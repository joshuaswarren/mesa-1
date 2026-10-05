# d3d12agx: full-pass review (t-1082)

Source reviewed: Mesa `ae23ffebb9d` (= a212d162f5e + t-1081 "restore G13/G14 graphics on
stock Asahi kernels") plus the microbenchmark commit on this branch. Scope:
`src/asahi/d3d12` (136 k lines: 15 k `d12_cmd.c`, 5.2 k `d12_device.c`, 5 k `d12_pipeline.c`,
3.3 k `d12_queue.c`, 3 k `d12_resource.c`, 2.5 k `d12_agx3_isa_core.c`, 13 k `dxil/`, 35 k
`umd/` Windows DDI, 40 k `tests/`). Detailed per-area notes with line references are in
`<aurora-ctl>/scratch/t-1082/review-{cmd,queue,device,pipeline}.md`.

Every number in section 2 was measured on 2026-10-05 with
`d3d12agx_microbench` (commit 6615044a005) on the M1 Pro (G13S) and M2 (G14G), stock Fedora
Asahi 44 kernels; raw logs in `<aurora-ctl>/reports/d3d12-stats/t-1082-raw/`.

## 1. Architecture map

```
 app ──COM vtables (plain C structs, d12_*.c)──────────────────────────────┐
   d12_device.c       ID3D12Device..Device2 + Device15/DeviceConfiguration1 tear-offs,
                      CheckFeatureSupport (d12_core_caps.c, one static table), Create*,
                      Create*View (writes struct d12_descriptor into a heap slot and
                      mirrors it into the heap's GPU texture/PBE/sampler tables)
   d12_resource.c     committed/placed/reserved resources = one agx_bo each; heaps;
                      Map/Unmap; WriteToSubresource/ReadFromSubresource (ail CPU tiling)
   d12_descriptor.c   descriptor heaps: CPU slot array + GPU-visible native tables
   d12_root_sig.c     RTS0 serialise/parse -> dense binding slots (d12_binding_map)
   d12_pipeline.c     PSO: DXIL -> NIR (dxil/) -> agx_compile_shader_nir -> prolog/epilog
                      -> agx_fast_link; separate AGX2 (USC words) and G17P (agx3 Ioto
                      program) containers in struct d12_gfx_stage
   d12_cmd.c          command list = array of up to 2048 d12_segment (CDM or RENDER),
                      control streams bump up / data bumps down in one 32 MiB stream BO,
                      USC words in a 32 MiB low-VA BO; per-draw binding upload, USC/PPP
                      emission, read/write trackers, indirect patch dispatches,
                      GS/tess/mesh software pre-raster compute stream (pregfx)
   d12_queue.c        ExecuteCommandLists builds a drm_asahi_submit payload (one command
                      per segment) under queue->lock, enqueues to a worker thread; a
                      second retire thread turns syncobj completion into fence values
   d12_fence.c        fences are userspace u64 + mutex/condvar; GPU waits are CPU hops
   d12_platform_{linux,windows}.c   shared handles, event signalling, sync provider
   d12_agx3_*_core.c  G17P command-stream producers (thin wrappers over
                      src/asahi/compiler/agx3_*_core and src/asahi/lib/agx3_*)
   dxil/              self-contained DXIL bitcode reader + DXIL->NIR; d12_dxbc.c = SM5
 ──substrate──────────────────────────────────────────────────────────────────────
   src/asahi/lib (agx_device, agx_bo, agx_submit, scratch, bg/eot), layout (ail),
   genxml (agx_pack), compiler (agx + agx3), libagx (precompiled helper kernels),
   poly (GS/tess lowering)
```

Two hardware generations share every file: "legacy"/AGX2 (G13, G14: USC words, agx_usc_*
helpers, `agx_cdm_launch`) and "native"/G17P (agx3 Ioto programs, agx3 VDM/PPP state
producers). The split is made at run time by ~30 scattered `dev.chip == AGX_CHIP_G17P`
tests plus four environment gates (`AURORA_D3D12_PHYSICAL_BASELINE`,
`AURORA_D3D12_PHYSICAL_D32`, `AURORA_D3D12_SHIM_DEPTH_METADATA`,
`AURORA_D3D12_SHIM_TEXTURE_METADATA`) evaluated inside the draw path (`d12_cmd.c:77-130`).

## 2. Measured baseline (M1 Pro, G13S; M2 in brackets)

| metric | value | comment |
|---|---|---|
| record, plain draw | 982 ns [946] | 20 000 `DrawInstanced` in one list, best of 5 |
| record, draw + `SetGraphicsRoot32BitConstant` | **15 811 ns** [15 299] | 16x plain: every root change misses the single-entry binding cache |
| record, draw + PSO toggle | **6 000 ns** [5 813] | USC cache keyed on the wrong things, rebuilt |
| record, draw + viewport change | 1 001 ns | ok |
| record, dispatch | 1 061 ns [1 016] | |
| record, dispatch + table switch | **2 644 ns** [2 538] | |
| CBV write (staging / shader-visible heap) | 51 / 74 ns | registry walk under global lock |
| CopyDescriptorsSimple, bulk / single | 30 / 31 ns per descriptor | |
| graphics PSO create, first / median | 1.6 / 0.96 ms | 7 full DXIL parses, nothing cached |
| compute PSO create, first / median | 2.0 / 1.8 ms | |
| empty list Reset+Reset+Close | 3.9 us | |
| Reset after a submission | **264 us** [313] | two fresh 32 MiB BOs + 365 KiB memset |
| `ExecuteCommandLists`, 20 k draws (1 segment) | 20 us | |
| `ExecuteCommandLists`, 20 k dispatches | **3 668 us** [3 371] | per-dispatch work at submit |
| GPU wall, 20 k draws / 20 k dispatches | 8.2 / 131 ms | 6.5 us per dispatch on the GPU: one barrier per launch |
| one-draw submit->fence round trip | 363 us median, p95 407 [261 / 361] | 1 ms polling waits + CPU fence hop |

Frozen 17-suite baseline on this base (M1 Pro and M2 identical): **11/17**
(g1 3 FAIL, g5_depth 3 FAIL = the documented test contradictions; g6-g9 NO VERDICT:
`CreateGraphicsPipelineState: incompatible shader signatures` for the frozen GS/tess/mesh
PSO descriptions). This is up from 6/17 at a212 and validates t-1081 on both chips. Neo:
8/17 on a212 (h-5099); the ae23 rebuild is queued.

## 3. Findings, ranked by impact

### A. Crash and correctness hazards

| # | finding | where | impact |
|---|---|---|---|
| A1 | **ID3D12Device15 vtable is 80/85 NULL.** QI answers `IID_ID3D12Device15` but the tear-off fills only QI/AddRef/Release/CreateQueryHeap1/ResolveQueryData; Device3..14 return E_NOINTERFACE. Any app (vkd3d-proton tests, every modern engine) that asks for Device3+ either fails or segfaults on first inherited call. | `d12_device.c:79-83, 4729-4735`; `gen_stubs.py` does not know Device15 | crash |
| A2 | `SetName(NULL)` dereferences NULL in `d12_private_data_set_name` | `d12_pipeline.c:378-390` | crash |
| A3 | PSO refcount race: `d12_pso_Release` and `d12_pipeline_state_library_unref` use two independent atomics; `device_ref_active` is a plain bool. A library and an app releasing concurrently can free twice. | `d12_pipeline.c:693-724` | UAF |
| A4 | Bundles drop root arguments / IA state set inside the bundle; only draws are replayed. | `d12_cmd.c:10052-10105, 11496-11555` | wrong rendering with bundles |
| A5 | ExecuteIndirect count buffer is read on the CPU at submit time instead of on the GPU. | `d12_cmd.c:5413-5430` | wrong draw counts when the GPU writes the count |
| A6 | Read/write trackers hold unretained `d12_descriptor_heap*` / `d12_resource*` pointers dereferenced at Execute; descriptor-kind refs are read without `descriptor_heap_lock`. | `d12_cmd.c:602-680`, `d12_queue.c:686, 1133` | UAF if the app releases a heap between Close and Execute (legal) |
| A7 | Binding cache keys on stale descriptor snapshot contents: two draws in one list using different contents of the same heap slot share one upload. | `d12_cmd.c:3720-3729` | wrong bindings |
| A8 | `retire_available` condvar is shared by three waiter classes but signalled with `cnd_signal`; the 1 ms timed-poll primitives hide the lost wakeup. | `d12_queue.c:112-166, 1344` | latency, potential stall when the polls go |
| A9 | Per-PSO leaks on the success path: every NIR shader and every GS/tess/mesh sub-part binary; ~13 error returns leak `vs_part`/`fs_part`. `struct d12_module` (3.44 MB) lives on the stack in three places. | `d12_pipeline.c:1500, 1777`, `dxil/d12_dxil_nir.c:6558` | memory growth, stack overflow risk |
| A10 | Pre-gfx segment insertion invalidates `seg` after scratch/sampler-heap were bound to it. | `d12_cmd.c:9290-9303` vs `9910-9930` | stale segment pointer |
| A11 | Fixed stream-space margin (0x300) regardless of bytes emitted; per-ioctl command limit hard-coded to 64 instead of `params.max_commands_per_submission`. | `d12_cmd.c:1061`, `d12_queue.c:1781, 2447` | overflow / EINVAL on large lists |
| A12 | Fixed DXIL parser limits (`MAX_INSTS 4096`, `MAX_VALUES 8192`, `MAX_METADATA 4096`) reject real uber-shaders. | `dxil/d12_dxil.h:474-478` | PSO creation fails for games |
| A13 | `D3D12GetInterface` and `EnableExperimentalFeatures(0)` fail; `GetDebugInterface` E_NOINTERFACE. | `d12_device.c:5193-5207` | app start-up failures |
| A14 | `texture_shapes[]` filled only for compute PSOs but read for all. | `d12_cmd.c:3696, 3805` | wrong texture dimension on graphics |

### B. CPU cost (host side)

| # | finding | where | measured / estimated |
|---|---|---|---|
| B1 | **Binding upload cache is one entry per pipeline type, compares ~33 KiB per draw and misses on any A/B/A/B root change.** On a miss it rebuilds the root planes (two 8 KiB stack arrays), re-uploads, and re-packs texture + PBE descriptors for every heap slot from the table base to the *end* of the bound heap (a 1 M-slot heap = millions of pack/unpack round trips and 24-32 MiB of the 32 MiB arena). | `d12_cmd.c:3671-4630` (3692-3729, 3772-3815, 4080-4400, 4578-4596) | root-constant draw 15.8 us vs 1.0 us |
| B2 | USC caches (`gfx_upload_cache[1024]`, `fragment_state_cache[2048]`) have no eviction and key on `draw_gpu`; `direct_draw_params` caches only the last pair, so varying base vertex fills the table and then every lookup probes all 1024 entries and rebuilds USC. | `d12_cmd.c:8606-8686, 9050-9067` | PSO toggle 6.0 us |
| B3 | Per-dispatch submit work: each `Dispatch` is its own CDM launch with a full barrier and the queue re-walks per-dispatch state at submit. | `d12_queue.c:2353-2583`, `d12_cmd.c:6372` | 3.7 ms CPU and 6.5 us GPU per 20 k dispatches |
| B4 | Submit builds a complete "read set" (O(max_handle) BO-map scans under `bo_map_lock`, 256-entry sorted inserts, hundreds of BO refs) that the Linux DRM backend never consumes (`agx_submit` ignores `virt`). The whole payload is built under `queue->lock`. | `d12_queue.c:630, 1043, 2542`, `agx_device.c:787-790` | all of the 20 us to 3.7 ms submit cost |
| B5 | `sparse_resource_lock` taken ~41 times per draw (32 VB slots, IB, 8 SO) and per root CBV/SRV/UAV set. | `d12_cmd.c:8875-8917, 10879` | ~100-200 ns per draw |
| B6 | Read/write trackers dedupe by linear scan (54 call sites); 30-60 appends per draw. | `d12_cmd.c:432-686` | quadratic in list size |
| B7 | `getenv()` in hot paths: ~30 knob names, 80+ call sites, 1 per draw on the hit path, ~12 per gfx miss, ~10 per dispatch, 14 in `d12_queue.c`, one in `Fence::GetCompletedValue`. | everywhere | 1-3 us per call |
| B8 | Reset-after-submit allocates two fresh 32 MiB BOs (no pooling) and memsets ~365 KiB of caches/tables; `struct d12_cmd_list` is ~3.2 MiB (`segments[2048]` ~2.8 MiB, 512 B of viewport snapshots per segment). | `d12_cmd.c:11825-11856, 807-890` | 264 us per Reset |
| B9 | Descriptor writes take a global device lock and walk every live descriptor heap to find the owner; `Create*View` calls it 2-3x; sampler heaps rebuild a 2048-entry dedup table per write (O(N^2)); shader-visible heap creation unpack/repacks the null descriptor per slot. | `d12_descriptor.c:91-218, 440-456` | 74 ns now; O(#heaps) with thousands of staging heaps |
| B10 | PSO creation: 7 full DXIL parses per VS+PS PSO (each signature query re-parses), identical `static_vi` prolog recompiled per PSO, no prolog/epilog cache, no fast-link table, no `disk_cache`, `GetCachedBlob` E_NOTIMPL, pipeline library recompiles on load and scans linearly without a mutex. hk caches all of these. | `d12_pipeline.c:4480-4503`, `dxil/d12_dxil_nir.c:6545-6615`, `d12_device.c:2158-2345` | 1-2 ms per PSO; thousands per game |
| B11 | Fences are userspace counters: every GPU->GPU dependency is a CPU round trip through the retire thread (one `drmSyncobjTimelineWait` per Signal), and every blocking wait is a 1 ms timed poll (`d12_sync_wait_for`, `d12_sync_lock` = trylock + yield + 1 ms sleep), on Linux too. | `d12_queue.c:112-166, 2155, 2184`, `d12_fence.c:124-190` | 363 us round trip, 1 kHz polls |
| B12 | Sampler lookups copy an 8 KiB `d12_binding_map` by value per static sampler per stage per miss. | `d12_cmd.c:5645-5690`, `dxil/d12_dxil.h:1055` | |
| B13 | Every resource/heap BO uses `AGX_BO_ZERO_ON_REUSE` even with `CREATE_NOT_ZEROED`; no suballocation for small buffers. | `d12_resource.c:2068, 2878` | memset per allocation |
| B14 | G17P: `d12_gfx_stage_g17p` regenerates the Ioto program per draw; `close_cdm` syncs the whole 32 MiB stream BO per pass; `get_pregfx` memsets 512 KiB per GS/tess/mesh pass. | `d12_cmd.c:6586-6680, 1270-1280, 1348` | |

### C. GPU side

| # | finding | where |
|---|---|---|
| C1 | Every submitted command barriers on all prior commands; no VDM/CDM overlap, no `_with_prior` or `PER_COMMAND_INPUT_SYNCS` as hk does. | `d12_queue.c:2447`, hk_queue.c:887-891 |
| C2 | Every dispatch is a launch + full CDM barrier (6.5 us GPU each). hk batches launches and barriers only on hazards. | `d12_cmd.c:6372-6586` |
| C3 | Outside a render pass, every attachment is loaded and stored (no clear/discard inference beyond the first clear); render passes split on any binding/tilebuffer-layout change and whenever 256 scissors or 64 depth biases are used. | `d12_cmd.c:2636-3156, 8947-8975` |
| C4 | Uploads: CopyTextureRegion/WriteToSubresource go through CPU tiling or compute copies; no blit engine use; no staging ring. | `d12_resource.c:613`, `d12_cmd.c:11977-12843` |

### D. Structure, duplication, dead code, knobs

| # | finding |
|---|---|
| D1 | Nine function pairs exist in AGX2 and G17P flavours (bg/eot, stage USC vs Ioto, descriptor refresh, compute launch, draw emit, precomp dispatch, sampler push, bind, PSO append) selected by 15 chip tests and 9 generation switches in the recording path, plus 10 env-gate calls. No per-chip ops table. |
| D2 | `d12_agx3_isa_core.c` (2514 lines) is a pure enum-remap wrapper over `compiler/agx3_*_core`; the `d12_agx3_vdm_*`/`ppp_*` producers are one-call wrappers 3-6x larger than hk's equivalents; six producer files have only test callers. |
| D3 | The native G17P draw path hard-codes ~20 negative gates ("bounded triangle": no VBs, no textures, triangles only, D32 only, 1 sample, no preamble, instance_count == 1, no indexed/indirect). Each feature lane must delete gates rather than add a path. |
| D4 | `d12_pipeline_state` stores 11 ad-hoc BO/VA/info triples for sw_vs/gs/tcs/tes/ms/as/setup programs; no `d12_program[]` abstraction. |
| D5 | 31 `D3D12AGX_*` debug knobs and 4 `AURORA_D3D12_*` qualification gates read per call; registered-shim "metadata-only" modes (`g17p_txf_metadata_only`, `g17p_txf_compiler_only`) live in production draw/dispatch/queue paths. |
| D6 | ~250 lines of D3DKMT shared-handle code compile on Linux where every op is E_NOTIMPL; DDI-only queue code (three copies of the submit sequence) sits in `d12_queue.c`; `d12_heap_*` names collide between ID3D12Heap and ID3D12DescriptorHeap. |
| D7 | `d12_core_caps.c` is one static table: 12_0 / heap tier 2 / mesh tier 1 claimed for all chips; wave width hard-coded 32; OPTIONS16/19 bypass the table. G17P overclaims mesh/tess/texture support it refuses at draw time. |
| D8 | Stale docs: `README.md:7-11` still claims 17/17; comments cite 1 MiB/2 MiB arenas (32 MiB real), "6 KiB root planes", unused `expected` parameter, double `release_segment_scratch` per Reset. |
| D9 | Platform seam: the declared 13-function `d12_platform_*` interface is shared-handle plumbing; the real platform surface used by the queue is `agx_device_ops` + `util_sync_provider` + event signalling + direct `agx_bo_*`/`bo_map_lock` access. A Wine unixlib backend (t-1084) slots in at the libasahi vtable level only if the frontend stops scanning the BO map and gates the read set on a winsys capability. |

### E. API coverage gaps

Device3..14 absent (A1). `ID3D12GraphicsCommandList` 1..9 exist. Missing or stubbed: enhanced
barriers (`Barrier()` on List7), `CreateCommittedResource3`/`CreatePlacedResource2`
(Device10), sampler feedback, DRED, video, work graphs execution, ray tracing execution
(`d12_raytracing_core.h` is CPU-only), `OpenExistingHeapFromAddress`, `GetCachedBlob` for
graphics, `SetResidencyPriority` beyond bookkeeping.

## 4. Ranked change list (what this task executes, in order)

Scoring: crash fixes and measurable wins first; mechanical moves last.

1. **A1 + A2 + A13**: full `ID3D12Device15` vtable via the generator, accept Device..Device15
   on one pointer, NULL-safe SetName, GetInterface via the config tear-off.
2. **B7**: hoist every `getenv` into `device->debug` flags parsed once at device creation
   (`DEBUG_GET_ONCE_*`); turn the four `AURORA_D3D12_*` gates into device booleans.
3. **B1 + A7**: dirty-bit the root tables; replace the single-entry binding cache with a hashed
   multi-entry cache keyed on a 64-bit hash of (rs, heap ids, table contents); never repack
   "to end of heap" (use the heap's own GPU-native tables + base offset).
4. **B2**: drop `draw_gpu` from the USC cache key (push draw params as their own uniform
   range), add LRU/hashed replacement, cache `vb_params`.
5. **B4 + B3**: gate the read-set build on a winsys capability (off on Linux DRM); move
   payload construction out of `queue->lock`; use `max_commands_per_submission`.
6. **B8**: pool stream/USC BOs per allocator; invalidate caches with a generation counter
   instead of memset; shrink `d12_segment` (viewport snapshots out).
7. **B10 + A9**: parse each DXIL stage once per PSO (heap-allocated module), free NIR and
   sub-part binaries, prolog/epilog + fast-link caches keyed like hk.
8. **B11 + A8**: real condvar wakeups (broadcast), kernel-side timeline deps for monotonic
   fences (hk_queue model), no 1 ms polls on Linux.
9. **B9**: descriptor heap registry as a sorted range table without the global walk; don't
   register non-shader-visible heaps; incremental sampler dedup.
10. **D1 + D3 + D5**: `struct d12_chip_ops` per device (bg/eot, stage program, descriptor
    refresh, launch, draw, bind) with `d12_cmd_agx2.c` / `d12_cmd_g17p.c`; qualification
    gates become ops booleans.
11. **A3, A4, A5, A6, A10, A11, A12, A14** as individual small fixes with tests.
12. **D2, D6, D8**: delete the isa_core wrapper layer, `#ifdef _WIN32` the KMT-only code, fix
    docs.

## 5. Refactor plan: target structure and sequence

Target module layout (same `src/asahi/d3d12/` tree, no new library):

```
d12_debug.c/.h        one-time env parsing -> struct d12_debug_flags on the device
d12_chip_ops.h        per-generation ops table; d12_chip_agx2.c, d12_chip_g17p.c implement it
d12_device.c          COM device + QI (Device..Device15 on one vtable), caps per chip
d12_resource.c        resources/heaps (BO pool, no forced zeroing)
d12_descriptor.c      heaps (range registry, incremental native mirrors)
d12_pipeline.c        PSO front end; d12_shader_cache.c (prolog/epilog/fast-link/disk)
d12_cmd_state.c       API state, root tables + dirty bits, caches, trackers (hash sets)
d12_cmd_draw.c        draw/dispatch front end (chip-independent), calls chip ops
d12_cmd_copy.c        copies/clears/resolves
d12_cmd_prerast.c     GS/tess/mesh/amplification pregfx stream (lanes g6-g9)
d12_queue.c           submission (payload outside the lock, capability-gated read set)
d12_queue_ddi.c       DDI-only submit sequences (Windows)
d12_fence.c           timeline-backed fences
d12_platform_*.c      shared handles + event signalling only
```

Sequence, each step buildable, tested on m1pro/m2 (17 suites + microbench) and Neo:

| step | kind | content | gate |
|---|---|---|---|
| S1 | fix | A1/A2/A13 Device15 vtable, SetName, GetInterface + `tests/d12_device_qi_tests.c` | 17 suites unchanged |
| S2 | mechanical | `d12_debug.c`: all getenv -> flags; `AURORA_*` gates -> device bools | microbench no regression |
| S3 | semantic | binding cache rewrite (B1/A7), root dirty bits | rootconst draw near plain draw |
| S4 | semantic | USC cache key/replacement (B2), vb_params cache | psotoggle draw < 2 us |
| S5 | semantic | submit: no read set on DRM, payload outside lock, max_commands (B4/B3) | dispatch submit << 1 ms |
| S6 | semantic | allocator BO pool + generation-counter invalidation (B8) | Reset < 50 us |
| S7 | semantic | PSO parse-once + frees + prolog/epilog/fast-link cache (B10/A9) | PSO median < 0.5 ms, no leak under valgrind |
| S8 | semantic | queue wakeups + timeline deps (B11/A8) | round trip < 150 us |
| S9 | mechanical | chip ops table + file split (D1/D3/D5) | byte-identical streams on both chips |
| S10 | fixes | A3-A6, A10-A12, A14; D6/D8 cleanups | tests |

### Feature-lane accommodation

The lanes in PLAN.md (P03 Texture.Load, P04-P08 VB/indexed/instancing, P09-P17 GS/tess/
mesh/AS on G17P) need, from this structure:

* `d12_chip_ops.draw()` receiving a chip-independent `struct d12_draw_desc` (topology,
  counts, index/VB bindings, indirect args, generated-stage programs) so G17P adds support
  by removing gates in `d12_chip_g17p.c`, not by branching in the front end.
* `struct d12_program` (binary, BO, VA, info, ioto/usc words) replacing the 11 ad-hoc
  triples in `d12_pipeline_state`, with `hw_vertex_stage` pointing at whichever program
  rasterises (VS, GS raster, TES, MS raster).
* A per-stage descriptor-usage summary at PSO creation (textures, samplers, VBs, UAVs)
  replacing `g17p_root_only`, so the native binding producer knows exactly which prefix
  halves to publish.
* `d12_cmd_prerast.c` owning the pregfx stream with chip-ops launches (`launch_compute`,
  `barrier`) so GS/tess/mesh run natively once `d12_chip_g17p.launch_compute` exists.
* The VB parameter ABI (`vb_params` uniform) cached in command-list state and published by
  the chip op, so P04/P05 only add the G17P publisher.

## 6. Out of scope here, recommended follow-ups

* Enhanced barriers / resource state tracking (none exists; barriers are no-ops beyond
  segment splits) once vkd3d-proton tests exercise them.
* Native G17P tessellation (hk `hk_g17p_tess_is_native`) after the software route is green.
* Per-chip capability audit (D7) with the orchestrator's contract for G1.

## 7. Results of the executed series (2026-10-05)

Series on branch `ctl/t-1082-...`, on top of t-1081's ae23ffebb9d:

| step | commit | change |
|---|---|---|
| S0 | 6615044a005 | `d3d12agx_microbench` |
| S1 | e7bcec55e79 | ID3D12Device3..15 on one complete vtable, CreateCommandList1, SetName(NULL), EnableExperimentalFeatures(0); `tests/device_interfaces.c` |
| S3 | 0a7ed778d1f | root-generation keyed binding cache + plane-only re-root |
| S4+S5 | 4414b585468 | 4-way binding cache, bounded USC probes, read set gated off on DRM (`device->submit_read_set`) |
| S2 | 1d5c3a64d7a | `d12_debug()`: all knobs parsed once |
| S6 | 554d3e7a777 | allocator-owned command storage pool, lists reference their allocator |
| S7 | b67b8dc4dd1 | DXIL signature cache, module off the stack |

Microbenchmark, median of 3 runs, debugoptimized build (M2 in brackets):

| metric | before (ae23) | after (b67b) | gain |
|---|---|---|---|
| record plain draw | 984 ns [942] | 94 ns [86] | 10.5x |
| record draw + root constant | 15 779 ns [15 299] | 661 ns [628] | 24x |
| record draw + viewport | 1 001 ns [962] | 105 ns [100] | 9.5x |
| record draw + PSO/root-sig toggle | 6 000 ns [5 819] | 142 ns [116] | 42x |
| record dispatch | 1 055 ns [1 019] | 65 ns [62] | 16x |
| record dispatch + table switch | 2 635 ns [2 538] | 127 ns [109] | 21x |
| graphics PSO create (median) | 1.06 ms [1.10] | 0.53 ms [0.48] | 2.0x |
| compute PSO create (median) | 2.08 ms [2.04] | 2.24 ms [1.98] | compile-bound, unchanged |
| allocator + list Reset after submit | 335 us [307] | 11.7 us [6.8] | 29x |
| ExecuteCommandLists, 20k dispatches | 3 623 us [3 395] | 5.3 us [3.7] | 690x |
| ExecuteCommandLists, 20k draws | 15.3 us [8.7] | 7.6 us [6.6] | 2x |
| one-draw submit round trip (median) | 362 us [264] | 278 us [168] | 1.3x |
| CBV descriptor write (visible heap) | 77 ns | 84 ns [73] | unchanged (B9 not done) |

Frozen 17 suites: 11/17 on M1 Pro and M2 at every step (g1 x3 and g5_depth x3 are the
documented contradictions; g6-g9 are refused at PSO creation with "incompatible shader
signatures" for the 2026-10-05 fixture descriptions, unchanged from the base).
`device_interfaces`: ALL PASS at every step. Raw logs and JSON records:
`<aurora-ctl>/reports/d3d12-stats/t-1082-*`.

Not executed from the plan (next in order): S8 queue wakeups/timeline deps (B11), S9
descriptor registry (B9), the chip-ops split (S9/D1), the per-PSO NIR/binary frees and
prolog/epilog/fast-link caches (A9/B10 rest), the "to end of heap" texture repack (B1
core, not exercised by the microbenchmark), and the correctness items A3-A6, A10-A14.

## 8. Second pass (2026-10-06): rebase, bindless ranges, agents' lanes

The series was rebased onto the t-1099 integration head `87c78f894a1` (a212 + four hk
commits + t-1081 + t-1087's G17P Texture.Load admission). The Neo was never compiled on
again: Neo packages come from a Fedora 45 podman image on the M1 Pro (`aurora-f45-mesa`,
`scratch/t-1082/f45-build.sh`) and run under the Phase 2 kit environment
(`AURORA_D3D12_PHYSICAL_BASELINE=1 AURORA_D3D12_PHYSICAL_D32=1`, shim variables unset).
The previous attempt's "6/17 on the Neo" was that environment missing, not a regression:
the same build is 8/17 (= a212) under the kit environment (h-6158), and the rebased series
is 9/17 (= t-1099) before and after every commit below.

### Commits (on top of `1d2ee857bd4`, the rebased S0-S10)

| sha | lane | change |
|---|---|---|
| 10a737f9062 | bench | `dispatch_texture` (texture table at the start of a 65536-slot heap) and `dispatch_miss` (eight tables, every dispatch misses the 4-way cache) modes |
| d85918323fe | B1 core | dynamically indexed texture ranges address the heap's native mirrors through a heap index in the range's first resource slot; no per-command copy to the end of the heap, no per-slot submit fixups |
| bd19552e5be | B1 | every binding-upload pass and cache snapshot bounded by the root signature's resource_count instead of all 256 slots |
| 7bb38b08b45 | B14/G17P | retired native VDM slabs pooled on the device, alias kept bound (hk's pooled command BOs) |
| f896dc14cf4 | D8 | README records the measured suite status |
| 0fa2c1a68a2 .. 4cd90d34538 | B9 (agent) | sorted range table + seqlock for heap lookup; one refresh per view; incremental sampler dedup; per-range copies; null natives packed once per heap; one-walk CBV sparse resolve |
| e2c37723815 .. d242ca01ec5 | B10/A9 (agent) | every NIR and sub-part binary freed; device prolog/epilog cache (hk_get_prolog_epilog model); compiled-stage + fast-link cache keyed on bytecode hash + binding map + stage keys; pipeline library locked, indexed, graphics GetCachedBlob |
| 0902159130b .. a63e8981e40 | A3/A4/A5/A10/A11/A12 (agent) | DXIL parser tables grow (uber-shaders parse); one atomic PSO lifetime (library race); pre-graphics slot reserved at render begin (no stale segment pointer); ExecuteIndirect count proven GPU-side (test only, it already was); stream-space checks sized from the emission; bundles replay every recorded command |

New live/unit tests: `dxil_limits` (meson), `pso_library_race`, `pso_threads`,
`execute_indirect_count`, `bundle_state`; the microbench gained sampler-heap, texture-table
and binding-miss sections.

### Measured (debugoptimized; medians of 3 unless stated)

| metric | M1 Pro before (1d2ee) | M1 Pro after (a63e8) | M2 after | Neo before (87c78) | Neo after (7bb38) |
|---|---|---|---|---|---|
| record plain draw | 89 ns | 93 ns | 84 | 960 ns | 223 ns |
| record draw + root constant | 665 ns | 248 ns | 229 | 12 308 ns | 883 ns |
| record draw + viewport | 105 ns | 108 ns | 100 | 1 032 ns | 252 ns |
| record draw + PSO toggle | 140 ns | 142 ns | 114 | 3 971 ns | 274 ns |
| record dispatch / table hit / table miss | 65 / 112 / 2 522 ns | 67 / 114 / 403 ns | 62 / 107 / 397 | 1 385 / 2 868 / - | 715 / 727 / 1 292 ns |
| dispatch_texture, 65536-slot heap (record / Execute call) | 26 909 ns / 3 380 us | 137 ns / 14 us | 111 ns / 8 us | - | - |
| CBV write staging / visible | 50 / 73 ns | 8 / 16 ns | 40 / 64 (pre-B9) | 11 / 139 ns | 11 / 126 ns |
| CopyDescriptorsSimple bulk / single | 29 / 31 ns | 7 / 15 ns | | 66 / 75 ns | 61 / 65 ns |
| CreateSampler into a 2048-slot visible heap | 176 us | 28 ns | | | |
| graphics PSO create, median (repeat) | 0.54 ms | 0.01 ms | | 1.54 ms | 0.53-0.71 ms |
| compute PSO create, median (repeat) | 1.9 ms | 0.00 ms | | 1.95 ms | 1.70 ms |
| ExecuteCommandLists 20k draws / 20k dispatches | 5 / 12 us | 9 / 13 us | 3 / 3 | 86 / 5 850 us | 49 / 5 us |
| Reset after submit | 9 us | 11 us | 7 | 740 us | 180 us |
| one-draw round trip median | 200 us | 228 us (loaded host) | 170 | 258 us | 251 us |

PSO leak check (valgrind, 64+64 creations): 1.27 MB definitely + 46 MB indirectly lost
before, nothing after. Frozen 17 suites: M1 Pro 11/17 and M2 11/17 after every commit;
Neo 9/17 at 87c78, 1d2ee and 7bb38 (a63e8 pending). The Neo's remaining Reset cost
(180 us) is not the slab churn; it needs a profile on the Neo.

### What remains (ranked)

1. B11 queue/fence latency (agent lane, in review at the time of writing: plain locks and
   condvar waits, inline submit when idle, direct timeline waits; a hang in its live test
   must be fixed first).
2. Static texture handles through the heap mirrors too, removing the per-command texture
   and PBE tables entirely (needs the G17P txf prefix publisher to follow).
3. Neo per-draw cost (223 ns vs 93 on G13) and Reset (180 us): profile on the Neo.
4. Disk cache for compiled stages and pipeline libraries; caching of the GS/tess/mesh
   sub-programs.
5. D1/D3 chip-ops split, D2 wrapper deletion, D6 `#ifdef _WIN32` of the KMT code, A6
   retained descriptor heaps across Close/Execute, D7 per-chip capability table.

## 9. Third pass (2026-10-06): queue lane, GPU-side traffic, G17P

Series on top of `a63e8981e40`, Neo packages built in the Fedora 45 container
on the M1 Pro and run under the Phase 2 kit environment.

### Commits

| sha | lane | change |
|---|---|---|
| 715b2a8beab .. d83fc735eaf | B11 (agent, reviewed) | plain locks and untimed condition waits; submit and signal in the calling thread when the queue is idle; direct timeline waits for pending GPU signals; drain runnable ops before stopping the worker; `--idle` microbench mode; live queue/fence contract test |
| 407e18662b1 | B10 | compiled stages persist in the Mesa shader disk cache (keyed on the library's build id and the AGX/D3D12 environment) |
| 9b1afcfb219 | B11 review | shared-fence waits use WAIT_FOR_SUBMIT, notify-handle ABA, wait-any polls shared fences |
| cc55230bb1e | C (GPU) | AGX2 compute cache flush deferred from every dispatch to the next barrier, helper launch or segment end |
| 91113c90ba9 | G17P | identical scissor and depth-bias records reused: no more render cut every 64 draws |
| 1261cc02d4b | G17P | native Texture.Load trace (three BLAKE3 hashes and two log lines per dispatch) behind `D3D12AGX_DUMP_TEX` |
| 684f32ab025 | G17P | API dispatches end in the weak barrier word; the strong flush is applied in place at the next barrier (Honeykrisp's model); `D3D12AGX_FLUSH_EACH_DISPATCH` covers G17P too |
| 1b20e71cef2 | G17P | the Ioto state-loader program is reused while the PSO and uniform ranges repeat |
| ec8fed30c9f | C (GPU) | full clears deferred to the render that writes the view; dead attachment stores retired by a later clear or DiscardResource; DiscardResource turns off the next load |
| 7f8f0eed552 | bench | `--only passes`: GPU wall of 16 frames of clear/bind/draw at 1080p in three shapes |
| 89c05dd4c2e | A (correctness) | a split declared render pass resumes with PRESERVE and the split render stores; a declared CLEAR processes every tile |
| 2d65607c6de | test | `render_pass_split` live test (fails against cc55230bb1e: the clear reaches a quarter of the target) |
| 39bffe4c69b | A (correctness) | a UAV or aliasing barrier, or a transition out of UNORDERED_ACCESS, inside a render whose draws write memory ends the render (hk's split for a memory-scope barrier in a pass) |
| 53e8807e597 | G17P | bind-state and draw-index records reused across identical draws (256 B of VDM slab per draw before) |
| fa3e5c0d645 | review fixes | a declared pass's entry traffic applied once per attachment (a split pass lost the CLEAR of targets the first render did not write); EndRenderPass applies clears no draw's render applied; depth/stencil stores retired only for renders without draws; a colour-less pipeline absorbs clears only with one bound format; bind descriptor padding; Clear* rejected in bundles |
| 107ee4f204c | test | `render_pass_split` declares a second cleared target no draw writes |
| 5758992ff6a, 030f0cafab3 | bench | the passes bench skips a shape the device cannot record (the Neo: linear 1080p RTV) and records each shape on its own list, since a failed Close poisons a list for every later Reset |

### How the G17P costs were found

The Neo's 20k-draw Execute call cost 2.2 ms where the G13's cost microseconds.
The native triangle producer appended a scissor and a depth-bias record per
draw with no reuse, so the 64-entry depth-bias array cut the pass every 64
draws: 313 hardware renders, five sliced submit ioctls, and at record time a
full 32 MiB stream sync plus four background/end-of-tile program builds per
render. The 10 us Texture.Load dispatch was the qualification trace left on.
The 44 ms of GPU time for 20k dispatches is the strong barrier word after
every launch; Honeykrisp already emits the weak form and upgrades it at a
pipeline barrier, which the driver now copies.

### Render-pass traffic

Before this pass a ClearRenderTargetView opened a render of its own and a
draw merged into it only with exactly that one target bound; every other
shape stored and reloaded. Clears are now pending on the list until the render
that writes the view absorbs them (per attachment, so MRT clears merge too),
or until a command that could observe the view forces a clear-only render.
Because clears are recorded rather than emitted, a later clear or discard of
an attachment retires the previous render's write-back of it. Colour only:
the ZLS unit has one store bit for the whole render, partial renders
included, so a render with draws keeps its depth and stencil stores
(Honeykrisp's `hk_pack_zls_control` forces them for the same reason; its
`HK_ZLS_PRECISE_DEPTH_STORE` mode, off by default, is the path to follow when
it becomes the default).

`d3d12agx_microbench --only passes`, 16 frames at 1920x1080 RGBA8 + D32, GPU
wall, best of three, before = cc55230bb1e library with the same binary:

| shape | M1 Pro before | M1 Pro after (ec8fed) | M1 Pro head (depth stores kept, final bench, 3 runs) | M2 before | M2 after (ec8fed) | M2 head |
|---|---|---|---|---|---|---|
| ClearRTV, bind, 4 draws | 2.69 ms (2.8-3.2 final bench) | 2.30 ms | 1.8-2.5 ms | 3.3-3.9 ms | 2.6-2.7 ms | 2.6-2.7 ms |
| ClearDSV, ClearRTV, bind, 4 draws | 6.82 ms (7.6-8.2) | 2.37 ms | 3.0-3.3 ms | 7.4 ms | 2.4-2.8 ms | 2.9-3.0 ms |
| the same + DiscardResource(depth) | 6.81 ms (7.6-7.8) | 2.30 ms | 3.0-3.2 ms | 6.5-7.2 ms | 2.3-2.8 ms | 2.8-3.0 ms |

The remaining 140 us per frame is the fixed cost of one 1080p render (tile
walk with the clear background program); solid-colour stores compress well on
AGX, which is why the colour-only shape gained less than the depth shapes. The
head keeps the depth store of every render with draws (about 45 us per 1080p
D32 frame on G13), the price of the review's partial-render argument above.
On the Neo the bench reports 0: a 1080p RTV created without
SIMULTANEOUS_ACCESS is a linear texture (`d12_resource.c`, `linear_ok`) and
the native G17P background/end-of-tile builder admits only GPU-tiled or
twiddled targets. That default deserves its own change: linear render targets
are the wrong default on a tiler for AGX2 too.

### Measured (debugoptimized; medians of 3 unless stated)

| metric | M1 Pro a63e8 | M1 Pro 2d656 | M2 2d656 | Neo cc552 (before G17P fixes) | Neo 1b20e (G17P fixes, h-7241) | Neo 57589 head (h-7340) |
|---|---|---|---|---|---|---|
| record plain draw | 93 ns | 96 ns | 83 ns | 218 ns | 191 ns | 86-90 ns |
| record draw + root constant | 248 ns | 239 ns | 221 ns | 839 ns | 816 ns | 800 ns |
| record dispatch | 67 ns | 65 ns | 62 ns | 710 ns | 39 ns | 43-44 ns |
| record dispatch_texture | 137 ns | 132 ns | 110 ns | 9 864 ns | 5 133 ns | 4.7-4.9 us |
| ExecuteCommandLists 20k draws, call | 9 us | 33 us (includes the inline submit ioctl since B11) | 23 us | 2 188 us | 24 us | (bench aborted, see passes) |
| 20k draws GPU wall | 7.9 ms | 7.4 ms | 6.3 ms | 31.4 ms | 21.0 ms | |
| ExecuteCommandLists 20k dispatches, call | 13 us | 54 us (inline ioctl) | 16 us | 33 us | 22 us | |
| 20k dispatches GPU wall | 105 ms | 2.0 ms | 2.0 ms | 44 ms | 39.8 ms (45.7 with the strong word, A/B) | |
| Reset after submit | 11 us | 4.7 us | 2.5 us | 8.2 us | 3.3 us | |
| one-draw round trip, median | 228 us | 210 us | 149 us | 256 us | 132 us | |
| graphics PSO, first (cold disk cache after a rebuild / warm) | 540 us | 370 us warm | 540 us | 2 290 / 670 us | 2 560 / 680 us | |

The Neo's 20k dispatches still cost 2 us of GPU each with the weak barrier
word: the launch itself, not the flush, is the cost on G17P (A/B: strong
45.7 ms, weak 39.8 ms). The remaining 5 us per native Texture.Load dispatch at
record time is the per-dispatch prefix work (sampler cell upload, range
building, syncs), the next thing to cache.

The 20k-draw GPU wall on the M1 Pro moves between 4.5 and 7.5 ms from run to
run with either library (GPU clock state), so only the record and call numbers
are comparable there.

Frozen 17 suites: M1 Pro 11/17 and M2 11/17 after every commit of this pass;
Neo 9/17 at 9b1afcfb219 and cc55230bb1e (h-6529), at 1b20e71cef2 and
53e8807e597 (h-7241), at 5758992ff6a (h-7340). `device_interfaces`, `render_pass_split`,
`pso_threads`, `pso_library_race`, `bundle_state`, `execute_indirect_count`
and the queue/fence contract test pass on the M1 Pro at the head; meson host
tests OK. Raw logs and JSON records: `reports/d3d12-stats/t-1082-*`.
