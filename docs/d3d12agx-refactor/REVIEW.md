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
