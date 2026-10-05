# d3d12agx roadmap — from compiling to a conformance number

Written 2026-08-04, at `dev`. **Rewritten the same day, twice**: the
geometry, tessellation, mesh and amplification runtimes all landed, and
mesh support in `nir_to_dxil` closed the last dependency on the LGPL
corpus. §1.0, §1.0b, §1.1, §2 and §3 are CLOSED. The highest-leverage
remaining work is §1.4 (the attribute-fetching vertex prolog) and §5
(executing vkd3d-proton's tests for real).

This document exists to be **attacked**. Every step states what it changes,
how you know it worked, and what would prove it wrong. Where a decision is
contestable it says so explicitly, in a **CONTESTED** note — those are the
places a second opinion is worth the most.

---

## 0. Where we actually are

**321 of 334 corpus shaders compile DXIL → NIR → AGX machine code**, and
**all seventeen GPU acceptance suites pass on real hardware with nothing
skipped** — from a clean checkout, with no corpus present.
No SPIR-V, no Vulkan layer, no dxc at run time.

The corpus is vkd3d-proton's test shaders used as a **compile-only proxy**:

- **Compiling** means the translator and the AGX backend accepted it.
- **Conforming** means it produces the right pixels/bytes on hardware.

**Every graphics stage that replaces fixed function now executes**:
vertex, geometry, tessellation, mesh, amplification, pixel. Each is
proven by a suite that derives its expectation from the shader rather
than from an observed number — `g6_geometry` compares a pass-through GS
against the no-GS control pixel for pixel, `g7_tessellation` against a
derived quad, `g8_mesh` and `g9_amplification` against a full-screen quad
whose vertices and indices are dumped and checked.

**The compile/execute gap has narrowed to two mechanisms.** Work graphs
and ray queries still compile without dispatching; everything else in the
graphics path runs, and since G10/G11 an application's own vertex data
reaches it — indexed, instanced, and from interleaved buffers. See "What 'done' means for tiers 2-4" below, and
§4.3.

### The probes are self-hosting

`g8_mesh` and `g9_amplification` used to be gated on `D3D12AGX_CORPUS`
because `nir_to_dxil` had no mesh support and no mesh probe could be
built. It has one now — shader kinds, `MSState`/`ASState` entry
properties, `setMeshOutputCounts` / `emitIndices` / `storeVertexOutput` /
`storePrimitiveOutput` / `dispatchMesh`, and PSV runtime info — so
`tests/make_probe.c` emits `mesh_quad.dxil` and `as_dispatch.dxil`
alongside the geometry and tessellation probes. **No suite depends on
LGPL material any more.**

### The remaining 13 — all Tier 5

Full analysis, including the two that cannot be done correctly without a
robustness tradeoff, is in
[`tier5-hardware-limits.md`](tier5-hardware-limits.md).


| Bucket | N | Why |
|---|---|---|
| fp64 arithmetic | 6 | AGX has no fp64 ALU; needs soft-float |
| WaveMMA | 3 | SM 6.9 cooperative matrix lowering |
| 64-bit atomics | 3 | hk reports all three Int64Atomics features false |
| `attributeAtVertex` | 1 | hk reports `fragmentShaderBarycentric` false |

Tiers 1–4 are **done**: named refusals (222), geometry and tessellation
on the `poly` emulation (269), mesh and amplification as compute (291),
work graphs and inline raytracing (321).

### What "done" means for tiers 2–4

Each shipped a documented ABI the runtime must honour. Four of the five
runtimes now exist and are proven on hardware:

- **Geometry** — DONE (`g6_geometry`). Two compute dispatches ordered
  ahead of the render pass, plus a rasterization program bound as the
  hardware vertex shader.
- **Tessellation** — DONE (`g7_tessellation`). Five compute dispatches
  and an indexed indirect draw whose index count the GPU writes.
- **Mesh/amplification** — DONE (`g8_mesh`, `g9_amplification`). Buffer
  allocation and the amplification → mesh hand-off through
  `D12_UNIFORM_MESH_PARAMS`, the mesh grid dispatched indirectly off
  words the amplification shader writes.
- **Work graphs** — still need a scheduler (`D12_UNIFORM_NODE_PARAMS`).
  ~21 corpus shaders compile against an ABI nothing drives.
- **Ray queries** — still compile to "traversal found nothing", which is
  correct only because the device reports `RaytracingTier 0`. Real
  traversal needs BVH build plus a traversal kernel.

---|---|---|
| Work Graphs (SM 6.8 GPU scheduling) | ~21 | §3.6 — needs a scheduler |
| Raytracing (library + `rayquery`) | 9 | §3.6 — needs BVH + traversal |
| fp64 arithmetic | 6 | §4.2 — soft-float |
| WaveMMA (SM 6.9 cooperative matrix) | 4 | §4.4 |
| 64-bit atomics | 3 | §4.1 — hk reports the feature absent |

Tiers 1–3 of the earlier plan are **done**: the three named refusals
(222), geometry and tessellation via the `poly` compute emulation (269),
and mesh/amplification translated directly to compute (291).

---|---|---|---|
| **1 — no emulation** | `waveMultiPrefixOp`, immediate constant buffers, SM 6.8 draw params (§2) | 3 | **222** |
| **2 — emulation ALREADY IN TREE** | geometry 15, hull 18, domain 14 (§3) | 47 | **269** |
| **3 — greenfield, moderate** | mesh 19 + amplification 3 (§3.5) | 22 | **291** |
| **4 — greenfield, large** | Work Graphs ~21, raytracing 9 (§3.6) | 30 | **321** |
| **5 — numeric / ISA** | fp64 6, WaveMMA 3, 64-bit atomics 3, barycentrics 1 (§4) | 13 | **334** |

Two corrections to earlier readings of this corpus:

- The 29 "library" shaders are **not** all raytracing. About 21 are **SM 6.8
  Work Graphs** (`broadcast_custom_input`, `thread_custom_input`,
  `coalesce_custom_input`, `two_level_*`, `embedded_subobject`,
  `local_root_signature`). Emulating those means writing a GPU work
  scheduler — a different and much larger problem than RT traversal.
- Honeykrisp exposes `geometryShader` and `tessellationShader` as **true** on
  this GPU but exposes **no** mesh, task, raytracing or ray-query features.
  So Tier 2 is wiring; Tiers 3 and 4 are design.

---|---|---|
| Stage-locked (hull/domain/geometry/mesh/amplification/library/RT/node) | 98 | needs the GS/tess campaign (§3) |
| fp64 arithmetic | 6 | needs soft-float (§4.2) |
| WaveMMA (SM 6.9 cooperative matrix) | 4 | out of scope until it matters |
| Hardware-absent: 64-bit atomics ×3, attributeAtVertex, rayquery | 5 | §4.1, §4.3 |
| Mechanisms not built, each refusing BY NAME | 3 | §2 |

**222** is the ceiling of the *current architecture* — not the conformance
limit, and not a hardware limit. Every tier above it is reachable; they
differ in how much of the machinery already exists.

---

## 1. Runtime: make the compiled work actually run

This is the critical path. Everything in §1 gates real conformance numbers.

### 1.0 — ~~Geometry shader dispatch~~ — CLOSED 2026-08-04 (`g6_geometry`)

102 shaders compile that cannot execute. Geometry is the cheapest entry
into that set, and it is **much cheaper than it looked** — established by
measurement, not estimate.

`d3d12agx_dxil_compile` now prints the `poly_gs_info` the lowering
computes. Across all 15 corpus geometry shaders:

```
xfb=0  prefix_sum=0   — every one of them
```

That matters because hk only dispatches the count, prefix-sum and pre-GS
programs when `main->info.gs.xfb` is set or a GS pipeline-statistics
query is active (`hk_cmd_draw.c:1535-1538`). **Three of the four programs
are never dispatched for this entire corpus.** What is left is:

1. a compute dispatch of the software VS (`poly_nir_lower_vs_before_gs`)
   writing `poly_vertex_params::output_buffer`;
2. a compute dispatch of the main GS body;
3. the hardware draw, with the **rast** program bound as the hardware
   vertex shader (`hk_bound_hw_vs` returns the RAST variant).

One shader is cheaper still. `gs_topology_point` is the only
`STATIC_PER_INSTANCE` shape in the corpus (`max_indices=1`), so it needs
**no output index buffer, no restart index, and a plain non-indexed
draw** — and its main program compiles to 18 bytes / 1 GPR, which is an
empty shader that hk would skip outright (`agx_is_shader_empty`). For
that one shader the whole runtime is *one compute dispatch and one
ordinary draw*. It is the correct first target.

The other 14 are `DYNAMIC_INDEXED`, which additionally need the output
index buffer, the rewritten restart index
(`BITFIELD_MASK(8 * poly_gs_index_size(shape))`) and an indexed draw.

**Already built:** `libpoly`/`libpoly_nir` are linked, the translator
lowers the poly parameter intrinsics to reserved uniforms
(`D12_UNIFORM_GEOM_PARAMS`, `VERTEX_PARAMS`, `VS_OUTPUTS`), and all four
programs compile. **Not built:** anything that publishes those uniforms.
No `.c` in the driver references them.

#### Step 0 — the actual blocker: the command list cannot order compute before a draw

This is not in the original plan and it gates everything else. Measured,
not assumed:

- `D12_SEG_RENDER` is created in **exactly one place**,
  `ClearRenderTargetView` (`d12_cmd.c:1863`). There is no way to open a
  render pass that **loads** the target instead of clearing it.
- The command list has **one** stream cursor (`stream_cur`,
  `d12_private.h:393`) and two open-flags (`cdm_open`, `render_open`).
- Therefore `d12_list_get_cdm()` handles the collision the only way it
  can: *"A render pass may still be open; compute work ends it"*
  (`d12_cmd.c:315-317`).

A GS draw needs compute dispatches to run **before** the draw that
consumes their output, while the render pass containing that draw is
already open. Today that closes the pass, and there is no way to reopen
it without clearing — so the draw would either land in a new pass that
wipes everything drawn so far, or not run at all.

hk does not have this problem because it keeps **two** control streams
per command buffer, `current_cs.gfx` and `current_cs.pre_gfx`, and
splices the pre-gfx CDM stream in *before* the graphics stream
(`hk_cmd_buffer.h:615-627`). `hk_draw` deliberately fetches the graphics
stream before the pre-gfx one (`hk_cmd_draw.c:3515`, `:3541`) so the list
insertion finds it.

Two ways out, and this is a **CONTESTED** choice:

- **(a) Mirror hk — a second stream cursor.** Add a pre-gfx CDM cursor to
  `d12_cmd_list`, and insert its segment before the open render segment
  (the segment array is ordered, so this is a `memmove` plus a second
  cursor). Structurally correct, no per-draw tilebuffer traffic, and it
  is the shape every later stage needs too — tessellation, mesh and work
  graphs all dispatch compute before rasterising. More invasive.
- **(b) Load-preserving pass resume.** Let compute split a render pass
  and reopen one whose background program *loads* the attachment. Smaller
  change, but it costs a full store + load of the render target per GS
  draw, and it needs the background/eot programs that today only exist
  for the clear path.

**(a) is the recommendation** — (b) buys a smaller diff now and pays for
it at every stage after geometry.

**Steps, in order:**

| # | change | how you know it worked |
|---|---|---|
| 0 | pre-gfx CDM stream (above) — **BUILT**, unproven until step 4 | a dispatch recorded during an open render pass appears BEFORE it in `agxdecode`, and the pass still draws |
| 1 | ~~device-side geometry heap~~ — **NOT NEEDED**, see below | — |
| 2 | PSO compiles the SW-VS variant and the rast program; rast becomes the hardware VS | PSO with a GS creates instead of returning `E_NOTIMPL` |
| 3 | upload `poly_vertex_params` (16 words) and `poly_geometry_params` (79 words), align 8 | `agxdecode` shows the uniforms published at the reserved registers |
| 4 | dispatch SW VS, then the main body, with a CDM barrier between | `agxdecode` shows two dispatches before the VDM stream |
| 5 | non-indexed draw for `STATIC_PER_INSTANCE` | **`g6_geometry`**: a point GS whose output pixels are compared against a CPU model, the way `g4_triangle` now is |
| 6 | index buffer + restart + indexed draw | the remaining 14 shapes |

**The geometry heap is not on this path.** hk allocates a 128 MiB
device-wide `poly_heap` and it looked like a prerequisite. It is not, for
a *direct* draw: `poly_nir_lower_gs.c` never emits a heap reference at
all (`grep heap` finds only a comment), and every actual
`poly_heap_alloc` call site is in code this path does not run —
`poly_gs_setup_indirect` (`geometry.h:640-650`, indirect draws),
`restart.h` (primitive-restart unrolling) and `tessellator.h`. For a
direct draw the CPU allocates `output_buffer` itself. The heap comes back
with indirect draws, restart, and tessellation; it is not a geometry
prerequisite.

**Traps already identified** (full list in the session research, and in
`hk_cmd_draw.c`): grid counts are in THREADS not workgroups; the
workgroup size 64 must match in three places with no cross-check;
`poly_pre_gs` unconditionally dereferences its counter pointers so they
need scratch/zero-page sentinels; params must be uploaded *before* the
root table; the compute programs read the **graphics** root descriptor;
and the 79-word/16-word struct sizes are static-asserted because the
shaders bake `offsetof` values.

**CONTESTED — the test shader.** `dxc` does not run on this box and the
corpus is LGPL, so `g6_geometry` cannot ship a committed GS probe the way
`g4_triangle` ships `tri.dxil`. Two options: (a) the suite takes a GS
path as an argument and reports "no shader supplied" otherwise, exactly
as `corpus-passset.sh` already does for an empty corpus directory; or
(b) generate probe DXIL with Mesa's own **NIR→DXIL** backend
(`src/microsoft/compiler`, MIT, already in tree — see
`architecture.md`), which would give clean-room committable probe
shaders and remove the `dxc` dependency permanently. (b) is more work and
strictly better; (a) unblocks step 5 immediately. **This needs a
decision.**

### 1.0b — ~~Tessellation: the patch constant function~~ — CLOSED 2026-08-04 (`g7_tessellation`)

Everything else in the tessellation path is built and measured working
(see the commit that added `d12_list_tess_prerast`): heap, three
compiled programs, five dispatches, indexed indirect draw, patch-list
topology, probe shaders. On hardware the pipeline creates, the GPU
completes the whole stream, and the software vertex shader writes its
control points correctly.

**Nothing tessellates because `counts[0] = 0`.**

A DXIL hull shader has **two function bodies**. The entry runs once per
output control point. A separate **patch constant function** runs once
per patch and is the *only* thing that writes `SV_TessFactor`. This
frontend translates only the entry, so the factors stay zero and the
tessellator correctly emits nothing. There is no error anywhere — the
draw is simply empty.

`d3d12agx_dxil_compile -n` now lists the module's functions and marks the
patch constant one; `HS_STATE` operand 0 identifies it.

Two attempts, both reverted, both worth not repeating blind:

1. **Skipping it when picking the entry** cost 18 corpus hull shaders
   (321 → 303). So the identification via `HS_STATE` operand 0 is not
   matching the right function for real DXC output — check how the
   metadata value wraps the function reference before trusting it.
2. **Concatenating the two bodies** onto one instruction list is sound in
   principle: the translator resolves values by absolute module index and
   never reads `func->first_value`, so the two bodies' value ranges do
   not collide. It hit a `CALL` whose callee resolved to value 10 — an
   instruction, not `dx.op.storePatchConstant`. That is a real operand
   decoding question, not a design flaw in the merge.

Whichever route wins, the patch constant body should end up in the same
NIR shader as the entry; poly's TCS lowering follows GLSL, where one
shader writes both the control points and the levels. No invocation
guard is needed — a GLSL tessellation control shader writes its levels
from every invocation too, and the values are per-patch.

**Falsification:** `g7_tessellation` grades against a derived quad —
constant tess factors of 2 on a quad domain, bilinear domain shader,
control points at (±0.5, ±0.5), so exactly 64×64 = 4096 pixels. A bowtie
covers half, a dropped ring leaves a hole, an empty tessellator covers
zero. It reports `counts[]` and the GPU-written draw descriptor under
`D3D12AGX_DUMP_TESS`.

### 1.1 — ~~Break the g5_texture wall~~ — CLOSED 2026-08-04

**Symptom.** 100 of 256 `Texture2D.Load` fetches return transparent black
(out of bounds) on a 16×16 texture with coordinates 0–15. Deterministic.

**Already verified correct by measurement — do NOT re-check these:**

- Descriptor unpacks to Linear / R8G8B8A8 / Unorm / 16×16, address exactly
  the BO VA, stride field 48 (the field's convention is bytes−16 ⇒ 64).
  Checked by unpacking the bytes against `cmdbuf.xml`.
- Texture memory at that VA holds the right texels (`px[0]`, `px[1]`,
  `px[16]` all match the CPU fill).
- Pointer chain at dispatch: `textures_gpu` → cell holding `heap_gpu` →
  descriptor. Printed and compared: MATCH.
- Lowered NIR is `txf`, `vec2(x,y)` in backend1, 2D, lod 0.
- Bindless offset is `slot * 24` bytes, matching `agx_bg_eot.c`.
- `uses_txf=1`, `sampler_state_count=1`, `texture_state_count=0`.
- hk imposes no restriction on sampling linear 2D images.

**Falsified hypotheses — do NOT re-run:** tiling, stride encoding,
descriptor address, bindless offset units, USC overflow, uniform register
count, missing rodata, missing sampler, sampler-count encoding,
`promote_constants`.

**Next moves, in order:**

1. Wire `agxdecode` into our submit path (`d12_queue.c`). It is not called
   today, so `ASAHI_MESA_DEBUG=trace` prints nothing useful. Mesa's own
   decoder parsing our command stream is the highest-information tool we
   have not used.
2. Dump the complete USC word sequence and diff it against one hk emits for
   a compute `texelFetch`. The remaining suspect is texture-unit *setup*,
   not the descriptor.
3. If both are inconclusive: build a minimal Vulkan compute `texelFetch`
   through Honeykrisp on the same box and diff the two command streams.
   Heavy, but it is the definitive oracle.

**Done when:** `g5_texture` passes, and the fix is explained in one sentence
naming the mechanism.

**CONTESTED:** it may be faster to skip straight to (3). The counter-argument
is that (1) is reusable infrastructure for every future runtime bug.

### 1.2 — Publish sampler state registers

**Contract already implied by the compiler** (landed in `0da20af`): a
constant sampler slot becomes `tex->sampler_index`, a computed one a 16-bit
`sampler_offset` source. Both index **sampler state registers**, which the
USC SAMPLER word loads — *not* a device sampler heap.

**Work:** build sampler state registers from bound sampler heaps and static
samplers, exactly as the device-wide `txf_sampler` at `ss0` is built today
(`d12_device.c`, mirroring `hk_device.c`).

**Done when:** a new `g5_sampler` GPU test samples a texture with a
non-default filter/address mode and matches a CPU reference.

**Depends on §1.1** — a sampling test cannot pass while fetches are broken.

### 1.3 — Publish the SM 6.6 descriptor heap

**Contract already implied** (landed in `dac8799`, `b6de66a`): heap handles
index `struct d12_descriptor` **in memory** at `D12_UNIFORM_HEAP`, with
`address` at offset 0 and `counter_address` at offset 8 — this is
shader-visible ABI, asserted in `d12_private.h`.

**Work:** publish the bound shader-visible heap's GPU address in the
reserved uniform, and make `CreateShaderResourceView` write hardware texture
descriptors into a parallel heap so heap-handle texture ops resolve.

**Unlocks:** every `ResourceDescriptorHeap[]` shader already compiling.

**Done when:** a `g5_heap` test binds a resource only via
`ResourceDescriptorHeap[i]` and reads correct data.

### 1.4 — ~~Attribute-fetching vertex fetch~~ — CLOSED 2026-08-04 (`g10_vertex_buffers`)

Compile side is done: `agx_nir_lower_vs_input_to_prolog` runs, so VS inputs
become prolog-exported registers. The runtime still builds only a
`static_vi` prolog that fixes up vertex/instance IDs.

**Done, and not as a prolog.** AGX's prolog vertex-fetch ABI starts at
uniform register 0 and collides with this driver's reserved block, so the
fetch is compiled into the MAIN shader with `agx_nir_lower_vbo` and the
base/clamp/stride come from a driver parameter block — the same choice
Honeykrisp makes. The prolog stays `static_vi`. See `findings.md`.

**Done when:** a draw with a real input layout renders correctly.

**The falsification was written first and then met.** `g10_vertex_buffers`
draws one triangle twice — once from `SV_VertexID` (`tri.dxil`), once
from a bound buffer (`vb_vs.dxil`) — and requires them to agree pixel for
pixel, both against a CPU rasterisation. 10965 pixels each, zero
differing. The buffer is interleaved (position at offset 0, colour at
offset 8, stride 24) so a fetch that ignored attribute offsets would pass
the position check and fail the colour one.

**Both follow-ups are now closed too**, by `g11_indexed_instanced`:
per-instance attributes (`PER_INSTANCE_DATA` with a step rate) and
indexed draws through a bound index buffer, including non-zero
`StartIndexLocation` and `BaseVertexLocation`. `IASetIndexBuffer` and
`DrawIndexedInstanced` were stubs until then.

---

## 2. Close the three named compile mechanisms (→ 222)

Each currently refuses by name. Small, independent, verifiable.

| # | Item | Approach | Done when |
|---|---|---|---|
| 2.1 | `waveMultiPrefixOp` | partitioned waterfall — same shape as the WaveMatch loop in `6da7bea`, scanning within each mask partition | shader compiles; a `g5_wave` case checks the scan against a CPU model |
| 2.2 | Immediate constant buffers | upload the initializer data, lower `load_constant` to global reads (AGX backend has none) — needs the data-carrying-initializer path that `pointer_for_value` deliberately refuses today | `ps_immediate_constant_buffer` compiles AND a GPU test reads the right table values |
| 2.3 | SM 6.8 draw parameters | publish a per-draw sysval carrying StartVertexLocation/StartInstanceLocation; `AGX_ABI_VUNI_FIRST_VERTEX`/`BASE_INSTANCE` slots exist but only the prolog-building side maps them | `vs_draw_args` compiles and a draw with nonzero start vertex is correct |

**Note on 2.3:** a constant 0 would compile it today and be *wrong* for any
draw with a nonzero start vertex — which the runtime supports. It refuses on
purpose.

---

## 3. The geometry/tessellation campaign (→ ~290)  ★ BIGGEST WIN

**98 shaders are stage-locked** — the single largest bucket, ~29% of the
corpus. Almost all are hull/domain/geometry/mesh.

**The key fact:** this is *not* invention. Honeykrisp already reports
`geometryShader = true` and `tessellationShader = true` on this GPU,
implemented by compute emulation whose kernels are already in our tree:
`src/asahi/libagx/geometry.cl`, `tessellation.cl`, `tessellator.cl`, plus
`hk_nir_passthrough_gs.c` and the NIR lowering that feeds them.

**The work is wiring DXIL's GS/HS/DS into machinery that already exists.**

Suggested decomposition (each independently testable):

1. **Recon** — read how hk drives `agx_nir_lower_gs` / the tessellator:
   what NIR shape goes in, what dispatches come out, what the index/param
   buffers look like. Produce a written map before writing code.
2. **GS first** — geometry is the simpler emulation, and
   `hk_nir_passthrough_gs.c` shows the minimal case. Translate
   `emitStream`/`cutStream`/`storeVertexOutput` onto it.
3. **Tessellation** — hull (patch constants, control points) and domain
   (`domainLocation`, `loadPatchConstant`), driving `tessellator.cl`.
4. **Mesh/amplification** — different model again (`setMeshOutputCounts`,
   `getMeshPayload`); decide from the recon whether it rides the same path.

**Done when:** each stage's corpus shaders compile *and* a GPU test draws
the expected geometry — not compile-only, given emulation's many moving
parts.

**CONTESTED:** whether this belongs before §1. My position is no — §1 gates
the *ability to verify* anything in §3, and shipping 98 more compiling-but-
unrunnable shaders would repeat the mistake this project already corrected
once (the fake 109). But it is the biggest single number on the board, so
the argument is worth having.

---

## 4. Hardware-limited and out-of-scope

### 4.1 — 64-bit atomics (3 shaders) — DEAD unless the substrate is wrong
hk reports `shaderBufferInt64Atomics`, `shaderSharedInt64Atomics`,
`shaderImageInt64Atomics` all **false**. `cs_64bit_atomics_shared` *compiles*
today — flagged runtime-suspect in WORKLOG; do not trust it until measured.
**Worth one cheap experiment:** run the compiled shared-atomic shader and see
what the hardware actually does. If it works, hk is being conservative and
this bucket reopens.

### 4.2 — fp64 arithmetic (6 shaders)
AGX has no fp64 ALU. `makeDouble`/`splitDouble` already work (pure bit
assembly). Real arithmetic needs soft-float — a large deliberate build with
prior art in Mesa. Not scheduled.

### 4.3 — attributeAtVertex (1) / rayquery + RT pipelines
`fragmentShaderBarycentric = false` in hk today — an hk gap, **not proven
hardware absence**; worth a substrate check before writing it off. RT needs
compute BVH traversal: an entire project.

### 4.4 — WaveMMA (4)
SM 6.9 cooperative matrix. Skip until something needs it.

---

## 5. The real conformance number — FIRST MEASUREMENT, 2026-08-04

**2026-10-05 update:** the DXBC front end now exists. On the refactor base
`c4518ba3ce5`, the same 63 shader registrations measure 57 pass / 6 fail on
M1 Pro and M2; t-1108 takes both to 58 pass / 5 fail, equal to the current
vkd3d-proton control. The August measurements below remain historical.
See [shader-gaps.md](shader-gaps.md) for the per-registration limitations,
DXVK opcode audit, and distinction between translation and execution.

**Command-side reassessment, 2026-10-05 UTC (t-1110):** the historical
counts below are not a baseline for the refactored driver. Native M1 Pro
(G13S) and M2 (G14G) measurements of the same 96 registrations give identical
results on both machines. Tuples are PASS / FAIL / CRASH / SKIP; a partial
subtest skip is SKIP and an outstanding todo is FAIL.

| file | d3d12agx `c4518ba3ce5` | d3d12agx `2f6523f03b4` | vkd3d-proton `31d1f89ca5b3` |
|---|---:|---:|---:|
| `d3d12_command.c` | 32/4/0/4 | 32/4/0/4 | 31/1/0/8 |
| `d3d12_pso.c` | 22/2/0/5 | 23/1/0/5 | 21/4/0/4 |
| `d3d12_query.c` | 11/0/0/0 | 11/0/0/0 | 9/2/0/0 |
| `d3d12_streamout.c` | 4/0/0/0 | 4/0/0/0 | 4/0/0/0 |
| `d3d12_geometry_shader.c` | 4/0/0/2 | 4/0/0/2 | 6/0/0/0 |
| `d3d12_mesh_shader.c` | 0/0/0/6 | 0/0/0/6 | 0/0/0/6 |
| total | 73/6/0/17 | 74/5/0/17 | 71/7/0/18 |

The separate eight-function root-signature slice improves from 7/1/0/0 to
8/0/0/0; the reference remains 7/1/0/0. Query and SO heaps already existed
in the base. The new execution-time timestamp, multi-fence rewind, and dynamic
SO-topology probes expose defects the existing corpus did not distinguish.
Those live probes pass on both Macs. The frozen suites remain 11/17 and the
270-shader compile passset is unchanged.

These numbers do not qualify the Neo, full pipeline-statistics accuracy,
or mesh execution. The two GS-file partial skips are the *without-GS*
layer/viewport-index subtests; their GS paths execute successfully. Mesh
capability remains unadvertised. Pipeline statistics still include inherited
recording-time estimates and require GPU instrumentation before claiming full
correctness. Evidence and exact commands are in the t-1110 worker report and
`<aurora-ctl>/reports/d3d12-t1110-20261005/`.

**185 of 557** vkd3d-proton D3D12 test functions complete with zero
failures against `libd3d12agx.so`. Of those, **10 execute with nothing
skipped at all**; the other 163 pass in part because a feature check
skips the body (no SM 6.7, no 64-bit shader ops, and so on). Both numbers
are the honest headline — the first is "did not fail", the second is
"actually ran and was right".

```
              first    now
PASS            173    185    (10 with zero skipped sub-tests)
FAIL            171    188
CRASH           207    179
TIMEOUT           5      4
SKIP              1      1
```

Reproduce with `d3d12agx/scripts/vkd3d-suite.sh` in the ntasp repo. It
builds vkd3d's OWN test harness against our driver and never builds
vkd3d's D3D12 implementation — reference and oracle only, as the standing
rule requires.

### Two things that made this possible

* **The harness needs almost nothing from vkd3d.** Only `vkd3d-common`
  (seven files) plus the D3D12 entry points, which we export. Every
  `vkd3d_*` helper the tests call is defined in
  `tests/d3d12_crosstest.h`. No Vulkan, no dxil-spirv, no SPIRV-Tools.

* **The aggregate-return ABI, which was the open question all along.**
  32 methods return a struct by value. DirectX-Headers on non-Windows
  declares them by value; widl declares them with an explicit hidden
  pointer. On AArch64 those are not compatible — by-value returns through
  x8, widl's form passes a pointer in x1 and never sets x8. **327 of 557
  tests segfaulted on this**, and the backtrace pointed straight at our
  `GetDesc`, which is exactly what a driver bug would look like. Fixed in
  the generated header, not in the driver.

### What the 207 crashes actually are

Now that the ABI is right, these are OUR bugs, and they cluster. Sampled
from the first 60:

| cause | note |
|---|---|
| `ExecuteCommandLists: list not closed` | our error path leaves the list unusable, then something dereferences it |
| `SetEventOnCompletion: non-NULL event` | real event objects are not implemented |
| `CommandAllocator::Reset with lists recording` | same shape as the first |
| `CreateCommandAllocator: type 1` | bundles |
| `CreateDescriptorHeap: type 1` | sampler heaps |
| `GetResourceAllocationInfo`, `CreateHeap`, `CopyDescriptors` | unimplemented, and the test then uses the result |

**The pattern is not "we do the wrong thing", it is "we refuse and then
crash".** G5a made the vtables crash-safe by filling them with
E_NOTIMPL stubs; this is the next layer down — the paths that DO exist,
fail correctly, and leave the caller holding something unusable. Fixing
that class is likely worth more than any single feature, because it
converts crashes into ordinary failures and lets the remaining tests in
each file run.

### Where the addressable work is

Every test mapped to its source file, after the crash-to-failure pass:

| file | pass | fail | crash | note |
|---|--:|--:|--:|---|
| `d3d12_shaders.c` | 2 | 26 | 35 | **the biggest single block** |
| `d3d12_descriptors.c` | 10 | 17 | 25 | |
| `d3d12_command.c` | 6 | 18 | 16 | |
| `d3d12_copy.c` | 4 | 12 | 19 | twiddled copies refused |
| `d3d12_resource.c` | 4 | 8 | 19 | |
| `d3d12_pso.c` | 4 | 18 | 7 | |
| `d3d12_query.c` | 0 | 0 | 11 | **no query heaps at all** |
| `d3d12_geometry_shader.c` | 0 | 0 | 6 | |
| `d3d12_clear.c` | 3 | 0 | 7 | |
| `d3d12_mesh_shader.c` | 0 | 6 | 0 | |
| `d3d12_streamout.c` | 0 | 4 | 0 | no stream output |

### Already at their ceiling

These pass by correctly reporting the feature absent. That IS the right
answer and needs no work:

| file | pass | why |
|---|--:|---|
| `d3d12_sm_advanced.c` | 40 | shader models above ours skip |
| `d3d12_raytracing.c` | 22 | `RaytracingTier 0` |
| `d3d12_sampler_feedback.c` | 13 | feature not reported |
| `d3d12_workgraphs.c` | 10 | feature not reported |
| `d3d12_enhanced_barriers.c` | 10 | |
| `d3d12_sparse.c` | 10 | |

### The theoretical limit

**It is close to 557, not far below it** — and that is the useful finding.
The features this driver will never have (raytracing, work graphs,
sampler feedback, fp64, 64-bit atomics, WaveMMA) are ones the suite
already SKIPS when the capability bits say no, and a clean skip counts as
"no failures". **The ceiling is set by how much ordinary D3D12 surface
gets implemented, not by the hardware.**

The exception is a test that uses a feature UNCONDITIONALLY after
checking nothing: `d3d12_query.c` releases a query heap whether or not
creation succeeded, so nulling the out-pointer cannot save it. Those need
the feature or nothing.

**A crash is never at the ceiling.** Each of the 179 is a defect whether
or not the underlying feature is in scope — the correct answer for
something unimplemented is a failure the caller survives.

### THE BLOCKER: there is no DXBC front end

Measured 2026-08-05, by running every failing test and grouping the
driver's own errors:

```
210  PSO creations fail with "no DXIL part in DXBC container"
202  DISTINCT TESTS blocked by that, and by nothing else first
```

vkd3d-proton embeds most of its shaders as fxc-compiled **Shader Model
5.x DXBC** — a `SHEX` chunk, no DXIL anywhere in the container. Decoded
from the test source directly:

```
chunks: 3
   ISGN 44
   OSGN 44
   SHEX 240  sm5.0 kind=1
```

SM 5.1 is legal D3D12, so this is a real conformance gap, not a test
artifact. DXBC is a **token stream**, nothing like DXIL's LLVM bitcode:
a different decoder and a different translator. The container parsing
already exists (it is how we find the DXIL part); what is missing is the
`SHDR`/`SHEX` token decoder and the instruction translation.

**This single gap sets the ceiling at 350.** Every other fix competes for
the remaining 171.

### Why incremental features barely move the number

Root 32-bit constants appeared in the error counts 35 times. Implementing
them moved the score by **+2**.

The causes are **LAYERED**: a test needs root constants AND a shader we
can compile AND a texture shape we support. Only the first failure
appears in the log, so an error histogram badly overstates what any one
fix is worth. Expect roughly 1-2 tests per bounded feature until the
layer beneath is gone.

This is the single most useful planning fact in this document: **the way
to a materially higher score is the DXBC front end, not a longer list of
small features.**

### Ordering from here

1. **Query heaps** — 11 tests, 0 passing, one self-contained feature.
   The cheapest whole-file win on the board.
2. **`d3d12_shaders.c`** — 61 failing or crashing, the largest block, and
   the compiler surface this project is actually about.
3. **Descriptors (42) and copies (31)** — large crash counts, so likely a
   few shared root causes rather than 73 separate bugs.
4. **Twiddled texture copies** — currently refused; needed across
   `d3d12_copy.c` and `d3d12_resource.c`.
5. **Stream output** — 4 tests, no implementation at all.

---

## 6. Standing rules for whoever picks this up

- **Measure, don't guess.** State what one number would falsify a hypothesis,
  then go get it. Every expensive mistake in this campaign was a confident
  hypothesis acted on without measurement.
- **A loud failure beats a plausible wrong answer.** Every refusal in the
  translator names its mechanism. Do not replace one with a plausible
  constant.
- **After every change:** `tests/run-suites.sh` (all suites) and
  `tests/corpus-tests/corpus-passset.sh` (corpus). Diff the passset with `comm -23 old new` — the
  count alone hides regressions that trade one shader for another.
- **Never build while an agent holds the tree.** This produced one confident,
  wrong conclusion already.
- **Keep the offline mirror current.** The fork is also mirrored as
  format-patches over the pinned `mesa-26.1.4` tag; refresh it at session end
  and verify with a clean `git am` + `diff -r`. That is what proves the
  rebuild recipe, and it is the copy that survives losing the remote.
