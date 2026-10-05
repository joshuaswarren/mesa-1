# d3d12agx — Direct3D 12 on Apple AGX

A native D3D12 user-mode driver on the Mesa/Asahi substrate: DXIL is
translated straight to NIR and compiled to AGX machine code. **No Vulkan
layer, no SPIR-V, no `dxc` at run time.**

Status (2026-10-06, frozen ba4c reference suites, stock Fedora Asahi
kernels): **11 of 17 on G13 (M1 Pro) and G14 (M2), 9 of 17 on G17P (J700)
under the Phase 2 kit environment.** The six that do not pass on the Macs
are fixture contradictions, not driver failures: `g1_smoke` (three
capability expectations the driver cannot honour), `g5_depth` (three cull
expectations that vkd3d-proton contradicts too) and `g6`-`g9`, whose
2026-10-05 pipeline descriptions are refused at creation with
"incompatible shader signatures". The paragraphs below describe what each
suite proves when it passes; see `docs/d3d12agx-refactor/REVIEW.md` for
the measured state and the optimisation series.

- `g11_indexed_instanced` — **indexed draws and per-instance
  attributes**. An indexed draw with a non-zero `StartIndexLocation` and
  `BaseVertexLocation`, over an index buffer whose padding and decoy
  vertices make every one of those three links load-bearing, reproduces
  the `SV_VertexID` control pixel for pixel. Two instances with a
  per-instance offset attribute land on opposite sides of the target.
- `g10_vertex_buffers` — **an application's own vertex data reaches a
  shader**. One triangle drawn twice, once from `SV_VertexID` and once
  from a bound vertex buffer, matching each other and a CPU rasterisation
  on all 65536 pixels. The buffer is interleaved, so the second
  attribute's byte offset is load-bearing.
- `g6_geometry` — **geometry shaders execute**, as two compute dispatches
  ordered ahead of the render pass plus a rasterization program bound as
  the hardware vertex shader. A pass-through geometry shader reproduces
  the no-GS control **pixel for pixel** — all 16384 compared, zero differ.
- `g9_amplification` — **amplification shaders execute**: the AS writes
  its `DispatchMesh` grid on the GPU and the mesh shader is dispatched
  INDIRECTLY off it, through a six-word indirect-local grid that
  `as_setup` builds.
- `g8_mesh` — **mesh shaders execute**, via
  `ID3D12Device2::CreatePipelineState` and
  `ID3D12GraphicsCommandList6::DispatchMesh`. The first stage with no
  Honeykrisp reference: hk implements no mesh shaders, so the parameter
  block, draw-setup shader and rasterisation shader are all this
  driver's own. Its probe is generated here too — `nir_to_dxil` gained
  mesh and amplification support so that both suites run without the
  LGPL corpus.
- `g7_tessellation` — **tessellation executes**: five compute dispatches
  (software VS, hull, tessellator count, prefix sum, tessellator emit)
  and an indexed indirect draw whose index count the GPU writes. The
  tessellated patch matches a derived quad exactly — 4096 pixels, none
  wrong inside or out.
- `g5_texture` — reproduces `Texture2D.Load` for all 256 texels of a
  16×16 image.
- `g4_triangle` — matches a CPU rasterisation of the shader's own
  vertices on all 65536 pixels, D3D top-left fill rule included.

The remaining 13 corpus shaders are documented in
[`docs/tier5-hardware-limits.md`](docs/tier5-hardware-limits.md).

**First real conformance number, 2026-08-04**: **173 of 557**
vkd3d-proton D3D12 test functions complete with no failures against this
driver, 10 of them with nothing skipped. See `docs/roadmap.md` §5 for the
breakdown, the 207 crashes it exposed, and how to reproduce it.

The corpus figure is a **compile** proxy. Geometry, tessellation, mesh,
amplification, vertex attribute fetch, indexed draws and instancing now
execute; work graphs and ray query still compile without executing. See
`docs/roadmap.md` — the next step is running vkd3d-proton's tests for
real, which is the only thing that turns the proxy into a conformance
number.

## Start here

| doc | what |
|---|---|
| [`docs/replication.md`](docs/replication.md) | build it and reproduce the numbers — **read first** |
| [`docs/architecture.md`](docs/architecture.md) | the stack, the shader path, the binding model, licensing |
| [`docs/roadmap.md`](docs/roadmap.md) | what to do next, ordered by leverage, with falsification criteria |
| [`docs/findings.md`](docs/findings.md) | traps that each cost a wrong hypothesis — read before debugging |
| [`docs/tier5-hardware-limits.md`](docs/tier5-hardware-limits.md) | the last 13 shaders: what each needs, and the two that need a decision |
| [`docs/wddm-umd.md`](docs/wddm-umd.md) | native WDDM D3D12 UMD source inventory, pinned DDI ABI and staged implementation gates |

## Build

```bash
meson setup _build -Dvulkan-drivers=asahi -Dgallium-drivers= \
                   -Dplatforms=wayland -Dtools=asahi -Dasahi-d3d12=true
ninja -C _build
```

Needs Asahi Linux on AGX hardware, a kernel whose drm/asahi UAPI matches
Mesa 26.1.4, and **DirectX-Headers 1.619.1 installed system-wide** — the
tests `#include <directx/d3d12.h>` and `<wsl/winadapter.h>` directly.
Details and the verified procedure are in `docs/replication.md`.

## Test

```bash
src/asahi/d3d12/tests/run-suites.sh            # GPU acceptance suites
src/asahi/d3d12/tests/corpus-passset.sh > now  # corpus compile proxy
```

`tests/shaders/` holds the probe shaders. The older ones are HLSL sources
**and** their compiled DXIL, committed together because `dxc` does not run
on the target platform; the newer ones (`gs_*`, `tess_*`, `mesh_quad`,
`as_dispatch`, `vb_*`) are built as NIR by `tests/make_probe.c` and emitted
through Mesa's own `nir_to_dxil`, so they need nothing but this tree. `tests/corpus/` is empty in this repository: those shaders are
LGPL-derived from vkd3d-proton and are regenerated rather than vendored
(see `docs/replication.md`).

## Layout

```
d12_device.c       ID3D12Device, adapter/feature queries, view creation
d12_cmd.c          command lists, draws, dispatches, USC/PPP encoding
d12_queue.c        submission, fences (DRM syncobj timelines)
d12_resource.c     CreateCommittedResource, ail layout selection
d12_descriptor.c   descriptor heaps
d12_pipeline.c     PSO compile, prolog/epilog linking
d12_root_sig.c     root signature serialise/parse (RTS0)
d12_format.c       DXGI ↔ pipe format
dxil/              DXIL → NIR: container, bitstream, module, translator
tests/             GPU acceptance suites, standalone tools, shaders
```

## Branches

| branch | contents |
|---|---|
| `mesa/main` | mirror of upstream `main` |
| `main` | pinned base, `mesa-26.1.4`, no d3d12agx work |
| `dev` | this work |

Upstream is `gitlab.freedesktop.org/mesa/mesa` (the `upstream` remote). This
is not a GitHub fork because Mesa has no GitHub mirror to fork from; the
relationship is carried by that remote and by real shared ancestry.

## Licensing

Driver code is MIT, matching Mesa, SPDX-tagged. Anything adapted from
Honeykrisp or the AGX libraries keeps its license and attribution, noted in
the commit that introduces it. **vkd3d-proton and dxil-spirv are LGPL and are
used as reference and test oracle only — never copied into this tree.**
