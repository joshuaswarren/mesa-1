# P09 native internal compute program ABI

This extends the existing D3D12 frontend over the Asahi compiler, Poly and
libagx. It does not admit geometry, tessellation, mesh or amplification draws
on a physical G17P fd. The existing preraster and raster guards remain closed.
`d12_list_build_compute_usc` also checks the AGX2-only gate at its entry.

`compile_poly_program` uses G17P preprocessing and a monolithic, no-preamble
key on G17P. Every internal compute upload reuses the CODE container used by
API compute. `code_va` names the CODE entry, not the BO base. The existing PSO
BO slots and `agx_shader_info` retain code, rodata, workgroup, staging, GPR,
shared-memory and scratch metadata. `native_stages[]` records certification and
the required uniform source cells. AGX2 continues using its existing uploads.

The compiler records optimized NIR `load_preamble` declarations in half-register
units. This mask is a publication contract; it is not a CProg TPS import layout
or a substitute for texture/sampler state metadata. Compiler rodata is retained
and published separately. Unknown driver halves, overlapping/odd rodata,
missing/misaligned cells and a mismatch with the retained ABI refuse before
changing the output ranges.

Each four-half driver slot reads eight bytes from a source cell. The cell
address is passed to `agx3_ioto_uniform_range`; the value inside the cell is
what the shader receives. Required null pointer **values** still need a valid
source cell. Supplying the parameter buffer address in place of its pointer
cell would load the first eight parameter bytes as an address.

| Half range | Existing D3D12 declaration | Cell contents |
| --- | --- | --- |
| 4–7 | ROOT | Root binding/header address |
| 16–19 | TEXTURES | Texture table address |
| 20–23 | DRAW | Start vertex and start instance, two u32 values |
| 24–27 | GEOM_PARAMS | `poly_geometry_params` address |
| 28–31 | TESS_PARAMS | `poly_tess_params` address |
| 32–35 | VS_OUTPUTS | u64 output-location mask, not a pointer |
| 36–39 | VERTEX_PARAMS | `poly_vertex_params` address |
| 40–43 | MESH_PARAMS | Existing eight-pointer D12_MESH_OFF_* block address |
| 44–47 | NODE_PARAMS | Existing D3D12 node parameter block address |
| 48–51 | STAT_QUERY | Statistics query address (may contain zero) |
| 52–55 | VB_PARAMS | Existing 256-byte vertex-fetch parameter table address |
| 56–59 | HEAP | Descriptor metadata heap address |
| 60–63 | PBES | PBE table address |
| 64–67 | HEAP_TEXTURES | Native texture descriptor array address |
| 68–71 | HEAP_PBES | Native PBE descriptor array address |
| 72–75 | HEAP_SAMPLERS | Sampler-index array address |
| 76–79 | CUSTOM_BORDERS | Custom border table address |
| ≥80 | Compiler rodata | Compiler constants, exact base and half count |

Slots 0, 8 and the fragment-only root at 12 have no internal-compute producer.
A program publishes only slots used by its compiler declarations. A scalar
read within DRAW still uses the existing full eight-byte DRAW cell. The same
applies to the VS_OUTPUTS mask. The exact initial GS maps are emitted by the
certification kit; the table above describes the common ABI, not a claim that
every stage reads every slot.

The common generator uses `agx3_g17p_compute_program_generate` and the existing
Ioto loader. It refuses legacy scratch, CProg, uniform DMA, imageblock,
texture/sampler and traversal requirements. The native compiler's SPECTPR
footprint and supported shared-memory metadata go to the shared generator.
P13/P15 must certify their exact stage shapes before execution; this change
does not imply that every tessellation, mesh or helper program compiles.

## Certification and queued verification

Build in a new directory with the pinned P00 configuration and frozen reference
files. `build-native-stage-kit.py` takes the committed source archive, its
trusted SHA256, candidate commit, new build directory and installed P00 prefix.
Run it on the Neo only through an exclusive hardware request, using a transient
unit with `AllowedCPUs=1-4`, `CPUQuota=200%`, `Nice=10` and Ninja `-j3`.
It stages one library, all17 original suites, the actual registered DRM shim,
and the certification binary under a new kit; it never installs globally.
The diagnostic binary links the same runtime archive as the shared library so
it can inspect private retained compiler/BO metadata. Both binaries' compile
and link commands and hashes are in the kit receipts.

Place `run-native-stage-certification.sh` in `PREFIX/diagnostics/P09/` and run
it through the queue. With no arguments it derives PREFIX there; explicit
arguments are `PREFIX CANDIDATE_SHA OUTPUT_DIR`. It requires native Linux,
aarch64 and 16K pages, checks the staged artifact/frozen-reference manifests,
and uses the actual registered noop fd with G17P chip/stream validation.
It never creates a command queue. It does not run the original G6 pixel suite.

The helper preserves the ordinary API PSO creation result. If API signature
validation refuses before compilation, an actual-registered-fd-only diagnostic
entry runs the same software VS/GS lowering and native upload functions. This
does not alter API validation, fix the G6 PSO refusal tracked by t-1092, or
admit a command list.

The helper compiles the original `gs_input_vs.dxil`, `solid.dxil` and
`gs_point.dxil`, retains software VS/GS CODE and the linked raster companion,
and the pre-GS patch-up program, and checks Ioto generation and refusal of missing cells, misalignment,
undeclared halves, ABI mismatch, unknown staging and preamble requirements.
The Ioto certificate uses explicitly synthetic source-cell addresses and is
never submitted. P10 must supply real command-owned source cells and prove
buffer ownership, synchronization, sequencing and grid publication.

Artifacts include final NIR, binary, program name, exact compiler refusal,
compiler metadata JSON, retained CODE, synthetic Ioto bytes, ABI range JSON,
SHA256 hashes, complete stdout/stderr and the supplementary certificate verdict.
A refusal produces a failing certificate, not a certified program or a suite
PASS. Run the frozen G2 and G4 via the plan's `verify-reference.sh` with this
same candidate library to establish the physical regression result separately.

Later stage tasks can use `d12_native_compute_generate` with their retained
`native_stages[]` metadata. They must validate/publish each required source
cell and its pointee span, retain/synchronize CODE/rodata and recording-owned
Ioto, and enforce actual-fd admission plus queue revalidation. This P09
certificate alone is insufficient to open those guards.
