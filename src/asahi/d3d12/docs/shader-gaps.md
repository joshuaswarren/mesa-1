# Shader coverage measured on the refactor base

2026-10-05 UTC, task t-1108. Base: `c4518ba3ce5`. Reference corpus:
vkd3d-proton `31d1f89ca5b3`, 63 registrations from `d3d12_shaders.c`.
This supersedes the August roadmap's shader counts for this base and these
machines; it does not make a claim about every shader or the Neo.

| Machine | A before | A after | B | A assertion failures before/after |
| --- | --- | --- | --- | --- |
| M1 Pro | 57 pass / 6 fail | 58 pass / 5 fail | 58 pass / 5 fail | 191 / 169 |
| M2 | 57 pass / 6 fail | 58 pass / 5 fail | 58 pass / 5 fail | 191 / 169 |

No crashes or timeouts occurred in these shader runs. Accounting follows
t-1083's strict runner: unresolved TODO and BUG results fail a registration;
feature skips remain skips. A registration with zero ordinary assertion
failures can therefore still fail this scoreboard. The fp64 subcases were
skipped by the feature checks, not passed.

Changes:

* DXIL division/remainder by zero: eight ordinary instruction-test
  mismatches removed; the registration still contains half-conversion TODOs.
* 3D SRV descriptor depth: fourteen RESINFO mismatches removed. The heap
  descriptor must carry base volume depth, not the SRV's layer count of one.
  Existing libagx texture queries perform mip minification.
* SM5 texture feedback: thirteen texture instruction forms plus
  `check_access_fully_mapped` now use the existing NIR/AGX residency path.
  Null feedback destinations avoid the extra residency result. UNORM/SNORM
  UAV declarations return floats. The separate
  `test_texture_feedback_instructions_sm51` registration progresses from a
  first-opcode refusal/crash to a completed wrong-result failure. Its mip/LOD
  failures remain, as do failures in the corresponding DXIL registration.

The remaining five shader registrations are:

* `test_shader_instructions`: legacy half-conversion TODOs and fp64 skips.
* `test_shader_instructions_dxil`: legacy half-conversion TODOs.
* `test_uav_load`: `CopyTextureRegion` tiled-geometry refusal; later readback
  assertions follow the errored command list. This is not a shader refusal.
* `test_register_space_sm51`: four zero readbacks, with successful PSO
  compilation. The cause has not been isolated.
* `test_root_constant_indexing_dxil`: a TODO expecting PSO creation to fail;
  both A and B accept it. Preserve the upstream expectation in accounting.

## DXVK comparison and provenance

[Opcode-by-opcode table](../../../../docs/d3d12-shaders/dxbc-opcodes.csv)
compares the translator's enumerants/cases with
[DXVK v2.7.1](https://github.com/doitsujin/dxvk/tree/v2.7.1/src/dxbc), pinned
to `c3dd74be6baec53786d4e064a572185b70347a17`. The
[DXVK licence](https://github.com/doitsujin/dxvk/blob/v2.7.1/LICENSE) is zlib.
197 of its 231 non-reserved named opcodes now have d12 enumerants, up from
183. **This is an audit of dispatch coverage, not a conformance percentage.**
Dimensions, operand forms, binding maps and hardware generation impose
additional restrictions even when an enumerant is present. Some DXVK
enumerants have no implementation class either.

Remaining DXVK-supported families include call/label flow, LOD queries,
snapped interpolation, raw/structured buffer feedback and fp64 instructions.
Buffer forms of `ld_s` and `ld_uav_typed_s` still fail by name: the DXBC path
must consume the real buffer residency metadata before it can report them
mapped. Texture residency is taken from hardware, never replaced by a
constant. Sparse status codes are opaque and interpreted through NIR's
target residency intrinsic.

Instruction formats were checked against DXVK; this NIR implementation and
the token-stream tests are independently written. Integer division behavior
was checked against the MIT-licensed
[dxil-spirv implementation](https://github.com/HansKristian-Work/dxil-spirv/blob/master/opcodes/opcodes_llvm_builtins.cpp)
and [DXC issue 3150](https://github.com/microsoft/DirectXShaderCompiler/issues/3150).
No vkd3d-shader source was copied into the driver or its unit tests. Upstream
bytecode/assertions remain external behavioral reference material. No private
macOS reverse-engineering material was used.

## Verification and limits

The new CPU tests exercise 75 integer cases (16/32/64-bit, signed/unsigned,
division/remainder), production descriptor packing for three volume/array
views, and 39 independently assembled sparse texture token streams. They
open no GPU device. `meson test --suite d3d12` passes all five registered
tests.

Every measured code revision ran `tests/run-suites.sh` and
`tests/corpus-passset.sh`. M1 Pro remained at the established 11/17 frozen
suite ceiling with the same failed suites. The compile passset remained at
270 with empty `comm -23` regressions.

The original 334-shader corpus was not available in this worktree or the
reference build tree. The check used 1,053 external DXBC/DXIL arrays extracted
from the pinned Proton tests. The standalone tool's DXBC branch stops at
translation and supplies an empty binding map; it does **not** compile DXBC
to AGX. The standing passset script only counts `NIR->AGX OK`, so its 270
entries describe the DXIL backend subset. It does not certify DXBC execution
or the entire 1,053-array set. GPU registrations and PSO logs provide the
runtime evidence above.

A separate before/after CPU audit confirms 157 DXBC translations and 270
DXIL-to-AGX compiles on both revisions, with identical name sets. Its DXBC
failures include the tool's empty binding-map limitation; those refusals
must not be scored as hardware conformance failures.

fp64 ALU, 64-bit atomics, WaveMMA and barycentrics remain unsupported. See
[the hardware limits](tier5-hardware-limits.md). Neither passing translation
nor a feature skip establishes any of these features.

Raw logs, commands, identities and per-step stats are in the task's private
human-readable report and the orchestrator's d3d12-stats folder. The
[runner instructions](../../../../docs/d3d12-shaders/README.md) explain the
explicit Neo environment controls.
