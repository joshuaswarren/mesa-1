# AGX3 integer uniform witnesses

This corpus covers 22 expressions across `uint4`, `int4`, `ushort4`, and
`short4`: add/subtract (both operand orders), multiply, multiply-add, bitwise
logic, shifts, extraction, min/max, select, and rotate. Each kernel reads varying
input and uniform parameters and writes observable output.

On 2026-09-21, Apple Metal 32023.921 (metalfe-32023.921.6), using
`-fno-fast-math`, compiled all 88 kernels for both `applegpu_g17p` and
`applegpu_g16g`. Each pair has identical extracted machine code. `manifest.json`
records the 176 text hashes and sizes. The compiling host was `ssh m5`; the
explicit targets select the encoding, independently of that host's GPU.

## Reproduce

On a Mac with that toolchain, copy this directory and run:

```sh
sh compile-apple.sh /tmp/agx3-integer-results
```

For local analysis, use the neighboring
`../agx3-optimization-corpus/extract.mjs` on the downloaded results. Its optional
decoder argument writes candidate instruction boundaries. This toolchain's
`metal-objdump` cannot initialize the Apple GPU disassembler. The in-tree
decoder also reports unknown bits; none of those diagnostics is treated as a
correctness oracle.

The following commands run from the Mesa repository root. Substitute an
existing native Mesa build for `$native_build`; its AGX3B disassembler archive
must already be built.

```sh
probe=src/asahi/compiler/test/probes/agx3-integer-uniform-corpus
compose=src/asahi/compiler/agx3_compose
native_build=../.mesa-native-build
cc -O2 -I"$compose" "$probe/decompose.c" \
  "$compose/agx3_compose.c" "$compose/agx3b_tables.c" \
  "$compose/agx3c_tables.c" -o /tmp/agx3-decompose
cc -O2 -DHAVE_ENDIAN_H -Isrc -Iinclude -I"$native_build/src" \
  "$probe/disassemble.c" "$native_build/src/asahi/isa/libagx3b_disasm.a" \
  -o /tmp/agx3-disassemble
node "$probe/../agx3-optimization-corpus/extract.mjs" \
  /tmp/agx3-integer-results /tmp/agx3-disassemble
node "$probe/audit.mjs" /tmp/agx3-integer-results /tmp/agx3-decompose
```

The audited capture produced 984 exact AGX3B structural round trips across
249 distinct selected instruction candidates, with zero unmatched candidates.
Every candidate is checked against the bytes at its reported file offset and
then reconstructed by the production composition tables. The generated
`integer-witnesses.json` preserves each byte sequence and matching field vector.
This verifies structural encodings; it does not execute shaders, establish
unobserved operand combinations, or prove the decoder's complete correctness.

## Findings and limits

- `iadd`, `isub`, and `imadd` read uniform operands directly in both widths.
  Subtraction witnesses cover each operand order. Half-width lanes include odd
  GPR and uniform indices. Multiply-add witnesses include uniform scale and
  addend operands together.
- Integer min/max uses tied `isel` operands, including compact half-width
  endpoints. Signed and unsigned conditions differ and must remain distinct.
- Bitwise and shift examples include direct uniform operands. Some expressions
  also have uniform preprocessing in a separate constant program, so their
  main instruction cannot be enabled without preserving that preprocessing.
- Dynamic shifts and rotate include masking/count preparation. A shorter
  sequence must preserve shift-count semantics. No one-instruction rotate
  implementation is established by this capture.
- This corpus does not cover high uniform-index boundaries, all dependency
  masks, saturation, mixed-width integer arithmetic, every source hint, or
  every immediate combination. Those remain integration-test requirements.

The validated encoder and lowering have since been integrated into the compiler
as `agx3_uniform_integer.[ch]` and `agx3_uniform_integer_instr.h`. These probe
files supply the independent native capture and paired measurement harnesses.

## Paired Mesa baseline

`mesa-corpus.c` builds the same 88 expressions in NIR, using Mesa's own ABI.
It covers uniform bases 12, 248, 256, and 504 for 352 variants. Load/store
alignment follows the four-component element size, including eight-byte half
vectors. The selected invocation index only supplies distinct varying inputs;
it is not intended to reproduce Apple's prologue.

From the Mesa root, with a native build configured for compiler tests:

```sh
node "$probe/build-mesa.mjs" ../.mesa-agx3-opt-build /tmp/agx3-mesa-integers
mkdir -p /tmp/agx3-mesa-integers/binaries
AGX_CORPUS_REPORT=1 AGX_CORPUS_OUTPUT=/tmp/agx3-mesa-integers/binaries \
  /tmp/agx3-mesa-integers/mesa-corpus > /tmp/agx3-mesa-integers/metrics.csv
```

All 352 variants compiled in the initial capture: 46,536 main-code bytes and
5,904 instructions in total, at most 24 GPR halves, zero scratch. This is a dirty
workspace at Mesa HEAD `3590ec7d0851418ad3ec9186a8d818ce9b716450`, not a pristine
commit baseline. Relevant source hashes at capture:

| File | SHA256 |
| --- | --- |
| `agx_optimizer.c` | `7f743f7974c4b988e6e6cb555b51f7210e24a714c95d01a8ef8fe0806124edd0` |
| `agx3_pack_binary.c` | `96be6d952bc305d4e6fa21a688b6cfa98e49cce3622ab07a99aa7d6ee1aa8d04` |
| `agx_lower_uniform_sources.c` | `574cc270ff9fe04972a3e7ea80e4af5696ca24f7170d404b4c0c94f848498f2d` |
| `mesa-corpus.c` | `cce22003538bb9fe42bade5e7fe2dfc8967852c5d51145ca67bf2f3138f6c075` |

The local full capture is `.tmp/agx3-integer-uniform-corpus/baseline.csv` under
the AuroraSilicon workspace, with emitted binaries in its `baseline/` directory.

## Separate source-complement fusion experiment

Inspection found that `agx_optimizer_bitop` blocks source-NOT fusion for all
AGX3B widths, although the packer already handles complemented word-wide truth
tables. A scratch-only replacement optimizer object permits that existing
fusion when the destination, both inputs, and NOT input are all 32-bit.

All 352 variants still compile. Sixteen improve, with no size, instruction,
GPR, or scratch regression: totals fall to 45,832 bytes and 5,840 instructions.
For example, four-lane `u32_andnot` falls from 140 bytes/17 instructions to
100/13, and `u32_notand` from 148/17 to 100/13. Signed cases have the same
reductions. GPR/scratch counts remain unchanged.

`../../test-agx3-integer-fusion.cpp` supplies focused tests for AND/OR/XOR with either
or both sources complemented, including register/uniform operands, plus
half-width preservation. The word-fusion test fails on the original optimizer
and passes on the scratch optimizer; the half test passes on both. Production
integration enables only the same-width word expressions with a uniform
operand. Register-only fusion increased triangle tessellation by 56 instructions
in the larger built-in corpus, despite unchanged spill/fill counts. That broader
experiment is not enabled. Normal Meson tests cover uniform fusion and preserve
the existing behavior for half-width and register-only expressions.
Pass histograms localize the increase to register shuffling: fusion removes
eight NOTs, allocation adds 25 SWAPs and removes 11 MOVs, and pseudo lowering
turns the extra swaps into 75 XORs. The resulting `75 - 11 - 8 = 56` extra
bitops exactly match the final instruction increase, without more spills/fills.
Relinking the existing compiler and graphics tests
with the scratch object leaves their baseline failure sets unchanged.

This experiment does not enable uniform operands in the packer, and does not
measure GPU performance. Native half NAND/NOR witnesses establish another
candidate. The encoder now supports these forms; relaxing half-width optimizer
guards remains pending.

## Integer uniform encoder

`agx3_uniform_integer.[ch]` implements the encoder for same-width
16/32-bit ADD, SUB, MAD, AND, OR, XOR, NAND, NOR, and AND-with-complement. It is
linked into the production compiler. Binary arithmetic requires one GPR
and one uniform; MAD additionally supports the witnessed register, uniform, or
zero addend. Unsupported operand orders, shifts, widths, and input domains
return failure without changing output. Commutative canonicalization preserves
GPR hints and never commutes a shifted add or subtraction.

The register bound is 192 halves, matching `AGX3B_NUM_REGS` and the production
allocator. Some bitwise fields can represent indices 192–255, but these are not
allocatable GPRs. Initial probing exposed this distinction; the probe now
rejects those indices even when a composition table would encode them.

Validation on 2026-09-21:

- 27 exact Apple byte specimens, including odd half registers, cached GPR
  addends, both subtraction orders, and shifted add.
- 10,368 boundary uniform-index/wait-mask combinations, with pairwise checks
  that all 64 dependency masks retain distinct encodings for each shape.
- Rejected inputs preserve output, and commuted operands preserve their hints.
- 338,015 total checks pass under Clang ASan and UBSan. This count includes
  byte comparisons and pairwise wait-mask comparisons, not that many shaders.
- `replay-native.mjs` reproduces 448 native instruction instances across 104
  distinct descriptors. It explicitly excludes the other 536 captured
  instructions: unsupported operations/domains and noncanonical operand orders.

From the Mesa root, build the standalone checks with:

```sh
probe=src/asahi/compiler/test/probes/agx3-integer-uniform-corpus
compose=src/asahi/compiler/agx3_compose
clang -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
  -DHAVE_ENDIAN_H -Isrc -Iinclude -Isrc/asahi/compiler \
  src/asahi/compiler/agx3_uniform_integer.c \
  src/asahi/compiler/test/test-agx3-uniform-integer.c \
  src/asahi/compiler/agx3_binder.c "$compose/agx3_compose.c" \
  "$compose/agx3b_tables.c" "$compose/agx3c_tables.c" \
  -o /tmp/agx3-test-integer-encode
/tmp/agx3-test-integer-encode
cc -O2 -DHAVE_ENDIAN_H -Isrc -Iinclude -Isrc/asahi/compiler \
  src/asahi/compiler/agx3_uniform_integer.c "$probe/encode-cli.c" \
  src/asahi/compiler/agx3_binder.c "$compose/agx3_compose.c" \
  "$compose/agx3b_tables.c" "$compose/agx3c_tables.c" \
  -o /tmp/agx3-encode-cli
node "$probe/replay-native.mjs" /tmp/agx3-integer-results /tmp/agx3-encode-cli
```

These checks establish encoding agreement and input handling. Neither the
generated boundary variants nor the candidate encoder have hardware execution
validation. Shader lowering now preserves these supported operand forms;
unsupported forms retain their existing materialization paths.

## Integrated measurements and outstanding validation

With the integer encoder, lowering, and word source-complement fusion together,
all 352 Mesa variants compile: 46,536 → 41,592 bytes and 5,904 → 4,780
instructions. 252 variants shrink; 16 use fewer GPR halves; none increases
bytes, instructions, GPRs, or scratch. These are code-generation metrics,
not GPU execution measurements.

The expanded Asahi host suite passes 138/146 targets. The three compiler GTest
failures and graphics failure set remain unchanged. Baseline reconstruction
matches all three source hashes above and reproduces the f16 I/O and command
lifetime failures. The precompiled-fill fixture initially had 64 baseline
failures and 83 after the broad experiment. Updating the size/GPR expectations
affected by the patch and the fill binary fingerprint leaves 63 failures, all
present in the baseline. Unrelated old expectations were not refreshed. The
suite is not green, and these remaining failures are not resolved by this work.

`libagx-metrics.c` compares generated built-in kernels and their launch fields.
Both AGX2 tables (214 kernels total) remain identical. Of 55 G17P kernels, 40
shrink; none grows or increases GPRs/TPS. Total G17P bytes fall 195,278 →
194,426; measured launch fields do not change. The triangle and quad
tessellation variants retain their baseline code. Instrumented compilation of
199 built-ins finds no instruction, spill, fill, size, register, or scratch
increase. Understanding the register-only fusion regression remains audit work;
hardware execution of the integrated integer patch is pending.

Final local artifacts under `.tmp/agx3-integer-uniform-corpus/` include
`restricted.csv`, `restricted-original-corpus.csv`, `restricted-libagx.csv`,
`metrics-restricted.err`, and `baseline-*` comparisons. Host-suite logs are
`.mesa-agx3-opt-build/meson-logs/integer-restricted.*`; the final fixture-only
rerun is `integer-fixture-final.*`.
