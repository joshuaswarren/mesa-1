# sin() DenormFlushToZero keyed emission — device A/B receipt (jw14m2 M2 Max, 2026-09-20)

Fix under test: `hk/fdiv-fckey` @ `095cb7e1b0a` — the sin zero-compare select
returns a sign-preserving flushed zero when the shader declares
`DenormFlushToZero` fp32, and is byte-identical otherwise
(agx_nir_lower_math.c; default fdiv/frcp op histograms pinned by
test-lower-math.cpp, 61/61 host tests on x86 and native aarch64).

## Window (Main-owned; three M2 GPU windows, lock read-only fd9)

| window | outcome |
|---|---|
| v1 | SPIR-V parse rejection: raw uint StorageBuffer variables (vtn: struct+Block required). Modules kept as `*.invalid.spvasm/.spv`; log `failures/v1-e167-spirv-parse.log` |
| v2 | output-path bug in the probe driver (`inbin.replace("in-", …)` matched the directory name `jw14-sin-ftz-main-v2` → `/tmp/jw14-sout-…`); mathrepro returned 0 despite the ofstream failure — second defect. Log `failures/v2-e167-output-path.log`. Both fixed: probe uses `os.path.join(here, …)` (Main's line-52 fix, in `sin-denorm-probe.py`), mathrepro now checks the output stream (exit 3) |
| v3 | GREEN: both driver arms exit 0, digest pins all PASS, semantic PASS on the actual A/B |

## Build and pins (all aarch64, generic armv8-a flags, no -mcpu; isolated /tmp/mesa-fckey-095cb on jw14m2-linux)

| artifact | sha256 |
|---|---|
| candidate driver (hk/fdiv-fckey 095cb7e1b0a) | `55c3bd4aa61286907e75cc986574b47aeaa491898d0e469f38268ae5f54ebf50` |
| matched baseline driver (e1677564284) | `f52f1cf10bb8f496b192690dbc19dec72039d95911904823cab47010ceda70f3` |
| mathrepro (precise-math harness, native) | `399730d86af7d852c00324d6e9450c496ec23e36324b8d7befbc49590eb099db` |

Shaders: `sin-denorm-ftz-v2.spv` `0f8342512fec1bb7…`, `sin-denorm-default-v2.spv`
`82775187a68ea6ca…` (spirv-val --target-env vulkan1.2 clean on both; assembler
deterministic dev-box vs M2). Shader cache OFF both arms; driver+helper pins
independently verified by Main (`drivers.sha256`).

Caps (candidate, in-window capture; fields under
VkPhysicalDeviceVulkan12Properties): `shaderDenormFlushToZeroFloat32 = true`,
`shaderDenormPreserveFloat32 = false` (full dump `window-v3/candidate-vulkaninfo.txt`),
matching the stock 26.2.3 jw14 dump Main captured earlier.

## Result (OpGLSL450.sin, 64 lanes, SSBO-loaded bit patterns)

| input | e167 default | e167 FTZ | fckey default | fckey FTZ |
|---|---|---|---|---|
| 0x00000001 pos smallest subnormal | 0x00000001 | 0x00000001 | 0x00000001 | **0x00000000** |
| 0x007fffff pos largest subnormal | 0x007fffff | 0x007fffff | 0x007fffff | **0x00000000** |
| 0x80000001 neg smallest subnormal | 0x80000001 | 0x80000001 | 0x80000001 | **0x80000000** |
| 0x807fffff neg largest subnormal | 0x807fffff | 0x807fffff | 0x807fffff | **0x80000000** |
| ±0 | unchanged | unchanged | unchanged | unchanged |
| FLT_MIN 0x00800000 | 0x00000000 | 0x00000000 | 0x00000000 | 0x00000000 |
| normals (1.0, ln2, 1e-8), inf, nan | identical in all four legs | | | |

- The four FTZ subnormal edges: leaked subnormal on both arms of the
  baseline; sign-preserving flushed zero on the candidate FTZ arm only.
- Candidate default arm is word-exact with the baseline default arm (64/64)
  — the keying changes nothing without the mode declaration.
- sin(FLT_MIN) stays flushed zero on both drivers and both arms: the input is
  a normal whose reduced argument is subnormal, so this is pre-existing
  flushing behavior, correct under the declared FTZ mode and legal under no
  mode — noted, no full-math claim.
- Independent assertion: `verify-sin-ftz.py` (this directory) re-asserts the
  contract over all 64 words x 4 legs from the collected bins: **0 failures**.
  Main's independent verify also ran PASS in-window.

## Scope limits

Single probe shader (GLSL.std.450 Sin over 13 named patterns, one workgroup);
no CTS executed in this window; no performance or full-math claim; M1
(jwm1/t8103) compatibility is by generic-flags build only — not yet exercised
on M1 hardware. Validation chain: spirv-val vulkan1.2 both modules, host
gtest 61/61 (x86 + native aarch64), device A/B above.
