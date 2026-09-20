# sin() DenormFlushToZero keyed emission — device A/B receipt (jw16 M1 Max, 2026-09-20)

Second-host replication of `2026-09-20-sin-ftz-window.md` (M2 Max) on
jw16mbp1-linux (Apple M1 Max, T6001, G13C C0). Main-delegated execution;
existing `/tmp/m1-gpu.lock` (inode 12) held exclusive via flock; llm-inference
stopped before the lock, restarted after release, confirmed `active` with
health `{"status":"ok"}` post-load. No system driver install — isolated ICDs
(`VK_DRIVER_FILES`/`VK_ICD_FILENAMES`), shader cache disabled
(`MESA_SHADER_CACHE_DISABLE=true`), shader cache OFF both arms. System Mesa
and services otherwise untouched; jwm1 not touched.

## Builds (native jw16, generic armv8-a, isolated /tmp/mesa-sin-ftz-ab)

| artifact | sha256 |
|---|---|
| candidate driver (hk/fdiv-fckey 095cb7e1b0a) | `aade269367b42001adb20156990570bf6ee1afd1576b43f778ee5c6a507f50b3` |
| matched baseline driver (e1677564284) | `7087accecede1f556ed604d28fe88a8f6e5e90df202bc8cdf50c4cea51c76bfd` |
| mathrepro (native) | `399730d86af7d852c00324d6e9450c496ec23e36324b8d7befbc49590eb099db` (bit-identical to the M2 build) |

Pre-window gates: `PINS.sha256` 5/5 OK; `spirv-val --target-env vulkan1.2`
both modules OK; lock inode verified 12; agx_tests 61/61 native on jw16.

## Caps (candidate, in-window)

`Apple M1 Max (G13C C0)`, Honeykrisp `Mesa 26.3.0-devel`,
`shaderDenormFlushToZeroFloat32 = true`, `shaderDenormPreserveFloat32 = false`
— identical shape to the M2 Max dump; the keyed mode is legal on G13C
(`window-jw16-attempt2-green/caps-fckey-props.txt`).

## Result (OpGLSL450.sin, 64 lanes, SSBO-loaded bit patterns)

Semantic assert (in-window, `window-jw16-attempt2-green/semantic-assert.txt`):
**256 words x 4 legs, 0 failures.** Key rows (full dumps
`jw16-ab/probe/jw16-{fckey,e167}.json`):

| input | e167 default | e167 FTZ | fckey default | fckey FTZ |
|---|---|---|---|---|
| 0x00000001 | 0x00000001 | 0x00000001 | 0x00000001 | **0x00000000** |
| 0x807fffff | 0x807fffff | 0x807fffff | 0x807fffff | **0x80000000** |
| ±0, FLT_MIN, normals, inf, nan | identical across all four legs | | | |

Candidate default arm word-exact with baseline; the four FTZ subnormal edges
fix on the candidate only — same shape as M2 Max, now on G13C.

## Attempt history (preserved)

- attempt 1 (in `failures/window-jw16-attempt1/`): ICD jsons shipped into
  `probe/` but referenced at the staging root (loader: file not found), plus
  a `--env` flag the probe driver does not parse. Repaired CPU-side: ICDs
  copied to the staging root, cache-off moved to an exported variable.
- attempt 2 (green): `window-jw16-attempt2-green/`.

## Scope limits

Single probe shader; no CTS executed (deqp-vk not installed on jw16 —
caselist prepared at `jw16-ab/fp32-caselist-1.4.6.txt`, 1995 cases, for a
future Main-owned window); no performance claim; M1 (jwm1/t8103) not
exercised (box offline, Main+owner recovery in progress).
