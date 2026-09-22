# G17P three-target blend and packed-float witnesses

Captured with Apple MetalToolchain v27.1.5194.15 (`AIR-NT 32023.917`) and
the real `applegpu_g17p` backend:

```sh
xcrun metal -c input.metal -o input.air
xcrun metallib input.air -o input.metallib
xcrun metal-nt -arch applegpu_g17p -platform_version macos 27.0 27.0 \
  -N rgba32-blend.mtlp-json input.metallib -o rgba32-blend.nat
xcrun metal-nt -arch applegpu_g17p -platform_version macos 27.0 27.0 \
  -N r11.mtlp-json input.metallib -o r11.nat
xcrun metal -c rgba32-subset.metal -o rgba32-subset.air
xcrun metallib rgba32-subset.air -o rgba32-subset.metallib
xcrun metal-nt -arch applegpu_g17p -platform_version macos 27.0 27.0 \
  -N rgba32-subset.mtlp-json rgba32-subset.metallib -o rgba32-subset.nat
```

The three blended RGBA32Float outputs produce 412 bytes of fragment text,
SHA-256 `df8a3ed7a5fd061304e7220648ae52e76fd3b93721be61778d770571553698d3`.
Its tile sequence loads offsets 32, 16 and 0, then stores them from register
halves 16, 8 and 0. All three payloads are live together, so Apple's program
uses the complete low bank through half 23. The outer native object SHA-256 is
`c6ef0a81d0feb5ed064244ee4b1eea11681cc58a980bab1f73efde650e31fdeb`.

The four-target subset pipeline masks RT0 and blends RT1..3. Apple's program
does not access RT0 and uses the following transaction order for the other
three partitions:

```
W512 W8 L48 W128 L32 W32 L16 W4 S48 R12 W64 S32 R192 W16 S16 R560
```

Its 412-byte fragment text SHA-256 is
`2cee690172c6fba77c97ecc9ec8848bffb74313104d2fa40cd21a5e213ba7198`;
the outer native object SHA-256 is
`ab2a1831dc04280aa5fd40e538f5a26306bc60453883dd371ced29b85a99490f`.
This grounds keyed subsets that load and blend each stored partition while
leaving omitted partitions untouched.

The `RG11B10Float` pipeline establishes that Apple's G17P compiler admits the
three-target render layout and emits three scalar 32-bit tile stores. Its
188-byte fragment text SHA-256 is
`d12117c8d8bfc8f4138db1a23480237f25fcdcf88977bf0b7f1da888846fe2c0`;
the outer native object SHA-256 is
`82e9029689e28718b00f93e877b2f013c62c127ff9cd31530f2044810d352039`.
Mesa packs this logical format into raw `R32_UINT` before the tile store and
uses an `R32_UINT` PBE. Therefore its generic multi-target EOT uses the same
grounded scalar-I32 selector (`0x35`) as the existing native raw32 witnesses;
the special two-target R11G11B10 EOT independently uses that same selector.

These are offline compiler witnesses. The corresponding Vulkan CTS cases are
the hardware qualification for the Mesa paths.

## Hardware qualification

On Neo, 2026-09-16, the 69-case unresolved blend/MRT batch passed 69/69 with
the native Vulkan API, no version override, no skipped GPU-killer cases,
zero recorded kernel fault markers, and Hyprland remaining alive.
This covers the targeted monolithic, library and shader-object paths; it is
not a full Vulkan 1.4 mustpass qualification.

- Mesa candidate: `k4-blend-mrt3-r2`, SHA-256
  `daffdeba4c6e63e33e24a67ee385a9026c11f6506c82b8dab297cdcc0bfca028`.
- DRM: `3e0a5ee07e0b1` integrated runtime, module SHA-256
  `c3080a3c4674bcf085927037d0869b613dd88962c3f0ca0c88a7b9a4d563f4cf`.
- Boot ID: `a6a1aae8-4d7b-4d09-adb6-d373505405c1`.
- Evidence: `/opt/neo-gpu/results/blend-unresolved69-graph-r12`, including
  the complete case list, individual results, hashes and kernel log.

The packaged builder records base `a8fb2f6836da` plus dirty-patch SHA-256
`4222313c225599e35e404e33baf36d7eb0b486346e53552cd422c152b25ebeb9`.
That exact patch hash was rechecked, and all eight changed C/C++ source files
match this tree, including the physical-format clear fix already committed
as `4d05233b3337`. The older base recorded in the manifest does not omit that
fix. The MRT epilog, quad and mixed-format host tests also pass (3/3).
