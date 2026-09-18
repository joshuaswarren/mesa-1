# G17P witnesses for shared AGX3B compiler fixes

These two Apple-compiler witnesses were captured while importing Eryk's M4
compiler fixes on 2026-09-16.  They answer whether the same mechanisms apply
to G17P instead of relying on an M4 result or a Mesa-generated golden.

Use MetalToolchain v27.1.5194.15 or newer:

```sh
base=src/asahi/compiler/test/probes/g17p-agx3b-shared-tail
MT=$(dirname "$(xcrun -f metal)")
"$MT/metal" -mmacosx-version-min=26.0 -c "$base.metal" -o probe.air
"$MT/metallib" probe.air -o probe.metallib
"$MT/metal-nt" -arch applegpu_g17p -N "$base.mtlp-json" \
  -platform_version macos 26.0 26.0 probe.metallib -o probe.nat
```

Extract every embedded Mach-O `__TEXT,__text` section as described in
`../../../isa/README-metal-witness.md`.  The relevant fragment texts are:

- `half_uniform_clamp`: 176 bytes, SHA-256
  `a684c7a3c488d8bc3f2c13a679e65eec34c18bcda7ebd4d2c48cdcf6acb84aa4`.
  Its eight `fsel` instructions use the same register field for the compared
  and selected value.  This validates reusing one materialized uniform per
  instruction on G17P, matching G16G.
- `cube_array_register_lod`: 326 bytes, SHA-256
  `3f60f5d4f3d082af78bd7d0f6dd1994cb92ee07dcfcc1206936bfc4c6e82a243`.
  It contains the fused layer/face/LOD `PACK_SMPIR` form at byte 266.  The
  layer register occupies its seven-bit field, so the coordinate vector must
  remain below half 128 on G17P as on G16G.

The uniform-reuse fix changed only the R11G11B10 two-target background-clear
golden in `test-agx-g17p-mrt-mixed.c`: 442 bytes became 394 bytes with FNV-1a
`0xbdc85483d269636f`.  The epilog, reload, clear-plus-reload, and EOT siblings
remained byte-identical.  That narrow change plus the native tied-FSEL witness
justifies the new golden; a G17P chip exception would contradict Apple's code.
