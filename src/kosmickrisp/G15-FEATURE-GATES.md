# Chip-gated feature predicates vs the G15 / gen-15 family

Audit scope: every chip-generation gate in `src/kosmickrisp`, with the
G15 (next Apple family, `gpu_apple_family >= 11`) column. Written for the
Q-99 audit row; the two files named there (`kk_physical_device.c`,
`kk_device.c`) plus every other gate site in the driver are tabled.

## Family numbering

`gpu_apple_family` comes from `mtl_device_get_gpu_apple_family()`
(`kk_physical_device.c:1052`). In this tree: 7/8 = the G13 pair
(M1/M2-era), 9 = M3+, 10 = M5, and G15 is the unnumbered successor
(11). No site pins the number 11 yet; every site below is either
open-ended or an exact match on 7/8.

## Gate table

| Site | Predicate | What it gates | G15 admits today? | Notes |
|---|---|---|---|---|
| `kk_physical_device.c:90` | `gpu_apple_family >= 10` | `EXT_sampler_filter_minmax` feature flag | yes, open-ended | G15 inherits the M5 treatment |
| `kk_physical_device.c:229` | `gpu_apple_family >= 10` | `depthBounds` feature flag | yes, open-ended | same |
| `kk_physical_device.c:1113` | `gpu_apple_family >= 10` | re-enables workaround 2 on M5+ (`disabled_workarounds &= ~BIT(2)`) | yes, open-ended | workaround state, not a feature |
| `kk_physical_device.c:1116` | `gpu_apple_family < 10` | disables workaround 16 on pre-M5 | yes, open-ended | G15 skips the branch |
| `kk_physical_device.c:1120` | `gpu_apple_family < 9` | disables workaround 18 on M3+ ("M3+ only") | yes, open-ended | G15 skips the branch |
| `kk_image.c:207` | `gpu_apple_family >= 10` | mip-map size limit 32768 vs 16384 | yes, open-ended | limit bumps with family 10+ |
| `kk_image_layout.c:98` | `gpu_apple_family >= 10` | an image-layout path | yes, open-ended | one conjunct of a larger condition |
| `kk_shader.c:579-580` | `== 7 \|\| == 8` | a pre-G13-only compiler workaround | yes, excluded by design | the only exact-match gate; G15 correctly falls outside |

`kk_device.c` contains no chip-gated predicate: its behavior is uniform
across families (queue setup, draw dispatch, memory plumbing).

## The row's named features

`cdm_flush` (gen-14), cooperative matrix (`coopmat`), and the hoist flags
do not exist anywhere in this tree — those are the honeykrisp (`hk_*`)
fork's names. This tree is kosmickrisp (`kk_*`); its equivalents are the
`>= 10` feature-flag rows above. There is no gen-14 gate and no
cooperative-matrix feature flag to audit here.

## Conclusion

- Every gate that a G15 device can reach is `>=` or `<` against 9 or 10,
  so a G15 enumerates with the family-10 feature set: minmax filtering,
  depth bounds, the 32768 image limit, and the family-10 workaround
  state.
- The single exact-match gate (`kk_shader.c:579`) excludes G15
  deliberately and correctly: it guards the two oldest families.
- Nothing silently excludes G15. The live risk is the opposite one:
  G15 silently inherits the M5 assumptions (workaround 2 re-enable,
  depth bounds, the raised image limit). Those need verification on
  real G15 silicon before the family number is minted; none of the
  sites above refuses to enumerate it.
