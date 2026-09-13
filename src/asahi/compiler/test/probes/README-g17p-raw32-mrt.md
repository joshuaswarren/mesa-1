# G17P two-to-four-target raw32 contracts

The native EOT factory accepts two, three and four identical RGBA32 targets
in one imageblock. Signed integer, unsigned integer and float attachment
formats10/11/17 produce identical text at each attachment count. Texture type
stays2, 2D, components stay4 and every attachment uses partition0.

| Targets | EOT bytes | Staging halves | Text SHA256 |
| --- | --- | --- | --- |
| 2 | 120 | 16 | f419419f18a2d8a330657d586336537a0fffa693921e374c76fb4043f3c52e0b |
| 3 | 164 | 24 | 37cec0add9eada87ec7aa3b8b10d6007fbabfd38214b9c755156d30bf81499ad |
| 4 | 208 | 32 | e37d70ea63e4a588ddc29a692dbec01191aa50b2fdaf9cbba0bbf2ba99f2b961 |

The adjacent two/three-target BTNF inputs retain the complete attachment
vector and vtables. Decode a `.btnf.hex` file with `xxd -r -p`, then run:

```sh
xcrun metal-nt -N input.btnf -arch applegpu_g17p \
  -platform_version macos 26.0 26.0 -o output.nat
```

The fragment object is nested in the outer `__TEXT,__builtin` section; hash
its `__TEXT,__text`. Metadata common field0 stays3 GPR dwords. Fields38/42 are
16 times the attachment count, bytes of staging. The ESL has a kind3 qword
load at destination0 and a kind0 load of `2 * count - 1` dwords at destination2.
Driver kind5 arguments select each target at destinations0,2,4,6. The RT store
mask is `(1 << count) - 1`.

The PBE pointer occupies uniform halves0..3. Per-target byte offsets occupy
halves4,8,12,16; unused intervening words are not offset arguments. Each target
has its own aligned32-byte PBE descriptor. Each PACK_IMGWIR names staging half
`4 + 8 * rt` and uniform half `4 + 4 * rt`. IMGWBLK names staging half `8 * rt`
and PBE target `rt`. All stores are 2D and non-array.

Fresh ordinary fragment witnesses retain descending target stores. Two targets
use balanced classes48 and12; three use192,48,12. Entry class512 precedes the
transactions and the final release is524. The four-target path keeps its shared
class12 lifetime, acquired by target3 and released by target0.

Fresh clear/reload constructor witnesses for two and three RGBA32 targets have
ascending stores at byte offsets0,16,32, no shader PIXWAIT/PIXREL, and loader
release512. Mesa preserves its existing clear and texture-pointer ABI. Clear
uniforms use `4 + 8 * count` reserved halves and texture tables retain the
32-byte gap between target entries, indices0,2,4,6.

The generic tile allocator is unchanged: two RGBA32 targets select32x16 tiles,
three and four select16x16. The existing four-target scalar R32 path remains
32x32. Mixed formats, sparse attachments, spills, layered targets and MSAA do
not gain admission through this change. No per-draw synchronization is added.

Compiler and loader regressions validate these contracts and reject malformed
offsets, counts, staging sizes, table alignment and overflow. Native witnesses
are offline evidence; exact pixels still require hardware qualification.
