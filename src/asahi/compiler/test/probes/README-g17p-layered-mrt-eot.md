# G17P layered multi-target EOT witnesses

The multi-target end-of-tile programs in `agx3_pack_binary.c` are byte-exact
Apple builtins, and every shipped one stores 2D and non-array
(`README-g17p-raw32-mrt.md`: "All stores are 2D and non-array"). That is why
`agx_g17p_classify_mixed_mrt()` excludes `tib->layered`, and why every
`dEQP-VK.renderpasses.*.multisample_resolve.layers_*` case fails at
`vkEndCommandBuffer` with `VK_ERROR_FEATURE_NOT_PRESENT` once a geometry
shader lets CTS reach it -- the packer reports

    no grounded AGX3 encoding for block_image_store
      [... s3=32:reg8x3 ... dim=3 ...]

for the three-component array coordinate.

Apple's builtin factory does emit the array form. It is selected by the
attachment's **texture type**: 2 is 2D, 3 is the 2D array. The attachment
payload is `00 <format> <type> 00 00 <components> 00`, so a layered witness is
the shipped input with one byte changed per attachment.

```sh
xxd -r -p g17p-native-rgba8-mrt4-array.btnf.hex > in.btnf
xcrun metal-nt -N in.btnf -arch applegpu_g17p \
  -platform_version macos 26.0 26.0 -o out.nat
```

The fragment object is nested in the outer `__TEXT,__builtin`; hash its
`__TEXT,__text`, exactly as `README-g17p-native-rt0-eot.md` describes. The
`.text.hex` files beside each input are that text.

## Witness table

Every 2D row reproduces an artefact already in the tree, byte for byte, so
the array rows rest on the same extraction.

| Targets | Class | Type | Bytes | Text SHA256 | Reproduces |
| --- | --- | --- | --- | --- | --- |
| 2 | raw32 | 2 | 120 | f419419f18a2d8a330657d586336537a0fffa693921e374c76fb4043f3c52e0b | README-g17p-raw32-mrt.md |
| 2 | raw32 | 3 | 150 | 0161736a8ab5f875d6d7523990b574f1095460a45ef632d6cf0bf7ddf60d852f | new |
| 3 | raw32 | 2 | 164 | 37cec0add9eada87ec7aa3b8b10d6007fbabfd38214b9c755156d30bf81499ad | README-g17p-raw32-mrt.md |
| 3 | raw32 | 3 | 204 | 0e45e8506633e84dc0759d84de2ede9be0300ce886b1cf1e506663de1d516f47 | new |
| 4 | raw32 | 2 | 208 | e37d70ea63e4a588ddc29a692dbec01191aa50b2fdaf9cbba0bbf2ba99f2b961 | README-g17p-raw32-mrt.md |
| 4 | raw32 | 3 | 258 | 97cc3a9dba4b126502b03953205c7256620a4b54fe698fa129f84fbcad7aaa96 | new |
| 3 | rgba8 | 2 | 164 | eeb93747c76af1ba5d29d7827677c2b4be7022d6c93da63bc712023a681952c8 | `agx3_g17p_rgba8_3_eot` |
| 3 | rgba8 | 3 | 204 | b2331b7f196fd529a78cb337eeac053d913d34a7d872f738bef2df08baa2b029 | new |
| 4 | rgba8 | 2 | 208 | 2b4d0f4c382275afa49fa9fc322394c5a5d063a3fef50901a1a29beaeac6ab27 | `agx3_g17p_rgba8_4_eot` |
| 4 | rgba8 | 3 | 258 | deb70c1374fc82126a62c2495a1540b7d338157c30d1b8f731b73027fb743737 | new |

Length is `32 + 44 * targets` for 2D and `42 + 54 * targets` for the array
form.

## The four-attachment input

No four-target BTNF shipped. `g17p-native-raw32-mrt4.btnf.hex` is built from
the three-target one by duplicating a 32-byte attachment record, setting the
enabled mask to 0xf and the count to 4, and rewriting the four self-relative
offsets in the vector at 0x58. It is not a guess: unmodified it reproduces the
documented four-target raw32 text, and with the format byte set to 3 it
reproduces the shipped `agx3_g17p_rgba8_4_eot` byte for byte.

## What the array form changes

Aligning the 2D and array programs at two, three and four targets, the array
program is not the 2D one with a patched byte:

- the preamble's second instruction loses its `14` field (`24a81406` ->
  `24a81006`), the fourth gains a `44` (`24a91026` -> `44a91026`), and one
  instruction is inserted after it (`34c31046`) -- the same shape the 84-byte
  single-target layered EOT has;
- the two coordinate setups shift (`9f105402...14` -> `9f105404...10`,
  `9f205400...10` -> `9f205402...20`);
- one array wire is inserted before the first PACK_IMGWIR,
  `97455400030c1000f0880000` -- "layer first, then the XY tile origin", the
  same instruction the layered RT0 program uses;
- every PACK_IMGWIR staging half moves up by two, and the per-target
  element and uniform-copy chains each gain one entry;
- every IMGWBLK lane-class wire byte changes from `a8` to `28`.

That is why the templates need capturing rather than deriving.

## Still to capture

The multisample variants. The shipped 2D templates carry multisample as a
bit patch (`imgwir_wire[rt] |= 0x02`, `imgwblk + 8 &= 0x7f`, selector low
nibble 2); whether the array form patches the same way is **unknown** and must
be captured, not assumed -- the CTS cases that need all this are
`samples_2` layered resolves.

No GPU is involved; this is an offline oracle.
