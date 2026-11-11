# G17G and G18G exp2 on high registers

llama.cpp `soft_max_f32` on G18G computes `f32exp2` in place on half 134.
The funop encoder refused every register at half 128 or above, so the shader
failed to compile. This probe keeps 120 values live, so Apple's compiler
allocates `f32exp2` operands across the whole register file.

Reproduce the native references in an output directory with Metal toolchain
32023.922 (metalfe-32023.922.4, macOS build 25G83). Substitute the path to
this directory for `$source`:

```sh
xcrun metal -std=metal3.0 -mmacosx-version-min=27.0 \
  -c "$source/exp2-high-registers.metal" -o probe.air
xcrun metallib probe.air -o probe.metallib
xcrun metal-nt -arch applegpu_g18g -platform_version macos 27.0 27.0 \
  -N "$source/exp2-high-registers.mtlp-json" probe.metallib -o probe.gpubin
```

Repeat the final command with `-arch applegpu_g17g` and another output path
for the G17G reference. The program SHA-256 values were:

| Target | `gpubin` SHA-256 |
| --- | --- |
| G18G | `f30ecec88959cb0da19d1cff3af4ea4a988e27096838a84871380521821666ca` |
| G17G | `0d2f6bc6b80516baa71a3fafa48b7b3d1a26708c522942235428433b1625b559` |

Each program holds 120 `f32exp2` words. In the G18G program, 31 have a
destination at half 128 or above and 49 have such a source. The highest
destination is half 214 and the highest source is half 250. These words
carry no wait, and the G17G program holds the same bytes:

| Destination | Source | G18G offset | G17G offset | Native bytes |
| --- | --- | --- | --- | --- |
| 134 | 90 | `0x10ca` | `0x107a` | `37020caa2a20a50a3100` |
| 38 | 248 | `0x0fee` | `0x0fa8` | `370004ea2220a52a3300` |
| 128 | 170 | `0x113a` | `0x10f0` | `27020c6a2220a5023200` |
| 132 | 132 | `0x1722` | `0x16fe` | `27000c1a2a20a50a3200` |
| 166 | 166 | `0x1682` | `0x1652` | `37020c5a2a20a52a3200` |

Field 2 holds the destination and field 18 the source, 9 bits each. Field 29
marks the last use of the source. Field 11 caches the destination where a
later instruction reads it again. The agx3c table composes these field
vectors to the same bytes as the agx3d table, so the G18G transcode passes
them through unchanged.

This evidence admits the G17G and G18G funop runtime forms on halves 0 to 255.
It does not change the AGX3b bound, and it does not establish hardware or CTS
qualification from native byte equivalence alone.
