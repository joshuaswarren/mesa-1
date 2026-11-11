# G18G uniform copy to high registers

llama.cpp prompt-processing shaders on G18G move a 32-bit uniform into
half 130. The copy encoder refused every G18G destination at half 128 or
above, so the shader failed to compile. This probe keeps 120 device values
live and seeds 48 loop-carried accumulators from a constant struct, so
Apple's compiler copies a uniform into a high register at loop entry.

Reproduce the native references in an output directory with Metal toolchain
32023.922 (metalfe-32023.922.4, macOS build 25G83). Substitute the path to
this directory for `$source`:

```sh
xcrun metal -std=metal3.0 -mmacosx-version-min=27.0 \
  -c "$source/uniform-copy-high-registers.metal" -o probe.air
xcrun metallib probe.air -o probe.metallib
xcrun metal-nt -arch applegpu_g18g -platform_version macos 27.0 27.0 \
  -N "$source/uniform-copy-high-registers.mtlp-json" probe.metallib \
  -o probe.gpubin
```

| Target | `gpubin` SHA-256 |
| --- | --- |
| G18G | `737f4b95abe9b1031ff8ccefcceb70a0e64dbe667f50be5be27bf32efcd3df21` |
| G17G | `d1a418295fb84fbb35d71e5dbdc16fc8b859f3fd50ee7957f57cb18b7958b6d3` |

The G18G program holds two `copy` words. The uniform copy is:

| Destination | Uniform | G18G offset | Native bytes |
| --- | --- | --- | --- |
| 136 | 102 | `0x0dee` | `a308660a100000003000` |

The agx3d table decodes it as form 121 with f2=136, f9=2, f10=2, f17=102,
f23=3, f24=2, f28=3, f29=35 and f42=1. That is the compact copy's field
vector plus f42=1. The compact form's f2 can also name half 136, but Apple
does not use it there. The G17G program holds no `copy` word, so the AGX3c
bound does not change.

Mesa packs G18G shaders as AGX3c words and re-encodes them for G18G. For
these destinations it packs the AGX3c eight-byte `copy` with the same vector
and f42=1. `agx3_transcode_agx3d()` composes that vector in Apple order as
form 121, which gives the native bytes above.

This evidence admits the G18G uniform-to-register copy on halves 0 to 255. It
does not establish hardware or CTS qualification from native byte
equivalence alone.
