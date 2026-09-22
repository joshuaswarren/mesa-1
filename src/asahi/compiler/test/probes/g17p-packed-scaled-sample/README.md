# G17P packed unsigned-scaled sample witness

This probe asks Apple's compiler to recover the numeric texel domain of a
normalized 10:10:10:2 texture sample. RGB uses `1023.0`; alpha uses `3.0`.
That is the per-channel conversion needed when Honeykrisp samples Vulkan
`A2R10G10B10_USCALED_PACK32` or `A2B10G10R10_USCALED_PACK32` through the
corresponding normalized hardware texture format.

The probe was compiled offline on 2026-09-16 with Apple metal 32023.917 and
AIR-NT 32023.917, targeting `applegpu_g17p`:

```sh
xcrun metal -c packed-scaled-sample.metal -o packed-scaled-sample.air
xcrun metallib packed-scaled-sample.air -o packed-scaled-sample.metallib
xcrun metal-nt -arch applegpu_g17p -platform_version macos 27.0 27.0 \
  -N unpack-rgb10a2.mtlp-json packed-scaled-sample.metallib \
  -o unpack-rgb10a2.nat
```

The G17P native object is SHA-256
`8ab59d6d617f0bc0ae7b68ad02555a20ca10a779263f593c1e687458f3a2d9db`.
Its 276-byte `__TEXT,__text` section is SHA-256
`f8f44d6cef9502b95547c334624b9d4d9b3843f87333c09153c35f4457b1f740`.
Input identities are:

```text
d5bfc68749fd4e43ab000603d63dd7908ebb95d31a61542a8b723cee6ad1e9e7  packed-scaled-sample.metal
4e6b1343d634699ed1599e2ca2cd0a38056e551e48316083e88e02f63b98a9e2  unpack-rgb10a2.mtlp-json
```

This is an Apple-compiler witness for the exact G17P sample-and-rescale
mechanism. Metal texture arguments are not specialized by source pixel format,
so it does not independently prove a Vulkan format mapping. The format mapping
and descriptor metadata follow the already qualified M1/M2 Honeykrisp path;
G17P CTS execution supplies the hardware result.
