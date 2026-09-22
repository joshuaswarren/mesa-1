# G17P packed unsigned-scaled store witness

This probe asks Apple's compiler to round floating-point values, clamp packed
10:10:10:2 channels to their individual unsigned ranges, convert them to
integers, and store through a texture. RGB clamps to 1023; alpha clamps to 3.
Honeykrisp needs the same per-channel conversion before writing the integer
physical render target for Vulkan packed USCALED attachments.

The probe was compiled offline on 2026-09-16 with Apple metal 32023.917 and
AIR-NT 32023.917, targeting `applegpu_g17p`:

```sh
xcrun metal -c packed-scaled-store.metal -o packed-scaled-store.air
xcrun metallib packed-scaled-store.air -o packed-scaled-store.metallib
xcrun metal-nt -arch applegpu_g17p -platform_version macos 27.0 27.0 \
  -N store-rgb10a2.mtlp-json packed-scaled-store.metallib \
  -o store-rgb10a2.nat
```

The G17P native object is SHA-256
`c89c1ffeb41adc2b837e4fb171b3a8980edae43413e849033d7b938ec903e294`.
Its 386-byte `__TEXT,__text` section is SHA-256
`7b4e7b52257ae7e9079b0a1bb99508a6a868e5a590ad091ef99ba1b288da4e8a`.
Input identities are:

```text
a256df0d082bbeb228865af7297d75ff07b21f6b1d6d3c55d24548acbbe9f57a  packed-scaled-store.metal
bf0d7b5aca1c691dcf8f0ba39734a836f4330a4861d6e0ddb0da772ca1679cb9  store-rgb10a2.mtlp-json
```

This is an Apple-compiler witness for the exact G17P round, per-channel clamp,
integer-convert, and texture-store mechanism. Metal texture arguments are not
specialized by destination pixel format, so the Vulkan format mapping remains
Honeykrisp state. CTS execution supplies the hardware result.
