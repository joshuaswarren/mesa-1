# G17P padded RGB clear witness

This probe asks Apple's compiler to perform the operation used by Honeykrisp's
optimal RGB image storage: write a logical `uchar3` clear value into one
physical `uchar4` element and set the padding/alpha byte to 255.

Build with the Xcode 27.1 Metal toolchain:

```sh
xcrun metal -c rgb8-clear-fill.metal -o rgb8-clear-fill.air
xcrun metallib rgb8-clear-fill.air -o rgb8-clear-fill.metallib
xcrun metal-nt -arch applegpu_g17p -platform_version macos 27.0 27.0 \
  -N rgb8-clear-fill.mtlp-json rgb8-clear-fill.metallib \
  -o rgb8-clear-fill.g17p.bin
```

`metal-nt` emits a G17P Mach-O GPU executable. The nested compute object has a
174-byte `__TEXT,__text` section, proving that Apple's G17P compiler accepts
and lowers this exact padded write instead of requiring a packed 3-byte
destination.

Reproducibility hashes from 2026-09-16:

- AIR: `6e57cc344cacb8ed917c4b86b6464397eb82728881a7517ea6f88742b59c1c0b`
- metallib: `661b8eb6b322cc226f2a407cedad60b42cf657581cf3f9577b29c4c2111a6e67`
- G17P executable: `93463ac0a488ff0aec7a6792830ec835eb868aa1bb5deed35294647e98d728fb`
- nested compute object: `40d2e2a071feef4509530b62c0534c618e7c11a7b83eee28af559f034f66ad0b`
- native text: `9d1a6997125784751743110cd3ffb2ecbc6fe04a48b208ae2e57179f8ba8ed6b`

The generated binaries are intentionally not checked in. The source,
translator script, commands, target, sizes, and hashes are the durable oracle.
