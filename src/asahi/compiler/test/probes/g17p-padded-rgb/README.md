# G17P padded RGB transfer witness

Optimal three-component Vulkan images use four-component physical storage on
Apple GPUs.  The two kernels in `padded-rgb-copy.metal` isolate the required
packed-buffer transfer in both directions: three byte components at a
three-byte stride and the same components at a four-byte image stride.

The witness was compiled on 2026-09-16 with Apple metal 32023.917 and AIR-NT
32023.917, targeting `applegpu_g17p`:

```
xcrun metal -c padded-rgb-copy.metal -o padded-rgb-copy.air
xcrun metallib padded-rgb-copy.air -o padded-rgb-copy.metallib
metal-nt -arch applegpu_g17p -platform_version macos 27.0 27.0 \
   -N to-padded.mtlp-json padded-rgb-copy.metallib -o to-padded.nat
metal-nt -arch applegpu_g17p -platform_version macos 27.0 27.0 \
   -N from-padded.mtlp-json padded-rgb-copy.metallib -o from-padded.nat
```

SHA-256 identities:

```
463ca37bc5deec7ab287e8923b2d89284f5e1e4cc4f579731196337c2d4b8c0c  padded-rgb-copy.metal
53ba7df407a7ef209f71acc53984236ac495e1846520f284858a0cf58b882ca1  to-padded.mtlp-json
7addbfa0f4debb82ca70e12bb4afe7ecdc11dc8dc004d1dc4359acf2c9aaa898  from-padded.mtlp-json
310e6e825a33952269b0deb99835ba015a93adf2830f7a2e75cf0abbf53101c4  to-padded.nat
9731286ee4a7bb4a2b294dd23d971fc2d7db1a68e9247deab3053703d245f08d  from-padded.nat
```

Both 170-byte native code bodies were decoded with the in-tree AGX3 compose
tables and every reported instruction was re-composed byte-for-byte.  The
to-padded kernel contains the expected integer stride arithmetic, device loads
and device stores; the reverse kernel likewise loads the padded element and
emits three component stores into packed storage.  This is direct evidence
that G17P supports the three-component, 3/4-stride transfer used by Honeykrisp.
Mesa keeps the component accesses explicit to match this witnessed contract.
