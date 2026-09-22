# G17P scaled attachment-store witness

Vulkan scaled attachments expose floating-point shader values but use integer
physical storage.  `scaled-store.metal` isolates the required round-to-even,
clamp and float-to-integer conversion for signed and unsigned 8-bit stores.

The witness was compiled on 2026-09-16 with Apple metal 32023.917 and AIR-NT
32023.917, targeting `applegpu_g17p`:

```
xcrun metal -c scaled-store.metal -o scaled-store.air
xcrun metallib scaled-store.air -o scaled-store.metallib
metal-nt -arch applegpu_g17p -platform_version macos 27.0 27.0 \
   -N uscaled.mtlp-json scaled-store.metallib -o uscaled.nat
metal-nt -arch applegpu_g17p -platform_version macos 27.0 27.0 \
   -N sscaled.mtlp-json scaled-store.metallib -o sscaled.nat
```

SHA-256 identities:

```
3ee642ec885e2022c02d6d032371e7cd4ff9c08754b57ba4fcaa3c1d5abf07c2  scaled-store.metal
f63fb3c7031975100729e0c0a16e6235dd6e6e374f5317483d6a8bf6face758e  uscaled.mtlp-json
f8decc955525186edd80fd847e33573dfc8ee21f42a93477f57fb263e9b1fe7e  sscaled.mtlp-json
3a0dfb7580f48496ad7ddc48da83ceee130ceea708cd41e59ee7973a01bf42fd  uscaled.nat
e5710eaa6c4b5501aba7689add615d2766f162bda66ef73b4d04ab6f5e0bcaa4  sscaled.nat
```

Both 300-byte native code bodies were decoded with the in-tree AGX3 compose
tables and every instruction row was re-composed byte-for-byte.  Apple's
unsigned path contains `f32rint`, clamp-select operations, `flt2int`, and the
final device store; the signed path has the corresponding signed bounds and
conversion.  This directly witnesses the conversion sequence used by the
Honeykrisp epilog before integer tilebuffer storage.
