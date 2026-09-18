# G17P packed 5:5:5:1 attachment-store witness

The render pipelines in this directory isolate Apple's fragment/EOT path for
the packed 16-bit `A1BGR5Unorm` and `BGR5A1Unorm` Metal formats, with
`RGBA16Float` as the 16-bit float-lane control.  They were compiled offline
on 2026-09-16 with Apple metal 32023.917 and AIR-NT 32023.917, targeting
`applegpu_g17p`:

```sh
xcrun metal -c packed-5551.metal -o packed-5551.air
xcrun metallib packed-5551.air -o packed-5551.metallib
xcrun metal-nt -arch applegpu_g17p -platform_version macos 27.0 27.0 \
  -N a1bgr5.mtlp-json packed-5551.metallib -o a1bgr5.nat
xcrun metal-nt -arch applegpu_g17p -platform_version macos 27.0 27.0 \
  -N bgr5a1.mtlp-json packed-5551.metallib -o bgr5a1.nat
xcrun metal-nt -arch applegpu_g17p -platform_version macos 27.0 27.0 \
  -N rgba16float.mtlp-json packed-5551.metallib -o rgba16float.nat
```

Both native objects contain the same 112-byte fragment text, SHA-256
`3375bab60bd450e6303a23d93b9fb8976c6e0eaa4dfc5163dd3f5697a3bf4430`.
Only the pipeline descriptor changes with the packed memory channel order.
Both end in the same 12-byte `lstcol` encoding as the independently compiled
RGBA16Float control (`e70654000000010e00000000`).  This is direct
Apple-compiler evidence that the packed formats use its native fragment-store
lane class.  The existing G17P builtin sweep identifies that class as the
two-word F16 store with IMGWBLK selector `0x15`.

The complete native objects have SHA-256 values
`f372025eae3012c267dd18f529d56ed0f75679df63da6b89e6db755e840766d9`
for A1BGR5 and
`e0fad1f646b005ccb24f229caafcf35ef2fc3f890c7d61fb43917e5dc0a18dd6`
for BGR5A1.  The RGBA16Float control object is
`9aa0c5addeecfe699397b437618078df89971088f79ba38db6a96c62f9d7efbf`;
its 138-byte fragment text is
`6975831af852460c2542c3b3e656606d0bcb2329f7908ff1b60d6a51aec14b94`.
Input identities are:

```text
fc97173b908003643cd30115afc03cb432ddcd38b930c3818bd54fb55bf1c0d7  packed-5551.metal
dcd75c635001cd9ae8332eabf6ef1eb48fca7a57d4a1262851fccdb9881107e0  a1bgr5.mtlp-json
2f72a6460a3768570e8d8c386f177b7d11947e36c3b811764849ce0779802627  bgr5a1.mtlp-json
b8c4dc3a7bf728bcf5f52af5e8c611704191aed2f8583041557d8c7e4b3242e5  rgba16float.mtlp-json
```

Vulkan `R5G5B5A1_UNORM_PACK16` and `B5G5R5A1_UNORM_PACK16` map to
Gallium's low-alpha `A1B5G5R5_UNORM` and `A1R5G5B5_UNORM` formats.  Those
logical aliases therefore need the same two-word F16 EOT key as the already
supported high-alpha packed forms; channel packing remains the PBE's job.
This witness does not claim hardware execution or a CTS result.
