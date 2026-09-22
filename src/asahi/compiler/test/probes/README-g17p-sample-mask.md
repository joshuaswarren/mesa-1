# G17P sample mask and alpha coverage

Run `python3 capture-g17p-sample-mask.py NEW_DIRECTORY` on a Mac with the
Metal toolchain. The probe only compiles native G17P code, without GPU execution.
It preserves the MSL, pipeline configuration, compiler output and native objects.

The fragment matches CTS `pipeline.*.multisample.z_export.sample_mask_static_atc`:
alpha is zero on the left and one on the right; SampleMask kills the lower half.
The probe captures mask-only, mask plus alpha coverage, and conditional discard.

The 2026-09-16 capture of mask plus alpha coverage has one MOVMSK. The compiler
combines the two masks before it. Its SR39 word is split into an upper API-mask
nibble and low coverage nibble, then recombined as:

```
mid  = api_mask + (coverage << 4)
word = live_mask + (mid << 4)
```

Native fragment text SHA256:

- mask-only: `f12ee9aa8b05d442ea823017996fc0eed76a7e563c1ae0c643538bc10fe8eb93`
- mask plus alpha coverage: `238ff279f802181fb0ee08ae7a1fdff9d5be06c2a35f7b67548f710cd8791cf4`
- conditional discard: `2ea4ec3910943e3f6c6865c07dabb681ee08174c8026cbe3f7b3fe92166fdf03`

The previous split Mesa program emitted a mask update in the main fragment and
another for alpha coverage in the epilog. The exact CTS case produced a clear
image instead of its surviving quadrant. Passing the late pixel-shaded SampleMask
through the export ABI, then combining it with alpha coverage in the epilog,
fixes the case. The nibble correction alone did not fix it.

Qualification: the exact monolithic case passes, three historical shader-object
cases pass, and the related 81-case gate passes with no GPU faults. Per-sample
alpha-coverage mismatches and sample-interpolation failures remain separate work.
