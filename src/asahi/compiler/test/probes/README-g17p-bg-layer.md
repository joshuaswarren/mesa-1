# G17P background layer input

The paired builtin BG reload inputs differ only in the array-key field.
They were generated from the retained UMD background serializer using the
installed `metal-nt -arch applegpu_g17p` builtin factory. These are offline
compiler witnesses, not runtime captures.

The array form `g17p-bg-d1-m1.btnf.hex` introduces `34c31046`, a read of
normalized SR12 into h6 on slot 3. This is the native 16-bit BG layer input,
not normalized SR28 used for compute-grid Z. The layer then feeds the array
sampler input. `AGX3SrCore.G17PBackgroundLayerInput` pins the read bytes, and
the mixed-MRT test compiles a layered BG reload and requires that read.

Reproduce by converting each hex fixture to binary and running:

```sh
xcrun metal-nt -N input.btnf -arch applegpu_g17p \
  -platform_version macos 26.0 26.0 -o output.nat
```

Neo's original layered secondary-command-buffer reproducer passes after this
fix. All 30 layered geometry target cases and ten controls pass without faults.
No change to draw counts, batching, or the ordinary fragment path is required.
