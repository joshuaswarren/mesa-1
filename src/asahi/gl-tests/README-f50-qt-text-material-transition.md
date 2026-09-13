# F50 Qt GL 2.1 text-material transition oracle

The failing full-shell capture differs from the clean minimal SDF fixture in a
specific way F49 does not cover. It runs a real OpenGL 2.1 compatibility
context and mixes explicit `Text.NativeRendering` nodes with default
distance-field text. The capture prepared three distance-field caches and 256
glyph vertices, but its only emitted text shader was Qt's native 8-bit alpha
mask program. Its thick colored notification halo is also the visual signature
expected when an SDF texture is sampled as an ordinary alpha mask.

F50 alternates those two Qt materials through one GL 2.1 context:

- `--sdf-only` is the clean minimal-scene control;
- `--native-sdf` draws native, SDF, native, SDF bands, switching program,
  alpha atlas, nearest/linear filtering and incompatible uniform layouts;
- `--forced-wrong-mask` deliberately draws the SDF atlas with the native mask
  shader, providing an exact-pixel reference for the observed halo failure.

The native vertex/fragment sources reproduce the capture line-for-line (apart
from its trailing blank line) from
`VS_a941...glsl` and `FS_8f22...glsl` from
`qs-full-sdf-capture-20260910-0737.tar.gz`. They use the captured two-matrix,
texture-scale, DPR and color layout, alpha `.w` sample, pixel snapping and
`_qt_order` input. The SDF pair follows Qt's distance-field material source but
uses the same GLSL 1.20 compatibility and alpha-swizzled texture conventions.
Compile-time assertions pin Qt's native offsets (0/64/128/136/144, 160 bytes)
and SDF offsets (0/64/80/96/100, 104 bytes). This matters even though QRhi's
GL 2.1 backend expands those logical blocks into classic struct uniforms.

Every mode recreates both bounded 128x16 alpha atlases, orphans shared vertex
and index buffers, and issues four indexed glyph draws for 64 cycles without a
readback in the loop. `--finish-between` is a fully serialized control;
`--zink-sync` leaves the application tight and enables Zink's pre-draw barrier.

Run inside the AArch64 Mesa builder mounted at `/src`:

```sh
./src/asahi/gl-tests/run-f50-qt-text-material-transition.sh \
  /src/_build-hypr-desktop /src/_build-full \
  --software --sdf-only --tight
./src/asahi/gl-tests/run-f50-qt-text-material-transition.sh \
  /src/_build-hypr-desktop /src/_build-full \
  --software --native-sdf --tight
./src/asahi/gl-tests/run-f50-qt-text-material-transition.sh \
  /src/_build-hypr-desktop /src/_build-full \
  --shim --native-sdf --tight
```

The runner forces `MESA_GL_VERSION_OVERRIDE=2.1` and rejects a non-AArch64
binary. Software modes validate every pixel against CPU mask/SDF coverage and
premultiplied source-over blending. No-op modes accept success only with the
matching completion marker and no invalid-stream, device-loss or GL-error
diagnostic.

If `sdf-only` is exact but `native-sdf` corrupts, the missing discriminator is
the cross-material program/uniform/alpha-atlas transition rather than cache
size. Comparing `native-sdf` with `forced-wrong-mask` distinguishes stale mask
program state from unrelated corruption. If both hardware modes are exact,
the remaining full-shell factor is above this GL state sequence, such as Qt's
pipeline cache or command ordering across its multiple QRhi contexts.
