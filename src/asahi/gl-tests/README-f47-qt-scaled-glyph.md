# F47 Qt scaled-glyph compositor oracle

F47 isolates the two-stage text path used by a Qt Quick client under
Hyprland/Zink:

1. Qt's distance-field shader samples either an `R8` atlas through `.r` or a
   legacy `GL_ALPHA` atlas through `.a`, then source-over blends premultiplied
   glyph colour into a `BGRA8` client attachment.
2. A compositor shader linearly samples that attachment into a second render
   target.

For each atlas representation, the compositor runs at 1:1, exact 2x, and
exact 2x with a quarter-output-pixel translation. Every output pixel is
checked against a CPU implementation of Qt's smoothstep coverage,
premultiplied blending, quantization to the client attachment, and bilinear
sampling. The tolerance is two in each 8-bit channel.

Local controls:

```sh
./src/asahi/gl-tests/run-f47-qt-scaled-glyph.sh \
  /src/_build-hypr-desktop /src/_build-full --software
./src/asahi/gl-tests/run-f47-qt-scaled-glyph.sh \
  /src/_build-hypr-desktop /src/_build-full --shim
```

The software run must print six exact-pixel lines, one for every atlas/scale
combination. The G17P no-op run validates submission of both indexed Qt glyph
draws and all six compositor draws, then reports that pixels were skipped.

The standalone AArch64 binary accepts no arguments. Under a hardware Zink
environment it must end with:

```text
GL47 Qt R8/.r and GL_ALPHA/.a glyph client 1x/2x/2x+0.25 pixels exact
```

A mismatch names the atlas representation, scale mode, output coordinate,
observed RGBA, and CPU-expected RGBA. That separates client-atlas swizzle from
compositor scaling/interpolation without relying on a screenshot.
