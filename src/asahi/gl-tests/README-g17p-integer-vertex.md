# G17P integer vertex oracle

Native Gallium now uses the G17P raw vertex prolog already used by Honeykrisp.
AGX3 loads return packed raw bytes and halfwords; the legacy prolog expects
formatted loads to widen each component. Chromium's Skia glyph atlas uses
`R16G16_UINT` coordinates, so the legacy path reads `(x, y)` as
`(x | (y << 16), 0)` and samples outside the populated glyph region. The text
disappears even though the atlas upload is correct. AGX2 keeps its existing
prolog.

Build and run against native Mesa on Neo:

```sh
cc -O2 test-g17p-gl-vertex-integer.c -o vertex-integer -lEGL -lGLESv2
ASAHI_GL_EXPECT_RENDERER='A18 Pro' ./vertex-integer
```

The surfaceless GLES 3 test checks all pixels in 48 signed/unsigned integer
vertex cases: 8/16/32-bit storage, one through four components, and two element
offsets. It verifies sign extension and the default missing components using
an integer framebuffer, without floating-point tolerance. The unpatched Neo
passes 36/48; the raw-prolog fix passes 48/48.

Hardware validation also samples a partially populated 2048-square R8 atlas
through Skia-shaped U16 coordinates and checks Chromium's UI and a local page
with sans, serif, monospace, and large text. Chromium must use native GPU
rendering with its normal sandbox, no `no16` setting, and no cache-disable flag.
Software fallback after a GPU-process crash is not a passing result.
