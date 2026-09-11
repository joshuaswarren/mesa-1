# F49 Qt glyph-cache churn oracle

F49 targets the state transition that distinguishes the corrupt full Omarchy
shell from the clean minimal distance-field fixture. Qt's RHI distance-field
cache grows by replacing its `RED_OR_ALPHA8` texture, initializing the new
allocation, restoring the old allocation from a CPU image, and appending new
glyph subimages. Before the next glyph draw, its material replaces the sampled
texture wrapper and changes `textureScale` in the 104-byte material UBO.

The full shell has hundreds of `Text` nodes, many unique glyphs and multiple
materials. The minimal fixture uses a small stable glyph set, so it need not
exercise cache growth after existing nodes and materials exist.

## Workloads

- `--static-sdf` uses the reconstructed Qt distance-field vertex/fragment
  shaders and final atlas contents, but creates the R8 atlas once.
- `--growth-sdf` recreates the R8 atlas on every cycle using Qt's OpenGLES
resize-workaround upload order: one full zero initialization, one preserved
CPU-image upload, then sixteen separately queued glyph subimage uploads. It
then draws an old atlas region, the newly appended glyphs, a second font/cache,
and the first cache again through four different materials.
- `--growth-mask` keeps the same texture, upload, UBO, indexed geometry, blend,
  scissor, and inter-material transitions but replaces `smoothstep` with a raw
  8-bit mask sample. This is the driver-level counterpart of the clean
  `QML_DISABLE_DISTANCEFIELD=1` full shell.

All modes run for 64 cycles without a readback or finish in the loop. Each
cycle orphans and rewrites the shared vertex, index and aligned UBO buffers,
changes the R8 texture binding/normalization around an RGBA image material, and
uses Qt's six `uint16_t` indices per glyph quad. Final readback checks every
pixel of four asymmetric bands against CPU distance-field or mask coverage and
premultiplied source-over blending.

## Local controls

Run in the AArch64 Mesa builder mounted at `/src`:

```sh
./src/asahi/gl-tests/run-f49-qt-glyph-cache-churn.sh \
  /src/_build-hypr-desktop /src/_build-full \
  --software --static-sdf --tight
./src/asahi/gl-tests/run-f49-qt-glyph-cache-churn.sh \
  /src/_build-hypr-desktop /src/_build-full \
  --software --growth-sdf --tight
./src/asahi/gl-tests/run-f49-qt-glyph-cache-churn.sh \
  /src/_build-hypr-desktop /src/_build-full \
  --software --growth-mask --tight
./src/asahi/gl-tests/run-f49-qt-glyph-cache-churn.sh \
  /src/_build-hypr-desktop /src/_build-full \
  --shim --growth-sdf --tight
```

`--api-barrier` calls `glMemoryBarrier(GL_ALL_BARRIER_BITS)` before every draw.
`--zink-sync` leaves the program in tight mode but enables Zink's draw barrier.
This tells apart an API-visible upload/binding problem from the separate broad
inter-draw ordering problem already exposed by F48.

The strongest hardware discriminator is:

1. `static-sdf` exact;
2. `growth-mask` exact;
3. tight `growth-sdf` corrupt;
4. barriered `growth-sdf` exact or still corrupt.

The first three reproduce the full-shell/minimal-shell boundary. Whether the
fourth changes the result says if the SDF-specific failure needs an inter-draw
visibility break or survives one and instead follows texture/UBO descriptor
replacement. Any mismatch reports its band and exact coordinate.

## Source correspondence

The shader, uniform offsets, texture filtering and update order follow Qt 6's
`QSGDistanceFieldTextMaterialRhiShader` and
`QSGRhiDistanceFieldGlyphCache`. The glyph geometry follows
`QSGDistanceFieldGlyphNode::updateGeometry`: four textured float vertices and
indices `0, 2, 3, 3, 1, 0` per quad. The atlas-growth control deliberately uses
the OpenGLES CPU-copy workaround Qt selects automatically for that RHI backend.

Primary source references:

- `qsgrhidistancefieldglyphcache.cpp` (request/store/resize and OpenGLES
  workaround): <https://codebrowser.dev/qt6/qtdeclarative/src/quick/scenegraph/qsgrhidistancefieldglyphcache.cpp.html>
- `qsgdistancefieldglyphnode.cpp` (four-vertex/six-index glyph geometry):
  <https://codebrowser.dev/qt6/qtdeclarative/src/quick/scenegraph/qsgdistancefieldglyphnode.cpp.html>
- `qsgdistancefieldglyphnode_p.cpp` (104-byte UBO, textureScale, thresholds,
  texture wrapper and filtering):
  <https://codebrowser.dev/qt6/qtdeclarative/src/quick/scenegraph/qsgdistancefieldglyphnode_p.cpp.html>
