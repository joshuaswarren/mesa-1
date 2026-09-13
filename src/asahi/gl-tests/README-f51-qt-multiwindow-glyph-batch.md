# F51 Qt GL 2.1 multi-window glyph batching oracle

F50 proved that a native-mask to distance-field transition in one context is
well-formed. The full Omarchy shell adds two driver-visible dimensions:

- the capture's threaded render loop created one QRhi/OpenGL context per
  window (three distinct QRhIs were logged before the notification);
- each render context owns glyph caches whose OpenGL resize path allocates a
  replacement, uploads a full zero image, uploads the preserved CPU image and
  then applies all queued glyph subimages before the first material draw.

F51 creates either one or three **unshared** EGL OpenGL 2.1 contexts, matching
the default QRhi setup when `Qt::AA_ShareOpenGLContexts` is not enabled. Every
context owns its programs, two SDF caches, native cache, BGRA image, buffers
and pbuffer. Context-local GLuint names intentionally overlap, exercising
state restoration when independent QRhi windows submit to one driver/device.
It time-slices those contexts deterministically on one CPU thread. This keeps
the oracle and no-op validator reproducible while preserving GPU-visible
context-local namespaces and independently flushed submissions. Host
render-thread concurrency remains a later discriminator if F51 is clean.

Each 128x64 frame has four batches in notification-like order:

1. native 8-bit alpha-mask text with nearest filtering;
2. SDF font/material A with linear filtering;
3. a premultiplied BGRA image with linear filtering;
4. SDF font/material B with different thresholds.

Each band contains 16 glyph quads. The combined frame is 256 vertices and 384
U16 indices, matching the failing capture's 256 prepared glyph vertices while
remaining bounded. There are 16 frames. Growth stops at 128x144; deferred old
textures are retained for two frames, so memory remains below one MiB.

The workloads form a discriminator ladder:

- `--single-window-growth`: cache growth and batching without context changes;
- `--multi-window-static`: three independent contexts without cache growth;
- `--multi-window-growth`: the full combined sequence.
- `--multi-window-draw-only`: initializes immutable resources, waits for them,
  prints `draw-phase-begin`, then performs only context switches, clears and
  draws. This is the hardware-safe discriminator to try while the unrelated
  first-client Zink async-compute timeout remains unresolved. It removes all
  post-marker texture and buffer uploads; it cannot promise that Zink will
  never select an internal compute path for a draw.

`--tight` flushes each multi-window frame without waiting, like independently
paced window frames; the single-window control defers submission until its
final validation, as F50 does. `--finish-between` serializes every window. The
runner's `--zink-sync` keeps the application tight while asking Zink for its
pre-draw barrier.

Run in the AArch64 Mesa builder mounted at `/src`:

```sh
./src/asahi/gl-tests/run-f51-qt-multiwindow-glyph-batch.sh \
  /src/_build-hypr-desktop /src/_build-full \
  --software --multi-window-growth --tight
./src/asahi/gl-tests/run-f51-qt-multiwindow-glyph-batch.sh \
  /src/_build-hypr-desktop /src/_build-full \
  --shim --multi-window-growth --tight
```

Software runs validate every final-window pixel against CPU alpha-mask, SDF
smoothstep and premultiplied BGRA references. No-op runs require the exact
completion marker and reject invalid G17P streams, device loss and GL errors.

Interpretation:

- only `multi-window-static` failing isolates context/pipeline state restore;
- only growth variants failing implicates replacement/upload ordering;
- `multi-window-growth` failing while both controls pass requires their
  interaction;
- tight failing while either serialization mode passes implicates submission
  visibility/ordering, not text math;
- all F51 modes passing leaves Qt's higher-level threaded timing, render-loop
  lifecycle or QSG batch construction as the remaining discriminator.

The sequence is grounded in the local Qt 6.11.2 sources:

- `qsgthreadedrenderloop.cpp` creates a QRhi for each render-window thread;
- `qsgrhitextureglyphcache.cpp` disables texture-copy resize on OpenGL, queues
  full zero and preserved-image uploads, then appends glyph subimages;
- `qsgdistancefieldglyphnode_p.cpp` merges those uploads into the render
  resource-update batch before binding the linearly filtered glyph texture.
