# F48 Hyprland mixed-draw/state-transition oracle

F48 reproduces the visual paths implicated by the Omarchy screenshots in one
long-lived GLES context, rather than proving them independently:

- Hyprland 0.56.1's exact preprocessed rounded-border shaders;
- an indexed Qt-style R8 glyph mask with transparent, antialiased, and opaque
  coverage values;
- a little-endian `WL_SHM_FORMAT_ARGB8888` client texture uploaded through
  `GL_BGRA_EXT`, including row length and skip state.

It rotates all six pairwise orders for 64 cycles. Between visible passes it
changes viewport, scissor, blend equation/factors/constants, colour mask,
depth, stencil, cull/front-face, triangle/line/point topology, VBO storage,
texture unit/filter/swizzle, and unpack state. There is no readback or finish in
the loop. The final readback checks the four straight border segments, rounded
corner discard, an untouched gap, every asymmetric BGRA texel, and four glyph
coverage levels.

## Synchronization discriminator

The same workload has three synchronization modes:

- `--tight`: no application barrier between draws. This is the closest local
  reproduction of the ordinary Hyprland command stream.
- `--api-barrier`: calls `glMemoryBarrier(GL_ALL_BARRIER_BITS)` immediately
  before every draw. This is a portable, deliberately redundant application
  barrier control.
- `--zink-sync`: keeps the binary in `--tight` mode and sets only
  `ZINK_DEBUG=sync`. Zink then exits the render pass and inserts
  `VkMemoryBarrier { MEMORY_WRITE -> MEMORY_READ }` at
  `ALL_COMMANDS -> ALL_COMMANDS` before each draw. This is the exact control
  corresponding to the hardware observation that `ZINK_DEBUG=sync` removes
  black border marks and survives 30 Foot cycles while the baseline crashes.

If `--tight` corrupts/crashes on hardware while `--zink-sync` is exact and
stable, the result implicates missing inter-draw ordering/cache visibility, not
the border shader math, BGRA channel mapping, or glyph coverage computation.
If both fail in different pixels, the printed probe name separates stale
border state, stale/client texture data, and stale R8 glyph data.

## Build and local controls

Run inside the AArch64 Mesa builder with the repository mounted at `/src`:

```sh
cd /src
./src/asahi/gl-tests/run-f48-hypr-mixed-draw.sh \
  /src/_build-hypr-desktop /src/_build-full --software --tight
./src/asahi/gl-tests/run-f48-hypr-mixed-draw.sh \
  /src/_build-hypr-desktop /src/_build-full --software --api-barrier
./src/asahi/gl-tests/run-f48-hypr-mixed-draw.sh \
  /src/_build-hypr-desktop /src/_build-full --shim --tight
./src/asahi/gl-tests/run-f48-hypr-mixed-draw.sh \
  /src/_build-hypr-desktop /src/_build-full --shim --zink-sync
```

The runner compiles the source directly, even when the selected build directory
predates the optional Meson target, and rejects a non-AArch64 executable. The
software modes must identify llvmpipe and end with an `exact` line. The no-op
Zink modes set `ASAHI_SHIM_VALIDATE_G17P_STREAM=1`; the shim walks every
submitted G17P render stream. The runner accepts success only when the process
returns zero, the matching success marker is present, and the captured output
contains no invalid-stream, device-loss, or GL-error diagnostic. This prevents
a late Zink submit failure from being misreported as a passing no-op readback.
The test ends with:

```text
F48 tight no-op G17P submission validated; pixels skipped
```

For `--zink-sync`, its preceding mode line must say `ZINK_DEBUG=sync`. The
no-op driver deliberately does not execute GPU work, so
pixel readback is skipped there. A nonzero exit, GL error, invalid G17P stream,
missing extension, pixel mismatch, or timeout is a failure.

The standalone executable accepts:

```sh
f48-hypr-mixed-draw hypr-0.56.1-tex300.vert \
  hypr-0.56.1-border-rounded.frag --tight
f48-hypr-mixed-draw hypr-0.56.1-tex300.vert \
  hypr-0.56.1-border-rounded.frag --api-barrier
```
