# F54: offline Firefox WebGL coverage for G17P

F54 is a bounded, network-free qualification bundle for the Firefox 151
WebGL path through Zink and Honeykrisp. It does not launch or contact Neo.

## What is covered

`test-g17p-gl-firefox-webgl.c` is the driver-side discriminator. It creates an
ES 3 context, compiles both ESSL 1.00 (WebGL 1) and ESSL 3.00 (WebGL 2)
programs, and performs U16-indexed full-screen draws. The WebGL 2 vertex
shader combines `gl_VertexID`, integer arithmetic, `texelFetch`, and
`textureLod` against a nonuniform 4x4 texture. Three dynamically selected
texels have the same exact sentinel so the pixel result remains deterministic
without reducing the texture operation to the old 1x1 case. Twelve bounded
compile/link/delete cycles exercise pipeline churn, followed by twelve exact
draw/readbacks. Total memory is under one MiB and every command has a hard
timeout.

`firefox-g17p-offline.html` performs the same public WebGL/WebGL2 workload in
a normal browser page. It uses only a `file:` URL and exposes the complete
result as `window.__f54ResultPromise`.

`run-f54-firefox-offline.py` creates a fresh disposable profile, drives a
local Firefox through Marionette, saves JSON, stderr, and a screenshot, and
fails closed on wrong pixels, fewer than 12 WebGL2 draws, a software fallback
when requested, or any AGX compiler/pipeline/device-loss signature. It never
uses SSH or an HTTP server. The report records the Firefox version, page hash,
preferences, and an allowlisted Mesa/Wayland/Vulkan environment snapshot:

```
python3 run-f54-firefox-offline.py \
  --firefox /usr/bin/firefox \
  --output /tmp/new-f54-firefox-result \
  --require-hardware --require-g17p-zink
```

The browser process must be launched with the intended Mesa environment by
the caller. The script records the exact argv and fresh result; it does not
guess or alter driver paths.

## Driver-side controls

From the AArch64 Mesa build container:

```
./src/asahi/gl-tests/run-f54-firefox-g17p-offline.sh \
  ./_build-hypr-desktop ./_build-full --software

./src/asahi/gl-tests/run-f54-firefox-g17p-offline.sh \
  ./_build-hypr-desktop ./_build-full --shim

./src/asahi/gl-tests/run-f54-firefox-g17p-offline.sh \
  ./_build-hypr-desktop ./_build-full --compiler
```

The software control requires exact pixels. The no-op G17P run skips pixels,
rejects compiler or command-stream errors, requires the explicit F54 success
marker, and requires exactly 13 indexed VDM validations (one WebGL1 plus
twelve WebGL2 draws).
The compiler mode builds and runs the existing exact Firefox instruction,
control-flow, indexed vertex-texture, and six-range graphics-program tests.

## Gap analysis and local evidence

The older browser probe used a 1x1 texture and one WebGL2 draw. It did not
cover WebGL1 shader translation, meaningful texel coordinates, shuffled U16
indices, explicit vertex LOD/fetch together, fragment integer/control flow,
resource retirement, or repeated compile/link pressure. F54 adds those
discriminators while retaining exact output.

Two saved Firefox compiler captures identify the historical first failures:

- the vertex capture refused an `iadd` with a cached 32-bit destination,
  sign-extended 16-bit register source, and immediate zero;
- the fragment capture refused `if_fcmp` with a full register and minifloat
  immediate before control-flow constants were materialized.

Current source contains focused regressions named
`FirefoxVertexHalfSignExtendUsesNativeIadd`,
`FirefoxFragmentSignedHalfAccumulatorUsesNativeIadd`,
`FirefoxHalfArithmeticShiftMatchesNative`, and
`FirefoxFloatIfMinifloatsMaterializeToGroundedRegisters`, plus native indexed
vertex texture fetch/sample compile cases.

Local AArch64 results on 2026-09-10:

- llvmpipe: `F54 PASS: Firefox WebGL/WebGL2 pixels exact`;
- Zink/Honeykrisp G17P no-op: `F54 PASS: G17P Firefox WebGL submission
  validated; pixels skipped`, with exactly 13 indexed VDM validations;
- `agx3_iadd_core_test`: 196 checks, 0 failures;
- `agx3_alu_core_test`: 71 checks passed (13 deliberately unsupported forms);
- `agx3_graphics_compile_test`: vertex and fragment shaders compile;
- focused `agx3_pack_test --gtest_filter="*Firefox*:*VertexIndexedTexture*"`:
  10,232,294 proven-subset pack checks, 0 failures.

That September run did not include live Firefox because no local Firefox
binary was available and Neo was explicitly out of scope.

## Native Gallium staging regression (2026-10-03)

The same driver-side WebGL2 shader exposed a native G17P failure on a MacBook
Neo: combining `texelFetch` and `textureLod` returned `{2,5,106,128}` rather
than `{17,123,201,255}`. Each texture operation passed in isolation. Fetches
and filtered samples tracked their shared sampler-private banks independently,
allowing one asynchronous consumer's coordinates to be overwritten by the other.

The packer now accounts for both consumers in the shared staging-bank ledger.
It selects a free bank when possible and waits only when an overlapping bank
must be reused. `FetchAndSampleKeepIndependentStagingBanks` covers both operation
orders and checks that a free bank avoids a full drain. The regression fails
without this fix, passes with it, and the exact-pixel WebGL2 test passes on Neo.
Native OpenGL Minecraft 26.3 also renders its world and UI on that device with
the graphics-uniform-range and dword-alignment fixes applied. This is bounded
application evidence, not an OpenGL conformance claim.
