# F52 Qt GL 2.1 host-threaded multi-window oracle

The failing QuickShell capture reports Qt's threaded render loop and three
distinct QRhi/OpenGL 2.1 contexts. F51 reproduced their context-local state and
submission order on one host thread. F52 adds the final missing dimension:
three persistent pthreads, each permanently owning one unshared EGL context.

F52 deliberately includes F51's implementation substrate rather than copying
it. Therefore the material shaders, two SDF atlases per context, GL cache
resize sequence, two-frame texture retirement, 256-vertex/384-index batch and
strict CPU pixel reference remain identical. It adds only orchestration.

Schedules:

- `--overlapped`: all three workers cross a start barrier, issue their cache
  updates concurrently, cross an update-complete barrier, then issue and flush
  native-mask → SDF A → BGRA image → SDF B draws concurrently;
- `--serialized`: all workers and contexts stay alive, but six barriers per
  cycle give each window an exclusive update/draw/finish turn.

Workloads:

- `--threaded-static`: repeated full/subimage updates without reallocation;
- `--threaded-growth`: staggered replacement growth for both SDF caches in
  every context, including full zero, preserved image and glyph subuploads.

Both run for 16 cycles with three 128x64 targets. Each cache stops at 128x144,
and old textures retire after two cycles. Total retained test data remains
below one MiB. A hard 30-second execution timeout and 60-second build timeout
bound every runner invocation.

Run in the AArch64 Mesa builder mounted at `/src`:

```sh
./src/asahi/gl-tests/run-f52-qt-threaded-multiwindow.sh \
  /src/_build-hypr-desktop /src/_build-full \
  --software --threaded-growth --overlapped
./src/asahi/gl-tests/run-f52-qt-threaded-multiwindow.sh \
  /src/_build-hypr-desktop /src/_build-full \
  --shim --threaded-growth --serialized
```

Software mode reads every final target and validates alpha-mask, SDF
smoothstep and premultiplied BGRA output within two UNORM steps. The no-op gate
requires the exact F52 completion marker and rejects invalid command streams,
device loss, GL errors and crashes.

The no-op shim's mapping table is process-global and its VM-bind path is not
thread-safe. For `--shim` only, the runner builds F52 with an `ioctl()` mutex
around entry into that shim. This prevents validator bookkeeping races without
serializing GL updates, draw recording or flushes. The interposer is not
compiled into software or real-hardware runs.

Interpretation:

- static-overlapped failing isolates host-thread/context state restoration;
- growth-overlapped failing while static-overlapped passes adds resource
  replacement and deferred retirement;
- overlapped failing while serialized passes isolates concurrent submission
  rather than glyph math or command construction;
- both hardware schedules rendering exact leaves QSG batch construction or
  real Wayland swapchain timing above this oracle as the next discriminator.
