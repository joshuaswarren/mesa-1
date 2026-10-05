# d3d12agx under Wine/Proton: ARM64EC PE `d3d12.dll` + unixlib DRM backend

Task t-1084, 2026-10-05. Clean-room: Wine's public unixlib ABI, Linux DRM UAPI,
DXVK's public `dxgi_interfaces.h`, our own measurements. No macOS research.

## Goal
Let x86-64 Windows games running under Proton (FEX ARM64EC, 4K opt-in userspace,
no muvm) on the Neo use the native D3D12 driver d3d12agx instead of vkd3d-proton.
First a triangle (D3D12HelloTriangle), then the 17 frozen suites as Windows exes,
then real titles; pass/fail and frame time recorded against the current tool.

## Alternatives considered
| design | what runs where | verdict |
|---|---|---|
| **A. ARM64EC PE `d3d12.dll` = whole Mesa driver; tiny unixlib for kernel calls (chosen)** | DXIL->NIR->AGX compiler, command encoding, BO/VA management all in PE (native AArch64 via ARM64EC, no emulation); unix `.so` only opens the render node and does `ioctl`/`mmap`/`munmap` | One shared process, BO maps read directly by PE code, zero marshalling (ioctl structs are passed by pointer, identical layout both sides). Driver core unchanged; the Windows `_WIN32` aggregate-return ABI already exists. Only new files + 6 one-line gate edits. |
| B. winevulkan-style full COM thunking to `libd3d12agx.so` | PE stub forwards every ID3D12* method to a unix-side COM object | ~1500 methods to thunk; descriptor handles, `Map()` pointers and fence events cross the boundary; PE-side callbacks (destruction notifier, SetEventOnCompletion) need reverse thunks. Weeks of work, worse call latency. |
| C. DXGI/D3D12 split with our own `dxgi.dll` | replace DXVK's dxgi too | Breaks D3D11 games (DXVK d3d11 and dxgi share private interfaces). Rejected; instead we implement DXVK's `IDXGIVkSwapChainFactory` on our command queue so DXVK's dxgi presents us. |

## Design A in detail
* **Build**: Mesa cross build, host `windows`/`aarch64`, compiler
  `arm64ec-w64-mingw32-clang` (llvm-mingw 20260922, aarch64 Linux build), new
  option `-Dasahi-d3d12-platform=wine`. The build host is **not the Neo**
  (no compiles there since 2026-10-06): `build-wine-arm64ec.sh` runs inside
  the Fedora 45 `aurora-f45-mesa` podman image on m1pro (`~/aurora-d3d12/t1084/
  t1084-m1pro-build.sh`), so the unixlib is linked against the Neo's glibc and
  the native host tools (mesa_clc, vtn_bindgen2, asahi_clc) are built from the
  same tree first. The `wine/include` shim directory is on the cross file's
  global `c_args` because `asahi_drm.h` is reached from `agx_chip.h` by the
  compiler library, not only by `src/asahi/d3d12`. The *Linux* DRM body of `agx_device.c`
  (`HAVE_LIBDRM`) is compiled into the PE; virtio is compiled out
  (`AGX_NO_VIRTIO`). `xf86drm.h`, `sys/ioccom.h` (Linux ioctl encoding, not
  BSD), `linux/types.h` and `os_mmap` come from `wine/include/` shims that route
  to the unixlib. `src/util/u_sync_provider.c` (DRM syncobj provider) is compiled
  in through the same shim.
* **Unixlib** `d3d12.so` (plain aarch64 ELF, libc only): calls OPEN (first
  `/dev/dri/renderD*` whose DRM_IOCTL_VERSION name is `asahi`), IOCTL
  (EINTR/EAGAIN retry), MMAP, MUNMAP, SYNCOBJ_WAIT (relative timeout re-based on
  CLOCK_MONOTONIC on the unix side so PE and unix clocks never mix),
  IS_RENDER_NODE, CLOSE. Found by Wine's loader as
  `files/lib/wine/aarch64-unix/d3d12.so` next to `x86_64-windows/d3d12.dll`.
* **Loading**: the PE is marked a Wine builtin (`"Wine builtin DLL"` at DOS header
  offset 0x40, same as `winebuild --builtin`). For an x64 process on an ARM64
  host Wine's `load_builtin` forces the search into `aarch64-windows/` and
  requires the image machine to be AMD64 after ARM64X processing; an ARM64EC
  PE is already machine AMD64, so ours is installed as
  `files/lib/wine/aarch64-windows/d3d12.dll` (the ARM64X Wine d3d12 kept as
  `.wine-orig`), with `files/lib/wine/aarch64-unix/d3d12.so` beside it. The
  COFF header machine field must be `IMAGE_FILE_MACHINE_AMD64` (0x8664):
  wineserver's `is_machine_64bit()`/`is_machine_supported()` know only AMD64
  and ARM64 and refuse anything else with STATUS_INVALID_IMAGE_FORMAT; the
  hybrid nature comes from `CHPEMetadataPointer` in the load config
  (`server/mapping.c`: `is_hybrid`). lld writes 0x8664 for a pure ARM64EC
  link; `llvm-readobj` *reports* such an image as `COFF-ARM64EC` /
  `IMAGE_FILE_MACHINE_ARM64EC`, derived from the metadata, which is why the
  packaging check accepts that string and `mark-wine-builtin.py` verifies the
  raw field instead. The
  tool wrapper sets `WINEDLLOVERRIDES=d3d12=b;d3d12core=b` and the tool's
  vkd3d-proton `d3d12*.dll` are removed so Proton stops copying them into the
  prefix. Unix handle via `NtQueryVirtualMemory(..., 1000 /*MemoryWineLoadUnixLib*/)`;
  on ARM64EC the call must branch straight to ntdll's
  `__wine_unix_call_dispatcher_arm64ec` (a `br x16` stub) because ARM64EC
  indirect calls to non-EC code would be routed to the x64 emulator.
* **Platform backend** `d12_platform_wine.c`: open via unixlib, Win32 events
  (`SetEvent`), shared handles `E_NOTIMPL` (as Linux). `GetAdapterLuid` returns the
  LUID of the DXGI adapter the game passed in.
* **Presentation** `d12_wine_swapchain.c`: `IDXGIVkSwapChainFactory` on
  `ID3D12CommandQueue`; `IDXGIVkSwapChain` whose back buffers are our own
  RENDER_TARGET textures; `Present` executes a texture->buffer copy on the
  game's queue, waits, and blits with GDI `SetDIBitsToDevice`. CPU copy of the
  frame (3.7 MB at 1080p) is the known cost; a Vulkan/dma-buf presenter is the
  follow-up. Vsync: SyncInterval honoured by sleeping to the display period.
* **Gates**: the `#ifndef _WIN32` guards that select Linux behaviour
  (G17P buffer cache ops, baseline/depth opt-ins) become
  `#ifndef D12_PLATFORM_D3DKMT`, so the Wine build behaves like Linux; the
  weak `asahi_noop_*` shim hooks are Linux-only (no weak undefined in PE).
* **Tests**: the frozen suites are already `_WIN32`-clean (`d12_test_abi.h`);
  cross-built as x64 exes with `x86_64-w64-mingw32-clang`, run with
  `proton runinprefix` in the user session. The four D3D12HelloWorld samples
  (Triangle, Texture, Bundles, ConstBuffers) are Microsoft's unmodified
  sources built with llvm-mingw around `samples/aurora_sample_main.cpp`
  (`-DSAMPLE_CLASS=...`): bounded `--frames N`, a `--selftest` back-buffer
  readback (corner must be the clear colour, the triangle must cover a sane
  share of the frame) and one JSON line of frame-time statistics, so
  pass/fail and ms/frame are objective and identical under both tools.
  HelloTexture expects Visual Studio's precompiled `.cso` pair; the entry
  point produces it at start-up through `d3dcompiler_47` (Wine's) so the
  sample source stays untouched. `samples/include/d3dx12/d3dx12.h` is a small
  shim of the public helpers (mingw's d3d12.h predates the real d3dx12.h).

## Running the tests on the Neo (what the first runs taught)
* Every attempt-3 run under *both* tools timed out at 240 s without the test
  exe ever loading. The Proton log showed why: a fresh prefix (wineboot,
  wine-mono, explorer, tabtip) plus Proton's accessibility helper `xalia.exe`
  (.NET under FEX) consumed the whole budget. The harness now warms each
  tool's prefix once with a 9-minute budget and sets `PROTON_USE_XALIA=0`
  for both tools (the `proton` script honours it) so later runs measure the
  driver, not the prefix bring-up.
* The tool wrapper logs `page=16384` for `runinprefix` because it records
  `getconf PAGE_SIZE` *before* re-executing itself through the 4K page
  helper, which then asserts 4096; Steam launches show `page=4096` only
  because Steam already runs in 4K mode. The tests are 4K processes either way.
* The wrapper must not override a caller's `PROTON_LOG_DIR`, otherwise the
  per-run Proton logs vanish into the tool's `aurora-game-logs`.
* `install-proton-tool.sh --refresh` swaps a rebuilt pair + wrapper into the
  installed tool; `SUITE_ONLY` splits the 17 suites into short batches so an
  exclusive hardware request stays under ten minutes.

## Risks / open
* ARM64EC Mesa build has never been done: expect a few compiler/ABI fixes.
  Found so far: clang's arm64ec target predefines `__x86_64__` (not
  `__aarch64__`), so `src/util/detect_arch.h` needed an `__arm64ec__` clause
  or x86 inline asm (bitscan popcnt, cpuid) is emitted for an AArch64 target.
* `drm_handle_t` is `unsigned long` (32-bit on Win64) in the BSD branch of
  `drm.h`; only legacy structs use it, none of ours.
* Wine's 4K-page processes on the 16K kernel already mmap GEM BOs (Honeykrisp
  through winevulkan), so the same path is expected to work for us.
