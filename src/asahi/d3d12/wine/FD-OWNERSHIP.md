# Wine fd ownership

The PE CRT and the Linux process have separate descriptor tables. Render-node,
dma-buf and sync-file descriptors returned by unixlib or DRM ioctls belong to
Linux. Their integer values must never be passed to a CRT descriptor function.

| Resource/path | Owner and operations |
| --- | --- |
| Wine render node, successful open | Wine platform retains the fd. Common `agx_close_device` releases BOs/maps before `d12_platform_close_device` closes it through unixlib and sets it to -1. |
| Wine render node, failed common open | Common partial-init cleanup releases only completed stages and retains the fd. Platform closes it through unixlib and sets it to -1. |
| Wine sync-provider failure | Platform invokes its complete close path after successful common initialization. |
| Native DRM render node | Caller retains the fd on failed open. Successful open transfers it to common teardown, which uses native `close`. |
| D3DKMT | No Unix fd: `fd` is -1. The existing common teardown delegates opaque-handle destruction to the D3DKMT winsys; failed common open closes through the caller's D3DKMT cleanup. |
| Imported dma-buf | Caller retains its original fd. Common import queries its size with Unix `lseek`, duplicates with Unix `F_DUPFD_CLOEXEC`, and owns/closes only the duplicate. |
| Exported dma-buf | BO retains the PRIME fd. Export returns a separate Unix CLOEXEC duplicate. The consumer owns that returned fd. |
| Exported sync file | DRM returns a Unix fd. Common sharing closes the temporary fd after import and on error through unixlib. A returned export belongs to its consumer. |
| BO mappings | `os_mmap` and `os_munmap` already forward to unixlib; the direct common `munmap` call uses the Wine inline shim. |

Audit scope: common `agx_device.c`, the Wine platform/DRM bridge and its Unix
call table, the sources selected by the Wine Meson build, and DRM sync provider
`u_sync_provider.c`. The latter borrows the render fd and only calls the bridged
DRM helpers; finalization frees its allocation without closing the fd. Virtio,
native Linux platform, D3DKMT runtime and native test executables are excluded
from the Wine PE build. Common fd helpers select unixlib for Wine and native
POSIX calls for Linux. Generic Mesa `os_file` remains a CRT implementation in
the PE, but no GPU descriptor from the audited paths reaches it. Shader-cache,
trace and dump files use their matching CRT `fopen`/`fclose` operations; they do
not receive GPU descriptors. Shared D3D12 resource/fence platform entry points
still return `E_NOTIMPL` on Wine. No consumer is added for exported Unix fds.

The appended Unix ABI entries for duplication and size queries require shipping
the PE and unixlib from the same build. Existing entry numbers are unchanged.

Build the regression with `tools/build-strict-teardown-test.sh SOURCE OUT`, with
`LLVM_MINGW_ROOT` set on an aarch64 Linux build host. Run `strict-teardown.exe`
as x64 under the experimental Proton tool. Its handler aborts on any UCRT
invalid parameter. The builder verifies a dynamic UCRT handler import so it
shares the driver's handler. It exercises final Release, NULL-output adapter
probes, unsupported feature levels, and unsupported interfaces. No GPU command
is submitted. The two FD markers pause for an external `/proc/PID/fd` snapshot
after warmup and after repeated teardown: render fds must be absent at both,
and the live descriptor set must balance. A strict test with the old DLL must
abort, making the check sensitive to the original bug.

Run `python3 tests/unix-fds.py /absolute/path/to/d3d12.so` on the Linux build host
to check CLOEXEC duplication, a size exceeding 32 bits, shared mapping, close,
invalid-descriptor errors and render-node rejection without touching a GPU.
Hardware runs must use the project's approved hardware queue and GPU guard.
Partial common initialization is statically audited; the strict PE test does
not inject allocation failures into every initialization stage.
