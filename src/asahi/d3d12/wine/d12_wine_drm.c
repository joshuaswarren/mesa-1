/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * PE side of the Wine unixlib bridge: libdrm-compatible entry points and
 * os_mmap/os_munmap, each forwarded to d3d12.so in the same process.
 *
 * The unixlib ABI is Wine's public one (include/wine/unixlib.h): the handle
 * comes from NtQueryVirtualMemory(MemoryWineLoadUnixLib) on our module base,
 * and calls go through ntdll's dispatcher. On ARM64EC the dispatcher lives in
 * ntdll.so (native, non-EC code); an ordinary indirect call would be routed to
 * the x64 emulator, so a bare `br` stub is used, the same way winecrt0 does it.
 */

#include "agx_wine_unixlib.h"
#include "d12_wine_unixlib.h"
#include "xf86drm.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <windows.h>

#include "util/os_time.h"

#define D12_WINE_MEMORY_LOAD_UNIX_LIB 1000 /* MemoryWineLoadUnixLib */

/* The request numbers travel verbatim to the Linux kernel: make sure the
 * asm-generic encoding from the sys/ioccom.h shim is in effect here. */
_Static_assert(DRM_IOCTL_VERSION == 0xc0406400u, "Linux ioctl encoding lost");
_Static_assert(DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT == 0xc03064cau,
               "Linux ioctl encoding lost");

typedef UINT64 d12_unixlib_handle_t;
typedef LONG d12_ntstatus_t;
typedef d12_ntstatus_t (WINAPI *d12_unix_dispatcher_t)(d12_unixlib_handle_t,
                                                       unsigned int, void *);
/* mingw's winternl.h does not declare this; resolve it from ntdll at runtime. */
typedef d12_ntstatus_t (WINAPI *d12_nt_query_virtual_memory_t)(
   HANDLE, PVOID, int, PVOID, SIZE_T, SIZE_T *);

static d12_unixlib_handle_t d12_wine_handle;
static d12_unix_dispatcher_t d12_wine_dispatcher __attribute__((used));
static int d12_wine_state; /* 0 untried, 1 ok, -1 failed */

extern IMAGE_DOS_HEADER __ImageBase;

#ifdef __arm64ec__
/* Must not be reached through an EC indirect-call check: branch straight to the
 * native dispatcher stored in d12_wine_dispatcher. */
__attribute__((naked)) static d12_ntstatus_t
d12_wine_unix_call_raw(d12_unixlib_handle_t handle, unsigned int code,
                       void *args)
{
   __asm__(".seh_proc \"#d12_wine_unix_call_raw\"\n\t"
           ".seh_endprologue\n\t"
           "adrp x16, d12_wine_dispatcher\n\t"
           "ldr x16, [x16, #:lo12:d12_wine_dispatcher]\n\t"
           "br x16\n\t"
           ".seh_endproc");
}
#define D12_WINE_DISPATCHER_EXPORT "__wine_unix_call_dispatcher_arm64ec"
#else
static d12_ntstatus_t
d12_wine_unix_call_raw(d12_unixlib_handle_t handle, unsigned int code,
                       void *args)
{
   return d12_wine_dispatcher(handle, code, args);
}
#define D12_WINE_DISPATCHER_EXPORT "__wine_unix_call_dispatcher"
#endif

bool
d12_wine_unixlib_init(void)
{
   if (d12_wine_state)
      return d12_wine_state > 0;

   HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
   void **dispatcher_var =
      ntdll ? (void **)GetProcAddress(ntdll, D12_WINE_DISPATCHER_EXPORT) : NULL;
   if (!dispatcher_var || !*dispatcher_var) {
      fprintf(stderr, "d3d12agx: not running under Wine (%s missing)\n",
              D12_WINE_DISPATCHER_EXPORT);
      d12_wine_state = -1;
      return false;
   }
   d12_wine_dispatcher = (d12_unix_dispatcher_t)*dispatcher_var;

   d12_nt_query_virtual_memory_t query =
      (d12_nt_query_virtual_memory_t)(void *)GetProcAddress(ntdll,
                                                            "NtQueryVirtualMemory");
   if (!query) {
      d12_wine_state = -1;
      return false;
   }
   d12_ntstatus_t status =
      query(GetCurrentProcess(), &__ImageBase, D12_WINE_MEMORY_LOAD_UNIX_LIB,
            &d12_wine_handle, sizeof(d12_wine_handle), NULL);
   if (status) {
      fprintf(stderr,
              "d3d12agx: cannot load unixlib d3d12.so (NTSTATUS 0x%08lx); "
              "is d3d12.dll installed as a Wine builtin next to "
              "aarch64-unix/d3d12.so?\n",
              (unsigned long)status);
      d12_wine_state = -1;
      return false;
   }
   d12_wine_state = 1;
   return true;
}

static int
d12_wine_call(enum d12_wine_call code, void *args)
{
   if (!d12_wine_unixlib_init()) {
      errno = ENOSYS;
      return -1;
   }
   d12_ntstatus_t status = d12_wine_unix_call_raw(d12_wine_handle, code, args);
   if (status) {
      errno = ENOSYS;
      return -1;
   }
   return 0;
}

int
d12_wine_open_render_node(void)
{
   struct d12_wine_open_args a = {0};
   if (d12_wine_call(D12_WINE_CALL_OPEN, &a))
      return -1;
   if (a.fd < 0) {
      errno = a.err;
      return -1;
   }
   return a.fd;
}

void
d12_wine_close_fd(int fd)
{
   struct d12_wine_close_args a = {.fd = fd};
   d12_wine_call(D12_WINE_CALL_CLOSE, &a);
}

int
d12_wine_dupfd_cloexec(int fd)
{
   struct d12_wine_dupfd_args a = {.fd = fd};
   if (d12_wine_call(D12_WINE_CALL_DUPFD_CLOEXEC, &a))
      return -1;
   if (a.new_fd < 0)
      errno = a.err;
   return a.new_fd;
}

int64_t
d12_wine_dmabuf_size(int fd)
{
   struct d12_wine_dmabuf_size_args a = {.fd = fd};
   if (d12_wine_call(D12_WINE_CALL_DMABUF_SIZE, &a))
      return -1;
   if (a.size < 0)
      errno = a.err;
   return a.size;
}

bool
d12_wine_fd_is_asahi_render_node(int fd)
{
   struct d12_wine_is_render_node_args a = {.fd = fd};
   if (d12_wine_call(D12_WINE_CALL_IS_RENDER_NODE, &a))
      return false;
   return a.result == 1;
}

/* ---- mmap ---------------------------------------------------------------- */

void *
os_mmap(void *addr, size_t length, int prot, int flags, int fd, int64_t offset)
{
   struct d12_wine_mmap_args a = {
      .addr = (uintptr_t)addr,
      .length = length,
      .prot = prot,
      .flags = flags,
      .fd = fd,
      .offset = (uint64_t)offset,
   };
   if (d12_wine_call(D12_WINE_CALL_MMAP, &a))
      return MAP_FAILED;
   if (!a.addr) {
      errno = a.err;
      return MAP_FAILED;
   }
   return (void *)(uintptr_t)a.addr;
}

int
os_munmap(void *addr, size_t length)
{
   struct d12_wine_munmap_args a = {.addr = (uintptr_t)addr, .length = length};
   if (d12_wine_call(D12_WINE_CALL_MUNMAP, &a))
      return -1;
   if (a.ret)
      errno = a.err;
   return a.ret;
}

/* ---- libdrm subset ------------------------------------------------------- */

int
drmIoctl(int fd, unsigned request, void *arg)
{
   struct d12_wine_ioctl_args a = {
      .fd = fd,
      .request = request,
      .arg = (uintptr_t)arg,
   };
   if (d12_wine_call(D12_WINE_CALL_IOCTL, &a))
      return -1;
   if (a.ret)
      errno = a.err;
   return a.ret;
}

/* libdrm convention for the helpers below: 0 on success, else -errno. */
static int
drm_ioctl_neg(int fd, unsigned request, void *arg)
{
   if (drmIoctl(fd, request, arg))
      return -errno;
   return 0;
}

drmVersionPtr
drmGetVersion(int fd)
{
   struct drm_version v = {0};
   if (drmIoctl(fd, DRM_IOCTL_VERSION, &v))
      return NULL;

   drmVersionPtr r = calloc(1, sizeof(*r));
   if (!r)
      return NULL;
   r->name = calloc(1, v.name_len + 1);
   r->date = calloc(1, v.date_len + 1);
   r->desc = calloc(1, v.desc_len + 1);
   if (!r->name || !r->date || !r->desc) {
      drmFreeVersion(r);
      return NULL;
   }
   v.name = r->name;
   v.date = r->date;
   v.desc = r->desc;
   if (drmIoctl(fd, DRM_IOCTL_VERSION, &v)) {
      drmFreeVersion(r);
      return NULL;
   }
   r->version_major = v.version_major;
   r->version_minor = v.version_minor;
   r->version_patchlevel = v.version_patchlevel;
   r->name_len = (int)v.name_len;
   r->date_len = (int)v.date_len;
   r->desc_len = (int)v.desc_len;
   return r;
}

void
drmFreeVersion(drmVersionPtr v)
{
   if (!v)
      return;
   free(v->name);
   free(v->date);
   free(v->desc);
   free(v);
}

int
drmGetCap(int fd, uint64_t capability, uint64_t *value)
{
   struct drm_get_cap cap = {.capability = capability};
   int ret = drm_ioctl_neg(fd, DRM_IOCTL_GET_CAP, &cap);
   if (ret)
      return ret;
   *value = cap.value;
   return 0;
}

int
drmPrimeHandleToFD(int fd, uint32_t handle, uint32_t flags, int *prime_fd)
{
   struct drm_prime_handle args = {.handle = handle, .flags = flags};
   int ret = drm_ioctl_neg(fd, DRM_IOCTL_PRIME_HANDLE_TO_FD, &args);
   if (ret)
      return ret;
   *prime_fd = args.fd;
   return 0;
}

int
drmPrimeFDToHandle(int fd, int prime_fd, uint32_t *handle)
{
   struct drm_prime_handle args = {.fd = prime_fd};
   int ret = drm_ioctl_neg(fd, DRM_IOCTL_PRIME_FD_TO_HANDLE, &args);
   if (ret)
      return ret;
   *handle = args.handle;
   return 0;
}

int
drmSyncobjCreate(int fd, uint32_t flags, uint32_t *handle)
{
   struct drm_syncobj_create args = {.flags = flags};
   int ret = drm_ioctl_neg(fd, DRM_IOCTL_SYNCOBJ_CREATE, &args);
   if (ret)
      return ret;
   *handle = args.handle;
   return 0;
}

int
drmSyncobjDestroy(int fd, uint32_t handle)
{
   struct drm_syncobj_destroy args = {.handle = handle};
   return drm_ioctl_neg(fd, DRM_IOCTL_SYNCOBJ_DESTROY, &args);
}

int
drmSyncobjHandleToFD(int fd, uint32_t handle, int *obj_fd)
{
   struct drm_syncobj_handle args = {.handle = handle};
   int ret = drm_ioctl_neg(fd, DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD, &args);
   if (ret)
      return ret;
   *obj_fd = args.fd;
   return 0;
}

int
drmSyncobjFDToHandle(int fd, int obj_fd, uint32_t *handle)
{
   struct drm_syncobj_handle args = {.fd = obj_fd};
   int ret = drm_ioctl_neg(fd, DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE, &args);
   if (ret)
      return ret;
   *handle = args.handle;
   return 0;
}

int
drmSyncobjImportSyncFile(int fd, uint32_t handle, int sync_file_fd)
{
   struct drm_syncobj_handle args = {
      .handle = handle,
      .flags = DRM_SYNCOBJ_FD_TO_HANDLE_FLAGS_IMPORT_SYNC_FILE,
      .fd = sync_file_fd,
   };
   return drm_ioctl_neg(fd, DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE, &args);
}

int
drmSyncobjExportSyncFile(int fd, uint32_t handle, int *sync_file_fd)
{
   struct drm_syncobj_handle args = {
      .handle = handle,
      .flags = DRM_SYNCOBJ_HANDLE_TO_FD_FLAGS_EXPORT_SYNC_FILE,
   };
   int ret = drm_ioctl_neg(fd, DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD, &args);
   if (ret)
      return ret;
   *sync_file_fd = args.fd;
   return 0;
}

/* Callers pass absolute CLOCK_MONOTONIC-style deadlines derived from
 * os_time_get_nano(); rebase to a relative wait so the unix side can apply its
 * own monotonic clock. INT64_MAX means forever. */
static int64_t
d12_wine_relative_timeout(int64_t timeout_nsec)
{
   if (timeout_nsec == INT64_MAX)
      return -1;
   int64_t now = os_time_get_nano();
   return timeout_nsec > now ? timeout_nsec - now : 0;
}

static int
d12_wine_syncobj_wait(int fd, uint32_t *handles, uint64_t *points,
                      unsigned num_handles, int64_t timeout_nsec,
                      unsigned flags, uint32_t *first_signaled)
{
   struct d12_wine_syncobj_wait_args a = {
      .fd = fd,
      .count = num_handles,
      .handles = (uintptr_t)handles,
      .points = (uintptr_t)points,
      .rel_timeout_ns = d12_wine_relative_timeout(timeout_nsec),
      .flags = flags,
   };
   if (d12_wine_call(D12_WINE_CALL_SYNCOBJ_WAIT, &a))
      return -ENOSYS;
   if (a.ret)
      return a.ret;
   if (first_signaled)
      *first_signaled = a.first_signaled;
   return 0;
}

int
drmSyncobjWait(int fd, uint32_t *handles, unsigned num_handles,
               int64_t timeout_nsec, unsigned flags, uint32_t *first_signaled)
{
   return d12_wine_syncobj_wait(fd, handles, NULL, num_handles, timeout_nsec,
                                flags, first_signaled);
}

int
drmSyncobjTimelineWait(int fd, uint32_t *handles, uint64_t *points,
                       unsigned num_handles, int64_t timeout_nsec,
                       unsigned flags, uint32_t *first_signaled)
{
   return d12_wine_syncobj_wait(fd, handles, points, num_handles, timeout_nsec,
                                flags, first_signaled);
}

int
drmSyncobjReset(int fd, const uint32_t *handles, uint32_t handle_count)
{
   struct drm_syncobj_array args = {
      .handles = (uintptr_t)handles,
      .count_handles = handle_count,
   };
   return drm_ioctl_neg(fd, DRM_IOCTL_SYNCOBJ_RESET, &args);
}

int
drmSyncobjSignal(int fd, const uint32_t *handles, uint32_t handle_count)
{
   struct drm_syncobj_array args = {
      .handles = (uintptr_t)handles,
      .count_handles = handle_count,
   };
   return drm_ioctl_neg(fd, DRM_IOCTL_SYNCOBJ_SIGNAL, &args);
}

int
drmSyncobjTimelineSignal(int fd, const uint32_t *handles, uint64_t *points,
                         uint32_t handle_count)
{
   struct drm_syncobj_timeline_array args = {
      .handles = (uintptr_t)handles,
      .points = (uintptr_t)points,
      .count_handles = handle_count,
   };
   return drm_ioctl_neg(fd, DRM_IOCTL_SYNCOBJ_TIMELINE_SIGNAL, &args);
}

int
drmSyncobjQuery2(int fd, uint32_t *handles, uint64_t *points,
                 uint32_t handle_count, uint32_t flags)
{
   struct drm_syncobj_timeline_array args = {
      .handles = (uintptr_t)handles,
      .points = (uintptr_t)points,
      .count_handles = handle_count,
      .flags = flags,
   };
   return drm_ioctl_neg(fd, DRM_IOCTL_SYNCOBJ_QUERY, &args);
}

int
drmSyncobjTransfer(int fd, uint32_t dst_handle, uint64_t dst_point,
                   uint32_t src_handle, uint64_t src_point, uint32_t flags)
{
   struct drm_syncobj_transfer args = {
      .src_handle = src_handle,
      .dst_handle = dst_handle,
      .src_point = src_point,
      .dst_point = dst_point,
      .flags = flags,
   };
   return drm_ioctl_neg(fd, DRM_IOCTL_SYNCOBJ_TRANSFER, &args);
}
