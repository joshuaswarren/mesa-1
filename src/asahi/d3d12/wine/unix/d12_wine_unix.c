/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Wine unixlib for d3d12agx: the only Linux-side code of the driver. It opens
 * the asahi render node and forwards ioctl/mmap/munmap/syncobj waits for the
 * ARM64EC PE d3d12.dll running in the same process. No libdrm: raw UAPI only.
 *
 * Build (aarch64 Linux):
 *   cc -O2 -fPIC -shared -fvisibility=hidden -o d3d12.so d12_wine_unix.c
 */

#define _GNU_SOURCE
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <time.h>
#include <unistd.h>

#include "../d12_wine_unixlib.h"

/* drm.h subset: version query and syncobj waits (public Linux UAPI). */
struct drm_version {
   int version_major;
   int version_minor;
   int version_patchlevel;
   size_t name_len;
   char *name;
   size_t date_len;
   char *date;
   size_t desc_len;
   char *desc;
};
#define DRM_IOCTL_VERSION _IOWR('d', 0x00, struct drm_version)

struct drm_syncobj_wait {
   uint64_t handles;
   int64_t timeout_nsec;
   uint32_t count_handles;
   uint32_t flags;
   uint32_t first_signaled;
   uint32_t pad;
};
struct drm_syncobj_timeline_wait {
   uint64_t handles;
   uint64_t points;
   int64_t timeout_nsec;
   uint32_t count_handles;
   uint32_t flags;
   uint32_t first_signaled;
   uint32_t pad;
};
#define DRM_IOCTL_SYNCOBJ_WAIT          _IOWR('d', 0xC3, struct drm_syncobj_wait)
#define DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT _IOWR('d', 0xCA, struct drm_syncobj_timeline_wait)

typedef uint32_t NTSTATUS;
#define STATUS_SUCCESS 0

static int
retry_ioctl(int fd, unsigned long req, void *arg)
{
   int ret;
   do {
      ret = ioctl(fd, req, arg);
   } while (ret == -1 && (errno == EINTR || errno == EAGAIN));
   return ret;
}

static int
drm_name(int fd, char *name, size_t len)
{
   char buf[64] = {0};
   struct drm_version v = {.name = buf, .name_len = sizeof(buf) - 1};
   if (retry_ioctl(fd, DRM_IOCTL_VERSION, &v))
      return -1;
   snprintf(name, len, "%s", buf);
   return 0;
}

static NTSTATUS
call_open(void *args)
{
   struct d12_wine_open_args *a = args;
   a->fd = -1;
   a->err = ENOENT;
   a->name[0] = 0;

   /* Honour an explicit node (tests) before scanning. */
   const char *forced = getenv("D3D12AGX_RENDER_NODE");
   for (int i = 128; i < 192; i++) {
      char path[64];
      if (forced && i == 128)
         snprintf(path, sizeof(path), "%s", forced);
      else if (forced)
         break;
      else
         snprintf(path, sizeof(path), "/dev/dri/renderD%d", i);
      int fd = open(path, O_RDWR | O_CLOEXEC);
      if (fd < 0) {
         if (errno != ENOENT)
            a->err = errno;
         continue;
      }
      char name[32];
      if (!drm_name(fd, name, sizeof(name)) && !strcmp(name, "asahi")) {
         a->fd = fd;
         a->err = 0;
         snprintf(a->name, sizeof(a->name), "%s", name);
         return STATUS_SUCCESS;
      }
      close(fd);
   }
   return STATUS_SUCCESS;
}

static NTSTATUS
call_close(void *args)
{
   struct d12_wine_close_args *a = args;
   close(a->fd);
   return STATUS_SUCCESS;
}

static NTSTATUS
call_ioctl(void *args)
{
   struct d12_wine_ioctl_args *a = args;
   a->ret = retry_ioctl(a->fd, a->request, (void *)(uintptr_t)a->arg);
   a->err = a->ret ? errno : 0;
   return STATUS_SUCCESS;
}

static NTSTATUS
call_mmap(void *args)
{
   struct d12_wine_mmap_args *a = args;
   int prot = ((a->prot & D12_WINE_PROT_READ) ? PROT_READ : 0) |
              ((a->prot & D12_WINE_PROT_WRITE) ? PROT_WRITE : 0);
   int flags = ((a->flags & D12_WINE_MAP_SHARED) ? MAP_SHARED : MAP_PRIVATE) |
               ((a->flags & D12_WINE_MAP_FIXED) ? MAP_FIXED : 0);
   void *p = mmap((void *)(uintptr_t)a->addr, a->length, prot, flags, a->fd,
                  (off_t)a->offset);
   if (p == MAP_FAILED) {
      a->err = errno;
      a->addr = 0;
   } else {
      a->err = 0;
      a->addr = (uintptr_t)p;
   }
   return STATUS_SUCCESS;
}

static NTSTATUS
call_munmap(void *args)
{
   struct d12_wine_munmap_args *a = args;
   a->ret = munmap((void *)(uintptr_t)a->addr, a->length);
   a->err = a->ret ? errno : 0;
   return STATUS_SUCCESS;
}

static int64_t
monotonic_ns(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return (int64_t)ts.tv_sec * 1000000000ll + ts.tv_nsec;
}

static NTSTATUS
call_syncobj_wait(void *args)
{
   struct d12_wine_syncobj_wait_args *a = args;
   int64_t abs_ns;
   if (a->rel_timeout_ns < 0)
      abs_ns = INT64_MAX;
   else {
      int64_t now = monotonic_ns();
      abs_ns = (a->rel_timeout_ns > INT64_MAX - now) ? INT64_MAX
                                                      : now + a->rel_timeout_ns;
   }
   int ret;
   if (a->points) {
      struct drm_syncobj_timeline_wait w = {
         .handles = a->handles,
         .points = a->points,
         .timeout_nsec = abs_ns,
         .count_handles = a->count,
         .flags = a->flags,
      };
      ret = retry_ioctl(a->fd, DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, &w);
      a->first_signaled = w.first_signaled;
   } else {
      struct drm_syncobj_wait w = {
         .handles = a->handles,
         .timeout_nsec = abs_ns,
         .count_handles = a->count,
         .flags = a->flags,
      };
      ret = retry_ioctl(a->fd, DRM_IOCTL_SYNCOBJ_WAIT, &w);
      a->first_signaled = w.first_signaled;
   }
   a->ret = ret ? -errno : 0;
   return STATUS_SUCCESS;
}

static NTSTATUS
call_is_render_node(void *args)
{
   struct d12_wine_is_render_node_args *a = args;
   struct stat st;
   char name[32];
   a->result = 0;
   if (a->fd < 0 || fstat(a->fd, &st) || !S_ISCHR(st.st_mode))
      return STATUS_SUCCESS;
   /* DRM render nodes are minor 128..191 of the DRM major. */
   if (minor(st.st_rdev) < 128)
      return STATUS_SUCCESS;
   if (drm_name(a->fd, name, sizeof(name)) || strcmp(name, "asahi"))
      return STATUS_SUCCESS;
   a->result = 1;
   return STATUS_SUCCESS;
}

static NTSTATUS
call_clock(void *args)
{
   struct d12_wine_clock_args *a = args;
   a->monotonic_ns = monotonic_ns();
   return STATUS_SUCCESS;
}

static NTSTATUS
call_dupfd_cloexec(void *args)
{
   struct d12_wine_dupfd_args *a = args;
   a->new_fd = fcntl(a->fd, F_DUPFD_CLOEXEC, 3);
   a->err = a->new_fd < 0 ? errno : 0;
   return STATUS_SUCCESS;
}

static NTSTATUS
call_dmabuf_size(void *args)
{
   struct d12_wine_dmabuf_size_args *a = args;
   a->size = lseek(a->fd, 0, SEEK_END);
   a->err = a->size < 0 ? errno : 0;
   return STATUS_SUCCESS;
}

typedef NTSTATUS (*unixlib_entry_t)(void *args);

__attribute__((visibility("default")))
const unixlib_entry_t __wine_unix_call_funcs[] = {
   [D12_WINE_CALL_OPEN] = call_open,
   [D12_WINE_CALL_CLOSE] = call_close,
   [D12_WINE_CALL_IOCTL] = call_ioctl,
   [D12_WINE_CALL_MMAP] = call_mmap,
   [D12_WINE_CALL_MUNMAP] = call_munmap,
   [D12_WINE_CALL_SYNCOBJ_WAIT] = call_syncobj_wait,
   [D12_WINE_CALL_IS_RENDER_NODE] = call_is_render_node,
   [D12_WINE_CALL_CLOCK] = call_clock,
   [D12_WINE_CALL_DUPFD_CLOEXEC] = call_dupfd_cloexec,
   [D12_WINE_CALL_DMABUF_SIZE] = call_dmabuf_size,
};

_Static_assert(sizeof(__wine_unix_call_funcs) / sizeof(__wine_unix_call_funcs[0]) ==
               D12_WINE_CALL_COUNT, "call table");
