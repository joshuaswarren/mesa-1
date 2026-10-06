/* SPDX-License-Identifier: MIT */
/* Diagnostic LD_PRELOAD shim; compile against Mesa's drm-uapi headers.
 * cc -shared -fPIC -O2 -Wall -Wextra -Iinclude/drm-uapi ioctl-timing.c -ldl
 * Records large GEM_CREATE / VM_BIND calls without changing the request. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include "asahi_drm.h"

static int (*real_ioctl)(int, unsigned long, void *);
static pthread_once_t once = PTHREAD_ONCE_INIT;

static void resolve(void)
{
   real_ioctl = dlsym(RTLD_NEXT, "drmIoctl");
   /* Vulkan loads its ICD (and libdrm) with local symbol visibility. That
    * dependency may therefore be absent from the preload's RTLD_NEXT scope. */
   if (!real_ioctl) {
      void *drm = dlopen("libdrm.so.2", RTLD_LAZY | RTLD_LOCAL);
      if (drm)
         real_ioctl = dlsym(drm, "drmIoctl");
   }
}

static double now(clockid_t id)
{
   struct timespec ts;
   clock_gettime(id, &ts);
   return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int drmIoctl(int fd, unsigned long request, void *arg)
{
   pthread_once(&once, resolve);
   if (!real_ioctl) { errno = ENOSYS; return -1; }
   uint64_t bytes = 0, addr = 0;
   unsigned flags = 0, handle = 0;
   const char *kind = NULL;
   if (request == DRM_IOCTL_ASAHI_GEM_CREATE) {
      struct drm_asahi_gem_create *c = arg;
      bytes = c->size; flags = c->flags;
      kind = "GEM_CREATE";
   } else if (request == DRM_IOCTL_ASAHI_VM_BIND) {
      struct drm_asahi_vm_bind *b = arg;
      if (b->num_binds == 1) {
         struct drm_asahi_gem_bind_op *op = (void *)(uintptr_t)b->userptr;
         bytes = op->range; addr = op->addr;
         flags = op->flags; handle = op->handle;
         kind = "VM_BIND";
      }
   }
   if (bytes < 268435456)
      return real_ioctl(fd, request, arg);
   double wall = now(CLOCK_MONOTONIC), cpu = now(CLOCK_THREAD_CPUTIME_ID);
   int rc = real_ioctl(fd, request, arg);
   int saved = errno;
   cpu = now(CLOCK_THREAD_CPUTIME_ID) - cpu;
   wall = now(CLOCK_MONOTONIC) - wall;
   if (request == DRM_IOCTL_ASAHI_GEM_CREATE)
      handle = ((struct drm_asahi_gem_create *)arg)->handle;
   fprintf(stderr, "bo-ioctl,%s,%llu,%#llx,%u,%#x,%d,%.9f,%.9f\n",
      kind, (unsigned long long)bytes, (unsigned long long)addr,
      handle, flags, rc, wall, cpu);
   errno = saved;
   return rc;
}
