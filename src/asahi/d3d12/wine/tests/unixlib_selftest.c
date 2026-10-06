/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Native (no Wine) self-test of the d3d12agx unixlib: dlopen d3d12.so, pull
 * __wine_unix_call_funcs and drive the call table exactly as the PE side
 * would. Covers render-node discovery, DRM_IOCTL_VERSION through the IOCTL
 * call, GEM create + mmap/munmap through the unixlib, syncobj waits (signaled,
 * timeout) with the relative-timeout re-basing, and the clock call.
 *
 * Build:  cc -O2 -o unixlib_selftest unixlib_selftest.c -ldl
 * Run:    ./unixlib_selftest path/to/d3d12.so     (needs an asahi render node)
 */
#include <dlfcn.h>
#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>

#include "../d12_wine_unixlib.h"
#include "drm-uapi/drm.h"
#include "drm-uapi/asahi_drm.h"

typedef uint32_t (*entry_t)(void *args);
static entry_t *funcs;
static int failures;

#define CHECK(cond, ...)                                                      \
   do {                                                                       \
      if (cond) {                                                             \
         printf("PASS " __VA_ARGS__);                                         \
         printf("\n");                                                        \
      } else {                                                                \
         printf("FAIL " __VA_ARGS__);                                         \
         printf("\n");                                                        \
         failures++;                                                          \
      }                                                                       \
   } while (0)

static int
do_ioctl(int fd, uint32_t req, void *arg, int *err)
{
   struct d12_wine_ioctl_args a = {.fd = fd, .request = req, .arg = (uintptr_t)arg};
   funcs[D12_WINE_CALL_IOCTL](&a);
   if (err)
      *err = a.err;
   return a.ret;
}

int
main(int argc, char **argv)
{
   const char *so = argc > 1 ? argv[1] : "./d3d12.so";
   void *h = dlopen(so, RTLD_NOW | RTLD_LOCAL);
   if (!h) {
      fprintf(stderr, "dlopen %s: %s\n", so, dlerror());
      return 2;
   }
   funcs = dlsym(h, "__wine_unix_call_funcs");
   CHECK(funcs != NULL, "__wine_unix_call_funcs exported");
   if (!funcs)
      return 2;
   for (int i = 0; i < D12_WINE_CALL_COUNT; i++)
      CHECK(funcs[i] != NULL, "call %d populated", i);

   struct d12_wine_open_args o = {0};
   funcs[D12_WINE_CALL_OPEN](&o);
   CHECK(o.fd >= 0 && !strcmp(o.name, "asahi"), "OPEN -> fd %d name '%s' err %d", o.fd, o.name, o.err);
   if (o.fd < 0)
      return failures ? 1 : 0;
   int fd = o.fd;

   struct d12_wine_is_render_node_args rn = {.fd = fd};
   funcs[D12_WINE_CALL_IS_RENDER_NODE](&rn);
   CHECK(rn.result == 1, "IS_RENDER_NODE(fd) -> %d", rn.result);
   struct d12_wine_is_render_node_args rn0 = {.fd = 0};
   funcs[D12_WINE_CALL_IS_RENDER_NODE](&rn0);
   CHECK(rn0.result == 0, "IS_RENDER_NODE(stdin) -> %d", rn0.result);

   /* Request numbers must be the Linux encoding (the PE side computes them
    * with the sys/ioccom.h shim; here the real header). */
   char name[32] = {0};
   struct drm_version v = {.name = name, .name_len = sizeof(name) - 1};
   int err;
   int r = do_ioctl(fd, DRM_IOCTL_VERSION, &v, &err);
   CHECK(r == 0 && !strcmp(name, "asahi"), "IOCTL DRM_IOCTL_VERSION (0x%08x) -> %d '%s' %d.%d.%d",
         (unsigned)DRM_IOCTL_VERSION, r, name, v.version_major, v.version_minor, v.version_patchlevel);

   struct drm_asahi_params_global params = {0};
   struct drm_asahi_get_params gp = {.param_group = 0, .pointer = (uintptr_t)&params, .size = sizeof(params)};
   r = do_ioctl(fd, DRM_IOCTL_ASAHI_GET_PARAMS, &gp, &err);
   CHECK(r == 0, "IOCTL ASAHI_GET_PARAMS -> %d err %d (gpu gen %u variant %u rev %u)", r, err,
         params.gpu_generation, params.gpu_variant, params.gpu_revision);

   /* GEM create + mmap through the unixlib, written and read in this process. */
   struct drm_asahi_gem_create gc = {.size = 1 << 20, .flags = 0, .vm_id = 0};
   r = do_ioctl(fd, DRM_IOCTL_ASAHI_GEM_CREATE, &gc, &err);
   CHECK(r == 0 && gc.handle, "IOCTL ASAHI_GEM_CREATE 1 MiB -> %d err %d handle %u", r, err, gc.handle);
   if (r == 0) {
      struct drm_asahi_gem_mmap_offset mo = {.handle = gc.handle};
      r = do_ioctl(fd, DRM_IOCTL_ASAHI_GEM_MMAP_OFFSET, &mo, &err);
      CHECK(r == 0, "IOCTL ASAHI_GEM_MMAP_OFFSET -> %d err %d offset 0x%" PRIx64, r, err, (uint64_t)mo.offset);
      struct d12_wine_mmap_args m = {.addr = 0, .length = gc.size,
                                     .prot = D12_WINE_PROT_READ | D12_WINE_PROT_WRITE,
                                     .flags = D12_WINE_MAP_SHARED, .fd = fd, .offset = mo.offset};
      funcs[D12_WINE_CALL_MMAP](&m);
      CHECK(m.addr != 0 && m.err == 0, "MMAP BO -> 0x%" PRIx64 " err %d", (uint64_t)m.addr, m.err);
      if (m.addr) {
         volatile uint32_t *p = (volatile uint32_t *)(uintptr_t)m.addr;
         p[0] = 0xA18A5A5A;
         p[(gc.size / 4) - 1] = 0x5A5AA18A;
         CHECK(p[0] == 0xA18A5A5A && p[(gc.size / 4) - 1] == 0x5A5AA18A, "BO mapping read/write in-process");
         struct d12_wine_munmap_args um = {.addr = m.addr, .length = gc.size};
         funcs[D12_WINE_CALL_MUNMAP](&um);
         CHECK(um.ret == 0, "MUNMAP -> %d err %d", um.ret, um.err);
      }
      struct drm_gem_close gcl = {.handle = gc.handle};
      r = do_ioctl(fd, DRM_IOCTL_GEM_CLOSE, &gcl, &err);
      CHECK(r == 0, "IOCTL GEM_CLOSE -> %d err %d", r, err);
   }

   /* Syncobj waits: a signaled binary syncobj returns at once; an unsignaled
    * one times out with -ETIME after the relative timeout. */
   struct drm_syncobj_create sc = {.flags = DRM_SYNCOBJ_CREATE_SIGNALED};
   r = do_ioctl(fd, DRM_IOCTL_SYNCOBJ_CREATE, &sc, &err);
   CHECK(r == 0, "IOCTL SYNCOBJ_CREATE(signaled) -> %d handle %u", r, sc.handle);
   struct drm_syncobj_create su = {.flags = 0};
   r = do_ioctl(fd, DRM_IOCTL_SYNCOBJ_CREATE, &su, &err);
   CHECK(r == 0, "IOCTL SYNCOBJ_CREATE(unsignaled) -> %d handle %u", r, su.handle);

   struct d12_wine_clock_args c0 = {0}, c1 = {0};
   funcs[D12_WINE_CALL_CLOCK](&c0);
   uint32_t hs = sc.handle;
   struct d12_wine_syncobj_wait_args w = {.fd = fd, .count = 1, .handles = (uintptr_t)&hs,
                                          .rel_timeout_ns = 1000000000ll, .flags = 0};
   funcs[D12_WINE_CALL_SYNCOBJ_WAIT](&w);
   CHECK(w.ret == 0, "SYNCOBJ_WAIT signaled -> %d first %u", w.ret, w.first_signaled);

   uint32_t hu = su.handle;
   struct d12_wine_syncobj_wait_args wu = {.fd = fd, .count = 1, .handles = (uintptr_t)&hu,
                                           .rel_timeout_ns = 50000000ll,
                                           .flags = DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT};
   funcs[D12_WINE_CALL_SYNCOBJ_WAIT](&wu);
   funcs[D12_WINE_CALL_CLOCK](&c1);
   int64_t waited_ms = (c1.monotonic_ns - c0.monotonic_ns) / 1000000;
   CHECK(wu.ret == -ETIME && waited_ms >= 45 && waited_ms < 2000,
         "SYNCOBJ_WAIT unsignaled 50 ms -> %d (ETIME=%d) after %" PRId64 " ms", wu.ret, -ETIME, waited_ms);

   uint32_t hz = su.handle;
   struct d12_wine_syncobj_wait_args wz = {.fd = fd, .count = 1, .handles = (uintptr_t)&hz,
                                           .rel_timeout_ns = 0, .flags = DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT};
   funcs[D12_WINE_CALL_SYNCOBJ_WAIT](&wz);
   CHECK(wz.ret == -ETIME, "SYNCOBJ_WAIT unsignaled poll (0 ns) -> %d", wz.ret);

   /* Timeline wait on point 0 of a signaled syncobj. */
   uint64_t pt = 0;
   struct d12_wine_syncobj_wait_args wt = {.fd = fd, .count = 1, .handles = (uintptr_t)&hs,
                                           .points = (uintptr_t)&pt, .rel_timeout_ns = 100000000ll, .flags = 0};
   funcs[D12_WINE_CALL_SYNCOBJ_WAIT](&wt);
   CHECK(wt.ret == 0, "SYNCOBJ_TIMELINE_WAIT point 0 of signaled -> %d", wt.ret);

   struct drm_syncobj_destroy sd = {.handle = sc.handle};
   do_ioctl(fd, DRM_IOCTL_SYNCOBJ_DESTROY, &sd, &err);
   sd.handle = su.handle;
   do_ioctl(fd, DRM_IOCTL_SYNCOBJ_DESTROY, &sd, &err);

   struct d12_wine_close_args cl = {.fd = fd};
   funcs[D12_WINE_CALL_CLOSE](&cl);
   printf("%s: %d failure(s)\n", failures ? "FAILURES" : "ALL PASS", failures);
   return failures ? 1 : 0;
}
