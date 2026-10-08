/*
 * Copyright © 2018 Broadcom
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

/** @file
 *
 * Implements core GEM support (particularly ioctls) underneath the libc ioctl
 * wrappers, and calls into the driver-specific code as necessary.
 */

#include <c11/threads.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/dma-buf.h>
#include <linux/memfd.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/eventfd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include "drm-uapi/drm.h"
#include "drm_shim.h"
#include "util/hash_table.h"
#include "util/u_atomic.h"

#define SHIM_MEM_SIZE (4ull * 1024 * 1024 * 1024)

#ifndef HAVE_MEMFD_CREATE
#include <sys/syscall.h>

static inline int
memfd_create(const char *name, unsigned int flags)
{
   return syscall(SYS_memfd_create, name, flags);
}
#endif

/* Global state for the shim shared between libc, core, and driver. */
struct shim_device shim_device;

long shim_page_size;
static bool failed_dmabuf_export_once;
static bool failed_dmabuf_import_once;
static bool timed_out_sync_wait_once;

#define TIMELINE_HOST_MAX_SYNCOBJS 4096
static struct {
   bool alive;
   uint64_t value;
} timeline_host_syncobjs[TIMELINE_HOST_MAX_SYNCOBJS];
static uint32_t timeline_host_next_handle = 1;

static bool
timeline_host_values_enabled(void)
{
   return getenv("ASAHI_SHIM_TIMELINE_HOST_VALUES") != NULL;
}

static bool
timeline_host_handle_valid(uint32_t handle)
{
   return handle > 0 && handle < TIMELINE_HOST_MAX_SYNCOBJS &&
          timeline_host_syncobjs[handle].alive;
}

static void __attribute__((destructor))
check_injected_sync_failures(void)
{
   if (failed_dmabuf_export_once && failed_dmabuf_import_once)
      fprintf(stderr, "shim: injected dma-buf sync failures exactly once\n");
   if (timed_out_sync_wait_once)
      fprintf(stderr, "shim: injected syncobj wait timeout exactly once\n");
}

static uint32_t
uint_key_hash(const void *key)
{
   return (uintptr_t)key;
}

static bool
uint_key_compare(const void *a, const void *b)
{
   return a == b;
}

/**
 * Called when the first libc shim is called, to initialize GEM simulation
 * state (other than the shims themselves).
 */
void
drm_shim_device_init(void)
{
   shim_device.fd_map = _mesa_hash_table_create(NULL,
                                                uint_key_hash,
                                                uint_key_compare);
   shim_device.prime_fd_map = _mesa_hash_table_create(NULL,
                                                      uint_key_hash,
                                                      uint_key_compare);
   mtx_init(&shim_device.prime_fd_lock, mtx_plain);

   shim_device.offset_map = _mesa_hash_table_u64_create(NULL);

   mtx_init(&shim_device.mem_lock, mtx_plain);

   shim_device.mem_fd = memfd_create("shim mem", MFD_CLOEXEC);
   assert(shim_device.mem_fd != -1);

   ASSERTED int ret = ftruncate(shim_device.mem_fd, SHIM_MEM_SIZE);
   assert(ret == 0);

   /* The man page for mmap() says
    *
    *    offset must be a multiple of the page size as returned by
    *    sysconf(_SC_PAGE_SIZE).
    *
    * Depending on the configuration of the kernel, this may not be 4096. Get
    * this page size once and use it as the page size throughout, ensuring that
    * are offsets are page-size aligned as required. Otherwise, mmap will fail
    * with EINVAL.
    */

   shim_page_size = sysconf(_SC_PAGESIZE);

   util_vma_heap_init(&shim_device.mem_heap, shim_page_size,
                      SHIM_MEM_SIZE - shim_page_size);

   drm_shim_driver_init();
}

static struct shim_fd *
drm_shim_file_create(int fd)
{
   struct shim_fd *shim_fd = calloc(1, sizeof(*shim_fd));

   shim_fd->fd = fd;
   p_atomic_set(&shim_fd->refcount, 1);
   mtx_init(&shim_fd->handle_lock, mtx_plain);
   shim_fd->handles = _mesa_hash_table_create(NULL,
                                              uint_key_hash,
                                              uint_key_compare);

   return shim_fd;
}

/**
 * Called when the libc shims have interposed an open or dup of our simulated
 * DRM device.
 */
void drm_shim_fd_register(int fd, struct shim_fd *shim_fd)
{
   if (!shim_fd)
      shim_fd = drm_shim_file_create(fd);
   else
      p_atomic_inc(&shim_fd->refcount);

   _mesa_hash_table_insert(shim_device.fd_map, (void *)(uintptr_t)(fd + 1), shim_fd);
}

static void handle_delete_fxn(struct hash_entry *entry)
{
   drm_shim_bo_put(entry->data);
}

void drm_shim_fd_unregister(int fd)
{
   if (fd == -1)
      return;

   struct hash_entry *entry =
         _mesa_hash_table_search(shim_device.fd_map, (void *)(uintptr_t)(fd + 1));
   if (!entry)
      return;
   struct shim_fd *shim_fd = entry->data;
   _mesa_hash_table_remove(shim_device.fd_map, entry);

   if (!p_atomic_dec_zero(&shim_fd->refcount))
      return;

   _mesa_hash_table_destroy(shim_fd->handles, handle_delete_fxn);
   free(shim_fd);
}

struct shim_fd *
drm_shim_fd_lookup(int fd)
{
   if (fd == -1)
      return NULL;

   struct hash_entry *entry =
      _mesa_hash_table_search(shim_device.fd_map, (void *)(uintptr_t)(fd + 1));

   if (!entry)
      return NULL;
   return entry->data;
}

void
drm_shim_prime_fd_register(int fd, struct shim_bo *bo)
{
   if (fd < 0 || !bo)
      return;

   mtx_lock(&shim_device.prime_fd_lock);
   drm_shim_bo_get(bo);
   _mesa_hash_table_insert(shim_device.prime_fd_map,
                           (void *)(uintptr_t)(fd + 1), bo);
   mtx_unlock(&shim_device.prime_fd_lock);
}

struct shim_bo *
drm_shim_prime_fd_lookup(int fd)
{
   if (fd < 0)
      return NULL;

   mtx_lock(&shim_device.prime_fd_lock);
   struct hash_entry *entry = _mesa_hash_table_search(
      shim_device.prime_fd_map, (void *)(uintptr_t)(fd + 1));
   struct shim_bo *bo = entry ? entry->data : NULL;
   if (bo)
      drm_shim_bo_get(bo);
   mtx_unlock(&shim_device.prime_fd_lock);
   return bo;
}

void
drm_shim_prime_fd_unregister(int fd)
{
   if (fd < 0)
      return;

   mtx_lock(&shim_device.prime_fd_lock);
   struct hash_entry *entry = _mesa_hash_table_search(
      shim_device.prime_fd_map, (void *)(uintptr_t)(fd + 1));
   struct shim_bo *bo = entry ? entry->data : NULL;
   if (entry)
      _mesa_hash_table_remove(shim_device.prime_fd_map, entry);
   mtx_unlock(&shim_device.prime_fd_lock);

   if (bo)
      drm_shim_bo_put(bo);
}

int
drm_shim_prime_ioctl(int fd, unsigned long request, void *arg)
{
   struct shim_bo *bo = drm_shim_prime_fd_lookup(fd);
   if (!bo)
      return -EBADF;
   drm_shim_bo_put(bo);

   switch (request) {
   case DMA_BUF_IOCTL_SYNC: {
      const struct dma_buf_sync *sync = arg;
      if (!sync)
         return -EINVAL;
      const uint64_t access = sync->flags & DMA_BUF_SYNC_RW;
      const uint64_t phase = sync->flags & (DMA_BUF_SYNC_START |
                                             DMA_BUF_SYNC_END);
      const uint64_t known = DMA_BUF_SYNC_RW | DMA_BUF_SYNC_START |
                             DMA_BUF_SYNC_END;
      const bool valid = access && (phase == DMA_BUF_SYNC_START ||
                                    phase == DMA_BUF_SYNC_END) &&
                         !(sync->flags & ~known);
      return valid ? 0 : -EINVAL;
   }
   case DMA_BUF_IOCTL_EXPORT_SYNC_FILE: {
      struct dma_buf_export_sync_file *export = arg;
      if (!export || !(export->flags & DMA_BUF_SYNC_RW) ||
          (export->flags & ~DMA_BUF_SYNC_RW))
         return -EINVAL;
      if (getenv("ASAHI_SHIM_FAIL_DMABUF_EXPORT_ONCE") &&
          !failed_dmabuf_export_once) {
         failed_dmabuf_export_once = true;
         errno = EMFILE;
         return -EMFILE;
      }
      export->fd = eventfd(1, EFD_CLOEXEC | EFD_NONBLOCK);
      return export->fd < 0 ? -errno : 0;
   }
   case DMA_BUF_IOCTL_IMPORT_SYNC_FILE: {
      const struct dma_buf_import_sync_file *import = arg;
      if (!import || import->fd < 0 ||
          !(import->flags & DMA_BUF_SYNC_RW) ||
          (import->flags & ~DMA_BUF_SYNC_RW))
         return -EINVAL;
      if (getenv("ASAHI_SHIM_FAIL_DMABUF_IMPORT_ONCE") &&
          failed_dmabuf_export_once && !failed_dmabuf_import_once) {
         failed_dmabuf_import_once = true;
         errno = ENOMEM;
         return -ENOMEM;
      }
      return 0;
   }
   default:
      return -ENOTTY;
   }
}

/* ioctl used by drmGetVersion() */
static int
drm_shim_ioctl_version(int fd, unsigned long request, void *arg)
{
   struct drm_version *args = arg;
   const char *date = "20190320";
   const char *desc = "shim";

   args->version_major = shim_device.version_major;
   args->version_minor = shim_device.version_minor;
   args->version_patchlevel = shim_device.version_patchlevel;

   if (args->name)
      strncpy(args->name, shim_device.driver_name, args->name_len);
   if (args->date)
      strncpy(args->date, date, args->date_len);
   if (args->desc)
      strncpy(args->desc, desc, args->desc_len);
   args->name_len = strlen(shim_device.driver_name);
   args->date_len = strlen(date);
   args->desc_len = strlen(desc);

   return 0;
}

static int
drm_shim_ioctl_get_unique(int fd, unsigned long request, void *arg)
{
   struct drm_unique *gu = arg;

   if (gu->unique && shim_device.unique)
      strncpy(gu->unique, shim_device.unique, gu->unique_len);
   gu->unique_len = shim_device.unique ? strlen(shim_device.unique) : 0;

   return 0;
}

static int
drm_shim_ioctl_get_cap(int fd, unsigned long request, void *arg)
{
   struct drm_get_cap *gc = arg;

   switch (gc->capability) {
   case DRM_CAP_PRIME:
      gc->value = DRM_PRIME_CAP_IMPORT | DRM_PRIME_CAP_EXPORT;
      return 0;
   case DRM_CAP_SYNCOBJ:
   case DRM_CAP_SYNCOBJ_TIMELINE:
   case DRM_CAP_ADDFB2_MODIFIERS:
      gc->value = 1;
      return 0;

   default:
      fprintf(stderr, "DRM_IOCTL_GET_CAP: unhandled 0x%x\n",
              (int)gc->capability);
      return -1;
   }
}

static int
drm_shim_ioctl_gem_close(int fd, unsigned long request, void *arg)
{
   struct shim_fd *shim_fd = drm_shim_fd_lookup(fd);
   struct drm_gem_close *c = arg;

   if (!c->handle)
      return 0;

   mtx_lock(&shim_fd->handle_lock);
   struct hash_entry *entry =
      _mesa_hash_table_search(shim_fd->handles, (void *)(uintptr_t)c->handle);
   if (!entry) {
      mtx_unlock(&shim_fd->handle_lock);
      return -EINVAL;
   }

   struct shim_bo *bo = entry->data;
   _mesa_hash_table_remove(shim_fd->handles, entry);
   drm_shim_bo_put(bo);
   mtx_unlock(&shim_fd->handle_lock);
   return 0;
}

static int
drm_shim_ioctl_syncobj_create(int fd, unsigned long request, void *arg)
{
   struct drm_syncobj_create *create = arg;

   if (timeline_host_values_enabled()) {
      if (timeline_host_next_handle >= TIMELINE_HOST_MAX_SYNCOBJS)
         return -ENOMEM;

      create->handle = timeline_host_next_handle++;
      timeline_host_syncobjs[create->handle].alive = true;
      timeline_host_syncobjs[create->handle].value =
         (create->flags & DRM_SYNCOBJ_CREATE_SIGNALED) ? 1 : 0;
   } else {
      create->handle = 1; /* 0 is invalid */
   }

   return 0;
}

static int
drm_shim_ioctl_syncobj_destroy(int fd, unsigned long request, void *arg)
{
   const struct drm_syncobj_destroy *destroy = arg;

   if (timeline_host_values_enabled()) {
      if (!timeline_host_handle_valid(destroy->handle))
         return -EINVAL;
      timeline_host_syncobjs[destroy->handle].alive = false;
   }

   return 0;
}

static int
drm_shim_ioctl_syncobj_transfer(int fd, unsigned long request, void *arg)
{
   return 0;
}

static int
drm_shim_ioctl_syncobj_fd_to_handle(int fd, unsigned long request, void *arg)
{
   return 0;
}

static int
drm_shim_ioctl_syncobj_timeline_signal(int fd, unsigned long request,
                                       void *arg)
{
   const struct drm_syncobj_timeline_array *signal = arg;

   if (timeline_host_values_enabled()) {
      const uint32_t *handles = (const void *)(uintptr_t)signal->handles;
      const uint64_t *points = (const void *)(uintptr_t)signal->points;
      if (!handles || !points || signal->flags)
         return -EINVAL;

      for (uint32_t i = 0; i < signal->count_handles; ++i) {
         if (!timeline_host_handle_valid(handles[i]) ||
             points[i] < timeline_host_syncobjs[handles[i]].value)
            return -EINVAL;
         timeline_host_syncobjs[handles[i]].value = points[i];
      }
   }

   return 0;
}

static int
drm_shim_ioctl_syncobj_query(int fd, unsigned long request, void *arg)
{
   struct drm_syncobj_timeline_array *query = arg;
   if (!timeline_host_values_enabled())
      return 0;

   const uint32_t *handles = (const void *)(uintptr_t)query->handles;
   uint64_t *points = (void *)(uintptr_t)query->points;
   if (!handles || !points ||
       (query->flags & ~DRM_SYNCOBJ_QUERY_FLAGS_LAST_SUBMITTED))
      return -EINVAL;

   for (uint32_t i = 0; i < query->count_handles; ++i) {
      if (!timeline_host_handle_valid(handles[i]))
         return -EINVAL;
      points[i] = timeline_host_syncobjs[handles[i]].value;
   }

   return 0;
}

static int
drm_shim_ioctl_syncobj_timeline_wait(int fd, unsigned long request, void *arg)
{
   struct drm_syncobj_timeline_wait *wait = arg;
   if (!timeline_host_values_enabled())
      return 0;

   const uint32_t *handles = (const void *)(uintptr_t)wait->handles;
   const uint64_t *points = (const void *)(uintptr_t)wait->points;
   if (!handles || !points || !wait->count_handles)
      return -EINVAL;

   const bool wait_all = wait->flags & DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL;
   bool ready = wait_all;
   for (uint32_t i = 0; i < wait->count_handles; ++i) {
      if (!timeline_host_handle_valid(handles[i]))
         return -EINVAL;

      const bool this_ready =
         timeline_host_syncobjs[handles[i]].value >= points[i];
      if (!wait_all && this_ready) {
         wait->first_signaled = i;
         ready = true;
         break;
      }
      if (wait_all && !this_ready)
         ready = false;
   }

   if (!ready) {
      errno = ETIME;
      return -ETIME;
   }

   return 0;
}

static int
drm_shim_ioctl_syncobj_handle_to_fd(int fd, unsigned long request, void *arg)
{
   struct drm_syncobj_handle *handle = arg;

   /* No-op submissions complete immediately, so their completion sync file
    * must be immediately pollable. Ordinary syncobj-fd export stays closed:
    * an eventfd models a sync file, not another syncobj handle. */
   if (!handle->handle ||
       handle->flags != DRM_SYNCOBJ_HANDLE_TO_FD_FLAGS_EXPORT_SYNC_FILE)
      return -EINVAL;

   handle->fd = eventfd(1, EFD_CLOEXEC | EFD_NONBLOCK);
   return handle->fd < 0 ? -errno : 0;
}

static int
drm_shim_ioctl_syncobj_reset(int fd, unsigned long request, void *arg)
{
   return 0;
}

static int
drm_shim_ioctl_prime_handle_to_fd(int fd, unsigned long request, void *arg)
{
   struct shim_fd *shim_fd = drm_shim_fd_lookup(fd);
   struct drm_prime_handle *prime = arg;
   if (!shim_fd || !prime->handle ||
       (prime->flags & ~(DRM_CLOEXEC | DRM_RDWR)))
      return -EINVAL;

   struct shim_bo *bo = drm_shim_bo_lookup(shim_fd, prime->handle);
   if (!bo)
      return -ENOENT;

   const unsigned flags = (prime->flags & DRM_CLOEXEC) ? MFD_CLOEXEC : 0;
   const int prime_fd = memfd_create("drm-shim-prime", flags);
   if (prime_fd < 0 || ftruncate(prime_fd, bo->size)) {
      const int error = errno;
      if (prime_fd >= 0)
         close(prime_fd);
      drm_shim_bo_put(bo);
      return -error;
   }

   drm_shim_prime_fd_register(prime_fd, bo);
   drm_shim_bo_put(bo);
   prime->fd = prime_fd;
   return 0;
}

static int
drm_shim_ioctl_prime_fd_to_handle(int fd, unsigned long request, void *arg)
{
   struct shim_fd *shim_fd = drm_shim_fd_lookup(fd);
   struct drm_prime_handle *prime = arg;
   if (!shim_fd || prime->fd < 0 || prime->flags)
      return -EINVAL;

   struct shim_bo *bo = drm_shim_prime_fd_lookup(prime->fd);
   if (!bo)
      return -ENOENT;

   prime->handle = drm_shim_bo_get_handle(shim_fd, bo);
   drm_shim_bo_put(bo);
   return prime->handle ? 0 : -ENOMEM;
}

static int
drm_shim_ioctl_syncobj_wait(int fd, unsigned long request, void *arg)
{
   if (getenv("ASAHI_SHIM_TIMEOUT_SYNC_WAIT_ONCE") &&
       !timed_out_sync_wait_once) {
      timed_out_sync_wait_once = true;
      errno = ETIME;
      return -ETIME;
   }

   return 0;
}

ioctl_fn_t core_ioctls[] = {
   [_IOC_NR(DRM_IOCTL_VERSION)] = drm_shim_ioctl_version,
   [_IOC_NR(DRM_IOCTL_GET_UNIQUE)] = drm_shim_ioctl_get_unique,
   [_IOC_NR(DRM_IOCTL_GET_CAP)] = drm_shim_ioctl_get_cap,
   [_IOC_NR(DRM_IOCTL_GEM_CLOSE)] = drm_shim_ioctl_gem_close,
   [_IOC_NR(DRM_IOCTL_SYNCOBJ_CREATE)] = drm_shim_ioctl_syncobj_create,
   [_IOC_NR(DRM_IOCTL_SYNCOBJ_DESTROY)] = drm_shim_ioctl_syncobj_destroy,
   [_IOC_NR(DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD)] =
      drm_shim_ioctl_syncobj_handle_to_fd,
   [_IOC_NR(DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE)] =
      drm_shim_ioctl_syncobj_fd_to_handle,
   [_IOC_NR(DRM_IOCTL_SYNCOBJ_WAIT)] = drm_shim_ioctl_syncobj_wait,
   [_IOC_NR(DRM_IOCTL_SYNCOBJ_TRANSFER)] = drm_shim_ioctl_syncobj_transfer,
   [_IOC_NR(DRM_IOCTL_SYNCOBJ_RESET)] = drm_shim_ioctl_syncobj_reset,
   [_IOC_NR(DRM_IOCTL_SYNCOBJ_TIMELINE_SIGNAL)] =
      drm_shim_ioctl_syncobj_timeline_signal,
   [_IOC_NR(DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT)] =
      drm_shim_ioctl_syncobj_timeline_wait,
   [_IOC_NR(DRM_IOCTL_SYNCOBJ_QUERY)] = drm_shim_ioctl_syncobj_query,
   [_IOC_NR(DRM_IOCTL_PRIME_HANDLE_TO_FD)] = drm_shim_ioctl_prime_handle_to_fd,
   [_IOC_NR(DRM_IOCTL_PRIME_FD_TO_HANDLE)] = drm_shim_ioctl_prime_fd_to_handle,
};

/**
 * Implements the GEM core ioctls, and calls into driver-specific ioctls.
 */
int
drm_shim_ioctl(int fd, unsigned long request, void *arg)
{
   ASSERTED int type = _IOC_TYPE(request);
   int nr = _IOC_NR(request);

   assert(type == DRM_IOCTL_BASE);

   if (nr >= DRM_COMMAND_BASE && nr < DRM_COMMAND_END) {
      int driver_nr = nr - DRM_COMMAND_BASE;

      if (driver_nr < shim_device.driver_ioctl_count &&
          shim_device.driver_ioctls[driver_nr]) {
         return shim_device.driver_ioctls[driver_nr](fd, request, arg);
      }
   } else {
      if (nr < ARRAY_SIZE(core_ioctls) && core_ioctls[nr]) {
         return core_ioctls[nr](fd, request, arg);
      }
   }

   if (nr >= DRM_COMMAND_BASE && nr < DRM_COMMAND_END) {
      fprintf(stderr,
              "DRM_SHIM: unhandled driver DRM ioctl %d (0x%08lx)\n",
              nr - DRM_COMMAND_BASE, request);
   } else {
      fprintf(stderr,
              "DRM_SHIM: unhandled core DRM ioctl 0x%X (0x%08lx)\n",
              nr, request);
   }

   return -EINVAL;
}

int
drm_shim_bo_init(struct shim_bo *bo, size_t size)
{

   mtx_lock(&shim_device.mem_lock);
   bo->mem_addr = util_vma_heap_alloc(&shim_device.mem_heap, size, shim_page_size);
   mtx_unlock(&shim_device.mem_lock);

   if (!bo->mem_addr)
      return -ENOMEM;

   bo->size = size;
   p_atomic_set(&bo->refcount, 1);

   return 0;
}

struct shim_bo *
drm_shim_bo_lookup(struct shim_fd *shim_fd, int handle)
{
   if (!handle)
      return NULL;

   mtx_lock(&shim_fd->handle_lock);
   struct hash_entry *entry =
      _mesa_hash_table_search(shim_fd->handles, (void *)(uintptr_t)handle);
   struct shim_bo *bo = entry ? entry->data : NULL;

   /* Keep the handle's reference alive until this lookup owns its own. */
   if (bo)
      p_atomic_inc(&bo->refcount);
   mtx_unlock(&shim_fd->handle_lock);

   return bo;
}

void
drm_shim_bo_get(struct shim_bo *bo)
{
   p_atomic_inc(&bo->refcount);
}

void
drm_shim_bo_put(struct shim_bo *bo)
{
   if (p_atomic_dec_return(&bo->refcount) != 0)
      return;

   if (shim_device.driver_bo_free)
      shim_device.driver_bo_free(bo);

   /* Give the pages back. The backing memfd is a 4 GiB sparse file that is
    * never truncated, so without this every page a process ever touched
    * stays resident until it exits -- a long CTS run on the shim then holds
    * gigabytes of dead buffers. It also hands the next allocation zeroed
    * memory, as a fresh kernel BO would. */
   fallocate(shim_device.mem_fd, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE,
             bo->mem_addr, ALIGN_POT(bo->size, shim_page_size));

   mtx_lock(&shim_device.mem_lock);
   util_vma_heap_free(&shim_device.mem_heap, bo->mem_addr, bo->size);
   mtx_unlock(&shim_device.mem_lock);
   free(bo);
}

int
drm_shim_bo_get_handle(struct shim_fd *shim_fd, struct shim_bo *bo)
{
   /* We should probably have some real datastructure for finding the free
    * number.
    */
   mtx_lock(&shim_fd->handle_lock);
   for (int new_handle = 1; ; new_handle++) {
      void *key = (void *)(uintptr_t)new_handle;
      if (!_mesa_hash_table_search(shim_fd->handles, key)) {
         drm_shim_bo_get(bo);
         _mesa_hash_table_insert(shim_fd->handles, key, bo);
         mtx_unlock(&shim_fd->handle_lock);
         return new_handle;
      }
   }
   mtx_unlock(&shim_fd->handle_lock);

   return 0;
}

/* Creates an mmap offset for the BO in the DRM fd.
 */
uint64_t
drm_shim_bo_get_mmap_offset(struct shim_fd *shim_fd, struct shim_bo *bo)
{
   mtx_lock(&shim_device.mem_lock);
   _mesa_hash_table_u64_insert(shim_device.offset_map, bo->mem_addr, bo);
   mtx_unlock(&shim_device.mem_lock);

   /* reuse the buffer address as the mmap offset: */
   return bo->mem_addr;
}

void
drm_shim_init_iomem_region(off64_t offset, size_t size,
                           void *(*mmap_handler)(size_t, int, int, off64_t))
{
   shim_device.iomem_region.mmap = mmap_handler;
   shim_device.iomem_region.start = offset;
   shim_device.iomem_region.size = size;
}

/* For mmap() on the DRM fd, look up the BO from the "offset" and map the BO's
 * fd.
 */
void *
drm_shim_mmap(struct shim_fd *shim_fd, size_t length, int prot, int flags,
              int fd, off64_t offset)
{
   if (shim_device.iomem_region.mmap &&
       offset >= shim_device.iomem_region.start &&
       offset + length <= shim_device.iomem_region.start + shim_device.iomem_region.size) {
      return shim_device.iomem_region.mmap(length, prot, flags, offset);
   }

   mtx_lock(&shim_device.mem_lock);
   struct shim_bo *bo = _mesa_hash_table_u64_search(shim_device.offset_map, offset);
   mtx_unlock(&shim_device.mem_lock);

   if (!bo)
      return MAP_FAILED;

   if (length > bo->size)
      return MAP_FAILED;

   /* The offset we pass to mmap must be aligned to the page size */
   assert((bo->mem_addr & (shim_page_size - 1)) == 0);

   return mmap(NULL, length, prot, flags, shim_device.mem_fd, bo->mem_addr);
}
