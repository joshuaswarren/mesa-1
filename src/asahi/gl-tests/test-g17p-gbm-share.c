/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <fcntl.h>
#include <gbm.h>
#include <linux/dma-buf.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <xf86drm.h>
#include <drm_fourcc.h>

#define WIDTH 64
#define HEIGHT 32
#define ROW_BYTES (WIDTH * 4)
#define DATA_BYTES (ROW_BYTES * HEIGHT)
#define GUARD_SIZE 32
#define GUARD_BYTE 0xa5

struct guarded_image {
   uint8_t before[GUARD_SIZE];
   uint8_t data[DATA_BYTES];
   uint8_t after[GUARD_SIZE];
};

static bool
guard_ok(const uint8_t guard[GUARD_SIZE])
{
   for (unsigned i = 0; i < GUARD_SIZE; ++i) {
      if (guard[i] != GUARD_BYTE)
         return false;
   }

   return true;
}

static void
make_pattern(struct guarded_image *image, uint8_t salt)
{
   memset(image, GUARD_BYTE, sizeof(*image));
   for (unsigned y = 0; y < HEIGHT; ++y) {
      for (unsigned x = 0; x < WIDTH; ++x) {
         uint8_t *p = &image->data[(y * WIDTH + x) * 4];
         p[0] = (uint8_t)(salt + x * 3);
         p[1] = (uint8_t)(salt ^ (y * 5));
         p[2] = (uint8_t)(x ^ y ^ salt);
         p[3] = 0xff;
      }
   }
}

static bool
map_write(struct gbm_bo *bo, const struct guarded_image *image)
{
   uint32_t stride = 0;
   void *map_data = NULL;
   uint8_t *map = gbm_bo_map(bo, 0, 0, WIDTH, HEIGHT,
                             GBM_BO_TRANSFER_WRITE, &stride, &map_data);
   if (!map || !map_data || stride < ROW_BYTES) {
      fprintf(stderr, "GBM write map failed: map=%p token=%p stride=%u\n",
              map, map_data, stride);
      return false;
   }

   for (unsigned y = 0; y < HEIGHT; ++y)
      memcpy(map + y * stride, image->data + y * ROW_BYTES, ROW_BYTES);
   gbm_bo_unmap(bo, map_data);
   return true;
}

static bool
map_read_exact(struct gbm_bo *bo, const struct guarded_image *expected)
{
   struct guarded_image readback;
   memset(&readback, GUARD_BYTE, sizeof(readback));

   uint32_t stride = 0;
   void *map_data = NULL;
   const uint8_t *map = gbm_bo_map(bo, 0, 0, WIDTH, HEIGHT,
                                   GBM_BO_TRANSFER_READ, &stride, &map_data);
   if (!map || !map_data || stride < ROW_BYTES) {
      fprintf(stderr, "GBM read map failed: map=%p token=%p stride=%u\n",
              map, map_data, stride);
      return false;
   }

   for (unsigned y = 0; y < HEIGHT; ++y)
      memcpy(readback.data + y * ROW_BYTES, map + y * stride, ROW_BYTES);
   gbm_bo_unmap(bo, map_data);

   const bool exact = guard_ok(readback.before) && guard_ok(readback.after) &&
                      memcmp(readback.data, expected->data, DATA_BYTES) == 0;
   if (!exact)
      fprintf(stderr, "GBM mapped readback or guard mismatch\n");
   return exact;
}

static bool
dma_buf_cpu_sync(int fd, uint64_t access)
{
   struct dma_buf_sync sync = {.flags = DMA_BUF_SYNC_START | access};
   if (ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync))
      return false;
   sync.flags = DMA_BUF_SYNC_END | access;
   return ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync) == 0;
}

int
main(void)
{
   const bool noop = getenv("ASAHI_GL_NOOP_SHIM") != NULL;
   const int drm_fd = open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
   if (drm_fd < 0) {
      perror("open render node");
      return 1;
   }

   struct gbm_device *device = gbm_create_device(drm_fd);
   if (!device) {
      fprintf(stderr, "gbm_create_device failed\n");
      return 1;
   }

   const uint32_t use = GBM_BO_USE_RENDERING | GBM_BO_USE_LINEAR;
   struct gbm_bo *original =
      gbm_bo_create(device, WIDTH, HEIGHT, GBM_FORMAT_ARGB8888, use);
   if (!original || gbm_bo_get_width(original) != WIDTH ||
       gbm_bo_get_height(original) != HEIGHT ||
       gbm_bo_get_format(original) != GBM_FORMAT_ARGB8888 ||
       gbm_bo_get_plane_count(original) != 1 ||
       gbm_bo_get_stride(original) < ROW_BYTES ||
       gbm_bo_get_modifier(original) != DRM_FORMAT_MOD_LINEAR) {
      fprintf(stderr,
              "unexpected GBM allocation layout: %ux%u format=%#x planes=%d "
              "stride=%u modifier=%#llx\n",
              original ? gbm_bo_get_width(original) : 0,
              original ? gbm_bo_get_height(original) : 0,
              original ? gbm_bo_get_format(original) : 0,
              original ? gbm_bo_get_plane_count(original) : 0,
              original ? gbm_bo_get_stride(original) : 0,
              (unsigned long long)(original ? gbm_bo_get_modifier(original)
                                            : 0));
      return 1;
   }

   struct guarded_image first, second;
   make_pattern(&first, 0x19);
   make_pattern(&second, 0xc3);
   if (!map_write(original, &first) || !map_read_exact(original, &first) ||
       !guard_ok(first.before) || !guard_ok(first.after) ||
       !guard_ok(second.before) || !guard_ok(second.after)) {
      fprintf(stderr, "initial GBM map/guard gate failed\n");
      return 1;
   }

   struct drm_prime_handle invalid = {
      .handle = 0,
      .flags = DRM_CLOEXEC | DRM_RDWR,
      .fd = -1,
   };
   if (drmIoctl(drm_fd, DRM_IOCTL_PRIME_HANDLE_TO_FD, &invalid) == 0) {
      fprintf(stderr, "PRIME export accepted handle zero\n");
      return 1;
   }
   invalid.handle = gbm_bo_get_handle(original).u32;
   invalid.flags = UINT32_MAX;
   if (drmIoctl(drm_fd, DRM_IOCTL_PRIME_HANDLE_TO_FD, &invalid) == 0) {
      fprintf(stderr, "PRIME export accepted invalid flags\n");
      return 1;
   }
   invalid = (struct drm_prime_handle){.fd = -1};
   if (drmIoctl(drm_fd, DRM_IOCTL_PRIME_FD_TO_HANDLE, &invalid) == 0) {
      fprintf(stderr, "PRIME import accepted fd -1\n");
      return 1;
   }

   int export_fd = gbm_bo_get_fd(original);
   if (export_fd < 0 ||
       !(fcntl(export_fd, F_GETFD, 0) & FD_CLOEXEC)) {
      fprintf(stderr, "GBM dma-buf export failed or lost CLOEXEC\n");
      return 1;
   }

   const off_t exported_size = lseek(export_fd, 0, SEEK_END);
   if (exported_size < DATA_BYTES || (!noop && !dma_buf_cpu_sync(
                                            export_fd, DMA_BUF_SYNC_RW))) {
      fprintf(stderr, "dma-buf size/synchronization contract failed\n");
      return 1;
   }

   const int retained_fd = fcntl(export_fd, F_DUPFD_CLOEXEC, 0);
   close(export_fd);
   if (retained_fd < 0 ||
       !(fcntl(retained_fd, F_GETFD, 0) & FD_CLOEXEC)) {
      fprintf(stderr, "retained dma-buf fd contract failed\n");
      return 1;
   }

   struct gbm_import_fd_modifier_data import = {
      .width = WIDTH,
      .height = HEIGHT,
      .format = GBM_FORMAT_ARGB8888,
      .num_fds = 1,
      .fds = {retained_fd},
      .strides = {(int)gbm_bo_get_stride(original)},
      .offsets = {(int)gbm_bo_get_offset(original, 0)},
      .modifier = gbm_bo_get_modifier(original),
   };

   /* Unknown import types are an exact negative and must not consume the fd. */
   if (gbm_bo_import(device, UINT32_C(0xdeadbeef), &import, use) != NULL) {
      fprintf(stderr, "unknown GBM import type was accepted\n");
      return 1;
   }

   struct gbm_bo *imported =
      gbm_bo_import(device, GBM_BO_IMPORT_FD_MODIFIER, &import, use);
   close(retained_fd);
   if (!imported || gbm_bo_get_plane_count(imported) != 1 ||
       gbm_bo_get_modifier(imported) != DRM_FORMAT_MOD_LINEAR ||
       !map_read_exact(imported, &first)) {
      fprintf(stderr, "GBM dma-buf import/first read failed\n");
      return 1;
   }

   /* Both handles must name the same storage in both directions. */
   if (!map_write(imported, &second) || !map_read_exact(original, &second)) {
      fprintf(stderr, "GBM alias coherence failed\n");
      return 1;
   }

   gbm_bo_destroy(original);
   original = NULL;
   if (!map_read_exact(imported, &second)) {
      fprintf(stderr, "import did not retain storage ownership\n");
      return 1;
   }

   const int reexport_fd = gbm_bo_get_fd(imported);
   if (reexport_fd < 0 ||
       !(fcntl(reexport_fd, F_GETFD, 0) & FD_CLOEXEC) ||
       (!noop && !dma_buf_cpu_sync(reexport_fd, DMA_BUF_SYNC_READ))) {
      fprintf(stderr, "GBM re-export contract failed\n");
      return 1;
   }
   close(reexport_fd);

   if (!guard_ok(first.before) || !guard_ok(first.after) ||
       !guard_ok(second.before) || !guard_ok(second.after)) {
      fprintf(stderr, "source guard changed during sharing\n");
      return 1;
   }

   gbm_bo_destroy(imported);
   gbm_device_destroy(device);
   close(drm_fd);

   puts("GL05 GBM linear dma-buf export/import, aliasing, retention and guards exact");
   return 0;
}
