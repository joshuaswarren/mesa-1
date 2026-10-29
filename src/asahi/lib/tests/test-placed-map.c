/*
 * Copyright 2026 Joshua Warren
 * SPDX-License-Identifier: MIT
 */

/*
 * CPU-only regression for the omacom/mesa#8 review finding
 * (https://github.com/omacom/mesa/pull/8#discussion_r4418383484):
 *
 * A non-sparse image bind retains an internal host-copy mapping
 * (hk_image_plane_bind, hk_image.c:1334: plane->map = agx_bo_map(mem->bo) +
 * offset) without setting mem->map or taking a BO reference. A subsequent
 * application vkMapMemory2KHR with VK_MEMORY_MAP_PLACED_BIT_EXT reaches
 * agx_bo_map_placed (agx_device.h) with a fixed address, which unconditionally
 * drops the BO's current mapping. The retained plane->map then dangles, and
 * the host-copy paths (hk_image.c:1444, :1523, :1610) read and write through
 * it. The plain unmap path (hk_device_memory.c:405) and the
 * VK_MEMORY_UNMAP_RESERVE_BIT_EXT path (:394) invalidate the same retained
 * pointer.
 *
 * This test drives the real production mapping code: agx_bo_map,
 * agx_bo_map_placed and agx_bo_unmap exactly as defined in agx_device.h, with
 * the exact call expressions used by hk_image_plane_bind, hk_MapMemory2KHR and
 * hk_UnmapMemory2KHR. The only mock is the kernel mmap: mock_bo_mmap hands
 * back anonymous pages for the request, like the DRM GEM mmap path does. There
 * is no GPU, no device node and no Vulkan entrypoint; those are covered by the
 * admitted device gate (dEQP-VK host-copy coverage with a placed map after
 * image bind), not here.
 *
 * Fails at head ddfb4cf114a6: the retained image mapping is destroyed and the
 * host-copy read faults or reads foreign memory. Passes once the internal
 * image mapping is independent of the application mapping.
 */

#include <setjmp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "agx_device.h"

/* The BO cache can hand back a BO larger than the request (agx_bo_cache_fetch);
 * a placed map covers the allocation only. Same numbers as the PR: 0x100000
 * allocation in a 0x104000 BO. */
#define ALLOC_SIZE 0x100000
#define BO_SIZE    0x104000

/* Two images (or planes) bound to one memory object at different offsets, as
 * hk_BindImageMemory2 allows. */
#define BIND_OFF_A 0x1000
#define BIND_OFF_B 0x80000

#define CANARY_LEN 0x1000

struct mock_state {
   struct agx_device dev;
   struct agx_bo bo;
};

/* The single mock: the kernel side of a GEM mmap. Body mirrors agx_bo_mmap
 * (agx_device.c) minus the ioctls. */
static void
mock_bo_mmap(struct agx_device *dev, struct agx_bo *bo, void *fixed_addr,
             size_t size)
{
   (void)dev;
   if (bo->_map != NULL) {
      fprintf(stderr, "mock: bo double mapped\n");
      exit(99);
   }

   bo->_map = mmap(fixed_addr, size, PROT_READ | PROT_WRITE,
                   MAP_SHARED | MAP_ANONYMOUS | (fixed_addr ? MAP_FIXED : 0),
                   -1, 0);
   if (bo->_map == MAP_FAILED) {
      bo->_map = NULL;
      bo->_map_size = 0;
   } else {
      bo->_map_size = size;
   }
}

static void
mock_bo_init(struct mock_state *m, size_t bo_size)
{
   memset(m, 0, sizeof(*m));
   m->dev.ops.bo_mmap = mock_bo_mmap;
   m->bo.dev = &m->dev;
   m->bo.size = bo_size;
   m->bo.refcnt = 1;
}

/* Grab a page-aligned address that is (almost certainly) free, so the placed
 * map target cannot alias the retained mapping. */
static void *
reserve_free_address(size_t size)
{
   void *p = mmap(NULL, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
   if (p == MAP_FAILED)
      return NULL;
   munmap(p, size);
   return p;
}

static sigjmp_buf fault_env;
static volatile sig_atomic_t faulted;

static void
fault_handler(int sig)
{
   (void)sig;
   faulted = 1;
   siglongjmp(fault_env, 1);
}

/* Reads through a possibly-dangling retained pointer the way the host-copy
 * paths do (hk_image.c:1523). Returns false when the access faults. */
static bool
guarded_copy(void *dst, const void *src, size_t n)
{
   void *prev_segv = signal(SIGSEGV, fault_handler);
   void *prev_bus = signal(SIGBUS, fault_handler);

   faulted = 0;
   if (sigsetjmp(fault_env, 1) == 0)
      memcpy(dst, src, n);

   signal(SIGSEGV, prev_segv);
   signal(SIGBUS, prev_bus);
   return !faulted;
}

/* Bind (hk_image.c:1333-1335), placed map (hk_device_memory.c:362), plain
 * unmap (:405), host-copy read through the retained pointer (hk_image.c:1523).
 */
static int
placed_map_after_bind_then_unmap(void)
{
   struct mock_state m;
   mock_bo_init(&m, BO_SIZE);

   uint8_t *image_map = (uint8_t *)agx_bo_map(&m.bo) + BIND_OFF_A;
   uint8_t pattern[CANARY_LEN];
   memset(pattern, 0x5A, sizeof(pattern));
   memcpy(image_map, pattern, sizeof(pattern));

   void *placed = reserve_free_address(ALLOC_SIZE);
   if (!placed) {
      printf("SKIP placed_map_after_bind_then_unmap: no free address\n");
      return 0;
   }

   uint8_t *mem_map = agx_bo_map_placed(&m.bo, placed, ALLOC_SIZE);
   if (mem_map != placed) {
      printf("FAIL placed_map_after_bind_then_unmap: placed map returned %p, "
             "requested %p\n",
             (void *)mem_map, placed);
      return 1;
   }

   agx_bo_unmap(&m.bo);

   uint8_t readback[CANARY_LEN];
   if (!guarded_copy(readback, image_map, CANARY_LEN)) {
      printf("FAIL placed_map_after_bind_then_unmap: host copy through the "
             "retained image mapping faults after placed map + unmap "
             "(omacom/mesa#8 r4418383484)\n");
      return 1;
   }
   if (memcmp(readback, pattern, CANARY_LEN) != 0) {
      printf("FAIL placed_map_after_bind_then_unmap: retained image mapping "
             "was destroyed or reused: canary mismatch\n");
      return 1;
   }

   printf("PASS placed_map_after_bind_then_unmap\n");
   return 0;
}

/* Same sequence, but the application unmaps with
 * VK_MEMORY_UNMAP_RESERVE_BIT_EXT: the production path (hk_device_memory.c:394)
 * overlays the BO mapping with PROT_NONE, which must not touch the internal
 * image mapping either. */
static int
reserve_unmap_after_bind(void)
{
   struct mock_state m;
   mock_bo_init(&m, BO_SIZE);

   uint8_t *image_map = (uint8_t *)agx_bo_map(&m.bo) + BIND_OFF_A;
   uint8_t pattern[CANARY_LEN];
   memset(pattern, 0xA5, sizeof(pattern));
   memcpy(image_map, pattern, sizeof(pattern));

   void *placed = reserve_free_address(ALLOC_SIZE);
   if (!placed) {
      printf("SKIP reserve_unmap_after_bind: no free address\n");
      return 0;
   }

   uint8_t *mem_map = agx_bo_map_placed(&m.bo, placed, ALLOC_SIZE);
   if (mem_map != placed) {
      printf("FAIL reserve_unmap_after_bind: placed map returned %p, "
             "requested %p\n",
             (void *)mem_map, placed);
      return 1;
   }

   /* Production reserve expression (hk_device_memory.c:395). */
   void *err = mmap(m.bo._map, agx_bo_map_len(&m.bo), PROT_NONE,
                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
   if (err == MAP_FAILED) {
      printf("SKIP reserve_unmap_after_bind: reserve mmap failed\n");
      return 0;
   }
   m.bo._map = NULL;
   m.bo._map_size = 0;

   uint8_t readback[CANARY_LEN];
   if (!guarded_copy(readback, image_map, CANARY_LEN)) {
      printf("FAIL reserve_unmap_after_bind: host copy through the retained "
             "image mapping faults after placed map + reserve unmap\n");
      return 1;
   }
   if (memcmp(readback, pattern, CANARY_LEN) != 0) {
      printf("FAIL reserve_unmap_after_bind: retained image mapping was "
             "destroyed: canary mismatch\n");
      return 1;
   }

   printf("PASS reserve_unmap_after_bind\n");
   return 0;
}

/* Two images bound to one memory object: one placed map must not invalidate
 * either retained mapping (hk_BindImageMemory2 binds each plane of an aliasing
 * image from the same mem->bo). */
static int
aliased_images_survive_placed_map(void)
{
   struct mock_state m;
   mock_bo_init(&m, BO_SIZE);

   uint8_t *image_a = (uint8_t *)agx_bo_map(&m.bo) + BIND_OFF_A;
   uint8_t *image_b = (uint8_t *)agx_bo_map(&m.bo) + BIND_OFF_B;
   uint8_t pattern_a[CANARY_LEN], pattern_b[CANARY_LEN];
   memset(pattern_a, 0x5A, sizeof(pattern_a));
   memset(pattern_b, 0xC3, sizeof(pattern_b));
   memcpy(image_a, pattern_a, sizeof(pattern_a));
   memcpy(image_b, pattern_b, sizeof(pattern_b));

   void *placed = reserve_free_address(ALLOC_SIZE);
   if (!placed) {
      printf("SKIP aliased_images_survive_placed_map: no free address\n");
      return 0;
   }

   uint8_t *mem_map = agx_bo_map_placed(&m.bo, placed, ALLOC_SIZE);
   if (mem_map != placed) {
      printf("FAIL aliased_images_survive_placed_map: placed map returned "
             "%p, requested %p\n",
             (void *)mem_map, placed);
      return 1;
   }

   agx_bo_unmap(&m.bo);

   int rc = 0;
   uint8_t readback[CANARY_LEN];
   if (!guarded_copy(readback, image_a, CANARY_LEN) ||
       memcmp(readback, pattern_a, CANARY_LEN) != 0) {
      printf("FAIL aliased_images_survive_placed_map: first image's retained "
             "mapping is dangling after placed map + unmap\n");
      rc = 1;
   }
   if (!guarded_copy(readback, image_b, CANARY_LEN) ||
       memcmp(readback, pattern_b, CANARY_LEN) != 0) {
      printf("FAIL aliased_images_survive_placed_map: second image's retained "
             "mapping is dangling after placed map + unmap\n");
      rc = 1;
   }
   if (!rc)
      printf("PASS aliased_images_survive_placed_map\n");
   return rc;
}

int
main(void)
{
   int failures = 0;

   failures += placed_map_after_bind_then_unmap();
   failures += reserve_unmap_after_bind();
   failures += aliased_images_survive_placed_map();

   if (failures) {
      printf("%d case(s) failed: an application placed map or unmap destroys "
             "the internal mapping retained by a bound image\n",
             failures);
      return 1;
   }
   return 0;
}
