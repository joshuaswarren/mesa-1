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
 * hk_UnmapMemory2KHR. The only mock is the kernel mmap: mock_bo_mmap backs the
 * BO with one tmpfile (ftruncate to the BO size) and hands out MAP_SHARED
 * mappings of it, so two live mappings of the same BO show the same bytes,
 * like two CPU mappings of one GEM object. The placed target is a PROT_NONE
 * reservation that stays mapped until the MAP_FIXED placed map replaces it.
 * Every resource failure FAILs the case; there are no SKIP paths. There is no
 * GPU, no device node and no Vulkan entrypoint; those are covered by the
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
#include <unistd.h>

#include "agx_device.h"

/* The BO cache can hand back a BO larger than the request (agx_bo_cache_fetch);
 * a placed map covers the allocation only. Same numbers as the PR: 0x100000
 * allocation in a 0x104000 BO. */
#define ALLOC_SIZE 0x100000
#define BO_SIZE    0x104000

/* Two images (or planes) bound to one memory object at different offsets, as
 * hk_BindImageMemory2 allows. Both offsets sit inside ALLOC_SIZE so the
 * placed mapping can be checked for the same contents. */
#define BIND_OFF_A 0x1000
#define BIND_OFF_B 0x80000

#define CANARY_LEN 0x1000

/* Case selection bits. Plain unmap (hk_device_memory.c:405) is the default
 * application teardown; CASE_RESERVE uses the :394 overlay instead. */
#define CASE_RESERVE   (1u << 0)
#define CASE_ALIAS     (1u << 1)
#define CASE_MAP_FIRST (1u << 2)

static const struct {
   const char *name;
   unsigned flags;
} cases[] = {
   {"placed_map_after_bind_then_unmap", 0},
   {"reserve_unmap_after_bind", CASE_RESERVE},
   {"aliased_images_survive_placed_map", CASE_ALIAS},
   /* The proposed production fix stashes an existing application mapping
    * around the internal map; this control exercises that order. */
   {"map_before_bind_control", CASE_MAP_FIRST},
};

#define MOCK_MAX_MAPS 8

struct mock_map {
   void *addr;
   size_t len;
};

struct mock_state {
   /* Must stay first: ops callbacks only get the agx_device pointer. */
   struct agx_device dev;
   struct agx_bo bo;
   FILE *file;
   struct mock_map maps[MOCK_MAX_MAPS];
   unsigned map_count;
};

static void
mock_track(struct mock_state *m, void *addr, size_t len)
{
   for (unsigned i = 0; i < m->map_count; i++) {
      if (m->maps[i].addr == addr) {
         m->maps[i].len = len;
         return;
      }
   }
   if (m->map_count == MOCK_MAX_MAPS) {
      fprintf(stderr, "mock: mapping table full\n");
      exit(99);
   }
   m->maps[m->map_count].addr = addr;
   m->maps[m->map_count].len = len;
   m->map_count++;
}

/* The single mock: the kernel side of a GEM mmap. One tmpfile is the BO
 * storage; every mapping is MAP_SHARED over it, at fixed_addr with MAP_FIXED
 * when requested. Body mirrors agx_bo_mmap (agx_device.c) minus the ioctls;
 * like the production primitive it only hands back a pointer (or NULL) and
 * leaves the slot bookkeeping to the production inlines. */
static void *
mock_bo_mmap(struct agx_device *dev, struct agx_bo *bo, void *fixed_addr,
             size_t size)
{
   struct mock_state *m = (struct mock_state *)dev;

   void *p = mmap(fixed_addr, size, PROT_READ | PROT_WRITE,
                  MAP_SHARED | (fixed_addr ? MAP_FIXED : 0), fileno(m->file),
                  0);
   if (p == MAP_FAILED)
      return NULL;

   mock_track(m, p, size);
   return p;
}

static bool
mock_bo_init(struct mock_state *m, size_t bo_size)
{
   memset(m, 0, sizeof(*m));
   m->dev.ops.bo_mmap = mock_bo_mmap;
   m->bo.dev = &m->dev;
   m->bo.size = bo_size;
   m->bo.refcnt = 1;

   m->file = tmpfile();
   if (!m->file)
      return false;
   if (ftruncate(fileno(m->file), (off_t)bo_size) != 0) {
      fclose(m->file);
      m->file = NULL;
      return false;
   }
   return true;
}

/* Per-case cleanup: drop every mapping the shim made and the backing file. */
static void
mock_bo_cleanup(struct mock_state *m)
{
   for (unsigned i = 0; i < m->map_count; i++)
      munmap(m->maps[i].addr, m->maps[i].len);
   if (m->file)
      fclose(m->file);
}

/* Reserve the placed target with PROT_NONE and keep it mapped: the MAP_FIXED
 * placed map replaces it in place, never releasing the address first. */
static void *
reserve_placed_address(struct mock_state *m, size_t size)
{
   void *p = mmap(NULL, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
   if (p == MAP_FAILED)
      return NULL;
   mock_track(m, p, size);
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

#define CASE_FAIL(...)              \
   do {                             \
      printf("FAIL %s: ", name);    \
      printf(__VA_ARGS__);          \
      putchar('\n');                \
      goto out;                     \
   } while (0)

static int
run_case(const char *name, unsigned flags)
{
   struct mock_state m;
   int rc = 1;

   if (!mock_bo_init(&m, BO_SIZE))
      CASE_FAIL("tmpfile/ftruncate for the BO failed");

   /* Optional plain application map before bind, hk_device_memory.c:362 with
    * a NULL fixed address. */
   uint8_t *app_map = NULL;
   if (flags & CASE_MAP_FIRST) {
      app_map = agx_bo_map_placed(&m.bo, NULL, m.bo.size);
      if (app_map == NULL)
         CASE_FAIL("plain application map returned NULL");
   }

   /* Bind (hk_image.c:1333-1335): internal mapping retained by the image. */
   void *internal = agx_bo_map(&m.bo);
   if (internal == NULL)
      CASE_FAIL("internal map returned NULL");
   uint8_t *image_a = (uint8_t *)internal + BIND_OFF_A;
   uint8_t pat_a[CANARY_LEN], pat_b[CANARY_LEN];
   memset(pat_a, 0x5A, sizeof(pat_a));
   memset(pat_b, 0xC3, sizeof(pat_b));
   memcpy(image_a, pat_a, sizeof(pat_a));

   uint8_t *image_b = NULL;
   if (flags & CASE_ALIAS) {
      image_b = (uint8_t *)agx_bo_map(&m.bo) + BIND_OFF_B;
      memcpy(image_b, pat_b, sizeof(pat_b));
   }

   /* Coherence: the retained mapping and an earlier application mapping of
    * the same BO must show the same bytes. */
   if (app_map != NULL &&
       memcmp(app_map + BIND_OFF_A, pat_a, CANARY_LEN) != 0)
      CASE_FAIL("application and internal mappings disagree on BO contents");

   /* Placed map at the reserved address (hk_device_memory.c:362). */
   void *placed = reserve_placed_address(&m, ALLOC_SIZE);
   if (placed == NULL)
      CASE_FAIL("could not reserve the placed target address");
   uint8_t *mem_map = agx_bo_map_placed(&m.bo, placed, ALLOC_SIZE);
   if (mem_map != placed) {
      CASE_FAIL("placed map returned %p, requested %p", (void *)mem_map,
                placed);
   }

   /* Placed/internal coherence: the placed mapping covers the allocation and
    * must show the same BO bytes at the bound offsets. */
   if (memcmp(mem_map + BIND_OFF_A, pat_a, CANARY_LEN) != 0)
      CASE_FAIL("placed mapping does not show the BO contents at the bind "
                "offset");
   if ((flags & CASE_ALIAS) &&
       memcmp(mem_map + BIND_OFF_B, pat_b, CANARY_LEN) != 0)
      CASE_FAIL("placed mapping does not show the BO contents at the alias "
                "offset");

   /* Application teardown. Reserve: production :394-396; else :405. */
   if (flags & CASE_RESERVE) {
      void *err = mmap(m.bo._map, agx_bo_map_len(&m.bo), PROT_NONE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
      if (err == MAP_FAILED)
         CASE_FAIL("reserve overlay mmap failed");
      m.bo._map = NULL;
      m.bo._map_size = 0;
   } else {
      agx_bo_unmap(&m.bo);
   }

   /* Lifetime: host-copy read through the retained pointers (hk_image.c:1523).
    * The placed map and unmap must never touch the internal mapping. */
   uint8_t readback[CANARY_LEN];
   if (!guarded_copy(readback, image_a, CANARY_LEN))
      CASE_FAIL("host copy through the retained image mapping faults after "
                "placed map + unmap (omacom/mesa#8 r4418383484)");
   if (memcmp(readback, pat_a, CANARY_LEN) != 0)
      CASE_FAIL("retained image mapping was destroyed or reused: canary "
                "mismatch");
   if (flags & CASE_ALIAS) {
      if (!guarded_copy(readback, image_b, CANARY_LEN))
         CASE_FAIL("aliased image's retained mapping faults after placed map "
                   "+ unmap");
      if (memcmp(readback, pat_b, CANARY_LEN) != 0)
         CASE_FAIL("aliased image's retained mapping canary mismatch");
   }

   printf("PASS %s\n", name);
   rc = 0;

out:
   mock_bo_cleanup(&m);
   return rc;
}

int
main(void)
{
   int failures = 0;

   for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
      failures += run_case(cases[i].name, cases[i].flags);

   if (failures) {
      printf("%d case(s) failed: an application placed map or unmap destroys "
             "the internal mapping retained by a bound image\n",
             failures);
      return 1;
   }
   return 0;
}
