/*
 * Copyright 2026 Joshua Warren
 * SPDX-License-Identifier: MIT
 */

/*
 * CPU-only regression for the omacom/mesa#8 review finding
 * (https://github.com/omacom/mesa/pull/8#discussion_r4418383484) and its
 * follow-ups:
 *
 * A non-sparse image bind retains an internal host-copy mapping
 * (hk_image_plane_bind, hk_image.c:1334: plane->map = agx_bo_map(mem->bo) +
 * offset) without setting mem->map or taking a BO reference. While the
 * application mapping lived on the BO, any vkMapMemory2KHR placed map,
 * vkUnmapMemory2KHR (plain or VK_MEMORY_UNMAP_RESERVE_BIT_EXT) or
 * vkFreeMemory of the same memory destroyed that retained pointer, and two
 * memory objects sharing an imported BO shared one replaceable mapping. The
 * fix gives the BO a single stable internal mapping (agx_bo_map ->
 * bo->_map, read directly by decode.c and populated by wrap.c) and gives
 * each VkDeviceMemory its own application mapping from the bo_mmap
 * primitive, dropped only by that object's unmap, reserve unmap and free.
 *
 * This test drives the real production mapping code shared with the driver:
 * hk_memory_map_from_bo, hk_memory_unmap and hk_memory_unmap_reserve from
 * hk_device_memory.h operating on real struct hk_device_memory objects, plus
 * agx_bo_map exactly as defined in agx_device.h with the bind expression of
 * hk_image_plane_bind, the real agx_bo_unreference, and the real
 * agx_virtio_bo_mmap. A mutation of the shared helpers back to BO-owned
 * application mappings, or of the virtio primitive away from the per-length
 * ops contract, fails these cases. The mocks stand in for the kernel: the
 * bo_mmap shim backs the BO with one tmpfile (ftruncate to the BO size) and
 * hands out MAP_SHARED mappings of it, so two live mappings of the same BO
 * show the same bytes, like two CPU mappings of one GEM object, and the
 * vdrm host-side shim records what the virtio primitive asks the host to
 * map. The placed target is a PROT_NONE reservation that stays mapped until
 * the MAP_FIXED placed map replaces it. Every resource failure FAILs the
 * case; there are no SKIP paths. There is no GPU, no device node and no
 * Vulkan entrypoint; those are covered by the admitted device gate
 * (dEQP-VK host-copy coverage with a placed map after image bind), not
 * here.
 *
 * The stable-map case pins the decoder contract: the internal mapping must
 * exist, match agx_bo_map, stay pointer-stable and coherent through the
 * application map lifecycle. The two-importer and free cases pin shared-BO
 * lifetime: distinct mappings per memory object, mutual content coherence,
 * and a survivor whose mapping outlives the other importer's unmap or
 * free. The heap accounting, ext-BO tracking and vk_device_memory_destroy
 * tail of hk_FreeMemory need a live vk_device and stay behind the admitted
 * device gate.
 *
 * Red was parent-observed on the pre-fix lineage (fixture heads 8af285bd30c
 * and e14884a59ca: stable _map absent, importer B reusing importer A's
 * placed mapping; the free case stopped at the same pointer-independence
 * assertion before reaching the survivor checks). Passes with per-memory
 * ownership.
 */

#include <setjmp.h>
#include <signal.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "hk_device_memory.h"
#include "asahi/lib/agx_device_virtio.h"
#include "virtio/vdrm/vdrm.h"


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
   /* The app-first order must leave the internal BO map independent. */
   {"map_before_bind_keeps_maps_independent", CASE_MAP_FIRST},
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

   /* Host side of the virtio backend under test. */
   struct vdrm_device vdrm;
   size_t vdrm_map_size;
   void *vdrm_map_placed;
   bool vdrm_fail_next;
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
 * like the production primitive it writes the mapping slot on the supplied BO
 * copy, which exercises agx_bo_map_owned's real backend contract. */
static void
mock_bo_mmap(struct agx_device *dev, struct agx_bo *bo, void *fixed_addr,
             size_t size)
{
   struct mock_state *m = (struct mock_state *)dev;
   size = size ?: bo->size;
   void *p = mmap(fixed_addr, size, PROT_READ | PROT_WRITE,
                  MAP_SHARED | (fixed_addr ? MAP_FIXED : 0), fileno(m->file),
                  0);
   bo->_map = p == MAP_FAILED ? NULL : p;
   bo->_map_size = bo->_map ? size : 0;
   if (bo->_map)
      mock_track(m, bo->_map, size);
}

static void
mock_bo_munmap(struct agx_device *dev, struct agx_bo *bo)
{
   (void)dev;
   if (bo->_map)
      munmap(bo->_map, agx_bo_map_size(bo));
}

static bool
mock_bo_init(struct mock_state *m, size_t bo_size)
{
   memset(m, 0, sizeof(*m));
   m->dev.ops.bo_munmap = mock_bo_munmap;
   m->dev.ops.bo_mmap_independent = true;
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

static void
mock_mem_init(struct hk_device_memory *mem, struct agx_bo *bo)
{
   memset(mem, 0, sizeof(*mem));
   mem->bo = bo;
}

/* The memory object must own exactly the mapping the shared helper made. */
#define ASSERT_MAPPED(mem, want_ptr, want_size)                             \
   do {                                                                     \
      if ((mem)->map != (want_ptr) || (mem)->map_size != (want_size))       \
         CASE_FAIL("memory object does not own its mapping: map=%p "        \
                   "map_size=%zu", (mem)->map, (mem)->map_size);            \
   } while (0)

#define ASSERT_UNMAPPED(mem)                                                \
   do {                                                                     \
      if ((mem)->map != NULL || (mem)->map_size != 0)                       \
         CASE_FAIL("memory object mapping state not cleared: map=%p "       \
                   "map_size=%zu", (mem)->map, (mem)->map_size);            \
   } while (0)

static int
run_case(const char *name, unsigned flags)
{
   struct mock_state m;
   int rc = 1;

   if (!mock_bo_init(&m, BO_SIZE))
      CASE_FAIL("tmpfile/ftruncate for the BO failed");

   /* Real per-memory objects: an earlier memory object mapping the BO
    * plainly, and the one whose placed mapping this case tears down. */
   struct hk_device_memory pre_mem, mem;
   mock_mem_init(&pre_mem, &m.bo);
   mock_mem_init(&mem, &m.bo);

   /* Optional plain application map before bind: a mapping owned by its own
    * memory object (the hk_MapMemory2KHR ordinary expression via the shared
    * helper). */
   uint8_t *app_map = NULL;
   if (flags & CASE_MAP_FIRST) {
      app_map = hk_memory_map_from_bo(&pre_mem, NULL, m.bo.size);
      if (app_map == NULL)
         CASE_FAIL("plain application map returned NULL");
      ASSERT_MAPPED(&pre_mem, app_map, m.bo.size);
      if (m.bo._map != NULL)
         CASE_FAIL("application map populated stable BO mapping before bind");
   }

   /* Bind (hk_image.c:1333-1335): internal mapping retained by the image. */
   void *internal = agx_bo_map(&m.bo);
   if (internal == NULL)
      CASE_FAIL("internal map returned NULL");
   if ((flags & CASE_MAP_FIRST) && m.bo._map == app_map)
      CASE_FAIL("internal map reused the earlier application mapping");
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

   /* Placed map at the reserved address: a mapping owned by its own memory
    * object (the hk_MapMemory2KHR placed expression via the shared helper). */
   void *placed = reserve_placed_address(&m, ALLOC_SIZE);
   if (placed == NULL)
      CASE_FAIL("could not reserve the placed target address");
   uint8_t *mem_map = hk_memory_map_from_bo(&mem, placed, ALLOC_SIZE);
   if (mem_map != placed) {
      CASE_FAIL("placed map returned %p, requested %p", (void *)mem_map,
                placed);
   }
   ASSERT_MAPPED(&mem, placed, ALLOC_SIZE);

   /* Placed/internal coherence: the placed mapping covers the allocation and
    * must show the same BO bytes at the bound offsets. */
   if (memcmp(mem_map + BIND_OFF_A, pat_a, CANARY_LEN) != 0)
      CASE_FAIL("placed mapping does not show the BO contents at the bind "
                "offset");
   if ((flags & CASE_ALIAS) &&
       memcmp(mem_map + BIND_OFF_B, pat_b, CANARY_LEN) != 0)
      CASE_FAIL("placed mapping does not show the BO contents at the alias "
                "offset");

   /* Application teardown, owned by the mapped memory object
    * (hk_UnmapMemory2KHR): reserve overlays exactly that mapping, a plain
    * unmap releases it. The stable BO mapping is never touched. */
   if (flags & CASE_RESERVE) {
      if (!hk_memory_unmap_reserve(&mem))
         CASE_FAIL("reserve overlay mmap failed");
   } else {
      hk_memory_unmap(&mem);
   }
   ASSERT_UNMAPPED(&mem);

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

/* Follow-up finding: decode.c (agxdecode_grab_mapped :102, fetch :136,
 * cmdstream :976) and wrap.c (:210, :230) read and populate bo->_map directly.
 * The stable BO mapping must exist, match agx_bo_map, stay pointer-stable and
 * coherent through the application map lifecycle. */
static int
stable_map_survives_application_maps(void)
{
   const char *name = "stable_map_survives_application_maps";
   struct mock_state m;
   int rc = 1;

   if (!mock_bo_init(&m, BO_SIZE))
      CASE_FAIL("tmpfile/ftruncate for the BO failed");

   /* Decoder-style stable mapping. */
   uint8_t *stable = agx_bo_map(&m.bo);
   if (stable == NULL)
      CASE_FAIL("internal map returned NULL");
   if (m.bo._map == NULL)
      CASE_FAIL("internal map did not populate the stable BO mapping _map "
                "read by decode.c and wrap.c");
   if (m.bo._map != (void *)stable)
      CASE_FAIL("agx_bo_map result does not match the stable BO mapping");
   memset(stable, 0x77, CANARY_LEN);

   /* Application placed map over the same BO: a mapping owned by its own
    * memory object (the hk_MapMemory2KHR placed expression via the shared
    * helper). */
   struct hk_device_memory mem;
   mock_mem_init(&mem, &m.bo);

   void *placed = reserve_placed_address(&m, ALLOC_SIZE);
   if (placed == NULL)
      CASE_FAIL("could not reserve the placed target address");
   uint8_t *mem_map = hk_memory_map_from_bo(&mem, placed, ALLOC_SIZE);
   if (mem_map != placed) {
      CASE_FAIL("placed map returned %p, requested %p", (void *)mem_map,
                placed);
   }
   ASSERT_MAPPED(&mem, placed, ALLOC_SIZE);
   uint8_t readback[CANARY_LEN];
   if (!guarded_copy(readback, stable, CANARY_LEN))
      CASE_FAIL("stable BO mapping faults while checking map coherence");
   if (memcmp(mem_map, readback, CANARY_LEN) != 0)
      CASE_FAIL("placed mapping disagrees with the stable BO mapping");

   /* Application unmap of that memory object's mapping
    * (hk_UnmapMemory2KHR). */
   hk_memory_unmap(&mem);
   ASSERT_UNMAPPED(&mem);

   /* The stable mapping survives: same pointer, same bytes. */
   if (m.bo._map != (void *)stable)
      CASE_FAIL("application maps replaced the stable BO mapping pointer");
   if (!guarded_copy(readback, stable, CANARY_LEN))
      CASE_FAIL("stable BO mapping faults after application placed map + "
                "unmap");
   for (size_t i = 0; i < CANARY_LEN; i++) {
      if (readback[i] != 0x77)
         CASE_FAIL("stable BO mapping was destroyed or reused: canary "
                   "mismatch at offset %zu",
                   i);
   }

   printf("PASS %s\n", name);
   rc = 0;

out:
   mock_bo_cleanup(&m);
   return rc;
}

/* Follow-up finding: two VkDeviceMemory importers share one BO. An ordinary
 * map of the second importer must not reuse the first importer's placed
 * mapping, and the first importer's unmap must not destroy the second's
 * mapping. Both mappings must show the same BO contents. */
static int
shared_bo_importers_map_independently(void)
{
   const char *name = "shared_bo_importers_map_independently";
   struct mock_state m;
   int rc = 1;

   if (!mock_bo_init(&m, BO_SIZE))
      CASE_FAIL("tmpfile/ftruncate for the BO failed");
   m.bo.refcnt = 2; /* imported share: two memory objects, one BO */

   struct hk_device_memory mem_a, mem_b;
   mock_mem_init(&mem_a, &m.bo);
   mock_mem_init(&mem_b, &m.bo);

   /* Importer A: placed map (hk_MapMemory2KHR placed expression via the
    * shared helper, fixed address). */
   void *placed_a = reserve_placed_address(&m, ALLOC_SIZE);
   if (placed_a == NULL)
      CASE_FAIL("could not reserve the placed target address");
   uint8_t *map_a = hk_memory_map_from_bo(&mem_a, placed_a,
                                          ALLOC_SIZE);
   if (map_a != placed_a) {
      CASE_FAIL("importer A placed map returned %p, requested %p",
                (void *)map_a, placed_a);
   }
   ASSERT_MAPPED(&mem_a, placed_a, ALLOC_SIZE);

   uint8_t pat_a[CANARY_LEN], pat_b[CANARY_LEN];
   memset(pat_a, 0x5A, sizeof(pat_a));
   memset(pat_b, 0xC3, sizeof(pat_b));
   memcpy(map_a + BIND_OFF_A, pat_a, sizeof(pat_a));

   /* Importer B: ordinary map (hk_MapMemory2KHR ordinary expression via the
    * shared helper, NULL fixed, whole BO). It must be its own mapping, not
    * importer A's. */
   uint8_t *map_b = hk_memory_map_from_bo(&mem_b, NULL, m.bo.size);
   if (map_b == NULL)
      CASE_FAIL("importer B map returned NULL");
   if (map_b == map_a)
      CASE_FAIL("importer B ordinary map reused importer A's placed mapping");
   ASSERT_MAPPED(&mem_b, map_b, m.bo.size);

   /* Coherence: both mappings show the same BO contents. */
   if (memcmp(map_b + BIND_OFF_A, pat_a, CANARY_LEN) != 0)
      CASE_FAIL("importer B mapping disagrees with importer A on BO "
                "contents");
   memcpy(map_b + BIND_OFF_B, pat_b, sizeof(pat_b));
   if (memcmp(map_a + BIND_OFF_B, pat_b, CANARY_LEN) != 0)
      CASE_FAIL("importer A mapping disagrees with importer B on BO "
                "contents");

   /* Importer A unmaps its own mapping (hk_UnmapMemory2KHR). Only A's
    * mapping may die. */
   hk_memory_unmap(&mem_a);
   ASSERT_UNMAPPED(&mem_a);

   uint8_t readback[CANARY_LEN];
   if (!guarded_copy(readback, map_b + BIND_OFF_A, CANARY_LEN))
      CASE_FAIL("importer B mapping faults after importer A unmap");
   if (memcmp(readback, pat_a, CANARY_LEN) != 0)
      CASE_FAIL("importer B mapping was destroyed or reused by importer A's "
                "unmap: canary mismatch");

   printf("PASS %s\n", name);
   rc = 0;

out:
   mock_bo_cleanup(&m);
   return rc;
}

/* The user-required free path: two memory objects share one imported BO;
 * freeing one runs the BO-relevant hk_FreeMemory sequence (placed-unmap
 * condition, hk_device_memory.c:283, plus the real agx_bo_unreference) and
 * the survivor's mapping must stay valid and coherent. The heap accounting,
 * ext-BO tracking and vk_device_memory_destroy tail of hk_FreeMemory need a
 * live vk_device and stay behind the admitted device gate. */
static int
free_one_importer_survivor_survives(void)
{
   const char *name = "free_one_importer_survivor_survives";
   struct mock_state m;
   int rc = 1;

   if (!mock_bo_init(&m, BO_SIZE))
      CASE_FAIL("tmpfile/ftruncate for the BO failed");
   m.bo.refcnt = 2; /* imported share: two memory objects, one BO */

   struct hk_device_memory mem_a, mem_b;
   mock_mem_init(&mem_a, &m.bo);
   mock_mem_init(&mem_b, &m.bo);

   /* Importer A: placed map (hk_MapMemory2KHR placed expression via the
    * shared helper, fixed address). */
   void *placed_a = reserve_placed_address(&m, ALLOC_SIZE);
   if (placed_a == NULL)
      CASE_FAIL("could not reserve the placed target address");
   uint8_t *map_a = hk_memory_map_from_bo(&mem_a, placed_a,
                                          ALLOC_SIZE);
   if (map_a != placed_a) {
      CASE_FAIL("importer A placed map returned %p, requested %p",
                (void *)map_a, placed_a);
   }
   ASSERT_MAPPED(&mem_a, placed_a, ALLOC_SIZE);

   /* Importer B: ordinary map (hk_MapMemory2KHR ordinary expression via the
    * shared helper, NULL fixed). */
   uint8_t *map_b = hk_memory_map_from_bo(&mem_b, NULL, m.bo.size);
   if (map_b == NULL)
      CASE_FAIL("importer B map returned NULL");
   if (map_b == map_a)
      CASE_FAIL("importer B ordinary map reused importer A's placed mapping");
   ASSERT_MAPPED(&mem_b, map_b, m.bo.size);

   uint8_t pat_a[CANARY_LEN], pat_b[CANARY_LEN];
   memset(pat_a, 0x5A, sizeof(pat_a));
   memset(pat_b, 0xC3, sizeof(pat_b));
   memcpy(map_a + BIND_OFF_A, pat_a, sizeof(pat_a));
   memcpy(map_b + BIND_OFF_B, pat_b, sizeof(pat_b));

   /* vkFreeMemory(A): the mapping belongs to the freed memory object, so
    * hk_FreeMemory drops exactly A's mapping via the shared helper, and the
    * real refcount drop returns the BO alive at refcount 1 for B. */
   hk_memory_unmap(&mem_a);
   ASSERT_UNMAPPED(&mem_a);
   agx_bo_unreference(&m.dev, &m.bo);
   if (p_atomic_read(&m.bo.refcnt) != 1)
      CASE_FAIL("freeing one importer did not leave the shared BO at "
                "refcount 1 (got %d)",
                p_atomic_read(&m.bo.refcnt));
   ASSERT_MAPPED(&mem_b, map_b, m.bo.size);

   /* Survivor B: mapping still valid, both canaries coherent. */
   uint8_t readback[CANARY_LEN];
   if (!guarded_copy(readback, map_b + BIND_OFF_A, CANARY_LEN))
      CASE_FAIL("survivor mapping faults after the other importer was "
                "freed");
   if (memcmp(readback, pat_a, CANARY_LEN) != 0)
      CASE_FAIL("survivor mapping was destroyed or reused by the other "
                "importer's free: canary mismatch");
   if (!guarded_copy(readback, map_b + BIND_OFF_B, CANARY_LEN))
      CASE_FAIL("survivor mapping faults on its own written range after "
                "the other importer was freed");
   if (memcmp(readback, pat_b, CANARY_LEN) != 0)
      CASE_FAIL("survivor's own written range corrupted after the other "
                "importer was freed");

   printf("PASS %s\n", name);
   rc = 0;

out:
   mock_bo_cleanup(&m);
   return rc;
}

/* Boundary case: the shared helpers must treat a never-mapped memory object
 * as a no-op (hk_UnmapMemory2KHR on unmapped memory must not touch the BO's
 * stable mapping), and a later real map on the same object must work. */
static int
unmap_without_map_is_noop(void)
{
   const char *name = "unmap_without_map_is_noop";
   struct mock_state m;
   int rc = 1;

   if (!mock_bo_init(&m, BO_SIZE))
      CASE_FAIL("tmpfile/ftruncate for the BO failed");

   struct hk_device_memory mem;
   mock_mem_init(&mem, &m.bo);
   ASSERT_UNMAPPED(&mem);

   hk_memory_unmap(&mem);
   if (!hk_memory_unmap_reserve(&mem))
      CASE_FAIL("reserve unmap of a never-mapped object failed");
   ASSERT_UNMAPPED(&mem);

   /* The BO's stable mapping is untouched and a real map works after. */
   uint8_t *stable = agx_bo_map(&m.bo);
   if (stable == NULL)
      CASE_FAIL("internal map returned NULL");
   ASSERT_UNMAPPED(&mem);

   uint8_t *map = hk_memory_map_from_bo(&mem, NULL, m.bo.size);
   if (map == NULL)
      CASE_FAIL("map after no-op unmaps returned NULL");
   ASSERT_MAPPED(&mem, map, m.bo.size);
   hk_memory_unmap(&mem);
   ASSERT_UNMAPPED(&mem);

   printf("PASS %s\n", name);
   rc = 0;

out:
   mock_bo_cleanup(&m);
   return rc;
}

/* The virtio mapping primitive is the one bo_mmap implementer the ops
 * contract binds by length: vdrm_bo_map must receive exactly the requested
 * placed length (the application reserved only that much address space) and
 * record it, while a length of 0 keeps the whole cached BO. Drives the real
 * agx_virtio_bo_mmap with a stubbed vdrm host side; MAP_FAILED propagation
 * keeps the cleanup path covered. */
static void *
mock_vdrm_bo_map(struct vdrm_device *vdev, uint32_t handle, size_t size,
                 void *placed_addr)
{
   struct mock_state *m = (struct mock_state *)((char *)vdev -
                                                offsetof(struct mock_state,
                                                         vdrm));

   if (m->vdrm_fail_next)
      return MAP_FAILED;

   m->vdrm_map_size = size;
   m->vdrm_map_placed = placed_addr;

   void *p = mmap(placed_addr, size, PROT_READ | PROT_WRITE,
                  MAP_SHARED | (placed_addr ? MAP_FIXED : 0), fileno(m->file),
                  0);
   if (p == MAP_FAILED)
      return MAP_FAILED;

   mock_track(m, p, size);
   return p;
}

static int
virtio_mmap_honors_length(void)
{
   const char *name = "virtio_mmap_honors_length";
   struct mock_state m;
   int rc = 1;

   if (!mock_bo_init(&m, BO_SIZE))
      CASE_FAIL("tmpfile/ftruncate for the BO failed");

   /* The production primitive only reads dev->vdrm and dev->fd. */
   memset(&m.vdrm, 0, sizeof(m.vdrm));
   static const struct vdrm_device_funcs funcs = {
      .bo_map = mock_vdrm_bo_map,
   };
   m.vdrm.funcs = &funcs;
   m.dev.vdrm = &m.vdrm;
   m.dev.fd = fileno(m.file);
   m.bo.handle = 1;

   /* Placed: the BO cache may hold a mapping up to twice the allocation, so
    * the primitive must ask the host for exactly the requested length. The
    * reservation extends past the placed target, so the pre-fix oversized
    * MAP_FIXED lands in reserved address space instead of corrupting live
    * memory and the assertions report the defect. */
   void *placed = reserve_placed_address(&m, BO_SIZE + ALLOC_SIZE);
   if (placed == NULL)
      CASE_FAIL("could not reserve the placed target address");
   agx_virtio_bo_mmap(&m.dev, &m.bo, placed, ALLOC_SIZE);
   if (m.bo._map != placed || m.bo._map_size != ALLOC_SIZE)
      CASE_FAIL("virtio placed map returned map=%p size=%zu, wanted %p+%zu",
                m.bo._map, m.bo._map_size, placed, (size_t)ALLOC_SIZE);
   if (m.vdrm_map_size != ALLOC_SIZE || m.vdrm_map_placed != placed)
      CASE_FAIL("virtio host map requested size=%zu placed=%p, wanted "
                "%u+%p",
                m.vdrm_map_size, m.vdrm_map_placed,
                (unsigned)ALLOC_SIZE, placed);

   /* Ordinary: a length of 0 maps and records the whole cached BO. */
   agx_virtio_bo_mmap(&m.dev, &m.bo, NULL, 0);
   if (m.bo._map == NULL || m.bo._map == placed ||
       m.bo._map_size != m.bo.size)
      CASE_FAIL("virtio ordinary map returned map=%p size=%zu, wanted the "
                "whole BO",
                m.bo._map, m.bo._map_size);
   if (m.vdrm_map_size != m.bo.size)
      CASE_FAIL("virtio host ordinary map asked for %zu, wanted %zu",
                m.vdrm_map_size, m.bo.size);
   munmap(m.bo._map, m.bo._map_size);

   /* MAP_FAILED from the host must leave the BO slot cleared. */
   m.vdrm_fail_next = true;
   agx_virtio_bo_mmap(&m.dev, &m.bo, NULL, 0);
   if (m.bo._map != NULL || m.bo._map_size != 0)
      CASE_FAIL("virtio MAP_FAILED left map=%p size=%zu on the BO",
                m.bo._map, m.bo._map_size);

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

   failures += stable_map_survives_application_maps();
   failures += shared_bo_importers_map_independently();
   failures += free_one_importer_survivor_survives();
   failures += unmap_without_map_is_noop();
   failures += virtio_mmap_honors_length();

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
