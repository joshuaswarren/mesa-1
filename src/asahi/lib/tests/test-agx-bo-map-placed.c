/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "asahi/lib/agx_device.h"

#include <assert.h>
#include <stdint.h>

static unsigned mmap_count;
static unsigned munmap_count;
static uint8_t default_mapping;

static size_t last_mmap_length;

static void
fake_bo_mmap(struct agx_device *dev, struct agx_bo *bo, void *fixed_addr,
             size_t length)
{
   assert(dev == bo->dev);
   assert(bo->_map == NULL);
   mmap_count++;
   last_mmap_length = length;
   bo->_map = fixed_addr ? fixed_addr : &default_mapping;
   bo->_map_size = length ? length : bo->size;
}

static void *last_munmap;
static size_t last_munmap_size;

static void
fake_bo_munmap(struct agx_device *dev, struct agx_bo *bo)
{
   assert(dev == bo->dev);
   assert(bo->_map != NULL);
   munmap_count++;
   last_munmap = bo->_map;
   last_munmap_size = bo->_map_size;
}

int
main(void)
{
   struct agx_device dev = {
      .ops = {
         .bo_mmap = fake_bo_mmap,
         .bo_munmap = fake_bo_munmap,
      },
   };
   struct agx_bo bo = {
      .dev = &dev,
      .size = 16384,
   };
   uint8_t cached_mapping, placed_mapping;

   bo._map = &cached_mapping;

   /* Ordinary maps retain the BO cache's process mapping. */
   assert(agx_bo_map(&bo) == &cached_mapping);
   assert(mmap_count == 0 && munmap_count == 0);

   /* Requesting the same placed address is also a no-op. */
   assert(agx_bo_map_placed(&bo, &cached_mapping, 4096) == &cached_mapping);
   assert(mmap_count == 0 && munmap_count == 0);

   /* A different placed address must move, rather than reuse, the mapping,
    * and must map only the length the caller asked for.  The BO cache returns
    * BOs up to twice the requested size, so mapping bo->size at an address the
    * application reserved for a smaller allocation would overwrite whatever
    * follows it -- the guard pages dEQP-VK.memory.map_placed.* checks. */
   assert(agx_bo_map_placed(&bo, &placed_mapping, 4096) == &placed_mapping);
   assert(mmap_count == 1 && munmap_count == 1);
   assert(last_mmap_length == 4096 && "placed map bounded by the caller");

   /* Once moved, ordinary and identical placed maps reuse the new address. */
   assert(agx_bo_map(&bo) == &placed_mapping);
   assert(agx_bo_map_placed(&bo, &placed_mapping, 4096) == &placed_mapping);
   assert(mmap_count == 1 && munmap_count == 1);

   /* Owned maps need a backend that maps a BO more than once. */
   size_t owned_size = 0;
   assert(agx_bo_map_owned(&bo, NULL, 0, &owned_size) == NULL);
   assert(mmap_count == 1);

   /* An owned map is a new mapping that leaves the shared one alone, and
    * unmapping it releases exactly that mapping. */
   dev.ops.bo_mmap_independent = true;
   uint8_t owned_mapping;
   assert(agx_bo_map_owned(&bo, NULL, 0, &owned_size) == &default_mapping);
   assert(owned_size == bo.size && mmap_count == 2);
   assert(bo._map == &placed_mapping && "shared mapping untouched");
   assert(agx_bo_map_owned(&bo, &owned_mapping, 4096, &owned_size) ==
          &owned_mapping);
   assert(last_mmap_length == 4096 && owned_size == 4096 &&
          "owned placed map bounded by the caller");
   agx_bo_unmap_owned(&bo, &owned_mapping, owned_size);
   assert(munmap_count == 2 && last_munmap == &owned_mapping &&
          last_munmap_size == 4096);
   assert(bo._map == &placed_mapping && "shared mapping untouched");

   return 0;
}
