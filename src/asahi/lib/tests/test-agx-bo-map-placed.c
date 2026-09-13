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

static void
fake_bo_mmap(struct agx_device *dev, struct agx_bo *bo, void *fixed_addr)
{
   assert(dev == bo->dev);
   assert(bo->_map == NULL);
   mmap_count++;
   bo->_map = fixed_addr ? fixed_addr : &default_mapping;
}

static void
fake_bo_munmap(struct agx_device *dev, struct agx_bo *bo)
{
   assert(dev == bo->dev);
   assert(bo->_map != NULL);
   munmap_count++;
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
   assert(agx_bo_map_placed(&bo, &cached_mapping) == &cached_mapping);
   assert(mmap_count == 0 && munmap_count == 0);

   /* A different placed address must move, rather than reuse, the mapping. */
   assert(agx_bo_map_placed(&bo, &placed_mapping) == &placed_mapping);
   assert(mmap_count == 1 && munmap_count == 1);

   /* Once moved, ordinary and identical placed maps reuse the new address. */
   assert(agx_bo_map(&bo) == &placed_mapping);
   assert(agx_bo_map_placed(&bo, &placed_mapping) == &placed_mapping);
   assert(mmap_count == 1 && munmap_count == 1);

   return 0;
}
