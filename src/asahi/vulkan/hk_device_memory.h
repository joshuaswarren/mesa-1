/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "hk_private.h"

#include "asahi/lib/agx_device.h"

#include "vk_device_memory.h"

#include "util/list.h"

#include <sys/mman.h>

struct hk_device;
struct hk_image_plane;

struct hk_device_memory {
   struct vk_device_memory vk;

   struct agx_bo *bo;

   /* Application mapping owned only by this memory object. */
   void *map;

   /* Length of the live mapping in map, valid when map is not NULL. */
   size_t map_size;

};

/* Map this memory object's own application mapping from the BO: a placed
 * mapping covers exactly `size` bytes at `fixed_addr`, an ordinary mapping
 * covers the whole BO. The mapping is never stored on the BO, so memory
 * objects sharing an imported BO map independently. Returns the mapping, or
 * NULL when the primitive failed. */
static inline void *
hk_memory_map_from_bo(struct agx_device *dev, struct hk_device_memory *mem,
                      void *fixed_addr, size_t size)
{
   mem->map = dev->ops.bo_mmap(dev, mem->bo, fixed_addr, size);
   if (mem->map)
      mem->map_size = size;

   return mem->map;
}

/* Drop this memory object's own application mapping. Other memory objects
 * sharing the BO keep theirs, and the BO's stable internal mapping is never
 * touched. A never-mapped object is a no-op. */
static inline void
hk_memory_unmap(struct hk_device_memory *mem)
{
   if (mem->map) {
      munmap(mem->map, mem->map_size);
      mem->map = NULL;
      mem->map_size = 0;
   }
}

/* VK_MEMORY_UNMAP_RESERVE_BIT_EXT: overlay this memory object's own mapping
 * with PROT_NONE instead of releasing the address. Returns false when the
 * overlay fails, leaving the mapping recorded. */
static inline bool
hk_memory_unmap_reserve(struct hk_device_memory *mem)
{
   if (!mem->map)
      return true;

   void *err = mmap(mem->map, mem->map_size, PROT_NONE,
                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
   if (err == MAP_FAILED)
      return false;

   mem->map = NULL;
   mem->map_size = 0;
   return true;
}

VK_DEFINE_NONDISP_HANDLE_CASTS(hk_device_memory, vk.base, VkDeviceMemory,
                               VK_OBJECT_TYPE_DEVICE_MEMORY)

extern const VkExternalMemoryProperties hk_opaque_fd_mem_props;
extern const VkExternalMemoryProperties hk_dma_buf_mem_props;
