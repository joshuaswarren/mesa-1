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

   /* Opaque application-visible token used for BDA capture/replay. This is
    * deliberately separate from the BO's current mapping: replay buffers own
    * the fixed VA that determines their device address. */
   uint64_t opaque_capture_address;

   void *map;

   /* map is this object's own mapping of map_size bytes rather than the
    * BO's shared one: the BO may be aliased by other memory objects. */
   size_t map_size;
};


/* Application mappings belong to one VkDeviceMemory, never the shared BO. */
static inline void *
hk_memory_map_from_bo(struct agx_device *dev, struct hk_device_memory *mem,
                      void *fixed_addr, size_t length)
{
   (void)dev;
   mem->map = agx_bo_map_owned(mem->bo, fixed_addr, length, &mem->map_size);
   return mem->map;
}

static inline void
hk_memory_unmap(struct hk_device_memory *mem)
{
   if (mem->map) {
      agx_bo_unmap_owned(mem->bo, mem->map, mem->map_size);
      mem->map = NULL;
      mem->map_size = 0;
   }
}

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
