/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "hk_private.h"

#include "vk_device_memory.h"

#include "util/list.h"

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
   bool map_owned;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(hk_device_memory, vk.base, VkDeviceMemory,
                               VK_OBJECT_TYPE_DEVICE_MEMORY)

extern const VkExternalMemoryProperties hk_opaque_fd_mem_props;
extern const VkExternalMemoryProperties hk_dma_buf_mem_props;
