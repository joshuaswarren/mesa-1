/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "hk_private.h"
#include "util/macros.h"
#include "asahi/lib/agx_device.h"
#include "util/cache_ops.h"

#include "vk_command_pool.h"

#define HK_CMD_BO_SIZE     1024 * 128
#define HK_CMD_POOL_BO_MAX 32
#define HK_G17P_CMD_POOL_MAIN_BO_MAX 256

enum hk_cmd_bo_type {
   HK_CMD_BO_MAIN,
   HK_CMD_BO_USC,
   HK_CMD_BO_ROOT,
   HK_CMD_BO_TYPE_COUNT,
};

/* Recyclable command buffer BO, used for both push buffers and upload */
struct hk_cmd_bo {
   struct agx_bo *bo;

   void *map;

   /* Full VA of the G17P VDM alias in the architectural user window. */
   uint64_t g17p_vdm_alias;

   /** Link in a matching command-pool or command-buffer BO list. */
   struct list_head link;
};

struct hk_vdm_slab;

struct hk_vdm_allocation {
   struct list_head link;
   struct hk_vdm_slab *slab;
   uint64_t addr;
   uint64_t alias;
   uint64_t size;
};

struct agx_device;
struct agx_ptr;
struct hk_device;

/* G17P command-buffer memory (main/USC pools, VDM slabs, Ioto arenas) is
 * CPU-cached; recording cleans what it wrote before submission
 * (hk_cmd_buffer_publish_cached_uploads()). */
static inline bool
hk_g17p_cached_cmd_memory(const struct agx_device *dev)
{
   return dev->chip == AGX_CHIP_G17P && util_has_cache_ops();
}

void hk_cmd_pool_clean_vdm(const struct hk_vdm_allocation *allocation);

/* A VDM record's pointer is 40 bits and hardware reads bit 39 as the UAT
 * root select, adding 0x1000000000 when it is clear.  G17P's main heap sits
 * at 0x8000000000, so every main-heap pointer a VDM record carried would have
 * bit 39 set and be fetched from the upper root instead: the dVDM_SSD
 * unmapped read at (1<<42)|0x7ffee0400 hardware reported for the first PPP
 * state record after the root chunk (0x87ffee0400).  Native keeps these
 * payloads inside the architectural 0x1000000000 window, and so does G17P
 * recording. */
static inline bool
hk_g17p_vdm_low_window_enabled(const struct agx_device *dev)
{
   return dev->chip == AGX_CHIP_G17P;
}

struct hk_cmd_pool {
   struct vk_command_pool vk;

   /** Recyclable hk_cmd_bo lists, kept separate by mapping and VA policy. */
   struct list_head free_bos[HK_CMD_BO_TYPE_COUNT];
   uint32_t num_free_bos[HK_CMD_BO_TYPE_COUNT];
};

VK_DEFINE_NONDISP_HANDLE_CASTS(hk_cmd_pool, vk.base, VkCommandPool,
                               VK_OBJECT_TYPE_COMMAND_POOL)

static inline struct hk_device *
hk_cmd_pool_device(struct hk_cmd_pool *pool)
{
   return (struct hk_device *)pool->vk.base.device;
}

VkResult hk_cmd_pool_alloc_bo(struct hk_cmd_pool *pool,
                              enum hk_cmd_bo_type type,
                              struct hk_cmd_bo **bo_out);

void hk_cmd_pool_free_bo_list(struct hk_cmd_pool *pool,
                              enum hk_cmd_bo_type type,
                              struct list_head *bos);
VkResult hk_cmd_bo_ensure_vdm_alias(struct hk_cmd_pool *pool,
                                    struct hk_cmd_bo *bo);

VkResult hk_cmd_pool_alloc_vdm(struct hk_cmd_pool *pool, uint64_t size,
                               uint64_t alignment, struct agx_ptr *ptr,
                               struct hk_vdm_allocation **allocation);
void hk_cmd_pool_free_vdm(struct hk_cmd_pool *pool,
                          struct hk_vdm_allocation *allocation);
void hk_device_finish_vdm_slabs(struct hk_device *dev);
