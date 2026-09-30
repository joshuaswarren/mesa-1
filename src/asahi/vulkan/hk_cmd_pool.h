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

/* -1 until the environment has been read (hk_g17p_vdm_low_window_init()),
 * then 0 or 1.  Queried several times per draw, so the cached answer is read
 * inline. */
extern int hk_g17p_vdm_low_window_state;
int hk_g17p_vdm_low_window_init(void);

static inline bool
hk_g17p_vdm_low_window_enabled(const struct agx_device *dev)
{
   int enabled = hk_g17p_vdm_low_window_state;
   if (unlikely(enabled < 0))
      enabled = hk_g17p_vdm_low_window_init();

   return enabled && dev->chip == AGX_CHIP_G17P;
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
bool hk_g17p_upload_quarantine_enabled(const struct hk_device *dev);
void hk_g17p_quarantine_command_uploads(
   struct hk_device *dev, struct list_head *main_bos,
   struct list_head *usc_bos, struct list_head *root_bos,
   struct list_head *vdm_allocations);
VkResult hk_cmd_bo_ensure_vdm_alias(struct hk_cmd_pool *pool,
                                    struct hk_cmd_bo *bo);

VkResult hk_cmd_pool_alloc_vdm(struct hk_cmd_pool *pool, uint64_t size,
                               uint64_t alignment, struct agx_ptr *ptr,
                               struct hk_vdm_allocation **allocation);
void hk_cmd_pool_free_vdm(struct hk_cmd_pool *pool,
                          struct hk_vdm_allocation *allocation);
void hk_device_finish_vdm_slabs(struct hk_device *dev);
