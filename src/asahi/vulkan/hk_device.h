/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "asahi/lib/agx_device.h"
#include "util/rwlock.h"
#include "util/simple_mtx.h"
#include "util/u_dynarray.h"
#include "agx_bg_eot.h"
#include "agx_pack.h"
#include "agx_scratch.h"
#include "decode.h"
#include "vk_cmd_queue.h"
#include "vk_dispatch_table.h"

#include "hk_private.h"

#include "hk_descriptor_table.h"
#include "hk_native_sampler.h"
#include "hk_queue.h"
#include "vk_device.h"
#include "vk_acceleration_structure.h"
#include "vk_meta.h"
#include "vk_queue.h"

struct hk_physical_device;
struct vk_pipeline_cache;

typedef void (*hk_internal_builder_t)(struct nir_builder *b, const void *key);

struct hk_internal_key {
   hk_internal_builder_t builder;
   size_t key_size;
   uint8_t key[];
};

struct hk_internal_shaders {
   simple_mtx_t lock;
   struct hash_table *ht;
};

struct hk_rc_sampler {
   struct agx_sampler_packed key;

   /* Reference count for this hardware sampler, protected by the heap mutex */
   uint16_t refcount;

   /* Index of this hardware sampler in the hardware sampler heap */
   uint16_t index;
};

struct hk_sampler_heap {
   simple_mtx_t lock;

   struct hk_descriptor_table table;

   /* Map of agx_sampler_packed to hk_rc_sampler */
   struct hash_table *ht;
};

struct hk_device {
   struct vk_device vk;
   struct agx_device dev;
   struct agxdecode_ctx *decode_ctx;

   struct hk_descriptor_table occlusion_queries;
   struct hk_sampler_heap samplers;

   /* Native sampled-image indices are stable for an image-view lifetime.
    * G17P shaders dereference this table on the GPU, including descriptors
    * updated after command recording. Other generations keep inline state.
    */
   struct hk_descriptor_table sampled_images;

   struct vk_meta_device meta;

   /* VK_KHR_acceleration_structure.  radix_sort is created on first use
    * behind radix_sort_lock, because building it compiles a dozen compute
    * pipelines that most devices never need. */
   struct vk_acceleration_structure_build_args accel_struct_args;
   struct radix_sort_vk *radix_sort;
   bool radix_sort_failed;
   simple_mtx_t radix_sort_lock;
   struct agx_bg_eot_cache bg_eot;

   struct {
      struct agx_bo *bo;
      uint64_t heap;
   } rodata;

   struct hk_internal_shaders prolog_epilog;
   struct hk_internal_shaders kernels;
   struct hk_api_shader *null_fs;

   /* Device-wide compact VDM backing. Command buffers from different pools
    * suballocate immutable spans from the same finite G17P aperture. */
   struct {
      simple_mtx_t lock;
      struct list_head slabs;

      /* Opt-in command-upload lifetime falsifier. These lists retain complete
       * upload BOs and VDM slab leases after command-buffer reset, preventing
       * either address from being recycled before device destruction. */
      struct list_head quarantined_bos;
      struct list_head quarantined_vdm_allocations;
   } vdm;

   /* Native G17P indirect compute grid helper.  A compact indirect helper
    * record carries no dimensions and no resource table, so the only way a
    * runtime grid can reach the main record is for this program to publish
    * the six-u32 geometry block.  Compiled once, from NIR, through the same
    * path as any other Honeykrisp compute shader, so it uses the driver's own
    * proven root/uniform ABI rather than a replayed one.  NULL on every other
    * chip, and NULL on G17P if the backend refused it -- in which case
    * indirect dispatch stays refused instead of launching a program that does
    * not exist. */
   struct hk_api_shader *g17p_grid_helper;

   /* Indirected for common secondary emulation */
   struct vk_device_dispatch_table cmd_dispatch;

   /* Heap used for GPU-side memory allocation for geometry/tessellation.
    *
    * Control streams accessing the heap must be serialized. This is not
    * expected to be a legitimate problem. If it is, we can rework later.
    */
   struct agx_bo *heap;
   util_once_flag heap_init_once;

   struct {
      struct agx_scratch vs, fs, cs;
      simple_mtx_t lock;
   } scratch;

   uint32_t perftest;

   struct {
      struct u_rwlock lock;
      struct util_dynarray list;
      struct util_dynarray counts;
   } external_bos;
};

VK_DEFINE_HANDLE_CASTS(hk_device, vk.base, VkDevice, VK_OBJECT_TYPE_DEVICE)

enum hk_perftest {
   HK_PERF_NOTESS = BITFIELD_BIT(0),
   HK_PERF_NOBORDER = BITFIELD_BIT(1),
   HK_PERF_NOBARRIER = BITFIELD_BIT(2),
   HK_PERF_BATCH = BITFIELD_BIT(3),
   HK_PERF_NOROBUST = BITFIELD_BIT(4),
};

#define HK_PERF(dev, flag) unlikely((dev)->perftest &HK_PERF_##flag)

static inline struct hk_physical_device *
hk_device_physical(struct hk_device *dev)
{
   return (struct hk_physical_device *)dev->vk.physical;
}

VkResult hk_device_init_meta(struct hk_device *dev);
void hk_device_finish_meta(struct hk_device *dev);

VkResult hk_sampler_heap_add(struct hk_device *dev,
                             const struct hk_native_sampler_descriptor *desc,
                             struct hk_rc_sampler **out);

void hk_sampler_heap_remove(struct hk_device *dev, struct hk_rc_sampler *rc);

static inline struct agx_scratch *
hk_device_scratch_locked(struct hk_device *dev, mesa_shader_stage stage)
{
   simple_mtx_assert_locked(&dev->scratch.lock);

   switch (stage) {
   case MESA_SHADER_FRAGMENT:
      return &dev->scratch.fs;
   case MESA_SHADER_VERTEX:
      return &dev->scratch.vs;
   default:
      return &dev->scratch.cs;
   }
}

static inline void
hk_device_alloc_scratch(struct hk_device *dev, mesa_shader_stage stage,
                        unsigned size)
{
   simple_mtx_lock(&dev->scratch.lock);
   agx_scratch_alloc(hk_device_scratch_locked(dev, stage), size, 0);
   simple_mtx_unlock(&dev->scratch.lock);
}
