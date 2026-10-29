/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "hk_cmd_pool.h"
#include "asahi/lib/agx_bo.h"
#include "asahi/lib/agx_device.h"
#include "asahi/lib/agx_va_runtime_hole.h"
#include "drm-uapi/asahi_drm.h"
#include "util/log.h"
#include "util/u_debug.h"

struct hk_vdm_slab {
   struct list_head link;
   struct agx_bo *bo;
   void *map;
   uint64_t alias;
   struct util_vma_heap heap;
};

#include "hk_cmd_buffer.h"
#include "hk_gen1_rt_indirect_contract.h"
#include "util/cache_ops.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_physical_device.h"

#include <string.h>

static void
hk_g17p_vdm_alias_free(struct agx_device *dev, uint64_t alias, uint64_t size);

static uint64_t
hk_g17p_vdm_alias_alloc(struct agx_device *dev, uint64_t size)
{
   simple_mtx_lock(&dev->vma_lock);
   const uint64_t alias = util_vma_heap_alloc(
      &dev->g17p_vdm_alias_heap, size, AIL_PAGESIZE);
   simple_mtx_unlock(&dev->vma_lock);

   if (agx_g17p_vdm_alias_range_valid(alias, size))
      return alias;

   if (alias)
      hk_g17p_vdm_alias_free(dev, alias, size);
   return 0;
}

static void
hk_g17p_vdm_alias_free(struct agx_device *dev, uint64_t alias, uint64_t size)
{
   simple_mtx_lock(&dev->vma_lock);
   util_vma_heap_free(&dev->g17p_vdm_alias_heap, alias, size);
   simple_mtx_unlock(&dev->vma_lock);
}

VkResult
hk_cmd_bo_ensure_vdm_alias(struct hk_cmd_pool *pool, struct hk_cmd_bo *bo)
{
   struct hk_device *dev = hk_cmd_pool_device(pool);
   /* Command pools are externally synchronized. A mapping is established
    * only when this BO first supplies a VDM pointer, then remains immutable
    * through submission and cached BO reuse. Ordinary data/compute uploads
    * consume no space in the finite VDM aperture. */
   if (bo->g17p_vdm_alias || !hk_g17p_vdm_low_window_enabled(&dev->dev))
      return VK_SUCCESS;

   const uint64_t alias = hk_g17p_vdm_alias_alloc(&dev->dev, bo->bo->size);
   const uint32_t flags = DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_WRITE;
   if (!alias || agx_bo_bind(&dev->dev, bo->bo, alias, bo->bo->size, 0, flags)) {
      if (alias)
         hk_g17p_vdm_alias_free(&dev->dev, alias, bo->bo->size);
      return vk_error(pool, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   }

   bo->g17p_vdm_alias = alias;
   return VK_SUCCESS;
}

static VkResult
hk_cmd_bo_create(struct hk_cmd_pool *pool, enum hk_cmd_bo_type type,
                 struct hk_cmd_bo **bo_out)
{
   struct hk_device *dev = hk_cmd_pool_device(pool);
   struct hk_cmd_bo *bo;
   const VkAllocationCallbacks *alloc = &pool->vk.alloc;

   bo = vk_zalloc(alloc, sizeof(*bo), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (bo == NULL)
      return vk_error(pool, VK_ERROR_OUT_OF_HOST_MEMORY);

   static const char *labels[HK_CMD_BO_TYPE_COUNT] = {
      [HK_CMD_BO_MAIN] = "Main command pool",
      [HK_CMD_BO_USC] = "USC command pool",
      [HK_CMD_BO_ROOT] = "Root descriptor pool",
   };
   const enum agx_bo_flags flags =
      (type == HK_CMD_BO_USC    ? AGX_BO_LOW_VA
       : type == HK_CMD_BO_ROOT ? AGX_BO_WRITEBACK
                                : 0) |
      (hk_g17p_cached_cmd_memory(&dev->dev) ? AGX_BO_WRITEBACK : 0);

   bo->bo = agx_bo_create(&dev->dev, HK_CMD_BO_SIZE, 0, flags, labels[type]);
   if (bo->bo == NULL) {
      vk_free(alloc, bo);
      return vk_error(pool, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   }

   bo->map = agx_bo_map(bo->bo);
   if (!bo->map) {
      agx_bo_unreference(&dev->dev, bo->bo);
      vk_free(alloc, bo);
      return vk_error(pool, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   }

   /* G17P command recording must not take first-write faults midway through a
    * draw. These BOs are recycled by the command pool, so populate and
    * initialize their shared mappings once at creation; subsequent command
    * buffers reuse already-resident pages without adding per-draw work. */
   if (agx_native_graphics(&dev->dev)) {
      memset(bo->map, 0, bo->bo->size);
      if (hk_g17p_cached_cmd_memory(&dev->dev))
         util_flush_range(bo->map, bo->bo->size);
   }

   *bo_out = bo;
   return VK_SUCCESS;
}

static void
hk_cmd_bo_destroy(struct hk_device *dev, const VkAllocationCallbacks *alloc,
                   struct hk_cmd_bo *bo)
{
   if (bo->g17p_vdm_alias) {
      /* A failed unbind may leave the old mapping installed. Keep its
       * address reserved until VM teardown rather than alias a later BO. */
      if (!agx_bo_bind(&dev->dev, NULL, bo->g17p_vdm_alias, bo->bo->size, 0,
                       DRM_ASAHI_BIND_UNBIND))
         hk_g17p_vdm_alias_free(&dev->dev, bo->g17p_vdm_alias, bo->bo->size);
   }
   agx_bo_unreference(&dev->dev, bo->bo);
   vk_free(alloc, bo);
}

static struct hk_vdm_slab *
hk_device_create_vdm_slab(struct hk_device *dev)
{
   struct hk_vdm_slab *slab =
      vk_zalloc(&dev->vk.alloc, sizeof(*slab), 8,
                VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!slab)
      return NULL;

   slab->bo = agx_bo_create(&dev->dev, HK_CMD_BO_SIZE, 0,
                            hk_g17p_cached_cmd_memory(&dev->dev)
                               ? AGX_BO_WRITEBACK : 0,
                            "VDM command slab");
   if (!slab->bo) {
      vk_free(&dev->vk.alloc, slab);
      return NULL;
   }

   /* Map before creating the GPU alias so a failed CPU mapping cannot
    * publish a slab with an invalid recording pointer. */
   slab->map = agx_bo_map(slab->bo);
   if (!slab->map) {
      agx_bo_unreference(&dev->dev, slab->bo);
      vk_free(&dev->vk.alloc, slab);
      return NULL;
   }

   slab->alias = hk_g17p_vdm_alias_alloc(&dev->dev, slab->bo->size);
   const uint32_t flags = DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_WRITE;
   if (!slab->alias || agx_bo_bind(&dev->dev, slab->bo, slab->alias,
                                   slab->bo->size, 0, flags)) {
      if (slab->alias)
         hk_g17p_vdm_alias_free(&dev->dev, slab->alias, slab->bo->size);
      agx_bo_unreference(&dev->dev, slab->bo);
      vk_free(&dev->vk.alloc, slab);
      return NULL;
   }

   util_vma_heap_init(&slab->heap, slab->bo->va->addr, slab->bo->size);
   slab->heap.alloc_high = false;
   list_addtail(&slab->link, &dev->vdm.slabs);

   return slab;
}

VkResult
hk_cmd_pool_alloc_vdm(struct hk_cmd_pool *pool, uint64_t size,
                      uint64_t alignment, struct agx_ptr *ptr,
                      struct hk_vdm_allocation **allocation)
{
   struct hk_device *dev = hk_cmd_pool_device(pool);
   if (!size || size > HK_CMD_BO_SIZE || !alignment || !ptr || !allocation)
      return vk_error(pool, VK_ERROR_OUT_OF_DEVICE_MEMORY);

   simple_mtx_lock(&dev->vdm.lock);
   uint64_t addr = 0;
   struct hk_vdm_slab *owner = NULL;
   list_for_each_entry(struct hk_vdm_slab, slab, &dev->vdm.slabs, link) {
      addr = util_vma_heap_alloc(&slab->heap, size, alignment);
      if (addr) {
         owner = slab;
         break;
      }
   }

   if (!owner) {
      owner = hk_device_create_vdm_slab(dev);
      if (!owner) {
         simple_mtx_unlock(&dev->vdm.lock);
         return vk_error(pool, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      }

      addr = util_vma_heap_alloc(&owner->heap, size, alignment);
      if (!addr) {
         simple_mtx_unlock(&dev->vdm.lock);
         return vk_error(pool, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      }
   }

   struct hk_vdm_allocation *lease = calloc(1, sizeof(*lease));
   if (!lease) {
      util_vma_heap_free(&owner->heap, addr, size);
      simple_mtx_unlock(&dev->vdm.lock);
      return vk_error(pool, VK_ERROR_OUT_OF_HOST_MEMORY);
   }

   *lease = (struct hk_vdm_allocation){
      .slab = owner,
      .addr = addr,
      .alias = owner->alias + (addr - owner->bo->va->addr),
      .size = size,
   };
   *ptr = (struct agx_ptr){
      .gpu = addr,
      .cpu = (uint8_t *)owner->map + (addr - owner->bo->va->addr),
   };
   *allocation = lease;
   simple_mtx_unlock(&dev->vdm.lock);
   return VK_SUCCESS;
}

void
hk_cmd_pool_free_vdm(struct hk_cmd_pool *pool,
                     struct hk_vdm_allocation *allocation)
{
   if (!allocation)
      return;

   struct hk_device *dev = hk_cmd_pool_device(pool);
   simple_mtx_lock(&dev->vdm.lock);
   util_vma_heap_free(&allocation->slab->heap, allocation->addr,
                      allocation->size);
   simple_mtx_unlock(&dev->vdm.lock);
   free(allocation);
}

void
hk_device_finish_vdm_slabs(struct hk_device *dev)
{
   list_for_each_entry_safe(struct hk_vdm_slab, slab, &dev->vdm.slabs, link) {
      list_del(&slab->link);
      util_vma_heap_finish(&slab->heap);
      if (!agx_bo_bind(&dev->dev, NULL, slab->alias, slab->bo->size, 0,
                       DRM_ASAHI_BIND_UNBIND))
         hk_g17p_vdm_alias_free(&dev->dev, slab->alias, slab->bo->size);
      agx_bo_unreference(&dev->dev, slab->bo);
      vk_free(&dev->vk.alloc, slab);
   }
   list_inithead(&dev->vdm.slabs);
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_CreateCommandPool(VkDevice _device,
                     const VkCommandPoolCreateInfo *pCreateInfo,
                     const VkAllocationCallbacks *pAllocator,
                     VkCommandPool *pCmdPool)
{
   VK_FROM_HANDLE(hk_device, device, _device);
   struct hk_cmd_pool *pool;

   pool = vk_zalloc2(&device->vk.alloc, pAllocator, sizeof(*pool), 8,
                     VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (pool == NULL)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   VkResult result =
      vk_command_pool_init(&device->vk, &pool->vk, pCreateInfo, pAllocator);
   if (result != VK_SUCCESS) {
      vk_free2(&device->vk.alloc, pAllocator, pool);
      return result;
   }

   for (enum hk_cmd_bo_type type = 0; type < HK_CMD_BO_TYPE_COUNT; ++type) {
      list_inithead(&pool->free_bos[type]);
      pool->num_free_bos[type] = 0;
   }

   for (unsigned i = 0; i < 2; ++i) {
      list_inithead(&pool->free_gen1_as_chunks[i]);
      pool->num_free_gen1_as_chunks[i] = 0;
   }

   *pCmdPool = hk_cmd_pool_to_handle(pool);

   return VK_SUCCESS;
}

static void
hk_cmd_pool_destroy_gen1_as_chunk(struct hk_device *dev,
                                  struct hk_gen1_as_table_chunk *chunk)
{
   agx_va_free(&dev->dev, chunk->va, true);
   agx_bo_unreference(&dev->dev, chunk->tail);
   agx_bo_unreference(&dev->dev, chunk->bank);
   vk_free(&dev->vk.alloc, chunk);
}

static struct hk_gen1_as_table_chunk *
hk_device_take_gen1_as_chunk(struct hk_device *dev, uint32_t kind)
{
   struct hk_gen1_as_table_chunk *chunk = NULL;
   simple_mtx_lock(&dev->gen1_as_chunk_cache_lock);
   if (!list_is_empty(&dev->gen1_as_free_chunks[kind])) {
      chunk = list_first_entry(&dev->gen1_as_free_chunks[kind],
                               struct hk_gen1_as_table_chunk, link);
      list_del(&chunk->link);
      --dev->gen1_as_free_chunk_count[kind];
   }
   simple_mtx_unlock(&dev->gen1_as_chunk_cache_lock);
   return chunk;
}

static void
hk_device_cache_gen1_as_chunk(struct hk_device *dev,
                               struct hk_gen1_as_table_chunk *chunk)
{
   const uint32_t kind = chunk->count == 1 ? 0 : 1;
   bool saved = false;
   simple_mtx_lock(&dev->gen1_as_chunk_cache_lock);
   if (dev->gen1_as_free_chunk_count[kind] == 0) {
      chunk->next = 0;
      list_addtail(&chunk->link, &dev->gen1_as_free_chunks[kind]);
      ++dev->gen1_as_free_chunk_count[kind];
      saved = true;
   }
   simple_mtx_unlock(&dev->gen1_as_chunk_cache_lock);
   if (!saved)
      hk_cmd_pool_destroy_gen1_as_chunk(dev, chunk);
}

void
hk_device_finish_gen1_as_chunks(struct hk_device *dev)
{
   if (!dev->gen1_as_chunk_cache_lock_ready)
      return;
   for (unsigned kind = 0; kind < 2; ++kind) {
      struct hk_gen1_as_table_chunk *chunk;
      while ((chunk = hk_device_take_gen1_as_chunk(dev, kind)))
         hk_cmd_pool_destroy_gen1_as_chunk(dev, chunk);
   }
   simple_mtx_destroy(&dev->gen1_as_chunk_cache_lock);
   dev->gen1_as_chunk_cache_lock_ready = false;
}

VkResult
hk_cmd_pool_alloc_gen1_as_chunk(struct hk_cmd_pool *pool, uint32_t count,
                               struct hk_gen1_as_table_chunk **out)
{
   struct hk_device *dev = hk_cmd_pool_device(pool);
   const uint32_t kind = count == 1 ? 0 : 1;
   assert(count == 1 || count == 4);
   if (!list_is_empty(&pool->free_gen1_as_chunks[kind])) {
      struct hk_gen1_as_table_chunk *chunk =
         list_first_entry(&pool->free_gen1_as_chunks[kind],
                          struct hk_gen1_as_table_chunk, link);
      list_del(&chunk->link);
      --pool->num_free_gen1_as_chunks[kind];
      chunk->next = 0;
      *out = chunk;
      return VK_SUCCESS;
   }
   struct hk_gen1_as_table_chunk *chunk =
      hk_device_take_gen1_as_chunk(dev, kind);
   if (chunk) {
      chunk->next = 0;
      *out = chunk;
      return VK_SUCCESS;
   }
   chunk = vk_zalloc(&dev->vk.alloc, sizeof(*chunk), 8,
                 VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!chunk)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   const uint32_t prefix = HK_GEN1_AS_TABLE_AS_SLOTS * HK_GEN1_AS_TABLE_ENTRY_BYTES;
   const uint32_t tail = HK_GEN1_RQ_PHYSICAL_SLOTS * HK_GEN1_AS_TABLE_ENTRY_BYTES;
   chunk->count = count;
   chunk->bank = dev->gen1_as_query_bank;
   agx_bo_reference(chunk->bank);
   chunk->tail = agx_bo_create(&dev->dev, count * tail, 16384,
                               0, "Gen1 cached query table tails");
   chunk->va = agx_va_alloc(&dev->dev, count * HK_GEN1_AS_TABLE_BYTES,
                            16384, 0, 0);
   if (!chunk->tail || !chunk->va || !agx_bo_map(chunk->tail) ||
       agx_va_overlaps_runtime_graph_hole(dev->dev.chip, chunk->va->addr,
                                         count * HK_GEN1_AS_TABLE_BYTES)) {
      agx_va_free(&dev->dev, chunk->va, false);
      agx_bo_unreference(&dev->dev, chunk->tail);
      agx_bo_unreference(&dev->dev, chunk->bank);
      vk_free(&dev->vk.alloc, chunk);
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   }
   struct drm_asahi_gem_bind_op ops[8];
   for (uint32_t i = 0; i < count; ++i) {
      const uint64_t address = chunk->va->addr + i * HK_GEN1_AS_TABLE_BYTES;
      ops[2 * i] = (struct drm_asahi_gem_bind_op) {
         .flags = DRM_ASAHI_BIND_READ,
         .handle = chunk->bank->uapi_handle,
         .addr = address, .range = prefix,
      };
      ops[2 * i + 1] = (struct drm_asahi_gem_bind_op) {
         .flags = DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_WRITE,
         .handle = chunk->tail->uapi_handle,
         .addr = address + prefix, .range = tail, .offset = i * tail,
      };
   }
   if (dev->dev.ops.bo_bind(&dev->dev, ops, 2 * count)) {
      hk_cmd_pool_destroy_gen1_as_chunk(hk_cmd_pool_device(pool), chunk);
      return VK_ERROR_UNKNOWN;
   }
   *out = chunk;
   return VK_SUCCESS;
}

void
hk_cmd_pool_free_gen1_as_chunks(struct hk_cmd_pool *pool,
                               struct list_head *chunks)
{
   list_for_each_entry_safe(struct hk_gen1_as_table_chunk, chunk, chunks, link) {
      list_del(&chunk->link);
      const uint32_t kind = chunk->count == 1 ? 0 : 1;
      if (pool->num_free_gen1_as_chunks[kind] == 4) {
         hk_cmd_pool_destroy_gen1_as_chunk(hk_cmd_pool_device(pool), chunk);
      } else {
         chunk->next = 0;
         list_addtail(&chunk->link, &pool->free_gen1_as_chunks[kind]);
         ++pool->num_free_gen1_as_chunks[kind];
      }
   }
}

static void
hk_cmd_pool_destroy_bos(struct hk_cmd_pool *pool)
{
   struct hk_device *dev = hk_cmd_pool_device(pool);
   for (unsigned kind = 0; kind < 2; ++kind) {
      list_for_each_entry_safe(struct hk_gen1_as_table_chunk, chunk,
                               &pool->free_gen1_as_chunks[kind], link) {
         list_del(&chunk->link);
         hk_device_cache_gen1_as_chunk(dev, chunk);
      }
      pool->num_free_gen1_as_chunks[kind] = 0;
   }
   for (enum hk_cmd_bo_type type = 0; type < HK_CMD_BO_TYPE_COUNT; ++type) {
      list_for_each_entry_safe(struct hk_cmd_bo, bo, &pool->free_bos[type],
                               link)
         hk_cmd_bo_destroy(dev, &pool->vk.alloc, bo);

      list_inithead(&pool->free_bos[type]);
      pool->num_free_bos[type] = 0;
   }
}

VkResult
hk_cmd_pool_alloc_bo(struct hk_cmd_pool *pool, enum hk_cmd_bo_type type,
                     struct hk_cmd_bo **bo_out)
{
   struct hk_cmd_bo *bo = NULL;
   if (!list_is_empty(&pool->free_bos[type])) {
      bo = list_first_entry(&pool->free_bos[type], struct hk_cmd_bo, link);
      pool->num_free_bos[type]--;
   }

   if (bo) {
      list_del(&bo->link);
      *bo_out = bo;
      return VK_SUCCESS;
   }

   return hk_cmd_bo_create(pool, type, bo_out);
}

void
hk_cmd_pool_free_bo_list(struct hk_cmd_pool *pool, enum hk_cmd_bo_type type,
                         struct list_head *bos)
{
   const struct hk_device *dev = hk_cmd_pool_device(pool);
   const unsigned max_free =
      (agx_native_graphics(&dev->dev) || agx_chip_g17_m5(dev->dev.chip)) &&
         type == HK_CMD_BO_MAIN
         ? HK_G17P_CMD_POOL_MAIN_BO_MAX
         : HK_CMD_POOL_BO_MAX;

   list_for_each_entry_safe(struct hk_cmd_bo, bo, bos, link) {
      list_del(&bo->link);
      if (pool->num_free_bos[type] >= max_free) {
         hk_cmd_bo_destroy(hk_cmd_pool_device(pool),
                           &pool->vk.alloc, bo);
      } else {
         list_addtail(&bo->link, &pool->free_bos[type]);
         pool->num_free_bos[type]++;
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_DestroyCommandPool(VkDevice _device, VkCommandPool commandPool,
                      const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(hk_device, device, _device);
   VK_FROM_HANDLE(hk_cmd_pool, pool, commandPool);

   if (!pool)
      return;

   vk_command_pool_finish(&pool->vk);
   hk_cmd_pool_destroy_bos(pool);
   vk_free2(&device->vk.alloc, pAllocator, pool);
}

VKAPI_ATTR void VKAPI_CALL
hk_TrimCommandPool(VkDevice device, VkCommandPool commandPool,
                   VkCommandPoolTrimFlags flags)
{
   VK_FROM_HANDLE(hk_cmd_pool, pool, commandPool);

   vk_command_pool_trim(&pool->vk, flags);
   hk_cmd_pool_destroy_bos(pool);
}

void
hk_cmd_pool_clean_vdm(const struct hk_vdm_allocation *allocation)
{
   const struct hk_vdm_slab *slab = allocation->slab;
   util_flush_range_no_fence(
      (uint8_t *)slab->map + (allocation->addr - slab->bo->va->addr),
      allocation->size);
}
