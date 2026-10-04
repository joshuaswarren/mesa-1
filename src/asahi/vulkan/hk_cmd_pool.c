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
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_physical_device.h"

#include <string.h>

int hk_g17p_vdm_low_window_state = -1;

int
hk_g17p_vdm_low_window_init(void)
{
   /* On by default: a VDM record's pointer is 40 bits and hardware reads bit
    * 39 as the UAT root select, adding 0x1000000000 when it is clear. G17P's
    * main heap sits at 0x8000000000, so every main-heap pointer a VDM record
    * carries has bit 39 set and is fetched from the upper root instead. That
    * is exactly the dVDM_SSD unmapped read at (1<<42)|0x7ffee0400 hardware
    * reported for the first PPP state record after the root chunk
    * (0x87ffee0400). Native keeps these payloads inside the architectural
    * 0x1000000000 window, which is what this path reproduces. The environment
    * variable now only exists to turn the correct behaviour back off for A/B
    * testing. */
   const int enabled =
      debug_get_bool_option("ASAHI_G17P_VDM_LOW_WINDOW", true) ? 1 : 0;
   hk_g17p_vdm_low_window_state = enabled;
   return enabled;
}

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
   if (hk_g17p_verbose_enabled())
      fprintf(stderr,
           "asahi: G17P VDM command BO full=%#llx alias=%#llx low=%#llx size=%#llx\n",
           (unsigned long long)bo->bo->va->addr,
           (unsigned long long)alias,
           (unsigned long long)(alias - AGX_G17P_VDM_USER_BASE),
           (unsigned long long)bo->bo->size);
   return VK_SUCCESS;
}

/* Ordinary command BO wrappers are owned by their command pool.  The
 * default-off upload quarantine deliberately transfers them to the device
 * until destruction, so choose device-lifetime allocation from the start in
 * that mode.  A quarantined wrapper must not retain a pool allocator whose
 * pUserData may cease to exist when the pool is destroyed.
 */
static const VkAllocationCallbacks *
hk_cmd_bo_allocator(struct hk_cmd_pool *pool)
{
   struct hk_device *dev = hk_cmd_pool_device(pool);
   return hk_g17p_upload_quarantine_enabled(dev) ? &dev->vk.alloc
                                                : &pool->vk.alloc;
}

static VkResult
hk_cmd_bo_create(struct hk_cmd_pool *pool, enum hk_cmd_bo_type type,
                 struct hk_cmd_bo **bo_out)
{
   struct hk_device *dev = hk_cmd_pool_device(pool);
   struct hk_cmd_bo *bo;
   const VkAllocationCallbacks *alloc = hk_cmd_bo_allocator(pool);

   bo = vk_zalloc(alloc, sizeof(*bo), 8,
                  alloc == &dev->vk.alloc ? VK_SYSTEM_ALLOCATION_SCOPE_DEVICE
                                         : VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (bo == NULL)
      return vk_error(pool, VK_ERROR_OUT_OF_HOST_MEMORY);

   static const char *labels[HK_CMD_BO_TYPE_COUNT] = {
      [HK_CMD_BO_MAIN] = "Main command pool",
      [HK_CMD_BO_USC] = "USC command pool",
      [HK_CMD_BO_ROOT] = "Root descriptor pool",
   };
   const enum agx_bo_flags flags =
      type == HK_CMD_BO_USC    ? AGX_BO_LOW_VA
      : type == HK_CMD_BO_ROOT ? AGX_BO_WRITEBACK
                               : 0;

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
   if (dev->dev.chip == AGX_CHIP_G17P)
      memset(bo->map, 0, bo->bo->size);

   *bo_out = bo;
   return VK_SUCCESS;
}

static void
hk_cmd_bo_destroy(struct hk_device *dev, const VkAllocationCallbacks *alloc,
                   struct hk_cmd_bo *bo)
{
   if (bo->g17p_vdm_alias) {
      agx_bo_bind(&dev->dev, NULL, bo->g17p_vdm_alias, bo->bo->size, 0,
                  DRM_ASAHI_BIND_UNBIND);
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

   slab->bo = agx_bo_create(&dev->dev, HK_CMD_BO_SIZE, 0, 0,
                            "VDM command slab");
   if (!slab->bo) {
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

   slab->map = agx_bo_map(slab->bo);
   util_vma_heap_init(&slab->heap, slab->bo->va->addr, slab->bo->size);
   slab->heap.alloc_high = false;
   list_addtail(&slab->link, &dev->vdm.slabs);

   if (hk_g17p_verbose_enabled())
      fprintf(stderr,
              "asahi: G17P VDM command BO slab full=%#llx alias=%#llx low=%#llx "
              "size=%#llx\n",
              (unsigned long long)slab->bo->va->addr,
              (unsigned long long)slab->alias,
              (unsigned long long)(slab->alias - AGX_G17P_VDM_USER_BASE),
              (unsigned long long)slab->bo->size);
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

bool
hk_g17p_upload_quarantine_enabled(const struct hk_device *dev)
{
   static int enabled = -1;
   if (enabled < 0)
      enabled = HK_ENV_BOOL_ONCE("HK_G17P_UPLOAD_QUARANTINE", false);

   return dev->dev.chip == AGX_CHIP_G17P && enabled;
}

void
hk_g17p_quarantine_command_uploads(
   struct hk_device *dev, struct list_head *main_bos,
   struct list_head *usc_bos, struct list_head *root_bos,
   struct list_head *vdm_allocations)
{
   assert(hk_g17p_upload_quarantine_enabled(dev));

   const unsigned bo_count = list_length(main_bos) + list_length(usc_bos) +
                             list_length(root_bos);
   const unsigned vdm_count = list_length(vdm_allocations);

   simple_mtx_lock(&dev->vdm.lock);
   list_splicetail(main_bos, &dev->vdm.quarantined_bos);
   list_inithead(main_bos);
   list_splicetail(usc_bos, &dev->vdm.quarantined_bos);
   list_inithead(usc_bos);
   list_splicetail(root_bos, &dev->vdm.quarantined_bos);
   list_inithead(root_bos);
   list_splicetail(vdm_allocations,
                   &dev->vdm.quarantined_vdm_allocations);
   list_inithead(vdm_allocations);
   simple_mtx_unlock(&dev->vdm.lock);

   if (bo_count || vdm_count)
      mesa_logi_once("G17P command-upload quarantine active; retained until "
                     "device destruction");
}

void
hk_device_finish_vdm_slabs(struct hk_device *dev)
{
   util_dynarray_foreach(&dev->vdm.quarantined_ioto_bos, struct agx_bo *, bo)
      agx_bo_unreference(&dev->dev, *bo);
   util_dynarray_fini(&dev->vdm.quarantined_ioto_bos);

   list_for_each_entry_safe(struct hk_cmd_bo, bo,
                            &dev->vdm.quarantined_bos, link) {
      list_del(&bo->link);
      hk_cmd_bo_destroy(dev, &dev->vk.alloc, bo);
   }
   list_inithead(&dev->vdm.quarantined_bos);

   list_for_each_entry_safe(struct hk_vdm_allocation, allocation,
                            &dev->vdm.quarantined_vdm_allocations, link) {
      list_del(&allocation->link);
      free(allocation);
   }
   list_inithead(&dev->vdm.quarantined_vdm_allocations);

   list_for_each_entry_safe(struct hk_vdm_slab, slab, &dev->vdm.slabs, link) {
      list_del(&slab->link);
      util_vma_heap_finish(&slab->heap);
      agx_bo_bind(&dev->dev, NULL, slab->alias, slab->bo->size, 0,
                  DRM_ASAHI_BIND_UNBIND);
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

   *pCmdPool = hk_cmd_pool_to_handle(pool);

   return VK_SUCCESS;
}

static void
hk_cmd_pool_destroy_bos(struct hk_cmd_pool *pool)
{
   struct hk_device *dev = hk_cmd_pool_device(pool);
   for (enum hk_cmd_bo_type type = 0; type < HK_CMD_BO_TYPE_COUNT; ++type) {
      list_for_each_entry_safe(struct hk_cmd_bo, bo, &pool->free_bos[type],
                               link)
         hk_cmd_bo_destroy(dev, hk_cmd_bo_allocator(pool), bo);

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
      dev->dev.chip == AGX_CHIP_G17P && type == HK_CMD_BO_MAIN
         ? HK_G17P_CMD_POOL_MAIN_BO_MAX
         : HK_CMD_POOL_BO_MAX;

   list_for_each_entry_safe(struct hk_cmd_bo, bo, bos, link) {
      list_del(&bo->link);
      if (pool->num_free_bos[type] >= max_free) {
         hk_cmd_bo_destroy(hk_cmd_pool_device(pool),
                           hk_cmd_bo_allocator(pool), bo);
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
