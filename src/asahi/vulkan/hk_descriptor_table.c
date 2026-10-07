/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "hk_descriptor_table.h"

#include "hk_device.h"
#include "hk_physical_device.h"

#include "asahi/lib/agx_bo.h"
#include "asahi/lib/agx_device.h"
#include "util/u_math.h"

static VkResult
hk_descriptor_table_resize_free_table(struct hk_device *dev,
                                      struct hk_descriptor_table *table,
                                      uint32_t new_alloc)
{
   const size_t new_free_table_size = new_alloc * sizeof(uint32_t);
   uint32_t *new_free_table =
      vk_realloc(&dev->vk.alloc, table->free_table, new_free_table_size, 4,
                 VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (new_free_table == NULL) {
      return vk_errorf(dev, VK_ERROR_OUT_OF_HOST_MEMORY,
                       "Failed to allocate image descriptor free table");
   }

   table->free_table = new_free_table;
   return VK_SUCCESS;
}

/* Every slot is read exactly as it came from the allocator until
 * hk_descriptor_table_add() writes it. A fresh kernel BO is zeroed -- entry
 * 0's reserved null texture semantics depend on that -- but a BO recycled from
 * the userspace cache is not, and a stale slot's 44-bit address field is
 * dereferenced verbatim by an out-of-range heap index. Hence ZERO_ON_REUSE.
 */
static VkResult
hk_descriptor_table_alloc_single(struct hk_device *dev,
                                 struct hk_descriptor_table *table)
{
   const uint32_t new_alloc = table->max_alloc;
   const uint32_t new_bo_size = new_alloc * table->desc_size;

   struct agx_bo *new_bo = agx_bo_create(&dev->dev, new_bo_size, 0,
                                         AGX_BO_ZERO_ON_REUSE,
                                         "Descriptor table");
   if (new_bo == NULL) {
      return vk_errorf(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                       "Failed to allocate the descriptor table");
   }

   void *map = agx_bo_map(new_bo);
   if (map == NULL) {
      agx_bo_unreference(&dev->dev, new_bo);
      return vk_errorf(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                       "Failed to map the descriptor table");
   }

   VkResult result =
      hk_descriptor_table_resize_free_table(dev, table, new_alloc);
   if (result != VK_SUCCESS) {
      agx_bo_unreference(&dev->dev, new_bo);
      return result;
   }

   /* Publish only a fully initialized table. Init's failure cleanup must not
    * release the local BO a second time. */
   table->bo = new_bo;
   table->map = map;
   table->base_va = new_bo->va->addr;
   table->alloc = new_alloc;
   return VK_SUCCESS;
}

static VkResult
hk_descriptor_table_add_chunk_locked(struct hk_device *dev,
                                     struct hk_descriptor_table *table)
{
   const uint32_t chunk_entries = 1u << table->chunk_shift;
   const uint32_t chunk_B = chunk_entries * table->desc_size;

   assert(table->alloc == table->chunk_count * chunk_entries);
   assert(table->alloc + chunk_entries <= table->max_alloc);
   assert(table->chunk_count < HK_DESCRIPTOR_TABLE_MAX_CHUNKS);

   VkResult result = hk_descriptor_table_resize_free_table(
      dev, table, table->alloc + chunk_entries);
   if (result != VK_SUCCESS)
      return result;

   struct agx_bo *bo = agx_bo_create(&dev->dev, chunk_B, 0,
                                     AGX_BO_ZERO_ON_REUSE, "Descriptor table");
   if (bo == NULL) {
      return vk_errorf(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                       "Failed to allocate a descriptor table chunk");
   }

   void *map = agx_bo_map(bo);
   const uint64_t addr =
      table->base_va + (uint64_t)table->chunk_count * chunk_B;

   /* Replaces the zero-page mapping of this range. Entries in it have never
    * been handed out, so in-flight work cannot observe the switch: both the
    * zero page and the new chunk read as zero.
    */
   if (map == NULL ||
       agx_bo_bind(&dev->dev, bo, addr, chunk_B, 0,
                   DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_WRITE)) {
      agx_bo_unreference(&dev->dev, bo);
      return vk_errorf(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                       "Failed to bind a descriptor table chunk");
   }

   table->chunks[table->chunk_count] = bo;
   table->chunk_maps[table->chunk_count] = map;
   table->chunk_count++;
   table->alloc += chunk_entries;
   return VK_SUCCESS;
}

static VkResult
hk_descriptor_table_reserve(struct hk_device *dev,
                            struct hk_descriptor_table *table)
{
   const uint64_t size_B = (uint64_t)table->max_alloc * table->desc_size;

   table->va = agx_va_alloc(&dev->dev, size_B, AIL_PAGESIZE, 0, 0);
   if (table->va == NULL) {
      return vk_errorf(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                       "Failed to reserve the descriptor table");
   }

   if (agx_bo_bind(&dev->dev, dev->dev.zero_bo, table->va->addr, size_B, 0,
                   DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_SINGLE_PAGE)) {
      agx_va_free(&dev->dev, table->va, false);
      table->va = NULL;
      return vk_errorf(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                       "Failed to bind the descriptor table reservation");
   }

   table->base_va = table->va->addr;
   return VK_SUCCESS;
}

VkResult
hk_descriptor_table_init(struct hk_device *dev,
                         struct hk_descriptor_table *table,
                         uint32_t descriptor_size,
                         uint32_t min_descriptor_count,
                         uint32_t max_descriptor_count)
{
   memset(table, 0, sizeof(*table));
   VkResult result;

   simple_mtx_init(&table->mutex, mtx_plain);

   assert(util_is_power_of_two_nonzero(min_descriptor_count));
   assert(util_is_power_of_two_nonzero(max_descriptor_count));
   assert(min_descriptor_count <= max_descriptor_count);

   table->desc_size = descriptor_size;
   table->alloc = 0;
   table->max_alloc = max_descriptor_count;
   table->next_desc = 0;
   table->free_count = 0;

   const uint64_t max_B = (uint64_t)max_descriptor_count * descriptor_size;

   if (max_B <= HK_DESCRIPTOR_TABLE_CHUNK_B ||
       !util_is_power_of_two_nonzero(descriptor_size)) {
      result = hk_descriptor_table_alloc_single(dev, table);
   } else {
      table->chunk_shift =
         util_logbase2(HK_DESCRIPTOR_TABLE_CHUNK_B / descriptor_size);
      assert((max_descriptor_count >> table->chunk_shift) <=
             HK_DESCRIPTOR_TABLE_MAX_CHUNKS);

      result = hk_descriptor_table_reserve(dev, table);
      while (result == VK_SUCCESS && table->alloc < min_descriptor_count)
         result = hk_descriptor_table_add_chunk_locked(dev, table);
   }

   if (result != VK_SUCCESS) {
      hk_descriptor_table_finish(dev, table);
      return result;
   }

   return VK_SUCCESS;
}

void
hk_descriptor_table_finish(struct hk_device *dev,
                           struct hk_descriptor_table *table)
{
   if (table->va) {
      const uint64_t size_B = (uint64_t)table->max_alloc * table->desc_size;
      const bool unbound =
         !agx_bo_bind(&dev->dev, NULL, table->va->addr, size_B, 0,
                      DRM_ASAHI_BIND_UNBIND);

      /* Never recycle the address while a stale mapping could remain. */
      if (unbound)
         agx_va_free(&dev->dev, table->va, false);
   }

   for (uint32_t i = 0; i < table->chunk_count; ++i)
      agx_bo_unreference(&dev->dev, table->chunks[i]);

   agx_bo_unreference(&dev->dev, table->bo);
   vk_free(&dev->vk.alloc, table->free_table);
   simple_mtx_destroy(&table->mutex);
}

#define HK_IMAGE_DESC_INVALID

static VkResult
hk_descriptor_table_alloc_locked(struct hk_device *dev,
                                 struct hk_descriptor_table *table,
                                 uint32_t *index_out)
{
   VkResult result;

   if (table->free_count > 0) {
      *index_out = table->free_table[--table->free_count];
      return VK_SUCCESS;
   }

   if (table->next_desc < table->alloc) {
      *index_out = table->next_desc++;
      return VK_SUCCESS;
   }

   if (table->next_desc >= table->max_alloc) {
      return vk_errorf(dev, VK_ERROR_OUT_OF_HOST_MEMORY,
                       "Descriptor table not large enough");
   }

   /* A single-chunk table is allocated at full capacity */
   assert(table->va != NULL);
   result = hk_descriptor_table_add_chunk_locked(dev, table);
   if (result != VK_SUCCESS)
      return result;

   assert(table->next_desc < table->alloc);
   *index_out = table->next_desc++;

   return VK_SUCCESS;
}

static VkResult
hk_descriptor_table_add_locked(struct hk_device *dev,
                               struct hk_descriptor_table *table,
                               const void *desc_data, size_t desc_size,
                               uint32_t *index_out)
{
   VkResult result = hk_descriptor_table_alloc_locked(dev, table, index_out);
   if (result != VK_SUCCESS)
      return result;

   void *map = hk_descriptor_table_entry(table, *index_out);

   assert(desc_size == table->desc_size);
   memcpy(map, desc_data, table->desc_size);

   return VK_SUCCESS;
}

VkResult
hk_descriptor_table_add(struct hk_device *dev,
                        struct hk_descriptor_table *table,
                        const void *desc_data, size_t desc_size,
                        uint32_t *index_out)
{
   simple_mtx_lock(&table->mutex);
   VkResult result = hk_descriptor_table_add_locked(dev, table, desc_data,
                                                    desc_size, index_out);
   simple_mtx_unlock(&table->mutex);

   return result;
}

void
hk_descriptor_table_remove(struct hk_device *dev,
                           struct hk_descriptor_table *table, uint32_t index)
{
   simple_mtx_lock(&table->mutex);

   memset(hk_descriptor_table_entry(table, index), 0, table->desc_size);

   /* Sanity check for double-free */
   assert(table->free_count < table->alloc);
   for (uint32_t i = 0; i < table->free_count; i++)
      assert(table->free_table[i] != index);

   table->free_table[table->free_count++] = index;

   simple_mtx_unlock(&table->mutex);
}
