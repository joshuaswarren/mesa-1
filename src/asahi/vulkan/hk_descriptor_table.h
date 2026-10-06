/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "hk_private.h"

#include "asahi/lib/agx_bo.h"
#include "util/simple_mtx.h"

struct hk_device;

/* A table whose full capacity exceeds one chunk reserves its GPU VA once and
 * adds backing one chunk at a time as entries are handed out, so the base
 * address never moves but the memory follows use rather than capacity.
 */
#define HK_DESCRIPTOR_TABLE_CHUNK_B    (256u * 1024u)
#define HK_DESCRIPTOR_TABLE_MAX_CHUNKS 128

struct hk_descriptor_table {
   simple_mtx_t mutex;

   uint32_t desc_size;  /**< Size of a descriptor */
   uint32_t alloc;      /**< Number of descriptors backed by memory */
   uint32_t max_alloc;  /**< Maximum possible number of descriptors */
   uint32_t next_desc;  /**< Next unallocated descriptor */
   uint32_t free_count; /**< Size of free_table */

   /** GPU address of entry 0, fixed for the lifetime of the table */
   uint64_t base_va;

   /* Single-chunk table: the whole capacity in one BO */
   struct agx_bo *bo;
   void *map;

   /* Chunked table: base_va..base_va + max_alloc * desc_size is reserved up
    * front. The unbacked tail maps the zero page, so it reads exactly like the
    * never-written slots of a fully allocated table.
    */
   struct agx_va *va;
   uint32_t chunk_shift; /**< log2(descriptors per chunk) */
   uint32_t chunk_count;
   struct agx_bo *chunks[HK_DESCRIPTOR_TABLE_MAX_CHUNKS];
   void *chunk_maps[HK_DESCRIPTOR_TABLE_MAX_CHUNKS];

   /* Stack for free descriptor elements */
   uint32_t *free_table;
};

static inline void *
hk_descriptor_table_entry(const struct hk_descriptor_table *table,
                          uint32_t index)
{
   assert(index < table->alloc);

   if (table->va) {
      const uint32_t mask = (1u << table->chunk_shift) - 1;
      return (char *)table->chunk_maps[index >> table->chunk_shift] +
             (size_t)(index & mask) * table->desc_size;
   }

   return (char *)table->map + (size_t)index * table->desc_size;
}

VkResult hk_descriptor_table_init(struct hk_device *dev,
                                  struct hk_descriptor_table *table,
                                  uint32_t descriptor_size,
                                  uint32_t min_descriptor_count,
                                  uint32_t max_descriptor_count);

void hk_descriptor_table_finish(struct hk_device *dev,
                                struct hk_descriptor_table *table);

VkResult hk_descriptor_table_add(struct hk_device *dev,
                                 struct hk_descriptor_table *table,
                                 const void *desc_data, size_t desc_size,
                                 uint32_t *index_out);

void hk_descriptor_table_remove(struct hk_device *dev,
                                struct hk_descriptor_table *table,
                                uint32_t index);
