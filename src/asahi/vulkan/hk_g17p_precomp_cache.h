/* SPDX-License-Identifier: MIT */
#ifndef HK_G17P_PRECOMP_CACHE_H
#define HK_G17P_PRECOMP_CACHE_H

#include "util/hash_table.h"
#include <string.h>

/* Prefix of each cached program. Argument storage is immutable and owned by
 * the recording. Chains use one-based indices so dynarray growth cannot
 * invalidate the index; zero denotes the end of a chain. */
struct hk_g17p_precomp_cache_key {
   uint32_t idx;
   uint32_t data_size;
   const void *data;
   uint32_t next;
};

static inline uint32_t
hk_g17p_precomp_cache_hash(uint32_t idx, const void *data, size_t data_size)
{
   const uint32_t identity[2] = {idx, (uint32_t)data_size};
   return _mesa_hash_data_with_seed(data, data_size,
                                    _mesa_hash_data(identity, sizeof(identity)));
}

/* Hashes only narrow the search: every hit still compares the entire exact
 * key. The explicit hash argument permits collision tests without changing
 * production hashing. No pointer into entries escapes a table operation. */
static inline uint32_t
hk_g17p_precomp_cache_search(struct hash_table *index, const void *entries,
                             size_t stride, uint32_t idx, const void *data,
                             size_t data_size, uint32_t hash)
{
   if (!index)
      return 0;

   struct hash_entry *bucket = _mesa_hash_table_search_pre_hashed(
      index, hash, (const void *)(uintptr_t)hash);
   uint32_t cursor = bucket ? (uintptr_t)bucket->data : 0;
   while (cursor) {
      const struct hk_g17p_precomp_cache_key *key =
         (const void *)((const uint8_t *)entries + (cursor - 1) * stride);
      if (key->idx == idx && key->data_size == data_size &&
          !memcmp(key->data, data, data_size))
         return cursor;
      cursor = key->next;
   }
   return 0;
}

static inline bool
hk_g17p_precomp_cache_insert(struct hash_table **index,
                             struct hk_g17p_precomp_cache_key *key,
                             uint32_t entry, uint32_t hash)
{
   if (!*index) {
      /* No hash callback: the caller already hashed the exact bytes once. */
      *index = _mesa_hash_table_create(NULL, NULL, _mesa_key_pointer_equal);
      if (!*index)
         return false;
   }
   struct hash_entry *bucket = _mesa_hash_table_search_pre_hashed(
      *index, hash, (const void *)(uintptr_t)hash);
   key->next = bucket ? (uintptr_t)bucket->data : 0;
   return _mesa_hash_table_insert_pre_hashed(
      *index, hash, (const void *)(uintptr_t)hash,
      (void *)(uintptr_t)entry) != NULL;
}

#endif
