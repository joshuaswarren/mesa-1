/* SPDX-License-Identifier: MIT */
#ifndef HK_G17P_EXACT_COMPUTE_CACHE_H
#define HK_G17P_EXACT_COMPUTE_CACHE_H

#include "util/hash_table.h"
#include <string.h>

/* Explicit words avoid hashing C structure padding. The eight words preserve
 * shader, linked shader, root, textures, draw params, draw ID, kind and internal
 * graphics identity. Entries are command-recording-owned and may relocate. */
struct hk_g17p_exact_compute_key {
   uint64_t words[8];
};

struct hk_g17p_exact_compute_cache_entry {
   struct hk_g17p_exact_compute_key key;
   uint32_t next;
};

static inline uint32_t
hk_g17p_exact_compute_cache_hash(const struct hk_g17p_exact_compute_key *key)
{
   return _mesa_hash_data(key->words, sizeof(key->words));
}

/* Store one-based entry indices, never pointers into a movable dynarray.
 * Hash collisions always compare all eight words before returning a hit. */
static inline uint32_t
hk_g17p_exact_compute_cache_search(
   struct hash_table *index, const void *entries, size_t stride,
   const struct hk_g17p_exact_compute_key *key, uint32_t hash)
{
   if (!index)
      return 0;

   struct hash_entry *bucket = _mesa_hash_table_search_pre_hashed(
      index, hash, (const void *)(uintptr_t)hash);
   uint32_t cursor = bucket ? (uintptr_t)bucket->data : 0;
   while (cursor) {
      const struct hk_g17p_exact_compute_cache_entry *entry =
         (const void *)((const uint8_t *)entries + (cursor - 1) * stride);
      if (!memcmp(entry->key.words, key->words, sizeof(key->words)))
         return cursor;
      cursor = entry->next;
   }
   return 0;
}

static inline bool
hk_g17p_exact_compute_cache_insert(
   struct hash_table **index, struct hk_g17p_exact_compute_cache_entry *entry,
   uint32_t position, uint32_t hash)
{
   if (!*index) {
      *index = _mesa_hash_table_create(NULL, NULL, _mesa_key_pointer_equal);
      if (!*index)
         return false;
   }
   struct hash_entry *bucket = _mesa_hash_table_search_pre_hashed(
      *index, hash, (const void *)(uintptr_t)hash);
   entry->next = bucket ? (uintptr_t)bucket->data : 0;
   return _mesa_hash_table_insert_pre_hashed(
      *index, hash, (const void *)(uintptr_t)hash,
      (void *)(uintptr_t)position) != NULL;
}

#endif
