/* SPDX-License-Identifier: MIT */
#include "util/hash_table.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CHECK(x) do { if (!(x)) { \
   fprintf(stderr, "FAIL line %u: %s\n", __LINE__, #x); exit(1); } } while (0)

static bool fail_create, fail_insert;
static struct hash_table *
checked_create(void *ctx, uint32_t (*hash)(const void *),
               bool (*equal)(const void *, const void *))
{
   return fail_create ? NULL : _mesa_hash_table_create(ctx, hash, equal);
}

static struct hash_entry *
checked_insert(struct hash_table *index, uint32_t hash, const void *key,
               void *data)
{
   return fail_insert ? NULL :
      _mesa_hash_table_insert_pre_hashed(index, hash, key, data);
}

#define _mesa_hash_table_create checked_create
#define _mesa_hash_table_insert_pre_hashed checked_insert
#include "../hk_g17p_exact_compute_cache.h"
#undef _mesa_hash_table_create
#undef _mesa_hash_table_insert_pre_hashed

struct entry {
   struct hk_g17p_exact_compute_cache_entry cache;
   uint64_t payload;
};

static uint32_t
lookup(struct hash_table *index, struct entry *entries,
       const struct hk_g17p_exact_compute_key *key, uint32_t hash)
{
   return hk_g17p_exact_compute_cache_search(index, entries, sizeof(*entries),
                                              key, hash);
}

int main(void)
{
   enum { N = 12000, COLLISIONS = 257 };
   struct hash_table *index = NULL;
   struct entry *entries = NULL;
   unsigned capacity = 0, relocations = 0;
   struct hk_g17p_exact_compute_key base = {.words = {
      0x123, 0x456, UINT64_C(0x100000000), 0x789, 0xabc, 0xdef, 1, 1,
   }};
   CHECK(!lookup(NULL, NULL, &base, 0));

   /* New roots are a common miss pattern. Moving the entire storage at every
    * growth catches any accidentally retained entry/key pointers. */
   for (unsigned i = 0; i < N; ++i) {
      struct hk_g17p_exact_compute_key key = base;
      key.words[2] += i * 16;
      const uint32_t hash = hk_g17p_exact_compute_cache_hash(&key);
      CHECK(!lookup(index, entries, &key, hash));
      if (i == capacity) {
         const unsigned next_capacity = capacity ? capacity * 2 : 1;
         struct entry *next = malloc(next_capacity * sizeof(*next));
         CHECK(next && next != entries);
         if (capacity)
            memcpy(next, entries, i * sizeof(*entries));
         free(entries);
         entries = next;
         capacity = next_capacity;
         relocations++;
      }
      entries[i] = (struct entry){.cache.key = key, .payload = 0x12340000 + i};
      CHECK(hk_g17p_exact_compute_cache_insert(&index, &entries[i].cache,
                                                i + 1, hash));
   }

   for (unsigned i = 0; i < N; ++i) {
      struct hk_g17p_exact_compute_key key = entries[i].cache.key;
      const uint32_t hash = hk_g17p_exact_compute_cache_hash(&key);
      CHECK(lookup(index, entries, &key, hash) == i + 1);
      CHECK(entries[i].payload == 0x12340000 + i);
      /* Force hash equality so a missing equality-field cannot hide behind
       * that field's effect on the hash. Low root bit is never populated. */
      for (unsigned field = 0; field < 8; ++field) {
         key.words[field] ^= 1;
         CHECK(!lookup(index, entries, &key, hash));
         key.words[field] ^= 1;
      }
   }
   _mesa_hash_table_clear(index, NULL);
   CHECK(index->entries == 0);
   CHECK(!lookup(index, entries, &base,
                 hk_g17p_exact_compute_cache_hash(&base)));

   /* Hash zero is legal. Retain every exact key through long collisions. */
   for (unsigned i = 0; i < COLLISIONS; ++i) {
      struct hk_g17p_exact_compute_key key = base;
      key.words[2] += i;
      entries[i] = (struct entry){.cache.key = key};
      CHECK(!lookup(index, entries, &key, 0));
      CHECK(hk_g17p_exact_compute_cache_insert(&index, &entries[i].cache,
                                                i + 1, 0));
   }
   CHECK(index->entries == 1);
   for (unsigned i = 0; i < COLLISIONS; ++i)
      CHECK(lookup(index, entries, &entries[i].cache.key, 0) == i + 1);

   /* Insertion failure leaves existing chains intact. A recording reports
    * host OOM and reset can discard its unpublished candidate safely. */
   entries[COLLISIONS] = (struct entry){.cache.key = base};
   entries[COLLISIONS].cache.key.words[2] += COLLISIONS;
   fail_insert = true;
   CHECK(!hk_g17p_exact_compute_cache_insert(
      &index, &entries[COLLISIONS].cache, COLLISIONS + 1, 0));
   CHECK(!lookup(index, entries, &entries[COLLISIONS].cache.key, 0));
   for (unsigned i = 0; i < COLLISIONS; ++i)
      CHECK(lookup(index, entries, &entries[i].cache.key, 0) == i + 1);
   fail_insert = false;
   _mesa_hash_table_destroy(index, NULL);
   index = NULL;
   fail_create = true;
   CHECK(!hk_g17p_exact_compute_cache_insert(&index, &entries[0].cache, 1, 0));
   CHECK(!index);
   fail_create = false;
   entries[0] = (struct entry){0};
   CHECK(hk_g17p_exact_compute_cache_insert(&index, &entries[0].cache, 1, 0));
   CHECK(lookup(index, entries, &entries[0].cache.key, 0) == 1);
   _mesa_hash_table_destroy(index, NULL);
   free(entries);
   printf("exact compute cache: %u keys, %u relocations, all 8 fields, "
          "%u-way collisions, zero key/hash, reset and OOM passed\n",
          N, relocations, COLLISIONS);
   return 0;
}
