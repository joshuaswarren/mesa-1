/* SPDX-License-Identifier: MIT */
#include "../hk_g17p_precomp_cache.h"
#include <stdio.h>
#include <stdlib.h>

#define CHECK(x) do { if (!(x)) { \
   fprintf(stderr, "FAIL line %u: %s\n", __LINE__, #x); exit(1); } } while (0)

struct entry {
   struct hk_g17p_precomp_cache_key key;
   uint64_t payload;
};

static uint32_t
lookup(struct hash_table *index, struct entry *entries, uint32_t idx,
       const void *data, size_t size, uint32_t hash, size_t *total_probes)
{
   size_t probes, comparisons;
   uint32_t result = hk_g17p_precomp_cache_search(
      index, entries, sizeof(*entries), idx, data, size, hash,
      &probes, &comparisons);
   CHECK(comparisons <= probes);
   *total_probes += probes;
   return result;
}

int main(void)
{
   enum { N = 12000, COLLISIONS = 257 };
   uint8_t (*data)[64] = calloc(N, sizeof(*data));
   CHECK(data);
   struct hash_table *index = NULL;
   struct entry *entries = NULL;
   unsigned capacity = 0, relocations = 0;
   size_t miss_probes = 0, hit_probes = 0;
   CHECK(!lookup(NULL, NULL, 1, data[0], 64, 0, &miss_probes));

   for (unsigned i = 0; i < N; ++i) {
      for (unsigned b = 0; b < sizeof(data[i]); ++b)
         data[i][b] = (i * 29 + b * 17) & 255;
      memcpy(data[i], &i, sizeof(i));
      const uint32_t idx = 7 + i % 3;
      const size_t size = 48 - (i % 3) * 8;
      const uint32_t hash = hk_g17p_precomp_cache_hash(idx, data[i], size);
      CHECK(!lookup(index, entries, idx, data[i], size, hash, &miss_probes));
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
      entries[i] = (struct entry){
         .key = {.idx = idx, .data_size = size, .data = data[i]},
         .payload = UINT64_C(0x123400000000) + i,
      };
      CHECK(hk_g17p_precomp_cache_insert(&index, &entries[i].key, i + 1, hash));
   }

   /* Relocated storage, original insertion identity and pointer-independent
    * exact hits all survive hash-table growth. */
   for (unsigned i = 0; i < N; ++i) {
      uint8_t copy[64];
      memcpy(copy, data[i], sizeof(copy));
      const uint32_t idx = 7 + i % 3;
      const size_t size = 48 - (i % 3) * 8;
      const uint32_t hash = hk_g17p_precomp_cache_hash(idx, copy, size);
      CHECK(lookup(index, entries, idx, copy, size, hash, &hit_probes) == i + 1);
      CHECK(entries[i].payload == UINT64_C(0x123400000000) + i);
      copy[size - 1] ^= 1;
      CHECK(!lookup(index, entries, idx, copy, size, hash, &hit_probes));
      CHECK(!lookup(index, entries, idx + 99, data[i], size, hash, &hit_probes));
      CHECK(!lookup(index, entries, idx, data[i], size - 1, hash, &hit_probes));
   }
   CHECK(miss_probes < N);
   CHECK(index->entries <= N);
   _mesa_hash_table_clear(index, NULL);
   CHECK(index->entries == 0);
   CHECK(!lookup(index, entries, 7, data[0], 48,
                 hk_g17p_precomp_cache_hash(7, data[0], 48), &miss_probes));

   /* Force all keys into hash zero, including identical bytes with distinct
    * program IDs and sizes. Every collision must remain exact. */
   size_t collision_probes = 0;
   for (unsigned i = 0; i < COLLISIONS; ++i) {
      entries[i] = (struct entry){
         .key = {.idx = 21, .data_size = 64, .data = data[i]},
      };
      CHECK(!lookup(index, entries, 21, data[i], 64, 0, &collision_probes));
      CHECK(hk_g17p_precomp_cache_insert(&index, &entries[i].key, i + 1, 0));
   }
   CHECK(index->entries == 1);
   for (unsigned i = 0; i < COLLISIONS; ++i) {
      CHECK(lookup(index, entries, 21, data[i], 64, 0, &collision_probes) == i + 1);
      CHECK(!lookup(index, entries, 22, data[i], 64, 0, &collision_probes));
      CHECK(!lookup(index, entries, 21, data[i], 63, 0, &collision_probes));
   }
   _mesa_hash_table_clear(index, NULL);
   entries[0] = (struct entry){
      .key = {.idx = 0, .data_size = 0, .data = data[0]},
   };
   CHECK(hk_g17p_precomp_cache_insert(&index, &entries[0].key, 1, 0));
   CHECK(lookup(index, entries, 0, data[1], 0, 0, &collision_probes) == 1);
   _mesa_hash_table_destroy(index, NULL);
   free(entries);
   free(data);
   printf("precomp cache: %u keys, %u relocations, %zu unique-miss probes; "
          "exact hits, %u-way collisions, reset/reuse passed\n",
          N, relocations, miss_probes, COLLISIONS);
   return 0;
}
