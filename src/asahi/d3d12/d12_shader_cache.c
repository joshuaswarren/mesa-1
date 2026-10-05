/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Device-level shader caches for pipeline-state creation.
 *
 * Games create pipeline states by the thousand, from many threads, and the
 * driver-generated parts of a pipeline repeat far more than the application
 * shaders do: every vertex pipeline links the same static_vi prolog, and
 * the fragment epilog is a function of a handful of blend/format bits.
 * Honeykrisp keeps those in hk_device::prolog_epilog, a hash table keyed on
 * the builder function and its key bytes (hk_get_prolog_epilog); this is
 * the same table for the D3D12 front end.
 *
 * Locking: the table lock is held only to look up and to insert. Compiles
 * run outside it so one thread's backend compile does not serialise every
 * other thread's pipeline creation; a thread that loses the race adopts the
 * entry the winner inserted and discards its own.
 */

#include "d12_private.h"
#include "agx_compile.h"
#include "agx_nir.h"
#include "nir.h"
#include "nir_builder.h"
#include "util/hash_table.h"
#include "util/ralloc.h"

/* Prolog/epilog keys are small structs (agx_vs_prolog_key is the largest);
 * bound them so the lookup key can live on the stack. */
#define D12_PART_KEY_MAX 1024

struct d12_part_key {
   d12_part_builder_t builder;
   uint32_t key_size;
   /* The hash covers sizeof(struct d12_part_key) + key_size bytes from the
    * start of the struct, which is wider than the members above plus the
    * key data: every producer zeroes the whole allocation first so no
    * padding byte reaches the hash uninitialised (hk_internal_key). */
   uint8_t key[];
};

static uint32_t
d12_part_key_hash(const void *data)
{
   const struct d12_part_key *key = data;
   return _mesa_hash_data(key, sizeof(*key) + key->key_size);
}

static bool
d12_part_key_equal(const void *a_, const void *b_)
{
   const struct d12_part_key *a = a_, *b = b_;
   return a->builder == b->builder && a->key_size == b->key_size &&
          !memcmp(a->key, b->key, a->key_size);
}

void
d12_shader_cache_init(struct d12_shader_cache *cache)
{
   simple_mtx_init(&cache->lock, mtx_plain);
   cache->parts =
      _mesa_hash_table_create(NULL, d12_part_key_hash, d12_part_key_equal);
}

void
d12_shader_cache_finish(struct d12_device *device)
{
   struct d12_shader_cache *cache = &device->shader_cache;

   if (cache->parts) {
      hash_table_foreach(cache->parts, ent) {
         struct agx_shader_part *part = ent->data;
         free(part->binary);
      }
      /* Keys and parts are ralloc children of the table. */
      _mesa_hash_table_destroy(cache->parts, NULL);
      cache->parts = NULL;
   }

   simple_mtx_destroy(&cache->lock);
}

const struct agx_shader_part *
d12_shader_cache_part(struct d12_device *device, d12_part_builder_t builder,
                      const void *key_data, size_t key_size,
                      const struct agx_shader_key *backend_key,
                      bool preprocess)
{
   struct d12_shader_cache *cache = &device->shader_cache;
   const size_t total = sizeof(struct d12_part_key) + key_size;
   uint8_t storage[sizeof(struct d12_part_key) + D12_PART_KEY_MAX];
   struct d12_part_key *key = (struct d12_part_key *)storage;

   assert(key_size <= D12_PART_KEY_MAX);
   memset(storage, 0, total);
   key->builder = builder;
   key->key_size = key_size;
   memcpy(key->key, key_data, key_size);

   simple_mtx_lock(&cache->lock);
   struct hash_entry *ent = _mesa_hash_table_search(cache->parts, key);
   simple_mtx_unlock(&cache->lock);
   if (ent)
      return ent->data;

   /* Miss: build and compile outside the lock. */
   nir_builder b = nir_builder_init_simple_shader(0, &agx_nir_options, NULL);
   builder(&b, key_data);
   if (preprocess) {
      if (device->dev.chip == AGX_CHIP_G17P)
         agx_preprocess_nir_g17p(b.shader);
      else
         agx_preprocess_nir(b.shader);
   }

   struct agx_shader_part *part = rzalloc(NULL, struct agx_shader_part);
   if (!part) {
      ralloc_free(b.shader);
      return NULL;
   }

   /* The backend takes a mutable key; it does not change ours. */
   struct agx_shader_key key_copy = *backend_key;
   const bool compiled = agx_compile_shader_nir(b.shader, &key_copy, part);
   ralloc_free(b.shader);
   if (!compiled) {
      /* A refused backend instruction leaves an empty part. Caching that
       * as success would let fast-link silently omit required code. */
      free(part->binary);
      ralloc_free(part);
      return NULL;
   }

   simple_mtx_lock(&cache->lock);
   ent = _mesa_hash_table_search(cache->parts, key);
   if (ent) {
      /* Another thread compiled the same part meanwhile: adopt theirs. */
      simple_mtx_unlock(&cache->lock);
      free(part->binary);
      ralloc_free(part);
      return ent->data;
   }

   struct d12_part_key *clone = ralloc_size(cache->parts, total);
   if (!clone) {
      simple_mtx_unlock(&cache->lock);
      free(part->binary);
      ralloc_free(part);
      return NULL;
   }
   memcpy(clone, key, total);
   ralloc_steal(cache->parts, part);
   _mesa_hash_table_insert(cache->parts, clone, part);
   simple_mtx_unlock(&cache->lock);
   return part;
}
