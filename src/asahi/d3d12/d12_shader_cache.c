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
 * The application's own stages repeat too: a game creates many PSOs that
 * differ only in fixed-function state over the same vertex and pixel
 * bytecode, and it recreates the same PSO after a pipeline-library load.
 * The second table keys a compiled stage on the bytecode's content hash,
 * the root-signature binding map and the per-PSO compile keys that reach
 * the NIR (vkd3d-proton and hk's vk_pipeline_cache usage do the same with
 * a blake3 of the SPIR-V plus layout). A hit skips the DXIL parse, the
 * DXIL->NIR translation, every lowering pass and the backend compile; the
 * entry also keeps its executable upload and its fast-linked variants so
 * repeated PSOs share GPU memory instead of re-uploading it.
 *
 * Locking: the table lock is held only to look up and to insert. Compiles
 * and links run outside it so one thread's backend compile does not
 * serialise every other thread's pipeline creation; a thread that loses
 * the race adopts the entry the winner inserted and discards its own.
 * Stage entries are reference counted so an entry evicted under a
 * concurrent user lives until that user is done.
 */

#include "d12_private.h"
#include "agx_compile.h"
#include "agx_nir.h"
#include "nir.h"
#include "nir_builder.h"
#include "util/hash_table.h"
#include "util/ralloc.h"
#define XXH_INLINE_ALL
#include "util/xxhash.h"

/* Compiled API stages kept once their last PSO is gone. Each entry holds
 * its binary, a copy of its bytecode and, lazily, one executable upload
 * plus its fast-linked variants; PSOs own references on those BOs, so an
 * eviction never pulls code from under a live pipeline. */
#define D12_SHADER_CACHE_MAX_STAGES 2048

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

static uint32_t
d12_stage_key_hash(const void *data)
{
   return _mesa_hash_data(data, sizeof(struct d12_stage_cache_key));
}

static bool
d12_stage_key_equal(const void *a, const void *b)
{
   return !memcmp(a, b, sizeof(struct d12_stage_cache_key));
}

static uint32_t
d12_link_key_hash(const void *data)
{
   return _mesa_hash_data(data, sizeof(struct d12_link_key));
}

static bool
d12_link_key_equal(const void *a, const void *b)
{
   return !memcmp(a, b, sizeof(struct d12_link_key));
}

void
d12_shader_cache_init(struct d12_shader_cache *cache)
{
   simple_mtx_init(&cache->lock, mtx_plain);
   cache->parts =
      _mesa_hash_table_create(NULL, d12_part_key_hash, d12_part_key_equal);
   cache->stages =
      _mesa_hash_table_create(NULL, d12_stage_key_hash, d12_stage_key_equal);
   list_inithead(&cache->stage_lru);
   cache->nr_stages = 0;
}

static void
d12_cached_stage_destroy(struct d12_device *device,
                         struct d12_cached_stage *stage)
{
   if (stage->links) {
      hash_table_foreach(stage->links, ent) {
         struct agx_linked_shader *linked = ent->data;
         if (linked->bo)
            agx_bo_unreference(&device->dev, linked->bo);
      }
      /* Keys and linked shaders are ralloc children of the table. */
      _mesa_hash_table_destroy(stage->links, NULL);
   }
   if (stage->code_bo)
      agx_bo_unreference(&device->dev, stage->code_bo);
   free(stage->part.binary);
   free(stage->blob);
   free(stage);
}

/* Drop the table's reference on an entry; the lock is held. */
static void
d12_shader_cache_evict_locked(struct d12_device *device,
                              struct d12_cached_stage *stage)
{
   struct d12_shader_cache *cache = &device->shader_cache;

   assert(stage->in_table);
   _mesa_hash_table_remove_key(cache->stages, &stage->key);
   list_del(&stage->lru_link);
   stage->in_table = false;
   cache->nr_stages--;
   if (--stage->refcount == 0)
      d12_cached_stage_destroy(device, stage);
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

   if (cache->stages) {
      /* Every PSO holds a device reference, so no lookup is live. */
      list_for_each_entry_safe(struct d12_cached_stage, stage,
                               &cache->stage_lru, lru_link) {
         assert(stage->refcount == 1);
         d12_shader_cache_evict_locked(device, stage);
      }
      _mesa_hash_table_destroy(cache->stages, NULL);
      cache->stages = NULL;
   }

   simple_mtx_destroy(&cache->lock);
}

void
d12_stage_cache_key_init(struct d12_stage_cache_key *key,
                         mesa_shader_stage stage, const void *blob,
                         size_t blob_size, const struct d12_binding_map *map)
{
   memset(key, 0, sizeof(*key));
   key->stage = stage;
   key->blob_size = blob_size;
   key->blob_hash = blob_size ? XXH64(blob, blob_size, 0) : 0;

   /* The map's live entries and its counts decide every register -> slot
    * assignment the translator makes. Entries past `count` are zero. */
   if (map) {
      key->has_map = true;
      for (unsigned seed = 0; seed < 2; seed++) {
         XXH64_state_t state;
         XXH64_reset(&state, seed);
         XXH64_update(&state, map->entries,
                      MIN2(map->count, ARRAY_SIZE(map->entries)) *
                         sizeof(map->entries[0]));
         XXH64_update(&state, &map->count, sizeof(map->count));
         XXH64_update(&state, &map->resource_count,
                      sizeof(map->resource_count));
         XXH64_update(&state, &map->sampler_count,
                      sizeof(map->sampler_count));
         XXH64_update(&state, &map->overflow, sizeof(map->overflow));
         XXH64_update(&state, &map->visibility_filter,
                      sizeof(map->visibility_filter));
         key->map_hash[seed] = XXH64_digest(&state);
      }
   }
}

struct d12_cached_stage *
d12_shader_cache_find_stage(struct d12_device *device,
                            const struct d12_stage_cache_key *key,
                            const void *blob)
{
   struct d12_shader_cache *cache = &device->shader_cache;
   struct d12_cached_stage *stage = NULL;

   simple_mtx_lock(&cache->lock);
   struct hash_entry *ent = _mesa_hash_table_search(cache->stages, key);
   if (ent) {
      stage = ent->data;
      /* The hash names the bytecode; the bytes confirm it. */
      if (key->blob_size && memcmp(stage->blob, blob, key->blob_size)) {
         stage = NULL;
      } else {
         stage->refcount++;
         list_del(&stage->lru_link);
         list_addtail(&stage->lru_link, &cache->stage_lru);
      }
   }
   simple_mtx_unlock(&cache->lock);
   return stage;
}

struct d12_cached_stage *
d12_shader_cache_add_stage(struct d12_device *device,
                           const struct d12_stage_cache_key *key,
                           const void *blob, struct agx_shader_part *part,
                           const struct d12_stage_outputs *out, bool insert)
{
   struct d12_shader_cache *cache = &device->shader_cache;

   struct d12_cached_stage *stage = calloc(1, sizeof(*stage));
   if (!stage)
      return NULL;
   if (key->blob_size) {
      stage->blob = malloc(key->blob_size);
      if (!stage->blob) {
         free(stage);
         return NULL;
      }
      memcpy(stage->blob, blob, key->blob_size);
   }
   stage->key = *key;
   stage->out = *out;
   stage->links = _mesa_hash_table_create(NULL, d12_link_key_hash,
                                          d12_link_key_equal);
   if (!stage->links) {
      free(stage->blob);
      free(stage);
      return NULL;
   }

   /* Private entries (debug dumps) never enter the table. */
   if (!insert) {
      stage->part = *part;
      part->binary = NULL;
      stage->refcount = 1;
      return stage;
   }

   simple_mtx_lock(&cache->lock);
   struct hash_entry *ent = _mesa_hash_table_search(cache->stages, key);
   if (ent) {
      struct d12_cached_stage *existing = ent->data;
      if (!key->blob_size || !memcmp(existing->blob, blob, key->blob_size)) {
         /* Another thread compiled the same stage meanwhile: adopt it. */
         existing->refcount++;
         simple_mtx_unlock(&cache->lock);
         _mesa_hash_table_destroy(stage->links, NULL);
         free(stage->blob);
         free(stage);
         free(part->binary);
         part->binary = NULL;
         return existing;
      }
      /* A content-hash collision: the new compile replaces the entry. */
      d12_shader_cache_evict_locked(device, existing);
   }

   stage->part = *part;
   part->binary = NULL;
   stage->refcount = 2; /* the table and the caller */
   stage->in_table = true;
   _mesa_hash_table_insert(cache->stages, &stage->key, stage);
   list_addtail(&stage->lru_link, &cache->stage_lru);
   cache->nr_stages++;

   /* Bound the table: drop the least recently used entries no pipeline
    * creation is reading. */
   if (cache->nr_stages > D12_SHADER_CACHE_MAX_STAGES) {
      list_for_each_entry_safe(struct d12_cached_stage, old,
                               &cache->stage_lru, lru_link) {
         if (cache->nr_stages <= D12_SHADER_CACHE_MAX_STAGES)
            break;
         if (old->refcount == 1)
            d12_shader_cache_evict_locked(device, old);
      }
   }
   simple_mtx_unlock(&cache->lock);
   return stage;
}

void
d12_cached_stage_unref(struct d12_device *device,
                       struct d12_cached_stage *stage)
{
   struct d12_shader_cache *cache = &device->shader_cache;

   simple_mtx_lock(&cache->lock);
   const bool last = --stage->refcount == 0;
   simple_mtx_unlock(&cache->lock);
   if (last)
      d12_cached_stage_destroy(device, stage);
}

struct agx_bo *
d12_cached_stage_code_bo(struct d12_device *device,
                         struct d12_cached_stage *stage, const char *name)
{
   struct d12_shader_cache *cache = &device->shader_cache;

   simple_mtx_lock(&cache->lock);
   struct agx_bo *bo = stage->code_bo;
   if (bo)
      agx_bo_reference(bo);
   simple_mtx_unlock(&cache->lock);
   if (bo)
      return bo;

   /* Shader code must live in executable, low-VA memory so the USC can
    * address it 32-bit-relative to the shader base. Allocate and upload
    * outside the lock. */
   bo = agx_bo_create(&device->dev, stage->part.info.binary_size, 0,
                      AGX_BO_EXEC | AGX_BO_LOW_VA, name);
   if (!bo)
      return NULL;
   if (!d12_upload_executable(bo, stage->part.binary,
                              stage->part.info.binary_size)) {
      agx_bo_unreference(&device->dev, bo);
      return NULL;
   }

   simple_mtx_lock(&cache->lock);
   if (stage->code_bo) {
      /* Lost the race: use the other upload. */
      struct agx_bo *theirs = stage->code_bo;
      agx_bo_reference(theirs);
      simple_mtx_unlock(&cache->lock);
      agx_bo_unreference(&device->dev, bo);
      return theirs;
   }
   stage->code_bo = bo;
   agx_bo_reference(bo); /* the caller's reference */
   simple_mtx_unlock(&cache->lock);
   return bo;
}

bool
d12_cached_stage_link(struct d12_device *device,
                      struct d12_cached_stage *stage,
                      const struct d12_link_key *key,
                      const struct agx_shader_part *prolog,
                      const struct agx_shader_part *epilog,
                      struct agx_linked_shader *out)
{
   struct d12_shader_cache *cache = &device->shader_cache;

   simple_mtx_lock(&cache->lock);
   struct hash_entry *ent = _mesa_hash_table_search(stage->links, key);
   if (ent) {
      *out = *(struct agx_linked_shader *)ent->data;
      agx_bo_reference(out->bo);
      simple_mtx_unlock(&cache->lock);
      return true;
   }
   simple_mtx_unlock(&cache->lock);

   /* Link outside the lock: this uploads executable memory. The linker
    * only reads its parts, so the shared cached parts are safe to pass. */
   struct agx_linked_shader linked;
   memset(&linked, 0, sizeof(linked));
   if (!agx_fast_link(&linked, &device->dev, key->fragment, &stage->part,
                      (struct agx_shader_part *)prolog,
                      (struct agx_shader_part *)epilog,
                      key->nr_samples_shaded))
      return false;

   simple_mtx_lock(&cache->lock);
   ent = _mesa_hash_table_search(stage->links, key);
   if (ent) {
      *out = *(struct agx_linked_shader *)ent->data;
      agx_bo_reference(out->bo);
      simple_mtx_unlock(&cache->lock);
      agx_bo_unreference(&device->dev, linked.bo);
      return true;
   }

   struct d12_link_key *key_clone =
      ralloc_memdup(stage->links, key, sizeof(*key));
   struct agx_linked_shader *clone =
      ralloc_memdup(stage->links, &linked, sizeof(linked));
   if (!key_clone || !clone) {
      /* Not cached, still linked: the caller owns this upload alone. */
      simple_mtx_unlock(&cache->lock);
      ralloc_free(key_clone);
      ralloc_free(clone);
      *out = linked;
      return true;
   }
   _mesa_hash_table_insert(stage->links, key_clone, clone);
   agx_bo_reference(linked.bo); /* the caller's reference */
   simple_mtx_unlock(&cache->lock);
   *out = linked;
   return true;
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
