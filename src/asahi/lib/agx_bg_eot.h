/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "agx_bg_eot_key.h"
#include "util/simple_mtx.h"
#include "agx_tilebuffer.h"
#include "libagx_dgc.h"
#include "libagx_shaders.h"
#include "pool.h"

struct agx_precompiled_shader {
   struct agx_shader b;
   struct agx_bo *bo;
   uint64_t ptr;
   uint32_t entry_offset;
   /* G17P only: byte offset of the promoted-constant block within the
    * uploaded image, valid when info.rodata.size_16 is nonzero. The legacy
    * USC path reaches the same bytes through agx_usc_immediates() and does
    * not need this. */
   uint32_t rodata_offset;
   struct agx_precompiled_kernel_info info;
};

struct agx_bg_eot_cache {
   struct agx_device *dev;
   struct agx_pool pool;
   simple_mtx_t lock;

   /* Map from agx_bg_eot_key to agx_bg_eot_shader */
   struct hash_table *ht;

   struct agx_precompiled_shader *precomp[LIBAGX_NUM_PROGRAMS];
};

/*
 * Get a precompiled shader, uploading if necessary. This is thread-safe.
 */
struct agx_precompiled_shader *
agx_get_precompiled(struct agx_bg_eot_cache *cache, unsigned program);

/*
 * Get the address of the cached helper program. This is thread-safe.
 */
uint64_t agx_helper_program(struct agx_bg_eot_cache *cache);

enum agx_bg_eot_op {
   AGX_BG_EOT_NONE,
   AGX_BG_CLEAR,
   AGX_BG_LOAD,
   AGX_EOT_STORE,
};

struct agx_bg_eot_key {
   struct agx_tilebuffer_layout tib;
   enum agx_bg_eot_op op[8];
   unsigned reserved_preamble;
   /* Replicate a single-sample WSI colour into every multisample tile lane. */
   bool single_sample_reload;
   /* Partial direct resolve: retain destination pixels outside its area. */
   bool preserve_resolve_outside;
   /* Store the multisample tile unresolved into a multisample target rather
    * than resolving it through a single-sample one. Only meaningful when
    * tib.nr_samples > 1; the caller decides, because storeOp = STORE is the
    * only Vulkan path that wants the unresolved surface. */
   bool multisample_store;
   /* Store RT0 samples and also resolve that same tile through PBE1.
    * The two EOT outputs share an imageblock offset, not storage. */
   bool store_and_resolve;
   /* M5 final EOT stores samples and resolves the same tile in one launch.
    * Kept separate from the G17P store_and_resolve path above; the two
    * selectors drive different EOT programs. */
   bool m5_store_and_resolve;
   /* The framebuffer is an AGX_G17P_MRT_MIXED layout. The tile layout keeps
    * only the targets this program touches, at their framebuffer offsets. */
   bool mixed_mrt;
   /* G17P packed clear: the single RT0 R8G8B8A8_UNORM clear value
    * arrives already packed in the first 32-bit clear uniform, so the
    * background program stores it as one tile word instead of converting four
    * floats with two FLTNORMs per pixel. Only agx_g17p_packed_clear_ok()
    * layouts set it. */
   bool g17p_packed_clear;
};

/* The layouts whose RT0 clear may be stored pre-packed: one single-sample,
 * unspilled R8G8B8A8_UNORM target at tile offset 0. Its tile word is the
 * byte image FLTNORM would produce (R, G, B, A in byte order), and the
 * end-of-tile PBE and LLDCOL read those bytes back unchanged. */
static inline bool
agx_g17p_packed_clear_ok(const struct agx_tilebuffer_layout *tib)
{
   if (tib->nr_samples != 1 || tib->spilled[0] ||
       tib->logical_format[0] != PIPE_FORMAT_R8G8B8A8_UNORM ||
       agx_tilebuffer_offset_B((struct agx_tilebuffer_layout *)tib, 0) != 0)
      return false;
   for (unsigned rt = 1; rt < ARRAY_SIZE(tib->logical_format); ++rt) {
      if (tib->logical_format[rt] != PIPE_FORMAT_NONE)
         return false;
   }
   return true;
}

struct agx_bg_eot_shader {
   /* Driver-owned immutable launch templates, ralloc children of this
    * shader. The cache lock serializes creation and list access. */
   void *m5_tile_templates;
   struct agx_bg_eot_key key;
   struct agx_shader_info info;
   struct agx_bo *bo;
   uint64_t ptr;
};

struct agx_bg_eot_shader *agx_get_bg_eot_shader(struct agx_bg_eot_cache *cache,
                                                struct agx_bg_eot_key *key);

void agx_bg_eot_init(struct agx_bg_eot_cache *cache, struct agx_device *dev);
void agx_bg_eot_cleanup(struct agx_bg_eot_cache *cache);
