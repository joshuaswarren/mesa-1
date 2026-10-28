/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "asahi/layout/layout.h"

struct hk_g17p_zls_plane {
   uint64_t buffer, meta;
   uint32_t stride, meta_stride;
   uint32_t width, height;
   enum ail_tiling tiling;
};

/* Build one aspect's attachment state. The caller supplies that aspect's
 * allocation, not the other plane of a combined depth/stencil image. */
static inline struct hk_g17p_zls_plane
hk_g17p_zls_plane(const struct ail_layout *layout, uint64_t base,
                  unsigned layer, unsigned level)
{
   assert(level < layout->levels && layer < layout->depth_px);
   assert(layout->tiling == AIL_TILING_GPU ||
          layout->tiling == AIL_TILING_TWIDDLED);

   struct hk_g17p_zls_plane state = {
      .buffer = base + ail_get_layer_level_B(layout, layer, level),
      .width = u_minify(layout->width_px, level),
      .height = u_minify(layout->height_px, level),
      .tiling = layout->tiling,
   };
   assert(state.width <= 16384 && state.height <= 16384);
   assert(util_is_aligned(state.buffer, AIL_CACHELINE));

   assert(layout->depth_px == 1 ||
          util_is_aligned(layout->layer_stride_B, AIL_PAGESIZE));
   unsigned pages = layout->layer_stride_B / AIL_PAGESIZE;
   /* A single layer does not use the stride. Keep sub-page strides zero,
    * as required by the native kernel's ZLS validation. */
   if (pages)
      state.stride = ((pages - 1) << 14) | 1;

   if (layout->compressed) {
      /* The narrow tail can have reserved metadata without being compressed.
       * It uses twiddled data and must not address those metadata slots. */
      state.tiling = ail_is_level_twiddled_uncompressed(layout, level)
                        ? AIL_TILING_TWIDDLED
                        : layout->tiling;
      if (base && ail_is_level_logically_compressed(layout, level) &&
          layout->compression_layer_stride_B >= AIL_CACHELINE) {
         assert(
            util_is_aligned(layout->compression_layer_stride_B, AIL_CACHELINE));
         state.meta = base + layout->metadata_offset_B +
                      (uint64_t)layer * layout->compression_layer_stride_B +
                      layout->level_offsets_compressed_B[level];
         assert(util_is_aligned(state.meta, AIL_CACHELINE));
         state.meta_stride =
            (layout->compression_layer_stride_B / AIL_CACHELINE - 1) << 14;
      }
   }

   return state;
}
