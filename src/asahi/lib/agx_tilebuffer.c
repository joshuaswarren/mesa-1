/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "agx_tilebuffer.h"
#include <assert.h>
#include "util/format/u_format.h"
#include "layout.h"

/* Maximum number of bytes per tile on G13G. This may change in future versions
 * of the architecture.
 */
#define MAX_BYTES_PER_TILE (32768 - 1)

/* Maximum bytes per sample in the tilebuffer. Greater allocations require
 * spilling render targets to memory.
 */
#define MAX_BYTES_PER_SAMPLE (64)

/* Minimum tile size in pixels, architectural. */
#define MIN_TILE_SIZE_PX (16 * 16)

/* Select the largest tile size that fits */
static uint16_t
agx_select_tile_size(unsigned px_size_B)
{
   assert(px_size_B <= (MAX_BYTES_PER_TILE / MIN_TILE_SIZE_PX));

   return ((px_size_B * 32 * 32) <= MAX_BYTES_PER_TILE)   ? (32 * 32)
          : ((px_size_B * 32 * 16) <= MAX_BYTES_PER_TILE) ? (32 * 16)
                                                          : MIN_TILE_SIZE_PX;
}

static inline unsigned
format_align_B(enum pipe_format format)
{
   /* For some reason util_format_get_blocksize(NONE) = 1 */
   enum pipe_format phys = ail_pixel_format[format].renderable;
   return (format != PIPE_FORMAT_NONE) ? util_format_get_blocksize(phys) : 0;
}

struct agx_tilebuffer_layout
agx_build_tilebuffer_layout(const enum pipe_format *formats, uint8_t nr_cbufs,
                            uint8_t nr_samples, bool layered)
{
   struct agx_tilebuffer_layout tib = {
      .nr_samples = nr_samples,
      .layered = layered,
   };

   uint32_t offset_B = 0;
   uint8_t order[] = {0, 1, 2, 3, 4, 5, 6, 7};

   /* Sort render targets in descending order of alignment, eliminating padding
    * and giving the optimal order of render targets. We use insertion sort
    * because it is simple, stable, fast for small n, and free for n=1.
    */
   for (int i = 1; i < nr_cbufs; ++i) {
      for (int j = i; j > 0 && format_align_B(formats[order[j - 1]]) <
                                  format_align_B(formats[order[j]]);
           --j) {
         SWAP(order[j], order[j - 1]);
      }
   }

   for (unsigned i = 0; i < nr_cbufs; ++i) {
      unsigned rt = order[i];
      enum pipe_format format = formats[rt];
      tib.logical_format[rt] = format;

      assert(util_is_aligned(offset_B, MAX2(format_align_B(formats[rt]), 1)) &&
             "loop invariant ensured by the sort");

      unsigned size_B = format_align_B(format);
      enum pipe_format phys = ail_pixel_format[format].renderable;
      if (util_format_get_nr_components(phys) == 1) {
         size_B *= util_format_get_nr_components(format);
      }

      /* If allocating this render target would exceed any tilebuffer limits, we
       * need to spill it to memory. Otherwise, allocate it to the tilebuffer.
       */
      unsigned new_offset_B = offset_B + size_B;
      bool fits = (new_offset_B <= MAX_BYTES_PER_SAMPLE) &&
                  (ALIGN_POT(new_offset_B, 8) * MIN_TILE_SIZE_PX *
                   nr_samples) <= MAX_BYTES_PER_TILE;
      if (fits) {
         tib._offset_B[rt] = offset_B;
         offset_B = new_offset_B;
      } else {
         tib.spilled[rt] = true;
      }
   }

   assert(offset_B <= MAX_BYTES_PER_SAMPLE && "loop invariant");

   /* Multisampling needs a nonempty allocation.
    * XXX: Check this against hw
    */
   if (nr_samples > 1)
      offset_B = MAX2(offset_B, 1);

   tib.sample_size_B = align(offset_B, 8);
   tib.tile_size = agx_select_tile_size(tib.sample_size_B * nr_samples);

   agx_tilebuffer_pack_usc(&tib);
   return tib;
}

/*
 * With attachmentless rendering in Vulkan, the sample count may not known until
 * draw-time. It's convenient to construct an agx_tilebuffer_layout anyway when
 * beginning rendering, updating the sample count later. This helper allows the
 * driver to set the sample count in a partial agx_tilebuffer_layout.
 *
 * When doing so, we need to rebuild entirely since e.g. tile size might change.
 */
void
agx_tilebuffer_set_samples(struct agx_tilebuffer_layout *tib,
                           unsigned nr_samples)
{
   assert(tib->nr_samples == 0 && "must not be initialized");

   *tib = agx_build_tilebuffer_layout(tib->logical_format,
                                      ARRAY_SIZE(tib->logical_format),
                                      nr_samples, tib->layered);
}

enum pipe_format
agx_tilebuffer_physical_format(struct agx_tilebuffer_layout *tib, unsigned rt)
{
   return ail_pixel_format[tib->logical_format[rt]].renderable;
}

bool
agx_tilebuffer_supports_mask(struct agx_tilebuffer_layout *tib, unsigned rt)
{
   /* We don't bother support masking with spilled render targets. This might be
    * optimized in the future but spilling is so rare anyway it's not worth it.
    */
   if (tib->spilled[rt])
      return false;

   enum pipe_format fmt = agx_tilebuffer_physical_format(tib, rt);
   return ail_isa_format_supports_mask((enum ail_isa_format)fmt);
}

uint32_t
agx_tilebuffer_total_size(struct agx_tilebuffer_layout *tib)
{
   return tib->sample_size_B * tib->nr_samples * tib->tile_size;
}

void
agx_tilebuffer_pack_usc(struct agx_tilebuffer_layout *tib)
{
   agx_pack(&tib->usc, USC_SHARED, cfg) {
      if (tib->nr_samples > 0) {
         cfg.uses_shared_memory = true;
         cfg.sample_stride_in_8_bytes = tib->sample_size_B / 8;
         cfg.sample_count = tib->nr_samples;
         cfg.bytes_per_threadgroup = agx_tilebuffer_total_size(tib);

         if (tib->tile_size == 32 * 32)
            cfg.layout = AGX_SHARED_LAYOUT_32X32;
         else if (tib->tile_size == 32 * 16)
            cfg.layout = AGX_SHARED_LAYOUT_32X16;
         else if (tib->tile_size == 16 * 16)
            cfg.layout = AGX_SHARED_LAYOUT_16X16;
         else
            UNREACHABLE("Invalid tile size");
      } else {
         cfg.layout = AGX_SHARED_LAYOUT_VERTEX_COMPUTE;
         cfg.bytes_per_threadgroup = 65536;
      }
   }
}

bool
agx_g17p_rt0_tile_layout_supported(struct agx_tilebuffer_layout *tib)
{
   if (!tib ||
       (tib->nr_samples != 1 && tib->nr_samples != 2 &&
        tib->nr_samples != 4) ||
       tib->layered || tib->spilled[0])
      return false;

   /* Four 8-byte samples fill exactly 32 KiB at 32x32, but the hardware
    * limit is strictly smaller. Keep the generic allocator's valid 32x16
    * layout; BG/EOT use pixel coordinates and the render UAPI publishes the
    * utile geometry to firmware. */
   if (tib->tile_size != 32 * 32 && tib->tile_size != 32 * 16)
      return false;

   if (tib->nr_samples > 1 &&
       tib->logical_format[0] != PIPE_FORMAT_R8G8B8A8_UNORM &&
       tib->logical_format[0] != PIPE_FORMAT_B8G8R8A8_UNORM)
      return false;

   const bool mixed_mrt =
      tib->nr_samples == 1 &&
      tib->logical_format[0] == PIPE_FORMAT_R16G16B16A16_FLOAT &&
      tib->logical_format[1] == PIPE_FORMAT_R8_UNORM && !tib->spilled[1] &&
      tib->_offset_B[0] == 0 && tib->_offset_B[1] == 8 &&
      tib->sample_size_B == 16;

   for (unsigned rt = mixed_mrt ? 2 : 1; rt < AGX_MAX_RENDER_TARGETS; ++rt) {
      if (tib->logical_format[rt] != PIPE_FORMAT_NONE)
         return false;
   }

   if (tib->logical_format[0] == PIPE_FORMAT_NONE)
      return tib->sample_size_B == 0;

   const bool scalar_r32f = tib->nr_samples == 1 &&
                            tib->logical_format[0] == PIPE_FORMAT_R32_FLOAT;
   const bool scalar_r32i = tib->nr_samples == 1 &&
                            tib->logical_format[0] == PIPE_FORMAT_R32_SINT;
   const bool pair_r32f = tib->nr_samples == 1 &&
                          tib->logical_format[0] == PIPE_FORMAT_R32G32_FLOAT;
   const bool pair_r32i = tib->nr_samples == 1 &&
                          tib->logical_format[0] == PIPE_FORMAT_R32G32_SINT;
   const bool pair_r32u = tib->nr_samples == 1 &&
                          tib->logical_format[0] == PIPE_FORMAT_R32G32_UINT;
   const bool scalar_r8u = tib->nr_samples == 1 &&
                           tib->logical_format[0] == PIPE_FORMAT_R8_UINT;
   const bool scalar_r8i = tib->nr_samples == 1 &&
                           tib->logical_format[0] == PIPE_FORMAT_R8_SINT;
   const bool rgba16u = tib->nr_samples == 1 &&
      tib->logical_format[0] == PIPE_FORMAT_R16G16B16A16_UINT;
   const bool rgba16i = tib->nr_samples == 1 &&
      tib->logical_format[0] == PIPE_FORMAT_R16G16B16A16_SINT;
   const bool rgba32u = tib->nr_samples == 1 && tib->sample_size_B == 16 &&
      tib->logical_format[0] == PIPE_FORMAT_R32G32B32A32_UINT &&
      agx_tilebuffer_physical_format(tib, 0) == PIPE_FORMAT_R32_UINT;
   const bool rgba32i = tib->nr_samples == 1 && tib->sample_size_B == 16 &&
      tib->logical_format[0] == PIPE_FORMAT_R32G32B32A32_SINT &&
      agx_tilebuffer_physical_format(tib, 0) == PIPE_FORMAT_R32_UINT;
   const bool rgba32f = tib->nr_samples == 1 && tib->sample_size_B == 16 &&
      tib->logical_format[0] == PIPE_FORMAT_R32G32B32A32_FLOAT &&
      agx_tilebuffer_physical_format(tib, 0) == PIPE_FORMAT_R32_UINT;
   const bool rgb10a2u = tib->nr_samples == 1 && tib->sample_size_B == 8 &&
      tib->logical_format[0] == PIPE_FORMAT_R10G10B10A2_UINT &&
      agx_tilebuffer_physical_format(tib, 0) == PIPE_FORMAT_R16_UINT;

   return mixed_mrt || rgba32u || rgba32i || rgba32f || rgb10a2u ||
          (tib->sample_size_B == 8 &&
           (agx_tilebuffer_physical_format(tib, 0) == PIPE_FORMAT_R8_UNORM ||
            agx_tilebuffer_physical_format(tib, 0) == PIPE_FORMAT_R16_FLOAT ||
            agx_tilebuffer_physical_format(tib, 0) ==
               PIPE_FORMAT_R10G10B10A2_UNORM ||
            tib->logical_format[0] == PIPE_FORMAT_R16G16_UNORM ||
            tib->logical_format[0] == PIPE_FORMAT_R16G16_SNORM ||
            tib->logical_format[0] == PIPE_FORMAT_R32_UINT ||
            tib->logical_format[0] == PIPE_FORMAT_R8_SNORM ||
            tib->logical_format[0] == PIPE_FORMAT_R8G8_SNORM ||
            tib->logical_format[0] == PIPE_FORMAT_R8G8_UINT ||
            tib->logical_format[0] == PIPE_FORMAT_R8G8_SINT ||
            tib->logical_format[0] == PIPE_FORMAT_R8G8B8A8_SNORM ||
            tib->logical_format[0] == PIPE_FORMAT_B8G8R8A8_SNORM ||
            tib->logical_format[0] == PIPE_FORMAT_R8G8B8A8_UINT ||
            tib->logical_format[0] == PIPE_FORMAT_R8G8B8A8_SINT ||
            tib->logical_format[0] == PIPE_FORMAT_R16_UNORM ||
            tib->logical_format[0] == PIPE_FORMAT_R16_SNORM ||
            tib->logical_format[0] == PIPE_FORMAT_R16_UINT ||
            tib->logical_format[0] == PIPE_FORMAT_R16_SINT ||
            tib->logical_format[0] == PIPE_FORMAT_R16_FLOAT ||
            tib->logical_format[0] == PIPE_FORMAT_R16G16_FLOAT ||
            tib->logical_format[0] == PIPE_FORMAT_R16G16B16A16_FLOAT ||
            tib->logical_format[0] == PIPE_FORMAT_R16G16B16A16_SNORM ||
            tib->logical_format[0] == PIPE_FORMAT_R16G16_UINT ||
            tib->logical_format[0] == PIPE_FORMAT_R16G16_SINT ||
            tib->logical_format[0] == PIPE_FORMAT_R16G16B16A16_UNORM ||
            scalar_r32f || scalar_r32i || pair_r32f || pair_r32i ||
            pair_r32u || scalar_r8u || scalar_r8i || rgba16u || rgba16i));
}
