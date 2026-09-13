/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "agx_tilebuffer.h"
#include <assert.h>
#include "util/format/u_format.h"
#include "util/u_debug.h"
#include "layout.h"

/* Maximum number of bytes per tile on G13G. This may change in future versions
 * of the architecture.
 *
 * Do not "fix" the minus one for G17P. Four 8-byte samples of RGBA8 at 32x32
 * need exactly 32768 bytes, so 4x colour misses this bound by a single byte
 * and falls to a 32x16 tile, which looks like an off-by-one worth testing. It
 * was tested: admitting exactly 32768 and letting 4x keep a 32x32 tile loses
 * the device outright (vkWaitForFences -> VK_ERROR_DEVICE_LOST on A18 Pro,
 * 2026-09-10). The bound is real; a 32x32 tile at 4x is not something this
 * hardware will do.
 */
#define MAX_BYTES_PER_TILE (32768 - 1)

/* Maximum bytes per sample in the tilebuffer. Greater allocations require
 * spilling render targets to memory.
 */
#define MAX_BYTES_PER_SAMPLE (64)

/* Minimum tile size in pixels, architectural. */
#define MIN_TILE_SIZE_PX (16 * 16)

/* Diagnostic: admit a tile of exactly 32768 bytes rather than one byte less.
 *
 * This exists for one specific question. Four 8-byte samples of RGBA8 at 32x32
 * need exactly 32768 bytes, so 4x colour misses the 32x32 tile by a single
 * byte and falls to 32x16 -- and 32x16 is a tile geometry that essentially
 * nothing else on G17P reaches, because at one sample you would need more than
 * 31 bytes per pixel to get there and the G17P tile gate admits almost no such
 * configuration. The hardware symptom of 4x (A01b: rows 0-15 exact, rows 16-31
 * absent) is the tile height, not a sample count.
 *
 * So either the limit really is 32767 and the 32x16 path is broken, or the
 * limit is 32768 on this part and 4x should never have left 32x32. Setting
 * AGX_TILE_32K=1 picks the second and changes exactly that one comparison;
 * the matching SPECLM bound in agx_ioto_state_core.c moves with it. Off by
 * default: the value here is inherited from G13G and has no G17P witness in
 * either direction.
 */
static bool
agx_tile_allows_full_32k(void)
{
   static int enabled = -1;
   if (enabled < 0)
      enabled = debug_get_bool_option("AGX_TILE_32K", false);
   return enabled != 0;
}

/* Select the largest tile size that fits */
static uint16_t
agx_select_tile_size(unsigned px_size_B)
{
   const unsigned limit =
      agx_tile_allows_full_32k() ? 32768 : MAX_BYTES_PER_TILE;

   assert(px_size_B <= (limit / MIN_TILE_SIZE_PX));

   return ((px_size_B * 32 * 32) <= limit)   ? (32 * 32)
          : ((px_size_B * 32 * 16) <= limit) ? (32 * 16)
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
   const enum agx_g17p_mrt_layout mrt = agx_g17p_classify_mrt(tib);
   const bool quad = mrt == AGX_G17P_MRT_RGBA32_QUAD ||
                     mrt == AGX_G17P_MRT_R32_QUAD;
   if (tib->tile_size != 32 * 32 && tib->tile_size != 32 * 16 &&
       !(mrt == AGX_G17P_MRT_RGBA32_QUAD && tib->tile_size == 16 * 16))
      return false;

   if (tib->nr_samples > 1 &&
       tib->logical_format[0] != PIPE_FORMAT_R8G8B8A8_UNORM &&
       tib->logical_format[0] != PIPE_FORMAT_B8G8R8A8_UNORM &&
       tib->logical_format[0] != PIPE_FORMAT_R8G8B8X8_UNORM &&
       tib->logical_format[0] != PIPE_FORMAT_B8G8R8X8_UNORM)
      return false;

   for (unsigned rt = quad ? 4 : mrt != AGX_G17P_MRT_NONE ? 2 : 1;
        rt < AGX_MAX_RENDER_TARGETS; ++rt) {
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
   const bool rgba16u_or_x = tib->nr_samples == 1 &&
      (tib->logical_format[0] == PIPE_FORMAT_R16G16B16A16_UINT ||
       tib->logical_format[0] == PIPE_FORMAT_R16G16B16X16_UINT);
   const bool rgba16i_or_x = tib->nr_samples == 1 &&
      (tib->logical_format[0] == PIPE_FORMAT_R16G16B16A16_SINT ||
       tib->logical_format[0] == PIPE_FORMAT_R16G16B16X16_SINT);
   /* API RGB32 surfaces use the four-lane RGBX physical family. The unused
    * lane has the same tile footprint and EOT protocol as RGBA32. */
   const bool rgbx32u = tib->nr_samples == 1 && tib->sample_size_B == 16 &&
      (tib->logical_format[0] == PIPE_FORMAT_R32G32B32A32_UINT ||
       tib->logical_format[0] == PIPE_FORMAT_R32G32B32X32_UINT) &&
      agx_tilebuffer_physical_format(tib, 0) == PIPE_FORMAT_R32_UINT;
   const bool rgbx32i = tib->nr_samples == 1 && tib->sample_size_B == 16 &&
      (tib->logical_format[0] == PIPE_FORMAT_R32G32B32A32_SINT ||
       tib->logical_format[0] == PIPE_FORMAT_R32G32B32X32_SINT) &&
      agx_tilebuffer_physical_format(tib, 0) == PIPE_FORMAT_R32_UINT;
   const bool rgbx32f = tib->nr_samples == 1 && tib->sample_size_B == 16 &&
      (tib->logical_format[0] == PIPE_FORMAT_R32G32B32A32_FLOAT ||
       tib->logical_format[0] == PIPE_FORMAT_R32G32B32X32_FLOAT) &&
      agx_tilebuffer_physical_format(tib, 0) == PIPE_FORMAT_R32_UINT;
   const bool rgb10a2u = tib->nr_samples == 1 && tib->sample_size_B == 8 &&
      tib->logical_format[0] == PIPE_FORMAT_R10G10B10A2_UINT &&
      agx_tilebuffer_physical_format(tib, 0) == PIPE_FORMAT_R16_UINT;
   const bool r11g11b10f = tib->nr_samples == 1 && tib->sample_size_B == 8 &&
      tib->logical_format[0] == PIPE_FORMAT_R11G11B10_FLOAT &&
      agx_tilebuffer_physical_format(tib, 0) ==
         PIPE_FORMAT_R11G11B10_FLOAT;

   return mrt != AGX_G17P_MRT_NONE || rgbx32u || rgbx32i || rgbx32f ||
          rgb10a2u || r11g11b10f ||
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
            tib->logical_format[0] == PIPE_FORMAT_R8G8B8X8_UINT ||
            tib->logical_format[0] == PIPE_FORMAT_R8G8B8A8_SNORM ||
            tib->logical_format[0] == PIPE_FORMAT_B8G8R8A8_SNORM ||
            tib->logical_format[0] == PIPE_FORMAT_R8_SRGB ||
            tib->logical_format[0] == PIPE_FORMAT_R8G8_SRGB ||
            tib->logical_format[0] == PIPE_FORMAT_R8G8B8A8_SRGB ||
            tib->logical_format[0] == PIPE_FORMAT_B8G8R8A8_SRGB ||
            tib->logical_format[0] == PIPE_FORMAT_R8G8B8A8_UINT ||
            tib->logical_format[0] == PIPE_FORMAT_B8G8R8X8_UINT ||
            tib->logical_format[0] == PIPE_FORMAT_B8G8R8A8_UINT ||
            tib->logical_format[0] == PIPE_FORMAT_R8G8B8X8_SINT ||
            tib->logical_format[0] == PIPE_FORMAT_R8G8B8A8_SINT ||
            tib->logical_format[0] == PIPE_FORMAT_B8G8R8X8_SINT ||
            tib->logical_format[0] == PIPE_FORMAT_B8G8R8A8_SINT ||
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
            pair_r32u || scalar_r8u || scalar_r8i || rgba16u_or_x ||
            rgba16i_or_x));
}

enum agx_g17p_mrt_layout
agx_g17p_classify_mrt(const struct agx_tilebuffer_layout *tib)
{
   if (!tib || tib->nr_samples != 1 || tib->spilled[0] || tib->spilled[1] ||
       tib->_offset_B[0] != 0)
      return AGX_G17P_MRT_NONE;

   /* Native G17P four-target fragment/BG/EOT constructors cover identical
    * raw 32-bit targets in one imageblock: 0/16/32/48 or 0/4/8/12. */
   const enum pipe_format format = tib->logical_format[0];
   const bool rgba32 = format == PIPE_FORMAT_R32G32B32A32_FLOAT ||
                       format == PIPE_FORMAT_R32G32B32A32_UINT ||
                       format == PIPE_FORMAT_R32G32B32A32_SINT;
   if (rgba32 || format == PIPE_FORMAT_R32_FLOAT) {
      const unsigned stride = rgba32 ? 16 : 4;
      bool quad = !tib->layered && tib->sample_size_B == 4 * stride;
      for (unsigned rt = 0; rt < AGX_MAX_RENDER_TARGETS; ++rt) {
         quad &= rt < 4 ? tib->logical_format[rt] == format &&
                             !tib->spilled[rt] && tib->_offset_B[rt] == rt * stride
                       : tib->logical_format[rt] == PIPE_FORMAT_NONE;
      }
      if (quad)
         return rgba32 ? AGX_G17P_MRT_RGBA32_QUAD : AGX_G17P_MRT_R32_QUAD;
   }

   for (unsigned rt = 2; rt < AGX_MAX_RENDER_TARGETS; ++rt) {
      if (tib->logical_format[rt] != PIPE_FORMAT_NONE)
         return AGX_G17P_MRT_NONE;
   }

   if (tib->logical_format[0] == PIPE_FORMAT_R16G16B16A16_FLOAT &&
       tib->logical_format[1] == PIPE_FORMAT_R8_UNORM &&
       tib->_offset_B[1] == 8 && tib->sample_size_B == 16)
      return AGX_G17P_MRT_WBOIT;

   const bool rt0_rgba8 =
      tib->logical_format[0] == PIPE_FORMAT_R8G8B8A8_UNORM ||
      tib->logical_format[0] == PIPE_FORMAT_B8G8R8A8_UNORM ||
      tib->logical_format[0] == PIPE_FORMAT_R8G8B8X8_UNORM ||
      tib->logical_format[0] == PIPE_FORMAT_B8G8R8X8_UNORM;
   const bool rt1_rgba8 =
      tib->logical_format[1] == PIPE_FORMAT_R8G8B8A8_UNORM ||
      tib->logical_format[1] == PIPE_FORMAT_B8G8R8A8_UNORM ||
      tib->logical_format[1] == PIPE_FORMAT_R8G8B8X8_UNORM ||
      tib->logical_format[1] == PIPE_FORMAT_B8G8R8X8_UNORM;

   /* These logical channel orders and alpha-ignored variants all use the
    * same four-lane U8 tile representation. Swizzle and X-channel discard
    * live in the per-attachment PBE descriptor, outside the shared MRT EOT
    * program. */
   if (rt0_rgba8 && rt1_rgba8 &&
       tib->_offset_B[1] == 4 && tib->sample_size_B == 8)
      return AGX_G17P_MRT_RGBA8;

   if (tib->logical_format[0] == PIPE_FORMAT_R11G11B10_FLOAT &&
       tib->logical_format[1] == PIPE_FORMAT_R11G11B10_FLOAT &&
       tib->_offset_B[1] == 4 && tib->sample_size_B == 8)
      return AGX_G17P_MRT_R11G11B10;

   return AGX_G17P_MRT_NONE;
}
