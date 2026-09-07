/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_native_descriptor.h"

#include "asahi/genxml/agx_pack.h"
#include "asahi/lib/agx_abi.h"
#include "asahi/lib/agx_pbe_agx3_core.h"

#include <string.h>

bool
agx_gallium_pack_native_texture(
   enum agx_descriptor_generation generation,
   struct agx_gallium_texture_descriptor *out,
   const struct AGX_TEXTURE *semantic)
{
   const unsigned descriptor_size =
      agx_gallium_texture_descriptor_size(generation);

   if (!out || !semantic || !descriptor_size)
      return false;

   memset(out, 0, descriptor_size);

   if (agx_gallium_descriptor_generation_is_agx2(generation)) {
      AGX_TEXTURE_pack(out->opaque, semantic);
      return true;
   }

   const bool multisampled =
      semantic->dimension == AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED ||
      semantic->dimension == AGX_TEXTURE_DIMENSION_2D_ARRAY_MULTISAMPLED;

   if (!agx_gallium_native_texture_supported(
          generation, multisampled, semantic->buffer_offset_sw))
      return false;

   if (multisampled) {
      unsigned samples;
      switch (semantic->samples) {
      case AGX_SAMPLE_COUNT_2:
         samples = 2;
         break;
      case AGX_SAMPLE_COUNT_4:
         samples = 4;
         break;
      case AGX_SAMPLE_COUNT_8:
         samples = 8;
         break;
      default:
         return false;
      }

      if (!agx_gallium_native_texture_sample_count_representable(
             generation, samples))
         return false;
   } else if (semantic->samples != AGX_SAMPLE_COUNT_2) {
      /* Sample Count has no 1x value, so zero/2x is the inert default for a
       * non-multisampled dimension.  Do not silently accept other selectors
       * where the hardware field has no defined meaning.
       */
      return false;
   }

   const bool has_levels = semantic->first_level || semantic->last_level ||
                           semantic->unk_mipmapped;
   const bool extended = semantic->extended || has_levels;

   if ((semantic->address & 0xf) || semantic->address >> 48 ||
       (semantic->compressed &&
        ((semantic->acceleration_buffer & 0xf) ||
         semantic->acceleration_buffer >> 48)))
      return false;

   if (generation == AGX_DESCRIPTOR_GENERATION_9) {
      if (!semantic->width || semantic->width > 16384 || !semantic->height ||
          semantic->height > 16384)
         return false;

      struct AGX_TEXTURE_GEN9 cfg = {
         .dimension = semantic->dimension,
         .layout = semantic->layout,
         .channels = semantic->channels,
         .type = semantic->type,
         .swizzle_r = semantic->swizzle_r,
         .swizzle_g = semantic->swizzle_g,
         .swizzle_b = semantic->swizzle_b,
         .swizzle_a = semantic->swizzle_a,
         .width = semantic->width,
         .height = semantic->height,
         .samples = semantic->samples,
         .has_levels = has_levels,
         .state_control = semantic->mode | (semantic->compression << 2),
         .address = semantic->address,
         .srgb = semantic->srgb,
         .srgb_2_channel = semantic->srgb_2_channel,
         .stride = semantic->layout == AGX_LAYOUT_LINEAR ? semantic->stride : 0,
         .depth = semantic->layout == AGX_LAYOUT_LINEAR ? 1 : semantic->depth,
         .page_aligned_layers = semantic->page_aligned_layers,
         .extended = extended,
         .acceleration_buffer =
            semantic->compressed && semantic->extended
               ? semantic->acceleration_buffer
               : 0,
         .depth_linear = semantic->layout == AGX_LAYOUT_LINEAR &&
                               semantic->extended
                            ? semantic->depth_linear
                            : 1,
         .layer_stride_linear = semantic->layout == AGX_LAYOUT_LINEAR &&
                                       semantic->extended
                                    ? semantic->layer_stride_linear
                                    : 0,
         .first_level = semantic->first_level,
         .last_level = semantic->last_level,
      };

      AGX_TEXTURE_GEN9_pack(out->opaque, &cfg);
      return true;
   }

   if (generation == AGX_DESCRIPTOR_GENERATION_10) {
      if (!semantic->width || semantic->width > 32768 || !semantic->height ||
          semantic->height > 32768)
         return false;

      struct AGX_TEXTURE_GEN10 cfg = {
         .dimension = semantic->dimension,
         .layout = semantic->layout,
         .channels = semantic->channels,
         .type = semantic->type,
         .swizzle_r = semantic->swizzle_r,
         .swizzle_g = semantic->swizzle_g,
         .swizzle_b = semantic->swizzle_b,
         .swizzle_a = semantic->swizzle_a,
         .width = semantic->width,
         .height = semantic->height,
         .samples = semantic->samples,
         .unk_mipmapped = semantic->unk_mipmapped,
         .compressed = semantic->compressed,
         .image_mode_enable = semantic->mode != AGX_IMAGE_MODE_NORMAL,
         .image_mode = semantic->mode,
         .address = semantic->address,
         .srgb = semantic->srgb,
         .srgb_2_channel = semantic->srgb_2_channel,
         .stride = semantic->layout == AGX_LAYOUT_LINEAR ? semantic->stride : 0,
         .depth = semantic->layout == AGX_LAYOUT_LINEAR ? 1 : semantic->depth,
         .page_aligned_layers = semantic->page_aligned_layers,
         .extended = extended,
         .acceleration_buffer =
            semantic->compressed && semantic->extended
               ? semantic->acceleration_buffer
               : 0,
         .depth_linear = semantic->layout == AGX_LAYOUT_LINEAR &&
                               semantic->extended
                            ? semantic->depth_linear
                            : 1,
         .layer_stride_linear = semantic->layout == AGX_LAYOUT_LINEAR &&
                                       semantic->extended
                                    ? semantic->layer_stride_linear
                                    : 0,
         .first_level = semantic->first_level,
         .last_level = semantic->last_level,
      };

      AGX_TEXTURE_GEN10_pack(out->opaque, &cfg);
      return true;
   }

   return false;
}

bool
agx_gallium_set_null_native_texture(
   enum agx_descriptor_generation generation,
   struct agx_gallium_texture_descriptor *out)
{
   const struct AGX_TEXTURE semantic = {
      AGX_TEXTURE_header,
      .layout = AGX_LAYOUT_TWIDDLED,
      .channels = AGX_CHANNELS_R8,
      .type = AGX_TEXTURE_TYPE_UNORM,
      .swizzle_r = AGX_CHANNEL_0,
      .swizzle_g = AGX_CHANNEL_0,
      .swizzle_b = AGX_CHANNEL_0,
      .swizzle_a = AGX_CHANNEL_0,
      .address = AGX_ZERO_PAGE_ADDRESS,
   };

   return agx_gallium_pack_native_texture(generation, out, &semantic);
}

/* Map the descriptor-generation enum onto the shared codec's numeric
 * generation. Zero means "no recovered AGX3 encoding". */
static uint32_t
agx_gallium_pbe_agx3_generation(enum agx_descriptor_generation generation)
{
   switch (generation) {
   case AGX_DESCRIPTOR_GENERATION_9:
      return AGX_PBE_AGX3_GENERATION_9;
   case AGX_DESCRIPTOR_GENERATION_10:
      return AGX_PBE_AGX3_GENERATION_10;
   default:
      return 0;
   }
}

bool
agx_gallium_pack_native_pbe(
   enum agx_descriptor_generation generation,
   struct agx_gallium_pbe_descriptor *out,
   const struct AGX_PBE *legacy)
{
   const unsigned descriptor_size =
      agx_gallium_pbe_descriptor_size(generation);

   if (!out || !legacy || !descriptor_size)
      return false;

   memset(out, 0, descriptor_size);
   if (!agx_gallium_native_pbe_supported(generation))
      return false;

   if (agx_gallium_descriptor_generation_is_agx2(generation)) {
      AGX_PBE_pack(out->opaque, legacy);
      return true;
   }

   const uint32_t agx3_generation =
      agx_gallium_pbe_agx3_generation(generation);
   if (!agx3_generation)
      return false;

   /* Exactly the subset agx_pbe_agx3_core.h documents as recovered, and the
    * same gate hk_pack_native_pbe() applies field for field -- the one that
    * already runs on this hardware through Honeykrisp. Every other axis
    * (arrays, multisample, compression, sRGB-on-store, image mode, the
    * extended word, and the software addressing sideband Gallium uses for
    * image atomics, EMRT and spilled render targets) has no proven Gen9/Gen10
    * placement, so refuse rather than drop the bits.
    */
   if (legacy->dimension != AGX_TEXTURE_DIMENSION_2D ||
       (legacy->layout == AGX_LAYOUT_LINEAR &&
        (agx3_generation != AGX_PBE_AGX3_GENERATION_9 || legacy->level != 0 ||
         legacy->levels != 1 || legacy->unk_mipmapped)) ||
       (legacy->layout != AGX_LAYOUT_LINEAR &&
        legacy->layout != AGX_LAYOUT_TWIDDLED &&
        legacy->layout != AGX_LAYOUT_GPU) ||
       legacy->unk_52 || legacy->rotate_90 || legacy->flip_vertical ||
       legacy->samples != AGX_SAMPLE_COUNT_2 || legacy->compressed ||
       legacy->mode != AGX_IMAGE_MODE_NORMAL || legacy->layers != 1 ||
       (agx3_generation != AGX_PBE_AGX3_GENERATION_9 &&
        (legacy->level != 0 || legacy->levels != 1 ||
         legacy->unk_mipmapped)) ||
       legacy->srgb || legacy->extended ||
       (legacy->layout == AGX_LAYOUT_LINEAR
           ? legacy->stride < 12 || legacy->stride > UINT32_MAX - 4
           : legacy->stride != 0) ||
       legacy->acceleration_buffer || legacy->depth_linear != 1 ||
       legacy->layer_stride_linear || legacy->level_offset_sw ||
       legacy->aligned_width_msaa_sw || legacy->sample_count_log2_sw ||
       legacy->tile_width_sw != 1 || legacy->tile_height_sw != 1 ||
       legacy->layer_stride_sw || legacy->buffer_offset_sw)
      return false;

   const struct agx_pbe_agx3_2d_fields fields = {
      .layout = legacy->layout,
      .channels = legacy->channels,
      .type = legacy->type,
      .swizzle_r = legacy->swizzle_r,
      .swizzle_g = legacy->swizzle_g,
      .swizzle_b = legacy->swizzle_b,
      .swizzle_a = legacy->swizzle_a,
      .width = legacy->width,
      .height = legacy->height,
      .buffer = legacy->buffer,
      .stride_B =
         legacy->layout == AGX_LAYOUT_LINEAR ? legacy->stride + 4 : 0,
      .page_aligned_layers = legacy->page_aligned_layers,
   };

   struct agx_pbe_agx3_descriptor packed;
   const bool valid =
      agx3_generation == AGX_PBE_AGX3_GENERATION_9
         ? agx_pbe_agx3_2d_mip_pack(agx3_generation, &fields, legacy->level,
                                    legacy->levels, &packed)
         : agx_pbe_agx3_2d_pack(agx3_generation, &fields, &packed);
   if (!valid)
      return false;

   static_assert(sizeof(packed.bytes) == AGX_GALLIUM_DESCRIPTOR_AGX3_SIZE,
                 "the AGX3 PBE descriptor is the AGX3 descriptor stride");
   memcpy(out->opaque, packed.bytes, sizeof(packed.bytes));
   return true;
}

bool
agx_gallium_set_null_native_pbe(
   enum agx_descriptor_generation generation,
   struct agx_gallium_pbe_descriptor *out)
{
   /* AGX_PBE_header already supplies the 1s for width, height, levels,
    * layers, depth_linear and the two software tile dimensions. */
   struct AGX_PBE legacy = {
      AGX_PBE_header,
      .layout = AGX_LAYOUT_TWIDDLED,
      .channels = AGX_CHANNELS_R8,
      .type = AGX_TEXTURE_TYPE_UNORM,
      .swizzle_r = AGX_CHANNEL_R,
      .swizzle_g = AGX_CHANNEL_R,
      .swizzle_b = AGX_CHANNEL_R,
      .swizzle_a = AGX_CHANNEL_R,
      .buffer = AGX_SCRATCH_PAGE_ADDRESS,
   };

   /* AGX3's recovered PBE geometry is 2D only, and the null descriptor
    * Honeykrisp already runs on this hardware is 2D. The AGX2 null
    * descriptor keeps the dimension it has always had.
    */
   if (agx_gallium_descriptor_generation_is_agx3(generation))
      legacy.dimension = AGX_TEXTURE_DIMENSION_2D;

   return agx_gallium_pack_native_pbe(generation, out, &legacy);
}
