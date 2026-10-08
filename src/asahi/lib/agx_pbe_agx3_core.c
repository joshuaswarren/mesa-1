/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_pbe_agx3_core.h"
#include "agx_bytes.h"

#include <string.h>

#define AGX_PBE_DIMENSION_2D 2u
#define AGX_PBE_DIMENSION_2D_ARRAY 3u
#define AGX_PBE_DIMENSION_2D_MULTISAMPLE 4u
#define AGX_PBE_DIMENSION_3D 5u
#define AGX_PBE_DIMENSION_2D_ARRAY_MULTISAMPLE 8u
#define AGX_PBE_SAMPLE_SELECTOR_SHIFT 56u
#define AGX_PBE_LAYOUT_LINEAR 0u
#define AGX_PBE_LAYOUT_TWIDDLED 1u
#define AGX_PBE_LAYOUT_GPU 2u
/* G17P Interchange, Gen9 2D only: the Linear row-stride field, in macroblocks
 * per row (HAL200 PBEStateEncoderGen9::StateFields handles layouts 0 and 3 on
 * one path; Texture::getStride returns macroblocks for Interchange). */
#define AGX_PBE_LAYOUT_INTERCHANGE 3u

static bool
channel_is_uncompressed(uint8_t channels)
{
   switch (channels) {
   case 0x00: /* R8 */
   case 0x09: /* R16 */
   case 0x0a: /* R8G8 */
   case 0x0b: /* R5G6B5 */
   case 0x0c: /* R4G4B4A4 */
   case 0x0d: /* A1R5G5B5 */
   case 0x0e: /* R5G5B5A1 */
   case 0x21: /* R32 */
   case 0x23: /* R16G16 */
   case 0x25: /* R11G11B10 */
   case 0x26: /* R10G10B10A2 */
   case 0x27: /* R9G9B9E5 */
   case 0x28: /* R8G8B8A8 */
   case 0x31: /* R32G32 */
   case 0x32: /* R16G16B16A16 */
   case 0x38: /* R32G32B32A32 */
      return true;
   default:
      return false;
   }
}

static bool
generation_geometry(uint32_t generation, unsigned *height_shift,
                    unsigned *dimension_bits)
{
   switch (generation) {
   case AGX_PBE_AGX3_GENERATION_9:
      *height_shift = 38;
      *dimension_bits = 14;
      return true;
   case AGX_PBE_AGX3_GENERATION_10:
      *height_shift = 39;
      *dimension_bits = 15;
      return true;
   default:
      *height_shift = 0;
      *dimension_bits = 0;
      return false;
   }
}

static bool
fields_valid(uint32_t generation,
             const struct agx_pbe_agx3_2d_fields *fields)
{
   unsigned height_shift, dimension_bits;
   if (!fields ||
       !generation_geometry(generation, &height_shift, &dimension_bits))
      return false;

   const uint32_t maximum_dimension = UINT32_C(1) << dimension_bits;
   const bool linear = fields->layout == AGX_PBE_LAYOUT_LINEAR;
   const bool interchange = fields->layout == AGX_PBE_LAYOUT_INTERCHANGE;
   const bool strided = linear || interchange;
   const bool stride_valid =
      strided ? fields->stride_B >= 16 && !(fields->stride_B & 0xf) &&
                   fields->stride_B - 16 < (UINT32_C(1) << 21)
              : fields->stride_B == 0;
   const bool address_valid =
      fields->buffer && !(fields->buffer & 0xf) &&
      !(fields->buffer >> (strided ? 44 : 48));
   const bool compression_valid = fields->compressed
      ? generation == AGX_PBE_AGX3_GENERATION_9 &&
           (fields->layout == AGX_PBE_LAYOUT_GPU || interchange) &&
           fields->acceleration_buffer &&
           !(fields->acceleration_buffer & 0xf) &&
           !(fields->acceleration_buffer >> 48)
      : !fields->acceleration_buffer;
   return (!strided || generation == AGX_PBE_AGX3_GENERATION_9) &&
          (!fields->srgb || generation == AGX_PBE_AGX3_GENERATION_9) &&
          (strided || fields->layout == AGX_PBE_LAYOUT_TWIDDLED ||
           fields->layout == AGX_PBE_LAYOUT_GPU) &&
          channel_is_uncompressed(fields->channels) && fields->type <= 5 &&
          fields->swizzle_r <= 3 && fields->swizzle_g <= 3 &&
          fields->swizzle_b <= 3 && fields->swizzle_a <= 3 && fields->width &&
          fields->width <= maximum_dimension && fields->height &&
          fields->height <= maximum_dimension && address_valid && stride_valid &&
          compression_valid;
}

bool
agx_pbe_agx3_2d_pack(uint32_t generation,
                     const struct agx_pbe_agx3_2d_fields *fields,
                     struct agx_pbe_agx3_descriptor *out)
{
   if (out)
      memset(out, 0, sizeof(*out));

   if (!out || !fields_valid(generation, fields))
      return false;

   unsigned height_shift, dimension_bits;
   if (!generation_geometry(generation, &height_shift, &dimension_bits))
      return false;

   uint64_t word0 = AGX_PBE_DIMENSION_2D;
   word0 |= (uint64_t)fields->layout << 4;
   word0 |= (uint64_t)fields->channels << 6;
   word0 |= (uint64_t)fields->type << 13;
   word0 |= (uint64_t)fields->swizzle_r << 16;
   word0 |= (uint64_t)fields->swizzle_g << 18;
   word0 |= (uint64_t)fields->swizzle_b << 20;
   word0 |= (uint64_t)fields->swizzle_a << 22;
   word0 |= (uint64_t)(fields->width - 1u) << 24;
   word0 |= (uint64_t)(fields->height - 1u) << height_shift;
   if (fields->compressed)
      word0 |= UINT64_C(1) << 59;

   /* Gen9 StateFields (0x34f188..0x34f1f8) only writes word1 bit60 for
    * array/type-flagged or kind3/4 images. This codec admits ordinary kind2,
    * one layer only: its implicit extent and bit60 remain zero, even when
    * the allocator padded that layer to a page. The legacy allocator flag
    * is not that Gen9 field. Preserve the existing Gen10 encoding.
    */
   const bool layer_flag = generation == AGX_PBE_AGX3_GENERATION_10 &&
                           fields->page_aligned_layers;
   uint64_t word1 = fields->buffer >> 4;
   if (fields->layout == AGX_PBE_LAYOUT_LINEAR ||
       fields->layout == AGX_PBE_LAYOUT_INTERCHANGE)
      word1 |= (uint64_t)(fields->stride_B - 16) << 40;
   else
      word1 |= (uint64_t)layer_flag << 60;

   /* PBEStateEncoderGen9<kAGXRevHAL200>::StateFields sets word1 bit 61 in
    * every layout whenever the texture format's gamma encoding (format
    * descriptor +0x54) is nonzero, except for a few extended pixel formats
    * that select bit 62 as well (+0x22c..+0x2dc). TextureGen4's
    * encodeImageReadState turns the same field into the texture descriptor's
    * sRGB bits, and AGX2's PBE keeps its sRGB store conversion at this bit. */
   word1 |= (uint64_t)fields->srgb << 61;
   if (fields->compressed)
      word1 |= UINT64_C(1) << 63;
   agx_store_le64(&out->bytes[0], word0);
   agx_store_le64(&out->bytes[8], word1);
   if (fields->compressed)
      agx_store_le64(&out->bytes[16], fields->acceleration_buffer >> 4);
   out->length = AGX_PBE_AGX3_DESCRIPTOR_LENGTH;
   return true;
}

/* PBEStateEncoderGen9::StateFields 0x34ef68 and 0x34f00c..0x34f028:
 * level | (levels - 1) selects the extended descriptor. The level is
 * word2[47:44], levels-1 is word2[51:48], with word0 bit58 and word1 bit63
 * marking the extension. Using the native chain fields preserves the
 * allocation's mip tail addressing rather than pretending each level is a
 * standalone image.
 *
 * In the macOS 27 AGXMetalG17P the deposit is StateFields+0x1ec..+0x21c. It
 * reads only the level arguments, and nothing later on the tiled 2D-array or
 * 3D path rewrites word2 (their layer count lands in word1 at +0x374), so
 * layered targets take the same extension.
 */
static void
mark_mip_chain(struct agx_pbe_agx3_descriptor *out, uint32_t level,
               uint32_t levels)
{
   if (!level && levels == 1)
      return;

   agx_store_le64(&out->bytes[0], agx_load_le64(&out->bytes[0]) | (UINT64_C(1) << 58));
   agx_store_le64(&out->bytes[8], agx_load_le64(&out->bytes[8]) | (UINT64_C(1) << 63));
   agx_store_le64(&out->bytes[16],
              (agx_load_le64(&out->bytes[16]) & ((UINT64_C(1) << 44) - 1)) |
              ((uint64_t)level << 44) | ((uint64_t)(levels - 1) << 48));
}

/* PBEStateEncoderGen9<kAGXRevHAL200>::StateFields (AGXMetalG17P, macOS 27)
 * deposits a layer count at word1[59:44] and page-aligned layers at bit60 for
 * every layered write target, keeping the 2D geometry. A texture's own write
 * descriptor (WriteType 1, from TextureGen4::texBaseAddressesUpdated) takes
 * arrayLength - 1 for an array dimension whatever the length (+0x32c..+0x374),
 * so a one-layer array is still dimension 3, and depth - 1 for a 3D texture
 * (+0x4ac, reached from the kind-3 dimension select at +0x4a4).
 */
static bool
pack_layered(uint32_t generation, const struct agx_pbe_agx3_2d_fields *fields,
             unsigned dimension, uint32_t layers, uint32_t level,
             uint32_t levels, struct agx_pbe_agx3_descriptor *out)
{
   if (out)
      memset(out, 0, sizeof(*out));

   /* The exact HAL200 producer keeps layered targets tiled in this packet.
    * Linear stride and Gen10's shifted mode fields remain independently
    * gated. */
   if (generation != AGX_PBE_AGX3_GENERATION_9 || !fields ||
       fields->compressed ||
       fields->layout == AGX_PBE_LAYOUT_LINEAR || layers < 1 ||
       layers > UINT16_MAX + 1u || !levels || levels > 16 ||
       level >= levels ||
       !agx_pbe_agx3_2d_pack(generation, fields, out))
      return false;

   uint64_t word0 = agx_load_le64(&out->bytes[0]);
   uint64_t word1 = agx_load_le64(&out->bytes[8]);
   word0 = (word0 & ~UINT64_C(0xf)) | dimension;
   word1 |= (uint64_t)(layers - 1) << 44;
   word1 |= (uint64_t)fields->page_aligned_layers << 60;
   agx_store_le64(&out->bytes[0], word0);
   agx_store_le64(&out->bytes[8], word1);
   mark_mip_chain(out, level, levels);
   return true;
}

bool
agx_pbe_agx3_2d_array_pack(
   uint32_t generation, const struct agx_pbe_agx3_2d_fields *fields,
   uint32_t layers, struct agx_pbe_agx3_descriptor *out)
{
   return pack_layered(generation, fields, AGX_PBE_DIMENSION_2D_ARRAY, layers,
                       0, 1, out);
}

bool
agx_pbe_agx3_2d_array_mip_pack(
   uint32_t generation, const struct agx_pbe_agx3_2d_fields *fields,
   uint32_t layers, uint32_t level, uint32_t levels,
   struct agx_pbe_agx3_descriptor *out)
{
   return pack_layered(generation, fields, AGX_PBE_DIMENSION_2D_ARRAY, layers,
                       level, levels, out);
}

bool
agx_pbe_agx3_3d_mip_pack(
   uint32_t generation, const struct agx_pbe_agx3_2d_fields *fields,
   uint32_t depth, uint32_t level, uint32_t levels,
   struct agx_pbe_agx3_descriptor *out)
{
   return pack_layered(generation, fields, AGX_PBE_DIMENSION_3D, depth, level,
                       levels, out);
}

bool
agx_pbe_agx3_2d_multisample_pack(
   uint32_t generation, const struct agx_pbe_agx3_2d_fields *fields,
   uint32_t samples, struct agx_pbe_agx3_descriptor *out)
{
   if (out)
      memset(out, 0, sizeof(*out));

   /* A multisample colour target is never linear and never mipmapped: the
    * end-of-tile store addresses it through the sample-expanded twiddled
    * layout. Refuse 8x and every other count: the encoder's selector covers
    * them, but no tilebuffer geometry does. Gen10 has its own encoder and
    * its own witness to recover, so it is refused here too.
    */
   if (generation != AGX_PBE_AGX3_GENERATION_9 || !fields ||
       fields->compressed ||
       (samples != 2 && samples != 4) ||
       fields->layout == AGX_PBE_LAYOUT_LINEAR ||
       !agx_pbe_agx3_2d_pack(generation, fields, out))
      return false;

   /* PBEStateEncoderGen9 clears word0[59:53] before depositing the selector,
    * so the two selector bits are the only ones this adds on top of the
    * single-sample packet.
    */
   const uint64_t selector = (uint64_t)((samples >> 2) & 3u);
   uint64_t word0 = agx_load_le64(&out->bytes[0]);
   word0 = (word0 & ~UINT64_C(0xf)) | AGX_PBE_DIMENSION_2D_MULTISAMPLE;
   word0 |= selector << AGX_PBE_SAMPLE_SELECTOR_SHIFT;
   agx_store_le64(&out->bytes[0], word0);
   return true;
}

bool
agx_pbe_agx3_2d_array_multisample_pack(
   uint32_t generation, const struct agx_pbe_agx3_2d_fields *fields,
   uint32_t layers, uint32_t samples,
   struct agx_pbe_agx3_descriptor *out)
{
   if (out)
      memset(out, 0, sizeof(*out));

   /* HAL200's array+multisample path selects dimension 8 while retaining the
    * layered word1 fields and the ordinary multisample selector. */
   if ((samples != 2 && samples != 4) ||
       !pack_layered(generation, fields,
                     AGX_PBE_DIMENSION_2D_ARRAY_MULTISAMPLE, layers, 0, 1,
                     out))
      return false;

   uint64_t word0 = agx_load_le64(&out->bytes[0]);
   word0 |= (uint64_t)((samples >> 2) & 3u)
            << AGX_PBE_SAMPLE_SELECTOR_SHIFT;
   agx_store_le64(&out->bytes[0], word0);
   return true;
}

bool
agx_pbe_agx3_2d_mip_pack(
   uint32_t generation, const struct agx_pbe_agx3_2d_fields *fields,
   uint32_t level, uint32_t levels, struct agx_pbe_agx3_descriptor *out)
{
   if (out)
      memset(out, 0, sizeof(*out));
   if (generation != AGX_PBE_AGX3_GENERATION_9 || !levels || levels > 16 ||
       (fields && fields->layout == AGX_PBE_LAYOUT_LINEAR &&
        (level || levels != 1)) ||
       level >= levels || !agx_pbe_agx3_2d_pack(generation, fields, out))
      return false;

   mark_mip_chain(out, level, levels);
   return true;
}

bool
agx_pbe_agx3_2d_unpack(uint32_t generation, const void *bytes, size_t size,
                       struct agx_pbe_agx3_2d_fields *fields)
{
   if (fields)
      memset(fields, 0, sizeof(*fields));

   unsigned height_shift, dimension_bits;
   if (!bytes || !fields || size != AGX_PBE_AGX3_DESCRIPTOR_LENGTH ||
       !generation_geometry(generation, &height_shift, &dimension_bits))
      return false;

   const uint8_t *raw = bytes;
   const uint64_t word0 = agx_load_le64(&raw[0]);
   const uint64_t word1 = agx_load_le64(&raw[8]);
   const uint64_t word2 = agx_load_le64(&raw[16]);
   const uint64_t word3 = agx_load_le64(&raw[24]);
   const uint64_t word0_mask =
      generation == AGX_PBE_AGX3_GENERATION_9
         ? (UINT64_C(1) << 52) - 1
         : (UINT64_C(1) << 54) - 1;

   const uint8_t layout = (word0 >> 4) & 0x3;
   const bool compressed = generation == AGX_PBE_AGX3_GENERATION_9 &&
                           (word0 & (UINT64_C(1) << 59));
   const uint64_t srgb_bit = generation == AGX_PBE_AGX3_GENERATION_9
      ? UINT64_C(1) << 61 : 0;
   const uint64_t word1_mask = layout == AGX_PBE_LAYOUT_LINEAR
      ? ((UINT64_C(1) << 61) - 1) | srgb_bit
      : layout == AGX_PBE_LAYOUT_INTERCHANGE
      ? ((UINT64_C(1) << 61) - 1) | srgb_bit |
           (compressed ? UINT64_C(1) << 63 : 0)
      : ((UINT64_C(1) << 44) - 1) | srgb_bit |
           (compressed ? UINT64_C(1) << 63 : 0) |
           (generation == AGX_PBE_AGX3_GENERATION_10 ? UINT64_C(1) << 60 : 0);
   const uint64_t permitted_word0 =
      word0_mask | (compressed ? UINT64_C(1) << 59 : 0);
   if ((word0 & 0xf) != AGX_PBE_DIMENSION_2D ||
       (word0 & ~permitted_word0) || (word1 & ~word1_mask) ||
       (compressed ? (word2 & ~((UINT64_C(1) << 44) - 1)) : word2) ||
       word3)
      return false;

   fields->layout = layout;
   fields->channels = (word0 >> 6) & 0x7f;
   fields->type = (word0 >> 13) & 0x7;
   fields->swizzle_r = (word0 >> 16) & 0x3;
   fields->swizzle_g = (word0 >> 18) & 0x3;
   fields->swizzle_b = (word0 >> 20) & 0x3;
   fields->swizzle_a = (word0 >> 22) & 0x3;
   fields->width = ((word0 >> 24) & ((UINT64_C(1) << dimension_bits) - 1)) + 1;
   fields->height =
      ((word0 >> height_shift) & ((UINT64_C(1) << dimension_bits) - 1)) + 1;
   fields->srgb = (word1 & srgb_bit) != 0;
   fields->compressed = compressed;
   if (compressed)
      fields->acceleration_buffer = word2 << 4;
   if (layout == AGX_PBE_LAYOUT_LINEAR || layout == AGX_PBE_LAYOUT_INTERCHANGE) {
      fields->buffer = (word1 & ((UINT64_C(1) << 40) - 1)) << 4;
      fields->stride_B = ((word1 >> 40) & ((UINT64_C(1) << 21) - 1)) + 16;
   } else {
      fields->buffer = (word1 & ((UINT64_C(1) << 44) - 1)) << 4;
      fields->page_aligned_layers = (word1 >> 60) & 1;
   }

   if (!fields_valid(generation, fields)) {
      memset(fields, 0, sizeof(*fields));
      return false;
   }

   return true;
}
