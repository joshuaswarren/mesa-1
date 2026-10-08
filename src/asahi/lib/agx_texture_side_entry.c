/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_texture_side_entry.h"

#include <limits.h>
#include <stddef.h>
#include <string.h>

#define AGX_TEXTURE_SIDE_ENTRY_MAX_EXACT_BINARY32_INTEGER UINT64_C(0x1000000)

static bool
agx_texture_side_entry_generation_valid(uint32_t generation)
{
   return generation == AGX_TEXTURE_SIDE_ENTRY_GENERATION_9 ||
          generation == AGX_TEXTURE_SIDE_ENTRY_GENERATION_10;
}

static bool
agx_texture_side_entry_power_of_two(uint64_t value)
{
   return value != 0 && (value & (value - 1)) == 0;
}

static bool
agx_texture_side_entry_align_up(uint64_t value, uint64_t alignment,
                                uint64_t *aligned)
{
   if (!aligned || !agx_texture_side_entry_power_of_two(alignment) ||
       value > UINT64_MAX - (alignment - 1))
      return false;

   *aligned = (value + alignment - 1) & ~(alignment - 1);
   return true;
}

/* Raw, lossless normal-form low-payload/high-auxiliary composition/decode. */
static bool
agx_texture_side_entry_compose(uint32_t payload, uint32_t auxiliary_heap_index,
                               uint64_t *packed)
{
   if (!packed)
      return false;

   *packed = (uint64_t)payload | ((uint64_t)auxiliary_heap_index << 32);
   return true;
}

bool
agx_texture_side_entry_decode(uint64_t packed,
                              struct agx_texture_side_entry_decoded *decoded)
{
   if (!decoded)
      return false;

   decoded->payload = (uint32_t)packed;
   decoded->auxiliary_heap_index = (uint32_t)(packed >> 32);
   return true;
}

static bool
agx_texture_side_entry_gen9_atomic_class_simple(
   enum agx_texture_side_entry_gen9_atomic_class layout_class)
{
   return layout_class == AGX_TEXTURE_SIDE_ENTRY_GEN9_ATOMIC_CLASS_1D ||
          layout_class == AGX_TEXTURE_SIDE_ENTRY_GEN9_ATOMIC_CLASS_2D;
}

static bool
agx_texture_side_entry_gen9_atomic_class_complex(
   enum agx_texture_side_entry_gen9_atomic_class layout_class)
{
   return layout_class == AGX_TEXTURE_SIDE_ENTRY_GEN9_ATOMIC_CLASS_3D ||
          layout_class == AGX_TEXTURE_SIDE_ENTRY_GEN9_ATOMIC_CLASS_1D_ARRAY ||
          layout_class == AGX_TEXTURE_SIDE_ENTRY_GEN9_ATOMIC_CLASS_2D_ARRAY;
}

/*
 * Partial Gen9 atomic-image payload packer.  This is kept separate so callers
 * cannot mistake recovered descriptor representation for execution support.
 * On failure, payload is cleared.
 */
static bool
agx_texture_side_entry_pack_gen9_atomic_payload(
   const struct agx_texture_side_entry_gen9_atomic_info *info,
   uint32_t *payload)
{
   if (!payload)
      return false;

   *payload = 0;
   if (!info || !info->eligible)
      return false;

   if (agx_texture_side_entry_gen9_atomic_class_simple(info->layout_class)) {
      const uint64_t shifted = info->level_offset_bytes >> 7;

      if (shifted > UINT32_MAX)
         return false;

      *payload = (uint32_t)shifted;
      return true;
   }

   if (!agx_texture_side_entry_gen9_atomic_class_complex(info->layout_class))
      return false;

   if (info->width_texels == 0 || info->width_texels > UINT32_MAX ||
       info->height_texels == 0 || info->height_texels > UINT32_MAX ||
       (info->format_element_bytes != 4 && info->format_element_bytes != 8) ||
       info->base_alignment_bytes < 128 ||
       !agx_texture_side_entry_power_of_two(info->base_alignment_bytes))
      return false;

   if (info->width_texels > UINT64_MAX / info->height_texels)
      return false;

   const uint64_t texels = info->width_texels * info->height_texels;
   const unsigned scale = info->format_element_bytes == 8 ? 3 : 2;
   if (texels > (UINT64_MAX >> scale))
      return false;

   uint64_t plane_size;
   if (!agx_texture_side_entry_align_up(texels << scale, UINT64_C(0x4000),
                                        &plane_size))
      return false;

   const bool high_alignment = info->base_alignment_bytes > 128;
   uint64_t high_source = info->mip_size_bytes;
   unsigned high_shift = 7;

   if (high_alignment) {
      if (high_source < plane_size)
         return false;

      high_source -= plane_size;
      high_shift = 14;
   }

   const uint64_t high = high_source >> high_shift;
   if (high > UINT16_MAX)
      return false;

   const bool level_aligned =
      (info->level_offset_bytes & UINT64_C(0x3fff)) == 0;
   uint64_t low;

   if (high_alignment && level_aligned) {
      if (info->mip_size_bytes < info->level_offset_bytes)
         return false;

      low = (info->mip_size_bytes - info->level_offset_bytes) >> 14;
   } else {
      uint64_t adjusted = 0;

      if (info->level_offset_bytes != 0) {
         if (info->level_offset_bytes < plane_size)
            return false;
         adjusted = info->level_offset_bytes - plane_size;
      }

      adjusted >>= 7;
      if (adjusted > UINT64_MAX - UINT64_C(0xae99))
         return false;
      low = adjusted + UINT64_C(0xae99);
   }

   if (low > UINT16_MAX)
      return false;

   *payload = ((uint32_t)high << 16) | (uint32_t)low;
   return true;
}

static bool
agx_texture_side_entry_pack_sparse_residency(
   const struct agx_texture_side_entry_sparse_residency_info *info,
   uint64_t *packed)
{
   if (!info || !packed || info->width_tiles == 0 || info->height_tiles == 0 ||
       info->width_tiles > AGX_TEXTURE_SIDE_ENTRY_MAX_EXACT_BINARY32_INTEGER ||
       info->height_tiles > AGX_TEXTURE_SIDE_ENTRY_MAX_EXACT_BINARY32_INTEGER)
      return false;

   const float width = (float)info->width_tiles;
   const float height = (float)info->height_tiles;
   uint32_t width_bits;
   uint32_t height_bits;

   _Static_assert(sizeof(width) == sizeof(width_bits), "binary32 is required");
   memcpy(&width_bits, &width, sizeof(width_bits));
   memcpy(&height_bits, &height, sizeof(height_bits));

   *packed = (uint64_t)width_bits | ((uint64_t)height_bits << 32);
   return true;
}

bool
agx_texture_side_entry_pack(const struct agx_texture_side_entry_info *info,
                            uint64_t *packed)
{
   if (!packed)
      return false;

   *packed = 0;
   if (!info || !agx_texture_side_entry_generation_valid(info->generation))
      return false;

   if (info->form == AGX_TEXTURE_SIDE_ENTRY_FORM_DIRECT) {
      if (info->kind !=
             AGX_TEXTURE_SIDE_ENTRY_KIND_TIER1_SPARSE_RESIDENCY ||
          info->auxiliary_heap_index != 0)
         return false;

      return agx_texture_side_entry_pack_sparse_residency(
         &info->value.sparse_residency, packed);
   }

   if (info->form != AGX_TEXTURE_SIDE_ENTRY_FORM_NORMAL)
      return false;

   uint32_t payload = 0;

   switch (info->kind) {
   case AGX_TEXTURE_SIDE_ENTRY_KIND_BUFFER:
      if (info->value.buffer_width_texels == 0 ||
          info->value.buffer_width_texels > UINT32_MAX)
         return false;
      payload = (uint32_t)info->value.buffer_width_texels;
      break;

   case AGX_TEXTURE_SIDE_ENTRY_KIND_ORDINARY_IMAGE:
      if (info->generation != AGX_TEXTURE_SIDE_ENTRY_GENERATION_10)
         return false;
      break;

   case AGX_TEXTURE_SIDE_ENTRY_KIND_GEN9_ATOMIC_IMAGE:
      if (info->generation != AGX_TEXTURE_SIDE_ENTRY_GENERATION_9 ||
          !agx_texture_side_entry_pack_gen9_atomic_payload(
             &info->value.gen9_atomic, &payload))
         return false;
      break;

   case AGX_TEXTURE_SIDE_ENTRY_KIND_TIER1_SPARSE_RESIDENCY:
      return false;

   case AGX_TEXTURE_SIDE_ENTRY_KIND_UNKNOWN:
   default:
      return false;
   }

   return agx_texture_side_entry_compose(payload, info->auxiliary_heap_index,
                                         packed);
}
