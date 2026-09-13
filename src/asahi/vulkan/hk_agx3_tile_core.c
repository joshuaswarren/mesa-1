/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_agx3_tile_core.h"

#include <string.h>

_Static_assert(HK_AGX3_TILE_MAX_LENGTH == AGX3_TILE_MAX_LENGTH,
               "Honeykrisp tile storage must match the shared codec");

static bool
hk_agx3_tile_target_encoding(enum agx_chip chip,
                             enum agx_isa_encoding *encoding)
{
   switch (chip) {
   case AGX_CHIP_G15G:
   case AGX_CHIP_G16G:
      *encoding = AGX_ISA_ENCODING_AGX3B;
      return true;
   case AGX_CHIP_G17S:
      *encoding = AGX_ISA_ENCODING_AGX3C;
      return true;
   default:
      return false;
   }
}

static bool
hk_agx3_tile_width(enum hk_agx3_tile_width in, enum agx3_tile_width *out)
{
   switch (in) {
   case HK_AGX3_TILE_WIDTH_8:
      *out = AGX3_TILE_W8;
      return true;
   case HK_AGX3_TILE_WIDTH_16:
      *out = AGX3_TILE_W16;
      return true;
   case HK_AGX3_TILE_WIDTH_32:
      *out = AGX3_TILE_W32;
      return true;
   default:
      return false;
   }
}

bool
hk_agx3_pack_tile_store(enum agx_chip chip,
                        const struct hk_agx3_tile_store_desc *desc,
                        struct hk_agx3_tile_instruction *out)
{
   if (!out)
      return false;
   memset(out, 0, sizeof(*out));

   enum agx_isa_encoding encoding;
   enum agx3_tile_width width;
   uint8_t length = 0;
   if (!desc || !hk_agx3_tile_target_encoding(chip, &encoding) ||
       !hk_agx3_tile_width(desc->width, &width) ||
       (encoding == AGX_ISA_ENCODING_AGX3B && desc->data >= 16) ||
       !agx3_tile_store(encoding, desc->data, width, desc->component_mask,
                        desc->tile_offset, AGX3_TILE_SAMPLE_1X, desc->wait_mask,
                        out->bytes, &length) ||
       length > sizeof(out->bytes)) {
      memset(out, 0, sizeof(*out));
      return false;
   }

   out->length = length;
   return true;
}

bool
hk_agx3_pack_tile_load(enum agx_chip chip,
                       const struct hk_agx3_tile_load_desc *desc,
                       struct hk_agx3_tile_instruction *out)
{
   if (!out)
      return false;
   memset(out, 0, sizeof(*out));

   enum agx_isa_encoding encoding;
   enum agx3_tile_width width;
   uint8_t length = 0;
   if (!desc || !hk_agx3_tile_target_encoding(chip, &encoding) ||
       !hk_agx3_tile_width(desc->width, &width) ||
       !agx3_tile_load(encoding, desc->dest, width, desc->slot,
                       desc->component_mask, desc->tile_offset,
                       AGX3_TILE_SAMPLE_1X, 0, desc->first, desc->last,
                       out->bytes, &length) ||
       length > sizeof(out->bytes)) {
      memset(out, 0, sizeof(*out));
      return false;
   }

   out->length = length;
   return true;
}
