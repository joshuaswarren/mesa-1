/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_vdm_barrier_core.h"

#include <string.h>

static void
agx_vdm_barrier_store_le32(uint8_t *dst, uint32_t value)
{
   dst[0] = (uint8_t)value;
   dst[1] = (uint8_t)(value >> 8);
   dst[2] = (uint8_t)(value >> 16);
   dst[3] = (uint8_t)(value >> 24);
}

static uint32_t
agx_vdm_barrier_load_le32(const uint8_t *src)
{
   return (uint32_t)src[0] | ((uint32_t)src[1] << 8) |
          ((uint32_t)src[2] << 16) | ((uint32_t)src[3] << 24);
}

bool
agx_vdm_barrier_core_supported(enum agx_chip chip)
{
   switch (chip) {
   case AGX_CHIP_G15G:
   case AGX_CHIP_G15S:
   case AGX_CHIP_G15C:
   case AGX_CHIP_G15D:
   case AGX_CHIP_G16G:
   case AGX_CHIP_G16S:
   case AGX_CHIP_G16C:
   case AGX_CHIP_G17P:
   case AGX_CHIP_G17G:
   case AGX_CHIP_G17S:
   case AGX_CHIP_G17C:
   /* 26A428 AGXMetalG18G RenderContext::memoryBarrier appends the same word
    * (0x219642398..3a0, hv/G18-RASTERIZER-DISCARD.md section 7). */
   case AGX_CHIP_G18G:
      return true;

   case AGX_CHIP_G13G:
   case AGX_CHIP_G13X:
   case AGX_CHIP_G14G:
   case AGX_CHIP_G14X:
   case AGX_CHIP_G15P:
   case AGX_CHIP_G16P:
   case AGX_CHIP_G17A:
   case AGX_CHIP_UNKNOWN:
   default:
      return false;
   }
}

bool
agx_vdm_barrier_core_pack(enum agx_chip chip, struct agx_vdm_barrier_core *out)
{
   if (out)
      memset(out, 0, sizeof(*out));

   if (!out || !agx_vdm_barrier_core_supported(chip))
      return false;

   agx_vdm_barrier_store_le32(out->bytes, AGX_VDM_BARRIER_CORE_WORD);
   out->length = AGX_VDM_BARRIER_CORE_LENGTH;
   return true;
}

bool
agx_vdm_barrier_core_unpack(enum agx_chip chip, const void *bytes, size_t size)
{
   if (!bytes || size != AGX_VDM_BARRIER_CORE_LENGTH ||
       !agx_vdm_barrier_core_supported(chip))
      return false;

   return agx_vdm_barrier_load_le32(bytes) == AGX_VDM_BARRIER_CORE_WORD;
}
