/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_ppp_viewport_payload_core.h"

#include <string.h>

static void
agx_ppp_viewport_store_le32(uint8_t *dst, uint32_t value)
{
   dst[0] = (uint8_t)value;
   dst[1] = (uint8_t)(value >> 8);
   dst[2] = (uint8_t)(value >> 16);
   dst[3] = (uint8_t)(value >> 24);
}

static void
agx_ppp_viewport_store_le64(uint8_t *dst, uint64_t value)
{
   agx_ppp_viewport_store_le32(dst, (uint32_t)value);
   agx_ppp_viewport_store_le32(dst + 4, (uint32_t)(value >> 32));
}

static uint32_t
agx_ppp_viewport_load_le32(const uint8_t *src)
{
   return (uint32_t)src[0] | ((uint32_t)src[1] << 8) |
          ((uint32_t)src[2] << 16) | ((uint32_t)src[3] << 24);
}

static uint64_t
agx_ppp_viewport_load_le64(const uint8_t *src)
{
   return (uint64_t)agx_ppp_viewport_load_le32(src) |
          ((uint64_t)agx_ppp_viewport_load_le32(src + 4) << 32);
}

static bool
agx_ppp_viewport_header_layout(enum agx_chip chip, uint32_t *base_header,
                               unsigned *count_shift)
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
      *base_header = AGX_PPP_VIEWPORT_PAYLOAD_GEN6_BASE_HEADER;
      *count_shift = 12;
      return true;

   case AGX_CHIP_G17G:
   case AGX_CHIP_G17S:
   case AGX_CHIP_G17C:
   case AGX_CHIP_G18G:
      *base_header = AGX_PPP_VIEWPORT_PAYLOAD_GEN7_BASE_HEADER;
      *count_shift = 14;
      return true;

   default:
      *base_header = 0;
      *count_shift = 0;
      return false;
   }
}

bool
agx_ppp_viewport_payload_core_supported(enum agx_chip chip)
{
   uint32_t base_header;
   unsigned count_shift;
   return agx_ppp_viewport_header_layout(chip, &base_header, &count_shift);
}

static size_t
agx_ppp_viewport_payload_length(unsigned viewport_count)
{
   return 8u + (32u * viewport_count);
}

bool
agx_ppp_viewport_payload_core_pack(
   enum agx_chip chip, const struct agx_ppp_viewport_payload_fields *fields,
   struct agx_ppp_viewport_payload_core *out)
{
   if (out)
      memset(out, 0, sizeof(*out));

   if (!out || !fields || !agx_ppp_viewport_payload_core_supported(chip) ||
       fields->viewport_count == 0 ||
       fields->viewport_count > AGX_PPP_VIEWPORT_PAYLOAD_MAX_VIEWPORTS)
      return false;

   uint32_t base_header;
   unsigned shift;
   if (!agx_ppp_viewport_header_layout(chip, &base_header, &shift))
      return false;

   const uint32_t header = base_header |
                           ((uint32_t)(fields->viewport_count - 1) << shift);
   size_t offset = 0;

   agx_ppp_viewport_store_le32(&out->bytes[offset], header);
   offset += 4;

   for (unsigned i = 0; i < fields->viewport_count; ++i) {
      agx_ppp_viewport_store_le64(&out->bytes[offset],
                                  fields->region_clip_raw[i]);
      offset += 8;
   }

   agx_ppp_viewport_store_le32(&out->bytes[offset],
                               fields->viewport_control_raw);
   offset += 4;

   for (unsigned i = 0; i < fields->viewport_count; ++i) {
      const struct agx_ppp_viewport_transform_raw *viewport =
         &fields->viewports[i];
      const uint32_t words[6] = {
         viewport->translate_x, viewport->scale_x,     viewport->translate_y,
         viewport->scale_y,     viewport->translate_z, viewport->scale_z,
      };

      for (unsigned w = 0; w < 6; ++w) {
         agx_ppp_viewport_store_le32(&out->bytes[offset], words[w]);
         offset += 4;
      }
   }

   if (offset != agx_ppp_viewport_payload_length(fields->viewport_count)) {
      memset(out, 0, sizeof(*out));
      return false;
   }

   out->length = (uint16_t)offset;
   return true;
}

bool
agx_ppp_viewport_payload_core_unpack(
   enum agx_chip chip, const void *bytes, size_t size,
   struct agx_ppp_viewport_payload_fields *fields)
{
   if (fields)
      memset(fields, 0, sizeof(*fields));

   if (!bytes || !fields || !agx_ppp_viewport_payload_core_supported(chip) ||
       size < sizeof(uint32_t))
      return false;

   const uint8_t *raw = bytes;
   uint32_t base_header;
   unsigned shift;
   if (!agx_ppp_viewport_header_layout(chip, &base_header, &shift))
      return false;

   const uint32_t count_mask = UINT32_C(0xf) << shift;
   const uint32_t header = agx_ppp_viewport_load_le32(raw);

   if ((header & ~count_mask) != base_header)
      return false;

   const unsigned viewport_count = ((header & count_mask) >> shift) + 1;
   if (size != agx_ppp_viewport_payload_length(viewport_count))
      return false;

   fields->viewport_count = (uint8_t)viewport_count;
   size_t offset = 4;

   for (unsigned i = 0; i < viewport_count; ++i) {
      fields->region_clip_raw[i] = agx_ppp_viewport_load_le64(&raw[offset]);
      offset += 8;
   }

   fields->viewport_control_raw = agx_ppp_viewport_load_le32(&raw[offset]);
   offset += 4;

   for (unsigned i = 0; i < viewport_count; ++i) {
      struct agx_ppp_viewport_transform_raw *viewport = &fields->viewports[i];
      uint32_t *const words[6] = {
         &viewport->translate_x, &viewport->scale_x,     &viewport->translate_y,
         &viewport->scale_y,     &viewport->translate_z, &viewport->scale_z,
      };

      for (unsigned w = 0; w < 6; ++w) {
         *words[w] = agx_ppp_viewport_load_le32(&raw[offset]);
         offset += 4;
      }
   }

   if (offset != size) {
      memset(fields, 0, sizeof(*fields));
      return false;
   }

   return true;
}
