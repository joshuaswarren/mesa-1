/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_ppp_hal300_depth_stencil_core.h"
#include "agx_bytes.h"

#include <string.h>

bool
agx_ppp_hal300_depth_stencil_core_supported(enum agx_chip chip)
{
   switch (chip) {
   case AGX_CHIP_G17G:
   case AGX_CHIP_G17S:
   case AGX_CHIP_G17C:
      return true;

   default:
      return false;
   }
}

bool
agx_ppp_hal300_depth_stencil_core_pack(
   enum agx_chip chip,
   const struct agx_ppp_hal300_depth_stencil_fields *fields,
   struct agx_ppp_hal300_depth_stencil_core *out)
{
   if (out)
      memset(out, 0, sizeof(*out));

   if (!out || !fields ||
       !agx_ppp_hal300_depth_stencil_core_supported(chip))
      return false;

   agx_store_le32(
      &out->bytes[0], AGX_PPP_HAL300_DEPTH_STENCIL_HEADER);
   agx_store_le32(
      &out->bytes[4], fields->fragment_control_raw);
   agx_store_le32(
      &out->bytes[8], fields->fragment_front_face_raw);
   agx_store_le32(
      &out->bytes[12], fields->fragment_back_face_raw);
   agx_store_le32(
      &out->bytes[16], fields->fragment_front_stencil_raw);
   agx_store_le32(
      &out->bytes[20], fields->fragment_back_stencil_raw);
   out->length = AGX_PPP_HAL300_DEPTH_STENCIL_CORE_LENGTH;
   return true;
}

bool
agx_ppp_hal300_depth_stencil_core_unpack(
   enum agx_chip chip, const void *bytes, size_t size,
   struct agx_ppp_hal300_depth_stencil_fields *fields)
{
   if (fields)
      memset(fields, 0, sizeof(*fields));

   if (!bytes || !fields || size != AGX_PPP_HAL300_DEPTH_STENCIL_CORE_LENGTH ||
       !agx_ppp_hal300_depth_stencil_core_supported(chip))
      return false;

   const uint8_t *raw = bytes;
   if (agx_load_le32(raw) !=
       AGX_PPP_HAL300_DEPTH_STENCIL_HEADER)
      return false;

   fields->fragment_control_raw =
      agx_load_le32(&raw[4]);
   fields->fragment_front_face_raw =
      agx_load_le32(&raw[8]);
   fields->fragment_back_face_raw =
      agx_load_le32(&raw[12]);
   fields->fragment_front_stencil_raw =
      agx_load_le32(&raw[16]);
   fields->fragment_back_stencil_raw =
      agx_load_le32(&raw[20]);
   return true;
}
