/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_ppp_hal200_raster_core.h"
#include "agx_bytes.h"

#include <string.h>

bool
agx_ppp_hal200_raster_state_build(
   enum agx_chip chip, const struct agx_ppp_hal200_raster_fields *fields,
   struct agx_ppp_hal200_raster_state *out)
{
   if (out)
      memset(out, 0, sizeof(*out));
   if (chip != AGX_CHIP_G17P || !fields || !out)
      return false;

   out->fragment_control = AGX_PPP_HAL200_FRAGMENT_CONTROL_BASE;
   if (fields->scissor_enable) {
      out->fragment_control |= AGX_PPP_HAL200_FRAGMENT_CONTROL_SCISSOR;
      out->draw_state_scissor_flag = 1;
   }
   if (fields->depth_bias_enable)
      out->fragment_control |= AGX_PPP_HAL200_FRAGMENT_CONTROL_DEPTH_BIAS;

   agx_store_le16(&out->scissor[0], fields->max_x);
   agx_store_le16(&out->scissor[2], fields->min_x);
   agx_store_le16(&out->scissor[4], fields->max_y);
   agx_store_le16(&out->scissor[6], fields->min_y);
   agx_store_le32(&out->scissor[8], fields->min_z_raw);
   agx_store_le32(&out->scissor[12], fields->max_z_raw);

   agx_store_le32(&out->depth_bias[0], fields->depth_bias_raw);
   agx_store_le32(&out->depth_bias[4], fields->slope_scale_raw);
   agx_store_le32(&out->depth_bias[8], fields->clamp_raw);
   return true;
}
