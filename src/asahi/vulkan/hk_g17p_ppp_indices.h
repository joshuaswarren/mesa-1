/* SPDX-License-Identifier: MIT */
#pragma once

#include "asahi/lib/agx_g17p_draw_state_upload.h"

static inline bool
hk_g17p_ppp_indices_pack(size_t scissor_bytes, size_t depth_bias_bytes,
                         uint8_t output[8])
{
   return agx_g17p_ppp_indices_pack(scissor_bytes, depth_bias_bytes, output);
}
