/* SPDX-License-Identifier: MIT */
#pragma once

#include "asahi/lib/agx_g17p_draw_state_upload.h"

#define HK_G17P_BIND_GROUP_BYTES AGX_G17P_BIND_GROUP_BYTES
#define HK_G17P_VIEWPORT_RECORD_BYTES AGX_G17P_VIEWPORT_RECORD_BYTES
#define HK_G17P_VARYING_RECORD_BYTES AGX_G17P_VARYING_RECORD_BYTES
#define HK_G17P_VARYING_RECORD_OFFSET AGX_G17P_VARYING_RECORD_OFFSET
#define HK_G17P_BIND_STATE_BYTES AGX_G17P_BIND_STATE_BYTES
#define hk_g17p_bind_state_upload agx_g17p_bind_state_upload

static inline bool
hk_g17p_draw_state_range_valid(uint64_t compact, size_t size)
{
   return agx_g17p_draw_state_range_valid(compact, size);
}

static inline bool
hk_g17p_upload_bind_state(void *cpu, size_t capacity, uint64_t compact,
                          const uint8_t *bind_group, const uint8_t *viewport,
                          size_t viewport_size, uint32_t counts_32,
                          uint32_t counts_16,
                          struct hk_g17p_bind_state_upload *out)
{
   return agx_g17p_upload_bind_state(cpu, capacity, compact, bind_group,
                                     viewport, viewport_size, counts_32,
                                     counts_16, out);
}

static inline bool
hk_g17p_upload_draw_indices(void *cpu, size_t capacity, uint64_t compact,
                            size_t scissor_bytes, size_t depth_bias_bytes,
                            uint32_t *offset)
{
   return agx_g17p_upload_draw_indices(cpu, capacity, compact, scissor_bytes,
                                       depth_bias_bytes, offset);
}
