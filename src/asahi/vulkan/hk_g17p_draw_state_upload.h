/* SPDX-License-Identifier: MIT */
#pragma once

#include "asahi/lib/agx_g17p_draw_state_upload.h"

#define hk_g17p_bind_state_upload agx_g17p_bind_state_upload

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
                            unsigned viewport_count, uint32_t *offset)
{
   return agx_g17p_upload_draw_indices(cpu, capacity, compact, scissor_bytes,
                                       depth_bias_bytes, viewport_count,
                                       offset);
}
