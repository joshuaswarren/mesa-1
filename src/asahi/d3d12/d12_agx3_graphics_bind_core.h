/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT */
#pragma once

#include "asahi/lib/agx_g17p_draw_state_upload.h"
#include "asahi/lib/agx_ppp_hal200_raster_core.h"

/* Baseline opaque solid triangle only: one viewport, no user coefficients,
 * depth/stencil, CProg or shader memory side effects. The caller still owns
 * the compact VDM allocation, stage CODE/Ioto and render-context lifetime. */
struct d12_agx3_triangle_bind_desc {
   uint64_t compact_va;
   uint64_t fragment_program;
   uint32_t linked_fragment_control;
   uint32_t raw_output_select;
   uint8_t uvs_size;
   uint8_t user_size;
   uint8_t coefficient_bindings;
   bool reads_z;
   bool writes_memory;
   bool needs_quad_derivatives;
   bool writes_depth;
   bool has_preamble;
   unsigned cull_mode;
   bool front_face_ccw;
   bool depth_clip;
   struct agx_ppp_hal200_raster_fields raster;
   struct agx_ppp_viewport_payload_fields viewport;
};

struct d12_agx3_triangle_bind {
   uint8_t bytes[AGX_G17P_BIND_STATE_BYTES];
   struct agx_g17p_bind_state_upload upload;
   struct agx_ppp_hal200_raster_state raster;
};

/* Transactional byte/state producer, not a draw admission or submit API. */
bool d12_agx3_triangle_bind_generate(
   enum agx_chip chip, const struct d12_agx3_triangle_bind_desc *desc,
   struct d12_agx3_triangle_bind *output);
