/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT */
#pragma once

#include "asahi/lib/agx_g17p_draw_state_upload.h"
#include "asahi/lib/agx_ppp_hal200_raster_core.h"

/* One viewport, no stencil, CProg or shader memory side effects. Smooth user
 * coefficients and native pass/quad controls use the shared Asahi producers.
 * The caller validates the coefficient layout and still owns
 * the compact VDM allocation, stage CODE/Ioto and render-context lifetime. */
struct d12_agx3_triangle_bind_desc {
   bool mesh;
   unsigned conservative_depth;
   bool early_fragment_tests;
   uint64_t compact_va;
   uint64_t fragment_program;
   uint32_t linked_fragment_control;
   uint32_t raw_output_select;
   uint8_t uvs_size;
   uint8_t user_size;
   uint8_t coefficient_bindings;
   uint64_t varying_table;
   uint32_t varying_counts_32;
   uint32_t varying_counts_16;
   bool reads_z;
   bool writes_memory;
   bool needs_quad_derivatives;
   bool writes_depth;
   bool has_preamble;
   unsigned cull_mode;
   bool front_face_ccw;
   bool depth_clip;
   /* Cold fixed-function depth words. The caller must own/validate D32 ZLS
    * storage and live admission independently; these bytes admit no draw. */
   bool has_depth;
   bool depth_test;
   bool depth_write;
   uint8_t depth_compare;
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
