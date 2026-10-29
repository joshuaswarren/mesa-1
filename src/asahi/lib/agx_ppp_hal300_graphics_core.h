/* SPDX-License-Identifier: MIT */
#pragma once

#include <stdbool.h>
#include <stdint.h>
#include "agx_chip.h"

#define AGX_PPP_HAL300_GRAPHICS_RECORDS 9u
#define AGX_PPP_HAL300_GRAPHICS_BYTES 172u

struct agx_ppp_hal300_graphics_fields {
   uint64_t fragment_program;
   uint64_t varying_table;
   unsigned cf_binding_count;

   uint32_t fragment_ti_spill_B;
   uint32_t fragment_ti_staging_B;
   unsigned cull_mode;

   bool scissor_enable;
   bool depth_bias_enable;
   bool depth_clip_disable;
   unsigned polygon_mode;
   bool primitive_msaa;
   /* Encoded 4:4 fixed-point line width, with the hardware off-by-one. */
   unsigned line_width;

   bool no_color_attachment;
   bool no_color_writes;

   unsigned fragment_pass;
   bool disable_tri_merging;

   bool fragment_writes_depth;
   /* Native conservative-depth mode after early-test/side-effect policy. */
   unsigned fragment_depth_mode;
   bool fragment_reads_depth;
   /* 1/W uses the existing position coefficient, with no user UVS word. */
   bool fragment_reads_reciprocal_w;
   bool front_face_ccw;
   bool point_size_output;
   bool layer_output;

   unsigned clip_distance_words;

   enum agx_ppp_hal300_object_type {
      AGX_PPP_HAL300_TRIANGLES = 0,
      AGX_PPP_HAL300_POINTS = 1,
      AGX_PPP_HAL300_LINES = 2,
   } object_type;

   unsigned vertex_varying_components;
   unsigned fragment_varying_components;

   enum agx_ppp_hal300_interpolation {
      AGX_PPP_HAL300_SMOOTH = 0,
      AGX_PPP_HAL300_FLAT = 1,
      AGX_PPP_HAL300_LINEAR = 2,
   } interpolation;

   bool depth_active;
   uint8_t depth_compare;
   bool depth_write;
   bool stencil_active;

   uint8_t front_stencil_reference;
   uint8_t back_stencil_reference;
   uint32_t front_stencil;
   uint32_t back_stencil;

   bool rasterizer_discard;
   /* FRAGMENT_CONTROL bit 21 of the linked fragment: no part writes the
    * tile, so the fragment must not take the visibility tag. */
   bool tag_write_disable;

   /* Checked all-f32 shared VIT/UVS layout, including mixed interpolation. */
   bool general_varyings;
   unsigned cf_register_count;
   uint32_t varying_counts_32;
   uint32_t varying_counts_16;
   uint32_t output_select;
   uint32_t uvs_words;

   bool needs_primitive_id;

   /* Visibility (occlusion) result: AGX_VISIBILITY_MODE_* (0, 2 counting, 3
    * boolean) and the 64-bit slot index in the render's result buffer. */
   uint8_t visibility_mode;
   uint16_t visibility_index;

   unsigned flat_shading_vertex;

   /* Zero retains the single-viewport default for standalone builders. */
   unsigned viewport_count;
};

#define AGX_PPP_HAL300_STENCIL_KEEP_ALWAYS UINT32_C(0x0e000000)
#define AGX_PPP_HAL300_STENCIL_REPLACE_ALWAYS UINT32_C(0x0e020000)
#define AGX_PPP_HAL300_STENCIL_KEEP_EQUAL UINT32_C(0x04000000)

struct agx_ppp_hal300_graphics_core {
   uint8_t bytes[AGX_PPP_HAL300_GRAPHICS_BYTES];
   uint16_t offsets[AGX_PPP_HAL300_GRAPHICS_RECORDS];
   uint8_t lengths[AGX_PPP_HAL300_GRAPHICS_RECORDS];
};

bool agx_ppp_hal300_graphics_core_pack(
   enum agx_chip chip, const struct agx_ppp_hal300_graphics_fields *fields,
   struct agx_ppp_hal300_graphics_core *out);
