/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT */
#include "d12_agx3_graphics_bind_core.h"
#include "asahi/lib/agx_ppp_hal200_graphics_core.h"
#include "asahi/lib/agx_ppp_w_clamp_cull2_core.h"

bool
d12_agx3_triangle_bind_generate(
   enum agx_chip chip, const struct d12_agx3_triangle_bind_desc *d,
   struct d12_agx3_triangle_bind *output)
{
   if (chip != AGX_CHIP_G17P || !d || !output ||
       (d->uvs_size != 4 && d->uvs_size != 5) || d->user_size ||
       d->raw_output_select != (d->uvs_size == 5 ? (1u << 18) : 0) ||
       d->coefficient_bindings || d->reads_z || d->writes_memory ||
       d->writes_depth || d->has_preamble || d->needs_quad_derivatives ||
       (d->linked_fragment_control >> 29) != 0 ||
       d->viewport.viewport_count != 1 || d->raster.depth_bias_enable ||
       d->depth_compare > 7)
      return false;
   struct d12_agx3_triangle_bind staged = {0};
   uint32_t output_size, output_select;
   if (!agx_ppp_hal200_output_size_pack(d->uvs_size, 0, &output_size) ||
       !agx_ppp_hal200_output_select_pack(chip, d->raw_output_select,
                                         false, false, &output_select) ||
       !agx_ppp_hal200_raster_state_build(chip, &d->raster, &staged.raster))
      return false;
   struct agx_ppp_hal200_fragment_fields fs = {
      .stage_program = d->fragment_program,
   };
   struct agx_ppp_hal200_graphics_fields outputs = {
      .words = {output_select, 0, 0, output_size,
                AGX_PPP_HAL200_PRIMARY_FACE_HEADER, 0},
   };
   struct agx_ppp_hal200_graphics_fields primary = {
      .words = {0, 0x07200f00, 0x0e000000, 0x07200f00, 0x0e000000, 0},
   };
   struct agx_ppp_hal200_graphics_fields secondary = {
      .words = {0x00000200, 0x07e00000, 0x07e00000, 0x0001ffff},
   };
   struct agx_ppp_hal200_graphics_fields cull = {.words = {0x00000480}};
   const struct agx_ppp_hal200_fragment_control_fields control = {
      .linked_control = d->linked_fragment_control,
   };
   if (!agx_ppp_hal200_fragment_controls_pack(
          chip, &control, staged.raster.fragment_control, secondary.words[0],
          &primary.words[0], &secondary.words[0]) ||
       !agx_ppp_hal200_fragment_face_raster_pack(chip, 15, 0,
                                              primary.words[1], &primary.words[1]) ||
       !agx_ppp_hal200_fragment_face_raster_pack(chip, 15, 0,
                                              primary.words[3], &primary.words[3]) ||
       !agx_ppp_hal200_fragment_face_object_pack(chip, 0, secondary.words[1],
                                                &secondary.words[1]) ||
       !agx_ppp_hal200_fragment_face_object_pack(chip, 0, secondary.words[2],
                                                &secondary.words[2]) ||
       !agx_ppp_hal200_cull_mode_pack(chip, d->cull_mode, d->front_face_ccw,
                                      cull.words[0], &cull.words[0]) ||
       !agx_ppp_hal200_cull_provoking_pack(chip, 0, cull.words[0], &cull.words[0]) ||
       !agx_ppp_hal200_depth_clip_pack(chip, d->depth_clip, cull.words[0],
                                      &cull.words[0]))
      return false;
   /* Reuse HK's exact native PRIMARY_FACE compare/write encoder for both
    * rasterizer faces. Without storage the API depth test is inert. */
   if (!agx_ppp_hal200_depth_face_pack(chip, d->has_depth && d->depth_test,
          d->depth_write, d->depth_compare, primary.words[1], &primary.words[1]) ||
       !agx_ppp_hal200_depth_face_pack(chip, d->has_depth && d->depth_test,
          d->depth_write, d->depth_compare, primary.words[3], &primary.words[3]))
      return false;
   outputs.words[5] = primary.words[0];
   struct agx_ppp_hal200_graphics_core f, o, p, s, c;
   struct agx_ppp_w_clamp_cull2_core wc;
   const struct agx_ppp_w_clamp_cull2_fields wc_fields = {.cull2_raw = 0x80};
   if (!agx_ppp_hal200_fragment_core_pack(chip, &fs, &f) ||
       !agx_ppp_hal200_graphics_core_pack(chip, AGX_PPP_HAL200_GRAPHICS_OUTPUT,
                                          &outputs, &o) ||
       !agx_ppp_hal200_graphics_core_pack(chip, AGX_PPP_HAL200_GRAPHICS_PRIMARY_FACE,
                                          &primary, &p) ||
       !agx_ppp_hal200_graphics_core_pack(chip, AGX_PPP_HAL200_GRAPHICS_SECONDARY_FACE,
                                          &secondary, &s) ||
       !agx_ppp_hal200_graphics_core_pack(chip, AGX_PPP_HAL200_GRAPHICS_CULL,
                                          &cull, &c) ||
       !agx_ppp_w_clamp_cull2_core_pack(chip, &wc_fields, &wc))
      return false;
   uint8_t group[AGX_G17P_BIND_GROUP_BYTES] = {0};
   memcpy(group + 0x00, f.bytes, f.length);
   memcpy(group + 0x1c, o.bytes, o.length);
   memcpy(group + 0x30, p.bytes, p.length);
   memcpy(group + 0x4c, s.bytes, s.length);
   memcpy(group + 0x60, wc.bytes, wc.length);
   memcpy(group + 0x6c, c.bytes, c.length);
   struct agx_ppp_viewport_payload_core viewport;
   if (!agx_ppp_viewport_payload_core_pack(chip, &d->viewport, &viewport) ||
       !agx_g17p_upload_bind_state(staged.bytes, sizeof(staged.bytes), d->compact_va,
                                   group, viewport.bytes, viewport.length,
                                   0, 0, &staged.upload))
      return false;
   *output = staged;
   return true;
}
