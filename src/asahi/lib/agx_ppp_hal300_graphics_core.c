/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "agx_ppp_hal300_graphics_core.h"
#include "agx_vdm_state_core.h"
#include <string.h>

bool
agx_ppp_hal300_graphics_core_pack(
   enum agx_chip chip, const struct agx_ppp_hal300_graphics_fields *fields,
   struct agx_ppp_hal300_graphics_core *out)
{
   /* Every field below but lines was grounded on G17G. The J873 G18G
    * captures reproduce byte for byte for each shape they contain --
    * triangles, points, lines, the three pass types, depth less/always with
    * writes, stencil control, two- and four-component smooth varyings --
    * apart from one stencil face bit that no caller can set on either chip
    * (see the stencil note in the header). The shapes grounded
    * only on G17G (flat, noperspective, colour-free, depth output, clip
    * distances, culling and winding) are carried to G18G on the strength of
    * that shared layout, without a G18G capture of their own. */
   if (!agx_chip_hal300_records(chip) || !fields || !out ||
       !fields->fragment_program || fields->cull_mode > 3 ||
       fields->fragment_pass > 2 || fields->vertex_varying_components > 8 ||
       fields->fragment_varying_components > fields->vertex_varying_components)
      return false;

   if (fields->fragment_varying_components) {
      if (!fields->varying_table ||
          (fields->varying_table & 63) ||
          fields->varying_table >= (UINT64_C(1) << 48) ||
          !fields->cf_binding_count || fields->cf_binding_count > 3 ||
          (fields->interpolation != AGX_PPP_HAL300_SMOOTH &&
           fields->cf_binding_count > 2))
         return false;
   } else if (fields->varying_table || fields->cf_binding_count ||
              fields->interpolation != AGX_PPP_HAL300_SMOOTH) {
      return false;
   }
   if (fields->interpolation > AGX_PPP_HAL300_LINEAR)
      return false;

   /* Only the witnessed point shape: Apple's point draw exports a point
    * size, and a point without one has no capture behind it. Lines are
    * captured on G18G only, without a point size. */
   if (fields->object_type > AGX_PPP_HAL300_LINES ||
       (fields->object_type == AGX_PPP_HAL300_POINTS &&
        !fields->point_size_output) ||
       (fields->object_type == AGX_PPP_HAL300_LINES &&
        (chip != AGX_CHIP_G18G || fields->point_size_output)))
      return false;

   /* Only the interpolated colour-free shape is witnessed. */
   if (fields->no_color_attachment &&
       fields->interpolation == AGX_PPP_HAL300_FLAT &&
       fields->fragment_varying_components)
      return false;

   /* never .. always; see the header's decode table. An inactive depth unit
    * carries no comparison of its own. */
   if (fields->depth_compare > 7 ||
       (!fields->depth_active && (fields->depth_compare || fields->depth_write)))
      return false;

   /* 0x40000 tracks whether the depth test is enabled at all, not what the
    * test does. The capture matrix settles it: every render with the test
    * enabled carries 0x200, including the one with comparison always and
    * depth writes off, while the renders with no test at all -- no
    * attachment, an attachment with no depth-stencil state, and the stencil
    * render whose depth test was off -- all carry 0x40000. An earlier model
    * that derived the bit from "always and no write" produced 0x40200 for
    * that first case and was wrong. */
   const uint8_t compare = fields->depth_active ? fields->depth_compare : 7;
   const bool write = fields->depth_active && fields->depth_write;
   const bool depth_inactive = !fields->depth_active;
   const bool points = fields->object_type == AGX_PPP_HAL300_POINTS;
   const bool lines = fields->object_type == AGX_PPP_HAL300_LINES;
   const uint32_t control = 0x00000200 | (points || lines ? 0x04000000 : 0) |
                            (fields->scissor_enable ? 0x00010000 : 0) |
                            (depth_inactive ? 0x00040000 : 0) |
                            (fields->stencil_active ? 0x00080000 : 0);
   const uint32_t face = ((uint32_t)compare << 24) |
                         (write ? 0 : 0x00200000) | 0xf00;

   /* The ISP may resolve depth before the shader runs only while the shader
    * cannot change it; a fragment that writes gl_FragDepth drops both of
    * these bits. See fragment_writes_depth in the header. */
   const uint32_t early_test =
      (fields->fragment_writes_depth ? 0x07200000 : 0x07e00000) |
      (points ? 0x40000000 : 0) | (lines ? 0x10000000 : 0);

   struct agx_vdm_state_vertex_pipeline_pair program;
   if (!agx_vdm_state_vertex_pipeline_pair_pack(
          chip, fields->fragment_program,
          fields->fragment_writes_depth ? 0x40 : 0, &program))
      return false;

   /* M5 native VDM 0x1000018000 references these payloads at offsets
    * 0x34/3c/44/4c/54/5c/6c/74/84. HAL300 has different selectors from
    * HAL200 and groups front/back face before front/back stencil. The
    * OUTPUT record is exactly five words, with no overlapping face header.
    * Viewport and scissor-base updates are independently emitted by caller.
    */
   /* Native color/flat-depth captures on M5/25G76 independently validate
    * counts32 = 4/0x400, absolute 48-bit VIT addresses, and the D32 LESS/write
    * face group. The VIT count occupies bits16..22 above address bits32..47.
    * The fragment program still uses an Ioto pair, not this address format.
    */
   const uint32_t records[][7] = {
      {0x10100000,
       fields->fragment_varying_components <<
          (fields->no_color_attachment
              ? 16
              : fields->interpolation == AGX_PPP_HAL300_FLAT ? 8 : 0),
       0, 0, 0},
      {0x01000000, (uint32_t)(fields->varying_table >> 32) |
                      (fields->cf_binding_count << 16),
       (uint32_t)fields->varying_table, 0, 0, 0, 0},
      {0x00000001, 0x000007aa, 0xe2e00000, 0xf0c00000,
       0x00023020, 0x0001ffff},
      {0x0c280000, (fields->point_size_output ? 0x40000 : 0) |
                      (fields->vertex_varying_components ? 0x10000 : 0) |
                      (fields->clip_distance_words ? 0x1 : 0), 0, 0,
       4 + (fields->point_size_output ? 1 : 0) +
           fields->clip_distance_words +
           fields->vertex_varying_components},
      {0x020000e0, control, face, face,
       fields->front_stencil ? fields->front_stencil : 0x0e000000,
       fields->back_stencil ? fields->back_stencil : 0x0e000000, 0},
      /* HAL300 updateFragmentShaderISPState (M5/25G76, 4d2b10..4d2b38)
       * stores pass bits29..31 at context+780c. encodeAndEmitRenderState
       * loads them at 4e831c (7500+30c), preserves them at 4e83ac..4e83b8,
       * and emits this secondary record's word1 at 4e83d0. They do not
       * belong in the primary face record, unlike the HAL200 layout. */
      {0x02000062, 0x00000200 | (fields->fragment_pass << 29) |
                      (fields->disable_tri_merging || points || lines
                          ? 0x04000000
                          : 0),
       early_test, early_test,
       fields->no_color_attachment ? 0 : 0x0001ffff},
      {0x00840002, 0, 0x80},
      {0x00800000, 0x480 | fields->cull_mode |
                      ((uint32_t)fields->front_face_ccw << 16)},
      {0x00000400, 0, 0},
   };
   const uint8_t lengths[] = {20, 28, 24, 20, 28, 20, 12, 8, 12};
   struct agx_ppp_hal300_graphics_core staged = {0};
   unsigned offset = 0;
   for (unsigned i = 0; i < AGX_PPP_HAL300_GRAPHICS_RECORDS; ++i) {
      staged.offsets[i] = offset;
      staged.lengths[i] = lengths[i];
      for (unsigned j = 0; j < lengths[i]; ++j)
         staged.bytes[offset + j] = records[i][j / 4] >> (8 * (j % 4));
      offset += lengths[i];
   }

   /* Native fragment record uses the same full 48-bit Ioto pair as VDM
    * vertex state. The captured 01000000/00005c1c points to 0x10000170700.
    * Native setCullMode 4ee1b0..4ee1c4 and setFrontFacingWinding
    * 4ee73c..4ee750 establish cull bits0..1 and winding bit16 above. */
   memcpy(staged.bytes + staged.offsets[1] + 16, program.bytes, 8);
   *out = staged;
   return true;
}
