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
   if (chip != AGX_CHIP_G17G || !fields || !out ||
       !fields->fragment_program || fields->cull_mode > 3 ||
       fields->fragment_pass > 1 || fields->vertex_varying_components > 8 ||
       fields->fragment_varying_components > fields->vertex_varying_components)
      return false;

   if (fields->fragment_varying_components) {
      if (!fields->varying_table ||
          (fields->varying_table & 63) ||
          fields->varying_table >= (UINT64_C(1) << 48) ||
          !fields->cf_binding_count || fields->cf_binding_count > 3 ||
          (fields->flat_varying && fields->cf_binding_count > 2))
         return false;
   } else if (fields->varying_table || fields->cf_binding_count ||
              fields->flat_varying) {
      return false;
   }

   struct agx_vdm_state_vertex_pipeline_pair program;
   if (!agx_vdm_state_vertex_pipeline_pair_pack(
          chip, fields->fragment_program, 0, &program))
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
       fields->fragment_varying_components << (fields->flat_varying ? 8 : 0),
       0, 0, 0},
      {0x01000000, (uint32_t)(fields->varying_table >> 32) |
                      (fields->cf_binding_count << 16),
       (uint32_t)fields->varying_table, 0, 0, 0, 0},
      {0x00000001, 0x000007aa, 0xe2e00000, 0xf0c00000,
       0x00023020, 0x0001ffff},
      {0x0c280000, (fields->point_size_output ? 0x40000 : 0) |
                      (fields->vertex_varying_components ? 0x10000 : 0), 0, 0,
       4 + (fields->point_size_output ? 1 : 0) +
           fields->vertex_varying_components},
      {0x020000e0, fields->depth_less_write ? 0x00000200 : 0x00040200,
       fields->depth_less_write ? 0x01000f00 : 0x07200f00,
       fields->depth_less_write ? 0x01000f00 : 0x07200f00,
       0x0e000000, 0x0e000000, 0},
      /* HAL300 updateFragmentShaderISPState (M5/25G76, 4d2b10..4d2b38)
       * stores pass bits29..31 at context+780c. encodeAndEmitRenderState
       * loads them at 4e831c (7500+30c), preserves them at 4e83ac..4e83b8,
       * and emits this secondary record's word1 at 4e83d0. They do not
       * belong in the primary face record, unlike the HAL200 layout. */
      {0x02000062, 0x00000200 | (fields->fragment_pass << 29),
       0x07e00000, 0x07e00000, 0x0001ffff},
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
