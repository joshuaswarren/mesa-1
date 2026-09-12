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
       !fields->fragment_program || fields->cull_mode > 3)
      return false;

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
   const uint32_t records[][7] = {
      {0x10100000, 0, 0, 0, 0},
      {0x01000000, 0, 0, 0, 0, 0, 0},
      {0x00000001, 0x000007aa, 0xe2e00000, 0xf0c00000,
       0x00023020, 0x0001ffff},
      {0x0c280000, fields->point_size_output ? 0x40000 : 0, 0, 0,
       fields->point_size_output ? 5 : 4},
      {0x020000e0, 0x00040200, 0x07200f00, 0x07200f00,
       0x0e000000, 0x0e000000, 0},
      {0x02000062, 0x00000200, 0x07e00000, 0x07e00000, 0x0001ffff},
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
