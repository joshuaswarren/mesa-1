/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT */
#include "../d12_agx3_graphics_bind_core.h"
#include "asahi/lib/agx_ppp_hal200_graphics_core.h"
#include "asahi/lib/agx_vdm_state_core.h"
#include <assert.h>
#include <stdio.h>

int main(void)
{
   struct d12_agx3_triangle_bind_desc d = {
      .compact_va = AGX_G17P_VDM_ALIAS_OFFSET_START,
      .fragment_program = 0x10000400000,
      .uvs_size = 4, .depth_clip = true,
      .viewport.viewport_count = 1,
      .raster.max_x = 256, .raster.max_y = 256,
      .raster.max_z_raw = 0x3f800000,
   };
   struct d12_agx3_triangle_bind out;
   assert(d12_agx3_triangle_bind_generate(AGX_CHIP_G17P, &d, &out));
   assert(out.upload.tail.length == AGX_VDM_HAL200_BIND_TAIL_LENGTH);
   assert(out.upload.pairs[0].context_offset == d.compact_va + 0xa8);
   assert(out.upload.pairs[1].context_offset == d.compact_va);
   assert(out.upload.pairs[5].context_offset == d.compact_va + 0x80);
   assert(out.upload.pairs[7].context_offset == d.compact_va + 0x6c);
   uint64_t fragment_program;
   uint32_t fragment_template;
   assert(agx_vdm_state_vertex_pipeline_pair_unpack(AGX_CHIP_G17P,
          out.bytes + 16, 8, &fragment_program, &fragment_template));
   assert(fragment_program == d.fragment_program && fragment_template == 0);
   struct agx_ppp_hal200_graphics_fields fs;
   assert(agx_ppp_hal200_graphics_core_unpack(AGX_CHIP_G17P,
          AGX_PPP_HAL200_GRAPHICS_FS, out.bytes,
          AGX_PPP_HAL200_FS_LENGTH, &fs));
   assert(fs.words[0] == 0 && fs.words[1] == 0);
   struct agx_ppp_hal200_graphics_fields o;
   assert(agx_ppp_hal200_graphics_core_unpack(AGX_CHIP_G17P,
          AGX_PPP_HAL200_GRAPHICS_OUTPUT, out.bytes + 0x1c,
          AGX_PPP_HAL200_OUTPUT_LENGTH, &o));
   assert(o.words[3] == 4);
   assert(o.words[4] == AGX_PPP_HAL200_PRIMARY_FACE_HEADER);
   /* Shared physical output-window/primary record overlap stays coherent. */
   assert(out.bytes[0x30] == (uint8_t)AGX_PPP_HAL200_PRIMARY_FACE_HEADER);
   assert(out.bytes[0xa8] == (uint8_t)AGX_G17P_VARYING_RECORD_WORD0);
   for (unsigned byte = 4; byte < AGX_G17P_VARYING_RECORD_BYTES; ++byte)
      assert(out.bytes[0xa8 + byte] == 0);

   const struct d12_agx3_triangle_bind_desc good = d;
   for (unsigned fault = 0; fault < 12; ++fault) {
      d = good;
      switch (fault) {
      case 0: d.compact_va = 0; break;
      case 1: d.compact_va = AGX_G17P_VDM_ALIAS_OFFSET_END - 4; break;
      case 2: d.fragment_program |= 1; break;
      case 3: d.uvs_size = 6; break;
      case 4: d.user_size = 1; break;
      case 5: d.coefficient_bindings = 1; break;
      case 6: d.reads_z = true; break;
      case 7: d.writes_memory = true; break;
      case 8: d.writes_depth = true; break;
      case 9: d.has_preamble = true; break;
      case 10: d.viewport.viewport_count = 2; break;
      case 11: d.raster.depth_bias_enable = true; break;
      }
      memset(&out, 0xa5, sizeof(out));
      const struct d12_agx3_triangle_bind poison = out;
      assert(!d12_agx3_triangle_bind_generate(AGX_CHIP_G17P, &d, &out));
      assert(memcmp(&out, &poison, sizeof(out)) == 0);
   }
   d = good;
   assert(!d12_agx3_triangle_bind_generate(AGX_CHIP_G17S, &d, &out));
   assert(!d12_agx3_triangle_bind_generate(AGX_CHIP_G17P, NULL, &out));
   assert(!d12_agx3_triangle_bind_generate(AGX_CHIP_G17P, &d, NULL));
   puts("PASS native triangle PPP/viewport/varying bind-tail roundtrip and fail-closed cases");
   return 0;
}
