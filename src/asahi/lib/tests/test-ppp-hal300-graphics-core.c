/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "agx_ppp_hal300_graphics_core.h"
#include "agx_vdm_state_core.h"
#include <stdio.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
   fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); return 1; \
} } while (0)

int
main(void)
{
   /* Independently extracted M5/25G76 VDM-referenced payloads. All nine
    * payload boundaries matter: joining these behind one selector is wrong.
    * This witness uses clockwise winding and Ioto VA 0x10000170700. */
   const uint8_t golden[] = {
      0,0,0x10,0x10, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0,
      0,0,0,1, 0,0,0,0, 0,0,0,0, 0,0,0,0,
      0,0,0,1, 0x1c,0x5c,0,0, 0,0,0,0,
      1,0,0,0, 0xaa,7,0,0, 0,0,0xe0,0xe2, 0,0,0xc0,0xf0,
      0x20,0x30,2,0, 0xff,0xff,1,0,
      0,0,0x28,0x0c, 0,0,0,0, 0,0,0,0, 0,0,0,0, 4,0,0,0,
      0xe0,0,0,2, 0,2,4,0, 0,0x0f,0x20,7, 0,0x0f,0x20,7,
      0,0,0,0x0e, 0,0,0,0x0e, 0,0,0,0,
      0x62,0,0,2, 0,2,0,0, 0,0,0xe0,7, 0,0,0xe0,7, 0xff,0xff,1,0,
      2,0,0x84,0, 0,0,0,0, 0x80,0,0,0,
      0,0,0x80,0, 0x80,4,0,0,
      0,4,0,0, 0,0,0,0, 0,0,0,0,
   };
   const uint16_t offsets[] = {0,20,48,72,92,120,140,152,160};
   const uint8_t lengths[] = {20,28,24,20,28,20,12,8,12};
   struct agx_ppp_hal300_graphics_fields f = {
      .fragment_program = UINT64_C(0x10000170700),
   };
   struct agx_ppp_hal300_graphics_core out;
   CHECK(sizeof(golden) == sizeof(out.bytes));
   CHECK(agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G17G, &f, &out));
   CHECK(memcmp(out.bytes, golden, sizeof(golden)) == 0);
   CHECK(memcmp(out.offsets, offsets, sizeof(offsets)) == 0);
   CHECK(memcmp(out.lengths, lengths, sizeof(lengths)) == 0);

   /* Relocation must retain all 48 address bits, not only the low word. */
   f.fragment_program = UINT64_C(0xa5c001234540);
   f.cull_mode = 3;
   f.front_face_ccw = true;
   CHECK(agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G17G, &f, &out));
   uint64_t address = 0;
   uint32_t raw_template = 1;
   CHECK(agx_vdm_state_vertex_pipeline_pair_unpack(
      AGX_CHIP_G17G, out.bytes + 36, 8, &address, &raw_template));
   CHECK(address == f.fragment_program && raw_template == 0);
   CHECK(out.bytes[156] == 0x83 && out.bytes[157] == 4 && out.bytes[158] == 1);

   const struct agx_ppp_hal300_graphics_core before = out;
   f.fragment_program++;
   CHECK(!agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G17G, &f, &out));
   CHECK(memcmp(&out, &before, sizeof(out)) == 0);
   f.fragment_program = UINT64_C(1) << 48;
   CHECK(!agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G17G, &f, &out));
   f.fragment_program = 0;
   CHECK(!agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G17G, &f, &out));
   f.fragment_program = UINT64_C(0x10000170700);
   CHECK(!agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G17P, &f, &out));
   f.cull_mode = 4;
   CHECK(!agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G17G, &f, &out));
   CHECK(memcmp(&out, &before, sizeof(out)) == 0);
   CHECK(!agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G17G, NULL, &out));
   CHECK(!agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G17G, &f, NULL));
   puts("M5 native PPP graphics records: PASS");
   return 0;
}
