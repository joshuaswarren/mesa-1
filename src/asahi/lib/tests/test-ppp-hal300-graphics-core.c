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

   /* Native HAL300 disassembly routes context+780c to word1 of the
    * 02000062 secondary record. For pass=1 it retains 20000200 there;
    * every primary-face byte and all other records must stay unchanged. */
   f.fragment_pass = 1;
   CHECK(agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G17G, &f, &out));
   for (unsigned i = 0; i < sizeof(golden); ++i)
      CHECK(out.bytes[i] == (i == 127 ? 0x20 : golden[i]));
   const struct agx_ppp_hal300_graphics_core translucent = out;
   for (unsigned pass = 2; pass < 8; ++pass) {
      f.fragment_pass = pass;
      CHECK(!agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G17G, &f, &out));
      CHECK(memcmp(&out, &translucent, sizeof(out)) == 0);
   }
   f.fragment_pass = 0;
   CHECK(agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G17G, &f, &out));
   CHECK(memcmp(out.bytes, golden, sizeof(golden)) == 0);

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

   /* Independently captured smooth-color and flat-color/D32 draws. Both
    * native outputs passed the 60928-pixel oracle after these snapshots.
    * The flat reference has one CF binding, exactly as Zink gears does. */
   const uint32_t color_words[] = {
      0x10100000, 4, 0, 0, 0,
      0x01000000, 0x00020100, 0x4c0, 0, 0x01000000, 0x5c1c, 0,
      1, 0x7aa, 0xe2e00000, 0xf0c00000, 0x23020, 0x1ffff,
      0x0c280000, 0x50000, 0, 0, 9,
      0x020000e0, 0x40200, 0x07200f00, 0x07200f00, 0x0e000000, 0x0e000000, 0,
      0x02000062, 0x200, 0x07e00000, 0x07e00000, 0x1ffff,
      0x00840002, 0, 0x80, 0x00800000, 0x480, 0x400, 0, 0,
   };
   const uint32_t flat_depth_words[] = {
      0x10100000, 0x400, 0, 0, 0,
      0x01000000, 0x00010100, 0x480, 0, 0x01000000, 0x6e1c, 0,
      1, 0x7aa, 0xe2e00000, 0xf0c00000, 0x23020, 0x1ffff,
      0x0c280000, 0x50000, 0, 0, 9,
      0x020000e0, 0x200, 0x01000f00, 0x01000f00, 0x0e000000, 0x0e000000, 0,
      0x02000062, 0x200, 0x07e00000, 0x07e00000, 0x1ffff,
      0x00840002, 0, 0x80, 0x00800000, 0x480, 0x400, 0, 0,
   };
   CHECK(sizeof(color_words) == sizeof(out.bytes));
   CHECK(sizeof(flat_depth_words) == sizeof(out.bytes));
   f = (struct agx_ppp_hal300_graphics_fields) {
      .fragment_program = UINT64_C(0x10000170700),
      .varying_table = UINT64_C(0x100000004c0),
      .cf_binding_count = 2,
      .point_size_output = true,
      .vertex_varying_components = 4,
      .fragment_varying_components = 4,
   };
   CHECK(agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G17G, &f, &out));
   for (unsigned i = 0; i < sizeof(out.bytes); ++i)
      CHECK(out.bytes[i] == (uint8_t)(color_words[i / 4] >> (8 * (i % 4))));
   f.fragment_program = UINT64_C(0x100001b8700);
   f.varying_table = UINT64_C(0x10000000480);
   f.cf_binding_count = 1;
   f.flat_varying = true;
   f.depth_less_write = true;
   CHECK(agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G17G, &f, &out));
   for (unsigned i = 0; i < sizeof(out.bytes); ++i)
      CHECK(out.bytes[i] == (uint8_t)(flat_depth_words[i / 4] >> (8 * (i % 4))));

   const struct agx_ppp_hal300_graphics_core color_before = out;
   f.varying_table++;
   CHECK(!agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G17G, &f, &out));
   f.varying_table = UINT64_C(1) << 48;
   CHECK(!agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G17G, &f, &out));
   f.varying_table = 0;
   CHECK(!agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G17G, &f, &out));
   f.varying_table = UINT64_C(0x10000000480);
   f.cf_binding_count = 0;
   CHECK(!agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G17G, &f, &out));
   f.cf_binding_count = 3;
   CHECK(!agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G17G, &f, &out));
   f.flat_varying = false;
   f.cf_binding_count = 4;
   CHECK(!agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G17G, &f, &out));
   f.cf_binding_count = 1;
   f.fragment_varying_components = 0;
   CHECK(!agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G17G, &f, &out));
   CHECK(memcmp(&out, &color_before, sizeof(out)) == 0);

   /* A constant fragment shader has no CF bindings even when a reusable
    * vertex shader exports float4 color. Preserve its nine UVS words and
    * point-size selection without inventing a fragment interpolation input. */
   f = (struct agx_ppp_hal300_graphics_fields) {
      .fragment_program = UINT64_C(0x10000170700),
      .point_size_output = true,
      .vertex_varying_components = 4,
   };
   CHECK(agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G17G, &f, &out));
   CHECK(out.bytes[4] == 0); /* No interpolated coefficients. */
   for (unsigned i = 24; i < 36; ++i)
      CHECK(out.bytes[i] == 0); /* No VIT pointer or CF binding count. */
   CHECK(out.bytes[78] == 5); /* PSIZ and color vertex outputs. */
   CHECK(out.bytes[88] == 9); /* All vertex output words stay live. */
   const struct agx_ppp_hal300_graphics_core unused_before = out;
   f.fragment_varying_components = 4;
   f.vertex_varying_components = 0;
   f.varying_table = UINT64_C(0x10000000480);
   f.cf_binding_count = 1;
   CHECK(!agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G17G, &f, &out));
   CHECK(memcmp(&out, &unused_before, sizeof(out)) == 0);

   /* The Varying Counts ABI counts scalar coefficients: smooth occupies
    * byte 0 and flat byte 1. Changing vector width also changes the UVS
    * word count, but no record boundary, pointer or face-state byte. */
   for (unsigned count = 1; count <= 8; ++count) {
      for (unsigned flat = 0; flat <= 1; ++flat) {
         f = (struct agx_ppp_hal300_graphics_fields) {
            .fragment_program = UINT64_C(0x10000170700),
            .varying_table = UINT64_C(0x100000004c0),
            .cf_binding_count = flat ? 1 : 2,
            .point_size_output = true,
            .vertex_varying_components = count,
            .fragment_varying_components = count,
            .flat_varying = flat,
         };
         CHECK(agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G17G, &f, &out));
         for (unsigned i = 0; i < sizeof(out.bytes); ++i) {
            uint8_t expected = color_words[i / 4] >> (8 * (i % 4));
            if (i == 4) expected = flat ? 0 : count;
            if (i == 5) expected = flat ? count : 0;
            if (i == 26) expected = flat ? 1 : 2;
            if (i == 88) expected = 5 + count;
            CHECK(out.bytes[i] == expected);
         }
      }
   }
   f.vertex_varying_components = 9;
   CHECK(!agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G17G, &f, &out));
   f.vertex_varying_components = 2;
   f.fragment_varying_components = 3;
   CHECK(!agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G17G, &f, &out));
   puts("M5 native PPP graphics records: PASS");
   return 0;
}
