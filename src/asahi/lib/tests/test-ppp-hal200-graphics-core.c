/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_ppp_hal200_graphics_core.h"

#include <assert.h>
#include <string.h>

static void
check(enum agx_ppp_hal200_graphics_bundle bundle, const uint32_t *words,
      unsigned word_count, const uint8_t *expected, size_t expected_size)
{
   struct agx_ppp_hal200_graphics_fields fields = {0};
   memcpy(fields.words, words, word_count * sizeof(words[0]));

   struct agx_ppp_hal200_graphics_core packed;
   assert(agx_ppp_hal200_graphics_core_pack(AGX_CHIP_G17P, bundle, &fields,
                                             &packed));
   assert(packed.length == expected_size);
   assert(memcmp(packed.bytes, expected, expected_size) == 0);

   struct agx_ppp_hal200_graphics_fields unpacked;
   assert(agx_ppp_hal200_graphics_core_unpack(
      AGX_CHIP_G17P, bundle, packed.bytes, packed.length, &unpacked));
   assert(memcmp(unpacked.words, fields.words, sizeof(fields.words)) == 0);
}

int
main(void)
{
   /* Native depth-only, conservative, late atomic, sample-mask and explicit
    * early witnesses. Only bits22:23 may change in either face2 word. */
   for (unsigned layout = 0; layout < 4; ++layout) {
      for (unsigned early = 0; early < 2; ++early) {
         for (unsigned memory = 0; memory < 2; ++memory) {
            uint32_t face;
            unsigned expected = early ? 3 : memory ? 0 : layout;
            assert(agx_ppp_hal200_fragment_face_depth_pack(
               AGX_CHIP_G17P, layout, early, memory, 0xa7ffffff, &face));
            assert(((face >> 22) & 3) == expected);
            assert((face & ~UINT32_C(0x00c00000)) ==
                   (UINT32_C(0xa7ffffff) & ~UINT32_C(0x00c00000)));
         }
      }
   }
   uint32_t bad_depth_mode = 0x12345678;
   assert(!agx_ppp_hal200_fragment_face_depth_pack(
      AGX_CHIP_G17G, 0, false, false, 0, &bad_depth_mode));
   assert(!agx_ppp_hal200_fragment_face_depth_pack(
      AGX_CHIP_G17P, 4, false, false, 0, &bad_depth_mode));
   assert(!agx_ppp_hal200_fragment_face_depth_pack(
      AGX_CHIP_G17P, 0, false, false, 0, NULL));
   assert(bad_depth_mode == UINT32_C(0x12345678));

   for (unsigned mode = 0; mode < 4; ++mode) {
      uint32_t control = UINT32_C(0xdead3eef);
      uint32_t offset = UINT32_C(0xdeadbeef);
      const bool valid = mode != 1;
      assert(agx_ppp_hal200_occlusion_pack(
                AGX_CHIP_G17P, mode, 0x7fff, UINT32_C(0x1234ffff),
                &control, &offset) == valid);
      if (valid) {
         assert(control == ((UINT32_C(0x1234ffff) & ~UINT32_C(0xc000)) |
                            (mode << 14)));
         assert(offset == UINT32_C(0xfffe0000));
      } else {
         assert(control == UINT32_C(0xdead3eef));
         assert(offset == UINT32_C(0xdeadbeef));
      }
   }
   uint32_t unchanged_control = UINT32_C(0xdead3eef);
   uint32_t unchanged_offset = UINT32_C(0xdeadbeef);
   assert(!agx_ppp_hal200_occlusion_pack(
      AGX_CHIP_G17G, 2, 0, 0, &unchanged_control, &unchanged_offset));
   assert(!agx_ppp_hal200_occlusion_pack(
      AGX_CHIP_G17P, 2, 32768, 0, &unchanged_control, &unchanged_offset));
   assert(!agx_ppp_hal200_occlusion_pack(
      AGX_CHIP_G17P, 2, 0, 0, NULL, &unchanged_offset));
   assert(!agx_ppp_hal200_occlusion_pack(
      AGX_CHIP_G17P, 2, 0, 0, &unchanged_control, NULL));
   assert(unchanged_control == UINT32_C(0xdead3eef));
   assert(unchanged_offset == UINT32_C(0xdeadbeef));

   /* Derived directly from native458898..45891c and vector constants
    * 811700/811710: asymmetric masks and distinct operations expose swaps. */
   assert(agx_ppp_hal200_stencil_pack(false, 0, 7, 6, 5, 0xaa, 0x55) == 0x0e000000);
   assert(agx_ppp_hal200_stencil_pack(true, 7, 0, 0, 2, 0xff, 0xff) == 0x0e02ffff);
   assert(agx_ppp_hal200_stencil_pack(true, 2, 3, 4, 5, 0x3c, 0xa5) == 0x04e53ca5);
   assert(agx_ppp_hal200_stencil_pack(true, 6, 7, 1, 0, 0x196, 0x269) == 0x0dc89669);

   static const uint32_t fs_words[] = {
      0x00020100, 0x000004c0, 0, 0x01000000, 0x00004c19, 0,
   };
   static const uint8_t fs_expected[] = {
      0x00, 0x00, 0x80, 0x00, 0x00, 0x01, 0x02, 0x00,
      0xc0, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x01, 0x19, 0x4c, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
   };
   check(AGX_PPP_HAL200_GRAPHICS_FS, fs_words, 6, fs_expected,
         sizeof(fs_expected));

   /* Valid G1a vary0 is the exact no-varying shape: it has no VIT allocation,
    * and its separately uploaded Ioto stage program is the split pointer in
    * words 4/5. */
   static const uint8_t no_vary_expected[] = {
      0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
      0x19, 0x4c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   const struct agx_ppp_hal200_fragment_fields live_fs = {
      .resource_counts = 0,
      .varying_table = 0,
      .unknown_0c = 0,
      .stage_program = UINT64_C(0x10000130640),
      .stage_program_template = 0,
      .flags = 0,
   };
   struct agx_ppp_hal200_graphics_core live_fs_packed;
   assert(agx_ppp_hal200_fragment_core_pack(AGX_CHIP_G17P, &live_fs,
                                            &live_fs_packed));
   assert(live_fs_packed.length == sizeof(no_vary_expected));
   assert(memcmp(live_fs_packed.bytes, no_vary_expected,
                 sizeof(no_vary_expected)) == 0);

   /* Pinned native finalization maps 32 TI storage bytes to 0x40 in the
    * stage-program template. IPR is independently zero in all five retained
    * pointer-bitcast CProg metadata witnesses, despite nonzero temporaries. */
   struct agx_ppp_hal200_fragment_fields ti = live_fs;
   assert(agx_ppp_hal200_fragment_ti_resources(AGX_CHIP_G17P, 32, 0, &ti));
   assert(ti.unknown_0c == 0 && ti.stage_program_template == 0x40);
   assert(agx_ppp_hal200_fragment_core_pack(AGX_CHIP_G17P, &ti, &live_fs_packed));
   uint32_t ti_words[7];
   memcpy(ti_words, live_fs_packed.bytes, sizeof(ti_words));
   assert(ti_words[3] == 0 && ti_words[4] == 0x01000040);
   assert(ti_words[5] == 0x4c19);

   const uint32_t spill_inputs[] = {0, 1, 3, 4, 5, 16, 32, 1040, 65536, 1048572};
   const uint32_t spill_words[] = {0, 8, 8, 8, 16, 32, 64, 2080, 131072, 0x1ffff8};
   const uint32_t ipr_inputs[] = {0, 1, 4, 5, 8, 9, 16, 17, 16376};
   const uint32_t ipr_words[] = {0, 2, 2, 2, 2, 4, 4, 6, 4094};
   for (unsigned s = 0; s < sizeof(spill_inputs) / sizeof(spill_inputs[0]); ++s) {
      for (unsigned i = 0; i < sizeof(ipr_inputs) / sizeof(ipr_inputs[0]); ++i) {
         ti = live_fs;
         ti.stage_program_template = 0x200007;
         ti.unknown_0c = 0xabcd0000;
         assert(agx_ppp_hal200_fragment_ti_resources(
            AGX_CHIP_G17P, spill_inputs[s], ipr_inputs[i], &ti));
         assert(ti.stage_program_template == (0x200007 | spill_words[s]));
         assert(ti.unknown_0c == (0xabcd0000 | ipr_words[i]));
         assert(agx_ppp_hal200_fragment_core_pack(AGX_CHIP_G17P, &ti, &live_fs_packed));
         memcpy(ti_words, live_fs_packed.bytes, sizeof(ti_words));
         assert(ti_words[4] == (0x01200007 | spill_words[s]));
         assert(ti_words[5] == 0x4c19);
      }
   }
   struct agx_ppp_hal200_fragment_fields saved_ti = ti;
   assert(!agx_ppp_hal200_fragment_ti_resources(AGX_CHIP_G16G, 32, 0, &ti));
   assert(!agx_ppp_hal200_fragment_ti_resources(AGX_CHIP_G17P, 1048573, 0, &ti));
   assert(!agx_ppp_hal200_fragment_ti_resources(AGX_CHIP_G17P, 0, 16377, &ti));
   assert(!agx_ppp_hal200_fragment_ti_resources(AGX_CHIP_G17P, UINT32_MAX, 0, &ti));
   assert(!agx_ppp_hal200_fragment_ti_resources(AGX_CHIP_G17P, 0, UINT32_MAX, &ti));
   assert(!memcmp(&ti, &saved_ti, sizeof(ti)));

   /* Native 0x4c5e38 packs absolute VIT 0x100000004c0 and 2 iterations as
    * counts 0x00020100 / low-pointer 0x4c0, independently of the Ioto VA. */
   struct agx_ppp_hal200_fragment_fields varying_fs = live_fs;
   varying_fs.resource_counts = UINT32_C(2) << 16;
   varying_fs.varying_table = UINT64_C(0x100000004c0);
   assert(agx_ppp_hal200_fragment_core_pack(AGX_CHIP_G17P, &varying_fs,
                                            &live_fs_packed));
   assert(memcmp(live_fs_packed.bytes, fs_expected, sizeof(fs_expected)) == 0);

   /* Relocation across all 48 address bits must survive the split fields. */
   for (unsigned bit = 6; bit < 48; ++bit) {
      varying_fs.varying_table = UINT64_C(1) << bit;
      assert(agx_ppp_hal200_fragment_core_pack(AGX_CHIP_G17P, &varying_fs,
                                               &live_fs_packed));
      uint32_t words[7];
      memcpy(words, live_fs_packed.bytes, sizeof(words));
      uint64_t decoded = ((uint64_t)(words[1] & 0xffff) << 32) | words[2];
      assert(decoded == varying_fs.varying_table);
      assert(words[1] >> 16 == 2);
   }

   struct agx_ppp_hal200_fragment_fields bad_fs = live_fs;
   bad_fs.varying_table = 4;
   assert(!agx_ppp_hal200_fragment_core_pack(AGX_CHIP_G17P, &bad_fs,
                                             &live_fs_packed));
   bad_fs = live_fs;
   bad_fs.varying_table = UINT64_C(1) << 48;
   assert(!agx_ppp_hal200_fragment_core_pack(AGX_CHIP_G17P, &bad_fs,
                                             &live_fs_packed));
   bad_fs = live_fs;
   bad_fs.resource_counts = 1;
   assert(!agx_ppp_hal200_fragment_core_pack(AGX_CHIP_G17P, &bad_fs,
                                             &live_fs_packed));
   bad_fs = live_fs;
   bad_fs.stage_program |= 4;
   assert(!agx_ppp_hal200_fragment_core_pack(AGX_CHIP_G17P, &bad_fs,
                                             &live_fs_packed));
   bad_fs = live_fs;
   bad_fs.stage_program = UINT64_C(1) << 48;
   assert(!agx_ppp_hal200_fragment_core_pack(AGX_CHIP_G17P, &bad_fs,
                                             &live_fs_packed));
   bad_fs = live_fs;
   bad_fs.stage_program_template = 0x00400000;
   assert(!agx_ppp_hal200_fragment_core_pack(AGX_CHIP_G17P, &bad_fs,
                                             &live_fs_packed));

   uint32_t output_size;
   assert(agx_ppp_hal200_output_size_pack(4, 0, &output_size));
   assert(output_size == 4);
   assert(agx_ppp_hal200_output_size_pack(6, 0, &output_size));
   assert(output_size == 6);
   assert(agx_ppp_hal200_output_size_pack(8, 0, &output_size));
   assert(output_size == 8);
   assert(!agx_ppp_hal200_output_size_pack(256, 0, &output_size));
   assert(!agx_ppp_hal200_output_size_pack(4, 256, &output_size));
   assert(!agx_ppp_hal200_output_size_pack(4, 2, NULL));

   uint32_t output_select;
   assert(agx_ppp_hal200_output_select_pack(
      AGX_CHIP_G17P, UINT32_C(0x00180000), true, false,
      &output_select));
   assert(output_select == UINT32_C(0x00190000));
   assert(agx_ppp_hal200_output_select_pack(
      AGX_CHIP_G17P, UINT32_C(0x00040000), false, true,
      &output_select));
   assert(output_select == UINT32_C(0x00240000));
   /* Apple position + smooth float2 + [[point_size]] keeps size 7 and sets
    * the point-size selector; merging the fragment varying gives 0x50000. */
   assert(agx_ppp_hal200_output_select_pack(
      AGX_CHIP_G17P, UINT32_C(0x00040000), true, false,
      &output_select));
   assert(output_select == UINT32_C(0x00050000));
   /* Native clip3 preserves the compiler's first three clip-distance bits
    * and merges the fragment-owned varying bit into the same HAL200 word. */
   assert(agx_ppp_hal200_output_select_pack(
      AGX_CHIP_G17P, UINT32_C(0x00000007), true, false,
      &output_select));
   assert(output_select == UINT32_C(0x00010007));
   assert(!agx_ppp_hal200_output_select_pack(
      AGX_CHIP_G17P, UINT32_C(0x00080000), false, false,
      &output_select));
   assert(!agx_ppp_hal200_output_select_pack(
      AGX_CHIP_G17P, UINT32_C(0x00020000), false, false,
      &output_select));
   assert(!agx_ppp_hal200_output_select_pack(
      AGX_CHIP_G17S, 0, false, false, &output_select));
   assert(!agx_ppp_hal200_output_select_pack(
      AGX_CHIP_G17P, 0, false, false, NULL));
   uint32_t viewport_count;
   for (unsigned count = 1; count <= 16; ++count) {
      assert(agx_ppp_hal200_viewport_count_pack(
         AGX_CHIP_G17P, count, &viewport_count));
      assert(viewport_count == count - 1);
   }
   assert(!agx_ppp_hal200_viewport_count_pack(
      AGX_CHIP_G17P, 0, &viewport_count));
   assert(!agx_ppp_hal200_viewport_count_pack(
      AGX_CHIP_G17P, 17, &viewport_count));
   assert(!agx_ppp_hal200_viewport_count_pack(
      AGX_CHIP_G17G, 1, &viewport_count));

   static const uint32_t output_words[] = {
      0, 0, 0, 4, AGX_PPP_HAL200_PRIMARY_FACE_HEADER, 0x00010200,
   };
   static const uint8_t output_expected[] = {
      0x00, 0x00, 0x0a, 0x0c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
      0xb4, 0x00, 0x00, 0x01, 0x00, 0x02, 0x01, 0x00,
   };
   check(AGX_PPP_HAL200_GRAPHICS_OUTPUT, output_words, 6, output_expected,
         sizeof(output_expected));

   /* Captured 194053 doorbell1 output record: varying enabled, middle
    * words zero, total f32 footprint6. Empty iteration metadata stays off. */
   static const uint32_t varying_output_words[] = {
      0x10000, 0, 0, 6, AGX_PPP_HAL200_PRIMARY_FACE_HEADER, 0x00040200,
   };
   static const uint8_t varying_output_expected[] = {
      0x00, 0x00, 0x0a, 0x0c, 0x00, 0x00, 0x01, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x06, 0x00, 0x00, 0x00, 0xb4, 0x00, 0x00, 0x01,
      0x00, 0x02, 0x04, 0x00,
   };
   check(AGX_PPP_HAL200_GRAPHICS_OUTPUT, varying_output_words, 6,
         varying_output_expected, sizeof(varying_output_expected));

   static const uint32_t primary_words[] = {
      0x00010200, 0x07200f00, 0x0e00ffff,
      0x07200f00, 0x0e00ffff, 0,
   };
   static const uint8_t primary_expected[] = {
      0xb4, 0x00, 0x00, 0x01, 0x00, 0x02, 0x01, 0x00,
      0x00, 0x0f, 0x20, 0x07, 0xff, 0xff, 0x00, 0x0e,
      0x00, 0x0f, 0x20, 0x07, 0xff, 0xff, 0x00, 0x0e,
      0x00, 0x00, 0x00, 0x00,
   };
   check(AGX_PPP_HAL200_GRAPHICS_PRIMARY_FACE, primary_words, 6,
         primary_expected, sizeof(primary_expected));

   static const uint32_t secondary_words[] = {
      0x00000200, 0x07e00000, 0x07e00000, 0x0001ffff,
   };
   static const uint8_t secondary_expected[] = {
      0x48, 0x00, 0x00, 0x02, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00,
      0xe0, 0x07, 0x00, 0x00, 0xe0, 0x07, 0xff, 0xff, 0x01, 0x00,
   };
   check(AGX_PPP_HAL200_GRAPHICS_SECONDARY_FACE, secondary_words, 4,
         secondary_expected, sizeof(secondary_expected));

   static const uint32_t cull_words[] = {0x00010480};
   static const uint8_t cull_expected[] = {
      0x00, 0x00, 0x20, 0x00, 0x80, 0x04, 0x01, 0x00,
   };
   check(AGX_PPP_HAL200_GRAPHICS_CULL, cull_words, 1, cull_expected,
         sizeof(cull_expected));

   /* A 3px line is 0x2f in 4:4-minus-one form. Polygon mode line occupies
    * PRIMARY_FACE bits18:19 on both faces. */
   uint32_t face = UINT32_C(0x07200f5a);
   assert(agx_ppp_hal200_fragment_face_raster_pack(
      AGX_CHIP_G17P, 0x2f, 1, face, &face));
   assert(face == UINT32_C(0x07242f5a));

   uint32_t face2 = UINT32_C(0x07e00000);
   assert(agx_ppp_hal200_fragment_face_object_pack(
      AGX_CHIP_G17P, 1, face2, &face2));
   assert(face2 == UINT32_C(0x17e00000));

   uint32_t bresenham_cull = cull_words[0];
   assert(agx_ppp_hal200_primitive_msaa_pack(
      AGX_CHIP_G17P, true, bresenham_cull, &bresenham_cull));
   assert(bresenham_cull == UINT32_C(0x00018480));

   /* Every helper is fail-closed and leaves caller state untouched. */
   uint32_t rejected = UINT32_C(0xa5a5a5a5);
   assert(!agx_ppp_hal200_fragment_face_raster_pack(
      AGX_CHIP_G17G, 0x0f, 1, 0, &rejected));
   assert(!agx_ppp_hal200_fragment_face_raster_pack(
      AGX_CHIP_G17P, 0x100, 1, 0, &rejected));
   assert(!agx_ppp_hal200_fragment_face_raster_pack(
      AGX_CHIP_G17P, 0x0f, 3, 0, &rejected));
   assert(!agx_ppp_hal200_fragment_face_object_pack(
      AGX_CHIP_G17P, 3, 0, &rejected));
   assert(!agx_ppp_hal200_primitive_msaa_pack(
      AGX_CHIP_G17G, true, 0, &rejected));
   assert(rejected == UINT32_C(0xa5a5a5a5));

   /* Grounded native depth bitfields; all unrelated bits survive. */
   uint32_t depth_face = 0;
   assert(agx_ppp_hal200_depth_face_pack(AGX_CHIP_G17P, true, true,
                                         1, 0x07200f00, &depth_face));
   assert(depth_face == UINT32_C(0x01000f00));
   assert(agx_ppp_hal200_depth_face_pack(AGX_CHIP_G17P, true, false,
                                         1, 0x07200f00, &depth_face));
   assert(depth_face == UINT32_C(0x01200f00));
   for (unsigned compare = 0; compare < 8; ++compare) {
      for (unsigned write = 0; write < 2; ++write) {
         assert(agx_ppp_hal200_depth_face_pack(AGX_CHIP_G17P, true, write,
                                               compare, 0xf8dabcde, &depth_face));
         assert(((depth_face >> 24) & 7) == compare);
         assert(((depth_face >> 21) & 1) == !write);
         assert((depth_face & ~UINT32_C(0x07200000)) ==
                (UINT32_C(0xf8dabcde) & ~UINT32_C(0x07200000)));
         assert(agx_ppp_hal200_depth_face_pack(AGX_CHIP_G17P, false, write,
                                               compare, 0x07200f00, &depth_face));
         assert(depth_face == UINT32_C(0x07200f00));
      }
   }
   depth_face = UINT32_C(0xa5a5a5a5);
   assert(!agx_ppp_hal200_depth_face_pack(AGX_CHIP_G17G, true, true,
                                          1, 0, &depth_face));
   assert(!agx_ppp_hal200_depth_face_pack(AGX_CHIP_G17P, true, true,
                                          8, 0, &depth_face));
   assert(depth_face == UINT32_C(0xa5a5a5a5));
   assert(!agx_ppp_hal200_depth_face_pack(AGX_CHIP_G17P, true, true,
                                          1, 0, NULL));

   struct agx_ppp_hal200_graphics_fields fields = {0};
   struct agx_ppp_hal200_graphics_core packed;
   assert(!agx_ppp_hal200_graphics_core_pack(
      AGX_CHIP_G17G, AGX_PPP_HAL200_GRAPHICS_FS, &fields, &packed));
   assert(!agx_ppp_hal200_fragment_core_pack(AGX_CHIP_G17G, &live_fs,
                                             &packed));
   for (unsigned selector = 0; selector < 3; ++selector) {
      uint32_t cull;
      assert(agx_ppp_hal200_cull_provoking_pack(AGX_CHIP_G17P, selector,
                                               0x480, &cull));
      assert((cull == (uint32_t[]){0x480, 0x500, 0x580}[selector]));
      assert(agx_ppp_hal200_cull_provoking_pack(AGX_CHIP_G17P, selector,
                                               0xdeadbeef, &cull));
      assert((cull & ~0x180u) == (0xdeadbeefu & ~0x180u));
      assert(((cull >> 7) & 3) == selector + 1);
   }
   uint32_t unchanged = 0xdeadbeef;
   assert(!agx_ppp_hal200_cull_provoking_pack(AGX_CHIP_G17P, 3, 0, &unchanged));
   assert(!agx_ppp_hal200_cull_provoking_pack(AGX_CHIP_G17G, 0, 0, &unchanged));
   assert(!agx_ppp_hal200_cull_provoking_pack(AGX_CHIP_G17P, 0, 0, NULL));
   assert(unchanged == 0xdeadbeef);
   for (unsigned mode = 0; mode < 4; ++mode) {
      for (unsigned ccw = 0; ccw < 2; ++ccw) {
         uint32_t cull;
         assert(agx_ppp_hal200_cull_mode_pack(AGX_CHIP_G17P, mode, ccw,
                                               0x480, &cull));
         assert(cull == (0x480u | mode | (ccw << 16)));
         assert(agx_ppp_hal200_cull_mode_pack(AGX_CHIP_G17P, mode, ccw,
                                               0xdeadbeef, &cull));
         assert((cull & ~0x10003u) == (0xdeadbeefu & ~0x10003u));
         assert((cull & 3) == mode && ((cull >> 16) & 1) == ccw);
      }
   }
   unchanged = 0xdeadbeef;
   assert(!agx_ppp_hal200_cull_mode_pack(AGX_CHIP_G17P, 4, 0, 0, &unchanged));
   assert(!agx_ppp_hal200_cull_mode_pack(AGX_CHIP_G17G, 0, 0, 0, &unchanged));
   assert(!agx_ppp_hal200_cull_mode_pack(AGX_CHIP_G17P, 0, 0, 0, NULL));
   assert(unchanged == 0xdeadbeef);

   /* EXP-0019 and the HAL200 Metal setter isolate the two-bit mode exactly:
    * clip=01, clamp=10. All other raster fields must survive the update. */
   uint32_t depth_clip;
   assert(agx_ppp_hal200_depth_clip_pack(AGX_CHIP_G17P, true,
                                         0xdeadbeef, &depth_clip));
   assert(depth_clip ==
          ((UINT32_C(0xdeadbeef) & ~UINT32_C(0xc00)) | UINT32_C(0x400)));
   assert(agx_ppp_hal200_depth_clip_pack(AGX_CHIP_G17P, false,
                                         0xdeadbeef, &depth_clip));
   assert(depth_clip ==
          ((UINT32_C(0xdeadbeef) & ~UINT32_C(0xc00)) | UINT32_C(0x800)));
   assert(agx_ppp_hal200_depth_clip_pack(AGX_CHIP_G17P, true,
                                         0x480, &depth_clip));
   assert(depth_clip == UINT32_C(0x480));
   assert(agx_ppp_hal200_depth_clip_pack(AGX_CHIP_G17P, false,
                                         0x480, &depth_clip));
   assert(depth_clip == UINT32_C(0x880));
   unchanged = UINT32_C(0xdeadbeef);
   assert(!agx_ppp_hal200_depth_clip_pack(AGX_CHIP_G17G, true, 0,
                                          &unchanged));
   assert(!agx_ppp_hal200_depth_clip_pack(AGX_CHIP_G17P, true, 0, NULL));
   assert(unchanged == UINT32_C(0xdeadbeef));

   uint32_t secondary_discard, cull_discard;
   assert(agx_ppp_hal200_rasterizer_discard_pack(
      AGX_CHIP_G17P, true, UINT32_C(0x00000200), UINT32_C(0x00000480),
      &secondary_discard, &cull_discard));
   assert(secondary_discard == UINT32_C(0x00200200));
   assert(cull_discard == UINT32_C(0x00020480));
   assert(agx_ppp_hal200_rasterizer_discard_pack(
      AGX_CHIP_G17P, false, secondary_discard, cull_discard,
      &secondary_discard, &cull_discard));
   assert(secondary_discard == UINT32_C(0x00000200));
   assert(cull_discard == UINT32_C(0x00000480));
   assert(!agx_ppp_hal200_rasterizer_discard_pack(
      AGX_CHIP_G17G, true, 0, 0, &secondary_discard, &cull_discard));
   assert(!agx_ppp_hal200_rasterizer_discard_pack(
      AGX_CHIP_G17P, true, 0, 0, NULL, &cull_discard));
   assert(!agx_ppp_hal200_rasterizer_discard_pack(
      AGX_CHIP_G17P, true, 0, 0, &secondary_discard, NULL));

   /* GNOME's ordinary textured fragment sets the legacy helper-invocation
    * bit. Apple G17P f_plain/f_grad/f_lod/f_shadow all publish native ff2=0,
    * so HAL200 must retain triangle merging and the exact base record. */
   struct agx_ppp_hal200_fragment_control_fields fragment_fields = {
      .linked_control = UINT32_C(1) << 26,
   };
   uint32_t primary_control = UINT32_C(0x00040200);
   uint32_t secondary_control = UINT32_C(0x00000200);
   assert(agx_ppp_hal200_fragment_controls_pack(
      AGX_CHIP_G17P, &fragment_fields, primary_control, secondary_control,
      &primary_control, &secondary_control));
   assert(primary_control == UINT32_C(0x00040200));
   assert(secondary_control == UINT32_C(0x00000200));
   const uint32_t gnome_secondary_words[] = {
      secondary_control, UINT32_C(0x07e00000), UINT32_C(0x07e00000),
      UINT32_C(0x0001ffff),
   };
   check(AGX_PPP_HAL200_GRAPHICS_SECONDARY_FACE, gnome_secondary_words, 4,
         secondary_expected, sizeof(secondary_expected));

   fragment_fields.linked_control =
      (UINT32_C(1) << 29) | (UINT32_C(1) << 26);
   assert(agx_ppp_hal200_fragment_controls_pack(
      AGX_CHIP_G17P, &fragment_fields, UINT32_C(0x00040200),
      UINT32_C(0x00000200), &primary_control, &secondary_control));
   assert(primary_control == UINT32_C(0x00040200));
   assert(secondary_control == UINT32_C(0x20000200));

   /* Native multisampled depth metadata and coverage disable merging only in
    * SECONDARY; the primitive override is the sole dual-control producer. */
   fragment_fields = (struct agx_ppp_hal200_fragment_control_fields){
      .shader_tri_merge_disable = true,
   };
   assert(agx_ppp_hal200_fragment_controls_pack(
      AGX_CHIP_G17P, &fragment_fields, UINT32_C(0x00040200),
      UINT32_C(0x00200200), &primary_control, &secondary_control));
   assert(primary_control == UINT32_C(0x00040200));
   assert(secondary_control == UINT32_C(0x04200200));

   fragment_fields = (struct agx_ppp_hal200_fragment_control_fields){
      .linked_control = UINT32_C(2) << 29,
      .coverage_tri_merge_disable = true,
   };
   assert(agx_ppp_hal200_fragment_controls_pack(
      AGX_CHIP_G17P, &fragment_fields, UINT32_C(0x00040200),
      UINT32_C(0x00000200), &primary_control, &secondary_control));
   assert(primary_control == UINT32_C(0x00040200));
   assert(secondary_control == UINT32_C(0x44000200));

   fragment_fields = (struct agx_ppp_hal200_fragment_control_fields){
      .primitive_tri_merge_disable = true,
   };
   assert(agx_ppp_hal200_fragment_controls_pack(
      AGX_CHIP_G17P, &fragment_fields, UINT32_C(0x00040200),
      UINT32_C(0x00000200), &primary_control, &secondary_control));
   assert(primary_control == UINT32_C(0x04040200));
   assert(secondary_control == UINT32_C(0x04000200));

   primary_control = secondary_control = UINT32_C(0xdeadbeef);
   fragment_fields.linked_control = UINT32_C(4) << 29;
   assert(!agx_ppp_hal200_fragment_controls_pack(
      AGX_CHIP_G17P, &fragment_fields, 0, 0, &primary_control,
      &secondary_control));
   assert(!agx_ppp_hal200_fragment_controls_pack(
      AGX_CHIP_G17G, &fragment_fields, 0, 0, &primary_control,
      &secondary_control));
   assert(!agx_ppp_hal200_fragment_controls_pack(
      AGX_CHIP_G17P, NULL, 0, 0, &primary_control, &secondary_control));
   assert(!agx_ppp_hal200_fragment_controls_pack(
      AGX_CHIP_G17P, &fragment_fields, 0, 0, NULL, &secondary_control));
   assert(!agx_ppp_hal200_fragment_controls_pack(
      AGX_CHIP_G17P, &fragment_fields, 0, 0, &primary_control, NULL));
   assert(primary_control == UINT32_C(0xdeadbeef));
   assert(secondary_control == UINT32_C(0xdeadbeef));
   return 0;
}
