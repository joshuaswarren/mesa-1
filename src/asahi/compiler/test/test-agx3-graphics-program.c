/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_graphics_program.h"

#include <stdio.h>
#include <string.h>

#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__,  \
                 #x);                                                          \
         return 1;                                                             \
      }                                                                        \
   } while (0)

static int
test_g17g_programs(void)
{
   /* Native M5 triangle's complete state records, independently captured
    * from the currently compiled shader. Runtime D remains unqualified
    * until these generated programs execute with Mesa-owned resources. */
   static const uint8_t expected_tail[2][28] = {
      {0x07,0x00,0x00,0x0c,0x00,0x00,0x00,0x00,
       0x87,0x04,0x48,0x0e,0x00,0x00,0x00,0x00,0x00,0x01,
       0x87,0x00,0x00,0x0d,0x00,0x00,0x00,0x00,0x00,0x00},
      {0x17,0x00,0x00,0x2c,0x00,0x00,0x18,0x08,
       0xc7,0x03,0x40,0x0e,0x00,0x00,0x00,0x00,0x00,0x01,
       0x07,0x00,0x00,0x0d,0x00,0x00,0x00,0x00,0x00,0x00},
   };
   for (unsigned fragment = 0; fragment < 2; ++fragment) {
      struct agx3_g17p_graphics_program_desc desc = {
         .program_va = UINT64_C(0x10000100000),
         .shader_va = fragment ? UINT64_C(0x100000003c0)
                              : UINT64_C(0x10000000480),
         .shader_binary_size = fragment ? 60 : 128,
         .nr_gprs = 16,
         .compiler_d_valid = true,
         .compiler_d = fragment ? 0 : 32,
         .fragment = fragment,
         .imageblock = AGX_IOTO_G17P_IMAGEBLOCK_8B_1X_32X32,
      };
      struct agx3_g17p_graphics_program output;
      size_t empty_bound;
      CHECK(agx3_g17g_graphics_program_bound(AGX_CHIP_G17G, NULL, 0,
                                              &empty_bound));
      CHECK(empty_bound == 90);
      CHECK(agx3_g17g_graphics_program_generate(AGX_CHIP_G17G, &desc,
                                                 &output));
      CHECK(output.length == (fragment ? 90 : 84));
      CHECK(memcmp(output.bytes, expected_tail[fragment], 28) == 0);

      struct agx3_ioto_uniform_range ranges[] = {
         {.start_halfs = 0, .size_halfs = 4, .address = 0x4000001000},
         {.start_halfs = 4, .size_halfs = 2, .address = 0x4000002000},
      };
      desc.uniform_ranges = ranges;
      desc.uniform_range_count = 2;
      size_t bound;
      CHECK(agx3_g17g_graphics_program_bound(AGX_CHIP_G17G, ranges, 2, &bound));
      CHECK(agx3_g17g_graphics_program_generate(AGX_CHIP_G17G, &desc,
                                                 &output));
      CHECK(output.length > 90 && output.length <= bound);
      struct agx3_g17p_graphics_program baseline = output;
      for (unsigned i = 0; i < 2; ++i) {
         ranges[i].address += 64;
         CHECK(agx3_g17g_graphics_program_generate(AGX_CHIP_G17G, &desc,
                                                    &output));
         CHECK(output.length == baseline.length &&
               memcmp(output.bytes, baseline.bytes, output.length) != 0);
         ranges[i].address -= 64;
      }

      baseline = output;
      desc.scratch_size = 4;
      CHECK(!agx3_g17g_graphics_program_generate(AGX_CHIP_G17G, &desc, &output));
      desc.scratch_size = 0;
      desc.compiler_d_valid = false;
      CHECK(!agx3_g17g_graphics_program_generate(AGX_CHIP_G17G, &desc, &output));
      desc.compiler_d_valid = true;
      desc.shader_va++;
      CHECK(!agx3_g17g_graphics_program_generate(AGX_CHIP_G17G, &desc, &output));
      desc.shader_va--;
      CHECK(!agx3_g17g_graphics_program_generate(AGX_CHIP_G17P, &desc, &output));
      CHECK(memcmp(&output, &baseline, sizeof(output)) == 0);
      if (fragment) {
         static const uint8_t bg_end[] = {7, 0x10, 0, 4, 0, 0, 14, 0, 0, 0};
         static const uint8_t eot_end[] = {14, 0, 0, 0};
         CHECK(agx3_g17g_tile_program_generate(AGX_CHIP_G17G, &desc, false,
                                               &output));
         CHECK(output.length == baseline.length);
         CHECK(memcmp(output.bytes + output.length - sizeof(bg_end),
                      bg_end, sizeof(bg_end)) == 0);
         CHECK(agx3_g17g_tile_program_generate(AGX_CHIP_G17G, &desc, true,
                                               &output));
         CHECK(output.length + 6 == baseline.length);
         CHECK(memcmp(output.bytes + output.length - sizeof(eot_end),
                      eot_end, sizeof(eot_end)) == 0);
      } else {
         CHECK(!agx3_g17g_tile_program_generate(AGX_CHIP_G17G, &desc, false,
                                                &output));
      }
   }
   return 0;
}

int
main(void)
{
   CHECK(test_g17g_programs() == 0);
   /* Glamor's instanced rectangle needs six independent VS ranges: VBO
    * bases, clamps, draw parameters, UBO pointer, provoking vertex and clip
    * coefficient. The old five-range limit silently left its pipeline zero. */
   struct agx3_ioto_uniform_range rectangle_ranges[] = {
      {.start_halfs = 0, .size_halfs = 8, .address = 0x4000001078},
      {.start_halfs = 8, .size_halfs = 4, .address = 0x4000001178},
      {.start_halfs = 12, .size_halfs = 4, .address = 0x400000000c},
      {.start_halfs = 16, .size_halfs = 4, .address = 0x4000000068},
      {.start_halfs = 20, .size_halfs = 2, .address = 0x4000001290},
      {.start_halfs = 22, .size_halfs = 2, .address = 0x40000012a0},
   };
   const size_t rectangle_count =
      sizeof(rectangle_ranges) / sizeof(rectangle_ranges[0]);
   size_t rectangle_bound = 0;
   CHECK(agx3_g17p_graphics_program_bound(
      AGX_CHIP_G17P, rectangle_ranges, rectangle_count, &rectangle_bound));
   struct agx3_g17p_graphics_program_desc rectangle_desc = {
      .program_va = UINT64_C(0x10000100000),
      .shader_va = UINT64_C(0x100002003c0),
      .uniform_ranges = rectangle_ranges,
      .uniform_range_count = rectangle_count,
      .shader_binary_size = 282,
      .nr_gprs = 29,
      .compiler_d_valid = true,
      .compiler_d = 20,
   };
   struct agx3_g17p_graphics_program rectangle;
   CHECK(agx3_g17p_graphics_program_generate(
      AGX_CHIP_G17P, &rectangle_desc, &rectangle));
   CHECK(rectangle.length > 0 && rectangle.length <= rectangle_bound);
   const struct agx3_g17p_graphics_program rectangle_baseline = rectangle;
   for (unsigned i = 0; i < rectangle_count; ++i) {
      rectangle_ranges[i].address += 0x40;
      CHECK(agx3_g17p_graphics_program_generate(
         AGX_CHIP_G17P, &rectangle_desc, &rectangle));
      CHECK(rectangle.length == rectangle_baseline.length);
      CHECK(memcmp(rectangle.bytes, rectangle_baseline.bytes,
                   rectangle.length) != 0);
      rectangle_ranges[i].address -= 0x40;
   }

   /* More ranges must not bypass the existing program byte budget. */
   struct agx3_ioto_uniform_range oversized_ranges[128];
   for (unsigned i = 0; i < 128; ++i) {
      oversized_ranges[i] = (struct agx3_ioto_uniform_range){
         .start_halfs = 2 * i, .size_halfs = 2,
         .address = UINT64_C(0x4000010000) + 4 * i,
      };
   }
   const size_t previous_bound = rectangle_bound;
   CHECK(!agx3_g17p_graphics_program_bound(
      AGX_CHIP_G17P, oversized_ranges, 128, &rectangle_bound));
   CHECK(rectangle_bound == previous_bound);
   rectangle_desc.uniform_ranges = oversized_ranges;
   rectangle_desc.uniform_range_count = 128;
   const struct agx3_g17p_graphics_program previous_program = rectangle;
   CHECK(!agx3_g17p_graphics_program_generate(
      AGX_CHIP_G17P, &rectangle_desc, &rectangle));
   CHECK(memcmp(&rectangle, &previous_program, sizeof(rectangle)) == 0);

   uint16_t blend_base = UINT16_C(0xa5a5);
   CHECK(agx3_g17p_fragment_blend_constant_base(AGX_CHIP_G17P, 0,
                                                 &blend_base));
   CHECK(blend_base == 8);
   CHECK(agx3_g17p_fragment_blend_constant_base(AGX_CHIP_G17P, 16,
                                                 &blend_base));
   CHECK(blend_base == 16);
   CHECK(agx3_g17p_fragment_blend_constant_base(AGX_CHIP_G17P, 17,
                                                 &blend_base));
   CHECK(blend_base == 18);
   CHECK(agx3_g17p_fragment_blend_constant_base(AGX_CHIP_G17P, 504,
                                                 &blend_base));
   CHECK(blend_base == 504);
   const uint16_t valid_blend_base = blend_base;
   CHECK(!agx3_g17p_fragment_blend_constant_base(AGX_CHIP_G17P, 505,
                                                  &blend_base));
   CHECK(blend_base == valid_blend_base);
   CHECK(!agx3_g17p_fragment_blend_constant_base(AGX_CHIP_G17S, 16,
                                                  &blend_base));
   CHECK(blend_base == valid_blend_base);
   CHECK(!agx3_g17p_fragment_blend_constant_base(AGX_CHIP_G17P, 16, NULL));

   /* Exact compositor uniform geometry: the texture/sampler pointer pair,
    * root pointer, and relocated four-float blend constant each occupy a
    * disjoint native range.  Every source must independently change the
    * generated CProg bytes. */
   CHECK(agx3_g17p_fragment_blend_constant_base(AGX_CHIP_G17P, 16,
                                                 &blend_base));
   struct agx3_ioto_uniform_range compositor_ranges[] = {
      {.start_halfs = 0, .size_halfs = 8, .address = 0x1000},
      {.start_halfs = 12, .size_halfs = 4, .address = 0x2000},
      {.start_halfs = blend_base, .size_halfs = 8, .address = 0x3000},
   };
   const size_t compositor_range_count =
      sizeof(compositor_ranges) / sizeof(compositor_ranges[0]);
   struct agx3_g17p_graphics_program_desc compositor_desc = {
      .program_va = UINT64_C(0x10000100000),
      .shader_va = UINT64_C(0x100002003c0),
      .uniform_ranges = compositor_ranges,
      .uniform_range_count = compositor_range_count,
      .shader_binary_size = 128,
      .nr_gprs = 16,
      .fragment = true,
      .imageblock = AGX_IOTO_G17P_IMAGEBLOCK_8B_1X_32X32,
   };
   struct agx3_g17p_graphics_program compositor_program;
   CHECK(agx3_g17p_graphics_program_generate(
      AGX_CHIP_G17P, &compositor_desc, &compositor_program));
   CHECK(compositor_program.length >
         AGX_IOTO_G17P_FRAGMENT_NO_RESOURCE_PROGRAM_SIZE);
   const struct agx3_g17p_graphics_program compositor_baseline =
      compositor_program;
   struct agx3_g17p_graphics_program_desc mixed_desc = compositor_desc;
   mixed_desc.imageblock.sample_size_B = 16;
   struct agx3_g17p_graphics_program mixed_program;
   CHECK(agx3_g17p_graphics_program_generate(AGX_CHIP_G17P, &mixed_desc,
                                              &mixed_program));
   CHECK(mixed_program.length == compositor_baseline.length);
   unsigned imageblock_differences = 0;
   for (unsigned i = 0; i < mixed_program.length; ++i) {
      if (mixed_program.bytes[i] != compositor_baseline.bytes[i]) {
         CHECK(compositor_baseline.bytes[i] == 0x43);
         CHECK(mixed_program.bytes[i] == 0x45);
         imageblock_differences++;
      }
   }
   CHECK(imageblock_differences == 1);
   for (unsigned i = 0; i < compositor_range_count; ++i) {
      compositor_ranges[i].address += 0x40;
      CHECK(agx3_g17p_graphics_program_generate(
         AGX_CHIP_G17P, &compositor_desc, &compositor_program));
      CHECK(compositor_program.length == compositor_baseline.length);
      CHECK(memcmp(compositor_program.bytes, compositor_baseline.bytes,
                   compositor_program.length) != 0);
      compositor_ranges[i].address -= 0x40;
   }

   uint64_t clear_spec = 0, reload_spec = 0, none_spec = 0;
   CHECK(agx3_g17p_rt0_bg_resource_spec(
      AGX_CHIP_G17P, AGX3_G17P_RT0_BG_CLEAR, &clear_spec));
   CHECK(agx3_g17p_rt0_bg_resource_spec(
      AGX_CHIP_G17P, AGX3_G17P_RT0_BG_RELOAD, &reload_spec));
   CHECK(agx3_g17p_rt0_bg_resource_spec(
      AGX_CHIP_G17P, AGX3_G17P_RT0_BG_NONE, &none_spec));
   CHECK(clear_spec == reload_spec && clear_spec != none_spec);
   CHECK((clear_spec >> 47) == UINT64_C(0x1ffff));
   CHECK(!agx3_g17p_rt0_bg_resource_spec(
      AGX_CHIP_G17S, AGX3_G17P_RT0_BG_CLEAR, &clear_spec));
   CHECK(agx3_g17p_rt0_bg_shader_resources_match(
      AGX3_G17P_RT0_BG_CLEAR, AGX3_G17P_RT0_CLEAR_PUSH_HALFS, false));
   CHECK(agx3_g17p_rt0_bg_shader_resources_match(
      AGX3_G17P_RT0_BG_CLEAR, AGX3_G17P_RGBA8_MRT_CLEAR_PUSH_HALFS, false));
   CHECK(!agx3_g17p_rt0_bg_shader_resources_match(
      AGX3_G17P_RT0_BG_CLEAR, AGX3_G17P_RT0_CLEAR_PUSH_HALFS, true));
   CHECK(agx3_g17p_rt0_bg_shader_resources_match(
      AGX3_G17P_RT0_BG_RELOAD, 0, true));

   struct agx3_g17p_rt0_bg_program_desc bg = {
      .program_va = UINT64_C(0x10000100000),
      .shader_va = UINT64_C(0x100002003c0),
      .clear_value_va = UINT64_C(0x4000010000),
      .shader_binary_size = 84,
      .nr_gprs = 8,
      .staging_known = true,
      .clear_push_halfs = AGX3_G17P_RT0_CLEAR_PUSH_HALFS,
      .op = AGX3_G17P_RT0_BG_CLEAR,
      .imageblock = AGX_IOTO_G17P_IMAGEBLOCK_8B_1X_32X32,
   };
   struct agx3_g17p_rt0_bg_program bg_program;
   memset(&bg_program, 0xa5, sizeof(bg_program));
   CHECK(agx3_g17p_rt0_bg_program_generate(AGX_CHIP_G17P, &bg,
                                           &bg_program));
   CHECK(bg_program.length > 0 &&
         bg_program.length <= AGX_IOTO_G17P_RT0_CLEAR_PROGRAM_SIZE);
   bg.clear_push_halfs = AGX3_G17P_RGBA8_MRT_CLEAR_PUSH_HALFS;
   CHECK(agx3_g17p_rt0_bg_program_generate(AGX_CHIP_G17P, &bg,
                                           &bg_program));
   CHECK(bg_program.length > 0 &&
         bg_program.length <= AGX_IOTO_G17P_RGBA8_MRT_CLEAR_PROGRAM_SIZE);
   bg.clear_value_va = 0;
   CHECK(!agx3_g17p_rt0_bg_program_generate(AGX_CHIP_G17P, &bg,
                                            &bg_program));

   bg.clear_value_va = 0;
   bg.texture_pointer_va = UINT64_C(0x4000020000);
   bg.sampler_pointer_va = UINT64_C(0x4000020008);
   bg.clear_push_halfs = 0;
   bg.op = AGX3_G17P_RT0_BG_RELOAD;
   CHECK(agx3_g17p_rt0_bg_program_generate(AGX_CHIP_G17P, &bg,
                                           &bg_program));
   CHECK(bg_program.length > 0 &&
         bg_program.length <= AGX_IOTO_G17P_RT0_RELOAD_PROGRAM_MAX_SIZE);
   bg.sampler_pointer_va = 0;
   CHECK(!agx3_g17p_rt0_bg_program_generate(AGX_CHIP_G17P, &bg,
                                            &bg_program));

   struct agx3_g17p_rt0_eot_program_desc eot = {
      .program_va = UINT64_C(0x10000300000),
      .shader_va = UINT64_C(0x100004003c0),
      .pbe_argument_va = UINT64_C(0x4000030000),
      .attachment_count = 1,
      .shader_binary_size = 84,
      .nr_gprs = 8,
      .staging_halfs = 8,
      .imageblock = AGX_IOTO_G17P_IMAGEBLOCK_8B_1X_32X32,
   };
   struct agx3_g17p_rt0_eot_program eot_program;
   CHECK(agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &eot,
                                            &eot_program));
   CHECK(eot_program.length > 0 &&
         eot_program.length <= AGX_IOTO_G17P_RT0_EOT_PROGRAM_SIZE);
   eot.pbe_argument_va++;
   CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &eot,
                                             &eot_program));
   eot.pbe_argument_va = UINT64_C(0x4000030000);
   eot.shader_binary_size = 120;
   eot.staging_halfs = 16;
   eot.attachment_count = 2;
   eot.imageblock_offset1_B = 4;
   CHECK(agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &eot,
                                            &eot_program));
   eot.imageblock_offset1_B = 8;
   CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &eot,
                                             &eot_program));
   eot.shader_binary_size = 238;
   eot.imageblock.sample_size_B = 16;
   CHECK(agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &eot,
                                            &eot_program));
   eot.imageblock_offset1_B = 4;
   CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &eot,
                                             &eot_program));
   eot.imageblock_offset1_B = 12;
   CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &eot,
                                             &eot_program));

   /* The exact layered array-wire EOT grows to 110 bytes and its r0..r6
    * coordinate prefix rounds to a sixteen-half GPR allocation. It keeps the
    * same single-attachment Ioto resource ABI and 128-byte LDSHDR fetch span. */
   eot.pbe_argument_va = UINT64_C(0x4000030000);
   eot.shader_binary_size = 110;
   eot.imageblock.sample_size_B = 8;
   eot.nr_gprs = 16;
   eot.staging_halfs = 8;
   eot.attachment_count = 1;
   eot.imageblock_offset1_B = 0;
   CHECK(agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &eot,
                                            &eot_program));
   CHECK(eot_program.length > 0 &&
         eot_program.length <= AGX_IOTO_G17P_RT0_EOT_PROGRAM_SIZE);
   eot.nr_gprs = 15;
   CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &eot,
                                             &eot_program));

   return 0;
}
