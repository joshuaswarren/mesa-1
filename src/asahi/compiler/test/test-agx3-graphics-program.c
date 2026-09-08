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

int
main(void)
{
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
   eot.imageblock_offset1_B = 12;
   CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &eot,
                                             &eot_program));

   /* The exact layered array-wire EOT grows to 110 bytes and its r0..r6
    * coordinate prefix rounds to a sixteen-half GPR allocation. It keeps the
    * same single-attachment Ioto resource ABI and 128-byte LDSHDR fetch span. */
   eot.pbe_argument_va--;
   eot.shader_binary_size = 110;
   eot.nr_gprs = 16;
   CHECK(agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &eot,
                                            &eot_program));
   CHECK(eot_program.length > 0 &&
         eot_program.length <= AGX_IOTO_G17P_RT0_EOT_PROGRAM_SIZE);
   eot.nr_gprs = 15;
   CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &eot,
                                             &eot_program));

   return 0;
}
