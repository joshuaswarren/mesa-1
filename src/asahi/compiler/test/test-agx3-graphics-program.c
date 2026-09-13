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

   /* Firefox/WebGL through Zink reached this exact six-source vertex shape:
    * one vertex buffer's base and clamp after the indexed texture prefix,
    * baseVertex/firstInstance, bindless texture/sampler heap pointers, a root
    * plus three descriptor-set pointers, and one promoted float constant at
    * half72.  Range count is only composer bookkeeping; each range becomes
    * ordinary LDDEV/COPY records, so it must not refuse the sixth source. */
   struct agx3_ioto_uniform_range webgl_vertex_ranges[] = {
      {.start_halfs = 8, .size_halfs = 4, .address = 0x1000},
      {.start_halfs = 12, .size_halfs = 2, .address = 0x2000},
      {.start_halfs = 14, .size_halfs = 4, .address = 0x3000},
      {.start_halfs = 0, .size_halfs = 8, .address = 0x4000},
      {.start_halfs = 28, .size_halfs = 16, .address = 0x5000},
      {.start_halfs = 72, .size_halfs = 2, .address = 0x6000},
   };
   const size_t webgl_vertex_range_count =
      sizeof(webgl_vertex_ranges) / sizeof(webgl_vertex_ranges[0]);
   struct agx3_g17p_graphics_program_desc webgl_vertex_desc = {
      .program_va = UINT64_C(0x10000300000),
      .shader_va = UINT64_C(0x100004003c0),
      .uniform_ranges = webgl_vertex_ranges,
      .uniform_range_count = webgl_vertex_range_count,
      .shader_binary_size = 192,
      .nr_gprs = 16,
   };
   struct agx3_g17p_graphics_program webgl_vertex_program;
   size_t webgl_vertex_bound = 0;
   CHECK(agx3_g17p_graphics_program_bound(
      AGX_CHIP_G17P, webgl_vertex_ranges, webgl_vertex_range_count,
      &webgl_vertex_bound));
   CHECK(agx3_g17p_graphics_program_generate(
      AGX_CHIP_G17P, &webgl_vertex_desc, &webgl_vertex_program));
   CHECK(webgl_vertex_program.length <= webgl_vertex_bound);
   const struct agx3_g17p_graphics_program webgl_vertex_baseline =
      webgl_vertex_program;
   for (unsigned i = 0; i < webgl_vertex_range_count; ++i) {
      webgl_vertex_ranges[i].address += 0x40;
      CHECK(agx3_g17p_graphics_program_generate(
         AGX_CHIP_G17P, &webgl_vertex_desc, &webgl_vertex_program));
      CHECK(webgl_vertex_program.length == webgl_vertex_baseline.length);
      CHECK(memcmp(webgl_vertex_program.bytes, webgl_vertex_baseline.bytes,
                   webgl_vertex_program.length) != 0);
      webgl_vertex_ranges[i].address -= 0x40;
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
   CHECK(agx3_g17p_rt0_bg_shader_resources_match(
      AGX3_G17P_RT0_BG_CLEAR,
      AGX3_G17P_R11G11B10_MRT_CLEAR_PUSH_HALFS, false));
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
   bg.clear_push_halfs = AGX3_G17P_R11G11B10_MRT_CLEAR_PUSH_HALFS;
   CHECK(agx3_g17p_rt0_bg_program_generate(AGX_CHIP_G17P, &bg,
                                           &bg_program));
   CHECK(bg_program.length > 0 &&
         bg_program.length <= AGX_IOTO_G17P_RGBA8_MRT_CLEAR_PROGRAM_SIZE);
   bg.clear_push_halfs = AGX3_G17P_R11G11B10_MRT_CLEAR_PUSH_HALFS - 1;
   CHECK(!agx3_g17p_rt0_bg_program_generate(AGX_CHIP_G17P, &bg,
                                            &bg_program));
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

   /* The native layered array EOT is an 84-byte, eight-half program. It keeps
    * the same single-attachment Ioto resource ABI and 128-byte LDSHDR fetch
    * span as the other admitted RT0 builtins. */
   eot.pbe_argument_va = UINT64_C(0x4000030000);
   eot.shader_binary_size = 84;
   eot.imageblock.sample_size_B = 8;
   eot.nr_gprs = 8;
   eot.staging_halfs = 8;
   eot.attachment_count = 1;
   eot.imageblock_offset1_B = 0;
   CHECK(agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &eot,
                                            &eot_program));
   CHECK(eot_program.length > 0 &&
         eot_program.length <= AGX_IOTO_G17P_RT0_EOT_PROGRAM_SIZE);
   eot.shader_binary_size = 110;
   eot.nr_gprs = 16;
   CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &eot,
                                             &eot_program));

   return 0;
}
