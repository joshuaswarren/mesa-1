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

static uint8_t *
eot_offset(struct agx3_g17p_rt0_eot_program_desc *desc, unsigned rt)
{
   uint8_t *const offsets[AGX3_G17P_MRT_MAX_TARGETS] = {
      &desc->imageblock_offset_B,  &desc->imageblock_offset1_B,
      &desc->imageblock_offset2_B, &desc->imageblock_offset3_B,
      &desc->imageblock_offset4_B, &desc->imageblock_offset5_B,
      &desc->imageblock_offset6_B, &desc->imageblock_offset7_B,
   };
   return offsets[rt];
}

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
   bg.clear_push_halfs = AGX3_G17P_TRIPLE_CLEAR_PUSH_HALFS;
   CHECK(agx3_g17p_rt0_bg_shader_resources_match(
      AGX3_G17P_RT0_BG_CLEAR, bg.clear_push_halfs, false));
   CHECK(agx3_g17p_rt0_bg_program_generate(AGX_CHIP_G17P, &bg,
                                           &bg_program));
   CHECK(bg_program.length > 0 &&
         bg_program.length <= AGX_IOTO_G17P_QUAD_CLEAR_PROGRAM_MAX_SIZE);
   for (unsigned count = 5; count <= AGX3_G17P_MRT_MAX_TARGETS; ++count) {
      bg.clear_push_halfs = 4 + 8 * count;
      CHECK(agx3_g17p_rt0_bg_program_generate(AGX_CHIP_G17P, &bg,
                                              &bg_program));
      CHECK(bg_program.length > 0 &&
            bg_program.length <=
               AGX_IOTO_G17P_OCTA_BACKGROUND_PROGRAM_MAX_SIZE);
   }
   bg.clear_push_halfs = AGX3_G17P_OCTA_CLEAR_PUSH_HALFS + 8;
   CHECK(!agx3_g17p_rt0_bg_program_generate(AGX_CHIP_G17P, &bg,
                                            &bg_program));
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

   /* Reloaded targets precede cleared ones. Each clear value keeps its
    * target's eight halves, which start after the texture/sampler pointers
    * only when at least one target reloads. */
   CHECK(agx3_g17p_rt0_bg_resource_spec(
      AGX_CHIP_G17P, AGX3_G17P_RT0_BG_CLEAR_RELOAD, &clear_spec));
   CHECK(clear_spec == reload_spec);
   CHECK(agx3_g17p_rt0_bg_shader_resources_match(
      AGX3_G17P_RT0_BG_CLEAR_RELOAD, AGX3_G17P_QUAD_CLEAR_PUSH_HALFS, true));
   CHECK(!agx3_g17p_rt0_bg_shader_resources_match(
      AGX3_G17P_RT0_BG_CLEAR_RELOAD, AGX3_G17P_QUAD_CLEAR_PUSH_HALFS, false));
   CHECK(!agx3_g17p_rt0_bg_shader_resources_match(
      AGX3_G17P_RT0_BG_CLEAR_RELOAD, AGX3_G17P_RT0_CLEAR_PUSH_HALFS, true));
   /* Five to eight mixed-format targets: eight halves per clear value. */
   for (unsigned count = 5; count <= AGX3_G17P_MRT_MAX_TARGETS; ++count) {
      CHECK(agx3_g17p_rt0_bg_shader_resources_match(
         AGX3_G17P_RT0_BG_CLEAR, 4 + 8 * count, false));
      CHECK(agx3_g17p_rt0_bg_shader_resources_match(
         AGX3_G17P_RT0_BG_CLEAR_RELOAD, 4 + 8 * count, true));
   }
   CHECK(!agx3_g17p_rt0_bg_shader_resources_match(
      AGX3_G17P_RT0_BG_CLEAR, AGX3_G17P_OCTA_CLEAR_PUSH_HALFS + 8, false));
   bg.sampler_pointer_va = UINT64_C(0x4000020008);
   bg.clear_value_va = UINT64_C(0x4000010000);
   bg.op = AGX3_G17P_RT0_BG_CLEAR_RELOAD;
   for (unsigned count = 2; count <= AGX3_G17P_MRT_MAX_TARGETS; ++count) {
      bg.clear_push_halfs = 4 + 8 * count;
      for (unsigned reloads = 1; reloads < count; ++reloads) {
         bg.clear_start_halfs = 4 + 8 * reloads;
         CHECK(agx3_g17p_rt0_bg_program_generate(AGX_CHIP_G17P, &bg,
                                                 &bg_program));
         CHECK(bg_program.length > 0 &&
               bg_program.length <=
                  (count > 4 ? AGX_IOTO_G17P_OCTA_BACKGROUND_PROGRAM_MAX_SIZE
                             : AGX_IOTO_G17P_CLEAR_RELOAD_PROGRAM_MAX_SIZE));
      }
      bg.clear_start_halfs = 4;
      CHECK(!agx3_g17p_rt0_bg_program_generate(AGX_CHIP_G17P, &bg,
                                               &bg_program));
      bg.clear_start_halfs = 4 + 8 * count;
      CHECK(!agx3_g17p_rt0_bg_program_generate(AGX_CHIP_G17P, &bg,
                                               &bg_program));
      bg.clear_start_halfs = 13;
      CHECK(!agx3_g17p_rt0_bg_program_generate(AGX_CHIP_G17P, &bg,
                                               &bg_program));
   }
   bg.clear_start_halfs = 12;
   bg.clear_value_va = 0;
   CHECK(!agx3_g17p_rt0_bg_program_generate(AGX_CHIP_G17P, &bg,
                                            &bg_program));
   bg.clear_value_va = UINT64_C(0x4000010000);
   bg.texture_pointer_va = 0;
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
   /* Mixed RGBA16F/R8 uses the same 120-byte control flow, with offsets0/8
    * inside sample stride16; the native format selector differs. */
   eot.imageblock.sample_size_B = 16;
   CHECK(agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &eot,
                                            &eot_program));
   eot.imageblock_offset1_B = 4;
   CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &eot,
                                             &eot_program));
   eot.imageblock_offset1_B = 12;
   CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &eot,
                                              &eot_program));
   /* The old partition-one 238-byte program is invalid for this layout. */
   eot.imageblock_offset1_B = 8;
   eot.shader_binary_size = 238;
   CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &eot,
                                              &eot_program));

   /* Four native PBE entries share one argument table. Both witnessed tile
    * strides use the same shader and sparse offset uniforms. */
   for (unsigned stride = 4; stride <= 16; stride *= 4) {
      struct agx3_g17p_rt0_eot_program_desc quad = eot;
      quad.shader_binary_size = 208;
      quad.staging_halfs = 32;
      quad.attachment_count = 4;
      quad.imageblock_offset1_B = stride;
      quad.imageblock_offset2_B = 2 * stride;
      quad.imageblock_offset3_B = 3 * stride;
      quad.imageblock.sample_size_B = 4 * stride;
      quad.imageblock.tile_size_px = stride == 16 ? 16 * 16 : 32 * 32;
      CHECK(agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &quad,
                                             &eot_program));
      CHECK(eot_program.length == 144);
      quad.imageblock_offset3_B++;
      CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &quad,
                                              &eot_program));
      quad.imageblock_offset3_B--;
      quad.staging_halfs = 16;
      CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &quad,
                                              &eot_program));
      quad.staging_halfs = 32;
      quad.attachment_count = 3;
      CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &quad,
                                              &eot_program));
      quad.attachment_count = 4;
      quad.pbe_argument_va++;
      CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &quad,
                                              &eot_program));
      quad.pbe_argument_va = UINT64_C(0xffffffffffe0);
      CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &quad,
                                              &eot_program));
   }

   /* Three RGBA8 attachments occupy twelve bytes but the tilebuffer rounds
    * the sample stride to sixteen. The native EOT still publishes offsets
    * 0, 4 and 8, with no fourth offset operand. */
   struct agx3_g17p_rt0_eot_program_desc rgba8_three = eot;
   rgba8_three.shader_binary_size = 164;
   rgba8_three.staging_halfs = 24;
   rgba8_three.attachment_count = 3;
   rgba8_three.imageblock_offset1_B = 4;
   rgba8_three.imageblock_offset2_B = 8;
   rgba8_three.imageblock_offset3_B = 0;
   rgba8_three.imageblock.sample_size_B = 16;
   rgba8_three.imageblock.tile_size_px = 32 * 32;
   CHECK(agx3_g17p_rt0_eot_program_generate(
      AGX_CHIP_G17P, &rgba8_three, &eot_program));
   CHECK(eot_program.length > 0 &&
         eot_program.length <= AGX_IOTO_G17P_QUAD_EOT_PROGRAM_MAX_SIZE);
   rgba8_three.imageblock.sample_size_B = 12;
   CHECK(!agx3_g17p_rt0_eot_program_generate(
      AGX_CHIP_G17P, &rgba8_three, &eot_program));

   /* Two and three RGBA32 targets retain the same sparse per-attachment
    * offset ABI, with only their actual descriptor and staging counts. */
   for (unsigned count = 2; count <= 3; ++count) {
      struct agx3_g17p_rt0_eot_program_desc raw = eot;
      raw.shader_binary_size = 32 + 44 * count;
      raw.staging_halfs = 8 * count;
      raw.attachment_count = count;
      raw.imageblock_offset1_B = 16;
      raw.imageblock_offset2_B = count > 2 ? 32 : 0;
      raw.imageblock_offset3_B = 0;
      raw.imageblock.sample_size_B = 16 * count;
      raw.imageblock.tile_size_px = count == 2 ? 32 * 16 : 16 * 16;
      CHECK(agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &raw,
                                             &eot_program));
      CHECK(eot_program.length > 0 && eot_program.length <= 144);
      raw.imageblock_offset3_B = 48;
      CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &raw,
                                              &eot_program));
      raw.imageblock_offset3_B = 0;
      raw.staging_halfs -= 8;
      CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &raw,
                                              &eot_program));
      raw.staging_halfs += 8;
      raw.attachment_count++;
      CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &raw,
                                              &eot_program));
      raw.attachment_count--;
      raw.pbe_argument_va = UINT64_C(0xffffffffffe0);
      CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &raw,
                                              &eot_program));
   }

   /* Targets of arbitrary formats keep the multi-target program and offset
    * uniforms, but each target may sit at any offset inside the sample. */
   for (unsigned count = 2; count <= AGX3_G17P_MRT_MAX_TARGETS; ++count) {
      struct agx3_g17p_rt0_eot_program_desc mixed = eot;
      static const uint8_t offsets[AGX3_G17P_MRT_MAX_TARGETS] = {
         16, 24, 0, 8, 32, 40, 48, 56};
      mixed.shader_binary_size = 32 + 44 * count;
      mixed.staging_halfs = 8 * count;
      mixed.attachment_count = count;
      for (unsigned rt = 0; rt < AGX3_G17P_MRT_MAX_TARGETS; ++rt)
         *eot_offset(&mixed, rt) = rt < count ? offsets[rt] : 0;
      mixed.imageblock.sample_size_B = count > 4 ? 64 : 32;
      mixed.imageblock.tile_size_px = count > 4 ? 16 * 16 : 32 * 16;
      CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &mixed,
                                              &eot_program));
      mixed.mixed_mrt = true;
      CHECK(agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &mixed,
                                             &eot_program));
      /* Each target beyond the second adds one offset LDIMM and two COPYs. */
      CHECK(eot_program.length > 0 &&
            eot_program.length <= (count > 4
                                      ? AGX_IOTO_G17P_OCTA_EOT_PROGRAM_MAX_SIZE
                                      : 144));

      struct agx3_g17p_rt0_eot_program_desc bad = mixed;
      bad.imageblock_offset1_B = mixed.imageblock.sample_size_B;
      CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &bad,
                                              &eot_program));
      bad = mixed;
      if (count < AGX3_G17P_MRT_MAX_TARGETS) {
         *eot_offset(&bad, count) = 4;
         CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &bad,
                                                 &eot_program));
      }
      bad = mixed;
      bad.staging_halfs -= 8;
      CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &bad,
                                              &eot_program));
      /* The multisample program has the same shape; the imageblock carries
       * the count. */
      if (count <= 4) {
         bad = mixed;
         bad.imageblock.nr_samples = 2;
         bad.imageblock.tile_size_px = 16 * 16;
         CHECK(agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &bad,
                                                &eot_program));
      }
      bad = mixed;
      bad.imageblock.nr_samples = 3;
      CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &bad,
                                              &eot_program));
      bad = mixed;
      bad.pbe_argument_va = UINT64_C(0xffffffffffe0);
      CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &bad,
                                              &eot_program));
   }

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

   /* Native builtin metadata: text lengths, GPR allocation and staging D
    * are independently captured for each dense attachment count. */
   const unsigned native_sizes[] = {164, 208, 252, 296, 340, 384};
   const unsigned native_staging[] = {24, 32, 40, 48, 56, 64};
   /* All logical masks use physical partition zero; only the number of
    * compact active PBEs changes the native builtin and its staging bound. */
   const unsigned native_all_sizes[] = {68, 120, 164, 208, 252, 296, 340, 384};
   for (unsigned mask = 1; mask < 256; ++mask) {
      unsigned count = __builtin_popcount(mask);
      eot.attachment_mask = mask;
      eot.attachment_count = count;
      eot.shader_binary_size = native_all_sizes[count - 1];
      eot.nr_gprs = 8;
      eot.staging_halfs = 8 * count;
      eot.imageblock.sample_size_B = (4 * count + 7) & ~7u;
      eot.imageblock.tile_size_px = count >= 7 ? 512 : 1024;
      for (unsigned i = 0; i < count; ++i)
         eot.imageblock_offsets_B[i] = 4 * i;
      CHECK(agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &eot,
                                               &eot_program));
      const struct agx3_g17p_rt0_eot_program before = eot_program;
      eot.attachment_count++;
      CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &eot,
                                                &eot_program));
      CHECK(!memcmp(&before, &eot_program, sizeof(before)));
      eot.attachment_count = count;

      /* An explicit active mask must not fall back to a legacy RT0/dual
       * descriptor when its compact-layout metadata is invalid. */
      eot.imageblock_offset_B = 4;
      CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &eot,
                                                &eot_program));
      CHECK(!memcmp(&before, &eot_program, sizeof(before)));
      eot.imageblock_offset_B = 0;

      eot.imageblock.nr_samples = 2;
      CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &eot,
                                                &eot_program));
      CHECK(!memcmp(&before, &eot_program, sizeof(before)));
      eot.imageblock.nr_samples = 1;
      eot.imageblock.sample_size_B += 8;
      CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &eot,
                                                &eot_program));
      CHECK(!memcmp(&before, &eot_program, sizeof(before)));
      eot.imageblock.sample_size_B -= 8;

      if (count == 1) {
         eot.attachment_count = 0;
         CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &eot,
                                                   &eot_program));
         CHECK(!memcmp(&before, &eot_program, sizeof(before)));
      }
   }
   eot.attachment_mask = 0;
   for (unsigned count = 3; count <= 8; ++count) {
      fprintf(stderr, "Testing dense MRT loader count=%u\n", count);
      eot.attachment_count = count;
      eot.shader_binary_size = native_sizes[count - 3];
      eot.nr_gprs = 8;
      eot.staging_halfs = native_staging[count - 3];
      eot.imageblock.sample_size_B = (4 * count + 7) & ~7u;
      eot.imageblock.tile_size_px = count >= 7 ? 512 : 1024;
      for (unsigned rt = 0; rt < count; ++rt)
         eot.imageblock_offsets_B[rt] = 4 * rt;
      CHECK(agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &eot,
                                               &eot_program));
      CHECK(eot_program.length <= AGX3_G17P_WIDE_MRT_EOT_PROGRAM_MAX_SIZE);
      const struct agx3_g17p_rt0_eot_program before = eot_program;
      eot.imageblock_offsets_B[count - 1] += 4;
      CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &eot,
                                                &eot_program));
      CHECK(memcmp(&eot_program, &before, sizeof(before)) == 0);
      eot.imageblock_offsets_B[count - 1] -= 4;
      eot.staging_halfs--;
      CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &eot,
                                                &eot_program));
      eot.staging_halfs++;
      eot.shader_binary_size--;
      CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &eot,
                                                &eot_program));

      bg.op = AGX3_G17P_RT0_BG_CLEAR;
      bg.clear_value_va = UINT64_C(0x4000010000);
      bg.texture_pointer_va = 0;
      bg.sampler_pointer_va = 0;
      bg.clear_push_halfs = 4 + 8 * count;
      bg.imageblock = eot.imageblock;
      CHECK(agx3_g17p_rt0_bg_shader_resources_match(bg.op,
                                                    bg.clear_push_halfs, false));
      CHECK(agx3_g17p_rt0_bg_program_generate(AGX_CHIP_G17P, &bg,
                                              &bg_program));
      CHECK(bg_program.length <= AGX3_G17P_WIDE_MRT_BG_PROGRAM_MAX_SIZE);
      bg.clear_push_halfs--;
      CHECK(!agx3_g17p_rt0_bg_program_generate(AGX_CHIP_G17P, &bg,
                                               &bg_program));
   }

   return 0;
}
