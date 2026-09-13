/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_graphics_program.h"

#include "agx3_binder.h"
#include "agx3_pixel_core.h"
#include "asahi/lib/agx_ioto_state_core.h"
#include "util/macros.h"

#include <string.h>

static bool
graphics_state_tail(const struct agx3_g17p_graphics_program_desc *desc,
                    struct agx_ioto_state_tail_output *tail)
{
   if (!desc || !tail || desc->shader_binary_size == 0 || !desc->nr_gprs)
      return false;

   static const uint8_t vertex_speclm[AGX_IOTO_SPECLM_SIZE] = {
      0x77, 0x00, 0x2a, 0x41, 0x00, 0x00, 0x00, 0x00,
   };

   struct agx_ioto_state ldshdr = {0};
   ldshdr.f[0] = desc->shader_va;
   const uint64_t padded_shader_size =
      ((uint64_t)desc->shader_binary_size + 63) & ~UINT64_C(63);
   const uint64_t fetch_span = padded_shader_size >> 7;
   ldshdr.f[1] = fetch_span > 3 ? 3 : fetch_span;
   ldshdr.f[23] = 1;

   struct agx_ioto_state spectpr = {0};
   const bool exact_no_resource_fragment =
      desc->fragment && desc->uniform_range_count == 0 && !desc->scratch_size;
   const uint32_t compiler_d =
      desc->compiler_d_valid ? desc->compiler_d : desc->nr_gprs;
   if ((desc->compiler_d_valid || !exact_no_resource_fragment) &&
       !agx_ioto_g17p_compiler_spectpr_build(desc->scratch_size, compiler_d,
                                             &spectpr))
      return false;

   struct agx_ioto_state_tail_output staged = {
      .length = AGX_IOTO_STATE_TAIL_TYPE0_SIZE,
   };
   if (desc->fragment) {
      if (!agx_ioto_g17p_fragment_speclm_build(
             &desc->imageblock, staged.bytes, AGX_IOTO_SPECLM_SIZE))
         return false;
   } else {
      memcpy(staged.bytes, vertex_speclm, AGX_IOTO_SPECLM_SIZE);
   }
   if (!agx_ioto_state_pack_ldshdr(
          AGX_CHIP_G17P, &ldshdr, staged.bytes + AGX_IOTO_SPECLM_SIZE,
          AGX_IOTO_LDSHDR_SIZE) ||
       !agx_ioto_state_pack_spectpr(
          AGX_CHIP_G17P, &spectpr,
          staged.bytes + AGX_IOTO_SPECLM_SIZE + AGX_IOTO_LDSHDR_SIZE,
          AGX_IOTO_SPECTPR_SIZE))
      return false;

   *tail = staged;
   return true;
}

bool
agx3_g17p_graphics_program_bound(
   enum agx_chip chip, const struct agx3_ioto_uniform_range *ranges,
   size_t range_count, size_t *size)
{
   size_t body, chunks;
   if (!size || chip != AGX_CHIP_G17P ||
       range_count > AGX3_G17P_GRAPHICS_MAX_RANGES ||
       !agx3_ioto_uniform_program_body_bound(chip, ranges, range_count, &body,
                                             &chunks) ||
       chunks > AGX3_G17P_GRAPHICS_MAX_LOAD_CHUNKS ||
       body > AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE -
                 AGX_IOTO_G17P_NO_RESOURCE_PROGRAM_SIZE)
      return false;

   *size = body + AGX_IOTO_G17P_NO_RESOURCE_PROGRAM_SIZE;
   return true;
}

bool
agx3_g17p_graphics_program_generate(
   enum agx_chip chip, const struct agx3_g17p_graphics_program_desc *desc,
   struct agx3_g17p_graphics_program *program)
{
   if (!program || chip != AGX_CHIP_G17P || !desc ||
       (desc->writes_sample_mask && !desc->fragment) ||
       (!desc->fragment && desc->colour_partitions) ||
       desc->colour_partitions > AGX3_G17P_MAX_COLOUR_PARTITIONS ||
       desc->uniform_range_count > AGX3_G17P_GRAPHICS_MAX_RANGES ||
       (desc->uniform_range_count && !desc->uniform_ranges))
      return false;

   /* Zero remains the conservative no-colour/default-fragment spelling.  It
    * has the same loader release mask as one active colour partition. */
   const unsigned colour_partitions =
      desc->fragment ? MAX2(desc->colour_partitions, 1) : 0;

   struct agx_ioto_state_tail_output tail;
   if (!graphics_state_tail(desc, &tail))
      return false;

   struct agx3_g17p_graphics_program staged = {0};
   size_t written = 0;
   if (!desc->uniform_range_count) {
      const bool composed =
         desc->fragment && desc->writes_sample_mask
            ? agx_ioto_g17p_coverage_fragment_program_compose(
                 desc->program_va, NULL, 0, NULL, 0, &tail,
                 colour_partitions, staged.bytes, sizeof(staged.bytes),
                 &written)
            : desc->fragment
            ? agx_ioto_g17p_fragment_no_resource_program_compose(
                 desc->program_va, &tail, colour_partitions, staged.bytes,
                 sizeof(staged.bytes), &written)
            : agx_ioto_g17p_no_resource_program_compose(
                 desc->program_va, &tail, staged.bytes, sizeof(staged.bytes),
                 &written);
      const size_t expected =
         desc->fragment ? AGX_IOTO_G17P_FRAGMENT_NO_RESOURCE_PROGRAM_SIZE
                        : AGX_IOTO_G17P_NO_RESOURCE_PROGRAM_SIZE;
      if (!composed || written != expected)
         return false;
   } else {
      size_t bound;
      if (!agx3_g17p_graphics_program_bound(
             chip, desc->uniform_ranges, desc->uniform_range_count, &bound))
         return false;

      uint8_t body[AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE];
      struct agx_ioto_inflight_load loads[AGX3_G17P_GRAPHICS_MAX_LOAD_CHUNKS];
      struct agx_ioto_inflight_load sort[AGX3_G17P_GRAPHICS_MAX_LOAD_CHUNKS];
      struct agx_ioto_load_cache_entry cache[AGX3_G17P_GRAPHICS_MAX_LOAD_CHUNKS];
      const struct agx3_ioto_uniform_program_desc load_desc = {
         .program_va = desc->program_va,
         .ranges = desc->uniform_ranges,
         .range_count = desc->uniform_range_count,
      };
      const struct agx3_ioto_uniform_program_workspace workspace = {
         .body = body,
         .body_capacity = sizeof(body),
         .loads = loads,
         .load_capacity = ARRAY_SIZE(loads),
         .sort = sort,
         .sort_capacity = ARRAY_SIZE(sort),
         .cache = cache,
         .cache_capacity = ARRAY_SIZE(cache),
      };
      size_t body_size, copy_offset;
      if (!agx3_ioto_uniform_program_body_generate(
             chip, &load_desc, workspace, &body_size, &copy_offset) ||
          copy_offset > body_size ||
          !(desc->fragment && desc->writes_sample_mask
               ? agx_ioto_g17p_coverage_fragment_program_compose(
                    desc->program_va, body, copy_offset, body + copy_offset,
                    body_size - copy_offset, &tail, colour_partitions,
                    staged.bytes, sizeof(staged.bytes), &written)
               : desc->fragment
               ? agx_ioto_g17p_fragment_resource_program_compose(
                    desc->program_va, body, copy_offset, body + copy_offset,
                    body_size - copy_offset, &tail, colour_partitions,
                    staged.bytes, sizeof(staged.bytes), &written)
               : agx_ioto_g17p_resource_program_compose(
                    desc->program_va, body, copy_offset, body + copy_offset,
                    body_size - copy_offset, &tail, staged.bytes,
                    sizeof(staged.bytes), &written)))
         return false;
   }

   if (written > UINT16_MAX)
      return false;
   staged.length = (uint16_t)written;
   *program = staged;
   return true;
}

bool
agx3_g17p_fragment_blend_constant_base(enum agx_chip chip,
                                        unsigned occupied_halfs,
                                        uint16_t *base)
{
   if (chip != AGX_CHIP_G17P || !base)
      return false;

   const unsigned start = ALIGN_POT(MAX2(occupied_halfs, 8), 2);
   if (start > 512 - 8)
      return false;

   *base = (uint16_t)start;
   return true;
}

bool
agx3_g17p_rt0_bg_resource_spec(enum agx_chip chip,
                               enum agx3_g17p_rt0_bg_op op,
                               uint64_t *resource_spec)
{
   if (chip != AGX_CHIP_G17P || !resource_spec)
      return false;

   uint64_t pass_mask;
   switch (op) {
   case AGX3_G17P_RT0_BG_CLEAR:
   case AGX3_G17P_RT0_BG_RELOAD:
      pass_mask = (UINT64_C(1) << 17) - 1;
      break;
   case AGX3_G17P_RT0_BG_NONE:
      pass_mask = 0;
      break;
   default:
      return false;
   }

   const uint64_t resources =
      AGX_IOTO_G17P_BG_RESOURCE_SPEC & ((UINT64_C(1) << 47) - 1);
   *resource_spec = resources | (pass_mask << 47);
   return true;
}

bool
agx3_g17p_rt0_bg_shader_resources_match(enum agx3_g17p_rt0_bg_op op,
                                        unsigned push_count, bool uses_txf)
{
   switch (op) {
   case AGX3_G17P_RT0_BG_CLEAR:
      return (push_count == AGX3_G17P_RT0_CLEAR_PUSH_HALFS ||
              push_count == AGX3_G17P_WBOIT_MRT_CLEAR_PUSH_HALFS ||
              push_count == AGX3_G17P_R11G11B10_MRT_CLEAR_PUSH_HALFS ||
              push_count == AGX3_G17P_RGBA8_MRT_CLEAR_PUSH_HALFS ||
              push_count == AGX3_G17P_TRIPLE_CLEAR_PUSH_HALFS ||
              push_count == AGX3_G17P_QUAD_CLEAR_PUSH_HALFS) &&
             !uses_txf;
   case AGX3_G17P_RT0_BG_RELOAD:
      return push_count == 0 && uses_txf;
   default:
      return false;
   }
}

bool
agx3_g17p_rt0_bg_program_generate(
   enum agx_chip chip, const struct agx3_g17p_rt0_bg_program_desc *desc,
   struct agx3_g17p_rt0_bg_program *program)
{
   if (!program || chip != AGX_CHIP_G17P || !desc)
      return false;

   struct agx3_ioto_uniform_range ranges[2];
   size_t range_count;
   switch (desc->op) {
   case AGX3_G17P_RT0_BG_CLEAR: {
      const uint16_t clear_push_halfs =
         desc->clear_push_halfs ? desc->clear_push_halfs
                                : AGX3_G17P_RT0_CLEAR_PUSH_HALFS;
      if (!desc->clear_value_va || desc->texture_pointer_va ||
          desc->sampler_pointer_va ||
          (clear_push_halfs != AGX3_G17P_RT0_CLEAR_PUSH_HALFS &&
           clear_push_halfs != AGX3_G17P_WBOIT_MRT_CLEAR_PUSH_HALFS &&
           clear_push_halfs != AGX3_G17P_R11G11B10_MRT_CLEAR_PUSH_HALFS &&
           clear_push_halfs != AGX3_G17P_RGBA8_MRT_CLEAR_PUSH_HALFS &&
           clear_push_halfs != AGX3_G17P_TRIPLE_CLEAR_PUSH_HALFS &&
           clear_push_halfs != AGX3_G17P_QUAD_CLEAR_PUSH_HALFS))
         return false;
      ranges[0] = (struct agx3_ioto_uniform_range){
         .start_halfs = 4,
         .size_halfs = clear_push_halfs - 4,
         .address = desc->clear_value_va,
      };
      range_count = 1;
      break;
   }
   case AGX3_G17P_RT0_BG_RELOAD:
      if (desc->clear_value_va || !desc->texture_pointer_va ||
          !desc->sampler_pointer_va)
         return false;
      ranges[0] = (struct agx3_ioto_uniform_range){
         .start_halfs = 0,
         .size_halfs = 4,
         .address = desc->texture_pointer_va,
      };
      ranges[1] = (struct agx3_ioto_uniform_range){
         .start_halfs = 4,
         .size_halfs = 4,
         .address = desc->sampler_pointer_va,
      };
      range_count = 2;
      break;
   default:
      return false;
   }

   const struct agx3_g17p_graphics_program_desc shader = {
      .program_va = desc->program_va,
      .shader_va = desc->shader_va,
      .shader_binary_size = desc->shader_binary_size,
      .scratch_size = desc->scratch_size,
      .nr_gprs = desc->nr_gprs,
      .compiler_d_valid = desc->staging_known,
      .compiler_d = (uint32_t)desc->staging_halfs * 2,
      .fragment = true,
      .imageblock = desc->imageblock,
   };
   struct agx_ioto_state_tail_output tail;
   if (!graphics_state_tail(&shader, &tail))
      return false;

   uint8_t body[512];
   struct agx_ioto_inflight_load loads[64];
   struct agx_ioto_inflight_load sort[64];
   struct agx_ioto_load_cache_entry cache[64];
   const struct agx3_ioto_uniform_program_desc load_desc = {
      .program_va = desc->program_va,
      .ranges = ranges,
      .range_count = range_count,
   };
   const struct agx3_ioto_uniform_program_workspace workspace = {
      .body = body,
      .body_capacity = sizeof(body),
      .loads = loads,
      .load_capacity = ARRAY_SIZE(loads),
      .sort = sort,
      .sort_capacity = ARRAY_SIZE(sort),
      .cache = cache,
      .cache_capacity = ARRAY_SIZE(cache),
   };

   size_t body_size, copy_offset;
   if (!agx3_ioto_uniform_program_body_generate(chip, &load_desc, workspace,
                                                &body_size, &copy_offset) ||
       copy_offset > body_size)
      return false;

   struct agx3_g17p_rt0_bg_program staged = {0};
   size_t written = 0;
   if (!agx_ioto_g17p_background_resource_program_compose(
          desc->program_va, body, copy_offset, body + copy_offset,
          body_size - copy_offset, &tail, staged.bytes, sizeof(staged.bytes),
          &written) ||
       written > UINT16_MAX ||
       (desc->op == AGX3_G17P_RT0_BG_CLEAR &&
        written >
           (desc->clear_push_halfs >= AGX3_G17P_TRIPLE_CLEAR_PUSH_HALFS
               ? AGX_IOTO_G17P_QUAD_CLEAR_PROGRAM_MAX_SIZE
            : desc->clear_push_halfs == AGX3_G17P_RGBA8_MRT_CLEAR_PUSH_HALFS
               ? AGX_IOTO_G17P_RGBA8_MRT_CLEAR_PROGRAM_SIZE
            : desc->clear_push_halfs == AGX3_G17P_R11G11B10_MRT_CLEAR_PUSH_HALFS
               ? AGX_IOTO_G17P_RGBA8_MRT_CLEAR_PROGRAM_SIZE
            : desc->clear_push_halfs == AGX3_G17P_WBOIT_MRT_CLEAR_PUSH_HALFS
               ? AGX_IOTO_G17P_WBOIT_MRT_CLEAR_PROGRAM_SIZE
               : AGX_IOTO_G17P_RT0_CLEAR_PROGRAM_SIZE)))
      return false;

   staged.length = (uint16_t)written;
   *program = staged;
   return true;
}

bool
agx3_g17p_rt0_eot_program_generate(
   enum agx_chip chip, const struct agx3_g17p_rt0_eot_program_desc *desc,
   struct agx3_g17p_rt0_eot_program *program)
{
   const bool rt0 = desc &&
                    (desc->shader_binary_size == 68 ||
                     desc->shader_binary_size == 84) &&
                    desc->nr_gprs == 8 && desc->staging_halfs == 8 &&
                    (desc->attachment_count == 0 ||
                     desc->attachment_count == 1);
   const bool rgba8_mrt = desc && desc->shader_binary_size == 120 &&
                          desc->nr_gprs == 8 &&
                          desc->staging_halfs == 16 &&
                          desc->attachment_count == 2 &&
                          desc->imageblock_offset_B == 0 &&
                          desc->imageblock_offset1_B == 4;
   const bool wboit_mrt = desc && desc->shader_binary_size == 238 &&
                          desc->nr_gprs == 8 &&
                          desc->staging_halfs == 16 &&
                          desc->attachment_count == 2 &&
                          desc->imageblock_offset_B == 0 &&
                          desc->imageblock_offset1_B == 8;
   const bool dual_mrt = rgba8_mrt || wboit_mrt;
   const unsigned count = desc ? desc->attachment_count : 0;
   const bool raw32_mrt =
      desc && count >= 2 && count <= 4 &&
      desc->shader_binary_size == 32 + 44 * count && desc->nr_gprs == 8 &&
      desc->staging_halfs == 8 * count &&
      desc->imageblock_offset_B == 0 &&
      (desc->imageblock_offset1_B == 4 || desc->imageblock_offset1_B == 16) &&
      desc->imageblock_offset2_B == (count > 2 ? 2 * desc->imageblock_offset1_B : 0) &&
      desc->imageblock_offset3_B == (count > 3 ? 3 * desc->imageblock_offset1_B : 0) &&
      desc->imageblock.sample_size_B ==
         ALIGN_POT(count * desc->imageblock_offset1_B, 8) &&
      desc->imageblock.nr_samples == 1;
   if (!program || chip != AGX_CHIP_G17P || !desc ||
       (!rt0 && !dual_mrt && !raw32_mrt) ||
       !desc->pbe_argument_va ||
       desc->pbe_argument_va > UINT64_C(0xffffffffffff) -
          (raw32_mrt ? 32 * count - 1 : dual_mrt ? 63 : 31) ||
       (desc->pbe_argument_va & 31))
      return false;

   const struct agx3_g17p_graphics_program_desc shader = {
      .program_va = desc->program_va,
      .shader_va = desc->shader_va,
      .shader_binary_size = desc->shader_binary_size,
      .scratch_size = desc->scratch_size,
      .nr_gprs = desc->nr_gprs,
      .compiler_d_valid = true,
      .compiler_d = desc->staging_halfs * 2,
      .fragment = true,
      .imageblock = desc->imageblock,
   };
   struct agx_ioto_state_tail_output tail;
   if (!graphics_state_tail(&shader, &tail))
      return false;

   uint8_t body[6 * AGX_IOTO_LOAD_LDIMM_MAX_SIZE];
   size_t body_size = 0;
   for (unsigned i = 0; i < 2; ++i) {
      uint8_t bytes[AGX_IOTO_LOAD_LDIMM_MAX_SIZE], length;
      if (!agx_ioto_load_pack_ldimm(
             chip, 4 + 2 * i,
             (uint32_t)(desc->pbe_argument_va >> (32 * i)), bytes, &length))
         return false;
      memcpy(body + body_size, bytes, length);
      body_size += length;
   }

   uint8_t offset_length;
   if (!agx_ioto_load_pack_ldimm(chip, 8, desc->imageblock_offset_B,
                                 body + body_size, &offset_length))
      return false;
   body_size += offset_length;
   if (dual_mrt || raw32_mrt) {
      if (!agx_ioto_load_pack_ldimm(chip, 10, desc->imageblock_offset1_B,
                                    body + body_size, &offset_length))
         return false;
      body_size += offset_length;
   }

   if (raw32_mrt && count > 2) {
      const unsigned offsets[] = {
         desc->imageblock_offset2_B, desc->imageblock_offset3_B};
      for (unsigned i = 0; i < count - 2; ++i) {
         if (!agx_ioto_load_pack_ldimm(chip, 12 + 2 * i, offsets[i],
                                      body + body_size, &offset_length))
            return false;
         body_size += offset_length;
      }
   }

   uint8_t copies[36];
   const struct agx3_field_kv baseline[] = {{9, 3}, {10, 2}, {23, 2}, {24, 2}};
   /* Native offset uniforms are u4/u8/u12/u16, not contiguous dwords.
    * Padding positions are unused but initialized deterministically. */
   const unsigned copy_count = raw32_mrt ? 2 * count + 1 : dual_mrt ? 5 : 3;
   const unsigned copy_sources[] = {4, 6, 8, 8, 10, 8, 12, 8, 14};
   for (unsigned i = 0; i < copy_count; ++i) {
      const struct agx3_field_kv operands[] = {
         {3, 2u * i},
         {16, copy_sources[i]},
      };
      uint8_t bytes[AGX3_MAX_WORDS * 2], length = 0;
      if (!agx3_bind_compose(AGX_ISA_ENCODING_AGX3B, "copy", 0, baseline,
                             ARRAY_SIZE(baseline), operands,
                             ARRAY_SIZE(operands), bytes, &length) ||
          length != 4)
         return false;
      memcpy(copies + 4 * i, bytes, length);
   }

   struct agx3_g17p_rt0_eot_program staged = {0};
   size_t written = 0;
   if (!agx_ioto_g17p_resource_program_compose(
          desc->program_va, body, body_size, copies, 4 * copy_count, &tail,
          staged.bytes, sizeof(staged.bytes), &written) ||
       written > sizeof(staged.bytes))
      return false;

   staged.length = (uint16_t)written;
   *program = staged;
   return true;
}
