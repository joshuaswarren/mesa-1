/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_graphics_program.h"

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
   static const uint8_t fragment_speclm[AGX_IOTO_SPECLM_SIZE] = {
      0x77, 0x00, 0x2a, 0x43, 0x4b, 0x00, 0x00, 0x00,
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
   memcpy(staged.bytes, desc->fragment ? fragment_speclm : vertex_speclm,
          AGX_IOTO_SPECLM_SIZE);
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
       desc->uniform_range_count > AGX3_G17P_GRAPHICS_MAX_RANGES ||
       (desc->uniform_range_count && !desc->uniform_ranges))
      return false;

   struct agx_ioto_state_tail_output tail;
   if (!graphics_state_tail(desc, &tail))
      return false;

   struct agx3_g17p_graphics_program staged = {0};
   size_t written = 0;
   if (!desc->uniform_range_count) {
      const bool composed =
         desc->fragment && desc->writes_sample_mask
            ? agx_ioto_g17p_coverage_fragment_program_compose(
                 desc->program_va, NULL, 0, NULL, 0, &tail, staged.bytes,
                 sizeof(staged.bytes), &written)
            : desc->fragment
            ? agx_ioto_g17p_fragment_no_resource_program_compose(
                 desc->program_va, &tail, staged.bytes, sizeof(staged.bytes),
                 &written)
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
                    body_size - copy_offset, &tail, staged.bytes,
                    sizeof(staged.bytes), &written)
               : desc->fragment
               ? agx_ioto_g17p_fragment_resource_program_compose(
                    desc->program_va, body, copy_offset, body + copy_offset,
                    body_size - copy_offset, &tail, staged.bytes,
                    sizeof(staged.bytes), &written)
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
