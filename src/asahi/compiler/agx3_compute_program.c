/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_compute_program.h"

#include "asahi/lib/agx_ioto_program_core.h"
#include "util/macros.h"

static bool
agx3_g17p_compute_state_tail(
   const struct agx3_g17p_compute_program_desc *desc,
   struct agx_ioto_state_tail_output *tail)
{
   /* SPECTPR carries the thread-private spill frame in quads. The returned
    * device atomic's 16-byte TPS word is one such frame; register spills
    * publish theirs the same way, as Apple's kernel metadata does. */
   if (!desc || !tail || !desc->shader_binary_size || !desc->nr_gprs ||
       desc->scratch_size > AGX3_G17P_MAX_SPILL_B ||
       (desc->scratch_size & 3))
      return false;

   struct agx_ioto_state_tail_output staged;
   const bool tail_ok = desc->threadgroup_memory_size_B
      ? agx_ioto_g17p_runtime_compute_tail_build_shared(
           desc->shader_va, desc->ldshdr_f1,
           desc->threadgroup_memory_size_B, &staged)
      : agx_ioto_g17p_runtime_compute_tail_build(desc->shader_va,
                                                 desc->ldshdr_f1, &staged);
   if (!tail_ok)
      return false;

   if (!desc->compiler_d_valid ||
       !agx_ioto_g17p_compiler_spectpr_pack(
          desc->scratch_size, desc->compiler_d,
          staged.bytes + AGX_IOTO_SPECLM_SIZE + AGX_IOTO_LDSHDR_SIZE,
          AGX_IOTO_SPECTPR_SIZE))
      return false;

   *tail = staged;
   return true;
}

bool
agx3_g17p_compute_program_generate(
   enum agx_chip chip, const struct agx3_g17p_compute_program_desc *desc,
   struct agx3_g17p_compute_program *program)
{
   if (!program || chip != AGX_CHIP_G17P || !desc ||
       (desc->uniform_range_count && !desc->uniform_ranges))
      return false;

   struct agx_ioto_state_tail_output tail;
   if (!agx3_g17p_compute_state_tail(desc, &tail))
      return false;

   struct agx3_g17p_compute_program staged = {0};
   size_t written = 0;
   if (!desc->uniform_range_count) {
      if (!agx_ioto_g17p_no_resource_program_compose_trusted(
             desc->program_va, &tail, staged.bytes, sizeof(staged.bytes),
             &written))
         return false;
   } else {
      uint8_t body[AGX3_G17P_COMPUTE_PROGRAM_MAX_SIZE];
      struct agx_ioto_inflight_load loads[AGX3_G17P_COMPUTE_MAX_LOAD_CHUNKS];
      struct agx_ioto_inflight_load sort[AGX3_G17P_COMPUTE_MAX_LOAD_CHUNKS];
      struct agx_ioto_load_cache_entry cache[AGX3_G17P_COMPUTE_MAX_LOAD_CHUNKS];
      size_t body_bound, chunk_count;
      if (!agx3_ioto_uniform_program_body_bound(
             chip, desc->uniform_ranges, desc->uniform_range_count,
             &body_bound, &chunk_count) ||
          body_bound > sizeof(body) || chunk_count > ARRAY_SIZE(loads))
         return false;

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
          !agx_ioto_g17p_resource_program_compose_trusted(
             desc->program_va, body, copy_offset, body + copy_offset,
             body_size - copy_offset, &tail, staged.bytes,
             sizeof(staged.bytes), &written))
         return false;
   }

   if (written > UINT16_MAX)
      return false;

   const struct agx3_g17p_native_compute_program_desc launch_desc = {
      .program_va = desc->program_va,
      .shader_va = desc->shader_va,
      .launch_sources = desc->launch_sources,
   };
   if (!agx3_g17p_native_compute_program_bind(&launch_desc, &staged.launch))
      return false;

   staged.length = written;
   *program = staged;
   return true;
}
