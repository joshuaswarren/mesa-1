/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_graphics_program.h"

#include "agx3_binder.h"
#include "agx3_copy_core.h"
#include "agx3_ioto_load_packer.h"
#include "agx3_pixel_core.h"
#include "asahi/lib/agx_ioto_preshader_core.h"
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

   const uint64_t padded_shader_size =
      ((uint64_t)desc->shader_binary_size + 63) & ~UINT64_C(63);
   const uint64_t fetch_span = padded_shader_size >> 7;
   const uint32_t ldshdr_fetch_span = fetch_span > 3 ? 3 : fetch_span;

   const bool exact_no_resource_fragment =
      desc->fragment && desc->uniform_range_count == 0 && !desc->scratch_size;
   const uint32_t compiler_d =
      desc->compiler_d_valid ? desc->compiler_d : desc->nr_gprs;
   const bool compiler_spectpr =
      desc->compiler_d_valid || !exact_no_resource_fragment;

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
   uint8_t *spectpr_out =
      staged.bytes + AGX_IOTO_SPECLM_SIZE + AGX_IOTO_LDSHDR_SIZE;
   static const struct agx_ioto_state zero_spectpr;
   if (!agx_ioto_g17p_compiler_ldshdr_pack(
          desc->shader_va, ldshdr_fetch_span,
          staged.bytes + AGX_IOTO_SPECLM_SIZE, AGX_IOTO_LDSHDR_SIZE) ||
       (compiler_spectpr
           ? !agx_ioto_g17p_compiler_spectpr_pack(
                desc->scratch_size, compiler_d, spectpr_out,
                AGX_IOTO_SPECTPR_SIZE)
           : !agx_ioto_state_pack_spectpr(
                AGX_CHIP_G17P, &zero_spectpr, spectpr_out,
                AGX_IOTO_SPECTPR_SIZE)))
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

static bool graphics_root_body_generate(
   const struct agx3_g17p_graphics_program_desc *desc, void *output,
   size_t capacity, size_t *written, bool root_is_uniform);

static bool
graphics_input_body_bound(const struct agx3_g17p_graphics_program_desc *desc,
                           size_t *bytes, size_t *chunks)
{
   if (!desc || (desc->cprog_inputs && desc->cprog_inputs->count &&
                 !desc->constant_program.code_size))
      return false;
   const struct agx3_ioto_uniform_program_desc loads = {
      .ranges = desc->uniform_ranges, .range_count = desc->uniform_range_count,
      .cprog_inputs = desc->cprog_inputs,
   };
   return agx3_ioto_uniform_program_desc_body_bound(
      AGX_CHIP_G17P, &loads, bytes, chunks);
}

static bool
graphics_measure_cprog_body(const struct agx3_g17p_graphics_program_desc *desc,
                             size_t *size)
{
   /* The range-only worst case assumes a fresh address pair and round for
    * every chunk. With up to 512 input halves that loose bound can exceed
    * capacity even though the real batched body fits. Measure the native
    * schedule when building this persistent setup, as for root loaders. */
   uint8_t body[AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE];
   struct agx_ioto_inflight_load loads[AGX3_G17P_GRAPHICS_MAX_LOAD_CHUNKS];
   struct agx_ioto_inflight_load sort[AGX3_G17P_GRAPHICS_MAX_LOAD_CHUNKS];
   struct agx_ioto_load_cache_entry cache[AGX3_G17P_GRAPHICS_MAX_LOAD_CHUNKS];
   const struct agx3_ioto_uniform_program_desc load_desc = {
      .ranges = desc->uniform_ranges, .range_count = desc->uniform_range_count,
      .cprog_inputs = desc->cprog_inputs,
   };
   const struct agx3_ioto_uniform_program_workspace workspace = {
      .body = body, .body_capacity = sizeof(body),
      .loads = loads, .load_capacity = ARRAY_SIZE(loads),
      .sort = sort, .sort_capacity = ARRAY_SIZE(sort),
      .cache = cache, .cache_capacity = ARRAY_SIZE(cache),
   };
   size_t copies;
   return agx3_ioto_uniform_program_body_generate(
      AGX_CHIP_G17P, &load_desc, workspace, size, &copies);
}

bool
agx3_g17p_graphics_program_desc_bound(
   enum agx_chip chip, const struct agx3_g17p_graphics_program_desc *desc,
   size_t *size)
{
   size_t base, input_body, chunks;
   if (!desc || !size || chip != AGX_CHIP_G17P ||
       desc->uniform_range_count > AGX3_G17P_GRAPHICS_MAX_RANGES ||
       (desc->uniform_range_count && !desc->uniform_ranges) ||
       (desc->constant_program.code_size & 1) ||
       !graphics_input_body_bound(desc, &input_body, &chunks) ||
       chunks > AGX3_G17P_GRAPHICS_MAX_LOAD_CHUNKS)
      return false;

   if (desc->preloaded_root) {
      /* A root range includes a dependent pointer load absent from the
       * direct-address bound. Measure the shared encoder's exact schedule
       * when constructing this persistent program, not per recorded draw. */
      uint8_t body[AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE];
      if (!graphics_root_body_generate(desc, body, sizeof(body), &base, false) ||
          base > AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE -
                    AGX_IOTO_G17P_NO_RESOURCE_PROGRAM_SIZE)
         return false;
      base += AGX_IOTO_G17P_NO_RESOURCE_PROGRAM_SIZE;
   } else {
      if (desc->cprog_inputs && desc->cprog_inputs->count &&
          input_body > AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE -
                          AGX_IOTO_G17P_NO_RESOURCE_PROGRAM_SIZE &&
          !graphics_measure_cprog_body(desc, &input_body))
         return false;
      if (input_body > AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE -
                          AGX_IOTO_G17P_NO_RESOURCE_PROGRAM_SIZE)
         return false;
      base = input_body + AGX_IOTO_G17P_NO_RESOURCE_PROGRAM_SIZE;
   }

   const struct agx_ioto_constant_program *cp = &desc->constant_program;
   size_t suffix = AGX_IOTO_PRESHADER_ABSENT_SIZE;
   if (cp->code_size) {
      if (cp->force_external ||
          cp->code_size > AGX_IOTO_PRESHADER_INLINE_LIMIT) {
         suffix = AGX_IOTO_PRESHADER_BRANCH_SIZE +
                  (cp->request_prelude && !cp->suppress_prelude
                      ? AGX_IOTO_PRESHADER_PRELUDE_SIZE : 0);
      } else {
         if (!cp->code)
            return false;
         suffix = MAX2(cp->code_size, AGX_IOTO_PRESHADER_ABSENT_SIZE);
      }
   }

   base -= AGX_IOTO_PRESHADER_ABSENT_SIZE;
   if (suffix > AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE - base)
      return false;
   *size = base + suffix;
   return true;
}

static bool
graphics_constant_suffix_emit(
   enum agx_chip chip, const struct agx3_g17p_graphics_program_desc *desc,
   size_t offset, struct agx_ioto_preshader_output *suffix)
{
   const uint64_t limit = UINT64_C(1) << 48;
   return desc->program_va && desc->program_va < limit &&
          offset < limit - desc->program_va &&
          agx_ioto_constant_program_emit(
             chip, &desc->constant_program, desc->program_va + offset,
             suffix);
}

/* All qualified resource composers end in the same absent-program END.
 * Replace that suffix only: profile/PIXREL ordering and the load schedule
 * are independent of constant-program selection. This writes unpublished
 * storage; the public transactional generator stages the whole result. */
static bool
graphics_constant_suffix_replace(
   enum agx_chip chip, const struct agx3_g17p_graphics_program_desc *desc,
   void *output, size_t capacity, size_t *written)
{
   if (!desc->constant_program.code_size)
      return true;

   static const uint8_t absent[] = {0x0e, 0, 0, 0};
   if (*written < sizeof(absent) || *written > capacity)
      return false;
   const size_t offset = *written - sizeof(absent);
   if (memcmp((uint8_t *)output + offset, absent, sizeof(absent)))
      return false;

   struct agx_ioto_preshader_output suffix;
   if (!graphics_constant_suffix_emit(chip, desc, offset, &suffix) ||
       suffix.length > capacity - offset)
      return false;

   memcpy((uint8_t *)output + offset, suffix.bytes, suffix.length);
   *written = offset + suffix.length;
   return true;
}

static bool
graphics_root_body_generate(
   const struct agx3_g17p_graphics_program_desc *desc, void *output,
   size_t capacity, size_t *written, bool root_is_uniform)
{
   size_t bound, chunks;
   if (!desc || !output || !written || !desc->preloaded_root ||
       !graphics_input_body_bound(desc, &bound, &chunks) ||
       chunks > AGX3_G17P_GRAPHICS_MAX_LOAD_CHUNKS)
      return false;

   size_t offset = 0;
   for (size_t i = 0; i < desc->uniform_range_count; ++i) {
      const struct agx3_ioto_uniform_range *range = &desc->uniform_ranges[i];
      if (range->address > INT16_MAX)
         return false;

      size_t range_size = 0;
      const uint16_t root_half =
         root_is_uniform ? AGX3_G17P_GRAPHICS_EXTERNAL_ROOT_UNIFORM_HALF
                         : AGX3_G17P_GRAPHICS_ROOT_BASE_HALF;
      const bool loaded = agx3_ioto_load_indirect_uniform_range_cprog(
         AGX_CHIP_G17P, root_half, root_is_uniform, (int16_t)range->address,
         range->start_halfs, range->size_halfs, desc->cprog_inputs,
         (uint8_t *)output + offset, capacity - offset, &range_size);
      if (!loaded)
         return false;
      offset += range_size;
   }

   *written = offset;
   return true;
}

static bool
graphics_external_root_uniform_free(
   const struct agx3_g17p_graphics_program_desc *desc)
{
   const unsigned root_start =
      AGX3_G17P_GRAPHICS_EXTERNAL_ROOT_UNIFORM_HALF;
   const unsigned root_end = root_start + 4;

   for (size_t i = 0; i < desc->uniform_range_count; ++i) {
      const unsigned start = desc->uniform_ranges[i].start_halfs;
      const unsigned end = start + desc->uniform_ranges[i].size_halfs;
      if (start < root_end && end > root_start)
         return false;
   }

   return true;
}

bool
agx3_g17p_graphics_external_setup_suffix_generate(
   enum agx_chip chip, const struct agx3_g17p_graphics_program_desc *desc,
   void *output, size_t capacity, size_t *written)
{
   if (chip != AGX_CHIP_G17P || !desc || !output || !written ||
       !desc->preloaded_root ||
       (desc->writes_sample_mask && !desc->fragment) ||
       (!desc->fragment && desc->colour_classes) ||
       desc->uniform_range_count > AGX3_G17P_GRAPHICS_MAX_RANGES ||
       (desc->uniform_range_count && !desc->uniform_ranges) ||
       !graphics_external_root_uniform_free(desc))
      return false;

   uint8_t body[AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE];
   size_t body_size = 0;
   if (!graphics_root_body_generate(desc, body, sizeof(body), &body_size,
                                    true))
      return false;

   size_t size;
   if (!agx_ioto_g17p_external_setup_suffix_compose(
      body, body_size, desc->fragment,
      desc->fragment && desc->writes_sample_mask, desc->colour_classes,
      output, capacity, &size) ||
       !graphics_constant_suffix_replace(chip, desc, output, capacity, &size))
      return false;

   *written = size;
   return true;
}

bool
agx3_g17p_graphics_external_launcher_generate(
   enum agx_chip chip, const struct agx3_g17p_graphics_program_desc *desc,
   uint64_t program_va, uint64_t root_va, uint64_t setup_va, void *output,
   size_t capacity, size_t *written)
{
   if (chip != AGX_CHIP_G17P || !desc || !output || !written ||
       !desc->preloaded_root || !graphics_external_root_uniform_free(desc))
      return false;

   struct agx3_g17p_graphics_root_prefix prefix;
   struct agx_ioto_state_tail_output tail;
   if (!agx3_g17p_graphics_root_prefix_generate(chip, root_va, &prefix) ||
       !graphics_state_tail(desc, &tail))
      return false;

   uint8_t launch_prefix[AGX3_G17P_GRAPHICS_ROOT_PREFIX_MAX_SIZE +
                         2 * AGX3_COPY_MAX_LENGTH];
   memcpy(launch_prefix, prefix.bytes, prefix.length);
   size_t prefix_length = prefix.length;
   for (unsigned dword = 0; dword < 2; ++dword) {
      uint8_t copy[AGX3_COPY_MAX_LENGTH];
      uint8_t copy_length = 0;
      if (!agx3_copy_encode_reg_to_uniform(
             AGX_ISA_ENCODING_AGX3B,
             AGX3_G17P_GRAPHICS_EXTERNAL_ROOT_UNIFORM_HALF + 2 * dword,
             AGX3_G17P_GRAPHICS_ROOT_BASE_HALF + 2 * dword, false, copy,
             &copy_length) ||
          copy_length > sizeof(launch_prefix) - prefix_length)
         return false;
      memcpy(launch_prefix + prefix_length, copy, copy_length);
      prefix_length += copy_length;
   }

   return agx_ioto_g17p_external_setup_launcher_compose_trusted(
      program_va, launch_prefix, prefix_length, &tail, setup_va, output,
      capacity, written);
}

bool
agx3_g17p_graphics_external_launcher_template_generate(
   enum agx_chip chip, const struct agx3_g17p_graphics_program_desc *desc,
   uint64_t setup_va,
   struct agx3_g17p_graphics_external_launcher_template *launcher)
{
   if (!launcher)
      return false;

   /* The command-pool root heap occupies this same compact high-dword class
    * in production.  If that ever changes, relocation refuses and HK keeps
    * the fully qualified inline fallback. */
   const uint64_t template_root_va = UINT64_C(0x40a5a5a5a5);
   struct agx3_g17p_graphics_root_prefix root;
   struct agx3_g17p_graphics_external_launcher_template staged = {0};
   size_t written = 0;
   if (!agx3_g17p_graphics_root_prefix_generate(chip, template_root_va,
                                                 &root) ||
       !agx3_g17p_graphics_external_launcher_generate(
          chip, desc, AGX_IOTO_PROGRAM_ALIGNMENT, template_root_va, setup_va,
          staged.bytes, sizeof(staged.bytes), &written) ||
       written > UINT8_MAX || written < AGX_IOTO_PRESHADER_BRANCH_SIZE)
      return false;

   staged.length = written;
   staged.root_prefix_length = root.length;
   staged.branch_offset = written - AGX_IOTO_PRESHADER_BRANCH_SIZE;
   *launcher = staged;
   return true;
}

bool
agx3_g17p_graphics_external_launcher_template_relocate(
   enum agx_chip chip,
   const struct agx3_g17p_graphics_external_launcher_template *launcher,
   uint64_t program_va, uint64_t root_va, uint64_t setup_va, void *output,
   size_t capacity, size_t *written)
{
   if (chip != AGX_CHIP_G17P || !launcher || !output || !written ||
       launcher->length > sizeof(launcher->bytes) ||
       launcher->length > capacity ||
       launcher->branch_offset + AGX_IOTO_PRESHADER_BRANCH_SIZE !=
          launcher->length ||
       program_va == 0 || program_va >= (UINT64_C(1) << 48) ||
       (program_va & (AGX_IOTO_PROGRAM_ALIGNMENT - 1)) != 0 ||
       setup_va == 0 || setup_va >= (UINT64_C(1) << 48) ||
       (setup_va & 1) != 0)
      return false;

   /* The command-pool heap's ordinary G17P address class packs as one full
    * low-dword LDIMM followed by one compact high-dword LDIMM.  Deposit those
    * exact fields directly instead of running two generic packers per draw.
    * Any uncommon address class keeps the already-qualified inline fallback. */
   const uint32_t root_low = (uint32_t)root_va;
   const uint32_t root_high = (uint32_t)(root_va >> 32);
   if (!root_va || root_va >= (UINT64_C(1) << 48) || root_low < 128 ||
       root_high >= 128 || launcher->root_prefix_length != 10)
      return false;

   const uint64_t branch_va = program_va + launcher->branch_offset;
   if (branch_va < program_va)
      return false;
   const int64_t displacement = (int64_t)setup_va - (int64_t)branch_va;
   if (displacement < -(INT64_C(1) << 47) ||
       displacement >= (INT64_C(1) << 47) || (displacement & 1) != 0)
      return false;

   /* Every refusal is above, so the output can be written in place: the
    * relocation stays transactional without a staging copy (it runs once
    * per draw). */
   uint8_t *staged = output;
   memcpy(staged, launcher->bytes, launcher->length);

   const uint16_t root_w0 =
      UINT16_C(0x800c) | (uint16_t)((root_low & 0x7f) << 8);
   const uint16_t root_w1 =
      UINT16_C(0x0002) | (uint16_t)(((root_low >> 25) & 0x7f) << 9);
   const uint16_t root_w2 = (uint16_t)(((root_low >> 7) & 0xf) << 1) |
                            (uint16_t)(((root_low >> 11) & 3) << 10);
   const uint16_t root_w3 = (uint16_t)((root_low >> 13) & 0xf) |
                            (uint16_t)(((root_low >> 17) & 0xff) << 4);
   staged[0] = root_w0;
   staged[1] = root_w0 >> 8;
   staged[2] = root_w1;
   staged[3] = root_w1 >> 8;
   staged[4] = root_w2;
   staged[5] = root_w2 >> 8;
   staged[6] = root_w3;
   staged[7] = root_w3 >> 8;
   staged[8] = 0x1c;
   staged[9] = root_high;

   /* G17P's complete signed 48-bit byte displacement is contiguous at branch
    * bits 24..71.  The template owns the invariant opcode/LUT bytes. */
   const uint64_t encoded_displacement =
      (uint64_t)displacement & ((UINT64_C(1) << 48) - 1);
   for (unsigned byte = 0; byte < 6; ++byte)
      staged[launcher->branch_offset + 3 + byte] =
         encoded_displacement >> (8 * byte);

   *written = launcher->length;
   return true;
}

static bool
graphics_program_write_unpublished(
   enum agx_chip chip, const struct agx3_g17p_graphics_program_desc *desc,
   void *output, size_t capacity, size_t *written_out,
   struct agx_ioto_load_relocation *relocations,
   size_t relocation_capacity, size_t *relocation_count,
   size_t *constant_suffix_offset)
{
   size_t input_body, input_chunks;
   if (!output || !written_out || chip != AGX_CHIP_G17P || !desc ||
       (desc->writes_sample_mask && !desc->fragment) ||
       (!desc->fragment && desc->colour_classes) ||
       (desc->colour_classes &
        ~BITFIELD_MASK(AGX3_G17P_MAX_COLOUR_PARTITIONS)) ||
       desc->uniform_range_count > AGX3_G17P_GRAPHICS_MAX_RANGES ||
       (desc->uniform_range_count && !desc->uniform_ranges) ||
       (relocation_capacity && !relocations) ||
       (relocations && !relocation_count) ||
       !graphics_input_body_bound(desc, &input_body, &input_chunks) ||
       input_chunks > AGX3_G17P_GRAPHICS_MAX_LOAD_CHUNKS)
      return false;

   /* Zero remains the conservative no-colour/default-fragment spelling.  It
    * has the same loader release mask as one active colour partition. */
   const unsigned colour_classes =
      !desc->fragment ? 0 : desc->colour_classes ? desc->colour_classes : 1;

   struct agx_ioto_state_tail_output tail;
   if (!graphics_state_tail(desc, &tail))
      return false;

   size_t written = 0;
   size_t generated_relocation_count = 0;
   if (!desc->uniform_range_count) {
      const bool composed =
         desc->fragment && desc->writes_sample_mask
            ? agx_ioto_g17p_coverage_fragment_program_compose_trusted(
                 desc->program_va, NULL, 0, NULL, 0, &tail,
                 colour_classes, output, capacity,
                 &written)
            : desc->fragment
            ? agx_ioto_g17p_fragment_no_resource_program_compose_trusted(
                 desc->program_va, &tail, colour_classes, output,
                 capacity, &written)
            : agx_ioto_g17p_no_resource_program_compose_trusted(
                 desc->program_va, &tail, output, capacity,
                 &written);
      const size_t expected =
         desc->fragment ? AGX_IOTO_G17P_FRAGMENT_NO_RESOURCE_PROGRAM_SIZE
                        : AGX_IOTO_G17P_NO_RESOURCE_PROGRAM_SIZE;
      if (!composed || written != expected)
         return false;
   } else {
      uint8_t body[AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE];
      struct agx_ioto_inflight_load loads[AGX3_G17P_GRAPHICS_MAX_LOAD_CHUNKS];
      struct agx_ioto_inflight_load sort[AGX3_G17P_GRAPHICS_MAX_LOAD_CHUNKS];
      struct agx_ioto_load_cache_entry cache[AGX3_G17P_GRAPHICS_MAX_LOAD_CHUNKS];
      const struct agx3_ioto_uniform_program_desc load_desc = {
         .program_va = desc->program_va,
         .ranges = desc->uniform_ranges,
         .range_count = desc->uniform_range_count,
         .cprog_inputs = desc->cprog_inputs,
         .initial_gpr_dwords =
            desc->preloaded_root ? AGX3_G17P_GRAPHICS_ROOT_GPR_DWORDS : 0,
         .preloaded_base_valid = desc->preloaded_root,
         .preloaded_base_half = AGX3_G17P_GRAPHICS_ROOT_BASE_HALF,
         .preloaded_base_address = 0,
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
         .relocations = relocations,
         .relocation_capacity = relocation_capacity,
         .relocation_count = &generated_relocation_count,
      };
      size_t body_size = 0, copy_offset = 0;
      const bool body_generated = desc->preloaded_root
         ? graphics_root_body_generate(desc, body, sizeof(body), &body_size,
                                       false)
         : agx3_ioto_uniform_program_body_generate(
              chip, &load_desc, workspace, &body_size, &copy_offset);
      if (desc->preloaded_root) {
         /* Each root-pointer range is completed serially before the state
          * tail, so the fixed payload window can be reused without retaining
          * per-range GPRs across the tail. */
         copy_offset = body_size;
         generated_relocation_count = 0;
      }

      if (!body_generated ||
          copy_offset > body_size ||
          !(desc->fragment && desc->writes_sample_mask
               ? agx_ioto_g17p_coverage_fragment_program_compose_trusted(
                    desc->program_va, body, copy_offset, body + copy_offset,
                    body_size - copy_offset, &tail, colour_classes,
                    output, capacity, &written)
               : desc->fragment
               ? agx_ioto_g17p_fragment_resource_program_compose_trusted(
                    desc->program_va, body, copy_offset, body + copy_offset,
                    body_size - copy_offset, &tail, colour_classes,
                    output, capacity, &written)
               : agx_ioto_g17p_resource_program_compose_trusted(
                    desc->program_va, body, copy_offset, body + copy_offset,
                    body_size - copy_offset, &tail, output,
                    capacity, &written)))
         return false;

      if (relocations)
         *relocation_count = generated_relocation_count;
   }

   if (!desc->uniform_range_count && relocations)
      *relocation_count = 0;

   const size_t suffix_offset = written - AGX_IOTO_PRESHADER_ABSENT_SIZE;
   if (!graphics_constant_suffix_replace(chip, desc, output, capacity, &written))
      return false;

   if (constant_suffix_offset)
      *constant_suffix_offset = suffix_offset;
   *written_out = written;
   return true;
}

bool
agx3_g17p_graphics_root_prefix_generate(
   enum agx_chip chip, uint64_t root_va,
   struct agx3_g17p_graphics_root_prefix *prefix)
{
   if (!prefix || chip != AGX_CHIP_G17P || !root_va)
      return false;

   struct agx3_g17p_graphics_root_prefix staged = {0};
   uint8_t low[AGX_IOTO_LOAD_LDIMM_MAX_SIZE];
   uint8_t high[AGX_IOTO_LOAD_LDIMM_MAX_SIZE];
   uint8_t low_length, high_length;
   if (!agx_ioto_load_pack_ldimm(
          chip, AGX3_G17P_GRAPHICS_ROOT_BASE_HALF, (uint32_t)root_va,
          low, &low_length) ||
       !agx_ioto_load_pack_ldimm(
          chip, AGX3_G17P_GRAPHICS_ROOT_BASE_HALF + 2,
          (uint32_t)(root_va >> 32), high, &high_length) ||
       (size_t)low_length + high_length > sizeof(staged.bytes))
      return false;

   memcpy(staged.bytes, low, low_length);
   memcpy(staged.bytes + low_length, high, high_length);
   staged.length = low_length + high_length;
   *prefix = staged;
   return true;
}

bool
agx3_g17p_graphics_program_write_unpublished(
   enum agx_chip chip, const struct agx3_g17p_graphics_program_desc *desc,
   void *output, size_t capacity, size_t *written_out)
{
   return graphics_program_write_unpublished(chip, desc, output, capacity,
                                              written_out, NULL, 0, NULL, NULL);
}

bool
agx3_g17p_graphics_program_generate(
   enum agx_chip chip, const struct agx3_g17p_graphics_program_desc *desc,
   struct agx3_g17p_graphics_program *program)
{
   if (!program)
      return false;

   uint8_t staged[AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE];
   size_t written;
   if (!agx3_g17p_graphics_program_write_unpublished(
          chip, desc, staged, sizeof(staged), &written) ||
       written > UINT16_MAX)
      return false;

   memcpy(program->bytes, staged, written);
   program->length = (uint16_t)written;
   return true;
}

bool
agx3_g17p_graphics_program_template_generate(
   enum agx_chip chip, const struct agx3_g17p_graphics_program_desc *desc,
   struct agx3_g17p_graphics_program_template *program)
{
   if (!program)
      return false;

   size_t written, relocation_count, suffix_offset;
   if (!graphics_program_write_unpublished(
          chip, desc, program->program.bytes,
          sizeof(program->program.bytes), &written, program->relocations,
          ARRAY_SIZE(program->relocations), &relocation_count, &suffix_offset) ||
       written > UINT16_MAX || relocation_count > UINT16_MAX)
      return false;

   program->program.length = (uint16_t)written;
   program->relocation_count = (uint16_t)relocation_count;
   program->constant_suffix_offset = (uint16_t)suffix_offset;
   return true;
}

static bool
graphics_range_end(const struct agx3_ioto_uniform_range *range,
                   uint64_t *end)
{
   if (!range || !end || !range->size_halfs ||
       (range->size_halfs & 1) || (range->start_halfs & 1) ||
       (range->address & 3))
      return false;

   const uint64_t bytes = (uint64_t)range->size_halfs * 2;
   if (range->address > UINT64_MAX - (bytes - 4))
      return false;

   *end = range->address + bytes - 4;
   return true;
}

static bool
graphics_ranges_close(const struct agx3_ioto_uniform_range *a,
                      const struct agx3_ioto_uniform_range *b,
                      bool *close)
{
   uint64_t a_end, b_end;
   if (!close || !graphics_range_end(a, &a_end) ||
       !graphics_range_end(b, &b_end))
      return false;

   if (a->address <= b_end && b->address <= a_end) {
      *close = true;
   } else if (a_end < b->address) {
      *close = b->address - a_end <= INT16_MAX;
   } else {
      *close = a->address - b_end <= INT16_MAX;
   }
   return true;
}

static unsigned
graphics_component_root(unsigned parent[AGX3_G17P_GRAPHICS_MAX_RANGES],
                        unsigned i)
{
   while (parent[i] != i) {
      parent[i] = parent[parent[i]];
      i = parent[i];
   }
   return i;
}

static bool
graphics_range_components(
   const struct agx3_ioto_uniform_range *ranges, size_t count,
   unsigned parent[AGX3_G17P_GRAPHICS_MAX_RANGES])
{
   for (unsigned i = 0; i < count; ++i) {
      uint64_t ignored;
      if (!graphics_range_end(&ranges[i], &ignored))
         return false;
      parent[i] = i;
   }

   for (unsigned i = 0; i < count; ++i) {
      for (unsigned j = i + 1; j < count; ++j) {
         bool close;
         if (!graphics_ranges_close(&ranges[i], &ranges[j], &close))
            return false;
         if (close) {
            unsigned a = graphics_component_root(parent, i);
            unsigned b = graphics_component_root(parent, j);
            parent[b] = a;
         }
      }
   }

   for (unsigned i = 0; i < count; ++i)
      parent[i] = graphics_component_root(parent, i);
   return true;
}

static bool
graphics_program_shape_equal(
   const struct agx3_g17p_graphics_program_desc *a,
   const struct agx3_g17p_graphics_program_desc *b)
{
   if (!a || !b || a->uniform_range_count != b->uniform_range_count ||
       a->shader_va != b->shader_va ||
       a->shader_binary_size != b->shader_binary_size ||
       a->scratch_size != b->scratch_size || a->nr_gprs != b->nr_gprs ||
       a->compiler_d_valid != b->compiler_d_valid ||
       a->compiler_d != b->compiler_d || a->fragment != b->fragment ||
       a->preloaded_root != b->preloaded_root ||
       a->writes_sample_mask != b->writes_sample_mask ||
       a->colour_classes != b->colour_classes ||
       a->imageblock.sample_size_B != b->imageblock.sample_size_B ||
       a->imageblock.nr_samples != b->imageblock.nr_samples ||
       a->imageblock.tile_size_px != b->imageblock.tile_size_px ||
       a->constant_program.code_size != b->constant_program.code_size ||
       a->constant_program.force_external != b->constant_program.force_external ||
       a->constant_program.request_prelude != b->constant_program.request_prelude ||
       a->constant_program.suppress_prelude != b->constant_program.suppress_prelude ||
       a->constant_program.absolute_branch != b->constant_program.absolute_branch)
      return false;

   static const struct agx3_cprog_inputs empty;
   const struct agx3_cprog_inputs *ai = a->cprog_inputs ? a->cprog_inputs : &empty;
   const struct agx3_cprog_inputs *bi = b->cprog_inputs ? b->cprog_inputs : &empty;
   if (ai->count != bi->count || ai->storage_bytes != bi->storage_bytes ||
       memcmp(ai->halves, bi->halves, sizeof(ai->halves)))
      return false;

   unsigned ap[AGX3_G17P_GRAPHICS_MAX_RANGES];
   unsigned bp[AGX3_G17P_GRAPHICS_MAX_RANGES];
   if (!graphics_range_components(a->uniform_ranges, a->uniform_range_count,
                                  ap) ||
       !graphics_range_components(b->uniform_ranges, b->uniform_range_count,
                                  bp))
      return false;

   for (unsigned i = 0; i < a->uniform_range_count; ++i) {
      if (a->uniform_ranges[i].start_halfs !=
             b->uniform_ranges[i].start_halfs ||
          a->uniform_ranges[i].size_halfs !=
             b->uniform_ranges[i].size_halfs)
         return false;

      for (unsigned j = 0; j < a->uniform_range_count; ++j) {
         const bool a_same = ap[i] == ap[j];
         const bool b_same = bp[i] == bp[j];
         if (a_same != b_same ||
             (a_same && a->uniform_ranges[i].address -
                           a->uniform_ranges[j].address !=
                           b->uniform_ranges[i].address -
                           b->uniform_ranges[j].address))
            return false;
      }
   }
   return true;
}

bool
agx3_g17p_graphics_program_template_relocate(
   enum agx_chip chip,
   const struct agx3_g17p_graphics_program_desc *template_desc,
   const struct agx3_g17p_graphics_program_desc *target_desc,
   const struct agx3_g17p_graphics_program_template *program,
   void *output, size_t capacity, uint16_t *written)
{
   if (chip != AGX_CHIP_G17P || !program || !output || !written ||
       program->program.length == 0 ||
       program->program.length > sizeof(program->program.bytes) ||
       program->program.length > capacity ||
       program->relocation_count > ARRAY_SIZE(program->relocations) ||
       !graphics_program_shape_equal(template_desc, target_desc))
      return false;

   uint8_t staged[AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE];
   memcpy(staged, program->program.bytes, program->program.length);

   for (unsigned r = 0; r < program->relocation_count; ++r) {
      const struct agx_ioto_load_relocation *rel = &program->relocations[r];
      unsigned range = template_desc->uniform_range_count;
      uint64_t range_end;
      for (unsigned i = 0; i < template_desc->uniform_range_count; ++i) {
         if (!graphics_range_end(&template_desc->uniform_ranges[i],
                                 &range_end))
            return false;
         if (rel->address >= template_desc->uniform_ranges[i].address &&
             rel->address <= range_end &&
             !((rel->address - template_desc->uniform_ranges[i].address) & 3)) {
            range = i;
            break;
         }
      }
      if (range == template_desc->uniform_range_count)
         return false;

      const uint64_t delta =
         rel->address - template_desc->uniform_ranges[range].address;
      if (target_desc->uniform_ranges[range].address > UINT64_MAX - delta)
         return false;
      const uint64_t target =
         target_desc->uniform_ranges[range].address + delta;

      uint8_t old_low[AGX_IOTO_LOAD_LDIMM_MAX_SIZE];
      uint8_t old_high[AGX_IOTO_LOAD_LDIMM_MAX_SIZE];
      uint8_t new_low[AGX_IOTO_LOAD_LDIMM_MAX_SIZE];
      uint8_t new_high[AGX_IOTO_LOAD_LDIMM_MAX_SIZE];
      uint8_t old_low_len, old_high_len, new_low_len, new_high_len;
      if (!agx_ioto_load_pack_ldimm(chip, rel->dest_half,
                                    (uint32_t)rel->address, old_low,
                                    &old_low_len) ||
          !agx_ioto_load_pack_ldimm(chip, rel->dest_half + 2,
                                    (uint32_t)(rel->address >> 32), old_high,
                                    &old_high_len) ||
          !agx_ioto_load_pack_ldimm(chip, rel->dest_half, (uint32_t)target,
                                    new_low, &new_low_len) ||
          !agx_ioto_load_pack_ldimm(chip, rel->dest_half + 2,
                                    (uint32_t)(target >> 32), new_high,
                                    &new_high_len) ||
          old_low_len != rel->low_length ||
          old_high_len != rel->high_length ||
          new_low_len != old_low_len || new_high_len != old_high_len ||
          rel->offset > program->program.length ||
          old_low_len + old_high_len > program->program.length - rel->offset ||
          memcmp(program->program.bytes + rel->offset, old_low,
                 old_low_len) ||
          memcmp(program->program.bytes + rel->offset + old_low_len,
                 old_high, old_high_len))
         return false;

      memcpy(staged + rel->offset, new_low, new_low_len);
      memcpy(staged + rel->offset + new_low_len, new_high, new_high_len);
   }

   if (template_desc->constant_program.code_size) {
      const size_t offset = program->constant_suffix_offset;
      struct agx_ioto_preshader_output old_suffix, new_suffix;
      if (offset > program->program.length ||
          !graphics_constant_suffix_emit(chip, template_desc, offset,
                                         &old_suffix) ||
          !graphics_constant_suffix_emit(chip, target_desc, offset,
                                         &new_suffix) ||
          old_suffix.length != program->program.length - offset ||
          old_suffix.length != new_suffix.length ||
          old_suffix.mode != new_suffix.mode ||
          memcmp(staged + offset, old_suffix.bytes, old_suffix.length) ||
          (old_suffix.mode != AGX_IOTO_PRESHADER_MODE_EXTERNAL &&
           memcmp(old_suffix.bytes, new_suffix.bytes, old_suffix.length)))
         return false;

      memcpy(staged + offset, new_suffix.bytes, new_suffix.length);
   }

   memcpy(output, staged, program->program.length);
   *written = program->program.length;
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
   case AGX3_G17P_RT0_BG_CLEAR_RELOAD:
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

/* Clear values of five to eight mixed-format targets, eight halves each after
 * the four reserved halves. */
static bool
agx3_g17p_wide_mrt_clear_push_halfs(unsigned push_count)
{
   return push_count > AGX3_G17P_QUAD_CLEAR_PUSH_HALFS &&
          push_count <= AGX3_G17P_OCTA_CLEAR_PUSH_HALFS &&
          (push_count - 4) % 8 == 0;
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
              push_count == AGX3_G17P_QUAD_CLEAR_PUSH_HALFS ||
              agx3_g17p_wide_mrt_clear_push_halfs(push_count)) &&
             !uses_txf;
   case AGX3_G17P_RT0_BG_RELOAD:
      return push_count == 0 && uses_txf;
   case AGX3_G17P_RT0_BG_CLEAR_RELOAD:
      return (push_count == AGX3_G17P_RGBA8_MRT_CLEAR_PUSH_HALFS ||
              push_count == AGX3_G17P_TRIPLE_CLEAR_PUSH_HALFS ||
              push_count == AGX3_G17P_QUAD_CLEAR_PUSH_HALFS ||
              agx3_g17p_wide_mrt_clear_push_halfs(push_count)) &&
             uses_txf;
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

   struct agx3_ioto_uniform_range ranges[3];
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
           clear_push_halfs != AGX3_G17P_QUAD_CLEAR_PUSH_HALFS &&
           !agx3_g17p_wide_mrt_clear_push_halfs(clear_push_halfs)))
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
   case AGX3_G17P_RT0_BG_CLEAR_RELOAD:
      /* Texture and sampler pointers occupy halves 0..7 and each target's
       * clear value its own eight halves from 4 + 8 * target, so a cleared
       * target must follow at least one reloaded target. */
      if (!desc->clear_value_va || !desc->texture_pointer_va ||
          !desc->sampler_pointer_va ||
          !agx3_g17p_rt0_bg_shader_resources_match(
             desc->op, desc->clear_push_halfs, true) ||
          desc->clear_start_halfs < 12 ||
          desc->clear_start_halfs > desc->clear_push_halfs - 8 ||
          (desc->clear_start_halfs - 4) % 8)
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
      ranges[2] = (struct agx3_ioto_uniform_range){
         .start_halfs = desc->clear_start_halfs,
         .size_halfs = desc->clear_push_halfs - desc->clear_start_halfs,
         .address = desc->clear_value_va,
      };
      range_count = 3;
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

   uint8_t body[768];
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
   if (!agx_ioto_g17p_background_resource_program_compose_trusted(
          desc->program_va, body, copy_offset, body + copy_offset,
          body_size - copy_offset, &tail, staged.bytes, sizeof(staged.bytes),
          &written) ||
       written > UINT16_MAX ||
       (desc->op == AGX3_G17P_RT0_BG_CLEAR &&
        written >
           (desc->clear_push_halfs > AGX3_G17P_QUAD_CLEAR_PUSH_HALFS
               ? AGX_IOTO_G17P_OCTA_BACKGROUND_PROGRAM_MAX_SIZE
            : desc->clear_push_halfs >= AGX3_G17P_TRIPLE_CLEAR_PUSH_HALFS
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
   const uint8_t offsets[AGX3_G17P_MRT_MAX_TARGETS] = {
      desc ? desc->imageblock_offset_B : 0,
      desc ? desc->imageblock_offset1_B : 0,
      desc ? desc->imageblock_offset2_B : 0,
      desc ? desc->imageblock_offset3_B : 0,
      desc ? desc->imageblock_offset4_B : 0,
      desc ? desc->imageblock_offset5_B : 0,
      desc ? desc->imageblock_offset6_B : 0,
      desc ? desc->imageblock_offset7_B : 0,
   };
   const bool high_offsets = offsets[4] || offsets[5] || offsets[6] ||
                             offsets[7];
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
      !high_offsets && desc->imageblock.nr_samples == 1;
   /* The same program reads every target's offset from its own uniform, so
    * targets of other formats may sit at any offset inside the sample. Its
    * multisample form has the same length and registers (texture type 4
    * changes three bytes per target); the imageblock carries the count.
    *
    * A layered framebuffer uses Apple's 2D-ARRAY builtin instead. It has the
    * same uniform contract and differs only in also publishing the layer wire,
    * which lengthens the program and costs one more GPR dword. Its multisample
    * form likewise has the same length and registers. */
   const bool mixed_2d = desc && desc->shader_binary_size == 32 + 44 * count &&
                         desc->nr_gprs == 8;
   const bool mixed_array = desc &&
                            desc->shader_binary_size == 42 + 54 * count &&
                            desc->nr_gprs == 10;
   bool mixed_mrt =
      desc && desc->mixed_mrt && count >= 2 &&
      count <= AGX3_G17P_MRT_MAX_TARGETS &&
      (mixed_2d || mixed_array) &&
      desc->staging_halfs == 8 * count &&
      (desc->imageblock.nr_samples == 1 ||
       desc->imageblock.nr_samples == 2 ||
       desc->imageblock.nr_samples == 4);
   for (unsigned rt = 0; mixed_mrt && rt < AGX3_G17P_MRT_MAX_TARGETS; ++rt) {
      mixed_mrt = rt < count ? offsets[rt] < desc->imageblock.sample_size_B
                             : offsets[rt] == 0;
   }
   const bool multi_mrt = raw32_mrt || mixed_mrt;
   if (!program || chip != AGX_CHIP_G17P || !desc ||
       (!rt0 && !dual_mrt && !multi_mrt) ||
       (!mixed_mrt && high_offsets) ||
       !desc->pbe_argument_va ||
       desc->pbe_argument_va > UINT64_C(0xffffffffffff) -
          (multi_mrt ? 32 * count - 1 : dual_mrt ? 63 : 31) ||
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

   uint8_t body[(2 + AGX3_G17P_MRT_MAX_TARGETS) * AGX_IOTO_LOAD_LDIMM_MAX_SIZE];
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

   const unsigned targets = multi_mrt ? count : dual_mrt ? 2 : 1;
   for (unsigned rt = 0; rt < targets; ++rt) {
      uint8_t offset_length;
      if (!agx_ioto_load_pack_ldimm(chip, 8 + 2 * rt, offsets[rt],
                                    body + body_size, &offset_length))
         return false;
      body_size += offset_length;
   }

   uint8_t copies[4 * (2 * AGX3_G17P_MRT_MAX_TARGETS + 1)];
   /* Native offset uniforms are u4/u8/.../u32, not contiguous dwords: target
    * n's offset is copied from GPR 8 + 2n to uniform 4 + 4n. Padding
    * positions are unused but initialized deterministically from GPR 8. */
   const unsigned copy_count = 2 * targets + 1;
   for (unsigned i = 0; i < copy_count; ++i) {
      const uint16_t destination_half = 2u * i;
      const uint16_t source_half =
         i < 2 ? 4 + 2 * i : i % 2 ? 8 : 8 + i - 2;
      uint8_t bytes[AGX3_COPY_MAX_LENGTH], length = 0;
      if (!agx3_copy_encode_reg_to_uniform(
             AGX_ISA_ENCODING_AGX3B, destination_half, source_half, false,
             bytes, &length) ||
          length != 4)
         return false;
      memcpy(copies + 4 * i, bytes, length);
   }

   struct agx3_g17p_rt0_eot_program staged = {0};
   size_t written = 0;
   if (!agx_ioto_g17p_resource_program_compose_trusted(
          desc->program_va, body, body_size, copies, 4 * copy_count, &tail,
          staged.bytes, sizeof(staged.bytes), &written) ||
       written > sizeof(staged.bytes))
      return false;

   staged.length = (uint16_t)written;
   *program = staged;
   return true;
}
