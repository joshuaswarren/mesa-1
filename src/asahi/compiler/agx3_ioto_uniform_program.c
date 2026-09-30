/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_ioto_uniform_program.h"

#include <limits.h>
#include <string.h>

#define GPU0_CHUNK_BODY_BOUND 78u
#define GPU1_CHUNK_BODY_BOUND 64u

static bool
uniform_range_valid(const struct agx3_ioto_uniform_range *range)
{
   if (range == NULL || range->size_halfs == 0 ||
       (range->start_halfs & 1) != 0 || (range->size_halfs & 1) != 0 ||
       range->start_halfs > 512 ||
       range->size_halfs > 512 - range->start_halfs ||
       (range->address & 3) != 0)
      return false;

   const uint64_t bytes = (uint64_t)(range->size_halfs / 2) * 4;
   return bytes != 0 && range->address <= UINT64_MAX - (bytes - 1);
}

static bool
add_size(size_t *value, size_t add)
{
   if (*value > SIZE_MAX - add)
      return false;
   *value += add;
   return true;
}

bool
agx3_ioto_uniform_program_body_bound(
   enum agx_chip chip, const struct agx3_ioto_uniform_range *ranges,
   size_t range_count, size_t *bytes, size_t *chunks)
{
   if (bytes == NULL || chunks == NULL ||
       agx_ioto_load_target_encoding(chip) ==
          AGX_IOTO_LOAD_ENCODING_UNSUPPORTED ||
       (range_count != 0 && ranges == NULL))
      return false;

   size_t staged_bytes =
      range_count == 0 &&
            agx_ioto_load_target_encoding(chip) == AGX_IOTO_LOAD_ENCODING_GPU0
         ? AGX_IOTO_LOAD_GPU0_ROUND_NOP_SIZE
         : 0;
   size_t staged_chunks = 0;
   const size_t per_chunk =
      agx_ioto_load_target_encoding(chip) == AGX_IOTO_LOAD_ENCODING_GPU0
         ? GPU0_CHUNK_BODY_BOUND
         : GPU1_CHUNK_BODY_BOUND;

   for (size_t i = 0; i < range_count; ++i) {
      if (!uniform_range_valid(&ranges[i]))
         return false;

      const size_t range_chunks = (ranges[i].size_halfs + 7u) / 8u;
      if (!add_size(&staged_chunks, range_chunks) ||
          range_chunks > SIZE_MAX / per_chunk ||
          !add_size(&staged_bytes, range_chunks * per_chunk))
         return false;
   }

   *bytes = staged_bytes;
   *chunks = staged_chunks;
   return true;
}

bool
agx3_ioto_uniform_program_desc_body_bound(
   enum agx_chip chip, const struct agx3_ioto_uniform_program_desc *desc,
   size_t *bytes, size_t *chunks)
{
   size_t bound, count;
   if (!desc || !bytes || !chunks ||
       !agx3_ioto_uniform_program_body_bound(chip, desc->ranges,
          desc->range_count, &bound, &count))
      return false;
   if (desc->cprog_inputs) {
      struct agx3_cprog_inputs checked;
      const struct agx3_cprog_inputs *inputs = desc->cprog_inputs;
      if (!agx3_cprog_inputs_init(inputs->halves, &checked) ||
          checked.count != inputs->count ||
          checked.storage_bytes != inputs->storage_bytes ||
          (inputs->count && (chip != AGX_CHIP_G17P || desc->fast_path_enabled)))
         return false;
      uint64_t covered[AGX3_CPROG_INPUT_WORDS] = {0};
      for (size_t i = 0; i < desc->range_count; ++i) {
         const struct agx3_ioto_uniform_range *range = &desc->ranges[i];
         for (unsigned h = range->start_halfs;
              h < range->start_halfs + range->size_halfs; ++h) {
            const uint64_t bit = UINT64_C(1) << (h % 64);
            if (!(inputs->halves[h / 64] & bit))
               continue;
            if (covered[h / 64] & bit)
               return false;
            covered[h / 64] |= bit;
         }
      }
      for (unsigned w = 0; w < AGX3_CPROG_INPUT_WORDS; ++w) {
         if (covered[w] != inputs->halves[w])
            return false;
      }
      /* At worst each selected half uses its own extended COPY. */
      if (!add_size(&bound, 10u * inputs->count))
         return false;
   }
   *bytes = bound;
   *chunks = count;
   return true;
}

static bool
spans_overlap(const void *a, size_t a_size, const void *b, size_t b_size)
{
   if (a_size == 0 || b_size == 0)
      return false;
   if (a == NULL || b == NULL)
      return true;

   const uintptr_t a0 = (uintptr_t)a, b0 = (uintptr_t)b;
   if (a0 > UINTPTR_MAX - a_size || b0 > UINTPTR_MAX - b_size)
      return true;
   return a0 < b0 + b_size && b0 < a0 + a_size;
}

static bool
array_overlaps(const void *array, size_t count, size_t element_size,
               const void *output, size_t output_size)
{
   return count > SIZE_MAX / element_size ||
          spans_overlap(array, count * element_size, output, output_size);
}

static bool
workspace_overlaps(struct agx3_ioto_uniform_program_workspace workspace)
{
   if (workspace.load_capacity >
          SIZE_MAX / sizeof(struct agx_ioto_inflight_load) ||
       workspace.sort_capacity >
          SIZE_MAX / sizeof(struct agx_ioto_inflight_load) ||
       workspace.cache_capacity >
          SIZE_MAX / sizeof(struct agx_ioto_load_cache_entry))
      return true;

   const size_t loads_size =
      workspace.load_capacity * sizeof(struct agx_ioto_inflight_load);
   const size_t sort_size =
      workspace.sort_capacity * sizeof(struct agx_ioto_inflight_load);
   const size_t cache_size =
      workspace.cache_capacity * sizeof(struct agx_ioto_load_cache_entry);

   return spans_overlap(workspace.body, workspace.body_capacity,
                        workspace.loads, loads_size) ||
          spans_overlap(workspace.body, workspace.body_capacity, workspace.sort,
                        sort_size) ||
          spans_overlap(workspace.body, workspace.body_capacity,
                        workspace.cache, cache_size) ||
          spans_overlap(workspace.loads, loads_size, workspace.sort,
                        sort_size) ||
          spans_overlap(workspace.loads, loads_size, workspace.cache,
                        cache_size) ||
          spans_overlap(workspace.sort, sort_size, workspace.cache, cache_size);
}

/* COPY GPR -> uniform with no wait and the source last-use hint: exactly the
 * direct ep0/ep1 forms agx3_ioto_load_packer.c's pack_copy_fields() emits for
 * absolute-load completion (checked byte for byte there against the
 * composer). */
static bool
immediate_copy_record(uint16_t destination_half, uint16_t source_half,
                      uint8_t out[10], size_t *length)
{
   if (destination_half < 128 && source_half < 256) {
      out[0] = 0x0b | ((destination_half & 0x1e) << 3);
      out[1] = source_half & 0x7f;
      out[2] = 0x09 | ((destination_half & 0x60) << 1);
      out[3] = 0x04 | (destination_half & 1) | ((source_half & 0x80) >> 3);
      *length = 4;
      return true;
   }
   if (destination_half < 256 && source_half < 256) {
      out[0] = 0x0b | ((destination_half & 0x1e) << 3);
      out[1] = 0x01 | (source_half & 0x7e);
      out[2] = 0x0f | ((destination_half & 0x60) << 1);
      out[3] = 0x00;
      out[4] = 0x40 | ((destination_half & 0x01) << 2) |
               ((source_half & 0x01) << 3) | ((destination_half & 0x80) >> 6);
      out[5] = 0x10 | ((source_half & 0x80) >> 7);
      memset(&out[6], 0, 4);
      *length = 10;
      return true;
   }
   return false;
}

/* Values go to every other 32-bit register (halves 4, 8, 12, ...): no two
 * LDIMM destinations are adjacent, so no pair can be mistaken for a 64-bit
 * load address by the stream validator, and the r0/r1 profile scratch is
 * never touched. Halves stay below the GPU0 allocator limit (97 dwords). */
#define AGX3_IOTO_IMMEDIATE_FIRST_HALF 4u
#define AGX3_IOTO_IMMEDIATE_HALF_STRIDE 4u
#define AGX3_IOTO_IMMEDIATE_MAX_DWORDS 44u

/* All-immediate body: LDIMM every source dword into GPR halves 4, 6, ...
 * (clear of the r0/r1 profile scratch), then the COPY suffix. Returns false
 * (without writing) when the shape is not eligible; the caller then uses the
 * ordinary load path. */
static bool
immediate_body_generate(enum agx_chip chip,
                        const struct agx3_ioto_uniform_program_desc *desc,
                        struct agx3_ioto_uniform_program_workspace workspace,
                        size_t *written, size_t *copy_offset)
{
   if (!desc->range_cpu || desc->cprog_inputs || desc->preloaded_base_valid ||
       desc->initial_gpr_dwords || chip != AGX_CHIP_G17P ||
       agx_ioto_load_target_encoding(chip) != AGX_IOTO_LOAD_ENCODING_GPU0 ||
       desc->range_count == 0)
      return false;

   unsigned total = 0;
   for (size_t i = 0; i < desc->range_count; ++i) {
      const struct agx3_ioto_uniform_range *range = &desc->ranges[i];
      if (!desc->range_cpu[i] || (range->start_halfs & 1) ||
          (range->size_halfs & 1) || !range->size_halfs)
         return false;
      total += range->size_halfs / 2;
   }
   if (total > AGX3_IOTO_IMMEDIATE_MAX_DWORDS)
      return false;

   size_t length = 0;
   unsigned slot = 0;
   for (size_t i = 0; i < desc->range_count; ++i) {
      const struct agx3_ioto_uniform_range *range = &desc->ranges[i];
      const uint8_t *src = desc->range_cpu[i];
      for (unsigned d = 0; d < range->size_halfs / 2; ++d, ++slot) {
         uint32_t value;
         memcpy(&value, src + 4 * d, sizeof(value));
         uint8_t bytes[AGX_IOTO_LOAD_LDIMM_MAX_SIZE], n;
         if (!agx_ioto_load_pack_ldimm(
                chip,
                AGX3_IOTO_IMMEDIATE_FIRST_HALF +
                   AGX3_IOTO_IMMEDIATE_HALF_STRIDE * slot,
                value,
                bytes, &n) ||
             length + n > workspace.body_capacity)
            return false;
         memcpy(workspace.body + length, bytes, n);
         length += n;
      }
   }

   *copy_offset = length;
   slot = 0;
   for (size_t i = 0; i < desc->range_count; ++i) {
      const struct agx3_ioto_uniform_range *range = &desc->ranges[i];
      for (unsigned d = 0; d < range->size_halfs / 2; ++d, ++slot) {
         uint8_t bytes[10];
         size_t n;
         if (!immediate_copy_record(range->start_halfs + 2 * d,
                                    AGX3_IOTO_IMMEDIATE_FIRST_HALF +
                                       AGX3_IOTO_IMMEDIATE_HALF_STRIDE * slot,
                                    bytes, &n) ||
             length + n > workspace.body_capacity)
            return false;
         memcpy(workspace.body + length, bytes, n);
         length += n;
      }
   }

   *written = length;
   return true;
}

bool
agx3_ioto_uniform_program_body_generate(
   enum agx_chip chip, const struct agx3_ioto_uniform_program_desc *desc,
   struct agx3_ioto_uniform_program_workspace workspace, size_t *written,
   size_t *copy_offset)
{
   size_t body_bound, chunk_count;
   if (desc == NULL || written == NULL || copy_offset == NULL ||
       !agx3_ioto_uniform_program_desc_body_bound(
          chip, desc, &body_bound, &chunk_count) ||
       (body_bound > workspace.body_capacity &&
        !(desc->cprog_inputs && desc->cprog_inputs->count)) ||
       (workspace.body_capacity != 0 && workspace.body == NULL) ||
       workspace_overlaps(workspace) ||
       (workspace.relocation_capacity && !workspace.relocations) ||
       (workspace.relocations && !workspace.relocation_count))
      return false;

   (void)chunk_count;
   if (immediate_body_generate(chip, desc, workspace, written, copy_offset)) {
      if (workspace.relocations)
         *workspace.relocation_count = 0;
      return true;
   }

   struct agx_ioto_load_core core;
   if (!agx_ioto_load_core_init(
          &core, chip,
          (struct agx_ioto_load_sink){workspace.body, workspace.body_capacity,
                                      0},
          workspace.loads, workspace.load_capacity, workspace.cache,
          workspace.cache_capacity, desc->initial_gpr_dwords,
          desc->fast_path_enabled,
          agx_ioto_load_target_encoding(chip) == AGX_IOTO_LOAD_ENCODING_GPU0 &&
             !(chip == AGX_CHIP_G17P &&
               agx_ioto_g17p_native_esl_wait_enabled())))
      return false;
   /* v51y: only G17P has the native copy-carried wait witness. */
   core.gpu0_copy_wait =
      chip == AGX_CHIP_G17P && agx_ioto_g17p_native_esl_wait_enabled();

   core.relocations = workspace.relocations;
   core.relocation_capacity = workspace.relocation_capacity;
   if (desc->preloaded_base_valid &&
       !agx_ioto_load_core_set_fixed_base(
          &core, desc->preloaded_base_address,
          desc->preloaded_base_half))
      return false;

   const struct agx_ioto_load_workspace sort = {
      workspace.sort,
      workspace.sort_capacity,
   };
   uint32_t cprog_dest_half = 0;

   for (size_t i = 0; i < desc->range_count; ++i) {
      const struct agx3_ioto_uniform_range *range = &desc->ranges[i];
      uint32_t dword = 0;
      const uint32_t dword_count = range->size_halfs / 2;
      while (dword < dword_count) {
         const uint8_t count =
            (uint8_t)((dword_count - dword) < 4 ? dword_count - dword : 4);
         const uint8_t lane_mask = (uint8_t)((1u << (2u * count)) - 1u);
         if (cprog_dest_half > UINT16_MAX)
            return false;

         struct agx_ioto_load_absolute load = {
            .native_slot = range->start_halfs / 2 + dword,
            .count_dwords = count,
            .address = range->address + (uint64_t)dword * 4,
            .lane_mask = lane_mask,
            .cprog_dest_half = (uint16_t)cprog_dest_half,
         };
         if (desc->cprog_inputs) {
            for (unsigned h = 0; h < 2u * count; ++h) {
               uint16_t tps;
               if (!agx3_cprog_input_tps_half(desc->cprog_inputs,
                          range->start_halfs + 2u * dword + h, &tps))
                  continue;
               if (!load.cprog_input_mask)
                  load.cprog_input_tps_half = tps;
               load.cprog_input_mask |= 1u << h;
            }
         }
         struct agx_ioto_load_allocation allocation;
         if (!agx3_ioto_load_absolute_native(&core, &load, sort, &allocation))
            return false;

         cprog_dest_half += 2u * (2u * count);
         dword += count;
      }
   }

   if (!agx3_ioto_load_finish_round_native(&core, true, sort))
      return false;

   *written = core.sink.length;
   *copy_offset = core.copy_suffix_offset;
   if (workspace.relocations)
      *workspace.relocation_count = core.relocation_count;
   return true;
}

bool
agx3_ioto_uniform_program_generate(
   enum agx_chip chip, const struct agx3_ioto_uniform_program_desc *desc,
   struct agx3_ioto_uniform_program_workspace workspace, void *output,
   size_t capacity, size_t *written)
{
   size_t body_bound, chunk_count;
   if (desc == NULL || output == NULL || written == NULL ||
       !agx3_ioto_uniform_program_desc_body_bound(
          chip, desc, &body_bound, &chunk_count) ||
       (body_bound > workspace.body_capacity &&
        !(desc->cprog_inputs && desc->cprog_inputs->count)) ||
       (workspace.body_capacity != 0 && workspace.body == NULL) ||
       workspace_overlaps(workspace) ||
       spans_overlap(workspace.body, workspace.body_capacity, output,
                     capacity) ||
       array_overlaps(workspace.loads, workspace.load_capacity,
                      sizeof(workspace.loads[0]), output, capacity) ||
       array_overlaps(workspace.sort, workspace.sort_capacity,
                      sizeof(workspace.sort[0]), output, capacity) ||
       array_overlaps(workspace.cache, workspace.cache_capacity,
                      sizeof(workspace.cache[0]), output, capacity))
      return false;

   (void)body_bound;
   (void)chunk_count;
   size_t body_size, copy_offset;
   if (!agx3_ioto_uniform_program_body_generate(chip, desc, workspace,
                                                &body_size, &copy_offset))
      return false;
   (void)copy_offset;

   const struct agx_ioto_profile_control profile = {
      .chip = chip,
      .payload0 = desc->profile_payload0,
      .payload1 = desc->profile_payload1,
   };
   return agx_ioto_program_compose_no_pixrel(
      chip, desc->program_va, workspace.body, body_size, &desc->tail, &profile,
      &desc->preshader, output, capacity, written);
}
