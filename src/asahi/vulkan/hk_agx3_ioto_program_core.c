/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_agx3_ioto_program_core.h"

#include "asahi/compiler/agx3_graphics_program.h"
#include "asahi/compiler/agx3_compute_program.h"
#include "asahi/lib/agx_ioto_core.h"
#include "asahi/lib/agx_tilebuffer.h"
#include "asahi/compiler/agx3_binder.h"

#include <string.h>

bool
hk_agx3_ioto_program_generate(
   enum agx_chip chip, const struct hk_agx3_ioto_program_desc *desc,
   void *output, size_t capacity, size_t *written)
{
   if (!desc || !agx_ioto_target_supported(chip))
      return false;

   const struct agx_ioto_profile_control profile = {
      .chip = chip,
      .payload0 = desc->profile_payload0,
      .payload1 = desc->profile_payload1,
   };

   return agx_ioto_straight_line_program_compose(
      chip, desc->program_va, desc->body, desc->body_size, &desc->tail,
      &profile, output, capacity, written);
}

bool
hk_agx3_ioto_program_generate_no_pixrel(
   enum agx_chip chip, const struct hk_agx3_ioto_program_desc *desc,
   const struct agx_ioto_preshader_desc *preshader, void *output,
   size_t capacity, size_t *written)
{
   if (!desc || !agx_ioto_target_supported(chip))
      return false;

   const struct agx_ioto_profile_control profile = {
      .chip = chip,
      .payload0 = desc->profile_payload0,
      .payload1 = desc->profile_payload1,
   };

   return agx_ioto_program_compose_no_pixrel(
      chip, desc->program_va, desc->body, desc->body_size, &desc->tail,
      &profile, preshader, output, capacity, written);
}

bool
hk_agx3_g17p_no_resource_program_generate(
   enum agx_chip chip, uint64_t program_va,
   const struct agx_ioto_state_tail_output *packed_tail, void *output,
   size_t capacity, size_t *written)
{
   if (chip != AGX_CHIP_G17P)
      return false;

   return agx_ioto_g17p_no_resource_program_compose(program_va, packed_tail,
                                                    output, capacity, written);
}

bool
hk_agx3_g17p_single_layer_rendering(uint32_t view_mask, uint32_t layer_count)
{
   return view_mask == 0 && layer_count == 1;
}

bool
hk_agx3_g17p_rt0_tile_layout_supported(struct agx_tilebuffer_layout *tib)
{
   if (!tib)
      return false;

   /* Layering changes only the background/end-of-tile image coordinates and
    * descriptors.  It does not change the physical tilebuffer allocation.
    * Validate the latter through the same bounded RT0 contract, while the
    * G17P array BG/EOT programs independently carry the layer coordinate.
    */
   struct agx_tilebuffer_layout physical = *tib;
   physical.layered = false;
   return agx_g17p_rt0_tile_layout_supported(&physical);
}

bool
hk_agx3_g17p_direct_color_resolve_supported(
   struct agx_tilebuffer_layout *tib, unsigned color_attachment_count,
   bool has_resolve_target, bool average_resolve, bool store_source)
{
   return tib && (tib->nr_samples == 2 || tib->nr_samples == 4) &&
          color_attachment_count == 1 && has_resolve_target &&
          average_resolve && !store_source &&
          hk_agx3_g17p_rt0_tile_layout_supported(tib);
}

static bool
hk_agx3_g17p_graphics_state_tail(
   const struct hk_agx3_g17p_graphics_program_desc *desc,
   struct agx_ioto_state_tail_output *tail, bool ordinary)
{
   if (!desc || !tail || desc->shader_binary_size == 0 ||
       desc->nr_gprs == 0)
      return false;

   static const uint8_t vertex_speclm[AGX_IOTO_SPECLM_SIZE] = {
      0x77, 0x00, 0x2a, 0x41, 0x00, 0x00, 0x00, 0x00,
   };
   static const uint8_t fragment_speclm[AGX_IOTO_SPECLM_SIZE] = {
      0x77, 0x00, 0x2a, 0x43, 0x4b, 0x00, 0x00, 0x00,
   };

   struct agx_ioto_state ldshdr = {0};
   ldshdr.f[0] = desc->shader_va;
   /* Native G17P graphics code containers reserve an 0x80-byte framing area
    * before the 64-byte-aligned main code. LDSHDR f1 is the saturated number
    * of complete 0x80-byte spans in that padded main-code region. Valid
    * captures cover container sizes 0xc0/0x100/0x140/0x1c0/0x240 and yield
    * f1 0/1/1/2/3 respectively, including the cold 88-byte VS and 128-byte
    * EOT. Compute uses a separate state builder and remains unchanged.
    */
   const uint64_t padded_shader_size =
      ((uint64_t)desc->shader_binary_size + 63) & ~UINT64_C(63);
   const uint64_t fetch_span = padded_shader_size >> 7;
   ldshdr.f[1] = fetch_span > 3 ? 3 : fetch_span;
   ldshdr.f[23] = 1;

   struct agx_ioto_state spectpr = {0};
   /* The valid cold G1a no-vary fragment and the matched Apple Metal
    * 32023.917 constant-colour probe both omit ProgramVariant field 38.
    * FragmentProgramVariantESLState supplies D=0, whose rounded SPECTPR
    * payload is entirely zero. Mesa's nr_gprs is not this Apple metadata
    * field; using even a two-GPR shader here incorrectly sets f22=2.
    *
    * Resource fragments retain the compiler-footprint path below until their
    * ProgramVariant metadata contract is independently recovered.
   */
   const bool exact_no_resource_fragment =
      ordinary && desc->fragment && desc->uniform_range_count == 0 &&
      desc->scratch_size == 0;
   const uint32_t compiler_d =
      desc->compiler_d_valid ? desc->compiler_d : desc->nr_gprs;
   if ((desc->compiler_d_valid || !exact_no_resource_fragment) &&
       !agx_ioto_g17p_compiler_spectpr_build(desc->scratch_size,
                                             compiler_d, &spectpr)) {
      return false;
   }

   struct agx_ioto_state_tail_output staged = {
      .length = AGX_IOTO_STATE_TAIL_TYPE0_SIZE,
   };
   memcpy(staged.bytes, desc->fragment ? fragment_speclm : vertex_speclm,
          AGX_IOTO_SPECLM_SIZE);
   if (!agx_ioto_state_pack_ldshdr(AGX_CHIP_G17P, &ldshdr,
                                   staged.bytes + AGX_IOTO_SPECLM_SIZE,
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
hk_agx3_g17p_graphics_program_bound(
   enum agx_chip chip, const struct agx3_ioto_uniform_range *ranges,
   size_t range_count, size_t *size)
{
   return agx3_g17p_graphics_program_bound(chip, ranges, range_count, size);
}

bool
hk_agx3_g17p_basic_graphics_program_generate(
   enum agx_chip chip, const struct hk_agx3_g17p_graphics_program_desc *desc,
   struct hk_agx3_g17p_graphics_program *program)
{
   if (!program || !desc)
      return false;
   const struct agx3_g17p_graphics_program_desc shared_desc = {
      .program_va = desc->program_va,
      .shader_va = desc->shader_va,
      .uniform_ranges = desc->uniform_ranges,
      .uniform_range_count = desc->uniform_range_count,
      .shader_binary_size = desc->shader_binary_size,
      .scratch_size = desc->scratch_size,
      .nr_gprs = desc->nr_gprs,
      .compiler_d_valid = desc->compiler_d_valid,
      .compiler_d = desc->compiler_d,
      .fragment = desc->fragment,
      .writes_sample_mask = desc->writes_sample_mask,
      .colour_partitions = desc->colour_partitions,
   };
   struct agx3_g17p_graphics_program shared;
   if (!agx3_g17p_graphics_program_generate(chip, &shared_desc, &shared))
      return false;
   memset(program, 0, sizeof(*program));
   memcpy(program->bytes, shared.bytes, shared.length);
   program->length = shared.length;
   return true;
}

bool
hk_agx3_g17p_combined_sampler_pointers(
   const struct hk_agx3_g17p_combined_sampler_heap *heap,
   uint64_t pointers[2])
{
   if (!heap || !pointers || heap->descriptor_set_va == 0 ||
       heap->sampler_heap_va == 0 || (heap->texture_offset_B & 7) != 0 ||
       heap->sampler_element_size_B != 32 ||
       heap->sampler_index < HK_AGX3_G17P_BINDLESS_SAMPLER_BASE_INDEX)
      return false;

   const uint32_t native_index =
      heap->sampler_index - HK_AGX3_G17P_BINDLESS_SAMPLER_BASE_INDEX;
   if (native_index >= heap->sampler_count)
      return false;

   const uint64_t texture_pointer_va =
      heap->descriptor_set_va + heap->texture_offset_B;
   const uint64_t sampler_offset =
      (uint64_t)native_index * heap->sampler_element_size_B;
   const uint64_t sampler_pointer_va = heap->sampler_heap_va + sampler_offset;
   if (texture_pointer_va < heap->descriptor_set_va ||
       sampler_pointer_va < heap->sampler_heap_va ||
       texture_pointer_va > UINT64_MAX - 31 || sampler_pointer_va > UINT64_MAX - 31)
      return false;

   pointers[0] = texture_pointer_va;
   pointers[1] = sampler_pointer_va;
   return true;
}

bool
hk_agx3_g17p_sampled_storage_pointers(
   const struct hk_agx3_g17p_sampled_storage_heap *heap,
   uint64_t pointers[4])
{
   if (!heap || !pointers || !heap->pbe_descriptor_set_va ||
       (heap->pbe_offset_B & 7) != 0 ||
       heap->pbe_descriptor_set_va > UINT64_MAX - heap->pbe_offset_B ||
       heap->pbe_descriptor_set_va + heap->pbe_offset_B > UINT64_MAX - 31)
      return false;

   uint64_t staged[4] = {0};
   if (!hk_agx3_g17p_combined_sampler_pointers(&heap->sampled, staged))
      return false;

   staged[2] = heap->pbe_descriptor_set_va + heap->pbe_offset_B;
   staged[3] = UINT64_C(0xffff);
   memcpy(pointers, staged, sizeof(staged));
   return true;
}

bool
hk_agx3_g17p_indexed_storage_pointers(
   const struct hk_agx3_g17p_indexed_storage_heap *heap,
   uint64_t pointers[4])
{
   if (!heap || !pointers || !heap->texture_heap_va ||
       !heap->sampler_heap_va || !heap->pbe_descriptor_set_va ||
       (heap->pbe_offset_B & 7) != 0 ||
       heap->pbe_descriptor_set_va > UINT64_MAX - heap->pbe_offset_B ||
       heap->pbe_descriptor_set_va + heap->pbe_offset_B > UINT64_MAX - 31)
      return false;

   const uint64_t staged[4] = {
      heap->texture_heap_va,
      heap->sampler_heap_va,
      heap->pbe_descriptor_set_va + heap->pbe_offset_B,
      UINT64_C(0xffff),
   };
   memcpy(pointers, staged, sizeof(staged));
   return true;
}

bool
hk_agx3_g17p_root_range(uint16_t start_halfs, uint8_t set_count,
                        uint64_t root_va,
                        struct agx3_ioto_uniform_range *range)
{
   if (!range || root_va == 0 || (start_halfs & 1) != 0 ||
       set_count > HK_AGX3_G17P_MAX_ROOT_SETS)
      return false;

   *range = (struct agx3_ioto_uniform_range){
      .start_halfs = start_halfs,
      .size_halfs = 4 * (1 + set_count),
      .address = root_va,
   };
   return true;
}

bool
hk_agx3_g17p_fragment_blend_constants_range(
   uint16_t start_halfs, uint64_t blend_constants_va,
   struct agx3_ioto_uniform_range *range)
{
   if (!blend_constants_va || !range || (start_halfs & 1) != 0 ||
       start_halfs > 512 - 8)
      return false;

   *range = (struct agx3_ioto_uniform_range){
      .start_halfs = start_halfs,
      .size_halfs = 8,
      .address = blend_constants_va,
   };
   return true;
}

bool
hk_agx3_g17p_rt0_bg_resource_spec(
   enum agx_chip chip, enum hk_agx3_g17p_rt0_bg_op op, uint64_t *resource_spec)
{
   return agx3_g17p_rt0_bg_resource_spec(
      chip, (enum agx3_g17p_rt0_bg_op)op, resource_spec);
}

bool
hk_agx3_g17p_rt0_bg_shader_resources_match(
   enum hk_agx3_g17p_rt0_bg_op op, unsigned push_count, bool uses_txf)
{
   return agx3_g17p_rt0_bg_shader_resources_match(
      (enum agx3_g17p_rt0_bg_op)op, push_count, uses_txf);
}

bool
hk_agx3_g17p_rt0_bg_program_generate(
   enum agx_chip chip, const struct hk_agx3_g17p_rt0_bg_program_desc *desc,
   struct hk_agx3_g17p_rt0_bg_program *program)
{
   if (!program || chip != AGX_CHIP_G17P || !desc)
      return false;

   struct agx3_ioto_uniform_range ranges[2];
   size_t range_count;
   switch (desc->op) {
   case HK_AGX3_G17P_RT0_BG_CLEAR: {
      const uint16_t clear_push_halfs = desc->clear_push_halfs
         ? desc->clear_push_halfs : HK_AGX3_G17P_RT0_CLEAR_PUSH_HALFS;
      if (!desc->clear_value_va || desc->texture_pointer_va ||
          desc->sampler_pointer_va ||
          (clear_push_halfs != HK_AGX3_G17P_RT0_CLEAR_PUSH_HALFS &&
           clear_push_halfs != HK_AGX3_G17P_WBOIT_MRT_CLEAR_PUSH_HALFS &&
           clear_push_halfs != HK_AGX3_G17P_RGBA8_MRT_CLEAR_PUSH_HALFS))
         return false;
      ranges[0] = (struct agx3_ioto_uniform_range){
         .start_halfs = 4,
         .size_halfs = clear_push_halfs - 4,
         .address = desc->clear_value_va,
      };
      range_count = 1;
      break;
   }
   case HK_AGX3_G17P_RT0_BG_RELOAD:
      if (desc->clear_value_va || !desc->texture_pointer_va || !desc->sampler_pointer_va)
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

   const struct hk_agx3_g17p_graphics_program_desc shader = {
      .program_va = desc->program_va,
      .shader_va = desc->shader_va,
      .shader_binary_size = desc->shader_binary_size,
      .scratch_size = desc->scratch_size,
      .nr_gprs = desc->nr_gprs,
      /* BG uses the same independent compiler staging requirement as the
       * main graphics stages. In particular a uniform-fed clear needs D=0
       * despite using GPRs. Unknown texel-fetch staging keeps the existing
       * fallback instead of falsely advertising zero. */
      .compiler_d_valid = desc->staging_known,
      .compiler_d = (uint32_t)desc->staging_halfs * 2,
      .fragment = true,
   };
   struct agx_ioto_state_tail_output tail;
   if (!hk_agx3_g17p_graphics_state_tail(&shader, &tail, false))
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
      .load_capacity = sizeof(loads) / sizeof(loads[0]),
      .sort = sort,
      .sort_capacity = sizeof(sort) / sizeof(sort[0]),
      .cache = cache,
      .cache_capacity = sizeof(cache) / sizeof(cache[0]),
   };

   size_t body_size, copy_offset;
   if (!agx3_ioto_uniform_program_body_generate(chip, &load_desc, workspace,
                                                &body_size, &copy_offset) ||
       copy_offset > body_size)
      return false;

   /*
    * The clear program's length is a bound, not a constant. It embeds
    * clear_value_va as two LDIMM records, and agx_ioto_load_pack_ldimm() is
    * variable length: a record whose destination half is below 32 and whose
    * immediate is below 128 takes a 2-byte compact form instead of the 8-byte
    * long one. The address's high word is `clear_value_va >> 32`, so whenever
    * the clear value happens to live below the 128 GiB mark the program comes
    * out 6 bytes shorter -- 140 versus 134 with the background PIXREL terminal
    * protocol.
    *
    * AGX_IOTO_G17P_RT0_CLEAR_PROGRAM_SIZE is therefore the long-form maximum,
    * which is also what the caller allocates (hk_cmd_draw.c:349-351), so a
    * shorter program is written and used as-is. Demanding equality rejected
    * every clear whose colour landed at a low address, which is to say the
    * background program of an entire class of render passes.
    */
   struct hk_agx3_g17p_rt0_bg_program staged;
   size_t written = 0;
   /* Both bounded auxiliary shaders are compiled by agx_bg_eot.c with
    * ignore_tib_dependencies. Their release descriptor is native BG 0x200;
    * ordinary FS 0x2f3 releases a different set of pixel classes.
    */
   if (!agx_ioto_g17p_background_resource_program_compose(
          desc->program_va, body, copy_offset, body + copy_offset,
          body_size - copy_offset, &tail, staged.bytes, sizeof(staged.bytes),
          &written) ||
       written > UINT16_MAX ||
       (desc->op == HK_AGX3_G17P_RT0_BG_CLEAR &&
        written >
           (desc->clear_push_halfs == HK_AGX3_G17P_RGBA8_MRT_CLEAR_PUSH_HALFS
               ? AGX_IOTO_G17P_RGBA8_MRT_CLEAR_PROGRAM_SIZE
            : desc->clear_push_halfs == HK_AGX3_G17P_WBOIT_MRT_CLEAR_PUSH_HALFS
               ? AGX_IOTO_G17P_WBOIT_MRT_CLEAR_PROGRAM_SIZE
               : AGX_IOTO_G17P_RT0_CLEAR_PROGRAM_SIZE)))
      return false;

   staged.length = (uint16_t)written;
   *program = staged;
   return true;
}

bool
hk_agx3_g17p_rt0_eot_program_generate(
   enum agx_chip chip, const struct hk_agx3_g17p_rt0_eot_program_desc *desc,
   struct hk_agx3_g17p_rt0_eot_program *program)
{
   const bool rt0 = desc &&
                    (desc->shader_binary_size == 68 ||
                     desc->shader_binary_size == 84) &&
                    desc->nr_gprs == 8 && desc->staging_halfs == 8 &&
                    (desc->attachment_count == 0 ||
                     desc->attachment_count == 1);
   const bool layered_rt0 = desc && desc->shader_binary_size == 110 &&
                            desc->nr_gprs == 16 &&
                            desc->staging_halfs == 8 &&
                            desc->attachment_count == 1;
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
   if (!program || chip != AGX_CHIP_G17P || !desc ||
       (!rt0 && !layered_rt0 && !dual_mrt) ||
       !desc->pbe_argument_va ||
       desc->pbe_argument_va > UINT64_C(0xffffffffffff) -
          (dual_mrt ? 63 : 31) ||
       (desc->pbe_argument_va & 31))
      return false;

   const struct hk_agx3_g17p_graphics_program_desc shader = {
      .program_va = desc->program_va,
      .shader_va = desc->shader_va,
      .shader_binary_size = desc->shader_binary_size,
      .scratch_size = desc->scratch_size,
      .nr_gprs = desc->nr_gprs,
      .compiler_d_valid = true,
      .compiler_d = desc->staging_halfs * 2,
      .fragment = true,
   };
   struct agx_ioto_state_tail_output tail;
   if (!hk_agx3_g17p_graphics_state_tail(&shader, &tail, false))
      return false;

   /* The EOT consumes a PBE table pointer, not inline descriptor words.
    * Matching G17P eot_uber metadata binds driver argument 0x32 through
    * the qword gather to uniform halves 0..3. The driver supplies a raw,
    * 32-byte-aligned GPU VA to its flat PBE32 table, separately from the
    * image address encoded inside each descriptor.
    *
    * Materialize the pointer in GPR halves4..7: the unchanged profile loader
    * uses0..3 before the final
    * COPYs. No descriptor-memory load or load-completion wait is needed.
    */
   uint8_t body[4 * AGX_IOTO_LOAD_LDIMM_MAX_SIZE];
   size_t body_size = 0;
   for (unsigned i = 0; i < 2; ++i) {
      uint8_t bytes[AGX_IOTO_LOAD_LDIMM_MAX_SIZE], length;
      if (!agx_ioto_load_pack_ldimm(chip, 4 + 2 * i,
                                     (uint32_t)(desc->pbe_argument_va >> (32 * i)),
                                     bytes, &length))
         return false;
      memcpy(body + body_size, bytes, length);
      body_size += length;
   }
   /* Native kind5 supplies the byte offset inside an imageblock sample.
    * Keep it in CProg GPR8 across profile scratch0..3, independently of the
    * fragment program's register allocation. */
   uint8_t offset_length;
   if (!agx_ioto_load_pack_ldimm(chip, 8, desc->imageblock_offset_B,
                                   body + body_size, &offset_length))
      return false;
   body_size += offset_length;

   if (dual_mrt) {
      if (!agx_ioto_load_pack_ldimm(chip, 10, desc->imageblock_offset1_B,
                                    body + body_size, &offset_length))
         return false;
      body_size += offset_length;
   }

   /* Preserve the proven EOT COPY source-file and no-kill policy. Publish
    * pointer dwords in halves0/2 and the imageblock byte offset in half4.
    */
   uint8_t copies[20];
   const struct agx3_field_kv baseline[] = {{9,3},{10,2},{23,2},{24,2}};
   const unsigned copy_count = dual_mrt ? 5 : 3;
   const unsigned copy_sources[] = {4, 6, 8, 8, 10};
   for (unsigned i = 0; i < copy_count; ++i) {
      const struct agx3_field_kv operands[] = {
         {3, 2u * i}, {16, copy_sources[i]},
      };
      uint8_t bytes[AGX3_MAX_WORDS * 2], length = 0;
      if (!agx3_bind_compose(AGX_ISA_ENCODING_AGX3B, "copy", 0,
                               baseline, 4, operands, 2, bytes, &length) ||
          length != 4)
         return false;
      memcpy(copies + 4 * i, bytes, length);
   }

   /* Keep the existing190-byte allocation capacity. A single target with a
    * zero offset needs at most120 bytes; the second LDIMM/COPY pair keeps the
    * admitted dual-target forms at130 bytes. Offsets128..255 grow by six.
    */
   struct hk_agx3_g17p_rt0_eot_program staged = {0};
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

bool
hk_agx3_g17p_direct_add3_program_upload_prepare(
   enum agx_chip chip,
   const struct agx3_g17p_direct_add3_program_desc *desc,
   const struct agx3_compute_program_upload *upload, size_t *written,
   struct agx3_compute_program *program)
{
   return agx3_g17p_direct_add3_program_upload_prepare(
      chip, desc, upload, written, program);
}

bool
hk_agx3_g17p_direct_add3_dispatch_append(
   const struct agx3_compute_program *program,
   const struct agx3_compute_direct_grid *grid,
   struct agx3_compute_dispatch_sink *sink)
{
   return agx3_g17p_direct_add3_dispatch_append(program, grid, sink);
}

bool
hk_agx3_g17p_native_indirect_compute_dispatch_append(
   const struct agx3_compute_program *program,
   const struct agx3_g17p_native_indirect_dispatch *dispatch,
   struct agx3_compute_dispatch_sink *sink)
{
   return agx3_g17p_native_indirect_compute_dispatch_append(program, dispatch,
                                                             sink);
}

bool
hk_agx3_g17p_compute_program_generate(
   enum agx_chip chip, const struct hk_agx3_g17p_compute_program_desc *desc,
   struct hk_agx3_g17p_compute_program *program)
{
   if (!program || !desc)
      return false;
   const struct agx3_g17p_compute_program_desc common_desc = {
      .program_va = desc->program_va,
      .shader_va = desc->shader_va,
      .uniform_ranges = desc->uniform_ranges,
      .uniform_range_count = desc->uniform_range_count,
      .shader_binary_size = desc->shader_binary_size,
      .scratch_size = desc->scratch_size,
      .nr_gprs = desc->nr_gprs,
      .compiler_d_valid = desc->compiler_d_valid,
      .compiler_d = desc->compiler_d,
      .threadgroup_memory_size_B = desc->threadgroup_memory_size_B,
      .launch_sources = desc->launch_sources,
      .ldshdr_f1 = desc->ldshdr_f1,
   };
   struct agx3_g17p_compute_program common;
   if (!agx3_g17p_compute_program_generate(chip, &common_desc, &common))
      return false;

   struct hk_agx3_g17p_compute_program staged = {0};
   memcpy(staged.bytes, common.bytes, common.length);
   staged.length = common.length;
   staged.launch = common.launch;
   *program = staged;
   return true;
}

bool
hk_agx3_g17p_native_compute_dispatch_append(
   const struct agx3_compute_program *program,
   const struct agx3_compute_direct_grid *grid,
   struct agx3_compute_dispatch_sink *sink)
{
   return agx3_g17p_native_compute_dispatch_append(program, grid, sink);
}

bool
hk_agx3_g17s_minimal_compute_program_prepare(
   enum agx_chip chip,
   const struct agx3_g17s_minimal_compute_program_desc *desc, void *output,
   size_t capacity, size_t *written, struct agx3_compute_program *program)
{
   if (chip != AGX_CHIP_G17S)
      return false;

   return agx3_g17s_minimal_compute_program_prepare(
      desc, output, capacity, written, program);
}

bool
hk_agx3_g17s_minimal_compute_program_upload_prepare(
   enum agx_chip chip,
   const struct agx3_g17s_minimal_compute_program_desc *desc,
   const struct agx3_compute_program_upload *upload, size_t *written,
   struct agx3_compute_program *program)
{
   return agx3_g17s_minimal_compute_program_upload_prepare(
      chip, desc, upload, written, program);
}

bool
hk_agx3_g17s_minimal_compute_program_live_upload_prepare(
   enum agx_chip chip,
   const struct agx3_g17s_minimal_compute_program_desc *desc,
   const struct agx_ioto_g17s_minimal_compute_resources *resources,
   const struct agx3_compute_program_upload *upload, size_t *written,
   struct agx3_compute_program *program)
{
   return agx3_g17s_minimal_compute_program_live_upload_prepare(
      chip, desc, resources, upload, written, program);
}

bool
hk_agx3_g17s_minimal_compute_dispatch_append(
   const struct agx3_compute_program *program,
   const struct agx3_compute_direct_grid *grid,
   struct agx3_compute_dispatch_sink *sink)
{
   return agx3_g17s_minimal_compute_dispatch_append(program, grid, sink);
}
