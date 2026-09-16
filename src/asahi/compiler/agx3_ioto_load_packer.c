/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_ioto_load_packer.h"

#include "agx3_binder.h"
#include "agx3_static_spill_core.h"

#include <string.h>

#define KV(f, v) {(uint16_t)(f), (uint64_t)(v)}

bool
agx3_ioto_pack_cprog_copy(enum agx_chip chip,
                         const struct agx3_ioto_cprog_copy_desc *desc,
                         struct agx3_ioto_cprog_copy_output *output)
{
   if (chip != AGX_CHIP_G17P || desc == NULL || output == NULL ||
       (desc->destination_bytes & 1) || desc->source_half >= 256 ||
       (desc->source_half & 1) || desc->lane_mask < 1 || desc->lane_mask > 3 ||
       desc->source_wait_slot < -1 || desc->source_wait_slot >= 6)
      return false;

   struct agx3_ioto_cprog_copy_output staged = {0};
   uint32_t destination = 8u + desc->destination_bytes / 2u;
   uint16_t source = desc->source_half + (desc->lane_mask == 2);
   unsigned remaining = desc->lane_mask == 3 ? 2 : 1;
   staged.written_bytes = 2 * remaining;

   while (remaining) {
      const unsigned halves = remaining == 2 && !(destination & 1) ? 2 : 1;
      const struct agx3_static_spill_store_desc store = {
         .tps_half = destination,
         .src = source,
         .width = halves == 2 ? AGX3_MEM_W32 : AGX3_MEM_W16,
         .src_kill = true,
         .slot = 6,
         .wait_mask = desc->source_wait_slot < 0
                         ? 0
                         : (uint16_t)(1u << desc->source_wait_slot),
      };
      uint8_t encoded[AGX3_STATIC_SPILL_MAX_LENGTH], length;
      if (!agx3_static_spill_encode_store(AGX_ISA_ENCODING_AGX3B, &store,
                                         encoded, &length) ||
          length > sizeof(staged.bytes) - staged.length)
         return false;

      memcpy(staged.bytes + staged.length, encoded, length);
      staged.length += length;
      destination += halves;
      source += halves;
      remaining -= halves;
   }

   *output = staged;
   return true;
}

static enum agx_ioto_load_raw_kind
gpu1_lddev_kind(uint8_t ep)
{
   return (enum agx_ioto_load_raw_kind)(AGX_IOTO_LOAD_RAW_LDDEV_EP0 + ep);
}

static bool
direct_load_registers_ok(const struct agx_ioto_inflight_load *load)
{
   return load != NULL && load->data.type == 3 && load->data.value == 1 &&
          load->base.type == 4 && load->base.value == 8 &&
          load->count_dwords >= 1 && load->count_dwords <= 4;
}

static bool
pack_absolute_load(void *context, enum agx_chip chip,
                   const struct agx_ioto_inflight_load *load,
                   struct agx_ioto_load_raw_record *record,
                   uint8_t *signal_slot)
{
   struct agx3_ioto_load_packer *packer = context;
   if (packer == NULL || record == NULL || signal_slot == NULL ||
       packer->chip != chip || !direct_load_registers_ok(load))
      return false;

   const uint8_t mask = (uint8_t)((1u << load->count_dwords) - 1u);
   const struct agx3_mem_desc desc = {
      .data = load->data.half,
      .width = AGX3_MEM_W32,
      .mask = mask,
      .base_is_uniform = false,
      .base = load->base.half,
      .base_kill = false,
      .idx_kind = AGX3_MEM_IDX_IMM,
      .idx_scaled = false,
      .byte_off = load->byte_offset,
      .slot = 1,
      .wait_mask = 0,
   };

   memset(record, 0, sizeof(*record));
   uint8_t length = 0;
   if (packer->encoding == AGX_IOTO_LOAD_ENCODING_GPU0) {
      if (!agx3_mem_encode_load_form(AGX_ISA_ENCODING_AGX3B, &desc, 0,
                                     record->bytes, &length) ||
          length != AGX_IOTO_LOAD_INLINE_LDDEV14_SIZE)
         return false;
      record->kind = AGX_IOTO_GPU0_INLINE_LDDEV14;
   } else if (packer->encoding == AGX_IOTO_LOAD_ENCODING_GPU1) {
      /* Apple's generic HAL300 emitter tries 0,1,3,4,2,5.  The fixed
       * loadAbsolute register forms are in the 0/1/2 family, have no waits,
       * and therefore select ep0 for an unsigned byte offset <= 255 and ep2
       * for the remaining signed-16-bit offsets. */
      const uint8_t ep =
         load->byte_offset >= 0 && load->byte_offset <= 255 ? 0 : 2;
      if (!agx3_mem_encode_load_form(AGX_ISA_ENCODING_AGX3C, &desc, ep,
                                     record->bytes, &length))
         return false;
      record->kind = gpu1_lddev_kind(ep);
   } else {
      return false;
   }

   record->length = length;
   *signal_slot = 1;
   return true;
}

static bool
add_half(uint16_t base, uint16_t increment, uint16_t *result)
{
   if ((unsigned)base + increment > UINT16_MAX)
      return false;
   *result = (uint16_t)(base + increment);
   return true;
}

static bool
pack_copy_fields(enum agx_isa_encoding enc, uint16_t destination_half,
                 uint16_t source_half, int wait_slot, bool source_kill,
                 struct agx_ioto_load_raw_record *record)
{
   /* f23 selects the register source file; f27 is carried as a last-use
    * hint by this all-dwords gathered-copy path. Native EOT's final gathered
    * copies have it set, but native BG also has f27-clear copies with
    * nonzero sources followed by further reads of those source halves
    * (valid194053 first Ioto page+0x10a: 4b2c0144). Therefore f27 must not be
    * described as selecting gathered data instead of register zero. Keep
    * the source live when a constant-program input copy follows; otherwise
    * retain the existing last-use policy.
    */
   struct agx3_field_kv baseline[7] = {
      KV(9, 3),
      KV(10, 2),
      KV(23, 2),
      KV(24, 2),
      KV(27, source_kill),
   };
   unsigned nbaseline = 5;
   if (enc == AGX_ISA_ENCODING_AGX3C) {
      baseline[nbaseline++] = (struct agx3_field_kv)KV(28, 1);
      baseline[nbaseline++] = (struct agx3_field_kv)KV(29, 35);
   }

   struct agx3_field_kv overrides[3] = {
      KV(3, destination_half),
      KV(16, source_half),
   };
   unsigned noverrides = 2;
   if (wait_slot >= 0) {
      if (wait_slot >= (enc == AGX_ISA_ENCODING_AGX3C ? 8 : 6))
         return false;
      overrides[noverrides++] = (struct agx3_field_kv)KV(30 + wait_slot, 1);
   }

   const uint8_t max_ep = 1;
   for (uint8_t ep = 0; ep <= max_ep; ++ep) {
      uint8_t length = 0;
      if (!agx3_bind_compose(enc, "copy", ep, baseline, nbaseline, overrides,
                             noverrides, record->bytes, &length))
         continue;

      record->kind =
         ep == 0 ? AGX_IOTO_LOAD_RAW_COPY_EP0 : AGX_IOTO_LOAD_RAW_COPY_EP1;
      record->length = length;
      return true;
   }

   return false;
}

static bool
cprog_lanes(const struct agx3_ioto_load_packer *packer,
             const struct agx_ioto_inflight_load *load, uint8_t dword,
             uint8_t *mask, uint16_t *offset)
{
   *mask = 0;
   *offset = 0;
   if (packer->cprog_inputs == NULL)
      return true;
   if (packer->chip != AGX_CHIP_G17P || !packer->cprog_inputs->known ||
       !direct_load_registers_ok(load) || dword >= load->count_dwords)
      return false;

   const unsigned uniform = load->native_slot_half + 2u * dword;
   if (uniform >= AGX_IOTO_CPROG_UNIFORM_HALFS - 1)
      return false;
   for (unsigned h = 0; h < 2; ++h) {
      unsigned half = uniform + h;
      if (packer->cprog_inputs->uniform_halfs[half / 64] &
          (UINT64_C(1) << (half % 64)))
         *mask |= 1u << h;
   }
   return *mask == 0 ||
          agx_ioto_cprog_input_offset(packer->chip, packer->cprog_inputs,
                                      uniform + (*mask == 2), offset);
}

static bool
pack_absolute_copy(void *context, enum agx_chip chip,
                   const struct agx_ioto_inflight_load *load, uint8_t dword,
                   uint8_t lane_code, struct agx_ioto_load_raw_record *record)
{
   struct agx3_ioto_load_packer *packer = context;
   if (packer == NULL || record == NULL || packer->chip != chip ||
       !direct_load_registers_ok(load) || dword >= load->count_dwords ||
       lane_code == 0 || lane_code > 3)
      return false;

   uint16_t destination_half, source_half;
   const uint16_t destination_stride =
      packer->encoding == AGX_IOTO_LOAD_ENCODING_GPU0 ? 2 : 4;
   if (!add_half(load->native_slot_half, (uint16_t)(destination_stride * dword),
                 &destination_half) ||
       !add_half(load->data.half, (uint16_t)(2u * dword), &source_half))
      return false;

   int wait_slot = -1;
   enum agx_isa_encoding enc;
   if (packer->encoding == AGX_IOTO_LOAD_ENCODING_GPU0) {
      /* The first of Apple's four NOPs drains slot 1 before any GPU0 COPY. */
      enc = AGX_ISA_ENCODING_AGX3B;
   } else if (packer->encoding == AGX_IOTO_LOAD_ENCODING_GPU1) {
      enc = AGX_ISA_ENCODING_AGX3C;
      if (!packer->copy_wait_emitted) {
         if (load->signal_slot == 0 || load->signal_slot > 8)
            return false;
         wait_slot = load->signal_slot;
      }
   } else {
      return false;
   }

   uint8_t cprog_mask;
   uint16_t cprog_offset;
   if (!cprog_lanes(packer, load, dword, &cprog_mask, &cprog_offset))
      return false;

   memset(record, 0, sizeof(*record));
   if (!pack_copy_fields(enc, destination_half, source_half, wait_slot,
                         cprog_mask == 0, record))
      return false;

   if (packer->encoding == AGX_IOTO_LOAD_ENCODING_GPU1)
      packer->copy_wait_emitted = true;
   return true;
}

static bool
cprog_copy_bound(void *context, enum agx_chip chip,
                  const struct agx_ioto_inflight_load *load, size_t *bytes)
{
   const struct agx3_ioto_load_packer *packer = context;
   if (packer == NULL || packer->chip != chip || bytes == NULL ||
       !direct_load_registers_ok(load))
      return false;

   /* Keep ordinary uniform uploads independent of the preamble input mask. */
   const unsigned full_mask = (1u << (2u * load->count_dwords)) - 1;
   if ((load->lane_mask & full_mask) != full_mask)
      return false;

   size_t total = 0;
   for (unsigned dword = 0; dword < load->count_dwords; ++dword) {
      uint8_t mask;
      uint16_t offset;
      if (!cprog_lanes(packer, load, dword, &mask, &offset))
         return false;
      total += ((mask & 1) != 0) * AGX_IOTO_LOAD_GPU0_COPY_MAX_SIZE;
      total += ((mask & 2) != 0) * AGX_IOTO_LOAD_GPU0_COPY_MAX_SIZE;
   }
   *bytes = total;
   return true;
}

static bool
pack_cprog_copies(void *context, enum agx_chip chip,
                   const struct agx_ioto_inflight_load *load, uint8_t dword,
                   struct agx_ioto_load_raw_record records[2], uint8_t *count)
{
   const struct agx3_ioto_load_packer *packer = context;
   uint8_t mask;
   uint16_t offset, source;
   if (packer == NULL || packer->chip != chip || records == NULL || count == NULL ||
       !cprog_lanes(packer, load, dword, &mask, &offset) ||
       !add_half(load->data.half, 2u * dword, &source))
      return false;
   *count = 0;
   if (mask == 0)
      return true;

   const struct agx3_ioto_cprog_copy_desc desc = {
      .destination_bytes = offset,
      .source_half = source,
      .lane_mask = mask,
      /* The existing first round NOP retires the loaded registers. */
      .source_wait_slot = -1,
   };
   struct agx3_ioto_cprog_copy_output output;
   if (!agx3_ioto_pack_cprog_copy(chip, &desc, &output))
      return false;

   size_t cursor = 0;
   while (cursor < output.length) {
      struct agx_ioto_record decoded;
      size_t consumed;
      if (*count >= 2 ||
          !agx_ioto_validate_kind(chip, AGX_IOTO_KIND_COPY_EP1,
                                  output.bytes + cursor, output.length - cursor,
                                  &decoded, &consumed) ||
          consumed != AGX_IOTO_LOAD_GPU0_COPY_MAX_SIZE)
         return false;
      records[*count] = (struct agx_ioto_load_raw_record){
         .kind = AGX_IOTO_LOAD_RAW_COPY_EP1,
         .length = consumed,
      };
      memcpy(records[*count].bytes, output.bytes + cursor, consumed);
      ++*count;
      cursor += consumed;
   }
   return true;
}

bool
agx3_ioto_load_packer_init(struct agx3_ioto_load_packer *packer,
                           enum agx_chip chip)
{
   if (packer == NULL)
      return false;
   const enum agx_ioto_load_encoding encoding =
      agx_ioto_load_target_encoding(chip);
   if (encoding == AGX_IOTO_LOAD_ENCODING_UNSUPPORTED)
      return false;

   *packer = (struct agx3_ioto_load_packer){
      .chip = chip,
      .encoding = encoding,
   };
   return true;
}

bool
agx3_ioto_load_absolute_packer_ready(enum agx_chip chip)
{
   return agx_ioto_load_target_encoding(chip) !=
          AGX_IOTO_LOAD_ENCODING_UNSUPPORTED;
}

void
agx3_ioto_load_packer_ops(struct agx3_ioto_load_packer *packer,
                          struct agx_ioto_load_emit_ops *ops)
{
   if (ops == NULL)
      return;
   *ops = (struct agx_ioto_load_emit_ops){
      .pack_load = pack_absolute_load,
      .pack_copy = pack_absolute_copy,
      .extra_copy_bound = packer != NULL && packer->cprog_inputs != NULL
                             ? cprog_copy_bound : NULL,
      .pack_extra_copies = packer != NULL && packer->cprog_inputs != NULL
                             ? pack_cprog_copies : NULL,
      .context = packer,
   };
}

bool
agx3_ioto_load_finish_round_native(struct agx_ioto_load_core *core,
                                   bool complete_round,
                                   struct agx_ioto_load_workspace workspace)
{
   if (core == NULL || (core->encoding == AGX_IOTO_LOAD_ENCODING_GPU0 &&
                        complete_round && !core->gpu0_round_nops))
      return false;

   struct agx3_ioto_load_packer packer;
   struct agx_ioto_load_emit_ops ops;
   if (!agx3_ioto_load_packer_init(&packer, core->chip))
      return false;
   agx3_ioto_load_packer_ops(&packer, &ops);
   return agx_ioto_load_finish_round(core, complete_round, &ops, workspace);
}

bool
agx3_ioto_load_absolute_native(struct agx_ioto_load_core *core,
                               const struct agx_ioto_load_absolute *load,
                               struct agx_ioto_load_workspace workspace,
                               struct agx_ioto_load_allocation *allocation)
{
   if (core == NULL || (core->encoding == AGX_IOTO_LOAD_ENCODING_GPU0 &&
                        !core->gpu0_round_nops))
      return false;

   struct agx3_ioto_load_packer packer;
   struct agx_ioto_load_emit_ops ops;
   if (!agx3_ioto_load_packer_init(&packer, core->chip))
      return false;
   agx3_ioto_load_packer_ops(&packer, &ops);
   return agx_ioto_load_absolute(core, load, &ops, workspace, allocation);
}

bool
agx3_ioto_load_from_scs_native(struct agx_ioto_load_core *core,
                               const struct agx_ioto_load_from_fixed *load,
                               struct agx_ioto_load_workspace workspace,
                               struct agx_ioto_load_allocation *allocation)
{
   if (core == NULL || (core->encoding == AGX_IOTO_LOAD_ENCODING_GPU0 &&
                        !core->gpu0_round_nops))
      return false;

   struct agx3_ioto_load_packer packer;
   struct agx_ioto_load_emit_ops ops;
   if (!agx3_ioto_load_packer_init(&packer, core->chip))
      return false;
   agx3_ioto_load_packer_ops(&packer, &ops);
   return agx_ioto_load_from_scs(core, load, &ops, workspace, allocation);
}

bool
agx3_ioto_load_from_abi_buffer_native(
   struct agx_ioto_load_core *core, enum agx_ioto_abi_buffer_slot source,
   const struct agx_ioto_load_from_fixed *load,
   struct agx_ioto_load_workspace workspace,
   struct agx_ioto_load_allocation *allocation)
{
   if (core == NULL || (core->encoding == AGX_IOTO_LOAD_ENCODING_GPU0 &&
                        !core->gpu0_round_nops))
      return false;

   struct agx3_ioto_load_packer packer;
   struct agx_ioto_load_emit_ops ops;
   if (!agx3_ioto_load_packer_init(&packer, core->chip))
      return false;
   agx3_ioto_load_packer_ops(&packer, &ops);
   return agx_ioto_load_from_abi_buffer(core, source, load, &ops, workspace,
                                        allocation);
}
