/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Exact AGX3 EslBuilder direct device-load record packer.
 *
 * This is intentionally narrower than a general shader-ISA encoder.  It maps
 * the fixed Register forms queued by Apple's G15/G16/G17P and G17G/S/C
 * EslBuilder implementations to native LDDEV/COPY records and supplies them
 * to the bounded agx_ioto_load_core scheduler.
 */

#pragma once

#include "asahi/lib/agx_ioto_load_core.h"
#include "agx3_mem_core.h"
#include "agx3_cprog_input.h"

#ifdef __cplusplus
extern "C" {
#endif

struct agx3_ioto_load_packer {
   enum agx_chip chip;
   enum agx_ioto_load_encoding encoding;
   bool copy_wait_emitted;
   /* HK_G17P_NATIVE_ESL_WAIT (v51y, default off): GPU0 carries the slot
    * wait on the first COPY of a round instead of the four-NOP drain. */
   bool gpu0_copy_wait;
};

bool agx3_ioto_load_packer_init(struct agx3_ioto_load_packer *packer,
                                enum agx_chip chip);

/* True only for the exact direct-load register subset implemented here.  This
 * is not the broad production-readiness gate for arbitrary resources. */
bool agx3_ioto_load_absolute_packer_ready(enum agx_chip chip);

void agx3_ioto_load_packer_ops(struct agx3_ioto_load_packer *packer,
                               struct agx_ioto_load_emit_ops *ops);

bool
agx3_ioto_load_finish_round_native(struct agx_ioto_load_core *core,
                                   bool complete_round,
                                   struct agx_ioto_load_workspace workspace);

bool
agx3_ioto_load_absolute_native(struct agx_ioto_load_core *core,
                               const struct agx_ioto_load_absolute *load,
                               struct agx_ioto_load_workspace workspace,
                               struct agx_ioto_load_allocation *allocation);

bool agx3_ioto_load_from_abi_buffer_native(
   struct agx_ioto_load_core *core, enum agx_ioto_abi_buffer_slot source,
   const struct agx_ioto_load_from_fixed *load,
   struct agx_ioto_load_workspace workspace,
   struct agx_ioto_load_allocation *allocation);

/* Emit one persistent-root range without embedding its source address in
 * executable bytes.  root_base_half names the preloaded root pointer;
 * pointer_offset selects a qword in that root containing the actual source
 * VA.  The emitted straight-line sequence loads that pointer, consumes it in
 * one or more dependent LDDEVs, and copies every dword to its final uniform.
 * It is allocation-free and may be concatenated for every structural range. */
bool agx3_ioto_load_indirect_uniform_range(
   enum agx_chip chip, uint16_t root_base_half, int16_t pointer_offset,
   uint16_t uniform_half, uint16_t size_halfs, void *output, size_t capacity,
   size_t *written);

/* External setup runs after LDSHDR, which does not preserve the launcher's
 * ordinary GPRs.  This form reads the root pointer from a CProg-uniform qword
 * previously written by the launcher, then uses the same qualified two-level
 * LDDEV schedule as the ordinary-GPR form above. */
bool agx3_ioto_load_indirect_uniform_range_from_uniform_root(
   enum agx_chip chip, uint16_t root_base_half, int16_t pointer_offset,
   uint16_t uniform_half, uint16_t size_halfs, void *output, size_t capacity,
   size_t *written);

/* v51y HK_G17P_NATIVE_ESL_WAIT: the native-shape schedule for several
 * root-relative ranges at once.  All pointer LDDEVs issue back to back, one
 * witnessed drain retires them before the dependent payload LDDEVs, and the
 * payload loads are retired by the slot-2 wait carried on the first COPY
 * (native 2b240944) instead of a second drain per load.  Returns false,
 * writing nothing, when the GPR window would exceed the native-witnessed
 * r25 (half 51) or any input is out of range; the caller then keeps the
 * per-range schedule.  CProg-input ranges are not handled here. */
struct agx3_ioto_indirect_range {
   int16_t pointer_offset;
   uint16_t uniform_half;
   uint16_t size_halfs;
};

bool agx3_ioto_load_indirect_uniform_ranges_native(
   enum agx_chip chip, uint16_t root_base_half, bool root_is_uniform,
   const struct agx3_ioto_indirect_range *ranges, unsigned count,
   void *output, size_t capacity, size_t *written);

/* Exposed for the offline encoding check: the AGX3B 32-bit register-to-
 * uniform COPY, optionally carrying scoreboard wait index `wait_slot`
 * (f30 + wait_slot; -1 for none). */
bool agx3_ioto_load_pack_copy_for_test(uint16_t destination_half,
                                       uint16_t source_half, int wait_slot,
                                       bool source_kill, uint8_t out[10],
                                       uint8_t *length);

bool agx3_ioto_load_indirect_uniform_range_cprog(
   enum agx_chip chip, uint16_t root_base_half, bool root_is_uniform,
   int16_t pointer_offset, uint16_t uniform_half, uint16_t size_halfs,
   const struct agx3_cprog_inputs *inputs, void *output, size_t capacity,
   size_t *written);

#ifdef __cplusplus
}
#endif
