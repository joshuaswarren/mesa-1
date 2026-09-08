/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * AGX3 f32 ALU with source/output modifiers (abs, neg, saturate).
 *
 * The grade-A no-modifier fadd/fmul/ffma are the compact hand-written packers.
 * When a modifier is present the compact forms cannot express it, so these
 * encoders emit the always-capable long form through the composer: both
 * commutative source slots (f35, f18) carry independent abs/neg bits, the ffma
 * addend (f52) carries its own, and f14 saturates the result. There is no slot
 * swap in those legacy interfaces. The runtime-aware ADD additionally tries
 * the equivalent commuted order, moving both modifiers and lifetime hints,
 * to keep subtraction in the shortest native form.
 *
 * Evidence: research/gpu/GPU_AGX3_FLOAT_MODIFIERS_2026-08-24.md (96/96 byte
 * exact from baseline + operands + modifiers).
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "asahi/lib/agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

struct agx3_falu_mods {
   bool src0_abs; /* f47 on the f35 slot */
   bool src0_neg; /* f48 */
   bool src1_abs; /* f30 on the f18 slot */
   bool src1_neg; /* f31 */
   bool addend_abs; /* f64, ffma only */
   bool addend_neg; /* f65, ffma only */
   bool saturate;   /* f14 */
};

#define AGX3_FALU_MOD_MAX_LENGTH 16u

/* dest = sat?(srcmods(src0) OP srcmods(src1)). abs applies before neg. */
bool agx3_falu_add_mod(enum agx_isa_encoding enc, uint16_t dest, uint16_t src0,
                       uint16_t src1, const struct agx3_falu_mods *m,
                       uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len);
bool agx3_falu_mul_mod(enum agx_isa_encoding enc, uint16_t dest, uint16_t src0,
                       uint16_t src1, const struct agx3_falu_mods *m,
                       uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len);
/* dest = sat?(srcmods(src0) * srcmods(src1) + addendmods(addend)). */
bool agx3_falu_fma_mod(enum agx_isa_encoding enc, uint16_t dest, uint16_t src0,
                       uint16_t src1, uint16_t addend,
                       const struct agx3_falu_mods *m,
                       uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len);

/* Actual post-RA hints, in API source order; no specimen lifetime defaults. */
struct agx3_falu_runtime {
   bool cache[3];
   bool discard[3];
   bool dest_cache;
   uint8_t wait_mask;
};

bool agx3_falu_add_mod_runtime(enum agx_isa_encoding enc, uint16_t dest,
                              uint16_t src0, uint16_t src1,
                              const struct agx3_falu_mods *m,
                              const struct agx3_falu_runtime *runtime,
                              uint8_t out[AGX3_FALU_MOD_MAX_LENGTH],
                              uint8_t *len);

bool agx3_falu_mul_mod_runtime(enum agx_isa_encoding enc, uint16_t dest,
                              uint16_t src0, uint16_t src1,
                              const struct agx3_falu_mods *m,
                              const struct agx3_falu_runtime *runtime,
                              uint8_t out[AGX3_FALU_MOD_MAX_LENGTH],
                              uint8_t *len);

/* f32 to f16 conversion, including odd destination halves. Runtime source0
 * is the sole real operand; the other input is exact negative zero. */
bool agx3_falu_narrow_f32(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                         bool saturate, const struct agx3_falu_runtime *runtime,
                         uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len);

bool agx3_falu_fma_mod_runtime(enum agx_isa_encoding enc, uint16_t dest,
                              uint16_t src0, uint16_t src1, uint16_t addend,
                              const struct agx3_falu_mods *m,
                              const struct agx3_falu_runtime *runtime,
                              uint8_t out[AGX3_FALU_MOD_MAX_LENGTH],
                              uint8_t *len);

bool agx3_falu_fma_addend_imm(enum agx_isa_encoding enc, uint16_t dest,
                             uint16_t src0, uint16_t src1, uint8_t immediate,
                             const struct agx3_falu_runtime *runtime,
                             uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len);

bool agx3_falu_fma_addend_imm_mod(
   enum agx_isa_encoding enc, uint16_t dest, uint16_t src0, uint16_t src1,
   uint8_t immediate, bool src0_abs, bool src0_neg, bool src1_abs,
   bool src1_neg,
   const struct agx3_falu_runtime *runtime,
   uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len);

bool agx3_falu_fma_two_imm(enum agx_isa_encoding enc, uint16_t dest,
                          uint16_t src, uint8_t multiplier, uint8_t addend,
                          const struct agx3_falu_runtime *runtime,
                          uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len);

#ifdef __cplusplus
}
#endif
