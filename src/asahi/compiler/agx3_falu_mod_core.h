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

/*
 * A slot's WIDTH is a field of the same encoding point as its modifier bits,
 * not a different instruction: f43 belongs to the f35 slot exactly as f47 and
 * f48 do, f26 to the f18 slot alongside f30 and f31, and f60 to the ffma
 * addend alongside f64 and f65. Apple composes every combination. One kernel
 * per cell, `float a`, `half b`, recovered through tools/decomp.c:
 *
 *    a + b        09011c0400c0   f35=4 f43=1  f18=0 f26=2
 *    a - b        09011c0400c8   the same, plus f48
 *    b - a        09041c0100c8   f35=0 f43=2  f18=4 f26=1, plus f48
 *    a + fabs(b)  09011c04020000800200   f47
 *    fabs(a) + b  09011c04020000800100   f30
 *    b - fabs(a)  09041c01020800800200   f47 and f48 together
 *   -a - b        09041c0101080280       f48 and f31 together
 *    b * -a       09041d0100c8           the same pair on f32mul
 *    fma(a,c,h)   09011e05810800c0       f52=8 f60=1, the addend narrow
 *    fma(a,c,-h)  09011e05910800c0       the same, plus f65
 *    fma(h,c,a)   09081e01810402c0       f18=8 f26=1, the multiplicand narrow
 *
 * so the width axis and the modifier axis are independent. Keeping them in
 * separate encoders is what refused `half + (-float)`: it is on both axes and
 * had no home in either.
 */
struct agx3_falu_mods {
   bool src0_abs; /* f47 on the f35 slot */
   bool src0_neg; /* f48 */
   bool src0_16;  /* f43 */
   bool src1_abs; /* f30 on the f18 slot */
   bool src1_neg; /* f31 */
   bool src1_16;  /* f26 */
   bool addend_abs; /* f64, ffma only */
   bool addend_neg; /* f65, ffma only */
   bool addend_16;  /* f60, ffma only */
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
