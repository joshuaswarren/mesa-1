/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * AGX3 transcendental / rounding funops: the single-source 10-byte family.
 *
 * floor/ceil/trunc/roundeven are one hardware operation, f32rint,
 * discriminated by the 2-bit rounding-mode field f43 (0=rint 1=trunc 2=ceil
 * 3=floor -- the composer applies Apple's LUT). log2, exp2, sinc and the
 * fragment gradient ops dfdx/dfdy are distinct mnemonics on AGX3b and one
 * f34-op-selected encoding class on AGX3c. Operand roles are uniform across
 * the family: dest=f2, src=f18, abs=f30, neg=f31, and on AGX3c only a native
 * saturate bit f14 (Apple lowers saturate on AGX3b as a separate f32add, so a
 * saturating funop is refused there rather than mis-encoded).
 *
 * This core exists beside agx3_alu_table.h rather than inside it: that table
 * is machine-generated from the hand packers and its generator is not in any
 * checked-out tree, and its row shape (dest/src0/src1/src2/imm) cannot carry
 * the f43 mode or the abs/neg/saturate modifier bits anyway.
 *
 * Grounding: research/gpu/GPU_AGX3_TRANSCENDENTAL_INT_OPS_2026-08-24.md and
 * its baselines JSON (funop section, grade A both encodings); dfdx/dfdy from
 * the fragment corpus transcendental-int-re/corpus_frag. Every encoding this
 * core can produce is replayed byte-for-byte against Apple's own compiler
 * output in test/test-agx3-funop-core.c.
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "asahi/lib/agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

/* All funops are 10 bytes on both encodings. */
#define AGX3_FUNOP_MAX_LENGTH 10u

enum agx3_funop {
   AGX3_FUNOP_FLOOR,     /* f32rint mode 3 */
   AGX3_FUNOP_CEIL,      /* f32rint mode 2 */
   AGX3_FUNOP_TRUNC,     /* f32rint mode 1 */
   AGX3_FUNOP_RINT,      /* f32rint mode 0 (roundeven) */
   AGX3_FUNOP_LOG2,      /* f32log2 */
   AGX3_FUNOP_EXP2,      /* f32exp2 */
   AGX3_FUNOP_SINC,      /* f32sinc on signed reduced quadrants */
   AGX3_FUNOP_DFDX,      /* f32dfdx, fragment-only gradient */
   AGX3_FUNOP_DFDY,      /* f32dfdy, fragment-only gradient */
   AGX3_FUNOP_COUNT,
};

/*
 * Encode one funop. `dest` and `src` are 32-bit registers in half units
 * (even, < 128, matching every witness). `wait_mask` is the scoreboard wait
 * set (bit i = slot i+1); it is accepted only for the ops whose wait-bit
 * fields are grade A in the wait-field map (f32rint/f32log2/f32exp2, fields
 * f35..f40 on AGX3b and f35..f42 on AGX3c). saturate is AGX3c-only.
 */
bool agx3_funop_encode(enum agx_isa_encoding enc, enum agx3_funop op,
                       uint16_t dest, uint16_t src, bool src_abs, bool src_neg,
                       bool saturate, uint16_t wait_mask,
                       uint8_t out[AGX3_FUNOP_MAX_LENGTH], uint8_t *len);

/* Preserve a live G17P rounding/log/exp source. Other instruction families
 * retain their canonical encoding; the wrapper above is the last-use form. */
bool agx3_funop_encode_source_lifetime(
   enum agx_isa_encoding enc, enum agx3_funop op, uint16_t dest, uint16_t src,
   bool src_abs, bool src_neg, bool saturate, uint16_t wait_mask, bool src_live,
   uint8_t out[AGX3_FUNOP_MAX_LENGTH], uint8_t *len);

#ifdef __cplusplus
}
#endif
