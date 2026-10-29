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
 * Every encoding this core produces matches Apple's own compiler output
 * byte for byte on both encodings; dfdx/dfdy come from its fragment programs.
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

/* Preserve a live G17P rounding/log/exp source. Other instruction families
 * retain their canonical encoding; the wrapper above is the last-use form. */
bool agx3_funop_encode_source_lifetime(
   enum agx_isa_encoding enc, enum agx3_funop op, uint16_t dest, uint16_t src,
   bool src_abs, bool src_neg, bool saturate, uint16_t wait_mask, bool src_live,
   uint8_t out[AGX3_FUNOP_MAX_LENGTH], uint8_t *len);

/* Whether the witnessed form of `op` has scoreboard wait fields. A caller
 * with an outstanding dependency on a form without them must retire it
 * first (as for the gradients). */
bool agx3_funop_carries_wait(enum agx_isa_encoding enc, enum agx3_funop op);

#ifdef __cplusplus
}
#endif
