/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * AGX3 control-flow encoders.
 *
 * AGX3 splits AGX2's fused compare-and-branch. A compare (icmp/fcmp) writes a
 * persistent per-lane flag slot; the exec-mask stack ops (cndst push, cndlt
 * loop/break test, cndef else, cndend pop) consume `flag[slot] ^ invert`; and
 * b/bn are the actual branches, carrying a signed byte displacement patched in
 * after block offsets are known. Conditions are AGX2's own icond/fcond enums.
 *
 * Every encoder here is grade A: it reproduces Apple's witnessed instruction
 * bytes exactly via the generated composer. What is NOT here is the lowering --
 * turning IR structured control flow into these sequences with flag-slot
 * allocation and branch fixups -- which lives in the backend. See
 * research/gpu/GPU_AGX3_CONTROL_FLOW_2026-08-24.md.
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "asahi/lib/agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

/* AGX3 integer compare conditions (== enum agx_icond). le/ge/ne are expressed
 * as the inverse condition plus the consumer's invert bit. */
enum agx3_cf_icond {
   AGX3_CF_ICOND_ULT = 1,
   AGX3_CF_ICOND_UGT = 2,
   AGX3_CF_ICOND_EQ = 4,
   AGX3_CF_ICOND_SLT = 5,
   AGX3_CF_ICOND_SGT = 6,
};

/* AGX3 float compare conditions (== enum agx_fcond base values). With or_nan
 * set, lt/gt become the IEEE ordered-or-unordered variants Apple uses to build
 * ge/le/ne by inversion. */
enum agx3_cf_fcond {
   AGX3_CF_FCOND_EQ = 0,
   AGX3_CF_FCOND_LT = 1,
   AGX3_CF_FCOND_GT = 2,
};

#define AGX3_CF_MAX_LENGTH 10u
#define AGX3_CF_MAX_FLAG_SLOT 5u /* slots 0..5 are witnessed */

struct agx3_cf_source_hints {
   bool cache;
   bool discard;
};

bool agx3_cf_icmp_reg16(enum agx_isa_encoding enc, enum agx3_cf_icond cond,
                        unsigned slot, const uint16_t src[2],
                        const struct agx3_cf_source_hints hints[2], uint16_t wait_mask,
                        uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len);

/* Compares: write flag[slot] = (src0 cond src1). */
bool agx3_cf_icmp_reg(enum agx_isa_encoding enc, enum agx3_cf_icond cond,
                      unsigned slot, uint16_t src0, uint16_t src1,
                      uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len);
bool agx3_cf_icmp_imm(enum agx_isa_encoding enc, enum agx3_cf_icond cond,
                      unsigned slot, uint16_t src0, uint8_t imm,
                      uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len);
/*
 * Compare a 16-bit register against the immediate zero -- the shape every
 * branch on a boolean takes, since Mesa keeps booleans in 16-bit halves.
 *
 * The compare width lives in f10 (1 = 16-bit, 2 = 32-bit); Apple's own
 * `if (ushort != 0)` sets it to 1 and adds f14, where the 32-bit form has
 * neither. Probe-witnessed with `metal-nt -arch applegpu_g16g`; the non-zero
 * 16-bit immediate forms are not covered here. Register/register compares
 * use agx3_cf_icmp_reg16(), including its compact unsigned encoding point.
 */
bool agx3_cf_icmp_zero16_runtime(enum agx_isa_encoding enc, enum agx3_cf_icond cond,
                                 unsigned slot, uint16_t src,
                                 struct agx3_cf_source_hints hint, uint16_t wait_mask,
                                 uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len);

bool agx3_cf_icmp_zero16(enum agx_isa_encoding enc, enum agx3_cf_icond cond,
                         unsigned slot, uint16_t src0,
                         uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len);

bool agx3_cf_fcmp_reg(enum agx_isa_encoding enc, enum agx3_cf_fcond cond,
                      bool or_nan, unsigned slot, uint16_t src0, uint16_t src1,
                      uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len);

/* G17P register comparison with exactly one negated f32 source. */
bool agx3_cf_fcmp_reg_neg(enum agx_isa_encoding enc,
                          enum agx3_cf_fcond cond, bool or_nan, unsigned slot,
                          const uint16_t src[2], const bool neg[2],
                          const struct agx3_cf_source_hints hints[2],
                          uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len);

/* Float comparison against literal zero, retaining the actual source lifetime. */
bool agx3_cf_fcmp_zero(enum agx_isa_encoding enc, enum agx3_cf_fcond cond,
                        bool or_nan, unsigned slot, uint16_t source, bool source_16,
                        struct agx3_cf_source_hints hint, uint16_t wait_mask,
                        uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len);

/* Exec-mask stack. cndst pushes (an `if`), with push count 1 or 2 (2 = a loop
 * entry that reserves a nesting level). cndlt is the loop/break test (push
 * count fixed at 2). cndef is the `else` flip. cndend pops `count` levels. */
bool agx3_cf_cndst(enum agx_isa_encoding enc, unsigned slot, bool invert,
                   unsigned push_count, uint8_t out[AGX3_CF_MAX_LENGTH],
                   uint8_t *len);
bool agx3_cf_cndlt(enum agx_isa_encoding enc, unsigned slot, bool invert,
                   uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len);
bool agx3_cf_cndef(enum agx_isa_encoding enc, uint8_t out[AGX3_CF_MAX_LENGTH],
                   uint8_t *len);
bool agx3_cf_cndend(enum agx_isa_encoding enc, unsigned pop_count,
                    uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len);

/* Branches. Emit with displacement 0 (a reserved 10-byte template), then patch
 * the byte displacement once the target offset is known. is_bn selects
 * branch-if-none-active (forward, an exec-mask skip) vs branch-if-any-active
 * (the backward loop edge). */
bool agx3_cf_branch(enum agx_isa_encoding enc, bool is_bn,
                    uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len);

/* OR the signed byte displacement (target - branch_start, must be even) into a
 * reserved branch template in place. Returns false on an odd or out-of-range
 * displacement. */
bool agx3_cf_patch_branch(enum agx_isa_encoding enc, bool is_bn,
                          uint8_t *branch, int64_t displacement);

#ifdef __cplusplus
}
#endif
