/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Composer-driven AGX3 ALU encoder that carries scoreboard wait bits.
 *
 * The compact hand packers (agx3_select_core) cannot express a wait mask, but an
 * ALU instruction that consumes an async load result must wait its slot. This
 * encodes the same grade-A ALU forms through the composer so the per-mnemonic
 * wait-bit fields can be set. With wait_mask == 0 it reproduces the hand packers
 * byte-for-byte; the backend uses the compact packer when no wait is needed and
 * this when one is.
 */
#pragma once
#include <stdbool.h>
#include <stdint.h>
#include "asahi/lib/agx_chip.h"
#include "agx3_alu_table.h"
#ifdef __cplusplus
extern "C" {
#endif
#define AGX3_ALU_MAX_LENGTH 16u
/* Ops that alu_compose can carry a wait on (fmin/fmax's tied-operand fsel form
 * is not modeled here; the backend keeps those on the compact packer). */
bool agx3_alu_wait_supported(enum agx3_alu_op op);
bool agx3_alu_compose(enum agx_isa_encoding enc, enum agx3_alu_op op,
                      uint16_t dest, uint16_t src0, uint16_t src1,
                      uint16_t src2, uint32_t imm, uint16_t wait_mask,
                      uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Post-RA source hints. These are distinct from the SSA kill flag. */
enum agx3_alu_hint {
   AGX3_ALU_HINT_NONE = 0,
   AGX3_ALU_HINT_DISCARD = 1,
   AGX3_ALU_HINT_CACHE = 2,
};

/* G17P 32-bit AND/OR/XOR with actual post-RA operand hints. */
bool agx3_alu_bit_reg(enum agx_isa_encoding enc, enum agx3_alu_op op,
                      uint16_t dest, const uint16_t src[2],
                      const enum agx3_alu_hint hints[2], uint16_t wait_mask,
                      uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_bit_reg16(enum agx_isa_encoding enc, enum agx3_alu_op op,
                          uint16_t dest, const uint16_t src[2],
                          const enum agx3_alu_hint hints[2], uint16_t wait_mask,
                          uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_shift_add16(enum agx_isa_encoding enc, uint16_t dest,
                            const uint16_t src[2],
                            const enum agx3_alu_hint hints[2], unsigned shift,
                            bool dest_cache, uint16_t wait_mask,
                            uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* G17P unshifted IADD/ISUB, preserving the canonical operation form. */
bool agx3_alu_integer_reg(enum agx_isa_encoding enc, enum agx3_alu_op op,
                          uint16_t dest, const uint16_t src[2],
                          const enum agx3_alu_hint hints[2], uint16_t wait_mask,
                          uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native signed/unsigned32 to float32 with post-RA source hints. */
bool agx3_alu_int_to_float(enum agx_isa_encoding enc, uint16_t dest,
                            uint16_t src, bool is_signed,
                            enum agx3_alu_hint hint, uint16_t wait_mask,
                            uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native f32 comparison with byte-sized integer alternatives and a16/32-bit result. */
bool agx3_alu_fcmp_select_zero(enum agx_isa_encoding enc, uint16_t dest,
                             const uint16_t src[3], bool compare_16,
                             const enum agx3_alu_hint hints[3], unsigned condition,
                             bool dest_cache, uint16_t wait_mask,
                             uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_fcmp_select_imm(enum agx_isa_encoding enc, uint16_t dest,
                             const uint16_t src[2],
                             const enum agx3_alu_hint hints[2],
                             unsigned condition, uint8_t if_true,
                             uint8_t if_false, bool dest_16, bool dest_cache, uint16_t wait_mask,
                             uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_fcmp_select_imm_width(enum agx_isa_encoding enc, uint16_t dest,
                             const uint16_t src[2], const bool src_16[2],
                             const enum agx3_alu_hint hints[2],
                             unsigned condition, uint8_t if_true,
                             uint8_t if_false, bool dest_16, bool dest_cache,
                             uint16_t wait_mask,
                             uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Plain f32 register FADD/FMUL/FMA with actual post-RA source hints and
 * scheduler waits. Compact witnessed forms only; unsupported wait combinations
 * return false so the caller can retire them explicitly before retrying. */
bool agx3_alu_float_reg(enum agx_isa_encoding enc, enum agx3_alu_op op,
                         uint16_t dest, const uint16_t src[3],
                         const enum agx3_alu_hint hints[3], uint16_t wait_mask,
                         uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_fma_minifloat(enum agx_isa_encoding enc, uint16_t dest,
                             uint16_t src, uint8_t immediate, uint16_t addend,
                             enum agx3_alu_hint src_hint,
                             enum agx3_alu_hint addend_hint, bool dest_cache,
                             uint16_t wait_mask,
                             uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native f32 FMA with a minifloat multiplier and an optionally negated f32
 * register addend. The addend modifier remains fused and rounds only once. */
bool agx3_alu_fma_minifloat_mod(enum agx_isa_encoding enc, uint16_t dest,
                                uint16_t src, uint8_t immediate,
                                uint16_t addend, bool negate_addend,
                                enum agx3_alu_hint src_hint,
                                enum agx3_alu_hint addend_hint,
                                bool dest_cache, uint16_t wait_mask,
                                uint8_t out[AGX3_ALU_MAX_LENGTH],
                                uint8_t *len);

bool agx3_alu_mov16(enum agx_isa_encoding enc, uint16_t dest, uint32_t bits,
                     uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                     uint8_t *len);
bool agx3_alu_fmul_mixed(enum agx_isa_encoding enc, uint16_t dest,
                          uint16_t f32, uint16_t f16,
                          enum agx3_alu_hint hint32, enum agx3_alu_hint hint16,
                          uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                          uint8_t *len);

/* G17P's unsigned16 GPR half -> float32, RTE. Every U16 is exactly
 * representable, so no precision is discarded. Native compiler witnesses
 * include both last-use and reused input, odd half indices and load waits.
 * This does not admit signed, byte, float16, vector, uniform or AGX3c forms.
 */
bool agx3_alu_s16_to_f32(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                        enum agx3_alu_hint hint, uint16_t wait_mask,
                        uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);
bool agx3_alu_add_imm16(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                       uint16_t immediate, enum agx3_alu_hint hint, bool dest_cache,
                       uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_u16_to_f32(enum agx_isa_encoding enc, uint16_t dest,
                        uint16_t src, enum agx3_alu_hint hint, uint16_t wait_mask,
                        uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native G17P f32 GPR + 8-bit minifloat, normal rounding, no modifiers.
 * The six-byte form carries no wait, one of six waits, or slots1+2 together.
 * Other wait combinations, operand files/widths and AGX3c are not admitted.
 */
bool agx3_alu_fadd_minifloat(enum agx_isa_encoding enc, uint16_t dest,
                            uint16_t src, uint16_t immediate, enum agx3_alu_hint hint,
                            uint16_t wait_mask,
                            uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Same f32/minifloat inputs with GPR negation and/or output saturation.
 * Native ep2 carries saturation and all six independent wait bits; the
 * unsaturated ep1 retains the compact helper's bounded wait combinations.
 * Source hints apply to the GPR even when the native operand slots commute.
 */
bool agx3_alu_fadd_minifloat_mod(enum agx_isa_encoding enc, uint16_t dest,
                                uint16_t src, uint16_t immediate,
                                bool negate_src, bool saturate,
                                enum agx3_alu_hint hint, uint16_t wait_mask,
                                uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Exact G17P 1.0 - abs(GPR) form used by the sky blend path. */
bool agx3_alu_fadd_one_absneg(enum agx_isa_encoding enc, uint16_t dest,
                             uint16_t src, bool dest_cache,
                             enum agx3_alu_hint hint, uint16_t wait_mask,
                             uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_fadd_abs_negzero(enum agx_isa_encoding enc, uint16_t dest,
                               uint16_t src, bool dest_cache,
                               enum agx3_alu_hint hint, uint16_t wait_mask,
                               uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_imul_high_div3(enum agx_isa_encoding enc, uint16_t dest,
                             uint16_t src, enum agx3_alu_hint hint,
                             uint16_t wait_mask,
                             uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Same bounded operand/wait domain as fadd_minifloat, native f32 multiply. */
bool agx3_alu_fmul_minifloat(enum agx_isa_encoding enc, uint16_t dest,
                            uint16_t src, uint16_t immediate, enum agx3_alu_hint hint,
                            uint16_t wait_mask,
                            uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);
/* Unsigned G17P extract into a zero base, with a fixed1..31-bit width.
 * Native srl performs shift+mask in one instruction. Dynamic shifts and
 * immediate14, dead/live sources and slot6 waits are native witnesses. */
bool agx3_alu_bfeil(enum agx_isa_encoding enc, uint16_t dest, uint16_t source,
                    uint16_t shift, bool shift_is_register, uint8_t width,
                    const enum agx3_alu_hint hints[2], uint16_t wait_mask,
                    uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *length);

/* Extract32 bits from a low/high register pair at an immediate offset1..31.
 * Native EXTRACT is one instruction and carries both source hints and waits. */
bool agx3_alu_extract_imm32(enum agx_isa_encoding enc, uint16_t dest,
                            const uint16_t low_high[2], unsigned shift,
                            const enum agx3_alu_hint hints[2], bool dest_cache,
                            uint16_t wait_mask,
                            uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *length);

/* Signed right shift by an immediate1..31, with native source lifetime/waits. */
bool agx3_alu_asr_imm32(enum agx_isa_encoding enc, uint16_t dest, uint16_t source,
                        unsigned shift, enum agx3_alu_hint hint, bool dest_cache,
                        uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                        uint8_t *length);

bool agx3_alu_fcmp_select_reg_imm(enum agx_isa_encoding enc, uint16_t dest,
                                  const uint16_t src[4], bool compare_zero, const bool immediate[2],
                                  const enum agx3_alu_hint hints[4], unsigned condition,
                                  bool dest_16, bool dest_cache, uint16_t wait_mask,
                                  uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native square-root step, multiplied by the input by common Mesa lowering. */
bool agx3_alu_sqrt_step(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                       enum agx3_alu_hint hint, bool dest_cache, uint16_t wait_mask,
                       uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_fmul_narrow(enum agx_isa_encoding enc, uint16_t dest,
                           const uint16_t src[2], const enum agx3_alu_hint hints[2],
                           bool dest_cache, uint16_t wait_mask,
                           uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_hmul(enum agx_isa_encoding enc, uint16_t dest,
                   const uint16_t src[2], const enum agx3_alu_hint hints[2],
                   bool dest_cache, uint16_t wait_mask,
                   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_widen_half(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                         enum agx3_alu_hint hint, bool dest_cache, uint16_t wait_mask,
                         uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

#ifdef __cplusplus
}
#endif
