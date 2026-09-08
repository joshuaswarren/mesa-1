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

/* Native G17P population count.  Apple's exact scalar ushort and uint forms
 * are both eight bytes; the source is consumed and no dependency wait is
 * encoded in this form. */
bool agx3_alu_popcount(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                       bool half, enum agx3_alu_hint hint, bool dest_cache,
                       uint16_t wait_mask,
                       uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* G17P 32-bit AND/OR/XOR with actual post-RA operand hints. */
bool agx3_alu_bit_reg(enum agx_isa_encoding enc, enum agx3_alu_op op,
                      uint16_t dest, const uint16_t src[2],
                      const enum agx3_alu_hint hints[2], uint16_t wait_mask,
                      uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_bit_reg16(enum agx_isa_encoding enc, enum agx3_alu_op op,
                          uint16_t dest, const uint16_t src[2],
                          const enum agx3_alu_hint hints[2], uint16_t wait_mask,
                          uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* G17P 32-bit a & ~b.  Apple exposes this truth-table shape as bit_andc;
 * unlike AND it is not commutative, so the source order is preserved. */
bool agx3_alu_bit_andc(enum agx_isa_encoding enc, uint16_t dest,
                      const uint16_t src[2],
                      const enum agx3_alu_hint hints[2], bool dest_cache,
                      uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                      uint8_t *len);

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

/* G17P f32-register == minifloat comparison selecting a half register or
 * immediate.  This is the exact vectorized native form used for half masks. */
bool agx3_alu_fcmp_minifloat_select_half(
   enum agx_isa_encoding enc, uint16_t dest, uint16_t compare,
   uint8_t minifloat, uint16_t if_true, uint8_t if_false,
   enum agx3_alu_hint compare_hint, enum agx3_alu_hint true_hint,
   unsigned condition, bool dest_cache, uint16_t wait_mask,
   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Compare f32 against a minifloat and select the observed mixed register /
 * byte-immediate result shapes used by Hyprland shaders. */
bool agx3_alu_fcmp_minifloat_select_mixed(
   enum agx_isa_encoding enc, uint16_t dest, uint16_t compare,
   uint8_t minifloat, const uint16_t selected[2],
   const bool selected_immediate[2], const enum agx3_alu_hint hints[3],
   unsigned condition, bool dest_16, bool dest_cache, uint16_t wait_mask,
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

/* Native f32-by-f16 multiply rounded directly to a scalar half destination. */
bool agx3_alu_fmul_mixed_half(enum agx_isa_encoding enc, uint16_t dest,
                              uint16_t f32, uint16_t f16,
                              enum agx3_alu_hint hint32,
                              enum agx3_alu_hint hint16, bool dest_cache,
                              uint16_t wait_mask,
                              uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* G17P's unsigned16 GPR half -> float32, RTE. Every U16 is exactly
 * representable, so no precision is discarded. Native compiler witnesses
 * include both last-use and reused input, odd half indices and load waits.
 * This does not admit signed, byte, float16, vector, uniform or AGX3c forms.
 */
bool agx3_alu_s16_to_f32(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                        enum agx3_alu_hint hint, uint16_t wait_mask,
                        uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);
bool agx3_alu_f32_to_s16(enum agx_isa_encoding enc, uint16_t dest,
                        uint16_t src, enum agx3_alu_hint hint,
                        bool dest_cache, uint16_t wait_mask,
                        uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);
bool agx3_alu_add_imm16(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                       uint16_t immediate, enum agx3_alu_hint hint, bool dest_cache,
                       uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);
bool agx3_alu_sub_imm16(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                       uint16_t immediate, enum agx3_alu_hint hint,
                       bool dest_cache, uint16_t wait_mask,
                       uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_u16_to_f32(enum agx_isa_encoding enc, uint16_t dest,
                        uint16_t src, enum agx3_alu_hint hint, uint16_t wait_mask,
                        uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* G17P's find-top-one operation.  This is the native operation
 * selected for llvm.agx2.ft1 and is the direct hardware implementation of
 * Mesa's ufind_msb/FFS instruction.  The zero input result is -1, matching
 * NIR.  Only the witnessed f32-width register form and its native
 * cache/discard and six dependency-wait axes are admitted. */
bool agx3_alu_ft1_u32(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                     enum agx3_alu_hint hint, bool dest_cache,
                     uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                     uint8_t *len);

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

/* G17P register/immediate FSEL. Absolute compare modifiers are grounded for
 * f32 eq/lt/gt with register-register comparators and f32 selected values. */
bool agx3_alu_fcmp_select_reg_imm(enum agx_isa_encoding enc, uint16_t dest,
                                  const uint16_t src[4], bool compare_zero,
                                  const bool compare_abs[2],
                                  const bool immediate[2],
                                  const enum agx3_alu_hint hints[4], unsigned condition,
                                  bool dest_16, bool dest_cache, uint16_t wait_mask,
                                  uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/*
 * Compare a 32-bit float against a minifloat immediate and select either of
 * two 32-bit registers -- what `x < k ? a : b` becomes once Mesa carries the
 * constant as a minifloat. Conditions are enum agx_fcond; eq, lt, gt, ge and
 * le are witnessed, ltn and gtn are not and are refused. An inverted
 * condition needs no encoding of its own: Apple spells `!=` as `==` with the
 * two selected operands exchanged.
 */
bool agx3_alu_fcmp_select_f32_imm(enum agx_isa_encoding enc, uint16_t dest,
                                  uint16_t src0, uint8_t minifloat,
                                  uint16_t src2, uint16_t src3,
                                  const enum agx3_alu_hint hints[4],
                                  unsigned condition, bool dest_cache,
                                  uint16_t wait_mask,
                                  uint8_t out[AGX3_ALU_MAX_LENGTH],
                                  uint8_t *len);

/* Native scalar half comparison selecting either of two scalar halves. */
bool agx3_alu_fcmp_select_half(enum agx_isa_encoding enc, uint16_t dest,
                               const uint16_t src[4],
                               const enum agx3_alu_hint hints[4],
                               unsigned condition, bool dest_cache,
                               uint16_t wait_mask,
                               uint8_t out[AGX3_ALU_MAX_LENGTH],
                               uint8_t *len);

bool agx3_alu_fmin_half(enum agx_isa_encoding enc, uint16_t dest,
                        const uint16_t src[4],
                        const enum agx3_alu_hint hints[4], bool dest_cache,
                        uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                        uint8_t *len);

bool agx3_alu_fmax_half(enum agx_isa_encoding enc, uint16_t dest,
                        const uint16_t src[4],
                        const enum agx3_alu_hint hints[4], bool dest_cache,
                        uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                        uint8_t *len);

/*
 * 32-bit tied fmin/fmax -- the shape NIR produces for max(x, 0.0), and so for
 * normalize(), clamp(), saturate() and discard.  Apple encodes it as a compact
 * six-byte `fsel`, not as the general mixed-width select:
 *
 *   fmax(a, b)     f32   12 03 1e 05 00 c0   (fsel.ep7, register second operand)
 *   fmax(a, 0.0f)  f32   12 03 0a 81 00 c2   (fsel.ep2, minifloat second operand)
 *   fmin           f32   the same, condition 3 instead of 7
 *
 * both witnessed byte-exactly from `metal-nt -arch applegpu_g17p`.  The
 * immediate is an ordinary AGX minifloat, deposited twice because the compared
 * and selected operands are tied.  wait_mask carries at most one scoreboard
 * slot; a wider wait must be retired before the instruction.
 */
bool agx3_alu_fmin_f32(enum agx_isa_encoding enc, uint16_t dest,
                       const uint16_t src[4],
                       const enum agx3_alu_hint hints[4], bool dest_cache,
                       uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                       uint8_t *len);

bool agx3_alu_fmax_f32(enum agx_isa_encoding enc, uint16_t dest,
                       const uint16_t src[4],
                       const enum agx3_alu_hint hints[4], bool dest_cache,
                       uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                       uint8_t *len);

bool agx3_alu_fmin_f32_imm(enum agx_isa_encoding enc, uint16_t dest,
                           uint16_t src, uint8_t minifloat,
                           enum agx3_alu_hint hint, bool dest_cache,
                           uint16_t wait_mask,
                           uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_fmax_f32_imm(enum agx_isa_encoding enc, uint16_t dest,
                           uint16_t src, uint8_t minifloat,
                           enum agx3_alu_hint hint, bool dest_cache,
                           uint16_t wait_mask,
                           uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_hrcp(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                   enum agx3_alu_hint hint, bool dest_cache,
                   uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                   uint8_t *len);

/*
 * Full-precision reciprocal and reciprocal square root.
 *
 * These are the same two Apple forms the half encoders above use; f26 selects
 * the source width (1 = one 16-bit half, 2 = a 32-bit pair) and, for rsqrt,
 * f10 selects the destination's. Only the half widths were wired up, so every
 * fragment shader containing a float divide or a normalize() -- which NIR
 * lowers to frcp / frsq on 32-bit values -- was refused outright.
 *
 * Witnessed byte-exactly against applegpu_g17p:
 *   o[i] = 1.0f / a[i]    af 00 56 00 02 00 10 48 20 00
 *   o[i] = rsqrt(a[i])    af 01 56 00 02 00 b0 40 00 00
 * both with the loaded source dead (f29 = 1) and waiting the load's slot 6.
 */
bool agx3_alu_f32rcp(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                     enum agx3_alu_hint hint, bool dest_cache,
                     uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                     uint8_t *len);

bool agx3_alu_f32rsqrt(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                       enum agx3_alu_hint hint, bool dest_cache,
                       uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                       uint8_t *len);

/*
 * Full-precision base-2 logarithm and exponential.
 *
 * The same two Apple forms as the half encoders below, with both width
 * selectors raised: f10 for the destination and f26 for the source, 1 for one
 * 16-bit half and 2 for a 32-bit pair, exactly as reciprocal square root does
 * it. Only the half widths were wired up, so every 32-bit log2 or exp2 was
 * refused -- and with them every `pow`, which NIR lowers to
 * fexp2(fmul(y, flog2(x))). That is what an sRGB fragment epilog is made of.
 *
 * Witnessed byte-exactly against applegpu_g17p over ten compute kernels:
 *   o[i] = log2(a[i])                      2f 02 56 00 02 00 b0 40 00 00
 *   o[i] = exp2(a[i])                      af 02 56 00 02 00 b0 40 00 00
 *   l = log2(a[i]); o[i] = exp2(l) + l     2f 02 56 04 03 00 92 40 00 00
 *   l = log2(a[i]); o[i] = exp2(l * b[i])  2f 02 56 00 03 00 b0 40 00 00
 *                                          af 02 54 00 02 00 b0 40 00 00
 *
 * One further bit, f30 at bit 63, is deliberately not modelled. Across
 * nineteen witnesses it is set on exactly one shape: the `f32log2` of Metal's
 * `pow(x, y)` built-in, whose result is defined for a negative base with an
 * integral exponent and for a zero base. `powr(x, y)`, which promises a
 * non-negative base, leaves it clear, and so does the hand-written
 * exp2(log2(x) * y) -- which is byte-identical to the built-in's log2 apart
 * from that one bit, and is exactly the shape NIR produces. So the encodings
 * below are Apple's for every shape this compiler can emit; f30 belongs to a
 * domain rule the lowered form does not carry.
 */
bool agx3_alu_f32log2(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                      enum agx3_alu_hint hint, bool dest_cache,
                      uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                      uint8_t *len);

bool agx3_alu_f32exp2(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                      enum agx3_alu_hint hint, bool dest_cache,
                      uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                      uint8_t *len);

/*
 * The same two forms with the grounded source modifiers: f31 negates the
 * source, f30 takes its absolute value. Both are witnessed as single-field
 * differences against the plain forms above, on applegpu_g17p:
 *
 *   o[i] = exp2(a[i])        af 02 56 00 02 00 b0 40 00 00
 *   o[i] = exp2(-a[i])       af 02 56 00 02 00 b0 40 01 00   f31
 *   o[i] = exp2(fabs(a[i]))  af 02 56 00 02 00 b0 c0 00 00   f30
 *   o[i] = exp2(-fabs(a))    af 02 56 00 02 00 b0 c0 01 00   both
 *   o[i] = log2(-a[i])       2f 02 56 00 02 00 b0 40 01 00
 *   o[i] = rsqrt(-a[i])      af 01 56 00 02 00 b0 40 01 00
 *   o[i] = floor(-a[i])      2f 00 56 00 02 00 b0 40 03 00
 *
 * The half forms carry the same field: `half o[i] = exp2(-a[i])` is
 * af 02 56 02 02 04 ac 60 01 00, the plain half form with f31 set.
 *
 * This is also what the f30 note above was missing. Metal's pow(x, y) is
 * defined for a negative base with an integral exponent, so its lowering
 * takes log2 of |x| -- f30 -- while powr() and the hand-written
 * exp2(log2(x) * y) do not. f30 is the absolute-value modifier, not a domain
 * flag.
 *
 * NIR folds a negation into a float source, so `exp(-x)` -- vkQuake's fog
 * term, in every one of its fragment shaders -- reaches the packer as an
 * exp2 with a negated source and refused the whole shader.
 */
bool agx3_alu_f32log2_mod(enum agx_isa_encoding enc, uint16_t dest,
                          uint16_t src, enum agx3_alu_hint hint,
                          bool dest_cache, uint16_t wait_mask, bool src_neg,
                          bool src_abs, uint8_t out[AGX3_ALU_MAX_LENGTH],
                          uint8_t *len);

bool agx3_alu_f32exp2_mod(enum agx_isa_encoding enc, uint16_t dest,
                          uint16_t src, enum agx3_alu_hint hint,
                          bool dest_cache, uint16_t wait_mask, bool src_neg,
                          bool src_abs, uint8_t out[AGX3_ALU_MAX_LENGTH],
                          uint8_t *len);

/*
 * Full-precision floor and ceiling. Apple names the instruction f32rint and
 * selects the rounding in f43; the width selectors f10 and f26 are 1 for a
 * scalar half and 2 for a 32-bit pair, the same relationship the 32-bit
 * logarithm has to the half one. Only the half widths were wired up, so
 * floor() on a float -- vkQuake's quantized lightmap coordinate -- refused
 * the shader.
 *
 * Witnessed byte-exactly against applegpu_g17p, one kernel per mode:
 *   o[i] = rint(a[i])    2f 00 56 00 02 00 b0 40 00 00   f43 absent
 *   o[i] = trunc(a[i])   2f 00 56 00 02 00 b0 40 06 00   f43 = 1
 *   o[i] = ceil(a[i])    2f 00 56 00 02 00 b0 40 04 00   f43 = 2
 *   o[i] = floor(a[i])   2f 00 56 00 02 00 b0 40 02 00   f43 = 3
 * and, for the lifetime and wait fields this form shares with the family,
 *   float x = a[i]; o[i] = floor(x) + x
 *                        2f 00 56 04 03 00 92 40 02 00   f27 retained, f11
 *   o[i] = floor(a[i]) * 4 + floor(b[i])
 *                        2f 10 54 04 03 08 b0 40 02 00   f35, wait slot 1
 */
bool agx3_alu_f32floor(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                       enum agx3_alu_hint hint, bool dest_cache,
                       uint16_t wait_mask, bool src_neg, bool src_abs,
                       uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_f32ceil(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                      enum agx3_alu_hint hint, bool dest_cache,
                      uint16_t wait_mask, bool src_neg, bool src_abs,
                      uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_fcmp_select_f32_f16(enum agx_isa_encoding enc, uint16_t dest,
                                  const uint16_t src[4],
                                  const enum agx3_alu_hint hints[4],
                                  unsigned condition, bool dest_cache,
                                  uint16_t wait_mask,
                                  uint8_t out[AGX3_ALU_MAX_LENGTH],
                                  uint8_t *len);

/* Native scalar half reciprocal square root. */
bool agx3_alu_hrsqrt(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                     enum agx3_alu_hint hint, bool dest_cache,
                     uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                     uint8_t *len);

/* Native scalar half base-2 exponential. */
bool agx3_alu_hexp2(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                    enum agx3_alu_hint hint, bool dest_cache,
                    uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                    uint8_t *len);

/* Native scalar half base-2 logarithm. */
bool agx3_alu_hlog2(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                    enum agx3_alu_hint hint, bool dest_cache,
                    uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                    uint8_t *len);

/* Native scalar half floor. */
bool agx3_alu_hfloor(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                     enum agx3_alu_hint hint, bool dest_cache,
                     uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                     uint8_t *len);

/* Native scalar half ceiling. */
bool agx3_alu_hceil(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                    enum agx3_alu_hint hint, bool dest_cache,
                    uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                    uint8_t *len);

/* Native square-root step, multiplied by the input by common Mesa lowering. */
bool agx3_alu_sqrt_step(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                       enum agx3_alu_hint hint, bool dest_cache, uint16_t wait_mask,
                       uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native square-root step with a scalar half source and float32 temporary. */
bool agx3_alu_sqrt_step_half(enum agx_isa_encoding enc, uint16_t dest,
                             uint16_t src, enum agx3_alu_hint hint,
                             bool dest_cache, uint16_t wait_mask,
                             uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_fmul_narrow(enum agx_isa_encoding enc, uint16_t dest,
                           const uint16_t src[2], const enum agx3_alu_hint hints[2],
                           bool dest_cache, uint16_t wait_mask,
                           uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_hmul(enum agx_isa_encoding enc, uint16_t dest,
                   const uint16_t src[2], const enum agx3_alu_hint hints[2],
                   bool dest_cache, uint16_t wait_mask,
                   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native scalar half register multiplied by an 8-bit minifloat. */
bool agx3_alu_hmul_minifloat(enum agx_isa_encoding enc, uint16_t dest,
                             uint16_t src, uint16_t immediate,
                             enum agx3_alu_hint hint, bool dest_cache,
                             uint16_t wait_mask,
                             uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_hadd(enum agx_isa_encoding enc, uint16_t dest,
                   const uint16_t src[2], const enum agx3_alu_hint hints[2],
                   bool dest_cache, uint16_t wait_mask,
                   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_hfma(enum agx_isa_encoding enc, uint16_t dest,
                   const uint16_t src[3], const enum agx3_alu_hint hints[3],
                   bool product_neg, bool src2_neg, bool dest_cache,
                   uint16_t wait_mask,
                   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_widen_half(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                         enum agx3_alu_hint hint, bool src_abs, bool src_neg,
                         bool dest_cache, uint16_t wait_mask,
                         uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

#ifdef __cplusplus
}
#endif
