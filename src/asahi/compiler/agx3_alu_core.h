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

/* Native G17P population count of a scalar ushort or uint, with post-RA
 * source lifetime and producer waits. */
bool agx3_alu_popcount(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                       bool half, enum agx3_alu_hint hint, bool dest_cache,
                       uint16_t wait_mask,
                       uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Whether agx3_alu_compose()'s form for `op` marks source `i` (in
 * f_src0/f_src1/f_src2 order) dead. */
bool agx3_alu_form_ends_source(enum agx_isa_encoding enc, enum agx3_alu_op op,
                               unsigned i);

/* G17P 32-bit AND/OR/XOR with actual post-RA operand hints. */
bool agx3_alu_bit_reg(enum agx_isa_encoding enc, enum agx3_alu_op op,
                      uint16_t dest, const uint16_t src[2],
                      const enum agx3_alu_hint hints[2], uint16_t wait_mask,
                      uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* `reg & reg` into reg: a no-op carrying a scoreboard wait mask, which keeps
 * the register live whatever it holds. */
bool agx3_alu_band_identity_wait(enum agx_isa_encoding enc, uint16_t reg,
                                 uint16_t wait_mask,
                                 uint8_t out[AGX3_ALU_MAX_LENGTH],
                                 uint8_t *len);

bool agx3_alu_bit_reg16(enum agx_isa_encoding enc, enum agx3_alu_op op,
                          uint16_t dest, const uint16_t src[2],
                          const enum agx3_alu_hint hints[2], uint16_t wait_mask,
                          uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/*
 * The same AND/OR/XOR with the second operand an IMMEDIATE. The operand kind
 * is a field, f37, on the very encoding point the register form already uses,
 * with the constant in f29 -- so `x & 0xff` is not a different instruction, it
 * is the register form with one field changed. Witnessed with
 * metal-nt -arch applegpu_g17p and recovered through tools/decomp.c:
 *
 *   a & c      0b051f01000000800000   f16=4 f37=2 f38=2
 *   a & 0xff   0b011fff100200800000   f29=255 f37=1 f38=2
 *   a | 0xff   0b011fff120a00800000   bit_or, same fields
 *   a ^ 0xff   0b011eff120a00800000   bit_xor, same fields
 *   a & 0x7f   0b010fff100200800000   f29=127
 *   a ^ 1      0b010e81120a00800000   f29=1
 *   (ushort) a & 0xff   13021fff100200800000   f10=1 f24=1, f38 still 2
 *   (ushort) a | 0x80   13021f81020a00800000   f29=128
 *
 * The narrow rows are what settle f38: a 16-bit result narrows f10 and f24 but
 * NOT f38, because an immediate operand declares width 2 whatever the
 * destination is -- the same rule the compare-select's immediate slot follows.
 */
bool agx3_alu_bit_imm(enum agx_isa_encoding enc, enum agx3_alu_op op,
                      bool narrow, uint16_t dest, uint16_t src,
                      uint8_t immediate, enum agx3_alu_hint hint,
                      uint16_t wait_mask,
                      uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* G17P 16-bit ~a, Apple's unary `inv` at width 1. The destination cache is
 * f11, the source is f16 with its cache/discard hints in f25/f27, and
 * f30..f35 wait on producer slots 1..6. Witnessed with metal-nt -arch
 * applegpu_g17p on `ushort m = ~s` fragments and kernels:
 *
 *   03002e00010800000000   inv $h0, ^h0
 *   03002e00012800000000   inv $h0, ^h0  waiting on slot 1
 *   03802600052800000000   inv $h1, $h0  waiting on slot 1
 *   03822600050800000000   inv $h1, $h2
 *   13802600050800000000   inv $h3, $h0
 *   13022e00010800800000   inv $h2, ^h2  read twice, waiting on slot 6
 *   13020e00010800800000   inv h2, ^h2   stored once, waiting on slot 6
 *   13020600050800800000   inv h3, h2    waiting on slot 6
 */
bool agx3_alu_bit_not_reg16(enum agx_isa_encoding enc, uint16_t dest,
                            uint16_t src, enum agx3_alu_hint hint,
                            bool dest_cache, uint16_t wait_mask,
                            uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* G17P 32-bit a & ~b.  Apple exposes this truth-table shape as bit_andc;
 * unlike AND it is not commutative, so the source order is preserved. */
bool agx3_alu_bit_andc(enum agx_isa_encoding enc, uint16_t dest,
                      const uint16_t src[2],
                      const enum agx3_alu_hint hints[2], bool dest_cache,
                      uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                      uint8_t *len);

/* G17P 32-bit ~(a | b), Apple's bit_nor. */
bool agx3_alu_bit_nor(enum agx_isa_encoding enc, uint16_t dest,
                      const uint16_t src[2],
                      const enum agx3_alu_hint hints[2], bool dest_cache,
                      uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                      uint8_t *len);

/* G17P 32-bit ~(a & b), ~(a ^ b) and a | ~b: Apple's bit_nand, bit_xnor and
 * bit_orc. bit_orc keeps its source order. */
bool agx3_alu_bit_nand(enum agx_isa_encoding enc, uint16_t dest,
                       const uint16_t src[2],
                       const enum agx3_alu_hint hints[2], bool dest_cache,
                       uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                       uint8_t *len);
bool agx3_alu_bit_xnor(enum agx_isa_encoding enc, uint16_t dest,
                       const uint16_t src[2],
                       const enum agx3_alu_hint hints[2], bool dest_cache,
                       uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                       uint8_t *len);
bool agx3_alu_bit_orc(enum agx_isa_encoding enc, uint16_t dest,
                      const uint16_t src[2],
                      const enum agx3_alu_hint hints[2], bool dest_cache,
                      uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                      uint8_t *len);

bool agx3_alu_shift_add16(enum agx_isa_encoding enc, uint16_t dest,
                            const uint16_t src[2],
                            const enum agx3_alu_hint hints[2], unsigned shift,
                            bool dest_cache, uint16_t wait_mask,
                            uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native scalar-half saturating register addition, signed or unsigned. */
bool agx3_alu_add_sat16(enum agx_isa_encoding enc, uint16_t dest,
                        const uint16_t src[2],
                        const enum agx3_alu_hint hints[2], bool is_signed,
                        bool dest_cache, uint16_t wait_mask,
                        uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native scalar-half register subtraction, src[0] - src[1]. */
bool agx3_alu_sub16(enum agx_isa_encoding enc, uint16_t dest,
                    const uint16_t src[2], const enum agx3_alu_hint hints[2],
                    bool dest_cache, uint16_t wait_mask,
                    uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native scalar-half integer negation, 0 - src. */
bool agx3_alu_neg16(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                    enum agx3_alu_hint hint, bool dest_cache,
                    uint16_t wait_mask,
                    uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* G17P logical shift-left with register value and register shift count. */
bool agx3_alu_sll_reg(enum agx_isa_encoding enc, uint16_t dest, bool dest_16,
                      bool dest_cache, const uint16_t src[2],
                      const bool src_16[2],
                      const enum agx3_alu_hint hints[2], uint16_t wait_mask,
                      uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native scalar-half `1 << uint_register` form. */
bool agx3_alu_sll_one_reg32(enum agx_isa_encoding enc, uint16_t dest,
                            uint16_t shift, enum agx3_alu_hint shift_hint,
                            bool dest_cache, uint16_t wait_mask,
                            uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* G17P logical shift-right with register value and signed boundary. */
bool agx3_alu_srl_reg(enum agx_isa_encoding enc, uint16_t dest, bool dest_16,
                      bool dest_cache, const uint16_t src[2],
                      const bool src_16[2],
                      const enum agx3_alu_hint hints[2], uint16_t wait_mask,
                      uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native signed 16-bit arithmetic shift-right by a 32-bit register count. */
bool agx3_alu_asr_reg16(enum agx_isa_encoding enc, uint16_t dest,
                        uint16_t source, uint16_t shift,
                        const enum agx3_alu_hint hints[2], bool dest_cache,
                        uint16_t wait_mask,
                        uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native 16-bit logical shift-right by an immediate bit count. */
bool agx3_alu_srl_imm16(enum agx_isa_encoding enc, uint16_t dest,
                        uint16_t src, unsigned shift,
                        enum agx3_alu_hint hint, bool dest_cache,
                        uint16_t wait_mask,
                        uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* G17P unshifted IADD/ISUB, preserving the canonical operation form. */
bool agx3_alu_integer_reg(enum agx_isa_encoding enc, enum agx3_alu_op op,
                          uint16_t dest, const uint16_t src[2],
                          const enum agx3_alu_hint hints[2], uint16_t wait_mask,
                          uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* The same, with slot B shifted left by `shift` first (f54). Witnessed for
 * ISUB at shifts 1..4; see agx3_alu_core.c for the byte ladder. */
bool agx3_alu_integer_reg_shift(enum agx_isa_encoding enc, enum agx3_alu_op op,
                                uint16_t dest, const uint16_t src[2],
                                const enum agx3_alu_hint hints[2],
                                uint8_t shift, uint16_t wait_mask,
                                uint8_t out[AGX3_ALU_MAX_LENGTH],
                                uint8_t *len);

/* Native signed/unsigned32 to float32 with post-RA source hints. */
bool agx3_alu_int_to_float(enum agx_isa_encoding enc, uint16_t dest,
                            uint16_t src, bool is_signed,
                            enum agx3_alu_hint hint, uint16_t wait_mask,
                            uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native signed/unsigned f32-to-i32 with post-RA source and destination cache
 * state. */
bool agx3_alu_float_to_int32(enum agx_isa_encoding enc, uint16_t dest,
                             uint16_t src, bool is_signed,
                             enum agx3_alu_hint hint, bool dest_cache,
                             uint16_t wait_mask,
                             uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native int8 or int16 to scalar-half or float32 conversion. A byte source
 * reads only the low eight bits of its half; unsigned8 converts only to
 * float32. */
bool agx3_alu_narrow_int_to_float(enum agx_isa_encoding enc, uint16_t dest,
                                  bool dest_16, uint16_t src, bool src_8,
                                  bool is_signed, enum agx3_alu_hint hint,
                                  bool dest_cache, uint16_t wait_mask,
                                  uint8_t out[AGX3_ALU_MAX_LENGTH],
                                  uint8_t *len);

/* Native signed/unsigned32 to scalar-half conversion. */
bool agx3_alu_i32_to_f16(enum agx_isa_encoding enc, uint16_t dest,
                         uint16_t src, bool is_signed,
                         enum agx3_alu_hint hint, bool dest_cache,
                         uint16_t wait_mask,
                         uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native scalar float16 to signed/unsigned32 conversion, RTZ. */
bool agx3_alu_f16_to_int32(enum agx_isa_encoding enc, uint16_t dest,
                           uint16_t src, bool is_signed,
                           enum agx3_alu_hint hint, bool dest_cache,
                           uint16_t wait_mask,
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

/*
 * The general compare-select.
 *
 * `fcmpsel` is four operands -- two compared, two selected -- and each carries
 * an independent (kind, width, value) triple. Selected values use kind 1 for
 * an immediate, kind 2 for a GPR, and kind 3 for a uniform. Fresh G17P
 * select-kind.metal probes change only GPR versus constant-buffer alternatives
 * in fabs(f32) < f16 ? a : b. Kind 3 was previously mislabeled as a register,
 * causing the trig small-input guard to read pointer words from uniforms.
 *
 * Field map, recovered from the 84 real Apple `fsel` instructions in
 * test/agx3_compose_corpus_vectors.c:
 *
 *   dest          f3    width f11   cache f12
 *   compare 0     f35   kind  f42   width f43   cache f44  last-use f46  ABS f47
 *   compare 1     f52 (register) or f51 (immediate)
 *                       kind  f59   width f60   cache f61  last-use f63  ABS f64
 *   select true   f69 (GPR), f70 (uniform), f68 (immediate)  kind f76 width f77
 *   select false  f86 (GPR), f87 (uniform), f85 (immediate)  kind f93 width f94
 *   condition f110, scoreboard f107
 *
 * Absolute sources move the instruction to a longer endpoint exactly as they
 * do elsewhere in the family, so the endpoint range is scanned rather than
 * computed; the endpoints are mutually exclusive on their fixed bits.
 *
 * GPR alternatives carry cache/discard fields f78/f80 and f95/f97. Uniform
 * alternatives have no GPR lifetime hints.
 */
enum agx3_fsel_kind {
   /* compare slots */
   AGX3_FSEL_CMP_IMMEDIATE = 1,
   AGX3_FSEL_CMP_REGISTER = 2,
   /* select slots */
   AGX3_FSEL_SEL_IMMEDIATE = 1,
   AGX3_FSEL_SEL_REGISTER = 2,
   AGX3_FSEL_SEL_UNIFORM = 3,
};

struct agx3_fsel_desc {
   uint16_t dest;
   bool dest_16, dest_cache;

   uint16_t cmp0;               /* register half; slot 0 is always a register */
   enum agx3_alu_hint cmp0_hint;
   bool cmp0_abs;
   bool cmp0_neg;
   bool cmp0_16;

   bool cmp1_is_immediate;
   uint16_t cmp1;               /* register half, or the immediate value */
   enum agx3_alu_hint cmp1_hint;
   bool cmp1_abs;
   bool cmp1_neg;
   bool cmp1_16;                /* register sources only; an immediate slot
                                 * always declares width 2 */

   uint8_t sel0_kind, sel1_kind; /* agx3_fsel_kind, select-slot numbering */
   uint16_t sel0, sel1;
   enum agx3_alu_hint sel0_hint, sel1_hint;
   bool sel_16;

   unsigned condition;
   uint16_t wait_mask;
};

bool agx3_alu_fcmp_select(enum agx_isa_encoding enc,
                          const struct agx3_fsel_desc *d,
                          uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* The same select with an absolute value on either comparison source. ABS is
 * not a bit on the form above: it moves to a longer endpoint (ten bytes ->
 * fourteen) whose integer-select destination block is the same f3/f11/f12.
 * f47 and f64 are the two sources' bits and are separable. See
 * agx3_alu_core.c for the witnessed byte ladder. */
bool agx3_alu_fcmp_select_imm_width_abs(
   enum agx_isa_encoding enc, uint16_t dest, const uint16_t src[2],
   const bool src_16[2], const enum agx3_alu_hint hints[2],
   unsigned condition, uint8_t if_true, uint8_t if_false, bool dest_16,
   bool dest_cache, uint16_t wait_mask, const bool src_abs[2],
   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* G17P `b > -a` and its relatives. f65 negates the second full-width
 * comparator and composes with every relation, both result widths and any
 * pair of selected immediates; see agx3_alu_core.c for the byte ladder. */
bool agx3_alu_fcmp_select_imm_width_neg_second(
   enum agx_isa_encoding enc, uint16_t dest, const uint16_t src[2],
   const bool src_16[2], const enum agx3_alu_hint hints[2],
   unsigned condition, uint8_t if_true, uint8_t if_false, bool dest_16,
   bool dest_cache, uint16_t wait_mask,
   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* The same, with the negate on either comparator. There is no bit for a
 * negated FIRST source: Apple spells `-a > b` as `b < -a`, so this swaps the
 * operands and reverses the relation rather than refusing. Negating both is
 * the relation flip alone. */
bool agx3_alu_fcmp_select_imm_width_neg(
   enum agx_isa_encoding enc, uint16_t dest, const uint16_t src[2],
   const bool src_16[2], const enum agx3_alu_hint hints[2],
   const bool neg[2], unsigned condition, uint8_t if_true, uint8_t if_false,
   bool dest_16, bool dest_cache, uint16_t wait_mask,
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
 * byte-immediate result shapes used by Hyprland and GTK shaders. The
 * half-result form carries any one of the six native scoreboard waits. */
bool agx3_alu_fcmp_minifloat_select_mixed(
   enum agx_isa_encoding enc, uint16_t dest, uint16_t compare,
   uint8_t minifloat, const uint16_t selected[2],
   const bool selected_immediate[2], const enum agx3_alu_hint hints[3],
   unsigned condition, bool dest_16, bool dest_cache, uint16_t wait_mask,
   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Compare a scalar half against a minifloat and select f32 values, with at
 * most one byte-immediate alternative. */
bool agx3_alu_fcmp_half_minifloat_select_f32(
   enum agx_isa_encoding enc, uint16_t dest, uint16_t compare,
   uint8_t minifloat, const uint16_t selected[2],
   const bool selected_immediate[2], const enum agx3_alu_hint hints[3],
   unsigned condition, bool dest_cache, uint16_t wait_mask,
   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Plain f32 register FADD/FMUL/FMA with actual post-RA source hints and
 * scheduler waits. Compact witnessed forms only; unsupported wait combinations
 * return false so the caller can retire them explicitly before retrying. */
bool agx3_alu_float_reg(enum agx_isa_encoding enc, enum agx3_alu_op op,
                         uint16_t dest, const uint16_t src[3],
                         const enum agx3_alu_hint hints[3], uint16_t wait_mask,
                         uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Write a scalar f32 add directly to the texture staging file.  Native
 * fragment shaders use this destination class for the last coordinate
 * arithmetic before an implicit sample; a later GPR-to-staging COPY loses
 * helper-lane derivative provenance at an odd hardware-scissor edge. */
bool agx3_alu_fadd_stage(enum agx_isa_encoding enc, uint16_t staging_dest,
                         uint8_t staging_slot, const uint16_t src[2],
                         const enum agx3_alu_hint hints[2],
                         uint16_t wait_mask,
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
                                uint16_t addend, bool negate_addend, bool saturate,
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
                          bool dest_cache, uint16_t wait_mask,
                          uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native ABS(f32)-by-scalar-half multiply with an f32 result. */
bool agx3_alu_fmul_abs_mixed(
   enum agx_isa_encoding enc, uint16_t dest, uint16_t f32, uint16_t f16,
   enum agx3_alu_hint hint32, enum agx3_alu_hint hint16, bool dest_cache,
   uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native unmodified f32 plus scalar-half register. */
bool agx3_alu_fadd_mixed_half(
   enum agx_isa_encoding enc, uint16_t dest, uint16_t f32, uint16_t f16,
   enum agx3_alu_hint hint32, enum agx3_alu_hint hint16, bool dest_cache,
   uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native f32 ABS(register) plus scalar-half register. */
bool agx3_alu_fadd_abs_mixed_half(
   enum agx_isa_encoding enc, uint16_t dest, uint16_t f32, uint16_t f16,
   enum agx3_alu_hint hint32, enum agx3_alu_hint hint16, bool dest_cache,
   uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

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
/* RTZ float-to-16-bit-integer conversion. The source width (f24) and the
 * signed/unsigned mode (f41) are separate fields from the 16-bit result width
 * (f10), so a HALF source is the same instruction with one field changed; see
 * agx3_alu_core.c for the kernel pairs. */
bool agx3_alu_float_to_int16(enum agx_isa_encoding enc, uint16_t dest,
                        uint16_t src, bool src_16, bool is_signed,
                        enum agx3_alu_hint hint, bool dest_cache,
                        uint16_t wait_mask,
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
 * Destination retention and the GPR's cache/discard state are independent.
 * The six-byte form carries no wait, one of six waits, or slots1+2 together.
 * Other wait combinations, operand files/widths and AGX3c are not admitted.
 */
bool agx3_alu_fadd_minifloat(enum agx_isa_encoding enc, uint16_t dest,
                            uint16_t src, uint16_t immediate, bool dest_cache,
                            enum agx3_alu_hint hint, uint16_t wait_mask,
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

/* Long G17P add form with an absolute register and minifloat operand. */
bool agx3_alu_fadd_abs_minifloat(
   enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
   uint16_t immediate, bool dest_cache, bool negate_src, bool saturate,
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

/* Full grounded G17P minifloat multiply form, including destination
 * retention. `src_abs` is retained as a fail-closed API guard but is always
 * rejected; lower ABS through fadd_abs_negzero followed by a plain FMUL. */
bool agx3_alu_fmul_minifloat_mod(
   enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
   uint16_t immediate, bool dest_cache, bool src_abs,
   enum agx3_alu_hint hint, uint16_t wait_mask,
   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/*
 * The same multiply with an independently selected destination width: f10
 * chooses a scalar half (1) or a float (2) result while f26 keeps the source
 * at 32 bits. `packHalf2x16` narrows a 32-bit product to a half, and the
 * optimizer folds that f2f16 into the multiply, so this mixed-width form is
 * what actually reaches the packer. See the witnesses in agx3_alu_core.c.
 *
 * An absolute source is refused for either destination width. The six-byte
 * endpoint has no grounded fused ABS form; use explicit FABS plus plain FMUL.
 */
bool agx3_alu_fmul_minifloat_narrow(
   enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
   uint16_t immediate, bool dest_cache, bool src_abs, bool dest_half,
   enum agx3_alu_hint hint, uint16_t wait_mask,
   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native scalar-half times minifloat with a float32 result. */
bool agx3_alu_fmul_half_minifloat_to_f32(
   enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
   uint16_t immediate, enum agx3_alu_hint hint, uint16_t wait_mask,
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

/* Signed right shifts by an immediate, with native source lifetime/waits. */
bool agx3_alu_asr_imm32(enum agx_isa_encoding enc, uint16_t dest, uint16_t source,
                        unsigned shift, enum agx3_alu_hint hint, bool dest_cache,
                        uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                        uint8_t *length);
bool agx3_alu_asr_imm16(enum agx_isa_encoding enc, uint16_t dest, uint16_t source,
                        unsigned shift, enum agx3_alu_hint hint, bool dest_cache,
                        uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                        uint8_t *length);

/*
 * G17P register/immediate FSEL. Absolute compare modifiers are grounded for
 * f32 eq/lt/gt, plus GE/LE with exactly one absolute comparator, with f32
 * selected values.
 *
 * `compare_neg` carries a negate on EITHER comparator. f65 is the only negate
 * field the encoding has and it names the second comparator; every other
 * placement is rewritten inside this function rather than refused -- a negated
 * first comparator exchanges the operands and reverses the relation (Apple's
 * own canonicalisation, observed), negating both is the relation flip alone,
 * a negate against the exact immediate zero is a no-op because -0.0 and +0.0
 * compare equal, and a modifier on a minifloat comparator folds into the
 * constant's sign bit. abs and neg compose on the same instruction. The byte
 * ladder for all of it is in agx3_alu_core.c.
 *
 * This function absorbed agx3_alu_fcmp_select_reg_imm_neg(), which was a
 * second entry point for the f65 half of the same encoding; there is one
 * encoder now, not two.
 */
bool agx3_alu_fcmp_select_reg_imm(enum agx_isa_encoding enc, uint16_t dest,
                                  const uint16_t src[4], bool compare_zero,
                                  const bool compare_abs[2],
                                  const bool compare_neg[2],
                                  const bool immediate[2],
                                  const enum agx3_alu_hint hints[4], unsigned condition,
                                  bool dest_16, bool dest_cache, uint16_t wait_mask,
                                  uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/*
 * Compare a 32-bit float against a minifloat immediate and select either of
 * two 32-bit registers -- what `x < k ? a : b` becomes once Mesa carries the
 * constant as a minifloat. The compared register may carry fabs; that native
 * form uses the long endpoint. Conditions are enum agx_fcond; eq, lt, gt, ge
 * and le are witnessed, ltn and gtn are not and are refused. An inverted
 * condition needs no encoding of its own: Apple spells `!=` as `==` with the
 * two selected operands exchanged.
 */
bool agx3_alu_fcmp_select_f32_imm(enum agx_isa_encoding enc, uint16_t dest,
                                  uint16_t src0, uint8_t minifloat,
                                  bool src0_abs,
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

/* Native reciprocal with independently selected result/source widths and
 * source modifiers. Both 16- and 32-bit registers use the same f32rcp form. */
bool agx3_alu_rcp(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                  bool dest_half, bool src_half, bool src_abs, bool src_neg,
                  enum agx3_alu_hint hint, bool dest_cache,
                  uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                  uint8_t *len);

bool agx3_alu_hrcp(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                   enum agx3_alu_hint hint, bool dest_cache,
                   uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                   uint8_t *len);

/*
 * Full-precision reciprocal and reciprocal square root.
 *
 * These are the same two Apple forms the half encoders above use; f26 selects
 * the source width (1 = one 16-bit half, 2 = a 32-bit pair) and f10 selects
 * the destination's. Only the half widths were wired up, so every
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
 * Reciprocal square root carrying the source's ABS (f30) and NEG (f31)
 * modifiers, the same two bits f32log2/f32exp2/f32rcp already model on their
 * own forms. `inversesqrt(abs(x))` is the standard safe form of the idiom and
 * reached the packer as `rsqrt [s0=32:reg.abs]`, which had no encoding at all.
 *
 * Witnessed with `metal-nt -arch applegpu_g17p`, one kernel per modifier,
 * each a single-field difference from the plain control and every vector
 * recovered by re-composing through this encoder (tools/decomp.c):
 *
 *   o[i] = rsqrt(a[i])         af 01 56 00 02 00 b0 40 00 00
 *   o[i] = rsqrt(-a[i])        af 01 56 00 02 00 b0 40 01 00   f31
 *   o[i] = rsqrt(fabs(a[i]))   af 01 56 00 02 00 b0 c0 00 00   f30
 *   o[i] = rsqrt(-fabs(a[i]))  af 01 56 00 02 00 b0 c0 01 00   both
 *
 * The two are separable exactly as they are on exp2: `-fabs` sets both.
 */
bool agx3_alu_f32rsqrt_mod(enum agx_isa_encoding enc, uint16_t dest,
                           uint16_t src, enum agx3_alu_hint hint,
                           bool dest_cache, uint16_t wait_mask, bool src_neg,
                           bool src_abs, uint8_t out[AGX3_ALU_MAX_LENGTH],
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

/* Native logarithm with independently selected scalar result/source widths.
 * Any half-width form is currently grounded for no wait or slot 6. */
bool agx3_alu_log2(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                   bool dest_half, bool src_half, bool src_abs, bool src_neg,
                   enum agx3_alu_hint hint, bool dest_cache,
                   uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
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

/*
 * Round-to-nearest-even is the same form with f43 absent, the first line of
 * the witness table above.
 *
 * The select core also encodes a 32-bit roundeven, through
 * agx3_pack_f32_rint_reg(), but that encoding point deposits its source in a
 * six-bit field and so refuses any source above r63. Register allocation puts
 * one there in any shader with real pressure, and the refusal is fatal: the
 * whole shader is rejected and pipeline creation fails. This form carries the
 * family's full eight-bit register fields, its source modifiers and its wait
 * slots, so it covers the register file.
 */
bool agx3_alu_f32roundeven(enum agx_isa_encoding enc, uint16_t dest,
                           uint16_t src, enum agx3_alu_hint hint,
                           bool dest_cache, uint16_t wait_mask, bool src_neg,
                           bool src_abs, uint8_t out[AGX3_ALU_MAX_LENGTH],
                           uint8_t *len);

bool agx3_alu_fcmp_select_f32_f16(enum agx_isa_encoding enc, uint16_t dest,
                                  const uint16_t src[4],
                                  const enum agx3_alu_hint hints[4],
                                  unsigned condition, bool dest_cache,
                                  uint16_t wait_mask,
                                  uint8_t out[AGX3_ALU_MAX_LENGTH],
                                  uint8_t *len);

bool agx3_alu_fcmp_select_f16_f16_f32(
   enum agx_isa_encoding enc, uint16_t dest, const uint16_t src[4],
   const enum agx3_alu_hint hints[4], const bool src_abs[2],
   unsigned condition, bool dest_cache, uint16_t wait_mask,
   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

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

/* Native scalar-half base-2 exponential with the recovered source negate. */
bool agx3_alu_hexp2_neg(enum agx_isa_encoding enc, uint16_t dest,
                        uint16_t src, enum agx3_alu_hint hint,
                        bool dest_cache, uint16_t wait_mask, bool src_neg,
                        uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

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

/* Native scalar half round-to-nearest-even and truncate. */
bool agx3_alu_hroundeven(enum agx_isa_encoding enc, uint16_t dest,
                         uint16_t src, enum agx3_alu_hint hint,
                         bool dest_cache, uint16_t wait_mask,
                         uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);
bool agx3_alu_htrunc(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                     enum agx3_alu_hint hint, bool dest_cache,
                     uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                     uint8_t *len);

/* Native square-root step, multiplied by the input by common Mesa lowering. */
bool agx3_alu_sqrt_step(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                       enum agx3_alu_hint hint, bool dest_cache, uint16_t wait_mask,
                       uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/*
 * The same step with the source's absolute value folded in (f30), which is
 * what `sqrt(abs(x))` -- the standard safe-sqrt idiom -- lowers to. Witnessed
 * against Apple's own kernels, a single-field difference from the control:
 *
 *   o[i] = sqrt(a[i])          2f 01 56 04 03 00 92 40 00 00
 *   o[i] = sqrt(fabs(a[i]))    2f 01 56 04 03 00 92 c0 00 00   f30
 *
 * NEG is deliberately not exposed here: sqrt of a negative is NaN, no Apple
 * witness exercises it, and an unwitnessed bit is worse than a refusal.
 */
bool agx3_alu_sqrt_step_mod(enum agx_isa_encoding enc, uint16_t dest,
                            uint16_t src, enum agx3_alu_hint hint,
                            bool dest_cache, uint16_t wait_mask, bool src_abs,
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
bool agx3_alu_hmul_abs(enum agx_isa_encoding enc, uint16_t dest,
                       const uint16_t src[2],
                       const enum agx3_alu_hint hints[2],
                       const bool src_abs[2], bool dest_cache,
                       uint16_t wait_mask,
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

/* Saturating scalar-half multiply and add: sat(a * b) and sat(a + b).
 * G17P spells the saturate as f14, which the compact ep0/ep1 points cannot
 * carry, so these land on the two-byte-longer ep2 form. Apple witnesses:
 *   saturate(a[i] * b[i])   10 03 1d 02 01 00 00 82  (vs 10 03 1d 02 00 c0)
 *   saturate(a[i] + b[i])   10 03 1c 02 01 00 00 82  (vs 10 03 1c 02 00 c0)
 * Their field vectors differ from the unsaturated ones in f14 alone. */
bool agx3_alu_hmul_saturate(enum agx_isa_encoding enc, uint16_t dest,
                            const uint16_t src[2],
                            const enum agx3_alu_hint hints[2],
                            bool dest_cache, uint16_t wait_mask,
                            uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);
bool agx3_alu_hadd_saturate(enum agx_isa_encoding enc, uint16_t dest,
                            const uint16_t src[2],
                            const enum agx3_alu_hint hints[2],
                            bool dest_cache, uint16_t wait_mask,
                            uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native scalar-half saturate: sat(src + -0.0h). */
bool agx3_alu_hadd_saturate_negzero(
   enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
   enum agx3_alu_hint hint, bool dest_cache, uint16_t wait_mask,
   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native -src + -0.0h used by half negate and matrix inverse lowering. */
bool agx3_alu_hadd_neg_negzero(
   enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
   enum agx3_alu_hint hint, bool dest_cache, uint16_t wait_mask,
   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native scalar-half inverse: 1.0h + -src. */
bool agx3_alu_hadd_one_neg(enum agx_isa_encoding enc, uint16_t dest,
                           uint16_t src, enum agx3_alu_hint hint,
                           bool dest_cache, uint16_t wait_mask,
                           uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_hfma(enum agx_isa_encoding enc, uint16_t dest,
                   const uint16_t src[3], const enum agx3_alu_hint hints[3],
                   bool product_neg, bool src2_neg, bool dest_cache,
                   uint16_t wait_mask,
                   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native scalar half FMA with an 8-bit minifloat addend. */
bool agx3_alu_hfma_addend_minifloat(
   enum agx_isa_encoding enc, uint16_t dest, const uint16_t src[2],
   uint16_t immediate, const enum agx3_alu_hint hints[2], bool dest_cache,
   uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_widen_half(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                         enum agx3_alu_hint hint, bool src_abs, bool src_neg,
                         bool dest_cache, uint16_t wait_mask,
                         uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

#ifdef __cplusplus
}
#endif
