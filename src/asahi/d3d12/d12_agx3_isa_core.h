/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Bounded D3D12 scalar-ALU mapping into the source-grounded AGX3 packer.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "asahi/lib/agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Encoding-scoped target gates in this bridge cover every M3--M5/A18 Pro
 * identity whose Apple compiler family is known, plus the existing G16P
 * target.  G15C/G16C/G17C use the exact g15d/g16s/g17s compiler target shared
 * with their sibling identity.  G17A remains rejected because no toolchain
 * family or compiler device class has been recovered for it. */

#define D12_AGX3_BINARY_ALU_MAX_LENGTH 14
#define D12_AGX3_TERNARY_ALU_MAX_LENGTH 14
#define D12_AGX3_FADD_F32_REG_REG_LENGTH 6
#define D12_AGX3_FMUL_F32_REG_REG_LENGTH 6
#define D12_AGX3_FMUL_F32_REG_REG_LONG_LENGTH 14
#define D12_AGX3_FMA_F32_REG_REG_REG_MAX_LENGTH 16
#define D12_AGX3_FMIN_F32_REG_REG_MAX_LENGTH 14
#define D12_AGX3_FMAX_F32_REG_REG_MAX_LENGTH 14
#define D12_AGX3_I32_TO_F32_REG_MAX_LENGTH 10
#define D12_AGX3_U32_TO_F32_REG_MAX_LENGTH 10
#define D12_AGX3_F32_RINT_REG_LENGTH 10
#define D12_AGX3_F32_RCP_REG_LENGTH 10
#define D12_AGX3_F32_RSQRT_REG_LENGTH 10
#define D12_AGX3_I32_COMPARE_SELECT_IMM_5_7_REG_REG_MAX_LENGTH 14
#define D12_AGX3_F32_TO_INT_RTZ_REG_LENGTH 10
#define D12_AGX3_YIELD_I16_REG_LENGTH 6
#define D12_AGX3_WAVE_READ_LANE_AT_F32_IMM_LENGTH 10
#define D12_AGX3_SUBGROUP_MAX_LENGTH 16
#define D12_AGX3_END_WITH_WAIT_FENCE_MASK_LENGTH 4
/* Compare plus cnd: the longest compare is fcmp.ep2 at fourteen bytes and
 * cnd is four. */
#define D12_AGX3_CF_MAX_LENGTH 18
#define D12_AGX3_VARY_MAX_LENGTH 10
/* Texture ops: the texel fetch is 14 bytes (AGX3b) / 8 (AGX3c); the
 * filtered sample reaches 22 bytes (AGX3c smp.ep3). */
#define D12_AGX3_TEX_MAX_LENGTH 22
#define D12_AGX3_TILE_MAX_LENGTH 14
#define D12_AGX3_LOCAL_TPS_MAX_LENGTH 16
#define D12_AGX3_STATIC_SPILL_MAX_LENGTH 10

/* These are D3D/DXIL scalar semantics, not raw Apple opcode numbers. */
enum d12_agx3_binary_alu_op {
   D12_AGX3_BINARY_ALU_IADD_U32 = 0,
   D12_AGX3_BINARY_ALU_IAND_U32 = 1,
   D12_AGX3_BINARY_ALU_IOR_U32 = 2,
   D12_AGX3_BINARY_ALU_IXOR_U32 = 3,
   D12_AGX3_BINARY_ALU_ISUB_U32 = 4,
   D12_AGX3_BINARY_ALU_ISHL_U32 = 5,
   D12_AGX3_BINARY_ALU_IASHR_I32 = 6,
   D12_AGX3_BINARY_ALU_ILSHR_U32 = 7,
   D12_AGX3_BINARY_ALU_IMUL_U32 = 8,
};

/*
 * Selection presented to the target-aware integer-multiply bridge.  Only the
 * low-u32 product is encoded.  Naming the rejected selections here lets a
 * lowering caller fail closed instead of accidentally treating an addend or
 * high product as the canonical three-register multiply.
 */
enum d12_agx3_integer_multiply_selection {
   D12_AGX3_INTEGER_MULTIPLY_LOW_U32 = 0,
   D12_AGX3_INTEGER_MULTIPLY_WITH_ADDEND_U32 = 1,
   D12_AGX3_INTEGER_MULTIPLY_HIGH_I32 = 2,
   D12_AGX3_INTEGER_MULTIPLY_HIGH_U32 = 3,
};

/* These selections are validation inputs, not claims of encoded support. */
enum d12_agx3_integer_multiply_add_selection {
   D12_AGX3_INTEGER_MULTIPLY_ADD_LOW_U32 = 0,
   D12_AGX3_INTEGER_MULTIPLY_ADD_IMPLICIT_ZERO_IMUL = 1,
   D12_AGX3_INTEGER_MULTIPLY_ADD_ALTERNATE_P3_LOW_U32 = 2,
   D12_AGX3_INTEGER_MULTIPLY_ADD_ALTERNATE_Q_LOW_U32 = 3,
   D12_AGX3_INTEGER_MULTIPLY_ADD_HIGH_I32 = 4,
   D12_AGX3_INTEGER_MULTIPLY_ADD_HIGH_U32 = 5,
};

enum d12_agx3_ternary_alu_op {
   D12_AGX3_TERNARY_ALU_IMADD_U32 = 0,
};

enum d12_agx3_unary_alu_op {
   D12_AGX3_UNARY_ALU_INOT_U32 = 0,
};

enum d12_agx3_f32_to_int_rtz_mode {
   D12_AGX3_F32_TO_I32_RTZ = 0,
   D12_AGX3_F32_TO_U32_RTZ = 1,
};

enum d12_agx3_i32_compare_predicate {
   D12_AGX3_I32_COMPARE_EQ = 0,
   D12_AGX3_I32_COMPARE_SLT = 1,
};

/* Structured-control-flow predicates expressed in D3D/DXIL semantics. The
 * bridge maps these to AGX3's flag compare plus cnd consumer; the raw Apple
 * predicate numbers are not exposed to callers. */
enum d12_agx3_i32_cf_predicate {
   D12_AGX3_I32_CF_EQ = 0,
   D12_AGX3_I32_CF_ULT = 1,
   D12_AGX3_I32_CF_UGT = 2,
   D12_AGX3_I32_CF_SLT = 3,
   D12_AGX3_I32_CF_SGT = 4,
};

enum d12_agx3_f32_cf_predicate {
   D12_AGX3_F32_CF_EQ = 0,
   D12_AGX3_F32_CF_LT = 1,
   D12_AGX3_F32_CF_GT = 2,
   D12_AGX3_F32_CF_LT_OR_NAN = 3,
   D12_AGX3_F32_CF_GT_OR_NAN = 4,
   D12_AGX3_F32_CF_GE = 5,
   D12_AGX3_F32_CF_LE = 6,
};

enum d12_agx3_cf_condition_op {
   D12_AGX3_CF_IF = 0,
   D12_AGX3_CF_LOOP_OR_BREAK_TEST = 1,
};

enum d12_agx3_cf_stack_op {
   D12_AGX3_CF_LOOP_ENTRY = 0,
   D12_AGX3_CF_ELSE = 1,
   D12_AGX3_CF_POP_1 = 2,
   D12_AGX3_CF_POP_2 = 3,
};

enum d12_agx3_cf_branch_op {
   D12_AGX3_CF_BRANCH_EXEC_NONE = 0,
   D12_AGX3_CF_BRANCH_EXEC_ANY = 1,
};

enum d12_agx3_vary_location {
   D12_AGX3_VARY_CENTER = 0,
   D12_AGX3_VARY_CENTROID = 1,
};

enum d12_agx3_texture_result_width {
   D12_AGX3_TEXTURE_RESULT_16 = 16,
   D12_AGX3_TEXTURE_RESULT_32 = 32,
};

struct d12_agx3_texture_fetch_desc {
   uint16_t dest;
   uint16_t coord;
   enum d12_agx3_texture_result_width width;
   uint8_t mask;
   uint8_t texture;
   uint8_t slot;
   uint16_t wait_mask;
   bool coord_kill;
   bool final;
};

enum d12_agx3_tile_width {
   D12_AGX3_TILE_WIDTH_8 = 8,
   D12_AGX3_TILE_WIDTH_16 = 16,
   D12_AGX3_TILE_WIDTH_32 = 32,
};

struct d12_agx3_tile_store_desc {
   uint16_t data;
   enum d12_agx3_tile_width width;
   uint8_t mask;
   uint16_t tile_offset;
   uint16_t wait_mask;
};

struct d12_agx3_tile_load_desc {
   uint16_t dest;
   enum d12_agx3_tile_width width;
   uint8_t mask;
   uint16_t tile_offset;
   uint8_t slot;
   bool first;
   bool last;
};

enum d12_agx3_memory_width {
   D12_AGX3_MEMORY_WIDTH_8 = 8,
   D12_AGX3_MEMORY_WIDTH_16 = 16,
   D12_AGX3_MEMORY_WIDTH_32 = 32,
};

enum d12_agx3_threadgroup_base_kind {
   D12_AGX3_THREADGROUP_BASE_IMMEDIATE = 0,
   D12_AGX3_THREADGROUP_BASE_REGISTER = 1,
   D12_AGX3_THREADGROUP_BASE_UNIFORM = 2,
};

/* Normalized ldtg/sttg roles used by DXIL groupshared-memory lowering.  All
 * register operands are Mesa half-register indices. */
struct d12_agx3_threadgroup_desc {
   uint16_t data; /* load destination / store value */
   enum d12_agx3_memory_width width;
   uint8_t mask;
   uint8_t index_scale_log2;

   enum d12_agx3_threadgroup_base_kind base_kind;
   uint16_t base;
   bool base_kill;

   bool index_is_register;
   uint16_t index;
   bool index_kill;
   uint16_t byte_offset;

   bool value_kill; /* store only */
   uint8_t slot;    /* load only */
   uint16_t wait_mask;
};

/* Normalized dynamic ldtps/sttps roles.  frame_region is the witnessed 1..6
 * structural selector; its hardware taxonomy is deliberately not inferred.
 * Static spill copy/rld forms and RT-only SCS memory are separate families. */
struct d12_agx3_tps_desc {
   uint16_t data; /* load destination / store value */
   enum d12_agx3_memory_width width;
   uint8_t mask;
   uint8_t index_scale_log2;
   uint16_t byte_offset;

   uint16_t index;
   bool index_kill;
   bool index_signed;
   uint8_t frame_region;

   bool value_kill; /* store only */
   uint8_t slot;    /* load only */
   uint16_t wait_mask;
};

/* Normalized scalar-32 static spill roles. byte_offset is the D3D/DXIL-facing
 * byte address; the bridge converts it to Apple's TPS half-register index.
 * reg_kill is valid only for a store source. */
struct d12_agx3_static_spill_desc {
   uint16_t reg; /* load destination / store source */
   uint32_t byte_offset;
   bool reg_kill;
   uint8_t slot;
   uint16_t wait_mask;
};

/* Register-ready wave/quad forms used after DXIL subgroup lowering. */
enum d12_agx3_subgroup_scope {
   D12_AGX3_SUBGROUP_SCOPE_WAVE = 0,
   D12_AGX3_SUBGROUP_SCOPE_QUAD,
};

enum d12_agx3_subgroup_shuffle_op {
   /* WaveReadLaneAt / QuadReadLaneAt after the lane is materialized. */
   D12_AGX3_SUBGROUP_SHUFFLE_READ_LANE_AT = 0,
   D12_AGX3_SUBGROUP_SHUFFLE_XOR,
   D12_AGX3_SUBGROUP_SHUFFLE_UP,
   D12_AGX3_SUBGROUP_SHUFFLE_DOWN,
};

enum d12_agx3_subgroup_lane_source {
   D12_AGX3_SUBGROUP_LANE_REGISTER = 0,
   D12_AGX3_SUBGROUP_LANE_UNIFORM,
};

struct d12_agx3_subgroup_shuffle_desc {
   enum d12_agx3_subgroup_scope scope;
   enum d12_agx3_subgroup_shuffle_op op;
   enum d12_agx3_subgroup_lane_source lane_source;
   uint16_t dest;
   uint16_t src;
   uint16_t lane;
};

enum d12_agx3_zero_ballot_type {
   D12_AGX3_ZERO_BALLOT_I32 = 0,
   D12_AGX3_ZERO_BALLOT_F32,
};

/* Integer ordering is signed. Float ordering follows the recovered f32
 * condition forms; NOT_EQUAL is the exact inverted-equality witness. */
enum d12_agx3_zero_ballot_predicate {
   D12_AGX3_ZERO_BALLOT_EQUAL = 0,
   D12_AGX3_ZERO_BALLOT_NOT_EQUAL,
   D12_AGX3_ZERO_BALLOT_LESS,
   D12_AGX3_ZERO_BALLOT_GREATER_EQUAL,
   D12_AGX3_ZERO_BALLOT_GREATER,
   D12_AGX3_ZERO_BALLOT_LESS_EQUAL,
};

struct d12_agx3_zero_ballot_desc {
   enum d12_agx3_subgroup_scope scope;
   enum d12_agx3_zero_ballot_type type;
   enum d12_agx3_zero_ballot_predicate predicate;
   uint16_t dest;
   uint16_t src;
};

/* WaveActive* and WavePrefix* register-ready operations. Prefix is grounded
 * only for FADD_F32, FMUL_F32 and IADD_U32; the shared encoder rejects every
 * unsupported wave/quad/operation combination. */
enum d12_agx3_subgroup_reduce_op {
   D12_AGX3_SUBGROUP_REDUCE_AND_U32 = 0,
   D12_AGX3_SUBGROUP_REDUCE_FADD_F32,
   D12_AGX3_SUBGROUP_REDUCE_OR_U32,
   D12_AGX3_SUBGROUP_REDUCE_FMUL_F32,
   D12_AGX3_SUBGROUP_REDUCE_XOR_U32,
   D12_AGX3_SUBGROUP_REDUCE_FMIN_F32,
   D12_AGX3_SUBGROUP_REDUCE_FMAX_F32,
   D12_AGX3_SUBGROUP_REDUCE_IADD_U32,
   D12_AGX3_SUBGROUP_REDUCE_SMIN_I32,
   D12_AGX3_SUBGROUP_REDUCE_SMAX_I32,
   D12_AGX3_SUBGROUP_REDUCE_UMIN_U32,
   D12_AGX3_SUBGROUP_REDUCE_UMAX_U32,
};

struct d12_agx3_subgroup_reduce_desc {
   enum d12_agx3_subgroup_scope scope;
   enum d12_agx3_subgroup_reduce_op op;
   bool prefix;
   uint16_t dest;
   uint16_t src;
};

struct d12_agx3_binary_alu_encoding {
   uint8_t bytes[D12_AGX3_BINARY_ALU_MAX_LENGTH];
   uint8_t length;
};

struct d12_agx3_unary_alu_encoding {
   uint8_t bytes[D12_AGX3_BINARY_ALU_MAX_LENGTH];
   uint8_t length;
};

struct d12_agx3_ternary_alu_encoding {
   uint8_t bytes[D12_AGX3_TERNARY_ALU_MAX_LENGTH];
   uint8_t length;
};

struct d12_agx3_fadd_f32_reg_reg_encoding {
   uint8_t bytes[D12_AGX3_FADD_F32_REG_REG_LENGTH];
   uint8_t length;
};

struct d12_agx3_fmul_f32_reg_reg_encoding {
   uint8_t bytes[D12_AGX3_FMUL_F32_REG_REG_LENGTH];
   uint8_t length;
};

struct d12_agx3_fmul_f32_reg_reg_long_encoding {
   uint8_t bytes[D12_AGX3_FMUL_F32_REG_REG_LONG_LENGTH];
   uint8_t length;
};

struct d12_agx3_fma_f32_reg_reg_reg_encoding {
   uint8_t bytes[D12_AGX3_FMA_F32_REG_REG_REG_MAX_LENGTH];
   uint8_t length;
};

struct d12_agx3_fmin_f32_reg_reg_encoding {
   uint8_t bytes[D12_AGX3_FMIN_F32_REG_REG_MAX_LENGTH];
   uint8_t length;
};

struct d12_agx3_fmax_f32_reg_reg_encoding {
   uint8_t bytes[D12_AGX3_FMAX_F32_REG_REG_MAX_LENGTH];
   uint8_t length;
};

struct d12_agx3_i32_to_f32_reg_encoding {
   uint8_t bytes[D12_AGX3_I32_TO_F32_REG_MAX_LENGTH];
   uint8_t length;
};

struct d12_agx3_u32_to_f32_reg_encoding {
   uint8_t bytes[D12_AGX3_U32_TO_F32_REG_MAX_LENGTH];
   uint8_t length;
};

struct d12_agx3_f32_rint_reg_encoding {
   uint8_t bytes[D12_AGX3_F32_RINT_REG_LENGTH];
   uint8_t length;
};

struct d12_agx3_f32_rcp_reg_encoding {
   uint8_t bytes[D12_AGX3_F32_RCP_REG_LENGTH];
   uint8_t length;
};

struct d12_agx3_f32_rsqrt_reg_encoding {
   uint8_t bytes[D12_AGX3_F32_RSQRT_REG_LENGTH];
   uint8_t length;
};

struct d12_agx3_i32_compare_select_imm_5_7_reg_reg_encoding {
   uint8_t bytes[D12_AGX3_I32_COMPARE_SELECT_IMM_5_7_REG_REG_MAX_LENGTH];
   uint8_t length;
};

struct d12_agx3_f32_to_int_rtz_reg_encoding {
   uint8_t bytes[D12_AGX3_F32_TO_INT_RTZ_REG_LENGTH];
   uint8_t length;
};

struct d12_agx3_yield_i16_reg_encoding {
   uint8_t bytes[D12_AGX3_YIELD_I16_REG_LENGTH];
   uint8_t length;
};

struct d12_agx3_wave_read_lane_at_f32_imm_encoding {
   uint8_t bytes[D12_AGX3_WAVE_READ_LANE_AT_F32_IMM_LENGTH];
   uint8_t length;
};

struct d12_agx3_subgroup_encoding {
   uint8_t bytes[D12_AGX3_SUBGROUP_MAX_LENGTH];
   uint8_t length;
};

struct d12_agx3_end_with_wait_fence_mask_encoding {
   uint8_t bytes[D12_AGX3_END_WITH_WAIT_FENCE_MASK_LENGTH];
   uint8_t length;
};

struct d12_agx3_cf_encoding {
   uint8_t bytes[D12_AGX3_CF_MAX_LENGTH];
   uint8_t length;
};

struct d12_agx3_vary_encoding {
   uint8_t bytes[D12_AGX3_VARY_MAX_LENGTH];
   uint8_t length;
};

struct d12_agx3_texture_fetch_encoding {
   uint8_t bytes[D12_AGX3_TEX_MAX_LENGTH];
   uint8_t length;
};

struct d12_agx3_tile_encoding {
   uint8_t bytes[D12_AGX3_TILE_MAX_LENGTH];
   uint8_t length;
};

struct d12_agx3_local_tps_encoding {
   uint8_t bytes[D12_AGX3_LOCAL_TPS_MAX_LENGTH];
   uint8_t length;
};

struct d12_agx3_static_spill_encoding {
   uint8_t bytes[D12_AGX3_STATIC_SPILL_MAX_LENGTH];
   uint8_t length;
};

/*
 * Encode one canonical three-register scalar operation. Register arguments
 * are Mesa half-register indices, so 32-bit operands must be even. This is a
 * compiler building block only: success does not imply that AGX3 pipeline
 * compilation, scheduling, command submission, or device admission exists.
 * ISHL, IASHR and ILSHR map only the proved register-register `sll`, `sra`
 * and `srl` instructions; they do not emit or claim the separate shift-count
 * masking seen in compiler output. IMUL maps only the implicit-zero-addend
 * low-u32 product selected by the shared packer, not integer add or high
 * multiply forms.
 */
bool d12_agx3_encode_binary_alu(
   enum agx_isa_encoding encoding, enum d12_agx3_binary_alu_op op,
   uint16_t dest, uint16_t src0, uint16_t src1,
   struct d12_agx3_binary_alu_encoding *out);

/*
 * Encode the canonical integer multiply for an encoding-scoped Apple compiler
 * target described above.  This is still only an ISA/lowering building block
 * and does not admit any AGX3 device.
 */
bool d12_agx3_encode_integer_multiply(
   enum agx_chip chip, enum d12_agx3_integer_multiply_selection selection,
   uint16_t dest, uint16_t src0, uint16_t src1,
   struct d12_agx3_binary_alu_encoding *out);

/*
 * Encode the proved real-register-addend, low-u32 multiply-add selection.
 * Signed and unsigned low products are identical modulo 2^32.  The separate
 * implicit-zero multiply, alternate low selections, and high-half forms are
 * deliberately represented only as rejected selections.  The target gate is
 * the same compiler-witness allowlist as d12_agx3_encode_integer_multiply;
 * this does not enable live NIR/DXIL lowering or any AGX3 runtime path.
 */
bool d12_agx3_encode_integer_multiply_add(
   enum agx_chip chip,
   enum d12_agx3_integer_multiply_add_selection selection, uint16_t dest,
   uint16_t mul0, uint16_t mul1, uint16_t addend,
   struct d12_agx3_ternary_alu_encoding *out);

/*
 * Encode/decode only the shared packer's exact compact register-register f32
 * add on the encoding-scoped targets described above. `encoding` must match
 * `chip`. All operands are even half-register indices in 0..254, and their
 * physical source-field order is preserved even though addition is
 * commutative.
 *
 * This is a strict ISA representation only.  It neither accepts a canonical
 * prefix with trailing bytes nor enables DXIL lowering, scheduling, command
 * submission, device admission, runtime use, or any D3D12 feature.
 */
bool d12_agx3_encode_fadd_f32_reg_reg(
   enum agx_chip chip, enum agx_isa_encoding encoding, uint16_t dest,
   uint16_t src0, uint16_t src1,
   struct d12_agx3_fadd_f32_reg_reg_encoding *out);
bool d12_agx3_decode_fadd_f32_reg_reg(
   enum agx_chip chip, enum agx_isa_encoding encoding,
   const void *instruction, uint32_t instruction_size, uint16_t *dest,
   uint16_t *src0, uint16_t *src1);

/*
 * Encode/decode only the exact compact register-register f32 multiply form
 * witnessed by the L/P2 AGX3b corpus and the Q AGX3c corpus, including the
 * exact compiler-target aliases described above. G17G remains rejected
 * because the pinned corpus witnesses only its longer multiply form.
 * `encoding` must match the target, and all operands are even half-register
 * indices in the conservatively proved 0..126 domain.
 *
 * This strict six-byte ISA bridge does not accept trailing bytes and does not
 * enable DXIL lowering, scheduling, command submission, device admission,
 * runtime use, or any D3D12 feature.
 */
bool d12_agx3_encode_fmul_f32_reg_reg(
   enum agx_chip chip, enum agx_isa_encoding encoding, uint16_t dest,
   uint16_t src0, uint16_t src1,
   struct d12_agx3_fmul_f32_reg_reg_encoding *out);
bool d12_agx3_decode_fmul_f32_reg_reg(
   enum agx_chip chip, enum agx_isa_encoding encoding,
   const void *instruction, uint32_t instruction_size, uint16_t *dest,
   uint16_t *src0, uint16_t *src1);

/*
 * Encode/decode the exact fourteen-byte register-register f32 multiply form
 * independently witnessed on G17G and the shared G17S/G17C compiler target.
 * The explicit long-form API keeps its fixed anonymous/context selection
 * separate from the compact G17S/G17C form. Operands are even half-register
 * indices in the conservatively proved 0..126 domain.
 *
 * This is an exact ISA bridge only.  It does not enable DXIL lowering,
 * scheduling, command submission, device admission, runtime use, or any
 * D3D12 feature.
 */
bool d12_agx3_encode_fmul_f32_reg_reg_long(
   enum agx_chip chip, enum agx_isa_encoding encoding, uint16_t dest,
   uint16_t src0, uint16_t src1,
   struct d12_agx3_fmul_f32_reg_reg_long_encoding *out);
bool d12_agx3_decode_fmul_f32_reg_reg_long(
   enum agx_chip chip, enum agx_isa_encoding encoding,
   const void *instruction, uint32_t instruction_size, uint16_t *dest,
   uint16_t *src0, uint16_t *src1);

/*
 * Encode/decode the exact four-register f32 fused multiply-add form on the
 * encoding-scoped targets described above. AGX3b uses the eight-byte form and
 * AGX3c uses the sixteen-byte form. `encoding` must match `chip`. All operands
 * are even half-register indices in the conservatively proved 0..126 domain,
 * and the physical multiply inputs preserve their encoded order.
 *
 * This strict bridge rejects trailing bytes, anonymous/context alternates,
 * unfused-mad claims, G17A, and all unsupported registers. It is an ISA
 * representation only: it does
 * not enable DXIL lowering, contraction policy, scheduling, command
 * submission, device admission, runtime use, or any D3D12 feature.
 */
bool d12_agx3_encode_fma_f32_reg_reg_reg(
   enum agx_chip chip, enum agx_isa_encoding encoding, uint16_t dest,
   uint16_t mul0, uint16_t mul1, uint16_t addend,
   struct d12_agx3_fma_f32_reg_reg_reg_encoding *out);
bool d12_agx3_decode_fma_f32_reg_reg_reg(
   enum agx_chip chip, enum agx_isa_encoding encoding,
   const void *instruction, uint32_t instruction_size, uint16_t *dest,
   uint16_t *mul0, uint16_t *mul1, uint16_t *addend);

/*
 * Encode/decode the exact f32-minimum compiler selection on the encoding-
 * scoped targets described above. AGX3b uses the six-byte form and AGX3c the
 * fourteen-byte form. `encoding` must match `chip`, and all operands are even
 * half-register indices in 0..126. The AGX3c bridge
 * also enforces equality of the duplicated comparison and selection sources.
 *
 * This exact representation rejects other predicates, fmax/generic select,
 * modifiers, trailing bytes, G17A, and unproven registers. It does not
 * independently define DXIL NaN or
 * signed-zero policy and does not enable lowering, scheduling, runtime use,
 * device admission, command submission, or a D3D12 feature.
 */
bool d12_agx3_encode_fmin_f32_reg_reg(
   enum agx_chip chip, enum agx_isa_encoding encoding, uint16_t dest,
   uint16_t src0, uint16_t src1,
   struct d12_agx3_fmin_f32_reg_reg_encoding *out);
bool d12_agx3_decode_fmin_f32_reg_reg(
   enum agx_chip chip, enum agx_isa_encoding encoding,
   const void *instruction, uint32_t instruction_size, uint16_t *dest,
   uint16_t *src0, uint16_t *src1);

/*
 * Encode/decode the exact f32-maximum compiler selection on the encoding-
 * scoped targets described above. AGX3b uses the six-byte form and AGX3c the
 * fourteen-byte form. `encoding` must match `chip`, and all operands are even
 * half-register indices in 0..126. The AGX3c bridge
 * also enforces equality of the duplicated comparison and selection sources.
 *
 * This exact representation rejects other predicates, fmin/generic select,
 * modifiers, trailing bytes, G17A, and unproven registers. It does not
 * independently define DXIL NaN or
 * signed-zero policy and does not enable lowering, scheduling, runtime use,
 * device admission, command submission, or a D3D12 feature.
 */
bool d12_agx3_encode_fmax_f32_reg_reg(
   enum agx_chip chip, enum agx_isa_encoding encoding, uint16_t dest,
   uint16_t src0, uint16_t src1,
   struct d12_agx3_fmax_f32_reg_reg_encoding *out);
bool d12_agx3_decode_fmax_f32_reg_reg(
   enum agx_chip chip, enum agx_isa_encoding encoding,
   const void *instruction, uint32_t instruction_size, uint16_t *dest,
   uint16_t *src0, uint16_t *src1);

/*
 * Encode/decode the exact signed-i32-to-f32 register conversion on the
 * encoding-scoped targets described above. AGX3b uses the eight-byte form and
 * AGX3c the ten-byte form. `encoding` must match `chip`, and both operands are
 * even half-register indices in 0..126.
 *
 * This strict ISA bridge rejects unsigned and float-to-integer conversions,
 * alternate source selections, modifiers, trailing bytes, G17A, and unproven
 * registers. It does not enable DXIL
 * lowering, conversion policy, scheduling, command submission, device
 * admission, runtime use, or any D3D12 feature.
 */
bool d12_agx3_encode_i32_to_f32_reg(
   enum agx_chip chip, enum agx_isa_encoding encoding, uint16_t dest,
   uint16_t src, struct d12_agx3_i32_to_f32_reg_encoding *out);
bool d12_agx3_decode_i32_to_f32_reg(
   enum agx_chip chip, enum agx_isa_encoding encoding,
   const void *instruction, uint32_t instruction_size, uint16_t *dest,
   uint16_t *src);

/*
 * Encode/decode the exact unsigned-u32-to-f32 register conversion on the
 * encoding-scoped targets described above. AGX3b uses the eight-byte form and
 * AGX3c the ten-byte form. `encoding` must match `chip`, and both operands are
 * even half-register indices in 0..126.
 *
 * This strict ISA bridge rejects the distinct signed and float-to-integer
 * conversions, alternate source selections, modifiers, trailing bytes, G17A,
 * and unproven registers. It does not
 * enable DXIL lowering, conversion policy, scheduling, command submission,
 * device admission, runtime use, or any D3D12 feature.
 */
bool d12_agx3_encode_u32_to_f32_reg(
   enum agx_chip chip, enum agx_isa_encoding encoding, uint16_t dest,
   uint16_t src, struct d12_agx3_u32_to_f32_reg_encoding *out);
bool d12_agx3_decode_u32_to_f32_reg(
   enum agx_chip chip, enum agx_isa_encoding encoding,
   const void *instruction, uint32_t instruction_size, uint16_t *dest,
   uint16_t *src);

/*
 * Encode/decode only the exact target-gated Metal `rint` selection observed
 * immediately before PC_f2ir's signed conversion on the encoding-scoped
 * targets described above. Both operands are even half-register indices in
 * 0..126 and `encoding` must match the chip.
 *
 * Other rounding selectors, exceptional-value/tie policy, saturation,
 * modifiers, alternate scheduler/last-use contexts, trailing bytes, and
 * G17A and unknown targets fail closed. This bridge does not enable DXIL
 * lowering, scheduling, runtime use, submission, device admission, or any
 * D3D12 capability.
 */
bool d12_agx3_encode_f32_rint_reg(
   enum agx_chip chip, enum agx_isa_encoding encoding, uint16_t dest,
   uint16_t src, struct d12_agx3_f32_rint_reg_encoding *out);
bool d12_agx3_decode_f32_rint_reg(
   enum agx_chip chip, enum agx_isa_encoding encoding,
   const void *instruction, uint32_t instruction_size, uint16_t *dest,
   uint16_t *src);

/*
 * Encode/decode only the directly witnessed scalar f32 reciprocal selection.
 * The encoding-scoped targets described above use their AGX3b/AGX3c table.
 * Both operands are even half-register indices in 0..126, and `encoding`
 * must match `chip`.
 *
 * The exact direct last-use/dependency context is fixed.  Live-source and
 * alternate dependency contexts, rsqrt and other special functions,
 * accuracy/refinement policy, modifiers, trailing bytes, high registers, and
 * G17A and unknown identities fail closed. This does not enable DXIL
 * lowering, scheduling, runtime use, submission, device admission, or a
 * D3D12 capability.
 */
bool d12_agx3_encode_f32_rcp_reg(
   enum agx_chip chip, enum agx_isa_encoding encoding, uint16_t dest,
   uint16_t src, struct d12_agx3_f32_rcp_reg_encoding *out);
bool d12_agx3_decode_f32_rcp_reg(
   enum agx_chip chip, enum agx_isa_encoding encoding,
   const void *instruction, uint32_t instruction_size, uint16_t *dest,
   uint16_t *src);

/*
 * Encode/decode only the directly witnessed scalar f32 reciprocal-square-root
 * selection. The encoding-scoped targets described above use their AGX3b/
 * AGX3c table. Both operands are even half-register indices in 0..126, and
 * `encoding` must match `chip`.
 *
 * The exact direct last-use/dependency context is fixed.  Live-source and
 * alternate dependency contexts, reciprocal and other special functions,
 * accuracy/refinement policy, modifiers, trailing bytes, high registers, and
 * G17A and unknown identities fail closed. This does not enable DXIL
 * lowering, scheduling, runtime use, submission, device admission, or a
 * D3D12 capability.
 */
bool d12_agx3_encode_f32_rsqrt_reg(
   enum agx_chip chip, enum agx_isa_encoding encoding, uint16_t dest,
   uint16_t src, struct d12_agx3_f32_rsqrt_reg_encoding *out);
bool d12_agx3_decode_f32_rsqrt_reg(
   enum agx_chip chip, enum agx_isa_encoding encoding,
   const void *instruction, uint32_t instruction_size, uint16_t *dest,
   uint16_t *src);

/*
 * Encode/decode only the directly witnessed fused i32 compare and fixed
 * immediate select `(src0 OP src1) ? 5 : 7`.  OP is limited to equality or
 * signed less-than. The encoding-scoped targets described above use their
 * AGX3b/AGX3c table. All operands are even half-register indices in 0..126,
 * and `encoding` must match the chip identity.
 *
 * Other predicates, arbitrary immediate/register-selected values, alternate
 * contexts, trailing bytes, high registers, G17A, and unknown identities fail
 * closed. This representation does not enable DXIL lowering,
 * control-flow formation, scheduling, runtime use, submission, admission, or
 * a D3D12 capability.
 */
bool d12_agx3_encode_i32_compare_select_imm_5_7_reg_reg(
   enum agx_chip chip, enum agx_isa_encoding encoding,
   enum d12_agx3_i32_compare_predicate predicate, uint16_t dest,
   uint16_t src0, uint16_t src1,
   struct d12_agx3_i32_compare_select_imm_5_7_reg_reg_encoding *out);
bool d12_agx3_decode_i32_compare_select_imm_5_7_reg_reg(
   enum agx_chip chip, enum agx_isa_encoding encoding,
   const void *instruction, uint32_t instruction_size,
   enum d12_agx3_i32_compare_predicate *predicate, uint16_t *dest,
   uint16_t *src0, uint16_t *src1);

/*
 * Encode/decode only the exact target-gated f32-to-i32/u32 RTZ language-cast
 * selections.  Both operands are even half-register indices in 0..126, and
 * `encoding` must match the directly witnessed chip identity.
 *
 * The bridge rejects every other rounding/source-reuse/context selection,
 * saturation, trailing bytes, high registers, G17A, and unknown targets.
 * NaN/out-of-range policy, DXIL lowering, scheduling, runtime use, submission,
 * device admission, and D3D12 capability exposure remain unimplemented.
 */
bool d12_agx3_encode_f32_to_int_rtz_reg(
   enum agx_chip chip, enum agx_isa_encoding encoding,
   enum d12_agx3_f32_to_int_rtz_mode mode, uint16_t dest, uint16_t src,
   struct d12_agx3_f32_to_int_rtz_reg_encoding *out);
bool d12_agx3_decode_f32_to_int_rtz_reg(
   enum agx_chip chip, enum agx_isa_encoding encoding,
   const void *instruction, uint32_t instruction_size,
   enum d12_agx3_f32_to_int_rtz_mode *mode, uint16_t *dest, uint16_t *src);

/*
 * Encode/decode only the canonical no-kill i16-register `yield` form on the
 * encoding-scoped Apple compiler targets described above. The source is a
 * 16-bit half-register index; AGX3b accepts 0..191 and AGX3c accepts 0..255.
 * Alternate operand selections, kill and scheduler/context fields, and the
 * AGX3c ninth register bit are deliberately absent and fail closed.
 *
 * This is a research-only ISA representation.  It does not enable D3D/DXIL
 * lowering, scheduling, command submission, device admission, API features,
 * or ray tracing.
 */
bool d12_agx3_encode_yield_i16_reg(
   enum agx_chip chip, uint16_t src,
   struct d12_agx3_yield_i16_reg_encoding *out);
bool d12_agx3_decode_yield_i16_reg(enum agx_chip chip,
                                   const void *instruction,
                                   uint32_t instruction_size, uint16_t *src);

/*
 * Encode/decode the exact Apple `simdbcast` register-source f32 form as the
 * D3D/DXIL WaveReadLaneAt semantic. Exposure is limited to the encoding-
 * scoped identities described above and the directly witnessed immediate
 * lanes 0 and 3. Operands are even half-register indices in
 * 0..190 on AGX3b and 0..254 on AGX3c.
 *
 * Dynamic lanes, other immediate lanes, integer/half types, WaveReadLaneFirst,
 * quad operations, lowering, scheduling, runtime use, device admission and
 * D3D12 wave-op capability advertisement remain absent.
 */
bool d12_agx3_encode_wave_read_lane_at_f32_imm(
   enum agx_chip chip, uint16_t dest, uint16_t src, uint8_t lane,
   struct d12_agx3_wave_read_lane_at_f32_imm_encoding *out);
bool d12_agx3_decode_wave_read_lane_at_f32_imm(
   enum agx_chip chip, const void *instruction, uint32_t instruction_size,
   uint16_t *dest, uint16_t *src, uint8_t *lane);

/* Encode only the exact dynamic-lane, zero-compare and aggregation families
 * on the encoding-scoped identities described above. Operands are scalar
 * f32/i32 half-register indices; a uniform lane is an even 0..510 uniform
 * half-register index.
 * These APIs are a bounded post-lowering ISA bridge only: they do not lower
 * DXIL, schedule dependency markers, expose wave caps, or enable runtime use.
 */
bool d12_agx3_encode_subgroup_shuffle(
   enum agx_chip chip, const struct d12_agx3_subgroup_shuffle_desc *desc,
   struct d12_agx3_subgroup_encoding *out);
bool d12_agx3_encode_zero_compare_ballot(
   enum agx_chip chip, const struct d12_agx3_zero_ballot_desc *desc,
   struct d12_agx3_subgroup_encoding *out);
bool d12_agx3_encode_subgroup_reduce(
   enum agx_chip chip, const struct d12_agx3_subgroup_reduce_desc *desc,
   struct d12_agx3_subgroup_encoding *out);

/*
 * Encode/decode only the dependency mask carried by a canonical terminal
 * `end` on the encoding-scoped Apple compiler targets described above.
 * AGX3b exposes slots 0..5 (masks <= 0x3f); AGX3c additionally exposes slots
 * 6 and 7.  Every other `end` field remains canonical zero.
 *
 * This is a research-only representation of an already terminal wait mask.
 * It is not a standalone wait instruction and does not enable D3D/DXIL
 * lowering, scheduling or memory-ordering policy, command submission, device
 * admission, or API capabilities.
 */
bool d12_agx3_encode_end_with_wait_fence_mask(
   enum agx_chip chip, uint8_t wait_fence_mask,
   struct d12_agx3_end_with_wait_fence_mask_encoding *out);
bool d12_agx3_decode_end_with_wait_fence_mask(
   enum agx_chip chip, const void *instruction, uint32_t instruction_size,
   uint8_t *wait_fence_mask);

/*
 * Encode the exact register-ready AGX3 structured-control-flow building
 * blocks on the encoding-scoped compiler targets described above. A
 * conditional form
 * is a compare into flag slot zero followed immediately by cndst (if) or
 * cndlt (loop/conditional break). Integer src1 may be an imm8; float sources
 * are registers. Register indices are 32-bit-aligned half-register numbers in
 * the witnessed 0..126 range. GE/LE are normalized to inverted ordered-or-NaN
 * tests exactly as Apple's compiler does.
 *
 * The stack helper exposes only the unconditional TRUE-slot loop entry,
 * canonical else, and one/two-level pops. Branch displacements are signed
 * byte offsets from the branch start: exec-none must be forward and exec-any
 * must be backward. Unknown predicates/forms, wrong-direction or odd branch
 * offsets, G17A, and unknown chip identities clear the result and fail.
 *
 * These are bounded ISA/lowering blocks. They do not implement DXIL control
 * flow, compare/cnd load-hazard scheduling, submission, device admission, or
 * a D3D12 capability.
 */
bool d12_agx3_encode_i32_cf(
   enum agx_chip chip, enum d12_agx3_cf_condition_op op,
   enum d12_agx3_i32_cf_predicate predicate, bool invert, uint16_t src0,
   bool src1_is_immediate, uint16_t src1_or_imm,
   struct d12_agx3_cf_encoding *out);
bool d12_agx3_encode_f32_cf(
   enum agx_chip chip, enum d12_agx3_cf_condition_op op,
   enum d12_agx3_f32_cf_predicate predicate, bool invert, uint16_t src0,
   uint16_t src1, struct d12_agx3_cf_encoding *out);
bool d12_agx3_encode_cf_stack(enum agx_chip chip,
                              enum d12_agx3_cf_stack_op op,
                              struct d12_agx3_cf_encoding *out);
bool d12_agx3_encode_cf_branch(enum agx_chip chip,
                               enum d12_agx3_cf_branch_op op,
                               int64_t byte_displacement,
                               struct d12_agx3_cf_encoding *out);

/*
 * Target-gated AGX3 varying building blocks. Interpolation is one scalar
 * f32interp; when `perspective` is true the caller must lower the surrounding
 * RHW interpolation, reciprocal, and multiply sequence. Explicit at-sample
 * coordinates and dynamic coefficient/UVS indices are intentionally absent.
 * LDCF is an asynchronous coefficient load; UVSW carries its scheduler slot,
 * value kill, and waits. These do not by themselves expose a D3D12 feature.
 */
bool d12_agx3_encode_vary_interp(
   enum agx_chip chip, uint16_t dest, uint16_t coefficient,
   bool perspective, enum d12_agx3_vary_location location,
   struct d12_agx3_vary_encoding *out);
bool d12_agx3_encode_vary_ldcf(enum agx_chip chip, uint16_t dest,
                               uint16_t coefficient, uint8_t slot,
                               struct d12_agx3_vary_encoding *out);
bool d12_agx3_encode_vary_uvsw(enum agx_chip chip, uint16_t value,
                               uint16_t uvs_index, uint8_t slot,
                               bool value_kill, uint16_t wait_mask,
                               struct d12_agx3_vary_encoding *out);

/* Target-gated ISA bridge for the directly witnessed Texture2D.Load LOD0
 * subset: non-array, immediate texture index, sampler zero, no offset,
 * compare, gather, sparse, bindless, register-selected state or MSAA.  The
 * descriptor carries only axes whose bit placements are grounded. */
bool d12_agx3_encode_texture_fetch(
   enum agx_chip chip, const struct d12_agx3_texture_fetch_desc *desc,
   struct d12_agx3_texture_fetch_encoding *out);
bool d12_agx3_decode_texture_fetch(
   enum agx_chip chip, const void *instruction, uint32_t instruction_size,
   struct d12_agx3_texture_fetch_desc *out);

/* Strict ISA bridge for the single-sample tile-buffer subset witnessed on the
 * tile corpus's exact G15G, G16G and G17S compiler targets.  Stores expose the
 * grounded value/width/mask/byte-offset and zero-or-one dependency wait axes.
 * Loads expose the full-mask destination, scoreboard slot and first/last load
 * markers; their corpus carries no incoming wait.  Per-sample selection,
 * explicit pixel coordinates, AGX3b's unresolved high-register f13 selection,
 * other chip identities, lowering, runtime use and capability exposure remain
 * deliberately absent. */
bool d12_agx3_encode_tile_store(
   enum agx_chip chip, const struct d12_agx3_tile_store_desc *desc,
   struct d12_agx3_tile_encoding *out);
bool d12_agx3_encode_tile_load(
   enum agx_chip chip, const struct d12_agx3_tile_load_desc *desc,
   struct d12_agx3_tile_encoding *out);

/* Target-gated bridges for byte-exact ldtg/sttg and dynamic ldtps/sttps.
 * The descriptors expose only source-grounded operand roles and scoreboard
 * state.  Eight-bit loads remain rejected because the policy for Apple's
 * structural narrow-extension bit is not grounded.  These functions do not
 * enable live DXIL lowering, AGX3 device admission, static spill copy/rld, or
 * the ungrounded RT-only ldscs/stscs family. */
bool d12_agx3_encode_threadgroup_load(
   enum agx_chip chip, const struct d12_agx3_threadgroup_desc *desc,
   struct d12_agx3_local_tps_encoding *out);
bool d12_agx3_encode_threadgroup_store(
   enum agx_chip chip, const struct d12_agx3_threadgroup_desc *desc,
   struct d12_agx3_local_tps_encoding *out);
bool d12_agx3_encode_tps_load(
   enum agx_chip chip, const struct d12_agx3_tps_desc *desc,
   struct d12_agx3_local_tps_encoding *out);
bool d12_agx3_encode_tps_store(
   enum agx_chip chip, const struct d12_agx3_tps_desc *desc,
   struct d12_agx3_local_tps_encoding *out);

/* Target-gated scalar-32 bridge for Apple's static copy/rld TPS spill forms.
 * Only roles with exact source-level meaning are exposed. Structural f11 and
 * f39/f41/f42/f44 stay clear; SCS/RT scratch is not part of this family. These
 * functions do not enable chip admission, submission, or a WDDM capability. */
bool d12_agx3_encode_static_spill_store(
   enum agx_chip chip, const struct d12_agx3_static_spill_desc *desc,
   struct d12_agx3_static_spill_encoding *out);
bool d12_agx3_decode_static_spill_store(
   enum agx_chip chip, const void *instruction, size_t length,
   struct d12_agx3_static_spill_desc *out);
bool d12_agx3_encode_static_spill_load(
   enum agx_chip chip, const struct d12_agx3_static_spill_desc *desc,
   struct d12_agx3_static_spill_encoding *out);
bool d12_agx3_decode_static_spill_load(
   enum agx_chip chip, const void *instruction, size_t length,
   struct d12_agx3_static_spill_desc *out);

/* Encoding-level form used by the target-aware lowering contract above. */
bool d12_agx3_encode_ternary_alu(
   enum agx_isa_encoding encoding, enum d12_agx3_ternary_alu_op op,
   uint16_t dest, uint16_t src0, uint16_t src1, uint16_t src2,
   struct d12_agx3_ternary_alu_encoding *out);

/* Same bounded contract as the binary bridge, for one-register semantics. */
bool d12_agx3_encode_unary_alu(
   enum agx_isa_encoding encoding, enum d12_agx3_unary_alu_op op,
   uint16_t dest, uint16_t src, struct d12_agx3_unary_alu_encoding *out);

#ifdef __cplusplus
}
#endif
