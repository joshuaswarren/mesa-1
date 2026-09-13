/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "d12_agx3_isa_core.h"

#include "asahi/compiler/agx3_cf_core.h"
#include "asahi/compiler/agx3_local_tps_core.h"
#include "asahi/compiler/agx3_pack.h"
#include "asahi/compiler/agx3_static_spill_core.h"
#include "asahi/compiler/agx3_subgroup_core.h"
#include "asahi/compiler/agx3_tex_core.h"
#include "asahi/compiler/agx3_tile_core.h"
#include "asahi/compiler/agx3_vary_core.h"

#include <string.h>

_Static_assert(AGX3_SLL_U32_REG_REG_MAX_LENGTH <=
                  D12_AGX3_BINARY_ALU_MAX_LENGTH,
               "D3D AGX3 ALU storage must hold the canonical sll form");
_Static_assert(AGX3_ASR_I32_REG_REG_MAX_LENGTH <=
                  D12_AGX3_BINARY_ALU_MAX_LENGTH,
               "D3D AGX3 ALU storage must hold the canonical sra form");
_Static_assert(AGX3_LSR_U32_REG_REG_MAX_LENGTH <=
                  D12_AGX3_BINARY_ALU_MAX_LENGTH,
               "D3D AGX3 ALU storage must hold the canonical srl form");
_Static_assert(AGX3_IMUL_U32_REG_REG_MAX_LENGTH <=
                  D12_AGX3_BINARY_ALU_MAX_LENGTH,
               "D3D AGX3 ALU storage must hold the canonical imul form");
_Static_assert(AGX3_IMADD_U32_REG_REG_REG_MAX_LENGTH <=
                  D12_AGX3_TERNARY_ALU_MAX_LENGTH,
               "D3D AGX3 ALU storage must hold the canonical imadd form");
_Static_assert(AGX3_FADD_F32_REG_REG_LENGTH ==
                  D12_AGX3_FADD_F32_REG_REG_LENGTH,
               "D3D AGX3 f32add storage must match the canonical form");
_Static_assert(AGX3_FMUL_F32_REG_REG_LENGTH ==
                  D12_AGX3_FMUL_F32_REG_REG_LENGTH,
               "D3D AGX3 f32mul storage must match the canonical form");
_Static_assert(AGX3_FMUL_F32_REG_REG_LONG_LENGTH ==
                  D12_AGX3_FMUL_F32_REG_REG_LONG_LENGTH,
               "D3D AGX3 long f32mul storage must match the canonical form");
_Static_assert(AGX3_FMAX_F32_REG_REG_MAX_LENGTH ==
                  D12_AGX3_FMAX_F32_REG_REG_MAX_LENGTH,
               "D3D AGX3 f32max storage must match the canonical form");
_Static_assert(AGX3_I32_TO_F32_REG_MAX_LENGTH ==
                  D12_AGX3_I32_TO_F32_REG_MAX_LENGTH,
               "D3D AGX3 i32-to-f32 storage must match the canonical form");
_Static_assert(AGX3_U32_TO_F32_REG_MAX_LENGTH ==
                  D12_AGX3_U32_TO_F32_REG_MAX_LENGTH,
               "D3D AGX3 u32-to-f32 storage must match the canonical form");
_Static_assert(AGX3_F32_RINT_REG_LENGTH == D12_AGX3_F32_RINT_REG_LENGTH,
               "D3D AGX3 f32-rint storage must match the canonical form");
_Static_assert(AGX3_F32_RCP_REG_LENGTH == D12_AGX3_F32_RCP_REG_LENGTH,
               "D3D AGX3 f32-rcp storage must match the canonical form");
_Static_assert(AGX3_F32_RSQRT_REG_LENGTH == D12_AGX3_F32_RSQRT_REG_LENGTH,
               "D3D AGX3 f32-rsqrt storage must match the canonical form");
_Static_assert(AGX3_I32_COMPARE_SELECT_IMM_5_7_REG_REG_MAX_LENGTH ==
                  D12_AGX3_I32_COMPARE_SELECT_IMM_5_7_REG_REG_MAX_LENGTH,
               "D3D AGX3 i32 compare/select storage must match canonical form");
_Static_assert(AGX3_F32_TO_INT_RTZ_REG_LENGTH ==
                  D12_AGX3_F32_TO_INT_RTZ_REG_LENGTH,
               "D3D AGX3 f32-to-int storage must match the canonical form");
_Static_assert(AGX3_YIELD_I16_REG_LENGTH == D12_AGX3_YIELD_I16_REG_LENGTH,
               "D3D AGX3 yield storage must match the canonical form");
_Static_assert(AGX3_SIMD_BROADCAST_F32_IMM_LENGTH ==
                  D12_AGX3_WAVE_READ_LANE_AT_F32_IMM_LENGTH,
               "D3D AGX3 wave-lane storage must match the canonical form");
_Static_assert(AGX3_SUBGROUP_MAX_LENGTH == D12_AGX3_SUBGROUP_MAX_LENGTH,
               "D3D AGX3 subgroup storage must match the shared encoder");
_Static_assert(AGX3_END_LENGTH ==
                  D12_AGX3_END_WITH_WAIT_FENCE_MASK_LENGTH,
               "D3D AGX3 terminal wait storage must match canonical end");
_Static_assert(AGX3_CF_MAX_LENGTH <= D12_AGX3_CF_MAX_LENGTH,
               "D3D AGX3 control-flow storage must hold one instruction");
_Static_assert(AGX3_CF_MAX_LENGTH + 4 <= D12_AGX3_CF_MAX_LENGTH,
               "D3D AGX3 control-flow storage must hold compare plus cnd");
_Static_assert(AGX3_VARY_MAX_LENGTH == D12_AGX3_VARY_MAX_LENGTH,
               "D3D AGX3 varying storage must match the canonical forms");
_Static_assert(AGX3_TEX_MAX_LENGTH == D12_AGX3_TEX_MAX_LENGTH,
               "D3D AGX3 texture storage must match the canonical forms");
_Static_assert(AGX3_TILE_MAX_LENGTH == D12_AGX3_TILE_MAX_LENGTH,
               "D3D AGX3 tile storage must match the canonical forms");
_Static_assert(AGX3_LOCAL_TPS_MAX_LENGTH == D12_AGX3_LOCAL_TPS_MAX_LENGTH,
               "D3D AGX3 local/TPS storage must match canonical forms");
_Static_assert(AGX3_STATIC_SPILL_MAX_LENGTH ==
                  D12_AGX3_STATIC_SPILL_MAX_LENGTH,
               "D3D AGX3 static-spill storage must match canonical forms");

static bool
d12_agx3_compiler_target_encoding(enum agx_chip chip,
                                  enum agx_isa_encoding *encoding)
{
   switch (chip) {
   /* C-tier identities use the same exact Apple compiler target as their
    * sibling (G15C -> g15d, G16C -> g16s).  G17A remains excluded because
    * the pinned compiler/runtime corpus has no toolchain family for it. */
   case AGX_CHIP_G15G:
   case AGX_CHIP_G15S:
   case AGX_CHIP_G15C:
   case AGX_CHIP_G15D:
   case AGX_CHIP_G16P:
   case AGX_CHIP_G16G:
   case AGX_CHIP_G16S:
   case AGX_CHIP_G16C:
   case AGX_CHIP_G17P:
      *encoding = AGX_ISA_ENCODING_AGX3B;
      return true;

   /* G17C and G17S share the exact g17s compiler target. */
   case AGX_CHIP_G17G:
   case AGX_CHIP_G17S:
   case AGX_CHIP_G17C:
      *encoding = AGX_ISA_ENCODING_AGX3C;
      return true;

   default:
      return false;
   }
}

static bool
d12_agx3_tile_target_encoding(enum agx_chip chip,
                              enum agx_isa_encoding *encoding)
{
   /* Do not infer tile state across identities: the pinned tile corpus has
    * exact artifacts only for G15G, G16G, and G17S, and AGX3b's high-register
    * f13 selection is still unresolved. */
   switch (chip) {
   case AGX_CHIP_G15G:
   case AGX_CHIP_G16G:
      *encoding = AGX_ISA_ENCODING_AGX3B;
      return true;
   case AGX_CHIP_G17S:
      *encoding = AGX_ISA_ENCODING_AGX3C;
      return true;
   default:
      return false;
   }
}

static bool
d12_agx3_subgroup_target_encoding(enum agx_chip chip,
                                  enum agx_isa_encoding *encoding)
{
   return d12_agx3_compiler_target_encoding(chip, encoding);
}

static bool
d12_agx3_map_subgroup_scope(enum d12_agx3_subgroup_scope scope, bool *quad)
{
   switch (scope) {
   case D12_AGX3_SUBGROUP_SCOPE_WAVE:
      *quad = false;
      return true;
   case D12_AGX3_SUBGROUP_SCOPE_QUAD:
      *quad = true;
      return true;
   default:
      return false;
   }
}

static bool
d12_agx3_map_subgroup_shuffle(
   enum d12_agx3_subgroup_shuffle_op op,
   enum agx3_subgroup_shuffle_op *shared)
{
   switch (op) {
   case D12_AGX3_SUBGROUP_SHUFFLE_READ_LANE_AT:
      *shared = AGX3_SUBGROUP_SHUFFLE_BROADCAST;
      return true;
   case D12_AGX3_SUBGROUP_SHUFFLE_XOR:
      *shared = AGX3_SUBGROUP_SHUFFLE_XOR;
      return true;
   case D12_AGX3_SUBGROUP_SHUFFLE_UP:
      *shared = AGX3_SUBGROUP_SHUFFLE_UP;
      return true;
   case D12_AGX3_SUBGROUP_SHUFFLE_DOWN:
      *shared = AGX3_SUBGROUP_SHUFFLE_DOWN;
      return true;
   default:
      return false;
   }
}

static bool
d12_agx3_map_subgroup_lane(
   enum d12_agx3_subgroup_lane_source source,
   enum agx3_subgroup_lane_kind *shared)
{
   switch (source) {
   case D12_AGX3_SUBGROUP_LANE_REGISTER:
      *shared = AGX3_SUBGROUP_LANE_REGISTER;
      return true;
   case D12_AGX3_SUBGROUP_LANE_UNIFORM:
      *shared = AGX3_SUBGROUP_LANE_UNIFORM;
      return true;
   default:
      return false;
   }
}

static bool
d12_agx3_map_subgroup_reduce(
   enum d12_agx3_subgroup_reduce_op op,
   enum agx3_subgroup_reduce_op *shared)
{
   switch (op) {
   case D12_AGX3_SUBGROUP_REDUCE_AND_U32:
      *shared = AGX3_SUBGROUP_REDUCE_AND;
      return true;
   case D12_AGX3_SUBGROUP_REDUCE_FADD_F32:
      *shared = AGX3_SUBGROUP_REDUCE_FADD;
      return true;
   case D12_AGX3_SUBGROUP_REDUCE_OR_U32:
      *shared = AGX3_SUBGROUP_REDUCE_OR;
      return true;
   case D12_AGX3_SUBGROUP_REDUCE_FMUL_F32:
      *shared = AGX3_SUBGROUP_REDUCE_FMUL;
      return true;
   case D12_AGX3_SUBGROUP_REDUCE_XOR_U32:
      *shared = AGX3_SUBGROUP_REDUCE_XOR;
      return true;
   case D12_AGX3_SUBGROUP_REDUCE_FMIN_F32:
      *shared = AGX3_SUBGROUP_REDUCE_FMIN;
      return true;
   case D12_AGX3_SUBGROUP_REDUCE_FMAX_F32:
      *shared = AGX3_SUBGROUP_REDUCE_FMAX;
      return true;
   case D12_AGX3_SUBGROUP_REDUCE_IADD_U32:
      *shared = AGX3_SUBGROUP_REDUCE_IADD;
      return true;
   case D12_AGX3_SUBGROUP_REDUCE_SMIN_I32:
      *shared = AGX3_SUBGROUP_REDUCE_SMIN;
      return true;
   case D12_AGX3_SUBGROUP_REDUCE_SMAX_I32:
      *shared = AGX3_SUBGROUP_REDUCE_SMAX;
      return true;
   case D12_AGX3_SUBGROUP_REDUCE_UMIN_U32:
      *shared = AGX3_SUBGROUP_REDUCE_UMIN;
      return true;
   case D12_AGX3_SUBGROUP_REDUCE_UMAX_U32:
      *shared = AGX3_SUBGROUP_REDUCE_UMAX;
      return true;
   default:
      return false;
   }
}

static bool
d12_agx3_fmul_compiler_target_encoding(enum agx_chip chip,
                                       enum agx_isa_encoding *encoding)
{
   switch (chip) {
   case AGX_CHIP_G15G:
   case AGX_CHIP_G15S:
   case AGX_CHIP_G15C:
   case AGX_CHIP_G15D:
   case AGX_CHIP_G16P:
   case AGX_CHIP_G16G:
   case AGX_CHIP_G16S:
   case AGX_CHIP_G16C:
   case AGX_CHIP_G17P:
      *encoding = AGX_ISA_ENCODING_AGX3B;
      return true;

   case AGX_CHIP_G17S:
   case AGX_CHIP_G17C:
      *encoding = AGX_ISA_ENCODING_AGX3C;
      return true;

   default:
      return false;
   }
}

static bool
d12_agx3_fmul_long_compiler_target_encoding(
   enum agx_chip chip, enum agx_isa_encoding *encoding)
{
   switch (chip) {
   case AGX_CHIP_G17G:
   case AGX_CHIP_G17S:
   case AGX_CHIP_G17C:
      *encoding = AGX_ISA_ENCODING_AGX3C;
      return true;

   default:
      return false;
   }
}

static bool
d12_agx3_fma_compiler_target_encoding(enum agx_chip chip,
                                      enum agx_isa_encoding *encoding)
{
   return d12_agx3_compiler_target_encoding(chip, encoding);
}

static bool
d12_agx3_fmin_compiler_target_encoding(enum agx_chip chip,
                                       enum agx_isa_encoding *encoding)
{
   return d12_agx3_compiler_target_encoding(chip, encoding);
}

static bool
d12_agx3_fmax_compiler_target_encoding(enum agx_chip chip,
                                       enum agx_isa_encoding *encoding)
{
   return d12_agx3_compiler_target_encoding(chip, encoding);
}

static bool
d12_agx3_cf_reg32(uint16_t reg)
{
   return (reg & 1) == 0 && reg < 128;
}

static bool
d12_agx3_map_i32_cf_predicate(enum d12_agx3_i32_cf_predicate predicate,
                              enum agx3_cf_icond *out)
{
   switch (predicate) {
   case D12_AGX3_I32_CF_EQ:
      *out = AGX3_CF_ICOND_EQ;
      return true;
   case D12_AGX3_I32_CF_ULT:
      *out = AGX3_CF_ICOND_ULT;
      return true;
   case D12_AGX3_I32_CF_UGT:
      *out = AGX3_CF_ICOND_UGT;
      return true;
   case D12_AGX3_I32_CF_SLT:
      *out = AGX3_CF_ICOND_SLT;
      return true;
   case D12_AGX3_I32_CF_SGT:
      *out = AGX3_CF_ICOND_SGT;
      return true;
   }

   return false;
}

static bool
d12_agx3_map_f32_cf_predicate(enum d12_agx3_f32_cf_predicate predicate,
                              enum agx3_cf_fcond *out, bool *or_nan,
                              bool *invert)
{
   *or_nan = false;

   switch (predicate) {
   case D12_AGX3_F32_CF_EQ:
      *out = AGX3_CF_FCOND_EQ;
      return true;
   case D12_AGX3_F32_CF_LT:
      *out = AGX3_CF_FCOND_LT;
      return true;
   case D12_AGX3_F32_CF_GT:
      *out = AGX3_CF_FCOND_GT;
      return true;
   case D12_AGX3_F32_CF_LT_OR_NAN:
      *out = AGX3_CF_FCOND_LT;
      *or_nan = true;
      return true;
   case D12_AGX3_F32_CF_GT_OR_NAN:
      *out = AGX3_CF_FCOND_GT;
      *or_nan = true;
      return true;
   case D12_AGX3_F32_CF_GE:
      *out = AGX3_CF_FCOND_LT;
      *or_nan = true;
      *invert = !*invert;
      return true;
   case D12_AGX3_F32_CF_LE:
      *out = AGX3_CF_FCOND_GT;
      *or_nan = true;
      *invert = !*invert;
      return true;
   }

   return false;
}

static bool
d12_agx3_append_cf(struct d12_agx3_cf_encoding *out, const uint8_t *bytes,
                   uint8_t length)
{
   if (out->length + length > sizeof(out->bytes))
      return false;

   memcpy(out->bytes + out->length, bytes, length);
   out->length += length;
   return true;
}

bool
d12_agx3_encode_i32_cf(enum agx_chip chip,
                       enum d12_agx3_cf_condition_op op,
                       enum d12_agx3_i32_cf_predicate predicate, bool invert,
                       uint16_t src0, bool src1_is_immediate,
                       uint16_t src1_or_imm,
                       struct d12_agx3_cf_encoding *out)
{
   if (!out)
      return false;
   memset(out, 0, sizeof(*out));

   enum agx_isa_encoding encoding;
   enum agx3_cf_icond condition;
   if (!d12_agx3_compiler_target_encoding(chip, &encoding) ||
       !d12_agx3_map_i32_cf_predicate(predicate, &condition) ||
       !d12_agx3_cf_reg32(src0) ||
       (op != D12_AGX3_CF_IF && op != D12_AGX3_CF_LOOP_OR_BREAK_TEST))
      return false;

   uint8_t bytes[AGX3_CF_MAX_LENGTH];
   uint8_t length = 0;
   const bool compare_ok =
      src1_is_immediate
         ? src1_or_imm <= UINT8_MAX &&
              agx3_cf_icmp_imm(encoding, condition, 0, src0,
                               (uint8_t)src1_or_imm, bytes, &length)
         : d12_agx3_cf_reg32(src1_or_imm) &&
              agx3_cf_icmp_reg(encoding, condition, 0, src0, src1_or_imm,
                               bytes, &length);
   if (!compare_ok || !d12_agx3_append_cf(out, bytes, length))
      goto fail;

   const bool cnd_ok =
      op == D12_AGX3_CF_IF
         ? agx3_cf_cndst(encoding, 0, invert, 1, bytes, &length)
         : agx3_cf_cndlt(encoding, 0, invert, bytes, &length);
   if (!cnd_ok || !d12_agx3_append_cf(out, bytes, length))
      goto fail;

   return true;

fail:
   memset(out, 0, sizeof(*out));
   return false;
}

bool
d12_agx3_encode_f32_cf(enum agx_chip chip,
                       enum d12_agx3_cf_condition_op op,
                       enum d12_agx3_f32_cf_predicate predicate, bool invert,
                       uint16_t src0, uint16_t src1,
                       struct d12_agx3_cf_encoding *out)
{
   if (!out)
      return false;
   memset(out, 0, sizeof(*out));

   enum agx_isa_encoding encoding;
   enum agx3_cf_fcond condition;
   bool or_nan;
   if (!d12_agx3_compiler_target_encoding(chip, &encoding) ||
       !d12_agx3_map_f32_cf_predicate(predicate, &condition, &or_nan,
                                      &invert) ||
       !d12_agx3_cf_reg32(src0) || !d12_agx3_cf_reg32(src1) ||
       (op != D12_AGX3_CF_IF && op != D12_AGX3_CF_LOOP_OR_BREAK_TEST))
      return false;

   uint8_t bytes[AGX3_CF_MAX_LENGTH];
   uint8_t length = 0;
   if (!agx3_cf_fcmp_reg(encoding, condition, or_nan, 0, src0, src1, bytes,
                         &length) ||
       !d12_agx3_append_cf(out, bytes, length))
      goto fail;

   const bool cnd_ok =
      op == D12_AGX3_CF_IF
         ? agx3_cf_cndst(encoding, 0, invert, 1, bytes, &length)
         : agx3_cf_cndlt(encoding, 0, invert, bytes, &length);
   if (!cnd_ok || !d12_agx3_append_cf(out, bytes, length))
      goto fail;

   return true;

fail:
   memset(out, 0, sizeof(*out));
   return false;
}

bool
d12_agx3_encode_cf_stack(enum agx_chip chip, enum d12_agx3_cf_stack_op op,
                         struct d12_agx3_cf_encoding *out)
{
   if (!out)
      return false;
   memset(out, 0, sizeof(*out));

   enum agx_isa_encoding encoding;
   if (!d12_agx3_compiler_target_encoding(chip, &encoding))
      return false;

   uint8_t bytes[AGX3_CF_MAX_LENGTH];
   uint8_t length = 0;
   bool ok;
   switch (op) {
   case D12_AGX3_CF_LOOP_ENTRY:
      ok = agx3_cf_cndst(encoding, 6, false, 2, bytes, &length);
      break;
   case D12_AGX3_CF_ELSE:
      ok = agx3_cf_cndef(encoding, bytes, &length);
      break;
   case D12_AGX3_CF_POP_1:
      ok = agx3_cf_cndend(encoding, 1, bytes, &length);
      break;
   case D12_AGX3_CF_POP_2:
      ok = agx3_cf_cndend(encoding, 2, bytes, &length);
      break;
   default:
      return false;
   }

   if (!ok || !d12_agx3_append_cf(out, bytes, length)) {
      memset(out, 0, sizeof(*out));
      return false;
   }
   return true;
}

bool
d12_agx3_encode_cf_branch(enum agx_chip chip,
                          enum d12_agx3_cf_branch_op op,
                          int64_t byte_displacement,
                          struct d12_agx3_cf_encoding *out)
{
   if (!out)
      return false;
   memset(out, 0, sizeof(*out));

   enum agx_isa_encoding encoding;
   if (!d12_agx3_compiler_target_encoding(chip, &encoding) ||
       (op != D12_AGX3_CF_BRANCH_EXEC_NONE &&
        op != D12_AGX3_CF_BRANCH_EXEC_ANY))
      return false;

   const bool is_bn = op == D12_AGX3_CF_BRANCH_EXEC_NONE;
   if ((is_bn && byte_displacement <= 0) ||
       (!is_bn && byte_displacement >= 0))
      return false;

   uint8_t bytes[AGX3_CF_MAX_LENGTH];
   uint8_t length = 0;
   if (!agx3_cf_branch(encoding, is_bn, bytes, &length) ||
       !agx3_cf_patch_branch(encoding, is_bn, bytes, byte_displacement) ||
       !d12_agx3_append_cf(out, bytes, length)) {
      memset(out, 0, sizeof(*out));
      return false;
   }
   return true;
}

static bool
d12_agx3_finish_vary(struct d12_agx3_vary_encoding *out, bool ok,
                     const uint8_t bytes[AGX3_VARY_MAX_LENGTH], uint8_t length)
{
   if (!ok || length > sizeof(out->bytes)) {
      memset(out, 0, sizeof(*out));
      return false;
   }

   memcpy(out->bytes, bytes, length);
   out->length = length;
   return true;
}

bool
d12_agx3_encode_vary_interp(enum agx_chip chip, uint16_t dest,
                            uint16_t coefficient, bool perspective,
                            enum d12_agx3_vary_location location,
                            struct d12_agx3_vary_encoding *out)
{
   if (!out)
      return false;
   memset(out, 0, sizeof(*out));

   enum agx_isa_encoding encoding;
   enum agx3_vary_location apple_location;
   if (!d12_agx3_compiler_target_encoding(chip, &encoding))
      return false;

   switch (location) {
   case D12_AGX3_VARY_CENTER:
      apple_location = AGX3_VARY_CENTER;
      break;
   case D12_AGX3_VARY_CENTROID:
      apple_location = AGX3_VARY_CENTROID;
      break;
   default:
      return false;
   }

   uint8_t bytes[AGX3_VARY_MAX_LENGTH];
   uint8_t length = 0;
   const bool ok = agx3_vary_interp_at(encoding, dest, coefficient,
                                       perspective, apple_location, bytes,
                                       &length);
   return d12_agx3_finish_vary(out, ok, bytes, length);
}

bool
d12_agx3_encode_vary_ldcf(enum agx_chip chip, uint16_t dest,
                          uint16_t coefficient, uint8_t slot,
                          struct d12_agx3_vary_encoding *out)
{
   if (!out)
      return false;
   memset(out, 0, sizeof(*out));

   enum agx_isa_encoding encoding;
   if (!d12_agx3_compiler_target_encoding(chip, &encoding))
      return false;

   uint8_t bytes[AGX3_VARY_MAX_LENGTH];
   uint8_t length = 0;
   const bool ok =
      agx3_vary_ldcf(encoding, dest, coefficient, slot, bytes, &length);
   return d12_agx3_finish_vary(out, ok, bytes, length);
}

bool
d12_agx3_encode_vary_uvsw(enum agx_chip chip, uint16_t value,
                          uint16_t uvs_index, uint8_t slot, bool value_kill,
                          uint16_t wait_mask,
                          struct d12_agx3_vary_encoding *out)
{
   if (!out)
      return false;
   memset(out, 0, sizeof(*out));

   enum agx_isa_encoding encoding;
   if (!d12_agx3_compiler_target_encoding(chip, &encoding))
      return false;

   uint8_t bytes[AGX3_VARY_MAX_LENGTH];
   uint8_t length = 0;
   const bool ok = agx3_vary_uvsw(encoding, value, uvs_index, slot,
                                  value_kill, wait_mask, bytes, &length);
   return d12_agx3_finish_vary(out, ok, bytes, length);
}

bool
d12_agx3_encode_texture_fetch(
   enum agx_chip chip, const struct d12_agx3_texture_fetch_desc *desc,
   struct d12_agx3_texture_fetch_encoding *out)
{
   if (!out)
      return false;
   memset(out, 0, sizeof(*out));

   enum agx_isa_encoding encoding;
   if (!desc || !d12_agx3_compiler_target_encoding(chip, &encoding))
      return false;

   enum agx3_tex_width width;
   if (desc->width == D12_AGX3_TEXTURE_RESULT_16)
      width = AGX3_TEX_W16;
   else if (desc->width == D12_AGX3_TEXTURE_RESULT_32)
      width = AGX3_TEX_W32;
   else
      return false;

   const struct agx3_tex_fetch_desc d = {
      .dest = desc->dest,
      .coord = desc->coord,
      .width = width,
      .mask = desc->mask,
      .texture = desc->texture,
      .slot = desc->slot,
      .wait_mask = desc->wait_mask,
      .coord_kill = desc->coord_kill,
      .final = desc->final,
   };
   uint8_t length = 0;
   if (!agx3_tex_encode_fetch(encoding, &d, out->bytes, &length) ||
       length > sizeof(out->bytes)) {
      memset(out, 0, sizeof(*out));
      return false;
   }
   out->length = length;
   return true;
}

bool
d12_agx3_decode_texture_fetch(
   enum agx_chip chip, const void *instruction, uint32_t instruction_size,
   struct d12_agx3_texture_fetch_desc *out)
{
   if (!out)
      return false;
   memset(out, 0, sizeof(*out));

   enum agx_isa_encoding encoding;
   struct agx3_tex_fetch_desc d;
   if (!d12_agx3_compiler_target_encoding(chip, &encoding) ||
       !agx3_tex_decode_fetch(encoding, instruction, instruction_size, &d))
      return false;

   out->dest = d.dest;
   out->coord = d.coord;
   out->width = d.width == AGX3_TEX_W16 ? D12_AGX3_TEXTURE_RESULT_16
                                        : D12_AGX3_TEXTURE_RESULT_32;
   out->mask = d.mask;
   out->texture = d.texture;
   out->slot = d.slot;
   out->wait_mask = d.wait_mask;
   out->coord_kill = d.coord_kill;
   out->final = d.final;
   return true;
}

static bool
d12_agx3_tile_width(enum d12_agx3_tile_width in,
                    enum agx3_tile_width *out)
{
   switch (in) {
   case D12_AGX3_TILE_WIDTH_8:
      *out = AGX3_TILE_W8;
      return true;
   case D12_AGX3_TILE_WIDTH_16:
      *out = AGX3_TILE_W16;
      return true;
   case D12_AGX3_TILE_WIDTH_32:
      *out = AGX3_TILE_W32;
      return true;
   default:
      return false;
   }
}

bool
d12_agx3_encode_tile_store(enum agx_chip chip,
                           const struct d12_agx3_tile_store_desc *desc,
                           struct d12_agx3_tile_encoding *out)
{
   if (!out)
      return false;
   memset(out, 0, sizeof(*out));

   enum agx_isa_encoding encoding;
   enum agx3_tile_width width;
   uint8_t length = 0;
   if (!desc || !d12_agx3_tile_target_encoding(chip, &encoding) ||
       !d12_agx3_tile_width(desc->width, &width) ||
       (encoding == AGX_ISA_ENCODING_AGX3B && desc->data >= 16) ||
       !agx3_tile_store(encoding, desc->data, width, desc->mask,
                        desc->tile_offset, AGX3_TILE_SAMPLE_1X, desc->wait_mask,
                        out->bytes, &length) ||
       length > sizeof(out->bytes)) {
      memset(out, 0, sizeof(*out));
      return false;
   }

   out->length = length;
   return true;
}

bool
d12_agx3_encode_tile_load(enum agx_chip chip,
                          const struct d12_agx3_tile_load_desc *desc,
                          struct d12_agx3_tile_encoding *out)
{
   if (!out)
      return false;
   memset(out, 0, sizeof(*out));

   enum agx_isa_encoding encoding;
   enum agx3_tile_width width;
   uint8_t length = 0;
   if (!desc || !d12_agx3_tile_target_encoding(chip, &encoding) ||
       !d12_agx3_tile_width(desc->width, &width) ||
       !agx3_tile_load(encoding, desc->dest, width, desc->slot, desc->mask,
                       desc->tile_offset, AGX3_TILE_SAMPLE_1X, 0, desc->first,
                       desc->last, out->bytes, &length) ||
       length > sizeof(out->bytes)) {
      memset(out, 0, sizeof(*out));
      return false;
   }

   out->length = length;
   return true;
}

static bool
d12_agx3_memory_width(enum d12_agx3_memory_width in,
                      enum agx3_mem_width *out)
{
   switch (in) {
   case D12_AGX3_MEMORY_WIDTH_8:
      *out = AGX3_MEM_W8;
      return true;
   case D12_AGX3_MEMORY_WIDTH_16:
      *out = AGX3_MEM_W16;
      return true;
   case D12_AGX3_MEMORY_WIDTH_32:
      *out = AGX3_MEM_W32;
      return true;
   default:
      return false;
   }
}

static bool
d12_agx3_threadgroup_base(enum d12_agx3_threadgroup_base_kind in,
                          enum agx3_local_base_kind *out)
{
   switch (in) {
   case D12_AGX3_THREADGROUP_BASE_IMMEDIATE:
      *out = AGX3_LOCAL_BASE_IMMEDIATE;
      return true;
   case D12_AGX3_THREADGROUP_BASE_REGISTER:
      *out = AGX3_LOCAL_BASE_REGISTER;
      return true;
   case D12_AGX3_THREADGROUP_BASE_UNIFORM:
      *out = AGX3_LOCAL_BASE_UNIFORM;
      return true;
   default:
      return false;
   }
}

static bool
d12_agx3_encode_threadgroup(enum agx_chip chip,
                            const struct d12_agx3_threadgroup_desc *desc,
                            bool is_load,
                            struct d12_agx3_local_tps_encoding *out)
{
   if (!out)
      return false;
   memset(out, 0, sizeof(*out));

   enum agx_isa_encoding encoding;
   enum agx3_mem_width width;
   enum agx3_local_base_kind base_kind;
   if (!desc || !d12_agx3_compiler_target_encoding(chip, &encoding) ||
       !d12_agx3_memory_width(desc->width, &width) ||
       !d12_agx3_threadgroup_base(desc->base_kind, &base_kind) ||
       (is_load &&
        (desc->value_kill || desc->width == D12_AGX3_MEMORY_WIDTH_8)) ||
       (desc->base_kind != D12_AGX3_THREADGROUP_BASE_REGISTER &&
        desc->base_kill) ||
       (!desc->index_is_register && desc->index_kill))
      return false;

   const struct agx3_local_desc d = {
      .data = desc->data,
      .width = width,
      .mask = desc->mask,
      .shift = desc->index_scale_log2,
      .base_kind = base_kind,
      .base = desc->base,
      .base_kill = desc->base_kill,
      .index_is_register = desc->index_is_register,
      .index = desc->index,
      .index_kill = desc->index_kill,
      .byte_offset = desc->byte_offset,
      .value_kill = desc->value_kill,
      .slot = desc->slot,
      .wait_mask = desc->wait_mask,
   };

   uint8_t length = 0;
   const bool ok = is_load ? agx3_local_encode_load(encoding, &d, out->bytes,
                                                     &length)
                           : agx3_local_encode_store(encoding, &d, out->bytes,
                                                      &length);
   if (!ok || length > sizeof(out->bytes)) {
      memset(out, 0, sizeof(*out));
      return false;
   }

   out->length = length;
   return true;
}

bool
d12_agx3_encode_threadgroup_load(
   enum agx_chip chip, const struct d12_agx3_threadgroup_desc *desc,
   struct d12_agx3_local_tps_encoding *out)
{
   return d12_agx3_encode_threadgroup(chip, desc, true, out);
}

bool
d12_agx3_encode_threadgroup_store(
   enum agx_chip chip, const struct d12_agx3_threadgroup_desc *desc,
   struct d12_agx3_local_tps_encoding *out)
{
   return d12_agx3_encode_threadgroup(chip, desc, false, out);
}

static bool
d12_agx3_encode_tps(enum agx_chip chip, const struct d12_agx3_tps_desc *desc,
                    bool is_load, struct d12_agx3_local_tps_encoding *out)
{
   if (!out)
      return false;
   memset(out, 0, sizeof(*out));

   enum agx_isa_encoding encoding;
   enum agx3_mem_width width;
   if (!desc || !d12_agx3_compiler_target_encoding(chip, &encoding) ||
       !d12_agx3_memory_width(desc->width, &width) ||
       (is_load &&
        (desc->value_kill || desc->width == D12_AGX3_MEMORY_WIDTH_8)))
      return false;

   const struct agx3_tps_desc d = {
      .data = desc->data,
      .width = width,
      .mask = desc->mask,
      .shift = desc->index_scale_log2,
      .byte_offset = desc->byte_offset,
      .index = desc->index,
      .index_kill = desc->index_kill,
      .index_signed = desc->index_signed,
      .frame_region = desc->frame_region,
      .value_kill = desc->value_kill,
      .slot = desc->slot,
      .wait_mask = desc->wait_mask,
   };

   uint8_t length = 0;
   const bool ok = is_load ? agx3_tps_encode_load(encoding, &d, out->bytes,
                                                   &length)
                           : agx3_tps_encode_store(encoding, &d, out->bytes,
                                                    &length);
   if (!ok || length > sizeof(out->bytes)) {
      memset(out, 0, sizeof(*out));
      return false;
   }

   out->length = length;
   return true;
}

bool
d12_agx3_encode_tps_load(enum agx_chip chip,
                         const struct d12_agx3_tps_desc *desc,
                         struct d12_agx3_local_tps_encoding *out)
{
   return d12_agx3_encode_tps(chip, desc, true, out);
}

bool
d12_agx3_encode_tps_store(enum agx_chip chip,
                          const struct d12_agx3_tps_desc *desc,
                          struct d12_agx3_local_tps_encoding *out)
{
   return d12_agx3_encode_tps(chip, desc, false, out);
}

static bool
d12_agx3_static_spill_input(
   enum agx_chip chip, const struct d12_agx3_static_spill_desc *desc,
   bool is_load, enum agx_isa_encoding *encoding, uint32_t *tps_half)
{
   if (desc == NULL ||
       !d12_agx3_compiler_target_encoding(chip, encoding) ||
       (desc->byte_offset & 3u) != 0 ||
       desc->byte_offset / 2u > AGX3_STATIC_SPILL_MAX_TPS_HALF ||
       (is_load && desc->reg_kill))
      return false;

   *tps_half = desc->byte_offset / 2u;
   return true;
}

static bool
d12_agx3_encode_static_spill(
   enum agx_chip chip, const struct d12_agx3_static_spill_desc *desc,
   bool is_load, struct d12_agx3_static_spill_encoding *out)
{
   if (out == NULL)
      return false;
   memset(out, 0, sizeof(*out));

   enum agx_isa_encoding encoding;
   uint32_t tps_half;
   if (!d12_agx3_static_spill_input(chip, desc, is_load, &encoding,
                                    &tps_half))
      return false;

   uint8_t length = 0;
   bool ok;
   if (is_load) {
      const struct agx3_static_spill_load_desc d = {
         .dest = desc->reg,
         .tps_half = tps_half,
         .width = AGX3_MEM_W32,
         .slot = desc->slot,
         .slot_mirror = desc->slot,
         .wait_mask = desc->wait_mask,
      };
      ok = agx3_static_spill_encode_load(encoding, &d, out->bytes, &length);
   } else {
      const struct agx3_static_spill_store_desc d = {
         .tps_half = tps_half,
         .src = desc->reg,
         .width = AGX3_MEM_W32,
         .src_kill = desc->reg_kill,
         .slot = desc->slot,
         .wait_mask = desc->wait_mask,
      };
      ok = agx3_static_spill_encode_store(encoding, &d, out->bytes, &length);
   }

   if (!ok || length > sizeof(out->bytes)) {
      memset(out, 0, sizeof(*out));
      return false;
   }
   out->length = length;
   return true;
}

bool
d12_agx3_encode_static_spill_store(
   enum agx_chip chip, const struct d12_agx3_static_spill_desc *desc,
   struct d12_agx3_static_spill_encoding *out)
{
   return d12_agx3_encode_static_spill(chip, desc, false, out);
}

bool
d12_agx3_encode_static_spill_load(
   enum agx_chip chip, const struct d12_agx3_static_spill_desc *desc,
   struct d12_agx3_static_spill_encoding *out)
{
   return d12_agx3_encode_static_spill(chip, desc, true, out);
}

bool
d12_agx3_decode_static_spill_store(
   enum agx_chip chip, const void *instruction, size_t length,
   struct d12_agx3_static_spill_desc *out)
{
   if (out == NULL)
      return false;
   memset(out, 0, sizeof(*out));

   enum agx_isa_encoding encoding;
   struct agx3_static_spill_store_desc d;
   if (!d12_agx3_compiler_target_encoding(chip, &encoding) ||
       !agx3_static_spill_decode_store(encoding, instruction, length, &d) ||
       d.width != AGX3_MEM_W32 || d.structural_flag)
      return false;

   *out = (struct d12_agx3_static_spill_desc){
      .reg = d.src,
      .byte_offset = d.tps_half * 2u,
      .reg_kill = d.src_kill,
      .slot = d.slot,
      .wait_mask = d.wait_mask,
   };
   return true;
}

bool
d12_agx3_decode_static_spill_load(
   enum agx_chip chip, const void *instruction, size_t length,
   struct d12_agx3_static_spill_desc *out)
{
   if (out == NULL)
      return false;
   memset(out, 0, sizeof(*out));

   enum agx_isa_encoding encoding;
   struct agx3_static_spill_load_desc d;
   if (!d12_agx3_compiler_target_encoding(chip, &encoding) ||
       !agx3_static_spill_decode_load(encoding, instruction, length, &d) ||
       d.width != AGX3_MEM_W32 || d.slot_mirror != d.slot ||
       d.narrow_extend || d.structural_flag)
      return false;

   *out = (struct d12_agx3_static_spill_desc){
      .reg = d.dest,
      .byte_offset = d.tps_half * 2u,
      .slot = d.slot,
      .wait_mask = d.wait_mask,
   };
   return true;
}

bool
d12_agx3_encode_binary_alu(
   enum agx_isa_encoding encoding, enum d12_agx3_binary_alu_op op,
   uint16_t dest, uint16_t src0, uint16_t src1,
   struct d12_agx3_binary_alu_encoding *out)
{
   if (!out)
      return false;

   memset(out, 0, sizeof(*out));

   switch (op) {
   case D12_AGX3_BINARY_ALU_IADD_U32: {
      struct agx3_encoded_iadd_u32_reg_reg encoded;
      if (!agx3_pack_iadd_u32_reg_reg(encoding, dest, src0, src1, &encoded))
         return false;

      memcpy(out->bytes, encoded.bytes, encoded.length);
      out->length = encoded.length;
      return true;
   }

   case D12_AGX3_BINARY_ALU_IAND_U32: {
      struct agx3_encoded_bit_and_u32_reg_reg encoded;
      if (!agx3_pack_bit_and_u32_reg_reg(encoding, dest, src0, src1,
                                         &encoded))
         return false;

      memcpy(out->bytes, encoded.bytes, encoded.length);
      out->length = encoded.length;
      return true;
   }

   case D12_AGX3_BINARY_ALU_IOR_U32: {
      struct agx3_encoded_bit_or_u32_reg_reg encoded;
      if (!agx3_pack_bit_or_u32_reg_reg(encoding, dest, src0, src1,
                                        &encoded))
         return false;

      memcpy(out->bytes, encoded.bytes, encoded.length);
      out->length = encoded.length;
      return true;
   }

   case D12_AGX3_BINARY_ALU_IXOR_U32: {
      struct agx3_encoded_bit_xor_u32_reg_reg encoded;
      if (!agx3_pack_bit_xor_u32_reg_reg(encoding, dest, src0, src1,
                                         &encoded))
         return false;

      memcpy(out->bytes, encoded.bytes, encoded.length);
      out->length = encoded.length;
      return true;
   }

   case D12_AGX3_BINARY_ALU_ISUB_U32: {
      struct agx3_encoded_isub_u32_reg_reg encoded;
      if (!agx3_pack_isub_u32_reg_reg(encoding, dest, src0, src1,
                                      &encoded))
         return false;

      memcpy(out->bytes, encoded.bytes, encoded.length);
      out->length = encoded.length;
      return true;
   }

   case D12_AGX3_BINARY_ALU_ISHL_U32: {
      struct agx3_encoded_sll_u32_reg_reg encoded;
      if (!agx3_pack_sll_u32_reg_reg(encoding, dest, src0, src1, &encoded))
         return false;

      memcpy(out->bytes, encoded.bytes, encoded.length);
      out->length = encoded.length;
      return true;
   }

   case D12_AGX3_BINARY_ALU_IASHR_I32: {
      struct agx3_encoded_asr_i32_reg_reg encoded;
      if (!agx3_pack_asr_i32_reg_reg(encoding, dest, src0, src1, &encoded))
         return false;

      memcpy(out->bytes, encoded.bytes, encoded.length);
      out->length = encoded.length;
      return true;
   }

   case D12_AGX3_BINARY_ALU_ILSHR_U32: {
      struct agx3_encoded_lsr_u32_reg_reg encoded;
      if (!agx3_pack_lsr_u32_reg_reg(encoding, dest, src0, src1, &encoded))
         return false;

      memcpy(out->bytes, encoded.bytes, encoded.length);
      out->length = encoded.length;
      return true;
   }

   case D12_AGX3_BINARY_ALU_IMUL_U32: {
      struct agx3_encoded_imul_u32_reg_reg encoded;
      if (!agx3_pack_imul_u32_reg_reg(encoding, dest, src0, src1, &encoded))
         return false;

      memcpy(out->bytes, encoded.bytes, encoded.length);
      out->length = encoded.length;
      return true;
   }

   default:
      return false;
   }
}

bool
d12_agx3_encode_integer_multiply(
   enum agx_chip chip, enum d12_agx3_integer_multiply_selection selection,
   uint16_t dest, uint16_t src0, uint16_t src1,
   struct d12_agx3_binary_alu_encoding *out)
{
   if (!out)
      return false;

   memset(out, 0, sizeof(*out));

   if (selection != D12_AGX3_INTEGER_MULTIPLY_LOW_U32)
      return false;

   enum agx_isa_encoding encoding;
   if (!d12_agx3_compiler_target_encoding(chip, &encoding))
      return false;

   return d12_agx3_encode_binary_alu(
      encoding, D12_AGX3_BINARY_ALU_IMUL_U32, dest, src0, src1, out);
}

bool
d12_agx3_encode_ternary_alu(
   enum agx_isa_encoding encoding, enum d12_agx3_ternary_alu_op op,
   uint16_t dest, uint16_t src0, uint16_t src1, uint16_t src2,
   struct d12_agx3_ternary_alu_encoding *out)
{
   if (!out)
      return false;

   memset(out, 0, sizeof(*out));

   switch (op) {
   case D12_AGX3_TERNARY_ALU_IMADD_U32: {
      struct agx3_encoded_imadd_u32_reg_reg_reg encoded;
      if (!agx3_pack_imadd_u32_reg_reg_reg(encoding, dest, src0, src1, src2,
                                           &encoded))
         return false;

      memcpy(out->bytes, encoded.bytes, encoded.length);
      out->length = encoded.length;
      return true;
   }

   default:
      return false;
   }
}

bool
d12_agx3_encode_integer_multiply_add(
   enum agx_chip chip,
   enum d12_agx3_integer_multiply_add_selection selection, uint16_t dest,
   uint16_t mul0, uint16_t mul1, uint16_t addend,
   struct d12_agx3_ternary_alu_encoding *out)
{
   if (!out)
      return false;

   memset(out, 0, sizeof(*out));

   if (selection != D12_AGX3_INTEGER_MULTIPLY_ADD_LOW_U32)
      return false;

   enum agx_isa_encoding encoding;
   if (!d12_agx3_compiler_target_encoding(chip, &encoding))
      return false;

   return d12_agx3_encode_ternary_alu(
      encoding, D12_AGX3_TERNARY_ALU_IMADD_U32, dest, mul0, mul1, addend,
      out);
}

bool
d12_agx3_encode_fadd_f32_reg_reg(
   enum agx_chip chip, enum agx_isa_encoding encoding, uint16_t dest,
   uint16_t src0, uint16_t src1,
   struct d12_agx3_fadd_f32_reg_reg_encoding *out)
{
   if (!out)
      return false;

   memset(out, 0, sizeof(*out));

   enum agx_isa_encoding target_encoding;
   if (!d12_agx3_compiler_target_encoding(chip, &target_encoding) ||
       encoding != target_encoding)
      return false;

   struct agx3_encoded_fadd_f32_reg_reg encoded;
   if (!agx3_pack_fadd_f32_reg_reg(encoding, dest, src0, src1, &encoded) ||
       encoded.length != sizeof(out->bytes))
      return false;

   memcpy(out->bytes, encoded.bytes, encoded.length);
   out->length = encoded.length;
   return true;
}

bool
d12_agx3_decode_fadd_f32_reg_reg(
   enum agx_chip chip, enum agx_isa_encoding encoding,
   const void *instruction, uint32_t instruction_size, uint16_t *dest,
   uint16_t *src0, uint16_t *src1)
{
   if (dest)
      *dest = 0;
   if (src0)
      *src0 = 0;
   if (src1)
      *src1 = 0;

   if (!instruction || !dest || !src0 || !src1 ||
       instruction_size != D12_AGX3_FADD_F32_REG_REG_LENGTH)
      return false;

   enum agx_isa_encoding target_encoding;
   if (!d12_agx3_compiler_target_encoding(chip, &target_encoding) ||
       encoding != target_encoding)
      return false;

   uint16_t decoded_dest = 0, decoded_src0 = 0, decoded_src1 = 0;
   uint8_t decoded_length = 0;
   if (!agx3_unpack_fadd_f32_reg_reg(
          encoding, instruction, instruction_size, &decoded_dest,
          &decoded_src0, &decoded_src1, &decoded_length) ||
       decoded_length != D12_AGX3_FADD_F32_REG_REG_LENGTH)
      return false;

   *dest = decoded_dest;
   *src0 = decoded_src0;
   *src1 = decoded_src1;
   return true;
}

bool
d12_agx3_encode_fmul_f32_reg_reg(
   enum agx_chip chip, enum agx_isa_encoding encoding, uint16_t dest,
   uint16_t src0, uint16_t src1,
   struct d12_agx3_fmul_f32_reg_reg_encoding *out)
{
   if (!out)
      return false;

   memset(out, 0, sizeof(*out));

   enum agx_isa_encoding target_encoding;
   if (!d12_agx3_fmul_compiler_target_encoding(chip, &target_encoding) ||
       encoding != target_encoding)
      return false;

   struct agx3_encoded_fmul_f32_reg_reg encoded;
   if (!agx3_pack_fmul_f32_reg_reg(encoding, dest, src0, src1, &encoded) ||
       encoded.length != sizeof(out->bytes))
      return false;

   memcpy(out->bytes, encoded.bytes, encoded.length);
   out->length = encoded.length;
   return true;
}

bool
d12_agx3_decode_fmul_f32_reg_reg(
   enum agx_chip chip, enum agx_isa_encoding encoding,
   const void *instruction, uint32_t instruction_size, uint16_t *dest,
   uint16_t *src0, uint16_t *src1)
{
   if (dest)
      *dest = 0;
   if (src0)
      *src0 = 0;
   if (src1)
      *src1 = 0;

   if (!instruction || !dest || !src0 || !src1 ||
       instruction_size != D12_AGX3_FMUL_F32_REG_REG_LENGTH)
      return false;

   enum agx_isa_encoding target_encoding;
   if (!d12_agx3_fmul_compiler_target_encoding(chip, &target_encoding) ||
       encoding != target_encoding)
      return false;

   uint16_t decoded_dest = 0, decoded_src0 = 0, decoded_src1 = 0;
   uint8_t decoded_length = 0;
   if (!agx3_unpack_fmul_f32_reg_reg(
          encoding, instruction, instruction_size, &decoded_dest,
          &decoded_src0, &decoded_src1, &decoded_length) ||
       decoded_length != D12_AGX3_FMUL_F32_REG_REG_LENGTH)
      return false;

   *dest = decoded_dest;
   *src0 = decoded_src0;
   *src1 = decoded_src1;
   return true;
}

bool
d12_agx3_encode_fmul_f32_reg_reg_long(
   enum agx_chip chip, enum agx_isa_encoding encoding, uint16_t dest,
   uint16_t src0, uint16_t src1,
   struct d12_agx3_fmul_f32_reg_reg_long_encoding *out)
{
   if (!out)
      return false;

   memset(out, 0, sizeof(*out));

   enum agx_isa_encoding target_encoding;
   if (!d12_agx3_fmul_long_compiler_target_encoding(
          chip, &target_encoding) ||
       encoding != target_encoding)
      return false;

   struct agx3_encoded_fmul_f32_reg_reg_long encoded;
   if (!agx3_pack_fmul_f32_reg_reg_long(
          encoding, dest, src0, src1, &encoded) ||
       encoded.length != sizeof(out->bytes))
      return false;

   memcpy(out->bytes, encoded.bytes, encoded.length);
   out->length = encoded.length;
   return true;
}

bool
d12_agx3_decode_fmul_f32_reg_reg_long(
   enum agx_chip chip, enum agx_isa_encoding encoding,
   const void *instruction, uint32_t instruction_size, uint16_t *dest,
   uint16_t *src0, uint16_t *src1)
{
   if (dest)
      *dest = 0;
   if (src0)
      *src0 = 0;
   if (src1)
      *src1 = 0;

   if (!instruction || !dest || !src0 || !src1 ||
       instruction_size != D12_AGX3_FMUL_F32_REG_REG_LONG_LENGTH)
      return false;

   enum agx_isa_encoding target_encoding;
   if (!d12_agx3_fmul_long_compiler_target_encoding(
          chip, &target_encoding) ||
       encoding != target_encoding)
      return false;

   uint16_t decoded_dest = 0, decoded_src0 = 0, decoded_src1 = 0;
   uint8_t decoded_length = 0;
   if (!agx3_unpack_fmul_f32_reg_reg_long(
          encoding, instruction, instruction_size, &decoded_dest,
          &decoded_src0, &decoded_src1, &decoded_length) ||
       decoded_length != D12_AGX3_FMUL_F32_REG_REG_LONG_LENGTH)
      return false;

   *dest = decoded_dest;
   *src0 = decoded_src0;
   *src1 = decoded_src1;
   return true;
}

bool
d12_agx3_encode_fma_f32_reg_reg_reg(
   enum agx_chip chip, enum agx_isa_encoding encoding, uint16_t dest,
   uint16_t mul0, uint16_t mul1, uint16_t addend,
   struct d12_agx3_fma_f32_reg_reg_reg_encoding *out)
{
   if (!out)
      return false;

   memset(out, 0, sizeof(*out));

   enum agx_isa_encoding target_encoding;
   if (!d12_agx3_fma_compiler_target_encoding(chip, &target_encoding) ||
       encoding != target_encoding)
      return false;

   struct agx3_encoded_fma_f32_reg_reg_reg encoded;
   if (!agx3_pack_fma_f32_reg_reg_reg(
          encoding, dest, mul0, mul1, addend, &encoded) ||
       encoded.length > sizeof(out->bytes))
      return false;

   memcpy(out->bytes, encoded.bytes, encoded.length);
   out->length = encoded.length;
   return true;
}

bool
d12_agx3_decode_fma_f32_reg_reg_reg(
   enum agx_chip chip, enum agx_isa_encoding encoding,
   const void *instruction, uint32_t instruction_size, uint16_t *dest,
   uint16_t *mul0, uint16_t *mul1, uint16_t *addend)
{
   if (dest)
      *dest = 0;
   if (mul0)
      *mul0 = 0;
   if (mul1)
      *mul1 = 0;
   if (addend)
      *addend = 0;

   const uint32_t expected_length =
      encoding == AGX_ISA_ENCODING_AGX3B ? 8 :
      encoding == AGX_ISA_ENCODING_AGX3C ? 16 : 0;
   if (!instruction || !dest || !mul0 || !mul1 || !addend ||
       expected_length == 0 || instruction_size != expected_length)
      return false;

   enum agx_isa_encoding target_encoding;
   if (!d12_agx3_fma_compiler_target_encoding(chip, &target_encoding) ||
       encoding != target_encoding)
      return false;

   uint16_t decoded_dest = 0, decoded_mul0 = 0, decoded_mul1 = 0;
   uint16_t decoded_addend = 0;
   uint8_t decoded_length = 0;
   if (!agx3_unpack_fma_f32_reg_reg_reg(
          encoding, instruction, instruction_size, &decoded_dest,
          &decoded_mul0, &decoded_mul1, &decoded_addend, &decoded_length) ||
       decoded_length != expected_length)
      return false;

   *dest = decoded_dest;
   *mul0 = decoded_mul0;
   *mul1 = decoded_mul1;
   *addend = decoded_addend;
   return true;
}

bool
d12_agx3_encode_fmin_f32_reg_reg(
   enum agx_chip chip, enum agx_isa_encoding encoding, uint16_t dest,
   uint16_t src0, uint16_t src1,
   struct d12_agx3_fmin_f32_reg_reg_encoding *out)
{
   if (!out)
      return false;

   memset(out, 0, sizeof(*out));

   enum agx_isa_encoding target_encoding;
   if (!d12_agx3_fmin_compiler_target_encoding(chip, &target_encoding) ||
       encoding != target_encoding)
      return false;

   struct agx3_encoded_fmin_f32_reg_reg encoded;
   if (!agx3_pack_fmin_f32_reg_reg(
          encoding, dest, src0, src1, &encoded) ||
       encoded.length > sizeof(out->bytes))
      return false;

   memcpy(out->bytes, encoded.bytes, encoded.length);
   out->length = encoded.length;
   return true;
}

bool
d12_agx3_decode_fmin_f32_reg_reg(
   enum agx_chip chip, enum agx_isa_encoding encoding,
   const void *instruction, uint32_t instruction_size, uint16_t *dest,
   uint16_t *src0, uint16_t *src1)
{
   if (dest)
      *dest = 0;
   if (src0)
      *src0 = 0;
   if (src1)
      *src1 = 0;

   const uint32_t expected_length =
      encoding == AGX_ISA_ENCODING_AGX3B ? 6 :
      encoding == AGX_ISA_ENCODING_AGX3C ? 14 : 0;
   if (!instruction || !dest || !src0 || !src1 || expected_length == 0 ||
       instruction_size != expected_length)
      return false;

   enum agx_isa_encoding target_encoding;
   if (!d12_agx3_fmin_compiler_target_encoding(chip, &target_encoding) ||
       encoding != target_encoding)
      return false;

   uint16_t decoded_dest = 0, decoded_src0 = 0, decoded_src1 = 0;
   uint8_t decoded_length = 0;
   if (!agx3_unpack_fmin_f32_reg_reg(
          encoding, instruction, instruction_size, &decoded_dest,
          &decoded_src0, &decoded_src1, &decoded_length) ||
       decoded_length != expected_length)
      return false;

   *dest = decoded_dest;
   *src0 = decoded_src0;
   *src1 = decoded_src1;
   return true;
}

bool
d12_agx3_encode_fmax_f32_reg_reg(
   enum agx_chip chip, enum agx_isa_encoding encoding, uint16_t dest,
   uint16_t src0, uint16_t src1,
   struct d12_agx3_fmax_f32_reg_reg_encoding *out)
{
   if (!out)
      return false;

   memset(out, 0, sizeof(*out));

   enum agx_isa_encoding target_encoding;
   if (!d12_agx3_fmax_compiler_target_encoding(chip, &target_encoding) ||
       encoding != target_encoding)
      return false;

   struct agx3_encoded_fmax_f32_reg_reg encoded;
   if (!agx3_pack_fmax_f32_reg_reg(
          encoding, dest, src0, src1, &encoded) ||
       encoded.length > sizeof(out->bytes))
      return false;

   memcpy(out->bytes, encoded.bytes, encoded.length);
   out->length = encoded.length;
   return true;
}

bool
d12_agx3_decode_fmax_f32_reg_reg(
   enum agx_chip chip, enum agx_isa_encoding encoding,
   const void *instruction, uint32_t instruction_size, uint16_t *dest,
   uint16_t *src0, uint16_t *src1)
{
   if (dest)
      *dest = 0;
   if (src0)
      *src0 = 0;
   if (src1)
      *src1 = 0;

   const uint32_t expected_length =
      encoding == AGX_ISA_ENCODING_AGX3B ? 6 :
      encoding == AGX_ISA_ENCODING_AGX3C ? 14 : 0;
   if (!instruction || !dest || !src0 || !src1 || expected_length == 0 ||
       instruction_size != expected_length)
      return false;

   enum agx_isa_encoding target_encoding;
   if (!d12_agx3_fmax_compiler_target_encoding(chip, &target_encoding) ||
       encoding != target_encoding)
      return false;

   uint16_t decoded_dest = 0, decoded_src0 = 0, decoded_src1 = 0;
   uint8_t decoded_length = 0;
   if (!agx3_unpack_fmax_f32_reg_reg(
          encoding, instruction, instruction_size, &decoded_dest,
          &decoded_src0, &decoded_src1, &decoded_length) ||
       decoded_length != expected_length)
      return false;

   *dest = decoded_dest;
   *src0 = decoded_src0;
   *src1 = decoded_src1;
   return true;
}

bool
d12_agx3_encode_i32_to_f32_reg(
   enum agx_chip chip, enum agx_isa_encoding encoding, uint16_t dest,
   uint16_t src, struct d12_agx3_i32_to_f32_reg_encoding *out)
{
   if (!out)
      return false;

   memset(out, 0, sizeof(*out));

   enum agx_isa_encoding target_encoding;
   if (!d12_agx3_compiler_target_encoding(chip, &target_encoding) ||
       encoding != target_encoding)
      return false;

   struct agx3_encoded_i32_to_f32_reg encoded;
   if (!agx3_pack_i32_to_f32_reg(encoding, dest, src, &encoded) ||
       encoded.length > sizeof(out->bytes))
      return false;

   memcpy(out->bytes, encoded.bytes, encoded.length);
   out->length = encoded.length;
   return true;
}

bool
d12_agx3_decode_i32_to_f32_reg(
   enum agx_chip chip, enum agx_isa_encoding encoding,
   const void *instruction, uint32_t instruction_size, uint16_t *dest,
   uint16_t *src)
{
   if (dest)
      *dest = 0;
   if (src)
      *src = 0;

   const uint32_t expected_length =
      encoding == AGX_ISA_ENCODING_AGX3B ? 8 :
      encoding == AGX_ISA_ENCODING_AGX3C ? 10 : 0;
   if (!instruction || !dest || !src || expected_length == 0 ||
       instruction_size != expected_length)
      return false;

   enum agx_isa_encoding target_encoding;
   if (!d12_agx3_compiler_target_encoding(chip, &target_encoding) ||
       encoding != target_encoding)
      return false;

   uint16_t decoded_dest = 0, decoded_src = 0;
   uint8_t decoded_length = 0;
   if (!agx3_unpack_i32_to_f32_reg(
          encoding, instruction, instruction_size, &decoded_dest,
          &decoded_src, &decoded_length) ||
       decoded_length != expected_length)
      return false;

   *dest = decoded_dest;
   *src = decoded_src;
   return true;
}

bool
d12_agx3_encode_u32_to_f32_reg(
   enum agx_chip chip, enum agx_isa_encoding encoding, uint16_t dest,
   uint16_t src, struct d12_agx3_u32_to_f32_reg_encoding *out)
{
   if (!out)
      return false;

   memset(out, 0, sizeof(*out));

   enum agx_isa_encoding target_encoding;
   if (!d12_agx3_compiler_target_encoding(chip, &target_encoding) ||
       encoding != target_encoding)
      return false;

   struct agx3_encoded_u32_to_f32_reg encoded;
   if (!agx3_pack_u32_to_f32_reg(encoding, dest, src, &encoded) ||
       encoded.length > sizeof(out->bytes))
      return false;

   memcpy(out->bytes, encoded.bytes, encoded.length);
   out->length = encoded.length;
   return true;
}

bool
d12_agx3_decode_u32_to_f32_reg(
   enum agx_chip chip, enum agx_isa_encoding encoding,
   const void *instruction, uint32_t instruction_size, uint16_t *dest,
   uint16_t *src)
{
   if (dest)
      *dest = 0;
   if (src)
      *src = 0;

   const uint32_t expected_length =
      encoding == AGX_ISA_ENCODING_AGX3B ? 8 :
      encoding == AGX_ISA_ENCODING_AGX3C ? 10 : 0;
   if (!instruction || !dest || !src || expected_length == 0 ||
       instruction_size != expected_length)
      return false;

   enum agx_isa_encoding target_encoding;
   if (!d12_agx3_compiler_target_encoding(chip, &target_encoding) ||
       encoding != target_encoding)
      return false;

   uint16_t decoded_dest = 0, decoded_src = 0;
   uint8_t decoded_length = 0;
   if (!agx3_unpack_u32_to_f32_reg(
          encoding, instruction, instruction_size, &decoded_dest,
          &decoded_src, &decoded_length) ||
       decoded_length != expected_length)
      return false;

   *dest = decoded_dest;
   *src = decoded_src;
   return true;
}

bool
d12_agx3_encode_f32_rint_reg(
   enum agx_chip chip, enum agx_isa_encoding encoding, uint16_t dest,
   uint16_t src, struct d12_agx3_f32_rint_reg_encoding *out)
{
   if (!out)
      return false;

   memset(out, 0, sizeof(*out));

   enum agx_isa_encoding target_encoding;
   if (!d12_agx3_compiler_target_encoding(chip, &target_encoding) ||
       encoding != target_encoding)
      return false;

   struct agx3_encoded_f32_rint_reg encoded;
   if (!agx3_pack_f32_rint_reg(encoding, dest, src, &encoded) ||
       encoded.length != sizeof(out->bytes))
      return false;

   memcpy(out->bytes, encoded.bytes, encoded.length);
   out->length = encoded.length;
   return true;
}

bool
d12_agx3_decode_f32_rint_reg(
   enum agx_chip chip, enum agx_isa_encoding encoding,
   const void *instruction, uint32_t instruction_size, uint16_t *dest,
   uint16_t *src)
{
   if (dest)
      *dest = 0;
   if (src)
      *src = 0;

   if (!instruction || !dest || !src ||
       instruction_size != D12_AGX3_F32_RINT_REG_LENGTH)
      return false;

   enum agx_isa_encoding target_encoding;
   if (!d12_agx3_compiler_target_encoding(chip, &target_encoding) ||
       encoding != target_encoding)
      return false;

   uint16_t decoded_dest = 0, decoded_src = 0;
   uint8_t decoded_length = 0;
   if (!agx3_unpack_f32_rint_reg(
          encoding, instruction, instruction_size, &decoded_dest,
          &decoded_src, &decoded_length) ||
       decoded_length != D12_AGX3_F32_RINT_REG_LENGTH)
      return false;

   *dest = decoded_dest;
   *src = decoded_src;
   return true;
}

bool
d12_agx3_encode_f32_rcp_reg(
   enum agx_chip chip, enum agx_isa_encoding encoding, uint16_t dest,
   uint16_t src, struct d12_agx3_f32_rcp_reg_encoding *out)
{
   if (!out)
      return false;

   memset(out, 0, sizeof(*out));

   enum agx_isa_encoding target_encoding;
   if (!d12_agx3_compiler_target_encoding(chip, &target_encoding) ||
       encoding != target_encoding)
      return false;

   struct agx3_encoded_f32_rcp_reg encoded;
   if (!agx3_pack_f32_rcp_reg(encoding, dest, src, &encoded) ||
       encoded.length != sizeof(out->bytes))
      return false;

   memcpy(out->bytes, encoded.bytes, encoded.length);
   out->length = encoded.length;
   return true;
}

bool
d12_agx3_decode_f32_rcp_reg(
   enum agx_chip chip, enum agx_isa_encoding encoding,
   const void *instruction, uint32_t instruction_size, uint16_t *dest,
   uint16_t *src)
{
   if (dest)
      *dest = 0;
   if (src)
      *src = 0;

   if (!instruction || !dest || !src ||
       instruction_size != D12_AGX3_F32_RCP_REG_LENGTH)
      return false;

   enum agx_isa_encoding target_encoding;
   if (!d12_agx3_compiler_target_encoding(chip, &target_encoding) ||
       encoding != target_encoding)
      return false;

   uint16_t decoded_dest = 0, decoded_src = 0;
   uint8_t decoded_length = 0;
   if (!agx3_unpack_f32_rcp_reg(
          encoding, instruction, instruction_size, &decoded_dest,
          &decoded_src, &decoded_length) ||
       decoded_length != D12_AGX3_F32_RCP_REG_LENGTH)
      return false;

   *dest = decoded_dest;
   *src = decoded_src;
   return true;
}

bool
d12_agx3_encode_f32_rsqrt_reg(
   enum agx_chip chip, enum agx_isa_encoding encoding, uint16_t dest,
   uint16_t src, struct d12_agx3_f32_rsqrt_reg_encoding *out)
{
   if (!out)
      return false;

   memset(out, 0, sizeof(*out));

   enum agx_isa_encoding target_encoding;
   if (!d12_agx3_compiler_target_encoding(chip, &target_encoding) ||
       encoding != target_encoding)
      return false;

   struct agx3_encoded_f32_rsqrt_reg encoded;
   if (!agx3_pack_f32_rsqrt_reg(encoding, dest, src, &encoded) ||
       encoded.length != sizeof(out->bytes))
      return false;

   memcpy(out->bytes, encoded.bytes, encoded.length);
   out->length = encoded.length;
   return true;
}

bool
d12_agx3_decode_f32_rsqrt_reg(
   enum agx_chip chip, enum agx_isa_encoding encoding,
   const void *instruction, uint32_t instruction_size, uint16_t *dest,
   uint16_t *src)
{
   if (dest)
      *dest = 0;
   if (src)
      *src = 0;

   if (!instruction || !dest || !src ||
       instruction_size != D12_AGX3_F32_RSQRT_REG_LENGTH)
      return false;

   enum agx_isa_encoding target_encoding;
   if (!d12_agx3_compiler_target_encoding(chip, &target_encoding) ||
       encoding != target_encoding)
      return false;

   uint16_t decoded_dest = 0, decoded_src = 0;
   uint8_t decoded_length = 0;
   if (!agx3_unpack_f32_rsqrt_reg(
          encoding, instruction, instruction_size, &decoded_dest,
          &decoded_src, &decoded_length) ||
       decoded_length != D12_AGX3_F32_RSQRT_REG_LENGTH)
      return false;

   *dest = decoded_dest;
   *src = decoded_src;
   return true;
}

static bool
d12_agx3_i32_compare_predicate_to_shared(
   enum d12_agx3_i32_compare_predicate predicate,
   enum agx3_i32_compare_predicate *shared)
{
   switch (predicate) {
   case D12_AGX3_I32_COMPARE_EQ:
      *shared = AGX3_I32_COMPARE_EQ;
      return true;
   case D12_AGX3_I32_COMPARE_SLT:
      *shared = AGX3_I32_COMPARE_SLT;
      return true;
   default:
      return false;
   }
}

bool
d12_agx3_encode_i32_compare_select_imm_5_7_reg_reg(
   enum agx_chip chip, enum agx_isa_encoding encoding,
   enum d12_agx3_i32_compare_predicate predicate, uint16_t dest,
   uint16_t src0, uint16_t src1,
   struct d12_agx3_i32_compare_select_imm_5_7_reg_reg_encoding *out)
{
   if (!out)
      return false;

   memset(out, 0, sizeof(*out));

   enum agx_isa_encoding target_encoding;
   enum agx3_i32_compare_predicate shared_predicate;
   if (!d12_agx3_compiler_target_encoding(chip, &target_encoding) ||
       encoding != target_encoding ||
       !d12_agx3_i32_compare_predicate_to_shared(predicate,
                                                  &shared_predicate))
      return false;

   struct agx3_encoded_i32_compare_select_imm_5_7_reg_reg encoded;
   if (!agx3_pack_i32_compare_select_imm_5_7_reg_reg(
          encoding, shared_predicate, dest, src0, src1, &encoded) ||
       encoded.length > sizeof(out->bytes))
      return false;

   memcpy(out->bytes, encoded.bytes, encoded.length);
   out->length = encoded.length;
   return true;
}

bool
d12_agx3_decode_i32_compare_select_imm_5_7_reg_reg(
   enum agx_chip chip, enum agx_isa_encoding encoding,
   const void *instruction, uint32_t instruction_size,
   enum d12_agx3_i32_compare_predicate *predicate, uint16_t *dest,
   uint16_t *src0, uint16_t *src1)
{
   if (predicate)
      *predicate = D12_AGX3_I32_COMPARE_EQ;
   if (dest)
      *dest = 0;
   if (src0)
      *src0 = 0;
   if (src1)
      *src1 = 0;

   const uint32_t expected_length =
      encoding == AGX_ISA_ENCODING_AGX3B ? 10 :
      encoding == AGX_ISA_ENCODING_AGX3C ? 14 : 0;
   if (!instruction || !predicate || !dest || !src0 || !src1 ||
       expected_length == 0 || instruction_size != expected_length)
      return false;

   enum agx_isa_encoding target_encoding;
   if (!d12_agx3_compiler_target_encoding(chip, &target_encoding) ||
       encoding != target_encoding)
      return false;

   enum agx3_i32_compare_predicate shared_predicate;
   uint16_t decoded_dest = 0, decoded_src0 = 0, decoded_src1 = 0;
   uint8_t decoded_length = 0;
   if (!agx3_unpack_i32_compare_select_imm_5_7_reg_reg(
          encoding, instruction, instruction_size, &shared_predicate,
          &decoded_dest, &decoded_src0, &decoded_src1, &decoded_length) ||
       decoded_length != expected_length)
      return false;

   switch (shared_predicate) {
   case AGX3_I32_COMPARE_EQ:
      *predicate = D12_AGX3_I32_COMPARE_EQ;
      break;
   case AGX3_I32_COMPARE_SLT:
      *predicate = D12_AGX3_I32_COMPARE_SLT;
      break;
   default:
      return false;
   }

   *dest = decoded_dest;
   *src0 = decoded_src0;
   *src1 = decoded_src1;
   return true;
}

static bool
d12_agx3_f32_to_int_rtz_mode_to_shared(
   enum d12_agx3_f32_to_int_rtz_mode mode,
   enum agx3_f32_to_int_rtz_mode *shared)
{
   switch (mode) {
   case D12_AGX3_F32_TO_I32_RTZ:
      *shared = AGX3_F32_TO_I32_RTZ;
      return true;
   case D12_AGX3_F32_TO_U32_RTZ:
      *shared = AGX3_F32_TO_U32_RTZ;
      return true;
   default:
      return false;
   }
}

bool
d12_agx3_encode_f32_to_int_rtz_reg(
   enum agx_chip chip, enum agx_isa_encoding encoding,
   enum d12_agx3_f32_to_int_rtz_mode mode, uint16_t dest, uint16_t src,
   struct d12_agx3_f32_to_int_rtz_reg_encoding *out)
{
   if (!out)
      return false;

   memset(out, 0, sizeof(*out));

   enum agx_isa_encoding target_encoding;
   enum agx3_f32_to_int_rtz_mode shared_mode;
   if (!d12_agx3_compiler_target_encoding(chip, &target_encoding) ||
       encoding != target_encoding ||
       !d12_agx3_f32_to_int_rtz_mode_to_shared(mode, &shared_mode))
      return false;

   struct agx3_encoded_f32_to_int_rtz_reg encoded;
   if (!agx3_pack_f32_to_int_rtz_reg(
          encoding, shared_mode, dest, src, &encoded) ||
       encoded.length > sizeof(out->bytes))
      return false;

   memcpy(out->bytes, encoded.bytes, encoded.length);
   out->length = encoded.length;
   return true;
}

bool
d12_agx3_decode_f32_to_int_rtz_reg(
   enum agx_chip chip, enum agx_isa_encoding encoding,
   const void *instruction, uint32_t instruction_size,
   enum d12_agx3_f32_to_int_rtz_mode *mode, uint16_t *dest, uint16_t *src)
{
   if (mode)
      *mode = D12_AGX3_F32_TO_I32_RTZ;
   if (dest)
      *dest = 0;
   if (src)
      *src = 0;

   if (!instruction || !mode || !dest || !src ||
       instruction_size != D12_AGX3_F32_TO_INT_RTZ_REG_LENGTH)
      return false;

   enum agx_isa_encoding target_encoding;
   if (!d12_agx3_compiler_target_encoding(chip, &target_encoding) ||
       encoding != target_encoding)
      return false;

   enum agx3_f32_to_int_rtz_mode shared_mode;
   uint16_t decoded_dest = 0, decoded_src = 0;
   uint8_t decoded_length = 0;
   if (!agx3_unpack_f32_to_int_rtz_reg(
          encoding, instruction, instruction_size, &shared_mode,
          &decoded_dest, &decoded_src, &decoded_length) ||
       decoded_length != D12_AGX3_F32_TO_INT_RTZ_REG_LENGTH)
      return false;

   switch (shared_mode) {
   case AGX3_F32_TO_I32_RTZ:
      *mode = D12_AGX3_F32_TO_I32_RTZ;
      break;
   case AGX3_F32_TO_U32_RTZ:
      *mode = D12_AGX3_F32_TO_U32_RTZ;
      break;
   default:
      return false;
   }

   *dest = decoded_dest;
   *src = decoded_src;
   return true;
}

bool
d12_agx3_encode_yield_i16_reg(
   enum agx_chip chip, uint16_t src,
   struct d12_agx3_yield_i16_reg_encoding *out)
{
   if (!out)
      return false;

   memset(out, 0, sizeof(*out));

   enum agx_isa_encoding encoding;
   if (!d12_agx3_compiler_target_encoding(chip, &encoding))
      return false;

   struct agx3_encoded_yield_i16_reg encoded;
   if (!agx3_pack_yield_i16_reg(encoding, src, &encoded) ||
       encoded.length != sizeof(out->bytes))
      return false;

   memcpy(out->bytes, encoded.bytes, encoded.length);
   out->length = encoded.length;
   return true;
}

bool
d12_agx3_decode_yield_i16_reg(enum agx_chip chip, const void *instruction,
                              uint32_t instruction_size, uint16_t *src)
{
   if (src)
      *src = 0;
   if (!instruction || !src ||
       instruction_size != D12_AGX3_YIELD_I16_REG_LENGTH)
      return false;

   enum agx_isa_encoding encoding;
   if (!d12_agx3_compiler_target_encoding(chip, &encoding))
      return false;

   uint16_t decoded_src = 0;
   uint8_t decoded_length = 0;
   struct agx3_encoded_yield_i16_reg canonical;
   if (!agx3_unpack_yield_i16_reg(encoding, instruction, instruction_size,
                                  &decoded_src, &decoded_length) ||
       decoded_length != D12_AGX3_YIELD_I16_REG_LENGTH ||
       !agx3_pack_yield_i16_reg(encoding, decoded_src, &canonical) ||
       canonical.length != decoded_length ||
       memcmp(instruction, canonical.bytes, decoded_length) != 0)
      return false;

   *src = decoded_src;
   return true;
}

bool
d12_agx3_encode_wave_read_lane_at_f32_imm(
   enum agx_chip chip, uint16_t dest, uint16_t src, uint8_t lane,
   struct d12_agx3_wave_read_lane_at_f32_imm_encoding *out)
{
   if (!out)
      return false;
   memset(out, 0, sizeof(*out));

   enum agx_isa_encoding encoding;
   struct agx3_encoded_simd_broadcast_f32_imm encoded;
   if (!d12_agx3_subgroup_target_encoding(chip, &encoding) ||
       !agx3_pack_simd_broadcast_f32_imm(encoding, dest, src, lane,
                                         &encoded) ||
       encoded.length != sizeof(out->bytes))
      return false;

   memcpy(out->bytes, encoded.bytes, encoded.length);
   out->length = encoded.length;
   return true;
}

bool
d12_agx3_decode_wave_read_lane_at_f32_imm(
   enum agx_chip chip, const void *instruction, uint32_t instruction_size,
   uint16_t *dest, uint16_t *src, uint8_t *lane)
{
   if (dest)
      *dest = 0;
   if (src)
      *src = 0;
   if (lane)
      *lane = 0;
   if (!instruction || !dest || !src || !lane ||
       instruction_size != D12_AGX3_WAVE_READ_LANE_AT_F32_IMM_LENGTH)
      return false;

   enum agx_isa_encoding encoding;
   uint16_t decoded_dest = 0, decoded_src = 0;
   uint8_t decoded_lane = 0, decoded_length = 0;
   if (!d12_agx3_subgroup_target_encoding(chip, &encoding) ||
       !agx3_unpack_simd_broadcast_f32_imm(
          encoding, instruction, instruction_size, &decoded_dest,
          &decoded_src, &decoded_lane, &decoded_length) ||
       decoded_length != instruction_size)
      return false;

   *dest = decoded_dest;
   *src = decoded_src;
   *lane = decoded_lane;
   return true;
}

static bool
d12_agx3_store_subgroup(const uint8_t *bytes, uint8_t length,
                        struct d12_agx3_subgroup_encoding *out)
{
   if (length == 0 || length > sizeof(out->bytes))
      return false;

   memcpy(out->bytes, bytes, length);
   out->length = length;
   return true;
}

bool
d12_agx3_encode_subgroup_shuffle(
   enum agx_chip chip, const struct d12_agx3_subgroup_shuffle_desc *desc,
   struct d12_agx3_subgroup_encoding *out)
{
   if (!out)
      return false;
   memset(out, 0, sizeof(*out));
   if (!desc)
      return false;

   enum agx_isa_encoding encoding;
   bool quad;
   enum agx3_subgroup_shuffle_op op;
   enum agx3_subgroup_lane_kind lane_kind;
   if (!d12_agx3_subgroup_target_encoding(chip, &encoding) ||
       !d12_agx3_map_subgroup_scope(desc->scope, &quad) ||
       !d12_agx3_map_subgroup_shuffle(desc->op, &op) ||
       !d12_agx3_map_subgroup_lane(desc->lane_source, &lane_kind))
      return false;

   const struct agx3_subgroup_shuffle_desc shared = {
      .quad = quad,
      .op = op,
      .lane_kind = lane_kind,
      .dest = desc->dest,
      .src = desc->src,
      .lane = desc->lane,
   };
   uint8_t bytes[AGX3_SUBGROUP_MAX_LENGTH], length = 0;
   return agx3_subgroup_encode_shuffle(encoding, &shared, bytes, &length) &&
          d12_agx3_store_subgroup(bytes, length, out);
}

bool
d12_agx3_encode_zero_compare_ballot(
   enum agx_chip chip, const struct d12_agx3_zero_ballot_desc *desc,
   struct d12_agx3_subgroup_encoding *out)
{
   if (!out)
      return false;
   memset(out, 0, sizeof(*out));
   if (!desc)
      return false;

   enum agx_isa_encoding encoding;
   bool quad, invert = false;
   enum agx3_subgroup_ballot_type type;
   uint8_t condition;
   if (!d12_agx3_subgroup_target_encoding(chip, &encoding) ||
       !d12_agx3_map_subgroup_scope(desc->scope, &quad))
      return false;

   switch (desc->type) {
   case D12_AGX3_ZERO_BALLOT_I32:
      type = AGX3_SUBGROUP_BALLOT_INTEGER;
      switch (desc->predicate) {
      case D12_AGX3_ZERO_BALLOT_EQUAL:
         condition = 4;
         break;
      case D12_AGX3_ZERO_BALLOT_NOT_EQUAL:
         condition = 4;
         invert = true;
         break;
      case D12_AGX3_ZERO_BALLOT_LESS:
         condition = 5;
         break;
      case D12_AGX3_ZERO_BALLOT_GREATER_EQUAL:
         condition = 5;
         invert = true;
         break;
      case D12_AGX3_ZERO_BALLOT_GREATER:
         condition = 6;
         break;
      case D12_AGX3_ZERO_BALLOT_LESS_EQUAL:
         condition = 6;
         invert = true;
         break;
      default:
         return false;
      }
      break;

   case D12_AGX3_ZERO_BALLOT_F32:
      type = AGX3_SUBGROUP_BALLOT_FLOAT;
      switch (desc->predicate) {
      case D12_AGX3_ZERO_BALLOT_EQUAL:
         condition = 0;
         break;
      case D12_AGX3_ZERO_BALLOT_NOT_EQUAL:
         condition = 0;
         invert = true;
         break;
      case D12_AGX3_ZERO_BALLOT_LESS:
         condition = 1;
         break;
      case D12_AGX3_ZERO_BALLOT_GREATER_EQUAL:
         condition = 5;
         break;
      case D12_AGX3_ZERO_BALLOT_GREATER:
         condition = 2;
         break;
      case D12_AGX3_ZERO_BALLOT_LESS_EQUAL:
         condition = 6;
         break;
      default:
         return false;
      }
      break;

   default:
      return false;
   }

   const struct agx3_subgroup_ballot_desc shared = {
      .quad = quad,
      .type = type,
      .dest = desc->dest,
      .src = desc->src,
      .condition = condition,
      .invert = invert,
   };
   uint8_t bytes[AGX3_SUBGROUP_MAX_LENGTH], length = 0;
   return agx3_subgroup_encode_ballot(encoding, &shared, bytes, &length) &&
          d12_agx3_store_subgroup(bytes, length, out);
}

bool
d12_agx3_encode_subgroup_reduce(
   enum agx_chip chip, const struct d12_agx3_subgroup_reduce_desc *desc,
   struct d12_agx3_subgroup_encoding *out)
{
   if (!out)
      return false;
   memset(out, 0, sizeof(*out));
   if (!desc)
      return false;

   enum agx_isa_encoding encoding;
   bool quad;
   enum agx3_subgroup_reduce_op op;
   if (!d12_agx3_subgroup_target_encoding(chip, &encoding) ||
       !d12_agx3_map_subgroup_scope(desc->scope, &quad) ||
       !d12_agx3_map_subgroup_reduce(desc->op, &op))
      return false;

   const struct agx3_subgroup_reduce_desc shared = {
      .quad = quad,
      .prefix = desc->prefix,
      .op = op,
      .dest = desc->dest,
      .src = desc->src,
   };
   uint8_t bytes[AGX3_SUBGROUP_MAX_LENGTH], length = 0;
   return agx3_subgroup_encode_reduce(encoding, &shared, bytes, &length) &&
          d12_agx3_store_subgroup(bytes, length, out);
}

bool
d12_agx3_encode_end_with_wait_fence_mask(
   enum agx_chip chip, uint8_t wait_fence_mask,
   struct d12_agx3_end_with_wait_fence_mask_encoding *out)
{
   if (!out)
      return false;

   memset(out, 0, sizeof(*out));

   enum agx_isa_encoding encoding;
   if (!d12_agx3_compiler_target_encoding(chip, &encoding))
      return false;

   struct agx3_encoded_end encoded;
   if (!agx3_pack_end_with_wait_fence_mask(encoding, wait_fence_mask,
                                           &encoded) ||
       encoded.length != sizeof(out->bytes))
      return false;

   memcpy(out->bytes, encoded.bytes, encoded.length);
   out->length = encoded.length;
   return true;
}

bool
d12_agx3_decode_end_with_wait_fence_mask(enum agx_chip chip,
                                         const void *instruction,
                                         uint32_t instruction_size,
                                         uint8_t *wait_fence_mask)
{
   if (wait_fence_mask)
      *wait_fence_mask = 0;
   if (!instruction || !wait_fence_mask ||
       instruction_size != D12_AGX3_END_WITH_WAIT_FENCE_MASK_LENGTH)
      return false;

   enum agx_isa_encoding encoding;
   if (!d12_agx3_compiler_target_encoding(chip, &encoding))
      return false;

   uint8_t decoded_mask = 0;
   uint8_t decoded_length = 0;
   struct agx3_encoded_end canonical;
   if (!agx3_unpack_end_with_wait_fence_mask(
          encoding, instruction, instruction_size, &decoded_mask,
          &decoded_length) ||
       decoded_length != D12_AGX3_END_WITH_WAIT_FENCE_MASK_LENGTH ||
       !agx3_pack_end_with_wait_fence_mask(encoding, decoded_mask,
                                           &canonical) ||
       canonical.length != decoded_length ||
       memcmp(instruction, canonical.bytes, decoded_length) != 0)
      return false;

   *wait_fence_mask = decoded_mask;
   return true;
}

bool
d12_agx3_encode_unary_alu(
   enum agx_isa_encoding encoding, enum d12_agx3_unary_alu_op op,
   uint16_t dest, uint16_t src, struct d12_agx3_unary_alu_encoding *out)
{
   if (!out)
      return false;

   memset(out, 0, sizeof(*out));

   switch (op) {
   case D12_AGX3_UNARY_ALU_INOT_U32: {
      struct agx3_encoded_bit_not_u32_reg encoded;
      if (!agx3_pack_bit_not_u32_reg(encoding, dest, src, &encoded))
         return false;

      memcpy(out->bytes, encoded.bytes, encoded.length);
      out->length = encoded.length;
      return true;
   }

   default:
      return false;
   }
}
