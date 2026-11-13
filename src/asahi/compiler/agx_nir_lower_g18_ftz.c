/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/*
 * G18G fp32 arithmetic preserves denorms and the AGX3 encoding has no
 * flush-to-zero mode, so a shader that declares DenormFlushToZero(32) gets
 * the flush in software: every fp32 value produced by an ALU instruction,
 * by a load that carries a 32-bit float dest type, or by a texture is
 * flushed at its definition. The flush is value-exact and passes NaN, the
 * infinities and both zeros through.
 *
 * Not covered: load_const, loads that carry no dest type (UBO, SSBO,
 * shared and similar), and sparse texture and image loads, whose extra
 * residency component must stay a bit pattern. A denormal from those
 * sources is not flushed at its definition: it reaches a store or a
 * comparison unchanged, and an ALU instruction that consumes it sees the
 * denormal as an operand.
 */

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "agx_nir.h"

static bool
fp32_def(nir_alu_type type, unsigned bit_size)
{
   return bit_size == 32 &&
          nir_alu_type_get_base_type(type) == nir_type_float;
}

static void
flush_def(nir_builder *b, nir_instr *instr, nir_def *x)
{
   b->cursor = nir_after_instr(instr);
   b->fp_math_ctrl = nir_fp_no_fast_math;

   /* |x| < FLT_MIN and x != 0 selects zero with the sign of x: a NaN fails
    * the magnitude test and both zeros fail the nonzero test, so they pass
    * through unchanged. */
   nir_def *cond = nir_iand(b,
      nir_flt(b, nir_fabs(b, x), nir_imm_float(b, 0x1p-126f)),
      nir_fneu(b, x, nir_imm_float(b, 0.0)));
   nir_def *zero = nir_bcsel(b, nir_flt(b, x, nir_imm_float(b, 0.0)),
                             nir_imm_float(b, -0.0), nir_imm_float(b, 0.0));
   nir_def *flushed = nir_bcsel(b, cond, zero, x);
   nir_def_rewrite_uses_after(x, flushed);
}

static bool
agx_lower_g18_ftz_instr(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      const nir_op_info *info = &nir_op_infos[alu->op];

      /* Exact conversions and sign moves cannot produce a denorm. Negates
       * and absolutes of flushed values cannot either, but of constants
       * they can, so only constants force a flush. Widening and
       * quantizing to 16-bit precision are 16-bit paths: an fp16 value can
       * never widen to an fp32 denormal, so these are never flushed. */
      if (alu->op == nir_op_fsign || alu->op == nir_op_f2f32 ||
          alu->op == nir_op_f2fmp ||
          alu->op == nir_op_unpack_half_2x16 ||
          alu->op == nir_op_fquantize2f16)
         return false;
      if (!fp32_def(info->output_type, alu->def.bit_size))
         return false;
      bool passthrough = alu->op == nir_op_fabs || alu->op == nir_op_fneg;
      if (passthrough && info->num_inputs > 0 &&
          !nir_src_is_const(alu->src[0].src))
         return false;
      flush_def(b, &alu->instr, &alu->def);
      return true;
   }

   if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      /* A sparse load returns its residency code in an extra component of
       * the same def, and the code 1 is a denormal when read as a float. */
      if (intr->intrinsic == nir_intrinsic_image_sparse_load ||
          intr->intrinsic == nir_intrinsic_image_deref_sparse_load ||
          intr->intrinsic == nir_intrinsic_bindless_image_sparse_load)
         return false;

      /* A declared 16-bit type widened into a 32-bit def is a 16-bit path. */
      if (nir_intrinsic_has_dest_type(intr) &&
          nir_alu_type_get_type_size(nir_intrinsic_dest_type(intr)) ==
             intr->def.bit_size &&
          fp32_def(nir_intrinsic_dest_type(intr), intr->def.bit_size)) {
         flush_def(b, &intr->instr, &intr->def);
         return true;
      }
      return false;
   }

   if (instr->type == nir_instr_type_tex) {
      nir_tex_instr *tex = nir_instr_as_tex(instr);
      if (tex->is_sparse)
         return false;
      if (nir_alu_type_get_type_size(tex->dest_type) == tex->def.bit_size &&
          fp32_def(tex->dest_type, tex->def.bit_size)) {
         flush_def(b, &tex->instr, &tex->def);
         return true;
      }
      return false;
   }

   return false;
}

bool
agx_nir_lower_g18_ftz(nir_shader *nir)
{
   return nir_shader_instructions_pass(
      nir, agx_lower_g18_ftz_instr,
      nir_metadata_block_index | nir_metadata_dominance, NULL);
}
