/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/*
 * G18G preserves fp32 denormals in hardware. Flush floating-point results
 * in software when the shader requests DenormFlushToZero(32). Preserve loads
 * bit for bit, including the residency component of sparse texture results.
 * Negate and abs still need a flush: they can consume an unflushed value.
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

static bool
is_texel_fetch(nir_texop op)
{
   switch (op) {
   case nir_texop_txf:
   case nir_texop_txf_ms:
   case nir_texop_txf_ms_fb:
   case nir_texop_fragment_fetch_amd:
   case nir_texop_fragment_mask_fetch_amd:
      return true;
   default:
      return false;
   }
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
agx_lower_g18_ftz_instr(nir_builder *b, nir_instr *instr, UNUSED void *data)
{
   if (instr->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      const nir_op_info *info = &nir_op_infos[alu->op];

      /* Exact conversions and sign moves cannot produce a denorm. Widening
       * and quantizing to 16-bit precision are 16-bit paths: an fp16 value
       * can never widen to an fp32 denormal, so these are never flushed. */
      if (alu->op == nir_op_fsign || alu->op == nir_op_f2f32 ||
          alu->op == nir_op_f2fmp ||
          alu->op == nir_op_unpack_half_2x16 ||
          alu->op == nir_op_fquantize2f16)
         return false;
      if (!fp32_def(info->output_type, alu->def.bit_size))
         return false;
      flush_def(b, &alu->instr, &alu->def);
      return true;
   }

   if (instr->type == nir_instr_type_tex) {
      nir_tex_instr *tex = nir_instr_as_tex(instr);
      if (tex->is_sparse || is_texel_fetch(tex->op))
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
