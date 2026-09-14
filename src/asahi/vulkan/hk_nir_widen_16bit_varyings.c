/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_nir_widen_16bit_varyings.h"

#include "nir.h"
#include "nir_builder.h"

/* storageInputOutput16 varyings travel in 32-bit UVS words. Widen them while
 * their SPIR-V types are still known: nir_opt_varyings retypes every varying
 * it relocates as float, after which a 16-bit integer would be widened with
 * f2f32 and narrowed with f2f16. G17P's float conversions canonicalize NaNs,
 * so an int16 whose bits spell one arrived changed.
 */
static bool
widen_16bit_varying(nir_builder *b, nir_intrinsic_instr *intr,
                    UNUSED void *data)
{
   const mesa_shader_stage stage = b->shader->info.stage;

   switch (intr->intrinsic) {
   case nir_intrinsic_store_output:
   case nir_intrinsic_store_per_vertex_output: {
      if (stage == MESA_SHADER_FRAGMENT || intr->src[0].ssa->bit_size != 16)
         return false;

      const nir_alu_type type =
         nir_alu_type_get_base_type(nir_intrinsic_src_type(intr));

      b->cursor = nir_before_instr(&intr->instr);
      nir_src_rewrite(&intr->src[0],
                      nir_convert_to_bit_size(b, intr->src[0].ssa, type, 32));
      nir_intrinsic_set_src_type(intr, type | 32);
      return true;
   }

   case nir_intrinsic_load_input:
   case nir_intrinsic_load_per_vertex_input:
   case nir_intrinsic_load_interpolated_input: {
      if (stage == MESA_SHADER_VERTEX || intr->def.bit_size != 16)
         return false;

      const nir_alu_type type =
         nir_alu_type_get_base_type(nir_intrinsic_dest_type(intr));

      intr->def.bit_size = 32;
      nir_intrinsic_set_dest_type(intr, type | 32);

      b->cursor = nir_after_instr(&intr->instr);
      nir_def *narrow = nir_convert_to_bit_size(b, &intr->def, type, 16);
      nir_def_rewrite_uses_after(&intr->def, narrow);
      return true;
   }

   default:
      return false;
   }
}

bool
hk_nir_widen_16bit_varyings(nir_shader *nir)
{
   return nir_shader_intrinsics_pass(nir, widen_16bit_varying,
                                     nir_metadata_control_flow, NULL);
}
