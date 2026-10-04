/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include "agx_abi.h"
#include "agx_compile.h"
#include "agx_tilebuffer.h"
#include "nir_builder.h"

static bool
test(unsigned samples, bool conditional)
{
   static const nir_shader_compiler_options options = {0};
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &options, "conditional sample exports");
   nir_def *condition = nir_load_front_face(&b, 1);
   if (conditional)
      nir_push_if(&b, condition);
   if (conditional)
      nir_export_agx(&b, nir_imm_int(&b, 11), .base = AGX_ABI_FOUT_Z);
   nir_export_agx(&b, nir_imm_int(&b, 22), .base = AGX_ABI_FOUT_Z);
   if (conditional) {
      nir_push_else(&b, NULL);
      nir_export_agx(&b, nir_imm_int(&b, 33), .base = AGX_ABI_FOUT_Z);
      nir_pop_if(&b, NULL);
   }
   nir_export_agx(&b, nir_imm_vec4(&b, 1, 0, 0, 1),
                  .base = AGX_ABI_FOUT_COLOUR(0));
   agx_nir_unroll_g17p_samples(b.shader, samples, 1);
   nir_opt_copy_prop(b.shader);
   nir_opt_dce(b.shader);
   nir_function_impl *impl = nir_shader_get_entrypoint(b.shader);
   nir_metadata_require(impl, nir_metadata_control_flow);
   unsigned invalid = 0, depth = 0, good_phi = 0;
   nir_foreach_block(block, impl)
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
         if (in->intrinsic != nir_intrinsic_export_agx)
            continue;
         nir_def *v = in->src[0].ssa;
         if (!nir_block_dominates(nir_def_block(v), block))
            invalid++;
         unsigned base = nir_intrinsic_base(in);
         bool is_depth = base == AGX_ABI_FOUT_Z;
         for (unsigned i = 1; i < samples; i++)
            is_depth |= base == agx_g17p_sample_zs_base(samples, 1, i);
         if (!is_depth)
            continue;
         depth++;
         if (nir_def_instr(v)->type == nir_instr_type_phi) {
            nir_phi_instr *phi = nir_instr_as_phi(nir_def_instr(v));
            unsigned n = 0, mask = 0;
            nir_foreach_phi_src(src, phi) {
               n++;
               if (nir_src_is_const(src->src)) {
                  unsigned c = nir_src_as_uint(src->src);
                  mask |= (c == 22) ? 1 : ((c == 33) ? 2 : 4);
               }
            }
            good_phi += (n == 2 && mask == 3);
         }
      }
   printf("samples=%u conditional=%d invalid=%u depth=%u expected_phi=%u\n",
          samples, conditional, invalid, depth, good_phi);
   bool ok =
      invalid == 0 && depth == samples && (!conditional || good_phi == samples);
   if (ok)
      nir_validate_shader(b.shader, "conditional export unroll");
   ralloc_free(b.shader);
   return ok;
}
int
main(void)
{
   glsl_type_singleton_init_or_ref();
   bool ok = true;
   ok &= test(2, true);
   ok &= test(4, true);
   ok &= test(2, false);
   ok &= test(4, false);
   glsl_type_singleton_decref();
   return !ok;
}
