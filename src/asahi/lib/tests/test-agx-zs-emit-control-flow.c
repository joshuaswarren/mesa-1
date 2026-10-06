/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include "agx_compile.h"
#include "nir_builder.h"

/* A body cloned into both arms of an if/else (Honeykrisp's ordinary-descriptor
 * specialization) stores FragDepth once per arm. The zs lowering must leave a
 * single store_zs_agx in the last block, fed by a phi of both values.
 */
static bool
test(bool stencil_at_end)
{
   static const nir_shader_compiler_options options = {0};
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &options, "zs emit in control flow");
   b.shader->info.outputs_written |= BITFIELD64_BIT(FRAG_RESULT_DEPTH);

   nir_push_if(&b, nir_load_front_face(&b, 1));
   nir_store_output(&b, nir_imm_float(&b, 0.25), nir_imm_int(&b, 0),
                    .io_semantics.location = FRAG_RESULT_DEPTH,
                    .src_type = nir_type_float32);
   nir_push_else(&b, NULL);
   nir_store_output(&b, nir_imm_float(&b, 0.75), nir_imm_int(&b, 0),
                    .io_semantics.location = FRAG_RESULT_DEPTH,
                    .src_type = nir_type_float32);
   nir_pop_if(&b, NULL);

   if (stencil_at_end) {
      b.shader->info.outputs_written |= BITFIELD64_BIT(FRAG_RESULT_STENCIL);
      nir_store_output(&b, nir_imm_int(&b, 7), nir_imm_int(&b, 0),
                       .io_semantics.location = FRAG_RESULT_STENCIL,
                       .src_type = nir_type_uint32);
   }

   agx_nir_lower_discard_zs_emit(b.shader);
   nir_validate_shader(b.shader, "zs emit in control flow");

   nir_function_impl *impl = nir_shader_get_entrypoint(b.shader);
   nir_block *last = nir_impl_last_block(impl);
   unsigned stores = 0, in_last = 0, phi_depth = 0, base = 0;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
         if (in->intrinsic != nir_intrinsic_store_zs_agx)
            continue;
         stores++;
         in_last += block == last;
         base = nir_intrinsic_base(in);
         /* Depth is converted to 32-bit after the merge */
         nir_instr *depth = nir_def_instr(in->src[1].ssa);
         if (depth->type == nir_instr_type_alu)
            depth = nir_def_instr(nir_instr_as_alu(depth)->src[0].src.ssa);
         phi_depth += depth->type == nir_instr_type_phi;
      }
   }

   unsigned expected_base = stencil_at_end ? 3 : 1;
   printf("stencil=%d stores=%u in_last=%u phi_depth=%u base=%u\n",
          stencil_at_end, stores, in_last, phi_depth, base);
   ralloc_free(b.shader);
   return stores == 1 && in_last == 1 && phi_depth == 1 &&
          base == expected_base;
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   bool ok = test(false);
   ok &= test(true);
   glsl_type_singleton_decref();
   return !ok;
}
