/* SPDX-License-Identifier: MIT */
#include "nir_builder.h"
#include "poly_nir.h"

#include <stdio.h>

static bool
check_barrier(const char *name, nir_variable_mode modes,
              mesa_scope memory_scope, nir_variable_mode expected,
              bool removed)
{
   static const nir_shader_compiler_options options = {0};
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_TESS_CTRL, &options, "%s", name);
   b.shader->info.tess.tcs_vertices_out = 3;
   nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
               .memory_scope = memory_scope,
               .memory_semantics = NIR_MEMORY_ACQUIRE | NIR_MEMORY_RELEASE,
               .memory_modes = modes);

   poly_nir_lower_tcs(b.shader);
   unsigned count = 0;
   bool valid = true;
   nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_barrier)
            continue;
         ++count;
         valid &= nir_intrinsic_memory_modes(intr) == expected;
         valid &= nir_intrinsic_memory_scope(intr) == memory_scope;
         valid &= nir_intrinsic_execution_scope(intr) == SCOPE_WORKGROUP;
         valid &= nir_intrinsic_memory_semantics(intr) ==
                  (NIR_MEMORY_ACQUIRE | NIR_MEMORY_RELEASE);
      }
   }
   valid &= count == (removed ? 0 : 1);
   if (!valid)
      fprintf(stderr, "%s: TCS barrier did not preserve lowered memory ordering\n",
              name);
   ralloc_free(b.shader);
   return valid;
}

int
main(void)
{
   bool valid = true;
   valid &= check_barrier("patch outputs", nir_var_shader_out, SCOPE_WORKGROUP,
                          nir_var_mem_global, false);
   valid &= check_barrier("outputs and images",
                          nir_var_shader_out | nir_var_image,
                          SCOPE_DEVICE, nir_var_mem_global | nir_var_image,
                          false);
   valid &= check_barrier("existing global ordering", nir_var_mem_global,
                          SCOPE_WORKGROUP, nir_var_mem_global, false);
   valid &= check_barrier("image ordering", nir_var_image, SCOPE_DEVICE,
                          nir_var_image, false);
   valid &= check_barrier("execution only", 0, SCOPE_NONE, 0, true);
   return valid ? 0 : 1;
}
