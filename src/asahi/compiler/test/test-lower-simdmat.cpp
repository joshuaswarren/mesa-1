/*
 * Copyright 2026 Joshua Warren
 * SPDX-License-Identifier: MIT
 */

#include <cstdlib>

#include "nir_builder.h"

extern "C" {
#include "agx_compiler.h"
}

#include <gtest/gtest.h>

/*
 * With AGX_HWMAT_COLPAIR set, a column-major 8x8 fp16 fragment loads as one
 * vec2 (two adjacent rows of one column) plus one quad swap, not as two
 * strided scalar loads.
 */
TEST(LowerSimdmat, ColumnMajorPairLoad)
{
   setenv("AGX_HWMAT_COLPAIR", "1", 1);
   static const nir_shader_compiler_options options = {};
   glsl_type_singleton_init_or_ref();
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &options, "simdmat test");

   struct glsl_cmat_description desc = {};
   desc.rows = 8;
   desc.cols = 8;
   desc.element_type = GLSL_TYPE_FLOAT16;
   desc.scope = SCOPE_SUBGROUP;
   desc.use = GLSL_CMAT_USE_B;

   nir_variable *var =
      nir_local_variable_create(b.impl, glsl_cmat_type(&desc), "m");
   nir_deref_instr *mat = nir_build_deref_var(&b, var);
   nir_deref_instr *ptr =
      nir_build_deref_cast(&b, nir_imm_int64(&b, 0x10000), nir_var_mem_global,
                           glsl_float16_t_type(), 2);
   nir_cmat_load(&b, &mat->def, &ptr->def, nir_imm_int(&b, 64),
                 GLSL_MATRIX_LAYOUT_COLUMN_MAJOR);

   EXPECT_TRUE(agx_nir_lower_simdmat(b.shader, 32));
   nir_validate_shader(b.shader, "after agx_nir_lower_simdmat");

   unsigned vec2_loads = 0, scalar_loads = 0, swaps = 0;
   nir_foreach_block(block, b.impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_quad_swap_vertical) {
            swaps++;
         } else if (intr->intrinsic == nir_intrinsic_load_deref &&
                    nir_deref_mode_is(nir_src_as_deref(intr->src[0]),
                                      nir_var_mem_global)) {
            if (intr->def.num_components == 2)
               vec2_loads++;
            else
               scalar_loads++;
         }
      }
   }
   EXPECT_EQ(vec2_loads, 1u);
   EXPECT_EQ(scalar_loads, 0u);
   EXPECT_EQ(swaps, 1u);

   ralloc_free(b.shader);
   glsl_type_singleton_decref();
   unsetenv("AGX_HWMAT_COLPAIR");
}
