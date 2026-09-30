/*
 * Copyright 2026 Joshua Warren
 * SPDX-License-Identifier: MIT
 */

#include "nir_builder.h"

extern "C" {
#include "agx_compiler.h"
}

#include <gtest/gtest.h>

/*
 * The software cooperative matrix lowering must index memory at the width of
 * the pointer. A PhysicalStorageBuffer pointer is 64-bit and the SPIR-V stride
 * is 32-bit.
 */
TEST(LowerCmat, PhysicalStorageBufferLoadStore)
{
   static const nir_shader_compiler_options options = {};
   glsl_type_singleton_init_or_ref();
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &options, "lower cmat test");

   struct glsl_cmat_description desc = {};
   desc.rows = 8;
   desc.cols = 8;
   desc.element_type = GLSL_TYPE_FLOAT;
   desc.scope = SCOPE_SUBGROUP;
   desc.use = GLSL_CMAT_USE_ACCUMULATOR;

   nir_variable *var =
      nir_local_variable_create(b.impl, glsl_cmat_type(&desc), "m");
   nir_deref_instr *mat = nir_build_deref_var(&b, var);
   nir_deref_instr *ptr =
      nir_build_deref_cast(&b, nir_imm_int64(&b, 0x10000), nir_var_mem_global,
                           glsl_float_type(), 4);
   nir_def *stride = nir_imm_int(&b, 8);

   nir_cmat_load(&b, &mat->def, &ptr->def, stride,
                 GLSL_MATRIX_LAYOUT_ROW_MAJOR);
   nir_cmat_store(&b, &ptr->def, &mat->def, stride,
                  GLSL_MATRIX_LAYOUT_ROW_MAJOR);

   EXPECT_TRUE(agx_nir_lower_cmat(b.shader));
   nir_validate_shader(b.shader, "after agx_nir_lower_cmat");

   ralloc_free(b.shader);
   glsl_type_singleton_decref();
}
