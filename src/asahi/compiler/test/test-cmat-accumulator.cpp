/*
 * SPDX-License-Identifier: MIT
 */

#include "agx_compile.h"
extern "C" {
#include "agx_nir.h"
}
#include "compiler/glsl_types.h"
#include "nir.h"
#include "nir_builder.h"
#include <gtest/gtest.h>

static const glsl_type *
cmat_type(enum glsl_base_type element, enum glsl_cmat_use use)
{
   struct glsl_cmat_description desc = {
      .rows = 16,
      .cols = 16,
      .element_type = (uint16_t)element,
      .scope = SCOPE_SUBGROUP,
      .use = (uint8_t)use,
   };
   return glsl_cmat_type(&desc);
}

static nir_deref_instr *
cmat_var(nir_builder *b, enum glsl_base_type element, enum glsl_cmat_use use,
         const char *name)
{
   return nir_build_deref_var(
      b, nir_local_variable_create(b->impl, cmat_type(element, use), name));
}

/* A 16x16x16 multiply-add of half inputs with the given accumulator type, as
 * the flash attention shaders of llama.cpp write it for the PV product.
 */
static unsigned
lowered_muladd_count(enum glsl_base_type accum)
{
   glsl_type_singleton_init_or_ref();
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "cmat");
   b.shader->info.cs.has_cooperative_matrix = true;

   nir_deref_instr *a = cmat_var(&b, GLSL_TYPE_FLOAT16, GLSL_CMAT_USE_A, "a");
   nir_deref_instr *bb = cmat_var(&b, GLSL_TYPE_FLOAT16, GLSL_CMAT_USE_B, "b");
   nir_deref_instr *c = cmat_var(&b, accum, GLSL_CMAT_USE_ACCUMULATOR, "c");
   nir_deref_instr *d = cmat_var(&b, accum, GLSL_CMAT_USE_ACCUMULATOR, "d");

   nir_cmat_construct(&b, &a->def, nir_imm_float16(&b, 1.0f));
   nir_cmat_construct(&b, &bb->def, nir_imm_float16(&b, 1.0f));
   nir_cmat_construct(&b, &c->def,
                      accum == GLSL_TYPE_FLOAT16 ? nir_imm_float16(&b, 0.0f)
                                                 : nir_imm_float(&b, 0.0f));
   nir_cmat_muladd(&b, &d->def, &a->def, &bb->def, &c->def);

   agx_nir_lower_cmat(b.shader);

   unsigned left = 0, agx = 0;
   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            left += intr->intrinsic == nir_intrinsic_cmat_muladd;
            /* The hardware accumulates in 32 bits whatever the matrix type. */
            agx += intr->intrinsic == nir_intrinsic_cmat_muladd_agx &&
                   intr->src[2].ssa->bit_size == 32;
         }
      }
   }
   ralloc_free(b.shader);
   return left == 0 ? agx : 0;
}

TEST(CmatLower, HalfInputsFloatAccumulator)
{
   EXPECT_EQ(lowered_muladd_count(GLSL_TYPE_FLOAT), 1u);
}

TEST(CmatLower, HalfInputsHalfAccumulator)
{
   EXPECT_EQ(lowered_muladd_count(GLSL_TYPE_FLOAT16), 1u);
}
