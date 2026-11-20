/* Copyright 2026 Aurora Silicon; SPDX-License-Identifier: MIT */

#include <gtest/gtest.h>
#include "agx_test.h"

extern "C" {
#include "agx_nir.h"
}

static agx_instr *
def_of(agx_context *s, agx_index value)
{
   agx_foreach_instr_global(s, I) {
      agx_foreach_dest(I, d) {
         if (agx_is_equiv(I->dest[d], value))
            return I;
      }
   }
   return NULL;
}

TEST(G18GIcmpsel, SignedHalfUniformComparandSignExtends)
{
   for (bool is_signed : {false, true}) {
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      struct agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
      key.dev.emit_agx3d = true;
      b->shader->key = &key;

      /* r60 (32-bit) SLT/ULT u90 (16-bit) ? r62 : r63 */
      agx_icmpsel_to(b, agx_register(41, AGX_SIZE_32),
                     agx_register(60, AGX_SIZE_32),
                     agx_uniform(90, AGX_SIZE_16),
                     agx_register(62, AGX_SIZE_32),
                     agx_register(63, AGX_SIZE_32),
                     is_signed ? AGX_ICOND_SLT : AGX_ICOND_ULT);
      agx_stop(b);

      ASSERT_TRUE(agx_lower_agx3_g18g_half_icmpsel(b->shader));

      agx_instr *sel = NULL;
      agx_foreach_instr_global(b->shader, I) {
         if (I->op == AGX_OPCODE_ICMPSEL)
            sel = I;
      }
      ASSERT_NE(sel, nullptr);

      agx_instr *def = def_of(b->shader, sel->src[1]);
      ASSERT_NE(def, nullptr);
      if (is_signed) {
         ASSERT_EQ(def->op, AGX_OPCODE_SIGNEXT)
            << "signed select must sign-extend the uniform comparand";
         agx_instr *inner = def_of(b->shader, def->src[0]);
         ASSERT_NE(inner, nullptr);
         EXPECT_EQ(inner->op, AGX_OPCODE_MOV);
         EXPECT_TRUE(agx_is_equiv(inner->src[0],
                                  agx_uniform(90, AGX_SIZE_16)));
      } else {
         ASSERT_EQ(def->op, AGX_OPCODE_MOV)
            << "unsigned select must zero-extend the uniform comparand";
         EXPECT_TRUE(agx_is_equiv(def->src[0],
                                  agx_uniform(90, AGX_SIZE_16)));
      }
      ralloc_free(mem);
   }
}

TEST(G18GFtz, LoadsKeepBitsAndFloatOperationsFlush)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
                                                   &agx_nir_options,
                                                   "g18-ftz-scope");
   nir_def *offset = nir_imm_int(&b, 0);
   nir_def *input = nir_load_input(&b, 1, 32, offset);
   nir_io_semantics semantics = {};
   semantics.location = VARYING_SLOT_VAR0;
   nir_intrinsic_set_io_semantics(nir_def_as_intrinsic(input), semantics);
   nir_intrinsic_set_dest_type(nir_def_as_intrinsic(input), nir_type_float32);

   nir_def *raw = nir_load_ubo(&b, 1, 32, offset, offset);
   nir_intrinsic_set_range(nir_def_as_intrinsic(raw), 4);
   nir_intrinsic_set_align(nir_def_as_intrinsic(raw), 4, 0);
   nir_def *negative = nir_imm_int(&b, 0x80000001u);
   nir_def *positive = nir_imm_int(&b, 1);
   nir_def *values = nir_vec4(&b, input, nir_fneg(&b, raw), nir_fabs(&b, raw),
                              nir_fadd(&b, raw, raw));
   nir_intrinsic_instr *store = agx_test_store_output(
      &b, values, offset, nir_type_float32, FRAG_RESULT_DATA0);

   agx_nir_lower_g18_ftz(b.shader);
   nir_validate_shader(b.shader, "g18-ftz-scope");

   nir_def_rewrite_uses(input, positive);
   nir_def_rewrite_uses(raw, negative);
   nir_opt_constant_folding(b.shader);

   ASSERT_TRUE(nir_src_is_const(store->src[0]));
   const nir_const_value *out = nir_src_as_const_value(store->src[0]);
   EXPECT_EQ(out[0].u32, 1u);
   EXPECT_EQ(out[1].u32, 0u);
   EXPECT_EQ(out[2].u32, 0u);
   EXPECT_EQ(out[3].u32, 0x80000000u);
   ralloc_free(b.shader);
}
