/* SPDX-License-Identifier: MIT */
#include <gtest/gtest.h>
#include "agx3_uniform_integer_instr.h"
#include "agx_builder.h"
#include "agx_test.h"

TEST(AGX3UniformInteger, LoweringRetainsNativeOperands)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   for (auto size : {AGX_SIZE_16, AGX_SIZE_32}) {
      for (unsigned op = 0; op < 10; ++op) {
         agx_builder *b = agx_test_builder(mem);
         b->shader->key = &key;
         agx_index r = agx_temp(b->shader, size);
         agx_index u = agx_uniform(504, size);
         agx_index dest = agx_temp(b->shader, size);
         agx_instr *I;
         if (op < 3) {
            agx_index rhs = r;
            rhs.neg = op == 1;
            I = agx_iadd_to(b, dest, u, rhs, op == 2 ? 3 : 0);
         } else if (op == 3) {
            I = agx_imad_to(b, dest, u, r, agx_uniform(508, size), 0);
         } else {
            const unsigned tables[] = {0x8, 0xe, 0x6, 0x7, 0x1, 0x2};
            I = agx_bitop_to(b, dest, u, r, tables[op - 4]);
         }
         agx_lower_uniform_sources(b->shader);
         ASSERT_EQ(I->src[0].type, AGX_INDEX_UNIFORM);
         EXPECT_EQ(I->src[0].value, 504u);
         if (op == 3) {
            EXPECT_EQ(I->src[2].type, AGX_INDEX_UNIFORM);
         }

         /* The preserved operation must remain encodable after allocation,
          * including register cache hints and a full dependency mask. */
         I->dest[0] = agx_register(2, size);
         I->src[1].type = AGX_INDEX_REGISTER;
         I->src[1].value = 0;
         I->src[1].cache = true;
         agx3_uniform_integer_desc d;
         ASSERT_TRUE(agx3_uniform_integer_instruction(I, true, &d));
         d.wait_mask = 63;
         uint8_t bytes[16], length;
         EXPECT_TRUE(
            agx3_uniform_integer_encode(key.dev.encoding, &d, bytes, &length));
      }
   }
   ralloc_free(mem);
}

TEST(AGX3UniformInteger, LoweringCopiesUnsupportedDomains)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   for (unsigned variant = 0; variant < 7; ++variant) {
      agx_builder *b = agx_test_builder(mem);
      b->shader->key = &key;
      agx_index r = agx_temp(b->shader, AGX_SIZE_32);
      agx_index u = agx_uniform(12, AGX_SIZE_32);
      if (variant == 0)
         u.size = AGX_SIZE_16;
      if (variant == 1)
         u.value = 511;
      if (variant == 2)
         r = agx_uniform(14, AGX_SIZE_32);
      if (variant == 3)
         u.neg = true;
      if (variant == 4)
         u.abs = true;
      agx_instr *I = agx_iadd_to(b, agx_temp(b->shader, AGX_SIZE_32), u, r,
                                 variant == 5 ? 2 : 0);
      if (variant == 6)
         I->saturate = true;
      agx3_uniform_integer_desc d;
      EXPECT_FALSE(agx3_uniform_integer_instruction(I, false, &d));
      agx_lower_uniform_sources(b->shader);
      EXPECT_NE(I->src[0].type, AGX_INDEX_UNIFORM);
   }
   ralloc_free(mem);
}
