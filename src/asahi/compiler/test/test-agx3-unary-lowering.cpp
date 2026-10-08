/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include <gtest/gtest.h>
#include "agx_builder.h"
#include "agx_test.h"

/* Refused shape: roundeven d32 <- s16. Copy propagation folds a widening
 * source move into the rounding instruction, but G17P's rounding forms have
 * width-matched operands. Cover the exact instruction and sign modifiers. */
TEST(AGX3UnaryLowering, RoundevenHalfSourceToWordResult)
{
   void *mem = ralloc_context(nullptr);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   for (auto stage : {MESA_SHADER_COMPUTE, MESA_SHADER_FRAGMENT}) {
      for (unsigned signs = 0; signs < 4; ++signs) {
         agx_builder *b = agx_test_builder(mem);
         b->shader->key = &key;
         b->shader->stage = stage;
         agx_index half = agx_temp(b->shader, AGX_SIZE_16);
         half.abs = signs & 1;
         half.neg = signs & 2;
         agx_instr *round = agx_roundeven_to(
            b, agx_temp(b->shader, AGX_SIZE_32), half);
         agx_lower_agx3_mixed_width_unary(b->shader);
         ASSERT_EQ(round->src[0].size, AGX_SIZE_32);
         EXPECT_FALSE(round->src[0].abs);
         EXPECT_FALSE(round->src[0].neg);
         agx_instr *move = list_entry(round->link.prev, agx_instr, link);
         EXPECT_EQ(move->op, AGX_OPCODE_FADD);
         EXPECT_EQ(move->src[0].size, AGX_SIZE_16);
         EXPECT_EQ(move->src[0].abs, bool(signs & 1));
         EXPECT_EQ(move->src[0].neg, bool(signs & 2));
         EXPECT_TRUE(agx_is_equiv(move->dest[0], round->src[0]));
      }
   }
   ralloc_free(mem);
}

TEST(AGX3UnaryLowering, PreservesMatchedWidthsAndOtherGenerations)
{
   void *mem = ralloc_context(nullptr);
   for (auto encoding : {AGX_ISA_ENCODING_AGX2, AGX_ISA_ENCODING_AGX3B}) {
      agx_shader_key key = {};
      key.dev.encoding = encoding;
      for (auto size : {AGX_SIZE_16, AGX_SIZE_32}) {
         agx_builder *b = agx_test_builder(mem);
         b->shader->key = &key;
         agx_index source = agx_temp(b->shader, size);
         agx_instr *round = agx_roundeven_to(b, agx_temp(b->shader, size), source);
         agx_lower_agx3_mixed_width_unary(b->shader);
         EXPECT_TRUE(agx_is_equiv(round->src[0], source));
      }
   }
   ralloc_free(mem);
}
