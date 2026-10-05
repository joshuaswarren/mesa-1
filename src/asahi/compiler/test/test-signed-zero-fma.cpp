/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "agx_test.h"
#include <gtest/gtest.h>

TEST(SignedZeroFMA, PreservesProductSignAndAddendModifiers)
{
   for (unsigned imm = 0; imm < 256; imm++) {
      for (unsigned zero : {0u, 128u}) {
         for (bool swap : {false, true}) {
            for (unsigned flags = 0; flags < 16; flags++) {
               if ((flags & 12) == 12)
                  continue;
               SCOPED_TRACE(imm);
               SCOPED_TRACE(zero);
               SCOPED_TRACE(swap);
               SCOPED_TRACE(flags);
               void *mem = ralloc_context(NULL);
               agx_builder *b = agx_test_builder(mem);
               agx_shader_key key = {};
               key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
               b->shader->key = &key;
               agx_index addend = agx_register(48, AGX_SIZE_32);
               addend.abs = flags & 1;
               addend.neg = (flags >> 1) & 1;
               addend.cache = (flags >> 2) & 1;
               addend.discard = (flags >> 3) & 1;
               agx_instr *I = agx_ffma_to(
                  b, agx_register(32, AGX_SIZE_32),
                  agx_immediate(swap ? zero : imm),
                  agx_immediate(swap ? imm : zero), addend);
               agx_lower_agx3_fma_sources(b->shader);
               ASSERT_EQ(I->op, AGX_OPCODE_FADD);
               EXPECT_EQ(I->nr_srcs, 2u);
               EXPECT_EQ(memcmp(&I->src[0], &addend, sizeof(addend)), 0);
               EXPECT_EQ(I->src[1].type, AGX_INDEX_IMMEDIATE);
               EXPECT_EQ(I->src[1].value, (imm ^ zero) & 128);
               EXPECT_FALSE(I->src[1].neg);
               EXPECT_EQ(I->dest[0].value, 32u);
               EXPECT_EQ(I->dest[0].size, AGX_SIZE_32);
               ralloc_free(mem);
            }
         }
      }
   }
}

TEST(SignedZeroFMA, RetainsUnsupportedForms)
{
   for (unsigned mode = 0; mode < 10; mode++) {
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.encoding = mode == 6 ? AGX_ISA_ENCODING_AGX2
                                  : AGX_ISA_ENCODING_AGX3B;
      b->shader->key = &key;
      agx_instr *I = agx_ffma_to(
         b, agx_register(32, AGX_SIZE_32), agx_immediate(128),
         agx_immediate(64), agx_register(48, AGX_SIZE_32));
      switch (mode) {
      case 0: I->saturate = true; break;
      case 1: I->src[0].neg = true; break;
      case 2: I->src[0].abs = true; break;
      case 3: I->src[0].memory = true; break;
      case 4: I->src[0] = agx_immediate(0x100); break;
      case 5: I->src[0] = agx_immediate(0x38); break;
      case 6: break;
      case 7: I->src[0] = agx_register(20, AGX_SIZE_32); break;
      case 8: I->dest[0].size = AGX_SIZE_16; break;
      case 9: I->dest[0].channels_m1 = 1; break;
      }
      agx_lower_agx3_fma_sources(b->shader);
      EXPECT_EQ(I->op, AGX_OPCODE_FFMA) << mode;
      ralloc_free(mem);
   }
}

/* Mixed FMA shapes the G17P packer encodes with their modifiers fused
 * (agx3_falu_fma_addend_imm_mod / agx3_falu_fma_mul_imm_mod) keep them; other
 * mixed shapes still have their modifiers materialized as sign bits. */
TEST(SignedZeroFMA, KeepsNativeModifierShapes)
{
   for (unsigned mode = 0; mode < 6; mode++) {
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      b->shader->key = &key;
      agx_index x = agx_temp(b->shader, AGX_SIZE_32);
      agx_index y = agx_temp(b->shader, AGX_SIZE_32);
      agx_index h = agx_temp(b->shader, AGX_SIZE_16);
      (void)h;
      agx_instr *I = NULL;
      bool keep = true;
      switch (mode) {
      case 0: /* fma(-x, y, 1.0) */
         I = agx_ffma_to(b, agx_temp(b->shader, AGX_SIZE_32), agx_neg(x), y,
                         agx_immediate(48));
         break;
      case 1: /* fma(x, |y|, 0.5) */
         I = agx_ffma_to(b, agx_temp(b->shader, AGX_SIZE_32), x, agx_abs(y),
                         agx_immediate(32));
         break;
      case 2: /* fma(2.0, -x, |y|) */
         I = agx_ffma_to(b, agx_temp(b->shader, AGX_SIZE_32), agx_immediate(64),
                         agx_neg(x), agx_abs(y));
         break;
      case 3: /* fma(-x, 2.0, half) */
         I = agx_ffma_to(b, agx_temp(b->shader, AGX_SIZE_32), agx_neg(x),
                         agx_immediate(64), h);
         break;
      case 4: /* fma(2.0, -x, 1.0): two minifloats, still materialized */
         I = agx_ffma_to(b, agx_temp(b->shader, AGX_SIZE_32), agx_immediate(64),
                         agx_neg(x), agx_immediate(48));
         keep = false;
         break;
      case 5: /* fma(|x|, 2.0, 1.0): likewise */
         I = agx_ffma_to(b, agx_temp(b->shader, AGX_SIZE_32), agx_abs(x),
                         agx_immediate(64), agx_immediate(48));
         keep = false;
         break;
      }
      agx_lower_agx3_fma_sources(b->shader);
      bool modified = false;
      for (unsigned s = 0; s < 3; ++s)
         modified |= I->src[s].abs || I->src[s].neg;
      EXPECT_EQ(modified, keep) << mode;
      ralloc_free(mem);
   }
}
