/*
 * Copyright 2023 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "agx_builder.h"
#include "agx_compile.h"
#include "agx_compiler.h"
#include "agx_test.h"

#include "util/macros.h"
#include <gtest/gtest.h>

#define CASE(instr, expected)                                                  \
   do {                                                                        \
      agx_builder *A = agx_test_builder(mem_ctx);                              \
      agx_builder *B = agx_test_builder(mem_ctx);                              \
      {                                                                        \
         agx_builder *b = A;                                                   \
         instr;                                                                \
      }                                                                        \
      {                                                                        \
         agx_builder *b = B;                                                   \
         expected;                                                             \
      }                                                                        \
      A->shader->scratch_size_B = 1024;                                        \
      agx_lower_spill(A->shader);                                              \
      ASSERT_SHADER_EQUAL(A->shader, B->shader);                               \
   } while (0)

class LowerSpill : public testing::Test {
 protected:
   LowerSpill()
   {
      mem_ctx = ralloc_context(NULL);

      wx = agx_register(0, AGX_SIZE_32);
      hy = agx_register(2, AGX_SIZE_16);

      mw4 = agx_memory_register(0, AGX_SIZE_32);
      mh4 = agx_memory_register(0, AGX_SIZE_16);
      mw4.channels_m1 = 4 - 1;
      mh4.channels_m1 = 4 - 1;

      wx4 = wx;
      wx4.channels_m1 = 4 - 1;

      hy4 = hy;
      hy4.channels_m1 = 4 - 1;
   }

   ~LowerSpill()
   {
      ralloc_free(mem_ctx);
   }

   void *mem_ctx;
   agx_index wx, hy, wx4, hy4;
   agx_index mw4, mh4;

   unsigned scalar = BITFIELD_MASK(1);
   unsigned vec4 = BITFIELD_MASK(4);

   enum agx_format i16 = AGX_FORMAT_I16;
   enum agx_format i32 = AGX_FORMAT_I32;
};

TEST_F(LowerSpill, ScalarSpills)
{
   CASE(agx_mov_to(b, agx_memory_register(11, AGX_SIZE_16), hy),
        agx_stack_store(b, hy, agx_immediate(22), i16, scalar));

   CASE(agx_mov_to(b, agx_memory_register(18, AGX_SIZE_32), wx),
        agx_stack_store(b, wx, agx_immediate(36), i32, scalar));
}

TEST_F(LowerSpill, ScalarFills)
{
   CASE(agx_mov_to(b, hy, agx_memory_register(11, AGX_SIZE_16)),
        agx_stack_load_to(b, hy, agx_immediate(22), i16, scalar));

   CASE(agx_mov_to(b, wx, agx_memory_register(18, AGX_SIZE_32)),
        agx_stack_load_to(b, wx, agx_immediate(36), i32, scalar));
}

TEST_F(LowerSpill, VectorSpills)
{
   CASE(agx_mov_to(b, mh4, hy4),
        agx_stack_store(b, hy4, agx_immediate(0), i16, vec4));

   CASE(agx_mov_to(b, mw4, wx4),
        agx_stack_store(b, wx4, agx_immediate(0), i32, vec4));
}

TEST_F(LowerSpill, VectorFills)
{
   CASE(agx_mov_to(b, hy4, mh4),
        agx_stack_load_to(b, hy4, agx_immediate(0), i16, vec4));

   CASE(agx_mov_to(b, wx4, mw4),
        agx_stack_load_to(b, wx4, agx_immediate(0), i32, vec4));
}

TEST_F(LowerSpill, ScalarSpill64)
{
   CASE(agx_mov_to(b, agx_memory_register(16, AGX_SIZE_64),
                   agx_register(8, AGX_SIZE_64)),
        agx_stack_store(b, agx_register(8, AGX_SIZE_64), agx_immediate(32), i32,
                        BITFIELD_MASK(2)));
}

TEST_F(LowerSpill, ScalarFill64)
{
   CASE(agx_mov_to(b, agx_register(16, AGX_SIZE_64),
                   agx_memory_register(8, AGX_SIZE_64)),
        agx_stack_load_to(b, agx_register(16, AGX_SIZE_64), agx_immediate(16),
                          i32, BITFIELD_MASK(2)));
}

TEST_F(LowerSpill, Vec6Spill)
{
   CASE(
      {
         agx_index mvec6 = agx_memory_register(16, AGX_SIZE_32);
         agx_index vec6 = agx_register(8, AGX_SIZE_32);
         vec6.channels_m1 = 6 - 1;
         mvec6.channels_m1 = 6 - 1;

         agx_mov_to(b, mvec6, vec6);
      },
      {
         agx_index vec4 = agx_register(8, AGX_SIZE_32);
         agx_index vec2 = agx_register(8 + (4 * 2), AGX_SIZE_32);
         vec4.channels_m1 = 4 - 1;
         vec2.channels_m1 = 2 - 1;

         agx_stack_store(b, vec4, agx_immediate(32), i32, BITFIELD_MASK(4));
         agx_stack_store(b, vec2, agx_immediate(32 + 4 * 4), i32,
                         BITFIELD_MASK(2));
      });
}

TEST_F(LowerSpill, Vec6Fill)
{
   CASE(
      {
         agx_index mvec6 = agx_memory_register(16, AGX_SIZE_32);
         agx_index vec6 = agx_register(8, AGX_SIZE_32);
         vec6.channels_m1 = 6 - 1;
         mvec6.channels_m1 = 6 - 1;

         agx_mov_to(b, vec6, mvec6);
      },
      {
         agx_index vec4 = agx_register(8, AGX_SIZE_32);
         agx_index vec2 = agx_register(8 + (4 * 2), AGX_SIZE_32);
         vec4.channels_m1 = 4 - 1;
         vec2.channels_m1 = 2 - 1;

         agx_stack_load_to(b, vec4, agx_immediate(32), i32, BITFIELD_MASK(4));
         agx_stack_load_to(b, vec2, agx_immediate(32 + 4 * 4), i32,
                           BITFIELD_MASK(2));
      });
}

TEST_F(LowerSpill, AGX3SplitsVectorStackAccess)
{
   agx_builder *A = agx_test_builder(mem_ctx);
   agx_builder *B = agx_test_builder(mem_ctx);

   {
      agx_builder *b = A;
      agx_index h3 = agx_register(6, AGX_SIZE_16);
      h3.channels_m1 = 3 - 1;
      agx_index w2 = agx_register(4, AGX_SIZE_32);
      w2.channels_m1 = 2 - 1;

      agx_stack_store(b, h3, agx_immediate(12), i16, BITFIELD_MASK(3));
      agx_stack_load_to(b, w2, agx_immediate(8), i32, BITFIELD_MASK(2));
   }
   {
      agx_builder *b = B;
      for (unsigned c = 0; c < 3; ++c) {
         agx_stack_store(b, agx_register(6 + c, AGX_SIZE_16),
                         agx_immediate(12 + 2 * c), i16, 1);
      }
      for (unsigned c = 0; c < 2; ++c) {
         agx_stack_load_to(b, agx_register(4 + 2 * c, AGX_SIZE_32),
                           agx_immediate(8 + 4 * c), i32, 1);
      }
   }

   agx3_split_vector_stack_access(A->shader);
   ASSERT_SHADER_EQUAL(A->shader, B->shader);
}

class M5SpillConstantPhi : public testing::Test {
protected:
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   M5SpillConstantPhi() {
      key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
      b->shader->key = &key;
   }
   ~M5SpillConstantPhi() { ralloc_free(mem); }
   agx_index literal(uint64_t bits, enum agx_size size = AGX_SIZE_32) {
      agx_index x = agx_temp(b->shader, size);
      agx_mov_imm_to(b, x, bits);
      return x;
   }
   void join() { b->cursor = agx_after_block(agx_test_block(b->shader)); }
   agx_instr *phi(agx_index x, agx_index y, agx_index dest) {
      agx_instr *I = agx_phi_to(b, dest, 2);
      I->src[0] = x; I->src[1] = y;
      return I;
   }
   agx_instr *definition(agx_index x) {
      agx_foreach_instr_global(b->shader, I) {
         if (I->nr_dests == 1 && agx_is_equiv(I->dest[0], x)) return I;
      }
      return nullptr;
   }
};

TEST_F(M5SpillConstantPhi, SameLiteralCopiesJoin)
{
   agx_index x = literal(0x3f800000), y = literal(0x3f800000);
   agx_index copy = agx_mov(b, x);
   join();
   agx_index out = agx_temp(b->shader, AGX_SIZE_32);
   phi(copy, y, out);
   ASSERT_TRUE(agx_fold_m5_spill_constant_phis(b->shader));
   ASSERT_EQ(definition(out)->op, AGX_OPCODE_MOV_IMM);
   EXPECT_EQ(definition(out)->imm, 0x3f800000u);
}

TEST_F(M5SpillConstantPhi, GroundedLoopWebPreservesLiteralPayload)
{
   agx_index root = literal(0x7fc01234);
   join();
   agx_index a = agx_temp(b->shader, AGX_SIZE_32);
   agx_index c = agx_temp(b->shader, AGX_SIZE_32);
   phi(root, c, a); phi(root, a, c);
   ASSERT_TRUE(agx_fold_m5_spill_constant_phis(b->shader));
   EXPECT_EQ(definition(a)->op, AGX_OPCODE_MOV_IMM);
   EXPECT_EQ(definition(c)->op, AGX_OPCODE_MOV_IMM);
   EXPECT_EQ(definition(a)->imm, 0x7fc01234u);
   EXPECT_EQ(definition(c)->imm, 0x7fc01234u);
}

TEST_F(M5SpillConstantPhi, ChangingLoopInputRemainsPhi)
{
   agx_index zero = literal(0), one = literal(1);
   join();
   agx_index a = agx_temp(b->shader, AGX_SIZE_32);
   agx_index c = agx_temp(b->shader, AGX_SIZE_32);
   phi(zero, c, a); phi(one, a, c);
   EXPECT_FALSE(agx_fold_m5_spill_constant_phis(b->shader));
   EXPECT_EQ(definition(a)->op, AGX_OPCODE_PHI);
   EXPECT_EQ(definition(c)->op, AGX_OPCODE_PHI);
}

TEST_F(M5SpillConstantPhi, RootFreeCycleCannotBorrowAnotherRootsValue)
{
   agx_index root = literal(1);
   join();
   agx_index a = agx_temp(b->shader, AGX_SIZE_32);
   agx_index c = agx_temp(b->shader, AGX_SIZE_32);
   agx_index out = agx_temp(b->shader, AGX_SIZE_32);
   phi(c, c, a); phi(a, a, c); phi(root, a, out);
   EXPECT_FALSE(agx_fold_m5_spill_constant_phis(b->shader));
   EXPECT_EQ(definition(out)->op, AGX_OPCODE_PHI);
}

TEST_F(M5SpillConstantPhi, RejectsUnprovenOrModifiedSources)
{
   for (unsigned mode = 0; mode < 9; ++mode) {
      agx_builder *old = b;
      b = agx_test_builder(mem); b->shader->key = &key;
      agx_index root = literal(1), other = root;
      switch (mode) {
      case 0: other.abs = true; break;
      case 1: other.neg = true; break;
      case 2: other.memory = true; break;
      case 3: other.size = AGX_SIZE_16; break;
      case 4: other.channels_m1 = 1; break;
      case 5: other = agx_uniform(2, AGX_SIZE_32); break;
      case 6: other = agx_register(2, AGX_SIZE_32); break;
      case 7: other = agx_immediate(1); break;
      case 8: other = agx_undef(AGX_SIZE_32); break;
      }
      join();
      agx_index out = agx_temp(b->shader, AGX_SIZE_32);
      phi(root, other, out);
      EXPECT_FALSE(agx_fold_m5_spill_constant_phis(b->shader));
      EXPECT_EQ(definition(out)->op, AGX_OPCODE_PHI);
      b = old;
   }
}

TEST_F(M5SpillConstantPhi, LiteralWidthsAndSignedZeroBits)
{
   agx_index zero = literal(0), negzero = literal(0x80000000);
   agx_index short1 = literal(0x10001, AGX_SIZE_16);
   agx_index short2 = literal(1, AGX_SIZE_16);
   join();
   agx_index wide = agx_temp(b->shader, AGX_SIZE_32);
   agx_index narrow = agx_temp(b->shader, AGX_SIZE_16);
   phi(zero, negzero, wide); phi(short1, short2, narrow);
   ASSERT_TRUE(agx_fold_m5_spill_constant_phis(b->shader));
   EXPECT_EQ(definition(wide)->op, AGX_OPCODE_PHI);
   EXPECT_EQ(definition(narrow)->op, AGX_OPCODE_MOV_IMM);
   EXPECT_EQ(definition(narrow)->imm, 1u);
}

TEST_F(M5SpillConstantPhi, PreservesParallelPhiPrefix)
{
   agx_index zero = literal(0), one = literal(1);
   join();
   agx_index folded = agx_temp(b->shader, AGX_SIZE_32);
   agx_index varying = agx_temp(b->shader, AGX_SIZE_32);
   phi(zero, zero, folded); phi(zero, one, varying);
   agx_index result = agx_iadd(b, folded, varying, 0);
   ASSERT_TRUE(agx_fold_m5_spill_constant_phis(b->shader));
   EXPECT_EQ(definition(varying)->op, AGX_OPCODE_PHI);
   EXPECT_EQ(definition(folded)->op, AGX_OPCODE_MOV_IMM);
   EXPECT_EQ(definition(result)->op, AGX_OPCODE_IADD);
   bool body = false;
   agx_foreach_block(b->shader, block) {
      body = false;
      agx_foreach_instr_in_block(block, I) {
         if (I->op != AGX_OPCODE_PHI) body = true;
         else EXPECT_FALSE(body);
      }
   }
}

TEST_F(M5SpillConstantPhi, G17PAndConstantProgramsRemainUnchanged)
{
   agx_index root = literal(1); join();
   agx_index out = agx_temp(b->shader, AGX_SIZE_32);
   phi(root, root, out);
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   EXPECT_FALSE(agx_fold_m5_spill_constant_phis(b->shader));
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   b->shader->is_preamble = true;
   EXPECT_FALSE(agx_fold_m5_spill_constant_phis(b->shader));
   EXPECT_EQ(definition(out)->op, AGX_OPCODE_PHI);
}
