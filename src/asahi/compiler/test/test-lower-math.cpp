/*
 * Copyright 2026 Joshua Warren
 * SPDX-License-Identifier: MIT
 */

#include "nir_builder.h"

extern "C" {
#include "agx_nir.h"
}

#include <gtest/gtest.h>
#include <algorithm>
#include <map>
#include <vector>

/*
 * Behavioral pins for the float-control keying in agx_nir_lower_math.c.
 *
 * Division and reciprocal do not key on float controls. Their bits are pinned
 * on hardware by tests/precise_math.py, ulp_bound.py and div_window.py; here
 * the pin is that DenormFlushToZero emits exactly the default lowering.
 *
 * The DenormFlushToZero sin test pins the one keyed change: a zero-comparing
 * subnormal argument must produce a sign-preserving flushed zero instead of
 * leaking the unflushed argument through the select.
 */

class LowerMathTest : public ::testing::Test {
 protected:
   LowerMathTest()
   {
      mem_ctx = ralloc_context(NULL);
      b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options,
                                         "lower math test");
   }

   ~LowerMathTest() override { ralloc_free(mem_ctx); }

   void *mem_ctx;
   nir_builder b;
   static const nir_shader_compiler_options options;

   void set_mode(unsigned mode)
   {
      b.shader->info.float_controls_execution_mode = mode;
   }

   unsigned count_alu(nir_op op)
   {
      unsigned n = 0;
      nir_foreach_function_impl(impl, b.shader) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type == nir_instr_type_alu &&
                   nir_instr_as_alu(instr)->op == op)
                  n++;
            }
         }
      }
      return n;
   }

   unsigned count_iand_imm(uint32_t mask)
   {
      unsigned n = 0;
      nir_foreach_function_impl(impl, b.shader) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_alu)
                  continue;

               nir_alu_instr *alu = nir_instr_as_alu(instr);
               if (alu->op == nir_op_iand &&
                   nir_src_is_const(alu->src[1].src) &&
                   nir_src_as_uint(alu->src[1].src) == mask)
                  n++;
            }
         }
      }
      return n;
   }

   /* ALU op histogram of fdiv (or frcp) lowered under one float-control mode */
   std::map<nir_op, unsigned> lowered_division(unsigned mode, bool reciprocal)
   {
      nir_builder s = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                     &options, "lower fdiv");
      s.shader->info.float_controls_execution_mode = mode;
      nir_def *x = nir_undef(&s, 1, 32);
      if (reciprocal)
         nir_frcp(&s, x);
      else
         nir_fdiv(&s, nir_undef(&s, 1, 32), x);

      EXPECT_TRUE(agx_nir_lower_fdiv(s.shader));

      std::map<nir_op, unsigned> ops;
      nir_foreach_function_impl(impl, s.shader) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type == nir_instr_type_alu)
                  ops[nir_instr_as_alu(instr)->op]++;
            }
         }
      }
      ralloc_free(s.shader);
      return ops;
   }
};

const nir_shader_compiler_options LowerMathTest::options = {};

TEST_F(LowerMathTest, DivFtzModeUnchanged)
{
   auto ftz = lowered_division(FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP32, false);
   EXPECT_EQ(ftz.count(nir_op_fdiv), 0u);
   EXPECT_EQ(ftz, lowered_division(0, false));
}

TEST_F(LowerMathTest, FrcpFtzModeUnchanged)
{
   auto ftz = lowered_division(FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP32, true);
   EXPECT_EQ(ftz.count(nir_op_frcp), 1u);
   EXPECT_EQ(ftz, lowered_division(0, true));
}

TEST_F(LowerMathTest, SinDefaultKeepsSelect)
{
   set_mode(0);
   nir_def *x = nir_undef(&b, 1, 32);
   nir_fsin(&b, x);

   agx_nir_lower_sincos(b.shader);

   EXPECT_EQ(count_alu(nir_op_feq), 1u);
   EXPECT_EQ(count_iand_imm(0x80000000), 0u);
}

TEST_F(LowerMathTest, SinFtzReturnsSignedZero)
{
   set_mode(FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP32);
   nir_def *x = nir_undef(&b, 1, 32);
   nir_fsin(&b, x);

   agx_nir_lower_sincos(b.shader);

   EXPECT_EQ(count_alu(nir_op_feq), 1u);
   EXPECT_EQ(count_iand_imm(0x80000000), 1u);
}

TEST_F(LowerMathTest, SinPreserveKeepsSelect)
{
   set_mode(FLOAT_CONTROLS_DENORM_PRESERVE_FP32);
   nir_def *x = nir_undef(&b, 1, 32);
   nir_fsin(&b, x);

   agx_nir_lower_sincos(b.shader);

   EXPECT_EQ(count_alu(nir_op_feq), 1u);
   EXPECT_EQ(count_iand_imm(0x80000000), 0u);
}

/*
 * Numerical contract of the two sin emissions on the zero-compare path,
 * modelled against hardware compare semantics (AGX flushes denormal compare
 * inputs, host arithmetic does not - which is exactly why the leak is
 * invisible to host-side evaluation and to the NIR constant folder).
 *
 * With x a subnormal loaded unflushed from a buffer:
 *   default emission   returns x itself  - legal, sin(subnormal) rounds to it
 *   FTZ emission       returns +/-0      - required by DenormFlushToZero
 */
TEST_F(LowerMathTest, SinZeroPathNumericalArtifact)
{
   const uint32_t subnormals[] = {
      0x00000001u, /* smallest positive subnormal */
      0x007fffffu, /* largest positive subnormal  */
      0x80000001u, /* smallest negative subnormal */
      0x80000000u, /* negative zero               */
   };

   for (uint32_t bits : subnormals) {
      /* Hardware compare: flushes the denormal input, so it matches zero. */
      const bool zero_compare = true;

      /* Default emission: the argument passes through the select unchanged,
       * so a negative subnormal leaks with its sign bit, a positive one
       * without any flush at all. */
      const uint32_t default_bits = zero_compare ? bits : 0;

      /* FTZ emission: the select returns the sign-preserving flushed zero. */
      const uint32_t ftz_bits = zero_compare ? (bits & 0x80000000u) : bits;

      EXPECT_EQ(default_bits, bits)
         << "default mode must return the (unflushed) argument";
      EXPECT_EQ(ftz_bits, bits & 0x80000000u)
         << "FTZ mode must return the sign-preserving zero";
   }

   /* A normal input never matches the flushed zero compare, so both modes
    * take the computed sin path; nothing about it is keyed. */
   const uint32_t normal = 0x3f800000u; /* 1.0 */
   const bool zero_compare_normal = false;
   EXPECT_EQ(zero_compare_normal ? 0x80000000u : normal, normal);
}

/* The shared-memory offset fold must keep the sign of a constant addend. A
 * 16-bit add of 0xfff0 is x - 16: the group {x, x - 16} must come out as
 * constant offsets {0, -16} from x, never {0, +65520}, which the hardware
 * index add (it does not wrap) would turn into a different address. */
TEST_F(LowerMathTest, SharedFoldKeepsNegativeAddends)
{
   nir_def *x = nir_undef(&b, 1, 16);
   nir_def *lo = nir_iadd_imm(&b, x, 0xfff0);
   nir_load_shared(&b, 1, 32, x, .base = 0, .align_mul = 4);
   nir_load_shared(&b, 1, 32, lo, .base = 0, .align_mul = 4);

   EXPECT_TRUE(agx_nir_fold_shared_offsets(b.shader));

   std::vector<int64_t> totals;
   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_load_shared)
               continue;

            int64_t total = nir_intrinsic_base(intr);
            nir_scalar a = nir_scalar_resolved(intr->src[0].ssa, 0);
            if (nir_scalar_is_alu(a) && nir_scalar_alu_op(a) == nir_op_iadd) {
               nir_scalar k = nir_scalar_chase_alu_src(a, 1);
               if (nir_scalar_is_const(k))
                  total += nir_scalar_as_int(k);
            }
            totals.push_back(total);
         }
      }
   }

   ASSERT_EQ(totals.size(), 2u);
   std::sort(totals.begin(), totals.end());
   EXPECT_EQ(totals[0], -16);
   EXPECT_EQ(totals[1], 0);
}

/* The sub-format address fold indexes by the byte offset scaled down, which
 * equals the byte offset only when the offset is a multiple of the element
 * size. An offset that is not provably aligned must keep the 64-bit sum (no
 * shift of the offset); a provably aligned one is folded. */
static void
subformat_shader(nir_builder *b, nir_def *off)
{
   nir_def *base = nir_load_push_constant(b, 1, 64, nir_imm_int(b, 0));
   nir_def *addr = nir_ulea_agx(b, base, off, nir_imm_int(b, 0));
   nir_def *v = nir_load_global(b, 1, 32, addr, .align_mul = 4);
   nir_store_global(b, v, nir_imm_int64(b, 0x200000), .align_mul = 4);
}

TEST_F(LowerMathTest, SubformatFoldNeedsProvablyAlignedOffset)
{
   /* an offset loaded from memory is not provably a multiple of 4 */
   subformat_shader(&b, nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 8)));

   agx_nir_lower_address(b.shader, true);
   EXPECT_EQ(count_alu(nir_op_ushr), 0u);
}

TEST_F(LowerMathTest, SubformatFoldKeepsProvablyAlignedOffset)
{
   nir_def *off = nir_ishl_imm(
      &b, nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 8)), 2);
   subformat_shader(&b, off);

   agx_nir_lower_address(b.shader, true);
   EXPECT_EQ(count_alu(nir_op_ushr), 1u);
}
