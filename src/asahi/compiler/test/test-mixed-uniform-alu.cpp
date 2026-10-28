/* SPDX-License-Identifier: MIT */
#include <array>
#include <cstring>
#include <gtest/gtest.h>
#include "agx3_uniform_alu.h"
#include "agx_test.h"
extern "C" {
#include "asahi/layout/layout.h"
#include "agx_nir.h"
}

TEST(MixedUniformALU, MetalWordArithmeticHalfUniformVectors)
{
   struct vector {
      agx3_uniform_float_op op;
      unsigned half_slot;
      std::array<uint8_t, 8> bytes;
      unsigned length;
   };
   const vector vectors[] = {
      {AGX3_UNIFORM_FADD, 1, {0x09, 0x01, 0x0c, 0x0e, 0x00, 0xc2}, 6},
      {AGX3_UNIFORM_FMUL, 1, {0x09, 0x01, 0x0d, 0x0e, 0x00, 0xc2}, 6},
      {AGX3_UNIFORM_FMA, 0, {0x09, 0x0e, 0x16, 0x01, 0x21, 0x0c, 0x82, 0xc0}, 8},
      {AGX3_UNIFORM_FMA, 2, {0x09, 0x01, 0x0e, 0x0d, 0x21, 0x0e, 0x00, 0xc2}, 8},
   };
   for (const auto &v : vectors) {
      agx3_uniform_float_src src[3] = {};
      src[0].kind = AGX3_FLOAT_REGISTER;
      src[0].discard = true;
      src[1].kind = src[2].kind = AGX3_FLOAT_UNIFORM;
      src[1].value = src[2].value = 12;
      if (v.half_slot == 0) {
         src[0].kind = AGX3_FLOAT_UNIFORM;
         src[0].discard = false;
         src[1].kind = AGX3_FLOAT_REGISTER;
         src[1].value = 0;
         src[1].discard = true;
      }
      src[v.half_slot].value = 14;
      src[v.half_slot].half = true;
      uint8_t bytes[16], length = 0;
      ASSERT_TRUE(agx3_uniform_float_encode(AGX_ISA_ENCODING_AGX3B, v.op, 0,
                                            src, false, false, 32, bytes,
                                            &length));
      ASSERT_EQ(length, v.length);
      EXPECT_EQ(memcmp(bytes, v.bytes.data(), length), 0);
   }
}

TEST(MixedUniformALU, LoweringKeepsWitnessedHalfSources)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   b->shader->key = &key;
   for (unsigned value : {0u, 1u, 14u, 255u, 511u}) {
      agx_instr *I = agx_fmul_to(b, agx_temp(b->shader, AGX_SIZE_32),
                                 agx_register(0, AGX_SIZE_32),
                                 agx_uniform(value, AGX_SIZE_16));
      agx_lower_uniform_sources(b->shader);
      EXPECT_EQ(I->src[1].type, AGX_INDEX_UNIFORM);
      EXPECT_EQ(I->src[1].size, AGX_SIZE_16);
      EXPECT_EQ(I->src[1].value, value);
   }
   ralloc_free(mem);
}

TEST(MixedUniformALU, RetainsSaturatedAndAllUniformFallbacks)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   b->shader->key = &key;
   agx_instr *sat =
      agx_fmul_to(b, agx_temp(b->shader, AGX_SIZE_32),
                  agx_register(0, AGX_SIZE_32), agx_uniform(14, AGX_SIZE_16));
   sat->saturate = true;
   agx_instr *uniform =
      agx_fmul_to(b, agx_temp(b->shader, AGX_SIZE_32),
                  agx_uniform(12, AGX_SIZE_32), agx_uniform(14, AGX_SIZE_16));
   agx_lower_uniform_sources(b->shader);
   EXPECT_NE(sat->src[1].type, AGX_INDEX_UNIFORM);
   EXPECT_NE(uniform->src[0].type, AGX_INDEX_UNIFORM);
   EXPECT_NE(uniform->src[1].type, AGX_INDEX_UNIFORM);
   ralloc_free(mem);
}

TEST(MixedUniformALU, PreservesEveryDependencySet)
{
   for (auto op : {AGX3_UNIFORM_FADD, AGX3_UNIFORM_FMUL, AGX3_UNIFORM_FMA}) {
      for (unsigned value : {0u, 1u, 14u, 255u, 511u}) {
         std::array<std::array<uint8_t, 16>, 64> seen = {};
         std::array<uint8_t, 64> lengths = {};
         for (unsigned wait = 0; wait < 64; ++wait) {
            agx3_uniform_float_src src[3] = {};
            src[0].kind = AGX3_FLOAT_REGISTER;
            src[1].kind = src[2].kind = AGX3_FLOAT_UNIFORM;
            src[1].value = value;
            src[1].half = true;
            src[2].value = 12;
            ASSERT_TRUE(agx3_uniform_float_encode(
               AGX_ISA_ENCODING_AGX3B, op, 0, src, false, false, wait,
               seen[wait].data(), &lengths[wait]));
            for (unsigned previous = 0; previous < wait; ++previous)
               EXPECT_TRUE(lengths[previous] != lengths[wait] ||
                           memcmp(seen[previous].data(), seen[wait].data(),
                                  lengths[wait]) != 0);
         }
      }
   }
}

TEST(MixedUniformALU, OptimizerFoldsUniformWideningOnlyIntoNativeShapes)
{
   for (bool uniform_other : {false, true}) {
      for (bool saturate : {false, true}) {
         void *mem = ralloc_context(NULL);
         agx_builder *b = agx_test_builder(mem);
         agx_shader_key key = {};
         key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
         b->shader->key = &key;
         agx_index wide = agx_temp(b->shader, AGX_SIZE_32);
         agx_fmov_to(b, wide, agx_uniform(14, AGX_SIZE_16));
         agx_index other = uniform_other ? agx_uniform(12, AGX_SIZE_32)
                                         : agx_register(0, AGX_SIZE_32);
         agx_instr *I =
            agx_fmul_to(b, agx_temp(b->shader, AGX_SIZE_32), other, wide);
         I->saturate = saturate;
         agx_optimizer_forward(b->shader);
         EXPECT_EQ(I->src[1].type == AGX_INDEX_UNIFORM,
                   !uniform_other && !saturate);
         if (!uniform_other && !saturate) {
            EXPECT_EQ(I->src[1].size, AGX_SIZE_16);
         }
         ralloc_free(mem);
      }
   }
}

TEST(MixedUniformALU, FullCompilerKeepsHalfUniformInFloatArithmetic)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "mixed-width uniform multiply");
   nir_def *half = agx_test_load_preamble(&b, 1, 16, 14);
   nir_def *x = nir_channel(&b, nir_load_frag_coord(&b), 0);
   nir_def *value = nir_fmul(&b, x, nir_f2f32(&b, half));
   agx_test_store_local_pixel_agx(&b, nir_vec4(&b, value, value, value, value),
                                  nir_imm_intN_t(&b, 0xff, 16),
                                  nir_undef(&b, 2, 16), 0, 0xf,
                                  (enum pipe_format)AIL_ISA_FORMAT_I32);
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   key.no_preamble = true;
   key.reserved_preamble = 16;
   key.fs.nr_samples = 1;
   agx_preprocess_nir(b.shader);
   agx_shader_part part = {};
   ASSERT_TRUE(agx_compile_shader_nir(b.shader, &key, &part));
   bool found = false;
   for (unsigned dest = 0; dest < 32; dest += 2) {
      for (unsigned reg = 0; reg < 32; reg += 2) {
         for (unsigned hint = 0; hint < 3; ++hint) {
            agx3_uniform_float_src src[3] = {};
            src[0].kind = AGX3_FLOAT_REGISTER;
            src[0].value = reg;
            src[0].cache = hint == 1;
            src[0].discard = hint == 2;
            src[1].kind = AGX3_FLOAT_UNIFORM;
            src[1].value = 14;
            src[1].half = true;
            for (bool cache : {false, true}) {
               uint8_t bytes[16], length;
               if (!agx3_uniform_float_encode(key.dev.encoding,
                                              AGX3_UNIFORM_FMUL, dest, src,
                                              false, cache, 0, bytes, &length))
                  continue;
               for (unsigned off = 0; off + length <= part.info.main_size;
                    off += 2)
                  found |= memcmp((const uint8_t *)part.binary +
                                     part.info.main_offset + off,
                                  bytes, length) == 0;
            }
         }
      }
   }
   EXPECT_TRUE(found);
   free(part.binary);
   ralloc_free(b.shader);
}

TEST(MixedUniformALU, AGX2ControlFlowComparesKeepUniforms)
{
   /* AGX2 control-flow compares read a uniform in either operand, so a loop
    * bound or branch condition held in the uniform file is used in place.
    * AGX3 keeps one in place only beside a 32-bit register first operand or
    * an 8-bit immediate, and copies it otherwise. */
   for (auto encoding : {AGX_ISA_ENCODING_AGX2, AGX_ISA_ENCODING_AGX3B}) {
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.encoding = encoding;
      b->shader->key = &key;
      agx_block *target = agx_test_block(b->shader);
      agx_instr *loop = agx_while_icmp(b, agx_temp(b->shader, AGX_SIZE_32),
                                       agx_uniform(14, AGX_SIZE_32), 1,
                                       AGX_ICOND_ULT, false, target);
      agx_instr *imm = agx_if_icmp(b, agx_immediate(300),
                                   agx_uniform(20, AGX_SIZE_32), 1,
                                   AGX_ICOND_UEQ, true, target);
      agx_instr *half = agx_break_if_icmp(b, agx_temp(b->shader, AGX_SIZE_16),
                                          agx_uniform(31, AGX_SIZE_16), 1,
                                          true, AGX_ICOND_UEQ, target);
      agx_lower_uniform_sources(b->shader);

      const bool agx2 = encoding == AGX_ISA_ENCODING_AGX2;
      EXPECT_EQ(loop->src[1].type, AGX_INDEX_UNIFORM);
      EXPECT_EQ(imm->src[1].type == AGX_INDEX_UNIFORM, agx2);
      EXPECT_EQ(half->src[1].type == AGX_INDEX_UNIFORM, agx2);
      ralloc_free(mem);
   }
}
