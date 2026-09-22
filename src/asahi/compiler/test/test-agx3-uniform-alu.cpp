/* SPDX-License-Identifier: MIT */
#include <array>
#include <cstring>
#include <set>
#include <string>
#include <gtest/gtest.h>
#include "agx3_binder.h"
#include "agx3_uniform_alu.h"
#include "agx_test.h"

TEST(AGX3UniformALU, AppleArithmeticWitnesses)
{
   /* Fresh applegpu_g17p output, also byte-identical on applegpu_g16g. */
   struct {
      agx3_uniform_float_op op;
      std::array<uint8_t, 8> bytes;
      unsigned length;
   } cases[] = {
      {AGX3_UNIFORM_FADD, {0x09, 0x01, 0x0c, 0x0d, 0x00, 0xc2}, 6},
      {AGX3_UNIFORM_FMUL, {0x09, 0x01, 0x0d, 0x0d, 0x00, 0xc2}, 6},
      {AGX3_UNIFORM_FMA, {0x09, 0x01, 0x0e, 0x0d, 0x21, 0x0e, 0x02, 0xc2}, 8},
   };
   agx3_uniform_float_src src[3] = {};
   src[0].kind = AGX3_FLOAT_REGISTER;
   src[0].discard = true;
   src[1].kind = src[2].kind = AGX3_FLOAT_UNIFORM;
   src[1].value = 12;
   src[2].value = 14;
   for (const auto &c : cases) {
      uint8_t bytes[16], length = 0;
      ASSERT_TRUE(agx3_uniform_float_encode(AGX_ISA_ENCODING_AGX3B, c.op, 0,
                                            src, false, false, 32, bytes,
                                            &length));
      ASSERT_EQ(length, c.length);
      EXPECT_EQ(memcmp(bytes, c.bytes.data(), length), 0);
   }
}

TEST(AGX3UniformALU, AppleImmediateUniformFma)
{
   agx3_uniform_float_src src[3] = {};
   src[0].kind = AGX3_FLOAT_REGISTER;
   src[0].discard = true;
   src[1].kind = AGX3_FLOAT_IMMEDIATE;
   src[1].value = 0x20;
   src[2].kind = AGX3_FLOAT_UNIFORM;
   src[2].value = 12;
   const uint8_t expected[] = {0x09, 0x01, 0x0e, 0xa1, 0x21, 0x0c, 0x02, 0xc2};
   uint8_t bytes[16], length = 0;
   ASSERT_TRUE(agx3_uniform_float_encode(AGX_ISA_ENCODING_AGX3B,
                                         AGX3_UNIFORM_FMA, 0, src, false, false,
                                         32, bytes, &length));
   ASSERT_EQ(length, sizeof(expected));
   EXPECT_EQ(memcmp(bytes, expected, length), 0);
}

TEST(AGX3UniformALU, AppleHalfArithmeticWitnesses)
{
   struct {
      agx3_uniform_float_op op;
      std::array<uint8_t, 8> bytes;
      unsigned length;
   } cases[] = {
      {AGX3_UNIFORM_FADD, {0x00, 0x00, 0x0c, 0x0c, 0x00, 0xc2}, 6},
      {AGX3_UNIFORM_FMUL, {0x00, 0x00, 0x0d, 0x0c, 0x00, 0xc2}, 6},
      {AGX3_UNIFORM_FMA, {0x00, 0x00, 0x0e, 0x0c, 0x21, 0x0e, 0x00, 0xc2}, 8},
   };
   agx3_uniform_float_src src[3] = {};
   src[0].kind = AGX3_FLOAT_REGISTER;
   src[0].discard = true;
   src[1].kind = src[2].kind = AGX3_FLOAT_UNIFORM;
   src[1].value = 12;
   src[2].value = 14;
   for (const auto &c : cases) {
      uint8_t bytes[16], length = 0;
      ASSERT_TRUE(agx3_uniform_half_encode(AGX_ISA_ENCODING_AGX3B, c.op, 0, src,
                                           false, false, 32, bytes, &length));
      ASSERT_EQ(length, c.length);
      EXPECT_EQ(memcmp(bytes, c.bytes.data(), length), 0);
   }
}

TEST(AGX3UniformALU, HalfParityAndDistinctDependencySets)
{
   for (unsigned value : {0u, 1u, 12u, 13u, 255u, 256u, 511u}) {
      for (unsigned reg : {0u, 1u, 126u, 127u, 254u, 255u}) {
         for (auto op :
              {AGX3_UNIFORM_FADD, AGX3_UNIFORM_FMUL, AGX3_UNIFORM_FMA}) {
            std::set<std::string> encodings;
            for (unsigned wait = 0; wait < 64; ++wait) {
               agx3_uniform_float_src src[3] = {};
               src[0].kind = AGX3_FLOAT_REGISTER;
               src[0].value = reg;
               src[1].kind = src[2].kind = AGX3_FLOAT_UNIFORM;
               src[1].value = src[2].value = value;
               uint8_t bytes[16], length;
               ASSERT_TRUE(agx3_uniform_half_encode(AGX_ISA_ENCODING_AGX3B, op,
                                                    reg ^ 1, src, false, false,
                                                    wait, bytes, &length))
                  << "op=" << op << " uniform=" << value << " reg=" << reg
                  << " wait=" << wait;
               EXPECT_TRUE(
                  encodings.insert(std::string((char *)bytes, length)).second)
                  << "dependency mask aliased at " << wait;
            }
         }
      }
   }
}

TEST(AGX3UniformALU, AppleUniformClampWitnesses)
{
   struct {
      bool half, maximum, dest_cache;
      uint16_t dest, reg, uniform;
      uint8_t wait;
      std::array<uint8_t, 6> bytes;
   } cases[] = {
      {false, true, true, 0, 0, 12, 32, {0x02, 0x01, 0x2e, 0x0d, 0x00, 0xc2}},
      {false, false, false, 0, 0, 14, 0, {0x02, 0x01, 0x0e, 0x0f, 0x01, 0x02}},
      {true, true, true, 0, 0, 12, 32, {0x02, 0x00, 0x2c, 0x0c, 0x00, 0xc2}},
      {true, true, true, 1, 1, 12, 0, {0x0a, 0x00, 0x2c, 0x0c, 0x08, 0x02}},
      {true, false, false, 0, 0, 13, 0, {0x02, 0x00, 0x0c, 0x0c, 0x11, 0x02}},
   };
   for (const auto &c : cases) {
      uint8_t bytes[16], length;
      ASSERT_TRUE(agx3_uniform_minmax_encode(
         AGX_ISA_ENCODING_AGX3B, c.dest, c.reg, c.uniform, c.half, c.maximum,
         false, true, c.dest_cache, c.wait, bytes, &length));
      ASSERT_EQ(length, c.bytes.size());
      EXPECT_EQ(memcmp(bytes, c.bytes.data(), length), 0);
   }
}

TEST(AGX3UniformALU, MinmaxBoundariesAndNoWaitAliasing)
{
   for (bool half : {false, true}) {
      for (unsigned uniform : {0u, 12u, 254u, 256u, 510u}) {
         for (unsigned reg : {0u, 126u, 254u}) {
            for (bool maximum : {false, true}) {
               std::set<std::string> encodings;
               for (unsigned wait = 0; wait < 64; ++wait) {
                  uint8_t bytes[16], length;
                  bool ok = agx3_uniform_minmax_encode(
                     AGX_ISA_ENCODING_AGX3B, reg, reg, uniform, half, maximum,
                     false, false, false, wait, bytes, &length);
                  ASSERT_EQ(ok, (wait & (wait - 1)) == 0)
                     << "half=" << half << " uniform=" << uniform
                     << " reg=" << reg;
                  if (ok)
                     EXPECT_TRUE(
                        encodings.insert(std::string((char *)bytes, length))
                           .second);
               }
            }
         }
      }
   }
}

TEST(AGX3UniformALU, UniformBoundariesAndDependencySets)
{
   for (unsigned value : {0u, 12u, 254u, 256u, 510u}) {
      for (unsigned wait = 0; wait < 64; ++wait) {
         agx3_uniform_float_src src[3] = {};
         src[0].kind = AGX3_FLOAT_REGISTER;
         src[0].value = 2;
         src[1].kind = src[2].kind = AGX3_FLOAT_UNIFORM;
         src[1].value = src[2].value = value;
         for (auto op :
              {AGX3_UNIFORM_FADD, AGX3_UNIFORM_FMUL, AGX3_UNIFORM_FMA}) {
            uint8_t bytes[16], length;
            ASSERT_TRUE(agx3_uniform_float_encode(AGX_ISA_ENCODING_AGX3B, op, 0,
                                                  src, false, true, wait, bytes,
                                                  &length))
               << "op=" << op << " uniform=" << value << " wait=" << wait;
            EXPECT_LE(length, 16);
         }
      }
   }
}

TEST(AGX3UniformALU, RefusesInvalidInputsWithoutWriting)
{
   agx3_uniform_float_src src[3] = {};
   src[0].kind = AGX3_FLOAT_REGISTER;
   src[1].kind = AGX3_FLOAT_UNIFORM;
   for (unsigned value : {1u, 255u, 511u, 512u, 65534u}) {
      src[1].value = value;
      uint8_t bytes[16];
      memset(bytes, 0xa5, sizeof(bytes));
      uint8_t length = 99;
      EXPECT_FALSE(agx3_uniform_float_encode(AGX_ISA_ENCODING_AGX3B,
                                             AGX3_UNIFORM_FADD, 0, src, false,
                                             false, 0, bytes, &length));
      EXPECT_EQ(length, 99);
      for (auto byte : bytes)
         EXPECT_EQ(byte, 0xa5);
   }
   src[1].value = 12;
   src[1].discard = true;
   uint8_t bytes[16], length;
   EXPECT_FALSE(agx3_uniform_float_encode(AGX_ISA_ENCODING_AGX3B,
                                          AGX3_UNIFORM_FADD, 0, src, false,
                                          false, 0, bytes, &length));
   src[1].discard = false;
   EXPECT_FALSE(agx3_uniform_float_encode(AGX_ISA_ENCODING_AGX3C,
                                          AGX3_UNIFORM_FADD, 0, src, false,
                                          false, 0, bytes, &length));
}

TEST(AGX3UniformALU, LoweringKeepsF32AndCopiesMixedWidths)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = agx_test_builder(mem);
   b->shader->key = &key;
   agx_instr *same =
      agx_fadd_to(b, agx_temp(b->shader, AGX_SIZE_32),
                  agx_register(0, AGX_SIZE_32), agx_uniform(14, AGX_SIZE_32));
   agx_instr *mixed = agx_fadd_to(b, agx_temp(b->shader, AGX_SIZE_16),
                                  agx_uniform(12, AGX_SIZE_32), agx_negzero());
   agx_lower_uniform_sources(b->shader);
   EXPECT_EQ(same->src[0].type, AGX_INDEX_REGISTER);
   EXPECT_EQ(same->src[1].type, AGX_INDEX_UNIFORM);
   EXPECT_NE(mixed->src[0].type, AGX_INDEX_UNIFORM);
   ralloc_free(mem);
}
