/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include <vector>
#include <gtest/gtest.h>
#include "agx3_alu_core.h"
#include "agx3_pack.h"
#include "agx_test.h"

static unsigned
reverse(unsigned c)
{
   switch (c) {
   case AGX_FCOND_LT:
      return AGX_FCOND_GT;
   case AGX_FCOND_GT:
      return AGX_FCOND_LT;
   case AGX_FCOND_LE:
      return AGX_FCOND_GE;
   case AGX_FCOND_GE:
      return AGX_FCOND_LE;
   default:
      return c;
   }
}

static std::vector<uint8_t>
pack(bool compare, unsigned imm, bool neg, bool swap, bool abs, unsigned hint,
     bool narrow, unsigned condition = AGX_FCOND_GE, bool invert = false,
     bool loaded = false, unsigned invalid = 0)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   agx_builder *b = agx_test_builder(mem);
   b->shader->key = &key;
   agx_index r = agx_register(6, AGX_SIZE_32);
   if (loaded)
      agx_device_load_to(b, r, agx_register(16, AGX_SIZE_64),
                         agx_register(20, AGX_SIZE_32), AGX_FORMAT_I32, 1, 0,
                         false);
   r.neg = neg;
   r.abs = abs;
   r.cache = hint == 1;
   r.discard = hint == 2;
   agx_index d = agx_register(6, narrow ? AGX_SIZE_16 : AGX_SIZE_32);
   d.cache = true;
   agx_index k = agx_immediate(imm);
   agx_instr *I;
   if (compare) {
      agx_index t = agx_register(22, AGX_SIZE_32),
                f = agx_register(8, AGX_SIZE_32);
      t.discard = true;
      f.cache = true;
      I = agx_fcmpsel_to(b, d, swap ? k : r, swap ? r : k, t, f,
                         (enum agx_fcond)condition);
      I->invert_cond = invert;
   } else
      I = agx_fmul_to(b, d, swap ? k : r, swap ? r : k);
   if (invalid == 1)
      I->src[swap ? 1 : 0].kill = true;
   if (invalid == 2)
      I->src[swap ? 0 : 1].neg = true;
   if (invalid == 3 && compare)
      I->src[2].neg = true;
   if (invalid == 4)
      I->saturate = true;
   if (invalid == 5)
      I->src[swap ? 1 : 0].cache = I->src[swap ? 1 : 0].discard = true;
   agx_stop(b);
   struct util_dynarray out;
   util_dynarray_init(&out, mem);
   bool ok = agx3_pack_binary(b->shader, &out);
   std::vector<uint8_t> bytes;
   if (ok)
      bytes.assign((uint8_t *)out.data, (uint8_t *)out.data + out.size);
   ralloc_free(mem);
   return bytes;
}

TEST(NegatedMinifloat, MultiplyMatchesSignedImmediateBytes)
{
   for (unsigned k = 0; k < 256; k++)
      for (bool swap : {false, true})
         for (unsigned h = 0; h < 3; h++)
            for (unsigned form = 0; form < 3; form++) {
               bool abs = form == 2, narrow = form == 1;
               SCOPED_TRACE(k);
               SCOPED_TRACE(swap);
               SCOPED_TRACE(h);
               SCOPED_TRACE(form);
               auto expected =
                  pack(false, k ^ 128, false, swap, abs, h, narrow);
               ASSERT_FALSE(expected.empty());
               auto actual = pack(false, k, true, swap, abs, h, narrow);
               ASSERT_FALSE(actual.empty());
               EXPECT_EQ(actual, expected);
            }
}
TEST(NegatedMinifloat, OrderedSelectMatchesSignedImmediateBytes)
{
   for (unsigned k :
        {0u, 1u, 0x30u, 0x40u, 0x7fu, 0x80u, 0x81u, 0xb0u, 0xc0u, 0xffu})
      for (unsigned c : {AGX_FCOND_EQ, AGX_FCOND_LT, AGX_FCOND_GT, AGX_FCOND_GE,
                         AGX_FCOND_LE})
         for (bool swap : {false, true})
            for (bool abs : {false, true})
               for (bool invert : {false, true})
                  for (unsigned h = 0; h < 3; h++) {
                     SCOPED_TRACE(k);
                     SCOPED_TRACE(c);
                     SCOPED_TRACE(swap);
                     SCOPED_TRACE(abs);
                     uint8_t encoded[AGX3_ALU_MAX_LENGTH], length = 0;
                     const enum agx3_alu_hint hints[] = {
                        h == 1   ? AGX3_ALU_HINT_CACHE
                        : h == 2 ? AGX3_ALU_HINT_DISCARD
                                 : AGX3_ALU_HINT_NONE,
                        AGX3_ALU_HINT_NONE,
                        invert ? AGX3_ALU_HINT_CACHE : AGX3_ALU_HINT_DISCARD,
                        invert ? AGX3_ALU_HINT_DISCARD : AGX3_ALU_HINT_CACHE};
                     ASSERT_TRUE(agx3_alu_fcmp_select_f32_imm(
                        AGX_ISA_ENCODING_AGX3B, 6, 6, k ^ 128, abs,
                        invert ? 8 : 22, invert ? 22 : 8, hints,
                        swap ? c : reverse(c), true, 0, encoded, &length));
                     std::vector<uint8_t> expected(encoded, encoded + length);
                     expected.insert(expected.end(), {0x0e, 0, 0, 0});
                     auto actual =
                        pack(true, k, true, swap, abs, h, false, c, invert);
                     ASSERT_FALSE(actual.empty());
                     EXPECT_EQ(actual, expected);
                  }
}
TEST(NegatedMinifloat, PreservesLoadWaitAndRejectsOtherModifiers)
{
   for (bool compare : {false, true})
      for (bool abs : {false, true})
         for (bool swap : {false, true}) {
            auto expected =
               pack(compare, 0xc0, false, swap, abs, 2, false,
                    compare ? AGX_FCOND_LE : AGX_FCOND_GE, false, true);
            ASSERT_FALSE(expected.empty());
            auto actual = pack(compare, 0x40, true, swap, abs, 2, false,
                               AGX_FCOND_GE, false, true);
            ASSERT_FALSE(actual.empty());
            EXPECT_EQ(actual, expected);
         }
   for (bool compare : {false, true})
      for (unsigned mode : {1u, 2u, 4u, 5u})
         EXPECT_TRUE(pack(compare, 0x40, true, false, false, 0, false,
                          AGX_FCOND_GE, false, false, mode)
                        .empty())
            << compare << mode;
   EXPECT_TRUE(pack(true, 0x40, true, false, false, 0, false, AGX_FCOND_GE,
                    false, false, 3)
                  .empty());
   for (unsigned c : {AGX_FCOND_LTN, AGX_FCOND_GTN})
      EXPECT_FALSE(pack(true, 0x40, true, false, false, 0, false, c).empty())
         << c;
}

TEST(NegatedMinifloat, KeepsExistingUnorderedPaths)
{
   for (unsigned c : {AGX_FCOND_LTN, AGX_FCOND_GTN}) {
      auto bytes = pack(true, 0x40, true, false, false, 0, false, c);
      ASSERT_FALSE(bytes.empty());
      printf("UNCHANGED_UNORDERED cond=%u hex=", c);
      for (uint8_t x : bytes)
         printf("%02x", x);
      printf("\n");
   }
}
