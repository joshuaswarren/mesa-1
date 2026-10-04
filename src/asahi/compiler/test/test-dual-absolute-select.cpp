/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include <cstdio>
#include <vector>
#include <gtest/gtest.h>
#include "agx3_alu_core.h"
#include "agx3_pack.h"
#include "agx_test.h"
static std::vector<uint8_t>
reference(unsigned dest, const uint16_t s[4], const bool ab[2],
          const bool neg[2], const bool imm[2], const agx3_alu_hint h[4],
          unsigned cond, bool half, bool cache, unsigned wait)
{
   agx3_fsel_desc d = {};
   d.dest = dest;
   d.dest_16 = half;
   d.dest_cache = cache;
   d.cmp0 = s[0];
   d.cmp1 = s[1];
   d.cmp0_abs = ab[0];
   d.cmp1_abs = ab[1];
   d.cmp0_neg = neg[0];
   d.cmp1_neg = neg[1];
   d.cmp0_hint = h[0];
   d.cmp1_hint = h[1];
   d.sel0 = s[2];
   d.sel1 = s[3];
   d.sel_16 = half;
   d.sel0_kind = imm[0] ? AGX3_FSEL_SEL_IMMEDIATE : AGX3_FSEL_SEL_REGISTER;
   d.sel1_kind = imm[1] ? AGX3_FSEL_SEL_IMMEDIATE : AGX3_FSEL_SEL_REGISTER;
   d.sel0_hint = h[2];
   d.sel1_hint = h[3];
   d.condition = cond;
   d.wait_mask = wait;
   uint8_t b[AGX3_ALU_MAX_LENGTH], n = 0;
   if (!agx3_alu_fcmp_select(AGX_ISA_ENCODING_AGX3B, &d, b, &n))
      return {};
   return {b, b + n};
}
static std::vector<uint8_t>
legacy(unsigned dest, const uint16_t s[4], const bool ab[2], const bool neg[2],
       const bool imm[2], const agx3_alu_hint h[4], unsigned cond, bool half,
       bool cache, unsigned wait)
{
   uint8_t b[AGX3_ALU_MAX_LENGTH], n = 0;
   if (!agx3_alu_fcmp_select_reg_imm(AGX_ISA_ENCODING_AGX3B, dest, s, false, ab,
                                     neg, imm, h, cond, half, cache, wait, b,
                                     &n))
      return {};
   return {b, b + n};
}
TEST(DualAbsoluteSelect, CapturedHardwareOpcode)
{
   const unsigned values[][3] = {{4, 50, 54}, {4, 56, 60}, {22, 10, 4}};
   for (auto &v : values) {
      void *mem = ralloc_context(NULL);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      agx_builder *b = agx_test_builder(mem);
      b->shader->key = &key;
      agx_index a = agx_register(v[1], AGX_SIZE_32),
                c = agx_register(v[2], AGX_SIZE_32);
      a.abs = c.abs = true;
      a.cache = c.cache = true;
      agx_index selected = agx_register(v[0], AGX_SIZE_16);
      selected.discard = true;
      agx_fcmpsel_to(b, agx_register(v[0], AGX_SIZE_16), a, c, selected,
                     agx_immediate(0), AGX_FCOND_GE);
      agx_stop(b);
      util_dynarray binary;
      util_dynarray_init(&binary, mem);
      const bool ok = agx3_pack_binary(b->shader, &binary);
      EXPECT_TRUE(ok);
      const uint16_t s[] = {uint16_t(v[1]), uint16_t(v[2]), uint16_t(v[0]), 0};
      const bool ab[] = {true, true}, neg[] = {false, false},
                 imm[] = {false, true};
      const agx3_alu_hint h[] = {AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_CACHE,
                                 AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_NONE};
      auto expected =
         reference(v[0], s, ab, neg, imm, h, AGX_FCOND_GE, true, false, 0);
      ASSERT_EQ(expected.size(), 14u);
      agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
      expected.insert(expected.end(), end.bytes, end.bytes + end.length);
      if (ok)
         EXPECT_EQ(std::vector<uint8_t>((uint8_t *)binary.data,
                                        (uint8_t *)binary.data + binary.size),
                   expected);
      ralloc_free(mem);
   }
}
TEST(DualAbsoluteSelect, WidthsHintsAndRelations)
{
   for (unsigned cond : {5u, 6u})
      for (bool half : {false, true})
         for (unsigned shape = 0; shape < 3; shape++)
            for (unsigned flags = 0; flags < 4; flags++)
               for (unsigned hint = 0; hint < 3; hint++)
                  for (unsigned wait : {0u, 32u}) {
                     const uint16_t s[] = {50, 54,
                                           uint16_t(shape == 2 ? 7
                                                    : half     ? 5
                                                               : 4),
                                           uint16_t(shape == 1 ? 0
                                                    : half     ? 9
                                                               : 8)};
                     const bool ab[] = {true, true},
                                neg[] = {bool(flags & 1), bool(flags & 2)};
                     const bool imm[] = {shape == 2, shape == 1};
                     const agx3_alu_hint h[] = {
                        agx3_alu_hint(hint), agx3_alu_hint(hint),
                        imm[0] ? AGX3_ALU_HINT_NONE : agx3_alu_hint(hint),
                        imm[1] ? AGX3_ALU_HINT_NONE : agx3_alu_hint(hint)};
                     auto want = reference(half ? 3 : 2, s, ab, neg, imm, h,
                                           cond, half, true, wait);
                     ASSERT_EQ(want.size(), 14u);
                     EXPECT_EQ(legacy(half ? 3 : 2, s, ab, neg, imm, h, cond,
                                      half, true, wait),
                               want);
                  }
}
TEST(DualAbsoluteSelect, KeepsInvalidInputsRejected)
{
   const bool ab[] = {true, true}, neg[] = {false, false},
              imm[] = {false, true};
   for (unsigned mode = 0; mode < 5; mode++) {
      uint16_t s[] = {50, 54, 4, 0};
      agx3_alu_hint h[] = {AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_CACHE,
                           AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_NONE};
      unsigned dest = 4, wait = 0;
      if (mode == 0)
         s[0] = 51;
      if (mode == 1)
         s[1] = 256;
      if (mode == 2)
         s[3] = 256;
      if (mode == 3)
         h[3] = AGX3_ALU_HINT_CACHE;
      if (mode == 4)
         wait = 2;
      EXPECT_TRUE(
         legacy(dest, s, ab, neg, imm, h, 5, true, false, wait).empty());
   }
}
