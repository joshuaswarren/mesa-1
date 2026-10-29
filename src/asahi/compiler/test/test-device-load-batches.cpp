/* Copyright 2026 Aurora Silicon; SPDX-License-Identifier: MIT */
#include <gtest/gtest.h>
#include <array>
extern "C" {
#include "agx3_slot_core.h"
}

static agx3_slot_instr
load(unsigned reg, bool join = true, bool prefer = false)
{
   agx3_slot_instr i{};
   i.is_async = true;
   i.nr_dsts = 1;
   i.dst[0] = reg;
   i.join_class = join ? 1 : 0;
   i.join_limit = join ? 8 : 0;
   i.prefer_join = prefer;
   return i;
}

TEST(DeviceLoadBatches, IndependentReadsShareQueues)
{
   std::array<agx3_slot_instr, 32> i;
   std::array<agx3_slot_result, 32> r;
   for (unsigned n = 0; n < i.size(); ++n) i[n] = load(n);
   ASSERT_TRUE(agx3_slot_assign(i.data(), i.size(), 8, r.data()));
   for (unsigned n = 0; n < i.size(); ++n) {
      EXPECT_EQ(r[n].wait_mask, 0);
      EXPECT_EQ(r[n].slot, n % 8 + 1);
   }
}

TEST(DeviceLoadBatches, RetiredAliasesDoNotConsumeNewGeneration)
{
   std::array<agx3_slot_instr, 37> i{};
   std::array<agx3_slot_result, 37> r;
   for (unsigned n = 0; n < 32; ++n) i[n] = load(n);
   i[32].nr_srcs = i[33].nr_srcs = i[35].nr_srcs = i[36].nr_srcs = 1;
   i[32].src[0] = 0; i[33].src[0] = 8;
   i[34] = load(40); i[34].preferred_slot = 1;
   i[35].src[0] = 16; i[36].src[0] = 40;
   ASSERT_TRUE(agx3_slot_assign(i.data(), i.size(), 8, r.data()));
   EXPECT_EQ(r[32].wait_mask, 1);
   EXPECT_EQ(r[33].wait_mask, 0);
   EXPECT_EQ(r[35].wait_mask, 0);
   EXPECT_EQ(r[36].wait_mask, 1);
}

TEST(DeviceLoadBatches, BoundAndMixedEngineRetirement)
{
   std::array<agx3_slot_instr, 9> i;
   std::array<agx3_slot_result, 9> r;
   for (unsigned n = 0; n < i.size(); ++n) i[n] = load(n);
   ASSERT_TRUE(agx3_slot_assign(i.data(), i.size(), 1, r.data()));
   for (unsigned n = 0; n < 8; ++n) EXPECT_EQ(r[n].wait_mask, 0);
   EXPECT_EQ(r[8].wait_mask, 1);
   i[2] = load(15, false); i[2].preferred_slot = 1;
   ASSERT_TRUE(agx3_slot_assign(i.data(), 3, 1, r.data()));
   EXPECT_EQ(r[2].wait_mask, 1);
   i[2] = {}; i[2].nr_dsts = 1; i[2].dst[0] = 0;
   ASSERT_TRUE(agx3_slot_assign(i.data(), 3, 1, r.data()));
   EXPECT_EQ(r[2].wait_mask, 1);
}

TEST(DeviceLoadBatches, GroupedPolicyAndLegacyPolicy)
{
   std::array<agx3_slot_instr, 32> i;
   std::array<agx3_slot_result, 32> r;
   for (unsigned n = 0; n < i.size(); ++n) i[n] = load(n, true, true);
   ASSERT_TRUE(agx3_slot_assign(i.data(), i.size(), 8, r.data()));
   for (unsigned n = 0; n < i.size(); ++n) {
      EXPECT_EQ(r[n].wait_mask, 0);
      EXPECT_EQ(r[n].slot, n / 8 + 1);
   }
   for (unsigned n = 0; n < i.size(); ++n) i[n] = load(n, false);
   ASSERT_TRUE(agx3_slot_assign(i.data(), 9, 8, r.data()));
   EXPECT_NE(r[8].wait_mask, 0);
}

TEST(DeviceLoadBatches, InvalidContractsFailClosed)
{
   agx3_slot_instr i = load(0);
   agx3_slot_result r;
   i.join_limit = 0; EXPECT_FALSE(agx3_slot_assign(&i, 1, 8, &r));
   i.join_limit = 9; EXPECT_FALSE(agx3_slot_assign(&i, 1, 8, &r));
   i.join_limit = 8; i.is_async = false;
   EXPECT_FALSE(agx3_slot_assign(&i, 1, 8, &r));
}
