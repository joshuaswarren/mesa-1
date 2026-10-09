/*
 * SPDX-License-Identifier: MIT
 */

#include "agx_va_layout.h"

#include <gtest/gtest.h>

static constexpr uint64_t GiB = 1ull << 30;

/* The read-write heap, and its read-only shadow, must be disjoint, below the
 * kernel window, and told apart by the single bit ro_offset.
 */
static void
check(uint64_t user_start, uint64_t kernel_start, uint64_t want_size)
{
   uint64_t ro = 0, size = 0;

   ASSERT_TRUE(agx_sparse_layout(user_start, kernel_start, &ro, &size));
   EXPECT_EQ(size, want_size);

   EXPECT_EQ(ro & (ro - 1), 0u);
   EXPECT_EQ(user_start & ro, 0u);
   EXPECT_EQ((user_start + size - 1) & ro, 0u);
   EXPECT_LE(user_start + ro + size, kernel_start);
}

TEST(SparseLayout, Window512GiBHeapAt72GiB)
{
   /* G13 and G14 as reported by the kernel: 408 GiB of user window; the old
    * code left 64 GiB of it
    */
   check(72 * GiB, 480 * GiB, 152 * GiB);
}

TEST(SparseLayout, Window512GiBHeapAt128GiB)
{
   check(128 * GiB, 480 * GiB, 96 * GiB);
}

TEST(SparseLayout, SmallWindow)
{
   check(8 * GiB, 120 * GiB, 48 * GiB);
}

TEST(SparseLayout, NoRoom)
{
   uint64_t ro = 0, size = 0;
   EXPECT_FALSE(agx_sparse_layout(72 * GiB, 72 * GiB, &ro, &size));
}
