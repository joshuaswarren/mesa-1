/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_memory_budget_core.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#define MiB(x) (UINT64_C(x) << 20)
#define GiB(x) (UINT64_C(x) << 30)

static void
check_invariants(uint64_t heap, uint64_t used, uint64_t available)
{
   uint64_t budget = hk_memory_budget_B(heap, used, available);
   assert(budget <= heap);
   if (heap) {
      assert(budget > 0);
      assert(budget >= (used < heap ? used : heap));
   } else {
      assert(budget == 0);
   }
}

int
main(void)
{
   assert(hk_memory_budget_B(GiB(8), GiB(2), GiB(4)) ==
          GiB(2) + MiB(3686));
   assert(hk_memory_budget_B(GiB(8), GiB(2), UINT64_MAX) == GiB(8));
   assert(hk_memory_budget_B(GiB(8), GiB(7), GiB(4)) == GiB(8));
   assert(hk_memory_budget_B(GiB(8), 0, 0) == MiB(1));
   assert(hk_memory_budget_B(4096, 0, 0) == 4096);
   assert(hk_memory_budget_B(GiB(8), GiB(9), GiB(1)) == GiB(8));

   const uint64_t heaps[] = {0, 4096, MiB(1), GiB(8), UINT64_MAX};
   const uint64_t usages[] = {0, 4096, MiB(3), GiB(7), UINT64_MAX};
   const uint64_t free_space[] = {0, 4096, MiB(5), GiB(16), UINT64_MAX};
   unsigned checks = 0;
   for (unsigned h = 0; h < sizeof(heaps) / sizeof(heaps[0]); ++h)
      for (unsigned u = 0; u < sizeof(usages) / sizeof(usages[0]); ++u)
         for (unsigned a = 0; a < sizeof(free_space) / sizeof(free_space[0]);
              ++a) {
            check_invariants(heaps[h], usages[u], free_space[a]);
            ++checks;
         }

   printf("Honeykrisp memory-budget core: %u invariant cases passed\n", checks);
   return 0;
}
