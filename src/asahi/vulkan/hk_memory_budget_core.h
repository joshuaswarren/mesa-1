/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdint.h>

#define HK_MEMORY_BUDGET_GRANULARITY_B (UINT64_C(1) << 20)

static inline uint64_t
hk_memory_budget_B(uint64_t heap_size, uint64_t used, uint64_t available)
{
   if (!heap_size)
      return 0;

   if (used > heap_size)
      used = heap_size;

   /* Keep ten percent of currently available system memory in reserve.  The
    * process's live allocations are already committed, so the reserve applies
    * only to free memory: heapBudget must continue to include heapUsage.
    */
   uint64_t allocatable = available - (available / 10);
   allocatable &= ~(HK_MEMORY_BUDGET_GRANULARITY_B - 1);
   if (allocatable > heap_size - used)
      allocatable = heap_size - used;

   uint64_t budget = used + allocatable;

   /* EXT_memory_budget requires a nonzero value for each real heap, including
    * the low-memory/query-failure case where both estimates above are zero.
    */
   if (!budget)
      budget = heap_size < HK_MEMORY_BUDGET_GRANULARITY_B
                  ? heap_size
                  : HK_MEMORY_BUDGET_GRANULARITY_B;

   return budget;
}
