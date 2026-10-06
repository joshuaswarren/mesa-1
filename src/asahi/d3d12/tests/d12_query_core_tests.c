/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "../d12_query_core.h"

#include <stdio.h>
#include <stdlib.h>

#define CHECK(condition)                                                     \
   do {                                                                      \
      if (!(condition)) {                                                    \
         fprintf(stderr, "d12_query_core_tests: failed at line %d\n",        \
                 __LINE__);                                                  \
         exit(1);                                                            \
      }                                                                      \
   } while (0)

int
main(void)
{
   CHECK(d12_query_core_heap_supported(D12_QUERY_CORE_HEAP_OCCLUSION));
   CHECK(d12_query_core_heap_supported(D12_QUERY_CORE_HEAP_TIMESTAMP));
   CHECK(d12_query_core_heap_supported(
      D12_QUERY_CORE_HEAP_COPY_QUEUE_TIMESTAMP));
   CHECK(!d12_query_core_heap_supported(4));
   CHECK(!d12_query_core_heap_supported(6));

   CHECK(d12_query_core_type_matches(D12_QUERY_CORE_HEAP_TIMESTAMP,
                                     D12_QUERY_CORE_TYPE_TIMESTAMP));
   CHECK(d12_query_core_type_matches(
      D12_QUERY_CORE_HEAP_COPY_QUEUE_TIMESTAMP,
      D12_QUERY_CORE_TYPE_TIMESTAMP));
   CHECK(!d12_query_core_type_matches(
      D12_QUERY_CORE_HEAP_COPY_QUEUE_TIMESTAMP,
      D12_QUERY_CORE_TYPE_OCCLUSION));
   CHECK(d12_query_core_type_matches(
      D12_QUERY_CORE_HEAP_SO_STATISTICS,
      D12_QUERY_CORE_TYPE_SO_STATISTICS_STREAM3));

   CHECK(d12_query_core_is_timestamp_heap(D12_QUERY_CORE_HEAP_TIMESTAMP));
   CHECK(d12_query_core_is_timestamp_heap(
      D12_QUERY_CORE_HEAP_COPY_QUEUE_TIMESTAMP));
   CHECK(!d12_query_core_is_timestamp_heap(D12_QUERY_CORE_HEAP_OCCLUSION));

   CHECK(d12_query_core_command_compatible(
      D12_QUERY_CORE_HEAP_COPY_QUEUE_TIMESTAMP,
      D12_QUERY_CORE_COMMAND_COPY));
   CHECK(!d12_query_core_command_compatible(
      D12_QUERY_CORE_HEAP_COPY_QUEUE_TIMESTAMP,
      D12_QUERY_CORE_COMMAND_DIRECT));
   CHECK(d12_query_core_command_compatible(
      D12_QUERY_CORE_HEAP_TIMESTAMP, D12_QUERY_CORE_COMMAND_DIRECT));
   CHECK(d12_query_core_command_compatible(
      D12_QUERY_CORE_HEAP_TIMESTAMP, D12_QUERY_CORE_COMMAND_COMPUTE));
   CHECK(!d12_query_core_command_compatible(
      D12_QUERY_CORE_HEAP_TIMESTAMP, D12_QUERY_CORE_COMMAND_COPY));

   const uint32_t heaps[] = {
      D12_QUERY_CORE_HEAP_OCCLUSION,
      D12_QUERY_CORE_HEAP_PIPELINE_STATISTICS,
      D12_QUERY_CORE_HEAP_SO_STATISTICS,
   };
   for (unsigned i = 0; i < sizeof(heaps) / sizeof(heaps[0]); i++) {
      CHECK(d12_query_core_command_compatible(
         heaps[i], D12_QUERY_CORE_COMMAND_DIRECT));
      CHECK(!d12_query_core_command_compatible(
         heaps[i], D12_QUERY_CORE_COMMAND_BUNDLE));
      CHECK(!d12_query_core_command_compatible(
         heaps[i], D12_QUERY_CORE_COMMAND_COMPUTE));
      CHECK(!d12_query_core_command_compatible(
         heaps[i], D12_QUERY_CORE_COMMAND_COPY));
   }
   CHECK(!d12_query_core_command_compatible(4,
                                             D12_QUERY_CORE_COMMAND_DIRECT));
   CHECK(!d12_query_core_command_compatible(
      D12_QUERY_CORE_HEAP_TIMESTAMP, D12_QUERY_CORE_COMMAND_BUNDLE));

   CHECK(d12_query_core_resolve_range_valid(4, 1, 3, 8, 32, 8));
   CHECK(!d12_query_core_resolve_range_valid(4, 1, 4, 8, 64, 0));
   CHECK(!d12_query_core_resolve_range_valid(4, UINT32_MAX, 1, 8, 64, 0));
   CHECK(!d12_query_core_resolve_range_valid(4, 1, UINT32_MAX, 8, 64, 0));
   CHECK(!d12_query_core_resolve_range_valid(4, 0, 1, 8, 16, 1));
   CHECK(!d12_query_core_resolve_range_valid(4, 0, 1, 8, 16, UINT64_MAX));
   CHECK(!d12_query_core_resolve_range_valid(4, 0, 1, 8, 7, 0));
   CHECK(!d12_query_core_resolve_range_valid(4, 0, 0, 8, 64, 0));
   CHECK(!d12_query_core_resolve_range_valid(4, 0, 1, 0, 64, 0));
   /* Statistics results can be much wider than a scalar timestamp. */
   CHECK(d12_query_core_resolve_range_valid(4, 2, 2, 88, 184, 8));
   CHECK(!d12_query_core_resolve_range_valid(4, 2, 2, 88, 183, 8));
   CHECK(d12_query_core_resolve_range_valid(UINT32_MAX, 0, UINT32_MAX,
                                            UINT32_MAX, UINT64_MAX, 0));

   puts("d12_query_core_tests: PASS");
   return 0;
}
