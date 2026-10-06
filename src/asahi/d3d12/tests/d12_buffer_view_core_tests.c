/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "../d12_buffer_view_core.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>

int
main(void)
{
   uint64_t offset = 0, size = 0;

   /* The reference runtime accepts near-4GiB raw/structured views. */
   assert(d12_buffer_view_range(UINT64_C(0x100000000), 1,
                                UINT32_C(0x40000000) - 9, 4, &offset, &size));
   assert(offset == 4 && size == UINT64_C(0xffffffdc));
   assert(d12_buffer_view_range(UINT64_C(0x100000000), 2,
                                UINT32_C(0x80000000) - 10, 2, &offset, &size));
   assert(offset == 4 && size == UINT64_C(0xffffffec));
   const uint64_t max_structured_size = d12_buffer_view_size_B(UINT32_MAX, 64);
   assert(max_structured_size == UINT64_C(0x3fffffffc0));
   assert(d12_buffer_view_range(max_structured_size, 0, UINT32_MAX, 64,
                                &offset, &size));
   assert(offset == 0 && size == max_structured_size);

   assert(d12_buffer_view_range(UINT64_C(0x100000080), 0,
                                UINT32_C(0x02000001), 128, &offset, &size));
   assert(offset == 0);
   assert(size == UINT64_C(0x100000080));
   assert(size > UINT32_MAX);
   assert(d12_buffer_view_size_B(UINT32_C(0x02000001), 128) == size);

   assert(d12_buffer_view_range(UINT64_C(0x200000100), UINT64_C(0x01000000),
                                UINT32_C(0x01000001), 128, &offset, &size));
   assert(offset == UINT64_C(0x80000000));
   assert(size == UINT64_C(0x80000080));

   assert(!d12_buffer_view_range(UINT64_MAX, UINT64_MAX, 1, 2,
                                 &offset, &size));
   assert(!d12_buffer_view_range(UINT64_C(0x100000000), 0,
                                 UINT32_C(0x02000001), 128, &offset, &size));
   assert(!d12_buffer_view_range(4096, 0, 1, 0, &offset, &size));

   assert(d12_buffer_access_in_bounds(8, 0, 8));
   assert(!d12_buffer_access_in_bounds(8, 4, 8));
   assert(!d12_buffer_access_in_bounds(8, 8, 1));
   assert(!d12_buffer_access_in_bounds(8, 0, 0));
   assert(d12_buffer_access_in_bounds(UINT64_C(0x100000008),
                                      UINT64_C(0x100000000), 8));
   assert(!d12_buffer_access_in_bounds(UINT64_C(0x100000008),
                                       UINT64_C(0x100000004), 8));
   assert(!d12_buffer_access_in_bounds(UINT64_MAX, UINT64_MAX - 3, 8));

   /* Windows preserve the full logical count and a 64-bit byte offset,
    * including a format whose fourth window begins beyond 4 GiB. */
   for (unsigned window = 0; window < D12_TYPED_BUFFER_WINDOWS; ++window) {
      uint32_t count;
      assert(d12_typed_buffer_window(D12_LEGACY_TYPED_BUFFER_MAX_ELEMENTS, 16,
                                     window, &offset, &count));
      assert(offset ==
             (uint64_t)window * D12_TYPED_BUFFER_WINDOW_ELEMENTS * 16);
      assert(count == D12_TYPED_BUFFER_WINDOW_ELEMENTS);
   }
   uint32_t count;
   assert(d12_typed_buffer_window(D12_TYPED_BUFFER_WINDOW_ELEMENTS + 1, 2, 1,
                                  &offset, &count));
   assert(offset == UINT64_C(0x10000000) && count == 1);
   assert(!d12_typed_buffer_window(D12_TYPED_BUFFER_WINDOW_ELEMENTS, 2, 1,
                                   &offset, &count));
   assert(!d12_typed_buffer_window(UINT32_MAX, 4, D12_TYPED_BUFFER_WINDOWS,
                                   &offset, &count));
   assert(!d12_typed_buffer_window(1, 0, 0, &offset, &count));
   assert(!d12_typed_buffer_window(0, 4, 0, &offset, &count));

   puts("d12_buffer_view_core_tests: PASS");
   return 0;
}
