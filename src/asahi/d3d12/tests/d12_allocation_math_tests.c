/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT */

#include "../d12_allocation_math.h"
#include <assert.h>
#include <stdio.h>

int
main(void)
{
   uint64_t total = 0, alignment = 1, offset, size, rounded;

   /* A normal buffer followed by a small texture: individual sizes and
    * offsets are padded to their own alignment; the aggregate is padded to
    * 64 KiB. The last resource does not determine the heap alignment. */
   assert(d12_allocation_append(32, 65536, &total, &alignment, &offset, &size));
   assert(offset == 0 && size == 65536);
   assert(d12_allocation_append(1, 4096, &total, &alignment, &offset, &size));
   assert(offset == 65536 && size == 4096 && total == 69632);
   assert(d12_allocation_align(total, alignment, &rounded));
   assert(rounded == 131072);

   /* The documented 64 KiB / 4 MiB / 64 KiB ordering requires 12 MiB;
    * moving the large-alignment resource to the end requires only 8 MiB. */
   for (unsigned large = 0; large < 3; large++) {
      total = 0;
      alignment = 1;
      for (unsigned i = 0; i < 3; i++) {
         uint64_t a = i == large ? 4194304 : 65536;
         assert(d12_allocation_append(1, a, &total, &alignment, &offset, &size));
      }
      assert(d12_allocation_align(total, alignment, &rounded));
      assert(rounded == (large == 1 ? 12582912 : 8388608));
   }

   assert(d12_allocation_align(0, 1, &rounded) && rounded == 0);
   assert(d12_allocation_align(UINT64_MAX, 1, &rounded));
   assert(!d12_allocation_align(UINT64_MAX, 65536, &rounded));
   assert(!d12_allocation_align(0, 0, &rounded));
   assert(!d12_allocation_align(4, 3, &rounded));

   /* Distinguish size rounding, offset rounding, sum overflow and final
    * aggregate rounding. Failed appends must not mutate the accumulator. */
   total = 0;
   alignment = 1;
   assert(!d12_allocation_append(UINT64_MAX, 65536, &total, &alignment,
                                 &offset, &size));
   assert(total == 0 && alignment == 1);
   total = UINT64_MAX;
   assert(!d12_allocation_append(1, 65536, &total, &alignment, &offset, &size));
   total = UINT64_MAX - 65535;
   assert(!d12_allocation_append(1, 65536, &total, &alignment, &offset, &size));
   assert(total == UINT64_MAX - 65535 && alignment == 1);
   assert(d12_allocation_append(1, 4096, &total, &alignment, &offset, &size));
   assert(!d12_allocation_align(total, 65536, &rounded));

   /* A BO may round a 1 KiB heap to a page. Only the declared 1 KiB belongs
    * to the application; an enormous offset must not wrap back into it. */
   assert(d12_allocation_fits(0, 1024, 1024));
   assert(d12_allocation_fits(1024, 0, 1024));
   assert(!d12_allocation_fits(0, 1025, 1024));
   assert(!d12_allocation_fits(65536, 1, 1024));
   assert(!d12_allocation_fits(UINT64_MAX - 65535, 65536, 1024));

   puts("allocation math: ALL PASS");
   return 0;
}
