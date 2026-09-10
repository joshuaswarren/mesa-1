/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * The placement arithmetic behind the G17P Ioto program arena. Every program
 * must land on the 64-byte boundary the launch pointer requires, no placement
 * may run past the chunk, and the caller must be told to start a new chunk
 * rather than being handed an address outside the one it has.
 */

#include "hk_ioto_arena_core.h"

#include <assert.h>
#include <stdio.h>

#define ALIGN 64u

static void
test_chunk_size(void)
{
   /* Ordinary programs share the standard chunk. */
   assert(hk_ioto_arena_chunk_size(1, ALIGN) == HK_IOTO_ARENA_CHUNK_SIZE);
   assert(hk_ioto_arena_chunk_size(246, ALIGN) == HK_IOTO_ARENA_CHUNK_SIZE);
   assert(hk_ioto_arena_chunk_size(HK_IOTO_ARENA_CHUNK_SIZE, ALIGN) ==
          HK_IOTO_ARENA_CHUNK_SIZE);

   /* A program larger than the standard chunk gets one sized to it, rounded
    * up to the launch alignment. */
   assert(hk_ioto_arena_chunk_size(HK_IOTO_ARENA_CHUNK_SIZE + 1, ALIGN) ==
          HK_IOTO_ARENA_CHUNK_SIZE + ALIGN);

   /* Unrepresentable requests are refused, not clamped. */
   assert(hk_ioto_arena_chunk_size(0, ALIGN) == 0);
   assert(hk_ioto_arena_chunk_size(64, 0) == 0);
   assert(hk_ioto_arena_chunk_size(64, 96) == 0);
}

static void
test_placement_is_aligned_and_packed(void)
{
   struct hk_ioto_arena_span span;

   assert(hk_ioto_arena_place(0, HK_IOTO_ARENA_CHUNK_SIZE, 246, ALIGN, &span));
   assert(span.offset == 0);
   assert(span.next_offset == 246);

   /* The next program rounds up to the launch alignment, and the cursor is
    * left at the program's true end so an over-reserved composition gives its
    * slack back. */
   assert(hk_ioto_arena_place(246, HK_IOTO_ARENA_CHUNK_SIZE, 190, ALIGN,
                              &span));
   assert(span.offset == 256);
   assert((span.offset % ALIGN) == 0);
   assert(span.next_offset == 446);

   /* An already-aligned cursor is not pushed forward. */
   assert(hk_ioto_arena_place(256, HK_IOTO_ARENA_CHUNK_SIZE, 64, ALIGN,
                              &span));
   assert(span.offset == 256);
}

static void
test_placement_refuses_overflow(void)
{
   struct hk_ioto_arena_span span;

   /* Exactly filling the chunk is allowed. */
   assert(hk_ioto_arena_place(0, 256, 256, ALIGN, &span));
   assert(span.offset == 0 && span.next_offset == 256);

   /* One byte more is not. */
   assert(!hk_ioto_arena_place(0, 256, 257, ALIGN, &span));

   /* Nor is a program that only fails once alignment is applied. */
   assert(!hk_ioto_arena_place(200, 256, 64, ALIGN, &span));

   /* A full chunk refuses everything. */
   assert(!hk_ioto_arena_place(256, 256, 1, ALIGN, &span));

   /* Degenerate inputs are refused rather than wrapping. */
   assert(!hk_ioto_arena_place(0, 256, 0, ALIGN, &span));
   assert(!hk_ioto_arena_place(0, 256, 64, 0, &span));
   assert(!hk_ioto_arena_place(0, 256, 64, 96, &span));
   assert(!hk_ioto_arena_place(300, 256, 64, ALIGN, &span));
   assert(!hk_ioto_arena_place(UINT32_MAX, UINT32_MAX, 64, ALIGN, &span));
   assert(!hk_ioto_arena_place(0, 256, (size_t)UINT32_MAX + 1, ALIGN, &span));
}

static void
test_a_chunk_holds_a_frame_of_programs(void)
{
   /* The point of the arena: a chunk holds hundreds of programs, where one
    * BO each would cost a 16KiB page plus a 16KiB guard page apiece. */
   struct hk_ioto_arena_span span;
   uint32_t cursor = 0;
   unsigned placed = 0;

   while (hk_ioto_arena_place(cursor, HK_IOTO_ARENA_CHUNK_SIZE, 246, ALIGN,
                              &span)) {
      cursor = span.next_offset;
      ++placed;
   }

   assert(placed >= 400);
}

int
main(void)
{
   test_chunk_size();
   test_placement_is_aligned_and_packed();
   test_placement_refuses_overflow();
   test_a_chunk_holds_a_frame_of_programs();

   puts("hk_ioto_arena_core: ok");
   return 0;
}
