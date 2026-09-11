/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_ioto_arena_core.h"

static bool
power_of_two(uint32_t value)
{
   return value != 0 && (value & (value - 1)) == 0;
}

size_t
hk_ioto_arena_chunk_size(size_t program_size, uint32_t align)
{
   if (program_size == 0 || !power_of_two(align))
      return 0;

   if (program_size > SIZE_MAX - (size_t)(align - 1))
      return 0;

   const size_t aligned =
      (program_size + (size_t)align - 1) & ~((size_t)align - 1);

   /* One oversized program still gets a chunk of its own rather than a
    * separate allocation path; the next program starts a fresh chunk. */
   return aligned > HK_IOTO_ARENA_CHUNK_SIZE ? aligned
                                             : HK_IOTO_ARENA_CHUNK_SIZE;
}

bool
hk_ioto_arena_place(uint32_t cursor, uint32_t capacity, size_t program_size,
                    uint32_t align, struct hk_ioto_arena_span *span)
{
   if (span == NULL || program_size == 0 || !power_of_two(align))
      return false;

   if (program_size > UINT32_MAX || cursor > capacity)
      return false;

   const uint32_t mask = align - 1;
   if (cursor > UINT32_MAX - mask)
      return false;

   const uint32_t offset = (cursor + mask) & ~mask;
   if (offset > capacity || (uint32_t)program_size > capacity - offset)
      return false;

   *span = (struct hk_ioto_arena_span){
      .offset = offset,
      .next_offset = offset + (uint32_t)program_size,
   };
   return true;
}
