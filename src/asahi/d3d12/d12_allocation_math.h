/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT */

#pragma once

#include <stdbool.h>
#include <stdint.h>

/* Placement queries must refuse overflow rather than return a small heap.
 * Keep both rounding and accumulation checked, including the final padding
 * to the largest alignment in a heterogeneous resource array. */
static inline bool
d12_allocation_align(uint64_t size, uint64_t alignment, uint64_t *result)
{
   if (!alignment || (alignment & (alignment - 1)) ||
       size > UINT64_MAX - (alignment - 1))
      return false;
   *result = (size + alignment - 1) & ~(alignment - 1);
   return true;
}

static inline bool
d12_allocation_append(uint64_t size, uint64_t alignment, uint64_t *total,
                      uint64_t *max_alignment, uint64_t *offset,
                      uint64_t *padded_size)
{
   uint64_t start, padded;
   if (!size || !d12_allocation_align(*total, alignment, &start) ||
       !d12_allocation_align(size, alignment, &padded) ||
       padded > UINT64_MAX - start)
      return false;

   *offset = start;
   *padded_size = padded;
   *total = start + padded;
   if (alignment > *max_alignment)
      *max_alignment = alignment;
   return true;
}

static inline bool
d12_allocation_fits(uint64_t offset, uint64_t size, uint64_t capacity)
{
   return offset <= capacity && size <= capacity - offset;
}
