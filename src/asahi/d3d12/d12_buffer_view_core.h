/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#ifndef D12_BUFFER_VIEW_CORE_H
#define D12_BUFFER_VIEW_CORE_H

#include <stdbool.h>
#include <stdint.h>

/* Qualified formatted-buffer extent; raw/structured views use byte bounds. */
#define D12_TYPED_BUFFER_VIEW_MAX_ELEMENTS (UINT32_C(1) << 27)

/*
 * D3D12 describes a buffer view as a 64-bit first element and a 32-bit
 * element count/stride.  Keep that representation until the final byte-range
 * calculation: a valid structured view may be much larger than 4 GiB even
 * though each individual field fits in 32 bits.
 */
static inline bool
d12_buffer_view_range(uint64_t resource_size_B, uint64_t first_element,
                      uint32_t num_elements, uint32_t stride_B,
                      uint64_t *offset_B, uint64_t *size_B)
{
   if (!stride_B || first_element > UINT64_MAX / stride_B)
      return false;

   const uint64_t offset = first_element * stride_B;
   const uint64_t size = (uint64_t)num_elements * stride_B;
   if (offset > resource_size_B || size > resource_size_B - offset)
      return false;

   if (offset_B)
      *offset_B = offset;
   if (size_B)
      *size_B = size;
   return true;
}

static inline uint64_t
d12_buffer_view_size_B(uint32_t num_elements, uint32_t stride_B)
{
   return (uint64_t)num_elements * stride_B;
}

/* Test the complete byte interval, not just its first byte. Shader robustness
 * uses the same rule for every scalar lane, including 64-bit loads and stores
 * that begin inside a view but end past it. */
static inline bool
d12_buffer_access_in_bounds(uint64_t view_size_B, uint64_t offset_B,
                            uint32_t access_size_B)
{
   return access_size_B && offset_B < view_size_B &&
          access_size_B <= view_size_B - offset_B;
}

#endif
