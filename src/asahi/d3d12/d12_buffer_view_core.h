/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#ifndef D12_BUFFER_VIEW_CORE_H
#define D12_BUFFER_VIEW_CORE_H

#include <stdbool.h>
#include <stdint.h>

/* Each legacy native descriptor covers a bounded linear 2D window. */
#define D12_TYPED_BUFFER_WINDOW_LOG2 27
#define D12_TYPED_BUFFER_WINDOW_ELEMENTS                                       \
   (UINT32_C(1) << D12_TYPED_BUFFER_WINDOW_LOG2)
#define D12_TYPED_BUFFER_WINDOWS 4
/* Keep the unqualified AGX3 path at its previous extent. */
#define D12_TYPED_BUFFER_VIEW_MAX_ELEMENTS D12_TYPED_BUFFER_WINDOW_ELEMENTS
#define D12_LEGACY_TYPED_BUFFER_MAX_ELEMENTS                                   \
   (D12_TYPED_BUFFER_WINDOWS * D12_TYPED_BUFFER_WINDOW_ELEMENTS)

static inline bool
d12_typed_buffer_window(uint32_t count, uint32_t stride_B, unsigned window,
                        uint64_t *offset_B, uint32_t *window_count)
{
   const uint64_t first = (uint64_t)window * D12_TYPED_BUFFER_WINDOW_ELEMENTS;
   if (!stride_B || window >= D12_TYPED_BUFFER_WINDOWS || first >= count)
      return false;
   const uint32_t remaining = count - first;
   *offset_B = first * stride_B;
   *window_count = remaining < D12_TYPED_BUFFER_WINDOW_ELEMENTS
                      ? remaining
                      : D12_TYPED_BUFFER_WINDOW_ELEMENTS;
   return true;
}

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
