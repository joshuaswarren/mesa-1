/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_image_layout_core.h"

#include <limits.h>

static bool
add_u64(uint64_t a, uint64_t b, uint64_t *sum)
{
   if (b > UINT64_MAX - a)
      return false;

   *sum = a + b;
   return true;
}

bool
hk_image_explicit_linear_stride(uint64_t row_pitch_B,
                                uint64_t minimum_row_pitch_B,
                                uint32_t *stride_B_out)
{
   if (!stride_B_out || !row_pitch_B || row_pitch_B > UINT32_MAX ||
       row_pitch_B < minimum_row_pitch_B ||
       (row_pitch_B % HK_IMAGE_LINEAR_STRIDE_ALIGN_B) != 0)
      return false;

   *stride_B_out = row_pitch_B;
   return true;
}

bool
hk_image_explicit_plane_range(uint64_t offset_B, uint64_t size_B,
                              uint64_t *end_B_out)
{
   if (!end_B_out || (offset_B % HK_IMAGE_PLANE_ALIGN_B) != 0)
      return false;

   return add_u64(offset_B, size_B, end_B_out);
}

bool
hk_image_plane_ranges_overlap(uint64_t offset_a_B, uint64_t size_a_B,
                              uint64_t offset_b_B, uint64_t size_b_B)
{
   uint64_t end_a_B, end_b_B;
   if (!add_u64(offset_a_B, size_a_B, &end_a_B) ||
       !add_u64(offset_b_B, size_b_B, &end_b_B))
      return true;

   return offset_a_B < end_b_B && offset_b_B < end_a_B;
}

bool
hk_image_implicit_plane_range(uint64_t cursor_B, uint64_t size_B,
                              uint64_t *offset_B_out, uint64_t *end_B_out)
{
   if (!offset_B_out || !end_B_out ||
       cursor_B > UINT64_MAX - (HK_IMAGE_PLANE_ALIGN_B - 1))
      return false;

   const uint64_t offset_B = (cursor_B + (HK_IMAGE_PLANE_ALIGN_B - 1)) &
                             ~(uint64_t)(HK_IMAGE_PLANE_ALIGN_B - 1);
   if (!add_u64(offset_B, size_B, end_B_out))
      return false;

   *offset_B_out = offset_B;
   return true;
}

bool
hk_image_plane_binding(uint64_t bo_address, uint64_t bo_size_B,
                       uint64_t memory_offset_B, uint64_t plane_offset_B,
                       uint64_t plane_size_B, uint64_t *address_out,
                       uint64_t *bo_offset_out)
{
   if (!address_out || !bo_offset_out)
      return false;

   uint64_t bo_offset_B;
   if (!add_u64(memory_offset_B, plane_offset_B, &bo_offset_B) ||
       bo_offset_B > bo_size_B || plane_size_B > bo_size_B - bo_offset_B)
      return false;

   uint64_t address;
   if (!add_u64(bo_address, bo_offset_B, &address) ||
       plane_size_B > UINT64_MAX - address)
      return false;

   *address_out = address;
   *bo_offset_out = bo_offset_B;
   return true;
}
