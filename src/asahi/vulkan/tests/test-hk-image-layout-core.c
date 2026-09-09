/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_image_layout_core.h"

#include <assert.h>
#include <limits.h>
#include <stdio.h>

static void
check_explicit_linear_stride(void)
{
   uint32_t stride = 0;
   assert(hk_image_explicit_linear_stride(800, 796, &stride));
   assert(stride == 800);
   assert(!hk_image_explicit_linear_stride(0, 0, &stride));
   assert(!hk_image_explicit_linear_stride(0, 0, NULL));
   assert(!hk_image_explicit_linear_stride(792, 800, &stride));
   assert(!hk_image_explicit_linear_stride(801, 800, &stride));
   assert(!hk_image_explicit_linear_stride(UINT64_C(1) << 32, 16, &stride));
}

static void
check_plane_ranges(void)
{
   uint64_t offset, end;

   assert(hk_image_explicit_plane_range(0x4000, 0x18000, &end));
   assert(end == 0x1c000);
   assert(!hk_image_explicit_plane_range(0x4040, 0x1000, &end));
   assert(!hk_image_explicit_plane_range(UINT64_MAX - 127, 256, &end));

   assert(!hk_image_plane_ranges_overlap(0x1000, 0x1000, 0x2000, 0x1000));
   assert(hk_image_plane_ranges_overlap(0x1000, 0x1001, 0x2000, 0x1000));
   assert(hk_image_plane_ranges_overlap(UINT64_MAX - 127, 256, 0, 1));

   assert(hk_image_implicit_plane_range(0x18001, 0x2000, &offset, &end));
   assert(offset == 0x18080);
   assert(end == 0x1a080);
   assert(!hk_image_implicit_plane_range(UINT64_MAX - 63, 1, &offset, &end));
}

static void
check_binding_adds_plane_offset_once(void)
{
   uint64_t address, bo_offset;

   assert(hk_image_plane_binding(0x100000, 0x20000, 0x2000, 0x4000, 0x8000,
                                 &address, &bo_offset));
   assert(bo_offset == 0x6000);
   assert(address == 0x106000);

   assert(!hk_image_plane_binding(0x100000, 0x8000, 0x2000, 0x4000, 0x4000,
                                  &address, &bo_offset));
   assert(!hk_image_plane_binding(0x100000, UINT64_MAX, UINT64_MAX, 1, 0,
                                  &address, &bo_offset));
   assert(!hk_image_plane_binding(UINT64_MAX - 0x100, 0x1000, 0x80, 0x80, 1,
                                  &address, &bo_offset));
}

int
main(void)
{
   check_explicit_linear_stride();
   check_plane_ranges();
   check_binding_adds_plane_offset_once();
   puts("Honeykrisp explicit image-plane layout PASS");
   return 0;
}
