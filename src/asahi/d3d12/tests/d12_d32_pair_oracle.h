/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 * Independent CPU pixel-centre model for the immutable tri.dxil first pair.
 */
#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static inline bool
d12_d32_pair_covered(unsigned x, unsigned y)
{
   /* Twice the window coordinates of (1,-1), (-1,-1), (5,1). */
   const int64_t p[3][2] = {{512,512}, {0,512}, {1536,0}};
   for (unsigned e = 0; e < 3; ++e) {
      unsigned n = (e + 1) % 3;
      int64_t dx = p[n][0] - p[e][0], dy = p[n][1] - p[e][1];
      int64_t edge = dx * ((int64_t)y * 2 + 1 - p[e][1]) -
                     dy * ((int64_t)x * 2 + 1 - p[e][0]);
      bool top_left = dy == 0 ? dx > 0 : dy < 0;
      if (edge < 0 || (edge == 0 && !top_left)) return false;
   }
   return true;
}

static inline unsigned
d12_d32_pair_pixel_mismatches(const uint8_t *pixels, size_t stride,
                             bool draw_passes, unsigned *covered)
{
   const uint8_t red[4] = {255,0,0,255}, blue[4] = {64,128,191,255};
   if (!pixels || !covered || stride < 256 * 4 || stride > SIZE_MAX / 256) return UINT32_MAX;
   unsigned bad = 0, triangle = 0;
   for (unsigned y = 0; y < 256; ++y)
      for (unsigned x = 0; x < 256; ++x) {
         bool inside = d12_d32_pair_covered(x, y);
         triangle += inside;
         const uint8_t *want = inside && draw_passes ? blue : red;
         bad += memcmp(pixels + y * stride + x * 4, want, 4) != 0;
      }
   *covered = triangle;
   return bad;
}
