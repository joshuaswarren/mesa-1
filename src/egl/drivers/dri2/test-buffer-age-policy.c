/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>

#include "egl_dri2_buffer_age_policy.h"

#define CHECK(condition)                                                       \
   do {                                                                        \
      if (!(condition)) {                                                      \
         fprintf(stderr, "buffer-age policy check failed: %s\n", #condition); \
         return 1;                                                             \
      }                                                                        \
   } while (0)

int
main(void)
{
   const char *honeykrisp =
      "zink Vulkan 1.4(Apple A18 Pro (G17P B1) (MESA_HONEYKRISP))";

   CHECK(dri2_force_zero_buffer_age_for_renderer("zink", honeykrisp));

   CHECK(!dri2_force_zero_buffer_age_for_renderer(NULL, honeykrisp));
   CHECK(!dri2_force_zero_buffer_age_for_renderer("zink", NULL));
   CHECK(!dri2_force_zero_buffer_age_for_renderer("asahi", honeykrisp));
   CHECK(!dri2_force_zero_buffer_age_for_renderer(
      "zink", "zink Vulkan 1.4(AMD Radeon (MESA_RADV))"));
   CHECK(!dri2_force_zero_buffer_age_for_renderer(
      "zink", "zink Vulkan 1.4(Apple GPU (MESA_KOSMICKRISP))"));
   CHECK(!dri2_force_zero_buffer_age_for_renderer(
      "zink", "zink Vulkan 1.4(Apple GPU MESA_HONEYKRISP)"));

   puts("EGL DRI2 Honeykrisp buffer-age policy: passed");
   return 0;
}
