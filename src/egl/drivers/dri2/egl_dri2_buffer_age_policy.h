/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#ifndef EGL_DRI2_BUFFER_AGE_POLICY_H
#define EGL_DRI2_BUFFER_AGE_POLICY_H

#include <stdbool.h>
#include <string.h>

/*
 * Zink includes the Vulkan driver ID in its renderer name.  Keep this
 * containment tied to Honeykrisp rather than disabling buffer age for every
 * Zink driver (or for native Asahi GL).
 */
static inline bool
dri2_force_zero_buffer_age_for_renderer(const char *dri_driver_name,
                                        const char *renderer_name)
{
   return dri_driver_name != NULL && renderer_name != NULL &&
          strcmp(dri_driver_name, "zink") == 0 &&
          strstr(renderer_name, "(MESA_HONEYKRISP)") != NULL;
}

#endif
