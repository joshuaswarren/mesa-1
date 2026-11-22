/* SPDX-License-Identifier: MIT */
#ifndef AGX_DRM_NAME_H
#define AGX_DRM_NAME_H

#include <stdbool.h>
#include <string.h>

/*
 * Kernel driver names that bind Apple Silicon GPUs with the UAPI this
 * driver implements. Mainline uses the historical name; the linux-aurora
 * distribution kernels rename the driver but keep the UAPI, so both names
 * advertise the same contract. Anything else is not this device.
 */
static inline bool
agx_drm_name_is_native(const char *name)
{
   return strcmp(name, "asahi") == 0 || strcmp(name, "aurora") == 0;
}

#endif
