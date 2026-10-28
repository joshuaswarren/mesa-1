/* SPDX-License-Identifier: MIT */
#pragma once

#include <stdbool.h>
#include <stdint.h>
#include "compiler/nir/nir.h"
#include "drm-uapi/asahi_drm.h"

/* The external-write predicate itself is hk_shader_writes_external_memory()
 * in hk_shader.c. */
static inline void
hk_m5_accumulate_shader_writes(bool *command_writes, bool shader_writes)
{
   *command_writes |= shader_writes;
}

static inline uint32_t
hk_m5_shader_write_class_flags(uint64_t features, bool command_writes)
{
   return command_writes &&
          (features & DRM_ASAHI_FEATURE_RENDER_SHADER_WRITES)
             ? DRM_ASAHI_RENDER_SHADER_WRITES : 0;
}
