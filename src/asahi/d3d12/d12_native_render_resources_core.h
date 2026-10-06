/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT */
#pragma once
#include <stdint.h>
#include "drm-uapi/asahi_drm.h"
struct d12_native_render_resource_specs {
   uint64_t bg, eot, partial_bg, partial_eot;
};
void d12_native_render_resource_specs_pack(
   const struct d12_native_render_resource_specs *specs,
   struct drm_asahi_cmd_render *cmd);
