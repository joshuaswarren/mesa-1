/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>
#include "agx_chip.h"

/* The opaque triangle profile, initially captured on M5/25G76 without PSIZ.
 * Each range is a complete PPP payload with its own VDM pointer. */
#define AGX_PPP_HAL300_GRAPHICS_RECORDS 9u
#define AGX_PPP_HAL300_GRAPHICS_BYTES 172u

struct agx_ppp_hal300_graphics_fields {
   uint64_t fragment_program;
   uint64_t varying_table;
   unsigned cf_binding_count;
   unsigned cull_mode;
   bool front_face_ccw;
   bool point_size_output;
   bool color_varying;
   bool flat_varying;
   bool depth_less_write;
};

struct agx_ppp_hal300_graphics_core {
   uint8_t bytes[AGX_PPP_HAL300_GRAPHICS_BYTES];
   uint16_t offsets[AGX_PPP_HAL300_GRAPHICS_RECORDS];
   uint8_t lengths[AGX_PPP_HAL300_GRAPHICS_RECORDS];
};

/* Opaque triangles, optionally one float4 smooth/flat color and D32 LESS/write.
 * No stencil, queries, discard, or MSAA. Attachment storage and clear/load/store
 * setup are separate; packing this state alone does not enable depth rendering.
 * Caller must validate those semantic restrictions before using this profile.
 * On failure out is untouched. */
bool agx_ppp_hal300_graphics_core_pack(
   enum agx_chip chip, const struct agx_ppp_hal300_graphics_fields *fields,
   struct agx_ppp_hal300_graphics_core *out);
