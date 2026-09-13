/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>
#include "agx_chip.h"

/* Triangle state, initially captured on M5/25G76 without PSIZ.
 * Each range is a complete PPP payload with its own VDM pointer. */
#define AGX_PPP_HAL300_GRAPHICS_RECORDS 9u
#define AGX_PPP_HAL300_GRAPHICS_BYTES 172u

struct agx_ppp_hal300_graphics_fields {
   uint64_t fragment_program;
   uint64_t varying_table;
   unsigned cf_binding_count;
   unsigned cull_mode;
   /* Native pass values: 0 opaque, 1 translucent. Punch-through also needs
    * coverage state and is not supported by this profile yet. */
   unsigned fragment_pass;
   bool front_face_ccw;
   bool point_size_output;
   /* Vertex exports and fragment interpolation are independent. A fragment
    * shader may leave the vertex shader's output unused. Counts are scalar
    * 32-bit UVS slots, not source-language vectors or CF bindings. The
    * interpolator reserves all exported user slots even when a CF binding
    * reads only a subvector; its offset is carried by the VIT separately. */
   unsigned vertex_varying_components;
   unsigned fragment_varying_components;
   bool flat_varying;
   bool depth_less_write;
};

struct agx_ppp_hal300_graphics_core {
   uint8_t bytes[AGX_PPP_HAL300_GRAPHICS_BYTES];
   uint16_t offsets[AGX_PPP_HAL300_GRAPHICS_RECORDS];
   uint8_t lengths[AGX_PPP_HAL300_GRAPHICS_RECORDS];
};

/* Opaque/translucent triangles, optionally up to two vec4 varyings
 * and D32 LESS/write.
 * No stencil, queries, discard, or MSAA. Attachment storage and clear/load/store
 * setup are separate; packing this state alone does not enable depth rendering.
 * Caller must validate those semantic restrictions before using this profile.
 * On failure out is untouched. */
bool agx_ppp_hal300_graphics_core_pack(
   enum agx_chip chip, const struct agx_ppp_hal300_graphics_fields *fields,
   struct agx_ppp_hal300_graphics_core *out);
