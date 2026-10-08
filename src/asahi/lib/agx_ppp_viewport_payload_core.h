/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Bounded, target-aware AGX3 viewport-only PPP payload representation.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AGX_PPP_VIEWPORT_PAYLOAD_MAX_VIEWPORTS 16u
#define AGX_PPP_VIEWPORT_PAYLOAD_MIN_LENGTH    40u
#define AGX_PPP_VIEWPORT_PAYLOAD_MAX_LENGTH    520u
#define AGX_PPP_VIEWPORT_PAYLOAD_GEN6_BASE_HEADER UINT32_C(0x00000c00)
#define AGX_PPP_VIEWPORT_PAYLOAD_GEN7_BASE_HEADER UINT32_C(0x00003000)

struct agx_ppp_viewport_transform_raw {
   uint32_t translate_x;
   uint32_t scale_x;
   uint32_t translate_y;
   uint32_t scale_y;
   uint32_t translate_z;
   uint32_t scale_z;
};

struct agx_ppp_viewport_payload_fields {
   uint8_t viewport_count;
   uint64_t region_clip_raw[AGX_PPP_VIEWPORT_PAYLOAD_MAX_VIEWPORTS];
   uint32_t viewport_control_raw;
   struct agx_ppp_viewport_transform_raw
      viewports[AGX_PPP_VIEWPORT_PAYLOAD_MAX_VIEWPORTS];
};

struct agx_ppp_viewport_payload_core {
   uint8_t bytes[AGX_PPP_VIEWPORT_PAYLOAD_MAX_LENGTH];
   uint16_t length;
};

bool agx_ppp_viewport_payload_core_pack(
   enum agx_chip chip, const struct agx_ppp_viewport_payload_fields *fields,
   struct agx_ppp_viewport_payload_core *out);

bool agx_ppp_viewport_payload_core_unpack(
   enum agx_chip chip, const void *bytes, size_t size,
   struct agx_ppp_viewport_payload_fields *fields);

#ifdef __cplusplus
}
#endif
