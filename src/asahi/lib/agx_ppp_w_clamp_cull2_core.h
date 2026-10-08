/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Bounded, target-aware AGX3 PPP W-clamp/Cull-2 payload representation.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AGX_PPP_W_CLAMP_CULL2_CORE_LENGTH 12u
#define AGX_PPP_W_CLAMP_CULL2_GEN6_HEADER UINT32_C(0x00410000)
#define AGX_PPP_W_CLAMP_CULL2_GEN7_HEADER UINT32_C(0x00840002)

struct agx_ppp_w_clamp_cull2_fields {
   uint32_t w_clamp_raw;
   uint32_t cull2_raw;
};

struct agx_ppp_w_clamp_cull2_core {
   uint8_t bytes[AGX_PPP_W_CLAMP_CULL2_CORE_LENGTH];
   uint8_t length;
};

bool agx_ppp_w_clamp_cull2_core_pack(
   enum agx_chip chip, const struct agx_ppp_w_clamp_cull2_fields *fields,
   struct agx_ppp_w_clamp_cull2_core *out);

bool agx_ppp_w_clamp_cull2_core_unpack(
   enum agx_chip chip, const void *bytes, size_t size,
   struct agx_ppp_w_clamp_cull2_fields *fields);

#ifdef __cplusplus
}
#endif
