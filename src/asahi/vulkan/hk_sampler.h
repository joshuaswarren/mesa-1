/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "hk_device.h"
#include "hk_physical_device.h"
#include "hk_private.h"

#include "vk_sampler.h"
#include "vk_ycbcr_conversion.h"

#include "vk_format.h"

struct hk_sampler {
   struct vk_sampler vk;
   VkClearColorValue custom_border;
   bool has_border;
   /* The recorded border is VkBorderColor integer data. A float-typed
    * sample of a scaled view converts it to the texel value. */
   bool border_is_int;

   /* Shader-visible immutable sampling state for explicit reconstruction. */
   struct vk_ycbcr_conversion_state ycbcr;
   VkFilter min_filter, mag_filter;
   float mip_lod_bias;

   uint8_t plane_count;
   uint16_t lod_bias_fp16;

   struct {
      struct hk_rc_sampler *hw;
   } planes[2];
};

VK_DEFINE_NONDISP_HANDLE_CASTS(hk_sampler, vk.base, VkSampler,
                               VK_OBJECT_TYPE_SAMPLER)
