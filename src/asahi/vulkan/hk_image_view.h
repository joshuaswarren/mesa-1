/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "agx_pack.h"
#include "hk_private.h"
#include "hk_native_descriptor_pack.h"
#include "vk_image.h"

struct hk_device;

#define HK_MAX_PLANES      3
#define HK_MAX_IMAGE_DESCS (10 * HK_MAX_PLANES)

struct hk_image_view {
   struct vk_image_view vk;

   uint8_t plane_count;
   struct {
      uint8_t image_plane;
      uint32_t sampled_heap_index;
      uint32_t storage_heap_index;
      uint32_t input_heap_index;

      struct hk_native_texture_descriptor ia, sampled, ro_storage, background,
         layered_background, emrt_texture;

      struct hk_native_pbe_descriptor storage, eot, layered_eot, emrt_pbe;
      bool storage_pbe_valid, eot_pbe_valid, layered_eot_pbe_valid,
         emrt_pbe_valid;
   } planes[3];
};

VK_DEFINE_NONDISP_HANDLE_CASTS(hk_image_view, vk.base, VkImageView,
                               VK_OBJECT_TYPE_IMAGE_VIEW)
