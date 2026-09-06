/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "asahi/lib/agx_chip.h"
#include "vulkan/vulkan_core.h"

#include <stdbool.h>
#include <stdint.h>

struct hk_g17p_buffer_to_image_copy {
   enum agx_chip chip;
   VkFormat format;
   VkImageLayout layout;
   VkImageType image_type;
   VkSampleCountFlagBits samples;
   uint8_t plane_count;
   uint32_t mip_levels;
   uint32_t array_layers;
   VkExtent3D image_extent;

   VkImageAspectFlags aspect;
   uint32_t mip_level;
   uint32_t base_array_layer;
   uint32_t layer_count;
   VkOffset3D image_offset;
   VkExtent3D copy_extent;
   uint32_t buffer_row_length;
   uint32_t buffer_image_height;
};

/* Native uncompressed color uploads preserve rectangles and mip/layer
 * subresources. Multi-layer regions are split into native single-layer
 * passes without changing source layout or copying index/pixel data. */
bool hk_g17p_buffer_to_image_copy_supported(
   const struct hk_g17p_buffer_to_image_copy *copy);

/* HAL200's qualified tile programs are single-layer.  Convert one layer of a
 * valid 2D-array image copy without changing its rectangle or mip levels.
 * The resolved source/destination layer counts must describe the same copy. */
bool hk_g17p_image_copy_slice(enum agx_chip chip, VkImageType src_type,
                              VkImageType dst_type, uint32_t src_layers,
                              uint32_t dst_layers,
                              const VkImageCopy2 *region, uint32_t slice,
                              VkImageCopy2 *out);

bool hk_g17p_image_blit_slice(enum agx_chip chip, VkImageType src_type,
                              VkImageType dst_type, uint32_t src_layers,
                              uint32_t dst_layers,
                              const VkImageBlit2 *region, uint32_t slice,
                              VkImageBlit2 *out);
