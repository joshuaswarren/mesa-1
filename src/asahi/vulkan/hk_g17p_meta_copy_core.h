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

struct hk_g17p_compressed_copy_plan {
   uint32_t block_width;
   uint32_t block_height;
   uint32_t block_size_B;
   uint32_t base_x_el;
   uint32_t base_y_el;
   uint32_t blocks_x;
   uint32_t blocks_y;
   uint32_t first_slice;
   uint32_t slice_count;
   uint64_t buffer_row_stride_B;
   uint64_t buffer_slice_stride_B;
   uint64_t buffer_span_B;
};

/* The three layouts vkCmdCopyBufferToImage accepts as dstImageLayout:
 * VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL and
 * VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR.  Honeykrisp never acts on a layout, so
 * this is an API-validity check rather than a capability check: it exists only
 * so the G17P fast paths reject a destination the caller could not have meant.
 * Keep the whole legal set, because a client that only ever passed
 * VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL starts passing VK_IMAGE_LAYOUT_GENERAL
 * for every copy the moment it enables VK_KHR_unified_image_layouts. */
bool hk_g17p_transfer_dst_layout_supported(VkImageLayout layout);

/* The corresponding legal set for vkCmdCopyImageToBuffer srcImageLayout. */
bool hk_g17p_transfer_src_layout_supported(VkImageLayout layout);

/* The complete BC family has captured G17P sampled descriptors and shares the
 * hardware-validated Morton-of-4x4-blocks layout. */
bool hk_g17p_bc_format_supported(VkFormat format);

enum hk_g17p_block_format_family {
   HK_G17P_BLOCK_FORMAT_NONE,
   HK_G17P_BLOCK_FORMAT_BC,
   HK_G17P_BLOCK_FORMAT_ETC2_EAC,
   HK_G17P_BLOCK_FORMAT_ASTC_LDR,
};

/* Classify every Vulkan 1.0 LDR block-compressed family carried by the
 * native G17P texture descriptors. ASTC HDR/SFLOAT is deliberately separate
 * from textureCompressionASTC_LDR and is not admitted here. */
enum hk_g17p_block_format_family
hk_g17p_block_format_family(VkFormat format);

/* Native uncompressed color uploads preserve rectangles and mip/layer
 * subresources. Multi-layer regions are split into native single-layer
 * passes without changing source layout or copying index/pixel data. 8-bit
 * multi-planar YCbCr uploads use the same per-plane R8/RG8 render path. */
bool hk_g17p_buffer_to_image_copy_supported(
   const struct hk_g17p_buffer_to_image_copy *copy);

/* Resolve one block-compressed buffer upload into source strides and AIL
 * layer/slice coordinates. This is shared by BC, ETC2/EAC and ASTC LDR: only
 * their Vulkan block geometry differs. The returned byte span reaches the
 * final source byte read and therefore covers padding gaps between rows and
 * slices. */
bool hk_g17p_compressed_buffer_to_image_copy_plan(
   const struct hk_g17p_buffer_to_image_copy *copy,
   VkDeviceSize buffer_offset, VkDeviceSize buffer_size,
   struct hk_g17p_compressed_copy_plan *plan);

/* Resolve the same block geometry for a raw compressed image readback. */
bool hk_g17p_compressed_image_to_buffer_copy_plan(
   const struct hk_g17p_buffer_to_image_copy *copy,
   VkDeviceSize buffer_offset, VkDeviceSize buffer_size,
   struct hk_g17p_compressed_copy_plan *plan);

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
