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
   /* Whether the destination plane's format is renderable, i.e. whether the
    * driver advertises COLOR_ATTACHMENT for it. The path below the gate is
    * generic vk_meta, which renders the staged texels through a native EOT
    * view, so that -- not membership of a hardcoded list -- is what decides
    * whether the copy can be serviced. The caller supplies it because the
    * answer lives in ail_pixel_format[], which needs the device. */
   bool plane_format_renderable;
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

/* A raw linear-buffer to AIL-tiled upload.  This is intentionally narrower
 * than the generic render-based copy path: it covers the byte-exact R8 and
 * four-channel 8-bit texture shapes used by STK without interpreting their
 * colour space or channel order. */
struct hk_g17p_uncompressed_copy_plan {
   uint32_t block_size_B;
   uint32_t base_x_el;
   uint32_t base_y_el;
   uint32_t width_el;
   uint32_t height_el;
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

/* Native uncompressed color uploads preserve rectangles and mip/layer/Z
 * subresources. Array and 3D regions are split into native single-slice
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

/* Resolve an R8/RGBA8/BGRA8 upload into one raw compute copy per 2D array
 * layer.  The returned byte span includes row and layer padding and is checked
 * against the source buffer before any command is emitted. */
bool hk_g17p_uncompressed_buffer_to_image_copy_plan(
   const struct hk_g17p_buffer_to_image_copy *copy,
   VkDeviceSize buffer_offset, VkDeviceSize buffer_size,
   struct hk_g17p_uncompressed_copy_plan *plan);

/* HAL200's qualified tile programs are single-layer.  Convert one logical
 * slice of a valid 2D array, 3D volume, or 2D-array<->3D image copy without
 * changing its rectangle or mip levels.  The resolved source/destination
 * slice counts must describe the same copy. */
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
