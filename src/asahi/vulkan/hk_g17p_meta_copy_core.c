/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_g17p_meta_copy_core.h"

bool
hk_g17p_buffer_to_image_copy_supported(
   const struct hk_g17p_buffer_to_image_copy *copy)
{
   if (!copy || copy->chip != AGX_CHIP_G17P ||
       (copy->format != VK_FORMAT_R8_UNORM &&
        copy->format != VK_FORMAT_B8G8R8A8_UNORM &&
        copy->format != VK_FORMAT_R8G8B8A8_UNORM &&
        copy->format != VK_FORMAT_A2B10G10R10_UNORM_PACK32) ||
       (copy->layout != VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
        copy->layout != VK_IMAGE_LAYOUT_GENERAL) ||
       copy->image_type != VK_IMAGE_TYPE_2D ||
       copy->samples != VK_SAMPLE_COUNT_1_BIT || copy->plane_count != 1 ||
       copy->mip_levels == 0 || copy->array_layers == 0 ||
       copy->aspect != VK_IMAGE_ASPECT_COLOR_BIT ||
       copy->mip_level >= copy->mip_levels || copy->mip_level >= 32 ||
       copy->base_array_layer >= copy->array_layers ||
       copy->layer_count == 0 ||
       (copy->layer_count != VK_REMAINING_ARRAY_LAYERS &&
        copy->layer_count > copy->array_layers - copy->base_array_layer) ||
       copy->image_offset.x < 0 || copy->image_offset.y < 0 ||
       copy->image_offset.z != 0 || copy->image_extent.depth != 1 ||
       copy->copy_extent.depth != 1 ||
       copy->image_extent.width == 0 || copy->image_extent.height == 0 ||
       copy->copy_extent.width == 0 || copy->copy_extent.height == 0)
      return false;

   const uint32_t width = copy->image_extent.width >> copy->mip_level;
   const uint32_t height = copy->image_extent.height >> copy->mip_level;
   const uint32_t mip_width = width ? width : 1;
   const uint32_t mip_height = height ? height : 1;
   const uint32_t x = copy->image_offset.x, y = copy->image_offset.y;
   if (x >= mip_width || y >= mip_height ||
       copy->copy_extent.width > mip_width - x ||
       copy->copy_extent.height > mip_height - y)
      return false;

   return (copy->buffer_row_length == 0 ||
           copy->buffer_row_length >= copy->copy_extent.width) &&
          (copy->buffer_image_height == 0 ||
           copy->buffer_image_height >= copy->copy_extent.height);
}

bool
hk_g17p_image_copy_slice(enum agx_chip chip, VkImageType src_type,
                         VkImageType dst_type, uint32_t src_layers,
                         uint32_t dst_layers, const VkImageCopy2 *region,
                         uint32_t slice, VkImageCopy2 *out)
{
   if (out)
      *out = (VkImageCopy2){0};

   if (chip != AGX_CHIP_G17P || !region || !out ||
       src_type != VK_IMAGE_TYPE_2D || dst_type != VK_IMAGE_TYPE_2D ||
       src_layers == 0 || src_layers != dst_layers || slice >= src_layers ||
       region->extent.depth != 1 ||
       slice > UINT32_MAX - region->srcSubresource.baseArrayLayer ||
       slice > UINT32_MAX - region->dstSubresource.baseArrayLayer)
      return false;

   *out = *region;
   out->extent.depth = 1;
   out->srcSubresource.baseArrayLayer += slice;
   out->srcSubresource.layerCount = 1;
   out->dstSubresource.baseArrayLayer += slice;
   out->dstSubresource.layerCount = 1;
   return true;
}

bool
hk_g17p_image_blit_slice(enum agx_chip chip, VkImageType src_type,
                         VkImageType dst_type, uint32_t src_layers,
                         uint32_t dst_layers, const VkImageBlit2 *region,
                         uint32_t slice, VkImageBlit2 *out)
{
   if (out)
      *out = (VkImageBlit2){0};

   if (chip != AGX_CHIP_G17P || !region || !out ||
       src_type != VK_IMAGE_TYPE_2D || dst_type != VK_IMAGE_TYPE_2D ||
       src_layers == 0 || src_layers != dst_layers || slice >= src_layers ||
       region->srcOffsets[0].z != 0 || region->srcOffsets[1].z != 1 ||
       region->dstOffsets[0].z != 0 || region->dstOffsets[1].z != 1 ||
       slice > UINT32_MAX - region->srcSubresource.baseArrayLayer ||
       slice > UINT32_MAX - region->dstSubresource.baseArrayLayer)
      return false;

   *out = *region;
   out->srcSubresource.baseArrayLayer += slice;
   out->srcSubresource.layerCount = 1;
   out->dstSubresource.baseArrayLayer += slice;
   out->dstSubresource.layerCount = 1;
   return true;
}
