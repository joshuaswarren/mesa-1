/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_g17p_meta_copy_core.h"
#include "vk_format.h"

bool
hk_g17p_transfer_dst_layout_supported(VkImageLayout layout)
{
   return layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL ||
          layout == VK_IMAGE_LAYOUT_GENERAL ||
          layout == VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR;
}

bool
hk_g17p_bc_format_supported(VkFormat format)
{
   return hk_g17p_block_format_family(format) == HK_G17P_BLOCK_FORMAT_BC;
}

enum hk_g17p_block_format_family
hk_g17p_block_format_family(VkFormat format)
{
   if (format >= VK_FORMAT_BC1_RGB_UNORM_BLOCK &&
       format <= VK_FORMAT_BC7_SRGB_BLOCK)
      return HK_G17P_BLOCK_FORMAT_BC;

   if (format >= VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK &&
       format <= VK_FORMAT_EAC_R11G11_SNORM_BLOCK)
      return HK_G17P_BLOCK_FORMAT_ETC2_EAC;

   if (format >= VK_FORMAT_ASTC_4x4_UNORM_BLOCK &&
       format <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK)
      return HK_G17P_BLOCK_FORMAT_ASTC_LDR;

   return HK_G17P_BLOCK_FORMAT_NONE;
}

bool
hk_g17p_buffer_to_image_copy_supported(
   const struct hk_g17p_buffer_to_image_copy *copy)
{
   if (!copy || copy->chip != AGX_CHIP_G17P ||
       !hk_g17p_transfer_dst_layout_supported(copy->layout) ||
       copy->image_type != VK_IMAGE_TYPE_2D ||
       copy->samples != VK_SAMPLE_COUNT_1_BIT ||
       copy->mip_levels == 0 || copy->array_layers == 0 ||
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

   VkFormat plane_format = copy->format;
   uint32_t width_scale = 1, height_scale = 1;
   const struct vk_format_ycbcr_info *ycbcr =
      vk_format_get_ycbcr_info(copy->format);

   if (ycbcr && ycbcr->n_planes > 1) {
      uint32_t plane;
      switch (copy->aspect) {
      case VK_IMAGE_ASPECT_PLANE_0_BIT:
         plane = 0;
         break;
      case VK_IMAGE_ASPECT_PLANE_1_BIT:
         plane = 1;
         break;
      case VK_IMAGE_ASPECT_PLANE_2_BIT:
         plane = 2;
         break;
      default:
         return false;
      }

      if (copy->plane_count != ycbcr->n_planes || plane >= ycbcr->n_planes ||
          copy->mip_levels != 1)
         return false;

      plane_format = ycbcr->planes[plane].format;
      width_scale = ycbcr->planes[plane].denominator_scales[0];
      height_scale = ycbcr->planes[plane].denominator_scales[1];
      if (!width_scale || !height_scale ||
          copy->image_extent.width % width_scale ||
          copy->image_extent.height % height_scale)
         return false;
   } else if (copy->plane_count != 1 ||
              copy->aspect != VK_IMAGE_ASPECT_COLOR_BIT) {
      return false;
   }

   switch (plane_format) {
   case VK_FORMAT_R8_UNORM:
   case VK_FORMAT_R8G8_UNORM:
   case VK_FORMAT_B8G8R8A8_UNORM:
   case VK_FORMAT_R8G8B8A8_UNORM:
   case VK_FORMAT_R8G8B8A8_UINT:
   case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
      break;
   default:
      return false;
   }

   const uint32_t width =
      (copy->image_extent.width / width_scale) >> copy->mip_level;
   const uint32_t height =
      (copy->image_extent.height / height_scale) >> copy->mip_level;
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
