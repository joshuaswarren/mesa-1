/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_g17p_meta_copy_core.h"
#include "util/compiler.h"
#include "vk_format.h"

#include <limits.h>
#include <string.h>

bool
hk_g17p_transfer_dst_layout_supported(VkImageLayout layout)
{
   return layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL ||
          layout == VK_IMAGE_LAYOUT_GENERAL ||
          layout == VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR;
}

bool
hk_g17p_transfer_src_layout_supported(VkImageLayout layout)
{
   return layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL ||
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

static bool
hk_u64_mul(uint64_t a, uint64_t b, uint64_t *out)
{
   if (a && b > UINT64_MAX / a)
      return false;

   *out = a * b;
   return true;
}

static bool
hk_u64_add(uint64_t a, uint64_t b, uint64_t *out)
{
   if (b > UINT64_MAX - a)
      return false;

   *out = a + b;
   return true;
}

static uint32_t
hk_minify(uint32_t value, uint32_t level)
{
   const uint32_t minified = value >> level;
   return minified ? minified : 1;
}

static bool
hk_g17p_compressed_copy_plan(
   const struct hk_g17p_buffer_to_image_copy *copy,
   VkDeviceSize buffer_offset, VkDeviceSize buffer_size,
   bool image_is_source, struct hk_g17p_compressed_copy_plan *plan)
{
   if (plan)
      memset(plan, 0, sizeof(*plan));

   if (!copy || !plan || copy->chip != AGX_CHIP_G17P ||
       hk_g17p_block_format_family(copy->format) ==
          HK_G17P_BLOCK_FORMAT_NONE ||
       !(image_is_source
            ? hk_g17p_transfer_src_layout_supported(copy->layout)
            : hk_g17p_transfer_dst_layout_supported(copy->layout)) ||
       copy->samples != VK_SAMPLE_COUNT_1_BIT || copy->plane_count != 1 ||
       copy->aspect != VK_IMAGE_ASPECT_COLOR_BIT ||
       copy->mip_levels == 0 || copy->mip_level >= copy->mip_levels ||
       copy->mip_level >= 32 || copy->array_layers == 0 ||
       copy->image_extent.width == 0 || copy->image_extent.height == 0 ||
       copy->image_extent.depth == 0 || copy->copy_extent.width == 0 ||
       copy->copy_extent.height == 0 || copy->copy_extent.depth == 0 ||
       copy->image_offset.x < 0 || copy->image_offset.y < 0 ||
       copy->image_offset.z < 0)
      return false;

   const uint32_t block_w = vk_format_get_blockwidth(copy->format);
   const uint32_t block_h = vk_format_get_blockheight(copy->format);
   const uint32_t block_B = vk_format_get_blocksize(copy->format);
   if (!block_w || !block_h || (block_B != 8 && block_B != 16))
      return false;

   const uint32_t mip_w =
      hk_minify(copy->image_extent.width, copy->mip_level);
   const uint32_t mip_h =
      hk_minify(copy->image_extent.height, copy->mip_level);
   const uint32_t mip_d =
      hk_minify(copy->image_extent.depth, copy->mip_level);
   const uint32_t x = copy->image_offset.x;
   const uint32_t y = copy->image_offset.y;
   const uint32_t z = copy->image_offset.z;
   const uint32_t width = copy->copy_extent.width;
   const uint32_t height = copy->copy_extent.height;

   uint32_t first_slice, slice_count;
   switch (copy->image_type) {
   case VK_IMAGE_TYPE_1D:
      if (copy->image_extent.height != 1 || copy->image_extent.depth != 1 ||
          y != 0 || z != 0 || height != 1 || copy->copy_extent.depth != 1)
         return false;
      FALLTHROUGH;
   case VK_IMAGE_TYPE_2D:
      if (copy->image_extent.depth != 1 || z != 0 ||
          copy->copy_extent.depth != 1 ||
          copy->base_array_layer >= copy->array_layers ||
          copy->layer_count == 0)
         return false;

      first_slice = copy->base_array_layer;
      slice_count = copy->layer_count == VK_REMAINING_ARRAY_LAYERS
                       ? copy->array_layers - copy->base_array_layer
                       : copy->layer_count;
      if (slice_count > copy->array_layers - copy->base_array_layer)
         return false;
      break;

   case VK_IMAGE_TYPE_3D:
      if (copy->array_layers != 1 || copy->base_array_layer != 0 ||
          copy->layer_count != 1 || z >= mip_d ||
          copy->copy_extent.depth > mip_d - z)
         return false;

      first_slice = z;
      slice_count = copy->copy_extent.depth;
      break;

   default:
      return false;
   }

   if (x >= mip_w || y >= mip_h || width > mip_w - x ||
       height > mip_h - y || (x % block_w) || (y % block_h) ||
       ((width % block_w) && x + width != mip_w) ||
       ((height % block_h) && y + height != mip_h) ||
       (copy->buffer_row_length &&
        (copy->buffer_row_length < width ||
         copy->buffer_row_length % block_w)) ||
       (copy->buffer_image_height &&
        (copy->buffer_image_height < height ||
         copy->buffer_image_height % block_h)) ||
       buffer_offset % block_B)
      return false;

   const uint32_t row_px =
      copy->buffer_row_length ? copy->buffer_row_length : width;
   const uint32_t image_h_px =
      copy->buffer_image_height ? copy->buffer_image_height : height;
   const uint32_t blocks_x =
      ((uint64_t)width + block_w - 1) / block_w;
   const uint32_t blocks_y =
      ((uint64_t)height + block_h - 1) / block_h;
   const uint64_t buffer_row_blocks =
      ((uint64_t)row_px + block_w - 1) / block_w;
   const uint64_t buffer_image_rows =
      ((uint64_t)image_h_px + block_h - 1) / block_h;
   uint64_t buffer_row_B, buffer_slice_B;
   if (!hk_u64_mul(buffer_row_blocks, block_B, &buffer_row_B) ||
       buffer_row_B > INT32_MAX ||
       !hk_u64_mul(buffer_image_rows, buffer_row_B, &buffer_slice_B))
      return false;

   uint64_t span = 0, term;
   if (!hk_u64_mul(slice_count - 1, buffer_slice_B, &span) ||
       !hk_u64_mul(blocks_y - 1, buffer_row_B, &term) ||
       !hk_u64_add(span, term, &span) ||
       !hk_u64_mul(blocks_x, block_B, &term) ||
       !hk_u64_add(span, term, &span) || buffer_offset > buffer_size ||
       span > buffer_size - buffer_offset)
      return false;

   *plan = (struct hk_g17p_compressed_copy_plan){
      .block_width = block_w,
      .block_height = block_h,
      .block_size_B = block_B,
      .base_x_el = x / block_w,
      .base_y_el = y / block_h,
      .blocks_x = blocks_x,
      .blocks_y = blocks_y,
      .first_slice = first_slice,
      .slice_count = slice_count,
      .buffer_row_stride_B = buffer_row_B,
      .buffer_slice_stride_B = buffer_slice_B,
      .buffer_span_B = span,
   };
   return true;
}

bool
hk_g17p_compressed_buffer_to_image_copy_plan(
   const struct hk_g17p_buffer_to_image_copy *copy,
   VkDeviceSize buffer_offset, VkDeviceSize buffer_size,
   struct hk_g17p_compressed_copy_plan *plan)
{
   return hk_g17p_compressed_copy_plan(copy, buffer_offset, buffer_size,
                                       false, plan);
}

bool
hk_g17p_compressed_image_to_buffer_copy_plan(
   const struct hk_g17p_buffer_to_image_copy *copy,
   VkDeviceSize buffer_offset, VkDeviceSize buffer_size,
   struct hk_g17p_compressed_copy_plan *plan)
{
   return hk_g17p_compressed_copy_plan(copy, buffer_offset, buffer_size,
                                       true, plan);
}

bool
hk_g17p_buffer_to_image_copy_supported(
   const struct hk_g17p_buffer_to_image_copy *copy)
{
   const bool volume = copy && copy->image_type == VK_IMAGE_TYPE_3D;
   if (!copy || copy->chip != AGX_CHIP_G17P ||
       !hk_g17p_transfer_dst_layout_supported(copy->layout) ||
       (copy->image_type != VK_IMAGE_TYPE_2D && !volume) ||
       copy->samples != VK_SAMPLE_COUNT_1_BIT ||
       copy->mip_levels == 0 || copy->array_layers == 0 ||
       copy->mip_level >= copy->mip_levels || copy->mip_level >= 32 ||
       copy->base_array_layer >= copy->array_layers ||
       copy->layer_count == 0 ||
       (copy->layer_count != VK_REMAINING_ARRAY_LAYERS &&
        copy->layer_count > copy->array_layers - copy->base_array_layer) ||
       (volume && (copy->array_layers != 1 || copy->base_array_layer != 0 ||
                   copy->layer_count != 1)) ||
       copy->image_offset.x < 0 || copy->image_offset.y < 0 ||
       (volume ? copy->image_offset.z < 0 : copy->image_offset.z != 0) ||
       (!volume && copy->image_extent.depth != 1) ||
       (!volume && copy->copy_extent.depth != 1) ||
       copy->image_extent.width == 0 || copy->image_extent.height == 0 ||
       copy->image_extent.depth == 0 || copy->copy_extent.width == 0 ||
       copy->copy_extent.height == 0 || copy->copy_extent.depth == 0 ||
       (volume && copy->copy_extent.depth - 1 >
                     (uint32_t)(INT32_MAX - copy->image_offset.z)))
      return false;

   VkFormat plane_format = copy->format;
   uint32_t width_scale = 1, height_scale = 1;
   const struct vk_format_ycbcr_info *ycbcr =
      vk_format_get_ycbcr_info(copy->format);

   if (ycbcr && ycbcr->n_planes > 1) {
      if (volume)
         return false;
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

   /* A hardcoded allowlist here refused copies the driver advertises as
    * supported, and the consequence is far worse than one lost image: the
    * refusal fails vkEndCommandBuffer, so Zink skips vkQueueSubmit for the
    * WHOLE frame and latches device_lost, after which nothing that process
    * draws ever reaches the GPU again -- with no GL error at any point.
    * SuperTuxKart hangs that way, and vkQuake's black world was one refused
    * R32_UINT lightmap upload invalidating the staging batch it shares with
    * every other texture.
    *
    * What the generic vk_meta path below actually needs is colour-attachment
    * support, which is exactly ail_pixel_format[].renderable -- the same bit
    * hk_get_format_features() reports as COLOR_ATTACHMENT. Ask that instead.
    * Formats that are not renderable still fail closed and fall back to the
    * ordinary copy path. */
   if (!copy->plane_format_renderable)
      return false;
   /* Multi-plane destinations select their plane format above; a renderable
    * plane is still required for each one. */
   (void)plane_format;

   const uint32_t width =
      (copy->image_extent.width / width_scale) >> copy->mip_level;
   const uint32_t height =
      (copy->image_extent.height / height_scale) >> copy->mip_level;
   const uint32_t depth = copy->image_extent.depth >> copy->mip_level;
   const uint32_t mip_width = width ? width : 1;
   const uint32_t mip_height = height ? height : 1;
   const uint32_t mip_depth = depth ? depth : 1;
   const uint32_t x = copy->image_offset.x, y = copy->image_offset.y,
                  z = copy->image_offset.z;
   if (x >= mip_width || y >= mip_height ||
       copy->copy_extent.width > mip_width - x ||
       copy->copy_extent.height > mip_height - y ||
       (volume &&
        (z >= mip_depth || copy->copy_extent.depth > mip_depth - z)))
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

   const bool src_3d = src_type == VK_IMAGE_TYPE_3D;
   const bool dst_3d = dst_type == VK_IMAGE_TYPE_3D;
   const bool valid_types =
      (src_type == VK_IMAGE_TYPE_2D || src_3d) &&
      (dst_type == VK_IMAGE_TYPE_2D || dst_3d);
   const bool same_type = src_type == dst_type;
   const bool two_d_three_d = (src_type == VK_IMAGE_TYPE_2D && dst_3d) ||
                              (dst_type == VK_IMAGE_TYPE_2D && src_3d);
   if (chip != AGX_CHIP_G17P || !region || !out || !valid_types ||
       (!same_type && !two_d_three_d) || src_layers == 0 ||
       src_layers != dst_layers || slice >= src_layers ||
       ((src_3d || dst_3d) && region->extent.depth != src_layers) ||
       (!src_3d && !dst_3d && region->extent.depth != 1) ||
       (!src_3d && region->srcOffset.z != 0) ||
       (!dst_3d && region->dstOffset.z != 0) ||
       (src_3d &&
        (region->srcOffset.z < 0 ||
         slice > (uint32_t)(INT32_MAX - region->srcOffset.z))) ||
       (!src_3d &&
        slice > UINT32_MAX - region->srcSubresource.baseArrayLayer) ||
       (dst_3d &&
        (region->dstOffset.z < 0 ||
         slice > (uint32_t)(INT32_MAX - region->dstOffset.z))) ||
       (!dst_3d &&
        slice > UINT32_MAX - region->dstSubresource.baseArrayLayer))
      return false;

   *out = *region;
   out->extent.depth = 1;
   if (src_3d)
      out->srcOffset.z += slice;
   else
      out->srcSubresource.baseArrayLayer += slice;
   out->srcSubresource.layerCount = 1;
   if (dst_3d)
      out->dstOffset.z += slice;
   else
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
