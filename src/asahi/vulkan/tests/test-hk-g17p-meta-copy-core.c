/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_g17p_meta_copy_core.h"

#include <stdio.h>

#define CHECK(c)                                                              \
   do {                                                                       \
      if (!(c)) {                                                             \
         fprintf(stderr, "check failed at line %u: %s\n", __LINE__, #c);     \
         return 1;                                                            \
      }                                                                       \
   } while (0)

int
main(void)
{
   const struct hk_g17p_buffer_to_image_copy vkcube = {
      .chip = AGX_CHIP_G17P,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .image_type = VK_IMAGE_TYPE_2D,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .plane_count = 1,
      .mip_levels = 1,
      .array_layers = 1,
      .image_extent = {256, 256, 1},
      .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
      .layer_count = 1,
      .copy_extent = {256, 256, 1},
   };
   CHECK(hk_g17p_buffer_to_image_copy_supported(&vkcube));
   struct hk_g17p_buffer_to_image_copy explicit_stride = vkcube;
   explicit_stride.buffer_row_length = explicit_stride.copy_extent.width;
   explicit_stride.buffer_image_height = explicit_stride.copy_extent.height;
   CHECK(hk_g17p_buffer_to_image_copy_supported(&explicit_stride));

   explicit_stride.buffer_row_length += 8;
   explicit_stride.buffer_image_height += 4;
   CHECK(hk_g17p_buffer_to_image_copy_supported(&explicit_stride));

   struct hk_g17p_buffer_to_image_copy bad = vkcube;
   bad.format = VK_FORMAT_B8G8R8A8_UNORM;
   CHECK(hk_g17p_buffer_to_image_copy_supported(&bad));
   bad.format = VK_FORMAT_R8_UNORM;
   CHECK(hk_g17p_buffer_to_image_copy_supported(&bad));
   bad.format = VK_FORMAT_R16G16B16A16_SFLOAT;
   CHECK(!hk_g17p_buffer_to_image_copy_supported(&bad));
   bad = vkcube;
   bad.copy_extent.width--;
   CHECK(hk_g17p_buffer_to_image_copy_supported(&bad));
   bad = vkcube;
   bad.layer_count = 2;
   CHECK(!hk_g17p_buffer_to_image_copy_supported(&bad));
   bad = vkcube;
   bad.buffer_row_length = 255;
   CHECK(!hk_g17p_buffer_to_image_copy_supported(&bad));
   bad = vkcube;
   bad.chip = AGX_CHIP_G17S;
   CHECK(!hk_g17p_buffer_to_image_copy_supported(&bad));
   bad = vkcube;
   bad.mip_levels = 2;
   CHECK(hk_g17p_buffer_to_image_copy_supported(&bad));
   bad = vkcube;
   bad.image_offset.x = 1;
   CHECK(!hk_g17p_buffer_to_image_copy_supported(&bad));
   bad = vkcube;
   bad.layout = VK_IMAGE_LAYOUT_GENERAL;
   CHECK(hk_g17p_buffer_to_image_copy_supported(&bad));
   bad.format = VK_FORMAT_A2B10G10R10_UNORM_PACK32;
   CHECK(hk_g17p_buffer_to_image_copy_supported(&bad));
   bad.layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
   CHECK(hk_g17p_buffer_to_image_copy_supported(&bad));
   bad.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
   CHECK(!hk_g17p_buffer_to_image_copy_supported(&bad));


   /* Atlas uploads must preserve arbitrary offsets and padded source rows. */
   struct hk_g17p_buffer_to_image_copy atlas = vkcube;
   atlas.image_offset = (VkOffset3D){11, 13, 0};
   atlas.copy_extent = (VkExtent3D){37, 29, 1};
   atlas.buffer_row_length = 80;
   atlas.buffer_image_height = 72;
   CHECK(hk_g17p_buffer_to_image_copy_supported(&atlas));
   unsigned tested = 0;
   for (unsigned mip = 0; mip < 9; ++mip) {
      struct hk_g17p_buffer_to_image_copy level = atlas;
      level.mip_levels = 9;
      level.mip_level = mip;
      const unsigned dim = 256 >> mip;
      level.image_offset = (VkOffset3D){dim - 1, dim - 1, 0};
      level.copy_extent = (VkExtent3D){1, 1, 1};
      for (unsigned layers = 1; layers <= 16; ++layers) {
         level.array_layers = layers;
         for (unsigned first = 0; first < layers; ++first) {
            level.base_array_layer = first;
            level.layer_count = layers - first;
            CHECK(hk_g17p_buffer_to_image_copy_supported(&level));
            level.layer_count = VK_REMAINING_ARRAY_LAYERS;
            CHECK(hk_g17p_buffer_to_image_copy_supported(&level));
            level.layer_count = layers - first + 1;
            CHECK(!hk_g17p_buffer_to_image_copy_supported(&level));
            tested += 3;
         }
      }
      level.layer_count = 1;
      level.copy_extent.width = 2;
      CHECK(!hk_g17p_buffer_to_image_copy_supported(&level));
   }
   bad = atlas;
   bad.image_offset.x = -1;
   CHECK(!hk_g17p_buffer_to_image_copy_supported(&bad));
   bad = atlas;
   bad.image_offset.x = INT32_MAX;
   bad.copy_extent.width = UINT32_MAX;
   CHECK(!hk_g17p_buffer_to_image_copy_supported(&bad));
   bad = atlas;
   bad.mip_level = 32;
   bad.mip_levels = 33;
   CHECK(!hk_g17p_buffer_to_image_copy_supported(&bad));
   bad = atlas;
   bad.array_layers = UINT32_MAX;
   bad.base_array_layer = UINT32_MAX - 1;
   bad.layer_count = 2;
   CHECK(!hk_g17p_buffer_to_image_copy_supported(&bad));

   const VkImageCopy2 layered = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_COPY_2,
      .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 1, 2, 3},
      .srcOffset = {7, 5, 0},
      .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 2, 5, 3},
      .dstOffset = {11, 13, 0},
      .extent = {37, 29, 1},
   };
   for (uint32_t slice = 0; slice < 3; ++slice) {
      VkImageCopy2 one;
      CHECK(hk_g17p_image_copy_slice(
         AGX_CHIP_G17P, VK_IMAGE_TYPE_2D, VK_IMAGE_TYPE_2D, 3, 3,
         &layered, slice, &one));
      CHECK(one.srcSubresource.baseArrayLayer == 2 + slice);
      CHECK(one.dstSubresource.baseArrayLayer == 5 + slice);
      CHECK(one.srcSubresource.layerCount == 1);
      CHECK(one.dstSubresource.layerCount == 1);
      CHECK(one.extent.width == 37 && one.extent.height == 29 &&
            one.extent.depth == 1);
      CHECK(one.srcOffset.x == 7 && one.srcOffset.y == 5 &&
            one.dstOffset.x == 11 && one.dstOffset.y == 13);
   }

   VkImageCopy2 untouched = {.sType = VK_STRUCTURE_TYPE_IMAGE_COPY_2};
   CHECK(!hk_g17p_image_copy_slice(
      AGX_CHIP_G17G, VK_IMAGE_TYPE_2D, VK_IMAGE_TYPE_2D, 3, 3,
      &layered, 0, &untouched));
   CHECK(untouched.sType == 0);
   CHECK(!hk_g17p_image_copy_slice(
      AGX_CHIP_G17P, VK_IMAGE_TYPE_2D, VK_IMAGE_TYPE_2D, 3, 2,
      &layered, 0, &untouched));
   CHECK(!hk_g17p_image_copy_slice(
      AGX_CHIP_G17P, VK_IMAGE_TYPE_2D, VK_IMAGE_TYPE_2D, 3, 3,
      &layered, 3, &untouched));
   VkImageCopy2 overflow = layered;
   overflow.srcSubresource.baseArrayLayer = UINT32_MAX;
   CHECK(!hk_g17p_image_copy_slice(
      AGX_CHIP_G17P, VK_IMAGE_TYPE_2D, VK_IMAGE_TYPE_2D, 3, 3,
      &overflow, 1, &untouched));
   CHECK(!hk_g17p_image_copy_slice(
      AGX_CHIP_G17P, VK_IMAGE_TYPE_3D, VK_IMAGE_TYPE_3D, 3, 3,
      &layered, 0, &untouched));

   const VkImageBlit2 blit = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2,
      .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 1, 2, 3},
      .srcOffsets = {{7, 5, 0}, {26, 18, 1}},
      .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 2, 5, 3},
      .dstOffsets = {{11, 13, 0}, {48, 42, 1}},
   };
   for (uint32_t layer = 0; layer < 3; ++layer) {
      VkImageBlit2 one;
      CHECK(hk_g17p_image_blit_slice(
         AGX_CHIP_G17P, VK_IMAGE_TYPE_2D, VK_IMAGE_TYPE_2D, 3, 3,
         &blit, layer, &one));
      CHECK(one.srcSubresource.baseArrayLayer == 2 + layer);
      CHECK(one.dstSubresource.baseArrayLayer == 5 + layer);
      CHECK(one.srcSubresource.layerCount == 1);
      CHECK(one.dstSubresource.layerCount == 1);
      CHECK(one.srcOffsets[0].x == 7 && one.srcOffsets[1].x == 26);
      CHECK(one.dstOffsets[0].y == 13 && one.dstOffsets[1].y == 42);
   }
   VkImageBlit2 bad_blit = blit;
   bad_blit.dstOffsets[1].z = 2;
   VkImageBlit2 no_blit = {.sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2};
   CHECK(!hk_g17p_image_blit_slice(
      AGX_CHIP_G17P, VK_IMAGE_TYPE_2D, VK_IMAGE_TYPE_2D, 3, 3,
      &bad_blit, 0, &no_blit));
   CHECK(no_blit.sType == 0);
   CHECK(!hk_g17p_image_blit_slice(
      AGX_CHIP_G17P, VK_IMAGE_TYPE_2D, VK_IMAGE_TYPE_2D, 3, 2,
      &blit, 0, &no_blit));
   CHECK(!hk_g17p_image_blit_slice(
      AGX_CHIP_G17P, VK_IMAGE_TYPE_3D, VK_IMAGE_TYPE_3D, 3, 3,
      &blit, 0, &no_blit));
   printf("Honeykrisp G17P meta copy region/mip/layer checks passed (%u exhaustive layer cases)\n", tested);
   return 0;
}
