/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_g17p_meta_copy_core.h"
#include "vulkan/runtime/vk_meta.h"

#include <stdio.h>
#include <stdlib.h>

#define CHECK(c)                                                              \
   do {                                                                       \
      if (!(c)) {                                                             \
         fprintf(stderr, "check failed at line %u: %s\n", __LINE__, #c);     \
         return 1;                                                            \
      }                                                                       \
   } while (0)

static uint8_t
mip_source_byte(uint32_t x, uint32_t y, uint32_t channel)
{
   uint32_t v = x * 0x9e3779b1u ^ y * 0x85ebca6bu ^
                channel * 0xc2b2ae35u ^ 0x6d2b79f5u;
   v ^= v >> 16;
   v *= 0x7feb352du;
   v ^= v >> 15;
   return v >> 24;
}

static uint64_t
mip_oracle_hash(uint32_t src_width, uint32_t src_height,
                uint32_t dst_width, uint32_t dst_height)
{
   uint32_t *axis = malloc((dst_width + dst_height) * sizeof(*axis));
   if (!axis || !hk_g17p_build_mip_axis(src_width, dst_width, axis) ||
       !hk_g17p_build_mip_axis(src_height, dst_height, axis + dst_width)) {
      free(axis);
      return 0;
   }

   uint64_t hash = UINT64_C(14695981039346656037);
   for (uint32_t y = 0; y < dst_height; ++y) {
      const uint32_t ye = axis[dst_width + y];
      const uint32_t y0 = ye & 0xffffu;
      const uint8_t wy = (ye >> 16) & 0xffu;
      const uint32_t y1 = y0 + (wy != 0);

      for (uint32_t x = 0; x < dst_width; ++x) {
         const uint32_t xe = axis[x];
         const uint32_t x0 = xe & 0xffffu;
         const uint8_t wx = (xe >> 16) & 0xffu;
         const uint32_t x1 = x0 + (wx != 0);

         for (uint32_t channel = 0; channel < 4; ++channel) {
            const uint8_t out = hk_g17p_filter_unorm8(
               mip_source_byte(x0, y0, channel),
               mip_source_byte(x1, y0, channel),
               mip_source_byte(x0, y1, channel),
               mip_source_byte(x1, y1, channel), wx, wy);
            hash ^= out;
            hash *= UINT64_C(1099511628211);
         }
      }
   }

   free(axis);
   return hash;
}

static uint32_t
tile_space_bits(uint32_t x)
{
   uint32_t accum = 0;
   for (uint32_t i = 0; i < 14; ++i)
      accum |= (x & (1u << i)) << i;
   return accum;
}

static uint32_t
tile_space_bits_7(uint32_t x)
{
   uint32_t accum = 0;
   for (uint32_t i = 0; i < 7; ++i)
      accum |= (x & (1u << i)) << i;
   return accum;
}

static uint32_t
tile_offset_division(uint32_t x, uint32_t y, uint32_t stride_el,
                     uint32_t tile_width_el, uint32_t tile_height_el,
                     uint32_t block_size_B)
{
   const uint32_t tiles_per_row =
      (stride_el + tile_width_el - 1) / tile_width_el;
   const uint32_t tile_idx = (y / tile_height_el) * tiles_per_row +
                             (x / tile_width_el);
   const uint32_t morton = tile_space_bits(x & (tile_width_el - 1)) |
                           (tile_space_bits(y & (tile_height_el - 1)) << 1);
   return (tile_idx * tile_width_el * tile_height_el + morton) *
          block_size_B;
}

static uint32_t
tile_offset_shifted(uint32_t x, uint32_t y,
                    const struct hk_g17p_tile_address_config *config,
                    uint32_t block_size_B)
{
   const uint32_t tile_width_el = 1u << config->tile_width_log2;
   const uint32_t tile_height_el = 1u << config->tile_height_log2;
   const uint32_t tile_idx =
      (y >> config->tile_height_log2) * config->tiles_per_row +
      (x >> config->tile_width_log2);
   const uint32_t morton = tile_space_bits_7(x & (tile_width_el - 1)) |
                           (tile_space_bits_7(y & (tile_height_el - 1)) << 1);
   return (tile_idx << (config->tile_width_log2 +
                        config->tile_height_log2)) *
             block_size_B +
          morton * block_size_B;
}

int
main(void)
{
   const VkImageAspectFlags color = VK_IMAGE_ASPECT_COLOR_BIT;
   const VkImageAspectFlags depth = VK_IMAGE_ASPECT_DEPTH_BIT;
   const VkImageAspectFlags stencil = VK_IMAGE_ASPECT_STENCIL_BIT;
   const VkImageAspectFlags pairs[][2] = {
      {color, color}, {depth, depth}, {stencil, stencil},
      {depth | stencil, depth | stencil},
      {color, depth}, {depth, color}, {color, stencil}, {stencil, color},
   };
   for (unsigned src = 0; src < 128; ++src) {
      for (unsigned dst = 0; dst < 128; ++dst) {
         bool expected = false;
         for (unsigned i = 0; i < sizeof(pairs) / sizeof(pairs[0]); ++i)
            expected |= src == pairs[i][0] && dst == pairs[i][1];
         struct hk_g17p_copy_aspect_plan plan = {.count = 99};
         CHECK(hk_g17p_copy_aspect_plan(src, dst, &plan) == expected);
         if (!expected) {
            CHECK(plan.count == 0 && plan.src[0] == 0 && plan.dst[0] == 0);
         } else if (src == (depth | stencil)) {
            CHECK(plan.count == 2);
            CHECK(plan.src[0] == depth && plan.dst[0] == depth);
            CHECK(plan.src[1] == stencil && plan.dst[1] == stencil);
         } else {
            CHECK(plan.count == 1 && plan.src[0] == src && plan.dst[0] == dst);
         }
      }
   }
   CHECK(!hk_g17p_copy_aspect_plan(color, depth, NULL));

   VkImageCopy2 element_region = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_COPY_2,
      .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .srcOffset = {12, 3, 0},
      .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .dstOffset = {2, 1, 0},
      .extent = {4, 4, 1},
   };
   struct hk_g17p_image_copy_element_plan elements;
   CHECK(hk_g17p_image_copy_element_plan(
      VK_FORMAT_BC1_RGB_UNORM_BLOCK, VK_FORMAT_R16G16B16A16_UINT,
      &element_region, &elements));
   CHECK(elements.block_size_B == 8);
   CHECK(elements.src_x_el == 3 && elements.src_y_el == 0);
   CHECK(elements.dst_x_el == 2 && elements.dst_y_el == 1);
   CHECK(elements.width_el == 1 && elements.height_el == 1);

   element_region.srcOffset = (VkOffset3D){3, 5, 0};
   element_region.dstOffset = (VkOffset3D){7, 11, 0};
   element_region.extent = (VkExtent3D){13, 17, 1};
   CHECK(hk_g17p_image_copy_element_plan(
      VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_E5B9G9R9_UFLOAT_PACK32,
      &element_region, &elements));
   CHECK(elements.block_size_B == 4);
   CHECK(elements.src_x_el == 3 && elements.src_y_el == 5);
   CHECK(elements.dst_x_el == 7 && elements.dst_y_el == 11);
   CHECK(elements.width_el == 13 && elements.height_el == 17);

   elements.block_size_B = 19;
   CHECK(!hk_g17p_image_copy_element_plan(
      VK_FORMAT_R8_UINT, VK_FORMAT_R16_UINT, &element_region, &elements));
   CHECK(elements.block_size_B == 0);
   element_region.srcOffset.x = -1;
   CHECK(!hk_g17p_image_copy_element_plan(
      VK_FORMAT_R8_UINT, VK_FORMAT_R8_UINT, &element_region, &elements));
   element_region.srcOffset.x = 0;
   element_region.extent.width = 0;
   CHECK(!hk_g17p_image_copy_element_plan(
      VK_FORMAT_R8_UINT, VK_FORMAT_R8_UINT, &element_region, &elements));

   struct hk_g17p_tile_address_config tile_config;
   CHECK(!hk_g17p_tile_address_config_init(0, 64, 64, &tile_config));
   CHECK(!hk_g17p_tile_address_config_init(64, 0, 64, &tile_config));
   CHECK(!hk_g17p_tile_address_config_init(64, 48, 64, &tile_config));
   CHECK(!hk_g17p_tile_address_config_init(64, 256, 128, &tile_config));
   CHECK(!hk_g17p_tile_address_config_init(64, 256, 1, &tile_config));
   CHECK(!hk_g17p_tile_address_config_init(64, 64, 64, NULL));

   static const struct {
      uint32_t block_size_B;
      uint32_t width_el;
      uint32_t height_el;
   } tile_shapes[] = {
      {1, 128, 128}, {2, 128, 64}, {4, 64, 64}, {8, 64, 32},
      {16, 32, 32},  {32, 32, 16}, {64, 16, 16}, {128, 16, 8},
      {256, 8, 8},
   };
   uint64_t tile_cases = 0;
   for (unsigned i = 0; i < sizeof(tile_shapes) / sizeof(tile_shapes[0]); ++i) {
      const uint32_t w = tile_shapes[i].width_el;
      const uint32_t h = tile_shapes[i].height_el;
      const uint32_t strides[] = {1, w - 1, w, w + 1, 2 * w + 3};
      for (unsigned s = 0; s < sizeof(strides) / sizeof(strides[0]); ++s) {
         CHECK(hk_g17p_tile_address_config_init(strides[s], w, h,
                                                 &tile_config));
         for (uint32_t y = 0; y < 2 * h + 3; ++y) {
            for (uint32_t x = 0; x < 2 * w + 3; ++x) {
               CHECK(tile_offset_shifted(x, y, &tile_config,
                                         tile_shapes[i].block_size_B) ==
                     tile_offset_division(x, y, strides[s], w, h,
                                          tile_shapes[i].block_size_B));
               ++tile_cases;
            }
         }
      }
   }

   uint32_t axis[303];
   CHECK(!hk_g17p_build_mip_axis(0, 1, axis));
   CHECK(!hk_g17p_build_mip_axis(3, 2, axis));
   CHECK(!hk_g17p_build_mip_axis(65537, 32768, axis));
   CHECK(!hk_g17p_build_mip_axis(3, 1, NULL));
   CHECK(hk_g17p_build_mip_axis(165, 82, axis));
   CHECK(axis[0] == 0x00820000u);
   CHECK(axis[20] == 0x00c00028u);
   CHECK(axis[43] == 0x00080057u);
   CHECK(hk_g17p_build_mip_axis(251, 125, axis));
   /* Division would produce weight 106 here. vk_meta's pushed reciprocal and
    * fragment FFMA produce 107, which is the graphics path being replaced. */
   CHECK(axis[114] == 0x006b00e5u);
   CHECK(hk_g17p_build_mip_axis(607, 303, axis));
   CHECK(axis[122] == 0x00e800f4u);
   CHECK(axis[180] == 0x00190169u);
   CHECK(axis[302] == 0x0080025du);

   CHECK(hk_g17p_filter_unorm8(0, 0, 0, 2, 128, 128) == 1);
   CHECK(hk_g17p_filter_unorm8(126, 126, 126, 128, 128, 128) == 127);
   CHECK(hk_g17p_filter_unorm8(127, 127, 127, 129, 128, 128) == 128);
   CHECK(hk_g17p_filter_unorm8(128, 128, 128, 130, 128, 128) == 128);
   CHECK(hk_g17p_filter_unorm8(254, 254, 255, 255, 128, 128) == 254);

   /* These hashes were captured on Apple M5 Pro from both a native Metal
    * linear sample into RGBA8 storage and a colour-attachment render. The two
    * paths were byte-identical. The focused G17P Vulkan hardware oracle must
    * agree before this path is enabled in a deployed build. */
   static const struct {
      uint32_t sw, sh, dw, dh;
      uint64_t hash;
   } mip_oracles[] = {
      {64, 64, 32, 32, UINT64_C(0x757b41d55f5fefc5)},
      {165, 165, 82, 82, UINT64_C(0x7d11df552c632a11)},
      {256, 89, 128, 44, UINT64_C(0x1ca8d5bd3e1aaee5)},
      {64, 63, 32, 31, UINT64_C(0x47a8e73e11743e54)},
      {607, 476, 303, 238, UINT64_C(0xfa614cc5cdf07a92)},
      {50, 45, 25, 22, UINT64_C(0x2c95ab15195eead6)},
      {75, 75, 37, 37, UINT64_C(0x477eb04369d02df1)},
   };
   for (unsigned i = 0; i < sizeof(mip_oracles) / sizeof(mip_oracles[0]); ++i)
      CHECK(mip_oracle_hash(mip_oracles[i].sw, mip_oracles[i].sh,
                            mip_oracles[i].dw, mip_oracles[i].dh) ==
            mip_oracles[i].hash);

   static const VkFormat bc[] = {
      VK_FORMAT_BC1_RGB_UNORM_BLOCK,  VK_FORMAT_BC1_RGB_SRGB_BLOCK,
      VK_FORMAT_BC1_RGBA_UNORM_BLOCK, VK_FORMAT_BC1_RGBA_SRGB_BLOCK,
      VK_FORMAT_BC2_UNORM_BLOCK,      VK_FORMAT_BC2_SRGB_BLOCK,
      VK_FORMAT_BC3_UNORM_BLOCK,      VK_FORMAT_BC3_SRGB_BLOCK,
      VK_FORMAT_BC4_UNORM_BLOCK,      VK_FORMAT_BC4_SNORM_BLOCK,
      VK_FORMAT_BC5_UNORM_BLOCK,      VK_FORMAT_BC5_SNORM_BLOCK,
      VK_FORMAT_BC6H_UFLOAT_BLOCK,    VK_FORMAT_BC6H_SFLOAT_BLOCK,
      VK_FORMAT_BC7_UNORM_BLOCK,      VK_FORMAT_BC7_SRGB_BLOCK,
   };
   for (unsigned i = 0; i < sizeof(bc) / sizeof(bc[0]); ++i)
      CHECK(hk_g17p_bc_format_supported(bc[i]));

   CHECK(hk_g17p_block_format_family(VK_FORMAT_BC1_RGB_UNORM_BLOCK) ==
         HK_G17P_BLOCK_FORMAT_BC);
   for (VkFormat format = VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK;
        format <= VK_FORMAT_EAC_R11G11_SNORM_BLOCK; ++format) {
      CHECK(hk_g17p_block_format_family(format) ==
            HK_G17P_BLOCK_FORMAT_ETC2_EAC);
      CHECK(!hk_g17p_bc_format_supported(format));
   }
   for (VkFormat format = VK_FORMAT_ASTC_4x4_UNORM_BLOCK;
        format <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK; ++format) {
      CHECK(hk_g17p_block_format_family(format) ==
            HK_G17P_BLOCK_FORMAT_ASTC_LDR);
      CHECK(!hk_g17p_bc_format_supported(format));
   }
   CHECK(!hk_g17p_bc_format_supported(VK_FORMAT_R8G8B8A8_UNORM));
   CHECK(hk_g17p_block_format_family(VK_FORMAT_R8G8B8A8_UNORM) ==
         HK_G17P_BLOCK_FORMAT_NONE);

   const struct hk_g17p_buffer_to_image_copy volume = {
      .chip = AGX_CHIP_G17P,
      .plane_format_renderable = true,
      .format = VK_FORMAT_BC1_RGBA_UNORM_BLOCK,
      .layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .image_type = VK_IMAGE_TYPE_3D,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .plane_count = 1,
      .mip_levels = 4,
      .array_layers = 1,
      .image_extent = {16, 12, 8},
      .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
      .mip_level = 1,
      .layer_count = 1,
      .image_offset = {4, 0, 1},
      .copy_extent = {4, 6, 2},
      .buffer_row_length = 12,
      .buffer_image_height = 8,
   };
   struct hk_g17p_compressed_copy_plan plan;
   CHECK(hk_g17p_compressed_buffer_to_image_copy_plan(
      &volume, 64, 144, &plan));
   CHECK(plan.block_width == 4 && plan.block_height == 4 &&
         plan.block_size_B == 8);
   CHECK(plan.base_x_el == 1 && plan.base_y_el == 0 &&
         plan.blocks_x == 1 && plan.blocks_y == 2);
   CHECK(plan.first_slice == 1 && plan.slice_count == 2);
   CHECK(plan.buffer_row_stride_B == 24 &&
         plan.buffer_slice_stride_B == 48 && plan.buffer_span_B == 80);

   /* A partial block short of the level edge still copies the whole block,
    * matching the AGX2 compute copy. */
   struct hk_g17p_buffer_to_image_copy partial = volume;
   partial.image_offset.x = 0;
   partial.copy_extent.width = 3;
   partial.copy_extent.height = 5;
   CHECK(hk_g17p_compressed_buffer_to_image_copy_plan(
      &partial, 64, 144, &plan));
   CHECK(plan.base_x_el == 0 && plan.blocks_x == 1 && plan.blocks_y == 2);

   struct hk_g17p_buffer_to_image_copy readback = volume;
   readback.layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
   CHECK(!hk_g17p_compressed_buffer_to_image_copy_plan(
      &readback, 64, 144, &plan));
   CHECK(hk_g17p_compressed_image_to_buffer_copy_plan(
      &readback, 64, 144, &plan));
   CHECK(plan.base_x_el == 1 && plan.blocks_x == 1 && plan.blocks_y == 2);
   CHECK(plan.first_slice == 1 && plan.slice_count == 2);
   CHECK(plan.buffer_row_stride_B == 24 &&
         plan.buffer_slice_stride_B == 48 && plan.buffer_span_B == 80);
   CHECK(!hk_g17p_compressed_image_to_buffer_copy_plan(
      &readback, 64, 143, &plan));
   readback.layout = VK_IMAGE_LAYOUT_GENERAL;
   CHECK(hk_g17p_compressed_image_to_buffer_copy_plan(
      &readback, 64, 144, &plan));
   readback.layout = VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR;
   CHECK(hk_g17p_compressed_image_to_buffer_copy_plan(
      &readback, 64, 144, &plan));
   readback.layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
   CHECK(!hk_g17p_compressed_image_to_buffer_copy_plan(
      &readback, 64, 144, &plan));

   struct hk_g17p_buffer_to_image_copy etc2 = volume;
   etc2.format = VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK;
   CHECK(hk_g17p_compressed_buffer_to_image_copy_plan(
      &etc2, 64, 144, &plan));
   CHECK(plan.block_size_B == 8 && plan.buffer_row_stride_B == 24 &&
         plan.buffer_slice_stride_B == 48 && plan.buffer_span_B == 80);

   struct hk_g17p_buffer_to_image_copy block16 = volume;
   block16.format = VK_FORMAT_BC3_UNORM_BLOCK;
   CHECK(hk_g17p_compressed_buffer_to_image_copy_plan(
      &block16, 64, 224, &plan));
   CHECK(plan.block_size_B == 16 && plan.buffer_row_stride_B == 48 &&
         plan.buffer_slice_stride_B == 96 && plan.buffer_span_B == 160);
   block16.format = VK_FORMAT_EAC_R11G11_UNORM_BLOCK;
   CHECK(hk_g17p_compressed_buffer_to_image_copy_plan(
      &block16, 64, 224, &plan));
   CHECK(plan.block_size_B == 16 && plan.buffer_span_B == 160);

   struct hk_g17p_buffer_to_image_copy astc = volume;
   astc.format = VK_FORMAT_ASTC_5x4_UNORM_BLOCK;
   astc.mip_levels = 1;
   astc.mip_level = 0;
   astc.image_extent = (VkExtent3D){13, 9, 4};
   astc.image_offset = (VkOffset3D){5, 4, 1};
   astc.copy_extent = (VkExtent3D){8, 5, 2};
   astc.buffer_row_length = 15;
   astc.buffer_image_height = 12;
   CHECK(hk_g17p_compressed_buffer_to_image_copy_plan(
      &astc, 32, 256, &plan));
   CHECK(plan.block_width == 5 && plan.block_height == 4 &&
         plan.block_size_B == 16);
   CHECK(plan.base_x_el == 1 && plan.base_y_el == 1 &&
         plan.blocks_x == 2 && plan.blocks_y == 2);
   CHECK(plan.first_slice == 1 && plan.slice_count == 2 &&
         plan.buffer_row_stride_B == 48 &&
         plan.buffer_slice_stride_B == 144 && plan.buffer_span_B == 224);

   struct hk_g17p_buffer_to_image_copy line = volume;
   line.image_type = VK_IMAGE_TYPE_1D;
   line.mip_levels = 1;
   line.mip_level = 0;
   line.array_layers = 3;
   line.image_extent = (VkExtent3D){7, 1, 1};
   line.base_array_layer = 1;
   line.layer_count = VK_REMAINING_ARRAY_LAYERS;
   line.image_offset = (VkOffset3D){4, 0, 0};
   line.copy_extent = (VkExtent3D){3, 1, 1};
   line.buffer_row_length = 8;
   line.buffer_image_height = 0;
   CHECK(hk_g17p_compressed_buffer_to_image_copy_plan(
      &line, 8, 32, &plan));
   CHECK(plan.base_x_el == 1 && plan.blocks_x == 1 && plan.blocks_y == 1);
   CHECK(plan.first_slice == 1 && plan.slice_count == 2 &&
         plan.buffer_row_stride_B == 16 &&
         plan.buffer_slice_stride_B == 16 && plan.buffer_span_B == 24);

   struct hk_g17p_buffer_to_image_copy array = block16;
   array.image_type = VK_IMAGE_TYPE_2D;
   array.mip_levels = 1;
   array.mip_level = 0;
   array.array_layers = 4;
   array.image_extent = (VkExtent3D){8, 8, 1};
   array.base_array_layer = 1;
   array.layer_count = VK_REMAINING_ARRAY_LAYERS;
   array.image_offset = (VkOffset3D){0, 0, 0};
   array.copy_extent = (VkExtent3D){8, 8, 1};
   array.buffer_row_length = 0;
   array.buffer_image_height = 0;
   CHECK(hk_g17p_compressed_buffer_to_image_copy_plan(
      &array, 16, 208, &plan));
   CHECK(plan.first_slice == 1 && plan.slice_count == 3 &&
         plan.buffer_row_stride_B == 32 &&
         plan.buffer_slice_stride_B == 64 && plan.buffer_span_B == 192);
   array.layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
   CHECK(hk_g17p_compressed_image_to_buffer_copy_plan(
      &array, 16, 208, &plan));
   CHECK(plan.first_slice == 1 && plan.slice_count == 3 &&
         plan.buffer_row_stride_B == 32 &&
         plan.buffer_slice_stride_B == 64 && plan.buffer_span_B == 192);

   struct hk_g17p_buffer_to_image_copy malformed = volume;
   malformed.chip = AGX_CHIP_G17G;
   CHECK(!hk_g17p_compressed_buffer_to_image_copy_plan(
      &malformed, 64, 144, &plan));
   CHECK(plan.buffer_span_B == 0);
   malformed = volume;
   malformed.format = VK_FORMAT_R8G8B8A8_UNORM;
   CHECK(!hk_g17p_compressed_buffer_to_image_copy_plan(
      &malformed, 64, 144, &plan));
   malformed = volume;
   malformed.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
   CHECK(!hk_g17p_compressed_buffer_to_image_copy_plan(
      &malformed, 64, 144, &plan));
   malformed = volume;
   malformed.samples = VK_SAMPLE_COUNT_2_BIT;
   CHECK(!hk_g17p_compressed_buffer_to_image_copy_plan(
      &malformed, 64, 144, &plan));
   malformed = volume;
   malformed.aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
   CHECK(!hk_g17p_compressed_buffer_to_image_copy_plan(
      &malformed, 64, 144, &plan));
   malformed = readback;
   malformed.aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
   CHECK(!hk_g17p_compressed_image_to_buffer_copy_plan(
      &malformed, 64, 144, &plan));
   malformed = volume;
   malformed.layer_count = VK_REMAINING_ARRAY_LAYERS;
   CHECK(!hk_g17p_compressed_buffer_to_image_copy_plan(
      &malformed, 64, 144, &plan));
   malformed = volume;
   malformed.image_offset.z = 3;
   CHECK(!hk_g17p_compressed_buffer_to_image_copy_plan(
      &malformed, 64, 144, &plan));
   malformed = volume;
   malformed.image_offset.x = 2;
   CHECK(!hk_g17p_compressed_buffer_to_image_copy_plan(
      &malformed, 64, 144, &plan));
   malformed = volume;
   malformed.copy_extent.width = 5;
   CHECK(!hk_g17p_compressed_buffer_to_image_copy_plan(
      &malformed, 64, 144, &plan));
   malformed = volume;
   malformed.buffer_row_length = 10;
   CHECK(!hk_g17p_compressed_buffer_to_image_copy_plan(
      &malformed, 64, 144, &plan));
   malformed = volume;
   malformed.buffer_image_height = 7;
   CHECK(!hk_g17p_compressed_buffer_to_image_copy_plan(
      &malformed, 64, 144, &plan));
   malformed = array;
   malformed.image_offset.z = 1;
   CHECK(!hk_g17p_compressed_buffer_to_image_copy_plan(
      &malformed, 16, 208, &plan));
   malformed = line;
   malformed.copy_extent.height = 2;
   CHECK(!hk_g17p_compressed_buffer_to_image_copy_plan(
      &malformed, 8, 32, &plan));
   malformed = astc;
   malformed.buffer_row_length = 14;
   CHECK(!hk_g17p_compressed_buffer_to_image_copy_plan(
      &malformed, 32, 256, &plan));
   CHECK(!hk_g17p_compressed_buffer_to_image_copy_plan(
      &volume, 65, 145, &plan));
   CHECK(!hk_g17p_compressed_buffer_to_image_copy_plan(
      &volume, 64, 143, &plan));

   struct hk_g17p_buffer_to_image_copy huge = volume;
   huge.format = VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
   huge.mip_levels = 1;
   huge.mip_level = 0;
   huge.image_extent =
      (VkExtent3D){UINT32_C(1073741820), UINT32_C(4294967292), 9};
   huge.image_offset = (VkOffset3D){0, 0, 0};
   huge.copy_extent = huge.image_extent;
   huge.buffer_row_length = UINT32_C(1073741820);
   huge.buffer_image_height = UINT32_C(4294967292);
   CHECK(!hk_g17p_compressed_buffer_to_image_copy_plan(
      &huge, 0, UINT64_MAX, &plan));
   CHECK(!hk_g17p_compressed_buffer_to_image_copy_plan(
      NULL, 0, 0, &plan));
   CHECK(!hk_g17p_compressed_buffer_to_image_copy_plan(
      &volume, 0, 0, NULL));

   const struct hk_g17p_buffer_to_image_copy vkcube = {
      .chip = AGX_CHIP_G17P,
      .plane_format_renderable = true,
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

   /* Compressed Minecraft D32 uploads reach the graphics fallback. Exercise
    * the complete image and a padded-row subrectangle crossing 16x16 tiles. */
   struct hk_g17p_buffer_to_image_copy depth_upload = vkcube;
   depth_upload.format = VK_FORMAT_D32_SFLOAT;
   depth_upload.aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
   depth_upload.image_extent = (VkExtent3D){2408, 1506, 1};
   depth_upload.copy_extent = depth_upload.image_extent;
   CHECK(hk_g17p_buffer_to_image_copy_supported(&depth_upload));
   depth_upload.layout = VK_IMAGE_LAYOUT_GENERAL;
   depth_upload.image_offset = (VkOffset3D){15, 31, 0};
   depth_upload.copy_extent = (VkExtent3D){33, 17, 1};
   depth_upload.buffer_row_length = 40;
   depth_upload.buffer_image_height = 20;
   CHECK(hk_g17p_buffer_to_image_copy_supported(&depth_upload));
   depth_upload.buffer_row_length = 32;
   CHECK(!hk_g17p_buffer_to_image_copy_supported(&depth_upload));
   depth_upload.buffer_row_length = 40;
   depth_upload.image_offset.x = 2400;
   CHECK(!hk_g17p_buffer_to_image_copy_supported(&depth_upload));
   depth_upload.image_offset.x = 15;
   depth_upload.plane_format_renderable = false;
   CHECK(!hk_g17p_buffer_to_image_copy_supported(&depth_upload));
   depth_upload.plane_format_renderable = true;
   depth_upload.aspect = VK_IMAGE_ASPECT_STENCIL_BIT;
   CHECK(!hk_g17p_buffer_to_image_copy_supported(&depth_upload));
   depth_upload.aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
   depth_upload.format = VK_FORMAT_D32_SFLOAT_S8_UINT;
   CHECK(!hk_g17p_buffer_to_image_copy_supported(&depth_upload));
   depth_upload.format = VK_FORMAT_D32_SFLOAT;
   depth_upload.array_layers = 2;
   CHECK(!hk_g17p_buffer_to_image_copy_supported(&depth_upload));
   depth_upload.array_layers = 1;
   depth_upload.mip_levels = 2;
   CHECK(!hk_g17p_buffer_to_image_copy_supported(&depth_upload));
   depth_upload.mip_levels = 1;
   depth_upload.image_type = VK_IMAGE_TYPE_3D;
   CHECK(!hk_g17p_buffer_to_image_copy_supported(&depth_upload));
   depth_upload.image_type = VK_IMAGE_TYPE_2D;
   depth_upload.samples = VK_SAMPLE_COUNT_2_BIT;
   CHECK(!hk_g17p_buffer_to_image_copy_supported(&depth_upload));
   depth_upload.samples = VK_SAMPLE_COUNT_1_BIT;

   /* D16 uses the same bounded fallback through an R16_UINT view. */
   depth_upload.format = VK_FORMAT_D16_UNORM;
   CHECK(hk_g17p_buffer_to_image_copy_supported(&depth_upload));
   depth_upload.format = VK_FORMAT_D16_UNORM_S8_UINT;
   CHECK(!hk_g17p_buffer_to_image_copy_supported(&depth_upload));
   depth_upload.format = VK_FORMAT_X8_D24_UNORM_PACK32;
   CHECK(!hk_g17p_buffer_to_image_copy_supported(&depth_upload));

   struct hk_g17p_buffer_to_image_copy explicit_stride = vkcube;
   explicit_stride.buffer_row_length = explicit_stride.copy_extent.width;
   explicit_stride.buffer_image_height = explicit_stride.copy_extent.height;
   CHECK(hk_g17p_buffer_to_image_copy_supported(&explicit_stride));

   explicit_stride.buffer_row_length += 8;
   explicit_stride.buffer_image_height += 4;
   CHECK(hk_g17p_buffer_to_image_copy_supported(&explicit_stride));

   /* STK uploads hundreds of R8 and RGBA8/BGRA8 base levels immediately
    * before generating their mip chains.  Resolve padded rows and array
    * layers once so the driver can copy them through the raw tiled compute
    * kernel instead of allocating one firmware render job per texture. */
   struct hk_g17p_buffer_to_image_copy raw_upload = vkcube;
   raw_upload.array_layers = 3;
   raw_upload.base_array_layer = 1;
   raw_upload.layer_count = VK_REMAINING_ARRAY_LAYERS;
   raw_upload.image_offset = (VkOffset3D){11, 13, 0};
   raw_upload.copy_extent = (VkExtent3D){37, 29, 1};
   raw_upload.buffer_row_length = 80;
   raw_upload.buffer_image_height = 72;
   struct hk_g17p_uncompressed_copy_plan raw_plan;
   CHECK(hk_g17p_uncompressed_buffer_to_image_copy_plan(
      &raw_upload, 64, 32212, &raw_plan));
   CHECK(raw_plan.block_size_B == 4 && raw_plan.base_x_el == 11 &&
         raw_plan.base_y_el == 13 && raw_plan.width_el == 37 &&
         raw_plan.height_el == 29);
   CHECK(raw_plan.first_slice == 1 && raw_plan.slice_count == 2 &&
         raw_plan.buffer_row_stride_B == 320 &&
         raw_plan.buffer_slice_stride_B == 23040 &&
         raw_plan.buffer_span_B == 32148);
   CHECK(!hk_g17p_uncompressed_buffer_to_image_copy_plan(
      &raw_upload, 64, 32211, &raw_plan));
   CHECK(raw_plan.buffer_span_B == 0);

   raw_upload.format = VK_FORMAT_R8_UNORM;
   CHECK(hk_g17p_uncompressed_buffer_to_image_copy_plan(
      &raw_upload, 7, 8044, &raw_plan));
   CHECK(raw_plan.block_size_B == 1 &&
         raw_plan.buffer_row_stride_B == 80 &&
         raw_plan.buffer_slice_stride_B == 5760 &&
         raw_plan.buffer_span_B == 8037);

   struct hk_g17p_buffer_to_image_copy one_d_upload = raw_upload;
   one_d_upload.image_type = VK_IMAGE_TYPE_1D;
   one_d_upload.array_layers = 2;
   one_d_upload.base_array_layer = 0;
   one_d_upload.layer_count = 2;
   one_d_upload.image_extent = (VkExtent3D){256, 1, 1};
   one_d_upload.image_offset = (VkOffset3D){17, 0, 0};
   one_d_upload.copy_extent = (VkExtent3D){83, 1, 1};
   one_d_upload.buffer_row_length = 96;
   one_d_upload.buffer_image_height = 0;
   CHECK(hk_g17p_uncompressed_buffer_to_image_copy_plan(
      &one_d_upload, 3, 182, &raw_plan));
   CHECK(raw_plan.block_size_B == 1 && raw_plan.base_x_el == 17 &&
         raw_plan.base_y_el == 0 && raw_plan.width_el == 83 &&
         raw_plan.height_el == 1 && raw_plan.first_slice == 0 &&
         raw_plan.slice_count == 2 && raw_plan.buffer_row_stride_B == 96 &&
         raw_plan.buffer_slice_stride_B == 96 &&
         raw_plan.buffer_span_B == 179);
   one_d_upload.image_offset.y = 1;
   CHECK(!hk_g17p_uncompressed_buffer_to_image_copy_plan(
      &one_d_upload, 3, 182, &raw_plan));
   one_d_upload.image_offset.y = 0;
   one_d_upload.copy_extent.height = 2;
   CHECK(!hk_g17p_uncompressed_buffer_to_image_copy_plan(
      &one_d_upload, 3, 182, &raw_plan));

   struct hk_g17p_buffer_to_image_copy raw_volume_upload = raw_upload;
   raw_volume_upload.image_type = VK_IMAGE_TYPE_3D;
   raw_volume_upload.array_layers = 1;
   raw_volume_upload.base_array_layer = 0;
   raw_volume_upload.layer_count = 1;
   raw_volume_upload.image_extent = (VkExtent3D){64, 32, 9};
   raw_volume_upload.image_offset = (VkOffset3D){7, 5, 2};
   raw_volume_upload.copy_extent = (VkExtent3D){17, 11, 3};
   raw_volume_upload.buffer_row_length = 24;
   raw_volume_upload.buffer_image_height = 16;
   CHECK(hk_g17p_uncompressed_buffer_to_image_copy_plan(
      &raw_volume_upload, 8, 1033, &raw_plan));
   CHECK(raw_plan.block_size_B == 1 && raw_plan.base_x_el == 7 &&
         raw_plan.base_y_el == 5 && raw_plan.width_el == 17 &&
         raw_plan.height_el == 11 && raw_plan.first_slice == 2 &&
         raw_plan.slice_count == 3 && raw_plan.buffer_row_stride_B == 24 &&
         raw_plan.buffer_slice_stride_B == 384 &&
         raw_plan.buffer_span_B == 1025);
   raw_volume_upload.copy_extent.depth = 8;
   CHECK(!hk_g17p_uncompressed_buffer_to_image_copy_plan(
      &raw_volume_upload, 8, UINT64_MAX, &raw_plan));
   raw_upload.format = VK_FORMAT_B8G8R8A8_SRGB;
   CHECK(hk_g17p_uncompressed_buffer_to_image_copy_plan(
      &raw_upload, 64, 32212, &raw_plan));
   raw_upload.format = VK_FORMAT_R8G8B8A8_UINT;
   CHECK(hk_g17p_uncompressed_buffer_to_image_copy_plan(
      &raw_upload, 64, 32212, &raw_plan));
   raw_upload.format = VK_FORMAT_R16G16_UNORM;
   CHECK(hk_g17p_uncompressed_buffer_to_image_copy_plan(
      &raw_upload, 64, 32212, &raw_plan));
   raw_upload.format = VK_FORMAT_R8G8_SRGB;
   CHECK(hk_g17p_uncompressed_buffer_to_image_copy_plan(
      &raw_upload, 8, 16148, &raw_plan));
   raw_upload.format = VK_FORMAT_R32G32_UINT;
   CHECK(hk_g17p_uncompressed_buffer_to_image_copy_plan(
      &raw_upload, 64, 64360, &raw_plan));
   CHECK(raw_plan.block_size_B == 8 &&
         raw_plan.buffer_row_stride_B == 640 &&
         raw_plan.buffer_slice_stride_B == 46080 &&
         raw_plan.buffer_span_B == 64296);
   raw_upload.format = VK_FORMAT_R16G16B16A16_SFLOAT;
   CHECK(hk_g17p_uncompressed_buffer_to_image_copy_plan(
      &raw_upload, 64, UINT64_MAX, &raw_plan));
   CHECK(raw_plan.block_size_B == 8);
   raw_upload.format = VK_FORMAT_R32G32B32A32_UINT;
   CHECK(hk_g17p_uncompressed_buffer_to_image_copy_plan(
      &raw_upload, 64, UINT64_MAX, &raw_plan));
   CHECK(raw_plan.block_size_B == 16);
   raw_upload.format = VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
   CHECK(!hk_g17p_uncompressed_buffer_to_image_copy_plan(
      &raw_upload, 64, UINT64_MAX, &raw_plan));
   raw_upload = vkcube;
   CHECK(!hk_g17p_uncompressed_buffer_to_image_copy_plan(
      &raw_upload, 2, 262146, &raw_plan));
   raw_upload.image_type = VK_IMAGE_TYPE_3D;
   raw_upload.array_layers = 2;
   CHECK(!hk_g17p_uncompressed_buffer_to_image_copy_plan(
      &raw_upload, 0, UINT64_MAX, &raw_plan));
   CHECK(!hk_g17p_uncompressed_buffer_to_image_copy_plan(
      NULL, 0, 0, &raw_plan));
   CHECK(!hk_g17p_uncompressed_buffer_to_image_copy_plan(
      &vkcube, 0, 0, NULL));

   struct hk_g17p_buffer_to_image_copy bad = vkcube;
   bad.format = VK_FORMAT_B8G8R8A8_UNORM;
   CHECK(hk_g17p_buffer_to_image_copy_supported(&bad));
   bad.format = VK_FORMAT_B8G8R8A8_SRGB;
   CHECK(hk_g17p_buffer_to_image_copy_supported(&bad));
   bad.format = VK_FORMAT_R8G8B8A8_SRGB;
   CHECK(hk_g17p_buffer_to_image_copy_supported(&bad));
   bad.format = VK_FORMAT_R8_UNORM;
   CHECK(hk_g17p_buffer_to_image_copy_supported(&bad));
   bad.format = VK_FORMAT_R8G8B8A8_UINT;
   CHECK(hk_g17p_buffer_to_image_copy_supported(&bad));
   /* R16G16B16A16_SFLOAT is renderable, so the driver advertises
    * COLOR_ATTACHMENT/TRANSFER_DST for it and the copy must be serviced.
    * It used to be refused for not being in a hardcoded allowlist, and that
    * refusal is not local: it fails vkEndCommandBuffer, so Zink drops the
    * whole frame and latches device_lost permanently. GL_RGBA16F is one of
    * the formats that hung SuperTuxKart that way. */
   bad.format = VK_FORMAT_R16G16B16A16_SFLOAT;
   CHECK(hk_g17p_buffer_to_image_copy_supported(&bad));
   /* A format the driver does NOT advertise as renderable still fails closed:
    * the generic vk_meta path below the gate renders through an EOT view and
    * has nowhere to put a non-renderable destination. */
   bad.plane_format_renderable = false;
   CHECK(!hk_g17p_buffer_to_image_copy_supported(&bad));
   bad.plane_format_renderable = true;
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
   bad.layout = VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR;
   CHECK(hk_g17p_buffer_to_image_copy_supported(&bad));

   /* Multi-planar copies address one plane at a time. The chroma extent is
    * expressed in plane texels, so 4:2:0 has a 32x32 RG8 plane beneath this
    * 64x64 image while 4:2:2 retains the full 64-pixel height. */
   struct hk_g17p_buffer_to_image_copy ycbcr = vkcube;
   ycbcr.format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
   ycbcr.plane_count = 2;
   ycbcr.image_extent = (VkExtent3D){64, 64, 1};
   ycbcr.aspect = VK_IMAGE_ASPECT_PLANE_0_BIT;
   ycbcr.copy_extent = (VkExtent3D){64, 64, 1};
   CHECK(hk_g17p_buffer_to_image_copy_supported(&ycbcr));
   ycbcr.aspect = VK_IMAGE_ASPECT_PLANE_1_BIT;
   ycbcr.copy_extent = (VkExtent3D){32, 32, 1};
   CHECK(hk_g17p_buffer_to_image_copy_supported(&ycbcr));
   ycbcr.copy_extent.width = 33;
   CHECK(!hk_g17p_buffer_to_image_copy_supported(&ycbcr));
   ycbcr.copy_extent = (VkExtent3D){32, 32, 1};
   ycbcr.aspect = VK_IMAGE_ASPECT_PLANE_2_BIT;
   CHECK(!hk_g17p_buffer_to_image_copy_supported(&ycbcr));
   ycbcr.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
   CHECK(!hk_g17p_buffer_to_image_copy_supported(&ycbcr));
   ycbcr.aspect = VK_IMAGE_ASPECT_PLANE_1_BIT;
   ycbcr.plane_count = 1;
   CHECK(!hk_g17p_buffer_to_image_copy_supported(&ycbcr));
   ycbcr.plane_count = 2;
   ycbcr.mip_levels = 2;
   CHECK(!hk_g17p_buffer_to_image_copy_supported(&ycbcr));
   ycbcr.mip_levels = 1;
   ycbcr.image_extent.width = 63;
   CHECK(!hk_g17p_buffer_to_image_copy_supported(&ycbcr));

   ycbcr.image_extent = (VkExtent3D){64, 64, 1};
   ycbcr.format = VK_FORMAT_G8_B8R8_2PLANE_422_UNORM;
   ycbcr.copy_extent = (VkExtent3D){32, 64, 1};
   CHECK(hk_g17p_buffer_to_image_copy_supported(&ycbcr));
   ycbcr.copy_extent.height = 65;
   CHECK(!hk_g17p_buffer_to_image_copy_supported(&ycbcr));

   ycbcr.format = VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM;
   ycbcr.plane_count = 3;
   ycbcr.aspect = VK_IMAGE_ASPECT_PLANE_2_BIT;
   ycbcr.copy_extent = (VkExtent3D){32, 32, 1};
   CHECK(hk_g17p_buffer_to_image_copy_supported(&ycbcr));

   /* The 10-bit-in-16 plane is not renderable, so the caller reports it as
    * such and the copy falls back rather than being serviced through an EOT
    * view it has no format for. */
   ycbcr.format = VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16;
   ycbcr.plane_count = 2;
   ycbcr.aspect = VK_IMAGE_ASPECT_PLANE_1_BIT;
   ycbcr.plane_format_renderable = false;
   CHECK(!hk_g17p_buffer_to_image_copy_supported(&ycbcr));
   ycbcr.plane_format_renderable = true;

   /* A volume upload is split into depth-one EOT views. The buffer image
    * stride remains the API rowLength*imageHeight product while Z advances
    * independently inside the selected mip. */
   struct hk_g17p_buffer_to_image_copy volume_upload = vkcube;
   volume_upload.image_type = VK_IMAGE_TYPE_3D;
   volume_upload.image_extent = (VkExtent3D){64, 32, 9};
   volume_upload.image_offset = (VkOffset3D){3, 4, 2};
   volume_upload.copy_extent = (VkExtent3D){17, 11, 4};
   volume_upload.buffer_row_length = 24;
   volume_upload.buffer_image_height = 16;
   CHECK(hk_g17p_buffer_to_image_copy_supported(&volume_upload));

   /* G17P renders each volume slice through a depth-one attachment view.
    * Layer is therefore zero inside every draw, even when the destination
    * view selects a nonzero Z. The source buffer already advances by one
    * image stride per slice and must retain that relative coordinate. */
   struct vk_meta_device meta = {.use_view_relative_3d_layer = true};
   for (uint32_t slice = 0; slice < volume_upload.copy_extent.depth; ++slice) {
      const VkOffset3D destination = {
         volume_upload.image_offset.x,
         volume_upload.image_offset.y,
         volume_upload.image_offset.z + (int32_t)slice,
      };
      const VkOffset3D source = vk_meta_buffer_to_image_gfx_source_offset(
         &meta, VK_IMAGE_TYPE_3D, destination);
      CHECK(source.x == destination.x && source.y == destination.y &&
            source.z == 0);
   }

   /* The correction is opt-in and never changes 1D/2D or another driver's
    * layered 3D copy coordinates. */
   VkOffset3D destination = {3, 4, 5};
   VkOffset3D source = vk_meta_buffer_to_image_gfx_source_offset(
      &meta, VK_IMAGE_TYPE_2D, destination);
   CHECK(source.x == 3 && source.y == 4 && source.z == 5);
   meta.use_view_relative_3d_layer = false;
   source = vk_meta_buffer_to_image_gfx_source_offset(
      &meta, VK_IMAGE_TYPE_3D, destination);
   CHECK(source.x == 3 && source.y == 4 && source.z == 5);

   struct hk_g17p_buffer_to_image_copy bad_volume = volume_upload;
   bad_volume.image_offset.z = -1;
   CHECK(!hk_g17p_buffer_to_image_copy_supported(&bad_volume));
   bad_volume = volume_upload;
   bad_volume.copy_extent.depth = 8;
   CHECK(!hk_g17p_buffer_to_image_copy_supported(&bad_volume));
   bad_volume = volume_upload;
   bad_volume.copy_extent.depth = 0;
   CHECK(!hk_g17p_buffer_to_image_copy_supported(&bad_volume));
   bad_volume = volume_upload;
   bad_volume.image_extent.depth = 0;
   CHECK(!hk_g17p_buffer_to_image_copy_supported(&bad_volume));
   bad_volume = volume_upload;
   bad_volume.image_extent.depth = UINT32_MAX;
   bad_volume.image_offset.z = INT32_MAX;
   bad_volume.copy_extent.depth = 2;
   CHECK(!hk_g17p_buffer_to_image_copy_supported(&bad_volume));
   bad_volume = volume_upload;
   bad_volume.array_layers = 2;
   CHECK(!hk_g17p_buffer_to_image_copy_supported(&bad_volume));
   bad_volume = volume_upload;
   bad_volume.layer_count = VK_REMAINING_ARRAY_LAYERS;
   CHECK(!hk_g17p_buffer_to_image_copy_supported(&bad_volume));
   bad_volume = volume_upload;
   bad_volume.image_type = VK_IMAGE_TYPE_1D;
   CHECK(!hk_g17p_buffer_to_image_copy_supported(&bad_volume));
   bad_volume = vkcube;
   bad_volume.copy_extent.depth = 2;
   CHECK(!hk_g17p_buffer_to_image_copy_supported(&bad_volume));

   volume_upload.mip_levels = 2;
   volume_upload.mip_level = 1;
   volume_upload.image_offset = (VkOffset3D){3, 2, 1};
   volume_upload.copy_extent = (VkExtent3D){17, 11, 3};
   CHECK(hk_g17p_buffer_to_image_copy_supported(&volume_upload));
   volume_upload.image_offset.z = 2;
   CHECK(!hk_g17p_buffer_to_image_copy_supported(&volume_upload));

   /* Exactly the layouts vkCmdCopyBufferToImage accepts as dstImageLayout.
    * VK_IMAGE_LAYOUT_GENERAL is what a VK_KHR_unified_image_layouts client
    * (Zink with driver_workarounds.general_layout) passes for every copy, so
    * rejecting it would turn every G17P block-compressed upload into
    * VK_ERROR_FEATURE_NOT_PRESENT. */
   CHECK(hk_g17p_transfer_dst_layout_supported(
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL));
   CHECK(hk_g17p_transfer_dst_layout_supported(VK_IMAGE_LAYOUT_GENERAL));
   CHECK(hk_g17p_transfer_dst_layout_supported(
      VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR));
   CHECK(!hk_g17p_transfer_dst_layout_supported(VK_IMAGE_LAYOUT_UNDEFINED));
   CHECK(!hk_g17p_transfer_dst_layout_supported(
      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL));
   CHECK(!hk_g17p_transfer_dst_layout_supported(
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
   CHECK(!hk_g17p_transfer_dst_layout_supported(
      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL));
   CHECK(!hk_g17p_transfer_dst_layout_supported(
      VK_IMAGE_LAYOUT_PRESENT_SRC_KHR));
   CHECK(hk_g17p_transfer_src_layout_supported(
      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL));
   CHECK(hk_g17p_transfer_src_layout_supported(VK_IMAGE_LAYOUT_GENERAL));
   CHECK(hk_g17p_transfer_src_layout_supported(
      VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR));
   CHECK(!hk_g17p_transfer_src_layout_supported(
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL));
   CHECK(!hk_g17p_transfer_src_layout_supported(
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));

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
      &layered, 3, &untouched));
   CHECK(!hk_g17p_image_copy_slice(
      AGX_CHIP_G17P, VK_IMAGE_TYPE_3D, VK_IMAGE_TYPE_3D, 3, 3,
      &layered, 0, &untouched));

   const VkImageCopy2 volume_copy = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_COPY_2,
      .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 2, 0, 1},
      .srcOffset = {7, 5, 3},
      .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 1, 0, 1},
      .dstOffset = {11, 13, 5},
      .extent = {37, 29, 3},
   };
   for (uint32_t slice = 0; slice < 3; ++slice) {
      VkImageCopy2 one;
      CHECK(hk_g17p_image_copy_slice(
         AGX_CHIP_G17P, VK_IMAGE_TYPE_3D, VK_IMAGE_TYPE_3D, 3, 3,
         &volume_copy, slice, &one));
      CHECK(one.srcOffset.z == 3 + (int32_t)slice);
      CHECK(one.dstOffset.z == 5 + (int32_t)slice);
      CHECK(one.srcSubresource.baseArrayLayer == 0 &&
            one.dstSubresource.baseArrayLayer == 0);
      CHECK(one.srcSubresource.layerCount == 1 &&
            one.dstSubresource.layerCount == 1 && one.extent.depth == 1);
      CHECK(one.extent.width == 37 && one.extent.height == 29);
   }

   /* Vulkan permits copies between 1D and 2D images when height and depth
    * are one. Both endpoints are ordinary single-layer EOT views on G17P. */
   const VkImageCopy2 one_d_two_d = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_COPY_2,
      .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 1, 2, 1},
      .srcOffset = {7, 0, 0},
      .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 2, 5, 1},
      .dstOffset = {11, 0, 0},
      .extent = {37, 1, 1},
   };
   VkImageCopy2 one_d_out;
   CHECK(hk_g17p_image_copy_slice(
      AGX_CHIP_G17P, VK_IMAGE_TYPE_1D, VK_IMAGE_TYPE_2D, 1, 1,
      &one_d_two_d, 0, &one_d_out));
   CHECK(one_d_out.srcSubresource.baseArrayLayer == 2);
   CHECK(one_d_out.dstSubresource.baseArrayLayer == 5);
   CHECK(one_d_out.srcSubresource.layerCount == 1);
   CHECK(one_d_out.dstSubresource.layerCount == 1);
   CHECK(one_d_out.extent.width == 37 && one_d_out.extent.height == 1 &&
         one_d_out.extent.depth == 1);
   CHECK(hk_g17p_image_copy_slice(
      AGX_CHIP_G17P, VK_IMAGE_TYPE_2D, VK_IMAGE_TYPE_1D, 1, 1,
      &one_d_two_d, 0, &one_d_out));
   CHECK(!hk_g17p_image_copy_slice(
      AGX_CHIP_G17P, VK_IMAGE_TYPE_2D, VK_IMAGE_TYPE_1D, 2, 2,
      &one_d_two_d, 0, &one_d_out));

   /* A 1D array copies to or from 3D slices exactly as a 2D array does: its
    * layers map to the volume's depth. */
   const VkImageCopy2 one_d_volume = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_COPY_2,
      .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 4},
      .srcOffset = {3, 0, 0},
      .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .dstOffset = {5, 0, 2},
      .extent = {17, 1, 4},
   };
   for (uint32_t slice = 0; slice < 4; ++slice) {
      CHECK(hk_g17p_image_copy_slice(
         AGX_CHIP_G17P, VK_IMAGE_TYPE_1D, VK_IMAGE_TYPE_3D, 4, 4,
         &one_d_volume, slice, &one_d_out));
      CHECK(one_d_out.srcSubresource.baseArrayLayer == 1 + slice);
      CHECK(one_d_out.dstOffset.z == 2 + (int32_t)slice);
      CHECK(one_d_out.extent.depth == 1);
   }
   CHECK(!hk_g17p_image_copy_slice(
      AGX_CHIP_G17P, VK_IMAGE_TYPE_1D, VK_IMAGE_TYPE_3D, 3, 3,
      &one_d_volume, 0, &one_d_out));

   VkImageCopy2 volume_array = volume_copy;
   volume_array.dstSubresource.baseArrayLayer = 7;
   volume_array.dstOffset.z = 0;
   for (uint32_t slice = 0; slice < 3; ++slice) {
      VkImageCopy2 one;
      CHECK(hk_g17p_image_copy_slice(
         AGX_CHIP_G17P, VK_IMAGE_TYPE_3D, VK_IMAGE_TYPE_2D, 3, 3,
         &volume_array, slice, &one));
      CHECK(one.srcOffset.z == 3 + (int32_t)slice);
      CHECK(one.dstOffset.z == 0);
      CHECK(one.dstSubresource.baseArrayLayer == 7 + slice);
      CHECK(one.extent.depth == 1);
   }

   VkImageCopy2 array_volume = volume_copy;
   array_volume.srcSubresource.baseArrayLayer = 7;
   array_volume.srcOffset.z = 0;
   for (uint32_t slice = 0; slice < 3; ++slice) {
      VkImageCopy2 one;
      CHECK(hk_g17p_image_copy_slice(
         AGX_CHIP_G17P, VK_IMAGE_TYPE_2D, VK_IMAGE_TYPE_3D, 3, 3,
         &array_volume, slice, &one));
      CHECK(one.srcSubresource.baseArrayLayer == 7 + slice);
      CHECK(one.dstOffset.z == 5 + (int32_t)slice);
      CHECK(one.extent.depth == 1);
   }

   VkImageCopy2 offset_overflow = volume_copy;
   offset_overflow.srcOffset.z = INT32_MAX;
   CHECK(!hk_g17p_image_copy_slice(
      AGX_CHIP_G17P, VK_IMAGE_TYPE_3D, VK_IMAGE_TYPE_3D, 3, 3,
      &offset_overflow, 1, &untouched));
   offset_overflow = volume_copy;
   offset_overflow.dstOffset.z = -1;
   CHECK(!hk_g17p_image_copy_slice(
      AGX_CHIP_G17P, VK_IMAGE_TYPE_3D, VK_IMAGE_TYPE_3D, 3, 3,
      &offset_overflow, 0, &untouched));
   offset_overflow = volume_array;
   offset_overflow.dstSubresource.baseArrayLayer = UINT32_MAX;
   CHECK(!hk_g17p_image_copy_slice(
      AGX_CHIP_G17P, VK_IMAGE_TYPE_3D, VK_IMAGE_TYPE_2D, 3, 3,
      &offset_overflow, 1, &untouched));
   offset_overflow = volume_copy;
   offset_overflow.extent.depth = 2;
   CHECK(!hk_g17p_image_copy_slice(
      AGX_CHIP_G17P, VK_IMAGE_TYPE_3D, VK_IMAGE_TYPE_3D, 3, 3,
      &offset_overflow, 0, &untouched));
   CHECK(!hk_g17p_image_copy_slice(
      AGX_CHIP_G17P, VK_IMAGE_TYPE_1D, VK_IMAGE_TYPE_3D, 3, 3,
      &volume_copy, 0, &untouched));
   CHECK(!hk_g17p_image_copy_slice(
      AGX_CHIP_G17P, VK_IMAGE_TYPE_1D, VK_IMAGE_TYPE_1D, 3, 3,
      &volume_copy, 0, &untouched));
   CHECK(!hk_g17p_image_copy_slice(
      AGX_CHIP_G17P, (VkImageType)99, VK_IMAGE_TYPE_3D, 3, 3,
      &volume_copy, 0, &untouched));

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

   /* A cube-compatible image is still a 2D array for transfer commands.
    * Exercise the exact public fixture: cube 1 at mip 0 is blitted to cube 0
    * at mip 1 through six independent one-layer views. The sampled coordinate
    * in each source view is layer zero, independent of unequal API bases. */
   VkImageBlit2 cube_mip = blit;
   cube_mip.srcSubresource.mipLevel = 0;
   cube_mip.srcSubresource.baseArrayLayer = 6;
   cube_mip.srcSubresource.layerCount = 6;
   cube_mip.dstSubresource.mipLevel = 1;
   cube_mip.dstSubresource.baseArrayLayer = 0;
   cube_mip.dstSubresource.layerCount = 6;
   for (uint32_t face = 0; face < 6; ++face) {
      VkImageBlit2 one;
      CHECK(hk_g17p_image_blit_slice(
         AGX_CHIP_G17P, VK_IMAGE_TYPE_2D, VK_IMAGE_TYPE_2D, 6, 6,
         &cube_mip, face, &one));
      CHECK(one.srcSubresource.mipLevel == 0 &&
            one.srcSubresource.baseArrayLayer == 6 + face &&
            one.srcSubresource.layerCount == 1);
      CHECK(one.dstSubresource.mipLevel == 1 &&
            one.dstSubresource.baseArrayLayer == face &&
            one.dstSubresource.layerCount == 1);
      const int64_t legacy_layer =
         (int64_t)one.dstSubresource.baseArrayLayer -
         (int64_t)one.srcSubresource.baseArrayLayer;
      CHECK(legacy_layer == -6 &&
            (legacy_layer < 0 ||
             legacy_layer >= one.srcSubresource.layerCount));
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
   CHECK(!hk_g17p_image_blit_slice(
      AGX_CHIP_G17P, VK_IMAGE_TYPE_2D, VK_IMAGE_TYPE_2D, 6, 6,
      &cube_mip, 6, &no_blit));
   bad_blit = cube_mip;
   bad_blit.srcSubresource.baseArrayLayer = UINT32_MAX;
   CHECK(!hk_g17p_image_blit_slice(
      AGX_CHIP_G17P, VK_IMAGE_TYPE_2D, VK_IMAGE_TYPE_2D, 6, 6,
      &bad_blit, 1, &no_blit));
   bad_blit = cube_mip;
   bad_blit.srcOffsets[0].z = 1;
   CHECK(!hk_g17p_image_blit_slice(
      AGX_CHIP_G17P, VK_IMAGE_TYPE_2D, VK_IMAGE_TYPE_2D, 6, 6,
      &bad_blit, 0, &no_blit));
   printf("Honeykrisp G17P meta copy checks passed (%u layer and %llu tiled-address cases)\n",
          tested, (unsigned long long)tile_cases);
   return 0;
}
