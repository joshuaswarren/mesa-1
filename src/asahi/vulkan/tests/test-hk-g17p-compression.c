/* Copyright 2026 Aurora Silicon; SPDX-License-Identifier: MIT
 *
 * Pure checks of the G17P lossless-compression admission policy
 * (hk_g17p_compression.h). The generic hk_can_compress() checks are applied
 * separately by the driver.
 */
#include "asahi/vulkan/hk_g17p_compression.h"
#include <stdio.h>
#include <stdlib.h>

static unsigned checks;
#define CHECK(x)                                                              \
   do {                                                                       \
      checks++;                                                               \
      if (!(x)) {                                                             \
         fprintf(stderr, "line %u: %s\n", __LINE__, #x);                      \
         exit(1);                                                             \
      }                                                                       \
   } while (0)

#define ELIGIBLE(i) hk_g17p_compression_eligible(AGX_CHIP_G17P, 0, &(i))

static VkImageCreateInfo
image(VkFormat format, uint32_t w, uint32_t h, uint32_t mips,
      VkImageUsageFlags usage)
{
   return (VkImageCreateInfo){
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = format,
      .extent = {w, h, 1},
      .mipLevels = mips,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
}

static void
check_formats(void)
{
   const VkImageUsageFlags color_usage =
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

   /* Every core format below VK_FORMAT_ASTC_12x12_SRGB_BLOCK. */
   unsigned color = 0, depth = 0;
   for (unsigned f = 1; f <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK; ++f) {
      const bool c = hk_g17p_compression_color_format((VkFormat)f);
      const bool d = hk_g17p_compression_depth_format((VkFormat)f);
      CHECK(!(c && d));
      color += c;
      depth += d;

      VkImageCreateInfo i = image((VkFormat)f, 64, 64, 1, color_usage);
      CHECK(ELIGIBLE(i) == (c || d));
   }
   CHECK(color == 46);
   CHECK(depth == 3);

   /* No 16-byte texels (tilebuffer spill), no packed/padded RGB, no stencil
    * or combined depth/stencil, no block-compressed or YCbCr formats. */
   static const VkFormat rejected[] = {
      VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R32G32B32A32_UINT,
      VK_FORMAT_R32G32B32A32_SINT,   VK_FORMAT_R32G32B32_SFLOAT,
      VK_FORMAT_R8G8B8_UNORM,        VK_FORMAT_R16G16B16_SFLOAT,
      VK_FORMAT_B10G11R11_UFLOAT_PACK32,
      VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, VK_FORMAT_R4G4B4A4_UNORM_PACK16,
      VK_FORMAT_A1R5G5B5_UNORM_PACK16, VK_FORMAT_S8_UINT,
      VK_FORMAT_D24_UNORM_S8_UINT,   VK_FORMAT_D16_UNORM_S8_UINT,
      VK_FORMAT_X8_D24_UNORM_PACK32,
      VK_FORMAT_BC1_RGBA_UNORM_BLOCK, VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK,
      VK_FORMAT_G8_B8R8_2PLANE_420_UNORM, VK_FORMAT_G8B8G8R8_422_UNORM,
      VK_FORMAT_R64_UINT,
   };
   for (unsigned r = 0; r < sizeof(rejected) / sizeof(rejected[0]); ++r) {
      VkImageCreateInfo i = image(rejected[r], 64, 64, 1, color_usage);
      CHECK(!ELIGIBLE(i));
   }
}

static void
check_shape(VkFormat format, bool is_depth)
{
   const VkImageUsageFlags usage =
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
      VK_IMAGE_USAGE_SAMPLED_BIT |
      (is_depth ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
   const VkImageCreateInfo base = image(format, 2408, 1508, 1, usage);
   CHECK(ELIGIBLE(base));

   /* Chip and plane. */
   for (unsigned chip = 0; chip <= AGX_CHIP_UNKNOWN; ++chip)
      CHECK(hk_g17p_compression_eligible((enum agx_chip)chip, 0, &base) ==
            (chip == AGX_CHIP_G17P));
   CHECK(!hk_g17p_compression_eligible(AGX_CHIP_G17P, 1, &base));
   CHECK(!hk_g17p_compression_eligible(AGX_CHIP_G17P, 0, NULL));

   /* Usage: only texture/PBE/ZLS usages are admitted. */
   VkImageUsageFlags allowed =
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
      VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT |
      VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
   if (is_depth)
      allowed |= VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT;
   for (unsigned bit = 0; bit < 32; ++bit) {
      VkImageCreateInfo i = base;
      i.usage |= 1u << bit;
      CHECK(ELIGIBLE(i) == !!((1u << bit) & allowed));
   }
   for (unsigned usage_bits = 0; usage_bits < 256; ++usage_bits) {
      VkImageCreateInfo i = base;
      i.usage = usage_bits & allowed;
      CHECK(ELIGIBLE(i));
   }
   static const VkImageUsageFlags refused[] = {
      VK_IMAGE_USAGE_STORAGE_BIT, VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT,
      VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT,
      VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR,
      VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR,
      VK_IMAGE_USAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR,
      VK_IMAGE_USAGE_FRAGMENT_DENSITY_MAP_BIT_EXT,
   };
   for (unsigned r = 0; r < sizeof(refused) / sizeof(refused[0]); ++r) {
      VkImageCreateInfo i = base;
      i.usage |= refused[r];
      CHECK(ELIGIBLE(i) == (is_depth && refused[r] ==
                           VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT));
   }

   /* Sample-location tracking is also admitted for single-sample depth. */
   for (unsigned bit = 0; bit < 32; ++bit) {
      VkImageCreateInfo i = base;
      i.flags = 1u << bit;
      CHECK(ELIGIBLE(i) ==
            ((1u << bit) == VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT ||
             (1u << bit) == VK_IMAGE_CREATE_EXTENDED_USAGE_BIT ||
             (is_depth && (1u << bit) ==
                          VK_IMAGE_CREATE_SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_BIT_EXT)));
   }

   /* Geometry. */
   VkImageCreateInfo i = base;
   i.imageType = VK_IMAGE_TYPE_1D;
   CHECK(!ELIGIBLE(i));
   i = base; i.imageType = VK_IMAGE_TYPE_3D;
   CHECK(!ELIGIBLE(i));
   i = base; i.extent.depth = 2;
   CHECK(!ELIGIBLE(i));
   i = base; i.arrayLayers = 2;
   CHECK(!ELIGIBLE(i));
   i = base; i.arrayLayers = 6;
   CHECK(!ELIGIBLE(i));
   for (unsigned s = 2; s <= 64; s <<= 1) {
      i = base; i.samples = (VkSampleCountFlagBits)s;
      CHECK(!ELIGIBLE(i));
   }
   i = base; i.tiling = VK_IMAGE_TILING_LINEAR;
   CHECK(!ELIGIBLE(i));
   i = base; i.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
   CHECK(!ELIGIBLE(i));
   i = base; i.mipLevels = 0;
   CHECK(!ELIGIBLE(i));
   i = base; i.mipLevels = 12;
   CHECK(ELIGIBLE(i) == !is_depth);
   i = base; i.extent.width = 15;
   CHECK(!ELIGIBLE(i));
   i = base; i.extent.height = 15;
   CHECK(!ELIGIBLE(i));
   i = base; i.extent.width = 16; i.extent.height = 16;
   CHECK(ELIGIBLE(i));
   i = base; i.extent.width = 17; i.extent.height = 31;
   CHECK(ELIGIBLE(i));
   i = base; i.extent.width = 16384; i.extent.height = 16384;
   CHECK(ELIGIBLE(i));
   i = base; i.sharingMode = VK_SHARING_MODE_CONCURRENT;
   CHECK(ELIGIBLE(i));

   /* Shared images stay plain; an empty external handle set does not. */
   VkExternalMemoryImageCreateInfo external = {
      .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
      .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
   };
   VkImageFormatListCreateInfo list = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO,
      .pNext = &external,
   };
   i = base; i.pNext = &external;
   CHECK(!ELIGIBLE(i));
   i = base; i.pNext = &list;
   CHECK(!ELIGIBLE(i));
   external.handleTypes = 0;
   CHECK(ELIGIBLE(i));
   list.pNext = NULL;
   CHECK(ELIGIBLE(i));
   VkBaseInStructure wsi = {.sType = HK_G17P_WSI_IMAGE_CREATE_INFO_MESA};
   i = base; i.pNext = &wsi;
   CHECK(!ELIGIBLE(i));
}

static void
check_minecraft(void)
{
   /* The live Minecraft 26.3 images: two 2408x1508 RGBA8 colour targets
    * (usage 0x17), the 2048^2 five-level blocks atlas (0x17) and the
    * 2408x1508 D32 depth target (0x27). All take the compressed path. */
   const VkImageCreateInfo color =
      image(VK_FORMAT_R8G8B8A8_UNORM, 2408, 1508, 1, 0x17);
   const VkImageCreateInfo atlas =
      image(VK_FORMAT_R8G8B8A8_UNORM, 2048, 2048, 5, 0x17);
   const VkImageCreateInfo depth = image(VK_FORMAT_D32_SFLOAT, 2408, 1508, 1,
                                         0x27);
   CHECK(ELIGIBLE(color));
   CHECK(ELIGIBLE(atlas));
   CHECK(ELIGIBLE(depth));
   /* Blit-generated RGBA8/BGRA8 UNORM mip chains the application never
    * renders to stay on the raw compute path; everything else compresses. */
   const VkImageUsageFlags texture_usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
      VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
   const VkFormat compute_blit[] = {VK_FORMAT_R8G8B8A8_UNORM,
                                    VK_FORMAT_B8G8R8A8_UNORM};
   for (unsigned f = 0; f < 2; ++f) {
      VkImageCreateInfo t = image(compute_blit[f], 256, 256, 9, texture_usage);
      CHECK(!ELIGIBLE(t));
      t.mipLevels = 1;
      CHECK(ELIGIBLE(t));
      t.mipLevels = 9;
      t.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
      CHECK(ELIGIBLE(t));
   }
   VkImageCreateInfo srgb = image(VK_FORMAT_R8G8B8A8_SRGB, 256, 256, 9,
                                  texture_usage);
   CHECK(ELIGIBLE(srgb));
   VkImageCreateInfo half = image(VK_FORMAT_R16G16B16A16_SFLOAT, 256, 256, 9,
                                  texture_usage);
   CHECK(ELIGIBLE(half));

   /* The swapchain is external/WSI memory and stays plain. */
   VkExternalMemoryImageCreateInfo external = {
      .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
      .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
   };
   VkImageCreateInfo i = image(VK_FORMAT_B8G8R8A8_UNORM, 2408, 1508, 1, 0x13);
   CHECK(ELIGIBLE(i));
   i.pNext = &external;
   CHECK(!ELIGIBLE(i));
}

static void
check_zink(void)
{
   /* Portal 2 under zink (t-1079): GL render targets declare
    * feedback-loop usage and keep the uncompressed layout. The GL Z24S8
    * depth buffer is D32S8 on Honeykrisp; its depth plane compresses and
    * its stencil plane stays plain. */
   const VkImageUsageFlags rt = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
      VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT |
      VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT;
   const VkImageUsageFlags ds = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
      VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
      VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT;
   VkImageCreateInfo color = image(VK_FORMAT_R8G8B8A8_SRGB, 2408, 1506, 1, rt);
   color.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
   const VkFormat views[] = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SRGB};
   VkImageFormatListCreateInfo list = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO,
      .viewFormatCount = 2,
      .pViewFormats = views,
   };
   color.pNext = &list;
   CHECK(!ELIGIBLE(color));
   VkImageCreateInfo depth =
      image(VK_FORMAT_D32_SFLOAT_S8_UINT, 2408, 1506, 1, ds);
   depth.flags = VK_IMAGE_CREATE_SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_BIT_EXT;
   CHECK(ELIGIBLE(depth));
   CHECK(!hk_g17p_compression_eligible(AGX_CHIP_G17P, 1, &depth));
   /* This is Portal's actual shape. A device-copy performance query must
    * find its compressed layout when host transfer is removed; otherwise
    * zink retains HOST_TRANSFER and never uses depth compression. */
   depth.usage |= VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT;
   CHECK(!ELIGIBLE(depth));
   depth.usage &= ~VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT;
   CHECK(ELIGIBLE(depth));
   depth.samples = VK_SAMPLE_COUNT_4_BIT;
   CHECK(!ELIGIBLE(depth));
   const VkImageCreateInfo d32 = image(VK_FORMAT_D32_SFLOAT, 2408, 1506, 1, ds);
   CHECK(ELIGIBLE(d32));
   /* A host-transfer image keeps the plain layout host copies require. */
   VkImageCreateInfo hic = color;
   hic.usage |= VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT;
   CHECK(!ELIGIBLE(hic));
}

int
main(void)
{
   check_formats();
   check_shape(VK_FORMAT_R8G8B8A8_UNORM, false);
   check_shape(VK_FORMAT_B8G8R8A8_SRGB, false);
   check_shape(VK_FORMAT_R16G16B16A16_SFLOAT, false);
   check_shape(VK_FORMAT_R8_UNORM, false);
   check_shape(VK_FORMAT_D32_SFLOAT, true);
   check_shape(VK_FORMAT_D16_UNORM, true);
   check_shape(VK_FORMAT_D32_SFLOAT_S8_UINT, true);
   check_minecraft();
   check_zink();
   printf("G17P compression policy: %u checks passed\n", checks);
   return 0;
}
