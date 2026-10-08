/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include "asahi/lib/agx_chip.h"
#include "vulkan/vulkan_core.h"
#include <stdbool.h>
#include <string.h>

/* G17P lossless compression policy.
 *
 * The compressed Gen9 PBE (write) packet matches Apple's HAL200 StateFields
 * encoder byte-for-byte for GPU-tiled one-layer 2D images, the read
 * descriptor carries the same compression bit and metadata address, and the
 * G17P ZLS control word Apple's Framebuffer builds uses the same load/store
 * compression bits (2/4/6/8), 16-bit depth format bit and metadata stride
 * encoding as Mesa's agx_pack_zls_control. The shapes below are the ones
 * whose every write path in hk goes through one of those three units or the
 * libagx_fast_clear metadata kernel:
 *
 *  - Color: single-plane, 1-8 byte texels, one layer, one sample. Layered
 *    and multisampled compressed PBE packets are refused by
 *    agx_pbe_agx3_core.c (no witness), so arrays/MSAA stay plain. 16-byte
 *    texels are excluded so that a single-sample tilebuffer never spills a
 *    compressed target (spilling needs libagx_decompress, which has no G17P
 *    binary): 8 targets x 8 B fit in the 120 B/sample G17P budget.
 *  - Depth: D16_UNORM, D32_SFLOAT and the D32_SFLOAT depth plane (plane 0)
 *    of D32_SFLOAT_S8_UINT, one level, one layer, one sample. Stencil
 *    planes stay plain. Zink stores GL Z24S8 as D32S8, so its depth plane
 *    is every GL depth buffer's ZLS traffic; buffer uploads into it take
 *    the graphics copy (R32_UINT PBE) and the stencil plane keeps the raw
 *    upload kernel.
 *
 * Images that may be shared with another process or device (external
 * memory, WSI, DRM modifiers) stay plain: the compositor and display paths
 * have their own compressed-format contracts.
 *
 * This is the only G17P policy: every eligible image is compressed, with no
 * runtime opt-out.
 */

/* Colour formats admitted on G17P: plain single-plane formats of 1-8 bytes
 * whose physical G17P storage equals the API format (no RGB padding) and
 * whose channel class has a bitwise copy view. hk_can_compress() additionally
 * requires ail_can_compress() (renderable, at least 16x16). */
static inline bool
hk_g17p_compression_color_format(VkFormat format)
{
   switch (format) {
   /* 1 byte */
   case VK_FORMAT_R8_UNORM:
   case VK_FORMAT_R8_SNORM:
   case VK_FORMAT_R8_UINT:
   case VK_FORMAT_R8_SINT:
   /* 2 bytes */
   case VK_FORMAT_R8G8_UNORM:
   case VK_FORMAT_R8G8_SNORM:
   case VK_FORMAT_R8G8_UINT:
   case VK_FORMAT_R8G8_SINT:
   case VK_FORMAT_R16_UNORM:
   case VK_FORMAT_R16_SNORM:
   case VK_FORMAT_R16_UINT:
   case VK_FORMAT_R16_SINT:
   case VK_FORMAT_R16_SFLOAT:
   case VK_FORMAT_R5G6B5_UNORM_PACK16:
   case VK_FORMAT_B5G6R5_UNORM_PACK16:
   /* 4 bytes */
   case VK_FORMAT_R8G8B8A8_UNORM:
   case VK_FORMAT_R8G8B8A8_SNORM:
   case VK_FORMAT_R8G8B8A8_UINT:
   case VK_FORMAT_R8G8B8A8_SINT:
   case VK_FORMAT_R8G8B8A8_SRGB:
   case VK_FORMAT_B8G8R8A8_UNORM:
   case VK_FORMAT_B8G8R8A8_SRGB:
   case VK_FORMAT_A8B8G8R8_UNORM_PACK32:
   case VK_FORMAT_A8B8G8R8_SNORM_PACK32:
   case VK_FORMAT_A8B8G8R8_UINT_PACK32:
   case VK_FORMAT_A8B8G8R8_SINT_PACK32:
   case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
   case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
   case VK_FORMAT_A2B10G10R10_UINT_PACK32:
   case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
   /* Not B10G11R11_UFLOAT: G17P packs it into one tile word and stores it
    * through an R32_UINT PBE (agx_g17p_rt0_pbe_format), a different
    * compression class from the R11G11B10 texture that reads it back. */
   case VK_FORMAT_R16G16_UNORM:
   case VK_FORMAT_R16G16_SNORM:
   case VK_FORMAT_R16G16_UINT:
   case VK_FORMAT_R16G16_SINT:
   case VK_FORMAT_R16G16_SFLOAT:
   case VK_FORMAT_R32_UINT:
   case VK_FORMAT_R32_SINT:
   case VK_FORMAT_R32_SFLOAT:
   /* 8 bytes */
   case VK_FORMAT_R16G16B16A16_UNORM:
   case VK_FORMAT_R16G16B16A16_SNORM:
   case VK_FORMAT_R16G16B16A16_UINT:
   case VK_FORMAT_R16G16B16A16_SINT:
   case VK_FORMAT_R16G16B16A16_SFLOAT:
   case VK_FORMAT_R32G32_UINT:
   case VK_FORMAT_R32G32_SINT:
   case VK_FORMAT_R32G32_SFLOAT:
      return true;
   default:
      return false;
   }
}

static inline bool
hk_g17p_compression_depth_format(VkFormat format)
{
   return format == VK_FORMAT_D16_UNORM || format == VK_FORMAT_D32_SFLOAT ||
          format == VK_FORMAT_D32_SFLOAT_S8_UINT;
}

/* Mesa-private WSI image marker (src/vulkan/wsi/wsi_common_private.h). */
#define HK_G17P_WSI_IMAGE_CREATE_INFO_MESA ((VkStructureType)1000001002)

/* Whether plane `plane` of an image created with `info` is compressed on
 * `chip`. There is no opt-out: every eligible G17P image takes the compressed
 * path. The generic hk_can_compress() checks (sparse, mutable-format
 * compatibility, ail_can_compress, the AGX-wide nocompress debug flag) still
 * apply. */
static inline bool
hk_g17p_compression_eligible(enum agx_chip chip, unsigned plane,
                             const VkImageCreateInfo *info)
{
   if (chip != AGX_CHIP_G17P || plane != 0 || !info)
      return false;

   /* Shared images keep their plain layout contract. Read each header by
    * copy: the chain's structures have other types, and Mesa builds with
    * strict aliasing (gcc -O2 dropped the stores of a stack chain). */
   for (const void *next = info->pNext; next;) {
      VkBaseInStructure ext;
      memcpy(&ext, next, sizeof(ext));
      if (ext.sType == VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO &&
          ((const VkExternalMemoryImageCreateInfo *)next)->handleTypes)
         return false;
      if (ext.sType == HK_G17P_WSI_IMAGE_CREATE_INFO_MESA)
         return false;
      next = ext.pNext;
   }

   VkImageCreateFlags allowed_flags =
      VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
   /* Zink marks every depth attachment sample-location compatible, including
    * single-sample images. This changes sample-location tracking, not the
    * depth plane's storage class. The one-sample restriction below remains.
    * Refusing the flag also makes the host-copy performance query advertise
    * a plain device layout, so zink adds HOST_TRANSFER and keeps depth plain.
    */
   if (hk_g17p_compression_depth_format(info->format))
      allowed_flags |= VK_IMAGE_CREATE_SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_BIT_EXT;
   /* Only usages whose every access goes through the texture unit, a PBE or
    * ZLS. Storage and host-transfer access is raw; feedback loops read what
    * is being written; video, shading-rate and density-map units have their
    * own (unknown) compression contracts. */
   VkImageUsageFlags allowed_usage =
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
      VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT |
      VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;

   /* Zink depth feedback reads use the depth plane's ZLS layout. Color
    * feedback keeps the plain layout required by texture barriers. */
   if (hk_g17p_compression_depth_format(info->format))
      allowed_usage |= VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT;

   if (info->imageType != VK_IMAGE_TYPE_2D || info->extent.depth != 1 ||
       info->arrayLayers != 1 || info->samples != VK_SAMPLE_COUNT_1_BIT ||
       info->tiling != VK_IMAGE_TILING_OPTIMAL || info->mipLevels < 1 ||
       (info->flags & ~allowed_flags) || (info->usage & ~allowed_usage) ||
       info->extent.width < 16 || info->extent.height < 16)
      return false;

   if (hk_g17p_compression_color_format(info->format)) {
      /* hk_CmdBlitImage2 generates RGBA8/BGRA8 UNORM mip chains with the raw
       * AIL compute kernels (hk_g17p_compute_blit), which cannot write
       * compression metadata; a compressed destination falls back to one
       * render job per level. An image the application never renders to is
       * written only by uploads and those blits, so keep it on the compute
       * path. Images that are also attachments stay compressed. */
      if ((info->format == VK_FORMAT_R8G8B8A8_UNORM ||
           info->format == VK_FORMAT_B8G8R8A8_UNORM) &&
          info->mipLevels > 1 &&
          !(info->usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT))
         return false;
      return true;
   }

   return hk_g17p_compression_depth_format(info->format) &&
          info->mipLevels == 1;
}
