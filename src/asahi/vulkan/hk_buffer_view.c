/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "hk_buffer_view.h"
#include "asahi/lib/agx_abi.h"
#include "asahi/layout/layout.h"
#include "asahi/lib/agx_nir_lower_vbo.h"
#include "util/bitscan.h"
#include "util/u_debug.h"
#include "util/u_math.h"
#include "util/format/u_format.h"
#include "util/format/u_formats.h"

#include "agx_helpers.h"
#include "agx_nir_texture.h"
#include "agx_pack.h"
#include "hk_buffer.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_image.h"
#include "hk_physical_device.h"
#include "vk_acceleration_structure.h"

VkFormatFeatureFlags2
hk_get_buffer_format_features(struct hk_physical_device *pdev,
                              VkFormat vk_format)
{
   VkFormatFeatureFlags2 features = 0;
   enum pipe_format p_format = hk_format_to_pipe_format(vk_format);

   if (p_format == PIPE_FORMAT_NONE)
      return 0;

   if (agx_vbo_supports_format(p_format))
      features |= VK_FORMAT_FEATURE_2_VERTEX_BUFFER_BIT;

   if (pdev->vk.supported_extensions.KHR_acceleration_structure &&
       vk_acceleration_struct_vtx_format_supported(vk_format))
      features |= VK_FORMAT_FEATURE_2_ACCELERATION_STRUCTURE_VERTEX_BUFFER_BIT_KHR;

   if (ail_pixel_format[p_format].texturable &&
       !util_format_is_depth_or_stencil(p_format)) {

      features |= VK_FORMAT_FEATURE_2_UNIFORM_TEXEL_BUFFER_BIT;

      /* RGB32 specially supported for uniform texel buffers only. */
      if (util_is_power_of_two_nonzero(util_format_get_blocksize(p_format))) {
         /* Reads obtain their format from the buffer texture descriptor. */
         features |= VK_FORMAT_FEATURE_2_STORAGE_TEXEL_BUFFER_BIT |
                     VK_FORMAT_FEATURE_2_STORAGE_READ_WITHOUT_FORMAT_BIT |
                     VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT;
      }

      if (p_format == PIPE_FORMAT_R32_UINT || p_format == PIPE_FORMAT_R32_SINT ||
          (p_format == PIPE_FORMAT_R32_FLOAT &&
           (pdev->dev.chip == AGX_CHIP_G17P ||
            agx_chip_g17_m5(pdev->dev.chip))))
         features |= VK_FORMAT_FEATURE_2_STORAGE_TEXEL_BUFFER_ATOMIC_BIT;
   }

   return features;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_CreateBufferView(VkDevice _device, const VkBufferViewCreateInfo *pCreateInfo,
                    const VkAllocationCallbacks *pAllocator,
                    VkBufferView *pBufferView)
{
   VK_FROM_HANDLE(hk_device, device, _device);
   VK_FROM_HANDLE(hk_buffer, buffer, pCreateInfo->buffer);
   struct hk_buffer_view *view;

   view = vk_buffer_view_create(&device->vk, pCreateInfo, pAllocator,
                                sizeof(*view));
   if (!view)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   enum pipe_format format = hk_format_to_pipe_format(view->vk.format);
   const struct util_format_description *desc = util_format_description(format);
   const bool g17p = agx_native_graphics(&device->dev) ||
                      agx_chip_g17_m5(device->dev.chip);
   const bool rgb32 =
      ail_pixel_format[format].channels == AGX_CHANNELS_R32G32B32_EMULATED;

   uint8_t format_swizzle[4] = {
      desc->swizzle[0],
      desc->swizzle[1],
      desc->swizzle[2],
      desc->swizzle[3],
   };

   if (util_format_is_depth_or_stencil(format)) {
      assert(!util_format_is_depth_and_stencil(format) &&
             "separate stencil always used");

      /* Broadcast depth and stencil */
      format_swizzle[0] = 0;
      format_swizzle[1] = 0;
      format_swizzle[2] = 0;
      format_swizzle[3] = 0;
   }

   /* Decompose the offset into a multiple of 16-bytes (which we can include in
    * the address) and an extra texel-aligned tail offset of up to 15 bytes.
    *
    * This lets us offset partially in the shader instead, getting
    * around alignment restrictions on the base address pointer.
    */
   uint64_t head_offset_B = view->vk.offset & ~0xf;
   uint64_t rw_base = hk_buffer_address_rw(buffer, 0) + head_offset_B;
   uint64_t ro_base = hk_buffer_address_ro(buffer, 0) + head_offset_B;

   /* Which mapping the sampled texel-buffer descriptor addresses on G17P.
    *
    * The read-only shadow at +sparse_ro_offset is the mechanism every other
    * read-only descriptor uses to get sparseResidencyNonResidentStrict for
    * free: an unbound page reads as zero there.  For a texel buffer G17P
    * cannot rely on it, so the descriptor names the primary mapping and the
    * shader masks a non-resident texel to zero from the residency bitmap
    * instead -- one address and one residency source for both the sparse
    * fetch and the ordinary read.  See buffer_texel_resident() in
    * agx_nir_lower_texture.c.
    *
    * For a non-sparse buffer hk_buffer_address() returns the same address
    * either way (the shadow only exists for buffer->va != NULL), so this
    * cannot move an ordinary texel buffer.
    */
   const uint64_t sampled_base = g17p ? rw_base : ro_base;

   uint32_t tail_offset_B = view->vk.offset & 0xf;
   /* RGB32 permits component-aligned offsets, which cannot in general be
    * expressed as a whole-texel tail. Its G17P raw-load metadata can carry
    * the exact address instead; the unused hardware texture stays aligned.
    */
   const uint64_t metadata_base =
      sampled_base + ((g17p && rgb32) ? tail_offset_B : 0);
   if (g17p && rgb32)
      tail_offset_B = 0;
   uint32_t tail_offset_el = tail_offset_B / util_format_get_blocksize(format);
   assert(tail_offset_el * util_format_get_blocksize(format) == tail_offset_B &&
          "must be texel aligned");

   const unsigned generation =
      agx_chip_descriptor_generation(device->dev.chip);

   struct AGX_TEXTURE texture = {AGX_TEXTURE_header};
   {
      texture.dimension = AGX_TEXTURE_DIMENSION_2D;
      texture.layout = AGX_LAYOUT_LINEAR;
      texture.channels = ail_pixel_format[format].channels;
      texture.type = ail_pixel_format[format].type;
      texture.swizzle_r = agx_channel_from_pipe(format_swizzle[0]);
      texture.swizzle_g = agx_channel_from_pipe(format_swizzle[1]);
      texture.swizzle_b = agx_channel_from_pipe(format_swizzle[2]);
      texture.swizzle_a = agx_channel_from_pipe(format_swizzle[3]);

      texture.width = AGX_TEXTURE_BUFFER_WIDTH;
      texture.height = DIV_ROUND_UP(view->vk.elements, texture.width);
      texture.first_level = texture.last_level = 0;

      texture.address = sampled_base;
      texture.buffer_size_sw = view->vk.elements;
      texture.buffer_offset_sw = tail_offset_el;

      texture.srgb = (desc->colorspace == UTIL_FORMAT_COLORSPACE_SRGB);
      texture.srgb_2_channel =
         texture.srgb && util_format_colormask(desc) == 0x3;

      texture.depth = 1;
      texture.stride =
         (texture.width * util_format_get_blocksize(format)) - 16;
   }
   if (g17p && rgb32) {
      texture.channels = AGX_CHANNELS_R32;
      texture.swizzle_g = AGX_CHANNEL_0;
      texture.swizzle_b = AGX_CHANNEL_0;
      texture.stride = texture.width * sizeof(uint32_t) - 16;
   }
   if (g17p) {
      /* A sparse-resident buffer carries a per-page residency bitmap, and the
       * buffer texture descriptor has no hardware sparse mode to report
       * residency with, so a sparse texel fetch reads that bitmap instead.
       * Pack the map and the element size the shader needs to find the page;
       * see AGX_G17P_BUFFER_TEXTURE_SPARSE_OFFSET.  A non-power-of-two
       * element size (the emulated RGB32 layout) takes the raw load path,
       * which claims residency anyway, so leave the word zero for it. */
      uint32_t sparse = 0;
      const unsigned blocksize_B = util_format_get_blocksize(format);
      if (buffer && buffer->sparse_map &&
          util_is_power_of_two_nonzero(blocksize_B)) {
         const uint64_t map_page = buffer->sparse_map->va->addr / AIL_PAGESIZE;
         assert((buffer->sparse_map->va->addr % AIL_PAGESIZE) == 0);
         /* The packed field leaves 28 bits for the map page.  Fail loudly
          * rather than silently truncating if the VA layout ever outgrows it;
          * a wrong map address would read residency from arbitrary memory. */
         if (map_page < (1u << (32 - AGX_G17P_BUFFER_TEXTURE_SPARSE_MAP_SHIFT))) {
            sparse = (uint32_t)(map_page
                                << AGX_G17P_BUFFER_TEXTURE_SPARSE_MAP_SHIFT) |
                     (uint32_t)util_logbase2(blocksize_B);
         } else {
            assert(!"sparse residency map outside the packed VA range");
         }
      }

      view->buffer_metadata = (struct hk_g17p_buffer_texture_metadata){
         .elements = view->vk.elements,
         .offset_el = tail_offset_el,
         .address = metadata_base,
         .rgb32 = rgb32,
         .sparse = sparse,
      };
      /* Gen9 has no AGX2 software tail fields. The shader consumes the
       * explicit metadata, while hardware sees the aligned base texture.
       */
      texture.buffer_offset_sw = 0;
      texture.height = DIV_ROUND_UP(view->vk.elements + tail_offset_el,
                                    texture.width);
   }
   if (!hk_pack_native_texture(generation, &view->tex, &texture)) {
      vk_buffer_view_destroy(&device->vk, pAllocator, &view->vk);
      return vk_error(device, VK_ERROR_FORMAT_NOT_SUPPORTED);
   }

   struct AGX_PBE pbe = {AGX_PBE_header};
   {
      pbe.dimension = AGX_TEXTURE_DIMENSION_2D;
      pbe.layout = AGX_LAYOUT_LINEAR;
      pbe.channels = ail_pixel_format[format].channels;
      pbe.type = ail_pixel_format[format].type;
      pbe.srgb = util_format_is_srgb(format);

      assert(desc->nr_channels >= 1 && desc->nr_channels <= 4);

      for (unsigned i = 0; i < desc->nr_channels; ++i) {
         if (desc->swizzle[i] == 0)
            pbe.swizzle_r = i;
         else if (desc->swizzle[i] == 1)
            pbe.swizzle_g = i;
         else if (desc->swizzle[i] == 2)
            pbe.swizzle_b = i;
         else if (desc->swizzle[i] == 3)
            pbe.swizzle_a = i;
      }

      pbe.buffer = rw_base;
      pbe.buffer_offset_sw = tail_offset_el;

      pbe.width = AGX_TEXTURE_BUFFER_WIDTH;
      pbe.height = DIV_ROUND_UP(view->vk.elements, pbe.width);
      pbe.level = 0;
      pbe.stride = (pbe.width * util_format_get_blocksize(format)) - 4;
      pbe.layers = 1;
      pbe.levels = 1;
   }

   if (hk_native_pbe_supported(generation)) {
      hk_pack_native_pbe(generation, &view->pbe, &pbe);
   } else if (agx_native_graphics(&device->dev) ||
              agx_chip_g17_m5(device->dev.chip)) {
      /* AGX3's linear PBE is architectural and has no software tail field,
       * exactly like the texture above: the store lowering adds the tail from
       * the buffer metadata. A format the native packer cannot represent (the
       * emulated RGB32 layout) has no storage usage, so its view keeps the
       * null PBE.
       */
      pbe.buffer_offset_sw = 0;
      pbe.height = DIV_ROUND_UP(view->vk.elements + tail_offset_el, pbe.width);
      if (!hk_pack_native_pbe(generation, &view->pbe, &pbe))
         hk_set_null_native_pbe(generation, &view->pbe);
   } else {
      memset(&view->pbe, 0, sizeof(view->pbe));
   }

   /* G17P reaches a sampled texture through the device heap, so the view's
    * texture descriptor has to live there and the descriptor carries only its
    * index.  Without this the index stays 0 and every texel-buffer fetch
    * resolves to the null texture.  Mirrors hk_image_view.c's per-plane
    * hk_descriptor_table_add(). */
   if (g17p) {
      VkResult result = hk_descriptor_table_add(
         device, &device->sampled_images, &view->tex,
         hk_texture_descriptor_size(generation), &view->sampled_heap_index);
      if (result != VK_SUCCESS) {
         vk_buffer_view_destroy(&device->vk, pAllocator, &view->vk);
         return result;
      }
   }

   *pBufferView = hk_buffer_view_to_handle(view);

   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
hk_DestroyBufferView(VkDevice _device, VkBufferView bufferView,
                     const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(hk_device, device, _device);
   VK_FROM_HANDLE(hk_buffer_view, view, bufferView);

   if (!view)
      return;

   if (view->sampled_heap_index)
      hk_descriptor_table_remove(device, &device->sampled_images,
                                 view->sampled_heap_index);

   vk_buffer_view_destroy(&device->vk, pAllocator, &view->vk);
}
