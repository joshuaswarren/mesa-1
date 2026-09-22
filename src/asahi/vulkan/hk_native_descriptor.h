/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

/* Keep this header independent of Vulkan and generated GenXML headers.  Apart
 * from making the descriptor ABI easy to test on the host, this prevents a
 * recognised-but-disabled GPU generation from accidentally selecting the
 * legacy C struct layout.
 */
/* G13 uses Gen7, G14 uses Gen8, G15G/S/C/D, G16P/G/S/C, and G17P/A use
 * Gen9, while G17G/S/C uses Gen10.  G15P remains deliberately unknown. */
#define HK_TEXTURE_DESCRIPTOR_AGX2_SIZE 24
#define HK_TEXTURE_DESCRIPTOR_AGX3_SIZE 32
#define HK_PBE_DESCRIPTOR_AGX2_SIZE     24
#define HK_PBE_DESCRIPTOR_AGX3_SIZE     32
#define HK_SAMPLED_IMAGE_SW_SIZE        40
#define HK_G17P_STORAGE_SW_SIZE         16
#define HK_G17P_STORAGE_ALIGNMENT       32
#define HK_G17P_STORAGE_STRIDE          96

/* G17P imgwci indexes whole PBE records in the heap it addresses. */
#define HK_G17P_PBE_RECORD_SIZE         HK_PBE_DESCRIPTOR_AGX3_SIZE

static inline bool
hk_pack_g17p_image_atomic_sw(uint64_t level_offset_B, unsigned tile_width_px,
                             unsigned tile_height_px, uint64_t layer_stride_B,
                             unsigned samples, uint64_t *out)
{
   if (!out || (level_offset_B & 0x7f) || (layer_stride_B & 0x7f) ||
       tile_width_px == 0 || (tile_width_px & (tile_width_px - 1)) ||
       tile_height_px == 0 || (tile_height_px & (tile_height_px - 1)) ||
       (samples != 1 && samples != 2 && samples != 4 && samples != 8))
      return false;

   unsigned tile_width_log2 = 0, tile_height_log2 = 0, samples_log2 = 0;
   for (unsigned x = tile_width_px; x > 1; x >>= 1)
      ++tile_width_log2;
   for (unsigned x = tile_height_px; x > 1; x >>= 1)
      ++tile_height_log2;
   for (unsigned x = samples; x > 1; x >>= 1)
      ++samples_log2;

   const uint64_t level_offset = level_offset_B >> 7;
   const uint64_t layer_stride = layer_stride_B >> 7;
   if (level_offset >= (UINT64_C(1) << 27) ||
       layer_stride >= (UINT64_C(1) << 27) || tile_width_log2 >= 16 ||
       tile_height_log2 >= 16 || samples_log2 >= 4)
      return false;

   *out = level_offset | ((uint64_t)tile_width_log2 << 27) |
          ((uint64_t)tile_height_log2 << 31) | (layer_stride << 35) |
          ((uint64_t)samples_log2 << 62);
   return true;
}

struct hk_g17p_buffer_texture_metadata {
   uint32_t elements;
   uint32_t offset_el;
   uint64_t address;
   uint32_t rgb32;
   /* See AGX_G17P_BUFFER_TEXTURE_SPARSE_OFFSET in agx_abi.h. */
   uint32_t sparse;
};

struct hk_sampled_image_descriptor_layout {
   uint8_t texture;
   uint8_t sampler;
   uint8_t sampler_index;
   uint8_t clamp_0_sampler_index;
   uint8_t lod_bias;
   uint8_t min_lod;
   uint8_t min_lod_uint;
   uint8_t border;
   uint8_t stride;
};

struct hk_storage_image_descriptor_layout {
   uint8_t texture;
   uint8_t pbe;
   uint8_t stride;
};

static inline bool
hk_descriptor_generation_is_agx2(unsigned generation)
{
   return generation == 7 || generation == 8;
}

static inline bool
hk_descriptor_generation_is_agx3(unsigned generation)
{
   return generation == 9 || generation == 10;
}

static inline unsigned
hk_texture_descriptor_size(unsigned generation)
{
   return hk_descriptor_generation_is_agx2(generation)
             ? HK_TEXTURE_DESCRIPTOR_AGX2_SIZE
          : hk_descriptor_generation_is_agx3(generation)
             ? HK_TEXTURE_DESCRIPTOR_AGX3_SIZE
             : 0;
}

static inline unsigned
hk_pbe_descriptor_size(unsigned generation)
{
   return hk_descriptor_generation_is_agx2(generation)
             ? HK_PBE_DESCRIPTOR_AGX2_SIZE
          : hk_descriptor_generation_is_agx3(generation)
             ? HK_PBE_DESCRIPTOR_AGX3_SIZE
             : 0;
}

static inline struct hk_sampled_image_descriptor_layout
hk_sampled_image_descriptor_layout(unsigned generation)
{
   const unsigned texture_size = hk_texture_descriptor_size(generation);

   if (!texture_size)
      return (struct hk_sampled_image_descriptor_layout){0};

   return (struct hk_sampled_image_descriptor_layout){
      .texture = 0,
      .sampler = texture_size,
      .sampler_index = texture_size + 8,
      .clamp_0_sampler_index = texture_size + 10,
      .lod_bias = texture_size + 12,
      .min_lod = texture_size + 14,
      .min_lod_uint = texture_size + 16,
      .border = texture_size + 20,
      .stride = texture_size + HK_SAMPLED_IMAGE_SW_SIZE,
   };
}

static inline struct hk_storage_image_descriptor_layout
hk_storage_image_descriptor_layout(unsigned generation)
{
   const unsigned texture_size = hk_texture_descriptor_size(generation);
   const unsigned pbe_size = hk_pbe_descriptor_size(generation);

   if (!texture_size || !pbe_size)
      return (struct hk_storage_image_descriptor_layout){0};

   return (struct hk_storage_image_descriptor_layout){
      .texture = 0,
      .pbe = texture_size,
      .stride = texture_size + pbe_size,
   };
}

/* AGX3 buffer descriptors have no recovered equivalent for AGX2's software
 * tail offset.  Keep those unavailable rather than aliasing unknown bits in a
 * native descriptor.  Multisample descriptor representation is recovered on
 * both AGX3 generations; this is not by itself an execution/capability gate.
 */
static inline bool
hk_native_texture_supported(unsigned generation, bool multisampled,
                            unsigned buffer_tail_offset)
{
   if (hk_descriptor_generation_is_agx2(generation))
      return true;

   if (!hk_descriptor_generation_is_agx3(generation))
      return false;

   return buffer_tail_offset == 0;
}

/* Return whether the sampled-texture descriptor has a recovered representation
 * for a sample count.  This is not a Vulkan capability predicate: the live
 * G17S corpus proves Gen10 creation/layout and descriptor bits but contains no
 * executed sampling command, so public 8x exposure remains gated.
 */
static inline bool
hk_native_texture_sample_count_representable(unsigned generation,
                                             unsigned samples)
{
   if (samples == 1)
      return hk_texture_descriptor_size(generation) != 0;

   if (hk_descriptor_generation_is_agx2(generation))
      return samples == 2 || samples == 4;

   if (generation == 9 || generation == 10)
      return samples == 2 || samples == 4 || samples == 8;

   return false;
}

/* Hardware PBE descriptors have an exact bounded Gen9/10 subset. */
static inline bool
hk_native_pbe_hardware_supported(unsigned generation)
{
   return hk_pbe_descriptor_size(generation) != 0;
}

/* Writable-image atomics, MSAA and buffers still rely on the AGX2 software
 * sideband decode. Keep those execution paths separate from ordinary native
 * PBE stores.
 */
static inline bool
hk_native_pbe_supported(unsigned generation)
{
   return hk_descriptor_generation_is_agx2(generation);
}
