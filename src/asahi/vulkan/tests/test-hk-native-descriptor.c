/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_native_descriptor_pack.h"

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

struct legacy_sampled_image_descriptor {
   uint8_t texture[24];
   uint8_t sampler[8];
   uint16_t sampler_index;
   int16_t clamp_0_sampler_index;
   uint16_t lod_bias;
   uint16_t min_lod;
   uint16_t min_lod_uint;
   uint16_t pad;
   uint32_t border[4];
   uint8_t pad2[4];
};

static void
check_layout(unsigned generation, unsigned texture_size,
             unsigned sampled_stride, unsigned storage_stride)
{
   const struct hk_sampled_image_descriptor_layout sampled =
      hk_sampled_image_descriptor_layout(generation);
   const struct hk_storage_image_descriptor_layout storage =
      hk_storage_image_descriptor_layout(generation);

   assert(hk_texture_descriptor_size(generation) == texture_size);
   assert(sampled.texture == 0);
   assert(sampled.sampler == texture_size);
   assert(sampled.sampler_index == texture_size + 8);
   assert(sampled.clamp_0_sampler_index == texture_size + 10);
   assert(sampled.lod_bias == texture_size + 12);
   assert(sampled.min_lod == texture_size + 14);
   assert(sampled.min_lod_uint == texture_size + 16);
   assert(sampled.border == texture_size + 20);
   assert(sampled.stride == sampled_stride);
   assert(storage.texture == 0);
   assert(storage.pbe == texture_size);
   assert(storage.stride == storage_stride);
}

static void
check_gen9_layer_mode_preserved(struct AGX_TEXTURE semantic)
{
   struct hk_native_texture_descriptor unaligned, aligned;
   semantic.page_aligned_layers = false;
   assert(hk_pack_native_texture(9, &unaligned, &semantic));
   semantic.page_aligned_layers = true;
   assert(hk_pack_native_texture(9, &aligned, &semantic));

   /* Outside the corrected contract, retain exactly the inherited bit and
    * leave every other byte unchanged. This is a scope check, not native
    * qualification of the excluded descriptor kinds.
    */
   assert((unaligned.opaque[3] & (1u << 30)) == 0);
   aligned.opaque[3] ^= 1u << 30;
   assert(memcmp(&aligned, &unaligned, sizeof(aligned)) == 0);
}

static void
check_gen9_bc1_rgba_texture(void)
{
   /* Native G17P descriptor capture, EXP-0028 fmt_bc1_rgba. It pins the
    * BC1 codec selector (channels0x74 -> physical fmt bytes 0x1d22), ordinary
    * 2D layout, RGBA swizzle, 32x32 extent and the uncompressed-block VA. */
   static const uint8_t expected[32] = {
      0x22, 0x1d, 0x88, 0xf6, 0x01, 0x7c, 0x00, 0x00,
      0x50, 0x1c, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   const struct AGX_TEXTURE semantic = {
      AGX_TEXTURE_header,
      .dimension = AGX_TEXTURE_DIMENSION_2D,
      .layout = AGX_LAYOUT_GPU,
      .channels = AGX_CHANNELS_BC1,
      .type = AGX_TEXTURE_TYPE_UNORM,
      .swizzle_r = AGX_CHANNEL_R,
      .swizzle_g = AGX_CHANNEL_G,
      .swizzle_b = AGX_CHANNEL_B,
      .swizzle_a = AGX_CHANNEL_A,
      .width = 32,
      .height = 32,
      .depth = 1,
      .address = UINT64_C(0x1000001c500),
   };
   struct hk_native_texture_descriptor packed;
   assert(hk_pack_native_texture(9, &packed, &semantic));
   assert(memcmp(&packed, expected, sizeof(expected)) == 0);
}

static void
check_gen9_etc2_rgba8_texture(void)
{
   /* Native G17P descriptor capture, EXP-0028 fmt_eac_rgba8. It pins the
    * ETC2/EAC RGBA8 codec selector (channels0x59 -> physical fmt bytes
    * 0x1662), ordinary 2D layout, RGBA swizzle, 32x32 extent and VA. */
   static const uint8_t expected[32] = {
      0x62, 0x16, 0x88, 0xf6, 0x01, 0x7c, 0x00, 0x00,
      0x50, 0x1c, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   const struct AGX_TEXTURE semantic = {
      AGX_TEXTURE_header,
      .dimension = AGX_TEXTURE_DIMENSION_2D,
      .layout = AGX_LAYOUT_GPU,
      .channels = AGX_CHANNELS_ETC2_RGBA8,
      .type = AGX_TEXTURE_TYPE_UNORM,
      .swizzle_r = AGX_CHANNEL_R,
      .swizzle_g = AGX_CHANNEL_G,
      .swizzle_b = AGX_CHANNEL_B,
      .swizzle_a = AGX_CHANNEL_A,
      .width = 32,
      .height = 32,
      .depth = 1,
      .address = UINT64_C(0x1000001c500),
   };
   struct hk_native_texture_descriptor packed;
   assert(hk_pack_native_texture(9, &packed, &semantic));
   assert(memcmp(&packed, expected, sizeof(expected)) == 0);
}

int
main(void)
{
   check_gen9_bc1_rgba_texture();
   check_gen9_etc2_rgba8_texture();

   /* Ordinary HAL200 2D has no layer-mode bit, even when its backing BO
    * allocation is page aligned. Compare the entire packet, not only bit62.
    */
   struct AGX_TEXTURE plain = {
      AGX_TEXTURE_header,
      .dimension = AGX_TEXTURE_DIMENSION_2D,
      .layout = AGX_LAYOUT_GPU,
      .channels = AGX_CHANNELS_R8G8B8A8,
      .type = AGX_TEXTURE_TYPE_UNORM,
      .swizzle_r = AGX_CHANNEL_R, .swizzle_g = AGX_CHANNEL_G,
      .swizzle_b = AGX_CHANNEL_B, .swizzle_a = AGX_CHANNEL_A,
      .width = 2, .height = 2, .address = 0x87ffefc000ull,
   };
   struct hk_native_texture_descriptor plain_unaligned, plain_aligned;
   for (unsigned generation = 8; generation <= 10; ++generation) {
      plain.page_aligned_layers = false;
      assert(hk_pack_native_texture(generation, &plain_unaligned, &plain));
      plain.page_aligned_layers = true;
      assert(hk_pack_native_texture(generation, &plain_aligned, &plain));
      if (generation == 9) {
         static const uint8_t expected[32] = {
            0x22,0x0a,0x88,0x16,0x00,0x04,0x00,0x00,
            0x00,0xfc,0xfe,0x7f,0x08,0x00,0x00,0x00,
         };
         assert(memcmp(&plain_aligned, expected, sizeof(expected)) == 0);
         assert(memcmp(&plain_aligned, &plain_unaligned, sizeof(plain_aligned)) == 0);
      } else {
         assert(memcmp(&plain_aligned, &plain_unaligned, sizeof(plain_aligned)) != 0);
      }
   }

   /* Each case independently crosses one plain-2D predicate boundary. In
    * particular, a one-layer array must still preserve the old array policy.
    */
   struct AGX_TEXTURE excluded = plain;
   excluded.dimension = AGX_TEXTURE_DIMENSION_2D_ARRAY;
   check_gen9_layer_mode_preserved(excluded);

   excluded = plain;
   excluded.dimension = AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED;
   check_gen9_layer_mode_preserved(excluded);

   excluded = plain;
   excluded.depth = 2;
   check_gen9_layer_mode_preserved(excluded);

   excluded = plain;
   excluded.layout = AGX_LAYOUT_LINEAR;
   check_gen9_layer_mode_preserved(excluded);

   excluded = plain;
   excluded.layout = AGX_LAYOUT_TWIDDLED;
   check_gen9_layer_mode_preserved(excluded);

   excluded = plain;
   excluded.first_level = 1;
   check_gen9_layer_mode_preserved(excluded);

   excluded = plain;
   excluded.last_level = 1;
   check_gen9_layer_mode_preserved(excluded);

   excluded = plain;
   excluded.unk_mipmapped = true;
   check_gen9_layer_mode_preserved(excluded);

   excluded = plain;
   excluded.extended = true;
   check_gen9_layer_mode_preserved(excluded);

   excluded = plain;
   excluded.compressed = true;
   check_gen9_layer_mode_preserved(excluded);

   excluded = plain;
   excluded.mode = AGX_IMAGE_MODE_SPARSE;
   check_gen9_layer_mode_preserved(excluded);

   static const struct {
      const char *chip;
      unsigned generation;
      unsigned texture_size;
      unsigned sampled_stride;
      unsigned storage_stride;
   } matrix[] = {
      {"G13 (M1/M2 class)", 7, 24, 64, 48},
      {"G14 (M2 class)", 8, 24, 64, 48},
      {"G15G (M3)", 9, 32, 72, 64},
      {"G15S (M3 Pro)", 9, 32, 72, 64},
      {"G15C/G15D (M3 Max/Ultra)", 9, 32, 72, 64},
      {"G16P (phone mapping ungrounded)", 9, 32, 72, 64},
      {"G16G (M4)", 9, 32, 72, 64},
      {"G16S (M4 Pro)", 9, 32, 72, 64},
      {"G16C (M4 Max)", 9, 32, 72, 64},
      {"G17P (A18 Pro) / G17A", 9, 32, 72, 64},
      {"G17G (M5)", 10, 32, 72, 64},
      {"G17S (M5 Pro)", 10, 32, 72, 64},
      {"G17C (M5 Max)", 10, 32, 72, 64},
   };

   for (unsigned i = 0; i < sizeof(matrix) / sizeof(matrix[0]); ++i) {
      assert(matrix[i].chip != NULL);
      check_layout(matrix[i].generation, matrix[i].texture_size,
                   matrix[i].sampled_stride, matrix[i].storage_stride);
   }

   /* The A16/G15P descriptor ABI is absent from the measured corpus. */
   assert(hk_texture_descriptor_size(0) == 0);
   assert(hk_sampled_image_descriptor_layout(0).stride == 0);
   assert(!hk_native_texture_supported(0, false, 0));
   assert(!hk_native_pbe_supported(0));

   /* Every byte of the existing AGX2 ABI remains at its old offset. */
   const struct hk_sampled_image_descriptor_layout legacy =
      hk_sampled_image_descriptor_layout(8);
   assert(sizeof(struct legacy_sampled_image_descriptor) == legacy.stride);
   assert(offsetof(struct legacy_sampled_image_descriptor, sampler) ==
          legacy.sampler);
   assert(offsetof(struct legacy_sampled_image_descriptor, sampler_index) ==
          legacy.sampler_index);
   assert(offsetof(struct legacy_sampled_image_descriptor,
                   clamp_0_sampler_index) == legacy.clamp_0_sampler_index);
   assert(offsetof(struct legacy_sampled_image_descriptor, lod_bias) ==
          legacy.lod_bias);
   assert(offsetof(struct legacy_sampled_image_descriptor, min_lod) ==
          legacy.min_lod);
   assert(offsetof(struct legacy_sampled_image_descriptor, min_lod_uint) ==
          legacy.min_lod_uint);
   assert(offsetof(struct legacy_sampled_image_descriptor, border) ==
          legacy.border);

   assert(hk_native_texture_supported(7, true, 12));
   assert(hk_native_texture_supported(9, true, 0));
   assert(hk_native_texture_supported(9, false, 0));
   assert(!hk_native_texture_supported(9, false, 4));
   assert(hk_native_texture_supported(10, true, 0));
   assert(!hk_native_texture_supported(10, false, 12));
   assert(hk_native_texture_sample_count_representable(7, 2));
   assert(hk_native_texture_sample_count_representable(8, 4));
   assert(hk_native_texture_sample_count_representable(9, 2));
   assert(hk_native_texture_sample_count_representable(9, 4));
   assert(hk_native_texture_sample_count_representable(9, 8));
   assert(!hk_native_texture_sample_count_representable(9, 16));
   assert(hk_native_texture_sample_count_representable(10, 8));
   assert(!hk_native_texture_sample_count_representable(10, 16));
   assert(!hk_native_texture_sample_count_representable(0, 1));
   assert(hk_native_pbe_supported(7));
   assert(hk_native_pbe_supported(8));
   assert(!hk_native_pbe_supported(9));
   assert(!hk_native_pbe_supported(10));

   assert(hk_native_pbe_hardware_supported(7));
   assert(hk_native_pbe_hardware_supported(8));
   assert(hk_native_pbe_hardware_supported(9));
   assert(hk_native_pbe_hardware_supported(10));
   assert(!hk_native_pbe_hardware_supported(0));

   struct hk_native_texture_descriptor null9, null10;
   assert(hk_set_null_native_texture(9, &null9));
   assert(hk_set_null_native_texture(10, &null10));
   assert(memcmp(&null9, &(struct hk_native_texture_descriptor){0},
                 sizeof(null9)) != 0);
   assert(memcmp(&null10, &(struct hk_native_texture_descriptor){0},
                 sizeof(null10)) != 0);

   struct AGX_TEXTURE semantic = {
      AGX_TEXTURE_header,
      .dimension = AGX_TEXTURE_DIMENSION_2D,
      .layout = AGX_LAYOUT_GPU,
      .channels = AGX_CHANNELS_R8G8B8A8,
      .type = AGX_TEXTURE_TYPE_UNORM,
      .swizzle_r = AGX_CHANNEL_R,
      .swizzle_g = AGX_CHANNEL_G,
      .swizzle_b = AGX_CHANNEL_B,
      .swizzle_a = AGX_CHANNEL_A,
      .width = 256,
      .height = 128,
      .address = 0x80000,
      .compressed = true,
      .extended = true,
      .acceleration_buffer = 0xa0000,
   };
   struct agx_texture_packed legacy_packed;
   struct hk_native_texture_descriptor native_packed;
   AGX_TEXTURE_pack((uint32_t *)&legacy_packed, &semantic);
   assert(hk_pack_native_texture(8, &native_packed, &semantic));
   assert(memcmp(&legacy_packed, &native_packed, sizeof(legacy_packed)) == 0);
   for (unsigned i = sizeof(legacy_packed); i < sizeof(native_packed); ++i)
      assert(((const uint8_t *)&native_packed)[i] == 0);

   /* M5 native bytes are grounded in a live AGXG17XFamilyTexture readout. */
   static const uint8_t m5_texture[32] = {
      0x22, 0x0a, 0x88, 0xf6, 0x0f, 0xf8, 0x03, 0x20,
      0x00, 0x80, 0x00, 0x00, 0x10, 0x00, 0x00, 0x80,
      0x00, 0xa0, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   semantic.address = 0x10000080000ull;
   semantic.acceleration_buffer = 0x100000a0000ull;
   assert(hk_pack_native_texture(10, &native_packed, &semantic));
   assert(memcmp(&native_packed, m5_texture, sizeof(m5_texture)) == 0);

   enum agx_sample_count samples;
   assert(hk_translate_native_texture_sample_count(9, 2, &samples));
   assert(samples == AGX_SAMPLE_COUNT_2);
   assert(hk_translate_native_texture_sample_count(9, 4, &samples));
   assert(samples == AGX_SAMPLE_COUNT_4);
   assert(hk_translate_native_texture_sample_count(9, 8, &samples));
   assert(samples == AGX_SAMPLE_COUNT_8);
   assert(hk_translate_native_texture_sample_count(10, 8, &samples));
   assert(samples == AGX_SAMPLE_COUNT_8);
   assert(!hk_translate_native_texture_sample_count(10, 1, &samples));

   semantic.dimension = AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED;
   static const struct {
      enum agx_sample_count samples;
      unsigned selector;
   } gen9_msaa[] = {
      {AGX_SAMPLE_COUNT_2, 0},
      {AGX_SAMPLE_COUNT_4, 1},
      {AGX_SAMPLE_COUNT_8, 2},
   };
   for (unsigned i = 0; i < sizeof(gen9_msaa) / sizeof(gen9_msaa[0]); ++i) {
      semantic.samples = gen9_msaa[i].samples;
      assert(hk_pack_native_texture(9, &native_packed, &semantic));

      uint64_t gen9_msaa_word0;
      memcpy(&gen9_msaa_word0, native_packed.opaque,
             sizeof(gen9_msaa_word0));
      assert(((gen9_msaa_word0 >> 56) & 3) == gen9_msaa[i].selector);

      struct AGX_TEXTURE_GEN9 unpacked = {0};
      assert(AGX_TEXTURE_GEN9_unpack(NULL,
                                     (const uint8_t *)native_packed.opaque,
                                     &unpacked));
      assert(unpacked.samples == gen9_msaa[i].samples);
   }

   memset(&native_packed, 0xff, sizeof(native_packed));
   semantic.samples = (enum agx_sample_count)3;
   assert(!hk_pack_native_texture(9, &native_packed, &semantic));
   assert(memcmp(&native_packed, &(struct hk_native_texture_descriptor){0},
                 sizeof(native_packed)) == 0);
   memset(&native_packed, 0xff, sizeof(native_packed));
   assert(!hk_pack_native_texture(10, &native_packed, &semantic));
   assert(memcmp(&native_packed, &(struct hk_native_texture_descriptor){0},
                 sizeof(native_packed)) == 0);

   semantic.samples = AGX_SAMPLE_COUNT_8;
   assert(hk_pack_native_texture(10, &native_packed, &semantic));
   uint64_t m5_msaa_word0;
   memcpy(&m5_msaa_word0, native_packed.opaque, sizeof(m5_msaa_word0));
   assert(((m5_msaa_word0 >> 58) & 3) == AGX_SAMPLE_COUNT_8);

   semantic.dimension = AGX_TEXTURE_DIMENSION_2D;
   memset(&native_packed, 0xff, sizeof(native_packed));
   assert(!hk_pack_native_texture(9, &native_packed, &semantic));
   assert(memcmp(&native_packed, &(struct hk_native_texture_descriptor){0},
                 sizeof(native_packed)) == 0);
   memset(&native_packed, 0xff, sizeof(native_packed));
   assert(!hk_pack_native_texture(10, &native_packed, &semantic));
   assert(memcmp(&native_packed, &(struct hk_native_texture_descriptor){0},
                 sizeof(native_packed)) == 0);

   semantic.samples = AGX_SAMPLE_COUNT_2;
   semantic.buffer_offset_sw = 1;
   assert(!hk_pack_native_texture(10, &native_packed, &semantic));
   for (unsigned i = 0; i < sizeof(native_packed); ++i)
      assert(((const uint8_t *)&native_packed)[i] == 0);

   struct AGX_PBE pbe = {
      AGX_PBE_header,
      .dimension = AGX_TEXTURE_DIMENSION_2D,
      .layout = AGX_LAYOUT_GPU,
      .channels = AGX_CHANNELS_R8G8B8A8,
      .type = AGX_TEXTURE_TYPE_UNORM,
      .swizzle_r = AGX_CHANNEL_R,
      .swizzle_g = AGX_CHANNEL_G,
      .swizzle_b = AGX_CHANNEL_B,
      .swizzle_a = AGX_CHANNEL_A,
      .width = 256,
      .height = 128,
      .buffer = 0x80000,
      .compressed = true,
      .extended = true,
      .acceleration_buffer = 0xa0000,
   };
   struct hk_native_pbe_descriptor native_pbe;

   struct agx_pbe_packed legacy_pbe;
   AGX_PBE_pack(legacy_pbe.opaque, &pbe);
   assert(hk_pack_native_pbe(8, &native_pbe, &pbe));
   assert(memcmp(&legacy_pbe, &native_pbe, sizeof(legacy_pbe)) == 0);
   for (unsigned i = sizeof(legacy_pbe); i < sizeof(native_pbe); ++i)
      assert(((const uint8_t *)&native_pbe)[i] == 0);

   /* Honeykrisp's ordinary color-store subset uses the exact shared Gen9/10
    * word geometry and leaves all unresolved sideband fields zero.
    */
   static const uint8_t gen9_pbe[32] = {
      0x22, 0x0a, 0xe4, 0xff, 0xc0, 0x1f, 0x00, 0x00,
      0x00, 0x80, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   static const uint8_t gen10_pbe[32] = {
      0x22, 0x0a, 0xe4, 0xff, 0x80, 0x3f, 0x00, 0x00,
      0x00, 0x80, 0x00, 0x00, 0x10, 0x00, 0x00, 0x10,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   pbe.buffer = 0x10000080000ull;
   pbe.compressed = false;
   pbe.extended = false;
   pbe.acceleration_buffer = 0;
   pbe.page_aligned_layers = true;
   /* Allocator padding is not Gen9's single-layer PBE word1 bit60.
    * Gen10 and the legacy-generation bridge retain their original bytes.
    */
   assert(hk_pack_native_pbe(9, &native_pbe, &pbe));
   assert(memcmp(&native_pbe, gen9_pbe, sizeof(gen9_pbe)) == 0);
   assert(hk_pack_native_pbe(10, &native_pbe, &pbe));
   assert(memcmp(&native_pbe, gen10_pbe, sizeof(gen10_pbe)) == 0);
   assert(!hk_native_pbe_supported(10));
   pbe.unk_mipmapped = true;
   for (unsigned levels = 2; levels <= 16; ++levels) {
      pbe.levels = levels;
      for (unsigned level = 0; level < levels; ++level) {
         pbe.level = level;
         assert(hk_pack_native_pbe(9, &native_pbe, &pbe));
         uint8_t expected[32];
         memcpy(expected, gen9_pbe, sizeof(expected));
         expected[7] |= 4;
         expected[15] |= 128;
         expected[21] = level << 4;
         expected[22] = levels - 1;
         assert(memcmp(&native_pbe, expected, sizeof(expected)) == 0);
      }
   }
   pbe.unk_mipmapped = false;
   pbe.level = 0;
   pbe.levels = 1;


#define CHECK_PBE_REJECT(statement_)                                           \
   do {                                                                        \
      statement_;                                                              \
      memset(&native_pbe, 0xff, sizeof(native_pbe));                           \
      assert(!hk_pack_native_pbe(9, &native_pbe, &pbe));                       \
      assert(memcmp(&native_pbe, &(struct hk_native_pbe_descriptor){0},        \
                    sizeof(native_pbe)) == 0);                                 \
   } while (0)

   CHECK_PBE_REJECT(pbe.buffer |= 1);
   pbe.buffer = 0x10000080000ull;
   pbe.layout = AGX_LAYOUT_LINEAR;
   pbe.stride = 1020; /* 1024-byte row pitch in the legacy minus-4 ABI. */
   static const uint8_t linear_pbe[32] = {
      0x02, 0x0a, 0xe4, 0xff, 0xc0, 0x1f, 0x00, 0x00,
      0x00, 0x80, 0x00, 0x00, 0x10, 0xf0, 0x03, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   assert(hk_pack_native_pbe(9, &native_pbe, &pbe));
   assert(memcmp(&native_pbe, linear_pbe, sizeof(linear_pbe)) == 0);
   pbe.stride = 0;
   pbe.layout = AGX_LAYOUT_GPU;
   CHECK_PBE_REJECT(pbe.dimension = AGX_TEXTURE_DIMENSION_2D_ARRAY);
   pbe.dimension = AGX_TEXTURE_DIMENSION_2D;
   CHECK_PBE_REJECT(pbe.compressed = true);
   pbe.compressed = false;
   CHECK_PBE_REJECT(pbe.level = 1);
   pbe.level = 0;
   CHECK_PBE_REJECT(pbe.levels = 17);
   pbe.levels = 1;
   CHECK_PBE_REJECT(pbe.srgb = true);
   pbe.srgb = false;
   CHECK_PBE_REJECT(pbe.mode = AGX_IMAGE_MODE_SPARSE);
   pbe.mode = AGX_IMAGE_MODE_NORMAL;
   CHECK_PBE_REJECT(pbe.level_offset_sw = 0x80);
   pbe.level_offset_sw = 0;
   CHECK_PBE_REJECT(pbe.aligned_width_msaa_sw = 64);
   pbe.aligned_width_msaa_sw = 0;
   CHECK_PBE_REJECT(pbe.sample_count_log2_sw = 1);
   pbe.sample_count_log2_sw = 0;
   CHECK_PBE_REJECT(pbe.tile_width_sw = 16);
   pbe.tile_width_sw = 1;
   CHECK_PBE_REJECT(pbe.tile_height_sw = 16);
   pbe.tile_height_sw = 1;
   CHECK_PBE_REJECT(pbe.layer_stride_sw = 0x80);
   pbe.layer_stride_sw = 0;
   CHECK_PBE_REJECT(pbe.buffer_offset_sw = 1);

#undef CHECK_PBE_REJECT

   struct hk_native_pbe_descriptor null_pbe9, null_pbe10;
   assert(hk_set_null_native_pbe(9, &null_pbe9));
   assert(hk_set_null_native_pbe(10, &null_pbe10));
   assert(memcmp(&null_pbe9, &(struct hk_native_pbe_descriptor){0},
                 sizeof(null_pbe9)) != 0);
   assert(memcmp(&null_pbe10, &(struct hk_native_pbe_descriptor){0},
                 sizeof(null_pbe10)) != 0);

   puts("honeykrisp native descriptor geometry/policy: passed");
   return 0;
}
