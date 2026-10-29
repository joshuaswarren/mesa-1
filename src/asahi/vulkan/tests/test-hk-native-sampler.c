/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/* Every check in this test is an assert(); keep them in NDEBUG builds. */
#undef NDEBUG

#include "hk_native_sampler.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static const struct hk_native_sampler_state base = {
   .minimum_lod = 1.0f,
   .maximum_lod = 100.0f,
   .lod_bias = -1.015625f,
   .maximum_anisotropy = 8,
   .magnify = AGX_FILTER_LINEAR,
   .minify = AGX_FILTER_NEAREST,
   .mip_filter = AGX_MIP_FILTER_LINEAR,
   .wrap_s = AGX_WRAP_REPEAT,
   .wrap_t = AGX_WRAP_MIRRORED_REPEAT,
   .wrap_r = AGX_WRAP_CLAMP_TO_BORDER,
   .compare_func = AGX_COMPARE_FUNC_LEQUAL,
   .border_colour = AGX_BORDER_COLOUR_OPAQUE_WHITE,
   .reduction_mode = HK_SAMPLER_REDUCTION_WEIGHTED_AVERAGE,
   .pixel_coordinates = true,
   .compare_enable = true,
   .seamful_cube_maps = true,
};

static void
check_chip(enum agx_chip chip, enum agx_sampler_generation generation,
           unsigned stride)
{
   assert(agx_chip_sampler_generation(chip) == generation);
   assert(agx_chip_sampler_heap_element_size(chip) == stride);

   struct hk_native_sampler_descriptor packed;
   const bool supported = generation != AGX_SAMPLER_GENERATION_UNKNOWN;
   assert(hk_pack_native_sampler(chip, &packed, &base) == supported);
   if (!supported) {
      assert(memcmp(&packed, &(struct hk_native_sampler_descriptor){0},
                    sizeof(packed)) == 0);
      return;
   }

   for (unsigned i = stride; i < sizeof(packed); ++i)
      assert(((const uint8_t *)&packed)[i] == 0);
   if (stride == 32) {
      for (unsigned i = 8; i < stride; ++i)
         assert(((const uint8_t *)&packed)[i] == 0);
   }
}

/* Full sampler words from the native HAL300 Gen5 constructor at
 * macOS 26A428, with normalized
 * coordinates, linear min/mag filters, no mip filter and clamp-to-edge.
 * Exercise reduction alongside both LOD bias signs and anisotropy. */
static void
check_native_minmax(void)
{
   static const struct {
      unsigned reduction, anisotropy;
      float bias;
      uint64_t word;
   } cases[] = {
      {1, 1, 0.00f, UINT64_C(0x00000080028e0000)},
      {1, 4, 0.00f, UINT64_C(0x0000008002ae0000)},
      {1, 16, 0.00f, UINT64_C(0x0000008002ce0000)},
      {1, 1, 1.50f, UINT64_C(0x00030080028e0000)},
      {1, 4, 1.50f, UINT64_C(0x0003008002ae0000)},
      {1, 16, 1.50f, UINT64_C(0x0003008002ce0000)},
      {1, 1, -2.25f, UINT64_C(0x003b8080028e0000)},
      {1, 4, -2.25f, UINT64_C(0x003b808002ae0000)},
      {1, 16, -2.25f, UINT64_C(0x003b808002ce0000)},
      {2, 1, 0.00f, UINT64_C(0x00000100028e0000)},
      {2, 4, 0.00f, UINT64_C(0x0000010002ae0000)},
      {2, 16, 0.00f, UINT64_C(0x0000010002ce0000)},
      {2, 1, 1.50f, UINT64_C(0x00030100028e0000)},
      {2, 4, 1.50f, UINT64_C(0x0003010002ae0000)},
      {2, 16, 1.50f, UINT64_C(0x0003010002ce0000)},
      {2, 1, -2.25f, UINT64_C(0x003b8100028e0000)},
      {2, 4, -2.25f, UINT64_C(0x003b810002ae0000)},
      {2, 16, -2.25f, UINT64_C(0x003b810002ce0000)},
   };
   for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
      struct hk_native_sampler_state state = {
         .maximum_lod = 14.0f,
         .maximum_anisotropy = cases[i].anisotropy,
         .lod_bias = cases[i].bias,
         .magnify = AGX_FILTER_LINEAR,
         .minify = AGX_FILTER_LINEAR,
         .reduction_mode = (enum hk_sampler_reduction_mode)cases[i].reduction,
      };
      struct hk_native_sampler_descriptor out;
      assert(hk_pack_native_sampler(AGX_CHIP_G17G, &out, &state));
      uint64_t word;
      memcpy(&word, out.opaque, sizeof(word));
      assert(word == cases[i].word);

      /* hk_nir_lower_descriptors' Gen10 compute LOD query reads the
       * descriptor's copy of this word: bias is signed 5:6 at bits 43..53
       * of the high half (bits 11..21), and max LOD is 4:6 at bits 10..19. */
      const uint32_t high = (uint32_t)(word >> 32);
      assert((int32_t)(high << 10) >> 21 == (int32_t)(cases[i].bias * 64));
      assert(((uint32_t)word >> 10 & 0x3ff) == 14 * 64);
   }
}

int
main(void)
{
   check_native_minmax();
   check_chip(AGX_CHIP_G13G, AGX_SAMPLER_GENERATION_GEN4, 8);
   check_chip(AGX_CHIP_G13X, AGX_SAMPLER_GENERATION_GEN4, 8);
   check_chip(AGX_CHIP_G14G, AGX_SAMPLER_GENERATION_GEN4, 8);
   check_chip(AGX_CHIP_G14X, AGX_SAMPLER_GENERATION_GEN4, 8);
   check_chip(AGX_CHIP_G15P, AGX_SAMPLER_GENERATION_UNKNOWN, 0);
   check_chip(AGX_CHIP_G15G, AGX_SAMPLER_GENERATION_GEN4, 32);
   check_chip(AGX_CHIP_G15S, AGX_SAMPLER_GENERATION_GEN4, 32);
   check_chip(AGX_CHIP_G15C, AGX_SAMPLER_GENERATION_GEN4, 32);
   check_chip(AGX_CHIP_G15D, AGX_SAMPLER_GENERATION_GEN4, 32);
   check_chip(AGX_CHIP_G16P, AGX_SAMPLER_GENERATION_GEN4_HAL200, 32);
   check_chip(AGX_CHIP_G16G, AGX_SAMPLER_GENERATION_GEN4_HAL200, 32);
   check_chip(AGX_CHIP_G16S, AGX_SAMPLER_GENERATION_GEN4_HAL200, 32);
   check_chip(AGX_CHIP_G16C, AGX_SAMPLER_GENERATION_GEN4_HAL200, 32);
   check_chip(AGX_CHIP_G17P, AGX_SAMPLER_GENERATION_GEN4_HAL200, 32);
   check_chip(AGX_CHIP_G17A, AGX_SAMPLER_GENERATION_GEN4_HAL200, 32);
   check_chip(AGX_CHIP_G17G, AGX_SAMPLER_GENERATION_GEN5, 32);
   check_chip(AGX_CHIP_G17S, AGX_SAMPLER_GENERATION_GEN5, 32);
   check_chip(AGX_CHIP_G17C, AGX_SAMPLER_GENERATION_GEN5, 32);

   struct hk_native_sampler_descriptor gen4, hal200, g17p, gen5;
   assert(hk_pack_native_sampler(AGX_CHIP_G14G, &gen4, &base));
   assert(hk_pack_native_sampler(AGX_CHIP_G16G, &hal200, &base));
   assert(hk_pack_native_sampler(AGX_CHIP_G17P, &g17p, &base));
   assert(hk_pack_native_sampler(AGX_CHIP_G17G, &gen5, &base));

   static const uint8_t expected_gen4[8] = {
      0x40, 0x00, 0xbe, 0x30, 0x5a, 0x04, 0x00, 0x03,
   };
   static const uint8_t expected_hal200[8] = {
      0x40, 0x00, 0xbe, 0x30, 0x5a, 0xfc, 0x3d, 0xc0,
   };
   static const uint8_t expected_gen5[8] = {
      0x40, 0x00, 0xbf, 0x30, 0x5a, 0xfc, 0x3d, 0xc0,
   };
   assert(memcmp(&gen4, expected_gen4, sizeof(expected_gen4)) == 0);
   assert(memcmp(&hal200, expected_hal200, sizeof(expected_hal200)) == 0);
   assert(memcmp(&g17p, expected_hal200, sizeof(expected_hal200)) == 0);
   for (unsigned i = sizeof(expected_hal200); i < sizeof(g17p); ++i)
      assert(((const uint8_t *)&g17p)[i] == 0);
   assert(memcmp(&gen5, expected_gen5, sizeof(expected_gen5)) == 0);
   {
      /* The fields the Gen10 compute LOD query decodes from this word. */
      const uint32_t lo = gen5.opaque[0], hi = gen5.opaque[1];
      assert((lo & 0x3ff) == 64);                             /* min LOD 1 */
      assert((lo >> 10 & 0x3ff) == 15 * 64);       /* max LOD 100 -> 15 */
      assert((lo >> 27 & 3) == AGX_MIP_FILTER_LINEAR);
      assert((int32_t)(hi << 10) >> 21 == -65);     /* bias -1.015625 */
   }

   /* A disabled compare must retain the exact mature Gen4 bytes: compare
    * function bits were historically left at their header default. */
   struct hk_native_sampler_state noncompare = base;
   noncompare.compare_enable = false;
   noncompare.compare_func = AGX_COMPARE_FUNC_ALWAYS;
   assert(hk_pack_native_sampler(AGX_CHIP_G14G, &gen4, &noncompare));
   struct AGX_SAMPLER legacy = {
      .minimum_lod = noncompare.minimum_lod,
      .maximum_lod = noncompare.maximum_lod,
      .maximum_anisotropy = noncompare.maximum_anisotropy,
      .magnify = noncompare.magnify,
      .minify = noncompare.minify,
      .mip_filter = noncompare.mip_filter,
      .wrap_s = noncompare.wrap_s,
      .wrap_t = noncompare.wrap_t,
      .wrap_r = noncompare.wrap_r,
      .pixel_coordinates = noncompare.pixel_coordinates,
      .border_colour = noncompare.border_colour,
      .seamful_cube_maps = noncompare.seamful_cube_maps,
   };
   uint32_t legacy_packed[AGX_SAMPLER_LENGTH / sizeof(uint32_t)] = {0};
   AGX_SAMPLER_pack(legacy_packed, &legacy);
   assert(memcmp(&gen4, legacy_packed, sizeof(legacy_packed)) == 0);

   assert(hk_sampler_reduction_supported(AGX_SAMPLER_GENERATION_GEN5,
                                         HK_SAMPLER_REDUCTION_MINIMUM));
   assert(!hk_sampler_reduction_supported(AGX_SAMPLER_GENERATION_GEN4,
                                          HK_SAMPLER_REDUCTION_MINIMUM));
   struct hk_native_sampler_state reduced = base;
   reduced.compare_enable = false;
   reduced.reduction_mode = HK_SAMPLER_REDUCTION_MINIMUM;
   assert(hk_pack_native_sampler(AGX_CHIP_G17G, &gen5, &reduced));
   assert(((gen5.opaque[1] >> 7) & 3) ==
          AGX_SAMPLER_REDUCTION_MODE_MINIMUM);
   assert(!hk_pack_native_sampler(AGX_CHIP_G16G, &hal200, &reduced));

   struct hk_native_sampler_descriptor txf;
   assert(hk_pack_native_txf_sampler(AGX_CHIP_G14G, &txf));
   assert(hk_pack_native_txf_sampler(AGX_CHIP_G16G, &txf));
   assert(hk_pack_native_txf_sampler(AGX_CHIP_G17G, &txf));
   assert(!hk_pack_native_txf_sampler(AGX_CHIP_G15P, &txf));

   puts("honeykrisp native sampler matrix/packing: passed");
   return 0;
}
