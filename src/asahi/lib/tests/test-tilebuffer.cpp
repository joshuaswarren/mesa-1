/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "util/format/u_format.h"
#include "agx_tilebuffer.h"
#include "agx_bg_eot_key.h"

#include <gtest/gtest.h>

struct test {
   const char *name;
   uint8_t nr_samples;
   enum pipe_format formats[8];
   struct agx_tilebuffer_layout layout;
   uint32_t total_size;
};

/* clang-format off */
struct test tests[] = {
   {
      "Simple test",
      1,
      { PIPE_FORMAT_R8G8B8A8_UNORM },
      {
         ._offset_B = { 0 },
         .sample_size_B = 8,
         .nr_samples = 1,
         .tile_size = 32 * 32,
      },
      8192
   },
   {
      "MSAA 2x",
      2,
      { PIPE_FORMAT_R8G8B8A8_UNORM },
      {
         ._offset_B = { 0 },
         .sample_size_B = 8,
         .nr_samples = 2,
         .tile_size = 32 * 32,
      },
      16384
   },
   {
      "MSAA 4x",
      4,
      { PIPE_FORMAT_R8G8B8A8_UNORM },
      {
         ._offset_B = { 0 },
         .sample_size_B = 8,
         .nr_samples = 4,
         .tile_size = 32 * 16,
      },
      16384
   },
   {
      "MRT",
      1,
      {
         PIPE_FORMAT_R16_SINT,
         PIPE_FORMAT_R32G32_FLOAT,
         PIPE_FORMAT_R8_SINT,
         PIPE_FORMAT_R32G32_SINT,
      },
      {
         ._offset_B = { 16, 0, 18, 8 },
         .sample_size_B = 24,
         .nr_samples = 1,
         .tile_size = 32 * 32,
      },
      24576
   },
   {
      "MRT with MSAA 2x",
      2,
      {
         PIPE_FORMAT_R16_SINT,
         PIPE_FORMAT_R32G32_FLOAT,
         PIPE_FORMAT_R8_SINT,
         PIPE_FORMAT_R32G32_SINT,
      },
      {
         ._offset_B = { 16, 0, 18, 8 },
         .sample_size_B = 24,
         .nr_samples = 2,
         .tile_size = 32 * 16,
      },
      24576
   },
   {
      "MRT with MSAA 4x",
      4,
      {
         PIPE_FORMAT_R16_SINT,
         PIPE_FORMAT_R32G32_FLOAT,
         PIPE_FORMAT_R8_SINT,
         PIPE_FORMAT_R32G32_SINT,
      },
      {
         ._offset_B = { 16, 0, 18, 8 },
         .sample_size_B = 24,
         .nr_samples = 4,
         .tile_size = 16 * 16,
      },
      24576
   },
   {
      "MRT test requiring 2 alignment on the second RT",
      1,
      { PIPE_FORMAT_R8_UNORM, PIPE_FORMAT_R16G16_SNORM },
      {
         ._offset_B = { 4, 0 },
         .sample_size_B = 8,
         .nr_samples = 1,
         .tile_size = 32 * 32,
      },
      8192
   },
   {
      "Simple MRT test requiring 4 alignment on the second RT",
      1,
      { PIPE_FORMAT_R8_UNORM, PIPE_FORMAT_R10G10B10A2_UNORM },
      {
         ._offset_B = { 4, 0 },
         .sample_size_B = 8,
         .nr_samples = 1,
         .tile_size = 32 * 32,
      },
      8192
   },
   {
      "MRT test that requires spilling to consider alignment requirements",
      4,
      {
         PIPE_FORMAT_R32_FLOAT,
         PIPE_FORMAT_R32_FLOAT,
         PIPE_FORMAT_R32_FLOAT,
         PIPE_FORMAT_R32_FLOAT,
         PIPE_FORMAT_R32_FLOAT,
         PIPE_FORMAT_R32_FLOAT,
         PIPE_FORMAT_R32_FLOAT,
         PIPE_FORMAT_R32_FLOAT,
      },
      {
         .spilled = { false, false, false, false, false, false, true, true },
         ._offset_B = { 0, 4, 8, 12, 16, 20, 0, 0},
         .sample_size_B = 24,
         .nr_samples = 4,
         .tile_size = 16 * 16,
      },
      24576
   },

};
/* clang-format on */

TEST(Tilebuffer, Layouts)
{
   for (unsigned i = 0; i < ARRAY_SIZE(tests); ++i) {
      unsigned nr_cbufs;

      for (nr_cbufs = 0; nr_cbufs < ARRAY_SIZE(tests[i].formats) &&
                         tests[i].formats[nr_cbufs] != PIPE_FORMAT_NONE;
           ++nr_cbufs)
         ;

      struct agx_tilebuffer_layout actual = agx_build_tilebuffer_layout(
         tests[i].formats, nr_cbufs, tests[i].nr_samples, false);

      ASSERT_EQ(tests[i].layout.sample_size_B, actual.sample_size_B)
         << tests[i].name;
      ASSERT_EQ(tests[i].layout.nr_samples, actual.nr_samples) << tests[i].name;
      ASSERT_EQ(tests[i].layout.tile_size, actual.tile_size) << tests[i].name;
      ASSERT_EQ(tests[i].total_size, agx_tilebuffer_total_size(&tests[i].layout))
         << tests[i].name;

      for (unsigned j = 0; j < 8; ++j) {
         ASSERT_EQ(tests[i].layout._offset_B[j], actual._offset_B[j])
            << tests[i].name << ", render target " << j;
      }
   }
}

TEST(Tilebuffer, G17PDualRenderTargets)
{
   enum pipe_format wboit[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R16G16B16A16_FLOAT,
      PIPE_FORMAT_R8_UNORM,
   };
   struct agx_tilebuffer_layout tib =
      agx_build_tilebuffer_layout(wboit, 2, 1, false);
   EXPECT_EQ(agx_g17p_classify_mrt(&tib), AGX_G17P_MRT_WBOIT);
   EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&tib));
   EXPECT_EQ(agx_tilebuffer_offset_B(&tib, 0), 0u);
   EXPECT_EQ(agx_tilebuffer_offset_B(&tib, 1), 8u);

   enum pipe_format rgba8[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R8G8B8A8_UNORM,
      PIPE_FORMAT_R8G8B8A8_UNORM,
   };
   tib = agx_build_tilebuffer_layout(rgba8, 2, 1, false);
   EXPECT_EQ(agx_g17p_classify_mrt(&tib), AGX_G17P_MRT_RGBA8);
   EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&tib));
   EXPECT_EQ(agx_tilebuffer_offset_B(&tib, 1), 4u);

   tib = agx_build_tilebuffer_layout(wboit, 2, 2, false);
   EXPECT_EQ(agx_g17p_classify_mrt(&tib), AGX_G17P_MRT_NONE);
   EXPECT_FALSE(agx_g17p_rt0_tile_layout_supported(&tib));

   tib = agx_build_tilebuffer_layout(wboit, 2, 1, false);
   tib._offset_B[1] = 4;
   EXPECT_EQ(agx_g17p_classify_mrt(&tib), AGX_G17P_MRT_NONE);

   tib = agx_build_tilebuffer_layout(wboit, 2, 1, false);
   tib.spilled[1] = true;
   EXPECT_EQ(agx_g17p_classify_mrt(&tib), AGX_G17P_MRT_NONE);
   EXPECT_FALSE(agx_g17p_rt0_tile_layout_supported(&tib));

   enum pipe_format three_targets[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R16G16B16A16_FLOAT,
      PIPE_FORMAT_R8_UNORM,
      PIPE_FORMAT_R8_UNORM,
   };
   tib = agx_build_tilebuffer_layout(three_targets, 3, 1, false);
   EXPECT_EQ(agx_g17p_classify_mrt(&tib), AGX_G17P_MRT_NONE);
   EXPECT_FALSE(agx_g17p_rt0_tile_layout_supported(&tib));

   EXPECT_EQ(agx_g17p_classify_mrt(nullptr), AGX_G17P_MRT_NONE);
}

TEST(Tilebuffer, G17PRGB565EOTKey)
{
   struct agx_shader_key key = {};

   /* Seed an unrelated format first: configuring a new EOT format must clear
    * mutually exclusive selectors rather than inheriting a stale cache key. */
   agx_g17p_configure_rt0_eot_key(PIPE_FORMAT_R32G32_UINT, &key);
   EXPECT_EQ(key.g17p_rt0_eot_words, 2);
   EXPECT_TRUE(key.g17p_rt0_eot_i32);
   EXPECT_TRUE(key.g17p_rt0_eot_i32_selector_75);

   for (enum pipe_format format : {PIPE_FORMAT_R5G6B5_UNORM,
                                   PIPE_FORMAT_B5G6R5_UNORM}) {
      agx_g17p_configure_rt0_eot_key(format, &key);
      EXPECT_EQ(key.g17p_rt0_eot_words, 2);
      EXPECT_TRUE(key.g17p_rt0_eot_rgb565);
      EXPECT_FALSE(key.g17p_rt0_eot_i32);
      EXPECT_FALSE(key.g17p_rt0_eot_i32_selector_75);
      EXPECT_FALSE(key.g17p_rt0_eot_norm16_selector_95);
   }

   /* RGBA16F has the same physical format and word count, but Apple's native
    * builtin gives it selector 0x15 rather than RGB565's selector 0x35. */
   agx_g17p_configure_rt0_eot_key(PIPE_FORMAT_R16G16B16A16_FLOAT, &key);
   EXPECT_EQ(key.g17p_rt0_eot_words, 2);
   EXPECT_FALSE(key.g17p_rt0_eot_rgb565);

   agx_g17p_configure_rt0_eot_key(PIPE_FORMAT_R8G8B8A8_UNORM, &key);
   EXPECT_EQ(key.g17p_rt0_eot_words, 0);
   EXPECT_FALSE(key.g17p_rt0_eot_rgb565);
}

TEST(Tilebuffer, G17PPacked16EOTKey)
{
   struct agx_shader_key key = {};
   const enum pipe_format packed[] = {
      PIPE_FORMAT_R5G5B5A1_UNORM,
      PIPE_FORMAT_B5G5R5A1_UNORM,
      PIPE_FORMAT_R5G5B5X1_UNORM,
      PIPE_FORMAT_B5G5R5X1_UNORM,
      PIPE_FORMAT_R4G4B4A4_UNORM,
      PIPE_FORMAT_B4G4R4A4_UNORM,
      PIPE_FORMAT_A4B4G4R4_UNORM,
      PIPE_FORMAT_A4R4G4B4_UNORM,
   };

   for (enum pipe_format format : packed) {
      enum pipe_format formats[AGX_MAX_RENDER_TARGETS] = {format};
      struct agx_tilebuffer_layout tib =
         agx_build_tilebuffer_layout(formats, 1, 1, false);

      EXPECT_EQ(agx_tilebuffer_physical_format(&tib, 0),
                PIPE_FORMAT_R16_FLOAT);
      EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&tib));

      agx_g17p_configure_rt0_eot_key(format, &key);
      EXPECT_EQ(key.g17p_rt0_eot_words, 2);
      EXPECT_FALSE(key.g17p_rt0_eot_rgb565);
      EXPECT_FALSE(key.g17p_rt0_eot_i32);
      EXPECT_FALSE(key.g17p_rt0_eot_norm16_selector_95);
   }
}
