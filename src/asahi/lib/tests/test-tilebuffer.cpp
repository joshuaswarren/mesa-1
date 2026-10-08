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

TEST(Tilebuffer, G17PUnorm8MultisampleLayouts)
{
   /* The allocator keeps a minimal MSAA slot even without colour output. */
   const enum pipe_format none = PIPE_FORMAT_NONE;
   for (unsigned samples : {2u, 4u}) {
      auto tib = agx_build_tilebuffer_layout(&none, 0, samples, false);
      EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&tib));
      EXPECT_EQ(tib.sample_size_B, 8u);
      tib.sample_size_B = 16;
      EXPECT_FALSE(agx_g17p_rt0_tile_layout_supported(&tib));
   }

   const enum pipe_format formats[] = {
      PIPE_FORMAT_R8_UNORM,
      PIPE_FORMAT_R8G8_UNORM,
      PIPE_FORMAT_R8G8B8A8_UNORM,
   };
   for (enum pipe_format format : formats) {
      for (unsigned samples : {2u, 4u}) {
         auto tib = agx_build_tilebuffer_layout(&format, 1, samples, false);
         EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&tib));
         EXPECT_EQ(agx_tilebuffer_physical_format(&tib, 0), PIPE_FORMAT_R8_UNORM);
         EXPECT_EQ(tib.sample_size_B, 8u);
         EXPECT_EQ(agx_tilebuffer_offset_B(&tib, 0), 0u);
         EXPECT_LT(agx_tilebuffer_total_size(&tib), 32768u);
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

   enum pipe_format rgba8_aliases[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R8G8B8X8_UNORM,
      PIPE_FORMAT_B8G8R8A8_UNORM,
   };
   tib = agx_build_tilebuffer_layout(rgba8_aliases, 2, 1, false);
   EXPECT_EQ(agx_g17p_classify_mrt(&tib), AGX_G17P_MRT_RGBA8);
   EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&tib));
   EXPECT_EQ(agx_tilebuffer_offset_B(&tib, 1), 4u);

   /* Multisample layouts take the mixed-format programs. */
   tib = agx_build_tilebuffer_layout(wboit, 2, 2, false);
   EXPECT_EQ(agx_g17p_classify_mrt(&tib), AGX_G17P_MRT_MIXED);
   EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&tib));

   tib = agx_build_tilebuffer_layout(wboit, 2, 1, false);
   tib._offset_B[1] = 4;
   EXPECT_EQ(agx_g17p_classify_mrt(&tib), AGX_G17P_MRT_NONE);

   tib = agx_build_tilebuffer_layout(wboit, 2, 1, false);
   tib.spilled[1] = true;
   EXPECT_EQ(agx_g17p_classify_mrt(&tib), AGX_G17P_MRT_NONE);
   EXPECT_FALSE(agx_g17p_rt0_tile_layout_supported(&tib));

   /* A third target of another format uses the mixed-format layout. */
   enum pipe_format three_targets[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R16G16B16A16_FLOAT,
      PIPE_FORMAT_R8_UNORM,
      PIPE_FORMAT_R8_UNORM,
   };
   tib = agx_build_tilebuffer_layout(three_targets, 3, 1, false);
   EXPECT_EQ(agx_g17p_classify_mrt(&tib), AGX_G17P_MRT_MIXED);
   EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&tib));

   EXPECT_EQ(agx_g17p_classify_mrt(nullptr), AGX_G17P_MRT_NONE);
}

TEST(Tilebuffer, G17PCompactsSparseRGBA8Attachments)
{
   enum pipe_format sparse[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_NONE,
      PIPE_FORMAT_R8G8B8A8_UNORM,
      PIPE_FORMAT_NONE,
      PIPE_FORMAT_B8G8R8A8_UNORM,
   };
   const struct agx_tilebuffer_layout source =
      agx_build_tilebuffer_layout(sparse, 4, 1, false);
   struct agx_tilebuffer_layout compact;
   uint8_t map[AGX_MAX_RENDER_TARGETS] = {0};
   ASSERT_EQ(agx_tilebuffer_compact(&source, &compact, map), 2u);
   EXPECT_EQ(map[0], 1u);
   EXPECT_EQ(map[1], 3u);
   EXPECT_EQ(compact.logical_format[0], PIPE_FORMAT_R8G8B8A8_UNORM);
   EXPECT_EQ(compact.logical_format[1], PIPE_FORMAT_B8G8R8A8_UNORM);
   EXPECT_EQ(compact._offset_B[0], 0u);
   EXPECT_EQ(compact._offset_B[1], 4u);
   EXPECT_EQ(agx_g17p_classify_mrt(&source), AGX_G17P_MRT_RGBA8);
   EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&source));

   for (unsigned count : {3u, 4u}) {
      enum pipe_format rgba8[AGX_MAX_RENDER_TARGETS] = {
         PIPE_FORMAT_R8G8B8A8_UNORM,
         PIPE_FORMAT_R8G8B8A8_UNORM,
         PIPE_FORMAT_R8G8B8A8_UNORM,
         PIPE_FORMAT_R8G8B8A8_UNORM,
      };
      struct agx_tilebuffer_layout tib =
         agx_build_tilebuffer_layout(rgba8, count, 1, false);
      EXPECT_EQ(agx_g17p_classify_mrt(&tib), AGX_G17P_MRT_RGBA8);
      EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&tib));
   }

   enum pipe_format none[AGX_MAX_RENDER_TARGETS] = {PIPE_FORMAT_NONE};
   struct agx_tilebuffer_layout attachmentless =
      agx_build_tilebuffer_layout(none, 1, 0, false);
   EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&attachmentless));
   EXPECT_EQ(attachmentless.nr_samples, 0u);
}

TEST(Tilebuffer, G17PNarrowSRGB)
{
   for (enum pipe_format format : {PIPE_FORMAT_R8_SRGB,
                                   PIPE_FORMAT_R8G8_SRGB}) {
      enum pipe_format formats[AGX_MAX_RENDER_TARGETS] = {format};
      struct agx_tilebuffer_layout tib =
         agx_build_tilebuffer_layout(formats, 1, 1, false);

      EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&tib));
      EXPECT_EQ(tib.sample_size_B, 8);
      EXPECT_EQ(agx_tilebuffer_offset_B(&tib, 0), 0);
      EXPECT_EQ(agx_tilebuffer_physical_format(&tib, 0),
                PIPE_FORMAT_R8G8B8A8_SRGB);
      EXPECT_EQ(agx_g17p_rt0_pbe_format(format), util_format_linear(format));

      tib.spilled[0] = true;
      EXPECT_FALSE(agx_g17p_rt0_tile_layout_supported(&tib));

      tib = agx_build_tilebuffer_layout(formats, 1, 1, true);
      EXPECT_FALSE(agx_g17p_rt0_tile_layout_supported(&tib));

      /* These narrow sRGB targets share the R8G8B8A8_SRGB physical tile, and
       * KHR-GL30.framebuffer_blit.multisampled_to_singlesampled_blit_color_config
       * exercises that tile at four samples through GL_SRGB8_ALPHA8. */
      for (unsigned samples : {2u, 4u}) {
         tib = agx_build_tilebuffer_layout(formats, 1, samples, false);
         EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&tib));
      }

      formats[1] = format;
      tib = agx_build_tilebuffer_layout(formats, 2, 1, false);
      EXPECT_EQ(agx_g17p_classify_mrt(&tib), AGX_G17P_MRT_MIXED);
      EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&tib));
   }
}

TEST(Tilebuffer, G17PAllRGBA8AttachmentMasksUseCompactStorage)
{
   for (unsigned mask = 1; mask < 256; ++mask) {
      SCOPED_TRACE(mask);
      enum pipe_format formats[AGX_MAX_RENDER_TARGETS] = {};
      for (unsigned rt = 0; rt < AGX_MAX_RENDER_TARGETS; ++rt) {
         if (mask & (1u << rt))
            formats[rt] = PIPE_FORMAT_R8G8B8A8_UNORM;
      }
      auto tib = agx_build_tilebuffer_layout(formats, 8, 1, false);
      EXPECT_EQ(agx_g17p_rgba8_mrt_mask(&tib), mask);
      EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&tib));
      unsigned active = 0, last = 0;
      for (unsigned rt = 0; rt < AGX_MAX_RENDER_TARGETS; ++rt) {
         if (!(mask & (1u << rt)))
            continue;
         EXPECT_EQ(agx_tilebuffer_offset_B(&tib, rt), 4 * active++);
         EXPECT_LT(agx_tilebuffer_offset_B(&tib, rt), tib.sample_size_B);
         last = rt;
      }
      tib._offset_B[last] += 4;
      EXPECT_EQ(agx_g17p_rgba8_mrt_mask(&tib), 0u);
   }
   EXPECT_EQ(agx_g17p_rgba8_mrt_mask(nullptr), 0u);
}

TEST(Tilebuffer, G17PDenseWideRGBA8RenderTargets)
{
   enum pipe_format formats[AGX_MAX_RENDER_TARGETS];
   for (auto &format : formats)
      format = PIPE_FORMAT_R8G8B8A8_UNORM;

   for (unsigned count = 3; count <= AGX_MAX_RENDER_TARGETS; ++count) {
      auto tib = agx_build_tilebuffer_layout(formats, count, 1, false);
      EXPECT_EQ(agx_g17p_wide_rgba8_mrt_count(&tib), count);
      EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&tib));
      EXPECT_EQ(tib.sample_size_B, ((4 * count + 7) & ~7u));
      EXPECT_EQ(tib.tile_size, count >= 7 ? 32 * 16 : 32 * 32);
      for (unsigned rt = 0; rt < count; ++rt)
         EXPECT_EQ(agx_tilebuffer_offset_B(&tib, rt), 4 * rt);

      auto invalid = tib;
      invalid.logical_format[1] = PIPE_FORMAT_NONE;
      EXPECT_EQ(agx_g17p_wide_rgba8_mrt_count(&invalid), 0u);
      invalid = tib;
      invalid._offset_B[count - 1] += 4;
      EXPECT_EQ(agx_g17p_wide_rgba8_mrt_count(&invalid), 0u);
      invalid = tib;
      invalid.spilled[count - 1] = true;
      EXPECT_EQ(agx_g17p_wide_rgba8_mrt_count(&invalid), 0u);
      invalid = tib;
      invalid.layered = true;
      EXPECT_EQ(agx_g17p_wide_rgba8_mrt_count(&invalid), 0u);
      invalid = tib;
      invalid.nr_samples = 2;
      EXPECT_EQ(agx_g17p_wide_rgba8_mrt_count(&invalid), 0u);
   }
   EXPECT_EQ(agx_g17p_wide_rgba8_mrt_count(nullptr), 0u);
}

TEST(Tilebuffer, G17PRGB565EOTKey)
{
   struct agx_shader_key key = {};

   /* Seed an unrelated format first: configuring a new EOT format must clear
    * the I32 admission rather than inheriting a stale cache key. */
   agx_g17p_configure_rt0_eot_key(PIPE_FORMAT_R32G32_UINT, &key);
   EXPECT_EQ(key.g17p_rt0_eot_words, 2);
   EXPECT_TRUE(key.g17p_rt0_eot_i32);

   /* RGB565 keeps RGBA16F's two 16-bit float words and its selector. */
   for (enum pipe_format format : {PIPE_FORMAT_R5G6B5_UNORM,
                                   PIPE_FORMAT_B5G6R5_UNORM,
                                   PIPE_FORMAT_R16G16B16A16_FLOAT}) {
      agx_g17p_configure_rt0_eot_key(format, &key);
      EXPECT_EQ(key.g17p_rt0_eot_words, 2);
      EXPECT_FALSE(key.g17p_rt0_eot_i32);
      EXPECT_EQ(agx_g17p_mrt_eot_selector(format), 0x15);
   }

   agx_g17p_configure_rt0_eot_key(PIPE_FORMAT_R8G8B8A8_UNORM, &key);
   EXPECT_EQ(key.g17p_rt0_eot_words, 0);
}

TEST(Tilebuffer, G17PPacked16EOTKey)
{
   struct agx_shader_key key = {};
   const enum pipe_format packed[] = {
      /* Vulkan R5G5B5A1/B5G5R5A1 use these low-alpha Gallium forms. */
      PIPE_FORMAT_A1B5G5R5_UNORM,
      PIPE_FORMAT_A1R5G5B5_UNORM,
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
      EXPECT_FALSE(key.g17p_rt0_eot_i32);
   }
}

TEST(Tilebuffer, G17PSpecialFormatEOTKey)
{
   static const struct {
      enum pipe_format logical;
      enum pipe_format physical;
   } formats[] = {
      {PIPE_FORMAT_R16G16B16X16_FLOAT, PIPE_FORMAT_R16_FLOAT},
      {PIPE_FORMAT_R16G16B16X16_UINT, PIPE_FORMAT_R16_UINT},
      {PIPE_FORMAT_R16G16B16X16_SINT, PIPE_FORMAT_R16_UINT},
   };

   for (const auto &format : formats) {
      enum pipe_format rt[AGX_MAX_RENDER_TARGETS] = {format.logical};
      struct agx_tilebuffer_layout tib =
         agx_build_tilebuffer_layout(rt, 1, 1, false);
      EXPECT_EQ(tib.sample_size_B, 8);
      EXPECT_EQ(agx_tilebuffer_physical_format(&tib, 0), format.physical);
      EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&tib));

      struct agx_shader_key key = {};
      agx_g17p_configure_rt0_eot_key(format.logical, &key);
      EXPECT_EQ(key.g17p_rt0_eot_words, 2);
      EXPECT_EQ(agx_g17p_mrt_eot_selector(format.logical), 0x15);

      /* Multisample admission follows the physical tilebuffer format, and
       * R16_FLOAT is one of the families the four-sample hardware sweep
       * covers. R16_UINT is not. */
      tib = agx_build_tilebuffer_layout(rt, 1, 2, false);
      EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&tib));
   }

   enum pipe_format packed_float[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R11G11B10_FLOAT,
   };
   struct agx_tilebuffer_layout tib =
      agx_build_tilebuffer_layout(packed_float, 1, 1, false);
   EXPECT_EQ(tib.sample_size_B, 8);
   EXPECT_EQ(agx_tilebuffer_physical_format(&tib, 0),
             PIPE_FORMAT_R11G11B10_FLOAT);
   EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&tib));

   struct agx_shader_key key = {};
   agx_g17p_configure_rt0_eot_key(PIPE_FORMAT_R11G11B10_FLOAT, &key);
   EXPECT_EQ(key.g17p_rt0_eot_words, 1);
   EXPECT_TRUE(key.g17p_rt0_eot_i32);
   EXPECT_EQ(agx_g17p_rt0_pbe_format(PIPE_FORMAT_R11G11B10_FLOAT),
             PIPE_FORMAT_R32_UINT);
   EXPECT_EQ(agx_g17p_rt0_pbe_format(PIPE_FORMAT_R8G8B8A8_SRGB),
             PIPE_FORMAT_R8G8B8A8_UNORM);
   EXPECT_EQ(agx_g17p_rt0_pbe_format(PIPE_FORMAT_R10G10B10A2_UNORM),
             PIPE_FORMAT_R10G10B10A2_UNORM);

   /* The packed float tile is eight bytes per sample at any sample count, and
    * KHR-GL30.framebuffer_blit.multisampled_to_singlesampled_blit_color_config
    * exercises it at four samples through GL_R11F_G11F_B10F. Layering is a
    * separate contract and stays refused. */
   tib = agx_build_tilebuffer_layout(packed_float, 1, 2, false);
   EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&tib));
   tib = agx_build_tilebuffer_layout(packed_float, 1, 1, true);
   EXPECT_FALSE(agx_g17p_rt0_tile_layout_supported(&tib));

   /* RGB9E5 is packed natively into the same raw tile word. */
   enum pipe_format rgb9e5[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R9G9B9E5_FLOAT,
   };
   tib = agx_build_tilebuffer_layout(rgb9e5, 1, 1, false);
   EXPECT_EQ(tib.sample_size_B, 8);
   EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&tib));
   key = {};
   agx_g17p_configure_rt0_eot_key(PIPE_FORMAT_R9G9B9E5_FLOAT, &key);
   EXPECT_EQ(key.g17p_rt0_eot_words, 1);
   EXPECT_TRUE(key.g17p_rt0_eot_i32);
   EXPECT_EQ(agx_g17p_rt0_pbe_format(PIPE_FORMAT_R9G9B9E5_FLOAT),
             PIPE_FORMAT_R32_UINT);
   tib = agx_build_tilebuffer_layout(rgb9e5, 1, 4, false);
   EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&tib));
   tib = agx_build_tilebuffer_layout(rgb9e5, 1, 1, true);
   EXPECT_FALSE(agx_g17p_rt0_tile_layout_supported(&tib));
}

TEST(Tilebuffer, G17PRGBX8IntegerEOTKey)
{
   const enum pipe_format formats[] = {
      PIPE_FORMAT_R8G8B8X8_UINT,
      PIPE_FORMAT_R8G8B8A8_UINT,
      PIPE_FORMAT_B8G8R8X8_UINT,
      PIPE_FORMAT_B8G8R8A8_UINT,
      PIPE_FORMAT_R8G8B8X8_SINT,
      PIPE_FORMAT_R8G8B8A8_SINT,
      PIPE_FORMAT_B8G8R8X8_SINT,
      PIPE_FORMAT_B8G8R8A8_SINT,
   };

   for (enum pipe_format format : formats) {
      enum pipe_format rt[AGX_MAX_RENDER_TARGETS] = {format};
      struct agx_tilebuffer_layout tib =
         agx_build_tilebuffer_layout(rt, 1, 1, false);
      EXPECT_EQ(tib.sample_size_B, 8);
      EXPECT_EQ(agx_tilebuffer_physical_format(&tib, 0),
                PIPE_FORMAT_R8_UINT);
      EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&tib));

      struct agx_shader_key key = {};
      agx_g17p_configure_rt0_eot_key(format, &key);
      EXPECT_EQ(key.g17p_rt0_eot_words, 1);
      EXPECT_FALSE(key.g17p_rt0_eot_i32);
   }
}

TEST(Tilebuffer, G17PRGBX8Multisample)
{
   for (enum pipe_format format : {PIPE_FORMAT_R8G8B8X8_UNORM,
                                   PIPE_FORMAT_B8G8R8X8_UNORM}) {
      enum pipe_format rt[AGX_MAX_RENDER_TARGETS] = {format};
      for (unsigned samples : {2u, 4u}) {
         struct agx_tilebuffer_layout tib =
            agx_build_tilebuffer_layout(rt, 1, samples, false);
         EXPECT_EQ(tib.sample_size_B, 8);
         EXPECT_EQ(agx_tilebuffer_physical_format(&tib, 0),
                   PIPE_FORMAT_R8_UNORM);
         EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&tib));
      }
   }
}

TEST(Tilebuffer, G17PRGB32EOTKey)
{
   static const struct {
      enum pipe_format logical;
      bool signed_format;
   } formats[] = {
      {PIPE_FORMAT_R32G32B32X32_FLOAT, false},
      {PIPE_FORMAT_R32G32B32X32_UINT, false},
      {PIPE_FORMAT_R32G32B32X32_SINT, true},
   };

   for (const auto &format : formats) {
      enum pipe_format rt[AGX_MAX_RENDER_TARGETS] = {format.logical};
      struct agx_tilebuffer_layout tib =
         agx_build_tilebuffer_layout(rt, 1, 1, false);
      EXPECT_EQ(tib.sample_size_B, 16);
      EXPECT_EQ(agx_tilebuffer_physical_format(&tib, 0), PIPE_FORMAT_R32_UINT);
      EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&tib));

      struct agx_shader_key key = {};
      agx_g17p_configure_rt0_eot_key(format.logical, &key);
      EXPECT_EQ(key.g17p_rt0_eot_words, 4);
      EXPECT_TRUE(key.g17p_rt0_eot_i32);
      EXPECT_EQ(key.g17p_rt0_eot_i32_four_words, !format.signed_format);
      EXPECT_EQ(key.g17p_rt0_eot_i32_signed_four_words,
                format.signed_format);

      /* Apple stores the 32-bit tile at every sample count with the same
       * lstcol it uses at one: metal-nt -arch applegpu_g17p on one
       * RGBA32Float attachment fed by a multisample read emits
       * e71654000000012e00000000 at one sample and e71654000000082e08000008
       * at four. KHR-GL33.texture_size_promotion renders to it. */
      tib = agx_build_tilebuffer_layout(rt, 1, 2, false);
      EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&tib));
      tib = agx_build_tilebuffer_layout(rt, 1, 1, true);
      EXPECT_FALSE(agx_g17p_rt0_tile_layout_supported(&tib));
   }

   /* Compact three-lane RGB32 is an emulated texture-buffer-only format, not
    * the four-lane RGBX render-target family admitted above. */
   enum pipe_format compact[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R32G32B32_FLOAT,
   };
   struct agx_tilebuffer_layout tib =
      agx_build_tilebuffer_layout(compact, 1, 1, false);
   EXPECT_EQ(agx_tilebuffer_physical_format(&tib, 0), PIPE_FORMAT_NONE);
   EXPECT_FALSE(agx_g17p_rt0_tile_layout_supported(&tib));

   struct agx_shader_key key = {};
   agx_g17p_configure_rt0_eot_key(PIPE_FORMAT_R32G32B32_FLOAT, &key);
   EXPECT_EQ(key.g17p_rt0_eot_words, 0);
   EXPECT_FALSE(key.g17p_rt0_eot_i32);
}

TEST(Tilebuffer, IndependentNativeStorageAndSampleLimits)
{
   enum pipe_format mixed[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R8G8B8A8_UNORM,
      PIPE_FORMAT_R16G16B16A16_FLOAT,
      PIPE_FORMAT_R32G32B32A32_UINT,
   };
   const auto legacy = agx_build_tilebuffer_layout_bounded(
      mixed, 3, 4, false, AGX_G17P_MAX_BYTES_PER_SAMPLE);
   EXPECT_TRUE(legacy.spilled[0]);

   const auto native = agx_build_tilebuffer_layout_with_limits(
      mixed, 3, 4, false, AGX_G17P_MAX_BYTES_PER_SAMPLE, 32768, 2048);
   EXPECT_FALSE(agx_tilebuffer_spills(&native));
   EXPECT_EQ(native.sample_size_B, 32u);
   EXPECT_EQ(native.tile_size, 16u * 16u);
   EXPECT_EQ(native.sample_size_B * native.nr_samples * native.tile_size, 32768u);
   EXPECT_EQ(native._offset_B[2], 0u);
   EXPECT_EQ(native._offset_B[1], 16u);
   EXPECT_EQ(native._offset_B[0], 24u);

   enum pipe_format rgba8[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R8G8B8A8_UNORM,
   };
   for (unsigned samples : {1u, 2u, 4u}) {
      const auto tib = agx_build_tilebuffer_layout_with_limits(
         rgba8, 1, samples, false, AGX_G17P_MAX_BYTES_PER_SAMPLE, 32768, 2048);
      EXPECT_EQ(tib.tile_size, samples == 4 ? 32u * 16u : 32u * 32u);
      EXPECT_LE(tib.nr_samples * tib.tile_size, 2048u);
   }

   /* Limits must survive attachmentless rendering's deferred sample count. */
   auto deferred = agx_build_tilebuffer_layout_with_limits(
      mixed, 3, 0, false, AGX_G17P_MAX_BYTES_PER_SAMPLE, 32768, 2048);
   agx_tilebuffer_set_samples(&deferred, 4);
   EXPECT_EQ(deferred.max_bytes_per_tile, 32768u);
   EXPECT_EQ(deferred.max_samples_per_tile, 2048u);
   EXPECT_EQ(deferred.tile_size, native.tile_size);
   EXPECT_EQ(deferred.sample_size_B, native.sample_size_B);
   EXPECT_FALSE(agx_tilebuffer_spills(&deferred));

   /* True over-capacity layouts still spill. */
   enum pipe_format wide[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R32G32B32A32_UINT,
      PIPE_FORMAT_R32G32B32A32_UINT,
      PIPE_FORMAT_R32G32B32A32_UINT,
   };
   const auto spilling = agx_build_tilebuffer_layout_with_limits(
      wide, 3, 4, false, AGX_G17P_MAX_BYTES_PER_SAMPLE, 32768, 2048);
   EXPECT_TRUE(spilling.spilled[2]);
   EXPECT_FALSE(spilling.spilled[0]);
   EXPECT_FALSE(spilling.spilled[1]);
}
