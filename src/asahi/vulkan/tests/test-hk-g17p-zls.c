/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#undef NDEBUG
#include "asahi/lib/agx_helpers.h"
#include "hk_drm_render_core.h"
#include "hk_g17p_zls.h"

#include <stdio.h>

static unsigned checks;
#define CHECK(x)                                                               \
   do {                                                                        \
      checks++;                                                                \
      if (!(x)) {                                                              \
         fprintf(stderr, "line %u: %s\n", __LINE__, #x);                       \
         abort();                                                              \
      }                                                                        \
   } while (0)

static struct ail_layout
make_layout(enum pipe_format format, unsigned w, unsigned h, unsigned layers,
            bool compressed, unsigned samples, bool mipmapped)
{
   struct ail_layout layout = {
      .width_px = w,
      .height_px = h,
      .depth_px = layers,
      .sample_count_sa = samples,
      .levels = mipmapped ? util_logbase2(MAX2(w, h)) + 1 : 1,
      .tiling = AIL_TILING_GPU,
      .format = format,
      .compressed = compressed,
      .renderable = true,
      .writeable_image = !compressed,
   };
   ail_make_miptree(&layout);
   return layout;
}

/* Check the entire metadata reserve, including unused narrow-tail slots. */
static void
check_metadata(const struct ail_layout *layout)
{
   unsigned w =
      ALIGN_POT(layout->width_px * (layout->sample_count_sa == 4 ? 2 : 1), 16);
   unsigned h =
      ALIGN_POT(layout->height_px * (layout->sample_count_sa >= 2 ? 2 : 1), 16);
   uint64_t offset = 0;
   for (unsigned l = 0; l < layout->levels; l++) {
      const unsigned major = MAX2(
         ALIGN_POT(layout->width_px * (layout->sample_count_sa == 4 ? 2 : 1),
                   16),
         ALIGN_POT(layout->height_px * (layout->sample_count_sa >= 2 ? 2 : 1),
                   16));
      if ((major >> l) < 16)
         break;
      CHECK(layout->level_offsets_compressed_B[l] == offset);
      const unsigned tw = DIV_ROUND_UP(util_next_power_of_two(w), 16);
      const unsigned th = DIV_ROUND_UP(util_next_power_of_two(h), 16);
      offset += ALIGN_POT((uint64_t)tw * th * 8, 128);
      w = DIV_ROUND_UP(w, 2);
      h = DIV_ROUND_UP(h, 2);
   }
   CHECK(layout->compression_layer_stride_B == offset);
   CHECK(layout->size_B ==
         layout->metadata_offset_B + offset * layout->depth_px);
}

static struct ail_layout
selected_layout(const struct ail_layout *layout,
                const struct hk_g17p_zls_plane *state)
{
   struct ail_layout selected = *layout;
   selected.compressed = state->meta != 0;
   selected.tiling = state->tiling;
   return selected;
}

/* All load/store combinations cover the first, partial and final stream
 * words. Compression and tiling must not depend on those actions. */
static void
check_control(const struct ail_layout *z, const struct ail_layout *s)
{
   for (unsigned actions = 0; actions < 16; actions++) {
      struct agx_zls args = {
         .z_load = actions & 1,
         .z_store = actions & 2,
         .s_load = actions & 4,
         .s_store = actions & 8,
      };
      struct agx_zls_control_packed packed;
      agx_pack_zls_control(&packed, z, s, &args);
      struct AGX_ZLS_CONTROL cfg;
      CHECK(AGX_ZLS_CONTROL_unpack(NULL, (const uint8_t *)&packed, &cfg));
      CHECK(cfg.z_load == (z && args.z_load));
      CHECK(cfg.z_store == (z && args.z_store));
      CHECK(cfg.s_load == (s && args.s_load));
      CHECK(cfg.s_store == (s && args.s_store));
      CHECK(cfg.z_load_compress == (z && z->compressed));
      CHECK(cfg.z_store_compress == (z && z->compressed));
      CHECK(cfg.s_load_compress == (s && s->compressed));
      CHECK(cfg.s_store_compress == (s && s->compressed));
      CHECK(cfg.z_load_tiling == (z && z->tiling == AIL_TILING_TWIDDLED));
      CHECK(cfg.z_store_tiling == (z && z->tiling == AIL_TILING_TWIDDLED));
      CHECK(cfg.s_load_tiling == (s && s->tiling == AIL_TILING_TWIDDLED));
      CHECK(cfg.s_store_tiling == (s && s->tiling == AIL_TILING_TWIDDLED));
      CHECK(cfg.z_format == (z && z->format == PIPE_FORMAT_Z16_UNORM
                                ? AGX_ZLS_FORMAT_16
                                : AGX_ZLS_FORMAT_32F));
   }
}

static void
check_plane(const struct ail_layout *layout, uint64_t base)
{
   if (layout->compressed)
      check_metadata(layout);

   for (unsigned l = 0; l < layout->levels; l++) {
      const unsigned w = MAX2(layout->width_px >> l, 1);
      const unsigned h = MAX2(layout->height_px >> l, 1);
      const bool compressed =
         layout->compressed &&
         w * (layout->sample_count_sa == 4 ? 2 : 1) >= 16 &&
         h * (layout->sample_count_sa >= 2 ? 2 : 1) >= 16;
      for (unsigned k = 0; k < layout->depth_px; k++) {
         struct hk_g17p_zls_plane state = hk_g17p_zls_plane(layout, base, k, l);
         CHECK(state.width == w && state.height == h);
         const uint32_t pixels = hk_drm_g17p_zls_pixels(
            state.width, state.height, state.buffer, state.meta, 0, 0);
         CHECK(pixels == (w - 1) + ((h - 1) << 15));
         CHECK(!(pixels & ((1u << 14) | (1u << 29))));
         CHECK(state.buffer ==
               base + k * layout->layer_stride_B + layout->level_offsets_B[l]);
         CHECK(state.buffer % 128 == 0);
         CHECK(state.buffer < base + layout->metadata_offset_B ||
               !layout->compressed);
         const uint64_t stride = layout->layer_stride_B;
         CHECK(state.stride ==
               (stride >= 16384 ? (((stride / 16384) - 1) << 14) | 1 : 0));
         CHECK(state.tiling == (layout->compressed && !compressed
                                   ? AIL_TILING_TWIDDLED
                                   : layout->tiling));
         if (compressed) {
            CHECK(state.meta == base + layout->metadata_offset_B +
                                   k * layout->compression_layer_stride_B +
                                   layout->level_offsets_compressed_B[l]);
            CHECK(state.meta % 128 == 0);
            CHECK(state.meta < base + layout->size_B);
            CHECK(state.meta_stride ==
                  (layout->compression_layer_stride_B / 128 - 1) << 14);
            CHECK(state.meta + layout->compression_layer_stride_B ==
                  base + layout->metadata_offset_B +
                     (k + 1) * layout->compression_layer_stride_B +
                     layout->level_offsets_compressed_B[l]);
         } else {
            CHECK(state.meta == 0 && state.meta_stride == 0);
         }
         struct ail_layout selected = selected_layout(layout, &state);
         if (layout->format == PIPE_FORMAT_S8_UINT)
            check_control(NULL, &selected);
         else
            check_control(&selected, NULL);
      }
   }
}

static void
check_combined(unsigned w, unsigned h, unsigned layers)
{
   struct ail_layout z =
      make_layout(PIPE_FORMAT_Z32_FLOAT, w, h, layers, true, 1, true);
   struct ail_layout s =
      make_layout(PIPE_FORMAT_S8_UINT, w, h, layers, true, 1, true);
   const uint64_t zbase = UINT64_C(0x100000000);
   const uint64_t sbase = zbase + ALIGN_POT(z.size_B, 16384);
   CHECK(z.layer_stride_B >= s.layer_stride_B);
   const struct hk_g17p_zls_plane absent = hk_g17p_zls_plane(&z, 0, 0, 0);
   CHECK(absent.meta == 0 && absent.meta_stride == 0);
   for (unsigned l = 0; l < z.levels; l++) {
      for (unsigned k = 0; k < layers; k++) {
         const struct hk_g17p_zls_plane zs = hk_g17p_zls_plane(&z, zbase, k, l);
         const struct hk_g17p_zls_plane ss = hk_g17p_zls_plane(&s, sbase, k, l);
         struct ail_layout zl = selected_layout(&z, &zs);
         struct ail_layout sl = selected_layout(&s, &ss);
         check_control(&zl, &sl);
         CHECK(zs.buffer < sbase && ss.buffer >= sbase);
         if (zs.meta)
            CHECK(zs.meta < sbase && ss.meta >= sbase);

         /* A missing metadata allocation on either plane clears only that
          * aspect, regardless of the other aspect's admission. */
         struct ail_layout missing = z;
         missing.compression_layer_stride_B = 0;
         struct hk_g17p_zls_plane ms = hk_g17p_zls_plane(&missing, zbase, k, l);
         CHECK(ms.meta == 0 && ms.meta_stride == 0);
         struct ail_layout ml = selected_layout(&missing, &ms);
         check_control(&ml, &sl);
         missing = s;
         missing.compressed = false;
         ms = hk_g17p_zls_plane(&missing, sbase, k, l);
         CHECK(ms.meta == 0 && ms.meta_stride == 0);
         ml = selected_layout(&missing, &ms);
         check_control(&zl, &ml);
      }
   }
}

int
main(void)
{
   static const unsigned sizes[][2] = {
      {16, 16},  {17, 31},   {32, 32},     {33, 65},       {129, 127},
      {300, 17}, {17, 4096}, {1920, 1080}, {16384, 16384},
   };
   static const unsigned layers[] = {1, 2, 6, 12};
   static const enum pipe_format formats[] = {
      PIPE_FORMAT_Z16_UNORM,
      PIPE_FORMAT_Z32_FLOAT,
      PIPE_FORMAT_S8_UINT,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(sizes); i++) {
      for (unsigned j = 0; j < ARRAY_SIZE(layers); j++) {
         /* Keep the maximum-extent fixtures within maxResourceSize. */
         if (sizes[i][0] == 16384 && layers[j] > 2)
            continue;
         for (unsigned f = 0; f < ARRAY_SIZE(formats); f++) {
            struct ail_layout layout = make_layout(
               formats[f], sizes[i][0], sizes[i][1], layers[j], true, 1, true);
            check_plane(&layout, UINT64_C(0x100000000));
            for (unsigned samples = 1; samples <= (sizes[i][0] <= 4096 ? 4 : 1);
                 samples *= 2) {
               layout = make_layout(formats[f], sizes[i][0], sizes[i][1],
                                    layers[j], true, samples, false);
               check_plane(&layout, UINT64_C(0x100000000));
            }
            layout = make_layout(formats[f], sizes[i][0], sizes[i][1],
                                 layers[j], false, 1, true);
            check_plane(&layout, UINT64_C(0x100000000));
         }
         check_combined(sizes[i][0], sizes[i][1], layers[j]);
      }
   }
   printf("hk_g17p_zls: %u checks passed\n", checks);
   return 0;
}
