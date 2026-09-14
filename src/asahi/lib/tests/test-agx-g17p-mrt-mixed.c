/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * G17P render passes with three or four colour targets of arbitrary formats.
 *
 * Apple's builtin end-of-tile factory (metal-nt -arch applegpu_g17p, BTNF
 * root field 2 = 5) emits the same program for two, three and four
 * same-partition 2D targets whatever their formats: the raw32 text in
 * agx3_g17p_quad_eot.h with each target's IMGWBLK selector byte replaced.
 * Sweeping internal formats 0..23 and components 1..4 shows the selector is
 * a function of the target's own format alone, and 42 mixed combinations of
 * two to four targets reproduce template-plus-selectors byte for byte. The
 * layouts below are the nine dEQP-VK renderpasses attachment.{3,4}.* cases.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "util/format/u_format.h"
#include "agx3_g17p_quad_eot.h"
#include "agx_bg_eot_key.h"
#include "agx_compile.h"
#include "agx_linker.h"
#include "agx_nir.h"
#include "agx_nir_texture.h"
#include "agx_tilebuffer.h"
#include "layout.h"
#include "nir_builder.h"

static unsigned checks;

#define CHECK(cond)                                                            \
   do {                                                                        \
      checks++;                                                                \
      if (!(cond)) {                                                           \
         fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #cond);               \
         return 1;                                                             \
      }                                                                        \
   } while (0)

struct layout {
   const char *name;
   unsigned count;
   enum pipe_format formats[4];
   uint8_t offsets[4];
   uint8_t sample_size_B;
};

static const struct layout layouts[] = {
   {"3.296", 3,
    {PIPE_FORMAT_R16G16_SNORM, PIPE_FORMAT_R16_SNORM, PIPE_FORMAT_R16_FLOAT},
    {0, 4, 6}, 8},
   {"3.314", 3,
    {PIPE_FORMAT_R8_UNORM, PIPE_FORMAT_R8G8B8A8_UNORM, PIPE_FORMAT_R8G8_SINT},
    {0, 1, 5}, 8},
   {"3.336", 3,
    {PIPE_FORMAT_R16G16B16A16_SNORM, PIPE_FORMAT_R16_FLOAT,
     PIPE_FORMAT_R32G32B32A32_SINT},
    {16, 24, 0}, 32},
   {"3.397", 3,
    {PIPE_FORMAT_R16G16_SNORM, PIPE_FORMAT_R16G16_FLOAT,
     PIPE_FORMAT_R16G16B16A16_UINT},
    {0, 4, 8}, 16},
   {"4.418", 4,
    {PIPE_FORMAT_R8G8B8A8_SNORM, PIPE_FORMAT_R8G8B8A8_UNORM,
     PIPE_FORMAT_R32G32_UINT, PIPE_FORMAT_R16_UNORM},
    {10, 14, 0, 8}, 24},
   {"4.465", 4,
    {PIPE_FORMAT_R32_FLOAT, PIPE_FORMAT_R16_UNORM,
     PIPE_FORMAT_R32G32B32A32_FLOAT, PIPE_FORMAT_R8G8B8A8_UNORM},
    {0, 20, 4, 22}, 32},
   {"4.541", 4,
    {PIPE_FORMAT_R16G16B16A16_UINT, PIPE_FORMAT_R8_SINT,
     PIPE_FORMAT_R8G8_SNORM, PIPE_FORMAT_R8G8_SINT},
    {0, 8, 9, 11}, 16},
   {"4.568", 4,
    {PIPE_FORMAT_R16G16B16A16_FLOAT, PIPE_FORMAT_R16G16B16A16_SNORM,
     PIPE_FORMAT_R8G8B8A8_SINT, PIPE_FORMAT_B8G8R8A8_UNORM},
    {0, 8, 16, 20}, 24},
   {"4.587", 4,
    {PIPE_FORMAT_R8G8B8A8_SRGB, PIPE_FORMAT_B10G10R10A2_UNORM,
     PIPE_FORMAT_B8G8R8A8_UNORM, PIPE_FORMAT_R16G16_UINT},
    {0, 4, 12, 8}, 16},
};

/* The store format the end-of-tile builder hands IMGWBLK. */
static enum pipe_format
eot_store_format(enum pipe_format format)
{
   return format == PIPE_FORMAT_R11G11B10_FLOAT ? PIPE_FORMAT_R32_UINT
          : util_format_is_srgb(format)        ? PIPE_FORMAT_R8_UNORM
                                         : ail_pixel_format[format].renderable;
}

static uint64_t
fnv1a(const uint8_t *data, size_t size)
{
   uint64_t hash = 0xcbf29ce484222325ull;
   for (size_t i = 0; i < size; ++i)
      hash = (hash ^ data[i]) * 0x100000001b3ull;
   return hash;
}

static bool
compile_eot(unsigned count, const enum pipe_format *formats,
            const uint8_t *offsets, const struct agx_shader_key *base_key,
            struct agx_shader_part *part)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "eot");
   nir_def *xy = nir_trim_vector(&b, nir_load_global_invocation_id(&b, 32), 2);
   for (unsigned rt = 0; rt < count; ++rt) {
      nir_image_store_block_agx(&b, nir_imm_intN_t(&b, rt, 16),
                                nir_imm_int(&b, offsets[rt]), xy,
                                .format = eot_store_format(formats[rt]),
                                .image_dim = GLSL_SAMPLER_DIM_2D,
                                .explicit_coord = true);
   }
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = *base_key;
   const bool ok = agx_compile_shader_nir(b.shader, &key, part);
   ralloc_free(b.shader);
   return ok;
}

/* Every admitted format's multi-target selector is the selector of that
 * format's own single-target program. */
static int
test_selectors_match_single_target_programs(void)
{
   unsigned admitted = 0;
   for (unsigned f = 1; f < PIPE_FORMAT_COUNT; ++f) {
      const uint8_t selector = agx_g17p_mrt_eot_selector(f);
      if (!selector)
         continue;

      struct agx_shader_key key = {
         .dev.encoding = AGX_ISA_ENCODING_AGX3B,
         .no_preamble = true,
      };
      agx_g17p_configure_rt0_eot_key(f, &key);
      const enum pipe_format format = f;
      const uint8_t offset = 0;
      struct agx_shader_part part = {0};
      CHECK(compile_eot(1, &format, &offset, &key, &part));
      CHECK(part.info.binary_size == 68);
      CHECK(((uint8_t *)part.binary)[61] == selector);
      free(part.binary);
      admitted++;
   }

   CHECK(agx_g17p_mrt_eot_selector(PIPE_FORMAT_R16G16B16A16_SINT) == 0);
   CHECK(agx_g17p_mrt_eot_selector(PIPE_FORMAT_R5G6B5_UNORM) == 0);
   CHECK(agx_g17p_mrt_eot_selector(PIPE_FORMAT_R11G11B10_FLOAT) == 0);
   CHECK(agx_g17p_mrt_eot_selector(PIPE_FORMAT_R9G9B9E5_FLOAT) == 0);
   CHECK(admitted > 40);
   return 0;
}

static int
test_layout_classification(void)
{
   for (unsigned i = 0; i < ARRAY_SIZE(layouts); ++i) {
      const struct layout *l = &layouts[i];
      enum pipe_format formats[AGX_MAX_RENDER_TARGETS] = {0};
      memcpy(formats, l->formats, sizeof(l->formats));
      struct agx_tilebuffer_layout tib =
         agx_build_tilebuffer_layout(formats, l->count, 1, false);

      CHECK(tib.sample_size_B == l->sample_size_B);
      for (unsigned rt = 0; rt < l->count; ++rt)
         CHECK(agx_tilebuffer_offset_B(&tib, rt) == l->offsets[rt]);
      CHECK(agx_g17p_classify_mrt(&tib) == AGX_G17P_MRT_MIXED);
      CHECK(agx_g17p_rt0_tile_layout_supported(&tib));

      struct agx_tilebuffer_layout bad = tib;
      bad.spilled[l->count - 1] = true;
      CHECK(agx_g17p_classify_mrt(&bad) == AGX_G17P_MRT_NONE);

      bad = tib;
      bad._offset_B[l->count - 1] = bad._offset_B[0];
      CHECK(agx_g17p_classify_mrt(&bad) == AGX_G17P_MRT_NONE);

      bad = tib;
      bad._offset_B[l->count - 1] = tib.sample_size_B - 1;
      CHECK(agx_g17p_classify_mrt(&bad) == AGX_G17P_MRT_NONE);

      bad = tib;
      bad.logical_format[l->count - 1] = PIPE_FORMAT_R11G11B10_FLOAT;
      CHECK(agx_g17p_classify_mrt(&bad) == AGX_G17P_MRT_NONE);

      bad = tib;
      bad.nr_samples = 2;
      CHECK(agx_g17p_classify_mrt(&bad) == AGX_G17P_MRT_NONE);

      bad = tib;
      bad.layered = true;
      CHECK(agx_g17p_classify_mrt(&bad) == AGX_G17P_MRT_NONE);

      /* A fifth target has no native program. */
      formats[l->count] = PIPE_FORMAT_R8_UNORM;
      if (l->count == 4) {
         tib = agx_build_tilebuffer_layout(formats, 5, 1, false);
         CHECK(agx_g17p_classify_mrt(&tib) == AGX_G17P_MRT_NONE);
         CHECK(!agx_g17p_rt0_tile_layout_supported(&tib));
      }
   }

   /* Homogeneous layouts keep their own programs. */
   enum pipe_format rgba8[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R8G8B8A8_UNORM, PIPE_FORMAT_R8G8B8A8_UNORM,
      PIPE_FORMAT_R8G8B8A8_UNORM, PIPE_FORMAT_R8G8B8A8_UNORM};
   enum pipe_format rgba32[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R32G32B32A32_FLOAT, PIPE_FORMAT_R32G32B32A32_FLOAT,
      PIPE_FORMAT_R32G32B32A32_FLOAT, PIPE_FORMAT_R32G32B32A32_FLOAT};
   enum pipe_format r32[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R32_FLOAT, PIPE_FORMAT_R32_FLOAT, PIPE_FORMAT_R32_FLOAT,
      PIPE_FORMAT_R32_FLOAT};
   for (unsigned count = 3; count <= 4; ++count) {
      struct agx_tilebuffer_layout tib =
         agx_build_tilebuffer_layout(rgba8, count, 1, false);
      CHECK(agx_g17p_classify_mrt(&tib) == AGX_G17P_MRT_RGBA8);
      tib = agx_build_tilebuffer_layout(rgba32, count, 1, false);
      CHECK(agx_g17p_classify_mrt(&tib) == AGX_G17P_MRT_RGBA32);
   }
   struct agx_tilebuffer_layout tib =
      agx_build_tilebuffer_layout(r32, 4, 1, false);
   CHECK(agx_g17p_classify_mrt(&tib) == AGX_G17P_MRT_R32_QUAD);

   /* Two targets outside the two-target families use the same programs. */
   enum pipe_format pair[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R16G16B16A16_FLOAT, PIPE_FORMAT_R16_UNORM};
   tib = agx_build_tilebuffer_layout(pair, 2, 1, false);
   CHECK(agx_g17p_classify_mrt(&tib) == AGX_G17P_MRT_MIXED);
   CHECK(agx_g17p_rt0_tile_layout_supported(&tib));

   /* A full sample selects the 16x16 tile, as for four RGBA32 targets. */
   enum pipe_format wide[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R32G32B32A32_FLOAT, PIPE_FORMAT_R32G32B32A32_UINT,
      PIPE_FORMAT_R32G32B32A32_SINT, PIPE_FORMAT_R32G32B32A32_FLOAT};
   tib = agx_build_tilebuffer_layout(wide, 4, 1, false);
   CHECK(tib.tile_size == 16 * 16);
   CHECK(agx_g17p_classify_mrt(&tib) == AGX_G17P_MRT_MIXED);
   CHECK(agx_g17p_rt0_tile_layout_supported(&tib));
   return 0;
}

/* Expected multi-target EOT: the native raw32 text with each target's
 * IMGWBLK selector at the byte metal-nt changes for it. */
static const uint8_t *
expected_eot(unsigned count, const uint8_t *selectors, uint8_t *out)
{
   static const unsigned selector_offsets[][4] = {
      [2] = {101, 113},
      [3] = {133, 145, 157},
      [4] = {165, 177, 189, 201},
   };
   const uint8_t *text = count == 2   ? agx3_g17p_dual_raw32_eot
                         : count == 3 ? agx3_g17p_triple_raw32_eot
                                      : agx3_g17p_quad_eot;
   memcpy(out, text, 32 + 44 * count);
   for (unsigned rt = 0; rt < count; ++rt)
      out[selector_offsets[count][rt]] = selectors[rt];
   return out;
}

static int
check_eot_subset(unsigned count, const enum pipe_format *formats,
                 const uint8_t *offsets)
{
   struct agx_shader_key key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .no_preamble = true,
      .g17p_mrt_eot_count = count,
   };
   agx_g17p_configure_rt0_eot_key(formats[0], &key);
   for (unsigned rt = 0; rt < count; ++rt)
      key.g17p_mrt_eot_selectors[rt] = agx_g17p_mrt_eot_selector(formats[rt]);

   struct agx_shader_part part = {0};
   CHECK(compile_eot(count, formats, offsets, &key, &part));
   uint8_t expected[sizeof(agx3_g17p_quad_eot)];
   CHECK(part.info.binary_size == 32 + 44 * count);
   CHECK(part.info.nr_gprs == 8);
   CHECK(part.info.agx3_staging_known &&
         part.info.agx3_staging_halfs == 8 * count);
   CHECK(!memcmp(part.binary,
                 expected_eot(count, key.g17p_mrt_eot_selectors, expected),
                 part.info.binary_size));
   free(part.binary);

   /* A store whose format cannot carry its target's selector is refused
    * rather than published with a selector from another format. */
   for (unsigned rt = 0; rt < count; ++rt) {
      struct agx_shader_key wrong = key;
      wrong.g17p_mrt_eot_selectors[rt] =
         key.g17p_mrt_eot_selectors[rt] == 0x55 ? 0x95 : 0x55;
      part = (struct agx_shader_part){0};
      CHECK(!compile_eot(count, formats, offsets, &wrong, &part));
   }

   struct agx_shader_key short_key = key;
   short_key.g17p_mrt_eot_count = count - 1;
   part = (struct agx_shader_part){0};
   CHECK(!compile_eot(count, formats, offsets, &short_key, &part));
   return 0;
}

static int
test_eot_programs(void)
{
   for (unsigned i = 0; i < ARRAY_SIZE(layouts); ++i) {
      const struct layout *l = &layouts[i];
      for (unsigned mask = 1; mask < BITFIELD_MASK(l->count); ++mask) {
         if (util_bitcount(mask) < 2)
            continue;
         enum pipe_format formats[4];
         uint8_t offsets[4];
         unsigned count = 0;
         u_foreach_bit(rt, mask) {
            formats[count] = l->formats[rt];
            offsets[count++] = l->offsets[rt];
         }
         if (check_eot_subset(count, formats, offsets))
            return 1;
      }
      if (check_eot_subset(l->count, l->formats, l->offsets))
         return 1;
   }

   /* Homogeneous RGBA8 targets reproduce the separately captured programs. */
   const uint8_t rgba8[] = {0x75, 0x75, 0x75, 0x75};
   uint8_t expected[sizeof(agx3_g17p_quad_eot)];
   CHECK(!memcmp(expected_eot(3, rgba8, expected), agx3_g17p_rgba8_3_eot,
                 sizeof(agx3_g17p_rgba8_3_eot)));
   CHECK(!memcmp(expected_eot(4, rgba8, expected), agx3_g17p_rgba8_4_eot,
                 sizeof(agx3_g17p_rgba8_4_eot)));
   return 0;
}

static bool
lower_tex_handle_to_u0(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_texture_handle_agx)
      return false;

   b->cursor = nir_instr_remove(&intr->instr);
   nir_def_rewrite_uses(
      &intr->def,
      nir_bindless_image_agx(b, nir_imul_imm(b, intr->src[0].ssa, 32),
                             .desc_set = 0));
   return true;
}

/* Mirror agx_build_background_shader for reloads followed by clears. */
static bool
compile_background(const struct agx_tilebuffer_layout *layout,
                   unsigned reloads, bool mrt_bg, unsigned reserved_preamble,
                   struct agx_shader_part *part)
{
   struct agx_tilebuffer_layout tib = *layout;
   unsigned count = 0;
   while (count < AGX_MAX_RENDER_TARGETS &&
          tib.logical_format[count] != PIPE_FORMAT_NONE)
      count++;

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "G17P mixed BG");
   b.shader->info.fs.untyped_color_outputs = true;
   for (unsigned rt = 0; rt < count; ++rt) {
      const unsigned nr = util_format_get_nr_components(tib.logical_format[rt]);
      nir_def *value;
      if (rt < reloads) {
         nir_def *coord = nir_u2u32(&b, nir_load_pixel_coord(&b));
         nir_def *tex = nir_txf(&b, coord, .texture_index = rt * 2,
                                .dim = GLSL_SAMPLER_DIM_2D,
                                .dest_type = nir_type_uint32,
                                .backend_flags = AGX_TEXTURE_FLAG_NO_CLAMP);
         value = nir_trim_vector(&b, tex, nr);
      } else {
         value = nir_load_preamble(&b, nr, 32, 4 + rt * 8);
      }
      nir_store_output(&b, value, nir_imm_int(&b, 0),
                       .write_mask = BITFIELD_MASK(nr),
                       .src_type = nir_type_uint32,
                       .io_semantics.location = FRAG_RESULT_DATA0 + rt,
                       .io_semantics.num_slots = 1);
      b.shader->info.outputs_written |= BITFIELD64_BIT(FRAG_RESULT_DATA0 + rt);
   }

   agx_nir_lower_texture(b.shader);
   agx_preprocess_nir(b.shader);
   unsigned bindless_base = 0;
   NIR_PASS(_, b.shader, agx_nir_lower_tilebuffer, &tib, NULL, &bindless_base,
            NULL, NULL, false, true);
   NIR_PASS(_, b.shader, agx_nir_lower_monolithic_msaa, tib.nr_samples);
   NIR_PASS(_, b.shader, agx_nir_lower_multisampled_image_store);
   NIR_PASS(_, b.shader, agx_nir_lower_texture);
   NIR_PASS(_, b.shader, nir_shader_intrinsics_pass, lower_tex_handle_to_u0,
            nir_metadata_control_flow, NULL);
   NIR_PASS(_, b.shader, agx_nir_lower_g17p_tile_samples, tib.nr_samples);

   struct agx_shader_key key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .fs.ignore_tib_dependencies = true,
      .g17p_mrt_bg = mrt_bg,
      .reserved_preamble = reserved_preamble,
      .no_preamble = true,
   };
   const bool ok = agx_compile_shader_nir(b.shader, &key, part);
   ralloc_free(b.shader);
   return ok;
}

static int
test_background_programs(void)
{
   for (unsigned i = 0; i < ARRAY_SIZE(layouts); ++i) {
      const struct layout *l = &layouts[i];
      enum pipe_format formats[AGX_MAX_RENDER_TARGETS] = {0};
      memcpy(formats, l->formats, sizeof(l->formats));
      const struct agx_tilebuffer_layout tib =
         agx_build_tilebuffer_layout(formats, l->count, 1, false);

      for (unsigned reloads = 0; reloads <= l->count; ++reloads) {
         struct agx_shader_part part = {0};
         CHECK(compile_background(&tib, reloads, true,
                                  reloads < l->count ? 4 + 8 * l->count : 0,
                                  &part));
         CHECK(part.info.uses_txf == (reloads > 0));
         CHECK(part.info.push_count ==
               (reloads < l->count ? 4 + 8 * l->count : 0));
         CHECK(!part.info.scratch_size);
         CHECK(!part.info.agx3_colour_link.entry_emitted);
         free(part.binary);
      }
   }
   return 0;
}

static bool
compile_epilog(unsigned count, const enum pipe_format *formats, bool additive,
               bool write_z, struct agx_shader_part *part)
{
   const enum pipe_blendfactor dst =
      additive ? PIPE_BLENDFACTOR_ONE : PIPE_BLENDFACTOR_ZERO;
   struct agx_fs_epilog_key epilog = {
      .link.loc_written = BITFIELD_MASK(count),
      .link.size_32 = BITFIELD_MASK(count),
      .link.write_z = write_z,
      .nr_samples = 1,
   };
   memset(epilog.remap, -1, sizeof(epilog.remap));
   for (unsigned rt = 0; rt < count; ++rt) {
      epilog.remap[rt] = rt;
      epilog.rt_formats[rt] = formats[rt];
      epilog.blend.rt[rt].colormask =
         BITFIELD_MASK(util_format_get_nr_components(formats[rt]));
      epilog.blend.rt[rt].mode = agx_pack_blend_standard(
         PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, dst, PIPE_BLEND_ADD,
         PIPE_BLENDFACTOR_ONE, dst);
   }
   agx_fs_epilog_apply_device_policy(AGX_CHIP_G17P, &epilog);

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "G17P MRT epilog");
   agx_nir_fs_epilog(&b, &epilog);
   agx_nir_lower_g17p_tile_samples(b.shader, 1);
   agx_preprocess_nir(b.shader);
   struct agx_shader_key key = {.dev.encoding = AGX_ISA_ENCODING_AGX3B,
                                .secondary = true};
   const bool ok = agx_compile_shader_nir(b.shader, &key, part);
   ralloc_free(b.shader);
   return ok;
}

/* Find each pixel instruction of a native witness, in order. */
static bool
contains_in_order(const struct agx_shader_part *part,
                  const uint8_t (*ops)[6], unsigned count)
{
   const uint8_t *text = part->binary;
   size_t at = 0;
   for (unsigned i = 0; i < count; ++i) {
      while (at + 6 <= part->info.binary_size &&
             memcmp(text + at, ops[i], 6))
         at++;
      if (at + 6 > part->info.binary_size)
         return false;
      at += 6;
   }
   return true;
}

/* The pixel transactions of Apple's store-only MRT fragments for these
 * layouts, from metal-nt render pipelines with the same colour formats and a
 * [[depth(any)]] output: the entry, the class-1 depth transaction, then the
 * colour partitions exactly as without depth. */
static const uint8_t entry_512[6] = {0x87, 0x02, 0x54, 0x00, 0x06, 0x00};
static const uint8_t wait_1[6] = {0x87, 0x02, 0x54, 0x01, 0x00, 0x00};
static const uint8_t depthf[6] = {0xd7, 0x14, 0x54, 0x00, 0x00, 0x03};
static const uint8_t release_1[6] = {0x07, 0x02, 0x54, 0x01, 0x00, 0x00};
static const uint8_t wait_192[6] = {0x87, 0x02, 0x54, 0xc0, 0x08, 0x00};
static const uint8_t release_192[6] = {0x07, 0x02, 0x54, 0xc0, 0x00, 0x00};
static const uint8_t wait_48[6] = {0x87, 0x02, 0x54, 0x30, 0x08, 0x00};
static const uint8_t release_48[6] = {0x07, 0x02, 0x54, 0x30, 0x00, 0x00};
static const uint8_t wait_12[6] = {0x87, 0x02, 0x54, 0x0c, 0x08, 0x00};
static const uint8_t release_524[6] = {0x07, 0x02, 0x54, 0x0c, 0x02, 0x00};

static int
test_epilogs(void)
{
   for (unsigned i = 0; i < ARRAY_SIZE(layouts); ++i) {
      const struct layout *l = &layouts[i];
      for (unsigned write_z = 0; write_z < 2; ++write_z) {
         struct agx_shader_part part = {0};
         CHECK(compile_epilog(l->count, l->formats, false, write_z, &part));
         CHECK(part.info.agx3_colour_link.colour_partitions == 3);
         CHECK(part.info.agx3_colour_link.entry_emitted);

         unsigned n = 0;
         uint8_t sequence[16][6];
         memcpy(sequence[n++], entry_512, 6);
         if (write_z) {
            memcpy(sequence[n++], wait_1, 6);
            memcpy(sequence[n++], depthf, 6);
            memcpy(sequence[n++], release_1, 6);
         }
         if (l->count == 3) {
            memcpy(sequence[n++], wait_192, 6);
            memcpy(sequence[n++], release_192, 6);
            memcpy(sequence[n++], wait_48, 6);
            memcpy(sequence[n++], release_48, 6);
            memcpy(sequence[n++], wait_12, 6);
            memcpy(sequence[n++], release_524, 6);
         } else {
            memcpy(sequence[n++], wait_12, 6);
            memcpy(sequence[n++], wait_192, 6);
            memcpy(sequence[n++], release_192, 6);
            memcpy(sequence[n++], wait_48, 6);
            memcpy(sequence[n++], release_48, 6);
            memcpy(sequence[n++], release_524, 6);
         }
         CHECK(contains_in_order(&part, (const uint8_t (*)[6])sequence, n));
         free(part.binary);
      }
   }
   return 0;
}

/* Homogeneous and two-target layouts keep their programs: sizes and FNV-1a
 * hashes of each binary as origin/agx3 2966c53569e compiled them. Blending
 * is covered where those layouts admit it. */
struct golden_program {
   uint32_t size;
   uint64_t hash;
};

static const struct {
   const char *name;
   unsigned count;
   enum pipe_format format0, other_formats;
   unsigned clear_push_halfs;
   struct golden_program programs[5];
} homogeneous[] = {
   {"rgba8x1", 1, PIPE_FORMAT_R8G8B8A8_UNORM, PIPE_FORMAT_NONE, 12,
    {{54, 0x4046fbb80be63de7ull},
     {176, 0x5f0f2b44b4423126ull},
     {36, 0x44e827688295a342ull},
     {94, 0xdac760eb596de1ceull},
     {68, 0xc35f74153a6f5d4dull}}},
   {"rgba16fx1", 1, PIPE_FORMAT_R16G16B16A16_FLOAT, PIPE_FORMAT_NONE, 12,
    {{58, 0x9bb0d0a9dc901048ull},
     {124, 0xaacc42d2bae1faf7ull},
     {56, 0x13fce0f2bc037095ull},
     {98, 0x2d5b1f727d37fe35ull},
     {68, 0x191002cafff501edull}}},
   {"rgba8x2", 2, PIPE_FORMAT_R8G8B8A8_UNORM, PIPE_FORMAT_R8G8B8A8_UNORM, 20,
    {{98, 0xdb4a9c1199db533eull},
     {342, 0xbbaff21bc69ed97aull},
     {98, 0x3d556eefb6d79da2ull},
     {190, 0x537d1f57e7f03c12ull},
     {120, 0x1c83deaa038a3135ull}}},
   {"rgba8x3", 3, PIPE_FORMAT_R8G8B8A8_UNORM, PIPE_FORMAT_R8G8B8A8_UNORM, 28,
    {{142, 0x219beb85e9dcef53ull},
     {0, 0},
     {100, 0x7e4ec8188383fa60ull},
     {226, 0x777d88deacbc5f85ull},
     {164, 0x3a6773eb9825a63full}}},
   {"rgba8x4", 4, PIPE_FORMAT_R8G8B8A8_UNORM, PIPE_FORMAT_R8G8B8A8_UNORM, 36,
    {{174, 0xeca6c1a1fffa8aa6ull},
     {0, 0},
     {132, 0x254d0fd1f930dc33ull},
     {292, 0xe2f43523620bc947ull},
     {208, 0x343085049a8f7fd5ull}}},
   {"rgba32fx2", 2, PIPE_FORMAT_R32G32B32A32_FLOAT,
    PIPE_FORMAT_R32G32B32A32_FLOAT, 20,
    {{138, 0x8d1aa0edcf65a202ull},
     {208, 0x54f2d78621dd6b72ull},
     {60, 0x0acdf209897c54bbull},
     {168, 0xa6f1af8b4212793dull},
     {120, 0x1aa59cf1e9af7435ull}}},
   {"rgba32fx3", 3, PIPE_FORMAT_R32G32B32A32_FLOAT,
    PIPE_FORMAT_R32G32B32A32_FLOAT, 28,
    {{202, 0xed0c467782f8f8e0ull},
     {0, 0},
     {88, 0xb10acac55bc69275ull},
     {214, 0x90e525c9e03e950bull},
     {164, 0x41bfe3fe09b22dffull}}},
   {"rgba32ux4", 4, PIPE_FORMAT_R32G32B32A32_UINT,
    PIPE_FORMAT_R32G32B32A32_UINT, 36,
    {{254, 0x0a0b4e4dce2827feull},
     {0, 0},
     {116, 0x5d443c6410799bfbull},
     {260, 0xd2771c02d344217full},
     {208, 0x866cf34c0d0fbd55ull}}},
   {"r32fx4", 4, PIPE_FORMAT_R32_FLOAT, PIPE_FORMAT_R32_FLOAT, 36,
    {{104, 0x6b76264af3ddd455ull},
     {0, 0},
     {68, 0x9780d74361183b43ull},
     {212, 0x089fc7e5c4812c7full},
     {208, 0x866cf34c0d0fbd55ull}}},
   {"wboit", 2, PIPE_FORMAT_R16G16B16A16_FLOAT, PIPE_FORMAT_R8_UNORM, 14,
    {{92, 0xf6185dff0bbeefa5ull},
     {208, 0x6c14b87b171d6988ull},
     {108, 0x6e9526fc4e86abbfull},
     {184, 0x84331807ec970b81ull},
     {238, 0x1815f518eb05705full}}},
   {"r11g11b10x2", 2, PIPE_FORMAT_R11G11B10_FLOAT,
    PIPE_FORMAT_R11G11B10_FLOAT, 18,
    {{366, 0xb3beb69360423e63ull},
     {648, 0xae7f96ce9d251e4cull},
     {442, 0x19bb47a2d06c32adull},
     {458, 0x4b1a48b3d909cec3ull},
     {120, 0x1aa59cf1e9af7435ull}}},
};

static int
check_golden(const char *name, bool ok, struct agx_shader_part *part,
             const struct golden_program *golden)
{
   if (!golden->size)
      return 0;

   if (!ok || part->info.binary_size != golden->size ||
       fnv1a(part->binary, part->info.binary_size) != golden->hash) {
      fprintf(stderr, "FAIL: %s changed from origin/agx3\n", name);
      return 1;
   }

   checks++;
   free(part->binary);
   return 0;
}

static int
test_homogeneous_programs_unchanged(void)
{
   for (unsigned i = 0; i < ARRAY_SIZE(homogeneous); ++i) {
      const unsigned count = homogeneous[i].count;
      const struct golden_program *golden = homogeneous[i].programs;
      enum pipe_format formats[AGX_MAX_RENDER_TARGETS] = {0};
      for (unsigned rt = 0; rt < count; ++rt)
         formats[rt] = rt ? homogeneous[i].other_formats
                          : homogeneous[i].format0;
      const struct agx_tilebuffer_layout tib =
         agx_build_tilebuffer_layout(formats, count, 1, false);
      CHECK(agx_g17p_classify_mrt(&tib) != AGX_G17P_MRT_MIXED);

      struct agx_shader_part part = {0};
      bool ok = compile_epilog(count, formats, false, false, &part);
      if (check_golden(homogeneous[i].name, ok, &part, &golden[0]))
         return 1;

      part = (struct agx_shader_part){0};
      ok = golden[1].size && compile_epilog(count, formats, true, false, &part);
      if (check_golden(homogeneous[i].name, ok, &part, &golden[1]))
         return 1;

      for (unsigned reload = 0; reload < 2; ++reload) {
         part = (struct agx_shader_part){0};
         ok = compile_background(&tib, reload ? count : 0, false,
                                 reload ? 0 : homogeneous[i].clear_push_halfs,
                                 &part);
         if (check_golden(homogeneous[i].name, ok, &part, &golden[2 + reload]))
            return 1;
      }

      struct agx_shader_key key = {
         .dev.encoding = AGX_ISA_ENCODING_AGX3B,
         .no_preamble = true,
         .g17p_r11g11b10_mrt_eot =
            agx_g17p_classify_mrt(&tib) == AGX_G17P_MRT_R11G11B10,
      };
      agx_g17p_configure_rt0_eot_key(formats[0], &key);
      part = (struct agx_shader_part){0};
      ok = compile_eot(count, formats, tib._offset_B, &key, &part);
      if (check_golden(homogeneous[i].name, ok, &part, &golden[4]))
         return 1;
   }
   return 0;
}

int
main(void)
{
   if (test_selectors_match_single_target_programs() ||
       test_layout_classification() || test_eot_programs() ||
       test_background_programs() || test_epilogs() ||
       test_homogeneous_programs_unchanged())
      return 1;

   printf("G17P mixed-format MRT: %u checks PASS\n", checks);
   return 0;
}
