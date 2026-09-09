/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "agx_compile.h"
#include "agx3_pixel_core.h"
#include "agx_linker.h"
#include "nir_builder.h"
#include "util/format/u_formats.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t
blend_mode(bool additive)
{
   return agx_pack_blend_standard(
      PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE,
      additive ? PIPE_BLENDFACTOR_ONE : PIPE_BLENDFACTOR_ZERO,
      PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE,
      additive ? PIPE_BLENDFACTOR_ONE : PIPE_BLENDFACTOR_ZERO);
}

static bool
compile_epilog(const struct agx_fs_epilog_link_info *link,
               enum pipe_format rt1_format, bool rt0_additive,
               bool rt1_additive, struct agx_shader_part *part)
{
   struct agx_fs_epilog_key key = {
      .link = *link,
      .nr_samples = 1,
   };
   memset(key.remap, -1, sizeof(key.remap));
   key.remap[0] = 0;
   key.remap[1] = 1;
   key.rt_formats[0] = PIPE_FORMAT_R16G16B16A16_FLOAT;
   key.rt_formats[1] = rt1_format;
   key.blend.rt[0].colormask = 0xf;
   key.blend.rt[0].mode = blend_mode(rt0_additive);
   key.blend.rt[1].colormask = 0x1;
   key.blend.rt[1].mode = blend_mode(rt1_additive);

   nir_builder b =
      nir_builder_init_simple_shader(0, &agx_nir_options, "G17P MRT epilog");
   agx_nir_fs_epilog(&b, &key);
   agx_nir_lower_g17p_1x_tile_samples(b.shader);
   agx_preprocess_nir(b.shader);

   struct agx_shader_key backend_key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .secondary = true,
   };
   const bool ok = agx_compile_shader_nir(b.shader, &backend_key, part);
   ralloc_free(b.shader);
   return ok;
}

static bool
compile_rgba8_mrt_epilog(const struct agx_fs_epilog_link_info *link,
                         enum pipe_format rt1_format, bool rt1_additive,
                         struct agx_shader_part *part)
{
   struct agx_fs_epilog_key key = {
      .link = *link,
      .nr_samples = 1,
   };
   memset(key.remap, -1, sizeof(key.remap));
   key.remap[0] = 0;
   key.remap[1] = 1;
   key.rt_formats[0] = PIPE_FORMAT_R8G8B8A8_UNORM;
   key.rt_formats[1] = rt1_format;
   key.blend.rt[0].colormask = 0xf;
   key.blend.rt[0].mode = blend_mode(false);
   key.blend.rt[1].colormask = 0xf;
   key.blend.rt[1].mode = blend_mode(rt1_additive);

   nir_builder b = nir_builder_init_simple_shader(
      0, &agx_nir_options, "G17P two-RGBA8 MRT epilog");
   agx_nir_fs_epilog(&b, &key);
   agx_nir_lower_g17p_1x_tile_samples(b.shader);
   agx_preprocess_nir(b.shader);

   struct agx_shader_key backend_key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .secondary = true,
   };
   const bool ok = agx_compile_shader_nir(b.shader, &backend_key, part);
   ralloc_free(b.shader);
   return ok;
}

static bool
compile_vkquake_mboit_epilog(const struct agx_fs_epilog_link_info *link,
                             struct agx_shader_part *part)
{
   struct agx_fs_epilog_key key = {
      .link = *link,
      .nr_samples = 1,
      .lower_srgb = true,
      .preserve_zero_mask = true,
   };
   memset(key.remap, -1, sizeof(key.remap));
   key.remap[0] = 0;
   key.remap[1] = 1;
   key.rt_formats[0] = PIPE_FORMAT_R16G16B16A16_FLOAT;
   key.rt_formats[1] = PIPE_FORMAT_R8_UNORM;
   key.blend.rt[0].colormask = 0xf;
   key.blend.rt[0].mode = agx_pack_blend_standard(
      PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ONE,
      PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ONE);
   key.blend.rt[1].colormask = 0x1;
   key.blend.rt[1].mode = agx_pack_blend_standard(
      PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ZERO,
      PIPE_BLENDFACTOR_INV_SRC_COLOR, PIPE_BLEND_ADD,
      PIPE_BLENDFACTOR_ZERO, PIPE_BLENDFACTOR_INV_SRC_ALPHA);

   nir_builder b = nir_builder_init_simple_shader(
      0, &agx_nir_options, "G17P vkQuake MBOIT epilog");
   agx_nir_fs_epilog(&b, &key);
   agx_nir_lower_g17p_1x_tile_samples(b.shader);
   agx_preprocess_nir(b.shader);

   struct agx_shader_key backend_key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .secondary = true,
   };
   const bool ok = agx_compile_shader_nir(b.shader, &backend_key, part);
   ralloc_free(b.shader);
   return ok;
}

int
main(void)
{
   nir_builder mb = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "G17P MRT main");
   nir_store_output(&mb, nir_imm_vec4(&mb, 0.75f, 0.5f, 0.25f, 1.0f),
                    nir_imm_int(&mb, 0), .write_mask = 0xf,
                    .io_semantics.location = FRAG_RESULT_DATA0,
                    .src_type = nir_type_float32);
   nir_store_output(&mb, nir_imm_float(&mb, 0.5f), nir_imm_int(&mb, 0),
                    .write_mask = 0x1,
                    .io_semantics.location = FRAG_RESULT_DATA1,
                    .src_type = nir_type_float32);
   mb.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0) |
                                     BITFIELD64_BIT(FRAG_RESULT_DATA1);
   struct agx_fs_epilog_link_info actual_link = {0};
   agx_nir_lower_fs_output_to_epilog_g17p(mb.shader, &actual_link);
   agx_preprocess_nir(mb.shader);
   struct agx_shader_key main_key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .no_stop = true,
      .no_preamble = true,
      .promote_constants = true,
      .promote_textures = true,
   };
   struct agx_shader_part main_part = {0};
   if (!agx_compile_shader_nir(mb.shader, &main_key, &main_part)) {
      fprintf(stderr, "G17P RGBA16F+R8 MRT main did not compile\n");
      return 1;
   }
   if (actual_link.loc_written != 3 || actual_link.size_32 != 3 ||
       actual_link.loc0_w_1) {
      fprintf(stderr, "G17P MRT main did not preserve the full export ABI\n");
      return 1;
   }

   struct agx_shader_part vkquake_part = {0};
   const bool vkquake_ok =
      compile_vkquake_mboit_epilog(&actual_link, &vkquake_part) &&
      vkquake_part.binary && vkquake_part.info.main_offset == 0 &&
      vkquake_part.info.main_size == vkquake_part.info.binary_size &&
      vkquake_part.info.agx3_colour_link.entry_emitted &&
      vkquake_part.info.agx3_colour_link.straight_line &&
      vkquake_part.info.agx3_colour_link.entry_offset == 0;
   if (!vkquake_ok) {
      fprintf(stderr,
              "G17P vkQuake MBOIT epilog did not preserve one transaction\n");
      return 1;
   }
   free(vkquake_part.binary);

   static const uint8_t expected_main[] = {
      0x4b,0x06,0x01,0x08,0x5b,0x00,0x01,0x08,0x6b,0x04,
      0x01,0x08,0x7b,0x02,0x01,0x08,0x8b,0x00,0x01,0x08,
   };
   if (main_part.info.main_size != sizeof(expected_main) ||
       memcmp((uint8_t *)main_part.binary + main_part.info.main_offset,
              expected_main, sizeof(expected_main)) != 0) {
      fprintf(stderr, "G17P MRT main did not keep distinct RT0/RT1 exports\n");
      return 1;
   }

   struct agx_shader_part part = {0};
   const bool ok = compile_epilog(&actual_link, PIPE_FORMAT_R8_UNORM, false,
                                  true, &part) &&
                   part.info.agx3_colour_link.entry_emitted &&
                   part.info.agx3_colour_link.straight_line &&
                   part.info.agx3_colour_link.entry_offset == 0 &&
                   part.info.main_offset == 0 &&
                   part.info.main_size == part.info.binary_size;
   if (!ok)
      fprintf(stderr, "G17P RGBA16F+R8 MRT epilog did not compile\n");
   static const uint8_t expected_epilog[] = {
      0x87,0x02,0x54,0x00,0x06,0x00,0x87,0x02,0x54,0x0c,0x08,0x00,
      0x01,0x09,0x1c,0x81,0x00,0x02,0x01,0x0b,0x1c,0x81,0x04,0x02,
      0x11,0x0d,0x1c,0x81,0x00,0x02,0x11,0x0f,0x1c,0x81,0x04,0x02,
      0xe7,0x06,0x54,0x00,0x00,0x00,0x01,0x0e,0x00,0x00,0x00,0x00,
      0x67,0x0e,0x54,0x00,0x00,0x04,0x01,0xc2,0x02,0x00,0x00,0x00,
      0x17,0x04,0x56,0x00,0x01,0x00,0x14,0x6a,0x19,0x11,0x1c,0x81,
      0x01,0x02,0x00,0x02,0x09,0x01,0x18,0x03,0x97,0x04,0x54,0x00,
      0x02,0x00,0x02,0x10,0x44,0xc0,0xe7,0x06,0x54,0x00,0x00,0x04,
      0x01,0x42,0x00,0x00,0x00,0x00,0x07,0x02,0x54,0x0c,0x02,0x00,
      0x0e,0x00,0x00,0x00,
   };
   uint8_t entry[AGX3_PIXEL_MAX_LENGTH] = {0};
   uint8_t entry_size = 0;
   static const uint8_t stop[] = {0x0e, 0x00, 0x00, 0x00};
   if (ok &&
       (!agx3_pixel_colour_entry(AGX_ISA_ENCODING_AGX3B, entry,
                                 &entry_size) ||
        entry_size != AGX3_PIXEL_MAX_LENGTH ||
        memcmp((uint8_t *)part.binary + part.info.main_offset, entry,
               entry_size) != 0 ||
        memcmp((uint8_t *)part.binary + part.info.main_offset +
                  part.info.main_size - sizeof(stop),
               stop, sizeof(stop)) != 0)) {
      fprintf(stderr, "G17P MRT epilog entry/stop ABI changed\n");
      return 1;
   }
   if (ok && (part.info.main_size != sizeof(expected_epilog) ||
              memcmp((uint8_t *)part.binary + part.info.main_offset,
                     expected_epilog, sizeof(expected_epilog)) != 0)) {
      fprintf(stderr, "G17P MRT epilog bytes changed size=%u bytes=",
              part.info.main_size);
      for (unsigned i = 0; i < part.info.main_size; ++i)
         fprintf(stderr, "%02x", ((uint8_t *)part.binary +
                                   part.info.main_offset)[i]);
      fputc('\n', stderr);
      return 1;
   }
   free(part.binary);

   memset(&part, 0, sizeof(part));
   if (!compile_rgba8_mrt_epilog(&actual_link,
                                 PIPE_FORMAT_R8G8B8A8_UNORM, false, &part)) {
      fprintf(stderr, "G17P two-RGBA8 MRT did not compile\n");
      return 1;
   }
   static const uint8_t expected_rgba8_mrt[] = {
      0x97,0x0c,0x54,0x00,0x02,0x10,0x2a,0x54,0x44,0xc2,
      0x97,0x04,0x54,0x01,0x02,0x18,0x3a,0x54,0x44,0xc2,
      0x97,0x04,0x54,0x02,0x02,0x20,0x4a,0x54,0x44,0xc2,
      0x97,0x04,0x54,0x03,0x02,0x28,0x5a,0x54,0x44,0xc2,
      0x87,0x02,0x54,0x00,0x06,0x00,
      0x87,0x02,0x54,0x30,0x08,0x00,
      0xe7,0x06,0x54,0x02,0x00,0x02,0x01,0x4e,0x00,0x00,0x00,0x00,
      0x07,0x02,0x54,0x30,0x00,0x00,
      0x87,0x02,0x54,0x0c,0x08,0x00,
      0xe7,0x06,0x54,0x00,0x00,0x00,0x01,0x4e,0x00,0x00,0x00,0x00,
      0x07,0x02,0x54,0x0c,0x02,0x00,
      0x0e,0x00,0x00,0x00,
   };
   static const uint8_t apple_transaction[] = {
      0x87,0x02,0x54,0x00,0x06,0x00,
      0x87,0x02,0x54,0x30,0x08,0x00,
      0xe7,0x06,0x54,0x02,0x00,0x02,0x01,0x4e,0x00,0x00,0x00,0x00,
      0x07,0x02,0x54,0x30,0x00,0x00,
      0x87,0x02,0x54,0x0c,0x08,0x00,
      0xe7,0x06,0x54,0x00,0x00,0x00,0x01,0x4e,0x00,0x00,0x00,0x00,
      0x07,0x02,0x54,0x0c,0x02,0x00,
      0x0e,0x00,0x00,0x00,
   };
   const uint8_t *rgba8_code =
      (uint8_t *)part.binary + part.info.main_offset;
   if (part.info.main_size != sizeof(expected_rgba8_mrt) ||
       memcmp(rgba8_code, expected_rgba8_mrt,
              sizeof(expected_rgba8_mrt)) ||
       !part.info.agx3_colour_link.entry_emitted ||
       !part.info.agx3_colour_link.straight_line ||
       part.info.agx3_colour_link.entry_offset != 40 ||
       memcmp(rgba8_code + 40, apple_transaction,
              sizeof(apple_transaction))) {
      fprintf(stderr, "G17P two-RGBA8 MRT transaction changed\n");
      return 1;
   }
   free(part.binary);

   const struct {
      enum pipe_format rt1_format;
      bool rt1_additive;
   } rgba8_rejected[] = {
      {PIPE_FORMAT_R16G16B16A16_FLOAT, false},
      {PIPE_FORMAT_R8G8B8A8_UNORM, true},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(rgba8_rejected); ++i) {
      memset(&part, 0xa5, sizeof(part));
      if (compile_rgba8_mrt_epilog(&actual_link,
                                   rgba8_rejected[i].rt1_format,
                                   rgba8_rejected[i].rt1_additive, &part) ||
          part.binary || part.info.binary_size != 0) {
         fprintf(stderr, "unsupported G17P RGBA8 MRT shape encoded\n");
         return 1;
      }
   }

   const struct {
      enum pipe_format rt1_format;
      bool rt0_additive;
      bool rt1_additive;
   } rejected[] = {
      {PIPE_FORMAT_R8_UNORM, false, false},
      {PIPE_FORMAT_R8_UNORM, true, false},
      {PIPE_FORMAT_R8_UINT, false, true},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(rejected); ++i) {
      memset(&part, 0xa5, sizeof(part));
      if (compile_epilog(&actual_link, rejected[i].rt1_format,
                         rejected[i].rt0_additive,
                         rejected[i].rt1_additive, &part) || part.binary ||
          part.info.binary_size != 0) {
         fprintf(stderr, "unsupported G17P MRT shape did not fail closed\n");
         return 1;
      }
   }

   free(main_part.binary);
   ralloc_free(mb.shader);
   return ok ? 0 : 1;
}
