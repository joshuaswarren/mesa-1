/* SPDX-License-Identifier: MIT */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "util/format/u_formats.h"
#include "agx3_graphics_program.h"
#include "agx_compile.h"
#include "agx_linker.h"
#include "agx_nir.h"
#include "agx_tilebuffer.h"
#include "nir_builder.h"

static void
dump(const char *name, enum pipe_format format,
     const struct agx_shader_part *part)
{
   if (!getenv("AGX3_DUMP_BINARY"))
      return;
   printf("quad-%s format=%u gprs=%u staging=%u partitions=%u bytes=", name,
          format, part->info.nr_gprs, part->info.agx3_staging_halfs,
          part->info.agx3_colour_link.colour_partitions);
   for (unsigned i = 0; i < part->info.main_size; ++i)
      printf("%02x", ((uint8_t *)part->binary)[part->info.main_offset + i]);
   puts("");
}

static void
check_format(enum pipe_format format, unsigned components)
{
   enum pipe_format formats[AGX_MAX_RENDER_TARGETS] = {format, format, format,
                                                       format};
   struct agx_tilebuffer_layout tib =
      agx_build_tilebuffer_layout(formats, 4, 1, false);
   assert(agx_g17p_classify_mrt(&tib) ==
          (components == 4 ? AGX_G17P_MRT_RGBA32_QUAD : AGX_G17P_MRT_R32_QUAD));
   assert(agx_g17p_rt0_tile_layout_supported(&tib));
   assert(tib.sample_size_B == 16 * components);
   for (unsigned i = 0; i < 4; ++i)
      assert(agx_tilebuffer_offset_B(&tib, i) == i * components * 4);
   struct agx_tilebuffer_layout invalid = tib;
   invalid.spilled[3] = true;
   assert(agx_g17p_classify_mrt(&invalid) == AGX_G17P_MRT_NONE);
   invalid = tib;
   invalid._offset_B[3]++;
   assert(agx_g17p_classify_mrt(&invalid) == AGX_G17P_MRT_NONE);
   invalid = tib;
   invalid.logical_format[4] = format;
   assert(agx_g17p_classify_mrt(&invalid) == AGX_G17P_MRT_NONE);
   invalid = tib;
   invalid.nr_samples = 2;
   assert(agx_g17p_classify_mrt(&invalid) == AGX_G17P_MRT_NONE);

   struct agx_fs_epilog_key epilog = {.link.loc_written = 15,
                                      .link.size_32 = 15,
                                      .nr_samples = 1};
   memset(epilog.remap, -1, sizeof(epilog.remap));
   for (unsigned i = 0; i < 4; ++i) {
      epilog.remap[i] = i;
      epilog.rt_formats[i] = format;
      epilog.blend.rt[i].colormask = (1u << components) - 1;
      epilog.blend.rt[i].mode = agx_pack_blend_standard(
         PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO,
         PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO);
   }
   agx_fs_epilog_apply_device_policy(AGX_CHIP_G17P, &epilog);
   nir_builder b =
      nir_builder_init_simple_shader(0, &agx_nir_options, "G17P quad epilog");
   agx_nir_fs_epilog(&b, &epilog);
   agx_nir_lower_g17p_1x_tile_samples(b.shader);
   agx_preprocess_nir(b.shader);
   struct agx_shader_key key = {.dev.encoding = AGX_ISA_ENCODING_AGX3B,
                                .secondary = true};
   struct agx_shader_part part = {0};
   assert(agx_compile_shader_nir(b.shader, &key, &part));
   assert(part.info.agx3_colour_link.colour_partitions == 3);
   assert(part.info.agx3_colour_link.entry_emitted);
   assert(!part.info.scratch_size);
   dump("epilog", format, &part);
   free(part.binary);
   ralloc_free(b.shader);

   b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, &agx_nir_options,
                                      "G17P quad BG clear");
   b.shader->info.fs.untyped_color_outputs = true;
   for (unsigned i = 0; i < 4; ++i) {
      nir_store_output(&b,
                       nir_load_preamble(&b, components, 32, .base = 4 + i * 8),
                       nir_imm_int(&b, 0), .write_mask = (1u << components) - 1,
                       .src_type = nir_type_uint32,
                       .io_semantics.location = FRAG_RESULT_DATA0 + i,
                       .io_semantics.num_slots = 1);
      b.shader->info.outputs_written |= BITFIELD64_BIT(FRAG_RESULT_DATA0 + i);
   }
   agx_preprocess_nir(b.shader);
   unsigned bindless_base = 0;
   NIR_PASS(_, b.shader, agx_nir_lower_tilebuffer, &tib, NULL, &bindless_base,
            NULL, NULL, false, true);
   NIR_PASS(_, b.shader, agx_nir_lower_monolithic_msaa, tib.nr_samples);
   NIR_PASS(_, b.shader, agx_nir_lower_multisampled_image_store);
   NIR_PASS(_, b.shader, agx_nir_lower_g17p_tile_samples, tib.nr_samples);
   key = (struct agx_shader_key){
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .fs.ignore_tib_dependencies = true,
      .reserved_preamble = AGX3_G17P_QUAD_CLEAR_PUSH_HALFS,
      .no_preamble = true};
   part = (struct agx_shader_part){0};
   assert(agx_compile_shader_nir(b.shader, &key, &part));
   assert(part.info.push_count == AGX3_G17P_QUAD_CLEAR_PUSH_HALFS);
   assert(!part.info.scratch_size && !part.info.agx3_colour_link.entry_emitted);
   dump("clear", format, &part);
   free(part.binary);
   ralloc_free(b.shader);
}

int
main(void)
{
   check_format(PIPE_FORMAT_R32G32B32A32_FLOAT, 4);
   check_format(PIPE_FORMAT_R32G32B32A32_UINT, 4);
   check_format(PIPE_FORMAT_R32G32B32A32_SINT, 4);
   check_format(PIPE_FORMAT_R32_FLOAT, 1);
   puts("quad layouts, epilogs and clear shaders PASS");
   return 0;
}
