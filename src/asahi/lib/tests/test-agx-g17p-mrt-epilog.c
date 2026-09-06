/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "agx_compile.h"
#include "agx_linker.h"
#include "nir_builder.h"
#include "util/format/u_formats.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

   struct agx_fs_epilog_key key = {
      .link = actual_link,
      .nr_samples = 1,
   };
   memset(key.remap, -1, sizeof(key.remap));
   key.remap[0] = 0;
   key.remap[1] = 1;
   key.rt_formats[0] = PIPE_FORMAT_R16G16B16A16_FLOAT;
   key.rt_formats[1] = PIPE_FORMAT_R8_UNORM;
   key.blend.rt[0].colormask = 0xf;
   key.blend.rt[0].mode =
      agx_pack_blend_standard(PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE,
                              PIPE_BLENDFACTOR_ZERO, PIPE_BLEND_ADD,
                              PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO);
   key.blend.rt[1].colormask = 0x1;
   key.blend.rt[1].mode =
      agx_pack_blend_standard(PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE,
                              PIPE_BLENDFACTOR_ONE, PIPE_BLEND_ADD,
                              PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ONE);

   nir_builder b =
      nir_builder_init_simple_shader(0, &agx_nir_options, "G17P MRT epilog");
   agx_nir_fs_epilog(&b, &key);
   agx_nir_lower_g17p_1x_tile_samples(b.shader);
   agx_preprocess_nir(b.shader);

   struct agx_shader_key backend_key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .secondary = true,
   };
   struct agx_shader_part part = {0};
   const bool ok = agx_compile_shader_nir(b.shader, &backend_key, &part);
   if (!ok)
      fprintf(stderr, "G17P RGBA16F+R8 MRT epilog did not compile\n");
   free(part.binary);
   free(main_part.binary);
   ralloc_free(mb.shader);
   ralloc_free(b.shader);
   return ok ? 0 : 1;
}
