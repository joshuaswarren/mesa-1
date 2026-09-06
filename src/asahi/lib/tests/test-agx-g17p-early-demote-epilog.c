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

static unsigned
count_intrinsic(nir_shader *s, nir_intrinsic_op op)
{
   unsigned count = 0;
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == op)
               count++;
         }
      }
   }
   return count;
}

static bool
test_single_sample_shading(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "G17P 1x sample shading");
   b.shader->info.fs.uses_sample_shading = true;
   BITSET_SET(b.shader->info.system_values_read, SYSTEM_VALUE_SAMPLE_ID);
   BITSET_SET(b.shader->info.system_values_read, SYSTEM_VALUE_SAMPLE_POS);
   BITSET_SET(b.shader->info.system_values_read, SYSTEM_VALUE_SAMPLE_MASK_IN);

   nir_def *id = nir_load_sample_id(&b);
   nir_def *pos = nir_load_sample_pos(&b);
   nir_def *mask = nir_load_sample_mask_in(&b);
   nir_def *value = nir_vec4(&b, nir_u2f32(&b, id),
                             nir_channel(&b, pos, 0),
                             nir_channel(&b, pos, 1), nir_u2f32(&b, mask));
   nir_store_output(&b, value, nir_imm_int(&b, 0), .write_mask = 0xf,
                    .io_semantics.location = FRAG_RESULT_DATA0,
                    .src_type = nir_type_float32);

   const bool progress = agx_nir_lower_g17p_1x_sample_shading(b.shader);
   const bool ok = progress && !b.shader->info.fs.uses_sample_shading &&
      count_intrinsic(b.shader, nir_intrinsic_load_sample_id) == 0 &&
      count_intrinsic(b.shader, nir_intrinsic_load_sample_pos) == 0 &&
      count_intrinsic(b.shader, nir_intrinsic_load_sample_mask_in) == 0;
   ralloc_free(b.shader);
   return ok;
}

static nir_shader *
build_early_demote_fs(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "G17P early-demote split FS");
   b.shader->info.fs.early_fragment_tests = true;
   b.shader->info.fs.uses_discard = true;
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);

   nir_def *discarded = nir_load_preamble(&b, 1, 16, .base = 0);
   nir_demote_samples(&b, discarded);
   nir_store_output(&b, nir_load_preamble(&b, 4, 32, .base = 2),
                    nir_imm_int(&b, 0), .write_mask = 0xf,
                    .io_semantics.location = FRAG_RESULT_DATA0,
                    .src_type = nir_type_float32);
   return b.shader;
}

static bool
has_lstcol(const struct agx_shader_part *part)
{
   const uint8_t *binary = part->binary;
   for (unsigned i = 0; binary && i + 12 <= part->info.binary_size; ++i) {
      if (binary[i] == 0xe7 && binary[i + 1] == 0x06 &&
          binary[i + 2] == 0x54)
         return true;
   }
   return false;
}

static unsigned
count_dynamic_tile_samples(nir_shader *s)
{
   unsigned count = 0;
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            unsigned src;
            if (intr->intrinsic == nir_intrinsic_load_local_pixel_agx)
               src = 0;
            else if (intr->intrinsic == nir_intrinsic_store_local_pixel_agx)
               src = 1;
            else
               continue;

            count += !nir_src_is_const(intr->src[src]);
         }
      }
   }
   return count;
}

int
main(void)
{
   if (!test_single_sample_shading()) {
      fprintf(stderr, "G17P 1x sample shading was not folded to pixel shading\n");
      return 1;
   }

   nir_shader *main = build_early_demote_fs();
   struct agx_fs_epilog_link_info link = {0};
   if (!agx_nir_lower_fs_output_to_epilog_g17p(main, &link) ||
       link.sample_mask_after_force_early ||
       count_intrinsic(main, nir_intrinsic_demote_samples) != 1) {
      fprintf(stderr, "G17P split moved the early demote into LSTCOL\n");
      return 1;
   }

   agx_nir_lower_sample_mask(main);
   if (count_intrinsic(main, nir_intrinsic_demote_samples) != 0 ||
       count_intrinsic(main, nir_intrinsic_sample_mask_agx) != 2) {
      fprintf(stderr, "G17P early demote did not lower to MOVMSK NIR\n");
      return 1;
   }

   struct agx_fs_epilog_key epilog_key = {
      .link = link,
      .nr_samples = 1,
   };
   epilog_key.link.sample_shading = true;
   epilog_key.link.already_ran_zs = true;
   memset(epilog_key.remap, -1, sizeof(epilog_key.remap));
   epilog_key.remap[0] = 0;
   epilog_key.rt_formats[0] = PIPE_FORMAT_R8G8B8A8_UNORM;
   epilog_key.blend.rt[0].colormask = 0xf;
   epilog_key.blend.rt[0].mode =
      agx_pack_blend_standard(PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE,
                              PIPE_BLENDFACTOR_ZERO, PIPE_BLEND_ADD,
                              PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO);

   nir_builder b =
      nir_builder_init_simple_shader(0, &agx_nir_options, "G17P FS epilog");
   agx_nir_fs_epilog(&b, &epilog_key);
   if (count_dynamic_tile_samples(b.shader) != 1 ||
       !agx_nir_lower_g17p_1x_tile_samples(b.shader) ||
       count_dynamic_tile_samples(b.shader) != 0) {
      fprintf(stderr, "G17P 1x sample-shaded epilog was not canonicalized\n");
      return 1;
   }
   agx_preprocess_nir(b.shader);

   struct agx_shader_key backend_key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .secondary = true,
      .no_preamble = true,
   };
   struct agx_shader_part part = {0};
   bool ok = agx_compile_shader_nir(b.shader, &backend_key, &part) &&
             has_lstcol(&part);
   if (!ok)
      fprintf(stderr, "G17P early-demote epilog did not compile to LSTCOL\n");

   free(part.binary);
   ralloc_free(b.shader);

   /* Fully disabled G17P colour writes use an identity blend to materialize
    * the existing tile value. Validate the exact load/store shape and backend
    * widths independently of the Vulkan key construction. */
   epilog_key.link.sample_shading = false;
   epilog_key.blend.rt[0].mode =
      agx_pack_blend_standard(PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ZERO,
                              PIPE_BLENDFACTOR_ONE, PIPE_BLEND_ADD,
                              PIPE_BLENDFACTOR_ZERO, PIPE_BLENDFACTOR_ONE);
   b = nir_builder_init_simple_shader(0, &agx_nir_options,
                                      "G17P masked colour identity epilog");
   agx_nir_fs_epilog(&b, &epilog_key);
   ok &= count_intrinsic(b.shader, nir_intrinsic_load_local_pixel_agx) == 1;
   ok &= count_intrinsic(b.shader, nir_intrinsic_store_local_pixel_agx) == 1;
   agx_preprocess_nir(b.shader);
   memset(&part, 0, sizeof(part));
   ok &= agx_compile_shader_nir(b.shader, &backend_key, &part) &&
         has_lstcol(&part);
   if (!ok)
      fprintf(stderr, "G17P masked colour identity epilog failed\n");
   free(part.binary);
   ralloc_free(b.shader);
   ralloc_free(main);
   return ok ? 0 : 1;
}
