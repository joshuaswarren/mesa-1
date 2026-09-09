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

static bool
check_lowering(nir_shader *shader, unsigned expected_powers)
{
   unsigned loads = 0, stores = 0, powers = 0;

   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_alu) {
               powers += nir_instr_as_alu(instr)->op == nir_op_fpow;
               continue;
            }

            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_load_local_pixel_agx) {
               loads++;
               if (nir_intrinsic_format(intr) != PIPE_FORMAT_R8_UNORM)
                  return false;
            } else if (intr->intrinsic == nir_intrinsic_store_local_pixel_agx) {
               stores++;
               if (nir_intrinsic_format(intr) != PIPE_FORMAT_R8_UNORM)
                  return false;
            }
         }
      }
   }

   return loads == 1 && stores == 1 && powers == expected_powers;
}

static bool
compile_epilog(nir_shader *shader, const char *name)
{
   agx_preprocess_nir(shader);
   struct agx_shader_key backend_key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .secondary = true,
      .no_preamble = true,
   };
   struct agx_shader_part part = {0};
   const bool ok = agx_compile_shader_nir(shader, &backend_key, &part);
   if (!ok)
      fprintf(stderr, "%s did not compile\n", name);
   free(part.binary);
   return ok;
}

int
main(void)
{
   struct agx_fs_epilog_key policy = {
      .lower_srgb = true,
      .preserve_zero_mask = true,
   };
   agx_fs_epilog_apply_device_policy(AGX_CHIP_G13G, &policy);
   if (policy.lower_srgb || policy.preserve_zero_mask) {
      fprintf(stderr, "legacy device inherited G17P epilog policy\n");
      return 1;
   }
   agx_fs_epilog_apply_device_policy(AGX_CHIP_G17P, &policy);
   if (!policy.lower_srgb || !policy.preserve_zero_mask) {
      fprintf(stderr, "G17P framebuffer epilog policy is incomplete\n");
      return 1;
   }
   agx_fs_epilog_apply_device_policy(AGX_CHIP_G17P, NULL);

   struct agx_fs_epilog_key key = {
      .link.loc_written = BITFIELD_BIT(0),
      .link.size_32 = BITFIELD_BIT(0),
      .nr_samples = 1,
   };
   agx_fs_epilog_apply_device_policy(AGX_CHIP_G17P, &key);
   memset(key.remap, -1, sizeof(key.remap));
   key.remap[0] = 0;
   key.rt_formats[0] = PIPE_FORMAT_R8G8B8A8_SRGB;
   key.blend.rt[0].colormask = 0xf;
   key.blend.rt[0].mode =
      agx_pack_blend_standard(PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE,
                              PIPE_BLENDFACTOR_ONE, PIPE_BLEND_ADD,
                              PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ONE);

   nir_builder b =
      nir_builder_init_simple_shader(0, &agx_nir_options, "G17P sRGB epilog");
   agx_nir_fs_epilog(&b, &key);

   /* Additive blending decodes the old RGB and encodes the result. */
   if (!check_lowering(b.shader, 2)) {
      fprintf(stderr, "G17P sRGB epilog did not bracket blending with conversions\n");
      return 1;
   }

   bool ok = compile_epilog(b.shader, "G17P sRGB blend epilog");
   ralloc_free(b.shader);

   /* A disabled colour mask must preserve the raw encoded tile. It needs one
    * real U8 load/store pair but no sRGB round trip. */
   key.blend.rt[0].colormask = 0;
   key.blend.rt[0].mode =
      agx_pack_blend_standard(PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE,
                              PIPE_BLENDFACTOR_ZERO, PIPE_BLEND_ADD,
                              PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO);
   nir_builder masked = nir_builder_init_simple_shader(
      0, &agx_nir_options, "G17P masked sRGB epilog");
   agx_nir_fs_epilog(&masked, &key);
   if (!check_lowering(masked.shader, 0)) {
      fprintf(stderr, "G17P zero-mask epilog did not preserve the encoded tile\n");
      return 1;
   }
   ok &= compile_epilog(masked.shader, "G17P zero-mask sRGB epilog");
   ralloc_free(masked.shader);
   return ok ? 0 : 1;
}
