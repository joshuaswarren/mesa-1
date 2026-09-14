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

/*
 * A colour mask of zero writes nothing, so on G17P the epilog must still
 * reload the tile and store it back: a covered fragment that stores nothing
 * loses the tile it was drawn over.
 *
 * Formats whose tilebuffer writes cannot be masked -- RGB10A2 and RG11B10F --
 * used to leave the mask to nir_lower_blend, which deletes the store outright
 * for a zero mask, so the preserve path never ran and the tile was lost.
 * KHR-GL30.framebuffer_blit.scissor_blit found it on a ten-bit window, where
 * u_blitter clears depth with a full screen quad and colour writes masked off.
 *
 * Count the tilebuffer accesses the epilog keeps. One load and one store is
 * the preserve pair; none at all is the tile going missing.
 */

static bool
counts(nir_shader *shader, unsigned *loads, unsigned *stores)
{
   *loads = 0;
   *stores = 0;

   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_load_local_pixel_agx)
               (*loads)++;
            else if (intr->intrinsic == nir_intrinsic_store_local_pixel_agx)
               (*stores)++;
         }
      }
   }

   return *loads == 1 && *stores == 1;
}

static bool
check(const char *what, enum pipe_format format)
{
   struct agx_fs_epilog_key key = {
      .link.loc_written = BITFIELD_BIT(0),
      .link.size_32 = BITFIELD_BIT(0),
      .nr_samples = 1,
   };
   agx_fs_epilog_apply_device_policy(AGX_CHIP_G17P, &key);
   memset(key.remap, -1, sizeof(key.remap));
   key.remap[0] = 0;
   key.rt_formats[0] = format;
   key.blend.rt[0].colormask = 0;
   key.blend.rt[0].mode =
      agx_pack_blend_standard(PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE,
                              PIPE_BLENDFACTOR_ZERO, PIPE_BLEND_ADD,
                              PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO);

   nir_builder b =
      nir_builder_init_simple_shader(0, &agx_nir_options, "G17P zero mask");
   agx_nir_fs_epilog(&b, &key);

   unsigned loads, stores;
   const bool preserved = counts(b.shader, &loads, &stores);

   agx_preprocess_nir(b.shader);
   struct agx_shader_key backend_key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .secondary = true,
      .no_preamble = true,
   };
   struct agx_shader_part part = {0};
   const bool compiled = agx_compile_shader_nir(b.shader, &backend_key, &part);
   free(part.binary);
   ralloc_free(b.shader);

   printf("%-28s %u load %u store, %s %s\n", what, loads, stores,
          compiled ? "compiles" : "DID NOT COMPILE",
          preserved && compiled ? "" : "FAIL");

   return preserved && compiled;
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   bool ok = true;

   /* Maskable, so the mask always reached the tilebuffer lowering. */
   ok &= check("rgba8", PIPE_FORMAT_R8G8B8A8_UNORM);

   /* The formats ail_isa_format_supports_mask refuses. Both spellings of the
    * ten-bit tile appear: the window is the BGRA one. */
   ok &= check("rgb10_a2", PIPE_FORMAT_R10G10B10A2_UNORM);
   ok &= check("bgr10_a2", PIPE_FORMAT_B10G10R10A2_UNORM);
   ok &= check("r11g11b10f", PIPE_FORMAT_R11G11B10_FLOAT);

   glsl_type_singleton_decref();
   return ok ? 0 : 1;
}
