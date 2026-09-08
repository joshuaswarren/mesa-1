/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_abi.h"
#include "agx_compile.h"
#include "agx_linker.h"
#include "nir_builder.h"
#include "util/format/u_formats.h"

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

static bool
compile_blend_epilog(uint16_t requested_base, uint16_t expected_base)
{
   struct agx_fs_epilog_key key = {
      .link = {
         .size_32 = BITFIELD_BIT(0),
         .loc_written = BITFIELD_BIT(0),
      },
      .nr_samples = 1,
      .preserve_zero_mask = true,
      .blend_constant_base = requested_base,
   };
   memset(key.remap, -1, sizeof(key.remap));
   key.remap[0] = 0;
   key.rt_formats[0] = PIPE_FORMAT_R8G8B8A8_UNORM;
   key.blend.rt[0].colormask = 0xf;
   key.blend.rt[0].mode = agx_pack_blend_standard(
      PIPE_BLEND_ADD, PIPE_BLENDFACTOR_CONST_COLOR,
      PIPE_BLENDFACTOR_INV_CONST_COLOR, PIPE_BLEND_ADD,
      PIPE_BLENDFACTOR_CONST_ALPHA, PIPE_BLENDFACTOR_INV_CONST_ALPHA);

   nir_builder b = nir_builder_init_simple_shader(
      0, &agx_nir_options, "G17P relocated blend constants");
   agx_nir_fs_epilog(&b, &key);

   unsigned found = 0;
   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_load_preamble)
               continue;

            const unsigned base = nir_intrinsic_base(intr);
            if (base < expected_base || base > expected_base + 6 ||
                ((base - expected_base) & 1)) {
               ralloc_free(b.shader);
               return false;
            }
            found |= BITFIELD_BIT((base - expected_base) / 2);
         }
      }
   }

   if (found != BITFIELD_MASK(4)) {
      ralloc_free(b.shader);
      return false;
   }

   agx_preprocess_nir(b.shader);
   struct agx_shader_key backend_key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .secondary = true,
      .no_preamble = true,
   };
   struct agx_shader_part part = {0};
   const bool ok = agx_compile_shader_nir(b.shader, &backend_key, &part);

   free(part.binary);
   ralloc_free(b.shader);
   return ok;
}

int
main(void)
{
   return compile_blend_epilog(0, AGX_ABI_FUNI_BLEND_R) &&
          compile_blend_epilog(28, 28)
             ? 0
             : 1;
}
