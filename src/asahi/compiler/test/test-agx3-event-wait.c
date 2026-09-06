/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_compile.h"
#include "agx_compiler.h"
#include "nir_builder.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int
main(void)
{
   static const uint8_t end[] = {0x0e, 0x00, 0x00, 0x00};
   static const uint8_t loop_entry[] = {0x0f, 0x05, 0x54, 0x1a};

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "AGX3 event wait");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *address = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *expected = nir_load_preamble(&b, 1, 32, .base = 4);
   nir_loop *loop = nir_push_loop(&b);
   nir_def *value = nir_load_global(
      &b, 1, 32, address, .align_mul = sizeof(uint32_t),
      .access = ACCESS_COHERENT | ACCESS_VOLATILE);
   nir_break_if(&b, nir_ieq(&b, value, expected));
   nir_pop_loop(&b, loop);

   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.reserved_preamble = 8;
   key.has_scratch = true;
   key.promote_constants = true;
   key.promote_textures = true;
   key.no_preamble = true;
   struct agx_shader_part part = {0};

   const bool compiled = agx_compile_shader_nir(b.shader, &key, &part);
   const bool valid =
      compiled && part.binary != NULL && part.info.binary_size > sizeof(end) &&
      part.info.main_offset == 0 &&
      part.info.main_size == part.info.binary_size &&
      part.info.stats.instrs >= 4 &&
      memcmp((const uint8_t *)part.binary + part.info.binary_size - sizeof(end),
             end, sizeof(end)) == 0;

   bool has_loop_entry = false;
   for (unsigned i = 0;
        part.binary && i + sizeof(loop_entry) <= part.info.binary_size; ++i) {
      if (memcmp((const uint8_t *)part.binary + i, loop_entry,
                 sizeof(loop_entry)) == 0) {
         has_loop_entry = true;
         break;
      }
   }

   if (!valid || !has_loop_entry)
      fprintf(stderr, "AGX3b event-wait poll was not encoded safely\n");

   free(part.binary);
   ralloc_free(b.shader);
   return valid && has_loop_entry ? 0 : 1;
}
