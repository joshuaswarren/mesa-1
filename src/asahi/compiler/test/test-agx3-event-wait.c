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
   static const uint8_t outer_entry[] = {0x0f, 0x05, 0x54, 0x01};
   static const uint8_t loop_entry[] = {0x0f, 0x05, 0x54, 0x1a};
   static const uint8_t loop_exit[] = {0x0f, 0x06, 0x04, 0x02, 0x00, 0x00};
   static const uint8_t outer_exit[] = {0x0f, 0x06, 0x04, 0x01, 0x00, 0x00};
   static const uint8_t expected_binary[] = {
      0x1b, 0x04, 0x01, 0x08, 0x2b, 0x06, 0x01, 0x08,
      0x0a, 0x05, 0x23, 0x03, 0x02, 0x00, 0x05, 0x00, 0x00, 0x00,
      0x0f, 0x05, 0x54, 0x01,
      0x0f, 0x05, 0x54, 0x1a,
      0x1b, 0x04, 0x01, 0x08,
      0x67, 0x00, 0x54, 0x04, 0x00, 0x00, 0x00, 0x00,
      0x11, 0x00, 0x00, 0x40, 0x02, 0x00,
      0x0b, 0x01, 0x0f, 0x01, 0x00, 0x20, 0x00, 0x80, 0x00, 0x00,
      0x0a, 0x05, 0x23, 0x03, 0x06, 0x00, 0x07, 0x00, 0x00, 0x00,
      0x8f, 0x04, 0x54, 0x02,
      0x0f, 0x00, 0x54, 0xd6, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00,
      0x0f, 0x06, 0x04, 0x02, 0x00, 0x00,
      0x0f, 0x06, 0x04, 0x01, 0x00, 0x00,
      0x0e, 0x00, 0x00, 0x00,
   };

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "AGX3 event wait");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *address = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *expected = nir_load_preamble(&b, 1, 32, .base = 4);
   nir_def *zero = nir_load_preamble(&b, 1, 32, .base = 6);
   nir_if *guard = nir_push_if(&b, nir_ult(&b, zero, expected));
   nir_loop *loop = nir_push_loop(&b);
   nir_def *value = nir_load_global(
      &b, 1, 32, address, .align_mul = sizeof(uint32_t),
      .access = ACCESS_COHERENT | ACCESS_VOLATILE);
   nir_break_if(&b, nir_ieq(&b, value, expected));
   nir_pop_loop(&b, loop);
   nir_pop_if(&b, guard);

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
      part.info.binary_size == sizeof(expected_binary) &&
      memcmp(part.binary, expected_binary, sizeof(expected_binary)) == 0 &&
      memcmp((const uint8_t *)part.binary + part.info.binary_size - sizeof(end),
             end, sizeof(end)) == 0;

   unsigned control = 0;
   const uint8_t *need[] = {outer_entry, loop_entry, loop_exit, outer_exit};
   const size_t need_size[] = {sizeof(outer_entry), sizeof(loop_entry),
                               sizeof(loop_exit), sizeof(outer_exit)};
   for (unsigned i = 0; part.binary && i < part.info.binary_size && control < 4;
        ++i) {
      if (i + need_size[control] <= part.info.binary_size &&
          memcmp((const uint8_t *)part.binary + i, need[control],
                 need_size[control]) == 0) {
         ++control;
      }
   }

   if (!valid || control != 4) {
      fprintf(stderr, "AGX3b event-wait poll was not encoded safely\n");
      for (unsigned i = 0; part.binary && i < part.info.binary_size; ++i)
         fprintf(stderr, "%02x", ((const uint8_t *)part.binary)[i]);
      fputc('\n', stderr);
   }

   free(part.binary);
   ralloc_free(b.shader);
   return valid && control == 4 ? 0 : 1;
}
