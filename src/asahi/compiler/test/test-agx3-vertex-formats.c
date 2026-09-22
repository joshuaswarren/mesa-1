/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Exercise the complete NIR-to-native pipeline for the byte conversions and
 * normalized device loads used by glVertexAttribPointer. These are compiler
 * regressions; the Piglit vertexattribpointer test supplies the pixel oracle.
 */

#include "agx_compile.h"
#include "nir_builder.h"

#include <stdio.h>
#include <stdlib.h>

static bool
compile_format(enum pipe_format format, unsigned components, unsigned bits,
               bool convert, bool is_signed, bool dynamic_offset)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "G17P vertex format regression");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *input = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *output = nir_load_preamble(&b, 1, 64, .base = 4);
   nir_def *offset = dynamic_offset ? nir_load_preamble(&b, 1, 32, .base = 8)
                                    : nir_imm_int(&b, 3);
   nir_def *value;
   if (convert) {
      value = nir_load_global(&b, components, 8,
                             nir_iadd(&b, input, nir_u2u64(&b, offset)),
                             .align_mul = 1);
      value = is_signed ? nir_i2fN(&b, value, bits)
                        : nir_u2fN(&b, value, bits);
   } else {
      value = nir_load_agx(&b, components, bits, input, offset,
                           .format = format, .base = 1);
   }
   nir_store_global(&b, value, output, .align_mul = 4);
   agx_preprocess_nir_g17p(b.shader);

   struct agx_shader_key key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .reserved_preamble = 10,
      .no_preamble = true,
   };
   struct agx_shader_part part = {0};
   bool ok = agx_compile_shader_nir(b.shader, &key, &part);
   ok &= part.binary != NULL && part.info.main_size > 0 &&
         part.info.scratch_size == 0;
   if (!ok) {
      fprintf(stderr, "format=%u components=%u bits=%u convert=%u signed=%u "
                      "dynamic=%u failed native compilation\n",
              format, components, bits, convert, is_signed, dynamic_offset);
   }
   free(part.binary);
   ralloc_free(b.shader);
   return ok;
}

int
main(void)
{
   static const enum pipe_format formats[] = {
      PIPE_FORMAT_R8_UNORM, PIPE_FORMAT_R8_SNORM,
      PIPE_FORMAT_R16_UNORM, PIPE_FORMAT_R16_SNORM,
   };
   unsigned passed = 0, total = 0;
   for (unsigned components = 1; components <= 4; ++components) {
      for (unsigned bits = 16; bits <= 32; bits *= 2) {
         for (unsigned dynamic = 0; dynamic < 2; ++dynamic) {
            for (unsigned i = 0; i < ARRAY_SIZE(formats); ++i) {
               passed += compile_format(formats[i], components, bits,
                                        false, false, dynamic);
               ++total;
            }
            for (unsigned is_signed = 0; is_signed < 2; ++is_signed) {
               passed += compile_format(PIPE_FORMAT_NONE, components, bits,
                                        true, is_signed, dynamic);
               ++total;
            }
         }
      }
   }
   printf("agx3_vertex_formats: %u/%u native compile cases passed\n", passed, total);
   return passed == total ? 0 : 1;
}
