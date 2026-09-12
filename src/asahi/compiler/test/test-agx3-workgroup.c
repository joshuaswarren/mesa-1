/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_compile.h"
#include "nir_builder.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned
count_pattern(const uint8_t *bytes, size_t size, const uint8_t *pattern,
              size_t pattern_size)
{
   unsigned count = 0;
   for (size_t i = 0; i + pattern_size <= size; ++i)
      count += memcmp(bytes + i, pattern, pattern_size) == 0;
   return count;
}

int
main(void)
{
   static const uint8_t barrier[] = {0x07, 0x04, 0x54,
                                     0x61, 0x09, 0x00};
   static const uint8_t end[] = {0x0e, 0x00, 0x00, 0x00};
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "G17P shared barrier exchange");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.shared_size = 32 * sizeof(uint32_t);

   nir_def *lane = nir_load_local_invocation_index(&b);
   nir_def *reverse = nir_isub(&b, nir_imm_int(&b, 31), lane);
   nir_def *offset = nir_imul_imm(&b, reverse, sizeof(uint32_t));
   nir_store_shared(&b, nir_iadd_imm(&b, lane, 0x1234), offset,
                    .align_mul = 4);
   nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
               .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_shared);
   nir_def *peer = nir_load_shared(&b, 1, 32,
                                   nir_imul_imm(&b, lane, sizeof(uint32_t)),
                                   .align_mul = 4);
   nir_def *output = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_store_global(&b, nir_iadd(&b, peer, lane), output, .align_mul = 4);
   agx_preprocess_nir_g17p(b.shader);

   struct agx_shader_key key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .reserved_preamble = 4,
      .no_preamble = true,
   };
   struct agx_shader_part part = {0};
   const bool compiled = agx_compile_shader_nir(b.shader, &key, &part);
   const bool valid =
      compiled && part.binary != NULL &&
      part.info.local_size == 32 * sizeof(uint32_t) &&
      part.info.workgroup_size[0] == 32 && part.info.workgroup_size[1] == 1 &&
      part.info.workgroup_size[2] == 1 &&
      count_pattern(part.binary, part.info.binary_size, barrier,
                    sizeof(barrier)) == 1 &&
      part.info.binary_size >= sizeof(end) &&
      memcmp((const uint8_t *)part.binary + part.info.binary_size - sizeof(end),
             end, sizeof(end)) == 0;

   if (!valid)
      fprintf(stderr, "G17P shared-memory barrier compile oracle failed\n");

   free(part.binary);
   ralloc_free(b.shader);
   if (!valid)
      return 1;

   puts("agx3_workgroup: shared exchange, local index and barrier emitted");
   return 0;
}
