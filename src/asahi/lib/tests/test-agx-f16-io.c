/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_compile.h"
#include "agx_uvs.h"
#include "nir_builder.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Exact G17P binary after the f16 logical output has become an f32 UVS
 * export. The fifth UVSW is the user varying; the first four are position. */
static const uint8_t g17p_f16_output_vs[] = {
   0x0c,0x80,0x02,0x3e,0x00,0x00,0x00,0x0c,0x1b,0x00,0x00,0x00,
   0x2c,0x80,0x02,0xbe,0x00,0x00,0x00,0x0c,0x3c,0xdd,0x10,0x06,
   0xa7,0x17,0x54,0x06,0x02,0x0c,0xac,0x20,0x0b,0x06,0x21,0x04,
   0x1b,0x05,0x27,0x00,0x42,0x00,0x00,0x04,0x00,0x00,0x2b,0x03,
   0x27,0x00,0x42,0x00,0x00,0x08,0x00,0x00,0x3b,0x01,0x27,0x00,
   0x42,0x00,0x00,0x0c,0x00,0x00,0x4b,0x07,0x27,0x00,0x42,0x00,
   0x00,0x10,0x00,0x00,0x57,0x16,0x54,0x00,0x00,0x40,0x48,0x00,
   0x57,0x26,0x54,0x02,0x20,0x40,0x49,0x00,0x57,0x46,0x54,0x04,
   0x40,0x40,0x4a,0x00,0x57,0x86,0x54,0x06,0x60,0x40,0x4b,0x00,
   0x57,0x06,0x55,0x08,0x80,0x40,0x4c,0x00,0x0e,0x00,0x00,0x00,
};

static nir_shader *
build_f16_output(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "G17P f16 Vulkan output");

   nir_def *id = nir_u2f32(&b, nir_load_vertex_id(&b));

   nir_store_output(&b,
                    nir_vec4(&b, id, nir_imm_float(&b, -1.0f),
                             nir_imm_float(&b, 0.0f),
                             nir_imm_float(&b, 1.0f)),
                    nir_imm_int(&b, 0), .write_mask = 0xf,
                    .src_type = nir_type_float32,
                    .io_semantics.location = VARYING_SLOT_POS,
                    .io_semantics.num_slots = 1);

   nir_store_output(&b, nir_f2f16(&b, id),
                    nir_imm_int(&b, 0), .write_mask = 0x1,
                    .src_type = nir_type_float16,
                    .io_semantics.location = VARYING_SLOT_VAR0,
                    .io_semantics.num_slots = 1);

   b.shader->info.outputs_written =
      VARYING_BIT_POS | BITFIELD64_BIT(VARYING_SLOT_VAR0);
   return b.shader;
}

static bool
has_type_correct_varying_store(nir_shader *nir)
{
   unsigned stores = 0;
   unsigned converted = 0;

   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_store_uvs_agx)
               continue;

            ++stores;
            if (intr->src[0].ssa->bit_size != 32)
               return false;

            converted += nir_def_is_alu(intr->src[0].ssa) &&
                         nir_def_as_alu(intr->src[0].ssa)->op == nir_op_f2f32;
         }
      }
   }

   return stores == 5 && converted == 1;
}

static unsigned
count_native_uvsw(const struct agx_shader_part *part)
{
   const uint8_t *binary = part->binary;
   unsigned count = 0;

   for (unsigned i = 0; binary && i + 8 <= part->info.binary_size; ++i) {
      if (binary[i] == 0x57 && (binary[i + 2] & 0xfc) == 0x54) {
         ++count;
         i += 7;
      }
   }

   return count;
}

int
main(void)
{
   nir_shader *nir = build_f16_output();
   struct agx_unlinked_uvs_layout layout = {0};

   if (!agx_nir_lower_uvs(nir, &layout) || layout.size != 5 ||
       layout.user_size != 1 || !has_type_correct_varying_store(nir)) {
      fprintf(stderr, "f16 output was not widened to f32 before ST_VARY\n");
      ralloc_free(nir);
      return 1;
   }

   agx_preprocess_nir(nir);
   struct agx_shader_key key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .no_preamble = true,
   };
   struct agx_shader_part part = {0};
   const bool compiled = agx_compile_shader_nir(nir, &key, &part);
   const bool native_ok =
      compiled && part.binary &&
      part.info.binary_size == sizeof(g17p_f16_output_vs) &&
      memcmp(part.binary, g17p_f16_output_vs,
             sizeof(g17p_f16_output_vs)) == 0 &&
      count_native_uvsw(&part) == 5;

   if (!native_ok) {
      fprintf(stderr,
              "type-correct f16 output native mismatch: compiled=%u size=%u "
              "uvsw=%u\n",
              compiled, part.info.binary_size, count_native_uvsw(&part));
      for (unsigned i = 0; i < part.info.binary_size; ++i)
         fprintf(stderr, "%02x", ((const uint8_t *)part.binary)[i]);
      fputc('\n', stderr);
   }

   free(part.binary);
   ralloc_free(nir);
   return native_ok ? 0 : 1;
}
