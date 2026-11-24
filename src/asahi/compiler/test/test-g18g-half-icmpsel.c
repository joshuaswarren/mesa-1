/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/* mesa#739 / omarchy-mlx#662: on G18G the AGX3 packer has no grounded
 * encoding for a 16-bit icmpsel whose two comparands are half registers
 * under a signed relation (or any relation when the selects are both
 * immediates into a word result). The G18G compare-select lowering widens
 * those forms before register allocation; unfused icmp pseudo ops lower to
 * icmpsel only after RA, so they must be widened while they are still icmp.
 * These programs compile the shapes the issue's shaders reach:
 *
 *   abs16:      r[i] = abs(a[i])      -> icmpsel h,h,h,h  SGT
 *   sign16:     r[i] = sign(a[i])     -> icmpsel chains on half compares
 *   divmod_i64: i64 floor-divide/modulo fixups -> unfused icmp on halves
 */

#include "agx_compile.h"
#include "agx_compiler.h"
#include "nir_builder.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const uint8_t end_marker[] = {0x0e, 0x00, 0x00, 0x00};

static bool
compile_g18g(nir_builder *b, const char *what)
{
   b->shader->info.workgroup_size[0] = 1;
   b->shader->info.workgroup_size[1] = 1;
   b->shader->info.workgroup_size[2] = 1;
   agx_preprocess_nir(b->shader);

   struct agx_shader_key key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3C,
      .dev.emit_agx3d = true,
      .reserved_preamble = 8,
      .no_preamble = true,
   };
   struct agx_shader_part part = {0};
   const bool compiled = agx_compile_shader_nir(b->shader, &key, &part);
   const bool valid =
      compiled && part.binary && part.info.binary_size >= sizeof(end_marker) &&
      !memcmp((uint8_t *)part.binary + part.info.binary_size -
                 sizeof(end_marker),
              end_marker, sizeof(end_marker));

   if (!valid)
      fprintf(stderr, "G18G refused %s\n", what);

   free(part.binary);
   ralloc_free(b->shader);
   return valid;
}

static nir_builder
begin(const char *name, nir_def **input, nir_def **output)
{
   nir_builder b =
      nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &agx_nir_options,
                                     name);
   *input = nir_load_preamble(&b, 1, 64, .base = 0);
   *output = nir_load_preamble(&b, 1, 64, .base = 4);
   return b;
}

/* abs16's select: max(a, -a) at 16 bits emits the reg:reg compare-select
 * the issue's refusal names (icmpsel h,h,h,h, SGT). */
static bool
check_g18g_imax16(void)
{
   nir_def *input, *output;
   nir_builder b = begin("G18G imax on int16", &input, &output);
   nir_def *a = nir_load_global(&b, 1, 16, input, .align_mul = 2);
   nir_def *m = nir_imax(&b, a, nir_ineg(&b, a));
   nir_store_global(&b, m, output, .align_mul = 2);
   return compile_g18g(&b, "imax(int16, -int16)");
}

/* The unsigned twin (issue's ULT escape path) stays native and must keep
 * compiling. */
static bool
check_g18g_umin16(void)
{
   nir_def *input, *output;
   nir_builder b = begin("G18G umin on int16", &input, &output);
   nir_def *a = nir_load_global(&b, 1, 16, input, .align_mul = 2);
   nir_def *c = nir_load_global(&b, 1, 16,
                                nir_iadd_imm(&b, input, 2), .align_mul = 2);
   nir_def *m = nir_umin(&b, a, c);
   nir_store_global(&b, m, output, .align_mul = 2);
   return compile_g18g(&b, "umin(int16, int16)");
}

/* A signed 16-bit compare whose boolean feeds a second use keeps an
 * unfused icmp alive to the post-RA pseudo lowering, which spells it
 * `icmpsel h,h,#1,#0` -- refused for every relation when the result is a
 * word and for every relation but EQ/ULT when it is a half. */
static bool
check_g18g_cmp16(void)
{
   nir_def *input, *output;
   nir_builder b = begin("G18G compare int16", &input, &output);
   nir_def *a = nir_load_global(&b, 1, 16, input, .align_mul = 2);
   nir_def *c = nir_load_global(&b, 1, 16,
                                nir_iadd_imm(&b, input, 2), .align_mul = 2);
   nir_def *lt = nir_slt(&b, a, c);
   nir_def *sel0 = nir_bcsel(&b, lt, nir_imm_int16(&b, 7),
                             nir_imm_int16(&b, 3));
   nir_def *sel1 = nir_bcsel(&b, lt, nir_imm_int16(&b, 11),
                             nir_imm_int16(&b, 13));
   nir_store_global(&b, nir_iadd(&b, sel0, sel1), output, .align_mul = 2);
   return compile_g18g(&b, "shared slt(int16, int16)");
}

/* The issue's elementwise/divmod int64 refusals: nir_lower_idiv fixups and
 * 64-bit compares reach unfused half compares. */
static bool
check_g18g_i64_divmod(void)
{
   nir_def *input, *output;
   nir_builder b = begin("G18G int64 divide and remainder", &input, &output);
   nir_def *x = nir_load_global(&b, 1, 64, input, .align_mul = 8);
   nir_def *y = nir_load_global(&b, 1, 64,
                                nir_iadd_imm(&b, input, 8), .align_mul = 8);
   nir_store_global(&b, nir_idiv(&b, x, y), output, .align_mul = 8);
   nir_store_global(&b, nir_irem(&b, x, y),
                    nir_iadd_imm(&b, output, 8), .align_mul = 8);
   return compile_g18g(&b, "idiv/irem int64");
}

int
main(void)
{
   if (!check_g18g_imax16() || !check_g18g_umin16() ||
       !check_g18g_cmp16() || !check_g18g_i64_divmod())
      return 1;

   puts("g18g_half_icmpsel: 16-bit compare-selects compile for G18G");
   return 0;
}
