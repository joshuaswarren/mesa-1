/* SPDX-License-Identifier: MIT */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "agx_compile.h"
#include "agx_nir.h"
#include "nir_builder.h"

/* Paired with this directory's kernels.metal. Use Mesa's own pointer and
 * uniform ABI. Full native text sizes also include Apple's constant program;
 * compare arithmetic shapes, not the unrelated entrypoint overhead.
 */
static const char *cases[] = {
   "u32_add",         "u32_sub",          "u32_sub_reverse", "u32_mul",
   "u32_mad_uniform", "u32_mad_register", "u32_and",         "u32_or",
   "u32_xor",         "u32_nand",         "u32_nor",         "u32_xnor",
   "u32_andnot",      "u32_notand",       "u32_shift_left",  "u32_shift_right",
   "u32_shift_add",   "u32_extract",      "u32_min",         "u32_max",
   "u32_select",      "u32_rotate",       "i32_add",         "i32_sub",
   "i32_sub_reverse", "i32_mul",          "i32_mad_uniform", "i32_mad_register",
   "i32_and",         "i32_or",           "i32_xor",         "i32_nand",
   "i32_nor",         "i32_xnor",         "i32_andnot",      "i32_notand",
   "i32_shift_left",  "i32_shift_right",  "i32_shift_add",   "i32_extract",
   "i32_min",         "i32_max",          "i32_select",      "i32_rotate",
   "u16_add",         "u16_sub",          "u16_sub_reverse", "u16_mul",
   "u16_mad_uniform", "u16_mad_register", "u16_and",         "u16_or",
   "u16_xor",         "u16_nand",         "u16_nor",         "u16_xnor",
   "u16_andnot",      "u16_notand",       "u16_shift_left",  "u16_shift_right",
   "u16_shift_add",   "u16_extract",      "u16_min",         "u16_max",
   "u16_select",      "u16_rotate",       "i16_add",         "i16_sub",
   "i16_sub_reverse", "i16_mul",          "i16_mad_uniform", "i16_mad_register",
   "i16_and",         "i16_or",           "i16_xor",         "i16_nand",
   "i16_nor",         "i16_xnor",         "i16_andnot",      "i16_notand",
   "i16_shift_left",  "i16_shift_right",  "i16_shift_add",   "i16_extract",
   "i16_min",         "i16_max",          "i16_select",      "i16_rotate",
};

static nir_def *
expression(nir_builder *b, const char *name, nir_def *x, nir_def *p, unsigned c)
{
   const char *op = strchr(name, '_') + 1;
   const bool signed_value = name[0] == 'i';
   nir_def *a = nir_channel(b, x, c), *q = nir_channel(b, p, c);
   nir_def *px = nir_channel(b, p, 0), *py = nir_channel(b, p, 1);
   nir_def *rev = nir_channel(b, x, 3 - c);
   nir_def *shift = nir_u2u32(b, q);
   if (!strcmp(op, "add"))
      return nir_iadd(b, a, q);
   if (!strcmp(op, "sub"))
      return nir_isub(b, a, q);
   if (!strcmp(op, "sub_reverse"))
      return nir_isub(b, q, a);
   if (!strcmp(op, "mul"))
      return nir_imul(b, a, q);
   if (!strcmp(op, "mad_uniform"))
      return nir_iadd(b, nir_imul(b, a, px), py);
   if (!strcmp(op, "mad_register"))
      return nir_iadd(b, nir_imul(b, a, q), rev);
   if (!strcmp(op, "and"))
      return nir_iand(b, a, q);
   if (!strcmp(op, "or"))
      return nir_ior(b, a, q);
   if (!strcmp(op, "xor"))
      return nir_ixor(b, a, q);
   if (!strcmp(op, "nand"))
      return nir_inot(b, nir_iand(b, a, q));
   if (!strcmp(op, "nor"))
      return nir_inot(b, nir_ior(b, a, q));
   if (!strcmp(op, "xnor"))
      return nir_inot(b, nir_ixor(b, a, q));
   if (!strcmp(op, "andnot"))
      return nir_iand(b, a, nir_inot(b, q));
   if (!strcmp(op, "notand"))
      return nir_iand(b, nir_inot(b, a), q);
   if (!strcmp(op, "shift_left"))
      return nir_ishl(b, a, shift);
   if (!strcmp(op, "shift_right"))
      return signed_value ? nir_ishr(b, a, shift) : nir_ushr(b, a, shift);
   if (!strcmp(op, "shift_add"))
      return nir_iadd(b, nir_ishl_imm(b, a, 3), q);
   if (!strcmp(op, "extract")) {
      nir_def *count = nir_u2u32(b, px);
      nir_def *v = signed_value ? nir_ishr(b, a, count) : nir_ushr(b, a, count);
      return nir_iand_imm(b, v, 255);
   }
   if (!strcmp(op, "min"))
      return signed_value ? nir_imin(b, a, q) : nir_umin(b, a, q);
   if (!strcmp(op, "max"))
      return signed_value ? nir_imax(b, a, q) : nir_umax(b, a, q);
   if (!strcmp(op, "select")) {
      nir_def *cond = signed_value ? nir_ilt(b, a, q) : nir_ult(b, a, q);
      return nir_bcsel(b, cond, q, a);
   }
   if (!strcmp(op, "rotate"))
      return nir_ior(b, nir_ishl(b, a, shift),
                     nir_ushr(b, a, nir_ineg(b, shift)));
   abort();
}

static bool
compile_case(const char *name, unsigned base)
{
   const unsigned bits = strstr(name, "16_") ? 16 : 32;
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "%s", name);
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   nir_def *input = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *output = nir_load_preamble(&b, 1, 64, .base = 4);
   nir_def *id = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
   nir_def *offset = nir_u2u64(&b, nir_imul_imm(&b, id, 4 * bits / 8));
   nir_def *x = nir_load_global(&b, 4, bits, nir_iadd(&b, input, offset),
                                .align_mul = 4 * bits / 8);
   nir_def *p = nir_load_preamble(&b, 4, bits, .base = base);
   nir_def *result[4];
   for (unsigned c = 0; c < 4; ++c)
      result[c] = expression(&b, name, x, p, c);
   nir_store_global(&b, nir_vec(&b, result, 4), nir_iadd(&b, output, offset),
                    .align_mul = 4 * bits / 8);
   agx_preprocess_nir_g17p(b.shader);
   struct agx_shader_key key = {.dev.encoding = AGX_ISA_ENCODING_AGX3B,
                                .no_preamble = true};
   struct agx_shader_part part = {0};
   bool ok = agx_compile_shader_nir(b.shader, &key, &part);
   if (!ok)
      fprintf(stderr, "corpus refused: %s base=%u\n", name, base);
   else if (getenv("AGX_CORPUS_REPORT")) {
      printf("CORPUS,%s,%u,%u,%u,%u,%u\n", name, base, part.info.main_size,
             part.info.stats.instrs, part.info.nr_gprs, part.info.scratch_size);
   }
   const char *dir = getenv("AGX_CORPUS_OUTPUT");
   if (ok && dir) {
      char path[1024];
      snprintf(path, sizeof(path), "%s/%s-%u.bin", dir, name, base);
      FILE *f = fopen(path, "wb");
      if (!f)
         ok = false;
      else {
         ok &= fwrite((char *)part.binary + part.info.main_offset, 1,
                      part.info.main_size, f) == part.info.main_size;
         ok &= fclose(f) == 0;
      }
   }
   free(part.binary);
   ralloc_free(b.shader);
   return ok;
}

int
main(void)
{
   bool ok = true;
   const unsigned bases[] = {12, 248, 256, 504};
   for (unsigned i = 0; i < ARRAY_SIZE(cases); ++i)
      for (unsigned j = 0; j < ARRAY_SIZE(bases); ++j)
         ok &= compile_case(cases[i], bases[j]);
   return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
