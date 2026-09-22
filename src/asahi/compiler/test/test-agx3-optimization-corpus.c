/* SPDX-License-Identifier: MIT */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "agx_compile.h"
#include "agx_nir.h"
#include "nir_builder.h"

/* The names and expressions match probes/agx3-optimization-corpus/kernels.metal.
 * ABI setup is deliberately Mesa's own: compare arithmetic operands rather than
 * assuming Apple's argument layout or prologue is our ABI.
 */
static const char *cases[] = {
   "uniform_repeat",
   "uniform_chain",
   "uniform_swizzle",
   "uniform_dot",
   "uniform_add",
   "uniform_mul",
   "uniform_add_absneg",
   "uniform_mul_absneg",
   "uniform_fma_absneg",
   "uniform_fma_addend",
   "uniform_fma_immediate",
   "uniform_fma_uniform_imm",
   "float_fma",
   "float_clamp",
   "float_absneg",
   "float_minmax",
   "float_recip",
   "float_rsqrt",
   "float_round",
   "float_select",
   "half_repeat",
   "half_chain",
   "half_siblings",
   "half_add",
   "half_mul",
   "half_dot",
   "half_clamp",
   "half_absneg",
   "integer_mad",
   "integer_shift",
   "integer_rotate",
   "integer_extract",
   "integer_logic",
   "integer_select",
};

static nir_def *
expression(nir_builder *b, const char *name, nir_def *x, nir_def *p, unsigned c)
{
   nir_def *a = nir_channel(b, x, c), *q = nir_channel(b, p, c);
   nir_def *px = nir_channel(b, p, 0), *py = nir_channel(b, p, 1);
   nir_def *pz = nir_channel(b, p, 2), *rev = nir_channel(b, x, 3 - c);
   if (strstr(name, "repeat"))
      return nir_ffma(b, a, px, py);
   if (strstr(name, "chain"))
      return nir_ffma(b, nir_ffma(b, a, px, py), px, pz);
   if (strstr(name, "swizzle") || strstr(name, "siblings"))
      return nir_ffma(b, a, nir_channel(b, p, c % 2),
                      nir_channel(b, p, 2 + c % 2));
   if (strstr(name, "dot"))
      return nir_fdot4(b, x, p);
   if (!strcmp(name, "uniform_add") || !strcmp(name, "half_add"))
      return nir_fadd(b, a, q);
   if (!strcmp(name, "uniform_mul") || !strcmp(name, "half_mul"))
      return nir_fmul(b, a, q);
   if (!strcmp(name, "uniform_add_absneg"))
      return nir_fadd(b, nir_fneg(b, nir_fabs(b, a)), nir_fabs(b, q));
   if (!strcmp(name, "uniform_mul_absneg"))
      return nir_fmul(b, nir_fneg(b, nir_fabs(b, a)), nir_fabs(b, q));
   if (!strcmp(name, "uniform_fma_absneg"))
      return nir_ffma(b, nir_fneg(b, nir_fabs(b, a)), nir_fabs(b, q),
                      nir_fneg(b, nir_fabs(b, nir_channel(b, p, 3 - c))));
   if (!strcmp(name, "uniform_fma_addend"))
      return nir_ffma(b, a, rev, q);
   if (!strcmp(name, "uniform_fma_immediate"))
      return nir_ffma(b, a, nir_imm_float(b, 0.5), q);
   if (!strcmp(name, "uniform_fma_uniform_imm"))
      return nir_ffma(b, a, q, nir_imm_float(b, 0.5));
   if (!strcmp(name, "float_fma"))
      return nir_ffma(b, a, q, rev);
   if (strstr(name, "clamp") || strstr(name, "minmax"))
      return nir_fmin(b, nir_fmax(b, a, px), py);
   if (strstr(name, "absneg"))
      return nir_ffma(b, nir_fneg(b, nir_fabs(b, a)), q, nir_fabs(b, rev));
   if (!strcmp(name, "float_recip"))
      return nir_fdiv(b, a, q);
   if (!strcmp(name, "float_rsqrt"))
      return nir_frsq(b, nir_fadd(b, nir_fabs(b, a), q));
   if (!strcmp(name, "float_round"))
      return nir_fadd(b, nir_ffloor(b, a), nir_fceil(b, q));
   if (!strcmp(name, "float_select"))
      return nir_bcsel(b, nir_flt(b, a, q), nir_fneg(b, a), a);
   if (!strcmp(name, "integer_mad"))
      return nir_iadd(b, nir_imul(b, a, px), py);
   if (!strcmp(name, "integer_shift"))
      return nir_iadd(b, nir_ishl_imm(b, a, 3), q);
   if (!strcmp(name, "integer_rotate"))
      return nir_ior(b, nir_ishl(b, a, q), nir_ushr(b, a, nir_ineg(b, q)));
   if (!strcmp(name, "integer_extract"))
      return nir_iand_imm(b, nir_ushr(b, a, px), 255);
   if (!strcmp(name, "integer_logic"))
      return nir_ixor(b, nir_iand(b, a, px), py);
   if (!strcmp(name, "integer_select"))
      return nir_bcsel(b, nir_ult(b, a, q), q, a);
   abort();
}

static bool
compile_case(const char *name, unsigned base)
{
   const unsigned bits = !strncmp(name, "half_", 5) ? 16 : 32;
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "%s", name);
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   nir_def *input = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *output = nir_load_preamble(&b, 1, 64, .base = 4);
   nir_def *id = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
   nir_def *offset = nir_u2u64(&b, nir_imul_imm(&b, id, 4 * bits / 8));
   nir_def *x = nir_load_global(&b, 4, bits, nir_iadd(&b, input, offset),
                                .align_mul = 16);
   nir_def *p = nir_load_preamble(&b, 4, bits, .base = base);
   nir_def *result[4];
   for (unsigned c = 0; c < 4; ++c)
      result[c] = expression(&b, name, x, p, c);
   nir_store_global(&b, nir_vec(&b, result, 4), nir_iadd(&b, output, offset),
                    .align_mul = 16);
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
