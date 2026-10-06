/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_nir_lower_image_view_min_lod.h"
#include "agx_nir.h"
#include "compiler/nir/nir_builder.h"

#include <stdio.h>

static bool
check(nir_texop op, enum glsl_sampler_dim dim, bool array,
      nir_alu_type type, unsigned base, float minimum, int level)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, NULL, "image-view integer minimum LOD");
   const bool fetch = op == nir_texop_txf;
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 1 + fetch + (op == nir_texop_txf_ms));
   tex->op = op;
   tex->sampler_dim = dim;
   tex->is_array = array;
   tex->dest_type = type;
   tex->coord_components = (dim == GLSL_SAMPLER_DIM_3D ? 3 : 2) + array;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord,
                                   nir_imm_zero(&b, tex->coord_components, 32));
   if (fetch)
      tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_lod, nir_imm_int(&b, level));
   if (op == nir_texop_txf_ms)
      tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_ms_index, nir_imm_int(&b, 0));
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);
   nir_intrinsic_instr *store = nir_store_scratch(
      &b, &tex->def, nir_imm_int(&b, 0), .write_mask = 0xf, .align_mul = 16);

   bool ok = hk_nir_lower_image_view_min_lod(b.shader);
   ok &= nir_tex_instr_src_index(tex, nir_tex_src_min_lod) < 0;
   /* Lowering must keep the requested integer level unchanged. */
   if (fetch)
      ok &= nir_src_as_int(tex->src[1].src) == level;

   unsigned queries = 0;
   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_tex)
               continue;
            nir_tex_instr *t = nir_instr_as_tex(instr);
            b.cursor = nir_before_instr(instr);
            if (t->op == nir_texop_image_min_lod_agx) {
               queries++;
               ok &= t->dest_type == nir_type_uint16;
               /* Match the descriptor's floor and base-level rebasing. */
               unsigned threshold = minimum > base ? (unsigned)(minimum - base) : 0;
               nir_def_replace(&t->def, nir_imm_intN_t(&b, threshold, 16));
            } else {
               nir_def_replace(&t->def, nir_imm_ivec4(&b, 7, 8, 9, 10));
            }
         }
      }
   }
   ok &= queries == 1;
   nir_validate_shader(b.shader, "integer view min LOD regression");
   while (nir_opt_constant_folding(b.shader)) {}
   const unsigned absolute_level = base + (fetch ? level : 0);
   const bool oob = absolute_level < (unsigned)minimum;
   for (unsigned c = 0; c < 4; ++c)
      ok &= nir_src_is_const(store->src[0]) &&
            nir_src_comp_as_uint(store->src[0], c) == (oob ? 0 : 7 + c);
   if (!ok) {
      fprintf(stderr, "FAIL op=%u dim=%u array=%u type=%u base=%u min=%g lod=%d\n",
              op, dim, array, type, base, minimum, level);
      nir_print_shader(b.shader, stderr);
   }
   ralloc_free(b.shader);
   return ok;
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   const float minima[] = {0.0, 0.5, 1.0, 1.5, 2.0, 2.75, 4.0};
   const nir_alu_type types[] = {nir_type_float32, nir_type_int32, nir_type_uint32};
   unsigned count = 0;
   for (unsigned base = 0; base < 3; ++base) {
      for (unsigned m = 0; m < ARRAY_SIZE(minima); ++m) {
         for (unsigned t = 0; t < ARRAY_SIZE(types); ++t) {
            for (unsigned shape = 0; shape < 5; ++shape) {
               const nir_texop op = shape < 3 ? nir_texop_txf
                                    : shape == 3 ? nir_texop_tg4 : nir_texop_txf_ms;
               const enum glsl_sampler_dim dim = shape == 2 ? GLSL_SAMPLER_DIM_3D
                                                  : shape == 4 ? GLSL_SAMPLER_DIM_MS
                                                               : GLSL_SAMPLER_DIM_2D;
               for (int level = 0; level < (op == nir_texop_txf ? 4 : 1); ++level) {
                  if (!check(op, dim, shape == 1, types[t], base, minima[m], level))
                     return 1;
                  count++;
               }
            }
         }
      }
   }
   glsl_type_singleton_decref();
   printf("PASS: %u integer image-view min LOD cases\n", count);
   return 0;
}
