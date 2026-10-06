/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "../dxil/d12_nir_min_lod.h"
#include "compiler/glsl_types.h"
#include <assert.h>
#include <stdio.h>

static const nir_shader_compiler_options options = {0};

static void
check(nir_texop op, bool comparison, float minimum)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
                                                  &options, "sample clamp");
   nir_def *clamp = nir_imm_float(&b, minimum);
   unsigned extras = op == nir_texop_txb ? 1 : op == nir_texop_txd ? 2 : 0;
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 2 + extras + comparison);
   tex->op = op;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->dest_type = nir_type_float32;
   tex->coord_components = 2;
   tex->is_shadow = comparison;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_texture_handle,
                                    nir_imm_int64(&b, 0x10000));
   nir_def *coord = nir_trim_vector(&b, nir_load_frag_coord(&b), 2);
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_coord, coord);
   unsigned s = 2;
   if (op == nir_texop_txb)
      tex->src[s++] = nir_tex_src_for_ssa(nir_tex_src_bias, nir_imm_float(&b, 0.25));
   if (op == nir_texop_txd) {
      tex->src[s++] = nir_tex_src_for_ssa(nir_tex_src_ddx, nir_ddx(&b, coord));
      tex->src[s++] = nir_tex_src_for_ssa(nir_tex_src_ddy, nir_ddy(&b, coord));
   }
   if (comparison)
      tex->src[s++] = nir_tex_src_for_ssa(nir_tex_src_comparator,
                                         nir_imm_float(&b, 0.5));
   assert(s == tex->num_srcs);
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);
   d12_nir_sample_min_lod(&b, tex, clamp);
   int min = nir_tex_instr_src_index(tex, nir_tex_src_min_lod);
   assert((min >= 0) == (minimum != 0));
   if (min >= 0)
      assert(tex->src[min].src.ssa == clamp);
   assert(tex->op == (minimum != 0 && op == nir_texop_tex ? nir_texop_txd : op));
   assert(tex->is_shadow == comparison);
   assert((nir_tex_instr_src_index(tex, nir_tex_src_comparator) >= 0) == comparison);
   assert((nir_tex_instr_src_index(tex, nir_tex_src_bias) >= 0) == (op == nir_texop_txb));
   if (minimum != 0 && op == nir_texop_tex) {
      assert(nir_tex_instr_src_index(tex, nir_tex_src_ddx) >= 0);
      assert(nir_tex_instr_src_index(tex, nir_tex_src_ddy) >= 0);
   }
   nir_validate_shader(b.shader, "DXIL minimum LOD lowering");
   ralloc_free(b.shader);
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   const nir_texop ops[] = {nir_texop_tex, nir_texop_txb, nir_texop_txd};
   for (unsigned i = 0; i < 3; i++) {
      for (unsigned cmp = 0; cmp < 2; cmp++) {
         check(ops[i], cmp, 1.25f);
         check(ops[i], cmp, 0.0f);
      }
   }
   glsl_type_singleton_decref();
   puts("d12_nir_min_lod_tests: PASS");
   return 0;
}
