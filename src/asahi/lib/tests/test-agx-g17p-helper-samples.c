/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "agx_compile.h"
#include "agx_nir.h"
#include "nir_builder.h"
#include <stdio.h>
#include <stdlib.h>

#define CHECK(x) do { if (!(x)) { \
   fprintf(stderr, "%s:%u: %s\n", __FILE__, __LINE__, #x); abort(); \
} } while (0)

static nir_tex_instr *
texture(nir_builder *b, nir_def *coord)
{
   nir_tex_instr *tex = nir_tex_instr_create(b->shader, 1);
   tex->op = nir_texop_tex;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->dest_type = nir_type_uint32;
   tex->coord_components = 2;
   tex->src[0].src_type = nir_tex_src_coord;
   tex->src[0].src = nir_src_for_ssa(coord);
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(b, &tex->instr);
   return tex;
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   /* Integer sampling followed by a float conversion is the CTS shape.
    * Cover every derivative opcode, long chains, and SSA merges. */
   const nir_intrinsic_op derivatives[] = {
      nir_intrinsic_ddx, nir_intrinsic_ddy,
      nir_intrinsic_ddx_fine, nir_intrinsic_ddy_fine,
      nir_intrinsic_ddx_coarse, nir_intrinsic_ddy_coarse,
   };
   for (unsigned d = 0; d < ARRAY_SIZE(derivatives); ++d) {
      for (unsigned shape = 0; shape < 6; ++shape) {
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_FRAGMENT, &agx_nir_options, "helper texture derivative");
         nir_def *coord = nir_channels(&b, nir_load_frag_coord(&b), 3);
         nir_tex_instr *tex = texture(&b, coord);
         nir_def *value = nir_u2f32(&b, &tex->def);
         if (shape == 1) {
            for (unsigned i = 0; i < 32; ++i)
               value = nir_fadd_imm(&b, value, 1.0);
         } else if (shape == 2) {
            nir_push_if(&b, nir_load_front_face(&b, 1));
            nir_def *a = nir_fadd_imm(&b, value, 1.0);
            nir_push_else(&b, NULL);
            nir_def *c = nir_fmul_imm(&b, value, 2.0);
            nir_pop_if(&b, NULL);
            value = nir_if_phi(&b, a, c);
         } else if (shape >= 4) {
            nir_variable *acc = nir_local_variable_create(
               b.impl, glsl_vec4_type(), "loop sampled value");
            nir_store_var(&b, acc, value, 0xf);
            nir_push_loop(&b);
            nir_store_var(&b, acc, nir_fadd_imm(&b, nir_load_var(&b, acc), 1), 0xf);
            nir_push_if(&b, nir_load_front_face(&b, 1));
            nir_jump(&b, nir_jump_break);
            nir_pop_if(&b, NULL);
            nir_pop_loop(&b, NULL);
            value = nir_load_var(&b, acc);
         }
         /* A derivative of coordinates alone does not require sampled
          * results in helper lanes. This is the ordinary game path. */
         nir_def *result = nir_build_deriv(
            &b, (shape == 3 || shape == 5) ? coord : value, derivatives[d]);
         nir_export_agx(&b, result, .base = 0);
         nir_lower_vars_to_ssa(b.shader);
         bool progress = agx_nir_lower_g17p_helper_samples(b.shader);
         CHECK(progress == (shape != 3 && shape != 5));
         CHECK(tex->op == ((shape == 3 || shape == 5) ? nir_texop_tex : nir_texop_txd));
         if (progress) {
            CHECK(nir_tex_instr_src_index(tex, nir_tex_src_ddx) >= 0);
            CHECK(nir_tex_instr_src_index(tex, nir_tex_src_ddy) >= 0);
            CHECK(tex->src[0].src.ssa == coord);
         }
         CHECK(!agx_nir_lower_g17p_helper_samples(b.shader));
         nir_validate_shader(b.shader, "helper samples");
         ralloc_free(b.shader);
      }
   }
   glsl_type_singleton_decref();
   return 0;
}
