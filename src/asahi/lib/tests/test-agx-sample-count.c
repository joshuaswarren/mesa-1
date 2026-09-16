/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "agx_compile.h"
#include "agx_tilebuffer.h"
#include "nir_builder.h"
#include <stdio.h>
#include <stdlib.h>

#define CHECK(x) do { if (!(x)) { \
   fprintf(stderr, "%s:%u: %s\n", __FILE__, __LINE__, #x); abort(); \
} } while (0)

int
main(void)
{
   /* Exercise literal and register-sourced indices independently. */
   for (unsigned variant = 0; variant < 6; ++variant) {
      unsigned dynamic = variant / 3;
      unsigned samples = 1u << (variant % 3);
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "explicit sample count");
      CHECK(!agx_nir_needs_sample_count(b.shader));
      nir_def *sample = dynamic ? nir_load_preamble(&b, 1, 32, .base = 0)
                               : nir_imm_int(&b, 1);
      nir_def *bary = nir_load_barycentric_at_sample(
         &b, 32, sample, .interp_mode = INTERP_MODE_SMOOTH);
      nir_export_agx(&b, bary, .base = 0);
      CHECK(agx_nir_needs_sample_count(b.shader));
      CHECK(agx_nir_lower_sample_intrinsics(b.shader, false));
      CHECK(agx_nir_specialize_sample_count(b.shader, samples));
      CHECK(!agx_nir_specialize_sample_count(b.shader, samples));
      nir_opt_constant_folding(b.shader);
      nir_opt_algebraic(b.shader);
      nir_opt_dce(b.shader);
      unsigned interpolations = 0;
      nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            CHECK(intr->intrinsic != nir_intrinsic_load_samples_log2_agx);
            if (intr->intrinsic == nir_intrinsic_load_barycentric_at_sample) {
               if (dynamic && samples > 1) {
                  CHECK(intr->src[0].ssa == sample);
               } else {
                  CHECK(nir_src_is_const(intr->src[0]));
                  CHECK(nir_src_as_uint(intr->src[0]) == (samples == 1 ? 0 : 1));
               }
               interpolations++;
            }
         }
      }
      CHECK(interpolations == 1);
      CHECK(!b.shader->info.fs.uses_sample_shading);
      nir_validate_shader(b.shader, "specialized explicit sample count");
      ralloc_free(b.shader);
   }
   return 0;
}
