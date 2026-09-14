/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_compile.h"
#include "agx_nir.h"
#include "nir_builder.h"

#include <stdio.h>
#include <stdlib.h>

/*
 * A shader that shades one of several candidate records -- poly's geometry
 * rasterization shader, choosing which emitted vertex it copies -- loads every
 * candidate and selects between the results. Selecting the index instead loads
 * once, which is what keeps a wide transform feedback capture inside the AGX3B
 * register file. Count the loads that survive.
 */

static unsigned
count_loads(nir_shader *shader)
{
   unsigned n = 0;

   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_load_agx ||
                intr->intrinsic == nir_intrinsic_load_constant_agx)
               n++;
         }
      }
   }

   return n;
}

/* A byte index the address lowering keeps as a register index rather than
 * folding into the base, matching the varying offsets poly computes. */
static nir_def *
candidate_address(nir_builder *b, nir_def *base, unsigned record)
{
   nir_def *index =
      nir_imul_imm(b, nir_load_vertex_id(b), 12 * (record + 1));

   return nir_iadd(b, base, nir_u2u64(b, index));
}

static nir_shader *
build(unsigned varyings, unsigned candidates, bool distinct_bases)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "candidate record copy");

   nir_def *id = nir_load_vertex_id(&b);
   nir_def *base = nir_pack_64_2x32_split(&b, id, nir_imm_int(&b, 0));

   for (unsigned v = 0; v < varyings; ++v) {
      nir_def *value = NULL;

      for (unsigned c = 0; c < candidates; ++c) {
         /* Each candidate's record sits at its own offset, and a separate base
          * models the case the fold must decline. */
         nir_def *from = distinct_bases
                            ? nir_pack_64_2x32_split(&b, id, nir_imm_int(&b, c))
                            : base;
         nir_def *loaded = nir_load_global_constant(
            &b, 4, 32, candidate_address(&b, from, varyings * c + v),
            .align_mul = 4);

         value = value ? nir_bcsel(&b, nir_ieq_imm(&b, id, c), loaded, value)
                       : loaded;
      }

      nir_store_output(&b, value, nir_imm_int(&b, 0),
                       .io_semantics.location = VARYING_SLOT_VAR0 + v,
                       .write_mask = 0xf, .src_type = nir_type_float32);
   }

   nir_shader *shader = b.shader;
   nir_shader_gather_info(shader, nir_shader_get_entrypoint(shader));
   return shader;
}

static bool
check(const char *what, unsigned varyings, unsigned candidates,
      bool distinct_bases, unsigned expected)
{
   nir_shader *shader = build(varyings, candidates, distinct_bases);

   NIR_PASS(_, shader, agx_nir_lower_address, true);
   NIR_PASS(_, shader, nir_opt_dce);

   const unsigned before = count_loads(shader);

   NIR_PASS(_, shader, agx_nir_opt_select_load);
   NIR_PASS(_, shader, nir_opt_cse);
   NIR_PASS(_, shader, nir_opt_dce);
   nir_validate_shader(shader, "after agx_nir_opt_select_load");

   const unsigned after = count_loads(shader);
   const bool ok = before == varyings * candidates && after == expected;

   printf("%-44s %2u loads -> %2u (want %2u) %s\n", what, before, after,
          expected, ok ? "ok" : "FAIL");

   ralloc_free(shader);
   return ok;
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   bool ok = true;

   /* One load per varying survives, however many candidates there were: the
    * innermost select folds, and the one above it then sees a load on both
    * sides. */
   ok &= check("two candidates, four varyings", 4, 2, false, 4);
   ok &= check("three candidates, four varyings", 4, 3, false, 4);
   ok &= check("three candidates, sixteen varyings", 16, 3, false, 16);

   /* Nothing to fold when there is only one candidate. */
   ok &= check("one candidate, four varyings", 4, 1, false, 4);

   /* Loads from unrelated bases are not the same access, so the index alone
    * cannot choose between them. */
   ok &= check("two candidates, separate bases", 4, 2, true, 8);

   glsl_type_singleton_decref();
   return ok ? 0 : 1;
}
