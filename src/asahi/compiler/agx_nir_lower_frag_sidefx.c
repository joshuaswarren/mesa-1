/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "agx_compiler.h"

/* Fragment shaders with side effects require special handling to ensure the
 * side effects execute as intended. By default, they require late depth
 * testing, to ensure the side effects happen even for killed pixels. Trigger
 * an all-live sample test after the side effects instead of synthesizing a
 * FragDepth write. Besides avoiding needless position interpolation, this is
 * the native operation for running tests without changing depth.
 *
 * For side effects with force early testing forced, the sample mask is written
 * at the *beginning* of the shader.
 */

#define ALL_SAMPLES (0xFF)

static void
insert_late_test(nir_builder *b)
{
   nir_def *all_samples = nir_imm_intN_t(b, ALL_SAMPLES, 16);
   nir_sample_mask_agx(b, all_samples, all_samples);
}

static bool
pass(struct nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;

   /* Only lower once */
   bool *done = data;
   if (*done)
      return false;
   *done = true;

   b->cursor = nir_before_instr(&intr->instr);
   insert_late_test(b);
   return true;
}

bool
agx_nir_lower_frag_sidefx(nir_shader *s)
{
   assert(s->info.stage == MESA_SHADER_FRAGMENT);

   /* If there are no side effects, there's nothing to lower */
   if (!s->info.writes_memory)
      return false;

   /* Lower writes from helper invocations with the common pass. The hardware
    * will predicate simple writes for us, but that fails with sample shading so
    * we do that in software too.
    */
   NIR_PASS(_, s, nir_lower_helper_writes, s->info.fs.uses_sample_shading);

   bool writes_zs =
      s->info.outputs_written &
      (BITFIELD64_BIT(FRAG_RESULT_STENCIL) | BITFIELD64_BIT(FRAG_RESULT_DEPTH));

   /* If the shader wants early fragment tests, the sample mask lowering pass
    * will trigger an early test at the beginning of the shader. This lets us
    * use a Passthrough punch type, instead of Opaque which may result in the
    * shader getting skipped incorrectly and then the side effects not kicking
    * in. But this happens there to avoid it happening twice with a discard.
    */
   if (s->info.fs.early_fragment_tests)
      return false;

   /* If depth/stencil feedback is already used, we're done */
   if (writes_zs)
      return false;

   bool done = false;
   nir_shader_intrinsics_pass(s, pass, nir_metadata_control_flow, &done);

   /* If there's no render targets written, just put the write at the end */
   if (!done) {
      nir_function_impl *impl = nir_shader_get_entrypoint(s);
      nir_builder b = nir_builder_at(nir_after_impl(impl));

      insert_late_test(&b);
   }

   return true;
}
