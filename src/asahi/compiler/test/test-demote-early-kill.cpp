/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "compiler/nir/nir_builder.h"
#include "agx_nir.h"
}

/* agx_nir_lower_sample_mask_deferred keeps G17P demoted lanes alive as quad
 * helpers until the end of the shader unless it proves that demoted lanes
 * never reach a quad-dependent operation; then it kills and runs the
 * depth/stencil test right after the discard. */
enum shape {
   GUARDED_NOT,
   GUARDED_ELSE,
   GUARDED_COMPLEMENT,
   UNGUARDED,
   OTHER_CONDITION,
   SAME_CONDITION,
   IF_DEMOTE_THEN_GUARDED,
   QUAD_ONLY_BEFORE,
   QUAD_IN_LOOP,
   SUBGROUP_AFTER,
   NESTED_GUARDED,
};

static nir_shader *
build(enum shape s)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
                                                  &agx_nir_options, "demote");
   b.shader->info.fs.uses_discard = true;
   b.shader->info.fs.needs_coarse_quad_helper_invocations = true;

   nir_def *xy = nir_load_frag_coord(&b);
   nir_def *x = nir_channel(&b, xy, 0), *y = nir_channel(&b, xy, 1);
   nir_def *c = nir_flt(&b, x, y);
   nir_def *d = nir_flt(&b, y, nir_imm_float(&b, 3.0));
   nir_def *ix = nir_f2i32(&b, x), *iy = nir_f2i32(&b, y);
   nir_def *out = nir_imm_float(&b, 0.0);

   if (s == QUAD_ONLY_BEFORE)
      out = nir_ddx(&b, x);

   if (s == IF_DEMOTE_THEN_GUARDED) {
      nir_push_if(&b, c);
      nir_demote(&b);
      nir_pop_if(&b, NULL);
   } else {
      nir_demote_if(&b, s == GUARDED_COMPLEMENT ? nir_ieq(&b, ix, iy) : c);
   }

   nir_def *v;
   switch (s) {
   case GUARDED_NOT:
   case IF_DEMOTE_THEN_GUARDED:
   case NESTED_GUARDED: {
      nir_if *nif = nir_push_if(&b, nir_inot(&b, c));
      if (s == NESTED_GUARDED) {
         nir_push_if(&b, d);
         v = nir_ddx(&b, y);
         nir_pop_if(&b, NULL);
         v = nir_if_phi(&b, v, nir_imm_float(&b, 1.0));
      } else {
         v = nir_ddy(&b, y);
      }
      nir_pop_if(&b, nif);
      out = nir_if_phi(&b, v, nir_imm_float(&b, 2.0));
      break;
   }
   case GUARDED_ELSE: {
      nir_if *nif = nir_push_if(&b, c);
      nir_push_else(&b, nif);
      v = nir_ddx(&b, y);
      nir_pop_if(&b, nif);
      out = nir_if_phi(&b, nir_imm_float(&b, 2.0), v);
      break;
   }
   case GUARDED_COMPLEMENT: {
      nir_if *nif = nir_push_if(&b, nir_ine(&b, iy, ix));
      v = nir_ddx(&b, y);
      nir_pop_if(&b, nif);
      out = nir_if_phi(&b, v, nir_imm_float(&b, 2.0));
      break;
   }
   case UNGUARDED:
      out = nir_ddx(&b, y);
      break;
   case OTHER_CONDITION:
   case SAME_CONDITION: {
      nir_if *nif = nir_push_if(&b, s == OTHER_CONDITION ? d : c);
      v = nir_ddx(&b, y);
      nir_pop_if(&b, nif);
      out = nir_if_phi(&b, v, nir_imm_float(&b, 2.0));
      break;
   }
   case QUAD_IN_LOOP: {
      nir_loop *loop = nir_push_loop(&b);
      v = nir_ddx(&b, y);
      nir_break_if(&b, nir_flt(&b, v, x));
      nir_pop_loop(&b, loop);
      out = x;
      break;
   }
   case SUBGROUP_AFTER:
      out = nir_u2f32(&b, nir_channel(&b, nir_ballot(&b, 1, 32, d), 0));
      break;
   case QUAD_ONLY_BEFORE:
      out = nir_fadd(&b, out, x);
      break;
   }

   nir_store_output(&b, out, nir_imm_int(&b, 0),
                    .io_semantics.location = FRAG_RESULT_DATA0,
                    .src_type = nir_type_float32);
   return b.shader;
}

/* The deferred lowering appends the test-triggering update after all other
 * work; the immediate one places it where the discard was, ahead of the
 * output. */
static bool
kill_is_early(nir_shader *s)
{
   bool mask = false, after = false;
   nir_foreach_block(block, nir_shader_get_entrypoint(s)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         /* The test-triggering coverage update; kill-only updates (dead
          * quads) have a constant zero live mask. */
         if (intr->intrinsic == nir_intrinsic_sample_mask_agx &&
             !(nir_src_is_const(intr->src[1]) &&
               nir_src_as_uint(intr->src[1]) == 0))
            mask = true;
         else if (mask && intr->intrinsic == nir_intrinsic_store_output)
            after = true;
      }
   }
   EXPECT_TRUE(mask);
   return after;
}

static bool
lower(enum shape s)
{
   nir_shader *nir = build(s);
   agx_nir_lower_discard_zs_emit(nir);
   agx_nir_lower_sample_mask_deferred(nir);
   nir_validate_shader(nir, "demote lowering");
   bool early = kill_is_early(nir);
   ralloc_free(nir);
   return early;
}

class DemoteEarlyKill : public ::testing::Test {
 protected:
   DemoteEarlyKill() { glsl_type_singleton_init_or_ref(); }
   ~DemoteEarlyKill() { glsl_type_singleton_decref(); }
};

TEST_F(DemoteEarlyKill, GuardedQuadOperationsAllowEarlyKill)
{
   EXPECT_TRUE(lower(GUARDED_NOT));
   EXPECT_TRUE(lower(GUARDED_ELSE));
   EXPECT_TRUE(lower(GUARDED_COMPLEMENT));
   EXPECT_TRUE(lower(IF_DEMOTE_THEN_GUARDED));
   EXPECT_TRUE(lower(NESTED_GUARDED));
   EXPECT_TRUE(lower(QUAD_ONLY_BEFORE));
}

TEST_F(DemoteEarlyKill, ReachableQuadOperationsKeepDeferredKill)
{
   EXPECT_FALSE(lower(UNGUARDED));
   EXPECT_FALSE(lower(OTHER_CONDITION));
   EXPECT_FALSE(lower(SAME_CONDITION));
   EXPECT_FALSE(lower(QUAD_IN_LOOP));
   EXPECT_FALSE(lower(SUBGROUP_AFTER));
}

/* A fully demoted quad is killed right after the demote; a shader with a
 * non-quad subgroup operation keeps every demoted lane until the end. */
static unsigned
count_quad_ballots(nir_shader *s)
{
   unsigned n = 0;
   nir_foreach_block(block, nir_shader_get_entrypoint(s)) {
      nir_foreach_instr(instr, block) {
         n += instr->type == nir_instr_type_intrinsic &&
              nir_instr_as_intrinsic(instr)->intrinsic ==
                 nir_intrinsic_quad_ballot_agx;
      }
   }
   return n;
}

TEST_F(DemoteEarlyKill, FullyDemotedQuadsAreKilledEarly)
{
   for (enum shape s : {UNGUARDED, OTHER_CONDITION, SUBGROUP_AFTER}) {
      nir_shader *nir = build(s);
      agx_nir_lower_discard_zs_emit(nir);
      agx_nir_lower_sample_mask_deferred(nir);
      nir_validate_shader(nir, "dead quads");
      EXPECT_EQ(count_quad_ballots(nir), s == SUBGROUP_AFTER ? 0u : 1u);
      ralloc_free(nir);
   }
}
