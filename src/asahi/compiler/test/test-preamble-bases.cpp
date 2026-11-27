/*
 * Copyright 2026 Joshua Warren
 * SPDX-License-Identifier: MIT
 */

#include "nir_builder.h"

extern "C" {
#include "agx_compiler.h"
}

#include <gtest/gtest.h>

/*
 * Verify that agx_nir_place_preamble_bases_low() moves 64-bit load_agx base
 * pointers out of the high uniform half (>= 256 halfwords) into low uniform
 * slots (< 256 halfwords), prioritizing loop-nested base pointers first,
 * without overflowing a full 512-halfword uniform file.
 */
TEST(PreambleBasesLow, PacksHighLoopLoadBaseBelow256WithoutOverflow)
{
   static const nir_shader_compiler_options options = {};
   glsl_type_singleton_init_or_ref();
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options,
                                                  "preamble bases low test");

   nir_function *preamble_fn = nir_function_create(b.shader, "preamble");
   preamble_fn->is_preamble = true;
   nir_function_impl *preamble = nir_function_impl_create(preamble_fn);
   b.impl->preamble = preamble_fn;

   nir_builder pb = nir_builder_at(nir_before_impl(preamble));

   const unsigned reserved = 8;
   unsigned offset = reserved;

   /* Pack 132 32-bit general preamble exports (264 halfwords -> offsets 8..272)
    * so subsequent 64-bit load bases sit above halfword 256.
    */
   nir_def *acc = nir_imm_int(&b, 0);
   for (unsigned i = 0; i < 132; ++i) {
      nir_store_preamble(&pb, nir_imm_int(&pb, i + 1), .base = (int)offset);
      nir_def *u = nir_load_preamble(&b, 1, 32, .base = (int)offset);
      acc = nir_iadd(&b, acc, u);
      offset += 2;
   }

   /* Outer (non-loop) 64-bit load base at offset 272 (aligned to 4). */
   const unsigned outer_base_off = offset;
   nir_store_preamble(&pb, nir_imm_int64(&pb, 0x100000), .base = (int)outer_base_off);
   offset += 4;

   /* Inner-loop 64-bit load base at offset 276. */
   const unsigned loop_base_off = offset;
   nir_store_preamble(&pb, nir_imm_int64(&pb, 0x200000), .base = (int)loop_base_off);
   offset += 4;

   /* Outer load */
   nir_def *outer_ptr = nir_load_preamble(&b, 1, 64, .base = (int)outer_base_off);
   nir_def *outer_val = nir_load_agx(&b, 1, 32, outer_ptr, nir_imm_int(&b, 0),
                                     .base = 0, .format = PIPE_FORMAT_R32_UINT);
   acc = nir_iadd(&b, acc, outer_val);

   /* Inner-loop load */
   nir_loop *loop = nir_push_loop(&b);
   (void)loop;
   nir_def *loop_ptr = nir_load_preamble(&b, 1, 64, .base = (int)loop_base_off);
   nir_def *loop_val = nir_load_agx(&b, 1, 32, loop_ptr, nir_imm_int(&b, 0),
                                    .base = 0, .format = PIPE_FORMAT_R32_UINT);
   acc = nir_iadd(&b, acc, loop_val);
   nir_jump(&b, nir_jump_break);
   nir_pop_loop(&b, loop);

   /* Sink acc via store_agx using outer_ptr so nothing is dead. */
   nir_store_agx(&b, acc, outer_ptr, nir_imm_int(&b, 4),
                 .base = 0, .format = PIPE_FORMAT_R32_UINT);

   nir_validate_shader(b.shader, "before agx_nir_place_preamble_bases_low");

   unsigned allocated = offset;
   ASSERT_TRUE(agx_nir_place_preamble_bases_low(b.shader, reserved, offset,
                                                &allocated));
   nir_validate_shader(b.shader, "after agx_nir_place_preamble_bases_low");

   EXPECT_LE(allocated, offset);

   nir_intrinsic_instr *outer_lp = nir_instr_as_intrinsic(nir_def_instr(outer_ptr));
   nir_intrinsic_instr *loop_lp = nir_instr_as_intrinsic(nir_def_instr(loop_ptr));
   EXPECT_LT(nir_intrinsic_base(loop_lp), 256u);
   EXPECT_LT(nir_intrinsic_base(outer_lp), 256u);
   /* Loop-nested base pointer must be placed ahead of outer base pointer. */
   EXPECT_EQ(nir_intrinsic_base(loop_lp), reserved);
   EXPECT_EQ(nir_intrinsic_base(outer_lp), reserved + 4);

   ralloc_free(b.shader);
}
