/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "compiler/nir/nir_builder.h"
#include "agx_nir.h"
}

static nir_shader *
make_shader(unsigned hazard = 0)
{
   nir_builder n = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "live paired preamble halves");
   nir_def *lo = nir_load_preamble(&n, 1, 16, .base = 28);
   nir_def *hi = nir_load_preamble(&n, 1, 16, .base = 29);
   nir_store_uvs_agx(&n, nir_pack_32_2x16_split(&n, lo, hi),
                     nir_imm_int(&n, 0));
   nir_builder p = nir_builder_create(nir_shader_get_preamble(n.shader));
   p.cursor = nir_after_impl(p.impl);
   nir_def *count = nir_load_preamble(&p, 1, 32, .base = 0);
   nir_def *cond = nir_ilt(&p, nir_imm_int(&p, 0), count);
   nir_store_preamble(&p, cond, .base = 28);
   if (hazard != 6)
      nir_store_preamble(&p, nir_imm_intN_t(&p, 0x55, 16), .base = 29);
   if (hazard != 7)
      nir_push_if(&p, cond);
   if (hazard != 6) {
      nir_def *v = nir_ilt(&p, nir_imm_int(&p, 1), count);
      nir_store_preamble(&p, v, .base = 29);
      if (hazard != 9)
         nir_store_preamble(&p, nir_imm_intN_t(&p, 0x1234, 16), .base = 29);
   }
   if (hazard != 7)
      nir_pop_if(&p, NULL);
   if (hazard == 1)
      nir_load_preamble(&p, 1, 16, .base = 29);
   if (hazard == 2)
      nir_bindless_image_agx(&p, nir_imm_int64(&p, 0), .desc_set = 28);
   if (hazard == 3)
      nir_store_preamble(&p, nir_imm_int(&p, 0xbeef), .base = 28);
   if (hazard == 4)
      nir_load_preamble(&p, 4, 64, .base = 511);
   if (hazard == 8)
      nir_jump(&p, nir_jump_return);
   if (hazard == 10) {
      nir_variable *counter =
         nir_local_variable_create(p.impl, glsl_uint_type(), "counter");
      nir_store_var(&p, counter, nir_imm_int(&p, 0), 1);
      nir_push_loop(&p);
      nir_def *i = nir_load_var(&p, counter);
      nir_push_if(&p, nir_uge(&p, i, count));
      nir_jump(&p, nir_jump_break);
      nir_pop_if(&p, NULL);
      nir_store_preamble(&p, nir_u2u16(&p, i), .base = 28);
      nir_store_preamble(&p, nir_u2u16(&p, nir_iadd_imm(&p, i, 16)),
                         .base = 29);
      nir_store_var(&p, counter, nir_iadd_imm(&p, i, 1), 1);
      nir_pop_loop(&p, NULL);
      nir_lower_vars_to_ssa(n.shader);
   }
   return n.shader;
}
static unsigned
stores(nir_shader *s, unsigned bits)
{
   unsigned n = 0;
   nir_foreach_block(b, nir_shader_get_preamble(s))
      nir_foreach_instr(i, b)
         if (i->type == nir_instr_type_intrinsic) {
            auto *x = nir_instr_as_intrinsic(i);
            n += x->intrinsic == nir_intrinsic_store_preamble &&
                 x->src[0].ssa->bit_size == bits;
         }
   return n;
}
static bool
apply(nir_shader *s, unsigned reserved = 20, unsigned allocated = 32)
{
   return agx_nir_coalesce_preamble_half_exports(s, reserved, allocated);
}
TEST(LiveHalfPreamble, ReconcilesConditionalLastWriteWithoutChangingInputs)
{
   for (unsigned kind : {0u, 9u, 10u}) {
      nir_shader *s = make_shader(kind);
      ASSERT_TRUE(apply(s));
      EXPECT_EQ(stores(s, 32), 1u);
      EXPECT_EQ(stores(s, 16), 0u);
      EXPECT_EQ(stores(s, 1), 0u);
      nir_validate_shader(s, "paired preamble exports");
      for (unsigned input : {0u, 1u, 2u, 3u}) {
         nir_shader *c = nir_shader_clone(NULL, s);
         nir_foreach_block(b, nir_shader_get_preamble(c))
            nir_foreach_instr_safe(i, b) {
               if (i->type != nir_instr_type_intrinsic)
                  continue;
               auto *x = nir_instr_as_intrinsic(i);
               if (x->intrinsic != nir_intrinsic_load_preamble ||
                   nir_intrinsic_base(x) != 0)
                  continue;
               nir_builder n = nir_builder_at(nir_before_instr(i));
               nir_def_replace(&x->def, nir_imm_int(&n, input));
            }
         bool progress;
         do {
            progress = nir_opt_constant_folding(c);
            progress |= nir_opt_copy_prop(c);
            progress |= nir_opt_dead_cf(c);
            progress |= nir_opt_dce(c);
            progress |= nir_opt_remove_phis(c);
            progress |= nir_opt_loop_unroll(c);
         } while (progress);
         nir_validate_shader(c, "specialized paired exports");
         unsigned count = 0;
         nir_foreach_block(b, nir_shader_get_preamble(c))
            nir_foreach_instr(i, b) {
               if (i->type != nir_instr_type_intrinsic)
                  continue;
               auto *x = nir_instr_as_intrinsic(i);
               if (x->intrinsic != nir_intrinsic_store_preamble)
                  continue;
               ASSERT_TRUE(nir_src_is_const(x->src[0]));
               EXPECT_EQ(nir_intrinsic_base(x), 28u);
               unsigned expected =
                  input ? (kind == 9 ? ((input > 1 ? 1u : 0u) << 16) | 1u
                                     : 0x12340001u)
                        : 0x00550000u;
               if (kind == 10 && input)
                  expected = ((input - 1 + 16) << 16) | (input - 1);
               EXPECT_EQ(nir_src_as_uint(x->src[0]), expected);
               count++;
            }
         EXPECT_EQ(count, 1u);
         ralloc_free(c);
      }
      ralloc_free(s);
   }
}
TEST(LiveHalfPreamble, LeavesUnprovenPairsAndStraightLineAlone)
{
   for (unsigned hazard = 1; hazard <= 8; hazard++) {
      SCOPED_TRACE(hazard);
      nir_shader *s = make_shader(hazard);
      unsigned before1 = stores(s, 1), before16 = stores(s, 16),
               before32 = stores(s, 32);
      EXPECT_FALSE(apply(s, hazard == 5 ? 29 : 20));
      EXPECT_EQ(stores(s, 1), before1);
      EXPECT_EQ(stores(s, 16), before16);
      EXPECT_EQ(stores(s, 32), before32);
      ralloc_free(s);
   }
}

/* The failing CTS preamble always writes the low boolean but writes the high
 * boolean only when low is true.  Main reads both and masks high with low.
 * Exercise arbitrary legal choices for the unwritten high half, rather than
 * incorrectly requiring it to be initialized or zero. */
static nir_shader *
make_masked_shader(bool reverse = false)
{
   nir_builder n = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "conditionally unwritten half");
   nir_def *lo = nir_load_preamble(&n, 1, 1, .base = 28);
   nir_def *hi = nir_load_preamble(&n, 1, 1, .base = 29);
   nir_def *masked =
      nir_bcsel(&n, reverse ? lo : hi, reverse ? hi : lo, nir_imm_false(&n));
   nir_store_uvs_agx(&n, nir_b2i32(&n, masked), nir_imm_int(&n, 0));
   nir_builder p = nir_builder_create(nir_shader_get_preamble(n.shader));
   p.cursor = nir_after_impl(p.impl);
   nir_def *count = nir_load_preamble(&p, 1, 32, .base = 0);
   nir_def *cond = nir_ilt(&p, nir_imm_int(&p, 0), count);
   nir_store_preamble(&p, cond, .base = reverse ? 29 : 28);
   nir_push_if(&p, cond);
   nir_store_preamble(&p, nir_ilt(&p, nir_imm_int(&p, 1), count),
                      .base = reverse ? 28 : 29);
   nir_pop_if(&p, NULL);
   return n.shader;
}

static void
fold_shader(nir_shader *s)
{
   bool progress;
   do {
      progress = nir_opt_constant_folding(s);
      progress |= nir_opt_copy_prop(s);
      progress |= nir_opt_dead_cf(s);
      progress |= nir_opt_dce(s);
      progress |= nir_opt_remove_phis(s);
      progress |= nir_opt_undef(s);
   } while (progress);
}

TEST(LiveHalfPreamble,
     PreservesDefinedHalfAndMaskedConsumerWithUnwrittenNeighbor)
{
   for (bool reverse : {false, true}) {
      nir_shader *s = make_masked_shader(reverse);
      ASSERT_TRUE(apply(s));
      nir_validate_shader(s, "conditionally unwritten half after coalescing");
      EXPECT_EQ(stores(s, 32), 1u);
      EXPECT_EQ(stores(s, 1), 0u);
      for (unsigned input : {0u, 1u, 2u, 3u}) {
         for (unsigned arbitrary : {0u, 1u, 0x1234u, 0xffffu}) {
            SCOPED_TRACE(input);
            SCOPED_TRACE(arbitrary);
            nir_shader *c = nir_shader_clone(NULL, s);
            nir_foreach_block(block, nir_shader_get_preamble(c)) {
               nir_foreach_instr_safe(instr, block) {
                  nir_builder b = nir_builder_at(nir_before_instr(instr));
                  if (instr->type == nir_instr_type_undef) {
                     auto *u = nir_instr_as_undef(instr);
                     ASSERT_EQ(u->def.bit_size, 16u);
                     nir_def_replace(&u->def,
                                     nir_imm_intN_t(&b, arbitrary, 16));
                  } else if (instr->type == nir_instr_type_intrinsic) {
                     auto *intr = nir_instr_as_intrinsic(instr);
                     if (intr->intrinsic == nir_intrinsic_load_preamble &&
                         nir_intrinsic_base(intr) == 0)
                        nir_def_replace(&intr->def, nir_imm_int(&b, input));
                  }
               }
            }
            fold_shader(c);
            unsigned value = 0, exports = 0;
            nir_foreach_block(block, nir_shader_get_preamble(c)) {
               nir_foreach_instr(instr, block) {
                  if (instr->type != nir_instr_type_intrinsic)
                     continue;
                  auto *intr = nir_instr_as_intrinsic(instr);
                  if (intr->intrinsic != nir_intrinsic_store_preamble)
                     continue;
                  ASSERT_TRUE(nir_src_is_const(intr->src[0]));
                  value = nir_src_as_uint(intr->src[0]);
                  exports++;
               }
            }
            ASSERT_EQ(exports, 1u);
            unsigned unconditional = reverse ? value >> 16 : value & 0xffff;
            unsigned conditional = reverse ? value & 0xffff : value >> 16;
            EXPECT_EQ(unconditional, input > 0 ? 1u : 0u);
            if (input > 0)
               EXPECT_EQ(conditional, input > 1 ? 1u : 0u);
            else
               EXPECT_EQ(conditional, arbitrary);
            nir_foreach_block(block, nir_shader_get_entrypoint(c)) {
               nir_foreach_instr_safe(instr, block) {
                  if (instr->type != nir_instr_type_intrinsic)
                     continue;
                  auto *intr = nir_instr_as_intrinsic(instr);
                  if (intr->intrinsic != nir_intrinsic_load_preamble)
                     continue;
                  unsigned shift = 16 * (nir_intrinsic_base(intr) - 28);
                  nir_builder b = nir_builder_at(nir_before_instr(instr));
                  nir_def_replace(
                     &intr->def,
                     nir_imm_bool(&b, ((value >> shift) & 0xffff) != 0));
               }
            }
            fold_shader(c);
            unsigned outputs = 0;
            nir_foreach_block(block, nir_shader_get_entrypoint(c)) {
               nir_foreach_instr(instr, block) {
                  if (instr->type != nir_instr_type_intrinsic)
                     continue;
                  auto *intr = nir_instr_as_intrinsic(instr);
                  if (intr->intrinsic != nir_intrinsic_store_uvs_agx)
                     continue;
                  ASSERT_TRUE(nir_src_is_const(intr->src[0]));
                  EXPECT_EQ(nir_src_as_uint(intr->src[0]), input > 1 ? 1u : 0u);
                  outputs++;
               }
            }
            EXPECT_EQ(outputs, 1u);
            nir_validate_shader(c, "specialized masked consumer");
            ralloc_free(c);
         }
      }
      ralloc_free(s);
   }
}

TEST(LiveHalfPreamble, DoesNotInventValuesOutsideCompilerAllocatedSlots)
{
   for (unsigned end : {0u, 20u, 28u, 29u, 513u}) {
      nir_shader *s = make_masked_shader();
      EXPECT_FALSE(apply(s, 20, end));
      EXPECT_EQ(stores(s, 1), 2u);
      EXPECT_EQ(stores(s, 32), 0u);
      ralloc_free(s);
   }
}
