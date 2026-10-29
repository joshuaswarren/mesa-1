/* SPDX-License-Identifier: MIT */
#include <array>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "agx_nir.h"
}

static float
position_value(nir_scalar s, nir_def *world, const std::array<nir_def *, 3> &u,
               float x, const std::array<float, 3> &v)
{
   s = nir_scalar_chase_movs(s);
   if (s.def == world) return x;
   for (unsigned i = 0; i < u.size(); i++) if (s.def == u[i]) return v[i];
   if (nir_scalar_is_const(s)) {
      uint32_t bits = nir_scalar_as_uint(s); float f;
      std::memcpy(&f, &bits, 4); return f;
   }
   if (!nir_scalar_is_alu(s)) { ADD_FAILURE(); return 0; }
   auto src = [&](unsigned i) { return position_value(
      nir_scalar_chase_alu_src(s, i), world, u, x, v); };
   switch (nir_scalar_alu_op(s)) {
   case nir_op_fadd: return src(0) + src(1);
   case nir_op_fmul: return src(0) * src(1);
   case nir_op_ffma: return std::fma(src(0), src(1), src(2));
   case nir_op_fneg: return -src(0);
   default: ADD_FAILURE() << nir_op_infos[nir_scalar_alu_op(s)].name; return 0;
   }
}

static void
paired_position(bool keep_varying, bool protect, std::array<float, 8> *out)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "paired position order");
   nir_def *world = nir_load_input(&b, 1, 32, nir_imm_int(&b, 0),
      .base = 0, .dest_type = nir_type_float32,
      .io_semantics = {.location = VERT_ATTRIB_GENERIC0});
   std::array<nir_def *, 3> u = {
      agx_test_load_preamble(&b, 1, 32, 0),
      agx_test_load_preamble(&b, 1, 32, 2),
      agx_test_load_preamble(&b, 1, 32, 4)};
   nir_def *intermediate = nir_fadd(&b, nir_fadd(&b, world, u[0]), u[1]);
   nir_def *clip = nir_fadd(&b, intermediate, u[2]);
   auto *store = nir_store_output(&b, clip, nir_imm_int(&b, 0),
      .base = 0, .write_mask = 1, .component = 2,
      .src_type = nir_type_float32, .io_semantics = {.location = VARYING_SLOT_POS});
   if (keep_varying) nir_store_output(&b, intermediate, nir_imm_int(&b, 0),
      .base = 1, .write_mask = 1, .src_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_VAR0});
   nir_def *unrelated = nir_fadd(&b, u[0], u[1]);
   nir_store_output(&b, unrelated, nir_imm_int(&b, 0),
      .base = 2, .write_mask = 1, .src_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_VAR1});
   if (protect) {
      ASSERT_TRUE(agx_nir_preserve_position_order(b.shader));
      EXPECT_FALSE(agx_nir_preserve_position_order(b.shader));
      EXPECT_TRUE(nir_alu_instr_no_reassoc(nir_def_as_alu(clip)));
      EXPECT_TRUE(nir_alu_instr_no_transform(nir_def_as_alu(clip)));
      EXPECT_FALSE(nir_alu_instr_no_contract(nir_def_as_alu(clip)));
      EXPECT_FALSE(nir_alu_instr_no_reassoc(nir_def_as_alu(unrelated)));
   }
   nir_divergence_analysis(b.shader);
   nir_opt_reassociate_loop(b.shader, (nir_reassociate_options)(
      nir_reassociate_scalar_math | nir_reassociate_cse_heuristic |
      nir_reassociate_loop_invariant));
   const std::array<float, 3> values = {1.23e-5f, -6.7e-6f, 64.43448638916016f};
   for (unsigned i = 0; i < out->size(); i++) {
      float x = -63.93449f + float(i) * 0.01879f;
      (*out)[i] = position_value(nir_get_scalar(store->src[0].ssa, 0), world, u, x, values);
      if (protect) {
         EXPECT_EQ((*out)[i], ((x + values[0]) + values[1]) + values[2]);
      }
   }
   nir_validate_shader(b.shader, "position order");
   ralloc_free(b.shader);
}

TEST(PositionOrder, SamePositionWithLiveVaryingKeepsIdenticalRounding)
{
   std::array<float, 8> depth, color, protected_depth, protected_color;
   paired_position(false, false, &depth);
   paired_position(true, false, &color);
   paired_position(false, true, &protected_depth);
   paired_position(true, true, &protected_color);
   EXPECT_NE(depth, color);
   EXPECT_EQ(protected_depth, protected_color);
}

TEST(PositionOrder, KeepsMultiplyAddContractionAvailable)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "position contraction");
   nir_def *x = agx_test_load_preamble(&b, 1, 32, 0);
   nir_def *y = agx_test_load_preamble(&b, 1, 32, 2);
   nir_def *z = agx_test_load_preamble(&b, 1, 32, 4);
   nir_def *product = nir_fmul(&b, x, y);
   nir_def *value = nir_fadd(&b, product, z);
   auto *store = nir_store_output(&b, value, nir_imm_int(&b, 0),
      .base = 0, .write_mask = 1, .src_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_POS});
   ASSERT_TRUE(agx_nir_preserve_position_order(b.shader));
   EXPECT_FALSE(nir_alu_instr_no_contract(nir_def_as_alu(product)));
   EXPECT_FALSE(nir_alu_instr_no_contract(nir_def_as_alu(value)));
   nir_opt_algebraic_late(b.shader);
   EXPECT_EQ(nir_scalar_alu_op(nir_scalar_chase_movs(
      nir_get_scalar(store->src[0].ssa, 0))), nir_op_ffma);
   ralloc_free(b.shader);
}

TEST(PositionOrder, IncludesPhiControlDependencies)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "position control dependencies");
   nir_def *x = agx_test_load_preamble(&b, 1, 32, 0);
   nir_def *y = agx_test_load_preamble(&b, 1, 32, 2);
   nir_def *z = agx_test_load_preamble(&b, 1, 32, 4);
   nir_def *predicate_sum = nir_fadd(&b, x, y);
   nir_def *condition = nir_flt(&b, predicate_sum, z);
   nir_push_if(&b, condition);
   nir_def *then_value = nir_fadd(&b, x, z);
   nir_push_else(&b, NULL);
   nir_def *else_value = nir_fadd(&b, y, z);
   nir_pop_if(&b, NULL);
   nir_def *position = nir_if_phi(&b, then_value, else_value);
   nir_def *unrelated = nir_fadd(&b, x, y);
   nir_store_output(&b, position, nir_imm_int(&b, 0),
      .base = 0, .write_mask = 1, .src_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_POS});
   nir_store_output(&b, unrelated, nir_imm_int(&b, 0),
      .base = 1, .write_mask = 1, .src_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_VAR0});

   ASSERT_TRUE(agx_nir_preserve_position_order(b.shader));
   EXPECT_TRUE(nir_alu_instr_no_reassoc(nir_def_as_alu(predicate_sum)));
   EXPECT_TRUE(nir_alu_instr_no_reassoc(nir_def_as_alu(then_value)));
   EXPECT_TRUE(nir_alu_instr_no_reassoc(nir_def_as_alu(else_value)));
   EXPECT_FALSE(nir_alu_instr_no_reassoc(nir_def_as_alu(unrelated)));
   EXPECT_FALSE(nir_alu_instr_no_contract(nir_def_as_alu(predicate_sum)));
   EXPECT_FALSE(agx_nir_preserve_position_order(b.shader));
   nir_validate_shader(b.shader, "position phi control dependencies");
   ralloc_free(b.shader);
}
