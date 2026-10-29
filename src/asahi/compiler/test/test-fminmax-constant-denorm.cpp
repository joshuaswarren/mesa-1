/* Copyright 2026 Aurora Silicon; SPDX-License-Identifier: MIT */
#include <gtest/gtest.h>
#include "agx_test.h"
#include "compiler/nir/nir_builder.h"
extern "C" {
#include "agx_nir.h"
}

static unsigned
fadds(nir_shader *s)
{
   unsigned n = 0;
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            n += instr->type == nir_instr_type_alu &&
                 nir_instr_as_alu(instr)->op == nir_op_fadd;
         }
      }
   }
   return n;
}

static void
constants(bool is_max)
{
   const struct { uint32_t bits; bool flush; } cases[] = {
      {0, false}, {0x80000000u, false},
      {1, true}, {0x80000001u, true},
      {0x007fffffu, true}, {0x807fffffu, true},
      {0x00800000u, false}, {0x80800000u, false},
      {0x3f800000u, false}, {0xbf800000u, false},
      {0x7f800000u, false}, {0xff800000u, false},
      {0x7fc00000u, false}, {0x7f800001u, false},
   };
   for (const auto &test : cases) {
      const uint32_t bits = test.bits;
      SCOPED_TRACE(bits);
      nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
                                      &agx_nir_options, "constant-ftz");
      nir_def *a = nir_imm_int(&b, bits);
      nir_def *other = nir_imm_float(&b, 1.0f);
      nir_def *out = is_max ? nir_fmax(&b, a, other) : nir_fmin(&b, a, other);
      nir_intrinsic_instr *store = nir_store_output(&b, out, nir_imm_int(&b, 0));
      nir_io_semantics semantics = {};
      semantics.location = FRAG_RESULT_DATA0;
      nir_intrinsic_set_io_semantics(store, semantics);
      ASSERT_TRUE(agx_nir_lower_fminmax(b.shader, false));
      EXPECT_EQ(fadds(b.shader), unsigned(test.flush && !is_max));
      nir_validate_shader(b.shader, "constant-ftz");
      ralloc_free(b.shader);
   }
}
TEST(FminmaxConstantDenorm, DirectConstantMin) { constants(false); }
TEST(FminmaxConstantDenorm, DirectConstantMax) { constants(true); }

TEST(FminmaxConstantDenorm, NormalBoundsExcludeSubnormalResults)
{
   for (bool is_max : {false, true}) {
      for (unsigned side : {0u, 1u}) {
         for (uint32_t bits :
              {0u, 0x80000000u, 1u, 0x80000001u, 0x007fffffu, 0x807fffffu,
               0x00800000u, 0x80800000u, 0x3f800000u, 0xbf800000u, 0x7f800000u,
               0xff800000u, 0x7fc00000u}) {
            SCOPED_TRACE(bits);
            SCOPED_TRACE(is_max);
            SCOPED_TRACE(side);
            nir_builder b = nir_builder_init_simple_shader(
               MESA_SHADER_FRAGMENT, &agx_nir_options, "normal bound ftz");
            nir_def *input = agx_test_load_preamble(&b, 1, 32, 0);
            nir_def *bound = nir_imm_int(&b, bits);
            nir_def *x = side == 0 ? bound : input;
            nir_def *y = side == 0 ? input : bound;
            nir_def *out = is_max ? nir_fmax(&b, x, y) : nir_fmin(&b, x, y);
            agx_test_store_output(&b, out, nir_imm_int(&b, 0), nir_type_float32,
                                  FRAG_RESULT_DATA0);
            ASSERT_TRUE(agx_nir_lower_fminmax(b.shader, false));
            const unsigned exponent = (bits >> 23) & 255;
            const bool normal_bound =
               exponent && exponent != 255 && (bits >> 31) == !is_max;
            EXPECT_EQ(fadds(b.shader), unsigned(!normal_bound));
            nir_validate_shader(b.shader, "normal bound ftz");
            ralloc_free(b.shader);
         }
      }
   }
}

TEST(FminmaxConstantDenorm, SelectsPreservePossibleSubnormals)
{
   for (uint32_t x : {0u, 1u, 0x007fffffu, 0x00800000u, 0x3f800000u}) {
      for (uint32_t y : {0u, 0x80000001u, 0x80800000u, 0xbf800000u}) {
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_FRAGMENT, &agx_nir_options, "select ftz");
         nir_def *condition = nir_ieq_imm(&b, nir_load_sample_id(&b), 0);
         nir_def *selected =
            nir_bcsel(&b, condition, nir_imm_int(&b, x), nir_imm_int(&b, y));
         nir_def *out = nir_fmin(&b, selected, nir_imm_float(&b, 1.0));
         agx_test_store_output(&b, out, nir_imm_int(&b, 0), nir_type_float32,
                               FRAG_RESULT_DATA0);
         ASSERT_TRUE(agx_nir_lower_fminmax(b.shader, false));
         const bool denorm_x = !(x & 0x7f800000u) && (x & 0x007fffffu);
         const bool denorm_y = !(y & 0x7f800000u) && (y & 0x007fffffu);
         EXPECT_EQ(fadds(b.shader), unsigned(denorm_x || denorm_y));
         nir_validate_shader(b.shader, "select ftz");
         ralloc_free(b.shader);
      }
   }
}

TEST(FminmaxConstantDenorm, VectorLaneRequiresCanonicalization)
{
   for (bool is_max : {false, true}) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "vector ftz");
      nir_def *x = nir_imm_ivec2(&b, 0x3f800000, 1);
      nir_def *y = nir_imm_ivec2(&b, 0x3f800000, 2);
      nir_def *out = is_max ? nir_fmax(&b, x, y) : nir_fmin(&b, x, y);
      agx_test_store_output(&b, out, nir_imm_int(&b, 0), nir_type_float32,
                            FRAG_RESULT_DATA0);
      ASSERT_TRUE(agx_nir_lower_fminmax(b.shader, false));
      EXPECT_EQ(fadds(b.shader), 1u);
      nir_validate_shader(b.shader, "vector ftz");
      ralloc_free(b.shader);
   }
}

TEST(FminmaxConstantDenorm, PhiSourcesPreservePossibleSubnormals)
{
   for (uint32_t bits : {0u, 1u, 0x00800000u, 0xbf800000u}) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "phi ftz");
      nir_def *condition = nir_ieq_imm(&b, nir_load_sample_id(&b), 0);
      nir_push_if(&b, condition);
      nir_def *x = nir_imm_int(&b, bits);
      nir_push_else(&b, NULL);
      nir_def *y = nir_imm_float(&b, 1.0);
      nir_pop_if(&b, NULL);
      nir_def *value = nir_if_phi(&b, x, y);
      nir_def *out = nir_fmin(&b, value, nir_imm_float(&b, 1.0));
      agx_test_store_output(&b, out, nir_imm_int(&b, 0), nir_type_float32,
                            FRAG_RESULT_DATA0);
      ASSERT_TRUE(agx_nir_lower_fminmax(b.shader, false));
      EXPECT_EQ(fadds(b.shader), unsigned(bits == 1));
      nir_validate_shader(b.shader, "phi ftz");
      ralloc_free(b.shader);
   }
}

TEST(FminmaxConstantDenorm, DeepSelectorsKeepConservativeFlush)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "bounded select ftz");
   nir_def *condition = nir_ieq_imm(&b, nir_load_sample_id(&b), 0);
   nir_def *value = nir_imm_float(&b, 1.0);
   for (unsigned i = 0; i < 20; ++i)
      value = nir_bcsel(&b, condition, value, nir_imm_float(&b, 2.0));
   nir_def *out = nir_fmin(&b, value, nir_imm_float(&b, 4.0));
   agx_test_store_output(&b, out, nir_imm_int(&b, 0), nir_type_float32,
                         FRAG_RESULT_DATA0);
   ASSERT_TRUE(agx_nir_lower_fminmax(b.shader, false));
   EXPECT_EQ(fadds(b.shader), 1u);
   nir_validate_shader(b.shader, "bounded select ftz");
   ralloc_free(b.shader);
}

TEST(FminmaxConstantDenorm, RawMinmaxDoesNotProveCanonicalization)
{
   for (bool phi : {false, true}) {
      for (bool inner_max : {false, true}) {
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_FRAGMENT, &agx_nir_options, "raw minmax ftz");
         nir_def *input = agx_test_load_preamble(&b, 1, 32, 0);
         nir_def *bound = nir_imm_float(&b, inner_max ? -1.0 : 1.0);
         nir_def *raw = inner_max ? nir_fmax_agx(&b, input, bound)
                                  : nir_fmin_agx(&b, input, bound);
         nir_def *condition = nir_ieq_imm(&b, nir_load_sample_id(&b), 0);
         nir_def *value;
         if (phi) {
            nir_push_if(&b, condition);
            nir_def *x = nir_mov(&b, raw);
            nir_push_else(&b, NULL);
            nir_def *y = nir_imm_float(&b, 0.0);
            nir_pop_if(&b, NULL);
            value = nir_if_phi(&b, x, y);
         } else {
            value = nir_bcsel(&b, condition, raw, nir_imm_float(&b, 0.0));
         }
         nir_def *out =
            inner_max ? nir_fmax(&b, value, bound) : nir_fmin(&b, value, bound);
         agx_test_store_output(&b, out, nir_imm_int(&b, 0), nir_type_float32,
                               FRAG_RESULT_DATA0);
         ASSERT_TRUE(agx_nir_lower_fminmax(b.shader, false));
         EXPECT_EQ(fadds(b.shader), 1u);
         nir_validate_shader(b.shader, "raw minmax ftz");
         ralloc_free(b.shader);
      }
   }
}

/* In flush-to-zero-only mode (G17P), a min/max result may stay denormal unless
 * the shader declares flush-to-zero: no flushing add is inserted then, even
 * for a subnormal operand. A flush-to-zero shader keeps the flush. */
TEST(FminmaxConstantDenorm, FlushOnlyInFlushToZeroShaders)
{
   for (bool ftz : {false, true}) {
      for (bool is_max : {false, true}) {
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_FRAGMENT, &agx_nir_options, "ftz-only");
         if (ftz) {
            b.shader->info.float_controls_execution_mode |=
               FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP32;
         }
         nir_def *a = agx_test_load_preamble(&b, 1, 32, 0);
         nir_def *other = nir_imm_float(&b, 0.5f);
         nir_def *out =
            is_max ? nir_fmax(&b, a, other) : nir_fmin(&b, a, other);
         agx_test_store_output(&b, out, nir_imm_int(&b, 0), nir_type_float32,
                               FRAG_RESULT_DATA0);
         ASSERT_TRUE(agx_nir_lower_fminmax(b.shader, true));
         EXPECT_EQ(fadds(b.shader), unsigned(ftz && !is_max));
         nir_validate_shader(b.shader, "ftz-only");
         ralloc_free(b.shader);
      }
   }
}

static nir_def *
unknown_word(nir_builder *b)
{
   return nir_load_push_constant(b, 1, 32, nir_imm_int(b, 0), .range = 4);
}

static void
check_shared(uint32_t bits, bool arithmetic, bool mixed, bool narrow,
             unsigned alignment, bool preserve, bool expect_removed)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "shared-minmax");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   b.shader->info.shared_size = 16;
   if (preserve)
      b.shader->info.float_controls_execution_mode |=
         FLOAT_CONTROLS_DENORM_PRESERVE_FP32;
   nir_def *stored = arithmetic
                        ? nir_fmul(&b, unknown_word(&b), nir_imm_float(&b, 2))
                        : nir_imm_int(&b, bits);
   nir_store_shared(&b, stored, nir_imm_int(&b, 0), .align_mul = 4);
   if (mixed)
      nir_store_shared(&b, nir_imm_int(&b, 1), nir_imm_int(&b, 4),
                       .align_mul = 4);
   if (narrow)
      nir_store_shared(&b, nir_imm_intN_t(&b, 1, 16), nir_imm_int(&b, 8),
                       .align_mul = 2);
   nir_def *load =
      nir_load_shared(&b, 1, 32, nir_imm_int(&b, 0), .align_mul = alignment);
   nir_def *out = nir_fmin(&b, load, nir_imm_float(&b, 1));
   nir_store_ssbo(&b, out, nir_imm_int(&b, 0), nir_imm_int(&b, 0),
                  .write_mask = 1, .align_mul = 4);
   ASSERT_TRUE(agx_nir_lower_fminmax(b.shader, false));
   unsigned before = fadds(b.shader);
   ASSERT_EQ(before, 1u);
   EXPECT_EQ(agx_nir_opt_canonical_minmax(b.shader), expect_removed);
   EXPECT_EQ(fadds(b.shader), before - unsigned(expect_removed));
   nir_validate_shader(b.shader, "shared-minmax result");
   ralloc_free(b.shader);
}

TEST(FminmaxProvenance, SharedArithmetic)
{
   check_shared(0, true, false, false, 4, false, true);
}
TEST(FminmaxProvenance, SharedNormalConstant)
{
   check_shared(0x3f800000, false, false, false, 4, false, true);
}
TEST(FminmaxProvenance, SharedSubnormalConstant)
{
   check_shared(1, false, false, false, 4, false, false);
}
TEST(FminmaxProvenance, MixedUnknownWord)
{
   check_shared(0, true, true, false, 4, false, false);
}
TEST(FminmaxProvenance, NarrowWrite)
{
   check_shared(0, true, false, true, 4, false, false);
}
TEST(FminmaxProvenance, UnalignedRead)
{
   check_shared(0, true, false, false, 2, false, false);
}
TEST(FminmaxProvenance, DenormPreserve)
{
   check_shared(0, true, false, false, 4, true, false);
}
TEST(FminmaxProvenance, SharedSignallingNan)
{
   check_shared(0x7f800001, false, false, false, 4, false, false);
}

static void
check_select(bool unknown)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "select-minmax");
   nir_def *x = unknown_word(&b);
   nir_def *a = nir_fmul(&b, x, nir_imm_float(&b, 2));
   nir_def *value = nir_bcsel(&b, nir_ine_imm(&b, x, 0), a,
                              unknown ? x : nir_imm_float(&b, 0));
   nir_def *out = nir_fmax(&b, value, nir_imm_float(&b, -1));
   nir_store_ssbo(&b, out, nir_imm_int(&b, 0), nir_imm_int(&b, 0),
                  .write_mask = 1, .align_mul = 4);
   ASSERT_TRUE(agx_nir_lower_fminmax(b.shader, false));
   /* The lowering may already prove a canonical select; either pass must
    * leave exactly the unknown input's flush. */
   const unsigned before = fadds(b.shader);
   ASSERT_LE(before, 1u);
   EXPECT_EQ(agx_nir_opt_canonical_minmax(b.shader), before > unsigned(unknown));
   EXPECT_EQ(fadds(b.shader), unsigned(unknown));
   nir_validate_shader(b.shader, "select-minmax result");
   ralloc_free(b.shader);
}
TEST(FminmaxProvenance, CanonicalSelect)
{
   check_select(false);
}
TEST(FminmaxProvenance, UnknownSelect)
{
   check_select(true);
}

static void
check_phi(bool unknown)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "phi-minmax");
   nir_def *x = unknown_word(&b);
   nir_push_if(&b, nir_ine_imm(&b, x, 0));
   nir_def *a = nir_fmul(&b, x, nir_imm_float(&b, 2));
   nir_push_else(&b, nullptr);
   nir_def *other = unknown ? x : nir_imm_float(&b, 0);
   nir_pop_if(&b, nullptr);
   nir_def *value = nir_if_phi(&b, a, other);
   nir_def *out = nir_fmax(&b, value, nir_imm_float(&b, -1));
   nir_store_ssbo(&b, out, nir_imm_int(&b, 0), nir_imm_int(&b, 0),
                  .write_mask = 1, .align_mul = 4);
   ASSERT_TRUE(agx_nir_lower_fminmax(b.shader, false));
   const unsigned before = fadds(b.shader);
   ASSERT_LE(before, 1u);
   EXPECT_EQ(agx_nir_opt_canonical_minmax(b.shader), before > unsigned(unknown));
   EXPECT_EQ(fadds(b.shader), unsigned(unknown));
   nir_validate_shader(b.shader, "phi-minmax result");
   ralloc_free(b.shader);
}
TEST(FminmaxProvenance, CanonicalPhi)
{
   check_phi(false);
}
TEST(FminmaxProvenance, UnknownPhi)
{
   check_phi(true);
}
