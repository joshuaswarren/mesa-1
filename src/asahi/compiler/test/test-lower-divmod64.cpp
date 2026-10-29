/* SPDX-License-Identifier: MIT */
#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "agx_nir.h"
}

/* agx_nir_lower_divmod64 replaces each 64-bit division with a loop whose trip
 * count depends on the operands. Substituting constant operands lets loop
 * analysis count the trips, unroll the loop and fold the whole computation,
 * which evaluates the lowered NIR without hardware. */

static nir_shader_compiler_options
unroll_options(void)
{
   nir_shader_compiler_options options = agx_nir_options;
   options.max_unroll_iterations = 128;
   return options;
}

static nir_shader *
make_program(const nir_shader_compiler_options *options, nir_op op,
             unsigned components)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, options,
                                                  "divmod64");
   nir_def *n = agx_test_load_preamble(&b, components, 64, 0);
   nir_def *d = agx_test_load_preamble(&b, components, 64, 16);
   nir_def *result = nir_build_alu(&b, op, n, d, NULL, NULL);
   nir_store_global(&b, result, nir_imm_int64(&b, 0), .align_mul = 8);
   nir_validate_shader(b.shader, "before divmod64 lowering");
   return b.shader;
}

static bool
evaluate(nir_shader *shader, const uint64_t *n, const uint64_t *d,
         unsigned components, uint64_t *out)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   nir_builder builder = nir_builder_create(impl);
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_load_preamble)
            continue;
         const uint64_t *src = nir_intrinsic_base(intr) == 0 ? n : d;
         nir_const_value values[NIR_MAX_VEC_COMPONENTS];
         for (unsigned c = 0; c < components; ++c)
            values[c] = nir_const_value_for_uint(src[c], 64);
         builder.cursor = nir_before_instr(instr);
         nir_def_rewrite_uses(
            &intr->def, nir_build_imm(&builder, components, 64, values));
         nir_instr_remove(instr);
      }
   }

   bool progress;
   do {
      progress = false;
      progress |= nir_opt_constant_folding(shader);
      progress |= nir_opt_copy_prop(shader);
      progress |= nir_opt_dce(shader);
      progress |= nir_opt_dead_cf(shader);
      progress |= nir_opt_remove_phis(shader);
      progress |= nir_opt_if(shader, nir_opt_if_optimize_phi_true_false);
      progress |= nir_opt_loop_unroll(shader);
   } while (progress);

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_store_global)
            continue;
         if (!nir_src_is_const(intr->src[0]))
            return false;
         for (unsigned c = 0; c < components; ++c)
            out[c] = nir_src_comp_as_uint(intr->src[0], c);
         return true;
      }
   }
   return false;
}

static uint64_t
reference(nir_op op, uint64_t a, uint64_t b)
{
   const int64_t sa = (int64_t)a, sb = (int64_t)b;
   switch (op) {
   case nir_op_udiv:
      return a / b;
   case nir_op_umod:
      return a % b;
   case nir_op_idiv:
      return (uint64_t)(sa / sb);
   case nir_op_irem:
      return (uint64_t)(sa % sb);
   case nir_op_imod: {
      const int64_t r = sa % sb;
      return (uint64_t)(r != 0 && ((r < 0) != (sb < 0)) ? r + sb : r);
   }
   default:
      return 0;
   }
}

static const uint64_t values[] = {
   0,
   1,
   2,
   3,
   7,
   0x80,
   0xffffffffull,
   0x100000000ull,
   0x123456789abcdefull,
   0x7fffffffffffffffull,
   0x7fff859a3bf78592ull,
   0xffff859a3bf78592ull,
   0x8000000000000001ull,
   0xfffffffffffffffeull,
   0xffffffffffffffffull,
   0xdeadbeefcafef00dull,
};

TEST(LowerDivmod64, MatchesReference)
{
   const nir_shader_compiler_options options = unroll_options();
   const nir_op ops[] = {nir_op_udiv, nir_op_umod, nir_op_idiv, nir_op_irem,
                         nir_op_imod};
   /* Both forms of the loop's 64-bit add and subtract (G17P and M5). */
   for (bool native_add64 : {false, true}) {
      for (nir_op op : ops) {
         const bool is_signed = op == nir_op_idiv || op == nir_op_irem ||
                                op == nir_op_imod;
         for (uint64_t n : values) {
            for (uint64_t d : values) {
               if (d == 0)
                  continue;
               /* INT64_MIN / -1 overflows and is undefined. */
               if (is_signed && n == 0x8000000000000000ull &&
                   d == ~UINT64_C(0))
                  continue;
               nir_shader *shader = make_program(&options, op, 1);
               ASSERT_TRUE(agx_nir_lower_divmod64(shader, native_add64));
               nir_validate_shader(shader, "after divmod64 lowering");
               uint64_t got = 0;
               ASSERT_TRUE(evaluate(shader, &n, &d, 1, &got))
                  << nir_op_infos[op].name << " " << n << " " << d;
               EXPECT_EQ(got, reference(op, n, d))
                  << nir_op_infos[op].name << " " << std::hex << n << " / "
                  << d;
               ralloc_free(shader);
            }
         }
      }
   }
}

TEST(LowerDivmod64, VectorComponentsAreIndependent)
{
   const nir_shader_compiler_options options = unroll_options();
   const uint64_t n[4] = {100, 0xffff859a3bf78592ull, 7, 0x123456789abcdefull};
   const uint64_t d[4] = {7, 3, 0x100000000ull, 0xffffffffull};
   nir_shader *shader = make_program(&options, nir_op_idiv, 4);
   ASSERT_TRUE(agx_nir_lower_divmod64(shader, false));
   uint64_t got[4] = {0};
   ASSERT_TRUE(evaluate(shader, n, d, 4, got));
   for (unsigned c = 0; c < 4; ++c)
      EXPECT_EQ(got[c], reference(nir_op_idiv, n[c], d[c])) << c;
   ralloc_free(shader);
}

TEST(LowerDivmod64, LeavesNarrowDivisionAlone)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "divmod32");
   nir_def *n = agx_test_load_preamble(&b, 1, 32, 0);
   nir_def *d = agx_test_load_preamble(&b, 1, 32, 2);
   nir_store_global(&b, nir_udiv(&b, n, d), nir_imm_int64(&b, 0),
                    .align_mul = 4);
   EXPECT_FALSE(agx_nir_lower_divmod64(b.shader, false));
   ralloc_free(b.shader);
}
