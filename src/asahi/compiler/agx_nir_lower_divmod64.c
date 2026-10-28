/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/*
 * 64-bit integer division by normalized restoring division.
 *
 * nir_lower_int64 expands every 64-bit udiv/umod into a fully unrolled
 * bit-at-a-time division: 32 conditional steps for the high quotient word and
 * 32 for the low word, several hundred instructions per scalar division
 * whatever the operands. Instead, align the divisor with the dividend using
 * their leading-zero counts and run a shift-subtract loop only over the
 * quotient bits that can be set.
 *
 * This pass emits that algorithm in NIR; the nir_lower_int64 call that
 * follows lowers its 64-bit compares, shifts, negations and selects to 32-bit
 * operations. The loop runs msb(n) - msb(d) + 1 times when n >= d and not at
 * all otherwise.
 *
 * Division by zero is undefined in Vulkan/SPIR-V; it returns quotient 0 and
 * remainder n here without looping.
 */

#include "compiler/nir/nir_builder.h"
#include "agx_nir.h"

/* G17P has no 64-bit add or subtract, and nir_lower_int64 only lowers those
 * for shaders whose source has a 64-bit add, so build them from 32-bit halves
 * here as nir_lower_int64 would. M5 adds and subtracts 64-bit values natively
 * (native_add64). */
static nir_def *
iadd64(nir_builder *b, nir_def *x, nir_def *y, bool native_add64)
{
   if (native_add64)
      return nir_iadd(b, x, y);

   nir_def *x_lo = nir_unpack_64_2x32_split_x(b, x);
   nir_def *x_hi = nir_unpack_64_2x32_split_y(b, x);
   nir_def *y_lo = nir_unpack_64_2x32_split_x(b, y);
   nir_def *y_hi = nir_unpack_64_2x32_split_y(b, y);

   nir_def *lo = nir_iadd(b, x_lo, y_lo);
   nir_def *carry = nir_b2i32(b, nir_ult(b, lo, x_lo));
   nir_def *hi = nir_iadd(b, carry, nir_iadd(b, x_hi, y_hi));
   return nir_pack_64_2x32_split(b, lo, hi);
}

static nir_def *
isub64(nir_builder *b, nir_def *x, nir_def *y, bool native_add64)
{
   if (native_add64)
      return nir_isub(b, x, y);

   nir_def *x_lo = nir_unpack_64_2x32_split_x(b, x);
   nir_def *x_hi = nir_unpack_64_2x32_split_y(b, x);
   nir_def *y_lo = nir_unpack_64_2x32_split_x(b, y);
   nir_def *y_hi = nir_unpack_64_2x32_split_y(b, y);

   nir_def *lo = nir_isub(b, x_lo, y_lo);
   nir_def *borrow = nir_ineg(b, nir_b2i32(b, nir_ult(b, x_lo, y_lo)));
   nir_def *hi = nir_iadd(b, nir_isub(b, x_hi, y_hi), borrow);
   return nir_pack_64_2x32_split(b, lo, hi);
}

static void
udivmod64(nir_builder *b, nir_def *n, nir_def *d, bool native_add64,
          nir_def **q_out, nir_def **r_out)
{
   nir_variable *q_var =
      nir_local_variable_create(b->impl, glsl_uint64_t_type(), "divmod64_q");
   nir_variable *r_var =
      nir_local_variable_create(b->impl, glsl_uint64_t_type(), "divmod64_r");
   nir_variable *d_var =
      nir_local_variable_create(b->impl, glsl_uint64_t_type(), "divmod64_d");
   nir_variable *i_var =
      nir_local_variable_create(b->impl, glsl_uint_type(), "divmod64_i");

   nir_store_var(b, q_var, nir_imm_int64(b, 0), 1);
   nir_store_var(b, r_var, n, 1);

   nir_push_if(b, nir_iand(b, nir_ine_imm(b, d, 0), nir_uge(b, n, d)));
   {
      /* n >= d > 0, so msb(n) >= msb(d) and the shift is in [0, 63]. */
      nir_def *shift =
         nir_isub(b, nir_ufind_msb(b, n), nir_ufind_msb(b, d));
      nir_store_var(b, d_var, nir_ishl(b, d, shift), 1);
      nir_store_var(b, i_var, shift, 1);

      nir_push_loop(b);
      {
         nir_def *r = nir_load_var(b, r_var);
         nir_def *dd = nir_load_var(b, d_var);
         nir_def *ge = nir_uge(b, r, dd);

         nir_store_var(b, r_var, nir_bcsel(b, ge, isub64(b, r, dd, native_add64), r), 1);
         nir_store_var(b, q_var,
                       nir_ior(b, nir_ishl_imm(b, nir_load_var(b, q_var), 1),
                               nir_b2i64(b, ge)),
                       1);
         nir_store_var(b, d_var, nir_ushr_imm(b, dd, 1), 1);

         nir_def *i = nir_load_var(b, i_var);
         nir_break_if(b, nir_ieq_imm(b, i, 0));
         nir_store_var(b, i_var, nir_iadd_imm(b, i, -1), 1);
      }
      nir_pop_loop(b, NULL);
   }
   nir_pop_if(b, NULL);

   *q_out = nir_load_var(b, q_var);
   *r_out = nir_load_var(b, r_var);
}

static nir_def *
lower_scalar(nir_builder *b, nir_op op, nir_def *n, nir_def *d,
             bool native_add64)
{
   nir_def *q, *r;

   switch (op) {
   case nir_op_udiv:
      udivmod64(b, n, d, native_add64, &q, &r);
      return q;
   case nir_op_umod:
      udivmod64(b, n, d, native_add64, &q, &r);
      return r;
   default:
      break;
   }

   /* Signed: divide magnitudes. |INT64_MIN| is 2^63 as an unsigned value. */
   nir_def *n_neg = nir_ilt_imm(b, n, 0);
   nir_def *d_neg = nir_ilt_imm(b, d, 0);
   nir_def *n_abs = nir_bcsel(b, n_neg, nir_ineg(b, n), n);
   nir_def *d_abs = nir_bcsel(b, d_neg, nir_ineg(b, d), d);
   udivmod64(b, n_abs, d_abs, native_add64, &q, &r);

   switch (op) {
   case nir_op_idiv:
      /* Truncating: negative when exactly one operand is negative. */
      return nir_bcsel(b, nir_ixor(b, n_neg, d_neg), nir_ineg(b, q), q);
   case nir_op_irem:
      /* Sign of the dividend. */
      return nir_bcsel(b, n_neg, nir_ineg(b, r), r);
   case nir_op_imod: {
      /* Sign of the divisor: add it to a nonzero remainder of the other
       * sign. */
      nir_def *rem = nir_bcsel(b, n_neg, nir_ineg(b, r), r);
      nir_def *fix = nir_iand(b, nir_ine_imm(b, rem, 0),
                              nir_ine(b, n_neg, d_neg));
      return nir_bcsel(b, fix, iadd64(b, rem, d, native_add64), rem);
   }
   default:
      UNREACHABLE("checked by the caller");
   }
}

static bool
lower_divmod64(nir_builder *b, nir_alu_instr *alu, bool native_add64)
{
   switch (alu->op) {
   case nir_op_udiv:
   case nir_op_umod:
   case nir_op_idiv:
   case nir_op_irem:
   case nir_op_imod:
      break;
   default:
      return false;
   }
   if (alu->def.bit_size != 64)
      return false;

   b->cursor = nir_before_instr(&alu->instr);
   nir_def *n = nir_ssa_for_alu_src(b, alu, 0);
   nir_def *d = nir_ssa_for_alu_src(b, alu, 1);

   nir_def *comps[NIR_MAX_VEC_COMPONENTS];
   for (unsigned c = 0; c < alu->def.num_components; ++c) {
      comps[c] = lower_scalar(b, alu->op, nir_channel(b, n, c),
                              nir_channel(b, d, c), native_add64);
   }

   nir_def_replace(&alu->def, nir_vec(b, comps, alu->def.num_components));
   return true;
}

bool
agx_nir_lower_divmod64(nir_shader *shader, bool native_add64)
{
   bool progress = false;

   nir_foreach_function_impl(impl, shader) {
      /* Collect first: lowering inserts control flow at each site. */
      struct util_dynarray sites = UTIL_DYNARRAY_INIT;
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_alu)
               util_dynarray_append(&sites, nir_instr_as_alu(instr));
         }
      }

      bool impl_progress = false;
      nir_builder b = nir_builder_create(impl);
      util_dynarray_foreach(&sites, nir_alu_instr *, alu)
         impl_progress |= lower_divmod64(&b, *alu, native_add64);
      util_dynarray_fini(&sites);

      nir_progress(impl_progress, impl, nir_metadata_none);
      progress |= impl_progress;
   }

   if (progress) {
      NIR_PASS(_, shader, nir_lower_vars_to_ssa);
      NIR_PASS(_, shader, nir_remove_dead_variables, nir_var_function_temp,
               NULL);
   }
   return progress;
}
