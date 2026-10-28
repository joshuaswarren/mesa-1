/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "nir_test.h"

class nir_opt_reassociate_test : public nir_test {
protected:
   nir_opt_reassociate_test()
      : nir_test::nir_test("nir_opt_reassociate_test", MESA_SHADER_COMPUTE)
   {
   }

   nir_def *x, *inv, *root;

   /* Inside a loop, r = ((v1 + |x|) + v2) + x with v1 and v2 loaded in the
    * loop body and x loaded before it. */
   void build()
   {
      nir_def *addr = nir_undef(b, 1, 64);
      x = nir_load_global(b, 1, 32, addr);

      nir_loop *loop = nir_push_loop(b);
      nir_def *v1 = nir_load_global(b, 1, 32, addr);
      inv = nir_iabs(b, x);
      nir_def *v2 = nir_load_global(b, 1, 32, addr);
      root = nir_iadd(b, nir_iadd(b, nir_iadd(b, v1, inv), v2), x);
      nir_store_global(b, root, addr);
      nir_break_if(b, nir_undef(b, 1, 1));
      nir_pop_loop(b, loop);
   }

   /* Whether some iadd adds exactly the two loop-invariant terms. */
   bool invariant_pair()
   {
      nir_foreach_block(block, b->impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_alu)
               continue;
            nir_alu_instr *alu = nir_instr_as_alu(instr);
            if (alu->op != nir_op_iadd || list_is_empty(&alu->def.uses))
               continue;
            nir_def *a = alu->src[0].src.ssa, *c = alu->src[1].src.ssa;
            while (nir_def_is_alu(a) && nir_def_as_alu(a)->op == nir_op_mov)
               a = nir_def_as_alu(a)->src[0].src.ssa;
            while (nir_def_is_alu(c) && nir_def_as_alu(c)->op == nir_op_mov)
               c = nir_def_as_alu(c)->src[0].src.ssa;
            if ((a == x && c == inv) || (a == inv && c == x))
               return true;
         }
      }
      return false;
   }
};

TEST_F(nir_opt_reassociate_test, loop_invariant_terms_first)
{
   build();
   ASSERT_TRUE(nir_opt_reassociate(b->shader, nir_reassociate_loop_invariant));
   nir_validate_shader(b->shader, NULL);
   nir_opt_dce(b->shader);
   ASSERT_TRUE(invariant_pair());
}

TEST_F(nir_opt_reassociate_test, loop_invariance_off)
{
   /* Without the option every source has the same rank: nothing to do. */
   build();
   ASSERT_FALSE(nir_opt_reassociate(b->shader, (nir_reassociate_options)0));
   ASSERT_FALSE(invariant_pair());
}
