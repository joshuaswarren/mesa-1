/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "util/lut.h"
#include "agx_builder.h"
#include "agx_compiler.h"
#include "agx_opcodes.h"

/* Lower pseudo instructions created during optimization. */

static agx_instr *
while_for_break_if(agx_builder *b, agx_instr *I)
{
   if (I->op == AGX_OPCODE_BREAK_IF_FCMP) {
      return agx_while_fcmp(b, I->src[0], I->src[1], I->nest, I->fcond,
                            !I->invert_cond, NULL);
   } else {
      return agx_while_icmp(b, I->src[0], I->src[1], I->nest, I->icond,
                            !I->invert_cond, NULL);
   }
}

static agx_instr *
cmpsel_for_break_if(agx_builder *b, agx_instr *I)
{
   agx_index r0l = agx_register(0, AGX_SIZE_16);

   /* If the condition is true, set r0l to nest to break */
   agx_index t = agx_immediate(I->nest);
   agx_index f = r0l;

   if (I->invert_cond) {
      SWAP(t, f);
   }

   if (I->op == AGX_OPCODE_BREAK_IF_FCMP)
      agx_fcmpsel_to(b, r0l, I->src[0], I->src[1], t, f, I->fcond);
   else
      agx_icmpsel_to(b, r0l, I->src[0], I->src[1], t, f, I->icond);

   return agx_push_exec(b, 0);
}

static void
swap(agx_builder *b, agx_index x, agx_index y)
{
   assert(!x.memory && "already lowered");
   assert(!y.memory && "already lowered");

   /* We can swap lo/hi halves of a 32-bit register with a 32-bit extr */
   if (x.size == AGX_SIZE_16 && (x.value >> 1) == (y.value >> 1)) {

      assert(((x.value & 1) == (1 - (y.value & 1))) &&
             "no trivial swaps, and only 2 halves of a register");

      /* r0 = extr r0, r0, #16
       *    = (((r0 << 32) | r0) >> 16) & 0xFFFFFFFF
       *    = (((r0 << 32) >> 16) & 0xFFFFFFFF) | (r0 >> 16)
       *    = (r0l << 16) | r0h
       */
      agx_index reg32 = agx_register(x.value & ~1, AGX_SIZE_32);
      agx_extr_to(b, reg32, reg32, reg32, agx_immediate(16), 0);
   } else {
      /* Otherwise, we're swapping GPRs and fallback on a XOR swap. */
      agx_xor_to(b, x, x, y);
      agx_xor_to(b, y, x, y);
      agx_xor_to(b, x, x, y);
   }
}

static agx_instr *
lower(agx_builder *b, agx_instr *I)
{
   const bool agx3 = b->shader->key != NULL &&
                     b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2;

   switch (I->op) {

   /* Various instructions are implemented as bitwise truth tables */
   case AGX_OPCODE_MOV:
      if (agx3 && I->src[0].type == AGX_INDEX_IMMEDIATE)
         return agx_mov_imm_to(b, I->dest[0], I->src[0].value);
      return agx_bitop_to(b, I->dest[0], I->src[0], agx_zero(), UTIL_LUT2(a));

   case AGX_OPCODE_NOT:
      return agx_bitop_to(b, I->dest[0], I->src[0], agx_zero(), UTIL_LUT2(~a));

   /* We can sign-extend with an add */
   case AGX_OPCODE_SIGNEXT:
      return agx_iadd_to(b, I->dest[0], I->src[0], agx_zero(), 0);

   /* Unfused comparisons are fused with a 0/1 select */
   case AGX_OPCODE_ICMP:
      return agx_icmpsel_to(b, I->dest[0], I->src[0], I->src[1],
                            agx_immediate(I->invert_cond ? 0 : 1),
                            agx_immediate(I->invert_cond ? 1 : 0), I->icond);

   case AGX_OPCODE_FCMP:
      return agx_fcmpsel_to(b, I->dest[0], I->src[0], I->src[1],
                            agx_immediate(I->invert_cond ? 0 : 1),
                            agx_immediate(I->invert_cond ? 1 : 0), I->fcond);

   case AGX_OPCODE_BALLOT:
      return agx_icmp_ballot_to(b, I->dest[0], I->src[0], agx_zero(),
                                AGX_ICOND_UEQ, true /* invert */);

   case AGX_OPCODE_QUAD_BALLOT:
      return agx_icmp_quad_ballot_to(b, I->dest[0], I->src[0], agx_zero(),
                                     AGX_ICOND_UEQ, true /* invert */);

   /* Writes to the nesting counter lowered to the real register */
   case AGX_OPCODE_BEGIN_CF:
      /* AGX3 has an architectural exec-mask stack and no witnessed r0l
       * nesting-counter write. Keep the pseudo for its zero-byte AGX3 packer
       * mapping instead of silently emitting the AGX2 contract. */
      if (agx3)
         return NULL;
      return agx_mov_imm_to(b, agx_register(0, AGX_SIZE_16), 0);

   case AGX_OPCODE_BREAK:
      if (agx3) {
         /* AGX3 has no witnessed unconditional break instruction. Break on
          * the integer identity r0l == r0l instead, using the grounded
          * compare+cndlt sequence without clobbering a post-RA register.
          *
          * A depth-2 BREAK is the unfused form produced for a loop break in
          * one enclosing if. agx_opt_break_if proves that folding away that
          * if subtracts one from the nesting depth, so its always-true fused
          * form is the same depth-1 BREAK_IF accepted here. Deeper forms stay
          * fail-closed until an equivalent nested witness exists. */
         if (I->nest > 2)
            return NULL;

         agx_index r0l = agx_register(0, AGX_SIZE_16);
         return agx_break_if_icmp(b, r0l, r0l, I->nest, false, AGX_ICOND_UEQ,
                                  I->target);
      }
      agx_mov_imm_to(b, agx_register(0, AGX_SIZE_16), I->nest);
      return agx_pop_exec(b, 0);

   case AGX_OPCODE_BREAK_IF_ICMP:
   case AGX_OPCODE_BREAK_IF_FCMP: {
      /* AGX3 lowers these directly to compare + cndlt, retaining the
       * condition and avoiding AGX2's r0l/cmpsel fallback. */
      if (agx3)
         return NULL;
      if (I->nest == 1)
         return while_for_break_if(b, I);
      else
         return cmpsel_for_break_if(b, I);
   }

   case AGX_OPCODE_SWAP:
      swap(b, I->src[0], I->src[1]);
      return (void *)true;

   default:
      return NULL;
   }
}

void
agx_lower_pseudo(agx_context *ctx)
{
   agx_foreach_instr_global_safe(ctx, I) {
      agx_builder b = agx_init_builder(ctx, agx_before_instr(I));

      if (lower(&b, I))
         agx_remove_instruction(I);
   }
}
