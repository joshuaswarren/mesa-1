/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include <stdlib.h>
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
      /*
       * Diagnostic (AGX_G17P_SWAP_SCRATCH=1): swap through a fresh scratch
       * register instead of XOR-ing in place.
       *
       * The XOR swap reads and writes its two registers three times. When one
       * of them is a texture destination -- which is exactly what happens when
       * the exports of several indexed samples form a register cycle -- those
       * are ALU reads and writes aimed at a register the texture unit has just
       * asynchronously written. This is the shape at the tail of bug #1's
       * shader, and banning the allocator off r0/r1 (which removes the cycle,
       * and with it the XOR swap) takes that shader from 49/60 failures to
       * 0/300. This knob removes the XOR swap without touching allocation, so
       * the two can be told apart.
       *
       * Safe to take a register above the high-water mark here: this pass runs
       * after agx_ra and before nr_gprs is derived from ctx->max_reg
       * (agx_compile.c), so growing max_reg grows the shader's declared
       * footprint with it. It costs occupancy, hence diagnostic-only.
       */
      agx_context *ctx = b->shader;
      const unsigned n = agx_size_align_16(x.size);
      const unsigned base = ALIGN_POT(ctx->max_reg + 1, n);
      static int scratch = -1;

      if (scratch < 0) {
         const char *v = getenv("AGX_G17P_SWAP_SCRATCH");
         scratch = (v && v[0] && v[0] != '0') ? 1 : 0;
      }

      if (scratch && !x.memory && !y.memory && base + n <= AGX_NUM_REGS) {
         agx_index tmp = agx_register(base, x.size);

         ctx->max_reg = MAX2(ctx->max_reg, base + n - 1);

         /* Emit the already-lowered bitwise move directly: this runs inside
          * lower(), so a plain MOV would never be re-lowered and the packer
          * refuses it. */
         agx_bitop_to(b, tmp, x, agx_zero(), UTIL_LUT2(a));
         agx_bitop_to(b, x, y, agx_zero(), UTIL_LUT2(a));
         agx_bitop_to(b, y, tmp, agx_zero(), UTIL_LUT2(a));
      } else {
         /* Otherwise, we're swapping GPRs and fallback on a XOR swap. */
         agx_xor_to(b, x, x, y);
         agx_xor_to(b, y, x, y);
         agx_xor_to(b, x, x, y);
      }
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
      /*
       * A modifier on a move is agx_optimizer_fmov() having composed an fneg
       * or fabs into the copy's source: agx_is_fmov() matches `fadd(x, -0.0)`,
       * and agx_compose_float_src() carries its abs/neg across. The bitwise
       * move below has no field for one, so the packer refused the whole
       * shader for `-x` or `|x|` crossing a copy -- 19 of the 307 refusals in
       * a three.js frame. Put the float move back instead: Apple spells a move
       * as an add of the exact negative-zero minifloat, and that form carries
       * both modifier bits (agx3_alu_fadd_minifloat_mod, agx3_alu_fadd_abs_*),
       * so this is the pre-fold program rather than a new encoding.
       */
      if (agx3 && (I->src[0].abs || I->src[0].neg) &&
          I->src[0].size == AGX_SIZE_32 &&
          I->dest[0].size == AGX_SIZE_32)
         return agx_fadd_to(b, I->dest[0], I->src[0], agx_negzero());
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
         /* AGX3 has no unconditional break instruction. Break on the
          * integer identity r0l == r0l instead, using the compare+cnd
          * sequence without clobbering a post-RA register.
          *
          * NEST carries over unchanged: it is the number of exec-mask levels
          * the break unwinds in Mesa's model either way, and agx_opt_break_if
          * only subtracts one when it actually folds an enclosing if away.
          * The packer turns that into the native level count (nest + 1,
          * because a loop owns two architectural levels), which CNDCF
          * encodes at any depth. */
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

/*
 * AGX3 has no control-flow compare with an immediate on both sides. A constant
 * condition can still reach one: SPIR-V such as dEQP-VK.reconvergence's keeps
 * `if (true) { break; }` inside loops, and immediate inlining turns the
 * condition into an immediate. The construct itself must stay, since it owns
 * an exec-mask level, so keep the instruction and branch on the integer
 * identity r0l == r0l, as an AGX3 unconditional break already does, inverted
 * to preserve the constant outcome.
 */
static void
agx3_lower_constant_cf_compare(agx_context *ctx, agx_instr *I)
{
   if (I->op != AGX_OPCODE_IF_ICMP && I->op != AGX_OPCODE_ELSE_ICMP &&
       I->op != AGX_OPCODE_WHILE_ICMP && I->op != AGX_OPCODE_BREAK_IF_ICMP)
      return;

   bool cond;
   if (!agx_eval_immediate_icond(ctx, I->icond, I->src[0], I->src[1], &cond))
      return;

   agx_index r0l = agx_register(0, AGX_SIZE_16);
   I->src[0] = r0l;
   I->src[1] = r0l;
   I->icond = AGX_ICOND_UEQ;
   I->invert_cond = cond == I->invert_cond;
}

void
agx_lower_pseudo(agx_context *ctx)
{
   agx_foreach_instr_global_safe(ctx, I) {
      agx_builder b = agx_init_builder(ctx, agx_before_instr(I));

      if (ctx->key && ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B)
         agx3_lower_constant_cf_compare(ctx, I);

      if (lower(&b, I))
         agx_remove_instruction(I);
   }
}
