/*
 * Copyright 2024 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "util/bitset.h"
#include "util/half_float.h"
#include "agx_builder.h"
#include "agx_compiler.h"
#include "agx_opcodes.h"

/*
 * AGX can convert 16-bit sources to 32-bit for free, so it's beneficial to
 * compact 32-bit constants down to 16-bit when doing so is lossless. This
 * reduces register pressure (GPR or uniform, depending on whether the constant
 * is promoted).
 */
void
agx_opt_compact_constants(agx_context *ctx)
{
   /* TODO: Handle ints too */
   BITSET_WORD *src_float = calloc(ctx->alloc, sizeof(BITSET_WORD));
   BITSET_WORD *src_other = calloc(ctx->alloc, sizeof(BITSET_WORD));
   BITSET_WORD *src_wide = calloc(ctx->alloc, sizeof(BITSET_WORD));
   BITSET_WORD *replaced = calloc(ctx->alloc, sizeof(BITSET_WORD));

   /* AGX3 has no generally grounded f32 arithmetic over an f16 register, so
    * narrowing a constant that feeds a full-width add, multiply or FMA hands
    * the packer a mixed-width operand pair it cannot encode and the whole
    * shader is refused. agx_optimizer.c already declines to fold f2f32 into
    * these same opcodes for that reason; this is the other producer of the
    * same shape, and the one a transform-feedback vertex shader hits --
    * `v + 100.0` narrows the constant because 100.0 round-trips exactly
    * through f16. Widening stays explicit instead. */
   const bool agx3 = ctx->key && agx_encoding_is_agx3(ctx->key->dev.encoding);

   /* Analyze the types that we read constants as */
   agx_foreach_instr_global(ctx, I) {
      const bool wide_float_op =
         agx3 && I->nr_dests >= 1 && I->dest[0].size == AGX_SIZE_32 &&
         (I->op == AGX_OPCODE_FADD || I->op == AGX_OPCODE_FMUL ||
          I->op == AGX_OPCODE_FFMA);

      agx_foreach_ssa_src(I, s) {
         if (agx_is_float_src(I, s)) {
            BITSET_SET(src_float, I->src[s].value);
            if (wide_float_op)
               BITSET_SET(src_wide, I->src[s].value);
         } else
            BITSET_SET(src_other, I->src[s].value);
      }
   }

   agx_foreach_instr_global(ctx, I) {
      if (I->op == AGX_OPCODE_MOV_IMM && I->dest[0].size == AGX_SIZE_32) {
         unsigned v = I->dest[0].value;

         if (BITSET_TEST(src_float, v) && !BITSET_TEST(src_other, v) &&
             !BITSET_TEST(src_wide, v)) {
            /* Try to compact to f16 */
            uint16_t compact = _mesa_float_to_half(uif(I->imm));

            if (I->imm == fui(_mesa_half_to_float(compact))) {
               I->dest[0].size = AGX_SIZE_16;
               I->imm = compact;
               BITSET_SET(replaced, v);
            }
         }
      } else {
         agx_foreach_ssa_src(I, s) {
            if (BITSET_TEST(replaced, I->src[s].value)) {
               I->src[s].size = AGX_SIZE_16;
            }
         }
      }
   }

   free(replaced);
   free(src_wide);
   free(src_float);
   free(src_other);
}
