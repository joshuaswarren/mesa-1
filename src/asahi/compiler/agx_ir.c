/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "agx_compiler.h"
#include "agx_opcodes.h"

bool
agx_allows_16bit_immediate(agx_instr *I)
{
   return (I->op == AGX_OPCODE_DEVICE_LOAD) ||
          (I->op == AGX_OPCODE_DEVICE_STORE) ||
          (I->op == AGX_OPCODE_STACK_LOAD) ||
          (I->op == AGX_OPCODE_STACK_STORE) ||
          (I->op == AGX_OPCODE_UNIFORM_STORE) || (I->op == AGX_OPCODE_ATOMIC) ||
          (I->op == AGX_OPCODE_LOCAL_ATOMIC) || (I->op == AGX_OPCODE_PHI) ||
          (I->op == AGX_OPCODE_TEX_STATE_STORE) ||
          (I->op == AGX_OPCODE_SAMPLER_STATE_STORE);
}

unsigned
agx_negate_src_index(agx_instr *I)
{
   switch (I->op) {
   case AGX_OPCODE_IMAD:
      return 2;
   case AGX_OPCODE_IADD:
      return 1;
   default:
      UNREACHABLE("not allowed");
   }
}

/* A complete fragment may sample through structured control flow and finish
 * with an ordinary MRT epilog. Recognize only a reconverged, colour-only tail;
 * nothing from the prefix may enter a pixel transaction or have write effects.
 * Both ordering (SSA) and packing (physical registers) use this same proof.
 */
agx_block *
agx_reconverged_mrt_tail(agx_context *ctx)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B ||
       ctx->stage != MESA_SHADER_FRAGMENT || ctx->is_preamble ||
       ctx->key->secondary || ctx->key->no_stop ||
       ctx->key->fs.ignore_tib_dependencies || ctx->key->g17p_emrt ||
       ctx->key->fs.nr_samples > 1 || list_length(&ctx->blocks) <= 1 ||
       !ctx->nir || !ctx->indexed_nir_blocks)
      return NULL;

   agx_block *tail = list_last_entry(&ctx->blocks, agx_block, link);
   /* The compiler supports only entrypoint and preamble functions. The
    * preamble was excluded above, so this is the map for the entrypoint. */
   nir_function_impl *impl = nir_shader_get_entrypoint(ctx->nir);
   nir_block *nir_tail = nir_impl_last_block(impl);
   /* The last top-level NIR block is outside every if/loop. Do not use
    * agx_block::divergent here: after-blocks are conservatively allocated
    * before emit_if decrements the nesting counter.
    */
   if (ctx->indexed_nir_blocks[nir_tail->index] != tail ||
       agx_num_successors(tail) != 0)
      return NULL;

   bool tile = false, stop = false;
   agx_foreach_block(ctx, block) {
      /* A unique CFG exit means every normally terminating path reaches
       * the reconverged tail, even when the prefix contains a loop.
       */
      if (block != tail && !agx_num_successors(block))
         return NULL;
      agx_foreach_instr_in_block(block, I) {
         switch (I->op) {
         case AGX_OPCODE_LD_TILE:
         case AGX_OPCODE_ST_TILE:
            if (block != tail)
               return NULL;
            tile = true;
            break;
         case AGX_OPCODE_WAIT_PIX:
            if (block != tail || I->writeout == 1)
               return NULL;
            break;
         case AGX_OPCODE_STOP:
            if (block != tail || I->link.next != &tail->instructions)
               return NULL;
            stop = true;
            break;
         case AGX_OPCODE_BEGIN_CF:
         case AGX_OPCODE_IF_ICMP:
         case AGX_OPCODE_IF_FCMP:
         case AGX_OPCODE_WHILE_ICMP:
         case AGX_OPCODE_WHILE_FCMP:
         case AGX_OPCODE_BREAK:
         case AGX_OPCODE_BREAK_IF_ICMP:
         case AGX_OPCODE_BREAK_IF_FCMP:
         case AGX_OPCODE_ELSE_ICMP:
         case AGX_OPCODE_ELSE_FCMP:
         case AGX_OPCODE_POP_EXEC:
         case AGX_OPCODE_JMP_EXEC_ANY:
         case AGX_OPCODE_JMP_EXEC_NONE:
         case AGX_OPCODE_JMP_EXEC_NONE_AFTER:
            if (block == tail)
               return NULL;
            break;
         default:
            /* Includes depth/coverage writes, pixel releases, atomics,
             * stores and barriers. Reads and pure ALU remain in place.
             */
            if (!agx_opcodes_info[I->op].can_eliminate)
               return NULL;
            break;
         }
      }
   }
   return tile && stop ? tail : NULL;
}
