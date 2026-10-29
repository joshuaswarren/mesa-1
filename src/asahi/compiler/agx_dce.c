/*
 * Copyright 2021 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "agx_compiler.h"

/**
 * SSA-based scalar dead code elimination.
 * This pass assumes that no loop header phis are dead.
 */
void
agx_dce(agx_context *ctx, bool partial)
{
   BITSET_WORD *seen = BITSET_CALLOC(ctx->alloc);

   agx_foreach_block(ctx, block) {
      if (block->loop_header) {
         agx_foreach_phi_in_block(block, I) {
            agx_foreach_ssa_src(I, s) {
               BITSET_SET(seen, I->src[s].value);
            }
         }
      }
   }

   agx_foreach_block_rev(ctx, block) {
      agx_foreach_instr_in_block_safe_rev(block, I) {
         if (block->loop_header && I->op == AGX_OPCODE_PHI)
            break;

         /* RT-completion loads wait for a traversal producer even when their
          * byte result is unread. Ordinary dead SCS metadata loads remain
          * removable. Keep a real destination, as the native load writes it. */
         const bool rt_completion =
            (I->op == AGX_OPCODE_SCS_LOAD_REGISTER_INDEXED_AGX3 ||
             I->op == AGX_OPCODE_SCS_LOAD_REGISTER_BASE_AGX3) &&
            I->scs_rt_completion;
         bool needed = false;

         agx_foreach_ssa_dest(I, d) {
            /* Eliminate destinations that are never read, as RA needs to
             * handle them specially. Visible only for instructions that write
             * multiple destinations (splits) or that write a destination but
             * cannot be DCE'd (atomics).
             */
            if (BITSET_TEST(seen, I->dest[d].value)) {
               needed = true;
            } else if (partial && I->op != AGX_OPCODE_ENTER_QUERY_M5 &&
                       !rt_completion) {
               /* The M5 query entry always writes its result register, so
                * it must keep a real destination even when it is unread.
                */
               I->dest[d] = agx_null();
            }
         }

         if (!needed && agx_opcodes_info[I->op].can_eliminate &&
             !I->agx3_tex_coord_stage && !rt_completion) {
            agx_remove_instruction(I);
         } else {
            agx_foreach_ssa_src(I, s) {
               BITSET_SET(seen, I->src[s].value);
            }
         }
      }
   }

   free(seen);
}
