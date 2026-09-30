/*
 * Copyright 2025 Valve Corporation
 * Copyright 2019-2022 Collabora, Ltd.
 * SPDX-License-Identifier: MIT
 */

#include <stdlib.h>
#include "util/bitset.h"
#include "util/u_debug.h"
#include "agx_compiler.h"
#include "agx_debug.h"
#include "shader_enums.h"

/*
 * Diagnostic (AGX_G17P_NO_ASYNC_DISCARD=1): do not put the register-cache
 * "last use / may evict" hint (.discard) on the sources of an ASYNCHRONOUS
 * instruction.
 *
 * This pass derives .discard purely from post-RA register liveness, which
 * models every instruction as reading its sources at issue. A texture or image
 * operation does not: it is issued, the shader runs on, and the texture unit
 * reads the operands it was not handed a private copy of at some later point,
 * signalling its scoreboard slot when the result lands.
 *
 * Coordinates are immune because they are staged -- explicit copies publish
 * them into the sampler's private bank before the sample issues, so the
 * discard hint applies to the copy, not to the asynchronous read. The shadow
 * COMPARATOR is not staged. It stays a plain GPR operand of the SMP, so it is
 * the one operand whose register the texture unit must still read after the
 * instruction has issued -- and the only one whose eviction the shader can
 * authorise while that read is outstanding.
 *
 * In dEQP-VK.glsl.opaque_type_indexing.sampler.dynamically_uniform.fragment.
 * sampler1darrayshadow all four lookups share one comparator register, so only
 * the LAST sample's comparator is a last use and only it is marked ^r3. Every
 * recorded failure of that test is lookup 3, returning the correct texel of
 * the correct sampler uncompared.
 */
static bool
agx_no_async_discard(void)
{
   static int cached = -1;
   if (cached < 0) {
      const char *v = getenv("AGX_G17P_NO_ASYNC_DISCARD");
      cached = (v && *v && *v != '0') ? 1 : 0;
   }
   return cached;
}

/* Instructions whose operands the hardware reads after issue, rather than
 * latching them when the instruction is dispatched. */
static bool
agx_reads_sources_late(const agx_instr *I)
{
   switch (I->op) {
   case AGX_OPCODE_TEXTURE_SAMPLE:
   case AGX_OPCODE_TEXTURE_LOAD:
   case AGX_OPCODE_IMAGE_LOAD:
   case AGX_OPCODE_IMAGE_WRITE:
      return true;
   default:
      return false;
   }
}

/*
 * AGX_G17P_PHYSICAL_DISCARD (v52b, default off): liveness over the physical
 * CFG the exec-masked hardware actually walks, so divergent blocks can carry
 * discard hints too.
 *
 * Inside divergent control flow the lanes of one register can hold values
 * with different lifetimes. Two things differ from the logical CFG:
 *  - a divergent block also falls through to the next block in layout order
 *    (the else after a then, the rest of a loop body after a break), because
 *    lanes that left logically are only masked off; and
 *  - a write in a divergent block may be partial (masked), so it does not end
 *    the previous value's lifetime for the inactive lanes.
 * With both, "dead after this read" holds for every lane, which is what a
 * discard hint asserts. Uniform blocks keep the exact logical analysis.
 */
static bool
agx_physical_discard(const agx_context *ctx)
{
   return ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          (agx_get_alu_flags() & AGX_ALU_G17P_PHYSICAL_DISCARD);
}

static agx_block *
agx_physical_fallthrough(agx_context *ctx, agx_block *blk)
{
   if (!blk->cf_divergent || blk->link.next == &ctx->blocks)
      return NULL;
   return agx_next_block(blk);
}

static void
postra_liveness_ins_mode(BITSET_WORD *live, agx_instr *I, bool masked_defs)
{
   agx_foreach_reg_dest(I, d) {
      unsigned reg = I->dest[d].value;
      if (!masked_defs)
         BITSET_CLEAR_COUNT(live, reg, agx_index_size_16(I->dest[d]));
   }

   agx_foreach_reg_src(I, s) {
      unsigned reg = I->src[s].value;
      BITSET_SET_COUNT(live, reg, agx_index_size_16(I->src[s]));
   }
}

static void
postra_liveness_ins(BITSET_WORD *live, agx_instr *I)
{
   agx_foreach_reg_dest(I, d) {
      unsigned reg = I->dest[d].value;
      BITSET_CLEAR_COUNT(live, reg, agx_index_size_16(I->dest[d]));
   }

   agx_foreach_reg_src(I, s) {
      unsigned reg = I->src[s].value;
      BITSET_SET_COUNT(live, reg, agx_index_size_16(I->src[s]));
   }
}

/*
 * Globally, liveness analysis uses a fixed-point algorithm based on a
 * worklist. We initialize a work list with the exit block. We iterate the work
 * list to compute live_in from live_out for each block on the work list,
 * adding the predecessors of the block to the work list if we made progress.
 */
static void
postra_liveness(agx_context *ctx)
{
   const bool physical = agx_physical_discard(ctx);
   u_worklist worklist;
   agx_worklist_init(ctx, &worklist);

   agx_foreach_block(ctx, block) {
      BITSET_ZERO(block->reg_live_in);
      BITSET_ZERO(block->reg_live_out);

      agx_worklist_push_tail(&worklist, block);
   }

   while (!u_worklist_is_empty(&worklist)) {
      /* Pop off in reverse order since liveness is backwards */
      agx_block *blk = agx_worklist_pop_tail(&worklist);

      /* Calculate liveness locally */
      agx_foreach_successor(blk, succ) {
         BITSET_OR(blk->reg_live_out, blk->reg_live_out, succ->reg_live_in);
      }
      agx_block *fall = physical ? agx_physical_fallthrough(ctx, blk) : NULL;
      if (fall)
         BITSET_OR(blk->reg_live_out, blk->reg_live_out, fall->reg_live_in);

      BITSET_DECLARE(live, AGX_NUM_REGS);
      memcpy(live, blk->reg_live_out, sizeof(live));

      agx_foreach_instr_in_block_rev(blk, ins) {
         if (physical)
            postra_liveness_ins_mode(live, ins, blk->cf_divergent);
         else
            postra_liveness_ins(live, ins);
      }

      if (BITSET_EQUAL(blk->reg_live_in, live))
         continue;

      /* We made progress, so we need to reprocess the predecessors */
      memcpy(blk->reg_live_in, live, sizeof(live));
      agx_foreach_predecessor(blk, pred) {
         agx_worklist_push_head(&worklist, *pred);
      }
      if (physical && blk->link.prev != &ctx->blocks) {
         agx_block *prev = agx_prev_block(blk);
         if (prev->cf_divergent)
            agx_worklist_push_head(&worklist, prev);
      }
   }

   u_worklist_fini(&worklist);
}

static bool
writes_reg(const agx_instr *I, unsigned reg)
{
   agx_foreach_reg_dest(I, d) {
      unsigned count = agx_index_size_16(I->dest[d]);

      if (reg >= I->dest[d].value && (reg - I->dest[d].value) < count)
         return true;
   }

   return false;
}

/*
 * Mark last-use sources to allow the hardware to discard from the register
 * cache. Last use information follows immediately from (post-RA) liveness
 * analysis: a register is dead immediately after its last use.
 *
 * Mark cache hints on sources/destinations to encourage the hardware to make
 * better use of the register cache. This is a simple local analysis.
 */
void
agx_opt_register_cache(agx_context *ctx)
{
   const bool physical = agx_physical_discard(ctx);

   /* Analyze the shader globally */
   postra_liveness(ctx);

   agx_foreach_block(ctx, block) {
      /* Live-set at each point in the program */
      BITSET_DECLARE(live, AGX_NUM_REGS);
      memcpy(live, block->reg_live_out, sizeof(live));

      /* Set of registers read "soon" by an ALU instruction. These are
       * candidates for the .cache bit.
       */
      BITSET_DECLARE(alu_reads, AGX_NUM_REGS) = {0};

      agx_foreach_instr_in_block_rev(block, I) {
         agx_foreach_reg_dest(I, d) {
            uint64_t reg = I->dest[d].value;
            unsigned nr = agx_index_size_16(I->dest[d]);

            I->dest[d].cache = BITSET_TEST(alu_reads, I->dest[d].value);
            BITSET_CLEAR_COUNT(alu_reads, reg, nr);
         }

         agx_foreach_reg_src(I, s) {
            I->src[s].cache = BITSET_TEST(alu_reads, I->src[s].value);
         }

         agx_foreach_reg_src(I, s) {
            uint64_t reg = I->src[s].value;
            unsigned nr = agx_index_size_16(I->src[s]);

            /* If the register dead after this instruction, it's the last use.
             * That includes if the register is overwritten this cycle, but that
             * won't show up in the liveness analysis.
             */
            bool lu = !BITSET_TEST_COUNT(live, reg, nr) ||
                      (writes_reg(I, reg) && !(physical && block->cf_divergent));

            /* Handling divergent blocks would require physical CFG awareness.
             * Just bail for now, skipping this pass won't affect correctness.
             * Subgroup-uniform ifs/loops have no physical divergence.
             * AGX_G17P_PHYSICAL_DISCARD computes that physical liveness.
             */
            I->src[s].discard = lu && (physical || !block->cf_divergent) &&
                                !(agx_no_async_discard() &&
                                  agx_reads_sources_late(I));

            /* Mark any source read by an ALU instruction in the same block as
             * wanting a .cache hint. This is better than just marking
             * everything, although it overly hints for very long blocks and
             * underhints for registers used across block boundaries. It's
             * probably good enough, though, and it's not clear how to do much
             * better given our limited understanding of the hardware.
             */
            if (agx_is_alu(I))
               BITSET_SET_COUNT(alu_reads, reg, nr);

            assert(!(I->src[s].discard && I->src[s].cache));
         }

         if (physical)
            postra_liveness_ins_mode(live, I, block->cf_divergent);
         else
            postra_liveness_ins(live, I);
      }
   }

   /* AGX_G17P_EXIT_COMPARE (v53-c3): Apple's G17P fragment loop test keeps
    * its counter operand in the register cache (fs-apple.bin +0xbc, f11 on
    * `icmp i, u6`). The block-local analysis above cannot: the counter's next
    * reads are at the top of the loop, across the back edge. Cache a loop
    * compare's register source when the loop header reads it with an ALU
    * instruction before writing it, unless this read is its last use.
    * AGX_G17P_COMPARE_CACHE=0 disables only this hint (diagnostic). */
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B ||
       !(agx_get_codegen_flags() & AGX_CODEGEN_G17P_EXIT_COMPARE) ||
       !debug_get_bool_option("AGX_G17P_COMPARE_CACHE", true))
      return;

   agx_foreach_block(ctx, block) {
      agx_block *header = NULL;
      agx_foreach_successor(block, succ) {
         if (succ->loop_header && succ->index <= block->index)
            header = succ;
      }
      if (!header)
         continue;

      agx_foreach_instr_in_block(block, I) {
         if (I->op != AGX_OPCODE_IF_ICMP && I->op != AGX_OPCODE_WHILE_ICMP &&
             I->op != AGX_OPCODE_BREAK_IF_ICMP)
            continue;
         agx_foreach_reg_src(I, s) {
            if (I->src[s].discard || I->src[s].cache ||
                I->src[s].type != AGX_INDEX_REGISTER)
               continue;
            const unsigned reg = I->src[s].value;
            const unsigned nr = agx_index_size_16(I->src[s]);
            bool read = false;
            agx_foreach_instr_in_block(header, H) {
               bool reads = false;
               agx_foreach_reg_src(H, hs) {
                  const unsigned r = H->src[hs].value;
                  const unsigned rn = agx_index_size_16(H->src[hs]);
                  reads |= r < reg + nr && reg < r + rn;
               }
               if (reads) {
                  read = agx_is_alu(H);
                  break;
               }
               if (writes_reg(H, reg))
                  break;
            }
            I->src[s].cache = read;
         }
      }
   }
}
