/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_nir.h"
#include "compiler/nir/nir.h"

/*
 * Sink the ALU consumers of a loop body's texture results to the latest
 * top-level block of that body that still dominates every use.
 *
 * Mesa's order consumes each texture result right after its sample, so a
 * loop whose body samples several times completes those samples one at a
 * time: twelve serial texture round trips per iteration in the robustness2
 * sampled_image 3D vertex loop. Apple's compiler for the same straight-line
 * loop (native G17G witness) issues all twelve
 * SMP in source order, the three conditional ones inside their own
 * cndst/cndend, and places every compare and accumulate of the twelve
 * results after the last SMP, so the iteration waits for the samples once.
 *
 * nir_opt_sink leaves general ALU alone and nir_opt_move only reorders
 * within a block; the backend pressure and ILP schedules are per block too,
 * and a sample under a condition (c != 0 here) splits the body into several
 * blocks. Moving a pure ALU instruction to a block its definition dominates
 * and that dominates all of its uses changes neither its value nor the SSA
 * dominance of the program; it may only run less often (it no longer runs on
 * paths that leave the body early). Texture instructions themselves never
 * move, so no sample is speculated.
 *
 * The sunk results stay live until the end of the body, so the flagged
 * sample results are bounded per loop by AGX_NIR_LOOP_SAMPLE_SINK_HALVES
 * (twelve 32-bit vec4 results, the witness loop); consumers of later samples
 * keep their place.
 */

#define AGX_NIR_LOOP_SAMPLE_SINK_HALVES 96

/* pass_flags bit: the instruction is, or depends on, a sample of this loop
 * whose consumers may sink. */
#define SAMPLE_DEPENDENT 1

static unsigned
def_halves(const nir_def *def)
{
   return def->num_components * DIV_ROUND_UP(def->bit_size, 16);
}

/* nir_foreach_src callback: keep going while the source is independent, so
 * the walk returns false exactly when some source is sample dependent. */
static bool
src_independent(nir_src *src, void *data)
{
   return (nir_def_instr(src->ssa)->pass_flags & SAMPLE_DEPENDENT) == 0;
}

/* The top-level block of the loop body containing (or preceding the nested
 * control flow containing) block b, or NULL when b is outside the loop. */
static nir_block *
loop_top_level_block(nir_loop *loop, nir_block *b)
{
   nir_cf_node *node = &b->cf_node;
   while (node->parent != &loop->cf_node) {
      node = node->parent;
      if (!node)
         return NULL;
   }
   if (node->type == nir_cf_node_block)
      return nir_cf_node_as_block(node);
   /* Nested if or loop: the top-level block right before it. */
   nir_cf_node *prev = nir_cf_node_prev(node);
   assert(prev && prev->type == nir_cf_node_block);
   return nir_cf_node_as_block(prev);
}

static bool
sink_loop(nir_loop *loop)
{
   if (nir_loop_has_continue_construct(loop))
      return false;

   /* Mark the body's samples, within the result budget, and everything
    * computed from them. Loop-carried values reach their consumers through
    * header phis, so iterate to a fixed point. */
   unsigned budget = AGX_NIR_LOOP_SAMPLE_SINK_HALVES;
   nir_foreach_block_in_cf_node(block, &loop->cf_node) {
      nir_foreach_instr(instr, block)
         instr->pass_flags = 0;
   }
   bool any_sample = false;
   nir_foreach_block_in_cf_node(block, &loop->cf_node) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_tex)
            continue;
         nir_tex_instr *tex = nir_instr_as_tex(instr);
         const unsigned halves = def_halves(&tex->def);
         if (halves > budget)
            continue;
         budget -= halves;
         instr->pass_flags = SAMPLE_DEPENDENT;
         any_sample = true;
      }
   }
   if (!any_sample)
      return false;

   bool changed;
   do {
      changed = false;
      nir_foreach_block_in_cf_node(block, &loop->cf_node) {
         nir_foreach_instr(instr, block) {
            if (instr->pass_flags & SAMPLE_DEPENDENT)
               continue;
            if (instr->type != nir_instr_type_alu &&
                instr->type != nir_instr_type_phi)
               continue;
            if (!nir_foreach_src(instr, src_independent, NULL)) {
               instr->pass_flags = SAMPLE_DEPENDENT;
               changed = true;
            }
         }
      }
   } while (changed);

   /* Consumers first: visiting in reverse lets a producer follow the uses
    * that already moved. */
   bool progress = false;
   nir_foreach_block_in_cf_node_reverse_safe(block, &loop->cf_node) {
      if (block->cf_node.parent != &loop->cf_node)
         continue;
      nir_foreach_instr_reverse_safe(instr, block) {
         if (instr->type != nir_instr_type_alu ||
             !(instr->pass_flags & SAMPLE_DEPENDENT))
            continue;
         nir_alu_instr *alu = nir_instr_as_alu(instr);

         nir_block *lca = NULL;
         nir_foreach_use_including_if(use, &alu->def)
            lca = nir_dominance_lca(lca, nir_src_get_block(use));
         if (!lca)
            continue;

         nir_block *target = loop_top_level_block(loop, lca);
         if (!target || target->index <= block->index)
            continue;
         assert(nir_block_dominates(block, target));

         nir_instr_remove(instr);
         nir_instr_insert(nir_after_phis(target), instr);
         progress = true;
      }
   }
   return progress;
}

static bool
sink_cf_list(struct exec_list *list)
{
   bool progress = false;
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         progress |= sink_cf_list(&nif->then_list);
         progress |= sink_cf_list(&nif->else_list);
      } else if (node->type == nir_cf_node_loop) {
         nir_loop *loop = nir_cf_node_as_loop(node);
         progress |= sink_cf_list(&loop->body);
         progress |= sink_loop(loop);
      }
   }
   return progress;
}

bool
agx_nir_opt_sink_loop_sample_uses(nir_shader *shader)
{
   bool progress = false;
   nir_foreach_function_impl(impl, shader) {
      nir_metadata_require(impl, nir_metadata_block_index |
                                    nir_metadata_dominance |
                                    nir_metadata_dominance_lca);
      const bool impl_progress = sink_cf_list(&impl->body);
      progress |= nir_progress(impl_progress, impl, nir_metadata_control_flow);
   }
   return progress;
}

/*
 * Hoist a loop body's unconditional samples above the conditional blocks
 * before them.
 *
 * The M5 backend retires every outstanding scoreboard slot before a
 * control-flow instruction (agx3_try_control_flow), so each conditional
 * block of a loop body closes the window in which samples stay in flight.
 * The robustness2 vertex loops have a conditional texel fetch after each
 * axis's first fetch; even with every consumer sunk to the end of the body
 * their eleven or twelve samples completed in seven serial round trips per
 * iteration. The native compiler issues the unconditional samples of such a
 * body back to back.
 *
 * A sample in a later top-level block of the body runs exactly when the
 * top-level blocks before it run, as long as the conditional blocks in
 * between hold no jump, no loop and nothing with side effects. Moving it,
 * together with the pure computation of its sources from those later
 * blocks, to the end of the earliest such block changes neither which
 * samples run nor their values, and no sample is speculated. Conditional
 * blocks left adjacent with the same condition are then merged by
 * nir_opt_if_merge, after which the merged block's duplicate fetches CSE.
 *
 * Only samples whose consumers agx_nir_opt_sink_loop_sample_uses sinks
 * (the same per-loop budget) move, so the live sample results stay bounded
 * by AGX_NIR_LOOP_SAMPLE_SINK_HALVES.
 */

/* pass_flags bits for the hoist. */
#define HOIST_ELIGIBLE 1
#define HOIST_MOVE     2

/* Bound on the source chain walked for one sample. */
#define HOIST_MAX_DEPTH 32

/* An instruction without side effects whose value does not depend on its
 * position among the body's samples. */
static bool
instr_reorderable(nir_instr *instr)
{
   switch (instr->type) {
   case nir_instr_type_alu:
   case nir_instr_type_load_const:
   case nir_instr_type_undef:
   case nir_instr_type_tex:
      return true;
   case nir_instr_type_intrinsic:
      return nir_intrinsic_can_reorder(nir_instr_as_intrinsic(instr));
   default:
      return false;
   }
}

/* Whether a sample may move across this control flow: only if-statements
 * whose arms hold no jump, loop or side effect. */
static bool
cf_list_reorderable(struct exec_list *list)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      switch (node->type) {
      case nir_cf_node_block:
         nir_foreach_instr(instr, nir_cf_node_as_block(node)) {
            if (instr->type != nir_instr_type_phi && !instr_reorderable(instr))
               return false;
         }
         break;
      case nir_cf_node_if: {
         nir_if *nif = nir_cf_node_as_if(node);
         if (!cf_list_reorderable(&nif->then_list) ||
             !cf_list_reorderable(&nif->else_list))
            return false;
         break;
      }
      default:
         return false;
      }
   }
   return true;
}

static bool
cf_node_reorderable(nir_cf_node *node)
{
   if (!node || node->type != nir_cf_node_if)
      return false;
   nir_if *nif = nir_cf_node_as_if(node);
   return cf_list_reorderable(&nif->then_list) &&
          cf_list_reorderable(&nif->else_list);
}

struct hoist_state {
   nir_loop *loop;
   nir_block *target;
   unsigned depth;
};

static bool hoist_src(nir_src *src, void *data);

/* Whether def is available at the end of target, or is a pure value of a
 * later top-level block of the body whose sources are, flagging the
 * instructions that have to move. Samples never join a chain: a sample
 * feeding another only counts once it has been hoisted itself. */
static bool
hoist_def(nir_def *def, struct hoist_state *st)
{
   nir_instr *instr = nir_def_instr(def);
   if (nir_block_dominates(instr->block, st->target) ||
       (instr->pass_flags & HOIST_MOVE))
      return true;
   if (instr->block->cf_node.parent != &st->loop->cf_node ||
       instr->type == nir_instr_type_phi ||
       instr->type == nir_instr_type_tex || !instr_reorderable(instr) ||
       st->depth >= HOIST_MAX_DEPTH)
      return false;
   instr->pass_flags |= HOIST_MOVE;
   st->depth++;
   const bool ok = nir_foreach_src(instr, hoist_src, st);
   st->depth--;
   return ok;
}

static bool
hoist_src(nir_src *src, void *data)
{
   return hoist_def(src->ssa, data);
}

/* Clear the move flags of the top-level blocks after target up to and
 * including tex's block, the only instructions hoist_def flags. */
static void
clear_moves(nir_block *target, nir_instr *tex)
{
   for (nir_cf_node *node = nir_cf_node_next(&target->cf_node);
        node != nir_cf_node_next(&tex->block->cf_node);
        node = nir_cf_node_next(node)) {
      if (node->type != nir_cf_node_block)
         continue;
      nir_foreach_instr(instr, nir_cf_node_as_block(node))
         instr->pass_flags &= ~HOIST_MOVE;
   }
}

static bool
hoist_sample(nir_loop *loop, nir_instr *tex)
{
   /* Walk back over reorderable conditional blocks while the sample's
    * sources stay available. */
   nir_block *target = NULL;
   nir_cf_node *prev = nir_cf_node_prev(&tex->block->cf_node);
   while (cf_node_reorderable(prev)) {
      nir_block *candidate = nir_cf_node_as_block(nir_cf_node_prev(prev));
      struct hoist_state st = { .loop = loop, .target = candidate };
      const bool ok = nir_foreach_src(tex, hoist_src, &st);
      clear_moves(candidate, tex);
      if (!ok)
         break;
      target = candidate;
      prev = nir_cf_node_prev(&candidate->cf_node);
   }
   if (!target)
      return false;

   struct hoist_state st = { .loop = loop, .target = target };
   ASSERTED bool ok = nir_foreach_src(tex, hoist_src, &st);
   assert(ok);
   tex->pass_flags |= HOIST_MOVE;

   /* Append the flagged instructions to the target in their original order,
    * which keeps every definition ahead of its uses. */
   nir_cf_node *end = nir_cf_node_next(&tex->block->cf_node);
   for (nir_cf_node *node = nir_cf_node_next(&target->cf_node); node != end;
        node = nir_cf_node_next(node)) {
      if (node->type != nir_cf_node_block)
         continue;
      nir_foreach_instr_safe(instr, nir_cf_node_as_block(node)) {
         if (!(instr->pass_flags & HOIST_MOVE))
            continue;
         instr->pass_flags &= ~HOIST_MOVE;
         nir_instr_remove(instr);
         nir_instr_insert(nir_after_block(target), instr);
      }
   }
   return true;
}

static bool
hoist_loop(nir_loop *loop)
{
   if (nir_loop_has_continue_construct(loop))
      return false;

   /* The samples agx_nir_opt_sink_loop_sample_uses sinks consumers of. */
   unsigned budget = AGX_NIR_LOOP_SAMPLE_SINK_HALVES;
   nir_foreach_block_in_cf_node(block, &loop->cf_node) {
      nir_foreach_instr(instr, block) {
         instr->pass_flags = 0;
         if (instr->type != nir_instr_type_tex)
            continue;
         const unsigned halves = def_halves(&nir_instr_as_tex(instr)->def);
         if (halves > budget)
            continue;
         budget -= halves;
         instr->pass_flags = HOIST_ELIGIBLE;
      }
   }

   bool progress = false;
   foreach_list_typed(nir_cf_node, node, node, &loop->body) {
      if (node->type != nir_cf_node_block)
         continue;
      nir_foreach_instr_safe(instr, nir_cf_node_as_block(node)) {
         if (instr->type == nir_instr_type_tex &&
             (instr->pass_flags & HOIST_ELIGIBLE))
            progress |= hoist_sample(loop, instr);
      }
   }
   return progress;
}

static bool
hoist_cf_list(struct exec_list *list)
{
   bool progress = false;
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         progress |= hoist_cf_list(&nif->then_list);
         progress |= hoist_cf_list(&nif->else_list);
      } else if (node->type == nir_cf_node_loop) {
         nir_loop *loop = nir_cf_node_as_loop(node);
         progress |= hoist_cf_list(&loop->body);
         progress |= hoist_loop(loop);
      }
   }
   return progress;
}

bool
agx_nir_opt_hoist_loop_samples(nir_shader *shader)
{
   bool progress = false;
   nir_foreach_function_impl(impl, shader) {
      nir_metadata_require(impl, nir_metadata_block_index |
                                    nir_metadata_dominance);
      const bool impl_progress = hoist_cf_list(&impl->body);
      progress |= nir_progress(impl_progress, impl, nir_metadata_control_flow);
   }
   return progress;
}
