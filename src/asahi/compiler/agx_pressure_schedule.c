/*
 * Copyright 2023 Alyssa Rosenzweig
 * Copyright 2022 Collabora Ltd.
 * SPDX-License-Identifier: MIT
 */

/* Bottom-up local scheduler to reduce register pressure */

#include "util/dag.h"
#include "util/sparse_bitset.h"
#include "agx_compiler.h"
#include "agx_opcodes.h"

struct sched_ctx {
   /* Dependency graph */
   struct dag *dag;

   /* Live set */
   struct u_sparse_bitset live;
};

struct sched_node {
   struct dag_node dag;

   /* Instruction this node represents */
   agx_instr *instr;
};

static void
add_dep(struct sched_node *a, struct sched_node *b)
{
   assert(a != b && "no self-dependencies");

   if (a && b)
      dag_add_edge(&a->dag, &b->dag, 0);
}

static void
serialize(struct sched_node *a, struct sched_node **b)
{
   add_dep(a, *b);
   *b = a;
}

static struct dag *
create_dag(agx_context *ctx, agx_block *block, void *memctx)
{
   struct dag *dag = dag_create(ctx);

   struct sched_node **last_write =
      calloc(ctx->alloc, sizeof(struct sched_node *));
   struct sched_node *coverage = NULL;
   struct sched_node *preload = NULL;

   /* Last memory load, to serialize stores against */
   struct sched_node *memory_load = NULL;

   /* Last memory store, to serialize loads and stores against */
   struct sched_node *memory_store = NULL;

   agx_foreach_instr_in_block(block, I) {
      /* Don't touch control flow */
      if (instr_after_logical_end(I))
         break;

      struct sched_node *node = rzalloc(memctx, struct sched_node);
      node->instr = I;
      dag_init_node(dag, &node->dag);

      /* Reads depend on writes, no other hazards in SSA */
      agx_foreach_ssa_src(I, s) {
         add_dep(node, last_write[I->src[s].value]);
      }

      agx_foreach_ssa_dest(I, d) {
         assert(I->dest[d].value < ctx->alloc);
         last_write[I->dest[d].value] = node;
      }

      /* Classify the instruction and add dependencies according to the class */
      enum agx_schedule_class dep = agx_opcodes_info[I->op].schedule_class;
      assert(dep != AGX_SCHEDULE_CLASS_INVALID && "invalid instruction seen");

      bool barrier = dep == AGX_SCHEDULE_CLASS_BARRIER;
      bool discards =
         I->op == AGX_OPCODE_SAMPLE_MASK || I->op == AGX_OPCODE_ZS_EMIT;

      if (dep == AGX_SCHEDULE_CLASS_STORE)
         add_dep(node, memory_load);
      else if (dep == AGX_SCHEDULE_CLASS_ATOMIC || barrier)
         serialize(node, &memory_load);

      if (dep == AGX_SCHEDULE_CLASS_LOAD || dep == AGX_SCHEDULE_CLASS_STORE ||
          dep == AGX_SCHEDULE_CLASS_ATOMIC || barrier)
         serialize(node, &memory_store);

      if (dep == AGX_SCHEDULE_CLASS_COVERAGE || barrier)
         serialize(node, &coverage);

      /* Make sure side effects happen before a discard */
      if (discards)
         add_dep(node, memory_store);

      if (dep == AGX_SCHEDULE_CLASS_PRELOAD)
         serialize(node, &preload);
      else
         add_dep(node, preload);
   }

   free(last_write);

   return dag;
}

/*
 * Calculate the change in register pressure from scheduling a given
 * instruction. Equivalently, calculate the difference in the number of live
 * registers before and after the instruction, given the live set after the
 * instruction. This calculation follows immediately from the dataflow
 * definition of liveness:
 *
 *      live_in = (live_out - KILL) + GEN
 */
static signed
calculate_pressure_delta(agx_instr *I, struct u_sparse_bitset *live)
{
   signed delta = 0;

   /* Destinations must be unique */
   agx_foreach_ssa_dest(I, d) {
      if (u_sparse_bitset_test(live, I->dest[d].value))
         delta -= agx_index_size_16(I->dest[d]);
   }

   agx_foreach_ssa_src(I, src) {
      /* Filter duplicates */
      bool dupe = false;

      for (unsigned i = 0; i < src; ++i) {
         if (agx_is_equiv(I->src[i], I->src[src])) {
            dupe = true;
            break;
         }
      }

      if (!dupe && !u_sparse_bitset_test(live, I->src[src].value))
         delta += agx_index_size_16(I->src[src]);
   }

   return delta;
}

/*
 * Choose the next instruction, bottom-up. For now we use a simple greedy
 * heuristic: choose the instruction that has the best effect on liveness, while
 * hoisting sample_mask.
 */
static struct sched_node *
choose_instr(struct sched_ctx *s)
{
   int32_t min_delta = INT32_MAX;
   struct sched_node *best = NULL;

   list_for_each_entry(struct sched_node, n, &s->dag->heads, dag.link) {
      /* Heuristic: hoist sample_mask/zs_emit. This allows depth/stencil tests
       * to run earlier, and potentially to discard the entire quad invocation
       * earlier, reducing how much redundant fragment shader we run.
       *
       * Since we schedule backwards, we make that happen by only choosing
       * sample_mask when all other instructions have been exhausted.
       */
      if (n->instr->op == AGX_OPCODE_SAMPLE_MASK ||
          n->instr->op == AGX_OPCODE_ZS_EMIT) {

         if (!best) {
            best = n;
            assert(min_delta == INT32_MAX);
         }

         continue;
      }

      /* Heuristic: sink wait_pix to increase parallelism. Since wait_pix does
       * not read or write registers, this has no effect on pressure.
       */
      if (n->instr->op == AGX_OPCODE_WAIT_PIX)
         return n;

      int32_t delta = calculate_pressure_delta(n->instr, &s->live);

      if (delta < min_delta) {
         best = n;
         min_delta = delta;
      }
   }

   return best;
}

static void
pressure_schedule_block(agx_context *ctx, agx_block *block, struct sched_ctx *s)
{
   /* off by a constant, that's ok */
   signed pressure = 0;
   signed orig_max_pressure = 0;
   unsigned nr_ins = 0;

   u_sparse_bitset_dup(&s->live, &block->live_out);

   agx_foreach_instr_in_block_rev(block, I) {
      pressure += calculate_pressure_delta(I, &s->live);
      orig_max_pressure = MAX2(pressure, orig_max_pressure);
      agx_liveness_ins_update(&s->live, I);
      nr_ins++;
   }

   u_sparse_bitset_dup(&s->live, &block->live_out);

   /* off by a constant, that's ok */
   signed max_pressure = 0;
   pressure = 0;

   struct sched_node **schedule = calloc(nr_ins, sizeof(struct sched_node *));
   nr_ins = 0;

   while (!list_is_empty(&s->dag->heads)) {
      struct sched_node *node = choose_instr(s);
      pressure += calculate_pressure_delta(node->instr, &s->live);
      max_pressure = MAX2(pressure, max_pressure);
      dag_prune_head(s->dag, &node->dag);

      schedule[nr_ins++] = node;
      agx_liveness_ins_update(&s->live, node->instr);
   }

   /* Bail if it looks like it's worse */
   if (max_pressure >= orig_max_pressure) {
      free(schedule);
      return;
   }

   /* Apply the schedule */
   for (unsigned i = 0; i < nr_ins; ++i) {
      agx_remove_instruction(schedule[i]->instr);
      list_add(&schedule[i]->instr->link, &block->instructions);
   }

   free(schedule);
}

void
agx_pressure_schedule(agx_context *ctx)
{
   agx_compute_liveness(ctx);
   void *memctx = ralloc_context(ctx);

   agx_foreach_block(ctx, block) {
      struct sched_ctx sctx = {.dag = create_dag(ctx, block, memctx)};
      u_sparse_bitset_init(&sctx.live, ctx->alloc, memctx);

      pressure_schedule_block(ctx, block, &sctx);
   }

   /* Clean up after liveness analysis */
   agx_foreach_instr_global(ctx, I) {
      agx_foreach_ssa_src(I, s)
         I->src[s].kill = false;
   }

   ralloc_free(memctx);
}

/*
 * Latency hoist (compute): move each non-coherent device_load
 * up past ALU, threadgroup barriers and workgroup-memory accesses, to just
 * after the last instruction that produces its sources, but never more than
 * HOIST_WINDOW instructions. The load then overlaps that work instead of
 * being issued right before its first use. Stops at anything that could
 * order device memory: device stores, atomics, memory barriers, coherent
 * loads, image ops, control flow.
 */
#define HOIST_WINDOW 64

/*
 * A uniform_store writes at most four 16-bit uniform registers starting at its
 * index and nothing else. A load may move above it unless one of the load's
 * uniform sources overlaps [index, index + 16). The window is wider than the
 * store, which keeps the check conservative. An index that is not an
 * immediate is never crossed.
 */
static bool
hoist_uniform_store_conflict(const agx_instr *store, const agx_instr *load)
{
   if (store->src[1].type != AGX_INDEX_IMMEDIATE)
      return true;

   unsigned lo = store->src[1].value, hi = lo + 16;

   agx_foreach_src(load, s) {
      agx_index src = load->src[s];
      if (src.type != AGX_INDEX_UNIFORM)
         continue;

      unsigned src_lo = src.value, src_hi = src_lo + agx_index_size_16(src);
      if (src_lo < hi && lo < src_hi)
         return true;
   }

   return false;
}

static bool
hoist_can_cross(const agx_instr *P, const agx_instr *load)
{
   if (P->op == AGX_OPCODE_THREADGROUP_BARRIER ||
       P->op == AGX_OPCODE_LOCAL_LOAD || P->op == AGX_OPCODE_LOCAL_STORE)
      return true;

   if (P->op == AGX_OPCODE_DEVICE_LOAD)
      return !P->coherent;

   if (P->op == AGX_OPCODE_UNIFORM_STORE)
      return !hoist_uniform_store_conflict(P, load);

   /* Anything else must be pure: free of side effects and reorderable. The
    * default schedule class is none, so the class alone also admits
    * doorbell, stack_map and similar.
    */
   const struct agx_opcode_info *info = &agx_opcodes_info[P->op];
   return info->schedule_class == AGX_SCHEDULE_CLASS_NONE &&
          info->can_reorder && info->can_eliminate;
}

void
agx_hoist_loads(agx_context *ctx)
{
   agx_foreach_block(ctx, block) {
      agx_foreach_instr_in_block_safe(block, I) {
         if (instr_after_logical_end(I))
            break;

         if (I->op != AGX_OPCODE_DEVICE_LOAD || I->coherent)
            continue;

         agx_instr *target = NULL;
         unsigned n = 0;

         for (agx_instr *P = list_entry(I->link.prev, agx_instr, link);
              &P->link != &block->instructions && n < HOIST_WINDOW;
              P = list_entry(P->link.prev, agx_instr, link), ++n) {

            if (!hoist_can_cross(P, I))
               break;

            bool dep = false;
            agx_foreach_ssa_dest(P, d) {
               agx_foreach_ssa_src(I, s) {
                  if (I->src[s].value == P->dest[d].value)
                     dep = true;
               }
            }
            if (dep)
               break;

            target = P;
         }

         if (target) {
            agx_remove_instruction(I);
            list_addtail(&I->link, &target->link);
         }
      }
   }
}
