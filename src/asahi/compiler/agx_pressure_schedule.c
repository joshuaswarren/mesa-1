/*
 * Copyright 2023 Alyssa Rosenzweig
 * Copyright 2022 Collabora Ltd.
 * SPDX-License-Identifier: MIT
 */

/* Bottom-up local scheduler to reduce register pressure */

#include <stdio.h>
#include <string.h>
#include "util/dag.h"
#include "util/u_debug.h"
#include "agx_debug.h"
#include "util/sparse_bitset.h"
#include "agx_compiler.h"
#include "agx3_fixed_dependencies.h"
#include "agx_opcodes.h"

/* Diagnostic escape hatch for A/B testing the UVS store ordering below. */
DEBUG_GET_ONCE_BOOL_OPTION(agx_stvary_noorder, "HK_STVARY_NOORDER", false)

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

   /* Original position within the block (ILP scheduler only) */
   unsigned index;
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

static bool
native_uniform_write(const agx_instr *I)
{
   return I->op == AGX_OPCODE_UNIFORM_FADD_AGX3 ||
          I->op == AGX_OPCODE_UNIFORM_FMUL_AGX3 ||
          I->op == AGX_OPCODE_UNIFORM_FFMA_AGX3;
}

static void
add_fixed_dependency(void *data, uint32_t before, uint32_t after)
{
   struct sched_node **nodes = data;
   /* The pressure scheduler walks dependents before their dependencies. */
   add_dep(nodes[after], nodes[before]);
}

static bool
uniform_access(struct util_dynarray *accesses, unsigned instruction,
               unsigned first, unsigned count, bool write)
{
   if (!count || first >= AGX_NUM_UNIFORMS ||
       count > AGX_NUM_UNIFORMS - first)
      return false;
   struct agx3_fixed_access access = {
      .instruction = instruction, .first = first, .count = count, .write = write,
   };
   util_dynarray_append(accesses, access);
   return true;
}

static bool
add_uniform_dependencies(struct util_dynarray *nodes, void *memctx)
{
   struct util_dynarray accesses;
   util_dynarray_init(&accesses, memctx);
   unsigned instruction = 0;
   util_dynarray_foreach(nodes, struct sched_node *, node) {
      const agx_instr *I = (*node)->instr;
      agx_foreach_src(I, s) {
         const agx_index src = I->src[s];
         if (src.type == AGX_INDEX_UNIFORM &&
             !uniform_access(&accesses, instruction, src.value,
                              agx_index_size_16(src), false))
            return false;
      }
      if (native_uniform_write(I)) {
         if (I->imm > AGX_NUM_UNIFORMS - 2 ||
             !uniform_access(&accesses, instruction, I->imm, 2, true))
            return false;
      } else if (I->op == AGX_OPCODE_UNIFORM_STORE) {
         if (I->nr_srcs != 2 || I->src[1].type != AGX_INDEX_IMMEDIATE)
            return false;
         const unsigned stride = agx_size_align_16(I->src[0].size);
         const unsigned base = I->src[1].value;
         if (base >= AGX_NUM_UNIFORMS || I->mask > 15)
            return false;
         for (unsigned c = 0; c < 4; ++c) {
            if ((I->mask & (1u << c)) &&
                !uniform_access(&accesses, instruction, base + c * stride,
                                  stride, true))
               return false;
         }
      }
      ++instruction;
   }
   return agx3_fixed_dependencies(accesses.data,
      util_dynarray_num_elements(&accesses, struct agx3_fixed_access),
      AGX_NUM_UNIFORMS, add_fixed_dependency, nodes->data);
}

static struct dag *
create_dag(agx_context *ctx, agx_block *block, void *memctx, bool native_uniforms,
           struct util_dynarray *all_nodes)
{
   struct dag *dag = dag_create(ctx);
   struct util_dynarray nodes;
   util_dynarray_init(&nodes, memctx);

   struct sched_node **last_write =
      calloc(ctx->alloc, sizeof(struct sched_node *));
   struct sched_node *coverage = NULL;
   struct sched_node *preload = NULL;

   /* Last memory load, to serialize stores against */
   struct sched_node *memory_load = NULL;

   /* Last memory store, to serialize loads and stores against */
   struct sched_node *memory_store = NULL;

   /* Last unified-vertex-store write, to keep UVS writeout in program order.
    *
    * AGX3's UVSW is not a plain indexed write.  agx3_pack_binary.c copies
    * each value into the staging file first and then emits the UVSW against
    * it, and outside Apple's batched protocols the staging half and the
    * scoreboard slot a store uses are picked from the store's position in
    * the emission order (`next_uvs_slot` round-robin, `staging_half =
    * 2 * (stage_slot - 1)`).  That fallback is what an unlinked vertex
    * shader takes, because a run-time output index disqualifies every
    * batched form.  Permuting the stores therefore permutes which of them
    * share a staging generation, and the packer's slot bookkeeping does not
    * model those generations, so the resulting hazards are invisible to it:
    * on G17P this rendered a random subset of a tessellated+geometry draw's
    * primitives per run.  st_vary carries no schedule class, so nothing else
    * stops the reorder.  The stores sit at the end of the shader and each
    * one already depends on its own value, so ordering them among themselves
    * costs the scheduler essentially nothing (measured: identical
    * instruction, GPR and spill counts; 10 bytes smaller code on the shader
    * that exposed the bug).  AGX2 addresses the UVS directly and is left
    * alone.
    */
   struct sched_node *vary_store = NULL;
   const enum agx_isa_encoding enc = ctx->key->dev.encoding;
   const bool order_vary_stores =
      agx_encoding_is_agx3(enc) && !debug_get_option_agx_stvary_noorder();

   agx_foreach_instr_in_block(block, I) {
      /* Don't touch control flow */
      if (instr_after_logical_end(I))
         break;

      struct sched_node *node = rzalloc(memctx, struct sched_node);
      node->instr = I;
      dag_init_node(dag, &node->dag);
      if (native_uniforms)
         util_dynarray_append(&nodes, node);
      if (all_nodes) {
         node->index = util_dynarray_num_elements(all_nodes,
                                                  struct sched_node *);
         util_dynarray_append(all_nodes, node);
      }

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

      if (order_vary_stores && I->op == AGX_OPCODE_ST_VARY)
         serialize(node, &vary_store);

      if (dep == AGX_SCHEDULE_CLASS_PRELOAD)
         serialize(node, &preload);
      else
         add_dep(node, preload);
   }

   free(last_write);

   if (native_uniforms && !add_uniform_dependencies(&nodes, memctx))
      return NULL;
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
          n->instr->op == AGX_OPCODE_ZS_EMIT ||
          n->instr->op == AGX_OPCODE_ZS_EMIT_AGX3) {

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

   bool native_uniforms = false;
   agx_foreach_instr_global(ctx, I) {
      native_uniforms |= native_uniform_write(I);
   }

   agx_foreach_block(ctx, block) {
      struct sched_ctx sctx = {
         .dag = create_dag(ctx, block, memctx, native_uniforms, NULL),
      };
      /* Preserve the original order if a complete dependency graph cannot be
       * built (invalid range or workspace allocation failure). */
      if (!sctx.dag)
         continue;
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
 * v47 execution-efficiency candidate (AGX_G17P_ILP_SCHEDULE, default off).
 *
 * Measured on Neo, Mesa's ALU-bound loops (compute128, heavy fragment) retire
 * exactly one instruction per lane per clock on 5 cores x 128 lanes, while
 * Apple's code for the same 13-instruction loop retires 1.13-1.20. Apple's
 * compiler interleaves the independent induction-variable chain (i + c, i++,
 * i == n) between the dependent xorshift steps; Mesa's pressure scheduler
 * leaves the dependent chain back to back. This pass is a latency-aware
 * top-down list scheduler over the same dependency DAG: at each step it
 * issues the ready instruction whose operands are available soonest, breaking
 * ties by the longest remaining critical path. It never adds or removes
 * instructions and keeps every ordering edge of create_dag(). A block whose
 * register pressure would rise by more than AGX_G17P_ILP_SLACK halves keeps
 * its original order.
 *
 * Latencies are a model, not measured G17P numbers: AGX_G17P_ILP_LATENCY
 * (default 4) for ALU results, 4x that for memory, texture and tile loads.
 */
DEBUG_GET_ONCE_NUM_OPTION(agx_g17p_ilp_latency, "AGX_G17P_ILP_LATENCY", 4)
DEBUG_GET_ONCE_NUM_OPTION(agx_g17p_ilp_slack, "AGX_G17P_ILP_SLACK", 2)

static unsigned
ilp_latency(const agx_instr *I, unsigned alu)
{
   switch (I->op) {
   case AGX_OPCODE_DEVICE_LOAD:
   case AGX_OPCODE_LOCAL_LOAD:
   case AGX_OPCODE_STACK_LOAD:
   case AGX_OPCODE_TEXTURE_SAMPLE:
   case AGX_OPCODE_TEXTURE_LOAD:
   case AGX_OPCODE_IMAGE_LOAD:
   case AGX_OPCODE_LD_TILE:
   case AGX_OPCODE_ATOMIC:
   case AGX_OPCODE_LOCAL_ATOMIC:
      return 4 * alu;
   default:
      return alu;
   }
}

static signed
ilp_max_pressure(struct sched_node **order, unsigned n,
                 struct u_sparse_bitset *live_end,
                 struct u_sparse_bitset *scratch)
{
   u_sparse_bitset_dup(scratch, live_end);
   signed pressure = 0, max_pressure = 0;
   for (unsigned i = n; i-- > 0;) {
      pressure += calculate_pressure_delta(order[i]->instr, scratch);
      max_pressure = MAX2(pressure, max_pressure);
      agx_liveness_ins_update(scratch, order[i]->instr);
   }
   return max_pressure;
}

/*
 * AGX_G17P_EXIT_COMPARE (v53-c3): loop-exit compare placement.
 *
 * Lab ladder (v92d): on Apple's own loop, moving the `icmp` next to its
 * `cndlt` keeps only 38% of Apple's advantage over Mesa, and Mesa's order
 * (counter work at the end) keeps 8%; Apple's order puts the counter's
 * update and compare `sll, xor, srl, c+i, i++, icmp, xor, iadd, iadd, xor,
 * x+=, cndlt` exactly five instructions ahead of the CND.
 *
 * Rule, applied to the order the ILP scheduler produced for a loop block
 * whose trailing control flow is an integer compare T:
 *  - the cone of T = the block's instructions T's sources depend on; it must
 *    be small (<= 3) and depend only on values from outside the block (the
 *    counter increment, not the loop's main chain);
 *  - readers = cone-independent instructions with no in-block dependency
 *    that read a value the cone also reads from outside the block (the old
 *    counter, e.g. c+i), kept just ahead of the cone so both uses of the old
 *    value end together;
 *  - everything else keeps its relative order, and the group [readers, cone]
 *    goes where `distance` of those instructions still follow it, or earlier
 *    if one of them uses a group result. The packer's EARLY_COMPARE then
 *    emits T's compare right after the cone.
 * This is placement by dependency and distance, not a pattern: nothing
 * depends on the shift/xor opcodes. Pressure is checked by the caller. */
DEBUG_GET_ONCE_NUM_OPTION(agx_g17p_exit_compare_distance,
                          "AGX_G17P_EXIT_COMPARE_DISTANCE", 5)

static bool
ilp_defines(const agx_instr *I, unsigned ssa)
{
   agx_foreach_ssa_dest(I, d) {
      if (I->dest[d].value == ssa)
         return true;
   }
   return false;
}

/* Values produced by a non-phi instruction of this block. Phis stay at the
 * top and their values count as loop inputs. */
static bool
ilp_block_value(struct sched_node **nodes, unsigned n, unsigned ssa)
{
   for (unsigned k = 0; k < n; ++k) {
      if (nodes[k]->instr->op != AGX_OPCODE_PHI &&
          ilp_defines(nodes[k]->instr, ssa))
         return true;
   }
   return false;
}

static bool
ilp_place_exit_compare(agx_block *block, struct sched_node **nodes,
                       struct sched_node **order, unsigned n,
                       const struct util_dynarray *dependents, void *memctx)
{
   const int64_t distance_opt = debug_get_option_agx_g17p_exit_compare_distance();
   if (distance_opt < 1 || distance_opt > 64)
      return false;
   const unsigned distance = (unsigned)distance_opt;

   agx_instr *T = agx_last_instr(block);
   if (!T || (T->op != AGX_OPCODE_IF_ICMP && T->op != AGX_OPCODE_WHILE_ICMP &&
              T->op != AGX_OPCODE_BREAK_IF_ICMP))
      return false;

   bool *cone = rzalloc_array(memctx, bool, n);
   bool *group = rzalloc_array(memctx, bool, n);
   if (!cone || !group)
      return false;

   unsigned cone_n = 0;
   agx_foreach_ssa_src(T, s) {
      for (unsigned i = 0; i < n; ++i) {
         if (!cone[i] && nodes[i]->instr->op != AGX_OPCODE_PHI &&
             ilp_defines(nodes[i]->instr, T->src[s].value)) {
            cone[i] = true;
            cone_n++;
         }
      }
   }
   if (cone_n == 0)
      return false;

   /* Close over in-block dependencies other than phis. */
   for (bool grew = true; grew;) {
      grew = false;
      for (unsigned i = 0; i < n; ++i) {
         if (!cone[i])
            continue;
         util_dynarray_foreach(&nodes[i]->dag.edges, struct dag_edge, edge) {
            const struct sched_node *child =
               (const struct sched_node *)edge->child;
            if (child->instr->op != AGX_OPCODE_PHI && !cone[child->index]) {
               cone[child->index] = true;
               cone_n++;
               grew = true;
            }
         }
      }
   }
   if (cone_n > 3)
      return false;

   /* Readers: other instructions whose only in-block dependencies are phis
    * and which read a loop input the cone also reads (the old counter). */
   unsigned readers = 0;
   for (unsigned j = 0; j < n; ++j) {
      group[j] = cone[j];
      if (cone[j] || nodes[j]->instr->op == AGX_OPCODE_PHI)
         continue;
      bool independent = true;
      util_dynarray_foreach(&nodes[j]->dag.edges, struct dag_edge, edge) {
         const struct sched_node *child = (const struct sched_node *)edge->child;
         independent &= child->instr->op == AGX_OPCODE_PHI;
      }
      if (!independent)
         continue;
      bool shares = false;
      agx_foreach_ssa_src(nodes[j]->instr, sj) {
         const unsigned v = nodes[j]->instr->src[sj].value;
         if (ilp_block_value(nodes, n, v))
            continue;
         for (unsigned i = 0; i < n && !shares; ++i) {
            if (!cone[i])
               continue;
            agx_foreach_ssa_src(nodes[i]->instr, si)
               shares |= nodes[i]->instr->src[si].value == v;
         }
      }
      if (shares && readers < 2) {
         group[j] = true;
         readers++;
      }
   }

   struct sched_node **rest = rzalloc_array(memctx, struct sched_node *, n);
   struct sched_node **grp = rzalloc_array(memctx, struct sched_node *, n);
   struct sched_node **out = rzalloc_array(memctx, struct sched_node *, n);
   if (!rest || !grp || !out)
      return false;
   unsigned nr = 0, ng = 0, phis = 0;
   /* Readers first, then the cone, each in the scheduled order. */
   for (unsigned pass = 0; pass < 2; ++pass) {
      for (unsigned i = 0; i < n; ++i) {
         const unsigned idx = order[i]->index;
         if (group[idx] && cone[idx] == (pass == 1))
            grp[ng++] = order[i];
      }
   }
   for (unsigned i = 0; i < n; ++i) {
      if (!group[order[i]->index]) {
         phis += order[i]->instr->op == AGX_OPCODE_PHI;
         rest[nr++] = order[i];
      }
   }

   const unsigned movable = nr - phis;
   unsigned p = phis + (movable > distance ? movable - distance : 0);
   for (unsigned q = 0; q < nr; ++q) {
      bool uses_group = false;
      for (unsigned g = 0; g < ng && !uses_group; ++g) {
         util_dynarray_foreach(&dependents[grp[g]->index], unsigned, d)
            uses_group |= *d == rest[q]->index;
      }
      if (uses_group) {
         p = MIN2(p, q);
         break;
      }
   }
   /* Phis stay first; a group user among them means no valid placement. */
   if (p < phis)
      return false;

   if (debug_get_bool_option("AGX_G17P_EXIT_COMPARE_TRACE", false)) {
      fprintf(stderr, "exit-compare: n=%u cone=%u readers=%u p=%u rest:", n,
              cone_n, readers, p);
      for (unsigned i = 0; i < nr; ++i)
         fprintf(stderr, " %s", agx_opcodes_info[rest[i]->instr->op].name);
      fprintf(stderr, " | group:");
      for (unsigned i = 0; i < ng; ++i)
         fprintf(stderr, " %s", agx_opcodes_info[grp[i]->instr->op].name);
      fprintf(stderr, "\n");
   }

   unsigned k = 0;
   for (unsigned i = 0; i < p; ++i)
      out[k++] = rest[i];
   for (unsigned i = 0; i < ng; ++i)
      out[k++] = grp[i];
   for (unsigned i = p; i < nr; ++i)
      out[k++] = rest[i];
   assert(k == n);

   bool changed = false;
   for (unsigned i = 0; i < n; ++i) {
      changed |= out[i] != order[i];
      order[i] = out[i];
   }
   return changed;
}

/*
 * AGX_G17P_TILE_WAIT_SINK (default on; =0 restores the plain ILP order).
 *
 * WAIT_PIX has no SSA result; the list scheduler above gives it an ALU
 * latency and ranks it by the critical path to the tile access it guards, so
 * it tends to be hoisted as early as its barrier edges allow. In the colour
 * block of the light fragment it placed the class-4 (write) wait twelve
 * instructions ahead of LSTCOL, and the class-8 (read) wait three ahead of
 * LLDCOL, so the in-order warp could block on the pixel protocol before
 * independent arithmetic. Apple's G17P fragments put each wait immediately
 * ahead of the tile access it orders (fs-apple.bin +0x106 pixwait 8 ->
 * lldcol, +0x1a2 pixwait 4 -> lstcol), which is also what the pressure
 * scheduler's own "sink wait_pix" heuristic does.
 *
 * Move each WAIT_PIX down to just before its first dependent in the chosen
 * order. It only moves later, never past anything that depends on it, and
 * every instruction it depends on stays earlier, so all DAG edges hold.
 */
static void
ilp_sink_pixel_waits(struct sched_node **order, unsigned n,
                     const struct util_dynarray *dependents)
{
   for (unsigned p = 0; p < n; ++p) {
      if (order[p]->instr->op != AGX_OPCODE_WAIT_PIX)
         continue;

      const unsigned w = order[p]->index;
      unsigned q = p + 1;
      for (; q < n; ++q) {
         bool depends = false;
         util_dynarray_foreach(&dependents[w], unsigned, d)
            depends |= *d == order[q]->index;
         if (depends)
            break;
      }

      /* No dependent inside the block: leave it where it is. */
      if (q == n || q == p + 1)
         continue;

      struct sched_node *wait = order[p];
      memmove(&order[p], &order[p + 1], (q - 1 - p) * sizeof(order[0]));
      order[q - 1] = wait;
      /* The instruction now at p has not been examined yet. */
      --p;
   }
}

static void
ilp_schedule_block(agx_context *ctx, agx_block *block, void *memctx,
                   bool native_uniforms, unsigned alu_latency,
                   signed slack)
{
   struct util_dynarray all;
   util_dynarray_init(&all, memctx);
   struct dag *dag = create_dag(ctx, block, memctx, native_uniforms, &all);
   if (!dag)
      return;

   const unsigned n = util_dynarray_num_elements(&all, struct sched_node *);
   if (n < 3)
      return;
   struct sched_node **nodes = all.data;

   unsigned *ndeps = rzalloc_array(memctx, unsigned, n);
   unsigned *height = rzalloc_array(memctx, unsigned, n);
   unsigned *ready = rzalloc_array(memctx, unsigned, n);
   bool *done = rzalloc_array(memctx, bool, n);
   struct util_dynarray *dependents =
      rzalloc_array(memctx, struct util_dynarray, n);
   struct sched_node **order = rzalloc_array(memctx, struct sched_node *, n);
   if (!ndeps || !height || !ready || !done || !dependents || !order)
      return;

   for (unsigned i = 0; i < n; ++i)
      util_dynarray_init(&dependents[i], memctx);

   /* Edges run from a dependent (parent) to its dependency (child). */
   for (unsigned i = 0; i < n; ++i) {
      util_dynarray_foreach(&nodes[i]->dag.edges, struct dag_edge, edge) {
         const struct sched_node *child = (const struct sched_node *)edge->child;
         ndeps[i]++;
         util_dynarray_append(&dependents[child->index], i);
      }
   }

   /* Dependents always follow their dependencies in the original order, so
    * a reverse sweep computes the critical-path height. */
   for (unsigned i = n; i-- > 0;) {
      unsigned tail = 0;
      util_dynarray_foreach(&dependents[i], unsigned, d)
         tail = MAX2(tail, height[*d]);
      height[i] = ilp_latency(nodes[i]->instr, alu_latency) + tail;
   }

   unsigned cycle = 0;
   bool changed = false;
   for (unsigned step = 0; step < n; ++step) {
      int best = -1;
      unsigned best_t = 0;
      for (unsigned i = 0; i < n; ++i) {
         if (done[i] || ndeps[i])
            continue;
         const unsigned t = MAX2(ready[i], cycle);
         if (best < 0 || t < best_t ||
             (t == best_t && height[i] > height[best])) {
            best = i;
            best_t = t;
         }
      }
      if (best < 0)
         return; /* cycle in the DAG: leave the block alone */

      done[best] = true;
      order[step] = nodes[best];
      changed |= (unsigned)best != step;
      cycle = best_t + 1;
      const unsigned lat = ilp_latency(nodes[best]->instr, alu_latency);
      util_dynarray_foreach(&dependents[best], unsigned, d) {
         ndeps[*d]--;
         ready[*d] = MAX2(ready[*d], best_t + lat);
      }
   }
   /* The ILP order itself counts as a change from the source order. */
   for (unsigned i = 0; i < n; ++i)
      changed |= order[i]->index != i;
   if ((agx_get_codegen_flags() & AGX_CODEGEN_G17P_EXIT_COMPARE) &&
       ilp_place_exit_compare(block, nodes, order, n, dependents, memctx)) {
      changed = false;
      for (unsigned i = 0; i < n; ++i)
         changed |= order[i]->index != i;
   }
   if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
       (agx_get_codegen_flags() & AGX_CODEGEN_G17P_TILE_WAIT_SINK)) {
      ilp_sink_pixel_waits(order, n, dependents);
      changed = false;
      for (unsigned i = 0; i < n; ++i)
         changed |= order[i]->index != i;
   }
   if (!changed)
      return;

   /* Live set at the logical end: walk back over the trailing control-flow
    * and export instructions, which stay where they are. */
   struct u_sparse_bitset live_end, scratch;
   u_sparse_bitset_init(&live_end, ctx->alloc, memctx);
   u_sparse_bitset_init(&scratch, ctx->alloc, memctx);
   u_sparse_bitset_dup(&live_end, &block->live_out);
   agx_instr *first_after = NULL;
   agx_foreach_instr_in_block(block, I) {
      if (instr_after_logical_end(I)) {
         first_after = I;
         break;
      }
   }
   if (first_after) {
      agx_foreach_instr_in_block_rev(block, I) {
         agx_liveness_ins_update(&live_end, I);
         if (I == first_after)
            break;
      }
   }

   const signed before = ilp_max_pressure(nodes, n, &live_end, &scratch);
   const signed after = ilp_max_pressure(order, n, &live_end, &scratch);
   if (after > before + slack)
      return;

   for (unsigned i = 0; i < n; ++i)
      agx_remove_instruction(order[i]->instr);
   for (unsigned i = 0; i < n; ++i) {
      if (first_after)
         list_addtail(&order[i]->instr->link, &first_after->link);
      else
         list_addtail(&order[i]->instr->link, &block->instructions);
   }
}

/* v52b: only pure-ALU loop bodies are interleaved. Offline, scheduling the
 * crossos bw_copy grid-stride loops (vector device loads and stores) grew
 * that kernel from 23 to 39 GPR halves; the xorshift loops the issue-rate gap
 * was measured on contain no memory access. */
static bool
ilp_block_has_memory(const agx_block *block)
{
   agx_foreach_instr_in_block(block, I) {
      switch (I->op) {
      case AGX_OPCODE_DEVICE_LOAD:
      case AGX_OPCODE_DEVICE_STORE:
      case AGX_OPCODE_ATOMIC:
      case AGX_OPCODE_LOCAL_LOAD:
      case AGX_OPCODE_LOCAL_STORE:
      case AGX_OPCODE_LOCAL_ATOMIC:
      case AGX_OPCODE_TEXTURE_LOAD:
      case AGX_OPCODE_TEXTURE_SAMPLE:
      case AGX_OPCODE_IMAGE_LOAD:
      case AGX_OPCODE_IMAGE_WRITE:
      case AGX_OPCODE_STACK_LOAD:
      case AGX_OPCODE_STACK_STORE:
      case AGX_OPCODE_LD_TILE:
      case AGX_OPCODE_ST_TILE:
      case AGX_OPCODE_BLOCK_IMAGE_STORE:
         return true;
      default:
         break;
      }
   }
   return false;
}

/* Schedule only a small, explicit arithmetic/tile domain. Schedule-class
 * NONE alone is not a purity guarantee (e.g. DOORBELL and texture ops).
 * SSA operands expose all data dependencies; physical register accesses and
 * uniform writes stay on the original schedule. */
static bool
ilp_single_tile_blend(const agx_context *ctx, const agx_block *block)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B ||
       ctx->stage != MESA_SHADER_FRAGMENT ||
       !(agx_get_codegen_flags() & AGX_CODEGEN_G17P_COLOR_ILP))
      return false;

   unsigned loads = 0, stores = 0, load_target = 0, store_target = 0;
   agx_foreach_instr_in_block(block, I) {
      if (instr_after_logical_end(I))
         break;
      switch (I->op) {
      case AGX_OPCODE_LD_TILE:
         ++loads;
         load_target = I->colour_target;
         break;
      case AGX_OPCODE_ST_TILE:
         ++stores;
         store_target = I->colour_target;
         break;
      case AGX_OPCODE_PHI:
      case AGX_OPCODE_WAIT_PIX:
      case AGX_OPCODE_MOV:
      case AGX_OPCODE_MOV_IMM:
      case AGX_OPCODE_SPLIT:
      case AGX_OPCODE_COLLECT:
      case AGX_OPCODE_BITOP:
      case AGX_OPCODE_BFI:
      case AGX_OPCODE_BFEIL:
      case AGX_OPCODE_IADD:
      case AGX_OPCODE_CONVERT:
      case AGX_OPCODE_FADD:
      case AGX_OPCODE_FMUL:
      case AGX_OPCODE_FFMA:
      case AGX_OPCODE_FLTNORM_AGX3:
      case AGX_OPCODE_NORMFLT_AGX3:
         break;
      default:
         return false;
      }
      agx_foreach_dest(I, d) {
         if (I->dest[d].type != AGX_INDEX_NORMAL &&
             I->dest[d].type != AGX_INDEX_NULL)
            return false;
      }
      agx_foreach_src(I, src) {
         if (I->src[src].type == AGX_INDEX_REGISTER)
            return false;
      }
   }
   return loads == 1 && stores == 1 && load_target == store_target;
}

void
agx_ilp_schedule(agx_context *ctx)
{
   const int64_t latency = debug_get_option_agx_g17p_ilp_latency();
   const int64_t slack = debug_get_option_agx_g17p_ilp_slack();
   const unsigned alu_latency =
      latency >= 1 && latency <= 64 ? (unsigned)latency : 4;

   agx_compute_liveness(ctx);
   void *memctx = ralloc_context(ctx);

   bool native_uniforms = false;
   agx_foreach_instr_global(ctx, I) {
      native_uniforms |= native_uniform_write(I);
   }

   /* Only loop bodies: that is where Apple's interleaving pays and where the
    * issue-rate gap was measured. Straight-line code keeps the pressure
    * schedule (offline, scheduling it grew some meta shaders' registers).
    * In structured AGX control flow a loop spans the source-order interval
    * from its header to the block holding the back edge. */
   unsigned nr_blocks = 0;
   agx_foreach_block(ctx, block)
      nr_blocks = MAX2(nr_blocks, block->index + 1);
   bool *in_loop = rzalloc_array(memctx, bool, MAX2(nr_blocks, 1));
   if (!in_loop) {
      ralloc_free(memctx);
      return;
   }
   agx_foreach_block(ctx, block) {
      agx_foreach_successor(block, succ) {
         if (succ->loop_header && succ->index <= block->index) {
            for (unsigned i = succ->index; i <= block->index; ++i)
               in_loop[i] = true;
         }
      }
   }

   agx_foreach_block(ctx, block) {
      if (block->index >= nr_blocks)
         continue;
      if (in_loop[block->index] && !ilp_block_has_memory(block)) {
         ilp_schedule_block(ctx, block, memctx, native_uniforms, alu_latency,
                            slack >= 0 && slack <= 256 ? (signed)slack : 2);
      } else if (!in_loop[block->index] && ilp_single_tile_blend(ctx, block)) {
         /* Hardware-qualified color-only budget; leave loop pressure alone. */
         ilp_schedule_block(ctx, block, memctx, native_uniforms, alu_latency, 8);
      }
   }

   agx_foreach_instr_global(ctx, I) {
      agx_foreach_ssa_src(I, s)
         I->src[s].kill = false;
   }

   ralloc_free(memctx);
}
