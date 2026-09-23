/*
 * Copyright 2023 Alyssa Rosenzweig
 * Copyright 2022 Collabora Ltd.
 * SPDX-License-Identifier: MIT
 */

/* Bottom-up local scheduler to reduce register pressure */

#include "util/dag.h"
#include "util/u_debug.h"
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
create_dag(agx_context *ctx, agx_block *block, void *memctx, bool native_uniforms)
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
      (enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C) &&
      !debug_get_option_agx_stvary_noorder();

   agx_foreach_instr_in_block(block, I) {
      /* Don't touch control flow */
      if (instr_after_logical_end(I))
         break;

      struct sched_node *node = rzalloc(memctx, struct sched_node);
      node->instr = I;
      dag_init_node(dag, &node->dag);
      if (native_uniforms)
         util_dynarray_append(&nodes, node);

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
         .dag = create_dag(ctx, block, memctx, native_uniforms),
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
