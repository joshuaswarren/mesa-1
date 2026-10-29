/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * agx_nir_opt_sink_loop_sample_uses: a loop body's sample consumers move to
 * the latest top-level block dominating their uses; samples never move;
 * results beyond the per-loop budget keep their consumers in place.
 *
 * agx_nir_opt_hoist_loop_samples: unconditional samples move ahead of the
 * side-effect-free conditional blocks before them, which nir_opt_if_merge
 * then merges.
 */
#include "agx_nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/glsl_types.h"
#include <stdio.h>
#include <stdlib.h>

static unsigned failures;

#define CHECK(cond)                                                          \
   do {                                                                      \
      if (!(cond)) {                                                         \
         fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__,   \
                 #cond);                                                     \
         failures++;                                                         \
      }                                                                      \
   } while (0)

static nir_def *
txf3d(nir_builder *b, nir_def *c, unsigned bit_size)
{
   nir_tex_instr *tex = nir_tex_instr_create(b->shader, 2);
   tex->op = nir_texop_txf;
   tex->sampler_dim = GLSL_SAMPLER_DIM_3D;
   tex->dest_type = bit_size == 16 ? nir_type_int16 : nir_type_int32;
   tex->coord_components = 3;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord,
                                     nir_vec3(b, c, nir_imm_int(b, 0),
                                              nir_imm_int(b, 0)));
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_lod, nir_imm_int(b, 0));
   nir_def_init(&tex->instr, &tex->def, 4, bit_size);
   nir_builder_instr_insert(b, &tex->instr);
   return &tex->def;
}

static nir_block *
def_block(nir_def *def)
{
   return nir_def_block(def);
}

struct loop_shader {
   nir_builder b;
   nir_loop *loop;
   nir_phi_instr *c_phi, *acc_phi;
   nir_def *c, *acc;
   bool early_break;
};

/* for (c = 100, acc = 0;; --c) { body; if (c - 1 < 0) break; } */
static void
loop_begin(struct loop_shader *s)
{
   s->early_break = false;
   s->b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX, NULL, "loop");
   nir_builder *b = &s->b;
   nir_def *c0 = nir_imm_int(b, 100);
   nir_def *acc0 = nir_imm_int(b, 0);
   s->loop = nir_push_loop(b);
   s->c_phi = nir_phi_instr_create(b->shader);
   s->acc_phi = nir_phi_instr_create(b->shader);
   nir_def_init(&s->c_phi->instr, &s->c_phi->def, 1, 32);
   nir_def_init(&s->acc_phi->instr, &s->acc_phi->def, 1, 32);
   nir_block *preheader = nir_cf_node_as_block(nir_cf_node_prev(&s->loop->cf_node));
   nir_phi_instr_add_src(s->c_phi, preheader, c0);
   nir_phi_instr_add_src(s->acc_phi, preheader, acc0);
   nir_builder_instr_insert(b, &s->c_phi->instr);
   nir_builder_instr_insert(b, &s->acc_phi->instr);
   s->c = &s->c_phi->def;
   s->acc = &s->acc_phi->def;
}

static void
loop_end(struct loop_shader *s, nir_def *acc)
{
   nir_builder *b = &s->b;
   nir_def *next = nir_iadd_imm(b, s->c, -1);
   nir_push_if(b, nir_ilt_imm(b, next, 0));
   nir_jump(b, nir_jump_break);
   nir_pop_if(b, NULL);
   /* Phi sources only join their definitions' use lists on insertion. */
   nir_block *latch = nir_loop_last_block(s->loop);
   nir_block *header = nir_loop_first_block(s->loop);
   nir_instr_remove(&s->acc_phi->instr);
   nir_instr_remove(&s->c_phi->instr);
   nir_phi_instr_add_src(s->c_phi, latch, next);
   nir_phi_instr_add_src(s->acc_phi, latch, acc);
   nir_instr_insert(nir_before_block(header), &s->acc_phi->instr);
   nir_instr_insert(nir_before_block(header), &s->c_phi->instr);
   nir_pop_loop(b, s->loop);
   /* Only the header phi dominates an early break's exit as well. */
   nir_store_global(b, s->early_break ? s->acc : acc,
                    nir_imm_int64(b, 0x1000), .write_mask = 1,
                    .align_mul = 4);
   nir_validate_shader(b->shader, "before");
}

/* The witness shape: sample, consume, conditional sample, consume, sample,
 * consume. After the pass every consumer sits in the last top-level block
 * (the one holding the final sample), and no sample moved. */
static void
test_witness_shape(void)
{
   struct loop_shader s;
   loop_begin(&s);
   nir_builder *b = &s.b;

   nir_def *t0 = txf3d(b, s.c, 32);
   nir_def *n0 = nir_ineg(b, nir_channel(b, t0, 0));
   nir_def *a1 = nir_iadd(b, s.acc, n0);

   nir_def *zero = nir_imm_int(b, 0);
   nir_push_if(b, nir_ine_imm(b, s.c, 0));
   nir_def *t1 = txf3d(b, s.c, 32);
   nir_def *m1 = nir_imax(b, nir_channel(b, t1, 1), zero);
   nir_push_else(b, NULL);
   nir_pop_if(b, NULL);
   nir_def *p1 = nir_if_phi(b, m1, zero);
   nir_def *a2 = nir_iadd(b, a1, p1);
   nir_def *keep = nir_ieq_imm(b, s.c, 7); /* independent of any sample */

   nir_def *t2 = txf3d(b, s.c, 32);
   nir_def *n2 = nir_ineg(b, nir_channel(b, t2, 2));
   nir_def *a3 = nir_iadd(b, a2, n2);
   nir_def *a4 = nir_bcsel(b, keep, a3, a2);
   nir_block *b_t0 = def_block(t0), *b_t1 = def_block(t1), *b_t2 = def_block(t2);
   nir_block *b_m1 = def_block(m1), *b_keep = def_block(keep);
   CHECK(b_t0 != b_t2);
   loop_end(&s, a4);

   CHECK(agx_nir_opt_sink_loop_sample_uses(b->shader));
   nir_validate_shader(b->shader, "after");

   CHECK(def_block(t0) == b_t0);
   CHECK(def_block(t1) == b_t1);
   CHECK(def_block(t2) == b_t2);
   CHECK(def_block(n0) == b_t2);
   CHECK(def_block(a1) == b_t2);
   CHECK(def_block(a2) == b_t2);
   CHECK(def_block(a3) == b_t2);
   CHECK(def_block(a4) == b_t2);
   /* A consumer inside the conditional stays: its use is the phi. */
   CHECK(def_block(m1) == b_m1);
   /* Sample-independent ALU keeps its place. */
   CHECK(def_block(keep) == b_keep);
   /* Producers precede their consumers in the target block. */
   CHECK(nir_block_dominates(def_block(n0), def_block(a1)));
   bool seen_n0 = false, ordered = true;
   nir_foreach_instr(instr, b_t2) {
      if (instr == nir_def_instr(n0))
         seen_n0 = true;
      if (instr == nir_def_instr(a1) && !seen_n0)
         ordered = false;
   }
   CHECK(ordered);
   CHECK(!agx_nir_opt_sink_loop_sample_uses(b->shader));
   ralloc_free(b->shader);
}

/* Thirteen 32-bit vec4 samples: the first twelve (96 halves) sink their
 * consumers; the thirteenth's own consumer stays where it was. */
static void
test_budget(void)
{
   struct loop_shader s;
   loop_begin(&s);
   nir_builder *b = &s.b;
   nir_def *neg[13];
   nir_def *acc = s.acc;
   for (unsigned i = 0; i < 13; ++i) {
      nir_def *t = txf3d(b, s.c, 32);
      neg[i] = nir_ineg(b, nir_channel(b, t, 0));
      acc = nir_iadd(b, acc, neg[i]);
      nir_push_if(b, nir_ine_imm(b, s.c, (int)i + 1));
      nir_pop_if(b, NULL);
   }
   nir_block *before[13];
   for (unsigned i = 0; i < 13; ++i)
      before[i] = def_block(neg[i]);
   nir_def *last = txf3d(b, s.c, 32);
   acc = nir_iadd(b, acc, nir_channel(b, last, 3));
   nir_block *tail = def_block(last);
   loop_end(&s, acc);

   CHECK(agx_nir_opt_sink_loop_sample_uses(b->shader));
   nir_validate_shader(b->shader, "after");
   for (unsigned i = 0; i < 12; ++i)
      CHECK(def_block(neg[i]) == tail);
   CHECK(def_block(neg[12]) == before[12]);
   ralloc_free(b->shader);
}

/* No sample in the loop: nothing moves. */
static void
test_no_samples(void)
{
   struct loop_shader s;
   loop_begin(&s);
   nir_builder *b = &s.b;
   nir_def *a1 = nir_iadd(b, s.acc, s.c);
   nir_push_if(b, nir_ine_imm(b, s.c, 0));
   nir_pop_if(b, NULL);
   nir_def *a2 = nir_imul(b, a1, s.c);
   nir_block *b1 = def_block(a1);
   loop_end(&s, a2);
   CHECK(!agx_nir_opt_sink_loop_sample_uses(b->shader));
   CHECK(def_block(a1) == b1);
   ralloc_free(b->shader);
}

/* Count the if-statements directly in the loop body. */
static unsigned
loop_body_ifs(nir_loop *loop)
{
   unsigned n = 0;
   foreach_list_typed(nir_cf_node, node, node, &loop->body)
      n += node->type == nir_cf_node_if;
   return n;
}

static unsigned
count_tex(nir_shader *shader)
{
   unsigned n = 0;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block)
            n += instr->type == nir_instr_type_tex;
      }
   }
   return n;
}

/* The robustness2 shape: per axis an unconditional fetch at a coordinate
 * derived from c, then a conditional fetch of texel 0 at level c. After the
 * sink, every unconditional fetch and its coordinate math moves to the first
 * block, the two conditional blocks become adjacent and merge, and their
 * identical fetches CSE. */
static void
test_hoist_witness_shape(void)
{
   struct loop_shader s;
   loop_begin(&s);
   nir_builder *b = &s.b;
   nir_def *cond = nir_ine_imm(b, s.c, 0);
   nir_def *zero = nir_imm_int(b, 0);

   nir_def *t0 = txf3d(b, s.c, 32);
   nir_def *acc = nir_iadd(b, s.acc, nir_channel(b, t0, 0));
   nir_def *p[2], *t[2], *coord[2];
   for (unsigned i = 0; i < 2; ++i) {
      nir_push_if(b, cond);
      nir_def *f = txf3d(b, zero, 32);
      nir_def *m = nir_imax(b, nir_channel(b, f, 1), zero);
      nir_push_else(b, NULL);
      nir_pop_if(b, NULL);
      p[i] = nir_if_phi(b, m, zero);
      acc = nir_iadd(b, acc, p[i]);
      coord[i] = nir_iadd_imm(b, s.c, (int)i + 1);
      t[i] = txf3d(b, coord[i], 32);
      acc = nir_iadd(b, acc, nir_channel(b, t[i], 2));
   }
   nir_block *first = def_block(t0);
   CHECK(def_block(t[0]) != first && def_block(t[1]) != first);
   loop_end(&s, acc);
   CHECK(loop_body_ifs(s.loop) == 3);
   CHECK(count_tex(b->shader) == 5);

   CHECK(agx_nir_opt_sink_loop_sample_uses(b->shader));
   CHECK(agx_nir_opt_hoist_loop_samples(b->shader));
   nir_validate_shader(b->shader, "after hoist");
   CHECK(def_block(t[0]) == first);
   CHECK(def_block(t[1]) == first);
   CHECK(def_block(coord[0]) == first);
   CHECK(def_block(coord[1]) == first);
   /* The samples keep their source order. */
   bool seen_t0 = false, seen_t1 = false, ordered = true;
   nir_foreach_instr(instr, first) {
      if (instr == nir_def_instr(t0))
         seen_t0 = true;
      if (instr == nir_def_instr(t[0])) {
         ordered &= seen_t0;
         seen_t1 = true;
      }
      if (instr == nir_def_instr(t[1]))
         ordered &= seen_t1;
   }
   CHECK(ordered);

   CHECK(nir_opt_if_merge(b->shader));
   nir_validate_shader(b->shader, "after merge");
   CHECK(loop_body_ifs(s.loop) == 2); /* the merged block and the break */
   nir_opt_cse(b->shader);
   nir_opt_dce(b->shader);
   nir_validate_shader(b->shader, "after cse");
   CHECK(count_tex(b->shader) == 4);

   CHECK(!agx_nir_opt_hoist_loop_samples(b->shader));
   ralloc_free(b->shader);
}

/* A conditional block with a side effect, or one leaving the loop, keeps the
 * samples after it in place. */
static void
test_hoist_barriers(void)
{
   for (unsigned store = 0; store < 2; ++store) {
      struct loop_shader s;
      loop_begin(&s);
      s.early_break = !store;
      nir_builder *b = &s.b;
      nir_def *t0 = txf3d(b, s.c, 32);
      nir_push_if(b, nir_ine_imm(b, s.c, 3));
      if (store) {
         nir_store_global(b, s.c, nir_imm_int64(b, 0x2000), .write_mask = 1,
                          .align_mul = 4);
      } else {
         nir_jump(b, nir_jump_break);
      }
      nir_pop_if(b, NULL);
      nir_def *t1 = txf3d(b, nir_iadd_imm(b, s.c, 1), 32);
      nir_def *acc = nir_iadd(b, s.acc, nir_channel(b, t0, 0));
      acc = nir_iadd(b, acc, nir_channel(b, t1, 0));
      nir_block *b_t1 = def_block(t1);
      loop_end(&s, acc);

      agx_nir_opt_sink_loop_sample_uses(b->shader);
      CHECK(!agx_nir_opt_hoist_loop_samples(b->shader));
      CHECK(def_block(t1) == b_t1);
      ralloc_free(b->shader);
   }
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   test_witness_shape();
   test_budget();
   test_no_samples();
   test_hoist_witness_shape();
   test_hoist_barriers();
   glsl_type_singleton_decref();
   if (failures) {
      fprintf(stderr, "%u check(s) failed\n", failures);
      return EXIT_FAILURE;
   }
   printf("agx_nir_opt_sink_loop_sample_uses: all checks passed\n");
   return EXIT_SUCCESS;
}
