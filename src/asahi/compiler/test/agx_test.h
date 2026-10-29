/*
 * Copyright 2020-2021 Collabora Ltd.
 * SPDX-License-Identifier: MIT
 */

#ifndef __AGX_TEST_H
#define __AGX_TEST_H

#include <inttypes.h>
#include <stdio.h>
#include "compiler/nir/nir_builder.h"
#include "agx_builder.h"
#include "agx_compiler.h"

/* Helper to generate a agx_builder suitable for creating test instructions */
static inline agx_builder *
agx_test_builder(void *memctx)
{
   agx_context *ctx = rzalloc(memctx, agx_context);
   list_inithead(&ctx->blocks);

   agx_block *blk = rzalloc(ctx, agx_block);
   blk->predecessors = UTIL_DYNARRAY_INIT;
   ctx->num_blocks = 1;

   list_addtail(&blk->link, &ctx->blocks);
   list_inithead(&blk->instructions);

   agx_builder *b = rzalloc(memctx, agx_builder);
   b->shader = ctx;
   b->cursor = agx_after_block(blk);

   return b;
}

static inline agx_block *
agx_test_block(agx_context *ctx)
{
   agx_block *blk = rzalloc(ctx, agx_block);

   util_dynarray_init(&blk->predecessors, blk);
   list_addtail(&blk->link, &ctx->blocks);
   list_inithead(&blk->instructions);

   blk->index = ctx->num_blocks++;

   return blk;
}

/* Helper to compare for logical equality of instructions. Need to compare the
 * pointers, then compare raw data.
 */
static inline bool
agx_instr_equal(agx_instr *A, agx_instr *B)
{
   unsigned pointers = sizeof(struct list_head) + sizeof(agx_index *) * 2;

   if (A->nr_srcs != B->nr_srcs)
      return false;

   if (memcmp(A->src, B->src, A->nr_srcs * sizeof(agx_index)))
      return false;

   if (A->nr_dests != B->nr_dests)
      return false;

   if (memcmp(A->dest, B->dest, A->nr_dests * sizeof(agx_index)))
      return false;

   return memcmp((uint8_t *)A + pointers, (uint8_t *)B + pointers,
                 sizeof(agx_instr) - pointers) == 0;
}

static inline bool
agx_block_equal(agx_block *A, agx_block *B)
{
   if (list_length(&A->instructions) != list_length(&B->instructions))
      return false;

   list_pair_for_each_entry(agx_instr, insA, insB, &A->instructions,
                            &B->instructions, link) {
      if (!agx_instr_equal(insA, insB))
         return false;
   }

   return true;
}

static inline bool
agx_shader_equal(agx_context *A, agx_context *B)
{
   if (list_length(&A->blocks) != list_length(&B->blocks))
      return false;

   list_pair_for_each_entry(agx_block, blockA, blockB, &A->blocks, &B->blocks,
                            link) {
      if (!agx_block_equal(blockA, blockB))
         return false;
   }

   return true;
}

#define ASSERT_SHADER_EQUAL(A, B)                                              \
   if (!agx_shader_equal(A, B)) {                                              \
      ADD_FAILURE();                                                           \
      fprintf(stderr, "Pass produced unexpected results");                     \
      fprintf(stderr, "  Actual:\n");                                          \
      agx_print_shader(A, stderr);                                             \
      fprintf(stderr, " Expected:\n");                                         \
      agx_print_shader(B, stderr);                                             \
      fprintf(stderr, "\n");                                                   \
   }

#define INSTRUCTION_CASE(instr, expected, pass)                                \
   do {                                                                        \
      agx_builder *A = agx_test_builder(mem_ctx);                              \
      agx_builder *B = agx_test_builder(mem_ctx);                              \
      {                                                                        \
         agx_builder *b = A;                                                   \
         instr;                                                                \
      }                                                                        \
      {                                                                        \
         agx_builder *b = B;                                                   \
         expected;                                                             \
      }                                                                        \
      pass(A->shader);                                                         \
      ASSERT_SHADER_EQUAL(A->shader, B->shader);                               \
   } while (0)

#ifdef __cplusplus
/* The helpers below use C++ default arguments; C tests (for example
 * test-agx3-m5-instructions.c) only need the builder helpers above. */

/* nir_builder's named-argument macros expand to an initializer that mixes a
 * positional element with designators, which C++ rejects as a C99 extension.
 * Build the intrinsic first and set its indices through the accessors. */
static inline nir_def *
agx_test_load_preamble(nir_builder *b, unsigned num_components,
                       unsigned bit_size, int base,
                       nir_preamble_class cls = nir_preamble_class_general)
{
   nir_def *def = nir_load_preamble(b, num_components, bit_size);
   nir_intrinsic_set_base(nir_def_as_intrinsic(def), base);
   nir_intrinsic_set_preamble_class(nir_def_as_intrinsic(def), cls);
   return def;
}

static inline nir_intrinsic_instr *
agx_test_store_preamble(nir_builder *b, nir_def *value, int base,
                        nir_preamble_class cls = nir_preamble_class_general)
{
   nir_intrinsic_instr *intr = nir_store_preamble(b, value);
   nir_intrinsic_set_base(intr, base);
   nir_intrinsic_set_preamble_class(intr, cls);
   return intr;
}

static inline nir_def *
agx_test_bindless_image_agx(nir_builder *b, nir_def *handle, unsigned desc_set)
{
   nir_def *def = nir_bindless_image_agx(b, handle);
   nir_intrinsic_set_desc_set(nir_def_as_intrinsic(def), desc_set);
   return def;
}

static inline nir_def *
agx_test_load_global_constant(nir_builder *b, unsigned num_components,
                              unsigned bit_size, nir_def *addr,
                              unsigned align_mul, gl_access_qualifier access)
{
   nir_def *def = nir_load_global_constant(b, num_components, bit_size, addr);
   nir_intrinsic_set_align(nir_def_as_intrinsic(def), align_mul, 0);
   nir_intrinsic_set_access(nir_def_as_intrinsic(def), access);
   return def;
}

static inline nir_def *
agx_test_bindless_image_load(nir_builder *b, unsigned num_components,
                             unsigned bit_size, nir_def *handle,
                             nir_def *coords, nir_def *sample, nir_def *lod,
                             enum glsl_sampler_dim dim, enum pipe_format format,
                             gl_access_qualifier access, nir_alu_type dest_type)
{
   nir_def *def = nir_bindless_image_load(b, num_components, bit_size, handle,
                                          coords, sample, lod);
   nir_intrinsic_instr *intr = nir_def_as_intrinsic(def);
   nir_intrinsic_set_image_dim(intr, dim);
   nir_intrinsic_set_format(intr, format);
   nir_intrinsic_set_access(intr, access);
   nir_intrinsic_set_dest_type(intr, dest_type);
   return def;
}

static inline nir_intrinsic_instr *
agx_test_store_output(nir_builder *b, nir_def *value, nir_def *offset,
                      nir_alu_type src_type, unsigned location)
{
   nir_intrinsic_instr *intr = nir_store_output(b, value, offset);
   nir_io_semantics sem = {};
   sem.location = location;
   nir_intrinsic_set_src_type(intr, src_type);
   nir_intrinsic_set_io_semantics(intr, sem);
   return intr;
}

static inline nir_intrinsic_instr *
agx_test_store_local_pixel_agx(nir_builder *b, nir_def *value, nir_def *mask,
                               nir_def *coords, int base, unsigned write_mask,
                               enum pipe_format format)
{
   nir_intrinsic_instr *intr =
      nir_store_local_pixel_agx(b, value, mask, coords);
   nir_intrinsic_set_base(intr, base);
   nir_intrinsic_set_write_mask(intr, write_mask);
   nir_intrinsic_set_format(intr, format);
   return intr;
}

#endif /* __cplusplus */

#endif
