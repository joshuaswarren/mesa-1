/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_test.h"

#include <gtest/gtest.h>

/* M5 copy loops: reorderable image reads issue ahead of the writes that
 * consume them, as Apple's compiler does for a vertex copy loop (four SMP,
 * then four IMGWCI). */

namespace {

struct copy_shader {
   void *mem;
   agx_shader_key key;
   agx_builder *b;
   agx_block *body;
};

static void
copy_shader_init(copy_shader *s, mesa_shader_stage stage, bool loop)
{
   s->mem = ralloc_context(NULL);
   s->key = {};
   s->key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   s->key.dev.chip = AGX_CHIP_G17S;
   s->b = agx_test_builder(s->mem);
   s->b->shader->key = &s->key;
   s->b->shader->stage = stage;
   s->b->shader->nir = rzalloc(s->mem, nir_shader);
   s->b->shader->nir->info.stage = stage;
   s->body = agx_start_block(s->b->shader);
   if (loop) {
      agx_block *entry = s->body;
      s->body = agx_test_block(s->b->shader);
      agx_block *exit = agx_test_block(s->b->shader);
      s->body->loop_header = true;
      agx_block_add_successor(entry, s->body);
      agx_block_add_successor(s->body, s->body);
      agx_block_add_successor(s->body, exit);
   }
}

/* Copy n texels at constant x coordinates defined in the entry block. */
static void
emit_copies(copy_shader *s, unsigned n, bool reorderable)
{
   agx_builder *b = s->b;
   agx_index y = agx_preload(b, agx_register(2, AGX_SIZE_32));
   agx_index z = agx_preload(b, agx_register(4, AGX_SIZE_32));
   agx_index x[16];
   assert(n <= 16);
   for (unsigned i = 0; i < n; ++i)
      x[i] = agx_mov_imm(b, 32, i);

   b->cursor = agx_after_block(s->body);
   for (unsigned i = 0; i < n; ++i) {
      agx_index coords = agx_vec_temp(b->shader, AGX_SIZE_32, 3);
      agx_instr *collect = agx_collect_to(b, coords, 3);
      collect->src[0] = x[i];
      collect->src[1] = y;
      collect->src[2] = z;

      agx_index texel = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
      agx_instr *load = agx_image_load_to(
         b, texel, coords, agx_zero(), agx_zero(), agx_immediate(1),
         agx_immediate(0), agx_null(), AGX_DIM_3D, AGX_LOD_MODE_AUTO_LOD, 0,
         false, false, false);
      load->mask = 1;
      load->memory_can_reorder = reorderable;
      agx_image_write(b, texel, coords, agx_zero(),
                      agx_uniform(8, AGX_SIZE_64),
                      agx_register(28, AGX_SIZE_32), AGX_DIM_3D, false);
   }
}

/* Largest number of reads issued ahead of the first unissued write. */
static unsigned
max_reads_in_flight(agx_block *block, unsigned *first_write_at)
{
   unsigned loads = 0, writes = 0, max = 0, pos = 0;
   *first_write_at = ~0u;
   agx_foreach_instr_in_block(block, I) {
      if (I->op == AGX_OPCODE_IMAGE_LOAD) {
         ++loads;
         max = MAX2(max, loads - writes);
      } else if (I->op == AGX_OPCODE_IMAGE_WRITE) {
         if (writes == 0)
            *first_write_at = pos;
         ++writes;
      }
      ++pos;
   }
   return max;
}

static unsigned
loads_before_first_write(agx_block *block)
{
   unsigned loads = 0;
   agx_foreach_instr_in_block(block, I) {
      if (I->op == AGX_OPCODE_IMAGE_WRITE)
         break;
      loads += I->op == AGX_OPCODE_IMAGE_LOAD;
   }
   return loads;
}

} /* namespace */

TEST(M5CopySchedule, VertexReadsIssueAheadOfWrites)
{
   copy_shader s;
   copy_shader_init(&s, MESA_SHADER_VERTEX, false);
   emit_copies(&s, 8, true);
   agx_ilp_schedule(s.b->shader);

   unsigned first_write;
   EXPECT_EQ(loads_before_first_write(s.body), 4u);
   EXPECT_EQ(max_reads_in_flight(s.body, &first_write), 4u);
   ralloc_free(s.mem);
}

TEST(M5CopySchedule, OrderedReadsStayBehindWrites)
{
   /* A read that may observe an earlier write keeps memory order. */
   copy_shader s;
   copy_shader_init(&s, MESA_SHADER_VERTEX, false);
   emit_copies(&s, 8, false);
   agx_ilp_schedule(s.b->shader);

   unsigned first_write;
   EXPECT_EQ(loads_before_first_write(s.body), 1u);
   EXPECT_EQ(max_reads_in_flight(s.body, &first_write), 1u);
   ralloc_free(s.mem);
}

TEST(M5CopySchedule, LoopConstantsRematerializedInBody)
{
   copy_shader s;
   copy_shader_init(&s, MESA_SHADER_VERTEX, true);
   emit_copies(&s, 8, true);
   agx_ilp_schedule(s.b->shader);

   /* Every coordinate constant is now defined in the loop body. */
   unsigned body_constants = 0;
   agx_foreach_instr_in_block(s.body, I)
      body_constants += I->op == AGX_OPCODE_MOV_IMM;
   EXPECT_EQ(body_constants, 8u);
   agx_foreach_instr_in_block(agx_start_block(s.b->shader), I)
      EXPECT_NE(I->op, AGX_OPCODE_MOV_IMM);

   unsigned first_write;
   EXPECT_EQ(loads_before_first_write(s.body), 4u);
   EXPECT_EQ(max_reads_in_flight(s.body, &first_write), 4u);
   ralloc_free(s.mem);
}

TEST(M5CopySchedule, OtherGenerationsUnchanged)
{
   copy_shader s;
   copy_shader_init(&s, MESA_SHADER_VERTEX, false);
   s.key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   s.key.dev.chip = AGX_CHIP_G17P;
   emit_copies(&s, 8, true);
   agx_ilp_schedule(s.b->shader);

   unsigned first_write;
   EXPECT_EQ(loads_before_first_write(s.body), 1u);
   EXPECT_EQ(max_reads_in_flight(s.body, &first_write), 1u);
   ralloc_free(s.mem);
}
