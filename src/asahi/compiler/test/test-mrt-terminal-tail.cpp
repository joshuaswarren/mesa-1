/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "compiler/nir/nir_builder.h"
#include "agx_nir.h"
}
#include <vector>

static bool
compile_mrt(unsigned shape, unsigned targets)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "reconverged MRT tail");
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.no_preamble = true;
   key.reserved_preamble = 8;
   key.fs.nr_samples = 1;
   nir_def *input = nir_load_preamble(&b, 1, 32, .base = 4);
   nir_def *value = input;
   if (shape == 1) {
      nir_push_if(&b, nir_ilt_imm(&b, input, 8));
      nir_def *a = input;
      for (unsigned i = 0; i < 8; ++i)
         a = nir_fadd(
            &b, a,
            nir_load_global_constant(
               &b, 1, 32, nir_imm_int64(&b, 0x10000 + 4 * i), .align_mul = 4));
      nir_push_else(&b, NULL);
      nir_def *z = input;
      for (unsigned i = 0; i < 8; ++i)
         z = nir_fmul(
            &b, z,
            nir_load_global_constant(
               &b, 1, 32, nir_imm_int64(&b, 0x20000 + 4 * i), .align_mul = 4));
      nir_pop_if(&b, NULL);
      value = nir_if_phi(&b, a, z);
   } else if (shape == 2) {
      nir_variable *counter =
         nir_local_variable_create(b.impl, glsl_uint_type(), "counter");
      nir_variable *sum =
         nir_local_variable_create(b.impl, glsl_float_type(), "sum");
      nir_store_var(&b, counter, nir_imm_int(&b, 0), 1);
      nir_store_var(&b, sum, nir_imm_float(&b, 0), 1);
      nir_push_loop(&b);
      nir_def *i = nir_load_var(&b, counter);
      nir_push_if(&b, nir_uge(&b, i, input));
      nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, NULL);
      nir_def *addr =
         nir_iadd_imm(&b, nir_u2u64(&b, nir_ishl_imm(&b, i, 2)), 0x10000);
      nir_def *v = nir_load_global_constant(&b, 1, 32, addr, .align_mul = 4);
      nir_store_var(&b, sum, nir_fadd(&b, nir_load_var(&b, sum), v), 1);
      nir_store_var(&b, counter, nir_iadd_imm(&b, i, 1), 1);
      nir_pop_loop(&b, NULL);
      value = nir_load_var(&b, sum);
   }
   for (unsigned i = 0; i < targets; ++i) {
      nir_def *v = nir_f2f16(&b, value);
      nir_store_local_pixel_agx(
         &b, nir_vec4(&b, v, v, v, v), nir_imm_intN_t(&b, 0xff, 16),
         nir_undef(&b, 2, 16), .base = int(8 * i), .write_mask = 0xf,
         .format = PIPE_FORMAT_R16_FLOAT);
      b.shader->info.outputs_written |= BITFIELD64_BIT(FRAG_RESULT_DATA0 + i);
      BITSET_SET(key.g17p_tile_partitions, 8 * i);
   }
   agx_preprocess_nir(b.shader);
   agx_shader_part part = {};
   bool ok = agx_compile_shader_nir(b.shader, &key, &part);
   free(part.binary);
   return ok;
}

TEST(MRTTerminalTail, ExistingStraightLine)
{
   for (unsigned n : {2u, 4u, 8u})
      EXPECT_TRUE(compile_mrt(0, n));
}
TEST(MRTTerminalTail, ReconvergedBranch)
{
   for (unsigned n : {2u, 4u, 8u})
      EXPECT_TRUE(compile_mrt(1, n));
}
TEST(MRTTerminalTail, ReconvergedLoop)
{
   for (unsigned n : {2u, 4u, 8u})
      EXPECT_TRUE(compile_mrt(2, n));
}

/* These test only the shared eligibility proof, not packing instruction fields.
 */
struct TailFixture {
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_context *ctx = b->shader;
   agx_shader_key key = {};
   agx_block *prefix = list_first_entry(&ctx->blocks, agx_block, link);
   agx_block *tail = agx_test_block(ctx);
   TailFixture()
   {
      nir_builder nb = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "tail proof");
      ctx->nir = nb.shader;
      nir_index_blocks(nb.impl);
      ctx->indexed_nir_blocks =
         rzalloc_array(ctx, agx_block *, nb.impl->num_blocks);
      ctx->indexed_nir_blocks[nir_impl_last_block(nb.impl)->index] = tail;
      ctx->stage = MESA_SHADER_FRAGMENT;
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.fs.nr_samples = 1;
      ctx->key = &key;
      prefix->successors[0] = tail;
      op(prefix, AGX_OPCODE_DEVICE_LOAD);
      op(tail, AGX_OPCODE_WAIT_PIX)->writeout = 12;
      op(tail, AGX_OPCODE_ST_TILE);
      op(tail, AGX_OPCODE_STOP);
   }
   agx_instr *op(agx_block *block, enum agx_opcode opcode)
   {
      b->cursor = agx_after_block(block);
      agx_instr *i = agx_alloc_instr(b, opcode, 0, 0);
      agx_builder_insert(&b->cursor, i);
      return i;
   }
   agx_instr *before_stop(enum agx_opcode opcode)
   {
      b->cursor = agx_before_instr(
         list_last_entry(&tail->instructions, agx_instr, link));
      agx_instr *i = agx_alloc_instr(b, opcode, 0, 0);
      agx_builder_insert(&b->cursor, i);
      return i;
   }
   ~TailFixture()
   {
      ralloc_free(ctx->nir);
      ralloc_free(mem);
   }
};
TEST(MRTTerminalTail, EligibilityRejectsUnprovenPrefixOrExit)
{
   {
      TailFixture f;
      EXPECT_EQ(agx_reconverged_mrt_tail(f.ctx), f.tail);
   }
   for (enum agx_opcode op :
        {AGX_OPCODE_ST_TILE, AGX_OPCODE_LD_TILE, AGX_OPCODE_WAIT_PIX,
         AGX_OPCODE_SIGNAL_PIX, AGX_OPCODE_STOP, AGX_OPCODE_DEVICE_STORE,
         AGX_OPCODE_ATOMIC, AGX_OPCODE_LOCAL_ATOMIC, AGX_OPCODE_TRAP,
         AGX_OPCODE_MEMORY_BARRIER, AGX_OPCODE_ZS_EMIT_AGX3,
         AGX_OPCODE_SAMPLE_MASK_AGX3}) {
      SCOPED_TRACE(op);
      TailFixture f;
      f.op(f.prefix, op);
      EXPECT_EQ(agx_reconverged_mrt_tail(f.ctx), nullptr);
   }
   for (enum agx_opcode op : {AGX_OPCODE_ATOMIC, AGX_OPCODE_LOCAL_ATOMIC}) {
      TailFixture f;
      f.b->cursor = agx_after_block(f.prefix);
      agx_instr *i = agx_alloc_instr(f.b, op, 1, 3);
      i->dest[0] = agx_register(0, AGX_SIZE_32);
      agx_builder_insert(&f.b->cursor, i);
      EXPECT_EQ(agx_reconverged_mrt_tail(f.ctx), nullptr);
   }
   for (enum agx_opcode op :
        {AGX_OPCODE_ZS_EMIT_AGX3, AGX_OPCODE_SAMPLE_MASK_AGX3,
         AGX_OPCODE_SIGNAL_PIX, AGX_OPCODE_DEVICE_STORE,
         AGX_OPCODE_MEMORY_BARRIER}) {
      TailFixture f;
      f.before_stop(op);
      EXPECT_EQ(agx_reconverged_mrt_tail(f.ctx), nullptr);
   }
   {
      TailFixture f;
      f.before_stop(AGX_OPCODE_WAIT_PIX)->writeout = 1;
      EXPECT_EQ(agx_reconverged_mrt_tail(f.ctx), nullptr);
   }
   {
      TailFixture f;
      f.prefix->successors[0] = nullptr;
      EXPECT_EQ(agx_reconverged_mrt_tail(f.ctx), nullptr);
   }
   {
      TailFixture f;
      f.tail->successors[0] = f.prefix;
      EXPECT_EQ(agx_reconverged_mrt_tail(f.ctx), nullptr);
   }
   {
      TailFixture f;
      f.ctx->indexed_nir_blocks[0] = f.prefix;
      EXPECT_EQ(agx_reconverged_mrt_tail(f.ctx), nullptr);
   }
   {
      TailFixture f;
      f.key.fs.nr_samples = 4;
      EXPECT_EQ(agx_reconverged_mrt_tail(f.ctx), nullptr);
   }
   {
      TailFixture f;
      f.key.no_stop = true;
      EXPECT_EQ(agx_reconverged_mrt_tail(f.ctx), nullptr);
   }
   {
      TailFixture f;
      f.op(f.tail, AGX_OPCODE_MOV);
      EXPECT_EQ(agx_reconverged_mrt_tail(f.ctx), nullptr);
   }
}
