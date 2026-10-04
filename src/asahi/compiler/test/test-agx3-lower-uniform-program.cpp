/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "agx_test.h"
#include "agx_debug.h"
#include "asahi/lib/agx_ioto_preshader_core.h"
#include <gtest/gtest.h>
extern "C" {
#include "agx3_graphics_program.h"
#include "agx3_static_spill_core.h"
#include "agx_nir.h"
#include "compiler/nir/nir_builder.h"
}

class AGX3UniformProgram : public testing::Test {
protected:
   void *memctx = ralloc_context(NULL);
   agx_shader_key key = {};
   agx_builder *b = agx_test_builder(memctx);
   AGX3UniformProgram()
   {
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      b->shader->key = &key;
      b->shader->is_preamble = true;
   }
   ~AGX3UniformProgram() { ralloc_free(memctx); }

   agx_index input(unsigned half)
   {
      agx_index value = agx_temp(b->shader, AGX_SIZE_32);
      agx_preload_to(b, value, agx_register(half, AGX_SIZE_32));
      return value;
   }
};

class AGX3CProgImports : public AGX3UniformProgram {
protected:
   agx_shader_info info = {};
   AGX3CProgImports()
   {
      b->shader->out = &info;
      b->shader->stage = MESA_SHADER_COMPUTE;
      b->shader->nir = rzalloc(memctx, nir_shader);
      b->shader->nir->info.stage = MESA_SHADER_COMPUTE;
      for (unsigned d = 0; d < 3; ++d)
         b->shader->nir->info.workgroup_size[d] = 1;
   }
};

TEST_F(AGX3CProgImports, SparseWordsRetainLogicalIdentityAfterDCE)
{
   agx_mov(b, agx_uniform(1, AGX_SIZE_16)); /* Deliberately dead. */
   agx_index source = agx_uniform(4, AGX_SIZE_32);
   source.abs = source.neg = true;
   agx_instr *use = agx_uniform_store(b, source, agx_immediate(120), 1);
   agx_stop(b);
   ASSERT_TRUE(agx3_lower_cprog_inputs(b->shader));
   EXPECT_TRUE(use->src[0].abs);
   EXPECT_TRUE(use->src[0].neg);
   EXPECT_EQ(info.agx3_preamble_inputs.count, 3u);
   EXPECT_EQ(info.agx3_preamble_inputs.halves[0], UINT64_C(0x32));
   EXPECT_EQ(b->shader->scratch_size_B, 32u);
   agx_dce(b->shader, false);
   unsigned loads = 0;
   agx_foreach_instr_global(b->shader, I) {
      if (I->op != AGX_OPCODE_LOAD_CPROG_AGX3)
         continue;
      EXPECT_EQ(I->imm, 4u + loads);
      EXPECT_EQ(I->dest[0].size, AGX_SIZE_16);
      uint16_t tps = 0;
      ASSERT_TRUE(agx3_cprog_input_tps_half(&info.agx3_preamble_inputs,
                                          I->imm, &tps));
      EXPECT_EQ(tps, 9u + loads);
      ++loads;
   }
   EXPECT_EQ(loads, 2u);
   EXPECT_EQ(info.agx3_preamble_inputs.count, 3u);
   EXPECT_FALSE(agx3_lower_cprog_inputs(b->shader));
}

TEST_F(AGX3CProgImports, PointerImportsSurviveCSEAllocationAndNativePacking)
{
   for (unsigned i = 0; i < 2; ++i) {
      agx_index value = agx_temp(b->shader, AGX_SIZE_32);
      agx_device_load_to(b, value, agx_uniform(0, AGX_SIZE_64),
                        agx_immediate(i * 4), AGX_FORMAT_I32, 1, 0, false);
      agx_uniform_store(b, value, agx_immediate(120 + 2 * i), 1);
   }
   agx_stop(b);
   ASSERT_TRUE(agx3_lower_cprog_inputs(b->shader));
   agx_opt_cse(b->shader);
   agx_dce(b->shader, false);
   agx_pressure_schedule(b->shader);
   agx_validate(b->shader, "native CProg imports before RA");
   agx_ra(b->shader);
   agx_lower_pseudo(b->shader);
   agx_opt_register_cache(b->shader);
   unsigned imports = 0;
   agx_foreach_instr_global(b->shader, I) {
      agx_foreach_src(I, s) { EXPECT_NE(I->src[s].type, AGX_INDEX_UNIFORM); }
      if (I->op != AGX_OPCODE_LOAD_CPROG_AGX3)
         continue;
      EXPECT_EQ(I->dest[0].type, AGX_INDEX_REGISTER);
      EXPECT_EQ(I->dest[0].size, AGX_SIZE_32);
      EXPECT_EQ(I->imm & 1, 0u);
      ++imports;
   }
   EXPECT_EQ(imports, 2u);
   EXPECT_EQ(b->shader->scratch_size_B, 32u);
   struct util_dynarray binary = UTIL_DYNARRAY_INIT;
   ASSERT_TRUE(agx3_pack_binary(b->shader, &binary));
   ASSERT_GE(binary.size, 8u);
   struct agx3_static_spill_load_desc first = {};
   ASSERT_TRUE(agx3_static_spill_decode_load(AGX_ISA_ENCODING_AGX3B,
                  (const uint8_t *)binary.data, 8, &first));
   EXPECT_EQ(first.wait_mask, 0x20u);
   EXPECT_EQ(first.tps_half, 8u);
   EXPECT_EQ(first.width, AGX3_MEM_W32);
   util_dynarray_fini(&binary);
}

TEST_F(AGX3CProgImports, SparseUnalignedWordSurvivesAllocationAndPacking)
{
   agx_mov(b, agx_uniform(1, AGX_SIZE_16));
   agx_index value = agx_fadd(b, agx_uniform(4, AGX_SIZE_32), input(0));
   agx_uniform_store(b, value, agx_immediate(120), 1);
   agx_stop(b);
   ASSERT_TRUE(agx3_lower_cprog_inputs(b->shader));
   ASSERT_TRUE(agx3_lower_uniform_program(b->shader));
   agx_dce(b->shader, false);
   agx_pressure_schedule(b->shader);
   agx_validate(b->shader, "sparse CProg imports before RA");
   agx_ra(b->shader);
   agx_lower_pseudo(b->shader);
   agx_opt_register_cache(b->shader);
   unsigned imports = 0;
   agx_foreach_instr_global(b->shader, I) {
      if (I->op == AGX_OPCODE_LOAD_CPROG_AGX3) {
         EXPECT_EQ(I->dest[0].size, AGX_SIZE_16);
         ++imports;
      }
   }
   EXPECT_EQ(imports, 2u);
   struct util_dynarray binary = UTIL_DYNARRAY_INIT;
   EXPECT_TRUE(agx3_pack_binary(b->shader, &binary));
   EXPECT_GT(binary.size, 0u);
   util_dynarray_fini(&binary);
}

TEST_F(AGX3CProgImports, LocalScratchIsRebasedBeyondImports)
{
   b->shader->scratch_size_B = 64;
   agx_index value = agx_mov(b, agx_uniform(4, AGX_SIZE_32));
   agx_instr *store = agx_stack_store(b, value, agx_immediate(12),
                                     AGX_FORMAT_I32, 1);
   agx_index result = agx_temp(b->shader, AGX_SIZE_32);
   agx_instr *load = agx_stack_load_to(b, result, input(0), AGX_FORMAT_I32, 1);
   agx_index original = load->src[0];
   ASSERT_TRUE(agx3_lower_cprog_inputs(b->shader));
   EXPECT_EQ(b->shader->scratch_size_B, 96u);
   EXPECT_EQ(store->src[1].value, 44u);
   EXPECT_FALSE(agx_is_equiv(load->src[0], original));
   agx_instr *add = list_entry(load->link.prev, agx_instr, link);
   EXPECT_EQ(add->op, AGX_OPCODE_IADD);
   EXPECT_TRUE(agx_is_equiv(add->src[0], original));
   EXPECT_TRUE(agx_is_equiv(add->dest[0], load->src[0]));
}

TEST_F(AGX3CProgImports, InvalidScratchRebaseLeavesIRAndMetadataUntouched)
{
   agx_instr *read = agx_mov_to(b, agx_temp(b->shader, AGX_SIZE_32),
                              agx_uniform(0, AGX_SIZE_32));
   agx_instr *store = agx_stack_store(b, read->dest[0], agx_immediate(65520),
                                     AGX_FORMAT_I32, 1);
   const unsigned instructions = list_length(&agx_start_block(b->shader)->instructions);
   EXPECT_FALSE(agx3_lower_cprog_inputs(b->shader));
   EXPECT_EQ(read->src[0].type, AGX_INDEX_UNIFORM);
   EXPECT_EQ(store->src[1].value, 65520u);
   EXPECT_EQ(list_length(&agx_start_block(b->shader)->instructions), instructions);
   EXPECT_EQ(info.agx3_preamble_inputs.count, 0u);
   EXPECT_FALSE(b->shader->agx3_cprog_inputs_lowered);
   EXPECT_EQ(b->shader->scratch_size_B, 0u);
}

TEST_F(AGX3CProgImports, ReadWriteAliasingIsRefusedBeforeMutation)
{
   agx_instr *read = agx_mov_to(b, agx_temp(b->shader, AGX_SIZE_32),
                              agx_uniform(4, AGX_SIZE_32));
   agx_uniform_store(b, read->dest[0], agx_immediate(5), 1);
   EXPECT_FALSE(agx3_lower_cprog_inputs(b->shader));
   EXPECT_EQ(read->src[0].type, AGX_INDEX_UNIFORM);
   EXPECT_FALSE(b->shader->agx3_cprog_inputs_lowered);
   EXPECT_EQ(info.agx3_preamble_inputs.count, 0u);
}

TEST_F(AGX3CProgImports, PhiImportsAreDefinedOnTheirPredecessorEdges)
{
   agx_context *ctx = b->shader;
   agx_block *left = agx_start_block(ctx);
   agx_block *right = agx_test_block(ctx);
   agx_block *join = agx_test_block(ctx);
   left->successors[0] = right;
   left->successors[1] = join;
   right->successors[0] = join;
   util_dynarray_append(&right->predecessors, left);
   util_dynarray_append(&join->predecessors, left);
   util_dynarray_append(&join->predecessors, right);
   agx_builder jb = agx_init_builder(ctx, agx_after_block(join));
   agx_instr *phi = agx_phi_to(&jb, agx_temp(ctx, AGX_SIZE_32), 2);
   phi->src[0] = agx_uniform(0, AGX_SIZE_32);
   phi->src[1] = agx_uniform(2, AGX_SIZE_32);
   agx_uniform_store(&jb, phi->dest[0], agx_immediate(120), 1);
   agx_stop(&jb);
   ASSERT_TRUE(agx3_lower_cprog_inputs(ctx));
   agx_block *preds[] = {left, right};
   for (unsigned s = 0; s < 2; ++s) {
      ASSERT_EQ(list_length(&preds[s]->instructions), 1u);
      agx_instr *load = list_first_entry(&preds[s]->instructions, agx_instr, link);
      EXPECT_EQ(load->op, AGX_OPCODE_LOAD_CPROG_AGX3);
      EXPECT_EQ(load->imm, 2u * s);
      EXPECT_TRUE(agx_is_equiv(load->dest[0], phi->src[s]));
   }
   agx_validate(ctx, "native CProg phi imports");
}

TEST_F(AGX3CProgImports, FrozenInputsPreventPaddingOverLiveUniformHalves)
{
   agx_index imported = agx_mov(b, agx_uniform(249, AGX_SIZE_16));
   agx_uniform_store(b, imported, agx_immediate(120), 1);
   agx_index half = agx_temp(b->shader, AGX_SIZE_16);
   agx_preload_to(b, half, agx_register(1, AGX_SIZE_16));
   agx_instr *store = agx_uniform_store(b, half, agx_immediate(248), 1);
   ASSERT_TRUE(agx3_lower_cprog_inputs(b->shader));
   agx3_lower_uniform_half_exports(b->shader);
   EXPECT_EQ(store->src[0].size, AGX_SIZE_16);
}

TEST_F(AGX3CProgImports, OnlyAGX3BPreamblesCanImport)
{
   agx_instr *read = agx_mov_to(b, agx_temp(b->shader, AGX_SIZE_32),
                              agx_uniform(0, AGX_SIZE_32));
   key.dev.encoding = AGX_ISA_ENCODING_AGX2;
   EXPECT_FALSE(agx3_lower_cprog_inputs(b->shader));
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   b->shader->is_preamble = false;
   EXPECT_FALSE(agx3_lower_cprog_inputs(b->shader));
   EXPECT_EQ(read->src[0].type, AGX_INDEX_UNIFORM);
   EXPECT_EQ(info.agx3_preamble_inputs.count, 0u);
}

TEST_F(AGX3CProgImports, TerminalExportsStayContiguous)
{
   agx_instr *first = agx_export(b, agx_uniform(0, AGX_SIZE_32), 0);
   agx_instr *second = agx_export(b, agx_uniform(2, AGX_SIZE_32), 2);
   ASSERT_TRUE(agx3_lower_cprog_inputs(b->shader));
   EXPECT_EQ(first->link.next, &second->link);
   EXPECT_EQ(second->link.next, &agx_start_block(b->shader)->instructions);
   agx_instr *before = list_entry(first->link.prev, agx_instr, link);
   EXPECT_EQ(before->op, AGX_OPCODE_LOAD_CPROG_AGX3);
   EXPECT_EQ(before->imm, 2u);
   agx_validate(b->shader, "native CProg imports before terminal exports");
}

TEST_F(AGX3CProgImports, RepeatedImportsRetainTPSUntilLastRead)
{
   agx_mov_to(b, agx_register(0, AGX_SIZE_32), agx_uniform(4, AGX_SIZE_32));
   agx_mov_to(b, agx_register(2, AGX_SIZE_32), agx_uniform(4, AGX_SIZE_32));
   agx_stop(b);
   ASSERT_TRUE(agx3_lower_cprog_inputs(b->shader));
   /* Fix physical destinations while deliberately retaining both reads.
    * CSE normally removes this straight-line duplicate, but separate CFG
    * paths and spill decisions can preserve repeated reads. */
   unsigned dest = 0;
   agx_foreach_instr_global_safe(b->shader, I) {
      if (I->op == AGX_OPCODE_MOV) {
         agx_remove_instruction(I);
      } else if (I->op == AGX_OPCODE_LOAD_CPROG_AGX3) {
         I->dest[0] = agx_register(dest, AGX_SIZE_32);
         dest += 2;
      }
   }
   b->shader->max_reg = 3;
   struct util_dynarray binary = UTIL_DYNARRAY_INIT;
   ASSERT_TRUE(agx3_pack_binary(b->shader, &binary));
   ASSERT_GE(binary.size, 12u);
   struct agx3_static_spill_load_desc first = {}, second = {};
   ASSERT_TRUE(agx3_static_spill_decode_load(AGX_ISA_ENCODING_AGX3B,
                  (const uint8_t *)binary.data, 8, &first));
   ASSERT_TRUE(agx3_static_spill_decode_load(AGX_ISA_ENCODING_AGX3B,
                  (const uint8_t *)binary.data + 8, 4, &second));
   EXPECT_EQ(first.wait_mask, 0x20u);
   EXPECT_EQ(second.wait_mask, 0u);
   EXPECT_EQ(first.tps_half, 8u);
   EXPECT_EQ(second.tps_half, 8u);
   EXPECT_TRUE(first.tps_live);
   EXPECT_FALSE(second.tps_live);
   util_dynarray_fini(&binary);
}

TEST_F(AGX3UniformProgram, UniformInputsExcludeEarlierDefinitions)
{
   agx_uniform_fadd_agx3(b, input(0), input(2), 24);
   agx_mov(b, agx_uniform(24, AGX_SIZE_32));
   bool inputs = true;
   ASSERT_TRUE(agx3_analyze_preamble_uniform_inputs(b->shader, &inputs));
   EXPECT_FALSE(inputs);

   agx_mov(b, agx_uniform(0, AGX_SIZE_64));
   ASSERT_TRUE(agx3_analyze_preamble_uniform_inputs(b->shader, &inputs));
   EXPECT_TRUE(inputs);
}

TEST_F(AGX3UniformProgram, UniformInputsPreserveReadBeforeWriteAndPartialWords)
{
   agx_index half = agx_register(0, AGX_SIZE_16);
   agx_uniform_store(b, half, agx_immediate(24), 1);
   agx_instr *read = agx_mov_to(b, agx_temp(b->shader, AGX_SIZE_32),
                              agx_uniform(24, AGX_SIZE_32));
   bool inputs = false;
   ASSERT_TRUE(agx3_analyze_preamble_uniform_inputs(b->shader, &inputs));
   EXPECT_TRUE(inputs); /* Upper half is an input. */
   agx_uniform_store(b, half, agx_immediate(25), 1);
   ASSERT_TRUE(agx3_analyze_preamble_uniform_inputs(b->shader, &inputs));
   EXPECT_TRUE(inputs); /* A later write cannot initialize this read. */
   b->cursor = agx_before_instr(read);
   agx_uniform_store(b, half, agx_immediate(25), 1);
   ASSERT_TRUE(agx3_analyze_preamble_uniform_inputs(b->shader, &inputs));
   EXPECT_FALSE(inputs);
}

TEST_F(AGX3UniformProgram, UniformInputsThroughBranchesAndLoops)
{
   agx_context *ctx = b->shader;
   agx_block *entry = agx_start_block(ctx);
   agx_block *left = agx_test_block(ctx);
   agx_block *right = agx_test_block(ctx);
   agx_block *join = agx_test_block(ctx);
   entry->successors[0] = left;
   entry->successors[1] = right;
   left->successors[0] = right->successors[0] = join;
   agx_builder lb = agx_init_builder(ctx, agx_after_block(left));
   agx_builder rb = agx_init_builder(ctx, agx_after_block(right));
   agx_builder jb = agx_init_builder(ctx, agx_after_block(join));
   agx_uniform_store(&lb, agx_register(0, AGX_SIZE_32), agx_immediate(24), 1);
   agx_mov(&jb, agx_uniform(24, AGX_SIZE_32));
   bool inputs = false;
   ASSERT_TRUE(agx3_analyze_preamble_uniform_inputs(ctx, &inputs));
   EXPECT_TRUE(inputs);
   agx_uniform_store(&rb, agx_register(0, AGX_SIZE_32), agx_immediate(24), 1);
   ASSERT_TRUE(agx3_analyze_preamble_uniform_inputs(ctx, &inputs));
   EXPECT_FALSE(inputs);

   /* Enter at the read, with definitions only on the back edge. */
   entry->successors[0] = join;
   entry->successors[1] = NULL;
   join->successors[0] = left;
   ASSERT_TRUE(agx3_analyze_preamble_uniform_inputs(ctx, &inputs));
   EXPECT_TRUE(inputs);
   agx_uniform_store(b, agx_register(0, AGX_SIZE_32), agx_immediate(24), 1);
   ASSERT_TRUE(agx3_analyze_preamble_uniform_inputs(ctx, &inputs));
   EXPECT_FALSE(inputs);
}

TEST_F(AGX3UniformProgram, UniformInputAnalysisRefusesMalformedRanges)
{
   agx_mov(b, agx_uniform(511, AGX_SIZE_32));
   bool inputs = true;
   EXPECT_FALSE(agx3_analyze_preamble_uniform_inputs(b->shader, &inputs));
   EXPECT_TRUE(inputs);
   EXPECT_FALSE(agx3_analyze_preamble_uniform_inputs(b->shader, NULL));
}

TEST_F(AGX3UniformProgram, ArithmeticExportsUseNativeDestination)
{
   agx_index a = input(0), c = input(2), d = input(4);
   agx_index add = agx_fadd(b, a, c);
   agx_index mul = agx_fmul(b, a, c);
   agx_index fma = agx_ffma(b, a, c, d);
   agx_uniform_store(b, add, agx_immediate(4), 1);
   agx_uniform_store(b, mul, agx_immediate(6), 1);
   agx_uniform_store(b, fma, agx_immediate(8), 1);
   ASSERT_TRUE(agx3_lower_uniform_program(b->shader));
   agx_dce(b->shader, false);
   unsigned exports = 0;
   agx_foreach_instr_global(b->shader, I) {
      EXPECT_NE(I->op, AGX_OPCODE_UNIFORM_STORE);
      EXPECT_NE(I->op, AGX_OPCODE_FADD);
      EXPECT_NE(I->op, AGX_OPCODE_FMUL);
      EXPECT_NE(I->op, AGX_OPCODE_FFMA);
      if (I->op == AGX_OPCODE_PRELOAD)
         continue;
      EXPECT_EQ(I->nr_dests, 0);
      EXPECT_EQ(I->imm, 4u + 2 * exports);
      EXPECT_EQ(I->op, exports == 0 ? AGX_OPCODE_UNIFORM_FADD_AGX3 :
                       exports == 1 ? AGX_OPCODE_UNIFORM_FMUL_AGX3 :
                                      AGX_OPCODE_UNIFORM_FFMA_AGX3);
      ++exports;
   }
   EXPECT_EQ(exports, 3u);
   agx_validate(b->shader, "native constant-program destinations");
   EXPECT_FALSE(agx3_lower_uniform_program(b->shader));
}

TEST_F(AGX3UniformProgram, PaddedHalfHasARealWordOperandBeforeAllocation)
{
   struct agx_shader_info info = {};
   info.push_count = 249;
   b->shader->out = &info;
   b->shader->stage = MESA_SHADER_COMPUTE;
   b->shader->nir = rzalloc(memctx, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_COMPUTE;
   for (unsigned d = 0; d < 3; ++d)
      b->shader->nir->info.workgroup_size[d] = 1;
   agx_index half = agx_temp(b->shader, AGX_SIZE_16);
   agx_preload_to(b, half, agx_register(1, AGX_SIZE_16));
   agx_instr *store = agx_uniform_store(b, half, agx_immediate(248), 1);
   agx_stop(b);

   ASSERT_TRUE(agx3_lower_uniform_half_exports(b->shader));
   EXPECT_EQ(info.push_count, 250u);
   ASSERT_EQ(store->src[0].size, AGX_SIZE_32);
   unsigned collects = 0;
   agx_foreach_instr_global(b->shader, I) {
      if (I->op != AGX_OPCODE_COLLECT)
         continue;
      ++collects;
      EXPECT_TRUE(agx_is_equiv(I->src[0], half));
      EXPECT_EQ(I->src[1].type, AGX_INDEX_IMMEDIATE);
      EXPECT_EQ(I->src[1].value, 0u);
      EXPECT_EQ(I->src[1].size, AGX_SIZE_16);
   }
   EXPECT_EQ(collects, 1u);
   EXPECT_FALSE(agx3_lower_uniform_half_exports(b->shader));
   agx_dce(b->shader, false);
   agx_pressure_schedule(b->shader);
   agx_validate(b->shader, "native half export word before allocation");
   agx_ra(b->shader);
   agx_lower_pseudo(b->shader);
   agx_opt_register_cache(b->shader);
   EXPECT_EQ(store->src[0].type, AGX_INDEX_REGISTER);
   EXPECT_EQ(store->src[0].size, AGX_SIZE_32);
   EXPECT_EQ(store->src[0].value & 1, 0u);
   EXPECT_EQ(b->shader->scratch_size_B, 0u);
   struct util_dynarray binary = UTIL_DYNARRAY_INIT;
   EXPECT_TRUE(agx3_pack_binary(b->shader, &binary));
   EXPECT_GT(binary.size, 0u);
   util_dynarray_fini(&binary);
}

TEST_F(AGX3UniformProgram, HalfExportDoesNotOverwriteOtherUniformData)
{
   struct agx_shader_info info = {};
   info.rodata.base_uniform = 49;
   info.rodata.size_16 = 1;
   b->shader->out = &info;
   agx_index half = agx_temp(b->shader, AGX_SIZE_16);
   agx_preload_to(b, half, agx_register(1, AGX_SIZE_16));
   /* A real neighbouring export, an imported half read, and reserved ABI
    * padding each prohibit widening, for either lane of the word. */
   key.reserved_preamble = 26;
   agx_uniform_store(b, half, agx_immediate(24), 1);
   agx_uniform_store(b, half, agx_immediate(32), 1);
   agx_uniform_store(b, half, agx_immediate(33), 1);
   agx_uniform_store(b, half, agx_immediate(40), 1);
   agx_index read = agx_mov(b, agx_uniform(41, AGX_SIZE_16));
   agx_mov(b, agx_uniform(42, AGX_SIZE_16));
   agx_uniform_store(b, read, agx_immediate(43), 1);
   agx_uniform_store(b, half, agx_immediate(48), 1);
   EXPECT_FALSE(agx3_lower_uniform_half_exports(b->shader));
   agx_foreach_instr_global(b->shader, I)
      EXPECT_NE(I->op, AGX_OPCODE_COLLECT);
}

TEST_F(AGX3UniformProgram, NativeUniformWritesBlockHalfPairCoalescing)
{
   b->shader->allocated_vec = _mesa_hash_table_u64_create(b->shader);
   agx_index a = input(0), c = input(2);
   agx_index half = agx_temp(b->shader, AGX_SIZE_16);
   agx_preload_to(b, half, agx_register(5, AGX_SIZE_16));
   agx_uniform_store(b, half, agx_immediate(32), 1);
   agx_uniform_fadd_agx3(b, a, c, 32);
   agx_uniform_store(b, half, agx_immediate(33), 1);
   EXPECT_FALSE(agx3_coalesce_uniform_half_stores(b->shader));
   EXPECT_FALSE(agx3_lower_uniform_half_exports(b->shader));
   agx_validate(b->shader, "half exports around native uniform overwrite");
}

TEST_F(AGX3UniformProgram, HalfExportSelectionIsPreambleAndAGX3BOnly)
{
   agx_index half = agx_temp(b->shader, AGX_SIZE_16);
   agx_preload_to(b, half, agx_register(1, AGX_SIZE_16));
   agx_uniform_store(b, half, agx_immediate(248), 1);
   b->shader->is_preamble = false;
   EXPECT_FALSE(agx3_lower_uniform_half_exports(b->shader));
   b->shader->is_preamble = true;
   for (auto enc : {AGX_ISA_ENCODING_AGX2, AGX_ISA_ENCODING_AGX3C}) {
      key.dev.encoding = enc;
      EXPECT_FALSE(agx3_lower_uniform_half_exports(b->shader));
   }
}

TEST_F(AGX3UniformProgram, SharedValuesAndHighExportsKeepGeneralPath)
{
   agx_index a = input(0), c = input(2);
   agx_index shared = agx_fadd(b, a, c);
   agx_uniform_store(b, shared, agx_immediate(4), 1);
   agx_uniform_store(b, shared, agx_immediate(6), 1);
   agx_uniform_store(b, agx_fmul(b, a, c), agx_immediate(128), 1);
   agx_uniform_store(b, agx_fmul(b, a, c), agx_immediate(9), 1);
   EXPECT_FALSE(agx3_lower_uniform_program(b->shader));
}

TEST_F(AGX3UniformProgram, CollectedVectorExportsBecomeScalarNativeWrites)
{
   agx_index a = input(0), c = input(2), d = input(4);
   agx_index lanes[] = {agx_fadd(b, a, c), agx_fmul(b, a, c),
                        agx_ffma(b, a, c, d), agx_fadd(b, c, d)};
   agx_index vector = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
   agx_instr *collect = agx_collect_to(b, vector, 4);
   for (unsigned i = 0; i < 4; ++i)
      collect->src[i] = lanes[i];
   agx_uniform_store(b, vector, agx_immediate(120), 15);
   ASSERT_TRUE(agx3_lower_uniform_program(b->shader));
   agx_dce(b->shader, false);
   unsigned writes = 0;
   agx_foreach_instr_global(b->shader, I) {
      if (I->op == AGX_OPCODE_PRELOAD)
         continue;
      EXPECT_EQ(I->nr_dests, 0);
      EXPECT_EQ(I->imm, 120u + 2 * writes);
      EXPECT_NE(I->op, AGX_OPCODE_COLLECT);
      EXPECT_NE(I->op, AGX_OPCODE_UNIFORM_STORE);
      ++writes;
   }
   EXPECT_EQ(writes, 4u);
   agx_validate(b->shader, "vector constant-program destinations");
}

TEST_F(AGX3UniformProgram, MainProgramAndOtherIsasStayUnchanged)
{
   agx_index a = input(0), c = input(2);
   agx_uniform_store(b, agx_fadd(b, a, c), agx_immediate(4), 1);
   b->shader->is_preamble = false;
   EXPECT_FALSE(agx3_lower_uniform_program(b->shader));
   b->shader->is_preamble = true;
   for (auto enc : {AGX_ISA_ENCODING_AGX2, AGX_ISA_ENCODING_AGX3C}) {
      key.dev.encoding = enc;
      EXPECT_FALSE(agx3_lower_uniform_program(b->shader));
   }
}

TEST_F(AGX3UniformProgram, SchedulingPreservesUniformReadWriteHazards)
{
   agx_index a = input(0), c = input(2), d = input(4);
   agx_index before = agx_temp(b->shader, AGX_SIZE_32);
   agx_instr *read_before = agx_mov_to(b, before, agx_uniform(6, AGX_SIZE_32));
   agx_instr *first = agx_uniform_fadd_agx3(b, a, c, 6);
   agx_index after = agx_temp(b->shader, AGX_SIZE_32);
   agx_instr *read_after = agx_mov_to(b, after, agx_uniform(6, AGX_SIZE_32));
   agx_instr *second = agx_uniform_fmul_agx3(b, c, d, 6);
   agx_uniform_store(b, before, agx_immediate(20), 1);
   agx_uniform_store(b, after, agx_immediate(22), 1);

   agx_pressure_schedule(b->shader);
   unsigned pos = 0, rb = ~0u, w0 = ~0u, ra = ~0u, w1 = ~0u;
   agx_foreach_instr_global(b->shader, I) {
      if (I == read_before)
         rb = pos;
      if (I == first)
         w0 = pos;
      if (I == read_after)
         ra = pos;
      if (I == second)
         w1 = pos;
      ++pos;
   }
   EXPECT_LT(rb, w0);
   EXPECT_LT(w0, ra);
   EXPECT_LT(ra, w1);
   agx_validate(b->shader, "scheduled native uniform hazards");
}

TEST_F(AGX3UniformProgram, AsyncInputsSurviveSchedulingAllocationAndPacking)
{
   b->shader->stage = MESA_SHADER_COMPUTE;
   b->shader->nir = rzalloc(memctx, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_COMPUTE;
   for (unsigned d = 0; d < 3; ++d)
      b->shader->nir->info.workgroup_size[d] = 1;

   agx_index inputs[3];
   for (unsigned i = 0; i < 3; ++i) {
      inputs[i] = agx_temp(b->shader, AGX_SIZE_32);
      agx_device_load_to(b, inputs[i], agx_uniform(0, AGX_SIZE_64),
                        agx_immediate(i * 4), AGX_FORMAT_I32, 1, 0, false);
   }
   agx_uniform_store(b, agx_fadd(b, inputs[0], inputs[1]), agx_immediate(120), 1);
   agx_uniform_store(b, agx_fmul(b, inputs[1], inputs[2]), agx_immediate(122), 1);
   agx_uniform_store(b, agx_ffma(b, inputs[0], inputs[1], inputs[2]),
                     agx_immediate(124), 1);
   agx_stop(b);

   ASSERT_TRUE(agx3_lower_uniform_program(b->shader));
   agx_dce(b->shader, false);
   agx_pressure_schedule(b->shader);
   agx_validate(b->shader, "uniform program before RA");
   agx_ra(b->shader);
   agx_lower_pseudo(b->shader);
   agx_opt_register_cache(b->shader);

   unsigned exports = 0, loads = 0;
   unsigned destinations = 0;
   agx_foreach_instr_global(b->shader, I) {
      if (I->op == AGX_OPCODE_DEVICE_LOAD) {
         ++loads;
         continue;
      }
      if (I->op == AGX_OPCODE_STOP)
         continue;
      ASSERT_TRUE(I->op == AGX_OPCODE_UNIFORM_FADD_AGX3 ||
                  I->op == AGX_OPCODE_UNIFORM_FMUL_AGX3 ||
                  I->op == AGX_OPCODE_UNIFORM_FFMA_AGX3);
      EXPECT_EQ(I->nr_dests, 0);
      ASSERT_GE(I->imm, 120u);
      ASSERT_LE(I->imm, 124u);
      EXPECT_EQ(I->imm & 1, 0u);
      destinations |= 1u << ((I->imm - 120) / 2);
      agx_foreach_src(I, s) {
         EXPECT_EQ(I->src[s].type, AGX_INDEX_REGISTER);
         EXPECT_EQ(I->src[s].size, AGX_SIZE_32);
         EXPECT_FALSE(I->src[s].memory);
      }
      ++exports;
   }
   EXPECT_EQ(loads, 3u);
   EXPECT_EQ(exports, 3u);
   EXPECT_EQ(destinations, 7u);
   EXPECT_EQ(b->shader->scratch_size_B, 0u);
   struct util_dynarray binary = UTIL_DYNARRAY_INIT;
   EXPECT_TRUE(agx3_pack_binary(b->shader, &binary));
   EXPECT_GT(binary.size, 0u);
   util_dynarray_fini(&binary);
}

static void
compile_arithmetic_uniform_program(enum agx_isa_encoding encoding,
                                   bool no_preamble)
{
   nir_builder n = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "native arithmetic uniform program");
   n.shader->info.internal = false;
   for (unsigned d = 0; d < 3; ++d)
      n.shader->info.workgroup_size[d] = d == 0 ? 8 : 1;
   nir_def *base = nir_load_preamble(&n, 1, 64, .base = 0);
   nir_def *output = nir_load_preamble(&n, 1, 64, .base = 4);
   nir_def *u[3];
   for (unsigned i = 0; i < 3; ++i) {
      u[i] = nir_load_global_constant(&n, 4, 32, nir_iadd_imm(&n, base, 16 * i),
                                     .access = ACCESS_CAN_SPECULATE,
                                     .align_mul = 16);
   }
   nir_def *uniform = nir_ffma(&n, u[0], u[1], u[2]);
   nir_def *id = nir_channel(&n, nir_load_local_invocation_id(&n), 0);
   nir_def *scale = nir_fadd_imm(&n, nir_u2f32(&n, id), 1.0);
   nir_def *result = nir_fmul(&n, uniform, scale);
   nir_def *address = nir_iadd(&n, output, nir_u2u64(&n, nir_imul_imm(&n, id, 16)));
   nir_store_global(&n, result, address, .align_mul = 16);
   agx_preprocess_nir(n.shader);

   struct agx_shader_key key = {};
   key.dev.encoding = encoding;
   key.reserved_preamble = 8;
   key.no_preamble = no_preamble;
   struct agx_shader_part part = {};
   ASSERT_TRUE(agx_compile_shader_nir(n.shader, &key, &part));
   ASSERT_NE(part.binary, nullptr);
   EXPECT_EQ(part.info.has_preamble, !no_preamble);
   EXPECT_EQ(part.info.agx3_preamble_uniform_inputs,
             !no_preamble && encoding == AGX_ISA_ENCODING_AGX3B);
   if (!no_preamble && encoding == AGX_ISA_ENCODING_AGX3B) {
      EXPECT_EQ(part.info.agx3_preamble_inputs.count, 4u);
      EXPECT_EQ(part.info.agx3_preamble_inputs.storage_bytes, 32u);
      EXPECT_EQ(part.info.agx3_preamble_inputs.halves[0], UINT64_C(0xf));
      for (unsigned w = 1; w < AGX3_CPROG_INPUT_WORDS; ++w)
         EXPECT_EQ(part.info.agx3_preamble_inputs.halves[w], 0u);
   } else {
      EXPECT_EQ(part.info.agx3_preamble_inputs.count, 0u);
      EXPECT_EQ(part.info.agx3_preamble_inputs.storage_bytes, 0u);
   }
   const bool native_imports = !no_preamble &&
      encoding == AGX_ISA_ENCODING_AGX3B &&
      (agx_get_compiler_debug() & AGX_DBG_NATIVE_UNIFORMS);
   EXPECT_EQ(part.info.preamble_scratch_size, native_imports ? 32u : 0u);
   EXPECT_EQ(part.info.agx3_preamble_tps_inputs, native_imports);
   if (native_imports) {
      /* The retained native stream uses GPR halves 0..31. Its input TPS
       * region must not trigger the old 256-register scratch reservation. */
      EXPECT_EQ(part.info.nr_preamble_gprs, 32u);
      EXPECT_TRUE(part.info.agx3_preamble_staging_known);
      EXPECT_EQ(part.info.agx3_preamble_staging_halfs, 0u);
   }
   EXPECT_LT(part.info.main_offset, part.info.binary_size);
   EXPECT_GT(part.info.main_size, 0u);
   const struct agx_precompiled_kernel_info compact =
      agx_compact_kernel_info(&part.info);
   EXPECT_EQ(compact.preamble_size, part.info.preamble_size);
   EXPECT_EQ(compact.preamble_offset, part.info.preamble_offset);
   if (no_preamble) {
      EXPECT_EQ(part.info.preamble_size, 0u);
      EXPECT_EQ(part.info.nr_preamble_gprs, 0u);
      free(part.binary);
      ralloc_free(n.shader);
      return;
   }

   EXPECT_GT(part.info.push_count, 8u);
   EXPECT_GT(part.info.nr_preamble_gprs, 0u);
   ASSERT_LT(part.info.preamble_offset, part.info.binary_size);
   ASSERT_GT(part.info.preamble_size, 0u);
   ASSERT_LE(part.info.preamble_size,
             part.info.binary_size - part.info.preamble_offset);
   EXPECT_TRUE(part.info.preamble_offset + part.info.preamble_size <=
                  part.info.main_offset ||
               part.info.main_offset + part.info.main_size <=
                  part.info.preamble_offset);

   /* Two native pointer imports, three vector loads, four native uniform
    * FMAs, and STOP. Dead extra
    * uses once hid these exports from selection, leaving four FMA results
    * and four separate uniform copies (12 preamble instructions). */
   if (encoding == AGX_ISA_ENCODING_AGX3B) {
      const bool native = agx_get_compiler_debug() & AGX_DBG_NATIVE_UNIFORMS;
      if (native)
         EXPECT_EQ(part.info.stats.preamble_inst, 10u);
      /* These exact extents exclude the next function's alignment padding. */
      EXPECT_EQ(part.info.preamble_size, native ? 94u : 96u);

      struct agx_ioto_constant_program program = {};
      program.code = (uint8_t *)part.binary + part.info.preamble_offset;
      program.code_size = part.info.preamble_size;
      program.code_va = 0x4000 + part.info.preamble_offset;
      program.request_prelude = part.info.agx3_preamble_uniform_inputs;
      struct agx_ioto_preshader_output suffix;
      ASSERT_TRUE(agx_ioto_constant_program_emit(
         AGX_CHIP_G17P, &program, 0x20000, &suffix));
      EXPECT_EQ(suffix.mode, AGX_IOTO_PRESHADER_MODE_INLINE);
      ASSERT_EQ(suffix.length, part.info.preamble_size);
      EXPECT_EQ(memcmp(suffix.bytes, program.code, suffix.length), 0);

      program.force_external = true;
      ASSERT_TRUE(agx_ioto_constant_program_emit(
         AGX_CHIP_G17P, &program, 0x20000, &suffix));
      EXPECT_EQ(suffix.mode, AGX_IOTO_PRESHADER_MODE_EXTERNAL);
      ASSERT_EQ(suffix.branch_offset, 2u);
      uint64_t displacement = 0;
      for (unsigned i = 0; i < 6; ++i)
         displacement |= (uint64_t)suffix.bytes[suffix.branch_offset + 3 + i]
                         << (8 * i);
      EXPECT_EQ((0x20000 + suffix.branch_offset + displacement) &
                   ((UINT64_C(1) << 48) - 1), program.code_va);

      /* Connect real compiler output to the graphics resource composer,
       * including relocation of a cached branch after the loader moves. */
      const struct agx3_ioto_uniform_range root = {
         .start_halfs = 0, .size_halfs = 8, .address = 0x12340000,
      };
      struct agx3_g17p_graphics_program_desc desc = {};
      desc.program_va = 0x20000;
      desc.shader_va = 0x100000003c0;
      desc.uniform_ranges = &root;
      desc.uniform_range_count = 1;
      desc.shader_binary_size = part.info.main_size;
      desc.nr_gprs = part.info.nr_gprs;
      desc.fragment = true;
      desc.imageblock = AGX_IOTO_G17P_IMAGEBLOCK_8B_1X_32X32;
      struct agx3_g17p_graphics_program empty;
      ASSERT_TRUE(agx3_g17p_graphics_program_generate(
         AGX_CHIP_G17P, &desc, &empty));
      const size_t offset = empty.length - AGX_IOTO_PRESHADER_ABSENT_SIZE;
      program.force_external = false;
      program.request_prelude = false;
      desc.constant_program = program;
      struct agx3_g17p_graphics_program inlined;
      ASSERT_TRUE(agx3_g17p_graphics_program_generate(
         AGX_CHIP_G17P, &desc, &inlined));
      ASSERT_EQ(inlined.length, offset + part.info.preamble_size);
      EXPECT_EQ(memcmp(inlined.bytes, empty.bytes, offset), 0);
      EXPECT_EQ(memcmp(inlined.bytes + offset, program.code,
                       part.info.preamble_size), 0);

      desc.constant_program.force_external = true;
      desc.constant_program.request_prelude = true;
      struct agx3_g17p_graphics_program_template cached;
      ASSERT_TRUE(agx3_g17p_graphics_program_template_generate(
         AGX_CHIP_G17P, &desc, &cached));
      ASSERT_EQ(cached.constant_suffix_offset, offset);
      struct agx3_g17p_graphics_program_desc moved = desc;
      moved.program_va = 0x60000;
      struct agx3_g17p_graphics_program fresh, relocated;
      ASSERT_TRUE(agx3_g17p_graphics_program_generate(
         AGX_CHIP_G17P, &moved, &fresh));
      ASSERT_TRUE(agx3_g17p_graphics_program_template_relocate(
         AGX_CHIP_G17P, &desc, &moved, &cached, relocated.bytes,
         sizeof(relocated.bytes), &relocated.length));
      ASSERT_EQ(relocated.length, fresh.length);
      EXPECT_EQ(memcmp(relocated.bytes, fresh.bytes, fresh.length), 0);

      /* The native absolute CProg form can be copied to another launch VA
       * unchanged, including when the prelude precedes the branch. */
      desc.constant_program.absolute_branch = true;
      ASSERT_TRUE(agx3_g17p_graphics_program_template_generate(
         AGX_CHIP_G17P, &desc, &cached));
      moved = desc;
      moved.program_va = 0x60000;
      ASSERT_TRUE(agx3_g17p_graphics_program_generate(
         AGX_CHIP_G17P, &moved, &fresh));
      ASSERT_EQ(fresh.length, cached.program.length);
      EXPECT_EQ(memcmp(fresh.bytes, cached.program.bytes, fresh.length), 0);
      const size_t branch = cached.constant_suffix_offset + 2;
      ASSERT_NE(cached.program.bytes[branch + 3] & 1, 0);
      uint64_t absolute = 0;
      for (unsigned i = 0; i < 6; ++i)
         absolute |= (uint64_t)cached.program.bytes[branch + 3 + i] << (8 * i);
      EXPECT_EQ(absolute & ~UINT64_C(1), program.code_va);

      if (native_imports) {
         /* Connect this actual compiled RLD consumer to loader publication.
          * Both aligned pointer words add one native 10-byte TPS COPY. */
         desc.cprog_inputs = &part.info.agx3_preamble_inputs;
         struct agx3_g17p_graphics_program with_inputs;
         ASSERT_TRUE(agx3_g17p_graphics_program_generate(
            AGX_CHIP_G17P, &desc, &with_inputs));
         EXPECT_EQ(with_inputs.length, cached.program.length + 20u);
         ASSERT_TRUE(agx3_g17p_graphics_program_template_generate(
            AGX_CHIP_G17P, &desc, &cached));
         moved = desc;
         moved.program_va = 0x70000;
         ASSERT_TRUE(agx3_g17p_graphics_program_template_relocate(
            AGX_CHIP_G17P, &desc, &moved, &cached, relocated.bytes,
            sizeof(relocated.bytes), &relocated.length));
         ASSERT_TRUE(agx3_g17p_graphics_program_generate(
            AGX_CHIP_G17P, &moved, &fresh));
         EXPECT_EQ(relocated.length, fresh.length);
         EXPECT_EQ(memcmp(relocated.bytes, fresh.bytes, fresh.length), 0);
      }
   }
   free(part.binary);
   ralloc_free(n.shader);
}

TEST(AGX3UniformNIR, CompilesArithmeticPreambleAndMain)
{
   compile_arithmetic_uniform_program(AGX_ISA_ENCODING_AGX3B, false);
}

TEST(AGX3UniformNIR, DisabledPreambleHasZeroExtent)
{
   compile_arithmetic_uniform_program(AGX_ISA_ENCODING_AGX3B, true);
}

TEST(AGX3UniformNIR, AGX2PreambleRetainsItsExtent)
{
   compile_arithmetic_uniform_program(AGX_ISA_ENCODING_AGX2, false);
}

TEST(AGX3UniformNIR, LiteralPositiveZeroProductsCompile)
{
   for (float constant : {0.0f, 2.0f, 3.0f, 4.0f, 5.0f, 31.0f}) {
      for (bool swap : {false, true}) {
         nir_builder n = nir_builder_init_simple_shader(
            MESA_SHADER_COMPUTE, &agx_nir_options, "literal positive zero FMA");
         n.fp_math_ctrl = 0; /* Preserve NaNs, signed zero and fused semantics. */
         n.shader->info.internal = false;
         for (unsigned d = 0; d < 3; d++)
            n.shader->info.workgroup_size[d] = 1;
         nir_def *base = nir_load_preamble(&n, 1, 64, .base = 0);
         nir_def *output = nir_load_preamble(&n, 1, 64, .base = 4);
         nir_def *x = nir_load_global_constant(&n, 1, 32, base,
                                               .align_mul = 4);
         nir_def *c = nir_imm_float(&n, constant);
         nir_def *z = nir_imm_float(&n, 0.0f);
         nir_def *result = nir_ffma(&n, swap ? z : c, swap ? c : z, x);
         nir_store_global(&n, result, output, .align_mul = 4);
         agx_preprocess_nir(n.shader);
         struct agx_shader_key key = {};
         key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
         key.reserved_preamble = 8;
         key.no_preamble = true;
         struct agx_shader_part part = {};
         ASSERT_TRUE(agx_compile_shader_nir(n.shader, &key, &part))
            << "constant=" << constant << " swap=" << swap;
         EXPECT_GT(part.info.main_size, 0u);
         free(part.binary);
         ralloc_free(n.shader);
      }
   }
}

TEST_F(AGX3UniformProgram, ReverseHalvesPairAcrossDisjointWrite)
{
   b->shader->allocated_vec = _mesa_hash_table_u64_create(b->shader);
   agx_index high = agx_temp(b->shader, AGX_SIZE_16);
   agx_index low = agx_temp(b->shader, AGX_SIZE_16);
   agx_preload_to(b, high, agx_register(1, AGX_SIZE_16));
   agx_preload_to(b, low, agx_register(3, AGX_SIZE_16));
   agx_uniform_store(b, high, agx_immediate(281), 1);
   agx_uniform_store(b, input(4), agx_immediate(100), 1);
   agx_uniform_store(b, low, agx_immediate(280), 1);
   ASSERT_TRUE(agx3_coalesce_uniform_half_stores(b->shader));
   unsigned stores = 0, pairs = 0;
   agx_foreach_instr_global(b->shader, I) {
      if (I->op == AGX_OPCODE_COLLECT) {
         ASSERT_EQ(I->nr_srcs, 2u);
         EXPECT_TRUE(agx_is_equiv(I->src[0], low));
         EXPECT_TRUE(agx_is_equiv(I->src[1], high));
         pairs++;
      }
      if (I->op == AGX_OPCODE_UNIFORM_STORE) {
         EXPECT_EQ(I->src[0].size, AGX_SIZE_32);
         EXPECT_TRUE(I->src[1].value == 100 || I->src[1].value == 280);
         stores++;
      }
   }
   EXPECT_EQ(pairs, 1u);
   EXPECT_EQ(stores, 2u);
}

TEST_F(AGX3UniformProgram, HalfPairRetainsWideImmediateBits)
{
   b->shader->allocated_vec = _mesa_hash_table_u64_create(b->shader);
   agx_uniform_store(b, agx_immediate(65535), agx_immediate(280), 1);
   agx_uniform_store(b, agx_immediate(256), agx_immediate(281), 1);
   ASSERT_TRUE(agx3_coalesce_uniform_half_stores(b->shader));
   unsigned constants = 0;
   agx_foreach_instr_global(b->shader, I) {
      if (I->op == AGX_OPCODE_MOV_IMM) {
         EXPECT_TRUE(I->imm == 65535 || I->imm == 256);
         constants++;
      }
      if (I->op == AGX_OPCODE_COLLECT) {
         for (unsigned s = 0; s < I->nr_srcs; s++)
            EXPECT_EQ(I->src[s].type, AGX_INDEX_NORMAL);
      }
   }
   EXPECT_EQ(constants, 2u);
}

TEST_F(AGX3UniformProgram, HalfReadPreventsPairMotion)
{
   b->shader->allocated_vec = _mesa_hash_table_u64_create(b->shader);
   agx_index half = agx_temp(b->shader, AGX_SIZE_16);
   agx_preload_to(b, half, agx_register(1, AGX_SIZE_16));
   agx_uniform_store(b, half, agx_immediate(280), 1);
   agx_mov(b, agx_uniform(280, AGX_SIZE_16));
   agx_uniform_store(b, half, agx_immediate(281), 1);
   EXPECT_FALSE(agx3_coalesce_uniform_half_stores(b->shader));
}

/* A late preamble DCE removes export28 from the mutable-descriptor CTS
 * fragment while leaving its boolean at29. COPY must preserve that lane. */
TEST_F(AGX3UniformProgram, OddPaddedHalfKeepsItsUniformLane)
{
   agx_shader_info info = {};
   info.push_count = 30;
   b->shader->out = &info;
   b->shader->stage = MESA_SHADER_COMPUTE;
   b->shader->nir = rzalloc(memctx, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_COMPUTE;
   for (unsigned d = 0; d < 3; ++d)
      b->shader->nir->info.workgroup_size[d] = 1;
   agx_index half = agx_temp(b->shader, AGX_SIZE_16);
   agx_preload_to(b, half, agx_register(1, AGX_SIZE_16));
   agx_instr *store = agx_uniform_store(b, half, agx_immediate(29), 1);
   agx_stop(b);
   ASSERT_TRUE(agx3_lower_uniform_half_exports(b->shader));
   EXPECT_EQ(store->src[1].value, 28u);
   EXPECT_EQ(store->src[0].size, AGX_SIZE_32);
   EXPECT_EQ(info.push_count, 30u);
   agx_instr *collect = list_entry(store->link.prev, agx_instr, link);
   ASSERT_EQ(collect->op, AGX_OPCODE_COLLECT);
   EXPECT_EQ(collect->src[0].type, AGX_INDEX_IMMEDIATE);
   EXPECT_EQ(collect->src[0].value, 0u);
   EXPECT_TRUE(agx_is_equiv(collect->src[1], half));
   EXPECT_FALSE(agx3_lower_uniform_half_exports(b->shader));
   agx_validate(b->shader, "odd half export lane");
   agx_dce(b->shader, false);
   agx_pressure_schedule(b->shader);
   agx_ra(b->shader);
   agx_lower_pseudo(b->shader);
   agx_opt_register_cache(b->shader);
   EXPECT_EQ(store->src[0].value & 1, 0u);
   EXPECT_EQ(store->src[1].value, 28u);
   struct util_dynarray binary = UTIL_DYNARRAY_INIT;
   EXPECT_TRUE(agx3_pack_binary(b->shader, &binary));
   EXPECT_GT(binary.size, 0u);
   util_dynarray_fini(&binary);
}

TEST_F(AGX3UniformProgram, OddHalfAtLastUniformPairDoesNotOverflow)
{
   agx_shader_info info = {};
   info.push_count = 512;
   b->shader->out = &info;
   agx_index half = agx_temp(b->shader, AGX_SIZE_16);
   agx_preload_to(b, half, agx_register(1, AGX_SIZE_16));
   agx_instr *store = agx_uniform_store(b, half, agx_immediate(511), 1);
   ASSERT_TRUE(agx3_lower_uniform_half_exports(b->shader));
   EXPECT_EQ(store->src[1].value, 510u);
   EXPECT_EQ(info.push_count, 512u);
}

TEST_F(AGX3UniformProgram, OddHalfPreservesLowerNeighbourHazards)
{
   agx_shader_info info = {};
   info.rodata.base_uniform = 48;
   info.rodata.size_16 = 1;
   info.agx3_preamble_inputs.halves[0] = UINT64_C(1) << 56;
   b->shader->out = &info;
   b->shader->agx3_cprog_inputs_lowered = true;
   key.reserved_preamble = 29;
   agx_index half = agx_temp(b->shader, AGX_SIZE_16);
   agx_preload_to(b, half, agx_register(1, AGX_SIZE_16));
   agx_uniform_store(b, half, agx_immediate(29), 1); /* reserved28 */
   agx_uniform_store(b, half, agx_immediate(32), 1);
   agx_uniform_store(b, half, agx_immediate(33), 1); /* live32 */
   agx_mov(b, agx_uniform(40, AGX_SIZE_16));
   agx_uniform_store(b, half, agx_immediate(41), 1); /* read40 */
   agx_uniform_store(b, half, agx_immediate(49), 1); /* rodata48 */
   agx_uniform_store(b, half, agx_immediate(57), 1); /* imported56 */
   EXPECT_FALSE(agx3_lower_uniform_half_exports(b->shader));
}

TEST_F(AGX3CProgImports, VectorHalfExportsPreserveEveryLaneThroughPacking)
{
   for (unsigned halves : {2u, 4u}) {
      agx_index value = agx_vec_temp(b->shader, AGX_SIZE_16, halves);
      agx_instr *collect = agx_collect_to(b, value, halves);
      for (unsigned h = 0; h < halves; ++h)
         collect->src[h] = agx_immediate(0x40 + h);
      agx_instr *store = agx_uniform_store(b, value, agx_immediate(24 + halves),
                                           BITFIELD_MASK(halves));
      ASSERT_TRUE(agx3_lower_uniform_half_exports(b->shader));
      EXPECT_EQ(store->src[0].size, AGX_SIZE_32);
      EXPECT_EQ(agx_channels(store->src[0]), halves / 2);
      EXPECT_EQ(store->mask, BITFIELD_MASK(halves / 2));
      EXPECT_EQ(store->src[1].value, 24 + halves);
      agx_instr *words = list_entry(store->link.prev, agx_instr, link);
      ASSERT_EQ(words->op, AGX_OPCODE_COLLECT);
      agx_instr *split = list_entry(words->link.prev, agx_instr, link);
      ASSERT_EQ(split->op, AGX_OPCODE_SPLIT);
      EXPECT_TRUE(agx_is_equiv(split->src[0], value));
      for (unsigned h = 0; h < halves; ++h)
         EXPECT_TRUE(agx_is_equiv(words->src[h], split->dest[h]));
   }
   agx_stop(b);
   EXPECT_FALSE(agx3_lower_uniform_half_exports(b->shader));
   agx_validate(b->shader, "vector half export widths");
   agx_dce(b->shader, false);
   agx_pressure_schedule(b->shader);
   agx_ra(b->shader);
   agx_lower_pseudo(b->shader);
   agx_opt_register_cache(b->shader);
   struct util_dynarray binary = UTIL_DYNARRAY_INIT;
   EXPECT_TRUE(agx3_pack_binary(b->shader, &binary));
   EXPECT_GT(binary.size, 0u);
   util_dynarray_fini(&binary);
}

TEST_F(AGX3UniformProgram, VectorHalfExportRejectsUnalignedAndSparseWrites)
{
   agx_index pair = agx_vec_temp(b->shader, AGX_SIZE_16, 2);
   agx_index triple = agx_vec_temp(b->shader, AGX_SIZE_16, 3);
   agx_index quad = agx_vec_temp(b->shader, AGX_SIZE_16, 4);
   agx_uniform_store(b, pair, agx_immediate(25), 3);
   agx_uniform_store(b, triple, agx_immediate(40), 7);
   agx_mov(b, agx_uniform(43, AGX_SIZE_16));
   agx_uniform_store(b, quad, agx_immediate(48), 5);
   EXPECT_FALSE(agx3_lower_uniform_half_exports(b->shader));
}

TEST_F(AGX3CProgImports, ThreeHalfVectorPadsOnlyUnusedFourthLane)
{
   info.push_count = 27;
   agx_index value = agx_vec_temp(b->shader, AGX_SIZE_16, 3);
   agx_index halves[3];
   for (unsigned h = 0; h < 3; ++h) {
      halves[h] = agx_temp(b->shader, AGX_SIZE_16);
      agx_preload_to(b, halves[h], agx_register(1 + 2 * h, AGX_SIZE_16));
   }
   agx_instr *input = agx_collect_to(b, value, 3);
   for (unsigned h = 0; h < 3; ++h)
      input->src[h] = halves[h];
   agx_instr *store = agx_uniform_store(b, value, agx_immediate(24), 7);
   agx_stop(b);
   ASSERT_TRUE(agx3_lower_uniform_half_exports(b->shader));
   EXPECT_EQ(store->src[0].size, AGX_SIZE_32);
   EXPECT_EQ(agx_channels(store->src[0]), 2u);
   EXPECT_EQ(store->mask, 3u);
   EXPECT_EQ(store->src[1].value, 24u);
   EXPECT_EQ(info.push_count, 28u);
   agx_instr *words = list_entry(store->link.prev, agx_instr, link);
   ASSERT_EQ(words->op, AGX_OPCODE_COLLECT);
   EXPECT_EQ(words->nr_srcs, 4u);
   EXPECT_EQ(words->src[3].type, AGX_INDEX_IMMEDIATE);
   EXPECT_EQ(words->src[3].value, 0u);
   agx_validate(b->shader, "padded half vector");
   agx_dce(b->shader, false);
   agx_pressure_schedule(b->shader);
   agx_ra(b->shader);
   agx_lower_pseudo(b->shader);
   agx_opt_register_cache(b->shader);
   struct util_dynarray binary = UTIL_DYNARRAY_INIT;
   EXPECT_TRUE(agx3_pack_binary(b->shader, &binary));
   EXPECT_GT(binary.size, 0u);
   util_dynarray_fini(&binary);
}

TEST_F(AGX3UniformProgram, ThreeHalfVectorPreservesEachFourthLaneHazard)
{
   for (unsigned kind = 0; kind < 5; ++kind) {
      agx_builder *test = agx_test_builder(memctx);
      agx_shader_key test_key = {};
      test_key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      test->shader->key = &test_key;
      test->shader->is_preamble = true;
      agx_shader_info info = {};
      test->shader->out = &info;
      agx_index value = agx_vec_temp(test->shader, AGX_SIZE_16, 3);
      agx_uniform_store(test, value, agx_immediate(24), 7);
      if (kind == 0) test_key.reserved_preamble = 28;
      if (kind == 1) agx_uniform_store(test, agx_immediate(1), agx_immediate(27), 1);
      if (kind == 2) agx_mov(test, agx_uniform(27, AGX_SIZE_16));
      if (kind == 3) {
         test->shader->agx3_cprog_inputs_lowered = true;
         info.agx3_preamble_inputs.halves[0] = UINT64_C(1) << 27;
      }
      if (kind == 4) {info.rodata.base_uniform = 27; info.rodata.size_16 = 1;}
      EXPECT_FALSE(agx3_lower_uniform_half_exports(test->shader)) << kind;
   }
}

TEST_F(AGX3UniformProgram, ThreeHalfVectorLastPaddingStaysInUniformFile)
{
   agx_shader_info info = {};
   info.push_count = 511;
   b->shader->out = &info;
   agx_index value = agx_vec_temp(b->shader, AGX_SIZE_16, 3);
   agx_instr *store = agx_uniform_store(b, value, agx_immediate(508), 7);
   ASSERT_TRUE(agx3_lower_uniform_half_exports(b->shader));
   EXPECT_EQ(store->src[1].value, 508u);
   EXPECT_EQ(info.push_count, 512u);
}

static agx_index
half_vec3_input(agx_builder *b, unsigned first_reg)
{
   agx_index value = agx_vec_temp(b->shader, AGX_SIZE_16, 3);
   agx_index halves[3];
   agx_builder pre = agx_init_builder(b->shader, agx_before_block(agx_start_block(b->shader)));
   for (unsigned h = 0; h < 3; ++h) {
      halves[h] = agx_temp(b->shader, AGX_SIZE_16);
      agx_preload_to(&pre, halves[h], agx_register(first_reg + 2 * h, AGX_SIZE_16));
   }
   agx_instr *collect = agx_collect_to(b, value, 3);
   for (unsigned h = 0; h < 3; ++h)
      collect->src[h] = halves[h];
   return value;
}

TEST_F(AGX3CProgImports, AdjacentHalfVectorsPairEveryLiveLane)
{
   b->shader->allocated_vec = _mesa_hash_table_u64_create(b->shader);
   agx_index values[2] = {half_vec3_input(b, 1), half_vec3_input(b, 7)};
   agx_uniform_store(b, values[0], agx_immediate(24), 7);
   agx_uniform_store(b, values[1], agx_immediate(27), 7);
   agx_stop(b);
   ASSERT_TRUE(agx3_coalesce_uniform_half_stores(b->shader));
   agx_index halves[6] = {};
   unsigned splits = 0, stores = 0;
   agx_foreach_instr_global(b->shader, I) {
      if (I->op != AGX_OPCODE_SPLIT)
         continue;
      for (unsigned v = 0; v < 2; ++v) {
         if (!agx_is_equiv(I->src[0], values[v]))
            continue;
         ASSERT_EQ(I->nr_dests, 3u);
         for (unsigned h = 0; h < 3; ++h)
            halves[3 * v + h] = I->dest[h];
         splits++;
      }
   }
   ASSERT_EQ(splits, 2u);
   agx_foreach_instr_global(b->shader, I) {
      if (I->op != AGX_OPCODE_UNIFORM_STORE)
         continue;
      ASSERT_EQ(I->src[0].size, AGX_SIZE_32);
      ASSERT_EQ(I->mask, 1u);
      ASSERT_GE(I->src[1].value, 24u);
      ASSERT_LE(I->src[1].value, 28u);
      const unsigned offset = I->src[1].value - 24;
      ASSERT_EQ(offset & 1, 0u);
      agx_instr *collect = list_entry(I->link.prev, agx_instr, link);
      ASSERT_EQ(collect->op, AGX_OPCODE_COLLECT);
      ASSERT_EQ(collect->nr_srcs, 2u);
      EXPECT_TRUE(agx_is_equiv(collect->src[0], halves[offset]));
      EXPECT_TRUE(agx_is_equiv(collect->src[1], halves[offset + 1]));
      stores++;
   }
   EXPECT_EQ(stores, 3u);
   agx_validate(b->shader, "adjacent half exports");
   agx_dce(b->shader, false);
   agx_pressure_schedule(b->shader);
   agx_ra(b->shader);
   agx_lower_pseudo(b->shader);
   agx_opt_register_cache(b->shader);
   struct util_dynarray binary = UTIL_DYNARRAY_INIT;
   EXPECT_TRUE(agx3_pack_binary(b->shader, &binary));
   EXPECT_GT(binary.size, 0u);
   util_dynarray_fini(&binary);
}

TEST_F(AGX3UniformProgram, ScalarAndThreeHalfVectorShareExactWord)
{
   b->shader->allocated_vec = _mesa_hash_table_u64_create(b->shader);
   agx_index low = agx_temp(b->shader, AGX_SIZE_16);
   agx_preload_to(b, low, agx_register(1, AGX_SIZE_16));
   agx_index vector = half_vec3_input(b, 3);
   agx_uniform_store(b, low, agx_immediate(24), 1);
   agx_uniform_store(b, vector, agx_immediate(25), 7);
   ASSERT_TRUE(agx3_coalesce_uniform_half_stores(b->shader));
   unsigned stores = 0;
   agx_foreach_instr_global(b->shader, I) {
      if (I->op != AGX_OPCODE_UNIFORM_STORE)
         continue;
      EXPECT_EQ(I->src[0].size, AGX_SIZE_32);
      EXPECT_EQ(I->mask, 1u);
      EXPECT_TRUE(I->src[1].value == 24 || I->src[1].value == 26);
      if (I->src[1].value == 24) {
         agx_instr *collect = list_entry(I->link.prev, agx_instr, link);
         ASSERT_EQ(collect->op, AGX_OPCODE_COLLECT);
         EXPECT_TRUE(agx_is_equiv(collect->src[0], low));
      }
      stores++;
   }
   EXPECT_EQ(stores, 2u);
}

TEST_F(AGX3UniformProgram, ThreeHalfDecompositionPreservesReadBarrier)
{
   b->shader->allocated_vec = _mesa_hash_table_u64_create(b->shader);
   agx_index values[2] = {half_vec3_input(b, 1), half_vec3_input(b, 7)};
   agx_uniform_store(b, values[0], agx_immediate(24), 7);
   agx_mov(b, agx_uniform(26, AGX_SIZE_16));
   agx_uniform_store(b, values[1], agx_immediate(27), 7);
   ASSERT_TRUE(agx3_coalesce_uniform_half_stores(b->shader));
   unsigned half_stores = 0, word_stores = 0;
   agx_foreach_instr_global(b->shader, I) {
      if (I->op != AGX_OPCODE_UNIFORM_STORE)
         continue;
      if (I->src[0].size == AGX_SIZE_16) {
         EXPECT_TRUE(I->src[1].value == 26 || I->src[1].value == 27);
         half_stores++;
      } else {
         EXPECT_TRUE(I->src[1].value == 24 || I->src[1].value == 28);
         word_stores++;
      }
   }
   EXPECT_EQ(half_stores, 2u);
   EXPECT_EQ(word_stores, 2u);
}

TEST_F(AGX3UniformProgram, SafePaddedThreeHalfAndFullVectorsStayIntact)
{
   b->shader->allocated_vec = _mesa_hash_table_u64_create(b->shader);
   agx_index value = half_vec3_input(b, 1);
   agx_instr *store = agx_uniform_store(b, value, agx_immediate(24), 7);
   agx_instr *pair = agx_uniform_store(b, agx_vec_temp(b->shader, AGX_SIZE_16, 2), agx_immediate(32), 3);
   agx_instr *quad = agx_uniform_store(b, agx_vec_temp(b->shader, AGX_SIZE_16, 4), agx_immediate(36), 15);
   EXPECT_FALSE(agx3_coalesce_uniform_half_stores(b->shader));
   EXPECT_EQ(store->mask, 7u);
   EXPECT_EQ(pair->mask, 3u);
   EXPECT_EQ(quad->mask, 15u);
}
