#include "agx3_alu_core.h"
#include "agx3_pack.h"
#include "agx3_stage_core.h"
#include "agx3_tex_core.h"
#include "agx_test.h"
extern "C" {
#include "agx_nir.h"
#include "asahi/layout/layout.h"
}
#include <gtest/gtest.h>
#include <vector>
#include <csignal>
#include <cstdlib>

TEST(NativeForms, CollectSplitSameWidthAndChains)
{
   for (agx_size size : {AGX_SIZE_16, AGX_SIZE_32, AGX_SIZE_64}) {
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_index input[3];
      for (unsigned i = 0; i < 3; ++i)
         input[i] = agx_preload(b, agx_register(4 * i, size));
      agx_index previous = input[0];
      for (unsigned i = 0; i < 8; ++i) {
         agx_index vector = agx_vec_temp(b->shader, size, 2);
         agx_instr *collect = agx_collect_to(b, vector, 2);
         collect->src[0] = previous;
         collect->src[1] = input[1 + (i & 1)];
         agx_instr *split = agx_split(b, 2, vector);
         split->dest[0] = agx_temp(b->shader, size);
         split->dest[1] = agx_null();
         previous = split->dest[0];
      }
      agx_instr *use = agx_unit_test(b, previous);
      ASSERT_TRUE(agx_opt_collect_split(b->shader));
      EXPECT_TRUE(agx_is_equiv(use->src[0], input[0]));
      EXPECT_EQ(use->src[0].size, size);
      agx_dce(b->shader, true);
      agx_foreach_instr_global(b->shader, I) {
         EXPECT_NE(I->op, AGX_OPCODE_COLLECT);
         EXPECT_NE(I->op, AGX_OPCODE_SPLIT);
      }
      EXPECT_FALSE(agx_opt_collect_split(b->shader));
      ralloc_free(mem);
   }
}

TEST(NativeForms, CollectSplitPaddingRetainsScalarAffinity)
{
   for (bool dead_normal : {false, true}) {
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      b->shader->nir = rzalloc(mem, nir_shader);
      b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;
      agx_index occupied = agx_preload(b, agx_register(1, AGX_SIZE_16));
      agx_index input = agx_preload(b, agx_register(16, AGX_SIZE_32));
      agx_instr *producers[3];
      for (unsigned i = 0; i < 3; ++i)
         producers[i] = agx_fadd_to(b, agx_temp(b->shader, AGX_SIZE_32),
                                    input, input);
      agx_index padding = dead_normal ? agx_mov_imm(b, 32, 7)
                                     : agx_undef(AGX_SIZE_32);
      agx_index vector = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
      agx_instr *collect = agx_collect_to(b, vector, 4);
      for (unsigned i = 0; i < 3; ++i)
         collect->src[i] = producers[i]->dest[0];
      collect->src[3] = padding;
      agx_instr *split = agx_split(b, 4, vector);
      for (unsigned i = 0; i < 3; ++i) {
         split->dest[i] = agx_temp(b->shader, AGX_SIZE_32);
         agx_unit_test(b, split->dest[i]);
      }
      split->dest[3] = agx_null();
      agx_unit_test(b, occupied);
      ASSERT_TRUE(agx_opt_collect_split(b->shader));
      agx_dce(b->shader, true);
      agx_ra(b->shader);
      ASSERT_FALSE(b->shader->ra_failed);
      EXPECT_EQ(producers[0]->dest[0].value % 8, 0u);
      EXPECT_EQ(producers[1]->dest[0].value, producers[0]->dest[0].value + 2);
      EXPECT_EQ(producers[2]->dest[0].value, producers[0]->dest[0].value + 4);
      ralloc_free(mem);
   }
}

TEST(NativeForms, CollectSplitEarlyInteriorExportAffinity)
{
   for (unsigned mode = 0; mode < 5; ++mode) {
      bool all_exported = mode == 1;
      unsigned exported_component = mode == 2 ? 3 : (mode == 4 ? 0 : 1);
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      b->shader->nir = rzalloc(mem, nir_shader);
      b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;
      agx_index input = agx_preload(b, agx_register(0, AGX_SIZE_32));
      agx_index high =
         mode == 3 ? agx_preload(b, agx_register(31, AGX_SIZE_16)) : agx_null();
      agx_instr *producers[4];
      unsigned order[4] = {0, 1, 2, 3};
      if (mode == 2) {
         order[0] = 3;
         order[3] = 0;
      }
      for (unsigned component : order)
         producers[component] =
            agx_fadd_to(b, agx_temp(b->shader, AGX_SIZE_32), input, input);
      agx_index vector = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
      agx_instr *collect = agx_collect_to(b, vector, 4);
      for (unsigned i = 0; i < 4; ++i)
         collect->src[i] = producers[i]->dest[0];
      agx_instr *split = agx_split(b, 4, vector);
      for (unsigned i = 0; i < 4; ++i) {
         split->dest[i] = agx_temp(b->shader, AGX_SIZE_32);
         if (i == exported_component)
            agx_export(b, split->dest[i], 14);
         else if (all_exported)
            agx_export(b, split->dest[i], 12 + 2 * i);
         else
            agx_unit_test(b, split->dest[i]);
      }
      if (!agx_is_null(high))
         agx_unit_test(b, high);
      ASSERT_TRUE(agx_opt_collect_split(b->shader));
      agx_dce(b->shader, true);
      agx_ra(b->shader);
      ASSERT_FALSE(b->shader->ra_failed);
      EXPECT_EQ(producers[exported_component]->dest[0].value == 14u,
                all_exported || mode == 2 || mode == 3 || mode == 4);
      bool exported = false;
      agx_foreach_instr_global(b->shader, I) {
         if (I->op == AGX_OPCODE_EXPORT && I->imm == 14) {
            exported = true;
            EXPECT_EQ(I->src[0].type, AGX_INDEX_REGISTER);
            EXPECT_EQ(I->src[0].value, 14u);
         }
      }
      EXPECT_TRUE(exported);
      ralloc_free(mem);
   }
}

TEST(NativeForms, EarlyPairExportLeavesLaterWideGroupAvailable)
{
   for (unsigned mode = 0; mode < 6; ++mode) {
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      b->shader->nir = rzalloc(mem, nir_shader);
      b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;
      agx_index input = agx_preload(b, agx_register(0, AGX_SIZE_32));
      agx_instr *pair[2];
      for (unsigned i = 0; i < 2; ++i) {
         unsigned component = mode == 2 ? i : 1 - i;
         pair[component] =
            agx_fadd_to(b, agx_temp(b->shader, AGX_SIZE_32), input, input);
      }
      agx_index pair_vector = agx_vec_temp(b->shader, AGX_SIZE_32, 2);
      agx_instr *collect = agx_collect_to(b, pair_vector, 2);
      for (unsigned i = 0; i < 2; ++i)
         collect->src[i] = pair[i]->dest[0];
      agx_instr *pair_split = agx_split(b, 2, pair_vector);
      for (unsigned i = 0; i < 2; ++i)
         pair_split->dest[i] = agx_temp(b->shader, AGX_SIZE_32);
      agx_instr *wide[4];
      for (unsigned i = 0; i < 4; ++i)
         wide[i] =
            agx_fadd_to(b, agx_temp(b->shader, AGX_SIZE_32), input, input);
      agx_index vector = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
      agx_instr *wide_collect = agx_collect_to(b, vector, 4);
      for (unsigned i = 0; i < 4; ++i)
         wide_collect->src[i] = wide[i]->dest[0];
      agx_instr *split = agx_split(b, 4, vector);
      for (unsigned i = 0; i < 4; ++i) {
         split->dest[i] = agx_temp(b->shader, AGX_SIZE_32);
         agx_unit_test(b, split->dest[i]);
      }
      if (mode != 1) {
         if (mode >= 3) {
            agx_publish_texture_scalar_agx3(b, pair_split->dest[0], input,
                                            agx_null(), 1);
            if (mode >= 4) {
               agx_index resource = agx_get_vec_index(1, AGX_SIZE_32, 2);
               resource.type = AGX_INDEX_RESOURCE;
               agx_index result = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
               agx_texture_sample_to(
                  b, result, resource, agx_zero(), agx_zero(),
                  agx_uniform(24, AGX_SIZE_64), agx_uniform(28, AGX_SIZE_64),
                  agx_null(), AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 15, false,
                  false, false, AGX_GATHER_NONE, false);
               if (mode == 5) {
                  agx_index later = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
                  agx_texture_sample_to(
                     b, later, resource, agx_zero(), agx_zero(),
                     agx_uniform(32, AGX_SIZE_64), agx_uniform(36, AGX_SIZE_64),
                     agx_null(), AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 15, false,
                     false, false, AGX_GATHER_NONE, false);
                  agx_instr *later_split = agx_split(b, 4, later);
                  later_split->dest[0] = agx_temp(b->shader, AGX_SIZE_32);
                  for (unsigned i = 1; i < 4; ++i)
                     later_split->dest[i] = agx_null();
                  agx_unit_test(b, later_split->dest[0]);
               }
               agx_instr *result_split = agx_split(b, 4, result);
               result_split->dest[0] = agx_temp(b->shader, AGX_SIZE_32);
               for (unsigned i = 1; i < 4; ++i)
                  result_split->dest[i] = agx_null();
               agx_unit_test(b, agx_fmul(b, result_split->dest[0], input));
            }
         } else
            agx_unit_test(b, agx_fmul(b, pair_split->dest[0], input));
      }
      if (mode == 1)
         agx_export(b, pair_split->dest[0], 12);
      agx_export(b, pair_split->dest[1], 14);
      ASSERT_TRUE(agx_opt_collect_split(b->shader));
      agx_dce(b->shader, true);
      agx_validate(b->shader, "early pair export allocation");
      agx_ra(b->shader);
      ASSERT_FALSE(b->shader->ra_failed);
      EXPECT_EQ(pair[1]->dest[0].value == 14u, mode != 0 && mode != 4);
      if (mode == 0 || mode == 4) {
         EXPECT_LT(pair[1]->dest[0].value, 8u);
         for (unsigned i = 0; i < 4; ++i)
            EXPECT_EQ(wide[i]->dest[0].value, 8 + 2 * i);
      }
      bool exported = false;
      agx_foreach_instr_global(b->shader, I) {
         if (I->op == AGX_OPCODE_EXPORT && I->imm == 14) {
            exported = true;
            EXPECT_EQ(I->src[0].type, AGX_INDEX_REGISTER);
            EXPECT_EQ(I->src[0].value, 14u);
         }
      }
      EXPECT_TRUE(exported);
      ralloc_free(mem);
   }
}

TEST(NativeForms, CollectSplitKeepsVectorConsumersAndModifiers)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_index x = agx_preload(b, agx_register(0, AGX_SIZE_32));
   agx_index y = agx_preload(b, agx_register(2, AGX_SIZE_32));
   agx_index vector = agx_vec_temp(b->shader, AGX_SIZE_32, 2);
   agx_instr *collect = agx_collect_to(b, vector, 2);
   collect->src[0] = x;
   collect->src[1] = y;
   agx_instr *split = agx_split(b, 2, vector);
   split->dest[0] = agx_temp(b->shader, AGX_SIZE_32);
   split->dest[1] = agx_temp(b->shader, AGX_SIZE_32);
   agx_instr *vector_use = agx_unit_test(b, vector);
   agx_index result = agx_temp(b->shader, AGX_SIZE_32);
   agx_instr *add = agx_fadd_to(
      b, result, agx_neg(agx_abs(split->dest[0])), split->dest[1]);
   agx_unit_test(b, result);
   ASSERT_TRUE(agx_opt_collect_split(b->shader));
   EXPECT_TRUE(agx_is_equiv(add->src[0], agx_neg(agx_abs(x))));
   EXPECT_TRUE(add->src[0].abs);
   EXPECT_TRUE(add->src[0].neg);
   EXPECT_TRUE(agx_is_equiv(add->src[1], y));
   EXPECT_TRUE(agx_is_equiv(vector_use->src[0], vector));
   agx_dce(b->shader, true);
   bool kept_vector = false;
   agx_foreach_instr_global(b->shader, I) {
      kept_vector |= I == collect;
      EXPECT_NE(I->op, AGX_OPCODE_SPLIT);
   }
   EXPECT_TRUE(kept_vector);
   ralloc_free(mem);
}

TEST(NativeForms, CollectSplitDoesNotChangeWidthsOrOperandClasses)
{
   for (agx_size size : {AGX_SIZE_32, AGX_SIZE_64}) {
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_index x = agx_preload(b, agx_register(0, size));
      agx_index vector = agx_vec_temp(b->shader, size, 2);
      agx_instr *collect = agx_collect_to(b, vector, 2);
      collect->src[0] = collect->src[1] = x;
      agx_instr *split = agx_split(b, 4, vector);
      for (unsigned i = 0; i < 4; ++i) {
         split->dest[i] = agx_temp(b->shader, agx_size(size - 1));
         agx_unit_test(b, split->dest[i]);
      }
      EXPECT_FALSE(agx_opt_collect_split(b->shader));
      ralloc_free(mem);
   }
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_index x = agx_preload(b, agx_register(0, AGX_SIZE_32));
   agx_index vector = agx_vec_temp(b->shader, AGX_SIZE_32, 2);
   agx_instr *collect = agx_collect_to(b, vector, 2);
   collect->src[0] = x;
   collect->src[1] = agx_uniform(20, AGX_SIZE_32);
   agx_instr *split = agx_split(b, 2, vector);
   split->dest[0] = agx_temp(b->shader, AGX_SIZE_32);
   split->dest[1] = agx_temp(b->shader, AGX_SIZE_32);
   agx_instr *first = agx_unit_test(b, split->dest[0]);
   agx_instr *second = agx_unit_test(b, split->dest[1]);
   agx_index original_second = second->src[0];
   ASSERT_TRUE(agx_opt_collect_split(b->shader));
   EXPECT_TRUE(agx_is_equiv(first->src[0], x));
   EXPECT_TRUE(agx_is_equiv(second->src[0], original_second));
   agx_dce(b->shader, true);
   bool kept_split = false;
   agx_foreach_instr_global(b->shader, I)
      kept_split |= I == split;
   EXPECT_TRUE(kept_split);
   ralloc_free(mem);
}

TEST(NativeForms, BankedScalarCollectAffinity)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   key.secondary = true;
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   b->shader->nir = rzalloc(mem, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;
   agx_index occupied = agx_preload(b, agx_register(1, AGX_SIZE_16));
   agx_index inputs[4];
   for (unsigned i = 0; i < 4; ++i)
      inputs[i] = agx_preload(b, agx_register(8 + 2 * i, AGX_SIZE_32));
   agx_instr *packs[2];
   for (unsigned i = 0; i < 2; ++i) {
      agx_index dest = agx_temp(b->shader, AGX_SIZE_16);
      packs[i] = agx_fltnorm_agx3_to(
         b, dest, inputs[2 * i], inputs[2 * i + 1], AGX_FORMAT_U8NORM);
   }
   agx_index vector = agx_vec_temp(b->shader, AGX_SIZE_16, 2);
   agx_instr *collect = agx_collect_to(b, vector, 2);
   collect->src[0] = packs[0]->dest[0];
   collect->src[1] = packs[1]->dest[0];
   agx_unit_test(b, vector);
   for (unsigned i = 0; i < 2; ++i) {
      agx_instr *store = agx_alloc_instr(b, AGX_OPCODE_ST_TILE, 0, 3);
      store->src[0] = packs[i]->dest[0];
      store->src[1] = agx_immediate(1);
      store->src[2] = agx_null();
      store->format = AGX_FORMAT_I16;
      store->mask = 1;
      store->pixel_offset = i;
      agx_builder_insert(&b->cursor, store);
   }
   agx_unit_test(b, occupied);
   agx_ra(b->shader);
   ASSERT_FALSE(b->shader->ra_failed);
   EXPECT_EQ(packs[0]->dest[0].value & 1, 0u);
   EXPECT_EQ(packs[1]->dest[0].value, packs[0]->dest[0].value + 1);
   EXPECT_LT(packs[1]->dest[0].value, agx3_tile_store_bank(b->shader));
   ralloc_free(mem);
}

TEST(NativeForms, PhiWebAtRegisterBoundaryAvoidsEdgeCopies)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   key.is_helper = true;
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   b->shader->any_cf = true;
   b->shader->nir = rzalloc(mem, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;
   agx_block *entry = agx_start_block(b->shader);
   agx_block *join = agx_test_block(b->shader);
   agx_block_add_successor(entry, join);
   agx_index input = agx_preload(b, agx_register(30, AGX_SIZE_32));
   b->cursor = agx_after_block(join);
   agx_index value = agx_temp(b->shader, AGX_SIZE_32);
   agx_instr *phi = agx_phi_to(b, value, 1);
   phi->src[0] = input;
   agx_instr *use = agx_unit_test(b, value);
   agx_ra(b->shader);
   ASSERT_FALSE(b->shader->ra_failed);
   EXPECT_EQ(use->src[0].value, 30u);
   unsigned copies = 0;
   agx_foreach_instr_global(b->shader, I)
      copies += I->op == AGX_OPCODE_MOV;
   EXPECT_EQ(copies, 0u);
   ralloc_free(mem);
}

TEST(NativeForms, SpillPhiUsesActualOtherEdgeWithoutCopyingLiveSlot)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   key.is_helper = true;
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   b->shader->any_cf = true;
   b->shader->has_spill_pcopy_reserved = true;
   b->shader->nir = rzalloc(mem, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;
   agx_block *entry = agx_start_block(b->shader);
   agx_block *left = agx_test_block(b->shader);
   agx_block *right = agx_test_block(b->shader);
   agx_block *join = agx_test_block(b->shader);
   agx_block_add_successor(entry, left);
   agx_block_add_successor(entry, right);
   agx_block_add_successor(left, join);
   agx_block_add_successor(right, join);
   agx_index input = agx_preload(b, agx_register(20, AGX_SIZE_32));
   agx_index second = agx_preload(b, agx_register(22, AGX_SIZE_32));
   agx_index padding[4];
   for (agx_index &slot : padding) {
      slot = agx_temp(b->shader, AGX_SIZE_32);
      slot.memory = true;
      agx_mov_to(b, slot, input);
   }
   agx_index held = agx_temp(b->shader, AGX_SIZE_32);
   held.memory = true;
   agx_mov_to(b, held, input);
   agx_index other = agx_temp(b->shader, AGX_SIZE_32);
   other.memory = true;
   for (agx_block *block : {left, right}) {
      b->cursor = agx_after_block(block);
      if (block == right)
         agx_mov_to(b, other, second);
      for (agx_index slot : padding)
         agx_unit_test(b, agx_mov(b, slot));
   }
   b->cursor = agx_after_block(join);
   agx_index value = agx_temp(b->shader, AGX_SIZE_32);
   value.memory = true;
   agx_instr *phi = agx_phi_to(b, value, 2);
   phi->src[0] = held;
   phi->src[1] = other;
   agx_instr *reload = agx_mov_to(b, agx_temp(b->shader, AGX_SIZE_32), value);
   agx_unit_test(b, reload->dest[0]);
   agx_instr *held_reload =
      agx_mov_to(b, agx_temp(b->shader, AGX_SIZE_32), held);
   agx_unit_test(b, held_reload->dest[0]);
   agx_ra(b->shader);
   ASSERT_FALSE(b->shader->ra_failed);
   EXPECT_TRUE(reload->src[0].memory);
   EXPECT_EQ(reload->src[0].value, 10u);
   EXPECT_EQ(held_reload->src[0].value, 8u);
   unsigned stores = 0;
   agx_foreach_instr_global(b->shader, I)
      stores += I->op == AGX_OPCODE_MOV && I->dest[0].memory;
   EXPECT_EQ(stores, 7u);
   EXPECT_EQ(b->shader->scratch_size_B, 24u);
   ralloc_free(mem);
}

TEST(NativeForms, PhiAffinityKeepsCrossedEdgeInputsOutOfSwapCycles)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   key.is_helper = true;
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   b->shader->any_cf = true;
   b->shader->nir = rzalloc(mem, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;
   agx_block *entry = agx_start_block(b->shader);
   agx_block *left = agx_test_block(b->shader);
   agx_block *right = agx_test_block(b->shader);
   agx_block *join = agx_test_block(b->shader);
   agx_block_add_successor(entry, left);
   agx_block_add_successor(entry, right);
   agx_block_add_successor(left, join);
   agx_block_add_successor(right, join);
   agx_index x = agx_preload(b, agx_register(10, AGX_SIZE_16));
   agx_index y = agx_preload(b, agx_register(12, AGX_SIZE_16));
   b->cursor = agx_after_block(join);
   agx_index first = agx_temp(b->shader, AGX_SIZE_16);
   agx_index second = agx_temp(b->shader, AGX_SIZE_16);
   agx_instr *a = agx_phi_to(b, first, 2);
   a->src[0] = x;
   a->src[1] = y;
   agx_instr *c = agx_phi_to(b, second, 2);
   c->src[0] = y;
   c->src[1] = x;
   agx_instr *use_first = agx_unit_test(b, first);
   agx_instr *use_second = agx_unit_test(b, second);
   agx_ra(b->shader);
   ASSERT_FALSE(b->shader->ra_failed);
   EXPECT_EQ(use_first->src[0].value, 10u);
   EXPECT_EQ(use_second->src[0].value, 1u);
   unsigned copies = 0, swaps = 0;
   agx_foreach_instr_global(b->shader, I) {
      copies += I->op == AGX_OPCODE_MOV;
      swaps += I->op == AGX_OPCODE_SWAP;
   }
   EXPECT_EQ(copies, 3u);
   EXPECT_EQ(swaps, 0u);
   ralloc_free(mem);
}

TEST(NativeForms, PhiEdgeAffinityRespectsTileBankAndLiveValues)
{
   for (unsigned mode = 0; mode < 3; ++mode) {
      SCOPED_TRACE(mode);
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      key.is_helper = true;
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      b->shader->any_cf = true;
      b->shader->nir = rzalloc(mem, nir_shader);
      b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;
      agx_block *entry = agx_start_block(b->shader);
      agx_block *left = agx_test_block(b->shader);
      agx_block *right = agx_test_block(b->shader);
      agx_block *join = agx_test_block(b->shader);
      agx_block_add_successor(entry, left);
      agx_block_add_successor(entry, right);
      agx_block_add_successor(left, join);
      agx_block_add_successor(right, join);
      agx_index held =
         mode == 2 ? agx_preload(b, agx_register(10, AGX_SIZE_16)) : agx_null();
      agx_index incoming[2];
      unsigned side = 0;
      for (agx_block *block : {left, right}) {
         b->cursor = agx_after_block(block);
         unsigned reg = side == 0 ? 20 : (mode == 0 ? 10 : 22);
         incoming[side++] =
            mode == 2 ? held : agx_preload(b, agx_register(reg, AGX_SIZE_16));
      }
      b->cursor = agx_after_block(join);
      agx_index value = agx_temp(b->shader, AGX_SIZE_16);
      agx_instr *phi = agx_phi_to(b, value, 2);
      phi->src[0] = incoming[0];
      phi->src[1] = incoming[1];
      agx_instr *use = agx_unit_test(b, value);
      agx_instr *store = agx_alloc_instr(b, AGX_OPCODE_ST_TILE, 0, 3);
      store->src[0] = value;
      store->src[1] = agx_immediate(1);
      store->src[2] = agx_null();
      store->format = AGX_FORMAT_I16;
      store->mask = 1;
      agx_builder_insert(&b->cursor, store);
      agx_instr *held_use = mode == 2 ? agx_unit_test(b, held) : nullptr;
      agx_ra(b->shader);
      ASSERT_FALSE(b->shader->ra_failed);
      EXPECT_LT(use->src[0].value, agx3_tile_store_bank(b->shader));
      EXPECT_EQ(use->src[0].value, mode == 0 ? 10u : 1u);
      if (held_use) {
         EXPECT_EQ(held_use->src[0].value, 10u);
      }
      unsigned copies = 0;
      agx_foreach_instr_global(b->shader, I)
         copies += I->op == AGX_OPCODE_MOV;
      EXPECT_EQ(copies, mode == 0 ? 1u : 2u);
      ralloc_free(mem);
   }
}

TEST(NativeForms, PhiUsesOtherEdgeWhenPreferredRegisterIsLive)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   key.is_helper = true;
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   b->shader->any_cf = true;
   b->shader->nir = rzalloc(mem, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;
   agx_block *entry = agx_start_block(b->shader);
   agx_block *left = agx_test_block(b->shader);
   agx_block *right = agx_test_block(b->shader);
   agx_block *join = agx_test_block(b->shader);
   agx_block_add_successor(entry, left);
   agx_block_add_successor(entry, right);
   agx_block_add_successor(left, join);
   agx_block_add_successor(right, join);
   agx_index held = agx_preload(b, agx_register(10, AGX_SIZE_16));
   b->cursor = agx_after_block(right);
   agx_index other = agx_preload(b, agx_register(12, AGX_SIZE_16));
   b->cursor = agx_after_block(join);
   agx_index value = agx_temp(b->shader, AGX_SIZE_16);
   agx_instr *phi = agx_phi_to(b, value, 2);
   phi->src[0] = held;
   phi->src[1] = other;
   agx_instr *use = agx_unit_test(b, value);
   agx_instr *held_use = agx_unit_test(b, held);
   agx_ra(b->shader);
   ASSERT_FALSE(b->shader->ra_failed);
   EXPECT_EQ(use->src[0].value, 12u);
   EXPECT_EQ(held_use->src[0].value, 10u);
   unsigned copies = 0;
   agx_foreach_instr_global(b->shader, I)
      copies += I->op == AGX_OPCODE_MOV;
   EXPECT_EQ(copies, 1u);
   ralloc_free(mem);
}

TEST(NativeForms, PhiWithAgreeingInputsIsAssignedBeforeEarlierPhis)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   key.is_helper = true;
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   b->shader->any_cf = true;
   b->shader->nir = rzalloc(mem, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;
   agx_block *entry = agx_start_block(b->shader);
   agx_block *left = agx_test_block(b->shader);
   agx_block *right = agx_test_block(b->shader);
   agx_block *join = agx_test_block(b->shader);
   agx_block_add_successor(entry, left);
   agx_block_add_successor(entry, right);
   agx_block_add_successor(left, join);
   agx_block_add_successor(right, join);
   agx_index incoming[2];
   for (unsigned side = 0; side < 2; ++side) {
      b->cursor = agx_after_block(side ? right : left);
      incoming[side] = agx_preload(b, agx_register(2, AGX_SIZE_32));
   }
   b->cursor = agx_after_block(join);
   /* The constant phi comes first and has no register to follow. */
   agx_index constant = agx_temp(b->shader, AGX_SIZE_32);
   agx_instr *first = agx_phi_to(b, constant, 2);
   first->src[0] = agx_immediate(1);
   first->src[1] = agx_immediate(2);
   agx_index value = agx_temp(b->shader, AGX_SIZE_32);
   agx_instr *phi = agx_phi_to(b, value, 2);
   phi->src[0] = incoming[0];
   phi->src[1] = incoming[1];
   agx_instr *use = agx_unit_test(b, value);
   agx_instr *constant_use = agx_unit_test(b, constant);
   agx_ra(b->shader);
   ASSERT_FALSE(b->shader->ra_failed);
   EXPECT_EQ(use->src[0].value, 2u);
   EXPECT_NE(constant_use->src[0].value, 2u);
   unsigned copies = 0;
   agx_foreach_instr_global(b->shader, I)
      copies += I->op == AGX_OPCODE_MOV;
   EXPECT_EQ(copies, 0u);
   ralloc_free(mem);
}

TEST(NativeForms, PhiFallbackLeavesLaterPhiInputsFree)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   key.is_helper = true;
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   b->shader->any_cf = true;
   b->shader->nir = rzalloc(mem, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;
   agx_block *entry = agx_start_block(b->shader);
   agx_block *header = agx_test_block(b->shader);
   agx_block *body = agx_test_block(b->shader);
   agx_block *exit = agx_test_block(b->shader);
   agx_block_add_successor(entry, header);
   agx_block_add_successor(header, body);
   agx_block_add_successor(header, exit);
   agx_block_add_successor(body, header);
   header->loop_header = true;
   agx_index input = agx_preload(b, agx_register(2, AGX_SIZE_32));
   b->cursor = agx_after_block(header);
   /* The counter phi comes first and knows none of its inputs yet. */
   agx_index counter = agx_temp(b->shader, AGX_SIZE_32);
   agx_index next_counter = agx_temp(b->shader, AGX_SIZE_32);
   agx_instr *count_phi = agx_phi_to(b, counter, 2);
   count_phi->src[0] = agx_immediate(0);
   count_phi->src[1] = next_counter;
   agx_index value = agx_temp(b->shader, AGX_SIZE_32);
   agx_index next_value = agx_temp(b->shader, AGX_SIZE_32);
   agx_instr *phi = agx_phi_to(b, value, 2);
   phi->src[0] = input;
   phi->src[1] = next_value;
   b->cursor = agx_after_block(body);
   agx_iadd_to(b, next_value, value, agx_immediate(1), 0);
   agx_iadd_to(b, next_counter, counter, agx_immediate(1), 0);
   b->cursor = agx_after_block(exit);
   agx_instr *use = agx_unit_test(b, value);
   agx_unit_test(b, counter);
   agx_ra(b->shader);
   ASSERT_FALSE(b->shader->ra_failed);
   EXPECT_EQ(use->src[0].value, 2u);
   unsigned copies = 0;
   agx_foreach_instr_global(b->shader, I)
      copies += I->op == AGX_OPCODE_MOV;
   EXPECT_EQ(copies, 0u);
   ralloc_free(mem);
}

TEST(NativeForms, TransientTextureVectorBecomesScalarBeforeAnyEviction)
{
   for (unsigned mode = 0; mode < 4; ++mode) {
      SCOPED_TRACE(mode);
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      b->shader->nir = rzalloc(mem, nir_shader);
      b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;
      agx_index held[3];
      held[0] = agx_preload(b, agx_register(0, AGX_SIZE_16));
      held[1] = agx_preload(b, agx_register(8, AGX_SIZE_16));
      held[2] = agx_preload(b, agx_register(18, AGX_SIZE_16));
      agx_index coords = agx_vec_temp(b->shader, AGX_SIZE_32, 2);
      agx_index coords_reg = agx_register(4, AGX_SIZE_32);
      coords_reg.channels_m1 = 1;
      agx_preload_to(b, coords, coords_reg);
      agx_index result = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
      agx_instr *sample = agx_texture_sample_to(
         b, result, coords, agx_zero(), agx_zero(),
         agx_uniform(24, AGX_SIZE_64), agx_uniform(28, AGX_SIZE_64),
         agx_null(), AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD,
         mode == 3 ? 1 : 15, false, false, false, AGX_GATHER_NONE, false);
      agx_index extra = agx_null();
      if (mode == 2)
         extra = agx_fadd(b, agx_uniform(0, AGX_SIZE_32),
                          agx_uniform(2, AGX_SIZE_32));
      agx_instr *split = agx_split(b, 4, result);
      for (unsigned i = 0; i < 4; ++i)
         split->dest[i] = mode == 3 && i
                            ? agx_null() : agx_temp(b->shader, AGX_SIZE_32);
      if (mode == 3) {
         extra = agx_fadd(b, agx_uniform(0, AGX_SIZE_32),
                          agx_uniform(2, AGX_SIZE_32));
         agx_export(b, extra, 12);
      }
      agx_index second = mode == 3 ? agx_uniform(4, AGX_SIZE_32)
                                   : split->dest[3];
      agx_unit_test(b, agx_fadd(b, split->dest[0], second));
      if (mode != 3) {
         agx_unit_test(b, split->dest[1]);
         agx_unit_test(b, split->dest[2]);
      }
      if (mode == 1)
         agx_unit_test(b, result);
      if (mode == 2)
         agx_unit_test(b, extra);
      for (agx_index value : held)
         agx_unit_test(b, value);
      agx_ra(b->shader);
      ASSERT_FALSE(b->shader->ra_failed);
      EXPECT_EQ(sample->dest[0].value, mode == 0 ? 10u : 24u);
      EXPECT_EQ(agx_channels(sample->dest[0]), 4u);
      if (mode == 0) {
         EXPECT_EQ(b->shader->max_reg, 18u);
         for (unsigned i = 0; i < 4; ++i)
            EXPECT_EQ(split->dest[i].value, 10u + 2 * i);
      }
      ralloc_free(mem);
   }
}

TEST(NativeForms, TransientTextureResultPreservesEarlierAsyncWriteSpan)
{
   for (bool complete : {false, true}) {
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      b->shader->nir = rzalloc(mem, nir_shader);
      b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;
      agx_index low = agx_preload(b, agx_register(0, AGX_SIZE_16));
      agx_index high = agx_preload(b, agx_register(18, AGX_SIZE_16));
      agx_index x = agx_preload(b, agx_register(4, AGX_SIZE_32));
      agx_index y = agx_preload(b, agx_register(6, AGX_SIZE_32));
      agx_publish_texture_scalar_agx3(b, x, y, agx_null(), 1);
      agx_index resource = agx_get_vec_index(1, AGX_SIZE_32, 2);
      resource.type = AGX_INDEX_RESOURCE;
      agx_index previous = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
      agx_instr *first = agx_texture_sample_to(
         b, previous, resource, agx_zero(), agx_zero(),
         agx_uniform(24, AGX_SIZE_64), agx_uniform(28, AGX_SIZE_64),
         agx_null(), AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 8,
         false, false, false, AGX_GATHER_NONE, false);
      agx_unit_test(b, low);
      agx_instr *parts = agx_split(b, 4, previous);
      for (unsigned i = 0; i < 3; ++i)
         parts->dest[i] = agx_null();
      parts->dest[3] = agx_temp(b->shader, AGX_SIZE_32);
      agx_index retained = parts->dest[3];
      if (complete)
         retained = agx_fadd(b, retained, agx_uniform(0, AGX_SIZE_32));
      agx_index result = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
      agx_instr *sample = agx_texture_sample_to(
         b, result, resource, agx_zero(), agx_zero(),
         agx_uniform(24, AGX_SIZE_64), agx_uniform(28, AGX_SIZE_64),
         agx_null(), AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 15,
         false, false, false, AGX_GATHER_NONE, false);
      agx_instr *split = agx_split(b, 4, result);
      for (unsigned i = 0; i < 4; ++i)
         split->dest[i] = agx_temp(b->shader, AGX_SIZE_32);
      agx_unit_test(b, agx_fadd(b, split->dest[0], split->dest[3]));
      agx_unit_test(b, split->dest[1]);
      agx_unit_test(b, split->dest[2]);
      agx_unit_test(b, retained);
      if (complete)
         agx_unit_test(b, parts->dest[3]);
      agx_unit_test(b, low);
      agx_unit_test(b, high);
      agx_ra(b->shader);
      ASSERT_FALSE(b->shader->ra_failed);
      EXPECT_EQ(first->dest[0].value, 8u);
      EXPECT_EQ(sample->dest[0].value, complete ? 4u : 24u);
      EXPECT_EQ(agx_channels(sample->dest[0]), 4u);
      ralloc_free(mem);
   }
}

TEST(NativeForms, BoundedUnaryRegistersUnderPressure)
{
   for (agx_opcode op : {AGX_OPCODE_TRUNC, AGX_OPCODE_DFDX, AGX_OPCODE_DFDY}) {
      void *mem = ralloc_context(NULL);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      agx_builder *b = agx_test_builder(mem);
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      b->shader->nir = rzalloc(mem, nir_shader);
      b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;
      agx_index live[64];
      for (unsigned i = 0; i < 64; ++i)
         live[i] = agx_preload(b, agx_register(2 * i, AGX_SIZE_32));
      agx_index input = agx_mov_imm(b, 32, 0x40500000);
      agx_index output = agx_temp(b->shader, AGX_SIZE_32);
      agx_instr *unary = agx_alloc_instr(b, op, 1, 1);
      unary->src[0] = input;
      unary->dest[0] = output;
      agx_builder_insert(&b->cursor, unary);
      agx_index sum = output;
      for (unsigned i = 0; i < 64; ++i)
         sum = agx_fadd(b, sum, live[i]);
      agx_export(b, sum, 8);
      agx_stop(b);
      agx_ra(b->shader);
      EXPECT_LT(unary->src[0].value, 128u) << op;
      EXPECT_LT(unary->dest[0].value, 128u) << op;
      unsigned high = 0;
      agx_foreach_instr_global(b->shader, I) {
         agx_foreach_dest(I, d)
            high += I->dest[d].type == AGX_INDEX_REGISTER && I->dest[d].value >= 128;
      }
      EXPECT_GT(high, 0u) << "Unrelated live values must retain the upper bank";
      ralloc_free(mem);
   }
}

TEST(NativeForms, NegatedSqrtStepReachesPacker)
{
   for (bool absolute : {false, true}) {
      void *mem = ralloc_context(NULL);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      agx_builder *b = agx_test_builder(mem);
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_COMPUTE;
      agx_index src = agx_register(112, AGX_SIZE_32);
      src.neg = true;
      src.abs = absolute;
      src.cache = true;
      agx_instr *step = agx_alloc_instr(b, AGX_OPCODE_SRSQRT, 1, 1);
      step->src[0] = src;
      step->dest[0] = agx_register(122, AGX_SIZE_32);
      agx_builder_insert(&b->cursor, step);
      agx_stop(b);
      util_dynarray bytes;
      util_dynarray_init(&bytes, mem);
      EXPECT_TRUE(agx3_pack_binary(b->shader, &bytes));
      EXPECT_GT(bytes.size, 4u);
      ralloc_free(mem);
   }
}

TEST(NativeForms, G17PHalfSqrtStepAcceptsLastUse)
{
   for (unsigned hint : {0u, 1u, 2u}) {
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_COMPUTE;
      b->shader->max_reg = 7;
      agx_index src = agx_register(3, AGX_SIZE_16);
      src.cache = hint == 1;
      src.discard = hint == 2;
      agx_srsqrt_to(b, agx_register(4, AGX_SIZE_32), src);
      agx_stop(b);
      util_dynarray binary;
      util_dynarray_init(&binary, mem);
      ASSERT_TRUE(agx3_pack_binary(b->shader, &binary));
      uint8_t expected[AGX3_ALU_MAX_LENGTH], length;
      ASSERT_TRUE(agx3_alu_sqrt_step_half(
         AGX_ISA_ENCODING_AGX3B, 4, 3, AGX3_ALU_HINT_CACHE, true, 0,
         expected, &length));
      ASSERT_GE(binary.size, length);
      EXPECT_EQ(memcmp(binary.data, expected, length), 0);
      util_dynarray_fini(&binary);
      ralloc_free(mem);
   }
}

TEST(NativeForms, G17PHalfComparisonSelectsIntegerBits)
{
   for (unsigned selected : {1u, 2u, 0xffu, 0x3c00u, 0x8000u, 0xffffu}) {
      for (unsigned arm : {2u, 3u}) {
         SCOPED_TRACE(selected);
         SCOPED_TRACE(arm);
         void *mem = ralloc_context(NULL);
         agx_builder *b = agx_test_builder(mem);
         agx_shader_key key = {};
         key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
         key.dev.chip = AGX_CHIP_G17P;
         b->shader->key = &key;
         agx_index x = agx_preload(b, agx_register(4, AGX_SIZE_16));
         agx_index y = agx_preload(b, agx_register(5, AGX_SIZE_16));
         agx_index alternatives[] = {agx_immediate(selected), x};
         agx_instr *sel = agx_fcmpsel_to(
            b, agx_temp(b->shader, AGX_SIZE_16), x, y,
            alternatives[arm == 3], alternatives[arm == 2], AGX_FCOND_LT);
         agx_materialize_agx3_alu_constants(b->shader);
         const agx_instr *constant = NULL;
         agx_foreach_instr_global(b->shader, I) {
            if (I->nr_dests && agx_is_equiv(I->dest[0], sel->src[arm]))
               constant = I;
         }
         ASSERT_NE(constant, nullptr);
         EXPECT_EQ(constant->op, AGX_OPCODE_MOV_IMM);
         EXPECT_EQ(constant->imm, selected);
         EXPECT_EQ(constant->dest[0].size, AGX_SIZE_16);
         ralloc_free(mem);
      }
   }
}

TEST(NativeForms, HalfModifiedMultiplyAndSelect)
{
   for (bool select : {false, true}) {
      SCOPED_TRACE(select);
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "half modified operands");
      nir_def *input = agx_test_load_preamble(&b, 4, 32, 4);
      nir_def *x = nir_f2f16(&b, nir_channel(&b, input, 0));
      nir_def *result;
      if (select) {
         nir_def *cmp = agx_test_load_preamble(&b, 2, 16, 12);
         x = nir_channel(&b, cmp, 0);
         nir_def *y = nir_channel(&b, cmp, 1);
         result = nir_bcsel(&b, nir_flt(&b, x, nir_fneg(&b, y)),
                           nir_channel(&b, input, 2),
                           nir_channel(&b, input, 3));
      } else {
         result = nir_f2f32(
            &b, nir_fmul(&b, nir_fabs(&b, x), nir_imm_floatN_t(&b, 0.5, 16)));
      }
      agx_test_store_local_pixel_agx(
         &b, nir_vec4(&b, nir_channel(&b, input, 0), result, result, result),
         nir_imm_intN_t(&b, 0xff, 16), nir_undef(&b, 2, 16), 0, 0xf,
         PIPE_FORMAT_R8_UNORM);
      b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      key.no_preamble = true;
      key.reserved_preamble = 16;
      key.fs.nr_samples = 1;
      agx_preprocess_nir(b.shader);
      agx_shader_part part = {};
      EXPECT_TRUE(agx_compile_shader_nir(b.shader, &key, &part));
      EXPECT_GT(part.info.main_size, 0u);
      free(part.binary);
   }
}

TEST(NativeForms, ImageStoreUsesDataTypeAcrossChipsAndFormats)
{
   const struct {
      nir_alu_type type;
      pipe_format format;
   } cases[] = {
      {nir_type_uint32, PIPE_FORMAT_NONE},
      {nir_type_uint32, PIPE_FORMAT_R32_UINT},
      {nir_type_uint32, PIPE_FORMAT_R32G32B32A32_UINT},
      {nir_type_int32, PIPE_FORMAT_NONE},
      {nir_type_int32, PIPE_FORMAT_R32_SINT},
      {nir_type_int32, PIPE_FORMAT_R32G32B32A32_SINT},
      {nir_type_float32, PIPE_FORMAT_NONE},
      {nir_type_float32, PIPE_FORMAT_R32_FLOAT},
      {nir_type_float32, PIPE_FORMAT_R32G32B32A32_FLOAT},
      {nir_type_invalid, PIPE_FORMAT_R32_UINT},
   };
   agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   /* The store is the program's last instruction before its end. Code around
    * it may differ by chip (G17P runs its own uniform-copy CSE, which changes
    * register allocation), so compare only the store. Integer and float data
    * each select one encoding whatever the format; a typeless store takes its
    * type from the format. */
   std::vector<uint8_t> reference[2];
   for (const auto &c : cases) {
      const bool is_float =
         nir_alu_type_get_base_type(c.type) == nir_type_float;
      for (agx_chip chip : {AGX_CHIP_G17P, AGX_CHIP_G16G}) {
         SCOPED_TRACE(::testing::Message() << chip << " " << c.type << " "
                                          << c.format);
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_COMPUTE, &agx_nir_options, "image store data type");
         nir_def *input = agx_test_load_preamble(&b, 4, 32, 4);
         nir_intrinsic_instr *store = nir_image_store(
            &b, nir_imm_intN_t(&b, 1, 16),
            nir_imm_ivec4(&b, 0, 0, 0, 0), nir_imm_intN_t(&b, 0, 16),
            input, nir_imm_intN_t(&b, 0, 16));
         nir_intrinsic_set_image_dim(store, GLSL_SAMPLER_DIM_2D);
         nir_intrinsic_set_format(store, c.format);
         nir_intrinsic_set_src_type(store, c.type);
         agx_shader_key key = {};
         key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
         key.dev.chip = chip;
         key.no_preamble = true;
         key.reserved_preamble = 16;
         agx_preprocess_nir(b.shader);
         agx_shader_part part = {};
         ASSERT_TRUE(agx_compile_shader_nir(b.shader, &key, &part));
         ASSERT_GT(part.info.binary_size, 0u);
         const uint8_t *code = static_cast<const uint8_t *>(part.binary);
         const unsigned size = part.info.binary_size;
         ASSERT_GE(size, AGX3_IMAGE_WRITE_LENGTH + end.length);
         EXPECT_EQ(memcmp(code + size - end.length, end.bytes, end.length), 0);
         const uint8_t *write =
            code + size - end.length - AGX3_IMAGE_WRITE_LENGTH;
         std::vector<uint8_t> encoded(write, write + AGX3_IMAGE_WRITE_LENGTH);
         if (reference[is_float].empty())
            reference[is_float] = encoded;
         else
            EXPECT_EQ(encoded, reference[is_float]);
         free(part.binary);
         ralloc_free(b.shader);
      }
   }
   EXPECT_NE(reference[0], reference[1]);
}

TEST(NativeForms, VertexImageLoadRegisterLod)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "vertex image load register LOD");
   nir_def *id = nir_load_vertex_id(&b);
   nir_def *coord = nir_vec4(&b, id, id, id, nir_imm_int(&b, 0));
   nir_def *value = nir_image_load(
      &b, 4, 32, nir_imm_intN_t(&b, 3, 16), coord, nir_imm_int(&b, 0),
      agx_test_load_preamble(&b, 1, 16, 0));
   nir_intrinsic_instr *load = nir_def_as_intrinsic(value);
   nir_intrinsic_set_image_dim(load, GLSL_SAMPLER_DIM_3D);
   nir_intrinsic_set_format(load, PIPE_FORMAT_R32G32B32A32_UINT);
   nir_intrinsic_set_dest_type(load, nir_type_uint32);
   for (unsigned c = 0; c < 4; ++c) {
      nir_store_uvs_agx(&b, nir_imm_float(&b, c == 3 ? 1.0f : 0.0f),
                       nir_imm_intN_t(&b, c, 16));
      nir_store_uvs_agx(&b, nir_channel(&b, value, c),
                       nir_imm_intN_t(&b, 4 + c, 16));
   }
   b.shader->info.outputs_written = VARYING_BIT_POS |
                                   BITFIELD64_BIT(VARYING_SLOT_VAR0);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   key.no_preamble = true;
   key.reserved_preamble = 64;
   agx_preprocess_nir(b.shader);
   agx_shader_part part = {};
   EXPECT_TRUE(agx_compile_shader_nir(b.shader, &key, &part));
   EXPECT_GT(part.info.main_size, 0u);
   free(part.binary);
}

TEST(NativeForms, TexturePublicationWaitPreservesUnrelatedResult)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   agx_builder *b = agx_test_builder(mem);
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_COMPUTE;
   b->shader->max_reg = 31;
   agx_index coords = agx_register(10, AGX_SIZE_32);
   coords.channels_m1 = 1;
   agx_device_load_to(b, agx_register(0, AGX_SIZE_32),
                     agx_register(28, AGX_SIZE_64), agx_immediate(0),
                     AGX_FORMAT_I32, 1, 0, false);
   agx_device_load_to(b, coords, agx_register(28, AGX_SIZE_64),
                     agx_immediate(4), AGX_FORMAT_I32, 3, 0, false);
   agx_publish_texture_agx3(b, coords, agx_null(), 1);
   agx_index resource = coords;
   resource.type = AGX_INDEX_RESOURCE;
   resource.value = 1;
   agx_index result = agx_register(20, AGX_SIZE_32);
   result.channels_m1 = 3;
   agx_texture_sample_to(b, result, resource, agx_immediate(0),
                        agx_immediate(0), agx_register(24, AGX_SIZE_64),
                        agx_register(26, AGX_SIZE_64), agx_null(), AGX_DIM_2D,
                        AGX_LOD_MODE_LOD_MIN, 15, false, false, false,
                        AGX_GATHER_NONE, false);
   agx_stop(b);
   util_dynarray binary;
   util_dynarray_init(&binary, mem);
   ASSERT_TRUE(agx3_pack_binary(b->shader, &binary));
   uint8_t carrier[AGX3_ALU_MAX_LENGTH], len = 0;
   ASSERT_TRUE(agx3_alu_band_identity_wait(AGX_ISA_ENCODING_AGX3B, 10,
                                         2, carrier, &len));
   const uint8_t *bytes = (const uint8_t *)binary.data;
   bool found = false;
   for (unsigned i = 0; i + len <= binary.size; ++i)
      found |= memcmp(bytes + i, carrier, len) == 0;
   EXPECT_TRUE(found);
   util_dynarray_fini(&binary);
   ralloc_free(mem);
}

TEST(NativeForms, MinmaxReusesUniformCopiedForComparison)
{
   for (agx_fcond cond : {AGX_FCOND_LTN, AGX_FCOND_GTN}) {
      for (unsigned immediate : {0u, 176u}) {
         void *mem = ralloc_context(NULL);
         agx_shader_key key = {};
         key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
         key.dev.chip = AGX_CHIP_G17P;
         agx_builder *b = agx_test_builder(mem);
         b->shader->key = &key;
         agx_instr *I = agx_alloc_instr(b, AGX_OPCODE_FCMPSEL, 1, 4);
         I->dest[0] = agx_temp(b->shader, AGX_SIZE_32);
         I->src[0] = I->src[2] = agx_uniform(8, AGX_SIZE_32);
         I->src[1] = I->src[3] = agx_immediate(immediate);
         I->fcond = cond;
         agx_builder_insert(&b->cursor, I);
         agx_lower_uniform_sources(b->shader);
         EXPECT_EQ(I->src[0].type, AGX_INDEX_NORMAL);
         EXPECT_TRUE(agx_is_equiv(I->src[0], I->src[2]));
         ralloc_free(mem);
      }
   }
}

TEST(NativeForms, AtomicVertexOutputsReachTerminalBundle)
{
   for (bool position : {false, true}) {
      SCOPED_TRACE(position);
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_VERTEX, &agx_nir_options, "atomic vertex outputs");
      nir_def *address = agx_test_load_preamble(&b, 1, 64, 0);
      nir_def *value = nir_global_atomic_agx(
         &b, 32, address, nir_imm_int(&b, 0), nir_imm_int(&b, 1));
      nir_intrinsic_set_atomic_op(nir_def_as_intrinsic(value), nir_atomic_op_iadd);
      nir_store_uvs_agx(&b, value, nir_imm_intN_t(&b, 4, 16));
      nir_def *second = nir_global_atomic_agx(
         &b, 32, address, nir_imm_int(&b, 4), nir_imm_int(&b, -1));
      nir_intrinsic_set_atomic_op(nir_def_as_intrinsic(second), nir_atomic_op_iadd);
      nir_store_uvs_agx(&b, nir_iadd(&b, value, second),
                       nir_imm_intN_t(&b, 5, 16));
      if (position) {
         for (unsigned c = 0; c < 4; ++c)
            nir_store_uvs_agx(&b, nir_imm_float(&b, c == 3 ? 1.0 : 0.0),
                             nir_imm_intN_t(&b, c, 16));
      }
      b.shader->info.outputs_written = BITFIELD64_BIT(VARYING_SLOT_VAR0) |
                                     (position ? VARYING_BIT_POS : 0);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      key.no_preamble = true;
      key.reserved_preamble = 4;
      agx_preprocess_nir(b.shader);
      agx_shader_part part = {};
      EXPECT_TRUE(agx_compile_shader_nir(b.shader, &key, &part));
      EXPECT_GT(part.info.main_size, 0u);
      free(part.binary);
      ralloc_free(b.shader);
   }
}

TEST(NativeForms, RectangleImplicitLodOutsideFragment)
{
   for (mesa_shader_stage stage : {MESA_SHADER_VERTEX, MESA_SHADER_COMPUTE,
                                   MESA_SHADER_FRAGMENT}) {
      for (bool sampler_bias : {false, true}) {
         SCOPED_TRACE(stage);
         SCOPED_TRACE(sampler_bias);
         nir_builder b = nir_builder_init_simple_shader(
            stage, &agx_nir_options, "rectangle implicit lod");
         nir_tex_instr *tex = nir_tex_instr_create(b.shader, 1);
         tex->op = nir_texop_tex;
         tex->sampler_dim = GLSL_SAMPLER_DIM_RECT;
         tex->dest_type = nir_type_float32;
         tex->coord_components = 2;
         tex->src[0] = nir_tex_src_for_ssa(
            nir_tex_src_coord, nir_imm_vec2(&b, 3.0, 5.0));
         nir_def_init(&tex->instr, &tex->def, 4, 32);
         nir_builder_instr_insert(&b, &tex->instr);
         agx_nir_lower_g17p_texture_early(b.shader, sampler_bias, true);
         EXPECT_EQ(tex->sampler_dim, GLSL_SAMPLER_DIM_RECT);
         EXPECT_GE(nir_tex_instr_src_index(tex, nir_tex_src_coord), 0);
         if (stage == MESA_SHADER_FRAGMENT) {
            EXPECT_EQ(tex->op, sampler_bias ? nir_texop_txb : nir_texop_tex);
         } else {
            EXPECT_EQ(tex->op, nir_texop_txl);
            EXPECT_LT(nir_tex_instr_src_index(tex, nir_tex_src_bias), 0);
            int lod = nir_tex_instr_src_index(tex, nir_tex_src_lod);
            ASSERT_GE(lod, 0);
            EXPECT_EQ(nir_src_as_float(tex->src[lod].src), 0.0);
         }
         nir_validate_shader(b.shader, "rectangle level zero");
         ralloc_free(b.shader);
      }
   }
}

TEST(NativeForms, ImageWriteHighUniformHeap)
{
   for (unsigned uniform : {252u, 256u, 296u, 508u}) {
      void *mem = ralloc_context(NULL);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G16G;
      agx_builder *b = agx_test_builder(mem);
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_COMPUTE;
      agx_index data = agx_register(16, AGX_SIZE_32);
      data.channels_m1 = 3;
      agx_index coord = agx_register(24, AGX_SIZE_32);
      coord.channels_m1 = 1;
      agx_image_write(b, data, coord, agx_zero(),
                      agx_uniform(uniform, AGX_SIZE_64),
                      agx_register(28, AGX_SIZE_32), AGX_DIM_2D, true);
      agx_stop(b);
      util_dynarray binary;
      util_dynarray_init(&binary, mem);
      EXPECT_TRUE(agx3_pack_binary(b->shader, &binary)) << uniform;
      util_dynarray_fini(&binary);
      ralloc_free(mem);
   }
}

TEST(NativeForms, ImageFenceWaitsRotatingWriteSlots)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   agx_builder *b = agx_test_builder(mem);
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_COMPUTE;
   agx_index data = agx_register(16, AGX_SIZE_32);
   data.channels_m1 = 3;
   agx_index coord = agx_register(24, AGX_SIZE_32);
   coord.channels_m1 = 1;
   for (unsigned i = 0; i < 3; ++i) {
      agx_image_write(b, data, coord, agx_zero(), agx_uniform(8, AGX_SIZE_64),
                      agx_register(28, AGX_SIZE_32), AGX_DIM_2D, true);
      agx_scoped_barrier_agx3(b, AGX3_BARRIER_SCOPE_TEXTURE_FENCE_WAIT);
   }
   agx_stop(b);
   util_dynarray binary;
   util_dynarray_init(&binary, mem);
   ASSERT_TRUE(agx3_pack_binary(b->shader, &binary));

   agx3_encoded_scoped_barrier fence;
   ASSERT_TRUE(agx3_pack_scoped_barrier(AGX_ISA_ENCODING_AGX3B,
                                       AGX3_BARRIER_SCOPE_TEXTURE_FENCE_WAIT,
                                       &fence));
   const uint8_t *bytes = (const uint8_t *)binary.data;
   unsigned fences = 0;
   for (unsigned i = 0; i + fence.length <= binary.size; ++i) {
      if (memcmp(bytes + i, fence.bytes, fence.length))
         continue;
      /* The first write uses the fence's embedded slot1 wait. Later writes
       * rotate to slots2/3 and must finish before the cache fence, not after. */
      if (fences > 0) {
         uint8_t carrier[AGX3_ALU_MAX_LENGTH], len = 0;
         ASSERT_TRUE(agx3_alu_band_identity_wait(
            AGX_ISA_ENCODING_AGX3B, 0, 1u << fences, carrier, &len));
         ASSERT_GE(i, len);
         EXPECT_EQ(memcmp(bytes + i - len, carrier, len), 0) << fences;
      }
      ++fences;
      i += fence.length - 1;
   }
   EXPECT_EQ(fences, 3u);
   util_dynarray_fini(&binary);
   ralloc_free(mem);
}

TEST(NativeForms, ScalarCubeCoordinateAdmissionAndRetainedConsumers)
{
   for (unsigned invalid = 0; invalid < 19; ++invalid) {
      void *mem = ralloc_context(NULL);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      agx_builder *b = agx_test_builder(mem);
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_index sources[3];
      for (unsigned c = 0; c < 3; ++c)
         sources[c] = agx_preload(b, agx_register(4 + 2 * c, AGX_SIZE_32));
      if (invalid == 1)
         sources[0].abs = true;
      if (invalid == 2)
         sources[1].neg = true;
      if (invalid == 3)
         sources[2].size = AGX_SIZE_16;
      if (invalid == 4)
         sources[1] = agx_uniform(8, AGX_SIZE_32);
      if (invalid == 5)
         sources[2] = agx_register(128, AGX_SIZE_32);
      agx_index vector = agx_vec_temp(b->shader, AGX_SIZE_32, 3);
      agx_instr *collect = agx_collect_to(b, vector, 3);
      for (unsigned c = 0; c < 3; ++c)
         collect->src[c] = sources[c];
      agx_index out = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
      agx_instr *original = agx_texture_sample_to(
         b, out, vector, agx_zero(), agx_zero(), agx_uniform(24, AGX_SIZE_64),
         agx_uniform(28, AGX_SIZE_64), agx_null(), AGX_DIM_CUBE,
         AGX_LOD_MODE_AUTO_LOD, 15, false, false, false, AGX_GATHER_NONE,
         false);
      if (invalid == 6)
         original->dim = AGX_DIM_CUBE_ARRAY;
      if (invalid == 7)
         original->lod_mode = AGX_LOD_MODE_LOD_MIN;
      if (invalid == 8)
         original->shadow = true;
      if (invalid == 9)
         original->offset = true;
      if (invalid == 10)
         original->query_lod = true;
      if (invalid == 11)
         original->sparse = true;
      if (invalid == 12)
         original->gather = AGX_GATHER_R;
      if (invalid == 13)
         original->src[3] = agx_register(24, AGX_SIZE_64);
      if (invalid == 14)
         original->src[1] = agx_immediate(1);
      if (invalid == 15)
         b->shader->stage = MESA_SHADER_COMPUTE;
      if (invalid == 16)
         b->shader->is_preamble = true;
      if (invalid == 17)
         key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
      if (invalid == 18)
         agx_unit_test(b, vector);
      agx_instr *use = agx_unit_test(b, out);
      bool progress = agx3_lower_scalar_cube_coordinates(b->shader);
      EXPECT_EQ(progress, invalid == 0) << invalid;
      if (!progress) {
         EXPECT_EQ(original->nr_srcs, 6u);
         EXPECT_FALSE(original->agx3_cube_scalar_coords);
      } else {
         agx_instr *sample = nullptr;
         agx_foreach_instr_global(b->shader, I)
            if (I->op == AGX_OPCODE_TEXTURE_SAMPLE)
               sample = I;
         ASSERT_NE(sample, nullptr);
         EXPECT_EQ(sample->nr_srcs, 8u);
         EXPECT_TRUE(sample->agx3_cube_scalar_coords);
         EXPECT_TRUE(agx_is_equiv(sample->dest[0], out));
         for (unsigned c = 0; c < 3; ++c)
            EXPECT_TRUE(agx_is_equiv(sample->src[c ? c + 5 : 0], sources[c]));
         EXPECT_TRUE(agx_is_equiv(use->src[0], out));
         agx_dce(b->shader, true);
         unsigned collects = 0;
         agx_foreach_instr_global(b->shader, I)
            collects += I->op == AGX_OPCODE_COLLECT;
         EXPECT_EQ(collects, 0u);
         EXPECT_FALSE(agx3_lower_scalar_cube_coordinates(b->shader));
      }
      ralloc_free(mem);
   }
}

TEST(NativeForms, ScalarCubePacksNoncontiguousCemOperands)
{
   for (unsigned flags = 0; flags < 8; ++flags) {
      void *mem = ralloc_context(NULL);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      agx_builder *b = agx_test_builder(mem);
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      b->shader->max_reg = 31;
      agx_index vector = agx_vec_temp(b->shader, AGX_SIZE_32, 3);
      agx_instr *collect = agx_collect_to(b, vector, 3);
      const unsigned source_regs[] = {4, 6, 20};
      for (unsigned c = 0; c < 3; ++c) {
         collect->src[c] = agx_register(source_regs[c], AGX_SIZE_32);
         collect->src[c].discard = flags & (1u << c);
      }
      agx_index result = agx_register(24, AGX_SIZE_32);
      result.channels_m1 = 3;
      agx_texture_sample_to(b, result, vector, agx_zero(), agx_zero(),
                            agx_uniform(24, AGX_SIZE_64),
                            agx_uniform(28, AGX_SIZE_64), agx_null(),
                            AGX_DIM_CUBE, AGX_LOD_MODE_AUTO_LOD, 15, false,
                            false, false, AGX_GATHER_NONE, false);
      agx_stop(b);
      ASSERT_TRUE(agx3_lower_scalar_cube_coordinates(b->shader));
      agx_remove_instruction(collect);
      util_dynarray binary;
      util_dynarray_init(&binary, mem);
      ASSERT_TRUE(agx3_pack_binary(b->shader, &binary)) << flags;
      const uint8_t *bytes = (const uint8_t *)binary.data;
      for (unsigned axis = 0; axis < 3; ++axis) {
         uint8_t cem[AGX3_TEX_MAX_LENGTH], length = 0;
         const bool discard[3] = {
            axis == 2 && bool(flags & 1),
            axis == 2 && bool(flags & 2),
            axis == 2 && bool(flags & 4),
         };
         ASSERT_TRUE(
            agx3_tex_cube_cem_calc_hints_to(AGX_ISA_ENCODING_AGX3B, 4, 6, 20,
                                            axis == 0   ? 32
                                            : axis == 1 ? 36
                                                        : 38,
                                            axis, discard, cem, &length));
         unsigned matches = 0;
         for (unsigned i = 0; i + length <= binary.size; ++i)
            matches += memcmp(bytes + i, cem, length) == 0;
         EXPECT_EQ(matches, 1u) << flags << ":" << axis;
      }
      util_dynarray_fini(&binary);
      ralloc_free(mem);
   }
}

TEST(NativeForms, ScalarCubeWaitsForAsynchronousYZ)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   agx_builder *b = agx_test_builder(mem);
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   b->shader->max_reg = 31;
   agx_index input = agx_register(16, AGX_SIZE_32);
   input.channels_m1 = 1;
   agx_index pending = agx_register(4, AGX_SIZE_32);
   pending.channels_m1 = 3;
   agx_texture_sample_to(
      b, pending, input, agx_zero(), agx_zero(), agx_uniform(24, AGX_SIZE_64),
      agx_uniform(28, AGX_SIZE_64), agx_null(), AGX_DIM_2D,
      AGX_LOD_MODE_AUTO_LOD, 15, false, false, false, AGX_GATHER_NONE, false);
   agx_index vector = agx_vec_temp(b->shader, AGX_SIZE_32, 3);
   agx_instr *collect = agx_collect_to(b, vector, 3);
   const unsigned source_regs[] = {20, 4, 6};
   for (unsigned c = 0; c < 3; ++c)
      collect->src[c] = agx_register(source_regs[c], AGX_SIZE_32);
   agx_index result = agx_register(24, AGX_SIZE_32);
   result.channels_m1 = 3;
   agx_texture_sample_to(
      b, result, vector, agx_zero(), agx_zero(), agx_uniform(24, AGX_SIZE_64),
      agx_uniform(28, AGX_SIZE_64), agx_null(), AGX_DIM_CUBE,
      AGX_LOD_MODE_AUTO_LOD, 15, false, false, false, AGX_GATHER_NONE, false);
   agx_stop(b);
   ASSERT_TRUE(agx3_lower_scalar_cube_coordinates(b->shader));
   agx_remove_instruction(collect);
   util_dynarray binary;
   util_dynarray_init(&binary, mem);
   ASSERT_TRUE(agx3_pack_binary(b->shader, &binary));
   const uint8_t *bytes = (const uint8_t *)binary.data;
   uint8_t cem[AGX3_TEX_MAX_LENGTH], cem_len = 0;
   ASSERT_TRUE(agx3_tex_cube_cem_calc_to(AGX_ISA_ENCODING_AGX3B, 20, 4, 6, 32,
                                         0, false, cem, &cem_len));
   unsigned cem_offset = binary.size;
   for (unsigned i = 0; i + cem_len <= binary.size; ++i)
      if (memcmp(bytes + i, cem, cem_len) == 0)
         cem_offset = i;
   ASSERT_LT(cem_offset, binary.size);
   bool waits = false;
   for (unsigned reg : {0u, 24u}) {
      for (unsigned slot = 0; slot < 6; ++slot) {
         uint8_t carrier[AGX3_ALU_MAX_LENGTH], length = 0;
         ASSERT_TRUE(agx3_alu_band_identity_wait(AGX_ISA_ENCODING_AGX3B, reg,
                                                 1u << slot, carrier, &length));
         for (unsigned i = 0; i + length <= cem_offset; ++i)
            waits |= memcmp(bytes + i, carrier, length) == 0;
      }
   }
   EXPECT_TRUE(waits);
   util_dynarray_fini(&binary);
   ralloc_free(mem);
}

TEST(NativeForms, NativeCubeIndependentLastUses)
{
   struct fixture {
      unsigned x, y, z, dest, axis, discard;
      uint8_t bytes[12];
   };
   const fixture vectors[] = {
      {4,
       6,
       14,
       0,
       0,
       0,
       {0x17, 0x01, 0x54, 0x00, 0x03, 0x04, 0x0d, 0x3a, 0x54, 0x21, 0x92,
        0x00}},
      {4,
       6,
       14,
       8,
       1,
       0,
       {0x17, 0x01, 0x54, 0x04, 0x03, 0x04, 0x0d, 0x3a, 0x54, 0x21, 0x92,
        0x04}},
      {4,
       6,
       14,
       4,
       2,
       7,
       {0x17, 0x01, 0x54, 0x02, 0x03, 0x04, 0x0c, 0x38, 0x50, 0x2f, 0x92,
        0x08}},
      {8,
       4,
       6,
       0,
       0,
       0,
       {0x17, 0x01, 0x54, 0x00, 0x03, 0x08, 0x09, 0x1a, 0x54, 0x21, 0x92,
        0x00}},
      {8,
       4,
       6,
       10,
       1,
       0,
       {0x17, 0x01, 0x54, 0x05, 0x03, 0x08, 0x09, 0x1a, 0x54, 0x21, 0x92,
        0x04}},
      {8,
       4,
       6,
       4,
       2,
       6,
       {0x17, 0x01, 0x54, 0x02, 0x03, 0x08, 0x08, 0x18, 0x50, 0x2d, 0x92,
        0x08}},
      {4,
       8,
       6,
       0,
       0,
       0,
       {0x17, 0x01, 0x54, 0x00, 0x03, 0x04, 0x11, 0x1a, 0x54, 0x21, 0x92,
        0x00}},
      {4,
       8,
       6,
       10,
       1,
       0,
       {0x17, 0x01, 0x54, 0x05, 0x03, 0x04, 0x11, 0x1a, 0x54, 0x21, 0x92,
        0x04}},
      {4,
       8,
       6,
       4,
       2,
       5,
       {0x17, 0x01, 0x54, 0x02, 0x03, 0x04, 0x10, 0x18, 0x50, 0x2b, 0x92,
        0x08}},
      {4,
       6,
       8,
       0,
       0,
       0,
       {0x17, 0x01, 0x54, 0x00, 0x03, 0x04, 0x0d, 0x22, 0x54, 0x21, 0x92,
        0x00}},
      {4,
       6,
       8,
       10,
       1,
       0,
       {0x17, 0x01, 0x54, 0x05, 0x03, 0x04, 0x0d, 0x22, 0x54, 0x21, 0x92,
        0x04}},
      {4,
       6,
       8,
       4,
       2,
       3,
       {0x17, 0x01, 0x54, 0x02, 0x03, 0x04, 0x0c, 0x20, 0x50, 0x27, 0x92,
        0x08}},
   };
   for (const fixture &v : vectors) {
      const bool discard[3] = {bool(v.discard & 1), bool(v.discard & 2),
                               bool(v.discard & 4)};
      uint8_t bytes[AGX3_TEX_MAX_LENGTH], length = 0;
      ASSERT_TRUE(agx3_tex_cube_cem_calc_hints_to(AGX_ISA_ENCODING_AGX3B, v.x,
                                                  v.y, v.z, v.dest, v.axis,
                                                  discard, bytes, &length));
      ASSERT_EQ(length, 12u);
      EXPECT_EQ(memcmp(bytes, v.bytes, length), 0);
   }
}

TEST(NativeForms, ScalarCubeAllocationPreservesSharedInputs)
{
   for (unsigned mode = 0; mode < 6; ++mode) {
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      b->shader->nir = rzalloc(mem, nir_shader);
      b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;
      agx_index occupied = agx_preload(b, agx_register(0, AGX_SIZE_32));
      agx_index occupied2 = agx_preload(b, agx_register(2, AGX_SIZE_32));
      agx_index sources[3];
      for (unsigned c = 0; c < 3; ++c)
         sources[c] = agx_preload(b, agx_register(4 + 2 * c, AGX_SIZE_32));
      if (mode == 2)
         sources[2] = sources[1];
      agx_index vector = agx_vec_temp(b->shader, AGX_SIZE_32, 3);
      agx_instr *collect = agx_collect_to(b, vector, 3);
      for (unsigned c = 0; c < 3; ++c)
         collect->src[c] = sources[c];
      agx_index result = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
      agx_texture_sample_to(b, result, vector, agx_zero(), agx_zero(),
                            agx_uniform(24, AGX_SIZE_64),
                            agx_uniform(28, AGX_SIZE_64), agx_null(),
                            AGX_DIM_CUBE, AGX_LOD_MODE_AUTO_LOD, 15, false,
                            false, false, AGX_GATHER_NONE, false);
      if (mode == 3) {
         agx_index second = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
         agx_texture_sample_to(b, second, vector, agx_zero(), agx_zero(),
                               agx_uniform(32, AGX_SIZE_64),
                               agx_uniform(36, AGX_SIZE_64), agx_null(),
                               AGX_DIM_CUBE, AGX_LOD_MODE_AUTO_LOD, 15, false,
                               false, false, AGX_GATHER_NONE, false);
         agx_instr *split = agx_split(b, 4, second);
         for (unsigned c = 0; c < 4; ++c) {
            split->dest[c] = agx_temp(b->shader, AGX_SIZE_32);
            agx_unit_test(b, split->dest[c]);
         }
      }
      agx_instr *keep = mode == 1 ? agx_unit_test(b, sources[1]) : nullptr;
      agx_instr *split = agx_split(b, 4, result);
      for (unsigned c = 0; c < 4; ++c) {
         split->dest[c] = agx_temp(b->shader, AGX_SIZE_32);
         if (mode == 4)
            agx_export(b, split->dest[c], 8 + 2 * c);
         else
            agx_unit_test(b, split->dest[c]);
      }
      if (mode != 4) {
         agx_unit_test(b, occupied);
         agx_unit_test(b, occupied2);
      }
      ASSERT_TRUE(agx3_lower_scalar_cube_coordinates(b->shader));
      agx_dce(b->shader, true);
      agx_validate(b->shader, "scalar cube allocation test");
      if (mode == 5) {
         agx_reindex_ssa(b->shader);
         b->shader->scratch_size_B = 16;
      }
      agx_ra(b->shader);
      ASSERT_FALSE(b->shader->ra_failed);
      unsigned samples = 0;
      agx_foreach_instr_global(b->shader, I) {
         if (I->op != AGX_OPCODE_TEXTURE_SAMPLE)
            continue;
         ++samples;
         EXPECT_TRUE(I->agx3_cube_scalar_coords);
         EXPECT_EQ(I->dest[0].value % 8, 0u);
         for (unsigned src : {0u, 6u, 7u}) {
            EXPECT_EQ(I->src[src].type, AGX_INDEX_REGISTER);
            EXPECT_LT(I->src[src].value, 128u);
         }
         if (keep) {
            EXPECT_EQ(I->src[6].value, keep->src[0].value);
            EXPECT_TRUE(I->dest[0].value + 8 <= keep->src[0].value ||
                        I->dest[0].value >= keep->src[0].value + 2);
         }
         if (mode == 2) {
            EXPECT_EQ(I->src[6].value, I->src[7].value);
         }
         if (mode == 0 || mode == 4) {
            EXPECT_EQ(I->dest[0].value, 8u);
         }
      }
      EXPECT_EQ(samples, mode == 3 ? 2u : 1u);
      ralloc_free(mem);
   }
}

TEST(NativeForms, ScalarCubeSpillRebuildsCurrentAffinity)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   key.has_scratch = true;
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   b->shader->nir = rzalloc(mem, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;
   agx_index input = agx_preload(b, agx_register(16, AGX_SIZE_32));
   std::vector<agx_index> live;
   for (unsigned i = 0; i < 100; ++i)
      live.push_back(agx_fmul(b, input, agx_uniform(2 * i, AGX_SIZE_32)));
   agx_index vector = agx_vec_temp(b->shader, AGX_SIZE_32, 3);
   agx_instr *collect = agx_collect_to(b, vector, 3);
   for (unsigned c = 0; c < 3; ++c)
      collect->src[c] = live[c];
   agx_index result = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
   agx_texture_sample_to(
      b, result, vector, agx_zero(), agx_zero(), agx_uniform(240, AGX_SIZE_64),
      agx_uniform(244, AGX_SIZE_64), agx_null(), AGX_DIM_CUBE,
      AGX_LOD_MODE_AUTO_LOD, 15, false, false, false, AGX_GATHER_NONE, false);
   for (agx_index value : live)
      agx_unit_test(b, value);
   agx_instr *split = agx_split(b, 4, result);
   for (unsigned c = 0; c < 4; ++c) {
      split->dest[c] = agx_temp(b->shader, AGX_SIZE_32);
      agx_unit_test(b, split->dest[c]);
   }
   ASSERT_TRUE(agx3_lower_scalar_cube_coordinates(b->shader));
   agx_dce(b->shader, true);
   agx_validate(b->shader, "scalar cube before spilling");
   agx_ra(b->shader);
   ASSERT_FALSE(b->shader->ra_failed);
   EXPECT_GT(b->shader->scratch_size_B, 0u);
   bool sample = false;
   agx_foreach_instr_global(b->shader, I) {
      if (I->op != AGX_OPCODE_TEXTURE_SAMPLE)
         continue;
      sample = true;
      EXPECT_TRUE(I->agx3_cube_scalar_coords);
      for (unsigned src : {0u, 6u, 7u}) {
         EXPECT_EQ(I->src[src].type, AGX_INDEX_REGISTER);
         EXPECT_FALSE(I->src[src].memory);
         EXPECT_LT(I->src[src].value, 128u);
      }
   }
   EXPECT_TRUE(sample);
   ralloc_free(mem);
}

TEST(NativeForms, CubeScratchUsesDeadRegionsWithoutAsyncDestinations)
{
   for (unsigned mode = 0; mode < 5; ++mode) {
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      b->shader->nir = rzalloc(mem, nir_shader);
      b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;
      b->shader->max_reg = 103;
      if (mode == 3)
         b->shader->any_cf = true;
      if (mode == 1) {
         agx_index input = agx_register(8, AGX_SIZE_32);
         input.channels_m1 = 1;
         agx_index pending = agx_register(36, AGX_SIZE_32);
         pending.channels_m1 = 3;
         agx_texture_sample_to(b, pending, input, agx_zero(), agx_zero(),
                               agx_uniform(24, AGX_SIZE_64),
                               agx_uniform(28, AGX_SIZE_64), agx_null(),
                               AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 15, false,
                               false, false, AGX_GATHER_NONE, false);
      }
      agx_index coords = agx_register(4, AGX_SIZE_32);
      coords.channels_m1 = 2;
      agx_index result = agx_register(24, AGX_SIZE_32);
      result.channels_m1 = 3;
      agx_texture_sample_to(b, result, coords, agx_zero(), agx_zero(),
                            agx_uniform(24, AGX_SIZE_64),
                            agx_uniform(28, AGX_SIZE_64), agx_null(),
                            AGX_DIM_CUBE, AGX_LOD_MODE_AUTO_LOD, 15, false,
                            false, false, AGX_GATHER_NONE, false);
      if (mode == 4)
         agx_fmul_to(b, agx_register(36, AGX_SIZE_32),
                     agx_register(12, AGX_SIZE_32),
                     agx_uniform(8, AGX_SIZE_32));
      for (unsigned reg = 16; reg < 104; reg += 2) {
         bool hole =
            reg == 36 || (reg >= 64 && reg < 68) || (reg >= 70 && reg < 74);
         if (mode == 2 && (reg == 64 || reg == 70))
            hole = false;
         if (!hole)
            agx_fmul_to(b, agx_register(0, AGX_SIZE_32),
                        agx_register(reg, AGX_SIZE_32),
                        agx_uniform(8, AGX_SIZE_32));
      }
      agx_stop(b);
      util_dynarray binary;
      util_dynarray_init(&binary, mem);
      ASSERT_TRUE(agx3_pack_binary(b->shader, &binary)) << mode;
      bool reused = mode == 0 || mode == 1 || mode == 4;
      EXPECT_EQ(b->shader->max_reg, reused ? 103u : 111u);
      const unsigned expected[] = {
         reused ? 64u : 104u,
         reused ? ((mode == 1 || mode == 4) ? 70u : 36u) : 108u,
         reused ? ((mode == 1 || mode == 4) ? 72u : 70u) : 110u,
      };
      const uint8_t *bytes = (const uint8_t *)binary.data;
      for (unsigned axis = 0; axis < 3; ++axis) {
         uint8_t cem[AGX3_TEX_MAX_LENGTH], length = 0;
         ASSERT_TRUE(agx3_tex_cube_cem_calc_to(AGX_ISA_ENCODING_AGX3B, 4, 6, 8,
                                               expected[axis], axis, false, cem,
                                               &length));
         unsigned matches = 0;
         for (unsigned i = 0; i + length <= binary.size; ++i)
            matches += memcmp(bytes + i, cem, length) == 0;
         EXPECT_EQ(matches, 1u) << mode << ":" << axis;
      }
      util_dynarray_fini(&binary);
      ralloc_free(mem);
   }
}

TEST(NativeForms, ImmediateCubeGathersReserveCemScratch)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   agx_builder *b = agx_test_builder(mem);
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_COMPUTE;
   b->shader->max_reg = 31;

   /* This is the hardware reproducer's allocated shape. The red gather
    * remains live in r0 while the remaining gathers compute their cube
    * coordinates. The old immediate-descriptor path used r0 as CEM scratch
    * and overwrote red with the next face calculation. */
   agx_index coords = agx_register(16, AGX_SIZE_32);
   coords.channels_m1 = 3;
   const unsigned destinations[] = {0, 8, 24, 16};
   for (unsigned component = 0; component < 4; ++component) {
      agx_index dst = agx_register(destinations[component], AGX_SIZE_32);
      dst.channels_m1 = 3;
      agx_texture_sample_to(b, dst, coords, agx_immediate(0),
                           agx_immediate(0), agx_immediate(0),
                           agx_immediate(1), agx_null(), AGX_DIM_CUBE_ARRAY,
                           AGX_LOD_MODE_AUTO_LOD, 15, false, false, false,
                           (enum agx_gather)(AGX_GATHER_R + 2 * component), false);
   }
   agx_stop(b);

   util_dynarray binary;
   util_dynarray_init(&binary, mem);
   ASSERT_TRUE(agx3_pack_binary(b->shader, &binary));
   EXPECT_GE(b->shader->max_reg, 39u);

   /* Check the actual emitted CEM destination, not just the reservation.
    * All four axis-zero operations must write above the live result bank. */
   uint8_t cem[AGX3_TEX_MAX_LENGTH], len = 0;
   ASSERT_TRUE(agx3_tex_cube_cem_calc_to(AGX_ISA_ENCODING_AGX3B, 16, 18, 20,
                                         32, 0, false, cem, &len));
   const uint8_t *bytes = (const uint8_t *)binary.data;
   unsigned count = 0;
   for (unsigned i = 0; i + len <= binary.size; ++i)
      count += memcmp(bytes + i, cem, len) == 0;
   EXPECT_EQ(count, 4u);
   util_dynarray_fini(&binary);
   ralloc_free(mem);
}


TEST(NativeForms, LateCoverageStaysAfterImageSideEffects)
{
   for (bool depth : {false, true}) {
      void *mem = ralloc_context(NULL);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      agx_builder *b = agx_test_builder(mem);
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_index data = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
      agx_device_load_to(b, data, agx_uniform(8, AGX_SIZE_64), agx_immediate(0),
                         AGX_FORMAT_I32, 15, 0, false);
      agx_index coord = agx_vec_temp(b->shader, AGX_SIZE_32, 2);
      agx_device_load_to(b, coord, agx_uniform(8, AGX_SIZE_64), agx_immediate(4),
                         AGX_FORMAT_I32, 3, 0, false);
      agx_instr *store = agx_image_write(
         b, data, coord, agx_zero(), agx_uniform(12, AGX_SIZE_64),
         agx_mov_imm(b, 32, 1), AGX_DIM_2D, true);
      agx_index mask = agx_mov_imm(b, 16, 1);
      agx_index coverage[8];
      for (unsigned i = 0; i < 8; ++i)
         coverage[i] = agx_get_coverage_1x_agx3(b, 16);
      agx_index word = coverage[0];
      for (unsigned i = 1; i < 8; ++i)
         word = agx_iadd(b, word, coverage[i], 0);
      word = agx_iadd(b, mask, word, 4);
      agx_index wide = agx_temp(b->shader, AGX_SIZE_32);
      agx_mov_to(b, wide, word);
      agx_instr *tests = depth ? agx_zs_emit_agx3(b, wide, 1)
                              : agx_sample_mask_agx3(b, mask, wide, 2);
      agx_export(b, data, 8);
      agx_pressure_schedule(b->shader);
      bool saw_store = false;
      agx_foreach_instr_global(b->shader, I) {
         saw_store |= I == store;
         if (I == tests) {
            EXPECT_TRUE(saw_store) << "Native late tests preceded imageStore";
         }
      }
      ralloc_free(mem);
   }
}

TEST(NativeForms, VertexOutputsAllowTrailingMemoryFence)
{
   for (unsigned count : {5u, 6u, 11u}) {
      SCOPED_TRACE(count);
      void *mem = ralloc_context(NULL);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      agx_builder *b = agx_test_builder(mem);
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_VERTEX;
      for (unsigned i = 0; i < count; ++i)
         agx_st_vary(b, agx_immediate(i), agx_register(2 * i, AGX_SIZE_32));
      agx_scoped_barrier_agx3(b, AGX3_BARRIER_SCOPE_TEXTURE_FENCE);
      agx_scoped_barrier_agx3(b, AGX3_BARRIER_SCOPE_TEXTURE_FENCE_WAIT);
      agx_stop(b);
      util_dynarray bytes;
      util_dynarray_init(&bytes, mem);
      EXPECT_TRUE(agx3_pack_binary(b->shader, &bytes));
      EXPECT_GT(bytes.size, 0u);
      ralloc_free(mem);
   }
}

TEST(NativeForms, CubeArrayShadowZeroLodUsesNativeDescriptors)
{
   for (mesa_shader_stage stage : {MESA_SHADER_VERTEX, MESA_SHADER_TESS_CTRL,
                                MESA_SHADER_TESS_EVAL, MESA_SHADER_GEOMETRY,
                                MESA_SHADER_FRAGMENT, MESA_SHADER_COMPUTE}) {
      for (float lod : {0.0f, 1.0f}) {
         SCOPED_TRACE(::testing::Message() << "stage=" << stage << " lod=" << lod);
         nir_builder b = nir_builder_init_simple_shader(
            stage, &agx_nir_options, "cube array shadow native descriptors");
         nir_def *texture = nir_bindless_image_agx(&b, nir_imm_int(&b, 0));
         nir_def *sampler = nir_bindless_sampler_agx(
            &b, nir_imm_int(&b, 32), nir_imm_intN_t(&b, 5, 16));
         nir_tex_instr *tex = nir_tex_instr_create(b.shader, 5);
         tex->op = nir_texop_txl;
         tex->sampler_dim = GLSL_SAMPLER_DIM_CUBE;
         tex->is_array = true;
         tex->is_shadow = true;
         tex->is_new_style_shadow = true;
         tex->coord_components = 4;
         tex->dest_type = nir_type_float32;
         tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_backend1,
                                         nir_imm_vec4(&b, 1, 0, 0, 0));
         tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_comparator,
                                         nir_imm_float(&b, 0.5));
         tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_lod, nir_imm_float(&b, lod));
         tex->src[3] = nir_tex_src_for_ssa(nir_tex_src_texture_handle, texture);
         tex->src[4] = nir_tex_src_for_ssa(nir_tex_src_sampler_handle, sampler);
         nir_def_init(&tex->instr, &tex->def, 1, 32);
         nir_builder_instr_insert(&b, &tex->instr);
         nir_store_ssbo(&b, &tex->def, nir_imm_int(&b, 0), nir_imm_int(&b, 0));
         EXPECT_EQ(agx_nir_lower_g17p_indexed_samplers(b.shader), lod == 0.0f);
         ralloc_free(b.shader);
      }
   }
}

TEST(NativeForms, EndOfTilePreservesSamplesAndResolvesSameTile)
{
   /* Both outputs read offset zero. Their descriptors are distinct, and
    * only the second descriptor names a single-sample image. Keep all four
    * dimension combinations as controls; no host check establishes pixels. */
   for (unsigned resolves = 0; resolves < 4; ++resolves) {
      SCOPED_TRACE(resolves);
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "EOT store and resolve");
      nir_def *xy = nir_trim_vector(&b, nir_load_global_invocation_id(&b, 32), 2);
      for (unsigned pbe = 0; pbe < 2; ++pbe) {
         nir_intrinsic_instr *intr = nir_image_store_block_agx(
            &b, nir_imm_intN_t(&b, pbe, 16), nir_imm_int(&b, 0), xy);
         nir_intrinsic_set_format(intr, (enum pipe_format)AIL_ISA_FORMAT_U8NORM);
         nir_intrinsic_set_image_dim(intr, GLSL_SAMPLER_DIM_2D);
         nir_intrinsic_set_image_array(intr, false);
         nir_intrinsic_set_explicit_coord(intr, true);
      }
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      key.no_preamble = true;
      key.g17p_rt0_eot_multisample = true;
      key.g17p_mrt_eot_count = 2;
      key.g17p_mrt_eot_resolve_mask = resolves;
      key.g17p_mrt_eot_selectors[0] = 0x75;
      key.g17p_mrt_eot_selectors[1] = 0x75;
      agx_preprocess_nir(b.shader);
      agx_shader_part part = {};
      bool ok = agx_compile_shader_nir(b.shader, &key, &part);
      EXPECT_TRUE(ok);
      if (ok) {
         EXPECT_EQ(part.info.binary_size, 120u);
         EXPECT_EQ(part.info.main_size, 120u);
         EXPECT_TRUE(part.info.agx3_staging_known);
         EXPECT_EQ(part.info.agx3_staging_halfs, 16u);
         if (part.info.binary_size == 120) {
            const uint8_t *code = static_cast<const uint8_t *>(part.binary);
            /* Native single-target MS and SS dimension fields, independently
             * selected for each store of the existing dual-target program. */
            for (unsigned pbe = 0; pbe < 2; ++pbe) {
               bool ms = !(resolves & (1u << pbe));
               EXPECT_EQ(code[pbe ? 63 : 43],
                         (pbe ? 0x0c : 0x04) | (ms ? 0x02 : 0x00));
               EXPECT_EQ(code[92 + 12 * pbe + 8], ms ? 0x28 : 0xa8);
               EXPECT_EQ(code[92 + 12 * pbe + 9], ms ? 0x72 : 0x75);
            }
         }
      }
      free(part.binary);
      ralloc_free(b.shader);
   }
}

static agx_shader_info
compile_partial_varying(unsigned components, unsigned bits,
                        glsl_interp_mode mode)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "partial varying cache");
   nir_def *bary = nir_load_barycentric_pixel(&b, 32);
   nir_intrinsic_set_interp_mode(nir_def_as_intrinsic(bary), mode);
   nir_def *input = nir_load_interpolated_input(&b, components, bits, bary,
                                                nir_imm_int(&b, 0));
   nir_intrinsic_instr *load = nir_def_as_intrinsic(input);
   nir_intrinsic_set_component(load, components == 1 ? 3 : 0);
   nir_intrinsic_set_dest_type(
      load, bits == 16 ? nir_type_float16 : nir_type_float32);
   nir_intrinsic_set_io_semantics(
      load, nir_io_semantics{.location = VARYING_SLOT_VAR0});
   nir_def *value = nir_channel(&b, input, components - 1);
   if (bits == 16)
      value = nir_f2f32(&b, value);
   agx_test_store_local_pixel_agx(&b, nir_vec4(&b, value, value, value, value),
                                  nir_imm_intN_t(&b, 0xff, 16),
                                  nir_undef(&b, 2, 16), 0, 0xf,
                                  PIPE_FORMAT_R32_UINT);
   b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   key.no_preamble = true;
   key.fs.nr_samples = 1;
   agx_preprocess_nir(b.shader);
   agx_shader_part part = {};
   EXPECT_TRUE(agx_compile_shader_nir(b.shader, &key, &part));
   agx_shader_info info = part.info;
   free(part.binary);
   ralloc_free(b.shader);
   return info;
}

TEST(NativeForms, PartialVaryingOmitsUnusedChannels)
{
   for (unsigned bits : {16u, 32u}) {
      for (glsl_interp_mode mode :
           {INTERP_MODE_SMOOTH, INTERP_MODE_NOPERSPECTIVE}) {
         SCOPED_TRACE(bits);
         SCOPED_TRACE(mode);
         agx_shader_info scalar = compile_partial_varying(1, bits, mode);
         agx_shader_info vector = compile_partial_varying(4, bits, mode);
         ASSERT_GT(scalar.main_size, 0u);
         EXPECT_LE(vector.main_size, scalar.main_size + 8);
         EXPECT_LE(vector.nr_gprs, 2 * scalar.nr_gprs);
      }
   }
}

TEST(NativeForms, TextureFmaMatchesMetalStaging)
{
   struct vector {
      uint16_t dest;
      uint8_t slot;
      uint16_t src[3];
      bool live[3], discard[3];
      uint16_t wait;
      uint8_t bytes[16];
      uint8_t length;
   };
   const vector vectors[] = {
      {0,
       1,
       {46, 6, 4},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0x09, 0x87, 0x26, 0x2f, 0x01, 0x84, 0x42, 0x02},
       8},
      {4,
       2,
       {48, 6, 4},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0x29, 0x87, 0x26, 0x31, 0x02, 0x84, 0x42, 0x02, 0x00, 0x04},
       10},
      {8,
       2,
       {50, 6, 4},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0x49, 0x87, 0x26, 0x33, 0x02, 0x84, 0x42, 0x02, 0x00, 0x04},
       10},
      {12,
       2,
       {52, 6, 4},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0x69, 0x87, 0x26, 0x35, 0x02, 0x84, 0x42, 0x02, 0x00, 0x04},
       10},
      {2,
       1,
       {46, 2, 0},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0x19, 0x83, 0x26, 0x2f, 0x01, 0x80, 0x42, 0x02},
       8},
      {6,
       2,
       {48, 2, 0},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0x39, 0x83, 0x26, 0x31, 0x02, 0x80, 0x42, 0x02, 0x00, 0x04},
       10},
      {10,
       2,
       {50, 2, 0},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0x59, 0x83, 0x26, 0x33, 0x02, 0x80, 0x42, 0x02, 0x00, 0x04},
       10},
      {14,
       2,
       {52, 2, 0},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0x79, 0x83, 0x26, 0x35, 0x02, 0x80, 0x42, 0x02, 0x00, 0x04},
       10},
      {16,
       2,
       {54, 6, 4},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0x89, 0x87, 0x26, 0x37, 0x02, 0x84, 0x42, 0x02, 0x00, 0x04},
       10},
      {20,
       2,
       {56, 6, 4},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0xa9, 0x87, 0x26, 0x39, 0x02, 0x84, 0x42, 0x02, 0x00, 0x04},
       10},
      {28,
       4,
       {58, 6, 4},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0xe9, 0x87, 0x26, 0x3b, 0x02, 0x84, 0x42, 0x02, 0x00, 0x0c},
       10},
      {36,
       5,
       {62, 6, 4},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0x29, 0x87, 0x66, 0x3f, 0x02, 0x84, 0x42, 0x02, 0x00, 0x10},
       10},
      {40,
       5,
       {64, 6, 4},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0x49, 0x87, 0x66, 0x41, 0x02, 0x84, 0x42, 0x02, 0x00, 0x10},
       10},
      {42,
       5,
       {64, 2, 0},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0x59, 0x83, 0x66, 0x41, 0x02, 0x80, 0x42, 0x02, 0x00, 0x10},
       10},
      {38,
       5,
       {62, 2, 0},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0x39, 0x83, 0x66, 0x3f, 0x02, 0x80, 0x42, 0x02, 0x00, 0x10},
       10},
      {44,
       6,
       {60, 6, 4},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0x69, 0x87, 0x66, 0x3d, 0x02, 0x84, 0x42, 0x02, 0x00, 0x14},
       10},
      {46,
       6,
       {60, 2, 0},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0x79, 0x83, 0x66, 0x3d, 0x02, 0x80, 0x42, 0x02, 0x00, 0x14},
       10},
      {30,
       4,
       {58, 2, 0},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0xf9, 0x83, 0x26, 0x3b, 0x02, 0x80, 0x42, 0x02, 0x00, 0x0c},
       10},
      {22,
       2,
       {56, 2, 0},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0xb9, 0x83, 0x26, 0x39, 0x02, 0x80, 0x42, 0x02, 0x00, 0x04},
       10},
      {18,
       2,
       {54, 2, 0},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0x99, 0x83, 0x26, 0x37, 0x02, 0x80, 0x42, 0x02, 0x00, 0x04},
       10},
      {32,
       6,
       {66, 6, 4},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0x09, 0x87, 0x66, 0x43, 0x02, 0x84, 0x42, 0x02, 0x00, 0x14},
       10},
      {56,
       4,
       {68, 6, 4},
       {0, 0, 0},
       {0, 1, 1},
       0,
       {0xc9, 0x07, 0x6e, 0x45, 0x82, 0x04, 0x42, 0x02, 0x00, 0x0c},
       10},
      {58,
       4,
       {68, 2, 0},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0xd9, 0x83, 0x66, 0x45, 0x02, 0x80, 0x42, 0x02, 0x00, 0x0c},
       10},
      {34,
       6,
       {66, 2, 0},
       {0, 0, 0},
       {0, 1, 1},
       0,
       {0x19, 0x03, 0x6e, 0x43, 0x82, 0x00, 0x42, 0x02, 0x00, 0x14},
       10},
   };
   for (const auto &v : vectors) {
      uint8_t bytes[16], length = 0;
      ASSERT_TRUE(agx3_stage_vertex_alu(AGX_ISA_ENCODING_AGX3B, true, v.dest,
                                        v.slot, v.src, true, v.live, v.discard,
                                        v.wait, bytes, &length));
      ASSERT_EQ(length, v.length);
      EXPECT_EQ(memcmp(bytes, v.bytes, length), 0);
   }
}

TEST(NativeForms, TextureFmaStagingAdmission)
{
   for (unsigned invalid = 0; invalid < 15; ++invalid) {
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_index x = agx_preload(b, agx_register(0, AGX_SIZE_32));
      agx_index y = agx_preload(b, agx_register(2, AGX_SIZE_32));
      agx_index z = agx_preload(b, agx_register(4, AGX_SIZE_32));
      agx_index multiplier = agx_uniform(20, AGX_SIZE_32);
      if (invalid == 12)
         multiplier = agx_preload(b, agx_register(6, AGX_SIZE_32));
      if (invalid == 13)
         multiplier = agx_immediate_f(0.5f);
      if (invalid == 14)
         multiplier = agx_immediate(256);
      if (invalid == 1)
         multiplier.size = AGX_SIZE_16;
      if (invalid == 2)
         y = agx_uniform(22, AGX_SIZE_32);
      if (invalid == 3)
         z = agx_uniform(22, AGX_SIZE_32);
      if (invalid == 4)
         x.neg = true;
      if (invalid == 10)
         multiplier.value = 256;
      if (invalid == 11)
         multiplier.value = 21;
      agx_index u = agx_temp(b->shader, AGX_SIZE_32);
      agx_ffma_to(b, u, x, multiplier, z);
      agx_instr *first =
         agx_last_instr(list_first_entry(&b->shader->blocks, agx_block, link));
      if (invalid == 5)
         first->saturate = true;
      agx_index v = agx_temp(b->shader, AGX_SIZE_32);
      agx_ffma_to(b, v, multiplier, y, z);
      agx_instr *second =
         agx_last_instr(list_first_entry(&b->shader->blocks, agx_block, link));
      if (invalid == 6)
         agx_unit_test(b, u);
      if (invalid == 7) {
         agx_instr *store = agx_alloc_instr(b, AGX_OPCODE_UNIFORM_STORE, 0, 2);
         store->src[0] = z;
         store->src[1] = agx_immediate(20);
         store->mask = 1;
         agx_builder_insert(&b->cursor, store);
      }
      agx_index coord = agx_vec_temp(b->shader, AGX_SIZE_32, 2);
      agx_instr *collect = agx_collect_to(b, coord, 2);
      collect->src[0] = u;
      collect->src[1] = v;
      agx_index out = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
      agx_instr *sample = agx_texture_sample_to(
         b, out, coord, agx_zero(), agx_zero(), agx_uniform(24, AGX_SIZE_64),
         agx_uniform(28, AGX_SIZE_64), agx_null(), AGX_DIM_2D,
         AGX_LOD_MODE_AUTO_LOD, 0xf, false, false, false, AGX_GATHER_NONE,
         false);
      if (invalid == 8)
         sample->dim = AGX_DIM_3D;
      if (invalid == 9)
         sample->lod_mode = AGX_LOD_MODE_LOD_MIN;
      agx_unit_test(b, out);
      agx3_lower_scalar_texture_publications(b->shader);
      agx_dce(b->shader, true);
      agx_instr *publication = nullptr;
      agx_foreach_instr_global(b->shader, I) {
         if (I->op == AGX_OPCODE_PUBLISH_TEXTURE_SCALAR_AGX3)
            publication = I;
      }
      const bool progress = agx3_fuse_fma_texture_publications(b->shader);
      EXPECT_EQ(progress, (invalid == 0 || invalid == 9 || invalid == 12 ||
                           invalid == 13))
         << invalid;
      EXPECT_EQ(first->agx3_tex_coord_stage, (invalid == 0 || invalid == 9 ||
                                              invalid == 12 || invalid == 13))
         << invalid;
      EXPECT_EQ(second->agx3_tex_coord_stage, (invalid == 0 || invalid == 9 ||
                                               invalid == 12 || invalid == 13))
         << invalid;
      const bool supported =
         invalid == 0 || invalid == 9 || invalid == 12 || invalid == 13;
      EXPECT_FALSE(sample->agx3_tex_coords_staged);
      if (publication) {
         EXPECT_EQ(publication->agx3_tex_coords_staged, supported);
      }
      if (supported) {
         ASSERT_NE(publication, nullptr);
         EXPECT_TRUE(first->agx3_tex_coord_start);
         EXPECT_EQ(first->link.next, &second->link);
         EXPECT_EQ(second->link.next, &publication->link);
         EXPECT_EQ(sample->src[0].type, AGX_INDEX_RESOURCE);
      }
      ralloc_free(mem);
   }
}

TEST(NativeForms, TextureFmaImmediateMatchesMetal)
{
   const uint16_t sources[2][3] = {{32, 6, 4}, {32, 2, 0}};
   const uint8_t expected[2][10] = {
      {0xc9, 0x87, 0x26, 0xa1, 0x02, 0x84, 0x42, 0x02, 0x00, 0x04},
      {0xd9, 0x83, 0x26, 0xa1, 0x02, 0x80, 0x42, 0x02, 0x00, 0x04},
   };
   const bool live[3] = {false, true, true};
   const bool discard[3] = {};
   for (unsigned i = 0; i < 2; ++i) {
      uint8_t bytes[16], length = 0;
      ASSERT_TRUE(agx3_stage_fma(AGX_ISA_ENCODING_AGX3B, 24 + 2 * i, 2,
                                 sources[i], AGX3_STAGE_FMA_IMMEDIATE, live,
                                 discard, 0, bytes, &length));
      ASSERT_EQ(length, sizeof(expected[i]));
      EXPECT_EQ(memcmp(bytes, expected[i], length), 0);
   }
   uint8_t bytes[16], length = 0;
   const uint16_t invalid_sources[3] = {256, 2, 0};
   EXPECT_FALSE(agx3_stage_fma(AGX_ISA_ENCODING_AGX3B, 0, 1, invalid_sources,
                               AGX3_STAGE_FMA_IMMEDIATE, live, discard, 0,
                               bytes, &length));
   const bool invalid_live[3] = {true, true, true};
   EXPECT_FALSE(agx3_stage_fma(AGX_ISA_ENCODING_AGX3B, 0, 1, sources[0],
                               AGX3_STAGE_FMA_IMMEDIATE, invalid_live, discard,
                               0, bytes, &length));
}

TEST(NativeForms, TextureFmaStagingPreservesDemand)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_index input = agx_preload(b, agx_register(0, AGX_SIZE_32));
   agx_index operands[6];
   for (unsigned i = 0; i < 6; ++i)
      operands[i] = agx_iadd(b, input, agx_immediate(i + 1), 0);
   agx_index x = agx_ffma(b, operands[0], operands[1], operands[2]);
   agx_index live[10];
   for (unsigned i = 0; i < 10; ++i)
      live[i] = agx_iadd(b, input, agx_immediate(i + 10), 0);
   agx_index y = agx_ffma(b, operands[3], operands[4], operands[5]);
   agx_index coord = agx_vec_temp(b->shader, AGX_SIZE_32, 2);
   agx_instr *collect = agx_collect_to(b, coord, 2);
   collect->src[0] = x;
   collect->src[1] = y;
   agx_index out = agx_temp(b->shader, AGX_SIZE_32);
   agx_texture_sample_to(
      b, out, coord, agx_zero(), agx_zero(), agx_uniform(24, AGX_SIZE_64),
      agx_uniform(28, AGX_SIZE_64), agx_null(), AGX_DIM_2D,
      AGX_LOD_MODE_AUTO_LOD, 1, false, false, false, AGX_GATHER_NONE, false);
   for (agx_index value : live)
      agx_unit_test(b, value);
   agx_unit_test(b, out);
   ASSERT_TRUE(agx3_lower_scalar_texture_publications(b->shader));
   agx_dce(b->shader, true);
   std::vector<const agx_instr *> order;
   agx_foreach_instr_global(b->shader, I)
      order.push_back(I);
   agx_compute_liveness(b->shader);
   unsigned before = agx_calc_register_demand(b->shader, false);
   EXPECT_FALSE(agx3_fuse_fma_texture_publications(b->shader));
   agx_compute_liveness(b->shader);
   EXPECT_EQ(agx_calc_register_demand(b->shader, false), before);
   unsigned index = 0;
   agx_foreach_instr_global(b->shader, I) {
      EXPECT_FALSE(I->agx3_tex_coords_staged);
      if (I->op == AGX_OPCODE_FFMA) {
         EXPECT_EQ(I->dest[0].type, AGX_INDEX_NORMAL);
      }
      if (agx_is_texture_publication(I)) {
         EXPECT_EQ(I->src[0].type, AGX_INDEX_NORMAL);
         EXPECT_EQ(I->src[1].type, AGX_INDEX_NORMAL);
      }
      ASSERT_LT(index, order.size());
      EXPECT_EQ(I, order[index++]);
   }
   EXPECT_EQ(index, order.size());
   ralloc_free(mem);
}

TEST(NativeForms, MultipleTextureFmaPublications)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   b->shader->nir = rzalloc(mem, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;
   agx_index input = agx_preload(b, agx_register(0, AGX_SIZE_32));
   agx_index output[3];
   for (unsigned i = 0; i < 3; ++i) {
      agx_index coord = agx_vec_temp(b->shader, AGX_SIZE_32, 2);
      agx_index x = agx_ffma(b, input, agx_uniform(2 * i, AGX_SIZE_32), input);
      agx_index y = agx_ffma(b, agx_uniform(6 + 2 * i, AGX_SIZE_32), input, input);
      agx_instr *collect = agx_collect_to(b, coord, 2);
      collect->src[0] = x;
      collect->src[1] = y;
      output[i] = agx_temp(b->shader, AGX_SIZE_32);
      agx_index sampled = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
      agx_texture_sample_to(
         b, sampled, coord, agx_zero(), agx_zero(),
         agx_uniform(24, AGX_SIZE_64), agx_uniform(28, AGX_SIZE_64), agx_null(),
         AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 15, false, false, false,
         AGX_GATHER_NONE, false);
      agx_instr *split = agx_split(b, 4, sampled);
      split->dest[0] = output[i];
      for (unsigned c = 1; c < 4; ++c)
         split->dest[c] = agx_null();
   }
   for (unsigned i = 0; i < 3; ++i)
      agx_export(b, output[i], 8 + 2 * i);
   agx_stop(b);
   ASSERT_TRUE(agx3_lower_scalar_texture_publications(b->shader));
   agx_dce(b->shader, true);
   ASSERT_TRUE(agx3_fuse_fma_texture_publications(b->shader));
   agx_dce(b->shader, true);
   unsigned stages = 0, publications = 0;
   agx_foreach_instr_global(b->shader, I) {
      stages += I->agx3_tex_coord_stage;
      if (I->agx3_tex_coord_stage) {
         EXPECT_TRUE(agx_is_null(I->dest[0]));
      }
      if (agx_is_texture_publication(I)) {
         publications++;
         EXPECT_TRUE(I->agx3_tex_coords_staged);
         EXPECT_TRUE(agx_is_null(I->src[0]));
         EXPECT_TRUE(agx_is_null(I->src[1]));
      }
   }
   EXPECT_EQ(stages, 6u);
   EXPECT_EQ(publications, 3u);
   EXPECT_FALSE(agx3_fuse_fma_texture_publications(b->shader));
   agx_ra(b->shader);
   EXPECT_FALSE(b->shader->ra_failed);
   agx_lower_pseudo(b->shader);
   agx_opt_register_cache(b->shader);
   util_dynarray binary;
   util_dynarray_init(&binary, mem);
   EXPECT_TRUE(agx3_pack_binary(b->shader, &binary));
   EXPECT_GT(binary.size, 0u);
   util_dynarray_fini(&binary);
   ralloc_free(mem);
}

TEST(NativeForms, TextureFmaPublicationPrecedesEarlierSamples)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_index input = agx_preload(b, agx_register(0, AGX_SIZE_32));
   agx_index x = agx_ffma(b, input, agx_uniform(0, AGX_SIZE_32), input);
   agx_index y = agx_ffma(b, input, agx_uniform(2, AGX_SIZE_32), input);
   agx_instr *samples[2];
   for (unsigned i = 0; i < 2; ++i) {
      agx_index coordinates = agx_vec_temp(b->shader, AGX_SIZE_32, 2);
      agx_instr *collect = agx_collect_to(b, coordinates, 2);
      collect->src[0] = i ? x : input;
      collect->src[1] = i ? y : input;
      agx_index result = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
      samples[i] = agx_texture_sample_to(
         b, result, coordinates, agx_zero(), agx_zero(),
         agx_uniform(24, AGX_SIZE_64), agx_uniform(28, AGX_SIZE_64), agx_null(),
         AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 15, false, false, false,
         AGX_GATHER_NONE, false);
      agx_unit_test(b, result);
   }
   ASSERT_TRUE(agx3_lower_scalar_texture_publications(b->shader));
   agx_dce(b->shader, true);
   ASSERT_TRUE(agx3_fuse_fma_texture_publications(b->shader));
   bool published = false;
   agx_foreach_instr_global(b->shader, I) {
      if (I->op == AGX_OPCODE_PUBLISH_TEXTURE_SCALAR_AGX3 &&
          I->imm == agx_texture_resource(samples[1])) {
         EXPECT_TRUE(I->agx3_tex_coords_staged);
         published = true;
      }
      if (I == samples[0]) {
         EXPECT_TRUE(published);
      }
   }
   EXPECT_TRUE(published);
   ralloc_free(mem);
}

TEST(NativeForms, TextureFmaPublicationPressure)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   b->shader->nir = rzalloc(mem, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;
   agx_index input = agx_preload(b, agx_register(0, AGX_SIZE_32));
   agx_index coordinates[65][2];
   for (unsigned i = 0; i < 65; ++i) {
      coordinates[i][0] = agx_ffma(b, input, agx_uniform(2 * i, AGX_SIZE_32), input);
      coordinates[i][1] = agx_ffma(b, input, agx_uniform(2 * i + 2, AGX_SIZE_32), input);
   }
   agx_instr *first_sample = nullptr;
   agx_index sum = agx_null();
   for (unsigned i = 0; i < 65; ++i) {
      agx_index coord = agx_vec_temp(b->shader, AGX_SIZE_32, 2);
      agx_instr *collect = agx_collect_to(b, coord, 2);
      collect->src[0] = coordinates[i][0];
      collect->src[1] = coordinates[i][1];
      agx_index sampled = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
      agx_instr *sample = agx_texture_sample_to(
         b, sampled, coord, agx_zero(), agx_zero(),
         agx_uniform(200, AGX_SIZE_64), agx_uniform(204, AGX_SIZE_64), agx_null(),
         AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 15, false, false, false,
         AGX_GATHER_NONE, false);
      if (!i)
         first_sample = sample;
      agx_index x = agx_temp(b->shader, AGX_SIZE_32);
      agx_instr *split = agx_split(b, 4, sampled);
      split->dest[0] = x;
      for (unsigned c = 1; c < 4; ++c)
         split->dest[c] = agx_null();
      sum = i ? agx_fadd(b, sum, x) : x;
   }
   agx_export(b, sum, 8);
   agx_stop(b);
   ASSERT_TRUE(agx3_lower_scalar_texture_publications(b->shader));
   agx_dce(b->shader, true);
   ASSERT_TRUE(agx3_fuse_fma_texture_publications(b->shader));
   unsigned early = 0, fused = 0;
   bool before_sample = true;
   agx_foreach_instr_global(b->shader, I) {
      if (I == first_sample)
         before_sample = false;
      if (agx_is_texture_publication(I) && I->agx3_tex_coords_staged) {
         fused++;
         early += before_sample;
      }
   }
   EXPECT_EQ(fused, 65u);
   EXPECT_EQ(early, 64u);
   agx_ra(b->shader);
   ASSERT_FALSE(b->shader->ra_failed);
   agx_lower_pseudo(b->shader);
   agx_opt_register_cache(b->shader);
   util_dynarray binary;
   util_dynarray_init(&binary, mem);
   EXPECT_TRUE(agx3_pack_binary(b->shader, &binary));
   EXPECT_GT(binary.size, 0u);
   util_dynarray_fini(&binary);
   ralloc_free(mem);
}

TEST(NativeForms, TextureFmaStagingRespectsCoverageOrdering)
{
   for (unsigned position = 0; position < 3; ++position) {
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_index input = agx_preload(b, agx_register(0, AGX_SIZE_32));
      agx_index outputs[2];
      for (unsigned i = 0; i < 3; ++i) {
         if (i == position) {
            agx_instr *coverage =
               agx_alloc_instr(b, AGX_OPCODE_SAMPLE_MASK_AGX3, 0, 2);
            coverage->src[0] = input;
            coverage->src[1] = agx_zero();
            agx_builder_insert(&b->cursor, coverage);
         }
         if (i < 2)
            outputs[i] = agx_ffma(b, input, agx_uniform(2 * i, AGX_SIZE_32), input);
      }
      agx_instr *publication = agx_publish_texture_scalar_agx3(
         b, outputs[0], outputs[1], agx_null(), 1);
      std::vector<const agx_instr *> order;
      agx_foreach_instr_global(b->shader, I)
         order.push_back(I);
      EXPECT_EQ(agx3_fuse_fma_texture_publications(b->shader), position == 0);
      EXPECT_EQ(publication->agx3_tex_coords_staged, position == 0);
      if (position != 0) {
         unsigned index = 0;
         agx_foreach_instr_global(b->shader, I) {
            ASSERT_LT(index, order.size());
            EXPECT_EQ(I, order[index++]);
         }
         EXPECT_EQ(index, order.size());
      }
      ralloc_free(mem);
   }
}

TEST(NativeForms, TextureFmaPublicationsPreserveReusedCoordinates)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   b->shader->max_reg = 63;
   for (unsigned pair = 0; pair < 2; ++pair) {
      agx_index outputs[2];
      for (unsigned c = 0; c < 2; ++c) {
         outputs[c] = agx_register(16 + 4 * pair + 2 * c, AGX_SIZE_32);
         agx_instr *fma = agx_ffma_to(
            b, outputs[c], agx_uniform(12 + 4 * pair + 2 * c, AGX_SIZE_32),
            agx_register(2, AGX_SIZE_32), agx_register(4, AGX_SIZE_32));
         fma->agx3_tex_coord_stage = true;
         fma->agx3_tex_coord_start = c == 0;
         fma->agx3_tex_coord_component = c;
      }
      agx_instr *publication = agx_publish_texture_scalar_agx3(
         b, outputs[0], outputs[1], agx_null(), pair + 1);
      publication->agx3_tex_coords_staged = true;
   }
   const unsigned values[3] = {1, 2, 1};
   for (unsigned i = 0; i < 3; ++i) {
      agx_index resource = agx_get_vec_index(values[i], AGX_SIZE_32, 2);
      resource.type = AGX_INDEX_RESOURCE;
      agx_index result = agx_register(40 + 8 * i, AGX_SIZE_32);
      result.channels_m1 = 3;
      agx_texture_sample_to(
         b, result, resource, agx_zero(), agx_zero(),
         agx_uniform(24, AGX_SIZE_64), agx_uniform(28, AGX_SIZE_64), agx_null(),
         AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 15, false, false, false,
         AGX_GATHER_NONE, false);
   }
   agx_stop(b);
   util_dynarray binary;
   util_dynarray_init(&binary, mem);
   ASSERT_TRUE(agx3_pack_binary(b->shader, &binary));
   EXPECT_EQ(b->shader->agx3_staging_halfs, 8u);
   const uint8_t *bytes = (const uint8_t *)binary.data;
   const bool hints[3] = {};
   for (unsigned pair = 0; pair < 2; ++pair) {
      for (unsigned c = 0; c < 2; ++c) {
         const uint16_t src[3] = {uint16_t(12 + 4 * pair + 2 * c), 2, 4};
         unsigned found = 0;
         for (unsigned slot = 1; slot <= 6; ++slot) {
            for (unsigned wait = 0; wait < 64; ++wait) {
               uint8_t expected[16], length = 0;
               ASSERT_TRUE(agx3_stage_fma(
                  AGX_ISA_ENCODING_AGX3B, 4 * pair + 2 * c, slot, src,
                  AGX3_STAGE_FMA_UNIFORM, hints, hints, wait, expected, &length));
               for (unsigned offset = 0; offset + length <= binary.size; ++offset)
                  found += memcmp(bytes + offset, expected, length) == 0;
            }
         }
         EXPECT_EQ(found, 1u) << pair << ":" << c;
      }
   }
   for (unsigned i = 0; i < 3; ++i) {
      agx3_tex_sample_desc desc = {};
      desc.dest = 40 + 8 * i;
      desc.coord = 4 * (values[i] - 1);
      desc.width = AGX3_TEX_W32;
      desc.mask = 15;
      desc.lod = AGX3_TEX_LOD_AUTO;
      desc.texture_kind = desc.sampler_kind = AGX3_TEX_OPERAND_UNIFORM;
      desc.texture_value = 24;
      desc.sampler_value = 28;
      desc.sample_abi = AGX3_TEX_SAMPLE_ABI_DRIVER;
      desc.coord_kill = i != 0;
      desc.final = desc.sample_group_end = i == 2;
      desc.sample_group_nonterminal = i != 2;
      unsigned found = 0;
      for (unsigned slot = 1; slot <= 6; ++slot) {
         for (unsigned wait = 0; wait <= 6; ++wait) {
            desc.slot = slot;
            desc.wait_mask = wait ? 1u << (wait - 1) : 0;
            uint8_t expected[AGX3_TEX_MAX_LENGTH], length = 0;
            ASSERT_TRUE(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &desc,
                                              expected, &length));
            for (unsigned offset = 0; offset + length <= binary.size; ++offset)
               found += memcmp(bytes + offset, expected, length) == 0;
         }
      }
      EXPECT_EQ(found, 1u) << i;
   }
   util_dynarray_fini(&binary);
   ralloc_free(mem);
}

TEST(NativeForms, TextureFmaPublicationPacksWithoutCopy)
{
   for (bool loaded : {false, true}) {
      void *mem = ralloc_context(NULL);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      agx_builder *b = agx_test_builder(mem);
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      b->shader->max_reg = 31;
      if (loaded)
         agx_device_load_to(b, agx_register(2, AGX_SIZE_32),
                            agx_register(28, AGX_SIZE_64), agx_zero(),
                            AGX_FORMAT_I32, 1, 0, false);
      const agx_index outputs[2] = {
         agx_register(16, AGX_SIZE_32),
         agx_register(18, AGX_SIZE_32),
      };
      agx_instr *x = agx_ffma_to(b, outputs[0], agx_uniform(20, AGX_SIZE_32),
                                 agx_register(2, AGX_SIZE_32),
                                 agx_register(4, AGX_SIZE_32));
      agx_instr *y = agx_ffma_to(b, outputs[1], agx_register(6, AGX_SIZE_32),
                                 agx_uniform(20, AGX_SIZE_32),
                                 agx_register(8, AGX_SIZE_32));
      x->agx3_tex_coord_stage = x->agx3_tex_coord_start = true;
      y->agx3_tex_coord_stage = true;
      y->agx3_tex_coord_component = 1;
      agx_instr *publication = agx_publish_texture_scalar_agx3(
         b, outputs[0], outputs[1], agx_null(), 1);
      publication->agx3_tex_coords_staged = true;
      agx_index resource = agx_get_vec_index(1, AGX_SIZE_32, 2);
      resource.type = AGX_INDEX_RESOURCE;
      agx_index result = agx_register(20, AGX_SIZE_32);
      result.channels_m1 = 3;
      agx_texture_sample_to(b, result, resource, agx_zero(), agx_zero(),
                            agx_uniform(24, AGX_SIZE_64),
                            agx_uniform(28, AGX_SIZE_64), agx_null(),
                            AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 15, false, false,
                            false, AGX_GATHER_NONE, false);
      agx_stop(b);
      util_dynarray binary;
      util_dynarray_init(&binary, mem);
      ASSERT_TRUE(agx3_pack_binary(b->shader, &binary));
      const uint8_t *bytes = (const uint8_t *)binary.data;
      const uint16_t src[2][3] = {{20, 2, 4}, {20, 6, 8}};
      const bool hints[3] = {};
      unsigned found[2] = {};
      unsigned publication_slot[2] = {};
      for (unsigned slot = 1; slot <= 6; ++slot) {
         for (unsigned wait = 0; wait < 64; ++wait) {
            for (unsigned c = 0; c < 2; ++c) {
               uint8_t expected[16], length = 0;
               if (!agx3_stage_fma(AGX_ISA_ENCODING_AGX3B, 2 * c, slot, src[c],
                                   AGX3_STAGE_FMA_UNIFORM, hints, hints, wait,
                                   expected, &length))
                  continue;
               for (unsigned offset = 0; offset + length <= binary.size;
                    ++offset) {
                  if (memcmp(bytes + offset, expected, length))
                     continue;
                  found[c]++;
                  if (c == 0) {
                     EXPECT_EQ(wait != 0, loaded);
                  }
                  publication_slot[c] = slot;
               }
            }
         }
      }
      EXPECT_EQ(found[0], 1u);
      EXPECT_EQ(found[1], 1u);
      EXPECT_EQ(publication_slot[0], publication_slot[1]);
      for (unsigned c = 0; c < 2; ++c) {
         for (unsigned slot = 1; slot <= 6; ++slot) {
            uint8_t copy[AGX3_STAGE_MAX_LENGTH], length = 0;
            ASSERT_TRUE(agx3_stage_copy_source(AGX_ISA_ENCODING_AGX3B, 2 * c,
                                               outputs[c].value, false, slot, 0,
                                               copy, &length));
            for (unsigned offset = 0; offset + length <= binary.size; ++offset)
               EXPECT_NE(memcmp(bytes + offset, copy, length), 0);
         }
      }
      util_dynarray_fini(&binary);
      ralloc_free(mem);
   }
}

#ifndef NDEBUG

static agx_index
validation_span_value(agx_builder *b, bool memory, unsigned reg,
                      agx_size size, unsigned channels)
{
   agx_index value = agx_vec_temp(b->shader, size, channels);
   value.has_reg = true;
   value.reg = reg;
   value.memory = memory;
   agx_index uniform = agx_uniform(0, size);
   uniform.channels_m1 = channels - 1;
   agx_mov_to(b, value, uniform);
   return value;
}

static void
validation_span_diamond(bool memory, unsigned reg, agx_size size,
                        unsigned channels, bool clobber)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_block *entry = agx_start_block(b->shader);
   agx_block *left = agx_test_block(b->shader);
   agx_block *right = agx_test_block(b->shader);
   agx_block *join = agx_test_block(b->shader);
   agx_block_add_successor(entry, left);
   agx_block_add_successor(entry, right);
   agx_block_add_successor(left, join);
   agx_block_add_successor(right, join);
   agx_index value = validation_span_value(b, memory, reg, size, channels);
   if (clobber) {
      b->cursor = agx_after_block(right);
      validation_span_value(b, memory, reg + agx_index_size_16(value) - 1,
                            AGX_SIZE_16, 1);
   }
   b->cursor = agx_after_block(join);
   agx_unit_test(b, value);
   agx_validate_ra(b->shader);
   ralloc_free(mem);
}

TEST(NativeForms, RegisterValidationTracksMergeSpansAndClobbers)
{
   for (bool memory : {false, true}) {
      for (agx_size size : {AGX_SIZE_16, AGX_SIZE_32, AGX_SIZE_64}) {
         for (unsigned channels : {1u, 3u, 4u}) {
            unsigned width = channels * agx_size_align_16(size);
            for (unsigned reg : {8u, 128u, AGX_NUM_MODELED_REGS - width}) {
               for (bool clobber : {false, true}) {
                  EXPECT_EXIT({
                     std::signal(SIGABRT, +[](int) { std::_Exit(77); });
                     validation_span_diamond(memory, reg, size, channels, clobber);
                     std::_Exit(0);
                  }, ::testing::ExitedWithCode(clobber ? 77 : 0),
                     clobber ? "failed to validate RA" : "");
               }
            }
         }
      }
   }
}

TEST(NativeForms, RegisterValidationTracksPhysicalMovesAndSwaps)
{
   for (agx_size size : {AGX_SIZE_16, AGX_SIZE_32, AGX_SIZE_64}) {
      for (unsigned channels : {1u, 3u, 4u}) {
         EXPECT_EXIT({
            std::signal(SIGABRT, +[](int) { std::_Exit(77); });
            void *mem = ralloc_context(NULL);
            agx_builder *b = agx_test_builder(mem);
            agx_block *entry = agx_start_block(b->shader);
            agx_block *next = agx_test_block(b->shader);
            agx_block_add_successor(entry, next);
            agx_index a = validation_span_value(b, false, 16, size, channels);
            agx_index z = validation_span_value(b, false, 64, size, channels);
            unsigned spill_reg = AGX_NUM_MODELED_REGS - agx_index_size_16(a);
            b->cursor = agx_after_block(next);
            agx_index spill = agx_memory_register(spill_reg, size);
            spill.channels_m1 = channels - 1;
            spill.memory = true;
            agx_index source = agx_register(16, size);
            source.channels_m1 = channels - 1;
            agx_mov_to(b, spill, source);
            agx_index other = agx_register(64, size);
            other.channels_m1 = channels - 1;
            agx_swap(b, spill, other);
            a.reg = 64;
            z.memory = true;
            z.reg = spill_reg;
            agx_unit_test(b, a);
            agx_unit_test(b, z);
            agx_validate_ra(b->shader);
            ralloc_free(mem);
            std::_Exit(0);
         }, ::testing::ExitedWithCode(0), "");
      }
   }
}

TEST(NativeForms, RegisterValidationTracksPhiCopies)
{
   for (bool physical : {false, true}) {
      for (agx_size size : {AGX_SIZE_16, AGX_SIZE_32, AGX_SIZE_64}) {
         for (unsigned channels : {1u, 3u, 4u}) {
            EXPECT_EXIT({
               std::signal(SIGABRT, +[](int) { std::_Exit(77); });
               void *mem = ralloc_context(NULL);
               agx_builder *b = agx_test_builder(mem);
               agx_block *entry = agx_start_block(b->shader);
               agx_block *left = agx_test_block(b->shader);
               agx_block *right = agx_test_block(b->shader);
               agx_block *join = agx_test_block(b->shader);
               agx_block_add_successor(entry, left);
               agx_block_add_successor(entry, right);
               agx_block_add_successor(left, join);
               agx_block_add_successor(right, join);
               agx_index value = validation_span_value(b, false, 128, size, channels);
               b->cursor = agx_after_block(join);
               agx_index dest = value;
               dest.reg = 192;
               if (physical) {
                  dest = agx_register(192, size);
                  dest.channels_m1 = channels - 1;
               }
               agx_instr *phi = agx_phi_to(b, dest, 2);
               for (unsigned q = 0; q < 2; ++q) {
                  phi->src[q] = physical ? agx_register(128, size) : value;
                  phi->src[q].channels_m1 = channels - 1;
               }
               value.reg = 192;
               agx_unit_test(b, value);
               agx_validate_ra(b->shader);
               ralloc_free(mem);
               std::_Exit(0);
            }, ::testing::ExitedWithCode(0), "");
         }
      }
   }
}

TEST(NativeForms, RegisterValidationRetainsLoopHeaderRules)
{
   for (bool invalid_livein : {false, true}) {
      EXPECT_EXIT({
         std::signal(SIGABRT, +[](int) { std::_Exit(77); });
         void *mem = ralloc_context(NULL);
         agx_builder *b = agx_test_builder(mem);
         agx_block *entry = agx_start_block(b->shader);
         agx_block *header = agx_test_block(b->shader);
         agx_block *body = agx_test_block(b->shader);
         agx_block *exit = agx_test_block(b->shader);
         agx_block_add_successor(entry, header);
         agx_block_add_successor(header, body);
         agx_block_add_successor(header, exit);
         agx_block_add_successor(body, header);
         header->loop_header = true;
         agx_index incoming = validation_span_value(b, false, 128, AGX_SIZE_32, 1);
         b->cursor = agx_after_block(body);
         agx_index back = validation_span_value(b, false, 144, AGX_SIZE_32, 1);
         b->cursor = agx_after_block(header);
         agx_index value = agx_temp(b->shader, AGX_SIZE_32);
         value.has_reg = true;
         value.reg = 160;
         agx_instr *phi = agx_phi_to(b, value, 2);
         phi->src[0] = incoming;
         phi->src[1] = back;
         agx_unit_test(b, invalid_livein ? incoming : value);
         agx_validate_ra(b->shader);
         ralloc_free(mem);
         std::_Exit(0);
      }, ::testing::ExitedWithCode(invalid_livein ? 77 : 0),
         invalid_livein ? "failed to validate RA" : "");
   }
}

TEST(NativeForms, RegisterValidationTracksNarrowPhysicalCopySources)
{
   for (unsigned mode = 0; mode < 3; ++mode) {
      for (unsigned channels : {3u, 4u}) {
         EXPECT_EXIT({
            std::signal(SIGABRT, +[](int) { std::_Exit(77); });
            void *mem = ralloc_context(NULL);
            agx_builder *b = agx_test_builder(mem);
            agx_block *entry = agx_start_block(b->shader);
            agx_block *next = agx_test_block(b->shader);
            agx_block_add_successor(entry, next);
            bool source_memory = mode == 1;
            agx_index value = validation_span_value(
               b, source_memory, source_memory ? AGX_NUM_MODELED_REGS - 2 * channels
                                              : 128,
               AGX_SIZE_32, channels);
            agx_index source = agx_as_register(value);
            source.channels_m1 = 0;
            agx_index dest = agx_register(16, AGX_SIZE_32);
            dest.channels_m1 = channels - 1;
            dest.memory = mode == 0;
            b->cursor = agx_after_block(next);
            if (mode == 2) {
               agx_instr *phi = agx_phi_to(b, dest, 1);
               phi->src[0] = source;
            } else {
               agx_mov_to(b, dest, source);
            }
            value.reg = dest.value;
            value.memory = dest.memory;
            agx_unit_test(b, value);
            agx_validate_ra(b->shader);
            ralloc_free(mem);
            std::_Exit(0);
         }, ::testing::ExitedWithCode(0), "");
      }
   }
}

TEST(NativeForms, RegisterValidationRejectsOutOfModelCopySpans)
{
   for (bool phi_copy : {false, true}) {
      EXPECT_EXIT({
         std::signal(SIGABRT, +[](int) { std::_Exit(77); });
         std::signal(SIGSEGV, +[](int) { std::_Exit(78); });
         void *mem = ralloc_context(NULL);
         agx_builder *b = agx_test_builder(mem);
         agx_block *entry = agx_start_block(b->shader);
         agx_block *next = agx_test_block(b->shader);
         agx_block_add_successor(entry, next);
         b->cursor = agx_after_block(next);
         agx_index source = agx_register(0, AGX_SIZE_16);
         source.value = AGX_NUM_MODELED_REGS - 1;
         source.memory = !phi_copy;
         agx_index dest = agx_register(16, AGX_SIZE_32);
         if (phi_copy) {
            agx_instr *phi = agx_phi_to(b, dest, 1);
            phi->src[0] = source;
         } else {
            agx_mov_to(b, dest, source);
         }
         agx_validate_ra(b->shader);
         ralloc_free(mem);
         std::_Exit(0);
      }, ::testing::ExitedWithCode(77),
         "width <= AGX_NUM_MODELED_REGS - reg");
   }
}

#endif

TEST(NativeForms, SchedulerSsaBookkeepingStaysWithinEachBlock)
{
   for (bool ilp : {false, true}) {
      for (unsigned gap : {0u, 31u, 4095u}) {
         for (unsigned empty_blocks : {0u, 2u}) {
            SCOPED_TRACE(ilp);
            SCOPED_TRACE(gap);
            SCOPED_TRACE(empty_blocks);
            void *mem = ralloc_context(NULL);
            agx_shader_key key = {};
            key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
            key.dev.chip = AGX_CHIP_G17P;
            agx_builder *b = agx_test_builder(mem);
            b->shader->key = &key;
            b->shader->stage = MESA_SHADER_COMPUTE;
            b->shader->is_preamble = true;
            agx_block *last =
               list_first_entry(&b->shader->blocks, agx_block, link);
            agx_index a = agx_mov_imm(b, 32, 3);
            b->shader->alloc += gap;
            agx_index c = agx_mov_imm(b, 32, 5);
            agx_instr *observe_a = agx_unit_test(b, a);
            agx_instr *observe_c = agx_unit_test(b, c);
            auto next_block = [&]() {
               agx_block *next = agx_test_block(b->shader);
               last->successors[0] = next;
               util_dynarray_append(&next->predecessors, last);
               last = next;
               b->cursor = agx_after_block(last);
            };
            for (unsigned n = 0; n < empty_blocks; ++n)
               next_block();
            next_block();
            agx_index small = agx_mov_imm(b, 32, 7);
            next_block();
            agx_index sum = agx_iadd(b, a, c, 0);
            sum = agx_iadd(b, sum, agx_mov_imm(b, 32, 11), 0);
            sum = agx_iadd(b, sum, small, 0);
            agx_instr *observe_sum = agx_unit_test(b, sum);
            agx_index x = agx_mov_imm(b, 32, 13);
            agx_index y = agx_mov_imm(b, 32, 17);
            agx_instr *observe_other = agx_unit_test(b, agx_iadd(b, x, y, 0));

            std::vector<std::vector<agx_instr *>> original;
            agx_foreach_block(b->shader, block) {
               original.emplace_back();
               agx_foreach_instr_in_block(block, I)
                  original.back().push_back(I);
            }
            if (ilp)
               agx_ilp_schedule(b->shader);
            else
               agx_pressure_schedule(b->shader);

            std::vector<uint64_t> values(b->shader->alloc, UINT64_MAX);
            unsigned block_index = 0;
            unsigned observations = 0;
            agx_foreach_block(b->shader, block) {
               unsigned count = 0;
               agx_foreach_instr_in_block(block, I) {
                  ++count;
                  bool belonged = false;
                  for (agx_instr *before : original[block_index])
                     belonged |= before == I;
                  ASSERT_TRUE(belonged);
                  agx_foreach_ssa_src(I, src) {
                     ASSERT_LT(I->src[src].value, values.size());
                     ASSERT_NE(values[I->src[src].value], UINT64_MAX);
                  }
                  if (I->op == AGX_OPCODE_MOV_IMM) {
                     values[I->dest[0].value] = I->imm;
                  } else if (I->op == AGX_OPCODE_IADD) {
                     values[I->dest[0].value] =
                        uint32_t(values[I->src[0].value] +
                                 values[I->src[1].value]);
                  } else {
                     ASSERT_EQ(I->op, AGX_OPCODE_UNIT_TEST);
                     const unsigned expected = I == observe_a ? 3 :
                        I == observe_c ? 5 : I == observe_sum ? 26 : 30;
                     ASSERT_TRUE(I == observe_a || I == observe_c ||
                                 I == observe_sum || I == observe_other);
                     EXPECT_EQ(values[I->src[0].value], expected);
                     ++observations;
                  }
               }
               EXPECT_EQ(count, original[block_index].size());
               ++block_index;
            }
            EXPECT_EQ(block_index, original.size());
            EXPECT_EQ(observations, 4u);
            ralloc_free(mem);
         }
      }
   }
}

/* AGX2 preamble lowering replaces an unpromoted bindless sampler token with
 * its heap index. The main shader must still request the sampler heap. */
TEST(LegacySamplers, UnpromotedIndexNeedsHeap)
{
   for (bool dynamic : {false, true}) {
      for (bool border : {false, true}) {
         SCOPED_TRACE(::testing::Message() << "dynamic=" << dynamic
                                         << " border=" << border);
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_COMPUTE, &agx_nir_options, "AGX2 sampler heap");
         b.shader->info.workgroup_size[0] = 1;
         b.shader->info.workgroup_size[1] = 1;
         b.shader->info.workgroup_size[2] = 1;
         nir_def *index = dynamic
            ? nir_u2u16(&b, nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0))
            : nir_imm_intN_t(&b, 28, 16);
         nir_def *sampler = border ? index : nir_bindless_sampler_agx(
            &b, nir_imm_int(&b, 24), index);
         nir_def *texture = nir_bindless_image_agx(&b, nir_imm_int(&b, 0));
         nir_tex_instr *tex = nir_tex_instr_create(b.shader, 4);
         tex->op = nir_texop_txl;
         tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
         tex->coord_components = 2;
         tex->dest_type = nir_type_float32;
         tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_backend1,
            nir_vec2(&b, nir_imm_float(&b, 0.25), nir_imm_float(&b, 0.5)));
         tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_lod, nir_imm_float(&b, 0));
         tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_texture_handle, texture);
         tex->src[3] = nir_tex_src_for_ssa(nir_tex_src_sampler_handle, sampler);
         nir_def_init(&tex->instr, &tex->def, 4, 32);
         nir_builder_instr_insert(&b, &tex->instr);
         nir_store_global(&b, &tex->def, nir_imm_int64(&b, 0x1000), .align_mul = 16);
         agx_shader_key key = {};
         key.dev.encoding = AGX_ISA_ENCODING_AGX2;
         key.dev.chip = AGX_CHIP_G13G;
         key.promote_textures = false;
         agx_preprocess_nir(b.shader);
         agx_shader_part part = {};
         bool ok = agx_compile_shader_nir(b.shader, &key, &part);
         EXPECT_TRUE(ok);
         if (ok) {
            EXPECT_TRUE(part.info.uses_sampler_heap);
         }
         free(part.binary);
         ralloc_free(b.shader);
      }
   }
}

TEST(LegacyFragments, SideEffectsRetainUpstreamLateDepth)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "legacy side effects");
   b.shader->info.writes_memory = true;
   nir_store_global(&b, nir_imm_int(&b, 1), nir_imm_int64(&b, 0x1000),
                    .align_mul = 4);
   EXPECT_TRUE(agx_nir_lower_frag_sidefx(b.shader, false));
   EXPECT_NE(b.shader->info.outputs_written & BITFIELD64_BIT(FRAG_RESULT_DEPTH), 0u);
   unsigned never = 0, masks = 0;
   nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic) continue;
         auto op = nir_instr_as_intrinsic(instr)->intrinsic;
         never += op == nir_intrinsic_load_depth_never_agx;
         masks += op == nir_intrinsic_sample_mask_agx;
      }
   }
   EXPECT_EQ(never, 1u);
   EXPECT_EQ(masks, 0u);
   ralloc_free(b.shader);
}
