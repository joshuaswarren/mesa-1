/*
 * SPDX-License-Identifier: MIT
 */

#include "agx_builder.h"
#include "agx_compiler.h"
#include "agx_test.h"

#include <gtest/gtest.h>

class InsertWaits : public testing::Test {
 protected:
   InsertWaits()
   {
      mem_ctx = ralloc_context(NULL);
   }

   ~InsertWaits()
   {
      ralloc_free(mem_ctx);
   }

   void *mem_ctx;
};

static agx_instr *
emit_reg_load(agx_builder *b, unsigned reg_16)
{
   return agx_device_load_to(b, agx_register(reg_16, AGX_SIZE_32),
                             agx_uniform(0, AGX_SIZE_64), agx_immediate(0),
                             AGX_FORMAT_I32, 1, 0, false);
}

static unsigned
count_waits_before_first_alu(agx_block *blk)
{
   unsigned waits = 0;
   agx_foreach_instr_in_block(blk, I) {
      if (I->op == AGX_OPCODE_WAIT)
         waits++;
      else if (I->op != AGX_OPCODE_DEVICE_LOAD)
         break;
   }
   return waits;
}

TEST_F(InsertWaits, SmartWaitsAvoidsMidBurstOverflow)
{
   /* 1 early load + 8 late loads + 2 early loads = 11 async loads in one burst
    * (> AGX_MAX_PENDING = 8), followed by immediate use of the 3 early loads,
    * 70 independent ALU instructions, and finally use of the 8 late loads.
    */
   auto build_case = [&](bool smart, mesa_shader_stage stage) {
      agx_builder *b = agx_test_builder(mem_ctx);
      b->shader->stage = stage;
      struct agx_shader_key *key = rzalloc(mem_ctx, struct agx_shader_key);
      key->dev.smart_waits = smart;
      b->shader->key = key;

      /* First early load: r0 */
      emit_reg_load(b, 0);
      /* Eight late loads: r2..r16 */
      for (unsigned i = 0; i < 8; ++i)
         emit_reg_load(b, 2 + 2 * i);
      /* Two more early loads: r18, r20 */
      emit_reg_load(b, 18);
      emit_reg_load(b, 20);

      /* Early consumer uses r0, r18, r20 */
      agx_iadd_to(b, agx_register(30, AGX_SIZE_32),
                  agx_register(0, AGX_SIZE_32), agx_register(18, AGX_SIZE_32), 0);
      agx_iadd_to(b, agx_register(32, AGX_SIZE_32),
                  agx_register(30, AGX_SIZE_32), agx_register(20, AGX_SIZE_32), 0);

      /* 70 independent ALU ops separating early and late uses */
      for (unsigned i = 0; i < 70; ++i) {
         agx_iadd_to(b, agx_register(34, AGX_SIZE_32),
                     agx_register(34, AGX_SIZE_32), agx_immediate(1), 0);
      }

      /* Late consumers use r2..r16 */
      for (unsigned i = 0; i < 8; ++i) {
         agx_iadd_to(b, agx_register(34, AGX_SIZE_32),
                     agx_register(34, AGX_SIZE_32),
                     agx_register(2 + 2 * i, AGX_SIZE_32), 0);
      }

      agx_insert_waits(b->shader);
      return b->shader;
   };

   agx_context *stock = build_case(false, MESA_SHADER_COMPUTE);
   agx_context *smart = build_case(true, MESA_SHADER_COMPUTE);
   agx_context *frag = build_case(true, MESA_SHADER_FRAGMENT);

   agx_block *stock_blk = agx_start_block(stock);
   agx_block *smart_blk = agx_start_block(smart);
   agx_block *frag_blk = agx_start_block(frag);

   /* Stock dumps loads 2..9 onto slot 0 and stalls mid-burst on overflow. */
   EXPECT_GE(count_waits_before_first_alu(stock_blk), 2u);

   /* Smart waits keeps all 8 late loads on one slot and 3 early loads on the
    * other: zero mid-burst overflow stalls, only 1 wait (early slot) before
    * the first early consumer.
    */
   EXPECT_EQ(count_waits_before_first_alu(smart_blk), 1u);

   /* Non-compute shaders keep stock behavior even when smart_waits is set. */
   EXPECT_TRUE(agx_shader_equal(stock, frag));
}
