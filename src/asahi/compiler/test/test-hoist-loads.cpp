/*
 * SPDX-License-Identifier: MIT
 */

#include "agx_builder.h"
#include "agx_compiler.h"
#include "agx_test.h"

#include <gtest/gtest.h>

static void
pass(agx_context *ctx)
{
   agx_hoist_loads(ctx);
   agx_reindex_ssa(ctx);
}

#define CASE(instr, expected) INSTRUCTION_CASE(instr, expected, pass)
#define NEGCASE(instr)        INSTRUCTION_CASE(instr, instr, pass)

class HoistLoads : public testing::Test {
 protected:
   HoistLoads()
   {
      mem_ctx = ralloc_context(NULL);
   }

   ~HoistLoads()
   {
      ralloc_free(mem_ctx);
   }

   void *mem_ctx;
};

static agx_index
load(agx_builder *b, agx_index off, bool coherent)
{
   agx_index v = agx_temp(b->shader, AGX_SIZE_32);
   agx_device_load_to(b, v, agx_uniform(0, AGX_SIZE_64), off, AGX_FORMAT_I32, 1,
                      0, coherent);
   return v;
}

TEST_F(HoistLoads, AboveAluAndThreadgroupBarrier)
{
   CASE(
      {
         agx_index x = agx_mov_imm(b, AGX_SIZE_32, 0xcafe);
         agx_index off = agx_mov_imm(b, AGX_SIZE_32, 4);
         agx_index y = agx_fadd(b, x, x);
         agx_threadgroup_barrier(b);
         agx_index v = load(b, off, false);
         agx_unit_test(b, v);
         agx_unit_test(b, y);
      },
      {
         agx_index x = agx_mov_imm(b, AGX_SIZE_32, 0xcafe);
         agx_index off = agx_mov_imm(b, AGX_SIZE_32, 4);
         agx_index v = load(b, off, false);
         agx_index y = agx_fadd(b, x, x);
         agx_threadgroup_barrier(b);
         agx_unit_test(b, v);
         agx_unit_test(b, y);
      });
}

TEST_F(HoistLoads, StopsAtItsOwnSource)
{
   NEGCASE({
      agx_index x = agx_mov_imm(b, AGX_SIZE_32, 0xcafe);
      agx_index y = agx_fadd(b, x, x);
      agx_index off = agx_mov_imm(b, AGX_SIZE_32, 4);
      agx_index v = load(b, off, false);
      agx_unit_test(b, v);
      agx_unit_test(b, y);
   });
}

TEST_F(HoistLoads, NotAboveDeviceStore)
{
   NEGCASE({
      agx_index off = agx_mov_imm(b, AGX_SIZE_32, 4);
      agx_index x = agx_mov_imm(b, AGX_SIZE_32, 0xcafe);
      agx_device_store(b, x, agx_uniform(0, AGX_SIZE_64), off, AGX_FORMAT_I32,
                       1, 0, false);
      agx_index v = load(b, off, false);
      agx_unit_test(b, v);
   });
}

TEST_F(HoistLoads, NotAboveMemoryBarrier)
{
   NEGCASE({
      agx_index off = agx_mov_imm(b, AGX_SIZE_32, 4);
      agx_memory_barrier(b);
      agx_index v = load(b, off, false);
      agx_unit_test(b, v);
   });
}

TEST_F(HoistLoads, NotAboveDoorbell)
{
   NEGCASE({
      agx_index off = agx_mov_imm(b, AGX_SIZE_32, 4);
      agx_doorbell(b, 1);
      agx_index v = load(b, off, false);
      agx_unit_test(b, v);
   });
}

TEST_F(HoistLoads, NotAboveUniformStoreToItsBase)
{
   NEGCASE({
      agx_index off = agx_mov_imm(b, AGX_SIZE_32, 4);
      agx_index x = agx_mov_imm(b, AGX_SIZE_32, 0xcafe);
      agx_uniform_store(b, x, agx_immediate(0), 1);
      agx_index v = load(b, off, false);
      agx_unit_test(b, v);
   });
}

TEST_F(HoistLoads, AboveUniformStoreToOtherRegisters)
{
   CASE(
      {
         agx_index off = agx_mov_imm(b, AGX_SIZE_32, 4);
         agx_index x = agx_mov_imm(b, AGX_SIZE_32, 0xcafe);
         agx_uniform_store(b, x, agx_immediate(32), 1);
         agx_index v = load(b, off, false);
         agx_unit_test(b, v);
      },
      {
         agx_index off = agx_mov_imm(b, AGX_SIZE_32, 4);
         agx_index v = load(b, off, false);
         agx_index x = agx_mov_imm(b, AGX_SIZE_32, 0xcafe);
         agx_uniform_store(b, x, agx_immediate(32), 1);
         agx_unit_test(b, v);
      });
}

TEST_F(HoistLoads, AboveUniformStoreAdjacentRegisters)
{
   CASE(
      {
         agx_index off = agx_mov_imm(b, AGX_SIZE_32, 4);
         agx_index x = agx_mov_imm(b, AGX_SIZE_32, 0xcafe);
         agx_uniform_store(b, x, agx_immediate(4), 1);
         agx_index v = load(b, off, false);
         agx_unit_test(b, v);
      },
      {
         agx_index off = agx_mov_imm(b, AGX_SIZE_32, 4);
         agx_index v = load(b, off, false);
         agx_index x = agx_mov_imm(b, AGX_SIZE_32, 0xcafe);
         agx_uniform_store(b, x, agx_immediate(4), 1);
         agx_unit_test(b, v);
      });
}

TEST_F(HoistLoads, NotAboveUniformStorePartialOverlap)
{
   NEGCASE({
      agx_index x = agx_mov_imm(b, AGX_SIZE_32, 0xcafe);
      agx_uniform_store(b, x, agx_immediate(3), 1);
      agx_index v = agx_temp(b->shader, AGX_SIZE_32);
      agx_device_load_to(b, v, agx_uniform(0, AGX_SIZE_64),
                         agx_immediate(4), AGX_FORMAT_I32, 1, 0, false);
      agx_unit_test(b, v);
   });
}

TEST_F(HoistLoads, CoherentLoadStays)
{
   NEGCASE({
      agx_index x = agx_mov_imm(b, AGX_SIZE_32, 0xcafe);
      agx_index off = agx_mov_imm(b, AGX_SIZE_32, 4);
      agx_index y = agx_fadd(b, x, x);
      agx_index v = load(b, off, true);
      agx_unit_test(b, v);
      agx_unit_test(b, y);
   });
}
