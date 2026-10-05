#include "agx_test.h"
#include "agx3_alu_core.h"
#include "agx3_pack.h"
#include <gtest/gtest.h>
#include <vector>

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
