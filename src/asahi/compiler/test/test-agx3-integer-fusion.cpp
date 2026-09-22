/* SPDX-License-Identifier: MIT */
#include <gtest/gtest.h>
#include "agx_builder.h"
#include "agx_compiler.h"
#include "agx_test.h"

/* Expected truth tables from the four Boolean input combinations, independent
 * of util_lut2_invert_source used by the optimizer. Columns complement neither,
 * source zero, source one, or both sources. */
static const unsigned tables[3][4] = {
   {0x8, 0x4, 0x2, 0x1},
   {0xe, 0xd, 0xb, 0x7},
   {0x6, 0x9, 0x9, 0x6},
};

TEST(AGX3IntegerFusion, FuseWordUniformSourceComplements)
{
   void *mem = ralloc_context(NULL);
   struct agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   for (unsigned op = 0; op < 3; ++op) {
      for (unsigned mask = 1; mask < 4; ++mask) {
         agx_builder *actual = agx_test_builder(mem);
         agx_builder *expected = agx_test_builder(mem);
         actual->shader->key = expected->shader->key = &key;
         const agx_index x = agx_register(0, AGX_SIZE_32);
         const agx_index y = agx_uniform(12, AGX_SIZE_32);
         agx_index out = agx_temp(actual->shader, AGX_SIZE_32);
         agx_index wanted = agx_temp(expected->shader, AGX_SIZE_32);
         agx_index a = (mask & 1) ? agx_not(actual, x) : x;
         agx_index b = (mask & 2) ? agx_not(actual, y) : y;
         agx_bitop_to(actual, out, a, b, tables[op][0]);
         agx_unit_test(actual, out);
         agx_bitop_to(expected, wanted, x, y, tables[op][mask]);
         agx_unit_test(expected, wanted);
         agx_optimizer_backward(actual->shader);
         agx_optimizer_forward(actual->shader);
         agx_dce(actual->shader, true);
         ASSERT_SHADER_EQUAL(actual->shader, expected->shader);
      }
   }
   ralloc_free(mem);
}

TEST(AGX3IntegerFusion, PreserveRegisterSourceComplements)
{
   void *mem = ralloc_context(NULL);
   struct agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   for (auto size : {AGX_SIZE_16, AGX_SIZE_32}) {
      for (unsigned op = 0; op < 3; ++op) {
         for (unsigned mask = 1; mask < 4; ++mask) {
            agx_builder *actual = agx_test_builder(mem);
            agx_builder *expected = agx_test_builder(mem);
            for (agx_builder *builder : {actual, expected}) {
               builder->shader->key = &key;
               agx_index x = agx_register(0, size);
               agx_index y = agx_register(2, size);
               if (mask & 1)
                  x = agx_not(builder, x);
               if (mask & 2)
                  y = agx_not(builder, y);
               agx_unit_test(builder, agx_bitop(builder, x, y, tables[op][0]));
            }
            agx_optimizer_backward(actual->shader);
            agx_optimizer_forward(actual->shader);
            agx_dce(actual->shader, true);
            ASSERT_SHADER_EQUAL(actual->shader, expected->shader);
         }
      }
   }
   ralloc_free(mem);
}
