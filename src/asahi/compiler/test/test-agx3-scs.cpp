#include "agx_test.h"
#include "agx3_pack.h"
#include "agx3_scs_core.h"
#include <gtest/gtest.h>

TEST(SCSRegisterIR, NeoScalarAndVector)
{
   for (unsigned mask : {1u, 3u, 15u}) {
      void *memctx = ralloc_context(nullptr);
      agx_builder *b = agx_test_builder(memctx);
      agx_shader_key key = {};
      agx_shader_info info = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      b->shader->key = &key;
      b->shader->out = &info;
      b->shader->nir = rzalloc(memctx, nir_shader);
      b->shader->nir->info.stage = MESA_SHADER_COMPUTE;
      b->shader->stage = info.stage = MESA_SHADER_COMPUTE;
      b->shader->max_reg = 32;
      agx_index value = agx_register(4, AGX_SIZE_32);
      value.channels_m1 = util_last_bit(mask) - 1;
      agx_scs_store_register_indexed_agx3(
         b, value, agx_register(32, AGX_SIZE_32),
         agx_register(28, AGX_SIZE_32), 0, AGX_FORMAT_I32, mask, 3);
      agx_index result = agx_register(12, AGX_SIZE_32);
      result.channels_m1 = util_last_bit(mask) - 1;
      agx_scs_load_register_indexed_agx3_to(
         b, result, agx_register(32, AGX_SIZE_32),
         agx_register(28, AGX_SIZE_32), 512, AGX_FORMAT_I32, mask, 3, false, false);
      agx_stop(b);
      util_dynarray binary = UTIL_DYNARRAY_INIT;
      ASSERT_TRUE(agx3_pack_binary(b->shader, &binary));
      unsigned stores = 0, loads = 0;
      for (unsigned off = 0; off < binary.size; off += 2) {
         const uint8_t *bytes = static_cast<const uint8_t *>(binary.data) + off;
         for (unsigned len : {12u, 14u, 16u}) {
            if (off + len > binary.size)
               continue;
            agx3_scs_desc d = {};
            if (agx3_scs_decode_store(AGX_ISA_ENCODING_AGX3B, bytes, len, &d)) {
               EXPECT_EQ(d.data, 4);
               EXPECT_EQ(d.component_mask, mask);
               EXPECT_EQ(d.base_register, 32);
               EXPECT_EQ(d.index, 28);
               EXPECT_EQ(d.address_selector, 3);
               EXPECT_EQ(d.byte_offset, 0);
               stores++;
            }
            if (agx3_scs_decode_load(AGX_ISA_ENCODING_AGX3B, bytes, len, &d)) {
               EXPECT_EQ(d.data, 12);
               EXPECT_EQ(d.component_mask, mask);
               EXPECT_EQ(d.base_register, 32);
               EXPECT_EQ(d.index, 28);
               EXPECT_EQ(d.address_selector, 3);
               EXPECT_EQ(d.byte_offset, 512);
               EXPECT_GT(d.slot, 0);
               loads++;
            }
         }
      }
      EXPECT_EQ(stores, 1u);
      EXPECT_EQ(loads, 1u);
      util_dynarray_fini(&binary);
      ralloc_free(memctx);
   }
}

TEST(SCSRegisterIR, NeoRejectsVectorPastRegisterFile)
{
   agx3_scs_desc d = {};
   d.data = 254;
   d.width = AGX3_MEM_W32;
   d.address_kind = AGX3_SCS_ADDRESS_REGISTER_INDEXED;
   d.base_register = 32;
   d.index = 28;
   d.component_mask = 3;
   d.has_address_selector = true;
   d.address_selector = 3;
   uint8_t bytes[AGX3_SCS_MAX_LENGTH], length;
   EXPECT_FALSE(agx3_scs_encode_store(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   d.data = 4;
   EXPECT_TRUE(agx3_scs_encode_store(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
}
