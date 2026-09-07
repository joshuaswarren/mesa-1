/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * End-to-end backend test: build AGX IR directly, run agx3_pack_binary, and
 * check the emitted AGX3 byte stream against the independently validated
 * encoders. This exercises the real IR walk, per-instruction translation,
 * modifier routing, and the terminal-end footer -- the lowering glue that the
 * portable encoder-core host tests do not cover.
 */

#include "agx_test.h"
extern "C" {
#include "agx_nir.h"
}
#include "compiler/nir/nir_builder.h"

#include <algorithm>
#include <vector>
#include <gtest/gtest.h>

#include "agx3_alu_core.h"
#include "agx3_isel_core.h"
#include "agx3_atomic_core.h"
#include "agx3_cf_core.h"
#include "agx3_copy_core.h"
#include "agx3_falu_mod_core.h"
#include "agx3_fltnorm_core.h"
#include "agx3_local_tps_core.h"
#include "agx3_mem_core.h"
#include "agx3_pack.h"
#include "agx3_scs_core.h"
#include "agx3_slot_core.h"
#include "agx3_sr_core.h"
#include "agx3_stage_core.h"
#include "agx3_static_spill_core.h"
#include "agx3_subgroup_core.h"
#include "agx3_tex_core.h"
#include "agx3_tile_core.h"
#include "agx3_pixel_core.h"
#include "agx3_vary_core.h"

/* agx_nir_lower_tilebuffer's ALL_SAMPLES: the pixel-rate sample mask, and the
 * only immediate AGX3's single-sample tile group can represent. */
static constexpr unsigned AGX_ALL_SAMPLES = 0xFF;

static void
append(std::vector<uint8_t> &v, const uint8_t *b, unsigned n)
{
   v.insert(v.end(), b, b + n);
}

class AGX3PackBinary : public testing::Test {
 protected:
   void *mem_ctx;
   agx_shader_key key;

   AGX3PackBinary()
   {
      mem_ctx = ralloc_context(NULL);
      memset(&key, 0, sizeof(key));
   }
   ~AGX3PackBinary()
   {
      ralloc_free(mem_ctx);
   }

   agx_builder *builder(enum agx_isa_encoding enc)
   {
      key.dev.encoding = enc;
      agx_builder *b = agx_test_builder(mem_ctx);
      b->shader->key = &key;
      return b;
   }

   std::vector<uint8_t> pack(agx_builder *b, bool expect_ok = true)
   {
      struct util_dynarray bin;
      util_dynarray_init(&bin, mem_ctx);
      bool ok = agx3_pack_binary(b->shader, &bin);
      EXPECT_EQ(ok, expect_ok);
      const uint8_t *d = (const uint8_t *)bin.data;
      return std::vector<uint8_t>(d, d + bin.size);
   }
};

TEST_F(AGX3PackBinary, StraightLineALU)
{
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e == 0 ? AGX_ISA_ENCODING_AGX3B : AGX_ISA_ENCODING_AGX3C;
      agx_builder *b = builder(enc);
      agx_index w0 = agx_register(0, AGX_SIZE_32);
      agx_index w2 = agx_register(2, AGX_SIZE_32);
      agx_index w4 = agx_register(4, AGX_SIZE_32);
      agx_index w6 = agx_register(6, AGX_SIZE_32);

      agx_mov_imm_to(b, w0, 0x12345678);
      agx_fadd_to(b, w4, w0, w2);
      agx_fmul_to(b, w6, w4, w2);
      agx_iadd_to(b, w4, w0, w2, 0);
      agx_stop(b);

      std::vector<uint8_t> want;
      struct agx3_encoded_ldimm ld;
      ASSERT_TRUE(agx3_pack_ldimm_u32(enc, 0, 0x12345678, &ld));
      append(want, ld.bytes, ld.length);
      struct agx3_encoded_fadd_f32_reg_reg fa;
      ASSERT_TRUE(agx3_pack_fadd_f32_reg_reg(enc, 4, 0, 2, &fa));
      if (enc == AGX_ISA_ENCODING_AGX3B)
         want.insert(want.end(), {0x29, 3, 0, 1}); /* No hints or waits. */
      else
         append(want, fa.bytes, fa.length);
      struct agx3_encoded_fmul_f32_reg_reg fm;
      ASSERT_TRUE(agx3_pack_fmul_f32_reg_reg(enc, 6, 4, 2, &fm));
      if (enc == AGX_ISA_ENCODING_AGX3B)
         want.insert(want.end(), {0x39, 3, 1, 5}); /* Both inputs retained. */
      else
         append(want, fm.bytes, fm.length);
      struct agx3_encoded_iadd_u32_reg_reg ia;
      ASSERT_TRUE(agx3_pack_iadd_u32_reg_reg(enc, 4, 0, 2, &ia));
      append(want, ia.bytes, ia.length);
      struct agx3_encoded_end en;
      ASSERT_TRUE(agx3_pack_end(enc, &en));
      append(want, en.bytes, en.length);

      EXPECT_EQ(pack(b), want);
   }
}

TEST_F(AGX3PackBinary, G17PRG16IntegerEOTUsesExactNativeI16Program)
{
   static const uint8_t native[] = {
      0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06,
      0x0c, 0x9d, 0x10, 0x26, 0x24, 0xa9, 0x10, 0x26,
      0x9f, 0x10, 0x54, 0x00, 0x00, 0x04, 0x14, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x9f, 0x20, 0x54, 0x02, 0x00, 0x00, 0x10, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x97, 0x05, 0x54, 0x04, 0x00, 0x08, 0x00, 0x00,
      0xd0, 0x40, 0x10, 0x00,
      0x57, 0x10, 0x54, 0x00, 0x00, 0x00, 0x00, 0x80,
      0xa8, 0x15, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
   };

   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_COMPUTE;
   agx_index coords = agx_register(0, AGX_SIZE_32);
   coords.channels_m1 = 1;
   agx_get_sr_to(b, agx_register(0, AGX_SIZE_32),
                 AGX_SR_THREAD_POSITION_IN_GRID_X);
   agx_get_sr_to(b, agx_register(2, AGX_SIZE_32),
                 AGX_SR_THREAD_POSITION_IN_GRID_Y);
   agx_mov_imm_to(b, agx_register(4, AGX_SIZE_32), 0);
   agx_block_image_store(b, agx_immediate(0), agx_immediate(0),
                         agx_register(4, AGX_SIZE_32), coords,
                         AGX_FORMAT_I16, AGX_DIM_2D, true);
   agx_stop(b);

   EXPECT_EQ(pack(b), std::vector<uint8_t>(std::begin(native),
                                           std::end(native)));
   EXPECT_EQ(b->shader->max_reg, 7);
   EXPECT_TRUE(b->shader->agx3_staging_known);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 8);
}

TEST_F(AGX3PackBinary, G17PI8EOTUsesExactNativeProgram)
{
   static const uint8_t native[] = {
      0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06,
      0x0c, 0x9d, 0x10, 0x26, 0x24, 0xa9, 0x10, 0x26,
      0x9f, 0x10, 0x54, 0x00, 0x00, 0x04, 0x14, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x9f, 0x20, 0x54, 0x02, 0x00, 0x00, 0x10, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x97, 0x05, 0x54, 0x04, 0x00, 0x08, 0x00, 0x00,
      0xd0, 0x40, 0x10, 0x00,
      0x57, 0x10, 0x54, 0x00, 0x00, 0x00, 0x00, 0x80,
      0xa8, 0x55, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
   };

   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_COMPUTE;
   agx_index coords = agx_register(0, AGX_SIZE_32);
   coords.channels_m1 = 1;
   agx_get_sr_to(b, agx_register(0, AGX_SIZE_32),
                 AGX_SR_THREAD_POSITION_IN_GRID_X);
   agx_get_sr_to(b, agx_register(2, AGX_SIZE_32),
                 AGX_SR_THREAD_POSITION_IN_GRID_Y);
   agx_mov_imm_to(b, agx_register(4, AGX_SIZE_32), 0);
   agx_block_image_store(b, agx_immediate(0), agx_immediate(0),
                         agx_register(4, AGX_SIZE_32), coords,
                         AGX_FORMAT_I8, AGX_DIM_2D, true);
   agx_stop(b);

   EXPECT_EQ(pack(b), std::vector<uint8_t>(std::begin(native),
                                           std::end(native)));
   EXPECT_EQ(b->shader->max_reg, 7);
   EXPECT_TRUE(b->shader->agx3_staging_known);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 8);
}

TEST_F(AGX3PackBinary, G17PS8NormEOTUsesExactNativeProgram)
{
   static const uint8_t native[] = {
      0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06,
      0x0c, 0x9d, 0x10, 0x26, 0x24, 0xa9, 0x10, 0x26,
      0x9f, 0x10, 0x54, 0x00, 0x00, 0x04, 0x14, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x9f, 0x20, 0x54, 0x02, 0x00, 0x00, 0x10, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x97, 0x05, 0x54, 0x04, 0x00, 0x08, 0x00, 0x00,
      0xd0, 0x40, 0x10, 0x00,
      0x57, 0x10, 0x54, 0x00, 0x00, 0x00, 0x00, 0x80,
      0xa8, 0x75, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
   };

   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_COMPUTE;
   agx_index coords = agx_register(0, AGX_SIZE_32);
   coords.channels_m1 = 1;
   agx_get_sr_to(b, agx_register(0, AGX_SIZE_32),
                 AGX_SR_THREAD_POSITION_IN_GRID_X);
   agx_get_sr_to(b, agx_register(2, AGX_SIZE_32),
                 AGX_SR_THREAD_POSITION_IN_GRID_Y);
   agx_mov_imm_to(b, agx_register(4, AGX_SIZE_32), 0);
   agx_block_image_store(b, agx_immediate(0), agx_immediate(0),
                         agx_register(4, AGX_SIZE_32), coords,
                         AGX_FORMAT_S8NORM, AGX_DIM_2D, true);
   agx_stop(b);

   EXPECT_EQ(pack(b), std::vector<uint8_t>(std::begin(native),
                                           std::end(native)));
   EXPECT_EQ(b->shader->max_reg, 7);
   EXPECT_TRUE(b->shader->agx3_staging_known);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 8);
}

TEST_F(AGX3PackBinary, G17PRG16EOTUsesExactOneWordNativeProgram)
{
   key.g17p_rt0_eot_words = 1;
   static const uint8_t native[] = {
      0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06,
      0x0c, 0x9d, 0x10, 0x26, 0x24, 0xa9, 0x10, 0x26,
      0x9f, 0x10, 0x54, 0x00, 0x00, 0x04, 0x14, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x9f, 0x20, 0x54, 0x02, 0x00, 0x00, 0x10, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x97, 0x05, 0x54, 0x04, 0x00, 0x08, 0x00, 0x00,
      0xd0, 0x40, 0x10, 0x00,
      0x57, 0x10, 0x54, 0x00, 0x00, 0x00, 0x00, 0x80,
      0xa8, 0x75, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
   };

   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_COMPUTE;
   agx_index coords = agx_register(0, AGX_SIZE_32);
   coords.channels_m1 = 1;
   agx_get_sr_to(b, agx_register(0, AGX_SIZE_32),
                 AGX_SR_THREAD_POSITION_IN_GRID_X);
   agx_get_sr_to(b, agx_register(2, AGX_SIZE_32),
                 AGX_SR_THREAD_POSITION_IN_GRID_Y);
   agx_mov_imm_to(b, agx_register(4, AGX_SIZE_32), 0);
   agx_block_image_store(b, agx_immediate(0), agx_immediate(0),
                         agx_register(4, AGX_SIZE_32), coords,
                         AGX_FORMAT_U16NORM, AGX_DIM_2D, true);
   agx_stop(b);

   EXPECT_EQ(pack(b), std::vector<uint8_t>(std::begin(native),
                                           std::end(native)));
   EXPECT_EQ(b->shader->max_reg, 7);
   EXPECT_TRUE(b->shader->agx3_staging_known);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 8);
}

TEST_F(AGX3PackBinary, G17PScalarI32EOTUsesExactNativeProgram)
{
   /* Fresh Apple G17P builtin EOT key 0x04021100. The PBE descriptor carries
    * the logical UINT/FLOAT interpretation; this is the exact scalar store. */
   static const uint8_t native[] = {
      0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06,
      0x0c, 0x9d, 0x10, 0x26, 0x24, 0xa9, 0x10, 0x26,
      0x9f, 0x10, 0x54, 0x00, 0x00, 0x04, 0x14, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x9f, 0x20, 0x54, 0x02, 0x00, 0x00, 0x10, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x97, 0x05, 0x54, 0x04, 0x00, 0x08, 0x00, 0x00,
      0xd0, 0x40, 0x10, 0x00,
      0x57, 0x10, 0x54, 0x00, 0x00, 0x00, 0x00, 0x80,
      0xa8, 0x35, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
   };

   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_COMPUTE;
   key.g17p_rt0_eot_i32 = true;
   key.g17p_rt0_eot_words = 1;
   agx_index coords = agx_register(0, AGX_SIZE_32);
   coords.channels_m1 = 1;
   agx_get_sr_to(b, agx_register(0, AGX_SIZE_32),
                 AGX_SR_THREAD_POSITION_IN_GRID_X);
   agx_get_sr_to(b, agx_register(2, AGX_SIZE_32),
                 AGX_SR_THREAD_POSITION_IN_GRID_Y);
   agx_mov_imm_to(b, agx_register(4, AGX_SIZE_32), 0);
   agx_block_image_store(b, agx_immediate(0), agx_immediate(0),
                         agx_register(4, AGX_SIZE_32), coords,
                         AGX_FORMAT_I32, AGX_DIM_2D, true);
   agx_stop(b);

   EXPECT_EQ(pack(b), std::vector<uint8_t>(std::begin(native),
                                           std::end(native)));
   EXPECT_EQ(b->shader->max_reg, 7);
   EXPECT_TRUE(b->shader->agx3_staging_known);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 8);
}

TEST_F(AGX3PackBinary, G17PRG32FloatEOTUsesExactTwoWordNativeProgram)
{
   /* Fresh Apple G17P builtin EOT key 0x08021100. Its exact text is
    * byte-identical to scalar key 0x04021100; PBE carries channels R32G32.
    * Text SHA256: 60beeb253d8f81af993e93617ed3fca4559e0f614c8465f4cd2bb42c0f1f5fb6. */
   static const uint8_t native[] = {
      0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06,
      0x0c, 0x9d, 0x10, 0x26, 0x24, 0xa9, 0x10, 0x26,
      0x9f, 0x10, 0x54, 0x00, 0x00, 0x04, 0x14, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x9f, 0x20, 0x54, 0x02, 0x00, 0x00, 0x10, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x97, 0x05, 0x54, 0x04, 0x00, 0x08, 0x00, 0x00,
      0xd0, 0x40, 0x10, 0x00,
      0x57, 0x10, 0x54, 0x00, 0x00, 0x00, 0x00, 0x80,
      0xa8, 0x35, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
   };

   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_COMPUTE;
   key.g17p_rt0_eot_i32 = true;
   key.g17p_rt0_eot_words = 2;
   agx_index coords = agx_register(0, AGX_SIZE_32);
   coords.channels_m1 = 1;
   agx_get_sr_to(b, agx_register(0, AGX_SIZE_32),
                 AGX_SR_THREAD_POSITION_IN_GRID_X);
   agx_get_sr_to(b, agx_register(2, AGX_SIZE_32),
                 AGX_SR_THREAD_POSITION_IN_GRID_Y);
   agx_mov_imm_to(b, agx_register(4, AGX_SIZE_32), 0);
   agx_block_image_store(b, agx_immediate(0), agx_immediate(0),
                         agx_register(4, AGX_SIZE_32), coords,
                         AGX_FORMAT_I32, AGX_DIM_2D, true);
   agx_stop(b);

   EXPECT_EQ(pack(b), std::vector<uint8_t>(std::begin(native),
                                           std::end(native)));
   EXPECT_EQ(b->shader->max_reg, 7);
   EXPECT_TRUE(b->shader->agx3_staging_known);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 8);
}

TEST_F(AGX3PackBinary, G17PRG32UintEOTUsesExactTwoWordNativeProgram)
{
   /* Apple G17P builtin key0x08021300, text SHA256
    * e641858631bd55fb938f839920beb6b2921a8bbada65bd72b21102daed9651de. */
   static const uint8_t native[] = {
      0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06,
      0x0c, 0x9d, 0x10, 0x26, 0x24, 0xa9, 0x10, 0x26,
      0x9f, 0x10, 0x54, 0x00, 0x00, 0x04, 0x14, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x9f, 0x20, 0x54, 0x02, 0x00, 0x00, 0x10, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x97, 0x05, 0x54, 0x04, 0x00, 0x08, 0x00, 0x00,
      0xd0, 0x40, 0x10, 0x00,
      0x57, 0x10, 0x54, 0x00, 0x00, 0x00, 0x00, 0x80,
      0xa8, 0x75, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
   };

   for (unsigned words = 0; words <= 2; ++words) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_COMPUTE;
      key.g17p_rt0_eot_i32 = true;
      key.g17p_rt0_eot_i32_selector_75 = true;
      key.g17p_rt0_eot_words = words;
      agx_index coords = agx_register(0, AGX_SIZE_32);
      coords.channels_m1 = 1;
      agx_get_sr_to(b, agx_register(0, AGX_SIZE_32),
                    AGX_SR_THREAD_POSITION_IN_GRID_X);
      agx_get_sr_to(b, agx_register(2, AGX_SIZE_32),
                    AGX_SR_THREAD_POSITION_IN_GRID_Y);
      agx_mov_imm_to(b, agx_register(4, AGX_SIZE_32), 0);
      agx_block_image_store(b, agx_immediate(0), agx_immediate(0),
                            agx_register(4, AGX_SIZE_32), coords,
                            AGX_FORMAT_I32, AGX_DIM_2D, true);
      agx_stop(b);

      if (words == 2) {
         EXPECT_EQ(pack(b), std::vector<uint8_t>(std::begin(native),
                                                 std::end(native)));
      } else {
         EXPECT_TRUE(pack(b, false).empty());
      }
   }
}

TEST_F(AGX3PackBinary, G17PI32EOTRefusesConflictingSelectorOverrides)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_COMPUTE;
   key.g17p_rt0_eot_i32 = true;
   key.g17p_rt0_eot_i32_selector_15 = true;
   key.g17p_rt0_eot_i32_selector_75 = true;
   key.g17p_rt0_eot_words = 2;
   agx_index coords = agx_register(0, AGX_SIZE_32);
   coords.channels_m1 = 1;
   agx_get_sr_to(b, agx_register(0, AGX_SIZE_32),
                 AGX_SR_THREAD_POSITION_IN_GRID_X);
   agx_get_sr_to(b, agx_register(2, AGX_SIZE_32),
                 AGX_SR_THREAD_POSITION_IN_GRID_Y);
   agx_mov_imm_to(b, agx_register(4, AGX_SIZE_32), 0);
   agx_block_image_store(b, agx_immediate(0), agx_immediate(0),
                         agx_register(4, AGX_SIZE_32), coords,
                         AGX_FORMAT_I32, AGX_DIM_2D, true);
   agx_stop(b);
   EXPECT_TRUE(pack(b, false).empty());
}

TEST_F(AGX3PackBinary, G17PI32EOTRefusesAmbiguousWordCount)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_COMPUTE;
   key.g17p_rt0_eot_i32 = true;
   agx_index coords = agx_register(0, AGX_SIZE_32);
   coords.channels_m1 = 1;
   agx_get_sr_to(b, agx_register(0, AGX_SIZE_32),
                 AGX_SR_THREAD_POSITION_IN_GRID_X);
   agx_get_sr_to(b, agx_register(2, AGX_SIZE_32),
                 AGX_SR_THREAD_POSITION_IN_GRID_Y);
   agx_mov_imm_to(b, agx_register(4, AGX_SIZE_32), 0);
   agx_block_image_store(b, agx_immediate(0), agx_immediate(0),
                         agx_register(4, AGX_SIZE_32), coords,
                         AGX_FORMAT_I32, AGX_DIM_2D, true);
   agx_stop(b);

   EXPECT_TRUE(pack(b, false).empty());
}

TEST_F(AGX3PackBinary, G17PRGBA16UNORMEOTUsesNativeTwoWordProgram)
{
   key.g17p_rt0_eot_words = 2;

   /* Native builtin EOT key 0x10020c00. The final imgwblk differs from the
    * one-word RGBA8/RGB10/RG16 form in its independently decoded selector. */
   static const uint8_t native[] = {
      0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06,
      0x0c, 0x9d, 0x10, 0x26, 0x24, 0xa9, 0x10, 0x26,
      0x9f, 0x10, 0x54, 0x00, 0x00, 0x04, 0x14, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x9f, 0x20, 0x54, 0x02, 0x00, 0x00, 0x10, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x97, 0x05, 0x54, 0x04, 0x00, 0x08, 0x00, 0x00,
      0xd0, 0x40, 0x10, 0x00,
      0x57, 0x10, 0x54, 0x00, 0x00, 0x00, 0x00, 0x80,
      0xa8, 0x15, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
   };

   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_COMPUTE;
   agx_index coords = agx_register(0, AGX_SIZE_32);
   coords.channels_m1 = 1;
   agx_get_sr_to(b, agx_register(0, AGX_SIZE_32),
                 AGX_SR_THREAD_POSITION_IN_GRID_X);
   agx_get_sr_to(b, agx_register(2, AGX_SIZE_32),
                 AGX_SR_THREAD_POSITION_IN_GRID_Y);
   agx_mov_imm_to(b, agx_register(4, AGX_SIZE_32), 0);
   agx_block_image_store(b, agx_immediate(0), agx_immediate(0),
                         agx_register(4, AGX_SIZE_32), coords,
                         AGX_FORMAT_U16NORM, AGX_DIM_2D, true);
   agx_stop(b);

   EXPECT_EQ(pack(b), std::vector<uint8_t>(std::begin(native),
                                           std::end(native)));
   EXPECT_EQ(b->shader->max_reg, 7);
   EXPECT_TRUE(b->shader->agx3_staging_known);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 8);
}

TEST_F(AGX3PackBinary, G17PRG16SNORMEOTUsesExactOneWordNativeProgram)
{
   /* Fresh Apple G17P direct-imgwblk witnesses for U16NORM and S16NORM are
    * byte-identical, proving the normalized signed pair shares this one-word
    * EOT envelope. PBE carries the signed format. Witness SHA256:
    * 958b7c8e3a8f3747a2207365331f528622a82b9656fe58928f2d94fd4e0d7053. */
   static const uint8_t native[] = {
      0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06,
      0x0c, 0x9d, 0x10, 0x26, 0x24, 0xa9, 0x10, 0x26,
      0x9f, 0x10, 0x54, 0x00, 0x00, 0x04, 0x14, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x9f, 0x20, 0x54, 0x02, 0x00, 0x00, 0x10, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x97, 0x05, 0x54, 0x04, 0x00, 0x08, 0x00, 0x00,
      0xd0, 0x40, 0x10, 0x00,
      0x57, 0x10, 0x54, 0x00, 0x00, 0x00, 0x00, 0x80,
      0xa8, 0x75, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
   };

   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_COMPUTE;
   agx_index coords = agx_register(0, AGX_SIZE_32);
   coords.channels_m1 = 1;
   agx_get_sr_to(b, agx_register(0, AGX_SIZE_32),
                 AGX_SR_THREAD_POSITION_IN_GRID_X);
   agx_get_sr_to(b, agx_register(2, AGX_SIZE_32),
                 AGX_SR_THREAD_POSITION_IN_GRID_Y);
   agx_mov_imm_to(b, agx_register(4, AGX_SIZE_32), 0);
   agx_block_image_store(b, agx_immediate(0), agx_immediate(0),
                         agx_register(4, AGX_SIZE_32), coords,
                         AGX_FORMAT_S16NORM, AGX_DIM_2D, true);
   agx_stop(b);

   EXPECT_EQ(pack(b), std::vector<uint8_t>(std::begin(native),
                                           std::end(native)));
   EXPECT_EQ(b->shader->max_reg, 7);
   EXPECT_TRUE(b->shader->agx3_staging_known);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 8);
}

TEST_F(AGX3PackBinary, G17PU16NormEOTRefusesAmbiguousWordCount)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_COMPUTE;
   agx_index coords = agx_register(0, AGX_SIZE_32);
   coords.channels_m1 = 1;
   agx_get_sr_to(b, agx_register(0, AGX_SIZE_32),
                 AGX_SR_THREAD_POSITION_IN_GRID_X);
   agx_get_sr_to(b, agx_register(2, AGX_SIZE_32),
                 AGX_SR_THREAD_POSITION_IN_GRID_Y);
   agx_mov_imm_to(b, agx_register(4, AGX_SIZE_32), 0);
   agx_block_image_store(b, agx_immediate(0), agx_immediate(0),
                         agx_register(4, AGX_SIZE_32), coords,
                         AGX_FORMAT_U16NORM, AGX_DIM_2D, true);
   agx_stop(b);

   EXPECT_TRUE(pack(b, false).empty());
}

TEST_F(AGX3PackBinary, HalfUVPreservesSharedHalfAndOperandOrder)
{
   for (unsigned order = 0; order < 2; ++order) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index half = agx_register(10, AGX_SIZE_16);
      agx_mov_imm_to(b, half, 0x1800);
      agx_index x = agx_register(0, AGX_SIZE_32);
      agx_index y = agx_register(2, AGX_SIZE_32);
      x.discard = y.discard = true;
      half.cache = true;
      agx_fmul_to(b, agx_register(0, AGX_SIZE_32),
                    order ? half : x, order ? x : half);
      half.cache = false;
      half.discard = true;
      agx_fmul_to(b, agx_register(2, AGX_SIZE_32), y, half);
      agx_stop(b);
      /* Native LDIMM and shared-half MUL forms, independently decoded. */
      const std::vector<uint8_t> want = {
         0x54,0x80,0x02,0x00,0x00,0x0c,0x00,0x00,
         0x09,0x01,0x09,0x8a, 0x19,0x03,0x19,0x0a, 0x0e,0,0,0,
      };
      EXPECT_EQ(pack(b), want);
   }
}

TEST_F(AGX3PackBinary, HalfUVRefusesUngroundedModes)
{
   for (unsigned mode = 0; mode < 9; ++mode) {
      agx_builder *b = builder(mode == 0 ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B);
      agx_index half = agx_register(10, AGX_SIZE_16);
      agx_index wide = agx_register(0, AGX_SIZE_32);
      if (mode == 1) half.neg = true;
      if (mode == 2) half.abs = true;
      if (mode == 3) half.cache = half.discard = true;
      if (mode == 4) half.kill = true;
      if (mode == 5) wide.value = 1;
      if (mode == 6) half.value = 256;
      if (mode == 7) wide.size = AGX_SIZE_16;
      agx_instr *mul = agx_fmul_to(b, agx_register(2, AGX_SIZE_32), wide, half);
      mul->saturate = mode == 8;
      EXPECT_TRUE(pack(b, false).empty());
   }
}

TEST_F(AGX3PackBinary, U16ToF32PreservesHalfLivenessAndLoadWait)
{
   const enum agx_isa_encoding enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   agx_get_sr_to(b, agx_register(0, AGX_SIZE_32), AGX_SR_THREAD_POSITION_IN_GRID_X);
   agx_index even = agx_register(0, AGX_SIZE_16);
   agx_index odd = agx_register(1, AGX_SIZE_16);
   even.discard = false;
   odd.discard = true;
   agx_convert_to(b, agx_register(4, AGX_SIZE_32),
                   agx_immediate(AGX_CONVERT_U16_TO_F), even, AGX_ROUND_RTE);
   agx_convert_to(b, agx_register(6, AGX_SIZE_32),
                   agx_immediate(AGX_CONVERT_U16_TO_F), odd, AGX_ROUND_RTE);
   agx_stop(b);
   struct agx3_sr_desc sr = {};
   sr.dest = 0;
   sr.width = AGX3_SR_W32;
   sr.sr = 23;
   sr.slot = 1;
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   std::vector<uint8_t> want;
   ASSERT_TRUE(agx3_sr_encode(enc, &sr, bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_alu_u16_to_f32(enc, 4, 0, AGX3_ALU_HINT_NONE, 1, bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_alu_u16_to_f32(enc, 6, 1, AGX3_ALU_HINT_DISCARD, 0, bytes, &len));
   append(want, bytes, len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(enc, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, U16ToF32RejectsUnwitnessedDomains)
{
   for (unsigned variant = 0; variant < 6; ++variant) {
      agx_builder *b = builder(variant == 0 ? AGX_ISA_ENCODING_AGX3C
                                           : AGX_ISA_ENCODING_AGX3B);
      agx_index dest = agx_register(4, variant == 1 ? AGX_SIZE_16 : AGX_SIZE_32);
      agx_index src = variant == 2 ? agx_uniform(0, AGX_SIZE_16)
                                   : agx_register(0, AGX_SIZE_16);
      if (variant == 3)
         src.neg = true;
      agx_convert_to(b, dest,
         agx_immediate(variant == 4 ? AGX_CONVERT_S16_TO_F : AGX_CONVERT_U16_TO_F),
         src, variant == 5 ? AGX_ROUND_RTZ : AGX_ROUND_RTE);
      EXPECT_TRUE(pack(b, false).empty());
   }
}

TEST_F(AGX3PackBinary, FaddMinifloatPreservesWaitLivenessAndCommutativity)
{
   const enum agx_isa_encoding enc = AGX_ISA_ENCODING_AGX3B;
   for (unsigned order = 0; order < 2; ++order) {
      agx_builder *b = builder(enc);
      agx_get_sr_to(b, agx_register(0, AGX_SIZE_32),
                    AGX_SR_THREAD_POSITION_IN_GRID_X);
      agx_index live = agx_register(0, AGX_SIZE_32);
      agx_index last = live;
      live.discard = false;
      last.discard = true;
      agx_index imm = agx_immediate(0x20);
      agx_fadd_to(b, agx_register(4, AGX_SIZE_32),
                  order ? imm : live, order ? live : imm);
      agx_fadd_to(b, agx_register(6, AGX_SIZE_32), last,
                  agx_immediate(0x21));
      agx_stop(b);
      struct agx3_sr_desc sr = {};
      sr.dest = 0;
      sr.width = AGX3_SR_W32;
      sr.sr = 23;
      sr.slot = 1;
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      std::vector<uint8_t> want;
      ASSERT_TRUE(agx3_sr_encode(enc, &sr, bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_alu_fadd_minifloat(enc, 4, 0, 0x20, AGX3_ALU_HINT_NONE, 1,
                                         bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_alu_fadd_minifloat(enc, 6, 0, 0x21, AGX3_ALU_HINT_DISCARD, 0,
                                         bytes, &len));
      append(want, bytes, len);
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(enc, &end));
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want);
   }
}

TEST_F(AGX3PackBinary, FaddMinifloatRefusesUngroundedModes)
{
   for (unsigned variant = 0; variant < 10; ++variant) {
      agx_builder *b = builder(variant == 0 ? AGX_ISA_ENCODING_AGX3C
                                           : AGX_ISA_ENCODING_AGX3B);
      agx_index dest = agx_register(4, variant == 1 ? AGX_SIZE_16 : AGX_SIZE_32);
      agx_index src = variant == 2 ? agx_uniform(0, AGX_SIZE_32)
                                   : agx_register(0, AGX_SIZE_32);
      agx_index imm = agx_immediate(variant == 3 ? 256 : 0x20);
      if (variant == 4)
         src.neg = true;
      if (variant == 5)
         src.abs = true;
      if (variant == 6)
         imm.neg = true;
      if (variant == 7)
         src = agx_immediate(0x20);
      if (variant == 8)
         src.size = AGX_SIZE_16;
      agx_instr *add = agx_fadd_to(b, dest, src, imm);
      add->saturate = variant == 9;
      EXPECT_TRUE(pack(b, false).empty());
   }
}

TEST_F(AGX3PackBinary, FaddOneMinusAbsMatchesNativeLiveForm)
{
   const enum agx_isa_encoding enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   agx_index dest = agx_register(2, AGX_SIZE_32);
   dest.cache = true;
   agx_index src = agx_register(0, AGX_SIZE_32);
   src.abs = true;
   src.neg = true;
   src.cache = true;
   agx_fadd_to(b, dest, agx_immediate(0x30), src);
   agx_stop(b);

   std::vector<uint8_t> want = {
      0x19, 0xb1, 0x24, 0x81, 0x82, 0x08, 0x00, 0x00, 0x02, 0x00,
      0x0e, 0x00, 0x00, 0x00,
   };
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, SignedMulHighDiv3MatchesNativeVertexWitness)
{
   const enum agx_isa_encoding enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   agx_get_sr_to(b, agx_register(4, AGX_SIZE_32), AGX_SR_VERTEX_ID);
   agx_index source = agx_register(4, AGX_SIZE_32);
   source.cache = true;
   agx_imul_high_div3_agx3_to(b, agx_register(0, AGX_SIZE_32), source);
   agx_stop(b);

   std::vector<uint8_t> want = {
      0x2c, 0xdd, 0x10, 0x06,
      0x9f, 0x10, 0x54, 0x00, 0x03, 0x08, 0x02, 0x00,
      0xe0, 0x24, 0x16, 0x00,
      0x0e, 0x00, 0x00, 0x00,
   };
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, FabsMaterializationKeepsNegativeZeroAndWait)
{
   const enum agx_isa_encoding enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   agx_index source = agx_register(2, AGX_SIZE_32);
   source.cache = true;
   source.abs = true;
   agx_index dest = agx_register(8, AGX_SIZE_32);
   dest.cache = true;
   agx_fadd_to(b, dest, source, agx_negzero());
   agx_stop(b);

   std::vector<uint8_t> want = {
      0x49, 0x81, 0x2c, 0x83, 0x82, 0x00, 0x00, 0x00,
      0x02, 0x00, 0x0e, 0x00, 0x00, 0x00,
   };
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, WboitInfinityRepairUsesMixedWidthFsel)
{
   const enum agx_isa_encoding enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   agx_index compare = agx_register(8, AGX_SIZE_32);
   agx_index infinity = agx_register(10, AGX_SIZE_16);
   agx_index if_true = agx_register(6, AGX_SIZE_32);
   agx_index if_false = agx_register(4, AGX_SIZE_32);
   compare.cache = infinity.cache = if_true.cache = true;
   if_false.discard = true;
   agx_index dest = agx_register(4, AGX_SIZE_32);
   dest.cache = true;
   agx_fcmpsel_to(b, dest, compare, infinity, if_true, if_false,
                  AGX_FCOND_EQ);
   agx_stop(b);

   std::vector<uint8_t> want = {
      0x22, 0x89, 0x27, 0x8a, 0x05, 0x86, 0x00, 0x00,
      0x0e, 0x00, 0x00, 0x00,
   };
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, UpdateLightmapMixedWidthLtFsel)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index compare = agx_register(46, AGX_SIZE_32);
   agx_index threshold = agx_register(52, AGX_SIZE_16);
   agx_index if_true = agx_register(26, AGX_SIZE_32);
   agx_index if_false = agx_register(36, AGX_SIZE_32);
   compare.cache = threshold.cache = if_true.cache = true;
   agx_index dest = agx_register(36, AGX_SIZE_32);
   dest.cache = true;
   agx_fcmpsel_to(b, dest, compare, threshold, if_true, if_false,
                  AGX_FCOND_LT);
   agx_stop(b);

   const uint8_t native[] = {
      0x22, 0xaf, 0x67, 0xb4, 0x02, 0x9a, 0x03, 0x00, 0x00, 0x24,
      0x0e, 0x00, 0x00, 0x00,
   };
   EXPECT_EQ(pack(b), std::vector<uint8_t>(native, native + sizeof(native)));
}

TEST_F(AGX3PackBinary, AbsoluteCompareFselUsesNativeLongForm)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index a = agx_register(34, AGX_SIZE_32);
   agx_index c = agx_register(20, AGX_SIZE_32);
   agx_index selected = agx_register(24, AGX_SIZE_32);
   agx_index dest = agx_register(26, AGX_SIZE_32);
   a.abs = c.abs = true;
   a.cache = c.cache = selected.cache = dest.cache = true;
   agx_fcmpsel_to(b, dest, a, c, selected, agx_immediate(0), AGX_FCOND_EQ);
   agx_stop(b);

   const uint16_t src[] = {34, 20, 24, 0};
   const bool compare_abs[] = {true, true};
   const bool immediate[] = {false, true};
   const enum agx3_alu_hint hints[] = {
      AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_CACHE,
      AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_NONE};
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   ASSERT_TRUE(agx3_alu_fcmp_select_reg_imm(
      AGX_ISA_ENCODING_AGX3B, 26, src, false, compare_abs, immediate,
      hints, AGX_FCOND_EQ, false, true, 0, bytes, &len));
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   std::vector<uint8_t> want;
   append(want, bytes, len);
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, UpdateLightmapNativeIntegerAndMaskForms)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);

   agx_index r13 = agx_register(26, AGX_SIZE_32);
   r13.cache = true;
   agx_index r12 = agx_register(24, AGX_SIZE_32);
   r12.cache = true;
   agx_ffs_to(b, r13, r12);

   agx_index r10 = agx_register(20, AGX_SIZE_32);
   agx_index r3 = agx_register(6, AGX_SIZE_32);
   r10.cache = r3.cache = true;
   agx_bitop_to(b, agx_register(20, AGX_SIZE_32), r10, r3, 0x2);

   agx_index half = agx_register(6, AGX_SIZE_16);
   agx_index half_dest = agx_register(6, AGX_SIZE_16);
   half_dest.cache = true;
   agx_fcmpsel_to(b, half_dest, agx_immediate(176),
                  agx_register(30, AGX_SIZE_32), half, agx_immediate(0),
                  AGX_FCOND_LT);
   agx_stop(b);

   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   std::vector<uint8_t> want;
   ASSERT_TRUE(agx3_alu_ft1_u32(AGX_ISA_ENCODING_AGX3B, 26, 24,
                                 AGX3_ALU_HINT_CACHE, true, 1, bytes, &len));
   append(want, bytes, len);
   const uint16_t bit_src[] = {20, 6};
   const enum agx3_alu_hint bit_hints[] = {
      AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_CACHE};
   ASSERT_TRUE(agx3_alu_bit_andc(AGX_ISA_ENCODING_AGX3B, 20, bit_src,
                                  bit_hints, false, 0, bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_alu_fcmp_minifloat_select_half(
      AGX_ISA_ENCODING_AGX3B, 6, 30, 176, 6, 0,
      AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE, AGX_FCOND_GT, true, 0,
      bytes, &len));
   append(want, bytes, len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, FmulMinifloatPreservesWaitLivenessAndCommutativity)
{
   const enum agx_isa_encoding enc = AGX_ISA_ENCODING_AGX3B;
   for (unsigned order = 0; order < 2; ++order) {
      agx_builder *b = builder(enc);
      agx_get_sr_to(b, agx_register(0, AGX_SIZE_32),
                    AGX_SR_THREAD_POSITION_IN_GRID_X);
      agx_index live = agx_register(0, AGX_SIZE_32);
      agx_index last = live;
      live.discard = false;
      last.discard = true;
      agx_index imm = agx_immediate(0x02);
      agx_fmul_to(b, agx_register(4, AGX_SIZE_32),
                  order ? imm : live, order ? live : imm);
      agx_fmul_to(b, agx_register(6, AGX_SIZE_32), last,
                  agx_immediate(0x03));
      agx_stop(b);
      struct agx3_sr_desc sr = {};
      sr.dest = 0;
      sr.width = AGX3_SR_W32;
      sr.sr = 23;
      sr.slot = 1;
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      std::vector<uint8_t> want;
      ASSERT_TRUE(agx3_sr_encode(enc, &sr, bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_alu_fmul_minifloat(enc, 4, 0, 0x02, AGX3_ALU_HINT_NONE, 1,
                                         bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_alu_fmul_minifloat(enc, 6, 0, 0x03, AGX3_ALU_HINT_DISCARD, 0,
                                         bytes, &len));
      append(want, bytes, len);
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(enc, &end));
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want);
   }
}

TEST_F(AGX3PackBinary, FmulMinifloatRefusesUngroundedModes)
{
   for (unsigned variant = 0; variant < 10; ++variant) {
      agx_builder *b = builder(variant == 0 ? AGX_ISA_ENCODING_AGX3C
                                           : AGX_ISA_ENCODING_AGX3B);
      agx_index dest = agx_register(4, variant == 1 ? AGX_SIZE_16 : AGX_SIZE_32);
      agx_index src = variant == 2 ? agx_uniform(0, AGX_SIZE_32)
                                   : agx_register(0, AGX_SIZE_32);
      agx_index imm = agx_immediate(variant == 3 ? 256 : 0x02);
      if (variant == 4)
         src.neg = true;
      if (variant == 5)
         src.abs = true;
      if (variant == 6)
         imm.neg = true;
      if (variant == 7)
         src = agx_immediate(0x02);
      if (variant == 8)
         src.size = AGX_SIZE_16;
      agx_instr *mul = agx_fmul_to(b, dest, src, imm);
      mul->saturate = variant == 9;
      EXPECT_TRUE(pack(b, false).empty());
   }
}

TEST_F(AGX3PackBinary, DeviceAtomicsUseGenerationSpecificReturnRegisters)
{
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      const agx_index value = agx_register(6, AGX_SIZE_32);
      const agx_index result =
         agx_register(e ? 4 : 6, AGX_SIZE_32); /* b is physically in-place */
      agx_atomic_to(b, result, value, agx_uniform(0, AGX_SIZE_64),
                    agx_immediate(0), AGX_ATOMIC_OPC_XOR);
      agx_stop(b);

      struct agx3_device_atomic_desc d = {
         .op = AGX3_ATOMIC_XOR,
         .value = 6,
         .returns_value = true,
         .result = (uint16_t)(e ? 4 : 6),
         .slot = 1,
         .base_kind = AGX3_ATOMIC_BASE_UNIFORM,
         .index_kind = AGX3_ATOMIC_INDEX_IMMEDIATE,
      };
      uint8_t atomic[AGX3_ATOMIC_MAX_LENGTH], atomic_len = 0;
      ASSERT_TRUE(agx3_atomic_encode_device(enc, &d, atomic, &atomic_len));
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(enc, &end));

      std::vector<uint8_t> want;
      append(want, atomic, atomic_len);
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want);
   }
}

TEST_F(AGX3PackBinary, DeviceAtomicUsesNativeMaskedExecutionMode)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->cursor.block->divergent = true;
   agx_atomic_to(b, agx_register(6, AGX_SIZE_32),
                 agx_register(6, AGX_SIZE_32),
                 agx_uniform(0, AGX_SIZE_64), agx_immediate(0),
                 AGX_ATOMIC_OPC_ADD);
   agx_stop(b);

   struct agx3_device_atomic_desc atomic = {
      .op = AGX3_ATOMIC_ADD,
      .value = 6,
      .exec_masked = true,
      .returns_value = true,
      .result = 6,
      .slot = 1,
      .base_kind = AGX3_ATOMIC_BASE_UNIFORM,
      .index_kind = AGX3_ATOMIC_INDEX_IMMEDIATE,
   };
   uint8_t bytes[AGX3_ATOMIC_MAX_LENGTH], atomic_len;
   ASSERT_TRUE(agx3_atomic_encode_device(AGX_ISA_ENCODING_AGX3B, &atomic,
                                         bytes, &atomic_len));
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));

   std::vector<uint8_t> want;
   append(want, bytes, atomic_len);
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
   ASSERT_GE(want.size(), 2u);
   EXPECT_EQ(want[1], 0x11);
}

TEST_F(AGX3PackBinary, DeviceAtomicInPlaceReturnDiscardsConsumedValue)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_VERTEX;
   b->shader->nir = rzalloc(mem_ctx, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_VERTEX;

   agx_index value = agx_mov_imm(b, 32, 7);
   agx_index result = agx_temp(b->shader, AGX_SIZE_32);
   agx_atomic_to(b, result, value, agx_uniform(0, AGX_SIZE_64),
                 agx_immediate(0), AGX_ATOMIC_OPC_ADD);
   agx_stop(b);

   agx_ra(b->shader);

   unsigned atomics = 0;
   agx_foreach_instr_global(b->shader, I) {
      if (I->op != AGX_OPCODE_ATOMIC)
         continue;

      ++atomics;
      EXPECT_EQ(I->src[0].type, AGX_INDEX_REGISTER);
      EXPECT_EQ(I->dest[0].type, AGX_INDEX_REGISTER);
      EXPECT_EQ(I->src[0].value, I->dest[0].value);
      EXPECT_TRUE(I->src[0].discard);
   }
   EXPECT_EQ(atomics, 1u);
}

TEST_F(AGX3PackBinary, RegisterSubtractDestinationDoesNotAliasInputs)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_VERTEX;
   b->shader->nir = rzalloc(mem_ctx, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_VERTEX;

   agx_index lhs = agx_mov_imm(b, 32, 0x77777777);
   agx_index rhs = agx_mov_imm(b, 32, 0x11111111);
   agx_index result = agx_temp(b->shader, AGX_SIZE_32);
   agx_iadd_to(b, result, lhs, agx_neg(rhs), 0);
   agx_st_vary(b, agx_immediate(0), result);
   agx_stop(b);

   agx_ra(b->shader);

   unsigned subtracts = 0;
   agx_foreach_instr_global(b->shader, I) {
      if (I->op != AGX_OPCODE_IADD || I->nr_srcs != 2 || !I->src[1].neg)
         continue;

      ++subtracts;
      EXPECT_NE(I->dest[0].value, I->src[0].value);
      EXPECT_NE(I->dest[0].value, I->src[1].value);
   }
   EXPECT_EQ(subtracts, 1u);
}

TEST_F(AGX3PackBinary, RegisterAndDestinationDoesNotAliasInputs)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_VERTEX;
   b->shader->nir = rzalloc(mem_ctx, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_VERTEX;

   agx_index lhs = agx_mov_imm(b, 32, 0x77777777);
   agx_index rhs = agx_mov_imm(b, 32, 0x55555555);
   agx_index result = agx_and(b, lhs, rhs);
   agx_st_vary(b, agx_immediate(0), result);
   agx_stop(b);

   agx_ra(b->shader);

   unsigned ands = 0;
   agx_foreach_instr_global(b->shader, I) {
      if (I->op != AGX_OPCODE_BITOP ||
          (I->truth_table & 0xfu) != 0x8u)
         continue;

      ++ands;
      EXPECT_NE(I->dest[0].value, I->src[0].value);
      EXPECT_NE(I->dest[0].value, I->src[1].value);
   }
   EXPECT_EQ(ands, 1u);
}

TEST_F(AGX3PackBinary, DeviceAtomicFoldsElementOffsetToBytes)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3C);
   agx_atomic_to(b, agx_register(4, AGX_SIZE_32),
                 agx_register(6, AGX_SIZE_32),
                 agx_uniform(4, AGX_SIZE_64), agx_immediate(7),
                 AGX_ATOMIC_OPC_ADD);
   agx_stop(b);

   struct agx3_device_atomic_desc d = {
      .op = AGX3_ATOMIC_ADD,
      .value = 6,
      .returns_value = true,
      .result = 4,
      .slot = 1,
      .base_kind = AGX3_ATOMIC_BASE_UNIFORM,
      .base = 4,
      .index_kind = AGX3_ATOMIC_INDEX_IMMEDIATE,
      .byte_offset = 28,
   };
   uint8_t atomic[AGX3_ATOMIC_MAX_LENGTH], atomic_len = 0;
   ASSERT_TRUE(agx3_atomic_encode_device(AGX_ISA_ENCODING_AGX3C, &d, atomic,
                                         &atomic_len));
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3C, &end));

   std::vector<uint8_t> want;
   append(want, atomic, atomic_len);
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, DeviceAtomicMapsPostRaDiscardHints)
{
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);

      agx_index value = agx_register(6, AGX_SIZE_32);
      value.discard = true;
      agx_index base = agx_register(8, AGX_SIZE_64);
      base.discard = true; /* no atomicdev base-hint field: safely omitted */
      agx_index index = agx_register(12, AGX_SIZE_32);
      index.discard = true;
      const agx_index result = agx_register(e ? 4 : 6, AGX_SIZE_32);

      agx_atomic_to(b, result, value, base, index, AGX_ATOMIC_OPC_ADD);
      agx_stop(b);

      struct agx3_device_atomic_desc d = {
         .op = AGX3_ATOMIC_ADD,
         .value = 6,
         .value_kill = true,
         .returns_value = true,
         .result = (uint16_t)(e ? 4 : 6),
         .slot = 1,
         .base_kind = AGX3_ATOMIC_BASE_REGISTER,
         .base = 8,
         .index_kind = AGX3_ATOMIC_INDEX_REGISTER,
         .index = 12,
         .index_kill = true,
         .index_shift = 2,
      };
      uint8_t atomic[AGX3_ATOMIC_MAX_LENGTH], atomic_len = 0;
      ASSERT_TRUE(agx3_atomic_encode_device(enc, &d, atomic, &atomic_len));
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(enc, &end));

      std::vector<uint8_t> want;
      append(want, atomic, atomic_len);
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want);
   }
}

TEST_F(AGX3PackBinary, DeviceAtomicReturnIsTiedWithoutClobberingIndex)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_COMPUTE;
   b->shader->nir = rzalloc(mem_ctx, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_COMPUTE;

   agx_index value = agx_temp(b->shader, AGX_SIZE_32);
   agx_mov_imm_to(b, value, 0xfffffffa);
   agx_index index = agx_temp(b->shader, AGX_SIZE_32);
   agx_mov_imm_to(b, index, 7);
   agx_index result = agx_temp(b->shader, AGX_SIZE_32);
   agx_atomic_to(b, result, value, agx_uniform(0, AGX_SIZE_64), index,
                 AGX_ATOMIC_OPC_ADD);

   /* Keep the address index live after the atomic.  A pack-time value copy
    * into the result would corrupt it; the pre-RA two-address legalization
    * must instead tie only the dedicated update temporary and result. */
   agx_index sum = agx_temp(b->shader, AGX_SIZE_32);
   agx_iadd_to(b, sum, result, index, 0);
   agx_stop(b);

   ASSERT_TRUE(agx_lower_agx3_device_atomic_sources(b->shader));
   agx_pressure_schedule(b->shader);
   agx_ra(b->shader);

   agx_instr *atomic = NULL;
   agx_foreach_instr_global(b->shader, I) {
      if (I->op == AGX_OPCODE_ATOMIC)
         atomic = I;
   }
   ASSERT_NE(atomic, nullptr);
   ASSERT_EQ(atomic->dest[0].type, AGX_INDEX_REGISTER);
   ASSERT_EQ(atomic->src[0].type, AGX_INDEX_REGISTER);
   EXPECT_EQ(atomic->dest[0].value, atomic->src[0].value);
   EXPECT_NE(atomic->dest[0].value, atomic->src[2].value);
   EXPECT_FALSE(pack(b).empty());
}

TEST_F(AGX3PackBinary, LocalAtomicScalarZeroAddress)
{
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      agx_local_atomic_to(b, agx_register(6, AGX_SIZE_32),
                          agx_register(4, AGX_SIZE_32), agx_immediate(0),
                          agx_immediate(0), AGX_ATOMIC_OPC_UMIN);
      agx_stop(b);

      const struct agx3_local_atomic_desc d = {
         .op = AGX3_ATOMIC_UMIN,
         .value = 4,
         .returns_value = true,
         .result = 6,
         .slot = 1,
         .address_kind = AGX3_ATOMIC_INDEX_IMMEDIATE,
      };
      uint8_t atomic[AGX3_ATOMIC_MAX_LENGTH], atomic_len = 0;
      ASSERT_TRUE(agx3_atomic_encode_local(enc, &d, atomic, &atomic_len));
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(enc, &end));

      std::vector<uint8_t> want;
      append(want, atomic, atomic_len);
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want);
   }
}

TEST_F(AGX3PackBinary, NarrowingMoveUsesLowHalfCopy)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_bitop_to(b, agx_register(1, AGX_SIZE_16),
                agx_register(8, AGX_SIZE_32), agx_zero(), UTIL_LUT2(a));
   agx_stop(b);

   uint8_t copy[AGX3_COPY_MAX_LENGTH], copy_len = 0;
   ASSERT_TRUE(agx3_copy_encode_reg16(AGX_ISA_ENCODING_AGX3B, 1, 8,
                                      false, false, false, 0,
                                      copy, &copy_len));
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));

   std::vector<uint8_t> want;
   append(want, copy, copy_len);
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, AtomicsRefuseUngroundedStructuralAxes)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_atomic_to(b, agx_register(4, AGX_SIZE_32),
                 agx_register(6, AGX_SIZE_32),
                 agx_uniform(0, AGX_SIZE_64), agx_immediate(0),
                 AGX_ATOMIC_OPC_ADD);
   agx_stop(b);
   pack(b, /*expect_ok=*/false); /* AGX3b return was not RA-coalesced */

   b = builder(AGX_ISA_ENCODING_AGX3C);
   agx_local_atomic_to(b, agx_register(6, AGX_SIZE_32),
                       agx_register(4, AGX_SIZE_32),
                       agx_register(8, AGX_SIZE_16), agx_immediate(0),
                       AGX_ATOMIC_OPC_ADD);
   agx_stop(b);
   pack(b, /*expect_ok=*/false); /* dynamic returned local address not isolated */
}

TEST_F(AGX3PackBinary, LocalAtomicVoidAddressedG17P)
{
   for (int address : {-8, 8, 1}) {
      const bool dynamic = address >= 0;
      const unsigned half = dynamic ? (unsigned)address : 0;
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_local_atomic_to(b, agx_null(), agx_register(4, AGX_SIZE_32),
                          dynamic ? agx_register(half, AGX_SIZE_16)
                                  : agx_immediate(8),
                          agx_immediate(0), AGX_ATOMIC_OPC_OR);
      agx_stop(b);

      const struct agx3_local_atomic_desc d = {
         .op = AGX3_ATOMIC_OR,
         .value = 4,
         .address_kind = dynamic ? AGX3_ATOMIC_INDEX_REGISTER
                                 : AGX3_ATOMIC_INDEX_IMMEDIATE,
         .address = static_cast<uint16_t>(dynamic ? half : 8),
      };
      uint8_t atomic[AGX3_ATOMIC_MAX_LENGTH], atomic_len = 0;
      ASSERT_TRUE(agx3_atomic_encode_local(AGX_ISA_ENCODING_AGX3B, &d,
                                           atomic, &atomic_len));
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));

      std::vector<uint8_t> want;
      append(want, atomic, atomic_len);
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want);
   }
}

TEST_F(AGX3PackBinary, SimdBroadcastImmediateWitnesses)
{
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      for (uint8_t lane : {0, 3}) {
         agx_builder *b = builder(enc);
         agx_shuffle_to(b, agx_register(4, AGX_SIZE_32),
                        agx_register(0, AGX_SIZE_32), agx_immediate(lane));
         agx_stop(b);

         uint8_t broadcast[AGX3_SUBGROUP_MAX_LENGTH];
         uint8_t broadcast_length = 0;
         if (enc == AGX_ISA_ENCODING_AGX3B) {
            const struct agx3_subgroup_shuffle_desc desc = {
               .op = AGX3_SUBGROUP_SHUFFLE_BROADCAST,
               .lane_kind = AGX3_SUBGROUP_LANE_IMMEDIATE,
               .dest = 4,
               .src = 0,
               .lane = lane,
            };
            ASSERT_TRUE(agx3_subgroup_encode_shuffle(
               enc, &desc, broadcast, &broadcast_length));
         } else {
            struct agx3_encoded_simd_broadcast_f32_imm encoded;
            ASSERT_TRUE(agx3_pack_simd_broadcast_f32_imm(enc, 4, 0, lane,
                                                          &encoded));
            memcpy(broadcast, encoded.bytes, encoded.length);
            broadcast_length = encoded.length;
         }
         struct agx3_encoded_end end;
         ASSERT_TRUE(agx3_pack_end(enc, &end));

         std::vector<uint8_t> want;
         append(want, broadcast, broadcast_length);
         append(want, end.bytes, end.length);
         EXPECT_EQ(pack(b), want);
      }
   }
}

TEST_F(AGX3PackBinary, SimdBroadcastRefusesUnwitnessedLane)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_shuffle_to(b, agx_register(4, AGX_SIZE_32),
                  agx_register(0, AGX_SIZE_32), agx_immediate(1));
   agx_stop(b);
   pack(b, /*expect_ok=*/false);
}

TEST_F(AGX3PackBinary, SimdBroadcastWaitsForDeviceLoad)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   const agx_index value = agx_register(4, AGX_SIZE_32);
   agx_device_load_to(b, value, agx_register(8, AGX_SIZE_64),
                      agx_register(12, AGX_SIZE_32), AGX_FORMAT_I32, 1, 2,
                      false);
   agx_shuffle_to(b, agx_register(6, AGX_SIZE_32), value, agx_immediate(3));
   agx_stop(b);

   uint8_t bytes[AGX3_SUBGROUP_MAX_LENGTH], length = 0;
   std::vector<uint8_t> want;

   const struct agx3_subgroup_shuffle_desc shuffle = {
      .op = AGX3_SUBGROUP_SHUFFLE_BROADCAST,
      .lane_kind = AGX3_SUBGROUP_LANE_IMMEDIATE,
      .dest = 6,
      .src = 4,
      .lane = 3,
      /* G17P device loads use the native preferred slot 6. */
      .wait_mask = 1u << 5,
   };
   ASSERT_TRUE(agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &shuffle,
                                             bytes, &length));
   append(want, bytes, length);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);
   const std::vector<uint8_t> got = pack(b);
   ASSERT_GE(got.size(), want.size());
   const std::vector<uint8_t> tail(got.end() - want.size(), got.end());
   EXPECT_EQ(tail, want);
}

TEST_F(AGX3PackBinary, QuadImmediateNativeWitnesses)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   const agx_index value = agx_register(6, AGX_SIZE_32);
   agx_quad_shuffle_to(b, agx_register(0, AGX_SIZE_32), value,
                       agx_immediate(2));
   agx_quad_shuffle_xor_to(b, agx_register(2, AGX_SIZE_32), value,
                           agx_immediate(1));
   agx_quad_shuffle_xor_to(b, agx_register(4, AGX_SIZE_32), value,
                           agx_immediate(2));
   agx_index final_value = value;
   final_value.discard = true;
   agx_quad_shuffle_xor_to(b, agx_register(6, AGX_SIZE_32), final_value,
                           agx_immediate(3));
   agx_stop(b);

   const uint8_t native[] = {
      0x47, 0x00, 0x54, 0x00, 0x02, 0x0c, 0x04, 0x2c, 0x02, 0x00,
      0xc7, 0x00, 0x54, 0x02, 0x02, 0x0c, 0x02, 0x2c, 0x02, 0x00,
      0xc7, 0x00, 0x54, 0x04, 0x02, 0x0c, 0x04, 0x2c, 0x02, 0x00,
      0xc7, 0x00, 0x54, 0x06, 0x02, 0x0c, 0x06, 0x2c, 0x04, 0x00,
   };
   std::vector<uint8_t> want(native, native + sizeof(native));
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, QuadImmediateXorRetiresDeviceLoad)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   const agx_index value = agx_register(4, AGX_SIZE_32);
   agx_device_load_to(b, value, agx_register(8, AGX_SIZE_64),
                      agx_register(12, AGX_SIZE_32), AGX_FORMAT_I32, 1, 2,
                      false);
   agx_quad_shuffle_xor_to(b, agx_register(6, AGX_SIZE_32), value,
                           agx_immediate(1));
   agx_stop(b);

   std::vector<uint8_t> want;
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], length = 0;
   ASSERT_TRUE(agx3_alu_compose(AGX_ISA_ENCODING_AGX3B, AGX3_ALU_BAND, 0, 0,
                                0, 0, 0, 1u << 5, bytes, &length));
   append(want, bytes, length);
   const struct agx3_subgroup_shuffle_desc shuffle = {
      .quad = true,
      .op = AGX3_SUBGROUP_SHUFFLE_XOR,
      .lane_kind = AGX3_SUBGROUP_LANE_IMMEDIATE,
      .dest = 6,
      .src = 4,
      .lane = 1,
      .src_live = true,
   };
   ASSERT_TRUE(agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &shuffle,
                                             bytes, &length));
   append(want, bytes, length);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);

   const std::vector<uint8_t> got = pack(b);
   ASSERT_GE(got.size(), want.size());
   const std::vector<uint8_t> tail(got.end() - want.size(), got.end());
   EXPECT_EQ(tail, want);
}

TEST_F(AGX3PackBinary, SignedReduceWaitsForDeviceLoad)
{
   for (auto op : {AGX3_SUBGROUP_REDUCE_SMIN,
                   AGX3_SUBGROUP_REDUCE_SMAX}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      const agx_index value = agx_register(4, AGX_SIZE_32);
      agx_device_load_to(b, value, agx_register(8, AGX_SIZE_64),
                         agx_register(12, AGX_SIZE_32), AGX_FORMAT_I32, 1, 2,
                         false);
      agx_simd_reduce_to(b, agx_register(6, AGX_SIZE_32), value,
                         op == AGX3_SUBGROUP_REDUCE_SMIN ? AGX_SIMD_OP_SMIN
                                                        : AGX_SIMD_OP_SMAX);
      agx_stop(b);

      uint8_t bytes[AGX3_SUBGROUP_MAX_LENGTH], length = 0;
      std::vector<uint8_t> want;
      const struct agx3_subgroup_reduce_desc reduce = {
         .op = op,
         .dest = 6,
         .src = 4,
         /* G17P device loads use the native preferred slot 6. */
         .wait_mask = 1u << 5,
      };
      ASSERT_TRUE(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B,
                                               &reduce, bytes, &length));
      append(want, bytes, length);
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
      append(want, end.bytes, end.length);
      const std::vector<uint8_t> got = pack(b);
      ASSERT_GE(got.size(), want.size());
      const std::vector<uint8_t> tail(got.end() - want.size(), got.end());
      EXPECT_EQ(tail, want);
   }
}

TEST_F(AGX3PackBinary, SignedAddPrefixWaitsForDeviceLoad)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   const agx_index value = agx_register(4, AGX_SIZE_32);
   agx_device_load_to(b, value, agx_register(8, AGX_SIZE_64),
                      agx_register(12, AGX_SIZE_32), AGX_FORMAT_I32, 1, 2,
                      false);
   agx_simd_prefix_to(b, agx_register(6, AGX_SIZE_32), value,
                      AGX_SIMD_OP_IADD);
   agx_stop(b);

   uint8_t bytes[AGX3_SUBGROUP_MAX_LENGTH], length = 0;
   std::vector<uint8_t> want;
   const struct agx3_subgroup_reduce_desc prefix = {
      .prefix = true,
      .op = AGX3_SUBGROUP_REDUCE_IADD,
      .dest = 6,
      .src = 4,
      /* G17P device loads use the native preferred slot 6. */
      .wait_mask = 1u << 5,
   };
   ASSERT_TRUE(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &prefix,
                                             bytes, &length));
   append(want, bytes, length);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);

   const std::vector<uint8_t> got = pack(b);
   ASSERT_GE(got.size(), want.size());
   const std::vector<uint8_t> tail(got.end() - want.size(), got.end());
   EXPECT_EQ(tail, want);
}

static void
build_shuffle(agx_builder *b, bool quad, unsigned op, agx_index dest,
              agx_index src, agx_index lane)
{
   if (!quad) {
      switch (op) {
      case AGX3_SUBGROUP_SHUFFLE_BROADCAST:
         agx_shuffle_to(b, dest, src, lane);
         break;
      case AGX3_SUBGROUP_SHUFFLE_XOR:
         agx_shuffle_xor_to(b, dest, src, lane);
         break;
      case AGX3_SUBGROUP_SHUFFLE_UP:
         agx_shuffle_up_to(b, dest, src, lane);
         break;
      case AGX3_SUBGROUP_SHUFFLE_DOWN:
         agx_shuffle_down_to(b, dest, src, lane);
         break;
      default:
         UNREACHABLE("bad shuffle");
      }
   } else {
      switch (op) {
      case AGX3_SUBGROUP_SHUFFLE_BROADCAST:
         agx_quad_shuffle_to(b, dest, src, lane);
         break;
      case AGX3_SUBGROUP_SHUFFLE_XOR:
         agx_quad_shuffle_xor_to(b, dest, src, lane);
         break;
      case AGX3_SUBGROUP_SHUFFLE_UP:
         agx_quad_shuffle_up_to(b, dest, src, lane);
         break;
      case AGX3_SUBGROUP_SHUFFLE_DOWN:
         agx_quad_shuffle_down_to(b, dest, src, lane);
         break;
      default:
         UNREACHABLE("bad shuffle");
      }
   }
}

TEST_F(AGX3PackBinary, RegisterAndUniformSubgroupShuffles)
{
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      for (unsigned quad = 0; quad < 2; ++quad) {
         for (unsigned op = 0; op < AGX3_SUBGROUP_SHUFFLE_NUM; ++op) {
            for (unsigned uniform = 0; uniform < 2; ++uniform) {
               agx_builder *b = builder(enc);
               const agx_index lane = uniform ? agx_uniform(8, AGX_SIZE_32)
                                              : agx_register(8, AGX_SIZE_32);
               build_shuffle(b, quad, op, agx_register(4, AGX_SIZE_32),
                             agx_register(2, AGX_SIZE_32), lane);
               agx_stop(b);

               const struct agx3_subgroup_shuffle_desc d = {
                  .quad = (bool)quad,
                  .op = (enum agx3_subgroup_shuffle_op)op,
                  .lane_kind = uniform ? AGX3_SUBGROUP_LANE_UNIFORM
                                       : AGX3_SUBGROUP_LANE_REGISTER,
                  .dest = 4,
                  .src = 2,
                  .lane = 8,
               };
               uint8_t subgroup[AGX3_SUBGROUP_MAX_LENGTH], subgroup_len = 0;
               ASSERT_TRUE(agx3_subgroup_encode_shuffle(enc, &d, subgroup,
                                                         &subgroup_len));
               struct agx3_encoded_end end;
               ASSERT_TRUE(agx3_pack_end(enc, &end));
               std::vector<uint8_t> want;
               append(want, subgroup, subgroup_len);
               append(want, end.bytes, end.length);
               EXPECT_EQ(pack(b), want);
            }
         }
      }
   }
}

TEST_F(AGX3PackBinary, RegisterSubgroupShufflesWaitForDeviceLoad)
{
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      for (unsigned op = 0; op < AGX3_SUBGROUP_SHUFFLE_NUM; ++op) {
         agx_builder *b = builder(enc);
         const agx_index value = agx_register(4, AGX_SIZE_32);
         agx_device_load_to(b, value, agx_register(10, AGX_SIZE_64),
                            agx_register(14, AGX_SIZE_32), AGX_FORMAT_I32, 1,
                            2, false);
         build_shuffle(b, false, op, agx_register(6, AGX_SIZE_32), value,
                       agx_register(8, AGX_SIZE_32));
         agx_stop(b);

         std::vector<uint8_t> want;
         uint8_t bytes[AGX3_ALU_MAX_LENGTH], length = 0;
         /* The slot allocator selects slot6 for this AGX3b register-lane
          * graph and slot1 for the corresponding AGX3c graph. */
         const uint16_t wait =
            enc == AGX_ISA_ENCODING_AGX3B ? 1u << 5 : 1u;
         ASSERT_TRUE(agx3_alu_compose(enc, AGX3_ALU_BAND, 0, 0, 0, 0, 0,
                                      wait, bytes, &length));
         append(want, bytes, length);

         const struct agx3_subgroup_shuffle_desc shuffle = {
            .op = (enum agx3_subgroup_shuffle_op)op,
            .lane_kind = AGX3_SUBGROUP_LANE_REGISTER,
            .dest = 6,
            .src = 4,
            .lane = 8,
         };
         ASSERT_TRUE(agx3_subgroup_encode_shuffle(enc, &shuffle, bytes,
                                                   &length));
         append(want, bytes, length);
         struct agx3_encoded_end end;
         ASSERT_TRUE(agx3_pack_end(enc, &end));
         append(want, end.bytes, end.length);

         const std::vector<uint8_t> got = pack(b);
         ASSERT_GE(got.size(), want.size());
         const std::vector<uint8_t> tail(got.end() - want.size(), got.end());
         EXPECT_EQ(tail, want);
      }
   }
}

TEST_F(AGX3PackBinary, BallotCompareZero)
{
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      for (unsigned quad = 0; quad < 2; ++quad) {
         for (unsigned floating = 0; floating < 2; ++floating) {
            agx_builder *b = builder(enc);
            agx_index source = agx_register(2, AGX_SIZE_32);
            source.discard = true;
            if (floating) {
               if (quad)
                  agx_fcmp_quad_ballot_to(
                     b, agx_register(4, AGX_SIZE_32),
                     source, agx_zero(), AGX_FCOND_LT,
                     false);
               else
                  agx_fcmp_ballot_to(b, agx_register(4, AGX_SIZE_32),
                                     source, agx_zero(),
                                     AGX_FCOND_LT, false);
            } else {
               if (quad)
                  agx_icmp_quad_ballot_to(
                     b, agx_register(4, AGX_SIZE_32),
                     source, agx_zero(), AGX_ICOND_UEQ,
                     true);
               else
                  agx_icmp_ballot_to(b, agx_register(4, AGX_SIZE_32),
                                     source, agx_zero(),
                                     AGX_ICOND_UEQ, true);
            }
            agx_stop(b);

            const struct agx3_subgroup_ballot_desc d = {
               .quad = (bool)quad,
               .type = floating ? AGX3_SUBGROUP_BALLOT_FLOAT
                                : AGX3_SUBGROUP_BALLOT_INTEGER,
               .dest = 4,
               .src = 2,
               .condition = (uint8_t)(floating ? 1 : 4),
               .invert = !floating,
            };
            uint8_t subgroup[AGX3_SUBGROUP_MAX_LENGTH], subgroup_len = 0;
            ASSERT_TRUE(agx3_subgroup_encode_ballot(enc, &d, subgroup,
                                                     &subgroup_len));
            struct agx3_encoded_end end;
            ASSERT_TRUE(agx3_pack_end(enc, &end));
            std::vector<uint8_t> want;
            append(want, subgroup, subgroup_len);
            append(want, end.bytes, end.length);
            EXPECT_EQ(pack(b), want);
         }
      }
   }
}

TEST_F(AGX3PackBinary, BallotWaitsForDeviceLoad)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   const agx_index value = agx_register(4, AGX_SIZE_32);
   agx_device_load_to(b, value, agx_register(8, AGX_SIZE_64),
                      agx_register(12, AGX_SIZE_32), AGX_FORMAT_I32, 1, 2,
                      false);
   agx_icmp_ballot_to(b, agx_register(6, AGX_SIZE_32), value, agx_zero(),
                      AGX_ICOND_UEQ, false);
   agx_stop(b);

   std::vector<uint8_t> want;
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], length = 0;
   /* Device loads use slot6. Retire it through the byte-grounded identity
    * prelude because the native ballot has no scoreboard field. */
   ASSERT_TRUE(agx3_alu_compose(AGX_ISA_ENCODING_AGX3B, AGX3_ALU_BAND,
                                0, 0, 0, 0, 0, 1u << 5, bytes, &length));
   append(want, bytes, length);

   const struct agx3_subgroup_ballot_desc ballot = {
      .type = AGX3_SUBGROUP_BALLOT_INTEGER,
      .dest = 6,
      .src = 4,
      .condition = 4,
   };
   ASSERT_TRUE(agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3B, &ballot,
                                            bytes, &length));
   append(want, bytes, length);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);

   const std::vector<uint8_t> got = pack(b);
   ASSERT_GE(got.size(), want.size());
   const std::vector<uint8_t> tail(got.end() - want.size(), got.end());
   EXPECT_EQ(tail, want);
}

static enum agx_simd_op
mesa_reduce_op(unsigned op)
{
   static const enum agx_simd_op map[] = {
      AGX_SIMD_OP_AND,  AGX_SIMD_OP_FADD, AGX_SIMD_OP_OR,
      AGX_SIMD_OP_FMUL, AGX_SIMD_OP_XOR,  AGX_SIMD_OP_FMIN,
      AGX_SIMD_OP_FMAX, AGX_SIMD_OP_IADD, AGX_SIMD_OP_SMIN,
      AGX_SIMD_OP_SMAX, AGX_SIMD_OP_UMIN, AGX_SIMD_OP_UMAX,
   };
   return map[op];
}

TEST_F(AGX3PackBinary, SimdReductionsAndExactPrefixes)
{
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      for (unsigned op = 0; op < AGX3_SUBGROUP_REDUCE_NUM; ++op) {
         agx_builder *b = builder(enc);
         agx_simd_reduce_to(b, agx_register(4, AGX_SIZE_32),
                            agx_register(2, AGX_SIZE_32), mesa_reduce_op(op));
         agx_stop(b);
         const struct agx3_subgroup_reduce_desc d = {
            .op = (enum agx3_subgroup_reduce_op)op,
            .dest = 4,
            .src = 2,
            .src_live = true,
         };
         uint8_t subgroup[AGX3_SUBGROUP_MAX_LENGTH], subgroup_len = 0;
         ASSERT_TRUE(agx3_subgroup_encode_reduce(enc, &d, subgroup,
                                                  &subgroup_len));
         struct agx3_encoded_end end;
         ASSERT_TRUE(agx3_pack_end(enc, &end));
         std::vector<uint8_t> want;
         append(want, subgroup, subgroup_len);
         append(want, end.bytes, end.length);
         EXPECT_EQ(pack(b), want);
      }

      for (unsigned op : {AGX3_SUBGROUP_REDUCE_FADD,
                          AGX3_SUBGROUP_REDUCE_FMUL,
                          AGX3_SUBGROUP_REDUCE_FMIN,
                          AGX3_SUBGROUP_REDUCE_FMAX,
                          AGX3_SUBGROUP_REDUCE_IADD}) {
         agx_builder *b = builder(enc);
         agx_quad_reduce_to(b, agx_register(4, AGX_SIZE_32),
                            agx_register(2, AGX_SIZE_32), mesa_reduce_op(op));
         agx_stop(b);
         const struct agx3_subgroup_reduce_desc d = {
            .quad = true,
            .op = (enum agx3_subgroup_reduce_op)op,
            .dest = 4,
            .src = 2,
            .src_live = true,
         };
         uint8_t subgroup[AGX3_SUBGROUP_MAX_LENGTH], subgroup_len = 0;
         ASSERT_TRUE(agx3_subgroup_encode_reduce(enc, &d, subgroup,
                                                  &subgroup_len));
         struct agx3_encoded_end end;
         ASSERT_TRUE(agx3_pack_end(enc, &end));
         std::vector<uint8_t> want;
         append(want, subgroup, subgroup_len);
         append(want, end.bytes, end.length);
         EXPECT_EQ(pack(b), want);
      }

      for (unsigned op : {AGX3_SUBGROUP_REDUCE_FADD,
                          AGX3_SUBGROUP_REDUCE_FMUL,
                          AGX3_SUBGROUP_REDUCE_IADD}) {
         agx_builder *b = builder(enc);
         agx_simd_prefix_to(b, agx_register(4, AGX_SIZE_32),
                            agx_register(2, AGX_SIZE_32), mesa_reduce_op(op));
         agx_stop(b);
         const struct agx3_subgroup_reduce_desc d = {
            .prefix = true,
            .op = (enum agx3_subgroup_reduce_op)op,
            .dest = 4,
            .src = 2,
            .src_live = true,
         };
         uint8_t subgroup[AGX3_SUBGROUP_MAX_LENGTH], subgroup_len = 0;
         ASSERT_TRUE(agx3_subgroup_encode_reduce(enc, &d, subgroup,
                                                  &subgroup_len));
         struct agx3_encoded_end end;
         ASSERT_TRUE(agx3_pack_end(enc, &end));
         std::vector<uint8_t> want;
         append(want, subgroup, subgroup_len);
         append(want, end.bytes, end.length);
         EXPECT_EQ(pack(b), want);
      }
   }
}

TEST_F(AGX3PackBinary, G17PClusteredIntegerReductions)
{
   for (unsigned op : {AGX3_SUBGROUP_REDUCE_AND,
                       AGX3_SUBGROUP_REDUCE_OR,
                       AGX3_SUBGROUP_REDUCE_XOR,
                       AGX3_SUBGROUP_REDUCE_SMIN,
                       AGX3_SUBGROUP_REDUCE_SMAX,
                       AGX3_SUBGROUP_REDUCE_UMIN,
                       AGX3_SUBGROUP_REDUCE_UMAX}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_quad_reduce_to(b, agx_register(4, AGX_SIZE_32),
                         agx_register(2, AGX_SIZE_32), mesa_reduce_op(op));
      agx_stop(b);

      const struct agx3_subgroup_reduce_desc d = {
         .quad = true,
         .op = (enum agx3_subgroup_reduce_op)op,
         .dest = 4,
         .src = 2,
         .src_live = true,
      };
      uint8_t subgroup[AGX3_SUBGROUP_MAX_LENGTH], subgroup_len = 0;
      ASSERT_TRUE(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &d,
                                               subgroup, &subgroup_len));
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
      std::vector<uint8_t> want;
      append(want, subgroup, subgroup_len);
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want);

      ASSERT_FALSE(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3C, &d,
                                                subgroup, &subgroup_len));
   }
}

TEST_F(AGX3PackBinary, G17PClusteredFloatReductions)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   const agx_index value = agx_register(4, AGX_SIZE_32);
   agx_device_load_to(b, value, agx_register(8, AGX_SIZE_64),
                      agx_register(12, AGX_SIZE_32), AGX_FORMAT_I32, 1, 2,
                      false);
   agx_quad_reduce_to(b, agx_register(0, AGX_SIZE_32), value,
                      AGX_SIMD_OP_FADD);
   agx_quad_reduce_to(b, agx_register(2, AGX_SIZE_32), value,
                      AGX_SIMD_OP_FMIN);
   agx_index final_value = value;
   final_value.discard = true;
   agx_quad_reduce_to(b, agx_register(4, AGX_SIZE_32), final_value,
                      AGX_SIMD_OP_FMAX);
   agx_stop(b);

   const uint8_t native[] = {
      0x37, 0x06, 0x56, 0x00, 0x02, 0x08, 0x16, 0x02,
      0xb7, 0x05, 0x54, 0x02, 0x02, 0x08, 0x0a, 0x02,
      0xb7, 0x07, 0x54, 0x04, 0x02, 0x08, 0x08, 0x12,
   };
   std::vector<uint8_t> want(native, native + sizeof(native));
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);

   const std::vector<uint8_t> got = pack(b);
   ASSERT_GE(got.size(), want.size());
   const std::vector<uint8_t> tail(got.end() - want.size(), got.end());
   EXPECT_EQ(tail, want);
}

TEST_F(AGX3PackBinary, SubgroupUnknownAxesFailClosed)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3C);
   agx_icmp_ballot_to(b, agx_register(4, AGX_SIZE_32),
                      agx_register(2, AGX_SIZE_32), agx_immediate(1),
                      AGX_ICOND_SEQ, false);
   agx_stop(b);
   pack(b, /*expect_ok=*/false);

   b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_simd_reduce_to(b, agx_register(4, AGX_SIZE_16),
                      agx_register(2, AGX_SIZE_16), AGX_SIMD_OP_IADD);
   agx_stop(b);
   pack(b, /*expect_ok=*/false);

   b = builder(AGX_ISA_ENCODING_AGX3C);
   agx_index killed = agx_register(2, AGX_SIZE_32);
   killed.kill = true;
   agx_simd_reduce_to(b, agx_register(4, AGX_SIZE_32), killed,
                      AGX_SIMD_OP_IADD);
   agx_stop(b);
   pack(b, /*expect_ok=*/false);
}

TEST_F(AGX3PackBinary, ScopedMemoryBarriers)
{
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      for (unsigned s = AGX3_BARRIER_SCOPE_EXECUTION;
           s <= AGX3_BARRIER_SCOPE_TEXTURE; ++s) {
         agx_builder *b = builder(enc);
         agx_scoped_barrier_agx3(b, s);
         agx_stop(b);

         struct agx3_encoded_scoped_barrier barrier;
         ASSERT_TRUE(agx3_pack_scoped_barrier(
            enc, (enum agx3_barrier_scope)s, &barrier));
         struct agx3_encoded_end end;
         ASSERT_TRUE(agx3_pack_end(enc, &end));

         std::vector<uint8_t> want;
         append(want, barrier.bytes, barrier.length);
         append(want, end.bytes, end.length);
         EXPECT_EQ(pack(b), want);
      }
   }
}

TEST_F(AGX3PackBinary, ScopedMemoryBarrierRefusesUnknownScope)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3C);
   agx_scoped_barrier_agx3(b, AGX3_BARRIER_SCOPE_TEXTURE + 1);
   agx_stop(b);
   pack(b, /*expect_ok=*/false);
}

TEST_F(AGX3PackBinary, DirectScsLoadStoreCarriesWait)
{
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      const agx_index value = agx_register(2, AGX_SIZE_32);
      agx_scs_load_agx3_to(b, value, 4, AGX_FORMAT_I32);
      agx_scs_store_agx3(b, value, 8, AGX_FORMAT_I32);
      agx_stop(b);

      uint8_t bytes[AGX3_SCS_MAX_LENGTH];
      uint8_t length = 0;
      std::vector<uint8_t> want;
      struct agx3_scs_desc d = {
         .data = 2,
         .width = AGX3_MEM_W32,
         .byte_offset = 4,
         .slot = 1,
      };
      ASSERT_TRUE(agx3_scs_encode_load(enc, &d, bytes, &length));
      append(want, bytes, length);
      d.byte_offset = 8;
      d.slot = 0;
      d.wait_mask = 1;
      ASSERT_TRUE(agx3_scs_encode_store(enc, &d, bytes, &length));
      append(want, bytes, length);
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(enc, &end));
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want);
   }
}

TEST_F(AGX3PackBinary, DirectScsScalarWidthsAndSignedOffsets)
{
   struct {
      enum agx_format format;
      enum agx_size size;
      enum agx3_mem_width width;
      int16_t offset;
   } cases[] = {
      {AGX_FORMAT_I8, AGX_SIZE_16, AGX3_MEM_W8, -8},
      {AGX_FORMAT_I16, AGX_SIZE_16, AGX3_MEM_W16, 10},
      {AGX_FORMAT_I32, AGX_SIZE_32, AGX3_MEM_W32, -12},
   };

   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      for (const auto &c : cases) {
         agx_builder *b = builder(enc);
         const agx_index value = agx_register(6, c.size);
         agx_scs_load_agx3_to(b, value, (uint64_t)(int64_t)c.offset, c.format);
         agx_stop(b);

         uint8_t bytes[AGX3_SCS_MAX_LENGTH];
         uint8_t length = 0;
         std::vector<uint8_t> want;
         const struct agx3_scs_desc d = {
            .data = 6,
            .width = c.width,
            .byte_offset = c.offset,
            .slot = 1,
         };
         ASSERT_TRUE(agx3_scs_encode_load(enc, &d, bytes, &length));
         append(want, bytes, length);
         struct agx3_encoded_end end;
         ASSERT_TRUE(agx3_pack_end(enc, &end));
         append(want, end.bytes, end.length);
         EXPECT_EQ(pack(b), want);
      }
   }
}

TEST_F(AGX3PackBinary, DirectScsRefusesVectorAccess)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3C);
   agx_index value = agx_register(2, AGX_SIZE_32);
   value.channels_m1 = 1;
   agx_scs_load_agx3_to(b, value, 0, AGX_FORMAT_I32);
   agx_stop(b);
   pack(b, /*expect_ok=*/false);
}

TEST_F(AGX3PackBinary, UniformIndexedScsScalarWidthsAndSignedOffsets)
{
   struct {
      enum agx_format format;
      enum agx_size size;
      enum agx3_mem_width width;
      int16_t offset;
   } cases[] = {
      {AGX_FORMAT_I8, AGX_SIZE_16, AGX3_MEM_W8, INT16_MAX},
      {AGX_FORMAT_I16, AGX_SIZE_16, AGX3_MEM_W16, INT16_MIN},
      {AGX_FORMAT_I32, AGX_SIZE_32, AGX3_MEM_W32, -12},
   };

   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      for (const auto &c : cases) {
         agx_builder *b = builder(enc);
         const agx_index value = agx_register(6, c.size);
         const agx_index base = agx_uniform(126, AGX_SIZE_64);
         const agx_index index = agx_register(10, AGX_SIZE_32);
         const uint64_t offset = (uint64_t)(int64_t)c.offset;
         agx_scs_load_uniform_indexed_agx3_to(b, value, base, index, offset,
                                               c.format,
                                               enc == AGX_ISA_ENCODING_AGX3C &&
                                                  c.width != AGX3_MEM_W32,
                                               enc == AGX_ISA_ENCODING_AGX3C);
         agx_scs_store_uniform_indexed_agx3(b, value, base, index, offset,
                                             c.format);
         agx_stop(b);

         uint8_t bytes[AGX3_SCS_MAX_LENGTH];
         uint8_t length = 0;
         std::vector<uint8_t> want;
         struct agx3_scs_desc d = {
            .data = 6,
            .width = c.width,
            .byte_offset = c.offset,
            .address_kind = AGX3_SCS_ADDRESS_UNIFORM_INDEXED,
            .index = 10,
            .uniform_base = 126,
            .slot = 1,
            .zero_extend = enc == AGX_ISA_ENCODING_AGX3C &&
                           c.width != AGX3_MEM_W32,
            .c_load_marker = enc == AGX_ISA_ENCODING_AGX3C,
         };
         ASSERT_TRUE(agx3_scs_encode_load(enc, &d, bytes, &length));
         append(want, bytes, length);
         d.slot = 0;
         d.wait_mask = 1;
         d.zero_extend = false;
         d.c_load_marker = false;
         ASSERT_TRUE(agx3_scs_encode_store(enc, &d, bytes, &length));
         append(want, bytes, length);
         struct agx3_encoded_end end;
         ASSERT_TRUE(agx3_pack_end(enc, &end));
         append(want, end.bytes, end.length);
         EXPECT_EQ(pack(b), want);
      }
   }
}

TEST_F(AGX3PackBinary, UniformIndexedScsCarriesHighestSlotWait)
{
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      const unsigned slots = e ? 8 : 6;
      agx_builder *b = builder(enc);
      const agx_index base = agx_uniform(4, AGX_SIZE_64);
      const agx_index index = agx_register(32, AGX_SIZE_32);
      for (unsigned i = 0; i < slots; ++i) {
         agx_scs_load_uniform_indexed_agx3_to(
            b, agx_register(2 * i, AGX_SIZE_32), base, index, 0,
            AGX_FORMAT_I32, false, enc == AGX_ISA_ENCODING_AGX3C);
      }
      agx_scs_store_uniform_indexed_agx3(
         b, agx_register(2 * (slots - 1), AGX_SIZE_32), base, index, 0,
         AGX_FORMAT_I32);
      agx_stop(b);

      uint8_t bytes[AGX3_SCS_MAX_LENGTH];
      uint8_t length = 0;
      std::vector<uint8_t> want;
      for (unsigned i = 0; i < slots; ++i) {
         const struct agx3_scs_desc d = {
            .data = (uint16_t)(2 * i),
            .width = AGX3_MEM_W32,
            .address_kind = AGX3_SCS_ADDRESS_UNIFORM_INDEXED,
            .index = 32,
            .uniform_base = 4,
            .slot = (uint8_t)(i + 1),
            .c_load_marker = enc == AGX_ISA_ENCODING_AGX3C,
         };
         ASSERT_TRUE(agx3_scs_encode_load(enc, &d, bytes, &length));
         append(want, bytes, length);
      }
      const struct agx3_scs_desc store = {
         .data = (uint16_t)(2 * (slots - 1)),
         .width = AGX3_MEM_W32,
         .address_kind = AGX3_SCS_ADDRESS_UNIFORM_INDEXED,
         .index = 32,
         .uniform_base = 4,
         .wait_mask = (uint16_t)(1u << (slots - 1)),
      };
      ASSERT_TRUE(agx3_scs_encode_store(enc, &store, bytes, &length));
      append(want, bytes, length);
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(enc, &end));
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want);
   }
}

TEST_F(AGX3PackBinary, UniformIndexedScsSelectorBoundaries)
{
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      const unsigned index_value = e ? 254 : 190;
      agx_builder *b = builder(enc);
      agx_scs_load_uniform_indexed_agx3_to(
         b, agx_register(2, AGX_SIZE_32), agx_uniform(511, AGX_SIZE_64),
         agx_register(index_value, AGX_SIZE_32), 0, AGX_FORMAT_I32, false,
         enc == AGX_ISA_ENCODING_AGX3C);
      agx_stop(b);

      uint8_t bytes[AGX3_SCS_MAX_LENGTH];
      uint8_t length = 0;
      std::vector<uint8_t> want;
      const struct agx3_scs_desc d = {
         .data = 2,
         .width = AGX3_MEM_W32,
         .address_kind = AGX3_SCS_ADDRESS_UNIFORM_INDEXED,
         .index = (uint16_t)index_value,
         .uniform_base = 511,
         .slot = 1,
         .c_load_marker = enc == AGX_ISA_ENCODING_AGX3C,
      };
      ASSERT_TRUE(agx3_scs_encode_load(enc, &d, bytes, &length));
      append(want, bytes, length);
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(enc, &end));
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want);
   }
}

TEST_F(AGX3PackBinary, UniformIndexedScsRefusesUnprovenAxes)
{
   const auto expect_load_failure = [this](enum agx_isa_encoding enc,
                                            agx_index value, agx_index base,
                                            agx_index index, uint64_t offset) {
      agx_builder *b = builder(enc);
      agx_scs_load_uniform_indexed_agx3_to(b, value, base, index, offset,
                                            AGX_FORMAT_I32, false,
                                            enc == AGX_ISA_ENCODING_AGX3C);
      agx_stop(b);
      pack(b, /*expect_ok=*/false);
   };

   const agx_index value = agx_register(2, AGX_SIZE_32);
   const agx_index base = agx_uniform(4, AGX_SIZE_64);
   const agx_index index = agx_register(6, AGX_SIZE_32);

   expect_load_failure(AGX_ISA_ENCODING_AGX3C, value,
                       agx_register(4, AGX_SIZE_64), index, 0);
   expect_load_failure(AGX_ISA_ENCODING_AGX3C, value,
                       agx_uniform(4, AGX_SIZE_32), index, 0);

   agx_index killed_index = index;
   killed_index.kill = true;
   expect_load_failure(AGX_ISA_ENCODING_AGX3C, value, base, killed_index, 0);

   agx_index vector_index = index;
   vector_index.channels_m1 = 1;
   expect_load_failure(AGX_ISA_ENCODING_AGX3C, value, base, vector_index, 0);

   expect_load_failure(AGX_ISA_ENCODING_AGX3C, value, base,
                       agx_register(6, AGX_SIZE_16), 0);

   agx_index base_oob = base;
   base_oob.value = 512;
   expect_load_failure(AGX_ISA_ENCODING_AGX3C, value, base_oob, index, 0);

   agx_index b_index_oob = index;
   b_index_oob.value = 192;
   expect_load_failure(AGX_ISA_ENCODING_AGX3B, value, base, b_index_oob, 0);

   agx_index c_index_oob = index;
   c_index_oob.value = 256;
   expect_load_failure(AGX_ISA_ENCODING_AGX3C, value, base, c_index_oob, 0);

   expect_load_failure(AGX_ISA_ENCODING_AGX3C, value, base, index, 32768);

   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_scs_load_uniform_indexed_agx3_to(b, value, base, index, 0,
                                         AGX_FORMAT_I32, false, true);
   agx_stop(b);
   pack(b, /*expect_ok=*/false);

   b = builder(AGX_ISA_ENCODING_AGX3C);
   agx_scs_load_uniform_indexed_agx3_to(b, value, base, index, 0,
                                         AGX_FORMAT_I32, true, true);
   agx_stop(b);
   pack(b, /*expect_ok=*/false);

   b = builder(AGX_ISA_ENCODING_AGX3C);
   agx_index killed_value = value;
   killed_value.kill = true;
   agx_scs_store_uniform_indexed_agx3(b, killed_value, base, index, 0,
                                       AGX_FORMAT_I32);
   agx_stop(b);
   pack(b, /*expect_ok=*/false);
}

TEST_F(AGX3PackBinary, StaticSpillCopyReloadDependency)
{
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      agx_index w0 = agx_register(0, AGX_SIZE_32);
      agx_index w2 = agx_register(2, AGX_SIZE_32);
      agx_index w4 = agx_register(4, AGX_SIZE_32);

      agx_stack_store(b, w0, agx_immediate(32), AGX_FORMAT_I32, 1);
      agx_stack_load_to(b, w2, agx_immediate(32), AGX_FORMAT_I32, 1);
      agx_iadd_to(b, w4, w2, w2, 0);
      agx_stop(b);

      std::vector<uint8_t> want;
      uint8_t bytes[AGX3_ALU_MAX_LENGTH];
      uint8_t length = 0;
      const struct agx3_static_spill_store_desc store = {
         .tps_half = 16,
         .src = 0,
         .width = AGX3_MEM_W32,
         .slot = 1,
      };
      ASSERT_TRUE(agx3_static_spill_encode_store(enc, &store, bytes, &length));
      append(want, bytes, length);

      const struct agx3_static_spill_load_desc load = {
         .dest = 2,
         .tps_half = 16,
         .width = AGX3_MEM_W32,
         .slot = 2,
         .slot_mirror = 2,
         .wait_mask = 1,
      };
      ASSERT_TRUE(agx3_static_spill_encode_load(enc, &load, bytes, &length));
      append(want, bytes, length);

      ASSERT_TRUE(agx3_alu_compose(enc, AGX3_ALU_IADD, 4, 2, 2, 0, 0,
                                   /* wait reload slot 2 */ 2, bytes, &length));
      append(want, bytes, length);
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(enc, &end));
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want);
   }
}

TEST_F(AGX3PackBinary, StaticSpillRefusesNonScalarOrNarrowShapes)
{
   {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index h0 = agx_register(0, AGX_SIZE_16);
      agx_stack_store(b, h0, agx_immediate(0), AGX_FORMAT_I16, 1);
      agx_stop(b);
      pack(b, /*expect_ok=*/false);
   }

   {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3C);
      agx_index w0 = agx_register(0, AGX_SIZE_32);
      w0.channels_m1 = 1;
      agx_stack_store(b, w0, agx_immediate(0), AGX_FORMAT_I32, 3);
      agx_stop(b);
      pack(b, /*expect_ok=*/false);
   }

   {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3C);
      agx_stack_load_to(b, agx_register(2, AGX_SIZE_32), agx_immediate(2),
                        AGX_FORMAT_I32, 1);
      agx_stop(b);
      pack(b, /*expect_ok=*/false);
   }
}

TEST_F(AGX3PackBinary, FloatModifierRouting)
{
   /* A negated source must route to the long modifier form, not the compact
    * packer, and match the modifier encoder byte for byte. */
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index w0 = agx_register(0, AGX_SIZE_32);
   agx_index w2 = agx_register(2, AGX_SIZE_32);
   agx_index w4 = agx_register(4, AGX_SIZE_32);
   agx_fadd_to(b, w4, agx_neg(w0), w2);
   agx_stop(b);

   struct agx3_falu_mods m = {};
   m.src0_neg = true;
   uint8_t mod[AGX3_FALU_MOD_MAX_LENGTH];
   uint8_t mlen = 0;
   ASSERT_TRUE(
      agx3_falu_add_mod(AGX_ISA_ENCODING_AGX3B, 4, 0, 2, &m, mod, &mlen));
   struct agx3_encoded_end en;
   agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &en);

   std::vector<uint8_t> want;
   append(want, mod, mlen);
   append(want, en.bytes, en.length);
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, RefusesUnmappedOp)
{
   /* BITREV remains unmapped; the backend must fail the shader rather than
    * treating it like the now-grounded tied-register FT1 operation. */
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index w0 = agx_register(0, AGX_SIZE_32);
   agx_instr *I = agx_ffs_to(b, w0, w0);
   I->op = AGX_OPCODE_BITREV;
   agx_stop(b);
   pack(b, /*expect_ok=*/false);
}

TEST_F(AGX3PackBinary, TextureFetch2DLod0Immediate)
{
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      for (int half = 0; half < 2; ++half) {
         agx_builder *b = builder(enc);
         agx_index dest = agx_register(0, half ? AGX_SIZE_16 : AGX_SIZE_32);
         agx_index coord = agx_register(12, AGX_SIZE_32);
         dest.channels_m1 = 3;
         coord.channels_m1 = 1;

         agx_instr *I = agx_texture_sample_to(
            b, dest, coord, agx_immediate(0), agx_immediate(0),
            agx_immediate(0), agx_immediate(0), agx_null(), AGX_DIM_2D,
            AGX_LOD_MODE_AUTO_LOD, 0, false, false, false, AGX_GATHER_NONE,
            false);
         I->op = AGX_OPCODE_TEXTURE_LOAD;
         I->mask = 15;
         agx_stop(b);

         struct agx3_tex_fetch_desc d = {};
         d.dest = 0;
         d.coord = 12;
         d.width = half ? AGX3_TEX_W16 : AGX3_TEX_W32;
         d.mask = 15;
         d.texture = 0;
         d.slot = 1;
         d.final = true;
         uint8_t tb[AGX3_TEX_MAX_LENGTH];
         uint8_t tl = 0;
         ASSERT_TRUE(agx3_tex_encode_fetch(enc, &d, tb, &tl));
         struct agx3_encoded_end end;
         ASSERT_TRUE(agx3_pack_end(enc, &end));

         std::vector<uint8_t> want;
         append(want, tb, tl);
         append(want, end.bytes, end.length);
         EXPECT_EQ(pack(b), want);
      }
   }
}

TEST_F(AGX3PackBinary, TextureFetchRefusesUngroundedAxes)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3C);
   agx_index dest = agx_register(0, AGX_SIZE_32);
   agx_index coord = agx_register(12, AGX_SIZE_32);
   dest.channels_m1 = 3;
   coord.channels_m1 = 1;
   agx_instr *I = agx_texture_sample_to(
      b, dest, coord, agx_immediate(0), agx_immediate(0),
      agx_immediate(16), agx_immediate(0), agx_null(), AGX_DIM_2D,
      AGX_LOD_MODE_AUTO_LOD, 0, false, false, false, AGX_GATHER_NONE, false);
   I->op = AGX_OPCODE_TEXTURE_LOAD;
   I->mask = 15;
   agx_stop(b);
   pack(b, /*expect_ok=*/false);
}

TEST_F(AGX3PackBinary, TextureExportRetiresAtPartBoundary)
{
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      key.no_stop = true;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_index dest = agx_register(8, AGX_SIZE_32);
      agx_index coord = agx_register(0, AGX_SIZE_32);
      dest.channels_m1 = 3;
      coord.channels_m1 = 1;
      agx_instr *tex = agx_texture_sample_to(
         b, dest, coord, agx_immediate(0), agx_immediate(0),
         agx_immediate(0), agx_immediate(0), agx_null(), AGX_DIM_2D,
         AGX_LOD_MODE_AUTO_LOD, 0, false, false, false, AGX_GATHER_NONE,
         false);
      tex->mask = 15;
      for (unsigned channel = 0; channel < 4; ++channel)
         agx_export(b, agx_register(8 + 2 * channel, AGX_SIZE_32),
                     8 + 2 * channel);

      std::vector<uint8_t> want;
      uint8_t bytes[AGX3_TEX_MAX_LENGTH], len = 0;
      ASSERT_TRUE(agx3_tex_stage_copy(enc, 0, 0, 1, 0, bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_tex_stage_copy(enc, 2, 2, 1, 0, bytes, &len));
      append(want, bytes, len);
      struct agx3_tex_sample_desc d = {};
      d.dest = 8;
      d.coord = 0;
      d.width = AGX3_TEX_W32;
      d.mask = 15;
      d.slot = 1;
      d.wait_mask = 1;
      d.final = true;
      ASSERT_TRUE(agx3_tex_encode_sample(enc, &d, bytes, &len));
      append(want, bytes, len);

      /* Four zero-byte exports are one outstanding texture result. The
       * independently compiled epilog cannot inherit this slot implicitly. */
      uint8_t wait[AGX3_ALU_MAX_LENGTH];
      ASSERT_TRUE(agx3_alu_compose(enc, AGX3_ALU_BAND, 0, 0, 0, 0, 0,
                                  1, wait, &len));
      append(want, wait, len);
      EXPECT_EQ(pack(b), want);
   }
}

TEST_F(AGX3PackBinary, ExportDoesNotConsumeRealReadWait)
{
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      key.no_stop = true;
      agx_index value = agx_register(4, AGX_SIZE_32);
      agx_get_sr_to(b, value, AGX_SR_THREAD_POSITION_IN_GRID_X);
      agx_export(b, value, 4);
      agx_iadd_to(b, agx_register(6, AGX_SIZE_32), value, value, 0);
      agx_export(b, agx_register(6, AGX_SIZE_32), 6);

      std::vector<uint8_t> want;
      struct agx3_sr_desc sr = {};
      sr.dest = 4;
      sr.width = AGX3_SR_W32;
      sr.sr = 23;
      sr.slot = 1;
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
      ASSERT_TRUE(agx3_sr_encode(enc, &sr, bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_alu_compose(enc, AGX3_ALU_IADD, 6, 4, 4, 0, 0,
                                  1, bytes, &len));
      append(want, bytes, len);
      /* The real consumer already waited: no extra tail instruction. */
      EXPECT_EQ(pack(b), want);
   }
}

TEST_F(AGX3PackBinary, DeviceLoadExportsRetireEveryOutstandingSlot)
{
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      key.no_stop = true;
      b->shader->stage = MESA_SHADER_COMPUTE;
      const agx_index base = agx_register(0, AGX_SIZE_64);
      const agx_index index = agx_register(4, AGX_SIZE_32);
      std::vector<uint8_t> want;
      uint8_t bytes[AGX3_MEM_MAX_LENGTH], len = 0;
      for (unsigned i = 0; i < 2; ++i) {
         const agx_index value = agx_register(6 + 2 * i, AGX_SIZE_32);
         agx_device_load_to(b, value, base, index, AGX_FORMAT_I32, 1, 0,
                             false);
         agx_export(b, value, value.value);
         struct agx3_mem_desc ld = {};
         ld.data = value.value;
         ld.width = AGX3_MEM_W32;
         ld.mask = 1;
         ld.shift = 2;
         ld.base = 0;
         ld.idx_kind = AGX3_MEM_IDX_REG;
         ld.idx_reg = 4;
         ld.slot = i + 1;
         ASSERT_TRUE(agx3_mem_encode_load(enc, &ld, bytes, &len));
         append(want, bytes, len);
      }
      for (unsigned i = 0; i < 2; ++i) {
         ASSERT_TRUE(agx3_alu_compose(enc, AGX3_ALU_BAND, 0, 0, 0, 0, 0,
                                     1u << i, bytes, &len));
         append(want, bytes, len);
      }
      EXPECT_EQ(pack(b), want);
   }
}

TEST_F(AGX3PackBinary, SynchronousExportPartHasNoTailDrain)
{
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      key.no_stop = true;
      agx_mov_imm_to(b, agx_register(8, AGX_SIZE_32), 0x3f800000);
      agx_export(b, agx_register(8, AGX_SIZE_32), 8);
      struct agx3_encoded_ldimm ld;
      ASSERT_TRUE(agx3_pack_ldimm_u32(enc, 8, 0x3f800000, &ld));
      std::vector<uint8_t> want;
      append(want, ld.bytes, ld.length);
      EXPECT_EQ(pack(b), want);
   }
}

TEST_F(AGX3PackBinary, TextureFetchWaitSurvivesFallthroughBlock)
{
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      agx_block *first = agx_start_block(b->shader);
      agx_index dest = agx_register(0, AGX_SIZE_32);
      agx_index coord = agx_register(12, AGX_SIZE_32);
      dest.channels_m1 = 3;
      coord.channels_m1 = 1;

      agx_instr *I = agx_texture_sample_to(
         b, dest, coord, agx_immediate(0), agx_immediate(0), agx_immediate(0),
         agx_immediate(0), agx_null(), AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 0,
         false, false, false, AGX_GATHER_NONE, false);
      I->op = AGX_OPCODE_TEXTURE_LOAD;
      I->mask = 15;

      agx_block *second = agx_test_block(b->shader);
      agx_block_add_successor(first, second);
      b->cursor = agx_after_block(second);
      agx_index scalar = dest;
      scalar.channels_m1 = 0;
      agx_iadd_to(b, agx_register(8, AGX_SIZE_32), scalar, scalar, 0);
      agx_stop(b);

      struct agx3_tex_fetch_desc d = {};
      d.dest = 0;
      d.coord = 12;
      d.width = AGX3_TEX_W32;
      d.mask = 15;
      d.slot = 1;
      d.final = true;
      uint8_t tb[AGX3_TEX_MAX_LENGTH];
      uint8_t tl = 0;
      ASSERT_TRUE(agx3_tex_encode_fetch(enc, &d, tb, &tl));

      uint8_t ab[AGX3_ALU_MAX_LENGTH];
      uint8_t al = 0;
      ASSERT_TRUE(agx3_alu_compose(enc, AGX3_ALU_IADD, 8, 0, 0, 0, 0,
                                   /*wait*/ 0x1, ab, &al));
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(enc, &end));

      std::vector<uint8_t> want;
      append(want, tb, tl);
      append(want, ab, al);
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want);
   }
}

TEST_F(AGX3PackBinary, TileStoreSingleSampleFormats)
{
   const struct {
      enum agx_format format;
      enum agx3_tile_width width;
   } formats[] = {
      {AGX_FORMAT_I8, AGX3_TILE_W8},
      {AGX_FORMAT_U8NORM, AGX3_TILE_W8},
      {AGX_FORMAT_S8NORM, AGX3_TILE_W8},
      {AGX_FORMAT_SRGBA8, AGX3_TILE_W8},
      {AGX_FORMAT_I16, AGX3_TILE_W16},
      {AGX_FORMAT_F16, AGX3_TILE_W16},
      {AGX_FORMAT_U16NORM, AGX3_TILE_W16},
      {AGX_FORMAT_S16NORM, AGX3_TILE_W16},
      {AGX_FORMAT_I32, AGX3_TILE_W32},
      {AGX_FORMAT_RGB10A2, AGX3_TILE_W32},
      {AGX_FORMAT_RG11B10F, AGX3_TILE_W32},
      {AGX_FORMAT_RGB9E5, AGX3_TILE_W32},
   };

   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      for (const auto &f : formats) {
         agx_builder *b = builder(enc);
         const bool rgb10 = f.format == AGX_FORMAT_RGB10A2;
         const unsigned mask = rgb10 ? 1 : 15;
         agx_index value = agx_register(0, f.width == AGX3_TILE_W8 ?
                                          AGX_SIZE_16 : AGX_SIZE_32);
         value.channels_m1 = f.width == AGX3_TILE_W8 ? 1 : rgb10 ? 0 : 3;
         agx_st_tile(b, value, agx_immediate(AGX_ALL_SAMPLES), agx_null(),
                     f.format, mask, 24, false);
         agx_stop(b);

         uint8_t tile[AGX3_TILE_MAX_LENGTH];
         uint8_t tile_len = 0;
         ASSERT_TRUE(agx3_tile_store(enc, 0, f.width, mask, 24,
                                     AGX3_TILE_COV_1X, 0, tile, &tile_len));
         struct agx3_encoded_end end;
         ASSERT_TRUE(agx3_pack_end(enc, &end));
         std::vector<uint8_t> want;
         append(want, tile, tile_len);
         append(want, end.bytes, end.length);
         EXPECT_EQ(pack(b), want);
      }
   }
}

TEST_F(AGX3PackBinary, TileLoadStoreWaitSurvivesFallthroughBlock)
{
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      agx_block *first = agx_start_block(b->shader);
      agx_index value = agx_register(0, AGX_SIZE_32);
      value.channels_m1 = 3;
      agx_ld_tile_to(b, value, agx_immediate(AGX_ALL_SAMPLES), agx_null(),
                     AGX_FORMAT_I32, 15, 16, false);

      agx_block *second = agx_test_block(b->shader);
      agx_block_add_successor(first, second);
      b->cursor = agx_after_block(second);
      value.discard = true; /* nonsemantic post-RA last-use hint */
      agx_st_tile(b, value, agx_immediate(AGX_ALL_SAMPLES), agx_null(),
                  AGX_FORMAT_I32, 15, 16, false);
      agx_stop(b);

      uint8_t tile[AGX3_TILE_MAX_LENGTH];
      uint8_t tile_len = 0;
      std::vector<uint8_t> want;
      ASSERT_TRUE(agx3_tile_load(enc, 0, AGX3_TILE_W32, 1, 15, 16,
                                 AGX3_TILE_COV_1X, 0, true, true, tile,
                                 &tile_len));
      append(want, tile, tile_len);
      ASSERT_TRUE(agx3_tile_store(enc, 0, AGX3_TILE_W32, 15, 16,
                                  AGX3_TILE_COV_1X, 1, tile, &tile_len));
      append(want, tile, tile_len);
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(enc, &end));
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want);
   }
}

TEST_F(AGX3PackBinary, TileLoadSequenceBoundaryMarkers)
{
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      agx_index a = agx_register(0, AGX_SIZE_32);
      agx_index c = agx_register(8, AGX_SIZE_32);
      a.channels_m1 = c.channels_m1 = 3;
      agx_ld_tile_to(b, a, agx_immediate(AGX_ALL_SAMPLES), agx_null(),
                     AGX_FORMAT_I32, 15, 0, false);
      agx_ld_tile_to(b, c, agx_immediate(AGX_ALL_SAMPLES), agx_null(),
                     AGX_FORMAT_I32, 15, 16, false);
      agx_stop(b);

      uint8_t tile[AGX3_TILE_MAX_LENGTH];
      uint8_t tile_len = 0;
      std::vector<uint8_t> want;
      ASSERT_TRUE(agx3_tile_load(enc, 0, AGX3_TILE_W32, 1, 15, 0,
                                 AGX3_TILE_COV_1X, 0, true, false, tile,
                                 &tile_len));
      append(want, tile, tile_len);
      ASSERT_TRUE(agx3_tile_load(enc, 8, AGX3_TILE_W32, 2, 15, 16,
                                 AGX3_TILE_COV_1X, 0, false, true, tile,
                                 &tile_len));
      append(want, tile, tile_len);
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(enc, &end));
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want);
   }
}

TEST_F(AGX3PackBinary, TileRefusesUngroundedAxes)
{
   {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3C);
      agx_index value = agx_register(0, AGX_SIZE_32);
      value.channels_m1 = 3;
      agx_st_tile(b, value, agx_register(8, AGX_SIZE_16), agx_null(),
                  AGX_FORMAT_I32, 15, 0, false);
      agx_stop(b);
      EXPECT_TRUE(pack(b, false).empty());
   }
   {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3C);
      agx_ld_tile_to(b, agx_register(0, AGX_SIZE_32),
                     agx_immediate(AGX_ALL_SAMPLES), agx_null(),
                     AGX_FORMAT_I32, 1, 0, false);
      agx_stop(b);
      EXPECT_TRUE(pack(b, false).empty());
   }
   {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index value = agx_register(16, AGX_SIZE_32);
      value.channels_m1 = 3;
      agx_st_tile(b, value, agx_immediate(AGX_ALL_SAMPLES), agx_null(),
                  AGX_FORMAT_I32, 15, 0, false);
      agx_stop(b);
      EXPECT_TRUE(pack(b, false).empty());
   }
   {
      /* A partial sample mask is per-sample traffic. AGX3's single-sample
       * group has no sample selector, so encoding it there would silently
       * widen the store to the whole pixel: refuse instead. */
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3C);
      agx_index value = agx_register(0, AGX_SIZE_32);
      value.channels_m1 = 3;
      agx_st_tile(b, value, agx_immediate(0x3), agx_null(), AGX_FORMAT_I32, 15,
                  0, false);
      agx_stop(b);
      EXPECT_TRUE(pack(b, false).empty());
   }
}

TEST_F(AGX3PackBinary, GetSrIndexedLoadAddStoreCarriesWait)
{
   /* The thread-indexed compute shape: read thread_position_in_grid.x, use it
    * as the index of a load, add, store back through the same index. The SR
    * read is an async producer: it takes scoreboard slot 1, the load waits on
    * it (for the index) while signalling slot 2, and the add waits on the
    * load. The store re-waits nothing -- both slots are already satisfied. */
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      agx_index base = agx_register(0, AGX_SIZE_64);
      agx_index gid = agx_register(4, AGX_SIZE_32);
      agx_index v6 = agx_register(6, AGX_SIZE_32);
      agx_index v8 = agx_register(8, AGX_SIZE_32);

      agx_get_sr_to(b, gid, AGX_SR_THREAD_POSITION_IN_GRID_X);
      agx_device_load_to(b, v6, base, gid, AGX_FORMAT_I32, 1, 2, false);
      agx_iadd_to(b, v8, v6, v6, 0);
      agx_device_store(b, v8, base, gid, AGX_FORMAT_I32, 1, 2, false);
      agx_stop(b);
      std::vector<uint8_t> got = pack(b);

      /* Independently build the expected stream from the encoders. */
      struct agx3_sr_desc sr = {};
      sr.dest = 4;
      sr.width = AGX3_SR_W32;
      sr.sr = 23;
      sr.slot = 1;
      sr.wait_mask = 0;
      uint8_t rb[AGX3_SR_MAX_LENGTH];
      uint8_t rl = 0;
      ASSERT_TRUE(agx3_sr_encode(enc, &sr, rb, &rl));

      struct agx3_mem_desc ld = {};
      ld.data = 6;
      ld.width = AGX3_MEM_W32;
      ld.mask = 1;
      ld.shift = 2;
      ld.base_is_uniform = false;
      ld.base = 0;
      ld.idx_kind = AGX3_MEM_IDX_REG;
      ld.idx_reg = 4;
      ld.slot = 2;
      ld.wait_mask = 0x1;
      uint8_t lb[AGX3_MEM_MAX_LENGTH];
      uint8_t ll = 0;
      ASSERT_TRUE(agx3_mem_encode_load(enc, &ld, lb, &ll));

      uint8_t ab[AGX3_ALU_MAX_LENGTH];
      uint8_t al = 0;
      ASSERT_TRUE(agx3_alu_compose(enc, AGX3_ALU_IADD, 8, 6, 6, 0, 0,
                                   /*wait*/ 0x2, ab, &al));

      struct agx3_mem_desc st = ld;
      st.data = 8;
      st.slot = 0;
      st.wait_mask = 0;
      uint8_t sb[AGX3_MEM_MAX_LENGTH];
      uint8_t sl = 0;
      ASSERT_TRUE(agx3_mem_encode_store(enc, &st, sb, &sl));

      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(enc, &end));

      std::vector<uint8_t> want;
      append(want, rb, rl);
      append(want, lb, ll);
      append(want, ab, al);
      append(want, sb, sl);
      append(want, end.bytes, end.length);

      EXPECT_EQ(got, want);
   }
}

TEST_F(AGX3PackBinary, RefusesUnwitnessedSr)
{
   /* samples_log2 has no corpus witness tying it to an Apple SR number; the
    * backend must fail the shader, not guess an encoding. */
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index h0 = agx_register(0, AGX_SIZE_16);
   agx_get_sr_to(b, h0, AGX_SR_SAMPLES_LOG2);
   agx_stop(b);
   pack(b, /*expect_ok=*/false);
}

TEST_F(AGX3PackBinary, LoadAddStoreCarriesWait)
{
   /* A load, an add that consumes the load result, and a store. The add must
    * carry a wait on the load's scoreboard slot; the shim must produce exactly
    * what the memory + wait-carrying-ALU encoders produce for these operands. */
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      agx_index base = agx_register(0, AGX_SIZE_64);
      agx_index idx = agx_register(4, AGX_SIZE_32);
      agx_index v6 = agx_register(6, AGX_SIZE_32);
      agx_index v8 = agx_register(8, AGX_SIZE_32);

      agx_device_load_to(b, v6, base, idx, AGX_FORMAT_I32, 1, 2, false);
      agx_iadd_to(b, v8, v6, v6, 0);
      agx_device_store(b, v8, base, idx, AGX_FORMAT_I32, 1, 2, false);
      agx_stop(b);
      std::vector<uint8_t> got = pack(b);

      /* Independently build the expected stream from the encoders. */
      struct agx3_mem_desc ld = {};
      ld.data = 6;
      ld.width = AGX3_MEM_W32;
      ld.mask = 1;
      ld.shift = 2;
      ld.base_is_uniform = false;
      ld.base = 0;
      ld.idx_kind = AGX3_MEM_IDX_REG;
      ld.idx_reg = 4;
      ld.slot = 1;
      ld.wait_mask = 0;
      uint8_t lb[AGX3_MEM_MAX_LENGTH];
      uint8_t ll = 0;
      ASSERT_TRUE(agx3_mem_encode_load(enc, &ld, lb, &ll));

      uint8_t ab[AGX3_ALU_MAX_LENGTH];
      uint8_t al = 0;
      ASSERT_TRUE(agx3_alu_compose(enc, AGX3_ALU_IADD, 8, 6, 6, 0, 0,
                                   /*wait*/ 0x1, ab, &al));

      struct agx3_mem_desc st = ld;
      st.data = 8;
      st.slot = 0;
      st.wait_mask = 0;
      uint8_t sb[AGX3_MEM_MAX_LENGTH];
      uint8_t sl = 0;
      ASSERT_TRUE(agx3_mem_encode_store(enc, &st, sb, &sl));

      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(enc, &end));

      std::vector<uint8_t> want;
      append(want, lb, ll);
      append(want, ab, al);
      append(want, sb, sl);
      append(want, end.bytes, end.length);

      EXPECT_EQ(got, want);
   }
}

TEST_F(AGX3PackBinary, LocalLoadAddStoreCarriesWait)
{
   /* Current Mesa shared-memory lowering folds the byte address into the
    * 16-bit base and leaves an immediate-zero index. Exercise that live path,
    * including the async ldtg slot and the dependent ALU wait. */
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      agx_index base = agx_uniform(8, AGX_SIZE_16);
      agx_index v0 = agx_register(0, AGX_SIZE_32);
      agx_index v2 = agx_register(2, AGX_SIZE_32);

      agx_local_load_to(b, v0, base, agx_zero(), AGX_FORMAT_I32, 1);
      agx_iadd_to(b, v2, v0, v0, 0);
      agx_local_store(b, v2, base, agx_zero(), AGX_FORMAT_I32, 1);
      agx_stop(b);

      struct agx3_local_desc ld = {};
      ld.data = 0;
      ld.width = AGX3_MEM_W32;
      ld.mask = 1;
      ld.shift = 2;
      ld.base_kind = AGX3_LOCAL_BASE_UNIFORM;
      ld.base = 8;
      ld.slot = 1;
      uint8_t lb[AGX3_LOCAL_TPS_MAX_LENGTH];
      uint8_t ll = 0;
      ASSERT_TRUE(agx3_local_encode_load(enc, &ld, lb, &ll));

      uint8_t ab[AGX3_ALU_MAX_LENGTH];
      uint8_t al = 0;
      ASSERT_TRUE(agx3_alu_compose(enc, AGX3_ALU_IADD, 2, 0, 0, 0, 0,
                                   /*wait*/ 1, ab, &al));

      struct agx3_local_desc st = ld;
      st.data = 2;
      st.slot = 0;
      uint8_t sb[AGX3_LOCAL_TPS_MAX_LENGTH];
      uint8_t sl = 0;
      ASSERT_TRUE(agx3_local_encode_store(enc, &st, sb, &sl));

      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(enc, &end));
      std::vector<uint8_t> want;
      append(want, lb, ll);
      append(want, ab, al);
      append(want, sb, sl);
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want);
   }
}

TEST_F(AGX3PackBinary, VaryingInterpolationCoefficientAndUvs)
{
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      b->shader->stage = MESA_SHADER_COMPUTE; /* Synthetic opcode mixture. */

      agx_interp_agx3_to(b, agx_register(0, AGX_SIZE_32), agx_immediate(3),
                         agx_zero(), AGX_INTERPOLATION_CENTER, false);
      agx_interp_agx3_to(b, agx_register(2, AGX_SIZE_32), agx_immediate(4),
                         agx_zero(), AGX_INTERPOLATION_CENTROID, true);
      agx_ldcf_to(b, agx_register(4, AGX_SIZE_32), agx_immediate(5), 1);
      agx_instr *store =
         agx_st_vary(b, agx_immediate(7), agx_register(4, AGX_SIZE_32));
      store->src[1].kill = true;
      agx_stop(b);

      uint8_t bytes[MAX2(AGX3_STAGE_MAX_LENGTH, AGX3_VARY_MAX_LENGTH)];
      uint8_t len = 0;
      std::vector<uint8_t> want;
      ASSERT_TRUE(agx3_vary_interp_at(enc, 0, 3, false, AGX3_VARY_CENTER,
                                      bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_vary_interp_at(enc, 2, 4, true, AGX3_VARY_CENTROID,
                                      bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_vary_ldcf(enc, 4, 5, 1, bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_stage_copy_source(enc, 0, 4, true, 1, 1, bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_vary_uvsw(enc, 0, 7, 1, true, 1, bytes, &len));
      append(want, bytes, len);
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(enc, &end));
      append(want, end.bytes, end.length);

      EXPECT_EQ(pack(b), want);
      EXPECT_EQ(b->shader->agx3_staging_halfs, 2u);
   }
}

TEST_F(AGX3PackBinary, RefusesExplicitSampleInterpolation)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3C);
   agx_interp_agx3_to(b, agx_register(0, AGX_SIZE_32), agx_immediate(3),
                      agx_register(2, AGX_SIZE_16), AGX_INTERPOLATION_SAMPLE,
                      false);
   agx_stop(b);
   EXPECT_TRUE(pack(b, /*expect_ok=*/false).empty());
}

TEST_F(AGX3PackBinary, VertexStagingProducerMatchesAppleAndPreservesLiveness)
{
   static const uint8_t apple_x[] = {0x0b, 0x00, 0x29, 0xc4};
   static const uint8_t apple_y[] = {
      0x1b, 0x03, 0x2f, 0x00, 0x42, 0x00, 0x00, 0x04, 0x00, 0x00,
   };
   uint8_t bytes[AGX3_STAGE_MAX_LENGTH], len;
   ASSERT_TRUE(agx3_stage_copy_source(AGX_ISA_ENCODING_AGX3B, 0, 0, true,
                                      1, 0x20, bytes, &len));
   ASSERT_EQ(len, sizeof(apple_x));
   EXPECT_EQ(memcmp(bytes, apple_x, len), 0);
   ASSERT_TRUE(agx3_stage_copy_source(AGX_ISA_ENCODING_AGX3B, 2, 2, true,
                                      2, 0, bytes, &len));
   ASSERT_EQ(len, sizeof(apple_y));
   EXPECT_EQ(memcmp(bytes, apple_y, len), 0);

   ASSERT_TRUE(agx3_stage_copy_source(AGX_ISA_ENCODING_AGX3B, 2, 2, false,
                                      2, 0, bytes, &len));
   /* f27 is the GPR last-use flag, independent of staging f9/f5/f12. */
   EXPECT_EQ(bytes[2], apple_y[2] & ~8u);
   bytes[2] |= 8;
   EXPECT_EQ(memcmp(bytes, apple_y, len), 0);
}

TEST_F(AGX3PackBinary, VertexStagingBankAndScoreboardRoundTrip)
{
   for (unsigned slot = 1; slot <= 6; ++slot) {
      for (unsigned wait : {0u, 1u, 2u, 4u, 8u, 16u, 32u, 33u}) {
         uint8_t bytes[AGX3_STAGE_MAX_LENGTH], len;
         const unsigned staging = 2 * (slot - 1);
         ASSERT_TRUE(agx3_stage_copy_source(AGX_ISA_ENCODING_AGX3B, staging,
                                            200, false, slot, wait, bytes, &len));
         /* Decode the selected COPY destination bank and narrow indices.
          * This is the actual f9=5 form, not a GPR copy to numeric index s. */
         EXPECT_EQ(bytes[0] & 0xf, 0xbu);
         EXPECT_EQ(bytes[2] & 0xf7, len == 4 ? 0x21u : 0x27u);
         EXPECT_EQ((bytes[0] >> 4) * 2, staging);
         EXPECT_EQ((bytes[1] & 0xfe) | (len == 10 ? (bytes[5] & 1) << 7
                                                             : (bytes[3] & 16) << 3),
                   200u);
         unsigned decoded_slot, decoded_wait;
         if (len == 4) {
            decoded_slot = 1;
            const unsigned index = bytes[3] >> 5;
            decoded_wait = index ? 1u << (index - 1) : 0;
         } else {
            ASSERT_EQ(len, 10u);
            decoded_slot = 1 + ((bytes[7] >> 2) & 7);
            decoded_wait = (bytes[5] >> 5) | ((bytes[7] >> 5) << 3);
         }
         EXPECT_EQ(decoded_slot, slot);
         EXPECT_EQ(decoded_wait, wait);

         ASSERT_TRUE(agx3_vary_uvsw(AGX_ISA_ENCODING_AGX3B, staging, 0, slot,
                                    true, 1u << (slot - 1), bytes, &len));
         ASSERT_EQ(len, 8u);
         EXPECT_EQ(bytes[3], staging);
         EXPECT_EQ(1 + (bytes[6] & 7), decoded_slot);
         const unsigned consumer_wait = (bytes[1] >> 4) | ((bytes[2] & 3) << 4);
         EXPECT_EQ(consumer_wait, 1u << (decoded_slot - 1));
      }
   }
}

TEST_F(AGX3PackBinary, VertexStagingAllocationDoesNotUseGprHighWater)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->max_reg = 201;
   agx_st_vary(b, agx_immediate(0), agx_register(200, AGX_SIZE_32));
   agx_st_vary(b, agx_immediate(1), agx_register(200, AGX_SIZE_32));
   agx_stop(b);
   EXPECT_FALSE(pack(b).empty());
   EXPECT_EQ(b->shader->max_reg, 201u);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 4u);
   EXPECT_TRUE(b->shader->agx3_staging_known);
}

TEST_F(AGX3PackBinary, AllGroundedComparisonRelations)
{
   const struct {
      enum agx_icond mesa;
      enum agx3_cf_icond apple;
   } integer_cases[] = {
      {AGX_ICOND_UEQ, AGX3_CF_ICOND_EQ},
      {AGX_ICOND_SEQ, AGX3_CF_ICOND_EQ},
      {AGX_ICOND_ULT, AGX3_CF_ICOND_ULT},
      {AGX_ICOND_UGT, AGX3_CF_ICOND_UGT},
      {AGX_ICOND_SLT, AGX3_CF_ICOND_SLT},
      {AGX_ICOND_SGT, AGX3_CF_ICOND_SGT},
   };
   const struct {
      enum agx_fcond mesa;
      enum agx3_cf_fcond apple;
      bool or_nan;
      bool flips_invert;
   } float_cases[] = {
      {AGX_FCOND_EQ, AGX3_CF_FCOND_EQ, false, false},
      {AGX_FCOND_LT, AGX3_CF_FCOND_LT, false, false},
      {AGX_FCOND_GT, AGX3_CF_FCOND_GT, false, false},
      {AGX_FCOND_LTN, AGX3_CF_FCOND_LT, true, false},
      {AGX_FCOND_GTN, AGX3_CF_FCOND_GT, true, false},
      {AGX_FCOND_GE, AGX3_CF_FCOND_LT, true, true},
      {AGX_FCOND_LE, AGX3_CF_FCOND_GT, true, true},
   };

   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;

      for (const auto &test : integer_cases) {
         for (int invert = 0; invert < 2; ++invert) {
            agx_builder *b = builder(enc);
            agx_if_icmp(b, agx_register(0, AGX_SIZE_32),
                        agx_register(2, AGX_SIZE_32), 1, test.mesa, invert,
                        NULL);
            agx_stop(b);

            uint8_t bytes[AGX3_CF_MAX_LENGTH];
            uint8_t len = 0;
            std::vector<uint8_t> want;
            ASSERT_TRUE(agx3_cf_icmp_reg(enc, test.apple, 0, 0, 2, bytes,
                                         &len));
            append(want, bytes, len);
            ASSERT_TRUE(agx3_cf_cndst(enc, 0, invert, 1, bytes, &len));
            append(want, bytes, len);
            struct agx3_encoded_end end;
            ASSERT_TRUE(agx3_pack_end(enc, &end));
            append(want, end.bytes, end.length);
            EXPECT_EQ(pack(b), want);
         }
      }

      for (const auto &test : float_cases) {
         for (int invert = 0; invert < 2; ++invert) {
            agx_builder *b = builder(enc);
            agx_if_fcmp(b, agx_register(0, AGX_SIZE_32),
                        agx_register(2, AGX_SIZE_32), 1, test.mesa, invert,
                        NULL);
            agx_stop(b);

            uint8_t bytes[AGX3_CF_MAX_LENGTH];
            uint8_t len = 0;
            std::vector<uint8_t> want;
            ASSERT_TRUE(agx3_cf_fcmp_reg(enc, test.apple, test.or_nan, 0, 0,
                                         2, bytes, &len));
            append(want, bytes, len);
            ASSERT_TRUE(agx3_cf_cndst(enc, 0,
                                      (bool)invert ^ test.flips_invert, 1,
                                      bytes, &len));
            append(want, bytes, len);
            struct agx3_encoded_end end;
            ASSERT_TRUE(agx3_pack_end(enc, &end));
            append(want, end.bytes, end.length);
            EXPECT_EQ(pack(b), want);
         }
      }
   }
}

TEST_F(AGX3PackBinary, ImmediateLeftIntegerComparisonsCanonicalize)
{
   const struct {
      enum agx_icond mesa;
      enum agx3_cf_icond swapped;
   } cases[] = {
      {AGX_ICOND_UEQ, AGX3_CF_ICOND_EQ},
      {AGX_ICOND_SEQ, AGX3_CF_ICOND_EQ},
      {AGX_ICOND_ULT, AGX3_CF_ICOND_UGT},
      {AGX_ICOND_UGT, AGX3_CF_ICOND_ULT},
      {AGX_ICOND_SLT, AGX3_CF_ICOND_SGT},
      {AGX_ICOND_SGT, AGX3_CF_ICOND_SLT},
   };

   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;

      for (const auto &test : cases) {
         agx_builder *b = builder(enc);
         agx_if_icmp(b, agx_immediate(15),
                     agx_register(2, AGX_SIZE_32), 1, test.mesa, false, NULL);
         agx_stop(b);

         uint8_t bytes[AGX3_CF_MAX_LENGTH];
         uint8_t len = 0;
         std::vector<uint8_t> want;
         ASSERT_TRUE(agx3_cf_icmp_imm(enc, test.swapped, 0, 2, 15, bytes,
                                      &len));
         append(want, bytes, len);
         ASSERT_TRUE(agx3_cf_cndst(enc, 0, false, 1, bytes, &len));
         append(want, bytes, len);
         struct agx3_encoded_end end;
         ASSERT_TRUE(agx3_pack_end(enc, &end));
         append(want, end.bytes, end.length);
         EXPECT_EQ(pack(b), want);
      }
   }
}

TEST_F(AGX3PackBinary, StructuredIfAndForwardBranch)
{
   /* One fused AGX IR if becomes icmp + cndst. The optional exec-none jump is
    * a reserved bn patched from its own start to the target block's cndend. */
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      agx_block *pop = agx_test_block(b->shader);
      agx_index a = agx_register(0, AGX_SIZE_32);
      agx_index c = agx_register(2, AGX_SIZE_32);

      agx_begin_cf(b); /* architectural on AGX3: emits zero bytes */
      agx_if_icmp(b, a, c, 1, AGX_ICOND_ULT, false, pop);
      agx_jmp_exec_none(b, pop);

      agx_builder pb = agx_init_builder(b->shader, agx_after_block(pop));
      agx_pop_exec(&pb, 1);
      agx_stop(&pb);

      uint8_t bytes[AGX3_CF_MAX_LENGTH];
      uint8_t len = 0;
      std::vector<uint8_t> want;
      ASSERT_TRUE(
         agx3_cf_icmp_reg(enc, AGX3_CF_ICOND_ULT, 0, 0, 2, bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_cf_cndst(enc, 0, false, 1, bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_cf_branch(enc, true, bytes, &len));
      ASSERT_EQ(len, AGX3_CF_MAX_LENGTH);
      ASSERT_TRUE(agx3_cf_patch_branch(enc, true, bytes, 10));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_cf_cndend(enc, 1, bytes, &len));
      append(want, bytes, len);
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(enc, &end));
      append(want, end.bytes, end.length);

      EXPECT_EQ(pack(b), want);
   }
}

TEST_F(AGX3PackBinary, NestedLoopEntryBackedgeAndPop)
{
   /* Mesa's push_exec(1) + jmp_exec_any + pop_exec(1) loop skeleton maps to
    * AGX3's TRUE-slot cndst(push=2), backwards b, and cndend(pop=2). */
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      agx_index a = agx_register(0, AGX_SIZE_32);
      agx_index c = agx_register(2, AGX_SIZE_32);

      agx_push_exec(b, 1);
      agx_block *body = agx_test_block(b->shader);
      agx_builder lb = agx_init_builder(b->shader, agx_after_block(body));
      agx_while_icmp(&lb, a, c, 1, AGX_ICOND_ULT, false, NULL);
      agx_jmp_exec_any(&lb, body);
      agx_pop_exec(&lb, 1);
      agx_stop(&lb);

      uint8_t bytes[AGX3_CF_MAX_LENGTH];
      uint8_t len = 0;
      std::vector<uint8_t> want;
      ASSERT_TRUE(agx3_cf_cndst(enc, 6, false, 2, bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(
         agx3_cf_icmp_reg(enc, AGX3_CF_ICOND_ULT, 1, 0, 2, bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_cf_cndlt(enc, 1, false, bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_cf_branch(enc, false, bytes, &len));
      ASSERT_EQ(len, AGX3_CF_MAX_LENGTH);
      ASSERT_TRUE(agx3_cf_patch_branch(enc, false, bytes, -14));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_cf_cndend(enc, 2, bytes, &len));
      append(want, bytes, len);
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(enc, &end));
      append(want, end.bytes, end.length);

      EXPECT_EQ(pack(b), want);
   }
}

TEST_F(AGX3PackBinary, FloatRelationAndCanonicalElse)
{
   /* Ordered >= is invert(less-than-or-NaN); the inversion belongs to cndst,
    * not fcmp. Mesa's fixed else_fcmp(0,0,eq) becomes cndef with no compare. */
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      agx_index a = agx_register(0, AGX_SIZE_32);
      agx_index c = agx_register(2, AGX_SIZE_32);
      agx_if_fcmp(b, a, c, 1, AGX_FCOND_GE, false, NULL);
      agx_else_fcmp(b, agx_zero(), agx_zero(), 1, AGX_FCOND_EQ, false, NULL);
      agx_pop_exec(b, 1);
      agx_stop(b);

      uint8_t bytes[AGX3_CF_MAX_LENGTH];
      uint8_t len = 0;
      std::vector<uint8_t> want;
      ASSERT_TRUE(
         agx3_cf_fcmp_reg(enc, AGX3_CF_FCOND_LT, true, 0, 0, 2, bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_cf_cndst(enc, 0, true, 1, bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_cf_cndef(enc, bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_cf_cndend(enc, 1, bytes, &len));
      append(want, bytes, len);
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(enc, &end));
      append(want, end.bytes, end.length);

      EXPECT_EQ(pack(b), want);
   }
}

TEST_F(AGX3PackBinary, RefusesMemoryDependentControlFlowAndRollsBack)
{
   /* The compare/cnd hazard fields are not grounded yet. A load before a
    * condition therefore fails the whole pack and leaves no executable-looking
    * prefix, even when the condition happens to use a different register. */
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index base = agx_register(0, AGX_SIZE_64);
   agx_index index = agx_register(4, AGX_SIZE_32);
   agx_index loaded = agx_register(6, AGX_SIZE_32);
   agx_index cond = agx_register(8, AGX_SIZE_32);
   agx_device_load_to(b, loaded, base, index, AGX_FORMAT_I32, 1, 2, false);
   agx_if_icmp(b, cond, agx_immediate(0), 1, AGX_ICOND_UEQ, true, NULL);
   agx_stop(b);

   EXPECT_TRUE(pack(b, /*expect_ok=*/false).empty());
}

TEST_F(AGX3PackBinary, RefusesUngroundedUnconditionalBreak)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3C);
   agx_break(b, 1, NULL);
   agx_stop(b);
   EXPECT_TRUE(pack(b, /*expect_ok=*/false).empty());
}

TEST_F(AGX3PackBinary, FloatRegisterLivenessNativeWitnesses)
{
   const struct {
      enum agx3_alu_op op;
      uint16_t dest, src[3], wait;
      enum agx3_alu_hint hint[3];
      unsigned len;
      uint8_t bytes[8];
   } cases[] = {
      {AGX3_ALU_FADD,0,{4,2,0},32,{AGX3_ALU_HINT_NONE,AGX3_ALU_HINT_NONE,AGX3_ALU_HINT_NONE},6,{9,3,4,5,0,0xc0}},
      {AGX3_ALU_FMUL,0,{4,2,0},32,{AGX3_ALU_HINT_NONE,AGX3_ALU_HINT_NONE,AGX3_ALU_HINT_NONE},6,{9,3,5,5,0,0xc0}},
      {AGX3_ALU_FMA,0,{4,2,6},32,{AGX3_ALU_HINT_NONE,AGX3_ALU_HINT_NONE,AGX3_ALU_HINT_NONE},8,{9,3,6,5,1,6,2,0xc0}},
      {AGX3_ALU_FMUL,0,{8,6,0},32,{AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_NONE},6,{9,0x87,5,0x89,0,0xc0}},
      {AGX3_ALU_FMA,2,{10,6,12},0,{AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_DISCARD},6,{0x19,0x87,0x16,0x0b,0x80,0x0c}},
      {AGX3_ALU_FADD,4,{8,6,0},0,{AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_NONE,AGX3_ALU_HINT_NONE},4,{0x29,7,0x10,9}},
   };
   for (const auto &v : cases) {
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
      ASSERT_TRUE(agx3_alu_float_reg(AGX_ISA_ENCODING_AGX3B, v.op, v.dest,
                                      v.src, v.hint, v.wait, bytes, &len));
      ASSERT_EQ(len, v.len);
      EXPECT_EQ(memcmp(bytes, v.bytes, len), 0);
   }
}

TEST_F(AGX3PackBinary, FloatRegisterLivenessPreservedThroughBackend)
{
   /* Alpha blending repeatedly reads both factors. Exercise all three float
    * operations with independent hints; physical bit checks do not call the
    * production encoder to derive expected flags. */
   for (unsigned op = 0; op < 3; ++op) {
      for (unsigned h0 = 0; h0 < 3; ++h0) {
         for (unsigned h1 = 0; h1 < 3; ++h1) {
            for (unsigned h2 = 0; h2 < (op == 2 ? 3u : 1u); ++h2) {
               agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
               b->shader->stage = MESA_SHADER_FRAGMENT;
               agx_index src[3] = {agx_register(2, AGX_SIZE_32),
                  agx_register(4, AGX_SIZE_32), agx_register(6, AGX_SIZE_32)};
               const unsigned hints[] = {h0, h1, h2};
               for (unsigned i = 0; i < 3; ++i) {
                  src[i].cache = hints[i] == 2;
                  src[i].discard = hints[i] == 1;
               }
               const agx_index dest = agx_register(8, AGX_SIZE_32);
               if (op == 0) agx_fadd_to(b, dest, src[0], src[1]);
               else if (op == 1) agx_fmul_to(b, dest, src[0], src[1]);
               else agx_ffma_to(b, dest, src[0], src[1], src[2]);
               agx_stop(b);
               const auto bytes = pack(b);
               ASSERT_EQ(bytes.size(), op == 2 ? 10u : 8u);
               EXPECT_EQ((bytes[3] >> 7) & 1, h0 == 2);
               EXPECT_EQ((bytes[2] >> 4) & 1, h0 == 1);
               EXPECT_EQ((bytes[1] >> 7) & 1, h1 == 2);
               EXPECT_EQ((bytes[2] >> 3) & 1, h1 == 1);
               if (op == 2) {
                  EXPECT_EQ((bytes[5] >> 7) & 1, h2 == 2);
                  EXPECT_EQ((bytes[4] >> 7) & 1, h2 == 1);
               }
            }
         }
      }
   }
}

TEST_F(AGX3PackBinary, FloatRegisterLivenessRetiresUnsupportedCompactWaits)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_COMPUTE;
   const agx_index x = agx_register(2, AGX_SIZE_32);
   const agx_index y = agx_register(6, AGX_SIZE_32);
   agx_get_sr_to(b, x, AGX_SR_THREAD_POSITION_IN_GRID_X);
   agx_get_sr_to(b, agx_register(4, AGX_SIZE_32), AGX_SR_THREAD_POSITION_IN_GRID_Y);
   agx_get_sr_to(b, y, AGX_SR_THREAD_POSITION_IN_GRID_Z);
   agx_index retained = y;
   retained.cache = true;
   agx_fmul_to(b, agx_register(8, AGX_SIZE_32), x, retained);
   agx_stop(b);
   const auto bytes = pack(b);
   ASSERT_GE(bytes.size(), 8u);
   /* Last multiply keeps its second source: f27=cache, f29=0. Its trailer
    * is END, so this also checks that the fallback emitted the operation. */
   const unsigned at = bytes.size() - 8;
   EXPECT_EQ(bytes[at] & 0xf, 9);
   EXPECT_EQ((bytes[at + 1] >> 7) & 1, 1);
   EXPECT_EQ((bytes[at + 2] >> 3) & 1, 0);
}

TEST_F(AGX3PackBinary, FloatRegisterLivenessRefusesContradictoryHints)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index src = agx_register(2, AGX_SIZE_32);
   src.cache = src.discard = true;
   agx_fmul_to(b, agx_register(0, AGX_SIZE_32), src,
                    agx_register(4, AGX_SIZE_32));
   agx_stop(b);
   EXPECT_TRUE(pack(b, false).empty());
}

TEST_F(AGX3PackBinary, G17PAlphaProtocolAcquiresBeforeLoadReleasesAfterStore)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_index packed = agx_register(0, AGX_SIZE_16);
   packed.channels_m1 = 1;
   agx_wait_pix(b, 8);
   agx_ld_tile_to(b, packed, agx_immediate(AGX_ALL_SAMPLES), agx_null(),
                  AGX_FORMAT_U8NORM, 15, 0, false);
   agx_wait_pix(b, 4);
   agx_st_tile(b, packed, agx_immediate(AGX_ALL_SAMPLES), agx_null(),
               AGX_FORMAT_U8NORM, 15, 0, false);
   agx_stop(b);

   const auto bin = pack(b);
   const uint8_t prefix[] = {
      0x87,0x02,0x54,0x00,0x06,0x00, /* class12 entry */
      0x87,0x02,0x54,0x08,0x08,0x00, /* read wait */
      0x67,0x0e,0x54,0x00,0x00,0x00,0x01,0x4e,0x00,0x00,0x00,0x00,
      0x87,0x02,0x54,0x04,0x08,0x00, /* write wait */
   };
   const uint8_t suffix[] = {
      0x07,0x02,0x54,0x0c,0x02,0x00, /* class12 release */
      0x0e,0x00,0x00,0x00,
   };
   ASSERT_EQ(bin.size(), 52u);
   EXPECT_TRUE(std::equal(std::begin(prefix), std::end(prefix), bin.begin()));
   EXPECT_TRUE(std::equal(std::begin(suffix), std::end(suffix), bin.end()-sizeof(suffix)));
   /* The intervening store consumes the actual outstanding tile-load slot. */
   uint8_t store[AGX3_TILE_MAX_LENGTH], len = 0;
   ASSERT_TRUE(agx3_tile_store(AGX_ISA_ENCODING_AGX3B, 0, AGX3_TILE_W8,
                               15, 0, AGX3_TILE_COV_1X, 1, store, &len));
   ASSERT_EQ(len, 12u);
   EXPECT_TRUE(std::equal(store, store+len, bin.begin()+sizeof(prefix)));
}

TEST_F(AGX3PackBinary, G17PAlphaProtocolDoesNotBroadenFormatOffsetOrControlFlow)
{
   for (unsigned axis = 0; axis < 3; ++axis) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_block *first = agx_start_block(b->shader);
      const enum agx_format format = axis == 0 ? AGX_FORMAT_S8NORM
                                               : AGX_FORMAT_U8NORM;
      const unsigned offset = axis == 1 ? 16 : 0;
      agx_index packed = agx_register(0, AGX_SIZE_16);
      packed.channels_m1 = 1;
      agx_wait_pix(b, 8);
      agx_ld_tile_to(b, packed, agx_immediate(AGX_ALL_SAMPLES), agx_null(),
                     format, 15, offset, false);
      if (axis == 2) {
         agx_block *second = agx_test_block(b->shader);
         agx_block_add_successor(first, second);
         b->cursor = agx_after_block(second);
      }
      agx_wait_pix(b, 4);
      agx_st_tile(b, packed, agx_immediate(AGX_ALL_SAMPLES), agx_null(),
                  format, 15, offset, false);
      agx_stop(b);
      const auto bin = pack(b);
      const uint8_t entry[] = {0x87,0x02,0x54,0x00,0x06,0x00};
      const uint8_t release[] = {0x07,0x02,0x54,0x0c,0x02,0x00};
      EXPECT_EQ(bin.size(), 40u) << axis;
      EXPECT_EQ(std::search(bin.begin(),bin.end(),std::begin(entry),std::end(entry)),bin.end()) << axis;
      EXPECT_EQ(std::search(bin.begin(),bin.end(),std::begin(release),std::end(release)),bin.end()) << axis;
   }
}

TEST_F(AGX3PackBinary, G17PAlphaProtocolPreservesConstantColourBracket)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_index packed = agx_register(0, AGX_SIZE_16);
   packed.channels_m1 = 1;
   agx_wait_pix(b, 0xc);
   agx_st_tile(b, packed, agx_immediate(AGX_ALL_SAMPLES), agx_null(),
               AGX_FORMAT_U8NORM, 15, 0, false);
   agx_stop(b);
   const uint8_t exact[] = {
      0x87,0x02,0x54,0x00,0x06,0x00,
      0x87,0x02,0x54,0x0c,0x08,0x00,
      0xe7,0x06,0x54,0x00,0x00,0x00,0x01,0x4e,0x00,0x00,0x00,0x00,
      0x07,0x02,0x54,0x0c,0x02,0x00,
      0x0e,0x00,0x00,0x00,
   };
   EXPECT_EQ(pack(b), std::vector<uint8_t>(std::begin(exact),std::end(exact)));
}

TEST_F(AGX3PackBinary, FragmentInterpolationStartsOnce)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_interp_agx3_to(b, agx_register(0, AGX_SIZE_32), agx_immediate(0),
                        agx_immediate(0), AGX_INTERPOLATION_CENTER, true);
   agx_interp_agx3_to(b, agx_register(2, AGX_SIZE_32), agx_immediate(1),
                        agx_immediate(0), AGX_INTERPOLATION_CENTER, true);
   agx_stop(b);
   const std::vector<uint8_t> want = {
      0x2f,0x0d,0x54,0,0x03,0,0,0x02,0x10,0,
      0x2f,0x05,0x54,2,0x03,2,0,0x02,0x10,0,0x0e,0,0,0,
   };
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, ProjectiveMultiplyPreservesCoefficientAndHints)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index x = agx_register(0, AGX_SIZE_32);
   agx_index y = agx_register(4, AGX_SIZE_32);
   agx_index recip = agx_register(2, AGX_SIZE_32);
   x.discard = y.discard = true;
   recip.cache = true;
   agx_persp_mul_agx3_to(b, agx_register(0, AGX_SIZE_32), x, recip, 1);
   recip.cache = false;
   recip.discard = true;
   agx_persp_mul_agx3_to(b, agx_register(2, AGX_SIZE_32), y, recip, 2);
   agx_stop(b);
   /* Exact consecutive native gradient instructions: coefficient tags1/2. */
   const std::vector<uint8_t> want = {
      0x09,0x01,0x2f,0x83,0x00,0x01,0x00,0x00,
      0x19,0x05,0x3f,0x03,0x00,0x02,0x00,0x00,0x0e,0,0,0,
   };
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, ProjectiveMultiplyRefusesUnsupportedForms)
{
   for (unsigned mode = 0; mode < 8; ++mode) {
      agx_builder *b = builder(mode == 0 ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B);
      agx_index raw = agx_register(0, AGX_SIZE_32);
      agx_index recip = agx_register(2, AGX_SIZE_32);
      if (mode == 1) raw.neg = true;
      if (mode == 2) raw = agx_uniform(0, AGX_SIZE_32);
      if (mode == 3) recip.size = AGX_SIZE_16;
      if (mode == 4) raw.cache = raw.discard = true;
      if (mode == 5) raw.kill = true;
      agx_instr *I = agx_persp_mul_agx3_to(b, agx_register(4, AGX_SIZE_32),
                                         raw, recip, mode == 6 ? 128 : 1);
      I->saturate = mode == 7;
      EXPECT_TRUE(pack(b, false).empty());
   }
}

TEST_F(AGX3PackBinary, FourPositionProtocolRemainsByteExact)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_VERTEX;
   agx_index r0 = agx_register(0, AGX_SIZE_32);
   agx_index xy = r0;
   xy.channels_m1 = 1;
   agx_index idx = r0;
   idx.abs = true;
   idx.kill = true;
   agx_get_sr_to(b, r0, AGX_SR_VERTEX_ID);
   agx_device_load_to(b, xy, agx_uniform(0, AGX_SIZE_64), idx,
                        AGX_FORMAT_I32, 3, 1, false);
   for (unsigned component = 0; component < 4; ++component) {
      if (component >= 2)
         agx_mov_imm_to(b, r0, component == 2 ? 0 : 0x3f800000);
      agx_index value = agx_register(component == 1 ? 2 : 0, AGX_SIZE_32);
      value.discard = true;
      agx_st_vary(b, agx_immediate(component), value);
   }
   agx_stop(b);
   /* Qualified VS108 recording, excluding its 8-byte linked prolog. */
   const std::vector<uint8_t> want = {
      0x0c,0xdd,0x10,0x06,0x67,0x10,0x44,0x00,0x00,0x80,0x20,0x00,
      0x59,0x01,0x00,0x40,0x48,0x00,0x0b,0x00,0x21,0xc4,0x1b,0x03,
      0x27,0x00,0x42,0x00,0x00,0x04,0x00,0x00,0x0b,0x00,0x00,0x00,
      0x2b,0x01,0x27,0x00,0x42,0x00,0x00,0x08,0x00,0x00,0x0c,0x80,
      0x02,0x3e,0x00,0x00,0x00,0x0c,0x3b,0x01,0x27,0x00,0x42,0x00,
      0x00,0x0c,0x00,0x00,0x57,0x16,0x54,0x00,0x00,0x40,0x48,0x00,
      0x57,0x26,0x54,0x02,0x20,0x40,0x49,0x00,0x57,0x46,0x54,0x04,
      0x40,0x40,0x4a,0x00,0x57,0x86,0x54,0x06,0x60,0x40,0x4b,0x00,
      0x0e,0x00,0x00,0x00,
   };
   EXPECT_EQ(pack(b), want);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 8u);
}

TEST_F(AGX3PackBinary, SixVertexOutputsRetireSourcesBeforeReuseAndWriteInOrder)
{
   /* Deliberately reuse r0 after every store, with the failing gradient's
    * output order. A deferred COPY must finish its source read before the
    * next MOV_IMM overwrites r0. */
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_VERTEX;
   const unsigned order[] = {3, 2, 1, 5, 0, 4};
   std::vector<uint8_t> want;
   for (unsigned component : order) {
      agx_mov_imm_to(b, agx_register(0, AGX_SIZE_32), 0x3f000000 + component);
      agx_st_vary(b, agx_immediate(component), agx_register(0, AGX_SIZE_32));
      struct agx3_encoded_ldimm ld;
      ASSERT_TRUE(agx3_pack_ldimm_u32(AGX_ISA_ENCODING_AGX3B, 0,
                                     0x3f000000 + component, &ld));
      append(want, ld.bytes, ld.length);
      uint8_t bytes[AGX3_ALU_MAX_LENGTH];
      uint8_t len = 0;
      ASSERT_TRUE(agx3_stage_copy_source(AGX_ISA_ENCODING_AGX3B,
                     component * 2, 0, false, component + 1, 0, bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_alu_compose(AGX_ISA_ENCODING_AGX3B, AGX3_ALU_BAND,
                     0, 0, 0, 0, 0, 1u << component, bytes, &len));
      append(want, bytes, len);
   }
   for (unsigned component = 0; component < 6; ++component) {
      uint8_t bytes[AGX3_VARY_MAX_LENGTH];
      uint8_t len = 0;
      ASSERT_TRUE(agx3_vary_uvsw(AGX_ISA_ENCODING_AGX3B, component * 2,
                     component, component + 1, true, 1u << component,
                     bytes, &len));
      append(want, bytes, len);
   }
   agx_stop(b);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
   EXPECT_TRUE(b->shader->agx3_staging_known);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 12u);
}

TEST_F(AGX3PackBinary, SixVertexOutputsDrainUnrelatedSlotsBeforeInjectedCopy)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_VERTEX;
   /* Two live async results unrelated to r0, including a slot later used
    * by output0. Both must retire before the injected r0-preserving wait. */
   agx_get_sr_to(b, agx_register(2, AGX_SIZE_32), AGX_SR_VERTEX_ID);
   agx_get_sr_to(b, agx_register(4, AGX_SIZE_32), AGX_SR_VERTEX_ID);
   agx_mov_imm_to(b, agx_register(0, AGX_SIZE_32), 0x3f800000);
   const unsigned order[] = {3, 2, 1, 5, 0, 4};
   for (unsigned component : order)
      agx_st_vary(b, agx_immediate(component), agx_register(0, AGX_SIZE_32));
   agx_stop(b);
   const auto got = pack(b);
   uint8_t bytes[AGX3_STAGE_MAX_LENGTH];
   uint8_t len = 0;
   ASSERT_TRUE(agx3_stage_copy_source(AGX_ISA_ENCODING_AGX3B, 6, 0,
                                      false, 4, 3, bytes, &len));
   const auto at = std::search(got.begin(), got.end(), bytes, bytes + len);
   ASSERT_NE(at, got.end());
   uint8_t wait_bytes[AGX3_ALU_MAX_LENGTH], wait_len = 0;
   ASSERT_TRUE(agx3_alu_compose(AGX_ISA_ENCODING_AGX3B, AGX3_ALU_BAND,
                  0, 0, 0, 0, 0, 8, wait_bytes, &wait_len));
   ASSERT_GE(got.end() - at, len + wait_len);
   EXPECT_EQ(memcmp(&*(at + len), wait_bytes, wait_len), 0);
}

TEST_F(AGX3PackBinary, SixVertexOutputsRefuseUnqualifiedGraphs)
{
   for (unsigned mode = 0; mode < 7; ++mode) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_VERTEX;
      for (unsigned component = 0; component < 6; ++component) {
         if (mode == 0 && component == 0)
            continue; /* Missing position but user outputs remain. */
         unsigned index = mode == 1 && component == 0 ? 1 : component;
         if (mode == 2 && component == 5)
            index = 6;
         agx_st_vary(b, agx_immediate(index),
                     agx_register(0, mode == 3 ? AGX_SIZE_16 : AGX_SIZE_32));
      }
      if (mode == 4)
         agx_begin_cf(b);
      if (mode == 5)
         agx_test_block(b->shader);
      if (mode == 6)
         agx_device_store(b, agx_register(0, AGX_SIZE_32),
                          agx_uniform(0, AGX_SIZE_64), agx_immediate(0),
                          AGX_FORMAT_I32, 1, 2, false);
      agx_stop(b);
      EXPECT_TRUE(pack(b, false).empty()) << "mode=" << mode;
   }
}

TEST_F(AGX3PackBinary, ProjectiveReciprocalMatchesNativeOperandForm)
{
   const struct {
      unsigned dest;
      bool cache;
      uint8_t bytes[10];
   } witnesses[] = {
      {0, false, {0xaf,0,0x54,0,3,0,0,0x48,0x20,1}},
      {2, true,  {0xaf,0,0x54,2,3,0,2,0x48,0x20,1}},
      {4, true,  {0xaf,0,0x54,4,3,0,2,0x48,0x20,1}},
   };
   for (const auto &v : witnesses) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_index rhw = agx_register(0, AGX_SIZE_32);
      rhw.cache = v.cache;
      agx_persp_rcp_agx3_to(b, agx_register(v.dest, AGX_SIZE_32), rhw, 0);
      agx_stop(b);
      std::vector<uint8_t> want(v.bytes, v.bytes + 10);
      want.insert(want.end(), {0x0e, 0, 0, 0});
      EXPECT_EQ(pack(b), want);
   }
   /* Ordinary reciprocal keeps its previously qualified encoding. */
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_rcp_to(b, agx_register(0, AGX_SIZE_32), agx_register(0, AGX_SIZE_32));
   agx_stop(b);
   const std::vector<uint8_t> ordinary = {
      0xaf,0,0x56,0,2,0,0x10,0x48,0x20,0, 0x0e,0,0,0,
   };
   EXPECT_EQ(pack(b), ordinary);
}

TEST_F(AGX3PackBinary, ProjectiveReciprocalCoefficientAndWaitBits)
{
   /* Independent physical extraction catches coefficient/source overlap and
    * accidental retention of ordinary RCP's unconditional slot6 wait. */
   for (unsigned dest : {0u, 2u, 126u, 128u, 190u}) {
      for (unsigned rhw : {0u, 2u, 126u, 128u, 190u}) {
         for (unsigned coef : {0u, 1u, 4u, 63u, 127u}) {
            for (unsigned wait = 0; wait < 64; ++wait) {
               struct agx3_persp_rcp_desc d = {};
               d.dest = dest;
               d.rhw = rhw;
               d.coef_index = coef;
               d.rhw_cache = (dest != rhw);
               d.wait_mask = wait;
               uint8_t bytes[AGX3_FLTNORM_MAX_LENGTH], len = 0;
               ASSERT_TRUE(agx3_persp_rcp_encode(AGX_ISA_ENCODING_AGX3B,
                                                &d, bytes, &len));
               ASSERT_EQ(len, 10u);
               EXPECT_EQ(bytes[3], dest);
               EXPECT_EQ(((bytes[5] >> 1) & 0x7f) | ((bytes[6] & 1) << 7), rhw);
               EXPECT_EQ(((bytes[6] >> 2) | ((bytes[7] & 1) << 6)), coef);
               EXPECT_EQ((bytes[6] >> 1) & 1, unsigned(d.rhw_cache));
               EXPECT_EQ((bytes[1] >> 4) | ((bytes[2] & 3) << 4), wait);
               EXPECT_EQ(bytes[9] & 1, 1u);
               EXPECT_EQ(bytes[4] & 1, 1u);
            }
         }
      }
   }
}

TEST_F(AGX3PackBinary, ProjectiveReciprocalRefusesUnsupportedForms)
{
   for (unsigned mode = 0; mode < 8; ++mode) {
      agx_builder *b = builder(mode == 0 ? AGX_ISA_ENCODING_AGX3C :
                                         AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_index rhw = agx_register(mode == 1 ? 192 : 0, AGX_SIZE_32);
      rhw.abs = mode == 2;
      rhw.neg = mode == 3;
      rhw.channels_m1 = mode == 4 ? 1 : 0;
      rhw.cache = rhw.discard = mode == 5;
      agx_instr *I = agx_persp_rcp_agx3_to(b,
         agx_register(0, AGX_SIZE_32), rhw, mode == 6 ? 128 : 0);
      I->saturate = mode == 7;
      agx_stop(b);
      EXPECT_TRUE(pack(b, false).empty()) << "mode=" << mode;
   }
   struct agx3_persp_rcp_desc d = {};
   uint8_t bytes[AGX3_FLTNORM_MAX_LENGTH], len;
   d.wait_mask = 0x40;
   EXPECT_FALSE(agx3_persp_rcp_encode(AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
}

TEST_F(AGX3PackBinary, ProjectiveRhwMatchesNativeAndKeepsUserInterp)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_index r0 = agx_register(0, AGX_SIZE_32);
   agx_rhw_interp_agx3_to(b, r0, agx_immediate(0), agx_immediate(0),
                         AGX_INTERPOLATION_CENTER);
   agx_persp_rcp_agx3_to(b, r0, r0, 0);
   agx_interp_agx3_to(b, agx_register(2, AGX_SIZE_32), agx_immediate(1),
                     agx_immediate(0), AGX_INTERPOLATION_CENTER, true);
   agx_stop(b);
   /* Native smooth_scalar: private RHW f54=1, first f63=1; user f54=f63=0. */
   const std::vector<uint8_t> want = {
      0x2f,0x0d,0x54,0,3,0,4,2,0x10,0,
      0xaf,0,0x54,0,3,0,0,0x48,0x20,1,
      0x2f,0x05,0x54,2,3,2,0,2,0x10,0, 0x0e,0,0,0,
   };
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, ProjectiveRhwModeIndependentOfFirstAndRcpDestination)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_interp_agx3_to(b, agx_register(4, AGX_SIZE_32), agx_immediate(1),
                     agx_immediate(0), AGX_INTERPOLATION_CENTER, false);
   agx_index r0 = agx_register(0, AGX_SIZE_32);
   agx_rhw_interp_agx3_to(b, r0, agx_immediate(0), agx_immediate(0),
                         AGX_INTERPOLATION_CENTER);
   r0.cache = true;
   agx_persp_rcp_agx3_to(b, agx_register(2, AGX_SIZE_32), r0, 0);
   agx_stop(b);
   auto bytes = pack(b);
   ASSERT_EQ(bytes.size(), 34u);
   EXPECT_EQ(bytes[1] & 8, 8u);
   EXPECT_EQ(bytes[6] & 4, 0u);
   EXPECT_EQ(bytes[11] & 8, 0u);
   EXPECT_EQ(bytes[16] & 4, 4u);
   const std::vector<uint8_t> native_rcp =
      {0xaf,0,0x54,2,3,0,2,0x48,0x20,1};
   EXPECT_EQ(std::vector<uint8_t>(bytes.begin()+20,bytes.begin()+30), native_rcp);
}

TEST_F(AGX3PackBinary, ProjectiveRhwRefusesUnprovenDomainsAndSharedValues)
{
   for (unsigned mode = 0; mode < 10; ++mode) {
      agx_builder *b = builder(mode == 0 ? AGX_ISA_ENCODING_AGX3C :
                                         AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = mode == 1 ? MESA_SHADER_VERTEX : MESA_SHADER_FRAGMENT;
      agx_index r0 = agx_register(0, AGX_SIZE_32);
      agx_instr *producer = agx_rhw_interp_agx3_to(b, r0,
         agx_immediate(mode == 2 ? 128 : 0), agx_immediate(mode == 3 ? 1 : 0),
         mode == 4 ? AGX_INTERPOLATION_CENTROID : AGX_INTERPOLATION_CENTER);
      producer->saturate = mode == 5;
      if (mode != 6)
         agx_persp_rcp_agx3_to(b, agx_register(2, AGX_SIZE_32), r0,
                              mode == 7 ? 1 : 0);
      if (mode == 8)
         agx_fmul_to(b, agx_register(4, AGX_SIZE_32), r0, r0);
      if (mode == 9)
         agx_mov_to(b, agx_register(4, AGX_SIZE_16),
                    agx_register(1, AGX_SIZE_16));
      agx_stop(b);
      EXPECT_TRUE(pack(b, false).empty()) << "mode=" << mode;
   }
}

TEST_F(AGX3PackBinary, ProjectiveRhwCseCannotMergeOrdinaryPositionW)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index ordinary = agx_temp(b->shader, AGX_SIZE_32);
   agx_index private_rhw = agx_temp(b->shader, AGX_SIZE_32);
   agx_interp_agx3_to(b, ordinary, agx_immediate(0), agx_immediate(0),
                     AGX_INTERPOLATION_CENTER, true);
   agx_rhw_interp_agx3_to(b, private_rhw, agx_immediate(0), agx_immediate(0),
                         AGX_INTERPOLATION_CENTER);
   agx_instr *rcp = agx_persp_rcp_agx3_to(b, agx_temp(b->shader, AGX_SIZE_32),
                                        private_rhw, 0);
   agx_opt_cse(b->shader);
   EXPECT_EQ(rcp->src[0].value, private_rhw.value);
   EXPECT_NE(rcp->src[0].value, ordinary.value);
}

/* Native scene-crescent / scene-ring place their write fence after both
 * FLTNORMs, immediately before LSTCOL. Exercise actual packing, including an
 * asynchronous producer whose wait must remain on the first FLTNORM.
 */
TEST_F(AGX3PackBinary, G17PWriteWaitFollowsPackingAndPreservesScoreboard)
{
   for (bool blend : {false, true}) {
      for (bool async : {false, true}) {
         SCOPED_TRACE(blend);
         SCOPED_TRACE(async);
         const auto enc = AGX_ISA_ENCODING_AGX3B;
         agx_builder *b = builder(enc);
         b->shader->stage = MESA_SHADER_FRAGMENT;
         std::vector<uint8_t> want;
         uint8_t bytes[32], len;
         const uint8_t entry[] = {0x87,0x02,0x54,0,6,0};
         const uint8_t read[] = {0x87,0x02,0x54,8,8,0};
         const uint8_t release_end[] = {7,2,0x54,12,2,0,14,0,0,0};
         agx_index packed = agx_register(0, AGX_SIZE_16);
         packed.channels_m1 = 1;
         if (async) {
            agx_scs_load_agx3_to(b, agx_register(4, AGX_SIZE_32), 4,
                                AGX_FORMAT_I32);
            struct agx3_scs_desc d = {};
            d.data = 4; d.width = AGX3_MEM_W32;
            d.byte_offset = 4; d.slot = 1;
            ASSERT_TRUE(agx3_scs_encode_load(enc, &d, bytes, &len));
            append(want, bytes, len);
         }
         if (blend) {
            append(want, entry, sizeof(entry));
            append(want, read, sizeof(read));
            agx_wait_pix(b, 8);
            agx_index loaded = agx_register(12, AGX_SIZE_16);
            loaded.channels_m1 = 1;
            agx_ld_tile_to(b, loaded, agx_immediate(AGX_ALL_SAMPLES),
                          agx_null(), AGX_FORMAT_U8NORM, 15, 0, false);
            ASSERT_TRUE(agx3_tile_load(enc, 12, AGX3_TILE_W8, async ? 2 : 1, 15, 0,
               AGX3_TILE_COV_1X, 0, true, true, bytes, &len));
            append(want, bytes, len);
            agx_index unpacked = agx_register(8, AGX_SIZE_32);
            unpacked.channels_m1 = 1;
            agx_normflt_agx3_to(b, unpacked, agx_register(12, AGX_SIZE_16),
                               AGX_FORMAT_U8NORM);
            struct agx3_normflt_desc d = {};
            d.dest = 8; d.dest_hint = true; d.src = 12;
            d.two_lanes = true; d.wait_mask = async ? 2 : 1;
            ASSERT_TRUE(agx3_normflt_encode(enc, &d, bytes, &len));
            append(want, bytes, len);
         }
         agx_wait_pix(b, blend ? 4 : 12);
         for (unsigned i = 0; i < 2; ++i) {
            agx_fltnorm_agx3_to(b, agx_register(i, AGX_SIZE_16),
               agx_register(4 + 4*i, AGX_SIZE_32),
               agx_register(6 + 4*i, AGX_SIZE_32), AGX_FORMAT_U8NORM);
            struct agx3_fltnorm_desc d = {};
            d.dest = i; d.two_lanes = true;
            d.src[0] = {AGX3_FLTNORM_SRC_REGISTER, uint16_t(4+4*i), false};
            d.src[1] = {AGX3_FLTNORM_SRC_REGISTER, uint16_t(6+4*i), false};
            d.wait_mask = async && i == 0 ? 1 : 0;
            d.first_instruction = !blend && !async && i == 0;
            ASSERT_TRUE(agx3_fltnorm_encode(enc, &d, bytes, &len));
            append(want, bytes, len);
         }
         if (!blend)
            append(want, entry, sizeof(entry));
         const uint8_t write[] = {0x87,2,0x54,uint8_t(blend ? 4 : 12),8,0};
         append(want, write, sizeof(write));
         agx_st_tile(b, packed, agx_immediate(AGX_ALL_SAMPLES), agx_null(),
                     AGX_FORMAT_U8NORM, 15, 0, false);
         ASSERT_TRUE(agx3_tile_store(enc, 0, AGX3_TILE_W8, 15, 0,
                                    AGX3_TILE_COV_1X, 0, bytes, &len));
         append(want, bytes, len);
         append(want, release_end, sizeof(release_end));
         agx_stop(b);
         EXPECT_EQ(pack(b), want);
      }
   }
}

TEST_F(AGX3PackBinary, G17PWriteWaitDoesNotCrossUnprovenOperations)
{
   for (unsigned axis = 0; axis < 6; ++axis) {
      SCOPED_TRACE(axis);
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_block *first = agx_start_block(b->shader);
      if (axis == 5)
         agx_begin_cf(b);
      agx_wait_pix(b, 12);
      if (axis == 2) {
         agx_block *second = agx_test_block(b->shader);
         agx_block_add_successor(first, second);
         b->cursor = agx_after_block(second);
      } else if (axis == 3) {
         agx_fadd_to(b, agx_register(4, AGX_SIZE_32),
                     agx_register(6, AGX_SIZE_32), agx_register(8, AGX_SIZE_32));
      } else if (axis == 4) {
         agx_signal_pix(b, 12);
      }
      const auto format = axis == 0 ? AGX_FORMAT_S8NORM : AGX_FORMAT_U8NORM;
      agx_fltnorm_agx3_to(b, agx_register(0, AGX_SIZE_16),
                         agx_immediate(32), agx_immediate(48), format);
      agx_index packed = agx_register(0, AGX_SIZE_16);
      packed.channels_m1 = 1;
      agx_st_tile(b, packed, agx_immediate(AGX_ALL_SAMPLES), agx_null(),
                  format, 15, axis == 1 ? 16 : 0, false);
      agx_stop(b);
      const auto bin = pack(b);
      const uint8_t original_prefix[] = {
         0x87,2,0x54,0,6,0, 0x87,2,0x54,12,8,0,
      };
      ASSERT_GE(bin.size(), sizeof(original_prefix));
      EXPECT_TRUE(std::equal(std::begin(original_prefix),
                             std::end(original_prefix), bin.begin())) << axis;
   }
}

TEST_F(AGX3PackBinary, OperandlessPixelWaitDoesNotRetireAsyncRegisters)
{
   /* agx3_emit_pixel requires zero src/dest operands; WAIT_PIX is synchronous.
    * Thus its scheduler mask is zero even with all six slots outstanding.
    * The real first consumer must retain all six waits after fence deferral.
    */
   struct agx3_slot_instr instrs[8] = {};
   struct agx3_slot_result res[8] = {};
   for (unsigned i = 0; i < 6; ++i) {
      instrs[i].is_async = true;
      instrs[i].nr_dsts = 1;
      instrs[i].dst[0] = i;
      instrs[7].src[i] = i;
   }
   instrs[7].nr_srcs = 6;
   ASSERT_TRUE(agx3_slot_assign(instrs, 8, 6, res));
   EXPECT_EQ(res[6].slot, 0);
   EXPECT_EQ(res[6].wait_mask, 0);
   EXPECT_EQ(res[7].wait_mask, 0x3f);
}

TEST_F(AGX3PackBinary, G17PColourPackingKeepsNativeUniformSources)
{
   for (unsigned base : {0u, 16u, 256u, 508u}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_instr *I = agx_fltnorm_agx3_to(b, agx_register(0, AGX_SIZE_16),
         agx_uniform(base, AGX_SIZE_32), agx_uniform(base + 2, AGX_SIZE_32),
         AGX_FORMAT_U8NORM);
      agx_stop(b);
      agx_lower_uniform_sources(b->shader);
      EXPECT_EQ(I->src[0].type, AGX_INDEX_UNIFORM);
      EXPECT_EQ(I->src[1].type, AGX_INDEX_UNIFORM);
      struct agx3_fltnorm_desc d = {};
      d.two_lanes = true;
      d.src[0].kind = d.src[1].kind = AGX3_FLTNORM_SRC_UNIFORM;
      d.src[0].value = base; d.src[1].value = base + 2;
      uint8_t encoded[AGX3_FLTNORM_MAX_LENGTH], len = 0;
      ASSERT_TRUE(agx3_fltnorm_encode(AGX_ISA_ENCODING_AGX3B, &d, encoded, &len));
      std::vector<uint8_t> expected(encoded, encoded + len);
      const uint8_t stop[] = {14, 0, 0, 0};
      append(expected, stop, sizeof(stop));
      EXPECT_EQ(pack(b), expected);
   }
   for (unsigned value : {1u, 511u, 512u})
      EXPECT_FALSE(agx_instr_accepts_uniform(AGX_ISA_ENCODING_AGX3B,
         AGX_OPCODE_FLTNORM_AGX3, 0, value, AGX_SIZE_32));
   for (auto size : {AGX_SIZE_16, AGX_SIZE_64})
      EXPECT_FALSE(agx_instr_accepts_uniform(AGX_ISA_ENCODING_AGX3B,
         AGX_OPCODE_FLTNORM_AGX3, 0, 16, size));
   EXPECT_FALSE(agx_instr_accepts_uniform(AGX_ISA_ENCODING_AGX3B,
      AGX_OPCODE_FLTNORM_AGX3, 2, 16, AGX_SIZE_32));
   EXPECT_FALSE(agx_instr_accepts_uniform(AGX_ISA_ENCODING_AGX3B,
      AGX_OPCODE_IADD, 0, 16, AGX_SIZE_32));
}

TEST_F(AGX3PackBinary, G17PFltnormStartMatchesNativeF32Uniforms)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   /* Actual native uniform-f32.nat, constant float4 buffer input: uniform
    * half indices 4,6,8,10 with f24=f41=2 (32-bit). */
   for (unsigned i = 0; i < 2; ++i)
      agx_fltnorm_agx3_to(b, agx_register(i, AGX_SIZE_16),
         agx_uniform(4 + 4*i, AGX_SIZE_32),
         agx_uniform(6 + 4*i, AGX_SIZE_32), AGX_FORMAT_U8NORM);
   agx_wait_pix(b, 12);
   agx_index packed = agx_register(0, AGX_SIZE_16);
   packed.channels_m1 = 1;
   agx_st_tile(b, packed, agx_immediate(AGX_ALL_SAMPLES), agx_null(),
               AGX_FORMAT_U8NORM, 15, 0, false);
   agx_stop(b);
   const std::vector<uint8_t> native = {
      0x97,0x0c,0x54,0,2,8,0x18,0xd0,5,0xc8,
      0x97,4,0x54,1,2,0x10,0x28,0xd0,5,0xc8,
      0x87,2,0x54,0,6,0, 0x87,2,0x54,12,8,0,
      0xe7,6,0x54,0,0,0,1,0x4e,0,0,0,0,
      7,2,0x54,12,2,0, 14,0,0,0,
   };
   EXPECT_EQ(pack(b), native);
}

TEST_F(AGX3PackBinary, G17PFltnormStartIsPositionNotFirstUniformRead)
{
   for (bool preceding : {false, true}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      if (preceding)
         agx_fadd_to(b, agx_register(4, AGX_SIZE_32),
                    agx_register(6, AGX_SIZE_32), agx_register(8, AGX_SIZE_32));
      agx_fltnorm_agx3_to(b, agx_register(0, AGX_SIZE_16),
                         agx_immediate(16), agx_immediate(32), AGX_FORMAT_U8NORM);
      agx_fltnorm_agx3_to(b, agx_register(1, AGX_SIZE_16),
                         agx_uniform(8, AGX_SIZE_32),
                         agx_uniform(10, AGX_SIZE_32), AGX_FORMAT_U8NORM);
      agx_stop(b);
      const auto bin = pack(b);
      const uint8_t immediate[] = {
         0x97,uint8_t(preceding ? 4 : 12),0x54,0,2,0x20,0x80,0x50,4,0xc8,
      };
      const uint8_t later[] = {0x97,4,0x54,1,2,0x10,0x28,0xd0,5,0xc8};
      /* Native immediate-only first FLTNORM sets f68; native position-derived
       * arithmetic before the first uniform FLTNORM leaves it clear. */
      const auto at = std::search(bin.begin(),bin.end(),std::begin(immediate),
                                  std::end(immediate));
      ASSERT_NE(at, bin.end());
      EXPECT_TRUE(std::equal(std::begin(later),std::end(later),at+10));
   }
   struct agx3_fltnorm_desc d = {};
   d.src[0] = {AGX3_FLTNORM_SRC_IMMEDIATE, 16, false};
   d.first_instruction = true;
   uint8_t bytes[AGX3_FLTNORM_MAX_LENGTH], len;
   EXPECT_FALSE(agx3_fltnorm_encode(AGX_ISA_ENCODING_AGX3C, &d, bytes, &len));
}

TEST_F(AGX3PackBinary, G17PFltnormStartRelocatesAcrossLinkedParts)
{
   for (unsigned main_bytes : {0u, 12u}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_fltnorm_agx3_to(b, agx_register(0, AGX_SIZE_16),
                         agx_uniform(4, AGX_SIZE_32),
                         agx_uniform(6, AGX_SIZE_32), AGX_FORMAT_U8NORM);
      agx_stop(b);
      const auto epilog = pack(b);
      ASSERT_TRUE(b->shader->agx3_fltnorm_start);
      /* Real crescent main's three 4-byte uniform COPY exports. */
      const uint8_t main[] = {0x4b,0x10,1,8,0x5b,0x14,1,8,0x6b,0x12,1,8};
      std::vector<uint8_t> linked(main, main+main_bytes);
      linked.insert(linked.end(),epilog.begin(),epilog.end());
      ASSERT_TRUE(agx3_fltnorm_relocate_start(linked.data()+main_bytes,
                  epilog.size(),b->shader->agx3_fltnorm_start,main_bytes));
      EXPECT_EQ(linked[main_bytes+1], main_bytes ? 4 : 12);
      EXPECT_TRUE(std::equal(main,main+main_bytes,linked.begin()));
      EXPECT_EQ(epilog[1], 12); /* cached part remains reusable */
      EXPECT_TRUE(std::equal(epilog.begin()+2,epilog.end(),
                             linked.begin()+main_bytes+2));
   }
   uint8_t invalid[10] = {0x97,4,0x54};
   EXPECT_FALSE(agx3_fltnorm_relocate_start(invalid,10,true,12));
   invalid[1] = 12;
   EXPECT_FALSE(agx3_fltnorm_relocate_start(invalid,9,true,12));
   /* No provenance means no scanning or rewriting, even for matching bytes. */
   EXPECT_TRUE(agx3_fltnorm_relocate_start(invalid,10,false,12));
   EXPECT_EQ(invalid[1],12);
}

TEST_F(AGX3PackBinary, G17PColourFusionUsesCompilerProvenanceAndNativeUniforms)
{
   for (unsigned copies = 1; copies <= 4; ++copies) {
      agx_builder *main = builder(AGX_ISA_ENCODING_AGX3B);
      main->shader->stage = MESA_SHADER_FRAGMENT;
      key.no_stop = true; key.secondary = false;
      for (unsigned i = 0; i < copies; ++i) {
         agx_index reg = agx_register(8+2*i, AGX_SIZE_32);
         agx_bitop_to(main,reg,agx_uniform(4+2*i,AGX_SIZE_32),agx_immediate(0),0xa);
         agx_export(main,reg,reg.value);
      }
      agx_stop(main);
      const auto main_bin = pack(main);
      EXPECT_EQ(main_bin.size(),4*copies);
      const auto main_info = main->shader->agx3_colour_link;
      ASSERT_TRUE(main_info.copy_only);
      ASSERT_EQ(main_info.copy_count,copies);

      agx_builder *epi = builder(AGX_ISA_ENCODING_AGX3B);
      epi->shader->stage = MESA_SHADER_FRAGMENT;
      key.no_stop = false; key.secondary = true;
      agx_wait_pix(epi,12);
      for (unsigned i = 0; i < 2; ++i)
         agx_fltnorm_agx3_to(epi,agx_register(i,AGX_SIZE_16),
            agx_register(8+2*((2*i)%copies),AGX_SIZE_32),
            agx_register(8+2*((2*i+1)%copies),AGX_SIZE_32),AGX_FORMAT_U8NORM);
      agx_index packed = agx_register(0,AGX_SIZE_16);
      packed.channels_m1 = 1;
      agx_st_tile(epi,packed,agx_immediate(AGX_ALL_SAMPLES),agx_null(),
                  AGX_FORMAT_U8NORM,15,0,false);
      agx_stop(epi);
      const auto epi_bin = pack(epi);
      const auto epi_info = epi->shader->agx3_colour_link;
      ASSERT_TRUE(epi_info.opaque_pack);
      uint8_t fused[AGX3_COLOUR_LINK_SIZE];
      ASSERT_TRUE(agx3_fuse_colour_parts(&main_info,&epi_info,fused));
      /* Four distinct exports produce the exact native float4-buffer shader. */
      if (copies == 4) {
         const uint8_t native[] = {
            0x97,12,0x54,0,2,8,0x18,0xd0,5,0xc8,
            0x97,4,0x54,1,2,0x10,0x28,0xd0,5,0xc8,
            0x87,2,0x54,0,6,0,0x87,2,0x54,12,8,0,
            0xe7,6,0x54,0,0,0,1,0x4e,0,0,0,0,
            7,2,0x54,12,2,0,14,0,0,0,
         };
         EXPECT_EQ(std::vector<uint8_t>(fused,fused+sizeof(fused)),
                   std::vector<uint8_t>(std::begin(native),std::end(native)));
      }
      EXPECT_EQ(main_bin.size(),4*copies); /* cached inputs remain immutable */
      EXPECT_EQ(epi_bin.size(),54u);
      for (unsigned mode = 0; mode < 8; ++mode) {
         auto bad_main = main_info;
         auto bad_epi = epi_info;
         switch (mode) {
         case 0: bad_main.copy_only = false; break;
         case 1: bad_main.copies[0].uniform |= 1; break;
         case 2: bad_epi.opaque_pack = false; break;
         case 3: bad_epi.packs[1].src[0].value = 250; break;
         case 4: bad_epi.packs[0].wait_mask = 1; break;
         case 5: bad_epi.packs[0].snorm = true; break;
         case 6: bad_epi.store_reg = bad_main.copies[0].reg;
                 bad_epi.packs[0].dest = bad_epi.store_reg;
                 bad_epi.packs[1].dest = bad_epi.store_reg+1; break;
         case 7: bad_main.copy_count = 0; break;
         }
         memset(fused,0xa5,sizeof(fused));
         EXPECT_FALSE(agx3_fuse_colour_parts(&bad_main,&bad_epi,fused)) << mode;
         EXPECT_TRUE(std::all_of(fused,fused+sizeof(fused),[](uint8_t v){return v==0xa5;}));
      }
   }
}

TEST_F(AGX3PackBinary, G17PColourFusionRejectsIncompleteMainProof)
{
   for (unsigned mode = 0; mode < 4; ++mode) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      key.no_stop = true; key.secondary = false;
      agx_index reg = agx_register(8,AGX_SIZE_32);
      agx_bitop_to(b,reg,agx_uniform(16,AGX_SIZE_32),agx_immediate(0),0xa);
      if (mode != 0)
         agx_export(b,reg,8);
      if (mode == 1)
         agx_bitop_to(b,reg,agx_uniform(18,AGX_SIZE_32),agx_immediate(0),0xa);
      if (mode == 2)
         agx_fadd_to(b,agx_register(10,AGX_SIZE_32),reg,reg);
      if (mode == 3)
         agx_test_block(b->shader);
      agx_stop(b);
      pack(b);
      EXPECT_FALSE(b->shader->agx3_colour_link.copy_only) << mode;
   }
}

TEST_F(AGX3PackBinary, G17PColourFusionAcceptsS8NormAndRejectsOtherChanges)
{
   for (unsigned mode = 0; mode < 4; ++mode) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      key.no_stop = false; key.secondary = true;
      if (mode == 0)
         agx_fadd_to(b,agx_register(8,AGX_SIZE_32),
                    agx_register(10,AGX_SIZE_32),agx_register(12,AGX_SIZE_32));
      if (mode == 1)
         agx_begin_cf(b);
      agx_wait_pix(b,12);
      for (unsigned i = 0; i < 2; ++i)
         agx_fltnorm_agx3_to(b,agx_register(i,AGX_SIZE_16),
                            agx_register(8,AGX_SIZE_32),agx_immediate(48),
                            mode == 2 ? AGX_FORMAT_S8NORM : AGX_FORMAT_U8NORM);
      agx_index packed = agx_register(0,AGX_SIZE_16);
      packed.channels_m1 = 1;
      agx_st_tile(b,packed,agx_immediate(AGX_ALL_SAMPLES),agx_null(),
                  mode == 2 ? AGX_FORMAT_S8NORM : AGX_FORMAT_U8NORM,
                  15,mode == 3 ? 16 : 0,false);
      agx_stop(b);
      pack(b);
      EXPECT_EQ(b->shader->agx3_colour_link.opaque_pack, mode == 2) << mode;
   }
}

TEST_F(AGX3PackBinary, G17PLinearCenterInterpolationMatchesNative)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_interp_agx3_to(b, agx_register(0, AGX_SIZE_32), agx_immediate(0),
                     agx_zero(), AGX_INTERPOLATION_CENTER, false);
   agx_interp_agx3_to(b, agx_register(2, AGX_SIZE_32), agx_immediate(1),
                     agx_zero(), AGX_INTERPOLATION_CENTER, false);
   agx_stop(b);
   /* Native linear_scalar start; linear_vector next scalar with r1/cf1.
    * No projective RHW marker, reciprocal, or multiply. */
   const std::vector<uint8_t> want = {
      0x2f,0x0d,0x54,0,3,0,0,2,0x10,0,
      0x2f,0x05,0x54,2,3,2,0,2,0x10,0,
      0x0e,0,0,0,
   };
   EXPECT_EQ(pack(b), want);
}


TEST_F(AGX3PackBinary, SixOutputTailSinkingPreservesValuesThroughRA)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_VERTEX;
   b->shader->nir = rzalloc(mem_ctx, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_VERTEX;
   const unsigned order[] = {3, 2, 1, 5, 0, 4};
   agx_instr *stores[6];
   for (unsigned c : order) {
      agx_index value = agx_mov_imm(b, 32, 0x3f000000 + c);
      stores[c] = agx_st_vary(b, agx_immediate(c), value);
   }
   agx_stop(b);
   agx_pressure_schedule(b->shader);
   ASSERT_TRUE(agx3_sink_vertex_outputs(b->shader));
   unsigned next = 0;
   agx_foreach_instr_global(b->shader, I) {
      if (I->op == AGX_OPCODE_ST_VARY) {
         ASSERT_LT(next, 6u);
         EXPECT_EQ(I, stores[next++]);
      } else if (next) {
         EXPECT_EQ(I->op, AGX_OPCODE_STOP);
      }
   }
   EXPECT_EQ(next, 6u);
   agx_ra(b->shader);
   /* Independently interpret the physical assignments: all six original
    * constants must survive until their terminal stores after actual RA. */
   uint32_t regs[256] = {0};
   std::vector<uint8_t> want;
   unsigned stored = 0;
   agx_foreach_instr_global(b->shader, I) {
      if (I->op == AGX_OPCODE_MOV_IMM) {
         regs[I->dest[0].value] = I->imm;
         struct agx3_encoded_ldimm ld;
         ASSERT_TRUE(agx3_pack_ldimm_u32(AGX_ISA_ENCODING_AGX3B,
                         I->dest[0].value, I->imm, &ld));
         append(want, ld.bytes, ld.length);
      } else if (I->op == AGX_OPCODE_ST_VARY) {
         const unsigned c = I->src[0].value;
         EXPECT_EQ(regs[I->src[1].value], 0x3f000000u + c);
         uint8_t bytes[AGX3_STAGE_MAX_LENGTH], len;
         ASSERT_TRUE(agx3_stage_copy_source(AGX_ISA_ENCODING_AGX3B,
                       c * 2, I->src[1].value, I->src[1].kill,
                       c + 1, 0, bytes, &len));
         append(want, bytes, len);
         stored++;
      } else {
         ASSERT_EQ(I->op, AGX_OPCODE_STOP);
      }
   }
   EXPECT_EQ(stored, 6u);
   for (unsigned c = 0; c < 6; ++c) {
      uint8_t bytes[AGX3_VARY_MAX_LENGTH], len;
      ASSERT_TRUE(agx3_vary_uvsw(AGX_ISA_ENCODING_AGX3B, c * 2, c,
                                c + 1, true, 1u << c, bytes, &len));
      append(want, bytes, len);
   }
   want.insert(want.end(), {14, 0, 0, 0});
   EXPECT_EQ(pack(b), want); /* No intervening band waits. */
   EXPECT_EQ(b->shader->agx3_staging_halfs, 12u);
}

TEST_F(AGX3PackBinary, SixTerminalOutputsDrainOriginalProducersOnce)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_VERTEX;
   agx_get_sr_to(b, agx_register(2, AGX_SIZE_32), AGX_SR_VERTEX_ID);
   agx_get_sr_to(b, agx_register(4, AGX_SIZE_32), AGX_SR_VERTEX_ID);
   for (unsigned c = 0; c < 6; ++c)
      agx_st_vary(b, agx_immediate(c),
                     agx_register(c % 2 ? 4 : 2, AGX_SIZE_32));
   agx_stop(b);
   auto got = pack(b);
   std::vector<uint8_t> tail;
   for (unsigned c = 0; c < 6; ++c) {
      uint8_t bytes[AGX3_STAGE_MAX_LENGTH], len;
      ASSERT_TRUE(agx3_stage_copy_source(AGX_ISA_ENCODING_AGX3B, c * 2,
                 c % 2 ? 4 : 2, false, c + 1, c == 0 ? 3 : 0, bytes, &len));
      append(tail, bytes, len);
   }
   for (unsigned c = 0; c < 6; ++c) {
      uint8_t bytes[AGX3_VARY_MAX_LENGTH], len;
      ASSERT_TRUE(agx3_vary_uvsw(AGX_ISA_ENCODING_AGX3B, c * 2, c,
                                c + 1, true, 1u << c, bytes, &len));
      append(tail, bytes, len);
   }
   tail.insert(tail.end(), {14, 0, 0, 0});
   ASSERT_GE(got.size(), tail.size());
   EXPECT_TRUE(std::equal(tail.begin(), tail.end(), got.end() - tail.size()));
}

TEST_F(AGX3PackBinary, SixOutputTailSinkingRefusesWithoutMutation)
{
   for (unsigned mode = 0; mode < 9; ++mode) {
      agx_builder *b = builder(mode == 0 ? AGX_ISA_ENCODING_AGX3C :
                                           AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = mode == 1 ? MESA_SHADER_FRAGMENT : MESA_SHADER_VERTEX;
      b->shader->is_preamble = mode == 2;
      for (unsigned c = 0; c < (mode == 3 ? 4u : 6u); ++c) {
         agx_index value = agx_mov_imm(b, 32, 0x3f000000 + c);
         if (mode == 4) value = agx_register(0, AGX_SIZE_32);
         agx_st_vary(b, agx_immediate(mode == 5 && c == 5 ? 4 : c), value);
      }
      if (mode == 6) agx_begin_cf(b);
      if (mode != 7) agx_stop(b);
      if (mode == 8) agx_test_block(b->shader);
      std::vector<agx_instr *> before;
      agx_foreach_instr_global(b->shader, I) before.push_back(I);
      EXPECT_FALSE(agx3_sink_vertex_outputs(b->shader)) << mode;
      std::vector<agx_instr *> after;
      agx_foreach_instr_global(b->shader, I) after.push_back(I);
      EXPECT_EQ(before, after) << mode;
   }
}


TEST_F(AGX3PackBinary, SixOutputVertexAddressArithmeticSurvivesRA)
{
   for (unsigned shift : {0u, 3u}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_VERTEX;
      b->shader->nir = rzalloc(mem_ctx, nir_shader);
      b->shader->nir->info.stage = MESA_SHADER_VERTEX;
      agx_index vertex = agx_temp(b->shader, AGX_SIZE_32);
      agx_get_sr_to(b, vertex, AGX_SR_VERTEX_ID);
      /* Pure integer address arithmetic supplies the stride adjustment
       * before the device load. Exercise both unshifted and shifted forms
       * already supported by the normal IADD encoder. */
      agx_index index = agx_temp(b->shader, AGX_SIZE_32);
      agx_iadd_to(b, index, vertex, vertex, shift);
      agx_index loaded = agx_temp(b->shader, AGX_SIZE_32);
      agx_device_load_to(b, loaded, agx_uniform(0, AGX_SIZE_64), index,
                         AGX_FORMAT_I32, 1, 2, false);
      for (unsigned c : {3u, 2u, 1u, 5u, 0u, 4u}) {
         agx_index value = c >= 4 ? loaded :
                            agx_mov_imm(b, 32, 0x3f000000 + c);
         agx_st_vary(b, agx_immediate(c), value);
      }
      agx_stop(b);
      agx_pressure_schedule(b->shader);
      ASSERT_TRUE(agx3_sink_vertex_outputs(b->shader));
      agx_ra(b->shader);
      /* The first vertex load owns native slot 6. Its result stays live
       * through RA until both user stores, and the first terminal COPY
       * must retire that load even when copying a constant position. */
      unsigned loaded_reg = ~0u, stores = 0;
      std::vector<uint8_t> tail;
      agx_foreach_instr_global(b->shader, I) {
         if (I->op == AGX_OPCODE_DEVICE_LOAD)
            loaded_reg = I->dest[0].value;
         if (I->op != AGX_OPCODE_ST_VARY)
            continue;
         const unsigned c = I->src[0].value;
         ASSERT_EQ(c, stores++);
         if (c >= 4) {
            EXPECT_EQ(I->src[1].value, loaded_reg);
         }
         uint8_t bytes[AGX3_STAGE_MAX_LENGTH], len;
         ASSERT_TRUE(agx3_stage_copy_source(AGX_ISA_ENCODING_AGX3B,
                       c * 2, I->src[1].value, I->src[1].kill,
                       c + 1, c == 0 ? 32 : 0, bytes, &len));
         append(tail, bytes, len);
      }
      ASSERT_EQ(stores, 6u);
      for (unsigned c = 0; c < 6; ++c) {
         uint8_t bytes[AGX3_VARY_MAX_LENGTH], len;
         ASSERT_TRUE(agx3_vary_uvsw(AGX_ISA_ENCODING_AGX3B, c * 2, c,
                                   c + 1, true, 1u << c, bytes, &len));
         append(tail, bytes, len);
      }
      tail.insert(tail.end(), {14, 0, 0, 0});
      auto got = pack(b);
      ASSERT_GE(got.size(), tail.size());
      EXPECT_TRUE(std::equal(tail.begin(), tail.end(), got.end() - tail.size()));
      EXPECT_EQ(b->shader->agx3_staging_halfs, 12u);
   }
}

TEST_F(AGX3PackBinary, NativeHighVertexOutputIndices)
{
   const struct {
      unsigned component, slot, wait;
      uint8_t bytes[8];
   } native[] = {
      {4, 5, 16, {0x57,0x06,0x55,0x08,0x80,0x40,0x4c,0x00}},
      {8, 6, 16, {0x57,0x06,0x55,0x10,0x00,0x41,0x4d,0x00}},
      {9, 5, 0, {0x57,0x06,0x54,0x12,0x20,0x41,0x4c,0x00}},
      {10, 4, 8, {0x57,0x86,0x54,0x14,0x40,0x41,0x4b,0x00}},
      {31, 2, 2, {0x57,0x26,0x54,0x3e,0xe0,0x43,0x49,0x00}},
   };
   for (auto &v : native) {
      uint8_t out[AGX3_VARY_MAX_LENGTH], len;
      ASSERT_TRUE(agx3_vary_uvsw(AGX_ISA_ENCODING_AGX3B, v.component * 2,
                                v.component, v.slot, true, v.wait, out, &len));
      ASSERT_EQ(len, 8u);
      EXPECT_EQ(memcmp(out, v.bytes, 8), 0);
   }
}

TEST_F(AGX3PackBinary, WideVertexOutputsPreserveValuesAndRecycleSlotsAfterWrites)
{
   for (unsigned count : {5u, 7u, 8u, 9u, 11u, 16u, 32u}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_VERTEX;
      b->shader->nir = rzalloc(mem_ctx, nir_shader);
      b->shader->nir->info.stage = MESA_SHADER_VERTEX;
      for (unsigned c = count; c-- > 0;) {
         agx_index value = agx_mov_imm(b, 32, 0x3f000000 + c);
         agx_st_vary(b, agx_immediate(c), value);
      }
      agx_stop(b);
      agx_pressure_schedule(b->shader);
      ASSERT_TRUE(agx3_sink_vertex_outputs(b->shader));
      agx_ra(b->shader);
      uint32_t regs[256] = {0};
      unsigned stored = 0;
      agx_foreach_instr_global(b->shader, I) {
         if (I->op == AGX_OPCODE_MOV_IMM)
            regs[I->dest[0].value] = I->imm;
         else if (I->op == AGX_OPCODE_ST_VARY) {
            EXPECT_EQ(I->src[0].value, stored);
            EXPECT_EQ(regs[I->src[1].value], 0x3f000000u + stored++);
         } else
            ASSERT_EQ(I->op, AGX_OPCODE_STOP);
      }
      ASSERT_EQ(stored, count);
      const auto bytes = pack(b);
      /* Skip the independently sized immediate producers. Decode actual
       * emitted COPY/UVSW operands, tracking outstanding staging reads and
       * writeout slots. This catches premature reuse or lost UVS values. */
      size_t at = 0;
      agx_foreach_instr_global(b->shader, I) {
         if (I->op != AGX_OPCODE_MOV_IMM) break;
         struct agx3_encoded_ldimm ld;
         ASSERT_TRUE(agx3_pack_ldimm_u32(AGX_ISA_ENCODING_AGX3B,
                       I->dest[0].value, I->imm, &ld));
         at += ld.length;
      }
      unsigned copies = 0, writes = 0, pending = 0, active_writes = 0;
      bool ready[AGX3_MAX_VERTEX_OUTPUTS] = {false};
      while (at + 4 < bytes.size()) {
         const uint8_t *p = bytes.data() + at;
         if ((p[0] & 0xf) == 0xb) {
            const unsigned len = (p[2] & 7) == 1 ? 4 : 10;
            ASSERT_LE(at + len, bytes.size());
            const unsigned slot = len == 4 ? 1 : 1 + ((p[7] >> 2) & 7);
            const unsigned wait = len == 4 ?
               (p[3] >> 5 ? 1u << ((p[3] >> 5) - 1) : 0) :
               (p[5] >> 5) | ((p[7] >> 5) << 3);
            ASSERT_LE(slot, 6u);
            EXPECT_EQ(((p[3] & 1) | ((p[0] >> 4) << 1) |
                       ((p[2] >> 6) << 5)), 2 * copies);
            EXPECT_EQ((active_writes & (1u << (slot - 1))) & ~wait, 0u);
            active_writes &= ~wait;
            EXPECT_EQ(pending & (1u << (slot - 1)), 0u);
            pending |= 1u << (slot - 1);
            ready[copies++] = true;
            at += len;
         } else {
            ASSERT_EQ(p[0], 0x57);
            ASSERT_LE(at + 8, bytes.size());
            const unsigned component = (p[4] >> 5) | ((p[5] & 0x1f) << 3);
            const unsigned slot = 1 + (p[6] & 7);
            const unsigned wait = (p[1] >> 4) | ((p[2] & 3) << 4);
            ASSERT_EQ(component, writes);
            ASSERT_TRUE(ready[component]);
            EXPECT_EQ(p[3], 2 * component);
            EXPECT_EQ(wait & (1u << (slot - 1)), 1u << (slot - 1));
            pending &= ~wait;
            active_writes |= 1u << (slot - 1);
            ++writes;
            at += 8;
         }
      }
      EXPECT_EQ(copies, count);
      EXPECT_EQ(writes, count);
      EXPECT_EQ(pending, 0u);
      EXPECT_EQ(b->shader->agx3_staging_halfs, 2 * count);
      ASSERT_EQ(at + 4, bytes.size());
      EXPECT_EQ(bytes[at], 14);
   }
}

TEST_F(AGX3PackBinary, WideVertexOutputsRefuseNonterminalAndOutOfRange)
{
   for (unsigned count : {7u, 8u, 9u, 33u}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_VERTEX;
      for (unsigned c = 0; c < count; ++c) {
         agx_mov_imm_to(b, agx_register(0, AGX_SIZE_32), 0x3f000000 + c);
         agx_st_vary(b, agx_immediate(c), agx_register(0, AGX_SIZE_32));
      }
      agx_stop(b);
      EXPECT_TRUE(pack(b, false).empty());
   }
}

/* The ID remains live after both immediate adds in the transfer lowering.
 * Old encoding unconditionally set f42, discarding it before the store. */
TEST_F(AGX3PackBinary, ImmediateIaddPreservesLiveSource)
{
   for (bool killed : {false, true}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index source = agx_register(2, AGX_SIZE_32);
      source.discard = killed;
      agx_iadd_to(b, agx_register(0, AGX_SIZE_32), agx_immediate(0), source, 2);
      agx_stop(b);
      auto bytes = pack(b);
      const uint8_t native[] = {0x9f,0x01,0x54,0x00,0x02,0x00,0x08,0x88,
                                (uint8_t)(killed ? 0x14 : 0x10),0x14};
      ASSERT_GE(bytes.size(), sizeof(native));
      EXPECT_EQ(memcmp(bytes.data(), native, sizeof(native)), 0);
   }
}

TEST_F(AGX3PackBinary, LocalIndexZeroExtendUsesCopyAndClear)
{
   for (unsigned source : {0u, 1u, 255u}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index half = agx_register(source, AGX_SIZE_16);
      half.abs = true;
      half.discard = true;
      half.kill = source == 255;
      agx_iadd_to(b, agx_register(4, AGX_SIZE_32), agx_immediate(0), half, 0);
      agx_stop(b);

      uint8_t bytes[AGX3_COPY_MAX_LENGTH], len = 0;
      std::vector<uint8_t> want;
      ASSERT_TRUE(agx3_copy_encode_reg16(AGX_ISA_ENCODING_AGX3B, 4, source,
                                         false, false, true, 0, bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_clr(AGX_ISA_ENCODING_AGX3B, 5, true, false, bytes,
                           &len));
      append(want, bytes, len);
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want) << "source half=" << source;
   }
}

TEST_F(AGX3PackBinary, ByteAddressLoweringKeepsFullU32Offset)
{
   for (bool enabled : {false, true}) {
      nir_builder b=nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,NULL,
                                                   "byte-index-address");
      nir_def *base=nir_load_push_constant(&b,1,64,nir_imm_int(&b,0));
      nir_def *offset=nir_channel(&b,nir_load_global_invocation_id(&b,32),0);
      nir_def *value=nir_load_global(&b,1,32,
         nir_iadd(&b,base,nir_u2u64(&b,offset)),.align_mul=4);
      nir_store_global(&b,value,nir_imm_int64(&b,0x100000),.align_mul=4);
      ASSERT_TRUE(agx_nir_lower_address(b.shader,enabled));
      unsigned found=0;
      nir_foreach_function_impl(impl,b.shader) {
         nir_foreach_block(block,impl) {
            nir_foreach_instr(instr,block) {
               if(instr->type!=nir_instr_type_intrinsic)continue;
               auto intr=nir_instr_as_intrinsic(instr);
               if(intr->intrinsic!=nir_intrinsic_load_agx)continue;
               found++;
               EXPECT_EQ(nir_intrinsic_base(intr),
                         enabled?AGX_NIR_MEMORY_INDEX_BYTES:0u);
               if(enabled){EXPECT_EQ(intr->src[0].ssa,base);
                           EXPECT_EQ(intr->src[1].ssa,offset);}
            }
         }
      }
      EXPECT_EQ(found,1u);ralloc_free(b.shader);
   }
}

TEST_F(AGX3PackBinary, PackedVertexMemoryNativeBytes)
{
   /* Apple packed_uchar4/ushort4 records with pad then colour, stride5/10.
    * The unaligned whole vector is one native load into packed GPR words. */
   const uint8_t native[2][14] = {
      {0x67,0x00,0x44,0x00,0x00,0x00,0x00,0x00,0x67,0x25,0x00,0x40,0x22,0x00},
      {0x67,0x00,0x44,0x00,0x00,0x00,0x00,0x00,0x47,0x45,0x00,0x40,0x22,0x00},
   };
   for (unsigned half = 0; half < 2; ++half) {
      struct agx3_mem_desc desc = {};
      desc.width = half ? AGX3_MEM_W16 : AGX3_MEM_W8;
      desc.mask = 15;
      desc.base_kill = true;
      desc.idx_kind = AGX3_MEM_IDX_IMM;
      desc.byte_off = 1u << half;
      desc.slot = 6;
      desc.agx3b_last_load = true;
      uint8_t bytes[AGX3_MEM_MAX_LENGTH], length;
      ASSERT_TRUE(agx3_mem_encode_load(AGX_ISA_ENCODING_AGX3B, &desc, bytes, &length));
      ASSERT_EQ(length, sizeof(native[half]));
      EXPECT_EQ(memcmp(bytes, native[half], length), 0);
   }
}

/* The engine's instance transform branches finish before vertex writeout.
 * The sink must leave that CFG intact and move stores only inside its final
 * block; this is independent of the native branch encoder's validation. */
TEST_F(AGX3PackBinary, VertexTerminalOutputBlockPreservesEarlierControlFlow)
{
   for (unsigned count : {6u, 9u, 32u}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_VERTEX;
      agx_block *entry = agx_start_block(b->shader);
      agx_block *merge = agx_test_block(b->shader);
      agx_block *output = agx_test_block(b->shader);
      agx_begin_cf(b);
      agx_if_icmp(b, agx_register(0, AGX_SIZE_32),
                    agx_register(2, AGX_SIZE_32), 1, AGX_ICOND_ULT, false, merge);
      agx_jmp_exec_none(b, merge);
      agx_block_add_successor(entry, merge);
      b->cursor = agx_after_block(merge);
      agx_pop_exec(b, 1);
      agx_block_add_successor(merge, output);
      std::vector<agx_instr *> prefix;
      agx_foreach_instr_global(b->shader, I) prefix.push_back(I);
      b->cursor = agx_after_block(output);
      std::vector<agx_instr *> stores(count);
      for (unsigned c = count; c-- > 0;) {
         agx_index value = agx_mov_imm(b, 32, 0x3f000000 + c);
         stores[c] = agx_st_vary(b, agx_immediate(c), value);
      }
      agx_stop(b);
      ASSERT_TRUE(agx3_sink_vertex_outputs(b->shader));
      unsigned pos = 0, next = 0;
      agx_foreach_block(b->shader, block) {
         agx_foreach_instr_in_block(block, I) {
            if (block != output) {
               ASSERT_LT(pos, prefix.size());
               EXPECT_EQ(I, prefix[pos++]);
            } else if (I->op == AGX_OPCODE_ST_VARY) {
               ASSERT_LT(next, count);
               EXPECT_EQ(I, stores[next++]);
            } else if (next) {
               EXPECT_EQ(I->op, AGX_OPCODE_STOP);
            }
         }
      }
      EXPECT_EQ(pos, prefix.size());
      EXPECT_EQ(next, count);
   }
}

TEST_F(AGX3PackBinary, VertexOutputCannotMoveAcrossControlFlow)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_VERTEX;
   for (unsigned c = 0; c < 9; ++c) {
      agx_index value = agx_mov_imm(b, 32, 0x3f000000 + c);
      agx_st_vary(b, agx_immediate(c), value);
   }
   agx_block *end = agx_test_block(b->shader);
   b->cursor = agx_after_block(end);
   agx_stop(b);
   std::vector<agx_instr *> before, after;
   agx_foreach_instr_global(b->shader, I) before.push_back(I);
   EXPECT_FALSE(agx3_sink_vertex_outputs(b->shader));
   agx_foreach_instr_global(b->shader, I) after.push_back(I);
   EXPECT_EQ(before, after);
}

TEST_F(AGX3PackBinary, VertexCFGTerminalOutputBytesMatchSingleBlock)
{
   for (unsigned count : {6u, 9u, 32u}) {
      std::vector<uint8_t> straight;
      for (bool cfg : {false, true}) {
         agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
         b->shader->stage = MESA_SHADER_VERTEX;
         if (cfg) {
            agx_block *merge = agx_test_block(b->shader);
            agx_block *output = agx_test_block(b->shader);
            agx_begin_cf(b);
            agx_if_icmp(b, agx_register(0, AGX_SIZE_32),
                          agx_register(2, AGX_SIZE_32), 1, AGX_ICOND_ULT, false, merge);
            agx_jmp_exec_none(b, merge);
            b->cursor = agx_after_block(merge);
            agx_pop_exec(b, 1);
            b->cursor = agx_after_block(output);
         }
         for (unsigned c = 0; c < count; ++c)
            agx_st_vary(b, agx_immediate(c), agx_register(4, AGX_SIZE_32));
         agx_stop(b);
         auto code = pack(b);
         if (!cfg) {
            straight = code;
         } else {
            ASSERT_GT(code.size(), straight.size());
            EXPECT_TRUE(std::equal(straight.begin(), straight.end(),
                                    code.end() - straight.size()));
         }
      }
   }
}

TEST(AGX3NativeRgb10A2, NativeDeadLiveAndConstantOperands)
{
 const struct agx3_rgb10a2_desc d[3]={
  {.dest=0,.source={0,2,4,6},.kind={2,2,2,2},.discard={1,1,1,1}},
  {.dest=8,.source={6,4,2,0},.kind={2,2,2,2}},
  {.source={48,32,16,48},.kind={1,1,1,1},.first_instruction=true},
 };
 const unsigned char want[3][14]={{0xa7,0x06,0x54,0x00,0x02,0x00,0x08,0x20,0x60,0xa0,0xea,0x8b,0x24,0x11},{0xa7,0x06,0x54,0x08,0x02,0x0c,0x10,0x10,0x00,0xa0,0x2a,0x88,0x24,0x11},{0xa7,0x0e,0x54,0,2,0x60,0x80,0x80,0,0xa3,0x2a,8,0,0x10}};
 for(unsigned i=0;i<3;i++){
  unsigned char out[14]={0};unsigned char len=0;
  ASSERT_TRUE(agx3_rgb10a2_encode(&d[i],out,&len));
  ASSERT_EQ(len,14); EXPECT_EQ(memcmp(out,want[i],14),0);
 }
}

TEST(AGX3NativeRgb10A2, NativeUnpackAndTextureDependency)
{
   const uint8_t expected[3][8] = {
      {0x27,6,0x56,0,1,0,0x5a,8}, {0x27,6,0x56,4,1,2,0x5a,8},
      {0x27,0x16,0x54,0,1,0,0x5a,8},
   };
   for (unsigned i = 0; i < 3; ++i) {
      struct agx3_rgb10a2_unpack_desc d = {
         .dest = (uint16_t)(i == 1 ? 8 : 0),
         .source = (uint16_t)(i == 1 ? 2 : 0),
         .wait_mask = (uint16_t)(i == 2 ? 1 : 32), .discard = true,
      };
      uint8_t out[8], length;
      ASSERT_TRUE(agx3_rgb10a2_unpack_encode(&d, out, &length));
      ASSERT_EQ(length, 8); EXPECT_EQ(memcmp(out, expected[i], 8), 0);
   }
   struct agx3_rgb10a2_desc p = {
      .source = {0,2,4,6}, .kind = {2,2,2,2}, .wait_mask = 1,
      .discard = {true,true,true,true},
   };
   uint8_t bytes[14], length;
   const uint8_t want[14] = {0xa7,0x16,0x54,0,2,0,8,0x20,0x60,0xa0,0xea,0x8b,0x24,0x11};
   ASSERT_TRUE(agx3_rgb10a2_encode(&p, bytes, &length));
   EXPECT_EQ(memcmp(bytes, want, 14), 0);
}

TEST_F(AGX3PackBinary, Rgb10TileRejectsRawFloatChannels)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_st_tile(b, agx_register(0, AGX_SIZE_32), agx_immediate(255),
               agx_null(), AGX_FORMAT_RGB10A2, 15, 0, false);
   agx_stop(b);
   EXPECT_TRUE(pack(b, false).empty());
}

TEST_F(AGX3PackBinary, NativeBfeilExtractPreservesSourceLifetime)
{
   for (bool dynamic : {false,true}) {
      agx_builder *b=builder(AGX_ISA_ENCODING_AGX3B);
      agx_index value=agx_register(0,AGX_SIZE_32);value.discard=true;
      agx_index shift=dynamic?agx_register(2,AGX_SIZE_32):agx_immediate(14);
      if(dynamic)shift.discard=true;
      agx_bfeil_to(b,agx_register(0,AGX_SIZE_32),agx_immediate(0),value,shift,14);
      agx_stop(b);
      auto code=pack(b);
      const uint8_t expected[2][12]={
         {0xa7,0,0x54,0,2,0,0x38,0,0xf0,0x11,0xe1,0},
         {0xa7,0,0x54,0,2,0,8,0,0xf0,0x13,0xe5,0},
      };
      ASSERT_EQ(code.size(),16);
      EXPECT_EQ(memcmp(code.data(),expected[dynamic],12),0);
   }
}

TEST_F(AGX3PackBinary, BfeilKeepsNonzeroMergeBaseClosed)
{
   agx_builder *b=builder(AGX_ISA_ENCODING_AGX3B);
   agx_bfeil_to(b,agx_register(0,AGX_SIZE_32),agx_register(4,AGX_SIZE_32),
               agx_register(0,AGX_SIZE_32),agx_immediate(14),14);
   agx_stop(b);EXPECT_TRUE(pack(b,false).empty());
}

TEST(AGX3NativeBfeil, NativeWaitAndOperandBounds)
{
   const enum agx3_alu_hint hints[2]={AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_NONE};
   uint8_t bytes[AGX3_ALU_MAX_LENGTH],length;
   const uint8_t dead[12]={0xa7,0,0x56,0,2,0,0x38,0,0xf0,0x11,0xe1,0};
   ASSERT_TRUE(agx3_alu_bfeil(AGX_ISA_ENCODING_AGX3B,0,0,14,false,14,hints,32,bytes,&length));
   ASSERT_EQ(length,12);EXPECT_EQ(memcmp(bytes,dead,12),0);
   EXPECT_FALSE(agx3_alu_bfeil(AGX_ISA_ENCODING_AGX3B,1,0,14,false,14,hints,0,bytes,&length));
   EXPECT_FALSE(agx3_alu_bfeil(AGX_ISA_ENCODING_AGX3B,0,0,32,false,14,hints,0,bytes,&length));
   EXPECT_FALSE(agx3_alu_bfeil(AGX_ISA_ENCODING_AGX3B,0,0,14,false,32,hints,0,bytes,&length));
   EXPECT_FALSE(agx3_alu_bfeil(AGX_ISA_ENCODING_AGX3B,0,0,14,false,14,hints,64,bytes,&length));
}

TEST_F(AGX3PackBinary, NativeIselImmediateAndNarrowComparands)
{
   for (bool narrow : {false, true}) {
      for (bool lhs : {false, true}) {
         agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
         const auto size = narrow ? AGX_SIZE_16 : AGX_SIZE_32;
         agx_index a = agx_register(narrow ? 1 : 2, size);
         agx_index t = agx_register(narrow ? 3 : 4, size);
         agx_index f = agx_register(narrow ? 5 : 6, size);
         a.cache = true;
         t.discard = f.discard = true;
         agx_icmpsel_to(b, agx_register(0, size),
                        lhs ? agx_immediate(2) : a,
                        lhs ? a : agx_immediate(2), t, f,
                        lhs ? AGX_ICOND_UGT : AGX_ICOND_ULT);
         agx_stop(b);
         auto code = pack(b);
         ASSERT_EQ(code.size(), 14);
         struct agx3_isel_desc d = {};
         d.dest_16 = narrow;
         d.cmp_16 = narrow;
         d.cmp1_immediate = true;
         d.cmp0 = a.value;
         d.cmp1 = 2;
         d.cond = AGX3_ISEL_ICOND_ULT;
         d.sel_true = t.value;
         d.sel_false = f.value;
         d.cache[0] = true;
         d.discard[2] = d.discard[3] = true;
         uint8_t expected[AGX3_ISEL_MAX_LENGTH], length;
         ASSERT_TRUE(agx3_isel_encode(AGX_ISA_ENCODING_AGX3B, &d, expected, &length));
         EXPECT_EQ(memcmp(code.data(), expected, length), 0);
      }
   }
}

TEST_F(AGX3PackBinary, NativeIselRetainsLiveRegisterComparands)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index a = agx_register(2, AGX_SIZE_32);
   agx_index c = agx_register(4, AGX_SIZE_32);
   agx_index t = agx_register(6, AGX_SIZE_32);
   agx_index f = agx_register(8, AGX_SIZE_32);
   a.cache = c.cache = t.cache = f.cache = true;
   agx_icmpsel_to(b, agx_register(0, AGX_SIZE_32), a, c, t, f, AGX_ICOND_ULT);
   agx_stop(b);
   auto code = pack(b);
   const uint8_t native_nowait[10] = {0x02,0x83,0x07,0x85,0x02,0x86,0x05,0,0,0x88};
   ASSERT_EQ(code.size(), 14);
   EXPECT_EQ(memcmp(code.data(), native_nowait, 10), 0);
}

TEST_F(AGX3PackBinary, NativeIselImageCopyBooleanHalfRegisters)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index a = agx_register(1, AGX_SIZE_16);
   agx_index f = agx_register(10, AGX_SIZE_16);
   a.discard = f.discard = true;
   agx_icmpsel_to(b, agx_register(1, AGX_SIZE_16), a, agx_immediate(0),
                  agx_immediate(0), f, AGX_ICOND_UEQ);
   agx_index x = agx_register(4, AGX_SIZE_16);
   agx_index y = agx_register(6, AGX_SIZE_16);
   agx_icmpsel_to(b, agx_register(1, AGX_SIZE_16), x, y, x, y, AGX_ICOND_UEQ);
   agx_stop(b);
   /* Both original image-copy shapes stay fused: two10-byte selects+stop. */
   EXPECT_EQ(pack(b).size(), 24);
}

TEST_F(AGX3PackBinary, ProjectiveRhwTracksBranchesAndPartialRegisterLifetimes)
{
   for (unsigned mode = 0; mode < 8; ++mode) {
      SCOPED_TRACE(mode);
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_block *entry = list_first_entry(&b->shader->blocks, agx_block, link);
      agx_block *left = agx_test_block(b->shader);
      agx_block *right = agx_test_block(b->shader);
      entry->successors[0] = left;
      entry->successors[1] = right;
      const agx_index rhw = agx_register(0, AGX_SIZE_32);
      agx_rhw_interp_agx3_to(b, rhw, agx_immediate(0), agx_immediate(0),
                            AGX_INTERPOLATION_CENTER);
      if (mode == 0 || mode == 6)
         agx_persp_rcp_agx3_to(b, mode == 0 ? rhw : agx_register(2, AGX_SIZE_32), rhw, 0);
      b->cursor = agx_after_block(left);
      if (mode == 4 || mode == 5 || mode == 6)
         agx_mov_imm_to(b, agx_register(0, AGX_SIZE_16), 0);
      if (mode == 6)
         agx_mov_imm_to(b, agx_register(1, AGX_SIZE_16), 0);
      if (mode == 4)
         agx_mov_to(b, agx_register(4, AGX_SIZE_16), agx_register(1, AGX_SIZE_16));
      else if (mode == 6)
         agx_fmul_to(b, agx_register(4, AGX_SIZE_32), rhw, rhw);
      else if (mode != 0)
         agx_persp_rcp_agx3_to(b, agx_register(2, AGX_SIZE_32), rhw, mode == 7 ? 1 : 0);
      if (mode == 3)
         left->successors[0] = left;
      b->cursor = agx_after_block(right);
      if (mode == 2)
         agx_fmul_to(b, agx_register(4, AGX_SIZE_32), rhw, rhw);
      else if (mode != 0)
         agx_persp_rcp_agx3_to(b, agx_register(2, AGX_SIZE_32), rhw, 0);
      agx_stop(b);
      const bool valid = mode == 0 || mode == 1 || mode == 3 || mode == 6;
      EXPECT_EQ(!pack(b, valid).empty(), valid);
   }
}

TEST_F(AGX3PackBinary, ModifiedFaddWaitAndLifetimeStayFused)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index loaded = agx_register(0, AGX_SIZE_32);
   loaded.channels_m1 = 1;
   agx_device_load_to(b, loaded, agx_uniform(0, AGX_SIZE_64),
                      agx_immediate(0), AGX_FORMAT_I32, 3, 2, false);
   agx_index x = agx_register(0, AGX_SIZE_32);
   agx_index y = agx_register(2, AGX_SIZE_32);
   x.cache = true;
   y.neg = y.discard = true;
   agx_fadd_to(b, agx_register(4, AGX_SIZE_32), x, y);
   agx_stop(b);
   const auto code = pack(b);
   /* Native negate-first subtraction consumes the load directly, retains x
    * and discards y. No temporary negation or separate wait operation. */
   const uint8_t native[6] = {0x29,0x81,0x14,0x03,0,0xc8};
   ASSERT_GE(code.size(), 10u);
   EXPECT_EQ(memcmp(code.data()+code.size()-10, native, 6), 0);
}

TEST_F(AGX3PackBinary, NarrowFaddKeepsOddHalfAndLiveInput)
{
   for (bool swapped : {false,true}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index source = agx_register(2, AGX_SIZE_32);
      agx_index zero = agx_immediate(0x80);
      agx_fadd_to(b, agx_register(1, AGX_SIZE_16),
                  swapped ? zero : source, swapped ? source : zero);
      agx_stop(b);
      auto code = pack(b);
      /* One16-bit rounding ADD, with no source discard and no widening. */
      const uint8_t expected[6] = {0x01,0x03,0x14,0x81,0x04,0x02};
      ASSERT_EQ(code.size(),10);
      EXPECT_EQ(memcmp(code.data(),expected,6),0);
   }
}

TEST_F(AGX3PackBinary, HalfMaximumRoutesLifetimesWithoutPhantomWait)
{
   for (bool live : {false, true}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index a = agx_register(2, AGX_SIZE_16);
      agx_index c = agx_register(3, AGX_SIZE_16);
      agx_index dest = agx_register(live ? 0 : 2, AGX_SIZE_16);
      if (live) {
         a.cache = c.cache = dest.cache = true;
      } else {
         a.discard = c.discard = true;
      }
      agx_fcmpsel_to(b, dest, a, c, a, c, AGX_FCOND_GTN);
      agx_stop(b);
      const auto code = pack(b);
      const uint8_t retained[] = {0x02,0x82,0x24,0x82,0x10,0x00};
      const uint8_t dead[] = {0x12,0x02,0x1c,0x02,0x10,0x00};
      const uint8_t *native = live ? retained : dead;
      ASSERT_EQ(code.size(), 10u);
      EXPECT_EQ(memcmp(code.data(), native, 6), 0);
   }
}

TEST_F(AGX3PackBinary, HalfClampDoesNotCarryLoadWaitIntoSecondSelect)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index x = agx_register(2, AGX_SIZE_16);
   agx_index low = agx_register(3, AGX_SIZE_16);
   agx_index high = agx_register(4, AGX_SIZE_16);
   x.cache = low.cache = true;
   agx_index temporary = agx_register(0, AGX_SIZE_16);
   temporary.cache = true;
   agx_fcmpsel_to(b, temporary, x, low, x, low, AGX_FCOND_GTN);
   temporary.cache = false;
   temporary.discard = high.discard = true;
   agx_fcmpsel_to(b, agx_register(0, AGX_SIZE_16), temporary, high,
                  temporary, high, AGX_FCOND_LTN);
   agx_stop(b);
   const auto code = pack(b);
   const uint8_t expected_chain[] = {
      0x02,0x82,0x24,0x82,0x10,0x00,
      0x02,0x00,0x1c,0x04,0x01,0x00,
   };
   ASSERT_EQ(code.size(), sizeof(expected_chain) + 4);
   if (memcmp(code.data(), expected_chain, sizeof(expected_chain)) != 0) {
      fprintf(stderr, "half-clamp chain got ");
      for (unsigned i = 0; i < sizeof(expected_chain); ++i)
         fprintf(stderr, "%02x", code[i]);
      fputc('\n', stderr);
   }
   EXPECT_EQ(memcmp(code.data(), expected_chain, sizeof(expected_chain)), 0);
}

TEST_F(AGX3PackBinary, HalfReciprocalRoutesNoWaitAndRefusesModifier)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index dest = agx_register(0, AGX_SIZE_16);
   agx_index src = agx_register(3, AGX_SIZE_16);
   dest.cache = src.cache = true;
   agx_rcp_to(b, dest, src);
   agx_stop(b);
   const auto code = pack(b);
   const uint8_t expected[] = {
      0xaf,0x00,0x54,0x00,0x03,0x06,0x02,0x44,0x20,0x00,
   };
   ASSERT_EQ(code.size(), sizeof(expected) + 4);
   EXPECT_EQ(memcmp(code.data(), expected, sizeof(expected)), 0);

   b = builder(AGX_ISA_ENCODING_AGX3B);
   src.neg = true;
   agx_rcp_to(b, dest, src);
   agx_stop(b);
   EXPECT_TRUE(pack(b, false).empty());
}

TEST_F(AGX3PackBinary, HalfMaximumRefusesUngroundedShapes)
{
   for (unsigned mode = 0; mode < 3; ++mode) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index a = agx_register(2, AGX_SIZE_16);
      agx_index c = agx_register(3, AGX_SIZE_16);
      agx_index selected_a = a;
      if (mode == 0)
         selected_a.value = 4;
      if (mode == 1)
         a.abs = true;
      agx_instr *I = agx_fcmpsel_to(b, agx_register(0, AGX_SIZE_16), a, c,
                                    selected_a, c, AGX_FCOND_GTN);
      if (mode == 2)
         I->saturate = true;
      agx_stop(b);
      EXPECT_TRUE(pack(b, false).empty()) << "mode=" << mode;
   }
}

TEST_F(AGX3PackBinary, FselZeroMixedGnomeRegisterShape)
{
   for (unsigned mode = 0; mode < 4; ++mode) {
      const bool zero_first = mode & 1, invert = mode & 2;
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index half = agx_register(1, AGX_SIZE_16);
      agx_index zero = agx_immediate(0);
      agx_instr *I = agx_fcmpsel_to(b, agx_register(70, AGX_SIZE_32),
                     zero_first ? zero : half, zero_first ? half : zero,
                     agx_register(invert ? 2 : 4, AGX_SIZE_32),
                     agx_register(invert ? 4 : 2, AGX_SIZE_32),
                     AGX_FCOND_EQ);
      I->invert_cond = invert;
      agx_stop(b);
      auto code = pack(b);
      /* Original GNOME post-RA shape: odd half comparison, both full float
       * alternatives remain live, one FSEL followed by STOP. */
      ASSERT_EQ(code.size(), 14u);
      const uint8_t expected[] = {0x32,0x00,0x87,0x81,0x06,0x04,0x08,0x02,0x00,0x02};
      EXPECT_EQ(memcmp(code.data(),expected,sizeof(expected)),0);
   }
}

TEST_F(AGX3PackBinary, FselZeroNativeLifetimeAndCompareOrder)
{
   for (bool zero_first : {false, true}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index half = agx_register(10, AGX_SIZE_16);
      agx_index t = agx_register(0, AGX_SIZE_32), f = agx_register(2, AGX_SIZE_32);
      half.discard = t.discard = f.discard = true;
      agx_fcmpsel_to(b, agx_register(0, AGX_SIZE_32),
                     zero_first ? agx_immediate(0) : half,
                     zero_first ? half : agx_immediate(0), t, f,
                     zero_first ? AGX_FCOND_GT : AGX_FCOND_LT);
      agx_stop(b);
      auto code = pack(b);
      const uint8_t native[] = {0x02,0x0a,0x0f,0x81,0x80,0x02,0x03,0x02};
      ASSERT_EQ(code.size(), sizeof(native)+4);
      EXPECT_EQ(memcmp(code.data(), native, sizeof(native)),0);
   }
}

TEST_F(AGX3PackBinary, HalfCompareGnomeLoopBreakAndBothBranchFixups)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_push_exec(b, 1);
   agx_block *body = agx_test_block(b->shader);
   agx_block *pop = agx_test_block(b->shader);
   agx_block *after = agx_test_block(b->shader);
   agx_builder lb = agx_init_builder(b->shader, agx_after_block(body));
   /* Original blur-loop comparison. The limit and counter are both live
    * after the comparison, including on the next iteration. */
   agx_break_if_icmp(&lb, agx_register(18, AGX_SIZE_16),
                     agx_register(1, AGX_SIZE_16), 1, false, AGX_ICOND_SLT, after);
   agx_jmp_exec_none_after(&lb, pop);
   agx_builder pb = agx_init_builder(b->shader, agx_after_block(pop));
   agx_jmp_exec_any(&pb, body);
   agx_pop_exec(&pb, 1);
   agx_builder ab = agx_init_builder(b->shader, agx_after_block(after));
   agx_stop(&ab);
   const auto code = pack(b);
   const uint8_t prefix[] = {
      0x0f,0x05,0x54,0x1a,
      0x2a,0x12,0x23,0x00,0x02,0x00,0x17,0x08,0x02,0x00,
      0x8f,0x04,0x54,0x26,
   };
   ASSERT_EQ(code.size(),48u);
   EXPECT_EQ(memcmp(code.data(),prefix,sizeof(prefix)),0);
   /* Reserved native branches deposit a signed byte displacement at byte3.
    * Forward reaches the footer POP; backward re-enters the compare. */
   const uint8_t forward[] = {0x0f,0x01,0x54,0x14,0,0,0,0,0,0};
   const uint8_t backward[] = {0x0f,0x00,0x54,0xe8,0xff,0xff,0xff,0xff,0xff,0};
   EXPECT_EQ(memcmp(code.data()+18,forward,sizeof(forward)),0);
   EXPECT_EQ(memcmp(code.data()+28,backward,sizeof(backward)),0);
   const uint8_t pop_native[] = {0x0f,0x06,0x04,0x02,0,0};
   EXPECT_EQ(memcmp(code.data()+38,pop_native,sizeof(pop_native)),0);
}

TEST_F(AGX3PackBinary, CachedHalfIdentityUnconditionalLoopBreak)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_push_exec(b, 1);
   agx_block *body = agx_test_block(b->shader);
   agx_block *after = agx_test_block(b->shader);
   agx_builder lb = agx_init_builder(b->shader, agx_after_block(body));
   agx_index r0l = agx_register(0, AGX_SIZE_16);
   r0l.cache = true;
   r0l.kill = true;
   agx_break_if_icmp(&lb, r0l, r0l, 2, false, AGX_ICOND_UEQ, after);
   agx_jmp_exec_any(&lb, body);
   agx_pop_exec(&lb, 1);
   agx_builder ab = agx_init_builder(b->shader, agx_after_block(after));
   agx_stop(&ab);

   uint8_t bytes[AGX3_CF_MAX_LENGTH];
   uint8_t len = 0;
   std::vector<uint8_t> want;
   ASSERT_TRUE(agx3_cf_cndst(AGX_ISA_ENCODING_AGX3B, 6, false, 2, bytes,
                             &len));
   append(want, bytes, len);
   const size_t body_offset = want.size();
   const uint16_t src[] = {0, 0};
   const struct agx3_cf_source_hints live[] = {
      {false, false}, {false, false},
   };
   ASSERT_TRUE(agx3_cf_icmp_reg16(AGX_ISA_ENCODING_AGX3B,
                                  AGX3_CF_ICOND_EQ, 1, src, live, 0,
                                  bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_cf_cndlt(AGX_ISA_ENCODING_AGX3B, 1, true, bytes, &len));
   append(want, bytes, len);
   const size_t branch_offset = want.size();
   ASSERT_TRUE(agx3_cf_branch(AGX_ISA_ENCODING_AGX3B, false, bytes, &len));
   ASSERT_TRUE(agx3_cf_patch_branch(AGX_ISA_ENCODING_AGX3B, false, bytes,
                                    body_offset - branch_offset));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_cf_cndend(AGX_ISA_ENCODING_AGX3B, 2, bytes, &len));
   append(want, bytes, len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);

   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, HalfZeroGnomeSamplerIndexSurvivesBranch)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index index = agx_register(3, AGX_SIZE_16);
   agx_if_icmp(b,index,agx_immediate(0),1,AGX_ICOND_UEQ,true,NULL);
   agx_index zext = index; zext.abs = true;
   agx_index bias = agx_immediate(28); bias.neg = true;
   agx_iadd_to(b,agx_register(8,AGX_SIZE_32),zext,bias,0);
   agx_pop_exec(b,1);
   agx_stop(b);
   auto code=pack(b);
   const uint8_t native[] = {0x0a,0x02,0x23,0x80,0x06,0x00,0x0f,0x02,0x02,0x00};
   ASSERT_GT(code.size(),sizeof(native));
   /* Both original halves remain live; no specimen-only discard/wait6. */
   EXPECT_EQ(memcmp(code.data(),native,sizeof(native)),0);
}

TEST_F(AGX3PackBinary, BlurSignedHalfConversionAndCounterStayNative)
{
   for (bool convert : {true,false}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      if (convert)
         agx_convert_to(b,agx_register(44,AGX_SIZE_32),agx_immediate(AGX_CONVERT_S16_TO_F),
                         agx_register(1,AGX_SIZE_16),AGX_ROUND_RTE);
      else
         agx_iadd_to(b,agx_register(1,AGX_SIZE_16),agx_register(1,AGX_SIZE_16),agx_immediate(2),0);
      agx_stop(b);
      auto code = pack(b);
      ASSERT_EQ(code.size(),convert ? 12u : 14u);
      const uint8_t converted[] = {0xa7,0x07,0x54,0x2c,0x02,0x02,0x84,0x60};
      const uint8_t incremented[] = {0x9f,0x01,0x54,0x01,0x02,0x04,0x04,0x00,0x11,0x04};
      EXPECT_EQ(memcmp(code.data(),convert ? converted : incremented,
                       convert ? sizeof(converted) : sizeof(incremented)),0);
   }
}

TEST_F(AGX3PackBinary, NarrowBitwiseUsesWiderSourcesLowHalves)
{
   for (unsigned mask = 1; mask < 4; ++mask) {
      for (unsigned lifetime = 0; lifetime < 4; ++lifetime) {
         for (unsigned op = 0; op < 3; ++op) {
            const enum agx3_alu_op ops[] = {AGX3_ALU_BAND,AGX3_ALU_BOR,AGX3_ALU_BXOR};
            const unsigned tt[] = {8,14,6};
            auto *b = builder(AGX_ISA_ENCODING_AGX3B);
            agx_index a = agx_register(2,mask & 1 ? AGX_SIZE_32 : AGX_SIZE_16);
            agx_index z = agx_register(6,mask & 2 ? AGX_SIZE_32 : AGX_SIZE_16);
            a.discard = lifetime & 1; a.cache = !a.discard;
            z.discard = lifetime & 2; z.cache = !z.discard;
            agx_bitop_to(b,agx_register(1,AGX_SIZE_16),a,z,tt[op]);
            agx_stop(b);
            const uint16_t halves[2] = {2,6};
            const enum agx3_alu_hint hints[2] = {
               a.discard ? AGX3_ALU_HINT_DISCARD : AGX3_ALU_HINT_CACHE,
               z.discard ? AGX3_ALU_HINT_DISCARD : AGX3_ALU_HINT_CACHE,
            };
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], length;
            ASSERT_TRUE(agx3_alu_bit_reg16(AGX_ISA_ENCODING_AGX3B,ops[op],1,
                                          halves,hints,0,bytes,&length));
            std::vector<uint8_t> want;
            append(want,bytes,length);
            struct agx3_encoded_end end;
            ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B,&end));
            append(want,end.bytes,end.length);
            EXPECT_EQ(pack(b),want);
         }
      }
   }
}

TEST_F(AGX3PackBinary, ScalarByteStoreReadsOddNativeHalf)
{
   auto *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index index = agx_register(0,AGX_SIZE_32);
   index.kill = true;
   agx_device_store(b,agx_register(3,AGX_SIZE_16),agx_uniform(4,AGX_SIZE_64),
                     index,AGX_FORMAT_I8,1,0,false);
   agx_stop(b);
   /* Apple byte_halves_odd_zero, actual native W8 store of the second
    * ushort half with a uniform pointer and last-use register index. */
   const uint8_t native[] = {0xe7,0,0x54,3,1,0,0x21,0,0x21,0,0,0x90,0x10,0};
   std::vector<uint8_t> want(native,native+sizeof(native));
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B,&end));
   append(want,end.bytes,end.length);
   EXPECT_EQ(pack(b),want);
}

TEST_F(AGX3PackBinary, ExplicitLodStagingAndLifetime)
{
   for (unsigned lod_reg : {0u, 1u, 127u, 254u, 255u}) {
      for (unsigned hint = 0; hint < 3; ++hint) {
         auto enc = AGX_ISA_ENCODING_AGX3B;
         agx_builder *b = builder(enc);
         b->shader->stage = MESA_SHADER_FRAGMENT;
         key.no_stop = true;
         agx_index dest = agx_register(8, AGX_SIZE_32);
         dest.channels_m1 = 3;
         agx_index coord = agx_register(4, AGX_SIZE_32);
         coord.channels_m1 = 1;
         agx_index lod = agx_register(lod_reg, AGX_SIZE_16);
         lod.cache = hint == 1;
         lod.discard = hint == 2;
         auto *I = agx_texture_sample_to(b, dest, coord, lod,
            agx_immediate(0), agx_immediate(0), agx_immediate(0), agx_null(),
            AGX_DIM_2D, AGX_LOD_MODE_LOD_MIN, 0, false, false, false,
            AGX_GATHER_NONE, false);
         I->mask = 15;
         std::vector<uint8_t> want;
         uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
         ASSERT_TRUE(agx3_tex_stage_lod(enc, lod_reg, 1, 0,
                                       hint == 1, hint == 2, bytes, &len));
         append(want, bytes, len);
         ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 0, 4, 1, 0,
                                              false, false, bytes, &len));
         append(want, bytes, len);
         ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 2, 6, 1, 0,
                                              false, false, bytes, &len));
         append(want, bytes, len);
         struct agx3_tex_sample_desc d = {};
         d.dest=8; d.width=AGX3_TEX_W32; d.mask=15;
         d.lod=AGX3_TEX_LOD_REGISTER; d.slot=1; d.wait_mask=1; d.final=true;
         ASSERT_TRUE(agx3_tex_encode_sample(enc, &d, bytes, &len));
         append(want, bytes, len);
         ASSERT_TRUE(agx3_alu_compose(enc, AGX3_ALU_BAND, 0, 0, 0, 0, 0,
                                      1, bytes, &len));
         append(want, bytes, len);
         EXPECT_EQ(pack(b), want);
         EXPECT_EQ(b->shader->agx3_staging_halfs, 8);
      }
   }
}

TEST_F(AGX3PackBinary, ComputeImmediateLodSample)
{
   auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_COMPUTE;
   agx_index dest = agx_register(8, AGX_SIZE_32);
   dest.channels_m1 = 3;
   agx_index coord = agx_register(4, AGX_SIZE_32);
   coord.channels_m1 = 1;
   auto *I = agx_texture_sample_to(
      b, dest, coord, agx_immediate_f(2.0f), agx_immediate(0),
      agx_immediate(0), agx_immediate(0), agx_null(), AGX_DIM_2D,
      AGX_LOD_MODE_LOD_MIN, 0, false, false, false, AGX_GATHER_NONE, false);
   I->mask = 15;
   agx_stop(b);

   std::vector<uint8_t> want;
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 0, 4, 1, 0, false, false,
                                        bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 2, 6, 1, 0, false, false,
                                        bytes, &len));
   append(want, bytes, len);
   struct agx3_tex_sample_desc d = {};
   d.dest = 8;
   d.width = AGX3_TEX_W32;
   d.mask = 15;
   d.lod = AGX3_TEX_LOD_LEVEL;
   d.lod_x64 = 128;
   d.slot = 1;
   d.wait_mask = 1;
   d.final = true;
   ASSERT_TRUE(agx3_tex_encode_sample(enc, &d, bytes, &len));
   append(want, bytes, len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(enc, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 8);
}

TEST_F(AGX3PackBinary, ComputeSampleRefusesUnqualifiedLod)
{
   for (unsigned shape = 0; shape < 3; ++shape) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_COMPUTE;
      agx_index dest = agx_register(8, AGX_SIZE_32);
      dest.channels_m1 = 3;
      agx_index coord = agx_register(4, AGX_SIZE_32);
      coord.channels_m1 = 1;
      const bool implicit = shape == 0;
      agx_index lod = shape == 1 ? agx_register(0, AGX_SIZE_16)
                                 : agx_immediate_f(-1.0f);
      auto *I = agx_texture_sample_to(
         b, dest, coord, implicit ? agx_immediate(0) : lod,
         agx_immediate(0), agx_immediate(0), agx_immediate(0), agx_null(),
         AGX_DIM_2D,
         implicit ? AGX_LOD_MODE_AUTO_LOD : AGX_LOD_MODE_LOD_MIN, 0, false,
         false, false, AGX_GATHER_NONE, false);
      I->mask = 15;
      agx_stop(b);
      pack(b, /*expect_ok=*/false);
   }
}

TEST_F(AGX3PackBinary, ArraySampleStagingAndLifetime)
{
   for (unsigned hint = 0; hint < 3; ++hint) {
      auto enc = AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      key.no_stop = true;
      agx_index dest = agx_register(8, AGX_SIZE_32);
      dest.channels_m1 = 3;
      agx_index coord = agx_register(0, AGX_SIZE_32);
      coord.channels_m1 = 2;
      coord.cache = hint == 1;
      coord.discard = hint == 2;
      coord.kill = hint == 2;
      auto *I = agx_texture_sample_to(
         b, dest, coord, agx_immediate(0), agx_immediate(0),
         agx_immediate(0), agx_immediate(0), agx_null(), AGX_DIM_2D_ARRAY,
         AGX_LOD_MODE_AUTO_LOD, 0, false, false, false, AGX_GATHER_NONE,
         false);
      I->mask = 15;
      agx_stop(b);

      std::vector<uint8_t> want;
      uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
      ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 0, 0, 1, 0,
                                           hint == 1, hint == 2,
                                           bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 2, 2, 1, 0,
                                           hint == 1, hint == 2,
                                           bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_tex_stage_array_layer(enc, 4, 1, 0,
                                            hint == 1, hint == 2,
                                            bytes, &len));
      if (hint == 2) {
         const uint8_t native[] = {
            0x17, 0x05, 0x54, 0x06, 0x00, 0x08,
            0x00, 0x00, 0xf0, 0xc0, 0x04, 0x00,
         };
         EXPECT_EQ(std::vector<uint8_t>(bytes, bytes + len),
                   std::vector<uint8_t>(native, native + sizeof(native)));
      }
      append(want, bytes, len);
      struct agx3_tex_sample_desc d = {};
      d.dest = 8;
      d.width = AGX3_TEX_W32;
      d.mask = 15;
      d.lod = AGX3_TEX_LOD_AUTO;
      d.slot = 1;
      d.wait_mask = 1;
      d.coord_kill = hint == 2;
      d.final = true;
      d.array = true;
      ASSERT_TRUE(agx3_tex_encode_sample(enc, &d, bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_alu_compose(enc, AGX3_ALU_BAND, 0, 0, 0, 0, 0,
                                   1, bytes, &len));
      append(want, bytes, len);
      EXPECT_EQ(pack(b), want);
      EXPECT_EQ(b->shader->agx3_staging_halfs, 8);
   }
}

TEST_F(AGX3PackBinary, ArrayRegisterDescriptorsUseNativeStaging)
{
   auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   key.no_stop = true;
   agx_index dest = agx_register(8, AGX_SIZE_32);
   dest.channels_m1 = 3;
   agx_index coord = agx_register(20, AGX_SIZE_32);
   coord.channels_m1 = 2;
   auto *I = agx_texture_sample_to(
      b, dest, coord, agx_immediate(0), agx_immediate(0),
      agx_register(12, AGX_SIZE_64), agx_register(16, AGX_SIZE_64),
      agx_null(), AGX_DIM_2D_ARRAY, AGX_LOD_MODE_AUTO_LOD, 0, false,
      false, false, AGX_GATHER_NONE, false);
   I->mask = 15;
   agx_stop(b);

   std::vector<uint8_t> want;
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 0, 20, 6, 0, false, false,
                                         bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 2, 22, 6, 0, false, false,
                                         bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_array_layer(enc, 24, 6, 0, false, false,
                                          bytes, &len));
   EXPECT_EQ(std::vector<uint8_t>(bytes, bytes + len),
             (std::vector<uint8_t>{0x17,0x05,0x54,0x06,0,0x30,
                                   0,0,0x70,0xe4,0x04,0}));
   append(want, bytes, len);
   struct agx3_tex_sample_desc d = {};
   d.dest = 8;
   d.width = AGX3_TEX_W32;
   d.mask = 15;
   d.lod = AGX3_TEX_LOD_AUTO;
   d.slot = 1;
   d.wait_mask = 1u << 5;
   d.final = true;
   d.array = true;
   d.texture_kind = AGX3_TEX_OPERAND_REGISTER;
   d.sampler_kind = AGX3_TEX_OPERAND_REGISTER;
   d.texture_value = 12;
   d.sampler_value = 16;
   ASSERT_TRUE(agx3_tex_encode_sample(enc, &d, bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_alu_compose(enc, AGX3_ALU_BAND, 0, 0, 0, 0, 0,
                                1, bytes, &len));
   append(want, bytes, len);
   EXPECT_EQ(pack(b), want);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 8);
}
TEST_F(AGX3PackBinary, ArraySampleAcceptsHighCoordinateRegistersWithoutWrap)
{
   auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   key.no_stop = true;
   agx_index dest = agx_register(8, AGX_SIZE_32);
   dest.channels_m1 = 3;
   agx_index coord = agx_register(120, AGX_SIZE_32);
   coord.channels_m1 = 2;
   auto *I = agx_texture_sample_to(
      b, dest, coord, agx_immediate(0), agx_immediate(0),
      agx_register(12, AGX_SIZE_64), agx_register(16, AGX_SIZE_64),
      agx_null(), AGX_DIM_2D_ARRAY, AGX_LOD_MODE_AUTO_LOD, 0, false,
      false, false, AGX_GATHER_NONE, false);
   I->mask = 15;
   agx_stop(b);

   std::vector<uint8_t> want;
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 0, 120, 6, 0, false, false,
                                         bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 2, 122, 6, 0, false, false,
                                         bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_array_layer(enc, 124, 6, 0, false, false,
                                          bytes, &len));
   append(want, bytes, len);
   struct agx3_tex_sample_desc d = {};
   d.dest = 8;
   d.width = AGX3_TEX_W32;
   d.mask = 15;
   d.lod = AGX3_TEX_LOD_AUTO;
   d.slot = 1;
   d.wait_mask = 1u << 5;
   d.final = true;
   d.array = true;
   d.texture_kind = AGX3_TEX_OPERAND_REGISTER;
   d.sampler_kind = AGX3_TEX_OPERAND_REGISTER;
   d.texture_value = 12;
   d.sampler_value = 16;
   ASSERT_TRUE(agx3_tex_encode_sample(enc, &d, bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_alu_compose(enc, AGX3_ALU_BAND, 0, 0, 0, 0, 0,
                                1, bytes, &len));
   append(want, bytes, len);
   EXPECT_EQ(pack(b), want);

   agx_builder *bad = builder(enc);
   bad->shader->stage = MESA_SHADER_FRAGMENT;
   agx_index wrapped = agx_register(124, AGX_SIZE_32);
   wrapped.channels_m1 = 2;
   I = agx_texture_sample_to(
      bad, dest, wrapped, agx_immediate(0), agx_immediate(0),
      agx_register(12, AGX_SIZE_64), agx_register(16, AGX_SIZE_64),
      agx_null(), AGX_DIM_2D_ARRAY, AGX_LOD_MODE_AUTO_LOD, 0, false,
      false, false, AGX_GATHER_NONE, false);
   I->mask = 15;
   agx_stop(bad);
   EXPECT_TRUE(pack(bad, false).empty());
}

TEST_F(AGX3PackBinary, StorageImageWriteMatchesNativeWithReadyOrSlot1Data)
{
   static const uint8_t native[AGX3_IMAGE_WRITE_LENGTH] = {
      0xd7, 0x20, 0x54, 0x00, 0x20, 0x40, 0x00, 0x00,
      0x00, 0x04, 0x00, 0x00, 0x88, 0x1a, 0x09, 0x00,
   };
   uint8_t bytes[AGX3_IMAGE_WRITE_LENGTH];
   uint8_t len = 0;
   ASSERT_TRUE(agx3_tex_encode_image_write_rgba8_2d(
      AGX_ISA_ENCODING_AGX3B, 1, 0, false, bytes, &len));
   ASSERT_EQ(len, sizeof(native));
   EXPECT_EQ(memcmp(bytes, native, sizeof(native)), 0);

   ASSERT_TRUE(agx3_tex_encode_image_write_rgba8_2d(
      AGX_ISA_ENCODING_AGX3B, 0, 0, false, bytes, &len));
   ASSERT_EQ(len, sizeof(native));
   EXPECT_EQ(memcmp(bytes, native, sizeof(native)), 0);

   for (unsigned wait : {2u, 3u, 0x40u}) {
      memset(bytes, 0xa5, sizeof(bytes));
      len = 0xa5;
      EXPECT_FALSE(agx3_tex_encode_image_write_rgba8_2d(
         AGX_ISA_ENCODING_AGX3B, wait, 0, false, bytes, &len));
      EXPECT_EQ(len, 0);
      for (uint8_t byte : bytes)
         EXPECT_EQ(byte, 0);
   }

   memset(bytes, 0xa5, sizeof(bytes));
   len = 0xa5;
   EXPECT_FALSE(agx3_tex_encode_image_write_rgba8_2d(
      AGX_ISA_ENCODING_AGX3C, 1, 0, false, bytes, &len));
   EXPECT_EQ(len, 0);
   for (uint8_t byte : bytes)
      EXPECT_EQ(byte, 0);

   static const uint8_t sampled_write[AGX3_IMAGE_WRITE_LENGTH] = {
      0xd7, 0x20, 0x54, 0x00, 0x20, 0x40, 0x00, 0x01,
      0x00, 0x04, 0x00, 0x00, 0x88, 0x3a, 0x09, 0x00,
   };
   ASSERT_TRUE(agx3_tex_encode_image_write_rgba8_2d(
      AGX_ISA_ENCODING_AGX3B, 1, 1, true, bytes, &len));
   ASSERT_EQ(len, sizeof(sampled_write));
   EXPECT_EQ(memcmp(bytes, sampled_write, sizeof(sampled_write)), 0);

   EXPECT_FALSE(agx3_tex_encode_image_write_rgba8_2d(
      AGX_ISA_ENCODING_AGX3B, 1, 0, true, bytes, &len));
   EXPECT_FALSE(agx3_tex_encode_image_write_rgba8_2d(
      AGX_ISA_ENCODING_AGX3B, 1, 1, false, bytes, &len));
}

TEST_F(AGX3PackBinary, StorageImageRoundtripDerivesLoadDependencyAndRejectsShapes)
{
   static const uint8_t imgwci[AGX3_IMAGE_WRITE_LENGTH] = {
      0xd7, 0x20, 0x54, 0x00, 0x20, 0x40, 0x00, 0x00,
      0x00, 0x04, 0x00, 0x00, 0x88, 0x1a, 0x09, 0x00,
   };

   for (unsigned mode = 0; mode < 4; ++mode) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_COMPUTE;
      agx_index coord = agx_register(0, AGX_SIZE_32);
      coord.channels_m1 = 1;
      agx_index data = agx_register(4, AGX_SIZE_32);
      data.channels_m1 = 3;
      agx_instr *load = agx_image_load_to(
         b, data, coord, agx_immediate(0), agx_immediate(0),
         agx_immediate(0), agx_immediate(0), agx_null(), AGX_DIM_2D,
         AGX_LOD_MODE_AUTO_LOD, 15, false, false, false);
      load->format = AGX_FORMAT_I8;
      agx_image_write(
         b, data, coord, agx_immediate(0), agx_immediate(0),
         agx_immediate(0), mode == 1 ? AGX_DIM_2D_ARRAY
                            : mode == 2 ? AGX_DIM_2D_MS : AGX_DIM_2D,
         mode == 3);
      agx_stop(b);

      const std::vector<uint8_t> code = pack(b, mode == 0);
      if (mode == 0) {
         EXPECT_TRUE(b->shader->agx3_staging_known);
         EXPECT_EQ(b->shader->agx3_staging_halfs, 20);
         static const uint8_t read_coords[] = {
            0x8b, 0x00, 0x21, 0x04, 0x9b, 0x02, 0x21, 0x04,
         };
         EXPECT_NE(std::search(code.begin(), code.end(), read_coords,
                               read_coords + sizeof(read_coords)),
                   code.end());
         static const uint8_t coord_x[] = {
            0x02, 0x01, 0x2f, 0x0d, 0x83, 0x00, 0x45,
            0x02, 0x20, 0x0c, 0x00, 0x04, 0x00, 0x00,
         };
         static const uint8_t coord_y[] = {
            0x12, 0x03, 0x2f, 0x0d, 0x83, 0x02, 0x45,
            0x02, 0x20, 0x0c, 0x00, 0x04, 0x00, 0x00,
         };
         const auto first_coord = std::search(code.begin(), code.end(),
                                               coord_x,
                                               coord_x + sizeof(coord_x));
         ASSERT_NE(first_coord, code.end());
         ASSERT_GE(std::distance(first_coord, code.end()),
                   static_cast<ptrdiff_t>(sizeof(coord_x) + sizeof(coord_y)));
         EXPECT_TRUE(std::equal(coord_y, coord_y + sizeof(coord_y),
                                first_coord + sizeof(coord_x)));

         const auto first = std::search(code.begin(), code.end(), imgwci,
                                        imgwci + sizeof(imgwci));
         ASSERT_NE(first, code.end());
         /* The native witness consumes the image-load scoreboard signal once:
          * the first data-staging COPY waits on slot 1, while the remaining
          * components only join the same slot-2 staging transaction. */
         const unsigned data_dest[4] = {4, 6, 12, 14};
         std::vector<uint8_t> staging;
         for (unsigned c = 0; c < 4; ++c) {
            uint8_t copy[AGX3_STAGE_MAX_LENGTH], len = 0;
            ASSERT_TRUE(agx3_stage_copy_source(
               AGX_ISA_ENCODING_AGX3B, data_dest[c], 4 + 2 * c, false, 2,
               c == 0 ? 1 : 0, copy, &len));
            append(staging, copy, len);
         }
         ASSERT_GE(std::distance(code.begin(), first), staging.size());
         EXPECT_TRUE(std::equal(staging.begin(), staging.end(),
                                first - staging.size()));
         EXPECT_EQ(std::search(first + sizeof(imgwci), code.end(), imgwci,
                               imgwci + sizeof(imgwci)),
                   code.end());
      } else {
         EXPECT_TRUE(code.empty());
      }
   }
}

TEST_F(AGX3PackBinary, SampledRgba8WriteUsesNativeStateOneProtocol)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_COMPUTE;
   agx_index sample = agx_register(0, AGX_SIZE_32);
   sample.channels_m1 = 3;
   agx_index sample_coord = agx_register(8, AGX_SIZE_32);
   sample_coord.channels_m1 = 1;
   agx_instr *read = agx_texture_sample_to(
      b, sample, sample_coord, agx_immediate_f(0.0f), agx_immediate(0),
      agx_immediate(0), agx_immediate(0), agx_null(), AGX_DIM_2D,
      AGX_LOD_MODE_LOD_MIN, 0, false, false, false, AGX_GATHER_NONE, false);
   read->mask = 15;

   agx_index write_coord = agx_register(12, AGX_SIZE_32);
   write_coord.channels_m1 = 1;
   agx_instr *write = agx_image_write(
      b, sample, write_coord, agx_immediate(0), agx_immediate(0),
      agx_immediate(1), AGX_DIM_2D, false);
   write->format = AGX_FORMAT_U8NORM;
   agx_stop(b);

   const std::vector<uint8_t> code = pack(b);
   static const uint8_t imgwci[AGX3_IMAGE_WRITE_LENGTH] = {
      0xd7, 0x20, 0x54, 0x00, 0x20, 0x40, 0x00, 0x01,
      0x00, 0x04, 0x00, 0x00, 0x88, 0x3a, 0x09, 0x00,
   };
   ASSERT_GE(code.size(), sizeof(imgwci) + 4);
   EXPECT_TRUE(std::equal(imgwci, imgwci + sizeof(imgwci),
                          code.end() - 4 - sizeof(imgwci)));
   EXPECT_TRUE(b->shader->agx3_staging_known);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 16);
}

TEST_F(AGX3PackBinary, ComputedRgb10WriteUsesNativeStateOneProtocol)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_COMPUTE;
   agx_index data = agx_register(16, AGX_SIZE_32);
   data.channels_m1 = 3;
   agx_index coord = agx_register(12, AGX_SIZE_32);
   coord.channels_m1 = 1;
   agx_instr *write = agx_image_write(
      b, data, coord, agx_immediate(0), agx_immediate(0), agx_immediate(1),
      AGX_DIM_2D, false);
   write->format = AGX_FORMAT_RGB10A2;
   agx_stop(b);

   const std::vector<uint8_t> code = pack(b);
   static const uint8_t imgwci[AGX3_IMAGE_WRITE_LENGTH] = {
      0xd7, 0x20, 0x54, 0x00, 0x20, 0x40, 0x00, 0x01,
      0x00, 0x04, 0x00, 0x00, 0x88, 0x3a, 0x09, 0x00,
   };
   EXPECT_NE(std::search(code.begin(), code.end(), imgwci,
                         imgwci + sizeof(imgwci)),
             code.end());
}
