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
#include "agx3_funop_core.h"
#include "agx3_iadd_core.h"
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
#include "agx3_select_core.h"
#include "agx3_vary_core.h"
#include "agx3_zs_core.h"

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
      if (enc == AGX_ISA_ENCODING_AGX3B) {
         const uint16_t src[] = {0, 2};
         const enum agx3_alu_hint hints[] = {
            AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE};
         uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
         ASSERT_TRUE(agx3_alu_integer_reg(enc, AGX3_ALU_IADD, 4, src,
                                          hints, 0, bytes, &len));
         append(want, bytes, len);
      } else {
         append(want, ia.bytes, ia.length);
      }
      struct agx3_encoded_end en;
      ASSERT_TRUE(agx3_pack_end(enc, &en));
      append(want, en.bytes, en.length);

      EXPECT_EQ(pack(b), want);
   }
}

TEST_F(AGX3PackBinary, G17PRG16IntegerEOTUsesExactNativeI16Program)
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

TEST_F(AGX3PackBinary, G17PRGB565EOTUsesExactNativeSelector35)
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
      0xa8, 0x35, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
   };

   auto build = [&](enum agx_format format) {
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
                            agx_register(4, AGX_SIZE_32), coords, format,
                            AGX_DIM_2D, true);
      agx_stop(b);
      return b;
   };

   key.g17p_rt0_eot_words = 2;
   key.g17p_rt0_eot_rgb565 = true;
   agx_builder *b = build(AGX_FORMAT_I16);
   EXPECT_EQ(pack(b), std::vector<uint8_t>(std::begin(native),
                                           std::end(native)));
   EXPECT_EQ(b->shader->max_reg, 7);
   EXPECT_TRUE(b->shader->agx3_staging_known);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 8);

   key.g17p_rt0_eot_words = 1;
   EXPECT_TRUE(pack(build(AGX_FORMAT_I16), false).empty());
   key.g17p_rt0_eot_words = 2;
   EXPECT_TRUE(pack(build(AGX_FORMAT_F16), false).empty());
   key.g17p_rt0_eot_i32 = true;
   EXPECT_TRUE(pack(build(AGX_FORMAT_I16), false).empty());
}

TEST_F(AGX3PackBinary, G17PLayeredU8NormEOTUsesExactNativeArrayWire)
{
   key.g17p_rt0_eot_words = 0;
   static const uint8_t native[] = {
      0x0c,0xa0,0x10,0x06,0x1c,0xa1,0x10,0x26,
      0x2c,0x9e,0x10,0x46,0x3b,0x00,0x00,0x00,
      0x4c,0x80,0x02,0x00,0x00,0x20,0x00,0x00,
      0x9f,0x11,0x54,0x08,0x02,0x00,0x00,0x88,0x11,0x04,
      0x5c,0x80,0x02,0x00,0x00,0x40,0x00,0x00,
      0x9f,0x11,0x54,0x0a,0x02,0x00,0x08,0x88,0x11,0x04,
      0x6c,0x80,0x02,0x00,0x00,0x80,0x00,0x00,
      0x9f,0x11,0x54,0x0c,0x02,0x00,0x10,0x88,0x11,0x04,
      0x97,0x15,0x54,0x04,0x00,0x18,0x10,0x00,0xf0,0xc0,0x00,0x00,
      0x97,0x05,0x54,0x06,0x00,0x08,0x00,0x00,0xd0,0x40,0x10,0x00,
      0x57,0x10,0x54,0x00,0x00,0x00,0x00,0x80,0x28,0x75,0x00,0x00,
      0x0e,0x00,0x00,0x00,
   };

   auto build = [&](enum agx_dim dim, unsigned coord_base) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_COMPUTE;
      agx_get_sr_to(b, agx_register(0, AGX_SIZE_32),
                    AGX_SR_THREAD_POSITION_IN_GRID_X);
      agx_get_sr_to(b, agx_register(2, AGX_SIZE_32),
                    AGX_SR_THREAD_POSITION_IN_GRID_Y);
      agx_get_sr_to(b, agx_register(4, AGX_SIZE_32),
                    AGX_SR_THREADGROUP_POSITION_IN_GRID_Z);
      agx_mov_imm_to(b, agx_register(6, AGX_SIZE_32), 0);
      for (unsigned lane = 0; lane < 3; ++lane) {
         agx_index src = agx_register(2 * lane, AGX_SIZE_32);
         src.discard = true;
         agx_bitop_to(b, agx_register(8 + 2 * lane, AGX_SIZE_32), src,
                      agx_immediate(0), AGX3_SEL_TT_MOV);
      }
      agx_index offset = agx_register(6, AGX_SIZE_32);
      offset.discard = true;
      agx_index coords = agx_register(coord_base, AGX_SIZE_32);
      coords.channels_m1 = 2;
      coords.discard = true;
      agx_block_image_store(b, agx_immediate(0), agx_immediate(0), offset,
                            coords, AGX_FORMAT_U8NORM, dim, true);
      agx_stop(b);
      return b;
   };

   agx_builder *good = build(AGX_DIM_2D_ARRAY, 8);
   EXPECT_EQ(pack(good), std::vector<uint8_t>(std::begin(native),
                                              std::end(native)));
   EXPECT_EQ(good->shader->max_reg, 13);
   EXPECT_TRUE(good->shader->agx3_staging_known);
   EXPECT_EQ(good->shader->agx3_staging_halfs, 8);

   EXPECT_TRUE(pack(build(AGX_DIM_2D, 8), false).empty());
   EXPECT_TRUE(pack(build(AGX_DIM_2D_ARRAY, 10), false).empty());
}

TEST_F(AGX3PackBinary, G17PRGBA8MRTEOTUsesExactNativeProgram)
{
   static const uint8_t native[] = {
      0x1c,0x9c,0x10,0x06,0x24,0xa8,0x14,0x06,
      0x0c,0x9d,0x10,0x26,0x24,0xa9,0x10,0x26,
      0x9f,0x10,0x54,0x02,0x03,0x04,0x14,0x00,
      0x50,0x26,0x0a,0x00,0x9f,0x20,0x54,0x00,
      0x03,0x00,0x10,0x00,0x50,0x26,0x0a,0x00,
      0x97,0x05,0x54,0x04,0x00,0x08,0x00,0x00,
      0xd0,0x40,0x10,0x00,0x0b,0x82,0x21,0x04,
      0x1b,0x80,0x21,0x04,0x97,0x05,0x54,0x0c,
      0x00,0x10,0x00,0x00,0xd0,0x44,0x10,0x00,
      0x4b,0x03,0x2f,0x00,0x42,0x00,0x00,0x04,
      0x00,0x00,0x5b,0x01,0x2f,0x00,0x42,0x00,
      0x00,0x04,0x00,0x00,0x57,0x10,0x54,0x00,
      0x00,0x00,0x00,0x80,0xa8,0x75,0x00,0x00,
      0x57,0x20,0x54,0x08,0x00,0x10,0x00,0x90,
      0xa8,0x75,0x00,0x00,0x0e,0x00,0x00,0x00,
   };

   auto build = [&](unsigned second_index, unsigned second_offset,
                    enum agx_format second_format) {
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
                            AGX_FORMAT_U8NORM, AGX_DIM_2D, true);
      agx_mov_imm_to(b, agx_register(4, AGX_SIZE_32), second_offset);
      agx_block_image_store(b, agx_immediate(0),
                            agx_immediate(second_index),
                            agx_register(4, AGX_SIZE_32), coords,
                            second_format, AGX_DIM_2D, true);
      agx_stop(b);
      return b;
   };

   agx_builder *good = build(1, 4, AGX_FORMAT_U8NORM);
   EXPECT_EQ(pack(good), std::vector<uint8_t>(std::begin(native),
                                              std::end(native)));
   EXPECT_EQ(good->shader->max_reg, 7);
   EXPECT_TRUE(good->shader->agx3_staging_known);
   EXPECT_EQ(good->shader->agx3_staging_halfs, 16);

   EXPECT_TRUE(pack(build(0, 4, AGX_FORMAT_U8NORM), false).empty());
   EXPECT_TRUE(pack(build(1, 8, AGX_FORMAT_U8NORM), false).empty());
   EXPECT_TRUE(pack(build(1, 4, AGX_FORMAT_S8NORM), false).empty());
}

TEST_F(AGX3PackBinary, G17PF16EOTUsesExactNativeI16Program)
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
                         AGX_FORMAT_F16, AGX_DIM_2D, true);
   agx_stop(b);

   EXPECT_EQ(pack(b), std::vector<uint8_t>(std::begin(native),
                                           std::end(native)));
   EXPECT_EQ(b->shader->max_reg, 7);
   EXPECT_TRUE(b->shader->agx3_staging_known);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 8);
}

TEST_F(AGX3PackBinary, G17PI8EOTUsesExactOneWordNativeProgram)
{
   /* Fresh scalar R8_UINT key0x04020500 and scalar R8_SINT key0x04020400
    * both produce this exact program. Its 68-byte text SHA256 is
    * 7a5e8ed6d0d8fca6b23b335f416f069c0942332ad7becf7ea5ec2704ee59ce95. */
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

   for (unsigned words = 0; words <= 2; ++words) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_COMPUTE;
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
                            AGX_FORMAT_I8, AGX_DIM_2D, true);
      agx_stop(b);

      if (words == 1) {
         EXPECT_EQ(pack(b), std::vector<uint8_t>(std::begin(native),
                                                 std::end(native)));
         EXPECT_EQ(b->shader->max_reg, 7);
         EXPECT_TRUE(b->shader->agx3_staging_known);
         EXPECT_EQ(b->shader->agx3_staging_halfs, 8);
      } else {
         EXPECT_TRUE(pack(b, false).empty());
      }
   }
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
   key.g17p_rt0_eot_words = 1;

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

TEST_F(AGX3PackBinary, G17PRGBA16SNORMEOTUsesExactTwoWordNativeProgram)
{
   key.g17p_rt0_eot_words = 2;
   key.g17p_rt0_eot_norm16_selector_95 = true;

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
      0xa8, 0x95, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
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

TEST_F(AGX3PackBinary, G17PScalarNorm16EOTUsesExactNativeProgram)
{
   key.g17p_rt0_eot_words = 1;
   key.g17p_rt0_eot_norm16_selector_95 = true;

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
      0xa8, 0x95, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
   };

   for (enum agx_format format : {AGX_FORMAT_U16NORM,
                                  AGX_FORMAT_S16NORM}) {
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
                            agx_register(4, AGX_SIZE_32), coords, format,
                            AGX_DIM_2D, true);
      agx_stop(b);

      EXPECT_EQ(pack(b), std::vector<uint8_t>(std::begin(native),
                                              std::end(native)));
   }
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
      /* f14 is the saturate bit and it sits on the same encoding point as the
       * slot widths, so `saturate(a * b)` over a mixed pair is one
       * instruction: Apple's own is 09011d0401000082, f14 = 1 with f43 = 1.
       * The modifier rows above it stay refused -- no witness puts abs or neg
       * on a MULTIPLY's half slot. */
      if (mode == 8)
         EXPECT_FALSE(pack(b).empty()) << "mode=" << mode;
      else
         EXPECT_TRUE(pack(b, false).empty()) << "mode=" << mode;
   }
}

TEST_F(AGX3PackBinary, WebGLAbsScaleMixedWidthMatchesNative)
{
   for (bool half_first : {false, true}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index wide = agx_register(0, AGX_SIZE_32);
      agx_index half = agx_register(8, AGX_SIZE_16);
      wide.abs = wide.discard = true;
      half.cache = true;
      agx_fmul_to(b, agx_register(0, AGX_SIZE_32),
                  half_first ? half : wide, half_first ? wide : half);
      agx_stop(b);

      /* Reduced post-RA form of Skia's `abs(testInputs) * 100.0`. */
      const std::vector<uint8_t> native = {
         0x09, 0x01, 0x0d, 0x88, 0x02, 0x00, 0x00, 0x00,
         0x01, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
      };
      EXPECT_EQ(pack(b), native) << "half_first=" << half_first;
   }
}

TEST_F(AGX3PackBinary, WebGLAbsScaleMixedWidthRefusesUngroundedModes)
{
   for (unsigned mode = 0; mode < 16; ++mode) {
      agx_builder *b = builder(mode == 0 ? AGX_ISA_ENCODING_AGX3C
                                         : AGX_ISA_ENCODING_AGX3B);
      agx_index dest = agx_register(0, AGX_SIZE_32);
      agx_index wide = agx_register(0, AGX_SIZE_32);
      agx_index half = agx_register(8, AGX_SIZE_16);
      wide.abs = wide.discard = true;
      half.cache = true;

      if (mode == 1)
         dest.size = AGX_SIZE_16;
      if (mode == 2)
         wide.neg = true;
      if (mode == 3) {
         wide.abs = false;
         half.abs = true;
      }
      if (mode == 4)
         half.neg = true;
      if (mode == 5)
         wide.cache = true;
      if (mode == 6)
         half.discard = true;
      if (mode == 7)
         wide.kill = true;
      if (mode == 8)
         half.kill = true;
      if (mode == 9)
         dest.value = 1;
      if (mode == 10)
         wide.value = 1;
      if (mode == 11)
         dest.channels_m1 = 1;
      if (mode == 13)
         wide.memory = true;
      if (mode == 14)
         half.memory = true;
      if (mode == 15)
         dest.memory = true;

      agx_instr *mul = agx_fmul_to(b, dest, wide, half);
      mul->saturate = mode == 12;
      agx_stop(b);
      /* Saturation is f14 on the same encoding point as the slot widths and
       * the modifier bits: Apple's saturate(a * b) over a mixed pair is
       * 09011d0401000082. Everything else here is still out of domain. */
      if (mode == 12)
         EXPECT_FALSE(pack(b).empty()) << "mode=" << mode;
      else
         EXPECT_TRUE(pack(b, false).empty()) << "mode=" << mode;
   }
}

TEST_F(AGX3PackBinary, HalfMultiplyMinifloatMatchesNativeAndRefusesModifier)
{
   for (unsigned i = 0; i < 2; ++i) {
      for (unsigned order = 0; order < 2; ++order) {
         agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
         agx_index src = agx_register(2, AGX_SIZE_16);
         src.discard = true;
         agx_index dest = agx_register(2, AGX_SIZE_16);
         agx_index immediate = agx_immediate(i ? 0x50 : 0x40);
         agx_hmul_to(b, dest, order ? immediate : src,
                     order ? src : immediate);
         agx_stop(b);
         const auto code = pack(b);
         const uint8_t times2[] = {0x10, 0xc0, 0x15, 0x02, 0x80, 0x00};
         const uint8_t times4[] = {0x10, 0xd0, 0x15, 0x02, 0x80, 0x00};
         ASSERT_EQ(code.size(), 10u);
         EXPECT_EQ(memcmp(code.data(), i ? times4 : times2, 6), 0);
      }
   }

   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index modified = agx_register(2, AGX_SIZE_16);
   modified.abs = true;
   agx_hmul_to(b, agx_register(2, AGX_SIZE_16), modified,
               agx_immediate(0x40));
   agx_stop(b);
   EXPECT_TRUE(pack(b, false).empty());
}

TEST_F(AGX3PackBinary, MinifloatMixedSelectMatchesNativeAndRefusesModifier)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index compare = agx_register(4, AGX_SIZE_32);
   compare.cache = true;
   agx_index selected = agx_register(2, AGX_SIZE_32);
   selected.cache = true;
   agx_fcmpsel_to(b, agx_register(0, AGX_SIZE_32), compare,
                  agx_immediate(0x30), selected, agx_immediate(0),
                  AGX_FCOND_LT);
   agx_stop(b);
   const uint8_t native[] = {
      0x02, 0x85, 0x03, 0xb1, 0x02, 0x82, 0x03, 0x02, 0x20, 0x80,
   };
   const auto code = pack(b);
   ASSERT_EQ(code.size(), sizeof(native) + 4);
   EXPECT_EQ(memcmp(code.data(), native, sizeof(native)), 0);

   /* Exact post-RA ext.frag mask shape. */
   b = builder(AGX_ISA_ENCODING_AGX3B);
   compare = agx_register(14, AGX_SIZE_32);
   agx_index dest = agx_register(1, AGX_SIZE_16);
   dest.cache = true;
   agx_fcmpsel_to(b, dest, compare,
                  agx_immediate(0x30), agx_immediate(255), agx_immediate(0),
                  AGX_FCOND_EQ);
   agx_stop(b);
   EXPECT_GT(pack(b).size(), 4u);

   /* Exact post-RA glitch.frag shape. */
   b = builder(AGX_ISA_ENCODING_AGX3B);
   compare = agx_register(22, AGX_SIZE_32);
   compare.cache = true;
   selected = agx_register(2, AGX_SIZE_32);
   selected.discard = true;
   dest = agx_register(2, AGX_SIZE_32);
   dest.cache = true;
   agx_fcmpsel_to(b, dest, compare,
                  agx_immediate(0x40), agx_immediate(0), selected,
                  AGX_FCOND_LT);
   agx_stop(b);
   EXPECT_GT(pack(b).size(), 4u);

   /* An absolute comparator against a NON-ZERO minifloat. The refusal here
    * was the admission gate, not the encoding: against Apple's own
    * (fabs(a) < c) ? 1.0h : 0.0h, 1203190523b00300208000800100, the
    * immediate form 120309a123b00302208000800100 changes only f59 (the
    * second slot's kind) and the operand field (f52 -> f51). */
   b = builder(AGX_ISA_ENCODING_AGX3B);
   compare = agx_register(2, AGX_SIZE_32);
   compare.abs = true;
   agx_fcmpsel_to(b, agx_register(2, AGX_SIZE_16), compare,
                  agx_immediate(0x30), agx_immediate(255), agx_immediate(0),
                  AGX_FCOND_EQ);
   agx_stop(b);
   EXPECT_FALSE(pack(b).empty());
}

TEST_F(AGX3PackBinary, GtkNegatedMinifloatSelectCanonicalizesExactly)
{
   const enum agx_isa_encoding enc = AGX_ISA_ENCODING_AGX3B;

   /* Exact post-RA shape from current GTK gskgpuboxshadow.frag. Apple emits
    * the byte-identical x > +0 form for source -x < -0. */
   agx_builder *b = builder(enc);
   agx_index compare = agx_register(44, AGX_SIZE_32);
   compare.neg = true;
   agx_fcmpsel_to(b, agx_register(44, AGX_SIZE_16), compare,
                  agx_immediate(0x80), agx_immediate(1), agx_immediate(0),
                  AGX_FCOND_LT);
   agx_stop(b);
   const std::vector<uint8_t> gtk = {
      0x62, 0x2d, 0x45, 0x81, 0x22, 0x81, 0x02, 0x02, 0x20, 0x80,
      0x0e, 0x00, 0x00, 0x00,
   };
   EXPECT_EQ(pack(b), gtk);

   /* Sign-flipping the threshold and reversing the relation are independent
    * of zero. These are the no-wait forms of fresh Apple -x < +1 and
    * -x >= -0.5 witnesses. */
   b = builder(enc);
   compare = agx_register(2, AGX_SIZE_32);
   compare.neg = true;
   compare.discard = true;
   agx_fcmpsel_to(b, agx_register(2, AGX_SIZE_16), compare,
                  agx_immediate(0x30), agx_immediate(1), agx_immediate(0),
                  AGX_FCOND_LT);
   agx_stop(b);
   const std::vector<uint8_t> minus_one = {
      0x12, 0x03, 0x1d, 0xb1, 0x22, 0x81, 0x02, 0x02, 0x20, 0x80,
      0x0e, 0x00, 0x00, 0x00,
   };
   EXPECT_EQ(pack(b), minus_one);

   b = builder(enc);
   compare = agx_register(2, AGX_SIZE_32);
   compare.neg = true;
   compare.discard = true;
   agx_fcmpsel_to(b, agx_register(2, AGX_SIZE_16), compare,
                  agx_immediate(0xa0), agx_immediate(1), agx_immediate(0),
                  AGX_FCOND_GE);
   agx_stop(b);
   const std::vector<uint8_t> plus_half = {
      0x12, 0x03, 0x0d, 0xa1, 0x26, 0x81, 0x03, 0x02, 0x20, 0x80,
      0x0e, 0x00, 0x00, 0x00,
   };
   EXPECT_EQ(pack(b), plus_half);

   /* Current gskgpulineargradient has the minifloat in the first compare
    * slot. Fresh Apple kernels normalize 1 < x to x > 1, and 1 < -x to
    * x < -1. */
   b = builder(enc);
   compare = agx_register(2, AGX_SIZE_32);
   compare.discard = true;
   agx_fcmpsel_to(b, agx_register(2, AGX_SIZE_16), agx_immediate(0x30),
                  compare, agx_immediate(1), agx_immediate(0),
                  AGX_FCOND_LT);
   agx_stop(b);
   const std::vector<uint8_t> immediate_first = {
      0x12, 0x03, 0x0d, 0xb1, 0x22, 0x81, 0x02, 0x02, 0x20, 0x80,
      0x0e, 0x00, 0x00, 0x00,
   };
   EXPECT_EQ(pack(b), immediate_first);

   b = builder(enc);
   compare = agx_register(2, AGX_SIZE_32);
   compare.neg = true;
   compare.discard = true;
   agx_fcmpsel_to(b, agx_register(2, AGX_SIZE_16), agx_immediate(0x30),
                  compare, agx_immediate(1), agx_immediate(0),
                  AGX_FCOND_LT);
   agx_stop(b);
   const std::vector<uint8_t> immediate_first_negated = {
      0x12, 0x03, 0x1d, 0xb1, 0x22, 0x81, 0x03, 0x02, 0x20, 0x80,
      0x0e, 0x00, 0x00, 0x00,
   };
   EXPECT_EQ(pack(b), immediate_first_negated);

   /* Exact full-result post-RA forms reached later in current GTK's linear
    * and conic gradient shaders. */
   b = builder(enc);
   compare = agx_register(22, AGX_SIZE_32);
   compare.cache = true;
   agx_fcmpsel_to(b, agx_register(10, AGX_SIZE_32), compare,
                  agx_immediate(0x30), agx_immediate(0),
                  agx_register(10, AGX_SIZE_32), AGX_FCOND_GE);
   agx_stop(b);
   const std::vector<uint8_t> linear_full = {
      0x52, 0x97, 0x03, 0xb1, 0x26, 0x80, 0x02, 0x02, 0x00, 0x0a,
      0x0e, 0x00, 0x00, 0x00,
   };
   EXPECT_EQ(pack(b), linear_full);

   b = builder(enc);
   compare = agx_register(12, AGX_SIZE_32);
   compare.cache = true;
   agx_index fallback = agx_register(24, AGX_SIZE_32);
   fallback.discard = true;
   agx_fcmpsel_to(b, agx_register(24, AGX_SIZE_32), agx_immediate(0x30),
                  compare, agx_immediate(0), fallback, AGX_FCOND_GE);
   agx_stop(b);
   const std::vector<uint8_t> conic_full = {
      0xc2, 0x8d, 0x03, 0xb1, 0x26, 0x80, 0x03, 0x02, 0x80, 0x18,
      0x0e, 0x00, 0x00, 0x00,
   };
   EXPECT_EQ(pack(b), conic_full);

   /* Inverting the predicate swaps the selected bytes without changing the
    * compare canonicalization. */
   b = builder(enc);
   compare = agx_register(44, AGX_SIZE_32);
   compare.neg = true;
   agx_instr *fsel = agx_fcmpsel_to(
      b, agx_register(44, AGX_SIZE_16), compare, agx_immediate(0x80),
      agx_immediate(1), agx_immediate(0), AGX_FCOND_LT);
   fsel->invert_cond = true;
   agx_stop(b);
   const std::vector<uint8_t> inverted = {
      0x62, 0x2d, 0x45, 0x81, 0x22, 0x80, 0x02, 0x02, 0x20, 0x81,
      0x0e, 0x00, 0x00, 0x00,
   };
   EXPECT_EQ(pack(b), inverted);
}

TEST_F(AGX3PackBinary, GtkNegatedMinifloatSelectRefusesUngroundedForms)
{
   enum bad_form {
      WRONG_ARCH,
      FULL_DEST,
      DEST_VECTOR,
      DEST_MEMORY,
      COMPARE_HALF,
      COMPARE_ODD,
      COMPARE_ABS,
      COMPARE_ABS_NEG,
      COMPARE_KILL,
      COMPARE_HINT_CONFLICT,
      THRESHOLD_OVERFLOW,
      THRESHOLD_ABS,
      THRESHOLD_NEG,
      THRESHOLD_KILL,
      THRESHOLD_CACHE,
      TRUE_OVERFLOW,
      FALSE_OVERFLOW,
      TRUE_ABS,
      FALSE_NEG,
      TRUE_KILL,
      FALSE_MEMORY,
      TRUE_CACHE,
      NAN_CONDITION_LT,
      NAN_CONDITION_GT,
      SATURATE,
   };

   for (unsigned form = WRONG_ARCH; form <= SATURATE; ++form) {
      SCOPED_TRACE(form);
      agx_builder *b = builder(form == WRONG_ARCH ? AGX_ISA_ENCODING_AGX3C
                                                  : AGX_ISA_ENCODING_AGX3B);
      agx_index dest = agx_register(44, AGX_SIZE_16);
      agx_index compare = agx_register(44, AGX_SIZE_32);
      agx_index threshold = agx_immediate(0x80);
      agx_index if_true = agx_immediate(1);
      agx_index if_false = agx_immediate(0);
      compare.neg = true;
      enum agx_fcond condition = AGX_FCOND_LT;

      switch ((enum bad_form)form) {
      case WRONG_ARCH: break;
      case FULL_DEST: dest.size = AGX_SIZE_32; break;
      case DEST_VECTOR: dest.channels_m1 = 1; break;
      case DEST_MEMORY: dest.memory = true; break;
      case COMPARE_HALF: compare.size = AGX_SIZE_16; break;
      case COMPARE_ODD: compare.value = 45; break;
      case COMPARE_ABS: compare.abs = true; compare.neg = false; break;
      case COMPARE_ABS_NEG: compare.abs = true; break;
      case COMPARE_KILL: compare.kill = true; break;
      case COMPARE_HINT_CONFLICT:
         compare.cache = compare.discard = true;
         break;
      case THRESHOLD_OVERFLOW: threshold.value = 256; break;
      case THRESHOLD_ABS: threshold.abs = true; break;
      case THRESHOLD_NEG: threshold.neg = true; break;
      case THRESHOLD_KILL: threshold.kill = true; break;
      case THRESHOLD_CACHE: threshold.cache = true; break;
      case TRUE_OVERFLOW: if_true.value = 256; break;
      case FALSE_OVERFLOW: if_false.value = 256; break;
      case TRUE_ABS: if_true.abs = true; break;
      case FALSE_NEG: if_false.neg = true; break;
      case TRUE_KILL: if_true.kill = true; break;
      case FALSE_MEMORY: if_false.memory = true; break;
      case TRUE_CACHE: if_true.cache = true; break;
      case NAN_CONDITION_LT: condition = AGX_FCOND_LTN; break;
      case NAN_CONDITION_GT: condition = AGX_FCOND_GTN; break;
      case SATURATE: break;
      }

      agx_instr *I = agx_fcmpsel_to(b, dest, compare, threshold, if_true,
                                     if_false, condition);
      I->saturate = form == SATURATE;
      agx_stop(b);
      /*
       * Eight of these were recording what had been measured rather than what
       * the encoding refuses, and each now has an Apple witness:
       *
       *   FULL_DEST       the result width is a field, f11 / f25
       *   COMPARE_HALF    the comparator widths are f43 and f60, and f65
       *                   composes with them: 12051d06850302c8
       *   COMPARE_ABS     f47 with a minifloat comparator:
       *                   120309a123b00302208000800100
       *   COMPARE_ABS_NEG f47 and f65 on one instruction:
       *                   12051d0787030208800200800100
       *   THRESHOLD_ABS   a minifloat's sign is bit 7, so |k| and -k are
       *   THRESHOLD_NEG   constant folding, which is how Apple spells
       *                   `-a < 0.5f` as `a > -0.5f`: 120319a122b002c22080
       *   NAN_CONDITION_* ltn/gtn are the ordered relation with the two
       *                   selected values exchanged, which this path already
       *                   did for a zero comparator
       */
      const bool grounded =
         form == FULL_DEST || form == COMPARE_HALF || form == COMPARE_ABS ||
         form == COMPARE_ABS_NEG || form == THRESHOLD_ABS ||
         form == THRESHOLD_NEG || form == NAN_CONDITION_LT ||
         form == NAN_CONDITION_GT;
      if (grounded)
         EXPECT_FALSE(pack(b).empty());
      else
         EXPECT_TRUE(pack(b, false).empty());
   }
}

TEST_F(AGX3PackBinary, AlphaCoverageBoundarySelectMatchesNative)
{
   enum modifier {
      NONE,
      COMPARE_ABS,
      COMPARE_NEG,
      COMPARE_KILL,
      COMPARE_CONFLICTING_HINTS,
      THRESHOLD_ABS,
      SELECTED_ABS,
      SELECTED_NEG,
      SELECTED_KILL,
      SATURATE,
   };

   auto build = [&](enum modifier kind, bool fail_closed) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index compare = agx_register(2, AGX_SIZE_32);
      agx_index threshold = agx_immediate(0x6f); /* 15.5 */
      agx_index selected = agx_register(0, AGX_SIZE_32);
      compare.discard = selected.discard = true;

      switch (kind) {
      case NONE: break;
      case COMPARE_ABS: compare.abs = true; break;
      case COMPARE_NEG: compare.neg = true; break;
      case COMPARE_KILL: compare.kill = true; break;
      case COMPARE_CONFLICTING_HINTS: compare.cache = true; break;
      case THRESHOLD_ABS: threshold.abs = true; break;
      case SELECTED_ABS: selected.abs = true; break;
      case SELECTED_NEG: selected.neg = true; break;
      case SELECTED_KILL: selected.kill = true; break;
      case SATURATE: break;
      }

      agx_instr *I = agx_fcmpsel_to(
         b, agx_register(14, AGX_SIZE_32), compare, threshold, selected,
         agx_immediate(0), AGX_FCOND_GE);
      I->saturate = kind == SATURATE;
      agx_stop(b);
      return pack(b, !fail_closed);
   };

   /* Exact post-RA alpha-to-coverage shape. The physical destination needs
    * Apple's ten-byte ep22 form; the terminal instruction follows it. */
   const uint8_t expected[] = {
      0x72, 0x03, 0x0b, 0xef, 0x86, 0x00, 0x12, 0x02, 0x20, 0x80,
      0x0e, 0x00, 0x00, 0x00,
   };
   EXPECT_EQ(build(NONE, false),
             std::vector<uint8_t>(expected, expected + sizeof(expected)));

   /* A fresh full-result Apple witness grounds the same negate
    * canonicalization. */
   const uint8_t negated[] = {
      0x72, 0x03, 0x1b, 0xef, 0x86, 0x00, 0x13, 0x02, 0x20, 0x80,
      0x0e, 0x00, 0x00, 0x00,
   };
   EXPECT_EQ(build(COMPARE_NEG, false),
             std::vector<uint8_t>(negated, negated + sizeof(negated)));

   /* An absolute comparator against a minifloat, and an absolute value taken
    * on that minifloat, are both grounded now: f47 composes with an immediate
    * comparator (120309a123b00302208000800100) and a minifloat's sign is bit
    * 7, so |k| is constant folding rather than an encoding question. */
   EXPECT_FALSE(build(COMPARE_ABS, false).empty());
   EXPECT_FALSE(build(THRESHOLD_ABS, false).empty());

   for (unsigned bad = COMPARE_ABS; bad <= SATURATE; ++bad) {
      if (bad == COMPARE_NEG || bad == COMPARE_ABS || bad == THRESHOLD_ABS)
         continue;
      SCOPED_TRACE(bad);
      EXPECT_TRUE(build((enum modifier)bad, true).empty());
   }
}

TEST_F(AGX3PackBinary, HyprlandRemainingFloatSelectsMatchNativeForms)
{
   const enum agx_isa_encoding enc = AGX_ISA_ENCODING_AGX3B;
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(enc, &end));

   /* Exact post-RA blurprepare.frag shape: r4 >= 0.25 ? r5 : 0. */
   agx_builder *b = builder(enc);
   agx_index compare = agx_register(8, AGX_SIZE_32);
   agx_index selected = agx_register(10, AGX_SIZE_32);
   compare.cache = selected.cache = true;
   agx_fcmpsel_to(b, agx_register(12, AGX_SIZE_32), compare,
                  agx_immediate(0x20), selected, agx_immediate(0),
                  AGX_FCOND_GE);
   agx_stop(b);

   const uint16_t blur_selected[] = {10, 0};
   const bool blur_immediate[] = {false, true};
   const enum agx3_alu_hint blur_hints[] = {
      AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_NONE,
   };
   ASSERT_TRUE(agx3_alu_fcmp_minifloat_select_mixed(
      enc, 12, 8, 0x20, blur_selected, blur_immediate, blur_hints,
      AGX_FCOND_GE, false, false, 0, bytes, &len));
   std::vector<uint8_t> want;
   append(want, bytes, len);
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);

   /* Apple canonicalizes inner_glow's `-a < b` as `b > -a`. */
   b = builder(enc);
   agx_index a = agx_register(8, AGX_SIZE_32);
   a.neg = true;
   agx_index c = agx_register(16, AGX_SIZE_32);
   agx_fcmpsel_to(b, agx_register(1, AGX_SIZE_16), a, c,
                  agx_immediate(255), agx_immediate(0), AGX_FCOND_LT);
   agx_stop(b);

   const uint16_t glow_src[] = {16, 8};
   const bool glow_width[] = {false, false};
   const enum agx3_alu_hint glow_hints[] = {
      AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE,
   };
   ASSERT_TRUE(agx3_alu_fcmp_select_imm_width_neg_second(
      enc, 1, glow_src, glow_width, glow_hints, AGX_FCOND_GT, 255, 0,
      true, false, 0, bytes, &len));
   want.clear();
   append(want, bytes, len);
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);

   /* The negate on the OTHER operand is f65 with nothing rearranged: it is
    * the slot the field names, so the relation and the operand order both
    * stand. `a < -b` is Apple's 02011d03228103c82080, one bit away from its
    * 02011d03228102c02080 twin for `a > b`. */
   b = builder(enc);
   a = agx_register(8, AGX_SIZE_32);
   c = agx_register(16, AGX_SIZE_32);
   c.neg = true;
   agx_fcmpsel_to(b, agx_register(1, AGX_SIZE_16), a, c,
                  agx_immediate(255), agx_immediate(0), AGX_FCOND_LT);
   agx_stop(b);

   const uint16_t second_src[] = {8, 16};
   ASSERT_TRUE(agx3_alu_fcmp_select_imm_width_neg_second(
      enc, 1, second_src, glow_width, glow_hints, AGX_FCOND_LT, 255, 0,
      true, false, 0, bytes, &len));
   want.clear();
   append(want, bytes, len);
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);

   /* Mixing the negate with an absolute source still has no witness: ABS is
    * a longer endpoint that carries no f65, so it fails closed rather than
    * dropping one of the two modifiers. */
   b = builder(enc);
   a = agx_register(8, AGX_SIZE_32);
   a.abs = true;
   c = agx_register(16, AGX_SIZE_32);
   c.neg = true;
   agx_fcmpsel_to(b, agx_register(1, AGX_SIZE_16), a, c,
                  agx_immediate(255), agx_immediate(0), AGX_FCOND_LT);
   agx_stop(b);
   EXPECT_TRUE(pack(b, false).empty());
}

TEST_F(AGX3PackBinary, MinifloatSpecializationDeclinesOtherGroundedForms)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index compare = agx_register(8, AGX_SIZE_32);
   agx_index selected = agx_register(14, AGX_SIZE_32);
   compare.discard = selected.discard = true;

   /* Cube lowering produces this zero-threshold GE form.  It is not the
    * specialized Hyprland GE encoding, but the later general zero-select
    * path has its own grounded representation and must get a chance at it. */
   agx_fcmpsel_to(b, agx_register(8, AGX_SIZE_32), compare,
                  agx_immediate(0), agx_immediate(0), selected,
                  AGX_FCOND_GE);
   agx_stop(b);
   EXPECT_GT(pack(b).size(), 4u);
}

TEST_F(AGX3PackBinary, F32PlusHalfMatchesHyprlandNativeShapes)
{
   for (unsigned order = 0; order < 2; ++order) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index dest = agx_register(0, AGX_SIZE_32);
      agx_index f32 = agx_register(0, AGX_SIZE_32);
      agx_index f16 = agx_register(5, AGX_SIZE_16);
      dest.cache = true;
      f32.discard = true;
      f16.cache = true;
      agx_fadd_to(b, dest, order ? f16 : f32, order ? f32 : f16);
      agx_stop(b);
      const std::vector<uint8_t> hyprland = {
         0x09, 0x01, 0x2c, 0x84, 0x10, 0x00, 0x0e, 0x00, 0x00, 0x00,
      };
      const auto code = pack(b);
      EXPECT_EQ(code, hyprland);
   }

   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index f32 = agx_register(2, AGX_SIZE_32);
   agx_index f16 = agx_register(7, AGX_SIZE_16);
   f32.discard = f16.discard = true;
   agx_fadd_to(b, agx_register(2, AGX_SIZE_32), f32, f16);
   agx_stop(b);
   const uint8_t native_odd_no_wait[] = {
      0x19, 0x03, 0x1c, 0x06, 0x10, 0x00,
   };
   const auto code = pack(b);
   ASSERT_EQ(code.size(), sizeof(native_odd_no_wait) + 4);
   EXPECT_EQ(memcmp(code.data(), native_odd_no_wait,
                    sizeof(native_odd_no_wait)), 0);

   /* Exact post-RA shape from Hyprland's colour-management branches. */
   b = builder(AGX_ISA_ENCODING_AGX3B);
   f32 = agx_register(2, AGX_SIZE_32);
   f16 = agx_register(1, AGX_SIZE_16);
   f16.cache = true;
   agx_fadd_to(b, agx_register(2, AGX_SIZE_32), f32, f16);
   agx_stop(b);
   const uint8_t hyprland_ra[] = {
      0x19, 0x03, 0x04, 0x80, 0x10, 0x00,
   };
   const auto hyprland_code = pack(b);
   ASSERT_EQ(hyprland_code.size(), sizeof(hyprland_ra) + 4);
   EXPECT_EQ(memcmp(hyprland_code.data(), hyprland_ra,
                    sizeof(hyprland_ra)), 0);
}

TEST_F(AGX3PackBinary, F32PlusHalfRefusesUngroundedModes)
{
   for (unsigned mode = 0; mode < 9; ++mode) {
      agx_builder *b = builder(mode == 0 ? AGX_ISA_ENCODING_AGX3C
                                         : AGX_ISA_ENCODING_AGX3B);
      agx_index dest = agx_register(0, AGX_SIZE_32);
      agx_index f32 = agx_register(0, AGX_SIZE_32);
      agx_index f16 = agx_register(5, AGX_SIZE_16);
      if (mode == 1)
         f32.neg = true;
      if (mode == 2)
         f16.neg = true;
      if (mode == 3)
         f16.abs = true;
      if (mode == 4)
         f16.cache = f16.discard = true;
      if (mode == 5)
         f16.kill = true;
      if (mode == 6)
         f32.value = 1;
      if (mode == 7)
         dest.size = AGX_SIZE_16;
      agx_instr *add = agx_fadd_to(b, dest, f32, f16);
      add->saturate = mode == 8;
      agx_stop(b);
      /* A modifier on a mixed-width add is Apple's own instruction, not a
       * different one: the slot widths f43/f26 and the modifier bits
       * f47/f48/f30/f31 belong to one encoding point and compose freely.
       * `a - b`, `b - a`, `a + fabs(b)` and `saturate(a + b)` over a
       * float/half pair are 09011c0400c8, 09041c0100c8, 09011c04020000800200
       * and 09011c0401000082. Modes 4..7 are lifetime and alignment, which
       * stay refused. */
      if (mode >= 1 && mode <= 3)
         EXPECT_FALSE(pack(b).empty()) << "mode=" << mode;
      else if (mode == 8)
         EXPECT_FALSE(pack(b).empty()) << "mode=" << mode;
      else
         EXPECT_TRUE(pack(b, false).empty()) << "mode=" << mode;
   }
}

TEST_F(AGX3PackBinary, AbsoluteF32PlusHalfMatchesGlitchShaderShape)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index f32 = agx_register(8, AGX_SIZE_32);
   f32.abs = true;
   f32.discard = true;
   agx_index f16 = agx_register(28, AGX_SIZE_16);
   f16.cache = true;
   agx_fadd_to(b, agx_register(8, AGX_SIZE_32), f32, f16);
   agx_stop(b);
   const auto code = pack(b);
   ASSERT_EQ(code.size(), 14u);

   /* abs and neg on the same mixed-width source used to be refused as
    * unwitnessed. Apple's `b - fabs(a)` is 09041c01020800800200: f47 and f48
    * together, with f43 = 2 and f26 = 1. */
   b = builder(AGX_ISA_ENCODING_AGX3B);
   f32 = agx_register(8, AGX_SIZE_32);
   f32.abs = f32.neg = true;
   agx_fadd_to(b, agx_register(8, AGX_SIZE_32), f32, f16);
   agx_stop(b);
   EXPECT_FALSE(pack(b).empty());
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

TEST_F(AGX3PackBinary, G17PHalfImmediateRetiresOverwriteWait)
{
   const enum agx_isa_encoding enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);

   /* The 32-bit async result owns halves 14 and 15. Loading a constant into
    * its upper half therefore needs a wait, but native LDIMM16 has no wait
    * field. The packer must retire slot 1 before issuing the exact load. */
   agx_get_sr_to(b, agx_register(14, AGX_SIZE_32),
                 AGX_SR_THREAD_POSITION_IN_GRID_X);
   agx_mov_imm_to(b, agx_register(15, AGX_SIZE_16), 0x1972);
   agx_stop(b);

   std::vector<uint8_t> want;
   struct agx3_sr_desc sr = {};
   sr.dest = 14;
   sr.width = AGX3_SR_W32;
   sr.sr = 23;
   sr.slot = 1;
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
   ASSERT_TRUE(agx3_sr_encode(enc, &sr, bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_alu_compose(enc, AGX3_ALU_BAND, 0, 0, 0, 0, 0, 1,
                                bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_alu_mov16(enc, 15, 0x1972, 0, bytes, &len));
   append(want, bytes, len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(enc, &end));
   append(want, end.bytes, end.length);

   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, U16ToF32RejectsRemainingUnwitnessedDomains)
{
   for (unsigned variant = 0; variant < 6; ++variant) {
      SCOPED_TRACE(variant);
      /* S16-to-f32 is now independently witnessed alongside U16-to-f32. */
      if (variant == 4)
         continue;
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
      last.kill = last.discard = true;
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
      ASSERT_TRUE(agx3_alu_fadd_minifloat(enc, 4, 0, 0x20, false,
                                         AGX3_ALU_HINT_NONE, 1, bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_alu_fadd_minifloat(enc, 6, 0, 0x21, false,
                                         AGX3_ALU_HINT_DISCARD, 0, bytes, &len));
      append(want, bytes, len);
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(enc, &end));
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want);
   }
}

TEST_F(AGX3PackBinary, FaddMinifloatRetiresExcessWaitAndKeepsBothCaches)
{
   const enum agx_isa_encoding enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);

   /* Advance the round-robin timeline so the retained destination is an
    * outstanding slot-4 result and the retained source is on slot 5.  The
    * ADD therefore needs both a WAW wait and its source producer wait, the
    * exact 0x18 shape reached by vkQuake's sky_cube fragment after RA. */
   for (unsigned r : {20u, 22u, 24u, 14u, 2u})
      agx_get_sr_to(b, agx_register(r, AGX_SIZE_32),
                    AGX_SR_THREAD_POSITION_IN_GRID_X);

   agx_index dest = agx_register(14, AGX_SIZE_32);
   agx_index source = agx_register(2, AGX_SIZE_32);
   dest.cache = source.cache = true;
   agx_fadd_to(b, dest, source, agx_immediate(0xa0));
   agx_stop(b);

   const auto code = pack(b);
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   std::vector<uint8_t> expected;
   /* Slot 5 is retired first by the byte-grounded x&x identity. Slot 4 stays
    * fused on the ADD, which preserves both source and destination CACHE. */
   ASSERT_TRUE(agx3_alu_compose(enc, AGX3_ALU_BAND, 0, 0, 0, 0, 0,
                                1u << 4, bytes, &len));
   append(expected, bytes, len);
   ASSERT_TRUE(agx3_alu_fadd_minifloat(
      enc, 14, 2, 0xa0, true, AGX3_ALU_HINT_CACHE, 1u << 3, bytes, &len));
   append(expected, bytes, len);

   EXPECT_NE(std::search(code.begin(), code.end(), expected.begin(),
                         expected.end()),
             code.end());
}

TEST_F(AGX3PackBinary, FaddTwoMinifloatsFoldsToLdimm)
{
   for (bool swapped : {false, true}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_fadd_to(b, agx_register(2, AGX_SIZE_32),
                  agx_immediate(swapped ? 0x80 : 0x30),
                  agx_immediate(swapped ? 0x30 : 0x80));
      agx_stop(b);
      struct agx3_encoded_ldimm load;
      ASSERT_TRUE(agx3_pack_ldimm_u32(AGX_ISA_ENCODING_AGX3B, 2,
                                      0x3f800000, &load));
      std::vector<uint8_t> expected(load.bytes,
                                    load.bytes + load.length);
      const uint8_t stop[] = {0x0e, 0x00, 0x00, 0x00};
      append(expected, stop, sizeof(stop));
      EXPECT_EQ(pack(b), expected);
   }
}

TEST_F(AGX3PackBinary, FaddMinifloatRefusesRemainingUngroundedModes)
{
   for (unsigned variant = 0; variant < 10; ++variant) {
      SCOPED_TRACE(variant);
      /* Negation, absolute, two-minifloat folding and saturation all have
       * exact, independently tested G17P paths. */
      if (variant == 4 || variant == 5 || variant == 7 || variant == 9)
         continue;
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

TEST_F(AGX3PackBinary, FaddAbsoluteMinifloatMatchesNative)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index src = agx_abs(agx_register(0, AGX_SIZE_32));
   src.discard = true;
   agx_fadd_to(b, agx_register(0, AGX_SIZE_32), src,
               agx_immediate(0x20));
   agx_stop(b);

   std::vector<uint8_t> want = {
      0x09, 0xa1, 0x14, 0x01, 0x82, 0x00, 0x00, 0x00, 0x02, 0x00,
   };
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
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

/*
 * 32x32 -> 64. Both halves of the product are written, so this is the shape
 * `umul_high`/`imul_high` and NIR's `*mul_2x32_64` all reach the backend as.
 * The AGX2 source convention carries over: `abs` on an operand narrower than
 * the destination means zero-extend, its absence means sign-extend, which is
 * what the per-slot f28/f43 bits encode.
 */
TEST_F(AGX3PackBinary, G17PNarrowImmediateMADMatchesNative)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index source = agx_register(3, AGX_SIZE_16);
   source.discard = true;
   agx_imad_to(b, agx_register(3, AGX_SIZE_16), source,
               agx_immediate(11), agx_immediate(9), 0);
   agx_stop(b);

   static const uint8_t native[] = {
      0x9f,0x00,0x54,0x03,0x02,0x06,0x2c,0x48,0x00,0x22,0x02,0x00,
   };
   std::vector<uint8_t> want(std::begin(native), std::end(native));
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);

   /* Alpha-to-coverage lowers its sample-index calculation to this exact
    * narrow register multiply.  It must reach the grounded imadd form rather
    * than fail closing the separately compiled fragment epilog. */
   agx_builder *mul = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index dst = agx_register(0, AGX_SIZE_16);
   agx_index a = agx_register(0, AGX_SIZE_16);
   agx_index m = agx_register(4, AGX_SIZE_16);
   dst.cache = true;
   a.discard = true;
   m.discard = true;
   agx_imad_to(mul, dst, a, m, agx_zero(), 0);
   agx_stop(mul);

   static const uint8_t native_mul[] = {
      /* Apple witness with its independent slot-1 wait cleared. */
      0x9f,0x00,0x54,0x00,0x03,0x00,0x10,0x00,0x00,0x26,0x0a,0x00,
   };
   std::vector<uint8_t> want_mul(std::begin(native_mul),
                                 std::end(native_mul));
   append(want_mul, end.bytes, end.length);
   EXPECT_EQ(pack(mul), want_mul);
}

TEST_F(AGX3PackBinary, FirefoxNarrowRegisterMADMatchesGroundedCore)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index dest = agx_register(24, AGX_SIZE_16);
   agx_index a = agx_register(32, AGX_SIZE_16);
   agx_index m = agx_register(25, AGX_SIZE_16);
   agx_index c = agx_register(24, AGX_SIZE_16);
   agx_imad_to(b, dest, a, m, c, 0);
   agx_stop(b);

   const struct agx3_imad_desc d = {
      .dest = 24, .dest_16 = true,
      .src_a = 32, .src_a_16 = true,
      .multiplier_is_register = true, .multiplier_reg = 25,
      .multiplier_16 = true,
      .src_c = 24, .src_c_16 = true,
   };
   uint8_t bytes[AGX3_IADD_MAX_LENGTH], len = 0;
   ASSERT_TRUE(agx3_imad_encode(AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
   std::vector<uint8_t> want(bytes, bytes + len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);

   /* Only the plain half addend has a native witness for the narrow result. */
   for (unsigned variant = 0; variant < 3; ++variant) {
      agx_builder *bad = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index bad_c = agx_register(24, variant == 2 ? AGX_SIZE_32
                                                      : AGX_SIZE_16);
      bad_c.abs = variant == 0;
      bad_c.neg = variant == 1;
      agx_imad_to(bad, dest, a, m, bad_c, 0);
      agx_stop(bad);
      EXPECT_TRUE(pack(bad, false).empty()) << variant;
   }
}

TEST_F(AGX3PackBinary, FirefoxWideningHalfMultiplyMatchesNative)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index dest = agx_register(34, AGX_SIZE_32);
   agx_index a = agx_register(25, AGX_SIZE_16);
   agx_index m = agx_register(33, AGX_SIZE_16);
   agx_imad_to(b, dest, a, m, agx_zero(), 0);
   agx_stop(b);

   struct agx3_imad_desc d = {};
   d.dest = 34;
   d.src_a = 25;
   d.src_a_16 = true;
   d.src_a_sign_extend = true;
   d.multiplier_is_register = true;
   d.multiplier_reg = 33;
   d.multiplier_16 = true;
   d.multiplier_sign_extend = true;
   d.no_addend = true;
   uint8_t bytes[AGX3_IADD_MAX_LENGTH], len = 0;
   ASSERT_TRUE(agx3_imad_encode(AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
   std::vector<uint8_t> want(bytes, bytes + len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);

   for (unsigned variant = 0; variant < 5; ++variant) {
      agx_builder *bad = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index bad_a = variant == 0 ? agx_register(24, AGX_SIZE_32) : a;
      agx_index bad_m = m;
      if (variant == 4)
         bad_m.abs = true;
      agx_index bad_c = variant == 1 ? agx_immediate(1) : agx_zero();
      agx_instr *I = agx_imad_to(bad, dest, bad_a, bad_m, bad_c,
                                 variant == 2 ? 1 : 0);
      I->saturate = variant == 3;
      agx_stop(bad);
      EXPECT_TRUE(pack(bad, false).empty()) << variant;
   }
}

TEST_F(AGX3PackBinary, G17PNarrowShiftSevenMatchesNativePair)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index dest = agx_register(0, AGX_SIZE_16);
   agx_index value = agx_register(0, AGX_SIZE_16);
   dest.cache = true;
   value.discard = true;
   agx_bfi_to(b, dest, agx_immediate(0), value, agx_immediate(7), 0);
   agx_stop(b);

   /* Apple's first instruction also waits on its input load. The builder has
    * no async producer, so only that independently grounded wait bit clears. */
   static const uint8_t native[] = {
      0x9f,0x01,0x54,0x00,0x03,0x00,0x00,0x40,0x14,0x14,
      0x9f,0x01,0x54,0x00,0x03,0x00,0x00,0x00,0x14,0x04,
   };
   std::vector<uint8_t> want(std::begin(native), std::end(native));
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);

   agx_builder *bad = builder(AGX_ISA_ENCODING_AGX3B);
   agx_bfi_to(bad, agx_register(0, AGX_SIZE_16), agx_immediate(0),
              agx_register(2, AGX_SIZE_16), agx_immediate(9), 0);
   agx_stop(bad);
   pack(bad, /*expect_ok=*/false);
}

TEST_F(AGX3PackBinary, G17PNarrowDynamicShiftsMatchNativeFields)
{
   const enum agx_isa_encoding enc = AGX_ISA_ENCODING_AGX3B;
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(enc, &end));

   agx_builder *left = builder(enc);
   agx_index left_dest = agx_register(3, AGX_SIZE_16);
   agx_index left_value = agx_register(2, AGX_SIZE_16);
   agx_index left_shift = agx_register(12, AGX_SIZE_32);
   left_dest.cache = true;
   left_value.cache = true;
   left_shift.discard = true;
   agx_bfi_to(left, left_dest, agx_immediate(0), left_value, left_shift, 0);
   agx_stop(left);
   static const uint8_t native_left[] = {
      0x27,0x00,0x54,0x03,0x03,0x04,0x32,0x00,0xc0,0x12,0x05,0x00,
   };
   std::vector<uint8_t> want_left(std::begin(native_left),
                                  std::end(native_left));
   append(want_left, end.bytes, end.length);
   EXPECT_EQ(pack(left), want_left);

   agx_builder *right = builder(enc);
   agx_index right_dest = agx_register(6, AGX_SIZE_16);
   agx_index right_value = agx_register(2, AGX_SIZE_16);
   agx_index right_shift = agx_register(4, AGX_SIZE_32);
   right_dest.cache = true;
   right_value.discard = true;
   right_shift.discard = true;
   agx_bfeil_to(right, right_dest, agx_immediate(0), right_value,
                right_shift, 0);
   agx_stop(right);
   static const uint8_t native_right[] = {
      0xa7,0x00,0x54,0x06,0x03,0x04,0x10,0x00,0xc0,0x13,0x05,0x00,
   };
   std::vector<uint8_t> want_right(std::begin(native_right),
                                   std::end(native_right));
   append(want_right, end.bytes, end.length);
   EXPECT_EQ(pack(right), want_right);
}

TEST_F(AGX3PackBinary, G17PRegisterMADImmediateAddendUsesNativeEndpoint)
{
   const enum agx_isa_encoding enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   agx_index a = agx_register(4, AGX_SIZE_32);
   agx_index m = agx_register(0, AGX_SIZE_32);
   a.discard = true;
   m.discard = true;
   agx_imad_to(b, agx_register(0, AGX_SIZE_32), a, m,
               agx_immediate(1), 0);
   agx_stop(b);

   static const uint8_t native[] = {
      0x9f,0x00,0x54,0x00,0x02,0x08,0x00,0x08,0xd0,0x26,0x0a,0x00,
   };
   std::vector<uint8_t> want(std::begin(native), std::end(native));
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(enc, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, WideSignedMultiplyMatchesNativeWitness)
{
   const enum agx_isa_encoding enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   agx_index a = agx_register(0, AGX_SIZE_32);
   agx_index m = agx_register(2, AGX_SIZE_32);
   a.discard = true;
   m.discard = true;
   agx_imad_to(b, agx_register(0, AGX_SIZE_64), a, m, agx_zero(), 0);
   agx_stop(b);

   std::vector<uint8_t> want = {
      0x9f, 0x00, 0x54, 0x00, 0x02, 0x00, 0x08, 0x00,
      0xe0, 0x26, 0x1e, 0x00,
      0x0e, 0x00, 0x00, 0x00,
   };
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, WideUnsignedMultiplyZeroExtendsBothOperands)
{
   const enum agx_isa_encoding enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   agx_index a = agx_register(0, AGX_SIZE_32);
   agx_index m = agx_register(2, AGX_SIZE_32);
   a.discard = true;
   m.discard = true;
   agx_imad_to(b, agx_register(0, AGX_SIZE_64), agx_abs(a), agx_abs(m),
               agx_zero(), 0);
   agx_stop(b);

   std::vector<uint8_t> want = {
      0x9f, 0x00, 0x54, 0x00, 0x02, 0x00, 0x08, 0x00,
      0xe0, 0x26, 0x0a, 0x00,
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

TEST_F(AGX3PackBinary, HyprlandHalfMinifloatFselShapes)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index half = agx_register(6, AGX_SIZE_16);
   agx_fcmpsel_to(b, agx_register(8, AGX_SIZE_32), agx_immediate(32), half,
                  agx_register(8, AGX_SIZE_32),
                  agx_register(16, AGX_SIZE_32), AGX_FCOND_GE);
   agx_stop(b);
   const std::vector<uint8_t> half_threshold = {
      0x42, 0x06, 0x07, 0xa1, 0x06, 0x08, 0x03, 0x02, 0x00,
      0x10, 0x0e, 0x00, 0x00, 0x00,
   };
   EXPECT_EQ(pack(b), half_threshold);

   b = builder(AGX_ISA_ENCODING_AGX3B);
   half = agx_register(6, AGX_SIZE_16);
   half.cache = true;
   agx_index selected = agx_register(8, AGX_SIZE_32);
   selected.cache = true;
   agx_fcmpsel_to(b, agx_register(20, AGX_SIZE_32), half,
                  agx_immediate(0), selected, agx_immediate(0),
                  AGX_FCOND_LT);
   agx_stop(b);
   const std::vector<uint8_t> register_or_zero = {
      0xa2, 0x86, 0x03, 0x81, 0x02, 0x88, 0x03, 0x02, 0x20,
      0x80, 0x0e, 0x00, 0x00, 0x00,
   };
   EXPECT_EQ(pack(b), register_or_zero);
}

TEST_F(AGX3PackBinary, HyprlandHalfMinifloatFselRefusesUngroundedModes)
{
   for (unsigned mode = 0; mode < 10; ++mode) {
      agx_builder *b = builder(mode == 0 ? AGX_ISA_ENCODING_AGX3C
                                         : AGX_ISA_ENCODING_AGX3B);
      agx_index dest = agx_register(8, AGX_SIZE_32);
      agx_index half = agx_register(6, AGX_SIZE_16);
      agx_index immediate = agx_immediate(32);
      agx_index if_true = agx_register(8, AGX_SIZE_32);
      agx_index if_false = agx_register(16, AGX_SIZE_32);
      if (mode == 1)
         dest.size = AGX_SIZE_16;
      if (mode == 2)
         half.abs = true;
      if (mode == 3)
         half.neg = true;
      if (mode == 4)
         half.cache = half.discard = true;
      if (mode == 5)
         half.kill = true;
      if (mode == 6)
         immediate.value = 256;
      if (mode == 7)
         if_true.value = 9;
      if (mode == 8)
         if_false = agx_immediate(0);
      agx_instr *fsel = agx_fcmpsel_to(b, dest, immediate, half,
                                       mode == 8 ? agx_immediate(0) : if_true,
                                       if_false, AGX_FCOND_GE);
      fsel->saturate = mode == 9;
      agx_stop(b);
      /* The result width (mode 1) and an immediate selected value (mode 8)
       * are independent fields of this endpoint, not separate instructions:
       * Apple's own `select(v.x, v.y, a >= 0.5f)` is 12050da1850302c2, whose
       * f51 carries the minifloat and whose f11 carries the half result. */
      if (mode == 1 || mode == 8)
         EXPECT_FALSE(pack(b).empty()) << "mode=" << mode;
      else
         EXPECT_TRUE(pack(b, false).empty()) << "mode=" << mode;
   }
}

TEST_F(AGX3PackBinary, HyprlandAbsoluteSecondFsel)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index plain = agx_register(26, AGX_SIZE_32);
   agx_index absolute = agx_register(8, AGX_SIZE_32);
   plain.cache = absolute.cache = true;
   absolute.abs = true;
   agx_fcmpsel_to(b, agx_register(16, AGX_SIZE_32), plain, absolute,
                  agx_register(22, AGX_SIZE_32),
                  agx_register(16, AGX_SIZE_32), AGX_FCOND_GE);
   agx_stop(b);
   const std::vector<uint8_t> target = {
      0x82, 0x9b, 0x07, 0x89, 0x07, 0x16, 0x02, 0x00, 0x00,
      0x10, 0x00, 0x00, 0x02, 0x00, 0x0e, 0x00, 0x00, 0x00,
   };
   EXPECT_EQ(pack(b), target);
}

TEST_F(AGX3PackBinary, GtkConvertCicpAbsoluteMinifloatFsel)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index absolute = agx_register(2, AGX_SIZE_32);
   absolute.abs = true;
   agx_fcmpsel_to(b, agx_register(0, AGX_SIZE_32), agx_immediate(32),
                  absolute, agx_register(4, AGX_SIZE_32),
                  agx_register(6, AGX_SIZE_32), AGX_FCOND_GE);
   agx_stop(b);

   /* Exact current-toolchain applegpu_g17p bytes for the no-wait kernel
    * `0.5 >= fabs(x) ? y : z`, plus STOP. */
   const std::vector<uint8_t> native = {
      0x02, 0x03, 0x07, 0xa1, 0x07, 0x04, 0x03, 0x02,
      0x00, 0x06, 0x00, 0x00, 0x01, 0x00,
      0x0e, 0x00, 0x00, 0x00,
   };
   EXPECT_EQ(pack(b), native);
}

TEST_F(AGX3PackBinary, GtkConvertCicpAbsoluteMinifloatRefusesModifiers)
{
   for (unsigned mode = 0; mode < 8; ++mode) {
      agx_builder *b = builder(mode == 0 ? AGX_ISA_ENCODING_AGX3C
                                         : AGX_ISA_ENCODING_AGX3B);
      agx_index dest = agx_register(0, mode == 1 ? AGX_SIZE_16
                                                 : AGX_SIZE_32);
      agx_index threshold = agx_immediate(32);
      agx_index compare = agx_register(2, mode == 2 ? AGX_SIZE_16
                                                    : AGX_SIZE_32);
      agx_index if_true = agx_register(4, AGX_SIZE_32);
      agx_index if_false = agx_register(mode == 3 ? 7 : 6, AGX_SIZE_32);
      compare.abs = true;
      if (mode == 4)
         compare.neg = true;
      if (mode == 5)
         threshold.abs = true;
      if (mode == 6)
         if_true.abs = true;
      if (mode == 7)
         compare.cache = compare.discard = true;
      agx_fcmpsel_to(b, dest, threshold, compare, if_true, if_false,
                     AGX_FCOND_GE);
      /* The result width is its own field (f11, or f25 on the shifted
       * destination block) and composes with an absolute comparator; the
       * ledger already carried the pair. */
      if (mode == 1)
         EXPECT_FALSE(pack(b).empty()) << "mode=" << mode;
      else
         EXPECT_TRUE(pack(b, false).empty()) << "mode=" << mode;
   }
}

TEST_F(AGX3PackBinary, OrderedAbsoluteFselRefusesUnwitnessedDoubleAbs)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index a = agx_register(2, AGX_SIZE_32);
   agx_index c = agx_register(4, AGX_SIZE_32);
   a.abs = c.abs = true;
   agx_fcmpsel_to(b, agx_register(0, AGX_SIZE_32), a, c,
                  agx_register(6, AGX_SIZE_32),
                  agx_register(8, AGX_SIZE_32), AGX_FCOND_GE);
   agx_stop(b);
   EXPECT_TRUE(pack(b, false).empty());
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
   const bool compare_neg[] = {false, false};
   const bool immediate[] = {false, true};
   const enum agx3_alu_hint hints[] = {
      AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_CACHE,
      AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_NONE};
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   ASSERT_TRUE(agx3_alu_fcmp_select_reg_imm(
      AGX_ISA_ENCODING_AGX3B, 26, src, false, compare_abs, compare_neg,
      immediate, hints, AGX_FCOND_EQ, false, true, 0, bytes, &len));
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

TEST_F(AGX3PackBinary, FmulAbsoluteMinifloatMatchesQuickshell)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index src = agx_abs(agx_register(2, AGX_SIZE_32));
   src.cache = true;
   agx_fmul_to(b, agx_register(0, AGX_SIZE_32), src,
               agx_immediate(0x40));
   agx_stop(b);

   std::vector<uint8_t> want = {
      0x09, 0xc1, 0x05, 0x83, 0x82, 0x00,
   };
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, FmulMinifloatRefusesUngroundedModes)
{
   for (unsigned variant = 0; variant < 10; ++variant) {
      /* Absolute and scalar-half sources now have exact native witnesses.
       * So does variant 1, a 32-bit source narrowed into a half destination:
       * f10 selects the destination width independently of f26, and Apple
       * emits the mixed-width multiply directly. See
       * FmulMinifloatNarrowsIntoHalfDestination below. */
      if (variant == 1 || variant == 5 || variant == 8)
         continue;
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

/*
 * `packHalf2x16(vec2(x * k, ...))`: agx_optimizer_fmov_rev() folds the f2f16
 * into the multiply, so the packer sees a 32-bit source, a minifloat operand
 * and a HALF destination. That form is Apple's own -- f10 is the destination
 * width -- and used to fail the whole shader closed.
 */
TEST_F(AGX3PackBinary, FmulMinifloatNarrowsIntoHalfDestination)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index src = agx_register(2, AGX_SIZE_32);
   src.discard = true;
   agx_fmul_to(b, agx_register(2, AGX_SIZE_16), src, agx_immediate(0x20));
   agx_stop(b);
   const auto code = pack(b);
   /* Apple's witness is `11 a1 15 03 80 c0`; a bare test has no producer to
    * wait on, so only the scoreboard field (f85) differs. The literal Apple
    * bytes are asserted at encoder level in test-agx3-alu-core.c. */
   const uint8_t native[] = {0x11, 0xa1, 0x15, 0x03, 0x80, 0x00};
   ASSERT_EQ(code.size(), sizeof(native) + 4);
   EXPECT_EQ(memcmp(code.data(), native, sizeof(native)), 0);

   /* The 32-bit destination keeps the bytes it always had. */
   b = builder(AGX_ISA_ENCODING_AGX3B);
   src = agx_register(2, AGX_SIZE_32);
   src.discard = true;
   agx_fmul_to(b, agx_register(2, AGX_SIZE_32), src, agx_immediate(0x20));
   agx_stop(b);
   const auto wide = pack(b);
   const uint8_t native32[] = {0x19, 0xa1, 0x15, 0x03, 0x80, 0x00};
   ASSERT_EQ(wide.size(), sizeof(native32) + 4);
   EXPECT_EQ(memcmp(wide.data(), native32, sizeof(native32)), 0);
}

/*
 * `sqrt(abs(x))` and `inversesqrt(abs(x))`. A bare .abs on either member of
 * this family refused the shader; f30 carries it, witnessed one field at a
 * time against Apple.
 */
TEST_F(AGX3PackBinary, RsqrtFamilyCarriesAbsoluteSource)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index src = agx_register(0, AGX_SIZE_32);
   src.abs = true;
   src.discard = true;
   agx_rsqrt_to(b, agx_register(0, AGX_SIZE_32), src);
   agx_stop(b);
   const auto rsq = pack(b);
   /* Apple: `af 01 56 00 02 00 b0 c0 00 00`; byte 2 drops to 0x54 here
    * because there is no scoreboard slot to wait on. */
   const uint8_t rsq_abs[] = {0xaf, 0x01, 0x54, 0x00, 0x02,
                              0x00, 0xb0, 0xc0, 0x00, 0x00};
   ASSERT_EQ(rsq.size(), sizeof(rsq_abs) + 4);
   EXPECT_EQ(memcmp(rsq.data(), rsq_abs, sizeof(rsq_abs)), 0);

   /* A negated source is separately witnessed on the same form. */
   b = builder(AGX_ISA_ENCODING_AGX3B);
   src = agx_register(0, AGX_SIZE_32);
   src.neg = true;
   src.discard = true;
   agx_rsqrt_to(b, agx_register(0, AGX_SIZE_32), src);
   agx_stop(b);
   const auto rsq_n = pack(b);
   const uint8_t rsq_neg[] = {0xaf, 0x01, 0x54, 0x00, 0x02,
                              0x00, 0xb0, 0x40, 0x01, 0x00};
   ASSERT_EQ(rsq_n.size(), sizeof(rsq_neg) + 4);
   EXPECT_EQ(memcmp(rsq_n.data(), rsq_neg, sizeof(rsq_neg)), 0);

   /* The half form has no modifier witness and stays closed. */
   b = builder(AGX_ISA_ENCODING_AGX3B);
   src = agx_register(2, AGX_SIZE_16);
   src.abs = true;
   agx_rsqrt_to(b, agx_register(2, AGX_SIZE_16), src);
   agx_stop(b);
   EXPECT_TRUE(pack(b, false).empty());
}

/*
 * `int16_t(x) + k`. A bare 16-bit integer source is sign-extended -- that is
 * what agx_optimizer_signext() means by dropping the SIGNEXT and asserting
 * the survivor carries no ABS -- and the register+immediate form refused it
 * while the register+register form beside it already encoded.
 */
TEST_F(AGX3PackBinary, IaddImmediateSignExtendsBareHalfSource)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index src = agx_register(0, AGX_SIZE_16);
   src.discard = true;
   agx_iadd_to(b, agx_register(0, AGX_SIZE_32), src, agx_immediate(7), 0);
   agx_stop(b);
   const auto code = pack(b);
   /* Apple: `o[i] = int(short a[i]) + 7`, f15 = 7 and f43 = 1. */
   const uint8_t native[] = {0x9f, 0x01, 0x54, 0x00, 0x02,
                             0x0e, 0x00, 0x08, 0x15, 0x0c};
   ASSERT_EQ(code.size(), sizeof(native) + 4);
   EXPECT_EQ(memcmp(code.data(), native, sizeof(native)), 0);

   /* An ABS source is the zero-extending form, and keeps its own bytes:
    * f43 clear, exactly as Apple's `uint(ushort a[i]) + 7`. */
   b = builder(AGX_ISA_ENCODING_AGX3B);
   src = agx_register(0, AGX_SIZE_16);
   src.abs = true;
   src.discard = true;
   agx_iadd_to(b, agx_register(0, AGX_SIZE_32), src, agx_immediate(7), 0);
   agx_stop(b);
   const auto zext = pack(b);
   const uint8_t native_u[] = {0x9f, 0x01, 0x54, 0x00, 0x02,
                               0x0e, 0x00, 0x08, 0x15, 0x04};
   ASSERT_EQ(zext.size(), sizeof(native_u) + 4);
   EXPECT_EQ(memcmp(zext.data(), native_u, sizeof(native_u)), 0);
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

/* The lane is the whole seven-bit f29, so every lane a subgroup can name
 * encodes -- lane 31, which ends every subgroup scan, included. Only a lane
 * the field cannot hold is refused. */
TEST_F(AGX3PackBinary, SimdBroadcastEncodesEveryLane)
{
   for (unsigned lane : {1u, 5u, 31u, 63u, 127u}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_shuffle_to(b, agx_register(4, AGX_SIZE_32),
                     agx_register(0, AGX_SIZE_32), agx_immediate(lane));
      agx_stop(b);

      uint8_t bytes[AGX3_SUBGROUP_MAX_LENGTH], length = 0;
      const struct agx3_subgroup_shuffle_desc desc = {
         .op = AGX3_SUBGROUP_SHUFFLE_BROADCAST,
         .lane_kind = AGX3_SUBGROUP_LANE_IMMEDIATE,
         .dest = 4,
         .src = 0,
         .lane = (uint16_t)lane,
      };
      ASSERT_TRUE(
         agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &desc, bytes,
                                      &length));
      std::vector<uint8_t> want;
      append(want, bytes, length);
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want);
   }

   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_shuffle_to(b, agx_register(4, AGX_SIZE_32),
                  agx_register(0, AGX_SIZE_32), agx_immediate(128));
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

TEST_F(AGX3PackBinary, G17PScalarByteLoadRequiresAlignedHalf)
{
   agx_builder *good = builder(AGX_ISA_ENCODING_AGX3B);
   agx_device_load_to(good, agx_register(2, AGX_SIZE_32),
                      agx_uniform(4, AGX_SIZE_64),
                      agx_register(0, AGX_SIZE_32), AGX_FORMAT_I8, 1, 0,
                      false);
   agx_stop(good);
   EXPECT_FALSE(pack(good).empty());

   /* No Apple scalar-byte witness writes an odd half. The frontend models the
    * physical result as a 32-bit temporary so production RA cannot create
    * this shape; retain the packer refusal as the negative control. */
   agx_builder *bad = builder(AGX_ISA_ENCODING_AGX3B);
   agx_device_load_to(bad, agx_register(1, AGX_SIZE_16),
                      agx_uniform(4, AGX_SIZE_64),
                      agx_register(0, AGX_SIZE_32), AGX_FORMAT_I8, 1, 0,
                      false);
   agx_stop(bad);
   pack(bad, /*expect_ok=*/false);
}

TEST_F(AGX3PackBinary, G17PHalfFMAImmediateAddend)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index src0 = agx_register(2, AGX_SIZE_16);
   agx_index src1 = agx_register(3, AGX_SIZE_16);
   src0.discard = true;
   src1.discard = true;
   agx_hfma_to(b, agx_register(2, AGX_SIZE_16), src0, src1,
               agx_immediate(56));
   agx_stop(b);

   uint8_t fma[AGX3_ALU_MAX_LENGTH], fma_len;
   const uint16_t src[] = {2, 3};
   const enum agx3_alu_hint hints[] = {
      AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD,
   };
   ASSERT_TRUE(agx3_alu_hfma_addend_minifloat(
      AGX_ISA_ENCODING_AGX3B, 2, src, 56, hints, false, 0, fma, &fma_len));
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   std::vector<uint8_t> want;
   append(want, fma, fma_len);
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);

   agx_builder *bad = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index absolute = agx_register(2, AGX_SIZE_16);
   absolute.abs = true;
   agx_hfma_to(bad, agx_register(0, AGX_SIZE_16), absolute,
               agx_register(3, AGX_SIZE_16), agx_immediate(56));
   agx_stop(bad);
   pack(bad, /*expect_ok=*/false);
}

TEST_F(AGX3PackBinary, G17PHalfSaturateMatchesNative)
{
   static const uint8_t native[] = {
      0x00, 0x00, 0x1c, 0x80, 0x01, 0x02, 0x00, 0x02,
   };
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index src = agx_register(0, AGX_SIZE_16);
   src.discard = true;
   agx_instr *I = agx_hadd_to(b, agx_register(0, AGX_SIZE_16), src,
                              agx_immediate(0x80));
   I->saturate = true;
   agx_stop(b);

   std::vector<uint8_t> want(std::begin(native), std::end(native));
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, G17PHalfSaturateRefusesUnwitnessedOperands)
{
   for (unsigned mode = 0; mode < 8; ++mode) {
      agx_builder *b = builder(mode == 0 ? AGX_ISA_ENCODING_AGX3C
                                         : AGX_ISA_ENCODING_AGX3B);
      agx_index dst = agx_register(0, AGX_SIZE_16);
      agx_index src = agx_register(0, AGX_SIZE_16);
      agx_index imm = agx_immediate(mode == 1 ? 0x30 : 0x80);
      if (mode == 2)
         src.size = AGX_SIZE_32;
      if (mode == 3)
         src.channels_m1 = 1;
      if (mode == 4)
         src.abs = true;
      if (mode == 5)
         src.neg = true;
      if (mode == 6)
         src.kill = true;
      if (mode == 7)
         dst.channels_m1 = 1;
      agx_instr *I = agx_hadd_to(b, dst, src, imm);
      I->saturate = mode != 1;
      agx_stop(b);
      EXPECT_TRUE(pack(b, false).empty()) << "mode=" << mode;
   }
}

TEST_F(AGX3PackBinary, G17PHalfInverseMatchesNativeForm)
{
   static const uint8_t native[] = {
      0x08, 0xb0, 0x04, 0x80, 0x80, 0x08,
   };
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index src = agx_register(0, AGX_SIZE_16);
   src.neg = true;
   src.cache = true;
   agx_hadd_to(b, agx_register(1, AGX_SIZE_16), src, agx_immediate(0x30));
   agx_stop(b);

   std::vector<uint8_t> want(std::begin(native), std::end(native));
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, G17PHalfInverseRefusesUnwitnessedOperands)
{
   for (unsigned mode = 0; mode < 9; ++mode) {
      agx_builder *b = builder(mode == 0 ? AGX_ISA_ENCODING_AGX3C
                                         : AGX_ISA_ENCODING_AGX3B);
      agx_index dst = agx_register(1, AGX_SIZE_16);
      agx_index src = agx_register(0, AGX_SIZE_16);
      agx_index imm = agx_immediate(mode == 1 ? 0x80 : 0x30);
      src.neg = mode != 2;
      src.abs = mode == 4;
      src.kill = mode == 5;
      if (mode == 6)
         src.size = AGX_SIZE_32;
      if (mode == 7)
         src.channels_m1 = 1;
      if (mode == 8)
         dst.channels_m1 = 1;
      agx_instr *I = agx_hadd_to(b, dst, src, imm);
      I->saturate = mode == 3;
      agx_stop(b);
      EXPECT_TRUE(pack(b, false).empty()) << "mode=" << mode;
   }
}

TEST_F(AGX3PackBinary, G17PI32ToHalfMatchesNative)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index dest = agx_register(0, AGX_SIZE_16);
   dest.cache = true;
   agx_convert_to(b, dest, agx_immediate(AGX_CONVERT_U32_TO_F),
                  agx_register(2, AGX_SIZE_32), AGX_ROUND_RTE);
   agx_stop(b);

   /* The Apple witness carried a wait on its async source. This synthetic
    * builder has no producer, so only that dependency bit clears. */
   static const uint8_t native[] = {
      0xa7, 0x07, 0x54, 0x00, 0x03, 0x04, 0x88, 0x20,
   };
   std::vector<uint8_t> want(std::begin(native), std::end(native));
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);

   agx_builder *signed_dead = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index source = agx_register(0, AGX_SIZE_32);
   source.discard = true;
   agx_convert_to(signed_dead, agx_register(0, AGX_SIZE_16),
                  agx_immediate(AGX_CONVERT_S32_TO_F), source,
                  AGX_ROUND_RTE);
   agx_stop(signed_dead);
   static const uint8_t native_signed_dead[] = {
      0xa7, 0x07, 0x54, 0x00, 0x02, 0x00, 0xa8, 0x60,
   };
   std::vector<uint8_t> want_signed(std::begin(native_signed_dead),
                                    std::end(native_signed_dead));
   append(want_signed, end.bytes, end.length);
   EXPECT_EQ(pack(signed_dead), want_signed);

   /* A CACHED source used to be refused here.  f27 carries it, exactly as f29
    * carries discard, and test-agx3-alu-core.c holds the byte-exact Apple
    * vector for both.  What this level has to show is that the packer reaches
    * that encoder and that the cache hint is the ONLY thing it changes, so
    * pack the same convert twice and diff. */
   agx_builder *plain = builder(AGX_ISA_ENCODING_AGX3B);
   agx_convert_to(plain, agx_register(0, AGX_SIZE_16),
                  agx_immediate(AGX_CONVERT_U32_TO_F),
                  agx_register(2, AGX_SIZE_32), AGX_ROUND_RTE);
   agx_stop(plain);
   const std::vector<uint8_t> plain_bytes = pack(plain);

   agx_builder *cached_src = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index cached = agx_register(2, AGX_SIZE_32);
   cached.cache = true;
   agx_convert_to(cached_src, agx_register(0, AGX_SIZE_16),
                  agx_immediate(AGX_CONVERT_U32_TO_F), cached,
                  AGX_ROUND_RTE);
   agx_stop(cached_src);
   const std::vector<uint8_t> cached_bytes = pack(cached_src);

   ASSERT_EQ(plain_bytes.size(), cached_bytes.size());
   unsigned differing_bytes = 0, differing_bits = 0;
   for (size_t i = 0; i < plain_bytes.size(); ++i) {
      const uint8_t delta = plain_bytes[i] ^ cached_bytes[i];
      if (!delta)
         continue;
      differing_bytes++;
      differing_bits += __builtin_popcount(delta);
   }
   EXPECT_EQ(differing_bytes, 1u);
   EXPECT_EQ(differing_bits, 1u);
   EXPECT_NE(plain_bytes, cached_bytes);
}

TEST_F(AGX3PackBinary, GTKHalfToInt32UsesNativeWidthAndPostRAState)
{
   const enum agx_isa_encoding enc = AGX_ISA_ENCODING_AGX3B;

   /* Exact first refusal from GTK 4's current gskgputurbulence.frag after
    * register allocation: `$r17 = convert #F_TO_S32, r17l, RTZ`. */
   agx_builder *gtk = builder(enc);
   agx_index dest = agx_register(34, AGX_SIZE_32);
   dest.cache = true;
   agx_convert_to(gtk, dest, agx_immediate(AGX_CONVERT_F_TO_S32),
                  agx_register(34, AGX_SIZE_16), AGX_ROUND_RTZ);
   agx_stop(gtk);
   const uint8_t gtk_native[] = {
      0x27, 0x07, 0x54, 0x22, 0x03, 0x44, 0x8c,
      0x48, 0x03, 0x00, 0x0e, 0x00, 0x00, 0x00,
   };
   EXPECT_EQ(pack(gtk), std::vector<uint8_t>(std::begin(gtk_native),
                                              std::end(gtk_native)));

   /* A half selected from an async system-register result remains a half;
    * its native conversion consumes that producer's slot in-place. */
   agx_builder *dependent = builder(enc);
   agx_get_sr_to(dependent, agx_register(0, AGX_SIZE_32),
                 AGX_SR_THREAD_POSITION_IN_GRID_X);
   agx_index half = agx_register(0, AGX_SIZE_16);
   half.discard = true;
   agx_convert_to(dependent, agx_register(4, AGX_SIZE_32),
                  agx_immediate(AGX_CONVERT_F_TO_U32), half,
                  AGX_ROUND_RTZ);
   agx_stop(dependent);

   std::vector<uint8_t> want;
   struct agx3_sr_desc sr = {};
   sr.dest = 0;
   sr.width = AGX3_SR_W32;
   sr.sr = 23;
   sr.slot = 1;
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
   ASSERT_TRUE(agx3_sr_encode(enc, &sr, bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_alu_f16_to_int32(enc, 4, 0, false,
                                      AGX3_ALU_HINT_DISCARD, false, 1, bytes,
                                      &len));
   append(want, bytes, len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(enc, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(dependent), want);
}

TEST_F(AGX3PackBinary, G17PHalfToInt32RejectsUngroundedShapes)
{
   for (unsigned mode = 0; mode < 22; ++mode) {
      SCOPED_TRACE(mode);
      agx_builder *b = builder(mode == 0 ? AGX_ISA_ENCODING_AGX3C
                                         : AGX_ISA_ENCODING_AGX3B);
      agx_index dest = agx_register(mode == 17 ? 128 :
                                    mode == 15 ? 1 : 4,
                                    mode == 14 ? AGX_SIZE_16 : AGX_SIZE_32);
      agx_index src = agx_register(mode == 18 ? 128 : 0,
                                   mode == 11 ? AGX_SIZE_64 : AGX_SIZE_16);
      if (mode == 9)
         src = agx_uniform(0, AGX_SIZE_16);
      else if (mode == 10)
         src = agx_immediate(0);
      src.abs = mode == 3;
      src.neg = mode == 4;
      src.kill = mode == 5;
      src.cache = mode == 6 || mode == 7;
      src.discard = mode == 7;
      src.memory = mode == 8;
      src.channels_m1 = mode == 12;
      dest.channels_m1 = mode == 13;
      dest.abs = mode == 16;
      dest.memory = mode == 19;
      agx_instr *I = agx_convert_to(
         b, dest, agx_immediate(AGX_CONVERT_F_TO_S32), src,
         mode == 1 ? AGX_ROUND_RTE : AGX_ROUND_RTZ);
      I->saturate = mode == 2;
      I->src[0].neg = mode == 20;
      I->src[0].cache = mode == 21;
      agx_stop(b);
      EXPECT_TRUE(pack(b, false).empty());
   }
}

TEST_F(AGX3PackBinary, G17PHalfTimesMinifloatWidensToF32)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index half = agx_register(0, AGX_SIZE_16);
   agx_fmul_to(b, agx_register(0, AGX_SIZE_32), half, agx_immediate(112));
   agx_stop(b);

   static const uint8_t native[] = {
      0x09, 0xf1, 0x35, 0x00, 0x80, 0x00,
   };
   std::vector<uint8_t> want(std::begin(native), std::end(native));
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);

   agx_builder *bad = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index live = agx_register(0, AGX_SIZE_16);
   live.cache = true;
   agx_fmul_to(bad, agx_register(2, AGX_SIZE_32), live,
               agx_immediate(112));
   agx_stop(bad);
   pack(bad, /*expect_ok=*/false);
}

TEST_F(AGX3PackBinary, QuadImmediateNativeWitnesses)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   const agx_index value = agx_register(6, AGX_SIZE_32);
   agx_quad_shuffle_to(b, agx_register(0, AGX_SIZE_32), value,
                       agx_immediate(2));
   agx_quad_shuffle_xor_to(b, agx_register(2, AGX_SIZE_32), value,
                           agx_immediate(1));
   agx_index cached = agx_register(4, AGX_SIZE_32);
   cached.cache = true;
   agx_quad_shuffle_xor_to(b, cached, value, agx_immediate(2));
   agx_index final_value = value;
   final_value.discard = true;
   agx_quad_shuffle_xor_to(b, agx_register(6, AGX_SIZE_32), final_value,
                           agx_immediate(3));
   agx_stop(b);

   const uint8_t native[] = {
      0x47, 0x00, 0x54, 0x00, 0x02, 0x0c, 0x04, 0x2c, 0x02, 0x00,
      0xc7, 0x00, 0x54, 0x02, 0x02, 0x0c, 0x02, 0x2c, 0x02, 0x00,
      0xc7, 0x00, 0x54, 0x04, 0x03, 0x0c, 0x04, 0x2c, 0x02, 0x00,
      0xc7, 0x00, 0x54, 0x06, 0x02, 0x0c, 0x06, 0x2c, 0x04, 0x00,
   };
   std::vector<uint8_t> want(native, native + sizeof(native));
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
}

/* Apple's own quad butterfly carries the slot-6 wait for its device load in
 * f49, so this no longer costs a synchronous identity prelude to retire the
 * producer first. */
TEST_F(AGX3PackBinary, QuadImmediateXorWaitsForDeviceLoad)
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
   uint8_t bytes[AGX3_SUBGROUP_MAX_LENGTH], length = 0;
   const struct agx3_subgroup_shuffle_desc shuffle = {
      .quad = true,
      .op = AGX3_SUBGROUP_SHUFFLE_XOR,
      .lane_kind = AGX3_SUBGROUP_LANE_IMMEDIATE,
      .dest = 6,
      .src = 4,
      .lane = 1,
      .wait_mask = 1u << 5,
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
      .compare_immediate = true,
      .src_live = true,
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

TEST_F(AGX3PackBinary, G17PBallotUnsignedImmediate)
{
   static const uint8_t native_lt16[] = {
      0x17, 0x17, 0x54, 0x00, 0x02, 0x04, 0x40, 0x18, 0x22, 0x0c,
   };
   static const uint8_t native_gt15[] = {
      0x17, 0x17, 0x54, 0x00, 0x02, 0x04, 0x3c, 0x18, 0x22, 0x0a,
   };
   static const uint8_t native_eq0[] = {
      0x17, 0x17, 0x54, 0x00, 0x02, 0x04, 0x00, 0x18, 0x22, 0x02,
   };

   auto check = [this](enum agx_icond cond, unsigned imm, bool invert,
                       const uint8_t *native, size_t native_size) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index source = agx_register(2, AGX_SIZE_32);
      source.cache = true;
      agx_icmp_ballot_to(b, agx_register(0, AGX_SIZE_32), source,
                         agx_immediate(imm), cond, invert);
      agx_stop(b);

      std::vector<uint8_t> want(native, native + native_size);
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want);
   };

   check(AGX_ICOND_ULT, 16, false, native_lt16, sizeof(native_lt16));
   check(AGX_ICOND_UGT, 15, false, native_gt15, sizeof(native_gt15));
   /* !(x < 16) is canonicalized to Apple's byte-exact x > 15 form. */
   check(AGX_ICOND_ULT, 16, true, native_gt15, sizeof(native_gt15));
   check(AGX_ICOND_UEQ, 0, false, native_eq0, sizeof(native_eq0));

   agx_builder *bad = builder(AGX_ISA_ENCODING_AGX3B);
   agx_icmp_quad_ballot_to(bad, agx_register(0, AGX_SIZE_32),
                           agx_register(2, AGX_SIZE_32), agx_immediate(16),
                           AGX_ICOND_ULT, false);
   agx_stop(bad);
   pack(bad, /*expect_ok=*/false);

   bad = builder(AGX_ISA_ENCODING_AGX3B);
   agx_icmp_ballot_to(bad, agx_register(0, AGX_SIZE_32),
                      agx_register(2, AGX_SIZE_32), agx_immediate(7),
                      AGX_ICOND_SLT, false);
   agx_stop(bad);
   pack(bad, /*expect_ok=*/false);

   bad = builder(AGX_ISA_ENCODING_AGX3B);
   agx_icmp_ballot_to(bad, agx_register(0, AGX_SIZE_32),
                      agx_register(2, AGX_SIZE_32), agx_immediate(0),
                      AGX_ICOND_ULT, true);
   agx_stop(bad);
   pack(bad, /*expect_ok=*/false);
}

TEST_F(AGX3PackBinary, G17PBallotMaskStaysLiveThroughPopcount)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index predicate = agx_register(2, AGX_SIZE_32);
   predicate.cache = true;
   agx_index mask = agx_register(0, AGX_SIZE_32);
   mask.cache = true;
   agx_icmp_ballot_to(b, mask, predicate, agx_immediate(16), AGX_ICOND_ULT,
                      false);
   agx_index live_mask = agx_register(0, AGX_SIZE_32);
   live_mask.cache = true;
   agx_index count = agx_register(4, AGX_SIZE_32);
   count.cache = true;
   agx_popcount_to(b, count, live_mask);
   agx_stop(b);

   /* applegpu-nt 32023.917, the exact ballot+popcount prefix generated for
    * mask + popcount(mask), with both intermediate values still live. */
   static const uint8_t native[] = {
      0x17, 0x17, 0x54, 0x00, 0x03, 0x04, 0x40, 0x18, 0x22, 0x0c,
      0x27, 0x05, 0x54, 0x04, 0x03, 0x00, 0x4e, 0x04,
   };
   std::vector<uint8_t> want(std::begin(native), std::end(native));
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
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

      if (enc == AGX_ISA_ENCODING_AGX3B) {
         const uint16_t src[] = {2, 2};
         const enum agx3_alu_hint hints[] = {
            AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE};
         ASSERT_TRUE(agx3_alu_integer_reg(enc, AGX3_ALU_IADD, 4, src,
                                          hints, 2, bytes, &length));
      } else {
         ASSERT_TRUE(agx3_alu_compose(enc, AGX3_ALU_IADD, 4, 2, 2, 0, 0,
                                      2, bytes, &length));
      }
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
   struct agx3_falu_runtime runtime = {};
   ASSERT_TRUE(agx3_falu_add_mod_runtime(AGX_ISA_ENCODING_AGX3B, 4, 0, 2,
                                         &m, &runtime, mod, &mlen));
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

TEST_F(AGX3PackBinary, TextureCoordinateCollectStaysLowUnderPressure)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   b->shader->nir = rzalloc(mem_ctx, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;

   /* Keep the whole 128-half TEX source bank live when the coordinate COLLECT
    * is reached. Preloads make that pressure deterministic: the allocator
    * must relocate one live value to place this exact direct-COLLECT texture
    * source below half 128, rather than accidentally finding an unused hole. */
   agx_index live[64];
   for (unsigned i = 0; i < ARRAY_SIZE(live); ++i)
      live[i] = agx_preload(b, agx_register(2 * i, AGX_SIZE_32));

   agx_index xy[2] = {
      agx_mov_imm(b, 32, 0x41000000),
      agx_mov_imm(b, 32, 0x41800000),
   };
   agx_index coord = agx_vec_temp(b->shader, AGX_SIZE_32, 2);
   agx_instr *collect = agx_collect_to(b, coord, 2);
   collect->src[0] = xy[0];
   collect->src[1] = xy[1];

   agx_index tex_dest = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
   agx_instr *tex = agx_texture_sample_to(
      b, tex_dest, coord, agx_immediate(0), agx_immediate(0),
      agx_uniform(0, AGX_SIZE_64), agx_immediate(0), agx_null(), AGX_DIM_2D,
      AGX_LOD_MODE_LOD_MIN, 0, false, false, false, AGX_GATHER_NONE, false);
   tex->op = AGX_OPCODE_TEXTURE_LOAD;
   tex->mask = 15;
   agx_instr *split = agx_split(b, 4, tex_dest);
   for (unsigned i = 0; i < 4; ++i)
      split->dest[i] = agx_temp(b->shader, AGX_SIZE_32);

   /* Preserve the synthetic pressure across the COLLECT/TEX pair. */
   agx_index sum = agx_fadd(b, live[0], split->dest[0]);
   for (unsigned i = 1; i < ARRAY_SIZE(live); ++i)
      sum = agx_fadd(b, sum, live[i]);
   agx_export(b, sum, 8);
   agx_stop(b);

   agx_pressure_schedule(b->shader);
   agx_ra(b->shader);

   ASSERT_EQ(tex->op, AGX_OPCODE_TEXTURE_LOAD);
   ASSERT_EQ(tex->src[0].type, AGX_INDEX_REGISTER);
   EXPECT_LT(tex->src[0].value, 128u);
   EXPECT_LE(tex->src[0].value + 4, 128u);
}

TEST_F(AGX3PackBinary, ThreeJSVertexTextureCoordinatesStayLowUnderPressure)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_VERTEX;
   b->shader->nir = rzalloc(mem_ctx, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_VERTEX;

   /* Reproduce the captured three.js skinning allocation: all 128 halves of
    * TEX's staging-COPY source bank are live, while the dynamic descriptor is
    * independently valid in the high bank. The allocator must make a low-bank
    * hole for the direct coordinate COLLECT and preserve that constraint if a
    * later live-range split relocates another value. */
   agx_index live[64];
   for (unsigned i = 0; i < ARRAY_SIZE(live); ++i)
      live[i] = agx_preload(b, agx_register(2 * i, AGX_SIZE_32));
   agx_index high_live[6];
   for (unsigned i = 0; i < ARRAY_SIZE(high_live); ++i)
      high_live[i] = agx_preload(b, agx_register(128 + 2 * i, AGX_SIZE_32));
   agx_index descriptor = agx_register(172, AGX_SIZE_64);

   agx_index xy[2] = {
      agx_mov_imm(b, 32, 0x41000000),
      agx_mov_imm(b, 32, 0x41800000),
   };
   agx_index coord = agx_vec_temp(b->shader, AGX_SIZE_32, 2);
   agx_instr *collect = agx_collect_to(b, coord, 2);
   collect->src[0] = xy[0];
   collect->src[1] = xy[1];

   agx_index result = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
   agx_instr *tex = agx_texture_sample_to(
      b, result, coord, agx_immediate(0), agx_immediate(0), descriptor,
      agx_immediate(0), agx_null(), AGX_DIM_2D, AGX_LOD_MODE_LOD_MIN, 0,
      false, false, false, AGX_GATHER_NONE, false);
   tex->op = AGX_OPCODE_TEXTURE_LOAD;
   tex->mask = 15;
   agx_instr *split = agx_split(b, 4, result);
   for (unsigned i = 0; i < 4; ++i)
      split->dest[i] = agx_temp(b->shader, AGX_SIZE_32);

   agx_index sum = agx_fadd(b, live[0], split->dest[0]);
   for (unsigned i = 1; i < ARRAY_SIZE(live); ++i)
      sum = agx_fadd(b, sum, live[i]);
   for (unsigned i = 0; i < ARRAY_SIZE(high_live); ++i)
      sum = agx_fadd(b, sum, high_live[i]);
   agx_export(b, sum, 8);
   agx_stop(b);

   agx_pressure_schedule(b->shader);
   agx_ra(b->shader);

   ASSERT_EQ(tex->op, AGX_OPCODE_TEXTURE_LOAD);
   ASSERT_EQ(tex->dest[0].type, AGX_INDEX_REGISTER);
   ASSERT_EQ(tex->src[0].type, AGX_INDEX_REGISTER);
   EXPECT_EQ(tex->src[3].type, AGX_INDEX_REGISTER);
   EXPECT_EQ(tex->src[3].value, 172u);
   EXPECT_LT(tex->src[0].value, 128u);
   EXPECT_LE(tex->src[0].value + 4, 128u);

   agx_lower_pseudo(b->shader);
   agx_opt_register_cache(b->shader);
   EXPECT_TRUE(tex->src[0].discard);
   EXPECT_FALSE(pack(b).empty());

   /* The allocator repair must not widen TEX's seven-bit coordinate field.
    * A precoloured reproduction of the captured bad physical instruction
    * remains fail-closed. The independently grounded high descriptor stays
    * legal only when the coordinate itself was allocated in the low bank. */
   agx_builder *bad = builder(AGX_ISA_ENCODING_AGX3B);
   bad->shader->stage = MESA_SHADER_VERTEX;
   bad->shader->nir = rzalloc(mem_ctx, nir_shader);
   bad->shader->nir->info.stage = MESA_SHADER_VERTEX;
   agx_index bad_dest = agx_register(0, AGX_SIZE_32);
   agx_index bad_coord = agx_register(140, AGX_SIZE_32);
   bad_dest.channels_m1 = 3;
   bad_coord.channels_m1 = 1;
   bad_coord.discard = true;
   agx_instr *bad_tex = agx_texture_sample_to(
      bad, bad_dest, bad_coord, agx_immediate(0), agx_immediate(0),
      agx_register(172, AGX_SIZE_64), agx_immediate(0), agx_null(),
      AGX_DIM_2D, AGX_LOD_MODE_LOD_MIN, 0, false, false, false,
      AGX_GATHER_NONE, false);
   bad_tex->op = AGX_OPCODE_TEXTURE_LOAD;
   bad_tex->mask = 15;
   agx_stop(bad);
   EXPECT_TRUE(pack(bad, false).empty());
}

TEST_F(AGX3PackBinary, TextureResultShuffleKeepsHalfCopiesInLowBank)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   b->shader->nir = rzalloc(mem_ctx, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;

   /* Fill every low-bank pair except r6/r7, then keep only r7l live there.
    * This makes r6/r7 the unique cheapest region for the constrained vec2
    * coordinate. XBR has this exact lifetime: preserving the r7l predicate
    * used to generate the unsupported `mov1 r71h, r7l`. */
   agx_index live[62];
   unsigned next = 0;
   for (unsigned reg = 0; reg < 128; reg += 2) {
      if (reg != 12 && reg != 14)
         live[next++] = agx_preload(b, agx_register(reg, AGX_SIZE_32));
   }
   ASSERT_EQ(next, ARRAY_SIZE(live));
   agx_index half_live = agx_preload(b, agx_register(14, AGX_SIZE_16));

   agx_index xy[2] = {
      agx_preload(b, agx_register(128, AGX_SIZE_32)),
      agx_preload(b, agx_register(130, AGX_SIZE_32)),
   };
   agx_index coord = agx_vec_temp(b->shader, AGX_SIZE_32, 2);
   agx_instr *collect = agx_collect_to(b, coord, 2);
   collect->src[0] = xy[0];
   collect->src[1] = xy[1];

   agx_index tex_dest = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
   agx_instr *tex = agx_texture_sample_to(
      b, tex_dest, coord, agx_immediate(0), agx_immediate(0),
      agx_uniform(0, AGX_SIZE_64), agx_immediate(0), agx_null(), AGX_DIM_2D,
      AGX_LOD_MODE_LOD_MIN, 0, false, false, false, AGX_GATHER_NONE, false);
   tex->op = AGX_OPCODE_TEXTURE_LOAD;
   tex->mask = 15;
   agx_instr *split = agx_split(b, 4, tex_dest);
   for (unsigned i = 0; i < 4; ++i)
      split->dest[i] = agx_temp(b->shader, AGX_SIZE_32);

   agx_index sum = agx_fadd(b, live[0], split->dest[0]);
   for (unsigned i = 1; i < ARRAY_SIZE(live); ++i)
      sum = agx_fadd(b, sum, live[i]);
   agx_iadd_to(b, agx_temp(b->shader, AGX_SIZE_16), half_live,
               agx_immediate(0), 0);
   agx_export(b, sum, 8);
   agx_stop(b);

   agx_ra(b->shader);

   agx_foreach_instr_global(b->shader, I) {
      if (I->op == AGX_OPCODE_MOV && I->dest[0].size == AGX_SIZE_16)
         EXPECT_LT(I->dest[0].value, 128u);
   }

   agx_lower_pseudo(b->shader);
   agx_opt_register_cache(b->shader);
   EXPECT_FALSE(pack(b).empty());
}

TEST_F(AGX3PackBinary, TextureFetch2DLod2Immediate)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   agx_index dest = agx_register(0, AGX_SIZE_32);
   agx_index coord = agx_register(12, AGX_SIZE_32);
   dest.channels_m1 = 3;
   coord.channels_m1 = 1;

   agx_instr *I = agx_texture_sample_to(
      b, dest, coord, agx_immediate(2), agx_immediate(0), agx_immediate(0),
      agx_immediate(0), agx_null(), AGX_DIM_2D, AGX_LOD_MODE_LOD_MIN, 0,
      false, false, false, AGX_GATHER_NONE, false);
   I->op = AGX_OPCODE_TEXTURE_LOAD;
   I->mask = 15;
   agx_stop(b);

   struct agx3_tex_fetch_desc d = {};
   d.dest = 0;
   d.coord = 12;
   d.width = AGX3_TEX_W32;
   d.mask = 15;
   d.lod_x64 = 2 * 64;
   d.texture = 0;
   d.slot = 1;
   d.final = true;
   uint8_t bytes[AGX3_TEX_MAX_LENGTH];
   uint8_t length = 0;
   ASSERT_TRUE(agx3_tex_encode_fetch(enc, &d, bytes, &length));
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(enc, &end));

   std::vector<uint8_t> want;
   append(want, bytes, length);
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, TextureFetch3DBindlessStagesThirdCoordinate)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_index dest = agx_register(16, AGX_SIZE_32);
   agx_index coord = agx_register(4, AGX_SIZE_32);
   agx_index texture = agx_register(12, AGX_SIZE_64);
   dest.channels_m1 = 3;
   coord.channels_m1 = 2;
   coord.discard = true;
   texture.discard = true;
   agx_instr *I = agx_texture_sample_to(
      b, dest, coord, agx_immediate(0), agx_immediate(0), texture,
      agx_immediate(0), agx_null(), AGX_DIM_3D, AGX_LOD_MODE_LOD_MIN, 0,
      false, false, false, AGX_GATHER_NONE, false);
   I->op = AGX_OPCODE_TEXTURE_LOAD;
   I->mask = 15;
   agx_stop(b);

   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len = 0;
   std::vector<uint8_t> want;
   for (unsigned component = 0; component < 3; ++component) {
      ASSERT_TRUE(agx3_tex_stage_copy_hints(
         enc, 2 * component, 4 + 2 * component, 1, 0, false, true,
         bytes, &len));
      append(want, bytes, len);
   }
   const struct agx3_tex_fetch_desc fetch = {
      .dest = 16, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
      .slot = 1, .wait_mask = 1, .coord_kill = true, .final = true,
      .three_d = true, .texture_kill = true,
      .texture_kind = AGX3_TEX_OPERAND_REGISTER, .texture_value = 12,
   };
   ASSERT_TRUE(agx3_tex_encode_fetch(enc, &fetch, bytes, &len));
   append(want, bytes, len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(enc, &end));
   append(want, end.bytes, end.length);

   EXPECT_EQ(pack(b), want);
   EXPECT_TRUE(b->shader->agx3_staging_known);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 6);

   b = builder(AGX_ISA_ENCODING_AGX3C);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   dest = agx_register(16, AGX_SIZE_32);
   coord = agx_register(4, AGX_SIZE_32);
   dest.channels_m1 = 3;
   coord.channels_m1 = 2;
   I = agx_texture_sample_to(
      b, dest, coord, agx_immediate(0), agx_immediate(0), agx_immediate(0),
      agx_immediate(0), agx_null(), AGX_DIM_3D, AGX_LOD_MODE_LOD_MIN, 0,
      false, false, false, AGX_GATHER_NONE, false);
   I->op = AGX_OPCODE_TEXTURE_LOAD;
   I->mask = 15;
   agx_stop(b);
   pack(b, /*expect_ok=*/false);
}

TEST_F(AGX3PackBinary, VertexIndexedTextureFetchUsesNativeHeapIndex)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   auto build = [&](bool native_index) {
      agx_builder *b = builder(enc);
      b->shader->stage = MESA_SHADER_VERTEX;
      agx_index dest = agx_register(40, AGX_SIZE_32);
      agx_index coord = agx_register(36, AGX_SIZE_32);
      agx_index texture = agx_register(2, native_index ? AGX_SIZE_64
                                                       : AGX_SIZE_32);
      dest.channels_m1 = 3;
      coord.channels_m1 = 1;
      coord.discard = true;
      texture.discard = true;
      agx_index descriptor_base = native_index
         ? agx_immediate(0)
         : agx_uniform(44, AGX_SIZE_64);
      agx_instr *I = agx_texture_sample_to(
         b, dest, coord, agx_immediate(0), descriptor_base, texture,
         agx_immediate(0), agx_null(), AGX_DIM_2D, AGX_LOD_MODE_LOD_MIN,
         0, false, false, false, AGX_GATHER_NONE, false);
      I->op = AGX_OPCODE_TEXTURE_LOAD;
      I->mask = 15;
      agx_stop(b);
      return b;
   };

   /* Native uses one zero-extended 64-bit heap index. The original AGX2
    * descriptor-base plus 32-bit byte-offset pair is not that operand and
    * must remain fail-closed instead of being reinterpreted. */
   EXPECT_FALSE(pack(build(true)).empty());
   pack(build(false), /*expect_ok=*/false);
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

TEST_F(AGX3PackBinary, TextureSampleOffsetUsesNativeImmediateField)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_index dest = agx_register(8, AGX_SIZE_16);
   agx_index coord = agx_register(4, AGX_SIZE_32);
   dest.channels_m1 = 3;
   coord.channels_m1 = 1;
   coord.discard = true;

   agx_index offset = agx_immediate(0xf1);
   offset.size = AGX_SIZE_32;
   agx_instr *I = agx_texture_sample_to(
      b, dest, coord, agx_immediate(0), agx_immediate(0), agx_immediate(0),
      agx_immediate(0), offset, AGX_DIM_2D,
      AGX_LOD_MODE_AUTO_LOD, 0, true, false, false, AGX_GATHER_NONE, false);
   I->mask = 15;
   /* The packed offset reaches the packer on the instruction, not through
    * src[5]: a comparison sample needs that operand for its reference. */
   I->offset_xy = 0xf1;
   agx_stop(b);

   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len = 0;
   std::vector<uint8_t> want;
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 0, 4, 1, 0, false, true,
                                         bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 2, 6, 1, 0, false, true,
                                         bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_pixel_colour_entry(enc, bytes, &len));
   append(want, bytes, len);

   const struct agx3_tex_sample_desc sample = {
      .dest = 8, .coord = 0, .width = AGX3_TEX_W16, .mask = 15,
      .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
      .final = true, .offset = true, .offset_xy = 0xf1,
   };
   ASSERT_TRUE(agx3_tex_encode_sample(enc, &sample, bytes, &len));
   append(want, bytes, len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(enc, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
   EXPECT_TRUE(b->shader->agx3_colour_link.entry_emitted);
   EXPECT_EQ(b->shader->agx3_colour_link.entry_offset, 8);
   EXPECT_TRUE(b->shader->agx3_staging_known);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 4);
}

TEST_F(AGX3PackBinary, TextureSampleOffsetRefusesRegisterAndAGX3C)
{
   for (unsigned shape = 0; shape < 2; ++shape) {
      const auto enc = shape ? AGX_ISA_ENCODING_AGX3C
                             : AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      agx_index dest = agx_register(8, AGX_SIZE_16);
      agx_index coord = agx_register(4, AGX_SIZE_32);
      dest.channels_m1 = 3;
      coord.channels_m1 = 1;
      agx_index offset = shape ? agx_immediate(0xf1)
                               : agx_register(0, AGX_SIZE_32);
      offset.size = AGX_SIZE_32;
      agx_instr *I = agx_texture_sample_to(
         b, dest, coord, agx_immediate(0), agx_immediate(0), agx_immediate(0),
         agx_immediate(0), offset, AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 0,
         true, false, false, AGX_GATHER_NONE, false);
      I->mask = 15;
      agx_stop(b);
      pack(b, /*expect_ok=*/false);
   }
}

TEST_F(AGX3PackBinary, TextureFetch2DMSPackedSample)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_index dest = agx_register(8, AGX_SIZE_32);
   agx_index coord = agx_register(16, AGX_SIZE_32);
   dest.channels_m1 = 3;
   coord.channels_m1 = 2;
   coord.discard = true;
   agx_instr *I = agx_texture_sample_to(
      b, dest, coord, agx_immediate(0), agx_immediate(0),
      agx_immediate(0), agx_immediate(0), agx_null(), AGX_DIM_2D_MS,
      AGX_LOD_MODE_AUTO_LOD, 0, false, false, false, AGX_GATHER_NONE,
      false);
   I->op = AGX_OPCODE_TEXTURE_LOAD;
   I->mask = 15;
   agx_stop(b);

   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len = 0;
   std::vector<uint8_t> want;
   ASSERT_TRUE(agx3_tex_stage_copy_hints(AGX_ISA_ENCODING_AGX3B, 0, 16, 1,
                                         0, false, true, bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_copy_hints(AGX_ISA_ENCODING_AGX3B, 2, 18, 1,
                                         0, false, true, bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_ms_sample(AGX_ISA_ENCODING_AGX3B, 20, 0,
                                        false, true, bytes, &len));
   append(want, bytes, len);
   struct agx3_tex_fetch_desc fetch = {
      .dest = 8, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
      .slot = 1, .wait_mask = 1, .coord_kill = true, .final = true,
      .multisample = true,
   };
   ASSERT_TRUE(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &fetch,
                                     bytes, &len));
   append(want, bytes, len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, TextureFetch2DMSArrayBindless)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_index dest = agx_register(8, AGX_SIZE_32);
   agx_index coord = agx_register(16, AGX_SIZE_32);
   agx_index texture = agx_register(6, AGX_SIZE_32);
   dest.channels_m1 = 3;
   coord.channels_m1 = 2;
   coord.discard = true;
   agx_instr *I = agx_texture_sample_to(
      b, dest, coord, agx_immediate(0), agx_uniform(8, AGX_SIZE_64),
      texture, agx_immediate(0), agx_null(), AGX_DIM_2D_MS_ARRAY,
      AGX_LOD_MODE_AUTO_LOD, 0, false, false, false, AGX_GATHER_NONE,
      false);
   I->op = AGX_OPCODE_TEXTURE_LOAD;
   I->mask = 15;
   agx_stop(b);

   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len = 0;
   std::vector<uint8_t> want;
   ASSERT_TRUE(agx3_tex_stage_copy_hints(AGX_ISA_ENCODING_AGX3B, 16, 16, 2,
                                         0, false, true, bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_copy_hints(AGX_ISA_ENCODING_AGX3B, 18, 18, 2,
                                         0, false, true, bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_array_layer(AGX_ISA_ENCODING_AGX3B, 20, 2, 0,
                                          false, true, bytes, &len));
   append(want, bytes, len);
   struct agx3_tex_fetch_desc fetch = {
      .dest = 8, .coord = 16, .width = AGX3_TEX_W32, .mask = 15,
      .slot = 1, .wait_mask = 2, .coord_kill = true, .final = true,
      .array = true, .multisample = true,
      .texture_kind = AGX3_TEX_OPERAND_REGISTER,
      .texture_value = 6,
   };
   ASSERT_TRUE(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &fetch,
                                     bytes, &len));
   append(want, bytes, len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);

   /* The MS-array native form is G17P-only and requires its packed layer. */
   b = builder(AGX_ISA_ENCODING_AGX3C);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   dest = agx_register(8, AGX_SIZE_32);
   coord = agx_register(16, AGX_SIZE_32);
   texture = agx_register(6, AGX_SIZE_32);
   dest.channels_m1 = 3;
   coord.channels_m1 = 2;
   I = agx_texture_sample_to(
      b, dest, coord, agx_immediate(0), agx_uniform(8, AGX_SIZE_64),
      texture, agx_immediate(0), agx_null(), AGX_DIM_2D_MS_ARRAY,
      AGX_LOD_MODE_AUTO_LOD, 0, false, false, false, AGX_GATHER_NONE,
      false);
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
      ASSERT_TRUE(enc == AGX_ISA_ENCODING_AGX3B
                     ? agx3_tex_stage_copy_hints(enc, 0, 0, 1, 0, false,
                                                 false, bytes, &len)
                     : agx3_tex_stage_copy(enc, 0, 0, 1, 0, bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(enc == AGX_ISA_ENCODING_AGX3B
                     ? agx3_tex_stage_copy_hints(enc, 2, 2, 1, 0, false,
                                                 false, bytes, &len)
                     : agx3_tex_stage_copy(enc, 2, 2, 1, 0, bytes, &len));
      append(want, bytes, len);
      if (enc == AGX_ISA_ENCODING_AGX3B) {
         ASSERT_TRUE(agx3_pixel_colour_entry(enc, bytes, &len));
         append(want, bytes, len);
      }
      struct agx3_tex_sample_desc d = {};
      d.dest = 8;
      d.coord = 0;
      d.width = AGX3_TEX_W32;
      d.mask = 15;
      d.slot = 1;
      d.wait_mask = 1;
      d.final = true;
      d.sample_abi = AGX3_TEX_SAMPLE_ABI_DRIVER;
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

TEST_F(AGX3PackBinary, Implicit2DRebasesMaterializedCoordinates)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   key.no_stop = true;

   agx_index dest = agx_register(8, AGX_SIZE_32);
   agx_index coord = agx_register(20, AGX_SIZE_32);
   dest.channels_m1 = 3;
   coord.channels_m1 = 1;
   coord.discard = true;
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
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 0, 20, 1, 0, false, true,
                                         bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 2, 22, 1, 0, false, true,
                                         bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_pixel_colour_entry(enc, bytes, &len));
   append(want, bytes, len);
   const struct agx3_tex_sample_desc sample = {
      .dest = 8,
      .coord = 0,
      .width = AGX3_TEX_W32,
      .mask = 15,
      .lod = AGX3_TEX_LOD_AUTO,
      .slot = 1,
      .wait_mask = 1,
      .coord_kill = false,
      .final = true,
      .sample_abi = AGX3_TEX_SAMPLE_ABI_DRIVER,
   };
   ASSERT_TRUE(agx3_tex_encode_sample(enc, &sample, bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_alu_compose(enc, AGX3_ALU_BAND, 0, 0, 0, 0, 0, 1,
                                bytes, &len));
   append(want, bytes, len);

   EXPECT_EQ(pack(b), want);
   /* Fresh attachment-only applegpu_g17p f_retain_coords uses this exact
    * fixed staging bank for materialized r4/r5 coordinates and reports
    * ProgramVariant fields38/42 = 8 bytes. */
   EXPECT_TRUE(b->shader->agx3_staging_known);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 4);
}

TEST_F(AGX3PackBinary, Implicit2DCoordinateCopySourceBank)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;

   /* The first pair is the exact current GTK component-transfer placement;
    * the second ends on the last two legal COPY source halves. */
   for (const unsigned source : {84u, 124u}) {
      agx_builder *b = builder(enc);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      key.no_stop = true;

      agx_index dest = agx_register(8, AGX_SIZE_32);
      agx_index coord = agx_register(source, AGX_SIZE_32);
      dest.channels_m1 = 3;
      coord.channels_m1 = 1;
      coord.discard = true;
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
      ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 0, source, 1, 0, false,
                                            true, bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 2, source + 2, 1, 0,
                                            false, true, bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_pixel_colour_entry(enc, bytes, &len));
      append(want, bytes, len);
      const struct agx3_tex_sample_desc sample = {
         .dest = 8,
         .coord = 0,
         .width = AGX3_TEX_W32,
         .mask = 15,
         .lod = AGX3_TEX_LOD_AUTO,
         .slot = 1,
         .wait_mask = 1,
         .final = true,
         .sample_abi = AGX3_TEX_SAMPLE_ABI_DRIVER,
      };
      ASSERT_TRUE(agx3_tex_encode_sample(enc, &sample, bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_alu_compose(enc, AGX3_ALU_BAND, 0, 0, 0, 0, 0, 1,
                                   bytes, &len));
      append(want, bytes, len);
      EXPECT_EQ(pack(b), want);
   }

   /* Half 128 used to be masked onto half 0 by the low-level composer. The
    * complete post-RA instruction must now fail closed. */
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   key.no_stop = true;
   agx_index dest = agx_register(8, AGX_SIZE_32);
   agx_index coord = agx_register(128, AGX_SIZE_32);
   dest.channels_m1 = 3;
   coord.channels_m1 = 1;
   coord.discard = true;
   agx_instr *tex = agx_texture_sample_to(
      b, dest, coord, agx_immediate(0), agx_immediate(0), agx_immediate(0),
      agx_immediate(0), agx_null(), AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 0,
      false, false, false, AGX_GATHER_NONE, false);
   tex->mask = 15;
   pack(b, /*expect_ok=*/false);
}

TEST_F(AGX3PackBinary, FragmentImplicit2DStagesNativeCoordinates)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;

   agx_index raw_x = agx_register(0, AGX_SIZE_32);
   agx_index reciprocal = agx_register(2, AGX_SIZE_32);
   agx_index raw_y = agx_register(4, AGX_SIZE_32);
   raw_x.discard = raw_y.discard = true;
   reciprocal.cache = true;
   agx_instr *stage_x = agx_persp_mul_agx3_to(
      b, agx_register(20, AGX_SIZE_32), raw_x, reciprocal, 1);
   reciprocal.cache = false;
   reciprocal.discard = true;
   agx_instr *stage_y = agx_persp_mul_agx3_to(
      b, agx_register(22, AGX_SIZE_32), raw_y, reciprocal, 2);

   agx_index dest = agx_register(8, AGX_SIZE_32);
   agx_index coord = agx_register(20, AGX_SIZE_32);
   dest.channels_m1 = 3;
   coord.channels_m1 = 1;
   coord.discard = true;
   agx_instr *tex = agx_texture_sample_to(
      b, dest, coord, agx_immediate(0), agx_immediate(0),
      agx_register(12, AGX_SIZE_64), agx_register(16, AGX_SIZE_64),
      agx_null(), AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 0, false, false,
      false, AGX_GATHER_NONE, false);
   tex->mask = 15;
   stage_x->agx3_tex_coord_stage = true;
   stage_x->agx3_tex_coord_component = 0;
   stage_x->agx3_tex_coord_start = true;
   stage_y->agx3_tex_coord_stage = true;
   stage_y->agx3_tex_coord_component = 1;
   tex->agx3_tex_coords_staged = true;
   agx_stop(b);

   std::vector<uint8_t> want;
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len = 0;
   const struct agx3_persp_mul_desc stage_x_desc = {
      .stage_dest = true,
      .dest = 0,
      .slot = 6,
      .reciprocal = 2,
      .reciprocal_liveness = AGX3_OPERAND_LIVE,
      .raw = 0,
      .raw_liveness = AGX3_OPERAND_KILL,
      .vary_index = 1,
   };
   ASSERT_TRUE(agx3_persp_mul_encode(enc, &stage_x_desc, bytes, &len));
   append(want, bytes, len);
   struct agx3_persp_mul_desc stage_y_desc = stage_x_desc;
   stage_y_desc.dest = 2;
   stage_y_desc.raw = 4;
   stage_y_desc.raw_liveness = AGX3_OPERAND_KILL;
   stage_y_desc.reciprocal_liveness = AGX3_OPERAND_KILL;
   stage_y_desc.vary_index = 2;
   ASSERT_TRUE(agx3_persp_mul_encode(enc, &stage_y_desc, bytes, &len));
   append(want, bytes, len);
   const unsigned expected_entry_offset = want.size();
   ASSERT_TRUE(agx3_pixel_colour_entry(enc, bytes, &len));
   append(want, bytes, len);
   const struct agx3_tex_sample_desc sample = {
      .dest = 8,
      .coord = 0,
      .width = AGX3_TEX_W32,
      .mask = 15,
      .lod = AGX3_TEX_LOD_AUTO,
      .slot = 1,
      .wait_mask = 1u << 5,
      .coord_kill = true,
      .final = true,
      .texture_kind = AGX3_TEX_OPERAND_REGISTER,
      .sampler_kind = AGX3_TEX_OPERAND_REGISTER,
      .texture_value = 12,
      .sampler_value = 16,
      .sample_abi = AGX3_TEX_SAMPLE_ABI_DRIVER,
   };
   ASSERT_TRUE(agx3_tex_encode_sample(enc, &sample, bytes, &len));
   append(want, bytes, len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(enc, &end));
   append(want, end.bytes, end.length);

   EXPECT_EQ(pack(b), want);
   EXPECT_TRUE(b->shader->agx3_colour_link.entry_emitted);
   EXPECT_EQ(b->shader->agx3_colour_link.entry_offset,
             expected_entry_offset);
   EXPECT_TRUE(b->shader->agx3_staging_known);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 4);
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
      if (enc == AGX_ISA_ENCODING_AGX3B) {
         const uint16_t src[] = {4, 4};
         const enum agx3_alu_hint hints[] = {
            AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE};
         ASSERT_TRUE(agx3_alu_integer_reg(enc, AGX3_ALU_IADD, 6, src,
                                          hints, 1, bytes, &len));
      } else {
         ASSERT_TRUE(agx3_alu_compose(enc, AGX3_ALU_IADD, 6, 4, 4, 0, 0,
                                      1, bytes, &len));
      }
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
      if (enc == AGX_ISA_ENCODING_AGX3B) {
         const uint16_t src[] = {0, 0};
         const enum agx3_alu_hint hints[] = {
            AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE};
         ASSERT_TRUE(agx3_alu_integer_reg(enc, AGX3_ALU_IADD, 8, src,
                                          hints, 1, ab, &al));
      } else {
         ASSERT_TRUE(agx3_alu_compose(enc, AGX3_ALU_IADD, 8, 0, 0, 0, 0,
                                      1, ab, &al));
      }
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
                                     AGX3_TILE_SAMPLE_1X, 0, tile, &tile_len));
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
      const unsigned load_slot = enc == AGX_ISA_ENCODING_AGX3B ? 6 : 1;
      ASSERT_TRUE(agx3_tile_load(enc, 0, AGX3_TILE_W32, load_slot, 15, 16,
                                 AGX3_TILE_SAMPLE_1X, 0, true, true, tile,
                                 &tile_len));
      append(want, tile, tile_len);
      ASSERT_TRUE(agx3_tile_store(enc, 0, AGX3_TILE_W32, 15, 16,
                                  AGX3_TILE_SAMPLE_1X,
                                  1u << (load_slot - 1), tile, &tile_len));
      append(want, tile, tile_len);
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(enc, &end));
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want);
   }
}

TEST_F(AGX3PackBinary, G17PMultisampleTileSampleOperandIsARegister)
{
   /* Apple's multisample colour stores source the sample mask from a live
    * register -- the coverage word, or that word ANDed with one sample bit.
    * The packer must carry an IR register straight into that operand. */
   for (unsigned reg : {8u, 9u, 10u, 11u}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index value = agx_register(0, AGX_SIZE_16);
      value.channels_m1 = 1;
      agx_index mask = agx_register(reg, AGX_SIZE_16);
      mask.kill = true;
      agx_st_tile(b, value, mask, agx_null(), AGX_FORMAT_U8NORM, 15, 0, false);
      agx_stop(b);

      uint8_t tile[AGX3_TILE_MAX_LENGTH], tile_len = 0;
      ASSERT_TRUE(agx3_tile_store(AGX_ISA_ENCODING_AGX3B, 0, AGX3_TILE_W8, 15,
                                  0, agx3_tile_sample_reg(reg, true), 0, tile,
                                  &tile_len));
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
      std::vector<uint8_t> want;
      append(want, tile, tile_len);
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want);
   }

   /* The only immediate the operand accepts is Mesa's whole-pixel spelling,
    * which becomes Apple's 1x immediate. A raw numeric mask has no witness
    * and must not be invented. */
   for (unsigned imm : {0u, 1u, 0x3u, 0xfu, 0x20u, 0x40u}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index value = agx_register(0, AGX_SIZE_16);
      value.channels_m1 = 1;
      agx_st_tile(b, value, agx_immediate(imm), agx_null(),
                  AGX_FORMAT_U8NORM, 15, 0, false);
      agx_stop(b);
      EXPECT_TRUE(pack(b, false).empty());
   }
}

TEST_F(AGX3PackBinary, G17PBackgroundWritesCompleteMultisamplePartition)
{
   key.fs.ignore_tib_dependencies = true;

   for (unsigned samples : {2u, 4u}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_index value = agx_register(0, AGX_SIZE_16);
      value.channels_m1 = 1;

      std::vector<uint8_t> want;
      for (unsigned i = 0; i < samples; ++i) {
         agx_index mask = agx_register(8 + i, AGX_SIZE_16);
         mask.kill = true;
         agx_st_tile(b, value, mask, agx_null(), AGX_FORMAT_U8NORM, 15, 0,
                     false);

         uint8_t tile[AGX3_TILE_MAX_LENGTH], tile_len = 0;
         ASSERT_TRUE(agx3_tile_store(AGX_ISA_ENCODING_AGX3B, 0, AGX3_TILE_W8,
                                     15, 0, agx3_tile_sample_reg(8 + i, true),
                                     0, tile, &tile_len));
         append(want, tile, tile_len);
      }
      agx_stop(b);

      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want);
   }

   /* Register allocation is free to reuse one mask register for stores that
    * kill it, and on the real four-sample reload it does. A mask recomputed
    * between the stores is a different mask, so that shape is admitted --
    * refusing it refused the only shader this protocol exists for. */
   {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_index value = agx_register(0, AGX_SIZE_16);
      value.channels_m1 = 1;
      for (unsigned i = 0; i < 4; ++i) {
         agx_mov_imm_to(b, agx_register(8, AGX_SIZE_16), 1u << i);
         agx_index mask = agx_register(8, AGX_SIZE_16);
         mask.kill = true;
         agx_st_tile(b, value, mask, agx_null(), AGX_FORMAT_U8NORM, 15, 0,
                     false);
      }
      agx_stop(b);
      EXPECT_FALSE(pack(b).empty());
   }

   /* A repeated mask register with nothing redefining it is one mask twice,
    * not a complete sample domain. */
   {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_index value = agx_register(0, AGX_SIZE_16);
      value.channels_m1 = 1;
      agx_index mask = agx_register(8, AGX_SIZE_16);
      agx_st_tile(b, value, mask, agx_null(), AGX_FORMAT_U8NORM, 15, 0, false);
      agx_st_tile(b, value, mask, agx_null(), AGX_FORMAT_U8NORM, 15, 0, false);
      agx_stop(b);
      EXPECT_TRUE(pack(b, false).empty());
   }

   /* The same stores in an ordinary fragment require the colour protocol. */
   {
      key.fs.ignore_tib_dependencies = false;
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_index value = agx_register(0, AGX_SIZE_16);
      value.channels_m1 = 1;
      agx_st_tile(b, value, agx_register(8, AGX_SIZE_16), agx_null(),
                  AGX_FORMAT_U8NORM, 15, 0, false);
      agx_st_tile(b, value, agx_register(9, AGX_SIZE_16), agx_null(),
                  AGX_FORMAT_U8NORM, 15, 0, false);
      agx_stop(b);
      EXPECT_TRUE(pack(b, false).empty());
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
      const unsigned first_slot = enc == AGX_ISA_ENCODING_AGX3B ? 6 : 1;
      const unsigned second_slot = enc == AGX_ISA_ENCODING_AGX3B ? 6 : 2;
      ASSERT_TRUE(agx3_tile_load(enc, 0, AGX3_TILE_W32, first_slot, 15, 0,
                                 AGX3_TILE_SAMPLE_1X, 0, true, false, tile,
                                 &tile_len));
      append(want, tile, tile_len);
      ASSERT_TRUE(agx3_tile_load(enc, 8, AGX3_TILE_W32, second_slot, 15, 16,
                                 AGX3_TILE_SAMPLE_1X, 0, false, true, tile,
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

   /*
    * The expected stream is built from the encoders, so it has to state the
    * three things the packer decides and this test used to leave at their
    * defaults:
    *
    *  - the ENCODED shift is the IR shift plus the access format's log2 size
    *    (2 for I32). A register index is scaled only by the instruction's
    *    shift, so dropping the format scale addressed elements as bytes.
    *  - the first device load of a non-fragment shader takes scoreboard slot
    *    6 (agx3_pack_binary's G17P load policy), so its consumer's wait is
    *    bit 17 -- byte 2 = 0x56 -- not byte 1.
    *  - the last LDDEV of the stream carries AGX3b f79=2, which clears bit 4
    *    of byte 2 (0x54 -> 0x44). Apple sets it on the final LDDEV and only
    *    the final one; `metal-nt -arch applegpu_g17p` on a two-load kernel
    *    emits 0x54 then 0x44.
    *
    * The dependent add goes through agx3_alu_integer_reg(), which drives the
    * two operand-lifetime fields from post-RA hints, rather than
    * agx3_alu_compose(), whose per-form baselines still carry the witnessed
    * instruction's own wait and lifetime bits as if they were constants.
    */
      struct agx3_mem_desc ld = {};
      ld.data = 6;
      ld.width = AGX3_MEM_W32;
      ld.mask = 1;
      ld.shift = 4; /* IR shift 2 + log2(sizeof(uint32_t)) */
      ld.base_is_uniform = false;
      ld.base = 0;
      ld.idx_kind = AGX3_MEM_IDX_REG;
      ld.idx_reg = 4;
      /* The G17P load policy puts the first device load on slot 6;
       * AGX3c has no such policy and keeps the round-robin slot. */
      ld.slot = enc == AGX_ISA_ENCODING_AGX3B ? 6 : 2;
      ld.wait_mask = 0x1;
      ld.agx3b_last_load = enc == AGX_ISA_ENCODING_AGX3B;
      uint8_t lb[AGX3_MEM_MAX_LENGTH];
      uint8_t ll = 0;
      ASSERT_TRUE(agx3_mem_encode_load(enc, &ld, lb, &ll));

      /* AGX3b routes an unshifted register add through the lifetime-aware
       * encoder; AGX3c has no such form yet and keeps the composer. */
      const uint16_t add_src[2] = {6, 6};
      const enum agx3_alu_hint add_hints[2] = {AGX3_ALU_HINT_NONE,
                                               AGX3_ALU_HINT_NONE};
      uint8_t ab[AGX3_ALU_MAX_LENGTH];
      uint8_t al = 0;
      ASSERT_TRUE(enc == AGX_ISA_ENCODING_AGX3B
                     ? agx3_alu_integer_reg(enc, AGX3_ALU_IADD, 8, add_src,
                                            add_hints, 1u << 5, ab, &al)
                     : agx3_alu_compose(enc, AGX3_ALU_IADD, 8, 6, 6, 0, 0,
                                        1u << 1, ab, &al));

      struct agx3_mem_desc st = ld;
      st.data = 8;
      st.slot = 0;
      st.wait_mask = 0;
      st.agx3b_last_load = false;
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

TEST_F(AGX3PackBinary, G17PFrontFacingUsesNativeBackFacingSr)
{
   /* Apple's front_facing fragment reads a native W16 SR41. The value is
    * zero on a front face, so it is Mesa's existing BACKFACING semantic and
    * the shared front-face lowering keeps its logical inversion. */
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_get_sr_to(b, agx_register(0, AGX_SIZE_16), AGX_SR_BACKFACING);
   agx_stop(b);
   const std::vector<uint8_t> native = {
      0x04,0xc5,0x10,0x06, 0x0e,0x00,0x00,0x00,
   };
   EXPECT_EQ(pack(b), native);

   /* The semantic identity is grounded only on applegpu_g17p. */
   b = builder(AGX_ISA_ENCODING_AGX3C);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_get_sr_to(b, agx_register(0, AGX_SIZE_16), AGX_SR_BACKFACING);
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
   /*
    * The expected stream is built from the encoders, so it has to state the
    * three things the packer decides and this test used to leave at their
    * defaults:
    *
    *  - the ENCODED shift is the IR shift plus the access format's log2 size
    *    (2 for I32). A register index is scaled only by the instruction's
    *    shift, so dropping the format scale addressed elements as bytes.
    *  - the first device load of a non-fragment shader takes scoreboard slot
    *    6 (agx3_pack_binary's G17P load policy), so its consumer's wait is
    *    bit 17 -- byte 2 = 0x56 -- not byte 1.
    *  - the last LDDEV of the stream carries AGX3b f79=2, which clears bit 4
    *    of byte 2 (0x54 -> 0x44). Apple sets it on the final LDDEV and only
    *    the final one; `metal-nt -arch applegpu_g17p` on a two-load kernel
    *    emits 0x54 then 0x44.
    *
    * The dependent add goes through agx3_alu_integer_reg(), which drives the
    * two operand-lifetime fields from post-RA hints, rather than
    * agx3_alu_compose(), whose per-form baselines still carry the witnessed
    * instruction's own wait and lifetime bits as if they were constants.
    */
      struct agx3_mem_desc ld = {};
      ld.data = 6;
      ld.width = AGX3_MEM_W32;
      ld.mask = 1;
      ld.shift = 4; /* IR shift 2 + log2(sizeof(uint32_t)) */
      ld.base_is_uniform = false;
      ld.base = 0;
      ld.idx_kind = AGX3_MEM_IDX_REG;
      ld.idx_reg = 4;
      /* The G17P load policy puts the first device load on slot 6;
       * AGX3c has no such policy and keeps the round-robin slot. */
      ld.slot = enc == AGX_ISA_ENCODING_AGX3B ? 6 : 1;
      ld.wait_mask = 0;
      ld.agx3b_last_load = enc == AGX_ISA_ENCODING_AGX3B;
      uint8_t lb[AGX3_MEM_MAX_LENGTH];
      uint8_t ll = 0;
      ASSERT_TRUE(agx3_mem_encode_load(enc, &ld, lb, &ll));

      /* AGX3b routes an unshifted register add through the lifetime-aware
       * encoder; AGX3c has no such form yet and keeps the composer. */
      const uint16_t add_src[2] = {6, 6};
      const enum agx3_alu_hint add_hints[2] = {AGX3_ALU_HINT_NONE,
                                               AGX3_ALU_HINT_NONE};
      uint8_t ab[AGX3_ALU_MAX_LENGTH];
      uint8_t al = 0;
      ASSERT_TRUE(enc == AGX_ISA_ENCODING_AGX3B
                     ? agx3_alu_integer_reg(enc, AGX3_ALU_IADD, 8, add_src,
                                            add_hints, 1u << 5, ab, &al)
                     : agx3_alu_compose(enc, AGX3_ALU_IADD, 8, 6, 6, 0, 0,
                                        1u << 0, ab, &al));

      struct agx3_mem_desc st = ld;
      st.data = 8;
      st.slot = 0;
      st.wait_mask = 0;
      st.agx3b_last_load = false;
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

      const uint16_t add_src[2] = {0, 0};
      const enum agx3_alu_hint add_hints[2] = {AGX3_ALU_HINT_NONE,
                                               AGX3_ALU_HINT_NONE};
      uint8_t ab[AGX3_ALU_MAX_LENGTH];
      uint8_t al = 0;
      ASSERT_TRUE(enc == AGX_ISA_ENCODING_AGX3B
                     ? agx3_alu_integer_reg(enc, AGX3_ALU_IADD, 2, add_src,
                                            add_hints, 1, ab, &al)
                     : agx3_alu_compose(enc, AGX3_ALU_IADD, 2, 0, 0, 0, 0, 1,
                                        ab, &al));

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

TEST_F(AGX3PackBinary, ExplicitSampleInterpolationMatchesApple)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_FRAGMENT;

   agx_index sample = agx_register(4, AGX_SIZE_16);
   sample.discard = true;
   agx_itrcoords_agx3_to(b, agx_register(4, AGX_SIZE_32), sample);

   agx_index coords = agx_register(4, AGX_SIZE_32);
   coords.discard = true;
   agx_interp_agx3_to(b, agx_register(4, AGX_SIZE_32), agx_immediate(0), coords,
                      AGX_INTERPOLATION_SAMPLE, false);
   agx_stop(b);

   /* Fresh Apple sample_no_perspective fragment: tied terminal ITRCOORDS
    * followed by its final scalar f32interp and Mesa's canonical end. */
   const std::vector<uint8_t> native = {
      0xaf, 0x04, 0x54, 0x04, 0x03, 0x08, 0x0a, 0x03, 0x2f, 0x05, 0x54,
      0x04, 0x02, 0x00, 0x02, 0x02, 0x20, 0x00, 0x0e, 0x00, 0x00, 0x00,
   };
   EXPECT_EQ(pack(b), native);
}

TEST_F(AGX3PackBinary, ExplicitSampleInterpolationStaysFailClosed)
{
   {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3C);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_itrcoords_agx3_to(b, agx_register(4, AGX_SIZE_32),
                            agx_register(2, AGX_SIZE_16));
      agx_stop(b);
      EXPECT_TRUE(pack(b, /*expect_ok=*/false).empty());
   }
   {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      /* A register is not a coordinate token without a dominating native
       * ITRCOORDS producer, even though it fits the physical f32 field. */
      agx_interp_agx3_to(b, agx_register(0, AGX_SIZE_32), agx_immediate(3),
                         agx_register(2, AGX_SIZE_32), AGX_INTERPOLATION_SAMPLE,
                         false);
      agx_stop(b);
      EXPECT_TRUE(pack(b, /*expect_ok=*/false).empty());
   }
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
      ASSERT_EQ(len, AGX3_CF_BRANCH_LENGTH);
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
      ASSERT_EQ(len, AGX3_CF_BRANCH_LENGTH);
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

TEST_F(AGX3PackBinary, FirefoxFloatIfMinifloatsMaterializeToGroundedRegisters)
{
   for (float constant : {1.0f, -10.0f}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      b->shader->nir = rzalloc(mem_ctx, nir_shader);
      b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;

      agx_index source = agx_mov_imm(b, 32, 0x40000000);
      agx_if_fcmp(b, source, agx_immediate_f(constant), 1, AGX_FCOND_EQ,
                  false, NULL);
      agx_stop(b);

      agx_materialize_agx3_cf_constants(b->shader);

      uint32_t expected;
      memcpy(&expected, &constant, sizeof(expected));
      unsigned matching_constant = 0, compares = 0;
      agx_foreach_instr_global(b->shader, I) {
         matching_constant +=
            I->op == AGX_OPCODE_MOV_IMM && I->imm == expected;
         if (I->op == AGX_OPCODE_IF_FCMP) {
            ++compares;
            EXPECT_EQ(I->src[0].type, AGX_INDEX_NORMAL);
            EXPECT_EQ(I->src[1].type, AGX_INDEX_NORMAL);
            EXPECT_EQ(I->src[0].size, AGX_SIZE_32);
            EXPECT_EQ(I->src[1].size, AGX_SIZE_32);
         }
      }
      EXPECT_EQ(matching_constant, 1u);
      EXPECT_EQ(compares, 1u);

      agx_ra(b->shader);
      agx_lower_pseudo(b->shader);
      agx_opt_register_cache(b->shader);
      EXPECT_FALSE(pack(b).empty());
   }
}

TEST_F(AGX3PackBinary, MemoryDependentControlFlowRetiresOutstandingLoad)
{
   /* A load before divergent control flow is retired on the straight-line
    * edge, so both arms enter with the same empty scoreboard. */
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index base = agx_register(0, AGX_SIZE_64);
   agx_index index = agx_register(4, AGX_SIZE_32);
   agx_index loaded = agx_register(6, AGX_SIZE_32);
   agx_index cond = agx_register(8, AGX_SIZE_32);
   agx_device_load_to(b, loaded, base, index, AGX_FORMAT_I32, 1, 2, false);
   agx_if_icmp(b, cond, agx_immediate(0), 1, AGX_ICOND_UEQ, true, NULL);
   agx_stop(b);

   const auto bin = pack(b);
   uint8_t wait[AGX3_ALU_MAX_LENGTH], len = 0;
   bool found_wait = false;
   for (unsigned slot = 0; slot < 6; ++slot) {
      ASSERT_TRUE(agx3_alu_compose(AGX_ISA_ENCODING_AGX3B, AGX3_ALU_BAND,
                                   0, 0, 0, 0, 0, 1u << slot, wait, &len));
      found_wait |=
         std::search(bin.begin(), bin.end(), wait, wait + len) != bin.end();
   }
   EXPECT_TRUE(found_wait);
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

TEST_F(AGX3PackBinary, G17PAbsoluteMultiplierImmediateAddend)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_index src0 = agx_register(0, AGX_SIZE_32);
   agx_index src1 = agx_register(2, AGX_SIZE_32);
   src0.discard = true;
   src1.abs = true;
   src1.discard = true;
   agx_ffma_to(b, agx_register(0, AGX_SIZE_32), src0, src1,
               agx_immediate(48));
   agx_stop(b);

   /* applegpu-nt v32023.255, -arch applegpu_g17p, for
    * fma(a, fabs(b), 1.0f), followed by the canonical end. */
   static const uint8_t native[] = {
      0x09,0x01,0x1e,0x03,0x23,0xb0,0x02,0x00,0x00,0x80,0x02,0x00,
      0x0e,0x00,0x00,0x00,
   };
   EXPECT_EQ(pack(b), std::vector<uint8_t>(std::begin(native),
                                            std::end(native)));
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
      0x67,0x0e,0x54,0x00,0x00,0x00,0x01,0xce,0x02,0x00,0x00,0x00,
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
                               15, 0, AGX3_TILE_SAMPLE_1X, 1u << 5, store, &len));
   ASSERT_EQ(len, 12u);
   EXPECT_TRUE(std::equal(store, store+len, bin.begin()+sizeof(prefix)));
}

TEST_F(AGX3PackBinary, G17PAlphaProtocolDoesNotBroadenFormatOffsetOrControlFlow)
{
   /* S8NORM is now independently qualified; offset and CF remain excluded. */
   for (unsigned axis = 1; axis < 3; ++axis) {
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

TEST_F(AGX3PackBinary, NativeImplicitCoordinateStagingForms)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;

   const uint16_t add0_src[2] = {8, 0};
   const enum agx3_alu_hint add0_hints[2] = {
      AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD};
   const uint8_t add0_native[] = {0x09,0x01,0x3c,0x09,0x40,0xc0};
   ASSERT_TRUE(agx3_alu_fadd_stage(enc, 0, 1, add0_src, add0_hints,
                                   1u << 5, bytes, &len));
   EXPECT_EQ(std::vector<uint8_t>(bytes, bytes + len),
             std::vector<uint8_t>(std::begin(add0_native),
                                  std::end(add0_native)));

   const uint16_t add1_src[2] = {4, 2};
   const enum agx3_alu_hint add1_hints[2] = {
      AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD};
   const uint8_t add1_native[] = {0x19,0x03,0x3c,0x05,0x40,0x00};
   ASSERT_TRUE(agx3_alu_fadd_stage(enc, 2, 1, add1_src, add1_hints, 0,
                                   bytes, &len));
   EXPECT_EQ(std::vector<uint8_t>(bytes, bytes + len),
             std::vector<uint8_t>(std::begin(add1_native),
                                  std::end(add1_native)));

   const struct agx3_persp_mul_desc compact = {
      .stage_dest = true,
      .dest = 0,
      .slot = 1,
      .reciprocal = 0,
      .reciprocal_liveness = AGX3_OPERAND_LIVE,
      .raw = 4,
      .raw_liveness = AGX3_OPERAND_KILL,
      .vary_index = 1,
   };
   const uint8_t compact_native[] = {
      0x09,0x05,0x2f,0x81,0x00,0x01,0x40,0x00};
   ASSERT_TRUE(agx3_persp_mul_encode(enc, &compact, bytes, &len));
   EXPECT_EQ(std::vector<uint8_t>(bytes, bytes + len),
             std::vector<uint8_t>(std::begin(compact_native),
                                  std::end(compact_native)));

   struct agx3_persp_mul_desc extended = compact;
   extended.slot = 6;
   extended.reciprocal = 4;
   extended.raw = 8;
   const uint8_t extended_native[] = {
      0x09,0x09,0x2f,0x85,0x00,0x01,0x42,0x00,0x00,0x14,0x00,0x00};
   ASSERT_TRUE(agx3_persp_mul_encode(enc, &extended, bytes, &len));
   EXPECT_EQ(std::vector<uint8_t>(bytes, bytes + len),
             std::vector<uint8_t>(std::begin(extended_native),
                                  std::end(extended_native)));
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
      0xaf,0,0x54,0,2,0,0,0x48,0x20,0, 0x0e,0,0,0,
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

TEST_F(AGX3PackBinary, FragmentInputStartsOnceWithFlatFirst)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_ldcf_to(b, agx_register(2, AGX_SIZE_32), agx_immediate(4), 1);
   agx_index rhw = agx_register(8, AGX_SIZE_32);
   agx_rhw_interp_agx3_to(b, rhw, agx_immediate(0), agx_immediate(0),
                         AGX_INTERPOLATION_CENTER);
   agx_persp_rcp_agx3_to(b, agx_register(10, AGX_SIZE_32), rhw, 0);
   agx_stop(b);

   uint8_t bytes[AGX3_VARY_MAX_LENGTH], len = 0;
   ASSERT_TRUE(agx3_vary_ldcf_at_start(AGX_ISA_ENCODING_AGX3B, 2, 4, 1,
                                      true, bytes, &len));
   std::vector<uint8_t> prefix(bytes, bytes + len);
   /* Apple mixed_wide begins with marked LDCF `1f0b...`, followed by its
    * first F32INTERP as `2f05...`: the two families share one entry. */
   ASSERT_TRUE(agx3_vary_projective_rhw(AGX_ISA_ENCODING_AGX3B, 8, 0, false,
                                       bytes, &len));
   prefix.insert(prefix.end(), bytes, bytes + len);

   const auto code = pack(b);
   ASSERT_GE(code.size(), prefix.size());
   EXPECT_TRUE(std::equal(prefix.begin(), prefix.end(), code.begin()));
}

TEST_F(AGX3PackBinary, FragmentInputStartsOnceWithSmoothFirst)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_interp_agx3_to(b, agx_register(8, AGX_SIZE_32), agx_immediate(0),
                      agx_immediate(0), AGX_INTERPOLATION_CENTER, false);
   agx_ldcf_to(b, agx_register(2, AGX_SIZE_32), agx_immediate(4), 1);
   agx_stop(b);

   const auto code = pack(b);
   ASSERT_GE(code.size(), 16u);
   EXPECT_EQ(code[0], 0x2f);
   EXPECT_EQ(code[1] & 8, 8u); /* marked F32INTERP */
   EXPECT_EQ(code[10], 0x1f);
   EXPECT_EQ(code[11] & 8, 0u); /* unmarked LDCF */
}

TEST_F(AGX3PackBinary, FragmentHelperProtocolLeavesInputEntryUnmarked)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_ldcf_to(b, agx_register(2, AGX_SIZE_32), agx_immediate(4), 1);
   agx_dfdx_to(b, agx_register(4, AGX_SIZE_32),
               agx_register(6, AGX_SIZE_32));
   agx_stop(b);

   const auto code = pack(b);
   ASSERT_GE(code.size(), 16u);
   EXPECT_EQ(code[0], 0x1f);
   EXPECT_EQ(code[1] & 8, 0u); /* helper-capable LDCF: f48 clear */

   /* Removing the derivative is a negative control: the same plain LDCF is
    * the fragment-input entry and retains Apple's f48 marker. */
   agx_builder *plain = builder(AGX_ISA_ENCODING_AGX3B);
   plain->shader->stage = MESA_SHADER_FRAGMENT;
   agx_ldcf_to(plain, agx_register(2, AGX_SIZE_32), agx_immediate(4), 1);
   agx_stop(plain);
   const auto plain_code = pack(plain);
   ASSERT_GE(plain_code.size(), 6u);
   EXPECT_EQ(plain_code[0], 0x1f);
   EXPECT_EQ(plain_code[1] & 8, 8u);
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
            ASSERT_TRUE(agx3_tile_load(enc, 12, AGX3_TILE_W8, 6, 15, 0,
               AGX3_TILE_SAMPLE_1X, 0, true, true, bytes, &len));
            append(want, bytes, len);
            agx_index unpacked = agx_register(8, AGX_SIZE_32);
            unpacked.channels_m1 = 1;
            agx_normflt_agx3_to(b, unpacked, agx_register(12, AGX_SIZE_16),
                               AGX_FORMAT_U8NORM);
            struct agx3_normflt_desc d = {};
            d.dest = 8; d.dest_hint = true; d.src = 12;
            d.two_lanes = true; d.wait_mask = 1u << 5;
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
                                    AGX3_TILE_SAMPLE_1X, 0, bytes, &len));
         append(want, bytes, len);
         append(want, release_end, sizeof(release_end));
         agx_stop(b);
         EXPECT_EQ(pack(b), want);
      }
   }
}

TEST_F(AGX3PackBinary, G17PWriteWaitDoesNotCrossUnprovenOperations)
{
   /* S8NORM is now independently qualified for the deferred protocol. */
   for (unsigned axis = 1; axis < 6; ++axis) {
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

TEST_F(AGX3PackBinary, G17PTexelFetchLODMaterializesUniform)
{
   EXPECT_FALSE(agx_instr_accepts_uniform(AGX_ISA_ENCODING_AGX3B,
      AGX_OPCODE_TEXTURE_LOAD, 1, 16, AGX_SIZE_32));
   EXPECT_TRUE(agx_instr_accepts_uniform(AGX_ISA_ENCODING_AGX3B,
      AGX_OPCODE_TEXTURE_LOAD, 2, 16, AGX_SIZE_64));
   EXPECT_TRUE(agx_instr_accepts_uniform(AGX_ISA_ENCODING_AGX2,
      AGX_OPCODE_TEXTURE_LOAD, 1, 16, AGX_SIZE_16));
}

TEST_F(AGX3PackBinary, TextureFetchRegisterLodIsFullUint)
{
   auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   key.no_stop = true;

   agx_index dest = agx_register(8, AGX_SIZE_32);
   dest.channels_m1 = 3;
   agx_index coord = agx_register(12, AGX_SIZE_32);
   coord.channels_m1 = 1;
   agx_instr *I = agx_texture_sample_to(
      b, dest, coord, agx_register(4, AGX_SIZE_32), agx_immediate(0),
      agx_immediate(0), agx_immediate(0), agx_null(), AGX_DIM_2D,
      AGX_LOD_MODE_LOD_MIN, 0, false, false, false, AGX_GATHER_NONE, false);
   I->op = AGX_OPCODE_TEXTURE_LOAD;
   I->mask = 15;

   std::vector<uint8_t> want;
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
   ASSERT_TRUE(agx3_tex_stage_typed_lod(
      enc, 4, 2, 0, false, false, AGX3_TEX_LEVEL_INTEGER, bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 0, 12, 2, 0, false, false,
                                         bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 2, 14, 2, 0, false, false,
                                         bytes, &len));
   append(want, bytes, len);
   struct agx3_tex_fetch_desc d = {};
   d.dest = 8;
   d.width = AGX3_TEX_W32;
   d.mask = 15;
   d.slot = 1;
   d.wait_mask = 1u << 1;
   d.coord_kill = true;
   d.final = true;
   d.register_lod = true;
   ASSERT_TRUE(agx3_tex_encode_fetch(enc, &d, bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_alu_compose(enc, AGX3_ALU_BAND, 0, 0, 0, 0, 0, 1,
                                bytes, &len));
   append(want, bytes, len);
   EXPECT_EQ(pack(b), want);

   agx_builder *bad = builder(enc);
   bad->shader->stage = MESA_SHADER_FRAGMENT;
   key.no_stop = true;
   agx_index bad_dest = agx_register(8, AGX_SIZE_32);
   bad_dest.channels_m1 = 3;
   agx_index bad_coord = agx_register(12, AGX_SIZE_32);
   bad_coord.channels_m1 = 1;
   agx_instr *bad_I = agx_texture_sample_to(
      bad, bad_dest, bad_coord, agx_register(4, AGX_SIZE_16),
      agx_immediate(0), agx_immediate(0), agx_immediate(0), agx_null(),
      AGX_DIM_2D, AGX_LOD_MODE_LOD_MIN, 0, false, false, false,
      AGX_GATHER_NONE, false);
   bad_I->op = AGX_OPCODE_TEXTURE_LOAD;
   bad_I->mask = 15;
   pack(bad, /*expect_ok=*/false);
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

TEST_F(AGX3PackBinary, G17PNormalized16MatchesNativeColourPrograms)
{
   const enum agx_isa_encoding enc = AGX_ISA_ENCODING_AGX3B;

   {
      agx_builder *b = builder(enc);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_fltnorm_agx3_to(b, agx_register(0, AGX_SIZE_16),
                          agx_immediate_f(1.0f), agx_null(),
                          AGX_FORMAT_U16NORM);
      agx_stop(b);
      const std::vector<uint8_t> native = {
         0x97,0x0c,0x54,0x00,0x02,0x60,0x00,0x10,0x04,0x80,
         0x0e,0x00,0x00,0x00,
      };
      EXPECT_EQ(pack(b), native);
   }

   {
      agx_builder *b = builder(enc);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_index pair = agx_register(0, AGX_SIZE_16);
      pair.channels_m1 = 1;
      agx_fltnorm_agx3_to(b, pair, agx_immediate_f(1.0f),
                          agx_immediate_f(0.5f), AGX_FORMAT_U16NORM);
      agx_stop(b);
      const std::vector<uint8_t> native = {
         0x97,0x0c,0x54,0x00,0x02,0x60,0x80,0x50,0x04,0x88,
         0x0e,0x00,0x00,0x00,
      };
      EXPECT_EQ(pack(b), native);
   }

   for (bool snorm : {false, true}) {
      struct agx3_normflt_desc d = {};
      d.dest = 2;
      d.dest_hint = true;
      d.src = 2;
      d.is_16bit = true;
      d.snorm = snorm;
      d.wait_mask = 1u << 5;
      uint8_t bytes[AGX3_FLTNORM_MAX_LENGTH], len = 0;
      ASSERT_TRUE(agx3_normflt_encode(enc, &d, bytes, &len));
      const std::vector<uint8_t> native = snorm
         ? std::vector<uint8_t>{0x17,0x04,0x56,0x02,0x01,0x04,0x14,0x2a}
         : std::vector<uint8_t>{0x17,0x04,0x56,0x02,0x01,0x04,0x14,0x4a};
      EXPECT_EQ(std::vector<uint8_t>(bytes, bytes + len), native);
   }
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
      unsigned value, component, slot, wait;
      bool kill;
      uint8_t bytes[8];
   } native[] = {
      {8, 4, 5, 16, true, {0x57,0x06,0x55,0x08,0x80,0x40,0x4c,0x00}},
      {16, 8, 6, 16, true, {0x57,0x06,0x55,0x10,0x00,0x41,0x4d,0x00}},
      {18, 9, 5, 0, true, {0x57,0x06,0x54,0x12,0x20,0x41,0x4c,0x00}},
      {20, 10, 4, 8, true, {0x57,0x86,0x54,0x14,0x40,0x41,0x4b,0x00}},
      {62, 31, 2, 2, true, {0x57,0x26,0x54,0x3e,0xe0,0x43,0x49,0x00}},
      /* Fresh Apple G17P 36-scalar vertex output, AIR 0b42132b9e9a... /
       * native b92aedf08dfd...; the final user scalar is UVS index 35. */
      {72, 35, 4, 0, true, {0x57,0x06,0x54,0x48,0x60,0x44,0x4b,0x00}},
   };
   for (auto &v : native) {
      uint8_t out[AGX3_VARY_MAX_LENGTH], len;
      ASSERT_TRUE(agx3_vary_uvsw(AGX_ISA_ENCODING_AGX3B, v.value,
                                v.component, v.slot, v.kill, v.wait, out, &len));
      ASSERT_EQ(len, 8u);
      EXPECT_EQ(std::vector<uint8_t>(out, out + 8),
                std::vector<uint8_t>(v.bytes, v.bytes + 8));
   }
}

TEST_F(AGX3PackBinary, WideVertexOutputsPreserveValuesAndRecycleSlotsAfterWrites)
{
   for (unsigned count : {5u, 7u, 8u, 9u, 11u, 16u, 32u, 52u}) {
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

TEST_F(AGX3PackBinary, NativeMaximumVertexOutputs)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_VERTEX;

   /* The promoted Apple vs68 witness grounds the same dense UVSW format used
    * by the opt-in 72-output qualification shape. Exercise indices 0..71 and
    * staging half-indices 0..142. A shared source keeps this codec test
    * independent of the register allocator's synthetic no-spill pressure. */
   for (unsigned c = 0; c < AGX3_MAX_VERTEX_OUTPUTS; ++c)
      agx_st_vary(b, agx_immediate(c), agx_register(4, AGX_SIZE_32));
   agx_stop(b);

   const auto bytes = pack(b);
   EXPECT_FALSE(bytes.empty());
   EXPECT_EQ(b->shader->agx3_staging_halfs,
             2u * AGX3_MAX_VERTEX_OUTPUTS);

   agx_builder *too_wide = builder(enc);
   too_wide->shader->stage = MESA_SHADER_VERTEX;
   for (unsigned c = 0; c <= AGX3_MAX_VERTEX_OUTPUTS; ++c)
      agx_st_vary(too_wide, agx_immediate(c),
                  agx_register(4, AGX_SIZE_32));
   agx_stop(too_wide);
   EXPECT_TRUE(pack(too_wide, false).empty());
}

TEST_F(AGX3PackBinary, SideEffectBeforeWideTerminalVertexOutputs)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_VERTEX;

   /* Apple's explicit-level vertex texture operation is independent of the
    * terminal UVSW protocol. This is the shape emitted by Firefox WebRender:
    * a texture read followed by a dense, wide output suffix. */
   agx_index tex_dest = agx_register(20, AGX_SIZE_32);
   agx_index coord = agx_register(12, AGX_SIZE_32);
   tex_dest.channels_m1 = 3;
   coord.channels_m1 = 1;
   agx_instr *tex = agx_texture_sample_to(
      b, tex_dest, coord, agx_immediate(0), agx_immediate(0),
      agx_immediate(0), agx_immediate(0), agx_null(), AGX_DIM_2D,
      AGX_LOD_MODE_LOD_MIN, 0, false, false, false, AGX_GATHER_NONE, false);
   tex->op = AGX_OPCODE_TEXTURE_LOAD;
   tex->mask = 15;

   for (unsigned c = 0; c < 51; ++c)
      agx_st_vary(b, agx_immediate(c), agx_register(4, AGX_SIZE_32));
   agx_stop(b);

   const auto bytes = pack(b);
   EXPECT_FALSE(bytes.empty());
   EXPECT_EQ(b->shader->agx3_staging_halfs, 102u);
}

/* WebRender materializes the constant final output after the preceding fifty
 * stores.  Those stores are not a syntactically terminal bundle, but each is
 * copied to its unique staging address before the late producer can reuse its
 * GPR.  The first old refusal was therefore output 4, not an unsupported UVSW
 * field: the nonterminal staging mode was arbitrarily limited to six outputs. */
TEST_F(AGX3PackBinary, FirefoxDenseOutputsAllowLatePureProducer)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   for (unsigned count : {40u, 51u}) {
      agx_builder *b = builder(enc);
      b->shader->stage = MESA_SHADER_VERTEX;

      for (unsigned c = 0; c + 1 < count; ++c)
         agx_st_vary(b, agx_immediate(c), agx_register(24, AGX_SIZE_32));
      agx_mov_imm_to(b, agx_register(24, AGX_SIZE_32), 0x3f800000);
      agx_st_vary(b, agx_immediate(count - 1),
                  agx_register(24, AGX_SIZE_32));
      agx_stop(b);

      const auto bytes = pack(b);
      EXPECT_FALSE(bytes.empty()) << count;
      EXPECT_EQ(b->shader->agx3_staging_halfs, 2u * count) << count;
   }
}

/* Firefox 155's WebRender vertex program is the same shape with 49 dense
 * outputs: its final store's source GPR is also read by an earlier store, so
 * the late pure producer is only safe because every earlier value has already
 * been copied to its own staging address.  That safety argument does not
 * depend on the number of outputs, and the captured 40/51 pair was only ever
 * what happened to be captured -- pinning it refused Firefox at st_vary #4 and
 * left the browser with no WebRender pipeline at all. */
TEST_F(AGX3PackBinary, FirefoxDenseOutputsAcceptAnyGroundedWidth)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   for (unsigned count : {5u, 39u, 40u, 49u, 51u, AGX3_MAX_VERTEX_OUTPUTS}) {
      agx_builder *b = builder(enc);
      b->shader->stage = MESA_SHADER_VERTEX;

      for (unsigned c = 0; c + 1 < count; ++c)
         agx_st_vary(b, agx_immediate(c), agx_register(24, AGX_SIZE_32));
      agx_mov_imm_to(b, agx_register(24, AGX_SIZE_32), 0x3f800000);
      agx_st_vary(b, agx_immediate(count - 1),
                  agx_register(24, AGX_SIZE_32));
      agx_stop(b);

      const auto bytes = pack(b);
      EXPECT_FALSE(bytes.empty()) << count;
      EXPECT_EQ(b->shader->agx3_staging_halfs, 2u * count) << count;
   }

   /* The width ceiling still fails closed. */
   agx_builder *too_wide = builder(enc);
   too_wide->shader->stage = MESA_SHADER_VERTEX;
   for (unsigned c = 0; c < AGX3_MAX_VERTEX_OUTPUTS; ++c)
      agx_st_vary(too_wide, agx_immediate(c), agx_register(24, AGX_SIZE_32));
   agx_mov_imm_to(too_wide, agx_register(24, AGX_SIZE_32), 0x3f800000);
   agx_st_vary(too_wide, agx_immediate(AGX3_MAX_VERTEX_OUTPUTS),
               agx_register(24, AGX_SIZE_32));
   agx_stop(too_wide);
   EXPECT_TRUE(pack(too_wide, false).empty());
}

TEST_F(AGX3PackBinary, FirefoxDenseOutputsKeepOtherLateProducersClosed)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_VERTEX;

   for (unsigned c = 0; c < 50; ++c)
      agx_st_vary(b, agx_immediate(c), agx_register(24, AGX_SIZE_32));
   agx_mov_imm_to(b, agx_register(24, AGX_SIZE_32), 0x3f000000);
   agx_st_vary(b, agx_immediate(50), agx_register(24, AGX_SIZE_32));
   agx_stop(b);

   EXPECT_TRUE(pack(b, false).empty());

   /* The late producer must be the exclusive source of the final store.  If
    * the last store reads some other register the staged value it publishes is
    * unrelated to the producer, so the suffix is not the grounded shape. */
   b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_VERTEX;
   for (unsigned c = 0; c < 50; ++c)
      agx_st_vary(b, agx_immediate(c), agx_register(24, AGX_SIZE_32));
   agx_mov_imm_to(b, agx_register(26, AGX_SIZE_32), 0x3f800000);
   agx_st_vary(b, agx_immediate(50), agx_register(24, AGX_SIZE_32));
   agx_stop(b);
   EXPECT_TRUE(pack(b, false).empty());

   /* A late producer that lands before the final store's predecessors, rather
    * than immediately ahead of the final store, is likewise not the shape. */
   b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_VERTEX;
   for (unsigned c = 0; c < 49; ++c)
      agx_st_vary(b, agx_immediate(c), agx_register(24, AGX_SIZE_32));
   agx_mov_imm_to(b, agx_register(24, AGX_SIZE_32), 0x3f800000);
   agx_st_vary(b, agx_immediate(49), agx_register(24, AGX_SIZE_32));
   agx_st_vary(b, agx_immediate(50), agx_register(24, AGX_SIZE_32));
   agx_stop(b);
   EXPECT_TRUE(pack(b, false).empty());
}

TEST_F(AGX3PackBinary, ReorderedWideVertexOutputsMatchNativeStagingProtocol)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_VERTEX;
   const unsigned order[] = {
      4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,
      26,27,28,29,30,31,32,33,0,1,2,3,
   };
   for (unsigned component : order)
      agx_st_vary(b, agx_immediate(component),
                  agx_register(4, AGX_SIZE_32));
   agx_stop(b);

   std::vector<uint8_t> want;
   for (unsigned position = 0; position < ARRAY_SIZE(order); ++position) {
      const unsigned component = order[position];
      uint8_t bytes[AGX3_STAGE_MAX_LENGTH], len = 0;
      const uint16_t wait = position && (position % 6) == 0 ? 0x3f : 0;
      ASSERT_TRUE(agx3_stage_copy_source(enc, component * 2, 4, false,
                                        component % 6 + 1, wait, bytes, &len));
      append(want, bytes, len);
      if (((position + 1) % 6) == 0 || position + 1 == ARRAY_SIZE(order)) {
         const unsigned first = position - (position % 6);
         for (unsigned i = first; i <= position; ++i) {
            const unsigned output = order[i];
            ASSERT_TRUE(agx3_vary_uvsw(enc, output * 2, output,
                                      output % 6 + 1, true,
                                      1u << (output % 6), bytes, &len));
            append(want, bytes, len);
         }
      }
   }
   want.insert(want.end(), {0x0e, 0, 0, 0});
   const auto got = pack(b);
   if (got != want) {
      size_t first = 0;
      while (first < got.size() && first < want.size() &&
             got[first] == want[first])
         ++first;
      ADD_FAILURE() << "first reordered-output byte mismatch at " << first
                    << ", got size " << got.size() << ", want " << want.size();
   }
}

TEST_F(AGX3PackBinary, GradientRetiresAsyncSourceBeforeNativeDfdx)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_ldcf_to(b, agx_register(8, AGX_SIZE_32), agx_immediate(0), 1);
   agx_index source = agx_register(8, AGX_SIZE_32);
   source.cache = true;
   agx_dfdx_to(b, agx_register(2, AGX_SIZE_32), source);
   agx_stop(b);

   std::vector<uint8_t> want;
   uint8_t bytes[AGX3_VARY_MAX_LENGTH], len = 0;
   /* Apple's derivative fragments leave the shared f48/f63 input marker
    * clear so helper lanes survive into the quad operation. */
   ASSERT_TRUE(agx3_vary_ldcf_at_start(enc, 8, 0, 1, false, bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_alu_compose(enc, AGX3_ALU_BAND, 0, 0, 0, 0, 0, 1,
                                bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_funop_encode(enc, AGX3_FUNOP_DFDX, 2, 8, false, false,
                                 false, 0, bytes, &len));
   append(want, bytes, len);
   want.insert(want.end(), {0x0e, 0, 0, 0});
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, ProjectiveRhwRetiresReusedCoefficientDestination)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_ldcf_to(b, agx_register(8, AGX_SIZE_32), agx_immediate(0), 1);
   agx_index rhw = agx_register(8, AGX_SIZE_32);
   agx_rhw_interp_agx3_to(b, rhw, agx_immediate(0), agx_immediate(0),
                         AGX_INTERPOLATION_CENTER);
   agx_persp_rcp_agx3_to(b, rhw, rhw, 0);
   agx_stop(b);

   std::vector<uint8_t> want;
   uint8_t bytes[AGX3_VARY_MAX_LENGTH], len = 0;
   ASSERT_TRUE(agx3_vary_ldcf_at_start(enc, 8, 0, 1, true, bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_alu_compose(enc, AGX3_ALU_BAND, 0, 0, 0, 0, 0, 1,
                                bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_vary_projective_rhw(enc, 8, 0, false, bytes, &len));
   append(want, bytes, len);
   struct agx3_persp_rcp_desc rcp = {
      .dest = 8, .rhw = 8, .coef_index = 0,
   };
   ASSERT_TRUE(agx3_persp_rcp_encode(enc, &rcp, bytes, &len));
   append(want, bytes, len);
   want.insert(want.end(), {0x0e, 0, 0, 0});
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, LinearInterpRetiresReusedCoefficientDestination)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_ldcf_to(b, agx_register(8, AGX_SIZE_32), agx_immediate(13), 1);
   agx_interp_agx3_to(b, agx_register(8, AGX_SIZE_32), agx_immediate(4),
                      agx_immediate(0), AGX_INTERPOLATION_CENTER, true);
   agx_stop(b);

   std::vector<uint8_t> want;
   uint8_t bytes[AGX3_VARY_MAX_LENGTH], len = 0;
   ASSERT_TRUE(agx3_vary_ldcf_at_start(enc, 8, 13, 1, true, bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_alu_compose(enc, AGX3_ALU_BAND, 0, 0, 0, 0, 0, 1,
                                bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_vary_interp_at_start(
      enc, 8, 4, true, AGX3_VARY_CENTER, false, bytes, &len));
   append(want, bytes, len);
   want.insert(want.end(), {0x0e, 0, 0, 0});
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, FragmentHelperCoverageMatchesNativeAtomicPipeline)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_get_sr_coverage_to(b, agx_register(0, AGX_SIZE_16),
                          AGX_SR_IS_ACTIVE_THREAD);
   agx_stop(b);
   const std::vector<uint8_t> native = {
      0x04,0x84,0x10,0x06, 0x0e,0x00,0x00,0x00,
   };
   EXPECT_EQ(pack(b), native);
}

TEST_F(AGX3PackBinary, FragmentSideEffectLateDepthMatchesNativeProtocol)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_wait_pix(b, 1);
   agx_index depth = agx_register(4, AGX_SIZE_32);
   depth.discard = true;
   agx_zs_emit(b, agx_immediate(0xff), depth, 1);
   agx_signal_pix(b, 1);
   agx_stop(b);

   /* Fresh Apple fragment-atomic tail from the same d150404a... AIR /
    * 32cf4963... native pipeline: entry, stage Z, acquire class1, depthf,
    * release. Construct every instruction independently, then compare the
    * complete backend stream. */
   std::vector<uint8_t> want;
   uint8_t bytes[AGX3_STAGE_MAX_LENGTH], len = 0;
   ASSERT_TRUE(agx3_pixel_colour_entry(enc, bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_stage_copy_kill(enc, 0, 4, true, 1, 0, bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_pixel_wait(enc, 1, 0, bytes, &len));
   append(want, bytes, len);
   const struct agx3_zs_desc zs = {
      .value = 0, .value_kill = true, .slot = 1, .wait_mask = 1,
   };
   ASSERT_TRUE(agx3_zs_encode_depth_stencil(enc, &zs, true, false, bytes,
                                            &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_pixel_release(enc, 1, false, 0, bytes, &len));
   append(want, bytes, len);
   want.insert(want.end(), {0x0e, 0, 0, 0});
   EXPECT_EQ(pack(b), want);
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

/* SIGNEXT lowers this exact Firefox vertex-shader shape to IADD after RA.
 * A bare half is signed in AGX IR, while ABS denotes zero extension. */
TEST_F(AGX3PackBinary, FirefoxVertexHalfSignExtendUsesNativeIadd)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index dest = agx_register(34, AGX_SIZE_32);
   dest.cache = true;
   agx_iadd_to(b, dest, agx_register(26, AGX_SIZE_16), agx_immediate(0), 0);
   agx_stop(b);

   const struct agx3_iadd_desc d = {
      .dest = 34,
      .immediate = 0,
      .src = 26,
      .src_16 = true,
      .sign_extend = true,
      .dest_cache = true,
   };
   uint8_t bytes[AGX3_IADD_MAX_LENGTH], len = 0;
   ASSERT_TRUE(agx3_iadd_encode(AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
   std::vector<uint8_t> want(bytes, bytes + len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, FirefoxFragmentSignedHalfAccumulatorUsesNativeIadd)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index half = agx_register(22, AGX_SIZE_16);
   half.cache = true;
   agx_iadd_to(b, agx_register(30, AGX_SIZE_32),
               agx_register(30, AGX_SIZE_32), half, 0);
   agx_stop(b);

   struct agx3_iadd_desc d = {};
   d.src_a_is_register = true;
   d.src_a_reg = 30;
   d.dest = 30;
   d.src = 22;
   d.src_16 = true;
   d.sign_extend = true;
   uint8_t bytes[AGX3_IADD_MAX_LENGTH], len = 0;
   ASSERT_TRUE(agx3_iadd_encode(AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
   std::vector<uint8_t> want(bytes, bytes + len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);

   /* Addition commutes: reverse Firefox's second physical allocation onto
    * the same grounded slot-A32 + slot-B16 instruction. */
   agx_builder *bad = builder(AGX_ISA_ENCODING_AGX3B);
   agx_iadd_to(bad, agx_register(30, AGX_SIZE_32),
               agx_register(22, AGX_SIZE_16),
               agx_register(30, AGX_SIZE_32), 0);
   agx_stop(bad);
   EXPECT_EQ(pack(bad), want);

   bad = builder(AGX_ISA_ENCODING_AGX3B);
   agx_iadd_to(bad, agx_register(30, AGX_SIZE_32),
               agx_register(30, AGX_SIZE_32),
               agx_register(22, AGX_SIZE_16), 1);
   agx_stop(bad);
   EXPECT_TRUE(pack(bad, false).empty());

   bad = builder(AGX_ISA_ENCODING_AGX3B);
   agx_instr *sat = agx_iadd_to(bad, agx_register(30, AGX_SIZE_32),
                                agx_register(30, AGX_SIZE_32),
                                agx_register(22, AGX_SIZE_16), 0);
   sat->saturate = true;
   agx_stop(bad);
   EXPECT_TRUE(pack(bad, false).empty());
}

TEST_F(AGX3PackBinary, FirefoxHalfArithmeticShiftMatchesNative)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index src = agx_register(35, AGX_SIZE_16);
   src.cache = true;
   agx_index amount = agx_immediate(1);
   agx_asr_to(b, agx_register(25, AGX_SIZE_16), src, amount);
   agx_stop(b);

   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
   ASSERT_TRUE(agx3_alu_asr_imm16(
      AGX_ISA_ENCODING_AGX3B, 25, 35, 1, AGX3_ALU_HINT_CACHE, false, 0,
      bytes, &len));
   std::vector<uint8_t> want(bytes, bytes + len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);

   for (unsigned variant = 0; variant < 4; ++variant) {
      agx_builder *bad = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index bad_src = variant == 0 ? agx_register(34, AGX_SIZE_32)
                                       : agx_register(35, AGX_SIZE_16);
      agx_index bad_amount = agx_immediate(variant == 1 ? 16 : 1);
      if (variant == 2)
         bad_src.abs = true;
      agx_instr *bad_I = agx_asr_to(
         bad, agx_register(25, AGX_SIZE_16), bad_src, bad_amount);
      bad_I->saturate = variant == 3;
      agx_stop(bad);
      EXPECT_TRUE(pack(bad, false).empty()) << variant;
   }
}

/*
 * agx_lower_uniform_sources() materializes a move for every generic-ALU
 * uniform source. At 32 bits it reached agx3_copy_encode_uniform_to_reg32();
 * at 16 bits it had no encoder, so it stayed a bitop reading the uniform file
 * and the packer refused the whole shader. u_blitter's pass-through vertex
 * shader hits exactly that (`mov u9h` for the clip-Z coefficient), so before
 * this every native-Gallium blit on G17P refused its vertex program.
 */
TEST_F(AGX3PackBinary, G17PUniformHalfMoveUsesNativeCopy)
{
   for (unsigned uniform : {0u, 19u, 511u}) {
      for (bool cache : {false, true}) {
         agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
         agx_index dest = agx_register(2, AGX_SIZE_16);
         dest.cache = cache;
         /* agx_lower_pseudo() turns a MOV into exactly this bitop. */
         agx_bitop_to(b, dest, agx_uniform(uniform, AGX_SIZE_16), agx_zero(),
                      AGX3_SEL_TT_MOV);
         agx_stop(b);

         uint8_t bytes[AGX3_COPY_MAX_LENGTH], len = 0;
         std::vector<uint8_t> want;
         ASSERT_TRUE(agx3_copy_encode_uniform_to_reg16(
            AGX_ISA_ENCODING_AGX3B, 2, uniform, cache, bytes, &len));
         append(want, bytes, len);
         struct agx3_encoded_end end;
         ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
         append(want, end.bytes, end.length);
         EXPECT_EQ(pack(b), want)
            << "uniform half=" << uniform << " cache=" << cache;
      }
   }
}

TEST_F(AGX3PackBinary, G17PPaddedUniformHalfStoreUsesNativeDwordCopy)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index value = agx_register(4, AGX_SIZE_16);
   value.discard = true;
   agx_uniform_store(b, value, agx_immediate(24), 1);
   agx_stop(b);

   uint8_t bytes[AGX3_COPY_MAX_LENGTH], len = 0;
   std::vector<uint8_t> want;
   ASSERT_TRUE(agx3_copy_encode_reg_to_uniform(
      AGX_ISA_ENCODING_AGX3B, 24, 4, false, bytes, &len));
   append(want, bytes, len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, G17PUniformHalfStoreRequiresUnusedAlignedPadding)
{
   /* Adjacent half stores are real data, not a dword with free padding. */
   agx_builder *adjacent = builder(AGX_ISA_ENCODING_AGX3B);
   agx_uniform_store(adjacent, agx_register(4, AGX_SIZE_16),
                     agx_immediate(24), 1);
   agx_uniform_store(adjacent, agx_register(6, AGX_SIZE_16),
                     agx_immediate(25), 1);
   agx_stop(adjacent);
   pack(adjacent, false);

   agx_builder *odd_source = builder(AGX_ISA_ENCODING_AGX3B);
   agx_uniform_store(odd_source, agx_register(5, AGX_SIZE_16),
                     agx_immediate(24), 1);
   agx_stop(odd_source);
   pack(odd_source, false);
}

TEST_F(AGX3PackBinary, G17PCoalescesAdjacentPreambleUniformHalvesBeforeRA)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->is_preamble = true;
   b->shader->allocated_vec = _mesa_hash_table_u64_create(b->shader);
   agx_uniform_store(b, agx_register(4, AGX_SIZE_16), agx_immediate(32), 1);
   agx_uniform_store(b, agx_register(6, AGX_SIZE_16), agx_immediate(33), 1);
   agx_uniform_store(b, agx_register(8, AGX_SIZE_16), agx_immediate(34), 1);
   agx_stop(b);

   ASSERT_TRUE(agx3_coalesce_uniform_half_stores(b->shader));
   unsigned collects = 0, stores32 = 0, stores16 = 0;
   agx_foreach_instr_global(b->shader, I) {
      if (I->op == AGX_OPCODE_COLLECT) {
         ++collects;
         ASSERT_EQ(I->nr_srcs, 2);
         EXPECT_EQ(I->src[0].value, 4u);
         EXPECT_EQ(I->src[1].value, 6u);
      } else if (I->op == AGX_OPCODE_UNIFORM_STORE) {
         ASSERT_EQ(I->src[1].type, AGX_INDEX_IMMEDIATE);
         if (I->src[0].size == AGX_SIZE_32) {
            ++stores32;
            EXPECT_EQ(I->src[1].value, 32u);
         } else {
            ++stores16;
            EXPECT_EQ(I->src[1].value, 34u);
         }
      }
   }
   EXPECT_EQ(collects, 1u);
   EXPECT_EQ(stores32, 1u);
   EXPECT_EQ(stores16, 1u);
}

TEST_F(AGX3PackBinary, G17PDoesNotCoalesceOddUniformPair)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->is_preamble = true;
   agx_uniform_store(b, agx_register(4, AGX_SIZE_16), agx_immediate(33), 1);
   agx_uniform_store(b, agx_register(6, AGX_SIZE_16), agx_immediate(34), 1);
   agx_stop(b);
   EXPECT_FALSE(agx3_coalesce_uniform_half_stores(b->shader));
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

TEST_F(AGX3PackBinary, VertexTerminalOutputsCrossDirectEmptyFallthrough)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_VERTEX;
   b->shader->nir = rzalloc(mem_ctx, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_VERTEX;

   agx_block *stores_block = agx_start_block(b->shader);
   agx_index values[12];
   values[0] = agx_temp(b->shader, AGX_SIZE_32);
   values[1] = agx_temp(b->shader, AGX_SIZE_32);
   agx_get_sr_to(b, values[0], AGX_SR_VERTEX_ID);
   agx_get_sr_to(b, values[1], AGX_SR_INSTANCE_ID);
   for (unsigned c = 2; c < ARRAY_SIZE(values); ++c)
      values[c] = agx_mov_imm(b, 32, 0x3f000000 + c);

   /* Exact ordering from the live Settings/Zink shader. */
   for (unsigned c : {8u, 9u, 10u, 11u, 4u, 5u, 6u, 7u, 0u, 1u, 2u, 3u})
      agx_st_vary(b, agx_immediate(c), values[c]);

   agx_block *stop_block = agx_test_block(b->shader);
   agx_block_add_successor(stores_block, stop_block);
   b->cursor = agx_after_block(stop_block);
   agx_stop(b);

   agx_pressure_schedule(b->shader);
   ASSERT_TRUE(agx3_sink_vertex_outputs(b->shader));
   agx_foreach_instr_in_block(stores_block, I)
      EXPECT_NE(I->op, AGX_OPCODE_ST_VARY);

   unsigned next = 0;
   agx_foreach_instr_in_block(stop_block, I) {
      if (I->op == AGX_OPCODE_ST_VARY) {
         ASSERT_LT(next, ARRAY_SIZE(values));
         EXPECT_EQ(I->src[0].value, next++);
      } else {
         EXPECT_EQ(I->op, AGX_OPCODE_STOP);
         EXPECT_EQ(next, ARRAY_SIZE(values));
      }
   }
   EXPECT_EQ(next, ARRAY_SIZE(values));

   agx_ra(b->shader);
   EXPECT_FALSE(pack(b).empty());
}

TEST_F(AGX3PackBinary, VertexInstanceIdWithWideSingleBlockOutputs)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_VERTEX;
   b->shader->nir = rzalloc(mem_ctx, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_VERTEX;

   agx_index instance = agx_temp(b->shader, AGX_SIZE_32);
   agx_index vertex = agx_temp(b->shader, AGX_SIZE_32);
   agx_get_sr_to(b, instance, AGX_SR_INSTANCE_ID);
   agx_get_sr_to(b, vertex, AGX_SR_VERTEX_ID);
   for (unsigned c : {8u, 9u, 10u, 11u, 4u, 5u, 6u, 7u, 0u, 1u, 2u, 3u})
      agx_st_vary(b, agx_immediate(c), c & 1 ? instance : vertex);
   agx_stop(b);

   agx_pressure_schedule(b->shader);
   ASSERT_TRUE(agx3_sink_vertex_outputs(b->shader));
   agx_ra(b->shader);
   EXPECT_FALSE(pack(b).empty());
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
      for (bool dead : {false, true}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index source = agx_register(2, AGX_SIZE_32);
      source.kill = source.discard = dead;
      agx_index zero = agx_immediate(0x80);
      agx_fadd_to(b, agx_register(1, AGX_SIZE_16),
                  swapped ? zero : source, swapped ? source : zero);
      agx_stop(b);
      auto code = pack(b);
      /* One 16-bit rounding ADD. A post-RA kill is representable exactly
       * when DISCARD owns the same final use, as in Gallium's readback CS. */
      const uint8_t expected[2][6] = {
         {0x01,0x03,0x14,0x81,0x04,0x02},
         {0x01,0x03,0x1c,0x81,0x04,0x02},
      };
      ASSERT_EQ(code.size(),10);
      EXPECT_EQ(memcmp(code.data(),expected[dead],6),0);
      }
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

/*
 * Apple's own applegpu_g17p bytes for the 32-bit tied minimum/maximum.  Before
 * this form existed the packer refused every shader containing max(x, 0.0),
 * which is normalize(), clamp(), saturate() and the discard compare.
 *
 *   fmax(a, b)      12 03 1e 05 00 c0
 *   fmin(a, b)      12 03 1e 05 01 c0
 *   fmax(a, 0.0f)   12 03 0a 81 00 c2
 *   fmax(a, 1.0f)   12 03 0a b1 00 c2
 */
/* Diagnostic: enumerate the fcmpsel operand shape space and print refusals. */
TEST_F(AGX3PackBinary, DISABLED_FcmpselShapeSurvey)
{
   static const char *kindname[] = {"h16", "r32", "#0", "#255"};
   static const unsigned fconds[] = {0, 1, 2, 3, 5, 6, 7};
   static const char *fcondname[] = {"eq", "lt", "gt", "ltn", "?", "ge",
                                     "le", "gtn"};
   unsigned refused = 0, total = 0;

   for (unsigned dest_32 = 0; dest_32 < 2; ++dest_32) {
   for (unsigned k0 = 0; k0 < 4; ++k0)
   for (unsigned k1 = 0; k1 < 4; ++k1)
   for (unsigned k2 = 0; k2 < 4; ++k2)
   for (unsigned k3 = 0; k3 < 4; ++k3)
   for (unsigned ci = 0; ci < ARRAY_SIZE(fconds); ++ci)
   for (unsigned inv = 0; inv < 2; ++inv) {
      const unsigned k[4] = {k0, k1, k2, k3};
      /* A select needs at least one register operand somewhere. */
      if (k0 >= 2 && k1 >= 2)
         continue;

      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index src[4];
      for (unsigned i = 0; i < 4; ++i) {
         switch (k[i]) {
         case 0: src[i] = agx_register(2 + 2 * i, AGX_SIZE_16); break;
         case 1: src[i] = agx_register(2 + 2 * i, AGX_SIZE_32); break;
         case 2: src[i] = agx_immediate(0); break;
         default: src[i] = agx_immediate(255); break;
         }
      }
      /* Sources 2 and 3 are selected into the destination, so their width has
       * to be the destination's. */
      if ((k2 < 2 && (k2 == 1) != (dest_32 != 0)) ||
          (k3 < 2 && (k3 == 1) != (dest_32 != 0)))
         continue;

      agx_index dest =
         agx_register(0, dest_32 ? AGX_SIZE_32 : AGX_SIZE_16);
      agx_instr *I = agx_fcmpsel_to(b, dest, src[0], src[1], src[2], src[3],
                                    (enum agx_fcond)fconds[ci]);
      I->invert_cond = inv;
      agx_stop(b);

      struct util_dynarray bin;
      util_dynarray_init(&bin, mem_ctx);
      ++total;
      if (!agx3_pack_binary(b->shader, &bin)) {
         ++refused;
         fprintf(stderr, "REFUSED d%s = fcmpsel %s, %s, %s, %s, %s%s\n",
                 dest_32 ? "32" : "16", kindname[k0], kindname[k1],
                 kindname[k2], kindname[k3], fcondname[fconds[ci]],
                 inv ? " (inverted)" : "");
      }
   }
   }
   fprintf(stderr, "fcmpsel survey: %u shapes, %u refused\n", total, refused);
}

TEST_F(AGX3PackBinary, F32TiedMaximumMatchesNativeRegisterForm)
{
   for (bool maximum : {true, false}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index a = agx_register(2, AGX_SIZE_32);
      agx_index c = agx_register(4, AGX_SIZE_32);
      agx_index dest = agx_register(2, AGX_SIZE_32);
      a.discard = c.discard = true;
      agx_fcmpsel_to(b, dest, a, c, a, c,
                     maximum ? AGX_FCOND_GTN : AGX_FCOND_LTN);
      agx_stop(b);
      const auto code = pack(b);
      const uint8_t native[] = {
         0x12, 0x03, 0x1e, 0x05, (uint8_t)(maximum ? 0x00 : 0x01), 0x00,
      };
      ASSERT_EQ(code.size(), 10u);
      EXPECT_EQ(memcmp(code.data(), native, 6), 0);
   }
}

TEST_F(AGX3PackBinary, F32TiedMaximumMatchesNativeMinifloatForm)
{
   static const struct {
      float value;
      uint8_t byte3, byte4;
   } cases[] = {
      {0.0f, 0x81, 0x00}, {0.5f, 0xa1, 0x00}, {1.0f, 0xb1, 0x00},
      {2.0f, 0xc1, 0x00}, {-1.0f, 0xb1, 0x00}, {31.0f, 0xff, 0x10},
   };

   for (unsigned i = 0; i < ARRAY_SIZE(cases); ++i) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index a = agx_register(2, AGX_SIZE_32);
      agx_index dest = agx_register(2, AGX_SIZE_32);
      a.discard = true;
      const agx_index k = agx_immediate_f(cases[i].value);
      agx_fcmpsel_to(b, dest, a, k, a, k, AGX_FCOND_GTN);
      agx_stop(b);
      const auto code = pack(b);
      const uint8_t native[] = {
         0x12, 0x03, (uint8_t)(cases[i].value < 0.0f ? 0x1a : 0x0a),
         cases[i].byte3, cases[i].byte4, 0x02,
      };
      ASSERT_EQ(code.size(), 10u);
      EXPECT_EQ(memcmp(code.data(), native, 6), 0);
   }
}

/*
 * max(0.0, x) is max(x, 0.0). NIR declares fmin_agx and fmax_agx
 * 2src_commutative, and a tied GTN/LTN fcmpsel can only have come from one of
 * them, so the constant may be written on either side and both encode to the
 * bytes Apple emits for the constant-second form.
 */
TEST_F(AGX3PackBinary, F32TiedMaximumIsCommutative)
{
   for (bool constant_first : {false, true}) {
      for (bool maximum : {true, false}) {
         agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
         agx_index a = agx_register(2, AGX_SIZE_32);
         agx_index dest = agx_register(2, AGX_SIZE_32);
         a.discard = true;
         const agx_index k = agx_immediate_f(0.0f);
         const agx_index s0 = constant_first ? k : a;
         const agx_index s1 = constant_first ? a : k;
         agx_fcmpsel_to(b, dest, s0, s1, s0, s1,
                        maximum ? AGX_FCOND_GTN : AGX_FCOND_LTN);
         agx_stop(b);
         const auto code = pack(b);
         const uint8_t native[] = {
            0x12, 0x03, 0x0a, 0x81, (uint8_t)(maximum ? 0x00 : 0x01), 0x02,
         };
         ASSERT_EQ(code.size(), 10u) << "constant_first=" << constant_first;
         EXPECT_EQ(memcmp(code.data(), native, 6), 0)
            << "constant_first=" << constant_first;
      }
   }

   /* The same for two registers: the tied pair may be named either way. */
   for (bool swapped : {false, true}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index a = agx_register(2, AGX_SIZE_32);
      agx_index c = agx_register(4, AGX_SIZE_32);
      agx_index dest = agx_register(2, AGX_SIZE_32);
      a.discard = c.discard = true;
      const agx_index s0 = swapped ? c : a;
      const agx_index s1 = swapped ? a : c;
      agx_fcmpsel_to(b, dest, s0, s1, s0, s1, AGX_FCOND_GTN);
      agx_stop(b);
      const auto code = pack(b);
      ASSERT_EQ(code.size(), 10u);
      /* Both operands are registers, so nothing is normalized: the two
       * operand fields simply exchange. */
      const uint8_t native[] = {
         0x12, (uint8_t)(swapped ? 0x05 : 0x03), 0x1e,
         (uint8_t)(swapped ? 0x03 : 0x05), 0x00, 0x00,
      };
      EXPECT_EQ(memcmp(code.data(), native, 6), 0) << "swapped=" << swapped;
   }
}

/*
 * A half select whose selected operands are byte constants is not the half
 * fmin/fmax form and must fall through to the byte-constant encoder, not fail
 * the shader.  This is the shape a `discard` compare produces.
 */
TEST_F(AGX3PackBinary, HalfSelectOfByteConstantsIsNotRefusedByTheHalfForms)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index a = agx_register(4, AGX_SIZE_32);
   agx_index c = agx_register(2, AGX_SIZE_32);
   agx_index dest = agx_register(0, AGX_SIZE_16);
   a.discard = c.discard = true;
   agx_fcmpsel_to(b, dest, a, c, agx_immediate(255), agx_immediate(0),
                  AGX_FCOND_LT);
   agx_stop(b);
   const auto code = pack(b);
   EXPECT_GT(code.size(), 4u);
}

TEST_F(AGX3PackBinary, HalfCompareSelectMatchesNativeAndRefusesModifier)
{
   for (bool live : {false, true}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index a = agx_register(live ? 2 : 0, AGX_SIZE_16);
      agx_index c = agx_register(live ? 3 : 1, AGX_SIZE_16);
      agx_index if_true = agx_register(live ? 4 : 2, AGX_SIZE_16);
      agx_index if_false = agx_register(live ? 5 : 3, AGX_SIZE_16);
      agx_index dest = agx_register(0, AGX_SIZE_16);
      if (live) {
         a.cache = c.cache = if_true.cache = dest.cache = true;
         if_false.discard = true;
      } else {
         a.discard = c.discard = if_true.discard = if_false.discard = true;
      }
      agx_fcmpsel_to(b, dest, a, c, if_true, if_false, AGX_FCOND_LT);
      agx_stop(b);
      const auto code = pack(b);
      const uint8_t retained[] = {
         0x0a,0x82,0x25,0x82,0x02,0x84,0x13,0xc0,0x80,0x05,
      };
      const uint8_t dead[] = {
         0x02,0x00,0x1d,0x00,0x82,0x02,0x13,0xc0,0x80,0x03,
      };
      const uint8_t *native = live ? retained : dead;
      ASSERT_EQ(code.size(), 14u);
      EXPECT_EQ(memcmp(code.data(), native, 10), 0);
   }

   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index modified = agx_register(0, AGX_SIZE_16);
   modified.abs = true;
   agx_fcmpsel_to(b, agx_register(0, AGX_SIZE_16), modified,
                  agx_register(1, AGX_SIZE_16),
                  agx_register(2, AGX_SIZE_16),
                  agx_register(3, AGX_SIZE_16), AGX_FCOND_LT);
   agx_stop(b);
   EXPECT_TRUE(pack(b, false).empty());
}

TEST_F(AGX3PackBinary, HalfSqrtMatchesNativeSequenceAndRefusesModifier)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index half = agx_register(2, AGX_SIZE_16);
   half.cache = true;
   agx_index step = agx_register(4, AGX_SIZE_32);
   step.cache = true;
   agx_srsqrt_to(b, step, half);

   step.cache = false;
   step.discard = true;
   half.cache = false;
   half.discard = true;
   agx_index dest = agx_register(2, AGX_SIZE_16);
   dest.cache = true;
   agx_fmul_to(b, dest, step, half);
   agx_stop(b);

   const uint8_t native[] = {
      0x2f,0x01,0x54,0x04,0x03,0x04,0x8a,0x40,0x00,0x00,
      0x11,0x05,0x39,0x02,
   };
   const auto code = pack(b);
   ASSERT_EQ(code.size(), sizeof(native) + 4);
   EXPECT_EQ(memcmp(code.data(), native, sizeof(native)), 0);

   b = builder(AGX_ISA_ENCODING_AGX3B);
   half = agx_register(2, AGX_SIZE_16);
   half.cache = true;
   half.abs = true;
   step = agx_register(4, AGX_SIZE_32);
   step.cache = true;
   agx_srsqrt_to(b, step, half);
   agx_stop(b);
   EXPECT_TRUE(pack(b, false).empty());
}

TEST_F(AGX3PackBinary, MixedHalfMultiplyRetainsUncachedHalf)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index f32 = agx_register(16, AGX_SIZE_32);
   agx_index f16 = agx_register(13, AGX_SIZE_16);
   f32.discard = true;
   f16.cache = true;
   agx_fmul_to(b, agx_register(15, AGX_SIZE_16), f32, f16);
   agx_stop(b);

   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   ASSERT_TRUE(agx3_alu_fmul_mixed_half(
      AGX_ISA_ENCODING_AGX3B, 15, 16, 13, AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_NONE, false, 0, bytes, &len));
   std::vector<uint8_t> want;
   append(want, bytes, len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);

   EXPECT_EQ(pack(b), want);
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

TEST_F(AGX3PackBinary, GtkMixedReciprocalMatchesNative)
{
   struct width_case {
      enum agx_size dest_size, src_size;
      uint16_t dest, src;
      const uint8_t bytes[10];
   };
   const width_case widths[] = {
      {AGX_SIZE_16, AGX_SIZE_16, 2, 2,
       {0xaf, 0x00, 0x54, 0x02, 0x02, 0x04, 0x10, 0x46, 0x30, 0x00}},
      {AGX_SIZE_32, AGX_SIZE_16, 0, 0,
       {0xaf, 0x00, 0x54, 0x00, 0x02, 0x00, 0x10, 0x44, 0x20, 0x00}},
      {AGX_SIZE_16, AGX_SIZE_32, 2, 2,
       {0xaf, 0x00, 0x54, 0x02, 0x02, 0x04, 0x10, 0x4a, 0x30, 0x00}},
      {AGX_SIZE_32, AGX_SIZE_32, 0, 0,
       {0xaf, 0x00, 0x54, 0x00, 0x02, 0x00, 0x10, 0x48, 0x20, 0x00}},
   };
   for (const width_case &v : widths) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index src = agx_register(v.src, v.src_size);
      src.discard = true;
      agx_rcp_to(b, agx_register(v.dest, v.dest_size), src);
      agx_stop(b);
      std::vector<uint8_t> native(v.bytes, v.bytes + sizeof(v.bytes));
      native.insert(native.end(), {0x0e, 0x00, 0x00, 0x00});
      EXPECT_EQ(pack(b), native);
   }

   /* Exact post-RA blocker from GTK gskgpuconvertbuiltin: a cached odd half
    * texture channel feeds a cached full-precision reciprocal result. */
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index dest = agx_register(6, AGX_SIZE_32);
   agx_index src = agx_register(3, AGX_SIZE_16);
   dest.cache = src.cache = true;
   agx_rcp_to(b, dest, src);
   agx_stop(b);
   const std::vector<uint8_t> expected = {
      0xaf, 0x00, 0x54, 0x06, 0x03, 0x06, 0x02, 0x44, 0x20, 0x00,
      0x0e, 0x00, 0x00, 0x00,
   };
   EXPECT_EQ(pack(b), expected);

   /* Source ABS and NEG are native fields rather than legalization moves. */
   for (unsigned modifiers = 0; modifiers < 4; ++modifiers) {
      b = builder(AGX_ISA_ENCODING_AGX3B);
      dest = agx_register(0, AGX_SIZE_32);
      src = agx_register(0, AGX_SIZE_16);
      src.discard = true;
      src.abs = modifiers & 1;
      src.neg = modifiers & 2;
      agx_rcp_to(b, dest, src);
      agx_stop(b);
      const std::vector<uint8_t> native = {
         0xaf, 0x00, 0x54, 0x00, 0x02, 0x00, 0x10, 0x44,
         (uint8_t)(0x20 | (modifiers << 6)), 0x00,
         0x0e, 0x00, 0x00, 0x00,
      };
      EXPECT_EQ(pack(b), native) << "modifiers=" << modifiers;
   }
}

TEST_F(AGX3PackBinary, ReciprocalWidthsRefuseUngroundedModes)
{
   for (unsigned mode = 0; mode < 18; ++mode) {
      agx_builder *b = builder(mode == 0 ? AGX_ISA_ENCODING_AGX3C
                                         : AGX_ISA_ENCODING_AGX3B);
      agx_index dest = agx_register(0, AGX_SIZE_32);
      agx_index src = agx_register(0, AGX_SIZE_16);
      src.discard = true;

      if (mode == 1)
         dest.size = AGX_SIZE_64;
      if (mode == 2)
         src.size = AGX_SIZE_64;
      if (mode == 3)
         dest.memory = true;
      if (mode == 4)
         src.memory = true;
      if (mode == 5)
         src.kill = true;
      if (mode == 6)
         src.cache = true;
      if (mode == 7)
         dest.value = 1;
      if (mode == 8) {
         src.size = AGX_SIZE_32;
         src.value = 1;
      }
      if (mode == 9)
         dest.channels_m1 = 1;
      if (mode == 10)
         src.channels_m1 = 1;
      if (mode == 11)
         src = agx_immediate(0);
      if (mode == 12)
         src = agx_uniform(0, AGX_SIZE_16);
      /* 192, not 128: f2 and f18 of f32rcp ep0 accept [0, 191] in Apple's own
       * composer tables, with 192 distinct encodings and no aliasing, and the
       * packer's gate now agrees with them rather than refusing sixty-four
       * addressable registers. */
      if (mode == 14) {
         dest.size = AGX_SIZE_16;
         dest.value = 192;
      }
      if (mode == 15)
         src.value = 192;
      if (mode == 16)
         dest.value = 192;
      if (mode == 17) {
         src.size = AGX_SIZE_32;
         src.value = 192;
      }

      agx_instr *rcp = agx_rcp_to(b, dest, src);
      rcp->saturate = mode == 13;
      agx_stop(b);
      EXPECT_TRUE(pack(b, false).empty()) << "mode=" << mode;
   }
}

/* The gap the old gate left: a reciprocal whose destination is a register the
 * form can address but the packer refused.  Three of the ten distinct programs
 * three.js compiles for webgl_geometry_cube, webgl_animation_keyframes and
 * webgl_materials_physical_clearcoat died on this shape, and hk_shader.c turns
 * a packer refusal into VK_ERROR_FEATURE_NOT_PRESENT for the whole pipeline. */
TEST_F(AGX3PackBinary, ReciprocalEncodesAboveTheOldRegisterGate)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index dest = agx_register(140, AGX_SIZE_32);
   agx_index src = agx_register(86, AGX_SIZE_32);
   src.abs = true;
   src.cache = true;
   agx_rcp_to(b, dest, src);
   agx_stop(b);
   EXPECT_FALSE(pack(b).empty());
}

TEST_F(AGX3PackBinary, HalfReciprocalSqrtMatchesNativeAndRefusesModifier)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index src = agx_register(3, AGX_SIZE_16);
   src.discard = true;
   agx_rsqrt_to(b, agx_register(3, AGX_SIZE_16), src);
   agx_stop(b);
   const auto code = pack(b);
   const uint8_t native[] = {
      0xaf,0x01,0x54,0x03,0x02,0x06,0xac,0x60,0x00,0x00,
   };
   ASSERT_EQ(code.size(), sizeof(native) + 4);
   EXPECT_EQ(memcmp(code.data(), native, sizeof(native)), 0);

   b = builder(AGX_ISA_ENCODING_AGX3B);
   src = agx_register(2, AGX_SIZE_16);
   src.neg = true;
   agx_rsqrt_to(b, agx_register(2, AGX_SIZE_16), src);
   agx_stop(b);
   EXPECT_TRUE(pack(b, false).empty());
}

TEST_F(AGX3PackBinary, HalfExp2MatchesNativeNegateAndRefusesAbsolute)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index src = agx_register(3, AGX_SIZE_16);
   src.discard = true;
   agx_exp2_to(b, agx_register(3, AGX_SIZE_16), src);
   agx_stop(b);
   const auto code = pack(b);
   const uint8_t native[] = {
      0xaf,0x02,0x54,0x03,0x02,0x06,0xac,0x60,0x00,0x00,
   };
   ASSERT_EQ(code.size(), sizeof(native) + 4);
   EXPECT_EQ(memcmp(code.data(), native, sizeof(native)), 0);

   b = builder(AGX_ISA_ENCODING_AGX3B);
   src = agx_register(2, AGX_SIZE_16);
   src.neg = true;
   src.discard = true;
   agx_exp2_to(b, agx_register(2, AGX_SIZE_16), src);
   agx_stop(b);
   const uint8_t native_neg[] = {
      0xaf,0x02,0x54,0x02,0x02,0x04,0xac,0x60,0x01,0x00,
   };
   const auto neg_code = pack(b);
   ASSERT_EQ(neg_code.size(), sizeof(native_neg) + 4);
   EXPECT_EQ(memcmp(neg_code.data(), native_neg, sizeof(native_neg)), 0);

   b = builder(AGX_ISA_ENCODING_AGX3B);
   src = agx_register(2, AGX_SIZE_16);
   src.abs = true;
   agx_exp2_to(b, agx_register(2, AGX_SIZE_16), src);
   agx_stop(b);
   EXPECT_TRUE(pack(b, false).empty());
}

TEST_F(AGX3PackBinary, HalfLog2MatchesNativeAndModifiers)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index src = agx_register(3, AGX_SIZE_16);
   src.discard = true;
   agx_log2_to(b, agx_register(3, AGX_SIZE_16), src);
   agx_stop(b);
   const auto code = pack(b);
   const uint8_t native[] = {
      0x2f,0x02,0x54,0x03,0x02,0x06,0xac,0x60,0x00,0x00,
   };
   ASSERT_EQ(code.size(), sizeof(native) + 4);
   EXPECT_EQ(memcmp(code.data(), native, sizeof(native)), 0);

   b = builder(AGX_ISA_ENCODING_AGX3B);
   src = agx_register(2, AGX_SIZE_16);
   src.neg = true;
   agx_log2_to(b, agx_register(2, AGX_SIZE_16), src);
   agx_stop(b);
   const uint8_t native_neg[] = {
      0x2f,0x02,0x54,0x02,0x02,0x04,0x8c,0x60,0x01,0x00,
   };
   const auto neg_code = pack(b);
   ASSERT_EQ(neg_code.size(), sizeof(native_neg) + 4);
   EXPECT_EQ(memcmp(neg_code.data(), native_neg, sizeof(native_neg)), 0);
}

TEST_F(AGX3PackBinary, MixedWidthLog2MatchesCurrentGtkAndNativeModifiers)
{
   /* Exact unchanged GTK component-transfer post-RA shape. Half 76 is read
    * directly and the float result begins at the aligned pair 64. */
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_log2_to(b, agx_register(64, AGX_SIZE_32),
               agx_register(76, AGX_SIZE_16));
   agx_stop(b);
   const uint8_t gtk_native[] = {
      0x2f,0x02,0x54,0x40,0x02,0x98,0x88,0x40,0x00,0x00,
   };
   const auto gtk_code = pack(b);
   ASSERT_EQ(gtk_code.size(), sizeof(gtk_native) + 4);
   EXPECT_EQ(memcmp(gtk_code.data(), gtk_native, sizeof(gtk_native)), 0);

   /* Fresh Apple half-input/full-output form with an odd source and both
    * modifiers. */
   b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index src = agx_register(3, AGX_SIZE_16);
   src.abs = true;
   src.neg = true;
   src.discard = true;
   agx_log2_to(b, agx_register(0, AGX_SIZE_32), src);
   agx_stop(b);
   const uint8_t modified_native[] = {
      0x2f,0x02,0x54,0x00,0x02,0x06,0xa8,0xc0,0x01,0x00,
   };
   const auto modified_code = pack(b);
   ASSERT_EQ(modified_code.size(), sizeof(modified_native) + 4);
   EXPECT_EQ(memcmp(modified_code.data(), modified_native,
                    sizeof(modified_native)), 0);

   /* A full-width destination/source still requires aligned register pairs,
    * and mixed-width waits outside the witnessed no-wait/slot-6 domain must
    * remain a low-level refusal. */
   b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_log2_to(b, agx_register(3, AGX_SIZE_32),
               agx_register(2, AGX_SIZE_16));
   agx_stop(b);
   EXPECT_TRUE(pack(b, false).empty());

   b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_log2_to(b, agx_register(0, AGX_SIZE_32),
               agx_register(192, AGX_SIZE_16));
   agx_stop(b);
   EXPECT_TRUE(pack(b, false).empty());

   b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_log2_to(b, agx_register(192, AGX_SIZE_16),
               agx_register(0, AGX_SIZE_16));
   agx_stop(b);
   EXPECT_TRUE(pack(b, false).empty());
}

TEST_F(AGX3PackBinary, HalfFloorMatchesNativeAndRefusesModifier)
{
   for (unsigned cached = 0; cached < 2; ++cached) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index src = agx_register(3, AGX_SIZE_16);
      src.cache = cached;
      src.discard = !cached;
      agx_floor_to(b, agx_register(3, AGX_SIZE_16), src);
      agx_stop(b);
      const auto code = pack(b);
      const uint8_t native[][10] = {
         {0x2f,0x00,0x54,0x03,0x02,0x06,0xac,0x60,0x02,0x00},
         {0x2f,0x00,0x54,0x03,0x02,0x06,0x8c,0x60,0x02,0x00},
      };
      ASSERT_EQ(code.size(), sizeof(native[cached]) + 4);
      EXPECT_EQ(memcmp(code.data(), native[cached], sizeof(native[cached])), 0);
   }

   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index src = agx_register(2, AGX_SIZE_16);
   src.abs = true;
   agx_floor_to(b, agx_register(2, AGX_SIZE_16), src);
   agx_stop(b);
   EXPECT_TRUE(pack(b, false).empty());
}

TEST_F(AGX3PackBinary, HalfCeilMatchesNativeAndRefusesModifier)
{
   agx_builder *b=builder(AGX_ISA_ENCODING_AGX3B);
   agx_index src=agx_register(3,AGX_SIZE_16); src.discard=true;
   agx_ceil_to(b,agx_register(3,AGX_SIZE_16),src); agx_stop(b);
   const auto code=pack(b);
   const uint8_t native[]={0x2f,0x00,0x54,0x03,0x02,0x06,0xac,0x60,0x04,0x00};
   ASSERT_EQ(code.size(),sizeof(native)+4);
   EXPECT_EQ(memcmp(code.data(),native,sizeof(native)),0);
   b=builder(AGX_ISA_ENCODING_AGX3B); src=agx_register(2,AGX_SIZE_16);
   src.neg=true; agx_ceil_to(b,agx_register(2,AGX_SIZE_16),src); agx_stop(b);
   EXPECT_TRUE(pack(b,false).empty());
}

TEST_F(AGX3PackBinary, HalfMaximumRoutesUntiedAndRefusesModifiers)
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
      const auto code = pack(b, mode == 0);
      EXPECT_EQ(code.empty(), mode != 0) << "mode=" << mode;
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

TEST_F(AGX3PackBinary, ChromiumNegatedRegisterEqualityFselEncodesBySwap)
{
   /* `-a == c` with both selected values in registers. f65 names the SECOND
    * comparator only, so Apple spells this by exchanging the operands and
    * reversing the relation -- equality being its own reverse. The exchange
    * itself is witnessed on the register-select form:
    *
    *    a >= c    12051d07850302c0
    *   -a >= c    12071d05850303c8   operands exchanged, f110 5 -> 6, f65
    *
    * so the shape that used to fail Chromium's pipeline closed is the
    * plain select with one field set. */
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index a = agx_register(2, AGX_SIZE_32);
   a.neg = a.cache = true;
   agx_index c = agx_register(12, AGX_SIZE_32);
   agx_index t = agx_register(10, AGX_SIZE_32);
   agx_index f = agx_register(18, AGX_SIZE_32);
   c.discard = t.discard = f.discard = true;
   agx_fcmpsel_to(b, agx_register(10, AGX_SIZE_32), a, c, t, f,
                  AGX_FCOND_EQ);
   agx_stop(b);
   EXPECT_FALSE(pack(b).empty());
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

/*
 * agx_lower_pseudo spells an AGX3 unconditional break as the always-true
 * identity r0l == r0l; this pins the cached/killed normalization of that
 * identity, and the level count NEST 2 turns into.
 *
 * A loop owns two architectural exec-mask levels, so a break that escapes the
 * loop plus one enclosing block has to stay inactive for three -- CNDCF with
 * count 3, which is Apple's byte-exact form for that shape. CNDLT is the same
 * operation pinned to count 2 and covers only the NEST 1 break, which
 * HalfCompareGnomeLoopBreakAndBothBranchFixups above asserts.
 */
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
   ASSERT_TRUE(
      agx3_cf_cndcf(AGX_ISA_ENCODING_AGX3B, 1, true, 3, bytes, &len));
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

/*
 * The same break, but escaping enclosing ifs as well. Apple's compiler emits
 * CNDCF with a level count equal to the number of architectural levels open
 * inside the loop: the loop's two plus one per enclosing if, i.e. NEST + 1.
 * Witnessed byte for byte for two through seven nested ifs -- see
 * agx3_cf_cndcf()'s comment and test-agx3-cf-core.c.
 */
TEST_F(AGX3PackBinary, IdentityBreakEscapingNestedIfsUsesNativeLevelCount)
{
   for (unsigned ifs = 1; ifs <= 5; ++ifs) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_push_exec(b, 1);
      agx_block *body = agx_test_block(b->shader);
      agx_block *after = agx_test_block(b->shader);
      agx_builder lb = agx_init_builder(b->shader, agx_after_block(body));
      agx_index cond = agx_register(3, AGX_SIZE_16);
      for (unsigned i = 0; i < ifs; ++i)
         agx_if_icmp(&lb, cond, agx_immediate(0), 1, AGX_ICOND_UEQ, true,
                     NULL);
      agx_index r0l = agx_register(0, AGX_SIZE_16);
      r0l.cache = true;
      r0l.kill = true;
      agx_break_if_icmp(&lb, r0l, r0l, ifs + 1, false, AGX_ICOND_UEQ, after);
      for (unsigned i = 0; i < ifs; ++i)
         agx_pop_exec(&lb, 1);
      agx_jmp_exec_any(&lb, body);
      agx_pop_exec(&lb, 1);
      agx_builder ab = agx_init_builder(b->shader, agx_after_block(after));
      agx_stop(&ab);

      uint8_t bytes[AGX3_CF_MAX_LENGTH];
      uint8_t len = 0;
      std::vector<uint8_t> want;
      ASSERT_TRUE(
         agx3_cf_cndst(AGX_ISA_ENCODING_AGX3B, 6, false, 2, bytes, &len));
      append(want, bytes, len);
      const size_t body_offset = want.size();

      const struct agx3_cf_source_hints plain = {false, false};
      for (unsigned i = 0; i < ifs; ++i) {
         ASSERT_TRUE(agx3_cf_icmp_imm16_runtime(AGX_ISA_ENCODING_AGX3B,
                                                AGX3_CF_ICOND_EQ, 0, 3, 0,
                                                plain, 0, bytes, &len));
         append(want, bytes, len);
         ASSERT_TRUE(
            agx3_cf_cndst(AGX_ISA_ENCODING_AGX3B, 0, true, 1, bytes, &len));
         append(want, bytes, len);
      }

      const uint16_t src[] = {0, 0};
      const struct agx3_cf_source_hints live[] = {{false, false},
                                                  {false, false}};
      ASSERT_TRUE(agx3_cf_icmp_reg16(AGX_ISA_ENCODING_AGX3B,
                                     AGX3_CF_ICOND_EQ, 1, src, live, 0, bytes,
                                     &len));
      append(want, bytes, len);
      /* NEST is ifs + 1, so the native level count is ifs + 2. */
      ASSERT_TRUE(agx3_cf_cndcf(AGX_ISA_ENCODING_AGX3B, 1, true, ifs + 2,
                                bytes, &len));
      append(want, bytes, len);

      for (unsigned i = 0; i < ifs; ++i) {
         ASSERT_TRUE(
            agx3_cf_cndend(AGX_ISA_ENCODING_AGX3B, 1, bytes, &len));
         append(want, bytes, len);
      }

      const size_t branch_offset = want.size();
      ASSERT_TRUE(agx3_cf_branch(AGX_ISA_ENCODING_AGX3B, false, bytes, &len));
      ASSERT_TRUE(agx3_cf_patch_branch(AGX_ISA_ENCODING_AGX3B, false, bytes,
                                       (int64_t)body_offset -
                                          (int64_t)branch_offset));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_cf_cndend(AGX_ISA_ENCODING_AGX3B, 2, bytes, &len));
      append(want, bytes, len);
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
      append(want, end.bytes, end.length);

      EXPECT_EQ(pack(b), want) << "ifs=" << ifs;
   }
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

TEST_F(AGX3PackBinary, HyprlandGradientCounterSubtractStaysNative)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index float_source = agx_register(6, AGX_SIZE_32);
   float_source.cache = true;
   agx_index converted = agx_register(3, AGX_SIZE_16);
   converted.cache = true;
   agx_convert_to(b, converted, agx_immediate(AGX_CONVERT_F_TO_S16),
                  float_source, AGX_ROUND_RTZ);
   agx_index source = agx_register(3, AGX_SIZE_16);
   source.discard = true;
   agx_index immediate = agx_immediate(1);
   immediate.neg = true;
   agx_index narrow = agx_register(2, AGX_SIZE_16);
   narrow.cache = true;
   agx_iadd_to(b, narrow, source, immediate, 0);
   agx_index wide = agx_register(4, AGX_SIZE_32);
   wide.cache = true;
   agx_instr *widen = agx_iadd_to(b, wide, agx_immediate(0),
                                  agx_register(2, AGX_SIZE_16), 0);
   widen->shift = 4;
   agx_stop(b);

   const auto code = pack(b);
   const uint8_t native[] = {
      0x27,0x07,0x54,0x03,0x03,0x0c,0x90,0xc8,0x02,0x00,
      0x1f,0x01,0x54,0x02,0x03,0x06,0x04,0x00,0x13,0x01,
      0x9f,0x01,0x54,0x04,0x03,0x00,0x08,0x08,0x10,0x0c,
   };
   ASSERT_EQ(code.size(), sizeof(native) + 4);
   EXPECT_EQ(std::vector<uint8_t>(code.begin(), code.begin() + sizeof(native)),
             std::vector<uint8_t>(native, native + sizeof(native)));
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

TEST_F(AGX3PackBinary, Texture3DImmediateLodStagesThirdCoordinate)
{
   auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_index dest = agx_register(8, AGX_SIZE_32);
   dest.channels_m1 = 3;
   agx_index coord = agx_register(4, AGX_SIZE_32);
   coord.channels_m1 = 2;
   coord.discard = true;
   auto *I = agx_texture_sample_to(
      b, dest, coord, agx_immediate_f(2.0f), agx_immediate(0),
      agx_immediate(0), agx_immediate(0), agx_null(), AGX_DIM_3D,
      AGX_LOD_MODE_LOD_MIN, 0, false, false, false, AGX_GATHER_NONE, false);
   I->mask = 15;
   agx_stop(b);

   std::vector<uint8_t> want;
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
   for (unsigned component = 0; component < 3; ++component) {
      ASSERT_TRUE(agx3_tex_stage_copy_hints(
         enc, 2 * component, 4 + 2 * component, 1, 0, false, true,
         bytes, &len));
      append(want, bytes, len);
   }
   struct agx3_tex_sample_desc d = {};
   d.dest = 8;
   d.width = AGX3_TEX_W32;
   d.mask = 15;
   d.lod = AGX3_TEX_LOD_LEVEL;
   d.lod_x64 = 128;
   d.slot = 1;
   d.wait_mask = 1;
   d.coord_kill = false;
   d.final = true;
   d.three_d = true;
   ASSERT_TRUE(agx3_tex_encode_sample(enc, &d, bytes, &len));
   static const uint8_t native_sample[] = {
      0x25,0x80,0x0c,0x98,0xb0,0x00,0x39,
      0x00,0x00,0x00,0x10,0x08,0x01,0x00,
   };
   EXPECT_EQ(std::vector<uint8_t>(bytes, bytes + len),
             std::vector<uint8_t>(native_sample,
                                  native_sample + sizeof(native_sample)));
   append(want, bytes, len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(enc, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 8);
}

TEST_F(AGX3PackBinary, Texture3DImplicitLodStagesCoordinates)
{
   auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_index dest = agx_register(8, AGX_SIZE_32);
   dest.channels_m1 = 3;
   agx_index coord = agx_register(4, AGX_SIZE_32);
   coord.channels_m1 = 2;
   coord.discard = true;
   auto *I = agx_texture_sample_to(
      b, dest, coord, agx_immediate(0), agx_immediate(0),
      agx_immediate(0), agx_immediate(0), agx_null(), AGX_DIM_3D,
      AGX_LOD_MODE_AUTO_LOD, 0, false, false, false, AGX_GATHER_NONE, false);
   I->mask = 15;
   agx_stop(b);

   std::vector<uint8_t> want;
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
   for (unsigned component = 0; component < 3; ++component) {
      ASSERT_TRUE(agx3_tex_stage_copy_hints(
         enc, 2 * component, 4 + 2 * component, 1, 0, false, true,
         bytes, &len));
      append(want, bytes, len);
   }
   ASSERT_TRUE(agx3_pixel_colour_entry(enc, bytes, &len));
   append(want, bytes, len);
   struct agx3_tex_sample_desc d = {};
   d.dest = 8;
   d.width = AGX3_TEX_W32;
   d.mask = 15;
   d.lod = AGX3_TEX_LOD_AUTO;
   d.slot = 1;
   d.wait_mask = 1;
   d.final = true;
   d.three_d = true;
   ASSERT_TRUE(agx3_tex_encode_sample(enc, &d, bytes, &len));
   /* Rebased onto this test's destination/scoreboard fields from the Apple
    * A18 Pro fragment `texture3d::sample(s, uvw)`. The adjacent native
    * `sample(..., level(0))` witness is identical outside the LOD selector:
    * implicit uses f156=1 while explicit level zero uses f37=1/f139=2. */
   static const uint8_t native_sample[] = {
      0x25,0x80,0x0c,0x98,0xb0,0x00,0x80,
      0x01,0x01,0x00,0x10,0x00,0x01,0x00,
   };
   EXPECT_EQ(std::vector<uint8_t>(bytes, bytes + len),
             std::vector<uint8_t>(native_sample,
                                  native_sample + sizeof(native_sample)));
   append(want, bytes, len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(enc, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 8);
}

TEST_F(AGX3PackBinary, Texture3DRefusesUnwitnessedModes)
{
   for (unsigned shape = 0; shape < 2; ++shape) {
      const auto enc = shape == 1 ? AGX_ISA_ENCODING_AGX3C
                                  : AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_index dest = agx_register(8, AGX_SIZE_32);
      dest.channels_m1 = 3;
      agx_index coord = agx_register(8, AGX_SIZE_32);
      coord.channels_m1 = 2;
      const enum agx_lod_mode mode = shape == 0 ? AGX_LOD_MODE_LOD_GRAD
                                                : AGX_LOD_MODE_LOD_MIN;
      agx_index lod = shape == 0 ? agx_register(20, AGX_SIZE_32)
                                 : agx_immediate_f(1.0f);
      if (shape == 0)
         lod.channels_m1 = 3;
      auto *I = agx_texture_sample_to(
         b, dest, coord, lod, agx_immediate(0), agx_immediate(0),
         agx_immediate(0), agx_null(), AGX_DIM_3D, mode, 0, false, false,
         false, AGX_GATHER_NONE, false);
      I->mask = 15;
      agx_stop(b);
      pack(b, /*expect_ok=*/false);
   }
}

TEST_F(AGX3PackBinary, TextureCubeUsesNativeDimensionAndThirdCoordinate)
{
   auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_index dest = agx_register(8, AGX_SIZE_32);
   dest.channels_m1 = 3;
   agx_index coord = agx_register(4, AGX_SIZE_32);
   coord.channels_m1 = 2;
   coord.discard = true;
   auto *I = agx_texture_sample_to(
      b, dest, coord, agx_immediate(0), agx_immediate(0), agx_immediate(0),
      agx_immediate(0), agx_null(), AGX_DIM_CUBE, AGX_LOD_MODE_AUTO_LOD, 0,
      false, false, false, AGX_GATHER_NONE, false);
   I->mask = 15;
   agx_stop(b);

   std::vector<uint8_t> want;
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
   for (unsigned axis = 0; axis < 3; ++axis) {
      ASSERT_TRUE(agx3_tex_cube_cem_calc(enc, 4, 6, 8, axis, axis == 2,
                                         bytes, &len));
      append(want, bytes, len);
      if (axis == 0) {
         ASSERT_TRUE(agx3_tex_cube_cem_rcp(enc, bytes, &len));
         append(want, bytes, len);
      }
   }
   for (unsigned component = 0; component < 2; ++component) {
      ASSERT_TRUE(agx3_tex_cube_cem_finalize(enc, component,
                                             component == 1, bytes, &len));
      append(want, bytes, len);
   }
   ASSERT_TRUE(agx3_tex_stage_cube_z(enc, 2, 1, 0, false, true,
                                    bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_pixel_colour_entry(enc, bytes, &len));
   append(want, bytes, len);

   struct agx3_tex_sample_desc d = {};
   d.dest = 8;
   d.width = AGX3_TEX_W32;
   d.mask = 15;
   d.lod = AGX3_TEX_LOD_AUTO;
   d.slot = 1;
   d.wait_mask = 1;
   d.coord_kill = true;
   d.final = true;
   d.cube = true;
   ASSERT_TRUE(agx3_tex_encode_sample(enc, &d, bytes, &len));
   static const uint8_t native_sample[] = {
      0x25,0x80,0x0c,0xb8,0xb0,0x00,0x10,
      0x00,0x01,0x00,0x10,0x00,0x01,0x00,
   };
   EXPECT_EQ(std::vector<uint8_t>(bytes, bytes + len),
             std::vector<uint8_t>(native_sample,
                                  native_sample + sizeof(native_sample)));
   append(want, bytes, len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(enc, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
   /* Attachment-only applegpu_g17p cube fragments report fields38/42 = 16.
    * They describe sampler-private staging, not coordinate GPRs 4..8. */
   EXPECT_EQ(b->shader->agx3_staging_halfs, 8);
}

TEST_F(AGX3PackBinary, TextureCubeRegisterDescriptorsUseLiveCEMScratch)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_index dest = agx_register(8, AGX_SIZE_32);
   dest.channels_m1 = 3;
   agx_index coord = agx_register(32, AGX_SIZE_32);
   coord.channels_m1 = 2;
   agx_index texture = agx_register(4, AGX_SIZE_64);
   agx_index sampler = agx_register(24, AGX_SIZE_64);
   auto *I = agx_texture_sample_to(
      b, dest, coord, agx_immediate(0), agx_immediate(0), texture, sampler,
      agx_null(), AGX_DIM_CUBE, AGX_LOD_MODE_AUTO_LOD, 0, false, false,
      false, AGX_GATHER_NONE, false);
   I->mask = 15;
   agx_stop(b);

   /* The ordinary operands occupy through half37, so the reusable three-GPR
    * CEM bank is halves38/40/42. In particular, it cannot overwrite the
    * texture handle at r2:r3 as the old fixed-r2 axis destination did. */
   const uint16_t scratch[3] = {38, 40, 42};
   std::vector<uint8_t> want;
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
   for (unsigned axis = 0; axis < 3; ++axis) {
      ASSERT_TRUE(agx3_tex_cube_cem_calc_to(
         enc, 32, 34, 36, scratch[axis], axis, false, bytes, &len));
      append(want, bytes, len);
      if (axis == 0) {
         ASSERT_TRUE(agx3_tex_cube_cem_rcp_to(enc, scratch[0], bytes, &len));
         append(want, bytes, len);
      }
   }
   for (unsigned component = 0; component < 2; ++component) {
      ASSERT_TRUE(agx3_tex_cube_cem_finalize_to(
         enc, component, scratch[0], scratch[component + 1], 6, bytes, &len));
      append(want, bytes, len);
   }
   ASSERT_TRUE(agx3_tex_stage_cube_z(enc, 2, 6, 0, false, true, bytes,
                                     &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_pixel_colour_entry(enc, bytes, &len));
   append(want, bytes, len);

   const struct agx3_tex_sample_desc sample = {
      .dest = 8,
      .coord = 0,
      .width = AGX3_TEX_W32,
      .mask = 15,
      .lod = AGX3_TEX_LOD_AUTO,
      .slot = 1,
      .wait_mask = 1u << 5,
      .coord_kill = true,
      .final = true,
      .cube = true,
      .texture_kind = AGX3_TEX_OPERAND_REGISTER,
      .sampler_kind = AGX3_TEX_OPERAND_REGISTER,
      .texture_value = 4,
      .sampler_value = 24,
   };
   ASSERT_TRUE(agx3_tex_encode_sample(enc, &sample, bytes, &len));
   append(want, bytes, len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(enc, &end));
   append(want, end.bytes, end.length);

   EXPECT_EQ(pack(b), want);
   EXPECT_EQ(b->shader->max_reg, 43u);
   /* The lane-indexed cube uses slot 6 without resizing the private extent. */
   EXPECT_EQ(b->shader->agx3_staging_halfs, 8);
}

TEST_F(AGX3PackBinary, TextureCubeRefusesEarlyCEMScratchAlias)
{
   auto enc = AGX_ISA_ENCODING_AGX3B;

   for (unsigned coord_base : {0u, 12u}) {
      agx_builder *b = builder(enc);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_index dest = agx_register(24, AGX_SIZE_32);
      dest.channels_m1 = 3;
      agx_index coord = agx_register(coord_base, AGX_SIZE_32);
      coord.channels_m1 = 2;
      coord.discard = true;
      auto *I = agx_texture_sample_to(
         b, dest, coord, agx_immediate(0), agx_immediate(0),
         agx_immediate(0), agx_immediate(0), agx_null(), AGX_DIM_CUBE,
         AGX_LOD_MODE_AUTO_LOD, 0, false, false, false,
         AGX_GATHER_NONE, false);
      I->mask = 15;
      agx_stop(b);

      pack(b, /*expect_ok=*/false);
   }
}

TEST_F(AGX3PackBinary, TextureCubeArrayUsesCEMAndDirectLayer)
{
   auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_index dest = agx_register(8, AGX_SIZE_32);
   dest.channels_m1 = 3;
   agx_index coord = agx_register(16, AGX_SIZE_32);
   coord.channels_m1 = 3;
   coord.discard = true;
   auto *I = agx_texture_sample_to(
      b, dest, coord, agx_immediate(0), agx_immediate(0), agx_immediate(0),
      agx_immediate(0), agx_null(), AGX_DIM_CUBE_ARRAY,
      AGX_LOD_MODE_AUTO_LOD, 0, false, false, false, AGX_GATHER_NONE, false);
   I->mask = 15;
   agx_stop(b);

   std::vector<uint8_t> want;
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
   for (unsigned axis = 0; axis < 3; ++axis) {
      ASSERT_TRUE(agx3_tex_cube_cem_calc(enc, 16, 18, 20, axis, axis == 2,
                                         bytes, &len));
      append(want, bytes, len);
      if (axis == 0) {
         ASSERT_TRUE(agx3_tex_cube_cem_rcp(enc, bytes, &len));
         append(want, bytes, len);
      }
   }
   for (unsigned component = 0; component < 2; ++component) {
      ASSERT_TRUE(agx3_tex_cube_cem_finalize(enc, component,
                                             component == 1, bytes, &len));
      append(want, bytes, len);
   }
   ASSERT_TRUE(agx3_tex_stage_cube_z(enc, 2, 1, 0, false, true,
                                    bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_pixel_colour_entry(enc, bytes, &len));
   append(want, bytes, len);

   const struct agx3_tex_sample_desc sample = {
      .dest = 8, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
      .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
      .coord_kill = true, .final = true, .array = true,
      .array_layer = 22, .cube = true,
   };
   ASSERT_TRUE(agx3_tex_encode_sample(enc, &sample, bytes, &len));
   static const uint8_t native_sample[] = {
      0x25,0x80,0x0c,0xb8,0xb0,0x00,0x13,
      0xb5,0x01,0x00,0x10,0x00,0x01,0x00,
   };
   EXPECT_EQ(std::vector<uint8_t>(bytes, bytes + len),
             std::vector<uint8_t>(native_sample,
                                  native_sample + sizeof(native_sample)));
   append(want, bytes, len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(enc, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 8);
}

TEST_F(AGX3PackBinary, TextureCubeExplicitLodMatchesNativeSelectors)
{
   for (unsigned dynamic = 0; dynamic < 2; ++dynamic) {
      auto enc = AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_index dest = agx_register(8, AGX_SIZE_32);
      dest.channels_m1 = 3;
      agx_index coord = agx_register(4, AGX_SIZE_32);
      coord.channels_m1 = 2;
      coord.discard = true;
      agx_index lod = dynamic ? agx_register(20, AGX_SIZE_16)
                              : agx_immediate_f(2.0f);
      auto *I = agx_texture_sample_to(
         b, dest, coord, lod, agx_immediate(0), agx_immediate(0),
         agx_immediate(0), agx_null(), AGX_DIM_CUBE, AGX_LOD_MODE_LOD_MIN, 0,
         false, false, false, AGX_GATHER_NONE, false);
      I->mask = 15;
      agx_stop(b);

      std::vector<uint8_t> want;
      uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
      if (dynamic) {
         ASSERT_TRUE(agx3_tex_stage_lod(enc, 20, 1, 0, false, false,
                                        bytes, &len));
         append(want, bytes, len);
      }
      for (unsigned axis = 0; axis < 3; ++axis) {
         ASSERT_TRUE(agx3_tex_cube_cem_calc(enc, 4, 6, 8, axis, axis == 2,
                                            bytes, &len));
         append(want, bytes, len);
         if (axis == 0) {
            ASSERT_TRUE(agx3_tex_cube_cem_rcp(enc, bytes, &len));
            append(want, bytes, len);
         }
      }
      for (unsigned component = 0; component < 2; ++component) {
         ASSERT_TRUE(agx3_tex_cube_cem_finalize(enc, component,
                                                component == 1, bytes,
                                                &len));
         append(want, bytes, len);
      }
      ASSERT_TRUE(agx3_tex_stage_cube_z(enc, 2, 1, 0, false, true,
                                        bytes, &len));
      append(want, bytes, len);

      struct agx3_tex_sample_desc d = {};
      d.dest = 8;
      d.width = AGX3_TEX_W32;
      d.mask = 15;
      d.lod = dynamic ? AGX3_TEX_LOD_REGISTER : AGX3_TEX_LOD_LEVEL;
      d.lod_x64 = dynamic ? 0 : 128;
      d.slot = 1;
      d.wait_mask = 1;
      d.coord_kill = true;
      d.final = true;
      d.cube = true;
      ASSERT_TRUE(agx3_tex_encode_sample(enc, &d, bytes, &len));
      static const uint8_t native_immediate[] = {
         0x25, 0x80, 0x0c, 0xb8, 0xb0, 0x00, 0x13,
         0x00, 0x01, 0x00, 0x10, 0x08, 0x01, 0x00,
      };
      static const uint8_t native_register[] = {
         0x25, 0x80, 0x0c, 0xb8, 0xb0, 0x00, 0x00,
         0x03, 0x01, 0x00, 0x10, 0x00, 0x01, 0x00,
      };
      const uint8_t *native = dynamic ? native_register : native_immediate;
      EXPECT_EQ(std::vector<uint8_t>(bytes, bytes + len),
                std::vector<uint8_t>(native, native + sizeof(native_immediate)));
      append(want, bytes, len);
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(enc, &end));
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want);
      EXPECT_EQ(b->shader->agx3_staging_halfs, 8);
   }
}

TEST_F(AGX3PackBinary, ConsecutiveCubeSamplesRetireCEMScratchGeneration)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;

   for (unsigned i = 0; i < 2; ++i) {
      agx_index dest = agx_register(8 * i, AGX_SIZE_32);
      dest.channels_m1 = 3;
      agx_index coord = agx_register(16 + 8 * i, AGX_SIZE_32);
      coord.channels_m1 = 2;
      coord.discard = true;
      auto *I = agx_texture_sample_to(
         b, dest, coord, agx_immediate_f(1.0f), agx_immediate(0),
         agx_immediate(0), agx_immediate(0), agx_null(), AGX_DIM_CUBE,
         AGX_LOD_MODE_LOD_MIN, 0, false, false, false, AGX_GATHER_NONE,
         false);
      I->mask = 15;
   }
   agx_stop(b);

   std::vector<uint8_t> want;
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
   for (unsigned i = 0; i < 2; ++i) {
      const uint8_t slot = i + 1;
      if (i != 0) {
         ASSERT_TRUE(agx3_alu_compose(enc, AGX3_ALU_BAND, 0, 0, 0, 0, 0,
                                      1, bytes, &len));
         append(want, bytes, len);
      }
      for (unsigned axis = 0; axis < 3; ++axis) {
         ASSERT_TRUE(agx3_tex_cube_cem_calc(
            enc, 16 + 8 * i, 18 + 8 * i, 20 + 8 * i, axis, axis == 2,
            bytes, &len));
         append(want, bytes, len);
         if (axis == 0) {
            ASSERT_TRUE(agx3_tex_cube_cem_rcp(enc, bytes, &len));
            append(want, bytes, len);
         }
      }
      for (unsigned component = 0; component < 2; ++component) {
         ASSERT_TRUE(agx3_tex_cube_cem_finalize(enc, component,
                                                component == 1, bytes,
                                                &len));
         append(want, bytes, len);
      }
      ASSERT_TRUE(agx3_tex_stage_cube_z(enc, 2, slot, 0, false, true,
                                        bytes, &len));
      append(want, bytes, len);

      struct agx3_tex_sample_desc d = {};
      d.dest = 8 * i;
      d.width = AGX3_TEX_W32;
      d.mask = 15;
      d.lod = AGX3_TEX_LOD_LEVEL;
      d.lod_x64 = 64;
      d.slot = slot;
      d.wait_mask = 1u << i;
      d.coord_kill = true;
      d.final = i == 1;
      d.sample_group_nonterminal = i == 0;
      d.sample_group_end = i == 1;
      d.cube = true;
      ASSERT_TRUE(agx3_tex_encode_sample(enc, &d, bytes, &len));
      append(want, bytes, len);
   }
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(enc, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, TextureCubeRefusesModesButAcceptsRetainedCoordinates)
{
   for (unsigned shape = 0; shape < 3; ++shape) {
      const auto enc = shape == 1 ? AGX_ISA_ENCODING_AGX3C
                                  : AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_index dest = agx_register(8, AGX_SIZE_32);
      dest.channels_m1 = 3;
      agx_index coord = agx_register(4, AGX_SIZE_32);
      coord.channels_m1 = 2;
      coord.discard = shape != 2;
      const enum agx_lod_mode mode = shape == 0 ? AGX_LOD_MODE_LOD_GRAD
                                                : AGX_LOD_MODE_AUTO_LOD;
      agx_index lod = shape == 0 ? agx_register(20, AGX_SIZE_32)
                                 : agx_immediate(0);
      if (shape == 0)
         lod.channels_m1 = 3;
      auto *I = agx_texture_sample_to(
         b, dest, coord, lod, agx_immediate(0), agx_immediate(0),
         agx_immediate(0), agx_null(), AGX_DIM_CUBE, mode, 0, false, false,
         false, AGX_GATHER_NONE, false);
      I->mask = 15;
      agx_stop(b);
      pack(b, /*expect_ok=*/shape == 2);
   }
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

TEST_F(AGX3PackBinary, ArraySampleStagesDriverLayerAndLifetime)
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
      append(want, bytes, len);
      ASSERT_TRUE(agx3_pixel_colour_entry(enc, bytes, &len));
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
      d.array_layer = 4;
      ASSERT_TRUE(agx3_tex_encode_sample(enc, &d, bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_alu_compose(enc, AGX3_ALU_BAND, 0, 0, 0, 0, 0,
                                   1, bytes, &len));
      append(want, bytes, len);
      EXPECT_EQ(pack(b), want);
      EXPECT_EQ(b->shader->agx3_staging_halfs, 8);
   }
}

/*
 * textureGrad. The gradient collect is (ddx.x, ddy.x, ddx.y, ddy.y) and each
 * scalar is staged into the bank Apple's own kernels stage it into: 8, 10, 12
 * and 14 in half-register units, on one shared staging slot, after the two
 * coordinate copies. agx3_tex_core's own test asserts the sample byte-exact
 * against `xcrun metal-nt -arch applegpu_g17p` on the compute kernel
 * t.sample(s, g[0], gradient2d(g[1], g[2])); this test asserts that the
 * packer drives it with the right banks, sources and staging budget.
 */
TEST_F(AGX3PackBinary, GradientSampleStagesFourScalarsIntoTheirOwnBanks)
{
   auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   key.no_stop = true;
   agx_index dest = agx_register(8, AGX_SIZE_32);
   dest.channels_m1 = 3;
   agx_index coord = agx_register(20, AGX_SIZE_32);
   coord.channels_m1 = 1;
   agx_index grad = agx_register(24, AGX_SIZE_32);
   grad.channels_m1 = 3;
   auto *I = agx_texture_sample_to(
      b, dest, coord, grad, agx_immediate(0), agx_immediate(0),
      agx_immediate(0), agx_null(), AGX_DIM_2D, AGX_LOD_MODE_LOD_GRAD, 0,
      false, false, false, AGX_GATHER_NONE, false);
   I->mask = 15;
   agx_stop(b);

   std::vector<uint8_t> want;
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
   /* Coordinates go to bank 0, as every gradient witness names coord 0. */
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 0, 20, 1, 0, false, false,
                                         bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 2, 22, 1, 0, false, false,
                                         bytes, &len));
   append(want, bytes, len);
   const uint16_t bank[4] = {8, 10, 12, 14};
   for (unsigned i = 0; i < 4; ++i) {
      ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, bank[i], 24 + 2 * i, 1, 0,
                                            false, false, bytes, &len));
      append(want, bytes, len);
   }
   struct agx3_tex_sample_desc d = {};
   d.dest = 8;
   d.width = AGX3_TEX_W32;
   d.mask = 15;
   d.lod = AGX3_TEX_LOD_GRADIENT;
   d.slot = 1;
   d.wait_mask = 1;
   d.final = true;
   ASSERT_TRUE(agx3_tex_encode_sample(enc, &d, bytes, &len));
   /* Apple's own compute gradient sample is 05800cb8b0000401000010000100;
    * this one differs only in the destination register (8, not 0) and the
    * coordinate-kill hint. */
   EXPECT_EQ(std::vector<uint8_t>(bytes, bytes + len),
             (std::vector<uint8_t>{0x25, 0x80, 0x0c, 0x98, 0xb0, 0x00, 0x04,
                                   0x01, 0x00, 0x00, 0x10, 0x00, 0x01, 0x00}));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_alu_compose(enc, AGX3_ALU_BAND, 0, 0, 0, 0, 0,
                                1, bytes, &len));
   append(want, bytes, len);
   EXPECT_EQ(pack(b), want);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 16);
}

TEST_F(AGX3PackBinary, ArrayGradientNamesLayerDirectly)
{
   auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   key.no_stop = true;
   agx_index dest = agx_register(8, AGX_SIZE_32);
   dest.channels_m1 = 3;
   agx_index coord = agx_register(20, AGX_SIZE_32);
   coord.channels_m1 = 2;
   agx_index grad = agx_register(32, AGX_SIZE_32);
   grad.channels_m1 = 3;
   auto *I = agx_texture_sample_to(
      b, dest, coord, grad, agx_immediate(0), agx_immediate(0),
      agx_immediate(0), agx_null(), AGX_DIM_2D_ARRAY,
      AGX_LOD_MODE_LOD_GRAD, 0, false, false, false, AGX_GATHER_NONE, false);
   I->mask = 15;
   agx_stop(b);

   std::vector<uint8_t> want;
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 0, 20, 1, 0, false, false,
                                         bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 2, 22, 1, 0, false, false,
                                         bytes, &len));
   append(want, bytes, len);
   const uint16_t bank[4] = {8, 10, 12, 14};
   for (unsigned i = 0; i < 4; ++i) {
      ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, bank[i], 32 + 2 * i, 1, 0,
                                            false, false, bytes, &len));
      append(want, bytes, len);
   }
   struct agx3_tex_sample_desc d = {};
   d.dest = 8;
   d.width = AGX3_TEX_W32;
   d.mask = 15;
   d.lod = AGX3_TEX_LOD_GRADIENT;
   d.slot = 1;
   d.wait_mask = 1;
   d.final = true;
   d.array = true;
   d.array_layer = 24;
   ASSERT_TRUE(agx3_tex_encode_sample(enc, &d, bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_alu_compose(enc, AGX3_ALU_BAND, 0, 0, 0, 0, 0,
                                1, bytes, &len));
   append(want, bytes, len);
   EXPECT_EQ(pack(b), want);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 16);
}

/* A gradient whose four scalars are not one aligned 32-bit run must fail. */
TEST_F(AGX3PackBinary, GradientSampleRefusesUnwitnessedShapes)
{
   for (unsigned shape = 0; shape < 2; ++shape) {
      auto enc = AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_index dest = agx_register(8, AGX_SIZE_32);
      dest.channels_m1 = 3;
      agx_index coord = agx_register(20, AGX_SIZE_32);
      coord.channels_m1 = 1;
      agx_index grad = agx_register(24, AGX_SIZE_32);
      grad.channels_m1 = shape == 0 ? 1 : 3;
      if (shape == 1)
         grad = agx_register(25, AGX_SIZE_32), grad.channels_m1 = 3;
      auto *I = agx_texture_sample_to(
         b, dest, coord, grad, agx_immediate(0), agx_immediate(0),
         agx_immediate(0), agx_null(),
         AGX_DIM_2D, AGX_LOD_MODE_LOD_GRAD, 0,
         false, false, false, AGX_GATHER_NONE, false);
      I->mask = 15;
      agx_stop(b);
      pack(b, /*expect_ok=*/false);
   }
}

/* A zero bias is the plain implicit sample, byte for byte, exactly as Apple's
 * own compiler folds it. Nothing extra is staged and both use fixed bank 0. */
TEST_F(AGX3PackBinary, ZeroBiasFoldsIntoTheImplicitSample)
{
   auto enc = AGX_ISA_ENCODING_AGX3B;
   std::vector<uint8_t> packed[2];
   for (unsigned biased = 0; biased < 2; ++biased) {
      agx_builder *b = builder(enc);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      key.no_stop = true;
      agx_index dest = agx_register(8, AGX_SIZE_32);
      dest.channels_m1 = 3;
      agx_index coord = agx_register(20, AGX_SIZE_32);
      coord.channels_m1 = 1;
      auto *I = agx_texture_sample_to(
         b, dest, coord, agx_immediate(0), agx_immediate(0), agx_immediate(0),
         agx_immediate(0), agx_null(), AGX_DIM_2D,
         biased ? AGX_LOD_MODE_AUTO_LOD_BIAS : AGX_LOD_MODE_AUTO_LOD, 0,
         false, false, false, AGX_GATHER_NONE, false);
      I->mask = 15;
      agx_stop(b);
      packed[biased] = pack(b);
   }
   EXPECT_FALSE(packed[0].empty());
   EXPECT_EQ(packed[0], packed[1]);
}

/*
 * A per-lane LOD bias stages exactly like a per-lane level -- PACK_SMPIR into
 * operand bank 6, coordinates rebased to bank 0 -- and moves only f139, so
 * the emitted sample differs from the explicit-level one in bytes 6 and 7
 * alone, exactly as Apple's own fragment pair does.
 */
TEST_F(AGX3PackBinary, RegisterLodBiasStagesLikeRegisterLevel)
{
   auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   key.no_stop = true;
   agx_index dest = agx_register(8, AGX_SIZE_32);
   dest.channels_m1 = 3;
   agx_index coord = agx_register(20, AGX_SIZE_32);
   coord.channels_m1 = 1;
   auto *I = agx_texture_sample_to(
      b, dest, coord, agx_register(4, AGX_SIZE_16), agx_immediate(0),
      agx_immediate(0), agx_immediate(0), agx_null(), AGX_DIM_2D,
      AGX_LOD_MODE_AUTO_LOD_BIAS, 0, false, false, false, AGX_GATHER_NONE,
      false);
   I->mask = 15;
   agx_stop(b);

   std::vector<uint8_t> want;
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
   ASSERT_TRUE(agx3_tex_stage_lod(enc, 4, 1, 0, false, false, bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 0, 20, 1, 0, false, false,
                                         bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 2, 22, 1, 0, false, false,
                                         bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_pixel_colour_entry(enc, bytes, &len));
   append(want, bytes, len);
   struct agx3_tex_sample_desc d = {};
   d.dest = 8;
   d.width = AGX3_TEX_W32;
   d.mask = 15;
   d.lod = AGX3_TEX_LOD_BIAS_REGISTER;
   d.slot = 1;
   d.wait_mask = 1;
   d.final = true;
   ASSERT_TRUE(agx3_tex_encode_sample(enc, &d, bytes, &len));
   std::vector<uint8_t> bias(bytes, bytes + len);
   append(want, bytes, len);
   ASSERT_TRUE(agx3_alu_compose(enc, AGX3_ALU_BAND, 0, 0, 0, 0, 0,
                                1, bytes, &len));
   append(want, bytes, len);
   EXPECT_EQ(pack(b), want);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 8);

   d.lod = AGX3_TEX_LOD_REGISTER;
   ASSERT_TRUE(agx3_tex_encode_sample(enc, &d, bytes, &len));
   std::vector<uint8_t> level(bytes, bytes + len);
   ASSERT_EQ(level.size(), bias.size());
   for (size_t i = 0; i < level.size(); ++i) {
      if (i == 6) {
         EXPECT_EQ(level[i], 0x00);
         EXPECT_EQ(bias[i], 0x04);
      } else if (i == 7) {
         EXPECT_EQ(level[i], 0x01);
         EXPECT_EQ(bias[i], 0x00);
      } else {
         EXPECT_EQ(level[i], bias[i]);
      }
   }
}

/*
 * textureGather. AGX_GATHER_{R,G,B,A} are 1/3/5/7 in the IR but the native
 * component field counts 0..3, so the packer halves them; a mistake there
 * would silently gather the wrong channel. The four encodings themselves are
 * asserted against Apple's kernels in test-agx3-tex-core.c.
 */
TEST_F(AGX3PackBinary, GatherComponentMapsOntoTheNativeField)
{
   const enum agx_gather ir[4] = {AGX_GATHER_R, AGX_GATHER_G, AGX_GATHER_B,
                                  AGX_GATHER_A};
   const enum agx3_tex_gather native[4] = {
      AGX3_TEX_GATHER_R, AGX3_TEX_GATHER_G, AGX3_TEX_GATHER_B,
      AGX3_TEX_GATHER_A};

   for (unsigned c = 0; c < 4; ++c) {
      auto enc = AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      key.no_stop = true;
      agx_index dest = agx_register(8, AGX_SIZE_32);
      dest.channels_m1 = 3;
      agx_index coord = agx_register(20, AGX_SIZE_32);
      coord.channels_m1 = 1;
      auto *I = agx_texture_sample_to(
         b, dest, coord, agx_immediate(0), agx_immediate(0), agx_immediate(0),
         agx_immediate(0), agx_null(), AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 0,
         false, false, false, ir[c], false);
      I->mask = 15;
      agx_stop(b);

      std::vector<uint8_t> want;
      uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
      ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 0, 20, 1, 0, false, false,
                                            bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 2, 22, 1, 0, false, false,
                                            bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_pixel_colour_entry(enc, bytes, &len));
      append(want, bytes, len);
      struct agx3_tex_sample_desc d = {};
      d.dest = 8;
      d.coord = 0;
      d.width = AGX3_TEX_W32;
      d.mask = 15;
      d.lod = AGX3_TEX_LOD_AUTO;
      d.slot = 1;
      d.wait_mask = 1;
      d.final = true;
      d.gather = native[c];
      ASSERT_TRUE(agx3_tex_encode_sample(enc, &d, bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_alu_compose(enc, AGX3_ALU_BAND, 0, 0, 0, 0, 0,
                                   1, bytes, &len));
      append(want, bytes, len);
      EXPECT_EQ(pack(b), want);
      EXPECT_EQ(b->shader->agx3_staging_halfs, 4);
   }
}

/* Metal's gather takes no LOD argument, so a gather carrying one has no
 * witness and must fail closed.  The array layer does compose with it:
 * `texture2d_array::gather(s, uv, layer)` is 05800ca490000db4010000001200,
 * the ordinary array sample's f77/f31/f37 plus the gather's f144. */
TEST_F(AGX3PackBinary, GatherRefusesALodButTakesAnArrayLayer)
{
   for (unsigned shape = 0; shape < 3; ++shape) {
      const bool array = shape != 1;
      auto enc = AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_index dest = agx_register(8, AGX_SIZE_32);
      dest.channels_m1 = 3;
      agx_index coord = agx_register(20, AGX_SIZE_32);
      coord.channels_m1 = array ? 2 : 1;
      auto *I = agx_texture_sample_to(
         b, dest, coord,
         shape == 0 ? agx_immediate(0)
                    : agx_immediate(agx_minifloat_encode(0.5f)),
         agx_immediate(0), agx_immediate(0), agx_immediate(0), agx_null(),
         array ? AGX_DIM_2D_ARRAY : AGX_DIM_2D,
         shape == 0 ? AGX_LOD_MODE_AUTO_LOD : AGX_LOD_MODE_AUTO_LOD_BIAS, 0,
         false, false, false, AGX_GATHER_G, false);
      I->mask = 15;
      agx_stop(b);
      /* shape 0: array gather, no LOD -> encodes.
       * shape 1: 2D gather with an immediate bias -> refuses.
       * shape 2: array gather with an immediate bias -> refuses. */
      pack(b, /*expect_ok=*/shape == 0);
   }
}

/*
 * The coordinate offset reaches the sample instruction as an immediate, in
 * every LOD mode. Before this, the packer took an offset only on a plain
 * implicit sample -- and the NIR combined-sampler gate disowned any shader
 * carrying one, so even that path was unreachable and a bare textureOffset()
 * refused the whole shader.
 */
TEST_F(AGX3PackBinary, OffsetSampleRidesEveryLodMode)
{
   struct row {
      enum agx_lod_mode mode;
      const char *name;
   };
   static const row rows[] = {
      {AGX_LOD_MODE_AUTO_LOD, "implicit"},
      {AGX_LOD_MODE_AUTO_LOD_BIAS, "immediate bias"},
      {AGX_LOD_MODE_LOD_MIN, "immediate level"},
      {AGX_LOD_MODE_LOD_GRAD, "gradient"},
   };

   for (const row &r : rows) {
      auto enc = AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      key.no_stop = true;
      agx_index dest = agx_register(8, AGX_SIZE_32);
      dest.channels_m1 = 3;
      agx_index coord = agx_register(20, AGX_SIZE_32);
      coord.channels_m1 = 1;
      agx_index lod = agx_immediate(0);
      if (r.mode == AGX_LOD_MODE_AUTO_LOD_BIAS)
         lod = agx_immediate(agx_minifloat_encode(0.5f));
      else if (r.mode == AGX_LOD_MODE_LOD_MIN)
         lod = agx_immediate(agx_minifloat_encode(2.0f));
      else if (r.mode == AGX_LOD_MODE_LOD_GRAD) {
         lod = agx_register(24, AGX_SIZE_32);
         lod.channels_m1 = 3;
      }
      /* Signed nibble pair (y << 4) | x -- offset (1, 1), the value
       * agx_nir_lower_texture() packs into nir_tex_src_backend2. */
      agx_index offset = agx_immediate(0x11);
      offset.size = AGX_SIZE_32;
      auto *I = agx_texture_sample_to(
         b, dest, coord, lod, agx_immediate(0), agx_immediate(0),
         agx_immediate(0), offset, AGX_DIM_2D, r.mode, 0,
         /*offset=*/true, false, false, AGX_GATHER_NONE, false);
      I->mask = 15;
      I->offset_xy = 0x11;
      agx_stop(b);
      SCOPED_TRACE(r.name);
      pack(b, /*expect_ok=*/true);
   }
}

/*
 * The same offset on the sample instruction itself, checked byte for byte.
 * Apple's `t.sample(s, uv, int2(1,1))` is 05800cb8b0000102018810001200 and
 * differs from its offset-free control by exactly f43 = 17, f51 = 1 and
 * f141 = 1. The stream around the sample (coordinate staging, the fragment
 * pixel entry) is checked by its own tests, so this one only asserts that
 * the sample instruction the packer emitted carries that field triple.
 */
TEST_F(AGX3PackBinary, OffsetSampleMatchesNativeFieldTriple)
{
   auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   key.no_stop = true;
   agx_index dest = agx_register(8, AGX_SIZE_32);
   dest.channels_m1 = 3;
   agx_index coord = agx_register(20, AGX_SIZE_32);
   coord.channels_m1 = 1;
   agx_index offset = agx_immediate(0x11);
   offset.size = AGX_SIZE_32;
   auto *I = agx_texture_sample_to(
      b, dest, coord, agx_immediate(0), agx_immediate(0), agx_immediate(0),
      agx_immediate(0), offset, AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 0,
      /*offset=*/true, false, false, AGX_GATHER_NONE, false);
   I->mask = 15;
   I->offset_xy = 0x11;
   agx_stop(b);
   const std::vector<uint8_t> stream = pack(b);

   /* The same sample with no offset, so the difference really is the triple
    * and not something the shape changed along the way. */
   agx_builder *b2 = builder(enc);
   b2->shader->stage = MESA_SHADER_FRAGMENT;
   key.no_stop = true;
   auto *I2 = agx_texture_sample_to(
      b2, dest, coord, agx_immediate(0), agx_immediate(0), agx_immediate(0),
      agx_immediate(0), agx_null(), AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 0,
      false, false, false, AGX_GATHER_NONE, false);
   I2->mask = 15;
   agx_stop(b2);
   const std::vector<uint8_t> plain = pack(b2);

   struct agx3_tex_sample_desc d = {};
   d.dest = 8;
   d.width = AGX3_TEX_W32;
   d.mask = 15;
   d.lod = AGX3_TEX_LOD_AUTO;
   d.slot = 1;
   d.wait_mask = 1;
   d.final = true;
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
   ASSERT_TRUE(agx3_tex_encode_sample(enc, &d, bytes, &len));
   const std::vector<uint8_t> want_plain(bytes, bytes + len);
   d.offset = true;
   d.offset_xy = 0x11;
   ASSERT_TRUE(agx3_tex_encode_sample(enc, &d, bytes, &len));
   const std::vector<uint8_t> want_offset(bytes, bytes + len);

   EXPECT_NE(want_plain, want_offset);
   EXPECT_NE(std::search(stream.begin(), stream.end(), want_offset.begin(),
                         want_offset.end()),
             stream.end());
   EXPECT_NE(std::search(plain.begin(), plain.end(), want_plain.begin(),
                         want_plain.end()),
             plain.end());
}

/*
 * The depth comparison is orthogonal to the LOD shape. Apple's depth2d
 * sample_compare with level(2), bias(0.5), level(reg), bias(reg) and
 * gradient2d() each differ from the corresponding colour sample by f143
 * alone; before this, only the implicit form was admitted.
 */
TEST_F(AGX3PackBinary, ShadowSampleRidesEveryLodMode)
{
   struct row {
      enum agx_lod_mode mode;
      const char *name;
   };
   static const row rows[] = {
      {AGX_LOD_MODE_AUTO_LOD, "implicit"},
      {AGX_LOD_MODE_AUTO_LOD_BIAS, "immediate bias"},
      {AGX_LOD_MODE_LOD_MIN, "immediate level"},
      {AGX_LOD_MODE_LOD_GRAD, "gradient"},
   };

   for (const row &r : rows) {
      auto enc = AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      key.no_stop = true;
      agx_index dest = agx_register(0, AGX_SIZE_32);
      dest.channels_m1 = 3;
      agx_index coord = agx_register(8, AGX_SIZE_32);
      coord.channels_m1 = 1;
      coord.discard = true;
      agx_index compare = agx_register(2, AGX_SIZE_32);
      compare.discard = true;
      agx_index lod = agx_immediate(0);
      if (r.mode == AGX_LOD_MODE_AUTO_LOD_BIAS)
         lod = agx_immediate(agx_minifloat_encode(0.5f));
      else if (r.mode == AGX_LOD_MODE_LOD_MIN)
         lod = agx_immediate(agx_minifloat_encode(2.0f));
      else if (r.mode == AGX_LOD_MODE_LOD_GRAD) {
         lod = agx_register(16, AGX_SIZE_32);
         lod.channels_m1 = 3;
      }
      auto *I = agx_texture_sample_to(
         b, dest, coord, lod, agx_immediate(0), agx_immediate(0),
         agx_immediate(0), compare, AGX_DIM_2D, r.mode, 0, false,
         /*shadow=*/true, false, AGX_GATHER_NONE, false);
      I->mask = 1;
      agx_stop(b);
      SCOPED_TRACE(r.name);
      pack(b, /*expect_ok=*/true);
   }
}

/*
 * gather_compare: the ordinary gather plus f143, returning four texels and so
 * carrying the full f75 mask rather than the single component every other
 * comparison carries. A comparison whose mask does not match its shape must
 * still refuse.
 */
TEST_F(AGX3PackBinary, ShadowGatherKeepsTheFourTexelMask)
{
   for (unsigned mask : {15u, 1u}) {
      auto enc = AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      key.no_stop = true;
      agx_index dest = agx_register(0, AGX_SIZE_32);
      dest.channels_m1 = 3;
      agx_index coord = agx_register(8, AGX_SIZE_32);
      coord.channels_m1 = 1;
      coord.discard = true;
      agx_index compare = agx_register(2, AGX_SIZE_32);
      compare.discard = true;
      auto *I = agx_texture_sample_to(
         b, dest, coord, agx_immediate(0), agx_immediate(0), agx_immediate(0),
         agx_immediate(0), compare, AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 0,
         false, /*shadow=*/true, false, AGX_GATHER_R, false);
      I->mask = mask;
      agx_stop(b);
      pack(b, /*expect_ok=*/mask == 15u);
   }
}

/*
 * A comparison sample takes a coordinate offset. Apple's
 * depth2d::sample_compare(s, uv, ref, int2(1,1)) is
 * 1d800ca090002102018800001200 -- the plain comparison plus the same
 * f43/f51/f141 triple every other sample takes.
 *
 * The reason this could not be encoded before was not the ISA: agx_emit_tex()
 * collected the comparator and the packed offset into one src[5] vector, and
 * the comparator staging cannot take that apart. The offset now rides on the
 * instruction (I->offset_xy) and src[5] belongs to the comparator alone.
 */
TEST_F(AGX3PackBinary, ShadowSampleTakesACoordinateOffset)
{
   auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   key.no_stop = true;
   agx_index dest = agx_register(0, AGX_SIZE_32);
   dest.channels_m1 = 3;
   agx_index coord = agx_register(8, AGX_SIZE_32);
   coord.channels_m1 = 1;
   coord.discard = true;
   agx_index compare = agx_register(2, AGX_SIZE_32);
   compare.discard = true;
   auto *I = agx_texture_sample_to(
      b, dest, coord, agx_immediate(0), agx_immediate(0), agx_immediate(0),
      agx_immediate(0), compare, AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 0,
      /*offset=*/true, /*shadow=*/true, false, AGX_GATHER_NONE, false);
   I->mask = 1;
   I->offset_xy = 0x11;
   agx_stop(b);
   const std::vector<uint8_t> stream = pack(b);

   struct agx3_tex_sample_desc d = {};
   d.dest = 0;
   d.width = AGX3_TEX_W32;
   d.mask = 1;
   d.lod = AGX3_TEX_LOD_AUTO;
   d.slot = 1;
   d.wait_mask = 1;
   d.final = true;
   d.shadow = true;
   d.offset = true;
   d.offset_xy = 0x11;
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
   ASSERT_TRUE(agx3_tex_encode_sample(enc, &d, bytes, &len));
   const std::vector<uint8_t> want(bytes, bytes + len);
   EXPECT_NE(std::search(stream.begin(), stream.end(), want.begin(),
                         want.end()),
             stream.end());

   /* An offset the instruction does not carry must refuse rather than encode
    * as offset (0,0), which native spells as the plain form. */
   agx_builder *b2 = builder(enc);
   b2->shader->stage = MESA_SHADER_FRAGMENT;
   key.no_stop = true;
   auto *I2 = agx_texture_sample_to(
      b2, dest, coord, agx_immediate(0), agx_immediate(0), agx_immediate(0),
      agx_immediate(0), compare, AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 0,
      /*offset=*/true, /*shadow=*/true, false, AGX_GATHER_NONE, false);
   I2->mask = 1;
   agx_stop(b2);
   pack(b2, /*expect_ok=*/false);
}

/*
 * REGRESSION GUARD. agx_opt_cse.c's instrs_equal() is hand-written and
 * enumerates the instruction's immediates one at a time, so a new field that
 * is not added there lets two otherwise-identical instructions collapse into
 * one. I->offset_xy is such a field: without its line in instrs_equal(), two
 * samples of the same texture at the same coordinate but *different* texel
 * offsets would CSE together and the second would silently read the first's
 * texels.
 *
 * This asserts the two are not equal. Delete the offset_xy line from
 * instrs_equal() and this test fails.
 */
TEST_F(AGX3PackBinary, DifferentTexelOffsetsDoNotCSETogether)
{
   auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_index coord = agx_register(4, AGX_SIZE_32);
   coord.channels_m1 = 1;

   agx_index d0 = agx_register(8, AGX_SIZE_32);
   d0.channels_m1 = 3;
   agx_index d1 = agx_register(16, AGX_SIZE_32);
   d1.channels_m1 = 3;

   agx_index off = agx_immediate(0x11);
   off.size = AGX_SIZE_32;

   auto *a = agx_texture_sample_to(
      b, d0, coord, agx_immediate(0), agx_immediate(0), agx_immediate(0),
      agx_immediate(0), off, AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 0,
      /*offset=*/true, false, false, AGX_GATHER_NONE, false);
   a->mask = 15;
   a->offset_xy = 0x11;

   auto *c = agx_texture_sample_to(
      b, d1, coord, agx_immediate(0), agx_immediate(0), agx_immediate(0),
      agx_immediate(0), off, AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 0,
      /*offset=*/true, false, false, AGX_GATHER_NONE, false);
   c->mask = 15;
   c->offset_xy = 0x22; /* the only difference */

   EXPECT_NE(a->offset_xy, c->offset_xy);
   EXPECT_FALSE(agx_instrs_equal_for_cse(a, c));

   /* ... and two with the same offset still compare equal, so the guard is
    * not simply always-false. */
   c->offset_xy = 0x11;
   c->dest[0] = d0;
   EXPECT_TRUE(agx_instrs_equal_for_cse(a, c));
}

TEST_F(AGX3PackBinary, ShadowArrayStagesPackedLayer)
{
   auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   key.no_stop = true;
   agx_index dest = agx_register(0, AGX_SIZE_32);
   dest.channels_m1 = 3;
   agx_index coord = agx_register(8, AGX_SIZE_32);
   coord.channels_m1 = 2;
   coord.discard = true;
   agx_index compare = agx_register(2, AGX_SIZE_32);
   compare.discard = true;
   auto *I = agx_texture_sample_to(
      b, dest, coord, agx_immediate(0), agx_immediate(0), agx_immediate(0),
      agx_immediate(0), compare, AGX_DIM_2D_ARRAY, AGX_LOD_MODE_AUTO_LOD, 0,
      false, true, false, AGX_GATHER_NONE, false);
   I->mask = 1;
   agx_stop(b);

   std::vector<uint8_t> want;
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 0, 8, 1, 0, false, true,
                                         bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 2, 10, 1, 0, false, true,
                                         bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_array_layer(enc, 12, 1, 0, false, true,
                                          bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 4, 2, 1, 0, false, true,
                                         bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_pixel_colour_entry(enc, bytes, &len));
   append(want, bytes, len);
   struct agx3_tex_sample_desc d = {};
   d.dest = 0;
   d.width = AGX3_TEX_W32;
   d.mask = 1;
   d.lod = AGX3_TEX_LOD_AUTO;
   d.slot = 1;
   d.wait_mask = 1;
   d.final = true;
   d.array = true;
   d.array_layer = 12;
   d.shadow = true;
   ASSERT_TRUE(agx3_tex_encode_sample(enc, &d, bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_alu_compose(enc, AGX3_ALU_BAND, 0, 0, 0, 0, 0,
                                1, bytes, &len));
   append(want, bytes, len);

   EXPECT_EQ(pack(b), want);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 8);
}

TEST_F(AGX3PackBinary, ShadowCubeStagesCompareBeforeCEM)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_index dest = agx_register(0, AGX_SIZE_32);
   dest.channels_m1 = 3;
   agx_index coord = agx_register(16, AGX_SIZE_32);
   coord.channels_m1 = 2;
   coord.discard = true;
   /* Deliberately use CEM's r0 scratch as the comparison source. The packer
    * must publish it before CEM overwrites that register. */
   agx_index compare = agx_register(0, AGX_SIZE_32);
   compare.discard = true;
   auto *I = agx_texture_sample_to(
      b, dest, coord, agx_immediate(0), agx_immediate(0), agx_immediate(0),
      agx_immediate(0), compare, AGX_DIM_CUBE, AGX_LOD_MODE_AUTO_LOD, 0,
      false, true, false, AGX_GATHER_NONE, false);
   I->mask = 1;
   agx_stop(b);

   std::vector<uint8_t> want;
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 4, 0, 1, 0, false, true,
                                         bytes, &len));
   append(want, bytes, len);
   for (unsigned axis = 0; axis < 3; ++axis) {
      ASSERT_TRUE(agx3_tex_cube_cem_calc(enc, 16, 18, 20, axis,
                                         axis == 2, bytes, &len));
      append(want, bytes, len);
      if (axis == 0) {
         ASSERT_TRUE(agx3_tex_cube_cem_rcp(enc, bytes, &len));
         append(want, bytes, len);
      }
   }
   for (unsigned component = 0; component < 2; ++component) {
      ASSERT_TRUE(agx3_tex_cube_cem_finalize(enc, component,
                                             component == 1, bytes, &len));
      append(want, bytes, len);
   }
   ASSERT_TRUE(agx3_tex_stage_cube_z(enc, 2, 1, 0, false, true, bytes,
                                    &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_pixel_colour_entry(enc, bytes, &len));
   append(want, bytes, len);
   const struct agx3_tex_sample_desc sample = {
      .dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 1,
      .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
      .coord_kill = true, .final = true, .cube = true, .shadow = true,
   };
   ASSERT_TRUE(agx3_tex_encode_sample(enc, &sample, bytes, &len));
   static const uint8_t native_sample[] = {
      0x05,0x80,0x0c,0xa0,0x90,0x00,0x30,
      0x00,0x01,0x00,0x00,0x00,0x01,0x00,
   };
   EXPECT_EQ(std::vector<uint8_t>(bytes, bytes + len),
             std::vector<uint8_t>(std::begin(native_sample),
                                  std::end(native_sample)));
   append(want, bytes, len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(enc, &end));
   append(want, end.bytes, end.length);

   EXPECT_EQ(pack(b), want);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 8);
}

TEST_F(AGX3PackBinary, ShadowCubeArrayUsesCombinedFaceLayerPack)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_index dest = agx_register(0, AGX_SIZE_32);
   dest.channels_m1 = 3;
   agx_index coord = agx_register(16, AGX_SIZE_32);
   coord.channels_m1 = 3;
   coord.discard = true;
   /* r1 matches the Gallium cube-array compile shape and exercises the same
    * early comparison publication as the native cube-array shader. */
   agx_index compare = agx_register(2, AGX_SIZE_32);
   compare.discard = true;
   auto *I = agx_texture_sample_to(
      b, dest, coord, agx_immediate(0), agx_immediate(0), agx_immediate(0),
      agx_immediate(0), compare, AGX_DIM_CUBE_ARRAY,
      AGX_LOD_MODE_AUTO_LOD, 0, false, true, false, AGX_GATHER_NONE, false);
   I->mask = 1;
   agx_stop(b);

   std::vector<uint8_t> want;
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 4, 2, 1, 0, false, true,
                                         bytes, &len));
   append(want, bytes, len);
   for (unsigned axis = 0; axis < 3; ++axis) {
      ASSERT_TRUE(agx3_tex_cube_cem_calc(enc, 16, 18, 20, axis,
                                         axis == 2, bytes, &len));
      append(want, bytes, len);
      if (axis == 0) {
         ASSERT_TRUE(agx3_tex_cube_cem_rcp(enc, bytes, &len));
         append(want, bytes, len);
      }
   }
   for (unsigned component = 0; component < 2; ++component) {
      ASSERT_TRUE(agx3_tex_cube_cem_finalize(enc, component,
                                             component == 1, bytes, &len));
      append(want, bytes, len);
   }
   ASSERT_TRUE(agx3_tex_stage_cube_array_shadow(
      enc, 22, 1, 0, false, true, bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_pixel_colour_entry(enc, bytes, &len));
   append(want, bytes, len);
   const struct agx3_tex_sample_desc sample = {
      .dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 1,
      .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
      .coord_kill = true, .final = true, .array = true,
      .array_layer = 22, .cube = true, .shadow = true,
   };
   ASSERT_TRUE(agx3_tex_encode_sample(enc, &sample, bytes, &len));
   static const uint8_t native_sample[] = {
      0x05,0x80,0x0c,0xa0,0x90,0x00,0x30,
      0x04,0x01,0x00,0x00,0x00,0x01,0x00,
   };
   EXPECT_EQ(std::vector<uint8_t>(bytes, bytes + len),
             std::vector<uint8_t>(std::begin(native_sample),
                                  std::end(native_sample)));
   append(want, bytes, len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(enc, &end));
   append(want, end.bytes, end.length);

   EXPECT_EQ(pack(b), want);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 8);
}

TEST_F(AGX3PackBinary, ArrayRegisterDescriptorsStageDriverLayer)
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
   append(want, bytes, len);
   ASSERT_TRUE(agx3_pixel_colour_entry(enc, bytes, &len));
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
   d.array_layer = 24;
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
TEST_F(AGX3PackBinary, ArrayDriverLayerAcceptsHighRegistersWithoutWrap)
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
   ASSERT_TRUE(agx3_pixel_colour_entry(enc, bytes, &len));
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
   d.array_layer = 124;
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
   struct agx3_image_write_desc d = {
      .wait_mask = 1,
      .image_kind = AGX3_TEX_OPERAND_IMMEDIATE,
      .image_value = 0,
      .normalized = false,
   };
   ASSERT_TRUE(agx3_tex_encode_image_write_rgba8_2d(
      AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
   ASSERT_EQ(len, sizeof(native));
   EXPECT_EQ(memcmp(bytes, native, sizeof(native)), 0);

   d.wait_mask = 0;
   ASSERT_TRUE(agx3_tex_encode_image_write_rgba8_2d(
      AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
   ASSERT_EQ(len, sizeof(native));
   EXPECT_EQ(memcmp(bytes, native, sizeof(native)), 0);

   for (unsigned wait : {2u, 3u, 0x40u}) {
      memset(bytes, 0xa5, sizeof(bytes));
      len = 0xa5;
      d.wait_mask = wait;
      EXPECT_FALSE(agx3_tex_encode_image_write_rgba8_2d(
         AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
      EXPECT_EQ(len, 0);
      for (uint8_t byte : bytes)
         EXPECT_EQ(byte, 0);
   }

   memset(bytes, 0xa5, sizeof(bytes));
   len = 0xa5;
   d.wait_mask = 1;
   EXPECT_FALSE(agx3_tex_encode_image_write_rgba8_2d(
      AGX_ISA_ENCODING_AGX3C, &d, bytes, &len));
   EXPECT_EQ(len, 0);
   for (uint8_t byte : bytes)
      EXPECT_EQ(byte, 0);

   static const uint8_t sampled_write[AGX3_IMAGE_WRITE_LENGTH] = {
      0xd7, 0x20, 0x54, 0x00, 0x20, 0x40, 0x00, 0x01,
      0x00, 0x04, 0x00, 0x00, 0x88, 0x3a, 0x09, 0x00,
   };
   d.image_value = 1;
   d.normalized = true;
   ASSERT_TRUE(agx3_tex_encode_image_write_rgba8_2d(
      AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
   ASSERT_EQ(len, sizeof(sampled_write));
   EXPECT_EQ(memcmp(bytes, sampled_write, sizeof(sampled_write)), 0);

   d.image_value = 0;
   EXPECT_FALSE(agx3_tex_encode_image_write_rgba8_2d(
      AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
   d.image_value = 1;
   d.normalized = false;
   EXPECT_FALSE(agx3_tex_encode_image_write_rgba8_2d(
      AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));

   /* applegpu-nt 32023.917 / apple-g17p, dynamic argument-buffer image
    * selection. The retained index at r10 is a byte-exact native witness. */
   static const uint8_t indexed_live[AGX3_IMAGE_WRITE_LENGTH] = {
      0xd7, 0x20, 0x54, 0x00, 0x00, 0x40, 0x00, 0x14,
      0x20, 0x04, 0x00, 0x00, 0x88, 0x3a, 0x09, 0x01,
   };
   d = {
      .wait_mask = 1,
      .image_kind = AGX3_TEX_OPERAND_REGISTER,
      .image_value = 20,
      .normalized = true,
   };
   ASSERT_TRUE(agx3_tex_encode_image_write_rgba8_2d(
      AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
   ASSERT_EQ(len, sizeof(indexed_live));
   EXPECT_EQ(memcmp(bytes, indexed_live, sizeof(indexed_live)), 0);

   /* The killed-index form is independently witnessed at r4. Combining it
    * with the same slot-2 staging wait gives the production bindless form. */
   static const uint8_t indexed_kill[AGX3_IMAGE_WRITE_LENGTH] = {
      0xd7, 0x20, 0x54, 0x00, 0x00, 0x40, 0x00, 0x08,
      0x21, 0x04, 0x00, 0x00, 0x88, 0x3a, 0x89, 0x00,
   };
   d.image_value = 8;
   d.image_kill = true;
   ASSERT_TRUE(agx3_tex_encode_image_write_rgba8_2d(
      AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
   ASSERT_EQ(len, sizeof(indexed_kill));
   EXPECT_EQ(memcmp(bytes, indexed_kill, sizeof(indexed_kill)), 0);

   d.image_value = 9;
   EXPECT_FALSE(agx3_tex_encode_image_write_rgba8_2d(
      AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
   d.image_value = 8;
   d.image_kind = AGX3_TEX_OPERAND_UNIFORM;
   EXPECT_FALSE(agx3_tex_encode_image_write_rgba8_2d(
      AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
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

TEST_F(AGX3PackBinary, BindlessStorageImageWriteMatchesNativeIndexForm)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_COMPUTE;
   agx_index data = agx_register(8, AGX_SIZE_32);
   data.channels_m1 = 3;
   data.discard = true;
   agx_index coord = agx_register(16, AGX_SIZE_32);
   coord.channels_m1 = 1;
   coord.discard = true;
   agx_index index = agx_register(2, AGX_SIZE_32);
   index.discard = true;
   agx_instr *write = agx_image_write(
      b, data, coord, agx_immediate(0), agx_uniform(4, AGX_SIZE_64),
      index, AGX_DIM_2D, false);
   write->format = AGX_FORMAT_U8NORM;
   agx_stop(b);

   const std::vector<uint8_t> code = pack(b);
   static const uint8_t imgwci[] = {
      0xd7, 0x20, 0x54, 0x00, 0x00, 0x40, 0x00, 0x02,
      0x21, 0x04, 0x00, 0x00, 0x88, 0x3a, 0x89, 0x00,
   };
   EXPECT_NE(std::search(code.begin(), code.end(), imgwci,
                         imgwci + sizeof(imgwci)),
             code.end());

   agx_builder *bad = builder(AGX_ISA_ENCODING_AGX3B);
   bad->shader->stage = MESA_SHADER_COMPUTE;
   index = agx_register(3, AGX_SIZE_32);
   index.discard = true;
   write = agx_image_write(
      bad, data, coord, agx_immediate(0), agx_uniform(4, AGX_SIZE_64),
      index, AGX_DIM_2D, false);
   write->format = AGX_FORMAT_U8NORM;
   agx_stop(bad);
   EXPECT_TRUE(pack(bad, false).empty());
}

TEST_F(AGX3PackBinary, BindlessStorageImageReadMatchesNativeIndexAndLOD)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_COMPUTE;
   agx_index dest = agx_register(0, AGX_SIZE_32);
   dest.channels_m1 = 3;
   agx_index coord = agx_register(12, AGX_SIZE_32);
   coord.channels_m1 = 1;
   coord.discard = true;
   agx_index lod = agx_register(4, AGX_SIZE_32);
   lod.discard = true;
   agx_index texture = agx_register(0, AGX_SIZE_64);
   texture.discard = true;
   agx_instr *read = agx_image_load_to(
      b, dest, coord, lod, agx_uniform(0, AGX_SIZE_64), texture,
      agx_immediate(0), agx_null(), AGX_DIM_2D, AGX_LOD_MODE_LOD_MIN,
      15, false, false, false);
   read->mask = 15;
   agx_stop(b);

   const std::vector<uint8_t> code = pack(b);
   /* applegpu-nt 32023.917 / apple-g17p, lane-selected argument-buffer
    * texture plus a uint level widened from ushort. Coordinates and LOD are
    * published on slot 2; the image result independently occupies slot 1. */
   static const uint8_t native_lod[] = {
      0x17,0x05,0x54,0x06,0x00,0x08,0x00,0x04,0xf0,0xc4,0x0c,0x00,
   };
   static const uint8_t native_read[] = {
      0x05,0x80,0x0c,0xb8,0x30,0x01,0x80,0x24,0x00,0x02,
      0x80,0x00,0x01,0x00,0x00,0x28,0x00,0x00,0x00,0x00,
   };
   EXPECT_NE(std::search(code.begin(), code.end(), native_lod,
                         native_lod + sizeof(native_lod)), code.end());
   EXPECT_NE(std::search(code.begin(), code.end(), native_read,
                         native_read + sizeof(native_read)), code.end());

   agx_builder *bad = builder(AGX_ISA_ENCODING_AGX3B);
   bad->shader->stage = MESA_SHADER_COMPUTE;
   lod = agx_register(4, AGX_SIZE_16);
   lod.discard = true;
   read = agx_image_load_to(
      bad, dest, coord, lod, agx_uniform(0, AGX_SIZE_64), texture,
      agx_immediate(0), agx_null(), AGX_DIM_2D, AGX_LOD_MODE_LOD_MIN,
      15, false, false, false);
   read->mask = 15;
   agx_stop(bad);
   EXPECT_TRUE(pack(bad, false).empty());
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


/*
 * The six shapes a three.js frame was refused on.
 *
 * Measured on hardware: Chromium loading shadowmap_pointlight, lights_spotlight
 * and animation_keyframes with AGX_MESA_DEBUG=shaders printed 307 refusals in
 * one run, which reduce to six once register and immediate numbers are
 * normalised. Every one is a source modifier, an operand width, or an operand
 * KIND -- three axes that are independent fields of a single AGX3 encoding
 * point, and which the packer modelled as separate encoders. A shape sitting
 * on two axes at once then had no home in either.
 *
 * Each case below is the refusal line verbatim, and each encoding is grounded
 * in the byte ladders quoted at the encoders themselves.
 */
TEST_F(AGX3PackBinary, ThreeJsRefusedShapesEncode)
{
   const enum agx_isa_encoding enc = AGX_ISA_ENCODING_AGX3B;

   /* 113x  fcmpsel [d16:reg s0=32:reg.abs.discard s1=16:imm s2=16:imm
    *                s3=16:imm cond=1 chan=1]
    * An absolute comparator against a non-zero minifloat. The gate demanded
    * an immediate comparator of exactly zero; f51 holds any minifloat. */
   {
      agx_builder *b = builder(enc);
      agx_index a = agx_register(2, AGX_SIZE_32);
      a.abs = true;
      a.discard = true;
      agx_fcmpsel_to(b, agx_register(2, AGX_SIZE_16), a, agx_immediate(0x20),
                     agx_immediate(0x30), agx_immediate(0), AGX_FCOND_LT);
      agx_stop(b);
      uint8_t bytes[AGX3_ALU_MAX_LENGTH];
      uint8_t len = 0;
      struct agx3_fsel_desc d = {};
      d.dest = 2;
      d.dest_16 = true;
      d.cmp0 = 2;
      d.cmp0_hint = AGX3_ALU_HINT_DISCARD;
      d.cmp0_abs = true;
      d.cmp1_is_immediate = true;
      d.cmp1 = 0x20;
      d.sel0_kind = AGX3_FSEL_SEL_IMMEDIATE;
      d.sel0 = 0x30;
      d.sel1_kind = AGX3_FSEL_SEL_IMMEDIATE;
      d.sel1 = 0;
      d.sel_16 = true;
      d.condition = AGX_FCOND_LT;
      ASSERT_TRUE(agx3_alu_fcmp_select(enc, &d, bytes, &len));
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(enc, &end));
      std::vector<uint8_t> want;
      append(want, bytes, len);
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want);
   }

   /* 84x  fadd [d32:reg s0=16:reg s1=32:reg.neg cond=0 chan=1]
    * A half added to a negated full float. The width lives in f43/f26 and the
    * negate in f48/f31, on one encoding point; the packer had one encoder for
    * widths without modifiers and another for modifiers without widths. */
   {
      agx_builder *b = builder(enc);
      agx_index f16 = agx_register(4, AGX_SIZE_16);
      agx_index f32 = agx_register(0, AGX_SIZE_32);
      f32.neg = true;
      f32.discard = true;
      f16.discard = true;
      agx_fadd_to(b, agx_register(0, AGX_SIZE_32), f16, f32);
      agx_stop(b);
      /* Apple's own `b - a`, `float a`, `half b`: 09041c0100c8. The wait bit
       * f57 belongs to the loads in that kernel, which this shape has not
       * got, so the ladder row is reproduced through the encoder. */
      struct agx3_falu_mods m = {};
      m.src0_neg = true;
      m.src1_16 = true;
      struct agx3_falu_runtime rt = {};
      rt.discard[0] = true;
      rt.discard[1] = true;
      rt.wait_mask = 0x20;
      uint8_t bytes[AGX3_FALU_MOD_MAX_LENGTH];
      uint8_t len = 0;
      ASSERT_TRUE(
         agx3_falu_add_mod_runtime(enc, 0, 0, 4, &m, &rt, bytes, &len));
      EXPECT_EQ(std::vector<uint8_t>(bytes, bytes + len),
                (std::vector<uint8_t>{0x09, 0x04, 0x1c, 0x01, 0x00, 0xc8}));
      EXPECT_FALSE(pack(b).empty());
   }

   /* 80x  fcmpsel [d16:reg s0=32:reg.neg s1=32:reg.neg s2=16:reg s3=16:imm
    *               cond=5 chan=1]
    * Both comparators negated. Apple's own bytes for `-a >= -c` and `a <= c`
    * are the same instruction, so this is the relation flip and nothing else;
    * the encoder must produce exactly the plain select's bytes. */
   {
      auto build = [&](bool negated, enum agx_fcond cond) {
         agx_builder *b = builder(enc);
         agx_index a = agx_register(8, AGX_SIZE_32);
         agx_index c = agx_register(16, AGX_SIZE_32);
         a.neg = c.neg = negated;
         agx_fcmpsel_to(b, agx_register(2, AGX_SIZE_16), a, c,
                        agx_register(6, AGX_SIZE_16), agx_immediate(0), cond);
         agx_stop(b);
         return pack(b);
      };
      const auto negated = build(true, AGX_FCOND_GE);
      const auto flipped = build(false, AGX_FCOND_LE);
      EXPECT_FALSE(negated.empty());
      EXPECT_EQ(negated, flipped);

      /* One negate is f65, with the operands and the relation left alone. */
      agx_builder *b = builder(enc);
      agx_index a = agx_register(8, AGX_SIZE_32);
      agx_index c = agx_register(16, AGX_SIZE_32);
      c.neg = true;
      agx_fcmpsel_to(b, agx_register(2, AGX_SIZE_16), a, c,
                     agx_register(6, AGX_SIZE_16), agx_immediate(0),
                     AGX_FCOND_GE);
      agx_stop(b);
      const auto second = pack(b);
      const auto plain = build(false, AGX_FCOND_GE);
      ASSERT_EQ(second.size(), plain.size());
      unsigned differing = 0;
      for (unsigned i = 0; i < second.size(); ++i)
         differing += second[i] != plain[i];
      EXPECT_EQ(differing, 1u);
   }

   /* 19x  bitop [d32:reg s0=32:reg.neg s1=16:imm cond=0 chan=1]
    * A value move carrying a folded fneg or fabs. agx_lower_pseudo() now puts
    * the float move back rather than handing the bitwise encoding a modifier
    * it has no field for. */
   {
      for (unsigned k = 0; k < 3; ++k) {
         agx_builder *b = builder(enc);
         agx_index a = agx_register(8, AGX_SIZE_32);
         a.neg = k != 1;
         a.abs = k != 0;
         agx_mov_to(b, agx_register(0, AGX_SIZE_32), a);
         agx_stop(b);
         agx_lower_pseudo(b->shader);
         EXPECT_FALSE(pack(b).empty()) << "k=" << k;
      }
   }

   /* 9x  bitop [d32:reg s0=32:reg.discard s1=16:imm cond=0 chan=1]
    * A bitwise operation against an immediate. The operand kind is f37 on the
    * register form's own encoding point, with the constant in f29. */
   {
      const unsigned tables[] = {AGX3_SEL_TT_AND, AGX3_SEL_TT_OR,
                                 AGX3_SEL_TT_XOR};
      for (unsigned i = 0; i < ARRAY_SIZE(tables); ++i) {
         agx_builder *b = builder(enc);
         agx_index a = agx_register(0, AGX_SIZE_32);
         a.discard = true;
         agx_bitop_to(b, agx_register(0, AGX_SIZE_32), a, agx_immediate(0xff),
                      tables[i]);
         agx_stop(b);
         EXPECT_FALSE(pack(b).empty()) << "truth table " << tables[i];
      }
      /* Apple's own `a & 0xff`, with the load's wait on it. */
      uint8_t bytes[AGX3_ALU_MAX_LENGTH];
      uint8_t len = 0;
      ASSERT_TRUE(agx3_alu_bit_imm(enc, AGX3_ALU_BAND, false, 0, 0, 0xff,
                                   AGX3_ALU_HINT_DISCARD, 0x20, bytes, &len));
      EXPECT_EQ(std::vector<uint8_t>(bytes, bytes + len),
                (std::vector<uint8_t>{0x0b, 0x01, 0x1f, 0xff, 0x10, 0x02, 0x00,
                                      0x80, 0x00, 0x00}));
   }
}
