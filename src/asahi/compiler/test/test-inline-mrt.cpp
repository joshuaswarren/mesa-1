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
#include <algorithm>

static std::vector<uint8_t>
compile_mrt(bool secondary, unsigned targets, bool *ok, bool no_stop = false)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "standalone MRT compilation");
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.secondary = secondary;
   key.no_stop = no_stop;
   key.no_preamble = true;
   key.reserved_preamble = 8;
   key.fs.nr_samples = 1;
   for (unsigned i = 0; i < targets; ++i) {
      nir_def *value = agx_test_load_preamble(&b, 4, 16, 4);
      agx_test_store_local_pixel_agx(&b, value, nir_imm_intN_t(&b, 0xff, 16),
                                     nir_undef(&b, 2, 16), int(8 * i), 0xf,
                                     PIPE_FORMAT_R16_FLOAT);
      b.shader->info.outputs_written |= BITFIELD64_BIT(FRAG_RESULT_DATA0 + i);
      BITSET_SET(key.g17p_tile_partitions, 8 * i);
   }
   agx_preprocess_nir(b.shader);
   agx_shader_part part = {};
   *ok = agx_compile_shader_nir(b.shader, &key, &part);
   std::vector<uint8_t> bytes;
   if (*ok)
      bytes.assign(
         (uint8_t *)part.binary + part.info.main_offset,
         (uint8_t *)part.binary + part.info.main_offset + part.info.main_size);
   free(part.binary);
   return bytes;
}

TEST(InlineMRT, StandaloneMatchesExistingSecondaryProtocol)
{
   for (unsigned targets : {2u, 3u, 4u, 5u, 6u, 7u, 8u}) {
      SCOPED_TRACE(targets);
      bool secondary_ok = false, primary_ok = false;
      auto secondary = compile_mrt(true, targets, &secondary_ok);
      auto primary = compile_mrt(false, targets, &primary_ok);
      ASSERT_TRUE(secondary_ok);
      ASSERT_TRUE(primary_ok);
      EXPECT_EQ(primary, secondary);
   }
}

TEST(InlineMRT, IncompleteMainPartStillRejected)
{
   bool ok = true;
   auto bytes = compile_mrt(false, 2, &ok, true);
   EXPECT_FALSE(ok);
}

TEST(Background, M5MultisampleTileTransaction)
{
   for (unsigned samples : {2u, 4u, 8u}) {
      SCOPED_TRACE(samples);
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "M5 background transaction");
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
      key.dev.chip = AGX_CHIP_G17S;
      key.no_preamble = true;
      key.reserved_preamble = 8;
      key.fs.background = true;
      key.fs.nr_samples = samples;
      nir_def *value = nir_load_preamble(&b, 4, 32, .base = 4);
      nir_store_local_pixel_agx(
         &b, value, nir_u2u16(&b, nir_load_sample_mask(&b)),
         nir_undef(&b, 2, 16), .write_mask = 0xf,
         .format = PIPE_FORMAT_R8_UNORM);
      b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
      agx_preprocess_nir(b.shader);
      agx_shader_part part = {};
      ASSERT_TRUE(agx_compile_shader_nir(b.shader, &key, &part));
      const uint8_t acquire[] = {0x67, 0x40, 0, 5, 0, 0};
      const uint8_t release[] = {0x67, 0x10, 0, 4, 0, 0};
      const uint8_t *begin = (uint8_t *)part.binary + part.info.main_offset;
      const uint8_t *end = begin + part.info.main_size;
      const uint8_t *wait = std::search(begin, end, std::begin(acquire),
                                       std::end(acquire));
      const uint8_t *signal = std::search(begin, end, std::begin(release),
                                         std::end(release));
      EXPECT_NE(wait, end);
      EXPECT_NE(signal, end);
      EXPECT_LT(wait, signal);
      free(part.binary);
   }
}
