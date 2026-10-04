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
      nir_def *value = nir_load_preamble(&b, 4, 16, .base = 4);
      nir_store_local_pixel_agx(&b, value, nir_imm_intN_t(&b, 0xff, 16),
                                nir_undef(&b, 2, 16), .base = int(8 * i),
                                .write_mask = 0xf,
                                .format = PIPE_FORMAT_R16_FLOAT);
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
