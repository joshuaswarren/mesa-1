/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include <vector>
#include <gtest/gtest.h>
#include "agx3_pack.h"
#include "agx_test.h"
extern "C" {
#include "compiler/nir/nir_builder.h"
#include "agx_nir.h"
}

static std::vector<uint8_t>
image_program(bool marker, bool fragment, bool dynamic_lod, unsigned mask,
              unsigned hint, bool narrow = false, bool register_heap = false)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.fs.ignore_tib_dependencies = true;
   agx_builder *b = agx_test_builder(mem);
   b->shader->key = &key;
   b->shader->stage = fragment ? MESA_SHADER_FRAGMENT : MESA_SHADER_COMPUTE;
   agx_index dest = agx_register(0, AGX_SIZE_32);
   dest.channels_m1 = 3;
   agx_index coord = agx_register(12, AGX_SIZE_32);
   coord.channels_m1 = 1;
   coord.discard = hint == 1;
   coord.cache = hint == 2;
   agx_index index = agx_register(0, narrow ? AGX_SIZE_32 : AGX_SIZE_64);
   index.discard = hint == 1;
   index.cache = hint == 2;
   agx_index lod = dynamic_lod ? agx_register(4, AGX_SIZE_32) : agx_zero();
   lod.discard = dynamic_lod && hint == 1;
   lod.cache = dynamic_lod && hint == 2;
   agx_index base = marker ? agx_zero() : agx_uniform(24, AGX_SIZE_64);
   if (register_heap)
      base = agx_register(24, AGX_SIZE_64);
   agx_instr *I = agx_image_load_to(
      b, dest, coord, lod, base, index, agx_zero(), agx_null(), AGX_DIM_2D,
      dynamic_lod ? AGX_LOD_MODE_LOD_MIN : AGX_LOD_MODE_AUTO_LOD, 0, false,
      false, false);
   I->mask = mask;
   agx_stop(b);
   util_dynarray bin;
   util_dynarray_init(&bin, mem);
   bool ok = agx3_pack_binary(b->shader, &bin);
   EXPECT_EQ(ok, !narrow && !register_heap);
   EXPECT_EQ(I->src[3].value, index.value);
   EXPECT_EQ(I->src[3].size, index.size);
   EXPECT_EQ(I->src[3].discard, index.discard);
   EXPECT_EQ(I->src[3].cache, index.cache);
   std::vector<uint8_t> bytes;
   if (bin.size)
      bytes.assign((uint8_t *)bin.data, (uint8_t *)bin.data + bin.size);
   ralloc_free(mem);
   return bytes;
}
TEST(AGX3CProgCleanup, NativeImageMarkerByteEquivalent)
{
   for (bool fragment : {false, true})
      for (bool dynamic_lod : {false, true})
         for (unsigned mask : {1u, 15u})
            for (unsigned hint : {0u, 1u, 2u}) {
               auto before =
                  image_program(false, fragment, dynamic_lod, mask, hint);
               auto after =
                  image_program(true, fragment, dynamic_lod, mask, hint);
               ASSERT_FALSE(before.empty());
               EXPECT_EQ(before, after);
            }
}
TEST(AGX3CProgCleanup, InvalidImageOperandsStayRejected)
{
   EXPECT_TRUE(image_program(false, false, false, 15, false, true).empty());
   EXPECT_TRUE(image_program(true, false, false, 15, false, true).empty());
   EXPECT_TRUE(
      image_program(false, false, false, 15, false, false, true).empty());
}

static void
keep_load(nir_builder *b, unsigned base, unsigned bits = 32)
{
   nir_def *v = nir_load_preamble(b, 1, bits, .base = (int)base);
   nir_store_uvs_agx(b, nir_u2u32(b, v), nir_imm_int(b, base));
}
TEST(AGX3CProgCleanup, DeadExportsKeepBranchesAndLiveSlots)
{
   nir_builder n = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "dead preamble exports");
   keep_load(&n, 40);
   nir_def *handle =
      nir_bindless_image_agx(&n, nir_imm_int64(&n, 1), .desc_set = 48);
   (void)handle; /* Implicit heap reads conservatively count even before DCE. */
   nir_builder p = nir_builder_create(nir_shader_get_preamble(n.shader));
   p.cursor = nir_after_impl(p.impl);
   nir_def *input = nir_load_preamble(&p, 1, 32, .base = 12);
   nir_def *cond = nir_ieq_imm(&p, input, 0);
   nir_store_preamble(&p, cond, .base = 32);
   nir_if *branch = nir_push_if(&p, cond);
   nir_store_preamble(&p, cond, .base = 33);
   nir_pop_if(&p, branch);
   nir_store_preamble(&p, nir_imm_int(&p, 9), .base = 40);
   nir_store_preamble(&p, nir_imm_int64(&p, 9),
                      .base = 38); /* Partial live overlap. */
   nir_store_preamble(&p, nir_imm_int64(&p, 9),
                      .base = 48); /* Implicit heap read. */
   nir_store_preamble(&p, nir_imm_int(&p, 9), .base = 4); /* Reserved ABI. */
   nir_def *own = nir_load_preamble(&p, 1, 32, .base = 52);
   nir_store_preamble(&p, own, .base = 40);
   nir_store_preamble(&p, nir_imm_int(&p, 9),
                      .base = 52); /* Same-CProg read. */
   nir_store_preamble(&p, nir_imm_int(&p, 9), .base = 32,
                      .preamble_class = nir_preamble_class_image);
   nir_store_preamble(&p, nir_imm_int(&p, 9), .base = 33,
                      .preamble_class = nir_preamble_class_sampler);
   ASSERT_TRUE(agx_nir_remove_unused_preamble_exports(n.shader, 16));
   EXPECT_EQ(branch->condition.ssa, cond);
   unsigned general = 0, other = 0;
   nir_foreach_block(block, p.impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         auto *I = nir_instr_as_intrinsic(instr);
         if (I->intrinsic != nir_intrinsic_store_preamble)
            continue;
         if (nir_intrinsic_preamble_class(I) != nir_preamble_class_general) {
            other++;
            continue;
         }
         EXPECT_NE(nir_intrinsic_base(I), 32u);
         EXPECT_NE(nir_intrinsic_base(I), 33u);
         general++;
      }
   }
   EXPECT_EQ(general, 6u);
   EXPECT_EQ(other, 2u);
   EXPECT_FALSE(agx_nir_remove_unused_preamble_exports(n.shader, 16));
   nir_validate_shader(n.shader, "dead exports preserve control flow");
   ralloc_free(n.shader);
}
TEST(AGX3CProgCleanup, NoPreambleRemainsAbsent)
{
   nir_builder n = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "no preamble");
   EXPECT_FALSE(agx_nir_remove_unused_preamble_exports(n.shader, 16));
   EXPECT_EQ(nir_shader_get_entrypoint(n.shader)->preamble, nullptr);
   ralloc_free(n.shader);
}
TEST(AGX3CProgCleanup, UnknownReaderLeavesExportsUntouched)
{
   nir_builder n = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "unknown preamble range");
   keep_load(&n, 511, 64);
   nir_builder p = nir_builder_create(nir_shader_get_preamble(n.shader));
   p.cursor = nir_after_impl(p.impl);
   nir_store_preamble(&p, nir_imm_int(&p, 9), .base = 32);
   EXPECT_FALSE(agx_nir_remove_unused_preamble_exports(n.shader, 16));
   EXPECT_FALSE(exec_list_is_empty(&nir_start_block(p.impl)->instr_list));
   ralloc_free(n.shader);
}
