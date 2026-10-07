/*
 * Copyright 2026 Joshua Warren
 * SPDX-License-Identifier: MIT
 */

#include "nir_builder.h"

extern "C" {
#include "agx_nir.h"
}

#include <gtest/gtest.h>

/*
 * A load whose address is a lea with a shift below the format's shift (the
 * byte offsets the cooperative-matrix loads produce) keeps the base and,
 * with fold_subformat_address, indexes by the offset scaled down to elements.
 * Returns the lowered load_agx's constant index, or INT64_MIN if the load is
 * not indexed by a constant.
 */
static int64_t
lowered_index(bool fold, nir_op lea, int32_t byte_offset, unsigned bit_size,
              bool *same_base = nullptr)
{
   static const nir_shader_compiler_options options = {};
   glsl_type_singleton_init_or_ref();
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &options, "address test");
   nir_def *base = nir_load_push_constant(&b, 1, 64, nir_imm_int(&b, 0));
   nir_def *addr = nir_build_alu3(&b, lea, base, nir_imm_int(&b, byte_offset),
                                  nir_imm_int(&b, 0));
   nir_def *v = nir_load_global(&b, 2, bit_size, addr, .align_mul = 16);
   nir_store_global(&b, v, nir_imm_int64(&b, 0x200000), .align_mul = 16);

   agx_nir_lower_address(b.shader, fold);
   nir_opt_constant_folding(b.shader);
   nir_validate_shader(b.shader, "after agx_nir_lower_address");

   int64_t index = INT64_MIN;
   nir_foreach_block(block, b.impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_load_agx)
            continue;
         if (same_base)
            *same_base = intr->src[0].ssa == base;
         if (nir_src_is_const(intr->src[1]))
            index = nir_src_as_int(intr->src[1]);
      }
   }
   ralloc_free(b.shader);
   glsl_type_singleton_decref();
   return index;
}

TEST(LowerAddress, FoldScalesByteOffsetToElements)
{
   bool same_base = false;
   EXPECT_EQ(lowered_index(true, nir_op_ulea_agx, 24, 32, &same_base), 6);
   EXPECT_TRUE(same_base);
   EXPECT_EQ(lowered_index(true, nir_op_ulea_agx, 24, 16), 12);
   EXPECT_EQ(lowered_index(true, nir_op_ilea_agx, -8, 32), -2);
}

TEST(LowerAddress, NoFoldMaterializesTheSum)
{
   bool same_base = true;
   EXPECT_EQ(lowered_index(false, nir_op_ulea_agx, 24, 32, &same_base), 0);
   EXPECT_FALSE(same_base);
}
