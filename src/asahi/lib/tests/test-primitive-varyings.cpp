#include <gtest/gtest.h>
#include "asahi/compiler/agx_compile.h"
#include "compiler/nir/nir_builder.h"
extern "C" {
#include "asahi/vulkan/hk_nir_widen_16bit_varyings.h"
}

class PrimitiveVaryings : public ::testing::Test {
protected:
   PrimitiveVaryings() { glsl_type_singleton_init_or_ref(); }
   ~PrimitiveVaryings() { glsl_type_singleton_decref(); }
};

TEST_F(PrimitiveVaryings, MeshStoresWidenBeforeInterstageOptimization)
{
   for (nir_alu_type type : {nir_type_int16, nir_type_uint16, nir_type_float16}) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_MESH, &agx_nir_options, "primitive store widening");
      nir_def *value = nir_imm_intN_t(&b, 0xfc01, 16);
      nir_store_per_primitive_output(&b, value, nir_imm_int(&b, 0),
         nir_imm_int(&b, 0), .write_mask = 1, .src_type = type,
         .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
      ASSERT_TRUE(hk_nir_widen_16bit_varyings(b.shader, HK_NIR_WIDEN_16BIT_OUTPUTS));
      nir_intrinsic_instr *store = nir_instr_as_intrinsic(nir_block_last_instr(nir_cursor_current_block(b.cursor)));
      EXPECT_EQ(store->src[0].ssa->bit_size, 32u);
      EXPECT_EQ(nir_intrinsic_src_type(store), nir_alu_type_get_base_type(type) | 32);
      nir_alu_instr *convert = nir_instr_as_alu(nir_def_instr(store->src[0].ssa));
      EXPECT_EQ(convert->op, type == nir_type_float16 ? nir_op_f2f32 :
                            type == nir_type_int16 ? nir_op_i2i32 : nir_op_u2u32);
      EXPECT_EQ(convert->src[0].src.ssa, value);
      nir_opt_constant_folding(b.shader);
      if (type == nir_type_int16) {
         EXPECT_EQ(nir_src_as_uint(store->src[0]), 0xfffffc01u);
      }
      if (type == nir_type_uint16) {
         EXPECT_EQ(nir_src_as_uint(store->src[0]), 0xfc01u);
      }
      ralloc_free(b.shader);
   }
}

TEST_F(PrimitiveVaryings, PrimitiveLoadsWidenAndNarrowNumerically)
{
   for (bool output : {false, true}) {
      for (nir_alu_type type : {nir_type_int16, nir_type_uint16, nir_type_float16}) {
         nir_builder b = nir_builder_init_simple_shader(
            output ? MESA_SHADER_MESH : MESA_SHADER_FRAGMENT,
            &agx_nir_options, "primitive load widening");
         nir_def *value = output
            ? nir_load_per_primitive_output(&b, 1, 16, nir_imm_int(&b, 0),
                 nir_imm_int(&b, 0), .dest_type = type,
                 .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1})
            : nir_load_per_primitive_input(&b, 1, 16, nir_imm_int(&b, 0),
                 .dest_type = type,
                 .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
         nir_def *use = nir_iadd_imm(&b, value, 1);
         auto mode = output ? HK_NIR_WIDEN_16BIT_OUTPUTS : HK_NIR_WIDEN_16BIT_INPUTS;
         ASSERT_TRUE(hk_nir_widen_16bit_varyings(b.shader, mode));
         EXPECT_EQ(value->bit_size, 32u);
         auto *load = nir_instr_as_intrinsic(nir_def_instr(value));
         EXPECT_EQ(nir_intrinsic_dest_type(load), nir_alu_type_get_base_type(type) | 32);
         auto *user = nir_instr_as_alu(nir_def_instr(use));
         auto *convert = nir_instr_as_alu(nir_def_instr(user->src[0].src.ssa));
         EXPECT_EQ(convert->op, type == nir_type_float16 ? nir_op_f2f16 : nir_op_u2u16);
         EXPECT_EQ(convert->def.bit_size, 16u);
         EXPECT_EQ(convert->src[0].src.ssa, value);
         ralloc_free(b.shader);
      }
   }
}
