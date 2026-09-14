/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_nir_widen_16bit_varyings.h"

#include <stdio.h>

#include "compiler/nir/nir_builder.h"

static unsigned checks, failures;

#define CHECK(cond)                                                            \
   do {                                                                        \
      ++checks;                                                                \
      if (!(cond)) {                                                           \
         ++failures;                                                           \
         fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #cond);            \
      }                                                                        \
   } while (0)

static nir_shader *
shader(mesa_shader_stage stage)
{
   nir_builder b = nir_builder_init_simple_shader(stage, NULL, "widen");
   return b.shader;
}

static nir_intrinsic_instr *
find(nir_shader *nir, nir_intrinsic_op op)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == op)
               return nir_instr_as_intrinsic(instr);
         }
      }
   }
   return NULL;
}

/* The only use of a def, if it is an ALU instruction. */
static nir_alu_instr *
alu_user(nir_def *def)
{
   nir_foreach_use(src, def) {
      nir_instr *user = nir_src_parent_instr(src);
      if (user->type == nir_instr_type_alu)
         return nir_instr_as_alu(user);
   }
   return NULL;
}

static const nir_io_semantics var0 = {.location = VARYING_SLOT_VAR0,
                                      .num_slots = 1};

/* A 16-bit value an output store widens to 32 bits, with the conversion its
 * base type asks for. */
static void
test_store(mesa_shader_stage stage, nir_alu_type type, bool widened,
           nir_op conversion)
{
   nir_shader *nir = shader(stage);
   nir_builder b = nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(nir)));
   nir_def *value = type == nir_type_float
                       ? nir_imm_floatN_t(&b, 2.0, 16)
                       : nir_imm_intN_t(&b, -2, 16);
   nir_store_output(&b, value, nir_imm_int(&b, 0), .io_semantics = var0,
                    .src_type = type | 16);

   CHECK(hk_nir_widen_16bit_varyings(nir) == widened);
   nir_intrinsic_instr *store = find(nir, nir_intrinsic_store_output);
   CHECK(store && store->src[0].ssa->bit_size == (widened ? 32 : 16));
   CHECK(store && nir_intrinsic_src_type(store) == (type | (widened ? 32 : 16)));
   if (widened && store) {
      nir_instr *parent = nir_def_instr(store->src[0].ssa);
      CHECK(parent->type == nir_instr_type_alu &&
            nir_instr_as_alu(parent)->op == conversion);
   }
   ralloc_free(nir);
}

/* A 16-bit input a consumer reads 32 bits of and narrows. */
static void
test_load_input(mesa_shader_stage stage, nir_alu_type type, bool widened,
                nir_op conversion)
{
   nir_shader *nir = shader(stage);
   nir_builder b = nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(nir)));
   nir_def *zero = nir_imm_int(&b, 0);
   nir_def *load =
      stage == MESA_SHADER_TESS_EVAL
         ? nir_load_per_vertex_input(&b, 2, 16, zero, zero,
                                     .io_semantics = var0,
                                     .dest_type = type | 16)
         : nir_load_input(&b, 2, 16, zero, .io_semantics = var0,
                          .dest_type = type | 16);
   nir_fneg(&b, nir_u2f32(&b, load));

   CHECK(hk_nir_widen_16bit_varyings(nir) == widened);
   CHECK(load->bit_size == (widened ? 32 : 16));
   if (widened) {
      nir_alu_instr *narrow = alu_user(load);
      CHECK(narrow && narrow->op == conversion &&
            narrow->def.bit_size == 16);
   }
   ralloc_free(nir);
}

/* A 16-bit output read back, as a tessellation control shader reads the
 * outputs it wrote and a fragment shader reads its render target. */
static void
test_load_output(mesa_shader_stage stage, nir_alu_type type, bool widened,
                 nir_op conversion)
{
   nir_shader *nir = shader(stage);
   nir_builder b = nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(nir)));
   nir_def *zero = nir_imm_int(&b, 0);
   nir_io_semantics sem = var0;
   if (stage == MESA_SHADER_FRAGMENT)
      sem.location = FRAG_RESULT_DATA0;
   nir_def *load =
      stage == MESA_SHADER_TESS_CTRL
         ? nir_load_per_vertex_output(&b, 2, 16, zero, zero,
                                      .io_semantics = sem,
                                      .dest_type = type | 16)
         : nir_load_output(&b, 2, 16, zero, .io_semantics = sem,
                           .dest_type = type | 16);
   nir_fneg(&b, nir_u2f32(&b, load));

   CHECK(hk_nir_widen_16bit_varyings(nir) == widened);
   CHECK(load->bit_size == (widened ? 32 : 16));
   if (widened) {
      nir_alu_instr *narrow = alu_user(load);
      CHECK(narrow && narrow->op == conversion &&
            narrow->def.bit_size == 16);
   }
   ralloc_free(nir);
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();

   test_store(MESA_SHADER_VERTEX, nir_type_float, true, nir_op_f2f32);
   test_store(MESA_SHADER_TESS_CTRL, nir_type_int, true, nir_op_i2i32);
   test_store(MESA_SHADER_TESS_EVAL, nir_type_uint, true, nir_op_u2u32);
   test_store(MESA_SHADER_FRAGMENT, nir_type_float, false, nir_op_f2f32);

   test_load_input(MESA_SHADER_TESS_EVAL, nir_type_float, true, nir_op_f2f16);
   test_load_input(MESA_SHADER_FRAGMENT, nir_type_int, true, nir_op_i2i16);
   test_load_input(MESA_SHADER_VERTEX, nir_type_float, false, nir_op_f2f16);

   test_load_output(MESA_SHADER_TESS_CTRL, nir_type_float, true, nir_op_f2f16);
   test_load_output(MESA_SHADER_TESS_CTRL, nir_type_int, true, nir_op_i2i16);
   test_load_output(MESA_SHADER_FRAGMENT, nir_type_float, false,
                    nir_op_f2f16);

   glsl_type_singleton_decref();

   if (failures) {
      fprintf(stderr, "hk-nir-widen-16bit-varyings: %u/%u checks failed\n",
              failures, checks);
      return 1;
   }
   printf("hk-nir-widen-16bit-varyings: %u checks passed\n", checks);
   return 0;
}
