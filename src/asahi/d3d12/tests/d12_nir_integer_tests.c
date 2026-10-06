/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "../dxil/d12_nir_integer.h"
#include "compiler/glsl_types.h"

#include <assert.h>
#include <stdio.h>

static const nir_shader_compiler_options options = {0};

static void
check(unsigned bits, bool is_signed, bool remainder, uint64_t x, uint64_t y,
      uint64_t expected)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &options, "DXIL divmod");
   nir_def *result = d12_nir_integer_divmod(
      &b, nir_imm_intN_t(&b, x, bits), nir_imm_intN_t(&b, y, bits),
      is_signed, remainder);
   nir_store_ssbo(&b, result, nir_imm_int(&b, 0), nir_imm_int(&b, 0),
                  .align_mul = bits / 8);
   nir_validate_shader(b.shader, "DXIL divmod test");
   while (nir_opt_constant_folding(b.shader)) {}

   unsigned stores = 0;
   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_store_ssbo)
               continue;
            assert(nir_src_is_const(intr->src[0]));
            uint64_t mask = bits == 64 ? UINT64_MAX : (UINT64_C(1) << bits) - 1;
            assert(nir_src_as_uint(intr->src[0]) == (expected & mask));
            stores++;
         }
      }
   }
   assert(stores == 1);
   ralloc_free(b.shader);
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   for (unsigned bits = 16; bits <= 64; bits *= 2) {
      for (unsigned sign = 0; sign < 2; sign++) {
         for (unsigned rem = 0; rem < 2; rem++) {
            check(bits, sign, rem, 0, 0, UINT64_MAX);
            check(bits, sign, rem, 7, 0, UINT64_MAX);
            check(bits, sign, rem, 7, 3, rem ? 1 : 2);
            check(bits, sign, rem, 7, 1, rem ? 0 : 7);
            check(bits, sign, rem, 7, 9, rem ? 7 : 0);
         }
      }
      check(bits, true, false, -7, 3, -2);
      check(bits, true, true, -7, 3, -1);
      check(bits, true, false, 7, -3, -2);
      check(bits, true, true, 7, -3, 1);
      check(bits, false, true, -1, 2, 1);
   }
   glsl_type_singleton_decref();
   puts("d12_nir_integer_tests: PASS");
   return 0;
}
