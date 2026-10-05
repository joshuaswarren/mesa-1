/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "../dxil/d12_nir_sparse_buffer.h"
#include "compiler/glsl_types.h"
#include <assert.h>
#include <stdio.h>

static const nir_shader_compiler_options options = {0};

static void
check(uint64_t offset, uint64_t size, unsigned access, bool resident)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &options, "buffer residency");
   nir_def *status = d12_nir_buffer_residency_status(
      &b, nir_imm_int64(&b, 0), nir_imm_int64(&b, 0xfff8),
      nir_imm_int64(&b, offset), nir_imm_int64(&b, size),
      nir_imm_int(&b, access), NULL, nir_imm_int64(&b, 0x10000));
   nir_store_ssbo(&b, status, nir_imm_int(&b, 0), nir_imm_int(&b, 0),
                  .align_mul = 4);
   bool progress;
   do {
      progress = nir_opt_constant_folding(b.shader);
      progress |= nir_opt_algebraic(b.shader);
   } while (progress);
   unsigned stores = 0;
   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_store_ssbo) {
               assert(nir_src_is_const(intr->src[0]));
               assert(nir_src_as_uint(intr->src[0]) == (resident ? UINT32_MAX : 0));
               stores++;
            }
         }
      }
   }
   assert(stores == 1);
   nir_validate_shader(b.shader, "shared sparse-buffer status");
   ralloc_free(b.shader);
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   check(0, 0, 4, false);
   check(0, 16, 16, true);
   check(1, 16, 16, false);
   check(12, 16, 4, true);
   check(13, 16, 4, false);
   check(UINT64_MAX, 16, 16, false);
   check(UINT64_MAX - 4, UINT64_MAX, 16, false);
   glsl_type_singleton_decref();
   puts("d12_nir_sparse_buffer_tests: PASS");
   return 0;
}
