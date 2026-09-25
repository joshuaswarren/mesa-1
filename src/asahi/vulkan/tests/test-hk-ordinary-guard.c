/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "hk_ordinary_descriptor.h"
#include "nir_builder.h"
#include <stdio.h>
#include <stdlib.h>

#define CHECK(c) do { if (!(c)) { \
   fprintf(stderr, "ordinary guard: line %u: %s\n", __LINE__, #c); abort(); \
} } while (0)

static void
check_mode(enum hk_ordinary_guard_mode mode)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, NULL, "ordinary descriptor guard");
   nir_def *condition = nir_load_front_face(&b, 1);
   nir_def *guard = nir_load_ordinary_descriptor_guard_agx(&b, condition);
   nir_if *nif = nir_push_if(&b, guard);
   nir_store_global(&b, nir_imm_int(&b, 111), nir_imm_int64(&b, 0x1000), .align_mul = 4);
   nir_push_else(&b, nif);
   nir_store_global(&b, nir_imm_int(&b, 222), nir_imm_int64(&b, 0x1000), .align_mul = 4);
   nir_pop_if(&b, nif);
   CHECK(hk_nir_lower_ordinary_descriptor_guard(b.shader, mode));
   nir_opt_dead_cf(b.shader);
   nir_opt_dce(b.shader);
   nir_validate_shader(b.shader, "after ordinary guard specialization");
   unsigned stores = 0, values = 0;
   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            CHECK(intr->intrinsic != nir_intrinsic_load_ordinary_descriptor_guard_agx);
            if (intr->intrinsic == nir_intrinsic_store_global) {
               unsigned value = nir_src_as_uint(intr->src[0]);
               CHECK(value == 111 || value == 222);
               values |= value == 111 ? 1 : 2;
               ++stores;
            }
         }
      }
   }
   CHECK(stores == (mode == HK_ORDINARY_GUARD_DYNAMIC ? 2 : 1));
   CHECK(values == (mode == HK_ORDINARY_GUARD_DYNAMIC ? 3 :
                    mode == HK_ORDINARY_GUARD_TRUE ? 1 : 2));
   ralloc_free(b.shader);
}

int main(void)
{
   check_mode(HK_ORDINARY_GUARD_DYNAMIC);
   check_mode(HK_ORDINARY_GUARD_TRUE);
   check_mode(HK_ORDINARY_GUARD_FALSE);
   puts("ordinary guard: dynamic preserves both effects; true/false preserve only the selected effect");
   return 0;
}
