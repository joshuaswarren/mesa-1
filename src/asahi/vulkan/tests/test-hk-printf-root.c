/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_cmd_buffer.h"
#include "hk_buffer.h"
#include "hk_shader.h"

#include "nir_builder.h"
#include "vk_pipeline.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>

static bool
source_is_constant(nir_src src, uint64_t expected)
{
   if (nir_src_is_const(src))
      return nir_src_as_uint(src) == expected;

   if (!nir_def_is_alu(src.ssa))
      return false;

   nir_alu_instr *alu = nir_def_as_alu(src.ssa);
   return (alu->op == nir_op_u2u64 || alu->op == nir_op_i2i64) &&
          nir_src_is_const(alu->src[0].src) &&
          nir_src_as_uint(alu->src[0].src) == expected;
}

int
main(void)
{
   assert(hk_buffer_addr_format(
             VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT, false) ==
          nir_address_format_64bit_global_32bit_offset);
   assert(hk_buffer_addr_format(
             VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT, true) ==
          nir_address_format_64bit_bounded_global);
   assert(hk_buffer_addr_format(
             VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_EXT,
             false) == nir_address_format_64bit_bounded_global);
   const struct hk_addr_range null_buffer =
      hk_buffer_addr_range(NULL, 0, VK_WHOLE_SIZE, true);
   assert(null_buffer.addr == AGX_ZERO_PAGE_ADDRESS && null_buffer.range == 0);

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, NULL, "Honeykrisp runtime printf address");
   nir_def *address = nir_load_printf_buffer_address(&b, 64);
   nir_store_global(&b, address, nir_imm_int64(&b, UINT64_C(0x2000000000)),
                    .align_mul = 8);

   const struct vk_pipeline_robustness_state robustness = {
      .storage_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
      .uniform_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
      .vertex_inputs = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
      .images = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_DISABLED_EXT,
   };
   assert(hk_nir_lower_descriptors(b.shader, &robustness, 0, NULL));

   const uint32_t expected = hk_root_descriptor_offset(printf_buffer_address);
   bool saw_root = false;
   bool saw_offset = false;

   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            assert(intr->intrinsic != nir_intrinsic_load_printf_buffer_address);
            saw_root |= intr->intrinsic == nir_intrinsic_load_root_agx;

            if (intr->intrinsic != nir_intrinsic_load_global_constant)
               continue;

            nir_def *pointer = intr->src[0].ssa;
            if (!nir_def_is_alu(pointer))
               continue;

            nir_alu_instr *add = nir_def_as_alu(pointer);
            if (add->op != nir_op_iadd)
               continue;

            for (unsigned i = 0; i < 2; ++i) {
               if (source_is_constant(add->src[i].src, expected))
                  saw_offset = true;
            }
         }
      }
   }

   assert((expected & 7) == 0);
   if (!saw_root || !saw_offset)
      nir_print_shader(b.shader, stderr);
   assert(saw_root && saw_offset);
   ralloc_free(b.shader);
   return 0;
}
