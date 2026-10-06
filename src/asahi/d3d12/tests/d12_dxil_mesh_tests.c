/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 * CPU-only mesh/task DXIL round trips using Mesa's own MIT shader writer.
 */
#include <assert.h>
#include <stdio.h>
#include "compiler/glsl_types.h"
#include "dxil/d12_dxil.h"
#include "util/blob.h"
#include "d12_native_descriptor_core.h"
#include "nir.h"
#include "nir_builder.h"
#include "nir_to_dxil.h"

static void
check_stage(bool task)
{
   nir_shader_compiler_options options;
   dxil_get_nir_compiler_options(&options, SHADER_MODEL_6_5, 32, 32);
   nir_builder b = nir_builder_init_simple_shader(
      task ? MESA_SHADER_TASK : MESA_SHADER_MESH, &options, "mesh contract");
   b.shader->info.workgroup_size[0] = task ? 8 : 16;
   b.shader->info.workgroup_size[1] = task ? 1 : 8;
   b.shader->info.workgroup_size[2] = 1;
   if (task) {
      b.shader->info.task_payload_size = 4;
      nir_launch_mesh_workgroups(&b, nir_imm_ivec3(&b, 2, 3, 4));
   } else {
      b.shader->info.mesh.max_vertices_out = 128;
      b.shader->info.mesh.max_primitives_out = 210;
      b.shader->info.mesh.primitive_type = MESA_PRIM_TRIANGLES;
      nir_set_vertex_and_primitive_count(
         &b, nir_imm_int(&b, 128), nir_imm_int(&b, 210), nir_imm_int(&b, 0));
   }
   struct nir_to_dxil_options opts = {
      .environment = DXIL_ENVIRONMENT_VULKAN,
      .shader_model_max = SHADER_MODEL_6_5,
      .validator_version_max = DXIL_VALIDATOR_1_6,
   };
   struct blob blob;
   blob_init(&blob);
   assert(nir_to_dxil(b.shader, &opts, NULL, &blob));
   const uint8_t *bc;
   size_t size;
   const char *error = NULL;
   assert(d12_dxil_find_bitcode(blob.data, blob.size, &bc, &size, &error));
   struct d12_module *mod = d12_dxil_module_alloc();
   assert(mod && d12_dxil_module_parse(mod, bc, size));
   struct d12_binding_map map = {0};
   nir_shader *nir = d12_dxil_to_nir(mod, MESA_SHADER_COMPUTE, &map,
                                     D12_NATIVE_DESCRIPTOR_GEN7, &error);
   if (!nir)
      fprintf(stderr, "translation failed: %s\n", error);
   assert(nir);
   assert(nir->info.workgroup_size[0] == (task ? 8 : 16));
   assert(nir->info.workgroup_size[1] == (task ? 1 : 8));
   if (task) {
      assert(nir->info.task_payload_size == 4);
      unsigned barriers = 0;
      nir_foreach_function_impl(impl, nir) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic == nir_intrinsic_barrier &&
                   nir_intrinsic_execution_scope(intr) == SCOPE_WORKGROUP &&
                   (nir_intrinsic_memory_modes(intr) & nir_var_mem_shared))
                  barriers++;
            }
         }
      }
      assert(barriers >= 1);
   } else {
      assert(nir->info.mesh.max_vertices_out == 128);
      assert(nir->info.mesh.max_primitives_out == 210);
      assert(nir->info.derivative_group == DERIVATIVE_GROUP_QUADS);
   }
   d12_dxil_bound_mesh_workgroups(nir, task);
   nir_validate_shader(nir, "mesh bounds on structured translation");
   ralloc_free(nir);
   /* Corrupt the declaration independently of the instruction body. These
    * modules must fail before any out-of-bounds lowering is constructed. */
   mod->mesh_payload_size = D12_MESH_PAYLOAD_STRIDE + 4;
   assert(!d12_dxil_to_nir(mod, MESA_SHADER_COMPUTE, &map,
                           D12_NATIVE_DESCRIPTOR_GEN7, &error));
   mod->mesh_payload_size = 0;
   mod->mesh_max_vertices = 257;
   assert(!d12_dxil_to_nir(mod, MESA_SHADER_COMPUTE, &map,
                           D12_NATIVE_DESCRIPTOR_GEN7, &error));
   mod->mesh_max_vertices = 0;
   mod->mesh_max_primitives = 257;
   assert(!d12_dxil_to_nir(mod, MESA_SHADER_COMPUTE, &map,
                           D12_NATIVE_DESCRIPTOR_GEN7, &error));
   d12_dxil_module_free(mod);
   blob_finish(&blob);
   ralloc_free(b.shader);
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   check_stage(false);
   check_stage(true);
   glsl_type_singleton_decref();
   puts("mesh/task declarations, payload barrier and invalid bounds: PASS");
   return 0;
}
