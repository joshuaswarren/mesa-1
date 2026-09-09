/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_nir_lower_robust_image.h"

#include "compiler/nir/nir_builder.h"
#include "util/format/u_format.h"

enum access_kind {
   IMAGE_ACCESS_LOAD,
   IMAGE_ACCESS_STORE,
   IMAGE_ACCESS_ATOMIC,
};

static nir_shader *
build_image_access(enum access_kind kind)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, NULL, "Honeykrisp robust image lowering");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   const enum pipe_format format = kind == IMAGE_ACCESS_ATOMIC
                                      ? PIPE_FORMAT_R32_UINT
                                      : PIPE_FORMAT_R8G8B8A8_UINT;
   nir_variable *image = nir_variable_create(
      b.shader, nir_var_image,
      glsl_image_type(GLSL_SAMPLER_DIM_2D, false, GLSL_TYPE_UINT), "image");
   image->data.image.format = format;

   nir_def *deref = &nir_build_deref_var(&b, image)->def;
   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *zero = nir_imm_int(&b, 0);
   nir_def *coord = nir_vec4(&b, nir_channel(&b, gid, 0),
                             nir_channel(&b, gid, 1), zero, zero);

   if (kind == IMAGE_ACCESS_LOAD) {
      nir_def *value = nir_image_deref_load(
         &b, 4, 32, deref, coord, zero, zero,
         .image_dim = GLSL_SAMPLER_DIM_2D, .image_array = false,
         .format = format, .access = ACCESS_RESTRICT,
         .dest_type = nir_type_uint32);
      nir_store_scratch(&b, value, zero, .write_mask = 0xf, .align_mul = 16);
   } else if (kind == IMAGE_ACCESS_STORE) {
      nir_image_deref_store(
         &b, deref, coord, zero, nir_imm_ivec4(&b, 1, 2, 3, 4), zero,
         .image_dim = GLSL_SAMPLER_DIM_2D, .image_array = false,
         .format = format, .access = ACCESS_RESTRICT,
         .src_type = nir_type_uint32);
   } else {
      nir_def *value = nir_image_deref_atomic(
         &b, 32, deref, coord, zero, nir_imm_int(&b, 1),
         .image_dim = GLSL_SAMPLER_DIM_2D, .image_array = false,
         .format = format, .access = ACCESS_RESTRICT,
         .atomic_op = nir_atomic_op_iadd);
      nir_store_scratch(&b, value, zero, .write_mask = 0x1, .align_mul = 4);
   }

   return b.shader;
}

static bool
check_case(enum access_kind kind, bool lower_loads_and_stores)
{
   nir_shader *nir = build_image_access(kind);
   const bool progress =
      hk_nir_lower_robust_image_access(nir, lower_loads_and_stores);

   unsigned accesses = 0, in_bounds_accesses = 0, sizes = 0, phis = 0;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_phi) {
               phis++;
               continue;
            }

            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            switch (intr->intrinsic) {
            case nir_intrinsic_image_deref_load:
               accesses += kind == IMAGE_ACCESS_LOAD;
               in_bounds_accesses +=
                  kind == IMAGE_ACCESS_LOAD &&
                  (nir_intrinsic_access(intr) & ACCESS_IN_BOUNDS);
               break;
            case nir_intrinsic_image_deref_store:
               accesses += kind == IMAGE_ACCESS_STORE;
               in_bounds_accesses +=
                  kind == IMAGE_ACCESS_STORE &&
                  (nir_intrinsic_access(intr) & ACCESS_IN_BOUNDS);
               break;
            case nir_intrinsic_image_deref_atomic:
               accesses += kind == IMAGE_ACCESS_ATOMIC;
               in_bounds_accesses +=
                  kind == IMAGE_ACCESS_ATOMIC &&
                  (nir_intrinsic_access(intr) & ACCESS_IN_BOUNDS);
               break;
            case nir_intrinsic_image_deref_size:
               sizes++;
               break;
            default:
               break;
            }
         }
      }
   }

   const bool lowered =
      kind == IMAGE_ACCESS_ATOMIC || lower_loads_and_stores;
   const bool has_result = kind != IMAGE_ACCESS_STORE;
   const bool ok = progress == lowered && accesses == 1 &&
                   in_bounds_accesses == (lowered ? 1 : 0) &&
                   sizes == (lowered ? 1 : 0) &&
                   phis == (lowered && has_result ? 1 : 0);
   ralloc_free(nir);
   return ok;
}

int
main(void)
{
   for (unsigned kind = IMAGE_ACCESS_LOAD; kind <= IMAGE_ACCESS_ATOMIC;
        ++kind) {
      if (!check_case(kind, false) || !check_case(kind, true))
         return 1;
   }

   return 0;
}
