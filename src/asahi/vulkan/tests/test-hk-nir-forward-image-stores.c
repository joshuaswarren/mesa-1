/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_nir_forward_image_stores.h"

#include <stdio.h>

#include "compiler/nir/nir_builder.h"
#include "util/format/u_format.h"

enum shape {
   /* imageStore(u0, pos, x); imageStore(u2, pos, imageLoad(u0, pos)) */
   SHAPE_STORE_THEN_LOAD,
   /* The load sits in a later block than the store. */
   SHAPE_STORE_THEN_LOAD_ACROSS_IF,
   SHAPE_LOAD_ONLY,
   SHAPE_FORMATLESS,
   SHAPE_ATOMIC,
};

static nir_variable *
make_image(nir_builder *b, const char *name, enum pipe_format format)
{
   nir_variable *image = nir_variable_create(
      b->shader, nir_var_image,
      glsl_image_type(GLSL_SAMPLER_DIM_2D, false, GLSL_TYPE_UINT), name);
   image->data.image.format = format;
   return image;
}

static nir_shader *
build(enum shape shape)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, NULL, "Honeykrisp image store forwarding");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   /* Image atomics require a single-channel format. */
   const enum pipe_format format =
      shape == SHAPE_FORMATLESS ? PIPE_FORMAT_NONE
      : shape == SHAPE_ATOMIC   ? PIPE_FORMAT_R32_UINT
                                : PIPE_FORMAT_R32G32_UINT;
   nir_variable *u0 = make_image(&b, "u0", format);
   nir_variable *u2 = make_image(&b, "u2", PIPE_FORMAT_R32G32_UINT);

   nir_def *zero = nir_imm_int(&b, 0);
   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *coord = nir_vec4(&b, nir_channel(&b, gid, 0),
                             nir_channel(&b, gid, 1), zero, zero);

   if (shape != SHAPE_LOAD_ONLY) {
      nir_image_deref_store(
         &b, &nir_build_deref_var(&b, u0)->def, coord, zero,
         nir_imm_ivec4(&b, 5, 6, 7, 8), zero,
         .image_dim = GLSL_SAMPLER_DIM_2D, .image_array = false,
         .format = format, .access = ACCESS_RESTRICT,
         .src_type = nir_type_uint32);
   }

   if (shape == SHAPE_ATOMIC) {
      nir_image_deref_atomic(
         &b, 32, &nir_build_deref_var(&b, u0)->def, coord, zero,
         nir_imm_int(&b, 1), .image_dim = GLSL_SAMPLER_DIM_2D,
         .image_array = false, .format = format, .access = ACCESS_RESTRICT,
         .atomic_op = nir_atomic_op_iadd);
   }

   if (shape == SHAPE_STORE_THEN_LOAD_ACROSS_IF)
      nir_push_if(&b, nir_ieq_imm(&b, nir_channel(&b, gid, 0), 0));

   nir_def *loaded = nir_image_deref_load(
      &b, 4, 32, &nir_build_deref_var(&b, u0)->def, coord, zero, zero,
      .image_dim = GLSL_SAMPLER_DIM_2D, .image_array = false,
      .format = format, .access = ACCESS_RESTRICT,
      .dest_type = nir_type_uint32);
   nir_image_deref_store(
      &b, &nir_build_deref_var(&b, u2)->def, coord, zero, loaded, zero,
      .image_dim = GLSL_SAMPLER_DIM_2D, .image_array = false,
      .format = PIPE_FORMAT_R32G32_UINT, .access = ACCESS_RESTRICT,
      .src_type = nir_type_uint32);

   if (shape == SHAPE_STORE_THEN_LOAD_ACROSS_IF)
      nir_pop_if(&b, NULL);

   return b.shader;
}

struct census {
   unsigned stores, loads, sizes, bcsels;
   bool store_consumes_load;
};

static struct census
count(nir_shader *nir)
{
   struct census c = {0};
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_alu &&
                nir_instr_as_alu(instr)->op == nir_op_bcsel)
               c.bcsels++;
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            switch (intr->intrinsic) {
            case nir_intrinsic_image_deref_store: {
               c.stores++;
               nir_instr *src = nir_def_instr(intr->src[3].ssa);
               c.store_consumes_load |=
                  src->type == nir_instr_type_intrinsic &&
                  nir_instr_as_intrinsic(src)->intrinsic ==
                     nir_intrinsic_image_deref_load;
               break;
            }
            case nir_intrinsic_image_deref_load:
               c.loads++;
               break;
            case nir_intrinsic_image_deref_size:
               c.sizes++;
               break;
            default:
               break;
            }
         }
      }
   }
   return c;
}

static bool
check(const char *name, enum shape shape, bool robust, bool expect_progress)
{
   nir_shader *nir = build(shape);
   const bool progress = hk_nir_forward_image_stores(nir, robust);
   nir_validate_shader(nir, "after image store forwarding");
   const struct census c = count(nir);

   bool ok = progress == expect_progress;
   if (expect_progress) {
      /* The stores and the load stay; the second store now takes a select
       * between the recorded value and the load. */
      ok &= c.stores == 2 && c.loads == 1 && c.bcsels > 0 &&
            !c.store_consumes_load && (c.sizes > 0) == robust;
   } else {
      ok &= c.bcsels == 0 && c.sizes == 0 &&
            (shape == SHAPE_LOAD_ONLY || shape == SHAPE_ATOMIC ||
             c.store_consumes_load);
   }

   if (!ok) {
      fprintf(stderr,
              "%s: progress=%u stores=%u loads=%u sizes=%u bcsels=%u "
              "store_consumes_load=%u\n",
              name, progress, c.stores, c.loads, c.sizes, c.bcsels,
              c.store_consumes_load);
      nir_print_shader(nir, stderr);
   }

   ralloc_free(nir);
   return ok;
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();

   bool ok = true;
   ok &= check("store then load", SHAPE_STORE_THEN_LOAD, false, true);
   ok &= check("robust store then load", SHAPE_STORE_THEN_LOAD, true, true);
   ok &= check("store then load across if", SHAPE_STORE_THEN_LOAD_ACROSS_IF,
               false, true);
   ok &= check("load only", SHAPE_LOAD_ONLY, false, false);
   ok &= check("formatless", SHAPE_FORMATLESS, false, false);
   ok &= check("atomic", SHAPE_ATOMIC, false, false);

   glsl_type_singleton_decref();
   return ok ? 0 : 1;
}
