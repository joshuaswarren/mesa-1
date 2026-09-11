/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * The traversal stack is a fixed-size array indexed by a runtime pointer, so
 * the only thing standing between a too-deep tree and an out-of-bounds access
 * is the guard in hk_build_push_stack().  These cases pin that guard down:
 * that it exists, that it wraps the store rather than sitting next to it, and
 * that the bound comes from the array the caller supplied rather than a
 * constant that silently stops matching when someone resizes the stack.
 */

#include "hk_bvh.h"
#include "hk_nir_ray_tracing.h"

#include "compiler/nir/nir_builder.h"

#include <stdio.h>

struct traversal_shader {
   nir_shader *nir;
   nir_variable *stack;
   nir_variable *stack_ptr;
};

/*
 * Build a shader containing nothing but one traversal loop over a stack of
 * exactly `stack_entries` slots.  Both leaf callbacks are left NULL: the leaf
 * cases are not what is under test here, and omitting them keeps the shader
 * small enough to walk exhaustively.
 */
static struct traversal_shader
build_traversal(unsigned stack_entries)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, NULL, "Honeykrisp ray traversal stack bound");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   const struct glsl_type *vec3_type = glsl_vector_type(GLSL_TYPE_FLOAT, 3);
   const struct glsl_type *stack_type =
      glsl_array_type(glsl_uint_type(), stack_entries, 0);

   nir_variable *tmax =
      nir_local_variable_create(b.impl, glsl_float_type(), "tmax");
   nir_variable *origin =
      nir_local_variable_create(b.impl, vec3_type, "origin");
   nir_variable *dir = nir_local_variable_create(b.impl, vec3_type, "dir");
   nir_variable *inv_dir =
      nir_local_variable_create(b.impl, vec3_type, "inv_dir");
   nir_variable *bvh_base =
      nir_local_variable_create(b.impl, glsl_uint64_t_type(), "bvh_base");
   nir_variable *current_node =
      nir_local_variable_create(b.impl, glsl_uint_type(), "current_node");
   nir_variable *stack_base =
      nir_local_variable_create(b.impl, glsl_uint_type(), "stack_base");
   nir_variable *stack_ptr =
      nir_local_variable_create(b.impl, glsl_uint_type(), "stack_ptr");
   nir_variable *stack = nir_local_variable_create(b.impl, stack_type, "stack");
   nir_variable *instance_addr =
      nir_local_variable_create(b.impl, glsl_uint64_t_type(), "instance_addr");
   nir_variable *sbt_offset_and_flags = nir_local_variable_create(
      b.impl, glsl_uint_type(), "sbt_offset_and_flags");

   /* Derive the ray from an invocation ID so nothing folds away: a constant
    * ray would let the box test resolve at build time and delete the very
    * push being tested. */
   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *fgid = nir_u2f32(&b, gid);
   nir_def *scalar = nir_channel(&b, gid, 0);

   nir_store_var(&b, current_node, nir_imm_int(&b, HK_BVH_ROOT_NODE), 0x1);
   nir_store_var(&b, stack_ptr, nir_imm_int(&b, 0), 0x1);
   nir_store_var(&b, stack_base, nir_imm_int(&b, -1), 0x1);
   nir_store_var(&b, tmax, nir_channel(&b, fgid, 0), 0x1);
   nir_store_var(&b, origin, fgid, 0x7);
   nir_store_var(&b, dir, fgid, 0x7);
   nir_store_var(&b, inv_dir, nir_frcp(&b, fgid), 0x7);
   nir_store_var(&b, bvh_base, nir_u2u64(&b, scalar), 0x1);

   struct hk_ray_traversal_args args = {
      .root_bvh_base = nir_u2u64(&b, scalar),
      .flags = scalar,
      .cull_mask = nir_ishl_imm(&b, scalar, 24),
      .origin = fgid,
      .tmin = nir_channel(&b, fgid, 0),
      .dir = fgid,
      .vars =
         {
            .tmax = nir_build_deref_var(&b, tmax),
            .origin = nir_build_deref_var(&b, origin),
            .dir = nir_build_deref_var(&b, dir),
            .inv_dir = nir_build_deref_var(&b, inv_dir),
            .bvh_base = nir_build_deref_var(&b, bvh_base),
            .current_node = nir_build_deref_var(&b, current_node),
            .stack_base = nir_build_deref_var(&b, stack_base),
            .stack_ptr = nir_build_deref_var(&b, stack_ptr),
            .stack = nir_build_deref_var(&b, stack),
            .instance_addr = nir_build_deref_var(&b, instance_addr),
            .sbt_offset_and_flags =
               nir_build_deref_var(&b, sbt_offset_and_flags),
         },
      .aabb_cb = NULL,
      .triangle_cb = NULL,
      .data = NULL,
   };

   hk_build_ray_traversal(&b, &args);

   nir_validate_shader(b.shader, "after hk_build_ray_traversal");

   return (struct traversal_shader){
      .nir = b.shader,
      .stack = stack,
      .stack_ptr = stack_ptr,
   };
}

/* Whether this instruction sits inside at least one nir_if. */
static bool
inside_an_if(const nir_instr *instr)
{
   for (const nir_cf_node *node = &instr->block->cf_node; node;
        node = node->parent) {
      if (node->type == nir_cf_node_if)
         return true;
   }
   return false;
}

/* The variable a deref chain ultimately roots at, or NULL. */
static const nir_variable *
deref_root_var(nir_deref_instr *deref)
{
   while (deref && deref->deref_type != nir_deref_type_var)
      deref = nir_deref_instr_parent(deref);

   return deref ? deref->var : NULL;
}

/* Whether this value is a plain load of `var`. */
static bool
is_load_of(nir_def *def, const nir_variable *var)
{
   if (!nir_def_is_intrinsic(def))
      return false;

   nir_intrinsic_instr *intr = nir_def_as_intrinsic(def);
   if (intr->intrinsic != nir_intrinsic_load_deref)
      return false;

   return deref_root_var(nir_src_as_deref(intr->src[0])) == var;
}

static bool
check_case(unsigned stack_entries)
{
   struct traversal_shader shader = build_traversal(stack_entries);

   unsigned guards = 0;      /* stack-pointer compares against the capacity */
   unsigned wrong_bound = 0; /* ...against anything else */
   unsigned stack_stores = 0;
   unsigned unguarded_stores = 0;

   nir_foreach_function_impl (impl, shader.nir) {
      nir_foreach_block (block, impl) {
         nir_foreach_instr (instr, block) {
            if (instr->type == nir_instr_type_alu) {
               nir_alu_instr *alu = nir_instr_as_alu(instr);

               /* Only compares of the stack pointer itself count.  The
                * instance case also emits a ult against a constant (the
                * cull-mask test against 1 << 24) and that is not a bound. */
               if (alu->op != nir_op_ult ||
                   !is_load_of(alu->src[0].src.ssa, shader.stack_ptr))
                  continue;

               if (!nir_src_is_const(alu->src[1].src))
                  continue;

               if (nir_src_comp_as_uint(alu->src[1].src,
                                        alu->src[1].swizzle[0]) ==
                   stack_entries)
                  guards++;
               else
                  wrong_bound++;
               continue;
            }

            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_store_deref)
               continue;

            if (deref_root_var(nir_src_as_deref(intr->src[0])) != shader.stack)
               continue;

            stack_stores++;
            if (!inside_an_if(instr))
               unguarded_stores++;
         }
      }
   }

   /* The traversal pushes in exactly one place, so one guarded store and one
    * comparison against this stack's own size.  wrong_bound catches the
    * regression this test exists for: a guard that kept comparing against a
    * hardcoded HK_MAX_TLAS_DEPTH + HK_MAX_BLAS_DEPTH would pass the "a guard
    * exists" check while silently not bounding a differently sized stack. */
   const bool ok = guards == 1 && wrong_bound == 0 && stack_stores == 1 &&
                   unguarded_stores == 0;

   if (!ok) {
      fprintf(stderr,
              "stack_entries=%u: guards=%u wrong_bound=%u stack_stores=%u "
              "unguarded_stores=%u\n",
              stack_entries, guards, wrong_bound, stack_stores,
              unguarded_stores);
   }

   ralloc_free(shader.nir);
   return ok;
}

int
main(void)
{
   /* The size the ray-query lowering actually allocates, plus two sizes it
    * does not, so a constant that happens to equal the real one cannot pass. */
   const unsigned sizes[] = {HK_MAX_TLAS_DEPTH + HK_MAX_BLAS_DEPTH, 8, 96};

   /* Building glsl_types outside a driver means owning the type cache. */
   glsl_type_singleton_init_or_ref();

   bool ok = true;
   for (unsigned i = 0; i < ARRAY_SIZE(sizes); ++i)
      ok &= check_case(sizes[i]);

   glsl_type_singleton_decref();

   return ok ? 0 : 1;
}
