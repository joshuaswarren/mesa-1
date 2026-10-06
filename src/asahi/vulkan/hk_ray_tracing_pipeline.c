/*
 * Copyright © 2021 Google
 * Copyright © 2024 Valve Corporation
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * VK_KHR_ray_tracing_pipeline for Honeykrisp.
 *
 * Ported from lavapipe's src/gallium/frontends/lavapipe/lvp_ray_tracing_pipeline.c,
 * which is the same monolithic shape: every stage is inlined into one compute
 * megakernel and the shader binding table picks between the inlined copies with
 * an equality test on a group index.  hk's software traversal
 * (hk_nir_ray_tracing.c) is already a port of lavapipe's, so the two halves fit
 * without reinterpretation.
 *
 * The one real divergence is how the trace-rays record reaches the shader.
 * lavapipe binds it as an internal SSBO; hk has no internal descriptor set, so
 * the record's device address is published in the compute root descriptor table
 * (hk_root_descriptor_table::cs.trace_rays_addr) and the shader reads it with
 * the same load_root_agx + load_global construction hk_nir_lower_descriptors.c
 * uses for every other root sysval.
 */

#include "hk_ray_tracing_pipeline.h"

#include "hk_buffer.h"
#include "hk_bvh.h"
#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_nir_ray_tracing.h"
#include "hk_physical_device.h"
#include "hk_shader.h"

#include "vk_alloc.h"
#include "vk_common_entrypoints.h"
#include "vk_nir.h"
#include "vk_pipeline.h"
#include "vk_pipeline_layout.h"
#include "vk_shader_module.h"

#include "nir.h"
#include "nir_builder.h"

#include "spirv/spirv.h"

#include "util/u_debug.h"
#include "util/u_dynarray.h"

/* ------------------------------------------------------------------------ */
/* Stage NIR refcounting                                                     */
/* ------------------------------------------------------------------------ */

static struct hk_rt_stage_nir *
hk_rt_stage_nir_create(nir_shader *nir)
{
   struct hk_rt_stage_nir *stage = calloc(1, sizeof(*stage));
   if (!stage)
      return NULL;

   stage->ref_cnt = 1;
   stage->nir = nir;
   return stage;
}

static void
hk_rt_stage_nir_ref(struct hk_rt_stage_nir **dst, struct hk_rt_stage_nir *src)
{
   if (src)
      p_atomic_inc(&src->ref_cnt);
   *dst = src;
}

static void
hk_rt_stage_nir_unref(struct hk_rt_stage_nir *stage)
{
   if (!stage)
      return;

   if (p_atomic_dec_zero(&stage->ref_cnt)) {
      ralloc_free(stage->nir);
      free(stage);
   }
}

/* ------------------------------------------------------------------------ */
/* Reading the trace-rays record                                             */
/* ------------------------------------------------------------------------ */

/*
 * The megakernel's only driver-private input.  hk_cmd_trace_rays.c publishes a
 * VkTraceRaysIndirectCommand2KHR and puts its address here; every shader-binding
 * table pointer, stride and the launch size come out of that one record, so the
 * indirect entry points can hand the shader the application's own buffer
 * unchanged.
 */
static nir_def *
hk_load_trace_rays_record(nir_builder *b)
{
   nir_def *root = nir_load_root_agx(b);
   nir_def *addr = nir_iadd_imm(
      b, root, hk_root_descriptor_offset(cs.trace_rays_addr));

   return nir_load_global_constant(b, 1, 64, addr, .align_mul = 8,
                                   .access = ACCESS_CAN_SPECULATE);
}

static nir_def *
hk_load_trace_ray_command_field(nir_builder *b, uint32_t command_offset,
                                uint32_t num_components, uint32_t bit_size)
{
   nir_def *record = hk_load_trace_rays_record(b);

   return nir_load_global_constant(b, num_components, bit_size,
                                   nir_iadd_imm(b, record, command_offset),
                                   .align_mul = MIN2(bit_size / 8, 8));
}

struct hk_sbt_entry {
   nir_def *value;
   nir_def *shader_record_ptr;
};

/*
 * Fetch one shader binding table record.  `command_offset` names the region
 * inside VkTraceRaysIndirectCommand2KHR; `index` is the record index inside it,
 * or NULL for the ray-generation region, which has exactly one record and no
 * stride.
 *
 * The record is laid out as the specification requires: the opaque group handle
 * first (HK_RAY_TRACING_GROUP_HANDLE_SIZE bytes), the application's shader
 * record data immediately after it.
 */
static struct hk_sbt_entry
hk_load_sbt_entry(nir_builder *b, nir_def *index, uint32_t command_offset)
{
   nir_def *addr = hk_load_trace_ray_command_field(b, command_offset, 1, 64);

   if (index) {
      /* A stride is a VkDeviceSize but is bounded by maxShaderGroupStride, so
       * the high 32 bits cannot contribute. */
      nir_def *stride = hk_load_trace_ray_command_field(
         b, hk_rt_sbt_stride_offset(command_offset), 1, 32);
      addr = nir_iadd(b, addr, nir_u2u64(b, nir_imul(b, index, stride)));
   }

   return (struct hk_sbt_entry){
      .value = nir_load_global_constant(
         b, 1, 32,
         nir_iadd_imm(b, addr,
                      offsetof(struct hk_ray_tracing_group_handle, index)),
         .align_mul = 4),
      .shader_record_ptr =
         nir_iadd_imm(b, addr, HK_RAY_TRACING_GROUP_HANDLE_SIZE),
   };
}

/* ------------------------------------------------------------------------ */
/* Per-stage preparation                                                     */
/* ------------------------------------------------------------------------ */

/*
 * Ray payloads (nir_var_shader_call_data) and hit attributes
 * (nir_var_ray_hit_attrib) both live in the invocation's scratch stack.  A
 * payload is addressed relative to the caller's frame, which the megakernel
 * tracks in state->shader_call_data_offset; a hit attribute always sits at the
 * current stack pointer.  Rewriting both to function_temp derefs lets
 * nir_lower_explicit_io turn them into plain scratch access, which the final
 * lowering then biases by the live stack pointer.
 */
static bool
hk_lower_ray_tracing_derefs(nir_shader *shader)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   bool progress = false;

   nir_builder _b = nir_builder_at(nir_before_impl(impl));
   nir_builder *b = &_b;

   nir_def *arg_offset = nir_load_shader_call_data_offset_lvp(b);

   nir_foreach_block (block, impl) {
      nir_foreach_instr_safe (instr, block) {
         if (instr->type != nir_instr_type_deref)
            continue;

         nir_deref_instr *deref = nir_instr_as_deref(instr);
         if (!nir_deref_mode_is_one_of(deref, nir_var_shader_call_data |
                                                 nir_var_ray_hit_attrib))
            continue;

         const bool is_shader_call_data =
            nir_deref_mode_is(deref, nir_var_shader_call_data);

         deref->modes = nir_var_function_temp;
         progress = true;

         if (deref->deref_type == nir_deref_type_var) {
            b->cursor = nir_before_instr(&deref->instr);
            nir_def *offset =
               is_shader_call_data ? arg_offset : nir_imm_int(b, 0);
            nir_deref_instr *replacement = nir_build_deref_cast(
               b, offset, nir_var_function_temp, deref->var->type, 0);
            nir_def_replace(&deref->def, &replacement->def);
         }
      }
   }

   return nir_progress(progress, impl, nir_metadata_control_flow);
}

/*
 * The ray state a stage reads is the state of the trace that invoked it, and
 * the megakernel restores that state after the inlined call returns.  Hoisting
 * the reads to the top of the stage guarantees they happen before anything the
 * stage itself does can perturb it.
 */
static bool
hk_move_ray_tracing_intrinsic(nir_builder *b, nir_intrinsic_instr *instr,
                              void *data)
{
   switch (instr->intrinsic) {
   case nir_intrinsic_load_shader_record_ptr:
   case nir_intrinsic_load_ray_flags:
   case nir_intrinsic_load_ray_object_origin:
   case nir_intrinsic_load_ray_world_origin:
   case nir_intrinsic_load_ray_t_min:
   case nir_intrinsic_load_ray_object_direction:
   case nir_intrinsic_load_ray_world_direction:
   case nir_intrinsic_load_ray_t_max:
      nir_instr_move(nir_before_impl(b->impl), &instr->instr);
      return true;
   default:
      return false;
   }
}

/* ------------------------------------------------------------------------ */
/* The megakernel compiler                                                   */
/* ------------------------------------------------------------------------ */

struct hk_ray_traversal_state {
   nir_variable *origin;
   nir_variable *dir;
   nir_variable *inv_dir;
   nir_variable *bvh_base;
   nir_variable *current_node;
   nir_variable *stack_base;
   nir_variable *stack_ptr;
   nir_variable *stack;
   nir_variable *hit;

   nir_variable *instance_addr;
   nir_variable *sbt_offset_and_flags;
};

struct hk_rt_shader_state {
   nir_variable *bvh_base;
   nir_variable *flags;
   nir_variable *cull_mask;
   nir_variable *sbt_offset;
   nir_variable *sbt_stride;
   nir_variable *miss_index;
   nir_variable *origin;
   nir_variable *tmin;
   nir_variable *dir;
   nir_variable *tmax;

   nir_variable *instance_addr;
   nir_variable *primitive_addr;
   nir_variable *primitive_id;
   nir_variable *geometry_id_and_flags;
   nir_variable *hit_kind;
   nir_variable *sbt_index;

   nir_variable *shader_record_ptr;
   nir_variable *stack_ptr;
   nir_variable *shader_call_data_offset;

   nir_variable *accept;
   nir_variable *terminate;
   nir_variable *opaque;

   struct hk_ray_traversal_state traversal;
};

struct hk_rt_compiler {
   struct hk_ray_tracing_pipeline *pipeline;
   VkPipelineCreateFlags2KHR flags;
   bool position_fetch;

   struct hk_rt_shader_state state;

   struct hash_table *functions;

   struct hk_rt_stack_sizes sizes;
};

static uint32_t
hk_rt_function_stack_size(struct hk_rt_compiler *compiler,
                          nir_function *function)
{
   hash_table_foreach (compiler->functions, entry) {
      if (entry->data == function) {
         const nir_shader *shader = entry->key;
         return shader->scratch_size;
      }
   }
   return 0;
}

static void
hk_ray_tracing_state_init(nir_shader *nir, struct hk_rt_compiler *compiler)
{
   struct hk_rt_shader_state *state = &compiler->state;

#define VAR(field, type, name)                                                 \
   state->field = nir_variable_create(nir, nir_var_shader_temp, type, name)

   VAR(bvh_base, glsl_uint64_t_type(), "bvh_base");
   VAR(flags, glsl_uint_type(), "flags");
   VAR(cull_mask, glsl_uint_type(), "cull_mask");
   VAR(sbt_offset, glsl_uint_type(), "sbt_offset");
   VAR(sbt_stride, glsl_uint_type(), "sbt_stride");
   VAR(miss_index, glsl_uint_type(), "miss_index");
   VAR(origin, glsl_vec_type(3), "origin");
   VAR(tmin, glsl_float_type(), "tmin");
   VAR(dir, glsl_vec_type(3), "dir");
   VAR(tmax, glsl_float_type(), "tmax");

   VAR(instance_addr, glsl_uint64_t_type(), "instance_addr");
   VAR(primitive_id, glsl_uint_type(), "primitive_id");
   VAR(geometry_id_and_flags, glsl_uint_type(), "geometry_id_and_flags");
   VAR(hit_kind, glsl_uint_type(), "hit_kind");
   VAR(sbt_index, glsl_uint_type(), "sbt_index");

   VAR(shader_record_ptr, glsl_uint64_t_type(), "shader_record_ptr");
   VAR(stack_ptr, glsl_uint_type(), "stack_ptr");
   VAR(shader_call_data_offset, glsl_uint_type(), "shader_call_data_offset");

   VAR(accept, glsl_bool_type(), "accept");
   VAR(terminate, glsl_bool_type(), "terminate");
   VAR(opaque, glsl_bool_type(), "opaque");

   if (compiler->position_fetch)
      VAR(primitive_addr, glsl_uint64_t_type(), "primitive_addr");

#undef VAR
}

static void
hk_ray_traversal_state_init(nir_function_impl *impl,
                            struct hk_ray_traversal_state *state)
{
   state->origin =
      nir_local_variable_create(impl, glsl_vec_type(3), "traversal.origin");
   state->dir =
      nir_local_variable_create(impl, glsl_vec_type(3), "traversal.dir");
   state->inv_dir =
      nir_local_variable_create(impl, glsl_vec_type(3), "traversal.inv_dir");
   state->bvh_base = nir_local_variable_create(impl, glsl_uint64_t_type(),
                                               "traversal.bvh_base");
   state->current_node = nir_local_variable_create(impl, glsl_uint_type(),
                                                   "traversal.current_node");
   state->stack_base =
      nir_local_variable_create(impl, glsl_uint_type(), "traversal.stack_base");
   state->stack_ptr =
      nir_local_variable_create(impl, glsl_uint_type(), "traversal.stack_ptr");
   state->stack = nir_local_variable_create(
      impl,
      glsl_array_type(glsl_uint_type(), HK_MAX_TLAS_DEPTH + HK_MAX_BLAS_DEPTH,
                      0),
      "traversal.stack");
   state->hit =
      nir_local_variable_create(impl, glsl_bool_type(), "traversal.hit");

   state->instance_addr = nir_local_variable_create(impl, glsl_uint64_t_type(),
                                                    "traversal.instance_addr");
   state->sbt_offset_and_flags = nir_local_variable_create(
      impl, glsl_uint_type(), "traversal.sbt_offset_and_flags");
}

/*
 * Inline one application stage as a NIR function of the megakernel, creating
 * the function on first use so a stage referenced by several groups is emitted
 * once.
 */
static void
hk_call_ray_tracing_stage(nir_builder *b, struct hk_rt_compiler *compiler,
                          nir_shader *stage)
{
   nir_function *function;

   struct hash_entry *entry =
      _mesa_hash_table_search(compiler->functions, stage);
   if (entry) {
      function = entry->data;
   } else {
      nir_function_impl *stage_entrypoint = nir_shader_get_entrypoint(stage);
      nir_function_impl *copy = nir_function_impl_clone(b->shader,
                                                        stage_entrypoint);

      struct hash_table *var_remap = _mesa_pointer_hash_table_create(NULL);

      nir_foreach_block (block, copy) {
         nir_foreach_instr_safe (instr, block) {
            if (instr->type != nir_instr_type_deref)
               continue;

            nir_deref_instr *deref = nir_instr_as_deref(instr);
            if (deref->deref_type != nir_deref_type_var ||
                deref->var->data.mode == nir_var_function_temp)
               continue;

            struct hash_entry *var_entry =
               _mesa_hash_table_search(var_remap, deref->var);
            if (!var_entry) {
               nir_variable *new_var = nir_variable_clone(deref->var,
                                                          b->shader);
               nir_shader_add_variable(b->shader, new_var);
               var_entry = _mesa_hash_table_insert(var_remap, deref->var,
                                                   new_var);
            }
            deref->var = var_entry->data;
         }
      }

      function = nir_function_create(
         b->shader, _mesa_shader_stage_to_string(stage->info.stage));
      nir_function_set_impl(function, copy);

      ralloc_free(var_remap);

      _mesa_hash_table_insert(compiler->functions, stage, function);
   }

   nir_build_call(b, function, 0, NULL);

   switch (stage->info.stage) {
   case MESA_SHADER_RAYGEN:
      compiler->sizes.raygen =
         MAX2(compiler->sizes.raygen, stage->scratch_size);
      break;
   case MESA_SHADER_ANY_HIT:
      compiler->sizes.any_hit =
         MAX2(compiler->sizes.any_hit, stage->scratch_size);
      break;
   case MESA_SHADER_CLOSEST_HIT:
      compiler->sizes.closest_hit =
         MAX2(compiler->sizes.closest_hit, stage->scratch_size);
      break;
   case MESA_SHADER_MISS:
      compiler->sizes.miss = MAX2(compiler->sizes.miss, stage->scratch_size);
      break;
   case MESA_SHADER_INTERSECTION:
      compiler->sizes.intersection =
         MAX2(compiler->sizes.intersection, stage->scratch_size);
      break;
   case MESA_SHADER_CALLABLE:
      compiler->sizes.callable =
         MAX2(compiler->sizes.callable, stage->scratch_size);
      break;
   default:
      UNREACHABLE("Invalid ray tracing stage");
   }
}

static void
hk_execute_callable(nir_builder *b, struct hk_rt_compiler *compiler,
                    nir_intrinsic_instr *instr)
{
   struct hk_rt_shader_state *state = &compiler->state;

   nir_def *sbt_index = instr->src[0].ssa;
   nir_def *payload = instr->src[1].ssa;

   struct hk_sbt_entry callable_entry =
      hk_load_sbt_entry(b, sbt_index, HK_RT_CMD_CALLABLE_ADDRESS);
   nir_store_var(b, state->shader_record_ptr,
                 callable_entry.shader_record_ptr, 0x1);

   const uint32_t stack_size =
      hk_rt_function_stack_size(compiler, b->impl->function);
   nir_def *stack_ptr = nir_load_var(b, state->stack_ptr);
   nir_store_var(b, state->stack_ptr, nir_iadd_imm(b, stack_ptr, stack_size),
                 0x1);

   nir_store_var(b, state->shader_call_data_offset,
                 nir_iadd_imm(b, payload, -(int32_t)stack_size), 0x1);

   for (uint32_t i = 0; i < compiler->pipeline->group_count; i++) {
      struct hk_ray_tracing_group *group = compiler->pipeline->groups + i;
      if (group->recursive_index == VK_SHADER_UNUSED_KHR)
         continue;

      nir_shader *stage =
         compiler->pipeline->stages[group->recursive_index]->nir;
      if (stage->info.stage != MESA_SHADER_CALLABLE)
         continue;

      nir_push_if(b, nir_ieq_imm(b, callable_entry.value, group->handle.index));
      hk_call_ray_tracing_stage(b, compiler, stage);
      nir_pop_if(b, NULL);
   }

   nir_store_var(b, state->stack_ptr, stack_ptr, 0x1);
}

struct hk_lower_isec_state {
   struct hk_rt_compiler *compiler;
   nir_shader *ahit;
};

/*
 * reportIntersectionEXT inside an intersection shader.  The hit is accepted
 * only if it is inside the current [tmin, tmax] window and the ray has not
 * already been terminated; an any-hit shader in the same group then gets to
 * veto it, and ignoreIntersectionEXT rolls the state back.
 */
static bool
hk_lower_isec_intrinsic(nir_builder *b, nir_intrinsic_instr *instr, void *data)
{
   if (instr->intrinsic != nir_intrinsic_report_ray_intersection)
      return false;

   struct hk_lower_isec_state *isec_state = data;
   struct hk_rt_compiler *compiler = isec_state->compiler;
   struct hk_rt_shader_state *state = &compiler->state;

   b->cursor = nir_after_instr(&instr->instr);

   nir_def *t = instr->src[0].ssa;
   nir_def *hit_kind = instr->src[1].ssa;

   nir_def *prev_accept = nir_load_var(b, state->accept);
   nir_def *prev_tmax = nir_load_var(b, state->tmax);
   nir_def *prev_hit_kind = nir_load_var(b, state->hit_kind);

   nir_variable *commit =
      nir_local_variable_create(b->impl, glsl_bool_type(), "commit");
   nir_store_var(b, commit, nir_imm_false(b), 0x1);

   nir_def *in_range =
      nir_iand(b, nir_fge(b, t, nir_load_var(b, state->tmin)),
               nir_fge(b, nir_load_var(b, state->tmax), t));
   nir_def *terminated = nir_iand(b, nir_load_var(b, state->terminate),
                                  nir_load_var(b, state->accept));
   nir_push_if(b, nir_iand(b, in_range, nir_inot(b, terminated)));
   {
      nir_store_var(b, state->accept, nir_imm_true(b), 0x1);

      nir_store_var(b, state->tmax, t, 1);
      nir_store_var(b, state->hit_kind, hit_kind, 1);

      if (isec_state->ahit) {
         nir_def *prev_terminate = nir_load_var(b, state->terminate);
         nir_store_var(b, state->terminate, nir_imm_false(b), 0x1);

         nir_push_if(b, nir_inot(b, nir_load_var(b, state->opaque)));
         {
            hk_call_ray_tracing_stage(b, compiler, isec_state->ahit);
         }
         nir_pop_if(b, NULL);

         nir_def *terminate = nir_load_var(b, state->terminate);
         nir_store_var(b, state->terminate,
                       nir_ior(b, terminate, prev_terminate), 0x1);

         nir_push_if(b, terminate);
         nir_jump(b, nir_jump_return);
         nir_pop_if(b, NULL);
      }

      nir_push_if(b, nir_load_var(b, state->accept));
      {
         nir_store_var(b, commit, nir_imm_true(b), 0x1);
      }
      nir_push_else(b, NULL);
      {
         nir_store_var(b, state->accept, prev_accept, 0x1);
         nir_store_var(b, state->tmax, prev_tmax, 1);
         nir_store_var(b, state->hit_kind, prev_hit_kind, 1);
      }
      nir_pop_if(b, NULL);
   }
   nir_pop_if(b, NULL);

   nir_def_replace(&instr->def, nir_load_var(b, commit));

   return true;
}

static nir_def *
hk_hit_sbt_index(nir_builder *b, struct hk_rt_shader_state *state,
                 nir_def *geometry_id_and_flags)
{
   nir_def *geometry_id = nir_iand_imm(b, geometry_id_and_flags, 0xfffffff);

   return nir_iadd(
      b,
      nir_iadd(b, nir_load_var(b, state->sbt_offset),
               nir_iand_imm(b, nir_load_var(b, state->traversal.sbt_offset_and_flags),
                            0xffffff)),
      nir_imul(b, nir_load_var(b, state->sbt_stride), geometry_id));
}

static void
hk_handle_aabb_intersection(nir_builder *b,
                            struct hk_leaf_intersection *intersection,
                            const struct hk_ray_traversal_args *args,
                            const struct hk_ray_flags *ray_flags)
{
   struct hk_rt_compiler *compiler = args->data;
   struct hk_rt_shader_state *state = &compiler->state;

   bool has_isec = false;
   for (uint32_t i = 0; i < compiler->pipeline->group_count; i++) {
      if (compiler->pipeline->groups[i].isec_index != VK_SHADER_UNUSED_KHR) {
         has_isec = true;
         break;
      }
   }

   /* No intersection shader anywhere in the pipeline means an AABB can never
    * produce a hit.  Emitting nothing is not a silent drop: a procedural
    * geometry with no intersection shader has no defined intersection. */
   if (!has_isec)
      return;

   nir_store_var(b, state->accept, nir_imm_false(b), 0x1);
   nir_store_var(b, state->terminate, ray_flags->terminate_on_first_hit, 0x1);
   nir_store_var(b, state->opaque, intersection->opaque, 0x1);

   nir_def *prev_instance_addr = nir_load_var(b, state->instance_addr);
   nir_def *prev_primitive_id = nir_load_var(b, state->primitive_id);
   nir_def *prev_geometry_id_and_flags =
      nir_load_var(b, state->geometry_id_and_flags);

   nir_store_var(b, state->instance_addr,
                 nir_load_var(b, state->traversal.instance_addr), 0x1);
   nir_store_var(b, state->primitive_id, intersection->primitive_id, 0x1);
   nir_store_var(b, state->geometry_id_and_flags,
                 intersection->geometry_id_and_flags, 0x1);

   nir_def *sbt_index =
      hk_hit_sbt_index(b, state, intersection->geometry_id_and_flags);

   struct hk_sbt_entry isec_entry =
      hk_load_sbt_entry(b, sbt_index, HK_RT_CMD_HIT_ADDRESS);
   nir_store_var(b, state->shader_record_ptr, isec_entry.shader_record_ptr,
                 0x1);

   for (uint32_t i = 0; i < compiler->pipeline->group_count; i++) {
      struct hk_ray_tracing_group *group = compiler->pipeline->groups + i;
      if (group->isec_index == VK_SHADER_UNUSED_KHR)
         continue;

      nir_shader *stage = compiler->pipeline->stages[group->isec_index]->nir;

      nir_push_if(b, nir_ieq_imm(b, isec_entry.value, group->handle.index));
      hk_call_ray_tracing_stage(b, compiler, stage);
      nir_pop_if(b, NULL);

      nir_shader *ahit_stage = NULL;
      if (group->ahit_index != VK_SHADER_UNUSED_KHR)
         ahit_stage = compiler->pipeline->stages[group->ahit_index]->nir;

      struct hk_lower_isec_state isec_state = {
         .compiler = compiler,
         .ahit = ahit_stage,
      };
      nir_shader_intrinsics_pass(b->shader, hk_lower_isec_intrinsic,
                                 nir_metadata_none, &isec_state);
   }

   nir_push_if(b, nir_load_var(b, state->accept));
   {
      nir_store_var(b, state->sbt_index, sbt_index, 0x1);
      nir_store_var(b, state->traversal.hit, nir_imm_true(b), 0x1);

      nir_break_if(b, nir_load_var(b, state->terminate));
   }
   nir_push_else(b, NULL);
   {
      nir_store_var(b, state->instance_addr, prev_instance_addr, 0x1);
      nir_store_var(b, state->primitive_id, prev_primitive_id, 0x1);
      nir_store_var(b, state->geometry_id_and_flags,
                    prev_geometry_id_and_flags, 0x1);
   }
   nir_pop_if(b, NULL);
}

static void
hk_handle_triangle_intersection(nir_builder *b,
                                struct hk_triangle_intersection *intersection,
                                const struct hk_ray_traversal_args *args,
                                const struct hk_ray_flags *ray_flags)
{
   struct hk_rt_compiler *compiler = args->data;
   struct hk_rt_shader_state *state = &compiler->state;

   nir_store_var(b, state->accept, nir_imm_true(b), 0x1);
   nir_store_var(b, state->terminate, ray_flags->terminate_on_first_hit, 0x1);

   /* The hit attribute lives at the top of the caller's frame, which is where
    * an any-hit shader's hitAttributeEXT deref was rewritten to point. */
   nir_def *barycentrics_offset = nir_load_var(b, state->stack_ptr);

   nir_def *prev_tmax = nir_load_var(b, state->tmax);
   nir_def *prev_instance_addr = nir_load_var(b, state->instance_addr);
   nir_def *prev_primitive_id = nir_load_var(b, state->primitive_id);
   nir_def *prev_geometry_id_and_flags =
      nir_load_var(b, state->geometry_id_and_flags);
   nir_def *prev_hit_kind = nir_load_var(b, state->hit_kind);
   nir_def *prev_barycentrics =
      nir_load_scratch(b, 2, 32, barycentrics_offset);

   nir_store_var(b, state->tmax, intersection->t, 0x1);
   nir_store_var(b, state->instance_addr,
                 nir_load_var(b, state->traversal.instance_addr), 0x1);
   nir_store_var(b, state->primitive_id, intersection->base.primitive_id, 0x1);
   nir_store_var(b, state->geometry_id_and_flags,
                 intersection->base.geometry_id_and_flags, 0x1);
   nir_store_var(b, state->hit_kind,
                 nir_bcsel(b, intersection->frontface, nir_imm_int(b, 0xFE),
                           nir_imm_int(b, 0xFF)),
                 0x1);

   nir_def *prev_primitive_addr = NULL;
   if (state->primitive_addr) {
      prev_primitive_addr = nir_load_var(b, state->primitive_addr);
      nir_store_var(b, state->primitive_addr, intersection->base.node_addr,
                    0x1);
   }

   nir_store_scratch(b, intersection->barycentrics, barycentrics_offset);

   nir_def *sbt_index =
      hk_hit_sbt_index(b, state, intersection->base.geometry_id_and_flags);

   nir_push_if(b, nir_inot(b, intersection->base.opaque));
   {
      struct hk_sbt_entry ahit_entry =
         hk_load_sbt_entry(b, sbt_index, HK_RT_CMD_HIT_ADDRESS);
      nir_store_var(b, state->shader_record_ptr,
                    ahit_entry.shader_record_ptr, 0x1);

      for (uint32_t i = 0; i < compiler->pipeline->group_count; i++) {
         struct hk_ray_tracing_group *group = compiler->pipeline->groups + i;
         if (group->ahit_index == VK_SHADER_UNUSED_KHR)
            continue;

         nir_shader *stage = compiler->pipeline->stages[group->ahit_index]->nir;

         nir_push_if(b, nir_ieq_imm(b, ahit_entry.value, group->handle.index));
         hk_call_ray_tracing_stage(b, compiler, stage);
         nir_pop_if(b, NULL);
      }
   }
   nir_pop_if(b, NULL);

   nir_push_if(b, nir_load_var(b, state->accept));
   {
      nir_store_var(b, state->sbt_index, sbt_index, 0x1);
      nir_store_var(b, state->traversal.hit, nir_imm_true(b), 0x1);

      nir_break_if(b, nir_load_var(b, state->terminate));
   }
   nir_push_else(b, NULL);
   {
      nir_store_var(b, state->tmax, prev_tmax, 0x1);
      nir_store_var(b, state->instance_addr, prev_instance_addr, 0x1);
      nir_store_var(b, state->primitive_id, prev_primitive_id, 0x1);
      nir_store_var(b, state->geometry_id_and_flags,
                    prev_geometry_id_and_flags, 0x1);
      nir_store_var(b, state->hit_kind, prev_hit_kind, 0x1);
      nir_store_scratch(b, prev_barycentrics, barycentrics_offset);
      if (state->primitive_addr)
         nir_store_var(b, state->primitive_addr, prev_primitive_addr, 0x1);
   }
   nir_pop_if(b, NULL);
}

static void
hk_trace_ray(nir_builder *b, struct hk_rt_compiler *compiler,
             nir_intrinsic_instr *instr)
{
   struct hk_rt_shader_state *state = &compiler->state;

   nir_def *accel_struct = instr->src[0].ssa;
   nir_def *flags = instr->src[1].ssa;
   nir_def *cull_mask = instr->src[2].ssa;
   nir_def *sbt_offset = nir_iand_imm(b, instr->src[3].ssa, 0xF);
   nir_def *sbt_stride = nir_iand_imm(b, instr->src[4].ssa, 0xF);
   nir_def *miss_index = nir_iand_imm(b, instr->src[5].ssa, 0xFFFF);
   nir_def *origin = instr->src[6].ssa;
   nir_def *tmin = instr->src[7].ssa;
   nir_def *dir = instr->src[8].ssa;
   nir_def *tmax = instr->src[9].ssa;
   nir_def *payload = instr->src[10].ssa;

   const uint32_t stack_size =
      hk_rt_function_stack_size(compiler, b->impl->function);
   nir_def *stack_ptr = nir_load_var(b, state->stack_ptr);
   nir_store_var(b, state->stack_ptr, nir_iadd_imm(b, stack_ptr, stack_size),
                 0x1);

   nir_store_var(b, state->shader_call_data_offset,
                 nir_iadd_imm(b, payload, -(int32_t)stack_size), 0x1);

   hk_ray_traversal_state_init(b->impl, &state->traversal);

   nir_store_var(b, state->bvh_base, accel_struct, 0x1);
   nir_store_var(b, state->flags, flags, 0x1);
   nir_store_var(b, state->cull_mask, cull_mask, 0x1);
   nir_store_var(b, state->sbt_offset, sbt_offset, 0x1);
   nir_store_var(b, state->sbt_stride, sbt_stride, 0x1);
   nir_store_var(b, state->miss_index, miss_index, 0x1);
   nir_store_var(b, state->origin, origin, 0x7);
   nir_store_var(b, state->tmin, tmin, 0x1);
   nir_store_var(b, state->dir, dir, 0x7);
   nir_store_var(b, state->tmax, tmax, 0x1);

   nir_store_var(b, state->traversal.bvh_base, accel_struct, 0x1);
   nir_store_var(b, state->traversal.origin, origin, 0x7);
   nir_store_var(b, state->traversal.dir, dir, 0x7);
   nir_store_var(b, state->traversal.inv_dir, nir_frcp(b, dir), 0x7);
   nir_store_var(b, state->traversal.current_node,
                 nir_imm_int(b, HK_BVH_ROOT_NODE), 0x1);
   nir_store_var(b, state->traversal.stack_base, nir_imm_int(b, -1), 0x1);
   nir_store_var(b, state->traversal.stack_ptr, nir_imm_int(b, 0), 0x1);

   nir_store_var(b, state->traversal.hit, nir_imm_false(b), 0x1);

   struct hk_ray_traversal_vars vars = {
      .tmax = nir_build_deref_var(b, state->tmax),
      .origin = nir_build_deref_var(b, state->traversal.origin),
      .dir = nir_build_deref_var(b, state->traversal.dir),
      .inv_dir = nir_build_deref_var(b, state->traversal.inv_dir),
      .bvh_base = nir_build_deref_var(b, state->traversal.bvh_base),
      .current_node = nir_build_deref_var(b, state->traversal.current_node),
      .stack_base = nir_build_deref_var(b, state->traversal.stack_base),
      .stack_ptr = nir_build_deref_var(b, state->traversal.stack_ptr),
      .stack = nir_build_deref_var(b, state->traversal.stack),
      .instance_addr = nir_build_deref_var(b, state->traversal.instance_addr),
      .sbt_offset_and_flags =
         nir_build_deref_var(b, state->traversal.sbt_offset_and_flags),
   };

   struct hk_ray_traversal_args args = {
      .root_bvh_base = accel_struct,
      .flags = flags,
      .cull_mask = nir_ishl_imm(b, cull_mask, 24),
      .origin = origin,
      .tmin = tmin,
      .dir = dir,
      .vars = vars,
      .aabb_cb =
         (compiler->flags & VK_PIPELINE_CREATE_2_RAY_TRACING_SKIP_AABBS_BIT_KHR)
            ? NULL
            : hk_handle_aabb_intersection,
      .triangle_cb =
         (compiler->flags &
          VK_PIPELINE_CREATE_2_RAY_TRACING_SKIP_TRIANGLES_BIT_KHR)
            ? NULL
            : hk_handle_triangle_intersection,
      .data = compiler,
   };

   nir_push_if(b, nir_ine_imm(b, accel_struct, 0));
   hk_build_ray_traversal(b, &args);
   nir_pop_if(b, NULL);

   nir_push_if(b, nir_load_var(b, state->traversal.hit));
   {
      nir_def *skip_chit =
         nir_test_mask(b, flags, SpvRayFlagsSkipClosestHitShaderKHRMask);
      nir_push_if(b, nir_inot(b, skip_chit));

      struct hk_sbt_entry chit_entry = hk_load_sbt_entry(
         b, nir_load_var(b, state->sbt_index), HK_RT_CMD_HIT_ADDRESS);
      nir_store_var(b, state->shader_record_ptr,
                    chit_entry.shader_record_ptr, 0x1);

      for (uint32_t i = 0; i < compiler->pipeline->group_count; i++) {
         struct hk_ray_tracing_group *group = compiler->pipeline->groups + i;
         if (group->recursive_index == VK_SHADER_UNUSED_KHR)
            continue;

         nir_shader *stage =
            compiler->pipeline->stages[group->recursive_index]->nir;
         if (stage->info.stage != MESA_SHADER_CLOSEST_HIT)
            continue;

         nir_push_if(b, nir_ieq_imm(b, chit_entry.value, group->handle.index));
         hk_call_ray_tracing_stage(b, compiler, stage);
         nir_pop_if(b, NULL);
      }

      nir_pop_if(b, NULL);
   }
   nir_push_else(b, NULL);
   {
      struct hk_sbt_entry miss_entry =
         hk_load_sbt_entry(b, miss_index, HK_RT_CMD_MISS_ADDRESS);
      nir_store_var(b, state->shader_record_ptr,
                    miss_entry.shader_record_ptr, 0x1);

      for (uint32_t i = 0; i < compiler->pipeline->group_count; i++) {
         struct hk_ray_tracing_group *group = compiler->pipeline->groups + i;
         if (group->recursive_index == VK_SHADER_UNUSED_KHR)
            continue;

         nir_shader *stage =
            compiler->pipeline->stages[group->recursive_index]->nir;
         if (stage->info.stage != MESA_SHADER_MISS)
            continue;

         nir_push_if(b, nir_ieq_imm(b, miss_entry.value, group->handle.index));
         hk_call_ray_tracing_stage(b, compiler, stage);
         nir_pop_if(b, NULL);
      }
   }
   nir_pop_if(b, NULL);

   nir_store_var(b, state->stack_ptr, stack_ptr, 0x1);
}

static bool
hk_lower_ray_tracing_instr(nir_builder *b, nir_instr *instr, void *data)
{
   struct hk_rt_compiler *compiler = data;
   struct hk_rt_shader_state *state = &compiler->state;

   if (instr->type == nir_instr_type_jump) {
      nir_jump_instr *jump = nir_instr_as_jump(instr);
      if (jump->type == nir_jump_halt) {
         /* An inlined stage's terminateInvocation must end the stage, not the
          * megakernel: the caller still has state to restore. */
         jump->type = nir_jump_return;
         return true;
      }
      return false;
   } else if (instr->type != nir_instr_type_intrinsic) {
      return false;
   }

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   nir_def *def = NULL;

   b->cursor = nir_before_instr(instr);

   switch (intr->intrinsic) {
   /* Ray tracing instructions */
   case nir_intrinsic_execute_callable:
      hk_execute_callable(b, compiler, intr);
      break;
   case nir_intrinsic_trace_ray:
      hk_trace_ray(b, compiler, intr);
      break;
   case nir_intrinsic_ignore_ray_intersection:
      nir_store_var(b, state->accept, nir_imm_false(b), 0x1);
      nir_push_if(b, nir_imm_true(b));
      nir_jump(b, nir_jump_return);
      nir_pop_if(b, NULL);
      break;
   case nir_intrinsic_terminate_ray:
      nir_store_var(b, state->accept, nir_imm_true(b), 0x1);
      nir_store_var(b, state->terminate, nir_imm_true(b), 0x1);
      nir_push_if(b, nir_imm_true(b));
      nir_jump(b, nir_jump_return);
      nir_pop_if(b, NULL);
      break;

   /* Ray tracing system values */
   case nir_intrinsic_load_ray_launch_id:
      def = nir_load_global_invocation_id(b, 32);
      break;
   case nir_intrinsic_load_ray_launch_size:
      def = hk_load_trace_ray_command_field(b, HK_RT_CMD_WIDTH, 3, 32);
      break;
   case nir_intrinsic_load_shader_record_ptr:
      def = nir_load_var(b, state->shader_record_ptr);
      break;
   case nir_intrinsic_load_ray_t_min:
      def = nir_load_var(b, state->tmin);
      break;
   case nir_intrinsic_load_ray_t_max:
      def = nir_load_var(b, state->tmax);
      break;
   case nir_intrinsic_load_ray_world_origin:
      def = nir_load_var(b, state->origin);
      break;
   case nir_intrinsic_load_ray_world_direction:
      def = nir_load_var(b, state->dir);
      break;
   case nir_intrinsic_load_ray_instance_custom_index: {
      nir_def *instance_node_addr = nir_load_var(b, state->instance_addr);
      nir_def *custom_instance_and_mask = nir_load_global(
         b, 1, 32,
         nir_iadd_imm(b, instance_node_addr,
                      offsetof(struct hk_bvh_instance_node,
                               custom_instance_and_mask)));
      def = nir_iand_imm(b, custom_instance_and_mask, 0xFFFFFF);
      break;
   }
   case nir_intrinsic_load_primitive_id:
      def = nir_load_var(b, state->primitive_id);
      break;
   case nir_intrinsic_load_ray_geometry_index:
      def = nir_iand_imm(b, nir_load_var(b, state->geometry_id_and_flags),
                         0xFFFFFFF);
      break;
   case nir_intrinsic_load_instance_id: {
      nir_def *instance_node_addr = nir_load_var(b, state->instance_addr);
      def = nir_load_global(
         b, 1, 32,
         nir_iadd_imm(b, instance_node_addr,
                      offsetof(struct hk_bvh_instance_node, instance_id)));
      break;
   }
   case nir_intrinsic_load_ray_flags:
      def = nir_load_var(b, state->flags);
      break;
   case nir_intrinsic_load_ray_hit_kind:
      def = nir_load_var(b, state->hit_kind);
      break;
   case nir_intrinsic_load_ray_world_to_object: {
      const unsigned c = nir_intrinsic_column(intr);
      nir_def *instance_node_addr = nir_load_var(b, state->instance_addr);
      nir_def *wto_matrix[3];
      hk_load_wto_matrix(b, instance_node_addr, NULL, wto_matrix);

      nir_def *vals[3];
      for (unsigned i = 0; i < 3; ++i)
         vals[i] = nir_channel(b, wto_matrix[i], c);

      def = nir_vec(b, vals, 3);
      break;
   }
   case nir_intrinsic_load_ray_object_to_world: {
      const unsigned c = nir_intrinsic_column(intr);
      nir_def *instance_node_addr = nir_load_var(b, state->instance_addr);
      nir_def *rows[3];
      for (unsigned r = 0; r < 3; ++r)
         rows[r] = nir_load_global(
            b, 4, 32,
            nir_iadd_imm(b, instance_node_addr,
                         offsetof(struct hk_bvh_instance_node, otw_matrix) +
                            r * 16));
      def = nir_vec3(b, nir_channel(b, rows[0], c), nir_channel(b, rows[1], c),
                     nir_channel(b, rows[2], c));
      break;
   }
   case nir_intrinsic_load_ray_object_origin: {
      nir_def *instance_node_addr = nir_load_var(b, state->instance_addr);
      nir_def *wto_matrix[3];
      hk_load_wto_matrix(b, instance_node_addr, NULL, wto_matrix);
      def = hk_mul_vec3_mat(b, nir_load_var(b, state->origin), wto_matrix,
                            true);
      break;
   }
   case nir_intrinsic_load_ray_object_direction: {
      nir_def *instance_node_addr = nir_load_var(b, state->instance_addr);
      nir_def *wto_matrix[3];
      hk_load_wto_matrix(b, instance_node_addr, NULL, wto_matrix);
      def = hk_mul_vec3_mat(b, nir_load_var(b, state->dir), wto_matrix, false);
      break;
   }
   case nir_intrinsic_load_cull_mask:
      def = nir_iand_imm(b, nir_load_var(b, state->cull_mask), 0xFF);
      break;

   /* Ray tracing stack lowering: every stage's scratch access is relative to
    * its own frame. */
   case nir_intrinsic_load_scratch:
      nir_src_rewrite(&intr->src[0],
                      nir_iadd(b, nir_load_var(b, state->stack_ptr),
                               intr->src[0].ssa));
      return true;
   case nir_intrinsic_store_scratch:
      nir_src_rewrite(&intr->src[1],
                      nir_iadd(b, nir_load_var(b, state->stack_ptr),
                               intr->src[1].ssa));
      return true;

   case nir_intrinsic_load_ray_triangle_vertex_positions:
      def = hk_load_vertex_position(b, nir_load_var(b, state->primitive_addr),
                                    nir_intrinsic_column(intr));
      break;

   /* Internal system values.  These two intrinsics are shared with lavapipe
    * rather than duplicated: they are driver-internal markers with no encoding
    * of their own, and both are fully consumed here. */
   case nir_intrinsic_load_shader_call_data_offset_lvp:
      def = nir_load_var(b, state->shader_call_data_offset);
      break;

   default:
      return false;
   }

   if (def)
      nir_def_rewrite_uses(&intr->def, def);
   nir_instr_remove(instr);

   return true;
}

static void
hk_rt_optimize_nir(nir_shader *nir)
{
   bool progress;

   do {
      progress = false;

      NIR_PASS(progress, nir, nir_opt_loop);
      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_remove_phis);
      NIR_PASS(progress, nir, nir_opt_dce);
      NIR_PASS(progress, nir, nir_opt_dead_cf);
      NIR_PASS(progress, nir, nir_opt_cse);
      NIR_PASS(progress, nir, nir_opt_algebraic);
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      NIR_PASS(progress, nir, nir_opt_undef);
   } while (progress);
}

/* ------------------------------------------------------------------------ */
/* Pipeline creation                                                         */
/* ------------------------------------------------------------------------ */

static void
hk_ray_tracing_pipeline_destroy(struct vk_device *vk_dev,
                                struct vk_pipeline *vk_pipeline,
                                const VkAllocationCallbacks *pAllocator)
{
   struct hk_ray_tracing_pipeline *pipeline =
      hk_pipeline_to_ray_tracing(vk_pipeline);

   if (pipeline->shader)
      hk_api_shader_destroy(vk_dev, &pipeline->shader->vk, pAllocator);

   if (pipeline->stages) {
      for (uint32_t i = 0; i < pipeline->stage_count; i++)
         hk_rt_stage_nir_unref(pipeline->stages[i]);
      free(pipeline->stages);
   }

   free(pipeline->groups);

   if (pipeline->layout)
      vk_pipeline_layout_unref(vk_dev, pipeline->layout);

   vk_pipeline_free(vk_dev, pAllocator, &pipeline->base);
}

static void
hk_ray_tracing_pipeline_cmd_bind(struct vk_command_buffer *vk_cmd,
                                 struct vk_pipeline *vk_pipeline)
{
   struct hk_cmd_buffer *cmd =
      container_of(vk_cmd, struct hk_cmd_buffer, vk);

   cmd->state.rt.pipeline =
      vk_pipeline ? hk_pipeline_to_ray_tracing(vk_pipeline) : NULL;
}

static struct vk_shader *
hk_ray_tracing_pipeline_get_shader(struct vk_pipeline *vk_pipeline,
                                   mesa_shader_stage stage)
{
   struct hk_ray_tracing_pipeline *pipeline =
      hk_pipeline_to_ray_tracing(vk_pipeline);

   /* The pipeline is one compute shader; there is no per-stage binary to
    * report.  Saying so is better than handing back the megakernel under a
    * stage name it is not. */
   return NULL;
}

static const struct vk_pipeline_ops hk_ray_tracing_pipeline_ops = {
   .destroy = hk_ray_tracing_pipeline_destroy,
   .cmd_bind = hk_ray_tracing_pipeline_cmd_bind,
   .get_shader = hk_ray_tracing_pipeline_get_shader,
};

static VkResult
hk_rt_stage_to_nir(struct hk_device *dev,
                   VkPipelineCreateFlags2KHR pipeline_flags, const void *pNext,
                   const VkPipelineShaderStageCreateInfo *stage_info,
                   nir_shader **nir_out)
{
   struct hk_physical_device *pdev = hk_device_physical(dev);
   struct vk_pipeline_robustness_state rstate;

   vk_pipeline_robustness_state_fill(&dev->vk, &rstate, pNext,
                                     stage_info->pNext);

   const mesa_shader_stage stage =
      vk_to_mesa_shader_stage(stage_info->stage);
   const struct spirv_to_nir_options spirv_options =
      hk_get_spirv_options(&pdev->vk, stage, &rstate);
   const nir_shader_compiler_options *nir_options =
      hk_get_nir_options(&pdev->vk, stage, &rstate);

   return vk_pipeline_shader_stage_to_nir(&dev->vk, pipeline_flags, stage_info,
                                          &spirv_options, nir_options, NULL,
                                          nir_out);
}


/*
 * A `rayQueryEXT` declared inside a ray-tracing stage is an ordinary
 * function_temp variable, and hk_nir_lower_ray_queries() finds it by
 * `var->data.ray_query` and then requires every rq_* intrinsic's source to
 * still be a DEREF of it (`hk_nir_lower_ray_queries.c:607-617`).  The
 * per-stage nir_lower_explicit_io(nir_var_function_temp) below rewrites that
 * deref into a 32-bit scratch offset, after which the pass aborts on
 * nir_instr_as_deref -- `nir.h:3060: nir_instr_as_deref: Assertion `parent &&
 * parent->type == nir_instr_type_deref' failed`, and a plain SIGSEGV in a
 * release build.  lavapipe has the same shape and the same hazard
 * (`lvp_ray_tracing_pipeline.c:181`).
 *
 * Ray-query lowering cannot simply move earlier: it needs rq_initialize's
 * acceleration structure to already be a resolved 64-bit address, and
 * descriptor lowering happens inside hk_lower_nir(), long after the megakernel
 * is assembled (`hk_shader.c:1764-1783`).  So park the query variables in a
 * mode the stage lowering does not touch and put them back before the stage
 * joins the megakernel.  Nothing between those two points lowers
 * nir_var_shader_temp inside a stage, and nothing after the unpark lowers
 * function_temp until hk_lower_nir() has run the ray-query pass.
 */
static nir_variable *
hk_deref_base_var(nir_deref_instr *deref)
{
   while (deref->deref_type != nir_deref_type_var) {
      if (deref->deref_type == nir_deref_type_cast)
         return NULL;

      deref = nir_deref_instr_parent(deref);
      if (!deref)
         return NULL;
   }

   return deref->var;
}

static void
hk_park_ray_query_vars(nir_shader *nir, bool park)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   const nir_variable_mode to =
      park ? nir_var_shader_temp : nir_var_function_temp;
   struct exec_list *src = park ? &impl->locals : &nir->variables;
   struct exec_list *dst = park ? &nir->variables : &impl->locals;
   bool any = false;

   foreach_list_typed_safe (nir_variable, var, node, src) {
      if (!var->data.ray_query)
         continue;

      exec_node_remove(&var->node);
      var->data.mode = to;
      exec_list_push_tail(dst, &var->node);
      any = true;
   }

   if (!any)
      return;

   nir_foreach_block (block, impl) {
      nir_foreach_instr (instr, block) {
         if (instr->type != nir_instr_type_deref)
            continue;

         nir_deref_instr *deref = nir_instr_as_deref(instr);
         nir_variable *var = hk_deref_base_var(deref);
         if (var && var->data.ray_query)
            deref->modes = to;
      }
   }

   nir_progress(true, impl, nir_metadata_all);
}

static VkResult
hk_compile_ray_tracing_stages(
   struct hk_device *dev, struct hk_ray_tracing_pipeline *pipeline,
   const VkRayTracingPipelineCreateInfoKHR *create_info)
{
   const VkPipelineCreateFlags2KHR pipeline_flags =
      vk_rt_pipeline_create_flags(create_info);
   uint32_t i = 0;

   for (; i < create_info->stageCount; i++) {
      nir_shader *nir;
      VkResult result = hk_rt_stage_to_nir(dev, pipeline_flags,
                                           create_info->pNext,
                                           create_info->pStages + i, &nir);
      if (result != VK_SUCCESS)
         return result;

      /* Every stage's private storage is its frame of the megakernel's
       * scratch stack, so nothing may already be claiming scratch. */
      /* Turn system-value variables into intrinsics while the stage is still
       * its own shader.  gl_LaunchIDEXT and friends arrive from SPIR-V as
       * `decl_var system` plus a deref; hk_lower_ray_tracing_instr() lowers
       * the INTRINSIC form, and nothing downstream of the megakernel build
       * would reach a surviving deref -- the driver's own
       * nir_lower_system_values runs inside hk_preprocess_nir_internal, which
       * is after the ray-tracing lowering pass has already been and gone.
       *
       * Getting this wrong is invisible until the AGX backend, and the way it
       * announces itself is hostile: agx_emit_intrinsic()'s default prints the
       * name once and then UNREACHABLE()s, which in a release build spun out
       * 184 million lines before the process died.  Redirect to a file and
       * head it; the first line is the whole message. */
      NIR_PASS(_, nir, nir_lower_system_values);

      /* An outgoing ray payload (SpvStorageClassRayPayloadKHR) and an outgoing
       * callable payload arrive as nir_var_shader_temp
       * (`vtn_variables.c:1872-1875`), not function_temp, so they are NOT
       * covered by the explicit-type and explicit-io lowering below and their
       * derefs survive into the megakernel.  traceRayEXT's payload source must
       * be a scratch OFFSET by the time hk_trace_ray() reads it, or the
       * megakernel biases the callee's payload window by a pointer.
       *
       * Demote them to function_temp first.  lavapipe gets this for free
       * because its per-stage preprocessing already ran
       * nir_lower_global_vars_to_local; hk's does not, and the symptom is a
       * `deref_var &payload (function_temp uint)` reaching agx_emit_instr()
       * and asserting "should've been lowered" -- in a release build, a
       * segfault with no message. */
      NIR_PASS(_, nir, nir_split_var_copies);
      NIR_PASS(_, nir, nir_lower_var_copies);
      NIR_PASS(_, nir, nir_lower_global_vars_to_local);
      NIR_PASS(_, nir, nir_lower_vars_to_ssa);

      assert(!nir->scratch_size);
      if (nir->info.stage == MESA_SHADER_ANY_HIT ||
          nir->info.stage == MESA_SHADER_CLOSEST_HIT ||
          nir->info.stage == MESA_SHADER_INTERSECTION)
         nir->scratch_size = HK_RAY_HIT_ATTRIBS_SIZE;

      hk_park_ray_query_vars(nir, true);

      NIR_PASS(_, nir, nir_lower_vars_to_explicit_types,
               nir_var_function_temp | nir_var_shader_call_data |
                  nir_var_ray_hit_attrib,
               glsl_get_natural_size_align_bytes);

      NIR_PASS(_, nir, hk_lower_ray_tracing_derefs);

      NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_function_temp,
               nir_address_format_32bit_offset);

      /* A shaderRecordEXT block is not a variable.  vtn builds a deref_cast of
       * nir_load_shader_record_ptr() in nir_var_mem_constant
       * (`vtn_variables.c:448-457` and `:1884-1886`), so there is no
       * nir_variable for nir_lower_vars_to_explicit_types to find and the mode
       * is one hk_lower_nir() never lowers: it lowers nir_var_mem_global,
       * nir_var_mem_ssbo, nir_var_mem_ubo and nir_var_mem_shared and no other
       * (`hk_shader.c:1767-1775`, `:1811-1814`).  Without this the deref
       * survives every pass and reaches the AGX backend, where
       * agx_extract_nir_src() reads a def that was never emitted -- in a
       * release build a SIGSEGV inside agx_emit_alu() with no message, which
       * is nowhere near the deref.  64-bit global is the right format: the
       * record address is an ordinary device address and
       * agx_nir_lower_address() already lowers load_global_constant
       * (`agx_nir_lower_address.c:110-111`, `:240`). */
      NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_constant,
               nir_address_format_64bit_global);

      hk_park_ray_query_vars(nir, false);

      NIR_PASS(_, nir, nir_shader_intrinsics_pass,
               hk_move_ray_tracing_intrinsic, nir_metadata_control_flow, NULL);

      pipeline->stages[i] = hk_rt_stage_nir_create(nir);
      if (!pipeline->stages[i]) {
         ralloc_free(nir);
         return VK_ERROR_OUT_OF_HOST_MEMORY;
      }
   }

   if (!create_info->pLibraryInfo)
      return VK_SUCCESS;

   for (uint32_t library_index = 0;
        library_index < create_info->pLibraryInfo->libraryCount;
        library_index++) {
      VK_FROM_HANDLE(vk_pipeline, vk_library,
                     create_info->pLibraryInfo->pLibraries[library_index]);
      struct hk_ray_tracing_pipeline *library =
         hk_pipeline_to_ray_tracing(vk_library);

      for (uint32_t stage_index = 0; stage_index < library->stage_count;
           stage_index++) {
         hk_rt_stage_nir_ref(pipeline->stages + i,
                             library->stages[stage_index]);
         i++;
      }
   }

   return VK_SUCCESS;
}

static VkResult
hk_init_ray_tracing_groups(
   struct hk_device *dev, struct hk_ray_tracing_pipeline *pipeline,
   const VkRayTracingPipelineCreateInfoKHR *create_info)
{
   uint32_t i = 0;

   for (; i < create_info->groupCount; i++) {
      const VkRayTracingShaderGroupCreateInfoKHR *group_info =
         create_info->pGroups + i;

      if (!hk_rt_group_from_create_info(
             pipeline->groups + i, group_info->type,
             group_info->generalShader, group_info->closestHitShader,
             group_info->anyHitShader, group_info->intersectionShader))
         return VK_ERROR_FEATURE_NOT_PRESENT;

      /* A group index past the end of this pipeline's own stage array is
       * application error, and the validation layer catches it -- but the
       * megakernel builder dereferences it, so without a check the driver
       * faults in a release build instead of refusing.  This is the same
       * process-boundary argument agx_emit_imadshl_agx() makes: refuse
       * loudly rather than take the process down. */
      const struct hk_ray_tracing_group *g = pipeline->groups + i;
      const uint32_t own_stage_count = create_info->stageCount;
      if ((g->recursive_index != VK_SHADER_UNUSED_KHR &&
           g->recursive_index >= own_stage_count) ||
          (g->ahit_index != VK_SHADER_UNUSED_KHR &&
           g->ahit_index >= own_stage_count) ||
          (g->isec_index != VK_SHADER_UNUSED_KHR &&
           g->isec_index >= own_stage_count)) {
         return vk_errorf(dev, VK_ERROR_INITIALIZATION_FAILED,
                          "ray tracing group %u names a shader index past "
                          "stageCount %u",
                          i, own_stage_count);
      }

      /* Handles are unique per device, not per pipeline, so a handle an
       * application captured from a library keeps identifying the same group
       * after it is imported. */
      pipeline->groups[i].handle.index =
         p_atomic_inc_return(&dev->rt_group_handle_alloc);
   }

   if (!create_info->pLibraryInfo)
      return VK_SUCCESS;

   uint32_t stage_base_index = create_info->stageCount;
   for (uint32_t library_index = 0;
        library_index < create_info->pLibraryInfo->libraryCount;
        library_index++) {
      VK_FROM_HANDLE(vk_pipeline, vk_library,
                     create_info->pLibraryInfo->pLibraries[library_index]);
      struct hk_ray_tracing_pipeline *library =
         hk_pipeline_to_ray_tracing(vk_library);

      for (uint32_t group_index = 0; group_index < library->group_count;
           group_index++) {
         pipeline->groups[i] =
            hk_rt_group_rebase(library->groups[group_index], stage_base_index);
         i++;
      }

      stage_base_index += library->stage_count;
   }

   return VK_SUCCESS;
}

static VkResult
hk_compile_ray_tracing_pipeline(
   struct hk_device *dev, struct hk_ray_tracing_pipeline *pipeline,
   const VkRayTracingPipelineCreateInfoKHR *create_info,
   const VkAllocationCallbacks *pAllocator)
{
   struct hk_physical_device *pdev = hk_device_physical(dev);
   struct vk_pipeline_robustness_state rstate;

   /* The megakernel is one shader made of many stages, so there is no
    * per-stage robustness to honour: take the pipeline-level state, which is
    * what VK_EXT_pipeline_robustness defines for a stage that does not
    * override it. */
   vk_pipeline_robustness_state_fill(&dev->vk, &rstate, create_info->pNext,
                                     NULL);

   const nir_shader_compiler_options *nir_options =
      hk_get_nir_options(&pdev->vk, MESA_SHADER_COMPUTE, &rstate);

   nir_builder _b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, nir_options, "ray tracing pipeline");
   nir_builder *b = &_b;

   /* nir_builder_init_simple_shader() marks the shader internal, and
    * hk_compile_nir() turns that into `has_scratch = false`
    * (hk_shader.c:2633), which makes agx_ra() assert "internal shaders are
    * unspillable".  The megakernel's whole stack model IS scratch, so it is
    * not an internal shader in that sense: it is an application shader that
    * the driver happens to assemble.  A release build does not assert; it
    * spills into a scratch allocation that does not exist. */
   b->shader->info.internal = false;

   b->shader->info.workgroup_size[0] = HK_RT_WORKGROUP_SIZE_X;
   b->shader->info.workgroup_size[1] = HK_RT_WORKGROUP_SIZE_Y;
   b->shader->info.workgroup_size[2] = HK_RT_WORKGROUP_SIZE_Z;

   struct hk_rt_compiler compiler = {
      .pipeline = pipeline,
      .flags = vk_rt_pipeline_create_flags(create_info),
      .position_fetch = dev->vk.enabled_features.rayTracingPositionFetch,
   };
   hk_ray_tracing_state_init(b->shader, &compiler);
   compiler.functions = _mesa_pointer_hash_table_create(NULL);

   /* A launch is dispatched in whole workgroups, so the trailing workgroup of
    * a width that is not a multiple of the workgroup size runs invocations the
    * application never asked for.  Retire them before anything else. */
   nir_def *launch_id = nir_load_ray_launch_id(b);
   nir_def *launch_size = nir_load_ray_launch_size(b);
   nir_def *oob = nir_ige(b, nir_channel(b, launch_id, 0),
                          nir_channel(b, launch_size, 0));
   oob = nir_ior(b, oob,
                 nir_ige(b, nir_channel(b, launch_id, 1),
                         nir_channel(b, launch_size, 1)));
   oob = nir_ior(b, oob,
                 nir_ige(b, nir_channel(b, launch_id, 2),
                         nir_channel(b, launch_size, 2)));

   nir_push_if(b, oob);
   nir_jump(b, nir_jump_return);
   nir_pop_if(b, NULL);

   nir_store_var(b, compiler.state.stack_ptr,
                 nir_load_ray_tracing_stack_base_lvp(b), 0x1);

   struct hk_sbt_entry raygen_entry =
      hk_load_sbt_entry(b, NULL, HK_RT_CMD_RAYGEN_ADDRESS);
   nir_store_var(b, compiler.state.shader_record_ptr,
                 raygen_entry.shader_record_ptr, 0x1);

   for (uint32_t i = 0; i < pipeline->group_count; i++) {
      struct hk_ray_tracing_group *group = pipeline->groups + i;
      if (group->recursive_index == VK_SHADER_UNUSED_KHR)
         continue;

      nir_shader *stage = pipeline->stages[group->recursive_index]->nir;
      if (stage->info.stage != MESA_SHADER_RAYGEN)
         continue;

      nir_push_if(b, nir_ieq_imm(b, raygen_entry.value, group->handle.index));
      hk_call_ray_tracing_stage(b, &compiler, stage);
      nir_pop_if(b, NULL);
   }

   nir_shader_instructions_pass(b->shader, hk_lower_ray_tracing_instr,
                                nir_metadata_none, &compiler);

   NIR_PASS(_, b->shader, nir_lower_continue_constructs);
   NIR_PASS(_, b->shader, nir_lower_variable_initializers,
            nir_var_function_temp);
   NIR_PASS(_, b->shader, nir_lower_returns);

   /* The AGX backend has no call instruction: every stage was inlined as a
    * NIR function so it could be emitted once and called from several places,
    * and the calls are flattened here.  lavapipe leaves them standing because
    * its backend executes calls directly; ours cannot, and a call that reaches
    * agx_compile_shader_nir() faults rather than refusing. */
   NIR_PASS(_, b->shader, nir_inline_functions);
   nir_remove_non_entrypoints(b->shader);

   const struct nir_lower_compute_system_values_options compute_system_values = {
      0
   };
   NIR_PASS(_, b->shader, nir_lower_compute_system_values,
            &compute_system_values);
   NIR_PASS(_, b->shader, nir_lower_global_vars_to_local);
   NIR_PASS(_, b->shader, nir_lower_vars_to_ssa);

   hk_rt_optimize_nir(b->shader);

   NIR_PASS(_, b->shader, nir_lower_vars_to_explicit_types, nir_var_shader_temp,
            glsl_get_natural_size_align_bytes);

   NIR_PASS(_, b->shader, nir_lower_explicit_io, nir_var_shader_temp,
            nir_address_format_32bit_offset);

   /* Everything below the megakernel's own frame is the per-stage stack. */
   NIR_PASS(_, b->shader, nir_inline_sysval,
            nir_intrinsic_load_ray_tracing_stack_base_lvp,
            b->shader->scratch_size);

   pipeline->stack_sizes = compiler.sizes;
   pipeline->stack_size = hk_rt_pipeline_stack_size(
      &compiler.sizes, create_info->maxPipelineRayRecursionDepth);
   b->shader->scratch_size += pipeline->stack_size;

   hk_rt_optimize_nir(b->shader);

   _mesa_hash_table_destroy(compiler.functions, NULL);

   /* hk_lower_nir() gates hk_nir_lower_ray_queries on `info.ray_queries > 0`
    * (`hk_shader.c:1783`), and info.ray_queries is only ever set by
    * nir_shader_gather_info() -- which hk calls in exactly one place,
    * hk_preprocess_nir_internal() (`hk_shader.c:206`), the function this path
    * must NOT run (see below).  The megakernel starts from
    * nir_builder_init_simple_shader(), so its info.ray_queries is 0 no matter
    * what the stages contain, and a rayQueryEXT in any ray-tracing stage then
    * reaches agx_emit_intrinsic() as `rq_initialize`.  That is not a corner:
    * dEQP-VK.ray_query has a variant of nearly every case for each
    * ray-tracing-pipeline stage.
    *
    * Gather here, after inlining, when the stages' ray-query variables are
    * function_temp locals of the entrypoint and nir_shader_gather_info() can
    * see them (`nir_gather_info.c:1168-1182`).  This is also what every
    * ordinary shader gets; the megakernel simply reaches hk_compile_shader()
    * by a different road. */
   nir_shader_gather_info(b->shader, nir_shader_get_entrypoint(b->shader));

   /* Hand the megakernel to the ordinary compute path with `io_lowered` still
    * false, exactly as hk's own internal compute shaders do, so that
    * hk_compile_shader() runs hk_lower_nir() on it: descriptor lowering, push
    * constants and agx_preprocess_nir are all inside that one call, in that
    * order.
    *
    * Do NOT pre-run hk_preprocess_nir_internal() here.  It ends in
    * agx_preprocess_nir(), which sets info.io_lowered, and hk_compile_shader()
    * then SKIPS hk_lower_nir() entirely -- so the megakernel reaches the
    * backend with vulkan_resource_index still in it and faults.  (The plural
    * hk_compile_shaders() calls hk_lower_nir() unconditionally, which is why
    * the pipeline path does not have this hazard.) */
   struct vk_shader_compile_info info = {
      .stage = MESA_SHADER_COMPUTE,
      .flags = 0,
      .rt_flags = compiler.flags & MESA_VK_PIPELINE_RAY_TRACING_FLAGS,
      .next_stage_mask = 0,
      .nir = b->shader,
      .robustness = &rstate,
      .set_layout_count = pipeline->layout ? pipeline->layout->set_count : 0,
      .set_layouts = pipeline->layout ? pipeline->layout->set_layouts : NULL,
      .push_constant_range_count = 0,
      .push_constant_ranges = NULL,
   };

   return hk_compile_shader(dev, &info, NULL, &dev->vk.enabled_features,
                            pAllocator, &pipeline->shader);
}

static VkResult
hk_create_ray_tracing_pipeline(
   struct hk_device *dev, const VkAllocationCallbacks *pAllocator,
   const VkRayTracingPipelineCreateInfoKHR *create_info,
   VkPipeline *out_pipeline)
{
   VK_FROM_HANDLE(vk_pipeline_layout, layout, create_info->layout);
   VkResult result;

   const VkPipelineCreateFlags2KHR flags =
      vk_rt_pipeline_create_flags(create_info);

   struct hk_ray_tracing_pipeline *pipeline = vk_pipeline_zalloc(
      &dev->vk, &hk_ray_tracing_pipeline_ops,
      VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, flags, pAllocator,
      sizeof(*pipeline));
   if (!pipeline)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   if (layout) {
      vk_pipeline_layout_ref(layout);
      pipeline->layout = layout;
   }

   /* maxRayRecursionDepth is a property, and exceeding it is application
    * error the validation layer catches -- but recursion here is unrolled at
    * compile time, so a pipeline that asks for more would silently get a
    * megakernel that traces one level and stops.  Refuse instead: a wrong
    * image is worse than a failed create. */
   if (create_info->maxPipelineRayRecursionDepth > HK_MAX_RAY_RECURSION_DEPTH) {
      vk_pipeline_free(&dev->vk, pAllocator, &pipeline->base);
      return vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT,
                       "maxPipelineRayRecursionDepth %u exceeds the %u this "
                       "driver advertises",
                       create_info->maxPipelineRayRecursionDepth,
                       HK_MAX_RAY_RECURSION_DEPTH);
   }

   pipeline->stage_count = create_info->stageCount;
   pipeline->group_count = create_info->groupCount;

   if (create_info->pLibraryInfo) {
      for (uint32_t i = 0; i < create_info->pLibraryInfo->libraryCount; i++) {
         VK_FROM_HANDLE(vk_pipeline, vk_library,
                        create_info->pLibraryInfo->pLibraries[i]);
         struct hk_ray_tracing_pipeline *library =
            hk_pipeline_to_ray_tracing(vk_library);

         pipeline->stage_count += library->stage_count;
         pipeline->group_count += library->group_count;
      }
   }

   if (pipeline->stage_count) {
      pipeline->stages =
         calloc(pipeline->stage_count, sizeof(*pipeline->stages));
      if (!pipeline->stages) {
         result = vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
         goto fail;
      }
   }

   if (pipeline->group_count) {
      pipeline->groups =
         calloc(pipeline->group_count, sizeof(*pipeline->groups));
      if (!pipeline->groups) {
         result = vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
         goto fail;
      }
   }

   result = hk_compile_ray_tracing_stages(dev, pipeline, create_info);
   if (result != VK_SUCCESS)
      goto fail;

   result = hk_init_ray_tracing_groups(dev, pipeline, create_info);
   if (result != VK_SUCCESS)
      goto fail;

   for (uint32_t i = 0; i < pipeline->stage_count; i++)
      pipeline->base.stages |= mesa_to_vk_shader_stage(
         pipeline->stages[i]->nir->info.stage);

   /* A library is a container of stages and groups; it is never executable and
    * compiling a megakernel for it would be wasted work. */
   if (!(flags & VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR)) {
      result = hk_compile_ray_tracing_pipeline(dev, pipeline, create_info,
                                               pAllocator);
      if (result != VK_SUCCESS)
         goto fail;
   }

   *out_pipeline = vk_pipeline_to_handle(&pipeline->base);
   return VK_SUCCESS;

fail:
   hk_ray_tracing_pipeline_destroy(&dev->vk, &pipeline->base, pAllocator);
   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_CreateRayTracingPipelinesKHR(
   VkDevice _device, VkDeferredOperationKHR deferredOperation,
   VkPipelineCache pipelineCache, uint32_t createInfoCount,
   const VkRayTracingPipelineCreateInfoKHR *pCreateInfos,
   const VkAllocationCallbacks *pAllocator, VkPipeline *pPipelines)
{
   VK_FROM_HANDLE(hk_device, dev, _device);
   VkResult result = VK_SUCCESS;

   uint32_t i = 0;
   for (; i < createInfoCount; i++) {
      VkResult one = hk_create_ray_tracing_pipeline(
         dev, pAllocator, pCreateInfos + i, pPipelines + i);

      if (one != VK_SUCCESS) {
         result = one;
         pPipelines[i] = VK_NULL_HANDLE;

         if (vk_rt_pipeline_create_flags(&pCreateInfos[i]) &
             VK_PIPELINE_CREATE_2_EARLY_RETURN_ON_FAILURE_BIT_KHR) {
            i++;
            break;
         }
      }
   }

   for (; i < createInfoCount; i++)
      pPipelines[i] = VK_NULL_HANDLE;

   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_GetRayTracingShaderGroupHandlesKHR(VkDevice _device, VkPipeline _pipeline,
                                      uint32_t firstGroup, uint32_t groupCount,
                                      size_t dataSize, void *pData)
{
   VK_FROM_HANDLE(vk_pipeline, vk_pipeline, _pipeline);
   struct hk_ray_tracing_pipeline *pipeline =
      hk_pipeline_to_ray_tracing(vk_pipeline);

   uint8_t *data = pData;

   assert(dataSize >= (size_t)groupCount * HK_RAY_TRACING_GROUP_HANDLE_SIZE);

   for (uint32_t i = 0; i < groupCount; i++) {
      hk_rt_write_group_handle(data + i * HK_RAY_TRACING_GROUP_HANDLE_SIZE,
                               pipeline->groups[firstGroup + i].handle);
   }

   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_GetRayTracingCaptureReplayShaderGroupHandlesKHR(
   VkDevice _device, VkPipeline _pipeline, uint32_t firstGroup,
   uint32_t groupCount, size_t dataSize, void *pData)
{
   /* shaderGroupHandleCaptureReplaySize is zero and
    * rayTracingPipelineShaderGroupHandleCaptureReplay is not advertised, so
    * there is nothing an application may legally read here. */
   return VK_SUCCESS;
}

VKAPI_ATTR VkDeviceSize VKAPI_CALL
hk_GetRayTracingShaderGroupStackSizeKHR(VkDevice _device, VkPipeline _pipeline,
                                        uint32_t group,
                                        VkShaderGroupShaderKHR groupShader)
{
   VK_FROM_HANDLE(vk_pipeline, vk_pipeline, _pipeline);
   struct hk_ray_tracing_pipeline *pipeline =
      hk_pipeline_to_ray_tracing(vk_pipeline);

   if (group >= pipeline->group_count)
      return 0;

   const struct hk_ray_tracing_group *g = pipeline->groups + group;
   uint32_t recursive_size = 0;

   if (g->recursive_index != VK_SHADER_UNUSED_KHR) {
      switch (pipeline->stages[g->recursive_index]->nir->info.stage) {
      case MESA_SHADER_RAYGEN:
         recursive_size = pipeline->stack_sizes.raygen;
         break;
      case MESA_SHADER_CLOSEST_HIT:
         recursive_size = pipeline->stack_sizes.closest_hit;
         break;
      case MESA_SHADER_MISS:
         recursive_size = pipeline->stack_sizes.miss;
         break;
      case MESA_SHADER_CALLABLE:
         recursive_size = pipeline->stack_sizes.callable;
         break;
      default:
         recursive_size = 0;
         break;
      }
   }

   return hk_rt_group_stack_size(&pipeline->stack_sizes, g, groupShader,
                                 recursive_size);
}
