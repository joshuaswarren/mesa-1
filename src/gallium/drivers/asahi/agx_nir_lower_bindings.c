/*
 * Copyright 2023 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

#include "asahi/compiler/agx_nir.h"
#include "compiler/nir/nir_builtin_builder.h"
#include "compiler/nir/nir_builder.h"
#include "util/bitset.h"
#include "agx_state.h"
#include "nir.h"
#include "nir_builder_opcodes.h"
#include "nir_intrinsics.h"
#include "nir_intrinsics_indices.h"

/*
 * Lower binding table textures and images to texture state registers and (if
 * necessary) bindless access into an internal table mapped like additional
 * texture state registers. The following layout is used:
 *
 *    1. Textures
 *    2. Images (read/write interleaved; on G17P each image is followed by a
 *       software tail record, so images are options->image_records apart)
 */

static mesa_shader_stage
stage_for_descriptors(nir_shader *shader)
{
   mesa_shader_stage stage = shader->info.stage;

   if (stage == MESA_SHADER_VERTEX && shader->info.vs.tes_poly)
      stage = MESA_SHADER_TESS_EVAL;

   assert(stage < MESA_SHADER_STAGES);
   return stage;
}

static nir_def *
load_stage_array(nir_builder *b, nir_def *index, unsigned components,
                 unsigned bit_size, uintptr_t binding, unsigned stride,
                 unsigned align, unsigned count)
{
   const unsigned table = AGX_SYSVAL_STAGE(stage_for_descriptors(b->shader));
   nir_scalar scalar = nir_scalar_resolved(index, 0);

   if (nir_scalar_is_const(scalar)) {
      const unsigned slot = nir_scalar_as_uint(scalar);
      assert(slot < count);
      return nir_load_sysval_agx(b, components, bit_size, .desc_set = table,
                                 .binding = binding + (slot * stride));
   }

   /* Keep a bad dynamic GL index from escaping the fixed driver table. */
   index = nir_umin(b, index,
                    nir_imm_intN_t(b, count - 1,
                                   index->bit_size));

   struct agx_draw_uniforms *u = NULL;
   nir_def *table_base = nir_load_sysval_agx(
      b, 1, 64, .desc_set = AGX_SYSVAL_TABLE_ROOT,
      .binding = (uintptr_t)&u->tables[table]);
   nir_def *offset = nir_iadd_imm(b, nir_imul_imm(b, index, stride), binding);
   nir_def *address = nir_iadd(b, table_base, nir_u2u64(b, offset));
   return nir_load_global_constant(b, components, bit_size, address,
                                   .align_mul = align);
}

static nir_def *
sampler_index(nir_builder *b, nir_tex_instr *tex)
{
   nir_def *index = nir_steal_tex_src(tex, nir_tex_src_sampler_offset);
   if (index)
      return index;

   const int handle_idx =
      nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle);
   if (handle_idx >= 0) {
      nir_scalar handle =
         nir_scalar_chase_movs(nir_get_scalar(tex->src[handle_idx].src.ssa, 0));
      if (nir_scalar_is_alu(handle) &&
          nir_scalar_alu_op(handle) == nir_op_u2u64)
         handle = nir_scalar_chase_alu_src(handle, 0);

      nir_intrinsic_instr *load = nir_def_as_intrinsic_or_null(handle.def);
      if (handle.comp == 0 && load &&
          load->intrinsic == nir_intrinsic_load_sampler_handle_agx)
         return load->src[0].ssa;
   }

   return nir_imm_int(b, tex->sampler_index);
}

static nir_def *
load_custom_border_sampler(nir_builder *b, nir_def *index)
{
   struct agx_stage_uniforms *s = NULL;
   return load_stage_array(
      b, index, 1, 16, (uintptr_t)&s->custom_border_sampler_handle,
      sizeof(s->custom_border_sampler_handle[0]), 2, PIPE_MAX_SAMPLERS);
}

static bool
lower_custom_border_query(nir_builder *b, nir_tex_instr *tex)
{
   if (tex->op != nir_texop_has_custom_border_color_agx &&
       tex->op != nir_texop_custom_border_color_agx)
      return false;

   nir_def *index = sampler_index(b, tex);
   nir_def *replacement;

   if (tex->op == nir_texop_has_custom_border_color_agx) {
      replacement = nir_ige_imm(b, load_custom_border_sampler(b, index), 0);
   } else {
      struct agx_stage_uniforms *s = NULL;
      nir_def *border = load_stage_array(
         b, index, 4, 32, (uintptr_t)&s->custom_border,
         sizeof(s->custom_border[0]), 4, PIPE_MAX_SAMPLERS);
      replacement = nir_convert_to_bit_size(
         b, border, nir_alu_type_get_base_type(tex->dest_type),
         tex->def.bit_size);
   }

   nir_def_replace(&tex->def, replacement);
   return true;
}

static bool
lower_sampler(nir_builder *b, nir_tex_instr *tex, bool wide_native_handle,
              bool clamp_to_zero)
{
   if (!nir_tex_instr_need_sampler(tex))
      return false;

   nir_def *index = sampler_index(b, tex);

   if (clamp_to_zero)
      nir_steal_tex_src(tex, nir_tex_src_sampler_handle);

   nir_def *handle = clamp_to_zero ? load_custom_border_sampler(b, index)
                                   : nir_load_sampler_handle_agx(b, index);
   if (wide_native_handle)
      handle = nir_u2u64(b, handle);

   nir_tex_instr_add_src(tex, nir_tex_src_sampler_handle, handle);
   return true;
}

static bool
lower(nir_builder *b, nir_instr *instr, void *data)
{
   struct agx_lower_bindings_options *options = data;
   bool *uses_bindless_samplers = options->uses_bindless_samplers;
   bool progress = false;
   bool force_bindless = agx_nir_needs_texture_crawl(instr);
   if (options->g17p_plain_txf_crawl && instr->type == nir_instr_type_tex) {
      nir_tex_instr *tex = nir_instr_as_tex(instr);

      /* agx_nir_lower_texture() crawls a G17P integer fetch's descriptor to
       * predicate it: the explicit out-of-bounds path reads the size of a two-
       * or three-dimensional fetch, and the older robustness path reads the
       * descriptor for a fetch carrying a minimum level. A crawl needs the
       * descriptor, and only a bindless fetch has one, so every fetch that
       * lowering will crawl has to be made bindless here.
       *
       * agx_nir_needs_texture_crawl() already covers the shapes that crawl on
       * every chip -- a non-zero level, an array layer, a buffer texture --
       * so this adds the two that are particular to G17P. Keep it in step with
       * the conditions in agx_nir_lower_texture(): a fetch that reaches a
       * crawl without a descriptor asserts rather than falls back.
       */
      force_bindless |=
         tex->op == nir_texop_txf &&
         !(tex->backend_flags & AGX_TEXTURE_FLAG_NO_CLAMP) &&
         (tex->sampler_dim == GLSL_SAMPLER_DIM_2D ||
          tex->sampler_dim == GLSL_SAMPLER_DIM_3D ||
          nir_tex_instr_src_index(tex, nir_tex_src_min_lod) >= 0);
   }
   b->cursor = nir_before_instr(instr);

   if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);

      switch (intr->intrinsic) {
      case nir_intrinsic_image_load:
      case nir_intrinsic_image_store:
      case nir_intrinsic_image_size:
      case nir_intrinsic_image_samples:
      case nir_intrinsic_image_atomic:
      case nir_intrinsic_image_atomic_swap:
         break;
      default:
         return false;
      }

      /* A buffer image's logical extent is not the width/height of its 2D
       * hardware view. AGX2 stores that extent in descriptor software bits;
       * Gen9 does not. Read the bound view's texel count from the stage table,
       * including for the bounds queries inserted by robustness lowering. */
      if (options->g17p_bindless_storage &&
          intr->intrinsic == nir_intrinsic_image_size &&
          nir_intrinsic_image_dim(intr) == GLSL_SAMPLER_DIM_BUF) {
         struct agx_stage_uniforms *s = NULL;
         nir_def *size = load_stage_array(
            b, intr->src[0].ssa, 1, 32, (uintptr_t)&s->image_buffer_size,
            sizeof(s->image_buffer_size[0]), 4, PIPE_MAX_SHADER_IMAGES);
         nir_def_replace(&intr->def, nir_u2uN(b, size, intr->def.bit_size));
         return true;
      }


      nir_def *index = intr->src[0].ssa;
      nir_scalar index_scalar = nir_scalar_resolved(index, 0);

      /* Remap according to the driver layout */
      unsigned offset = BITSET_LAST_BIT(b->shader->info.textures_used);

      /* For reads and queries, we use the texture descriptor which is first.
       * Writes and atomics use the PBE descriptor.
       */
      if (intr->intrinsic != nir_intrinsic_image_load &&
          intr->intrinsic != nir_intrinsic_image_size &&
          intr->intrinsic != nir_intrinsic_image_samples)
         offset++;

      /* G17P has no texture state registers to name a PBE with: imgwci reads
       * its PBE from a heap -- a 64-bit uniform -- at a record index held in
       * a register. Bind stores and atomics through the stage texture table
       * so they carry that pair; agx_nir_lower_g17p_storage_records() turns
       * the byte offset into the record index once the table's uniform is
       * known.
       */
      if (options->g17p_bindless_storage &&
          ((intr->intrinsic != nir_intrinsic_image_load &&
            intr->intrinsic != nir_intrinsic_image_size &&
            intr->intrinsic != nir_intrinsic_image_samples) ||
           (intr->intrinsic == nir_intrinsic_image_load &&
            nir_intrinsic_image_dim(intr) == GLSL_SAMPLER_DIM_MS)))
         force_bindless = true;

      /* If we can determine statically that the image fits in texture state
       * registers, avoid lowering to bindless access.
       */
      if (nir_scalar_is_const(index_scalar) && !force_bindless) {
         unsigned idx =
            (nir_scalar_as_uint(index_scalar) * options->image_records) +
            offset;

         if (idx < AGX_NUM_TEXTURE_STATE_REGS) {
            nir_src_rewrite(&intr->src[0], nir_imm_intN_t(b, idx, 16));
            return true;
         }
      }

      /* Otherwise, lower to bindless...
       *
       * The driver uploads enough null texture/PBE descriptors for robustness
       * given the shader limit, but we still need to clamp since we're lowering
       * to bindless so the hardware doesn't know the limit.
       *
       * The GL spec says out-of-bounds image indexing is undefined, but
       * faulting is not acceptable for robustness.
       */
      index = nir_umin(
         b, index,
         nir_imm_intN_t(b, b->shader->info.num_images - 1, index->bit_size));

      index = nir_iadd_imm(b, nir_imul_imm(b, index, options->image_records),
                           offset);

      nir_rewrite_image_intrinsic(intr, nir_load_texture_handle_agx(b, index),
                                  nir_image_intrinsic_type_bindless);
   } else if (instr->type == nir_instr_type_tex) {
      nir_tex_instr *tex = nir_instr_as_tex(instr);

      if (lower_custom_border_query(b, tex))
         return true;

      const unsigned bindful_sampler_count =
         options->bindful_sampler_count ? options->bindful_sampler_count : 16;
      const bool dynamic_sampler =
         nir_tex_instr_src_index(tex, nir_tex_src_sampler_offset) >= 0;
      const bool clamp_to_zero =
         tex->backend_flags & AGX_TEXTURE_FLAG_CLAMP_TO_0;
      const bool sampler_outside_bindful =
         tex->sampler_index >= bindful_sampler_count ||
         (dynamic_sampler &&
          BITSET_LAST_BIT(b->shader->info.samplers_used) >
             bindful_sampler_count);

      if ((sampler_outside_bindful || clamp_to_zero) &&
          lower_sampler(b, tex, options->g17p_native_sampler_handles,
                        clamp_to_zero)) {
         progress = true;
         if (!clamp_to_zero)
            *uses_bindless_samplers = true;
      }

      /* Nothing to do for "real" bindless */
      if (nir_tex_instr_src_index(tex, nir_tex_src_texture_handle) >= 0)
         return progress;

      /* Textures are mapped 1:1, so if we can prove it fits in a texture state
       * register, use the texture state register.
       */
      if (tex->texture_index < AGX_NUM_TEXTURE_STATE_REGS &&
          nir_tex_instr_src_index(tex, nir_tex_src_texture_offset) == -1 &&
          !force_bindless)
         return progress;

      /* Otherwise, lower to bindless. Could be optimized. */
      nir_def *index = nir_steal_tex_src(tex, nir_tex_src_texture_offset);
      if (!index)
         index = nir_imm_int(b, tex->texture_index);

      /* As above */
      index = nir_umin(
         b, index,
         nir_imm_intN_t(b, b->shader->info.num_textures - 1, index->bit_size));

      nir_tex_instr_add_src(tex, nir_tex_src_texture_handle,
                            nir_load_texture_handle_agx(b, index));
   }

   return true;
}

static nir_def *
query_custom_border(nir_builder *b, nir_tex_instr *tex)
{
   return nir_build_texture_query(b, tex, nir_texop_custom_border_color_agx, 4,
                                  tex->dest_type, false, false);
}

static nir_def *
has_custom_border(nir_builder *b, nir_tex_instr *tex)
{
   return nir_build_texture_query(b, tex,
                                  nir_texop_has_custom_border_color_agx, 1,
                                  nir_type_bool1, false, false);
}

static bool
lower_custom_border(nir_builder *b, nir_tex_instr *tex, UNUSED void *data)
{
   if (!nir_tex_instr_need_sampler(tex) || nir_tex_instr_is_query(tex) ||
       tex->is_shadow)
      return false;

   b->cursor = nir_after_instr(&tex->instr);
   nir_def *has_custom = has_custom_border(b, tex);

   nir_instr *ordinary_instr = nir_instr_clone(b->shader, &tex->instr);
   nir_builder_instr_insert(b, ordinary_instr);
   nir_def *clamp_to_one = &nir_instr_as_tex(ordinary_instr)->def;

   nir_push_if(b, has_custom);
   nir_def *replaced;
   {
      nir_instr *zero_instr = nir_instr_clone(b->shader, &tex->instr);
      nir_builder_instr_insert(b, zero_instr);
      nir_tex_instr *tex_zero = nir_instr_as_tex(zero_instr);
      nir_def *clamp_to_zero = &tex_zero->def;
      tex_zero->backend_flags |= AGX_TEXTURE_FLAG_CLAMP_TO_0;

      nir_def *border = query_custom_border(b, tex_zero);
      if (tex->op == nir_texop_tg4)
         border = nir_replicate(b, nir_channel(b, border, tex->component), 4);

      if (nir_alu_type_get_base_type(tex->dest_type) == nir_type_float &&
          tex->op != nir_texop_tg4) {
         replaced = nir_flrp(b, clamp_to_zero, clamp_to_one, border);
         b->shader->info.flrp_lowered = false;
      } else {
         replaced = nir_bcsel(b, nir_ieq(b, clamp_to_zero, clamp_to_one),
                              clamp_to_zero, border);
      }
   }
   nir_pop_if(b, NULL);

   nir_def *result = nir_if_phi(b, replaced, clamp_to_one);
   nir_def_replace(&tex->def, result);
   return true;
}

bool
agx_nir_lower_custom_border_g17p(nir_shader *shader)
{
   bool progress = nir_shader_tex_pass(shader, lower_custom_border,
                                       nir_metadata_none, NULL);

   if (progress) {
      bool uses_bindless_samplers = false;
      struct agx_lower_bindings_options options = {
         .uses_bindless_samplers = &uses_bindless_samplers,
         .g17p_native_sampler_handles = true,
         .bindful_sampler_count = AGX_GALLIUM_G17P_API_SAMPLERS,
      };
      progress |= nir_shader_instructions_pass(
         shader, lower, nir_metadata_control_flow, &options);

      /* Gallium variants apply this pass after the common AGX preprocess so
       * ordinary shaders do not pay for border emulation. Lower the small
       * amount of high-level NIR introduced above before handing the variant
       * to the backend. */
      NIR_PASS(_, shader, nir_lower_flrp, 16 | 32 | 64, false);
      NIR_PASS(_, shader, nir_lower_alu_to_scalar, NULL, NULL);
      NIR_PASS(_, shader, nir_lower_all_phis_to_scalar);
      NIR_PASS(_, shader, nir_opt_dce);
   }

   return progress;
}

bool
agx_nir_lower_bindings(nir_shader *shader,
                       struct agx_lower_bindings_options *options)
{
   /* First lower index to offset so we can lower more naturally */
   bool progress = nir_lower_tex(
      shader, &(nir_lower_tex_options){.lower_index_to_offset = true});

   /* Next run constant folding so the constant optimizations above have a
    * chance.
    */
   progress |= nir_opt_constant_folding(shader);

   progress |= nir_shader_instructions_pass(
      shader, lower, nir_metadata_control_flow, options);
   return progress;
}

static nir_def *
load_g17p_texture_index(nir_builder *b, nir_def *index)
{
   struct agx_stage_uniforms *s = NULL;
   nir_scalar scalar = nir_scalar_resolved(index, 0);
   mesa_shader_stage stage = b->shader->info.stage;

   if (stage == MESA_SHADER_VERTEX && b->shader->info.vs.tes_poly)
      stage = MESA_SHADER_TESS_EVAL;

   assert(stage < MESA_SHADER_STAGES);
   const unsigned table = AGX_SYSVAL_STAGE(stage);

   if (nir_scalar_is_const(scalar)) {
      const unsigned slot = nir_scalar_as_uint(scalar);
      assert(slot < PIPE_MAX_SAMPLERS);
      return nir_u2u64(
         b, nir_load_sysval_agx(
               b, 1, 16, .desc_set = table,
               .binding = (uintptr_t)&s->texture_handle[slot]));
   }

   /* Dynamic Gallium sampler arrays select the matching texture table slot.
    * Load the stage-table address from the root, then index the compact
    * uint16_t identity table just like indirect sampler handles. */
   struct agx_draw_uniforms *u = NULL;
   nir_def *table_base = nir_load_sysval_agx(
      b, 1, 64, .desc_set = AGX_SYSVAL_TABLE_ROOT,
      .binding = (uintptr_t)&u->tables[table]);
   nir_def *offset = nir_iadd_imm(
      b, nir_imul_imm(b, index, sizeof(s->texture_handle[0])),
      (uintptr_t)&s->texture_handle);
   nir_def *address = nir_iadd(b, table_base, nir_u2u64(b, offset));
   return nir_u2u64(
      b, nir_load_global_constant(b, 1, 16, address, .align_mul = 2));
}

/*
 * A descriptor crawl needs an address, so binding lowering temporarily turns
 * an otherwise bindful texture into a bindless handle.  On AGX2 the texture
 * instruction itself can consume that base-plus-byte-offset pair.  G17P
 * cannot: its indirect operand is a native table index.
 *
 * Once texture lowering has made the crawl explicit, put a statically bound
 * access in the native texture table. Slots 0..15 use the immediate field;
 * larger or dynamically selected slots use a zero-extended native table
 * index. Keep the handle alive for independent
 * load_from_texture_handle_agx users which implement the Gen9 clamp. This
 * preserves the bounds operation without presenting AGX2's base-plus-byte-
 * offset pair to HAL200's fetch/sample instruction. Filtered array samples
 * need the same treatment because their layer clamp forces a crawl.
 */
bool
agx_nir_rebind_g17p_texture_crawls(nir_shader *shader)
{
   bool progress = false;

   nir_foreach_function_impl(impl, shader) {
      nir_builder b = nir_builder_create(impl);
      bool impl_progress = false;
      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_tex)
               continue;

            nir_tex_instr *tex = nir_instr_as_tex(instr);

            const int handle_idx =
               nir_tex_instr_src_index(tex, nir_tex_src_texture_handle);
            if (handle_idx < 0)
               continue;

            nir_intrinsic_instr *handle =
               nir_src_as_intrinsic(tex->src[handle_idx].src);
            if (!handle ||
                handle->intrinsic != nir_intrinsic_load_texture_handle_agx)
               continue;

            b.cursor = nir_before_instr(instr);
            nir_steal_tex_src(tex, nir_tex_src_texture_handle);
            if (nir_src_is_const(handle->src[0]) &&
                nir_src_as_uint(handle->src[0]) <
                   AGX_NUM_TEXTURE_STATE_REGS) {
               tex->texture_index = nir_src_as_uint(handle->src[0]);
            } else {
               tex->texture_index = 0;
               nir_tex_instr_add_src(
                  tex, nir_tex_src_texture_offset,
                  load_g17p_texture_index(&b, handle->src[0].ssa));
            }
            impl_progress = true;
            progress = true;
         }
      }

      if (impl_progress)
         nir_progress(true, impl, nir_metadata_control_flow);
   }

   if (progress)
      NIR_PASS(_, shader, nir_opt_dce);

   return progress;
}

/*
 * G17P's imgwci names its destination as a PBE heap -- a 64-bit uniform --
 * plus the index of a record in that heap, held in a register, and a bindless
 * image read names its texture the same way with a 64-bit index. Gallium's
 * bindless image handle is the stage texture table (the uniform) plus a byte
 * offset into it, and every descriptor in that table, texture or PBE, is one
 * record wide. Divide the offset by the record size for the stores that reach
 * the backend. This is the Gallium side of hk_g17p_lower_storage_records().
 *
 * It must run after the sysval layout has assigned the table's uniform to the
 * handle, and after texture lowering: atomics and multisampled stores are
 * rewritten there into texel addressing that reads the descriptor through the
 * byte-offset handle, so only genuine hardware accesses remain. Each access
 * gets its own handle rather than rewriting a shared one, so any remaining
 * descriptor reader keeps the byte offset.
 */
bool
agx_nir_lower_g17p_storage_records(nir_shader *shader, unsigned record_size_B)
{
   assert(util_is_power_of_two_nonzero(record_size_B));
   bool progress = false;

   nir_foreach_function_impl(impl, shader) {
      nir_builder b = nir_builder_create(impl);
      bool impl_progress = false;

      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            const bool load =
               intr->intrinsic == nir_intrinsic_bindless_image_load ||
               intr->intrinsic == nir_intrinsic_bindless_image_sparse_load;
            if (intr->intrinsic != nir_intrinsic_bindless_image_store && !load)
               continue;

            nir_intrinsic_instr *handle = nir_src_as_intrinsic(intr->src[0]);
            if (!handle || handle->intrinsic != nir_intrinsic_bindless_image_agx)
               continue;

            b.cursor = nir_before_instr(instr);
            nir_def *record = nir_ushr_imm(&b, handle->src[0].ssa,
                                           util_logbase2(record_size_B));
            /* A read names its texture descriptor the way a G17P texture
             * fetch does (agx_nir_rebind_g17p_texture_crawls()): a native
             * table index, zero-extended to the 64-bit register operand. */
            if (load)
               record = nir_u2u64(&b, record);
            nir_src_rewrite(&intr->src[0],
                            nir_bindless_image_agx(
                               &b, record,
                               .desc_set = nir_intrinsic_desc_set(handle)));
            impl_progress = true;
         }
      }

      if (impl_progress) {
         nir_progress(true, impl, nir_metadata_control_flow);
         progress = true;
      }
   }

   return progress;
}
