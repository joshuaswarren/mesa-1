/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_nir.h"
#include "agx_nir_texture.h"

#include "nir.h"
#include "nir_builder.h"

static bool
g17p_sample_shape(const nir_tex_instr *tex)
{
   if ((tex->op != nir_texop_tex && tex->op != nir_texop_txb &&
       tex->op != nir_texop_txl) ||
       tex->sampler_dim != GLSL_SAMPLER_DIM_2D ||
       tex->is_shadow || tex->is_sparse ||
       tex->coord_components != (tex->is_array ? 3 : 2) ||
       (tex->backend_flags & ~(AGX_TEXTURE_FLAG_CLAMP_TO_0 |
                               AGX_TEXTURE_FLAG_NO_CLAMP)) != 0 ||
       tex->backend_flags == (AGX_TEXTURE_FLAG_CLAMP_TO_0 |
                              AGX_TEXTURE_FLAG_NO_CLAMP) ||
       tex->dest_type != nir_type_float32 || tex->def.num_components != 4 ||
       tex->def.bit_size != 32)
      return false;

   const bool has_bias =
      nir_tex_instr_src_index(tex, nir_tex_src_bias) >= 0;
   const bool has_lod = nir_tex_instr_src_index(tex, nir_tex_src_lod) >= 0;
   if (tex->num_srcs != ((has_bias || has_lod) ? 4 : 3) ||
       has_lod != (tex->op == nir_texop_txl) ||
       has_bias != (tex->op == nir_texop_txb))
      return false;

   int coord = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   if (coord < 0)
      coord = nir_tex_instr_src_index(tex, nir_tex_src_backend1);

   return coord >= 0 &&
          nir_src_num_components(tex->src[coord].src) ==
             tex->coord_components &&
          nir_src_bit_size(tex->src[coord].src) == 32 &&
          nir_tex_instr_src_index(tex, nir_tex_src_texture_handle) >= 0 &&
          nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle) >= 0;
}

static bool
g17p_handle_binding(const nir_tex_instr *tex, nir_tex_src_type source,
                    nir_intrinsic_op intrinsic, uint8_t *set,
                    uint32_t *offset_B)
{
   const int index = nir_tex_instr_src_index(tex, source);
   if (index < 0)
      return false;

   nir_intrinsic_instr *handle = nir_src_as_intrinsic(tex->src[index].src);
   if (!handle || handle->intrinsic != intrinsic ||
       nir_intrinsic_desc_set(handle) > UINT8_MAX ||
       !nir_src_is_const(handle->src[0]))
      return false;

   *set = nir_intrinsic_desc_set(handle);
   *offset_B = nir_src_as_uint(handle->src[0]);
   return true;
}

static bool
g17p_handle_offset(const nir_tex_instr *tex, nir_tex_src_type source,
                   nir_intrinsic_op intrinsic, uint32_t *offset_B)
{
   uint8_t set;
   return g17p_handle_binding(tex, source, intrinsic, &set, offset_B) &&
          set == 0;
}

static bool
g17p_is_set_address(nir_def *def, uint8_t set)
{
   nir_scalar address = nir_scalar_resolved(def, 0);
   nir_intrinsic_instr *load =
      nir_def_as_intrinsic_or_null(address.def);

   return address.comp == 0 && load &&
          load->intrinsic == nir_intrinsic_load_descriptor_set_agx &&
          nir_intrinsic_desc_set(load) == set;
}

static bool
g17p_is_constant_scalar(nir_scalar offset, uint32_t expected)
{
   offset = nir_scalar_chase_movs(offset);
   if (nir_scalar_is_const(offset))
      return nir_scalar_as_uint(offset) == expected;

   nir_alu_instr *convert = nir_def_as_alu_or_null(offset.def);
   if (offset.comp != 0 || !convert || convert->op != nir_op_u2u64)
      return false;

   return g17p_is_constant_scalar(nir_scalar_chase_alu_src(offset, 0),
                                  expected);
}

static bool
g17p_is_constant_offset(nir_def *def, uint32_t expected)
{
   return g17p_is_constant_scalar(nir_get_scalar(def, 0), expected);
}

/* Match only the driver-generated software LOD-bias field belonging to this
 * sampled descriptor. API samplers carry the same bias in their G17P native
 * sampler word, so this zero-valued software load must not become a txb.
 */
static bool
g17p_is_sampler_bias(const nir_tex_instr *tex, uint8_t set,
                     uint32_t expected_offset_B)
{
   const int index = nir_tex_instr_src_index(tex, nir_tex_src_bias);
   if (index < 0)
      return tex->op == nir_texop_tex || tex->op == nir_texop_txl;

   nir_intrinsic_instr *load = nir_src_as_intrinsic(tex->src[index].src);
   if (!load || load->def.num_components != 1 || load->def.bit_size != 16)
      return false;

   if (load->intrinsic == nir_intrinsic_load_global_constant_offset ||
       load->intrinsic == nir_intrinsic_load_global_constant_bounded) {
      return g17p_is_set_address(load->src[0].ssa, set) &&
             g17p_is_constant_offset(load->src[1].ssa, expected_offset_B);
   }

   if (load->intrinsic != nir_intrinsic_load_global_constant)
      return false;

   nir_scalar address = nir_scalar_resolved(load->src[0].ssa, 0);
   nir_alu_instr *add = nir_def_as_alu_or_null(address.def);
   if (address.comp != 0 || !add || add->op != nir_op_iadd)
      return false;

   nir_scalar lhs = nir_scalar_chase_alu_src(address, 0);
   nir_scalar rhs = nir_scalar_chase_alu_src(address, 1);

   return (g17p_is_set_address(lhs.def, set) &&
           g17p_is_constant_scalar(rhs, expected_offset_B)) ||
          (g17p_is_set_address(rhs.def, set) &&
           g17p_is_constant_scalar(lhs, expected_offset_B));
}

/* The native texture/sampler table prefix is separate from the fragment
 * push-root ABI. Residual descriptor accesses need their own lowering, but a
 * root load used by push constants is valid alongside this sampled resource. */
static bool
g17p_has_descriptor_access(const nir_shader *shader)
{
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            const nir_intrinsic_op op =
               nir_instr_as_intrinsic(instr)->intrinsic;
            if (op == nir_intrinsic_bindless_image_agx ||
                op == nir_intrinsic_bindless_sampler_agx ||
                op == nir_intrinsic_load_descriptor_set_agx)
               return true;
         }
      }
   }

   return false;
}

bool
agx_nir_lower_g17p_combined_sampler(
   nir_shader *shader, struct agx_g17p_combined_sampler_binding *binding)
{
   if (!shader || !binding ||
       (shader->info.stage != MESA_SHADER_FRAGMENT &&
        shader->info.stage != MESA_SHADER_COMPUTE))
      return false;

   nir_tex_instr *sample = NULL;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_tex)
               continue;

            if (sample || !g17p_sample_shape(nir_instr_as_tex(instr)))
               return false;
            sample = nir_instr_as_tex(instr);
         }
      }
   }

   if (!sample)
      return false;

   uint32_t texture_offset_B, sampler_state_offset_B;
   if (!g17p_handle_offset(sample, nir_tex_src_texture_handle,
                           nir_intrinsic_bindless_image_agx,
                           &texture_offset_B) ||
       !g17p_handle_offset(sample, nir_tex_src_sampler_handle,
                           nir_intrinsic_bindless_sampler_agx,
                           &sampler_state_offset_B))
      return false;

   /* Gen9 sampled descriptors contain one 32-byte texture followed by the
    * eight-byte native sampler word and its 16-bit global-heap index.
    */
   if ((texture_offset_B & 7) != 0 ||
       sampler_state_offset_B != texture_offset_B + 32)
      return false;

   /* The Gen9 software LOD bias is at byte 12 of sampler state. */
   if (!g17p_is_sampler_bias(sample, 0, sampler_state_offset_B + 12))
      return false;

   nir_steal_tex_src(sample, nir_tex_src_texture_handle);
   nir_steal_tex_src(sample, nir_tex_src_sampler_handle);
   if (sample->op == nir_texop_txb) {
      nir_steal_tex_src(sample, nir_tex_src_bias);
      sample->op = nir_texop_tex;
   }
   sample->texture_index = 0;
   /* HAL200's bounded fragment ABI binds independent texture/sampler table
    * pointers. Each table is rebased to the selected API resource, so both
    * hardware indices are zero; the legacy texel-fetch reservation is not
    * an extra entry in this sampler table.
    */
   sample->sampler_index = 0;

   NIR_PASS(_, shader, nir_opt_dce);
   /* The initial fragment ABI replaces the whole descriptor root. Compute
    * reserves the same native texture/sampler pointer prefix but may keep
    * unrelated storage-image/SSBO descriptors in the ordinary root. */
   if (shader->info.stage == MESA_SHADER_FRAGMENT &&
       g17p_has_descriptor_access(shader))
      return false;

   *binding = (struct agx_g17p_combined_sampler_binding){
      .texture_offset_B = texture_offset_B,
      .sampler_index_offset_B = sampler_state_offset_B + 8,
   };
   return true;
}

/* Ordinary native integer-coordinate read. The descriptor selects float,
 * signed, or unsigned texel interpretation; the result is a 32-bit register
 * value in each case. Input attachments arrive as 2D-array reads. */
static bool
g17p_indexed_fetch_shape(const nir_tex_instr *tex)
{
   int coord = nir_tex_instr_src_index(tex, nir_tex_src_backend1);
   if (coord < 0)
      coord = nir_tex_instr_src_index(tex, nir_tex_src_coord);

   /* Buffer textures are lowered to a 2D hardware texture before this pass.
    * That lowering deliberately preserves coord_components == 1 while its
    * backend coordinate is the remapped vec2.  Admit exactly that canonical
    * mismatch alongside ordinary 2D/2D-array fetches.
    */
   const bool lowered_buffer =
      tex->sampler_dim == GLSL_SAMPLER_DIM_2D && !tex->is_array &&
      tex->coord_components == 1 && coord >= 0 &&
      tex->src[coord].src.ssa->num_components == 2;

   if (tex->op != nir_texop_txf || tex->sampler_dim != GLSL_SAMPLER_DIM_2D ||
       tex->is_shadow || tex->is_sparse || tex->texture_index ||
       tex->sampler_index || tex->def.bit_size != 32 ||
       tex->def.num_components != 4 ||
       (tex->dest_type != nir_type_float32 && tex->dest_type != nir_type_int32 &&
        tex->dest_type != nir_type_uint32) ||
       (!lowered_buffer &&
        tex->coord_components != (tex->is_array ? 3 : 2)) ||
       (tex->backend_flags & ~AGX_TEXTURE_FLAG_NO_CLAMP))
      return false;
   const int handle = nir_tex_instr_src_index(tex, nir_tex_src_texture_handle);
   const int lod = nir_tex_instr_src_index(tex, nir_tex_src_lod);
   return handle >= 0 && lod >= 0 && coord >= 0 && tex->num_srcs == 3 &&
      nir_src_is_const(tex->src[lod].src) &&
      nir_src_as_uint(tex->src[lod].src) == 0 &&
      nir_src_bit_size(tex->src[coord].src) == 32 &&
      nir_src_num_components(tex->src[coord].src) ==
         (lowered_buffer ? 2 : tex->coord_components);
}

/* Descriptor lowering normally wraps the software heap index in a bindless
 * sampler. The custom-border zero-clamp sample already carries the companion
 * index directly: its descriptor load and nonnegative runtime branch must
 * remain intact. Both indices include the same AGX2 reservation of 28. */
static nir_def *
g17p_sample_heap_index(const nir_tex_instr *tex)
{
   const int index = nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle);
   if (index < 0)
      return NULL;

   nir_src handle = tex->src[index].src;
   nir_intrinsic_instr *sampler = nir_src_as_intrinsic(handle);
   nir_def *value = NULL;
   if (sampler && sampler->intrinsic == nir_intrinsic_bindless_sampler_agx)
      value = sampler->src[1].ssa;
   else if (tex->backend_flags & AGX_TEXTURE_FLAG_CLAMP_TO_0)
      value = handle.ssa;

   return value && value->num_components == 1 && value->bit_size == 16
             ? value : NULL;
}

/* Native indexed samples consume zero-extended heap indices, loaded by the
 * GPU from the descriptor set. Keep descriptor-set roots alive so updates
 * after command recording and dynamically selected array elements are visible.
 */
bool
agx_nir_lower_g17p_indexed_samplers(nir_shader *shader)
{
   if (!shader || (shader->info.stage != MESA_SHADER_FRAGMENT &&
                   shader->info.stage != MESA_SHADER_COMPUTE))
      return false;

   unsigned samples = 0;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_tex)
               continue;
            nir_tex_instr *tex = nir_instr_as_tex(instr);
            const bool fetch = g17p_indexed_fetch_shape(tex);
            if (!fetch && (!g17p_sample_shape(tex) ||
                           (tex->op != nir_texop_tex && tex->op != nir_texop_txl)))
               return false;
            nir_intrinsic_instr *image = nir_src_as_intrinsic(
               tex->src[nir_tex_instr_src_index(tex, nir_tex_src_texture_handle)].src);
            if (!image || image->intrinsic != nir_intrinsic_bindless_image_agx ||
                (!fetch && !g17p_sample_heap_index(tex)))
               return false;
            samples++;
         }
      }
   }
   if (!samples)
      return false;

   nir_foreach_function_impl(impl, shader) {
      nir_builder b = nir_builder_create(impl);
      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_tex)
               continue;
            nir_tex_instr *tex = nir_instr_as_tex(instr);
            b.cursor = nir_before_instr(instr);
            nir_intrinsic_instr *image = nir_src_as_intrinsic(
               tex->src[nir_tex_instr_src_index(tex, nir_tex_src_texture_handle)].src);
            const bool fetch = tex->op == nir_texop_txf;
            nir_def *sampler = fetch ? NULL : g17p_sample_heap_index(tex);
            nir_def *set = nir_load_descriptor_set_agx(&b, nir_intrinsic_desc_set(image));
            /* Descriptor offsets are 32-bit and valid sets fit far below
             * 4GiB. Add the field offset before widening so native LDDEV can
             * consume base+byte-index without a separate 64-bit base add. */
            nir_def *offset = nir_iadd_imm(&b, image->src[0].ssa,
                                          AGX_G17P_SAMPLED_HEAP_INDEX_OFFSET);
            nir_def *address = nir_iadd(&b, set, nir_u2u64(&b, offset));
            nir_def *texture_index = nir_load_global_constant(&b, 1, 32, address, .align_mul = 4);
            /* The software sampler index includes AGX2's reservation of 28.
             * Native G17P heaps are zero-based, as the immediate path is. */
            texture_index = nir_u2u64(&b, texture_index);
            nir_steal_tex_src(tex, nir_tex_src_texture_handle);
            nir_tex_instr_add_src(tex, nir_tex_src_texture_offset, texture_index);
            if (!fetch) {
               nir_def *sampler_index = nir_iadd_imm(
                  &b, nir_u2u32(&b, sampler), -28);
               sampler_index = nir_u2u64(&b, sampler_index);
               nir_steal_tex_src(tex, nir_tex_src_sampler_handle);
               nir_tex_instr_add_src(tex, nir_tex_src_sampler_handle, sampler_index);
            }
            /* Fetch uses native sampler0, reserved in the same device heap
             * that ordinary samples use, so mixed reads share one prefix. */
         }
      }
      nir_progress(true, impl, nir_metadata_control_flow);
   }
   NIR_PASS(_, shader, nir_opt_dce);
   return true;
}

static bool
g17p_cube_sample_shape(const nir_tex_instr *tex)
{
   if (tex->op != nir_texop_tex ||
       tex->sampler_dim != GLSL_SAMPLER_DIM_CUBE || tex->is_array ||
       tex->is_shadow || tex->is_sparse || tex->coord_components != 3 ||
       tex->num_srcs != 3 ||
       (tex->backend_flags & ~AGX_TEXTURE_FLAG_CLAMP_TO_0) != 0 ||
       tex->dest_type != nir_type_float32 || tex->def.num_components != 4 ||
       tex->def.bit_size != 32 ||
       nir_tex_instr_src_index(tex, nir_tex_src_texture_handle) < 0 ||
       nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle) < 0)
      return false;

   int coord = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   if (coord < 0)
      coord = nir_tex_instr_src_index(tex, nir_tex_src_backend1);

   return coord >= 0 && nir_src_num_components(tex->src[coord].src) == 3 &&
          nir_src_bit_size(tex->src[coord].src) == 32;
}

/* Lower the exact implicit, non-array cube sample to the already-qualified
 * G17P 2D-array path. Cube images are six consecutive 2D layers; select the
 * major axis and map its face coordinate to [0, 1] before the ordinary array
 * layer marshalling pass. The face ordering and transforms are the Vulkan/GL
 * cube convention (the same mapping used by Mesa's software samplers). */
bool
agx_nir_lower_g17p_cube_samples(nir_shader *shader)
{
   if (!shader || shader->info.stage != MESA_SHADER_FRAGMENT)
      return false;

   unsigned cubes = 0;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_tex)
               continue;

            nir_tex_instr *tex = nir_instr_as_tex(instr);
            if (tex->sampler_dim != GLSL_SAMPLER_DIM_CUBE)
               continue;
            if (!g17p_cube_sample_shape(tex))
               return false;
            cubes++;
         }
      }
   }

   if (!cubes)
      return false;

   nir_foreach_function_impl(impl, shader) {
      nir_builder b = nir_builder_create(impl);
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_tex)
               continue;

            nir_tex_instr *tex = nir_instr_as_tex(instr);
            if (tex->sampler_dim != GLSL_SAMPLER_DIM_CUBE)
               continue;

            int coord_index = nir_tex_instr_src_index(tex, nir_tex_src_coord);
            if (coord_index < 0)
               coord_index =
                  nir_tex_instr_src_index(tex, nir_tex_src_backend1);

            nir_def *coord = tex->src[coord_index].src.ssa;
            b.cursor = nir_before_instr(instr);
            nir_def *x = nir_channel(&b, coord, 0);
            nir_def *y = nir_channel(&b, coord, 1);
            nir_def *z = nir_channel(&b, coord, 2);
            nir_def *x2 = nir_fmul(&b, x, x);
            nir_def *y2 = nir_fmul(&b, y, y);
            nir_def *z2 = nir_fmul(&b, z, z);
            nir_def *half = nir_imm_float(&b, 0.5f);

            nir_def *x_positive = nir_fge_imm(&b, x, 0.0f);
            nir_def *y_positive = nir_fge_imm(&b, y, 0.0f);
            nir_def *z_positive = nir_fge_imm(&b, z, 0.0f);

            /* rsqrt(c*c) gives 1/abs(c) without materializing fabs/fneg,
             * whose standalone AGX3 forms are not yet qualified. */
            nir_def *rx = nir_frsq(&b, x2);
            nir_def *ry = nir_frsq(&b, y2);
            nir_def *rz = nir_frsq(&b, z2);
            nir_def *px = nir_fmul_imm(&b, rx, 0.5f);
            nir_def *nx = nir_fmul_imm(&b, rx, -0.5f);
            nir_def *py = nir_fmul_imm(&b, ry, 0.5f);
            nir_def *ny = nir_fmul_imm(&b, ry, -0.5f);
            nir_def *pz = nir_fmul_imm(&b, rz, 0.5f);
            nir_def *nz = nir_fmul_imm(&b, rz, -0.5f);

            nir_def *face_x = nir_vec3(
               &b,
               nir_bcsel(&b, x_positive, nir_ffma(&b, nx, z, half),
                          nir_ffma(&b, px, z, half)),
               nir_ffma(&b, nx, y, half),
               nir_bcsel(&b, x_positive, nir_imm_float(&b, 0.0f),
                          nir_imm_float(&b, 1.0f)));
            nir_def *face_y = nir_vec3(
               &b, nir_ffma(&b, py, x, half),
               nir_bcsel(&b, y_positive, nir_ffma(&b, py, z, half),
                          nir_ffma(&b, ny, z, half)),
               nir_bcsel(&b, y_positive, nir_imm_float(&b, 2.0f),
                          nir_imm_float(&b, 3.0f)));
            nir_def *face_z = nir_vec3(
               &b,
               nir_bcsel(&b, z_positive, nir_ffma(&b, pz, x, half),
                          nir_ffma(&b, nz, x, half)),
               nir_ffma(&b, nz, y, half),
               nir_bcsel(&b, z_positive, nir_imm_float(&b, 4.0f),
                          nir_imm_float(&b, 5.0f)));

            nir_def *use_x = nir_iand(&b, nir_fge(&b, x2, y2),
                                      nir_fge(&b, x2, z2));
            nir_def *use_y = nir_iand(&b, nir_fge(&b, y2, x2),
                                      nir_fge(&b, y2, z2));
            nir_def *mapped = nir_bcsel(
               &b, use_x, face_x, nir_bcsel(&b, use_y, face_y, face_z));

            nir_src_rewrite(&tex->src[coord_index].src, mapped);
            tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
            tex->is_array = true;
            tex->array_is_lowered_cube = true;
         }
      }

      nir_progress(true, impl, nir_metadata_control_flow);
   }

   NIR_PASS(_, shader, nir_opt_dce);
   return true;
}

static bool
g17p_txf_shape(const nir_tex_instr *tex)
{
   if (tex->op != nir_texop_txf ||
       tex->sampler_dim != GLSL_SAMPLER_DIM_2D || tex->is_shadow ||
       tex->is_sparse || (tex->coord_components != 2 &&
                          tex->coord_components != 3) ||
       tex->is_array != (tex->coord_components == 3) ||
       (tex->backend_flags & ~AGX_TEXTURE_FLAG_NO_CLAMP) != 0 ||
       tex->dest_type != nir_type_uint32 || tex->def.num_components != 4 ||
       tex->def.bit_size != 32 || tex->texture_index != 0 ||
       tex->sampler_index != 0)
      return false;

   const int handle =
      nir_tex_instr_src_index(tex, nir_tex_src_texture_handle);
   const int lod = nir_tex_instr_src_index(tex, nir_tex_src_lod);
   int coord = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   if (coord < 0)
      coord = nir_tex_instr_src_index(tex, nir_tex_src_backend1);

   if (handle < 0 || lod < 0 || coord < 0 || tex->num_srcs != 3 ||
       nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle) >= 0 ||
       nir_tex_instr_src_index(tex, nir_tex_src_sampler_offset) >= 0 ||
       nir_tex_instr_src_index(tex, nir_tex_src_texture_offset) >= 0 ||
       !nir_src_is_const(tex->src[lod].src) ||
       nir_src_as_uint(tex->src[lod].src) != 0)
      return false;

   return nir_src_num_components(tex->src[coord].src) ==
             tex->coord_components &&
          nir_src_bit_size(tex->src[coord].src) == 32;
}

static bool
g17p_has_bindless_texture_access(const nir_shader *shader)
{
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            const nir_intrinsic_op op =
               nir_instr_as_intrinsic(instr)->intrinsic;
            if (op == nir_intrinsic_bindless_image_agx ||
                op == nir_intrinsic_bindless_sampler_agx)
               return true;
         }
      }
   }

   return false;
}

bool
agx_nir_lower_g17p_txf(nir_shader *shader,
                        struct agx_g17p_txf_binding *binding)
{
   if (!shader || !binding || shader->info.stage != MESA_SHADER_COMPUTE)
      return false;

   nir_tex_instr *fetch = NULL;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_tex)
               continue;

            if (fetch || !g17p_txf_shape(nir_instr_as_tex(instr)))
               return false;
            fetch = nir_instr_as_tex(instr);
         }
      }
   }

   if (!fetch)
      return false;

   uint32_t texture_offset_B;
   if (!g17p_handle_offset(fetch, nir_tex_src_texture_handle,
                           nir_intrinsic_bindless_image_agx,
                           &texture_offset_B) ||
       (texture_offset_B & 7) != 0)
      return false;

   nir_steal_tex_src(fetch, nir_tex_src_texture_handle);
   fetch->texture_index = 0;

   NIR_PASS(_, shader, nir_opt_dce);
   if (g17p_has_bindless_texture_access(shader))
      return false;

   *binding = (struct agx_g17p_txf_binding){
      .texture_offset_B = texture_offset_B,
   };
   return true;
}

static bool
g17p_storage_handle_binding(const nir_intrinsic_instr *intr, uint8_t *set,
                            uint32_t *offset_B)
{
   nir_intrinsic_instr *handle = nir_src_as_intrinsic(intr->src[0]);
   if (!handle || handle->intrinsic != nir_intrinsic_bindless_image_agx ||
       nir_intrinsic_desc_set(handle) > UINT8_MAX ||
       !nir_src_is_const(handle->src[0]))
      return false;

   *set = nir_intrinsic_desc_set(handle);
   *offset_B = nir_src_as_uint(handle->src[0]);
   return true;
}

static bool
g17p_storage_handle_offset(const nir_intrinsic_instr *intr,
                           uint32_t *offset_B)
{
   uint8_t set;
   return g17p_storage_handle_binding(intr, &set, offset_B) && set == 0;
}

static bool
g17p_storage_image_shape(const nir_intrinsic_instr *intr, bool load)
{
   if (intr->intrinsic != (load ? nir_intrinsic_bindless_image_load
                                : nir_intrinsic_bindless_image_store) ||
       nir_intrinsic_image_dim(intr) != GLSL_SAMPLER_DIM_2D ||
       nir_intrinsic_image_array(intr) ||
       nir_intrinsic_format(intr) != PIPE_FORMAT_R8G8B8A8_UINT ||
       nir_src_num_components(intr->src[1]) != 4 ||
       nir_src_bit_size(intr->src[1]) != 32 ||
       !nir_src_is_const(intr->src[2]) ||
       nir_src_as_uint(intr->src[2]) != 0)
      return false;

   if (load) {
      return intr->def.num_components == 4 && intr->def.bit_size == 32 &&
             !nir_intrinsic_image_array(intr) &&
             nir_src_is_const(intr->src[3]) &&
             nir_src_as_uint(intr->src[3]) == 0 &&
             !(nir_intrinsic_access(intr) &
               (ACCESS_COHERENT | ACCESS_VOLATILE));
   }

   return nir_src_num_components(intr->src[3]) == 4 &&
          nir_src_bit_size(intr->src[3]) == 32 &&
          nir_src_is_const(intr->src[4]) &&
          nir_src_as_uint(intr->src[4]) == 0 &&
          !(nir_intrinsic_access(intr) &
            (ACCESS_COHERENT | ACCESS_VOLATILE));
}

bool
agx_nir_lower_g17p_storage_image(
   nir_shader *shader, struct agx_g17p_storage_image_binding *binding)
{
   if (!shader || !binding || shader->info.stage != MESA_SHADER_COMPUTE)
      return false;

   nir_intrinsic_instr *load = NULL, *store = NULL;
   uint32_t texture_offset_B = 0, pbe_offset_B = 0;
   unsigned impl_count = 0, block_count = 0;
   nir_foreach_function_impl(impl, shader) {
      impl_count++;
      nir_foreach_block(block, impl) {
         block_count++;
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_tex)
               return false;
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_bindless_image_load) {
               if (load || !g17p_storage_image_shape(intr, true) ||
                   !g17p_storage_handle_offset(intr, &texture_offset_B))
                  return false;
               load = intr;
            } else if (intr->intrinsic == nir_intrinsic_bindless_image_store) {
               if (!load || store || !g17p_storage_image_shape(intr, false) ||
                   !g17p_storage_handle_offset(intr, &pbe_offset_B))
                  return false;
               store = intr;
            } else if (intr->intrinsic == nir_intrinsic_bindless_image_sparse_load ||
                       intr->intrinsic == nir_intrinsic_bindless_image_atomic ||
                       intr->intrinsic == nir_intrinsic_bindless_image_atomic_swap ||
                       intr->intrinsic == nir_intrinsic_bindless_image_size ||
                       intr->intrinsic == nir_intrinsic_bindless_image_samples) {
               return false;
            }
         }
      }
   }

   /* A Gen9 storage descriptor is texture32 followed by PBE32. Requiring this
    * exact pair proves both operations name the same API image. */
   if (impl_count != 1 || block_count != 1 || !load || !store ||
       store->src[3].ssa != &load->def || (texture_offset_B & 7) != 0 ||
       pbe_offset_B != texture_offset_B + 32)
      return false;

   nir_foreach_function_impl(impl, shader) {
      nir_builder b = nir_builder_create(impl);
      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr != &load->instr && instr != &store->instr)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            b.cursor = nir_before_instr(instr);
            nir_src_rewrite(&intr->src[0],
                            nir_imm_intN_t(&b, 0, 16));
            intr->intrinsic = intr == load ? nir_intrinsic_image_load
                                           : nir_intrinsic_image_store;
         }
      }
      nir_progress(true, impl, nir_metadata_control_flow);
   }

   NIR_PASS(_, shader, nir_opt_dce);
   if (g17p_has_bindless_texture_access(shader))
      return false;

   *binding = (struct agx_g17p_storage_image_binding){
      .texture_offset_B = texture_offset_B,
      .pbe_offset_B = pbe_offset_B,
   };
   return true;
}

static bool
g17p_sampled_storage_store_shape(const nir_intrinsic_instr *intr)
{
   return intr->intrinsic == nir_intrinsic_bindless_image_store &&
          nir_intrinsic_image_dim(intr) == GLSL_SAMPLER_DIM_2D &&
          !nir_intrinsic_image_array(intr) &&
          nir_intrinsic_format(intr) == PIPE_FORMAT_R8G8B8A8_UNORM &&
          nir_src_num_components(intr->src[1]) == 4 &&
          nir_src_bit_size(intr->src[1]) == 32 &&
          nir_src_num_components(intr->src[3]) == 4 &&
          nir_src_bit_size(intr->src[3]) == 32 &&
          nir_src_is_const(intr->src[2]) &&
          nir_src_as_uint(intr->src[2]) == 0 &&
          nir_src_is_const(intr->src[4]) &&
          nir_src_as_uint(intr->src[4]) == 0 &&
          !(nir_intrinsic_access(intr) &
            (ACCESS_COHERENT | ACCESS_VOLATILE));
}

bool
agx_nir_lower_g17p_sampled_storage(
   nir_shader *shader, struct agx_g17p_sampled_storage_binding *binding)
{
   if (!shader || !binding || shader->info.stage != MESA_SHADER_COMPUTE)
      return false;

   nir_tex_instr *sample = NULL;
   nir_intrinsic_instr *store = NULL;
   unsigned impl_count = 0, block_count = 0;
   nir_foreach_function_impl(impl, shader) {
      impl_count++;
      nir_foreach_block(block, impl) {
         block_count++;
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_tex) {
               if (sample || !g17p_sample_shape(nir_instr_as_tex(instr)))
                  return false;
               sample = nir_instr_as_tex(instr);
               continue;
            }

            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_bindless_image_store) {
               if (store || !g17p_sampled_storage_store_shape(intr))
                  return false;
               store = intr;
            } else if (intr->intrinsic == nir_intrinsic_bindless_image_load ||
                       intr->intrinsic == nir_intrinsic_bindless_image_sparse_load ||
                       intr->intrinsic == nir_intrinsic_bindless_image_atomic ||
                       intr->intrinsic == nir_intrinsic_bindless_image_atomic_swap ||
                       intr->intrinsic == nir_intrinsic_bindless_image_size ||
                       intr->intrinsic == nir_intrinsic_bindless_image_samples) {
               return false;
            }
         }
      }
   }

   /* This protocol is derived from Apple's straight-line kernel with one
    * explicit level-zero sample feeding one rgba8 write.  Keep all control
    * flow and independent data transforms outside the qualified shape. */
   if (impl_count != 1 || block_count != 1 || !sample || !store ||
       store->src[3].ssa != &sample->def || sample->op != nir_texop_txl)
      return false;

   const int lod = nir_tex_instr_src_index(sample, nir_tex_src_lod);
   if (lod < 0 || !nir_src_is_const(sample->src[lod].src) ||
       nir_src_as_uint(sample->src[lod].src) != 0)
      return false;

   uint8_t texture_set, sampler_set, pbe_set;
   uint32_t texture_offset_B, sampler_state_offset_B, storage_offset_B;
   if (!g17p_handle_binding(sample, nir_tex_src_texture_handle,
                            nir_intrinsic_bindless_image_agx, &texture_set,
                            &texture_offset_B) ||
       !g17p_handle_binding(sample, nir_tex_src_sampler_handle,
                            nir_intrinsic_bindless_sampler_agx, &sampler_set,
                            &sampler_state_offset_B) ||
       !g17p_storage_handle_binding(store, &pbe_set, &storage_offset_B) ||
       texture_set != sampler_set || (texture_offset_B & 7) != 0 ||
       sampler_state_offset_B != texture_offset_B + 32 ||
       !g17p_is_sampler_bias(sample, texture_set,
                             sampler_state_offset_B + 12) ||
       (storage_offset_B & 7) != 0)
      return false;

   nir_steal_tex_src(sample, nir_tex_src_texture_handle);
   nir_steal_tex_src(sample, nir_tex_src_sampler_handle);
   if (sample->op == nir_texop_txb) {
      nir_steal_tex_src(sample, nir_tex_src_bias);
      sample->op = nir_texop_tex;
   }
   sample->texture_index = 0;
   sample->sampler_index = 0;

   nir_builder b = nir_builder_create(nir_shader_get_entrypoint(shader));
   b.cursor = nir_before_instr(&store->instr);
   nir_src_rewrite(&store->src[0], nir_imm_intN_t(&b, 1, 16));
   store->intrinsic = nir_intrinsic_image_store;

   NIR_PASS(_, shader, nir_opt_dce);
   if (g17p_has_bindless_texture_access(shader))
      return false;

   *binding = (struct agx_g17p_sampled_storage_binding){
      .texture_offset_B = texture_offset_B,
      .sampler_index_offset_B = sampler_state_offset_B + 8,
      .pbe_offset_B = storage_offset_B,
      .texture_set = texture_set,
      .pbe_set = pbe_set,
   };
   return true;
}
