/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT */
#include "d12_nir_g17p_txf.h"
#include "dxil/d12_dxil.h"
#include "agx_nir.h"
#include "nir_builder.h"

bool
d12_nir_g17p_txf_supported(nir_shader *nir, bool *has_txf)
{
   if (!nir || !has_txf || nir->info.stage != MESA_SHADER_COMPUTE)
      return false;
   bool found = false;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_intrinsic_has_image_dim(nir_instr_as_intrinsic(instr)))
               return false;
            if (instr->type != nir_instr_type_tex)
               continue;
            nir_tex_instr *tex = nir_instr_as_tex(instr);
            int h = nir_tex_instr_src_index(tex, nir_tex_src_texture_handle);
            if (tex->op != nir_texop_txf ||
                tex->sampler_dim != GLSL_SAMPLER_DIM_2D || tex->is_array ||
                tex->is_shadow || tex->coord_components != 2 || h < 0 ||
                nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle) >= 0)
               return false;
            nir_intrinsic_instr *handle = nir_src_as_intrinsic(tex->src[h].src);
            if (!handle || handle->intrinsic != nir_intrinsic_bindless_image_agx ||
                nir_intrinsic_desc_set(handle) != D12_UNIFORM_TEXTURES ||
                nir_src_bit_size(handle->src[0]) != 32 ||
                nir_src_num_components(handle->src[0]) != 1 ||
                (nir_src_is_const(handle->src[0]) &&
                 (nir_src_as_uint(handle->src[0]) & 31)))
               return false;
            found = true;
         }
      }
   }
   *has_txf = found;
   return true;
}

bool
d12_nir_lower_g17p_txf(nir_shader *nir, uint32_t generation)
{
   bool found = false;
   if (generation != 9 || !d12_nir_g17p_txf_supported(nir, &found) || !found)
      return false;
   nir_foreach_function_impl(impl, nir) {
      nir_builder b = nir_builder_create(impl);
      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_tex)
               continue;
            nir_tex_instr *tex = nir_instr_as_tex(instr);
            int h = nir_tex_instr_src_index(tex, nir_tex_src_texture_handle);
            nir_intrinsic_instr *handle = nir_src_as_intrinsic(tex->src[h].src);
            b.cursor = nir_before_instr(instr);
            /* The D12 producer's ordinary table is a packed native32 array.
             * Its intrinsic already carries a byte offset in that array,
             * including dynamic declaration segment bases. HK's native
             * indexed fetch consumes the descriptor index, zero-extended to
             * a full GPR pair. No HK descriptor-set trailer exists here. */
            nir_def *index = nir_u2u64(&b, nir_ushr_imm(&b, handle->src[0].ssa, 5));
            nir_steal_tex_src(tex, nir_tex_src_texture_handle);
            nir_tex_instr_add_src(tex, nir_tex_src_texture_offset, index);
         }
      }
      nir_progress(true, impl, nir_metadata_control_flow);
   }
   NIR_PASS(_, nir, nir_opt_dce);
   return true;
}

static bool
shift_root(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   (void)b;
   (void)data;
   if (intr->intrinsic != nir_intrinsic_load_preamble ||
       nir_intrinsic_base(intr) != D12_UNIFORM_ROOT)
      return false;
   /* Native texture and txf-sampler pointers own halves0..7. Reserve root8
    * like HK; all other D12 ABI ranges begin at16 and remain unchanged. */
   nir_intrinsic_set_base(intr, AGX_G17P_TXF_ROOT_HALFS);
   return true;
}

bool
d12_nir_shift_g17p_txf_root(nir_shader *nir)
{
   return nir_shader_intrinsics_pass(nir, shift_root,
                                    nir_metadata_control_flow, NULL);
}
