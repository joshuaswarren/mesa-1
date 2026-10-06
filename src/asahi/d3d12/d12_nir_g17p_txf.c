/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT */
#include "d12_nir_g17p_txf.h"
#include "d12_debug.h"
#include <stdio.h>
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

/* The HAL200 fragment sample instruction takes immediate texture and sampler
 * indices against two table pointers in uniform halves 0..7. One sample per
 * shader keeps both tables one entry long, so index zero names the API
 * resource after the driver rebases each pointer onto it. */
/* Name the shape that kept a fragment shader on the bindless form; under
 * D3D12AGX_DUMP_TEX, like the compiler's own refusal census. */
static bool
g17p_fragment_decline(const char *why, const nir_tex_instr *tex)
{
   if (!d12_debug()->dump_tex)
      return false;
   if (tex) {
      fprintf(stderr,
              "G17P fragment sample decline (%s): op=%u dim=%u array=%u "
              "shadow=%u coords=%u srcs=%u comps=%u bits=%u flags=0x%x "
              "tex_idx=%u smp_idx=%u\n",
              why, tex->op, tex->sampler_dim, tex->is_array, tex->is_shadow,
              tex->coord_components, tex->num_srcs, tex->def.num_components,
              tex->def.bit_size, tex->backend_flags, tex->texture_index,
              tex->sampler_index);
      for (unsigned i = 0; i < tex->num_srcs; i++)
         fprintf(stderr, "   src[%u] type=%u const=%u bits=%u\n", i,
                 tex->src[i].src_type, nir_src_is_const(tex->src[i].src),
                 nir_src_bit_size(tex->src[i].src));
   } else {
      fprintf(stderr, "G17P fragment sample decline (%s)\n", why);
   }
   return false;
}

static bool
g17p_fragment_sample_shape(const nir_tex_instr *tex)
{
   if ((tex->op != nir_texop_tex && tex->op != nir_texop_txl &&
        tex->op != nir_texop_txb) ||
       tex->sampler_dim != GLSL_SAMPLER_DIM_2D || tex->is_array ||
       tex->is_shadow || tex->is_sparse || tex->coord_components != 2 ||
       tex->def.bit_size != 32 || tex->def.num_components != 4 ||
       tex->texture_index != 0 || tex->backend_flags)
      return false;
   /* agx_nir_lower_texture packs offsets into backend2 and a bias+min_lod
    * pair into lod_bias_min; neither has a witnessed immediate-descriptor
    * fragment form in this driver yet. */
   if (nir_tex_instr_src_index(tex, nir_tex_src_backend2) >= 0 ||
       nir_tex_instr_src_index(tex, nir_tex_src_lod_bias_min_agx) >= 0 ||
       nir_tex_instr_src_index(tex, nir_tex_src_min_lod) >= 0 ||
       nir_tex_instr_src_index(tex, nir_tex_src_offset) >= 0 ||
       nir_tex_instr_src_index(tex, nir_tex_src_sampler_offset) >= 0)
      return false;
   return true;
}

bool
d12_nir_lower_g17p_fragment_sample(nir_shader *nir, uint32_t generation,
                                   struct d12_nir_g17p_fragment_sample *out)
{
   if (!nir || !out || generation != 9 ||
       nir->info.stage != MESA_SHADER_FRAGMENT)
      return false;

   nir_tex_instr *sample = NULL;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic) {
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               /* Any other descriptor consumer (image ops, size queries
                * that crawl the descriptor, heap-indexed handles) is
                * outside the bounded two-pointer ABI. */
               if (nir_intrinsic_has_image_dim(intr) ||
                   intr->intrinsic ==
                      nir_intrinsic_load_from_texture_handle_agx ||
                   intr->intrinsic == nir_intrinsic_bindless_sampler_agx ||
                   intr->intrinsic ==
                      nir_intrinsic_load_custom_border_table_agx)
                  return g17p_fragment_decline("descriptor consumer", NULL);
               continue;
            }
            if (instr->type != nir_instr_type_tex)
               continue;
            if (sample)
               return g17p_fragment_decline("second texture instruction",
                                            sample);
            sample = nir_instr_as_tex(instr);
         }
      }
   }
   if (!sample)
      return false;
   /* The DXIL translator requests the residency lane on every sample so
    * sparse descriptors report exactly; a shader that never reads the
    * status (the ordinary Sample) can take the plain four-component form,
    * which is the one with an immediate-descriptor fragment witness. */
   if (sample->is_sparse && sample->def.num_components == 5 &&
       !(nir_def_components_read(&sample->def) & BITFIELD_BIT(4))) {
      sample->is_sparse = false;
      sample->def.num_components = 4;
   }
   if (!g17p_fragment_sample_shape(sample))
      return g17p_fragment_decline("sample shape", sample);

   const int th = nir_tex_instr_src_index(sample, nir_tex_src_texture_handle);
   const int sh = nir_tex_instr_src_index(sample, nir_tex_src_sampler_handle);
   if (th < 0 || sh < 0)
      return g17p_fragment_decline("missing handle", sample);
   nir_intrinsic_instr *handle = nir_src_as_intrinsic(sample->src[th].src);
   if (!handle || handle->intrinsic != nir_intrinsic_bindless_image_agx ||
       nir_intrinsic_desc_set(handle) != D12_UNIFORM_TEXTURES ||
       !nir_src_is_const(handle->src[0]) ||
       nir_src_bit_size(handle->src[0]) != 32 ||
       (nir_src_as_uint(handle->src[0]) & 31))
      return g17p_fragment_decline("texture handle", sample);
   /* The API sampler arrives as the widened native index s + 1 (state 0 is
    * the texel-fetch reservation); the static sampler the draw publishes is
    * found by the API slot. */
   if (!nir_src_is_const(sample->src[sh].src))
      return g17p_fragment_decline("dynamic sampler", sample);
   const uint64_t native_sampler = nir_src_as_uint(sample->src[sh].src);
   if (native_sampler < 1 || native_sampler > 16)
      return g17p_fragment_decline("sampler index", sample);

   /* Every bindless_image_agx must belong to this one sample; the handle
    * intrinsic itself dies with the source. */
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_bindless_image_agx &&
                intr != handle)
               return g17p_fragment_decline("second texture handle", sample);
         }
      }
   }

   out->texture_offset_B = (uint32_t)nir_src_as_uint(handle->src[0]);
   out->sampler_slot = (uint8_t)(native_sampler - 1);
   nir_steal_tex_src(sample, nir_tex_src_texture_handle);
   nir_steal_tex_src(sample, nir_tex_src_sampler_handle);
   sample->texture_index = 0;
   sample->sampler_index = 0;
   nir_foreach_function_impl(impl, nir)
      nir_progress(true, impl, nir_metadata_control_flow);
   NIR_PASS(_, nir, nir_opt_dce);
   return true;
}
