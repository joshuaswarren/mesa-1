/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_nir_lower_image_view_min_lod.h"
#include "agx_nir_texture.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_builtin_builder.h"

static nir_def *
query_min_lod(nir_builder *b, nir_tex_instr *tex, bool int_coords)
{
   nir_alu_type T = int_coords ? nir_type_uint16 : nir_type_float16;
   return nir_build_texture_query(b, tex, nir_texop_image_min_lod_agx, 1, T,
                                  false, false);
}

static bool
lower_min_lod(nir_builder *b, nir_tex_instr *tex, UNUSED void *_data)
{
   if (nir_tex_instr_is_query(tex))
      return false;

   /* Buffer textures don't have levels-of-detail */
   if (tex->sampler_dim == GLSL_SAMPLER_DIM_BUF)
      return false;

   if (tex->backend_flags & AGX_TEXTURE_FLAG_NO_CLAMP)
      return false;

   bool int_coords = tex->op == nir_texop_txf || tex->op == nir_texop_txf_ms ||
                     tex->op == nir_texop_tg4;

   b->cursor = nir_before_instr(&tex->instr);
   nir_def *min_lod = query_min_lod(b, tex, int_coords);
   nir_def *other_min_lod = nir_steal_tex_src(tex, nir_tex_src_min_lod);

   if (int_coords) {
      assert(other_min_lod == NULL && "no API min lod for integer fetches");

      /* The descriptor stores max(floor(view minLod) - baseMipLevel, 0).
       * Integer Texel Coordinate Operations select baseMipLevel + Lod;
       * they do not clamp or offset Lod by minLod. Below the minimum,
       * robustImageAccess2 requires all components to be zero. Gathering
       * and multisample fetches implicitly use level zero.
       *
       * Keep this integer threshold out of nir_tex_src_min_lod, which is
       * floating point and describes a sampling clamp. Ordinary coordinate
       * and level robustness remains in the common texture lowering.
       */
      int lod_idx = nir_tex_instr_src_index(tex, nir_tex_src_lod);
      nir_def *oob = lod_idx >= 0
                       ? nir_ult(b, nir_u2u32(b, tex->src[lod_idx].src.ssa),
                                 nir_u2u32(b, min_lod))
                       : nir_ine_imm(b, min_lod, 0);
      b->cursor = nir_after_instr(&tex->instr);
      nir_def *old = &tex->def;
      nir_def *zero = nir_imm_zero(b, old->num_components, old->bit_size);
      nir_def *new_ = nir_bcsel(b, oob, zero, old);
      nir_def_rewrite_uses_after(old, new_);
   } else if (tex->op == nir_texop_txl) {
      assert(other_min_lod == NULL && "txl doesn't have an API min lod");

      nir_def *lod = nir_steal_tex_src(tex, nir_tex_src_lod);
      if (lod) {
         min_lod = nir_fmax(b, nir_f2fN(b, lod, min_lod->bit_size), min_lod);
      }

      nir_tex_instr_add_src(tex, nir_tex_src_lod, min_lod);
   } else {
      if (other_min_lod) {
         assert(!int_coords && "no API min lod");
         min_lod = nir_fmax(b, min_lod, nir_f2f16(b, other_min_lod));
      }

      nir_tex_instr_add_src(tex, nir_tex_src_min_lod, min_lod);
   }

   return true;
}

bool
hk_nir_lower_image_view_min_lod(nir_shader *nir)
{
   return nir_shader_tex_pass(nir, lower_min_lod, nir_metadata_none, NULL);
}
