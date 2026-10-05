/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#ifndef D12_NIR_MIN_LOD_H
#define D12_NIR_MIN_LOD_H
#include "nir_builder.h"

/* AGX represents a clamped implicit sample through explicit coordinate
 * gradients, as the DXBC path already does. Preserve bias, gradient and
 * comparison operands. The instruction is inserted and minimum dominates it. */
static inline void
d12_nir_sample_min_lod(nir_builder *b, nir_tex_instr *tex, nir_def *minimum)
{
   if (nir_src_is_const(nir_src_for_ssa(minimum)) &&
       nir_src_as_float(nir_src_for_ssa(minimum)) == 0.0f)
      return;
   assert(tex->op == nir_texop_tex || tex->op == nir_texop_txb ||
          tex->op == nir_texop_txd);
   if (tex->op == nir_texop_tex) {
      int coord = nir_tex_instr_src_index(tex, nir_tex_src_coord);
      assert(coord >= 0);
      nir_def *value = tex->src[coord].src.ssa;
      b->cursor = nir_before_instr(&tex->instr);
      nir_def *dx = nir_ddx(b, value);
      nir_def *dy = nir_ddy(b, value);
      tex->op = nir_texop_txd;
      nir_tex_instr_add_src(tex, nir_tex_src_ddx, dx);
      nir_tex_instr_add_src(tex, nir_tex_src_ddy, dy);
   }
   nir_tex_instr_add_src(tex, nir_tex_src_min_lod, minimum);
   b->cursor = nir_after_instr(&tex->instr);
}
#endif
