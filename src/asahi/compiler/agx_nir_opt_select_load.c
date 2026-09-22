/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "compiler/nir/nir_builder.h"
#include "agx_nir.h"
#include "nir.h"
#include "nir_intrinsics.h"
#include "nir_intrinsics_indices.h"

/*
 * Select the index, not the loaded data.
 *
 * A shader that picks one of several equally addressable records -- poly's
 * geometry rasterization shader chooses which emitted vertex it shades, and
 * every one of its varyings sits at the same offset within each candidate --
 * lowers to a select over a pair of loads that differ only in their index:
 *
 *    a = load_agx(base, index_a)
 *    b = load_agx(base, index_b)
 *    r = bcsel(c, b, a)
 *
 * Both loads are issued, both results are staged, and one of each pair is
 * thrown away. Selecting the index instead issues one load and stages one
 * result:
 *
 *    r = load_agx(base, bcsel(c, index_b, index_a))
 *
 * The saving is register pressure rather than instruction count: a sixteen
 * varying transform feedback capture stages thirty-two vec4 loads at once,
 * which does not fit the AGX3B register file and spills. Selecting the index
 * halves the staged data and the live index count. A chain of selects -- one
 * per candidate vertex -- collapses to a single load, because rewriting the
 * innermost select leaves the one above it with a load on both sides.
 *
 * Only reorderable loads are folded, so no memory dependence is crossed: the
 * surviving load may be issued later than the one it replaces.
 */

static nir_intrinsic_instr *
reorderable_load(nir_scalar s)
{
   nir_instr *parent = nir_def_instr(s.def);

   if (parent->type != nir_instr_type_intrinsic)
      return NULL;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(parent);

   if (intr->intrinsic == nir_intrinsic_load_constant_agx)
      return intr;

   if (intr->intrinsic == nir_intrinsic_load_agx &&
       (nir_intrinsic_access(intr) & ACCESS_CAN_REORDER))
      return intr;

   return NULL;
}

/* The two loads must address the same memory in the same way, so that the only
 * thing left to choose between them is the index.
 */
static bool
same_shape(const nir_intrinsic_instr *a, const nir_intrinsic_instr *b)
{
   return a->intrinsic == b->intrinsic &&
          a->def.num_components == b->def.num_components &&
          a->def.bit_size == b->def.bit_size && a->src[0].ssa == b->src[0].ssa &&
          nir_intrinsic_base(a) == nir_intrinsic_base(b) &&
          nir_intrinsic_format(a) == nir_intrinsic_format(b) &&
          nir_intrinsic_access(a) == nir_intrinsic_access(b) &&
          nir_intrinsic_sign_extend(a) == nir_intrinsic_sign_extend(b);
}

static bool
opt_select_load(nir_builder *b, nir_alu_instr *alu, UNUSED void *data)
{
   if (alu->op != nir_op_bcsel)
      return false;

   const unsigned components = alu->def.num_components;
   nir_scalar cond = nir_get_scalar(alu->src[0].src.ssa, alu->src[0].swizzle[0]);
   nir_scalar t = nir_scalar_resolved(alu->src[1].src.ssa, alu->src[1].swizzle[0]);
   nir_scalar f = nir_scalar_resolved(alu->src[2].src.ssa, alu->src[2].swizzle[0]);
   unsigned swizzle[NIR_MAX_VEC_COMPONENTS] = {t.comp};

   /* One index is selected for the whole vector, so every component must agree
    * on the condition and read the same component of each load.
    */
   for (unsigned c = 1; c < components; ++c) {
      nir_scalar tc =
         nir_scalar_resolved(alu->src[1].src.ssa, alu->src[1].swizzle[c]);
      nir_scalar fc =
         nir_scalar_resolved(alu->src[2].src.ssa, alu->src[2].swizzle[c]);

      if (alu->src[0].swizzle[c] != cond.comp || tc.def != t.def ||
          fc.def != f.def || tc.comp != fc.comp)
         return false;

      swizzle[c] = tc.comp;
   }

   nir_intrinsic_instr *load_t = reorderable_load(t);
   nir_intrinsic_instr *load_f = reorderable_load(f);

   if (!load_t || !load_f || load_t == load_f || !same_shape(load_t, load_f) ||
       load_t->src[1].ssa == load_f->src[1].ssa)
      return false;

   /* Both loads dominate the select, so an index selected here does too. */
   b->cursor = nir_before_instr(&alu->instr);

   nir_def *index = nir_bcsel(b, nir_channel(b, cond.def, cond.comp),
                              load_t->src[1].ssa, load_f->src[1].ssa);

   nir_def *load =
      load_t->intrinsic == nir_intrinsic_load_constant_agx
         ? nir_load_constant_agx(b, load_t->def.num_components,
                                 load_t->def.bit_size, load_t->src[0].ssa, index,
                                 .access = nir_intrinsic_access(load_t),
                                 .base = nir_intrinsic_base(load_t),
                                 .format = nir_intrinsic_format(load_t),
                                 .sign_extend = nir_intrinsic_sign_extend(load_t))
         : nir_load_agx(b, load_t->def.num_components, load_t->def.bit_size,
                        load_t->src[0].ssa, index,
                        .access = nir_intrinsic_access(load_t),
                        .base = nir_intrinsic_base(load_t),
                        .format = nir_intrinsic_format(load_t),
                        .sign_extend = nir_intrinsic_sign_extend(load_t));

   nir_def_replace(&alu->def, nir_swizzle(b, load, swizzle, components));
   return true;
}

bool
agx_nir_opt_select_load(nir_shader *shader)
{
   return nir_shader_alu_pass(shader, opt_select_load,
                              nir_metadata_control_flow, NULL);
}
