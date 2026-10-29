/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "compiler/nir/nir_builder.h"
#include "util/hash_table.h"
#include "util/u_dynarray.h"
#include "agx_nir.h"
#include "nir.h"

/*
 * Share the end computation of robust buffer reads at constant distances.
 *
 * A robust read keeps its value only when its exclusive end lies within the
 * descriptor's byte size:
 *
 *    bounds_agx(data, end, size) = end <= size ? data : 0
 *
 * The reads of one structure in a loop (a light record, a constant buffer
 * row) differ only by a constant field offset, and each computes its own end:
 *
 *    end = uadd_sat(iadd(X, c), n)
 *
 * an add and a saturating add per read, on top of the compare-select. For the
 * reads of one block with the same X and size, take the smallest c and
 *
 *    R = usub_sat(size, iadd(X, c_min))
 *
 * once, after which each read checks the immediate c - c_min + n against R:
 *
 *    X + c + n <= size  <=>  c - c_min + n <= size - (X + c_min)
 *
 * when size >= X + c_min, and both sides are false otherwise (R = 0 while the
 * immediate is at least n > 0). This is exact, except where X + c wraps a
 * 32-bit offset: such a read was kept when its wrapped end fell inside the
 * buffer, and is now zeroed, as its real offset lies past the end.
 */

/* bounds_agx takes its end as an 8-bit immediate. */
#define SHARE_BOUNDS_MAX_IMM 255

struct bound_read {
   nir_alu_instr *alu;
   uint32_t c, n;
};

struct bound_group {
   nir_scalar x, size;
   struct util_dynarray reads;
};

static bool
scalar_const(nir_scalar s, uint32_t *value)
{
   if (!nir_scalar_is_const(s))
      return false;
   *value = nir_scalar_as_uint(s);
   return true;
}

/* Match end = uadd_sat(iadd(X, c), n) or uadd_sat(X, n), constants in either
 * position. */
static bool
match_end(nir_scalar end, nir_scalar *x, uint32_t *c, uint32_t *n)
{
   if (!nir_scalar_is_alu(end) || nir_scalar_alu_op(end) != nir_op_uadd_sat ||
       end.def->bit_size != 32)
      return false;

   nir_scalar a = nir_scalar_chase_alu_src(end, 0);
   nir_scalar b = nir_scalar_chase_alu_src(end, 1);
   nir_scalar inner;
   if (scalar_const(b, n))
      inner = a;
   else if (scalar_const(a, n))
      inner = b;
   else
      return false;

   if (*n == 0 || *n > SHARE_BOUNDS_MAX_IMM)
      return false;

   *c = 0;
   *x = inner;
   if (nir_scalar_is_alu(inner) && nir_scalar_alu_op(inner) == nir_op_iadd) {
      nir_scalar p = nir_scalar_chase_alu_src(inner, 0);
      nir_scalar q = nir_scalar_chase_alu_src(inner, 1);
      if (scalar_const(q, c))
         *x = p;
      else if (scalar_const(p, c))
         *x = q;
   }

   return !nir_scalar_is_const(*x);
}

static uint32_t
group_hash(const void *key)
{
   const struct bound_group *g = key;
   uint32_t h = _mesa_hash_pointer(g->x.def) ^ (g->x.comp * 31);
   return h ^ (_mesa_hash_pointer(g->size.def) * 7) ^ g->size.comp;
}

static bool
group_equal(const void *a, const void *b)
{
   const struct bound_group *x = a, *y = b;
   return nir_scalar_equal(x->x, y->x) && nir_scalar_equal(x->size, y->size);
}

static bool
share_group(nir_builder *b, struct bound_group *g)
{
   unsigned count = util_dynarray_num_elements(&g->reads, struct bound_read);
   if (count < 2)
      return false;

   uint32_t c_min = UINT32_MAX;
   util_dynarray_foreach(&g->reads, struct bound_read, r)
      c_min = MIN2(c_min, r->c);

   /* Reads whose immediate would not fit keep their own end. */
   unsigned fit = 0;
   util_dynarray_foreach(&g->reads, struct bound_read, r) {
      if ((uint64_t)r->c - c_min + r->n <= SHARE_BOUNDS_MAX_IMM)
         fit++;
   }
   if (fit < 2)
      return false;

   struct bound_read *first =
      util_dynarray_element(&g->reads, struct bound_read, 0);
   b->cursor = nir_before_instr(&first->alu->instr);

   nir_def *x = nir_mov_scalar(b, g->x);
   nir_def *base = c_min ? nir_iadd_imm(b, x, c_min) : x;
   nir_def *rest = nir_usub_sat(b, nir_mov_scalar(b, g->size), base);

   util_dynarray_foreach(&g->reads, struct bound_read, r) {
      uint64_t imm = (uint64_t)r->c - c_min + r->n;
      if (imm > SHARE_BOUNDS_MAX_IMM)
         continue;

      b->cursor = nir_before_instr(&r->alu->instr);
      nir_src_rewrite(&r->alu->src[1].src, nir_imm_int(b, imm));
      r->alu->src[1].swizzle[0] = 0;
      nir_src_rewrite(&r->alu->src[2].src, rest);
      r->alu->src[2].swizzle[0] = 0;
   }

   return true;
}

static bool
share_block(nir_builder *b, nir_block *block, void *mem_ctx)
{
   struct set *groups = _mesa_set_create(mem_ctx, group_hash, group_equal);
   struct util_dynarray order;
   util_dynarray_init(&order, mem_ctx);

   nir_foreach_instr(instr, block) {
      if (instr->type != nir_instr_type_alu)
         continue;
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      if (alu->op != nir_op_bounds_agx || alu->def.bit_size != 32 ||
          alu->def.num_components != 1)
         continue;

      nir_scalar end = nir_scalar_resolved(alu->src[1].src.ssa,
                                           alu->src[1].swizzle[0]);
      nir_scalar size = nir_scalar_resolved(alu->src[2].src.ssa,
                                            alu->src[2].swizzle[0]);
      if (size.def->bit_size != 32)
         continue;

      struct bound_read read = {.alu = alu};
      nir_scalar x;
      if (!match_end(end, &x, &read.c, &read.n))
         continue;

      struct bound_group key = {.x = x, .size = size};
      bool found = false;
      struct set_entry *e = _mesa_set_search_or_add(groups, &key, &found);
      struct bound_group *g;
      if (!found) {
         g = ralloc(mem_ctx, struct bound_group);
         *g = key;
         util_dynarray_init(&g->reads, mem_ctx);
         e->key = g;
         util_dynarray_append(&order, g);
      } else {
         g = (struct bound_group *)e->key;
      }
      util_dynarray_append(&g->reads, read);
   }

   bool progress = false;
   util_dynarray_foreach(&order, struct bound_group *, g)
      progress |= share_group(b, *g);

   return progress;
}

bool
agx_nir_share_bounds(nir_shader *s)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   if (!impl)
      return false;

   void *mem_ctx = ralloc_context(NULL);
   nir_builder b = nir_builder_create(impl);
   bool progress = false;

   nir_foreach_block(block, impl)
      progress |= share_block(&b, block, mem_ctx);

   ralloc_free(mem_ctx);
   return nir_progress(progress, impl, nir_metadata_control_flow);
}
