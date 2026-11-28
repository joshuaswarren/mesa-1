/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * agx3_decode.c -- agx3_table.py's Decoder, in C. See agx3_decode.h.
 *
 * Every bound below fails closed: a path that would overflow one is treated
 * as not decoding, never truncated.
 */
#include "agx3_decode.h"

#include <stdlib.h>
#include <string.h>

#define DEC_MAX_NODES   96
#define DEC_MAX_PENDING 96
#define DEC_MAX_LUTS    8
#define DEC_MAX_PREDS   24
#define DEC_POOL_LUTS   128
#define DEC_POOL_PREDS  384
#define DEC_PATH_LIMIT  20000

struct dec_lut {
   int32_t off;
   uint16_t n;
   uint16_t next;
   uint8_t src, width;
   uint64_t bits;
};

struct dec_pred_node {
   const agx3_pred *p;
   uint16_t next;
};

struct dec_field {
   uint64_t known_mask, known_value;
   uint16_t lut_head, pred_head;
   uint8_t nluts, npreds;
};

struct dec {
   const agx3_table *t;
   const uint8_t *data;
   unsigned size;
   uint32_t form;
   uint32_t nodes[DEC_MAX_NODES];
   unsigned nnodes;
   uint32_t pending[DEC_MAX_PENDING];
   unsigned npending;
   unsigned tried;
   bool aborted;
   uint64_t *fields;
   uint8_t touched[AGX3_MAX_FIELDS];
   uint8_t touched_list[AGX3_MAX_FIELDS];
   struct dec_field f[AGX3_MAX_FIELDS];
   struct dec_lut lut_pool[DEC_POOL_LUTS];
   struct dec_pred_node pred_pool[DEC_POOL_PREDS];
};

static inline uint64_t
wmask(unsigned width)
{
   return width >= 64 ? ~0ull : ((1ull << width) - 1);
}

static inline bool
word_at(const struct dec *d, unsigned w, uint16_t *out)
{
   if (2 * w + 1 >= d->size)
      return false;
   *out = d->data[2 * w] | (d->data[2 * w + 1] << 8);
   return true;
}

static bool
fixed_ok(const struct dec *d, uint32_t node)
{
   const agx3_node *n = &d->t->nodes[node];
   for (unsigned i = 0; i < n->fixed_n; ++i) {
      const agx3_fixed *fx = &d->t->fixeds[n->fixed_off + i];
      uint16_t w;
      if (!word_at(d, fx->word, &w) || (w & fx->mask) != fx->value)
         return false;
   }
   return true;
}

static bool
pred_holds(const struct dec *d, const agx3_pred *p, uint64_t v)
{
   switch (p->kind) {
   case 0:
      return v == p->a;
   case 1:
      for (uint64_t i = 0; i < p->b; ++i) {
         if (d->t->setpool[p->a + i] == v)
            return true;
      }
      return false;
   case 2:
      return p->a <= v && v <= p->b;
   case 3:
      return ((v >> p->shift) & wmask(p->width)) == p->a;
   default:
      return false;
   }
}

/* Field values that satisfy the deposits and predicates of d->nodes. */
static bool
solve(struct dec *d, uint64_t nonzero[3])
{
   const agx3_table *t = d->t;
   unsigned ntouched = 0;
   uint16_t nluts_pool = 0, npreds_pool = 0;
   bool ok = true;

   for (unsigned i = 0; i < d->nnodes && ok; ++i) {
      const agx3_node *n = &t->nodes[d->nodes[i]];
      for (unsigned k = 0; k < n->pred_n; ++k) {
         const agx3_pred *p = &t->preds[n->pred_off + k];
         if (p->field >= AGX3_MAX_FIELDS) {
            ok = false;
            break;
         }
         struct dec_field *f = &d->f[p->field];
         if (!d->touched[p->field]) {
            d->touched[p->field] = 1;
            d->touched_list[ntouched++] = (uint8_t)p->field;
            f->known_mask = 0;
            f->known_value = 0;
            f->lut_head = UINT16_MAX;
            f->pred_head = UINT16_MAX;
            f->nluts = 0;
            f->npreds = 0;
         }
         if (f->npreds == DEC_MAX_PREDS || npreds_pool == DEC_POOL_PREDS) {
            ok = false;
            break;
         }
         uint16_t idx = npreds_pool++;
         d->pred_pool[idx].p = p;
         d->pred_pool[idx].next = f->pred_head;
         f->pred_head = idx;
         f->npreds++;
      }
      for (unsigned k = 0; k < n->cmd_n && ok; ++k) {
         const agx3_cmd *c = &t->cmds[n->cmd_off + k];
         uint16_t w;
         if (c->field >= AGX3_MAX_FIELDS || !word_at(d, c->word, &w)) {
            ok = false;
            break;
         }
         struct dec_field *f = &d->f[c->field];
         if (!d->touched[c->field]) {
            d->touched[c->field] = 1;
            d->touched_list[ntouched++] = (uint8_t)c->field;
            f->known_mask = 0;
            f->known_value = 0;
            f->lut_head = UINT16_MAX;
            f->pred_head = UINT16_MAX;
            f->nluts = 0;
            f->npreds = 0;
         }
         const uint64_t bits = (w >> c->dst_shift) & wmask(c->width);
         if (c->lut_off >= 0) {
            if (f->nluts == DEC_MAX_LUTS || nluts_pool == DEC_POOL_LUTS) {
               ok = false;
               break;
            }
            uint16_t idx = nluts_pool++;
            d->lut_pool[idx] = (struct dec_lut){
               .off = c->lut_off, .n = c->lut_n, .next = f->lut_head,
               .src = c->src_shift, .width = c->width, .bits = bits,
            };
            f->lut_head = idx;
            f->nluts++;
         } else {
            const uint64_t m = wmask(c->width) << c->src_shift;
            if (f->known_mask & m & (f->known_value ^ (bits << c->src_shift))) {
               ok = false;
               break;
            }
            f->known_mask |= m;
            f->known_value = (f->known_value & ~m) | (bits << c->src_shift);
         }
      }
   }

   if (!ok) {
      for (unsigned ti = 0; ti < ntouched; ++ti)
         d->touched[d->touched_list[ti]] = 0;
      return false;
   }

   memset(d->fields, 0, sizeof(uint64_t) * AGX3_MAX_FIELDS);
   nonzero[0] = nonzero[1] = nonzero[2] = 0;

   for (unsigned ti = 0; ti < ntouched; ++ti) {
      const unsigned fi = d->touched_list[ti];
      struct dec_field *f = &d->f[fi];
      uint64_t km = f->known_mask, kv = f->known_value;

      /* Bitfield predicates fix slices of the value. */
      for (uint16_t pi = f->pred_head; pi != UINT16_MAX; pi = d->pred_pool[pi].next) {
         const agx3_pred *p = d->pred_pool[pi].p;
         if (p->kind != 3)
            continue;
         const uint64_t m = wmask(p->width) << p->shift;
         if (km & m & (kv ^ (p->a << p->shift))) {
            ok = false;
            break;
         }
         km |= m;
         kv = (kv & ~m) | (p->a << p->shift);
      }
      if (!ok)
         break;

      /* Candidate values: an equality predicate's, else every LUT index up
       * to the shortest LUT, else the known bits. Note pred_pool is prepended,
       * so iterate to the last eq or any eq (all preds must hold so any eq
       * uniquely constrains v). */
      uint64_t first = kv, count = 1;
      bool have_eq = false;
      for (uint16_t pi = f->pred_head; pi != UINT16_MAX; pi = d->pred_pool[pi].next) {
         if (d->pred_pool[pi].p->kind == 0) {
            first = d->pred_pool[pi].p->a;
            have_eq = true;
            break;
         }
      }
      if (!have_eq && f->nluts) {
         first = 0;
         count = UINT64_MAX;
         for (uint16_t li = f->lut_head; li != UINT16_MAX; li = d->lut_pool[li].next) {
            if (d->lut_pool[li].n < count)
               count = d->lut_pool[li].n;
         }
      }

      bool found = false;
      for (uint64_t c = 0; c < count && !found; ++c) {
         const uint64_t v = first + c;
         if ((v & km) != kv)
            continue;
         bool good = true;
         for (uint16_t li = f->lut_head; li != UINT16_MAX && good; li = d->lut_pool[li].next) {
            const struct dec_lut *l = &d->lut_pool[li];
            good = v < l->n &&
                   ((t->lutpool[l->off + v] >> l->src) & wmask(l->width)) ==
                      l->bits;
         }
         for (uint16_t pi = f->pred_head; pi != UINT16_MAX && good; pi = d->pred_pool[pi].next)
            good = pred_holds(d, d->pred_pool[pi].p, v);
         if (good) {
            d->fields[fi] = v;
            if (v)
               nonzero[fi >> 6] |= 1ull << (fi & 63);
            found = true;
         }
      }
      if (!found) {
         ok = false;
         break;
      }
   }

   for (unsigned ti = 0; ti < ntouched; ++ti)
      d->touched[d->touched_list[ti]] = 0;
   return ok;
}

/* Depth-first over the pending choice groups, in agx3_table.py's order: a
 * group's chosen alternative expands its own groups before the groups after
 * it, and later groups vary fastest. Pending is a stack, so groups are pushed
 * in reverse. When two paths reproduce the same bytes this picks the one the
 * Python decoder picks. Returns true to stop (found or aborted). */
static bool
expand(struct dec *d)
{
   if (d->npending == 0) {
      if (++d->tried > DEC_PATH_LIMIT) {
         d->aborted = true;
         return true;
      }
      uint64_t nonzero[3];
      uint8_t out[2 * AGX3_MAX_WORDS];
      return solve(d, nonzero) &&
             agx3_compose_known(d->t, d->form, d->fields, nonzero, out) &&
             !memcmp(out, d->data, d->size);
   }

   const uint32_t g = d->pending[--d->npending];
   const agx3_group *grp = &d->t->groups[g];
   for (unsigned i = 0; i < grp->alt_n; ++i) {
      const uint32_t alt = d->t->altpool[grp->alt_off + i];
      if (!fixed_ok(d, alt))
         continue;
      const agx3_node *an = &d->t->nodes[alt];
      if (d->nnodes == DEC_MAX_NODES ||
          d->npending + an->grp_n > DEC_MAX_PENDING)
         continue;
      d->nodes[d->nnodes++] = alt;
      for (unsigned k = an->grp_n; k-- > 0;)
         d->pending[d->npending++] = an->grp_off + k;
      if (expand(d))
         return true;
      d->npending -= an->grp_n;
      d->nnodes--;
   }
   d->pending[d->npending++] = g;
   return false;
}

int32_t
agx3_decode(const agx3_table *t, const uint8_t *data, unsigned size,
            const char *mnemonic, uint64_t fields[AGX3_MAX_FIELDS])
{
   if (!t || !data || !fields || size < 2 || size > 2 * AGX3_MAX_WORDS)
      return -1;

   uint32_t form_start = 0, form_end = t->num_forms;
   if (mnemonic) {
      /* t->forms is sorted by (mnemonic, ep); binary search the mnemonic span. */
      uint32_t lo = 0, hi = t->num_forms;
      while (lo < hi) {
         uint32_t mid = lo + ((hi - lo) >> 1);
         if (strcmp(t->forms[mid].mnemonic, mnemonic) < 0)
            lo = mid + 1;
         else
            hi = mid;
      }
      form_start = lo;
      form_end = lo;
      while (form_end < t->num_forms &&
             strcmp(t->forms[form_end].mnemonic, mnemonic) == 0)
         form_end++;
      if (form_start == form_end)
         return -1;
   }

   struct dec d;
   d.t = t;
   d.data = data;
   d.size = size;
   d.tried = 0;
   d.aborted = false;
   d.fields = fields;
   memset(d.touched, 0, sizeof(d.touched));

   int32_t result = -1;
   for (uint32_t form = form_start; form < form_end && !d.aborted; ++form) {
      const agx3_form *fm = &t->forms[form];
      if (fm->size != size)
         continue;
      if (!fixed_ok(&d, fm->node))
         continue;
      const agx3_node *root = &t->nodes[fm->node];
      if (root->grp_n > DEC_MAX_PENDING)
         continue;
      d.form = form;
      d.nnodes = 0;
      d.nodes[d.nnodes++] = fm->node;
      d.npending = 0;
      for (unsigned k = root->grp_n; k-- > 0;)
         d.pending[d.npending++] = root->grp_off + k;
      if (expand(&d) && !d.aborted) {
         result = (int32_t)form;
         break;
      }
   }
   return result;
}
