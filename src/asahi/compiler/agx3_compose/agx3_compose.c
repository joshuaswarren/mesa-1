/* agx3_compose.c -- generic AGX3 encoder over packgen.py tables.
 * Mirrors composer.py encode semantics; see agx3_compose.h. */
#include "agx3_compose.h"
#include <string.h>

typedef struct {
   const agx3_table *t;
   const uint64_t *fields;
   uint16_t words[AGX3_MAX_WORDS], written[AGX3_MAX_WORDS];
   uint8_t nwords;
   uint64_t consumed[3];              /* field ids consumed by cmds/preds */
   uint64_t is_lut[3], is_pred[3];    /* fields seen through a LUT / predicate */
   uint64_t srcmask[AGX3_MAX_FIELDS]; /* direct-deposit source coverage */
} ctx;

static inline void bset(uint64_t *bm, unsigned i) { bm[i >> 6] |= 1ull << (i & 63); }
static inline bool bget(const uint64_t *bm, unsigned i) { return (bm[i >> 6] >> (i & 63)) & 1; }
static inline uint64_t wmask(unsigned width) {
   return width >= 64 ? ~0ull : (1ull << width) - 1;
}

static bool pred_ok(const ctx *c, const agx3_pred *p)
{
   uint64_t v = c->fields[p->field];
   switch (p->kind) {
   case 0: return v == p->a;
   case 1: {
      if (v >= 0x10000) return false;
      for (uint16_t i = 0; i < p->b; i++)
         if (c->t->setpool[p->a + i] == v) return true;
      return false;
   }
   case 2: return p->a <= v && v <= p->b;
   case 3: return ((v >> p->shift) & wmask(p->width)) == p->a;
   default: return false;
   }
}

static bool preds_test(const ctx *c, uint32_t off, uint16_t n)
{
   for (uint16_t i = 0; i < n; i++) {
      const agx3_pred *p = &c->t->preds[off + i];
      if (p->field >= AGX3_MAX_FIELDS) return false;
      if (!pred_ok(c, p)) return false;
   }
   return true;
}

static void preds_mark(ctx *c, uint32_t off, uint16_t n)
{
   for (uint16_t i = 0; i < n; i++) {   /* mark consumed only if all hold */
      bset(c->consumed, c->t->preds[off + i].field);
      bset(c->is_pred, c->t->preds[off + i].field);
   }
}

static bool preds_ok(ctx *c, uint32_t off, uint16_t n)
{
   if (!preds_test(c, off, n)) return false;
   preds_mark(c, off, n);
   return true;
}

static bool deposit(ctx *c, uint8_t word, uint16_t mask, uint16_t value)
{
   if (word >= c->nwords) return false;
   if (c->written[word] & mask & (c->words[word] ^ value)) return false;
   c->words[word] = (uint16_t)((c->words[word] & ~mask) | (value & mask));
   c->written[word] |= mask;
   return true;
}

static bool apply_node(ctx *c, uint32_t node_id);

static bool apply_group(ctx *c, const agx3_group *g)
{
   for (uint16_t i = 0; i < g->alt_n; i++) {
      uint32_t alt_id = c->t->altpool[g->alt_off + i];
      const agx3_node *alt = &c->t->nodes[alt_id];
      /* Most alternatives fail their predicates without modifying state.
       * Only snapshot the encoder once an alternative can actually apply. */
      if (!preds_test(c, alt->pred_off, alt->pred_n))
         continue;
      ctx saved = *c;
      preds_mark(c, alt->pred_off, alt->pred_n);
      if (apply_node(c, alt_id))
         return true;
      *c = saved;
   }
   return false;
}

static bool apply_node(ctx *c, uint32_t node_id)
{
   const agx3_node *n = &c->t->nodes[node_id];
   for (uint16_t i = 0; i < n->fixed_n; i++) {
      const agx3_fixed *f = &c->t->fixeds[n->fixed_off + i];
      if (!deposit(c, f->word, f->mask, f->value)) return false;
   }
   for (uint16_t i = 0; i < n->cmd_n; i++) {
      const agx3_cmd *cmd = &c->t->cmds[n->cmd_off + i];
      if (cmd->field >= AGX3_MAX_FIELDS) return false;
      uint64_t value = c->fields[cmd->field], transformed;
      bset(c->consumed, cmd->field);
      if (cmd->lut_off >= 0) {
         bset(c->is_lut, cmd->field);
         if (value >= cmd->lut_n) return false;
         transformed = c->t->lutpool[cmd->lut_off + value];
      } else {
         transformed = value;
         c->srcmask[cmd->field] |= wmask(cmd->width) << cmd->src_shift;
      }
      uint16_t bits = (uint16_t)((transformed >> cmd->src_shift) & wmask(cmd->width));
      if (!deposit(c, cmd->word, (uint16_t)(wmask(cmd->width) << cmd->dst_shift),
                   (uint16_t)(bits << cmd->dst_shift)))
         return false;
   }
   for (uint16_t i = 0; i < n->grp_n; i++)
      if (!apply_group(c, &c->t->groups[n->grp_off + i]))
         return false;
   return true;
}

bool agx3_compose(const agx3_table *t, uint32_t form,
                  const uint64_t fields[AGX3_MAX_FIELDS], uint8_t *out)
{
   if (form >= t->num_forms) return false;
   const agx3_form *f = &t->forms[form];
   if (f->size % 2 || f->size / 2 > AGX3_MAX_WORDS) return false;

   /* Composition is used by parallel shader compilers.  Keep all mutable
    * traversal state on this invocation; the generated tables are immutable. */
   ctx c = {0};
   ctx *cp = &c;
   cp->t = t;
   cp->fields = fields;
   cp->nwords = f->size / 2;

   const agx3_node *root = &t->nodes[f->node];
   if (!preds_ok(cp, root->pred_off, root->pred_n)) return false;
   if (!apply_node(cp, f->node)) return false;

   /* strict: every nonzero field must have been consumed; direct-deposited
    * non-selector fields must fit their source slices. */
   for (unsigned i = 0; i < AGX3_MAX_FIELDS; i++) {
      if (fields[i] && !bget(cp->consumed, i)) return false;
      if (bget(cp->consumed, i) && !bget(cp->is_lut, i) && !bget(cp->is_pred, i) &&
          (fields[i] & ~cp->srcmask[i]))
         return false;
   }
   for (unsigned w = 0; w < cp->nwords; w++) {
      out[2 * w] = (uint8_t)cp->words[w];
      out[2 * w + 1] = (uint8_t)(cp->words[w] >> 8);
   }
   return true;
}

int32_t agx3_form_index(const agx3_table *t, const char *mnemonic, uint8_t ep)
{
   for (uint32_t i = 0; i < t->num_forms; i++)
      if (t->forms[i].ep == ep && !strcmp(t->forms[i].mnemonic, mnemonic))
         return (int32_t)i;
   return -1;
}

int32_t agx3_compose_mnemonic(const agx3_table *t, const char *mnemonic,
                              const uint64_t fields[AGX3_MAX_FIELDS],
                              uint8_t *out, uint8_t *size)
{
   for (uint32_t i = 0; i < t->num_forms; i++) {
      if (strcmp(t->forms[i].mnemonic, mnemonic)) continue;
      if (agx3_compose(t, i, fields, out)) {
         *size = t->forms[i].size;
         return (int32_t)i;
      }
   }
   return -1;
}
