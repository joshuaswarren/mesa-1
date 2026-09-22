/* SPDX-License-Identifier: MIT */
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "agx3_compose.h"

/* Structural decomposer.  Enumerates a form's choice-group alternatives,
 * derives the field vector implied by the target bytes (predicates pin
 * selectors, direct deposits recover operands, LUT deposits are inverted by
 * enumeration), then verifies by re-composing with the production encoder.
 * Only exact round-trips are reported, so a reported field vector is a fact
 * about the encoder, not a guess.
 *
 * usage: decomp <hex bytes> [mnemonic|*] */

static const agx3_table *T;
static uint16_t want[AGX3_MAX_WORDS];
static unsigned nwords;
static int hits;

#define MAXLUT 24
typedef struct {
   uint64_t fields[AGX3_MAX_FIELDS];
   uint8_t known[AGX3_MAX_FIELDS];
   const agx3_cmd *lut[MAXLUT];
   unsigned nlut;
} sol;

#define MAXPEND 64
typedef struct {
   const agx3_group *g[MAXPEND];
   unsigned n;
} pending;

static bool
fixed_ok(const agx3_node *n)
{
   for (uint16_t i = 0; i < n->fixed_n; i++) {
      const agx3_fixed *f = &T->fixeds[n->fixed_off + i];
      if (f->word >= nwords)
         return false;
      if ((want[f->word] & f->mask) != (f->value & f->mask))
         return false;
   }
   return true;
}

static void
apply_preds(sol *s, uint32_t off, uint16_t n)
{
   for (uint16_t i = 0; i < n; i++) {
      const agx3_pred *p = &T->preds[off + i];
      if (p->field >= AGX3_MAX_FIELDS)
         continue;
      if (p->kind == 0) {
         s->fields[p->field] = p->a;
         s->known[p->field] = 1;
      } else if (p->kind == 3)
         s->fields[p->field] |= (p->a << p->shift);
   }
}

static bool
apply_cmds(sol *s, const agx3_node *n)
{
   for (uint16_t i = 0; i < n->cmd_n; i++) {
      const agx3_cmd *c = &T->cmds[n->cmd_off + i];
      if (c->field >= AGX3_MAX_FIELDS || c->word >= nwords)
         return false;
      uint64_t mask = (c->width >= 64) ? ~0ull : ((1ull << c->width) - 1);
      uint64_t bits = (want[c->word] >> c->dst_shift) & mask;
      if (c->lut_off >= 0) {
         if (s->known[c->field]) { /* pinned by a predicate: check */
            uint64_t v = s->fields[c->field];
            if (v >= c->lut_n)
               return false;
            if ((((uint64_t)T->lutpool[c->lut_off + v] >> c->src_shift) &
                 mask) != bits)
               return false;
            continue;
         }
         if (s->nlut >= MAXLUT)
            return false;
         s->lut[s->nlut++] = c;
      } else {
         s->fields[c->field] |= bits << c->src_shift;
      }
   }
   return true;
}

static void
report(sol *s, uint32_t form)
{
   printf("MATCH form %u %s ep%u size%u :", form, T->forms[form].mnemonic,
          T->forms[form].ep, T->forms[form].size);
   for (unsigned i = 0; i < AGX3_MAX_FIELDS; i++)
      if (s->fields[i])
         printf(" %u:%llu", i, (unsigned long long)s->fields[i]);
   printf("\n");
   hits++;
}

static void
resolve_luts(sol *s, unsigned k, uint32_t form)
{
   if (k == s->nlut) {
      uint8_t out[32];
      if (!agx3_compose(T, form, s->fields, out))
         return;
      for (unsigned w = 0; w < nwords; w++)
         if (((uint16_t)out[2 * w] | ((uint16_t)out[2 * w + 1] << 8)) !=
             want[w])
            return;
      report(s, form);
      return;
   }
   const agx3_cmd *c = s->lut[k];
   uint64_t mask = (c->width >= 64) ? ~0ull : ((1ull << c->width) - 1);
   uint64_t bits = (want[c->word] >> c->dst_shift) & mask;
   uint64_t save = s->fields[c->field];
   for (uint16_t v = 0; v < c->lut_n; v++) {
      if ((((uint64_t)T->lutpool[c->lut_off + v] >> c->src_shift) & mask) !=
          bits)
         continue;
      s->fields[c->field] = v;
      resolve_luts(s, k + 1, form);
      s->fields[c->field] = save;
   }
}

static void
push_groups(pending *p, const agx3_node *n)
{
   for (uint16_t i = 0; i < n->grp_n && p->n < MAXPEND; i++)
      p->g[p->n++] = &T->groups[n->grp_off + i];
}

static void
solve(pending p, sol s, uint32_t form)
{
   if (p.n == 0) {
      resolve_luts(&s, 0, form);
      return;
   }
   const agx3_group *g = p.g[--p.n];
   for (uint16_t a = 0; a < g->alt_n; a++) {
      uint32_t alt = T->altpool[g->alt_off + a];
      const agx3_node *an = &T->nodes[alt];
      if (!fixed_ok(an))
         continue;
      sol s2 = s;
      pending p2 = p;
      apply_preds(&s2, an->pred_off, an->pred_n);
      if (!apply_cmds(&s2, an))
         continue;
      push_groups(&p2, an);
      solve(p2, s2, form);
   }
}

int
main(int argc, char **argv)
{
   if (argc < 2 || argc > 3) {
      fprintf(stderr, "usage: %s HEX_BYTES [MNEMONIC]\n", argv[0]);
      return 2;
   }
   uint8_t bytes[32];
   const char *h = argv[1];
   const size_t chars = strlen(h);
   if (!chars || chars > sizeof(bytes) * 2 || chars % 4)
      return 2;
   for (size_t i = 0; i < chars; ++i)
      if (!isxdigit((unsigned char)h[i]))
         return 2;
   unsigned nb = chars / 2;
   for (unsigned i = 0; i < nb; ++i) {
      unsigned v;
      if (sscanf(h + 2 * i, "%2x", &v) != 1)
         return 2;
      bytes[i] = v;
   }
   const char *mn = argc > 2 ? argv[2] : "*";
   const agx3_table *tabs[2] = {&agx3b_table, &agx3c_table};
   const char *names[2] = {"agx3b", "agx3c"};
   for (int ti = 0; ti < 2; ti++) {
      T = tabs[ti];
      nwords = nb / 2;
      for (unsigned w = 0; w < nwords; w++)
         want[w] = (uint16_t)bytes[2 * w] | ((uint16_t)bytes[2 * w + 1] << 8);
      for (uint32_t f = 0; f < T->num_forms; f++) {
         if (T->forms[f].size != nb)
            continue;
         if (strcmp(mn, "*") && strcmp(T->forms[f].mnemonic, mn))
            continue;
         const agx3_node *root = &T->nodes[T->forms[f].node];
         if (!fixed_ok(root))
            continue;
         sol s;
         memset(&s, 0, sizeof(s));
         apply_preds(&s, root->pred_off, root->pred_n);
         if (!apply_cmds(&s, root))
            continue;
         pending p;
         p.n = 0;
         push_groups(&p, root);
         int before = hits;
         solve(p, s, f);
         if (hits > before)
            printf("   ^ %s\n", names[ti]);
      }
   }
   if (!hits)
      printf("no structural match\n");
   return 0;
}
