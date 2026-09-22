/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * The in-tree AGX3 encoder must reproduce real Apple shader instructions. This
 * replays a pinned slice of the G16G/G17S corpus -- each instruction decoded to
 * its field vector by the Python composer -- and requires the C encoder to
 * produce identical bytes for every one.
 */

#include "agx3_compose/agx3_compose.h"
#include "agx3_compose_corpus_vectors.c"

#include <stdio.h>
#include <string.h>

int
main(void)
{
   unsigned pass = 0, fail = 0;

   const agx3_table *tables[] = {&agx3b_table, &agx3c_table};
   for (unsigned table = 0; table < 2; ++table) {
      const agx3_table *t = tables[table];

      for (uint32_t i = 0; i < t->num_forms; ++i) {
         const agx3_form *f = &t->forms[i];
         if (agx3_form_index(t, f->mnemonic, f->ep) != (int32_t)i) {
            fprintf(stderr, "FAIL form lookup %s.ep%u (agx3%c)\n",
                    f->mnemonic, f->ep, table == 0 ? 'b' : 'c');
            fail++;
         }

         if (i > 0) {
            const agx3_form *prev = &t->forms[i - 1];
            const int mnemonic_cmp = strcmp(prev->mnemonic, f->mnemonic);
            if (mnemonic_cmp > 0 ||
                (mnemonic_cmp == 0 && prev->ep >= f->ep)) {
               fprintf(stderr, "FAIL unsorted forms %s.ep%u, %s.ep%u "
                               "(agx3%c)\n",
                       prev->mnemonic, prev->ep, f->mnemonic, f->ep,
                       table == 0 ? 'b' : 'c');
               fail++;
            }
         }
      }

      if (agx3_form_index(t, "", 0) != -1 ||
          agx3_form_index(t, "~unknown", UINT8_MAX) != -1) {
         fprintf(stderr, "FAIL unknown form lookup (agx3%c)\n",
                 table == 0 ? 'b' : 'c');
         fail++;
      }
   }

   for (size_t i = 0; i < agx3_compose_corpus_case_count; ++i) {
      const struct agx3_compose_corpus_case *c = &agx3_compose_corpus_cases[i];
      const agx3_table *t = c->enc == 0 ? &agx3b_table : &agx3c_table;

      uint64_t fields[AGX3_MAX_FIELDS] = {0};
      for (uint16_t k = 0; k < c->nfields; ++k)
         fields[c->field_ids[k]] = c->field_vals[k];

      int32_t form = agx3_form_index(t, c->mnemonic, c->ep);
      uint8_t out[AGX3_MAX_WORDS * 2];

      if (form < 0 || !agx3_compose(t, (uint32_t)form, fields, out) ||
          memcmp(out, c->bytes, c->nbytes) != 0) {
         if (fail < 8)
            fprintf(stderr, "FAIL case %zu: %s.ep%d (agx3%c)\n", i, c->mnemonic,
                    c->ep, c->enc == 0 ? 'b' : 'c');
         fail++;
      } else {
         pass++;
      }
   }

   if (fail) {
      fprintf(stderr, "agx3_compose corpus: %u of %zu failed\n", fail,
              agx3_compose_corpus_case_count);
      return 1;
   }
   printf("agx3_compose corpus: all %u instructions reproduced\n", pass);
   return 0;
}
