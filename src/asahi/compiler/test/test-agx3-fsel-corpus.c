/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Drive every real Apple `fsel` instruction in the corpus through the general
 * compare-select encoder.
 *
 * The point is not the pass count. A vector whose operands the encoder does
 * not understand must FAIL CLOSED and is asserted to fail, so the split
 * between "reproduced byte-exactly" and "refused" is visible and cannot be
 * papered over by a green total. Select-slot kind 2 is the corpus majority and
 * is unidentified, so those are the refusals.
 */

#include "agx3_alu_core.h"
#include "agx3_compose/agx3_compose.h"
#include "agx3_compose_corpus_vectors.c"

#include <stdio.h>
#include <string.h>

static uint64_t
field(const struct agx3_compose_corpus_case *c, uint16_t id)
{
   for (uint16_t i = 0; i < c->nfields; ++i)
      if (c->field_ids[i] == id)
         return c->field_vals[i];
   return 0;
}

static enum agx3_alu_hint
hint_of(uint64_t cache, uint64_t discard)
{
   if (discard)
      return AGX3_ALU_HINT_DISCARD;
   if (cache)
      return AGX3_ALU_HINT_CACHE;
   return AGX3_ALU_HINT_NONE;
}

int
main(void)
{
   unsigned reproduced = 0, refused = 0, wrong = 0, imm_cmp = 0;

   for (size_t i = 0; i < agx3_compose_corpus_case_count; ++i) {
      const struct agx3_compose_corpus_case *c = &agx3_compose_corpus_cases[i];
      if (c->enc != 0 || strcmp(c->mnemonic, "fsel") != 0)
         continue;

      const uint64_t sel0_kind = field(c, 76), sel1_kind = field(c, 93);
      /* f43 and f60 are the comparators' widths. Every fsel vector in the
       * current corpus declares both at 2 (32-bit), so wiring them changes
       * nothing today; it is here so that the first half-width vector to
       * arrive is encoded rather than silently widened. */
      const uint64_t cmp1_kind = field(c, 59);
      const bool cmp1_imm = cmp1_kind == AGX3_FSEL_CMP_IMMEDIATE;

      /*
       * Loud invariant, not a silent assumption: every immediate compare
       * source in the corpus carries the value zero (32 of 32). The surviving
       * `isinf` refusal is exactly that shape, so if a future corpus vector
       * breaks this, the encoder's immediate path needs revisiting.
       */
      if (cmp1_imm) {
         ++imm_cmp;
         if (field(c, 51) != 0) {
            printf("FAIL: immediate compare source with value %llu, not 0\n",
                   (unsigned long long)field(c, 51));
            ++wrong;
         }
      }

      const struct agx3_fsel_desc d = {
         .dest = (uint16_t)field(c, 3),
         .dest_16 = field(c, 11) == 1,
         .dest_cache = field(c, 12) != 0,
         .cmp0 = (uint16_t)field(c, 35),
         .cmp0_hint = hint_of(field(c, 44), field(c, 46)),
         .cmp0_abs = field(c, 47) != 0,
         .cmp0_16 = field(c, 43) == 1,
         .cmp1_is_immediate = cmp1_imm,
         .cmp1 = (uint16_t)(cmp1_imm ? field(c, 51) : field(c, 52)),
         .cmp1_hint = hint_of(field(c, 61), field(c, 63)),
         .cmp1_abs = field(c, 64) != 0,
         .cmp1_16 = field(c, 60) == 1,
         .sel0_kind = (uint8_t)sel0_kind,
         .sel1_kind = (uint8_t)sel1_kind,
         .sel0 = (uint16_t)(sel0_kind == AGX3_FSEL_SEL_REGISTER ? field(c, 70)
                                                                : field(c, 68)),
         .sel1 = (uint16_t)(sel1_kind == AGX3_FSEL_SEL_REGISTER ? field(c, 87)
                                                                : field(c, 85)),
         .sel_16 = field(c, 77) == 1,
         .condition = (unsigned)field(c, 110),
         .wait_mask = field(c, 107) ? 32u : 0u,
      };

      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
      const bool ok = agx3_alu_fcmp_select(AGX_ISA_ENCODING_AGX3B, &d, bytes,
                                           &len);
      const bool understood = sel0_kind != AGX3_FSEL_SEL_UNKNOWN2 &&
                              sel1_kind != AGX3_FSEL_SEL_UNKNOWN2;

      if (!understood) {
         /* Must fail closed, not guess. */
         if (ok) {
            printf("FAIL: encoded a vector using select kind 2 (%zu)\n", i);
            ++wrong;
         } else {
            ++refused;
         }
         continue;
      }

      if (!ok || len != c->nbytes || memcmp(bytes, c->bytes, len) != 0) {
         printf("FAIL vector %zu: got", i);
         for (unsigned b = 0; ok && b < len; ++b)
            printf(" %02x", bytes[b]);
         if (!ok)
            printf(" <refused>");
         printf(", want");
         for (unsigned b = 0; b < c->nbytes; ++b)
            printf(" %02x", c->bytes[b]);
         printf("\n");
         ++wrong;
      } else {
         ++reproduced;
      }
   }

   printf("agx3 fsel corpus: %u reproduced byte-exactly, %u refused closed "
          "(select kind 2, unidentified), %u immediate compare sources\n",
          reproduced, refused, imm_cmp);
   if (wrong) {
      printf("%u FAILURES\n", wrong);
      return 1;
   }
   return 0;
}
