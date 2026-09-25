/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_binder.h"

#include <string.h>

/*
 * The agx3d (G18G) table is 10 MB of generated data. Only the builds that
 * link agx3_compose/agx3d_tables.c define AGX3_HAVE_AGX3D_TABLE; everywhere
 * else an AGX3D request fails closed here instead of failing to link.
 */
static const agx3_table *
agx3d_table_or_null(void)
{
#ifdef AGX3_HAVE_AGX3D_TABLE
   return &agx3d_table;
#else
   return NULL;
#endif
}

bool
agx3_bind_have_agx3d(void)
{
   return agx3d_table_or_null() != NULL;
}

static bool
agx3_bind_fields(const struct agx3_field_kv *baseline, unsigned nbaseline,
                 const struct agx3_field_kv *overrides, unsigned noverrides,
                 uint64_t fields[AGX3_MAX_FIELDS])
{
   memset(fields, 0, sizeof(uint64_t) * AGX3_MAX_FIELDS);
   for (unsigned i = 0; i < nbaseline; ++i) {
      if (baseline[i].field >= AGX3_MAX_FIELDS)
         return false;
      fields[baseline[i].field] = baseline[i].value;
   }
   for (unsigned i = 0; i < noverrides; ++i) {
      if (overrides[i].field >= AGX3_MAX_FIELDS)
         return false;
      fields[overrides[i].field] = overrides[i].value;
   }
   return true;
}

/*
 * Apple's encoder tries a mnemonic's encoding points by ascending size, the
 * ep number breaking ties, and keeps the first that encodes the field vector
 * (the forms[].size order of the generated table; see agx3_compose.h). The
 * forms of one mnemonic are contiguous in a table, which is sorted by
 * (mnemonic, ep).
 */
static int32_t
agx3_compose_apple_order(const agx3_table *t, const char *mnemonic,
                         const uint64_t fields[AGX3_MAX_FIELDS], uint8_t *out)
{
   int32_t lo = -1;
   for (uint8_t ep = 0; ep < 64 && lo < 0; ++ep)
      lo = agx3_form_index(t, mnemonic, ep);
   if (lo < 0)
      return -1;

   uint32_t first = (uint32_t)lo;
   while (first > 0 && strcmp(t->forms[first - 1].mnemonic, mnemonic) == 0)
      first--;
   uint32_t end = (uint32_t)lo;
   while (end < t->num_forms && strcmp(t->forms[end].mnemonic, mnemonic) == 0)
      end++;

   /* At most a few dozen forms per mnemonic: try them smallest first. */
   bool tried[64] = {false};
   if (end - first > 64)
      return -1;
   for (unsigned n = 0; n < end - first; ++n) {
      int32_t best = -1;
      for (uint32_t f = first; f < end; ++f) {
         if (tried[f - first])
            continue;
         if (best < 0 || t->forms[f].size < t->forms[best].size ||
             (t->forms[f].size == t->forms[best].size &&
              t->forms[f].ep < t->forms[best].ep))
            best = (int32_t)f;
      }
      tried[best - first] = true;
      if (agx3_compose(t, (uint32_t)best, fields, out))
         return best;
   }
   return -1;
}

bool
agx3_bind_compose(enum agx_isa_encoding encoding, const char *mnemonic, int ep,
                  const struct agx3_field_kv *baseline, unsigned nbaseline,
                  const struct agx3_field_kv *overrides, unsigned noverrides,
                  uint8_t *out, uint8_t *len)
{
   if (out == NULL || len == NULL || mnemonic == NULL)
      return false;

   /*
    * agx3d forms are chosen the way Apple's encoder chooses them, never by an
    * encoding-point number: agx3b/agx3c ep numbers do not name agx3d forms.
    * A caller has to ask for that explicitly, so that no agx3c field builder
    * reaches the agx3d table by forwarding its encoding.
    */
   if (encoding == AGX_ISA_ENCODING_AGX3D) {
      const agx3_table *t = agx3d_table_or_null();
      uint64_t fields[AGX3_MAX_FIELDS];
      if (ep != AGX3_BIND_APPLE_ORDER || t == NULL ||
          !agx3_bind_fields(baseline, nbaseline, overrides, noverrides, fields))
         return false;

      const int32_t form = agx3_compose_apple_order(t, mnemonic, fields, out);
      if (form < 0)
         return false;
      *len = t->forms[form].size;
      return true;
   }

   const agx3_table *t;
   if (encoding == AGX_ISA_ENCODING_AGX3B)
      t = &agx3b_table;
   else if (encoding == AGX_ISA_ENCODING_AGX3C)
      t = &agx3c_table;
   else
      return false;

   const int32_t form = agx3_form_index(t, mnemonic, (uint8_t)ep);
   if (form < 0)
      return false;

   uint64_t fields[AGX3_MAX_FIELDS];
   if (!agx3_bind_fields(baseline, nbaseline, overrides, noverrides, fields))
      return false;

   /* agx3_compose keeps its words invocation-local and does not touch out
    * until every predicate, deposit and field-consumption check succeeds.
    * Encode directly into the caller's buffer instead of copying the same
    * validated instruction through a second 32-byte stack buffer. */
   if (!agx3_compose(t, (uint32_t)form, fields, out))
      return false;

   const uint8_t size = t->forms[form].size;
   *len = size;
   return true;
}
