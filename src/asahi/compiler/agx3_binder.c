/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_binder.h"

static const agx3_table *
agx3_bind_table(enum agx_isa_encoding encoding)
{
   if (encoding == AGX_ISA_ENCODING_AGX3B)
      return &agx3b_table;
   else if (encoding == AGX_ISA_ENCODING_AGX3C)
      return &agx3c_table;
   else
      return NULL;
}

unsigned
agx3_bind_form_size(enum agx_isa_encoding encoding, const char *mnemonic,
                    int ep)
{
   const agx3_table *t = agx3_bind_table(encoding);
   const int32_t form = t ? agx3_form_index(t, mnemonic, (uint8_t)ep) : -1;
   return form < 0 ? 0 : t->forms[form].size;
}

bool
agx3_bind_compose(enum agx_isa_encoding encoding, const char *mnemonic, int ep,
                  const struct agx3_field_kv *baseline, unsigned nbaseline,
                  const struct agx3_field_kv *overrides, unsigned noverrides,
                  uint8_t *out, uint8_t *len)
{
   if (out == NULL || len == NULL || mnemonic == NULL)
      return false;

   const agx3_table *t = agx3_bind_table(encoding);
   if (!t)
      return false;

   const int32_t form = agx3_form_index(t, mnemonic, (uint8_t)ep);
   if (form < 0)
      return false;

   uint64_t fields[AGX3_MAX_FIELDS] = {0};
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

   /* Only the listed fields can be nonzero. */
   uint64_t nonzero[3] = {0};
   for (unsigned i = 0; i < nbaseline; ++i) {
      const unsigned field = baseline[i].field;
      if (fields[field])
         nonzero[field / 64] |= UINT64_C(1) << (field % 64);
   }
   for (unsigned i = 0; i < noverrides; ++i) {
      const unsigned field = overrides[i].field;
      if (fields[field])
         nonzero[field / 64] |= UINT64_C(1) << (field % 64);
   }

   /* agx3_compose keeps its words invocation-local and does not touch out
    * until every predicate, deposit and field-consumption check succeeds.
    * Encode directly into the caller's buffer instead of copying the same
    * validated instruction through a second 32-byte stack buffer. */
   if (!agx3_compose_known(t, (uint32_t)form, fields, nonzero, out))
      return false;

   const uint8_t size = t->forms[form].size;
   *len = size;
   return true;
}
