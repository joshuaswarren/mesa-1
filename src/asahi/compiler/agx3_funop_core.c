/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_funop_core.h"

#include "agx3_binder.h"

#include <string.h>

#define KV(f, v) {(uint16_t)(f), (uint64_t)(v)}
#define NELEM(a) (sizeof(a) / sizeof((a)[0]))

/*
 * Per-mnemonic canonical baselines, taken verbatim from the decoded bare
 * witnesses (transcendental-int-re analysis.json `baselines`, corpus_frag for
 * the gradients). f9/f10/f25/f26/f29 are the operand-shape selectors; f11,
 * f27, f29 and f40(b)/f42(c) retain the canonical witness values here. G17P's
 * rounding/log/exp source-discard field is overridden from the actual source
 * lifetime below. On AGX3c the class discriminator f33=3 plus the f34
 * op-select join the baseline (f32sinc uses a distinct word0 without f42).
 */
struct funop_form {
   const char *mnemonic;
   int rint_mode;                     /* f43, or -1 when not f32rint */
   bool wait_ok;                      /* wait fields grade A in the map */
   const struct agx3_field_kv *base_b;
   unsigned nbase_b;
   const struct agx3_field_kv *base_c;
   unsigned nbase_c;
};

/* AGX3b: the four-mode f32rint / f32log2 / f32exp2 share one shape. */
static const struct agx3_field_kv b_rint_like[] = {
   KV(9, 2), KV(10, 2), KV(25, 2), KV(26, 2), KV(29, 1), KV(40, 1),
};
/* f32sinc and f32dfdx carry the f11/f27 source-hint pair instead. */
static const struct agx3_field_kv b_sinc_like[] = {
   KV(9, 2), KV(10, 2), KV(11, 1), KV(25, 2), KV(26, 2), KV(27, 1),
};
/* The canonical f32dfdy witness has f11 but not f27. */
static const struct agx3_field_kv b_dfdy[] = {
   KV(9, 2), KV(10, 2), KV(11, 1), KV(25, 2), KV(26, 2),
};

/* AGX3c: same shapes plus the f33=3 class selector and per-op f34. */
static const struct agx3_field_kv c_rint[] = {
   KV(9, 2),  KV(10, 2), KV(25, 2), KV(26, 2), KV(29, 1),
   KV(33, 3), KV(34, 65), KV(42, 1),
};
static const struct agx3_field_kv c_log2[] = {
   KV(9, 2),  KV(10, 2), KV(25, 2), KV(26, 2), KV(29, 1),
   KV(33, 3), KV(34, 59), KV(42, 1),
};
static const struct agx3_field_kv c_exp2[] = {
   KV(9, 2),  KV(10, 2), KV(25, 2), KV(26, 2), KV(29, 1),
   KV(33, 3), KV(34, 52), KV(42, 1),
};
static const struct agx3_field_kv c_sinc[] = {
   KV(9, 2),  KV(10, 2), KV(11, 1), KV(25, 2), KV(26, 2), KV(27, 1),
   KV(33, 3), KV(34, 67),
};
static const struct agx3_field_kv c_dfdx[] = {
   KV(9, 2),  KV(10, 2), KV(11, 1), KV(25, 2), KV(26, 2), KV(27, 1),
   KV(33, 3), KV(34, 49),
};
static const struct agx3_field_kv c_dfdy[] = {
   KV(9, 2),  KV(10, 2), KV(11, 1), KV(25, 2), KV(26, 2),
   KV(33, 3), KV(34, 50),
};

static const struct funop_form forms[AGX3_FUNOP_COUNT] = {
   [AGX3_FUNOP_FLOOR] = {"f32rint", 3, true, b_rint_like, NELEM(b_rint_like),
                         c_rint, NELEM(c_rint)},
   [AGX3_FUNOP_CEIL] = {"f32rint", 2, true, b_rint_like, NELEM(b_rint_like),
                        c_rint, NELEM(c_rint)},
   [AGX3_FUNOP_TRUNC] = {"f32rint", 1, true, b_rint_like, NELEM(b_rint_like),
                         c_rint, NELEM(c_rint)},
   [AGX3_FUNOP_RINT] = {"f32rint", 0, true, b_rint_like, NELEM(b_rint_like),
                        c_rint, NELEM(c_rint)},
   [AGX3_FUNOP_LOG2] = {"f32log2", -1, true, b_rint_like, NELEM(b_rint_like),
                        c_log2, NELEM(c_log2)},
   [AGX3_FUNOP_EXP2] = {"f32exp2", -1, true, b_rint_like, NELEM(b_rint_like),
                        c_exp2, NELEM(c_exp2)},
   /* f32sinc is grade B and dfdx/dfdy are absent in the wait-field map, so
    * they fail closed on any wait instead of guessing the bit positions. */
   [AGX3_FUNOP_SINC] = {"f32sinc", -1, false, b_sinc_like, NELEM(b_sinc_like),
                        c_sinc, NELEM(c_sinc)},
   [AGX3_FUNOP_DFDX] = {"f32dfdx", -1, false, b_sinc_like, NELEM(b_sinc_like),
                        c_dfdx, NELEM(c_dfdx)},
   [AGX3_FUNOP_DFDY] = {"f32dfdy", -1, false, b_dfdy, NELEM(b_dfdy),
                        c_dfdy, NELEM(c_dfdy)},
};

static bool
is_b(enum agx_isa_encoding e)
{
   return e == AGX_ISA_ENCODING_AGX3B;
}

bool
agx3_funop_encode_source_lifetime(
   enum agx_isa_encoding enc, enum agx3_funop op, uint16_t dest, uint16_t src,
   bool src_abs, bool src_neg, bool saturate, uint16_t wait_mask, bool src_live,
   uint8_t out[AGX3_FUNOP_MAX_LENGTH], uint8_t *len)
{
   if (out == NULL || len == NULL)
      return false;
   memset(out, 0, AGX3_FUNOP_MAX_LENGTH);
   *len = 0;

   if ((enc != AGX_ISA_ENCODING_AGX3B && enc != AGX_ISA_ENCODING_AGX3C) ||
       op >= AGX3_FUNOP_COUNT)
      return false;

   /* Witnessed operand domain: even 32-bit half-register pairs below 128. */
   if (dest >= 128 || src >= 128 || (dest & 1) || (src & 1))
      return false;

   const struct funop_form *F = &forms[op];

   /* Native saturate is witnessed only on AGX3c (f14); Apple emits a
    * separate saturating f32add on AGX3b. */
   if (saturate && is_b(enc))
      return false;

   const unsigned nslots = is_b(enc) ? 6 : 8;
   if (wait_mask & ~(uint16_t)((1u << nslots) - 1))
      return false;
   if (wait_mask && !F->wait_ok)
      return false;

   struct agx3_field_kv ov[16];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(2, dest);
   ov[n++] = (struct agx3_field_kv)KV(18, src);
   /* G17P rounding/log/exp share f29 as the source-discard bit. Apple's
    * trunc(x) is 2f0056000200b0400600 when x dies and
    * 2f005600020490400600 when x is also stored: only the source register
    * changes and f29 clears. Floor, ceil, log2 and exp2 preserve x likewise.
    * Tessellation's int64 conversion rounds a coordinate which the position
    * calculation reads again, so the baseline's last-use hint is not safe. */
   if (is_b(enc) && op <= AGX3_FUNOP_EXP2)
      ov[n++] = (struct agx3_field_kv)KV(29, !src_live);
   if (src_abs)
      ov[n++] = (struct agx3_field_kv)KV(30, 1);
   if (src_neg)
      ov[n++] = (struct agx3_field_kv)KV(31, 1);
   if (saturate)
      ov[n++] = (struct agx3_field_kv)KV(14, 1);
   if (F->rint_mode >= 0)
      ov[n++] = (struct agx3_field_kv)KV(43, F->rint_mode);
   /* Wait bits f35..f40 (b) / f35..f42 (c), one per scoreboard slot; on the
    * canonical AGX3b baseline the slot-6 field doubles as the f40 hint. */
   for (unsigned i = 0; i < nslots; ++i) {
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(35 + i, 1);
   }

   const struct agx3_field_kv *base = is_b(enc) ? F->base_b : F->base_c;
   const unsigned nbase = is_b(enc) ? F->nbase_b : F->nbase_c;

   if (!agx3_bind_compose(enc, F->mnemonic, 0, base, nbase, ov, n, out, len)) {
      memset(out, 0, AGX3_FUNOP_MAX_LENGTH);
      *len = 0;
      return false;
   }
   return true;
}

bool
agx3_funop_encode(enum agx_isa_encoding enc, enum agx3_funop op, uint16_t dest,
                  uint16_t src, bool src_abs, bool src_neg, bool saturate,
                  uint16_t wait_mask, uint8_t out[AGX3_FUNOP_MAX_LENGTH],
                  uint8_t *len)
{
   return agx3_funop_encode_source_lifetime(enc, op, dest, src, src_abs, src_neg,
                                           saturate, wait_mask, false, out, len);
}
