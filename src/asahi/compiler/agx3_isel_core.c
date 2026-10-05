/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_isel_core.h"

#include "agx3_binder.h"

#include <string.h>

#define KV(f, v) {(uint16_t)(f), (uint64_t)(v)}

bool
agx3_bittest_encode(enum agx_isa_encoding enc,
                   const struct agx3_bittest_desc *d,
                   uint8_t out[AGX3_ISEL_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len)
      return false;
   memset(out, 0, AGX3_ISEL_MAX_LENGTH);
   *len = 0;
   if (!d || enc != AGX_ISA_ENCODING_AGX3B ||
       d->dest >= 128 || (d->dest & 1) ||
       (d->mask != 1 && d->mask != 2 && d->mask != 4) ||
       (d->negate[0] && d->negate[1]) || d->wait_mask >= 64)
      return false;
   for (unsigned i = 0; i < 3; ++i) {
      if (d->src[i] >= 128 || (d->src[i] & 1) ||
          (d->cache[i] && d->discard[i]))
         return false;
   }

   const bool negate = d->negate[0] || d->negate[1];
   const struct agx3_field_kv base[] = {
      KV(42, 2), KV(43, 2), KV(57, 1), KV(58, 1),
      KV(72, 2), KV(73, 2), KV(89, 2), KV(90, 2), KV(106, 3),
   };
   struct agx3_field_kv ov[24];
   unsigned n = 0;
   if (negate) {
      ov[n++] = (struct agx3_field_kv)KV(0, 1);
      ov[n++] = (struct agx3_field_kv)KV(17, d->dest);
      ov[n++] = (struct agx3_field_kv)KV(24, 2);
      ov[n++] = (struct agx3_field_kv)KV(25, 2);
   } else {
      ov[n++] = (struct agx3_field_kv)KV(3, d->dest);
      ov[n++] = (struct agx3_field_kv)KV(10, 2);
      ov[n++] = (struct agx3_field_kv)KV(11, 2);
   }
   ov[n++] = (struct agx3_field_kv)KV(35, d->src[0]);
   ov[n++] = (struct agx3_field_kv)KV(49, d->mask);
   ov[n++] = (struct agx3_field_kv)KV(65, d->src[1]);
   ov[n++] = (struct agx3_field_kv)KV(82, d->src[2]);
   ov[n++] = (struct agx3_field_kv)KV(44, d->cache[0]);
   ov[n++] = (struct agx3_field_kv)KV(46, d->discard[0]);
   ov[n++] = (struct agx3_field_kv)KV(74, d->cache[1]);
   ov[n++] = (struct agx3_field_kv)KV(76, d->discard[1]);
   ov[n++] = (struct agx3_field_kv)KV(91, d->cache[2]);
   ov[n++] = (struct agx3_field_kv)KV(93, d->discard[2]);
   ov[n++] = (struct agx3_field_kv)KV(78, d->negate[0]);
   ov[n++] = (struct agx3_field_kv)KV(95, d->negate[1]);
   for (unsigned slot = 0; slot < 6; ++slot) {
      if (d->wait_mask & (1u << slot))
         ov[n++] = (struct agx3_field_kv)KV(98 + slot, 1);
   }
   return agx3_bind_compose(enc, "isel", negate ? 22 : 27, base,
                           sizeof(base) / sizeof(base[0]), ov, n, out, len);
}

static bool
cond_ok(enum agx3_isel_icond cond)
{
   return cond == AGX3_ISEL_ICOND_ULT || cond == AGX3_ISEL_ICOND_UGT ||
          cond == AGX3_ISEL_ICOND_EQ || cond == AGX3_ISEL_ICOND_SLT ||
          cond == AGX3_ISEL_ICOND_SGT;
}

/* A 32-bit value spans two halves, so its index is 2-aligned and the pair
 * must fit the 256-half register file. A 16-bit one is a single half. */
static bool
reg32_ok(uint16_t half)
{
   return (half & 1) == 0 && half + 2 <= 256;
}

static bool
reg16_ok(uint16_t half)
{
   return half < 256;
}

/*
 * One entry per witnessed operand shape. `base` is the field vector of an
 * Apple-compiled instruction with the operands removed; `true_field` and
 * `false_field` name where this shape puts its two selected operands, which
 * differs by operand kind: an immediate goes to f64/f81, a kind-2 register to
 * f65/f82, and the 32-bit both-registers form's kind-3 registers to f66/f83.
 */
struct agx3_isel_form {
   uint8_t ep;
   uint8_t true_field;
   uint8_t false_field;
   uint8_t nbase;
   struct agx3_field_kv base[16];
};

/* 32-bit result: ep27. */
static const struct agx3_isel_form form_32_reg_reg = {
   27, 66, 83, 12,
   {KV(10, 2), KV(11, 2), KV(42, 2), KV(43, 2), KV(44, 1), KV(57, 2),
    KV(58, 2), KV(59, 1), KV(72, 3), KV(73, 2), KV(89, 3), KV(90, 2)}};
static const struct agx3_isel_form form_32_imm_imm = {
   27, 64, 81, 12,
   {KV(10, 2), KV(11, 2), KV(42, 2), KV(43, 2), KV(46, 1), KV(57, 2),
    KV(58, 2), KV(61, 1), KV(72, 1), KV(73, 2), KV(89, 1), KV(90, 2)}};
/* These legacy vectors predate explicit lifetime arguments. Both happened to
 * keep their results live and therefore carry destination-cache f12. New
 * production callers use encode_explicit(), which derives f12 from post-RA
 * state for every operand shape and result width. */
static const struct agx3_isel_form form_32_reg_imm = {
   27, 65, 81, 14,
   {KV(10, 2), KV(11, 2), KV(12, 1), KV(42, 2), KV(43, 2), KV(46, 1),
    KV(57, 2), KV(58, 2), KV(61, 1), KV(72, 2), KV(73, 2), KV(76, 1),
    KV(89, 1), KV(90, 2)}};
static const struct agx3_isel_form form_32_imm_reg = {
   27, 64, 82, 14,
   {KV(10, 2), KV(11, 2), KV(12, 1), KV(42, 2), KV(43, 2), KV(46, 1),
    KV(57, 2), KV(58, 2), KV(61, 1), KV(72, 1), KV(73, 2), KV(89, 2),
    KV(90, 2), KV(93, 1)}};

/* 16-bit result: ep25. */
static const struct agx3_isel_form form_16_imm_imm = {
   25, 64, 81, 12,
   {KV(10, 2), KV(11, 1), KV(42, 2), KV(43, 2), KV(46, 1), KV(57, 2),
    KV(58, 2), KV(61, 1), KV(72, 1), KV(73, 1), KV(89, 1), KV(90, 1)}};
static const struct agx3_isel_form form_16_reg_imm = {
   25, 65, 81, 13,
   {KV(10, 2), KV(11, 1), KV(42, 2), KV(43, 2), KV(46, 1), KV(57, 2),
    KV(58, 2), KV(61, 1), KV(72, 2), KV(73, 1), KV(76, 1), KV(89, 1),
    KV(90, 1)}};
static const struct agx3_isel_form form_16_imm_reg = {
   25, 64, 82, 13,
   {KV(10, 2), KV(11, 1), KV(42, 2), KV(43, 2), KV(46, 1), KV(57, 2),
    KV(58, 2), KV(61, 1), KV(72, 1), KV(73, 1), KV(89, 2), KV(90, 1),
    KV(93, 1)}};
static const struct agx3_isel_form form_16_reg_reg = {
   25, 65, 82, 14,
   {KV(10, 2), KV(11, 1), KV(42, 2), KV(43, 2), KV(46, 1), KV(57, 2),
    KV(58, 2), KV(61, 1), KV(72, 2), KV(73, 1), KV(76, 1), KV(89, 2),
    KV(90, 1), KV(93, 1)}};

/*
 * A 16-bit result in an odd half is a different encoding point, not a field:
 * Apple emits ep25 for an even half and ep26 for an odd one, with an
 * otherwise identical field layout. Witnessed by ushort2/ushort4 kernels,
 * whose components must land in adjacent halves, so the compiler had to
 * produce both. (Scalar code never does -- Apple's allocator always picks an
 * even half, which is why the first pass here saw only ep25.)
 */
static const struct agx3_isel_form *
pick_form(bool dest_16, enum agx3_isel_operand t, enum agx3_isel_operand f)
{
   const bool ti = t == AGX3_ISEL_IMMEDIATE, fi = f == AGX3_ISEL_IMMEDIATE;

   if (dest_16) {
      if (ti && fi)
         return &form_16_imm_imm;
      if (!ti && fi)
         return &form_16_reg_imm;
      if (ti && !fi)
         return &form_16_imm_reg;
      return &form_16_reg_reg;
   }

   if (ti && fi)
      return &form_32_imm_imm;
   if (!ti && fi)
      return &form_32_reg_imm;
   if (ti && !fi)
      return &form_32_imm_reg;
   return &form_32_reg_reg;
}

static bool
operand_ok(enum agx3_isel_operand kind, uint16_t value, bool narrow)
{
   if (kind == AGX3_ISEL_IMMEDIATE)
      return value <= 0xff;
   if (kind == AGX3_ISEL_REGISTER)
      return narrow ? reg16_ok(value) : reg32_ok(value);
   return false;
}

/* Fresh G17P eq0/eq0_live/lt2/gt255 native witnesses. This form uses kind-2
 * selected registers even with two register selections; it cannot reuse the
 * older register-comparand form's kind-3 fields. */
/* Uniform operands: aligned 32-bit words in the low uniform file. */
static bool
uniform32_ok(uint16_t half)
{
   return (half & 1) == 0 && half + 2 <= 256;
}

static bool
encode_explicit(const struct agx3_isel_desc *d,
                 uint8_t out[AGX3_ISEL_MAX_LENGTH], uint8_t *len)
{
   const bool tu = d->true_kind == AGX3_ISEL_UNIFORM;
   const bool fu = d->false_kind == AGX3_ISEL_UNIFORM;
   const bool any_uniform = d->cmp0_uniform || d->cmp1_uniform || tu || fu;

   /* Native G17P (metal-nt -arch applegpu_g17p) uniform isel witnesses:
    *   (u < r) ? r : r      02091701020285c00004   cmp0 f36, f42 = 3
    *   (r < u) ? r : r      02010f09020205c20004   cmp1 f51, f57 = 3
    *   (r == 7) ? u : r     02010f86260a17c20002   true f66, f72 = 3
    *   (r == 7) ? r : u     02010f86060217c2200c   false f83, f89 = 3
    *   (r == 7) ? u : u     02010f86260a17c2200c
    *   (r < r) ? u : 3      02010f03220a05c02083
    * Keep to those shapes: one uniform comparand beside a 32-bit register,
    * or uniform selects beside register/immediate comparands, 32-bit result. */
   if (any_uniform) {
      if (d->dest_16 || d->cmp_16 ||
          (d->cmp0_uniform && d->cmp1_uniform) ||
          ((d->cmp0_uniform || d->cmp1_uniform) &&
           (d->cmp1_immediate || tu || fu)) ||
          (d->cmp0_uniform && !uniform32_ok(d->cmp0)) ||
          (d->cmp1_uniform && !uniform32_ok(d->cmp1)) ||
          (tu && !uniform32_ok(d->sel_true)) ||
          (fu && !uniform32_ok(d->sel_false)) ||
          (d->cmp0_uniform && (d->cache[0] || d->discard[0])) ||
          (d->cmp1_uniform && (d->cache[1] || d->discard[1])) ||
          (tu && (d->cache[2] || d->discard[2])) ||
          (fu && (d->cache[3] || d->discard[3])) ||
          (!tu && !operand_ok(d->true_kind, d->sel_true, false)) ||
          (!fu && !operand_ok(d->false_kind, d->sel_false, false)))
         return false;
   }

   if ((!d->cmp0_uniform &&
        !(d->cmp_16 ? reg16_ok(d->cmp0) : reg32_ok(d->cmp0))) ||
       (!d->cmp1_uniform &&
        (d->cmp1_immediate ? d->cmp1 > 255 :
         !(d->cmp_16 ? reg16_ok(d->cmp1) : reg32_ok(d->cmp1)))))
      return false;

   for (unsigned i = 0; i < 4; ++i) {
      if (d->cache[i] && d->discard[i])
         return false;
   }

   const bool ti = d->true_kind == AGX3_ISEL_IMMEDIATE;
   const bool fi = d->false_kind == AGX3_ISEL_IMMEDIATE;
   if ((d->cmp1_immediate && (d->cache[1] || d->discard[1])) ||
       (ti && (d->cache[2] || d->discard[2])) ||
       (fi && (d->cache[3] || d->discard[3])))
      return false;

   const unsigned width = d->dest_16 ? 1 : 2;
   const struct agx3_field_kv base[] = {
      KV(10, 2), KV(11, width), KV(12, d->dest_cache),
      KV(42, d->cmp0_uniform ? 3 : 2), KV(43, d->cmp_16 ? 1 : 2),
      KV(57, d->cmp1_immediate ? 1 : d->cmp1_uniform ? 3 : 2),
      KV(58, (d->cmp1_immediate || d->cmp_16) ? 1 : 2),
      KV(72, ti ? 1 : tu ? 3 : 2), KV(73, width),
      KV(89, fi ? 1 : fu ? 3 : 2), KV(90, width),
   };
   struct agx3_field_kv ov[24];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(3, d->dest);
   ov[n++] = (struct agx3_field_kv)KV(d->cmp0_uniform ? 36 : 35, d->cmp0);
   ov[n++] = (struct agx3_field_kv)KV(d->cmp1_immediate ? 49 :
                                      d->cmp1_uniform ? 51 : 50, d->cmp1);
   ov[n++] = (struct agx3_field_kv)KV(106, d->cond);
   ov[n++] = (struct agx3_field_kv)KV(ti ? 64 : tu ? 66 : 65, d->sel_true);
   ov[n++] = (struct agx3_field_kv)KV(fi ? 81 : fu ? 83 : 82, d->sel_false);
   if (!d->cmp0_uniform) {
      ov[n++] = (struct agx3_field_kv)KV(44, d->cache[0]);
      ov[n++] = (struct agx3_field_kv)KV(46, d->discard[0]);
   }
   if (d->cmp_16 && d->cond >= AGX3_ISEL_ICOND_EQ)
      ov[n++] = (struct agx3_field_kv)KV(47, 1);
   if (!d->cmp1_immediate && !d->cmp1_uniform) {
      ov[n++] = (struct agx3_field_kv)KV(59, d->cache[1]);
      ov[n++] = (struct agx3_field_kv)KV(61, d->discard[1]);
      if (d->cmp_16 && d->cond >= AGX3_ISEL_ICOND_EQ)
         ov[n++] = (struct agx3_field_kv)KV(62, 1);
   }
   if (!ti && !tu) {
      ov[n++] = (struct agx3_field_kv)KV(74, d->cache[2]);
      ov[n++] = (struct agx3_field_kv)KV(76, d->discard[2]);
   }
   if (!fi && !fu) {
      ov[n++] = (struct agx3_field_kv)KV(91, d->cache[3]);
      ov[n++] = (struct agx3_field_kv)KV(93, d->discard[3]);
   }
   for (unsigned i = 0; i < 6; ++i)
      ov[n++] = (struct agx3_field_kv)KV(98 + i, (d->wait_mask >> i) & 1);

   const unsigned ep = d->dest_16 ? 25 + (d->dest & 1) : 27;
   return agx3_bind_compose(AGX_ISA_ENCODING_AGX3B, "isel", ep,
                           base, sizeof(base) / sizeof(base[0]), ov, n, out, len);
}

bool
agx3_isel_encode(enum agx_isa_encoding encoding,
                 const struct agx3_isel_desc *d,
                 uint8_t out[AGX3_ISEL_MAX_LENGTH], uint8_t *len)
{
   if (d == NULL || out == NULL || len == NULL)
      return false;

   /* The probes are AGX3b. AGX3c spells this `iseli32`/`iseli16` with its own
    * forms, which no probe here covers. */
   if (encoding != AGX_ISA_ENCODING_AGX3B)
      return false;

   /* The register comparands share a width; immediates carry no register. */
   if (!cond_ok(d->cond) ||
       (!d->cmp0_uniform &&
        !(d->cmp_16 ? reg16_ok(d->cmp0) : reg32_ok(d->cmp0))) ||
       (!d->cmp1_immediate && !d->cmp1_uniform &&
        !(d->cmp_16 ? reg16_ok(d->cmp1) : reg32_ok(d->cmp1))))
      return false;
   if (d->wait_mask >= (1u << 6))
      return false;

   if (d->dest_16 ? !reg16_ok(d->dest) : !reg32_ok(d->dest))
      return false;
   const bool uniform = d->cmp0_uniform || d->cmp1_uniform ||
                        d->true_kind == AGX3_ISEL_UNIFORM ||
                        d->false_kind == AGX3_ISEL_UNIFORM;
   if (uniform)
      return encode_explicit(d, out, len);

   if (!operand_ok(d->true_kind, d->sel_true, d->dest_16) ||
       !operand_ok(d->false_kind, d->sel_false, d->dest_16))
      return false;

   if (d->cmp1_immediate || d->cmp_16 || d->explicit_hints ||
       d->dest_cache)
      return encode_explicit(d, out, len);

   const struct agx3_isel_form *form =
      pick_form(d->dest_16, d->true_kind, d->false_kind);
   const uint8_t ep =
      (d->dest_16 && (d->dest & 1)) ? (uint8_t)(form->ep + 1) : form->ep;

   struct agx3_field_kv ov[12];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(3, d->dest);
   ov[n++] = (struct agx3_field_kv)KV(35, d->cmp0);
   ov[n++] = (struct agx3_field_kv)KV(50, d->cmp1);
   ov[n++] = (struct agx3_field_kv)KV(106, d->cond);
   ov[n++] = (struct agx3_field_kv)KV(form->true_field, d->sel_true);
   ov[n++] = (struct agx3_field_kv)KV(form->false_field, d->sel_false);

   /* Waits ride on f98..f103 for slots 1..6. */
   for (unsigned i = 0; i < 6; ++i) {
      if (d->wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(98 + i, 1);
   }

   return agx3_bind_compose(encoding, "isel", ep, form->base, form->nbase, ov,
                            n, out, len);
}
