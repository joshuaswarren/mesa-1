/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_tex_core.h"

#include "agx3_binder.h"

#include <string.h>

#define KV(f, v) {(uint16_t)(f), (uint64_t)(v)}

static bool
one_or_zero(uint16_t value)
{
   return (value & (value - 1)) == 0;
}

static bool
fetch_ok(enum agx_isa_encoding enc, const struct agx3_tex_fetch_desc *d)
{
   if (d == NULL || (enc != AGX_ISA_ENCODING_AGX3B &&
                     enc != AGX_ISA_ENCODING_AGX3C))
      return false;

   if ((d->width != AGX3_TEX_W16 && d->width != AGX3_TEX_W32) ||
       d->dest >= 128 || (d->width == AGX3_TEX_W32 && (d->dest & 1)) ||
       (d->coord & 3) || d->coord > 60 ||
       (enc == AGX_ISA_ENCODING_AGX3B ? (!d->mask || d->mask > 15)
                                    : (d->mask != 1 && d->mask != 15)) ||
       d->texture > 15)
      return false;

   const unsigned tk = d->texture_kind ?: AGX3_TEX_OPERAND_IMMEDIATE;
   const unsigned sk = d->sampler_kind ?: AGX3_TEX_OPERAND_IMMEDIATE;
   if (tk > AGX3_TEX_OPERAND_UNIFORM || sk > AGX3_TEX_OPERAND_UNIFORM ||
       (tk != AGX3_TEX_OPERAND_IMMEDIATE &&
        ((d->texture_value & 1) || d->texture_value > 252)) ||
       (sk == AGX3_TEX_OPERAND_IMMEDIATE ? d->sampler_value > 15
         : ((d->sampler_value & 1) || d->sampler_value > 252)) ||
       (d->texture_kill && tk != AGX3_TEX_OPERAND_REGISTER))
      return false;

   const unsigned slots = enc == AGX_ISA_ENCODING_AGX3C ? 8 : 6;
   const uint16_t waits = enc == AGX_ISA_ENCODING_AGX3C ? 0x7f : 0x3f;
   return d->slot >= 1 && d->slot <= slots &&
          (d->wait_mask & ~waits) == 0 && one_or_zero(d->wait_mask);
}

bool
agx3_tex_encode_fetch(enum agx_isa_encoding enc,
                      const struct agx3_tex_fetch_desc *d,
                      uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   if (out == NULL || len == NULL)
      return false;
   memset(out, 0, AGX3_TEX_MAX_LENGTH);
   *len = 0;
   if (!fetch_ok(enc, d))
      return false;

   struct agx3_field_kv base[20];
   unsigned nb = 0;
   base[nb++] = (struct agx3_field_kv)KV(9, 2);
   base[nb++] = (struct agx3_field_kv)KV(23, 5);
   base[nb++] = (struct agx3_field_kv)KV(38, 1);
   base[nb++] = (struct agx3_field_kv)KV(52, 1);
   base[nb++] = (struct agx3_field_kv)KV(89, 3);
   base[nb++] = (struct agx3_field_kv)KV(90, 3);
   const enum agx3_tex_operand_kind tk =
      d->texture_kind ? d->texture_kind : AGX3_TEX_OPERAND_IMMEDIATE;
   const enum agx3_tex_operand_kind sk =
      d->sampler_kind ? d->sampler_kind : AGX3_TEX_OPERAND_IMMEDIATE;
   const bool immediate_operands = tk == AGX3_TEX_OPERAND_IMMEDIATE &&
                                   sk == AGX3_TEX_OPERAND_IMMEDIATE;

   base[nb++] = (struct agx3_field_kv)KV(103, tk);
   base[nb++] = (struct agx3_field_kv)KV(104, tk == AGX3_TEX_OPERAND_IMMEDIATE ? 1 : 2);
   base[nb++] = (struct agx3_field_kv)KV(112, 4);
   base[nb++] = (struct agx3_field_kv)KV(118, 3);
   base[nb++] = (struct agx3_field_kv)KV(119, 3);
   base[nb++] = (struct agx3_field_kv)KV(132, sk);
   base[nb++] = (struct agx3_field_kv)KV(133, sk == AGX3_TEX_OPERAND_IMMEDIATE ? 1 : 2);
   base[nb++] = (struct agx3_field_kv)KV(138, 1);
   if (enc == AGX_ISA_ENCODING_AGX3C) {
      base[nb++] = (struct agx3_field_kv)KV(24, 2);
      base[nb++] = (struct agx3_field_kv)KV(60, 2);
      base[nb++] = (struct agx3_field_kv)KV(61, 164);
   }

   struct agx3_field_kv ov[24];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(2, d->dest);
   ov[n++] = (struct agx3_field_kv)KV(10, d->width);
   ov[n++] = (struct agx3_field_kv)KV(12, d->slot);
   ov[n++] = (struct agx3_field_kv)KV(19, d->coord);
   ov[n++] = (struct agx3_field_kv)KV(26, d->slot);
   ov[n++] = (struct agx3_field_kv)KV(27, d->coord_kill ? 1 : 0);
   ov[n++] = (struct agx3_field_kv)KV(37, 1);  /* immediate LOD */
   ov[n++] = (struct agx3_field_kv)KV(75, d->mask);
   ov[n++] = (struct agx3_field_kv)KV(76, 1);  /* 2D */
   if (d->array)
      ov[n++] = (struct agx3_field_kv)KV(77, 2); /* array, register layer */
   ov[n++] = (struct agx3_field_kv)KV(78, 1);  /* texel fetch */
   /* Each operand lands in the field its kind selects: 95/96/97 for the
    * texture, 124/125/126 for the sampler. */
   ov[n++] = (struct agx3_field_kv)KV(
      tk == AGX3_TEX_OPERAND_IMMEDIATE   ? 95
      : tk == AGX3_TEX_OPERAND_REGISTER  ? 96
                                         : 97,
      tk == AGX3_TEX_OPERAND_IMMEDIATE ? d->texture : d->texture_value);
   ov[n++] = (struct agx3_field_kv)KV(
      sk == AGX3_TEX_OPERAND_IMMEDIATE   ? 124
      : sk == AGX3_TEX_OPERAND_REGISTER  ? 125
                                         : 126,
      d->sampler_value);
   ov[n++] = (struct agx3_field_kv)KV(107, d->texture_kill);
   ov[n++] = (struct agx3_field_kv)KV(139, 2); /* LOD0 immediate shape */
   if (enc == AGX_ISA_ENCODING_AGX3B)
      ov[n++] = (struct agx3_field_kv)KV(147, d->final ? 2 : 0);
   else {
      ov[n++] = (struct agx3_field_kv)KV(150, d->final ? 2 : 0);
      ov[n++] = (struct agx3_field_kv)KV(155, 1);
   }
   for (unsigned i = 0; i < 8; ++i)
      if (d->wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(62 + i, 1);

   /*
    * Immediate operands are ep3; register and uniform operands are ep4. Only
    * AGX3b is witnessed for the non-immediate kinds, so AGX3c refuses them
    * rather than assuming its encoding points line up.
    */
   if (!immediate_operands && enc != AGX_ISA_ENCODING_AGX3B)
      return false;
   const int ep =
      enc == AGX_ISA_ENCODING_AGX3B ? (immediate_operands ? 3 : 4) : 0;
   if (!agx3_bind_compose(enc, "smp", ep, base, nb, ov, n, out, len)) {
      memset(out, 0, AGX3_TEX_MAX_LENGTH);
      *len = 0;
      return false;
   }
   return true;
}

static bool
mask_witnessed(uint8_t mask)
{
   /* All 15 nonempty RGBA subsets are witnessed by native fragment samples.
    * Results remain densely packed in selected-component order. */
   return mask != 0 && (mask & ~15u) == 0;
}

static bool
sample_ok(enum agx_isa_encoding enc, const struct agx3_tex_sample_desc *d)
{
   if (d == NULL || (enc != AGX_ISA_ENCODING_AGX3B &&
                     enc != AGX_ISA_ENCODING_AGX3C))
      return false;

   if ((d->width != AGX3_TEX_W16 && d->width != AGX3_TEX_W32) ||
       d->dest >= 128 || (d->width == AGX3_TEX_W32 && (d->dest & 1)) ||
       (d->coord & 3) || d->coord > 60 || !mask_witnessed(d->mask) ||
       d->texture > 15 || d->sampler > 15)
      return false;

   if (d->lod == AGX3_TEX_LOD_LEVEL) {
      /* 12-bit immediate level field; witnessed up to 464 (level 7.25). */
      if (d->lod_x64 > 0xfff)
         return false;
   } else if ((d->lod != AGX3_TEX_LOD_AUTO &&
               !(d->lod == AGX3_TEX_LOD_REGISTER &&
                 enc == AGX_ISA_ENCODING_AGX3B)) || d->lod_x64 != 0) {
      return false;
   }

   if (enc == AGX_ISA_ENCODING_AGX3C && d->mask != 1 && d->mask != 3 &&
       d->mask != 8 && d->mask != 12 && d->mask != 15)
      return false;

   /* The array selector and register-layer operand are grounded only for the
    * G17P implicit-LOD form. */
   if (d->array &&
       (enc != AGX_ISA_ENCODING_AGX3B || d->lod != AGX3_TEX_LOD_AUTO))
      return false;

   const unsigned tk = d->texture_kind ? d->texture_kind : 1;
   const unsigned sk = d->sampler_kind ? d->sampler_kind : 1;
   if (tk > 3 || sk > 3 || (tk == 1) != (sk == 1) ||
       ((tk != 1 || sk != 1) && enc != AGX_ISA_ENCODING_AGX3B) ||
       (tk != 1 && ((d->texture_value & 1) || d->texture_value > 252)) ||
       (sk != 1 && ((d->sampler_value & 1) || d->sampler_value > 252)) ||
       (d->texture_kill && tk != 2) || (d->sampler_kill && sk != 2))
      return false;

   const unsigned slots = enc == AGX_ISA_ENCODING_AGX3C ? 8 : 6;
   const uint16_t waits = enc == AGX_ISA_ENCODING_AGX3C ? 0xff : 0x3f;
   return d->slot >= 1 && d->slot <= slots &&
          (d->wait_mask & ~waits) == 0 && one_or_zero(d->wait_mask);
}

bool
agx3_tex_encode_sample(enum agx_isa_encoding enc,
                       const struct agx3_tex_sample_desc *d,
                       uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   if (out == NULL || len == NULL)
      return false;
   memset(out, 0, AGX3_TEX_MAX_LENGTH);
   *len = 0;
   if (!sample_ok(enc, d))
      return false;

   /* The fetch baseline plus f74=1: the sample witnesses carry the extra
    * mask-group bit on every mask value, where the fetch form does not. */
   const unsigned tk = d->texture_kind ? d->texture_kind : 1;
   const unsigned sk = d->sampler_kind ? d->sampler_kind : 1;
   const bool immediate = tk == 1 && sk == 1;
   struct agx3_field_kv base[20];
   unsigned nb = 0;
   base[nb++] = (struct agx3_field_kv)KV(9, 2);
   base[nb++] = (struct agx3_field_kv)KV(23, 5);
   base[nb++] = (struct agx3_field_kv)KV(38, 1);
   base[nb++] = (struct agx3_field_kv)KV(52, 1);
   base[nb++] = (struct agx3_field_kv)KV(74, 1);
   base[nb++] = (struct agx3_field_kv)KV(89, 3);
   base[nb++] = (struct agx3_field_kv)KV(90, 3);
   base[nb++] = (struct agx3_field_kv)KV(103, tk);
   base[nb++] = (struct agx3_field_kv)KV(104, immediate ? 1 : 2);
   base[nb++] = (struct agx3_field_kv)KV(112, 4);
   base[nb++] = (struct agx3_field_kv)KV(118, 3);
   base[nb++] = (struct agx3_field_kv)KV(119, 3);
   base[nb++] = (struct agx3_field_kv)KV(132, sk);
   base[nb++] = (struct agx3_field_kv)KV(133, immediate ? 1 : 2);
   base[nb++] = (struct agx3_field_kv)KV(138, 1);
   if (enc == AGX_ISA_ENCODING_AGX3C) {
      base[nb++] = (struct agx3_field_kv)KV(24, 2);
      base[nb++] = (struct agx3_field_kv)KV(60, 2);
      base[nb++] = (struct agx3_field_kv)KV(61, 164);
   }

   struct agx3_field_kv ov[26];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(2, d->dest);
   ov[n++] = (struct agx3_field_kv)KV(10, d->width);
   ov[n++] = (struct agx3_field_kv)KV(12, d->slot);
   ov[n++] = (struct agx3_field_kv)KV(19, d->coord);
   ov[n++] = (struct agx3_field_kv)KV(26, d->slot);
   ov[n++] = (struct agx3_field_kv)KV(27, d->coord_kill ? 1 : 0);
   ov[n++] = (struct agx3_field_kv)KV(75, d->mask);
   ov[n++] = (struct agx3_field_kv)KV(76, 1);  /* 2D */
   ov[n++] = (struct agx3_field_kv)KV(78, 2);  /* filtered sample */
   if (d->array)
      ov[n++] = (struct agx3_field_kv)KV(77, 2); /* register array layer */
   ov[n++] = (struct agx3_field_kv)KV(94 + tk, tk == 1 ? d->texture : d->texture_value);
   ov[n++] = (struct agx3_field_kv)KV(123 + sk, sk == 1 ? d->sampler : d->sampler_value);
   if (d->texture_kill)
      ov[n++] = (struct agx3_field_kv)KV(107, 1);
   if (d->sampler_kill)
      ov[n++] = (struct agx3_field_kv)KV(136, 1);
   if (d->lod == AGX3_TEX_LOD_AUTO) {
      ov[n++] = (struct agx3_field_kv)KV(156, 1); /* implicit-LOD marker */
   } else if (d->lod == AGX3_TEX_LOD_REGISTER) {
      ov[n++] = (struct agx3_field_kv)KV(28, 0); /* packed level bank */
      ov[n++] = (struct agx3_field_kv)KV(139, 4);
      ov[n++] = (struct agx3_field_kv)KV(156, 1); /* native fragment form */
   } else {
      ov[n++] = (struct agx3_field_kv)KV(37, 1);  /* immediate LOD operand */
      ov[n++] = (struct agx3_field_kv)KV(139, 2); /* explicit-level mode */
      ov[n++] = (struct agx3_field_kv)KV(29, d->lod_x64);
   }
   if (enc == AGX_ISA_ENCODING_AGX3B)
      ov[n++] = (struct agx3_field_kv)KV(147, d->final ? 2 : 0);
   else
      ov[n++] = (struct agx3_field_kv)KV(150, d->final ? 2 : 0);
   for (unsigned i = 0; i < 8; ++i)
      if (d->wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(62 + i, 1);

   if (!agx3_bind_compose(enc, "smp", immediate ? 3 : 4, base, nb, ov, n, out, len)) {
      memset(out, 0, AGX3_TEX_MAX_LENGTH);
      *len = 0;
      return false;
   }
   return true;
}

bool
agx3_tex_stage_copy_hints(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                    uint8_t slot, uint16_t wait_mask,
                          bool src_cache, bool src_discard,
                    uint8_t out[AGX3_TEX_COPY_MAX_LENGTH], uint8_t *len)
{
   if (out == NULL || len == NULL)
      return false;
   memset(out, 0, AGX3_TEX_COPY_MAX_LENGTH);
   *len = 0;
   if ((src_cache && src_discard) ||
       (enc == AGX_ISA_ENCODING_AGX3C && (src_cache || !src_discard)))
      return false;

   if (enc != AGX_ISA_ENCODING_AGX3B && enc != AGX_ISA_ENCODING_AGX3C)
      return false;

   /* One even 32-bit register each way, as in every witness. */
   if ((dest & 1) || (src & 1) || dest >= 256 || src >= 256)
      return false;

   const unsigned slots = enc == AGX_ISA_ENCODING_AGX3C ? 8 : 6;
   const uint16_t waits = enc == AGX_ISA_ENCODING_AGX3C ? 0xff : 0x3f;
   if (slot < 1 || slot > slots || (wait_mask & ~waits) ||
       !one_or_zero(wait_mask))
      return false;

   struct agx3_field_kv base[8];
   unsigned nb = 0;
   base[nb++] = (struct agx3_field_kv)KV(9, 5);
   base[nb++] = (struct agx3_field_kv)KV(10, 2);
   base[nb++] = (struct agx3_field_kv)KV(23, 2);
   base[nb++] = (struct agx3_field_kv)KV(24, 2);
   if (enc == AGX_ISA_ENCODING_AGX3C) {
      base[nb++] = (struct agx3_field_kv)KV(28, 1);
      base[nb++] = (struct agx3_field_kv)KV(29, 35);
   }

   struct agx3_field_kv ov[12];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(5, dest);
   ov[n++] = (struct agx3_field_kv)KV(16, src & 0x7f);
   ov[n++] = (struct agx3_field_kv)KV(25, src_cache);
   ov[n++] = (struct agx3_field_kv)KV(27, src_discard);
   ov[n++] = (struct agx3_field_kv)KV(12, slot);
   for (unsigned i = 0; i < 8; ++i)
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(30 + i, 1);

   /* The compact form first; Apple escalates the encoding point when a slot
    * or wait bit does not fit, and the composer's first-success order
    * reproduces that choice on every witness. */
   for (int ep = 0; ep <= 1; ++ep) {
      if (agx3_bind_compose(enc, "copy", ep, base, nb, ov, n, out, len))
         return true;
   }
   memset(out, 0, AGX3_TEX_COPY_MAX_LENGTH);
   *len = 0;
   return false;
}

bool
agx3_tex_stage_copy(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                    uint8_t slot, uint16_t wait_mask,
                    uint8_t out[AGX3_TEX_COPY_MAX_LENGTH], uint8_t *len)
{
   return agx3_tex_stage_copy_hints(enc,dest,src,slot,wait_mask,false,true,out,len);
}

bool
agx3_tex_encode_image_write_rgba8_2d(
   enum agx_isa_encoding enc, uint16_t wait_mask, uint8_t image_index,
   bool normalized,
   uint8_t out[AGX3_IMAGE_WRITE_LENGTH], uint8_t *len)
{
   if (out == NULL || len == NULL)
      return false;

   memset(out, 0, AGX3_IMAGE_WRITE_LENGTH);
   *len = 0;
   if (enc != AGX_ISA_ENCODING_AGX3B || wait_mask != 1 ||
       !((image_index == 0 && !normalized) ||
         (image_index == 1 && normalized)))
      return false;

   /* Nonzero field values recovered by decoding the native 16-byte
    * imgwci.ep0. Every omitted field is intentionally zero; the composer
    * validates the complete selected form. Fields 70/91 are the only
    * differences in Apple's sampled-float-to-write-state1 witness. */
   const struct agx3_field_kv fields[] = {
      KV(9, 5),   KV(12, 1),  KV(13, 1),   KV(24, 1), KV(38, 1),
      KV(51, 1),  KV(52, 1),  KV(60, 1),   KV(71, 2), KV(72, 1),
      KV(74, 1),  KV(79, 4),  KV(85, 3),   KV(86, 3), KV(99, 1),
      KV(100, 1), KV(113, 1), KV(70, normalized), KV(91, image_index),
   };

   if (!agx3_bind_compose(enc, "imgwci", 0, fields,
                          sizeof(fields) / sizeof(fields[0]),
                          NULL, 0, out, len) ||
       *len != AGX3_IMAGE_WRITE_LENGTH) {
      memset(out, 0, AGX3_IMAGE_WRITE_LENGTH);
      *len = 0;
      return false;
   }

   return true;
}

static uint16_t
word(const uint8_t *p, unsigned i)
{
   return (uint16_t)p[2 * i] | ((uint16_t)p[2 * i + 1] << 8);
}

static bool
decode_b(const uint8_t *p, struct agx3_tex_fetch_desc *d)
{
   const uint16_t w0 = word(p, 0), w1 = word(p, 1), w2 = word(p, 2),
                  w4 = word(p, 4), w5 = word(p, 5);
   d->width = (w0 & 0x8000) ? AGX3_TEX_W32 : AGX3_TEX_W16;
   d->dest = d->width == AGX3_TEX_W32
                ? (((w0 >> 3) & 0x1f) << 1) | (((w1 >> 6) & 1) << 6)
                : ((w0 >> 3) & 0x1f) | (((w1 >> 6) & 3) << 5);
   d->coord = (((w0 >> 8) & 7) << 1) | (((w1 >> 8) & 3) << 4);
   d->coord_kill = (w1 >> 13) & 1;
   d->final = (w1 >> 15) & 1;

   const unsigned mask_code = ((w1 >> 11) & 3) | (((w2 >> 5) & 1) << 2) |
                              (((w5 >> 6) & 1) << 3);
   static const uint8_t masks[16] = {
      1, 2, 4, 8, 9, 3, 7, 15, 5, 6, 10, 12, 13, 11, 14, 0,
   };
   d->mask = masks[mask_code];
   if (!d->mask)
      return false;
   d->sampler_value = ((w4 >> 8) & 1) | ((w2 & 0xf) << 1) |
                      (((w4 >> 1) & 7) << 5);

   const unsigned slot_code = (w2 >> 10) & 7;
   if (slot_code > 5)
      return false;
   d->slot = slot_code + 1;
   d->texture = ((w4 >> 7) & 1) | (((w0 >> 11) & 0xf) << 1) |
                (((w4 >> 4) & 7) << 5);

   const unsigned wait_code = (w2 >> 7) & 7;
   if (wait_code > 6)
      return false;
   d->wait_mask = wait_code == 0 ? 0 : (1u << (wait_code - 1));
   return true;
}

static bool
decode_c(const uint8_t *p, struct agx3_tex_fetch_desc *d)
{
   const uint16_t w0 = word(p, 0), w1 = word(p, 1), w2 = word(p, 2),
                  w3 = word(p, 3);
   d->width = (w0 & 8) ? AGX3_TEX_W32 : AGX3_TEX_W16;
   d->dest = ((w3 >> 2) & 1) | (((w0 >> 4) & 0xf) << 1) |
             (((w1 >> 6) & 3) << 5) | (((w3 >> 12) & 1) << 7);
   d->coord = (((w1 >> 10) & 0x3f) << 2) | (((w1 >> 9) & 1) << 8);
   d->coord_kill = (w1 >> 4) & 1;
   d->final = (w2 >> 15) & 1;

   const unsigned mask_code = ((w3 >> 5) & 3) | (((w3 >> 11) & 1) << 2);
   if (mask_code == 0)
      d->mask = 1;
   else if (mask_code == 7)
      d->mask = 15;
   else
      return false;

   d->slot = ((w2 >> 3) & 3) | (((w3 >> 4) & 1) << 2);
   d->slot++;
   d->texture = ((w3 >> 3) & 1) | (((w2 >> 12) & 1) << 1) |
                (((w2 >> 13) & 3) << 2);
   const unsigned wait_code = (w3 >> 13) & 7;
   d->wait_mask = wait_code == 0 ? 0 : (1u << (wait_code - 1));
   return true;
}

bool
agx3_tex_decode_fetch(enum agx_isa_encoding enc, const void *instruction,
                      uint32_t instruction_size,
                      struct agx3_tex_fetch_desc *out)
{
   if (out == NULL)
      return false;
   memset(out, 0, sizeof(*out));
   if (enc != AGX_ISA_ENCODING_AGX3B && enc != AGX_ISA_ENCODING_AGX3C)
      return false;
   if (enc == AGX_ISA_ENCODING_AGX3B && instruction && instruction_size == 20) {
      const uint8_t *p = instruction;
      struct agx3_tex_fetch_desc d = {0};
      if (!decode_b(p, &d))
         return false;
      d.texture = 0;
      d.texture_value = (((word(p, 0) >> 12) & 7) << 1) |
                        (((word(p, 4) >> 4) & 7) << 4) |
                        (((p[15] >> 2) & 1) << 7);
      d.wait_mask = ((p[4] >> 7) & 1) | ((p[5] & 3) << 1) |
                    (((p[18] >> 4) & 7) << 3);
      d.sampler_kind = AGX3_TEX_OPERAND_IMMEDIATE;
      /* Repack both native index kinds. Their context selector, upper bits,
       * and every reserved field are validated by exact byte comparison. */
      for (unsigned kind = AGX3_TEX_OPERAND_REGISTER;
           kind <= AGX3_TEX_OPERAND_UNIFORM; ++kind) {
         d.texture_kind = kind;
         d.texture_kill = kind == AGX3_TEX_OPERAND_REGISTER && (p[15] & 8);
         for (unsigned array = 0; array < 2; ++array) {
            d.array = array;
            uint8_t canonical[AGX3_TEX_MAX_LENGTH], length;
            if (agx3_tex_encode_fetch(enc, &d, canonical, &length) &&
                length == instruction_size &&
                !memcmp(canonical, instruction, length)) {
               *out = d;
               return true;
            }
         }
      }
      return false;
   }
   const uint32_t expected = enc == AGX_ISA_ENCODING_AGX3B ? 14 : 8;
   if (instruction == NULL || instruction_size != expected)
      return false;

   struct agx3_tex_fetch_desc d = {0};
   const uint8_t *p = instruction;
   if (!(enc == AGX_ISA_ENCODING_AGX3B ? decode_b(p, &d) : decode_c(p, &d)) ||
       !fetch_ok(enc, &d))
      return false;

   /* The array axis is a single selector the hand decoders above do not
    * extract; recover it the same way everything else here is validated, by
    * requiring one of the two readings to repack byte-for-byte. */
   for (unsigned a = 0; a < 2; ++a) {
      uint8_t canonical[AGX3_TEX_MAX_LENGTH];
      uint8_t length = 0;
      d.array = a != 0;
      if (agx3_tex_encode_fetch(enc, &d, canonical, &length) &&
          length == instruction_size &&
          memcmp(canonical, instruction, length) == 0) {
         *out = d;
         return true;
      }
   }

   return false;
}

/* Fresh native G17P LOD fragment witnesses use this 12-byte PACK_SMPIR for a
 * scalar half LOD, preserving the source when it remains live. f5=6 selects
 * operand bank 6, not architectural r3. */
bool
agx3_tex_stage_lod(enum agx_isa_encoding enc, uint16_t src, uint8_t slot,
                   uint16_t wait_mask, bool src_cache, bool src_discard,
                   uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len)
      return false;
   memset(out, 0, AGX3_TEX_MAX_LENGTH);
   *len = 0;
   if (enc != AGX_ISA_ENCODING_AGX3B || src >= 256 || slot < 1 || slot > 6 ||
       (wait_mask & ~0x3f) || (src_cache && src_discard))
      return false;

   const struct agx3_field_kv base[] = {
      KV(5, 6), KV(9, 5), KV(10, 2), KV(23, 2), KV(24, 1),
      KV(38, 1), KV(51, 1), KV(52, 2), KV(74, 2), KV(75, 16),
   };
   struct agx3_field_kv ov[10];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(16, src & 0x7f);
   ov[n++] = (struct agx3_field_kv)KV(12, slot);
   ov[n++] = (struct agx3_field_kv)KV(25, src_cache);
   ov[n++] = (struct agx3_field_kv)KV(27, src_discard);
   for (unsigned i = 0; i < 6; ++i)
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(58 + i, 1);
   if (!agx3_bind_compose(enc, "pack_smpir", 0, base,
                         sizeof(base) / sizeof(base[0]), ov, n, out, len))
      return false;

   /* The source is an eight-bit half-register index (f16 bits 0..6 at
    * instruction bits 41..47, bit 7 at bit 48). Apple's encoder's 191
    * bound is a compiler allocation policy, not a narrower wire field.
    * Keep the same native instruction for Mesa's full 256-half bank. */
   out[6] |= (src >> 7) & 1;
   return true;
}

/* The dynamic array-layer witness uses PACK_SMPIR like register LOD, but its
 * operand-type selectors differ. In particular, f74=4/f75=0 produces the
 * native 1705540600080000f0c00400 for discarded r2 on slot 1; the LOD
 * baseline's f74=2/f75=16 silently leaves the sample destination unwritten.
 * Register-indexed array samples use the independently witnessed slot-6 form
 * 1705540600180000f0e40400 for discarded r6. */
bool
agx3_tex_stage_array_layer(enum agx_isa_encoding enc, uint16_t src,
                           uint8_t slot, uint16_t wait_mask, bool src_cache,
                           bool src_discard,
                           uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len)
      return false;
   memset(out, 0, AGX3_TEX_MAX_LENGTH);
   *len = 0;
   if (enc != AGX_ISA_ENCODING_AGX3B || src >= 256 || slot < 1 || slot > 6 ||
       (wait_mask & ~0x3f) || (src_cache && src_discard))
      return false;

   const struct agx3_field_kv base[] = {
      KV(5, 6), KV(9, 5), KV(10, 2), KV(23, 2), KV(24, 1),
      KV(38, 1), KV(51, 1), KV(52, 2), KV(74, 4),
   };
   struct agx3_field_kv ov[10];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(16, src & 0x7f);
   ov[n++] = (struct agx3_field_kv)KV(12, slot);
   ov[n++] = (struct agx3_field_kv)KV(25, src_cache);
   ov[n++] = (struct agx3_field_kv)KV(27, src_discard);
   for (unsigned i = 0; i < 6; ++i)
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(58 + i, 1);
   if (!agx3_bind_compose(enc, "pack_smpir", 0, base,
                         sizeof(base) / sizeof(base[0]), ov, n, out, len))
      return false;

   out[6] |= (src >> 7) & 1;
   return true;
}
