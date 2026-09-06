/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_cf_core.h"

#include "agx3_binder.h"

#include <string.h>

/* Baselines recovered by decoding Apple's witnessed templates; only the
 * non-zero constant fields need setting (the composer defaults the rest to 0).
 * AGX3c compares, cnd ops and branches carry extra selector constants. */
#define KV(f, v) {(f), (v)}

static bool
agx3b(enum agx_isa_encoding enc)
{
   return enc == AGX_ISA_ENCODING_AGX3B;
}

static bool
slot_ok(unsigned slot)
{
   return slot <= AGX3_CF_MAX_FLAG_SLOT;
}

static bool
icond_ok(enum agx3_cf_icond cond)
{
   return cond == AGX3_CF_ICOND_ULT || cond == AGX3_CF_ICOND_UGT ||
          cond == AGX3_CF_ICOND_EQ || cond == AGX3_CF_ICOND_SLT ||
          cond == AGX3_CF_ICOND_SGT;
}

static bool
fcond_ok(enum agx3_cf_fcond cond)
{
   return cond == AGX3_CF_FCOND_EQ || cond == AGX3_CF_FCOND_LT ||
          cond == AGX3_CF_FCOND_GT;
}

/* Native half comparisons use the half indices directly. Signed relations
 * sign-extend both inputs; cache/discard belongs to each source independently.
 * The compact unsigned form is six bytes, signed forms are ten bytes. */
bool
agx3_cf_icmp_reg16(enum agx_isa_encoding enc, enum agx3_cf_icond cond,
                   unsigned slot, const uint16_t src[2],
                   const struct agx3_cf_source_hints hints[2], uint16_t wait_mask,
                   uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len)
{
   if (!agx3b(enc) || !icond_ok(cond) || !slot_ok(slot) || !src || !hints ||
       !out || !len || (wait_mask != 0 && wait_mask != 32))
      return false;
   for (unsigned i = 0; i < 2; ++i)
      if (src[i] >= 256 || (hints[i].cache && hints[i].discard))
         return false;
   const bool sign_extend = cond == AGX3_CF_ICOND_SLT || cond == AGX3_CF_ICOND_SGT;
   const struct agx3_field_kv base[] = {
      KV(9,2), KV(10,1), KV(24,2), KV(25,1),
   };
   const struct agx3_field_kv fields[] = {
      KV(2,src[0]), KV(17,src[1]), KV(31,slot), KV(42,cond),
      KV(11,hints[0].cache), KV(13,hints[0].discard), KV(14,sign_extend),
      KV(26,hints[1].cache), KV(28,hints[1].discard), KV(29,sign_extend),
      KV(39,wait_mask == 32),
   };
   for (unsigned ep = 0; ep < 2; ++ep)
      if (agx3_bind_compose(enc,"icmp",ep,base,4,fields,11,out,len))
         return true;
   return false;
}

bool
agx3_cf_icmp_reg(enum agx_isa_encoding enc, enum agx3_cf_icond cond,
                 unsigned slot, uint16_t src0, uint16_t src1,
                 uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len)
{
   if (!icond_ok(cond) || !slot_ok(slot) || src0 >= 128 || src1 >= 128)
      return false;
   const struct agx3_field_kv base_b[] = {KV(9, 2), KV(10, 2), KV(24, 2),
                                          KV(25, 2)};
   const struct agx3_field_kv base_c[] = {KV(9, 2),  KV(10, 2), KV(24, 2),
                                          KV(25, 2), KV(32, 3), KV(33, 82)};
   const struct agx3_field_kv ov[] = {KV(2, src0), KV(17, src1), KV(42, cond),
                                      KV(31, slot)};
   return agx3_bind_compose(enc, "icmp", 1, agx3b(enc) ? base_b : base_c,
                            agx3b(enc) ? 4 : 6, ov, 4, out, len);
}

bool
agx3_cf_icmp_imm(enum agx_isa_encoding enc, enum agx3_cf_icond cond,
                 unsigned slot, uint16_t src0, uint8_t imm,
                 uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len)
{
   if (!icond_ok(cond) || !slot_ok(slot) || src0 >= 128)
      return false;
   const struct agx3_field_kv base_b[] = {KV(9, 2), KV(10, 2), KV(24, 1),
                                          KV(25, 1)};
   const struct agx3_field_kv base_c[] = {KV(9, 2),  KV(10, 2), KV(24, 1),
                                          KV(25, 1), KV(32, 3), KV(33, 82)};
   const struct agx3_field_kv ov[] = {KV(2, src0), KV(16, imm), KV(42, cond),
                                      KV(31, slot)};
   return agx3_bind_compose(enc, "icmp", 1, agx3b(enc) ? base_b : base_c,
                            agx3b(enc) ? 4 : 6, ov, 4, out, len);
}

bool
agx3_cf_icmp_zero16_runtime(enum agx_isa_encoding enc, enum agx3_cf_icond cond,
                            unsigned slot, uint16_t src,
                            struct agx3_cf_source_hints hint, uint16_t wait_mask,
                            uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len)
{
   if (!agx3b(enc) || !icond_ok(cond) || !slot_ok(slot) || src >= 256 ||
       !out || !len || (hint.cache && hint.discard) ||
       (wait_mask != 0 && wait_mask != 32))
      return false;
   /* Native signed and unsigned zero tests both sign-extend the half.
    * This preserves equality and every relation to zero. Live witnesses
    * retain the source for the taken branch instead of discarding it. */
   const struct agx3_field_kv base[] = {
      KV(9,2), KV(10,1), KV(14,1), KV(24,1), KV(25,1),
   };
   const struct agx3_field_kv fields[] = {
      KV(2,src), KV(42,cond), KV(31,slot), KV(11,hint.cache),
      KV(13,hint.discard), KV(39,wait_mask == 32),
   };
   return agx3_bind_compose(enc,"icmp",1,base,5,fields,6,out,len);
}

bool
agx3_cf_icmp_zero16(enum agx_isa_encoding enc, enum agx3_cf_icond cond,
                    unsigned slot, uint16_t src0,
                    uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len)
{
   /* Preserve the historical native specimen API; production uses the
    * runtime variant with register-allocation hints and actual waits. */
   return src0 < 128 && agx3_cf_icmp_zero16_runtime(enc,cond,slot,src0,
                 (struct agx3_cf_source_hints){false,true},32,out,len);
}

bool
agx3_cf_fcmp_reg(enum agx_isa_encoding enc, enum agx3_cf_fcond cond,
                 bool or_nan, unsigned slot, uint16_t src0, uint16_t src1,
                 uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len)
{
   if (!fcond_ok(cond) || !slot_ok(slot) || src0 >= 128 || src1 >= 128)
      return false;
   const struct agx3_field_kv base_b[] = {KV(9, 2), KV(10, 2), KV(26, 2),
                                          KV(27, 2)};
   const struct agx3_field_kv base_c[] = {KV(9, 2),  KV(10, 2), KV(26, 2),
                                          KV(27, 2), KV(36, 3), KV(37, 69)};
   /* or_nan folds into the condition: lt->5, gt->6 (eq stays 0). */
   uint64_t fcond = cond;
   if (or_nan && cond == AGX3_CF_FCOND_LT)
      fcond = 5;
   else if (or_nan && cond == AGX3_CF_FCOND_GT)
      fcond = 6;
   else if (or_nan)
      return false;
   const struct agx3_field_kv ov[] = {KV(2, src0), KV(19, src1), KV(46, fcond),
                                      KV(34, or_nan ? 1 : 0), KV(35, slot)};
   return agx3_bind_compose(enc, "fcmp", 1, agx3b(enc) ? base_b : base_c,
                            agx3b(enc) ? 4 : 6, ov, 5, out, len);
}

bool
agx3_cf_fcmp_reg_neg(enum agx_isa_encoding enc, enum agx3_cf_fcond cond,
                     bool or_nan, unsigned slot, const uint16_t src_in[2],
                     const bool neg_in[2],
                     const struct agx3_cf_source_hints hints_in[2],
                     uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !src_in || !neg_in || !hints_in ||
       !out || !len || !slot_ok(slot) || or_nan ||
       (cond != AGX3_CF_FCOND_LT && cond != AGX3_CF_FCOND_GT) ||
       (neg_in[0] == neg_in[1]) || src_in[0] >= 128 || src_in[1] >= 128 ||
       (src_in[0] & 1) || (src_in[1] & 1) ||
       (hints_in[0].cache && hints_in[0].discard) ||
       (hints_in[1].cache && hints_in[1].discard))
      return false;

   uint16_t src[2] = {src_in[0], src_in[1]};
   struct agx3_cf_source_hints hints[2] = {hints_in[0], hints_in[1]};
   if (neg_in[0]) {
      const uint16_t tmp = src[0];
      src[0] = src[1];
      src[1] = tmp;
      const struct agx3_cf_source_hints hint = hints[0];
      hints[0] = hints[1];
      hints[1] = hint;
      cond = cond == AGX3_CF_FCOND_LT ? AGX3_CF_FCOND_GT
                                      : AGX3_CF_FCOND_LT;
   }

   const struct agx3_field_kv base[] = {
      KV(9, 2), KV(10, 2), KV(26, 2), KV(27, 2), KV(38, 1)};
   const struct agx3_field_kv ov[] = {
      KV(2, src[0]), KV(11, hints[0].cache), KV(13, hints[0].discard),
      KV(19, src[1]), KV(28, hints[1].discard), KV(30, hints[1].cache),
      KV(32, 1), KV(35, slot), KV(46, cond)};
   return agx3_bind_compose(enc, "fcmp", 0, base, 5, ov, 9, out, len);
}

bool
agx3_cf_cndst(enum agx_isa_encoding enc, unsigned slot, bool invert,
              unsigned push_count, uint8_t out[AGX3_CF_MAX_LENGTH],
              uint8_t *len)
{
   if ((slot != 6 && !slot_ok(slot)) || (push_count != 1 && push_count != 2))
      return false;
   const struct agx3_field_kv base_c[] = {KV(2, 3), KV(3, 33)};
   const struct agx3_field_kv ov[] = {KV(0, invert ? 1 : 0), KV(1, slot),
                                      KV(12, push_count)};
   return agx3_bind_compose(enc, "cndst", 0, base_c, agx3b(enc) ? 0 : 2, ov, 3,
                            out, len);
}

bool
agx3_cf_cndlt(enum agx_isa_encoding enc, unsigned slot, bool invert,
              uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len)
{
   if (!slot_ok(slot))
      return false;
   const struct agx3_field_kv base_b[] = {KV(12, 2)};
   const struct agx3_field_kv base_c[] = {KV(2, 3), KV(3, 31), KV(12, 2)};
   const struct agx3_field_kv ov[] = {KV(0, invert ? 1 : 0), KV(1, slot)};
   return agx3_bind_compose(enc, "cndlt", 0, agx3b(enc) ? base_b : base_c,
                            agx3b(enc) ? 1 : 3, ov, 2, out, len);
}

bool
agx3_cf_cndef(enum agx_isa_encoding enc, uint8_t out[AGX3_CF_MAX_LENGTH],
              uint8_t *len)
{
   const struct agx3_field_kv base_b[] = {KV(1, 6), KV(12, 1), KV(13, 2),
                                          KV(14, 1)};
   const struct agx3_field_kv base_c[] = {KV(1, 6),  KV(2, 3),  KV(3, 29),
                                          KV(12, 1), KV(16, 2), KV(17, 1),
                                          KV(18, 2)};
   return agx3_bind_compose(enc, "cndef", 0, agx3b(enc) ? base_b : base_c,
                            agx3b(enc) ? 4 : 7, NULL, 0, out, len);
}

bool
agx3_cf_cndend(enum agx_isa_encoding enc, unsigned pop_count,
               uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len)
{
   if (pop_count != 1 && pop_count != 2)
      return false;
   const struct agx3_field_kv base_b[] = {KV(11, 2), KV(12, 2)};
   const struct agx3_field_kv base_c[] = {KV(0, 3), KV(1, 30), KV(14, 2),
                                          KV(15, 2), KV(16, 2)};
   const struct agx3_field_kv ov[] = {KV(10, pop_count)};
   return agx3_bind_compose(enc, "cndend", 0, agx3b(enc) ? base_b : base_c,
                            agx3b(enc) ? 2 : 5, ov, 1, out, len);
}

bool
agx3_cf_branch(enum agx_isa_encoding enc, bool is_bn,
               uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len)
{
   const char *mnem = is_bn ? "bn" : "b";
   const int ep = agx3b(enc) ? 0 : 1;
   const struct agx3_field_kv base_c[] = {KV(1, 3), KV(2, is_bn ? 21 : 8)};
   return agx3_bind_compose(enc, mnem, ep, base_c, agx3b(enc) ? 0 : 2, NULL, 0,
                            out, len);
}

/* OR `width` bits of `value` into `buf` starting at instruction bit
 * `instr_bit` (little-endian byte order). The reserved template has zeros in
 * every displacement slice, so a pure OR is a faithful deposit. */
static void
deposit(uint8_t *buf, unsigned instr_bit, unsigned width, uint64_t value)
{
   for (unsigned i = 0; i < width; ++i) {
      if ((value >> i) & 1u) {
         const unsigned bit = instr_bit + i;
         buf[bit / 8] |= (uint8_t)(1u << (bit % 8));
      }
   }
}

bool
agx3_cf_patch_branch(enum agx_isa_encoding enc, bool is_bn, uint8_t *branch,
                     int64_t displacement)
{
   (void)is_bn; /* b and bn share the f0 scatter within an encoding. */
   if (branch == NULL || (displacement & 1) != 0)
      return false;

   /* 48-bit two's-complement range. */
   if (displacement < -(INT64_C(1) << 47) || displacement >= (INT64_C(1) << 47))
      return false;

   const uint64_t d = (uint64_t)displacement & ((INT64_C(1) << 48) - 1);

   if (enc == AGX_ISA_ENCODING_AGX3B) {
      deposit(branch, 1 * 16 + 9, 7, d >> 1);   /* disp[1:7]   */
      deposit(branch, 2 * 16 + 0, 16, d >> 8);  /* disp[8:23]  */
      deposit(branch, 3 * 16 + 0, 16, d >> 24); /* disp[24:39] */
      deposit(branch, 4 * 16 + 0, 8, d >> 40);  /* disp[40:47] */
      return true;
   }

   if (enc == AGX_ISA_ENCODING_AGX3C) {
      deposit(branch, 0 * 16 + 7, 1, d >> 1);   /* disp[1]     */
      deposit(branch, 0 * 16 + 12, 3, d >> 2);  /* disp[2:4]   */
      deposit(branch, 1 * 16 + 1, 1, d >> 5);   /* disp[5]     */
      deposit(branch, 1 * 16 + 3, 1, d >> 6);   /* disp[6]     */
      deposit(branch, 1 * 16 + 5, 3, d >> 7);   /* disp[7:9]   */
      deposit(branch, 0 * 16 + 8, 4, d >> 10);  /* disp[10:13] */
      deposit(branch, 1 * 16 + 12, 1, d >> 14); /* disp[14]    */
      deposit(branch, 2 * 16 + 0, 5, d >> 15);  /* disp[15:19] */
      deposit(branch, 3 * 16 + 0, 11, d >> 20); /* disp[20:30] */
      deposit(branch, 3 * 16 + 14, 2, d >> 31); /* disp[31:32] */
      deposit(branch, 4 * 16 + 0, 15, d >> 33); /* disp[33:47] */
      return true;
   }

   return false;
}

bool agx3_cf_fcmp_zero(enum agx_isa_encoding enc, enum agx3_cf_fcond cond,
                        bool or_nan, unsigned slot, uint16_t source, bool source_16,
                        struct agx3_cf_source_hints hint, uint16_t wait_mask,
                        uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len)
{
   if (!agx3b(enc) || !out || !len || !fcond_ok(cond) || !slot_ok(slot) ||
       source >= 256 || (!source_16 && (source & 1)) || (hint.cache && hint.discard) ||
       (wait_mask & ~0x3fu) || (or_nan && cond == AGX3_CF_FCOND_EQ))
      return false;
   const struct agx3_field_kv base[]={KV(9,2),KV(10,source_16?1:2),KV(26,1),KV(27,2)};
   struct agx3_field_kv fields[13]={
      KV(2,source),KV(18,0),KV(11,hint.cache),KV(13,hint.discard),
      KV(34,or_nan),KV(35,slot),KV(46,(unsigned)cond+(or_nan?4:0)),
   };
   for(unsigned i=0;i<6;i++)fields[7+i]=(struct agx3_field_kv)KV(38+i,!!(wait_mask&(1u<<i)));
   for(unsigned ep=0;ep<=1;ep++)
      if(agx3_bind_compose(enc,"fcmp",ep,base,4,fields,13,out,len))return true;
   return false;
}
