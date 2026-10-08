/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_falu_mod_core.h"

#include "agx3_binder.h"

#define KV(f, v) {(uint16_t)(f), (uint64_t)(v)}

/* Long-form baselines: the encoding point that carries every modifier, per
 * mnemonic and encoding. Decoded from Apple's witnessed instructions. */
static const struct agx3_field_kv add_b[] = {KV(9, 2),  KV(10, 2), KV(25, 2),
                                             KV(26, 2), KV(29, 1), KV(42, 2),
                                             KV(43, 2), KV(46, 1), KV(57, 1)};
static const struct agx3_field_kv add_c[] = {
   KV(9, 2),  KV(10, 2), KV(25, 2), KV(26, 2), KV(29, 1), KV(42, 2),
   KV(43, 2), KV(46, 1), KV(50, 3), KV(51, 48), KV(59, 1)};
static const struct agx3_field_kv mul_b[] = {
   KV(9, 2),  KV(10, 2), KV(25, 2), KV(26, 2), KV(29, 1), KV(42, 2),
   KV(43, 2), KV(46, 1), KV(60, 2), KV(74, 2), KV(85, 1)};
static const struct agx3_field_kv mul_c[] = {
   KV(9, 2),  KV(10, 2), KV(25, 2), KV(26, 2), KV(29, 1),  KV(42, 2), KV(43, 2),
   KV(46, 1), KV(60, 2), KV(74, 2), KV(78, 3), KV(79, 62), KV(87, 1)};
static const struct agx3_field_kv fma_b[] = {
   KV(9, 2),  KV(10, 2), KV(25, 2), KV(26, 2), KV(29, 1),  KV(42, 2), KV(43, 2),
   KV(46, 1), KV(59, 2), KV(60, 2), KV(63, 1), KV(77, 2), KV(91, 2), KV(102, 1)};
static const struct agx3_field_kv fma_c[] = {
   KV(9, 2),  KV(10, 2), KV(25, 2), KV(26, 2), KV(29, 1),  KV(42, 2),
   KV(43, 2), KV(46, 1), KV(59, 2), KV(60, 2), KV(63, 1),  KV(77, 2),
   KV(91, 2), KV(95, 3), KV(96, 55), KV(104, 1)};

static bool
b(enum agx_isa_encoding e)
{
   return e == AGX_ISA_ENCODING_AGX3B;
}

static unsigned
common_mods(uint16_t dest, uint16_t src0, uint16_t src1,
            const struct agx3_falu_mods *m, struct agx3_field_kv *ov)
{
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(2, dest);
   ov[n++] = (struct agx3_field_kv)KV(35, src0);
   ov[n++] = (struct agx3_field_kv)KV(18, src1);
   ov[n++] = (struct agx3_field_kv)KV(47, m->src0_abs ? 1 : 0);
   ov[n++] = (struct agx3_field_kv)KV(48, m->src0_neg ? 1 : 0);
   ov[n++] = (struct agx3_field_kv)KV(43, m->src0_16 ? 1 : 2);
   ov[n++] = (struct agx3_field_kv)KV(30, m->src1_abs ? 1 : 0);
   ov[n++] = (struct agx3_field_kv)KV(31, m->src1_neg ? 1 : 0);
   ov[n++] = (struct agx3_field_kv)KV(26, m->src1_16 ? 1 : 2);
   ov[n++] = (struct agx3_field_kv)KV(14, m->saturate ? 1 : 0);
   return n;
}

/* A 32-bit operand is a register pair and starts on an even half; a 16-bit
 * one is a single half and may sit on an odd index. */
static bool
slot_ok(uint16_t r, bool narrow)
{
   return r < 256 && (narrow || (r & 1) == 0);
}

static bool
regs_ok(uint16_t a, uint16_t b_, uint16_t c)
{
   return (a & 1) == 0 && a < 256 && (b_ & 1) == 0 && b_ < 256 &&
          (c & 1) == 0 && c < 256;
}

bool
agx3_falu_add_mod(enum agx_isa_encoding enc, uint16_t dest, uint16_t src0,
                  uint16_t src1, const struct agx3_falu_mods *m,
                  uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len)
{
   if (m == NULL || (dest & 1) || dest >= 256 ||
       !slot_ok(src0, m->src0_16) || !slot_ok(src1, m->src1_16) ||
       m->addend_abs || m->addend_neg)
      return false;
   struct agx3_field_kv ov[32];
   unsigned n = common_mods(dest, src0, src1, m, ov);
   return agx3_bind_compose(enc, "f32add", 3, b(enc) ? add_b : add_c,
                            b(enc) ? 9 : 11, ov, n, out, len);
}

/* Keep subtraction and other modified adds fused while preserving both live
 * operands and their producer slots. The native forms grow from4 to10bytes
 * only when modifiers or wait combinations require it. */
bool
agx3_falu_add_mod_runtime(enum agx_isa_encoding enc, uint16_t dest,
                         uint16_t src0, uint16_t src1,
                         const struct agx3_falu_mods *m,
                         const struct agx3_falu_runtime *runtime,
                         uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !m || !runtime || !out || !len ||
       (dest & 1) || dest >= 256 || !slot_ok(src0, m->src0_16) ||
       !slot_ok(src1, m->src1_16) || m->addend_abs || m->addend_neg ||
       runtime->wait_mask > 0x3f)
      return false;
   for (unsigned i = 0; i < 2; ++i)
      if (runtime->cache[i] && runtime->discard[i])
         return false;

   uint8_t best[AGX3_FALU_MOD_MAX_LENGTH], best_size = 0;
   /* Addition is commutative, but the compact form gives the first source
    * the cheap negate modifier. Native subtraction places its negated input
    * there. Swap both modifiers and lifetime hints when this saves bytes. */
   for (unsigned swap = 0; swap < 2; ++swap) {
      struct agx3_falu_mods mods = *m;
      if (swap) {
         mods.src0_abs = m->src1_abs;
         mods.src0_neg = m->src1_neg;
         mods.src0_16 = m->src1_16;
         mods.src1_abs = m->src0_abs;
         mods.src1_neg = m->src0_neg;
         mods.src1_16 = m->src0_16;
      }
      struct agx3_field_kv ov[32];
      unsigned n = common_mods(dest, swap ? src1 : src0, swap ? src0 : src1,
                               &mods, ov);
      ov[n++] = (struct agx3_field_kv)KV(11, runtime->dest_cache);
      ov[n++] = (struct agx3_field_kv)KV(44, runtime->cache[swap]);
      ov[n++] = (struct agx3_field_kv)KV(46, runtime->discard[swap]);
      ov[n++] = (struct agx3_field_kv)KV(27, runtime->cache[1 - swap]);
      ov[n++] = (struct agx3_field_kv)KV(29, runtime->discard[1 - swap]);
      for (unsigned i = 0; i < 6; ++i)
         ov[n++] = (struct agx3_field_kv)KV(52 + i,
                                          (runtime->wait_mask >> i) & 1);
      for (unsigned ep = 0; ep <= 3; ++ep) {
         uint8_t bytes[AGX3_FALU_MOD_MAX_LENGTH], size;
         if (agx3_bind_compose(enc, "f32add", ep, add_b, 9, ov, n, bytes, &size)) {
            if (!best_size || size < best_size) {
               for (unsigned i = 0; i < size; ++i)
                  best[i] = bytes[i];
               best_size = size;
            }
            break;
         }
      }
   }
   if (!best_size)
      return false;
   for (unsigned i = 0; i < best_size; ++i)
      out[i] = best[i];
   *len = best_size;
   return true;
}

/* Apple lowers float-to-half conversion to f32add with a16-bit destination
 * and the exact negative-zero minifloat in the other input. This preserves
 * signed zero and rounds only once; half4 witnesses include odd halves. */
bool
agx3_falu_narrow_f32(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                     bool saturate, const struct agx3_falu_runtime *runtime,
                     uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !runtime || !out || !len ||
       dest >= 256 || src >= 256 || (src & 1) || runtime->wait_mask > 0x3f ||
       (runtime->cache[0] && runtime->discard[0]))
      return false;
   const struct agx3_field_kv base[] = {
      KV(9, 2), KV(10, 1), KV(25, 2), KV(26, 2), KV(42, 1), KV(43, 2),
   };
   struct agx3_field_kv ov[32] = {
      KV(2, dest), KV(18, src), KV(34, 0x80), KV(11, runtime->dest_cache),
      KV(27, runtime->cache[0]), KV(29, runtime->discard[0]), KV(14, saturate),
   };
   unsigned n = 7;
   for (unsigned i = 0; i < 6; ++i)
      ov[n++] = (struct agx3_field_kv)KV(52 + i,
                                       (runtime->wait_mask >> i) & 1);
   for (unsigned ep = 0; ep <= 3; ++ep) {
      uint8_t bytes[AGX3_FALU_MOD_MAX_LENGTH], size;
      if (agx3_bind_compose(enc, "f32add", ep, base, 6, ov, n, bytes, &size)) {
         for (unsigned i = 0; i < size; ++i)
            out[i] = bytes[i];
         *len = size;
         return true;
      }
   }
   return false;
}

/* Half of a minifloat minus a register: f32add with a 16-bit destination,
 * the register negated in the first operand and the minifloat in the other.
 * Apple's half4(1.0 - x) and half4(-x - 1.0) (blend/native/negpack3.py
 * fs_h_onem, fs_h_negm1) differ only in the immediate; with exact negative
 * zero this is the negated float-to-half conversion. */
bool
agx3_falu_narrow_imm_sub_f32(enum agx_isa_encoding enc, uint16_t dest,
                             uint8_t minifloat, uint16_t src,
                             const struct agx3_falu_runtime *runtime,
                             uint8_t out[AGX3_FALU_MOD_MAX_LENGTH],
                             uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !runtime || !out || !len ||
       dest >= 256 || src >= 256 || (src & 1) || runtime->wait_mask > 0x3f ||
       (runtime->cache[0] && runtime->discard[0]))
      return false;
   const struct agx3_field_kv base[] = {
      KV(9, 2), KV(10, 1), KV(25, 1), KV(26, 2), KV(42, 2), KV(43, 2),
      KV(48, 1),
   };
   struct agx3_field_kv ov[16] = {
      KV(2, dest), KV(35, src), KV(17, minifloat),
      KV(11, runtime->dest_cache), KV(44, runtime->cache[0]),
      KV(46, runtime->discard[0]),
   };
   unsigned n = 6;
   for (unsigned i = 0; i < 6; ++i)
      ov[n++] = (struct agx3_field_kv)KV(52 + i,
                                       (runtime->wait_mask >> i) & 1);
   for (unsigned ep = 0; ep <= 3; ++ep) {
      if (agx3_bind_compose(enc, "f32add", ep, base, 7, ov, n, out, len))
         return true;
   }
   return false;
}

bool
agx3_falu_mul_mod(enum agx_isa_encoding enc, uint16_t dest, uint16_t src0,
                  uint16_t src1, const struct agx3_falu_mods *m,
                  uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len)
{
   if (m == NULL || (dest & 1) || dest >= 256 ||
       !slot_ok(src0, m->src0_16) || !slot_ok(src1, m->src1_16) ||
       m->addend_abs || m->addend_neg)
      return false;
   struct agx3_field_kv ov[32];
   unsigned n = common_mods(dest, src0, src1, m, ov);
   return agx3_bind_compose(enc, "f32mul", 3, b(enc) ? mul_b : mul_c,
                            b(enc) ? 11 : 13, ov, n, out, len);
}

/* Saturation and source modifiers stay fused with the multiply. Native
 * live/dead witnesses carry independent hints and the same six dependency
 * slots as plain FMUL; select the shortest encoding that fits them. */
bool
agx3_falu_mul_mod_runtime(enum agx_isa_encoding enc, uint16_t dest,
                          uint16_t src0, uint16_t src1,
                          const struct agx3_falu_mods *m,
                          const struct agx3_falu_runtime *runtime,
                          uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !m || !runtime || !out || !len ||
       (dest & 1) || dest >= 256 || !slot_ok(src0, m->src0_16) ||
       !slot_ok(src1, m->src1_16) || m->addend_abs || m->addend_neg ||
       runtime->wait_mask > 0x3f)
      return false;
   for (unsigned i = 0; i < 2; ++i)
      if (runtime->cache[i] && runtime->discard[i])
         return false;

   struct agx3_field_kv ov[32];
   unsigned n = common_mods(dest, src0, src1, m, ov);
   ov[n++] = (struct agx3_field_kv)KV(11, runtime->dest_cache);
   ov[n++] = (struct agx3_field_kv)KV(44, runtime->cache[0]);
   ov[n++] = (struct agx3_field_kv)KV(46, runtime->discard[0]);
   ov[n++] = (struct agx3_field_kv)KV(27, runtime->cache[1]);
   ov[n++] = (struct agx3_field_kv)KV(29, runtime->discard[1]);
   for (unsigned i = 0; i < 6; ++i)
      ov[n++] = (struct agx3_field_kv)KV(80 + i,
                                       (runtime->wait_mask >> i) & 1);
   for (unsigned ep = 0; ep <= 3; ++ep) {
      uint8_t bytes[AGX3_FALU_MOD_MAX_LENGTH], size;
      if (agx3_bind_compose(enc, "f32mul", ep, mul_b, 11, ov, n, bytes, &size)) {
         for (unsigned i = 0; i < size; ++i)
            out[i] = bytes[i];
         *len = size;
         return true;
      }
   }
   return false;
}

bool
agx3_falu_fma_mod(enum agx_isa_encoding enc, uint16_t dest, uint16_t src0,
                  uint16_t src1, uint16_t addend,
                  const struct agx3_falu_mods *m,
                  uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len)
{
   if (m == NULL || (dest & 1) || dest >= 256 ||
       !slot_ok(src0, m->src0_16) || !slot_ok(src1, m->src1_16) ||
       !slot_ok(addend, m->addend_16))
      return false;
   struct agx3_field_kv ov[32];
   unsigned n = common_mods(dest, src0, src1, m, ov);
   ov[n++] = (struct agx3_field_kv)KV(52, addend);
   ov[n++] = (struct agx3_field_kv)KV(64, m->addend_abs ? 1 : 0);
   ov[n++] = (struct agx3_field_kv)KV(65, m->addend_neg ? 1 : 0);
   ov[n++] = (struct agx3_field_kv)KV(60, m->addend_16 ? 1 : 2);
   return agx3_bind_compose(enc, "f32fma", b(enc) ? 5 : 7, b(enc) ? fma_b : fma_c,
                            b(enc) ? 14 : 16, ov, n, out, len);
}

/* Native dead/live modifier witnesses use the same source cache/discard
 * fields as ordinary FMA. Encoding points 2..5 carry progressively wider
 * modifiers and dependency waits; choose the shortest exact representation.
 * This keeps fog subtraction fused while waiting only on its dependencies.
 */
bool
agx3_falu_fma_mod_runtime(enum agx_isa_encoding enc, uint16_t dest,
                          uint16_t src0, uint16_t src1, uint16_t addend,
                          const struct agx3_falu_mods *m,
                          const struct agx3_falu_runtime *runtime,
                          uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || m == NULL || runtime == NULL ||
       out == NULL || len == NULL || (dest & 1) || dest >= 256 ||
       !slot_ok(src0, m->src0_16) || !slot_ok(src1, m->src1_16) ||
       !slot_ok(addend, m->addend_16) || runtime->wait_mask > 0x3f)
      return false;
   for (unsigned i = 0; i < 3; ++i)
      if (runtime->cache[i] && runtime->discard[i])
         return false;

   struct agx3_field_kv ov[32];
   unsigned n = common_mods(dest, src0, src1, m, ov);
   ov[n++] = (struct agx3_field_kv)KV(52, addend);
   ov[n++] = (struct agx3_field_kv)KV(64, m->addend_abs);
   ov[n++] = (struct agx3_field_kv)KV(65, m->addend_neg);
   ov[n++] = (struct agx3_field_kv)KV(60, m->addend_16 ? 1 : 2);
   ov[n++] = (struct agx3_field_kv)KV(11, runtime->dest_cache);
   ov[n++] = (struct agx3_field_kv)KV(44, runtime->cache[0]);
   ov[n++] = (struct agx3_field_kv)KV(46, runtime->discard[0]);
   ov[n++] = (struct agx3_field_kv)KV(27, runtime->cache[1]);
   ov[n++] = (struct agx3_field_kv)KV(29, runtime->discard[1]);
   ov[n++] = (struct agx3_field_kv)KV(61, runtime->cache[2]);
   ov[n++] = (struct agx3_field_kv)KV(63, runtime->discard[2]);
   for (unsigned i = 0; i < 6; ++i)
      ov[n++] = (struct agx3_field_kv)KV(97 + i,
                                       (runtime->wait_mask >> i) & 1);
   for (unsigned ep = 2; ep <= 5; ++ep) {
      uint8_t bytes[AGX3_FALU_MOD_MAX_LENGTH], size;
      if (agx3_bind_compose(enc, "f32fma", ep, fma_b, 14, ov, n,
                           bytes, &size)) {
         for (unsigned i = 0; i < size; ++i)
            out[i] = bytes[i];
         *len = size;
         return true;
      }
   }
   return false;
}

/* Native FMA has a minifloat addend form: keep a*b+c fused without a
 * temporary constant register, including absolute/negated multipliers, and
 * preserve each multiplier's lifetime. */
bool
agx3_falu_fma_addend_imm_mod(enum agx_isa_encoding enc, uint16_t dest,
                            uint16_t src0, uint16_t src1, uint8_t immediate,
                            bool src0_abs, bool src0_neg, bool src1_abs,
                            bool src1_neg,
                            const struct agx3_falu_runtime *runtime,
                            uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !runtime || !out || !len ||
       !regs_ok(dest, src0, src1) || runtime->wait_mask > 0x3f)
      return false;
   for (unsigned i = 0; i < 2; ++i)
      if (runtime->cache[i] && runtime->discard[i])
         return false;
   const struct agx3_field_kv base[] = {
      KV(9,2), KV(10,2), KV(25,2), KV(26,2), KV(42,2), KV(43,2),
      KV(59,1), KV(60,2), KV(77,2), KV(91,2), KV(102,1),
   };
   struct agx3_field_kv ov[32] = {
      KV(2,dest), KV(35,src0), KV(18,src1), KV(51,immediate),
      KV(11,runtime->dest_cache), KV(44,runtime->cache[0]),
      KV(46,runtime->discard[0]), KV(27,runtime->cache[1]),
      KV(29,runtime->discard[1]), KV(47,src0_abs), KV(48,src0_neg),
      KV(30,src1_abs), KV(31,src1_neg),
   };
   unsigned n = 13;
   for (unsigned i = 0; i < 6; ++i)
      if (runtime->wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(97+i,1);
   for (unsigned ep = 0; ep <= 5; ++ep) {
      if (agx3_bind_compose(enc,"f32fma",ep,base,11,ov,n,out,len))
         return true;
   }
   return false;
}

/* The immediate multiplier and addend occupy independent native fields.
 * This is still one fused operation with one rounding, e.g. depth*2-1. */
bool
agx3_falu_fma_two_imm(enum agx_isa_encoding enc, uint16_t dest,
                     uint16_t src, uint8_t multiplier, uint8_t addend,
                     const struct agx3_falu_runtime *runtime,
                     uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !runtime || !out || !len ||
       !regs_ok(dest, src, 0) || runtime->wait_mask > 0x3f ||
       (runtime->cache[0] && runtime->discard[0]))
      return false;
   const struct agx3_field_kv base[] = {
      KV(9,2), KV(10,2), KV(25,2), KV(26,2), KV(42,1), KV(43,2),
      KV(59,1), KV(60,2), KV(77,2), KV(91,2),
   };
   struct agx3_field_kv ov[32] = {
      KV(2,dest), KV(18,src), KV(34,multiplier), KV(51,addend),
      KV(11,runtime->dest_cache), KV(27,runtime->cache[0]),
      KV(29,runtime->discard[0]),
   };
   unsigned n = 7;
   for (unsigned i = 0; i < 6; ++i)
      if (runtime->wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(97+i,1);
   for (unsigned ep = 0; ep <= 5; ++ep)
      if (agx3_bind_compose(enc,"f32fma",ep,base,10,ov,n,out,len))
         return true;
   return false;
}

/* Immediate multiplier (f34, selected by f42=1) with a modifier-bearing
 * register multiplicand (f18 + f26/f30/f31) and addend (f52 + f60/f64/f65).
 * Baseline is the constant part of Apple's witnessed fma(0.5, b, c); every
 * operand, modifier, lifetime hint and wait bit is an override, so a field
 * the selected encoding point does not consume makes the composer refuse
 * rather than silently drop it. */
bool
agx3_falu_fma_mul_imm_mod(enum agx_isa_encoding enc, uint16_t dest,
                          uint8_t multiplier, uint16_t src1, uint16_t addend,
                          const struct agx3_falu_mods *m,
                          const struct agx3_falu_runtime *runtime,
                          uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || m == NULL || runtime == NULL ||
       out == NULL || len == NULL || (dest & 1) || dest >= 256 ||
       !slot_ok(src1, m->src1_16) || !slot_ok(addend, m->addend_16) ||
       runtime->wait_mask > 0x3f)
      return false;

   /* There is no f35 register slot in this form, so nothing can carry a
    * multiplier-side modifier or lifetime hint; a caller that has one has not
    * folded it into the immediate and must not be silently obeyed.
    *
    * f31 -- the multiplicand's negate -- is refused for a different reason:
    * it is not witnessed here. Apple never emits it beside an immediate
    * multiplier, folding the sign into the minifloat instead
    * (fma(0.5,-fabs(b),c) is f34=160 with f30, no f31), which is exact
    * because k * -b == -k * b. The caller folds it the same way. */
   if (m->src0_abs || m->src0_neg || m->src0_16 || m->src1_neg ||
       runtime->cache[0] || runtime->discard[0])
      return false;

   for (unsigned i = 1; i < 3; ++i)
      if (runtime->cache[i] && runtime->discard[i])
         return false;

   const struct agx3_field_kv base[] = {
      KV(9, 2), KV(10, 2), KV(25, 2), KV(26, 2), KV(42, 1),
      KV(43, 2), KV(59, 2), KV(60, 2), KV(77, 2), KV(91, 2),
   };
   /* Appended rather than counted: a designated-initialiser list with a
    * hand-written length silently grows a zeroed {field 0, value 0} entry the
    * moment an override is removed. */
   struct agx3_field_kv ov[32];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(2, dest);
   ov[n++] = (struct agx3_field_kv)KV(34, multiplier);
   ov[n++] = (struct agx3_field_kv)KV(18, src1);
   ov[n++] = (struct agx3_field_kv)KV(52, addend);
   ov[n++] = (struct agx3_field_kv)KV(26, m->src1_16 ? 1 : 2);
   ov[n++] = (struct agx3_field_kv)KV(30, m->src1_abs ? 1 : 0);
   ov[n++] = (struct agx3_field_kv)KV(60, m->addend_16 ? 1 : 2);
   ov[n++] = (struct agx3_field_kv)KV(64, m->addend_abs ? 1 : 0);
   ov[n++] = (struct agx3_field_kv)KV(65, m->addend_neg ? 1 : 0);
   ov[n++] = (struct agx3_field_kv)KV(14, m->saturate ? 1 : 0);
   ov[n++] = (struct agx3_field_kv)KV(11, runtime->dest_cache);
   ov[n++] = (struct agx3_field_kv)KV(27, runtime->cache[1]);
   ov[n++] = (struct agx3_field_kv)KV(29, runtime->discard[1]);
   ov[n++] = (struct agx3_field_kv)KV(61, runtime->cache[2]);
   ov[n++] = (struct agx3_field_kv)KV(63, runtime->discard[2]);
   for (unsigned i = 0; i < 6; ++i)
      ov[n++] = (struct agx3_field_kv)KV(97 + i, (runtime->wait_mask >> i) & 1);

   for (unsigned ep = 0; ep <= 5; ++ep) {
      if (agx3_bind_compose(enc, "f32fma", ep, base, 10, ov, n, out, len))
         return true;
   }
   return false;
}
