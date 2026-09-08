/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "agx3_alu_core.h"
#include "agx3_binder.h"

#include <string.h>

bool
agx3_alu_popcount(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                  bool half, enum agx3_alu_hint hint, bool dest_cache,
                  uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                  uint8_t *len)
{
   /* Exact Apple G17P witnesses:
    *   ushort: 27 05 56 02 02 04 50 04
    *   uint:   27 05 56 00 02 00 5c 04
    * byte3 is the destination half, byte5 is twice the source half, and the
    * source descriptor distinguishes 16 from 32 bits.  Both witnesses consume
    * their source.  Refuse unwitnessed cache/wait/lifetime combinations. */
   if (enc != AGX_ISA_ENCODING_AGX3B || !out || !len || dest >= 256 ||
       src >= 128 || (!half && ((dest | src) & 1)) ||
       hint != AGX3_ALU_HINT_DISCARD || wait_mask)
      return false;

   const uint8_t bytes[8] = {
      0x27, 0x05, 0x56, (uint8_t)dest, dest_cache ? 0x03 : 0x02,
      (uint8_t)(src << 1),
      half ? 0x50 : 0x5c, 0x04,
   };
   memcpy(out, bytes, sizeof(bytes));
   *len = sizeof(bytes);
   return true;
}

bool
agx3_alu_ft1_u32(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                 enum agx3_alu_hint hint, bool dest_cache,
                 uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                 uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !out || !len || dest >= 256 ||
       src >= 256 || ((dest | src) & 1) ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE ||
       (wait_mask & ~UINT16_C(0x3f)))
      return false;

   /* Direct llvm.agx2.ft1 witness:
    *   a705560202004c04  ft1 r1, wait0
    * A live ALU chain independently grounds destination/source caching:
    *   a705540003005c04  ft1 $r0, $r0
    * Apple's clz lowering grounds independent destination/source plus wait0:
    *   a705560603044e04  ft1 $r3, ^r2, wait0 */
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 2}, {23, 2}, {24, 2},
   };
   struct agx3_field_kv fields[11] = {
      {2, dest},
      {11, dest_cache},
      {25, hint == AGX3_ALU_HINT_DISCARD},
      {27, hint == AGX3_ALU_HINT_CACHE},
   };
   unsigned n = 4;
   for (unsigned slot = 0; slot < 6; ++slot)
      fields[n++] = (struct agx3_field_kv){36 + slot,
                                          !!(wait_mask & (1u << slot))};

   if (!agx3_bind_compose(enc, "ft1", 0, base, 4, fields, n, out, len))
      return false;

   /* The recovered composer form fixes the independent source byte to zero,
    * but fresh Apple clz kernels vary it directly with the source half:
    * a705560603004e04 (src half0) -> a705560603044e04 (src half4).
    * The full byte is the source field and every admitted register is even. */
   out[5] = (uint8_t)src;
   return true;
}

bool
agx3_alu_sqrt_step(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                  enum agx3_alu_hint hint, bool dest_cache, uint16_t wait_mask,
                  uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !out || !len || dest >= 256 ||
       src >= 256 || ((dest | src) & 1) ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE || (wait_mask & ~UINT16_C(0x3f)))
      return false;
   /* Apple sqrt emits this step followed by FMUL with the original input.
    * Ordinary rsqrt is a distinct opcode and cannot replace this operation. */
   const struct agx3_field_kv base[] = {{9,2},{10,2},{25,2},{26,2}};
   struct agx3_field_kv fields[11] = {
      {2,dest},{18,src},{11,dest_cache},
      {27,hint == AGX3_ALU_HINT_CACHE},{29,hint == AGX3_ALU_HINT_DISCARD},
   };
   for (unsigned i = 0; i < 6; ++i)
      fields[5+i] = (struct agx3_field_kv){35+i,!!(wait_mask & (1u<<i))};
   return agx3_bind_compose(enc,"f32sqrtstep",0,base,4,fields,11,out,len);
}

bool
agx3_alu_sqrt_step_half(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                        enum agx3_alu_hint hint, bool dest_cache,
                        uint16_t wait_mask,
                        uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   /* Apple emits the half-input square-root step only with its float32
    * temporary cached, its half retained for the following multiply, and
    * either no dependency or the observed slot-6 load dependency. */
   if (enc != AGX_ISA_ENCODING_AGX3B || !out || !len || dest >= 256 ||
       src >= 256 || (dest & 1) || hint != AGX3_ALU_HINT_CACHE ||
       !dest_cache || (wait_mask != 0 && wait_mask != 32))
      return false;

   const struct agx3_field_kv base[] = {{9,2},{10,2},{25,2},{26,1}};
   const struct agx3_field_kv fields[] = {
      {2,dest}, {18,src}, {11,1}, {27,1}, {40,wait_mask == 32},
   };
   return agx3_bind_compose(enc, "f32sqrtstep", 0, base, 4, fields, 5,
                            out, len);
}

bool
agx3_alu_fmul_minifloat(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                       uint16_t immediate, enum agx3_alu_hint hint, uint16_t wait_mask,
                       uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len || enc != AGX_ISA_ENCODING_AGX3B || dest >= 256 ||
       src >= 256 || ((dest | src) & 1) || immediate > 0xff ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE ||
       (wait_mask & ~UINT16_C(0x3f)))
      return false;

   /* Fresh Apple mul_scale: 0983150180c0 is loaded f32 * minifloat0x02
    * (1/32), wait6. mul_reuse: 0983050380c0 retains its half2 source.
    * The ordinary multiply's additional fixed fields60/74 are both2;
    * it shares add's operand routing, but its wait fields are80..85.
    */
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 2}, {25, 1}, {26, 2},
      {42, 2}, {43, 2}, {60, 2}, {74, 2},
   };
   struct agx3_field_kv fields[11] = {
      {2, dest}, {35, src}, {17, immediate}, {46, hint == AGX3_ALU_HINT_DISCARD},
      {44, hint == AGX3_ALU_HINT_CACHE},
   };
   unsigned n = 5;
   for (unsigned slot = 0; slot < 6; ++slot)
      if (wait_mask & (1u << slot))
         fields[n++] = (struct agx3_field_kv){80 + slot, 1};

   return agx3_bind_compose(enc, "f32mul", 1, base, 8, fields, n, out, len);
}

bool
agx3_alu_fadd_minifloat(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                       uint16_t immediate, enum agx3_alu_hint hint, uint16_t wait_mask,
                       uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len || enc != AGX_ISA_ENCODING_AGX3B || dest >= 256 ||
       src >= 256 || ((dest | src) & 1) || immediate > 0xff ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE ||
       (wait_mask & ~UINT16_C(0x3f)))
      return false;

   /* Apple add_half: 09a1140180c0 (f32 register + minifloat0x20, wait6).
    * add_reuse: 09a1040380c0 keeps its half2 source for a later store.
    * Fragment position uses the same add without a wait. f25=1 selects the
    * minifloat; f42=2 selects the GPR. The register is field35, not field18.
    */
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 2}, {25, 1}, {26, 2}, {42, 2}, {43, 2},
   };
   struct agx3_field_kv fields[11] = {
      {2, dest}, {35, src}, {17, immediate}, {46, hint == AGX3_ALU_HINT_DISCARD},
      {44, hint == AGX3_ALU_HINT_CACHE},
   };
   unsigned n = 5;
   for (unsigned slot = 0; slot < 6; ++slot)
      if (wait_mask & (1u << slot))
         fields[n++] = (struct agx3_field_kv){52 + slot, 1};

   /* This encoding point rejects unsupported wait combinations. Do not
    * silently clear waits or switch to an unwitnessed long-form variant.
    */
   return agx3_bind_compose(enc, "f32add", 1, base, 6, fields, n, out, len);
}

bool
agx3_alu_fadd_minifloat_mod(enum agx_isa_encoding enc, uint16_t dest,
                           uint16_t src, uint16_t immediate, bool negate_src,
                           bool saturate, enum agx3_alu_hint hint,
                           uint16_t wait_mask,
                           uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len || enc != AGX_ISA_ENCODING_AGX3B || dest >= 256 ||
       src >= 256 || ((dest | src) & 1) || immediate > 0xff ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE ||
       (wait_mask & ~UINT16_C(0x3f)))
      return false;

   /* Apple sat_only: 09011c8101020082, sat(r0 + -0), wait6.
    * Native puts this -0 minifloat in field34 and its GPR in field18.
    * Apple complement: 09b1140180c8, -r0 + 1, discard/wait6.
    * sat_add and sat_neg retain that field35/field17 routing, extending
    * the instruction to ep2 for saturation and independent wait bits.
    * Commuting the add's two unmodified operands retains signed-zero
    * semantics; do not commute a negation away from its source register.
    */
   const bool swap = saturate && immediate == 0x80 && !negate_src;
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 2}, {25, swap ? 2 : 1}, {26, 2},
      {42, swap ? 1 : 2}, {43, 2},
   };
   struct agx3_field_kv fields[13] = {
      {2, dest}, {swap ? 18 : 35, src}, {swap ? 34 : 17, immediate},
      {swap ? 27 : 44, hint == AGX3_ALU_HINT_CACHE},
      {swap ? 29 : 46, hint == AGX3_ALU_HINT_DISCARD},
      {swap ? 31 : 48, negate_src}, {14, saturate},
   };
   unsigned n = 7;
   for (unsigned slot = 0; slot < 6; ++slot)
      if (wait_mask & (1u << slot))
         fields[n++] = (struct agx3_field_kv){52 + slot, 1};

   return agx3_bind_compose(enc, "f32add", saturate ? 2 : 1,
                            base, 6, fields, n, out, len);
}

bool
agx3_alu_fadd_one_absneg(enum agx_isa_encoding enc, uint16_t dest,
                         uint16_t src, bool dest_cache,
                         enum agx3_alu_hint hint, uint16_t wait_mask,
                         uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len || enc != AGX_ISA_ENCODING_AGX3B || dest >= 256 ||
       src >= 256 || ((dest | src) & 1) ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE || (wait_mask & ~0x3f))
      return false;

   /* Native live/discard variants are 19b12481820800000200 and
    * 09b13401820800000200. f25 selects the minifloat input, while f47/f48
    * retain abs+neg on the GPR and f11 retains a partially live destination.
    */
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 2}, {25, 1}, {26, 2}, {42, 2}, {43, 2},
   };
   struct agx3_field_kv fields[20] = {
      {2, dest}, {35, src}, {17, 0x30}, {11, dest_cache}, {47, 1}, {48, 1},
      {44, hint == AGX3_ALU_HINT_CACHE},
      {46, hint == AGX3_ALU_HINT_DISCARD},
   };
   unsigned n = 8;
   for (unsigned slot = 0; slot < 6; ++slot)
      fields[n++] = (struct agx3_field_kv){52 + slot,
                                          (wait_mask >> slot) & 1};

   for (unsigned ep = 0; ep <= 3; ++ep) {
      if (agx3_bind_compose(enc, "f32add", ep, base, 6, fields, n, out, len))
         return true;
   }

   return false;
}

bool
agx3_alu_fadd_abs_negzero(enum agx_isa_encoding enc, uint16_t dest,
                          uint16_t src, bool dest_cache,
                          enum agx3_alu_hint hint, uint16_t wait_mask,
                          uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len || enc != AGX_ISA_ENCODING_AGX3B || dest >= 256 ||
       src >= 256 || ((dest | src) & 1) ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE || (wait_mask & ~0x3f))
      return false;

   /* This is the exact fmov/fabs form: |src| + -0. f17 selects negative
    * zero, while f47 is the independently grounded register absolute bit. */
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 2}, {25, 1}, {26, 2}, {42, 2}, {43, 2},
   };
   struct agx3_field_kv fields[20] = {
      {2, dest}, {35, src}, {17, 0x80}, {11, dest_cache}, {47, 1}, {48, 0},
      {44, hint == AGX3_ALU_HINT_CACHE},
      {46, hint == AGX3_ALU_HINT_DISCARD},
   };
   unsigned n = 8;
   for (unsigned slot = 0; slot < 6; ++slot)
      fields[n++] = (struct agx3_field_kv){52 + slot,
                                          (wait_mask >> slot) & 1};

   for (unsigned ep = 0; ep <= 3; ++ep) {
      if (agx3_bind_compose(enc, "f32add", ep, base, 6, fields, n, out, len))
         return true;
   }

   return false;
}

bool
agx3_alu_float_reg(enum agx_isa_encoding enc, enum agx3_alu_op op,
                     uint16_t dest, const uint16_t src[3],
                     const enum agx3_alu_hint hints[3], uint16_t wait_mask,
                     uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !src || !hints || !out || !len ||
       (op != AGX3_ALU_FADD && op != AGX3_ALU_FMUL && op != AGX3_ALU_FMA) ||
       (wait_mask & ~UINT16_C(0x3f)))
      return false;
   const unsigned count = op == AGX3_ALU_FMA ? 3 : 2;
   /* Native FADD/FMUL/FMA all encode eight-bit half-register operands.
    * TEX alone has a separate seven-bit destination restriction. */
   const unsigned limit = 256;
   if (dest >= limit || (dest & 1))
      return false;
   for (unsigned i = 0; i < count; ++i)
      if (src[i] >= limit || (src[i] & 1) ||
          (unsigned)hints[i] > AGX3_ALU_HINT_CACHE)
         return false;

   /* Retain the qualified operand classes, not a specimen's liveness or
    * unconditional waits. In Mesa order src0/src1/src2 map to f35/f18/f52.
    * Fresh add_live/mul_live/fma_live witnesses retain all their operands;
    * all_ops additionally witnesses independently cached/discarded sources.
    */
   const struct agx3_alu_form *f = &agx3_alu_forms[op][0];
   struct agx3_field_kv base[20], ov[16];
   unsigned nb = 0, n = 0;
   for (unsigned i = 0; i < f->nbase; ++i) {
      unsigned field = f->base[i].f;
      bool wait_field = false;
      for (unsigned w = 0; w < f->nwait; ++w)
         wait_field |= field == f->wait[w];
      if (field != 29 && field != 46 && field != 63 && !wait_field)
         base[nb++] = (struct agx3_field_kv){field, f->base[i].v};
   }
   ov[n++] = (struct agx3_field_kv){2, dest};
   const unsigned operands[] = {35, 18, 52};
   const unsigned cache[] = {44, 27, 61};
   const unsigned discard[] = {46, 29, 63};
   for (unsigned i = 0; i < count; ++i) {
      ov[n++] = (struct agx3_field_kv){operands[i], src[i]};
      ov[n++] = (struct agx3_field_kv){cache[i], hints[i] == AGX3_ALU_HINT_CACHE};
      ov[n++] = (struct agx3_field_kv){discard[i], hints[i] == AGX3_ALU_HINT_DISCARD};
   }
   for (unsigned i = 0; i < f->nwait; ++i)
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv){f->wait[i], 1};

   const unsigned first = op == AGX3_ALU_FMA ? 2 : 0;
   const unsigned last = op == AGX3_ALU_FMA ? 3 : 1;
   for (unsigned ep = first; ep <= last; ++ep)
      if (agx3_bind_compose(enc, f->mnemonic, ep, base, nb, ov, n, out, len))
         return true;
   return false;
}

bool
agx3_alu_fma_minifloat_mod(enum agx_isa_encoding enc, uint16_t dest,
                           uint16_t src, uint8_t immediate, uint16_t addend,
                           bool negate_addend, enum agx3_alu_hint src_hint,
                           enum agx3_alu_hint addend_hint, bool dest_cache,
                           uint16_t wait_mask,
                           uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len || enc != AGX_ISA_ENCODING_AGX3B ||
       dest >= 256 || src >= 256 || addend >= 256 ||
       ((dest | src | addend) & 1) || (wait_mask & ~UINT16_C(0x3f)) ||
       (unsigned)src_hint > AGX3_ALU_HINT_CACHE ||
       (unsigned)addend_hint > AGX3_ALU_HINT_CACHE)
      return false;
   const struct agx3_field_kv base[] = {
      {9,2},{10,2},{25,2},{26,2},{42,1},{43,2},{59,2},{60,2},{77,2},{91,2},
      {102,1},
   };
   struct agx3_field_kv fields[17] = {
      {2,dest},{11,dest_cache},{18,src},{34,immediate},{52,addend},
      {27,src_hint==AGX3_ALU_HINT_CACHE},{29,src_hint==AGX3_ALU_HINT_DISCARD},
      {61,addend_hint==AGX3_ALU_HINT_CACHE},{63,addend_hint==AGX3_ALU_HINT_DISCARD},
      {65,negate_addend},
   };
   unsigned n=10;
   for(unsigned slot=0;slot<6;slot++)
      if(wait_mask & (1u<<slot))fields[n++]=(struct agx3_field_kv){97+slot,1};
   for(unsigned ep=2;ep<=4;ep++)
      if(agx3_bind_compose(enc,"f32fma",ep,base,negate_addend ? 11 : 10,
                           fields,n,out,len))
         return true;
   return false;
}

bool
agx3_alu_fma_minifloat(enum agx_isa_encoding enc, uint16_t dest,
                       uint16_t src, uint8_t immediate, uint16_t addend,
                       enum agx3_alu_hint src_hint,
                       enum agx3_alu_hint addend_hint, bool dest_cache,
                       uint16_t wait_mask,
                       uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   return agx3_alu_fma_minifloat_mod(enc, dest, src, immediate, addend, false,
                                     src_hint, addend_hint, dest_cache,
                                     wait_mask, out, len);
}

static bool
agx3_alu_int16_to_f32(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                    enum agx3_alu_hint hint, bool is_signed, uint16_t wait_mask,
                    uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len || enc != AGX_ISA_ENCODING_AGX3B || dest >= 128 ||
       (dest & 1) || src >= 128 || (unsigned)hint > AGX3_ALU_HINT_CACHE ||
       (wait_mask & ~UINT16_C(0x3f)))
      return false;

   /* Apple ushort_cs: a70756000200a420 (r0f <- r0h, wait slot6).
    * ushort_pair_cs additionally witnesses odd source half3 and no wait;
    * ushort_reuse_cs has f29=0 with a later16-bit store of the same source.
    * Versus U32 int2flt only f26 changes2->1. The fresh short_float and
    * short_float_live witnesses establish signedness in f30.
    */
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 2}, {25, 2}, {26, 1},
   };
   struct agx3_field_kv fields[11] = {
      {2, dest}, {18, src}, {29, hint == AGX3_ALU_HINT_DISCARD},
      {27, hint == AGX3_ALU_HINT_CACHE}, {30, is_signed},
   };
   unsigned n = 5;
   for (unsigned slot = 0; slot < 6; ++slot)
      if (wait_mask & (1u << slot))
         fields[n++] = (struct agx3_field_kv){33 + slot, 1};
   return agx3_bind_compose(enc, "int2flt", 0, base, 4, fields, n, out, len);
}

bool
agx3_alu_u16_to_f32(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                     enum agx3_alu_hint hint, uint16_t wait_mask,
                     uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   return agx3_alu_int16_to_f32(enc,dest,src,hint,false,wait_mask,out,len);
}

bool
agx3_alu_s16_to_f32(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                     enum agx3_alu_hint hint, uint16_t wait_mask,
                     uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   return agx3_alu_int16_to_f32(enc,dest,src,hint,true,wait_mask,out,len);
}

bool
agx3_alu_f32_to_s16(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                    enum agx3_alu_hint hint, bool dest_cache,
                    uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                    uint8_t *len)
{
   if (!out || !len || enc != AGX_ISA_ENCODING_AGX3B || dest >= 256 ||
       src >= 128 || (src & 1) || (unsigned)hint > AGX3_ALU_HINT_CACHE ||
       (wait_mask != 0 && wait_mask != 32))
      return false;

   /* Fresh G17P Metal kernels ground the RTZ float-to-short form with odd and
    * even half destinations, sources 0 and 2, both destination-cache states,
    * preserved and discarded sources, and no wait versus slot 6. A kernel
    * stores the original float after conversion, proving 0x90 preserves it. */
   const uint8_t bytes[] = {
      0x27, 0x07, wait_mask ? 0x56 : 0x54, (uint8_t)dest,
      dest_cache ? 0x03 : 0x02, (uint8_t)(src << 1),
      hint == AGX3_ALU_HINT_DISCARD ? 0xb0 : 0x90,
      0xc8, 0x02, 0x00,
   };
   memcpy(out, bytes, sizeof(bytes));
   *len = sizeof(bytes);
   return true;
}

bool
agx3_alu_add_imm16(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                    uint16_t immediate, enum agx3_alu_hint hint, bool dest_cache,
                    uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !out || !len || dest >= 256 ||
       src >= 256 || immediate > 255 || (unsigned)hint > AGX3_ALU_HINT_CACHE ||
       (wait_mask & ~0x3fu))
      return false;
   /* Native short +2 and ushort2 +(2,17): result and register operand
    * are individual halves, with the immediate in slot A. Addition wraps
    * at16 bits without widening or allocating a temporary register. */
   const struct agx3_field_kv base[] = {
      {9,2}, {10,1}, {23,1}, {24,1}, {38,2}, {39,1},
   };
   struct agx3_field_kv fields[12] = {
      {2,dest}, {11,dest_cache}, {15,immediate}, {31,src},
      {40,hint == AGX3_ALU_HINT_CACHE}, {42,hint == AGX3_ALU_HINT_DISCARD},
   };
   unsigned n=6;
   for(unsigned i=0;i<6;++i)
      if(wait_mask & (1u<<i))fields[n++]=(struct agx3_field_kv){45+i,1};
   return agx3_bind_compose(enc,"iadd",0,base,6,fields,n,out,len);
}

bool
agx3_alu_sub_imm16(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                   uint16_t immediate, enum agx3_alu_hint hint,
                   bool dest_cache, uint16_t wait_mask,
                   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len || enc != AGX_ISA_ENCODING_AGX3B || dest >= 256 ||
       src >= 128 || immediate == 0 || immediate > 255 ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE ||
       (wait_mask != 0 && wait_mask != 32))
      return false;

   /* Apple G17P emits a distinct ten-byte subtract-immediate form rather
    * than negating the add form. Controlled Metal kernels witness immediate
    * values 1, 2, 17 and 255; half destinations 1, 2 and 3; half sources 2,
    * 3 and 4; both destination-cache states; both source lifetime hints; and
    * no wait versus wait slot 6. The immediate is an unsigned byte scaled by
    * four in bytes 6..7, exactly matching modulo-16-bit subtraction. */
   const uint16_t encoded_immediate = immediate << 2;
   const uint8_t bytes[] = {
      0x1f, 0x01, wait_mask ? 0x56 : 0x54, (uint8_t)dest,
      dest_cache ? 0x03 : 0x02, (uint8_t)(src << 1),
      (uint8_t)encoded_immediate, (uint8_t)(encoded_immediate >> 8),
      hint == AGX3_ALU_HINT_DISCARD ? 0x13 : 0x11, 0x01,
   };
   memcpy(out, bytes, sizeof(bytes));
   *len = sizeof(bytes);
   return true;
}

/* Exact native G17P witnesses include 0x1800 (1/512), adjacent literals,
 * odd destination halves and mixed multiplies with both source hints.
 * LDIMM's nonzero-wait form has not been witnessed and remains refused. */
bool
agx3_alu_mov16(enum agx_isa_encoding enc, uint16_t dest, uint32_t bits,
               uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len || enc != AGX_ISA_ENCODING_AGX3B || dest >= 256 ||
       bits > 0xffff || wait_mask)
      return false;
   const struct agx3_field_kv fields[] = {{9, 2}, {10, 1}, {2, dest}, {14, bits}};
   return agx3_bind_compose(enc, "ldimm", 1, fields, 4, NULL, 0, out, len);
}

bool
agx3_alu_fmul_mixed(enum agx_isa_encoding enc, uint16_t dest,
                    uint16_t f32, uint16_t f16,
                    enum agx3_alu_hint hint32, enum agx3_alu_hint hint16,
                    uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                    uint8_t *len)
{
   if (!out || !len || enc != AGX_ISA_ENCODING_AGX3B || dest >= 256 ||
       f32 >= 256 || f16 >= 256 || ((dest | f32) & 1) ||
       (unsigned)hint32 > AGX3_ALU_HINT_CACHE ||
       (unsigned)hint16 > AGX3_ALU_HINT_CACHE || (wait_mask & ~0x3f))
      return false;
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 2}, {25, 2}, {26, 2}, {42, 2}, {43, 1}, {60, 2}, {74, 2},
   };
   struct agx3_field_kv fields[13] = {
      {2, dest}, {18, f32}, {35, f16},
      {27, hint32 == AGX3_ALU_HINT_CACHE},
      {29, hint32 == AGX3_ALU_HINT_DISCARD},
      {44, hint16 == AGX3_ALU_HINT_CACHE},
      {46, hint16 == AGX3_ALU_HINT_DISCARD},
   };
   unsigned n = 7;
   for (unsigned bit = 0; bit < 6; ++bit)
      if (wait_mask & (1u << bit))
         fields[n++] = (struct agx3_field_kv){80 + bit, 1};
   return agx3_bind_compose(enc, "f32mul", 0, base, 8, fields, n, out, len) ||
          agx3_bind_compose(enc, "f32mul", 1, base, 8, fields, n, out, len);
}

bool
agx3_alu_fadd_abs_mixed_half(
   enum agx_isa_encoding enc, uint16_t dest, uint16_t f32, uint16_t f16,
   enum agx3_alu_hint hint32, enum agx3_alu_hint hint16, bool dest_cache,
   uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !out || !len || dest >= 256 ||
       f32 >= 256 || f16 >= 256 || ((dest | f32) & 1) ||
       (unsigned)hint32 > AGX3_ALU_HINT_CACHE ||
       (unsigned)hint16 > AGX3_ALU_HINT_CACHE ||
       (wait_mask & ~UINT16_C(0x3f)))
      return false;

   /* Fresh G17P Metal output uses f32add.ep3 directly: the first source is
    * absolute f32, while the second retains its physical half register. */
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 2}, {25, 2}, {26, 2}, {30, 1}, {42, 2}, {43, 1},
   };
   struct agx3_field_kv fields[15] = {
      {2, dest},
      {11, dest_cache},
      {18, f32},
      {35, f16},
      {27, hint32 == AGX3_ALU_HINT_CACHE},
      {29, hint32 == AGX3_ALU_HINT_DISCARD},
      {44, hint16 == AGX3_ALU_HINT_CACHE},
      {46, hint16 == AGX3_ALU_HINT_DISCARD},
   };
   for (unsigned i = 0; i < 6; ++i)
      fields[8 + i] = (struct agx3_field_kv){52 + i,
                                            !!(wait_mask & (1u << i))};

   return agx3_bind_compose(enc, "f32add", 3, base, 7, fields, 14, out,
                            len);
}

bool
agx3_alu_fmul_mixed_half(enum agx_isa_encoding enc, uint16_t dest,
                         uint16_t f32, uint16_t f16,
                         enum agx3_alu_hint hint32,
                         enum agx3_alu_hint hint16, bool dest_cache,
                         uint16_t wait_mask,
                         uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   /* This is the exact second half of Apple's scalar half sqrt sequence.
    * Field10 selects the half destination, including odd half registers.
    * The f32 step is consumed, while the original half may remain live. The
    * sampled-texture form uses the native f32mul long endpoint's slot-1 wait. */
   if (!out || !len || enc != AGX_ISA_ENCODING_AGX3B || dest >= 256 ||
       f32 >= 256 || f16 >= 256 || (f32 & 1) ||
       hint32 != AGX3_ALU_HINT_DISCARD ||
       (hint16 != AGX3_ALU_HINT_NONE && hint16 != AGX3_ALU_HINT_DISCARD) ||
       wait_mask > 1)
      return false;

   const struct agx3_field_kv base[] = {
      {9,2},{10,1},{25,2},{26,2},{42,2},{43,1},{60,2},{74,2},
   };
   struct agx3_field_kv fields[7] = {
      {2,dest}, {18,f32}, {35,f16}, {11,dest_cache}, {29,1},
      {46,hint16 == AGX3_ALU_HINT_DISCARD},
   };
   unsigned n = 6;
   if (wait_mask)
      fields[n++] = (struct agx3_field_kv){80, 1};
   for (unsigned ep = 0; ep < 2; ++ep)
      if (agx3_bind_compose(enc, "f32mul", ep, base, 8, fields, n,
                            out, len))
         return true;
   return false;
}

bool
agx3_alu_wait_supported(enum agx3_alu_op op)
{
   /* fmin/fmax use a tied-operand fsel form whose duplicated compare fields
    * this table does not carry; exclude them. */
   return op < AGX3_ALU_NUM && op != AGX3_ALU_FMIN && op != AGX3_ALU_FMAX;
}

bool
agx3_alu_compose(enum agx_isa_encoding enc, enum agx3_alu_op op, uint16_t dest,
                 uint16_t src0, uint16_t src1, uint16_t src2, uint32_t imm,
                 uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                 uint8_t *len)
{
   if (op >= AGX3_ALU_NUM)
      return false;
   const unsigned ei = (enc == AGX_ISA_ENCODING_AGX3C) ? 1
                       : (enc == AGX_ISA_ENCODING_AGX3B) ? 0
                                                         : 2;
   if (ei == 2)
      return false;
   const struct agx3_alu_form *F = &agx3_alu_forms[op][ei];
   if (F->mnemonic == 0)
      return false;
   if (wait_mask && !agx3_alu_wait_supported(op))
      return false;

   struct agx3_field_kv base[20];
   for (unsigned i = 0; i < F->nbase; ++i) {
      base[i].field = F->base[i].f;
      base[i].value = F->base[i].v;
   }

   struct agx3_field_kv ov[24];
   unsigned n = 0;
   if (F->f_dest >= 0) { ov[n].field = F->f_dest; ov[n].value = dest; n++; }
   if (F->f_src0 >= 0) { ov[n].field = F->f_src0; ov[n].value = src0; n++; }
   if (F->f_src1 >= 0) { ov[n].field = F->f_src1; ov[n].value = src1; n++; }
   if (F->f_src2 >= 0) { ov[n].field = F->f_src2; ov[n].value = src2; n++; }
   if (F->f_imm >= 0)  { ov[n].field = F->f_imm;  ov[n].value = imm;  n++; }
   for (unsigned i = 0; i < F->nwait; ++i)
      if (wait_mask & (1u << i)) { ov[n].field = F->wait[i]; ov[n].value = 1; n++; }

   return agx3_bind_compose(enc, F->mnemonic, F->ep, base, F->nbase, ov, n, out,
                            len);
}

/* Native and_live/or_live/xor_live retain their second input for a later
 * operation (f39), then discard it on its last use (f41). and_keep_both
 * clears both discard fields. The operation classes and register fields
 * are identical; only truth-table bits differ. */
static bool
agx3_alu_bit_width(enum agx_isa_encoding enc, enum agx3_alu_op op, bool narrow,
                 uint16_t dest, const uint16_t src[2],
                 const enum agx3_alu_hint hints[2], uint16_t wait_mask,
                 uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if(enc != AGX_ISA_ENCODING_AGX3B || !src || !hints || !out || !len ||
      (op != AGX3_ALU_BAND && op != AGX3_ALU_BOR && op != AGX3_ALU_BXOR) ||
      dest >= 256 || (!narrow && (dest & 1)) || (wait_mask & ~UINT16_C(0x3f)))
      return false;
   for(unsigned i=0;i<2;i++)
      if(src[i]>=256 || (!narrow && (src[i]&1)) || (unsigned)hints[i]>AGX3_ALU_HINT_CACHE)
         return false;
   const struct agx3_alu_form *f=&agx3_alu_forms[op][0];
   struct agx3_field_kv base[20],ov[13];unsigned nb=0,n=0;
   for(unsigned i=0;i<f->nbase;i++){
      unsigned field=f->base[i].f;bool wait=false;
      for(unsigned j=0;j<f->nwait;j++)wait|=field==f->wait[j];
      if(field!=27 && field!=41 && !wait)
         base[nb++]=(struct agx3_field_kv){field,
            narrow && (field == 10 || field == 24 || field == 38) ? 1 : f->base[i].v};
   }
   ov[n++]=(struct agx3_field_kv){2,dest};
   const unsigned operand[]={16,30},cache[]={25,39},discard[]={27,41};
   for(unsigned i=0;i<2;i++){
      ov[n++]=(struct agx3_field_kv){operand[i],src[i]};
      ov[n++]=(struct agx3_field_kv){cache[i],hints[i]==AGX3_ALU_HINT_CACHE};
      ov[n++]=(struct agx3_field_kv){discard[i],hints[i]==AGX3_ALU_HINT_DISCARD};
   }
   for(unsigned i=0;i<f->nwait;i++)
      if(wait_mask&(1u<<i))ov[n++]=(struct agx3_field_kv){f->wait[i],1};
   return agx3_bind_compose(enc,f->mnemonic,f->ep,base,nb,ov,n,out,len);
}

bool
agx3_alu_bit_reg(enum agx_isa_encoding enc, enum agx3_alu_op op,
                 uint16_t dest, const uint16_t src[2],
                 const enum agx3_alu_hint hints[2], uint16_t wait_mask,
                 uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   return agx3_alu_bit_width(enc, op, false, dest, src, hints, wait_mask, out, len);
}

bool
agx3_alu_bit_reg16(enum agx_isa_encoding enc, enum agx3_alu_op op,
                 uint16_t dest, const uint16_t src[2],
                 const enum agx3_alu_hint hints[2], uint16_t wait_mask,
                 uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   return agx3_alu_bit_width(enc, op, true, dest, src, hints, wait_mask, out, len);
}

bool
agx3_alu_bit_andc(enum agx_isa_encoding enc, uint16_t dest,
                  const uint16_t src[2],
                  const enum agx3_alu_hint hints[2], bool dest_cache,
                  uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                  uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !src || !hints || !out || !len ||
       dest >= 256 || src[0] >= 256 || src[1] >= 256 ||
       ((dest | src[0] | src[1]) & 1) || (wait_mask & ~UINT16_C(0x3f)) ||
       (unsigned)hints[0] > AGX3_ALU_HINT_CACHE ||
       (unsigned)hints[1] > AGX3_ALU_HINT_CACHE)
      return false;

   /* Apple `v.x & ~v.y`:
    *   0b010e03020000800000  bit_andc r0, ^r0, ^r1
    * Keeping the first input independently gives:
    *   0b031605020000800000  bit_andc r0, r1, ^r2
    * The operand/hint/wait topology is identical to the grounded ordinary
    * bitops; only the noncommutative mnemonic changes. */
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 2}, {23, 2}, {24, 2},
      {37, 2}, {38, 2},
   };
   struct agx3_field_kv fields[14] = {
      {2, dest}, {16, src[0]}, {30, src[1]}, {11, dest_cache},
      {25, hints[0] == AGX3_ALU_HINT_CACHE},
      {27, hints[0] == AGX3_ALU_HINT_DISCARD},
      {39, hints[1] == AGX3_ALU_HINT_CACHE},
      {41, hints[1] == AGX3_ALU_HINT_DISCARD},
   };
   unsigned n = 8;
   for (unsigned slot = 0; slot < 6; ++slot)
      fields[n++] = (struct agx3_field_kv){44 + slot,
                                          !!(wait_mask & (1u << slot))};
   return agx3_bind_compose(enc, "bit_andc", 0, base, 6, fields, n, out, len);
}

/* Native iadd_live/isub_live retain both inputs, while their dead variants
 * set f27/f42. Keep the same canonical opcode/register fields. */
bool
agx3_alu_integer_reg(enum agx_isa_encoding enc, enum agx3_alu_op op,
                 uint16_t dest, const uint16_t src[2],
                 const enum agx3_alu_hint hints[2], uint16_t wait_mask,
                 uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if(enc != AGX_ISA_ENCODING_AGX3B || !src || !hints || !out || !len ||
      (op != AGX3_ALU_IADD && op != AGX3_ALU_ISUB) ||
      dest >= 256 || (dest & 1) || (wait_mask & ~UINT16_C(0x3f)))
      return false;
   for(unsigned i=0;i<2;i++)
      if(src[i]>=256 || (src[i]&1) || (unsigned)hints[i]>AGX3_ALU_HINT_CACHE)
         return false;
   const struct agx3_alu_form *f=&agx3_alu_forms[op][0];
   struct agx3_field_kv base[20],ov[13];unsigned nb=0,n=0;
   for(unsigned i=0;i<f->nbase;i++){
      unsigned field=f->base[i].f;bool wait=false;
      for(unsigned j=0;j<f->nwait;j++)wait|=field==f->wait[j];
      if(field!=27 && field!=42 && !wait)
         base[nb++]=(struct agx3_field_kv){field,f->base[i].v};
   }
   ov[n++]=(struct agx3_field_kv){2,dest};
   const unsigned operand[]={16,31},cache[]={25,40},discard[]={27,42};
   for(unsigned i=0;i<2;i++){
      ov[n++]=(struct agx3_field_kv){operand[i],src[i]};
      ov[n++]=(struct agx3_field_kv){cache[i],hints[i]==AGX3_ALU_HINT_CACHE};
      ov[n++]=(struct agx3_field_kv){discard[i],hints[i]==AGX3_ALU_HINT_DISCARD};
   }
   for(unsigned i=0;i<f->nwait;i++)
      if(wait_mask&(1u<<i))ov[n++]=(struct agx3_field_kv){f->wait[i],1};
   return agx3_bind_compose(enc,f->mnemonic,f->ep,base,nb,ov,n,out,len);
}

/* Native float comparison selects two integer immediates. Field 11 selects
 * the result width; a 16-bit result can target either register half. */
bool
agx3_alu_fcmp_select_imm_width(enum agx_isa_encoding enc, uint16_t dest,
                        const uint16_t src[2], const bool src_16[2],
                        const enum agx3_alu_hint hints[2],
                        unsigned condition, uint8_t if_true, uint8_t if_false,
                        bool dest_16, bool dest_cache, uint16_t wait_mask,
                        uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !out || !len || !src || !src_16 || !hints ||
       (src_16[0] && src_16[1]) ||
       dest >= 256 || (!dest_16 && (dest & 1)) || (condition != 0 && condition != 1 && condition != 2 &&
        condition != 5 && condition != 6) ||
       (wait_mask != 0 && wait_mask != 32))
      return false;
   for (unsigned i = 0; i < 2; ++i)
      if (src[i] >= 256 || (!src_16[i] && (src[i] & 1)) ||
          (unsigned)hints[i] > AGX3_ALU_HINT_CACHE)
         return false;
   const struct agx3_field_kv base[] = {
      {10, 2}, {11, dest_16 ? 1 : 2}, {42, 2}, {43, src_16[0] ? 1 : 2},
      {59, 2}, {60, src_16[1] ? 1 : 2},
      {76, 1}, {77, dest_16 ? 1 : 2}, {93, 1}, {94, dest_16 ? 1 : 2},
   };
   const struct agx3_field_kv fields[] = {
      {3, dest}, {12, dest_cache}, {35, src[0]}, {52, src[1]},
      {44, hints[0] == AGX3_ALU_HINT_CACHE},
      {46, hints[0] == AGX3_ALU_HINT_DISCARD},
      {61, hints[1] == AGX3_ALU_HINT_CACHE},
      {63, hints[1] == AGX3_ALU_HINT_DISCARD},
      {68, if_true}, {85, if_false}, {110, condition}, {107, wait_mask == 32},
   };
   for (unsigned ep = 20; ep < 28; ++ep)
      if (agx3_bind_compose(enc, "fsel", ep, base, 10, fields, 12, out, len))
         return true;
   return false;
}

bool
agx3_alu_fcmp_minifloat_select_half(
   enum agx_isa_encoding enc, uint16_t dest, uint16_t compare,
   uint8_t minifloat, uint16_t if_true, uint8_t if_false,
   enum agx3_alu_hint compare_hint, enum agx3_alu_hint true_hint,
   unsigned condition, bool dest_cache, uint16_t wait_mask,
   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !out || !len || dest >= 256 ||
       compare >= 256 || (compare & 1) || if_true >= 256 ||
       (unsigned)compare_hint > AGX3_ALU_HINT_CACHE ||
       (unsigned)true_hint > AGX3_ALU_HINT_CACHE ||
       condition > 2 ||
       (wait_mask != 0 && wait_mask != 32))
      return false;

   /* Native vector `select(half(0), h, x == -1.0f)` lanes:
    *   020119b1860a00c22080
    *   0a0319b1860b00022080
    * field51 is the minifloat compare operand, field69 the selected half
    * register and field85 the false immediate. */
   const struct agx3_field_kv base[] = {
      {0, 1}, {24, 2}, {25, 1}, {42, 2}, {43, 2}, {59, 1},
      {60, 2}, {76, 2}, {77, 1}, {93, 1}, {94, 1},
   };
   const struct agx3_field_kv fields[] = {
      {17, dest}, {26, dest_cache}, {35, compare}, {51, minifloat},
      {69, if_true}, {85, if_false},
      {44, compare_hint == AGX3_ALU_HINT_CACHE},
      {46, compare_hint == AGX3_ALU_HINT_DISCARD},
      {78, true_hint == AGX3_ALU_HINT_CACHE},
      {80, true_hint == AGX3_ALU_HINT_DISCARD},
      {107, wait_mask == 32}, {110, condition},
   };
   for (unsigned ep = 20; ep < 28; ++ep)
      if (agx3_bind_compose(enc, "fsel", ep, base, 11, fields, 12, out,
                            len))
         return true;
   return false;
}

bool
agx3_alu_fcmp_minifloat_select_mixed(
   enum agx_isa_encoding enc, uint16_t dest, uint16_t compare,
   uint8_t minifloat, const uint16_t selected[2],
   const bool selected_immediate[2], const enum agx3_alu_hint hints[3],
   unsigned condition, bool dest_16, bool dest_cache, uint16_t wait_mask,
   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !selected ||
       !selected_immediate || !hints || !out || !len || dest >= 256 ||
       compare >= 256 || (compare & 1) || (!dest_16 && (dest & 1)) ||
       (condition != 0 && condition != 1) ||
       (wait_mask != 0 && wait_mask != 32) ||
       (unsigned)hints[0] > AGX3_ALU_HINT_CACHE)
      return false;

   /* The observed half result selects two byte constants; the full result
    * selects exactly one full register and one byte constant. */
   if (dest_16 ? (!selected_immediate[0] || !selected_immediate[1])
               : (selected_immediate[0] == selected_immediate[1]))
      return false;

   for (unsigned i = 0; i < 2; ++i) {
      if (selected[i] >= 256 ||
          (selected_immediate[i]
              ? hints[i + 1] != AGX3_ALU_HINT_NONE
              : ((!dest_16 && (selected[i] & 1)) ||
                 (unsigned)hints[i + 1] > AGX3_ALU_HINT_CACHE)))
         return false;
   }

   struct agx3_field_kv base[11];
   unsigned nbase = 0;
   if (dest_16) {
      base[nbase++] = (struct agx3_field_kv){10, 2};
      base[nbase++] = (struct agx3_field_kv){11, 1};
   } else {
      base[nbase++] = (struct agx3_field_kv){0, 1};
      base[nbase++] = (struct agx3_field_kv){24, 2};
      base[nbase++] = (struct agx3_field_kv){25, 2};
   }
   base[nbase++] = (struct agx3_field_kv){42, 2};
   base[nbase++] = (struct agx3_field_kv){43, 2};
   base[nbase++] = (struct agx3_field_kv){59, 1};
   base[nbase++] = (struct agx3_field_kv){60, 2};
   base[nbase++] = (struct agx3_field_kv){76,
                                          selected_immediate[0] ? 1 : 2};
   base[nbase++] = (struct agx3_field_kv){77, dest_16 ? 1 : 2};
   base[nbase++] = (struct agx3_field_kv){93,
                                          selected_immediate[1] ? 1 : 2};
   base[nbase++] = (struct agx3_field_kv){94, dest_16 ? 1 : 2};
   struct agx3_field_kv fields[16] = {
      {dest_16 ? 3 : 17, dest},
      {dest_16 ? 12 : 26, dest_cache},
      {35, compare},
      {51, minifloat},
      {selected_immediate[0] ? 68 : 69, selected[0]},
      {selected_immediate[1] ? 85 : 86, selected[1]},
      {44, hints[0] == AGX3_ALU_HINT_CACHE},
      {46, hints[0] == AGX3_ALU_HINT_DISCARD},
      {107, wait_mask == 32},
      {110, condition},
   };
   unsigned n = 10;
   const unsigned cache[] = {78, 95}, discard[] = {80, 97};
   for (unsigned i = 0; i < 2; ++i) {
      if (!selected_immediate[i]) {
         fields[n++] = (struct agx3_field_kv){cache[i],
                                              hints[i + 1] == AGX3_ALU_HINT_CACHE};
         fields[n++] = (struct agx3_field_kv){discard[i],
                                              hints[i + 1] == AGX3_ALU_HINT_DISCARD};
      }
   }

   const unsigned preferred = dest_16 ? ((dest & 1) ? 26 : 25) : 22;
   if (agx3_bind_compose(enc, "fsel", preferred, base, nbase, fields, n,
                         out, len))
      return true;
   for (unsigned ep = 0; ep < 28; ++ep) {
      if (ep != preferred &&
          agx3_bind_compose(enc, "fsel", ep, base, nbase, fields, n, out,
                            len))
         return true;
   }
   return false;
}

/* Compare a half/full float register with floating zero, selecting full float
 * registers. The three source lifetimes are independent (including aliases).
 * Native Metal uses these compact forms for mixed-precision shader selects. */
bool
agx3_alu_fcmp_select_zero(enum agx_isa_encoding enc, uint16_t dest,
                          const uint16_t src[3], bool compare_16,
                          const enum agx3_alu_hint hints[3], unsigned condition,
                          bool dest_cache, uint16_t wait_mask,
                          uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !out || !len || !src || !hints ||
       dest >= 256 || (dest & 1) ||
       (condition != 0 && condition != 1 && condition != 2 &&
        condition != 5 && condition != 6) ||
       (wait_mask != 0 && wait_mask != 32))
      return false;
   for (unsigned i = 0; i < 3; ++i)
      if (src[i] >= 256 || ((i != 0 || !compare_16) && (src[i] & 1)) ||
          (unsigned)hints[i] > AGX3_ALU_HINT_CACHE)
         return false;
   const struct agx3_field_kv base[] = {
      {10, 2}, {11, 2}, {42, 2}, {43, compare_16 ? 1 : 2},
      {59, 1}, {60, 2}, {76, 2}, {77, 2}, {93, 2}, {94, 2},
   };
   const struct agx3_field_kv fields[] = {
      {3, dest}, {12, dest_cache}, {35, src[0]}, {51, 0},
      {69, src[1]}, {86, src[2]},
      {44, hints[0] == AGX3_ALU_HINT_CACHE},
      {46, hints[0] == AGX3_ALU_HINT_DISCARD},
      {78, hints[1] == AGX3_ALU_HINT_CACHE},
      {80, hints[1] == AGX3_ALU_HINT_DISCARD},
      {95, hints[2] == AGX3_ALU_HINT_CACHE},
      {97, hints[2] == AGX3_ALU_HINT_DISCARD},
      {110, condition}, {107, wait_mask == 32},
   };
   /* Native forms are ordered by size: 6, 8, then 10 bytes. */
   for (unsigned ep = 0; ep < 28; ++ep)
      if (agx3_bind_compose(enc, "fsel", ep, base, 10, fields, 14, out, len))
         return true;
   return false;
}

bool
agx3_alu_fcmp_select_imm(enum agx_isa_encoding enc, uint16_t dest,
                        const uint16_t src[2], const enum agx3_alu_hint hints[2],
                        unsigned condition, uint8_t if_true, uint8_t if_false,
                        bool dest_16, bool dest_cache, uint16_t wait_mask,
                        uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   const bool src_16[2] = {false, false};
   return agx3_alu_fcmp_select_imm_width(enc, dest, src, src_16, hints, condition,
      if_true, if_false, dest_16, dest_cache, wait_mask, out, len);
}

/* Integer source liveness remains relevant after conversion: IDs can still
 * address later stores. Keep the native instruction and actual RA hints. */
bool
agx3_alu_int_to_float(enum agx_isa_encoding enc, uint16_t dest,
                      uint16_t src, bool is_signed, enum agx3_alu_hint hint,
                      uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                      uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !out || !len ||
       dest >= 256 || src >= 256 || ((dest | src) & 1) ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE || (wait_mask & ~0x3fu))
      return false;
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 2}, {25, 2}, {26, 2},
   };
   struct agx3_field_kv fields[12] = {
      {2, dest}, {18, src}, {27, hint == AGX3_ALU_HINT_CACHE},
      {29, hint == AGX3_ALU_HINT_DISCARD}, {30, is_signed},
   };
   unsigned n = 5;
   for (unsigned i = 0; i < 6; ++i)
      fields[n++] = (struct agx3_field_kv){33 + i, !!(wait_mask & (1u << i))};
   return agx3_bind_compose(enc, "int2flt", 0, base, 4, fields, n, out, len);
}

bool
agx3_alu_bfeil(enum agx_isa_encoding enc, uint16_t dest, uint16_t source,
               uint16_t shift, bool shift_is_register, uint8_t width,
               const enum agx3_alu_hint hints[2], uint16_t wait_mask,
               uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *length)
{
   if (!out || !length || !hints || enc != AGX_ISA_ENCODING_AGX3B ||
       dest >= 256 || source >= 256 || ((dest | source) & 1) ||
       (shift_is_register ? (shift >= 256 || (shift & 1)) : shift >= 32) ||
       width < 1 || width > 31 || (wait_mask & ~0x3f) ||
       (unsigned)hints[0] > AGX3_ALU_HINT_CACHE ||
       (unsigned)hints[1] > AGX3_ALU_HINT_CACHE ||
       (!shift_is_register && hints[1] != AGX3_ALU_HINT_NONE))
      return false;
   const struct agx3_field_kv base[] = {
      {9,2},{10,2},{23,2},{24,2},{39,2},{52,1},{53,2},
   };
   struct agx3_field_kv fields[16] = {
      {2,dest},{16,source},{25,hints[0] == AGX3_ALU_HINT_CACHE},
      {27,hints[0] == AGX3_ALU_HINT_DISCARD},
      {38,shift_is_register ? 2 : 1},{shift_is_register ? 31 : 30,shift},
      {44,0},{57,width},
   };
   unsigned count=8;
   if (shift_is_register) {
      fields[count++] = (struct agx3_field_kv){40,hints[1] == AGX3_ALU_HINT_CACHE};
      fields[count++] = (struct agx3_field_kv){42,hints[1] == AGX3_ALU_HINT_DISCARD};
   }
   for (unsigned i=0;i<6;++i)
      if (wait_mask & (1u<<i))
         fields[count++] = (struct agx3_field_kv){60+i,1};
   return agx3_bind_compose(enc,"srl",0,base,7,fields,count,out,length);
}

bool
agx3_alu_shift_add16(enum agx_isa_encoding enc, uint16_t dest,
                     const uint16_t src[2], const enum agx3_alu_hint hints[2],
                     unsigned shift, bool dest_cache, uint16_t wait_mask,
                     uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !src || !hints || !out || !len ||
       dest >= 192 || src[0] >= 192 || src[1] >= 192 || shift > 4 ||
       (unsigned)hints[0] > AGX3_ALU_HINT_CACHE ||
       (unsigned)hints[1] > AGX3_ALU_HINT_CACHE || (wait_mask & ~0x3fu))
      return false;
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 1}, {23, 2}, {24, 1}, {38, 2}, {39, 1},
   };
   struct agx3_field_kv fields[15] = {
      {2, dest}, {11, dest_cache}, {16, src[0]}, {31, src[1]},
      {25, hints[0] == AGX3_ALU_HINT_CACHE},
      {27, hints[0] == AGX3_ALU_HINT_DISCARD},
      {40, hints[1] == AGX3_ALU_HINT_CACHE},
      {42, hints[1] == AGX3_ALU_HINT_DISCARD}, {54, shift},
   };
   unsigned n = 9;
   for (unsigned i = 0; i < 6; ++i)
      if (wait_mask & (1u << i))
         fields[n++] = (struct agx3_field_kv){45 + i, 1};
   return agx3_bind_compose(enc, "iadd", 0, base, 6, fields, n, out, len);
}

bool
agx3_alu_extract_imm32(enum agx_isa_encoding enc, uint16_t dest,
                       const uint16_t low_high[2], unsigned shift,
                       const enum agx3_alu_hint hints[2], bool dest_cache,
                       uint16_t wait_mask,
                       uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *length)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !out || !length || !low_high ||
       !hints || dest>=192 || low_high[0]>=192 || low_high[1]>=192 ||
       ((dest | low_high[0] | low_high[1]) & 1) || shift<1 || shift>31 ||
       (unsigned)hints[0]>AGX3_ALU_HINT_CACHE ||
       (unsigned)hints[1]>AGX3_ALU_HINT_CACHE || (wait_mask & ~0x3fu))
      return false;
   const struct agx3_field_kv base[] = {
      {9,2},{10,2},{23,2},{24,2},{38,1},{39,2},{52,2},{53,2},
   };
   struct agx3_field_kv fields[16] = {
      {2,dest},{11,dest_cache},{16,low_high[1]},{30,shift},{45,low_high[0]},
      {25,hints[1]==AGX3_ALU_HINT_CACHE},{27,hints[1]==AGX3_ALU_HINT_DISCARD},
      {54,hints[0]==AGX3_ALU_HINT_CACHE},{56,hints[0]==AGX3_ALU_HINT_DISCARD},
      {57,0},
   };
   for(unsigned i=0;i<6;i++)
      fields[10+i]=(struct agx3_field_kv){60+i,!!(wait_mask & (1u<<i))};
   return agx3_bind_compose(enc,"extract",0,base,8,fields,16,out,length);
}

/* Native arithmetic right shift keeps signed fill without a second operation. */
bool
agx3_alu_asr_imm32(enum agx_isa_encoding enc, uint16_t dest, uint16_t source,
                   unsigned shift, enum agx3_alu_hint hint, bool dest_cache,
                   uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                   uint8_t *length)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !out || !length ||
       dest >= 192 || source >= 192 || ((dest | source) & 1) ||
       shift < 1 || shift > 31 || (unsigned)hint > AGX3_ALU_HINT_CACHE ||
       (wait_mask & ~0x3fu))
      return false;
   const struct agx3_field_kv base[] = {
      {9,2},{10,2},{23,2},{24,2},{38,1},{39,2},
   };
   struct agx3_field_kv fields[13] = {
      {2,dest},{11,dest_cache},{16,source},{30,shift},
      {25,hint == AGX3_ALU_HINT_CACHE},{27,hint == AGX3_ALU_HINT_DISCARD},
      {28,1},
   };
   for (unsigned i=0;i<6;i++)
      fields[7+i]=(struct agx3_field_kv){45+i,!!(wait_mask & (1u<<i))};
   return agx3_bind_compose(enc,"sra",0,base,6,fields,13,out,length);
}

bool agx3_alu_fcmp_select_reg_imm(enum agx_isa_encoding enc, uint16_t dest,
                                  const uint16_t src[4], bool compare_zero,
                                  const bool compare_abs[2],
                                  const bool immediate[2],
                                  const enum agx3_alu_hint hints[4], unsigned condition,
                                  bool dest_16, bool dest_cache, uint16_t wait_mask,
                                  uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !out || !len || !src ||
       !compare_abs || !immediate || !hints || dest>=256 ||
       (!dest_16 && (dest&1)) ||
       (condition!=0 && condition!=1 && condition!=2 && condition!=5 && condition!=6) ||
       (wait_mask!=0 && wait_mask!=32) ||
       (compare_zero && (src[1]!=0 || compare_abs[1])))
      return false;
   for(unsigned i=0;i<4;i++) {
      const bool imm=(i==1 && compare_zero) || (i>=2 && immediate[i-2]);
      if(src[i]>=256 || (!imm && (i<2 || !dest_16) && (src[i]&1)) ||
         (unsigned)hints[i]>AGX3_ALU_HINT_CACHE || (imm && hints[i]!=AGX3_ALU_HINT_NONE))
         return false;
   }
   const bool abs_compare = compare_abs[0] || compare_abs[1];
   const bool abs_immediate_select =
      abs_compare && (immediate[0] || immediate[1]);
   if (abs_compare && (compare_zero || condition > 2 || dest_16))
      return false;

   struct agx3_field_kv base[12];
   unsigned nbase=0;
   base[nbase++]=(struct agx3_field_kv){42,2};
   base[nbase++]=(struct agx3_field_kv){43,2};
   base[nbase++]=(struct agx3_field_kv){59,compare_zero?1:2};
   base[nbase++]=(struct agx3_field_kv){60,2};
   base[nbase++]=(struct agx3_field_kv){76,immediate[0]?1:2};
   base[nbase++]=(struct agx3_field_kv){77,dest_16?1:2};
   base[nbase++]=(struct agx3_field_kv){93,immediate[1]?1:2};
   base[nbase++]=(struct agx3_field_kv){94,dest_16?1:2};
   if (abs_immediate_select) {
      base[nbase++]=(struct agx3_field_kv){0,1};
      base[nbase++]=(struct agx3_field_kv){24,2};
      base[nbase++]=(struct agx3_field_kv){25,2};
   } else {
      base[nbase++]=(struct agx3_field_kv){10,2};
      base[nbase++]=(struct agx3_field_kv){11,dest_16?1:2};
   }
   struct agx3_field_kv fields[18]={
      {abs_immediate_select?17:3,dest},
      {abs_immediate_select?26:12,dest_cache},
      {35,src[0]},{compare_zero?51:52,src[1]},
      {immediate[0]?68:69,src[2]},{immediate[1]?85:86,src[3]},
      {110,condition},{107,wait_mask==32},{47,compare_abs[0]},
      {64,compare_abs[1]},
   };
   const unsigned cache[]={44,61,78,95},discard[]={46,63,80,97};
   unsigned n=10;
   for(unsigned i=0;i<4;i++) if((i<2 && !(i==1 && compare_zero)) || (i>=2 && !immediate[i-2])) {
      fields[n++]=(struct agx3_field_kv){cache[i],hints[i]==AGX3_ALU_HINT_CACHE};
      fields[n++]=(struct agx3_field_kv){discard[i],hints[i]==AGX3_ALU_HINT_DISCARD};
   }
   for(unsigned ep=0;ep<36;ep++)
      if(agx3_bind_compose(enc,"fsel",ep,base,nbase,fields,n,out,len))
         return true;
   return false;
}

bool
agx3_alu_fcmp_select_f32_f16(enum agx_isa_encoding enc, uint16_t dest,
                             const uint16_t src[4],
                             const enum agx3_alu_hint hints[4],
                             unsigned condition, bool dest_cache,
                             uint16_t wait_mask,
                             uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !out || !len || !src || !hints ||
       dest >= 256 || (dest & 1) ||
       /* f110 is enum agx_fcond (eq 0, lt 1, gt 2, ge 5, le 6), swept one
        * field at a time against Apple's own bytes for the same form; see
        * agx3_alu_fcmp_select_f32_imm. The NaN-ignoring ltn/gtn codes were
        * not witnessed here and stay out. */
       (condition != 0 && condition != 1 && condition != 2 &&
        condition != 5 && condition != 6) ||
       (wait_mask != 0 && wait_mask != 32))
      return false;

   for (unsigned i = 0; i < 4; ++i) {
      if (src[i] >= 256 || (i != 1 && (src[i] & 1)) ||
          (unsigned)hints[i] > AGX3_ALU_HINT_CACHE)
         return false;
   }

   /* Full/half comparison with two full-register alternatives. The operand
    * width and routing fields are shared with the native mixed comparison and
    * all-register FSEL witnesses already used by the adjacent encoders. */
   const struct agx3_field_kv base[] = {
      {10, 2}, {11, 2}, {42, 2}, {43, 2}, {59, 2},
      {60, 1}, {76, 2}, {77, 2}, {93, 2}, {94, 2},
   };
   struct agx3_field_kv fields[16] = {
      {3, dest}, {12, dest_cache}, {35, src[0]}, {52, src[1]},
      {69, src[2]}, {86, src[3]}, {110, condition}, {107, wait_mask == 32},
   };
   const unsigned cache[] = {44, 61, 78, 95};
   const unsigned discard[] = {46, 63, 80, 97};
   unsigned n = 8;
   for (unsigned i = 0; i < 4; ++i) {
      fields[n++] = (struct agx3_field_kv){cache[i],
                                           hints[i] == AGX3_ALU_HINT_CACHE};
      fields[n++] = (struct agx3_field_kv){discard[i],
                                           hints[i] == AGX3_ALU_HINT_DISCARD};
   }

   for (unsigned ep = 0; ep < 28; ++ep) {
      if (agx3_bind_compose(enc, "fsel", ep, base, 10, fields, n, out, len))
         return true;
   }

   return false;
}

/* Native Metal keeps all four scalar half operands in their assigned halves:
 * src0/src1 are compared and src2/src3 are the selected alternatives.  The
 * instruction carries independent post-RA lifetime hints.  Only the no-wait
 * form is witnessed; callers retire scoreboard slots before composing it.
 */
/*
 * Compare a 32-bit float against a minifloat immediate and select either of
 * two 32-bit registers.
 *
 * This is what an ordinary `x < k ? a : b` becomes: NIR keeps the comparison
 * at 32 bits and Mesa carries the constant as a minifloat, so the packer sees
 * a mixed-width select with no register to widen.  Apple's own bytes, from
 * `metal-nt -arch applegpu_g17p` compiling `(a[i] OP 0.5f) ? c[i] : d[i]`:
 *
 *   <    12 03 0f a1 22 06 03 02 20 0a     f110 = 1
 *   >    12 03 0f a1 22 06 02 02 20 0a     f110 = 2
 *   >=   12 03 0f a1 26 06 02 02 20 0a     f110 = 5
 *   <=   12 03 0f a1 26 06 03 02 20 0a     f110 = 6
 *   ==   12 03 0f a1 26 06 00 02 20 0a     f110 = 0
 *   !=   12 03 0f a1 26 08 00 02 20 06     f110 = 0, selected operands swapped
 *
 * Six kernels differing only in the operator, and only f110 (and, for `!=`,
 * which operand lands in which select slot) changes: the condition selector is
 * literally `enum agx_fcond`, and it is orthogonal to the operand routing.
 * The last row is why an inverted condition needs no separate encoding --
 * Apple spells it by exchanging the two selected values.
 */
bool
agx3_alu_fcmp_select_f32_imm(enum agx_isa_encoding enc, uint16_t dest,
                             uint16_t src0, uint8_t minifloat, uint16_t src2,
                             uint16_t src3,
                             const enum agx3_alu_hint hints[4],
                             unsigned condition, bool dest_cache,
                             uint16_t wait_mask,
                             uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !out || !len || !hints ||
       dest >= 256 || src0 >= 256 || src2 >= 256 || src3 >= 256 ||
       (dest & 1) || (src0 & 1) || (src2 & 1) || (src3 & 1))
      return false;

   /* Only the five conditions Apple was witnessed emitting for this form:
    * eq = 0, lt = 1, gt = 2, ge = 5, le = 6, matching enum agx_fcond. */
   if (condition != 0 && condition != 1 && condition != 2 &&
       condition != 5 && condition != 6)
      return false;

   /* One scoreboard slot at a time: the selector is a three-bit code. */
   if ((wait_mask & ~UINT16_C(0x3f)) || (wait_mask & (wait_mask - 1)))
      return false;

   for (unsigned i = 0; i < 4; ++i) {
      if ((unsigned)hints[i] > AGX3_ALU_HINT_CACHE)
         return false;
   }

   /* The compared immediate and the unused slot-1 register share a lifetime
    * that does not exist; refuse a hint on either. */
   if (hints[1] != AGX3_ALU_HINT_NONE)
      return false;

   struct agx3_field_kv fields[24];
   unsigned n = 0;
   fields[n++] = (struct agx3_field_kv){3, dest};
   fields[n++] = (struct agx3_field_kv){10, 2};
   fields[n++] = (struct agx3_field_kv){11, 2};
   fields[n++] = (struct agx3_field_kv){12, dest_cache};
   fields[n++] = (struct agx3_field_kv){35, src0};
   fields[n++] = (struct agx3_field_kv){42, 2};
   fields[n++] = (struct agx3_field_kv){43, 2};
   fields[n++] = (struct agx3_field_kv){44, hints[0] == AGX3_ALU_HINT_CACHE};
   fields[n++] = (struct agx3_field_kv){46, hints[0] == AGX3_ALU_HINT_DISCARD};
   fields[n++] = (struct agx3_field_kv){51, minifloat};
   fields[n++] = (struct agx3_field_kv){59, 1};
   fields[n++] = (struct agx3_field_kv){60, 2};
   fields[n++] = (struct agx3_field_kv){69, src2};
   fields[n++] = (struct agx3_field_kv){76, 2};
   fields[n++] = (struct agx3_field_kv){77, 2};
   fields[n++] = (struct agx3_field_kv){78, hints[2] == AGX3_ALU_HINT_CACHE};
   fields[n++] = (struct agx3_field_kv){80, hints[2] == AGX3_ALU_HINT_DISCARD};
   fields[n++] = (struct agx3_field_kv){86, src3};
   fields[n++] = (struct agx3_field_kv){93, 2};
   fields[n++] = (struct agx3_field_kv){94, 2};
   fields[n++] = (struct agx3_field_kv){95, hints[3] == AGX3_ALU_HINT_CACHE};
   fields[n++] = (struct agx3_field_kv){97, hints[3] == AGX3_ALU_HINT_DISCARD};
   fields[n++] = (struct agx3_field_kv){110, condition};
   for (unsigned i = 0; i < 6; ++i) {
      if (wait_mask & (1u << i)) {
         fields[n++] = (struct agx3_field_kv){(uint16_t)(102 + i), 1};
         break;
      }
   }

   return agx3_bind_compose(enc, "fsel", 27, NULL, 0, fields, n, out, len);
}

bool
agx3_alu_fcmp_select_half(enum agx_isa_encoding enc, uint16_t dest,
                          const uint16_t src[4],
                          const enum agx3_alu_hint hints[4],
                          unsigned condition, bool dest_cache,
                          uint16_t wait_mask,
                          uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !out || !len || !src || !hints ||
       dest >= 256 ||
       (condition != 0 && condition != 1 && condition != 2 &&
        condition != 5 && condition != 6) || wait_mask != 0)
      return false;

   for (unsigned i = 0; i < 4; ++i) {
      if (src[i] >= 256 || (unsigned)hints[i] > AGX3_ALU_HINT_CACHE)
         return false;
   }

   const struct agx3_field_kv base[] = {
      {10, 2}, {11, 1}, {42, 2}, {43, 1}, {59, 2}, {60, 1},
      {76, 2}, {77, 1}, {93, 2}, {94, 1},
   };
   const struct agx3_field_kv fields[] = {
      {3, dest}, {12, dest_cache}, {35, src[0]}, {52, src[1]},
      {69, src[2]}, {86, src[3]},
      {44, hints[0] == AGX3_ALU_HINT_CACHE},
      {46, hints[0] == AGX3_ALU_HINT_DISCARD},
      {61, hints[1] == AGX3_ALU_HINT_CACHE},
      {63, hints[1] == AGX3_ALU_HINT_DISCARD},
      {78, hints[2] == AGX3_ALU_HINT_CACHE},
      {80, hints[2] == AGX3_ALU_HINT_DISCARD},
      {95, hints[3] == AGX3_ALU_HINT_CACHE},
      {97, hints[3] == AGX3_ALU_HINT_DISCARD},
      {110, condition}, {107, 0},
   };

   for (unsigned ep = 0; ep < 28; ++ep) {
      uint8_t candidate[AGX3_ALU_MAX_LENGTH], candidate_len;
      if (!agx3_bind_compose(enc, "fsel", ep, base, 10, fields, 16,
                             candidate, &candidate_len))
         continue;

      /* The table admits an equivalent alias for two otherwise fixed bits.
       * Apple's G17P compiler canonicalizes scalar-half FSEL with bits62:63
       * set and duplicates destination-cache in its bit3 alias.  Apply
       * exactly that canonical choice, witnessed independently with cached
       * and all-dead operands.  These bits decode to the same named fields.
       */
      if (candidate_len != 10)
         continue;
      candidate[7] |= 0xc0;
      if (dest_cache)
         candidate[0] |= 0x08;
      for (unsigned i = 0; i < candidate_len; ++i)
         out[i] = candidate[i];
      *len = candidate_len;
      return true;
   }

   return false;
}

static bool
agx3_alu_fminmax_half(enum agx_isa_encoding enc, uint16_t dest,
                      const uint16_t src[4],
                      const enum agx3_alu_hint hints[4], bool maximum,
                      bool dest_cache, uint16_t wait_mask,
                      uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !src || !hints || !out || !len ||
       dest >= 256 || (wait_mask != 0 && wait_mask != 32))
      return false;
   for (unsigned i = 0; i < 4; ++i) {
      if (src[i] >= 256 || (unsigned)hints[i] > AGX3_ALU_HINT_CACHE)
         return false;
   }

   const struct agx3_field_kv base[] = {
      {10,2},{11,1},{42,2},{43,1},{59,2},{60,1},{76,2},{77,1},{93,2},{94,1},
   };
   struct agx3_field_kv fields[16] = {
      {3,dest},{12,dest_cache},{35,src[0]},{52,src[1]},
      {69,src[2]},{86,src[3]},{110,maximum ? 7 : 3},{107,wait_mask == 32},
   };
   const unsigned cache[] = {44,61,78,95};
   const unsigned discard[] = {46,63,80,97};
   unsigned n = 8;
   for (unsigned i = 0; i < 4; ++i) {
      fields[n++] = (struct agx3_field_kv){cache[i],
         hints[i] == AGX3_ALU_HINT_CACHE};
      fields[n++] = (struct agx3_field_kv){discard[i],
         hints[i] == AGX3_ALU_HINT_DISCARD};
   }
   for (unsigned ep = 0; ep < 28; ++ep) {
      if (agx3_bind_compose(enc, "fsel", ep, base, 10, fields, n, out, len))
         return true;
   }
   return false;
}

bool
agx3_alu_fmin_half(enum agx_isa_encoding enc, uint16_t dest,
                   const uint16_t src[4],
                   const enum agx3_alu_hint hints[4], bool dest_cache,
                   uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                   uint8_t *len)
{
   return agx3_alu_fminmax_half(enc, dest, src, hints, false, dest_cache,
                               wait_mask, out, len);
}

bool
agx3_alu_fmax_half(enum agx_isa_encoding enc, uint16_t dest,
                   const uint16_t src[4],
                   const enum agx3_alu_hint hints[4], bool dest_cache,
                   uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                   uint8_t *len)
{
   return agx3_alu_fminmax_half(enc, dest, src, hints, true, dest_cache,
                               wait_mask, out, len);
}

/*
 * 32-bit tied fmin/fmax.
 *
 * The 32-bit destination form of a tied fcmpsel is the same Apple `fsel`
 * family the half encoder above uses; only the four width selectors change
 * (f11/f43/f60/f77/f94: 1 = one 16-bit half, 2 = a 32-bit pair).  Which
 * encoding profile Apple picks depends on the second operand:
 *
 *   fmax(a, b)     f32   12 03 1e 05 00 c0    fsel.ep7, src1 a register
 *   fmax(a, 0.0f)  f32   12 03 0a 81 00 c2    fsel.ep2, src1 a minifloat
 *
 * both witnessed from `metal-nt -arch applegpu_g17p`.  ep7 names the
 * destination in f3, ep2 in f17; every other field id is shared.  The
 * immediate is a plain AGX minifloat and appears twice, in f51 and f85,
 * because the compared value and the selected value are the same operand --
 * which is exactly what makes the instruction a minimum or a maximum rather
 * than a general select.
 *
 * The wait mask is the ordinary three-bit scoreboard selector at the top of
 * the last byte: slot n sets f(101 + n).  Only a single slot is encodable in
 * this form, so a multi-slot wait must be retired separately.
 */
static bool
agx3_alu_fminmax32(enum agx_isa_encoding enc, uint16_t dest, uint16_t src0,
                   uint16_t src2, bool src1_immediate, uint16_t src1,
                   uint16_t src3, const enum agx3_alu_hint hints[4],
                   bool maximum, bool dest_cache, uint16_t wait_mask,
                   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !hints || !out || !len || dest >= 256 ||
       src0 >= 256 || src2 >= 256 || src1 >= 256 || src3 >= 256)
      return false;
   if (wait_mask & ~UINT16_C(0x3f))
      return false;
   /* One scoreboard slot at a time: the selector is a three-bit code. */
   if (wait_mask & (wait_mask - 1))
      return false;
   for (unsigned i = 0; i < 4; ++i) {
      if ((unsigned)hints[i] > AGX3_ALU_HINT_CACHE)
         return false;
   }
   if (src1_immediate && (src1 > 255 || src3 > 255))
      return false;

   struct agx3_field_kv fields[24];
   unsigned n = 0;
   const uint16_t dest_field = src1_immediate ? 17 : 3;
   const uint16_t dest_kind = src1_immediate ? 24 : 10;
   const uint16_t dest_size = src1_immediate ? 25 : 11;
   const uint16_t dest_cache_field = src1_immediate ? 26 : 12;
   if (src1_immediate)
      fields[n++] = (struct agx3_field_kv){0, 1};
   fields[n++] = (struct agx3_field_kv){dest_field, dest};
   fields[n++] = (struct agx3_field_kv){dest_kind, 2};
   fields[n++] = (struct agx3_field_kv){dest_size, 2};
   fields[n++] = (struct agx3_field_kv){dest_cache_field, dest_cache};

   /* src0 and src2 are the tied register operand. */
   fields[n++] = (struct agx3_field_kv){35, src0};
   fields[n++] = (struct agx3_field_kv){42, 2};
   fields[n++] = (struct agx3_field_kv){43, 2};
   fields[n++] = (struct agx3_field_kv){44, hints[0] == AGX3_ALU_HINT_CACHE};
   fields[n++] = (struct agx3_field_kv){46, hints[0] == AGX3_ALU_HINT_DISCARD};
   fields[n++] = (struct agx3_field_kv){69, src2};
   fields[n++] = (struct agx3_field_kv){76, 2};
   fields[n++] = (struct agx3_field_kv){77, 2};
   fields[n++] = (struct agx3_field_kv){78, hints[2] == AGX3_ALU_HINT_CACHE};
   fields[n++] = (struct agx3_field_kv){80, hints[2] == AGX3_ALU_HINT_DISCARD};

   /* src1 and src3 are the tied second operand: a register pair or the same
    * minifloat deposited twice. */
   fields[n++] = (struct agx3_field_kv){src1_immediate ? 51 : 52, src1};
   fields[n++] = (struct agx3_field_kv){59, src1_immediate ? 1 : 2};
   fields[n++] = (struct agx3_field_kv){60, 2};
   fields[n++] = (struct agx3_field_kv){src1_immediate ? 85 : 86, src3};
   fields[n++] = (struct agx3_field_kv){93, src1_immediate ? 1 : 2};
   fields[n++] = (struct agx3_field_kv){94, 2};
   if (!src1_immediate) {
      fields[n++] = (struct agx3_field_kv){61, hints[1] == AGX3_ALU_HINT_CACHE};
      fields[n++] = (struct agx3_field_kv){63, hints[1] == AGX3_ALU_HINT_DISCARD};
      fields[n++] = (struct agx3_field_kv){95, hints[3] == AGX3_ALU_HINT_CACHE};
      fields[n++] = (struct agx3_field_kv){97, hints[3] == AGX3_ALU_HINT_DISCARD};
   } else if (hints[1] != AGX3_ALU_HINT_NONE ||
              hints[3] != AGX3_ALU_HINT_NONE) {
      return false;             /* an immediate has no lifetime */
   }

   struct agx3_field_kv all[32];
   unsigned m = 0;
   for (unsigned i = 0; i < n; ++i)
      all[m++] = fields[i];
   for (unsigned i = 0; i < 6; ++i) {
      if (wait_mask & (1u << i))
         all[m++] = (struct agx3_field_kv){(uint16_t)(102 + i), 1};
   }
   all[m++] = (struct agx3_field_kv){110, maximum ? 7 : 3};

   return agx3_bind_compose(enc, "fsel", src1_immediate ? 2 : 7, NULL, 0, all,
                            m, out, len);
}

bool
agx3_alu_fmin_f32(enum agx_isa_encoding enc, uint16_t dest,
                  const uint16_t src[4], const enum agx3_alu_hint hints[4],
                  bool dest_cache, uint16_t wait_mask,
                  uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (!src)
      return false;
   return agx3_alu_fminmax32(enc, dest, src[0], src[2], false, src[1], src[3],
                             hints, false, dest_cache, wait_mask, out, len);
}

bool
agx3_alu_fmax_f32(enum agx_isa_encoding enc, uint16_t dest,
                  const uint16_t src[4], const enum agx3_alu_hint hints[4],
                  bool dest_cache, uint16_t wait_mask,
                  uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (!src)
      return false;
   return agx3_alu_fminmax32(enc, dest, src[0], src[2], false, src[1], src[3],
                             hints, true, dest_cache, wait_mask, out, len);
}

bool
agx3_alu_fmin_f32_imm(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                      uint8_t minifloat, enum agx3_alu_hint hint,
                      bool dest_cache, uint16_t wait_mask,
                      uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   const enum agx3_alu_hint hints[4] = {hint, AGX3_ALU_HINT_NONE, hint,
                                        AGX3_ALU_HINT_NONE};
   return agx3_alu_fminmax32(enc, dest, src, src, true, minifloat, minifloat,
                             hints, false, dest_cache, wait_mask, out, len);
}

bool
agx3_alu_fmax_f32_imm(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                      uint8_t minifloat, enum agx3_alu_hint hint,
                      bool dest_cache, uint16_t wait_mask,
                      uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   const enum agx3_alu_hint hints[4] = {hint, AGX3_ALU_HINT_NONE, hint,
                                        AGX3_ALU_HINT_NONE};
   return agx3_alu_fminmax32(enc, dest, src, src, true, minifloat, minifloat,
                             hints, true, dest_cache, wait_mask, out, len);
}

bool
agx3_alu_hrcp(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
              enum agx3_alu_hint hint, bool dest_cache, uint16_t wait_mask,
              uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !out || !len || dest >= 256 ||
       src >= 256 || (wait_mask & ~UINT16_C(0x3f)) ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE)
      return false;
   const struct agx3_field_kv base[] = {
      {9,2},{10,2},{25,2},{26,1},{43,2},
   };
   struct agx3_field_kv fields[11] = {
      {2,dest},{18,src},{11,dest_cache},
      {27,hint == AGX3_ALU_HINT_CACHE},{29,hint == AGX3_ALU_HINT_DISCARD},
   };
   for (unsigned i=0;i<6;i++)
      fields[5+i]=(struct agx3_field_kv){49+i,!!(wait_mask & (1u<<i))};
   return agx3_bind_compose(enc,"f32rcp",0,base,5,fields,11,out,len);
}

bool
agx3_alu_f32rcp(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                enum agx3_alu_hint hint, bool dest_cache, uint16_t wait_mask,
                uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   /* Identical to the half form except f26 = 2, a 32-bit source pair. */
   if (enc != AGX_ISA_ENCODING_AGX3B || !out || !len || dest >= 256 ||
       src >= 256 || (dest & 1) || (src & 1) ||
       (wait_mask & ~UINT16_C(0x3f)) ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE)
      return false;
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 2}, {25, 2}, {26, 2}, {43, 2},
   };
   struct agx3_field_kv fields[11] = {
      {2, dest}, {18, src}, {11, dest_cache},
      {27, hint == AGX3_ALU_HINT_CACHE},
      {29, hint == AGX3_ALU_HINT_DISCARD},
   };
   for (unsigned i = 0; i < 6; ++i)
      fields[5 + i] =
         (struct agx3_field_kv){49 + i, !!(wait_mask & (1u << i))};
   return agx3_bind_compose(enc, "f32rcp", 0, base, 5, fields, 11, out, len);
}

bool
agx3_alu_f32rsqrt(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                  enum agx3_alu_hint hint, bool dest_cache,
                  uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                  uint8_t *len)
{
   /* f10 and f26 are the destination's and source's widths; the half form
    * sets both to 1, the 32-bit form to 2. */
   if (enc != AGX_ISA_ENCODING_AGX3B || !out || !len || dest >= 256 ||
       src >= 256 || (dest & 1) || (src & 1) ||
       (wait_mask & ~UINT16_C(0x3f)) ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE)
      return false;
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 2}, {25, 2}, {26, 2},
   };
   struct agx3_field_kv fields[11] = {
      {2, dest}, {18, src}, {11, dest_cache},
      {27, hint == AGX3_ALU_HINT_CACHE},
      {29, hint == AGX3_ALU_HINT_DISCARD},
   };
   for (unsigned i = 0; i < 6; ++i)
      fields[5 + i] =
         (struct agx3_field_kv){35 + i, !!(wait_mask & (1u << i))};
   return agx3_bind_compose(enc, "f32rsqrt", 0, base, 4, fields, 11, out, len);
}

/*
 * The 32-bit log2/exp2 pair. Field-for-field the half forms below with f10 and
 * f26 raised from 1 to 2, and with all six wait slots reachable rather than
 * only slot 6, which is the same relationship f32rsqrt has to hrsqrt.
 */
static bool
agx3_alu_f32_unop(enum agx_isa_encoding enc, const char *mnemonic,
                  uint16_t dest, uint16_t src, enum agx3_alu_hint hint,
                  bool dest_cache, uint16_t wait_mask, bool src_neg,
                  bool src_abs, uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !out || !len || dest >= 256 ||
       src >= 256 || (dest & 1) || (src & 1) ||
       (wait_mask & ~UINT16_C(0x3f)) ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE)
      return false;

   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 2}, {25, 2}, {26, 2},
   };
   struct agx3_field_kv fields[13] = {
      {2, dest}, {18, src}, {11, dest_cache},
      {27, hint == AGX3_ALU_HINT_CACHE},
      {29, hint == AGX3_ALU_HINT_DISCARD},
      {30, src_abs}, {31, src_neg},
   };
   for (unsigned i = 0; i < 6; ++i)
      fields[7 + i] =
         (struct agx3_field_kv){35 + i, !!(wait_mask & (1u << i))};
   return agx3_bind_compose(enc, mnemonic, 0, base, 4, fields, 13, out, len);
}

bool
agx3_alu_f32log2(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                 enum agx3_alu_hint hint, bool dest_cache, uint16_t wait_mask,
                 uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   return agx3_alu_f32_unop(enc, "f32log2", dest, src, hint, dest_cache,
                            wait_mask, false, false, out, len);
}

bool
agx3_alu_f32exp2(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                 enum agx3_alu_hint hint, bool dest_cache, uint16_t wait_mask,
                 uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   return agx3_alu_f32_unop(enc, "f32exp2", dest, src, hint, dest_cache,
                            wait_mask, false, false, out, len);
}

bool
agx3_alu_f32log2_mod(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                     enum agx3_alu_hint hint, bool dest_cache,
                     uint16_t wait_mask, bool src_neg, bool src_abs,
                     uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   return agx3_alu_f32_unop(enc, "f32log2", dest, src, hint, dest_cache,
                            wait_mask, src_neg, src_abs, out, len);
}

bool
agx3_alu_f32exp2_mod(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                     enum agx3_alu_hint hint, bool dest_cache,
                     uint16_t wait_mask, bool src_neg, bool src_abs,
                     uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   return agx3_alu_f32_unop(enc, "f32exp2", dest, src, hint, dest_cache,
                            wait_mask, src_neg, src_abs, out, len);
}

/*
 * The 32-bit rounding modes. Field-for-field the half forms below with f10
 * and f26 raised from 1 to 2, exactly as the 32-bit logarithm relates to the
 * half one, plus the source modifiers f30 (absolute value) and f31 (negate)
 * that every member of this form family carries. f43 is the mode.
 */
static bool
agx3_alu_f32_rint(enum agx_isa_encoding enc, uint64_t mode, uint16_t dest,
                  uint16_t src, enum agx3_alu_hint hint, bool dest_cache,
                  uint16_t wait_mask, bool src_neg, bool src_abs,
                  uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !out || !len || dest >= 256 ||
       src >= 256 || (dest & 1) || (src & 1) ||
       (wait_mask & ~UINT16_C(0x3f)) ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE)
      return false;

   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 2}, {25, 2}, {26, 2}, {43, mode},
   };
   struct agx3_field_kv fields[13] = {
      {2, dest}, {18, src}, {11, dest_cache},
      {27, hint == AGX3_ALU_HINT_CACHE},
      {29, hint == AGX3_ALU_HINT_DISCARD},
      {30, src_abs}, {31, src_neg},
   };
   for (unsigned i = 0; i < 6; ++i)
      fields[7 + i] =
         (struct agx3_field_kv){35 + i, !!(wait_mask & (1u << i))};
   return agx3_bind_compose(enc, "f32rint", 0, base, 5, fields, 13, out, len);
}

bool
agx3_alu_f32floor(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                  enum agx3_alu_hint hint, bool dest_cache, uint16_t wait_mask,
                  bool src_neg, bool src_abs,
                  uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   return agx3_alu_f32_rint(enc, 3, dest, src, hint, dest_cache, wait_mask,
                            src_neg, src_abs, out, len);
}

bool
agx3_alu_f32ceil(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                 enum agx3_alu_hint hint, bool dest_cache, uint16_t wait_mask,
                 bool src_neg, bool src_abs,
                 uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   return agx3_alu_f32_rint(enc, 2, dest, src, hint, dest_cache, wait_mask,
                            src_neg, src_abs, out, len);
}

bool
agx3_alu_hrsqrt(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                enum agx3_alu_hint hint, bool dest_cache,
                uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                uint8_t *len)
{
   /* Apple names the instruction f32rsqrt, while field10 and field26 select
    * scalar half destination/source widths.  Fresh kernels cover even and odd
    * halves, retained/dead inputs, destination caching, and slot-6/no wait. */
   if (enc != AGX_ISA_ENCODING_AGX3B || !out || !len || dest >= 256 ||
       src >= 256 ||
       (hint != AGX3_ALU_HINT_NONE && hint != AGX3_ALU_HINT_DISCARD) ||
       (wait_mask != 0 && wait_mask != 32))
      return false;

   const struct agx3_field_kv base[] = {
      {9,2},{10,1},{25,2},{26,1},
   };
   const struct agx3_field_kv fields[] = {
      {2,dest}, {18,src}, {11,dest_cache},
      {29,hint == AGX3_ALU_HINT_DISCARD}, {40,wait_mask == 32},
   };
   return agx3_bind_compose(enc, "f32rsqrt", 0, base, 4, fields, 5,
                            out, len);
}

bool
agx3_alu_hexp2(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
               enum agx3_alu_hint hint, bool dest_cache,
               uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
               uint8_t *len)
{
   /* As with reciprocal sqrt, f32exp2 uses explicit scalar-half width
    * selectors.  Fresh Apple kernels cover odd halves, retained/dead input,
    * destination caching, and slot-6/no dependency forms. */
   if (enc != AGX_ISA_ENCODING_AGX3B || !out || !len || dest >= 256 ||
       src >= 256 ||
       (hint != AGX3_ALU_HINT_NONE && hint != AGX3_ALU_HINT_DISCARD) ||
       (wait_mask != 0 && wait_mask != 32))
      return false;

   const struct agx3_field_kv base[] = {
      {9,2},{10,1},{25,2},{26,1},
   };
   const struct agx3_field_kv fields[] = {
      {2,dest}, {18,src}, {11,dest_cache},
      {29,hint == AGX3_ALU_HINT_DISCARD}, {40,wait_mask == 32},
   };
   return agx3_bind_compose(enc, "f32exp2", 0, base, 4, fields, 5,
                            out, len);
}

bool
agx3_alu_hlog2(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
               enum agx3_alu_hint hint, bool dest_cache,
               uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
               uint8_t *len)
{
   /* f32log2 uses the same explicit half-width controls as exp2.  Fresh
    * Apple kernels independently cover both half lanes, source lifetime,
    * destination caching, and slot-6/no dependency forms. */
   if (enc != AGX_ISA_ENCODING_AGX3B || !out || !len || dest >= 256 ||
       src >= 256 ||
       (hint != AGX3_ALU_HINT_NONE && hint != AGX3_ALU_HINT_DISCARD) ||
       (wait_mask != 0 && wait_mask != 32))
      return false;

   const struct agx3_field_kv base[] = {
      {9,2},{10,1},{25,2},{26,1},
   };
   const struct agx3_field_kv fields[] = {
      {2,dest}, {18,src}, {11,dest_cache},
      {29,hint == AGX3_ALU_HINT_DISCARD}, {40,wait_mask == 32},
   };
   return agx3_bind_compose(enc, "f32log2", 0, base, 4, fields, 5,
                            out, len);
}

bool
agx3_alu_hfloor(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                enum agx3_alu_hint hint, bool dest_cache,
                uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !out || !len || dest >= 256 ||
       src >= 256 ||
       (hint != AGX3_ALU_HINT_NONE && hint != AGX3_ALU_HINT_DISCARD) ||
       (wait_mask != 0 && wait_mask != 32))
      return false;
   /* f32rint mode3 is floor. Field10/26 select scalar halves; Apple kernels
    * cover both lanes, lifetime, destination cache, and slot-6/no wait. */
   const struct agx3_field_kv base[] = {
      {9,2},{10,1},{25,2},{26,1},{43,3},
   };
   const struct agx3_field_kv fields[] = {
      {2,dest},{18,src},{11,dest_cache},
      {29,hint == AGX3_ALU_HINT_DISCARD},{40,wait_mask == 32},
   };
   return agx3_bind_compose(enc, "f32rint", 0, base, 5, fields, 5,
                            out, len);
}

bool
agx3_alu_hceil(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
               enum agx3_alu_hint hint, bool dest_cache,
               uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
               uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !out || !len || dest >= 256 ||
       src >= 256 ||
       (hint != AGX3_ALU_HINT_NONE && hint != AGX3_ALU_HINT_DISCARD) ||
       (wait_mask != 0 && wait_mask != 32))
      return false;
   /* f32rint mode2 is ceiling. Field10/26 select scalar halves; Apple kernels
    * cover both lanes, lifetime, destination cache, and slot-6/no wait. */
   const struct agx3_field_kv base[] = {
      {9,2},{10,1},{25,2},{26,1},{43,2},
   };
   const struct agx3_field_kv fields[] = {
      {2,dest},{18,src},{11,dest_cache},
      {29,hint == AGX3_ALU_HINT_DISCARD},{40,wait_mask == 32},
   };
   return agx3_bind_compose(enc, "f32rint", 0, base, 5, fields, 5,
                            out, len);
}

/* Native float32 multiplication rounds directly to the half destination. */
bool
agx3_alu_fmul_narrow(enum agx_isa_encoding enc, uint16_t dest,
                     const uint16_t src[2], const enum agx3_alu_hint hints[2],
                     bool dest_cache, uint16_t wait_mask,
                     uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !src || !hints || !out || !len ||
       dest >= 256 || src[0] >= 256 || src[1] >= 256 ||
       ((src[0] | src[1]) & 1) || (wait_mask & ~UINT16_C(0x3f)) ||
       (unsigned)hints[0] > AGX3_ALU_HINT_CACHE ||
       (unsigned)hints[1] > AGX3_ALU_HINT_CACHE)
      return false;
   const struct agx3_field_kv base[] = {
      {9,2},{10,1},{25,2},{26,2},{42,2},{43,2},{60,2},{74,2},
   };
   struct agx3_field_kv fields[14] = {
      {2,dest},{18,src[0]},{35,src[1]},{11,dest_cache},
      {27,hints[0] == AGX3_ALU_HINT_CACHE},{29,hints[0] == AGX3_ALU_HINT_DISCARD},
      {44,hints[1] == AGX3_ALU_HINT_CACHE},{46,hints[1] == AGX3_ALU_HINT_DISCARD},
   };
   for (unsigned i=0;i<6;i++)
      fields[8+i] = (struct agx3_field_kv){80+i,!!(wait_mask & (1u<<i))};
   for (unsigned ep=0;ep<2;ep++)
      if (agx3_bind_compose(enc,"f32mul",ep,base,8,fields,14,out,len))
         return true;
   return false;
}

bool
agx3_alu_hmul(enum agx_isa_encoding enc, uint16_t dest,
              const uint16_t src[2], const enum agx3_alu_hint hints[2],
              bool dest_cache, uint16_t wait_mask,
              uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !src || !hints || !out || !len ||
       dest >= 256 || src[0] >= 256 || src[1] >= 256 ||
       (wait_mask & ~UINT16_C(0x3f)) ||
       (unsigned)hints[0] > AGX3_ALU_HINT_CACHE ||
       (unsigned)hints[1] > AGX3_ALU_HINT_CACHE)
      return false;
   const struct agx3_field_kv base[] = {
      {9,2},{10,1},{24,2},{25,1},{41,2},{42,1},
   };
   struct agx3_field_kv fields[14] = {
      {2,dest},{17,src[0]},{34,src[1]},{11,dest_cache},
      {26,hints[0] == AGX3_ALU_HINT_CACHE},{28,hints[0] == AGX3_ALU_HINT_DISCARD},
      {43,hints[1] == AGX3_ALU_HINT_CACHE},{45,hints[1] == AGX3_ALU_HINT_DISCARD},
   };
   for (unsigned i=0;i<6;i++)
      fields[8+i] = (struct agx3_field_kv){51+i,!!(wait_mask & (1u<<i))};
   for (unsigned ep=0;ep<2;ep++)
      if (agx3_bind_compose(enc,"f16mul",ep,base,6,fields,14,out,len))
         return true;
   return false;
}

bool
agx3_alu_hmul_minifloat(enum agx_isa_encoding enc, uint16_t dest,
                        uint16_t src, uint16_t immediate,
                        enum agx3_alu_hint hint, bool dest_cache,
                        uint16_t wait_mask,
                        uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !out || !len || dest >= 256 ||
       src >= 256 || immediate > UINT8_MAX ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE ||
       (wait_mask & ~UINT16_C(0x3f)))
      return false;

   /* Fresh G17P Metal witnesses place the minifloat in field16, retain the
    * ordinary half register/lifetime fields, and carry all six wait bits. */
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 1}, {24, 1}, {25, 1}, {41, 2}, {42, 1},
   };
   struct agx3_field_kv fields[12] = {
      {2, dest},
      {16, immediate},
      {34, src},
      {11, dest_cache},
      {43, hint == AGX3_ALU_HINT_CACHE},
      {45, hint == AGX3_ALU_HINT_DISCARD},
   };
   for (unsigned i = 0; i < 6; ++i)
      fields[6 + i] = (struct agx3_field_kv){51 + i,
                                            !!(wait_mask & (1u << i))};

   for (unsigned ep = 0; ep < 2; ++ep)
      if (agx3_bind_compose(enc, "f16mul", ep, base, 6, fields, 12, out,
                            len))
         return true;

   return false;
}

/* Scalar half add uses the register, lifetime and six-slot wait fields pinned
 * by fresh G17P add_dead/add_live Metal witnesses. */
bool
agx3_alu_hadd(enum agx_isa_encoding enc, uint16_t dest,
              const uint16_t src[2], const enum agx3_alu_hint hints[2],
              bool dest_cache, uint16_t wait_mask,
              uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !src || !hints || !out || !len ||
       dest >= 256 || src[0] >= 256 || src[1] >= 256 ||
       (wait_mask & ~UINT16_C(0x3f)) ||
       (unsigned)hints[0] > AGX3_ALU_HINT_CACHE ||
       (unsigned)hints[1] > AGX3_ALU_HINT_CACHE)
      return false;

   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 1}, {24, 2}, {25, 1}, {41, 2}, {42, 1},
   };
   struct agx3_field_kv fields[14] = {
      {2, dest},
      {17, src[0]},
      {34, src[1]},
      {11, dest_cache},
      {26, hints[0] == AGX3_ALU_HINT_CACHE},
      {28, hints[0] == AGX3_ALU_HINT_DISCARD},
      {43, hints[1] == AGX3_ALU_HINT_CACHE},
      {45, hints[1] == AGX3_ALU_HINT_DISCARD},
   };
   for (unsigned i = 0; i < 6; ++i)
      fields[8 + i] =
         (struct agx3_field_kv){51 + i, !!(wait_mask & (1u << i))};

   for (unsigned ep = 0; ep < 4; ++ep) {
      if (agx3_bind_compose(enc, "f16add", ep, base, 6, fields, 14, out, len))
         return true;
   }

   return false;
}

/* Native scalar half FMA preserves the single-rounding f16 semantics. */
bool
agx3_alu_hfma(enum agx_isa_encoding enc, uint16_t dest,
              const uint16_t src[3], const enum agx3_alu_hint hints[3],
              bool product_neg, bool src2_neg, bool dest_cache,
              uint16_t wait_mask,
              uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !src || !hints || !out || !len ||
       dest >= 256 || src[0] >= 256 || src[1] >= 256 || src[2] >= 256 ||
       (wait_mask & ~UINT16_C(0x3f)) ||
       (unsigned)hints[0] > AGX3_ALU_HINT_CACHE ||
       (unsigned)hints[1] > AGX3_ALU_HINT_CACHE ||
       (unsigned)hints[2] > AGX3_ALU_HINT_CACHE)
      return false;

   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 1}, {24, 2}, {25, 1},
      {41, 2}, {42, 1}, {58, 2}, {59, 1},
   };
   struct agx3_field_kv fields[19] = {
      {2, dest},
      {17, src[0]},
      {34, src[1]},
      {51, src[2]},
      {11, dest_cache},
      {26, hints[0] == AGX3_ALU_HINT_CACHE},
      {28, hints[0] == AGX3_ALU_HINT_DISCARD},
      {43, hints[1] == AGX3_ALU_HINT_CACHE},
      {45, hints[1] == AGX3_ALU_HINT_DISCARD},
      {60, hints[2] == AGX3_ALU_HINT_CACHE},
      {62, hints[2] == AGX3_ALU_HINT_DISCARD},
      {47, product_neg},
      {64, src2_neg},
   };
   for (unsigned i = 0; i < 6; ++i)
      fields[13 + i] =
         (struct agx3_field_kv){68 + i, !!(wait_mask & (1u << i))};

   /* The three-register form begins at ep3.  The shorter entries alias a
    * two-source layout and cannot represent all three independent operands. */
   for (unsigned ep = 3; ep < 6; ++ep) {
      if (agx3_bind_compose(enc, "f16fma", ep, base, 8, fields, 19, out,
                            len)) {
         /* metal-nt canonicalizes the otherwise ignored tail bits of ep3. */
         if (*len == 8)
            out[7] |= 0xc0;
         return true;
      }
   }

   return false;
}

/* Exact half-to-float widening uses native mixed-width ADD with negative zero. */
bool
agx3_alu_widen_half(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                    enum agx3_alu_hint hint, bool src_abs, bool src_neg,
                    bool dest_cache, uint16_t wait_mask,
                    uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !out || !len || dest >= 256 ||
       src >= 256 || (dest & 1) || (wait_mask & ~UINT16_C(0x3f)) ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE || (src_abs && src_neg))
      return false;
   const struct agx3_field_kv base[] = {
      {9,2},{10,2},{25,2},{26,1},{42,1},{43,2},
   };
   struct agx3_field_kv fields[14] = {
      {2,dest},{18,src},{34,0x80},{11,dest_cache},
      {27,hint == AGX3_ALU_HINT_CACHE},{29,hint == AGX3_ALU_HINT_DISCARD},
      {30,src_abs},{31,src_neg},
   };
   for (unsigned i=0;i<6;i++)
      fields[8+i] = (struct agx3_field_kv){52+i,!!(wait_mask & (1u<<i))};
   for (unsigned ep=0;ep<4;ep++)
      if (agx3_bind_compose(enc,"f32add",ep,base,6,fields,14,out,len)) {
         /* metal-nt canonicalizes modifier-specific tail bits. */
         if (src_abs && *len == 10)
            out[7] |= 0x80;
         if (src_neg && *len == 8)
            out[7] |= 0x80;
         return true;
      }
   return false;
}
