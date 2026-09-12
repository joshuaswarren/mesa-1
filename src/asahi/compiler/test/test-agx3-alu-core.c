/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * The composer-driven wait-carrying ALU encoder must reproduce the compact hand
 * packers exactly when no wait is needed, and depositing a wait must change the
 * bytes (and keep the length) without disturbing the operands.
 */
#include "agx3_alu_core.h"
#include "agx3_alu_core_vectors.c"
#include <stdio.h>
#include <string.h>
#include <assert.h>

static void
test_native_popcount(void)
{
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   static const uint8_t u16[] = {0x27, 0x05, 0x56, 0x02,
                                 0x02, 0x04, 0x50, 0x04};
   static const uint8_t u32[] = {0x27, 0x05, 0x56, 0x00,
                                 0x02, 0x00, 0x5c, 0x04};
   static const uint8_t u16_cached[] = {0x27, 0x05, 0x56, 0x02,
                                        0x03, 0x04, 0x50, 0x04};
   /* applegpu-nt 32023.917: popcount of a still-live SIMD ballot mask. */
   static const uint8_t u32_live[] = {0x27, 0x05, 0x54, 0x04,
                                      0x03, 0x00, 0x4e, 0x04};
   assert(agx3_alu_popcount(AGX_ISA_ENCODING_AGX3B, 2, 2, true,
                            AGX3_ALU_HINT_DISCARD, false, 0, bytes, &len));
   assert(len == sizeof(u16) && !memcmp(bytes, u16, len));
   assert(agx3_alu_popcount(AGX_ISA_ENCODING_AGX3B, 0, 0, false,
                            AGX3_ALU_HINT_DISCARD, false, 0, bytes, &len));
   assert(len == sizeof(u32) && !memcmp(bytes, u32, len));
   assert(agx3_alu_popcount(AGX_ISA_ENCODING_AGX3B, 2, 2, true,
                            AGX3_ALU_HINT_DISCARD, true, 0, bytes, &len));
   assert(len == sizeof(u16_cached) && !memcmp(bytes, u16_cached, len));
   assert(agx3_alu_popcount(AGX_ISA_ENCODING_AGX3B, 4, 0, false,
                            AGX3_ALU_HINT_CACHE, true, 0, bytes, &len));
   assert(len == sizeof(u32_live) && !memcmp(bytes, u32_live, len));

   assert(!agx3_alu_popcount(AGX_ISA_ENCODING_AGX3C, 0, 0, false,
                             AGX3_ALU_HINT_DISCARD, false, 0, bytes, &len));
   assert(!agx3_alu_popcount(AGX_ISA_ENCODING_AGX3B, 0, 0, false,
                             AGX3_ALU_HINT_NONE, false, 0, bytes, &len));
   assert(!agx3_alu_popcount(AGX_ISA_ENCODING_AGX3B, 0, 0, false,
                             AGX3_ALU_HINT_DISCARD, false, 1, bytes, &len));
}

static void
test_native_ft1(void)
{
   static const struct {
      uint16_t dest, src, wait;
      enum agx3_alu_hint hint;
      bool dest_cache;
      uint8_t bytes[8];
   } witnesses[] = {
      {2, 0, 1, AGX3_ALU_HINT_NONE, false,
       {0xa7, 0x05, 0x56, 0x02, 0x02, 0x00, 0x4c, 0x04}},
      {0, 0, 0, AGX3_ALU_HINT_CACHE, true,
       {0xa7, 0x05, 0x54, 0x00, 0x03, 0x00, 0x5c, 0x04}},
      {6, 0, 1, AGX3_ALU_HINT_DISCARD, true,
       {0xa7, 0x05, 0x56, 0x06, 0x03, 0x00, 0x4e, 0x04}},
      {6, 4, 1, AGX3_ALU_HINT_DISCARD, true,
       {0xa7, 0x05, 0x56, 0x06, 0x03, 0x04, 0x4e, 0x04}},
   };
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   for (unsigned i = 0; i < sizeof(witnesses) / sizeof(witnesses[0]); ++i) {
      assert(agx3_alu_ft1_u32(AGX_ISA_ENCODING_AGX3B, witnesses[i].dest,
                              witnesses[i].src,
                              witnesses[i].hint, witnesses[i].dest_cache,
                              witnesses[i].wait, bytes, &len));
      assert(len == 8 && !memcmp(bytes, witnesses[i].bytes, len));
   }

   /* This exact post-RA shape is the vkQuake update_lightmap use. */
   assert(agx3_alu_ft1_u32(AGX_ISA_ENCODING_AGX3B, 26, 26,
                           AGX3_ALU_HINT_CACHE, true, 0, bytes, &len));
   assert(len == 8 &&
          !memcmp(bytes,
                  (uint8_t[]){0xa7, 0x05, 0x54, 0x1a,
                              0x03, 0x1a, 0x5c, 0x04},
                  8));

   assert(!agx3_alu_ft1_u32(AGX_ISA_ENCODING_AGX3C, 26, 26,
                            AGX3_ALU_HINT_CACHE, true, 0, bytes, &len));
   assert(!agx3_alu_ft1_u32(AGX_ISA_ENCODING_AGX3B, 27, 26,
                            AGX3_ALU_HINT_CACHE, true, 0, bytes, &len));
   assert(!agx3_alu_ft1_u32(AGX_ISA_ENCODING_AGX3B, 26, 26,
                            AGX3_ALU_HINT_CACHE, true, 64, bytes, &len));
   assert(!agx3_alu_ft1_u32(AGX_ISA_ENCODING_AGX3B, 26, 27,
                            AGX3_ALU_HINT_CACHE, true, 0, bytes, &len));
}

/*
 * Full-precision reciprocal and reciprocal square root.
 *
 * Only the half forms were wired up, so a 32-bit float divide or a
 * normalize() -- which NIR lowers to frcp / frsq on 32-bit values -- refused
 * the whole shader out of agx3_pack_binary. Both vectors are Apple's own,
 * from `metal-nt -N compute.mtlp -arch applegpu_g17p`:
 *
 *   o[i] = 1.0f / a[i]    af 00 56 00 02 00 10 48 20 00
 *   o[i] = rsqrt(a[i])    af 01 56 00 02 00 b0 40 00 00
 *
 * In both, the loaded source is dead (f29 = 1) and the instruction waits the
 * load's scoreboard slot 6.
 */
static void
test_native_f32_rcp_rsqrt(void)
{
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;

   static const uint8_t rcp[] = {0xaf, 0x00, 0x56, 0x00, 0x02,
                                 0x00, 0x10, 0x48, 0x20, 0x00};
   assert(agx3_alu_f32rcp(AGX_ISA_ENCODING_AGX3B, 0, 0,
                          AGX3_ALU_HINT_DISCARD, false, 1u << 5, bytes, &len));
   assert(len == sizeof(rcp) && !memcmp(bytes, rcp, len));

   static const uint8_t rsq[] = {0xaf, 0x01, 0x56, 0x00, 0x02,
                                 0x00, 0xb0, 0x40, 0x00, 0x00};
   assert(agx3_alu_f32rsqrt(AGX_ISA_ENCODING_AGX3B, 0, 0,
                            AGX3_ALU_HINT_DISCARD, false, 1u << 5, bytes,
                            &len));
   assert(len == sizeof(rsq) && !memcmp(bytes, rsq, len));

   /* The half and the 32-bit form are distinct encodings, not aliases. */
   uint8_t half[AGX3_ALU_MAX_LENGTH], hlen;
   assert(agx3_alu_hrcp(AGX_ISA_ENCODING_AGX3B, 0, 0, AGX3_ALU_HINT_DISCARD,
                        false, 1u << 5, half, &hlen));
   assert(hlen != len || memcmp(half, rcp, len) != 0);

   /* A 32-bit operand pair is aligned, and only AGX3b is witnessed. */
   assert(!agx3_alu_f32rcp(AGX_ISA_ENCODING_AGX3B, 1, 0, AGX3_ALU_HINT_NONE,
                           false, 0, bytes, &len));
   assert(!agx3_alu_f32rsqrt(AGX_ISA_ENCODING_AGX3B, 0, 1, AGX3_ALU_HINT_NONE,
                             false, 0, bytes, &len));
   assert(!agx3_alu_f32rcp(AGX_ISA_ENCODING_AGX3C, 0, 0, AGX3_ALU_HINT_NONE,
                           false, 0, bytes, &len));
}

/*
 * Full-precision base-2 logarithm and exponential.
 *
 * Only the half forms were wired up, so every 32-bit log2 or exp2 refused the
 * whole shader -- and with them every pow(), which NIR lowers to
 * fexp2(fmul(y, flog2(x))). An sRGB fragment epilog is made of exactly that,
 * which is why `asahi-g17p-srgb-epilog` could not compile and why a G17P
 * fragment shader writing an sRGB swapchain was refused outright.
 *
 * Ten vectors, all Apple's own from `metal-nt -arch applegpu_g17p` over three
 * probe sets. They vary the operand registers, the source lifetime, the
 * destination cache bit and the wait mask independently, so the table pins the
 * fields rather than merely covering the instruction.
 */
static void
test_native_f32_log2_exp2(void)
{
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;

   static const struct {
      const char *what;
      bool exp2;
      uint16_t dest, src;
      enum agx3_alu_hint hint;
      bool dest_cache;
      uint16_t wait;
      uint8_t bytes[10];
   } vectors[] = {
      {"log2(a[i])", false, 0, 0, AGX3_ALU_HINT_DISCARD, false, 1u << 5,
       {0x2f, 0x02, 0x56, 0x00, 0x02, 0x00, 0xb0, 0x40, 0x00, 0x00}},
      {"exp2(a[i])", true, 0, 0, AGX3_ALU_HINT_DISCARD, false, 1u << 5,
       {0xaf, 0x02, 0x56, 0x00, 0x02, 0x00, 0xb0, 0x40, 0x00, 0x00}},
      {"log2 kept live for two readers", false, 4, 0, AGX3_ALU_HINT_CACHE,
       true, 1u << 5,
       {0x2f, 0x02, 0x56, 0x04, 0x03, 0x00, 0x92, 0x40, 0x00, 0x00}},
      {"exp2 of a multiply, pow()", true, 8, 6, AGX3_ALU_HINT_DISCARD, true, 0,
       {0xaf, 0x02, 0x54, 0x08, 0x03, 0x0c, 0xb0, 0x40, 0x00, 0x00}},
      {"exp2 in an sRGB encode", true, 4, 4, AGX3_ALU_HINT_DISCARD, true, 0,
       {0xaf, 0x02, 0x54, 0x04, 0x03, 0x08, 0xb0, 0x40, 0x00, 0x00}},
      /* exp2(log2(x) * y) spelled out: the shape NIR lowers fpow to, and the
       * one an sRGB epilog is built from. */
      {"log2 of a hand-written pow", false, 0, 0, AGX3_ALU_HINT_DISCARD, true,
       1u << 5,
       {0x2f, 0x02, 0x56, 0x00, 0x03, 0x00, 0xb0, 0x40, 0x00, 0x00}},
      {"exp2 of a hand-written pow", true, 0, 0, AGX3_ALU_HINT_DISCARD, true, 0,
       {0xaf, 0x02, 0x54, 0x00, 0x03, 0x00, 0xb0, 0x40, 0x00, 0x00}},
      {"exp2 of powr()", true, 0, 0, AGX3_ALU_HINT_DISCARD, false, 0,
       {0xaf, 0x02, 0x54, 0x00, 0x02, 0x00, 0xb0, 0x40, 0x00, 0x00}},
      {"log2 with no wait", false, 0, 0, AGX3_ALU_HINT_DISCARD, false, 0,
       {0x2f, 0x02, 0x54, 0x00, 0x02, 0x00, 0xb0, 0x40, 0x00, 0x00}},
      {"log2 of powr()", false, 4, 4, AGX3_ALU_HINT_DISCARD, true, 1u << 5,
       {0x2f, 0x02, 0x56, 0x04, 0x03, 0x08, 0xb0, 0x40, 0x00, 0x00}},
   };

   for (unsigned i = 0; i < sizeof(vectors) / sizeof(vectors[0]); ++i) {
      assert((vectors[i].exp2 ? agx3_alu_f32exp2 : agx3_alu_f32log2)(
         AGX_ISA_ENCODING_AGX3B, vectors[i].dest, vectors[i].src,
         vectors[i].hint, vectors[i].dest_cache, vectors[i].wait, bytes,
         &len));
      assert(len == sizeof(vectors[i].bytes));
      assert(!memcmp(bytes, vectors[i].bytes, len));
   }

   /* The half and the 32-bit form are distinct encodings, not aliases: the
    * two width selectors are the whole difference. */
   uint8_t half[AGX3_ALU_MAX_LENGTH], hlen;
   assert(agx3_alu_hlog2(AGX_ISA_ENCODING_AGX3B, 2, 2, AGX3_ALU_HINT_DISCARD,
                         false, 1u << 5, half, &hlen));
   assert(agx3_alu_f32log2(AGX_ISA_ENCODING_AGX3B, 2, 2, AGX3_ALU_HINT_DISCARD,
                           false, 1u << 5, bytes, &len));
   assert(hlen == len && memcmp(half, bytes, len) != 0);

   /* A 32-bit operand pair is aligned, only AGX3b is witnessed, and the wait
    * mask has six slots. */
   assert(!agx3_alu_f32log2(AGX_ISA_ENCODING_AGX3B, 1, 0, AGX3_ALU_HINT_NONE,
                            false, 0, bytes, &len));
   assert(!agx3_alu_f32exp2(AGX_ISA_ENCODING_AGX3B, 0, 1, AGX3_ALU_HINT_NONE,
                            false, 0, bytes, &len));
   assert(!agx3_alu_f32log2(AGX_ISA_ENCODING_AGX3C, 0, 0, AGX3_ALU_HINT_NONE,
                            false, 0, bytes, &len));
   assert(!agx3_alu_f32exp2(AGX_ISA_ENCODING_AGX3B, 0, 0, AGX3_ALU_HINT_NONE,
                            false, 1u << 6, bytes, &len));
}

/*
 * The source modifiers this whole form family carries: f31 negates the
 * source, f30 takes its absolute value. Every vector is Apple's own
 * applegpu_g17p machine code for the named expression, and every one differs
 * from the unmodified form in exactly those bits.
 *
 * exp(-x) is vkQuake's fog term. NIR folds the negation into the source, so
 * the packer saw `exp2 d32, r.neg` and refused every fragment shader in the
 * game.
 */
static void
test_native_f32_log2_exp2_modifiers(void)
{
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   static const struct {
      const char *name;
      bool exp2;
      bool neg, abs;
      uint8_t bytes[10];
   } vectors[] = {
      {"exp2(a[i])", true, false, false,
       {0xaf, 0x02, 0x56, 0x00, 0x02, 0x00, 0xb0, 0x40, 0x00, 0x00}},
      {"exp2(-a[i])", true, true, false,
       {0xaf, 0x02, 0x56, 0x00, 0x02, 0x00, 0xb0, 0x40, 0x01, 0x00}},
      {"exp2(fabs(a[i]))", true, false, true,
       {0xaf, 0x02, 0x56, 0x00, 0x02, 0x00, 0xb0, 0xc0, 0x00, 0x00}},
      {"exp2(-fabs(a[i]))", true, true, true,
       {0xaf, 0x02, 0x56, 0x00, 0x02, 0x00, 0xb0, 0xc0, 0x01, 0x00}},
      {"log2(a[i])", false, false, false,
       {0x2f, 0x02, 0x56, 0x00, 0x02, 0x00, 0xb0, 0x40, 0x00, 0x00}},
      {"log2(-a[i])", false, true, false,
       {0x2f, 0x02, 0x56, 0x00, 0x02, 0x00, 0xb0, 0x40, 0x01, 0x00}},
      {"log2(fabs(a[i]))", false, false, true,
       {0x2f, 0x02, 0x56, 0x00, 0x02, 0x00, 0xb0, 0xc0, 0x00, 0x00}},
      {"log2(-fabs(a[i]))", false, true, true,
       {0x2f, 0x02, 0x56, 0x00, 0x02, 0x00, 0xb0, 0xc0, 0x01, 0x00}},
   };

   for (unsigned i = 0; i < sizeof(vectors) / sizeof(vectors[0]); ++i) {
      assert((vectors[i].exp2 ? agx3_alu_f32exp2_mod : agx3_alu_f32log2_mod)(
         AGX_ISA_ENCODING_AGX3B, 0, 0, AGX3_ALU_HINT_DISCARD, false, 1u << 5,
         vectors[i].neg, vectors[i].abs, bytes, &len));
      assert(len == sizeof(vectors[i].bytes));
      assert(!memcmp(bytes, vectors[i].bytes, len));
   }

   /* The unmodified entry point and the modifier one agree when no modifier
    * is asked for, so nothing silently changed for existing callers. */
   uint8_t plain[AGX3_ALU_MAX_LENGTH], plen;
   assert(agx3_alu_f32exp2(AGX_ISA_ENCODING_AGX3B, 4, 6, AGX3_ALU_HINT_CACHE,
                           true, 1u << 2, plain, &plen));
   assert(agx3_alu_f32exp2_mod(AGX_ISA_ENCODING_AGX3B, 4, 6,
                               AGX3_ALU_HINT_CACHE, true, 1u << 2, false,
                               false, bytes, &len));
   assert(plen == len && !memcmp(plain, bytes, len));

   /* A negated source is a different instruction, not a no-op. */
   assert(agx3_alu_f32exp2_mod(AGX_ISA_ENCODING_AGX3B, 4, 6,
                               AGX3_ALU_HINT_CACHE, true, 1u << 2, true,
                               false, bytes, &len));
   assert(plen == len && memcmp(plain, bytes, len) != 0);
}

/*
 * Full-precision floor and ceiling. Apple's f32rint with f43 selecting the
 * mode and the width selectors raised from the half form. floor() on a float
 * is vkQuake's quantized lightmap coordinate; only the half form was wired
 * up, so world.frag refused.
 */
static void
test_native_f32_floor_ceil(void)
{
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;

   /* o[i] = floor(a[i]) -- source dead, waiting the load's slot 6. */
   assert(agx3_alu_f32floor(AGX_ISA_ENCODING_AGX3B, 0, 0,
                            AGX3_ALU_HINT_DISCARD, false, 1u << 5, false,
                            false, bytes, &len));
   assert(len == 10);
   assert(!memcmp(bytes,
                  "\x2f\x00\x56\x00\x02\x00\xb0\x40\x02\x00", 10));

   /* o[i] = ceil(a[i]) -- the same instruction with f43 = 2. */
   assert(agx3_alu_f32ceil(AGX_ISA_ENCODING_AGX3B, 0, 0,
                           AGX3_ALU_HINT_DISCARD, false, 1u << 5, false, false,
                           bytes, &len));
   assert(len == 10);
   assert(!memcmp(bytes,
                  "\x2f\x00\x56\x00\x02\x00\xb0\x40\x04\x00", 10));

   /* o[i] = floor(-a[i]) -- the family's f31 negate. */
   assert(agx3_alu_f32floor(AGX_ISA_ENCODING_AGX3B, 0, 0,
                            AGX3_ALU_HINT_DISCARD, false, 1u << 5, true, false,
                            bytes, &len));
   assert(len == 10);
   assert(!memcmp(bytes,
                  "\x2f\x00\x56\x00\x02\x00\xb0\x40\x03\x00", 10));

   /* float x = a[i]; o[i] = floor(x) + x -- the source is retained, and the
    * destination is cached for the add. This is the CACHE hint the half
    * encoder still refuses for want of a witness. */
   assert(agx3_alu_f32floor(AGX_ISA_ENCODING_AGX3B, 4, 0, AGX3_ALU_HINT_CACHE,
                            true, 1u << 5, false, false, bytes, &len));
   assert(len == 10);
   assert(!memcmp(bytes,
                  "\x2f\x00\x56\x04\x03\x00\x92\x40\x02\x00", 10));

   /* o[i] = floor(a[i]) * 4 + floor(b[i]) -- the second floor waits slot 1,
    * so the wait mask reaches further than the half form's slot 6. */
   assert(agx3_alu_f32floor(AGX_ISA_ENCODING_AGX3B, 4, 4,
                            AGX3_ALU_HINT_DISCARD, true, 1u << 0, false, false,
                            bytes, &len));
   assert(len == 10);
   assert(!memcmp(bytes,
                  "\x2f\x10\x54\x04\x03\x08\xb0\x40\x02\x00", 10));

   /* o[i] = ceil(fabs(a[i])) and floor(fabs(a[i])) -- the family's f30. */
   assert(agx3_alu_f32ceil(AGX_ISA_ENCODING_AGX3B, 0, 0,
                           AGX3_ALU_HINT_DISCARD, false, 1u << 5, false, true,
                           bytes, &len));
   assert(len == 10);
   assert(!memcmp(bytes,
                  "\x2f\x00\x56\x00\x02\x00\xb0\xc0\x04\x00", 10));
   assert(agx3_alu_f32floor(AGX_ISA_ENCODING_AGX3B, 0, 0,
                            AGX3_ALU_HINT_DISCARD, false, 1u << 5, false, true,
                            bytes, &len));
   assert(len == 10);
   assert(!memcmp(bytes,
                  "\x2f\x00\x56\x00\x02\x00\xb0\xc0\x02\x00", 10));

   /* float x = a[i]; o[i] = ceil(x) + x, and the second of two ceils. */
   assert(agx3_alu_f32ceil(AGX_ISA_ENCODING_AGX3B, 4, 0, AGX3_ALU_HINT_CACHE,
                           true, 1u << 5, false, false, bytes, &len));
   assert(len == 10);
   assert(!memcmp(bytes,
                  "\x2f\x00\x56\x04\x03\x00\x92\x40\x04\x00", 10));
   assert(agx3_alu_f32ceil(AGX_ISA_ENCODING_AGX3B, 4, 4,
                           AGX3_ALU_HINT_DISCARD, true, 1u << 0, false, false,
                           bytes, &len));
   assert(len == 10);
   assert(!memcmp(bytes,
                  "\x2f\x10\x54\x04\x03\x08\xb0\x40\x04\x00", 10));

   /* The lifetime and wait fields are modelled, not pinned. The shared funop
    * baseline (agx3_funop_core.c) folds f29 and f40 into its form constant,
    * so it cannot express a live source or an absent slot-6 wait; these
    * encoders can, which is why floor and ceil do not fall through to it. */
   uint8_t live[AGX3_ALU_MAX_LENGTH], llen;
   assert(agx3_alu_f32floor(AGX_ISA_ENCODING_AGX3B, 0, 0,
                            AGX3_ALU_HINT_DISCARD, false, 1u << 5, false,
                            false, live, &llen));
   assert(agx3_alu_f32floor(AGX_ISA_ENCODING_AGX3B, 0, 0, AGX3_ALU_HINT_NONE,
                            false, 0, false, false, bytes, &len));
   assert(llen == len && memcmp(live, bytes, len) != 0);

   /* The half and 32-bit forms are distinct encodings, not aliases. */
   uint8_t half[AGX3_ALU_MAX_LENGTH], hlen;
   assert(agx3_alu_hfloor(AGX_ISA_ENCODING_AGX3B, 2, 2, AGX3_ALU_HINT_DISCARD,
                          false, 1u << 5, half, &hlen));
   assert(agx3_alu_f32floor(AGX_ISA_ENCODING_AGX3B, 2, 2,
                            AGX3_ALU_HINT_DISCARD, false, 1u << 5, false,
                            false, bytes, &len));
   assert(hlen == len && memcmp(half, bytes, len) != 0);

   /* A 32-bit operand pair is aligned, only AGX3b is witnessed, and the wait
    * mask has six slots. */
   assert(!agx3_alu_f32floor(AGX_ISA_ENCODING_AGX3B, 1, 0, AGX3_ALU_HINT_NONE,
                             false, 0, false, false, bytes, &len));
   assert(!agx3_alu_f32ceil(AGX_ISA_ENCODING_AGX3B, 0, 1, AGX3_ALU_HINT_NONE,
                            false, 0, false, false, bytes, &len));
   assert(!agx3_alu_f32floor(AGX_ISA_ENCODING_AGX3C, 0, 0, AGX3_ALU_HINT_NONE,
                             false, 0, false, false, bytes, &len));
   assert(!agx3_alu_f32ceil(AGX_ISA_ENCODING_AGX3B, 0, 0, AGX3_ALU_HINT_NONE,
                            false, 1u << 6, false, false, bytes, &len));
}

static void
test_native_bit_andc(void)
{
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   const uint16_t dead_src[] = {0, 2};
   const enum agx3_alu_hint dead[] = {
      AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_NONE,
   };
   assert(agx3_alu_bit_andc(AGX_ISA_ENCODING_AGX3B, 0, dead_src, dead,
                            false, 32, bytes, &len));
   assert(len == 10 &&
          !memcmp(bytes,
                  (uint8_t[]){0x0b, 0x01, 0x0e, 0x03, 0x02,
                              0x00, 0x00, 0x80, 0x00, 0x00},
                  10));

   const uint16_t live_src[] = {2, 4};
   const enum agx3_alu_hint live[] = {
      AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_DISCARD,
   };
   assert(agx3_alu_bit_andc(AGX_ISA_ENCODING_AGX3B, 0, live_src, live,
                            false, 32, bytes, &len));
   assert(len == 10 &&
          !memcmp(bytes,
                  (uint8_t[]){0x0b, 0x03, 0x16, 0x05, 0x02,
                              0x00, 0x00, 0x80, 0x00, 0x00},
                  10));

   /* Exact vkQuake update_lightmap shape: r10 = $r10 & ~$r3. */
   const uint16_t quake_src[] = {20, 6};
   const enum agx3_alu_hint cached[] = {
      AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_CACHE,
   };
   assert(agx3_alu_bit_andc(AGX_ISA_ENCODING_AGX3B, 20, quake_src, cached,
                            false, 0, bytes, &len));
   assert(len == 10 &&
          !memcmp(bytes,
                  (uint8_t[]){0xab, 0x95, 0x06, 0x87, 0x02,
                              0x00, 0x00, 0x00, 0x00, 0x00},
                  10));
   assert(!agx3_alu_bit_andc(AGX_ISA_ENCODING_AGX3C, 20, quake_src, cached,
                             false, 0, bytes, &len));
}

static void
test_native_minifloat_half_select(void)
{
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   assert(agx3_alu_fcmp_minifloat_select_half(
      AGX_ISA_ENCODING_AGX3B, 0, 0, 176, 10, 0,
      AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD, 0, false, 32,
      bytes, &len));
   assert(len == 10 &&
          !memcmp(bytes,
                  (uint8_t[]){0x02, 0x01, 0x19, 0xb1, 0x86,
                              0x0a, 0x00, 0xc2, 0x20, 0x80},
                  10));

   /* Exact Quake form after normalizing its commutative #176,r15 compare. */
   assert(agx3_alu_fcmp_minifloat_select_half(
      AGX_ISA_ENCODING_AGX3B, 6, 30, 176, 6, 0,
      AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE, 2, true, 0, bytes, &len));
   assert(len == 10 &&
          !memcmp(bytes,
                  (uint8_t[]){0x32, 0x1f, 0x31, 0xb1, 0x02,
                              0x06, 0x02, 0x02, 0x20, 0x80},
                  10));
}

static void
check_minifloat_bits(bool multiply, unsigned dest, unsigned src, unsigned imm,
                     bool kill, unsigned wait)
{
   static const unsigned wait_decode[8] = {0, 1, 2, 4, 8, 16, 32, 3};
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
   const bool encoded = multiply
      ? agx3_alu_fmul_minifloat(AGX_ISA_ENCODING_AGX3B, dest, src, imm,
                               kill, wait, bytes, &len)
      : agx3_alu_fadd_minifloat(AGX_ISA_ENCODING_AGX3B, dest, src, imm,
                               false, kill, wait, bytes, &len);
   assert(encoded);
   assert(len == 6);
   uint64_t bits = 0;
   for (unsigned i = 0; i < len; ++i)
      bits |= (uint64_t)bytes[i] << (8 * i);
   assert(((bits >> 16) & 7) == (multiply ? 5u : 4u));

   /* Independent physical bit extraction; no composer/decoder round trip. */
   assert((((bits >> 34) & 1) | (((bits >> 4) & 15) << 1) |
           (((bits >> 22) & 3) << 5) | (((bits >> 44) & 1) << 7)) == dest);
   assert((((bits >> 36) & 1) | (((bits >> 25) & 63) << 1) |
           (((bits >> 42) & 1) << 7)) == src);
   assert((((bits >> 35) & 1) | (((bits >> 9) & 63) << 1) |
           (((bits >> 19) & 1) << 7)) == imm);
   assert(((bits >> 20) & 1) == kill);
   assert(wait_decode[(bits >> 45) & 7] == wait);
   assert(((bits >> 3) & 1) && ((bits >> 8) & 1) &&
          ((bits >> 24) & 1)); /* f32 widths */
   assert(((bits >> 15) & 1) && ((bits >> 39) & 1)); /* minifloat */
   assert(!(bits & ((UINT64_C(1) << 37) | (UINT64_C(1) << 38) |
                    (UINT64_C(1) << 41) | (UINT64_C(1) << 43))));
}

static void
test_fadd_minifloat(void)
{
   /* Fresh Apple -fno-fast-math witnesses: real loaded float producer,
    * neighboring immediates, negative immediate, and positively reused input.
    */
   static const struct {
      unsigned dest, src, imm, dest_cache, hint, wait;
      uint8_t bytes[6];
   } witness[] = {
      {0, 0, 0x20, 0, AGX3_ALU_HINT_DISCARD, 0,
       {0x09, 0xa1, 0x14, 0x01, 0x80, 0x00}},
      {2, 2, 0x20, 0, AGX3_ALU_HINT_DISCARD, 0,
       {0x19, 0xa1, 0x14, 0x03, 0x80, 0x00}},
      {0, 0, 0x20, 0, AGX3_ALU_HINT_DISCARD, 32,
       {0x09, 0xa1, 0x14, 0x01, 0x80, 0xc0}},
      {0, 0, 0x1f, 0, AGX3_ALU_HINT_DISCARD, 32,
       {0x09, 0x9f, 0x14, 0x01, 0x88, 0xc0}},
      {0, 0, 0x21, 0, AGX3_ALU_HINT_DISCARD, 32,
       {0x09, 0xa1, 0x14, 0x01, 0x88, 0xc0}},
      {0, 0, 0xa0, 0, AGX3_ALU_HINT_DISCARD, 32,
       {0x09, 0xa1, 0x1c, 0x01, 0x80, 0xc0}},
      {0, 2, 0x20, 0, AGX3_ALU_HINT_NONE, 32,
       {0x09, 0xa1, 0x04, 0x03, 0x80, 0xc0}},
      /* applegpu_g17p pair.fragment.bin SHA256
       * d38342dd219c2e9b5d9ee2ffc44beb185929fdaf994ac67a64099127ab3fc428:
       * a loaded float is retained across x - 0.5, and the result is reused.
       * Apple keeps both lifetimes and the producer wait on this one ADD. */
      {0, 4, 0xa0, 1, AGX3_ALU_HINT_CACHE, 32,
       {0x09, 0xa1, 0x2c, 0x85, 0x80, 0xc0}},
   };
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
   for (unsigned i = 0; i < sizeof(witness) / sizeof(witness[0]); ++i) {
      assert(agx3_alu_fadd_minifloat(AGX_ISA_ENCODING_AGX3B,
                                    witness[i].dest, witness[i].src,
                                    witness[i].imm, witness[i].dest_cache,
                                    witness[i].hint,
                                    witness[i].wait, bytes, &len));
      assert(len == 6 && !memcmp(bytes, witness[i].bytes, 6));
   }
   const unsigned waits[] = {0, 1, 2, 4, 8, 16, 32, 3};
   for (unsigned dest = 0; dest < 256; dest += 2)
      for (unsigned src = 0; src < 256; src += 2)
         for (unsigned w = 0; w < 8; ++w)
            for (unsigned kill = 0; kill < 2; ++kill)
               check_minifloat_bits(false, dest, src, 0x20, kill, waits[w]);
   for (unsigned imm = 0; imm < 256; ++imm)
      for (unsigned w = 0; w < 8; ++w)
         check_minifloat_bits(false, 254, 128, imm, true, waits[w]);

   /* Pixel-centre arithmetic and adjacent minifloats are exact across the
    * complete U16 pixel coordinate range. This checks the mathematical
    * contract, not a claim of executing the instruction on GPU hardware.
    */
   const unsigned numerator[] = {31, 32, 34};
   for (unsigned i = 0; i < 3; ++i) {
      for (uint32_t u = 0; u <= 65535; ++u) {
         uint32_t n = u * 64 + numerator[i];
         unsigned bit = 0;
         while ((n >> (bit + 1)) != 0)
            ++bit;
         uint32_t want = ((121 + bit) << 23) |
                         ((n - (1u << bit)) << (23 - bit));
         float f = (float)u + (float)numerator[i] / 64.0f;
         uint32_t got;
         memcpy(&got, &f, sizeof(got));
         assert(got == want);
      }
   }
   /* Normal IEEE nearest-even addition, including ties where a truncating
    * interpretation would differ. The -fno-fast-math native probes select
    * this ordinary f32add; no rounding/conversion modifier is synthesized.
    */
   const uint32_t tie_input[] = {0x4b000000, 0x4b000001, 0x4b7fffff};
   const uint32_t tie_result[] = {0x4b000000, 0x4b000002, 0x4b800000};
   for (unsigned i = 0; i < 3; ++i) {
      float input;
      memcpy(&input, &tie_input[i], sizeof(input));
      volatile float live = input;
      float result = live + 0.5f;
      uint32_t bits;
      memcpy(&bits, &result, sizeof(bits));
      assert(bits == tie_result[i]);
   }
   for (unsigned wait = 0; wait < 64; ++wait) {
      bool supported = wait == 0 || wait == 3 || !(wait & (wait - 1));
      memset(bytes, 0xa5, sizeof(bytes));
      len = 99;
      assert(agx3_alu_fadd_minifloat(AGX_ISA_ENCODING_AGX3B, 0, 2, 0x20,
                                    false, AGX3_ALU_HINT_DISCARD, wait,
                                    bytes, &len) == supported);
      if (!supported)
         assert(len == 99 && bytes[0] == 0xa5);
   }
   /* Independently inspected AGX3c compact ADD retains these exact operand
    * and wait bits, while requiring its own dispatch/opcode fields. */
   const uint8_t m5_pixel_centre[] = {0x19,0xa1,0x14,0x07,0x80,0x00};
   assert(agx3_alu_fadd_minifloat(AGX_ISA_ENCODING_AGX3C, 2, 6, 32,
                                 false, AGX3_ALU_HINT_DISCARD, 0, bytes, &len));
   assert(len == sizeof(m5_pixel_centre) && !memcmp(bytes,m5_pixel_centre,len));
   for (unsigned i = 0; i < sizeof(witness) / sizeof(witness[0]); ++i) {
      assert(agx3_alu_fadd_minifloat(AGX_ISA_ENCODING_AGX3C,
                                    witness[i].dest, witness[i].src,
                                    witness[i].imm, witness[i].dest_cache,
                                    witness[i].hint, witness[i].wait, bytes, &len));
      assert(len == 6 && !memcmp(bytes, witness[i].bytes, 6));
   }
   for (unsigned wait = 0; wait < 256; ++wait) {
      bool supported = wait < 64 && (wait == 0 || wait == 3 || !(wait & (wait - 1)));
      assert(agx3_alu_fadd_minifloat(AGX_ISA_ENCODING_AGX3C, 0, 2, 32,
                                    false, AGX3_ALU_HINT_DISCARD, wait,
                                    bytes, &len) == supported);
   }
   assert(!agx3_alu_fadd_minifloat(AGX_ISA_ENCODING_AGX3B, 1, 0, 32,
                                  false, AGX3_ALU_HINT_NONE, 0, bytes, &len));
   assert(!agx3_alu_fadd_minifloat(AGX_ISA_ENCODING_AGX3B, 0, 1, 32,
                                  false, AGX3_ALU_HINT_NONE, 0, bytes, &len));
   assert(!agx3_alu_fadd_minifloat(AGX_ISA_ENCODING_AGX3B, 256, 0, 32,
                                  false, AGX3_ALU_HINT_NONE, 0, bytes, &len));
   assert(!agx3_alu_fadd_minifloat(AGX_ISA_ENCODING_AGX3B, 0, 256, 32,
                                  false, AGX3_ALU_HINT_NONE, 0, bytes, &len));
   assert(!agx3_alu_fadd_minifloat(AGX_ISA_ENCODING_AGX3B, 0, 0, 256,
                                  false, AGX3_ALU_HINT_NONE, 0, bytes, &len));
   assert(!agx3_alu_fadd_minifloat(AGX_ISA_ENCODING_AGX3B, 0, 0, 32,
                                  false, AGX3_ALU_HINT_NONE, 64, bytes, &len));
   assert(!agx3_alu_fadd_minifloat(AGX_ISA_ENCODING_AGX3B, 0, 0, 32,
                                  false, AGX3_ALU_HINT_NONE, 0, NULL, &len));
   assert(!agx3_alu_fadd_minifloat(AGX_ISA_ENCODING_AGX3B, 0, 0, 32,
                                  false, AGX3_ALU_HINT_NONE, 0, bytes, NULL));
   puts("G17P fadd minifloat: 8 Apple witnesses, register/wait/liveness bits,"
        " all256 immediates, all65536 pixel centres and adjacent values PASS");
}

static void
test_fmul_minifloat(void)
{
   static const struct {
      unsigned dest, src, imm, kill, wait;
      uint8_t bytes[6];
   } witness[] = {
      {0, 0, 0x02, 1, 0,  {0x09, 0x83, 0x15, 0x01, 0x80, 0x00}},
      {2, 2, 0x02, 1, 0,  {0x19, 0x83, 0x15, 0x03, 0x80, 0x00}},
      {0, 0, 0x02, 1, 32, {0x09, 0x83, 0x15, 0x01, 0x80, 0xc0}},
      {0, 0, 0x01, 1, 32, {0x09, 0x81, 0x15, 0x01, 0x88, 0xc0}},
      {0, 0, 0x03, 1, 32, {0x09, 0x83, 0x15, 0x01, 0x88, 0xc0}},
      {0, 0, 0x82, 1, 32, {0x09, 0x83, 0x1d, 0x01, 0x80, 0xc0}},
      {0, 2, 0x02, 0, 32, {0x09, 0x83, 0x05, 0x03, 0x80, 0xc0}},
   };
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
   for (unsigned i = 0; i < sizeof(witness) / sizeof(witness[0]); ++i) {
      assert(agx3_alu_fmul_minifloat(AGX_ISA_ENCODING_AGX3B,
                                    witness[i].dest, witness[i].src,
                                    witness[i].imm, witness[i].kill,
                                    witness[i].wait, bytes, &len));
      assert(len == 6 && !memcmp(bytes, witness[i].bytes, 6));
   }
   const unsigned waits[] = {0, 1, 2, 4, 8, 16, 32, 3};
   for (unsigned dest = 0; dest < 256; dest += 2)
      for (unsigned src = 0; src < 256; src += 2)
         for (unsigned w = 0; w < 8; ++w)
            for (unsigned kill = 0; kill < 2; ++kill)
               check_minifloat_bits(true, dest, src, 0x02, kill, waits[w]);
   for (unsigned imm = 0; imm < 256; ++imm)
      for (unsigned w = 0; w < 8; ++w)
         check_minifloat_bits(true, 254, 128, imm, true, waits[w]);

   /* Complete pixel-centre range times1/64,1/32,3/64. The result is the
    * exact rational (2*u+1)*scale/128, independently normalized below.
    */
   for (unsigned scale = 1; scale <= 3; ++scale) {
      for (uint32_t u = 0; u <= 65535; ++u) {
         uint32_t n = (2 * u + 1) * scale;
         unsigned bit = 0;
         while ((n >> (bit + 1)) != 0)
            ++bit;
         uint32_t want = ((120 + bit) << 23) |
                         ((n - (1u << bit)) << (23 - bit));
         float f = ((float)u + 0.5f) * ((float)scale / 64.0f);
         uint32_t got;
         memcpy(&got, &f, sizeof(got));
         assert(got == want);
      }
   }
   const uint32_t tie_input[] = {0x3f800001, 0x3f800003};
   const uint32_t tie_result[] = {0x3fc00002, 0x3fc00004};
   for (unsigned i = 0; i < 2; ++i) {
      float input;
      memcpy(&input, &tie_input[i], sizeof(input));
      volatile float live = input;
      float result = live * 1.5f;
      uint32_t bits;
      memcpy(&bits, &result, sizeof(bits));
      assert(bits == tie_result[i]);
   }
   for (unsigned wait = 0; wait < 64; ++wait) {
      bool supported = wait == 0 || wait == 3 || !(wait & (wait - 1));
      memset(bytes, 0xa5, sizeof(bytes));
      len = 99;
      assert(agx3_alu_fmul_minifloat(AGX_ISA_ENCODING_AGX3B, 0, 2, 2, true,
                                    wait, bytes, &len) == supported);
      if (!supported)
         assert(len == 99 && bytes[0] == 0xa5);
   }
   assert(!agx3_alu_fmul_minifloat(AGX_ISA_ENCODING_AGX3C, 0, 0, 2, 0,
                                  0, bytes, &len));
   assert(!agx3_alu_fmul_minifloat(AGX_ISA_ENCODING_AGX3B, 1, 0, 2, 0,
                                  0, bytes, &len));
   assert(!agx3_alu_fmul_minifloat(AGX_ISA_ENCODING_AGX3B, 0, 1, 2, 0,
                                  0, bytes, &len));
   assert(!agx3_alu_fmul_minifloat(AGX_ISA_ENCODING_AGX3B, 256, 0, 2, 0,
                                  0, bytes, &len));
   assert(!agx3_alu_fmul_minifloat(AGX_ISA_ENCODING_AGX3B, 0, 256, 2, 0,
                                  0, bytes, &len));
   assert(!agx3_alu_fmul_minifloat(AGX_ISA_ENCODING_AGX3B, 0, 0, 256, 0,
                                  0, bytes, &len));
   assert(!agx3_alu_fmul_minifloat(AGX_ISA_ENCODING_AGX3B, 0, 0, 2, 0,
                                  64, bytes, &len));
   assert(!agx3_alu_fmul_minifloat(AGX_ISA_ENCODING_AGX3B, 0, 0, 2, 0,
                                  0, NULL, &len));
   assert(!agx3_alu_fmul_minifloat(AGX_ISA_ENCODING_AGX3B, 0, 0, 2, 0,
                                  0, bytes, NULL));
   puts("G17P fmul minifloat: 7 Apple witnesses, register/wait/liveness bits,"
        " all256 immediates, all65536 scaled pixel centres PASS");
}

static void
test_fmul_minifloat_modifiers(void)
{
   static const struct {
      uint16_t src, imm;
      bool dest_cache;
      enum agx3_alu_hint hint;
      uint16_t wait;
      uint8_t bytes[6];
   } native[] = {
      {0, 0x40, false, AGX3_ALU_HINT_DISCARD, 0,
       {0x09, 0xc1, 0x15, 0x01, 0x82, 0x00}},
      {0, 0x20, false, AGX3_ALU_HINT_DISCARD, 0,
       {0x09, 0xa1, 0x15, 0x01, 0x82, 0x00}},
      {0, 0x38, false, AGX3_ALU_HINT_DISCARD, 0,
       {0x09, 0xb9, 0x15, 0x01, 0x82, 0x00}},
      {0, 0xc0, false, AGX3_ALU_HINT_DISCARD, 0,
       {0x09, 0xc1, 0x1d, 0x01, 0x82, 0x00}},
      {0, 0x40, true, AGX3_ALU_HINT_DISCARD, 1,
       {0x09, 0xc1, 0x35, 0x01, 0x82, 0x20}},
      {2, 0x40, false, AGX3_ALU_HINT_CACHE, 0,
       {0x09, 0xc1, 0x05, 0x83, 0x82, 0x00}},
   };
   uint8_t out[AGX3_ALU_MAX_LENGTH], len;
   for (unsigned i = 0; i < sizeof(native) / sizeof(native[0]); ++i) {
      assert(agx3_alu_fmul_minifloat_mod(
         AGX_ISA_ENCODING_AGX3B, 0, native[i].src, native[i].imm,
         native[i].dest_cache, true, native[i].hint,
         native[i].wait, out, &len));
      assert(len == sizeof(native[i].bytes));
      assert(!memcmp(out, native[i].bytes, len));
   }

   unsigned matrix = 0;
   for (unsigned imm = 0; imm <= UINT8_MAX; ++imm) {
      for (unsigned dest_cache = 0; dest_cache < 2; ++dest_cache) {
         for (unsigned hint = AGX3_ALU_HINT_NONE;
              hint <= AGX3_ALU_HINT_CACHE; ++hint) {
            for (unsigned wait = 0; wait < 64; ++wait) {
               uint8_t plain[AGX3_ALU_MAX_LENGTH], absolute[AGX3_ALU_MAX_LENGTH];
               uint8_t plain_len = 0xa5, absolute_len = 0xa5;
               bool plain_ok = agx3_alu_fmul_minifloat_mod(
                  AGX_ISA_ENCODING_AGX3B, 254, 128, imm, dest_cache,
                  false, hint, wait, plain, &plain_len);
               bool absolute_ok = agx3_alu_fmul_minifloat_mod(
                  AGX_ISA_ENCODING_AGX3B, 254, 128, imm, dest_cache,
                  true, hint, wait, absolute, &absolute_len);
               assert(plain_ok == absolute_ok);
               if (!plain_ok)
                  continue;
               assert(plain_len == 6 && absolute_len == plain_len);
               assert(!(plain[4] & 0x02) && (absolute[4] & 0x02));
               absolute[4] &= ~0x02;
               assert(!memcmp(plain, absolute, plain_len));
               matrix++;
            }
         }
      }
   }

   memset(out, 0xa5, sizeof(out));
   len = 99;
   assert(!agx3_alu_fmul_minifloat_mod(
      AGX_ISA_ENCODING_AGX3C, 0, 0, 0x40, false, true,
      AGX3_ALU_HINT_DISCARD, 0, out, &len));
   assert(len == 99 && out[0] == 0xa5);
   assert(!agx3_alu_fmul_minifloat_mod(
      AGX_ISA_ENCODING_AGX3B, 1, 0, 0x40, false, true,
      AGX3_ALU_HINT_DISCARD, 0, out, &len));
   assert(!agx3_alu_fmul_minifloat_mod(
      AGX_ISA_ENCODING_AGX3B, 0, 1, 0x40, false, true,
      AGX3_ALU_HINT_DISCARD, 0, out, &len));
   assert(!agx3_alu_fmul_minifloat_mod(
      AGX_ISA_ENCODING_AGX3B, 0, 0, 0x100, false, true,
      AGX3_ALU_HINT_DISCARD, 0, out, &len));

   printf("G17P absolute minifloat FMUL: five direct and one texture-wait "
          "Apple witnesses, %u orthogonal immediate/lifetime/wait forms and "
          "strict negatives PASS\n", matrix);
}

static void test_fmul_half_minifloat_to_f32(void)
{
   static const uint8_t native[] = {0x09, 0xf1, 0x35, 0x00, 0x80, 0x00};
   uint8_t out[AGX3_ALU_MAX_LENGTH], len;
   assert(agx3_alu_fmul_half_minifloat_to_f32(
      AGX_ISA_ENCODING_AGX3B, 0, 0, 112, AGX3_ALU_HINT_NONE, 0, out,
      &len));
   assert(len == sizeof(native) && !memcmp(out, native, len));
   assert(!agx3_alu_fmul_half_minifloat_to_f32(
      AGX_ISA_ENCODING_AGX3C, 2, 0, 112, AGX3_ALU_HINT_DISCARD, 0, out,
      &len));
   assert(!agx3_alu_fmul_half_minifloat_to_f32(
      AGX_ISA_ENCODING_AGX3B, 3, 0, 112, AGX3_ALU_HINT_DISCARD, 0, out,
      &len));
   assert(!agx3_alu_fmul_half_minifloat_to_f32(
      AGX_ISA_ENCODING_AGX3B, 2, 0, 112, AGX3_ALU_HINT_CACHE, 0, out,
      &len));
   assert(!agx3_alu_fmul_half_minifloat_to_f32(
      AGX_ISA_ENCODING_AGX3B, 2, 0, 112, AGX3_ALU_HINT_DISCARD, 1, out,
      &len));
   puts("Native half times minifloat to f32: exact Apple vector and "
        "refusals PASS");
}

static void test_u16_to_f32(void)
{
   /* Literal applegpu-nt32023 G17P ushort/ushort2/reused-ushort witnesses.
    * Each source comes from an actual16-bit LDDEV, then a float32 STDEV.
    */
   static const struct { unsigned d,s,kill,wait; uint8_t bytes[8]; } v[] = {
      {0,0,1,32,{0xa7,0x07,0x56,0x00,0x02,0x00,0xa4,0x20}},
      {0,2,1,32,{0xa7,0x07,0x56,0x00,0x02,0x04,0xa4,0x20}},
      {2,3,1,0, {0xa7,0x07,0x54,0x02,0x02,0x06,0xa4,0x20}},
      {2,4,0,32,{0xa7,0x07,0x56,0x02,0x02,0x08,0x84,0x20}},
   };
   uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
   for(unsigned i=0;i<sizeof(v)/sizeof(v[0]);++i) {
      assert(agx3_alu_u16_to_f32(AGX_ISA_ENCODING_AGX3B,v[i].d,v[i].s,
                                v[i].kill,v[i].wait,bytes,&len));
      assert(len==8 && !memcmp(bytes,v[i].bytes,8));
   }
   /* Independent literal bit decode: register halves, unsigned16 width,
    * source kill and all six wait bits. Adjacent source half is not consumed.
    */
   for(unsigned d=0;d<128;d+=2) for(unsigned s=0;s<128;++s)
      for(unsigned k=0;k<2;++k) for(unsigned w=0;w<64;++w) {
         assert(agx3_alu_u16_to_f32(AGX_ISA_ENCODING_AGX3B,d,s,k,w,bytes,&len));
         assert(len==8 && bytes[3]==d && bytes[5]==(s<<1));
         assert((bytes[6]&0x18)==0); /* source-width LUT[1]=0, U16 */
         assert(((bytes[6]>>5)&1)==k);
         assert((bytes[7]&0x40)==0); /* unsigned */
         assert((unsigned)((bytes[1]>>4)|((bytes[2]&3)<<4))==w);
      }
   /* Every U16 has an exact IEEE754f32 representation (<=16 significantbits).
    * Compare independent integer normalization to the C conversion over the
    * full domain; RTE/RTZ have no rounding divergence in this exact range.
    */
   for(uint32_t u=0;u<=65535;++u) {
      uint32_t expected=0;
      if(u) {
         unsigned bit=0;while((u>>(bit+1))!=0)++bit;
         expected=((127+bit)<<23)|((u-(1u<<bit))<<(23-bit));
      }
      float f=(float)(uint16_t)u;uint32_t actual;memcpy(&actual,&f,4);
      assert(actual==expected);
   }
   /* Independent AGX3c table compositions for the Zink copy shader's cast,
    * an odd source half, and signed conversion waiting on producer slot8.
    * These are encoding regressions, not hardware numerical witnesses. */
   const uint8_t m5_cast[] = {0x37,0x02,0x00,0x02,0x22,0x80,0xaf,0x0a,0x20,0x00};
   const uint8_t m5_odd[] = {0x27,0x03,0x00,0x02,0x22,0x80,0xaf,0x82,0x20,0x00};
   const uint8_t m5_signed[] = {0x3f,0x03,0x00,0x02,0x22,0x80,0xaf,0x02,0x60,0x00};
   assert(agx3_alu_u16_to_f32(AGX_ISA_ENCODING_AGX3C,6,2,
                             AGX3_ALU_HINT_DISCARD,0,bytes,&len));
   assert(len == sizeof(m5_cast) && !memcmp(bytes,m5_cast,len));
   assert(agx3_alu_u16_to_f32(AGX_ISA_ENCODING_AGX3C,0,3,
                             AGX3_ALU_HINT_DISCARD,32,bytes,&len));
   assert(len == sizeof(m5_odd) && !memcmp(bytes,m5_odd,len));
   assert(agx3_alu_s16_to_f32(AGX_ISA_ENCODING_AGX3C,2,3,
                             AGX3_ALU_HINT_DISCARD,128,bytes,&len));
   assert(len == sizeof(m5_signed) && !memcmp(bytes,m5_signed,len));
   assert(!agx3_alu_u16_to_f32(AGX_ISA_ENCODING_AGX3C,0,0,0,256,bytes,&len));
   assert(!agx3_alu_u16_to_f32(AGX_ISA_ENCODING_AGX2,0,0,0,0,bytes,&len));
   assert(!agx3_alu_u16_to_f32(AGX_ISA_ENCODING_AGX3B,1,0,0,0,bytes,&len));
   assert(!agx3_alu_u16_to_f32(AGX_ISA_ENCODING_AGX3B,128,0,0,0,bytes,&len));
   assert(!agx3_alu_u16_to_f32(AGX_ISA_ENCODING_AGX3B,0,128,0,0,bytes,&len));
   assert(!agx3_alu_u16_to_f32(AGX_ISA_ENCODING_AGX3B,0,0,0,64,bytes,&len));
   assert(!agx3_alu_u16_to_f32(AGX_ISA_ENCODING_AGX3B,0,0,0,0,NULL,&len));
   assert(!agx3_alu_u16_to_f32(AGX_ISA_ENCODING_AGX3B,0,0,0,0,bytes,NULL));
   puts("G17P U16->F32: Apple witnesses,1048576 register/kill/wait encodings, all65536 exact numeric values PASS");
}
static void exact(const uint8_t *bytes, uint8_t length, const char *hex)
{
   assert(strlen(hex) == (size_t)length*2);
   for (unsigned i=0;i<length;++i) {
      unsigned value;
      assert(sscanf(hex+2*i,"%2x",&value)==1);
      if (bytes[i] != value) {
         fprintf(stderr, "expected %s, got ", hex);
         for (unsigned j=0;j<length;++j) fprintf(stderr, "%02x", bytes[j]);
         fputc('\n', stderr);
         assert(bytes[i] == value);
      }
   }
}

static void
test_modified_minifloat_add(void)
{
   /* Fresh Apple Metal 32023.917, applegpu_g17p: sat_only, complement,
    * sat_add, sat_neg, complement_reuse, sat_reuse, sat_pair, cache_add.
    * Source and native objects are retained in .tmp/g17p-alpha-20260905.
    */
   const struct {
      unsigned dest, src, imm;
      bool neg, sat;
      enum agx3_alu_hint hint;
      unsigned wait;
      const char *hex;
   } cases[] = {
      {0, 0, 0x80, false, true, AGX3_ALU_HINT_DISCARD, 32, "09011c8101020082"},
      {0, 0, 0x30, true, false, AGX3_ALU_HINT_DISCARD, 32, "09b1140180c8"},
      {0, 0, 0x20, false, true, AGX3_ALU_HINT_DISCARD, 32, "09a1140181000082"},
      {0, 0, 0x30, true, true, AGX3_ALU_HINT_DISCARD, 32, "09b1140181080082"},
      {0, 2, 0x30, true, false, AGX3_ALU_HINT_NONE, 32, "09b1040380c8"},
      {0, 2, 0x80, false, true, AGX3_ALU_HINT_NONE, 32, "0903148101020082"},
      {2, 2, 0x80, false, true, AGX3_ALU_HINT_DISCARD, 0, "19031c8101020002"},
      {0, 6, 0x80, false, true, AGX3_ALU_HINT_CACHE, 32, "0987148101020082"},
      {2, 6, 0x30, true, false, AGX3_ALU_HINT_CACHE, 0, "19b104878008"},
   };
   uint8_t out[AGX3_ALU_MAX_LENGTH], len;
   for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
      assert(agx3_alu_fadd_minifloat_mod(
         AGX_ISA_ENCODING_AGX3B, cases[i].dest, cases[i].src, cases[i].imm,
         cases[i].neg, cases[i].sat, cases[i].hint, cases[i].wait, out, &len));
      exact(out, len, cases[i].hex);
   }

   /* Fresh Metal 32023.917 G17P fragments for 1.0f - abs(x), with x dead
    * and then retained for another output channel. */
   assert(agx3_alu_fadd_one_absneg(AGX_ISA_ENCODING_AGX3B, 0, 0, true,
                                   AGX3_ALU_HINT_DISCARD, 0, out, &len));
   exact(out, len, "09b13401820800000200");
   assert(agx3_alu_fadd_one_absneg(AGX_ISA_ENCODING_AGX3B, 2, 0, true,
                                   AGX3_ALU_HINT_CACHE, 0, out, &len));
   exact(out, len, "19b12481820800000200");
   assert(!agx3_alu_fadd_one_absneg(AGX_ISA_ENCODING_AGX3C, 0, 0, false,
                                    AGX3_ALU_HINT_NONE, 0, out, &len));
   assert(agx3_alu_fadd_abs_negzero(AGX_ISA_ENCODING_AGX3B, 8, 2, true,
                                    AGX3_ALU_HINT_CACHE, 1, out, &len));
   exact(out, len, "49812c83822000000200");
   assert(!agx3_alu_fadd_abs_negzero(AGX_ISA_ENCODING_AGX3C, 8, 2, true,
                                     AGX3_ALU_HINT_CACHE, 1, out, &len));

   static const struct {
      uint16_t imm;
      bool neg, sat;
      uint16_t wait;
      const char *hex;
   } abs_add[] = {
      {0x20, false, false, 32, "09a11401820000800200"},
      {0x38, false, false, 32, "09b91401820000800200"},
      {0x40, false, false, 32, "09c11401820000800200"},
      {0x20, true,  false, 32, "09a11401820800800200"},
      {0x20, false, true,  32, "09a11401820000820200"},
   };
   for (unsigned i = 0; i < sizeof(abs_add) / sizeof(abs_add[0]); ++i) {
      assert(agx3_alu_fadd_abs_minifloat(
         AGX_ISA_ENCODING_AGX3B, 0, 0, abs_add[i].imm, false,
         abs_add[i].neg, abs_add[i].sat, AGX3_ALU_HINT_DISCARD,
         abs_add[i].wait,
         out, &len));
      exact(out, len, abs_add[i].hex);
   }

   unsigned abs_shapes = 0;
   for (unsigned imm = 0; imm <= UINT8_MAX; ++imm)
      for (unsigned dest_cache = 0; dest_cache < 2; ++dest_cache)
         for (unsigned neg = 0; neg < 2; ++neg)
            for (unsigned sat = 0; sat < 2; ++sat)
               for (unsigned hint = AGX3_ALU_HINT_NONE;
                    hint <= AGX3_ALU_HINT_CACHE; ++hint)
                  for (unsigned wait = 0; wait < 64; ++wait) {
                     assert(agx3_alu_fadd_abs_minifloat(
                        AGX_ISA_ENCODING_AGX3B, 254, 128, imm,
                        dest_cache, neg, sat, hint, wait, out, &len));
                     assert(len == 10);
                     abs_shapes++;
                  }
   memset(out, 0xa5, sizeof(out));
   len = 0xa5;
   assert(!agx3_alu_fadd_abs_minifloat(
      AGX_ISA_ENCODING_AGX3C, 0, 0, 0x20, false, false, false,
      AGX3_ALU_HINT_NONE, 0, out, &len));
   assert(len == 0xa5 && out[0] == 0xa5);
   assert(!agx3_alu_fadd_abs_minifloat(
      AGX_ISA_ENCODING_AGX3B, 1, 0, 0x20, false, false, false,
      AGX3_ALU_HINT_NONE, 0, out, &len));
   assert(!agx3_alu_fadd_abs_minifloat(
      AGX_ISA_ENCODING_AGX3B, 0, 1, 0x20, false, false, false,
      AGX3_ALU_HINT_NONE, 0, out, &len));
   assert(!agx3_alu_fadd_abs_minifloat(
      AGX_ISA_ENCODING_AGX3B, 0, 0, 0x100, false, false, false,
      AGX3_ALU_HINT_NONE, 0, out, &len));

   const unsigned regs[] = {0, 2, 30, 62, 126, 128, 254};
   const unsigned immediates[] = {0, 0x20, 0x30, 0x80, 0xb0, 0xff};
   unsigned shapes = 0;
   for (unsigned d = 0; d < sizeof(regs) / sizeof(regs[0]); ++d)
      for (unsigned r = 0; r < sizeof(regs) / sizeof(regs[0]); ++r)
         for (unsigned k = 0; k < sizeof(immediates) / sizeof(immediates[0]); ++k)
            for (unsigned neg = 0; neg < 2; ++neg)
               for (unsigned hint = 0; hint <= AGX3_ALU_HINT_CACHE; ++hint)
                  for (unsigned wait = 0; wait < 64; ++wait) {
                     const unsigned imm = immediates[k];
                     const bool swap = imm == 0x80 && !neg;
                     assert(agx3_alu_fadd_minifloat_mod(
                        AGX_ISA_ENCODING_AGX3B, regs[d], regs[r], imm, neg,
                        true, hint, wait, out, &len));
                     assert(len == 8);
                     uint64_t bits = 0;
                     for (unsigned i = 0; i < len; ++i)
                        bits |= (uint64_t)out[i] << (i * 8);
                     /* Physical bit extraction independent of the composer. */
                     assert((((bits >> 34) & 1) | (((bits >> 4) & 15) << 1) |
                             (((bits >> 22) & 3) << 5) |
                             (((bits >> 44) & 1) << 7)) == regs[d]);
                     const unsigned low = swap ? 35 : 36;
                     const unsigned middle = swap ? 9 : 25;
                     const unsigned high = swap ? 40 : 42;
                     assert((((bits >> low) & 1) |
                             (((bits >> middle) & 63) << 1) |
                             (((bits >> high) & 1) << 7)) == regs[r]);
                     assert((((bits >> (swap ? 36 : 35)) & 1) |
                             (((bits >> (swap ? 25 : 9)) & 63) << 1) |
                             (((bits >> (swap ? 20 : 19)) & 1) << 7)) == imm);
                     assert(((bits >> (swap ? 15 : 31)) & 1) ==
                            (hint == AGX3_ALU_HINT_CACHE));
                     assert(((bits >> (swap ? 19 : 20)) & 1) ==
                            (hint == AGX3_ALU_HINT_DISCARD));
                     assert(((bits >> 43) & 1) == neg);
                     assert((bits >> 57) & 1);
                     assert((((bits >> 45) & 7) | (((bits >> 61) & 7) << 3)) == wait);
                     ++shapes;
                  }
   assert(!agx3_alu_fadd_minifloat_mod(AGX_ISA_ENCODING_AGX3C, 0, 0, 0x80,
                                     false, true, 0, 0, out, &len));
   assert(!agx3_alu_fadd_minifloat_mod(AGX_ISA_ENCODING_AGX3B, 1, 0, 0x80,
                                     false, true, 0, 0, out, &len));
   assert(!agx3_alu_fadd_minifloat_mod(AGX_ISA_ENCODING_AGX3B, 0, 1, 0x80,
                                     false, true, 0, 0, out, &len));
   assert(!agx3_alu_fadd_minifloat_mod(AGX_ISA_ENCODING_AGX3B, 256, 0, 0x80,
                                     false, true, 0, 0, out, &len));
   assert(!agx3_alu_fadd_minifloat_mod(AGX_ISA_ENCODING_AGX3B, 0, 256, 0x80,
                                     false, true, 0, 0, out, &len));
   assert(!agx3_alu_fadd_minifloat_mod(AGX_ISA_ENCODING_AGX3B, 0, 0, 256,
                                     false, true, 0, 0, out, &len));
   assert(!agx3_alu_fadd_minifloat_mod(AGX_ISA_ENCODING_AGX3B, 0, 0, 0x80,
                                     false, true, 3, 0, out, &len));
   assert(!agx3_alu_fadd_minifloat_mod(AGX_ISA_ENCODING_AGX3B, 0, 0, 0x80,
                                     false, true, 0, 64, out, &len));
   assert(!agx3_alu_fadd_minifloat_mod(AGX_ISA_ENCODING_AGX3B, 0, 0, 0x30,
                                     true, false, 0, 63, out, &len));
   printf("Modified minifloat add:14 native witnesses +%u regular and %u "
          "absolute physical forms passed\n", shapes, abs_shapes);
}

static void test_half_uv(void)
{
   uint8_t out[16], size;
   const struct {unsigned d,bits; const char *hex;} moves[] = {
      {2,0x1800,"14800200000c0000"}, {2,0x1801,"14810200000c0000"},
      {2,0x9800,"14800200000c0400"}, {3,0x1800,"14800600000c0000"},
      {4,0x1800,"24800200000c0000"},
   };
   for (unsigned i=0;i<sizeof(moves)/sizeof(moves[0]);++i) {
      assert(agx3_alu_mov16(AGX_ISA_ENCODING_AGX3B,moves[i].d,moves[i].bits,0,out,&size));
      exact(out,size,moves[i].hex);
   }
   const struct {unsigned d,f,h; enum agx3_alu_hint a,b; bool cache; unsigned wait; const char *hex;} muls[] = {
      {0,0,4,AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD,false,32,"09011d0400c0"},
      {0,2,0,AGX3_ALU_HINT_NONE,AGX3_ALU_HINT_DISCARD,false,1,"090315000020"},
      {0,0,10,AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD,false,32,"09011d0a00c0"},
      {2,2,11,AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD,false,0,"19031d0a1000"},
      {0,0,10,AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_CACHE,false,32,"09010d8a00c0"},
      {2,2,10,AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD,false,0,"1903190a"},
      {0,0,2,AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD,true,32,"09013d0200c0"},
   };
   for (unsigned i=0;i<sizeof(muls)/sizeof(muls[0]);++i) {
      assert(agx3_alu_fmul_mixed(AGX_ISA_ENCODING_AGX3B,muls[i].d,muls[i].f,muls[i].h,muls[i].a,muls[i].b,
                   muls[i].cache,muls[i].wait,out,&size));
      exact(out,size,muls[i].hex);
   }
   assert(!agx3_alu_mov16(AGX_ISA_ENCODING_AGX3B,256,0x1800,0,out,&size));
   assert(!agx3_alu_mov16(AGX_ISA_ENCODING_AGX3B,2,0x10000,0,out,&size));
   assert(!agx3_alu_mov16(AGX_ISA_ENCODING_AGX3B,2,0x1800,1,out,&size));
   assert(!agx3_alu_fmul_mixed(AGX_ISA_ENCODING_AGX3B,1,0,4,AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD,false,0,out,&size));
   assert(!agx3_alu_fmul_mixed(AGX_ISA_ENCODING_AGX3B,0,1,4,AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD,false,0,out,&size));
   assert(!agx3_alu_fmul_mixed(AGX_ISA_ENCODING_AGX3B,0,0,256,AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD,false,0,out,&size));
   assert(!agx3_alu_fmul_mixed(AGX_ISA_ENCODING_AGX3B,0,0,4,AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD,false,64,out,&size));
   unsigned shapes=0;
   for (unsigned d=0;d<256;d+=2)
      for (unsigned h=0;h<256;++h)
         for (unsigned wait=0;wait<=32;wait=wait ? wait*2 : 1) {
            assert(agx3_alu_fmul_mixed(AGX_ISA_ENCODING_AGX3B,d,(d+2)%256,h,AGX3_ALU_HINT_DISCARD, h&1 ? AGX3_ALU_HINT_DISCARD : AGX3_ALU_HINT_CACHE,
                         !!(h & 2),wait,out,&size));
            assert(size==4 || size==6);
            ++shapes;
         }
   puts("Half UV native candidate PASS:5 LDIMM16 +7 mixed-multiply byte-exact Apple witnesses; odd halves, source/result lifetimes and wait6/no-wait/compact variants; invalid widths/ranges/waits rejected");
   printf("Validated%u register/half/wait shapes through actual composer; offline production encoder coverage\n",shapes);
}

static void
test_native_fmul_abs_mixed(void)
{
   static const struct {
      uint16_t dest, f32, f16, wait;
      enum agx3_alu_hint hint32, hint16;
      bool dest_cache;
      const char *bytes;
   } vectors[] = {
      /* Both loaded inputs die, waiting on their producer slot. */
      {0, 0, 4, 32, AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD, false,
       "09011d040200008001000000"},
      /* The same native operation without a producer wait. */
      {0, 0, 4, 0, AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD, false,
       "09011d040200000001000000"},
      /* Retain the f32 input for a later store. */
      {0, 2, 0, 32, AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_DISCARD, false,
       "090315000200008001000000"},
      /* Cache the f32 input shared by two channel multiplies. */
      {0, 2, 6, 32, AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_DISCARD, false,
       "098315060200008001000000"},
      /* The second channel proves that scalar-half register 7 stays odd. */
      {2, 2, 7, 0, AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD, false,
       "19031d061200000001000000"},
      /* Reuse both the result and scalar-half source. */
      {0, 0, 2, 32, AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_CACHE, true,
       "09012d820200008001000000"},
      /* A separately scheduled conversion pins producer slot 0. */
      {0, 0, 8, 1, AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD, false,
       "09011d080220000001000000"},
   };
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   for (unsigned i = 0; i < sizeof(vectors) / sizeof(vectors[0]); ++i) {
      assert(agx3_alu_fmul_abs_mixed(
         AGX_ISA_ENCODING_AGX3B, vectors[i].dest, vectors[i].f32,
         vectors[i].f16, vectors[i].hint32, vectors[i].hint16,
         vectors[i].dest_cache, vectors[i].wait, bytes, &len));
      exact(bytes, len, vectors[i].bytes);
   }

   /* Every lifetime and producer-slot combination is an independent field on
    * this Apple endpoint. Exercise the full cross-product at the high bank
    * boundary, including a retained (non-cache, non-discard) half source. */
   for (unsigned hint32 = 0; hint32 <= AGX3_ALU_HINT_CACHE; ++hint32) {
      for (unsigned hint16 = 0; hint16 <= AGX3_ALU_HINT_CACHE; ++hint16) {
         for (unsigned dest_cache = 0; dest_cache < 2; ++dest_cache) {
            for (unsigned wait = 0; wait < 64; ++wait) {
               assert(agx3_alu_fmul_abs_mixed(
                  AGX_ISA_ENCODING_AGX3B, 254, 252, 255,
                  (enum agx3_alu_hint)hint32, (enum agx3_alu_hint)hint16,
                  dest_cache, wait, bytes, &len));
               assert(len == 12);
            }
         }
      }
   }

   memset(bytes, 0xa5, sizeof(bytes));
   len = 0xa5;
   assert(!agx3_alu_fmul_abs_mixed(
      AGX_ISA_ENCODING_AGX3C, 0, 0, 4, AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_DISCARD, false, 0, bytes, &len));
   assert(!agx3_alu_fmul_abs_mixed(
      AGX_ISA_ENCODING_AGX3B, 1, 0, 4, AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_DISCARD, false, 0, bytes, &len));
   assert(!agx3_alu_fmul_abs_mixed(
      AGX_ISA_ENCODING_AGX3B, 0, 1, 4, AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_DISCARD, false, 0, bytes, &len));
   assert(!agx3_alu_fmul_abs_mixed(
      AGX_ISA_ENCODING_AGX3B, 0, 0, 256, AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_DISCARD, false, 0, bytes, &len));
   assert(!agx3_alu_fmul_abs_mixed(
      AGX_ISA_ENCODING_AGX3B, 0, 0, 4, (enum agx3_alu_hint)3,
      AGX3_ALU_HINT_DISCARD, false, 0, bytes, &len));
   assert(!agx3_alu_fmul_abs_mixed(
      AGX_ISA_ENCODING_AGX3B, 0, 0, 4, AGX3_ALU_HINT_DISCARD,
      (enum agx3_alu_hint)3, false, 0, bytes, &len));
   assert(!agx3_alu_fmul_abs_mixed(
      AGX_ISA_ENCODING_AGX3B, 0, 0, 4, AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_DISCARD, false, 64, bytes, &len));
   assert(len == 0xa5);
   for (unsigned i = 0; i < sizeof(bytes); ++i)
      assert(bytes[i] == 0xa5);

   puts("Native abs-f32 times scalar-half: seven exact Apple vectors, "
        "1152 lifetime/wait forms and strict bounds PASS");
}

static void
test_bitwise_lifetimes(void)
{
   /* Apple g17p native kernels reuse the second operand, then release it;
    * keep_both stores both unmodified operands after the AND result. */
   const struct {
      enum agx3_alu_op op;
      uint16_t dest, src[2];
      enum agx3_alu_hint hints[2];
      unsigned wait;
      const char *hex;
   } cases[] = {
      {AGX3_ALU_BAND, 0, {0, 2}, {AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_CACHE}, 32, "0b010f83000000800000"},
      {AGX3_ALU_BAND, 2, {6, 2}, {AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD}, 1, "1b071f03002000000000"},
      {AGX3_ALU_BOR, 0, {0, 2}, {AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_CACHE}, 32, "0b010f83020800800000"},
      {AGX3_ALU_BOR, 2, {6, 2}, {AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD}, 1, "1b071f03022800000000"},
      {AGX3_ALU_BXOR, 0, {0, 2}, {AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_CACHE}, 32, "0b010e83020800800000"},
      {AGX3_ALU_BXOR, 2, {6, 2}, {AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD}, 1, "1b071e03022800000000"},
      {AGX3_ALU_BAND, 6, {2, 0}, {AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE}, 32, "3b030701000000800000"},
   };
   uint8_t out[AGX3_ALU_MAX_LENGTH], size;
   for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
      assert(agx3_alu_bit_reg(AGX_ISA_ENCODING_AGX3B, cases[i].op,
                            cases[i].dest, cases[i].src, cases[i].hints,
                            cases[i].wait, out, &size));
      exact(out, size, cases[i].hex);
   }
   puts("Bitwise lifetimes: seven exact native AND/OR/XOR live/dead witnesses PASS");
}

static void
test_integer_lifetimes(void)
{
   const struct {
      enum agx3_alu_op op;
      uint16_t dest, src[2];
      enum agx3_alu_hint hints[2];
      const char *hex;
   } cases[] = {
      {AGX3_ALU_IADD, 0, {4, 0}, {AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD}, "9f015600020800a81705"},
      {AGX3_ALU_IADD, 0, {4, 2}, {AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE}, "9f015600020808a81105"},
      {AGX3_ALU_ISUB, 0, {0, 4}, {AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD}, "1f015600020010a81705"},
      {AGX3_ALU_ISUB, 0, {2, 4}, {AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE}, "1f015600020410a81105"},
   };
   uint8_t out[AGX3_ALU_MAX_LENGTH], size;
   for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
      assert(agx3_alu_integer_reg(AGX_ISA_ENCODING_AGX3B, cases[i].op,
                                cases[i].dest, cases[i].src, cases[i].hints,
                                32, out, &size));
      exact(out, size, cases[i].hex);
   }
   puts("Integer lifetimes: four exact native IADD/ISUB live/dead witnesses PASS");
}

static void
test_float_compare_zero(void)
{
   const struct {
      unsigned dest, src[3], hints[3], cond, wait;
      bool half;
      const char *hex;
   } cases[] = {
      {0,{10,0,2},{1,1,1},0,32,true,"020a0f81840200c2"},
      {0,{2,4,6},{2,0,0},0,32,true,"02820781060400c20006"},
      {0,{10,0,2},{1,1,1},1,32,true,"020a0f81800203c2"},
      {0,{0,2,4},{1,1,1},0,32,false,"02010f81860200c28004"},
      {0,{6,2,4},{1,2,2},0,32,true,"02060f81068200c20084"},
      {2,{7,4,2},{1,1,1},0,0,true,"12060f8185040802"},
   };
   for (unsigned i=0;i<sizeof(cases)/sizeof(cases[0]);++i) {
      uint8_t out[AGX3_ALU_MAX_LENGTH], len;
      uint16_t src[3]; enum agx3_alu_hint hints[3];
      for(unsigned j=0;j<3;++j) {src[j]=cases[i].src[j];hints[j]=cases[i].hints[j];}
      assert(agx3_alu_fcmp_select_zero(AGX_ISA_ENCODING_AGX3B,cases[i].dest,
             src,cases[i].half,hints,cases[i].cond,false,cases[i].wait,out,&len));
      exact(out,len,cases[i].hex);
   }
   /* Rejection leaves the output untouched. Half comparisons may be odd,
    * but full float operands/results must be aligned, and unsupported wait
    * sets must be retired explicitly by the caller. */
   for (unsigned bad=0;bad<8;++bad) {
      uint16_t src[] = {1,4,2};
      enum agx3_alu_hint hints[] = {AGX3_ALU_HINT_NONE,AGX3_ALU_HINT_NONE,AGX3_ALU_HINT_NONE};
      uint8_t out[AGX3_ALU_MAX_LENGTH],len=0xa5;
      memset(out,0xa5,sizeof(out));
      unsigned dest=70,cond=0,wait=0;
      bool half=true;
      if(bad==0)dest=71;
      if(bad==1)src[1]=5;
      if(bad==2)src[2]=256;
      if(bad==3)half=false;
      if(bad==4)hints[0]=3;
      if(bad==5)wait=1;
      if(bad==6)cond=7;
      if(bad==7)dest=256;
      assert(!agx3_alu_fcmp_select_zero(AGX_ISA_ENCODING_AGX3B,dest,src,half,
                                      hints,cond,false,wait,out,&len));
      assert(len==0xa5);
      for(unsigned j=0;j<sizeof(out);++j)assert(out[j]==0xa5);
   }
   puts("FSEL zero: six exact mixed-width native lifetime witnesses PASS");
}

static void
test_float_compare_select(void)
{
   const uint16_t src[] = {0, 2};
   const enum agx3_alu_hint dead[] = {AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD};
   const enum agx3_alu_hint live[] = {AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE};
   uint8_t out[AGX3_ALU_MAX_LENGTH], len;
   assert(agx3_alu_fcmp_select_imm(AGX_ISA_ENCODING_AGX3B,0,src,dead,1,255,0,true,false,32,out,&len));
   exact(out,len,"02011d03a2ff03c02080");
   assert(agx3_alu_fcmp_select_imm(AGX_ISA_ENCODING_AGX3B,8,src,live,1,255,0,true,false,32,out,&len));
   exact(out,len,"42010503a2ff03c02080");
   const uint16_t mixed_src[] = {8, 10, 6, 4};
   const enum agx3_alu_hint mixed_hints[] = {
      AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_CACHE,
      AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_DISCARD,
   };
   assert(agx3_alu_fcmp_select_f32_f16(
      AGX_ISA_ENCODING_AGX3B, 4, mixed_src, mixed_hints, 0, true, 0,
      out, &len));
   exact(out, len, "2289278a05860000");
   const uint16_t lightmap_src[] = {46, 52, 26, 36};
   const enum agx3_alu_hint lightmap_hints[] = {
      AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_CACHE,
      AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_NONE,
   };
   assert(agx3_alu_fcmp_select_f32_f16(
      AGX_ISA_ENCODING_AGX3B, 36, lightmap_src, lightmap_hints,
      1, true, 0, out, &len));
   exact(out, len, "22af67b4029a03000024");
   /* The condition is orthogonal: gt, ge and le encode the same way, and
    * only the condition bytes move. Apple's own six-kernel sweep is in
    * test_native_f32_minifloat_select. The NaN-ignoring ltn/gtn codes were
    * not witnessed for this form and stay refused. */
   {
      uint8_t widened[AGX3_ALU_MAX_LENGTH];
      uint8_t widened_len;
      static const struct {
         unsigned condition;
         const char *bytes;
      } conditions[] = {
         {2 /* gt */, "22af67b4029a02000024"},
         {5 /* ge */, "22af67b4069a02000024"},
         {6 /* le */, "22af67b4069a03000024"},
      };
      for (unsigned i = 0; i < 3; ++i) {
         assert(agx3_alu_fcmp_select_f32_f16(
            AGX_ISA_ENCODING_AGX3B, 36, lightmap_src, lightmap_hints,
            conditions[i].condition, true, 0, widened, &widened_len));
         exact(widened, widened_len, conditions[i].bytes);
      }
      assert(!agx3_alu_fcmp_select_f32_f16(
         AGX_ISA_ENCODING_AGX3B, 36, lightmap_src, lightmap_hints, 3, true, 0,
         widened, &widened_len));
      assert(!agx3_alu_fcmp_select_f32_f16(
         AGX_ISA_ENCODING_AGX3B, 36, lightmap_src, lightmap_hints, 7, true, 0,
         widened, &widened_len));
   }
   assert(!agx3_alu_fcmp_select_f32_f16(
      AGX_ISA_ENCODING_AGX3C, 4, mixed_src, mixed_hints, 0, true, 0,
      out, &len));
   assert(agx3_alu_fcmp_select_imm(AGX_ISA_ENCODING_AGX3B,0,src,dead,5,255,0,true,false,32,out,&len));
   exact(out,len,"02011d03a6ff02c02080");
   assert(agx3_alu_fcmp_select_imm(AGX_ISA_ENCODING_AGX3B,8,src,live,5,255,0,true,false,32,out,&len));
   exact(out,len,"42010503a6ff02c02080");
   assert(agx3_alu_fcmp_select_imm(AGX_ISA_ENCODING_AGX3B,0,src,dead,2,255,0,true,false,32,out,&len));
   exact(out,len,"02011d03a2ff02c02080");
   assert(agx3_alu_fcmp_select_imm(AGX_ISA_ENCODING_AGX3B,8,src,live,2,255,0,true,false,32,out,&len));
   exact(out,len,"42010503a2ff02c02080");
   assert(agx3_alu_fcmp_select_imm(AGX_ISA_ENCODING_AGX3B,0,src,dead,6,255,0,true,false,32,out,&len));
   exact(out,len,"02011d03a6ff03c02080");
   assert(agx3_alu_fcmp_select_imm(AGX_ISA_ENCODING_AGX3B,8,src,live,6,255,0,true,false,32,out,&len));
   exact(out,len,"42010503a6ff03c02080");
   assert(agx3_alu_fcmp_select_imm(AGX_ISA_ENCODING_AGX3B,0,src,dead,0,255,0,true,false,32,out,&len));
   exact(out,len,"02011d03a6ff00c02080");
   assert(agx3_alu_fcmp_select_imm(AGX_ISA_ENCODING_AGX3B,8,src,live,0,255,0,true,false,32,out,&len));
   exact(out,len,"42010503a6ff00c02080");
   assert(agx3_alu_fcmp_select_imm(AGX_ISA_ENCODING_AGX3B,0,src,dead,0,0,255,true,false,32,out,&len));
   exact(out,len,"02011d03268000c0a0ff");
   assert(agx3_alu_fcmp_select_imm(AGX_ISA_ENCODING_AGX3B,8,src,live,0,0,255,true,false,32,out,&len));
   exact(out,len,"42010503268000c0a0ff");
   const uint16_t alpha_src[] = {6, 10};
   const enum agx3_alu_hint alpha_hint[] = {AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_DISCARD};
   assert(agx3_alu_fcmp_select_imm(AGX_ISA_ENCODING_AGX3B,11,alpha_src,alpha_hint,1,0,1,true,true,0,out,&len));
   exact(out,len,"5a07350b228003002081");
   assert(agx3_alu_fcmp_select_imm(AGX_ISA_ENCODING_AGX3B,0,src,dead,1,255,0,false,false,32,out,&len));
   exact(out,len,"02011f03a2ff03c02080");
   assert(agx3_alu_fcmp_select_imm(AGX_ISA_ENCODING_AGX3B,6,src,live,1,255,0,false,false,32,out,&len));
   exact(out,len,"32010703a2ff03c02080");
   assert(agx3_alu_fcmp_select_imm(AGX_ISA_ENCODING_AGX3B,0,src,dead,5,255,0,false,false,32,out,&len));
   exact(out,len,"02011f03a6ff02c02080");
   assert(agx3_alu_fcmp_select_imm(AGX_ISA_ENCODING_AGX3B,6,src,live,5,255,0,false,false,32,out,&len));
   exact(out,len,"32010703a6ff02c02080");
   assert(agx3_alu_fcmp_select_imm(AGX_ISA_ENCODING_AGX3B,0,src,dead,2,255,0,false,false,32,out,&len));
   exact(out,len,"02011f03a2ff02c02080");
   assert(agx3_alu_fcmp_select_imm(AGX_ISA_ENCODING_AGX3B,6,src,live,2,255,0,false,false,32,out,&len));
   exact(out,len,"32010703a2ff02c02080");
   assert(agx3_alu_fcmp_select_imm(AGX_ISA_ENCODING_AGX3B,0,src,dead,6,255,0,false,false,32,out,&len));
   exact(out,len,"02011f03a6ff03c02080");
   assert(agx3_alu_fcmp_select_imm(AGX_ISA_ENCODING_AGX3B,6,src,live,6,255,0,false,false,32,out,&len));
   exact(out,len,"32010703a6ff03c02080");
   assert(agx3_alu_fcmp_select_imm(AGX_ISA_ENCODING_AGX3B,0,src,dead,0,255,0,false,false,32,out,&len));
   exact(out,len,"02011f03a6ff00c02080");
   assert(agx3_alu_fcmp_select_imm(AGX_ISA_ENCODING_AGX3B,6,src,live,0,255,0,false,false,32,out,&len));
   exact(out,len,"32010703a6ff00c02080");
   assert(agx3_alu_fcmp_select_imm(AGX_ISA_ENCODING_AGX3B,0,src,dead,0,0,255,false,false,32,out,&len));
   exact(out,len,"02011f03268000c0a0ff");
   assert(agx3_alu_fcmp_select_imm(AGX_ISA_ENCODING_AGX3B,6,src,live,0,0,255,false,false,32,out,&len));
   exact(out,len,"32010703268000c0a0ff");
   puts("Float select:25 native comparison/alpha dead-live vectors PASS");
}

static void test_int_to_float_liveness(void)
{
   uint8_t out[AGX3_ALU_MAX_LENGTH], len;
   assert(agx3_alu_int_to_float(AGX_ISA_ENCODING_AGX3B,0,0,false,AGX3_ALU_HINT_DISCARD,32,out,&len));
   exact(out,len,"a70756000200ac20");
   assert(agx3_alu_int_to_float(AGX_ISA_ENCODING_AGX3B,0,2,false,AGX3_ALU_HINT_NONE,32,out,&len));
   exact(out,len,"a707560002048c20");
   assert(agx3_alu_int_to_float(AGX_ISA_ENCODING_AGX3B,0,0,true,AGX3_ALU_HINT_DISCARD,32,out,&len));
   exact(out,len,"a70756000200ac60");
   assert(agx3_alu_int_to_float(AGX_ISA_ENCODING_AGX3B,0,2,true,AGX3_ALU_HINT_NONE,32,out,&len));
   exact(out,len,"a707560002048c60");
   puts("Integer-to-float: four native signed/unsigned live/dead vectors PASS");
}

static void test_i32_to_f16(void)
{
   uint8_t out[AGX3_ALU_MAX_LENGTH], len;
   assert(agx3_alu_i32_to_f16(AGX_ISA_ENCODING_AGX3B, 0, 2, false,
                              AGX3_ALU_HINT_NONE, true, 1, out, &len));
   exact(out, len, "a717540003048820");
   assert(agx3_alu_i32_to_f16(AGX_ISA_ENCODING_AGX3B, 2, 0, false,
                              AGX3_ALU_HINT_NONE, false, 0, out, &len));
   exact(out, len, "a707540202008820");
   assert(agx3_alu_i32_to_f16(AGX_ISA_ENCODING_AGX3B, 0, 0, false,
                              AGX3_ALU_HINT_DISCARD, false, 0, out, &len));
   exact(out, len, "a70754000200a820");
   assert(agx3_alu_i32_to_f16(AGX_ISA_ENCODING_AGX3B, 0, 0, true,
                              AGX3_ALU_HINT_DISCARD, false, 0, out, &len));
   exact(out, len, "a70754000200a860");
   assert(agx3_alu_i32_to_f16(AGX_ISA_ENCODING_AGX3B, 2, 0, true,
                              AGX3_ALU_HINT_NONE, true, 1, out, &len));
   exact(out, len, "a717540203008860");
   assert(!agx3_alu_i32_to_f16(AGX_ISA_ENCODING_AGX3C, 0, 2, false,
                               AGX3_ALU_HINT_NONE, false, 0, out, &len));
   assert(!agx3_alu_i32_to_f16(AGX_ISA_ENCODING_AGX3B, 0, 3, false,
                               AGX3_ALU_HINT_NONE, false, 0, out, &len));
   assert(!agx3_alu_i32_to_f16(AGX_ISA_ENCODING_AGX3B, 0, 2, false,
                               AGX3_ALU_HINT_CACHE, false, 0, out, &len));
   assert(!agx3_alu_i32_to_f16(AGX_ISA_ENCODING_AGX3B, 0, 2, false,
                               AGX3_ALU_HINT_NONE, false, 0x40, out, &len));
   puts("Native i32-to-f16: five exact Apple vectors and refusals PASS");
}

static void
test_f16_to_int32(void)
{
   static const struct {
      bool sign, cache;
      uint16_t dest, src, wait;
      enum agx3_alu_hint hint;
      uint8_t bytes[10];
   } native[] = {
      {true, false, 0, 0, 32, AGX3_ALU_HINT_DISCARD,
       {0x27, 0x07, 0x56, 0x00, 0x02, 0x00, 0xac, 0x48, 0x03, 0x00}},
      {false, false, 0, 0, 32, AGX3_ALU_HINT_DISCARD,
       {0x27, 0x07, 0x56, 0x00, 0x02, 0x00, 0xac, 0x08, 0x02, 0x00}},
      {true, false, 0, 0, 0, AGX3_ALU_HINT_DISCARD,
       {0x27, 0x07, 0x54, 0x00, 0x02, 0x00, 0xac, 0x48, 0x03, 0x00}},
      {true, false, 2, 4, 32, AGX3_ALU_HINT_NONE,
       {0x27, 0x07, 0x56, 0x02, 0x02, 0x08, 0x8c, 0x48, 0x03, 0x00}},
      {true, true, 0, 0, 32, AGX3_ALU_HINT_DISCARD,
       {0x27, 0x07, 0x56, 0x00, 0x03, 0x00, 0xac, 0x48, 0x03, 0x00}},
      {true, true, 0, 6, 32, AGX3_ALU_HINT_NONE,
       {0x27, 0x07, 0x56, 0x00, 0x03, 0x0c, 0x8c, 0x48, 0x03, 0x00}},
      {true, false, 4, 1, 32, AGX3_ALU_HINT_DISCARD,
       {0x27, 0x07, 0x56, 0x04, 0x02, 0x02, 0xac, 0x48, 0x03, 0x00}},
      {true, false, 0, 0, 1, AGX3_ALU_HINT_DISCARD,
       {0x27, 0x17, 0x54, 0x00, 0x02, 0x00, 0xac, 0x48, 0x03, 0x00}},
   };

   uint8_t out[AGX3_ALU_MAX_LENGTH];
   uint8_t len = 0;
   for (unsigned i = 0; i < sizeof(native) / sizeof(native[0]); ++i) {
      assert(agx3_alu_f16_to_int32(
         AGX_ISA_ENCODING_AGX3B, native[i].dest, native[i].src,
         native[i].sign, native[i].hint, native[i].cache,
         native[i].wait, out, &len));
      assert(len == sizeof(native[i].bytes));
      assert(!memcmp(out, native[i].bytes, len));
   }

   /* Exercise the complete admitted Cartesian product, including the upper
    * register boundary and every producer wait combination. */
   for (unsigned sign = 0; sign < 2; ++sign)
      for (unsigned hint = AGX3_ALU_HINT_NONE;
           hint <= AGX3_ALU_HINT_DISCARD; ++hint)
         for (unsigned cache = 0; cache < 2; ++cache)
            for (unsigned wait = 0; wait < 64; ++wait) {
               assert(agx3_alu_f16_to_int32(
                  AGX_ISA_ENCODING_AGX3B, 126, 127, sign, hint, cache,
                  wait, out, &len));
               assert(len == 10);
            }

   const uint8_t sentinel[AGX3_ALU_MAX_LENGTH] = {
      0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5,
      0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5,
   };
#define REJECT(...)                                                            \
   do {                                                                        \
      memcpy(out, sentinel, sizeof(out));                                      \
      len = 0x5a;                                                              \
      assert(!agx3_alu_f16_to_int32(__VA_ARGS__));                             \
      assert(!memcmp(out, sentinel, sizeof(out)) && len == 0x5a);              \
   } while (0)
   REJECT(AGX_ISA_ENCODING_AGX3C, 0, 0, true,
          AGX3_ALU_HINT_NONE, false, 0, out, &len);
   REJECT(AGX_ISA_ENCODING_AGX3B, 128, 0, true,
          AGX3_ALU_HINT_NONE, false, 0, out, &len);
   REJECT(AGX_ISA_ENCODING_AGX3B, 0, 128, true,
          AGX3_ALU_HINT_NONE, false, 0, out, &len);
   REJECT(AGX_ISA_ENCODING_AGX3B, 1, 0, true,
          AGX3_ALU_HINT_NONE, false, 0, out, &len);
   REJECT(AGX_ISA_ENCODING_AGX3B, 0, 0, true,
          AGX3_ALU_HINT_CACHE, false, 0, out, &len);
   REJECT(AGX_ISA_ENCODING_AGX3B, 0, 0, true,
          (enum agx3_alu_hint)3, false, 0, out, &len);
   REJECT(AGX_ISA_ENCODING_AGX3B, 0, 0, true,
          AGX3_ALU_HINT_NONE, false, 64, out, &len);
   assert(!agx3_alu_f16_to_int32(
      AGX_ISA_ENCODING_AGX3B, 0, 0, true, AGX3_ALU_HINT_NONE,
      false, 0, NULL, &len));
   assert(!agx3_alu_f16_to_int32(
      AGX_ISA_ENCODING_AGX3B, 0, 0, true, AGX3_ALU_HINT_NONE,
      false, 0, out, NULL));
#undef REJECT

   puts("F16-to-int32: eight exact native vectors, 512 legal forms and "
        "strict refusals PASS");
}

static void
test_native_fma_minifloat(void)
{
   {
      static const struct {
         uint16_t dest, src, addend;
         uint8_t immediate;
         enum agx3_alu_hint src_hint, addend_hint;
         const char *bytes;
      } negative_addend[] = {
         {0, 0, 2, 64, AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD,
          "09010ec1910202c2"},
         {0, 2, 4, 64, AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE,
          "090306c1110402c2"},
         {0, 0, 2, 32, AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD,
          "09010ea1910202c2"},
         {0, 2, 4, 32, AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE,
          "090306a1110402c2"},
      };
      for (unsigned i = 0;
           i < sizeof(negative_addend) / sizeof(negative_addend[0]); ++i) {
         uint8_t out[16], len;
         assert(agx3_alu_fma_minifloat_mod(
            AGX_ISA_ENCODING_AGX3B, negative_addend[i].dest,
            negative_addend[i].src, negative_addend[i].immediate,
            negative_addend[i].addend, true, negative_addend[i].src_hint,
            negative_addend[i].addend_hint, false, 0, out, &len));
         exact(out, len, negative_addend[i].bytes);
      }
   }
   {const uint8_t native[]={0x49,0x11,0x2e,0xc1,0x81,0x08,0x02,0x42};uint8_t out[16],len;
    assert(agx3_alu_fma_minifloat(AGX_ISA_ENCODING_AGX3B,8,16,64,8,1,1,1,2,out,&len));
    assert(len==sizeof(native) && !memcmp(out,native,len));}
   {const uint8_t native[]={0x59,0x13,0x2e,0xc1,0xa0,0x0a};uint8_t out[16],len;
    assert(agx3_alu_fma_minifloat(AGX_ISA_ENCODING_AGX3B,10,18,64,10,1,1,1,0,out,&len));
    assert(len==sizeof(native) && !memcmp(out,native,len));}
   {const uint8_t native[]={0x09,0x01,0x2e,0xd1,0x81,0x08,0x02,0x22};uint8_t out[16],len;
    assert(agx3_alu_fma_minifloat(AGX_ISA_ENCODING_AGX3B,0,0,80,8,1,1,1,1,out,&len));
    assert(len==sizeof(native) && !memcmp(out,native,len));}
   {const uint8_t native[]={0x19,0x03,0x2e,0xd1,0xa0,0x0a};uint8_t out[16],len;
    assert(agx3_alu_fma_minifloat(AGX_ISA_ENCODING_AGX3B,2,2,80,10,1,1,1,0,out,&len));
    assert(len==sizeof(native) && !memcmp(out,native,len));}
   {const uint8_t native[]={0x19,0x15,0x2e,0xc1,0xa0,0x0c};uint8_t out[16],len;
    assert(agx3_alu_fma_minifloat(AGX_ISA_ENCODING_AGX3B,2,20,64,12,1,1,1,0,out,&len));
    assert(len==sizeof(native) && !memcmp(out,native,len));}
   {const uint8_t native[]={0x49,0x17,0x2e,0xc1,0xa0,0x0e};uint8_t out[16],len;
    assert(agx3_alu_fma_minifloat(AGX_ISA_ENCODING_AGX3B,8,22,64,14,1,1,1,0,out,&len));
    assert(len==sizeof(native) && !memcmp(out,native,len));}
   {const uint8_t native[]={0x19,0x05,0x2e,0xd1,0xa0,0x02};uint8_t out[16],len;
    assert(agx3_alu_fma_minifloat(AGX_ISA_ENCODING_AGX3B,2,4,80,2,1,1,1,0,out,&len));
    assert(len==sizeof(native) && !memcmp(out,native,len));}
   {const uint8_t native[]={0x29,0x07,0x2e,0xd1,0xa0,0x08};uint8_t out[16],len;
    assert(agx3_alu_fma_minifloat(AGX_ISA_ENCODING_AGX3B,4,6,80,8,1,1,1,0,out,&len));
    assert(len==sizeof(native) && !memcmp(out,native,len));}
   {const uint8_t native[]={0x49,0x09,0x2e,0xc1,0x01,0x80,0x02,0x42};uint8_t out[16],len;
    assert(agx3_alu_fma_minifloat(AGX_ISA_ENCODING_AGX3B,8,8,64,0,1,2,1,2,out,&len));
    assert(len==sizeof(native) && !memcmp(out,native,len));}
   {const uint8_t native[]={0x59,0x0b,0x2e,0xc1,0x20,0x82};uint8_t out[16],len;
    assert(agx3_alu_fma_minifloat(AGX_ISA_ENCODING_AGX3B,10,10,64,2,1,2,1,0,out,&len));
    assert(len==sizeof(native) && !memcmp(out,native,len));}
   {const uint8_t native[]={0x19,0x0d,0x2e,0xc1,0x20,0x84};uint8_t out[16],len;
    assert(agx3_alu_fma_minifloat(AGX_ISA_ENCODING_AGX3B,2,12,64,4,1,2,1,0,out,&len));
    assert(len==sizeof(native) && !memcmp(out,native,len));}
   {const uint8_t native[]={0x49,0x0f,0x2e,0xc1,0x20,0x86};uint8_t out[16],len;
    assert(agx3_alu_fma_minifloat(AGX_ISA_ENCODING_AGX3B,8,14,64,6,1,2,1,0,out,&len));
    assert(len==sizeof(native) && !memcmp(out,native,len));}
   {const uint8_t native[]={0x09,0x01,0x0e,0xc1,0x81,0x04,0x02,0xc2};uint8_t out[16],len;
    assert(agx3_alu_fma_minifloat(AGX_ISA_ENCODING_AGX3B,0,0,64,4,1,1,0,32,out,&len));
    assert(len==sizeof(native) && !memcmp(out,native,len));}
   {const uint8_t native[]={0x09,0x83,0x06,0xd1,0x01,0x86,0x02,0xc2};uint8_t out[16],len;
    assert(agx3_alu_fma_minifloat(AGX_ISA_ENCODING_AGX3B,0,2,80,6,2,2,0,32,out,&len));
    assert(len==sizeof(native) && !memcmp(out,native,len));}
   for(unsigned wait=0;wait<64;wait++) {
      uint8_t out[16],len;
      assert(agx3_alu_fma_minifloat(AGX_ISA_ENCODING_AGX3B,4,24,64,4,1,1,false,wait,out,&len));
      assert(len==6 || len==8 || len==10);
      if(len==10) {
         /* Extended FMA physical wait bits: slots1..3 at61..63 and
          * slots4..6 at77..79, independent of the composer. */
         const unsigned decoded=(out[7]>>5)|((out[9]>>5)<<3);
         assert(decoded==wait);
      }
   }
   puts("native FMA minifloat:14 byte vectors and64 wait combinations PASS");
}

static void test_native_u16_arithmetic(void)
{
   uint8_t out[AGX3_ALU_MAX_LENGTH], len;
   const uint16_t xy[] = {3, 4};
   const enum agx3_alu_hint cache[] = {AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_CACHE};
   const enum agx3_alu_hint dead[] = {AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD};
   assert(agx3_alu_bit_reg16(AGX_ISA_ENCODING_AGX3B,AGX3_ALU_BAND,2,xy,cache,32,out,&len));
   exact(out,len,"13820784080000800000");
   assert(agx3_alu_bit_reg16(AGX_ISA_ENCODING_AGX3B,AGX3_ALU_BOR,2,xy,cache,32,out,&len));
   exact(out,len,"138207840a0800800000");
   assert(agx3_alu_bit_reg16(AGX_ISA_ENCODING_AGX3B,AGX3_ALU_BXOR,3,xy,dead,0,out,&len));
   exact(out,len,"13021e040e0800000000");
   const uint16_t t[] = {16, 16}, n[] = {11, 10};
   assert(agx3_alu_shift_add16(AGX_ISA_ENCODING_AGX3B,10,t,dead,4,true,2,out,&len));
   exact(out,len,"9f21540a032040001605");
   assert(agx3_alu_shift_add16(AGX_ISA_ENCODING_AGX3B,10,n,dead,4,true,0,out,&len));
   exact(out,len,"9f01540a031628001605");
   const uint16_t sll_dead_src[] = {0, 1};
   const bool sll_half_src[] = {true, true};
   assert(agx3_alu_sll_reg(AGX_ISA_ENCODING_AGX3B, 0, true, true,
                           sll_dead_src, sll_half_src, dead, 2, out, &len));
   exact(out, len, "272054000300040080130500");
   const uint16_t sll_mixed_src[] = {2, 0};
   const bool sll_mixed_width[] = {true, false};
   const enum agx3_alu_hint sll_mixed_hints[] = {
      AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_DISCARD,
   };
   assert(agx3_alu_sll_reg(AGX_ISA_ENCODING_AGX3B, 0, false, true,
                           sll_mixed_src, sll_mixed_width, sll_mixed_hints,
                           2, out, &len));
   exact(out, len, "2720540003040200d0120500");
   const enum agx3_alu_hint sll_bad_hints[] = {
      AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_CACHE,
   };
   assert(!agx3_alu_sll_reg(AGX_ISA_ENCODING_AGX3B, 0, true, false,
                            sll_mixed_src, sll_mixed_width, sll_bad_hints,
                            0, out, &len));
   assert(agx3_alu_srl_reg(AGX_ISA_ENCODING_AGX3B, 0, false, false,
                           sll_mixed_src, sll_mixed_width, sll_mixed_hints,
                           2, out, &len));
   exact(out, len, "a720540002040200d0120500");
   assert(!agx3_alu_srl_reg(AGX_ISA_ENCODING_AGX3B, 0, true, false,
                            sll_mixed_src, sll_mixed_width, sll_bad_hints,
                            0, out, &len));
   const uint16_t mixed_src[] = {2, 4};
   const bool mixed_width[] = {false, true};
   assert(agx3_alu_fcmp_select_imm_width(AGX_ISA_ENCODING_AGX3B,2,mixed_src,mixed_width,dead,1,255,0,true,false,32,out,&len));
   exact(out,len,"12031d04a2ff03c02080");
   const uint16_t neg_src[] = {8, 0};
   const bool full_width[] = {false, false};
   assert(agx3_alu_fcmp_select_imm_width_neg_second(
      AGX_ISA_ENCODING_AGX3B, 0, neg_src, full_width, dead,
      2, 255, 0, true, false, 32, out, &len));
   exact(out, len, "02091d01a2ff02c82080");
   assert(!agx3_alu_fcmp_select_imm_width_neg_second(
      AGX_ISA_ENCODING_AGX3B, 0, neg_src, full_width, dead,
      1, 255, 0, true, false, 32, out, &len));
   assert(!agx3_alu_fcmp_select_imm_width_neg_second(
      AGX_ISA_ENCODING_AGX3B, 0, neg_src, full_width, dead,
      2, 255, 0, false, false, 32, out, &len));
   puts("Native u16 ALU and mixed-width FSEL: six native byte vectors PASS");
}

static void
test_native_u16_shift_right_immediate(void)
{
   static const struct {
      uint16_t dest, src, wait;
      enum agx3_alu_hint hint;
      bool dest_cache;
      const char *bytes;
   } witnesses[] = {
      /* Fresh applegpu_g17p [[sample_mask]] extraction. */
      {0, 0, 1, AGX3_ALU_HINT_DISCARD, true,
       "a710540003001000c0110100"},
      /* Apple's constant sample-mask fragment path. */
      {1, 0, 1, AGX3_ALU_HINT_CACHE, true,
       "a710540103001200c0100100"},
      /* Apple's per-sample fragment path. */
      {15, 14, 2, AGX3_ALU_HINT_CACHE, true,
       "a720540f031c1200c0100100"},
   };
   uint8_t out[AGX3_ALU_MAX_LENGTH], len;

   for (unsigned i = 0; i < sizeof(witnesses) / sizeof(witnesses[0]); ++i) {
      assert(agx3_alu_srl_imm16(
         AGX_ISA_ENCODING_AGX3B, witnesses[i].dest, witnesses[i].src, 4,
         witnesses[i].hint, witnesses[i].dest_cache, witnesses[i].wait, out,
         &len));
      exact(out, len, witnesses[i].bytes);
   }

   assert(!agx3_alu_srl_imm16(AGX_ISA_ENCODING_AGX3C, 0, 0, 4,
                              AGX3_ALU_HINT_NONE, false, 0, out, &len));
   assert(!agx3_alu_srl_imm16(AGX_ISA_ENCODING_AGX3B, 0, 0, 0,
                              AGX3_ALU_HINT_NONE, false, 0, out, &len));
   assert(!agx3_alu_srl_imm16(AGX_ISA_ENCODING_AGX3B, 0, 0, 32,
                              AGX3_ALU_HINT_NONE, false, 0, out, &len));
   assert(!agx3_alu_srl_imm16(AGX_ISA_ENCODING_AGX3B, 256, 0, 4,
                              AGX3_ALU_HINT_NONE, false, 0, out, &len));
   assert(!agx3_alu_srl_imm16(AGX_ISA_ENCODING_AGX3B, 0, 256, 4,
                              AGX3_ALU_HINT_NONE, false, 0, out, &len));
   assert(!agx3_alu_srl_imm16(AGX_ISA_ENCODING_AGX3B, 0, 0, 4,
                              (enum agx3_alu_hint)3, false, 0, out, &len));
   assert(!agx3_alu_srl_imm16(AGX_ISA_ENCODING_AGX3B, 0, 0, 4,
                              AGX3_ALU_HINT_NONE, false, 1u << 6, out,
                              &len));
   puts("Native u16 immediate SRL: three exact Apple vectors and refusals PASS");
}

static void
test_blur_half_arithmetic(void)
{
   uint8_t out[AGX3_ALU_MAX_LENGTH],len;
   assert(agx3_alu_s16_to_f32(AGX_ISA_ENCODING_AGX3B,0,0,AGX3_ALU_HINT_DISCARD,32,out,&len));
   exact(out,len,"a70756000200a460");
   assert(agx3_alu_s16_to_f32(AGX_ISA_ENCODING_AGX3B,2,4,AGX3_ALU_HINT_NONE,32,out,&len));
   exact(out,len,"a707560202088460");
   assert(agx3_alu_s16_to_f32(AGX_ISA_ENCODING_AGX3B,0,2,AGX3_ALU_HINT_DISCARD,32,out,&len));
   exact(out,len,"a70756000204a460");
   assert(agx3_alu_s16_to_f32(AGX_ISA_ENCODING_AGX3B,2,3,AGX3_ALU_HINT_DISCARD,0,out,&len));
   exact(out,len,"a70754020206a460");
   assert(agx3_alu_add_imm16(AGX_ISA_ENCODING_AGX3B,2,2,2,AGX3_ALU_HINT_DISCARD,false,32,out,&len));
   exact(out,len,"9f015602020408001504");
   assert(agx3_alu_add_imm16(AGX_ISA_ENCODING_AGX3B,2,3,2,AGX3_ALU_HINT_NONE,false,32,out,&len));
   exact(out,len,"9f01560202040c001104");
   assert(agx3_alu_add_imm16(AGX_ISA_ENCODING_AGX3B,3,3,17,AGX3_ALU_HINT_DISCARD,false,0,out,&len));
   exact(out,len,"9f01540302220c001504");
   assert(agx3_alu_sub_imm16(AGX_ISA_ENCODING_AGX3B,2,2,1,
                            AGX3_ALU_HINT_DISCARD,false,32,out,&len));
   exact(out,len,"1f015602020404001301");
   assert(agx3_alu_sub_imm16(AGX_ISA_ENCODING_AGX3B,2,3,2,
                            AGX3_ALU_HINT_CACHE,false,32,out,&len));
   exact(out,len,"1f015602020608001101");
   assert(agx3_alu_sub_imm16(AGX_ISA_ENCODING_AGX3B,3,3,17,
                            AGX3_ALU_HINT_DISCARD,false,0,out,&len));
   exact(out,len,"1f015403020644001301");
   assert(agx3_alu_sub_imm16(AGX_ISA_ENCODING_AGX3B,2,2,255,
                            AGX3_ALU_HINT_DISCARD,true,0,out,&len));
   exact(out,len,"1f0154020304fc031301");
   assert(!agx3_alu_sub_imm16(AGX_ISA_ENCODING_AGX3C,2,2,2,
                             AGX3_ALU_HINT_DISCARD,false,0,out,&len));
   assert(!agx3_alu_sub_imm16(AGX_ISA_ENCODING_AGX3B,2,2,0,
                             AGX3_ALU_HINT_DISCARD,false,0,out,&len));
   assert(agx3_alu_sub_imm16(AGX_ISA_ENCODING_AGX3B,2,3,2,
                            AGX3_ALU_HINT_NONE,false,32,out,&len));
   exact(out,len,"1f015602020608001101");
   assert(!agx3_alu_sub_imm16(AGX_ISA_ENCODING_AGX3B,2,2,2,
                             AGX3_ALU_HINT_DISCARD,false,1,out,&len));
   assert(agx3_alu_f32_to_s16(AGX_ISA_ENCODING_AGX3B,2,2,
                              AGX3_ALU_HINT_DISCARD,false,32,out,&len));
   exact(out,len,"270756020204b0c80200");
   assert(agx3_alu_f32_to_s16(AGX_ISA_ENCODING_AGX3B,1,2,
                              AGX3_ALU_HINT_NONE,false,0,out,&len));
   exact(out,len,"27075401020490c80200");
   assert(agx3_alu_f32_to_s16(AGX_ISA_ENCODING_AGX3B,2,0,
                              AGX3_ALU_HINT_CACHE,true,0,out,&len));
   exact(out,len,"27075402030090c80200");
   assert(!agx3_alu_f32_to_s16(AGX_ISA_ENCODING_AGX3C,2,0,
                               AGX3_ALU_HINT_NONE,false,0,out,&len));
   assert(!agx3_alu_f32_to_s16(AGX_ISA_ENCODING_AGX3B,2,1,
                               AGX3_ALU_HINT_NONE,false,0,out,&len));
   puts("Blur half arithmetic: fourteen exact native conversion/add/sub witnesses PASS");
}

static void test_native_extract32(void)
{
   uint8_t bytes[AGX3_ALU_MAX_LENGTH],length;
   const unsigned shifts[]={1,4,16,28,31};
   const char *golden[5][2]={
      {"2701560002040400f0150900","2701560002080410f0100900"},
      {"2701560002041000f0150900","2701560002081010f0100900"},
      {"2701560002044000f0150900","2701560002084010f0100900"},
      {"2701560002047000f0150900","2701560002087010f0100900"},
      {"2701560002047c00f0150900","2701560002087c10f0100900"},
   };
   for(unsigned n=0;n<5;n++)for(unsigned live=0;live<2;live++) {
      const uint16_t src[2]={live?2:0,live?4:2};
      const enum agx3_alu_hint hints[2]={live?AGX3_ALU_HINT_NONE:AGX3_ALU_HINT_DISCARD,live?AGX3_ALU_HINT_NONE:AGX3_ALU_HINT_DISCARD};
      assert(agx3_alu_extract_imm32(AGX_ISA_ENCODING_AGX3B,0,src,shifts[n],hints,false,32,bytes,&length));
      exact(bytes,length,golden[n][live]);
   }
   puts("Native EXTRACT32:10 exact Apple immediate/live/dead vectors PASS");
   /* Independent M5 table vector from Zink's descriptor-address extract:
    * discard low half-pair8, retain high half-pair10, wait for slot7. */
   const uint16_t m5_src[2] = {8,10};
   const enum agx3_alu_hint m5_hints[2] = {AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_CACHE};
   assert(agx3_alu_extract_imm32(AGX_ISA_ENCODING_AGX3C,8,m5_src,2,m5_hints,
                                false,64,bytes,&length));
   exact(bytes,length,"2782401222a0a21280805c000808");
   assert(!agx3_alu_extract_imm32(AGX_ISA_ENCODING_AGX3C,8,m5_src,2,m5_hints,
                                 false,256,bytes,&length));
   const enum agx3_alu_hint field_hints[2] = {AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_DISCARD};
   assert(agx3_alu_bfeil(AGX_ISA_ENCODING_AGX3C,20,10,20,true,14,field_hints,
                         0,bytes,&length));
   exact(bytes,length,"a782005a2200a10a0885203400e0");
   assert(!agx3_alu_bfeil(AGX_ISA_ENCODING_AGX3C,20,10,20,true,32,field_hints,
                          0,bytes,&length));
}

static void test_native_asr32(void)
{
   const unsigned shifts[]={1,4,28};
   const char *golden[3][2]={
      {"a7015600020004786200","a7015600020404386200"},
      {"a7015600020010786200","a7015600020410386200"},
      {"a7015600020070786200","a7015600020470386200"},
   };
   for(unsigned n=0;n<3;n++)for(unsigned live=0;live<2;live++) {
      uint8_t bytes[AGX3_ALU_MAX_LENGTH],length;
      assert(agx3_alu_asr_imm32(AGX_ISA_ENCODING_AGX3B,0,live?2:0,shifts[n],
                                live?AGX3_ALU_HINT_NONE:AGX3_ALU_HINT_DISCARD,
                                false,32,bytes,&length));
      exact(bytes,length,golden[n][live]);
   }
   puts("Native ASR32:6 exact Apple immediate/live/dead vectors PASS");
   /* Independent M5 native-composer vectors. The first discarded-source
    * case is Xorg glamor's vertex-ID shift before integer-to-float conversion.
    * Also cover retained sources, cached destinations, waits and high regs. */
   const struct {
      uint16_t dest, source;
      unsigned shift;
      enum agx3_alu_hint hint;
      bool dest_cache;
      uint16_t wait;
      const char *bytes;
   } m5[] = {
      {0,0,1,AGX3_ALU_HINT_DISCARD,false,0,"2700001a2600a30278804000"},
      {0,0,31,AGX3_ALU_HINT_NONE,false,0,"2700001a2200a302f8874000"},
      {4,6,7,AGX3_ALU_HINT_NONE,true,32,"2702003a2a00a38af8814000"},
      {190,188,16,AGX3_ALU_HINT_CACHE,true,63,"b7fc08fa2a80a3ba3a844000"},
   };
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], length;
   for (unsigned n = 0; n < sizeof(m5) / sizeof(m5[0]); ++n) {
      assert(agx3_alu_asr_imm32(AGX_ISA_ENCODING_AGX3C,
                                m5[n].dest,m5[n].source,m5[n].shift,m5[n].hint,
                                m5[n].dest_cache,m5[n].wait,bytes,&length));
      exact(bytes,length,m5[n].bytes);
   }
   assert(!agx3_alu_asr_imm32(AGX_ISA_ENCODING_AGX3C,0,0,0,
                              AGX3_ALU_HINT_NONE,false,0,bytes,&length));
   assert(!agx3_alu_asr_imm32(AGX_ISA_ENCODING_AGX3C,0,0,32,
                              AGX3_ALU_HINT_NONE,false,0,bytes,&length));
   assert(!agx3_alu_asr_imm32(AGX_ISA_ENCODING_AGX3C,1,0,1,
                              AGX3_ALU_HINT_NONE,false,0,bytes,&length));
   assert(!agx3_alu_asr_imm32(AGX_ISA_ENCODING_AGX3C,0,0,1,
                              AGX3_ALU_HINT_NONE,false,64,bytes,&length));
   puts("M5 ASR32:4 native vectors and invalid operands PASS");
}

static void test_native_narrow_multiply(void)
{
   const uint16_t src[]={0,2};
   for (unsigned live=0;live<2;live++) {
      enum agx3_alu_hint hint=live?AGX3_ALU_HINT_NONE:AGX3_ALU_HINT_DISCARD;
      const enum agx3_alu_hint hints[]={hint,hint};
      uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
      assert(agx3_alu_fmul_narrow(AGX_ISA_ENCODING_AGX3B,live?6:0,src,hints,
                                 false,32,bytes,&len));
      exact(bytes,len,live?"3101050300c0":"01011d0300c0");
   }
   const uint16_t odd_src[]={4,6};
   const enum agx3_alu_hint odd_hints[]={AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD};
   uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
   assert(agx3_alu_fmul_narrow(AGX_ISA_ENCODING_AGX3B,1,odd_src,odd_hints,
                              false,0,bytes,&len));
   exact(bytes,len,"01051d070400");
   puts("Native narrow multiply: three Apple live/dead/odd-half vectors PASS");
}

static void test_native_half_multiply(void)
{
   for (unsigned live=0;live<2;live++) {
      const uint16_t src[]={live?0:2,live?1:3};
      enum agx3_alu_hint hint=live?AGX3_ALU_HINT_NONE:AGX3_ALU_HINT_DISCARD;
      const enum agx3_alu_hint hints[]={hint,hint};
      uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
      assert(agx3_alu_hmul(AGX_ISA_ENCODING_AGX3B,live?4:2,src,hints,
                           false,32,bytes,&len));
      exact(bytes,len,live?"2000050100c0":"10021d0300c0");
   }
   puts("Native half multiply: two complete Apple live/dead vectors PASS");
}

static void test_native_half_multiply_minifloat(void)
{
   static const struct {
      uint16_t dest, src, immediate, wait;
      enum agx3_alu_hint hint;
      bool dest_cache;
      const char *bytes;
   } vectors[] = {
      /* Fresh applegpu_g17p Metal output: *2, *4, *0.5 and *-2. */
      {2, 2, 0x40, 32, AGX3_ALU_HINT_DISCARD, false, "10c0150280c0"},
      {2, 2, 0x40, 0, AGX3_ALU_HINT_DISCARD, false, "10c015028000"},
      {2, 2, 0x50, 32, AGX3_ALU_HINT_DISCARD, false, "10d0150280c0"},
      {2, 2, 0x20, 32, AGX3_ALU_HINT_DISCARD, false, "10a0150280c0"},
      {2, 2, 0xc0, 32, AGX3_ALU_HINT_DISCARD, false, "10c01d0280c0"},
      /* Odd retained source and cached destination are independent axes. */
      {2, 3, 0x40, 32, AGX3_ALU_HINT_NONE, false, "10c0050380c0"},
      {3, 2, 0x50, 32, AGX3_ALU_HINT_NONE, false, "18d0050280c0"},
      {2, 2, 0x50, 32, AGX3_ALU_HINT_DISCARD, true, "10d0350280c0"},
   };
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   for (unsigned i = 0; i < sizeof(vectors) / sizeof(vectors[0]); ++i) {
      assert(agx3_alu_hmul_minifloat(
         AGX_ISA_ENCODING_AGX3B, vectors[i].dest, vectors[i].src,
         vectors[i].immediate, vectors[i].hint, vectors[i].dest_cache,
         vectors[i].wait, bytes, &len));
      exact(bytes, len, vectors[i].bytes);
   }

   assert(!agx3_alu_hmul_minifloat(
      AGX_ISA_ENCODING_AGX3C, 2, 2, 0x40, AGX3_ALU_HINT_DISCARD, false, 32,
      bytes, &len));
   assert(!agx3_alu_hmul_minifloat(
      AGX_ISA_ENCODING_AGX3B, 2, 2, 0x100, AGX3_ALU_HINT_DISCARD, false, 32,
      bytes, &len));
   assert(!agx3_alu_hmul_minifloat(
      AGX_ISA_ENCODING_AGX3B, 2, 2, 0x40, AGX3_ALU_HINT_DISCARD, false, 64,
      bytes, &len));
   puts("Native half/minifloat multiply: eight exact Apple vectors and bounds PASS");
}

static void test_native_minifloat_mixed_select(void)
{
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;

   {
      const uint16_t selected[] = {2, 0};
      const bool immediate[] = {false, true};
      const enum agx3_alu_hint hints[] = {
         AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE,
      };
      assert(agx3_alu_fcmp_minifloat_select_mixed(
         AGX_ISA_ENCODING_AGX3B, 0, 0, 0x30, selected, immediate, hints,
         1, false, false, 32, bytes, &len));
      exact(bytes, len, "02010bb1020203c22080");
   }
   {
      const uint16_t selected[] = {4, 0};
      const bool immediate[] = {false, true};
      const enum agx3_alu_hint hints[] = {
         AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE,
      };
      assert(agx3_alu_fcmp_minifloat_select_mixed(
         AGX_ISA_ENCODING_AGX3B, 0, 0, 0x30, selected, immediate, hints,
         1, false, true, 32, bytes, &len));
      exact(bytes, len, "02012bb1020403c22080");
   }
   {
      const uint16_t selected[] = {0, 2};
      const bool immediate[] = {true, false};
      const enum agx3_alu_hint hints[] = {
         AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE,
      };
      assert(agx3_alu_fcmp_minifloat_select_mixed(
         AGX_ISA_ENCODING_AGX3B, 0, 0, 0x40, selected, immediate, hints,
         1, false, false, 32, bytes, &len));
      exact(bytes, len, "02010bc1228003c20002");
   }
   {
      /* Fresh G17P vector witness for `a >= 0.25 ? v : 0`. */
      const uint16_t selected[] = {0, 0};
      const bool immediate[] = {false, true};
      const enum agx3_alu_hint hints[] = {
         AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD,
         AGX3_ALU_HINT_NONE,
      };
      assert(agx3_alu_fcmp_minifloat_select_mixed(
         AGX_ISA_ENCODING_AGX3B, 0, 8, 0x10, selected, immediate, hints,
         5, false, false, 32, bytes, &len));
      exact(bytes, len, "02090b91248002c2");
   }
   {
      /* Exact native `a >= 15.5 ? v : 0` alpha-coverage boundary. */
      const uint16_t selected[] = {0, 0};
      const bool immediate[] = {false, true};
      const enum agx3_alu_hint hints[] = {
         AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD,
         AGX3_ALU_HINT_NONE,
      };
      assert(agx3_alu_fcmp_minifloat_select_mixed(
         AGX_ISA_ENCODING_AGX3B, 0, 4, 0x6f, selected, immediate, hints,
         5, false, false, 32, bytes, &len));
      exact(bytes, len, "02050bef248012c2");
   }
   {
      const uint16_t selected[] = {255, 0};
      const bool immediate[] = {true, true};
      const enum agx3_alu_hint hints[] = {
         AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE,
      };
      assert(agx3_alu_fcmp_minifloat_select_mixed(
         AGX_ISA_ENCODING_AGX3B, 2, 2, 0x30, selected, immediate, hints,
         0, true, false, 32, bytes, &len));
      exact(bytes, len, "12030db1a6ff00c22080");
      assert(agx3_alu_fcmp_minifloat_select_mixed(
         AGX_ISA_ENCODING_AGX3B, 2, 2, 0x30, selected, immediate, hints,
         1, true, false, 32, bytes, &len));
      exact(bytes, len, "12030db1a2ff03c22080");
      assert(agx3_alu_fcmp_minifloat_select_mixed(
         AGX_ISA_ENCODING_AGX3B, 3, 2, 0x30, selected, immediate, hints,
         0, true, false, 32, bytes, &len));
      exact(bytes, len, "1a030db1a6ff00c22080");
   }

   /* Fresh applegpu_g17p GTK box-shadow witnesses. Apple canonicalizes
    * `-x < -0` to `x > +0`; the latter is byte-identical to a source written
    * directly as `x > 0`. The remaining vectors independently cover every
    * ordered condition, signed minifloats, both lifetime hints, both half
    * destination endpoints, arbitrary selected bytes and texture/load waits.
    */
   {
      static const struct {
         uint16_t dest, compare;
         uint8_t threshold, if_true, if_false, condition;
         enum agx3_alu_hint hint;
         bool dest_cache;
         uint16_t wait;
         const char *bytes;
      } native[] = {
         {2, 2, 0x00, 1, 0, 0, AGX3_ALU_HINT_DISCARD, false, 32,
          "12030d81268100c22080"},
         {2, 2, 0x00, 1, 0, 1, AGX3_ALU_HINT_DISCARD, false, 32,
          "12030d81228103c22080"},
         {2, 2, 0x00, 1, 0, 2, AGX3_ALU_HINT_DISCARD, false, 32,
          "12030d81228102c22080"},
         {2, 2, 0x00, 1, 0, 5, AGX3_ALU_HINT_DISCARD, false, 32,
          "12030d81268102c22080"},
         {2, 2, 0x00, 1, 0, 6, AGX3_ALU_HINT_DISCARD, false, 32,
          "12030d81268103c22080"},
         {2, 2, 0xb0, 1, 0, 2, AGX3_ALU_HINT_DISCARD, false, 32,
          "12031db1228102c22080"},
         {2, 2, 0x20, 1, 0, 6, AGX3_ALU_HINT_DISCARD, false, 32,
          "12030da1268103c22080"},
         {2, 2, 0x30, 1, 0, 2, AGX3_ALU_HINT_DISCARD, false, 32,
          "12030db1228102c22080"},
         {2, 2, 0xb0, 1, 0, 1, AGX3_ALU_HINT_DISCARD, false, 32,
          "12031db1228103c22080"},
         {4, 0, 0x00, 1, 0, 2, AGX3_ALU_HINT_NONE, false, 32,
          "22010581228102c22080"},
         {2, 4, 0x00, 1, 0, 2, AGX3_ALU_HINT_CACHE, false, 32,
          "12850581228102c22080"},
         {4, 0, 0x00, 1, 0, 2, AGX3_ALU_HINT_CACHE, false, 0,
          "22810581228102022080"},
         {3, 2, 0x00, 1, 0, 2, AGX3_ALU_HINT_DISCARD, false, 32,
          "1a030d81228102c22080"},
         {2, 2, 0x00, 5, 7, 2, AGX3_ALU_HINT_DISCARD, false, 32,
          "12030d81228502c22087"},
         {2, 2, 0x00, 1, 0, 2, AGX3_ALU_HINT_DISCARD, false, 1,
          "12030d81228102222080"},
      };
      const bool immediate[] = {true, true};
      for (unsigned i = 0; i < sizeof(native) / sizeof(native[0]); ++i) {
         const uint16_t selected[] = {
            native[i].if_true, native[i].if_false};
         const enum agx3_alu_hint hints[] = {
            native[i].hint, AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE};
         assert(agx3_alu_fcmp_minifloat_select_mixed(
            AGX_ISA_ENCODING_AGX3B, native[i].dest, native[i].compare,
            native[i].threshold, selected, immediate, hints,
            native[i].condition, true, native[i].dest_cache, native[i].wait,
            bytes, &len));
         exact(bytes, len, native[i].bytes);
      }

      /* Apple canonicalizes `1 < -x ? 0 : y` to `x < -1 ? 0 : y`. */
      const uint16_t signed_selected[] = {0, 4};
      const bool signed_immediate[] = {true, false};
      const enum agx3_alu_hint signed_hints[] = {
         AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_NONE,
         AGX3_ALU_HINT_DISCARD};
      assert(agx3_alu_fcmp_minifloat_select_mixed(
         AGX_ISA_ENCODING_AGX3B, 0, 0, 0xb0, signed_selected,
         signed_immediate, signed_hints, 1, false, false, 32, bytes, &len));
      exact(bytes, len, "02011bb1228003c28004");
   }

   /* Fresh full-result sweep from the same Apple build. Each condition is
    * pinned with floating zero on either side of the selected register. */
   {
      static const struct {
         uint8_t condition;
         bool zero_true;
         const char *bytes;
      } native[] = {
         {0, true,  "02010bb1268000c28004"},
         {1, true,  "02010bb1228003c28004"},
         {2, true,  "02010bb1228002c28004"},
         {5, true,  "02010bb1268002c28004"},
         {6, true,  "02010bb1268003c28004"},
         {0, false, "02010bb1860400c22080"},
         {1, false, "02010bb1820403c22080"},
         {2, false, "02010bb1820402c22080"},
         {5, false, "02010bb1860402c22080"},
         {6, false, "02010bb1860403c22080"},
      };
      for (unsigned i = 0; i < sizeof(native) / sizeof(native[0]); ++i) {
         const uint16_t selected[] = {
            native[i].zero_true ? 0 : 4,
            native[i].zero_true ? 4 : 0,
         };
         const bool immediate[] = {
            native[i].zero_true, !native[i].zero_true,
         };
         const enum agx3_alu_hint hints[] = {
            AGX3_ALU_HINT_DISCARD,
            native[i].zero_true ? AGX3_ALU_HINT_NONE
                                : AGX3_ALU_HINT_DISCARD,
            native[i].zero_true ? AGX3_ALU_HINT_DISCARD
                                : AGX3_ALU_HINT_NONE,
         };
         assert(agx3_alu_fcmp_minifloat_select_mixed(
            AGX_ISA_ENCODING_AGX3B, 0, 0, 0x30, selected, immediate,
            hints, native[i].condition, false, false, 32, bytes, &len));
         exact(bytes, len, native[i].bytes);
      }
   }

   /* Complete admitted half-result field product. */
   {
      static const unsigned conditions[] = {0, 1, 2, 5, 6};
      static const unsigned waits[] = {0, 1, 2, 4, 8, 16, 32};
      const bool immediate[] = {true, true};
      const uint16_t selected[] = {255, 7};
      for (unsigned c = 0; c < sizeof(conditions) / sizeof(conditions[0]); ++c)
         for (unsigned hint = 0; hint <= AGX3_ALU_HINT_CACHE; ++hint)
            for (unsigned dest_cache = 0; dest_cache < 2; ++dest_cache)
               for (unsigned dest_odd = 0; dest_odd < 2; ++dest_odd)
                  for (unsigned w = 0; w < sizeof(waits) / sizeof(waits[0]);
                       ++w) {
                     const enum agx3_alu_hint hints[] = {
                        (enum agx3_alu_hint)hint, AGX3_ALU_HINT_NONE,
                        AGX3_ALU_HINT_NONE};
                     assert(agx3_alu_fcmp_minifloat_select_mixed(
                        AGX_ISA_ENCODING_AGX3B, dest_odd ? 255 : 254, 254,
                        0x80, selected, immediate, hints, conditions[c], true,
                        dest_cache, waits[w], bytes, &len));
                     assert(len == 10);
                  }

      const enum agx3_alu_hint dead[] = {
         AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE};
      for (unsigned value = 0; value < 256; ++value) {
         const uint16_t span[] = {value, 255 - value};
         assert(agx3_alu_fcmp_minifloat_select_mixed(
            AGX_ISA_ENCODING_AGX3B, value, value & 0xfe, value, span,
            immediate, dead, 2, true, value & 1, 1u << (value % 6),
            bytes, &len));
      }
   }

   /* Complete full-result condition/orientation/lifetime/wait product. */
   {
      static const unsigned conditions[] = {0, 1, 2, 5, 6};
      static const unsigned waits[] = {0, 32};
      for (unsigned c = 0; c < sizeof(conditions) / sizeof(conditions[0]); ++c)
         for (unsigned zero_true = 0; zero_true < 2; ++zero_true)
            for (unsigned compare_hint = 0;
                 compare_hint <= AGX3_ALU_HINT_CACHE; ++compare_hint)
               for (unsigned selected_hint = 0;
                    selected_hint <= AGX3_ALU_HINT_CACHE; ++selected_hint)
                  for (unsigned dest_cache = 0; dest_cache < 2; ++dest_cache)
                     for (unsigned w = 0;
                          w < sizeof(waits) / sizeof(waits[0]); ++w) {
                        const uint16_t selected[] = {
                           zero_true ? 0 : 252,
                           zero_true ? 252 : 0,
                        };
                        const bool immediate[] = {
                           (bool)zero_true, !zero_true,
                        };
                        const enum agx3_alu_hint hints[] = {
                           (enum agx3_alu_hint)compare_hint,
                           zero_true
                              ? AGX3_ALU_HINT_NONE
                              : (enum agx3_alu_hint)selected_hint,
                           zero_true
                              ? (enum agx3_alu_hint)selected_hint
                              : AGX3_ALU_HINT_NONE,
                        };
                        assert(agx3_alu_fcmp_minifloat_select_mixed(
                           AGX_ISA_ENCODING_AGX3B, 254, 254, 0x80,
                           selected, immediate, hints, conditions[c], false,
                           dest_cache, waits[w], bytes, &len));
                     }
   }

   {
      const uint16_t gtk_selected[] = {1, 0};
      const bool gtk_immediate[] = {true, true};
      const enum agx3_alu_hint gtk_hints[] = {
         AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE};
      assert(agx3_alu_fcmp_minifloat_select_mixed(
         AGX_ISA_ENCODING_AGX3B, 44, 44, 0, gtk_selected, gtk_immediate,
         gtk_hints, 2, true, false, 0, bytes, &len));
      exact(bytes, len, "622d4581228102022080");
   }
   {
      const uint16_t full_selected[] = {0, 0};
      const bool full_immediate[] = {false, true};
      const enum agx3_alu_hint full_hints[] = {
         AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD,
         AGX3_ALU_HINT_NONE};
      assert(agx3_alu_fcmp_minifloat_select_mixed(
         AGX_ISA_ENCODING_AGX3B, 14, 2, 0xef, full_selected,
         full_immediate, full_hints, 6, false, false, 0, bytes, &len));
      exact(bytes, len, "72031bef860013022080");
   }
   {
      const uint16_t linear_selected[] = {0, 10};
      const bool linear_immediate[] = {true, false};
      const enum agx3_alu_hint linear_hints[] = {
         AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE};
      assert(agx3_alu_fcmp_minifloat_select_mixed(
         AGX_ISA_ENCODING_AGX3B, 10, 22, 0x30, linear_selected,
         linear_immediate, linear_hints, 5, false, false, 0, bytes, &len));
      exact(bytes, len, "529703b126800202000a");
   }
   {
      const uint16_t conic_selected[] = {0, 24};
      const bool conic_immediate[] = {true, false};
      const enum agx3_alu_hint conic_hints[] = {
         AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_NONE,
         AGX3_ALU_HINT_DISCARD};
      assert(agx3_alu_fcmp_minifloat_select_mixed(
         AGX_ISA_ENCODING_AGX3B, 24, 12, 0x30, conic_selected,
         conic_immediate, conic_hints, 6, false, false, 0, bytes, &len));
      exact(bytes, len, "c28d03b1268003028018");
   }

   const uint16_t selected[] = {2, 0};
   const bool immediate[] = {false, true};
   const enum agx3_alu_hint hints[] = {
      AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE,
   };
   assert(!agx3_alu_fcmp_minifloat_select_mixed(
      AGX_ISA_ENCODING_AGX3C, 0, 0, 0x30, selected, immediate, hints, 1,
      false, false, 0, bytes, &len));
   assert(agx3_alu_fcmp_minifloat_select_mixed(
      AGX_ISA_ENCODING_AGX3B, 0, 0, 0x30, selected, immediate, hints, 2,
      false, false, 0, bytes, &len));
   {
      const uint16_t nonzero[] = {7, 2};
      const bool zero_first[] = {true, false};
      assert(!agx3_alu_fcmp_minifloat_select_mixed(
         AGX_ISA_ENCODING_AGX3B, 0, 0, 0x30, nonzero, zero_first, hints, 2,
         false, false, 0, bytes, &len));
   }
   {
      const uint16_t selected_half[] = {1, 0};
      const bool immediate_half[] = {true, true};
      const enum agx3_alu_hint half_hints[] = {
         AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE};
      for (unsigned condition = 3; condition <= 7; ++condition) {
         if (condition == 5 || condition == 6)
            continue;
         assert(!agx3_alu_fcmp_minifloat_select_mixed(
            AGX_ISA_ENCODING_AGX3B, 0, 0, 0, selected_half,
            immediate_half, half_hints, condition, true, false, 0, bytes,
            &len));
      }
      assert(!agx3_alu_fcmp_minifloat_select_mixed(
         AGX_ISA_ENCODING_AGX3B, 0, 0, 0, selected_half, immediate_half,
         half_hints, 2, true, false, 3, bytes, &len));
      assert(!agx3_alu_fcmp_minifloat_select_mixed(
         AGX_ISA_ENCODING_AGX3B, 0, 0, 0, selected_half, immediate_half,
         half_hints, 2, true, false, 64, bytes, &len));
      assert(!agx3_alu_fcmp_minifloat_select_mixed(
         AGX_ISA_ENCODING_AGX3B, 0, 1, 0, selected_half, immediate_half,
         half_hints, 2, true, false, 0, bytes, &len));
      assert(!agx3_alu_fcmp_minifloat_select_mixed(
         AGX_ISA_ENCODING_AGX3B, 256, 0, 0, selected_half,
         immediate_half, half_hints, 2, true, false, 0, bytes,
         &len));
   }
   /* The three measured Hyprland post-RA physical-register shapes. */
   {
      const uint16_t blur_selected[] = {24, 0};
      const bool blur_immediate[] = {false, true};
      const enum agx3_alu_hint blur_hints[] = {
         AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE,
      };
      assert(agx3_alu_fcmp_minifloat_select_mixed(
         AGX_ISA_ENCODING_AGX3B, 24, 20, 0x30, blur_selected,
         blur_immediate, blur_hints, 1, false, true, 0, bytes, &len));
   }
   {
      const uint16_t ext_selected[] = {255, 0};
      const bool ext_immediate[] = {true, true};
      const enum agx3_alu_hint ext_hints[] = {
         AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE,
      };
      assert(agx3_alu_fcmp_minifloat_select_mixed(
         AGX_ISA_ENCODING_AGX3B, 1, 14, 0x30, ext_selected,
         ext_immediate, ext_hints, 0, true, true, 0, bytes, &len));
   }
   {
      const uint16_t glitch_selected[] = {0, 2};
      const bool glitch_immediate[] = {true, false};
      const enum agx3_alu_hint glitch_hints[] = {
         AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_DISCARD,
      };
      assert(agx3_alu_fcmp_minifloat_select_mixed(
         AGX_ISA_ENCODING_AGX3B, 2, 22, 0x40, glitch_selected,
         glitch_immediate, glitch_hints, 1, false, true, 0, bytes, &len));
   }
   puts("Native minifloat mixed select: 34 exact Apple vectors, 676 half + "
        "360 full legal forms and strict bounds PASS");
}

static void test_native_half_minifloat_select_f32(void)
{
   static const struct {
      unsigned condition;
      const char *bytes;
   } vectors[] = {
      {6, "02000fa1060203c20004"},
      {1, "02000fa1020203c20004"},
      {2, "02000fa1020202c20004"},
      {5, "02000fa1060202c20004"},
      {0, "02000fa1060200c20004"},
   };
   const uint16_t selected[] = {2, 4};
   const bool selected_immediate[] = {false, false};
   const enum agx3_alu_hint hints[] = {
      AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE,
   };
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   for (unsigned i = 0; i < sizeof(vectors) / sizeof(vectors[0]); ++i) {
      assert(agx3_alu_fcmp_half_minifloat_select_f32(
         AGX_ISA_ENCODING_AGX3B, 0, 0, 32, selected, selected_immediate, hints,
         vectors[i].condition, false, 32, bytes, &len));
      exact(bytes, len, vectors[i].bytes);
   }

   const uint16_t register_zero[] = {2, 0};
   const bool register_zero_immediate[] = {false, true};
   assert(agx3_alu_fcmp_half_minifloat_select_f32(
      AGX_ISA_ENCODING_AGX3B, 0, 0, 0, register_zero,
      register_zero_immediate, hints, 1, false, 32, bytes, &len));
   exact(bytes, len, "02000b81020203c22080");
   const uint16_t zero_register[] = {0, 2};
   const bool zero_register_immediate[] = {true, false};
   assert(agx3_alu_fcmp_half_minifloat_select_f32(
      AGX_ISA_ENCODING_AGX3B, 0, 0, 0, zero_register,
      zero_register_immediate, hints, 1, false, 32, bytes, &len));
   exact(bytes, len, "02000b81228003c20002");

   assert(!agx3_alu_fcmp_half_minifloat_select_f32(
      AGX_ISA_ENCODING_AGX3C, 0, 0, 32, selected, selected_immediate, hints,
      6, false, 32, bytes, &len));
   assert(!agx3_alu_fcmp_half_minifloat_select_f32(
      AGX_ISA_ENCODING_AGX3B, 1, 0, 32, selected, selected_immediate, hints,
      6, false, 32, bytes, &len));
   assert(!agx3_alu_fcmp_half_minifloat_select_f32(
      AGX_ISA_ENCODING_AGX3B, 0, 256, 32, selected, selected_immediate, hints,
      6, false, 32, bytes, &len));
   const uint16_t odd_selected[] = {1, 4};
   assert(!agx3_alu_fcmp_half_minifloat_select_f32(
      AGX_ISA_ENCODING_AGX3B, 0, 0, 32, odd_selected, selected_immediate,
      hints, 6, false, 32, bytes, &len));
   assert(!agx3_alu_fcmp_half_minifloat_select_f32(
      AGX_ISA_ENCODING_AGX3B, 0, 0, 32, selected, selected_immediate, hints,
      3, false, 32, bytes, &len));
   assert(!agx3_alu_fcmp_half_minifloat_select_f32(
      AGX_ISA_ENCODING_AGX3B, 0, 0, 32, selected, selected_immediate, hints,
      6, false, 1, bytes, &len));
   const enum agx3_alu_hint bad_hints[] = {
      (enum agx3_alu_hint)3, AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE};
   assert(!agx3_alu_fcmp_half_minifloat_select_f32(
      AGX_ISA_ENCODING_AGX3B, 0, 0, 32, selected, selected_immediate,
      bad_hints, 6, false, 32, bytes, &len));
   const bool both_immediate[] = {true, true};
   assert(!agx3_alu_fcmp_half_minifloat_select_f32(
      AGX_ISA_ENCODING_AGX3B, 0, 0, 32, selected, both_immediate, hints, 6,
      false, 32, bytes, &len));
   puts("Native half/minifloat full FSEL: seven exact Apple vectors and bounds PASS");
}

static void test_native_fadd_abs_mixed_half(void)
{
   static const struct {
      uint16_t dest, f32, f16, wait;
      enum agx3_alu_hint hint32, hint16;
      bool dest_cache;
      const char *bytes;
   } vectors[] = {
      {0, 0, 4, 32, AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD, false,
       "09011c04020000800100"},
      {0, 0, 4, 1, AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_CACHE, false,
       "09010c84022000000100"},
      {0, 2, 0, 32, AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_DISCARD, false,
       "09031400020000800100"},
      {0, 0, 2, 32, AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD, true,
       "09013c02020000800100"},
      {0, 2, 3, 1, AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE, false,
       "09030402122000000100"},
   };
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   for (unsigned i = 0; i < sizeof(vectors) / sizeof(vectors[0]); ++i) {
      assert(agx3_alu_fadd_abs_mixed_half(
         AGX_ISA_ENCODING_AGX3B, vectors[i].dest, vectors[i].f32,
         vectors[i].f16, vectors[i].hint32, vectors[i].hint16,
         vectors[i].dest_cache, vectors[i].wait, bytes, &len));
      exact(bytes, len, vectors[i].bytes);
   }

   assert(agx3_alu_fadd_abs_mixed_half(
      AGX_ISA_ENCODING_AGX3B, 8, 8, 28, AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_CACHE, false, 0, bytes, &len));
   assert(!agx3_alu_fadd_abs_mixed_half(
      AGX_ISA_ENCODING_AGX3C, 8, 8, 28, AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_CACHE, false, 0, bytes, &len));
   assert(!agx3_alu_fadd_abs_mixed_half(
      AGX_ISA_ENCODING_AGX3B, 9, 8, 28, AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_CACHE, false, 0, bytes, &len));
   assert(!agx3_alu_fadd_abs_mixed_half(
      AGX_ISA_ENCODING_AGX3B, 8, 8, 28, AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_CACHE, false, 64, bytes, &len));
   puts("Native abs-f32 plus half add: five exact Apple vectors and bounds PASS");
}

static void test_native_fadd_mixed_half(void)
{
   static const struct {
      uint16_t dest, f32, f16, wait;
      enum agx3_alu_hint hint32, hint16;
      bool dest_cache;
      const char *bytes;
   } vectors[] = {
      /* add_dead */
      {0, 0, 4, 32, AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD, false,
       "09011c0400c0"},
      /* add_half_live */
      {0, 0, 8, 1, AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD, false,
       "09011c080020"},
      /* add_float_live */
      {0, 2, 0, 32, AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_DISCARD, false,
       "0903140000c0"},
      /* add_half_pair, second channel: physical half 7 is intentionally odd. */
      {2, 2, 7, 0, AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD, false,
       "19031c061000"},
      /* add_reuse_odd: the odd half and destination are both retained. */
      {0, 0, 5, 32, AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_CACHE, true,
       "09012c8410c0"},
   };
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   for (unsigned i = 0; i < sizeof(vectors) / sizeof(vectors[0]); ++i) {
      assert(agx3_alu_fadd_mixed_half(
         AGX_ISA_ENCODING_AGX3B, vectors[i].dest, vectors[i].f32,
         vectors[i].f16, vectors[i].hint32, vectors[i].hint16,
         vectors[i].dest_cache, vectors[i].wait, bytes, &len));
      exact(bytes, len, vectors[i].bytes);
   }

   assert(!agx3_alu_fadd_mixed_half(
      AGX_ISA_ENCODING_AGX3C, 0, 0, 0, AGX3_ALU_HINT_NONE,
      AGX3_ALU_HINT_NONE, false, 0, bytes, &len));
   assert(!agx3_alu_fadd_mixed_half(
      AGX_ISA_ENCODING_AGX3B, 1, 0, 0, AGX3_ALU_HINT_NONE,
      AGX3_ALU_HINT_NONE, false, 0, bytes, &len));
   assert(!agx3_alu_fadd_mixed_half(
      AGX_ISA_ENCODING_AGX3B, 0, 1, 0, AGX3_ALU_HINT_NONE,
      AGX3_ALU_HINT_NONE, false, 0, bytes, &len));
   assert(!agx3_alu_fadd_mixed_half(
      AGX_ISA_ENCODING_AGX3B, 0, 0, 256, AGX3_ALU_HINT_NONE,
      AGX3_ALU_HINT_NONE, false, 0, bytes, &len));
   assert(!agx3_alu_fadd_mixed_half(
      AGX_ISA_ENCODING_AGX3B, 0, 0, 0, (enum agx3_alu_hint)3,
      AGX3_ALU_HINT_NONE, false, 0, bytes, &len));
   assert(!agx3_alu_fadd_mixed_half(
      AGX_ISA_ENCODING_AGX3B, 0, 0, 0, AGX3_ALU_HINT_NONE,
      (enum agx3_alu_hint)3, false, 0, bytes, &len));
   assert(!agx3_alu_fadd_mixed_half(
      AGX_ISA_ENCODING_AGX3B, 0, 0, 0, AGX3_ALU_HINT_NONE,
      AGX3_ALU_HINT_NONE, false, 64, bytes, &len));
   puts("Native f32 plus half add: five exact Apple vectors and bounds PASS");
}

static void test_native_half_add(void)
{
   {
      const uint16_t src[] = {2, 3};
      const enum agx3_alu_hint hints[] = {
         AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD,
      };
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      assert(agx3_alu_hadd(AGX_ISA_ENCODING_AGX3B, 2, src, hints, false, 32,
                           bytes, &len));
      exact(bytes, len, "10021c0300c0");
   }
   {
      const uint16_t src[] = {2, 3};
      const enum agx3_alu_hint hints[] = {
         AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_CACHE,
      };
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      assert(agx3_alu_hadd(AGX_ISA_ENCODING_AGX3B, 1, src, hints, true, 32,
                           bytes, &len));
      exact(bytes, len, "0882248300c0");
   }

   const uint16_t src[] = {2, 3};
   const enum agx3_alu_hint hints[] = {
      AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE,
   };
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   assert(!agx3_alu_hadd(AGX_ISA_ENCODING_AGX3C, 0, src, hints, false, 0,
                         bytes, &len));
   assert(!agx3_alu_hadd(AGX_ISA_ENCODING_AGX3B, 256, src, hints, false, 0,
                         bytes, &len));
   assert(!agx3_alu_hadd(AGX_ISA_ENCODING_AGX3B, 0, src, hints, false, 0x40,
                         bytes, &len));
   puts("Native half add: two exact Apple vectors and bounds PASS");
}

static void test_native_half_saturate(void)
{
   static const struct {
      uint16_t dest, src, wait;
      const char *bytes;
   } witnesses[] = {
      {0, 0, 1, "00001c8001220002"},
      {1, 1, 0, "08011c8001020002"},
      {2, 2, 0, "10021c8001020002"},
      {3, 3, 0, "18031c8001020002"},
   };
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   for (unsigned i = 0; i < sizeof(witnesses) / sizeof(witnesses[0]); ++i) {
      assert(agx3_alu_hadd_saturate_negzero(
         AGX_ISA_ENCODING_AGX3B, witnesses[i].dest, witnesses[i].src,
         AGX3_ALU_HINT_DISCARD, false, witnesses[i].wait, bytes, &len));
      exact(bytes, len, witnesses[i].bytes);
   }

   assert(!agx3_alu_hadd_saturate_negzero(
      AGX_ISA_ENCODING_AGX3C, 0, 0, AGX3_ALU_HINT_NONE, false, 0, bytes,
      &len));
   assert(!agx3_alu_hadd_saturate_negzero(
      AGX_ISA_ENCODING_AGX3B, 256, 0, AGX3_ALU_HINT_NONE, false, 0, bytes,
      &len));
   assert(!agx3_alu_hadd_saturate_negzero(
      AGX_ISA_ENCODING_AGX3B, 0, 256, AGX3_ALU_HINT_NONE, false, 0, bytes,
      &len));
   assert(!agx3_alu_hadd_saturate_negzero(
      AGX_ISA_ENCODING_AGX3B, 0, 0, AGX3_ALU_HINT_NONE, false, 0x40, bytes,
      &len));
   puts("Native half saturate: four exact Apple vectors and bounds PASS");
}

static void test_native_half_inverse(void)
{
   static const struct {
      uint16_t dest, src, wait;
      enum agx3_alu_hint hint;
      const char *bytes;
   } witnesses[] = {
      {0, 0, 1, AGX3_ALU_HINT_DISCARD, "00b034008028"},
      {1, 1, 0, AGX3_ALU_HINT_DISCARD, "08b034018008"},
      {2, 2, 0, AGX3_ALU_HINT_DISCARD, "10b034028008"},
      {3, 3, 0, AGX3_ALU_HINT_DISCARD, "18b034038008"},
      {0, 1, 1, AGX3_ALU_HINT_CACHE, "00b024818028"},
   };
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   for (unsigned i = 0; i < sizeof(witnesses) / sizeof(witnesses[0]); ++i) {
      assert(agx3_alu_hadd_one_neg(
         AGX_ISA_ENCODING_AGX3B, witnesses[i].dest, witnesses[i].src,
         witnesses[i].hint, true, witnesses[i].wait, bytes, &len));
      exact(bytes, len, witnesses[i].bytes);
   }

   assert(!agx3_alu_hadd_one_neg(AGX_ISA_ENCODING_AGX3C, 0, 0,
                                  AGX3_ALU_HINT_NONE, false, 0, bytes, &len));
   assert(!agx3_alu_hadd_one_neg(AGX_ISA_ENCODING_AGX3B, 256, 0,
                                  AGX3_ALU_HINT_NONE, false, 0, bytes, &len));
   assert(!agx3_alu_hadd_one_neg(AGX_ISA_ENCODING_AGX3B, 0, 256,
                                  AGX3_ALU_HINT_NONE, false, 0, bytes, &len));
   assert(!agx3_alu_hadd_one_neg(AGX_ISA_ENCODING_AGX3B, 0, 0,
                                  AGX3_ALU_HINT_NONE, false, 0x40, bytes,
                                  &len));
   puts("Native half inverse: five exact Apple vectors and bounds PASS");
}

static void test_native_half_fma(void)
{
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   const uint16_t live_src[] = {2, 3, 4};
   const enum agx3_alu_hint live_hints[] = {
      AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_CACHE,
   };
   assert(agx3_alu_hfma(AGX_ISA_ENCODING_AGX3B, 1, live_src, live_hints,
                        false, false, true, 0, bytes, &len));
   exact(bytes, len, "08822683018400c0");

   const uint16_t dead_src[] = {0, 1, 2};
   const enum agx3_alu_hint dead_hints[] = {
      AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_DISCARD,
   };
   assert(agx3_alu_hfma(AGX_ISA_ENCODING_AGX3B, 0, dead_src, dead_hints,
                        false, false, false, 0, bytes, &len));
   exact(bytes, len, "00001e01810200c0");
   assert(agx3_alu_hfma(AGX_ISA_ENCODING_AGX3B, 1, live_src, live_hints,
                        false, true, true, 0, bytes, &len));
   exact(bytes, len, "08822683118400c0");
   assert(agx3_alu_hfma(AGX_ISA_ENCODING_AGX3B, 0, dead_src, dead_hints,
                        false, true, false, 0, bytes, &len));
   exact(bytes, len, "00001e01910200c0");
   assert(agx3_alu_hfma(AGX_ISA_ENCODING_AGX3B, 1, live_src, live_hints,
                        true, false, true, 0, bytes, &len));
   exact(bytes, len, "08822683018400c8");
   const uint16_t neg0_src[] = {3, 2, 4};
   assert(agx3_alu_hfma(AGX_ISA_ENCODING_AGX3B, 1, neg0_src, live_hints,
                        true, false, true, 0, bytes, &len));
   exact(bytes, len, "08832682018400c8");
   assert(!agx3_alu_hfma(AGX_ISA_ENCODING_AGX3C, 0, dead_src, dead_hints,
                         false, false, false, 0, bytes, &len));
   assert(!agx3_alu_hfma(AGX_ISA_ENCODING_AGX3B, 256, dead_src, dead_hints,
                         false, false, false, 0, bytes, &len));
   assert(!agx3_alu_hfma(AGX_ISA_ENCODING_AGX3B, 0, dead_src, dead_hints,
                         false, false, false, 0x40, bytes, &len));
   puts("Native half FMA/FMS: six complete Apple modifier/lifetime vectors "
        "and refusals PASS");
}

static void test_native_half_fma_minifloat_addend(void)
{
   /* applegpu-nt 32023.917, apple-g17p, fma(a, b, half(1.5/2.5)). */
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   const uint16_t src[] = {2, 3};
   const enum agx3_alu_hint dead_hints[] = {
      AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD,
   };

   assert(agx3_alu_hfma_addend_minifloat(
      AGX_ISA_ENCODING_AGX3B, 2, src, 56, dead_hints, false, 32, bytes,
      &len));
   exact(bytes, len, "10021e0321b800c0");
   assert(agx3_alu_hfma_addend_minifloat(
      AGX_ISA_ENCODING_AGX3B, 2, src, 68, dead_hints, false, 32, bytes,
      &len));
   exact(bytes, len, "10021e0321c400c0");

   assert(!agx3_alu_hfma_addend_minifloat(
      AGX_ISA_ENCODING_AGX3C, 2, src, 56, dead_hints, false, 32, bytes,
      &len));
   assert(!agx3_alu_hfma_addend_minifloat(
      AGX_ISA_ENCODING_AGX3B, 2, src, 256, dead_hints, false, 32, bytes,
      &len));
   assert(!agx3_alu_hfma_addend_minifloat(
      AGX_ISA_ENCODING_AGX3B, 2, src, 56, dead_hints, false, 64, bytes,
      &len));
   puts("Native half FMA minifloat addend: two exact Apple vectors and "
        "bounds PASS");
}

static void test_native_widen_half(void)
{
   const char *golden[]={"09021c8100c2","19021c810802","0984148100c2","198414810802"};
   for (unsigned i=0;i<4;i++) {
      uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
      assert(agx3_alu_widen_half(AGX_ISA_ENCODING_AGX3B,(i&1)*2,i+2,
         i<2?AGX3_ALU_HINT_DISCARD:AGX3_ALU_HINT_CACHE,false,false,false,
         (i&1)?0:32,bytes,&len));
      exact(bytes,len,golden[i]);
   }
   uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
   assert(agx3_alu_widen_half(AGX_ISA_ENCODING_AGX3B,0,4,
      AGX3_ALU_HINT_CACHE,true,false,false,0,bytes,&len));
   exact(bytes,len,"09841481020200800100");
   assert(agx3_alu_widen_half(AGX_ISA_ENCODING_AGX3B,0,0,
      AGX3_ALU_HINT_DISCARD,true,false,false,0,bytes,&len));
   exact(bytes,len,"09001c81020200800100");
   assert(agx3_alu_widen_half(AGX_ISA_ENCODING_AGX3B,0,4,
      AGX3_ALU_HINT_CACHE,false,true,false,0,bytes,&len));
   exact(bytes,len,"0984148101020280");
   assert(agx3_alu_widen_half(AGX_ISA_ENCODING_AGX3B,0,0,
      AGX3_ALU_HINT_DISCARD,false,true,false,0,bytes,&len));
   exact(bytes,len,"09001c8101020280");
   assert(!agx3_alu_widen_half(AGX_ISA_ENCODING_AGX3B,0,0,
      AGX3_ALU_HINT_DISCARD,true,true,false,0,bytes,&len));
   puts("Native half widening/abs/neg: eight Apple vectors plus combined-modifier refusal PASS");
}

static void test_native_sqrt_step(void)
{
   for (unsigned live=0;live<2;live++) {
      uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
      assert(agx3_alu_sqrt_step(AGX_ISA_ENCODING_AGX3B,live?0:2,live?2:0,
                               AGX3_ALU_HINT_CACHE,true,32,bytes,&len));
      exact(bytes,len,live?"2f015600030492400000":"2f015602030092400000");
   }
   puts("Native sqrt step: two complete Apple live/dead square-root kernels PASS");
}

static void test_native_half_sqrt(void)
{
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;

   assert(agx3_alu_sqrt_step_half(AGX_ISA_ENCODING_AGX3B, 2, 4,
                                  AGX3_ALU_HINT_CACHE, true, 32,
                                  bytes, &len));
   exact(bytes, len, "2f01560203088a400000");
   assert(agx3_alu_sqrt_step_half(AGX_ISA_ENCODING_AGX3B, 2, 4,
                                  AGX3_ALU_HINT_CACHE, true, 0,
                                  bytes, &len));
   exact(bytes, len, "2f01540203088a400000");
   assert(agx3_alu_sqrt_step_half(AGX_ISA_ENCODING_AGX3B, 4, 3,
                                  AGX3_ALU_HINT_CACHE, true, 0,
                                  bytes, &len));
   exact(bytes, len, "2f01540403068a400000");

   assert(agx3_alu_fmul_mixed_half(
      AGX_ISA_ENCODING_AGX3B, 2, 2, 4, AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_NONE, false, 0, bytes, &len));
   exact(bytes, len, "11030904");
   assert(agx3_alu_fmul_mixed_half(
      AGX_ISA_ENCODING_AGX3B, 2, 2, 4, AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_DISCARD, false, 0, bytes, &len));
   exact(bytes, len, "11031904");
   assert(agx3_alu_fmul_mixed_half(
      AGX_ISA_ENCODING_AGX3B, 3, 4, 3, AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_DISCARD, false, 0, bytes, &len));
   exact(bytes, len, "11051d021400");
   assert(agx3_alu_fmul_mixed_half(
      AGX_ISA_ENCODING_AGX3B, 2, 4, 2, AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_DISCARD, true, 0, bytes, &len));
   exact(bytes, len, "11053902");
   assert(agx3_alu_fmul_mixed_half(
      AGX_ISA_ENCODING_AGX3B, 8, 4, 0, AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_DISCARD, false, 0, bytes, &len));
   exact(bytes, len, "41051900");
   assert(agx3_alu_fmul_mixed_half(
      AGX_ISA_ENCODING_AGX3B, 8, 4, 0, AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_DISCARD, false, 1, bytes, &len));
   exact(bytes, len, "41051d000020");

   memset(bytes, 0xa5, sizeof(bytes));
   assert(!agx3_alu_sqrt_step_half(AGX_ISA_ENCODING_AGX3C, 2, 4,
                                   AGX3_ALU_HINT_CACHE, true, 0,
                                   bytes, &len));
   assert(!agx3_alu_sqrt_step_half(AGX_ISA_ENCODING_AGX3B, 3, 4,
                                   AGX3_ALU_HINT_CACHE, true, 0,
                                   bytes, &len));
   assert(!agx3_alu_sqrt_step_half(AGX_ISA_ENCODING_AGX3B, 2, 4,
                                   AGX3_ALU_HINT_DISCARD, true, 0,
                                   bytes, &len));
   assert(!agx3_alu_sqrt_step_half(AGX_ISA_ENCODING_AGX3B, 2, 4,
                                   AGX3_ALU_HINT_CACHE, false, 0,
                                   bytes, &len));
   assert(!agx3_alu_sqrt_step_half(AGX_ISA_ENCODING_AGX3B, 2, 4,
                                   AGX3_ALU_HINT_CACHE, true, 1,
                                   bytes, &len));
   assert(!agx3_alu_fmul_mixed_half(
      AGX_ISA_ENCODING_AGX3B, 2, 3, 4, AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_NONE, false, 0, bytes, &len));
   assert(!agx3_alu_fmul_mixed_half(
      AGX_ISA_ENCODING_AGX3B, 2, 2, 4, AGX3_ALU_HINT_CACHE,
      AGX3_ALU_HINT_NONE, false, 0, bytes, &len));
   assert(!agx3_alu_fmul_mixed_half(
      AGX_ISA_ENCODING_AGX3B, 2, 2, 4, AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_CACHE, false, 0, bytes, &len));
   assert(!agx3_alu_fmul_mixed_half(
      AGX_ISA_ENCODING_AGX3B, 2, 2, 4, AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_NONE, false, 32, bytes, &len));
   for (unsigned i = 0; i < sizeof(bytes); ++i)
      assert(bytes[i] == 0xa5);

   puts("Native half sqrt: nine exact Apple-derived step/multiply vectors and narrow refusal guards PASS");
}

static void test_native_select_reg_imm(void)
{
   const uint16_t src[4][4]={{0,2,10,1},{0,2,7,1},{0,2,10,1},{4,6,2,1}};
   const char *golden[]={"02011d03860a00c02081","32011d03060700c02081",
      "02011f03860a00c02081","02051f07060200c02081"};
   const bool no_abs[]={false,false};
   const bool imm[]={false,true};
   for(unsigned i=0;i<4;i++) {
      const enum agx3_alu_hint hints[]={AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD,
         (i&1)?AGX3_ALU_HINT_NONE:AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_NONE};
      uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
      assert(agx3_alu_fcmp_select_reg_imm(AGX_ISA_ENCODING_AGX3B,i==1?6:0,src[i],false,no_abs,imm,
         hints,0,i<2,false,32,bytes,&len));
      exact(bytes,len,golden[i]);
   }
   puts("Native retained-register/immediate FSEL: four exact16/32-bit live/dead vectors PASS");
   for(unsigned live=0;live<2;live++) {
      const uint16_t inputs[]={live?2:0,0,1,0};
      const bool immediate[]={true,true};
      const enum agx3_alu_hint hints[]={live?AGX3_ALU_HINT_NONE:AGX3_ALU_HINT_DISCARD,
         AGX3_ALU_HINT_NONE,AGX3_ALU_HINT_NONE,AGX3_ALU_HINT_NONE};
      uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
      assert(agx3_alu_fcmp_select_reg_imm(AGX_ISA_ENCODING_AGX3B,0,inputs,true,no_abs,immediate,
         hints,2,true,false,32,bytes,&len));
      exact(bytes,len,live?"02030581228102c22080":"02010d81228102c22080");
   }
   puts("Native zero-comparison boolean FSEL: two exact live/dead vectors PASS");
   /* Fresh complete Apple kernels select two loaded values, then optionally
    * store both operands again. Cover DISCARD, CACHE and retained NONE hints. */
   const uint16_t registers[4][4]={{4,6,8,9},{4,6,2,3},{4,6,8,10},{6,8,2,4}};
   const char *reg_golden[]={"02051d07820803c08009","0a053d07028203c00083",
      "02051f07820803c0800a","02071f09020203c00004"};
   const bool reg_immediate[]={false,false};
   for(unsigned i=0;i<4;i++) {
      const enum agx3_alu_hint live_hint=i==1?AGX3_ALU_HINT_CACHE:AGX3_ALU_HINT_NONE;
      const enum agx3_alu_hint hints[]={AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD,
         (i&1)?live_hint:AGX3_ALU_HINT_DISCARD,
         (i&1)?live_hint:AGX3_ALU_HINT_DISCARD};
      uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
      assert(agx3_alu_fcmp_select_reg_imm(AGX_ISA_ENCODING_AGX3B,i==1?1:0,registers[i],
         false,no_abs,reg_immediate,hints,1,i<2,i==1,32,bytes,&len));
      exact(bytes,len,reg_golden[i]);
   }
   puts("Native register-register FSEL: four exact 16/32-bit live/dead vectors PASS");
   const uint16_t zero_registers[4][4]={{0,0,8,9},{0,0,2,3},{14,0,4,6},{0,0,2,4}};
   const char *zero_golden[]={"02010d81820803c28009","0a012d81028203c20083",
      "020f0f81820403c28006","02010f81020203c20004"};
   for(unsigned i=0;i<4;i++) {
      const enum agx3_alu_hint live_hint=i==1?AGX3_ALU_HINT_CACHE:AGX3_ALU_HINT_NONE;
      const enum agx3_alu_hint hints[]={AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_NONE,
         (i&1)?live_hint:AGX3_ALU_HINT_DISCARD,
         (i&1)?live_hint:AGX3_ALU_HINT_DISCARD};
      uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
      assert(agx3_alu_fcmp_select_reg_imm(AGX_ISA_ENCODING_AGX3B,i==1?1:0,
         zero_registers[i],true,no_abs,reg_immediate,hints,1,i<2,i==1,32,bytes,&len));
      exact(bytes,len,zero_golden[i]);
   }
   puts("Native zero-compare register FSEL: four exact 16/32-bit live/dead vectors PASS");

   /* Own Metal kernels isolate fabs on each comparator and retain all three
    * selected-operand forms used by update_lightmap. */
   const bool both_abs[]={true,true};
   const uint16_t abs_reg0[]={0,2,4,0};
   const bool reg0_imm[]={false,true};
   const enum agx3_alu_hint no_hints[]={AGX3_ALU_HINT_NONE,AGX3_ALU_HINT_NONE,
      AGX3_ALU_HINT_NONE,AGX3_ALU_HINT_NONE};
   uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
   assert(agx3_alu_fcmp_select_reg_imm(AGX_ISA_ENCODING_AGX3B,6,abs_reg0,
      false,both_abs,reg0_imm,no_hints,0,false,false,32,bytes,&len));
   exact(bytes,len,"3201030307040000208000800300");

   const bool left_abs[]={true,false},right_abs[]={false,true};
   assert(agx3_alu_fcmp_select_reg_imm(AGX_ISA_ENCODING_AGX3B,6,abs_reg0,
      false,left_abs,reg0_imm,no_hints,0,false,false,32,bytes,&len));
   exact(bytes,len,"3201030307040000208000800100");
   assert(agx3_alu_fcmp_select_reg_imm(AGX_ISA_ENCODING_AGX3B,6,abs_reg0,
      false,right_abs,reg0_imm,no_hints,0,false,false,32,bytes,&len));
   exact(bytes,len,"3201030307040000208000800200");

   const uint16_t abs_regreg[]={0,2,4,6};
   const enum agx3_alu_hint regreg_hints[]={AGX3_ALU_HINT_NONE,
      AGX3_ALU_HINT_NONE,AGX3_ALU_HINT_NONE,AGX3_ALU_HINT_DISCARD};
   assert(agx3_alu_fcmp_select_reg_imm(AGX_ISA_ENCODING_AGX3B,6,abs_regreg,
      false,both_abs,reg_immediate,regreg_hints,0,false,false,32,bytes,&len));
   exact(bytes,len,"3201070307040000800600800300");

   const uint16_t abs_zeroreg[]={0,2,0,4};
   const bool zeroreg_imm[]={true,false};
   assert(agx3_alu_fcmp_select_reg_imm(AGX_ISA_ENCODING_AGX3B,6,abs_zeroreg,
      false,both_abs,zeroreg_imm,no_hints,0,false,false,32,bytes,&len));
   exact(bytes,len,"3201030327800000000400800300");

   assert(agx3_alu_fcmp_select_reg_imm(AGX_ISA_ENCODING_AGX3B,6,abs_reg0,
      false,both_abs,reg0_imm,no_hints,1,false,false,32,bytes,&len));
   exact(bytes,len,"3201030303040300208000800300");
   assert(agx3_alu_fcmp_select_reg_imm(AGX_ISA_ENCODING_AGX3B,6,abs_regreg,
      false,both_abs,reg_immediate,regreg_hints,1,false,false,32,bytes,&len));
   exact(bytes,len,"3201070303040300800600800300");
   assert(agx3_alu_fcmp_select_reg_imm(AGX_ISA_ENCODING_AGX3B,6,abs_zeroreg,
      false,both_abs,zeroreg_imm,no_hints,1,false,false,32,bytes,&len));
   exact(bytes,len,"3201030323800300000400800300");
   assert(agx3_alu_fcmp_select_reg_imm(AGX_ISA_ENCODING_AGX3B,6,abs_reg0,
      false,both_abs,reg0_imm,no_hints,2,false,false,32,bytes,&len));
   exact(bytes,len,"3201030303040200208000800300");

   /* Hyprland's colour conversion compares a full value against one
    * absolute full value.  Fresh direct-value Metal kernels cover every
    * ordered condition on the same long endpoint. */
   const uint16_t abs_second_src[]={2,4,6,0};
   const bool abs_second[]={false,true};
   const enum agx3_alu_hint abs_second_hints[]={AGX3_ALU_HINT_NONE,
      AGX3_ALU_HINT_NONE,AGX3_ALU_HINT_NONE,AGX3_ALU_HINT_DISCARD};
   static const struct {
      unsigned condition;
      const char *bytes;
   } abs_ordered[] = {
      {5,"0203070507060200800000800200"},
      {6,"0203070507060300800000800200"},
      {1,"0203070503060300800000800200"},
      {2,"0203070503060200800000800200"},
      {0,"0203070507060000800000800200"},
   };
   for (unsigned i=0;i<sizeof(abs_ordered)/sizeof(abs_ordered[0]);++i) {
      assert(agx3_alu_fcmp_select_reg_imm(
         AGX_ISA_ENCODING_AGX3B,0,abs_second_src,false,abs_second,
         reg_immediate,abs_second_hints,abs_ordered[i].condition,false,false,
         32,bytes,&len));
      exact(bytes,len,abs_ordered[i].bytes);
   }

   const uint16_t abs_first_src[]={4,2,6,0};
   const bool abs_first[]={true,false};
   assert(agx3_alu_fcmp_select_reg_imm(
      AGX_ISA_ENCODING_AGX3B,0,abs_first_src,false,abs_first,reg_immediate,
      abs_second_hints,6,false,false,32,bytes,&len));
   exact(bytes,len,"0205070307060300800000800100");
   assert(!agx3_alu_fcmp_select_reg_imm(
      AGX_ISA_ENCODING_AGX3B,0,abs_second_src,false,both_abs,reg_immediate,
      abs_second_hints,5,false,false,32,bytes,&len));
   puts("Native absolute-comparison FSEL: fifteen exact G17P vectors PASS");
}

static void test_native_half_min(void)
{
   uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
   const uint16_t internal_src[]={2,4,2,4};
   const enum agx3_alu_hint cached[]={AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_CACHE,
      AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_CACHE};
   assert(agx3_alu_fmin_half(AGX_ISA_ENCODING_AGX3B,0,internal_src,cached,
                            true,32,bytes,&len));
   exact(bytes,len,"0282248401c0");
   const uint16_t dead_src[]={2,3,2,3};
   const enum agx3_alu_hint dead[]={AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD};
   assert(agx3_alu_fmin_half(AGX_ISA_ENCODING_AGX3B,2,dead_src,dead,
                            false,32,bytes,&len));
   exact(bytes,len,"12021c0211c0");
   puts("Native half minimum: exact internal-clamp/dead Apple FSEL vectors PASS");
}

static void test_native_half_compare_select(void)
{
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   const uint16_t src[] = {2, 3, 4, 5};
   const enum agx3_alu_hint cached[] = {
      AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_CACHE,
      AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_DISCARD,
   };
   assert(agx3_alu_fcmp_select_half(AGX_ISA_ENCODING_AGX3B, 0, src,
                                    cached, 1, true, 0, bytes, &len));
   exact(bytes, len, "0a822582028413c08005");

   const uint16_t dead_src[] = {0, 1, 2, 3};
   const enum agx3_alu_hint dead[] = {
      AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD,
   };
   assert(agx3_alu_fcmp_select_half(AGX_ISA_ENCODING_AGX3B, 0, dead_src,
                                    dead, 1, false, 0, bytes, &len));
   exact(bytes, len, "02001d00820213c08003");

   /* Reject unwitnessed shapes without touching the caller's storage. */
   for (unsigned bad = 0; bad < 7; ++bad) {
      uint16_t bad_src[] = {0, 1, 2, 3};
      enum agx3_alu_hint bad_hints[] = {
         AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE,
         AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE,
      };
      enum agx_isa_encoding enc = AGX_ISA_ENCODING_AGX3B;
      unsigned dest = 0, cond = 1, wait = 0;
      if (bad == 0) enc = AGX_ISA_ENCODING_AGX3C;
      if (bad == 1) dest = 256;
      if (bad == 2) bad_src[0] = 256;
      if (bad == 3) bad_src[3] = 256;
      if (bad == 4) bad_hints[2] = 3;
      if (bad == 5) cond = 4;
      if (bad == 6) wait = 1;
      memset(bytes, 0xa5, sizeof(bytes));
      len = 0xa5;
      assert(!agx3_alu_fcmp_select_half(enc, dest, bad_src, bad_hints,
                                        cond, false, wait, bytes, &len));
      assert(len == 0xa5);
      for (unsigned i = 0; i < sizeof(bytes); ++i)
         assert(bytes[i] == 0xa5);
   }
   puts("Native scalar half comparison/select: two exact Apple vectors and fail-closed cases PASS");
}

static void test_native_half_max(void)
{
   uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
   const uint16_t src[]={2,3,2,3};
   const enum agx3_alu_hint cached[]={AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_CACHE,
      AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_CACHE};
   assert(agx3_alu_fmax_half(AGX_ISA_ENCODING_AGX3B,0,src,cached,
                            true,32,bytes,&len));
   exact(bytes,len,"0282248210c0");
   const enum agx3_alu_hint dead[]={AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD};
   assert(agx3_alu_fmax_half(AGX_ISA_ENCODING_AGX3B,2,src,dead,
                            false,32,bytes,&len));
   exact(bytes,len,"12021c0210c0");
   assert(!agx3_alu_fmax_half(AGX_ISA_ENCODING_AGX3C,0,src,cached,
                             true,32,bytes,&len));
   assert(!agx3_alu_fmax_half(AGX_ISA_ENCODING_AGX3B,256,src,cached,
                             true,32,bytes,&len));
   puts("Native half maximum: exact internal-clamp/dead Apple FSEL vectors and fail-closed bounds PASS");
}

/*
 * Apple's own applegpu_g17p bytes for the 32-bit tied minimum/maximum, from
 * `metal-nt`.  These are the shapes NIR produces for normalize(), clamp(),
 * saturate() and discard, none of which had an encoding before.
 */
static void test_native_f32_minmax(void)
{
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;

   /* o[i] = fmax(a[i], b[i]); loaded operands dead, waiting the load's slot. */
   const uint16_t rr[] = {2, 4, 2, 4};
   const enum agx3_alu_hint dead[] = {
      AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD,
   };
   assert(agx3_alu_fmax_f32(AGX_ISA_ENCODING_AGX3B, 2, rr, dead, false, 32,
                            bytes, &len));
   exact(bytes, len, "12031e0500c0");
   assert(agx3_alu_fmin_f32(AGX_ISA_ENCODING_AGX3B, 2, rr, dead, false, 32,
                            bytes, &len));
   exact(bytes, len, "12031e0501c0");

   /* The same maximum with the second operand carried as a minifloat.  The
    * immediate byte is a plain agx_minifloat_encode() of the constant. */
   static const struct {
      uint8_t minifloat;
      const char *bytes;
   } imm[] = {
      {0x00, "12030a8100c2"},   /*  0.0f      */
      {0x10, "12030a9100c2"},   /*  0.25f     */
      {0x20, "12030aa100c2"},   /*  0.5f      */
      {0x28, "12030aa900c2"},   /*  0.75f     */
      {0x30, "12030ab100c2"},   /*  1.0f      */
      {0x34, "12030ab500c2"},   /*  1.25f     */
      {0x38, "12030ab900c2"},   /*  1.5f      */
      {0x40, "12030ac100c2"},   /*  2.0f      */
      {0x48, "12030ac900c2"},   /*  3.0f      */
      {0x50, "12030ad100c2"},   /*  4.0f      */
      {0x60, "12030ae100c2"},   /*  8.0f      */
      {0x70, "12030af100c2"},   /* 16.0f      */
      {0x08, "12030a8900c2"},   /*  0.125f    */
      {0x01, "12030a8110c2"},   /*  0.015625f */
      {0x7f, "12030aff10c2"},   /* 31.0f      */
      {0x3f, "12030abf10c2"},   /*  1.9375f   */
      {0xa0, "12031aa100c2"},   /* -0.5f      */
      {0xb0, "12031ab100c2"},   /* -1.0f      */
   };
   for (unsigned i = 0; i < sizeof(imm) / sizeof(imm[0]); ++i) {
      assert(agx3_alu_fmax_f32_imm(AGX_ISA_ENCODING_AGX3B, 2, 2,
                                   imm[i].minifloat, AGX3_ALU_HINT_DISCARD,
                                   false, 32, bytes, &len));
      exact(bytes, len, imm[i].bytes);
   }
   assert(agx3_alu_fmin_f32_imm(AGX_ISA_ENCODING_AGX3B, 2, 2, 0x00,
                                AGX3_ALU_HINT_DISCARD, false, 32, bytes, &len));
   exact(bytes, len, "12030a8101c2");
   assert(agx3_alu_fmin_f32_imm(AGX_ISA_ENCODING_AGX3B, 2, 2, 0x30,
                                AGX3_ALU_HINT_DISCARD, false, 32, bytes, &len));
   exact(bytes, len, "12030ab101c2");

   /* Eight simultaneously-live maxima against zero: Apple's own register and
    * lifetime assignment, from one kernel. */
   static const struct {
      uint16_t dest, src;
      enum agx3_alu_hint hint;
      bool dest_cache;
      uint16_t wait;
      const char *bytes;
   } live[] = {
      { 8, 16, AGX3_ALU_HINT_CACHE,   true, 32, "4291228100c2"},
      {12, 18, AGX3_ALU_HINT_CACHE,   true,  0, "629322810002"},
      {12, 20, AGX3_ALU_HINT_CACHE,   true,  0, "629522810002"},
      {14, 22, AGX3_ALU_HINT_CACHE,   true,  0, "729722810002"},
      { 0,  0, AGX3_ALU_HINT_DISCARD, true,  0, "02012a810002"},
      {28,  2, AGX3_ALU_HINT_DISCARD, true,  0, "e2032a810002"},
      { 4,  4, AGX3_ALU_HINT_DISCARD, true,  0, "22052a810002"},
      { 6,  6, AGX3_ALU_HINT_DISCARD, true,  0, "32072a810002"},
   };
   for (unsigned i = 0; i < sizeof(live) / sizeof(live[0]); ++i) {
      assert(agx3_alu_fmax_f32_imm(AGX_ISA_ENCODING_AGX3B, live[i].dest,
                                   live[i].src, 0x00, live[i].hint,
                                   live[i].dest_cache, live[i].wait, bytes,
                                   &len));
      exact(bytes, len, live[i].bytes);
   }

   /* A maximum whose operand came from a slot-1 load, also Apple's. */
   assert(agx3_alu_fmax_f32(AGX_ISA_ENCODING_AGX3B, 2, rr, dead, false, 1,
                            bytes, &len));
   exact(bytes, len, "12031e050020");

   /* Fail closed: wrong encoding, out-of-range registers, a lifetime hint on
    * an immediate, and a wait spanning two slots. */
   const enum agx3_alu_hint none[] = {
      AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE,
      AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE,
   };
   /* M5 uses a different four-operand fseli32 field layout. Zero can be
    * selected directly; nonzero floating immediates remain rejected. */
   assert(agx3_alu_fmax_f32_imm(AGX_ISA_ENCODING_AGX3C,2,2,0,
                                AGX3_ALU_HINT_DISCARD,false,0,bytes,&len));
   exact(bytes,len,"3202070a2300a00214808a4c0200");
   const uint16_t m5_rr[4] = {0,2,0,2};
   assert(agx3_alu_fmin_f32(AGX_ISA_ENCODING_AGX3C,2,m5_rr,none,false,129,bytes,&len));
   exact(bytes,len,"3204470b2300a002900001400040");
   assert(!agx3_alu_fmax_f32_imm(AGX_ISA_ENCODING_AGX3C,2,2,0x30,
                                 AGX3_ALU_HINT_NONE,false,0,bytes,&len));
   assert(!agx3_alu_fmax_f32(AGX_ISA_ENCODING_AGX2, 2, rr, none, false, 0,
                             bytes, &len));
   assert(!agx3_alu_fmax_f32(AGX_ISA_ENCODING_AGX3B, 256, rr, none, false, 0,
                             bytes, &len));
   assert(!agx3_alu_fmax_f32(AGX_ISA_ENCODING_AGX3B, 2, rr, none, false, 33,
                             bytes, &len));
   assert(!agx3_alu_fmax_f32_imm(AGX_ISA_ENCODING_AGX3B, 2, 2, 0x00,
                                 AGX3_ALU_HINT_NONE, false, 3, bytes, &len));
   puts("Native f32 minimum/maximum: exact Apple register, minifloat and "
        "lifetime vectors plus fail-closed bounds PASS");
}

/*
 * A 32-bit float compared against a minifloat immediate, selecting two 32-bit
 * registers -- what an ordinary `x < k ? a : b` becomes.
 *
 * Apple's own bytes for `(a[i] OP 0.5f) ? c[i] : d[i]`, from
 * `metal-nt -arch applegpu_g17p`.  Six kernels differing only in the operator;
 * only the condition changes, and `!=` is `==` with the selected operands
 * exchanged:
 *
 *   <    12 03 0f a1 22 06 03 02 20 0a
 *   >    12 03 0f a1 22 06 02 02 20 0a
 *   >=   12 03 0f a1 26 06 02 02 20 0a
 *   <=   12 03 0f a1 26 06 03 02 20 0a
 *   ==   12 03 0f a1 26 06 00 02 20 0a
 *   !=   12 03 0f a1 26 08 00 02 20 06
 *
 * Those witnesses route their selected operands through Apple's kind-3 slots;
 * the encoder emits ordinary registers, which changes only the operand bits.
 * The condition lives in byte 4 bit 2 and byte 6, and the test pins those
 * against the witnesses directly as well as pinning the whole instruction.
 */
static void test_native_f32_minifloat_select(void)
{
   static const struct {
      unsigned condition;
      const char *bytes;
      uint8_t apple_byte4, apple_byte6;
   } cases[] = {
      {0 /* eq */, "12030fa106060002000a", 0x26, 0x00},
      {1 /* lt */, "12030fa102060302000a", 0x22, 0x03},
      {2 /* gt */, "12030fa102060202000a", 0x22, 0x02},
      {5 /* ge */, "12030fa106060202000a", 0x26, 0x02},
      {6 /* le */, "12030fa106060302000a", 0x26, 0x03},
   };
   const enum agx3_alu_hint hints[4] = {
      AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_NONE,
      AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE,
   };

   for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      assert(agx3_alu_fcmp_select_f32_imm(AGX_ISA_ENCODING_AGX3B, 2, 2, 0x20,
                                          false, 6, 10, hints, cases[i].condition,
                                          false, 0, bytes, &len));
      exact(bytes, len, cases[i].bytes);

      /* The condition bits are Apple's, not ours. */
      assert((bytes[4] & 0x04) == (cases[i].apple_byte4 & 0x04));
      assert(bytes[6] == cases[i].apple_byte6);
   }

   /* Fresh current-toolchain applegpu_g17p kernels for `0.5 OP fabs(x)`.
    * Apple reverses the operands, places fabs(x) in source0, and uses the
    * long endpoint whose field47 is the absolute modifier. */
   static const struct {
      unsigned condition;
      const char *bytes;
   } absolute[] = {
      {6 /* le */, "020307a107040302000600800100"},
      {5 /* ge */, "020307a107040202000600800100"},
      {2 /* gt */, "020307a103040202000600800100"},
      {1 /* lt */, "020307a103040302000600800100"},
      {0 /* eq */, "020307a107040002000600800100"},
   };
   const enum agx3_alu_hint absolute_hints[4] = {
      AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE,
      AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE,
   };
   for (unsigned i = 0; i < sizeof(absolute) / sizeof(absolute[0]); ++i) {
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      assert(agx3_alu_fcmp_select_f32_imm(
         AGX_ISA_ENCODING_AGX3B, 0, 2, 0x20, true, 4, 6,
         absolute_hints, absolute[i].condition, false, 32, bytes, &len));
      exact(bytes, len, absolute[i].bytes);
   }

   /* Apple spells != as EQ with the selected values exchanged here too. */
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   assert(agx3_alu_fcmp_select_f32_imm(
      AGX_ISA_ENCODING_AGX3B, 0, 2, 0x20, true, 6, 4,
      absolute_hints, 0, false, 32, bytes, &len));
   exact(bytes, len, "020307a107060002000400800100");

   /* The minifloat spans the same field as every other fsel immediate. */
   assert(agx3_alu_fcmp_select_f32_imm(AGX_ISA_ENCODING_AGX3B, 2, 2, 0x00, false, 6,
                                       10, hints, 1, false, 0, bytes, &len));
   exact(bytes, len, "12030f8102060302000a");
   assert(agx3_alu_fcmp_select_f32_imm(AGX_ISA_ENCODING_AGX3B, 2, 2, 0xff, false, 6,
                                       10, hints, 1, false, 0, bytes, &len));
   exact(bytes, len, "12031fff02061302000a");

   /* Waits, lifetimes and the cached destination stay orthogonal. */
   assert(agx3_alu_fcmp_select_f32_imm(AGX_ISA_ENCODING_AGX3B, 2, 2, 0x20, false, 6,
                                       10, hints, 2, true, 32, bytes, &len));
   exact(bytes, len, "12032fa1020602c2000a");

   /* Fail closed: the wrong encoding, an unwitnessed condition, a wait
    * spanning two slots, an odd register, and a lifetime on the immediate. */
   assert(!agx3_alu_fcmp_select_f32_imm(AGX_ISA_ENCODING_AGX3C, 2, 2, 0x20, false, 6,
                                        10, hints, 1, false, 0, bytes, &len));
   assert(!agx3_alu_fcmp_select_f32_imm(AGX_ISA_ENCODING_AGX3B, 2, 2, 0x20, false, 6,
                                        10, hints, 3, false, 0, bytes, &len));
   assert(!agx3_alu_fcmp_select_f32_imm(AGX_ISA_ENCODING_AGX3B, 2, 2, 0x20, false, 6,
                                        10, hints, 7, false, 0, bytes, &len));
   assert(!agx3_alu_fcmp_select_f32_imm(AGX_ISA_ENCODING_AGX3B, 2, 2, 0x20, false, 6,
                                        10, hints, 1, false, 3, bytes, &len));
   assert(!agx3_alu_fcmp_select_f32_imm(AGX_ISA_ENCODING_AGX3B, 3, 2, 0x20, false, 6,
                                        10, hints, 1, false, 0, bytes, &len));
   const enum agx3_alu_hint bad[4] = {
      AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE,
   };
   assert(!agx3_alu_fcmp_select_f32_imm(AGX_ISA_ENCODING_AGX3B, 2, 2, 0x20, false, 6,
                                        10, bad, 1, false, 0, bytes, &len));
   assert(!agx3_alu_fcmp_select_f32_imm(
      AGX_ISA_ENCODING_AGX3B, 2, 2, 0x20, true, 6, 10,
      absolute_hints, 1, false, 1, bytes, &len));
   assert(agx3_alu_fcmp_select_f32_imm(
      AGX_ISA_ENCODING_AGX3B, 254, 252, 0xff, true, 250, 248,
      absolute_hints, 1, true, 0, bytes, &len));
   assert(!agx3_alu_fcmp_select_f32_imm(
      AGX_ISA_ENCODING_AGX3B, 254, 253, 0xff, true, 250, 248,
      absolute_hints, 1, true, 0, bytes, &len));
   const enum agx3_alu_hint gtk_hints[4] = {
      AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_NONE,
      AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE,
   };
   assert(agx3_alu_fcmp_select_f32_imm(
      AGX_ISA_ENCODING_AGX3B, 20, 12, 0x20, true, 30, 20,
      gtk_hints, 6, false, 0, bytes, &len));
   exact(bytes, len, "a28d07a1071e0302001400000100");
   puts("Native f32 minifloat select: Apple's plain and absolute conditions, "
        "immediate span, wait/lifetime orthogonality and fail-closed PASS");
}

static void test_native_half_clamp(void)
{
   uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
   const uint16_t max_src[]={2,3,2,3};
   const enum agx3_alu_hint max_hints[]={AGX3_ALU_HINT_CACHE,
      AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_CACHE};
   assert(agx3_alu_fmax_half(AGX_ISA_ENCODING_AGX3B,0,max_src,max_hints,
                            true,32,bytes,&len));
   exact(bytes,len,"0282248210c0");

   const uint16_t min_src[]={0,4,0,4};
   const enum agx3_alu_hint min_hints[]={AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_CACHE};
   assert(agx3_alu_fmin_half(AGX_ISA_ENCODING_AGX3B,0,min_src,min_hints,
                            false,0,bytes,&len));
   exact(bytes,len,"02000c840100");
   puts("Native half clamp: exact internal Apple FSEL plus canonical no-wait tail PASS");
}

static void
test_native_reciprocal_widths(void)
{
   static const struct {
      uint16_t dest, src, wait;
      bool dest_half, src_half, src_abs, src_neg, dest_cache;
      enum agx3_alu_hint hint;
      const char *bytes;
   } cases[] = {
      /* The four independent destination/source width combinations. */
      {2, 2, 32, true, true, false, false, false,
       AGX3_ALU_HINT_DISCARD, "af005602020410463000"},
      {0, 0, 32, false, true, false, false, false,
       AGX3_ALU_HINT_DISCARD, "af005600020010442000"},
      {2, 2, 32, true, false, false, false, false,
       AGX3_ALU_HINT_DISCARD, "af0056020204104a3000"},
      {0, 0, 32, false, false, false, false, false,
       AGX3_ALU_HINT_DISCARD, "af005600020010482000"},

      /* Source arithmetic modifiers on GTK's float-from-half form. */
      {0, 0, 32, false, true, true, false, false,
       AGX3_ALU_HINT_DISCARD, "af005600020010446000"},
      {0, 0, 32, false, true, false, true, false,
       AGX3_ALU_HINT_DISCARD, "af00560002001044a000"},
      {0, 0, 32, false, true, true, true, false,
       AGX3_ALU_HINT_DISCARD, "af00560002001044e000"},

      /* Source/result lifetime fields are independent. */
      {0, 2, 32, false, true, false, false, false,
       AGX3_ALU_HINT_CACHE, "af005600020402442000"},
      {0, 0, 32, false, true, false, false, true,
       AGX3_ALU_HINT_DISCARD, "af005600030010442000"},
      {0, 2, 32, false, true, false, false, true,
       AGX3_ALU_HINT_CACHE, "af005600030402442000"},

      /* Odd half input, no dependency, and texture producer slot zero. */
      {0, 3, 32, false, true, false, false, false,
       AGX3_ALU_HINT_DISCARD, "af005600020610442000"},
      {0, 0, 0, false, true, false, false, false,
       AGX3_ALU_HINT_DISCARD, "af005400020010442000"},
      {0, 0, 1, false, true, false, false, false,
       AGX3_ALU_HINT_DISCARD, "af105400020010442000"},
   };

   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
      assert(agx3_alu_rcp(
         AGX_ISA_ENCODING_AGX3B, cases[i].dest, cases[i].src,
         cases[i].dest_half, cases[i].src_half, cases[i].src_abs,
         cases[i].src_neg, cases[i].hint, cases[i].dest_cache,
         cases[i].wait, bytes, &len));
      exact(bytes, len, cases[i].bytes);
   }

   unsigned forms = 0;
   for (unsigned dest_half = 0; dest_half < 2; ++dest_half) {
      for (unsigned src_half = 0; src_half < 2; ++src_half) {
         for (unsigned modifiers = 0; modifiers < 4; ++modifiers) {
            for (unsigned hint = 0; hint <= AGX3_ALU_HINT_CACHE; ++hint) {
               for (unsigned dest_cache = 0; dest_cache < 2; ++dest_cache) {
                  for (unsigned wait = 0; wait < 64; ++wait) {
                     const bool ok = agx3_alu_rcp(
                        AGX_ISA_ENCODING_AGX3B, dest_half ? 127 : 126,
                        src_half ? 127 : 126, dest_half, src_half,
                        modifiers & 1, modifiers & 2,
                        (enum agx3_alu_hint)hint, dest_cache, wait, bytes,
                        &len);
                     if (!ok)
                        fprintf(stderr,
                                "RCP compose failed: dh=%u sh=%u mod=%u "
                                "hint=%u dc=%u wait=%u\n",
                                dest_half, src_half, modifiers, hint,
                                dest_cache, wait);
                     assert(ok);
                     assert(len == 10);
                     ++forms;
                  }
               }
            }
         }
      }
   }
   assert(forms == 6144);

   memset(bytes, 0xa5, sizeof(bytes));
   len = 0xa5;
   assert(!agx3_alu_rcp(
      AGX_ISA_ENCODING_AGX3C, 0, 0, false, true, false, false,
      AGX3_ALU_HINT_NONE, false, 0, bytes, &len));
   assert(!agx3_alu_rcp(
      AGX_ISA_ENCODING_AGX3B, 128, 0, true, true, false, false,
      AGX3_ALU_HINT_NONE, false, 0, bytes, &len));
   assert(!agx3_alu_rcp(
      AGX_ISA_ENCODING_AGX3B, 0, 128, true, true, false, false,
      AGX3_ALU_HINT_NONE, false, 0, bytes, &len));
   assert(!agx3_alu_rcp(
      AGX_ISA_ENCODING_AGX3B, 1, 0, false, true, false, false,
      AGX3_ALU_HINT_NONE, false, 0, bytes, &len));
   assert(!agx3_alu_rcp(
      AGX_ISA_ENCODING_AGX3B, 0, 1, true, false, false, false,
      AGX3_ALU_HINT_NONE, false, 0, bytes, &len));
   assert(!agx3_alu_rcp(
      AGX_ISA_ENCODING_AGX3B, 0, 0, true, true, false, false,
      (enum agx3_alu_hint)3, false, 0, bytes, &len));
   assert(!agx3_alu_rcp(
      AGX_ISA_ENCODING_AGX3B, 0, 0, true, true, false, false,
      AGX3_ALU_HINT_NONE, false, 64, bytes, &len));
   assert(!agx3_alu_rcp(
      AGX_ISA_ENCODING_AGX3B, 0, 0, true, true, false, false,
      AGX3_ALU_HINT_NONE, false, 0, NULL, &len));
   assert(!agx3_alu_rcp(
      AGX_ISA_ENCODING_AGX3B, 0, 0, true, true, false, false,
      AGX3_ALU_HINT_NONE, false, 0, bytes, NULL));
   assert(len == 0xa5);
   for (unsigned i = 0; i < sizeof(bytes); ++i)
      assert(bytes[i] == 0xa5);

   puts("Native reciprocal widths/modifiers: thirteen exact Apple vectors, "
        "6144 field combinations and strict bounds PASS");
}

static void test_native_half_rsqrt(void)
{
   static const struct {
      uint16_t dest, src;
      enum agx3_alu_hint hint;
      bool dest_cache;
      uint16_t wait;
      const char *bytes;
   } cases[] = {
      {3,2,AGX3_ALU_HINT_NONE,    false,32,"af01560302048c600000"},
      {2,2,AGX3_ALU_HINT_DISCARD, false,32,"af0156020204ac600000"},
      {2,2,AGX3_ALU_HINT_DISCARD, false, 0,"af0154020204ac600000"},
      {3,3,AGX3_ALU_HINT_DISCARD, false, 0,"af0154030206ac600000"},
      {2,2,AGX3_ALU_HINT_DISCARD, true, 32,"af0156020304ac600000"},
   };
   for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      assert(agx3_alu_hrsqrt(AGX_ISA_ENCODING_AGX3B, cases[i].dest,
                             cases[i].src, cases[i].hint,
                             cases[i].dest_cache, cases[i].wait,
                             bytes, &len));
      exact(bytes, len, cases[i].bytes);
   }

   for (unsigned bad = 0; bad < 5; ++bad) {
      enum agx_isa_encoding enc = AGX_ISA_ENCODING_AGX3B;
      uint16_t dest = 2, src = 2, wait = 0;
      enum agx3_alu_hint hint = AGX3_ALU_HINT_DISCARD;
      if (bad == 0) enc = AGX_ISA_ENCODING_AGX3C;
      if (bad == 1) dest = 256;
      if (bad == 2) src = 256;
      if (bad == 3) hint = AGX3_ALU_HINT_CACHE;
      if (bad == 4) wait = 1;
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0xa5;
      memset(bytes, 0xa5, sizeof(bytes));
      assert(!agx3_alu_hrsqrt(enc, dest, src, hint, false, wait,
                              bytes, &len));
      assert(len == 0xa5);
      for (unsigned j = 0; j < sizeof(bytes); ++j)
         assert(bytes[j] == 0xa5);
   }
   puts("Native half reciprocal sqrt: five exact Apple vectors and narrow refusal guards PASS");
}

static void test_native_half_exp2(void)
{
   static const struct {
      uint16_t dest, src;
      enum agx3_alu_hint hint;
      bool dest_cache;
      uint16_t wait;
      const char *bytes;
      bool src_neg;
   } cases[] = {
      {3,2,AGX3_ALU_HINT_NONE,    false,32,"af02560302048c600000"},
      {2,2,AGX3_ALU_HINT_DISCARD, false,32,"af0256020204ac600000"},
      {2,2,AGX3_ALU_HINT_DISCARD, false, 0,"af0254020204ac600000"},
      {3,3,AGX3_ALU_HINT_DISCARD, false, 0,"af0254030206ac600000"},
      {2,2,AGX3_ALU_HINT_DISCARD, true, 32,"af0256020304ac600000"},
      {2,2,AGX3_ALU_HINT_DISCARD, false,32,"af0256020204ac600100", true},
   };
   for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      assert(agx3_alu_hexp2_neg(AGX_ISA_ENCODING_AGX3B, cases[i].dest,
                                cases[i].src, cases[i].hint,
                                cases[i].dest_cache, cases[i].wait,
                                cases[i].src_neg, bytes, &len));
      exact(bytes, len, cases[i].bytes);
   }

   for (unsigned bad = 0; bad < 5; ++bad) {
      enum agx_isa_encoding enc = AGX_ISA_ENCODING_AGX3B;
      uint16_t dest = 2, src = 2, wait = 0;
      enum agx3_alu_hint hint = AGX3_ALU_HINT_DISCARD;
      if (bad == 0) enc = AGX_ISA_ENCODING_AGX3C;
      if (bad == 1) dest = 256;
      if (bad == 2) src = 256;
      if (bad == 3) hint = AGX3_ALU_HINT_CACHE;
      if (bad == 4) wait = 1;
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0xa5;
      memset(bytes, 0xa5, sizeof(bytes));
      assert(!agx3_alu_hexp2(enc, dest, src, hint, false, wait,
                             bytes, &len));
      assert(len == 0xa5);
      for (unsigned j = 0; j < sizeof(bytes); ++j)
         assert(bytes[j] == 0xa5);
   }
   puts("Native half exp2: six exact Apple vectors and narrow refusal guards PASS");
}

static void test_native_half_log2(void)
{
   static const struct {
      uint16_t dest, src;
      enum agx3_alu_hint hint;
      bool dest_cache;
      uint16_t wait;
      const char *bytes;
   } cases[] = {
      {3,2,AGX3_ALU_HINT_NONE,    false,32,"2f02560302048c600000"},
      {2,2,AGX3_ALU_HINT_DISCARD, false,32,"2f0256020204ac600000"},
      {2,2,AGX3_ALU_HINT_DISCARD, false, 0,"2f0254020204ac600000"},
      {3,3,AGX3_ALU_HINT_DISCARD, false, 0,"2f0254030206ac600000"},
      {2,2,AGX3_ALU_HINT_DISCARD, true, 32,"2f0256020304ac600000"},
   };
   for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      assert(agx3_alu_hlog2(AGX_ISA_ENCODING_AGX3B, cases[i].dest,
                            cases[i].src, cases[i].hint,
                            cases[i].dest_cache, cases[i].wait,
                            bytes, &len));
      exact(bytes, len, cases[i].bytes);
   }

   for (unsigned bad = 0; bad < 5; ++bad) {
      enum agx_isa_encoding enc = AGX_ISA_ENCODING_AGX3B;
      uint16_t dest = 2, src = 2, wait = 0;
      enum agx3_alu_hint hint = AGX3_ALU_HINT_DISCARD;
      if (bad == 0) enc = AGX_ISA_ENCODING_AGX3C;
      if (bad == 1) dest = 256;
      if (bad == 2) src = 256;
      if (bad == 3) hint = (enum agx3_alu_hint)3;
      if (bad == 4) wait = 1;
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0xa5;
      memset(bytes, 0xa5, sizeof(bytes));
      assert(!agx3_alu_hlog2(enc, dest, src, hint, false, wait,
                             bytes, &len));
      assert(len == 0xa5);
      for (unsigned j = 0; j < sizeof(bytes); ++j)
         assert(bytes[j] == 0xa5);
   }
   puts("Native half log2: five exact Apple vectors and narrow refusal guards PASS");
}

static void test_native_mixed_width_log2(void)
{
   /* Fresh current-Xcode applegpu_g17p kernels. f10 and f26 independently
    * select the destination and source width. The matrix also pins an odd
    * half source, both lifetimes, result caching, modifiers, and wait/no-wait
    * without relying on a synthetic field composition. */
   static const struct {
      uint16_t dest, src;
      bool dest_half, src_half, src_abs, src_neg;
      enum agx3_alu_hint hint;
      bool dest_cache;
      uint16_t wait;
      const char *bytes;
   } cases[] = {
      {0,0,true, true, false,false,AGX3_ALU_HINT_DISCARD,true, 32,
       "2f0256000300ac600000"},
      {0,2,true, true, false,false,AGX3_ALU_HINT_CACHE,  true, 32,
       "2f02560003048e600000"},
      {0,0,false,true, false,false,AGX3_ALU_HINT_DISCARD,false,32,
       "2f0256000200a8400000"},
      {0,2,false,true, false,false,AGX3_ALU_HINT_CACHE,  false,32,
       "2f02560002048a400000"},
      {0,3,false,true, false,false,AGX3_ALU_HINT_DISCARD,false,32,
       "2f0256000206a8400000"},
      {0,0,false,true, false,false,AGX3_ALU_HINT_DISCARD,false,0,
       "2f0254000200a8400000"},
      {0,0,false,true, true, false,AGX3_ALU_HINT_DISCARD,false,32,
       "2f0256000200a8c00000"},
      {0,0,false,true, false,true, AGX3_ALU_HINT_DISCARD,false,32,
       "2f0256000200a8400100"},
      {0,0,false,true, true, true, AGX3_ALU_HINT_DISCARD,false,32,
       "2f0256000200a8c00100"},
      {0,0,false,true, false,false,AGX3_ALU_HINT_DISCARD,true, 32,
       "2f0256000300a8400000"},
      {2,2,true, false,false,false,AGX3_ALU_HINT_DISCARD,false,32,
       "2f0256020204b4600000"},
      {2,4,true, false,false,false,AGX3_ALU_HINT_CACHE,  false,32,
       "2f025602020896600000"},
      {2,2,true, false,true, false,AGX3_ALU_HINT_DISCARD,false,32,
       "2f0256020204b4e00000"},
      {2,2,true, false,false,true, AGX3_ALU_HINT_DISCARD,false,32,
       "2f0256020204b4600100"},
      {2,2,true, false,true, true, AGX3_ALU_HINT_DISCARD,false,32,
       "2f0256020204b4e00100"},
   };

   for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      assert(agx3_alu_log2(
         AGX_ISA_ENCODING_AGX3B, cases[i].dest, cases[i].src,
         cases[i].dest_half, cases[i].src_half, cases[i].src_abs,
         cases[i].src_neg, cases[i].hint, cases[i].dest_cache,
         cases[i].wait, bytes, &len));
      exact(bytes, len, cases[i].bytes);
   }

   /* The recovered operand grammar names halves 0..191. A full scalar
    * consumes an aligned pair; a half may independently select the final odd
    * half. */
   for (unsigned dest_half = 0; dest_half < 2; ++dest_half) {
      for (unsigned src_half = 0; src_half < 2; ++src_half) {
         uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
         assert(agx3_alu_log2(
            AGX_ISA_ENCODING_AGX3B, dest_half ? 191 : 190,
            src_half ? 191 : 190, dest_half, src_half, false, false,
            AGX3_ALU_HINT_NONE, false, 0,
            bytes, &len));
         assert(len == 10);
      }
   }

   struct {
      enum agx_isa_encoding enc;
      uint16_t dest, src;
      bool dest_half, src_half;
      enum agx3_alu_hint hint;
      uint16_t wait;
   } bad[] = {
      {AGX_ISA_ENCODING_AGX3C, 0,   0,   false,true, AGX3_ALU_HINT_NONE,0},
      {AGX_ISA_ENCODING_AGX3B, 256, 0,   true, true, AGX3_ALU_HINT_NONE,0},
      {AGX_ISA_ENCODING_AGX3B, 0,   256, false,true, AGX3_ALU_HINT_NONE,0},
      {AGX_ISA_ENCODING_AGX3B, 192, 0,   true, true, AGX3_ALU_HINT_NONE,0},
      {AGX_ISA_ENCODING_AGX3B, 0,   192, false,true, AGX3_ALU_HINT_NONE,0},
      {AGX_ISA_ENCODING_AGX3B, 255, 0,   false,true, AGX3_ALU_HINT_NONE,0},
      {AGX_ISA_ENCODING_AGX3B, 0,   255, false,false,AGX3_ALU_HINT_NONE,0},
      {AGX_ISA_ENCODING_AGX3B, 0,   0,   false,true, AGX3_ALU_HINT_NONE,1},
      {AGX_ISA_ENCODING_AGX3B, 0,   0,   false,true,
       (enum agx3_alu_hint)3, 0},
      {AGX_ISA_ENCODING_AGX3B, 0,   0,   false,false,AGX3_ALU_HINT_NONE,64},
   };
   for (unsigned i = 0; i < sizeof(bad) / sizeof(bad[0]); ++i) {
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0xa5;
      memset(bytes, 0xa5, sizeof(bytes));
      assert(!agx3_alu_log2(
         bad[i].enc, bad[i].dest, bad[i].src, bad[i].dest_half,
         bad[i].src_half, false, false, bad[i].hint, false, bad[i].wait,
         bytes, &len));
      assert(len == 0xa5);
      for (unsigned j = 0; j < sizeof(bytes); ++j)
         assert(bytes[j] == 0xa5);
   }

   puts("Native mixed-width log2: fifteen exact Apple vectors and strict width/register guards PASS");
}

static void test_native_half_floor(void)
{
   static const struct { uint16_t d,s; enum agx3_alu_hint h; bool dc;
      uint16_t w; const char *b; } cases[] = {
      {3,2,AGX3_ALU_HINT_NONE,false,32,"2f00560302048c600200"},
      {2,2,AGX3_ALU_HINT_DISCARD,false,32,"2f0056020204ac600200"},
      {2,2,AGX3_ALU_HINT_DISCARD,false,0,"2f0054020204ac600200"},
      {3,3,AGX3_ALU_HINT_DISCARD,false,0,"2f0054030206ac600200"},
      {2,2,AGX3_ALU_HINT_DISCARD,true,32,"2f0056020304ac600200"},
   };
   for (unsigned i=0;i<sizeof(cases)/sizeof(cases[0]);++i) {
      uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
      assert(agx3_alu_hfloor(AGX_ISA_ENCODING_AGX3B,cases[i].d,cases[i].s,
                             cases[i].h,cases[i].dc,cases[i].w,bytes,&len));
      exact(bytes,len,cases[i].b);
   }
   uint8_t bytes[AGX3_ALU_MAX_LENGTH],len=0xa5;
   memset(bytes,0xa5,sizeof(bytes));
   assert(!agx3_alu_hfloor(AGX_ISA_ENCODING_AGX3C,2,2,
                           AGX3_ALU_HINT_DISCARD,false,0,bytes,&len));
   assert(!agx3_alu_hfloor(AGX_ISA_ENCODING_AGX3B,2,2,
                           AGX3_ALU_HINT_CACHE,false,0,bytes,&len));
   assert(!agx3_alu_hfloor(AGX_ISA_ENCODING_AGX3B,2,2,
                           AGX3_ALU_HINT_DISCARD,false,1,bytes,&len));
   assert(len==0xa5);
   for(unsigned i=0;i<sizeof(bytes);++i) assert(bytes[i]==0xa5);
   puts("Native half floor: five exact Apple vectors and refusal guards PASS");
}

static void test_native_half_ceil(void)
{
   static const struct { uint16_t d,s; enum agx3_alu_hint h; bool dc;
      uint16_t w; const char *b; } cases[] = {
      {3,2,AGX3_ALU_HINT_NONE,false,32,"2f00560302048c600400"},
      {2,2,AGX3_ALU_HINT_DISCARD,false,32,"2f0056020204ac600400"},
      {2,2,AGX3_ALU_HINT_DISCARD,false,0,"2f0054020204ac600400"},
      {3,3,AGX3_ALU_HINT_DISCARD,false,0,"2f0054030206ac600400"},
      {2,2,AGX3_ALU_HINT_DISCARD,true,32,"2f0056020304ac600400"},
   };
   for (unsigned i=0;i<sizeof(cases)/sizeof(cases[0]);++i) {
      uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
      assert(agx3_alu_hceil(AGX_ISA_ENCODING_AGX3B,cases[i].d,cases[i].s,
                            cases[i].h,cases[i].dc,cases[i].w,bytes,&len));
      exact(bytes,len,cases[i].b);
   }
   uint8_t bytes[AGX3_ALU_MAX_LENGTH],len=0xa5;
   memset(bytes,0xa5,sizeof(bytes));
   assert(!agx3_alu_hceil(AGX_ISA_ENCODING_AGX3C,2,2,
                          AGX3_ALU_HINT_DISCARD,false,0,bytes,&len));
   assert(!agx3_alu_hceil(AGX_ISA_ENCODING_AGX3B,2,2,
                          AGX3_ALU_HINT_CACHE,false,0,bytes,&len));
   assert(!agx3_alu_hceil(AGX_ISA_ENCODING_AGX3B,2,2,
                          AGX3_ALU_HINT_DISCARD,false,1,bytes,&len));
   assert(len==0xa5);
   for(unsigned i=0;i<sizeof(bytes);++i) assert(bytes[i]==0xa5);
   puts("Native half ceil: five exact Apple vectors and refusal guards PASS");
}

static void test_native_high_float_registers(void)
{
   /* Original native80-value kernels force retained ALU values into the upper
    * register bank. These are complete-instruction byte vectors, not patched
    * low-register examples; they retain native hints and scoreboard waits. */
   static const struct {
      enum agx3_alu_op op; uint16_t dest,src[3];
      enum agx3_alu_hint hints[3]; uint16_t wait; const char *bytes;
   } vectors[]={
      {AGX3_ALU_FMUL,168,{0,2,0},{AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_NONE},4,"498345810070"},
      {AGX3_ALU_FMUL,172,{4,6,0},{AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_NONE},0,"698755050010"},
      {AGX3_ALU_FMUL,174,{6,160,0},{AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_NONE},0,"79a155070011"},
      {AGX3_ALU_FMUL,142,{142,128,0},{AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_NONE},0,"7981150f0015"},
      {AGX3_ALU_FMUL,110,{110,152,0},{AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_NONE},0,"7999d56f0001"},
      {AGX3_ALU_FMUL,70,{70,56,0},{AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_NONE},0,"39b99147"},
      {AGX3_ALU_FMUL,38,{38,24,0},{AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_NONE},0,"39995127"},
      {AGX3_ALU_FMUL,4,{184,186,0},{AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_NONE},0,"293b1d390005"},
      {AGX3_ALU_FMA,168,{2,0,4},{AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_CACHE},4,"4981468301840270"},
      {AGX3_ALU_FMA,172,{6,4,152},{AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_CACHE},0,"69054e8741980210"},
      {AGX3_ALU_FMA,174,{152,6,154},{AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_CACHE},0,"79074e99419a0214"},
      {AGX3_ALU_FMA,150,{128,150,130},{AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_CACHE},0,"b9170e8141820215"},
      {AGX3_ALU_FMA,118,{160,86,162},{AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_CACHE},0,"b957cea141a20204"},
      {AGX3_ALU_FMA,70,{48,70,50},{AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_CACHE},0,"39478eb100b2"},
      {AGX3_ALU_FMA,38,{104,24,106},{AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_CACHE},0,"39194ee900ea"},
      {AGX3_ALU_FMA,2,{102,100,104},{AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD},0,"19651e678068"},
   };
   for(unsigned i=0;i<sizeof(vectors)/sizeof(vectors[0]);i++) {
      uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
      assert(agx3_alu_float_reg(AGX_ISA_ENCODING_AGX3B,vectors[i].op,vectors[i].dest,
         vectors[i].src,vectors[i].hints,vectors[i].wait,bytes,&len));
      exact(bytes,len,vectors[i].bytes);
   }
   puts("Native full-bank FMUL/FMA:16 exact high-register/lifetime/wait vectors PASS");
}

static void test_gpu1_float_compare_select(void)
{
   /* Native GPU1 fseli32.ep4 composer vectors: gears' 0 < x ? y : 0,
    * reversed comparison, reversed alternatives, and upper registers/waits.
    * These check instruction encoding, not hardware shader semantics. */
   const uint8_t expected[4][14] = {
      {0x32,0x02,0x47,0x6a,0x23,0x00,0xa0,0x1a,0x14,0x80,0x0a,0x08,0x1a,0x00},
      {0x32,0x02,0x07,0x6a,0x23,0x00,0xa0,0x1a,0x14,0x80,0x0a,0x48,0x1a,0x00},
      {0x32,0x02,0x47,0x6b,0x23,0x00,0xa0,0x1a,0x14,0x80,0x4a,0x00,0x04,0x01},
      {0xb2,0x84,0x4f,0xea,0x2b,0x80,0xa1,0x3a,0x30,0x80,0x0a,0x08,0x9a,0x40},
   };
   const bool no_abs[2] = {false, false};
   uint8_t out[AGX3_ALU_MAX_LENGTH], len;
   for (unsigned i = 0; i < 4; ++i) {
      const uint16_t src[4] = {
         i == 3 ? 188 : 14, 0, i == 2 ? 0 : i == 3 ? 186 : 26,
         i == 2 ? 26 : 0,
      };
      const bool immediate[2] = {i == 2, i != 2};
      const enum agx3_alu_hint hints[4] = {
         i == 3 ? AGX3_ALU_HINT_CACHE : AGX3_ALU_HINT_DISCARD,
         AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE,
      };
      assert(agx3_alu_fcmp_select_reg_imm(AGX_ISA_ENCODING_AGX3C,
         i == 3 ? 190 : 14, src, true, no_abs, immediate, hints,
         i == 1 ? 1 : 2, false, i == 3, i == 3 ? 0x81 : 0, out, &len));
      assert(len == sizeof(expected[i]) && !memcmp(out, expected[i], len));
   }
   uint16_t src[4] = {14, 0, 26, 0};
   const bool immediate[2] = {false, true};
   const enum agx3_alu_hint hints[4] = {0};
   assert(!agx3_alu_fcmp_select_reg_imm(AGX_ISA_ENCODING_AGX3C,
      14, src, true, no_abs, immediate, hints, 2, true, false, 0, out, &len));
   src[3] = 1; /* A minifloat payload is not an integer result immediate. */
   assert(!agx3_alu_fcmp_select_reg_imm(AGX_ISA_ENCODING_AGX3C,
      14, src, true, no_abs, immediate, hints, 2, false, false, 0, out, &len));
}

int main(void){
   test_gpu1_float_compare_select();
   test_native_popcount();
   test_native_minifloat_half_select();
   test_native_bit_andc();
   test_native_ft1();
   test_native_f32_rcp_rsqrt();
   test_native_f32_log2_exp2();
   test_native_f32_log2_exp2_modifiers();
   test_native_f32_floor_ceil();
   test_native_high_float_registers();
   test_native_widen_half();
   test_native_half_add();
   test_native_half_saturate();
   test_native_half_inverse();
   test_native_half_multiply();
   test_native_half_multiply_minifloat();
   test_native_minifloat_mixed_select();
   test_native_half_minifloat_select_f32();
   test_native_fadd_mixed_half();
   test_native_fadd_abs_mixed_half();
   test_native_half_fma();
   test_native_half_fma_minifloat_addend();
   test_native_narrow_multiply();
   test_native_sqrt_step();
   test_native_half_sqrt();
   test_native_select_reg_imm();
   test_native_half_compare_select();
   test_native_half_min();
   test_native_half_max();
   test_native_f32_minmax();
   test_native_f32_minifloat_select();
   test_native_half_clamp();
   test_native_reciprocal_widths();
   test_native_half_rsqrt();
   test_native_half_exp2();
   test_native_half_log2();
   test_native_mixed_width_log2();
   test_native_half_floor();
   test_native_half_ceil();
   test_native_asr32();
   test_native_extract32();
   test_blur_half_arithmetic();
   test_native_u16_arithmetic();
   test_native_u16_shift_right_immediate();
   test_native_fma_minifloat();
   test_int_to_float_liveness();
   test_i32_to_f16();
   test_f16_to_int32();
   test_float_compare_select();
   test_float_compare_zero();
   test_integer_lifetimes();
   test_bitwise_lifetimes();
   test_modified_minifloat_add();
   test_half_uv();
   test_native_fmul_abs_mixed();
   test_fmul_minifloat();
   test_fmul_minifloat_modifiers();
   test_fmul_half_minifloat_to_f32();
   test_fadd_minifloat();
   test_u16_to_f32();
   unsigned pass=0,fail=0,unsupported=0;
   for(size_t i=0;i<avn;i++){
      const struct av*v=&avs[i];
      /* fmin/fmax use a tied-operand form this encoder does not model. */
      if(!agx3_alu_wait_supported((enum agx3_alu_op)v->op)){ unsupported++; continue; }
      enum agx_isa_encoding e=v->enc?AGX_ISA_ENCODING_AGX3C:AGX_ISA_ENCODING_AGX3B;
      uint8_t out[AGX3_ALU_MAX_LENGTH]; uint8_t len=0;
      bool ok=agx3_alu_compose(e,(enum agx3_alu_op)v->op,2,4,6,8,0x1234u,0,out,&len);
      if(!ok||len!=v->n||memcmp(out,v->b,v->n)){ if(fail<12) fprintf(stderr,"FAIL op=%d enc=%d\n",v->op,v->enc); fail++; } else pass++;
   }
   /* Wait deposit: same op/operands, one slot waited -> same length, different bytes. */
   for(int e=0;e<2;e++){
      enum agx_isa_encoding E=e?AGX_ISA_ENCODING_AGX3C:AGX_ISA_ENCODING_AGX3B;
      uint8_t a[16],b[16]; uint8_t la,lb;
      if(agx3_alu_compose(E,AGX3_ALU_IADD,2,4,6,0,0,0,a,&la) &&
         agx3_alu_compose(E,AGX3_ALU_IADD,2,4,6,0,0,0x1,b,&lb)){
         pass++;
         if(la!=lb||memcmp(a,b,la)==0){ fprintf(stderr,"FAIL wait deposit enc=%d\n",e); fail++; }
      }

      /* A wait-only prelude uses ldimm on a fresh destination or bit_and on an
       * in-place accumulator. Cover all six G17P slots and the same common
       * subset on AGX3c.
       */
      const unsigned slots=6;
      for(unsigned slot=0;slot<slots;slot++){
         if(!agx3_alu_compose(E,AGX3_ALU_LDIMM,2,0,0,0,0,
                              1u<<slot,b,&lb)){
            fprintf(stderr,"FAIL ldimm wait slot=%u enc=%d\n",slot,e);
            fail++;
         } else pass++;
         if(!agx3_alu_compose(E,AGX3_ALU_FMUL,2,4,6,0,0,
                              1u<<slot,b,&lb)){
            unsupported++;
         } else pass++;
         if(!agx3_alu_compose(E,AGX3_ALU_BAND,2,2,2,0,0,
                              1u<<slot,b,&lb)){
            fprintf(stderr,"FAIL band wait slot=%u enc=%d\n",slot,e);
            fail++;
         } else pass++;
      }
   }
   if(fail){ fprintf(stderr,"agx3_alu_core: %u failed (%u passed, %u unsupported)\n",fail,pass,unsupported); return 1; }
   printf("agx3_alu_core: %u checks passed (%u unsupported forms skipped)\n",pass,unsupported);
   return 0;
}
