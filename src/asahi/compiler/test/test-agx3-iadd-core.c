/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Every positive vector is an instruction Apple's own compiler emitted.
 * Kernels of the form `o[i] = (cast(x[i]) << S) + 7u` were compiled with
 * `metal-nt -arch applegpu_g16g`, varying one thing at a time: the shift
 * amount, the source width (uint vs ushort), and its signedness (ushort vs
 * short). The wait vectors come from kernels whose addend is a loaded value,
 * where the consuming add waits on the load's scoreboard slot.
 */

#include "agx3_iadd_core.h"

#include <stdio.h>
#include <string.h>

static unsigned checks, failures;

#define CHECK(x)                                                               \
   do {                                                                        \
      ++checks;                                                                \
      if (!(x)) {                                                              \
         fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x);          \
         ++failures;                                                           \
      }                                                                        \
   } while (0)

struct vec {
   const char *name;
   struct agx3_iadd_desc d;
   uint8_t bytes[10];
};

/* All ten bytes; every vector below is length 10. */
static const struct vec vectors[] = {
   {"u32 shift0 wait6",
    {.src_kill = true, .dest = 0, .immediate = 7, .src = 0, .wait_mask = 1u << 5},
    {0x9f, 0x01, 0x56, 0x00, 0x02, 0x0e, 0x00, 0x88, 0x15, 0x04}},
   {"u32 shift1 wait6",
    {.src_kill = true, .dest = 0, .immediate = 7, .src = 0, .shift = 1, .wait_mask = 1u << 5},
    {0x9f, 0x01, 0x56, 0x00, 0x02, 0x0e, 0x00, 0xc8, 0x14, 0x04}},
   {"u32 shift3 wait6",
    {.src_kill = true, .dest = 0, .immediate = 7, .src = 0, .shift = 3, .wait_mask = 1u << 5},
    {0x9f, 0x01, 0x56, 0x00, 0x02, 0x0e, 0x00, 0xc8, 0x14, 0x14}},
   {"u32 shift0 wait1",
    {.src_kill = true, .dest = 0, .immediate = 7, .src = 0, .wait_mask = 1u << 0},
    {0x9f, 0x11, 0x54, 0x00, 0x02, 0x0e, 0x00, 0x88, 0x15, 0x04}},
   {"u32 dest4 src2 wait6",
    {.src_kill = true, .dest = 4, .immediate = 7, .src = 2, .wait_mask = 1u << 5},
    {0x9f, 0x01, 0x56, 0x04, 0x02, 0x0e, 0x08, 0x88, 0x15, 0x04}},
   {"u16 zero-extend shift0",
    {.src_kill = true, .dest = 0, .immediate = 7, .src = 0, .src_16 = true,
     .wait_mask = 1u << 5},
    {0x9f, 0x01, 0x56, 0x00, 0x02, 0x0e, 0x00, 0x08, 0x15, 0x04}},
   {"u16 zero-extend shift2",
    {.src_kill = true, .dest = 0, .immediate = 7, .src = 0, .src_16 = true, .shift = 2,
     .wait_mask = 1u << 5},
    {0x9f, 0x01, 0x56, 0x00, 0x02, 0x0e, 0x00, 0x08, 0x14, 0x14}},
   {"s16 sign-extend shift0",
    {.src_kill = true, .dest = 0, .immediate = 7, .src = 0, .src_16 = true, .sign_extend = true,
     .wait_mask = 1u << 5},
    {0x9f, 0x01, 0x56, 0x00, 0x02, 0x0e, 0x00, 0x08, 0x15, 0x0c}},
   {"s16 sign-extend shift2",
    {.src_kill = true, .dest = 0, .immediate = 7, .src = 0, .src_16 = true, .sign_extend = true,
     .shift = 2, .wait_mask = 1u << 5},
    {0x9f, 0x01, 0x56, 0x00, 0x02, 0x0e, 0x00, 0x08, 0x14, 0x1c}},
};


/*
 * Apple's shifted-add and multiply forms, captured from five compute kernels
 * on an A18 Pro that differ only in the shift amount:
 *
 *   imad_s0    o[i] = a[i]*b[i] + c[i]
 *   imad_s1..3 o[i] = a[i]*b[i] + (c[i] << N)   for N = 1, 2, 3
 *   imad_addr  o[i] = a[i]*16u  + (i << 2)
 *
 * The decisive result is that s1, s2 and s3 have a BYTE-IDENTICAL imadd: the
 * shift never reaches the multiply. Apple emits an unshifted product and puts
 * the shift on a following add, where f54 carries it (1, 2, 3 and 4 across
 * these kernels). agx_emit_imadshl_agx() splits AGX3's shifted imad the same
 * way rather than inventing a shift field the instruction does not have.
 */
static int
test_apple_shift_witnesses(void)
{
   uint8_t out[AGX3_IADD_MAX_LENGTH];
   uint8_t len;

   /* imad_s0: a*b + c, register multiplier, waits slot 6. */
   {
      static const uint8_t want[] = {0x9f, 0x00, 0x56, 0x00, 0x02, 0x08,
                                     0x00, 0x40, 0xd0, 0x2f, 0x2a, 0x00};
      struct agx3_imad_desc d = {
         .dest = 0, .src_a = 4, .multiplier_is_register = true,
         .multiplier_reg = 0, .multiplier_kill = true, .src_c = 8,
         .src_a_kill = true, .src_c_kill = true,
         .wait_mask = 1u << 5,
      };
      len = 0;
      CHECK(agx3_imad_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
      CHECK(len == sizeof(want) && memcmp(out, want, sizeof(want)) == 0);
   }

   /* imsub: `a*b - c` is a DIFFERENT INSTRUCTION, not a modifier on imad.
    * AGX3b.xml carries `imsub` with a field layout identical to `imadd`;
    * only opcode bit 7 differs.  Same descriptor as imad_s0 above, one
    * variable changed, from metal-nt -arch applegpu_g17p:
    *
    *   o[i] = a[i]*b[i] + c[i]   9f 00 56 00 02 08 00 40 d0 2f 2a 00
    *   o[i] = a[i]*b[i] - c[i]   1f 00 56 00 02 08 00 40 d0 2f 2a 00
    *
    * and the same bit serves an immediate addend, from the same run:
    *
    *   o[i] = a[i]*b[i] + 1      9f 00 56 00 02 08 00 08 d0 26 0a 00
    *   o[i] = a[i]*b[i] - 1      1f 00 56 00 02 08 00 08 d0 26 0a 00
    *   o[i] = a[i]*b[i] + 2      9f 00 56 00 02 08 00 10 d0 26 0a 00
    *   o[i] = a[i]*b[i] - 2      1f 00 56 00 02 08 00 10 d0 26 0a 00
    *
    * so f45 (the slot-C immediate, value << 3) and the opcode bit are
    * independent.  Before this, a negated immediate addend was refused and a
    * negated REGISTER addend composed as plain `imadd` -- `a*b - c` silently
    * computing `a*b + c`.
    */
   {
      static const uint8_t want[] = {0x1f, 0x00, 0x56, 0x00, 0x02, 0x08,
                                     0x00, 0x40, 0xd0, 0x2f, 0x2a, 0x00};
      struct agx3_imad_desc d = {
         .dest = 0, .src_a = 4, .multiplier_is_register = true,
         .multiplier_reg = 0, .multiplier_kill = true, .src_c = 8,
         .src_a_kill = true, .src_c_kill = true,
         .negate_addend = true,
         .wait_mask = 1u << 5,
      };
      len = 0;
      CHECK(agx3_imad_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
      CHECK(len == sizeof(want) && memcmp(out, want, sizeof(want)) == 0);

      /* A pure multiply has no addend to negate: fail closed rather than
       * emit an unwitnessed form. */
      struct agx3_imad_desc bad = d;
      bad.no_addend = true;
      bad.src_c = 0;
      bad.src_c_kill = false;
      len = 0;
      CHECK(!agx3_imad_encode(AGX_ISA_ENCODING_AGX3B, &bad, out, &len));
   }

   /* imad_s1..s3: the pure multiply, identical for every shift.
    *
    * The product feeds the shifted add below, so its destination carries the
    * register-cache hint. That bit (f11) is not a baseline constant: Apple's
    * `o[0] = a[i] * b[i]`, whose product is consumed only by a store, is the
    * same instruction with the bit clear (test-agx3-wide-imad.c, umul_narrow).
    */
   {
      static const uint8_t want[] = {0x9f, 0x00, 0x56, 0x00, 0x03, 0x10,
                                     0x00, 0x00, 0xd0, 0x26, 0x0a, 0x00};
      struct agx3_imad_desc d = {
         .dest = 0, .dest_cache = true, .src_a = 8,
         .multiplier_is_register = true,
         .multiplier_reg = 0, .multiplier_kill = true, .no_addend = true,
         .src_a_kill = true,
         .wait_mask = 1u << 5,
      };
      len = 0;
      CHECK(agx3_imad_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
      CHECK(len == sizeof(want) && memcmp(out, want, sizeof(want)) == 0);
   }

   /* The shifted adds that carry what the multiply does not. */
   {
      static const struct {
         uint8_t shift, src_a, src, wait;
         uint8_t bytes[10];
      } adds[] = {
         {1, 0, 4, 1u << 0,
          {0x9f, 0x11, 0x54, 0x00, 0x02, 0x00, 0x10, 0xe8, 0x16, 0x05}},
         {2, 0, 4, 1u << 0,
          {0x9f, 0x11, 0x54, 0x00, 0x02, 0x00, 0x10, 0xa8, 0x16, 0x15}},
         {3, 0, 4, 1u << 0,
          {0x9f, 0x11, 0x54, 0x00, 0x02, 0x00, 0x10, 0xe8, 0x16, 0x15}},
         {4, 4, 0, 1u << 5,
          {0x9f, 0x01, 0x56, 0x00, 0x02, 0x08, 0x00, 0xa8, 0x16, 0x05}},
      };

      for (unsigned i = 0; i < sizeof(adds) / sizeof(adds[0]); ++i) {
         struct agx3_iadd_desc d = {
            .src_a_is_register = true,
            .src_a_reg = adds[i].src_a,
            .src_a_kill = true,
            .src = adds[i].src,
            .src_kill = true,
            .shift = adds[i].shift,
            .wait_mask = adds[i].wait,
         };
         len = 0;
         CHECK(agx3_iadd_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
         CHECK(len == 10 && memcmp(out, adds[i].bytes, 10) == 0);
      }
   }

   /*
    * Narrow register operands. f24 is slot A's width and f39 is slot B's, on
    * both encodings -- the register slot-A form used to pin both to 32-bit,
    * which made `iadd d32, h_a, h_b` inexpressible and refused every shader
    * that sums gl_LocalInvocationID's three 16-bit components.
    *
    * Every vector below is Apple's own, from
    * `metal-nt -N compute.mtlp -arch applegpu_g17{p,s}`:
    *
    *   o[i] = (uint)a[2i] + (uint)a[2i+1]   over ushort   (f24=1 f39=1)
    *   o[i] =  (int)a[2i] +  (int)a[2i+1]   over short    (f28=1 f43=1)
    *   o[0] = lid.x + lid.y + lid.z                       (16+16, then 32+16)
    */
   {
      static const uint8_t u16_u16[] = {0x9f, 0x01, 0x56, 0x00, 0x02,
                                        0x00, 0x04, 0x08, 0x17, 0x05};
      struct agx3_iadd_desc d = {
         .src_a_is_register = true, .src_a_reg = 0, .src_a_16 = true,
         .src = 1, .src_16 = true, .src_a_kill = true, .src_kill = true,
         .wait_mask = 1u << 5,
      };
      len = 0;
      CHECK(agx3_iadd_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
      CHECK(len == sizeof(u16_u16) && memcmp(out, u16_u16, len) == 0);
   }
   {
      static const uint8_t s16_s16[] = {0x9f, 0x01, 0x56, 0x00, 0x02,
                                        0x02, 0x00, 0x08, 0x17, 0x0f};
      struct agx3_iadd_desc d = {
         .src_a_is_register = true, .src_a_reg = 1, .src_a_16 = true,
         .src_a_sign_extend = true, .src = 0, .src_16 = true,
         .sign_extend = true, .src_a_kill = true, .src_kill = true,
         .wait_mask = 1u << 5,
      };
      len = 0;
      CHECK(agx3_iadd_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
      CHECK(len == sizeof(s16_s16) && memcmp(out, s16_s16, len) == 0);
   }
   {
      /* lid.x + lid.y: both halves, and Apple caches the result (f11). */
      static const uint8_t lid_a[] = {0x9f, 0x11, 0x54, 0x00, 0x03,
                                      0x00, 0x04, 0x08, 0x17, 0x05};
      struct agx3_iadd_desc d = {
         .src_a_is_register = true, .src_a_reg = 0, .src_a_16 = true,
         .src = 1, .src_16 = true, .src_a_kill = true, .src_kill = true,
         .dest_cache = true, .wait_mask = 1u << 0,
      };
      len = 0;
      CHECK(agx3_iadd_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
      CHECK(len == sizeof(lid_a) && memcmp(out, lid_a, len) == 0);
   }
   {
      /* ... + lid.z: a 32-bit slot A and a 16-bit slot B, no dest cache. */
      static const uint8_t lid_b[] = {0x9f, 0x21, 0x54, 0x00, 0x02,
                                      0x00, 0x08, 0x28, 0x17, 0x05};
      struct agx3_iadd_desc d = {
         .src_a_is_register = true, .src_a_reg = 0, .src = 2,
         .src_16 = true, .src_a_kill = true, .src_kill = true,
         .wait_mask = 1u << 1,
      };
      len = 0;
      CHECK(agx3_iadd_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
      CHECK(len == sizeof(lid_b) && memcmp(out, lid_b, len) == 0);
   }
   {
      /* The same pair on AGX3c (applegpu_g17s), same operand fields. */
      static const uint8_t lid_a[] = {0x27, 0x04, 0x04, 0x1a, 0x29, 0x00,
                                      0xa3, 0x02, 0x60, 0x00, 0x21, 0x00};
      static const uint8_t lid_b[] = {0x27, 0x08, 0x04, 0x1a, 0x21, 0x00,
                                      0xa3, 0x02, 0xa8, 0x00, 0x21, 0x00};
      struct agx3_iadd_desc a = {
         .src_a_is_register = true, .src_a_reg = 0, .src_a_16 = true,
         .src = 1, .src_16 = true, .src_a_kill = true, .src_kill = true,
         .dest_cache = true, .wait_mask = 1u << 0,
      };
      struct agx3_iadd_desc b = {
         .src_a_is_register = true, .src_a_reg = 0, .src = 2,
         .src_16 = true, .src_a_kill = true, .src_kill = true,
         .wait_mask = 1u << 1,
      };
      len = 0;
      CHECK(agx3_iadd_encode(AGX_ISA_ENCODING_AGX3C, &a, out, &len));
      CHECK(len == sizeof(lid_a) && memcmp(out, lid_a, len) == 0);
      len = 0;
      CHECK(agx3_iadd_encode(AGX_ISA_ENCODING_AGX3C, &b, out, &len));
      CHECK(len == sizeof(lid_b) && memcmp(out, lid_b, len) == 0);
   }
   {
      /* Slot A width and sign-extension exist only for a REGISTER slot A;
       * asking for them with an immediate must fail closed, not silently
       * drop them. */
      struct agx3_iadd_desc d = {.src_a_16 = true, .src = 0, .src_16 = true};
      len = 0;
      CHECK(!agx3_iadd_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
      d.src_a_16 = false;
      d.src_a_sign_extend = true;
      CHECK(!agx3_iadd_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
   }
   return 0;
}

static void
test_native_immediate_subtract(void)
{
   static const uint8_t native[24][10] = {
      {0x1f,0x01,0x56,0x00,0x02,0x00,0x04,0x08,0x13,0x01},
      {0x1f,0x01,0x56,0x00,0x02,0x04,0x04,0x08,0x11,0x01},
      {0x1f,0x01,0x56,0x00,0x02,0x00,0x1c,0x08,0x13,0x01},
      {0x1f,0x01,0x56,0x00,0x02,0x04,0x1c,0x08,0x11,0x01},
      {0x1f,0x01,0x56,0x00,0x02,0x00,0x70,0x08,0x13,0x01},
      {0x1f,0x01,0x56,0x00,0x02,0x04,0x70,0x08,0x11,0x01},
      {0x1f,0x01,0x56,0x00,0x02,0x00,0xfc,0x0b,0x13,0x01},
      {0x1f,0x01,0x56,0x00,0x02,0x04,0xfc,0x0b,0x11,0x01},
      {0x1f,0x01,0x56,0x00,0x02,0x00,0x04,0x08,0x13,0x03},
      {0x1f,0x01,0x54,0x00,0x02,0x04,0x04,0x08,0x11,0x03},
      {0x1f,0x01,0x56,0x00,0x02,0x00,0x1c,0x08,0x13,0x03},
      {0x1f,0x01,0x54,0x00,0x02,0x04,0x1c,0x08,0x11,0x03},
      {0x1f,0x01,0x56,0x00,0x02,0x00,0x70,0x08,0x13,0x03},
      {0x1f,0x01,0x54,0x00,0x02,0x04,0x70,0x08,0x11,0x03},
      {0x1f,0x01,0x56,0x00,0x02,0x00,0xfc,0x0b,0x13,0x03},
      {0x1f,0x01,0x54,0x00,0x02,0x04,0xfc,0x0b,0x11,0x03},
      {0x1f,0x01,0x56,0x00,0x02,0x00,0x04,0x28,0x13,0x01},
      {0x1f,0x01,0x56,0x00,0x02,0x04,0x04,0x28,0x11,0x01},
      {0x1f,0x01,0x56,0x00,0x02,0x00,0x1c,0x28,0x13,0x01},
      {0x1f,0x01,0x56,0x00,0x02,0x04,0x1c,0x28,0x11,0x01},
      {0x1f,0x01,0x56,0x00,0x02,0x00,0x70,0x28,0x13,0x01},
      {0x1f,0x01,0x56,0x00,0x02,0x04,0x70,0x28,0x11,0x01},
      {0x1f,0x01,0x56,0x00,0x02,0x00,0xfc,0x2b,0x13,0x01},
      {0x1f,0x01,0x56,0x00,0x02,0x04,0xfc,0x2b,0x11,0x01}
   };
   const unsigned values[]={1,7,28,255};
   for(unsigned type=0;type<3;type++)for(unsigned c=0;c<4;c++)for(unsigned live=0;live<2;live++) {
      const struct agx3_isub_imm_desc d={
         .dest=0,.src=live ? 2 : 0,.immediate=values[c],
         .src_16=type!=2,.sign_extend=type==1,.src_kill=!live,
         .wait_mask=(type==1 && live) ? 0 : 32,
      };
      uint8_t out[AGX3_IADD_MAX_LENGTH],len;
      CHECK(agx3_isub_immediate_encode(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
      CHECK(len==10 && !memcmp(out,native[type*8+c*2+live],10));
   }
   uint8_t out[AGX3_IADD_MAX_LENGTH],len;
   struct agx3_isub_imm_desc d={.src_16=true,.immediate=28};
   d.dest=255; CHECK(!agx3_isub_immediate_encode(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
   d.dest=0; d.immediate=256; CHECK(!agx3_isub_immediate_encode(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
   d.immediate=28; d.src_cache=d.src_kill=true;
   CHECK(!agx3_isub_immediate_encode(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
   d.src_cache=d.src_kill=false; d.wait_mask=64;
   CHECK(!agx3_isub_immediate_encode(AGX_ISA_ENCODING_AGX3B,&d,out,&len));

   /* Independently composed HAL300 AGX3c isub32 vectors: dimension minus
    * one, signed half widening, and cached source with dependency slot 7. */
   static const struct {
      struct agx3_isub_imm_desc d;
      uint8_t bytes[12];
   } m5[] = {
      {{.dest=8,.src=8,.immediate=1,.src_kill=true},
       {0x27,0x00,0x04,0x5a,0x25,0x00,0xa2,0x12,0x48,0x00,0x0b,0x00}},
      {{.dest=2,.src=3,.immediate=255,.src_16=true,.sign_extend=true,
        .wait_mask=128},
       {0x3f,0x03,0x04,0x1a,0x21,0x00,0xa2,0x02,0xc0,0x3f,0x4b,0x00}},
      {{.dest=0,.src=2,.immediate=28,.src_cache=true,.wait_mask=64},
       {0x27,0x82,0x44,0x1a,0x21,0x00,0xa2,0x02,0x08,0x07,0x0b,0x00}},
   };
   for (unsigned i=0;i<sizeof(m5)/sizeof(m5[0]);i++) {
      CHECK(agx3_isub_immediate_encode(AGX_ISA_ENCODING_AGX3C,
                                      &m5[i].d,out,&len));
      CHECK(len==12 && !memcmp(out,m5[i].bytes,12));
   }
   d=m5[0].d; d.wait_mask=256;
   CHECK(!agx3_isub_immediate_encode(AGX_ISA_ENCODING_AGX3C,&d,out,&len));
}

static void
test_native_imad_immediate_addend(void)
{
   static const uint8_t apple_reg_imm[12] = {
      0x9f,0x00,0x56,0x00,0x02,0x08,0x00,0x08,0xd0,0x26,0x0a,0x00,
   };
   /* Fresh Apple G17P AIR 7326f3e6841d... / native fbc46f52c6ff...
    * from `out[i] = a[i] * b[i] + 1u`. This is the exact register
    * multiplier plus immediate addend that libagx_increment_ia requires. */
   const struct agx3_imad_desc apple_reg_imm_desc = {
      .dest=0, .src_a=4, .src_a_kill=true,
      .multiplier_is_register=true, .multiplier_reg=0,
      .multiplier_kill=true, .addend_is_immediate=true,
      .addend_immediate=1, .wait_mask=32,
   };
   uint8_t apple_out[AGX3_IADD_MAX_LENGTH], apple_len = 0;
   CHECK(agx3_imad_encode(AGX_ISA_ENCODING_AGX3B, &apple_reg_imm_desc,
                          apple_out, &apple_len));
   CHECK(apple_len == 12 && !memcmp(apple_out, apple_reg_imm, 12));
   static const uint8_t native[6][12] = {
      {0x9f,0x00,0x56,0x00,0x02,0x00,0x94,0x28,0x50,0x22,0x02,0x00},
      {0x9f,0x00,0x56,0x00,0x02,0x04,0x94,0x28,0x50,0x20,0x02,0x00},
      {0x9f,0x00,0x56,0x00,0x02,0x00,0x24,0x89,0x50,0x22,0x02,0x00},
      {0x9f,0x00,0x56,0x00,0x02,0x04,0x24,0x89,0x50,0x20,0x02,0x00},
      {0x9f,0x00,0x56,0x00,0x02,0x00,0xfc,0x23,0x52,0x22,0x02,0x00},
      {0x9f,0x00,0x56,0x00,0x02,0x04,0xfc,0x23,0x52,0x20,0x02,0x00}
   };
   const unsigned mul[]={37,73,255}, add[]={5,17,68};
   for(unsigned i=0;i<3;i++)for(unsigned live=0;live<2;live++) {
      const struct agx3_imad_desc d={
         .src_a=live ? 2 : 0,.src_a_kill=!live,.multiplier=mul[i],
         .addend_is_immediate=true,.addend_immediate=add[i],.wait_mask=32,
      };
      uint8_t out[AGX3_IADD_MAX_LENGTH],len;
      CHECK(agx3_imad_encode(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
      CHECK(len==12 && !memcmp(out,native[i*2+live],12));
   }
   struct agx3_imad_desc d={.multiplier=9,.addend_is_immediate=true,.addend_immediate=256};
   uint8_t out[AGX3_IADD_MAX_LENGTH],len;
   CHECK(!agx3_imad_encode(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
   d.addend_immediate=5; d.src_c_kill=true;
   CHECK(!agx3_imad_encode(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
   d = apple_reg_imm_desc;
   d.addend_immediate = 256;
   CHECK(!agx3_imad_encode(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
   d = apple_reg_imm_desc;
   d.src_c = 2;
   CHECK(!agx3_imad_encode(AGX_ISA_ENCODING_AGX3B,&d,out,&len));

   /* applegpu-nt 32023.917, apple-g17p ushort MAD. The vector form pins
    * odd-half operands plus both no-wait and slot-6 wait encodings. */
   static const struct {
      struct agx3_imad_desc d;
      uint8_t bytes[12];
   } narrow[] = {
      {{.dest=2, .dest_16=true, .src_a=0, .src_a_16=true,
        .multiplier=7, .addend_is_immediate=true, .addend_immediate=3,
        .wait_mask=1},
       {0x9f,0x10,0x54,0x02,0x02,0x00,0x1c,0x18,0x00,0x20,0x02,0x00}},
      {{.dest=2, .dest_16=true, .src_a=2, .src_a_16=true,
        .src_a_kill=true, .multiplier=7, .addend_is_immediate=true,
        .addend_immediate=3, .wait_mask=32},
       {0x9f,0x00,0x56,0x02,0x02,0x04,0x1c,0x18,0x00,0x22,0x02,0x00}},
      {{.dest=3, .dest_16=true, .src_a=3, .src_a_16=true,
        .src_a_kill=true, .multiplier=11, .addend_is_immediate=true,
        .addend_immediate=9},
       {0x9f,0x00,0x54,0x03,0x02,0x06,0x2c,0x48,0x00,0x22,0x02,0x00}},
      /* Fresh applegpu_g17p Metal witnesses.  mul_u16 and mul_s16 are
       * byte-identical because only the low 16 product is retained. */
      {{.dest=0, .dest_16=true, .dest_cache=true,
        .src_a=0, .src_a_16=true, .src_a_kill=true,
        .multiplier_is_register=true, .multiplier_reg=4,
        .multiplier_16=true, .multiplier_kill=true,
        .no_addend=true, .wait_mask=1},
       {0x9f,0x10,0x54,0x00,0x03,0x00,0x10,0x00,0x00,0x26,0x0a,0x00}},
      {{.dest=5, .dest_16=true, .dest_cache=true,
        .src_a=0, .src_a_16=true, .src_a_kill=true,
        .multiplier_is_register=true, .multiplier_reg=4,
        .multiplier_16=true, .no_addend=true, .wait_mask=1},
       {0x9f,0x10,0x54,0x05,0x03,0x00,0x10,0x00,0x00,0x22,0x0a,0x00}},
      {{.dest=0, .dest_16=true, .dest_cache=true,
        .src_a=0, .src_a_16=true, .src_a_kill=true,
        .multiplier_is_register=true, .multiplier_reg=4,
        .multiplier_16=true, .multiplier_kill=true,
        .addend_is_immediate=true, .addend_immediate=1, .wait_mask=1},
       {0x9f,0x10,0x54,0x00,0x03,0x00,0x10,0x08,0x00,0x26,0x0a,0x00}},
      /* Current Metal 32023.255, applegpu_g17p, exact
       * `out[i] = a[i] * b[i] + c[i]` ushort MAD. */
      {{.dest=2, .dest_16=true,
        .src_a=3, .src_a_16=true, .src_a_kill=true,
        .multiplier_is_register=true, .multiplier_reg=2,
        .multiplier_16=true, .multiplier_kill=true,
        .src_c=4, .src_c_16=true, .src_c_kill=true, .wait_mask=32},
       {0x9f,0x00,0x56,0x02,0x02,0x06,0x08,0x20,0x00,0x2e,0x2a,0x00}},
      /* Current Metal 32023.255, applegpu_g17p, exact widening
       * `uint(ushort) * uint(ushort)` product. */
      {{.dest=0, .src_a=0, .src_a_16=true, .src_a_kill=true,
        .multiplier_is_register=true, .multiplier_reg=1,
        .multiplier_16=true, .multiplier_kill=true,
        .no_addend=true, .wait_mask=32},
       {0x9f,0x00,0x56,0x00,0x02,0x00,0x04,0x00,0x10,0x26,0x0a,0x00}},
      /* Same current-toolchain witness over signed short inputs. */
      {{.dest=0, .src_a=0, .src_a_16=true, .src_a_sign_extend=true,
        .src_a_kill=true,
        .multiplier_is_register=true, .multiplier_reg=1,
        .multiplier_16=true, .multiplier_sign_extend=true,
        .multiplier_kill=true, .no_addend=true, .wait_mask=32},
       {0x9f,0x00,0x56,0x00,0x02,0x00,0x04,0x00,0x10,0x26,0x1e,0x00}},
   };
   for (unsigned i=0; i<sizeof(narrow)/sizeof(narrow[0]); ++i) {
      CHECK(agx3_imad_encode(AGX_ISA_ENCODING_AGX3B,&narrow[i].d,out,&len));
      CHECK(len==12 && !memcmp(out,narrow[i].bytes,12));
   }
   struct agx3_imad_desc narrow_bad = narrow[3].d;
   narrow_bad.multiplier_16 = false;
   CHECK(!agx3_imad_encode(AGX_ISA_ENCODING_AGX3B,&narrow_bad,out,&len));
   narrow_bad = narrow[0].d;
   narrow_bad.dest_64 = true;
   CHECK(!agx3_imad_encode(AGX_ISA_ENCODING_AGX3B,&narrow_bad,out,&len));
   narrow_bad = narrow[3].d;
   narrow_bad.no_addend = false;
   narrow_bad.src_c = 2;
   narrow_bad.src_c_16 = false;
   CHECK(!agx3_imad_encode(AGX_ISA_ENCODING_AGX3B,&narrow_bad,out,&len));

   narrow_bad = narrow[8].d;
   narrow_bad.src_a_16 = false;
   CHECK(!agx3_imad_encode(AGX_ISA_ENCODING_AGX3B,&narrow_bad,out,&len));
   narrow_bad = narrow[8].d;
   narrow_bad.multiplier_16 = false;
   CHECK(!agx3_imad_encode(AGX_ISA_ENCODING_AGX3B,&narrow_bad,out,&len));
   narrow_bad = narrow[8].d;
   narrow_bad.multiplier_sign_extend = false;
   CHECK(!agx3_imad_encode(AGX_ISA_ENCODING_AGX3B,&narrow_bad,out,&len));
}

/*
 * Narrow imad operands widened into a 32-bit product.
 *
 * A bare narrow operand is SIGN-extended -- the convention
 * agx_optimizer_signext() enforces by dropping a SIGNEXT and asserting the
 * survivor carries no ABS. Both the addend (slot C, width f54, sign f58) and
 * the multiplicand (slot A, width f24, sign f28) can be narrow, and both were
 * refused: the packer gates required an explicit modifier, and the encoder
 * additionally allowed slot A's sign-extension only for a 64-bit result.
 *
 * Apple, one kernel per variable, everything else held fixed.
 */
static void
test_native_imad_narrow_operands(void)
{
   uint8_t out[AGX3_IADD_MAX_LENGTH];
   uint8_t len = 0;

   /* o[i] = a[i]*b[i] + int(short c[i]) -- narrow addend, sign-extended. */
   {
      static const uint8_t want[] = {0x9f, 0x00, 0x56, 0x00, 0x02, 0x08,
                                     0x00, 0x40, 0xd0, 0x2e, 0x6a, 0x00};
      /* Field vector: f16 = 4 (slot A), slot B a register at 0, f46 = 8
       * (slot C), f27/f42/f57 the three last-use hints, f65 the slot-6
       * wait, f54 = 1 and f58 = 1 the addend's width and sign. */
      struct agx3_imad_desc d = {
         .dest = 0, .src_a = 4, .src_a_kill = true,
         .multiplier_is_register = true, .multiplier_reg = 0,
         .multiplier_kill = true,
         .src_c = 8, .src_c_16 = true, .src_c_sign_extend = true,
         .src_c_kill = true, .wait_mask = 1u << 5,
      };
      len = 0;
      CHECK(agx3_imad_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
      CHECK(len == sizeof(want) && memcmp(out, want, len) == 0);

      /* ushort: the same instruction with f58 clear. */
      static const uint8_t want_u[] = {0x9f, 0x00, 0x56, 0x00, 0x02, 0x08,
                                       0x00, 0x40, 0xd0, 0x2e, 0x2a, 0x00};
      d.src_c_sign_extend = false;
      len = 0;
      CHECK(agx3_imad_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
      CHECK(len == sizeof(want_u) && memcmp(out, want_u, len) == 0);

      /* A 32-bit addend moves f54 from 1 to 2. */
      static const uint8_t want_32[] = {0x9f, 0x00, 0x56, 0x00, 0x02, 0x08,
                                        0x00, 0x40, 0xd0, 0x2f, 0x2a, 0x00};
      d.src_c_16 = false;
      len = 0;
      CHECK(agx3_imad_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
      CHECK(len == sizeof(want_32) && memcmp(out, want_32, len) == 0);
   }

   /* o[i] = int(short a[i])*129 + int(short c[i]) -- narrow multiplicand.
    * f28 is set with a 32-BIT result, which the encoder used to refuse. */
   {
      static const uint8_t want[] = {0x9f, 0x00, 0x56, 0x00, 0x02, 0x02,
                                     0x04, 0x02, 0x10, 0x2a, 0x66, 0x00};
      /* f16 = 1 (slot A, an odd half because it is 16-bit), f24 = 1 and
       * f28 = 1 its width and sign, f30 = 129 the immediate multiplier,
       * slot C at 0 with f54 = 1 and f58 = 1. */
      struct agx3_imad_desc d = {
         .dest = 0, .src_a = 1, .src_a_16 = true, .src_a_sign_extend = true,
         .src_a_kill = true, .multiplier = 129,
         .src_c = 0, .src_c_16 = true, .src_c_sign_extend = true,
         .src_c_kill = true, .wait_mask = 1u << 5,
      };
      len = 0;
      CHECK(agx3_imad_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
      CHECK(len == sizeof(want) && memcmp(out, want, len) == 0);

      static const uint8_t want_u[] = {0x9f, 0x00, 0x56, 0x00, 0x02, 0x02,
                                       0x04, 0x02, 0x10, 0x2a, 0x22, 0x00};
      d.src_a_sign_extend = false;
      d.src_c_sign_extend = false;
      len = 0;
      CHECK(agx3_imad_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
      CHECK(len == sizeof(want_u) && memcmp(out, want_u, len) == 0);
   }

   /* Still closed: a 32-bit multiplicand widens nothing, and a narrow
    * REGISTER multiplier with a 32-bit result has no witness. */
   {
      struct agx3_imad_desc d = {
         .dest = 0, .src_a = 0, .src_a_sign_extend = true, .multiplier = 3,
         .no_addend = true,
      };
      len = 0;
      CHECK(!agx3_imad_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
   }
   {
      struct agx3_imad_desc d = {
         .dest = 0, .src_a = 0, .multiplier_is_register = true,
         .multiplier_reg = 2, .multiplier_16 = true,
         .multiplier_sign_extend = true, .no_addend = true,
      };
      len = 0;
      CHECK(!agx3_imad_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
   }
}

static void
test_m5_imad(void)
{
   const struct {
      struct agx3_imad_desc d;
      uint8_t bytes[14];
   } vectors[] = {
      /* Native M5: retain all loaded operands after each calculation. */
      {{.src_a=4,.multiplier=37,.src_c=2,.wait_mask=128},
       {0x2f,0x00,0x04,0x3a,0x21,0x00,0xa1,0x02,0x48,0x09,0x4a,0x00,0x02,0x00}},
      {{.src_a=4,.multiplier_is_register=true,.multiplier_reg=6,
        .src_c=2,.wait_mask=128},
       {0x2f,0x00,0x04,0x3a,0x21,0x00,0xa1,0x02,0x88,0x81,0x40,0x00,0x02,0x00}},
      {{.src_a=2,.multiplier=37,.addend_is_immediate=true,
        .addend_immediate=1,.wait_mask=128},
       {0x2f,0x02,0x04,0x1a,0x21,0x00,0xa1,0x02,0x48,0x09,0x0e,0x00,0x01,0x02}},
      /* Image-copy addressing with independently consumed inputs. */
      {{.dest=2,.src_a=8,.multiplier=32,.src_c=2,
        .src_a_kill=true,.src_c_kill=true,.wait_mask=9},
       {0x37,0x24,0x04,0x5a,0x21,0x00,0xa1,0x02,0x28,0x08,0xca,0x00,0x02,0x00}},
   };
   uint8_t out[AGX3_IADD_MAX_LENGTH], len = 0;
   for (unsigned i = 0; i < sizeof(vectors)/sizeof(vectors[0]); ++i) {
      CHECK(agx3_imad_encode(AGX_ISA_ENCODING_AGX3C,&vectors[i].d,out,&len));
      CHECK(len == 14 && !memcmp(out,vectors[i].bytes,14));
   }
   struct agx3_imad_desc d = vectors[0].d;
   d.wait_mask = 255;
   CHECK(agx3_imad_encode(AGX_ISA_ENCODING_AGX3C,&d,out,&len));
   d.dest_64 = true;
   CHECK(!agx3_imad_encode(AGX_ISA_ENCODING_AGX3C,&d,out,&len));
   d.dest_64 = false;
   d.src_a_16 = true;
   CHECK(!agx3_imad_encode(AGX_ISA_ENCODING_AGX3C,&d,out,&len));
   d.src_a_16 = false;
   d.src_c_cache = d.src_c_kill = true;
   CHECK(!agx3_imad_encode(AGX_ISA_ENCODING_AGX3C,&d,out,&len));
}

int
main(void)
{
   test_m5_imad();
   test_native_imad_narrow_operands();
   for (unsigned i = 0; i < sizeof(vectors) / sizeof(vectors[0]); ++i) {
      const struct vec *v = &vectors[i];
      uint8_t out[AGX3_IADD_MAX_LENGTH];
      uint8_t len = 0;

      memset(out, 0xcd, sizeof(out));
      CHECK(agx3_iadd_encode(AGX_ISA_ENCODING_AGX3B, &v->d, out, &len));
      CHECK(len == 10);
      ++checks;
      if (len != 10 || memcmp(out, v->bytes, 10) != 0) {
         ++failures;
         fprintf(stderr, "FAIL %s: got", v->name);
         for (unsigned b = 0; b < len; ++b)
            fprintf(stderr, " %02x", out[b]);
         fprintf(stderr, ", want");
         for (unsigned b = 0; b < 10; ++b)
            fprintf(stderr, " %02x", v->bytes[b]);
         fprintf(stderr, "\n");
      }
   }

   /* Fresh G17P native witnesses: id remains live for the output address.
    * f42 is clear even though this add uses a shifted source. */
   {
      static const uint8_t native[][10] = {
         {0x9f,0x11,0x54,0x00,0x02,0x00,0x08,0xc8,0x10,0x04},
         {0x9f,0x11,0x54,0x00,0x02,0x00,0x08,0x88,0x10,0x14},
         {0x9f,0x11,0x54,0x00,0x02,0x00,0x08,0xc8,0x10,0x14},
         {0x9f,0x11,0x54,0x00,0x02,0x00,0x08,0x88,0x10,0x04},
      };
      for (unsigned shift = 1; shift <= 4; ++shift) {
         struct agx3_iadd_desc d = {.dest=0, .src=2, .shift=shift,
                                     .wait_mask=1, .src_kill=false};
         uint8_t out[AGX3_IADD_MAX_LENGTH], len=0;
         CHECK(agx3_iadd_encode(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
         CHECK(len==10 && !memcmp(out,native[shift-1],10));
      }
   }

   /* Fail closed outside the witnessed domain. */
   uint8_t out[AGX3_IADD_MAX_LENGTH];
   uint8_t len = 0;
   struct agx3_iadd_desc d = {.dest = 0, .immediate = 7, .src = 0};

   /* Shift 4 IS witnessed: Apple's `o[i] = a[i]*16u + (i << 2)` kernel emits
    * a shifted add with f54 = 4. Only 5 and above are out of domain. */
   d.shift = 4;
   CHECK(agx3_iadd_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
   d.shift = 5;
   CHECK(!agx3_iadd_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
   d.shift = 0;

   d.dest = 1; /* a 32-bit destination is an aligned pair */
   CHECK(!agx3_iadd_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
   d.dest = 0;

   d.src = 1; /* likewise a 32-bit source */
   CHECK(!agx3_iadd_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
   d.src = 1;
   d.src_16 = true; /* but a narrow source may sit in an odd half */
   CHECK(agx3_iadd_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
   d.src = 0;
   d.src_16 = false;

   d.wait_mask = 1u << 6; /* six scoreboard slots on AGX3b */
   CHECK(!agx3_iadd_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
   d.wait_mask = 0;

   CHECK(!agx3_iadd_encode(AGX_ISA_ENCODING_AGX2, &d, out, &len));

   /*
    * AGX3c, from the same ladder against `-arch applegpu_g17s`: `iadd32`,
    * the shift at f55, and eight wait slots. Its single load signals slot 8,
    * the top one, exactly as AGX3b's signals slot 6.
    */
   {
      static const struct {
         const char *name;
         struct agx3_iadd_desc d;
         uint8_t bytes[12];
      } cvec[] = {
         {"u32 shift0", {.dest = 0, .immediate = 7, .src = 0,
                         .wait_mask = 1u << 7},
          {0x2f, 0x03, 0x04, 0x3a, 0x25, 0x00, 0xa3, 0x02, 0x20, 0x80, 0x21,
           0x00}},
         {"u32 shift2", {.dest = 0, .immediate = 7, .src = 0, .shift = 2,
                         .wait_mask = 1u << 7},
          {0x2f, 0x03, 0x04, 0x3a, 0x25, 0x00, 0xa3, 0x02, 0x20, 0x80, 0x25,
           0x00}},
         {"u16 shift2", {.dest = 0, .immediate = 7, .src = 0, .src_16 = true,
                         .shift = 2, .wait_mask = 1u << 7},
          {0x2f, 0x03, 0x04, 0x3a, 0x25, 0x00, 0xa3, 0x02, 0x20, 0x00, 0x25,
           0x00}},
         {"s16 shift2", {.dest = 0, .immediate = 7, .src = 0, .src_16 = true,
                         .sign_extend = true, .shift = 2,
                         .wait_mask = 1u << 7},
          {0x2f, 0x03, 0x04, 0x3a, 0x25, 0x00, 0xa3, 0x02, 0x20, 0x00, 0xa5,
           0x00}},
      };
      for (unsigned i = 0; i < sizeof(cvec) / sizeof(cvec[0]); ++i) {
         uint8_t co[AGX3_IADD_MAX_LENGTH];
         uint8_t cl = 0;
         CHECK(agx3_iadd_encode(AGX_ISA_ENCODING_AGX3C, &cvec[i].d, co, &cl));
         CHECK(cl == 12);
         ++checks;
         if (cl != 12 || memcmp(co, cvec[i].bytes, 12) != 0) {
            ++failures;
            fprintf(stderr, "FAIL agx3c %s\n", cvec[i].name);
         }
      }

      /* AGX3c has eight slots, so bit 8 is still out of range. */
      struct agx3_iadd_desc bad = {.dest = 0, .immediate = 7, .src = 0,
                                   .wait_mask = 1u << 8};
      CHECK(!agx3_iadd_encode(AGX_ISA_ENCODING_AGX3C, &bad, out, &len));
   }

   /*
    * imad with a constant multiplier, and its addend in a narrow register.
    * From `o[i] = a[i] * 255u + <addend>` over a uint, a ushort and a short.
    */
   {
      static const struct {
         const char *name;
         struct agx3_imad_desc d;
         uint8_t bytes[12];
      } iv[] = {
         {"c 32-bit", {.dest = 0, .src_a = 0, .multiplier = 255, .src_c = 4,
           .src_a_kill = true, .src_c_kill = true,
                       .wait_mask = 1u << 5},
          {0x9f, 0x00, 0x56, 0x00, 0x02, 0x00, 0xfc, 0x23, 0x50, 0x2b, 0x22,
           0x00}},
         {"c 16-bit zero-extend",
          {.dest = 0, .src_a = 0, .multiplier = 255, .src_c = 4,
           .src_a_kill = true, .src_c_kill = true,
           .src_c_16 = true, .wait_mask = 1u << 5},
          {0x9f, 0x00, 0x56, 0x00, 0x02, 0x00, 0xfc, 0x23, 0x50, 0x2a, 0x22,
           0x00}},
         {"c 16-bit sign-extend",
          {.dest = 0, .src_a = 0, .multiplier = 255, .src_c = 4,
           .src_a_kill = true, .src_c_kill = true,
           .src_c_16 = true, .src_c_sign_extend = true, .wait_mask = 1u << 5},
          {0x9f, 0x00, 0x56, 0x00, 0x02, 0x00, 0xfc, 0x23, 0x50, 0x2a, 0x62,
           0x00}},
      };
      for (unsigned i = 0; i < sizeof(iv) / sizeof(iv[0]); ++i) {
         uint8_t o[AGX3_IADD_MAX_LENGTH];
         uint8_t l = 0;
         CHECK(agx3_imad_encode(AGX_ISA_ENCODING_AGX3B, &iv[i].d, o, &l));
         CHECK(l == 12);
         ++checks;
         if (l != 12 || memcmp(o, iv[i].bytes, 12) != 0) {
            ++failures;
            fprintf(stderr, "FAIL imad %s\n", iv[i].name);
         }
      }

      struct agx3_imad_desc bad = {.dest = 0, .src_a = 0, .multiplier = 255,
                                   .src_c = 4, .wait_mask = 1u << 6};
      CHECK(!agx3_imad_encode(AGX_ISA_ENCODING_AGX3B, &bad, out, &len));
      bad.wait_mask = 0;
      CHECK(!agx3_imad_encode(AGX_ISA_ENCODING_AGX2, &bad, out, &len));
   }

   /* Native imad_live retains all three inputs; imad_chain caches its
    * multiplier and addend across two operations, then discards both. */
   {
      const struct {
         struct agx3_imad_desc d;
         uint8_t bytes[12];
      } live[] = {
         {{.dest=0, .src_a=4, .multiplier_is_register=true,
           .multiplier_reg=2, .src_c=6, .wait_mask=32},
          {0x9f,0x00,0x56,0x00,0x02,0x08,0x08,0x30,0xd0,0x21,0x2a,0x00}},
         {{.dest=2, .src_a=2, .src_a_kill=true, .multiplier_is_register=true,
           .multiplier_reg=0, .multiplier_cache=true, .src_c=4,
           .src_c_cache=true, .wait_mask=32},
          {0x9f,0x00,0x56,0x02,0x02,0x04,0x00,0x24,0xd8,0x23,0x2a,0x00}},
         {{.dest=4, .src_a=6, .src_a_kill=true, .multiplier_is_register=true,
           .multiplier_reg=0, .multiplier_kill=true, .src_c=4,
           .src_c_kill=true, .wait_mask=0},
          {0x9f,0x00,0x54,0x04,0x02,0x0c,0x00,0x20,0xd0,0x2f,0x2a,0x00}},
      };
      for (unsigned i=0; i<sizeof(live)/sizeof(live[0]); ++i) {
         CHECK(agx3_imad_encode(AGX_ISA_ENCODING_AGX3B, &live[i].d, out, &len));
         CHECK(len==12 && !memcmp(out,live[i].bytes,12));
      }
   }

   /* Fresh native scalar constant multiplication without an addend. */
   {
      const uint8_t native[][12] = {
         {0x9f,0x00,0x56,0x00,0x02,0x00,0x94,0x00,0x50,0x22,0x02,0x00},
         {0x9f,0x00,0x56,0x00,0x02,0x00,0xfc,0x03,0x50,0x22,0x02,0x00},
      };
      for (unsigned i=0; i<2; ++i) {
         struct agx3_imad_desc m = {.src_a_kill=true, .no_addend=true,
            .multiplier=i ? 255 : 37, .wait_mask=32};
         CHECK(agx3_imad_encode(AGX_ISA_ENCODING_AGX3B,&m,out,&len));
         CHECK(len==12 && !memcmp(out,native[i],12));
      }
   }

   test_apple_shift_witnesses();
   test_native_immediate_subtract();
   test_native_imad_immediate_addend();

   printf("%u checks, %u failures\n", checks, failures);
   return failures ? 1 : 0;
}
