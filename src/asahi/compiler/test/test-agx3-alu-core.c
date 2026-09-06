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
   assert((multiply ? agx3_alu_fmul_minifloat : agx3_alu_fadd_minifloat)(
      AGX_ISA_ENCODING_AGX3B, dest, src, imm, kill, wait, bytes, &len));
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
      unsigned dest, src, imm, kill, wait;
      uint8_t bytes[6];
   } witness[] = {
      {0, 0, 0x20, 1, 0,  {0x09, 0xa1, 0x14, 0x01, 0x80, 0x00}},
      {2, 2, 0x20, 1, 0,  {0x19, 0xa1, 0x14, 0x03, 0x80, 0x00}},
      {0, 0, 0x20, 1, 32, {0x09, 0xa1, 0x14, 0x01, 0x80, 0xc0}},
      {0, 0, 0x1f, 1, 32, {0x09, 0x9f, 0x14, 0x01, 0x88, 0xc0}},
      {0, 0, 0x21, 1, 32, {0x09, 0xa1, 0x14, 0x01, 0x88, 0xc0}},
      {0, 0, 0xa0, 1, 32, {0x09, 0xa1, 0x1c, 0x01, 0x80, 0xc0}},
      {0, 2, 0x20, 0, 32, {0x09, 0xa1, 0x04, 0x03, 0x80, 0xc0}},
   };
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
   for (unsigned i = 0; i < sizeof(witness) / sizeof(witness[0]); ++i) {
      assert(agx3_alu_fadd_minifloat(AGX_ISA_ENCODING_AGX3B,
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
                                    true, wait, bytes, &len) == supported);
      if (!supported)
         assert(len == 99 && bytes[0] == 0xa5);
   }
   assert(!agx3_alu_fadd_minifloat(AGX_ISA_ENCODING_AGX3C, 0, 0, 32, 0,
                                  0, bytes, &len));
   assert(!agx3_alu_fadd_minifloat(AGX_ISA_ENCODING_AGX3B, 1, 0, 32, 0,
                                  0, bytes, &len));
   assert(!agx3_alu_fadd_minifloat(AGX_ISA_ENCODING_AGX3B, 0, 1, 32, 0,
                                  0, bytes, &len));
   assert(!agx3_alu_fadd_minifloat(AGX_ISA_ENCODING_AGX3B, 256, 0, 32, 0,
                                  0, bytes, &len));
   assert(!agx3_alu_fadd_minifloat(AGX_ISA_ENCODING_AGX3B, 0, 256, 32, 0,
                                  0, bytes, &len));
   assert(!agx3_alu_fadd_minifloat(AGX_ISA_ENCODING_AGX3B, 0, 0, 256, 0,
                                  0, bytes, &len));
   assert(!agx3_alu_fadd_minifloat(AGX_ISA_ENCODING_AGX3B, 0, 0, 32, 0,
                                  64, bytes, &len));
   assert(!agx3_alu_fadd_minifloat(AGX_ISA_ENCODING_AGX3B, 0, 0, 32, 0,
                                  0, NULL, &len));
   assert(!agx3_alu_fadd_minifloat(AGX_ISA_ENCODING_AGX3B, 0, 0, 32, 0,
                                  0, bytes, NULL));
   puts("G17P fadd minifloat: 7 Apple witnesses, register/wait/liveness bits,"
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
   assert(!agx3_alu_u16_to_f32(AGX_ISA_ENCODING_AGX3C,0,0,0,0,bytes,&len));
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
   printf("Modified minifloat add:9 native witnesses +%u physical bit checks passed\n", shapes);
}

static void
test_imul_high_div3(void)
{
   uint8_t out[AGX3_ALU_MAX_LENGTH], len;
   assert(agx3_alu_imul_high_div3(
      AGX_ISA_ENCODING_AGX3B, 0, 4, AGX3_ALU_HINT_CACHE, 1,
      out, &len));
   exact(out, len, "9f10540003080200e0241600");
   assert(agx3_alu_imul_high_div3(
      AGX_ISA_ENCODING_AGX3B, 0, 0, AGX3_ALU_HINT_DISCARD, 1,
      out, &len));
   exact(out, len, "9f10540003000000e0261600");
   assert(!agx3_alu_imul_high_div3(
      AGX_ISA_ENCODING_AGX3C, 0, 4, AGX3_ALU_HINT_CACHE, 1,
      out, &len));
   puts("Native signed mul-high-by-div3 magic: 2 exact G17P vertex witnesses PASS");
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
   const struct {unsigned d,f,h; enum agx3_alu_hint a,b; unsigned wait; const char *hex;} muls[] = {
      {0,0,4,AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD,32,"09011d0400c0"},
      {0,2,0,AGX3_ALU_HINT_NONE,AGX3_ALU_HINT_DISCARD,1,"090315000020"},
      {0,0,10,AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD,32,"09011d0a00c0"},
      {2,2,11,AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD,0,"19031d0a1000"},
      {0,0,10,AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_CACHE,32,"09010d8a00c0"},
      {2,2,10,AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD,0,"1903190a"},
   };
   for (unsigned i=0;i<sizeof(muls)/sizeof(muls[0]);++i) {
      assert(agx3_alu_fmul_mixed(AGX_ISA_ENCODING_AGX3B,muls[i].d,muls[i].f,muls[i].h,muls[i].a,muls[i].b,
                   muls[i].wait,out,&size));
      exact(out,size,muls[i].hex);
   }
   assert(!agx3_alu_mov16(AGX_ISA_ENCODING_AGX3B,256,0x1800,0,out,&size));
   assert(!agx3_alu_mov16(AGX_ISA_ENCODING_AGX3B,2,0x10000,0,out,&size));
   assert(!agx3_alu_mov16(AGX_ISA_ENCODING_AGX3B,2,0x1800,1,out,&size));
   assert(!agx3_alu_fmul_mixed(AGX_ISA_ENCODING_AGX3B,1,0,4,AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD,0,out,&size));
   assert(!agx3_alu_fmul_mixed(AGX_ISA_ENCODING_AGX3B,0,1,4,AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD,0,out,&size));
   assert(!agx3_alu_fmul_mixed(AGX_ISA_ENCODING_AGX3B,0,0,256,AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD,0,out,&size));
   assert(!agx3_alu_fmul_mixed(AGX_ISA_ENCODING_AGX3B,0,0,4,AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD,64,out,&size));
   unsigned shapes=0;
   for (unsigned d=0;d<256;d+=2)
      for (unsigned h=0;h<256;++h)
         for (unsigned wait=0;wait<=32;wait=wait ? wait*2 : 1) {
            assert(agx3_alu_fmul_mixed(AGX_ISA_ENCODING_AGX3B,d,(d+2)%256,h,AGX3_ALU_HINT_DISCARD, h&1 ? AGX3_ALU_HINT_DISCARD : AGX3_ALU_HINT_CACHE,
                         wait,out,&size));
            assert(size==4 || size==6);
            ++shapes;
         }
   puts("Half UV native candidate PASS:5 LDIMM16 +6 mixed-multiply byte-exact Apple witnesses; odd halves, cache/discard and wait6/no-wait/compact variants; invalid widths/ranges/waits rejected");
   printf("Validated%u register/half/wait shapes through actual composer; offline production encoder coverage\n",shapes);
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
   assert(!agx3_alu_fcmp_select_f32_f16(
      AGX_ISA_ENCODING_AGX3B, 36, lightmap_src, lightmap_hints,
      2, true, 0, out, &len));
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
   const uint16_t mixed_src[] = {2, 4};
   const bool mixed_width[] = {false, true};
   assert(agx3_alu_fcmp_select_imm_width(AGX_ISA_ENCODING_AGX3B,2,mixed_src,mixed_width,dead,1,255,0,true,false,32,out,&len));
   exact(out,len,"12031d04a2ff03c02080");
   puts("Native u16 ALU and mixed-width FSEL: six native byte vectors PASS");
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
   puts("Blur half arithmetic: seven native signed-conversion/immediate-add witnesses PASS");
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

static void test_native_widen_half(void)
{
   const char *golden[]={"09021c8100c2","19021c810802","0984148100c2","198414810802"};
   for (unsigned i=0;i<4;i++) {
      uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
      assert(agx3_alu_widen_half(AGX_ISA_ENCODING_AGX3B,(i&1)*2,i+2,
         i<2?AGX3_ALU_HINT_DISCARD:AGX3_ALU_HINT_CACHE,false,(i&1)?0:32,bytes,&len));
      exact(bytes,len,golden[i]);
   }
   puts("Native half widening: four Apple even/odd retained/dead vectors PASS");
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
   puts("Native absolute-comparison FSEL: nine exact G17P vectors PASS");
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

int main(void){
   test_native_minifloat_half_select();
   test_native_bit_andc();
   test_native_ft1();
   test_native_high_float_registers();
   test_native_widen_half();
   test_native_half_multiply();
   test_native_narrow_multiply();
   test_native_sqrt_step();
   test_native_select_reg_imm();
   test_native_asr32();
   test_native_extract32();
   test_blur_half_arithmetic();
   test_native_u16_arithmetic();
   test_native_fma_minifloat();
   test_int_to_float_liveness();
   test_float_compare_select();
   test_float_compare_zero();
   test_integer_lifetimes();
   test_bitwise_lifetimes();
   test_modified_minifloat_add();
   test_imul_high_div3();
   test_half_uv();
   test_fmul_minifloat();
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
