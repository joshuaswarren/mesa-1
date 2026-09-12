/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * The f32 modifier encoders must reproduce the composer's bytes -- the same
 * recipe (baseline + operands + modifiers) the RE proved byte-exact against
 * Apple -- across the abs/neg/saturate combinations of add, mul and fma.
 */

#include "agx3_falu_mod_core.h"
#include "agx3_falu_mod_core_vectors.c"

#include <stdio.h>
#include <string.h>

static unsigned
narrow_runtime_vectors(void)
{
   const struct {
      uint16_t dest, src;
      bool discard, dest_cache, saturate;
      uint8_t wait, size, bytes[8];
   } vectors[] = {
      {0,0,true,false,false,32,6,{0x01,0x01,0x1c,0x81,0,0xc2}},
      {1,2,true,false,false,0,6,{0x01,0x03,0x1c,0x81,0x04,0x02}},
      {3,6,true,false,false,0,6,{0x11,0x07,0x1c,0x81,0x04,0x02}},
      {0,2,false,true,false,32,6,{0x01,0x03,0x34,0x81,0,0xc2}},
      {2,2,true,false,true,32,8,{0x11,0x03,0x1c,0x81,0x01,0x02,0,0x82}},
   };
   unsigned fail = 0;
   for (unsigned i=0; i<sizeof(vectors)/sizeof(vectors[0]); ++i) {
      struct agx3_falu_runtime rt = {
         .discard={vectors[i].discard}, .dest_cache=vectors[i].dest_cache,
         .wait_mask=vectors[i].wait,
      };
      uint8_t bytes[16], length;
      if (!agx3_falu_narrow_f32(AGX_ISA_ENCODING_AGX3B, vectors[i].dest,
             vectors[i].src, vectors[i].saturate, &rt, bytes, &length) ||
          length!=vectors[i].size || memcmp(bytes,vectors[i].bytes,length)) {
         fprintf(stderr,"native narrowing vector %u failed\n",i);
         fail++;
      }
   }
   struct agx3_falu_runtime rt = {.cache={true},.wait_mask=63};
   uint8_t bytes[16], length=0;
   fail += !agx3_falu_narrow_f32(AGX_ISA_ENCODING_AGX3B,255,254,false,&rt,bytes,&length);
   fail += length>10;
   fail += agx3_falu_narrow_f32(AGX_ISA_ENCODING_AGX3B,256,254,false,&rt,bytes,&length);
   fail += agx3_falu_narrow_f32(AGX_ISA_ENCODING_AGX3B,0,255,false,&rt,bytes,&length);
   rt.discard[0]=true;
   fail += agx3_falu_narrow_f32(AGX_ISA_ENCODING_AGX3B,0,2,false,&rt,bytes,&length);
   printf("native f32-to-f16: odd/even halves, live/dead input, saturate, waits: %u failures\n",fail);
   return fail;
}

static unsigned
add_runtime_vectors(void)
{
   struct add_vector {
      uint16_t src0, src1;
      struct agx3_falu_mods mods;
      struct agx3_falu_runtime runtime;
      unsigned length;
      uint8_t bytes[10];
   };
   const struct add_vector vectors[] = {
      /* Original x-y ordering commutes to Apple's shorter negate-first form. */
      {0, 2, {.src1_neg=true}, {.discard={true,true}, .wait_mask=32}, 6,
       {0x09,0x01,0x1c,0x03,0,0xc8}},
      {2, 0, {.src0_neg=true}, {.discard={true,true}, .wait_mask=32}, 6,
       {0x09,0x01,0x1c,0x03,0,0xc8}},
      {4, 2, {.src0_neg=true}, {.wait_mask=32}, 6,
       {0x09,0x03,0x04,0x05,0,0xc8}},
      {2, 0, {.src0_abs=true, .src0_neg=true, .src1_abs=true},
       {.discard={true,true}, .wait_mask=32}, 10,
       {0x09,0x01,0x1c,0x03,0x02,0x08,0,0x80,0x03,0}},
      {2, 0, {.src0_neg=true, .saturate=true},
       {.discard={true,true}, .wait_mask=32}, 8,
       {0x09,0x01,0x1c,0x03,0x01,0x08,0,0x82}},
      {2, 0, {.src0_neg=true, .src1_neg=true},
       {.discard={true,true}, .wait_mask=32}, 8,
       {0x09,0x01,0x1c,0x03,0x01,0x08,0x02,0x80}},
   };
   unsigned fail = 0;
   for (unsigned i = 0; i < sizeof(vectors)/sizeof(vectors[0]); ++i) {
      const struct add_vector *v = &vectors[i];
      uint8_t out[16], length;
      if (!agx3_falu_add_mod_runtime(AGX_ISA_ENCODING_AGX3B, 0,
             v->src0, v->src1, &v->mods, &v->runtime, out, &length) ||
          length != v->length || memcmp(out, v->bytes, length)) {
         fprintf(stderr, "native modified ADD vector %u failed\n", i);
         fail++;
      }
   }
   struct agx3_falu_mods m = {.src0_neg=true, .src1_abs=true};
   struct agx3_falu_runtime rt = {.wait_mask=63};
   uint8_t out[16], length = 0;
   /* Long forms must carry multiple dependency slots without a separate
    * arithmetic operation or discarding either live input. */
   fail += !agx3_falu_add_mod_runtime(AGX_ISA_ENCODING_AGX3B, 254, 252, 250,
                                       &m, &rt, out, &length);
   fail += length > 10;
   memset(out, 0xa5, sizeof(out)); length = 0xa5;
   rt.cache[0] = rt.discard[0] = true;
   fail += agx3_falu_add_mod_runtime(AGX_ISA_ENCODING_AGX3B, 0, 2, 4,
                                      &m, &rt, out, &length);
   rt.cache[0] = false; rt.wait_mask = 64;
   fail += agx3_falu_add_mod_runtime(AGX_ISA_ENCODING_AGX3B, 0, 2, 4,
                                      &m, &rt, out, &length);
   rt.wait_mask = 0;
   fail += agx3_falu_add_mod_runtime(AGX_ISA_ENCODING_AGX3B, 0, 255, 4,
                                      &m, &rt, out, &length);
   fail += agx3_falu_add_mod_runtime(AGX_ISA_ENCODING_AGX2, 0, 2, 4,
                                      &m, &rt, out, &length);
   fail += length != 0xa5;
   for (unsigned i=0; i<sizeof(out); ++i) fail += out[i] != 0xa5;
   printf("native modified ADD: five Apple vectors plus commuted subtraction, live operands, waits and bounds: %u failures\n",fail);
   return fail;
}

static unsigned
m5_runtime_vectors(void)
{
   /* First three vectors are native G17G Metal kernels retaining the input
    * operands in their output. They therefore wait for slot8 without
    * discarding the sources. Remaining vectors exercise independent hints
    * and all eight waits against the G17G instruction composer. */
   const struct {
      unsigned op;
      uint16_t dest, src0, src1, addend;
      struct agx3_falu_mods mods;
      struct agx3_falu_runtime runtime;
      uint8_t length, bytes[16];
   } vectors[] = {
      {0,0,4,2,0,{.src0_neg=true},{.wait_mask=128},12,
       {0x29,0x02,0x24,0x0a,0x22,0x20,0xa0,0x02,0x02,0x00,0x12,0x00}},
      {1,0,2,4,0,{.src0_neg=true,.src1_abs=true,.saturate=true},
       {.wait_mask=128},14,
       {0x29,0x00,0x25,0x30,0x22,0x80,0xa0,0x02,0x98,0x00,0x04,0x02,0x00,0x10}},
      {2,0,4,2,6,{.src0_abs=true,.addend_neg=true},{.wait_mask=128},16,
       {0x29,0x02,0x07,0x0a,0x20,0x20,0xa3,0x02,0x18,0x01,0x80,0x08,0x06,0x00,0x00,0xc0}},
      {0,2,32,8,0,{.src1_neg=true},{.cache={false,true}},6,
       {0x19,0x21,0x04,0x89,0x00,0x08}},
      {0,2,32,8,0,{.src1_neg=true},{.cache={false,true},.wait_mask=255},12,
       {0x39,0xfc,0x44,0x2a,0x22,0x20,0xa0,0x82,0x20,0x00,0x12,0x04}},
      {1,8,32,6,0,{.src0_neg=true},
       {.cache={true},.discard={false,true},.wait_mask=255},14,
       {0x29,0x7e,0x65,0x30,0x22,0x80,0xa0,0x92,0x1c,0x08,0x10,0x00,0x00,0x10}},
      {2,8,32,6,2,{.src0_abs=true,.addend_neg=true},
       {.cache={true,false,true},.discard={false,true},.wait_mask=255},16,
       {0x29,0x7e,0x47,0x1a,0x20,0x20,0xa3,0x92,0x38,0x08,0x90,0x08,0x02,0x04,0x00,0xc0}},
   };
   unsigned fail = 0;
   for (unsigned i = 0; i < sizeof(vectors)/sizeof(vectors[0]); ++i) {
      uint8_t out[16], length = 0;
      bool ok;
      if (vectors[i].op == 0)
         ok = agx3_falu_add_mod_runtime(AGX_ISA_ENCODING_AGX3C,
            vectors[i].dest, vectors[i].src0, vectors[i].src1,
            &vectors[i].mods, &vectors[i].runtime, out, &length);
      else if (vectors[i].op == 1)
         ok = agx3_falu_mul_mod_runtime(AGX_ISA_ENCODING_AGX3C,
            vectors[i].dest, vectors[i].src0, vectors[i].src1,
            &vectors[i].mods, &vectors[i].runtime, out, &length);
      else
         ok = agx3_falu_fma_mod_runtime(AGX_ISA_ENCODING_AGX3C,
            vectors[i].dest, vectors[i].src0, vectors[i].src1, vectors[i].addend,
            &vectors[i].mods, &vectors[i].runtime, out, &length);
      if (!ok || length != vectors[i].length ||
          memcmp(out, vectors[i].bytes, length)) {
         fprintf(stderr, "M5 float runtime vector %u failed\n", i);
         fail++;
      }
   }
   struct agx3_falu_mods mods = {.src1_neg=true};
   struct agx3_falu_runtime rt = {.cache={true},.discard={true}};
   uint8_t out[16], length = 0xa5;
   memset(out, 0xa5, sizeof(out));
   fail += agx3_falu_add_mod_runtime(AGX_ISA_ENCODING_AGX3C,0,2,4,&mods,&rt,out,&length);
   fail += agx3_falu_mul_mod_runtime(AGX_ISA_ENCODING_AGX3C,0,2,4,&mods,&rt,out,&length);
   fail += agx3_falu_fma_mod_runtime(AGX_ISA_ENCODING_AGX3C,0,2,4,6,&mods,&rt,out,&length);
   rt.cache[0] = false;
   fail += agx3_falu_add_mod_runtime(AGX_ISA_ENCODING_AGX3C,0,3,4,&mods,&rt,out,&length);
   fail += agx3_falu_mul_mod_runtime(AGX_ISA_ENCODING_AGX3C,256,2,4,&mods,&rt,out,&length);
   fail += agx3_falu_fma_mod_runtime(AGX_ISA_ENCODING_AGX3C,0,2,4,7,&mods,&rt,out,&length);
   fail += length != 0xa5;
   for (unsigned i = 0; i < sizeof(out); ++i) fail += out[i] != 0xa5;
   printf("M5 float runtime: native ADD/MUL/FMA, live inputs, hints and eight waits: %u failures\n", fail);
   return fail;
}

static unsigned
runtime_vectors(void)
{
   unsigned fail = 0;
   {
      const uint8_t expected[] = {0x09, 0x01, 0x1e, 0x03, 0x91, 0x04, 0x02, 0xc0};
      const struct agx3_falu_mods m = {.src0_abs = 0, .src0_neg = 0, .src1_abs = 0, .src1_neg = 0, .addend_abs = 0, .addend_neg = 1, .saturate = 0};
      const struct agx3_falu_runtime rt = {.cache = {0, 0, 0}, .discard = {1, 1, 1}, .dest_cache = 0, .wait_mask = 32};
      uint8_t out[16], len;
      if (!agx3_falu_fma_mod_runtime(AGX_ISA_ENCODING_AGX3B, 0, 2, 0, 4,
                                      &m, &rt, out, &len) ||
          len != sizeof(expected) || memcmp(out, expected, len)) {
         fprintf(stderr, "native modified FMA vector 0 failed\n");
         fail++;
      }
   }
   {
      const uint8_t expected[] = {0x09, 0x03, 0x06, 0x05, 0x11, 0x06, 0x02, 0xc0};
      const struct agx3_falu_mods m = {.src0_abs = 0, .src0_neg = 0, .src1_abs = 0, .src1_neg = 0, .addend_abs = 0, .addend_neg = 1, .saturate = 0};
      const struct agx3_falu_runtime rt = {.cache = {0, 0, 0}, .discard = {0, 0, 0}, .dest_cache = 0, .wait_mask = 32};
      uint8_t out[16], len;
      if (!agx3_falu_fma_mod_runtime(AGX_ISA_ENCODING_AGX3B, 0, 4, 2, 6,
                                      &m, &rt, out, &len) ||
          len != sizeof(expected) || memcmp(out, expected, len)) {
         fprintf(stderr, "native modified FMA vector 1 failed\n");
         fail++;
      }
   }
   {
      const uint8_t expected[] = {0x09, 0x01, 0x1e, 0x03, 0x83, 0x04, 0x02, 0x00, 0x00, 0x80, 0x01, 0x00};
      const struct agx3_falu_mods m = {.src0_abs = 0, .src0_neg = 0, .src1_abs = 1, .src1_neg = 0, .addend_abs = 0, .addend_neg = 0, .saturate = 0};
      const struct agx3_falu_runtime rt = {.cache = {0, 0, 0}, .discard = {1, 1, 1}, .dest_cache = 0, .wait_mask = 32};
      uint8_t out[16], len;
      if (!agx3_falu_fma_mod_runtime(AGX_ISA_ENCODING_AGX3B, 0, 2, 0, 4,
                                      &m, &rt, out, &len) ||
          len != sizeof(expected) || memcmp(out, expected, len)) {
         fprintf(stderr, "native modified FMA vector 2 failed\n");
         fail++;
      }
   }
   {
      const uint8_t expected[] = {0x09, 0x03, 0x1e, 0x01, 0x81, 0x04, 0x02, 0xc8};
      const struct agx3_falu_mods m = {.src0_abs = 0, .src0_neg = 1, .src1_abs = 0, .src1_neg = 0, .addend_abs = 0, .addend_neg = 0, .saturate = 0};
      const struct agx3_falu_runtime rt = {.cache = {0, 0, 0}, .discard = {1, 1, 1}, .dest_cache = 0, .wait_mask = 32};
      uint8_t out[16], len;
      if (!agx3_falu_fma_mod_runtime(AGX_ISA_ENCODING_AGX3B, 0, 0, 2, 4,
                                      &m, &rt, out, &len) ||
          len != sizeof(expected) || memcmp(out, expected, len)) {
         fprintf(stderr, "native modified FMA vector 3 failed\n");
         fail++;
      }
   }
   {
      const uint8_t expected[] = {0x09, 0x01, 0x1e, 0x03, 0x89, 0x04, 0x02, 0xc0};
      const struct agx3_falu_mods m = {.src0_abs = 0, .src0_neg = 0, .src1_abs = 0, .src1_neg = 0, .addend_abs = 1, .addend_neg = 0, .saturate = 0};
      const struct agx3_falu_runtime rt = {.cache = {0, 0, 0}, .discard = {1, 1, 1}, .dest_cache = 0, .wait_mask = 32};
      uint8_t out[16], len;
      if (!agx3_falu_fma_mod_runtime(AGX_ISA_ENCODING_AGX3B, 0, 2, 0, 4,
                                      &m, &rt, out, &len) ||
          len != sizeof(expected) || memcmp(out, expected, len)) {
         fprintf(stderr, "native modified FMA vector 4 failed\n");
         fail++;
      }
   }
   {
      const uint8_t expected[] = {0x09, 0x01, 0x1e, 0x03, 0x82, 0x04, 0x02, 0x00, 0x00, 0x82};
      const struct agx3_falu_mods m = {.src0_abs = 0, .src0_neg = 0, .src1_abs = 0, .src1_neg = 0, .addend_abs = 0, .addend_neg = 0, .saturate = 1};
      const struct agx3_falu_runtime rt = {.cache = {0, 0, 0}, .discard = {1, 1, 1}, .dest_cache = 0, .wait_mask = 32};
      uint8_t out[16], len;
      if (!agx3_falu_fma_mod_runtime(AGX_ISA_ENCODING_AGX3B, 0, 2, 0, 4,
                                      &m, &rt, out, &len) ||
          len != sizeof(expected) || memcmp(out, expected, len)) {
         fprintf(stderr, "native modified FMA vector 5 failed\n");
         fail++;
      }
   }
   /* The witnessed abs source forces the long form. Independently inspect
    * all six physical wait bits and source hints, not the encoder's recipe. */
   for (unsigned waits = 0; waits < 64; ++waits) {
      const struct agx3_falu_mods m = {.src1_abs = true};
      struct agx3_falu_runtime rt = {.wait_mask = waits};
      uint8_t out[16], len;
      const unsigned wait_bits[] = {61, 62, 63, 77, 78, 79};
      if (!agx3_falu_fma_mod_runtime(AGX_ISA_ENCODING_AGX3B, 0, 2, 0, 4,
                                    &m, &rt, out, &len) || len != 12) {
         fail++;
         continue;
      }
      for (unsigned i = 0; i < 6; ++i)
         fail += ((out[wait_bits[i] / 8] >> (wait_bits[i] % 8)) & 1) !=
                 ((waits >> i) & 1);
      const unsigned cache_bits[] = {31, 15, 47};
      const unsigned discard_bits[] = {20, 19, 39};
      for (unsigned hints = 0; hints < 27; ++hints) {
         unsigned v = hints;
         for (unsigned i = 0; i < 3; ++i, v /= 3) {
            rt.cache[i] = (v % 3) == 1;
            rt.discard[i] = (v % 3) == 2;
         }
         if (!agx3_falu_fma_mod_runtime(AGX_ISA_ENCODING_AGX3B, 0, 2, 0, 4,
                                       &m, &rt, out, &len) || len != 12) {
            fail++;
            continue;
         }
         for (unsigned i = 0; i < 3; ++i) {
            fail += ((out[cache_bits[i] / 8] >> (cache_bits[i] % 8)) & 1)
                    != rt.cache[i];
            fail += ((out[discard_bits[i] / 8] >> (discard_bits[i] % 8)) & 1)
                    != rt.discard[i];
         }
      }
   }
   struct agx3_falu_mods m = {.addend_neg = true};
   struct agx3_falu_runtime rt = {.wait_mask = 64};
   uint8_t out[16], len = 0xa5;
   memset(out, 0xa5, sizeof(out));
   fail += agx3_falu_fma_mod_runtime(AGX_ISA_ENCODING_AGX3B, 0, 2, 4, 6,
                                    &m, &rt, out, &len);
   rt.wait_mask = 0;
   rt.cache[0] = rt.discard[0] = true;
   fail += agx3_falu_fma_mod_runtime(AGX_ISA_ENCODING_AGX3B, 0, 2, 4, 6,
                                    &m, &rt, out, &len);
   rt.cache[0] = false;
   fail += agx3_falu_fma_mod_runtime(AGX_ISA_ENCODING_AGX3B, 0, 2, 4, 255,
                                    &m, &rt, out, &len);
   fail += agx3_falu_fma_mod_runtime(AGX_ISA_ENCODING_AGX2, 0, 2, 4, 6,
                                    &m, &rt, out, &len);
   fail += len != 0xa5;
   for (unsigned i = 0; i < sizeof(out); ++i)
      fail += out[i] != 0xa5;
   printf("native modified FMA: six Apple byte vectors, 64 wait masks, "
          "27 independent lifetime combinations: %u failures\n", fail);
   return fail;
}

static unsigned
mul_runtime_vectors(void)
{
   const struct {
      uint16_t dst, s0, s1;
      struct agx3_falu_mods m;
      struct agx3_falu_runtime rt;
      unsigned size;
      uint8_t bytes[16];
   } vectors[] = {
      {0,6,4,{.src0_abs=0,.src0_neg=0,.src1_abs=0,.src1_neg=0,.saturate=1}, {.cache={0,0},.discard={1,1},.dest_cache=0,.wait_mask=32},8,{0x09,0x05,0x1d,0x07,0x01,0x00,0x00,0x82}},
      {0,4,2,{.src0_abs=0,.src0_neg=0,.src1_abs=0,.src1_neg=0,.saturate=1}, {.cache={0,0},.discard={0,0},.dest_cache=0,.wait_mask=32},8,{0x09,0x03,0x05,0x05,0x01,0x00,0x00,0x82}},
      {0,6,4,{.src0_abs=0,.src0_neg=0,.src1_abs=1,.src1_neg=0,.saturate=0}, {.cache={0,0},.discard={1,1},.dest_cache=0,.wait_mask=32},12,{0x09,0x05,0x1d,0x07,0x02,0x00,0x00,0x80,0x01,0x00,0x00,0x00}},
      {0,4,2,{.src0_abs=0,.src0_neg=0,.src1_abs=1,.src1_neg=0,.saturate=0}, {.cache={0,0},.discard={0,0},.dest_cache=0,.wait_mask=32},12,{0x09,0x03,0x05,0x05,0x02,0x00,0x00,0x80,0x01,0x00,0x00,0x00}},
      {0,4,6,{.src0_abs=0,.src0_neg=1,.src1_abs=0,.src1_neg=0,.saturate=0}, {.cache={0,0},.discard={1,1},.dest_cache=0,.wait_mask=32},6,{0x09,0x07,0x1d,0x05,0x00,0xc8}},
      {0,2,4,{.src0_abs=0,.src0_neg=1,.src1_abs=0,.src1_neg=0,.saturate=0}, {.cache={0,0},.discard={0,0},.dest_cache=0,.wait_mask=32},6,{0x09,0x05,0x05,0x03,0x00,0xc8}},
      {0,4,6,{.src0_abs=1,.src0_neg=1,.src1_abs=1,.src1_neg=0,.saturate=0}, {.cache={0,0},.discard={1,1},.dest_cache=0,.wait_mask=32},12,{0x09,0x07,0x1d,0x05,0x02,0x08,0x00,0x80,0x03,0x00,0x00,0x00}},
      {0,2,4,{.src0_abs=1,.src0_neg=1,.src1_abs=1,.src1_neg=0,.saturate=0}, {.cache={0,0},.discard={0,0},.dest_cache=0,.wait_mask=32},12,{0x09,0x05,0x05,0x03,0x02,0x08,0x00,0x80,0x03,0x00,0x00,0x00}},
   };
   unsigned fail = 0;
   for (unsigned i = 0; i < sizeof(vectors)/sizeof(vectors[0]); ++i) {
      uint8_t out[16], size;
      if (!agx3_falu_mul_mod_runtime(AGX_ISA_ENCODING_AGX3B,
              vectors[i].dst, vectors[i].s0, vectors[i].s1,
              &vectors[i].m, &vectors[i].rt, out, &size) ||
          size != vectors[i].size || memcmp(out, vectors[i].bytes, size)) {
         fprintf(stderr, "native modified FMUL vector %u failed size=%u: ", i, size);
         for (unsigned j=0;j<size;++j) fprintf(stderr,"%02x",out[j]);
         fprintf(stderr,"\n");
         fail++;
      }
   }
   printf("native modified FMUL: eight Apple live/dead byte vectors: %u failures\n", fail);
   return fail;
}

static unsigned
fma_addend_vectors(void)
{
   const struct {
      uint16_t dst,s0,s1; uint8_t imm;
      struct agx3_falu_runtime rt;
      unsigned size; uint8_t bytes[16];
   } vectors[] = {
      {0,6,4,48,{.cache={0,0},.discard={1,1},.dest_cache=0,.wait_mask=32},8,{0x09,0x05,0x1e,0x07,0x21,0xb0,0x02,0xc0}},
      {0,4,2,48,{.cache={0,0},.discard={0,0},.dest_cache=0,.wait_mask=32},8,{0x09,0x03,0x06,0x05,0x21,0xb0,0x02,0xc0}},
      {0,6,4,176,{.cache={0,0},.discard={1,1},.dest_cache=0,.wait_mask=32},8,{0x09,0x05,0x1e,0x07,0xa1,0xb0,0x02,0xc0}},
      {0,4,2,176,{.cache={0,0},.discard={0,0},.dest_cache=0,.wait_mask=32},8,{0x09,0x03,0x06,0x05,0xa1,0xb0,0x02,0xc0}},
      {0,6,4,32,{.cache={0,0},.discard={1,1},.dest_cache=0,.wait_mask=32},8,{0x09,0x05,0x1e,0x07,0x21,0xa0,0x02,0xc0}},
      {0,4,2,32,{.cache={0,0},.discard={0,0},.dest_cache=0,.wait_mask=32},8,{0x09,0x03,0x06,0x05,0x21,0xa0,0x02,0xc0}},
   };
   unsigned fail=0;
   for(unsigned i=0;i<sizeof(vectors)/sizeof(vectors[0]);++i){
      uint8_t out[16],size=0;
      if(!agx3_falu_fma_addend_imm(AGX_ISA_ENCODING_AGX3B,vectors[i].dst,
             vectors[i].s0,vectors[i].s1,vectors[i].imm,&vectors[i].rt,out,&size)||
         size!=vectors[i].size||memcmp(out,vectors[i].bytes,size)){
         fprintf(stderr,"FMA minifloat addend vector %u failed\n",i);fail++;
      }
   }
   {
      const struct {
         uint16_t dst, src0, src1;
         struct agx3_falu_runtime rt;
         uint8_t bytes[8];
      } negative_multiplier[] = {
         {0, 0, 2, {.discard={1,1}},
          {0x09,0x03,0x1e,0x01,0x21,0x98,0x02,0xc8}},
         {0, 2, 4, {.cache={0,0}, .discard={0,0}},
          {0x09,0x05,0x06,0x03,0x21,0x98,0x02,0xc8}},
      };
      for (unsigned i = 0;
           i < sizeof(negative_multiplier) / sizeof(negative_multiplier[0]);
           ++i) {
         uint8_t out[16], size = 0;
         if (!agx3_falu_fma_addend_imm_mod(
                AGX_ISA_ENCODING_AGX3B, negative_multiplier[i].dst,
                negative_multiplier[i].src0, negative_multiplier[i].src1,
                24, false, true, false, false,
                &negative_multiplier[i].rt, out, &size) ||
             size != sizeof(negative_multiplier[i].bytes) ||
             memcmp(out, negative_multiplier[i].bytes, size)) {
            fprintf(stderr,
                    "FMA minifloat addend negated-multiplier vector %u failed\n",
                    i);
            for (unsigned j = 0; j < size; ++j)
               fprintf(stderr, "%02x", out[j]);
            fprintf(stderr, " size=%u\n", size);
            fail++;
         }
      }
   }
   {
      /* applegpu-nt v32023.255, -arch applegpu_g17p:
       * fma(a, fabs(b), 1.0f).  This is the exact modifier-bearing
       * minifloat-addend form emitted by production GTK shaders. */
      static const uint8_t native[] = {
         0x09,0x01,0x1e,0x03,0x23,0xb0,0x02,0x00,0x00,0x80,0x02,0x00,
      };
      const struct agx3_falu_runtime rt = {.discard={true,true}};
      uint8_t out[16], size = 0;
      if (!agx3_falu_fma_addend_imm_mod(
             AGX_ISA_ENCODING_AGX3B, 0, 2, 0, 48,
             true, false, false, false, &rt, out, &size) ||
          size != sizeof(native) || memcmp(out, native, size)) {
         fprintf(stderr, "FMA absolute multiplier/immediate addend failed\n");
         for (unsigned i = 0; i < size; ++i)
            fprintf(stderr, "%02x", out[i]);
         fprintf(stderr, " size=%u\n", size);
         fail++;
      }
   }
   printf("native FMA minifloat addend: nine Apple live/dead/modifier vectors: "
          "%u failures\n", fail);
   return fail;
}

static unsigned
fma_two_vectors(void)
{
   const struct {
      uint16_t dst,src; uint8_t mul,add;
      struct agx3_falu_runtime rt;
      unsigned size; uint8_t bytes[16];
   } vectors[] = {
      {0,0,64,176,{.cache={0},.discard={1},.dest_cache=0,.wait_mask=32},8,{0x09,0x01,0x0e,0xc1,0xa1,0xb0,0x02,0xc2}},
      {0,2,64,176,{.cache={0},.discard={0},.dest_cache=0,.wait_mask=32},8,{0x09,0x03,0x06,0xc1,0xa1,0xb0,0x02,0xc2}},
      {0,0,32,16,{.cache={0},.discard={1},.dest_cache=0,.wait_mask=32},8,{0x09,0x01,0x0e,0xa1,0x21,0x90,0x02,0xc2}},
      {0,2,32,16,{.cache={0},.discard={0},.dest_cache=0,.wait_mask=32},8,{0x09,0x03,0x06,0xa1,0x21,0x90,0x02,0xc2}},
      {0,0,56,160,{.cache={0},.discard={1},.dest_cache=0,.wait_mask=32},8,{0x09,0x01,0x0e,0xb9,0xa1,0xa0,0x02,0xc2}},
      {0,2,56,160,{.cache={0},.discard={0},.dest_cache=0,.wait_mask=32},8,{0x09,0x03,0x06,0xb9,0xa1,0xa0,0x02,0xc2}},
   };
   unsigned fail=0;
   for(unsigned i=0;i<sizeof(vectors)/sizeof(vectors[0]);++i){
      uint8_t out[16],size=0;
      if(!agx3_falu_fma_two_imm(AGX_ISA_ENCODING_AGX3B,vectors[i].dst,
             vectors[i].src,vectors[i].mul,vectors[i].add,&vectors[i].rt,out,&size)||
         size!=vectors[i].size||memcmp(out,vectors[i].bytes,size)){
         fprintf(stderr,"FMA two-immediate vector %u failed\n",i);fail++;
      }
   }
   printf("native FMA two minifloats: six Apple live/dead vectors: %u failures\n",fail);
   return fail;
}

int
main(void)
{
   unsigned pass = 0, fail = runtime_vectors() + add_runtime_vectors() + narrow_runtime_vectors() + mul_runtime_vectors() + fma_addend_vectors() + fma_two_vectors() + m5_runtime_vectors();
   for (size_t i = 0; i < mod_vec_count; ++i) {
      const struct mod_vec *v = &mod_vecs[i];
      const enum agx_isa_encoding e =
         v->enc == 0 ? AGX_ISA_ENCODING_AGX3B : AGX_ISA_ENCODING_AGX3C;
      const struct agx3_falu_mods m = {
         .src0_abs = v->s0a, .src0_neg = v->s0n, .src1_abs = v->s1a,
         .src1_neg = v->s1n, .addend_abs = v->ada, .addend_neg = v->adn,
         .saturate = v->sat,
      };
      uint8_t out[AGX3_FALU_MOD_MAX_LENGTH];
      uint8_t len = 0;
      bool ok;
      if (!strcmp(v->mnem, "f32add"))
         ok = agx3_falu_add_mod(e, v->dest, v->s0, v->s1, &m, out, &len);
      else if (!strcmp(v->mnem, "f32mul"))
         ok = agx3_falu_mul_mod(e, v->dest, v->s0, v->s1, &m, out, &len);
      else
         ok = agx3_falu_fma_mod(e, v->dest, v->s0, v->s1, v->ad, &m, out, &len);

      if (!ok || len != v->nbytes || memcmp(out, v->bytes, v->nbytes) != 0) {
         if (fail < 10)
            fprintf(stderr, "FAIL %s enc=%d\n", v->mnem, v->enc);
         fail++;
      } else {
         pass++;
      }
   }
   if (fail) {
      fprintf(stderr, "agx3_falu_mod_core: %u of %zu failed\n", fail,
              mod_vec_count);
      return 1;
   }
   printf("agx3_falu_mod_core: all %u modifier forms reproduced\n", pass);
   return 0;
}
