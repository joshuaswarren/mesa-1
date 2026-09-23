/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "agx3_uniform_alu.h"
#include <stdio.h>
#include <string.h>

static unsigned checks, failures;
#define CHECK(x) do { \
   ++checks; \
   if (!(x)) { \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); \
      ++failures; \
   } \
} while (0)

static void
native_destinations(void)
{
   /* AIR-NT 32023.921 native-uniform-arithmetic.metal, G17P and G16G.
    * ADD/MUL offsets40/46/52/58, FMA54/62/70/78 in fragment __text.
    * These bytes come from the native compiler, not this encoder. */
   const uint8_t witness[3][4][8] = {
      {{0x29,0x09,0x1c,0x01,0x40,0x20},
       {0x39,0x0b,0x1c,0x03,0x40,0x00},
       {0x49,0x0d,0x1c,0x05,0x40,0x00},
       {0x59,0x0f,0x1c,0x07,0x40,0x00}},
      {{0x29,0x09,0x1d,0x01,0x40,0x20},
       {0x39,0x0b,0x1d,0x03,0x40,0x00},
       {0x49,0x0d,0x1d,0x05,0x40,0x00},
       {0x59,0x0f,0x1d,0x07,0x40,0x00}},
      {{0x29,0x01,0x1e,0x09,0x81,0x10,0x42,0x20},
       {0x39,0x03,0x1e,0x0b,0x81,0x12,0x42,0x00},
       {0x49,0x05,0x1e,0x0d,0x81,0x14,0x42,0x00},
       {0x59,0x07,0x1e,0x0f,0x81,0x16,0x42,0x00}},
   };
   for (unsigned op = 0; op < 3; ++op) {
      for (unsigned lane = 0; lane < 4; ++lane) {
         struct agx3_uniform_float_src src[3] = {
            {.kind=AGX3_FLOAT_REGISTER, .value=2*lane+(op==2?0:8), .discard=true},
            {.kind=AGX3_FLOAT_REGISTER, .value=2*lane+(op==2?8:0), .discard=true},
            {.kind=AGX3_FLOAT_REGISTER, .value=2*lane+16, .discard=true},
         };
         uint8_t bytes[16], length;
         CHECK(agx3_uniform_float_store_encode(AGX_ISA_ENCODING_AGX3B, op,
                    4 + 2 * lane, src, lane == 0 ? 1 : 0, bytes, &length));
         CHECK(length == (op == 2 ? 8 : 6));
         CHECK(memcmp(bytes, witness[op][lane], length) == 0);
      }
   }
   /* Existing texture corpus: direct uniform destination, reg * minifloat. */
   struct agx3_uniform_float_src src[3] = {
      {.kind=AGX3_FLOAT_IMMEDIATE, .value=64},
      {.kind=AGX3_FLOAT_REGISTER, .value=2, .cache=true},
   };
   const uint8_t immediate[] = {0x89,0xc1,0x05,0x83,0xc0,0x00};
   uint8_t bytes[16], length;
   CHECK(agx3_uniform_float_store_encode(AGX_ISA_ENCODING_AGX3B,
              AGX3_UNIFORM_FMUL, 16, src, 0, bytes, &length));
   CHECK(length == sizeof(immediate));
   CHECK(memcmp(bytes, immediate, length) == 0);
}

static void
main_program_contract(void)
{
   const uint8_t witness[3][8] = {
      {0x09,0x01,0x0c,0x0d,0x00,0xc2},
      {0x09,0x01,0x0d,0x0d,0x00,0xc2},
      {0x09,0x01,0x0e,0x0d,0x21,0x0e,0x02,0xc2},
   };
   const uint8_t half_witness[3][8] = {
      {0x00,0x00,0x0c,0x0c,0x00,0xc2},
      {0x00,0x00,0x0d,0x0c,0x00,0xc2},
      {0x00,0x00,0x0e,0x0c,0x21,0x0e,0x00,0xc2},
   };
   struct agx3_uniform_float_src src[3] = {
      {.kind=AGX3_FLOAT_REGISTER, .discard=true},
      {.kind=AGX3_FLOAT_UNIFORM, .value=12},
      {.kind=AGX3_FLOAT_UNIFORM, .value=14},
   };
   for (unsigned op = 0; op < 3; ++op) {
      uint8_t bytes[16], length;
      CHECK(agx3_uniform_float_encode(AGX_ISA_ENCODING_AGX3B, op, 0, src,
                                      false, false, 32, bytes, &length));
      CHECK(length == (op == 2 ? 8 : 6));
      CHECK(memcmp(bytes, witness[op], length) == 0);
      CHECK(agx3_uniform_half_encode(AGX_ISA_ENCODING_AGX3B, op, 0, src,
                                     false, false, 32, bytes, &length));
      CHECK(length == (op == 2 ? 8 : 6));
      CHECK(memcmp(bytes, half_witness[op], length) == 0);
      CHECK(!agx3_uniform_float_store_encode(AGX_ISA_ENCODING_AGX3B, op, 0,
                                             src, 32, bytes, &length));
   }
}

static void
reject_unsupported(void)
{
   struct agx3_uniform_float_src src[3] = {
      {.kind=AGX3_FLOAT_REGISTER}, {.kind=AGX3_FLOAT_REGISTER, .value=2},
   };
   uint8_t bytes[16], unchanged[16], length = 0x5a;
   memset(bytes, 0xa5, sizeof(bytes));
   memcpy(unchanged, bytes, sizeof(bytes));
   CHECK(!agx3_uniform_float_store_encode(AGX_ISA_ENCODING_AGX2,
              AGX3_UNIFORM_FADD, 0, src, 0, bytes, &length));
   CHECK(!agx3_uniform_float_store_encode(AGX_ISA_ENCODING_AGX3C,
              AGX3_UNIFORM_FADD, 0, src, 0, bytes, &length));
   CHECK(!agx3_uniform_float_store_encode(AGX_ISA_ENCODING_AGX3B,
              AGX3_UNIFORM_FADD, 1, src, 0, bytes, &length));
   CHECK(!agx3_uniform_float_store_encode(AGX_ISA_ENCODING_AGX3B,
              AGX3_UNIFORM_FADD, 128, src, 0, bytes, &length));
   CHECK(!agx3_uniform_float_store_encode(AGX_ISA_ENCODING_AGX3B,
              AGX3_UNIFORM_FADD, 0, src, 64, bytes, &length));
   src[0].neg = true;
   CHECK(!agx3_uniform_float_store_encode(AGX_ISA_ENCODING_AGX3B,
              AGX3_UNIFORM_FADD, 0, src, 0, bytes, &length));
   src[0].neg = false;
   src[0].cache = src[0].discard = true;
   CHECK(!agx3_uniform_float_store_encode(AGX_ISA_ENCODING_AGX3B,
              AGX3_UNIFORM_FADD, 0, src, 0, bytes, &length));
   CHECK(memcmp(bytes, unchanged, sizeof(bytes)) == 0);
   CHECK(length == 0x5a);
}

int main(void)
{
   native_destinations();
   main_program_contract();
   reject_unsupported();
   printf("AGX3 uniform destination: %u checks, %u failures\n", checks, failures);
   return failures != 0;
}
