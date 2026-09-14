/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "asahi/compiler/agx3_subgroup_core.h"
#include "util/macros.h"

#include <stdio.h>
#include <string.h>

#define CHECK(COND)                                                            \
   do {                                                                        \
      if (!(COND)) {                                                           \
         fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #COND);       \
         return 1;                                                             \
      }                                                                        \
   } while (0)

static bool
bytes_equal(const uint8_t *actual, uint8_t length, const uint8_t *expected,
            size_t expected_length)
{
   if (length == expected_length &&
       memcmp(actual, expected, expected_length) == 0)
      return true;

   fprintf(stderr, "length %u expected %zu; actual ", length, expected_length);
   for (unsigned i = 0; i < length; ++i)
      fprintf(stderr, "%02x", actual[i]);
   fprintf(stderr, " expected ");
   for (unsigned i = 0; i < expected_length; ++i)
      fprintf(stderr, "%02x", expected[i]);
   fprintf(stderr, "\n");
   return false;
}

static int
test_shuffle_vectors(void)
{
   static const uint8_t reg_b[] = {
      0x47, 0x04, 0x54, 0x04, 0x03, 0x00, 0x08, 0x2c, 0x1a, 0x00,
   };
   static const uint8_t reg_c[] = {
      0x27, 0x80, 0x01, 0x1a, 0x28, 0x00, 0xa8, 0x0a, 0x08, 0x81,
   };
   static const uint8_t uniform_b[] = {
      0x47, 0x04, 0x54, 0x00, 0x02, 0x00, 0x10, 0x2c, 0x0c, 0x00,
   };
   static const uint8_t uniform_c[] = {
      0x27, 0x00, 0x21, 0x1a, 0x20, 0x00, 0xa8, 0x02, 0x18, 0x02,
   };
   static const uint8_t quad_broadcast_wait6[4][10] = {
      {0x47, 0x00, 0x56, 0x00, 0x02, 0x0c, 0x00, 0x2c, 0x02, 0x00},
      {0x47, 0x00, 0x56, 0x00, 0x02, 0x0c, 0x02, 0x2c, 0x02, 0x00},
      {0x47, 0x00, 0x56, 0x00, 0x02, 0x0c, 0x04, 0x2c, 0x02, 0x00},
      {0x47, 0x00, 0x56, 0x00, 0x02, 0x0c, 0x06, 0x2c, 0x02, 0x00},
   };
   static const uint8_t quad_xor_lane1_live[] = {
      0xc7, 0x00, 0x54, 0x02, 0x02, 0x0c, 0x02, 0x2c, 0x02, 0x00,
   };
   static const uint8_t quad_xor_lane1_cached[] = {
      0xc7, 0x00, 0x54, 0x02, 0x03, 0x0c, 0x02, 0x2c, 0x02, 0x00,
   };
   static const uint8_t quad_xor_lane3_final[] = {
      0xc7, 0x00, 0x54, 0x06, 0x02, 0x0c, 0x06, 0x2c, 0x04, 0x00,
   };
   uint8_t out[AGX3_SUBGROUP_MAX_LENGTH], length = 0;

   struct agx3_subgroup_shuffle_desc d = {
      .op = AGX3_SUBGROUP_SHUFFLE_BROADCAST,
      .lane_kind = AGX3_SUBGROUP_LANE_REGISTER,
      .dest = 4,
      .src = 0,
      .lane = 4,
   };
   CHECK(
      agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &d, out, &length));
   CHECK(bytes_equal(out, length, reg_b, sizeof(reg_b)));
   /* G15G and G16G independently produce this same AGX3b word. */
   CHECK(
      agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &d, out, &length));
   CHECK(bytes_equal(out, length, reg_b, sizeof(reg_b)));
   CHECK(
      agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3C, &d, out, &length));
   CHECK(bytes_equal(out, length, reg_c, sizeof(reg_c)));

   d.dest = d.src = 0;
   d.lane = 8;
   d.lane_kind = AGX3_SUBGROUP_LANE_UNIFORM;
   CHECK(
      agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &d, out, &length));
   CHECK(bytes_equal(out, length, uniform_b, sizeof(uniform_b)));
   CHECK(
      agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3C, &d, out, &length));
   CHECK(bytes_equal(out, length, uniform_c, sizeof(uniform_c)));

   d = (struct agx3_subgroup_shuffle_desc){
      .quad = true,
      .op = AGX3_SUBGROUP_SHUFFLE_BROADCAST,
      .lane_kind = AGX3_SUBGROUP_LANE_IMMEDIATE,
      .src = 6,
      .lane = 2,
      .wait_mask = 1u << 5,
      .src_live = true,
   };
   for (unsigned lane = 0; lane < 4; ++lane) {
      d.lane = lane;
      CHECK(agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &d, out,
                                         &length));
      CHECK(bytes_equal(out, length, quad_broadcast_wait6[lane],
                        sizeof(quad_broadcast_wait6[lane])));
   }

   d.op = AGX3_SUBGROUP_SHUFFLE_XOR;
   d.dest = 2;
   d.lane = 1;
   d.wait_mask = 0;
   CHECK(agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &d, out,
                                       &length));
   CHECK(bytes_equal(out, length, quad_xor_lane1_live,
                     sizeof(quad_xor_lane1_live)));

   /* Fresh Apple G17P quad output retains f11 when its result remains live. */
   d.dest_hint = true;
   CHECK(agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &d, out,
                                      &length));
   CHECK(bytes_equal(out, length, quad_xor_lane1_cached,
                     sizeof(quad_xor_lane1_cached)));
   d.dest_hint = false;

   d.dest = 6;
   d.lane = 3;
   d.src_live = false;
   CHECK(agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &d, out,
                                       &length));
   CHECK(bytes_equal(out, length, quad_xor_lane3_final,
                     sizeof(quad_xor_lane3_final)));

   d.lane = 8;
   for (unsigned quad = 0; quad < 2; ++quad) {
      d.quad = quad;
      for (unsigned op = 0; op < AGX3_SUBGROUP_SHUFFLE_NUM; ++op) {
         d.op = op;
         for (unsigned kind = AGX3_SUBGROUP_LANE_REGISTER;
              kind <= AGX3_SUBGROUP_LANE_UNIFORM; ++kind) {
            d.lane_kind = kind;
            CHECK(agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &d, out,
                                               &length));
            CHECK(agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3C, &d, out,
                                               &length));
         }
      }
   }

   /*
    * Apple's own applegpu_g17p words for the immediate-lane family, one
    * kernel per lane (.tmp/g17p-rtgaps-evidence/{bcast,shuf,quadud}.metal).
    * The lane is f29 and nothing else moves with it, on all six mnemonics --
    * which is why `subgroupBroadcast(v, 31)`, the read that ends every
    * subgroup scan, needs no materialising mov any more.
    */
   static const struct {
      bool quad;
      enum agx3_subgroup_shuffle_op op;
      uint16_t dest, src, lane;
      bool src_live, dest_hint;
      uint8_t length;
      uint8_t bytes[10];
   } immediate[] = {
      {false, AGX3_SUBGROUP_SHUFFLE_BROADCAST, 0, 0, 0, false, false, 10,
       {0x47, 0x04, 0x56, 0x00, 0x02, 0x00, 0x00, 0x2c, 0x04, 0x00}},
      {false, AGX3_SUBGROUP_SHUFFLE_BROADCAST, 0, 0, 31, false, false, 10,
       {0x47, 0x04, 0x56, 0x00, 0x02, 0x00, 0x3e, 0x2c, 0x04, 0x00}},
      {false, AGX3_SUBGROUP_SHUFFLE_BROADCAST, 0, 0, 63, false, false, 10,
       {0x47, 0x04, 0x56, 0x00, 0x02, 0x00, 0x7e, 0x2c, 0x04, 0x00}},
      {false, AGX3_SUBGROUP_SHUFFLE_XOR, 0, 0, 63, false, false, 10,
       {0xc7, 0x04, 0x56, 0x00, 0x02, 0x00, 0x7e, 0x2c, 0x04, 0x00}},
      {false, AGX3_SUBGROUP_SHUFFLE_UP, 0, 0, 0, false, false, 10,
       {0x47, 0x05, 0x56, 0x00, 0x02, 0x00, 0x00, 0x2c, 0x04, 0x00}},
      {false, AGX3_SUBGROUP_SHUFFLE_UP, 4, 0, 31, true, true, 10,
       {0x47, 0x05, 0x56, 0x04, 0x03, 0x00, 0x3e, 0x2c, 0x02, 0x00}},
      {false, AGX3_SUBGROUP_SHUFFLE_DOWN, 4, 0, 31, true, true, 10,
       {0xc7, 0x05, 0x56, 0x04, 0x03, 0x00, 0x3e, 0x2c, 0x02, 0x00}},
      {true, AGX3_SUBGROUP_SHUFFLE_BROADCAST, 0, 0, 3, false, false, 10,
       {0x47, 0x00, 0x56, 0x00, 0x02, 0x00, 0x06, 0x2c, 0x04, 0x00}},
      {true, AGX3_SUBGROUP_SHUFFLE_XOR, 0, 0, 3, false, false, 10,
       {0xc7, 0x00, 0x56, 0x00, 0x02, 0x00, 0x06, 0x2c, 0x04, 0x00}},
      {true, AGX3_SUBGROUP_SHUFFLE_UP, 4, 0, 3, true, true, 10,
       {0x47, 0x01, 0x56, 0x04, 0x03, 0x00, 0x06, 0x2c, 0x02, 0x00}},
      {true, AGX3_SUBGROUP_SHUFFLE_DOWN, 4, 0, 3, true, true, 10,
       {0xc7, 0x01, 0x56, 0x04, 0x03, 0x00, 0x06, 0x2c, 0x02, 0x00}},
   };

   for (unsigned i = 0; i < ARRAY_SIZE(immediate); ++i) {
      const struct agx3_subgroup_shuffle_desc w = {
         .quad = immediate[i].quad,
         .op = immediate[i].op,
         .lane_kind = AGX3_SUBGROUP_LANE_IMMEDIATE,
         .dest = immediate[i].dest,
         .src = immediate[i].src,
         .lane = immediate[i].lane,
         .wait_mask = 1u << 5,
         .dest_hint = immediate[i].dest_hint,
         .src_live = immediate[i].src_live,
      };
      CHECK(agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &w, out,
                                         &length));
      CHECK(bytes_equal(out, length, immediate[i].bytes,
                        immediate[i].length));
   }
   return 0;
}

static int
test_ballot_vectors(void)
{
   static const uint8_t int_b[] = {
      0x17, 0x07, 0x54, 0x00, 0x02, 0x08, 0x00, 0xf8, 0xa2, 0x02,
   };
   static const uint8_t int_c[] = {
      0x27, 0x00, 0x00, 0x3a, 0x25, 0x00, 0xae, 0x02, 0x08, 0x80, 0x20, 0x04,
   };
   static const uint8_t float_b[] = {
      0x97, 0x07, 0x54, 0x00, 0x02, 0x00, 0x00, 0xa8, 0x89, 0x00, 0x00, 0x00,
   };
   static const uint8_t float_c[] = {
      0x27, 0x00, 0x00, 0x1a, 0x25, 0x00, 0xaf, 0x02, 0x30, 0x80, 0x02, 0x00,
   };
   static const uint8_t int_lt_b[] = {
      0x17, 0x07, 0x54, 0x00, 0x02, 0x00, 0x00, 0x58, 0x22, 0x06,
   };
   static const uint8_t int_lt_c[] = {
      0x27, 0x00, 0x40, 0x1a, 0x25, 0x00, 0xae, 0x02, 0x38, 0x00, 0x00, 0x04,
   };
   static const uint8_t quad_int_ne_b[] = {
      0x17, 0x02, 0x54, 0x00, 0x03, 0x08, 0x00, 0xf8, 0xa2, 0x12,
   };
   static const uint8_t quad_int_ne_c[] = {
      0x27, 0x00, 0x00, 0x3a, 0x2d, 0x00, 0xa6, 0x02, 0x08, 0x80, 0x28, 0x04,
   };
   static const uint8_t float_lt_b[] = {
      0x97, 0x07, 0x54, 0x00, 0x02, 0x00, 0x00, 0xa8, 0x89, 0x80, 0x00, 0x00,
   };
   static const uint8_t float_lt_c[] = {
      0x27, 0x00, 0x00, 0x1a, 0x25, 0x00, 0xaf, 0x02, 0x30, 0x80, 0x02, 0x40,
   };
   /* applegpu-nt 32023.917, apple-g17p, simd_ballot(tid < 16/5). */
   static const uint8_t uint_lt16_live_b[] = {
      0x17, 0x17, 0x54, 0x00, 0x02, 0x04, 0x40, 0x18, 0x22, 0x0c,
   };
   static const uint8_t uint_lt16_cached_b[] = {
      0x17, 0x17, 0x54, 0x00, 0x03, 0x04, 0x40, 0x18, 0x22, 0x0c,
   };
   static const uint8_t uint_lt5_live_b[] = {
      0x17, 0x17, 0x54, 0x00, 0x02, 0x04, 0x14, 0x18, 0x22, 0x0c,
   };
   static const uint8_t uint_lt16_dead_b[] = {
      0x17, 0x17, 0x54, 0x00, 0x02, 0x00, 0x40, 0x58, 0x22, 0x0c,
   };
   /* Apple Metal/applegpu-nt 32023.917, apple-g17p.  The compiler
    * canonicalizes tid >= 16 to the native unsigned tid > 15 form. */
   static const uint8_t uint_gt15_live_b[] = {
      0x17, 0x17, 0x54, 0x00, 0x02, 0x04, 0x3c, 0x18, 0x22, 0x0a,
   };
   static const uint8_t uint_eq0_live_b[] = {
      0x17, 0x17, 0x54, 0x00, 0x02, 0x04, 0x00, 0x18, 0x22, 0x02,
   };
   uint8_t out[AGX3_SUBGROUP_MAX_LENGTH], length = 0;
   struct agx3_subgroup_ballot_desc d = {
      .type = AGX3_SUBGROUP_BALLOT_INTEGER,
      .src = 4,
      .condition = 4,
   };
   CHECK(agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3B, &d, out, &length));
   CHECK(bytes_equal(out, length, int_b, sizeof(int_b)));
   CHECK(agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3C, &d, out, &length));
   CHECK(bytes_equal(out, length, int_c, sizeof(int_c)));

   d.dest = 4;
   CHECK(agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3B, &d, out, &length));
   d.dest = 0;

   d.src = 0;
   d.condition = 5;
   CHECK(agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3B, &d, out, &length));
   CHECK(bytes_equal(out, length, int_lt_b, sizeof(int_lt_b)));
   CHECK(agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3C, &d, out, &length));
   CHECK(bytes_equal(out, length, int_lt_c, sizeof(int_lt_c)));

   d.quad = true;
   d.src = 4;
   d.condition = 4;
   d.invert = true;
   CHECK(agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3B, &d, out, &length));
   CHECK(bytes_equal(out, length, quad_int_ne_b, sizeof(quad_int_ne_b)));
   CHECK(agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3C, &d, out, &length));
   CHECK(bytes_equal(out, length, quad_int_ne_c, sizeof(quad_int_ne_c)));

   d.type = AGX3_SUBGROUP_BALLOT_FLOAT;
   d.quad = false;
   d.src = 0;
   d.condition = 0;
   d.invert = false;
   CHECK(agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3B, &d, out, &length));
   CHECK(bytes_equal(out, length, float_b, sizeof(float_b)));
   CHECK(agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3C, &d, out, &length));
   CHECK(bytes_equal(out, length, float_c, sizeof(float_c)));

   d.condition = 1;
   CHECK(agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3B, &d, out, &length));
   CHECK(bytes_equal(out, length, float_lt_b, sizeof(float_lt_b)));
   CHECK(agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3C, &d, out, &length));
   CHECK(bytes_equal(out, length, float_lt_c, sizeof(float_lt_c)));

   d = (struct agx3_subgroup_ballot_desc) {
      .type = AGX3_SUBGROUP_BALLOT_INTEGER,
      .src = 2,
      .condition = 1,
      .compare_immediate = true,
      .compare_value = 16,
      .src_live = true,
   };
   CHECK(agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3B, &d, out, &length));
   CHECK(bytes_equal(out, length, uint_lt16_live_b,
                     sizeof(uint_lt16_live_b)));
   d.dest_cache = true;
   CHECK(agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3B, &d, out, &length));
   CHECK(bytes_equal(out, length, uint_lt16_cached_b,
                     sizeof(uint_lt16_cached_b)));
   d.dest_cache = false;
   d.compare_value = 5;
   CHECK(agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3B, &d, out, &length));
   CHECK(bytes_equal(out, length, uint_lt5_live_b,
                     sizeof(uint_lt5_live_b)));
   d.src = 0;
   d.compare_value = 16;
   d.src_live = false;
   CHECK(agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3B, &d, out, &length));
   CHECK(bytes_equal(out, length, uint_lt16_dead_b,
                     sizeof(uint_lt16_dead_b)));
   CHECK(!agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3C, &d, out,
                                      &length));

   d = (struct agx3_subgroup_ballot_desc) {
      .type = AGX3_SUBGROUP_BALLOT_INTEGER,
      .src = 2,
      .condition = 2,
      .compare_immediate = true,
      .compare_value = 15,
      .src_live = true,
   };
   CHECK(agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3B, &d, out,
                                     &length));
   CHECK(bytes_equal(out, length, uint_gt15_live_b,
                     sizeof(uint_gt15_live_b)));

   d.condition = 4;
   d.compare_value = 0;
   CHECK(agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3B, &d, out,
                                     &length));
   CHECK(bytes_equal(out, length, uint_eq0_live_b,
                     sizeof(uint_eq0_live_b)));

   /* metal-nt applegpu_g17p simd_broadcast_first(): the 16-bit active thread
    * index in h4 compared equal to zero, discarded, into a cached r2
    * (half 4). */
   static const uint8_t half_eq0_b[] = {0x17, 0x17, 0x54, 0x04, 0x03,
                                        0x08, 0x00, 0x48, 0x22, 0x02};
   d = (struct agx3_subgroup_ballot_desc) {
      .type = AGX3_SUBGROUP_BALLOT_INTEGER,
      .dest = 4,
      .src = 4,
      .condition = 4,
      .compare_immediate = true,
      .src_half = true,
      .dest_cache = true,
   };
   CHECK(agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3B, &d, out,
                                     &length));
   CHECK(bytes_equal(out, length, half_eq0_b, sizeof(half_eq0_b)));
   d.compare_value = 1;
   CHECK(!agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3B, &d, out,
                                      &length));
   d.compare_value = 0;
   d.condition = 1;
   CHECK(!agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3B, &d, out,
                                      &length));
   d.condition = 4;
   d.compare_immediate = false;
   CHECK(!agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3B, &d, out,
                                      &length));

   d = (struct agx3_subgroup_ballot_desc) {
      .type = AGX3_SUBGROUP_BALLOT_INTEGER,
      .src = 2,
      .condition = 2,
      .compare_immediate = true,
      .compare_value = 15,
      .src_live = true,
   };

   /* No immediate signed, inverted, quad, or AGX3c form was witnessed. */
   d.condition = 5;
   CHECK(!agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3B, &d, out,
                                      &length));
   d.condition = 2;
   d.invert = true;
   CHECK(!agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3B, &d, out,
                                      &length));
   d.invert = false;
   d.quad = true;
   CHECK(!agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3B, &d, out,
                                      &length));
   d.quad = false;
   CHECK(!agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3C, &d, out,
                                      &length));
   return 0;
}

static int
test_reduce_vectors(void)
{
   static const uint8_t fadd_b[] = {
      0x3f, 0x06, 0x54, 0x00, 0x02, 0x00, 0x14, 0x12,
   };
   static const uint8_t fadd_c[] = {
      0x27, 0x00, 0x04, 0x08, 0x27, 0x80, 0xac, 0x02, 0x08, 0x02,
   };
   static const uint8_t qfadd_b[] = {
      0x37, 0x06, 0x54, 0x00, 0x02, 0x00, 0x14, 0x12,
   };
   static const uint8_t qfadd_c[] = {
      0x27, 0x00, 0x00, 0x08, 0x27, 0x80, 0xac, 0x02, 0x08, 0x02,
   };
   static const uint8_t prefix_iadd_b[] = {
      0xbf, 0x01, 0x54, 0x00, 0x02, 0x00, 0x14, 0x0b,
   };
   static const uint8_t prefix_iadd_c[] = {
      0x27, 0x00, 0x04, 0x0a, 0x27, 0x80, 0xa3, 0x02, 0x00, 0x00,
   };
   static const uint8_t prefix_fadd_b[] = {
      0x3f, 0x06, 0x54, 0x00, 0x02, 0x00, 0x14, 0x32,
   };
   static const uint8_t prefix_fadd_c[] = {
      0x27, 0x00, 0x04, 0x08, 0x27, 0x80, 0xac, 0x02, 0x08, 0x00,
   };
   static const uint8_t and_b[] = {
      0x3f, 0x00, 0x54, 0x00, 0x02, 0x00, 0x14, 0x03,
   };
   static const uint8_t and_c[] = {
      0x27, 0x00, 0x04, 0x0a, 0x27, 0x80, 0xa0, 0x02, 0x00, 0x02,
   };
   static const uint8_t smin_b[] = {
      0x3f, 0x02, 0x54, 0x00, 0x02, 0x00, 0x14, 0x07,
   };
   static const uint8_t smin_c[] = {
      0x27, 0x00, 0x04, 0x0a, 0x27, 0x80, 0xa4, 0x02, 0x40, 0x02,
   };
   static const uint8_t smin_wait6_b[] = {
      0x3f, 0x02, 0x56, 0x00, 0x02, 0x00, 0x14, 0x07,
   };
   static const uint8_t smax_wait6_b[] = {
      0xbf, 0x02, 0x56, 0x00, 0x02, 0x00, 0x14, 0x07,
   };
   static const uint8_t qiadd_live_wait6_b[] = {
      0xb7, 0x01, 0x56, 0x00, 0x02, 0x0c, 0x16, 0x01,
   };
   static const uint8_t qumin_live_b[] = {
      0x37, 0x03, 0x54, 0x02, 0x02, 0x0c, 0x16, 0x01,
   };
   static const uint8_t qumax_live_b[] = {
      0xb7, 0x03, 0x54, 0x04, 0x02, 0x0c, 0x16, 0x01,
   };
   static const uint8_t qsmin_live_wait6_b[] = {
      0x37, 0x02, 0x56, 0x00, 0x02, 0x04, 0x16, 0x05,
   };
   static const uint8_t qsmax_final_b[] = {
      0xb7, 0x02, 0x54, 0x02, 0x02, 0x04, 0x14, 0x07,
   };
   static const uint8_t qand_live_wait6_b[] = {
      0x37, 0x00, 0x56, 0x00, 0x02, 0x08, 0x16, 0x01,
   };
   static const uint8_t qor_live_b[] = {
      0xb7, 0x00, 0x54, 0x02, 0x02, 0x08, 0x16, 0x01,
   };
   static const uint8_t qxor_final_b[] = {
      0x37, 0x01, 0x54, 0x04, 0x02, 0x08, 0x14, 0x03,
   };
   static const uint8_t qfadd_live_wait6_b[] = {
      0x37, 0x06, 0x56, 0x00, 0x02, 0x08, 0x16, 0x02,
   };
   static const uint8_t qfmin_live_b[] = {
      0xb7, 0x05, 0x54, 0x02, 0x02, 0x08, 0x0a, 0x02,
   };
   static const uint8_t qfmax_final_b[] = {
      0xb7, 0x07, 0x54, 0x04, 0x02, 0x08, 0x08, 0x12,
   };
   uint8_t out[AGX3_SUBGROUP_MAX_LENGTH], length = 0;
   struct agx3_subgroup_reduce_desc d = {
      .op = AGX3_SUBGROUP_REDUCE_FADD,
   };
   CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &d, out, &length));
   CHECK(bytes_equal(out, length, fadd_b, sizeof(fadd_b)));
   CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3C, &d, out, &length));
   CHECK(bytes_equal(out, length, fadd_c, sizeof(fadd_c)));

   d.prefix = true;
   CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &d, out, &length));
   CHECK(bytes_equal(out, length, prefix_fadd_b, sizeof(prefix_fadd_b)));
   CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3C, &d, out, &length));
   CHECK(bytes_equal(out, length, prefix_fadd_c, sizeof(prefix_fadd_c)));
   d.prefix = false;

   d.quad = true;
   CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &d, out,
                                      &length));
   CHECK(bytes_equal(out, length, qfadd_b, sizeof(qfadd_b)));
   CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3C, &d, out,
                                      &length));
   CHECK(bytes_equal(out, length, qfadd_c, sizeof(qfadd_c)));
   d.quad = false;

   for (unsigned op = 0; op < AGX3_SUBGROUP_REDUCE_NUM; ++op) {
      d.op = op;
      CHECK(
         agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &d, out, &length));
      CHECK(
         agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3C, &d, out, &length));
   }

   d.op = AGX3_SUBGROUP_REDUCE_AND;
   CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &d, out, &length));
   CHECK(bytes_equal(out, length, and_b, sizeof(and_b)));
   CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3C, &d, out, &length));
   CHECK(bytes_equal(out, length, and_c, sizeof(and_c)));

   d.op = AGX3_SUBGROUP_REDUCE_SMIN;
   CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &d, out, &length));
   CHECK(bytes_equal(out, length, smin_b, sizeof(smin_b)));
   CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3C, &d, out, &length));
   CHECK(bytes_equal(out, length, smin_c, sizeof(smin_c)));

   d.wait_mask = 1u << 5;
   CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &d, out, &length));
   CHECK(bytes_equal(out, length, smin_wait6_b, sizeof(smin_wait6_b)));
   CHECK(!agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3C, &d, out,
                                       &length));

   d.op = AGX3_SUBGROUP_REDUCE_SMAX;
   CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &d, out, &length));
   CHECK(bytes_equal(out, length, smax_wait6_b, sizeof(smax_wait6_b)));
   CHECK(!agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3C, &d, out,
                                       &length));
   d.wait_mask = 0;

   d.op = AGX3_SUBGROUP_REDUCE_IADD;
   d.prefix = true;
   CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &d, out, &length));
   CHECK(bytes_equal(out, length, prefix_iadd_b, sizeof(prefix_iadd_b)));
   CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3C, &d, out, &length));
   CHECK(bytes_equal(out, length, prefix_iadd_c, sizeof(prefix_iadd_c)));

   d = (struct agx3_subgroup_reduce_desc){
      .quad = true,
      .op = AGX3_SUBGROUP_REDUCE_IADD,
      .src = 6,
      .wait_mask = 1u << 5,
      .src_live = true,
   };
   CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &d, out,
                                      &length));
   CHECK(bytes_equal(out, length, qiadd_live_wait6_b,
                     sizeof(qiadd_live_wait6_b)));

   d.op = AGX3_SUBGROUP_REDUCE_UMIN;
   d.dest = 2;
   d.wait_mask = 0;
   CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &d, out,
                                      &length));
   CHECK(bytes_equal(out, length, qumin_live_b, sizeof(qumin_live_b)));

   d.op = AGX3_SUBGROUP_REDUCE_UMAX;
   d.dest = 4;
   CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &d, out,
                                      &length));
   CHECK(bytes_equal(out, length, qumax_live_b, sizeof(qumax_live_b)));

   d.op = AGX3_SUBGROUP_REDUCE_SMIN;
   d.dest = 0;
   d.src = 2;
   d.wait_mask = 1u << 5;
   CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &d, out,
                                      &length));
   CHECK(bytes_equal(out, length, qsmin_live_wait6_b,
                     sizeof(qsmin_live_wait6_b)));

   d.op = AGX3_SUBGROUP_REDUCE_SMAX;
   d.dest = 2;
   d.wait_mask = 0;
   d.src_live = false;
   CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &d, out,
                                      &length));
   CHECK(bytes_equal(out, length, qsmax_final_b, sizeof(qsmax_final_b)));
   CHECK(!agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3C, &d, out,
                                       &length));

   d = (struct agx3_subgroup_reduce_desc){
      .quad = true,
      .op = AGX3_SUBGROUP_REDUCE_AND,
      .src = 4,
      .wait_mask = 1u << 5,
      .src_live = true,
   };
   CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &d, out,
                                      &length));
   CHECK(bytes_equal(out, length, qand_live_wait6_b,
                     sizeof(qand_live_wait6_b)));

   d.op = AGX3_SUBGROUP_REDUCE_OR;
   d.dest = 2;
   d.wait_mask = 0;
   CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &d, out,
                                      &length));
   CHECK(bytes_equal(out, length, qor_live_b, sizeof(qor_live_b)));

   d.op = AGX3_SUBGROUP_REDUCE_XOR;
   d.dest = 4;
   d.src_live = false;
   CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &d, out,
                                      &length));
   CHECK(bytes_equal(out, length, qxor_final_b, sizeof(qxor_final_b)));
   CHECK(!agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3C, &d, out,
                                       &length));

   d = (struct agx3_subgroup_reduce_desc){
      .quad = true,
      .op = AGX3_SUBGROUP_REDUCE_FADD,
      .src = 4,
      .wait_mask = 1u << 5,
      .src_live = true,
   };
   CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &d, out,
                                      &length));
   CHECK(bytes_equal(out, length, qfadd_live_wait6_b,
                     sizeof(qfadd_live_wait6_b)));

   d.op = AGX3_SUBGROUP_REDUCE_FMIN;
   d.dest = 2;
   d.wait_mask = 0;
   CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &d, out,
                                      &length));
   CHECK(bytes_equal(out, length, qfmin_live_b, sizeof(qfmin_live_b)));

   d.op = AGX3_SUBGROUP_REDUCE_FMAX;
   d.dest = 4;
   d.src_live = false;
   CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &d, out,
                                      &length));
   CHECK(bytes_equal(out, length, qfmax_final_b, sizeof(qfmax_final_b)));
   return 0;
}

static int
test_rejection(void)
{
   uint8_t out[AGX3_SUBGROUP_MAX_LENGTH], length = 0;
   struct agx3_subgroup_shuffle_desc s = {
      .op = AGX3_SUBGROUP_SHUFFLE_XOR,
      .lane_kind = AGX3_SUBGROUP_LANE_REGISTER,
      .dest = 0,
      .src = 2,
      .lane = 4,
   };
   CHECK(
      !agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX2, &s, out, &length));
   s.wait_mask = 1;
   CHECK(
      !agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &s, out, &length));
   s.wait_mask = 0;
   /* An immediate lane is now the whole seven-bit field on every one of the
    * six mnemonics, so the shapes this used to refuse all encode. What still
    * fails closed is a lane the field cannot hold, a wait outside the six
    * witnessed slots, and the whole immediate form on AGX3c. */
   s.lane_kind = AGX3_SUBGROUP_LANE_IMMEDIATE;
   s.lane = 0x80;
   CHECK(
      !agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &s, out, &length));
   s.lane = 0x7f;
   CHECK(
      agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &s, out, &length));
   CHECK(
      !agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3C, &s, out, &length));
   s.wait_mask = 1u << 6;
   CHECK(
      !agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &s, out, &length));
   s.wait_mask = 0;
   s.quad = false;
   s.lane_kind = AGX3_SUBGROUP_LANE_REGISTER;
   s.lane = 3;
   CHECK(
      !agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &s, out, &length));
   s.lane = 4;
   s.op = (enum agx3_subgroup_shuffle_op)-1;
   CHECK(
      !agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &s, out, &length));

   struct agx3_subgroup_ballot_desc b = {
      .type = AGX3_SUBGROUP_BALLOT_INTEGER,
      .condition = 0,
   };
   CHECK(
      !agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3B, &b, out, &length));
   b.condition = 4;
   b.wait_mask = 1;
   CHECK(
      !agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3B, &b, out, &length));
   b.wait_mask = 0;
   b.type = (enum agx3_subgroup_ballot_type)-1;
   CHECK(
      !agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3B, &b, out, &length));

   struct agx3_subgroup_reduce_desc r = {
      .quad = true,
      .op = AGX3_SUBGROUP_REDUCE_AND,
   };
   CHECK(
      !agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3C, &r, out, &length));
   r.prefix = true;
   CHECK(
      !agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &r, out, &length));
   r.quad = false;
   CHECK(
      !agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &r, out, &length));
   r.prefix = false;
   r.op = (enum agx3_subgroup_reduce_op)-1;
   CHECK(
      !agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &r, out, &length));
   return 0;
}

int
main(void)
{
   CHECK(test_shuffle_vectors() == 0);
   CHECK(test_ballot_vectors() == 0);
   CHECK(test_reduce_vectors() == 0);
   CHECK(test_rejection() == 0);
   return 0;
}
