/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "asahi/compiler/agx3_subgroup_core.h"

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
   static const uint8_t quad_broadcast_lane2_wait6[] = {
      0x47, 0x00, 0x56, 0x00, 0x02, 0x0c, 0x04, 0x2c, 0x02, 0x00,
   };
   static const uint8_t quad_xor_lane1_live[] = {
      0xc7, 0x00, 0x54, 0x02, 0x02, 0x0c, 0x02, 0x2c, 0x02, 0x00,
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
   CHECK(agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &d, out,
                                       &length));
   CHECK(bytes_equal(out, length, quad_broadcast_lane2_wait6,
                     sizeof(quad_broadcast_lane2_wait6)));

   d.op = AGX3_SUBGROUP_SHUFFLE_XOR;
   d.dest = 2;
   d.lane = 1;
   d.wait_mask = 0;
   CHECK(agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &d, out,
                                       &length));
   CHECK(bytes_equal(out, length, quad_xor_lane1_live,
                     sizeof(quad_xor_lane1_live)));

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
   s.lane_kind = (enum agx3_subgroup_lane_kind)1;
   CHECK(
      !agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &s, out, &length));
   s.quad = true;
   s.op = AGX3_SUBGROUP_SHUFFLE_BROADCAST;
   s.lane = 1;
   CHECK(
      !agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &s, out, &length));
   s.op = AGX3_SUBGROUP_SHUFFLE_XOR;
   s.lane = 0;
   CHECK(
      !agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &s, out, &length));
   s.lane = 1;
   s.wait_mask = 1;
   CHECK(
      !agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &s, out, &length));
   s.wait_mask = 0;
   s.op = AGX3_SUBGROUP_SHUFFLE_UP;
   s.lane = 1;
   CHECK(
      !agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &s, out, &length));
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
      !agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &r, out, &length));
   r.quad = false;
   r.prefix = true;
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
