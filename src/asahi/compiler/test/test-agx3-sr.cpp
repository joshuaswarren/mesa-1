/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * agx3_sr_core against the Apple corpus: every clean SR-read rld witness from
 * the srops/memops/lsw probe corpora must be reproduced byte-exactly from its
 * normalized descriptor, and everything outside the witnessed domain must be
 * refused.
 */

#include <gtest/gtest.h>

#include <cstring>

#include "agx3_sr_core.h"

#include "agx3_sr_core_vectors.c"

static enum agx_isa_encoding
enc_of(int e)
{
   return e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
}

TEST(AGX3SrCore, CorpusVectors)
{
   for (size_t i = 0; i < sr_vec_count; ++i) {
      const struct sr_vec *v = &sr_vecs[i];
      struct agx3_sr_desc d = {};
      d.dest = (uint16_t)v->dest;
      d.width = (enum agx3_sr_width)v->width;
      d.sr = (uint8_t)v->sr;
      d.slot = (uint8_t)v->slot;
      d.wait_mask = (uint16_t)v->wait;

      uint8_t out[AGX3_SR_MAX_LENGTH];
      uint8_t len = 0;
      ASSERT_TRUE(agx3_sr_encode(enc_of(v->enc), &d, out, &len)) << "vec " << i;
      ASSERT_EQ(len, v->nbytes) << "vec " << i;
      ASSERT_EQ(0, memcmp(out, v->bytes, len)) << "vec " << i;
   }
}

TEST(AGX3SrCore, WitnessSpotChecks)
{
   /* thread_position_in_grid.x -> r0, slot 1: srops sr_tpig offset 0x40, the
    * identical 4 bytes on both encodings. */
   static const uint8_t tpig_x[] = {0x0c, 0xa0, 0x10, 0x06};
   /* thread_index_in_simdgroup (16-bit) -> h0, slot 1: srops sr_lane. */
   static const uint8_t lane[] = {0x04, 0x82, 0x10, 0x06};

   for (int e = 0; e < 2; ++e) {
      struct agx3_sr_desc d = {};
      d.dest = 0;
      d.width = AGX3_SR_W32;
      d.sr = 23;
      d.slot = 1;

      uint8_t out[AGX3_SR_MAX_LENGTH];
      uint8_t len = 0;
      ASSERT_TRUE(agx3_sr_encode(enc_of(e), &d, out, &len));
      ASSERT_EQ(len, sizeof(tpig_x));
      EXPECT_EQ(0, memcmp(out, tpig_x, len));

      d.width = AGX3_SR_W16;
      d.sr = 7;
      ASSERT_TRUE(agx3_sr_encode(enc_of(e), &d, out, &len));
      ASSERT_EQ(len, sizeof(lane));
      EXPECT_EQ(0, memcmp(out, lane, len));
   }
}

TEST(AGX3SrCore, HardwareVertexInvocationInputs)
{
   /* Apple's isolated hardware-vertex shaders begin with these byte-exact
    * 32-bit reads into r0. The encoder input is normalized f21, not the
    * hardware SR byte visible in the instruction. */
   static const struct {
      uint8_t normalized;
      uint8_t bytes[4];
   } cases[] = {
      {51, {0x0c, 0xdd, 0x10, 0x06}}, /* vertex_id */
      {46, {0x0c, 0xd8, 0x10, 0x06}}, /* instance_id */
   };

   for (int e = 0; e < 2; ++e) {
      for (const auto &c : cases) {
         struct agx3_sr_desc d = {};
         d.dest = 0;
         d.width = AGX3_SR_W32;
         d.sr = c.normalized;
         d.slot = 1;

         uint8_t out[AGX3_SR_MAX_LENGTH];
         uint8_t len = 0;
         ASSERT_TRUE(agx3_sr_encode(enc_of(e), &d, out, &len));
         ASSERT_EQ(len, sizeof(c.bytes));
         EXPECT_EQ(0, memcmp(out, c.bytes, len));
      }
   }
}

TEST(AGX3SrCore, FragmentCoverageInputs)
{
   /* Apple's isolated fragment probes read the per-invocation 1x coverage
    * seed from SR2 and API-visible SampleMaskIn from bits 4..7 of SR39. Both
    * are native half-register reads; neither has a witnessed 32-bit form. */
   static const struct {
      uint8_t sr;
      uint8_t bytes[4];
   } cases[] = {
      {2, {0x04, 0x84, 0x10, 0x06}},
      {39, {0x04, 0xc2, 0x10, 0x06}},
   };

   for (int e = 0; e < 2; ++e) {
      for (const auto &c : cases) {
         const struct agx3_sr_desc d = {
            .dest = 0,
            .width = AGX3_SR_W16,
            .sr = c.sr,
            .slot = 1,
         };
         uint8_t out[AGX3_SR_MAX_LENGTH];
         uint8_t len = 0;
         ASSERT_TRUE(agx3_sr_encode(enc_of(e), &d, out, &len));
         ASSERT_EQ(len, sizeof(c.bytes));
         EXPECT_EQ(0, memcmp(out, c.bytes, len));

         struct agx3_sr_desc wide = d;
         wide.width = AGX3_SR_W32;
         EXPECT_FALSE(agx3_sr_encode(enc_of(e), &wide, out, &len));
      }
   }
}

TEST(AGX3SrCore, G17PBackFacingInput)
{
   const struct agx3_sr_desc d = {
      .dest = 0,
      .width = AGX3_SR_W16,
      .sr = 41,
      .slot = 1,
   };
   static const uint8_t native[] = {0x04, 0xc5, 0x10, 0x06};
   uint8_t out[AGX3_SR_MAX_LENGTH];
   uint8_t len = 0;

   ASSERT_TRUE(agx3_sr_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
   ASSERT_EQ(len, sizeof(native));
   EXPECT_EQ(0, memcmp(out, native, len));

   struct agx3_sr_desc wide = d;
   wide.width = AGX3_SR_W32;
   EXPECT_FALSE(agx3_sr_encode(AGX_ISA_ENCODING_AGX3B, &wide, out, &len));
}

TEST(AGX3SrCore, G17PActiveThreadIndexInSubgroup)
{
   /* metal-nt -arch applegpu_g17p: simd_is_first() reads SR34 into h0 on slot
    * 1 with these bytes in compute and vertex shaders, and with f48 set in the
    * long form in fragment shaders. */
   static const uint8_t compute[] = {0x04, 0x95, 0x10, 0x06};
   static const uint8_t fragment[] = {0x04, 0x95, 0x11, 0x06,
                                      0x01, 0x00, 0x00, 0x00};
   struct agx3_sr_desc d = {};
   d.dest = 0;
   d.width = AGX3_SR_W16;
   d.sr = 34;
   d.slot = 1;
   uint8_t out[AGX3_SR_MAX_LENGTH];
   uint8_t len = 0;

   ASSERT_TRUE(agx3_sr_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
   ASSERT_EQ(len, sizeof(compute));
   EXPECT_EQ(0, memcmp(out, compute, len));

   d.fragment_active_index = true;
   ASSERT_TRUE(agx3_sr_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
   ASSERT_EQ(len, sizeof(fragment));
   EXPECT_EQ(0, memcmp(out, fragment, len));

   d.sr = 7;
   EXPECT_FALSE(agx3_sr_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
}

TEST(AGX3SrCore, WaitEscalatesToLongForm)
{
   /* A wait bit cannot ride the compact form; Apple's own at_dev_nr witness
    * (thread_position_in_grid.x, slot 1, wait slot 1) is the 8-byte form. */
   struct agx3_sr_desc d = {};
   d.dest = 0;
   d.width = AGX3_SR_W32;
   d.sr = 23;
   d.slot = 1;
   d.wait_mask = 0x1;

   for (int e = 0; e < 2; ++e) {
      uint8_t out[AGX3_SR_MAX_LENGTH];
      uint8_t len = 0;
      ASSERT_TRUE(agx3_sr_encode(enc_of(e), &d, out, &len));
      EXPECT_EQ(len, 8u);
   }
}

TEST(AGX3SrCore, FailsClosed)
{
   uint8_t out[AGX3_SR_MAX_LENGTH];
   uint8_t len = 0;

   struct agx3_sr_desc ok = {};
   ok.dest = 0;
   ok.width = AGX3_SR_W32;
   ok.sr = 23;
   ok.slot = 1;
   ASSERT_TRUE(agx3_sr_encode(AGX_ISA_ENCODING_AGX3B, &ok, out, &len));

   struct agx3_sr_desc d;

   d = ok;
   d.sr = 0; /* no witness ties SR 0 to anything */
   EXPECT_FALSE(agx3_sr_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));

   d = ok;
   d.width = AGX3_SR_W16; /* SR 23 witnessed only at 32-bit */
   EXPECT_FALSE(agx3_sr_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));

   d = ok;
   d.sr = 13;
   d.width = AGX3_SR_W32; /* SR 13 witnessed only at 16-bit */
   EXPECT_FALSE(agx3_sr_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));

   d = ok;
   d.dest = 1; /* 32-bit read of an unaligned pair */
   EXPECT_FALSE(agx3_sr_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));

   d = ok;
   d.slot = 0; /* an SR read always signals a slot */
   EXPECT_FALSE(agx3_sr_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));

   d = ok;
   d.slot = 7; /* agx3b has six slots; the same value is fine on c */
   EXPECT_FALSE(agx3_sr_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
   EXPECT_TRUE(agx3_sr_encode(AGX_ISA_ENCODING_AGX3C, &d, out, &len));

   d = ok;
   d.wait_mask = 1u << 6; /* wait on slot 7: b refuses, c encodes */
   EXPECT_FALSE(agx3_sr_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
   EXPECT_TRUE(agx3_sr_encode(AGX_ISA_ENCODING_AGX3C, &d, out, &len));
}
