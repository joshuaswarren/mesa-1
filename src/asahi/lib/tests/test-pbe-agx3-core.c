/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "../agx_pbe_agx3_core.h"

#include <stdio.h>
#include <string.h>

static unsigned checks;

#define CHECK(x)                                                               \
   do {                                                                        \
      checks++;                                                                \
      if (!(x)) {                                                              \
         fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x);                  \
         return 1;                                                             \
      }                                                                        \
   } while (0)

static const uint8_t channels[] = {
   0x00, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x21,
   0x23, 0x25, 0x26, 0x27, 0x28, 0x31, 0x32, 0x38,
};

static bool
same_fields(uint32_t generation, const struct agx_pbe_agx3_2d_fields *a,
            const struct agx_pbe_agx3_2d_fields *b)
{
   return a->layout == b->layout && a->channels == b->channels &&
          a->type == b->type && a->swizzle_r == b->swizzle_r &&
          a->swizzle_g == b->swizzle_g &&
          a->swizzle_b == b->swizzle_b &&
          a->swizzle_a == b->swizzle_a && a->width == b->width &&
          a->height == b->height && a->buffer == b->buffer &&
          a->stride_B == b->stride_B &&
          a->page_aligned_layers ==
             (generation == AGX_PBE_AGX3_GENERATION_10 && b->page_aligned_layers);
}

static int
check_exact_gen9_linear_vector(void)
{
   /* Own-MSL buffer-backed R32F storage-image capture: 64x64, base
    * 0x10000018000, 256-byte row pitch. Linear stride is rowPitch-16 at
    * descriptor bit104, sharing word1 with the shortened 40-bit address. */
   const struct agx_pbe_agx3_2d_fields fields = {
      .layout = 0, .channels = 0x21, .type = 4,
      .swizzle_r = 0, .swizzle_g = 0, .swizzle_b = 0, .swizzle_a = 0,
      .width = 64, .height = 64, .buffer = UINT64_C(0x10000018000),
      .stride_B = 256,
   };
   static const uint8_t expected[32] = {
      0x42, 0x88, 0x00, 0x3f, 0xc0, 0x0f, 0x00, 0x00,
      0x00, 0x18, 0x00, 0x00, 0x10, 0xf0, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   struct agx_pbe_agx3_descriptor packed;
   struct agx_pbe_agx3_2d_fields decoded;
   CHECK(agx_pbe_agx3_2d_pack(9, &fields, &packed));
   CHECK(memcmp(packed.bytes, expected, sizeof(expected)) == 0);
   CHECK(agx_pbe_agx3_2d_unpack(9, expected, sizeof(expected), &decoded));
   CHECK(same_fields(9, &decoded, &fields));

   struct agx_pbe_agx3_2d_fields bad = fields;
   bad.stride_B = 252;
   CHECK(!agx_pbe_agx3_2d_pack(9, &bad, &packed));
   bad = fields;
   bad.buffer = UINT64_C(1) << 44;
   CHECK(!agx_pbe_agx3_2d_pack(9, &bad, &packed));
   bad = fields;
   bad.layout = 2;
   CHECK(!agx_pbe_agx3_2d_pack(9, &bad, &packed));
   return 0;
}

static int
check_exact_generation_vectors(void)
{
   const struct agx_pbe_agx3_2d_fields fields = {
      .layout = 2,
      .channels = 0x28,
      .type = 0,
      .swizzle_r = 0,
      .swizzle_g = 1,
      .swizzle_b = 2,
      .swizzle_a = 3,
      .width = 256,
      .height = 128,
      .buffer = UINT64_C(0x0000123456789ab0),
   };
   static const uint8_t gen9[] = {
      0x22, 0x0a, 0xe4, 0xff, 0xc0, 0x1f, 0x00, 0x00,
      0xab, 0x89, 0x67, 0x45, 0x23, 0x01, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   static const uint8_t gen10[] = {
      0x22, 0x0a, 0xe4, 0xff, 0x80, 0x3f, 0x00, 0x00,
      0xab, 0x89, 0x67, 0x45, 0x23, 0x01, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   const struct {
      uint32_t generation;
      const uint8_t *expected;
   } cases[] = {
      {AGX_PBE_AGX3_GENERATION_9, gen9},
      {AGX_PBE_AGX3_GENERATION_10, gen10},
   };

   for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
      struct agx_pbe_agx3_descriptor packed;
      struct agx_pbe_agx3_2d_fields decoded;
      CHECK(agx_pbe_agx3_2d_pack(cases[i].generation, &fields, &packed));
      CHECK(packed.length == sizeof(gen9));
      CHECK(memcmp(packed.bytes, cases[i].expected, sizeof(gen9)) == 0);
      CHECK(agx_pbe_agx3_2d_unpack(cases[i].generation, packed.bytes,
                                   packed.length, &decoded));
      CHECK(same_fields(cases[i].generation, &decoded, &fields));
   }
   return 0;
}

static int
check_exact_gen9_array_vector(void)
{
   /* PBEStateEncoderGen9<HAL200> keeps the ordinary 2D words, changes the
    * dimension nibble to3, and writes layers-1/page alignment at word1
    * bits44..60. This exact two-layer vector exercises every changed field. */
   const struct agx_pbe_agx3_2d_fields fields = {
      .layout = 2,
      .channels = 0x28,
      .type = 0,
      .swizzle_r = 0,
      .swizzle_g = 1,
      .swizzle_b = 2,
      .swizzle_a = 3,
      .width = 256,
      .height = 128,
      .buffer = UINT64_C(0x0000123456789ab0),
      .page_aligned_layers = true,
   };
   static const uint8_t expected[32] = {
      0x23, 0x0a, 0xe4, 0xff, 0xc0, 0x1f, 0x00, 0x00,
      0xab, 0x89, 0x67, 0x45, 0x23, 0x11, 0x00, 0x10,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   struct agx_pbe_agx3_descriptor packed;
   CHECK(agx_pbe_agx3_2d_array_pack(9, &fields, 2, &packed));
   CHECK(packed.length == sizeof(expected));
   CHECK(memcmp(packed.bytes, expected, sizeof(expected)) == 0);

   struct agx_pbe_agx3_2d_fields bad = fields;
   bad.layout = 0;
   bad.stride_B = 1024;
   CHECK(!agx_pbe_agx3_2d_array_pack(9, &bad, 2, &packed));
   CHECK(!agx_pbe_agx3_2d_array_pack(10, &fields, 2, &packed));
   CHECK(!agx_pbe_agx3_2d_array_pack(9, &fields, 1, &packed));
   CHECK(!agx_pbe_agx3_2d_array_pack(9, &fields, 65537, &packed));
   return 0;
}

static int
check_exhaustive_format_axes(void)
{
   const uint32_t generations[] = {
      AGX_PBE_AGX3_GENERATION_9,
      AGX_PBE_AGX3_GENERATION_10,
   };

   for (unsigned g = 0; g < sizeof(generations) / sizeof(generations[0]); ++g)
      for (uint8_t layout = 1; layout <= 2; ++layout)
         for (unsigned c = 0; c < sizeof(channels) / sizeof(channels[0]); ++c)
            for (uint8_t type = 0; type <= 5; ++type)
               for (unsigned swizzle = 0; swizzle < 256; ++swizzle) {
                  const struct agx_pbe_agx3_2d_fields fields = {
                     .layout = layout,
                     .channels = channels[c],
                     .type = type,
                     .swizzle_r = swizzle & 3,
                     .swizzle_g = (swizzle >> 2) & 3,
                     .swizzle_b = (swizzle >> 4) & 3,
                     .swizzle_a = (swizzle >> 6) & 3,
                     .width = generations[g] == AGX_PBE_AGX3_GENERATION_9
                                 ? 16384
                                 : 32768,
                     .height = 1,
                     .buffer = UINT64_C(0x0000fffffffffff0),
                     .page_aligned_layers = (swizzle & 1) != 0,
                  };
                  struct agx_pbe_agx3_descriptor packed;
                  struct agx_pbe_agx3_2d_fields decoded;
                  CHECK(agx_pbe_agx3_2d_pack(generations[g], &fields,
                                             &packed));
                  CHECK(agx_pbe_agx3_2d_unpack(
                     generations[g], packed.bytes, packed.length, &decoded));
                  CHECK(same_fields(generations[g], &decoded, &fields));
               }
   return 0;
}

static int
check_dimension_and_address_bits(void)
{
   const uint32_t generations[] = {
      AGX_PBE_AGX3_GENERATION_9,
      AGX_PBE_AGX3_GENERATION_10,
   };

   for (unsigned g = 0; g < sizeof(generations) / sizeof(generations[0]); ++g) {
      const unsigned bits = generations[g] == AGX_PBE_AGX3_GENERATION_9
                               ? 14
                               : 15;
      for (unsigned bit = 0; bit < bits; ++bit) {
         struct agx_pbe_agx3_2d_fields fields = {
            .layout = 2,
            .channels = 0x28,
            .swizzle_r = 0,
            .swizzle_g = 1,
            .swizzle_b = 2,
            .swizzle_a = 3,
            .width = (uint16_t)(UINT32_C(1) << bit),
            .height = (uint16_t)(UINT32_C(1) << bit),
            .buffer = UINT64_C(0x1000),
         };
         struct agx_pbe_agx3_descriptor packed;
         struct agx_pbe_agx3_2d_fields decoded;
         CHECK(agx_pbe_agx3_2d_pack(generations[g], &fields, &packed));
         CHECK(agx_pbe_agx3_2d_unpack(generations[g], packed.bytes,
                                      packed.length, &decoded));
         CHECK(same_fields(generations[g], &decoded, &fields));
      }

      for (unsigned bit = 4; bit < 48; ++bit) {
         struct agx_pbe_agx3_2d_fields fields = {
            .layout = 1,
            .channels = 0x00,
            .width = 1,
            .height = 1,
            .buffer = UINT64_C(1) << bit,
         };
         struct agx_pbe_agx3_descriptor packed;
         struct agx_pbe_agx3_2d_fields decoded;
         CHECK(agx_pbe_agx3_2d_pack(generations[g], &fields, &packed));
         CHECK(agx_pbe_agx3_2d_unpack(generations[g], packed.bytes,
                                      packed.length, &decoded));
         CHECK(same_fields(generations[g], &decoded, &fields));
      }
   }
   return 0;
}

static int
check_fail_closed_axes(void)
{
   struct agx_pbe_agx3_2d_fields fields = {
      .layout = 2,
      .channels = 0x28,
      .width = 1,
      .height = 1,
      .buffer = UINT64_C(0x1000),
   };
   struct agx_pbe_agx3_descriptor packed;

#define REJECT(statement_)                                                     \
   do {                                                                        \
      statement_;                                                              \
      memset(&packed, 0xa5, sizeof(packed));                                   \
      CHECK(!agx_pbe_agx3_2d_pack(AGX_PBE_AGX3_GENERATION_9, &fields,         \
                                   &packed));                                   \
      CHECK(memcmp(&packed, &(struct agx_pbe_agx3_descriptor){0},              \
                   sizeof(packed)) == 0);                                      \
   } while (0)

   REJECT(fields.layout = 0);
   REJECT(fields.layout = 3);
   fields.layout = 2;
   REJECT(fields.channels = 0x36);
   REJECT(fields.channels = 0x50);
   fields.channels = 0x28;
   REJECT(fields.type = 6);
   fields.type = 0;
   REJECT(fields.swizzle_a = 4);
   fields.swizzle_a = 0;
   REJECT(fields.width = 0);
   REJECT(fields.width = 16385);
   fields.width = 1;
   REJECT(fields.height = 0);
   REJECT(fields.height = 16385);
   fields.height = 1;
   REJECT(fields.buffer = 0);
   REJECT(fields.buffer = UINT64_C(0x1001));
   REJECT(fields.buffer = UINT64_C(0x0001000000000000));

#undef REJECT

   CHECK(!agx_pbe_agx3_2d_pack(8, &fields, &packed));
   CHECK(!agx_pbe_agx3_2d_pack(0, &fields, &packed));
   CHECK(!agx_pbe_agx3_2d_pack(AGX_PBE_AGX3_GENERATION_9, NULL, &packed));
   CHECK(!agx_pbe_agx3_2d_pack(AGX_PBE_AGX3_GENERATION_9, &fields, NULL));

   fields.buffer = UINT64_C(0x1000);
   CHECK(agx_pbe_agx3_2d_pack(AGX_PBE_AGX3_GENERATION_9, &fields, &packed));
   for (unsigned bit = 52; bit < 256; ++bit) {
      if (bit >= 64 && bit < 108)
         continue;
      uint8_t malformed[AGX_PBE_AGX3_DESCRIPTOR_LENGTH];
      struct agx_pbe_agx3_2d_fields decoded;
      memcpy(malformed, packed.bytes, sizeof(malformed));
      malformed[bit / 8] ^= (uint8_t)(1u << (bit % 8));
      CHECK(!agx_pbe_agx3_2d_unpack(AGX_PBE_AGX3_GENERATION_9, malformed,
                                    sizeof(malformed), &decoded));
      CHECK(memcmp(&decoded, &(struct agx_pbe_agx3_2d_fields){0},
                   sizeof(decoded)) == 0);
   }
   return 0;
}

static int
check_gen9_single_layer_packet(void)
{
   /* Matching Gen9 MTL80 table878fa8[79] ->818c40 supplies channels0x28,
    * type0, bytes-per-pixel4. StateFields34edf4 takes depth1/base-slice0,
    * sparsePageSize0: layout2, no high control bits or extended words.
    * This is a constructor-derived vector, not a native32x32 capture.
    */
   static const uint8_t expected[32] = {
      0x22, 0x0a, 0xe4, 0x1f, 0xc0, 0x07, 0x00, 0x00,
      0x00, 0xf4, 0xfe, 0x7f, 0x08, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   struct agx_pbe_agx3_2d_fields fields = {
      .layout = 2, .channels = 0x28, .type = 0,
      .swizzle_r = 0, .swizzle_g = 1, .swizzle_b = 2, .swizzle_a = 3,
      .width = 32, .height = 32, .buffer = UINT64_C(0x87ffef4000),
      .page_aligned_layers = true,
   };
   struct agx_pbe_agx3_descriptor packed;
   CHECK(agx_pbe_agx3_2d_pack(9, &fields, &packed));
   CHECK(memcmp(packed.bytes, expected, sizeof(expected)) == 0);

   const uint64_t addresses[] = {
      UINT64_C(0x4000), UINT64_C(0x87ffef4000), UINT64_C(0x10002000000),
   };
   for (unsigned generation = 9; generation <= 10; ++generation) {
      for (unsigned layout = 1; layout <= 2; ++layout) {
         for (unsigned i = 0; i < sizeof(addresses) / sizeof(addresses[0]); ++i) {
            fields.layout = layout;
            fields.buffer = addresses[i];
            fields.page_aligned_layers = false;
            struct agx_pbe_agx3_descriptor unpadded;
            CHECK(agx_pbe_agx3_2d_pack(generation, &fields, &unpadded));
            fields.page_aligned_layers = true;
            CHECK(agx_pbe_agx3_2d_pack(generation, &fields, &packed));
            if (generation == 10)
               unpadded.bytes[15] |= 0x10;
            CHECK(memcmp(&packed, &unpadded, sizeof(packed)) == 0);
            struct agx_pbe_agx3_2d_fields decoded;
            CHECK(agx_pbe_agx3_2d_unpack(generation, packed.bytes,
                                        packed.length, &decoded));
            CHECK(same_fields(generation, &decoded, &fields));
         }
      }
   }
   return 0;
}

static int
check_gen9_native_mips(void)
{
   const struct agx_pbe_agx3_2d_fields f = {
      .layout = 1, .channels = 0x28, .type = 0,
      .swizzle_r = 0, .swizzle_g = 1, .swizzle_b = 2, .swizzle_a = 3,
      .width = 128, .height = 128, .buffer = UINT64_C(0x87fddc0000),
   };
   /* Recovered native StateFields output for level1 of a three-level chain:
    * base dimensions/address are preserved; extension is fully explicit. */
   static const uint8_t native_level1_of3[32] = {
      0x12,0x0a,0xe4,0x7f,0xc0,0x1f,0x00,0x04,
      0x00,0xc0,0xdd,0x7f,0x08,0x00,0x00,0x80,
      0x00,0x00,0x00,0x00,0x00,0x10,0x02,0x00,
      0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
   };
   struct agx_pbe_agx3_descriptor p, plain;
   CHECK(agx_pbe_agx3_2d_mip_pack(9,&f,1,3,&p));
   CHECK(memcmp(p.bytes,native_level1_of3,32)==0);
   CHECK(agx_pbe_agx3_2d_pack(9,&f,&plain));
   CHECK(agx_pbe_agx3_2d_mip_pack(9,&f,0,1,&p));
   CHECK(memcmp(&p,&plain,sizeof(p))==0);
   for(unsigned levels=1;levels<=16;levels++) {
      for(unsigned level=0;level<levels;level++) {
         CHECK(agx_pbe_agx3_2d_mip_pack(9,&f,level,levels,&p));
         CHECK(p.length==32);
         for(unsigned i=0;i<32;i++) {
            uint8_t expect=plain.bytes[i];
            if(levels>1) {
               if(i==7)expect|=4;
               if(i==15)expect|=128;
               if(i==21)expect=level*16;
               if(i==22)expect=levels-1;
            }
            CHECK(p.bytes[i]==expect);
         }
      }
   }
   const unsigned invalid[][3]={{9,0,0},{9,0,17},{9,1,1},{9,16,16},{10,0,1},{9,UINT32_MAX,2}};
   for(unsigned i=0;i<sizeof(invalid)/sizeof(invalid[0]);i++) {
      memset(&p,0xa5,sizeof(p));
      CHECK(!agx_pbe_agx3_2d_mip_pack(invalid[i][0],&f,invalid[i][1],invalid[i][2],&p));
      CHECK(memcmp(&p,&(struct agx_pbe_agx3_descriptor){0},sizeof(p))==0);
   }
   return 0;
}

int
main(void)
{
   if (check_exact_generation_vectors() || check_exact_gen9_array_vector() ||
       check_exact_gen9_linear_vector() ||
       check_exhaustive_format_axes() ||
       check_dimension_and_address_bits() || check_fail_closed_axes() ||
       check_gen9_single_layer_packet() || check_gen9_native_mips())
      return 1;

   printf("agx_pbe_agx3_core: %u checks passed\n", checks);
   return 0;
}
