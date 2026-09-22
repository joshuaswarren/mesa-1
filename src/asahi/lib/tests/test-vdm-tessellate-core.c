/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "../agx_vdm_tessellate_core.h"

#include <stdio.h>
#include <string.h>

static unsigned checks;

#define CHECK(condition)                                                       \
   do {                                                                        \
      ++checks;                                                                \
      if (!(condition)) {                                                      \
         fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__,  \
                 #condition);                                                  \
         return 1;                                                             \
      }                                                                        \
   } while (0)

static const enum agx_chip supported_chips[] = {
   AGX_CHIP_G15G, AGX_CHIP_G15S, AGX_CHIP_G15C, AGX_CHIP_G15D,
   AGX_CHIP_G16G, AGX_CHIP_G16S, AGX_CHIP_G16C, AGX_CHIP_G17P,
   AGX_CHIP_G17G, AGX_CHIP_G17S, AGX_CHIP_G17C,
};

static const enum agx_chip unsupported_chips[] = {
   AGX_CHIP_G13G, AGX_CHIP_G13X, AGX_CHIP_G14G, AGX_CHIP_G14X,
   AGX_CHIP_G15P, AGX_CHIP_G16P, AGX_CHIP_G17A, AGX_CHIP_UNKNOWN,
   (enum agx_chip)999,
};

static const struct agx_vdm_tessellate_direct_info witness_info = {
   .factor_buffer = UINT64_C(0x123456789abc),
   .patch_count = UINT32_C(0x10203040),
   .instance_count = UINT32_C(0x50607080),
   .base_patch = UINT32_C(0x90a0b0c0),
   .base_instance = UINT32_C(0xd0e0f001),
   .instance_stride_word = UINT32_C(0x71234),
   .factor_buffer_size_minus_one = UINT32_C(0xfedcba98),
};

static const uint8_t witness_bytes[AGX_VDM_TESSELLATE_DIRECT_LENGTH] = {
   0x34, 0x0c, 0xf0, 0xb2, 0xbc, 0x9a, 0x78, 0x56, 0x40,
   0x30, 0x20, 0x10, 0x80, 0x70, 0x60, 0x50, 0xc0, 0xb0,
   0xa0, 0x90, 0x01, 0xf0, 0xe0, 0xd0, 0x34, 0x12, 0x07,
   0x00, 0x98, 0xba, 0xdc, 0xfe, 0x12, 0x00, 0x00, 0x00,
};

static bool
info_is_zero(const struct agx_vdm_tessellate_direct_info *info)
{
   const uint8_t *raw = (const uint8_t *)info;
   for (size_t i = 0; i < sizeof(*info); ++i) {
      if (raw[i] != 0)
         return false;
   }
   return true;
}

static bool
encoded_is_zero(const struct agx_vdm_tessellate_direct *encoded)
{
   const uint8_t *raw = (const uint8_t *)encoded;
   for (size_t i = 0; i < sizeof(*encoded); ++i) {
      if (raw[i] != 0)
         return false;
   }
   return true;
}

static int
check_info(const struct agx_vdm_tessellate_direct_info *actual,
           const struct agx_vdm_tessellate_direct_info *expected)
{
   CHECK(actual->factor_buffer == expected->factor_buffer);
   CHECK(actual->patch_count == expected->patch_count);
   CHECK(actual->instance_count == expected->instance_count);
   CHECK(actual->base_patch == expected->base_patch);
   CHECK(actual->base_instance == expected->base_instance);
   CHECK(actual->instance_stride_word == expected->instance_stride_word);
   CHECK(actual->factor_buffer_size_minus_one ==
         expected->factor_buffer_size_minus_one);
   return 0;
}

static int
test_exact_witness_and_targets(void)
{
   uint8_t first_encoding[AGX_VDM_TESSELLATE_DIRECT_LENGTH] = {0};

   for (size_t i = 0; i < sizeof(supported_chips) / sizeof(supported_chips[0]);
        ++i) {
      struct agx_vdm_tessellate_direct encoded;
      struct agx_vdm_tessellate_direct_info decoded;

      CHECK(agx_vdm_tessellate_direct_target_supported(supported_chips[i]));
      CHECK(agx_vdm_tessellate_direct_pack(supported_chips[i], &witness_info,
                                            &encoded));
      CHECK(encoded.length == AGX_VDM_TESSELLATE_DIRECT_LENGTH);
      CHECK(memcmp(encoded.bytes, witness_bytes, sizeof(witness_bytes)) == 0);
      CHECK(agx_vdm_tessellate_direct_unpack(
         supported_chips[i], witness_bytes, sizeof(witness_bytes), &decoded));
      if (check_info(&decoded, &witness_info))
         return 1;

      if (i == 0)
         memcpy(first_encoding, encoded.bytes, encoded.length);
      else
         CHECK(memcmp(first_encoding, encoded.bytes, encoded.length) == 0);
   }

   /* The complete record is byte-identical across the three plugin families. */
   struct agx_vdm_tessellate_direct g15, hal300;
   CHECK(agx_vdm_tessellate_direct_pack(AGX_CHIP_G15G, &witness_info, &g15));
   CHECK(
      agx_vdm_tessellate_direct_pack(AGX_CHIP_G17S, &witness_info, &hal300));
   CHECK(memcmp(g15.bytes, hal300.bytes, g15.length) == 0);
   struct agx_vdm_tessellate_direct_info cross;
   CHECK(agx_vdm_tessellate_direct_unpack(AGX_CHIP_G17S, g15.bytes, g15.length,
                                           &cross));
   if (check_info(&cross, &witness_info))
      return 1;

   for (size_t i = 0;
        i < sizeof(unsupported_chips) / sizeof(unsupported_chips[0]); ++i) {
      struct agx_vdm_tessellate_direct encoded;
      struct agx_vdm_tessellate_direct_info decoded;
      memset(&encoded, 0xa5, sizeof(encoded));
      memset(&decoded, 0xa5, sizeof(decoded));
      CHECK(!agx_vdm_tessellate_direct_target_supported(unsupported_chips[i]));
      CHECK(!agx_vdm_tessellate_direct_pack(unsupported_chips[i],
                                             &witness_info, &encoded));
      CHECK(encoded_is_zero(&encoded));
      CHECK(!agx_vdm_tessellate_direct_unpack(
         unsupported_chips[i], witness_bytes, sizeof(witness_bytes), &decoded));
      CHECK(info_is_zero(&decoded));
   }

   return 0;
}

static int
test_domains(void)
{
   for (unsigned bit = 0; bit < 48; ++bit) {
      struct agx_vdm_tessellate_direct_info info = witness_info;
      struct agx_vdm_tessellate_direct_info decoded;
      struct agx_vdm_tessellate_direct encoded;
      info.factor_buffer = UINT64_C(1) << bit;
      CHECK(agx_vdm_tessellate_direct_pack(AGX_CHIP_G16G, &info, &encoded));
      CHECK(agx_vdm_tessellate_direct_unpack(
         AGX_CHIP_G16G, encoded.bytes, encoded.length, &decoded));
      if (check_info(&decoded, &info))
         return 1;
   }

   const uint32_t stride_values[] = {0, 1, UINT32_C(0x7fffe),
                                     UINT32_C(0x7ffff)};
   for (size_t i = 0; i < sizeof(stride_values) / sizeof(stride_values[0]);
        ++i) {
      struct agx_vdm_tessellate_direct_info info = witness_info;
      struct agx_vdm_tessellate_direct_info decoded;
      struct agx_vdm_tessellate_direct encoded;
      info.instance_stride_word = stride_values[i];
      CHECK(agx_vdm_tessellate_direct_pack(AGX_CHIP_G17C, &info, &encoded));
      CHECK(agx_vdm_tessellate_direct_unpack(
         AGX_CHIP_G17C, encoded.bytes, encoded.length, &decoded));
      if (check_info(&decoded, &info))
         return 1;
   }

   const uint32_t edge_values[] = {0, 1, UINT32_MAX};
   for (size_t i = 0; i < sizeof(edge_values) / sizeof(edge_values[0]); ++i) {
      struct agx_vdm_tessellate_direct_info info = witness_info;
      struct agx_vdm_tessellate_direct_info decoded;
      struct agx_vdm_tessellate_direct encoded;
      info.patch_count = edge_values[i];
      info.instance_count = edge_values[2 - i];
      info.base_patch = edge_values[(i + 1) % 3];
      info.base_instance = edge_values[(i + 2) % 3];
      info.factor_buffer_size_minus_one = edge_values[i];
      CHECK(agx_vdm_tessellate_direct_pack(AGX_CHIP_G15D, &info, &encoded));
      CHECK(agx_vdm_tessellate_direct_unpack(
         AGX_CHIP_G15D, encoded.bytes, encoded.length, &decoded));
      if (check_info(&decoded, &info))
         return 1;
   }

   return 0;
}

static int
test_corruption_classification(void)
{
   struct agx_vdm_tessellate_direct baseline;
   CHECK(agx_vdm_tessellate_direct_pack(AGX_CHIP_G17S, &witness_info,
                                         &baseline));

   for (unsigned bit = 0; bit < baseline.length * 8u; ++bit) {
      uint8_t changed[AGX_VDM_TESSELLATE_DIRECT_LENGTH];
      struct agx_vdm_tessellate_direct_info decoded;
      struct agx_vdm_tessellate_direct repacked;
      memcpy(changed, baseline.bytes, baseline.length);
      changed[bit / 8] ^= (uint8_t)(1u << (bit % 8));
      memset(&decoded, 0xa5, sizeof(decoded));

      const bool accepted = agx_vdm_tessellate_direct_unpack(
         AGX_CHIP_G17S, changed, sizeof(changed), &decoded);
      if (accepted) {
         CHECK(agx_vdm_tessellate_direct_pack(AGX_CHIP_G17S, &decoded,
                                               &repacked));
         CHECK(memcmp(changed, repacked.bytes, sizeof(changed)) == 0);
      } else {
         CHECK(info_is_zero(&decoded));
      }
   }

   return 0;
}

static int
test_rejection_and_clearing(void)
{
   struct agx_vdm_tessellate_direct encoded;
   CHECK(agx_vdm_tessellate_direct_pack(AGX_CHIP_G17S, &witness_info,
                                         &encoded));

   for (size_t size = 0; size < encoded.length; ++size) {
      struct agx_vdm_tessellate_direct_info decoded;
      memset(&decoded, 0xa5, sizeof(decoded));
      CHECK(!agx_vdm_tessellate_direct_unpack(AGX_CHIP_G17S, encoded.bytes,
                                               size, &decoded));
      CHECK(info_is_zero(&decoded));
   }

   uint8_t trailing[AGX_VDM_TESSELLATE_DIRECT_LENGTH + 1];
   memcpy(trailing, encoded.bytes, encoded.length);
   trailing[encoded.length] = 0xa5;
   struct agx_vdm_tessellate_direct_info decoded;
   memset(&decoded, 0xa5, sizeof(decoded));
   CHECK(!agx_vdm_tessellate_direct_unpack(AGX_CHIP_G17S, trailing,
                                            sizeof(trailing), &decoded));
   CHECK(info_is_zero(&decoded));

   uint8_t unaligned[AGX_VDM_TESSELLATE_DIRECT_LENGTH + 1];
   memcpy(&unaligned[1], encoded.bytes, encoded.length);
   CHECK(agx_vdm_tessellate_direct_unpack(AGX_CHIP_G17S, &unaligned[1],
                                           encoded.length, &decoded));
   if (check_info(&decoded, &witness_info))
      return 1;

   struct agx_vdm_tessellate_direct_info invalid = witness_info;
   invalid.factor_buffer = UINT64_C(1) << 48;
   memset(&encoded, 0xa5, sizeof(encoded));
   CHECK(!agx_vdm_tessellate_direct_pack(AGX_CHIP_G17S, &invalid, &encoded));
   CHECK(encoded_is_zero(&encoded));
   invalid = witness_info;
   invalid.instance_stride_word = UINT32_C(0x80000);
   memset(&encoded, 0xa5, sizeof(encoded));
   CHECK(!agx_vdm_tessellate_direct_pack(AGX_CHIP_G17S, &invalid, &encoded));
   CHECK(encoded_is_zero(&encoded));

   memset(&encoded, 0xa5, sizeof(encoded));
   CHECK(!agx_vdm_tessellate_direct_pack(AGX_CHIP_G17S, NULL, &encoded));
   CHECK(encoded_is_zero(&encoded));
   CHECK(!agx_vdm_tessellate_direct_pack(AGX_CHIP_G17S, &witness_info, NULL));
   CHECK(!agx_vdm_tessellate_direct_unpack(AGX_CHIP_G17S, NULL,
                                            sizeof(witness_bytes), &decoded));
   CHECK(info_is_zero(&decoded));
   CHECK(!agx_vdm_tessellate_direct_unpack(
      AGX_CHIP_G17S, witness_bytes, sizeof(witness_bytes), NULL));
   return 0;
}

/*
 * Tessellation state records captured from Metal drawPatches on the dev Mac,
 * one variable per capture, and packed identically by the G17P plugin's
 * HWTessellationContext::drawPatches.
 */
static int
test_tessellation_state_witnesses(void)
{
   const struct {
      uint32_t word1, word2;
      struct agx_vdm_tessellation_state_info info;
   } witnesses[] = {
      /* Pow2, clockwise, maximum 16, triangles, half factors, per patch */
      {0x07a20000, 0x3c00, {AGX_VDM_TESSELLATION_PARTITION_POW2, false, 16,
                            false, false, AGX_VDM_TESSELLATION_STEP_PER_PATCH,
                            false, 0x3c00}},
      {0x47a20000, 0x3c00, {AGX_VDM_TESSELLATION_PARTITION_INTEGER, false, 16,
                            false, false, AGX_VDM_TESSELLATION_STEP_PER_PATCH,
                            false, 0x3c00}},
      {0x87220000, 0x3c00,
       {AGX_VDM_TESSELLATION_PARTITION_FRACTIONAL_ODD, false, 16, false, false,
        AGX_VDM_TESSELLATION_STEP_PER_PATCH, false, 0x3c00}},
      {0xc7a20000, 0x3c00,
       {AGX_VDM_TESSELLATION_PARTITION_FRACTIONAL_EVEN, false, 16, false,
        false, AGX_VDM_TESSELLATION_STEP_PER_PATCH, false, 0x3c00}},
      {0x27a20000, 0x3c00, {AGX_VDM_TESSELLATION_PARTITION_POW2, true, 16,
                            false, false, AGX_VDM_TESSELLATION_STEP_PER_PATCH,
                            false, 0x3c00}},
      {0x1fa20000, 0x3c00, {AGX_VDM_TESSELLATION_PARTITION_POW2, false, 64,
                            false, false, AGX_VDM_TESSELLATION_STEP_PER_PATCH,
                            false, 0x3c00}},
      {0x07b20000, 0x3c00, {AGX_VDM_TESSELLATION_PARTITION_POW2, false, 16,
                            true, false, AGX_VDM_TESSELLATION_STEP_PER_PATCH,
                            false, 0x3c00}},
      {0x87320000, 0x3c00,
       {AGX_VDM_TESSELLATION_PARTITION_FRACTIONAL_ODD, false, 16, true, false,
        AGX_VDM_TESSELLATION_STEP_PER_PATCH, false, 0x3c00}},
      {0x07a20000, 0x4000, {AGX_VDM_TESSELLATION_PARTITION_POW2, false, 16,
                            false, false, AGX_VDM_TESSELLATION_STEP_PER_PATCH,
                            false, 0x4000}},
      {0x07a00000, 0x3c00, {AGX_VDM_TESSELLATION_PARTITION_POW2, false, 16,
                            false, false, AGX_VDM_TESSELLATION_STEP_CONSTANT,
                            false, 0x3c00}},
      {0x07a40000, 0x3c00,
       {AGX_VDM_TESSELLATION_PARTITION_POW2, false, 16, false, false,
        AGX_VDM_TESSELLATION_STEP_PER_INSTANCE, false, 0x3c00}},
      {0x07a60000, 0x3c00,
       {AGX_VDM_TESSELLATION_PARTITION_POW2, false, 16, false, false,
        AGX_VDM_TESSELLATION_STEP_PER_PATCH_AND_PER_INSTANCE, false, 0x3c00}},
   };

   for (unsigned i = 0; i < sizeof(witnesses) / sizeof(witnesses[0]); ++i) {
      const uint8_t expected[AGX_VDM_TESSELLATION_STATE_LENGTH] = {
         0x90, 0x00, 0x00, 0x40,
         (uint8_t)witnesses[i].word1, (uint8_t)(witnesses[i].word1 >> 8),
         (uint8_t)(witnesses[i].word1 >> 16),
         (uint8_t)(witnesses[i].word1 >> 24),
         (uint8_t)witnesses[i].word2, (uint8_t)(witnesses[i].word2 >> 8),
         0x00, 0x00,
      };
      uint8_t packed[AGX_VDM_TESSELLATION_STATE_LENGTH];
      CHECK(agx_vdm_tessellation_state_pack(AGX_CHIP_G17P, &witnesses[i].info,
                                            packed));
      CHECK(memcmp(packed, expected, sizeof(expected)) == 0);

      struct agx_vdm_tessellation_state_info decoded;
      CHECK(agx_vdm_tessellation_state_unpack(AGX_CHIP_G17P, expected,
                                              sizeof(expected), &decoded));
      CHECK(memcmp(&decoded, &witnesses[i].info, sizeof(decoded)) == 0);
   }

   struct agx_vdm_tessellation_state_info bad = witnesses[0].info;
   uint8_t packed[AGX_VDM_TESSELLATION_STATE_LENGTH];
   bad.max_factor = 65;
   CHECK(!agx_vdm_tessellation_state_pack(AGX_CHIP_G17P, &bad, packed));
   bad.max_factor = 0;
   CHECK(!agx_vdm_tessellation_state_pack(AGX_CHIP_G17P, &bad, packed));
   CHECK(!agx_vdm_tessellation_state_pack(AGX_CHIP_G13G, &witnesses[0].info,
                                          packed));

   uint8_t corrupt[AGX_VDM_TESSELLATION_STATE_LENGTH];
   CHECK(agx_vdm_tessellation_state_pack(AGX_CHIP_G17P, &witnesses[0].info,
                                         corrupt));
   struct agx_vdm_tessellation_state_info decoded;
   corrupt[6] &= (uint8_t)~0x20; /* the fixed bit 21 */
   CHECK(!agx_vdm_tessellation_state_unpack(AGX_CHIP_G17P, corrupt,
                                            sizeof(corrupt), &decoded));
   return 0;
}

/*
 * An indirect Tessellate record captured from Metal on the development Mac:
 * two patches drawn from a four-word argument buffer at 0x1000001cf00, with
 * factors in a 24-byte buffer at 0x10000018300. The G17P plugin packs the
 * same seven words.
 */
static int
test_tessellate_indirect(void)
{
   static const uint8_t witness[AGX_VDM_TESSELLATE_INDIRECT_LENGTH] = {
      0x00, 0x0c, 0x30, 0xb4, 0x00, 0x83, 0x01, 0x00, 0x00, 0x01,
      0x00, 0x00, 0x00, 0xcf, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x05, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
   };
   const struct agx_vdm_tessellate_indirect_info info = {
      .factor_buffer = UINT64_C(0x10000018300),
      .indirect_buffer = UINT64_C(0x1000001cf00),
      .instance_stride_word = 0,
      .factor_buffer_size_minus_one = 5,
   };

   struct agx_vdm_tessellate_indirect packed;
   CHECK(agx_vdm_tessellate_indirect_pack(AGX_CHIP_G17P, &info, &packed));
   CHECK(packed.length == sizeof(witness));
   CHECK(memcmp(packed.bytes, witness, sizeof(witness)) == 0);

   struct agx_vdm_tessellate_indirect_info decoded;
   CHECK(agx_vdm_tessellate_indirect_unpack(AGX_CHIP_G17P, witness,
                                            sizeof(witness), &decoded));
   CHECK(memcmp(&decoded, &info, sizeof(info)) == 0);

   /* Every split field reaches its word. */
   const struct agx_vdm_tessellate_indirect_info wide = {
      .factor_buffer = UINT64_C(0xfedcba987654),
      .indirect_buffer = UINT64_C(0xabcd12345678),
      .instance_stride_word = 0x71234,
      .factor_buffer_size_minus_one = 0x89abcdef,
   };
   CHECK(agx_vdm_tessellate_indirect_pack(AGX_CHIP_G17P, &wide, &packed));
   CHECK(agx_vdm_tessellate_indirect_unpack(AGX_CHIP_G17P, packed.bytes,
                                            packed.length, &decoded));
   CHECK(memcmp(&decoded, &wide, sizeof(wide)) == 0);

   struct agx_vdm_tessellate_indirect_info bad = info;
   bad.indirect_buffer |= 2;
   CHECK(!agx_vdm_tessellate_indirect_pack(AGX_CHIP_G17P, &bad, &packed));
   CHECK(packed.length == 0);
   bad = info;
   bad.factor_buffer = UINT64_C(1) << 48;
   CHECK(!agx_vdm_tessellate_indirect_pack(AGX_CHIP_G17P, &bad, &packed));
   bad = info;
   bad.instance_stride_word = 0x80000;
   CHECK(!agx_vdm_tessellate_indirect_pack(AGX_CHIP_G17P, &bad, &packed));
   CHECK(!agx_vdm_tessellate_indirect_pack(AGX_CHIP_G13G, &info, &packed));

   uint8_t corrupt[AGX_VDM_TESSELLATE_INDIRECT_LENGTH];
   memcpy(corrupt, witness, sizeof(corrupt));
   corrupt[12] |= 1;
   CHECK(!agx_vdm_tessellate_indirect_unpack(AGX_CHIP_G17P, corrupt,
                                             sizeof(corrupt), &decoded));
   memcpy(corrupt, witness, sizeof(corrupt));
   corrupt[10] = 1; /* above 48 bits */
   CHECK(!agx_vdm_tessellate_indirect_unpack(AGX_CHIP_G17P, corrupt,
                                             sizeof(corrupt), &decoded));
   memcpy(corrupt, witness, sizeof(corrupt));
   corrupt[3] = 0xb2;
   CHECK(!agx_vdm_tessellate_indirect_unpack(AGX_CHIP_G17P, corrupt,
                                             sizeof(corrupt), &decoded));
   CHECK(!agx_vdm_tessellate_indirect_unpack(AGX_CHIP_G17P, witness,
                                             sizeof(witness) - 4, &decoded));
   return 0;
}

int
main(void)
{
   CHECK(test_tessellation_state_witnesses() == 0);
   CHECK(test_tessellate_indirect() == 0);
   CHECK(test_exact_witness_and_targets() == 0);
   CHECK(test_domains() == 0);
   CHECK(test_corruption_classification() == 0);
   CHECK(test_rejection_and_clearing() == 0);
   printf("AGX3 VDM Tessellate core: %u checks passed\n", checks);
   return 0;
}
