/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "../d12_agx3_index_list_core.h"

#include "asahi/lib/agx_vdm_index_list_core.h"

#include <stdio.h>
#include <string.h>

static unsigned checks;

#define CHECK(x)                                                               \
   do {                                                                        \
      checks++;                                                                \
      if (!(x)) {                                                              \
         fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x);               \
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

static const uint8_t primitives[] = {0, 1, 3, 5, 6, 9, 11, 13, 14, 15};

static bool
is_hal300(enum agx_chip chip)
{
   return chip == AGX_CHIP_G17G || chip == AGX_CHIP_G17S ||
          chip == AGX_CHIP_G17C;
}

static int
check_zero_nonindexed(
   const struct d12_agx3_direct_nonindexed_index_list *encoded)
{
   for (unsigned i = 0; i < sizeof(encoded->bytes); ++i)
      CHECK(encoded->bytes[i] == 0);
   CHECK(encoded->length == 0);
   return 0;
}

static int
check_zero_indexed(const struct d12_agx3_direct_indexed_index_list *encoded)
{
   for (unsigned i = 0; i < sizeof(encoded->bytes); ++i)
      CHECK(encoded->bytes[i] == 0);
   CHECK(encoded->length == 0);
   return 0;
}

static int
check_nonindexed_info(
   const struct d12_agx3_direct_nonindexed_index_list_info *actual,
   const struct d12_agx3_direct_nonindexed_index_list_info *expected)
{
   CHECK(actual->primitive == expected->primitive);
   CHECK(actual->count == expected->count);
   CHECK(actual->instances == expected->instances);
   CHECK(actual->start == expected->start);
   return 0;
}

static int
check_indexed_info(
   const struct d12_agx3_direct_indexed_index_list_info *actual,
   const struct d12_agx3_direct_indexed_index_list_info *expected)
{
   CHECK(actual->primitive == expected->primitive);
   CHECK(actual->restart_enable == expected->restart_enable);
   CHECK(actual->index_size == expected->index_size);
   CHECK(actual->index_buffer == expected->index_buffer);
   CHECK(actual->count == expected->count);
   CHECK(actual->instances == expected->instances);
   CHECK(actual->start == expected->start);
   CHECK(actual->index_buffer_size_word == expected->index_buffer_size_word);
   return 0;
}

static int
check_zero_nonindexed_info(
   const struct d12_agx3_direct_nonindexed_index_list_info *info)
{
   const struct d12_agx3_direct_nonindexed_index_list_info zero = {0};
   return check_nonindexed_info(info, &zero);
}

static int
check_zero_indexed_info(
   const struct d12_agx3_direct_indexed_index_list_info *info)
{
   const struct d12_agx3_direct_indexed_index_list_info zero = {0};
   return check_indexed_info(info, &zero);
}

static int
check_witnesses_and_shared_equality(void)
{
   static const uint8_t expected_nonindexed_g15[] = {
      0x00, 0x06, 0xc4, 0x61, 0x44, 0x33, 0x22, 0x11,
      0x02, 0x00, 0x00, 0x00, 0xdd, 0xcc, 0xbb, 0xaa,
   };
   static const uint8_t expected_nonindexed_hal300[] = {
      0x00, 0x06, 0xc4, 0x69, 0x44, 0x33, 0x22, 0x11,
      0x02, 0x00, 0x00, 0x00, 0xdd, 0xcc, 0xbb, 0xaa,
      0x07, 0x00, 0x00, 0x00,
   };
   static const uint8_t expected_indexed_g15[] = {
      0x34, 0x09, 0xf5, 0x61, 0xbc, 0x9a, 0x78, 0x56,
      0x04, 0x03, 0x02, 0x01, 0x03, 0x00, 0x00, 0x00,
      0xf9, 0xff, 0xff, 0xff, 0x40, 0x30, 0x20, 0x10,
      0x12, 0x00, 0x00, 0x00,
   };
   static const uint8_t expected_indexed_hal300[] = {
      0x34, 0x09, 0xf5, 0x69, 0xbc, 0x9a, 0x78, 0x56,
      0x04, 0x03, 0x02, 0x01, 0x03, 0x00, 0x00, 0x00,
      0xf9, 0xff, 0xff, 0xff, 0x07, 0x00, 0x00, 0x00,
      0x40, 0x30, 0x20, 0x10, 0x12, 0x00, 0x00, 0x00,
   };
   const struct d12_agx3_direct_nonindexed_index_list_info nonindexed_info = {
      .primitive = 6,
      .count = UINT32_C(0x11223344),
      .instances = 2,
      .start = UINT32_C(0xaabbccdd),
   };
   const struct d12_agx3_direct_indexed_index_list_info indexed_info = {
      .primitive = 9,
      .restart_enable = true,
      .index_size = D12_AGX3_INDEX_SIZE_U32,
      .index_buffer = UINT64_C(0x123456789abc),
      .count = UINT32_C(0x01020304),
      .instances = 3,
      .start = UINT32_C(0xfffffff9),
      .index_buffer_size_word = UINT32_C(0x10203040),
   };

   for (unsigned i = 0; i < sizeof(supported_chips) / sizeof(supported_chips[0]);
        ++i) {
      struct d12_agx3_direct_nonindexed_index_list d12_nonindexed;
      struct d12_agx3_direct_indexed_index_list d12_indexed;
      struct d12_agx3_direct_nonindexed_index_list_info decoded_nonindexed;
      struct d12_agx3_direct_indexed_index_list_info decoded_indexed;
      struct agx_vdm_index_list_direct_nonindexed shared_nonindexed;
      struct agx_vdm_index_list_direct_indexed shared_indexed;
      const struct agx_vdm_index_list_direct_nonindexed_info shared_noninfo = {
         .primitive = nonindexed_info.primitive,
         .count = nonindexed_info.count,
         .instances = nonindexed_info.instances,
         .start = nonindexed_info.start,
      };
      const struct agx_vdm_index_list_direct_indexed_info shared_indexinfo = {
         .primitive = indexed_info.primitive,
         .restart_enable = indexed_info.restart_enable,
         .index_size = AGX_VDM_INDEX_LIST_INDEX_SIZE_U32,
         .index_buffer = indexed_info.index_buffer,
         .count = indexed_info.count,
         .instances = indexed_info.instances,
         .start = indexed_info.start,
         .index_buffer_size_word = indexed_info.index_buffer_size_word,
      };
      const uint8_t *expected_nonindexed =
         is_hal300(supported_chips[i]) ? expected_nonindexed_hal300
                                       : expected_nonindexed_g15;
      const uint8_t *expected_indexed =
         is_hal300(supported_chips[i]) ? expected_indexed_hal300
                                       : expected_indexed_g15;
      const size_t expected_nonindexed_size = is_hal300(supported_chips[i])
                                                 ? sizeof(expected_nonindexed_hal300)
                                                 : sizeof(expected_nonindexed_g15);
      const size_t expected_indexed_size = is_hal300(supported_chips[i])
                                              ? sizeof(expected_indexed_hal300)
                                              : sizeof(expected_indexed_g15);

      CHECK(d12_agx3_encode_direct_nonindexed_index_list(
         supported_chips[i], &nonindexed_info, &d12_nonindexed));
      CHECK(agx_vdm_index_list_direct_nonindexed_pack(
         supported_chips[i], &shared_noninfo, &shared_nonindexed));
      CHECK(d12_nonindexed.length == expected_nonindexed_size);
      CHECK(d12_nonindexed.length == shared_nonindexed.length);
      CHECK(memcmp(d12_nonindexed.bytes, expected_nonindexed,
                   expected_nonindexed_size) == 0);
      CHECK(memcmp(d12_nonindexed.bytes, shared_nonindexed.bytes,
                   expected_nonindexed_size) == 0);
      CHECK(d12_agx3_decode_direct_nonindexed_index_list(
         supported_chips[i], d12_nonindexed.bytes, d12_nonindexed.length,
         &decoded_nonindexed));
      if (check_nonindexed_info(&decoded_nonindexed, &nonindexed_info))
         return 1;

      CHECK(d12_agx3_encode_direct_indexed_index_list(
         supported_chips[i], &indexed_info, &d12_indexed));
      CHECK(agx_vdm_index_list_direct_indexed_pack(
         supported_chips[i], &shared_indexinfo, &shared_indexed));
      CHECK(d12_indexed.length == expected_indexed_size);
      CHECK(d12_indexed.length == shared_indexed.length);
      CHECK(memcmp(d12_indexed.bytes, expected_indexed,
                   expected_indexed_size) == 0);
      CHECK(memcmp(d12_indexed.bytes, shared_indexed.bytes,
                   expected_indexed_size) == 0);
      CHECK(d12_agx3_decode_direct_indexed_index_list(
         supported_chips[i], d12_indexed.bytes, d12_indexed.length,
         &decoded_indexed));
      if (check_indexed_info(&decoded_indexed, &indexed_info))
         return 1;
   }

   return 0;
}

static int
check_domains(void)
{
   for (unsigned c = 0; c < sizeof(supported_chips) / sizeof(supported_chips[0]);
        ++c) {
      for (unsigned p = 0; p < sizeof(primitives); ++p) {
         struct d12_agx3_direct_nonindexed_index_list_info nonindexed_info = {
            .primitive = primitives[p],
            .count = UINT32_MAX - p,
            .instances = p + 1,
            .start = UINT32_C(0x80000000) | p,
         };
         struct d12_agx3_direct_nonindexed_index_list_info decoded_nonindexed;
         struct d12_agx3_direct_nonindexed_index_list nonindexed;
         CHECK(d12_agx3_encode_direct_nonindexed_index_list(
            supported_chips[c], &nonindexed_info, &nonindexed));
         CHECK(d12_agx3_decode_direct_nonindexed_index_list(
            supported_chips[c], nonindexed.bytes, nonindexed.length,
            &decoded_nonindexed));
         if (check_nonindexed_info(&decoded_nonindexed, &nonindexed_info))
            return 1;

         for (unsigned size = 1; size <= 2; ++size) {
            for (unsigned restart = 0; restart <= 1; ++restart) {
               struct d12_agx3_direct_indexed_index_list_info indexed_info = {
                  .primitive = primitives[p],
                  .restart_enable = restart != 0,
                  .index_size = (enum d12_agx3_index_size)size,
                  .index_buffer = UINT64_C(0xabcdef123456),
                  .count = UINT32_MAX - p,
                  .instances = p + 1,
                  .start = UINT32_C(0x80000000) | p,
                  .index_buffer_size_word = UINT32_C(0xfedcba98),
               };
               struct d12_agx3_direct_indexed_index_list_info decoded_indexed;
               struct d12_agx3_direct_indexed_index_list indexed;
               CHECK(d12_agx3_encode_direct_indexed_index_list(
                  supported_chips[c], &indexed_info, &indexed));
               CHECK(d12_agx3_decode_direct_indexed_index_list(
                  supported_chips[c], indexed.bytes, indexed.length,
                  &decoded_indexed));
               if (check_indexed_info(&decoded_indexed, &indexed_info))
                  return 1;
            }
         }
      }
   }

   for (unsigned bit = 0; bit < 48; ++bit) {
      struct d12_agx3_direct_indexed_index_list_info info = {
         .primitive = 6,
         .index_size = D12_AGX3_INDEX_SIZE_U16,
         .index_buffer = UINT64_C(1) << bit,
         .instances = 1,
      };
      struct d12_agx3_direct_indexed_index_list_info decoded;
      struct d12_agx3_direct_indexed_index_list encoded;
      CHECK(d12_agx3_encode_direct_indexed_index_list(AGX_CHIP_G17S, &info,
                                                       &encoded));
      CHECK(d12_agx3_decode_direct_indexed_index_list(
         AGX_CHIP_G17S, encoded.bytes, encoded.length, &decoded));
      CHECK(decoded.index_buffer == info.index_buffer);
   }

   return 0;
}

static int
check_corruptions_and_rejections(void)
{
   const struct d12_agx3_direct_nonindexed_index_list_info nonindexed_info = {
      .primitive = 6,
      .count = UINT32_C(0x11223344),
      .instances = UINT32_C(0x55667788),
      .start = UINT32_C(0xaabbccdd),
   };
   const struct d12_agx3_direct_indexed_index_list_info indexed_info = {
      .primitive = 9,
      .restart_enable = true,
      .index_size = D12_AGX3_INDEX_SIZE_U32,
      .index_buffer = UINT64_C(0x123456789abc),
      .count = UINT32_C(0x01020304),
      .instances = UINT32_C(0x55667788),
      .start = UINT32_C(0xfffffff9),
      .index_buffer_size_word = UINT32_C(0x10203040),
   };

   for (unsigned g = 0; g < 2; ++g) {
      const enum agx_chip chip = g ? AGX_CHIP_G17S : AGX_CHIP_G15G;
      struct d12_agx3_direct_nonindexed_index_list baseline_nonindexed;
      struct d12_agx3_direct_indexed_index_list baseline_indexed;
      CHECK(d12_agx3_encode_direct_nonindexed_index_list(
         chip, &nonindexed_info, &baseline_nonindexed));
      CHECK(d12_agx3_encode_direct_indexed_index_list(chip, &indexed_info,
                                                      &baseline_indexed));

      for (unsigned bit = 0; bit < baseline_nonindexed.length * 8u; ++bit) {
         uint8_t changed[D12_AGX3_DIRECT_NONINDEXED_INDEX_LIST_MAX_LENGTH];
         struct d12_agx3_direct_nonindexed_index_list_info decoded;
         struct d12_agx3_direct_nonindexed_index_list repacked;
         memcpy(changed, baseline_nonindexed.bytes, baseline_nonindexed.length);
         changed[bit / 8] ^= (uint8_t)(1u << (bit % 8));
         memset(&decoded, 0xa5, sizeof(decoded));
         const bool accepted = d12_agx3_decode_direct_nonindexed_index_list(
            chip, changed, baseline_nonindexed.length, &decoded);
         if (accepted) {
            CHECK(d12_agx3_encode_direct_nonindexed_index_list(
               chip, &decoded, &repacked));
            CHECK(memcmp(changed, repacked.bytes, repacked.length) == 0);
         } else if (check_zero_nonindexed_info(&decoded)) {
            return 1;
         }
      }

      for (unsigned bit = 0; bit < baseline_indexed.length * 8u; ++bit) {
         uint8_t changed[D12_AGX3_DIRECT_INDEXED_INDEX_LIST_MAX_LENGTH];
         struct d12_agx3_direct_indexed_index_list_info decoded;
         struct d12_agx3_direct_indexed_index_list repacked;
         memcpy(changed, baseline_indexed.bytes, baseline_indexed.length);
         changed[bit / 8] ^= (uint8_t)(1u << (bit % 8));
         memset(&decoded, 0xa5, sizeof(decoded));
         const bool accepted = d12_agx3_decode_direct_indexed_index_list(
            chip, changed, baseline_indexed.length, &decoded);
         if (accepted) {
            CHECK(d12_agx3_encode_direct_indexed_index_list(chip, &decoded,
                                                            &repacked));
            CHECK(memcmp(changed, repacked.bytes, repacked.length) == 0);
         } else if (check_zero_indexed_info(&decoded)) {
            return 1;
         }
      }
   }

   struct d12_agx3_direct_nonindexed_index_list g15_nonindexed;
   struct d12_agx3_direct_nonindexed_index_list hal300_nonindexed;
   struct d12_agx3_direct_indexed_index_list g15_indexed;
   struct d12_agx3_direct_indexed_index_list hal300_indexed;
   CHECK(d12_agx3_encode_direct_nonindexed_index_list(
      AGX_CHIP_G15G, &nonindexed_info, &g15_nonindexed));
   CHECK(d12_agx3_encode_direct_nonindexed_index_list(
      AGX_CHIP_G17S, &nonindexed_info, &hal300_nonindexed));
   CHECK(d12_agx3_encode_direct_indexed_index_list(
      AGX_CHIP_G15G, &indexed_info, &g15_indexed));
   CHECK(d12_agx3_encode_direct_indexed_index_list(
      AGX_CHIP_G17S, &indexed_info, &hal300_indexed));

   {
      struct d12_agx3_direct_nonindexed_index_list_info decoded_nonindexed;
      struct d12_agx3_direct_indexed_index_list_info decoded_indexed;
      CHECK(!d12_agx3_decode_direct_nonindexed_index_list(
         AGX_CHIP_G15G, hal300_nonindexed.bytes, hal300_nonindexed.length,
         &decoded_nonindexed));
      if (check_zero_nonindexed_info(&decoded_nonindexed))
         return 1;
      CHECK(!d12_agx3_decode_direct_indexed_index_list(
         AGX_CHIP_G15G, hal300_indexed.bytes, hal300_indexed.length,
         &decoded_indexed));
      if (check_zero_indexed_info(&decoded_indexed))
         return 1;
      CHECK(!d12_agx3_decode_direct_nonindexed_index_list(
         AGX_CHIP_G17S, g15_nonindexed.bytes, g15_nonindexed.length,
         &decoded_nonindexed));
      if (check_zero_nonindexed_info(&decoded_nonindexed))
         return 1;
      CHECK(!d12_agx3_decode_direct_indexed_index_list(
         AGX_CHIP_G17S, g15_indexed.bytes, g15_indexed.length,
         &decoded_indexed));
      if (check_zero_indexed_info(&decoded_indexed))
         return 1;
   }

   for (unsigned i = 0;
        i < sizeof(unsupported_chips) / sizeof(unsupported_chips[0]); ++i) {
      struct d12_agx3_direct_nonindexed_index_list encoded_nonindexed;
      struct d12_agx3_direct_indexed_index_list encoded_indexed;
      struct d12_agx3_direct_nonindexed_index_list_info decoded_nonindexed;
      struct d12_agx3_direct_indexed_index_list_info decoded_indexed;
      memset(&encoded_nonindexed, 0xa5, sizeof(encoded_nonindexed));
      CHECK(!d12_agx3_encode_direct_nonindexed_index_list(
         unsupported_chips[i], &nonindexed_info, &encoded_nonindexed));
      if (check_zero_nonindexed(&encoded_nonindexed))
         return 1;
      memset(&encoded_indexed, 0xa5, sizeof(encoded_indexed));
      CHECK(!d12_agx3_encode_direct_indexed_index_list(
         unsupported_chips[i], &indexed_info, &encoded_indexed));
      if (check_zero_indexed(&encoded_indexed))
         return 1;
      CHECK(!d12_agx3_decode_direct_nonindexed_index_list(
         unsupported_chips[i], g15_nonindexed.bytes, g15_nonindexed.length,
         &decoded_nonindexed));
      if (check_zero_nonindexed_info(&decoded_nonindexed))
         return 1;
      CHECK(!d12_agx3_decode_direct_indexed_index_list(
         unsupported_chips[i], g15_indexed.bytes, g15_indexed.length,
         &decoded_indexed));
      if (check_zero_indexed_info(&decoded_indexed))
         return 1;
   }

   for (uint32_t size = 0; size < g15_nonindexed.length; ++size) {
      struct d12_agx3_direct_nonindexed_index_list_info decoded;
      CHECK(!d12_agx3_decode_direct_nonindexed_index_list(
         AGX_CHIP_G15G, g15_nonindexed.bytes, size, &decoded));
      if (check_zero_nonindexed_info(&decoded))
         return 1;
   }
   for (uint32_t size = 0; size < hal300_nonindexed.length; ++size) {
      struct d12_agx3_direct_nonindexed_index_list_info decoded;
      CHECK(!d12_agx3_decode_direct_nonindexed_index_list(
         AGX_CHIP_G17S, hal300_nonindexed.bytes, size, &decoded));
      if (check_zero_nonindexed_info(&decoded))
         return 1;
   }
   for (uint32_t size = 0; size < g15_indexed.length; ++size) {
      struct d12_agx3_direct_indexed_index_list_info decoded;
      CHECK(!d12_agx3_decode_direct_indexed_index_list(
         AGX_CHIP_G15G, g15_indexed.bytes, size, &decoded));
      if (check_zero_indexed_info(&decoded))
         return 1;
   }
   for (uint32_t size = 0; size < hal300_indexed.length; ++size) {
      struct d12_agx3_direct_indexed_index_list_info decoded;
      CHECK(!d12_agx3_decode_direct_indexed_index_list(
         AGX_CHIP_G17S, hal300_indexed.bytes, size, &decoded));
      if (check_zero_indexed_info(&decoded))
         return 1;
   }

   {
      struct d12_agx3_direct_nonindexed_index_list_info decoded_nonindexed;
      struct d12_agx3_direct_indexed_index_list_info decoded_indexed;
      uint8_t unaligned_nonindexed[21];
      uint8_t unaligned_indexed[33];
      memcpy(&unaligned_nonindexed[1], hal300_nonindexed.bytes,
             hal300_nonindexed.length);
      memcpy(&unaligned_indexed[1], hal300_indexed.bytes,
             hal300_indexed.length);
      CHECK(d12_agx3_decode_direct_nonindexed_index_list(
         AGX_CHIP_G17S, &unaligned_nonindexed[1], hal300_nonindexed.length,
         &decoded_nonindexed));
      CHECK(d12_agx3_decode_direct_indexed_index_list(
         AGX_CHIP_G17S, &unaligned_indexed[1], hal300_indexed.length,
         &decoded_indexed));
      CHECK(!d12_agx3_decode_direct_nonindexed_index_list(
         AGX_CHIP_G17S, hal300_nonindexed.bytes,
         hal300_nonindexed.length + 1, &decoded_nonindexed));
      CHECK(!d12_agx3_decode_direct_indexed_index_list(
         AGX_CHIP_G17S, hal300_indexed.bytes, hal300_indexed.length + 1,
         &decoded_indexed));
      CHECK(!d12_agx3_decode_direct_nonindexed_index_list(
         AGX_CHIP_G17S, NULL, hal300_nonindexed.length,
         &decoded_nonindexed));
      CHECK(!d12_agx3_decode_direct_nonindexed_index_list(
         AGX_CHIP_G17S, hal300_nonindexed.bytes, hal300_nonindexed.length,
         NULL));
      CHECK(!d12_agx3_decode_direct_indexed_index_list(
         AGX_CHIP_G17S, NULL, hal300_indexed.length, &decoded_indexed));
      CHECK(!d12_agx3_decode_direct_indexed_index_list(
         AGX_CHIP_G17S, hal300_indexed.bytes, hal300_indexed.length, NULL));
   }

   {
      struct d12_agx3_direct_nonindexed_index_list encoded_nonindexed;
      struct d12_agx3_direct_indexed_index_list encoded_indexed;
      struct d12_agx3_direct_nonindexed_index_list_info bad_nonindexed =
         nonindexed_info;
      struct d12_agx3_direct_indexed_index_list_info bad_indexed = indexed_info;
      bad_nonindexed.instances = 0;
      CHECK(!d12_agx3_encode_direct_nonindexed_index_list(
         AGX_CHIP_G15G, &bad_nonindexed, &encoded_nonindexed));
      bad_indexed.instances = 0;
      CHECK(!d12_agx3_encode_direct_indexed_index_list(
         AGX_CHIP_G15G, &bad_indexed, &encoded_indexed));
      bad_indexed = indexed_info;
      bad_indexed.index_buffer = UINT64_C(1) << 48;
      CHECK(!d12_agx3_encode_direct_indexed_index_list(
         AGX_CHIP_G17S, &bad_indexed, &encoded_indexed));
      bad_indexed = indexed_info;
      bad_indexed.index_size = (enum d12_agx3_index_size)0;
      CHECK(!d12_agx3_encode_direct_indexed_index_list(
         AGX_CHIP_G17S, &bad_indexed, &encoded_indexed));

      CHECK(!d12_agx3_encode_direct_nonindexed_index_list(
         AGX_CHIP_G15G, NULL, &encoded_nonindexed));
      CHECK(!d12_agx3_encode_direct_nonindexed_index_list(
         AGX_CHIP_G15G, &nonindexed_info, NULL));
      CHECK(!d12_agx3_encode_direct_indexed_index_list(
         AGX_CHIP_G15G, NULL, &encoded_indexed));
      CHECK(!d12_agx3_encode_direct_indexed_index_list(
         AGX_CHIP_G15G, &indexed_info, NULL));
   }

   return 0;
}

static int
check_indirect_bridge(void)
{
   const enum d12_agx3_index_size sizes[] = {D12_AGX3_INDEX_SIZE_U16,
                                             D12_AGX3_INDEX_SIZE_U32};

   for (unsigned i = 0;
        i < sizeof(supported_chips) / sizeof(supported_chips[0]); ++i) {
      const enum agx_chip chip = supported_chips[i];
      const bool hal300 = is_hal300(chip);

      for (unsigned p = 0; p < sizeof(primitives); ++p) {
         const uint8_t prim = primitives[p];
         const uint64_t A = 0x554433221100ull + ((uint64_t)prim << 4);

         struct d12_agx3_indirect_nonindexed_index_list_info ni = {
            .primitive = prim, .indirect_buffer = A};
         struct d12_agx3_indirect_nonindexed_index_list no;
         CHECK(d12_agx3_encode_indirect_nonindexed_index_list(chip, &ni, &no));
         CHECK(no.length == (hal300 ? 16 : 12));
         struct d12_agx3_indirect_nonindexed_index_list_info nback;
         CHECK(d12_agx3_decode_indirect_nonindexed_index_list(
            chip, no.bytes, no.length, &nback));
         CHECK(nback.primitive == prim &&
               nback.indirect_buffer == (A & ~(uint64_t)3));

         for (unsigned z = 0; z < 2; ++z) {
            const uint64_t IB = 0x0011ffee0040ull + ((uint64_t)prim << 32);
            struct d12_agx3_indirect_indexed_index_list_info ii = {
               .primitive = prim, .restart_enable = (z == 1),
               .index_size = sizes[z], .index_buffer = IB, .indirect_buffer = A,
               .restart_index = (z == 0) ? 0xffffu : 0xcafef00du,
               .index_buffer_size_word = 0x155u};
            struct d12_agx3_indirect_indexed_index_list io;
            CHECK(d12_agx3_encode_indirect_indexed_index_list(chip, &ii, &io));
            CHECK(io.length == (hal300 ? 36 : 32));
            struct d12_agx3_indirect_indexed_index_list_info iback;
            CHECK(d12_agx3_decode_indirect_indexed_index_list(
               chip, io.bytes, io.length, &iback));
            CHECK(iback.primitive == prim && iback.index_size == sizes[z] &&
                  iback.restart_enable == (z == 1) &&
                  iback.restart_index == ii.restart_index &&
                  iback.index_buffer == IB &&
                  iback.indirect_buffer == (A & ~(uint64_t)3) &&
                  iback.index_buffer_size_word == 0x155u);
         }
      }
   }

   for (unsigned u = 0;
        u < sizeof(unsupported_chips) / sizeof(unsupported_chips[0]); ++u) {
      struct d12_agx3_indirect_nonindexed_index_list_info ni = {
         .primitive = 0, .indirect_buffer = 0x2000};
      struct d12_agx3_indirect_nonindexed_index_list no;
      CHECK(!d12_agx3_encode_indirect_nonindexed_index_list(
         unsupported_chips[u], &ni, &no));
   }
   return 0;
}

static uint32_t
native_word(const struct d12_agx3_native_direct_draw *draw, unsigned word)
{
   const uint8_t *p = draw->bytes + word * 4;
   return p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
          ((uint32_t)p[3] << 24);
}

static int
check_native_api_draws(void)
{
   /* G11's first-index 3 selects a half-dword-aligned U16 range. The API
    * base vertex 3 biases fetched vertices, independently of index start. */
   struct d12_agx3_native_direct_draw_info info = {
      .draw_state_context_offset = 0x48000, .primitive = 6, .indexed = true,
      .index_size_B = 2, .index_buffer = UINT64_C(0x10000001000),
      .index_buffer_range_B = 12, .first_index = 3, .first_vertex = 3,
      .count = 3, .instances = 1,
   };
   struct d12_agx3_native_direct_draw draw;
   CHECK(d12_agx3_encode_native_direct_draw(AGX_CHIP_G17P, &info, &draw));
   const uint32_t expected[] = {
      0x48000, 0x40000001, 0, 0x61f20600, 0x1006, 3, 1, 3, 1, 1,
   };
   CHECK(draw.length == sizeof(expected));
   for (unsigned i = 0; i < sizeof(expected) / sizeof(expected[0]); ++i)
      CHECK(native_word(&draw, i) == expected[i]);

   const int32_t biases[] = {3, -3, INT32_MIN, INT32_MAX};
   for (unsigned width = 2; width <= 4; width *= 2) {
      info.index_size_B = width;
      info.index_buffer_range_B = 8 * width;
      info.count = 5;
      info.instances = 17;
      for (unsigned b = 0; b < sizeof(biases) / sizeof(biases[0]); ++b) {
         info.first_vertex = (uint32_t)biases[b];
         CHECK(d12_agx3_encode_native_direct_draw(AGX_CHIP_G17P, &info, &draw));
         CHECK(native_word(&draw, 4) == 0x1000 + 3 * width);
         CHECK(native_word(&draw, 5) == 5 && native_word(&draw, 6) == 17);
         CHECK(native_word(&draw, 7) == (uint32_t)biases[b]);
         CHECK(native_word(&draw, 8) == (5 * width + 3) / 4 - 1);
      }
   }
   /* Reject range overflow transactionally, never clamp a real draw count. */
   info.first_index = UINT32_MAX;
   memset(&draw, 0xff, sizeof(draw));
   CHECK(!d12_agx3_encode_native_direct_draw(AGX_CHIP_G17P, &info, &draw));
   CHECK(draw.length == 0);
   for (unsigned i = 0; i < sizeof(draw.bytes); ++i)
      CHECK(draw.bytes[i] == 0);

   info.index_buffer = 0;
   info.first_vertex = (uint32_t)-7;
   info.count = 4097;
   CHECK(d12_agx3_encode_native_direct_draw(AGX_CHIP_G17P, &info, &draw));
   CHECK(native_word(&draw, 4) == 0 && native_word(&draw, 9) == 0);
   CHECK(native_word(&draw, 5) == 4097 && native_word(&draw, 6) == 17);
   CHECK(native_word(&draw, 7) == 0 && native_word(&draw, 8) == 0);
   info.indexed = false;
   info.first_vertex = 5;
   info.count = 6;
   CHECK(d12_agx3_encode_native_direct_draw(AGX_CHIP_G17P, &info, &draw));
   CHECK(draw.length == 20);
   CHECK(native_word(&draw, 2) == 6 && native_word(&draw, 3) == 17);
   CHECK(native_word(&draw, 4) == 5);
   CHECK(!d12_agx3_encode_native_direct_draw(AGX_CHIP_G14G, &info, &draw));
   info.instances = 0;
   CHECK(!d12_agx3_encode_native_direct_draw(AGX_CHIP_G17P, &info, &draw));
   CHECK(!d12_agx3_encode_native_direct_draw(AGX_CHIP_G17P, NULL, &draw));
   CHECK(!d12_agx3_encode_native_direct_draw(AGX_CHIP_G17P, &info, NULL));
   return 0;
}

int
main(void)
{
   if (check_witnesses_and_shared_equality() || check_domains() ||
       check_corruptions_and_rejections() || check_indirect_bridge() ||
       check_native_api_draws())
      return 1;

   printf("d12_agx3_index_list_core: %u checks passed\n", checks);
   return 0;
}
