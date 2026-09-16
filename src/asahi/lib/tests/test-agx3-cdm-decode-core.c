/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "../agx3_cdm_decode_core.h"
#include "asahi/genxml/agx3_cmdstream.h"

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

static const enum agx_chip launch_targets[] = {
   AGX_CHIP_G15G, AGX_CHIP_G15S, AGX_CHIP_G15C, AGX_CHIP_G15D,
   AGX_CHIP_G16G, AGX_CHIP_G16S, AGX_CHIP_G16C, AGX_CHIP_G17P,
   AGX_CHIP_G17G, AGX_CHIP_G17S, AGX_CHIP_G17C,
};

static bool
record_is_cleared(const struct agx3_cdm_decoded_record *record)
{
   const struct agx3_cdm_decoded_record zero = {0};
   return memcmp(record, &zero, sizeof(zero)) == 0;
}

static struct agx3_cdm_launch_info
launch_info(enum agx_chip chip)
{
   return (struct agx3_cdm_launch_info){
      .chip = chip,
      .word0_payload = UINT32_C(0x1234567),
      .word1_payload = UINT32_C(0x23456),
      .pipeline = UINT64_C(0x123456789a80),
      .opaque_word3 = UINT32_C(0x89abcdef),
      .global_size = {UINT32_C(0x10203040), 7, UINT32_MAX},
      .local_size = {1, 32, 0x7ff},
   };
}

static int
test_launch_padding_and_terminate(void)
{
   for (unsigned t = 0; t < sizeof(launch_targets) / sizeof(launch_targets[0]);
        ++t) {
      const struct agx3_cdm_launch_info info = launch_info(launch_targets[t]);
      uint8_t launch[AGX3_CDM_LAUNCH_SIZE + 3] = {0};
      struct agx3_cdm_decoded_record decoded;
      CHECK(agx3_cdm_launch_pack(&info, launch));
      memset(launch + AGX3_CDM_LAUNCH_SIZE, 0xa5, 3);
      CHECK(agx3_cdm_decode_one(info.chip, launch, sizeof(launch), &decoded));
      CHECK(decoded.kind == AGX3_CDM_RECORD_LAUNCH);
      CHECK(decoded.length == AGX3_CDM_LAUNCH_SIZE);
      CHECK(decoded.chip == info.chip);
      CHECK(decoded.data.launch.word0_payload == info.word0_payload);
      CHECK(decoded.data.launch.word1_payload == info.word1_payload);
      CHECK(decoded.data.launch.pipeline == info.pipeline);
      CHECK(decoded.data.launch.opaque_word3 == info.opaque_word3);
      CHECK(memcmp(decoded.data.launch.global_size, info.global_size,
                   sizeof(info.global_size)) == 0);
      CHECK(memcmp(decoded.data.launch.local_size, info.local_size,
                   sizeof(info.local_size)) == 0);

      for (uint8_t count = 0; count < 8; ++count) {
         const struct agx3_cdm_link_padding_info padding = {
            .chip = info.chip,
            .count = count,
         };
         uint8_t bytes[AGX3_CDM_LINK_PADDING_SIZE];
         CHECK(agx3_cdm_link_padding_pack(&padding, bytes));
         CHECK(agx3_cdm_decode_one(info.chip, bytes, sizeof(bytes), &decoded));
         CHECK(decoded.kind == AGX3_CDM_RECORD_LINK_PADDING);
         CHECK(decoded.length == AGX3_CDM_LINK_PADDING_SIZE);
         CHECK(decoded.chip == info.chip);
         CHECK(decoded.data.link_padding.count == count);
      }

      uint8_t terminate[AGX3_CDM_STREAM_TERMINATE_SIZE];
      CHECK(agx3_cdm_stream_terminate_pack(info.chip, terminate));
      CHECK(agx3_cdm_decode_one(info.chip, terminate, sizeof(terminate),
                                &decoded));
      CHECK(decoded.kind == AGX3_CDM_RECORD_STREAM_TERMINATE);
      CHECK(decoded.length == AGX3_CDM_STREAM_TERMINATE_SIZE);
   }

   return 0;
}

static bool
indirect_launch_equal(const struct agx3_cdm_indirect_launch_info *a,
                      const struct agx3_cdm_indirect_launch_info *b)
{
   return a->chip == b->chip && a->word0_payload == b->word0_payload &&
          a->word1_payload == b->word1_payload &&
          a->pipeline == b->pipeline &&
          a->opaque_word3 == b->opaque_word3 &&
          a->indirect_word4_payload == b->indirect_word4_payload &&
          a->indirect_parameters == b->indirect_parameters;
}

static int
test_indirect_launch_codec(void)
{
   static const uint8_t witness[AGX3_CDM_INDIRECT_LAUNCH_SIZE] = {
      0x67, 0x45, 0x23, 0x01, 0xde, 0xbc, 0x0a, 0x01,
      0x23, 0x01, 0xef, 0x8d, 0xef, 0xbe, 0xad, 0x20,
      0xcd, 0xab, 0xbc, 0x9a, 0x78, 0x56, 0x34, 0x12,
   };
   const struct agx3_cdm_indirect_launch_info base = {
      .chip = AGX_CHIP_G17S,
      .word0_payload = UINT32_C(0x01234567),
      .word1_payload = UINT32_C(0x000abcde),
      .pipeline = UINT64_C(0x01237bc048c0),
      .opaque_word3 = UINT32_C(0x20adbeef),
      .indirect_word4_payload = UINT16_C(0x9abc),
      .indirect_parameters = UINT64_C(0xabcd12345678),
   };

   for (unsigned t = 0; t < sizeof(launch_targets) / sizeof(launch_targets[0]);
        ++t) {
      struct agx3_cdm_indirect_launch_info info = base, decoded;
      uint8_t packed[AGX3_CDM_INDIRECT_LAUNCH_SIZE];
      info.chip = launch_targets[t];
      CHECK(agx3_cdm_indirect_launch_pack(&info, packed));
      CHECK(memcmp(packed, witness, sizeof(witness)) == 0);
      CHECK(agx3_cdm_indirect_launch_unpack(info.chip, packed, sizeof(packed),
                                            &decoded));
      CHECK(indirect_launch_equal(&decoded, &info));
   }

   /* Every bit exposed by the bounded codec survives the six-word form. */
   for (unsigned bit = 0; bit < 189; ++bit) {
      struct agx3_cdm_indirect_launch_info info = {
         .chip = AGX_CHIP_G17G,
      };
      if (bit < 29)
         info.word0_payload = UINT32_C(1) << bit;
      else if (bit < 51)
         info.word1_payload = UINT32_C(1) << (bit - 29);
      else if (bit < 93)
         info.pipeline = UINT64_C(1) << (bit - 51 + 6);
      else if (bit < 125)
         info.opaque_word3 = UINT32_C(1) << (bit - 93);
      else if (bit < 141)
         info.indirect_word4_payload = UINT16_C(1) << (bit - 125);
      else
         info.indirect_parameters = UINT64_C(1) << (bit - 141);

      uint8_t packed[AGX3_CDM_INDIRECT_LAUNCH_SIZE];
      struct agx3_cdm_indirect_launch_info decoded;
      CHECK(agx3_cdm_indirect_launch_pack(&info, packed));
      CHECK(agx3_cdm_indirect_launch_unpack(info.chip, packed, sizeof(packed),
                                            &decoded));
      CHECK(indirect_launch_equal(&decoded, &info));
   }

   const enum agx_chip rejected[] = {
      AGX_CHIP_G13G, AGX_CHIP_G14X, AGX_CHIP_G15P, AGX_CHIP_G16P,
      AGX_CHIP_G17A, AGX_CHIP_UNKNOWN, (enum agx_chip)999,
   };
   for (unsigned i = 0; i < sizeof(rejected) / sizeof(rejected[0]); ++i) {
      struct agx3_cdm_indirect_launch_info info = base, decoded;
      uint8_t packed[AGX3_CDM_INDIRECT_LAUNCH_SIZE];
      info.chip = rejected[i];
      memset(packed, 0xa5, sizeof(packed));
      CHECK(!agx3_cdm_indirect_launch_pack(&info, packed));
      for (unsigned b = 0; b < sizeof(packed); ++b)
         CHECK(packed[b] == 0);
      memset(&decoded, 0xa5, sizeof(decoded));
      CHECK(!agx3_cdm_indirect_launch_unpack(
         info.chip, witness, sizeof(witness), &decoded));
      const struct agx3_cdm_indirect_launch_info zero = {0};
      CHECK(memcmp(&decoded, &zero, sizeof(zero)) == 0);
   }

   const struct agx3_cdm_indirect_launch_info invalid[] = {
      {.chip = AGX_CHIP_G17S,
       .word0_payload = UINT32_C(1) << 29},
      {.chip = AGX_CHIP_G17S,
       .word1_payload = UINT32_C(1) << 22},
      {.chip = AGX_CHIP_G17S,
       .pipeline = 1},
      {.chip = AGX_CHIP_G17S,
       .pipeline = UINT64_C(1) << 48},
      {.chip = AGX_CHIP_G17S,
       .indirect_parameters = UINT64_C(1) << 48},
   };
   for (unsigned i = 0; i < sizeof(invalid) / sizeof(invalid[0]); ++i) {
      uint8_t packed[AGX3_CDM_INDIRECT_LAUNCH_SIZE];
      memset(packed, 0xa5, sizeof(packed));
      CHECK(!agx3_cdm_indirect_launch_pack(&invalid[i], packed));
      for (unsigned b = 0; b < sizeof(packed); ++b)
         CHECK(packed[b] == 0);
   }

   struct agx3_cdm_indirect_launch_info decoded;
   memset(&decoded, 0xa5, sizeof(decoded));
   CHECK(!agx3_cdm_indirect_launch_unpack(
      base.chip, witness, sizeof(witness) - 1, &decoded));
   CHECK(decoded.pipeline == 0);
   CHECK(!agx3_cdm_indirect_launch_unpack(
      base.chip, witness, sizeof(witness) + 1, &decoded));
   CHECK(decoded.pipeline == 0);

   uint8_t malformed[sizeof(witness)];
   memcpy(malformed, witness, sizeof(malformed));
   malformed[3] |= 0x20;
   CHECK(!agx3_cdm_indirect_launch_unpack(
      base.chip, malformed, sizeof(malformed), &decoded));
   CHECK(decoded.pipeline == 0);
   CHECK(!agx3_cdm_indirect_launch_pack(&base, NULL));
   uint8_t null_input[AGX3_CDM_INDIRECT_LAUNCH_SIZE];
   memset(null_input, 0xa5, sizeof(null_input));
   CHECK(!agx3_cdm_indirect_launch_pack(NULL, null_input));
   for (unsigned b = 0; b < sizeof(null_input); ++b)
      CHECK(null_input[b] == 0);
   memset(&decoded, 0xa5, sizeof(decoded));
   CHECK(!agx3_cdm_indirect_launch_unpack(
      base.chip, NULL, sizeof(witness), &decoded));
   CHECK(decoded.pipeline == 0);
   CHECK(!agx3_cdm_indirect_launch_unpack(
      base.chip, witness, sizeof(witness), NULL));
   return 0;
}

static bool
indirect_launch_36_equal(
   const struct agx3_cdm_indirect_launch_36_info *a,
   const struct agx3_cdm_indirect_launch_36_info *b)
{
   return a->chip == b->chip && a->word0_payload == b->word0_payload &&
          a->word1_payload == b->word1_payload &&
          a->pipeline == b->pipeline &&
          a->opaque_word3 == b->opaque_word3 &&
          a->indirect_word4_payload == b->indirect_word4_payload &&
          a->indirect_parameters == b->indirect_parameters &&
          memcmp(a->trailing_words, b->trailing_words,
                 sizeof(a->trailing_words)) == 0;
}

static int
test_indirect_launch_36_codec(void)
{
   static const uint8_t witness[AGX3_CDM_INDIRECT_LAUNCH_36_SIZE] = {
      0x67, 0x45, 0x23, 0x01, 0xde, 0xbc, 0x0a, 0x01,
      0x23, 0x01, 0xef, 0x8d, 0xef, 0xbe, 0xad, 0x20,
      0xcd, 0xab, 0xbc, 0x9a, 0x78, 0x56, 0x34, 0x12,
      0x55, 0x01, 0x00, 0x00, 0xaa, 0x03, 0x00, 0x00,
      0xff, 0x07, 0x00, 0x00,
   };
   const struct agx3_cdm_indirect_launch_36_info base = {
      .chip = AGX_CHIP_G17S,
      .word0_payload = UINT32_C(0x01234567),
      .word1_payload = UINT32_C(0x000abcde),
      .pipeline = UINT64_C(0x01237bc048c0),
      .opaque_word3 = UINT32_C(0x20adbeef),
      .indirect_word4_payload = UINT16_C(0x9abc),
      .indirect_parameters = UINT64_C(0xabcd12345678),
      .trailing_words = {0x155, 0x3aa, 0x7ff},
   };

   for (unsigned t = 0; t < sizeof(launch_targets) / sizeof(launch_targets[0]);
        ++t) {
      struct agx3_cdm_indirect_launch_36_info info = base, decoded;
      uint8_t packed[AGX3_CDM_INDIRECT_LAUNCH_36_SIZE];
      info.chip = launch_targets[t];
      CHECK(agx3_cdm_indirect_launch_36_pack(&info, packed));
      CHECK(memcmp(packed, witness, sizeof(witness)) == 0);
      CHECK(agx3_cdm_indirect_launch_36_unpack(
         info.chip, packed, sizeof(packed), &decoded));
      CHECK(indirect_launch_36_equal(&decoded, &info));
   }

   /* Every independently writable bit survives this exact nine-word form. */
   for (unsigned bit = 0; bit < 222; ++bit) {
      struct agx3_cdm_indirect_launch_36_info info = {
         .chip = AGX_CHIP_G17G,
      };
      if (bit < 29)
         info.word0_payload = UINT32_C(1) << bit;
      else if (bit < 51)
         info.word1_payload = UINT32_C(1) << (bit - 29);
      else if (bit < 93)
         info.pipeline = UINT64_C(1) << (bit - 51 + 6);
      else if (bit < 125)
         info.opaque_word3 = UINT32_C(1) << (bit - 93);
      else if (bit < 141)
         info.indirect_word4_payload = UINT16_C(1) << (bit - 125);
      else if (bit < 189)
         info.indirect_parameters = UINT64_C(1) << (bit - 141);
      else {
         const unsigned trailing = bit - 189;
         info.trailing_words[trailing / 11] = UINT32_C(1) << (trailing % 11);
      }

      uint8_t packed[AGX3_CDM_INDIRECT_LAUNCH_36_SIZE];
      struct agx3_cdm_indirect_launch_36_info decoded;
      CHECK(agx3_cdm_indirect_launch_36_pack(&info, packed));
      CHECK(agx3_cdm_indirect_launch_36_unpack(
         info.chip, packed, sizeof(packed), &decoded));
      CHECK(indirect_launch_36_equal(&decoded, &info));
   }

   const enum agx_chip rejected[] = {
      AGX_CHIP_G13G, AGX_CHIP_G14X, AGX_CHIP_G15P, AGX_CHIP_G16P,
      AGX_CHIP_G17A, AGX_CHIP_UNKNOWN, (enum agx_chip)999,
   };
   for (unsigned i = 0; i < sizeof(rejected) / sizeof(rejected[0]); ++i) {
      struct agx3_cdm_indirect_launch_36_info info = base, decoded;
      uint8_t packed[AGX3_CDM_INDIRECT_LAUNCH_36_SIZE];
      info.chip = rejected[i];
      memset(packed, 0xa5, sizeof(packed));
      CHECK(!agx3_cdm_indirect_launch_36_pack(&info, packed));
      for (unsigned b = 0; b < sizeof(packed); ++b)
         CHECK(packed[b] == 0);
      memset(&decoded, 0xa5, sizeof(decoded));
      CHECK(!agx3_cdm_indirect_launch_36_unpack(
         info.chip, witness, sizeof(witness), &decoded));
      const struct agx3_cdm_indirect_launch_36_info zero = {0};
      CHECK(memcmp(&decoded, &zero, sizeof(zero)) == 0);
   }

   const struct agx3_cdm_indirect_launch_36_info invalid[] = {
      {.chip = AGX_CHIP_G17S, .word0_payload = UINT32_C(1) << 29},
      {.chip = AGX_CHIP_G17S, .word1_payload = UINT32_C(1) << 22},
      {.chip = AGX_CHIP_G17S, .pipeline = 1},
      {.chip = AGX_CHIP_G17S, .pipeline = UINT64_C(1) << 48},
      {.chip = AGX_CHIP_G17S,
       .indirect_parameters = UINT64_C(1) << 48},
      {.chip = AGX_CHIP_G17S, .trailing_words = {0, 0x800, 0}},
   };
   for (unsigned i = 0; i < sizeof(invalid) / sizeof(invalid[0]); ++i) {
      uint8_t packed[AGX3_CDM_INDIRECT_LAUNCH_36_SIZE];
      memset(packed, 0xa5, sizeof(packed));
      CHECK(!agx3_cdm_indirect_launch_36_pack(&invalid[i], packed));
      for (unsigned b = 0; b < sizeof(packed); ++b)
         CHECK(packed[b] == 0);
   }

   struct agx3_cdm_indirect_launch_36_info decoded;
   memset(&decoded, 0xa5, sizeof(decoded));
   CHECK(!agx3_cdm_indirect_launch_36_unpack(
      base.chip, witness, sizeof(witness) - 1, &decoded));
   CHECK(decoded.indirect_parameters == 0);
   CHECK(!agx3_cdm_indirect_launch_36_unpack(
      base.chip, witness, sizeof(witness) + 1, &decoded));
   CHECK(decoded.indirect_parameters == 0);

   uint8_t malformed[sizeof(witness)];
   memcpy(malformed, witness, sizeof(malformed));
   malformed[3] |= 0x20;
   CHECK(!agx3_cdm_indirect_launch_36_unpack(
      base.chip, malformed, sizeof(malformed), &decoded));
   CHECK(decoded.indirect_parameters == 0);
   memcpy(malformed, witness, sizeof(malformed));
   malformed[34] = 0x08;
   CHECK(!agx3_cdm_indirect_launch_36_unpack(
      base.chip, malformed, sizeof(malformed), &decoded));
   CHECK(decoded.indirect_parameters == 0);

   CHECK(!agx3_cdm_indirect_launch_36_pack(&base, NULL));
   uint8_t null_input[AGX3_CDM_INDIRECT_LAUNCH_36_SIZE];
   memset(null_input, 0xa5, sizeof(null_input));
   CHECK(!agx3_cdm_indirect_launch_36_pack(NULL, null_input));
   for (unsigned b = 0; b < sizeof(null_input); ++b)
      CHECK(null_input[b] == 0);
   CHECK(!agx3_cdm_indirect_launch_36_unpack(
      base.chip, witness, sizeof(witness), NULL));
   memset(&decoded, 0xa5, sizeof(decoded));
   CHECK(!agx3_cdm_indirect_launch_36_unpack(
      base.chip, NULL, sizeof(witness), &decoded));
   CHECK(decoded.indirect_parameters == 0);
   return 0;
}

static int
test_links_and_barriers(void)
{
   const enum agx_chip link_targets[] = {
      AGX_CHIP_G15G, AGX_CHIP_G15S, AGX_CHIP_G15C, AGX_CHIP_G15D,
      AGX_CHIP_G16G, AGX_CHIP_G16S, AGX_CHIP_G16C, AGX_CHIP_G17P,
      AGX_CHIP_G17G, AGX_CHIP_G17S, AGX_CHIP_G17C,
   };
   for (unsigned t = 0; t < sizeof(link_targets) / sizeof(link_targets[0]);
        ++t) {
      for (unsigned returning = 0; returning < 2; ++returning) {
         const struct agx3_cdm_stream_link_info info = {
            .chip = link_targets[t],
            .target = UINT64_C(0xabcdef012345),
            .with_return = returning,
         };
         uint8_t bytes[AGX3_CDM_STREAM_LINK_SIZE];
         struct agx3_cdm_decoded_record decoded;
         CHECK(agx3_cdm_stream_link_pack(&info, bytes));
         CHECK(agx3_cdm_decode_one(info.chip, bytes, sizeof(bytes), &decoded));
         CHECK(decoded.kind == AGX3_CDM_RECORD_STREAM_LINK);
         CHECK(decoded.length == AGX3_CDM_STREAM_LINK_SIZE);
         CHECK(decoded.chip == info.chip);
         CHECK(decoded.data.stream_link.target == info.target);
         CHECK(decoded.data.stream_link.with_return == info.with_return);
      }
   }

   for (unsigned t = 0; t < sizeof(launch_targets) / sizeof(launch_targets[0]);
        ++t) {
      const bool gen7 = launch_targets[t] == AGX_CHIP_G17G ||
                        launch_targets[t] == AGX_CHIP_G17S ||
                        launch_targets[t] == AGX_CHIP_G17C;
      for (uint8_t flags = 0; flags < 16; ++flags) {
         if (!gen7 && flags != 0 && flags != 5)
            continue;

         const struct agx3_cdm_barrier_info barrier = {
            .chip = launch_targets[t],
            .flags = flags,
         };
         uint8_t bytes[AGX3_CDM_BARRIER_SIZE];
         struct agx3_cdm_decoded_record decoded;
         CHECK(agx3_cdm_barrier_pack(&barrier, bytes));
         CHECK(
            agx3_cdm_decode_one(barrier.chip, bytes, sizeof(bytes), &decoded));
         CHECK(decoded.kind == AGX3_CDM_RECORD_BARRIER);
         CHECK(decoded.length == AGX3_CDM_BARRIER_SIZE);
         CHECK(decoded.data.barrier.flags == flags);

         const struct agx3_cdm_substream_barrier_info substream = {
            .chip = launch_targets[t],
            .flags = flags,
            .substream = 0xfe,
         };
         CHECK(agx3_cdm_substream_barrier_pack(&substream, bytes));
         CHECK(agx3_cdm_decode_one(substream.chip, bytes, sizeof(bytes),
                                   &decoded));
         CHECK(decoded.kind == AGX3_CDM_RECORD_SUBSTREAM_BARRIER);
         CHECK(decoded.length == AGX3_CDM_BARRIER_SIZE);
         CHECK(decoded.data.substream_barrier.flags == flags);
         CHECK(decoded.data.substream_barrier.substream == substream.substream);
      }

      if (gen7) {
         uint8_t bytes[AGX3_CDM_SECONDARY_BARRIER_SIZE];
         struct agx3_cdm_decoded_record decoded;
         CHECK(agx3_cdm_secondary_barrier_pack(launch_targets[t], bytes));
         CHECK(agx3_cdm_decode_one(launch_targets[t], bytes, sizeof(bytes),
                                   &decoded));
         CHECK(decoded.kind == AGX3_CDM_RECORD_SECONDARY_BARRIER);
         CHECK(decoded.length == AGX3_CDM_SECONDARY_BARRIER_SIZE);
      }
   }

   {
      uint8_t bytes[AGX3_CDM_SSTT_FLUSH_INVALIDATE_SIZE];
      struct agx3_cdm_decoded_record decoded;
      CHECK(agx3_cdm_sstt_flush_invalidate_pack(AGX_CHIP_G17P, bytes));
      CHECK(agx3_cdm_decode_one(AGX_CHIP_G17P, bytes, sizeof(bytes),
                                &decoded));
      CHECK(decoded.kind == AGX3_CDM_RECORD_SSTT_FLUSH_INVALIDATE);
      CHECK(decoded.length == AGX3_CDM_SSTT_FLUSH_INVALIDATE_SIZE);
      CHECK(decoded.chip == AGX_CHIP_G17P);

      bytes[0] ^= 1;
      CHECK(!agx3_cdm_sstt_flush_invalidate_validate(AGX_CHIP_G17P, bytes,
                                                      sizeof(bytes)));
      CHECK(!agx3_cdm_sstt_flush_invalidate_pack(AGX_CHIP_G17S, bytes));
      CHECK(!agx3_cdm_sstt_flush_invalidate_validate(
         AGX_CHIP_G17P, bytes, AGX3_CDM_SSTT_FLUSH_INVALIDATE_SIZE - 1));
   }

   return 0;
}

static int
test_sequence_and_rejections(void)
{
   uint8_t stream[AGX3_CDM_LAUNCH_SIZE + AGX3_CDM_LINK_PADDING_SIZE +
                  AGX3_CDM_BARRIER_SIZE + AGX3_CDM_SECONDARY_BARRIER_SIZE +
                  AGX3_CDM_STREAM_TERMINATE_SIZE] = {0};
   const struct agx3_cdm_launch_info launch = launch_info(AGX_CHIP_G17S);
   const struct agx3_cdm_link_padding_info padding = {
      .chip = AGX_CHIP_G17S,
      .count = 7,
   };
   const struct agx3_cdm_barrier_info barrier = {
      .chip = AGX_CHIP_G17S,
      .flags = 0xf,
   };
   size_t offset = 0;
   CHECK(agx3_cdm_launch_pack(&launch, &stream[offset]));
   offset += AGX3_CDM_LAUNCH_SIZE;
   CHECK(agx3_cdm_link_padding_pack(&padding, &stream[offset]));
   offset += AGX3_CDM_LINK_PADDING_SIZE;
   CHECK(agx3_cdm_barrier_pack(&barrier, &stream[offset]));
   offset += AGX3_CDM_BARRIER_SIZE;
   CHECK(agx3_cdm_secondary_barrier_pack(AGX_CHIP_G17S, &stream[offset]));
   offset += AGX3_CDM_SECONDARY_BARRIER_SIZE;
   CHECK(agx3_cdm_stream_terminate_pack(AGX_CHIP_G17S, &stream[offset]));
   offset += AGX3_CDM_STREAM_TERMINATE_SIZE;
   CHECK(offset == sizeof(stream));

   const enum agx3_cdm_record_kind expected[] = {
      AGX3_CDM_RECORD_LAUNCH,           AGX3_CDM_RECORD_LINK_PADDING,
      AGX3_CDM_RECORD_BARRIER,          AGX3_CDM_RECORD_SECONDARY_BARRIER,
      AGX3_CDM_RECORD_STREAM_TERMINATE,
   };
   offset = 0;
   for (unsigned i = 0; i < sizeof(expected) / sizeof(expected[0]); ++i) {
      struct agx3_cdm_decoded_record decoded;
      CHECK(agx3_cdm_decode_one(AGX_CHIP_G17S, &stream[offset],
                                sizeof(stream) - offset, &decoded));
      CHECK(decoded.kind == expected[i]);
      CHECK(decoded.length != 0);
      offset += decoded.length;
   }
   CHECK(offset == sizeof(stream));

   const enum agx_chip unsupported[] = {
      AGX_CHIP_G13G, AGX_CHIP_G15P,    AGX_CHIP_G16P,
      AGX_CHIP_G17A, AGX_CHIP_UNKNOWN, (enum agx_chip)999,
   };
   uint8_t launch_bytes[AGX3_CDM_LAUNCH_SIZE];
   CHECK(agx3_cdm_launch_pack(&launch, launch_bytes));
   for (unsigned i = 0; i < sizeof(unsupported) / sizeof(unsupported[0]); ++i) {
      struct agx3_cdm_decoded_record decoded;
      memset(&decoded, 0xa5, sizeof(decoded));
      CHECK(!agx3_cdm_decode_one(unsupported[i], launch_bytes,
                                 sizeof(launch_bytes), &decoded));
      CHECK(record_is_cleared(&decoded));
   }

   for (size_t size = 0; size < sizeof(launch_bytes); ++size) {
      struct agx3_cdm_decoded_record decoded;
      memset(&decoded, 0xa5, sizeof(decoded));
      CHECK(!agx3_cdm_decode_one(AGX_CHIP_G17S, launch_bytes, size, &decoded));
      CHECK(record_is_cleared(&decoded));
   }

   for (uint32_t type = 4; type < 8; ++type) {
      if (type == 5)
         continue;
      uint8_t unknown[8] = {0};
      agx3_cmdstream_store_le32(unknown, type << 29);
      struct agx3_cdm_decoded_record decoded;
      memset(&decoded, 0xa5, sizeof(decoded));
      CHECK(!agx3_cdm_decode_one(AGX_CHIP_G17S, unknown, sizeof(unknown),
                                 &decoded));
      CHECK(record_is_cleared(&decoded));
   }

   struct agx3_cdm_decoded_record decoded;
   memset(&decoded, 0xa5, sizeof(decoded));
   CHECK(!agx3_cdm_decode_one(AGX_CHIP_G17S, NULL, sizeof(stream), &decoded));
   CHECK(record_is_cleared(&decoded));
   CHECK(!agx3_cdm_decode_one(AGX_CHIP_G17S, stream, sizeof(stream), NULL));

   uint8_t malformed_padding[AGX3_CDM_LINK_PADDING_SIZE];
   CHECK(agx3_cdm_link_padding_pack(&padding, malformed_padding));
   malformed_padding[1] = 1;
   memset(&decoded, 0xa5, sizeof(decoded));
   CHECK(!agx3_cdm_decode_one(AGX_CHIP_G17S, malformed_padding,
                              sizeof(malformed_padding), &decoded));
   CHECK(record_is_cleared(&decoded));

   uint8_t malformed_terminate[AGX3_CDM_STREAM_TERMINATE_SIZE];
   CHECK(agx3_cdm_stream_terminate_pack(AGX_CHIP_G17S, malformed_terminate));
   malformed_terminate[0] ^= 1;
   memset(&decoded, 0xa5, sizeof(decoded));
   CHECK(!agx3_cdm_decode_one(AGX_CHIP_G17S, malformed_terminate,
                              sizeof(malformed_terminate), &decoded));
   CHECK(record_is_cleared(&decoded));
   return 0;
}

int
main(void)
{
   if (test_launch_padding_and_terminate() || test_indirect_launch_codec() ||
       test_indirect_launch_36_codec() || test_links_and_barriers() ||
       test_sequence_and_rejections())
      return 1;

   printf("agx3_cdm_decode_core: %u checks passed\n", checks);
   return 0;
}
