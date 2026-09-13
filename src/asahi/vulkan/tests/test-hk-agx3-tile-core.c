/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_agx3_tile_core.h"

#include <stdio.h>
#include <string.h>

static unsigned checks;

#define CHECK(condition)                                                       \
   do {                                                                        \
      ++checks;                                                                \
      if (!(condition)) {                                                      \
         fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__,    \
                 #condition);                                                  \
         return 1;                                                             \
      }                                                                        \
   } while (0)

struct tile_target {
   enum agx_chip chip;
   enum agx_isa_encoding encoding;
};

static const struct tile_target supported_targets[] = {
   {AGX_CHIP_G15G, AGX_ISA_ENCODING_AGX3B},
   {AGX_CHIP_G16G, AGX_ISA_ENCODING_AGX3B},
   {AGX_CHIP_G17S, AGX_ISA_ENCODING_AGX3C},
};

static const enum agx_chip unsupported_targets[] = {
   AGX_CHIP_G13G,  AGX_CHIP_G13X, AGX_CHIP_G14G, AGX_CHIP_G14X,
   AGX_CHIP_G15P,  AGX_CHIP_G15S, AGX_CHIP_G15C, AGX_CHIP_G15D,
   AGX_CHIP_G16P,  AGX_CHIP_G16S, AGX_CHIP_G16C, AGX_CHIP_G17P,
   AGX_CHIP_G17A,  AGX_CHIP_G17G, AGX_CHIP_G17C, AGX_CHIP_UNKNOWN,
   (enum agx_chip)999,
};

static int
check_instruction(const struct hk_agx3_tile_instruction *actual,
                  const uint8_t *expected, uint8_t length)
{
   CHECK(actual->length == length);
   CHECK(memcmp(actual->bytes, expected, length) == 0);
   for (unsigned i = length; i < sizeof(actual->bytes); ++i)
      CHECK(actual->bytes[i] == 0);
   return 0;
}

static int
check_cleared(const struct hk_agx3_tile_instruction *instruction)
{
   const uint8_t *bytes = (const uint8_t *)instruction;
   for (unsigned i = 0; i < sizeof(*instruction); ++i)
      CHECK(bytes[i] == 0);
   return 0;
}

static enum agx3_tile_width
shared_width(enum hk_agx3_tile_width width)
{
   switch (width) {
   case HK_AGX3_TILE_WIDTH_8:
      return AGX3_TILE_W8;
   case HK_AGX3_TILE_WIDTH_16:
      return AGX3_TILE_W16;
   case HK_AGX3_TILE_WIDTH_32:
      return AGX3_TILE_W32;
   default:
      return (enum agx3_tile_width)99;
   }
}

static int
check_grounded_witnesses(void)
{
   static const uint8_t store_b[] = {
      0xe7, 0x06, 0x54, 0x00, 0x00, 0x00, 0x01, 0x4e, 0, 0, 0, 0,
   };
   static const uint8_t store_c[] = {
      0x27, 0x01, 0x05, 0x08, 0x21, 0x01, 0xad,
      0x02, 0x00, 0x00, 0x61, 0x5c, 0x00, 0x00,
   };
   static const uint8_t load_b[] = {
      0x67, 0x0e, 0x54, 0x00, 0x00, 0x00, 0x01, 0xce, 0x02, 0, 0, 0,
   };
   static const uint8_t load_c[] = {
      0x27, 0x01, 0x05, 0x08, 0x01, 0x87, 0xbc,
      0x02, 0x00, 0x00, 0x40, 0x5c, 0x20, 0x00,
   };

   for (unsigned t = 0;
        t < sizeof(supported_targets) / sizeof(supported_targets[0]); ++t) {
      const struct tile_target *target = &supported_targets[t];
      const bool is_c = target->encoding == AGX_ISA_ENCODING_AGX3C;
      struct hk_agx3_tile_store_desc store = {
         .width = HK_AGX3_TILE_WIDTH_8,
         .component_mask = 15,
      };
      struct hk_agx3_tile_load_desc load = {
         .width = HK_AGX3_TILE_WIDTH_8,
         .component_mask = 15,
         .slot = is_c ? 8 : 6,
         .first = true,
         .last = true,
      };
      struct hk_agx3_tile_instruction instruction;

      CHECK(hk_agx3_pack_tile_store(target->chip, &store, &instruction));
      if (check_instruction(&instruction, is_c ? store_c : store_b,
                            is_c ? sizeof(store_c) : sizeof(store_b)))
         return 1;

      CHECK(hk_agx3_pack_tile_load(target->chip, &load, &instruction));
      if (check_instruction(&instruction, is_c ? load_c : load_b,
                            is_c ? sizeof(load_c) : sizeof(load_b)))
         return 1;
   }

   return 0;
}

static int
check_shared_codec_equivalence(void)
{
   static const enum hk_agx3_tile_width widths[] = {
      HK_AGX3_TILE_WIDTH_8,
      HK_AGX3_TILE_WIDTH_16,
      HK_AGX3_TILE_WIDTH_32,
   };

   for (unsigned t = 0;
        t < sizeof(supported_targets) / sizeof(supported_targets[0]); ++t) {
      const struct tile_target *target = &supported_targets[t];
      const bool is_c = target->encoding == AGX_ISA_ENCODING_AGX3C;
      const unsigned slots = is_c ? 8 : 6;
      const unsigned store_registers = is_c ? 256 : 16;
      const unsigned load_registers = is_c ? 256 : 192;
      struct hk_agx3_tile_store_desc store = {
         .width = HK_AGX3_TILE_WIDTH_8,
         .component_mask = 15,
      };
      struct hk_agx3_tile_load_desc load = {
         .width = HK_AGX3_TILE_WIDTH_8,
         .component_mask = 15,
         .slot = (uint8_t)slots,
         .first = true,
         .last = true,
      };
      struct hk_agx3_tile_instruction instruction;
      uint8_t expected[AGX3_TILE_MAX_LENGTH];
      uint8_t length = 0;

      for (unsigned data = 0; data < store_registers; ++data) {
         store.data = (uint16_t)data;
         CHECK(agx3_tile_store(target->encoding, store.data, AGX3_TILE_W8,
                               store.component_mask, store.tile_offset,
                               AGX3_TILE_SAMPLE_1X, store.wait_mask, expected,
                               &length));
         CHECK(hk_agx3_pack_tile_store(target->chip, &store, &instruction));
         if (check_instruction(&instruction, expected, length))
            return 1;
      }
      store.data = 0;

      for (unsigned dest = 0; dest < load_registers; ++dest) {
         load.dest = (uint16_t)dest;
         CHECK(agx3_tile_load(target->encoding, load.dest, AGX3_TILE_W8,
                              load.slot, load.component_mask, load.tile_offset,
                              AGX3_TILE_SAMPLE_1X, 0, load.first, load.last,
                              expected, &length));
         CHECK(hk_agx3_pack_tile_load(target->chip, &load, &instruction));
         if (check_instruction(&instruction, expected, length))
            return 1;
      }
      load.dest = 0;

      for (unsigned w = 0; w < sizeof(widths) / sizeof(widths[0]); ++w) {
         store.width = load.width = widths[w];
         CHECK(agx3_tile_store(target->encoding, store.data,
                               shared_width(store.width), store.component_mask,
                               store.tile_offset, AGX3_TILE_SAMPLE_1X,
                               store.wait_mask, expected, &length));
         CHECK(hk_agx3_pack_tile_store(target->chip, &store, &instruction));
         if (check_instruction(&instruction, expected, length))
            return 1;
         CHECK(agx3_tile_load(target->encoding, load.dest,
                              shared_width(load.width), load.slot,
                              load.component_mask, load.tile_offset,
                              AGX3_TILE_SAMPLE_1X, 0, load.first, load.last,
                              expected, &length));
         CHECK(hk_agx3_pack_tile_load(target->chip, &load, &instruction));
         if (check_instruction(&instruction, expected, length))
            return 1;
      }
      store.width = load.width = HK_AGX3_TILE_WIDTH_8;

      for (unsigned mask = 1; mask <= 15; ++mask) {
         store.component_mask = (uint8_t)mask;
         CHECK(agx3_tile_store(target->encoding, store.data, AGX3_TILE_W8,
                               store.component_mask, store.tile_offset,
                               AGX3_TILE_SAMPLE_1X, store.wait_mask, expected,
                               &length));
         CHECK(hk_agx3_pack_tile_store(target->chip, &store, &instruction));
         if (check_instruction(&instruction, expected, length))
            return 1;
      }
      store.component_mask = 15;

      for (unsigned offset = 0; offset <= UINT8_MAX; ++offset) {
         store.tile_offset = load.tile_offset = (uint16_t)offset;
         CHECK(agx3_tile_store(target->encoding, store.data, AGX3_TILE_W8,
                               store.component_mask, store.tile_offset,
                               AGX3_TILE_SAMPLE_1X, store.wait_mask, expected,
                               &length));
         CHECK(hk_agx3_pack_tile_store(target->chip, &store, &instruction));
         if (check_instruction(&instruction, expected, length))
            return 1;
         CHECK(agx3_tile_load(target->encoding, load.dest, AGX3_TILE_W8,
                              load.slot, load.component_mask, load.tile_offset,
                              AGX3_TILE_SAMPLE_1X, 0, load.first, load.last,
                              expected, &length));
         CHECK(hk_agx3_pack_tile_load(target->chip, &load, &instruction));
         if (check_instruction(&instruction, expected, length))
            return 1;
      }
      store.tile_offset = load.tile_offset = 0;

      for (unsigned wait = 0; wait <= slots; ++wait) {
         store.wait_mask = wait ? (uint16_t)(1u << (wait - 1)) : 0;
         CHECK(agx3_tile_store(target->encoding, store.data, AGX3_TILE_W8,
                               store.component_mask, store.tile_offset,
                               AGX3_TILE_SAMPLE_1X, store.wait_mask, expected,
                               &length));
         CHECK(hk_agx3_pack_tile_store(target->chip, &store, &instruction));
         if (check_instruction(&instruction, expected, length))
            return 1;
      }
      store.wait_mask = 0;

      for (unsigned slot = 1; slot <= slots; ++slot) {
         load.slot = (uint8_t)slot;
         for (unsigned boundary = 0; boundary < 4; ++boundary) {
            load.first = (boundary & 1) != 0;
            load.last = (boundary & 2) != 0;
            CHECK(agx3_tile_load(target->encoding, load.dest, AGX3_TILE_W8,
                                 load.slot, load.component_mask,
                                 load.tile_offset, AGX3_TILE_SAMPLE_1X, 0,
                                 load.first, load.last, expected, &length));
            CHECK(hk_agx3_pack_tile_load(target->chip, &load, &instruction));
            if (check_instruction(&instruction, expected, length))
               return 1;
         }
      }
   }

   return 0;
}

static int
check_rejection(void)
{
   struct hk_agx3_tile_store_desc store = {
      .width = HK_AGX3_TILE_WIDTH_8,
      .component_mask = 15,
   };
   struct hk_agx3_tile_load_desc load = {
      .width = HK_AGX3_TILE_WIDTH_8,
      .component_mask = 15,
      .slot = 8,
      .first = true,
      .last = true,
   };
   struct hk_agx3_tile_instruction instruction;

#define REJECT(call)                                                           \
   do {                                                                        \
      memset(&instruction, 0xff, sizeof(instruction));                         \
      CHECK(!(call));                                                          \
      if (check_cleared(&instruction))                                         \
         return 1;                                                             \
   } while (0)

   for (unsigned i = 0;
        i < sizeof(unsupported_targets) / sizeof(unsupported_targets[0]); ++i) {
      REJECT(hk_agx3_pack_tile_store(unsupported_targets[i], &store,
                                     &instruction));
      REJECT(hk_agx3_pack_tile_load(unsupported_targets[i], &load,
                                    &instruction));
   }

   store.data = 16;
   REJECT(hk_agx3_pack_tile_store(AGX_CHIP_G15G, &store, &instruction));
   REJECT(hk_agx3_pack_tile_store(AGX_CHIP_G16G, &store, &instruction));
   store.data = 256;
   REJECT(hk_agx3_pack_tile_store(AGX_CHIP_G17S, &store, &instruction));
   store.data = 1;
   store.width = HK_AGX3_TILE_WIDTH_32;
   REJECT(hk_agx3_pack_tile_store(AGX_CHIP_G17S, &store, &instruction));
   store.data = 0;
   store.width = HK_AGX3_TILE_WIDTH_8;
   store.component_mask = 0;
   REJECT(hk_agx3_pack_tile_store(AGX_CHIP_G17S, &store, &instruction));
   store.component_mask = 16;
   REJECT(hk_agx3_pack_tile_store(AGX_CHIP_G17S, &store, &instruction));
   store.component_mask = 15;
   store.wait_mask = 3;
   REJECT(hk_agx3_pack_tile_store(AGX_CHIP_G17S, &store, &instruction));
   store.wait_mask = 1u << 6;
   REJECT(hk_agx3_pack_tile_store(AGX_CHIP_G15G, &store, &instruction));
   store.wait_mask = 1u << 8;
   REJECT(hk_agx3_pack_tile_store(AGX_CHIP_G17S, &store, &instruction));
   store.wait_mask = 0;
   store.tile_offset = 256;
   REJECT(hk_agx3_pack_tile_store(AGX_CHIP_G17S, &store, &instruction));
   store.tile_offset = 0;
   store.width = (enum hk_agx3_tile_width)99;
   REJECT(hk_agx3_pack_tile_store(AGX_CHIP_G17S, &store, &instruction));

   load.component_mask = 7;
   REJECT(hk_agx3_pack_tile_load(AGX_CHIP_G17S, &load, &instruction));
   load.component_mask = 15;
   load.slot = 0;
   REJECT(hk_agx3_pack_tile_load(AGX_CHIP_G17S, &load, &instruction));
   load.slot = 9;
   REJECT(hk_agx3_pack_tile_load(AGX_CHIP_G17S, &load, &instruction));
   load.slot = 8;
   load.tile_offset = 256;
   REJECT(hk_agx3_pack_tile_load(AGX_CHIP_G17S, &load, &instruction));
   load.tile_offset = 0;
   load.dest = 256;
   REJECT(hk_agx3_pack_tile_load(AGX_CHIP_G17S, &load, &instruction));
   load.dest = 1;
   load.width = HK_AGX3_TILE_WIDTH_32;
   REJECT(hk_agx3_pack_tile_load(AGX_CHIP_G17S, &load, &instruction));
   load.dest = 0;
   load.width = (enum hk_agx3_tile_width)99;
   REJECT(hk_agx3_pack_tile_load(AGX_CHIP_G17S, &load, &instruction));

   CHECK(!hk_agx3_pack_tile_store(AGX_CHIP_G17S, &store, NULL));
   CHECK(!hk_agx3_pack_tile_load(AGX_CHIP_G17S, &load, NULL));
   REJECT(hk_agx3_pack_tile_store(AGX_CHIP_G17S, NULL, &instruction));
   REJECT(hk_agx3_pack_tile_load(AGX_CHIP_G17S, NULL, &instruction));
#undef REJECT

   return 0;
}

int
main(void)
{
   if (check_grounded_witnesses() || check_shared_codec_equivalence() ||
       check_rejection())
      return 1;

   printf("Honeykrisp AGX3 tile core: %u checks passed\n", checks);
   return 0;
}
