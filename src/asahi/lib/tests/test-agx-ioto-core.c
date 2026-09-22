/* Copyright 2026 Aurora Silicon */
/* SPDX-License-Identifier: MIT */

#include "../agx_ioto_core.h"

#include <stdio.h>
#include <string.h>

static unsigned checks;

#define CHECK(x)                                                               \
   do {                                                                        \
      ++checks;                                                                \
      if (!(x)) {                                                              \
         fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x);                 \
         return 1;                                                             \
      }                                                                        \
   } while (0)

enum test_encoding {
   TEST_GPU0,
   TEST_GPU1,
};

struct expected_fixed_word {
   uint8_t word;
   uint16_t value;
   uint16_t mask;
};

struct expected_form {
   enum agx_ioto_kind kind;
   uint8_t length;
   uint8_t fixed_word_count;
   struct expected_fixed_word fixed[5];
};

#define EW(index, val, msk)                                                     \
   { .word = (index), .value = UINT16_C(val), .mask = UINT16_C(msk) }
#define EF1(k, len, a)                                                         \
   { .kind = AGX_IOTO_KIND_##k, .length = (len), .fixed_word_count = 1,        \
     .fixed = {a} }
#define EF2(k, len, a, b)                                                      \
   { .kind = AGX_IOTO_KIND_##k, .length = (len), .fixed_word_count = 2,        \
     .fixed = {a, b} }
#define EF3(k, len, a, b, c)                                                   \
   { .kind = AGX_IOTO_KIND_##k, .length = (len), .fixed_word_count = 3,        \
     .fixed = {a, b, c} }
#define EF4(k, len, a, b, c, d)                                                \
   { .kind = AGX_IOTO_KIND_##k, .length = (len), .fixed_word_count = 4,        \
     .fixed = {a, b, c, d} }
#define EF5(k, len, a, b, c, d, e)                                             \
   { .kind = AGX_IOTO_KIND_##k, .length = (len), .fixed_word_count = 5,        \
     .fixed = {a, b, c, d, e} }

/* Independent literal copy of the recovered fixed-bit grammar. */
static const struct expected_form gpu0_forms[] = {
   EF1(NOP_EP0, 2, EW(0, 0x0006, 0x001f)),
   EF2(NOP_EP1, 4, EW(0, 0x001e, 0x001f), EW(1, 0x0100, 0x0f00)),
   EF2(NOP_EP2, 6, EW(0, 0x001e, 0x001f), EW(1, 0x0200, 0x0f00)),
   EF2(NOP_EP3, 8, EW(0, 0x001e, 0x001f), EW(1, 0x0300, 0x0f00)),
   EF2(NOP_EP4, 10, EW(0, 0x001e, 0x001f), EW(1, 0x0400, 0x0f00)),
   EF2(NOP_EP5, 12, EW(0, 0x001e, 0x001f), EW(1, 0x0500, 0x0f00)),
   EF2(NOP_EP6, 14, EW(0, 0x001e, 0x001f), EW(1, 0x0600, 0x0f00)),
   EF2(NOP_EP7, 16, EW(0, 0x001e, 0x001f), EW(1, 0x0700, 0x0f00)),
   EF2(NOP_EP8, 18, EW(0, 0x001e, 0x001f), EW(1, 0x0800, 0x0f00)),
   EF2(NOP_EP9, 20, EW(0, 0x001e, 0x001f), EW(1, 0x0900, 0x0f00)),
   EF1(LDIMM_EP0, 2, EW(0, 0x0004, 0x8007)),
   EF2(LDIMM_EP1, 8, EW(0, 0x8004, 0x8007), EW(1, 0x0002, 0x0003)),
   EF2(LDIMM_EP2, 10, EW(0, 0x8004, 0x8007), EW(1, 0x0003, 0x0003)),
   EF2(COPY_EP0, 4, EW(0, 0x0003, 0x0007), EW(1, 0x0001, 0x0007)),
   EF3(COPY_EP1, 10, EW(0, 0x0003, 0x0007), EW(1, 0x0007, 0x0007),
       EW(2, 0x0002, 0x0803)),
   EF2(RLD_EP1, 8, EW(0, 0x8004, 0x8007), EW(1, 0x0001, 0x0003)),
   EF1(SPECLM_EP0, 8, EW(0, 0x0077, 0x07ff)),
   EF1(LDSHDR_EP0, 10, EW(0, 0x0177, 0x07ff)),
   EF1(SPECTPR_EP0, 10, EW(0, 0x00f7, 0x07ff)),
   EF1(IADD_EP0, 10, EW(0, 0x019f, 0x07ff)),
   EF1(B_EP0, 10, EW(0, 0x000f, 0x07ff)),
   EF1(BL_EP0, 10, EW(0, 0x008f, 0x07ff)),
   EF1(CNDCALL_EP0, 6, EW(0, 0x020f, 0x07ff)),
   EF1(CNDRET_EP0, 4, EW(0, 0x028f, 0x07ff)),
   EF1(SETPROFILECTL_EP0, 4, EW(0, 0x03f7, 0x07ff)),
};

static const struct expected_form gpu1_forms[] = {
   EF1(NOP_EP0, 2, EW(0, 0x0006, 0x001f)),
   EF2(NOP_EP1, 4, EW(0, 0x001e, 0x001f), EW(1, 0x0000, 0x0f80)),
   EF2(NOP_EP2, 6, EW(0, 0x001e, 0x001f), EW(1, 0x0100, 0x0f80)),
   EF2(NOP_EP3, 8, EW(0, 0x001e, 0x001f), EW(1, 0x0200, 0x0f80)),
   EF2(NOP_EP4, 10, EW(0, 0x001e, 0x001f), EW(1, 0x0300, 0x0f80)),
   EF2(NOP_EP5, 12, EW(0, 0x001e, 0x001f), EW(1, 0x0400, 0x0f80)),
   EF2(NOP_EP6, 14, EW(0, 0x001e, 0x001f), EW(1, 0x0500, 0x0f80)),
   EF2(NOP_EP7, 16, EW(0, 0x001e, 0x001f), EW(1, 0x0600, 0x0f80)),
   EF2(NOP_EP8, 18, EW(0, 0x001e, 0x001f), EW(1, 0x0700, 0x0f80)),
   EF2(NOP_EP9, 20, EW(0, 0x001e, 0x001f), EW(1, 0x0800, 0x0f80)),
   EF2(NOP_EP10, 22, EW(0, 0x001e, 0x001f), EW(1, 0x0900, 0x0f80)),
   EF1(LDIMM_EP0, 2, EW(0, 0x0004, 0x800f)),
   EF1(LDIMM_EP1, 2, EW(0, 0x000c, 0x800f)),
   EF2(LDIMM_EP2, 8, EW(0, 0x8004, 0x8007), EW(1, 0x0002, 0x0003)),
   EF2(LDIMM_EP3, 10, EW(0, 0x8004, 0x8007), EW(1, 0x0003, 0x0003)),
   EF2(COPY_EP0, 4, EW(0, 0x0003, 0x0007), EW(1, 0x0001, 0x0007)),
   EF3(COPY_EP1, 8, EW(0, 0x0003, 0x0007), EW(1, 0x0007, 0x0007),
       EW(2, 0x0002, 0x0803)),
   EF2(RLD_EP0, 4, EW(0, 0x8004, 0x8007), EW(1, 0x0000, 0x0003)),
   EF2(RLD_EP1, 8, EW(0, 0x8004, 0x8007), EW(1, 0x0001, 0x0003)),
   EF3(SPECLM_EP0, 8, EW(0, 0x0007, 0x0007), EW(1, 0x0c00, 0x0f07),
       EW(2, 0x0000, 0x0003)),
   EF3(LDSHDR_EP0, 10, EW(0, 0x0007, 0x0007), EW(1, 0x0e00, 0x0f07),
       EW(2, 0x0000, 0x0003)),
   EF3(SPECTPR_EP0, 10, EW(0, 0x0007, 0x0007), EW(1, 0x0d00, 0x0f07),
       EW(2, 0x0000, 0x0003)),
   EF4(IADD_EP0, 12, EW(0, 0x0007, 0x0007), EW(1, 0x0004, 0x0007),
       EW(2, 0x0001, 0x0003), EW(3, 0x0003, 0x000f)),
   EF2(B_EP1_COMPACT, 4, EW(0, 0x005e, 0x007f),
       EW(1, 0x0e10, 0x0f11)),
   EF2(B_EP1, 10, EW(0, 0x005e, 0x007f), EW(1, 0x0e11, 0x0f11)),
   EF2(BL_EP1_COMPACT, 4, EW(0, 0x007e, 0x007f),
       EW(1, 0x0e10, 0x0f11)),
   EF2(BL_EP1, 10, EW(0, 0x007e, 0x007f), EW(1, 0x0e11, 0x0f11)),
   EF4(CNDCALL_EP0, 8, EW(0, 0x0007, 0x0007), EW(1, 0x0004, 0x0007),
       EW(2, 0x0000, 0x0003), EW(3, 0x0004, 0x000f)),
   EF4(CNDRET_EP0, 8, EW(0, 0x0007, 0x0007), EW(1, 0x0004, 0x0007),
       EW(2, 0x0000, 0x0003), EW(3, 0x0005, 0x000f)),
   EF4(LDDEV_EP0, 8, EW(0, 0x0007, 0x0007), EW(1, 0x0003, 0x0007),
       EW(2, 0x0000, 0x0003), EW(3, 0x0000, 0x8006)),
   EF5(LDDEV_EP1, 10, EW(0, 0x0007, 0x0007), EW(1, 0x0003, 0x0007),
       EW(2, 0x0000, 0x0003), EW(3, 0x8000, 0x8006),
       EW(4, 0x0000, 0x0001)),
   EF5(LDDEV_EP2, 14, EW(0, 0x0007, 0x0007), EW(1, 0x0003, 0x0007),
       EW(2, 0x0000, 0x0003), EW(3, 0x8000, 0x8006),
       EW(4, 0x0001, 0x0001)),
   EF4(LDDEV_EP3, 10, EW(0, 0x0007, 0x0007), EW(1, 0x0007, 0x0007),
       EW(3, 0x0000, 0x0003), EW(4, 0x0000, 0x8006)),
   EF5(LDDEV_EP4, 12, EW(0, 0x0007, 0x0007), EW(1, 0x0007, 0x0007),
       EW(3, 0x0000, 0x0003), EW(4, 0x8000, 0x8006),
       EW(5, 0x0000, 0x0001)),
   EF5(LDDEV_EP5, 16, EW(0, 0x0007, 0x0007), EW(1, 0x0007, 0x0007),
       EW(3, 0x0000, 0x0003), EW(4, 0x8000, 0x8006),
       EW(5, 0x0001, 0x0001)),
   EF3(PIXREL_EP0, 6, EW(0, 0x0007, 0x0007), EW(1, 0x0400, 0x0f07),
       EW(2, 0x0000, 0x0003)),
   EF3(PIXWAIT_EP0, 6, EW(0, 0x0007, 0x0007), EW(1, 0x0500, 0x0f07),
       EW(2, 0x0000, 0x0003)),
   EF3(SETPROFILECTL_EP0, 6, EW(0, 0x0007, 0x0007),
       EW(1, 0x0f00, 0x0f07), EW(2, 0x0000, 0x0003)),
};

#undef EF5
#undef EF4
#undef EF3
#undef EF2
#undef EF1
#undef EW

static const enum agx_chip gpu0_targets[] = {
   AGX_CHIP_G15G, AGX_CHIP_G15S, AGX_CHIP_G15C, AGX_CHIP_G15D,
   AGX_CHIP_G16G, AGX_CHIP_G16S, AGX_CHIP_G16C, AGX_CHIP_G17P,
};

static const enum agx_chip gpu1_targets[] = {
   AGX_CHIP_G17G,
   AGX_CHIP_G17S,
   AGX_CHIP_G17C,
};

static bool
record_clear(const struct agx_ioto_record *record)
{
   if (record->kind != AGX_IOTO_KIND_INVALID || record->length != 0)
      return false;

   for (size_t i = 0; i < sizeof(record->bytes); ++i) {
      if (record->bytes[i] != 0)
         return false;
   }

   return true;
}

static bool
program_record_clear(const struct agx_ioto_program_record *record)
{
   static const struct agx_ioto_program_record zero;
   return memcmp(record, &zero, sizeof(*record)) == 0;
}

static void
make_canonical(const struct expected_form *form,
               uint8_t bytes[AGX_IOTO_MAX_RECORD_SIZE],
               uint8_t mask[AGX_IOTO_MAX_RECORD_SIZE])
{
   memset(bytes, 0, AGX_IOTO_MAX_RECORD_SIZE);
   memset(mask, 0, AGX_IOTO_MAX_RECORD_SIZE);

   for (uint8_t i = 0; i < form->fixed_word_count; ++i) {
      const struct expected_fixed_word *fixed = &form->fixed[i];
      const size_t offset = (size_t)fixed->word * 2;
      bytes[offset] = (uint8_t)fixed->value;
      bytes[offset + 1] = (uint8_t)(fixed->value >> 8);
      mask[offset] = (uint8_t)fixed->mask;
      mask[offset + 1] = (uint8_t)(fixed->mask >> 8);
   }
}

static int
check_form(enum agx_chip chip, const struct expected_form *form)
{
   uint8_t canonical[AGX_IOTO_MAX_RECORD_SIZE];
   uint8_t fixed_mask[AGX_IOTO_MAX_RECORD_SIZE];
   make_canonical(form, canonical, fixed_mask);
   CHECK(agx_ioto_kind_size(chip, form->kind) == form->length);

   uint8_t longer[AGX_IOTO_MAX_RECORD_SIZE + 3];
   memset(longer, 0xa5, sizeof(longer));
   memcpy(longer, canonical, form->length);

   struct agx_ioto_record record;
   size_t consumed = 99;
   CHECK(agx_ioto_validate_kind(chip, form->kind, longer, sizeof(longer),
                                &record, &consumed));
   CHECK(record.kind == form->kind);
   CHECK(record.length == form->length);
   CHECK(consumed == form->length);
   CHECK(memcmp(record.bytes, canonical, form->length) == 0);
   for (size_t i = form->length; i < sizeof(record.bytes); ++i)
      CHECK(record.bytes[i] == 0);

   memset(&record, 0xff, sizeof(record));
   consumed = 99;
   CHECK(agx_ioto_decode_one(chip, longer, sizeof(longer), &record,
                             &consumed));
   CHECK(record.kind == form->kind);
   CHECK(record.length == form->length);
   CHECK(consumed == form->length);
   CHECK(memcmp(record.bytes, canonical, form->length) == 0);

   memset(&record, 0xff, sizeof(record));
   consumed = 99;
   CHECK(agx_ioto_decode_one(chip, canonical, form->length, &record,
                             &consumed));
   CHECK(record.kind == form->kind);
   CHECK(consumed == form->length);

   uint8_t copied[AGX_IOTO_MAX_RECORD_SIZE + 2];
   memset(copied, 0xcc, sizeof(copied));
   CHECK(agx_ioto_record_copy(chip, &record, &copied[1],
                              sizeof(copied) - 1));
   CHECK(copied[0] == 0xcc);
   CHECK(memcmp(&copied[1], canonical, form->length) == 0);
   for (size_t i = form->length + 1; i < sizeof(copied); ++i)
      CHECK(copied[i] == 0xcc);

   uint8_t unaligned[AGX_IOTO_MAX_RECORD_SIZE + 4];
   memset(unaligned, 0x5a, sizeof(unaligned));
   memcpy(&unaligned[1], canonical, form->length);
   CHECK(agx_ioto_validate_kind(chip, form->kind, &unaligned[1],
                                form->length, &record, &consumed));
   CHECK(memcmp(record.bytes, canonical, form->length) == 0);

   for (size_t truncated = 0; truncated < form->length; ++truncated) {
      memset(&record, 0xff, sizeof(record));
      consumed = 99;
      CHECK(!agx_ioto_validate_kind(chip, form->kind, canonical, truncated,
                                    &record, &consumed));
      CHECK(record_clear(&record));
      CHECK(consumed == 0);

      memset(&record, 0xff, sizeof(record));
      consumed = 99;
      CHECK(!agx_ioto_decode_one(chip, canonical, truncated, &record,
                                 &consumed));
      CHECK(record_clear(&record));
      CHECK(consumed == 0);
   }

   for (size_t byte = 0; byte < form->length; ++byte) {
      for (unsigned bit = 0; bit < 8; ++bit) {
         uint8_t mutated[AGX_IOTO_MAX_RECORD_SIZE];
         memcpy(mutated, canonical, sizeof(mutated));
         mutated[byte] ^= (uint8_t)(1u << bit);

         memset(&record, 0xff, sizeof(record));
         consumed = 99;
         const bool fixed = (fixed_mask[byte] & (1u << bit)) != 0;
         const bool valid = agx_ioto_validate_kind(
            chip, form->kind, mutated, form->length, &record, &consumed);
         CHECK(valid == !fixed);

         if (fixed) {
            CHECK(record_clear(&record));
            CHECK(consumed == 0);
         } else {
            CHECK(consumed == form->length);
            CHECK(record.length == form->length);
            CHECK(memcmp(record.bytes, mutated, form->length) == 0);
         }
      }
   }

   return 0;
}

static int
check_stream_pairs(enum agx_chip chip, const struct expected_form *forms,
                   size_t form_count)
{
   for (size_t first = 0; first < form_count; ++first) {
      for (size_t second = 0; second < form_count; ++second) {
         uint8_t stream[AGX_IOTO_MAX_RECORD_SIZE * 2];
         uint8_t ignored_mask[AGX_IOTO_MAX_RECORD_SIZE];
         make_canonical(&forms[first], stream, ignored_mask);
         make_canonical(&forms[second], stream + forms[first].length,
                        ignored_mask);
         const size_t total = forms[first].length + forms[second].length;

         struct agx_ioto_record record;
         size_t consumed = 99;
         CHECK(agx_ioto_decode_one(chip, stream, total, &record, &consumed));
         CHECK(record.kind == forms[first].kind);
         CHECK(consumed == forms[first].length);

         const size_t first_length = consumed;
         consumed = 99;
         CHECK(agx_ioto_decode_one(chip, stream + first_length,
                                   total - first_length, &record, &consumed));
         CHECK(record.kind == forms[second].kind);
         CHECK(consumed == forms[second].length);
      }
   }

   return 0;
}

static int
check_form_set(const enum agx_chip *targets, size_t target_count,
               const struct expected_form *forms, size_t form_count)
{
   for (size_t t = 0; t < target_count; ++t) {
      CHECK(agx_ioto_target_supported(targets[t]));
      for (size_t f = 0; f < form_count; ++f) {
         if (check_form(targets[t], &forms[f]))
            return 1;
      }
      if (check_stream_pairs(targets[t], forms, form_count))
         return 1;

      for (enum agx_ioto_kind kind = AGX_IOTO_KIND_INVALID;
           kind < AGX_IOTO_KIND_COUNT; ++kind) {
         bool listed = false;
         for (size_t f = 0; f < form_count; ++f)
            listed |= forms[f].kind == kind;
         CHECK((agx_ioto_kind_size(targets[t], kind) != 0) == listed);
      }
   }

   return 0;
}

struct corpus_vector {
   enum agx_chip chip;
   enum agx_ioto_kind kind;
   uint8_t length;
   uint8_t bytes[AGX_IOTO_MAX_RECORD_SIZE];
};

static const struct corpus_vector corpus_vectors[] = {
   {AGX_CHIP_G15G, AGX_IOTO_KIND_B_EP0, 10,
    {0x0f, 0x00, 0x54, 0x10, 0xfd, 0xff, 0xff, 0xff, 0xff, 0x00}},
   {AGX_CHIP_G16G, AGX_IOTO_KIND_B_EP0, 10,
    {0x0f, 0x00, 0x54, 0x48, 0xfd, 0xff, 0xff, 0xff, 0xff, 0x00}},
   {AGX_CHIP_G15G, AGX_IOTO_KIND_COPY_EP0, 4, {0x1b, 0x0c, 0x01, 0x08}},
   {AGX_CHIP_G15G, AGX_IOTO_KIND_COPY_EP1, 10,
    {0x03, 0x00, 0x07, 0x00, 0x02, 0x00, 0x00, 0x00, 0x60, 0x00}},
   {AGX_CHIP_G15G, AGX_IOTO_KIND_LDIMM_EP0, 2, {0x0c, 0x20}},
   {AGX_CHIP_G15G, AGX_IOTO_KIND_LDIMM_EP1, 8,
    {0x0c, 0x80, 0x22, 0x3e, 0x00, 0x00, 0x00, 0x08}},
   {AGX_CHIP_G16G, AGX_IOTO_KIND_LDIMM_EP2, 10,
    {0x2c, 0x80, 0x23, 0x3e, 0x00, 0x00, 0x00, 0x0c, 0x40, 0x40}},
   {AGX_CHIP_G16G, AGX_IOTO_KIND_NOP_EP0, 2, {0x06, 0x00}},
   {AGX_CHIP_G15G, AGX_IOTO_KIND_RLD_EP1, 8,
    {0x2c, 0x8c, 0x09, 0x27, 0x60, 0x00, 0x00, 0x80}},
   {AGX_CHIP_G15G, AGX_IOTO_KIND_IADD_EP0, 10,
    {0x9f, 0x11, 0x54, 0x00, 0x02, 0x00, 0x08, 0xa8, 0x10, 0x05}},
   {AGX_CHIP_G17G, AGX_IOTO_KIND_B_EP1, 10,
    {0x5e, 0x0f, 0xf3, 0x1e, 0x1f, 0x00, 0xff, 0xc7, 0xff, 0x7f}},
   {AGX_CHIP_G17G, AGX_IOTO_KIND_COPY_EP1, 8,
    {0x23, 0x00, 0x07, 0x02, 0x42, 0x20, 0xa0, 0x02}},
   {AGX_CHIP_G17G, AGX_IOTO_KIND_LDDEV_EP0, 8,
    {0x0f, 0x04, 0x03, 0x00, 0x18, 0x04, 0x90, 0x20}},
   {AGX_CHIP_G17G, AGX_IOTO_KIND_LDDEV_EP1, 10,
    {0x0f, 0x04, 0x03, 0x00, 0x18, 0x04, 0x10, 0xa0, 0x00, 0x20}},
   {AGX_CHIP_G17G, AGX_IOTO_KIND_LDDEV_EP2, 14,
    {0x0f, 0x04, 0x03, 0x00, 0x18, 0x04, 0x10, 0xb2, 0x01, 0x00, 0x00, 0x00,
     0x00, 0x01}},
   {AGX_CHIP_G17G, AGX_IOTO_KIND_LDDEV_EP4, 12,
    {0x2f, 0x04, 0x07, 0x10, 0x08, 0x00, 0xb8, 0x0a, 0x10, 0xc0, 0x00, 0x08}},
   {AGX_CHIP_G17G, AGX_IOTO_KIND_LDDEV_EP5, 16,
    {0x4f, 0x08, 0x07, 0x10, 0x08, 0x00, 0xb8, 0x0a, 0x10, 0xc0, 0x41, 0x08,
     0x80, 0x00, 0x00, 0x00}},
   {AGX_CHIP_G17G, AGX_IOTO_KIND_LDIMM_EP0, 2, {0x84, 0x01}},
   {AGX_CHIP_G17G, AGX_IOTO_KIND_LDIMM_EP2, 8,
    {0x8c, 0x80, 0x22, 0x00, 0x04, 0x00, 0x00, 0x00}},
   {AGX_CHIP_G17G, AGX_IOTO_KIND_NOP_EP0, 2, {0x06, 0x00}},
   {AGX_CHIP_G17G, AGX_IOTO_KIND_RLD_EP1, 8,
    {0x24, 0x80, 0x21, 0x02, 0x47, 0x01, 0xa0, 0x82}},
   {AGX_CHIP_G17G, AGX_IOTO_KIND_IADD_EP0, 12,
    {0x27, 0x04, 0x04, 0x1a, 0x21, 0x00, 0xa3, 0x02, 0x88, 0x80, 0x00,
     0x00}},
};

static int
check_corpus_vectors(void)
{
   for (size_t i = 0; i < sizeof(corpus_vectors) / sizeof(corpus_vectors[0]);
        ++i) {
      const struct corpus_vector *vector = &corpus_vectors[i];
      uint8_t stream[AGX_IOTO_MAX_RECORD_SIZE + 2];
      memset(stream, 0x5a, sizeof(stream));
      memcpy(stream, vector->bytes, vector->length);

      struct agx_ioto_record record;
      size_t consumed;
      CHECK(agx_ioto_validate_kind(vector->chip, vector->kind, stream,
                                   sizeof(stream), &record, &consumed));
      CHECK(consumed == vector->length);
      CHECK(record.length == vector->length);
      CHECK(memcmp(record.bytes, vector->bytes, vector->length) == 0);
      CHECK(agx_ioto_decode_one(vector->chip, stream, sizeof(stream), &record,
                                &consumed));
      CHECK(record.kind == vector->kind);
      CHECK(record.length == vector->length);
      CHECK(memcmp(record.bytes, vector->bytes, vector->length) == 0);
   }

   return 0;
}

static int
check_failures(void)
{
   static const enum agx_chip rejected[] = {
      AGX_CHIP_G13G, AGX_CHIP_G13X, AGX_CHIP_G14G, AGX_CHIP_G14X,
      AGX_CHIP_G15P, AGX_CHIP_G16P, AGX_CHIP_G17A, AGX_CHIP_UNKNOWN,
   };
   const uint8_t nop[] = {0x06, 0x00};

   for (size_t i = 0; i < sizeof(rejected) / sizeof(rejected[0]); ++i) {
      struct agx_ioto_record record;
      memset(&record, 0xff, sizeof(record));
      size_t consumed = 99;
      CHECK(!agx_ioto_target_supported(rejected[i]));
      CHECK(agx_ioto_kind_size(rejected[i], AGX_IOTO_KIND_NOP_EP0) == 0);
      CHECK(!agx_ioto_validate_kind(rejected[i], AGX_IOTO_KIND_NOP_EP0, nop,
                                    sizeof(nop), &record, &consumed));
      CHECK(record_clear(&record));
      CHECK(consumed == 0);
      memset(&record, 0xff, sizeof(record));
      consumed = 99;
      CHECK(!agx_ioto_decode_one(rejected[i], nop, sizeof(nop), &record,
                                 &consumed));
      CHECK(record_clear(&record));
      CHECK(consumed == 0);
   }

   struct agx_ioto_record record;
   size_t consumed = 99;
   memset(&record, 0xff, sizeof(record));
   CHECK(!agx_ioto_validate_kind(AGX_CHIP_G15G, AGX_IOTO_KIND_INVALID, nop,
                                 sizeof(nop), &record, &consumed));
   CHECK(record_clear(&record));
   CHECK(consumed == 0);

   CHECK(!agx_ioto_validate_kind(AGX_CHIP_G15G, AGX_IOTO_KIND_NOP_EP0,
                                 NULL, sizeof(nop), &record, &consumed));
   CHECK(record_clear(&record));
   CHECK(consumed == 0);
   CHECK(!agx_ioto_validate_kind(AGX_CHIP_G15G, AGX_IOTO_KIND_NOP_EP0,
                                 nop, sizeof(nop), NULL, &consumed));
   CHECK(consumed == 0);
   CHECK(!agx_ioto_validate_kind(AGX_CHIP_G15G, AGX_IOTO_KIND_NOP_EP0,
                                 nop, sizeof(nop), &record, NULL));

   memset(&record, 0xff, sizeof(record));
   consumed = 99;
   CHECK(!agx_ioto_decode_one(AGX_CHIP_G15G, NULL, sizeof(nop), &record,
                              &consumed));
   CHECK(record_clear(&record));
   CHECK(consumed == 0);
   CHECK(!agx_ioto_decode_one(AGX_CHIP_G15G, nop, sizeof(nop), NULL,
                              &consumed));
   CHECK(consumed == 0);
   CHECK(!agx_ioto_decode_one(AGX_CHIP_G15G, nop, sizeof(nop), &record, NULL));

   uint8_t copy[8];
   memset(copy, 0xaa, sizeof(copy));
   memset(&record, 0, sizeof(record));
   record.kind = AGX_IOTO_KIND_INVALID;
   record.length = 2;
   CHECK(!agx_ioto_record_copy(AGX_CHIP_G15G, &record, copy, sizeof(copy)));
   for (size_t i = 0; i < sizeof(copy); ++i)
      CHECK(copy[i] == 0xaa);

   memset(copy, 0xaa, sizeof(copy));
   record.kind = AGX_IOTO_KIND_NOP_EP0;
   record.length = 2;
   memcpy(record.bytes, nop, sizeof(nop));
   CHECK(!agx_ioto_record_copy(AGX_CHIP_G15G, &record, copy, 1));
   CHECK(copy[0] == 0xaa);

   memset(copy, 0xaa, sizeof(copy));
   record.length = 3;
   CHECK(!agx_ioto_record_copy(AGX_CHIP_G15G, &record, copy, sizeof(copy)));
   for (size_t i = 0; i < sizeof(copy); ++i)
      CHECK(copy[i] == 0xaa);

   CHECK(!agx_ioto_record_copy(AGX_CHIP_G15G, NULL, copy, sizeof(copy)));
   CHECK(!agx_ioto_record_copy(AGX_CHIP_G15G, &record, NULL, sizeof(copy)));

   static const uint8_t m5_lddev[] = {
      0x0f, 0x04, 0x03, 0x00, 0x18, 0x04, 0x90, 0x20,
   };
   CHECK(agx_ioto_validate_kind(AGX_CHIP_G17G, AGX_IOTO_KIND_LDDEV_EP0,
                                m5_lddev, sizeof(m5_lddev), &record,
                                &consumed));
   memset(copy, 0xaa, sizeof(copy));
   CHECK(!agx_ioto_record_copy(AGX_CHIP_G15G, &record, copy, sizeof(copy)));
   for (size_t i = 0; i < sizeof(copy); ++i)
      CHECK(copy[i] == 0xaa);

   return 0;
}

static int
check_program_records(void)
{
   static const enum agx_chip all_targets[] = {
      AGX_CHIP_G15G, AGX_CHIP_G15S, AGX_CHIP_G15C, AGX_CHIP_G15D,
      AGX_CHIP_G16G, AGX_CHIP_G16S, AGX_CHIP_G16C, AGX_CHIP_G17P,
      AGX_CHIP_G17G, AGX_CHIP_G17S, AGX_CHIP_G17C,
   };
   static const uint8_t nop[] = {0x06, 0x00, 0xa5, 0xa5};
   static const uint8_t terminate[] = {0x0e, 0x00, 0x00, 0x00, 0xa5};
   static const uint8_t inline_lddev[AGX_IOTO_GPU0_INLINE_LDDEV_SIZE] = {
      0x67, 0x00, 0x12, 0x34, 0x56, 0x78, 0x9a,
      0xbc, 0xde, 0xf0, 0x24, 0x68, 0xac, 0xe0,
   };
   /* Bytes three and four carry the release descriptor. applegpu_g17p
    * fragments consuming colour classes {0}, {1}, {2}, {0,1}, {0,2}, {1,2} and
    * {0,1,2} release 0x2f3/0x2cf/0x23f/0x2c3/0x233/0x20f/0x203, coverage
    * clears bit 0, and 0x200 is the bounded background release; decoding must
    * admit the whole family. */
   static const uint8_t g17p_pixrel[][6] = {
      {0x07, 0x02, 0x54, 0xf3, 0x02, 0x00},
      {0x07, 0x02, 0x54, 0xcf, 0x02, 0x00},
      {0x07, 0x02, 0x54, 0x3f, 0x02, 0x00},
      {0x07, 0x02, 0x54, 0xc3, 0x02, 0x00},
      {0x07, 0x02, 0x54, 0x33, 0x02, 0x00},
      {0x07, 0x02, 0x54, 0x0f, 0x02, 0x00},
      {0x07, 0x02, 0x54, 0x03, 0x02, 0x00},
      {0x07, 0x02, 0x54, 0xf2, 0x02, 0x00},
      {0x07, 0x02, 0x54, 0xce, 0x02, 0x00},
      {0x07, 0x02, 0x54, 0x3e, 0x02, 0x00},
      {0x07, 0x02, 0x54, 0xc2, 0x02, 0x00},
      {0x07, 0x02, 0x54, 0x32, 0x02, 0x00},
      {0x07, 0x02, 0x54, 0x0e, 0x02, 0x00},
      {0x07, 0x02, 0x54, 0x02, 0x02, 0x00},
      {0x07, 0x02, 0x54, 0x00, 0x02, 0x00},
   };
   /* Descriptors no program carries -- a split class field, a missing bit 1,
    * or no consumed class -- must still lose the walk rather than be silently
    * accepted as a release. */
   static const uint8_t g17p_not_pixrel[][6] = {
      {0x07, 0x02, 0x54, 0xc4, 0x02, 0x00},
      {0x07, 0x02, 0x54, 0xc7, 0x02, 0x00},
      {0x07, 0x02, 0x54, 0xf1, 0x02, 0x00},
      {0x07, 0x02, 0x54, 0xff, 0x02, 0x00},
      {0x07, 0x02, 0x54, 0xfe, 0x02, 0x00},
   };

   for (unsigned i = 0; i < sizeof(g17p_pixrel) / sizeof(g17p_pixrel[0]);
        ++i) {
      struct agx_ioto_program_record record;
      size_t consumed = 99;
      CHECK(agx_ioto_decode_program_record(
         AGX_CHIP_G17P, g17p_pixrel[i], sizeof(g17p_pixrel[i]), &record,
         &consumed));
      CHECK(record.kind == AGX_IOTO_PROGRAM_RECORD_IOTO);
      CHECK(record.ioto.kind == AGX_IOTO_KIND_PIXREL_EP0);
      CHECK(consumed == sizeof(g17p_pixrel[i]));
      CHECK(!agx_ioto_decode_program_record(
         AGX_CHIP_G16G, g17p_pixrel[i], sizeof(g17p_pixrel[i]), &record,
         &consumed));
   }

   for (unsigned i = 0;
        i < sizeof(g17p_not_pixrel) / sizeof(g17p_not_pixrel[0]); ++i) {
      struct agx_ioto_program_record record;
      size_t consumed = 99;
      CHECK(!agx_ioto_decode_program_record(AGX_CHIP_G17P, g17p_not_pixrel[i],
                                            sizeof(g17p_not_pixrel[i]),
                                            &record, &consumed));
   }

   for (size_t i = 0; i < sizeof(all_targets) / sizeof(all_targets[0]); ++i) {
      const enum agx_chip chip = all_targets[i];
      struct agx_ioto_program_record record;
      size_t consumed = 99;

      memset(&record, 0xff, sizeof(record));
      CHECK(agx_ioto_decode_program_record(chip, nop, sizeof(nop), &record,
                                           &consumed));
      CHECK(record.kind == AGX_IOTO_PROGRAM_RECORD_IOTO);
      CHECK(record.length == 2);
      CHECK(record.ioto.kind == AGX_IOTO_KIND_NOP_EP0);
      CHECK(consumed == 2);
      CHECK(memcmp(record.bytes, nop, 2) == 0);

      memset(&record, 0xff, sizeof(record));
      consumed = 99;
      CHECK(agx_ioto_decode_program_record(chip, terminate,
                                           sizeof(terminate), &record,
                                           &consumed));
      CHECK(record.kind == AGX_IOTO_PROGRAM_RECORD_TERMINATE);
      CHECK(record.length == AGX_IOTO_PROGRAM_TERMINATE_SIZE);
      CHECK(consumed == AGX_IOTO_PROGRAM_TERMINATE_SIZE);
      CHECK(memcmp(record.bytes, terminate,
                   AGX_IOTO_PROGRAM_TERMINATE_SIZE) == 0);

      for (size_t truncated = 0;
           truncated < AGX_IOTO_PROGRAM_TERMINATE_SIZE; ++truncated) {
         memset(&record, 0xff, sizeof(record));
         consumed = 99;
         CHECK(!agx_ioto_decode_program_record(chip, terminate, truncated,
                                               &record, &consumed));
         CHECK(program_record_clear(&record));
         CHECK(consumed == 0);
      }

      for (unsigned bit = 0; bit < 32; ++bit) {
         uint8_t mutated[sizeof(terminate)];
         memcpy(mutated, terminate, sizeof(mutated));
         mutated[bit / 8] ^= (uint8_t)(1u << (bit % 8));
         memset(&record, 0xff, sizeof(record));
         consumed = 99;
         const bool decoded = agx_ioto_decode_program_record(
            chip, mutated, sizeof(mutated), &record, &consumed);
         CHECK(!decoded || record.kind != AGX_IOTO_PROGRAM_RECORD_TERMINATE);
      }
   }

   for (size_t i = 0; i < sizeof(gpu0_targets) / sizeof(gpu0_targets[0]);
        ++i) {
      struct agx_ioto_program_record record;
      size_t consumed = 99;
      CHECK(agx_ioto_decode_program_record(
         gpu0_targets[i], inline_lddev, sizeof(inline_lddev), &record,
         &consumed));
      CHECK(record.kind == AGX_IOTO_PROGRAM_RECORD_GPU0_INLINE_LDDEV);
      CHECK(record.length == sizeof(inline_lddev));
      CHECK(consumed == sizeof(inline_lddev));
      CHECK(memcmp(record.bytes, inline_lddev, sizeof(inline_lddev)) == 0);

      for (size_t truncated = 0; truncated < sizeof(inline_lddev);
           ++truncated) {
         memset(&record, 0xff, sizeof(record));
         consumed = 99;
         CHECK(!agx_ioto_decode_program_record(
            gpu0_targets[i], inline_lddev, truncated, &record, &consumed));
         CHECK(program_record_clear(&record));
         CHECK(consumed == 0);
      }

      for (unsigned bit = 0; bit < 11; ++bit) {
         uint8_t mutated[sizeof(inline_lddev)];
         memcpy(mutated, inline_lddev, sizeof(mutated));
         mutated[bit / 8] ^= (uint8_t)(1u << (bit % 8));
         memset(&record, 0xff, sizeof(record));
         consumed = 99;
         const bool decoded = agx_ioto_decode_program_record(
            gpu0_targets[i], mutated, sizeof(mutated), &record, &consumed);
         CHECK(!decoded ||
               record.kind != AGX_IOTO_PROGRAM_RECORD_GPU0_INLINE_LDDEV);
      }
   }

   for (size_t i = 0; i < sizeof(gpu1_targets) / sizeof(gpu1_targets[0]);
        ++i) {
      struct agx_ioto_program_record record;
      size_t consumed = 99;
      memset(&record, 0xff, sizeof(record));
      CHECK(!agx_ioto_decode_program_record(
         gpu1_targets[i], inline_lddev, sizeof(inline_lddev), &record,
         &consumed));
      CHECK(program_record_clear(&record));
      CHECK(consumed == 0);
   }

   struct agx_ioto_program_record record;
   size_t consumed = 99;
   memset(&record, 0xff, sizeof(record));
   CHECK(!agx_ioto_decode_program_record(AGX_CHIP_G14G, nop, sizeof(nop),
                                         &record, &consumed));
   CHECK(program_record_clear(&record));
   CHECK(consumed == 0);
   CHECK(!agx_ioto_decode_program_record(AGX_CHIP_G15G, NULL, sizeof(nop),
                                         &record, &consumed));
   CHECK(program_record_clear(&record));
   CHECK(consumed == 0);
   CHECK(!agx_ioto_decode_program_record(AGX_CHIP_G15G, nop, sizeof(nop),
                                         NULL, &consumed));
   CHECK(consumed == 0);
   CHECK(!agx_ioto_decode_program_record(AGX_CHIP_G15G, nop, sizeof(nop),
                                         &record, NULL));
   return 0;
}

int
main(void)
{
   if (check_form_set(gpu0_targets,
                      sizeof(gpu0_targets) / sizeof(gpu0_targets[0]),
                      gpu0_forms, sizeof(gpu0_forms) / sizeof(gpu0_forms[0])))
      return 1;
   if (check_form_set(gpu1_targets,
                      sizeof(gpu1_targets) / sizeof(gpu1_targets[0]),
                      gpu1_forms, sizeof(gpu1_forms) / sizeof(gpu1_forms[0])))
      return 1;
   if (check_corpus_vectors() || check_failures() || check_program_records())
      return 1;

   printf("AGX3 Ioto raw core: %u checks passed\n", checks);
   return 0;
}
