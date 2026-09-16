/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "../agx_ioto_state_core.h"

#include "../agx_ioto_core.h"

#include <stdio.h>
#include <string.h>

static uint64_t checks;

#define CHECK(x)                                                               \
   do {                                                                        \
      ++checks;                                                                \
      if (!(x)) {                                                              \
         fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x);                \
         return 1;                                                             \
      }                                                                        \
   } while (0)

#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))
#define FM(field) (UINT32_C(1) << (field))

static const enum agx_chip gpu0_targets[] = {
   AGX_CHIP_G15G, AGX_CHIP_G15S, AGX_CHIP_G15C, AGX_CHIP_G15D,
   AGX_CHIP_G16G, AGX_CHIP_G16S, AGX_CHIP_G16C, AGX_CHIP_G17P,
};

static const enum agx_chip gpu1_targets[] = {
   AGX_CHIP_G17G,
   AGX_CHIP_G17S,
   AGX_CHIP_G17C,
};

static const enum agx_chip unsupported_targets[] = {
   AGX_CHIP_G13G,    AGX_CHIP_G13X,    AGX_CHIP_G14G,
   AGX_CHIP_G14X,    AGX_CHIP_G15P,    AGX_CHIP_G16P,
   AGX_CHIP_G17A,    AGX_CHIP_UNKNOWN, (enum agx_chip)-1,
   (enum agx_chip)999,
};

enum test_encoding {
   TEST_GPU0,
   TEST_GPU1,
};

enum test_record {
   TEST_SPECLM,
   TEST_LDSHDR,
   TEST_SPECTPR,
};

enum test_lut {
   TEST_LUT_NONE,
   TEST_LUT_E3,
   TEST_LUT_E4,
};

struct test_slice {
   uint8_t field;
   uint8_t word;
   uint8_t destination_shift;
   uint8_t width;
   uint8_t source_shift;
   uint8_t lut;
};

struct test_form {
   size_t size;
   enum agx_ioto_kind kind;
   uint16_t fixed[5];
   const struct test_slice *slices;
   size_t slice_count;
   uint32_t selector_mask;
};

#define S(field_, word_, destination_, width_, source_)                        \
   { field_, word_, destination_, width_, source_, TEST_LUT_NONE }
#define SL(field_, word_, destination_, width_, source_, lut_)                 \
   { field_, word_, destination_, width_, source_, lut_ }

/* Literal test-side transcription of the recovered ep0 tables. */
static const struct test_slice speclm_gpu0_slices[] = {
   S(22, 1, 9, 5, 3),  S(18, 2, 8, 8, 6),  S(18, 3, 0, 2, 14),
   S(2, 0, 11, 1, 0),  S(3, 0, 12, 1, 0),  S(4, 0, 13, 1, 0),
   S(5, 0, 14, 1, 0),  S(6, 0, 15, 1, 0),  S(7, 1, 0, 1, 0),
   SL(12, 1, 1, 2, 0, TEST_LUT_E3),
   SL(11, 1, 3, 2, 0, TEST_LUT_E3),
   SL(10, 1, 5, 2, 0, TEST_LUT_E3),
   SL(21, 1, 7, 2, 0, TEST_LUT_E4), S(25, 2, 6, 1, 0),
   S(24, 2, 7, 1, 0),
};

static const struct test_slice speclm_gpu1_slices[] = {
   S(24, 0, 3, 1, 0), S(25, 0, 4, 1, 0), S(21, 3, 6, 2, 0),
   S(26, 3, 0, 1, 0), S(9, 3, 2, 1, 0),  S(8, 3, 1, 1, 0),
   S(7, 2, 15, 1, 0), S(6, 2, 14, 1, 0), S(5, 2, 13, 1, 0),
   S(4, 2, 12, 1, 0), S(3, 2, 11, 1, 0), S(2, 2, 10, 1, 0),
   S(18, 0, 5, 11, 0), S(18, 1, 3, 5, 11), S(22, 3, 8, 8, 0),
};

static const struct test_slice ldshdr_gpu0_slices[] = {
   S(0, 1, 7, 9, 6), S(0, 2, 0, 16, 15), S(0, 3, 0, 16, 31),
   S(0, 4, 0, 1, 47), S(7, 0, 11, 1, 0), S(8, 0, 12, 1, 0),
   S(9, 0, 13, 1, 0), S(10, 0, 14, 1, 0), S(11, 0, 15, 1, 0),
   S(12, 1, 0, 1, 0), SL(17, 1, 1, 2, 0, TEST_LUT_E3),
   SL(16, 1, 3, 2, 0, TEST_LUT_E3),
   SL(15, 1, 5, 2, 0, TEST_LUT_E3),
   SL(1, 4, 1, 2, 0, TEST_LUT_E4),
};

static const struct test_slice ldshdr_gpu1_slices[] = {
   S(1, 1, 3, 2, 0), S(23, 1, 6, 1, 0), S(3, 1, 12, 1, 0),
   S(24, 1, 15, 1, 0), S(14, 1, 14, 1, 0), S(13, 1, 13, 1, 0),
   S(12, 2, 15, 1, 0), S(11, 2, 14, 1, 0), S(10, 2, 13, 1, 0),
   S(9, 2, 12, 1, 0), S(8, 2, 11, 1, 0), S(7, 2, 10, 1, 0),
   S(0, 0, 6, 10, 6), S(0, 3, 0, 16, 16), S(0, 4, 0, 16, 32),
};

static const struct test_slice spectpr_gpu0_slices[] = {
   S(10, 1, 7, 7, 0), S(22, 1, 15, 1, 1), S(22, 2, 0, 10, 2),
   S(20, 2, 10, 6, 0), S(20, 3, 0, 10, 6), S(21, 3, 10, 6, 0),
   S(21, 4, 0, 10, 6), S(2, 0, 11, 1, 0), S(3, 0, 12, 1, 0),
   S(4, 0, 13, 1, 0), S(5, 0, 14, 1, 0), S(6, 0, 15, 1, 0),
   S(7, 1, 0, 1, 0), SL(14, 1, 1, 2, 0, TEST_LUT_E3),
   SL(13, 1, 3, 2, 0, TEST_LUT_E3),
   SL(12, 1, 5, 2, 0, TEST_LUT_E3), S(11, 1, 14, 1, 0),
};

static const struct test_slice spectpr_gpu1_slices[] = {
   S(11, 1, 12, 1, 0), S(23, 1, 15, 1, 0), S(9, 1, 14, 1, 0),
   S(8, 1, 13, 1, 0), S(7, 2, 15, 1, 0), S(6, 2, 14, 1, 0),
   S(5, 2, 13, 1, 0), S(4, 2, 12, 1, 0), S(3, 2, 11, 1, 0),
   S(2, 2, 10, 1, 0), S(22, 0, 5, 11, 1), S(20, 3, 0, 16, 0),
   S(21, 4, 0, 16, 0), S(10, 0, 3, 2, 0), S(10, 1, 3, 5, 2),
};

#undef SL
#undef S

static const struct test_form forms[2][3] = {
   [TEST_GPU0] = {
      [TEST_SPECLM] = {
         AGX_IOTO_SPECLM_SIZE,
         AGX_IOTO_KIND_SPECLM_EP0,
         {0x0077, 0, 0, 0, 0},
         speclm_gpu0_slices,
         ARRAY_SIZE(speclm_gpu0_slices),
         FM(26),
      },
      [TEST_LDSHDR] = {
         AGX_IOTO_LDSHDR_SIZE,
         AGX_IOTO_KIND_LDSHDR_EP0,
         {0x0177, 0, 0, 0, 0},
         ldshdr_gpu0_slices,
         ARRAY_SIZE(ldshdr_gpu0_slices),
         FM(23) | FM(24),
      },
      [TEST_SPECTPR] = {
         AGX_IOTO_SPECTPR_SIZE,
         AGX_IOTO_KIND_SPECTPR_EP0,
         {0x00f7, 0, 0, 0, 0},
         spectpr_gpu0_slices,
         ARRAY_SIZE(spectpr_gpu0_slices),
         FM(23),
      },
   },
   [TEST_GPU1] = {
      [TEST_SPECLM] = {
         AGX_IOTO_SPECLM_SIZE,
         AGX_IOTO_KIND_SPECLM_EP0,
         {0x0007, 0x0c00, 0, 0, 0},
         speclm_gpu1_slices,
         ARRAY_SIZE(speclm_gpu1_slices),
         FM(0) | FM(1),
      },
      [TEST_LDSHDR] = {
         AGX_IOTO_LDSHDR_SIZE,
         AGX_IOTO_KIND_LDSHDR_EP0,
         {0x0007, 0x0e00, 0, 0, 0},
         ldshdr_gpu1_slices,
         ARRAY_SIZE(ldshdr_gpu1_slices),
         FM(5) | FM(6),
      },
      [TEST_SPECTPR] = {
         AGX_IOTO_SPECTPR_SIZE,
         AGX_IOTO_KIND_SPECTPR_EP0,
         {0x0007, 0x0d00, 0, 0, 0},
         spectpr_gpu1_slices,
         ARRAY_SIZE(spectpr_gpu1_slices),
         FM(0) | FM(1),
      },
   },
};

typedef bool (*pack_fn)(enum agx_chip, const struct agx_ioto_state *, void *,
                        size_t);

static pack_fn
record_packer(enum test_record record)
{
   switch (record) {
   case TEST_SPECLM:
      return agx_ioto_state_pack_speclm;
   case TEST_LDSHDR:
      return agx_ioto_state_pack_ldshdr;
   case TEST_SPECTPR:
      return agx_ioto_state_pack_spectpr;
   }
   return NULL;
}

static enum agx_chip
representative(enum test_encoding encoding)
{
   return encoding == TEST_GPU0 ? AGX_CHIP_G15G : AGX_CHIP_G17G;
}

static struct agx_ioto_state
canonical_state(enum test_encoding encoding, enum test_record record)
{
   struct agx_ioto_state state = {0};
   if (encoding == TEST_GPU0) {
      if (record == TEST_LDSHDR)
         state.f[23] = 1;
   } else {
      switch (record) {
      case TEST_SPECLM:
         state.f[0] = 3;
         state.f[1] = 165;
         break;
      case TEST_LDSHDR:
         state.f[5] = 3;
         state.f[6] = 107;
         break;
      case TEST_SPECTPR:
         state.f[0] = 3;
         state.f[1] = 166;
         break;
      }
   }
   return state;
}

static uint64_t
apply_lut(uint64_t value, enum test_lut lut)
{
   static const uint8_t e3[] = {1, 2, 0};
   static const uint8_t e4[] = {2, 0, 1, 3};
   switch (lut) {
   case TEST_LUT_NONE:
      return value;
   case TEST_LUT_E3:
      return e3[value];
   case TEST_LUT_E4:
      return e4[value];
   }
   return UINT64_MAX;
}

static void
reference_encode(const struct test_form *form,
                 const struct agx_ioto_state *state, uint8_t output[10])
{
   uint16_t words[5];
   memcpy(words, form->fixed, sizeof(words));

   for (size_t i = 0; i < form->slice_count; ++i) {
      const struct test_slice *slice = &form->slices[i];
      const uint64_t value =
         apply_lut(state->f[slice->field], (enum test_lut)slice->lut);
      const uint64_t mask = (UINT64_C(1) << slice->width) - 1;
      words[slice->word] |= (uint16_t)(((value >> slice->source_shift) & mask)
                                       << slice->destination_shift);
   }

   memset(output, 0, 10);
   for (size_t i = 0; i < form->size / 2; ++i) {
      output[i * 2] = (uint8_t)words[i];
      output[i * 2 + 1] = (uint8_t)(words[i] >> 8);
   }
}

static bool
bytes_are(const void *data, uint8_t value, size_t size)
{
   const uint8_t *bytes = data;
   for (size_t i = 0; i < size; ++i) {
      if (bytes[i] != value)
         return false;
   }
   return true;
}

static uint32_t
form_field_mask(const struct test_form *form)
{
   uint32_t mask = form->selector_mask;
   for (size_t i = 0; i < form->slice_count; ++i)
      mask |= FM(form->slices[i].field);
   return mask;
}

static enum test_lut
form_field_lut(const struct test_form *form, unsigned field)
{
   for (size_t i = 0; i < form->slice_count; ++i) {
      if (form->slices[i].field == field &&
          form->slices[i].lut != TEST_LUT_NONE)
         return (enum test_lut)form->slices[i].lut;
   }
   return TEST_LUT_NONE;
}

static uint64_t
form_field_raw_mask(const struct test_form *form, unsigned field)
{
   uint64_t mask = 0;
   for (size_t i = 0; i < form->slice_count; ++i) {
      const struct test_slice *slice = &form->slices[i];
      if (slice->field != field || slice->lut != TEST_LUT_NONE)
         continue;
      mask |= ((UINT64_C(1) << slice->width) - 1) << slice->source_shift;
   }
   return mask;
}

static int
check_success(enum test_encoding encoding, enum test_record record,
              const struct agx_ioto_state *state)
{
   const struct test_form *form = &forms[encoding][record];
   uint8_t expected[10];
   reference_encode(form, state, expected);

   uint8_t actual[12];
   memset(actual, 0xa5, sizeof(actual));
   CHECK(record_packer(record)(representative(encoding), state, &actual[1],
                               sizeof(actual) - 1));
   CHECK(actual[0] == 0xa5);
   CHECK(memcmp(&actual[1], expected, form->size) == 0);
   CHECK(bytes_are(&actual[1 + form->size], 0xa5,
                   sizeof(actual) - 1 - form->size));

   struct agx_ioto_record decoded;
   size_t consumed = 0;
   CHECK(agx_ioto_validate_kind(representative(encoding), form->kind,
                                &actual[1], form->size, &decoded, &consumed));
   CHECK(consumed == form->size);
   return 0;
}

static int
check_rejected(enum test_encoding encoding, enum test_record record,
               const struct agx_ioto_state *state)
{
   uint8_t output[12];
   memset(output, 0xa5, sizeof(output));
   CHECK(!record_packer(record)(representative(encoding), state, output,
                                sizeof(output)));
   CHECK(bytes_are(output, 0xa5, sizeof(output)));
   return 0;
}

static int
test_one_hot_scatter_and_widths(void)
{
   for (unsigned encoding = TEST_GPU0; encoding <= TEST_GPU1; ++encoding) {
      for (unsigned record = TEST_SPECLM; record <= TEST_SPECTPR; ++record) {
         const struct test_form *form = &forms[encoding][record];
         const uint32_t field_mask = form_field_mask(form);
         struct agx_ioto_state base = canonical_state(
            (enum test_encoding)encoding, (enum test_record)record);
         CHECK(check_success((enum test_encoding)encoding,
                             (enum test_record)record, &base) == 0);

         for (unsigned field = 0; field < AGX_IOTO_STATE_FIELD_COUNT;
              ++field) {
            if ((field_mask & FM(field)) == 0) {
               struct agx_ioto_state invalid = base;
               invalid.f[field] = 1;
               CHECK(check_rejected((enum test_encoding)encoding,
                                     (enum test_record)record,
                                     &invalid) == 0);
               continue;
            }

            const enum test_lut lut = form_field_lut(form, field);
            if (lut != TEST_LUT_NONE) {
               const uint64_t limit = lut == TEST_LUT_E3 ? 3 : 4;
               for (uint64_t value = 0; value < limit; ++value) {
                  struct agx_ioto_state state = base;
                  state.f[field] = value;
                  CHECK(check_success((enum test_encoding)encoding,
                                      (enum test_record)record, &state) == 0);
               }
               struct agx_ioto_state invalid = base;
               invalid.f[field] = limit;
               CHECK(check_rejected((enum test_encoding)encoding,
                                     (enum test_record)record,
                                     &invalid) == 0);
               continue;
            }

            const uint64_t raw_mask = form_field_raw_mask(form, field);
            if (raw_mask == 0)
               continue; /* Exact selector, covered separately. */

            for (unsigned bit = 0; bit < 64; ++bit) {
               const uint64_t value = UINT64_C(1) << bit;
               if ((raw_mask & value) == 0)
                  continue;
               struct agx_ioto_state state = base;
               state.f[field] = value;
               CHECK(check_success((enum test_encoding)encoding,
                                   (enum test_record)record, &state) == 0);
            }

            struct agx_ioto_state maximum = base;
            maximum.f[field] = raw_mask;
            CHECK(check_success((enum test_encoding)encoding,
                                (enum test_record)record, &maximum) == 0);

            const uint64_t low_unencoded = raw_mask & (0 - raw_mask);
            if (low_unencoded > 1) {
               struct agx_ioto_state misaligned = base;
               misaligned.f[field] = 1;
               CHECK(check_rejected((enum test_encoding)encoding,
                                     (enum test_record)record,
                                     &misaligned) == 0);
            }

            unsigned top = 63;
            while ((raw_mask & (UINT64_C(1) << top)) == 0)
               --top;
            if (top < 63) {
               struct agx_ioto_state too_wide = base;
               too_wide.f[field] = UINT64_C(1) << (top + 1);
               CHECK(check_rejected((enum test_encoding)encoding,
                                     (enum test_record)record,
                                     &too_wide) == 0);
            }
         }
      }
   }
   return 0;
}

static int
test_selectors(void)
{
   struct agx_ioto_state state = canonical_state(TEST_GPU0, TEST_SPECLM);
   state.f[26] = 1;
   CHECK(check_rejected(TEST_GPU0, TEST_SPECLM, &state) == 0);

   state = canonical_state(TEST_GPU0, TEST_LDSHDR);
   state.f[23] = 0;
   CHECK(check_rejected(TEST_GPU0, TEST_LDSHDR, &state) == 0);
   state = canonical_state(TEST_GPU0, TEST_LDSHDR);
   state.f[24] = 1;
   CHECK(check_rejected(TEST_GPU0, TEST_LDSHDR, &state) == 0);

   state = canonical_state(TEST_GPU0, TEST_SPECTPR);
   state.f[23] = 1;
   CHECK(check_rejected(TEST_GPU0, TEST_SPECTPR, &state) == 0);

   state = canonical_state(TEST_GPU1, TEST_SPECLM);
   state.f[0] = 2;
   CHECK(check_rejected(TEST_GPU1, TEST_SPECLM, &state) == 0);
   state = canonical_state(TEST_GPU1, TEST_SPECLM);
   state.f[1] = 164;
   CHECK(check_rejected(TEST_GPU1, TEST_SPECLM, &state) == 0);

   state = canonical_state(TEST_GPU1, TEST_LDSHDR);
   state.f[5] = 2;
   CHECK(check_rejected(TEST_GPU1, TEST_LDSHDR, &state) == 0);
   state = canonical_state(TEST_GPU1, TEST_LDSHDR);
   state.f[6] = 106;
   CHECK(check_rejected(TEST_GPU1, TEST_LDSHDR, &state) == 0);

   state = canonical_state(TEST_GPU1, TEST_SPECTPR);
   state.f[0] = 2;
   CHECK(check_rejected(TEST_GPU1, TEST_SPECTPR, &state) == 0);
   state = canonical_state(TEST_GPU1, TEST_SPECTPR);
   state.f[1] = 165;
   CHECK(check_rejected(TEST_GPU1, TEST_SPECTPR, &state) == 0);
   return 0;
}

static int
test_targets_capacity_and_nulls(void)
{
   for (unsigned encoding = TEST_GPU0; encoding <= TEST_GPU1; ++encoding) {
      const enum agx_chip *targets =
         encoding == TEST_GPU0 ? gpu0_targets : gpu1_targets;
      const size_t target_count = encoding == TEST_GPU0
                                     ? ARRAY_SIZE(gpu0_targets)
                                     : ARRAY_SIZE(gpu1_targets);

      for (unsigned record = TEST_SPECLM; record <= TEST_SPECTPR; ++record) {
         const struct test_form *form = &forms[encoding][record];
         const struct agx_ioto_state state = canonical_state(
            (enum test_encoding)encoding, (enum test_record)record);
         uint8_t expected[10];
         reference_encode(form, &state, expected);

         for (size_t t = 0; t < target_count; ++t) {
            uint8_t output[12];
            memset(output, 0xa5, sizeof(output));
            CHECK(record_packer((enum test_record)record)(
               targets[t], &state, output, sizeof(output)));
            CHECK(memcmp(output, expected, form->size) == 0);
            CHECK(bytes_are(output + form->size, 0xa5,
                            sizeof(output) - form->size));
         }

         uint8_t output[12];
         memset(output, 0xa5, sizeof(output));
         CHECK(!record_packer((enum test_record)record)(
            representative((enum test_encoding)encoding), &state, output,
            form->size - 1));
         CHECK(bytes_are(output, 0xa5, sizeof(output)));

         memset(output, 0xa5, sizeof(output));
         CHECK(!record_packer((enum test_record)record)(
            representative((enum test_encoding)encoding), NULL, output,
            sizeof(output)));
         CHECK(bytes_are(output, 0xa5, sizeof(output)));
         CHECK(!record_packer((enum test_record)record)(
            representative((enum test_encoding)encoding), &state, NULL,
            form->size));
      }
   }

   for (size_t t = 0; t < ARRAY_SIZE(unsupported_targets); ++t) {
      for (unsigned record = TEST_SPECLM; record <= TEST_SPECTPR; ++record) {
         const struct agx_ioto_state state = canonical_state(
            TEST_GPU0, (enum test_record)record);
         uint8_t output[12];
         memset(output, 0xa5, sizeof(output));
         CHECK(!record_packer((enum test_record)record)(
            unsupported_targets[t], &state, output, sizeof(output)));
         CHECK(bytes_are(output, 0xa5, sizeof(output)));
      }
   }
   return 0;
}

static struct agx_ioto_state_tail
canonical_tail(enum test_encoding encoding, enum agx_ioto_state_tail_type type)
{
   struct agx_ioto_state_tail tail = {
      .type = type,
      .ldshdr = canonical_state(encoding, TEST_LDSHDR),
      .spectpr = canonical_state(encoding, TEST_SPECTPR),
   };
   if (type == AGX_IOTO_STATE_TAIL_TYPE0)
      tail.speclm = canonical_state(encoding, TEST_SPECLM);

   tail.ldshdr.f[0] = UINT64_C(0x123456780);
   return tail;
}

static void
reference_tail(enum test_encoding encoding,
               const struct agx_ioto_state_tail *tail,
               struct agx_ioto_state_tail_output *output)
{
   memset(output, 0, sizeof(*output));
   size_t offset = 0;
   if (tail->type == AGX_IOTO_STATE_TAIL_TYPE0) {
      reference_encode(&forms[encoding][TEST_SPECLM], &tail->speclm,
                       output->bytes + offset);
      offset += AGX_IOTO_SPECLM_SIZE;
   }
   reference_encode(&forms[encoding][TEST_LDSHDR], &tail->ldshdr,
                    output->bytes + offset);
   offset += AGX_IOTO_LDSHDR_SIZE;
   reference_encode(&forms[encoding][TEST_SPECTPR], &tail->spectpr,
                    output->bytes + offset);
   offset += AGX_IOTO_SPECTPR_SIZE;
   output->length = offset;
}

static bool
tail_outputs_equal(const struct agx_ioto_state_tail_output *a,
                   const struct agx_ioto_state_tail_output *b)
{
   return a->length == b->length &&
          memcmp(a->bytes, b->bytes, sizeof(a->bytes)) == 0;
}

static int
test_tail_types_order_and_targets(void)
{
   for (unsigned encoding = TEST_GPU0; encoding <= TEST_GPU1; ++encoding) {
      const enum agx_chip *targets =
         encoding == TEST_GPU0 ? gpu0_targets : gpu1_targets;
      const size_t target_count = encoding == TEST_GPU0
                                     ? ARRAY_SIZE(gpu0_targets)
                                     : ARRAY_SIZE(gpu1_targets);

      for (unsigned type = AGX_IOTO_STATE_TAIL_TYPE0;
           type <= AGX_IOTO_STATE_TAIL_TYPE1; ++type) {
         const struct agx_ioto_state_tail tail = canonical_tail(
            (enum test_encoding)encoding,
            (enum agx_ioto_state_tail_type)type);
         struct agx_ioto_state_tail_output expected;
         reference_tail((enum test_encoding)encoding, &tail, &expected);

         for (size_t t = 0; t < target_count; ++t) {
            struct agx_ioto_state_tail_output output;
            memset(&output, 0xa5, sizeof(output));
            CHECK(agx_ioto_state_tail_pack(targets[t], &tail, &output));
            CHECK(tail_outputs_equal(&output, &expected));
            CHECK(output.length ==
                  (type == AGX_IOTO_STATE_TAIL_TYPE0
                      ? AGX_IOTO_STATE_TAIL_TYPE0_SIZE
                      : AGX_IOTO_STATE_TAIL_TYPE1_SIZE));

            size_t offset = 0;
            struct agx_ioto_record record;
            size_t consumed = 0;
            if (type == AGX_IOTO_STATE_TAIL_TYPE0) {
               CHECK(agx_ioto_validate_kind(
                  targets[t], AGX_IOTO_KIND_SPECLM_EP0,
                  output.bytes + offset, output.length - offset, &record,
                  &consumed));
               CHECK(consumed == AGX_IOTO_SPECLM_SIZE);
               offset += consumed;
            }
            CHECK(agx_ioto_validate_kind(
               targets[t], AGX_IOTO_KIND_LDSHDR_EP0, output.bytes + offset,
               output.length - offset, &record, &consumed));
            CHECK(consumed == AGX_IOTO_LDSHDR_SIZE);
            offset += consumed;
            CHECK(agx_ioto_validate_kind(
               targets[t], AGX_IOTO_KIND_SPECTPR_EP0, output.bytes + offset,
               output.length - offset, &record, &consumed));
            CHECK(consumed == AGX_IOTO_SPECTPR_SIZE);
            offset += consumed;
            CHECK(offset == output.length);
         }
      }
   }
   return 0;
}

static int
test_tail_failures_and_shader_convenience(void)
{
   struct agx_ioto_state_tail tail =
      canonical_tail(TEST_GPU1, AGX_IOTO_STATE_TAIL_TYPE1);
   struct agx_ioto_state_tail_output output;
   memset(&output, 0xa5, sizeof(output));
   tail.speclm.f[26] = 1;
   CHECK(!agx_ioto_state_tail_pack(AGX_CHIP_G17G, &tail, &output));
   CHECK(bytes_are(&output, 0xa5, sizeof(output)));

   tail = canonical_tail(TEST_GPU0, AGX_IOTO_STATE_TAIL_TYPE0);
   tail.type = (enum agx_ioto_state_tail_type)2;
   memset(&output, 0xa5, sizeof(output));
   CHECK(!agx_ioto_state_tail_pack(AGX_CHIP_G15G, &tail, &output));
   CHECK(bytes_are(&output, 0xa5, sizeof(output)));

   tail = canonical_tail(TEST_GPU0, AGX_IOTO_STATE_TAIL_TYPE0);
   tail.spectpr.f[22] = 1;
   memset(&output, 0xa5, sizeof(output));
   CHECK(!agx_ioto_state_tail_pack(AGX_CHIP_G15G, &tail, &output));
   CHECK(bytes_are(&output, 0xa5, sizeof(output)));

   tail = canonical_tail(TEST_GPU0, AGX_IOTO_STATE_TAIL_TYPE0);
   memset(&output, 0xa5, sizeof(output));
   CHECK(!agx_ioto_state_tail_pack(AGX_CHIP_G15P, &tail, &output));
   CHECK(bytes_are(&output, 0xa5, sizeof(output)));
   CHECK(!agx_ioto_state_tail_pack(AGX_CHIP_G15G, NULL, &output));
   CHECK(!agx_ioto_state_tail_pack(AGX_CHIP_G15G, &tail, NULL));

   const uint64_t shader_va = UINT64_C(0xabcdef00);
   for (unsigned encoding = TEST_GPU0; encoding <= TEST_GPU1; ++encoding) {
      struct agx_ioto_state_tail direct = canonical_tail(
         (enum test_encoding)encoding, AGX_IOTO_STATE_TAIL_TYPE0);
      direct.ldshdr.f[0] = shader_va;
      struct agx_ioto_state_tail_output expected;
      CHECK(agx_ioto_state_tail_pack(
         representative((enum test_encoding)encoding), &direct, &expected));

      struct agx_ioto_state_tail separate = direct;
      separate.ldshdr.f[0] = 0;
      memset(&output, 0xa5, sizeof(output));
      CHECK(agx_ioto_state_tail_pack_shader(
         representative((enum test_encoding)encoding), &separate, shader_va,
         &output));
      CHECK(tail_outputs_equal(&output, &expected));

      memset(&output, 0xa5, sizeof(output));
      CHECK(!agx_ioto_state_tail_pack_shader(
         representative((enum test_encoding)encoding), &direct, shader_va,
         &output));
      CHECK(bytes_are(&output, 0xa5, sizeof(output)));

      memset(&output, 0xa5, sizeof(output));
      CHECK(!agx_ioto_state_tail_pack_shader(
         representative((enum test_encoding)encoding), &separate,
         shader_va + 1, &output));
      CHECK(bytes_are(&output, 0xa5, sizeof(output)));

      memset(&output, 0xa5, sizeof(output));
      CHECK(!agx_ioto_state_tail_pack_shader(
         representative((enum test_encoding)encoding), &separate,
         UINT64_C(0x1000000000000), &output));
      CHECK(bytes_are(&output, 0xa5, sizeof(output)));
   }
   return 0;
}

static int
test_g17s_minimal_compute_tail(void)
{
   static const uint8_t expected[AGX_IOTO_STATE_TAIL_TYPE0_SIZE] = {
      0x07, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x00,
      0xc7, 0x06, 0x40, 0x0e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
      0x07, 0x00, 0x00, 0x0d, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   struct agx_ioto_state_tail tail;
   memset(&tail, 0xa5, sizeof(tail));
   CHECK(agx_ioto_g17s_minimal_compute_tail_build(
      UINT64_C(0x100000006c0), &tail));

   struct agx_ioto_state_tail_output packed;
   CHECK(agx_ioto_state_tail_pack(AGX_CHIP_G17S, &tail, &packed));
   CHECK(packed.length == sizeof(expected));
   CHECK(memcmp(packed.bytes, expected, sizeof(expected)) == 0);

   struct agx_ioto_state_tail before = tail;
   CHECK(!agx_ioto_g17s_minimal_compute_tail_build(
      UINT64_C(0x100000006c1), &tail));
   CHECK(memcmp(&tail, &before, sizeof(tail)) == 0);
   CHECK(!agx_ioto_g17s_minimal_compute_tail_build(
      UINT64_C(0x1000000000000), &tail));
   CHECK(memcmp(&tail, &before, sizeof(tail)) == 0);
   CHECK(!agx_ioto_g17s_minimal_compute_tail_build(
      UINT64_C(0x100000006c0), NULL));
   return 0;
}

static int
test_g17p_compute_ldshdr(void)
{
   static const uint8_t zero_va_expected[AGX_IOTO_LDSHDR_SIZE] = {
      0x77, 0x01, 0x2a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x06, 0x00,
   };
   struct agx_ioto_state state;
   memset(&state, 0xa5, sizeof(state));
   CHECK(agx_ioto_g17p_compute_ldshdr_build(0, &state));
   uint8_t packed[AGX_IOTO_LDSHDR_SIZE];
   CHECK(agx_ioto_state_pack_ldshdr(AGX_CHIP_G17P, &state, packed,
                                    sizeof(packed)));
   CHECK(memcmp(packed, zero_va_expected, sizeof(packed)) == 0);

   CHECK(agx_ioto_g17p_compute_ldshdr_build(UINT64_C(0x123456780),
                                            &state));
   CHECK(state.f[0] == UINT64_C(0x123456780));
   CHECK(state.f[1] == 3 && state.f[23] == 1);
   struct agx_ioto_state before = state;
   CHECK(!agx_ioto_g17p_compute_ldshdr_build(UINT64_C(0x123456781),
                                             &state));
   CHECK(memcmp(&state, &before, sizeof(state)) == 0);
   CHECK(!agx_ioto_g17p_compute_ldshdr_build(
      UINT64_C(0x1000000000000), &state));
   CHECK(memcmp(&state, &before, sizeof(state)) == 0);
   CHECK(!agx_ioto_g17p_compute_ldshdr_build(0, NULL));
   return 0;
}

static int
test_g17p_runtime_compute_tail(void)
{
   static const struct {
      uint64_t shader_va;
      enum agx_ioto_g17p_compute_ldshdr_f1 ldshdr_f1;
      uint8_t expected[AGX_IOTO_STATE_TAIL_TYPE0_SIZE];
   } cases[] = {
      {UINT64_C(0x100000003c0),
       AGX_IOTO_G17P_COMPUTE_LDSHDR_F1_DIRECT_ADD3,
       {0x77, 0x00, 0x2a, 0x41, 0x00, 0x00, 0x00, 0x00,
        0x77, 0x01, 0xaa, 0x07, 0x00, 0x00, 0x00, 0x02, 0x04, 0x00,
        0xf7, 0x00, 0x2a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}},
      {UINT64_C(0x10000000400),
       AGX_IOTO_G17P_COMPUTE_LDSHDR_F1_GRID_HELPER,
       {0x77, 0x00, 0x2a, 0x41, 0x00, 0x00, 0x00, 0x00,
        0x77, 0x01, 0x2a, 0x08, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00,
        0xf7, 0x00, 0x2a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}},
      {UINT64_C(0x10000000740),
       AGX_IOTO_G17P_COMPUTE_LDSHDR_F1_INDIRECT_ADD3,
       {0x77, 0x00, 0x2a, 0x41, 0x00, 0x00, 0x00, 0x00,
        0x77, 0x01, 0xaa, 0x0e, 0x00, 0x00, 0x00, 0x02, 0x06, 0x00,
        0xf7, 0x00, 0x2a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}},
   };

   struct agx_ioto_state_tail_output output;
   for (unsigned i = 0; i < ARRAY_SIZE(cases); ++i) {
      memset(&output, 0xa5, sizeof(output));
      CHECK(agx_ioto_g17p_runtime_compute_tail_build(
         cases[i].shader_va, cases[i].ldshdr_f1, &output));
      CHECK(output.length == sizeof(cases[i].expected));
      CHECK(memcmp(output.bytes, cases[i].expected,
                   sizeof(cases[i].expected)) == 0);
   }

   const struct agx_ioto_state_tail_output before = output;
   CHECK(!agx_ioto_g17p_runtime_compute_tail_build(
      UINT64_C(0x100000003c1),
      AGX_IOTO_G17P_COMPUTE_LDSHDR_F1_DIRECT_ADD3, &output));
   CHECK(memcmp(&output, &before, sizeof(output)) == 0);
   CHECK(!agx_ioto_g17p_runtime_compute_tail_build(
      UINT64_C(0x100000003c0),
      (enum agx_ioto_g17p_compute_ldshdr_f1)2, &output));
   CHECK(memcmp(&output, &before, sizeof(output)) == 0);
   return 0;
}

static int
test_g17p_runtime_compute_shared_tail(void)
{
   static const struct {
      uint32_t size_B;
      uint32_t speclm_word1;
   } cases[] = {
      /* Keep the complete Vulkan-qualified 32 KiB allocation range. */
      {256, UINT32_C(0x00000480)},
      {1024, UINT32_C(0x00001080)},
      {4096, UINT32_C(0x00004080)},
      {16384, UINT32_C(0x00010080)},
      {32768, UINT32_C(0x00020080)},
   };
   const uint64_t shader_va = UINT64_C(0x100000003c0);
   struct agx_ioto_state_tail_output output;

   for (unsigned i = 0; i < ARRAY_SIZE(cases); ++i) {
      memset(&output, 0xa5, sizeof(output));
      CHECK(agx_ioto_g17p_runtime_compute_tail_build_shared(
         shader_va, AGX_IOTO_G17P_COMPUTE_LDSHDR_F1_DIRECT_ADD3,
         cases[i].size_B, &output));
      CHECK(output.length == AGX_IOTO_STATE_TAIL_TYPE0_SIZE);
      CHECK(memcmp(output.bytes, "\x77\x00\x2a\x41", 4) == 0);
      uint32_t word1;
      memcpy(&word1, output.bytes + 4, sizeof(word1));
      CHECK(word1 == cases[i].speclm_word1);
   }

   const struct agx_ioto_state_tail_output before = output;
   static const uint32_t invalid_sizes[] = {0, 64, 255, 257, 32769};
   for (unsigned i = 0; i < ARRAY_SIZE(invalid_sizes); ++i) {
      CHECK(!agx_ioto_g17p_runtime_compute_tail_build_shared(
         shader_va, AGX_IOTO_G17P_COMPUTE_LDSHDR_F1_DIRECT_ADD3,
         invalid_sizes[i], &output));
      CHECK(memcmp(&output, &before, sizeof(output)) == 0);
   }
   return 0;
}

static int
test_g17p_compiler_spectpr(void)
{
   static const uint8_t pinned_expected[AGX_IOTO_SPECTPR_SIZE] = {
      0xf7, 0x00, 0x2a, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   struct agx_ioto_state state;
   memset(&state, 0xa5, sizeof(state));
   CHECK(agx_ioto_g17p_compiler_spectpr_build(0, 8, &state));
   CHECK(state.f[20] == 0 && state.f[21] == 0 && state.f[22] == 2);
   uint8_t packed[AGX_IOTO_SPECTPR_SIZE];
   CHECK(agx_ioto_state_pack_spectpr(AGX_CHIP_G17P, &state, packed,
                                     sizeof(packed)));
   CHECK(memcmp(packed, pinned_expected, sizeof(packed)) == 0);

   CHECK(agx_ioto_g17p_compiler_spectpr_build(UINT32_C(0x3fff9),
                                               UINT32_C(0x3ff8), &state));
   CHECK(state.f[20] == UINT16_MAX && state.f[21] == UINT16_MAX);
   CHECK(state.f[22] == UINT16_C(0x0ffe));
   struct agx_ioto_state before = state;
   CHECK(!agx_ioto_g17p_compiler_spectpr_build(UINT32_MAX, 0, &state));
   CHECK(memcmp(&state, &before, sizeof(state)) == 0);
   CHECK(!agx_ioto_g17p_compiler_spectpr_build(0, UINT32_MAX, &state));
   CHECK(memcmp(&state, &before, sizeof(state)) == 0);
   CHECK(!agx_ioto_g17p_compiler_spectpr_build(0, 8, NULL));
   return 0;
}

static int
test_g17p_fragment_imageblock_speclm(void)
{
   static const struct {
      struct agx_ioto_g17p_imageblock imageblock;
      uint8_t expected[AGX_IOTO_SPECLM_SIZE];
   } cases[] = {
      {{0, 0, 32 * 32}, {0x77, 0x00, 0x2a, 0x41, 0x00, 0, 0, 0}},
      {{0, 1, 32 * 32}, {0x77, 0x00, 0x2a, 0x41, 0x00, 0, 0, 0}},
      {{8, 1, 32 * 32}, {0x77, 0x00, 0x2a, 0x43, 0x4b, 0, 0, 0}},
      {{16, 1, 32 * 32}, {0x77, 0x00, 0x2a, 0x45, 0x4b, 0, 0, 0}},
      {{24, 1, 32 * 32}, {0x77, 0x00, 0x2a, 0x47, 0x4b, 0, 0, 0}},
      {{32, 1, 32 * 16}, {0x77, 0x00, 0x2a, 0x09, 0x43, 0, 0, 0}},
      {{64, 1, 16 * 16}, {0x77, 0x00, 0x2a, 0x11, 0x41, 0, 0, 0}},
      /* Past 64 bytes the stride keeps its five-bit field. */
      {{72, 1, 16 * 16}, {0x77, 0x00, 0x2a, 0x13, 0x41, 0, 0, 0}},
      {{80, 1, 16 * 16}, {0x77, 0x00, 0x2a, 0x15, 0x41, 0, 0, 0}},
      {{120, 1, 16 * 16}, {0x77, 0x00, 0x2a, 0x1f, 0x41, 0, 0, 0}},
      {{8, 2, 32 * 32}, {0x77, 0x00, 0x2a, 0x42, 0x4b, 0, 0, 0}},
      {{16, 2, 32 * 16}, {0x77, 0x00, 0x2a, 0x04, 0x43, 0, 0, 0}},
      {{8, 4, 32 * 16}, {0x77, 0x00, 0xaa, 0x02, 0x43, 0, 0, 0}},
      {{24, 4, 16 * 16}, {0x77, 0x00, 0xaa, 0x06, 0x41, 0, 0, 0}},
   };
   uint8_t output[AGX_IOTO_SPECLM_SIZE];
   for (unsigned i = 0; i < ARRAY_SIZE(cases); ++i) {
      memset(output, 0xa5, sizeof(output));
      CHECK(agx_ioto_g17p_fragment_speclm_build(
         &cases[i].imageblock, output, sizeof(output)));
      CHECK(memcmp(output, cases[i].expected, sizeof(output)) == 0);
   }

   const uint8_t before[AGX_IOTO_SPECLM_SIZE] = {
      0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5,
   };
   static const struct agx_ioto_g17p_imageblock invalid[] = {
      {0, 3, 32 * 32},   {8, 0, 32 * 32}, {8, 3, 32 * 32},
      {8, 8, 32 * 32},
      {4, 1, 32 * 32},   {9, 1, 32 * 32}, {72, 1, 32 * 32},
      {136, 1, 16 * 16}, {8, 1, 8 * 8},   {8, 1, 64 * 64},
      {64, 4, 16 * 16},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(invalid); ++i) {
      memcpy(output, before, sizeof(output));
      CHECK(!agx_ioto_g17p_fragment_speclm_build(
         &invalid[i], output, sizeof(output)));
      CHECK(memcmp(output, before, sizeof(output)) == 0);
   }
   CHECK(!agx_ioto_g17p_fragment_speclm_build(NULL, output,
                                               sizeof(output)));
   CHECK(!agx_ioto_g17p_fragment_speclm_build(
      &cases[0].imageblock, NULL, sizeof(output)));
   CHECK(!agx_ioto_g17p_fragment_speclm_build(
      &cases[0].imageblock, output, sizeof(output) - 1));
   return 0;
}

int
main(void)
{
   CHECK(test_one_hot_scatter_and_widths() == 0);
   CHECK(test_selectors() == 0);
   CHECK(test_targets_capacity_and_nulls() == 0);
   CHECK(test_tail_types_order_and_targets() == 0);
   CHECK(test_tail_failures_and_shader_convenience() == 0);
   CHECK(test_g17s_minimal_compute_tail() == 0);
   CHECK(test_g17p_compute_ldshdr() == 0);
   CHECK(test_g17p_runtime_compute_tail() == 0);
   CHECK(test_g17p_runtime_compute_shared_tail() == 0);
   CHECK(test_g17p_compiler_spectpr() == 0);
   CHECK(test_g17p_fragment_imageblock_speclm() == 0);

   printf("agx_ioto_state_core: %llu checks passed\n",
          (unsigned long long)checks);
   return 0;
}

#undef FM
