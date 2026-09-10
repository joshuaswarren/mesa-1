/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Controlled address-space(5) witnesses from Apple metal-nt targeting G17P
 * (AGX3b) and G17S (AGX3c).
 */

#include "agx3_scs_core.h"

#include <stdio.h>
#include <string.h>

static unsigned checks;
static unsigned failures;

#define CHECK(x)                                                               \
   do {                                                                        \
      ++checks;                                                                \
      if (!(x)) {                                                              \
         fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x);          \
         ++failures;                                                           \
      }                                                                        \
   } while (0)

struct scs_vec {
   enum agx_isa_encoding enc;
   bool load;
   struct agx3_scs_desc d;
   uint8_t bytes[AGX3_SCS_MAX_LENGTH];
   uint8_t length;
};

static const struct scs_vec vectors[] = {
   {AGX_ISA_ENCODING_AGX3B, true,
    {.data = 1, .width = AGX3_MEM_W8, .slot = 6},
    {0x5f, 0x02, 0x54, 0x01, 0x00, 0x00, 0x00, 0x00, 0x80, 0x0a, 0x00,
     0x00, 0x02, 0x00}, 14},
   {AGX_ISA_ENCODING_AGX3B, true,
    {.data = 12, .width = AGX3_MEM_W16, .byte_offset = 4, .slot = 6},
    {0x5f, 0x02, 0x54, 0x0c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0a, 0x02,
     0x00, 0x02, 0x00}, 14},
   {AGX_ISA_ENCODING_AGX3B, true,
    {.data = 2, .width = AGX3_MEM_W32, .byte_offset = 8, .slot = 1},
    {0x5f, 0x02, 0x54, 0x02, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x04,
     0x00, 0x02, 0x00}, 14},
   {AGX_ISA_ENCODING_AGX3B, false, {.width = AGX3_MEM_W8},
    {0xdf, 0x02, 0x54, 0x00, 0x00, 0x00, 0x00, 0x00, 0x84, 0x00, 0x00,
     0x40, 0x00, 0x00}, 14},
   {AGX_ISA_ENCODING_AGX3B, false,
    {.width = AGX3_MEM_W16, .byte_offset = 4},
    {0xdf, 0x02, 0x54, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x80, 0x00,
     0x40, 0x00, 0x00}, 14},
   {AGX_ISA_ENCODING_AGX3B, false,
    {.width = AGX3_MEM_W32, .byte_offset = 8},
    {0xdf, 0x02, 0x54, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x00, 0x01,
     0x40, 0x00, 0x00}, 14},

   {AGX_ISA_ENCODING_AGX3C, true,
    {.data = 1, .width = AGX3_MEM_W8, .slot = 8},
    {0x27, 0x00, 0x01, 0x09, 0x32, 0x07, 0xa8, 0x02, 0x00, 0x00, 0x00,
     0x00, 0x02, 0x84, 0x00, 0x00}, 16},
   {AGX_ISA_ENCODING_AGX3C, true,
    {.data = 12, .width = AGX3_MEM_W16, .byte_offset = 4, .slot = 8},
    {0x27, 0x00, 0x01, 0x00, 0x32, 0x07, 0xa8, 0x1a, 0x00, 0x40, 0x00,
     0x00, 0x02, 0x84, 0x00, 0x00}, 16},
   {AGX_ISA_ENCODING_AGX3C, true,
    {.data = 2, .width = AGX3_MEM_W32, .byte_offset = 8, .slot = 1},
    {0x37, 0x00, 0x01, 0x02, 0x32, 0x00, 0xa8, 0x02, 0x00, 0x80, 0x00,
     0x00, 0x02, 0x84, 0x00, 0x00}, 16},
   {AGX_ISA_ENCODING_AGX3C, false, {.width = AGX3_MEM_W8},
    {0x27, 0x00, 0x01, 0x08, 0x32, 0x00, 0xa9, 0x02, 0x00, 0x00, 0x00,
     0x00, 0x02, 0x84}, 14},
   {AGX_ISA_ENCODING_AGX3C, false,
    {.width = AGX3_MEM_W16, .byte_offset = 4},
    {0x27, 0x00, 0x01, 0x00, 0x32, 0x00, 0xa9, 0x02, 0x00, 0x40, 0x00,
     0x00, 0x02, 0x84}, 14},
   {AGX_ISA_ENCODING_AGX3C, false,
    {.data = 2, .width = AGX3_MEM_W32, .byte_offset = 8},
    {0x37, 0x00, 0x01, 0x02, 0x32, 0x00, 0xa9, 0x02, 0x00, 0x80, 0x00,
     0x00, 0x02, 0x84}, 14},

   {AGX_ISA_ENCODING_AGX3B, true,
    {.data = 6, .width = AGX3_MEM_W32, .slot = 6},
    {0x5f, 0x02, 0x54, 0x06, 0x00, 0x00, 0x00, 0x00, 0x40, 0x0a, 0x00,
     0x00, 0x02, 0x00}, 14},
   {AGX_ISA_ENCODING_AGX3B, true,
    {.data = 8, .width = AGX3_MEM_W32, .byte_offset = 4, .slot = 6},
    {0x5f, 0x02, 0x54, 0x08, 0x00, 0x00, 0x00, 0x00, 0x40, 0x0a, 0x02,
     0x00, 0x02, 0x00}, 14},
   {AGX_ISA_ENCODING_AGX3B, true,
    {.data = 4, .width = AGX3_MEM_W32, .byte_offset = 16, .slot = 1},
    {0x5f, 0x02, 0x54, 0x04, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x08,
     0x00, 0x02, 0x00}, 14},
   {AGX_ISA_ENCODING_AGX3B, true,
    {.data = 2, .width = AGX3_MEM_W32, .byte_offset = 0x6000, .slot = 2},
    {0x5f, 0x02, 0x54, 0x02, 0x00, 0x00, 0x00, 0x00, 0x40, 0x02, 0x00,
     0x30, 0x02, 0x00}, 14},
   {AGX_ISA_ENCODING_AGX3B, false, {.width = AGX3_MEM_W32},
    {0xdf, 0x02, 0x54, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x00, 0x00,
     0x40, 0x00, 0x00}, 14},
   {AGX_ISA_ENCODING_AGX3B, false,
    {.width = AGX3_MEM_W32, .byte_offset = 4},
    {0xdf, 0x02, 0x54, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x80, 0x00,
     0x40, 0x00, 0x00}, 14},
   {AGX_ISA_ENCODING_AGX3B, false,
    {.width = AGX3_MEM_W32, .byte_offset = 16},
    {0xdf, 0x02, 0x54, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x00, 0x02,
     0x40, 0x00, 0x00}, 14},
   {AGX_ISA_ENCODING_AGX3B, false,
    {.width = AGX3_MEM_W32, .byte_offset = 0x6000},
    {0xdf, 0x02, 0x54, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x00, 0x00,
     0x4c, 0x00, 0x00}, 14},

   {AGX_ISA_ENCODING_AGX3C, true,
    {.data = 6, .width = AGX3_MEM_W32, .slot = 8},
    {0x37, 0x00, 0x01, 0x02, 0x32, 0x07, 0xa8, 0x0a, 0x00, 0x00, 0x00,
     0x00, 0x02, 0x84, 0x00, 0x00}, 16},
   {AGX_ISA_ENCODING_AGX3C, true,
    {.data = 8, .width = AGX3_MEM_W32, .byte_offset = 4, .slot = 8},
    {0x27, 0x00, 0x01, 0x02, 0x32, 0x07, 0xa8, 0x12, 0x00, 0x40, 0x00,
     0x00, 0x02, 0x84, 0x00, 0x00}, 16},
   {AGX_ISA_ENCODING_AGX3C, true,
    {.data = 4, .width = AGX3_MEM_W32, .byte_offset = 16, .slot = 1},
    {0x27, 0x00, 0x01, 0x02, 0x32, 0x00, 0xa8, 0x0a, 0x10, 0x00, 0x00,
     0x00, 0x02, 0x84, 0x00, 0x00}, 16},
   {AGX_ISA_ENCODING_AGX3C, true,
    {.data = 2, .width = AGX3_MEM_W32, .byte_offset = 0x6000, .slot = 2,
     .c_load_marker = true},
    {0x37, 0x00, 0x01, 0x22, 0x12, 0x01, 0xa8, 0x02, 0x00, 0x00, 0x00,
     0x00, 0x02, 0x84, 0x00, 0x20}, 16},
   {AGX_ISA_ENCODING_AGX3C, false, {.width = AGX3_MEM_W32},
    {0x27, 0x00, 0x01, 0x02, 0x32, 0x00, 0xa9, 0x02, 0x00, 0x00, 0x00,
     0x00, 0x02, 0x84}, 14},
   {AGX_ISA_ENCODING_AGX3C, false,
    {.width = AGX3_MEM_W32, .byte_offset = 4},
    {0x27, 0x00, 0x01, 0x02, 0x32, 0x00, 0xa9, 0x02, 0x00, 0x40, 0x00,
     0x00, 0x02, 0x84}, 14},
   {AGX_ISA_ENCODING_AGX3C, false,
    {.width = AGX3_MEM_W32, .byte_offset = 16},
    {0x27, 0x00, 0x01, 0x02, 0x32, 0x00, 0xa9, 0x02, 0x10, 0x00, 0x00,
     0x00, 0x02, 0x84}, 14},
   {AGX_ISA_ENCODING_AGX3C, false,
    {.width = AGX3_MEM_W32, .byte_offset = 0x6000, .value_kill = true},
    {0x27, 0x00, 0x01, 0x22, 0x32, 0x04, 0xa9, 0x02, 0x00, 0x00, 0x01,
     0x00, 0x02, 0x84}, 14},

   /* Controlled uniform-base + register-index probes.  The Apple compiler
    * placed the uniform pointer in half4 and the index in half0 or half2.
    */
   {AGX_ISA_ENCODING_AGX3B, true,
    {.data = 2,
     .width = AGX3_MEM_W8,
     .address_kind = AGX3_SCS_ADDRESS_UNIFORM_INDEXED,
     .uniform_base = 4,
     .slot = 6,
     .wait_mask = 1},
    {0x5f, 0x12, 0x54, 0x02, 0x04, 0x00, 0x20, 0x00, 0x80, 0x1b, 0x00,
     0x00, 0x42, 0x00}, 14},
   {AGX_ISA_ENCODING_AGX3B, false,
    {.width = AGX3_MEM_W8,
     .address_kind = AGX3_SCS_ADDRESS_UNIFORM_INDEXED,
     .uniform_base = 4},
    {0xdf, 0x02, 0x54, 0x00, 0x04, 0x00, 0x20, 0x00, 0x84, 0x05, 0x00,
     0x40, 0x08, 0x00}, 14},
   {AGX_ISA_ENCODING_AGX3B, true,
    {.data = 2,
     .width = AGX3_MEM_W16,
     .address_kind = AGX3_SCS_ADDRESS_UNIFORM_INDEXED,
     .uniform_base = 4,
     .slot = 6,
     .wait_mask = 1},
    {0x5f, 0x12, 0x54, 0x02, 0x04, 0x00, 0x20, 0x00, 0x00, 0x1b, 0x00,
     0x00, 0x44, 0x00}, 14},
   {AGX_ISA_ENCODING_AGX3B, false,
    {.width = AGX3_MEM_W16,
     .address_kind = AGX3_SCS_ADDRESS_UNIFORM_INDEXED,
     .uniform_base = 4},
    {0xdf, 0x02, 0x54, 0x00, 0x04, 0x00, 0x20, 0x00, 0x04, 0x05, 0x00,
     0x80, 0x08, 0x00}, 14},
   {AGX_ISA_ENCODING_AGX3B, true,
    {.data = 2,
     .width = AGX3_MEM_W32,
     .address_kind = AGX3_SCS_ADDRESS_UNIFORM_INDEXED,
     .uniform_base = 4,
     .slot = 6,
     .wait_mask = 1},
    {0x5f, 0x12, 0x54, 0x02, 0x04, 0x00, 0x20, 0x00, 0x40, 0x1b, 0x00,
     0x00, 0x46, 0x00}, 14},
   {AGX_ISA_ENCODING_AGX3B, false,
    {.width = AGX3_MEM_W32,
     .address_kind = AGX3_SCS_ADDRESS_UNIFORM_INDEXED,
     .uniform_base = 4},
    {0xdf, 0x02, 0x54, 0x00, 0x04, 0x00, 0x20, 0x00, 0x44, 0x05, 0x00,
     0xc0, 0x08, 0x00}, 14},
   {AGX_ISA_ENCODING_AGX3B, true,
    {.data = 2,
     .width = AGX3_MEM_W32,
     .byte_offset = 16,
     .address_kind = AGX3_SCS_ADDRESS_UNIFORM_INDEXED,
     .uniform_base = 4,
     .slot = 6,
     .wait_mask = 1},
    {0x5f, 0x12, 0x54, 0x02, 0x04, 0x00, 0x20, 0x00, 0x40, 0x1b, 0x08,
     0x00, 0x46, 0x00}, 14},
   {AGX_ISA_ENCODING_AGX3B, false,
    {.width = AGX3_MEM_W32,
     .byte_offset = 16,
     .address_kind = AGX3_SCS_ADDRESS_UNIFORM_INDEXED,
     .uniform_base = 4},
    {0xdf, 0x02, 0x54, 0x00, 0x04, 0x00, 0x20, 0x00, 0x44, 0x05, 0x02,
     0xc0, 0x08, 0x00}, 14},
   {AGX_ISA_ENCODING_AGX3B, true,
    {.width = AGX3_MEM_W8,
     .address_kind = AGX3_SCS_ADDRESS_UNIFORM_INDEXED,
     .index = 2,
     .uniform_base = 4,
     .slot = 6,
     .wait_mask = 1},
    {0x5f, 0x12, 0x54, 0x00, 0x04, 0x04, 0x20, 0x00, 0x80, 0x1b, 0x00,
     0x00, 0x42, 0x00}, 14},

   {AGX_ISA_ENCODING_AGX3C, true,
    {.data = 2,
     .width = AGX3_MEM_W8,
     .address_kind = AGX3_SCS_ADDRESS_UNIFORM_INDEXED,
     .uniform_base = 4,
     .slot = 8,
     .wait_mask = 1,
     .zero_extend = true,
     .c_load_marker = true},
    {0x37, 0x04, 0x01, 0x08, 0x1a, 0x47, 0xa8, 0x02, 0x02, 0x08, 0x00,
     0x00, 0x01, 0x84, 0x00, 0x00}, 16},
   {AGX_ISA_ENCODING_AGX3C, false,
    {.width = AGX3_MEM_W8,
     .address_kind = AGX3_SCS_ADDRESS_UNIFORM_INDEXED,
     .uniform_base = 4},
    {0x27, 0x00, 0x01, 0x08, 0x32, 0x40, 0xa9, 0x02, 0x02, 0x08, 0x00,
     0x00, 0x01, 0x84}, 14},
   {AGX_ISA_ENCODING_AGX3C, true,
    {.data = 2,
     .width = AGX3_MEM_W16,
     .address_kind = AGX3_SCS_ADDRESS_UNIFORM_INDEXED,
     .uniform_base = 4,
     .slot = 8,
     .wait_mask = 1,
     .zero_extend = true,
     .c_load_marker = true},
    {0x37, 0x04, 0x01, 0x00, 0x1e, 0x47, 0xa8, 0x02, 0x02, 0x08, 0x00,
     0x00, 0x01, 0x04, 0x00, 0x00}, 16},
   {AGX_ISA_ENCODING_AGX3C, false,
    {.width = AGX3_MEM_W16,
     .address_kind = AGX3_SCS_ADDRESS_UNIFORM_INDEXED,
     .uniform_base = 4},
    {0x27, 0x00, 0x01, 0x00, 0x36, 0x40, 0xa9, 0x02, 0x02, 0x08, 0x00,
     0x00, 0x01, 0x04}, 14},
   {AGX_ISA_ENCODING_AGX3C, true,
    {.data = 2,
     .width = AGX3_MEM_W32,
     .address_kind = AGX3_SCS_ADDRESS_UNIFORM_INDEXED,
     .uniform_base = 4,
     .slot = 8,
     .wait_mask = 1,
     .c_load_marker = true},
    {0x37, 0x04, 0x01, 0x02, 0x16, 0x47, 0xa8, 0x02, 0x02, 0x08, 0x00,
     0x00, 0x01, 0x84, 0x00, 0x00}, 16},
   {AGX_ISA_ENCODING_AGX3C, false,
    {.width = AGX3_MEM_W32,
     .address_kind = AGX3_SCS_ADDRESS_UNIFORM_INDEXED,
     .uniform_base = 4},
    {0x27, 0x00, 0x01, 0x02, 0x36, 0x40, 0xa9, 0x02, 0x02, 0x08, 0x00,
     0x00, 0x01, 0x84}, 14},
   {AGX_ISA_ENCODING_AGX3C, true,
    {.data = 2,
     .width = AGX3_MEM_W32,
     .byte_offset = 16,
     .address_kind = AGX3_SCS_ADDRESS_UNIFORM_INDEXED,
     .uniform_base = 4,
     .slot = 8,
     .wait_mask = 1,
     .c_load_marker = true},
    {0x37, 0x04, 0x01, 0x02, 0x16, 0x47, 0xa8, 0x02, 0x12, 0x08, 0x00,
     0x00, 0x01, 0x84, 0x00, 0x00}, 16},
   {AGX_ISA_ENCODING_AGX3C, false,
    {.width = AGX3_MEM_W32,
     .byte_offset = 16,
     .address_kind = AGX3_SCS_ADDRESS_UNIFORM_INDEXED,
     .uniform_base = 4},
    {0x27, 0x00, 0x01, 0x02, 0x36, 0x40, 0xa9, 0x02, 0x12, 0x08, 0x00,
     0x00, 0x01, 0x84}, 14},
   {AGX_ISA_ENCODING_AGX3C, true,
    {.width = AGX3_MEM_W8,
     .address_kind = AGX3_SCS_ADDRESS_UNIFORM_INDEXED,
     .index = 2,
     .uniform_base = 4,
     .slot = 8,
     .wait_mask = 1,
     .c_load_marker = true},
    {0x27, 0x04, 0x01, 0x08, 0x12, 0x47, 0xa8, 0x02, 0x02, 0x08, 0x20,
     0x00, 0x01, 0x84, 0x00, 0x00}, 16},

   /* Live traversal compiler corpus: uniform half126, W8, offset 51, index
    * half0, scoreboard wait on slot 8.
    */
   {AGX_ISA_ENCODING_AGX3C, true,
    {.data = 4,
     .width = AGX3_MEM_W8,
     .byte_offset = 51,
     .address_kind = AGX3_SCS_ADDRESS_UNIFORM_INDEXED,
     .uniform_base = 126,
     .slot = 8,
     .wait_mask = 1u << 7},
    {0x2f, 0x00, 0x01, 0x8c, 0x32, 0xe7, 0xa8, 0x4a, 0x32, 0x38, 0x00,
     0x00, 0x01, 0x84, 0x00, 0x00}, 16},
};

static bool
descs_equal(const struct agx3_scs_desc *a, const struct agx3_scs_desc *b)
{
   return a->data == b->data && a->width == b->width &&
          a->byte_offset == b->byte_offset && a->slot == b->slot &&
          a->address_kind == b->address_kind && a->index == b->index &&
          a->uniform_base == b->uniform_base &&
          a->wait_mask == b->wait_mask && a->value_kill == b->value_kill &&
          a->zero_extend == b->zero_extend &&
          a->c_load_marker == b->c_load_marker;
}

static void
test_vectors(void)
{
   for (unsigned i = 0; i < sizeof(vectors) / sizeof(vectors[0]); ++i) {
      const struct scs_vec *v = &vectors[i];
      uint8_t encoded[AGX3_SCS_MAX_LENGTH];
      uint8_t length = 0;
      const bool ok =
         v->load ? agx3_scs_encode_load(v->enc, &v->d, encoded, &length)
                 : agx3_scs_encode_store(v->enc, &v->d, encoded, &length);
      CHECK(ok);
      CHECK(length == v->length);
      CHECK(memcmp(encoded, v->bytes, v->length) == 0);
      for (unsigned j = v->length; j < sizeof(encoded); ++j)
         CHECK(encoded[j] == 0);

      struct agx3_scs_desc decoded;
      const bool decoded_ok =
         v->load ? agx3_scs_decode_load(v->enc, v->bytes, v->length, &decoded)
                 : agx3_scs_decode_store(v->enc, v->bytes, v->length,
                                         &decoded);
      CHECK(decoded_ok);
      CHECK(descs_equal(&decoded, &v->d));
   }
}

static void
test_synthetic_roundtrip(void)
{
   const struct agx3_scs_desc descs[] = {
      {.data = 10, .width = AGX3_MEM_W16, .byte_offset = -4, .slot = 3,
       .wait_mask = (1u << 0) | (1u << 5)},
      {.data = 18, .width = AGX3_MEM_W32, .byte_offset = -8,
       .wait_mask = (1u << 0) | (1u << 7), .value_kill = true},
      {.data = 10,
       .width = AGX3_MEM_W16,
       .byte_offset = -4,
       .address_kind = AGX3_SCS_ADDRESS_UNIFORM_INDEXED,
       .index = 18,
       .uniform_base = 126,
       .slot = 3,
       .wait_mask = (1u << 0) | (1u << 5)},
      {.data = 18,
       .width = AGX3_MEM_W32,
       .byte_offset = -8,
       .address_kind = AGX3_SCS_ADDRESS_UNIFORM_INDEXED,
       .index = 300,
       .uniform_base = 511,
       .wait_mask = (1u << 0) | (1u << 7)},
   };
   const enum agx_isa_encoding encs[] = {AGX_ISA_ENCODING_AGX3B,
                                         AGX_ISA_ENCODING_AGX3C,
                                         AGX_ISA_ENCODING_AGX3B,
                                         AGX_ISA_ENCODING_AGX3C};
   const bool loads[] = {true, false, true, false};

   for (unsigned i = 0; i < 4; ++i) {
      uint8_t encoded[AGX3_SCS_MAX_LENGTH];
      uint8_t length = 0;
      CHECK(loads[i] ? agx3_scs_encode_load(encs[i], &descs[i], encoded,
                                            &length)
                     : agx3_scs_encode_store(encs[i], &descs[i], encoded,
                                             &length));
      struct agx3_scs_desc decoded;
      CHECK(loads[i] ? agx3_scs_decode_load(encs[i], encoded, length, &decoded)
                     : agx3_scs_decode_store(encs[i], encoded, length,
                                             &decoded));
      CHECK(descs_equal(&descs[i], &decoded));
   }
}

static void
test_fail_closed(void)
{
   uint8_t encoded[AGX3_SCS_MAX_LENGTH];
   uint8_t length = 0xff;
   struct agx3_scs_desc bad = {
      .data = 2, .width = AGX3_MEM_W32, .slot = 1,
   };

   memset(encoded, 0xcc, sizeof(encoded));
   CHECK(!agx3_scs_encode_load(AGX_ISA_ENCODING_AGX2, &bad, encoded,
                               &length));
   CHECK(length == 0);
   for (unsigned i = 0; i < sizeof(encoded); ++i)
      CHECK(encoded[i] == 0);

   bad.byte_offset = 2;
   CHECK(!agx3_scs_encode_load(AGX_ISA_ENCODING_AGX3B, &bad, encoded,
                               &length));
   bad.byte_offset = 0;
   bad.data = 3;
   CHECK(!agx3_scs_encode_load(AGX_ISA_ENCODING_AGX3B, &bad, encoded,
                               &length));
   bad.data = 2;
   bad.slot = 0;
   CHECK(!agx3_scs_encode_load(AGX_ISA_ENCODING_AGX3B, &bad, encoded,
                               &length));
   bad.slot = 1;
   bad.wait_mask = 1u << 6;
   CHECK(!agx3_scs_encode_load(AGX_ISA_ENCODING_AGX3B, &bad, encoded,
                               &length));
   bad.wait_mask = 0;
   bad.value_kill = true;
   CHECK(!agx3_scs_encode_load(AGX_ISA_ENCODING_AGX3C, &bad, encoded,
                               &length));
   bad.value_kill = false;
   bad.c_load_marker = true;
   CHECK(!agx3_scs_encode_load(AGX_ISA_ENCODING_AGX3B, &bad, encoded,
                               &length));
   bad.c_load_marker = false;
   bad.address_kind = (enum agx3_scs_address_kind)2;
   CHECK(!agx3_scs_encode_load(AGX_ISA_ENCODING_AGX3C, &bad, encoded,
                               &length));
   bad.address_kind = AGX3_SCS_ADDRESS_DIRECT;
   bad.index = 1;
   CHECK(!agx3_scs_encode_load(AGX_ISA_ENCODING_AGX3C, &bad, encoded,
                               &length));
   bad.index = 0;
   bad.uniform_base = 1;
   CHECK(!agx3_scs_encode_load(AGX_ISA_ENCODING_AGX3C, &bad, encoded,
                               &length));
   bad.address_kind = AGX3_SCS_ADDRESS_UNIFORM_INDEXED;
   bad.uniform_base = 512;
   CHECK(!agx3_scs_encode_load(AGX_ISA_ENCODING_AGX3C, &bad, encoded,
                               &length));
   bad.uniform_base = 4;
   bad.index = 192;
   CHECK(!agx3_scs_encode_load(AGX_ISA_ENCODING_AGX3B, &bad, encoded,
                               &length));
   bad.index = 512;
   CHECK(!agx3_scs_encode_load(AGX_ISA_ENCODING_AGX3C, &bad, encoded,
                               &length));
   bad.index = 0;
   bad.zero_extend = true;
   CHECK(!agx3_scs_encode_load(AGX_ISA_ENCODING_AGX3B, &bad, encoded,
                               &length));
   bad.width = AGX3_MEM_W32;
   CHECK(!agx3_scs_encode_load(AGX_ISA_ENCODING_AGX3C, &bad, encoded,
                               &length));
   bad.zero_extend = false;
   bad.value_kill = true;
   bad.slot = 0;
   CHECK(!agx3_scs_encode_store(AGX_ISA_ENCODING_AGX3C, &bad, encoded,
                                &length));

   static const uint8_t b_ld64[] = {
      0x5f, 0x02, 0x54, 0x04, 0x00, 0x00, 0x00, 0x00, 0x58, 0x02, 0x08,
      0x00, 0x02, 0x00,
   };
   static const uint8_t c_register_base_ld[] = {
      0x87, 0x00, 0x01, 0x02, 0x06, 0x47, 0x88, 0x4a,
      0x02, 0x08, 0xa0, 0x02, 0x01, 0x84, 0x00, 0x00,
   };
   static const uint8_t c_vector_uniform_st[] = {
      0x37, 0x00, 0x01, 0x86, 0x32, 0xe0, 0xa9,
      0x5a, 0x12, 0x08, 0x08, 0x40, 0x01, 0x84,
   };
   struct agx3_scs_desc decoded = bad;
   /* b_ld64 was pinned here as undecodable while this core could only express
    * a single component.  It is a real Apple form: f68=3, i.e. two 32-bit
    * components -- a 64-bit load -- and it now decodes as exactly that. */
   memset(&decoded, 0, sizeof(decoded));
   CHECK(agx3_scs_decode_load(AGX_ISA_ENCODING_AGX3B, b_ld64,
                              sizeof(b_ld64), &decoded));
   CHECK(decoded.component_mask == 3);
   CHECK(decoded.width == AGX3_MEM_W32);
   CHECK(decoded.address_kind == AGX3_SCS_ADDRESS_DIRECT);
   CHECK(decoded.byte_offset == 16);
   CHECK(decoded.data == 4);
   CHECK(!agx3_scs_decode_load(AGX_ISA_ENCODING_AGX3C, c_register_base_ld,
                               sizeof(c_register_base_ld), &decoded));
   CHECK(!agx3_scs_decode_store(AGX_ISA_ENCODING_AGX3C, c_vector_uniform_st,
                                sizeof(c_vector_uniform_st), &decoded));
   CHECK(!agx3_scs_decode_load(AGX_ISA_ENCODING_AGX3C, vectors[6].bytes,
                               vectors[6].length - 1, &decoded));
   CHECK(!agx3_scs_decode_load(AGX_ISA_ENCODING_AGX3C, NULL,
                               vectors[6].length, &decoded));
   CHECK(!agx3_scs_decode_load(AGX_ISA_ENCODING_AGX3C, vectors[6].bytes,
                               vectors[6].length, NULL));

   uint8_t changed[AGX3_SCS_MAX_LENGTH];
   memcpy(changed, vectors[6].bytes, vectors[6].length);
   changed[0] ^= 1;
   CHECK(!agx3_scs_decode_load(AGX_ISA_ENCODING_AGX3C, changed,
                               vectors[6].length, &decoded));
}


/*
 * The four address kinds and the vector component masks, as Apple's own
 * compiler emits them in G17P ray-query setup.
 *
 * Produced with the Metal toolchain, one MSL source per flag variant:
 *   metal -mmacosx-version-min=26.0 -c <src>.metal -o <src>.air
 *   metallib <src>.air -o <src>.metallib
 *   metal-nt -arch applegpu_g17p -N <src>.mtlp-json \
 *            -platform_version macos 26.0 26.0 <src>.metallib -o <src>.nat
 * then linear-disassembling __text and keeping every distinct stscs.
 *
 * These are the forms this core used to fail closed on: a component mask
 * other than one, and a slot-B base register (Apple f38=2) instead of a
 * uniform.  Each must round-trip byte-for-byte.  No hardware involved.
 */
static void
test_g17p_ray_setup_forms(void)
{
   static const struct {
      const char *note;
      uint8_t bytes[14];
      struct agx3_scs_desc d;
   } cases[] = {
      {"direct, 1 component",
       {0xdf, 0x02, 0x54, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x02, 0x0a,
        0x4c, 0x00, 0x00},
       {.data = 0, .width = AGX3_MEM_W32, .byte_offset = 24656,
        .address_kind = AGX3_SCS_ADDRESS_DIRECT, .component_mask = 1,
        .value_kill = true}},
      {"uniform-indexed, 4 components, selector 0",
       {0xdf, 0x02, 0x54, 0x00, 0x72, 0x68, 0x20, 0x00, 0x5c, 0x05, 0x00,
        0x40, 0x08, 0x00},
       {.data = 0, .width = AGX3_MEM_W32, .byte_offset = 0,
        .address_kind = AGX3_SCS_ADDRESS_UNIFORM_INDEXED, .index = 52,
        .uniform_base = 114, .component_mask = 15,
        .has_address_selector = true, .address_selector = 0}},
      {"uniform-indexed, 4 components, selector 2",
       {0xdf, 0x02, 0x54, 0x0a, 0x74, 0x40, 0x20, 0x00, 0x5c, 0x05, 0x20,
        0xc0, 0x08, 0x00},
       {.data = 10, .width = AGX3_MEM_W32, .byte_offset = 256,
        .address_kind = AGX3_SCS_ADDRESS_UNIFORM_INDEXED, .index = 32,
        .uniform_base = 116, .component_mask = 15,
        .has_address_selector = true, .address_selector = 2}},
      {"register-indexed, 2 components",
       {0xdf, 0x02, 0x54, 0x1a, 0x2c, 0x54, 0x20, 0x00, 0x64, 0x01, 0x00,
        0x00, 0x0d, 0x00},
       {.data = 26, .width = AGX3_MEM_W32, .byte_offset = 0,
        .address_kind = AGX3_SCS_ADDRESS_REGISTER_INDEXED, .index = 42,
        .base_register = 44, .component_mask = 3,
        .has_address_selector = true, .address_selector = 3}},
      {"register-indexed, 1 component",
       {0xdf, 0x02, 0x54, 0x34, 0x2c, 0x54, 0x20, 0x00, 0x44, 0x03, 0x00,
        0x00, 0x0d, 0x00},
       {.data = 52, .width = AGX3_MEM_W32, .byte_offset = 0,
        .address_kind = AGX3_SCS_ADDRESS_REGISTER_INDEXED, .index = 42,
        .base_register = 44, .component_mask = 1,
        .has_address_selector = true, .address_selector = 3,
        .value_kill = true}},
      {"register-base, no index",
       {0xdf, 0x02, 0x54, 0x08, 0x32, 0x00, 0x00, 0x00, 0x84, 0xe3, 0x00,
        0x00, 0x04, 0x00},
       {.data = 8, .width = AGX3_MEM_W8, .byte_offset = 7,
        .address_kind = AGX3_SCS_ADDRESS_REGISTER_BASE, .base_register = 50,
        .component_mask = 1, .has_address_selector = true,
        .address_selector = 4, .value_kill = true}},
   };

   for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
      uint8_t encoded[AGX3_SCS_MAX_LENGTH];
      uint8_t length = 0;
      CHECK(agx3_scs_encode_store(AGX_ISA_ENCODING_AGX3B, &cases[i].d,
                                  encoded, &length));
      CHECK(length == sizeof(cases[i].bytes));
      CHECK(memcmp(encoded, cases[i].bytes, sizeof(cases[i].bytes)) == 0);

      struct agx3_scs_desc decoded;
      memset(&decoded, 0, sizeof(decoded));
      CHECK(agx3_scs_decode_store(AGX_ISA_ENCODING_AGX3B, cases[i].bytes,
                                  sizeof(cases[i].bytes), &decoded));
      CHECK(descs_equal(&decoded, &cases[i].d));
   }

   /* Register-base and vector forms are AGX3b-only: no AGX3c witness. */
   struct agx3_scs_desc c_reg = cases[3].d;
   uint8_t out[AGX3_SCS_MAX_LENGTH];
   uint8_t out_len = 0;
   CHECK(!agx3_scs_encode_store(AGX_ISA_ENCODING_AGX3C, &c_reg, out,
                                &out_len));
}

int
main(void)
{
   test_vectors();
   test_g17p_ray_setup_forms();
   test_synthetic_roundtrip();
   test_fail_closed();

   if (failures != 0) {
      fprintf(stderr, "agx3_scs_core: %u/%u checks failed\n", failures,
              checks);
      return 1;
   }

   printf("agx3_scs_core: %zu controlled Apple G17P/G17S vectors and all "
          "%u checks passed\n",
          sizeof(vectors) / sizeof(vectors[0]), checks);
   return 0;
}
