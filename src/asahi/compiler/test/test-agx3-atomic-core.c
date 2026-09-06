/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_atomic_core.h"

#include <stdio.h>
#include <string.h>

static unsigned checks;

#define CHECK(x)                                                               \
   do {                                                                        \
      ++checks;                                                                \
      if (!(x)) {                                                              \
         fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x);               \
         return 1;                                                             \
      }                                                                        \
   } while (0)

static int
check_device(enum agx_isa_encoding enc, struct agx3_device_atomic_desc d,
             const uint8_t *expected, uint8_t expected_len)
{
   uint8_t bytes[AGX3_ATOMIC_MAX_LENGTH] = {0};
   uint8_t len = 0;
   CHECK(agx3_atomic_encode_device(enc, &d, bytes, &len));
   CHECK(len == expected_len);
   CHECK(memcmp(bytes, expected, len) == 0);
   return 0;
}
static int
check_local(enum agx_isa_encoding enc, struct agx3_local_atomic_desc d,
            const uint8_t *expected, uint8_t expected_len)
{
   uint8_t bytes[AGX3_ATOMIC_MAX_LENGTH] = {0};
   uint8_t len = 0;
   CHECK(agx3_atomic_encode_local(enc, &d, bytes, &len));
   CHECK(len == expected_len);
   if (memcmp(bytes, expected, len) != 0) {
      fprintf(stderr, "local atomic got: ");
      for (unsigned i = 0; i < len; ++i)
         fprintf(stderr, "%02x", bytes[i]);
      fprintf(stderr, "\n");
   }
   CHECK(memcmp(bytes, expected, len) == 0);
   return 0;
}

int
main(void)
{
   const struct agx3_device_atomic_desc base = {
      .op = AGX3_ATOMIC_ADD,
      .value_kill = true,
      .base_kind = AGX3_ATOMIC_BASE_UNIFORM,
      .index_kind = AGX3_ATOMIC_INDEX_IMMEDIATE,
   };

   static const uint8_t b_add[] =
      {0x67,0x01,0x54,0x00,0x00,0x00,0x00,0x00,0x00,0x40,0x00,0x00,0x20,0x00};
   static const uint8_t c_add[] =
      {0x0f,0x00,0x03,0x00,0x00,0x04,0x04,0x20,0x80,0x00};
   CHECK(check_device(AGX_ISA_ENCODING_AGX3B, base, b_add, sizeof(b_add)) == 0);
   CHECK(check_device(AGX_ISA_ENCODING_AGX3C, base, c_add, sizeof(c_add)) == 0);

   struct agx3_device_atomic_desc x = base;
   x.op = AGX3_ATOMIC_XOR;
   static const uint8_t b_xor[] =
      {0x67,0x01,0x54,0x00,0x00,0x00,0x00,0x00,0x00,0x40,0x00,0x00,0x3e,0x00};
   static const uint8_t c_xor[] =
      {0x0f,0x00,0x03,0x00,0x20,0x0c,0x0c,0xa0,0x80,0x00,0x00,0x10};
   CHECK(check_device(AGX_ISA_ENCODING_AGX3B, x, b_xor, sizeof(b_xor)) == 0);
   CHECK(check_device(AGX_ISA_ENCODING_AGX3C, x, c_xor, sizeof(c_xor)) == 0);

   struct agx3_device_atomic_desc ret = base;
   ret.value = 6;
   ret.result = 6;
   ret.returns_value = true;
   ret.slot = 1;
   ret.wait_mask = 1;
   static const uint8_t b_ret[] =
      {0x67,0x11,0x54,0x00,0x00,0x80,0x01,0x00,0x00,0x42,0x00,0x00,0x20,0x00};
   CHECK(check_device(AGX_ISA_ENCODING_AGX3B, ret, b_ret, sizeof(b_ret)) == 0);

   /* Fresh Apple G17P register-base fetch-add.  The single field16 value is
    * both the update input and old-value result; index4 remains independent.
    * The source pointer is retained for a second atomic and the update waits
    * slot6, eliminating lifetime ambiguity from the native witness. */
   struct agx3_device_atomic_desc g17p_reg_return = {
      .op = AGX3_ATOMIC_ADD,
      .value = 6,
      .returns_value = true,
      .result = 6,
      .slot = 1,
      .base_kind = AGX3_ATOMIC_BASE_REGISTER,
      .base = 0,
      .index_kind = AGX3_ATOMIC_INDEX_REGISTER,
      .index = 4,
      .index_shift = 2,
      .wait_mask = 1u << 5,
   };
   static const uint8_t b_g17p_reg_return[] = {
      0x67,0x01,0x56,0x00,0x00,0x80,0x01,0x81,0x00,0x02,0x00,0x00,0x60,0x03,
   };
   CHECK(check_device(AGX_ISA_ENCODING_AGX3B, g17p_reg_return,
                      b_g17p_reg_return, sizeof(b_g17p_reg_return)) == 0);

   ret.result = 4;
   ret.wait_mask = 0;
   static const uint8_t c_ret[] =
      {0x2f,0x00,0x03,0x00,0x01,0x04,0x04,0x20,0x90,0x06};
   CHECK(check_device(AGX_ISA_ENCODING_AGX3C, ret, c_ret, sizeof(c_ret)) == 0);

   struct agx3_device_atomic_desc indexed = base;
   indexed.index_kind = AGX3_ATOMIC_INDEX_REGISTER;
   indexed.index = 2;
   indexed.index_kill = true;
   indexed.index_shift = 2;
   indexed.wait_mask = 1u << 5;
   static const uint8_t b_idx[] =
      {0x67,0x01,0x56,0x00,0x00,0x00,0x80,0x80,0x01,0x40,0x00,0x00,0x60,0x02};
   CHECK(check_device(AGX_ISA_ENCODING_AGX3B, indexed, b_idx, sizeof(b_idx)) == 0);
   indexed.wait_mask = 1u << 7;
   static const uint8_t c_idx[] =
      {0x0f,0x00,0x03,0x02,0x00,0x06,0x04,0xc0,0x80,0x00,0x00,0x20};
   CHECK(check_device(AGX_ISA_ENCODING_AGX3C, indexed, c_idx, sizeof(c_idx)) == 0);

   struct agx3_device_atomic_desc offset = base;
   offset.byte_offset = 28;
   static const uint8_t b_off[] =
      {0x67,0x01,0x54,0x00,0x00,0x00,0x00,0x00,0x00,0x40,0x1c,0x00,0x20,0x00};
   static const uint8_t c_off[] =
      {0x0f,0x00,0x03,0x00,0x00,0x04,0x84,0x23,0x80,0x00};
   CHECK(check_device(AGX_ISA_ENCODING_AGX3B, offset, b_off, sizeof(b_off)) == 0);
   CHECK(check_device(AGX_ISA_ENCODING_AGX3C, offset, c_off, sizeof(c_off)) == 0);

   struct agx3_device_atomic_desc uniform_index = base;
   uniform_index.index_kind = AGX3_ATOMIC_INDEX_UNIFORM;
   uniform_index.index = 12;
   uniform_index.index_shift = 2;
   static const uint8_t b_uidx[] =
      {0x67,0x01,0x54,0x00,0x00,0x00,0x00,0x83,0x00,0x40,0x00,0x00,0x60,0x04};
   static const uint8_t c_uidx[] =
      {0x0f,0x00,0x03,0x0c,0x00,0x00,0x04,0x40,0x80,0x00};
   CHECK(check_device(AGX_ISA_ENCODING_AGX3B, uniform_index, b_uidx,
                      sizeof(b_uidx)) == 0);
   CHECK(check_device(AGX_ISA_ENCODING_AGX3C, uniform_index, c_uidx,
                      sizeof(c_uidx)) == 0);

   struct agx3_local_atomic_desc local = {
      .op = AGX3_ATOMIC_ADD,
      .value = 4,
      .result = 6,
      .value_kill = true,
      .returns_value = true,
      .slot = 1,
      .address_kind = AGX3_ATOMIC_INDEX_IMMEDIATE,
   };
   static const uint8_t b_tg_add[] =
      {0x67,0x03,0x54,0x03,0x00,0x02,0x00,0x00,0x22,0x00,0x00,0x04};
   static const uint8_t c_tg_add[] =
      {0x3f,0x00,0x23,0x00,0x03,0x05,0x04,0xa0,0x90,0x04,0x40,0x02};
   CHECK(check_local(AGX_ISA_ENCODING_AGX3B, local, b_tg_add,
                     sizeof(b_tg_add)) == 0);
   CHECK(check_local(AGX_ISA_ENCODING_AGX3C, local, c_tg_add,
                     sizeof(c_tg_add)) == 0);

   local.op = AGX3_ATOMIC_UMIN;
   static const uint8_t b_tg_min[] =
      {0x67,0x03,0x54,0x03,0x00,0x02,0x00,0x00,0x22,0x00,0x40,0x07};
   static const uint8_t c_tg_min[] =
      {0x3f,0x00,0x23,0x00,0x23,0x05,0x0c,0xa0,0x90,0x04,0x40,0x12};
   CHECK(check_local(AGX_ISA_ENCODING_AGX3B, local, b_tg_min,
                     sizeof(b_tg_min)) == 0);
   CHECK(check_local(AGX_ISA_ENCODING_AGX3C, local, c_tg_min,
                     sizeof(c_tg_min)) == 0);

   local.op = AGX3_ATOMIC_CMPXCHG;
   local.value = 0;
   local.result = 0;
   local.value_kill = false;
   static const uint8_t b_tg_cas[] =
      {0x67,0x03,0x56,0x00,0x00,0x00,0x00,0x00,0x02,0x00,0x80,0x04};
   static const uint8_t c_tg_cas[] =
      {0x0f,0x00,0x23,0x00,0x03,0x0d,0x04,0xa0,0x10,0x00,0x40,0x22};
   CHECK(check_local(AGX_ISA_ENCODING_AGX3B, local, b_tg_cas,
                     sizeof(b_tg_cas)) == 0);
   CHECK(check_local(AGX_ISA_ENCODING_AGX3C, local, c_tg_cas,
                     sizeof(c_tg_cas)) == 0);

   /* Own Metal source, compiled by Apple's offline G17P translator.  These
    * are the no-result OR forms for words[0], words[2], and an inttoptr byte
    * address held in r1. */
   struct agx3_local_atomic_desc local_void = {
      .op = AGX3_ATOMIC_OR,
      .value = 0,
      .value_kill = true,
      .address_kind = AGX3_ATOMIC_INDEX_IMMEDIATE,
   };
   static const uint8_t b_tg_void_0[] =
      {0x67,0x03,0x54,0x00,0x00,0x00,0x00,0x00,0x20,0x00,0x80,0x05};
   CHECK(check_local(AGX_ISA_ENCODING_AGX3B, local_void, b_tg_void_0,
                     sizeof(b_tg_void_0)) == 0);

   local_void.address = 8;
   static const uint8_t b_tg_void_8[] =
      {0x67,0x03,0x54,0x00,0x00,0x00,0x00,0x00,0x20,0x02,0x80,0x05};
   CHECK(check_local(AGX_ISA_ENCODING_AGX3B, local_void, b_tg_void_8,
                     sizeof(b_tg_void_8)) == 0);

   local_void.value_kill = false;
   local_void.address_kind = AGX3_ATOMIC_INDEX_REGISTER;
   local_void.address = 2;
   local_void.address_kill = true;
   static const uint8_t b_tg_void_reg[] =
      {0x67,0x03,0x54,0x00,0x01,0x00,0x00,0x00,0x10,0x00,0x80,0x25};
   CHECK(check_local(AGX_ISA_ENCODING_AGX3B, local_void, b_tg_void_reg,
                     sizeof(b_tg_void_reg)) == 0);

   /* Fail closed on the ungrounded and structurally invalid axes. */
   uint8_t bytes[AGX3_ATOMIC_MAX_LENGTH];
   uint8_t len = 0;
   struct agx3_device_atomic_desc bad = base;
   bad.op = (enum agx3_atomic_op)11;
   CHECK(!agx3_atomic_encode_device(AGX_ISA_ENCODING_AGX3B, &bad, bytes, &len));
   bad = ret;
   bad.result = 8;
   CHECK(!agx3_atomic_encode_device(AGX_ISA_ENCODING_AGX3B, &bad, bytes, &len));
   bad = base;
   bad.base = 2;
   CHECK(!agx3_atomic_encode_device(AGX_ISA_ENCODING_AGX3C, &bad, bytes, &len));
   bad = base;
   bad.index_kind = AGX3_ATOMIC_INDEX_REGISTER;
   bad.index = 3;
   CHECK(!agx3_atomic_encode_device(AGX_ISA_ENCODING_AGX3C, &bad, bytes, &len));

   local_void.address = 192;
   CHECK(!agx3_atomic_encode_local(AGX_ISA_ENCODING_AGX3B, &local_void,
                                   bytes, &len));
   local_void.address = 2;
   CHECK(!agx3_atomic_encode_local(AGX_ISA_ENCODING_AGX3C, &local_void,
                                   bytes, &len));

   printf("agx3_atomic_core: %u checks passed\n", checks);
   return 0;
}
