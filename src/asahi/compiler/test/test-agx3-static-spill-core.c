/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_static_spill_core.h"

#include <stdio.h>
#include <stdlib.h>
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

static void
check_store(enum agx_isa_encoding enc,
            const struct agx3_static_spill_store_desc *d, uint8_t ep,
            const uint8_t *expected, uint8_t expected_length, bool check_auto)
{
   uint8_t encoded[AGX3_STATIC_SPILL_MAX_LENGTH];
   uint8_t length = 0;
   CHECK(agx3_static_spill_encode_store_form(enc, d, ep, encoded, &length));
   CHECK(length == expected_length);
   CHECK(memcmp(encoded, expected, expected_length) == 0);
   for (unsigned i = expected_length; i < sizeof(encoded); ++i)
      CHECK(encoded[i] == 0);

   struct agx3_static_spill_store_desc decoded;
   CHECK(
      agx3_static_spill_decode_store(enc, expected, expected_length, &decoded));
   CHECK(memcmp(&decoded, d, sizeof(decoded)) == 0);

   if (check_auto) {
      memset(encoded, 0xcc, sizeof(encoded));
      length = 0xff;
      CHECK(agx3_static_spill_encode_store(enc, d, encoded, &length));
      CHECK(length == expected_length);
      CHECK(memcmp(encoded, expected, expected_length) == 0);
   }
}

static void
check_load(enum agx_isa_encoding enc,
           const struct agx3_static_spill_load_desc *d, uint8_t ep,
           const uint8_t *expected, uint8_t expected_length, bool check_auto)
{
   uint8_t encoded[AGX3_STATIC_SPILL_MAX_LENGTH];
   uint8_t length = 0;
   CHECK(agx3_static_spill_encode_load_form(enc, d, ep, encoded, &length));
   CHECK(length == expected_length);
   CHECK(memcmp(encoded, expected, expected_length) == 0);
   for (unsigned i = expected_length; i < sizeof(encoded); ++i)
      CHECK(encoded[i] == 0);

   struct agx3_static_spill_load_desc decoded;
   CHECK(
      agx3_static_spill_decode_load(enc, expected, expected_length, &decoded));
   CHECK(memcmp(&decoded, d, sizeof(decoded)) == 0);

   if (check_auto) {
      memset(encoded, 0xcc, sizeof(encoded));
      length = 0xff;
      CHECK(agx3_static_spill_encode_load(enc, d, encoded, &length));
      CHECK(length == expected_length);
      CHECK(memcmp(encoded, expected, expected_length) == 0);
   }
}

static void
hex_bytes(const char *hex, uint8_t *out, uint8_t *length)
{
   uint8_t n = 0;
   for (; hex[0] && hex[1]; hex += 2) {
      const char pair[3] = {hex[0], hex[1], 0};
      out[n++] = (uint8_t)strtoul(pair, NULL, 16);
   }
   *length = n;
}

/*
 * applegpu_g17p's own spills of 16-bit temporaries. metal-nt -arch
 * applegpu_g17p on kernels holding 100 live ushort3, 260 live ushort and 260
 * live half temporaries; Apple splits a ushort3 into a W32 pair and a W16
 * half. Every one of the 842 static copies and reloads in those kernels that
 * carries no source-cache (copy f25) or live-TPS (rld f27 clear) bit
 * re-encodes byte for byte through the automatic form choice; these are one of
 * each shape, plus two of the reloads that keep their cell live.
 */
static void
check_g17p_half_spills(void)
{
   const struct {
      const char *hex;
      struct agx3_static_spill_store_desc d;
      uint8_t ep;
   } stores[] = {
      {"930e07002e1000080000",
       {.tps_half = 147, .src = 15, .width = AGX3_MEM_W16, .slot = 3}, 1},
      {"53064f00221000040000",
       {.tps_half = 170, .src = 6, .width = AGX3_MEM_W16, .src_kill = true,
        .slot = 2}, 1},
      {"631887002e1000140000",
       {.tps_half = 205, .src = 25, .width = AGX3_MEM_W16, .slot = 6}, 1},
      {"4b054f00229000040000",
       {.tps_half = 168, .src = 4, .width = AGX3_MEM_W32, .src_kill = true,
        .slot = 2, .wait_mask = 1u << 2}, 1},
   };
   const struct {
      const char *hex;
      struct agx3_static_spill_load_desc d;
      uint8_t ep;
   } loads[] = {
      {"14fc0800",
       {.dest = 2, .tps_half = 124, .width = AGX3_MEM_W16, .slot = 1,
        .slot_mirror = 1}, 0},
      {"14aa092602000000",
       {.dest = 2, .tps_half = 170, .width = AGX3_MEM_W16, .slot = 2}, 1},
      {"14ae192002000000",
       {.dest = 2, .tps_half = 174, .width = AGX3_MEM_W16, .slot = 2,
        .slot_mirror = 2}, 1},
      {"14fe090000000040",
       {.dest = 2, .tps_half = 126, .width = AGX3_MEM_W16, .slot = 1,
        .slot_mirror = 1, .wait_mask = 1u << 4}, 1},
      {"0ca8192102000000",
       {.dest = 0, .tps_half = 168, .width = AGX3_MEM_W32, .slot = 2,
        .slot_mirror = 2}, 1},
      /* Reloads of a cell that is reloaded again before it is written. */
      {"14901420",
       {.dest = 3, .tps_half = 16, .width = AGX3_MEM_W16, .slot = 2,
        .slot_mirror = 2, .tps_live = true}, 0},
      {"148c112000200000",
       {.dest = 2, .tps_half = 12, .width = AGX3_MEM_W16, .slot = 2,
        .slot_mirror = 2, .wait_mask = 1, .tps_live = true}, 1},
   };

   uint8_t bytes[AGX3_STATIC_SPILL_MAX_LENGTH], length;
   for (unsigned i = 0; i < sizeof(stores) / sizeof(stores[0]); ++i) {
      hex_bytes(stores[i].hex, bytes, &length);
      check_store(AGX_ISA_ENCODING_AGX3B, &stores[i].d, stores[i].ep, bytes,
                  length, true);
   }
   for (unsigned i = 0; i < sizeof(loads) / sizeof(loads[0]); ++i) {
      hex_bytes(loads[i].hex, bytes, &length);
      check_load(AGX_ISA_ENCODING_AGX3B, &loads[i].d, loads[i].ep, bytes,
                 length, true);
   }
}

int
main(void)
{
   static const uint8_t b_copy_h[] = {0x03, 0x14, 0x09, 0xc2};
   const struct agx3_static_spill_store_desc b_copy_h_d = {
      .tps_half = 0,
      .src = 20,
      .width = AGX3_MEM_W16,
      .src_kill = true,
      .slot = 1,
      .wait_mask = 1u << 5,
   };
   check_store(AGX_ISA_ENCODING_AGX3B, &b_copy_h_d, 0, b_copy_h,
               sizeof(b_copy_h), true);

   static const uint8_t b_copy_w[] = {
      0x1b, 0x49, 0x0f, 0x00, 0x22, 0x00, 0x00, 0x04, 0x00, 0x00,
   };
   const struct agx3_static_spill_store_desc b_copy_w_d = {
      .tps_half = 2,
      .src = 72,
      .width = AGX3_MEM_W32,
      .src_kill = true,
      .slot = 2,
   };
   check_store(AGX_ISA_ENCODING_AGX3B, &b_copy_w_d, 1, b_copy_w,
               sizeof(b_copy_w), true);

   static const uint8_t b_rld_h[] = {0x24, 0xbf, 0x18, 0xa4};
   const struct agx3_static_spill_load_desc b_rld_h_d = {
      .dest = 4,
      .tps_half = 63,
      .width = AGX3_MEM_W16,
      .slot = 6,
      .slot_mirror = 6,
   };
   check_load(AGX_ISA_ENCODING_AGX3B, &b_rld_h_d, 0, b_rld_h, sizeof(b_rld_h),
              true);

   static const uint8_t b_rld_h_long[] = {
      0x24, 0xc5, 0x1d, 0xa4, 0x00, 0x20, 0x00, 0x00,
   };
   const struct agx3_static_spill_load_desc b_rld_h_long_d = {
      .dest = 5,
      .tps_half = 69,
      .width = AGX3_MEM_W16,
      .slot = 6,
      .slot_mirror = 6,
      .wait_mask = 1,
   };
   check_load(AGX_ISA_ENCODING_AGX3B, &b_rld_h_long_d, 1, b_rld_h_long,
              sizeof(b_rld_h_long), true);

   static const uint8_t b_rld_structural[] = {
      0x0c, 0x88, 0x09, 0x27, 0x60, 0x00, 0x00, 0x80,
   };
   const struct agx3_static_spill_load_desc b_rld_structural_d = {
      .dest = 0,
      .tps_half = 8,
      .width = AGX3_MEM_W32,
      .slot = 2,
      .slot_mirror = 0,
      .wait_mask = 1u << 5,
      .structural_flag = true,
   };
   check_load(AGX_ISA_ENCODING_AGX3B, &b_rld_structural_d, 1, b_rld_structural,
              sizeof(b_rld_structural), true);

   /* applegpu_g17p device-atomic return: atomicdev publishes its hidden old
    * value on slot 1, then this exact rld fetches TPS word 0 into r0 and
    * republishes the GPR result on slot 6. */
   static const uint8_t b_atomic_return_rld[] = {
      0x0c, 0x80, 0x19, 0xa5, 0x00, 0x20, 0x00, 0x00,
   };
   const struct agx3_static_spill_load_desc b_atomic_return_rld_d = {
      .dest = 0,
      .tps_half = 0,
      .width = AGX3_MEM_W32,
      .slot = 6,
      .slot_mirror = 6,
      .wait_mask = 1,
   };
   check_load(AGX_ISA_ENCODING_AGX3B, &b_atomic_return_rld_d, 1,
              b_atomic_return_rld, sizeof(b_atomic_return_rld), true);

   /* The same return in a kernel that also spills a thread-private array:
    * the old value sits above the spill frame at TPS half 130, and this
    * reload is scheduled on slot 1. */
   static const uint8_t b_atomic_return_rld_130[] = {
      0x0c, 0x82, 0x19, 0x05, 0x02, 0x20, 0x00, 0x00,
   };
   const struct agx3_static_spill_load_desc b_atomic_return_rld_130_d = {
      .dest = 0,
      .tps_half = 130,
      .width = AGX3_MEM_W32,
      .slot = 1,
      .slot_mirror = 6,
      .wait_mask = 1,
   };
   check_load(AGX_ISA_ENCODING_AGX3B, &b_atomic_return_rld_130_d, 1,
              b_atomic_return_rld_130, sizeof(b_atomic_return_rld_130), true);

   static const uint8_t c_copy_h[] = {
      0x2b, 0x00, 0x07, 0x10, 0x26, 0x00, 0xa0, 0x02,
   };
   const struct agx3_static_spill_store_desc c_copy_h_d = {
      .tps_half = 0,
      .src = 8,
      .width = AGX3_MEM_W16,
      .src_kill = true,
      .slot = 1,
      .wait_mask = 1u << 7,
   };
   check_store(AGX_ISA_ENCODING_AGX3C, &c_copy_h_d, 1, c_copy_h,
               sizeof(c_copy_h), true);

   static const uint8_t c_copy_w[] = {
      0x33, 0x00, 0x07, 0x90, 0x26, 0x90, 0xa4, 0x02,
   };
   const struct agx3_static_spill_store_desc c_copy_w_d = {
      .tps_half = 2,
      .src = 72,
      .width = AGX3_MEM_W32,
      .src_kill = true,
      .slot = 2,
   };
   check_store(AGX_ISA_ENCODING_AGX3C, &c_copy_w_d, 1, c_copy_w,
               sizeof(c_copy_w), true);

   static const uint8_t c_rld_h[] = {
      0x24, 0x81, 0x21, 0x11, 0x2c, 0x02, 0xa4, 0x0a,
   };
   const struct agx3_static_spill_load_desc c_rld_h_d = {
      .dest = 5,
      .tps_half = 9,
      .width = AGX3_MEM_W16,
      .slot = 3,
      .slot_mirror = 3,
      .narrow_extend = true,
   };
   check_load(AGX_ISA_ENCODING_AGX3C, &c_rld_h_d, 1, c_rld_h, sizeof(c_rld_h),
              false);

   static const uint8_t c_rld_h_compact[] = {0x24, 0x88, 0x1c, 0x20};
   const struct agx3_static_spill_load_desc c_rld_h_compact_d = {
      .dest = 5,
      .tps_half = 8,
      .width = AGX3_MEM_W16,
      .slot = 2,
      .slot_mirror = 2,
   };
   check_load(AGX_ISA_ENCODING_AGX3C, &c_rld_h_compact_d, 0, c_rld_h_compact,
              sizeof(c_rld_h_compact), true);

   static const uint8_t c_rld_w[] = {
      0x24, 0x86, 0xa1, 0x00, 0x26, 0x00, 0xa0, 0x12,
   };
   const struct agx3_static_spill_load_desc c_rld_w_d = {
      .dest = 72,
      .tps_half = 2,
      .width = AGX3_MEM_W32,
      .slot = 1,
      .slot_mirror = 1,
      .wait_mask = 1,
   };
   check_load(AGX_ISA_ENCODING_AGX3C, &c_rld_w_d, 1, c_rld_w, sizeof(c_rld_w),
              true);

   /* Fail closed on non-static widths, misaligned 32-bit operands, malformed
    * target/form/length, and any mutated fixed opcode bit. */
   uint8_t encoded[AGX3_STATIC_SPILL_MAX_LENGTH];
   uint8_t length = 0;
   struct agx3_static_spill_store_desc bad_store = b_copy_w_d;
   bad_store.width = AGX3_MEM_W8;
   memset(encoded, 0xcc, sizeof(encoded));
   length = 0xff;
   CHECK(!agx3_static_spill_encode_store(AGX_ISA_ENCODING_AGX3B, &bad_store,
                                         encoded, &length));
   CHECK(length == 0);
   for (unsigned i = 0; i < sizeof(encoded); ++i)
      CHECK(encoded[i] == 0);

   bad_store = b_copy_w_d;
   bad_store.tps_half = 1;
   CHECK(!agx3_static_spill_encode_store(AGX_ISA_ENCODING_AGX3B, &bad_store,
                                         encoded, &length));
   bad_store = b_copy_w_d;
   bad_store.slot = 0;
   CHECK(!agx3_static_spill_encode_store(AGX_ISA_ENCODING_AGX3B, &bad_store,
                                         encoded, &length));
   bad_store = b_copy_w_d;
   bad_store.wait_mask = 1u << 6;
   CHECK(!agx3_static_spill_encode_store(AGX_ISA_ENCODING_AGX3B, &bad_store,
                                         encoded, &length));

   struct agx3_static_spill_store_desc decoded_store = b_copy_w_d;
   CHECK(!agx3_static_spill_decode_store(AGX_ISA_ENCODING_AGX2, b_copy_w,
                                         sizeof(b_copy_w), &decoded_store));
   CHECK(memcmp(&decoded_store, &(struct agx3_static_spill_store_desc){0},
                sizeof(decoded_store)) == 0);
   CHECK(!agx3_static_spill_decode_store(AGX_ISA_ENCODING_AGX3B, b_copy_w,
                                         sizeof(b_copy_w) - 1, &decoded_store));
   uint8_t changed[sizeof(c_rld_w)];
   memcpy(changed, c_rld_w, sizeof(changed));
   changed[0] ^= 1;
   struct agx3_static_spill_load_desc decoded_load;
   CHECK(!agx3_static_spill_decode_load(AGX_ISA_ENCODING_AGX3C, changed,
                                        sizeof(changed), &decoded_load));
   CHECK(!agx3_static_spill_decode_load(AGX_ISA_ENCODING_AGX3C, c_rld_w,
                                        sizeof(c_rld_w) - 1, &decoded_load));
   CHECK(!agx3_static_spill_decode_load(AGX_ISA_ENCODING_AGX3C, NULL,
                                        sizeof(c_rld_w), &decoded_load));
   CHECK(!agx3_static_spill_decode_load(AGX_ISA_ENCODING_AGX3C, c_rld_w,
                                        sizeof(c_rld_w), NULL));

   check_g17p_half_spills();

   if (failures != 0) {
      fprintf(stderr, "agx3_static_spill_core: %u/%u checks failed\n", failures,
              checks);
      return 1;
   }

   printf("agx3_static_spill_core: 22 Apple copy/rld vectors and all %u "
          "checks passed\n",
          checks);
   return 0;
}
