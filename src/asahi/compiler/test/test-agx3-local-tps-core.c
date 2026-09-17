/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Byte-exact witnesses from the hash-pinned G16G/G17S local/stack corpus.
 */

#include "agx3_local_tps_core.h"

#include <stdio.h>
#include <string.h>

struct local_vec {
   enum agx_isa_encoding enc;
   bool load;
   uint8_t ep;
   struct agx3_local_desc d;
   uint8_t bytes[AGX3_LOCAL_TPS_MAX_LENGTH];
   uint8_t len;
};

static const struct local_vec local_vecs[] = {
   {AGX_ISA_ENCODING_AGX3B, false, 0,
    {.data = 0, .width = AGX3_MEM_W32, .mask = 1, .shift = 2,
     .base_kind = AGX3_LOCAL_BASE_UNIFORM, .base = 8,
     .index_is_register = true, .index = 5, .index_kill = true,
     .wait_mask = 1u << 5},
    {0xe7, 0x02, 0x56, 0x00, 0x08, 0x0a, 0x80, 0x00, 0x44, 0x02, 0x00,
     0x30, 0x02, 0x00}, 14},
   {AGX_ISA_ENCODING_AGX3B, true, 0,
    {.data = 0, .width = AGX3_MEM_W32, .mask = 1, .shift = 2,
     .base_kind = AGX3_LOCAL_BASE_UNIFORM, .base = 8,
     .index_is_register = true, .index = 4, .index_kill = true, .slot = 6},
    {0x67, 0x02, 0x54, 0x00, 0x08, 0x08, 0x80, 0x00, 0x44, 0x0d, 0x00,
     0xc0, 0x08, 0x00}, 14},
   {AGX_ISA_ENCODING_AGX3B, false, 0,
    {.data = 0, .width = AGX3_MEM_W32, .mask = 1,
     .base_kind = AGX3_LOCAL_BASE_IMMEDIATE, .value_kill = true},
    {0xe7, 0x02, 0x54, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x01, 0x00,
     0x10, 0x00, 0x00}, 14},
   {AGX_ISA_ENCODING_AGX3B, true, 0,
    {.data = 6, .width = AGX3_MEM_W32, .mask = 1,
     .base_kind = AGX3_LOCAL_BASE_IMMEDIATE, .slot = 6},
    {0x67, 0x02, 0x54, 0x06, 0x00, 0x00, 0x00, 0x00, 0x44, 0x05, 0x00,
     0x40, 0x00, 0x00}, 14},
   {AGX_ISA_ENCODING_AGX3B, false, 0,
    {.data = 4, .width = AGX3_MEM_W32, .mask = 1, .shift = 2,
     .base_kind = AGX3_LOCAL_BASE_REGISTER,
     .index_is_register = true, .index = 7, .index_kill = true,
     .wait_mask = 1},
    {0xe7, 0x12, 0x54, 0x04, 0x00, 0x0e, 0x80, 0x00, 0x44, 0x00, 0x00,
     0x30, 0x03, 0x00}, 14},
   {AGX_ISA_ENCODING_AGX3B, true, 0,
    {.data = 0, .width = AGX3_MEM_W32, .mask = 1, .shift = 2,
     .base_kind = AGX3_LOCAL_BASE_REGISTER, .base_kill = true,
     .index_is_register = true, .index = 1, .index_kill = true, .slot = 6},
    {0x67, 0x02, 0x54, 0x00, 0x00, 0x02, 0x80, 0x00, 0x44, 0x0d, 0x00,
     0xc0, 0x0c, 0x00}, 14},

   /*
    * applegpu_g17p witnesses, produced this session with
    *   metal -c tg.metal -o tg.air; metallib tg.air -o tg.metallib
    *   metal-nt -arch applegpu_g17p -N tg.mtlp-json \
    *            -platform_version macos 27.0 27.0 tg.metallib -o tg.nat
    * and read out of __TEXT,__text.  They pin two conventions that the G16G
    * corpus above only showed at one value each:
    *
    *  - every threadgroup access takes its base from a *uniform*, and the
    *    half is chosen per shader after the buffer arguments: 4 with one
    *    buffer, 16 with four, and a second block gets its own half (5).
    *    It is therefore a per-block base the runtime writes, not a fixed
    *    hardware-preloaded half.
    *  - every threadgroup load lands on scoreboard slot 6.
    */
   {AGX_ISA_ENCODING_AGX3B, false, 0,
    {.data = 0, .width = AGX3_MEM_W32, .mask = 1, .shift = 2,
     .base_kind = AGX3_LOCAL_BASE_UNIFORM, .base = 4,
     .index_is_register = true, .index = 5, .index_kill = true,
     .wait_mask = 1u << 1},
    {0xe7, 0x22, 0x54, 0x00, 0x04, 0x0a, 0x80, 0x00, 0x44, 0x02, 0x00,
     0x30, 0x02, 0x00}, 14},
   {AGX_ISA_ENCODING_AGX3B, true, 0,
    {.data = 0, .width = AGX3_MEM_W32, .mask = 1, .shift = 2,
     .base_kind = AGX3_LOCAL_BASE_UNIFORM, .base = 4,
     .index_is_register = true, .index = 4, .index_kill = true, .slot = 6,
     .wait_mask = 1u << 2},
    {0x67, 0x42, 0x54, 0x00, 0x04, 0x08, 0x80, 0x00, 0x44, 0x0d, 0x00,
     0xc0, 0x08, 0x00}, 14},
   /* A vec4 store into the second 16 bytes of the block: the index still
    * counts scalar elements and the static part rides byte_offset. */
   {AGX_ISA_ENCODING_AGX3B, false, 0,
    {.data = 30, .width = AGX3_MEM_W32, .mask = 15, .shift = 2,
     .base_kind = AGX3_LOCAL_BASE_UNIFORM, .base = 4,
     .index_is_register = true, .index = 4, .byte_offset = 16,
     .value_kill = true},
    {0xe7, 0x02, 0x54, 0x1e, 0x04, 0x08, 0x00, 0x00, 0x5c, 0x83, 0x00,
     0x30, 0x02, 0x00}, 14},
   /* Four buffer arguments push the same base to uniform half 16. */
   {AGX_ISA_ENCODING_AGX3B, false, 0,
    {.data = 0, .width = AGX3_MEM_W32, .mask = 1, .shift = 2,
     .base_kind = AGX3_LOCAL_BASE_UNIFORM, .base = 16,
     .index_is_register = true, .index = 10, .index_kill = true},
    {0xe7, 0x02, 0x54, 0x00, 0x10, 0x14, 0x80, 0x00, 0x44, 0x02, 0x00,
     0x30, 0x02, 0x00}, 14},
   /* A shader with two threadgroup blocks gets a base uniform each. */
   {AGX_ISA_ENCODING_AGX3B, false, 0,
    {.data = 2, .width = AGX3_MEM_W32, .mask = 1, .shift = 2,
     .base_kind = AGX3_LOCAL_BASE_UNIFORM, .base = 5,
     .index_is_register = true, .index = 10, .index_kill = true},
    {0xe7, 0x02, 0x54, 0x02, 0x05, 0x14, 0x80, 0x00, 0x44, 0x02, 0x00,
     0x30, 0x02, 0x00}, 14},

   {AGX_ISA_ENCODING_AGX3C, false, 1,
    {.data = 0, .width = AGX3_MEM_W32, .mask = 1, .shift = 2,
     .base_kind = AGX3_LOCAL_BASE_UNIFORM, .base = 8,
     .index_is_register = true, .index = 5, .index_kill = true,
     .value_kill = true, .wait_mask = 1u << 7},
    {0x0f, 0x08, 0x03, 0x04, 0x03, 0x06, 0x11, 0xc0, 0x00, 0x20}, 10},
   {AGX_ISA_ENCODING_AGX3C, true, 2,
    {.data = 0, .width = AGX3_MEM_W32, .mask = 1, .shift = 2,
     .base_kind = AGX3_LOCAL_BASE_UNIFORM, .base = 8,
     .index_is_register = true, .index = 4, .index_kill = true, .slot = 8,
     .c_load_marker = true},
    {0x0f, 0x08, 0x03, 0x04, 0x1a, 0x06, 0x10, 0xc0, 0x41, 0x00, 0x80,
     0x00, 0x00, 0x00}, 14},
   {AGX_ISA_ENCODING_AGX3C, false, 0,
    {.data = 0, .width = AGX3_MEM_W32, .mask = 1, .shift = 2,
     .base_kind = AGX3_LOCAL_BASE_UNIFORM, .base = 12,
     .index_is_register = true, .index = 8, .index_kill = true,
     .value_kill = true, .wait_mask = 1},
    {0x0f, 0x0c, 0x03, 0x08, 0x03, 0x26, 0x10, 0x40}, 8},
   {AGX_ISA_ENCODING_AGX3C, true, 0,
    {.data = 0, .width = AGX3_MEM_W32, .mask = 1, .shift = 2,
     .base_kind = AGX3_LOCAL_BASE_UNIFORM, .base = 8,
     .index_is_register = true, .index = 7, .index_kill = true, .slot = 8},
    {0x0f, 0x08, 0x03, 0x06, 0x1a, 0x06, 0x11, 0x40}, 8},
   {AGX_ISA_ENCODING_AGX3C, false, 0,
    {.data = 0, .width = AGX3_MEM_W32, .mask = 1,
     .base_kind = AGX3_LOCAL_BASE_IMMEDIATE, .value_kill = true},
    {0x0f, 0x00, 0x03, 0x00, 0x03, 0x05, 0x10, 0x20}, 8},
   {AGX_ISA_ENCODING_AGX3C, true, 2,
    {.data = 6, .width = AGX3_MEM_W32, .mask = 1,
     .base_kind = AGX3_LOCAL_BASE_IMMEDIATE, .slot = 8,
     .c_load_marker = true},
    {0x3f, 0x00, 0x03, 0x00, 0x1a, 0x05, 0x10, 0xa0, 0x41, 0x00, 0x80,
     0x00, 0x00, 0x00}, 14},
   {AGX_ISA_ENCODING_AGX3C, false, 2,
    {.data = 0, .width = AGX3_MEM_W32, .mask = 1, .shift = 2,
     .base_kind = AGX3_LOCAL_BASE_UNIFORM, .base = 8,
     .index_is_register = true, .index = 14, .byte_offset = 1200,
     .value_kill = true, .wait_mask = 1u << 1},
    {0x0f, 0x08, 0x03, 0x0e, 0x03, 0x42, 0x10, 0xd6, 0x01, 0x00, 0x00,
     0x00, 0x00, 0x04}, 14},
   {AGX_ISA_ENCODING_AGX3C, false, 0,
    {.data = 4, .width = AGX3_MEM_W32, .mask = 1, .shift = 2,
     .base_kind = AGX3_LOCAL_BASE_REGISTER,
     .index_is_register = true, .index = 7, .index_kill = true,
     .value_kill = true, .wait_mask = 1},
    {0x2f, 0x00, 0x03, 0x06, 0x83, 0x26, 0x11, 0x40}, 8},
   {AGX_ISA_ENCODING_AGX3C, true, 2,
    {.data = 0, .width = AGX3_MEM_W32, .mask = 1, .shift = 2,
     .base_kind = AGX3_LOCAL_BASE_REGISTER,
     .index_is_register = true, .index = 1, .slot = 8,
     .c_load_marker = true},
    {0x0f, 0x00, 0x03, 0x00, 0x9a, 0x02, 0x11, 0xc0, 0x41, 0x00, 0x80,
     0x00, 0x00, 0x00}, 14},
   {AGX_ISA_ENCODING_AGX3C, true, 2,
    {.data = 2, .width = AGX3_MEM_W8, .mask = 1,
     .base_kind = AGX3_LOCAL_BASE_UNIFORM, .base = 8,
     .index_is_register = true, .index = 4, .index_kill = true, .slot = 8,
     .narrow_extend = true, .c_load_marker = true},
    {0x17, 0x08, 0x23, 0x04, 0x1a, 0x06, 0x10, 0xa0, 0x41, 0x04, 0x80,
     0x00, 0x00, 0x00}, 14},
};

struct tps_vec {
   enum agx_isa_encoding enc;
   bool load;
   uint8_t ep;
   struct agx3_tps_desc d;
   uint8_t bytes[AGX3_LOCAL_TPS_MAX_LENGTH];
   uint8_t len;
};

static const struct tps_vec tps_vecs[] = {
   /* G16G repeat-tps.metal: repeated dynamic read, then final read. f66
    * consumes TPS on the last read; f27 independently consumes its GPR index. */
   {AGX_ISA_ENCODING_AGX3B, true, 0,
    {.data = 8, .width = AGX3_MEM_W32, .mask = 15,
     .byte_offset = 16, .index = 0, .frame_region = 6,
     .retain_tps = true, .slot = 1, .wait_mask = 1},
    {0x6f, 0x10, 0x54, 0x08, 0x00, 0x02, 0x17, 0x0a,
     0x04, 0x00, 0x02, 0x00}, 12},
   {AGX_ISA_ENCODING_AGX3B, true, 0,
    {.data = 0, .width = AGX3_MEM_W32, .mask = 15,
     .byte_offset = 16, .index = 0, .index_kill = true,
     .frame_region = 6, .slot = 6, .wait_mask = 1u << 5},
    {0x6f, 0x00, 0x56, 0x00, 0x00, 0x0a, 0x57, 0x0b,
     0x04, 0x40, 0x02, 0x00}, 12},
   {AGX_ISA_ENCODING_AGX3B, false, 0,
    {.data = 48, .width = AGX3_MEM_W32, .mask = 15, .shift = 2,
     .byte_offset = 4, .index = 4, .frame_region = 5, .value_kill = true,
     .wait_mask = 1u << 5},
    {0xef, 0x00, 0x56, 0x30, 0x04, 0x02, 0x17, 0x43, 0x00, 0x60}, 10},
   /* G16G g16g-dynamic-stack-store-static-reads.metal fsvec4: a float4
    * private-array element stored and reloaded through a byte index. The
    * store carries the wait of the preceding static copies and signals no
    * slot; the reload consumes the location (f66). */
   {AGX_ISA_ENCODING_AGX3B, false, 0,
    {.data = 0, .width = AGX3_MEM_W32, .mask = 15,
     .byte_offset = 16, .index = 34, .index_kill = true, .frame_region = 1,
     .wait_mask = 1},
    {0xef, 0x10, 0x54, 0x00, 0x22, 0x0a, 0x17, 0x00, 0x01, 0x40}, 10},
   {AGX_ISA_ENCODING_AGX3B, true, 0,
    {.data = 0, .width = AGX3_MEM_W32, .mask = 15,
     .byte_offset = 16, .index = 32, .index_kill = true, .frame_region = 1,
     .slot = 6},
    {0x6f, 0x00, 0x54, 0x00, 0x20, 0x0a, 0x57, 0x01, 0x04, 0x40, 0x02,
     0x00}, 12},
   {AGX_ISA_ENCODING_AGX3B, true, 0,
    {.data = 0, .width = AGX3_MEM_W16, .mask = 3, .shift = 2,
     .byte_offset = 4, .index = 0, .index_kill = true, .frame_region = 1,
     .slot = 6},
    {0x6f, 0x00, 0x54, 0x00, 0x00, 0x0a, 0x49, 0x01, 0x01, 0x40, 0x03,
     0x00}, 12},
   {AGX_ISA_ENCODING_AGX3C, false, 0,
    {.data = 48, .width = AGX3_MEM_W32, .mask = 15, .shift = 2,
     .byte_offset = 4, .index = 4, .frame_region = 1, .value_kill = true,
     .wait_mask = 1},
    {0x8f, 0x00, 0x43, 0x04, 0x61, 0x2a, 0x92, 0x40}, 8},
   {AGX_ISA_ENCODING_AGX3C, false, 1,
    {.data = 38, .width = AGX3_MEM_W32, .mask = 15, .shift = 2,
     .byte_offset = 20, .index = 4, .frame_region = 1, .value_kill = true,
     .wait_mask = 1u << 7},
    {0x3f, 0x00, 0x43, 0x04, 0x61, 0x0a, 0x92, 0xc2, 0x00, 0x20}, 10},
   {AGX_ISA_ENCODING_AGX3C, true, 0,
    {.data = 2, .width = AGX3_MEM_W16, .mask = 3, .shift = 2,
     .byte_offset = 4, .index = 2, .frame_region = 1, .slot = 8},
    {0x17, 0x01, 0x03, 0x02, 0x38, 0x0a, 0x92, 0x40}, 8},
   {AGX_ISA_ENCODING_AGX3C, true, 1,
    {.data = 2, .width = AGX3_MEM_W8, .mask = 1, .byte_offset = 4,
     .index = 2, .frame_region = 1, .narrow_extend = true, .slot = 8},
    {0x17, 0x01, 0x23, 0x02, 0x18, 0x02, 0x92, 0xa0, 0x00, 0x04}, 10},
};

static unsigned
test_local(void)
{
   unsigned failures = 0;
   for (unsigned i = 0; i < sizeof(local_vecs) / sizeof(local_vecs[0]); ++i) {
      const struct local_vec *v = &local_vecs[i];
      uint8_t out[AGX3_LOCAL_TPS_MAX_LENGTH] = {0};
      uint8_t len = 0;
      const bool ok = v->load ? agx3_local_encode_load(v->enc, &v->d, out, &len)
                              : agx3_local_encode_store(v->enc, &v->d, out, &len);
      if (!ok || len != v->len || memcmp(out, v->bytes, v->len) != 0) {
         fprintf(stderr, "local vector %u failed (ok=%u len=%u/%u)\n", i,
                 ok, len, v->len);
         failures++;
      }

      len = 0;
      const bool form_ok =
         v->load ? agx3_local_encode_load_form(v->enc, &v->d, v->ep, out, &len)
                 : agx3_local_encode_store_form(v->enc, &v->d, v->ep, out, &len);
      if (!form_ok || len != v->len || memcmp(out, v->bytes, v->len) != 0) {
         fprintf(stderr, "local form vector %u failed\n", i);
         failures++;
      }
   }
   return failures;
}

static unsigned
test_tps(void)
{
   unsigned failures = 0;
   for (unsigned i = 0; i < sizeof(tps_vecs) / sizeof(tps_vecs[0]); ++i) {
      const struct tps_vec *v = &tps_vecs[i];
      uint8_t out[AGX3_LOCAL_TPS_MAX_LENGTH] = {0};
      uint8_t len = 0;
      const bool ok = v->load ? agx3_tps_encode_load(v->enc, &v->d, out, &len)
                              : agx3_tps_encode_store(v->enc, &v->d, out, &len);
      if (!ok || len != v->len || memcmp(out, v->bytes, v->len) != 0) {
         fprintf(stderr, "TPS vector %u failed (ok=%u len=%u/%u)\n", i,
                 ok, len, v->len);
         failures++;
      }

      len = 0;
      const bool form_ok =
         v->load ? agx3_tps_encode_load_form(v->enc, &v->d, v->ep, out, &len)
                 : agx3_tps_encode_store_form(v->enc, &v->d, v->ep, out, &len);
      if (!form_ok || len != v->len || memcmp(out, v->bytes, v->len) != 0) {
         fprintf(stderr, "TPS form vector %u failed\n", i);
         failures++;
      }
   }
   return failures;
}

/*
 * f11, `narrow_extend`, is an AGX3c field: the AGX3b ldtg/sttg forms have no
 * encoding for it at any width. Setting it from the operand width regardless
 * of encoding refused every 8-bit threadgroup access on G17P, which is the
 * bulk of the Vulkan compute block's refusals once
 * VK_KHR_workgroup_memory_explicit_layout puts 8-bit types in shared memory.
 *
 * Pin both directions so the split cannot be re-broken: AGX3b must encode the
 * 8-bit access with the bit clear and must refuse it set, while AGX3c keeps
 * the bit (its own vectors above already cover the set case).
 */
static unsigned
test_local_narrow_extend(void)
{
   unsigned failures = 0;
   uint8_t out[AGX3_LOCAL_TPS_MAX_LENGTH];
   uint8_t len;

   for (unsigned store = 0; store < 2; ++store) {
      struct agx3_local_desc d = {
         .data = 4, .width = AGX3_MEM_W8, .mask = 0xf, .shift = 2,
         .base_kind = AGX3_LOCAL_BASE_REGISTER, .base = 0,
         .slot = store ? 0 : 6,
      };

      /* AGX3b: encodes with the bit clear ... */
      len = 0;
      d.narrow_extend = false;
      const bool clear_ok =
         store ? agx3_local_encode_store(AGX_ISA_ENCODING_AGX3B, &d, out, &len)
               : agx3_local_encode_load(AGX_ISA_ENCODING_AGX3B, &d, out, &len);
      if (!clear_ok) {
         fprintf(stderr, "FAIL: AGX3b 8-bit %s refused with f11 clear\n",
                 store ? "sttg" : "ldtg");
         ++failures;
      }

      /* ... and must refuse it set, because the form cannot express it. */
      len = 0;
      d.narrow_extend = true;
      const bool set_ok =
         store ? agx3_local_encode_store(AGX_ISA_ENCODING_AGX3B, &d, out, &len)
               : agx3_local_encode_load(AGX_ISA_ENCODING_AGX3B, &d, out, &len);
      if (set_ok) {
         fprintf(stderr, "FAIL: AGX3b 8-bit %s encoded f11 = 1, which has no "
                         "form\n", store ? "sttg" : "ldtg");
         ++failures;
      }
   }

   return failures;
}

int
main(void)
{
   const unsigned failures =
      test_local() + test_tps() + test_local_narrow_extend();
   if (failures)
      return 1;

   printf("agx3_local_tps_core: %zu Apple local/TPS vectors reproduced via "
          "auto and explicit forms\n",
          sizeof(local_vecs) / sizeof(local_vecs[0]) +
             sizeof(tps_vecs) / sizeof(tps_vecs[0]));
   return 0;
}
