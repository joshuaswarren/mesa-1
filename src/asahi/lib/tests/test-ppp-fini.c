/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "../agx_ppp.h"
#include "../agx_va_runtime_hole.h"

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

static const enum agx_chip agx2_targets[] = {
   AGX_CHIP_G13G,
   AGX_CHIP_G13X,
   AGX_CHIP_G14G,
   AGX_CHIP_G14X,
};

static const enum agx_chip agx3_targets[] = {
   AGX_CHIP_G15G, AGX_CHIP_G15S, AGX_CHIP_G15C, AGX_CHIP_G15D,
   AGX_CHIP_G16G, AGX_CHIP_G16S, AGX_CHIP_G16C, AGX_CHIP_G17P,
   AGX_CHIP_G17G, AGX_CHIP_G17S, AGX_CHIP_G17C,
};

static void
legacy_reference(uint64_t gpu, uint32_t size_words,
                 uint8_t expected[AGX_PPP_STATE_LENGTH])
{
   struct agx_ppp_state_packed packed;
   const struct AGX_PPP_STATE cfg = {
      .pointer_hi = gpu >> 32,
      .pointer_lo = (uint32_t)gpu,
      .size_words = size_words,
   };

   AGX_PPP_STATE_pack(packed.opaque, &cfg);
   memcpy(expected, &packed, AGX_PPP_STATE_LENGTH);
}

static int
check_outer_state_all_generations(void)
{
   const uint64_t gpu = UINT64_C(0x1234567898);
   const uint32_t size_words = 0x82;
   uint8_t legacy[AGX_PPP_STATE_LENGTH];
   legacy_reference(gpu, size_words, legacy);

   static const uint8_t witness[AGX_PPP_STATE_LENGTH] = {
      0x12, 0x82, 0x00, 0x00, 0x98, 0x78, 0x56, 0x34,
   };
   CHECK(memcmp(legacy, witness, sizeof(witness)) == 0);

   for (unsigned i = 0; i < sizeof(agx2_targets) / sizeof(agx2_targets[0]);
        ++i) {
      uint8_t stream[AGX_PPP_STATE_LENGTH + 2];
      memset(stream, 0xa5, sizeof(stream));
      uint8_t *out = &stream[1];
      CHECK(agx_ppp_emit_state(agx2_targets[i], &out, gpu,
                               size_words * 4u));
      CHECK(out == &stream[1 + AGX_PPP_STATE_LENGTH]);
      CHECK(memcmp(&stream[1], legacy, sizeof(legacy)) == 0);
      CHECK(stream[0] == 0xa5 && stream[sizeof(stream) - 1] == 0xa5);

      for (unsigned bit = 2; bit < 40; ++bit) {
         const uint64_t pointer = UINT64_C(1) << bit;
         uint8_t expected[AGX_PPP_STATE_LENGTH];
         legacy_reference(pointer, 0, expected);
         out = &stream[1];
         CHECK(agx_ppp_emit_state(agx2_targets[i], &out, pointer, 0));
         CHECK(memcmp(&stream[1], expected, sizeof(expected)) == 0);
      }

      for (unsigned bit = 0; bit < 8; ++bit) {
         const uint32_t words = UINT32_C(1) << bit;
         uint8_t expected[AGX_PPP_STATE_LENGTH];
         legacy_reference(0, words, expected);
         out = &stream[1];
         CHECK(agx_ppp_emit_state(agx2_targets[i], &out, 0, words * 4u));
         CHECK(memcmp(&stream[1], expected, sizeof(expected)) == 0);
      }
   }

   for (unsigned i = 0; i < sizeof(agx3_targets) / sizeof(agx3_targets[0]);
        ++i) {
      struct agx_vdm_ppp_state_core shared;
      uint8_t stream[AGX_PPP_STATE_LENGTH + 2];
      memset(stream, 0xa5, sizeof(stream));
      uint8_t *out = &stream[1];
      CHECK(agx_vdm_ppp_state_core_pack(agx3_targets[i], gpu, size_words,
                                        &shared));
      CHECK(agx_ppp_emit_state(agx3_targets[i], &out, gpu,
                               size_words * 4u));
      CHECK(out == &stream[1 + AGX_PPP_STATE_LENGTH]);
      CHECK(memcmp(&stream[1], shared.bytes, shared.length) == 0);
      CHECK(memcmp(&stream[1], witness, sizeof(witness)) == 0);
      CHECK(stream[0] == 0xa5 && stream[sizeof(stream) - 1] == 0xa5);

      for (unsigned bit = 2; bit < 40; ++bit) {
         const uint64_t pointer = UINT64_C(1) << bit;
         CHECK(agx_vdm_ppp_state_core_pack(agx3_targets[i], pointer, 0,
                                           &shared));
         out = &stream[1];
         CHECK(agx_ppp_emit_state(agx3_targets[i], &out, pointer, 0));
         CHECK(memcmp(&stream[1], shared.bytes, shared.length) == 0);
      }

      for (unsigned bit = 0; bit < 8; ++bit) {
         const uint32_t words = UINT32_C(1) << bit;
         CHECK(agx_vdm_ppp_state_core_pack(agx3_targets[i], 0, words,
                                           &shared));
         out = &stream[1];
         CHECK(agx_ppp_emit_state(agx3_targets[i], &out, 0, words * 4u));
         CHECK(memcmp(&stream[1], shared.bytes, shared.length) == 0);
      }
   }

   return 0;
}

static int
check_live_legacy_payload_gate(void)
{
   for (unsigned i = 0; i < sizeof(agx2_targets) / sizeof(agx2_targets[0]);
        ++i)
      CHECK(agx_ppp_legacy_payload_supported(agx2_targets[i]));

   for (unsigned i = 0; i < 8; ++i)
      CHECK(agx_ppp_legacy_payload_supported(agx3_targets[i]));

   for (unsigned i = 8; i < sizeof(agx3_targets) / sizeof(agx3_targets[0]);
        ++i)
      CHECK(!agx_ppp_legacy_payload_supported(agx3_targets[i]));

   CHECK(!agx_ppp_legacy_payload_supported(AGX_CHIP_G15P));
   CHECK(!agx_ppp_legacy_payload_supported(AGX_CHIP_G16P));
   CHECK(!agx_ppp_legacy_payload_supported(AGX_CHIP_G17A));
   CHECK(!agx_ppp_legacy_payload_supported(AGX_CHIP_UNKNOWN));
   return 0;
}

static int
check_complete_update_finalization(void)
{
   uint8_t payload[16] = {0};
   struct agx_ppp_update ppp = {
      .head = payload + sizeof(payload),
      .gpu_base = UINT64_C(0x1234567800),
      .total_size = sizeof(payload),
      .cpu_base = payload,
   };
   uint8_t expected[AGX_PPP_STATE_LENGTH];
   legacy_reference(ppp.gpu_base, sizeof(payload) / 4, expected);

   for (unsigned i = 0; i < sizeof(agx2_targets) / sizeof(agx2_targets[0]);
        ++i) {
      uint8_t stream[AGX_PPP_STATE_LENGTH + 1];
      memset(stream, 0xa5, sizeof(stream));
      uint8_t *out = stream;
      CHECK(agx_ppp_fini(agx2_targets[i], &out, &ppp));
      CHECK(out == stream + AGX_PPP_STATE_LENGTH);
      CHECK(memcmp(stream, expected, sizeof(expected)) == 0);
      CHECK(stream[AGX_PPP_STATE_LENGTH] == 0xa5);
   }

   for (unsigned i = 0; i < sizeof(agx3_targets) / sizeof(agx3_targets[0]);
        ++i) {
      uint8_t stream[AGX_PPP_STATE_LENGTH + 1];
      memset(stream, 0xa5, sizeof(stream));
      uint8_t *out = stream;
      CHECK(agx_ppp_fini(agx3_targets[i], &out, &ppp));
      CHECK(out == stream + AGX_PPP_STATE_LENGTH);
      CHECK(memcmp(stream, expected, sizeof(expected)) == 0);
      CHECK(stream[AGX_PPP_STATE_LENGTH] == 0xa5);
   }

   uint8_t stream[AGX_PPP_STATE_LENGTH];
   memset(stream, 0xa5, sizeof(stream));
   uint8_t before[sizeof(stream)];
   memcpy(before, stream, sizeof(stream));
   uint8_t *out = stream;

   ppp.head--;
   CHECK(!agx_ppp_fini(AGX_CHIP_G15G, &out, &ppp));
   CHECK(out == stream && memcmp(stream, before, sizeof(stream)) == 0);
   ppp.head++;
   ppp.cpu_base = NULL;
   CHECK(!agx_ppp_fini(AGX_CHIP_G15G, &out, &ppp));
   CHECK(out == stream && memcmp(stream, before, sizeof(stream)) == 0);
   CHECK(!agx_ppp_fini(AGX_CHIP_G15G, &out, NULL));
   CHECK(out == stream && memcmp(stream, before, sizeof(stream)) == 0);
   CHECK(!agx_ppp_fini(AGX_CHIP_G15G, NULL, &ppp));
   return 0;
}

static int
check_g17p_context_relative_payload(void)
{
   uint8_t payload[16] = {0};
   uint64_t compact = UINT64_MAX;
   CHECK(agx_g17p_vdm_payload_compact(UINT64_C(0x5fffee0000), &compact));
   CHECK(compact == UINT64_C(0x4fffee0000));

   struct agx_ppp_update ppp = {
      .head = payload + sizeof(payload),
      .gpu_base = compact,
      .total_size = sizeof(payload),
      .cpu_base = payload,
   };
   uint8_t stream[AGX_PPP_STATE_LENGTH + 2];
   memset(stream, 0xa5, sizeof(stream));
   uint8_t *out = &stream[1];
   CHECK(agx_ppp_fini(AGX_CHIP_G17P, &out, &ppp));

   static const uint8_t expected[AGX_PPP_STATE_LENGTH] = {
      0x4f, 0x04, 0x00, 0x00, 0x00, 0x00, 0xee, 0xff,
   };
   CHECK(out == &stream[1 + AGX_PPP_STATE_LENGTH]);
   CHECK(memcmp(&stream[1], expected, sizeof(expected)) == 0);
   CHECK(stream[0] == 0xa5 && stream[sizeof(stream) - 1] == 0xa5);
   return 0;
}

static int
check_transactional_rejections(void)
{
   static const enum agx_chip unsupported[] = {
      AGX_CHIP_G15P, AGX_CHIP_G16P, AGX_CHIP_G17A, AGX_CHIP_UNKNOWN,
      (enum agx_chip)999,
   };

   uint8_t stream[AGX_PPP_STATE_LENGTH];
   uint8_t before[sizeof(stream)];
   for (unsigned i = 0; i < sizeof(unsupported) / sizeof(unsupported[0]);
        ++i) {
      memset(stream, 0xa5, sizeof(stream));
      memcpy(before, stream, sizeof(stream));
      uint8_t *out = stream;
      CHECK(!agx_ppp_emit_state(unsupported[i], &out,
                                 UINT64_C(0x1234567800), 16));
      CHECK(out == stream && memcmp(stream, before, sizeof(stream)) == 0);
   }

   const uint64_t invalid_gpu[] = {
      1,
      2,
      3,
      UINT64_C(0x1234567801),
      UINT64_C(1) << 40,
      UINT64_MAX,
   };
   for (unsigned i = 0; i < sizeof(invalid_gpu) / sizeof(invalid_gpu[0]);
        ++i) {
      memset(stream, 0xa5, sizeof(stream));
      memcpy(before, stream, sizeof(stream));
      uint8_t *out = stream;
      CHECK(!agx_ppp_emit_state(AGX_CHIP_G15G, &out, invalid_gpu[i], 16));
      CHECK(out == stream && memcmp(stream, before, sizeof(stream)) == 0);
   }

   static const size_t invalid_sizes[] = {1, 2, 3, 1024, SIZE_MAX};
   for (unsigned i = 0; i < sizeof(invalid_sizes) / sizeof(invalid_sizes[0]);
        ++i) {
      memset(stream, 0xa5, sizeof(stream));
      memcpy(before, stream, sizeof(stream));
      uint8_t *out = stream;
      CHECK(!agx_ppp_emit_state(AGX_CHIP_G15G, &out,
                                 UINT64_C(0x1234567800), invalid_sizes[i]));
      CHECK(out == stream && memcmp(stream, before, sizeof(stream)) == 0);
   }

   uint8_t *null_out = NULL;
   CHECK(!agx_ppp_emit_state(AGX_CHIP_G15G, &null_out,
                              UINT64_C(0x1234567800), 16));
   CHECK(!agx_ppp_emit_state(AGX_CHIP_G15G, NULL,
                              UINT64_C(0x1234567800), 16));
   return 0;
}

int
main(void)
{
   if (check_outer_state_all_generations() ||
       check_live_legacy_payload_gate() ||
       check_complete_update_finalization() ||
       check_g17p_context_relative_payload() ||
       check_transactional_rejections())
      return 1;

   printf("agx PPP finalizer: %u checks passed\n", checks);
   return 0;
}
