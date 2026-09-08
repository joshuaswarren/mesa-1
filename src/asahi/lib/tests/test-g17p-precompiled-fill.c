/* Copyright 2026 Aurora Silicon; SPDX-License-Identifier: MIT */
#include "asahi/compiler/agx_compile.h"
#include "libagx_shaders.h"

#include <stdio.h>
#include <string.h>

static unsigned checks;
static unsigned failures;

#define CHECK(x)                                                               \
   do {                                                                        \
      ++checks;                                                                \
      if (!(x)) {                                                              \
         fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x);         \
         ++failures;                                                           \
      }                                                                        \
   } while (0)

static uint64_t
fnv1a64(const uint8_t *bytes, size_t size)
{
   uint64_t hash = UINT64_C(14695981039346656037);
   for (size_t i = 0; i < size; ++i) {
      hash ^= bytes[i];
      hash *= UINT64_C(1099511628211);
   }
   return hash;
}

int
main(void)
{
   /* Fresh metal-nt applegpu_g17p ground truth for the equivalent MSL kernel
    * is 54 bytes: two grid SR reads, one integer multiply-add, four value
    * moves, one 128-bit stdev and end.  Native artifact SHA-256:
    * 3474e39a65f602e913af23900d0205bd436c8c930dc4fa6f3571e539ae9437cd.
    *
    * This is the exact Mesa-ABI form of the same witnessed operations.  It
    * differs in register assignment and has one extra uniform move because
    * libagx receives its six arguments in contiguous uniform halves. */
   static const uint8_t expected[] = {
      0x0c, 0xa0, 0x10, 0x06, 0x1c, 0xa1, 0x10, 0x26,
      0x2b, 0x04, 0x01, 0x08, 0x9f, 0x30, 0x54, 0x00,
      0x02, 0x04, 0x10, 0x00, 0xd0, 0x2f, 0x2a, 0x00,
      0x4b, 0x06, 0x01, 0x08, 0x5b, 0x08, 0x01, 0x08,
      0x6b, 0x0a, 0x01, 0x08, 0x7b, 0x0c, 0x01, 0x08,
      0xe7, 0x00, 0x54, 0x08, 0x00, 0x00, 0x20, 0x00,
      0x17, 0x00, 0x00, 0x10, 0x10, 0x00, 0x0e, 0x00,
      0x00, 0x00,
   };

   for (unsigned i = 0; i < LIBAGX_NUM_PROGRAMS; ++i) {
      if (i == LIBAGX_FILL_UINT4 || i == LIBAGX_COPY_XFB_COUNTERS ||
          i == LIBAGX_PREDICATE_INDIRECT_0 ||
          i == LIBAGX_PREDICATE_INDIRECT_1)
         CHECK(libagx_g17p[i] != NULL);
      else
         CHECK(libagx_g17p[i] == NULL);
   }

   const uint32_t *blob = libagx_g17p[LIBAGX_FILL_UINT4];
   if (!blob)
      return 1;
   const struct agx_precompiled_kernel_info *info = (const void *)blob;
   const uint8_t *binary = (const uint8_t *)blob + sizeof(*info);
   CHECK(info->preamble_offset == 0);
   CHECK(info->main_offset == 0);
   CHECK(info->main_size == sizeof(expected));
   CHECK(info->binary_size == sizeof(expected));
   CHECK(info->nr_gprs == 16);
   CHECK(info->nr_preamble_gprs == 0);
   CHECK(info->push_count == 14);
   CHECK(info->workgroup_size[0] == 32);
   CHECK(info->workgroup_size[1] == 1);
   CHECK(info->workgroup_size[2] == 1);
   CHECK(info->local_size == 0);
   CHECK(info->imageblock_stride == 0);
   CHECK(!info->uses_txf);
   CHECK(info->agx3_staging_known);
   CHECK(info->agx3_staging_halfs == 0);
   CHECK(memcmp(binary, expected, sizeof(expected)) == 0);

   /* Transform feedback now uses its dedicated native compute counter-copy
    * kernel. Its byte-exact store behavior is covered by the XFB oracle; keep
    * this table test responsible for the generated launch contract. */
   const uint32_t *xfb_blob = libagx_g17p[LIBAGX_COPY_XFB_COUNTERS];
   CHECK(xfb_blob != NULL);
   if (!xfb_blob)
      return 1;
   const struct agx_precompiled_kernel_info *xfb_info =
      (const void *)xfb_blob;
   const uint8_t *xfb_binary =
      (const uint8_t *)xfb_blob + sizeof(*xfb_info);
   static const uint8_t end[] = {0x0e, 0x00, 0x00, 0x00};
   CHECK(xfb_info->preamble_offset == 0);
   CHECK(xfb_info->main_offset == 0);
   CHECK(xfb_info->main_size == 176);
   CHECK(xfb_info->binary_size == 176);
   CHECK(xfb_info->nr_gprs == 12);
   CHECK(xfb_info->push_count == 4);
   CHECK(xfb_info->workgroup_size[0] == 4);
   CHECK(xfb_info->workgroup_size[1] == 1);
   CHECK(xfb_info->workgroup_size[2] == 1);
   CHECK(xfb_info->local_size == 0);
   CHECK(!xfb_info->uses_txf);
   CHECK(xfb_info->agx3_staging_known);
   CHECK(xfb_info->agx3_staging_halfs == 0);
   CHECK(memcmp(xfb_binary + xfb_info->binary_size - sizeof(end), end,
                sizeof(end)) == 0);

   /* Apple G17P compiles the equivalent four- and five-word predicate/copy
    * kernels to the same control, address and device-store instruction
    * families used here. Fresh AIR/native SHA-256 pairs:
    *   ab91101e4182bd1656ac7602ad258174d1576451f2de55f770c64ad851ecc606
    *   b30f7801c83790963138415bf16dc9770faa22a00bb5bf08add355fd7a127fa3
    * Keep both generated variants byte-exact: the count path must never
    * silently fall back to an AGX2 precompiled binary on HAL200. */
   const struct {
      unsigned index;
      uint32_t size;
      uint64_t hash;
   } predicate[] = {
      {LIBAGX_PREDICATE_INDIRECT_0, 668, UINT64_C(0x3b6d443be271fbfb)},
      {LIBAGX_PREDICATE_INDIRECT_1, 814, UINT64_C(0x33746e6fd2961f9f)},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(predicate); ++i) {
      const uint32_t *predicate_blob = libagx_g17p[predicate[i].index];
      CHECK(predicate_blob != NULL);
      if (!predicate_blob)
         continue;
      const struct agx_precompiled_kernel_info *predicate_info =
         (const void *)predicate_blob;
      const uint8_t *predicate_binary =
         (const uint8_t *)predicate_blob + sizeof(*predicate_info);
      CHECK(predicate_info->preamble_offset == 0);
      CHECK(predicate_info->main_offset == 0);
      CHECK(predicate_info->main_size == predicate[i].size);
      CHECK(predicate_info->binary_size == predicate[i].size);
      CHECK(predicate_info->nr_gprs == 24);
      CHECK(predicate_info->nr_preamble_gprs == 0);
      CHECK(predicate_info->push_count == 14);
      CHECK(predicate_info->workgroup_size[0] == 32);
      CHECK(predicate_info->workgroup_size[1] == 1);
      CHECK(predicate_info->workgroup_size[2] == 1);
      CHECK(predicate_info->local_size == 0);
      CHECK(predicate_info->imageblock_stride == 0);
      CHECK(!predicate_info->uses_txf);
      CHECK(predicate_info->agx3_staging_known);
      CHECK(predicate_info->agx3_staging_halfs == 0);
      CHECK(fnv1a64(predicate_binary, predicate_info->binary_size) ==
            predicate[i].hash);
      CHECK(memcmp(predicate_binary + predicate_info->binary_size -
                      sizeof(end),
                   end, sizeof(end)) == 0);
   }

   if (failures) {
      fprintf(stderr, "g17p-precompiled-fill: %u/%u checks failed\n",
              failures, checks);
      return 1;
   }
   printf("g17p-precompiled-fill: %u checks passed\n", checks);
   return 0;
}
