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
      if (i == LIBAGX_FILL_UINT4)
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

   if (failures) {
      fprintf(stderr, "g17p-precompiled-fill: %u/%u checks failed\n",
              failures, checks);
      return 1;
   }
   printf("g17p-precompiled-fill: %u checks passed\n", checks);
   return 0;
}
