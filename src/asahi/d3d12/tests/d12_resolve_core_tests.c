/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "../d12_resolve_core.h"
#include "../../libagx/resolve.h"

#include <stdio.h>
#include <stdlib.h>

#define CHECK(condition)                                                     \
   do {                                                                      \
      if (!(condition)) {                                                    \
         fprintf(stderr, "d12_resolve_core_tests: failed at line %d\n",     \
                 __LINE__);                                                  \
         exit(1);                                                            \
      }                                                                      \
   } while (0)

static void
check_format(uint32_t format, uint32_t shader_format, uint32_t pixel_size_B,
             uint32_t value_class, bool average)
{
   struct d12_resolve_core_format_info info;

   CHECK(d12_resolve_core_get_format(format, &info));
   CHECK(info.shader_format == shader_format);
   CHECK(info.pixel_size_B == pixel_size_B);
   CHECK(info.value_class == value_class);
   CHECK(d12_resolve_core_supports_mode(format, D12_RESOLVE_CORE_MODE_MIN));
   CHECK(d12_resolve_core_supports_mode(format, D12_RESOLVE_CORE_MODE_MAX));
   CHECK(d12_resolve_core_supports_average(format) == average);
}

int
main(void)
{
   uint32_t supported_count = 0;
   struct d12_resolve_core_format_info untouched = {
      .shader_format = UINT32_C(0xfeed),
      .pixel_size_B = UINT32_C(0xbeef),
      .mode_mask = UINT32_C(0xabcd),
      .value_class = UINT32_C(0x1234),
   };

   check_format(28, LIBAGX_RESOLVE_FORMAT_UNORM8_4, 4,
                D12_RESOLVE_CORE_VALUE_FLOAT, true);
   check_format(29, LIBAGX_RESOLVE_FORMAT_RGBA8_SRGB, 4,
                D12_RESOLVE_CORE_VALUE_FLOAT, true);
   check_format(10, LIBAGX_RESOLVE_FORMAT_FLOAT16_4, 8,
                D12_RESOLVE_CORE_VALUE_FLOAT, true);
   check_format(26, LIBAGX_RESOLVE_FORMAT_R11G11B10_FLOAT, 4,
                D12_RESOLVE_CORE_VALUE_FLOAT, true);
   check_format(40, LIBAGX_RESOLVE_FORMAT_FLOAT32_1, 4,
                D12_RESOLVE_CORE_VALUE_FLOAT, true);
   check_format(55, LIBAGX_RESOLVE_FORMAT_UNORM16_1, 2,
                D12_RESOLVE_CORE_VALUE_FLOAT, true);
   check_format(21, LIBAGX_RESOLVE_FORMAT_FLOAT32_1, 4,
                D12_RESOLVE_CORE_VALUE_FLOAT, true);
   check_format(22, LIBAGX_RESOLVE_FORMAT_UINT8_1, 1,
                D12_RESOLVE_CORE_VALUE_UINT, false);
   check_format(46, LIBAGX_RESOLVE_FORMAT_FLOAT32_1, 4,
                D12_RESOLVE_CORE_VALUE_FLOAT, true);
   check_format(47, LIBAGX_RESOLVE_FORMAT_UINT8_1, 1,
                D12_RESOLVE_CORE_VALUE_UINT, false);
   check_format(25, LIBAGX_RESOLVE_FORMAT_RGB10A2_UINT, 4,
                D12_RESOLVE_CORE_VALUE_UINT, false);
   check_format(32, LIBAGX_RESOLVE_FORMAT_SINT8_4, 4,
                D12_RESOLVE_CORE_VALUE_SINT, false);

   CHECK(!d12_resolve_core_get_format(27, &untouched));
   CHECK(untouched.shader_format == UINT32_C(0xfeed));
   CHECK(!d12_resolve_core_get_format(71, NULL));
   CHECK(!d12_resolve_core_supports_mode(28, 0));
   CHECK(!d12_resolve_core_supports_mode(30,
                                         D12_RESOLVE_CORE_MODE_AVERAGE));
   CHECK(d12_resolve_core_resource_compatible(28, 28));
   CHECK(d12_resolve_core_resource_compatible(27, 28));
   CHECK(d12_resolve_core_resource_compatible(27, 29));
   CHECK(d12_resolve_core_resource_compatible(28, 29));
   CHECK(d12_resolve_core_resource_compatible(29, 28));
   CHECK(d12_resolve_core_resource_compatible(29, 28));
   CHECK(d12_resolve_core_resource_compatible(28, 29));
   CHECK(d12_resolve_core_resource_compatible(91, 87));
   CHECK(d12_resolve_core_resource_compatible(87, 91));
   CHECK(d12_resolve_core_resource_compatible(29, 30));
   CHECK(!d12_resolve_core_resource_compatible(28, 87));
   CHECK(d12_resolve_core_resource_compatible(54, 56));
   CHECK(d12_resolve_core_resource_compatible(56, 54));
   CHECK(!d12_resolve_core_resource_compatible(29, 27));
   CHECK(!d12_resolve_core_resource_compatible(28, 71));
   CHECK(!d12_resolve_core_resource_compatible(27, 41));
   CHECK(d12_resolve_core_resource_compatible(19, 21));
   CHECK(d12_resolve_core_resource_compatible(20, 22));
   CHECK(d12_resolve_core_resource_compatible(45, 46));
   CHECK(!d12_resolve_core_resource_compatible(45, 40));
   CHECK(!d12_resolve_core_resource_compatible(28, 71));

   for (uint32_t format = 0; format < 192; ++format) {
      struct d12_resolve_core_format_info info;
      if (!d12_resolve_core_get_format(format, &info))
         continue;

      ++supported_count;
      CHECK(info.shader_format > LIBAGX_RESOLVE_FORMAT_INVALID);
      CHECK(info.shader_format <= LIBAGX_RESOLVE_FORMAT_R11G11B10_FLOAT);
      CHECK(info.pixel_size_B == 1 || info.pixel_size_B == 2 ||
            info.pixel_size_B == 4 || info.pixel_size_B == 8 ||
            info.pixel_size_B == 16);
      CHECK(info.mode_mask & D12_RESOLVE_CORE_SUPPORT_MIN);
      CHECK(info.mode_mask & D12_RESOLVE_CORE_SUPPORT_MAX);
      CHECK(((info.mode_mask & D12_RESOLVE_CORE_SUPPORT_AVERAGE) != 0) ==
            (info.value_class == D12_RESOLVE_CORE_VALUE_FLOAT));
   }
   CHECK(supported_count == 52);

   puts("d12_resolve_core_tests: PASS");
   return 0;
}
