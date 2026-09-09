/* Copyright 2026 Aurora Silicon; SPDX-License-Identifier: MIT */
#include "hk_pipeline_statistics_core.h"

#include "util/macros.h"

#include <stdio.h>
#include <stdlib.h>
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

static const VkQueryPipelineStatisticFlagBits g17p_statistics[] = {
   VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_VERTICES_BIT,
   VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_PRIMITIVES_BIT,
   VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT,
   VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT,
   VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT,
   VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT,
   VK_QUERY_PIPELINE_STATISTIC_COMPUTE_SHADER_INVOCATIONS_BIT,
};

static void
test_compact_indices(void)
{
   VkQueryPipelineStatisticFlags reconstructed = 0;
   for (uint32_t i = 0; i < ARRAY_SIZE(g17p_statistics); ++i) {
      uint32_t index = UINT32_MAX;
      reconstructed |= g17p_statistics[i];
      CHECK(hk_pipeline_statistics_report_index(
         HK_G17P_PIPELINE_STATISTICS_MASK, g17p_statistics[i], &index));
      CHECK(index == i);
   }
   CHECK(reconstructed == HK_G17P_PIPELINE_STATISTICS_MASK);

   /* Compact ordering counts selected lower bits, including stages that G17P
    * does not expose. Query-pool ABI is still defined for such masks and the
    * inapplicable counters remain zero. */
   const VkQueryPipelineStatisticFlags sparse =
      VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_VERTICES_BIT |
      VK_QUERY_PIPELINE_STATISTIC_GEOMETRY_SHADER_INVOCATIONS_BIT |
      VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT |
      VK_QUERY_PIPELINE_STATISTIC_COMPUTE_SHADER_INVOCATIONS_BIT;
   uint32_t index = UINT32_MAX;
   CHECK(hk_pipeline_statistics_report_index(
      sparse, VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT,
      &index));
   CHECK(index == 2);
   CHECK(!hk_pipeline_statistics_report_index(
      sparse, VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT, &index));
   CHECK(!hk_pipeline_statistics_report_index(
      sparse, (VkQueryPipelineStatisticFlagBits)0, &index));
   CHECK(!hk_pipeline_statistics_report_index(
      sparse, (VkQueryPipelineStatisticFlagBits)3, &index));
   CHECK(!hk_pipeline_statistics_report_index(
      sparse, VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT,
      NULL));
}

static void
test_layout_reset_and_reuse(void)
{
   struct hk_pipeline_statistics_layout layout;
   CHECK(hk_pipeline_statistics_layout_init(
      3, HK_G17P_PIPELINE_STATISTICS_MASK, &layout));
   CHECK(layout.query_count == 3);
   CHECK(layout.reports_per_query == ARRAY_SIZE(g17p_statistics));
   CHECK(layout.query_start_B == 16);
   CHECK(layout.query_stride_B == 56);
   CHECK(layout.size_B == 184);
   CHECK(hk_pipeline_statistics_query_offset_B(&layout, 0) == 16);
   CHECK(hk_pipeline_statistics_query_offset_B(&layout, 1) == 72);
   CHECK(hk_pipeline_statistics_query_offset_B(&layout, 2) == 128);

   _Alignas(8) uint8_t guarded[8 + 184 + 8];
   memset(guarded, 0xa5, sizeof(guarded));
   void *map = guarded + 8;
   hk_pipeline_statistics_reset_host(map, &layout, 0, 3, false);
   for (unsigned i = 0; i < 8; ++i) {
      CHECK(guarded[i] == 0xa5);
      CHECK(guarded[8 + layout.size_B + i] == 0xa5);
   }

   uint32_t *availability = map;
   for (uint32_t q = 0; q < layout.query_count; ++q) {
      CHECK(availability[q] == 0);
      const uint64_t *reports =
         (const void *)((const uint8_t *)map +
                        hk_pipeline_statistics_query_offset_B(&layout, q));
      for (uint32_t i = 0; i < layout.reports_per_query; ++i)
         CHECK(reports[i] == 0);
   }

   uint64_t *q0 =
      (void *)((uint8_t *)map +
               hk_pipeline_statistics_query_offset_B(&layout, 0));
   uint64_t *q1 =
      (void *)((uint8_t *)map +
               hk_pipeline_statistics_query_offset_B(&layout, 1));
   uint64_t *q2 =
      (void *)((uint8_t *)map +
               hk_pipeline_statistics_query_offset_B(&layout, 2));
   for (uint32_t i = 0; i < layout.reports_per_query; ++i) {
      q0[i] = UINT64_C(0x100000000) + 11 * (i + 1);
      q1[i] = 100 + i;
      q2[i] = 200 + i;
   }
   availability[0] = 1;
   availability[1] = 1;
   availability[2] = 1;

   hk_pipeline_statistics_reset_host(map, &layout, 1, 1, false);
   CHECK(availability[0] == 1);
   CHECK(availability[1] == 0);
   CHECK(availability[2] == 1);
   for (uint32_t i = 0; i < layout.reports_per_query; ++i) {
      CHECK(q0[i] == UINT64_C(0x100000000) + 11 * (i + 1));
      CHECK(q1[i] == 0);
      CHECK(q2[i] == 200 + i);
   }

   /* Reuse is reset followed by new increments, not accumulation from the
    * prior generation. */
   for (uint32_t i = 0; i < layout.reports_per_query; ++i)
      q1[i] += 7 * (i + 1);
   availability[1] = 1;
   for (uint32_t i = 0; i < layout.reports_per_query; ++i)
      CHECK(q1[i] == 7 * (i + 1));

   hk_pipeline_statistics_reset_host(map, &layout, 2, 1, true);
   CHECK(availability[2] == 1);
   for (uint32_t i = 0; i < layout.reports_per_query; ++i)
      CHECK(q2[i] == 0);
}

static void
test_result_abi_and_guards(void)
{
   const uint64_t reports[] = {
      UINT64_C(0x10000000b), UINT64_C(0x200000016),
      UINT64_C(0x300000021), UINT64_C(0x40000002c),
      UINT64_C(0x500000037), UINT64_C(0x600000042),
      UINT64_C(0x70000004d),
   };
   struct {
      uint64_t prefix;
      uint64_t values[ARRAY_SIZE(reports)];
      uint64_t availability;
      uint64_t suffix;
   } out64;
   memset(&out64, 0xa5, sizeof(out64));
   const uint64_t guard = UINT64_C(0xa5a5a5a5a5a5a5a5);

   CHECK(hk_query_write_cpu_result(
      out64.values, reports, ARRAY_SIZE(reports), true,
      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT));
   CHECK(out64.prefix == guard && out64.suffix == guard);
   CHECK(out64.availability == 1);
   for (uint32_t i = 0; i < ARRAY_SIZE(reports); ++i)
      CHECK(out64.values[i] == reports[i]);

   memset(out64.values, 0xa5,
          sizeof(out64.values) + sizeof(out64.availability));
   CHECK(!hk_query_write_cpu_result(
      out64.values, reports, ARRAY_SIZE(reports), false,
      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT));
   for (uint32_t i = 0; i < ARRAY_SIZE(reports); ++i)
      CHECK(out64.values[i] == guard);
   CHECK(out64.availability == 0);
   CHECK(out64.prefix == guard && out64.suffix == guard);

   CHECK(hk_query_write_cpu_result(
      out64.values, reports, ARRAY_SIZE(reports), false,
      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_PARTIAL_BIT |
         VK_QUERY_RESULT_WITH_AVAILABILITY_BIT));
   for (uint32_t i = 0; i < ARRAY_SIZE(reports); ++i)
      CHECK(out64.values[i] == reports[i]);
   CHECK(out64.availability == 0);

   struct {
      uint32_t prefix;
      uint32_t values[ARRAY_SIZE(reports)];
      uint32_t availability;
      uint32_t suffix;
   } out32;
   memset(&out32, 0xa5, sizeof(out32));
   CHECK(hk_query_write_cpu_result(
      out32.values, reports, ARRAY_SIZE(reports), true,
      VK_QUERY_RESULT_WITH_AVAILABILITY_BIT));
   CHECK(out32.prefix == UINT32_C(0xa5a5a5a5));
   CHECK(out32.suffix == UINT32_C(0xa5a5a5a5));
   CHECK(out32.availability == 1);
   for (uint32_t i = 0; i < ARRAY_SIZE(reports); ++i)
      CHECK(out32.values[i] == (uint32_t)reports[i]);
}

static void
test_layout_limits(void)
{
   struct hk_pipeline_statistics_layout layout;
   CHECK(!hk_pipeline_statistics_layout_init(
      0, HK_G17P_PIPELINE_STATISTICS_MASK, &layout));
   CHECK(!hk_pipeline_statistics_layout_init(1, 0, &layout));
   CHECK(!hk_pipeline_statistics_layout_init(
      1, HK_G17P_PIPELINE_STATISTICS_MASK, NULL));

   CHECK(hk_pipeline_statistics_layout_init(
      UINT32_MAX, HK_G17P_PIPELINE_STATISTICS_MASK, &layout));
   CHECK(layout.query_start_B == UINT64_C(17179869184));
   CHECK(layout.query_stride_B == 56);
   CHECK(layout.size_B == UINT64_C(257698037704));
}

int
main(void)
{
   test_compact_indices();
   test_layout_reset_and_reuse();
   test_result_abi_and_guards();
   test_layout_limits();

   if (failures) {
      fprintf(stderr, "%u/%u pipeline-statistics checks failed\n", failures,
              checks);
      return EXIT_FAILURE;
   }

   printf("hk-pipeline-statistics-core: %u checks passed\n", checks);
   return EXIT_SUCCESS;
}
