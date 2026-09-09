/* Copyright 2026 Aurora Silicon; SPDX-License-Identifier: MIT */
#include "hk_pipeline_statistics_core.h"

#include <assert.h>
#include <string.h>

bool
hk_pipeline_statistics_layout_init(
   uint32_t query_count, VkQueryPipelineStatisticFlags statistics,
   struct hk_pipeline_statistics_layout *layout)
{
   if (!layout || !query_count || !statistics)
      return false;

   const uint32_t reports = util_bitcount(statistics);
   const uint64_t availability_B = (uint64_t)query_count * sizeof(uint32_t);
   const uint64_t query_start_B = (availability_B + 7) & ~UINT64_C(7);
   const uint64_t query_stride_B = (uint64_t)reports * sizeof(uint64_t);

   if (query_count > (UINT64_MAX - query_start_B) / query_stride_B)
      return false;

   *layout = (struct hk_pipeline_statistics_layout){
      .query_count = query_count,
      .reports_per_query = reports,
      .query_start_B = query_start_B,
      .query_stride_B = query_stride_B,
      .size_B = query_start_B + ((uint64_t)query_count * query_stride_B),
   };
   return true;
}

uint64_t
hk_pipeline_statistics_query_offset_B(
   const struct hk_pipeline_statistics_layout *layout, uint32_t query)
{
   assert(layout && query < layout->query_count);
   return layout->query_start_B + ((uint64_t)query * layout->query_stride_B);
}

void
hk_pipeline_statistics_reset_host(
   void *map, const struct hk_pipeline_statistics_layout *layout,
   uint32_t first_query, uint32_t query_count, bool set_available)
{
   assert(map && layout);
   assert(first_query <= layout->query_count);
   assert(query_count <= layout->query_count - first_query);

   uint32_t *availability = map;
   uint8_t *bytes = map;
   for (uint32_t i = 0; i < query_count; ++i) {
      const uint32_t query = first_query + i;
      availability[query] = 0;
      memset(bytes + hk_pipeline_statistics_query_offset_B(layout, query), 0,
             layout->query_stride_B);
      if (set_available)
         availability[query] = 1;
   }
}

static void
write_value(void *dst, uint32_t index, bool result_64, uint64_t value)
{
   if (result_64)
      ((uint64_t *)dst)[index] = value;
   else
      ((uint32_t *)dst)[index] = (uint32_t)value;
}

bool
hk_query_write_cpu_result(void *dst, const uint64_t *reports,
                          uint32_t report_count, bool available,
                          VkQueryResultFlags flags)
{
   assert(dst && reports && report_count);
   const bool write_reports =
      available || (flags & VK_QUERY_RESULT_PARTIAL_BIT);
   const bool result_64 = flags & VK_QUERY_RESULT_64_BIT;

   if (write_reports) {
      for (uint32_t i = 0; i < report_count; ++i)
         write_value(dst, i, result_64, reports[i]);
   }

   if (flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT)
      write_value(dst, report_count, result_64, available);

   return write_reports;
}
