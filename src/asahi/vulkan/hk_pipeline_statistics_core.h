/* Copyright 2026 Aurora Silicon; SPDX-License-Identifier: MIT */
#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "util/bitscan.h"
#include "vulkan/vulkan_core.h"

struct hk_pipeline_statistics_layout {
   uint32_t query_count;
   uint32_t reports_per_query;
   uint64_t query_start_B;
   uint64_t query_stride_B;
   uint64_t size_B;
};

bool hk_pipeline_statistics_layout_init(
   uint32_t query_count, VkQueryPipelineStatisticFlags statistics,
   struct hk_pipeline_statistics_layout *layout);

static inline bool
hk_pipeline_statistics_report_index(
   VkQueryPipelineStatisticFlags statistics,
   VkQueryPipelineStatisticFlagBits statistic, uint32_t *index)
{
   const uint32_t bit = (uint32_t)statistic;
   if (!index || !bit || (bit & (bit - 1)) || !(statistics & bit))
      return false;

   *index = util_bitcount(statistics & (bit - 1));
   return true;
}

uint64_t hk_pipeline_statistics_query_offset_B(
   const struct hk_pipeline_statistics_layout *layout, uint32_t query);

void hk_pipeline_statistics_reset_host(
   void *map, const struct hk_pipeline_statistics_layout *layout,
   uint32_t first_query, uint32_t query_count, bool set_available);

/* Writes one query using Vulkan's compact result ABI. Returns whether report
 * values (as opposed to availability alone) were written. */
bool hk_query_write_cpu_result(void *dst, const uint64_t *reports,
                               uint32_t report_count, bool available,
                               VkQueryResultFlags flags);
