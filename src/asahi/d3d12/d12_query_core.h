/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Frontend-neutral D3D12 query classification.  The public D3D12 and
 * baseline D3D12 DDI enums intentionally use the same numeric ABI values;
 * keeping the policy here prevents the COM and WDDM frontends from drifting.
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>

enum d12_query_core_heap_type {
   D12_QUERY_CORE_HEAP_OCCLUSION = 0,
   D12_QUERY_CORE_HEAP_TIMESTAMP = 1,
   D12_QUERY_CORE_HEAP_PIPELINE_STATISTICS = 2,
   D12_QUERY_CORE_HEAP_SO_STATISTICS = 3,
   D12_QUERY_CORE_HEAP_COPY_QUEUE_TIMESTAMP = 5,
};

enum d12_query_core_type {
   D12_QUERY_CORE_TYPE_OCCLUSION = 0,
   D12_QUERY_CORE_TYPE_BINARY_OCCLUSION = 1,
   D12_QUERY_CORE_TYPE_TIMESTAMP = 2,
   D12_QUERY_CORE_TYPE_PIPELINE_STATISTICS = 3,
   D12_QUERY_CORE_TYPE_SO_STATISTICS_STREAM0 = 4,
   D12_QUERY_CORE_TYPE_SO_STATISTICS_STREAM3 = 7,
};

enum d12_query_core_command_type {
   D12_QUERY_CORE_COMMAND_DIRECT = 0,
   D12_QUERY_CORE_COMMAND_BUNDLE = 1,
   D12_QUERY_CORE_COMMAND_COMPUTE = 2,
   D12_QUERY_CORE_COMMAND_COPY = 3,
};

bool d12_query_core_heap_supported(uint32_t heap_type);
bool d12_query_core_type_matches(uint32_t heap_type, uint32_t query_type);
bool d12_query_core_is_timestamp_heap(uint32_t heap_type);
bool d12_query_core_command_compatible(uint32_t heap_type,
                                        uint32_t command_type);
bool d12_query_core_resolve_range_valid(uint32_t heap_count, uint32_t first,
                                        uint32_t count, uint32_t stride,
                                        uint64_t destination_size,
                                        uint64_t destination_offset);
