/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT */
#pragma once
#include <stdbool.h>
#include <stdint.h>

struct d12_g17p_txf_range {
   uint16_t start_halfs, size_halfs;
   uint64_t address;
};
struct d12_g17p_txf_prefix_sources {
   uint64_t root_cell, texture_cell, sampler_cell;
   uint64_t texture_table, texture_table_size_B, sampler_table;
};
/* Native pointers0..7, root8..11, ordinary D12 query pointer16..19.
 * Output is transactional. The caller proves all cells/table ownership. */
bool d12_g17p_txf_prefix_ranges(
   const struct d12_g17p_txf_prefix_sources *sources,
   struct d12_g17p_txf_range ranges[4]);
