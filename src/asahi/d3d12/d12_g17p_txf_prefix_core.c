/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT */
#include "d12_g17p_txf_prefix_core.h"
#include <string.h>

static bool cell(uint64_t address)
{
   const uint64_t limit = UINT64_C(1) << 48;
   return address && !(address & 7) && address <= limit - 8;
}

bool
d12_g17p_txf_prefix_ranges(const struct d12_g17p_txf_prefix_sources *s,
                            struct d12_g17p_txf_range out[4])
{
   const uint64_t limit = UINT64_C(1) << 48;
   if (!s || !out || !cell(s->root_cell) || !cell(s->texture_cell) ||
       !cell(s->sampler_cell) || s->root_cell == s->texture_cell ||
       s->root_cell == s->sampler_cell || s->texture_cell == s->sampler_cell ||
       !s->texture_table ||
       (s->texture_table & 31) || !s->texture_table_size_B ||
       (s->texture_table_size_B & 31) || s->texture_table >= limit ||
       s->texture_table_size_B > limit - s->texture_table ||
       !s->sampler_table || (s->sampler_table & 31) ||
       s->sampler_table > limit - 32)
      return false;
   struct d12_g17p_txf_range staged[4];
   memset(staged, 0, sizeof(staged));
   staged[0] = (struct d12_g17p_txf_range){0, 4, s->texture_cell};
   staged[1] = (struct d12_g17p_txf_range){4, 4, s->sampler_cell};
   staged[2] = (struct d12_g17p_txf_range){8, 4, s->root_cell};
   staged[3] = (struct d12_g17p_txf_range){16, 4, s->texture_cell};
   memcpy(out, staged, sizeof(staged));
   return true;
}
