/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_va_runtime_hole.h"

#include "util/macros.h"
#include "util/u_math.h"

struct agx_va_interval {
   uint64_t start;
   uint64_t end;
};

bool
agx_g17p_vdm_alias_heap_init(struct util_vma_heap *heap)
{
   if (!heap)
      return false;

   util_vma_heap_init(heap, AGX_G17P_VDM_ALIAS_BASE,
                      AGX_G17P_VDM_ALIAS_SIZE);
   heap->alloc_high = false;
   return heap->free_size == AGX_G17P_VDM_ALIAS_SIZE;
}

bool
agx_g17p_vdm_alias_range_valid(uint64_t addr, uint64_t size)
{
   if (size == 0 || addr < AGX_G17P_VDM_ALIAS_BASE ||
       size > UINT64_MAX - addr)
      return false;

   return addr + size <=
          AGX_G17P_VDM_USER_BASE + AGX_G17P_VDM_ALIAS_OFFSET_END;
}

bool
agx_g17p_vdm_alias_compact(uint64_t addr, uint32_t *offset)
{
   if (!offset || !agx_g17p_vdm_alias_range_valid(addr, 1))
      return false;

   const uint64_t compact = addr - AGX_G17P_VDM_USER_BASE;
   if (compact > UINT32_MAX)
      return false;

   *offset = (uint32_t)compact;
   return true;
}

bool
agx_g17p_vdm_payload_compact(uint64_t addr, uint64_t *compact)
{
   if (!compact || addr < AGX_G17P_VDM_USER_BASE ||
       addr >= AGX_G17P_VDM_USER_LIMIT || (addr & 3))
      return false;

   *compact = addr - AGX_G17P_VDM_USER_BASE;
   return true;
}

bool
agx_va_split_user_window(uint64_t user_start, uint64_t user_size,
                         uint64_t *out_start, uint64_t *out_size)
{
   if (!out_start || !out_size || user_start == 0 || user_size == 0)
      return false;

   /* Round the user VA window to powers-of-two... */
   user_start = util_next_power_of_two64(user_start);
   user_size = util_next_power_of_two64(user_size + 1) >> 1;

   /* ...so when we cut user size in half to emulate sparse buffers... */
   user_size /= 2;

   /*
    * ...and again as many times as it takes for the whole window to sit below
    * the bit we are about to steal.  Upstream halves at most once more, which
    * is exactly enough for AGX2: a 39-bit user top leaves a window equal to
    * the rounded start, so one extra halving lands it below.  AGX3 needs the
    * loop -- T8140's 42-bit root leaves a window eight times the start, so a
    * single fixup would still straddle the sparse bit and the read-only alias
    * of a high allocation would carry out of its own half.
    */
   while (user_size >= user_start)
      user_size /= 2;

   if (user_size == 0)
      return false;

   /* ...we can distinguish the top/bottom half by an address bit */
   if ((user_start & user_size) != 0 ||
       ((user_start + (user_size - 1)) & user_size) != 0)
      return false;

   *out_start = user_start;
   *out_size = user_size;
   return true;
}

bool
agx_va_overlaps_runtime_graph_hole(enum agx_chip chip, uint64_t addr,
                                   uint64_t size)
{
   if (chip != AGX_CHIP_G17P || size == 0 ||
       addr >= AGX_G17_RUNTIME_GRAPH_VA_END)
      return false;

   return addr >= AGX_G17_RUNTIME_GRAPH_VA_START ||
          size > AGX_G17_RUNTIME_GRAPH_VA_START - addr;
}

static bool
range_end(uint64_t start, uint64_t size, uint64_t *end)
{
   if (size > UINT64_MAX - start)
      return false;

   *end = start + size;
   return true;
}

static void
add_intersection(struct agx_va_interval *intervals, unsigned *count,
                 uint64_t heap_start, uint64_t heap_end,
                 uint64_t reserved_start, uint64_t reserved_end)
{
   const uint64_t start = MAX2(heap_start, reserved_start);
   const uint64_t end = MIN2(heap_end, reserved_end);

   if (start < end)
      intervals[(*count)++] = (struct agx_va_interval){start, end};
}

static bool
reserve_intervals(struct util_vma_heap *heap,
                  struct agx_va_interval *intervals, unsigned count)
{
   if (count == 2 && intervals[1].start < intervals[0].start) {
      const struct agx_va_interval tmp = intervals[0];
      intervals[0] = intervals[1];
      intervals[1] = tmp;
   }

   if (count == 2 && intervals[1].start <= intervals[0].end) {
      intervals[0].end = MAX2(intervals[0].end, intervals[1].end);
      count = 1;
   }

   for (unsigned i = 0; i < count; ++i) {
      if (!util_vma_heap_alloc_addr(heap, intervals[i].start,
                                    intervals[i].end - intervals[i].start))
         return false;
   }

   return true;
}

bool
agx_va_reserve_runtime_graph_hole(
   enum agx_chip chip, struct util_vma_heap *main_heap, uint64_t main_start,
   uint64_t main_size, uint64_t sparse_ro_offset,
   struct util_vma_heap *usc_heap, uint64_t usc_start, uint64_t usc_size)
{
   if (chip != AGX_CHIP_G17P)
      return true;

   uint64_t main_end, usc_end;
   if (!range_end(main_start, main_size, &main_end) ||
       !range_end(usc_start, usc_size, &usc_end))
      return false;

   struct agx_va_interval main_intervals[2];
   unsigned main_count = 0;

   add_intersection(main_intervals, &main_count, main_start, main_end,
                    AGX_G17_RUNTIME_GRAPH_VA_START,
                    AGX_G17_RUNTIME_GRAPH_VA_END);

   /* Sparse buffers bind the main allocation again at sparse_ro_offset. */
   if (sparse_ro_offset < AGX_G17_RUNTIME_GRAPH_VA_END) {
      const uint64_t shadow_start =
         sparse_ro_offset < AGX_G17_RUNTIME_GRAPH_VA_START
            ? AGX_G17_RUNTIME_GRAPH_VA_START - sparse_ro_offset
            : 0;
      const uint64_t shadow_end =
         AGX_G17_RUNTIME_GRAPH_VA_END - sparse_ro_offset;

      add_intersection(main_intervals, &main_count, main_start, main_end,
                       shadow_start, shadow_end);
   }

   if (!reserve_intervals(main_heap, main_intervals, main_count))
      return false;

   struct agx_va_interval usc_interval[1];
   unsigned usc_count = 0;
   add_intersection(usc_interval, &usc_count, usc_start, usc_end,
                    AGX_G17_RUNTIME_GRAPH_VA_START,
                    AGX_G17_RUNTIME_GRAPH_VA_END);

   return reserve_intervals(usc_heap, usc_interval, usc_count);
}
