/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#ifndef HK_AGX3_AS_LAYOUT_H
#define HK_AGX3_AS_LAYOUT_H
#include <stdbool.h>
#include <stdint.h>

/* Encoding scratch aliases the common builder's finished sort workspace.
 * All offsets are relative to the first byte after vk_ir_header. */
struct hk_agx3_as_layout {
   uint32_t nodes, jobs, envelope, state, node_status, envelope_status;
   /* An update's dispatches of inner jobs, one per depth (see
    * HK_AGX3_REFIT_LEVELS). */
   uint32_t refit_levels;
   uint32_t encode_bytes, node_bound, slot_bound;
   uint64_t as_bytes;
};

static inline bool
hk_agx3_as_layout_init(uint32_t leaves, struct hk_agx3_as_layout *out)
{
   if (!out)
      return false;
   if (!leaves) {
      *out = (struct hk_agx3_as_layout){.slot_bound = 3, .as_bytes = 192};
      return true;
   }
   const uint64_t nodes = (uint64_t)leaves + (leaves > 1 ? leaves : 2) - 1;
   /* A tree has jobs-1 child-root slots. Each leaf cluster contributes its
    * triangles plus ceil(triangles/4) QTB headers. With at most leaves leaf
    * jobs and leaves-1 inner jobs, end_slot <= 4*leaves+1 (including slots0..2).
    */
   const uint64_t slots = (uint64_t)leaves * 4 + 1;
   const uint64_t jobs = nodes * 16;
   const uint64_t envelope = jobs + nodes * 64;
   const uint64_t state = envelope + 48;
   /* The plan state, then the node encoding dispatch arguments. */
   const uint64_t status = state + 48;
   const uint64_t refit_levels = (status + nodes * 4 + 4 + 7) & ~UINT64_C(7);
   /* HK_AGX3_REFIT_LEVELS VkDispatchIndirectCommands. */
   const uint64_t end = refit_levels + 16 * 12;
   if (nodes > UINT32_MAX / 8 || slots > UINT32_MAX || end > UINT32_MAX)
      return false;
   *out = (struct hk_agx3_as_layout){
      .nodes = 0, .jobs = jobs, .envelope = envelope, .state = state,
      .node_status = status, .envelope_status = status + nodes * 4,
      .refit_levels = refit_levels,
      .encode_bytes = end, .node_bound = nodes, .slot_bound = slots,
      .as_bytes = slots * 64 + 184,
   };
   return true;
}

static inline void
hk_agx3_as_empty_header(uint32_t header[28])
{
   for (unsigned i = 0; i < 28; ++i)
      header[i] = i >= 4 && i < 10 ? 0x7fc00000u : 0;
}
#endif
