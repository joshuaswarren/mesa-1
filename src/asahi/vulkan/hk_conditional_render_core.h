/* Copyright 2026 Aurora Silicon; SPDX-License-Identifier: MIT */
#pragma once

#include <stdbool.h>
#include <stdint.h>

#define HK_CONDITIONAL_RENDER_MAX_WORDS 5

struct hk_conditional_render_state {
   uint64_t address;
   bool active;
   bool inverted;

   /* Secondary command buffers with conditionalRenderingEnable replay using
    * their caller's state. This bit describes the secondary, not a predicate
    * which was begun in that command buffer. */
   bool inherited;
};

static inline bool
hk_conditional_render_is_active(
   const struct hk_conditional_render_state *state, bool in_meta,
   bool conditional_meta)
{
   return state && state->active && (!in_meta || conditional_meta);
}

/* Build the exact Vulkan indirect record corresponding to a direct draw.
 * Indexed and non-indexed records intentionally share this helper so the GPU
 * predicate always sees the API-defined 5- or 4-word layout. */
static inline uint32_t
hk_conditional_render_direct_draw_record(
   uint32_t out[HK_CONDITIONAL_RENDER_MAX_WORDS], bool indexed,
   uint32_t count, uint32_t instance_count, uint32_t first,
   int32_t vertex_offset, uint32_t first_instance)
{
   if (!out)
      return 0;

   out[0] = count;
   out[1] = instance_count;
   out[2] = first;
   if (indexed) {
      out[3] = (uint32_t)vertex_offset;
      out[4] = first_instance;
      return 5;
   }

   out[3] = first_instance;
   return 4;
}

/* Honeykrisp records secondary command buffers into a common replay queue.
 * Select its initial predicate state explicitly at replay: only a secondary
 * which opted into inheritance observes its primary's active predicate. */
static inline struct hk_conditional_render_state
hk_conditional_render_replay_state(
   const struct hk_conditional_render_state *primary,
   bool secondary_inherits)
{
   if (!primary || !secondary_inherits)
      return (struct hk_conditional_render_state){0};

   struct hk_conditional_render_state replay = *primary;
   replay.inherited = true;
   return replay;
}
