/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "d12_queue_timeline_core.h"

#include <limits.h>

_Static_assert(D12_QUEUE_SIGNAL_SLOT_COUNT == 64u,
               "signal-slot bitmap width changed");
_Static_assert(D12_QUEUE_SIGNAL_STREAM_STRIDE *
                     D12_QUEUE_SIGNAL_SLOT_COUNT <=
                  D12_QUEUE_SIGNAL_ARENA_SIZE,
               "signal stream slots exceed arena");
_Static_assert(D12_QUEUE_SIGNAL_DATA_OFFSET + 16u <=
                  D12_QUEUE_SIGNAL_STREAM_STRIDE,
               "signal arguments overlap next control stream");
_Static_assert(D12_QUEUE_SIGNAL_USC_STRIDE *
                     D12_QUEUE_SIGNAL_SLOT_COUNT <=
                  D12_QUEUE_SIGNAL_ARENA_SIZE,
               "signal USC slots exceed arena");

void
d12_queue_timeline_init(struct d12_queue_timeline_state *state)
{
   if (state)
      *state = (struct d12_queue_timeline_state){0};
}

bool
d12_queue_timeline_next_submit(struct d12_queue_timeline_state *state,
                               uint64_t *in_point, uint64_t *out_point)
{
   if (!state || !in_point || !out_point || state->submitted == UINT64_MAX)
      return false;

   *in_point = state->submitted;
   *out_point = ++state->submitted;
   return true;
}

uint64_t
d12_queue_timeline_marker(const struct d12_queue_timeline_state *state)
{
   return state ? state->submitted : 0;
}

bool
d12_queue_timeline_retire(struct d12_queue_timeline_state *state,
                          uint64_t point)
{
   if (!state || point < state->retired || point > state->submitted)
      return false;

   state->retired = point;
   return true;
}

uint64_t
d12_queue_timeline_inflight(const struct d12_queue_timeline_state *state)
{
   if (!state || state->retired > state->submitted)
      return 0;

   return state->submitted - state->retired;
}

bool
d12_queue_timeline_has_capacity(const struct d12_queue_timeline_state *state,
                                uint64_t max_inflight)
{
   return state && max_inflight > 0 &&
          d12_queue_timeline_inflight(state) < max_inflight;
}

bool
d12_queue_signal_slot_layout_program(
   uint32_t slot, uint32_t control_size, uint32_t program_size,
   uint32_t program_stride, uint32_t program_arena_size,
   uint32_t *control_offset, uint32_t *data_offset,
   uint32_t *program_offset)
{
   if (slot >= D12_QUEUE_SIGNAL_SLOT_COUNT ||
       control_size > D12_QUEUE_SIGNAL_DATA_OFFSET || !program_stride ||
       (program_stride & 63u) || program_size > program_stride ||
       !control_offset || !data_offset || !program_offset)
      return false;

   const uint32_t control = slot * D12_QUEUE_SIGNAL_STREAM_STRIDE;
   const uint32_t data = control + D12_QUEUE_SIGNAL_DATA_OFFSET;
   const uint64_t program = (uint64_t)slot * program_stride;
   if (control > D12_QUEUE_SIGNAL_ARENA_SIZE -
                    D12_QUEUE_SIGNAL_STREAM_STRIDE ||
       data > D12_QUEUE_SIGNAL_ARENA_SIZE - 16u ||
       program > program_arena_size ||
       program_size > program_arena_size - program)
      return false;

   *control_offset = control;
   *data_offset = data;
   *program_offset = program;
   return true;
}

bool
d12_queue_signal_slot_layout(uint32_t slot, uint32_t control_size,
                             uint32_t usc_size, uint32_t *control_offset,
                             uint32_t *data_offset, uint32_t *usc_offset)
{
   return d12_queue_signal_slot_layout_program(
      slot, control_size, usc_size, D12_QUEUE_SIGNAL_USC_STRIDE,
      D12_QUEUE_SIGNAL_ARENA_SIZE, control_offset, data_offset, usc_offset);
}

int
d12_queue_signal_slot_acquire(uint64_t *used_slots)
{
   if (!used_slots || *used_slots == UINT64_MAX)
      return -1;

   for (uint32_t slot = 0; slot < D12_QUEUE_SIGNAL_SLOT_COUNT; ++slot) {
      const uint64_t bit = UINT64_C(1) << slot;
      if (!(*used_slots & bit)) {
         *used_slots |= bit;
         return (int)slot;
      }
   }

   return -1;
}

bool
d12_queue_signal_slot_release(uint64_t *used_slots, uint32_t slot)
{
   if (!used_slots || slot >= D12_QUEUE_SIGNAL_SLOT_COUNT)
      return false;

   const uint64_t bit = UINT64_C(1) << slot;
   if (!(*used_slots & bit))
      return false;

   *used_slots &= ~bit;
   return true;
}
