/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "../d12_queue_timeline_core.h"

#include <assert.h>
#include <limits.h>
#include <stdio.h>

static void
test_submission_pipeline(void)
{
   struct d12_queue_timeline_state state;
   uint64_t in = UINT64_MAX;
   uint64_t out = UINT64_MAX;

   d12_queue_timeline_init(&state);
   assert(state.submitted == 0);
   assert(state.retired == 0);
   assert(d12_queue_timeline_inflight(&state) == 0);

   assert(d12_queue_timeline_next_submit(&state, &in, &out));
   assert(in == 0 && out == 1);
   assert(d12_queue_timeline_next_submit(&state, &in, &out));
   assert(in == 1 && out == 2);
   assert(d12_queue_timeline_next_submit(&state, &in, &out));
   assert(in == 2 && out == 3);
   assert(d12_queue_timeline_inflight(&state) == 3);
   assert(d12_queue_timeline_has_capacity(&state, 4));
   assert(!d12_queue_timeline_has_capacity(&state, 3));

   assert(d12_queue_timeline_retire(&state, 1));
   assert(d12_queue_timeline_inflight(&state) == 2);
   assert(d12_queue_timeline_has_capacity(&state, 3));
   assert(d12_queue_timeline_retire(&state, 3));
   assert(d12_queue_timeline_inflight(&state) == 0);
}

static void
test_markers_do_not_serialize_submission(void)
{
   struct d12_queue_timeline_state state;
   uint64_t in;
   uint64_t out;

   d12_queue_timeline_init(&state);
   assert(d12_queue_timeline_marker(&state) == 0);
   assert(d12_queue_timeline_next_submit(&state, &in, &out));
   assert(d12_queue_timeline_marker(&state) == 1);

   /* Multiple fence signals snapshot the same completion point. */
   assert(d12_queue_timeline_marker(&state) == 1);
   assert(d12_queue_timeline_marker(&state) == 1);

   /* Work after Signal may be submitted before the marker retires. */
   assert(d12_queue_timeline_next_submit(&state, &in, &out));
   assert(in == 1 && out == 2);
   assert(state.retired == 0);

   assert(d12_queue_timeline_retire(&state, 1));
   assert(d12_queue_timeline_retire(&state, 1));
   assert(d12_queue_timeline_retire(&state, 2));
}

static void
test_invalid_transitions(void)
{
   struct d12_queue_timeline_state state = {
      .submitted = UINT64_MAX,
      .retired = UINT64_MAX - 1,
   };
   uint64_t in = 7;
   uint64_t out = 9;

   assert(!d12_queue_timeline_next_submit(&state, &in, &out));
   assert(in == 7 && out == 9);
   assert(!d12_queue_timeline_retire(&state, UINT64_MAX - 2));
   assert(d12_queue_timeline_retire(&state, UINT64_MAX));

   d12_queue_timeline_init(&state);
   assert(!d12_queue_timeline_retire(&state, 1));
   assert(!d12_queue_timeline_next_submit(NULL, &in, &out));
   assert(!d12_queue_timeline_next_submit(&state, NULL, &out));
   assert(!d12_queue_timeline_next_submit(&state, &in, NULL));
   assert(!d12_queue_timeline_retire(NULL, 0));
   assert(d12_queue_timeline_marker(NULL) == 0);
   assert(d12_queue_timeline_inflight(NULL) == 0);
   assert(!d12_queue_timeline_has_capacity(NULL, 1));
   assert(!d12_queue_timeline_has_capacity(&state, 0));
}

static void
test_signal_arena_layout_and_slots(void)
{
   uint64_t slots = 0;
   uint32_t control;
   uint32_t data;
   uint32_t usc;

   for (uint32_t slot = 0; slot < D12_QUEUE_SIGNAL_SLOT_COUNT; ++slot) {
      assert(d12_queue_signal_slot_layout(slot, 128, 146, &control, &data,
                                          &usc));
      assert(control == slot * D12_QUEUE_SIGNAL_STREAM_STRIDE);
      assert(data == control + D12_QUEUE_SIGNAL_DATA_OFFSET);
      assert(usc == slot * D12_QUEUE_SIGNAL_USC_STRIDE);
      assert(data + 16 <= D12_QUEUE_SIGNAL_ARENA_SIZE);
      assert(usc + 146 <= D12_QUEUE_SIGNAL_ARENA_SIZE);
      assert(d12_queue_signal_slot_acquire(&slots) == (int)slot);
   }
   assert(slots == UINT64_MAX);
   assert(d12_queue_signal_slot_acquire(&slots) == -1);

   assert(d12_queue_signal_slot_release(&slots, 17));
   assert(!d12_queue_signal_slot_release(&slots, 17));
   assert(d12_queue_signal_slot_acquire(&slots) == 17);
   assert(!d12_queue_signal_slot_release(&slots,
                                         D12_QUEUE_SIGNAL_SLOT_COUNT));
   assert(!d12_queue_signal_slot_release(NULL, 0));
   assert(d12_queue_signal_slot_acquire(NULL) == -1);

   assert(!d12_queue_signal_slot_layout(D12_QUEUE_SIGNAL_SLOT_COUNT, 1, 1,
                                        &control, &data, &usc));
   assert(!d12_queue_signal_slot_layout(0, D12_QUEUE_SIGNAL_DATA_OFFSET + 1,
                                        1, &control, &data, &usc));
   assert(!d12_queue_signal_slot_layout(0, 1,
                                        D12_QUEUE_SIGNAL_USC_STRIDE + 1,
                                        &control, &data, &usc));
   assert(!d12_queue_signal_slot_layout(0, 1, 1, NULL, &data, &usc));
}

static void
test_native_signal_program_slots(void)
{
   const uint32_t stride = 6144;
   const uint32_t arena = stride * D12_QUEUE_SIGNAL_SLOT_COUNT;
   uint32_t control, data, program;
   for (uint32_t slot = 0; slot < D12_QUEUE_SIGNAL_SLOT_COUNT; slot++) {
      assert(d12_queue_signal_slot_layout_program(
         slot, 44, stride, stride, arena, &control, &data, &program));
      assert(control == slot * D12_QUEUE_SIGNAL_STREAM_STRIDE);
      assert(data == control + D12_QUEUE_SIGNAL_DATA_OFFSET);
      assert(program == slot * stride && !(program & 63));
      assert((uint64_t)program + stride <= arena);
   }
   control = data = program = 0xfeed;
   assert(!d12_queue_signal_slot_layout_program(
      63, 44, stride, stride, arena - 1, &control, &data, &program));
   assert(control == 0xfeed && data == 0xfeed && program == 0xfeed);
   assert(!d12_queue_signal_slot_layout_program(
      0, 44, stride + 1, stride, arena, &control, &data, &program));
   assert(!d12_queue_signal_slot_layout_program(
      0, 44, 16, stride + 1, arena, &control, &data, &program));
   assert(!d12_queue_signal_slot_layout_program(
      63, 44, 16, UINT32_MAX & ~63u, UINT32_MAX,
      &control, &data, &program));
   assert(!d12_queue_signal_slot_layout_program(
      64, 44, stride, stride, arena, &control, &data, &program));
}

int
main(void)
{
   test_submission_pipeline();
   test_markers_do_not_serialize_submission();
   test_invalid_transitions();
   test_signal_arena_layout_and_slots();
   test_native_signal_program_slots();
   puts("d12 queue timeline core tests: pass");
   return 0;
}
