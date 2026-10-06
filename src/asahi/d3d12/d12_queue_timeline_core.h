/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Platform-neutral command-queue timeline accounting.  The renderer uses
 * this state machine to submit work ahead while retaining its storage until
 * the corresponding hardware timeline point completes.
 */

#ifndef D12_QUEUE_TIMELINE_CORE_H
#define D12_QUEUE_TIMELINE_CORE_H

#include <stdbool.h>
#include <stdint.h>

struct d12_queue_timeline_state {
   uint64_t submitted;
   uint64_t retired;
};

#define D12_QUEUE_MAX_INFLIGHT_SUBMITS 64u
/* A submission whose renders recorded at least this many draws makes the
 * next one wait for it instead of overlapping (d12_queue_process_submit). */
#define D12_QUEUE_OVERLAP_MAX_DRAWS 512u
#define D12_QUEUE_SIGNAL_SLOT_COUNT 64u
#define D12_QUEUE_SIGNAL_STREAM_STRIDE 256u
#define D12_QUEUE_SIGNAL_DATA_OFFSET 192u
#define D12_QUEUE_SIGNAL_USC_STRIDE 192u
#define D12_QUEUE_SIGNAL_ARENA_SIZE 16384u

void d12_queue_timeline_init(struct d12_queue_timeline_state *state);

bool d12_queue_timeline_next_submit(struct d12_queue_timeline_state *state,
                                    uint64_t *in_point,
                                    uint64_t *out_point);

uint64_t
d12_queue_timeline_marker(const struct d12_queue_timeline_state *state);

bool d12_queue_timeline_retire(struct d12_queue_timeline_state *state,
                               uint64_t point);

uint64_t
d12_queue_timeline_inflight(const struct d12_queue_timeline_state *state);

bool d12_queue_timeline_has_capacity(
   const struct d12_queue_timeline_state *state, uint64_t max_inflight);

/* The stream/argument slots stay fixed while each generation owns a bounded,
 * 64-byte-aligned program stride and a separate program arena. */
bool d12_queue_signal_slot_layout_program(
   uint32_t slot, uint32_t control_size, uint32_t program_size,
   uint32_t program_stride, uint32_t program_arena_size,
   uint32_t *control_offset, uint32_t *data_offset,
   uint32_t *program_offset);

bool d12_queue_signal_slot_layout(uint32_t slot, uint32_t control_size,
                                  uint32_t usc_size,
                                  uint32_t *control_offset,
                                  uint32_t *data_offset,
                                  uint32_t *usc_offset);

int d12_queue_signal_slot_acquire(uint64_t *used_slots);

bool d12_queue_signal_slot_release(uint64_t *used_slots, uint32_t slot);

#endif
