/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <stddef.h>
#include <stdint.h>
#include "agx3_slot_core.h"

#ifdef __cplusplus
extern "C" {
#endif

/* An immutable value in the sampler-private register file. A publication may
 * consist of several native operations sharing one completion event. Its GPR
 * inputs belong to that publication, not to subsequent samplers. Values and
 * uses describe one straight-line region; a CFG join needs a separate merge.
 * UINT16_MAX means the value has no fixed ABI address.
 */
struct agx3_resource_value {
   uint32_t producer;
   uint16_t halves;
   uint16_t alignment;
   uint16_t fixed_base;
};

struct agx3_resource_use {
   uint32_t instruction;
   uint32_t value;
   uint16_t first;
   uint16_t count;
   bool through_completion;
};

/* Allocate logical lifetimes, then lower resource accesses into the same
 * dependency stream as GPR accesses. Storage remains reserved through the last
 * issued use; the scoreboard separately protects every asynchronous reader
 * after that point. Reusing storage can require a WAR completion wait, even
 * though the logical lifetimes no longer overlap.
 *
 * base contains only the existing instruction's non-resource dependencies.
 * Each publication must be asynchronous. Every read names a defined value and
 * may name a subrange. All outputs are unchanged on failure, including pressure
 * overflow. A caller must rematerialize/spill or keep its original lowering in
 * that case, never discard an access. This does not reorder any instruction.
 */
bool agx3_resource_lower(const struct agx3_resource_value *values,
                         size_t value_count,
                         const struct agx3_resource_use *uses, size_t use_count,
                         const struct agx3_slot_instr *base,
                         unsigned instruction_count, unsigned capacity,
                         uint16_t *placements, struct agx3_slot_instr *out);

#ifdef __cplusplus
}
#endif
