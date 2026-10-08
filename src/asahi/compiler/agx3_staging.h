/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>

/* Sampler-private half-registers. COPY's destination and SMP's coordinate
 * operands name this file independently of the GPR file. A range remains live
 * until every asynchronous consumer of that range retires. Multiple consumers
 * may share a completion slot, or overlapping ranges may have different slots.
 */
#define AGX3_STAGING_HALFS 256u

struct agx3_staging_banks {
   uint16_t readers[AGX3_STAGING_HALFS];
   uint16_t end;
   uint8_t next_write_slot;
};

static inline void
agx3_staging_release(struct agx3_staging_banks *banks, uint16_t outstanding)
{
   for (unsigned i = 0; i < banks->end; ++i)
      banks->readers[i] &= outstanding;
}

static inline uint16_t
agx3_staging_conflicts(const struct agx3_staging_banks *banks, unsigned start,
                      unsigned span)
{
   uint16_t conflicts = 0;
   if (start >= banks->end)
      return 0;

   unsigned count = banks->end - start;
   if (span < count)
      count = span;
   for (unsigned i = start; i < start + count; ++i)
      conflicts |= banks->readers[i];
   return conflicts;
}

static inline void
agx3_staging_forget(struct agx3_staging_banks *banks, uint16_t slots)
{
   agx3_staging_release(banks, (uint16_t)~slots);
}

static inline bool
agx3_staging_hold(struct agx3_staging_banks *banks, uint16_t base,
                 uint16_t span, uint8_t slot)
{
   if (!span || span > AGX3_STAGING_HALFS ||
       base > AGX3_STAGING_HALFS - span || !slot || slot > 8)
      return false;

   if (base + span > banks->end)
      banks->end = base + span;
   for (unsigned i = base; i < base + span; ++i)
      banks->readers[i] |= 1u << (slot - 1);
   return true;
}

/* Plan an aligned allocation without changing any lifetime. Prefer a free
 * range; otherwise return the range requiring the fewest consumer slots to
 * retire. The caller must emit those waits before overwriting the range and
 * publish the new consumer with hold(). limit expresses the instruction's
 * addressable extent, not an arbitrary number of operations in flight.
 */
static inline bool
agx3_staging_plan(const struct agx3_staging_banks *banks, unsigned span,
                 unsigned alignment, unsigned limit, uint16_t *base,
                 uint16_t *waits)
{
   if (!base || !waits || !span || !alignment ||
       (alignment & (alignment - 1)) || limit > AGX3_STAGING_HALFS || span > limit)
      return false;

   unsigned best_count = 17;
   for (unsigned i = 0; i <= limit - span; i += alignment) {
      uint16_t conflict = agx3_staging_conflicts(banks, i, span);
      unsigned count = __builtin_popcount(conflict);
      if (count < best_count) {
         *base = i;
         *waits = conflict;
         best_count = count;
         if (!count)
            return true;
      }
   }
   return best_count != 17;
}
