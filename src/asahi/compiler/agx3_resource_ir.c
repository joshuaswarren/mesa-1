/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "agx3_resource_ir.h"

#include <stdlib.h>
#include <string.h>

struct interval {
   uint32_t value, first, last, completion_last;
   uint16_t base;
};

/* The first existing dataflow dependency that must complete this event.
 * This is an allocation preference, not a substitute for scoreboard hazards:
 * slot pressure may complete it sooner, and reusing its range may itself need
 * a wait. Avoid such a wait when another otherwise-dead range is available.
 */
static unsigned
completion_before(const struct agx3_slot_instr *base, unsigned n, unsigned event)
{
   for (unsigned i = event + 1; i < n; ++i) {
      if (base[i].drain)
         return i;
      for (unsigned d = 0; d < base[event].nr_dsts; ++d) {
         const uint16_t token = base[event].dst[d];
         for (unsigned s = 0; s < base[i].nr_srcs; ++s)
            if (base[i].src[s] == token)
               return i;
         for (unsigned w = 0; w < base[i].nr_dsts; ++w)
            if (base[i].dst[w] == token)
               return i;
      }
   }
   return n;
}

static int
compare_intervals(const void *ap, const void *bp)
{
   const struct interval *a = ap, *b = bp;
   if (a->first != b->first)
      return a->first < b->first ? -1 : 1;
   return (a->value > b->value) - (a->value < b->value);
}

static bool
intersects(unsigned a, unsigned an, unsigned b, unsigned bn)
{
   return a < b + bn && b < a + an;
}

static bool
lifetimes_overlap(const struct interval *a, const struct interval *b)
{
   /* Inputs must coexist with outputs at one instruction. Completion of
    * asynchronous inputs is enforced later by the scoreboard.
    */
   return a->first <= b->last && b->first <= a->last;
}

static bool
append_token(struct agx3_slot_instr *I, uint16_t token, bool write, bool late)
{
   uint8_t *count = write ? &I->nr_dsts : &I->nr_srcs;
   uint16_t *tokens = write ? I->dst : I->src;
   unsigned i;
   for (i = 0; i < *count && tokens[i] != token; ++i)
      ;
   if (i == *count) {
      if (*count == AGX3_SLOT_MAX_REGS)
         return false;
      tokens[(*count)++] = token;
   }
   if (late)
      I->late_src_mask |= UINT32_C(1) << i;
   return true;
}

bool
agx3_resource_lower(const struct agx3_resource_value *values, size_t value_count,
                    const struct agx3_resource_use *uses, size_t use_count,
                    const struct agx3_slot_instr *base,
                    unsigned instruction_count, unsigned capacity,
                    uint16_t *placements, struct agx3_slot_instr *out)
{
   size_t instruction_bytes;
   if (!capacity || capacity > AGX3_SLOT_RESOURCE_HALVES ||
       (value_count && (!values || !placements)) || (use_count && !uses) ||
       (instruction_count && (!base || !out)) || value_count > UINT32_MAX ||
       value_count > SIZE_MAX / sizeof(struct interval) ||
       __builtin_mul_overflow((size_t)instruction_count,
                              sizeof(struct agx3_slot_instr),
                              &instruction_bytes))
      return false;

   struct interval *live = calloc(value_count ? value_count : 1, sizeof(*live));
   struct interval *order = calloc(value_count ? value_count : 1, sizeof(*order));
   struct agx3_slot_instr *lowered =
      calloc(instruction_count ? instruction_count : 1, sizeof(*lowered));
   bool ok = false;
   if (!live || !order || !lowered)
      goto done;

   for (unsigned i = 0; i < instruction_count; ++i) {
      if (base[i].nr_srcs > AGX3_SLOT_MAX_REGS ||
          base[i].nr_dsts > AGX3_SLOT_MAX_REGS ||
          (base[i].nr_srcs < 32 &&
           (base[i].late_src_mask >> base[i].nr_srcs)))
         goto done;
      /* Resource tokens belong exclusively to the lowering below. */
      for (unsigned s = 0; s < base[i].nr_srcs; ++s)
         if (base[i].src[s] >= AGX3_SLOT_NUM_TOKENS ||
             (base[i].src[s] >= AGX3_SLOT_RESOURCE_TOKEN_BASE &&
              base[i].src[s] < AGX3_SLOT_RESOURCE_TOKEN_BASE +
                                 AGX3_SLOT_RESOURCE_HALVES))
            goto done;
      for (unsigned d = 0; d < base[i].nr_dsts; ++d)
         if (base[i].dst[d] >= AGX3_SLOT_NUM_TOKENS ||
             (base[i].dst[d] >= AGX3_SLOT_RESOURCE_TOKEN_BASE &&
              base[i].dst[d] < AGX3_SLOT_RESOURCE_TOKEN_BASE +
                                 AGX3_SLOT_RESOURCE_HALVES))
            goto done;
      lowered[i] = base[i];
   }

   for (size_t v = 0; v < value_count; ++v) {
      const struct agx3_resource_value *V = &values[v];
      if (V->producer >= instruction_count || !base[V->producer].is_async ||
          !V->halves || V->halves > capacity || !V->alignment ||
          (V->alignment & (V->alignment - 1)) ||
          (V->fixed_base != UINT16_MAX &&
           (V->fixed_base % V->alignment ||
            V->fixed_base > capacity - V->halves)))
         goto done;
      live[v] = (struct interval){
         .value = v, .first = V->producer, .last = V->producer,
         .completion_last = V->producer,
      };
   }

   for (size_t u = 0; u < use_count; ++u) {
      const struct agx3_resource_use *U = &uses[u];
      if (U->value >= value_count || U->instruction >= instruction_count)
         goto done;
      const struct agx3_resource_value *V = &values[U->value];
      if (U->instruction <= V->producer || !U->count ||
          U->first >= V->halves || U->count > V->halves - U->first ||
          (U->through_completion && !base[U->instruction].is_async))
         goto done;
      if (U->instruction > live[U->value].last)
         live[U->value].last = U->instruction;
      if (U->through_completion) {
         const unsigned end = completion_before(base, instruction_count,
                                                 U->instruction) - 1;
         if (end > live[U->value].completion_last)
            live[U->value].completion_last = end;
      }
   }

   memcpy(order, live, value_count * sizeof(*order));
   qsort(order, value_count, sizeof(*order), compare_intervals);
   for (size_t i = 0; i < value_count; ++i) {
      struct interval *L = &live[order[i].value];
      const struct agx3_resource_value *V = &values[L->value];
      bool placed = false;
      size_t best_cost = SIZE_MAX;
      unsigned first = V->fixed_base == UINT16_MAX ? 0 : V->fixed_base;
      for (unsigned b = first; b <= capacity - V->halves;
           b += V->alignment) {
         bool conflict = false;
         size_t cost = 0;
         for (size_t j = 0; j < value_count && !conflict; ++j) {
            if (j == i)
               continue;
            const struct interval *P = &live[order[j].value];
            const struct agx3_resource_value *PV = &values[P->value];
            /* Account for future fixed ABI values as well as values already
             * placed. A movable interval must not occupy their reservation.
             */
            if (j > i && PV->fixed_base == UINT16_MAX)
               continue;
            unsigned pb = j < i ? P->base : PV->fixed_base;
            conflict = lifetimes_overlap(L, P) &&
                       intersects(b, V->halves, pb, PV->halves);
            if (!conflict && j < i && L->first <= P->completion_last &&
                intersects(b, V->halves, pb, PV->halves))
               ++cost;
         }
         if (!conflict && cost < best_cost) {
            L->base = b;
            placed = true;
            best_cost = cost;
            if (!cost)
               break;
         }
         if (V->fixed_base != UINT16_MAX)
            break;
      }
      if (!placed)
         goto done;
      for (unsigned h = 0; h < V->halves; ++h)
         if (!append_token(&lowered[V->producer],
               AGX3_SLOT_RESOURCE_TOKEN_BASE + L->base + h, true, false))
            goto done;
   }

   for (size_t u = 0; u < use_count; ++u) {
      const struct agx3_resource_use *U = &uses[u];
      const unsigned first_source = lowered[U->instruction].nr_srcs;
      for (unsigned h = 0; h < U->count; ++h)
         if (!append_token(&lowered[U->instruction],
               AGX3_SLOT_RESOURCE_TOKEN_BASE + live[U->value].base + U->first + h,
               false, U->through_completion))
            goto done;
      /* A resource consumer may recycle its publication's completed event
       * slot. Slot assignment checks the real producer generation; a value
       * already made synchronous by another consumer contributes no stale
       * slot. This preserves separate events without spending two unrelated
       * scoreboard slots on a dependency chain.
       */
      if (U->through_completion && !lowered[U->instruction].coalesce_src &&
          first_source < lowered[U->instruction].nr_srcs)
         lowered[U->instruction].coalesce_src = first_source + 1;
   }

   for (size_t v = 0; v < value_count; ++v)
      placements[v] = live[v].base;
   if (instruction_count)
      memcpy(out, lowered, instruction_bytes);
   ok = true;
done:
   free(live);
   free(order);
   free(lowered);
   return ok;
}
