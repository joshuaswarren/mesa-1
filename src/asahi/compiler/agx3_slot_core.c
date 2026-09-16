/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_slot_core.h"

#include <string.h>

bool
agx3_slot_assign_with_live(const struct agx3_slot_instr *instrs, unsigned n,
                           unsigned nslots,
                           const struct agx3_slot_live_range *live,
                           unsigned live_count, struct agx3_slot_result *out)
{
   if (instrs == NULL || out == NULL || nslots < 1 || nslots > 8 ||
       (live_count != 0 && live == NULL))
      return false;

   /* reg_slot[r] = slot (1..nslots) of the async op whose result currently
    * lives in half-register r, or 0 if r was last written by a synchronous op
    * (or never). */
   uint8_t reg_slot[AGX3_SLOT_NUM_TOKENS];
   memset(reg_slot, 0, sizeof(reg_slot));

   for (unsigned i = 0; i < live_count; ++i) {
      const struct agx3_slot_live_range *range = &live[i];
      if (range->count == 0 || range->slot == 0 || range->slot > nslots ||
          range->first >= AGX3_SLOT_NUM_TOKENS ||
          range->count > AGX3_SLOT_NUM_TOKENS - range->first)
         return false;
      for (unsigned r = range->first; r < range->first + range->count; ++r) {
         if (reg_slot[r] != 0)
            return false;
         reg_slot[r] = range->slot;
      }
   }

   /* reader_slots[r] = the slots of the in-flight stores that latch
    * half-register r as payload. A store has no destination, so the result
    * bookkeeping above never covers its payload: the transform feedback copy
    * program reuses one payload quad for all sixteen captured varyings, and
    * the next varying's setup raced the outstanding store. */
   uint16_t reader_slots[AGX3_SLOT_NUM_TOKENS];
   memset(reader_slots, 0, sizeof(reader_slots));

   /* slot_waited[s] = the slot has been synchronised since it was last
    * allocated, so later consumers need not wait it again. */
   bool slot_waited[9] = {false};

   unsigned next = 0; /* round-robin cursor into 0..nslots-1 */

   for (unsigned i = 0; i < n; ++i) {
      const struct agx3_slot_instr *I = &instrs[i];
      if (I->nr_srcs > AGX3_SLOT_MAX_REGS || I->nr_dsts > AGX3_SLOT_MAX_REGS)
         return false;
      if (I->preferred_slot > nslots || (!I->is_async && I->preferred_slot))
         return false;
      const uint16_t valid_slots = (uint16_t)((1u << nslots) - 1u);
      if ((I->transient_slots & ~valid_slots) ||
          (I->preferred_slot &&
           (I->transient_slots & (1u << (I->preferred_slot - 1)))))
         return false;

      uint16_t wait_mask = 0;
      for (unsigned s = 0; s < I->nr_srcs; ++s) {
         const uint16_t r = I->src[s];
         if (r >= AGX3_SLOT_NUM_TOKENS)
            return false;
         const uint8_t slot = reg_slot[r];
         if (slot != 0 && !slot_waited[slot]) {
            wait_mask |= (uint16_t)(1u << (slot - 1));
            slot_waited[slot] = true;
         }
      }

      /*
       * Write-after-write. An async result that is still in flight for this
       * register lands LATER than this instruction, so a synchronous write
       * here is silently overwritten by the load when it retires -- and any
       * consumer that waits the slot then reads the loaded word instead of
       * the computed one. Mesa's own AGX2 pass checks this case explicitly
       * (agx_insert_waits.c, "Check for write-after-write"); omitting it here
       * produced no diagnostic, just wrong values on hardware.
       *
       * This runs before the bookkeeping below, so reg_slot[] still names the
       * previous writer rather than this instruction.
       */
      for (unsigned d = 0; d < I->nr_dsts; ++d) {
         const uint16_t r = I->dst[d];
         if (r >= AGX3_SLOT_NUM_TOKENS)
            return false;
         const uint8_t slot = reg_slot[r];
         if (slot != 0 && !slot_waited[slot]) {
            wait_mask |= (uint16_t)(1u << (slot - 1));
            slot_waited[slot] = true;
         }

         /* Write-after-read: retire every async op still reading this
          * register before redefining it. */
         uint16_t readers = reader_slots[r];
         while (readers) {
            const unsigned bit = (unsigned)(readers & (~readers + 1u));
            const uint8_t reader = (uint8_t)(__builtin_ctz(bit) + 1);
            readers &= (uint16_t)~bit;
            if (!slot_waited[reader]) {
               wait_mask |= (uint16_t)bit;
               slot_waited[reader] = true;
            }
         }
      }

      /* A compound instruction may use a scoreboard slot internally, for
       * example PACK_SMPIR/COPY producers followed by a TEX which waits for
       * them. Retire a prior generation before reusing the slot, then forget
       * its aliases: the prewait has made those GPR values synchronous and
       * the internal generation is consumed before this logical instruction
       * returns. */
      for (unsigned slot = 1; slot <= nslots; ++slot) {
         const uint16_t bit = (uint16_t)(1u << (slot - 1));
         if (!(I->transient_slots & bit))
            continue;

         bool live = false;
         for (unsigned r = 0; r < AGX3_SLOT_NUM_TOKENS; ++r) {
            if (reg_slot[r] == slot) {
               live = true;
               reg_slot[r] = 0;
            }
         }
         if (live && !slot_waited[slot])
            wait_mask |= bit;
         slot_waited[slot] = true;
      }

      out[i].wait_mask = wait_mask;

      /* Every slot this instruction waited is now retired, so the registers
       * its async ops were reading are free again. */
      if (wait_mask) {
         for (unsigned r = 0; r < AGX3_SLOT_NUM_TOKENS; ++r)
            reader_slots[r] &= (uint16_t)~wait_mask;
      }

      if (I->is_async) {
         const uint8_t slot =
            I->preferred_slot ? I->preferred_slot : (uint8_t)(next + 1);
         if (!I->preferred_slot)
            next = (next + 1) % nslots;
         out[i].slot = slot;

         /* Reusing an already-synchronized slot starts a new dependency
          * generation. Retire the old generation's register aliases before
          * publishing the new result, otherwise an unrelated later write can
          * consume the new generation's wait and hide it from its consumer. */
         if (slot_waited[slot]) {
            for (unsigned r = 0; r < AGX3_SLOT_NUM_TOKENS; ++r) {
               if (reg_slot[r] == slot)
                  reg_slot[r] = 0;
            }
         }
         slot_waited[slot] = false; /* freshly outstanding */
         for (unsigned r = 0; r < AGX3_SLOT_NUM_TOKENS; ++r)
            reader_slots[r] &= (uint16_t)~(1u << (slot - 1));
         for (unsigned d = 0; d < I->nr_dsts; ++d) {
            if (I->dst[d] >= AGX3_SLOT_NUM_TOKENS)
               return false;
            reg_slot[I->dst[d]] = slot;
         }
         if (I->latches_srcs) {
            for (unsigned s = 0; s < I->nr_srcs; ++s)
               reader_slots[I->src[s]] |= (uint16_t)(1u << (slot - 1));
         }
      } else {
         out[i].slot = 0;
         for (unsigned d = 0; d < I->nr_dsts; ++d) {
            if (I->dst[d] >= AGX3_SLOT_NUM_TOKENS)
               return false;
            reg_slot[I->dst[d]] = 0; /* synchronous result, no wait needed */
         }
      }
   }

   return true;
}

bool
agx3_slot_assign(const struct agx3_slot_instr *instrs, unsigned n,
                 unsigned nslots, struct agx3_slot_result *out)
{
   return agx3_slot_assign_with_live(instrs, n, nslots, NULL, 0, out);
}
