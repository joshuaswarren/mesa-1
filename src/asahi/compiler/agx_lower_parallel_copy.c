/*
 * Copyright 2022 Alyssa Rosenzweig
 * Copyright 2021 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

#include "agx_builder.h"
#include "agx_compiler.h"

UNUSED static void
print_copy(const struct agx_copy *cp)
{
   printf("%sr%u = ", cp->dest_mem ? "m" : "", cp->dest);
   agx_print_index(cp->src, false, stdout);
   printf("\n");
}

UNUSED static void
print_copies(const struct agx_copy *copies, unsigned nr)
{
   printf("[\n");

   for (unsigned i = 0; i < nr; ++i) {
      printf("  ");
      print_copy(&copies[i]);
   }

   printf("]\n");
}

/*
 * Emits code for
 *
 *    for (int i = 0; i < n; ++i)
 *       registers[dests[i]] = registers[srcs[i]];
 *
 * ...with all copies happening in parallel.
 *
 * That is, emit machine instructions equivalent to a parallel copy. This is
 * used to lower not only parallel copies but also collects and splits, which
 * also have parallel copy semantics.
 *
 * We only handles register-register copies, not general agx_index sources. This
 * suffices for its internal use for register allocation.
 */

static void
do_copy(agx_builder *b, const struct agx_copy *copy)
{
   agx_index dst = copy->dest_mem
                      ? agx_memory_register(copy->dest, copy->src.size)
                      : agx_register(copy->dest, copy->src.size);

   if (copy->dest_mem && copy->src.memory) {
      /* Memory-memory copies need to be lowered to memory-register and
       * register-memory, using a reserved scratch register. Keep AGX3's r1
       * available for a full-width divergent-shuffle index when required.
       */
      const bool agx3_shuffle =
         b->shader->any_quad_divergent_shuffle &&
         b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2;
      agx_index scratch_reg =
         agx_register(agx3_shuffle ? 4 : 2, copy->src.size);
      agx_mov_to(b, scratch_reg, copy->src);
      agx_mov_to(b, dst, scratch_reg);
   } else if (copy->src.type == AGX_INDEX_IMMEDIATE) {
      agx_mov_imm_to(b, dst, copy->src.value);
   } else {
      agx_mov_to(b, dst, copy->src);
   }
}

static void
do_swap(agx_builder *b, const struct agx_copy *copy)
{
   assert(copy->src.type == AGX_INDEX_REGISTER && "only GPRs are swapped");

   if (copy->dest == copy->src.value)
      return;

   agx_index x = copy->dest_mem
                    ? agx_memory_register(copy->dest, copy->src.size)
                    : agx_register(copy->dest, copy->src.size);
   agx_index y = copy->src;
   assert(x.memory == y.memory);

   /* A copy's abs/neg is a float modifier composed into the move. AGX3's XOR
    * swap has no field for one, so swap the registers themselves and apply
    * the modifier to the moved value afterwards.
    */
   const bool agx3_modified =
      b->shader->key &&
      b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2 &&
      (y.abs || y.neg);
   if (agx3_modified)
      y.abs = y.neg = false;

   /* Memory-memory swaps lowered here, GPR swaps lowered later */
   if (x.memory) {
      agx_index temp1 = agx_register(4, copy->src.size);
      agx_index temp2 = agx_register(6, copy->src.size);

      agx_mov_to(b, temp1, x);
      agx_mov_to(b, temp2, y);
      agx_mov_to(b, y, temp1);
      agx_mov_to(b, x, temp2);
   } else {
      agx_swap(b, x, y);
   }

   if (agx3_modified) {
      agx_index modified = x;
      modified.abs = copy->src.abs;
      modified.neg = copy->src.neg;
      agx_mov_to(b, x, modified);
   }
}

struct copy_ctx {
   /* Number of copies being processed */
   unsigned entry_count;

   /* For each physreg, the number of pending copy entries that use it as a
    * source. Once this drops to zero, then the physreg is unblocked and can
    * be moved to.
    */
   unsigned physreg_use_count[AGX_NUM_MODELED_REGS];

   /* For each physreg, the pending copy_entry that uses it as a dest. */
   struct agx_copy *physreg_dest[AGX_NUM_MODELED_REGS];

   struct agx_copy entries[AGX_NUM_MODELED_REGS];
};

static bool
entry_blocked(struct agx_copy *entry, struct copy_ctx *ctx)
{
   for (unsigned i = 0; i < agx_size_align_16(entry->src.size); i++) {
      if (ctx->physreg_use_count[entry->dest + i] != 0)
         return true;
   }

   return false;
}

static bool
is_real(struct agx_copy *entry)
{
   return entry->src.type == AGX_INDEX_REGISTER &&
          entry->dest_mem == entry->src.memory;
}

/* TODO: Generalize to other bit sizes */
static void
split_32bit_copy(struct copy_ctx *ctx, struct agx_copy *entry)
{
   assert(!entry->done);
   assert(is_real(entry));
   assert(agx_size_align_16(entry->src.size) == 2);
   struct agx_copy *new_entry = &ctx->entries[ctx->entry_count++];

   new_entry->dest = entry->dest + 1;
   new_entry->dest_mem = entry->dest_mem;
   new_entry->src = entry->src;
   new_entry->src.value += 1;
   new_entry->done = false;
   entry->src.size = AGX_SIZE_16;
   new_entry->src.size = AGX_SIZE_16;
   ctx->physreg_dest[entry->dest + 1] = new_entry;
}

static void
agx_emit_parallel_copies_for_class(agx_builder *b, struct agx_copy *copies,
                                   unsigned num_copies, bool cls,
                                   agx_index temp)
{
   /* First, lower away 64-bit copies to smaller chunks, since we don't have
    * 64-bit ALU so we always want to split.
    */
   struct agx_copy *copies2 = calloc(sizeof(copies[0]), num_copies * 2);
   unsigned num_copies2 = 0;

   for (unsigned i = 0; i < num_copies; ++i) {
      struct agx_copy copy = copies[i];

      /* Filter by class */
      if (copy.dest_mem != cls)
         continue;

      assert(copy.dest < AGX_NUM_MODELED_REGS);

      if (copy.src.size == AGX_SIZE_64) {
         copy.src.size = AGX_SIZE_32;
         copies2[num_copies2++] = copy;

         if (copy.src.type == AGX_INDEX_IMMEDIATE) {
            static_assert(sizeof(copy.src.value) * 8 == 32, "known size");
            copy.src.value = 0;
         } else {
            assert(copy.src.type == AGX_INDEX_REGISTER ||
                   copy.src.type == AGX_INDEX_UNIFORM);

            copy.src.value += 2;
         }

         copy.dest += 2;
         copies2[num_copies2++] = copy;
      } else {
         copies2[num_copies2++] = copy;
      }
   }

   if (num_copies2 == 0) {
      free(copies2);
      return;
   }

   copies = copies2;
   num_copies = num_copies2;

   /* Set up the bookkeeping.  copy_ctx carries three AGX_NUM_MODELED_REGS
    * (2048) entry arrays -- roughly 72 KiB -- and this runs on whatever
    * thread the application called into the driver on.  A 1 MiB thread
    * stack does not have that to spare underneath the rest of the compiler,
    * so keep it on the heap.
    */
   struct copy_ctx *ctx = calloc(1, sizeof(*ctx));
   if (ctx == NULL) {
      free(copies2);
      UNREACHABLE("failed to allocate parallel-copy context");
   }

   ctx->entry_count = num_copies;

   for (unsigned i = 0; i < ctx->entry_count; i++) {
      struct agx_copy *entry = &copies[i];

      ctx->entries[i] = *entry;

      for (unsigned j = 0; j < agx_size_align_16(entry->src.size); j++) {
         if (is_real(entry))
            ctx->physreg_use_count[entry->src.value + j]++;

         /* Copies should not have overlapping destinations. */
         assert(!ctx->physreg_dest[entry->dest + j]);
         ctx->physreg_dest[entry->dest + j] = &ctx->entries[i];
      }
   }

   /* Try to vectorize aligned 16-bit copies to use 32-bit operations instead */
   for (unsigned i = 0; i < ctx->entry_count; i++) {
      struct agx_copy *entry = &ctx->entries[i];
      if (entry->src.size != AGX_SIZE_16)
         continue;

      if ((entry->dest & 1) || (entry->src.value & 1))
         continue;

      if (entry->src.type != AGX_INDEX_UNIFORM &&
          entry->src.type != AGX_INDEX_REGISTER)
         continue;

      unsigned next_dest = entry->dest + 1;
      assert(next_dest < ARRAY_SIZE(ctx->physreg_dest) && "aligned reg");

      struct agx_copy *next_copy = ctx->physreg_dest[next_dest];
      if (!next_copy)
         continue;

      assert(next_copy->dest == next_dest && "data structure invariant");
      assert(next_copy->src.size == AGX_SIZE_16 && "unaligned copy");

      if (next_copy->src.type != entry->src.type)
         continue;

      if (next_copy->src.value != (entry->src.value + 1))
         continue;

      /* Vectorize the copies */
      ctx->physreg_dest[next_dest] = entry;
      entry->src.size = AGX_SIZE_32;
      next_copy->done = true;
   }

   /* A GPR temporary lets a cycle be broken with one extra move: save the
    * value a blocked copy is about to overwrite, retarget the copies that read
    * it, and the cycle becomes a path for step 1.  Only GPR copies use it;
    * memory copies keep the swap path. */
   const bool use_temp = !cls && !agx_is_null(temp) &&
                         temp.type == AGX_INDEX_REGISTER && !temp.memory;

   bool progress = true;
   while (progress) {
      progress = false;

      /* Step 1: resolve paths in the transfer graph. This means finding
       * copies whose destination aren't blocked by something else and then
       * emitting them, continuing this process until every copy is blocked
       * and there are only cycles left.
       *
       * TODO: We should note that src is also available in dest to unblock
       * cycles that src is involved in.
       */

      for (unsigned i = 0; i < ctx->entry_count; i++) {
         struct agx_copy *entry = &ctx->entries[i];
         if (!entry->done && !entry_blocked(entry, ctx)) {
            entry->done = true;
            progress = true;
            do_copy(b, entry);
            for (unsigned j = 0; j < agx_size_align_16(entry->src.size); j++) {
               if (is_real(entry))
                  ctx->physreg_use_count[entry->src.value + j]--;
               ctx->physreg_dest[entry->dest + j] = NULL;
            }
         }
      }

      if (progress)
         continue;

      /* Step 2: Find partially blocked copies and split them. In the
       * mergedregs case, we can 32-bit copies which are only blocked on one
       * 16-bit half, and splitting them helps get things moving.
       *
       * We can skip splitting copies if the source isn't a register,
       * however, because it does not unblock anything and therefore doesn't
       * contribute to making forward progress with step 1. These copies
       * should still be resolved eventually in step 1 because they can't be
       * part of a cycle.
       */
      for (unsigned i = 0; i < ctx->entry_count; i++) {
         struct agx_copy *entry = &ctx->entries[i];
         if (entry->done || (agx_size_align_16(entry->src.size) != 2))
            continue;

         if (((ctx->physreg_use_count[entry->dest] == 0 ||
               ctx->physreg_use_count[entry->dest + 1] == 0)) &&
             is_real(entry)) {
            split_32bit_copy(ctx, entry);
            progress = true;
         }
      }

      if (progress || !use_temp)
         continue;

      /* Step 2b: only cycles remain.  Break one through the temporary: the
       * value in a blocked copy's destination is saved, the copy is emitted,
       * and every copy reading that destination now reads the temporary.
       * The cycle is then a path, so step 1 resumes.  Costs k + 1 moves for a
       * k-element cycle instead of 3 (k - 1) XORs.
       */
      for (unsigned i = 0; i < ctx->entry_count; i++) {
         struct agx_copy *entry = &ctx->entries[i];
         if (entry->done)
            continue;

         assert(is_real(entry));
         if (entry->dest == entry->src.value) {
            entry->done = true;
            progress = true;
            break;
         }

         const unsigned width = agx_size_align_16(entry->src.size);
         agx_index saved = agx_register(temp.value, entry->src.size);
         agx_index victim = agx_register(entry->dest, entry->src.size);
         agx_mov_to(b, saved, victim);

         do_copy(b, entry);
         entry->done = true;
         for (unsigned j = 0; j < width; j++) {
            ctx->physreg_use_count[entry->src.value + j]--;
            ctx->physreg_dest[entry->dest + j] = NULL;
         }

         /* Copies whose source lay (wholly or partly) inside the destination
          * now read the saved value.  A 32-bit reader straddling the
          * destination boundary is split first so each half can be redirected
          * independently. */
         if (width == 1) {
            for (unsigned j = 0; j < ctx->entry_count; j++) {
               struct agx_copy *blocking = &ctx->entries[j];
               if (blocking->done || !is_real(blocking))
                  continue;
               if (blocking->src.value <= entry->dest &&
                   blocking->src.value + 1 >= entry->dest &&
                   agx_size_align_16(blocking->src.size) == 2)
                  split_32bit_copy(ctx, blocking);
            }
         }

         for (unsigned j = 0; j < ctx->entry_count; j++) {
            struct agx_copy *blocking = &ctx->entries[j];
            if (blocking->done || !is_real(blocking))
               continue;
            if (blocking->src.value >= entry->dest &&
                blocking->src.value < entry->dest + width) {
               const unsigned offset = blocking->src.value - entry->dest;
               /* The temporary is live until the retargeted copies drain,
                * which step 1 does before the next cycle break: the cycle is
                * now a path whose every copy becomes unblocked in order. */
               ctx->physreg_use_count[blocking->src.value]--;
               blocking->src.value = temp.value + offset;
               ctx->physreg_use_count[blocking->src.value]++;
            }
         }

         progress = true;
         break;
      }
   }

   /* Step 3: resolve cycles through swapping.
    *
    * At this point, the transfer graph should consist of only cycles.
    * The reason is that, given any physreg n_1 that's the source of a
    * remaining entry, it has a destination n_2, which (because every
    * copy is blocked) is the source of some other copy whose destination
    * is n_3, and so we can follow the chain until we get a cycle. If we
    * reached some other node than n_1:
    *
    *  n_1 -> n_2 -> ... -> n_i
    *          ^             |
    *          |-------------|
    *
    *  then n_2 would be the destination of 2 copies, which is illegal
    *  (checked above in an assert). So n_1 must be part of a cycle:
    *
    *  n_1 -> n_2 -> ... -> n_i
    *  ^                     |
    *  |---------------------|
    *
    *  and this must be only cycle n_1 is involved in, because any other
    *  path starting from n_1 would also have to end in n_1, resulting in
    *  a node somewhere along the way being the destination of 2 copies
    *  when the 2 paths merge.
    *
    *  The way we resolve the cycle is through picking a copy (n_1, n_2)
    *  and swapping n_1 and n_2. This moves n_1 to n_2, so n_2 is taken
    *  out of the cycle:
    *
    *  n_1 -> ... -> n_i
    *  ^              |
    *  |--------------|
    *
    *  and we can keep repeating this until the cycle is empty.
    */

   for (unsigned i = 0; i < ctx->entry_count; i++) {
      struct agx_copy *entry = &ctx->entries[i];
      if (entry->done)
         continue;

      assert(is_real(entry));

      /* catch trivial copies */
      if (entry->dest == entry->src.value) {
         entry->done = true;
         continue;
      }

      do_swap(b, entry);

      /* Split any blocking copies whose sources are only partially
       * contained within our destination.
       */
      if (agx_size_align_16(entry->src.size) == 1) {
         for (unsigned j = 0; j < ctx->entry_count; j++) {
            struct agx_copy *blocking = &ctx->entries[j];

            if (blocking->done)
               continue;

            if (blocking->src.value <= entry->dest &&
                blocking->src.value + 1 >= entry->dest &&
                agx_size_align_16(blocking->src.size) == 2) {
               split_32bit_copy(ctx, blocking);
            }
         }
      }

      /* Update sources of blocking copies.
       *
       * Note: at this point, every blocking copy's source should be
       * contained within our destination.
       */
      for (unsigned j = 0; j < ctx->entry_count; j++) {
         struct agx_copy *blocking = &ctx->entries[j];
         if (blocking->src.value >= entry->dest &&
             blocking->src.value <
                entry->dest + agx_size_align_16(entry->src.size)) {
            blocking->src.value =
               entry->src.value + (blocking->src.value - entry->dest);
         }
      }

      entry->done = true;
   }

   free(ctx);
   free(copies2);
}

void
agx_emit_parallel_copies_temp(agx_builder *b, struct agx_copy *copies,
                              unsigned num_copies, agx_index temp)
{
   /* Emit copies fo reach register class separately because we don't have
    * register class awareness in the parallel copy lowering data structure.
    */
   if (num_copies == 0)
      return;

   agx_emit_parallel_copies_for_class(b, copies, num_copies, false, temp);
   agx_emit_parallel_copies_for_class(b, copies, num_copies, true, temp);
}

void
agx_emit_parallel_copies(agx_builder *b, struct agx_copy *copies,
                         unsigned num_copies)
{
   agx_emit_parallel_copies_temp(b, copies, num_copies, agx_null());
}
