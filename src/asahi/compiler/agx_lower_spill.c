/*
 * Copyright 2023 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "util/macros.h"
#include "agx_builder.h"
#include "agx_compile.h"
#include "agx_compiler.h"

/* Lower moves involving memory registers (created when spilling) to concrete
 * spills and fills.
 */

static void
spill_fill(agx_builder *b, agx_instr *I, enum agx_size size, unsigned channels,
           unsigned component_offset_el)
{
   unsigned size_B = agx_size_align_16(size) * 2;
   enum agx_format format = size_B == 2 ? AGX_FORMAT_I16 : AGX_FORMAT_I32;
   unsigned format_size_B = size_B == 2 ? 2 : 4;

   unsigned offset_B = component_offset_el * size_B;
   unsigned effective_chans = size == AGX_SIZE_64 ? (channels * 2) : channels;
   unsigned mask = BITFIELD_MASK(effective_chans);

   assert(effective_chans <= 4);

   /* Pick off the memory and register parts of the move */
   agx_index mem = I->dest[0].memory ? I->dest[0] : I->src[0];
   agx_index reg = I->dest[0].memory ? I->src[0] : I->dest[0];

   assert(mem.type == AGX_INDEX_REGISTER && mem.memory);
   assert(reg.type == AGX_INDEX_REGISTER && !reg.memory);

   /* Slice the register according to the part of the spill we're handling */
   if (component_offset_el > 0 || channels != agx_channels(reg)) {
      reg.value += component_offset_el * agx_size_align_16(reg.size);
      reg.channels_m1 = channels - 1;
   }

   /* Calculate stack offset in bytes. IR registers are 2-bytes each. */
   unsigned stack_offs_B = b->shader->spill_base_B + (mem.value * 2) + offset_B;
   unsigned stack_offs_end_B = stack_offs_B + (effective_chans * format_size_B);

   assert(stack_offs_end_B <= b->shader->scratch_size_B &&
          "RA allocates enough scratch");

   /* Emit the spill/fill */
   if (I->dest[0].memory) {
      agx_stack_store(b, reg, agx_immediate(stack_offs_B), format, mask);
   } else {
      agx_stack_load_to(b, reg, agx_immediate(stack_offs_B), format, mask);
   }
}

void
agx_lower_spill(agx_context *ctx)
{
   agx_foreach_instr_global_safe(ctx, I) {
      if (I->op != AGX_OPCODE_MOV || (!I->dest[0].memory && !I->src[0].memory))
         continue;

      enum agx_size size = I->dest[0].size;
      unsigned channels = agx_channels(I->dest[0]);

      assert(size == I->src[0].size);
      assert(channels == agx_channels(I->src[0]));

      /* Texture gradient sources can be vec6, and if such a vector is spilled,
       * we need to be able to spill/fill a vec6. Since stack_store/stack_load
       * only work up to vec4, we break up into (at most) vec4 components.
       */
      agx_builder b = agx_init_builder(ctx, agx_before_instr(I));

      for (unsigned c = 0; c < channels; c += 4) {
         spill_fill(&b, I, size, MIN2(channels - c, 4), c);
      }

      agx_remove_instruction(I);
   }
}


/*
 * Split a multi-component thread-private-storage access into one component
 * per instruction, for AGX3.
 *
 * G17P's grounded static TPS forms move a single 16-bit or 32-bit component:
 * agx3_translate_static_spill() admits `mask == 1` only, and the AGX3 slot
 * allocator describes the TPS tokens the same way. Any wider access failed
 * the whole shader closed -- so a vec4 spill, or a vector element of a
 * scratch-lowered array, was unencodable regardless of what the shader
 * computed. That silently capped shader complexity across the driver.
 *
 * Splitting needs no new encoding: `mask` is always contiguous from component
 * zero (both producers build it with BITFIELD_MASK), the components live in
 * consecutive registers, and their stack slots are consecutive, so N scalar
 * accesses at `offset + size*c` are exactly equivalent. Both producers are
 * covered by running after register allocation: agx_lower_spill()'s real
 * spills and agx_emit_{load,store}_scratch()'s NIR scratch access.
 *
 * Restricted to 16-bit and 32-bit components with an immediate offset -- the
 * domain the scalar TPS forms are witnessed at. A spilled 64-bit scalar is
 * represented as an I32 mask of two consecutive words, so reinterpret those
 * two halves as independent W32 copies as well. Anything else is left exactly
 * as it was and still refuses, rather than being split into something
 * unwitnessed. AGX2 never reaches here, so none of its bytes change.
 */
void
agx3_split_vector_stack_access(agx_context *ctx)
{
   agx_foreach_instr_global_safe(ctx, I) {
      const bool is_store = I->op == AGX_OPCODE_STACK_STORE;
      if (!is_store && I->op != AGX_OPCODE_STACK_LOAD)
         continue;

      const unsigned nr = util_bitcount(I->mask);
      if (nr <= 1)
         continue;

      /* Only a contiguous mask starting at component 0 is a plain vector. */
      if (I->mask != BITFIELD_MASK(nr))
         continue;

      const agx_index data = is_store ? I->src[0] : I->dest[0];
      const agx_index offset = I->src[is_store ? 1 : 0];
      const enum agx_size size = I->format == AGX_FORMAT_I32   ? AGX_SIZE_32
                                 : I->format == AGX_FORMAT_I16 ? AGX_SIZE_16
                                                               : AGX_SIZE_64;
      const bool split_64_as_32 =
         size == AGX_SIZE_32 && data.size == AGX_SIZE_64 &&
         nr == 2 * agx_channels(data);
      if (size == AGX_SIZE_64 || offset.type != AGX_INDEX_IMMEDIATE ||
          (data.size != size && !split_64_as_32))
         continue;

      /* IR register values are 16-bit halves: a 32-bit component is two of
       * them and occupies four stack bytes, a 16-bit one is one and two. */
      const unsigned halves = agx_size_align_16(size);
      agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
      for (unsigned c = 0; c < nr; ++c) {
         agx_index comp = data;
         comp.value += c * halves;
         comp.channels_m1 = 0;
         comp.size = size;

         const agx_index off = agx_immediate(offset.value + (c * halves * 2));
         if (is_store)
            agx_stack_store(&b, comp, off, I->format, 1);
         else
            agx_stack_load_to(&b, comp, off, I->format, 1);
      }

      agx_remove_instruction(I);
   }
}
