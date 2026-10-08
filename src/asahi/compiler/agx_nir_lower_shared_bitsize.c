/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "compiler/nir/nir_builder.h"
#include "agx_compiler.h"
#include "agx_nir.h"

/* Local memory instructions require 16-bit offsets, so we add conversions. */
static bool
pass(struct nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   nir_src *offset = nir_get_io_offset_src(intr);
   if (!nir_is_shared_access(intr))
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *address = offset->ssa;

   /* AGX3 ldtg/sttg register indices count scalar elements, while AGX2's
    * local-memory base remains byte addressed. Keep atomics on their separate
    * byte-addressed encoding and convert only ordinary loads/stores. */
   const bool element_indices = *(const bool *)data;
   if (element_indices &&
       (intr->intrinsic == nir_intrinsic_load_shared ||
        intr->intrinsic == nir_intrinsic_store_shared)) {
      const unsigned bit_size =
         intr->intrinsic == nir_intrinsic_load_shared
            ? intr->def.bit_size
            : nir_src_bit_size(intr->src[0]);
      const unsigned shift = util_logbase2(bit_size / 8);
      address = nir_ushr_imm(b, address, shift);
   }

   if (address->bit_size != 16)
      address = nir_u2u16(b, address);

   if (address == offset->ssa)
      return false;

   nir_src_rewrite(offset, address);
   return true;
}

bool
agx_nir_lower_shared_bitsize(nir_shader *shader, bool element_indices)
{
   return nir_shader_intrinsics_pass(shader, pass, nir_metadata_control_flow,
                                     &element_indices);
}

/*
 * Divide `s` by 2^shift exactly, or return NULL when that cannot be done
 * without a shift instruction.  Every rewrite here is exact for every value of
 * the input: a constant divides only when it has no low bits, a left shift only
 * when it shifts at least as far, a multiply only when its constant factor is a
 * multiple of the divisor.  This is the fold nir_opt_algebraic would perform
 * had the conversion run before it -- doing it here keeps the pass usable after
 * the optimizer, where `ushr(ishl(a, 2), 2)` is instead rewritten into a mixed
 * width `iand` that the AGX3 encoder has no form for.
 */
static nir_def *
divide_exact(nir_builder *b, nir_scalar s, unsigned shift, unsigned depth)
{
   const uint64_t mask = (UINT64_C(1) << shift) - 1;

   if (nir_scalar_is_const(s)) {
      const uint64_t c = nir_scalar_as_uint(s);
      if (c & mask)
         return NULL;
      return nir_imm_intN_t(b, c >> shift, s.def->bit_size);
   }

   if (depth == 0 || !nir_scalar_is_alu(s))
      return NULL;

   switch (nir_scalar_alu_op(s)) {
   case nir_op_ishl: {
      const nir_scalar amount = nir_scalar_chase_alu_src(s, 1);
      if (!nir_scalar_is_const(amount))
         return NULL;
      const uint64_t k = nir_scalar_as_uint(amount);
      if (k < shift)
         return NULL;
      const nir_scalar value = nir_scalar_chase_alu_src(s, 0);
      nir_def *kept = nir_channel(b, value.def, value.comp);
      return k == shift ? kept : nir_ishl_imm(b, kept, k - shift);
   }

   case nir_op_imul: {
      for (unsigned i = 0; i < 2; ++i) {
         const nir_scalar factor = nir_scalar_chase_alu_src(s, i);
         if (!nir_scalar_is_const(factor))
            continue;
         const uint64_t f = nir_scalar_as_uint(factor);
         if (f & mask)
            return NULL;
         const nir_scalar value = nir_scalar_chase_alu_src(s, 1 - i);
         return nir_imul_imm(b, nir_channel(b, value.def, value.comp),
                             f >> shift);
      }
      return NULL;
   }

   case nir_op_iadd: {
      nir_def *x =
         divide_exact(b, nir_scalar_chase_alu_src(s, 0), shift, depth - 1);
      if (!x)
         return NULL;
      nir_def *y =
         divide_exact(b, nir_scalar_chase_alu_src(s, 1), shift, depth - 1);
      return y ? nir_iadd(b, x, y) : NULL;
   }

   default:
      return NULL;
   }
}

/*
 * AGX3 ldtg/sttg address threadgroup memory as
 *
 *    base + (index << shift) + byte_offset
 *
 * where `shift` is the access width -- agx3_translate_local() sets
 * shift = width (agx3_pack_binary.c), grounded against Apple's own output.
 * The index register
 * therefore counts *scalar elements*, while NIR's shared offsets count bytes.
 * AGX2's local-memory base stays byte addressed and AGX3 threadgroup atomics
 * keep their own byte-addressed immediate field (agx_local_atomic_base), so
 * only ordinary AGX3 loads and stores convert.
 *
 * Getting this wrong is silent: the instruction encodes either way and the
 * hardware simply strides four times too far for a 32-bit element, so a kernel
 * can only use the first quarter of its threadgroup block.  That is what every
 * internal (libagx) kernel did on G17P, because the conversion used to live
 * behind agx_preprocess_nir_g17p() and asahi_clc -- which preprocesses one
 * shader for g13g, g13x and g17p at once -- cannot call it.
 */
static bool
element_index(struct nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_shared &&
       intr->intrinsic != nir_intrinsic_store_shared)
      return false;

   const unsigned bit_size = intr->intrinsic == nir_intrinsic_load_shared
                                ? intr->def.bit_size
                                : nir_src_bit_size(intr->src[0]);
   const unsigned shift = util_logbase2(bit_size / 8);
   if (shift == 0)
      return false;

   nir_src *offset = nir_get_io_offset_src(intr);
   nir_def *address = offset->ssa;

   b->cursor = nir_before_instr(&intr->instr);

   /* agx_nir_lower_shared_bitsize() may already have narrowed the byte offset
    * to 16 bits.  Divide the value it was narrowed from, so the arithmetic
    * that produced the byte offset stays visible and the narrowing is applied
    * to the element index, exactly as when this conversion runs first. */
   nir_scalar narrow = nir_scalar_resolved(address, 0);
   if (nir_scalar_is_alu(narrow) && nir_scalar_alu_op(narrow) == nir_op_u2u16) {
      const nir_scalar wide = nir_scalar_chase_alu_src(narrow, 0);
      nir_def *elements = divide_exact(b, wide, shift, 4);
      if (!elements) {
         elements =
            nir_ushr_imm(b, nir_channel(b, wide.def, wide.comp), shift);
      }
      nir_src_rewrite(offset, nir_u2u16(b, elements));
      return true;
   }

   nir_def *elements = divide_exact(b, narrow, shift, 4);
   nir_src_rewrite(offset, elements ? elements
                                    : nir_ushr_imm(b, address, shift));
   return true;
}

bool
agx_nir_lower_shared_element_index(nir_shader *shader)
{
   /* Records that the conversion is now owned, so the backstop in
    * agx_compile_shader_nir() does not apply it a second time. */
   shader->info.shared_element_index_agx = true;
   return nir_shader_intrinsics_pass(shader, element_index,
                                     nir_metadata_control_flow, NULL);
}
