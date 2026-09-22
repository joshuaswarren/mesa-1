/*
 * Copyright 2023 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */
#include "agx_builder.h"
#include "agx_compiler.h"
#include "agx3_uniform_integer_instr.h"

/*
 * Not all instructions can take uniforms. Memory instructions can take
 * uniforms, but only for their base (first) source and only in the
 * low-half of the uniform file.
 *
 * This pass lowers invalid uniforms.
 */
bool
agx_instr_accepts_uniform(enum agx_isa_encoding encoding, enum agx_opcode op,
                          unsigned src_index, unsigned value,
                          enum agx_size size)
{
   /* Some instructions only seem able to access uniforms in the low half */
   bool high = value >= 256;

   /* ALU cannot access 64-bit uniforms */
   bool is_64 = size == AGX_SIZE_64;
   /* AGX3 memory bases address four-half uniform groups. A pointer packed
    * at a two-half boundary must be copied to a GPR pair before use. */
   bool memory_base = !high &&
      (encoding == AGX_ISA_ENCODING_AGX2 || !(value & 3));

   switch (op) {
   case AGX_OPCODE_FCMPSEL:
      if (encoding == AGX_ISA_ENCODING_AGX3B)
         return src_index < 4 && value < 512 &&
                (size == AGX_SIZE_16 || (size == AGX_SIZE_32 && !(value & 1)));
      return encoding == AGX_ISA_ENCODING_AGX2 && !is_64;
   case AGX_OPCODE_HADD:
   case AGX_OPCODE_HMUL:
   case AGX_OPCODE_HFMA:
      if (encoding == AGX_ISA_ENCODING_AGX3B)
         return size == AGX_SIZE_16 && value < 512 &&
                src_index < (op == AGX_OPCODE_HFMA ? 3 : 2);
      return encoding == AGX_ISA_ENCODING_AGX2 && !is_64;
   case AGX_OPCODE_FADD:
   case AGX_OPCODE_FMUL:
   case AGX_OPCODE_FFMA:
      /* Apple G17P/G16G arithmetic reads aligned f32 uniforms directly.
       * The instruction-level check below retains copies for mixed widths. */
      if (encoding == AGX_ISA_ENCODING_AGX3B)
         return size == AGX_SIZE_32 && value < 512 && !(value & 1) &&
                src_index < (op == AGX_OPCODE_FFMA ? 3 : 2);
      return encoding == AGX_ISA_ENCODING_AGX2 && !is_64;
   case AGX_OPCODE_FLTNORM_AGX3:
      /* G17P colour packing has native uniform operands in both lanes.
       * Preserve aligned float32 sources instead of allocating temporary
       * GPRs and copying the constant buffer before every pack. */
      return encoding == AGX_ISA_ENCODING_AGX3B && src_index < 2 &&
             size == AGX_SIZE_32 && value < 512 && !(value & 1);
   case AGX_OPCODE_IMAGE_LOAD:
   case AGX_OPCODE_TEXTURE_LOAD:
      /* G17P publishes a dynamic integer LOD through PACK_SMPIR from a GPR;
       * there is no witnessed uniform-file source for that operand.  Keep
       * the heap in uniforms, but materialize the LOD into a register. */
      assert(!(src_index == 2 && high) && "texture heap always low");
      return !high &&
             (src_index == 2 ||
              (encoding == AGX_ISA_ENCODING_AGX2 && src_index == 1));
   case AGX_OPCODE_TEXTURE_SAMPLE:
      /* G17P's LOD operand is an immediate level or a packed GPR value, not a
       * uniform-file operand. Source 2 is still the bindless texture heap. */
      assert(!(src_index == 2 && high) && "texture heap always low");
      return !high && (src_index == 2 ||
                       (encoding == AGX_ISA_ENCODING_AGX2 && src_index == 1));

   case AGX_OPCODE_TEX_STATE_STORE:
   case AGX_OPCODE_SAMPLER_STATE_STORE:
      assert(!high && "texture heap always low");
      return !high && src_index == 0;

   case AGX_OPCODE_DEVICE_LOAD:
      return src_index == 0 && memory_base;
   case AGX_OPCODE_DEVICE_STORE:
   case AGX_OPCODE_ATOMIC:
      return src_index == 1 && memory_base;
   case AGX_OPCODE_LOCAL_LOAD:
   case AGX_OPCODE_LOCAL_STORE:
      /* AGX3 threadgroup accesses with a promoted-constant uniform base
       * silently miss on G16G once the shader carries many of them
       * (tools/shared-atomic-probe.c mode 4: a 145-word shared struct copy
       * left every result untouched; AGX_MESA_DEBUG=nopromote passes). The
       * only witnessed uniform base is u4 in one small kernel, so keep
       * threadgroup bases in GPRs on AGX3B and let this pass copy them.
       * AGX3C still uses the fixed u8 base and has no grounded half-uniform
       * copy; retain that existing encoding's local-memory contract. */
      if (encoding == AGX_ISA_ENCODING_AGX3B)
         return false;
      return src_index == (op == AGX_OPCODE_LOCAL_LOAD ? 0 : 1);
   case AGX_OPCODE_IMAGE_WRITE:
      return src_index == 3;
   case AGX_OPCODE_BLOCK_IMAGE_STORE:
   case AGX_OPCODE_ZS_EMIT:
   case AGX_OPCODE_ZS_EMIT_AGX3:
   case AGX_OPCODE_ST_TILE:
   case AGX_OPCODE_LD_TILE:
   case AGX_OPCODE_UNIFORM_STORE:
   case AGX_OPCODE_ST_VARY:
   case AGX_OPCODE_LOCAL_ATOMIC:
   case AGX_OPCODE_SAMPLE_MASK:
   case AGX_OPCODE_ITER:
   case AGX_OPCODE_ITERPROJ:
   case AGX_OPCODE_STACK_LOAD:
   case AGX_OPCODE_STACK_STORE:
   case AGX_OPCODE_BALLOT:
   case AGX_OPCODE_FCMP_BALLOT:
   case AGX_OPCODE_ICMP_BALLOT:
   case AGX_OPCODE_QUAD_BALLOT:
   case AGX_OPCODE_FCMP_QUAD_BALLOT:
   case AGX_OPCODE_ICMP_QUAD_BALLOT:
      return false;
   case AGX_OPCODE_EXPORT:
      /* Register allocation copies an export's uniform source into its
       * output register. AGX3 grounds that copy only for whole words; a
       * half uniform (a 16-bit constant such as a stencil reference) has no
       * witnessed uniform-to-half copy, so it is split from its aligned word
       * below, ahead of the export run. */
      return !(encoding == AGX_ISA_ENCODING_AGX3B && size == AGX_SIZE_16);
   case AGX_OPCODE_PHI:
      /* We would fail validation otherwise */
      return true;
   default:
      /* Keep copies for operands not yet supported by the AGX3 packer.
       * This is a compiler limitation, not an ISA-wide uniform restriction. */
      if (encoding != AGX_ISA_ENCODING_AGX2)
         return false;

      return !is_64;
   }
}

void
agx_lower_uniform_sources(agx_context *ctx)
{
   agx_foreach_instr_global_safe(ctx, I) {
      agx_instr *at = I;
      struct agx3_uniform_integer_desc integer_desc;
      const bool native_integer =
         ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
         agx3_uniform_integer_instruction(I, false, &integer_desc);
      bool copy_float_uniforms = false;
      unsigned first_uniform = UINT_MAX;
      const bool half_float = I->op == AGX_OPCODE_HADD ||
                              I->op == AGX_OPCODE_HMUL || I->op == AGX_OPCODE_HFMA;
      const bool full_float = I->op == AGX_OPCODE_FADD ||
                              I->op == AGX_OPCODE_FMUL || I->op == AGX_OPCODE_FFMA;
      bool native_minmax = I->op == AGX_OPCODE_FCMPSEL && I->nr_dests == 1 &&
                           I->nr_srcs == 4 && !I->saturate && !I->invert_cond &&
                           (I->fcond == AGX_FCOND_LTN || I->fcond == AGX_FCOND_GTN) &&
                           agx_is_equiv(I->src[0], I->src[2]) &&
                           agx_is_equiv(I->src[1], I->src[3]);
      if (native_minmax) {
         agx_foreach_src(I, s) {
            native_minmax &= I->src[s].type != AGX_INDEX_IMMEDIATE &&
                             I->src[s].size == I->dest[0].size &&
                             !I->src[s].abs && !I->src[s].neg;
         }
      }
      const bool both_minmax_uniform = native_minmax &&
         I->src[0].type == AGX_INDEX_UNIFORM && I->src[1].type == AGX_INDEX_UNIFORM;
      if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          (half_float || full_float)) {
         const enum agx_size size = half_float ? AGX_SIZE_16 : AGX_SIZE_32;
         copy_float_uniforms = I->nr_dests != 1 ||
                               I->dest[0].size != size || (half_float && I->saturate);
         bool has_gpr = false;
         agx_foreach_src(I, s) {
            if (I->src[s].type != AGX_INDEX_IMMEDIATE)
               copy_float_uniforms |= I->src[s].size != size;
            if (half_float)
               copy_float_uniforms |= I->src[s].abs || I->src[s].neg;
            has_gpr |= I->src[s].type != AGX_INDEX_UNIFORM &&
                       I->src[s].type != AGX_INDEX_IMMEDIATE;
            if (first_uniform == UINT_MAX && I->src[s].type == AGX_INDEX_UNIFORM)
               first_uniform = s;
         }
         if (has_gpr)
            first_uniform = UINT_MAX;
      }

      /* Exports end the block and admit no instruction between them; place
       * their materialized sources before the first export of the run. */
      if (I->op == AGX_OPCODE_EXPORT) {
         while (at->link.prev != &v_block->instructions) {
            agx_instr *prev = list_entry(at->link.prev, agx_instr, link);
            if (prev->op != AGX_OPCODE_EXPORT)
               break;
            at = prev;
         }
      }

      agx_builder b = agx_init_builder(ctx, agx_before_instr(at));

      /* A tied compare-select (fmin/fmax lowered to fcmpsel(a, b, a, b)) is
       * grounded on AGX3 only while its compared and selected sources stay the
       * same register. Materialize a uniform once per instruction and reuse
       * the copy for every source that named the same uniform. */
      agx_index materialized_from[8] = {0};
      agx_index materialized_to[8] = {0};
      unsigned materialized = 0;

      agx_foreach_src(I, s) {
         const bool copy_minmax = ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
            I->op == AGX_OPCODE_FCMPSEL &&
            (!native_minmax || (both_minmax_uniform && !(s & 1)));
         /* Apple may precompute abs(uniform) in its constant program. The
          * native main-program witness proves negation only on an FMA addend;
          * retain materialization for the other modified uniform operands. */
         const bool copy_float_modifier =
            ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B && full_float &&
            (I->src[s].abs ||
             (I->src[s].neg && !(I->op == AGX_OPCODE_FFMA && s == 2)));
         if (I->src[s].type == AGX_INDEX_UNIFORM &&
             (s == first_uniform || copy_minmax || copy_float_uniforms || copy_float_modifier ||
              !(native_integer || agx_instr_accepts_uniform(ctx->key->dev.encoding, I->op, s,
                                         I->src[s].value, I->src[s].size)))) {
            agx_index idx = I->src[s];
            idx.abs = idx.neg = false;
            bool reused = false;
            for (unsigned m = 0; m < materialized; ++m) {
               if (agx_is_equiv(materialized_from[m], idx) &&
                   materialized_from[m].size == idx.size) {
                  agx_replace_src(I, s, materialized_to[m]);
                  reused = true;
                  break;
               }
            }
            if (reused)
               continue;
            if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
                idx.size == AGX_SIZE_16) {
               /* Native uniform copies fetch an aligned word. Extract the
                * requested half through an SSA split, so RA can coalesce
                * it without reading the neighbouring half as the value.
                */
               agx_index word = agx_mov(&b, agx_uniform(idx.value & ~1u,
                                                      AGX_SIZE_32));
               agx_instr *split = agx_split(&b, 2, word);
               split->dest[0] = agx_temp(ctx, AGX_SIZE_16);
               split->dest[1] = agx_temp(ctx, AGX_SIZE_16);
               agx_replace_src(I, s, split->dest[idx.value & 1]);
               if (materialized < ARRAY_SIZE(materialized_from)) {
                  materialized_from[materialized] = idx;
                  materialized_to[materialized++] = split->dest[idx.value & 1];
               }
            } else {
               agx_index copy = agx_mov(&b, idx);
               agx_replace_src(I, s, copy);
               if (materialized < ARRAY_SIZE(materialized_from)) {
                  materialized_from[materialized] = idx;
                  materialized_to[materialized++] = copy;
               }
            }
         }
      }
   }
}
