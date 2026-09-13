/*
 * Copyright 2023 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */
#include "agx_builder.h"
#include "agx_compiler.h"

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

   switch (op) {
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
      return src_index == 0 && !high;
   case AGX_OPCODE_DEVICE_STORE:
   case AGX_OPCODE_ATOMIC:
      return src_index == 1 && !high;
   case AGX_OPCODE_LOCAL_LOAD:
      return src_index == 0;
   case AGX_OPCODE_LOCAL_STORE:
      return src_index == 1;
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
   case AGX_OPCODE_PHI:
      /* We would fail validation otherwise */
      return true;
   default:
      /* Generic ALU.  AGX2 can read the uniform file directly here; AGX3
       * cannot -- its ALU operands are register-or-immediate only
       * (agx3_pack_binary.c:141-145 agx3_src_is_register, and the whole
       * agx3_sel_instr operand form has no uniform encoding).  Every
       * AGX_INDEX_UNIFORM the AGX3 packer does accept is a memory-op base,
       * texture heap, or subgroup source, all named explicitly above.  So on
       * AGX3 the generic case must be lowered to a move, or the shader is
       * refused at pack time with no way to recover.
       */
      if (encoding != AGX_ISA_ENCODING_AGX2)
         return false;

      return !is_64;
   }
}

void
agx_lower_uniform_sources(agx_context *ctx)
{
   agx_foreach_instr_global_safe(ctx, I) {
      agx_builder b = agx_init_builder(ctx, agx_before_instr(I));

      agx_foreach_src(I, s) {
         if (I->src[s].type == AGX_INDEX_UNIFORM &&
             !agx_instr_accepts_uniform(ctx->key->dev.encoding, I->op, s,
                                        I->src[s].value, I->src[s].size)) {

            agx_index idx = I->src[s];
            idx.abs = idx.neg = false;
            if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
                idx.size == AGX_SIZE_16 && agx_channels(idx) == 1) {
               /* AGX3c has witnessed dword uniform reads and half-register
                * moves, but no witnessed uniform-to-half copy. Read the
                * containing aligned word and select the original half. This
                * preserves every bit, including an odd uniform's upper half,
                * without inventing an instruction encoding.
                */
               const unsigned half = idx.value & 1;
               idx.value &= ~1u;
               idx.size = AGX_SIZE_32;
               agx_index word = agx_mov(&b, idx);
               agx_instr *split = agx_split(&b, 2, word);
               split->dest[0] = agx_temp(ctx, AGX_SIZE_16);
               split->dest[1] = agx_temp(ctx, AGX_SIZE_16);
               agx_replace_src(I, s, split->dest[half]);
            } else {
               agx_replace_src(I, s, agx_mov(&b, idx));
            }
         }
      }
   }
}
