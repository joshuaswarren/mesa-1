/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_select_core.h"

#include <string.h>

#include "agx3_pack.h"

/* A 32-bit register source that carries no modifier -- the only source shape
 * the grounded no-modifier packers can encode. */
static bool
src_is_plain_reg32(const struct agx3_sel_src *s)
{
   return s->is_reg && !s->abs && !s->neg && s->size == AGX3_SEL_SIZE_32;
}

static bool
dest_is_reg32(const struct agx3_sel_instr *I)
{
   return I->has_dest && I->dest_size == AGX3_SEL_SIZE_32;
}

bool
agx3_select_encode(enum agx_isa_encoding encoding,
                   const struct agx3_sel_instr *I,
                   uint8_t out[AGX3_SEL_MAX_LENGTH], uint8_t *len)
{
   if (out == NULL || len == NULL || I == NULL)
      return false;

   /* Only the two AGX3 encodings; agx2 has its own packer. */
   if (encoding != AGX_ISA_ENCODING_AGX3B &&
       encoding != AGX_ISA_ENCODING_AGX3C)
      return false;

   memset(out, 0, AGX3_SEL_MAX_LENGTH);
   *len = 0;

   /* Milestone one never emits float modifiers or saturation, and never a
    * shift on the integer add/mad forms; the grounded packers cannot carry
    * them, so refuse rather than drop them silently. */
   if (I->saturate)
      return false;

   switch (I->op) {
   case AGX3_SEL_OP_END: {
      struct agx3_encoded_end e;
      if (!agx3_pack_end(encoding, &e))
         return false;
      memcpy(out, e.bytes, e.length);
      *len = e.length;
      return true;
   }

   case AGX3_SEL_OP_BARRIER: {
      struct agx3_encoded_threadgroup_barrier e;
      if (!agx3_pack_threadgroup_barrier(encoding, &e))
         return false;
      memcpy(out, e.bytes, e.length);
      *len = e.length;
      return true;
   }

   case AGX3_SEL_OP_MOV_IMM: {
      struct agx3_encoded_ldimm e;
      if (!dest_is_reg32(I))
         return false;
      if (!agx3_pack_ldimm_u32(encoding, I->dest, I->imm, &e))
         return false;
      memcpy(out, e.bytes, e.length);
      *len = e.length;
      return true;
   }

   case AGX3_SEL_OP_FADD: {
      struct agx3_encoded_fadd_f32_reg_reg e;
      if (!dest_is_reg32(I) || I->nr_srcs != 2 ||
          !src_is_plain_reg32(&I->src[0]) || !src_is_plain_reg32(&I->src[1]))
         return false;
      if (!agx3_pack_fadd_f32_reg_reg(encoding, I->dest, I->src[0].reg,
                                      I->src[1].reg, &e))
         return false;
      memcpy(out, e.bytes, e.length);
      *len = e.length;
      return true;
   }

   case AGX3_SEL_OP_FMUL: {
      struct agx3_encoded_fmul_f32_reg_reg e;
      if (!dest_is_reg32(I) || I->nr_srcs != 2 ||
          !src_is_plain_reg32(&I->src[0]) || !src_is_plain_reg32(&I->src[1]))
         return false;
      if (!agx3_pack_fmul_f32_reg_reg(encoding, I->dest, I->src[0].reg,
                                      I->src[1].reg, &e))
         return false;
      memcpy(out, e.bytes, e.length);
      *len = e.length;
      return true;
   }

   case AGX3_SEL_OP_FFMA: {
      struct agx3_encoded_fma_f32_reg_reg_reg e;
      if (!dest_is_reg32(I) || I->nr_srcs != 3 ||
          !src_is_plain_reg32(&I->src[0]) || !src_is_plain_reg32(&I->src[1]) ||
          !src_is_plain_reg32(&I->src[2]))
         return false;
      if (!agx3_pack_fma_f32_reg_reg_reg(encoding, I->dest, I->src[0].reg,
                                         I->src[1].reg, I->src[2].reg, &e))
         return false;
      memcpy(out, e.bytes, e.length);
      *len = e.length;
      return true;
   }

   case AGX3_SEL_OP_IADD: {
      if (!dest_is_reg32(I) || I->nr_srcs != 2 || I->shift != 0 ||
          !src_is_plain_reg32(&I->src[0]))
         return false;

      /*
       * Adding a small constant. Apple's iadd/imm8 form is witnessed across
       * the immediate boundary set (1, 63, 64, 128, 255), and the move
       * lowering below already relies on it with a zero immediate. Constant
       * address arithmetic, which every shader with more than one store
       * does, needs this form without a scoreboard wait.
       */
      if (!I->src[1].is_reg && !I->src[1].abs && !I->src[1].neg) {
         if (I->src[1].reg > 0xff)
            return false;

         struct agx3_encoded_iadd_u32_imm8 imm;
         if (!agx3_pack_iadd_u32_imm8(encoding, I->dest, I->src[0].reg,
                                      (uint8_t)I->src[1].reg, &imm))
            return false;
         memcpy(out, imm.bytes, imm.length);
         *len = imm.length;
         return true;
      }

      struct agx3_encoded_iadd_u32_reg_reg e;
      if (!src_is_plain_reg32(&I->src[1]))
         return false;
      if (!agx3_pack_iadd_u32_reg_reg(encoding, I->dest, I->src[0].reg,
                                      I->src[1].reg, &e))
         return false;
      memcpy(out, e.bytes, e.length);
      *len = e.length;
      return true;
   }

   case AGX3_SEL_OP_IMAD: {
      struct agx3_encoded_imadd_u32_reg_reg_reg e;
      if (!dest_is_reg32(I) || I->nr_srcs != 3 || I->shift != 0 ||
          !src_is_plain_reg32(&I->src[0]) || !src_is_plain_reg32(&I->src[1]) ||
          !src_is_plain_reg32(&I->src[2]))
         return false;
      if (!agx3_pack_imadd_u32_reg_reg_reg(encoding, I->dest, I->src[0].reg,
                                           I->src[1].reg, I->src[2].reg, &e))
         return false;
      memcpy(out, e.bytes, e.length);
      *len = e.length;
      return true;
   }

   case AGX3_SEL_OP_ASR: {
      struct agx3_encoded_asr_i32_reg_reg e;
      if (!dest_is_reg32(I) || I->nr_srcs != 2 ||
          !src_is_plain_reg32(&I->src[0]) || !src_is_plain_reg32(&I->src[1]))
         return false;
      if (!agx3_pack_asr_i32_reg_reg(encoding, I->dest, I->src[0].reg,
                                     I->src[1].reg, &e))
         return false;
      memcpy(out, e.bytes, e.length);
      *len = e.length;
      return true;
   }

   case AGX3_SEL_OP_ISUB: {
      struct agx3_encoded_isub_u32_reg_reg e;
      if (!dest_is_reg32(I) || I->nr_srcs != 2 ||
          !src_is_plain_reg32(&I->src[0]) || !src_is_plain_reg32(&I->src[1]))
         return false;
      if (!agx3_pack_isub_u32_reg_reg(encoding, I->dest, I->src[0].reg,
                                      I->src[1].reg, &e))
         return false;
      memcpy(out, e.bytes, e.length);
      *len = e.length;
      return true;
   }

   case AGX3_SEL_OP_IMUL: {
      struct agx3_encoded_imul_u32_reg_reg e;
      if (!dest_is_reg32(I) || I->nr_srcs != 2 ||
          !src_is_plain_reg32(&I->src[0]) || !src_is_plain_reg32(&I->src[1]))
         return false;
      if (!agx3_pack_imul_u32_reg_reg(encoding, I->dest, I->src[0].reg,
                                      I->src[1].reg, &e))
         return false;
      memcpy(out, e.bytes, e.length);
      *len = e.length;
      return true;
   }

   case AGX3_SEL_OP_SLL: {
      struct agx3_encoded_sll_u32_reg_reg e;
      if (!dest_is_reg32(I) || I->nr_srcs != 2 ||
          !src_is_plain_reg32(&I->src[0]) || !src_is_plain_reg32(&I->src[1]))
         return false;
      if (!agx3_pack_sll_u32_reg_reg(encoding, I->dest, I->src[0].reg,
                                     I->src[1].reg, &e))
         return false;
      memcpy(out, e.bytes, e.length);
      *len = e.length;
      return true;
   }

   case AGX3_SEL_OP_LSR: {
      struct agx3_encoded_lsr_u32_reg_reg e;
      if (!dest_is_reg32(I) || I->nr_srcs != 2 ||
          !src_is_plain_reg32(&I->src[0]) || !src_is_plain_reg32(&I->src[1]))
         return false;
      if (!agx3_pack_lsr_u32_reg_reg(encoding, I->dest, I->src[0].reg,
                                     I->src[1].reg, &e))
         return false;
      memcpy(out, e.bytes, e.length);
      *len = e.length;
      return true;
   }

   case AGX3_SEL_OP_F_TO_I32:
   case AGX3_SEL_OP_F_TO_U32: {
      struct agx3_encoded_f32_to_int_rtz_reg e;
      const enum agx3_f32_to_int_rtz_mode mode =
         (I->op == AGX3_SEL_OP_F_TO_I32) ? AGX3_F32_TO_I32_RTZ
                                         : AGX3_F32_TO_U32_RTZ;
      if (!dest_is_reg32(I) || I->nr_srcs != 1 ||
          !src_is_plain_reg32(&I->src[0]))
         return false;
      if (!agx3_pack_f32_to_int_rtz_reg(encoding, mode, I->dest, I->src[0].reg,
                                        &e))
         return false;
      memcpy(out, e.bytes, e.length);
      *len = e.length;
      return true;
   }

   case AGX3_SEL_OP_I32_TO_F: {
      struct agx3_encoded_i32_to_f32_reg e;
      if (!dest_is_reg32(I) || I->nr_srcs != 1 ||
          !src_is_plain_reg32(&I->src[0]))
         return false;
      if (!agx3_pack_i32_to_f32_reg(encoding, I->dest, I->src[0].reg, &e))
         return false;
      memcpy(out, e.bytes, e.length);
      *len = e.length;
      return true;
   }

   case AGX3_SEL_OP_U32_TO_F: {
      struct agx3_encoded_u32_to_f32_reg e;
      if (!dest_is_reg32(I) || I->nr_srcs != 1 ||
          !src_is_plain_reg32(&I->src[0]))
         return false;
      if (!agx3_pack_u32_to_f32_reg(encoding, I->dest, I->src[0].reg, &e))
         return false;
      memcpy(out, e.bytes, e.length);
      *len = e.length;
      return true;
   }

   case AGX3_SEL_OP_FMIN: {
      struct agx3_encoded_fmin_f32_reg_reg e;
      if (!dest_is_reg32(I) || I->nr_srcs != 2 ||
          !src_is_plain_reg32(&I->src[0]) || !src_is_plain_reg32(&I->src[1]))
         return false;
      if (!agx3_pack_fmin_f32_reg_reg(encoding, I->dest, I->src[0].reg,
                                      I->src[1].reg, &e))
         return false;
      memcpy(out, e.bytes, e.length);
      *len = e.length;
      return true;
   }

   case AGX3_SEL_OP_FMAX: {
      struct agx3_encoded_fmax_f32_reg_reg e;
      if (!dest_is_reg32(I) || I->nr_srcs != 2 ||
          !src_is_plain_reg32(&I->src[0]) || !src_is_plain_reg32(&I->src[1]))
         return false;
      if (!agx3_pack_fmax_f32_reg_reg(encoding, I->dest, I->src[0].reg,
                                      I->src[1].reg, &e))
         return false;
      memcpy(out, e.bytes, e.length);
      *len = e.length;
      return true;
   }

   case AGX3_SEL_OP_F_RCP: {
      struct agx3_encoded_f32_rcp_reg e;
      if (!dest_is_reg32(I) || I->nr_srcs != 1 ||
          !src_is_plain_reg32(&I->src[0]))
         return false;
      if (!agx3_pack_f32_rcp_reg(encoding, I->dest, I->src[0].reg, &e))
         return false;
      memcpy(out, e.bytes, e.length);
      *len = e.length;
      return true;
   }

   case AGX3_SEL_OP_F_RSQRT: {
      struct agx3_encoded_f32_rsqrt_reg e;
      if (!dest_is_reg32(I) || I->nr_srcs != 1 ||
          !src_is_plain_reg32(&I->src[0]))
         return false;
      if (!agx3_pack_f32_rsqrt_reg(encoding, I->dest, I->src[0].reg, &e))
         return false;
      memcpy(out, e.bytes, e.length);
      *len = e.length;
      return true;
   }

   case AGX3_SEL_OP_F_RINT: {
      struct agx3_encoded_f32_rint_reg e;
      if (!dest_is_reg32(I) || I->nr_srcs != 1 ||
          !src_is_plain_reg32(&I->src[0]))
         return false;
      if (!agx3_pack_f32_rint_reg(encoding, I->dest, I->src[0].reg, &e))
         return false;
      memcpy(out, e.bytes, e.length);
      *len = e.length;
      return true;
   }

   case AGX3_SEL_OP_BITOP: {
      if (!dest_is_reg32(I) || !src_is_plain_reg32(&I->src[0]))
         return false;

      /* The NOT truth table depends only on the first source, so Apple encodes
       * it as the unary `inv`; the second bitop source (a zero) is ignored. */
      if ((I->truth_table & 0xfu) == AGX3_SEL_TT_NOT) {
         struct agx3_encoded_bit_not_u32_reg e;
         if (I->nr_srcs < 1 ||
             !agx3_pack_bit_not_u32_reg(encoding, I->dest, I->src[0].reg, &e))
            return false;
         memcpy(out, e.bytes, e.length);
         *len = e.length;
         return true;
      }

      /* A move is a bitop that returns its first source unchanged. No AGX3
       * truth table is the identity, so Apple materializes it as `iadd d, s,
       * #0` -- a bit-exact copy that also serves a modifier-free fmov, since a
       * move is bitwise. */
      if ((I->truth_table & 0xfu) == AGX3_SEL_TT_MOV) {
         struct agx3_encoded_iadd_u32_imm8 e;
         if (I->nr_srcs < 1 ||
             !agx3_pack_iadd_u32_imm8(encoding, I->dest, I->src[0].reg, 0, &e))
            return false;
         memcpy(out, e.bytes, e.length);
         *len = e.length;
         return true;
      }

      if (I->nr_srcs != 2 || !src_is_plain_reg32(&I->src[1]))
         return false;

      switch (I->truth_table & 0xfu) {
      case AGX3_SEL_TT_AND: {
         struct agx3_encoded_bit_and_u32_reg_reg e;
         if (!agx3_pack_bit_and_u32_reg_reg(encoding, I->dest, I->src[0].reg,
                                            I->src[1].reg, &e))
            return false;
         memcpy(out, e.bytes, e.length);
         *len = e.length;
         return true;
      }
      case AGX3_SEL_TT_OR: {
         struct agx3_encoded_bit_or_u32_reg_reg e;
         if (!agx3_pack_bit_or_u32_reg_reg(encoding, I->dest, I->src[0].reg,
                                           I->src[1].reg, &e))
            return false;
         memcpy(out, e.bytes, e.length);
         *len = e.length;
         return true;
      }
      case AGX3_SEL_TT_XOR: {
         struct agx3_encoded_bit_xor_u32_reg_reg e;
         if (!agx3_pack_bit_xor_u32_reg_reg(encoding, I->dest, I->src[0].reg,
                                            I->src[1].reg, &e))
            return false;
         memcpy(out, e.bytes, e.length);
         *len = e.length;
         return true;
      }
      default:
         return false;
      }
   }

   case AGX3_SEL_OP_UNSUPPORTED:
   default:
      return false;
   }
}
