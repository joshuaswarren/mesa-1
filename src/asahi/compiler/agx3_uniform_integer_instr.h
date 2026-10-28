/* SPDX-License-Identifier: MIT */
#pragma once
#include "agx3_uniform_integer.h"
#include "agx_compiler.h"
#include "util/u_debug.h"

/* Shared pre/post-RA domain check: preserve only encodable uniform operands. */
static inline bool
agx3_uniform_integer_instruction_limit(const agx_instr *I, bool post_ra,
                                       unsigned gprs,
                                       struct agx3_uniform_integer_desc *d)
{
   if ((I->op != AGX_OPCODE_IADD && I->op != AGX_OPCODE_IMAD &&
        I->op != AGX_OPCODE_BITOP) ||
       I->nr_dests != 1 || I->saturate || I->dest[0].memory ||
       agx_channels(I->dest[0]) != 1 ||
       (I->dest[0].size != AGX_SIZE_16 && I->dest[0].size != AGX_SIZE_32) ||
       I->dest[0].type != (post_ra ? AGX_INDEX_REGISTER : AGX_INDEX_NORMAL))
      return false;

   if (post_ra &&
       (I->dest[0].value >= gprs ||
        (I->dest[0].size == AGX_SIZE_32 && (I->dest[0].value & 1))))
      return false;

   const unsigned count = I->op == AGX_OPCODE_IMAD ? 3 : 2;
   if (I->nr_srcs != count)
      return false;
   *d = (struct agx3_uniform_integer_desc){
      .half = I->dest[0].size == AGX_SIZE_16,
      .dest_cache = post_ra && I->dest[0].cache,
      .dest = (uint16_t)(post_ra ? I->dest[0].value : 0),
   };
   unsigned order[3] = {0, 1, 2};
   if (I->op == AGX_OPCODE_IADD) {
      if (I->src[0].neg || (I->src[1].neg && I->shift) ||
          (I->shift != 0 && I->shift != 3))
         return false;
      d->op = I->src[1].neg ? AGX3_UI_SUB : AGX3_UI_ADD;
      d->shift = I->shift;
   } else if (I->op == AGX_OPCODE_IMAD) {
      if (I->shift)
         return false;
      d->op = AGX3_UI_MAD;
   } else {
      switch (I->truth_table) {
      case 0x8:
         d->op = AGX3_UI_AND;
         break;
      case 0xe:
         d->op = AGX3_UI_OR;
         break;
      case 0x6:
         d->op = AGX3_UI_XOR;
         break;
      case 0x7:
         d->op = AGX3_UI_NAND;
         break;
      case 0x1:
         d->op = AGX3_UI_NOR;
         break;
      case 0x2:
         d->op = AGX3_UI_ANDC;
         break;
      case 0x4:
         d->op = AGX3_UI_ANDC;
         order[0] = 1;
         order[1] = 0;
         break;
      default:
         return false;
      }
   }

   for (unsigned i = 0; i < count; ++i) {
      const unsigned s = order[i];
      const agx_index src = I->src[s];
      const bool zero = I->op == AGX_OPCODE_IMAD && i == 2 &&
                        src.type == AGX_INDEX_IMMEDIATE && src.value == 0;
      if (src.memory || src.abs ||
          (src.neg && !(I->op == AGX_OPCODE_IADD && s == 1)) ||
          (post_ra && src.kill) || (src.cache && src.discard) ||
          agx_channels(src) != 1 || (!zero && src.size != I->dest[0].size))
         return false;
      if (zero) {
         d->src[i].kind = AGX3_UI_ZERO;
      } else if (src.type == AGX_INDEX_UNIFORM) {
         if (src.value >= 512 || (!d->half && (src.value & 1)) || src.cache ||
             src.discard)
            return false;
         d->src[i].kind = AGX3_UI_UNIFORM;
         d->src[i].value = src.value;
      } else if (src.type ==
                 (post_ra ? AGX_INDEX_REGISTER : AGX_INDEX_NORMAL)) {
         if (post_ra && (src.value >= gprs ||
                         (!d->half && (src.value & 1))))
            return false;
         d->src[i].kind = AGX3_UI_REGISTER;
         d->src[i].value = post_ra ? src.value : 0;
         d->src[i].cache = post_ra && src.cache;
         d->src[i].discard = post_ra && src.discard;
      } else {
         return false;
      }
   }
   const bool uniform_first =
      d->src[0].kind == AGX3_UI_UNIFORM && d->src[1].kind == AGX3_UI_REGISTER;
   const bool uniform_second =
      d->src[0].kind == AGX3_UI_REGISTER && d->src[1].kind == AGX3_UI_UNIFORM;
   if (!uniform_first && !uniform_second)
      return false;
   if ((d->shift || d->op == AGX3_UI_ANDC) && !uniform_first)
      return false;
   return true;
}

static inline bool
agx3_uniform_integer_instruction(const agx_instr *I, bool post_ra,
                                 struct agx3_uniform_integer_desc *d)
{
   return agx3_uniform_integer_instruction_limit(
      I, post_ra, AGX3_UNIFORM_INTEGER_GPRS, d);
}

/* M5 full-width uniform integer domains: ADD/SUB and native IMADD R/U/U
 * (cs_loop, ld_constant) or U/R/R (sr_ids). Multiplication commutes. */
static inline bool
agx3_uniform_integer_for_encoding(enum agx_isa_encoding enc,
                                  const agx_instr *I, bool post_ra,
                                  struct agx3_uniform_integer_desc *d)
{
   if (enc == AGX_ISA_ENCODING_AGX3B)
      return agx3_uniform_integer_instruction(I, post_ra, d);
   if (enc != AGX_ISA_ENCODING_AGX3C ||
       !agx3_uniform_integer_instruction_limit(
          I, post_ra, AGX3_UNIFORM_INTEGER_M5_GPRS, d) ||
       d->half || d->shift ||
       (d->op != AGX3_UI_ADD && d->op != AGX3_UI_SUB &&
        d->op != AGX3_UI_MAD))
      return false;
   if (d->op == AGX3_UI_MAD)
      return d->src[2].kind == AGX3_UI_REGISTER ||
             d->src[2].kind == AGX3_UI_UNIFORM;
   return d->op != AGX3_UI_SUB ||
          (d->src[0].kind == AGX3_UI_REGISTER &&
           d->src[1].kind == AGX3_UI_UNIFORM);
}
