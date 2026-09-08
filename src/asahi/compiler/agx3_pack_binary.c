/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * AGX3 binary emission: the thin shim between the shared Mesa pipeline and the
 * AGX3 instruction-selection core.
 *
 * The entire front end -- NIR lowering, optimization, register allocation and
 * scheduling -- is encoding-independent and reused verbatim from AGX2. Only the
 * final pack stage forks: where agx_pack_binary emits AGX2, this walks the same
 * scheduled agx_instr IR, normalizes each instruction into agx3_select_core's
 * operand struct, and emits AGX3 bytes.
 *
 * Everything that could be wrong -- the op-to-form mapping, the grounded domain
 * of each form, how operands bind -- lives in agx3_select_core and is tested
 * byte-for-byte in the host suite. This file is the translation and the block
 * walk, nothing more.
 *
 * The production subset includes grounded ALU, device-memory, special-register
 * and structured-control-flow forms, plus the graphics path: varying
 * interpolation and UVS stores for vertex shaders, and coefficient loads, tile
 * load/store and the pixel writeout barriers for fragment shaders. It fails the
 * shader -- never aborts, since this runs inside the UMD -- on any unmapped op
 * or out-of-domain instruction. In particular, a compare after an asynchronous
 * producer is refused until the compare/cnd hazard fields are grounded;
 * emitting the witnessed all-clear form there could consume stale data.
 */

#include "agx3_alu_core.h"
#include "agx3_atomic_core.h"
#include "agx3_binder.h"
#include "agx3_cf_core.h"
#include "agx3_copy_core.h"
#include "agx3_falu_mod_core.h"
#include "agx3_fltnorm_core.h"
#include "agx3_funop_core.h"
#include "agx3_iadd_core.h"
#include "agx3_isel_core.h"
#include "agx3_local_tps_core.h"
#include "agx3_mem_core.h"
#include "agx3_pack.h"
#include "agx3_pixel_core.h"
#include "agx3_scs_core.h"
#include "agx3_select_core.h"
#include "agx3_slot_core.h"
#include "agx3_sr_core.h"
#include "agx3_zs_core.h"
#include "agx3_stage_core.h"
#include "agx3_static_spill_core.h"
#include "agx3_subgroup_core.h"
#include "agx3_tex_core.h"
#include "agx3_tile_core.h"
#include "agx3_vary_core.h"
#include "agx3_writeout_core.h"
#include "agx_compiler.h"
#include "agx_debug.h"
#include "util/half_float.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

struct agx3_branch_fixup {
   agx_block *block;
   off_t offset;
   bool skip_to_end;
   bool is_bn;
};

/*
 * This backend is fail-closed per instruction: a shader containing anything
 * outside the grounded domain is refused, never mis-encoded. A bare refusal is
 * hard to act on, so name the instruction that closed the door under the
 * compiler's existing shader-debug flag (AGX_MESA_DEBUG=shaders). This is the
 * difference between "AGX3 rejected this shader" and a usable bug report.
 */
static const char *
agx3_index_kind(const agx_index x)
{
   switch (x.type) {
   case AGX_INDEX_NULL:      return "null";
   case AGX_INDEX_NORMAL:    return "ssa";
   case AGX_INDEX_IMMEDIATE: return "imm";
   case AGX_INDEX_UNIFORM:   return "uni";
   case AGX_INDEX_REGISTER:  return "reg";
   default:                  return "?";
   }
}

/*
 * Name the refused instruction, not just its opcode.
 *
 * A refusal is fatal for the whole shader, and the only thing that makes one
 * actionable is the operand shape: which sources are registers, immediates or
 * uniforms, what width each one is, and which modifiers are set. Reporting
 * only the mnemonic costs a round trip to hardware every time -- "no grounded
 * AGX3 encoding for fcmpsel" is true of the shapes that encode as well.
 */
static void
agx3_report_refusal(const agx_instr *I, unsigned line)
{
   if (!(agx_compiler_debug & AGX_DBG_SHADERS))
      return;

   char shape[512];
   int n = snprintf(shape, sizeof(shape), "d%u:%s%u",
                    I->nr_dests ? 16u << I->dest[0].size : 0u,
                    I->nr_dests ? agx3_index_kind(I->dest[0]) : "none",
                    I->nr_dests ? I->dest[0].value : 0u);

   for (unsigned i = 0; i < I->nr_srcs && n > 0 && n < (int)sizeof(shape); ++i) {
      const agx_index x = I->src[i];
      n += snprintf(shape + n, sizeof(shape) - n, " s%u=%u:%s%u%s%s%s%s%s", i,
                    16u << x.size, agx3_index_kind(x), x.value,
                    x.abs ? ".abs" : "", x.neg ? ".neg" : "",
                    x.kill ? ".kill" : "", x.cache ? ".cache" : "",
                    x.discard ? ".discard" : "");
   }

   fprintf(stderr,
           "agx3_pack_binary.c:%u: no grounded AGX3 encoding for %s "
           "[%s%s%s cond=%u chan=%u]\n",
           line, agx_opcodes_info[I->op].name, shape,
           I->saturate ? " sat" : "", I->invert_cond ? " inv" : "",
           (unsigned)I->fcond, I->nr_dests ? agx_channels(I->dest[0]) : 0u);
}

#define AGX3_REFUSE(I)                                                         \
   do {                                                                        \
      agx3_report_refusal((I), __LINE__);                                      \
      goto fail;                                                               \
   } while (0)

static bool
agx3_append(struct util_dynarray *emission, const uint8_t *bytes, uint8_t len)
{
   memcpy(util_dynarray_grow_bytes(emission, 1, len), bytes, len);
   return true;
}

static enum agx3_sel_op
agx3_map_opcode(unsigned op)
{
   switch (op) {
   case AGX_OPCODE_STOP:
      return AGX3_SEL_OP_END;
   case AGX_OPCODE_THREADGROUP_BARRIER:
      return AGX3_SEL_OP_BARRIER;
   case AGX_OPCODE_MOV_IMM:
      return AGX3_SEL_OP_MOV_IMM;
   case AGX_OPCODE_FADD:
      return AGX3_SEL_OP_FADD;
   case AGX_OPCODE_FMUL:
      return AGX3_SEL_OP_FMUL;
   case AGX_OPCODE_FFMA:
      return AGX3_SEL_OP_FFMA;
   case AGX_OPCODE_IADD:
      return AGX3_SEL_OP_IADD;
   case AGX_OPCODE_IMAD:
      return AGX3_SEL_OP_IMAD;
   case AGX_OPCODE_ASR:
      return AGX3_SEL_OP_ASR;
   case AGX_OPCODE_BITOP:
      return AGX3_SEL_OP_BITOP;
   case AGX_OPCODE_BFI:
      return AGX3_SEL_OP_SLL;
   case AGX_OPCODE_BFEIL:
      return AGX3_SEL_OP_LSR;
   case AGX_OPCODE_RCP:
      return AGX3_SEL_OP_F_RCP;
   case AGX_OPCODE_RSQRT:
      return AGX3_SEL_OP_F_RSQRT;
   case AGX_OPCODE_ROUNDEVEN:
      return AGX3_SEL_OP_F_RINT;
   default:
      return AGX3_SEL_OP_UNSUPPORTED;
   }
}

/* A source that is the immediate constant zero -- how Mesa spells the base of a
 * bitfield insert that is really a shift. */
static bool
agx3_src_is_zero_imm(agx_index s)
{
   return s.type == AGX_INDEX_IMMEDIATE && s.value == 0;
}

/* A source the no-modifier packers can consume is a plain register: not an
 * immediate, uniform, or memory operand. Milestone one refuses the rest, which
 * a pre-RA legalize pass will later materialize. */
static bool
agx3_src_is_register(agx_index s)
{
   return !s.memory &&
          (s.type == AGX_INDEX_REGISTER || s.type == AGX_INDEX_NORMAL);
}

/* Fill the destination and a single value source into the normalized form,
 * used by the ops the generic path cannot describe (convert). */
static void
agx3_set_dest_and_value(const agx_instr *I, agx_index value,
                        struct agx3_sel_instr *S)
{
   if (I->nr_dests >= 1 && I->dest[0].type != AGX_INDEX_NULL) {
      S->has_dest = true;
      S->dest = I->dest[0].value;
      S->dest_size = I->dest[0].size;
   }
   S->nr_srcs = 1;
   S->src[0].reg = value.value;
   S->src[0].size = value.size;
   S->src[0].is_reg = agx3_src_is_register(value);
}

static bool
agx3_translate_instr(const agx_instr *I, struct agx3_sel_instr *S)
{
   memset(S, 0, sizeof(*S));

   /* convert carries its direction as an immediate mode in the first source
    * and its rounding in `round`; the actual value is the second source. Only
    * the four 32-bit language casts NIR emits (f2i/f2u round-to-zero, i2f/u2f
    * round-to-nearest-even) are grounded. */
   if (I->op == AGX_OPCODE_CONVERT) {
      if (I->nr_srcs != 2 || I->src[0].type != AGX_INDEX_IMMEDIATE)
         return false;

      const unsigned mode = I->src[0].value;
      const enum agx_round round = I->round;

      if (mode == AGX_CONVERT_F_TO_S32 && round == AGX_ROUND_RTZ)
         S->op = AGX3_SEL_OP_F_TO_I32;
      else if (mode == AGX_CONVERT_F_TO_U32 && round == AGX_ROUND_RTZ)
         S->op = AGX3_SEL_OP_F_TO_U32;
      else if (mode == AGX_CONVERT_S32_TO_F && round == AGX_ROUND_RTE)
         S->op = AGX3_SEL_OP_I32_TO_F;
      else if (mode == AGX_CONVERT_U32_TO_F && round == AGX_ROUND_RTE)
         S->op = AGX3_SEL_OP_U32_TO_F;
      else
         return false;

      agx3_set_dest_and_value(I, I->src[1], S);
      return true;
   }

   /* fmin/fmax lower to fcmpsel(a, b, a, b, cond) -- a compare-select whose two
    * selected values are the two compared values, tied. cond LTN is a minimum,
    * GTN a maximum. Any other fcmpsel (a general ?: select) has no grounded
    * AGX3 mapping and is refused. */
   if (I->op == AGX_OPCODE_FCMPSEL) {
      if (I->nr_srcs != 4 || I->invert_cond ||
          !agx3_src_is_register(I->src[0]) ||
          !agx3_src_is_register(I->src[1]) || I->src[0].abs || I->src[0].neg ||
          I->src[1].abs || I->src[1].neg ||
          I->src[0].value != I->src[2].value ||
          I->src[1].value != I->src[3].value)
         return false;

      if (I->fcond == AGX_FCOND_LTN)
         S->op = AGX3_SEL_OP_FMIN;
      else if (I->fcond == AGX_FCOND_GTN)
         S->op = AGX3_SEL_OP_FMAX;
      else
         return false;

      if (I->nr_dests >= 1 && I->dest[0].type != AGX_INDEX_NULL) {
         S->has_dest = true;
         S->dest = I->dest[0].value;
         S->dest_size = I->dest[0].size;
      }
      S->nr_srcs = 2;
      S->src[0].reg = I->src[0].value;
      S->src[0].size = I->src[0].size;
      S->src[0].is_reg = true;
      S->src[1].reg = I->src[1].value;
      S->src[1].size = I->src[1].size;
      S->src[1].is_reg = true;
      return true;
   }

   S->op = agx3_map_opcode(I->op);
   if (S->op == AGX3_SEL_OP_UNSUPPORTED)
      return false;

   if (I->nr_srcs > 3)
      return false;

   S->saturate = I->saturate;
   S->shift = I->shift;
   S->imm = (uint32_t)I->imm;
   S->truth_table = I->truth_table;

   if (I->nr_dests >= 1 && I->dest[0].type != AGX_INDEX_NULL) {
      S->has_dest = true;
      S->dest = I->dest[0].value;
      S->dest_size = I->dest[0].size;
   }

   S->nr_srcs = I->nr_srcs;
   for (unsigned i = 0; i < I->nr_srcs; ++i) {
      S->src[i].reg = I->src[i].value;
      S->src[i].size = I->src[i].size;
      S->src[i].abs = I->src[i].abs;
      S->src[i].neg = I->src[i].neg;
      S->src[i].is_reg = agx3_src_is_register(I->src[i]);
   }

   /* NIR ishl/ushr arrive as a bitfield insert/extract with a zero base and a
    * zero insert mask: bfi(zero, value, amount) and bfeil(zero, value,
    * amount). That exact shape is a shift; anything else with these opcodes is
    * a genuine bitfield op with no grounded AGX3 mapping, so refuse it. Rewrite
    * the shift into the core's two-source (value, amount) form. */
   if (S->op == AGX3_SEL_OP_SLL || S->op == AGX3_SEL_OP_LSR) {
      if (I->nr_srcs != 3 || !agx3_src_is_zero_imm(I->src[0]) || I->imm != 0)
         return false;

      S->src[0] = S->src[1]; /* value  */
      S->src[1] = S->src[2]; /* amount */
      S->nr_srcs = 2;
      S->src[2] = (struct agx3_sel_src){0};
   }

   /* NIR isub is iadd(a, -b): an iadd whose second source is negated and whose
    * first is not. Map it to the grounded isub and drop the neg. A negate
    * (iadd(0, -a)) has an immediate-zero first source and is left unmapped. */
   if (S->op == AGX3_SEL_OP_IADD && I->nr_srcs == 2 && I->src[1].neg &&
       !I->src[0].neg && agx3_src_is_register(I->src[0]) &&
       agx3_src_is_register(I->src[1]) && I->shift == 0) {
      S->op = AGX3_SEL_OP_ISUB;
      S->src[1].neg = false;
   }

   /* NIR imul is imad(a, b, 0): a multiply-add with a zero addend. Map it to
    * the grounded imul. */
   if (S->op == AGX3_SEL_OP_IMAD && I->nr_srcs == 3 &&
       agx3_src_is_zero_imm(I->src[2]) && I->shift == 0) {
      S->op = AGX3_SEL_OP_IMUL;
      S->nr_srcs = 2;
      S->src[2] = (struct agx3_sel_src){0};
   }

   return true;
}

static bool
agx3_emit(struct util_dynarray *emission, enum agx_isa_encoding enc,
          const struct agx3_sel_instr *S)
{
   uint8_t bytes[AGX3_SEL_MAX_LENGTH];
   uint8_t len = 0;

   if (!agx3_select_encode(enc, S, bytes, &len))
      return false;

   memcpy(util_dynarray_grow_bytes(emission, 1, len), bytes, len);
   return true;
}

/*
 * Every source of a scalar half select is a plain, unmodified half register.
 * This is a shape test, not an encodability verdict: an instruction that fails
 * it belongs to one of the later fcmpsel forms (byte constants, minifloats,
 * 32-bit compares), so the caller must skip its handler rather than refuse the
 * shader.
 */
static bool
agx3_fcmpsel_all_half_regs(const agx_instr *I)
{
   for (unsigned i = 0; i < 4; ++i) {
      const agx_index x = I->src[i];
      if (!agx3_src_is_register(x) || x.memory || x.size != AGX_SIZE_16 ||
          agx_channels(x) != 1 || x.abs || x.neg || x.kill ||
          (x.cache && x.discard))
         return false;
   }
   return true;
}

/*
 * The five comparison codes witnessed in Apple's own selects. f110 is
 * literally `enum agx_fcond`; eq, lt, gt, ge and le were each swept from a
 * kernel differing only in the operator. The NaN-ignoring ltn/gtn codes carry
 * the tied minimum/maximum forms instead and are not witnessed here.
 */
static bool
agx3_fcond_witnessed_select(enum agx_fcond cond)
{
   return cond == AGX_FCOND_EQ || cond == AGX_FCOND_LT ||
          cond == AGX_FCOND_GT || cond == AGX_FCOND_GE ||
          cond == AGX_FCOND_LE;
}

/* A plain, unmodified scalar half register. */
static bool
agx3_half_reg(agx_index s)
{
   return agx3_src_is_register(s) && !s.memory && s.size == AGX_SIZE_16 &&
          agx_channels(s) == 1;
}

/* Reverse a comparison so the operands can be exchanged. */
static enum agx_fcond
agx3_fcond_reverse(enum agx_fcond cond)
{
   switch (cond) {
   case AGX_FCOND_LT: return AGX_FCOND_GT;
   case AGX_FCOND_GT: return AGX_FCOND_LT;
   case AGX_FCOND_LE: return AGX_FCOND_GE;
   case AGX_FCOND_GE: return AGX_FCOND_LE;
   default:           return cond;
   }
}

/* A source usable by the modifier long forms: a plain 32-bit register. */
static bool
agx3_mod_src_ok(agx_index s)
{
   return agx3_src_is_register(s) && !s.memory && s.size == AGX_SIZE_32 &&
          (s.value & 1) == 0 && s.value < 256;
}

/*
 * Route a modifier-carrying float add/mul/fma through the long-form modifier
 * encoder. Returns 1 if it emitted the instruction, 0 if this is not a
 * modifier-bearing float op (the caller should use the ordinary path), or -1
 * if it is one but out of the grounded domain (fail the shader).
 */
static int
agx3_try_float_modifier(struct util_dynarray *emission,
                        enum agx_isa_encoding enc, const agx_instr *I, unsigned wait)
{
   const bool is_add = I->op == AGX_OPCODE_FADD;
   const bool is_mul = I->op == AGX_OPCODE_FMUL;
   const bool is_fma = I->op == AGX_OPCODE_FFMA;
   if (!is_add && !is_mul && !is_fma)
      return 0;

   bool has_mod = I->saturate;
   for (unsigned s = 0; s < I->nr_srcs; ++s)
      has_mod |= I->src[s].abs || I->src[s].neg;
   if (!has_mod)
      return 0; /* the compact no-modifier packer handles it */

   const unsigned want = is_fma ? 3 : 2;
   if (I->nr_dests < 1 || I->dest[0].type == AGX_INDEX_NULL ||
       I->dest[0].size != AGX_SIZE_32 || (I->dest[0].value & 1) != 0 ||
       I->dest[0].value >= 256 || I->nr_srcs != want)
      return -1;
   for (unsigned s = 0; s < want; ++s)
      if (!agx3_mod_src_ok(I->src[s]))
         return -1;

   struct agx3_falu_mods m = {
      .src0_abs = I->src[0].abs,
      .src0_neg = I->src[0].neg,
      .src1_abs = I->src[1].abs,
      .src1_neg = I->src[1].neg,
      .saturate = I->saturate,
   };

   uint8_t bytes[AGX3_FALU_MOD_MAX_LENGTH];
   uint8_t len = 0;
   bool ok;
   const uint16_t d = I->dest[0].value, s0 = I->src[0].value,
                  s1 = I->src[1].value;
   if ((is_add || is_mul) && enc == AGX_ISA_ENCODING_AGX3B) {
      struct agx3_falu_runtime runtime = {
         .dest_cache = I->dest[0].cache,
         .wait_mask = wait,
      };
      for (unsigned i = 0; i < 2; ++i) {
         if (I->src[i].kill)
            return -1;
         runtime.cache[i] = I->src[i].cache;
         runtime.discard[i] = I->src[i].discard;
      }
      ok = is_add
         ? agx3_falu_add_mod_runtime(enc, d, s0, s1, &m, &runtime, bytes, &len)
         : agx3_falu_mul_mod_runtime(enc, d, s0, s1, &m, &runtime, bytes, &len);
   } else if (is_add)
      ok = agx3_falu_add_mod(enc, d, s0, s1, &m, bytes, &len);
   else if (is_mul)
      ok = agx3_falu_mul_mod(enc, d, s0, s1, &m, bytes, &len);
   else {
      m.addend_abs = I->src[2].abs;
      m.addend_neg = I->src[2].neg;
      if (enc == AGX_ISA_ENCODING_AGX3B) {
         struct agx3_falu_runtime runtime = {
            .dest_cache = I->dest[0].cache,
            .wait_mask = wait,
         };
         for (unsigned i = 0; i < 3; ++i) {
            if (I->src[i].kill)
               return -1;
            runtime.cache[i] = I->src[i].cache;
            runtime.discard[i] = I->src[i].discard;
         }
         ok = agx3_falu_fma_mod_runtime(enc, d, s0, s1, I->src[2].value,
                                       &m, &runtime, bytes, &len);
      } else {
         ok = agx3_falu_fma_mod(enc, d, s0, s1, I->src[2].value,
                               &m, bytes, &len);
      }
   }
   if (!ok || (wait && enc != AGX_ISA_ENCODING_AGX3B))
      return -1;

   memcpy(util_dynarray_grow_bytes(emission, 1, len), bytes, len);
   return 1;
}

/*
 * The transcendental / rounding / gradient funops with a corpus-grounded
 * AGX3 form. roundeven keeps its existing select-core route; sin_pt_1 and
 * sin_pt_2 are deliberately absent: AGX3 has no counterpart of the sin_pt_1
 * quadrant fixup at all (Apple computes it with an fma/isel polynomial), and
 * although f32sinc's *encoding* is grounded (agx3_funop_core emits it and
 * replays Apple's bytes), whether its input contract equals AGX2 sin_pt_2's
 * is not decidable offline -- Apple folds a phase constant into the preceding
 * polynomial. Routing sin_pt_2 onto f32sinc would be a semantic guess, so
 * fsin_agx stays fail-closed until an executed-numerics witness exists.
 */
static int
agx3_map_funop(unsigned op)
{
   switch (op) {
   case AGX_OPCODE_FLOOR:
      return AGX3_FUNOP_FLOOR;
   case AGX_OPCODE_CEIL:
      return AGX3_FUNOP_CEIL;
   case AGX_OPCODE_TRUNC:
      return AGX3_FUNOP_TRUNC;
   case AGX_OPCODE_LOG2:
      return AGX3_FUNOP_LOG2;
   case AGX_OPCODE_EXP2:
      return AGX3_FUNOP_EXP2;
   case AGX_OPCODE_DFDX:
      return AGX3_FUNOP_DFDX;
   case AGX_OPCODE_DFDY:
      return AGX3_FUNOP_DFDY;
   default:
      return -1;
   }
}

static bool
agx3_emit_funop(struct util_dynarray *emission, enum agx_isa_encoding enc,
                const agx_instr *I, enum agx3_funop op, uint16_t wait_mask)
{
   if (I->nr_dests != 1 || I->nr_srcs != 1 ||
       I->dest[0].type == AGX_INDEX_NULL ||
       !agx3_src_is_register(I->dest[0]) || I->dest[0].memory ||
       I->dest[0].size != AGX_SIZE_32 || !agx3_src_is_register(I->src[0]) ||
       I->src[0].memory || I->src[0].size != AGX_SIZE_32)
      return false;

   uint8_t bytes[AGX3_FUNOP_MAX_LENGTH];
   uint8_t len = 0;
   return agx3_funop_encode(enc, op, I->dest[0].value, I->src[0].value,
                            I->src[0].abs, I->src[0].neg, I->saturate,
                            wait_mask, bytes, &len) &&
          agx3_append(emission, bytes, len);
}

/* An async producer whose result later consumers must wait on. */
static bool
agx3_is_async(const agx_instr *I)
{
   if ((I->op == AGX_OPCODE_ATOMIC || I->op == AGX_OPCODE_LOCAL_ATOMIC) &&
       I->nr_dests == 1 && !agx_is_null(I->dest[0]))
      return true;

   /* The compound depth write publishes its staging word and consumes that
    * scoreboard generation with depthf. It needs an assigned slot even
    * though it has no general-register destination. */
   if (I->op == AGX_OPCODE_ZS_EMIT_AGX3)
      return true;

   return I->op == AGX_OPCODE_DEVICE_LOAD || I->op == AGX_OPCODE_LOCAL_LOAD ||
          I->op == AGX_OPCODE_LDCF || I->op == AGX_OPCODE_GET_SR ||
          I->op == AGX_OPCODE_GET_SR_COVERAGE ||
          I->op == AGX_OPCODE_GET_COVERAGE_1X_AGX3 ||
          I->op == AGX_OPCODE_TEXTURE_LOAD || I->op == AGX_OPCODE_IMAGE_LOAD ||
          I->op == AGX_OPCODE_TEXTURE_SAMPLE || I->op == AGX_OPCODE_LD_TILE ||
          I->op == AGX_OPCODE_SCS_LOAD_AGX3 ||
          I->op == AGX_OPCODE_SCS_LOAD_UNIFORM_INDEXED_AGX3 ||
          I->op == AGX_OPCODE_STACK_LOAD ||
          I->op == AGX_OPCODE_STACK_STORE;
}

/* Map a Mesa special register to the corpus-witnessed Apple SR number and
 * destination width. Only registers tied to a source-level builtin by an
 * isolating probe are mapped; everything else -- including any (SR, width)
 * combination Apple never emitted -- fails closed so the shader is honestly
 * rejected, not miscompiled. */
static bool
agx3_map_sr(enum agx_sr sr, uint8_t *apple, enum agx3_sr_width *width)
{
#define SR(mesa, num, w)                                                       \
   case mesa:                                                                  \
      *apple = (num);                                                          \
      *width = (w);                                                            \
      return true
   switch (sr) {
      SR(AGX_SR_THREADGROUP_POSITION_IN_GRID_X, 26, AGX3_SR_W32);
      SR(AGX_SR_THREADGROUP_POSITION_IN_GRID_Y, 27, AGX3_SR_W32);
      SR(AGX_SR_THREADGROUP_POSITION_IN_GRID_Z, 28, AGX3_SR_W32);
      SR(AGX_SR_THREADS_PER_THREADGROUP_X, 17, AGX3_SR_W16);
      SR(AGX_SR_THREADS_PER_THREADGROUP_Y, 18, AGX3_SR_W16);
      SR(AGX_SR_THREADS_PER_THREADGROUP_Z, 19, AGX3_SR_W16);
      SR(AGX_SR_DISPATCH_THREADS_PER_THREADGROUP_X, 20, AGX3_SR_W16);
      SR(AGX_SR_DISPATCH_THREADS_PER_THREADGROUP_Y, 21, AGX3_SR_W16);
      SR(AGX_SR_DISPATCH_THREADS_PER_THREADGROUP_Z, 22, AGX3_SR_W16);
      SR(AGX_SR_THREAD_POSITION_IN_THREADGROUP_X, 9, AGX3_SR_W16);
      SR(AGX_SR_THREAD_POSITION_IN_THREADGROUP_Y, 10, AGX3_SR_W16);
      SR(AGX_SR_THREAD_POSITION_IN_THREADGROUP_Z, 11, AGX3_SR_W16);
      SR(AGX_SR_THREAD_INDEX_IN_THREADGROUP, 13, AGX3_SR_W16);
      SR(AGX_SR_THREAD_INDEX_IN_SUBGROUP, 7, AGX3_SR_W16);
      SR(AGX_SR_SUBGROUP_INDEX_IN_THREADGROUP, 6, AGX3_SR_W16);
      SR(AGX_SR_THREAD_POSITION_IN_GRID_X, 23, AGX3_SR_W32);
      SR(AGX_SR_THREAD_POSITION_IN_GRID_Y, 24, AGX3_SR_W32);
      SR(AGX_SR_THREAD_POSITION_IN_GRID_Z, 25, AGX3_SR_W32);
      /* The semantic enum carries the hardware SR byte (0xd8/0xdd), while
       * agx3_sr_core consumes Apple's anonymous normalized f21 value. */
      SR(AGX_SR_INSTANCE_ID, 46, AGX3_SR_W32);
      SR(AGX_SR_VERTEX_ID, 51, AGX3_SR_W32);
   default:
      return false;
   }
#undef SR
}

/* Translate a get_sr into the normalized SR-read descriptor. The destination
 * size must match the witnessed width of the register: Apple reads 16-bit SRs
 * into a half register (consumers or an explicit zero-extend handle the rest),
 * and no witness reads one at another width. */
static bool
agx3_translate_sr(const agx_instr *I, uint8_t slot, uint16_t wait_mask,
                  struct agx3_sr_desc *d)
{
   memset(d, 0, sizeof(*d));

   if (I->nr_dests < 1 || I->dest[0].type == AGX_INDEX_NULL)
      return false;
   const agx_index dst = I->dest[0];
   if (!agx3_src_is_register(dst) || dst.memory || dst.value >= 256)
      return false;

   uint8_t apple;
   enum agx3_sr_width width;
   if (!agx3_map_sr(I->sr, &apple, &width))
      return false;
   if (width == AGX3_SR_W32 ? dst.size != AGX_SIZE_32 : dst.size != AGX_SIZE_16)
      return false;

   d->dest = dst.value;
   d->width = width;
   d->sr = apple;
   d->slot = slot;
   d->wait_mask = wait_mask;
   return true;
}

static bool
agx3_map_vary_location(enum agx_interpolation interpolation,
                        enum agx3_vary_location *location)
{
   switch (interpolation) {
   case AGX_INTERPOLATION_CENTER:
      *location = AGX3_VARY_CENTER;
      return true;
   case AGX_INTERPOLATION_CENTROID:
      *location = AGX3_VARY_CENTROID;
      return true;
   default:
      return false;
   }
}

static bool
agx3_vary_reg32(agx_index index)
{
   return agx3_src_is_register(index) && !index.memory &&
          index.size == AGX_SIZE_32 && (index.value & 1) == 0 &&
          index.value < 256 && !index.abs && !index.neg;
}

/* Projective RHW is private to reciprocal interpolation. Follow its physical
 * half-register lifetime through the CFG: unrelated branches do not change
 * the iterator encoding, and a full overwrite ends the old value's lifetime.
 * Track each half independently so an ordinary read after a partial overwrite
 * cannot accidentally be accepted as a read of an unrelated new value. */
static bool
agx3_private_rhw_use(const agx_context *ctx, const agx_block *block,
                     const agx_instr *producer)
{
   if (producer->nr_dests != 1 || producer->nr_srcs != 2 ||
       !agx3_vary_reg32(producer->dest[0]) ||
       agx_channels(producer->dest[0]) != 1 ||
       producer->src[0].type != AGX_INDEX_IMMEDIATE)
      return false;

   struct rhw_path {
      const agx_block *block;
      unsigned live_halves;
      const agx_instr *after;
   };
   struct util_dynarray paths;
   util_dynarray_init(&paths, NULL);
   uint8_t *visited = calloc(ctx->num_blocks, sizeof(*visited));
   if (!visited)
      return false;
   util_dynarray_append(&paths,
                       ((struct rhw_path){block, 3, producer}));
   const unsigned reg = producer->dest[0].value;
   bool consumer = false, valid = false;

   for (unsigned p = 0; p < util_dynarray_num_elements(&paths, struct rhw_path); ++p) {
      struct rhw_path path = *util_dynarray_element(&paths, struct rhw_path, p);
      bool after = path.after == NULL;
      unsigned live = path.live_halves;
      agx_foreach_instr_in_block(path.block, I) {
         if (!after) {
            after = I == path.after;
            continue;
         }
         agx_foreach_src(I, src) {
            agx_index value = I->src[src];
            if (!agx3_src_is_register(value) || value.memory)
               continue;
            unsigned overlap = 0;
            for (unsigned h = 0; h < 2; ++h) {
               if (value.value <= reg + h &&
                   value.value + agx_index_size_16(value) > reg + h)
                  overlap |= 1u << h;
            }
            if (!(overlap & live))
               continue;
            if (live != 3 || I->op != AGX_OPCODE_PERSP_RCP_AGX3 || src != 0 ||
                value.value != reg || value.size != AGX_SIZE_32 ||
                agx_channels(value) != 1 || I->imm != producer->src[0].value)
               goto out;
            consumer = true;
         }
         agx_foreach_dest(I, dst) {
            agx_index value = I->dest[dst];
            if (!agx3_src_is_register(value) || value.memory)
               continue;
            for (unsigned h = 0; h < 2; ++h) {
               if (value.value <= reg + h &&
                   value.value + agx_index_size_16(value) > reg + h)
                  live &= ~(1u << h);
            }
         }
         if (!live)
            break;
      }
      if (!live)
         continue;
      for (unsigned e = 0; e < ARRAY_SIZE(path.block->successors); ++e) {
         const agx_block *next = path.block->successors[e];
         if (!next)
            continue;
         assert(next->index < ctx->num_blocks);
         if (visited[next->index] & (1u << live))
            continue;
         visited[next->index] |= 1u << live;
         util_dynarray_append(&paths,
                             ((struct rhw_path){next, live, NULL}));
      }
   }
   valid = consumer;
out:
   free(visited);
   util_dynarray_fini(&paths);
   return valid;
}

static bool
agx3_emit_interp(struct util_dynarray *emission, enum agx_isa_encoding enc,
                 const agx_instr *I, uint16_t wait_mask, bool first, bool fragment)
{
   if (wait_mask != 0 || I->nr_dests != 1 || I->nr_srcs != 2 ||
       !agx3_vary_reg32(I->dest[0]) ||
       I->src[0].type != AGX_INDEX_IMMEDIATE ||
       I->src[1].type != AGX_INDEX_IMMEDIATE || I->src[1].value != 0)
      return false;

   enum agx3_vary_location location;
   if (!agx3_map_vary_location(I->interpolation, &location))
      return false;

   uint8_t bytes[AGX3_VARY_MAX_LENGTH];
   uint8_t len = 0;
   if (I->op == AGX_OPCODE_RHW_INTERP_AGX3) {
      if (I->saturate || location != AGX3_VARY_CENTER)
         return false;
      return agx3_vary_projective_rhw(enc, I->dest[0].value,
                                     I->src[0].value, first, bytes, &len) &&
             agx3_append(emission, bytes, len);
   }
   /* The native G17P center_no_perspective scalar/vector programs use
    * f11=1 as well. This bit does not select perspective correction: the
    * VIT shade model and explicit RCP/MUL sequence do that. Keep the raw
    * codec able to reproduce both native f11 forms. */
   const bool native_mode = I->perspective ||
      (fragment && enc == AGX_ISA_ENCODING_AGX3B &&
       location == AGX3_VARY_CENTER);
   return agx3_vary_interp_at_start(enc, I->dest[0].value, I->src[0].value,
                                    native_mode, location, first, bytes, &len) &&
          agx3_append(emission, bytes, len);
}

static bool
agx3_emit_ldcf(struct util_dynarray *emission, enum agx_isa_encoding enc,
               const agx_instr *I, uint8_t slot, uint16_t wait_mask, bool first)
{
   if (wait_mask != 0 || I->nr_dests != 1 || I->nr_srcs != 1 ||
       !agx3_vary_reg32(I->dest[0]) ||
       I->src[0].type != AGX_INDEX_IMMEDIATE || I->channels != 1)
      return false;

   uint8_t bytes[AGX3_VARY_MAX_LENGTH];
   uint8_t len = 0;
   return agx3_vary_ldcf_at_start(enc, I->dest[0].value, I->src[0].value,
                                 slot, first, bytes, &len) &&
          agx3_append(emission, bytes, len);
}

static bool
agx3_emit_uvsw(struct util_dynarray *emission, enum agx_isa_encoding enc,
               const agx_instr *I, uint16_t staging_half, uint8_t slot,
               uint16_t wait_mask, bool defer_writeout)
{
   if (I->nr_dests != 0 || I->nr_srcs != 2 ||
       I->src[0].type != AGX_INDEX_IMMEDIATE ||
       !agx3_vary_reg32(I->src[1]))
      return false;

   uint8_t bytes[MAX2(AGX3_STAGE_MAX_LENGTH, AGX3_VARY_MAX_LENGTH)];
   uint8_t len = 0;
   /* UVSW's f9=5/f5 operand is a staging-file address, not the GPR named
    * by ST_VARY IR. First transfer the source into that file and wait for
    * its actual producer; UVSW then waits for this staging write's slot. */
   if (!agx3_stage_copy_source(enc, staging_half, I->src[1].value,
                                I->src[1].kill, slot, wait_mask, bytes, &len) ||
       !agx3_append(emission, bytes, len))
      return false;
   return defer_writeout ||
          (agx3_vary_uvsw(enc, staging_half, I->src[0].value, slot, true,
                          (uint16_t)(1u << (slot - 1)), bytes, &len) &&
           agx3_append(emission, bytes, len));
}

static bool
agx3_tex_plain_reg(agx_index index)
{
   return agx3_src_is_register(index) && !index.memory && !index.abs &&
          !index.neg;
}

static bool
agx3_tex_imm(agx_index index, uint16_t value)
{
   return index.type == AGX_INDEX_IMMEDIATE && index.value == value &&
          !index.memory && !index.abs && !index.neg;
}

static bool
agx3_tex_plain_uniform64(agx_index index)
{
   return index.type == AGX_INDEX_UNIFORM && index.size == AGX_SIZE_64 &&
          agx_channels(index) == 1 && !(index.value & 1) &&
          index.value <= 252 && !index.memory && !index.abs && !index.neg &&
          !index.kill && !index.cache && !index.discard;
}

/* Translate the corpus-grounded 2D texel fetch. Dynamic integer LOD is
 * published into operand bank 6 on staging slot 2, then the fetch returns on
 * result slot 1 while waiting for that independent staging generation. Every
 * other texture axis remains a deliberate refusal. */
static bool
agx3_emit_texture_fetch(struct util_dynarray *emission,
                        enum agx_isa_encoding enc, const agx_instr *I,
                        uint8_t slot, uint16_t wait_mask, bool final,
                        bool stage_coordinates)
{
   const bool immediate = I->nr_srcs == 6 &&
      I->src[3].type == AGX_INDEX_IMMEDIATE;
   const bool immediate_lod = I->lod_mode == AGX_LOD_MODE_LOD_MIN &&
      I->src[1].type == AGX_INDEX_IMMEDIATE && !I->src[1].memory &&
      !I->src[1].abs && !I->src[1].neg && I->src[1].value <= 31;
   const bool register_lod =
      I->lod_mode == AGX_LOD_MODE_LOD_MIN &&
      agx3_tex_plain_reg(I->src[1]) && I->src[1].size == AGX_SIZE_32 &&
      agx_channels(I->src[1]) == 1;
   const bool ms_2d = I->dim == AGX_DIM_2D_MS;
   const bool ms_array = I->dim == AGX_DIM_2D_MS_ARRAY;
   const bool bindless_image = I->op == AGX_OPCODE_IMAGE_LOAD && !immediate &&
      agx3_tex_plain_reg(I->src[3]) && I->src[3].size == AGX_SIZE_64 &&
      agx_channels(I->src[3]) == 1 && !(I->src[3].value & 1) &&
      I->src[3].value <= 252 &&
      !(I->src[3].cache && (I->src[3].discard || I->src[3].kill)) &&
      agx3_tex_plain_uniform64(I->src[2]);
   /* Apple's lane-indexed image read publishes coordinates and uint LOD on
    * staging slot 2, then issues the texture result on slot 1. Keep those two
    * independent scoreboard domains; reusing slot 1 for both is not a native
    * dependency shape. */
   const uint8_t staging_slot =
      ms_array ? 2 : register_lod ? 2 : slot;
   if (I->nr_dests != 1 || I->nr_srcs != 6 ||
       !agx3_tex_plain_reg(I->dest[0]) ||
       (I->dest[0].size != AGX_SIZE_16 &&
        I->dest[0].size != AGX_SIZE_32) ||
       agx_channels(I->dest[0]) != 4 || I->dest[0].value >= 128 ||
       (I->dest[0].size == AGX_SIZE_32 && (I->dest[0].value & 1)) ||
       !agx3_tex_plain_reg(I->src[0]) ||
       I->src[0].size != AGX_SIZE_32 ||
       agx_channels(I->src[0]) !=
          ((I->dim == AGX_DIM_2D_ARRAY || ms_array || ms_2d) ? 3 : 2) ||
       (stage_coordinates ? ((I->src[0].value & 1) ||
         I->src[0].value + 2 * agx_channels(I->src[0]) > 256) :
         ((I->src[0].value & 3) || I->src[0].value > 60)) ||
       ((ms_array || ms_2d)
           ? !agx3_tex_imm(I->src[1], 0)
           : (!(I->lod_mode == AGX_LOD_MODE_AUTO_LOD &&
                agx3_tex_imm(I->src[1], 0)) &&
              !immediate_lod && !register_lod)) ||
       ((ms_array || ms_2d) && I->lod_mode != AGX_LOD_MODE_AUTO_LOD) ||
       (register_lod && !stage_coordinates) ||
       !(agx3_tex_imm(I->src[2], 0) ||
         (ms_array && agx3_tex_plain_uniform64(I->src[2])) ||
         bindless_image) ||
       (immediate ? I->src[3].value > 15 :
        (enc != AGX_ISA_ENCODING_AGX3B ||
         (I->src[3].type != AGX_INDEX_REGISTER && I->src[3].type != AGX_INDEX_UNIFORM) ||
         (I->src[3].size != AGX_SIZE_64 &&
          !(ms_array && I->src[3].size == AGX_SIZE_32)) ||
         agx_channels(I->src[3]) != 1 ||
         (I->src[3].value & 1) || I->src[3].value > 252)) ||
       I->src[3].memory || I->src[3].abs || I->src[3].neg ||
       !agx3_tex_imm(I->src[4], 0) ||
       !agx_is_null(I->src[5]) ||
       (I->dim != AGX_DIM_2D && I->dim != AGX_DIM_2D_ARRAY &&
        !ms_2d && !ms_array) ||
       (enc == AGX_ISA_ENCODING_AGX3B ? (!I->mask || (I->mask & ~15u))
                                    : (I->mask != 1 && I->mask != 15)) ||
       I->offset || I->shadow ||
       I->query_lod || I->gather != AGX_GATHER_NONE || I->sparse)
      return false;

   /* A read/write image stages its read coordinates at halves16/18, keeping
    * them disjoint from imgwci's halves0/2. Native read-only indexed images,
    * like ordinary texture loads, use the bank0 coordinate ABI. */
   const uint16_t staging_coord =
      (I->op == AGX_OPCODE_IMAGE_LOAD && !bindless_image) || ms_array ? 16 : 0;
   if (register_lod && slot != 1)
      return false;
   if (stage_coordinates) {
      uint16_t staging_wait = wait_mask;
      if (register_lod) {
         uint8_t lod[AGX3_TEX_MAX_LENGTH], lod_len = 0;
         if (!agx3_tex_stage_typed_lod(
                enc, I->src[1].value, staging_slot, staging_wait,
                I->src[1].cache, I->src[1].discard,
                AGX3_TEX_LEVEL_INTEGER, lod, &lod_len) ||
             !agx3_append(emission, lod, lod_len))
            return false;
         staging_wait = 0;
      }

      const unsigned plain_coords =
         (ms_2d || ms_array) ? 2 : agx_channels(I->src[0]);
      for (unsigned i = 0; i < plain_coords; i++) {
         uint8_t copy[AGX3_TEX_MAX_LENGTH], copy_len = 0;
         if (!agx3_tex_stage_copy_hints(enc, staging_coord + 2 * i,
                                  I->src[0].value + 2 * i,
                                  staging_slot, i == 0 ? staging_wait : 0,
                                  enc == AGX_ISA_ENCODING_AGX3B && I->src[0].cache,
                                  enc != AGX_ISA_ENCODING_AGX3B || I->src[0].discard,
                                  copy, &copy_len) ||
             !agx3_append(emission, copy, copy_len))
            return false;
      }

      /* The lowered MS-array coordinate packs sample in the low half and
       * layer in the high half of its third word. Native publishes that word
       * through the same PACK_SMPIR array operand as ordinary array fetches. */
      if (ms_array) {
         uint8_t packed[AGX3_TEX_MAX_LENGTH], packed_len = 0;
         if (!agx3_tex_stage_array_layer(
                enc, I->src[0].value + 4, staging_slot, 0,
                I->src[0].cache, I->src[0].discard,
                packed, &packed_len) ||
             !agx3_append(emission, packed, packed_len))
            return false;
      }

      /* For non-array MSAA, the third 32-bit coordinate is the sample index.
       * Publish it through native's dedicated integer PACK_SMPIR operand. */
      if (ms_2d) {
         uint8_t packed[AGX3_TEX_MAX_LENGTH], packed_len = 0;
         if (!agx3_tex_stage_ms_sample(
                enc, I->src[0].value + 4, 0,
                I->src[0].cache, I->src[0].discard,
                packed, &packed_len) ||
             !agx3_append(emission, packed, packed_len))
            return false;
      }
   }

   const struct agx3_tex_fetch_desc d = {
      .dest = I->dest[0].value,
      .coord = stage_coordinates ? staging_coord : I->src[0].value,
      .width = I->dest[0].size == AGX_SIZE_32 ? AGX3_TEX_W32 : AGX3_TEX_W16,
      .mask = I->mask,
      .texture = immediate ? I->src[3].value : 0,
      .texture_kind = immediate ? AGX3_TEX_OPERAND_IMMEDIATE :
         I->src[3].type == AGX_INDEX_UNIFORM ? AGX3_TEX_OPERAND_UNIFORM :
                                             AGX3_TEX_OPERAND_REGISTER,
      .texture_value = immediate ? 0 : I->src[3].value,
      .texture_kill = !immediate && I->src[3].type == AGX_INDEX_REGISTER &&
         (I->src[3].discard || I->src[3].kill),
      .slot = slot,
      .wait_mask = stage_coordinates ? (1u << (staging_slot - 1)) : wait_mask,
      .coord_kill = stage_coordinates || I->src[0].kill,
      .final = final,
      .array = I->dim == AGX_DIM_2D_ARRAY || ms_array,
      .multisample = ms_2d || ms_array,
      .lod_x64 = immediate_lod ? I->src[1].value * 64 : 0,
      .register_lod = register_lod,
   };
   uint8_t bytes[AGX3_TEX_MAX_LENGTH];
   uint8_t len = 0;
   return agx3_tex_encode_fetch(enc, &d, bytes, &len) &&
          agx3_append(emission, bytes, len);
}

/* Native G17P imgwci reads its integer coordinates through a dedicated pair
 * populated by this exact isel.ep35 form. Both selected values are the same,
 * so the select is a routing operation, not a source-level conditional. */
static bool
agx3_emit_image_coord(struct util_dynarray *emission,
                      enum agx_isa_encoding enc, uint16_t dest, uint16_t src)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || (dest & 1) || dest > 2 ||
       (src & 1) || src > 254)
      return false;

   const struct agx3_field_kv base[] = {
      {10, 5}, {11, 2}, {13, 2}, {42, 2}, {43, 2}, {46, 1},
      {51, 12}, {57, 3}, {58, 2}, {72, 2}, {73, 2}, {76, 1},
      {83, 12}, {89, 3}, {90, 2}, {106, 1},
   };
   const struct agx3_field_kv values[] = {
      {6, dest}, {35, src}, {65, src},
   };
   uint8_t bytes[14], len = 0;
   return agx3_bind_compose(enc, "isel", 35, base, ARRAY_SIZE(base), values,
                            ARRAY_SIZE(values), bytes, &len) &&
          len == sizeof(bytes) && agx3_append(emission, bytes, len);
}

/* Two bounded public storage-image forms: noncoherent 2D RGBA8UI state0 fed
 * by one image load, and normalized formatted-float state1 fed by one
 * explicit-LOD sample. The PBE descriptor selects RGBA8 versus RGB10A2.
 * The four slot2 COPYs route the packed vec4 into the exact imgwci data lanes.
 * The first COPY consumes the one-shot texture dependency when it is still
 * outstanding. An intervening bounds consumer may already have retired that
 * producer; either way native imgwci waits for the new staging-slot2 writes.
 * Validation happens before the first append, so a refusal leaves the output
 * transaction untouched. */
static bool
agx3_emit_storage_image_write(struct util_dynarray *emission,
                              enum agx_isa_encoding enc, const agx_instr *I,
                              uint16_t wait_mask)
{
   const bool normalized = I->format == AGX_FORMAT_U8NORM ||
                           I->format == AGX_FORMAT_RGB10A2;
   const bool immediate_state = agx3_tex_imm(I->src[3], 0) &&
      I->src[4].type == AGX_INDEX_IMMEDIATE &&
      !I->src[4].memory && !I->src[4].abs && !I->src[4].neg &&
      ((I->src[4].value == 0 && I->format == AGX_FORMAT_I8) ||
       (I->src[4].value == 1 && normalized));
   const bool register_state = normalized &&
      agx3_tex_plain_uniform64(I->src[3]) &&
      agx3_tex_plain_reg(I->src[4]) &&
      I->src[4].size == AGX_SIZE_32 && agx_channels(I->src[4]) == 1 &&
      !(I->src[4].value & 1) && I->src[4].value <= 126 &&
      !(I->src[4].cache && (I->src[4].discard || I->src[4].kill));
   if (enc != AGX_ISA_ENCODING_AGX3B || I->nr_dests != 0 ||
       I->nr_srcs != 5 ||
       I->dim != AGX_DIM_2D || I->coherent || (wait_mask & ~1u) != 0 ||
       !agx3_tex_plain_reg(I->src[0]) || I->src[0].size != AGX_SIZE_32 ||
       agx_channels(I->src[0]) != 4 || (I->src[0].value & 1) ||
       I->src[0].value > 248 || !agx3_tex_plain_reg(I->src[1]) ||
       I->src[1].size != AGX_SIZE_32 || agx_channels(I->src[1]) != 2 ||
       (I->src[1].value & 1) || I->src[1].value > 252 ||
       !agx3_tex_imm(I->src[2], 0) ||
       (!immediate_state && !register_state)) {
      if (agx_compiler_debug & AGX_DBG_SHADERS)
         fprintf(stderr, "G17P imgwci shape refusal: wait=0x%x data=%u/%u/%u coord=%u/%u/%u dim=%u coherent=%u\n",
                 wait_mask, I->src[0].value, I->src[0].size,
                 agx_channels(I->src[0]), I->src[1].value, I->src[1].size,
                 agx_channels(I->src[1]), I->dim, I->coherent);
      return false;
   }

   struct util_dynarray staged;
   util_dynarray_init(&staged, NULL);
   const uint16_t data_dest[4] = {4, 6, 12, 14};
   if (!agx3_emit_image_coord(&staged, enc, 0, I->src[1].value) ||
       !agx3_emit_image_coord(&staged, enc, 2, I->src[1].value + 2)) {
      if (agx_compiler_debug & AGX_DBG_SHADERS)
         fprintf(stderr, "G17P imgwci coordinate compose refusal\n");
      goto fail;
   }

   for (unsigned c = 0; c < 4; ++c) {
      uint8_t bytes[AGX3_STAGE_MAX_LENGTH], len = 0;
      if (!agx3_stage_copy_source(enc, data_dest[c],
                                  I->src[0].value + 2 * c,
                                  I->src[0].discard, 2,
                                  c == 0 ? wait_mask : 0,
                                  bytes, &len) ||
          !agx3_append(&staged, bytes, len)) {
         if (agx_compiler_debug & AGX_DBG_SHADERS)
            fprintf(stderr, "G17P imgwci data compose refusal: component=%u\n", c);
         goto fail;
      }
   }

   uint8_t imgwci[AGX3_IMAGE_WRITE_LENGTH], imgwci_len = 0;
   const struct agx3_image_write_desc d = {
      .wait_mask = wait_mask,
      .image_kind = register_state ? AGX3_TEX_OPERAND_REGISTER :
                                     AGX3_TEX_OPERAND_IMMEDIATE,
      .image_value = I->src[4].value,
      .image_kill = register_state &&
                    (I->src[4].discard || I->src[4].kill),
      .normalized = normalized,
   };
   if (!agx3_tex_encode_image_write_rgba8_2d(
          enc, &d, imgwci, &imgwci_len) ||
       !agx3_append(&staged, imgwci, imgwci_len) ||
       !agx3_append(emission, staged.data, staged.size)) {
      if (agx_compiler_debug & AGX_DBG_SHADERS)
         fprintf(stderr, "G17P imgwci instruction compose refusal\n");
      goto fail;
   }
   util_dynarray_fini(&staged);
   return true;

fail:
   util_dynarray_fini(&staged);
   return false;
}

/* Filtered 2D samples use scoreboard-signalling native staging producers.
 * Plain implicit 2D keeps its coordinate layout. Explicit per-lane LOD and
 * 2D-array samples stage coordinates into bank 0; their auxiliary scalar is
 * packed into bank 6. All input waits attach to the first producer; the sample
 * waits for their shared slot.
 */
static bool
agx3_emit_texture_sample(struct util_dynarray *emission,
                         enum agx_isa_encoding enc, const agx_instr *I,
                         uint8_t slot, uint16_t wait_mask, bool final)
{
   const bool explicit_lod = I->lod_mode == AGX_LOD_MODE_LOD_MIN;
   /* An implicit-LOD sample with a constant bias: the same 12-bit x64
    * immediate operand as an explicit level, selected by f139 = 1 instead
    * of 2, and keeping the implicit-LOD marker.
    *
    * A zero bias is not encoded at all: Apple's own compiler folds it into
    * the plain implicit form, and this packer follows it, so the operand
    * stays the immediate zero the implicit form already requires. Minifloat
    * -0.0 folds the same way. */
   const bool zero_bias = I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS &&
                          I->src[1].type == AGX_INDEX_IMMEDIATE &&
                          agx_minifloat_decode(I->src[1].value) == 0.0f;
   const bool biased_lod = I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS &&
                           I->src[1].type == AGX_INDEX_IMMEDIATE && !zero_bias;
   /* The same sample with a per-lane bias instead: PACK_SMPIR publishes the
    * scalar exactly as it does for a per-lane level, and only f139 moves
    * (3 rather than 4). */
   const bool register_bias = I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS &&
                              I->src[1].type != AGX_INDEX_IMMEDIATE;
   const bool implicit_lod = I->lod_mode == AGX_LOD_MODE_AUTO_LOD || zero_bias;
   /* Explicit gradients. The four scalars are staged into their own banks
    * and the sample names coordinate bank 0, as every witness does. */
   const bool gradient_lod = I->lod_mode == AGX_LOD_MODE_LOD_GRAD;
   /* textureGather. Every witness is a plain 2D implicit-LOD gather, so the
    * gather bit rides on the ordinary implicit sample and nothing else about
    * the instruction changes. The IR spells the components 1/3/5/7 in a
    * three-bit field, so map them one by one: an unspelled value must refuse
    * rather than alias onto a real channel. */
   const bool gather = I->gather != AGX_GATHER_NONE;
   const bool shadow = I->shadow;
   const bool offset = I->offset;
   enum agx3_tex_gather gather_component = AGX3_TEX_GATHER_NONE;
   switch (I->gather) {
   case AGX_GATHER_NONE: break;
   case AGX_GATHER_R: gather_component = AGX3_TEX_GATHER_R; break;
   case AGX_GATHER_G: gather_component = AGX3_TEX_GATHER_G; break;
   case AGX_GATHER_B: gather_component = AGX3_TEX_GATHER_B; break;
   case AGX_GATHER_A: gather_component = AGX3_TEX_GATHER_A; break;
   default: return false;
   }
   const bool immediate_lod =
      (explicit_lod || biased_lod) && I->src[1].type == AGX_INDEX_IMMEDIATE;
   const float immediate_level = immediate_lod ?
      agx_minifloat_decode(I->src[1].value) : 0.0f;
   const float immediate_level_x64 = immediate_level * 64.0f;
   /* f29 is a *signed* 12-bit LOD in 1/64 steps. Sweeping Apple's own
    * `t.sample(s, uv, bias(k))` over k = +-{1/64, 0.25, 0.5, 1, 1.5, 2, 4,
    * 31, 32, 33, 63, 64} gives f29 = k * 64 in two's complement -- bias(-1.0)
    * is 4032, bias(-32.0) is 2048 -- saturating at 2047 (+31.984375) and 2048
    * (-32). A minifloat immediate spans [-31, 31], so it always lands inside
    * that range, but the sign has to survive into the field. */
   const bool integral_x64 =
      immediate_level_x64 >= -2048.0f && immediate_level_x64 <= 2047.0f &&
      immediate_level_x64 == (float)(int16_t)immediate_level_x64;
   /* An explicit level is unsigned; only a bias may be negative. */
   const bool valid_immediate_lod =
      immediate_lod && immediate_level >= 0.0f && integral_x64;
   const bool valid_immediate_bias = immediate_lod && integral_x64;
   const uint16_t immediate_x64_field =
      (uint16_t)(((int)immediate_level_x64) & 0xfff);
   const bool array = I->dim == AGX_DIM_2D_ARRAY;
   const bool three_d = I->dim == AGX_DIM_3D;
   const bool cube = I->dim == AGX_DIM_CUBE;
   /* A per-lane level and a per-lane bias share one staging form, so they
    * share one operand test: a live single-channel 16-bit register. */
   const bool scalar_lod_reg = agx3_tex_plain_reg(I->src[1]) &&
      I->src[1].size == AGX_SIZE_16 && agx_channels(I->src[1]) == 1;
   /* A 2D gradient is four 32-bit scalars in one aligned run, which the four
    * staging copies name two halves apart. */
   const bool gradient_operand = agx3_tex_plain_reg(I->src[1]) &&
      I->src[1].size == AGX_SIZE_32 && agx_channels(I->src[1]) == 4 &&
      (I->src[1].value & 1) == 0 && I->src[1].value + 6 < 256;
   const bool valid_lod = explicit_lod ?
      (enc == AGX_ISA_ENCODING_AGX3B &&
       (valid_immediate_lod || scalar_lod_reg)) :
      biased_lod
         ? (enc == AGX_ISA_ENCODING_AGX3B && valid_immediate_bias &&
            immediate_level_x64 != 0.0f)
         : register_bias
         ? (enc == AGX_ISA_ENCODING_AGX3B && scalar_lod_reg)
         : gradient_lod
         ? (enc == AGX_ISA_ENCODING_AGX3B && gradient_operand)
         : (implicit_lod &&
            (zero_bias || agx3_tex_imm(I->src[1], 0)));
   const bool immediate = I->nr_srcs == 6 &&
      I->src[3].type == AGX_INDEX_IMMEDIATE &&
      I->src[4].type == AGX_INDEX_IMMEDIATE;
   const bool immediate_offset = offset &&
      I->src[5].type == AGX_INDEX_IMMEDIATE &&
      I->src[5].value > 0 && I->src[5].value <= UINT8_MAX &&
      !I->src[5].memory && !I->src[5].abs && !I->src[5].neg &&
      !I->src[5].cache && !I->src[5].discard && !I->src[5].kill;
   /* Array and 3D coordinates are copied to the fixed staging bank before SMP,
    * while the layer is named directly by the sample's 8-bit register field.
    * The third scalar must not wrap that field. Plain 2D samples name
    * the coordinate directly in SMP and retain its narrower bound. */
   const unsigned max_coord = array ? 120 : (three_d || cube) ? 248 : 60;
   if (I->nr_dests != 1 || I->nr_srcs != 6 ||
       !agx3_tex_plain_reg(I->dest[0]) ||
       (I->dest[0].size != AGX_SIZE_16 &&
        I->dest[0].size != AGX_SIZE_32) ||
       agx_channels(I->dest[0]) != 4 || I->dest[0].value >= 128 ||
       (I->dest[0].size == AGX_SIZE_32 && (I->dest[0].value & 1)) ||
       !agx3_tex_plain_reg(I->src[0]) ||
       I->src[0].size != AGX_SIZE_32 ||
       agx_channels(I->src[0]) != ((array || three_d || cube) ? 3 : 2) ||
       (I->src[0].value & 3) || I->src[0].value > max_coord ||
       !valid_lod || !agx3_tex_imm(I->src[2], 0) ||
       (immediate && (I->src[3].value > 15 || I->src[4].value > 15)) ||
       I->src[3].memory || I->src[3].abs || I->src[3].neg ||
       I->src[4].memory || I->src[4].abs || I->src[4].neg ||
       (!immediate &&
        ((I->src[3].type != AGX_INDEX_REGISTER && I->src[3].type != AGX_INDEX_UNIFORM) ||
         (I->src[4].type != AGX_INDEX_REGISTER && I->src[4].type != AGX_INDEX_UNIFORM) ||
         I->src[3].size != AGX_SIZE_64 || I->src[4].size != AGX_SIZE_64 ||
         agx_channels(I->src[3]) != 1 || agx_channels(I->src[4]) != 1 ||
         (I->src[3].value & 1) || (I->src[4].value & 1))) ||
       (shadow
           ? (!agx3_tex_plain_reg(I->src[5]) ||
              I->src[5].size != AGX_SIZE_32 ||
              agx_channels(I->src[5]) != 1 || I->src[5].value > 254)
           : offset ? !immediate_offset : !agx_is_null(I->src[5])) ||
       (I->dim != AGX_DIM_2D && I->dim != AGX_DIM_2D_ARRAY && !three_d &&
        !cube) ||
       (array && enc != AGX_ISA_ENCODING_AGX3B) ||
       (three_d &&
        (enc != AGX_ISA_ENCODING_AGX3B ||
         !valid_immediate_lod)) ||
       (cube &&
        (enc != AGX_ISA_ENCODING_AGX3B ||
         (!implicit_lod && !explicit_lod) || !immediate)) ||
       (I->mask == 0 || (I->mask & ~15u)) ||
       I->query_lod || I->sparse ||
       (offset &&
        (enc != AGX_ISA_ENCODING_AGX3B || !implicit_lod || !immediate ||
         array || three_d || cube || shadow || gather)) ||
       (shadow && (array || three_d || cube || !implicit_lod || gather || I->mask != 1 ||
                   enc != AGX_ISA_ENCODING_AGX3B)) ||
       (gather && (I->lod_mode != AGX_LOD_MODE_AUTO_LOD || array || three_d || cube ||
                   enc != AGX_ISA_ENCODING_AGX3B)))
      return false;

   /* The witnessed copy form carries at most one producer dependency. */
   if (wait_mask & (wait_mask - 1))
      return false;

   const bool register_operands = !immediate &&
      (I->src[3].type == AGX_INDEX_REGISTER ||
       I->src[4].type == AGX_INDEX_REGISTER);

   /* Native register-indexed array samples publish their coordinate/layer
    * operand bank on slot 6 and wait slot 6 in SMP, independently of SMP's
    * own result slot. Uniform/immediate array samples use the ordinary result
    * slot. Keeping these two timelines distinct is required by the witnessed
    * ep4 register-descriptor form. */
   const uint8_t stage_slot = array && register_operands ? 6 : slot;
   /* Every native array witness names coordinate bank 0 in SMP, including
    * register-indexed shaders whose physical coordinates live elsewhere. */
   const uint16_t coord = (explicit_lod || register_bias || array || three_d || cube ||
                           gradient_lod || shadow)
                             ? 0
                             : I->src[0].value;
   uint8_t bytes[AGX3_TEX_MAX_LENGTH];
   uint8_t len = 0;

   if ((explicit_lod && !immediate_lod) || register_bias) {
      if (!agx3_tex_stage_lod(enc, I->src[1].value, stage_slot, wait_mask,
                             I->src[1].cache, I->src[1].discard,
                             bytes, &len) ||
          !agx3_append(emission, bytes, len))
         return false;
      wait_mask = 0;
   }

   if (!agx3_tex_stage_copy_hints(enc, coord, I->src[0].value, stage_slot, wait_mask,
          enc == AGX_ISA_ENCODING_AGX3B && I->src[0].cache,
          enc != AGX_ISA_ENCODING_AGX3B || I->src[0].discard, bytes, &len) ||
       !agx3_append(emission, bytes, len))
      return false;
   if (!agx3_tex_stage_copy_hints(enc, coord + 2, I->src[0].value + 2, stage_slot, 0,
          enc == AGX_ISA_ENCODING_AGX3B && I->src[0].cache,
          enc != AGX_ISA_ENCODING_AGX3B || I->src[0].discard, bytes, &len) ||
       !agx3_append(emission, bytes, len))
      return false;

   if (three_d &&
       (!agx3_tex_stage_copy_hints(
           enc, coord + 4, I->src[0].value + 4, stage_slot, 0,
           I->src[0].cache, I->src[0].discard, bytes, &len) ||
        !agx3_append(emission, bytes, len)))
      return false;

   if (cube &&
       (!agx3_tex_stage_cube_z(enc, I->src[0].value + 4, stage_slot, 0,
                              I->src[0].cache, I->src[0].discard,
                              bytes, &len) ||
        !agx3_append(emission, bytes, len)))
      return false;

   if (shadow &&
       (!agx3_tex_stage_copy_hints(
           enc, 4, I->src[5].value, stage_slot, 0, I->src[5].cache,
           I->src[5].discard, bytes, &len) ||
        !agx3_append(emission, bytes, len)))
      return false;

   /* The gradient scalars follow the coordinates, in the interleaved order
    * (ddx.x, ddy.x, ddx.y, ddy.y) the native kernels stage them in. The
    * collect that produced them is laid out the same way, so the run is
    * copied straight through. */
   if (gradient_lod) {
      static const uint16_t bank[4] = {
         AGX3_TEX_GRADIENT_BANK_DDX_X, AGX3_TEX_GRADIENT_BANK_DDY_X,
         AGX3_TEX_GRADIENT_BANK_DDX_Y, AGX3_TEX_GRADIENT_BANK_DDY_Y,
      };

      for (unsigned i = 0; i < 4; ++i) {
         if (!agx3_tex_stage_copy_hints(enc, bank[i],
                                        I->src[1].value + (2 * i), stage_slot,
                                        0, I->src[1].cache, I->src[1].discard,
                                        bytes, &len) ||
             !agx3_append(emission, bytes, len))
            return false;
      }
   }

   const struct agx3_tex_sample_desc d = {
      .dest = I->dest[0].value,
      .coord = coord,
      .width = I->dest[0].size == AGX_SIZE_32 ? AGX3_TEX_W32 : AGX3_TEX_W16,
      .mask = I->mask,
      .texture = immediate ? I->src[3].value : 0,
      .sampler = immediate ? I->src[4].value : 0,
      .texture_kind = immediate ? AGX3_TEX_OPERAND_IMMEDIATE :
         I->src[3].type == AGX_INDEX_UNIFORM ? AGX3_TEX_OPERAND_UNIFORM : AGX3_TEX_OPERAND_REGISTER,
      .sampler_kind = immediate ? AGX3_TEX_OPERAND_IMMEDIATE :
         I->src[4].type == AGX_INDEX_UNIFORM ? AGX3_TEX_OPERAND_UNIFORM : AGX3_TEX_OPERAND_REGISTER,
      .texture_value = immediate ? 0 : I->src[3].value,
      .sampler_value = immediate ? 0 : I->src[4].value,
      .texture_kill = !immediate && I->src[3].kill,
      .sampler_kill = !immediate && I->src[4].kill,
      .gather = gather_component,
      .shadow = shadow,
      .offset = offset,
      .offset_xy = offset ? I->src[5].value : 0,
      .lod = biased_lod       ? AGX3_TEX_LOD_BIAS
             : immediate_lod  ? AGX3_TEX_LOD_LEVEL
             : explicit_lod   ? AGX3_TEX_LOD_REGISTER
             : register_bias  ? AGX3_TEX_LOD_BIAS_REGISTER
             : gradient_lod   ? AGX3_TEX_LOD_GRADIENT
                              : AGX3_TEX_LOD_AUTO,
      .lod_x64 = immediate_lod ? immediate_x64_field : 0,
      .array = array,
      .array_layer = array ? I->src[0].value + 4 : 0,
      .three_d = three_d,
      .cube = cube,
      .slot = slot,
      .wait_mask = (uint16_t)(1u << (stage_slot - 1)),
      .coord_kill = I->src[0].kill,
      .final = final,
   };
   return agx3_tex_encode_sample(enc, &d, bytes, &len) &&
          agx3_append(emission, bytes, len);
}

static int
agx3_tile_format_width(enum agx_format format)
{
   switch (format) {
   case AGX_FORMAT_I8:
   case AGX_FORMAT_U8NORM:
   case AGX_FORMAT_S8NORM:
   case AGX_FORMAT_SRGBA8:
      return AGX3_TILE_W8;
   case AGX_FORMAT_I16:
   case AGX_FORMAT_F16:
   case AGX_FORMAT_U16NORM:
   case AGX_FORMAT_S16NORM:
      return AGX3_TILE_W16;
   case AGX_FORMAT_I32:
   case AGX_FORMAT_RGB10A2:
   case AGX_FORMAT_RG11B10F:
   case AGX_FORMAT_RGB9E5:
      return AGX3_TILE_W32;
   default:
      return -1;
   }
}

/*
 * The register span a tile store or load names.
 *
 * At 16 and 32 bits that is one register per written channel. At 8 bits it is
 * not: `lstcol` carries a width, not a format, so an 8-bit access moves *two*
 * channels per 16-bit register and the value is half as wide as the mask is
 * long. Apple's fragment shaders show this directly -- a four-channel 8-bit
 * store reads two 16-bit registers, a two-channel one reads a single register,
 * and a one-channel one reads the low byte of a single register.
 *
 * Anything else at 8 bits is refused. Before the `fltnorm` pack existed this
 * function accepted a four-channel 32-bit float value with an 8-bit width,
 * which would have stored the low byte of each float's raw bit pattern.
 */
static bool
agx3_tile_value_ok(agx_index value, uint8_t mask, int width)
{
   if (!agx3_src_is_register(value) || value.memory || value.abs || value.neg ||
       (value.size != AGX_SIZE_16 && value.size != AGX_SIZE_32) ||
       (value.value & (agx_size_align_16(value.size) - 1)))
      return false;

   const unsigned channels = util_bitcount(mask);
   if (width == AGX3_TILE_W8) {
      /* Both directions: the store's value is packed by fltnorm, and the
       * load's destination is what normflt then unpacks. */
      if (value.size != AGX_SIZE_16 ||
          agx_channels(value) != DIV_ROUND_UP(channels, 2))
         return false;
   } else if (agx_channels(value) != channels) {
      return false;
   }

   return value.value + agx_index_size_16(value) <= 256;
}


/* agx_nir_lower_g17p_tile_samples tags each statically known sample-count
 * domain before backend compilation. Keeping the count in the immediate is
 * essential: native lstcol uses different selectors for 2x and 4x sample 0,
 * while lldcol uses a one-hot selector for both. */
#define AGX3_TILE_ALL_SAMPLES 0xFFu
#define AGX3_TILE_2X_TAG       0x20u
#define AGX3_TILE_4X_TAG       0x40u

static bool
agx3_tile_coverage(agx_index sample, enum agx3_tile_coverage *coverage)
{
   if (sample.type != AGX_INDEX_IMMEDIATE || sample.size != AGX_SIZE_16 ||
       sample.memory || sample.abs || sample.neg || sample.cache ||
       sample.discard || sample.kill)
      return false;

   if (sample.value == AGX3_TILE_ALL_SAMPLES) {
      *coverage = AGX3_TILE_COV_1X;
      return true;
   }

   const unsigned tag = sample.value & ~0xfu;
   const unsigned mask = sample.value & 0xfu;
   if (tag == AGX3_TILE_2X_TAG) {
      if (mask == 0)
         *coverage = AGX3_TILE_COV_2X_BROADCAST;
      else if (mask == 1 || mask == 2)
         *coverage = AGX3_TILE_COV_2X_SAMPLE_0 + util_logbase2(mask);
      else
         return false;
   } else if (tag == AGX3_TILE_4X_TAG) {
      if (mask == 0)
         *coverage = AGX3_TILE_COV_4X_BROADCAST;
      else if (util_is_power_of_two_nonzero(mask) && mask <= 8)
         *coverage = AGX3_TILE_COV_4X_SAMPLE_0 + util_logbase2(mask);
      else
         return false;
   } else {
      return false;
   }

   return true;
}

/* Bind the directly witnessed single-sample tile subset.  Explicit pixel
 * coordinates and the per-sample selector remain closed because f44's exact
 * register/selector meaning is not yet grounded. */
static bool
agx3_emit_tile(struct util_dynarray *emission, enum agx_isa_encoding enc,
               const agx_instr *I, uint8_t slot, uint16_t wait_mask,
               bool first_load, bool last_load)
{
   const bool load = I->op == AGX_OPCODE_LD_TILE;
   if ((!load && I->op != AGX_OPCODE_ST_TILE) ||
       I->nr_dests != (load ? 1 : 0) || I->nr_srcs != (load ? 2 : 3) ||
       I->mask < 1 || I->mask > 15 || I->pixel_offset > UINT8_MAX ||
       I->explicit_coords)
      return false;

   const agx_index value = load ? I->dest[0] : I->src[0];
   const agx_index sample = load ? I->src[0] : I->src[1];
   const agx_index coords = load ? I->src[1] : I->src[2];
   const int width = agx3_tile_format_width(I->format);
   enum agx3_tile_coverage coverage;
   if ((I->format == AGX_FORMAT_RGB10A2 &&
        (I->mask != 1 || value.size != AGX_SIZE_32)) ||
       width < 0 || !agx3_tile_value_ok(value, I->mask, width) ||
       !agx3_tile_coverage(sample, &coverage) || !agx_is_null(coords))
      return false;

   /* AGX3b f13 varies independently in high-register store witnesses.  The
    * low-register, single-sample subset is the only one with a stable zero
    * selection that production IR can infer today. */
   if (!load && enc == AGX_ISA_ENCODING_AGX3B && value.value >= 16)
      return false;

   uint8_t bytes[AGX3_TILE_MAX_LENGTH];
   uint8_t length = 0;
   const bool ok =
      load ? agx3_tile_load(enc, value.value, (enum agx3_tile_width)width,
                            slot, I->mask, I->pixel_offset, coverage,
                            wait_mask, first_load, last_load, bytes, &length)
           : agx3_tile_store(enc, value.value,
                             (enum agx3_tile_width)width, I->mask,
                             I->pixel_offset, coverage, wait_mask,
                             bytes, &length);
   return ok && agx3_append(emission, bytes, length);
}

/*
 * Fragment ops that stay refused, and why. Each is a missing *witness*, not
 * missing routing, so grounding one is a corpus session rather than a patch:
 *
 * - sample_mask (Mesa's mask+value write, from discard or an explicit
 *   gl_SampleMask). Apple's discard and [[sample_mask]] shaders both use the
 *   same writeout group as depthf -- a staging `copy` whose destination field
 *   is 0 in every witness, then `movmsk` under the class-1 pixwait/pixrel
 *   bracket -- and never `cmsk` (cmsk appears only in alpha-to-coverage
 *   shaders). movmsk's operand roles (f13/f17) are not isolated and the
 *   staged-register ABI below applies equally, so both spellings stay closed.
 *
 * - zs_emit stencil and combined depth/stencil. The depth-only staging ABI is
 *   now independently grounded by Apple's fragment-atomic late-test path.
 *   Stencil still reaches r0h through an extra conversion/copy sequence whose
 *   exact input contract has not been isolated, so those two forms remain
 *   closed rather than inheriting the depth-only route.
 *
 * See research/gpu/GPU_AGX3_TILE_OPS_2026-08-24.md sections 4 and 6.
 */

/*
 * `fltnorm`: pack two 32-bit floats, clamped and scaled, into the two halves
 * of one 16-bit register. agx_pack_normalized_agx3() in agx_compile.c creates
 * these immediately before an 8-bit normalized tile store, one per channel
 * pair, which is exactly what Apple's own G17P fragment shaders do.
 *
 * The second source is null for an odd trailing channel; that selects the
 * single-lane form Apple emits for a one-component 8-bit target.
 *
 * Both sources are marked "still live" rather than "last read". The bit pair
 * is a liveness hint -- Apple sets one or the other per operand, and both
 * shapes are witnessed -- but Mesa cannot supply the fact: agx_register_like
 * drops kill/discard when RA lowers SSA to registers, so the IR's own bit
 * reads false here regardless of the truth. Claiming a value is still live is
 * the conservative half of the pair.
 */
static bool
agx3_emit_fltnorm(struct util_dynarray *emission, enum agx_isa_encoding enc,
                  const agx_instr *I, uint16_t wait_mask, bool first_instruction,
                  struct agx3_fltnorm_desc *link_desc)
{
   if (I->nr_dests != 1 || I->nr_srcs != 2)
      return false;

   const agx_index dest = I->dest[0];
   const bool norm16 = I->format == AGX_FORMAT_U16NORM ||
                       I->format == AGX_FORMAT_S16NORM;
   if (I->format != AGX_FORMAT_U8NORM && I->format != AGX_FORMAT_S8NORM &&
       I->format != AGX_FORMAT_U16NORM && I->format != AGX_FORMAT_S16NORM)
      return false;
   const bool two_lanes = !agx_is_null(I->src[1]);
   if (!agx3_src_is_register(dest) || dest.memory || dest.size != AGX_SIZE_16 ||
       agx_channels(dest) != (norm16 && two_lanes ? 2 : 1) ||
       dest.value >= 256)
      return false;

   struct agx3_fltnorm_desc d = {
      .dest = dest.value,
      .two_lanes = two_lanes,
      .snorm = I->format == AGX_FORMAT_S8NORM ||
               I->format == AGX_FORMAT_S16NORM,
      .is_16bit = norm16,
      .wait_mask = wait_mask,
      .first_instruction = first_instruction,
   };

   for (unsigned i = 0; i < 2; ++i) {
      if (i == 1 && !d.two_lanes)
         break;
      const agx_index src = I->src[i];
      if (src.memory || src.abs || src.neg || agx_channels(src) != 1)
         return false;

      /* All three operand classes are witnessed: a register (every
       * interpolated shader), an 8-bit AGX minifloat immediate (a constant
       * colour, which the optimizer folds straight into this source), and a
       * uniform (a colour read from a buffer). */
      if (agx3_src_is_register(src)) {
         if (src.size != AGX_SIZE_32)
            return false;
         d.src[i].kind = AGX3_FLTNORM_SRC_REGISTER;
      } else if (src.type == AGX_INDEX_IMMEDIATE) {
         if (src.value > 0xff)
            return false;
         d.src[i].kind = AGX3_FLTNORM_SRC_IMMEDIATE;
      } else if (src.type == AGX_INDEX_UNIFORM) {
         if (src.size != AGX_SIZE_32)
            return false;
         d.src[i].kind = AGX3_FLTNORM_SRC_UNIFORM;
      } else {
         return false;
      }

      d.src[i].value = src.value;
      d.src[i].kill = false;
   }

   uint8_t bytes[AGX3_FLTNORM_MAX_LENGTH];
   uint8_t len = 0;
   if (!agx3_fltnorm_encode(enc, &d, bytes, &len) ||
       !agx3_append(emission, bytes, len))
      return false;
   if (link_desc)
      *link_desc = d;
   return true;
}

static bool
agx3_emit_rgb10a2(struct util_dynarray *emission, const agx_instr *I,
                 uint16_t wait, bool first)
{
   const bool unpack = I->op == AGX_OPCODE_UNPACK_RGB10A2_AGX3;
   if (I->nr_dests != 1 || I->nr_srcs != (unpack ? 1 : 4))
      return false;
   const agx_index dest = I->dest[0];
   if (!agx3_src_is_register(dest) || dest.memory || dest.size != AGX_SIZE_32 ||
       agx_channels(dest) != (unpack ? 4 : 1))
      return false;
   uint8_t bytes[14], length = 0;
   if (unpack) {
      const agx_index src = I->src[0];
      if (!agx3_src_is_register(src) || src.memory || src.abs || src.neg ||
          src.size != AGX_SIZE_32 || agx_channels(src) != 1)
         return false;
      const struct agx3_rgb10a2_unpack_desc d = {
         .dest = dest.value, .source = src.value,
         .wait_mask = wait, .discard = src.discard,
      };
      if (!agx3_rgb10a2_unpack_encode(&d, bytes, &length))
         return false;
   } else {
      struct agx3_rgb10a2_desc d = {.dest = dest.value,
                                   .first_instruction = first, .wait_mask = wait};
      for (unsigned i = 0; i < 4; ++i) {
         const agx_index src = I->src[i];
         if (src.memory || src.abs || src.neg || agx_channels(src) != 1 ||
             (agx3_src_is_register(src) && src.size != AGX_SIZE_32))
            return false;
         d.kind[i] = agx3_src_is_register(src) ? AGX3_FLTNORM_SRC_REGISTER :
                     src.type == AGX_INDEX_IMMEDIATE ? AGX3_FLTNORM_SRC_IMMEDIATE : 0;
         d.source[i] = src.value;
         d.discard[i] = src.discard;
         d.cache[i] = src.cache;
      }
      if (!agx3_rgb10a2_encode(&d, bytes, &length))
         return false;
   }
   return agx3_append(emission, bytes, length);
}

/*
 * `normflt`: the tilebuffer-read inverse of `fltnorm`. One 16-bit register of
 * packed normalized lanes in, a 32-bit float pair out.
 */
static bool
agx3_emit_normflt(struct util_dynarray *emission, enum agx_isa_encoding enc,
                  const agx_instr *I, uint16_t wait_mask)
{
   if (I->nr_dests != 1 || I->nr_srcs != 1)
      return false;

   const agx_index dest = I->dest[0];
   const agx_index src = I->src[0];

   if (!agx3_src_is_register(dest) || dest.memory || dest.size != AGX_SIZE_32 ||
       agx_channels(dest) > 2 || agx_channels(dest) < 1)
      return false;
   const bool norm16 = I->format == AGX_FORMAT_U16NORM ||
                       I->format == AGX_FORMAT_S16NORM;
   if (I->format != AGX_FORMAT_U8NORM && I->format != AGX_FORMAT_S8NORM &&
       I->format != AGX_FORMAT_U16NORM && I->format != AGX_FORMAT_S16NORM)
      return false;
   const bool two_lanes = agx_channels(dest) == 2;
   if (!agx3_src_is_register(src) || src.memory || src.abs || src.neg ||
       src.size != AGX_SIZE_16 ||
       agx_channels(src) != (norm16 && two_lanes ? 2 : 1))
      return false;

   const struct agx3_normflt_desc d = {
      .dest = dest.value,
      .dest_hint = true,
      .src = src.value,
      .two_lanes = two_lanes,
      .snorm = I->format == AGX_FORMAT_S8NORM ||
               I->format == AGX_FORMAT_S16NORM,
      .is_16bit = norm16,
      .wait_mask = wait_mask,
   };

   uint8_t bytes[AGX3_FLTNORM_MAX_LENGTH];
   uint8_t len = 0;
   return agx3_normflt_encode(enc, &d, bytes, &len) &&
          agx3_append(emission, bytes, len);
}

/*
 * Pixel writeout barriers: wait_pix -> pixwait, signal_pix -> pixrel.
 *
 * The IR carries the writeout-class bitmask in `writeout`, which is the same
 * mask Apple's fragment corpus puts in the instruction, so it passes straight
 * through; agx3_pixel_core refuses any class Apple never emitted.
 *
 * signal_pix has no producer in Mesa today. It is routed anyway because it is
 * grounded, but only for the release descriptors witnessed without the
 * end-of-sequence bit: Mesa's IR has no way to say "this is the final release",
 * and inventing that bit is exactly the kind of guess this backend refuses.
 */
static bool
agx3_emit_pixel(struct util_dynarray *emission, enum agx_isa_encoding enc,
                const agx_instr *I, uint16_t wait_mask)
{
   if (I->nr_dests != 0 || I->nr_srcs != 0)
      return false;

   uint8_t bytes[AGX3_PIXEL_MAX_LENGTH];
   uint8_t len = 0;
   const bool ok =
      I->op == AGX_OPCODE_WAIT_PIX
         ? agx3_pixel_wait(enc, I->writeout, wait_mask, bytes, &len)
         : agx3_pixel_release(enc, I->writeout, false, wait_mask, bytes, &len);

   return ok && agx3_append(emission, bytes, len);
}

/* Append the half-registers an operand occupies (empty for immediates,
 * uniforms, or null). */
static unsigned
agx3_expand_regs(agx_index idx, uint16_t *arr, unsigned n)
{
   if (!agx3_src_is_register(idx) || idx.memory)
      return n;
   const unsigned halves = agx_size_align_16(idx.size) * agx_channels(idx);
   for (unsigned i = 0; i < halves && n < AGX3_SLOT_MAX_REGS; ++i)
      arr[n++] = (uint16_t)(idx.value + i);
   return n;
}

static int
agx3_fmt_to_width(enum agx_format f)
{
   switch (f) {
   case AGX_FORMAT_I8:
      return AGX3_MEM_W8;
   case AGX_FORMAT_I16:
   case AGX_FORMAT_F16:
      return AGX3_MEM_W16;
   case AGX_FORMAT_I32:
      return AGX3_MEM_W32;
   default:
      return -1; /* norm/packed formats have no AGX3 device-op witness */
   }
}

static bool
agx3_scs_plain_index(agx_index index)
{
   return !index.kill && !index.cache && !index.discard && !index.abs &&
          !index.neg && !index.memory && !index.has_reg &&
          agx_channels(index) == 1;
}

static bool
agx3_translate_scs(const agx_instr *I, bool is_store, bool indexed,
                   uint8_t slot, uint16_t wait_mask,
                   struct agx3_scs_desc *d)
{
   memset(d, 0, sizeof(*d));
   const int width = agx3_fmt_to_width(I->format);
   const int64_t byte_offset = (int64_t)I->imm;
   const unsigned expected_srcs = indexed ? (is_store ? 3 : 2)
                                          : (is_store ? 1 : 0);
   if (width < 0 || byte_offset < INT16_MIN || byte_offset > INT16_MAX ||
       I->nr_dests != (is_store ? 0 : 1) ||
       I->nr_srcs != expected_srcs ||
       ((is_store || !indexed) &&
        (I->scs_zero_extend || I->scs_c_load_marker)))
      return false;

   const agx_index data = is_store ? I->src[0] : I->dest[0];
   const enum agx_size expected =
      width == AGX3_MEM_W32 ? AGX_SIZE_32 : AGX_SIZE_16;
   if (!agx3_src_is_register(data) || !agx3_scs_plain_index(data) ||
       data.size != expected || agx_channels(data) != 1 ||
       (data.value & (agx_size_align_16(expected) - 1)) ||
       data.value + agx_size_align_16(expected) > 256)
      return false;

   *d = (struct agx3_scs_desc){
      .data = data.value,
      .width = (enum agx3_mem_width)width,
      .byte_offset = (int16_t)byte_offset,
      .address_kind = indexed ? AGX3_SCS_ADDRESS_UNIFORM_INDEXED
                              : AGX3_SCS_ADDRESS_DIRECT,
      .slot = is_store ? 0 : slot,
      .wait_mask = wait_mask,
   };

   if (indexed) {
      const agx_index base = I->src[is_store ? 1 : 0];
      const agx_index index = I->src[is_store ? 2 : 1];

      /* The pseudo itself preserves the address-space provenance.  Require
       * the physical operands to preserve it too: a scalar 64-bit uniform
       * base and a scalar aligned 32-bit GPR index.  In particular, no
       * register base, vector, source modifier, or inferred last-use bit is
       * silently encoded into fields whose semantics are not yet grounded.
       */
      if (base.type != AGX_INDEX_UNIFORM || base.size != AGX_SIZE_64 ||
          !agx3_scs_plain_index(base) || base.value >= 512 ||
          !agx3_src_is_register(index) || index.size != AGX_SIZE_32 ||
          !agx3_scs_plain_index(index) || (index.value & 1) ||
          index.value + agx_size_align_16(index.size) > 256)
         return false;

      d->uniform_base = base.value;
      d->index = index.value;
      d->zero_extend = !is_store && I->scs_zero_extend;
      d->c_load_marker = !is_store && I->scs_c_load_marker;
   }

   return true;
}

/* Translate a device_load/device_store into the normalized memory descriptor.
 * Returns false for any operand shape outside the grounded domain. */
static bool
agx3_translate_mem(const agx_instr *I, bool is_store, uint8_t slot,
                   uint16_t wait_mask, struct agx3_mem_desc *d)
{
   memset(d, 0, sizeof(*d));

   const int w = agx3_fmt_to_width(I->format);
   if (w < 0 || I->mask < 1 || I->mask > 15 || I->shift > 4)
      return false;
   d->width = (enum agx3_mem_width)w;
   d->mask = (uint8_t)I->mask;
   d->shift = (uint8_t)I->shift;

   const agx_index data = is_store ? I->src[0] : I->dest[0];
   /* Native scalar byte loads write an aligned low half. Stores can also
    * address individual halves: Apple's byte_halves_odd_zero stores
    * ushort2.y from half3 with W8/mask1. */
   const bool scalar_half = data.size == AGX_SIZE_16 &&
      (w == AGX3_MEM_W16 || w == AGX3_MEM_W8) &&
      I->mask == 1 && agx_channels(data) == 1;
   const bool byte_load_half = !is_store && w == AGX3_MEM_W8 && scalar_half;
   if (!agx3_src_is_register(data) || data.memory ||
       ((!scalar_half || byte_load_half) && (data.value & 1)) ||
       data.value >= 256)
      return false;
   d->data = data.value;

   const agx_index base = I->src[is_store ? 1 : 0];
   if (base.type == AGX_INDEX_UNIFORM) {
      d->base_is_uniform = true;
      d->base = base.value;
   } else if (agx3_src_is_register(base) && !base.memory) {
      d->base_is_uniform = false;
      d->base = base.value;
      d->base_kill = base.kill;
   } else {
      return false;
   }

   const agx_index index = I->src[is_store ? 2 : 1];

   /*
    * The index -- immediate or register -- counts ELEMENTS, and
    * agx_nir_lower_address() has already subtracted the format shift from
    * I->shift (match.shift = shift - format_shift, the AGX2 convention where
    * the hardware rescales). AGX3 does not rescale, so the true scale is
    * I->shift + w, where w is log2(element bytes).
    */
   const unsigned total_shift = (unsigned)I->shift +
      (I->memory_index_is_bytes ? 0 : (unsigned)w);

   if (index.type == AGX_INDEX_IMMEDIATE) {
      d->idx_kind = AGX3_MEM_IDX_IMM;
      /*
       * A byte offset is absolute, so it must carry the FULL scale. Using
       * I->shift alone addressed element 9 of the root descriptor table at
       * byte 9 instead of byte 36 -- invisible on every probe whose only
       * immediate index is 0, and the reason the constant-data (rodata)
       * pointer came back wrong.
       */
      const int64_t bytes = (int64_t)index.value << total_shift;
      if (bytes >= -32768 && bytes <= 32767) {
         d->idx_scaled = false;
         d->byte_off = (int32_t)bytes;
         d->shift = 0;
      } else if (index.value <= 0xffff && total_shift <= 4) {
         d->idx_scaled = true;
         d->elem_idx = index.value;
         d->shift = (uint8_t)total_shift;
      } else {
         return false; /* needs address-math legalization */
      }
   } else if (agx3_src_is_register(index) && !index.memory) {
      d->idx_kind = AGX3_MEM_IDX_REG;
      if ((index.value & 1) != 0 || index.value > 191)
         return false;
      d->idx_reg = index.value;
      /* Tested on hardware 2026-08-31 and REFUTED: forcing this bit to 1 (so
       * the store's index operand cannot decode as "operand absent") did not
       * change the result -- the Vulkan dispatch still wrote nothing. Keep the
       * IR's own liveness. */
      d->idx_kill = index.kill;
      d->idx_signed = false;

      /*
       * A REGISTER index is scaled by the shift alone -- the hardware does
       * NOT additionally scale it by the access format. Apple's corpus is
       * unambiguous: every register-indexed lddev/stdev carries
       * shift == log2(element bytes) plus any vector stride (16-bit -> 1,
       * 32-bit -> 2, vec4-of-32-bit -> 4), and shift 0 occurs only on
       * immediate byte offsets.
       *
       * agx_nir_lower_address() has already SUBTRACTED the format shift
       * (match.shift = shift - format_shift) because that is the AGX2
       * convention, so add it back. `w` is log2(element bytes).
       *
       * Without this a `values[i]` store addressed base + i BYTES: 32 lanes
       * folded onto 8 elements (the hardware aligns the access down), which
       * read back as a bogus 4x scaling on the value and as 8-of-32 coverage.
       * The immediate-index path above never had the bug because it converts
       * to a byte offset itself and forces shift 0 -- which is why constant
       * indices worked while index-derived ones did not.
       */
      if (total_shift > 4)
         return false;
      d->shift = (uint8_t)total_shift;
   } else {
      return false;
   }

   d->slot = slot;
   d->wait_mask = wait_mask;
   return true;
}

static uint8_t
agx3_local_vector_shift(enum agx3_mem_width width, uint8_t mask)
{
   const unsigned channels = util_bitcount(mask);
   unsigned bytes = channels << width;
   uint8_t shift = 0;
   while ((1u << shift) < bytes)
      shift++;
   return shift;
}

/* Translate the shared-memory shape Mesa currently emits. Address arithmetic
 * is folded into the 16-bit base by agx_compile, leaving an immediate-zero
 * index. This is exactly the grounded Apple ldtg/sttg subset; a register index
 * is refused because Mesa's byte-addressed IR has not yet been legalized to
 * AGX3's scaled-index convention. */
static bool
agx3_translate_local(const agx_instr *I, bool is_store, uint8_t slot,
                     uint16_t wait_mask, struct agx3_local_desc *d)
{
   memset(d, 0, sizeof(*d));

   const int w = agx3_fmt_to_width(I->format);
   if (w < 0 || I->mask < 1 || I->mask > 15 ||
       I->mask != BITFIELD_MASK(util_bitcount(I->mask)))
      return false;
   d->width = (enum agx3_mem_width)w;
   d->mask = (uint8_t)I->mask;
   d->shift = agx3_local_vector_shift(d->width, d->mask);

   const agx_index data = is_store ? I->src[0] : I->dest[0];
   const enum agx_size expected =
      d->width == AGX3_MEM_W32 ? AGX_SIZE_32 : AGX_SIZE_16;
   const unsigned halves = agx_size_align_16(expected) * util_bitcount(I->mask);
   if (!agx3_src_is_register(data) || data.memory || data.size != expected ||
       (data.value & (agx_size_align_16(expected) - 1)) ||
       data.value + halves > 256)
      return false;
   d->data = data.value;
   d->value_kill = is_store && data.kill;

   const agx_index base = I->src[is_store ? 1 : 0];
   if (base.size != AGX_SIZE_16 || base.memory || base.abs || base.neg)
      return false;
   if (base.type == AGX_INDEX_IMMEDIATE && base.value == 0) {
      d->base_kind = AGX3_LOCAL_BASE_IMMEDIATE;
   } else if (base.type == AGX_INDEX_UNIFORM && base.value < 512) {
      d->base_kind = AGX3_LOCAL_BASE_UNIFORM;
      d->base = base.value;
   } else if (agx3_src_is_register(base) && base.value < 256) {
      d->base_kind = AGX3_LOCAL_BASE_REGISTER;
      d->base = base.value;
      d->base_kill = base.kill;
   } else {
      return false;
   }

   const agx_index index = I->src[is_store ? 2 : 1];
   if (index.type != AGX_INDEX_IMMEDIATE || index.size != AGX_SIZE_16 ||
       index.value != 0 || index.memory || index.abs || index.neg)
      return false;

   d->narrow_extend = d->width == AGX3_MEM_W8;
   d->slot = slot;
   d->wait_mask = wait_mask;
   return true;
}

static bool
agx3_atomic_plain_reg(agx_index index, enum agx_size size, unsigned channels)
{
   return agx3_src_is_register(index) && !index.memory && !index.abs &&
          !index.neg && !index.kill && !(index.cache && index.discard) &&
          index.size == size &&
          agx_channels(index) == channels &&
          (index.value & (agx_size_align_16(size) - 1)) == 0 &&
          index.value + agx_size_align_16(size) * channels <= 256;
}

static bool
agx3_map_atomic_op(enum agx_atomic_opc in, enum agx3_atomic_op *out)
{
   switch (in) {
#define OP(x)                                                                  \
   case AGX_ATOMIC_OPC_##x:                                                    \
      *out = AGX3_ATOMIC_##x;                                                  \
      return true
      OP(ADD);
      OP(SUB);
      OP(XCHG);
      OP(CMPXCHG);
      OP(UMIN);
      OP(IMIN);
      OP(UMAX);
      OP(IMAX);
      OP(AND);
      OP(OR);
      OP(XOR);
#undef OP
   }

   return false;
}

/* Translate Mesa's global atomic IR into the fully witnessed 32-bit AGX3
 * family.  The address lowering represents a constant byte displacement as a
 * 32-bit element index; fold it back into the signed byte-offset field when it
 * fits, matching Apple.  AGX3b returned atomics are physically in-place, so
 * only an RA-coalesced value/result pair is accepted on that encoding. */
static bool
agx3_translate_device_atomic(const agx_instr *I,
                             enum agx_isa_encoding enc, bool exec_masked,
                             uint8_t slot,
                             uint16_t wait_mask,
                             struct agx3_device_atomic_desc *d)
{
   memset(d, 0, sizeof(*d));
   if (I->op != AGX_OPCODE_ATOMIC || I->nr_dests != 1 || I->nr_srcs != 3 ||
       !agx3_map_atomic_op(I->atomic_opc, &d->op))
      return false;

   d->exec_masked = exec_masked;

   const unsigned value_channels =
      d->op == AGX3_ATOMIC_CMPXCHG ? 2 : 1;
   const agx_index value = I->src[0];
   if (!agx3_atomic_plain_reg(value, AGX_SIZE_32, value_channels))
      return false;
   d->value = value.value;
   d->value_kill = value.discard;

   const agx_index result = I->dest[0];
   d->returns_value = !agx_is_null(result);
   if (d->returns_value) {
      if (!agx3_atomic_plain_reg(result, AGX_SIZE_32, 1) ||
          (enc == AGX_ISA_ENCODING_AGX3B && result.value != value.value))
         return false;
      d->result = result.value;
      d->slot = slot;
   } else if (slot != 0) {
      return false;
   }

   const agx_index base = I->src[1];
   if (base.memory || base.abs || base.neg || base.kill ||
       (base.cache && base.discard) || base.size != AGX_SIZE_64 ||
       agx_channels(base) != 1) {
      return false;
   } else if (base.type == AGX_INDEX_UNIFORM) {
      d->base_kind = AGX3_ATOMIC_BASE_UNIFORM;
      d->base = base.value;
   } else if (agx3_src_is_register(base)) {
      d->base_kind = AGX3_ATOMIC_BASE_REGISTER;
      d->base = base.value;
      /* atomicdev has no base cache/discard field. As on AGX2, silently
       * dropping these allocator hints is semantics-preserving. */
   } else {
      return false;
   }

   const agx_index index = I->src[2];
   if (index.memory || index.abs || index.neg || index.kill ||
       (index.cache && index.discard) || agx_channels(index) != 1)
      return false;

   if (index.type == AGX_INDEX_IMMEDIATE) {
      if (index.kill || index.cache || index.discard ||
          index.value > UINT16_MAX)
         return false;
      const uint64_t byte_offset = (uint64_t)index.value << 2;
      if (byte_offset <= INT16_MAX) {
         d->index_kind = AGX3_ATOMIC_INDEX_IMMEDIATE;
         d->byte_offset = byte_offset;
      } else {
         d->index_kind = AGX3_ATOMIC_INDEX_IMMEDIATE;
         d->index = index.value;
         d->index_shift = 2;
      }
   } else if (agx3_src_is_register(index) && index.size == AGX_SIZE_32 &&
              !index.kill && !(index.cache && index.discard) &&
              (index.value & 1) == 0) {
      d->index_kind = AGX3_ATOMIC_INDEX_REGISTER;
      d->index = index.value;
      d->index_kill = index.discard;
      d->index_shift = 2;
   } else if (index.type == AGX_INDEX_UNIFORM && index.size == AGX_SIZE_32 &&
              index.value < 512 && !index.kill && !index.cache &&
              !index.discard) {
      d->index_kind = AGX3_ATOMIC_INDEX_UNIFORM;
      d->index = index.value;
      d->index_shift = 2;
   } else {
      return false;
   }

   d->wait_mask = wait_mask;
   return true;
}

/* Apple's AGX3b no-result form accepts either a literal byte displacement or
 * a 16-bit GPR containing the byte address.  This is the shape produced by
 * Vulkan shared atomics after agx_compile folds the index into the base.
 * Returned atomics retain the cross-generation address-zero restriction. */
static bool
agx3_translate_local_atomic(const agx_instr *I, uint8_t slot,
                            uint16_t wait_mask,
                            struct agx3_local_atomic_desc *d)
{
   memset(d, 0, sizeof(*d));
   if (I->op != AGX_OPCODE_LOCAL_ATOMIC || I->nr_dests != 1 ||
       I->nr_srcs != 3 ||
       !agx3_map_atomic_op(I->atomic_opc, &d->op))
      return false;

   const unsigned value_channels =
      d->op == AGX3_ATOMIC_CMPXCHG ? 2 : 1;
   if (!agx3_atomic_plain_reg(I->src[0], AGX_SIZE_32, value_channels))
      return false;

   d->returns_value = !agx_is_null(I->dest[0]);
   if (d->returns_value) {
      if (!agx3_atomic_plain_reg(I->dest[0], AGX_SIZE_32, 1))
         return false;
      d->result = I->dest[0].value;
      d->slot = slot;
   } else if (slot != 0) {
      return false;
   }

   const agx_index index = I->src[2];
   if (index.type != AGX_INDEX_IMMEDIATE || index.value != 0 ||
       index.size != AGX_SIZE_16 || index.memory || index.abs || index.neg ||
       index.cache || index.discard || index.kill)
      return false;

   const agx_index base = I->src[1];
   if (base.size != AGX_SIZE_16 || base.memory || base.abs || base.neg)
      return false;
   if (base.type == AGX_INDEX_IMMEDIATE && base.value <= UINT16_MAX &&
       (base.value & 3) == 0 && !base.cache && !base.discard && !base.kill) {
      d->address_kind = AGX3_ATOMIC_INDEX_IMMEDIATE;
      d->address = base.value;
   } else if (agx3_src_is_register(base) && base.value <= 191 &&
              !(base.cache && base.discard)) {
      d->address_kind = AGX3_ATOMIC_INDEX_REGISTER;
      d->address = base.value;
      d->address_kill = base.kill;
   } else {
      return false;
   }

   d->value = I->src[0].value;
   d->value_kill = I->src[0].discard;
   d->wait_mask = wait_mask;
   return true;
}

/* Mesa's scalar 32-bit stack operation is an exact match for Apple's static
 * copy/rld family: the immediate byte offset names a TPS half-register and the
 * single data operand names one aligned GPR pair.  Vector masks, 16-bit values,
 * dynamic offsets and modifiers stay on the existing fail-closed path because
 * they do not uniquely determine the witnessed structural fields/forms. */
static bool
agx3_translate_static_spill(const agx_instr *I, bool is_store, uint8_t slot,
                            uint16_t wait_mask,
                            struct agx3_static_spill_store_desc *store,
                            struct agx3_static_spill_load_desc *load)
{
   if (I->op != (is_store ? AGX_OPCODE_STACK_STORE : AGX_OPCODE_STACK_LOAD) ||
       I->nr_dests != (is_store ? 0 : 1) ||
       I->nr_srcs != (is_store ? 2 : 1) || I->format != AGX_FORMAT_I32 ||
       I->mask != 1)
      return false;

   const agx_index data = is_store ? I->src[0] : I->dest[0];
   const agx_index offset = I->src[is_store ? 1 : 0];
   if (!agx3_src_is_register(data) || data.memory || data.abs || data.neg ||
       data.cache || data.discard || data.size != AGX_SIZE_32 ||
       agx_channels(data) != 1 || (data.value & 1) || data.value > 254 ||
       (!is_store && data.kill) || offset.type != AGX_INDEX_IMMEDIATE ||
       offset.memory || offset.abs || offset.neg || offset.cache ||
       offset.discard || offset.kill || offset.size != AGX_SIZE_16 ||
       agx_channels(offset) != 1 || (offset.value & 3))
      return false;

   const uint32_t tps_half = offset.value >> 1;
   if (tps_half + 2 > AGX3_SLOT_TPS_HALF_TOKENS)
      return false;

   if (is_store) {
      if (store == NULL)
         return false;
      *store = (struct agx3_static_spill_store_desc){
         .tps_half = tps_half,
         .src = data.value,
         .width = AGX3_MEM_W32,
         .src_kill = data.kill,
         .slot = slot,
         .wait_mask = wait_mask,
      };
   } else {
      if (load == NULL)
         return false;
      *load = (struct agx3_static_spill_load_desc){
         .dest = data.value,
         .tps_half = tps_half,
         .width = AGX3_MEM_W32,
         .slot = slot,
         .slot_mirror = slot,
         .wait_mask = wait_mask,
      };
   }

   return true;
}

static bool
agx3_append_static_spill_tokens(const agx_instr *I,
                                struct agx3_slot_instr *slot_instr)
{
   const bool is_store = I->op == AGX_OPCODE_STACK_STORE;
   struct agx3_static_spill_store_desc store;
   struct agx3_static_spill_load_desc load;
   if (!agx3_translate_static_spill(I, is_store, 1, 0, &store, &load))
      return false;

   const uint32_t tps_half = is_store ? store.tps_half : load.tps_half;
   for (unsigned i = 0; i < 2; ++i) {
      if (slot_instr->nr_srcs >= AGX3_SLOT_MAX_REGS ||
          slot_instr->nr_dsts >= AGX3_SLOT_MAX_REGS)
         return false;
      const uint16_t token =
         (uint16_t)(AGX3_SLOT_TPS_TOKEN_BASE + tps_half + i);
      slot_instr->src[slot_instr->nr_srcs++] = token;
      slot_instr->dst[slot_instr->nr_dsts++] = token;
   }
   return true;
}

/* Map a resolved selection op to the composer ALU op that can carry a wait.
 * Returns AGX3_ALU_NUM for ops with no wait-carrying form. */
static enum agx3_alu_op
agx3_sel_to_alu(const struct agx3_sel_instr *S)
{
   switch (S->op) {
   case AGX3_SEL_OP_MOV_IMM:
      return AGX3_ALU_LDIMM;
   case AGX3_SEL_OP_IADD:
      return AGX3_ALU_IADD;
   case AGX3_SEL_OP_ISUB:
      return AGX3_ALU_ISUB;
   case AGX3_SEL_OP_IMAD:
      return AGX3_ALU_IMADD;
   case AGX3_SEL_OP_IMUL:
      return AGX3_ALU_IMUL;
   case AGX3_SEL_OP_FADD:
      return AGX3_ALU_FADD;
   case AGX3_SEL_OP_FMUL:
      return AGX3_ALU_FMUL;
   case AGX3_SEL_OP_FFMA:
      return AGX3_ALU_FMA;
   case AGX3_SEL_OP_ASR:
      return AGX3_ALU_ASR;
   case AGX3_SEL_OP_SLL:
      return AGX3_ALU_SLL;
   case AGX3_SEL_OP_LSR:
      return AGX3_ALU_LSR;
   case AGX3_SEL_OP_F_RCP:
      return AGX3_ALU_RCP;
   case AGX3_SEL_OP_F_RSQRT:
      return AGX3_ALU_RSQRT;
   case AGX3_SEL_OP_F_RINT:
      return AGX3_ALU_RINT;
   case AGX3_SEL_OP_I32_TO_F:
      return AGX3_ALU_I2F;
   case AGX3_SEL_OP_U32_TO_F:
      return AGX3_ALU_U2F;
   case AGX3_SEL_OP_F_TO_I32:
      return AGX3_ALU_F2I;
   case AGX3_SEL_OP_BITOP:
      switch (S->truth_table & 0xfu) {
      case AGX3_SEL_TT_AND:
         return AGX3_ALU_BAND;
      case AGX3_SEL_TT_OR:
         return AGX3_ALU_BOR;
      case AGX3_SEL_TT_XOR:
         return AGX3_ALU_BXOR;
      case AGX3_SEL_TT_NOT:
         return AGX3_ALU_BNOT;
      default:
         return AGX3_ALU_NUM;
      }
   default:
      return AGX3_ALU_NUM; /* F_TO_U32, FMIN/FMAX: no wait-carrying form yet */
   }
}

/* Emit one selected ALU instruction, carrying a wait mask when needed. With no
 * wait it uses the compact hand packer; with a wait it uses the composer form,
 * which is byte-identical for wait 0. */
static bool
agx3_emit_alu(struct util_dynarray *emission, enum agx_isa_encoding enc,
              const struct agx3_sel_instr *S, uint16_t wait_mask)
{
   if (wait_mask == 0)
      return agx3_emit(emission, enc, S);

   const enum agx3_alu_op op = agx3_sel_to_alu(S);

   /*
    * Every wait-carrying form deposits its sources into register fields
    * (agx3_alu_table.h gives f_src0/f_src1/f_src2; only ldimm has an f_imm,
    * and it has no sources). agx3_sel_instr keeps an immediate's value in the
    * same `reg` slot a register index uses, so handing one to the composer
    * would encode the constant AS a register number: `iadd d, r0, #1` became
    * `iadd d, r0, r1`, reading whatever half 1 happened to hold. That was
    * silently wrong code, not a refusal.
    *
    * An instruction with an immediate source therefore skips the composer
    * entirely and takes the prelude path below, which retires the wait and
    * then emits the ordinary no-wait form through the select core -- where
    * the real immediate encodings live.
    */
   bool all_sources_registers = true;
   for (unsigned i = 0; i < S->nr_srcs; ++i)
      all_sources_registers &= S->src[i].is_reg &&
                               S->src[i].size == AGX3_SEL_SIZE_32;

   uint8_t bytes[AGX3_ALU_MAX_LENGTH];
   uint8_t len = 0;
   const uint16_t s0 = S->nr_srcs > 0 ? S->src[0].reg : 0;
   const uint16_t s1 = S->nr_srcs > 1 ? S->src[1].reg : 0;
   const uint16_t s2 = S->nr_srcs > 2 ? S->src[2].reg : 0;

   if (op != AGX3_ALU_NUM && all_sources_registers &&
       S->has_dest && S->dest_size == AGX3_SEL_SIZE_32 &&
       agx3_alu_compose(enc, op, S->dest, s0, s1, s2, S->imm, wait_mask,
                        bytes, &len)) {
      return agx3_append(emission, bytes, len);
   }

   /* Some compact ALU forms cannot carry the needed scoreboard wait. Retire
    * it with a synchronous prelude, then emit the ordinary no-wait form.
    * A fresh destination can take an ldimm; an in-place accumulator instead
    * uses bit_and(x, x), which preserves every bit of its live source.
    */
   if (!S->has_dest || S->dest_size != AGX_SIZE_32)
      return false;

   bool overlaps_source = false;
   bool exact_in_place = false;
   for (unsigned i = 0; i < S->nr_srcs; ++i) {
      if (!S->src[i].is_reg)
         continue;
      const unsigned dest_halves = agx_size_align_16(S->dest_size);
      const unsigned halves = agx_size_align_16(S->src[i].size);
      if (S->dest < S->src[i].reg + halves &&
          S->src[i].reg < S->dest + dest_halves) {
         overlaps_source = true;
         exact_in_place |=
            S->src[i].reg == S->dest && S->src[i].size == AGX_SIZE_32;
      }
   }

   if (overlaps_source &&
       (!exact_in_place || (wait_mask & (wait_mask - 1)) != 0))
      return false;

   while (wait_mask != 0) {
      const uint16_t prewait = wait_mask & -wait_mask;
      uint8_t prelude[AGX3_ALU_MAX_LENGTH];
      uint8_t prelude_len = 0;
      const enum agx3_alu_op prelude_op =
         exact_in_place ? AGX3_ALU_BAND : AGX3_ALU_LDIMM;
      if (!agx3_alu_compose(enc, prelude_op, S->dest, S->dest, S->dest, 0, 0,
                            prewait, prelude, &prelude_len) ||
          !agx3_append(emission, prelude, prelude_len))
         return false;
      wait_mask &= ~prewait;
   }

   /* The wait is retired; emit the plain form. Go through the select core
    * rather than the composer so that immediate operands use their own
    * encodings instead of being mistaken for register numbers. */
   return agx3_emit(emission, enc, S);
}

static bool
agx3_retire_slots_noop(struct util_dynarray *emission,
                       enum agx_isa_encoding enc, uint16_t mask);

static bool
agx3_subgroup_plain(agx_index index, enum agx_index_type type)
{
   /* cache/discard are post-RA performance hints, not value modifiers.  The
    * bounded subgroup forms either carry a pinned liveness profile or expose
    * the relevant hint explicitly below, so refusing a legal public subgroup
    * operation merely because its source became a last use is incorrect. */
   return index.type == type && !index.memory && index.size == AGX_SIZE_32 &&
          agx_channels(index) == 1 && !index.abs && !index.neg && !index.kill &&
          !(index.cache && index.discard) && (index.value & 1) == 0;
}

/* Preserve the older bounded immediate broadcast witnesses. */
static bool
agx3_emit_simd_broadcast(struct util_dynarray *emission,
                         enum agx_isa_encoding enc, const agx_instr *I,
                         uint16_t wait_mask)
{
   if (I->op != AGX_OPCODE_SHUFFLE || I->nr_dests != 1 || I->nr_srcs != 2 ||
       !agx3_subgroup_plain(I->dest[0], AGX_INDEX_REGISTER) ||
       !agx3_subgroup_plain(I->src[0], AGX_INDEX_REGISTER) ||
       I->src[1].type != AGX_INDEX_IMMEDIATE || I->src[1].memory ||
       I->src[1].abs || I->src[1].neg ||
       agx_channels(I->src[1]) != 1 || I->saturate ||
       (I->src[1].value != 0 && I->src[1].value != 3 &&
        I->src[1].value != 5))
      return false;

   /* G17P's immediate-lane form has direct lane 0/3/5 witnesses, including a
    * device-load producer on slot 6.  Route AGX3b through that exact bounded
    * encoder so a public subgroupShuffle may carry its real dependency. */
   if (enc == AGX_ISA_ENCODING_AGX3B) {
      const struct agx3_subgroup_shuffle_desc d = {
         .op = AGX3_SUBGROUP_SHUFFLE_BROADCAST,
         .lane_kind = AGX3_SUBGROUP_LANE_IMMEDIATE,
         .dest = I->dest[0].value,
         .src = I->src[0].value,
         .lane = I->src[1].value,
         .wait_mask = wait_mask,
         .dest_hint = I->dest[0].cache,
      };
      uint8_t bytes[AGX3_SUBGROUP_MAX_LENGTH], length = 0;
      return agx3_subgroup_encode_shuffle(enc, &d, bytes, &length) &&
             agx3_append(emission, bytes, length);
   }

   if (wait_mask != 0 || I->src[1].value == 5)
      return false;

   struct agx3_encoded_simd_broadcast_f32_imm encoded;
   return agx3_pack_simd_broadcast_f32_imm(
             enc, I->dest[0].value, I->src[0].value, I->src[1].value,
             &encoded) &&
          agx3_append(emission, encoded.bytes, encoded.length);
}

static bool
agx3_map_shuffle(unsigned op, bool *quad,
                 enum agx3_subgroup_shuffle_op *shuffle)
{
   *quad = false;
   switch (op) {
   case AGX_OPCODE_SHUFFLE:
      *shuffle = AGX3_SUBGROUP_SHUFFLE_BROADCAST;
      return true;
   case AGX_OPCODE_SHUFFLE_XOR:
      *shuffle = AGX3_SUBGROUP_SHUFFLE_XOR;
      return true;
   case AGX_OPCODE_SHUFFLE_UP:
      *shuffle = AGX3_SUBGROUP_SHUFFLE_UP;
      return true;
   case AGX_OPCODE_SHUFFLE_DOWN:
      *shuffle = AGX3_SUBGROUP_SHUFFLE_DOWN;
      return true;
   case AGX_OPCODE_QUAD_SHUFFLE:
      *quad = true;
      *shuffle = AGX3_SUBGROUP_SHUFFLE_BROADCAST;
      return true;
   case AGX_OPCODE_QUAD_SHUFFLE_XOR:
      *quad = true;
      *shuffle = AGX3_SUBGROUP_SHUFFLE_XOR;
      return true;
   case AGX_OPCODE_QUAD_SHUFFLE_UP:
      *quad = true;
      *shuffle = AGX3_SUBGROUP_SHUFFLE_UP;
      return true;
   case AGX_OPCODE_QUAD_SHUFFLE_DOWN:
      *quad = true;
      *shuffle = AGX3_SUBGROUP_SHUFFLE_DOWN;
      return true;
   default:
      return false;
   }
}

static bool
agx3_emit_subgroup_shuffle(struct util_dynarray *emission,
                           enum agx_isa_encoding enc, const agx_instr *I,
                           uint16_t wait_mask)
{
   bool quad;
   enum agx3_subgroup_shuffle_op op;
   if (!agx3_map_shuffle(I->op, &quad, &op))
      return false;

   if (I->op == AGX_OPCODE_SHUFFLE && I->nr_srcs == 2 &&
       I->src[1].type == AGX_INDEX_IMMEDIATE)
      return agx3_emit_simd_broadcast(emission, enc, I, wait_mask);

   if (quad && I->nr_srcs == 2 && I->src[1].type == AGX_INDEX_IMMEDIATE) {
      if (I->nr_dests != 1 || I->saturate ||
          !agx3_subgroup_plain(I->dest[0], AGX_INDEX_REGISTER) ||
          !agx3_subgroup_plain(I->src[0], AGX_INDEX_REGISTER) ||
          I->src[1].memory || I->src[1].abs || I->src[1].neg ||
          agx_channels(I->src[1]) != 1)
         return false;

      /* The native quad broadcast carries the device-load wait directly.
       * The witnessed quad butterfly words have no wait, so settle any
       * producer before emitting those byte-exact forms. */
      if (op != AGX3_SUBGROUP_SHUFFLE_BROADCAST && wait_mask != 0) {
         if (!agx3_retire_slots_noop(emission, enc, wait_mask))
            return false;
         wait_mask = 0;
      }

      const struct agx3_subgroup_shuffle_desc d = {
         .quad = true,
         .op = op,
         .lane_kind = AGX3_SUBGROUP_LANE_IMMEDIATE,
         .dest = I->dest[0].value,
         .src = I->src[0].value,
         .lane = I->src[1].value,
         .wait_mask = wait_mask,
         .dest_hint = I->dest[0].cache,
         .src_live = !I->src[0].discard,
      };
      uint8_t bytes[AGX3_SUBGROUP_MAX_LENGTH], length = 0;
      return agx3_subgroup_encode_shuffle(enc, &d, bytes, &length) &&
             agx3_append(emission, bytes, length);
   }

   if (I->nr_dests != 1 || I->nr_srcs != 2 || I->saturate ||
       !agx3_subgroup_plain(I->dest[0], AGX_INDEX_REGISTER) ||
       !agx3_subgroup_plain(I->src[0], AGX_INDEX_REGISTER))
      return false;

   enum agx3_subgroup_lane_kind lane_kind;
   if (agx3_subgroup_plain(I->src[1], AGX_INDEX_REGISTER))
      lane_kind = AGX3_SUBGROUP_LANE_REGISTER;
   else if (agx3_subgroup_plain(I->src[1], AGX_INDEX_UNIFORM))
      lane_kind = AGX3_SUBGROUP_LANE_UNIFORM;
   else
      return false;

   /* The register/uniform-lane native shuffle forms have no witnessed
    * scoreboard field.  Public shuffles commonly consume values directly
    * from an asynchronous SSBO/device load, so retire only their concrete
    * producer slots through the same synchronous identity prelude used by
    * other no-wait subgroup forms. */
   if (wait_mask != 0 &&
       !agx3_retire_slots_noop(emission, enc, wait_mask))
      return false;

   const struct agx3_subgroup_shuffle_desc d = {
      .quad = quad,
      .op = op,
      .lane_kind = lane_kind,
      .dest = I->dest[0].value,
      .src = I->src[0].value,
      .lane = I->src[1].value,
      .wait_mask = 0,
   };
   uint8_t bytes[AGX3_SUBGROUP_MAX_LENGTH], length = 0;
   return agx3_subgroup_encode_shuffle(enc, &d, bytes, &length) &&
          agx3_append(emission, bytes, length);
}

static bool
agx3_map_reduce_op(enum agx_simd_op in, enum agx3_subgroup_reduce_op *out)
{
   switch (in) {
   case AGX_SIMD_OP_AND:  *out = AGX3_SUBGROUP_REDUCE_AND; return true;
   case AGX_SIMD_OP_FADD: *out = AGX3_SUBGROUP_REDUCE_FADD; return true;
   case AGX_SIMD_OP_OR:   *out = AGX3_SUBGROUP_REDUCE_OR; return true;
   case AGX_SIMD_OP_FMUL: *out = AGX3_SUBGROUP_REDUCE_FMUL; return true;
   case AGX_SIMD_OP_XOR:  *out = AGX3_SUBGROUP_REDUCE_XOR; return true;
   case AGX_SIMD_OP_FMIN: *out = AGX3_SUBGROUP_REDUCE_FMIN; return true;
   case AGX_SIMD_OP_FMAX: *out = AGX3_SUBGROUP_REDUCE_FMAX; return true;
   case AGX_SIMD_OP_IADD: *out = AGX3_SUBGROUP_REDUCE_IADD; return true;
   case AGX_SIMD_OP_SMIN: *out = AGX3_SUBGROUP_REDUCE_SMIN; return true;
   case AGX_SIMD_OP_SMAX: *out = AGX3_SUBGROUP_REDUCE_SMAX; return true;
   case AGX_SIMD_OP_UMIN: *out = AGX3_SUBGROUP_REDUCE_UMIN; return true;
   case AGX_SIMD_OP_UMAX: *out = AGX3_SUBGROUP_REDUCE_UMAX; return true;
   default: return false;
   }
}

static bool
agx3_emit_subgroup_reduce(struct util_dynarray *emission,
                          enum agx_isa_encoding enc, const agx_instr *I,
                          uint16_t wait_mask)
{
   const bool quad = I->op == AGX_OPCODE_QUAD_REDUCE ||
                     I->op == AGX_OPCODE_QUAD_PREFIX;
   const bool prefix = I->op == AGX_OPCODE_SIMD_PREFIX ||
                       I->op == AGX_OPCODE_QUAD_PREFIX;
   if (I->nr_dests != 1 || I->nr_srcs != 1 || I->saturate ||
       !agx3_subgroup_plain(I->dest[0], AGX_INDEX_REGISTER) ||
       !agx3_subgroup_plain(I->src[0], AGX_INDEX_REGISTER))
      return false;

   enum agx3_subgroup_reduce_op op;
   if (!agx3_map_reduce_op(I->simd_op, &op))
      return false;
   const struct agx3_subgroup_reduce_desc d = {
      .quad = quad,
      .prefix = prefix,
      .op = op,
      .dest = I->dest[0].value,
      .src = I->src[0].value,
      .wait_mask = wait_mask,
      .dest_hint = I->dest[0].cache,
      .src_live = !I->src[0].discard,
   };
   uint8_t bytes[AGX3_SUBGROUP_MAX_LENGTH], length = 0;
   return agx3_subgroup_encode_reduce(enc, &d, bytes, &length) &&
          agx3_append(emission, bytes, length);
}

static bool
agx3_emit_subgroup_ballot(struct util_dynarray *emission,
                          enum agx_isa_encoding enc, const agx_instr *I,
                          uint16_t wait_mask)
{
   const bool is_float = I->op == AGX_OPCODE_FCMP_BALLOT ||
                         I->op == AGX_OPCODE_FCMP_QUAD_BALLOT;
   const bool quad = I->op == AGX_OPCODE_ICMP_QUAD_BALLOT ||
                     I->op == AGX_OPCODE_FCMP_QUAD_BALLOT;
   if (I->nr_dests != 1 || I->nr_srcs != 2 || I->saturate ||
       !agx3_subgroup_plain(I->dest[0], AGX_INDEX_REGISTER) ||
       !agx3_subgroup_plain(I->src[0], AGX_INDEX_REGISTER) ||
       I->src[1].type != AGX_INDEX_IMMEDIATE ||
       I->src[1].memory || I->src[1].abs || I->src[1].neg ||
       I->src[1].cache || I->src[1].discard || I->src[1].kill)
      return false;

   bool compare_immediate = false;
   uint32_t compare_value = I->src[1].value;
   uint8_t condition;
   if (is_float) {
      if (compare_value != 0)
         return false;
      condition = I->fcond;
   } else {
      switch (I->icond) {
      case AGX_ICOND_UEQ:
         /* Apple's immediate equality form is grounded for the direct
          * predicate.  Keep != 0 on the older register-zero form, whose
          * inverted encoding is independently witnessed. */
         compare_immediate = !I->invert_cond;
         condition = 4;
         break;
      case AGX_ICOND_SEQ:
         if (compare_value != 0)
            return false;
         condition = 4;
         break;
      case AGX_ICOND_SLT:
         if (compare_value != 0)
            return false;
         condition = 5;
         break;
      case AGX_ICOND_SGT:
         if (compare_value != 0)
            return false;
         condition = 6;
         break;
      case AGX_ICOND_ULT:
         compare_immediate = true;
         if (I->invert_cond) {
            /* !(x < N) is x > N - 1.  Apple emits this canonical native
             * form for subgroup votes, and the native instruction has no
             * need for an invert bit. */
            if (compare_value == 0)
               return false;
            compare_value--;
            condition = AGX_ICOND_UGT;
         } else {
            condition = AGX_ICOND_ULT;
         }
         break;
      case AGX_ICOND_UGT:
         compare_immediate = true;
         if (I->invert_cond) {
            /* !(x > N) is x < N + 1. */
            if (compare_value == UINT16_MAX)
               return false;
            compare_value++;
            condition = AGX_ICOND_ULT;
         } else {
            condition = AGX_ICOND_UGT;
         }
         break;
      default: return false;
      }
      if (compare_immediate && compare_value > UINT16_MAX)
         return false;
   }

   /* The native ballot itself has no witnessed scoreboard field.  A ballot
    * predicate may still come directly from an asynchronous SSBO/device
    * load, so retire those exact producer slots with the existing synchronous
    * identity prelude before issuing the unchanged no-wait ballot. */
   if (wait_mask != 0 &&
       !agx3_retire_slots_noop(emission, enc, wait_mask))
      return false;

   const struct agx3_subgroup_ballot_desc d = {
      .quad = quad,
      .type = is_float ? AGX3_SUBGROUP_BALLOT_FLOAT
                       : AGX3_SUBGROUP_BALLOT_INTEGER,
      .dest = I->dest[0].value,
      .src = I->src[0].value,
      .compare_immediate = compare_immediate,
      .compare_value = compare_value,
      .dest_cache = I->dest[0].cache,
      .src_live = !I->src[0].discard,
      .condition = condition,
      .invert = compare_immediate ? false : I->invert_cond,
      .wait_mask = 0,
   };
   uint8_t bytes[AGX3_SUBGROUP_MAX_LENGTH], length = 0;
   return agx3_subgroup_encode_ballot(enc, &d, bytes, &length) &&
          agx3_append(emission, bytes, length);
}

static bool
agx3_emit_scoped_barrier(struct util_dynarray *emission,
                         enum agx_isa_encoding enc, const agx_instr *I,
                         uint16_t wait_mask)
{
   if (wait_mask != 0 || I->op != AGX_OPCODE_SCOPED_BARRIER_AGX3 ||
       I->nr_dests != 0 || I->nr_srcs != 0 ||
       I->imm > AGX3_BARRIER_SCOPE_TEXTURE)
      return false;

   struct agx3_encoded_scoped_barrier encoded;
   return agx3_pack_scoped_barrier(enc, I->imm, &encoded) &&
          agx3_append(emission, encoded.bytes, encoded.length);
}

/* The grounded AGX3 compare forms are 32-bit and address the lower 128
 * half-registers. Source modifiers and memory/uniform operands select forms
 * whose routing has not been proved. */
static bool
agx3_cf_reg32(agx_index src)
{
   return agx3_src_is_register(src) && !src.memory && src.size == AGX_SIZE_32 &&
          (src.value & 1) == 0 && src.value < 128 && !src.abs && !src.neg;
}

/* Native 16-bit register comparison addresses all 256 individual halves. */
static bool
agx3_cf_reg16(agx_index src)
{
   return agx3_src_is_register(src) && !src.memory && src.size == AGX_SIZE_16 &&
          src.value < 256 && !src.abs && !src.neg;
}

static bool
agx3_cf_map_icond(enum agx_icond in, enum agx3_cf_icond *out)
{
   switch (in) {
   case AGX_ICOND_UEQ:
   case AGX_ICOND_SEQ:
      *out = AGX3_CF_ICOND_EQ;
      return true;
   case AGX_ICOND_ULT:
      *out = AGX3_CF_ICOND_ULT;
      return true;
   case AGX_ICOND_UGT:
      *out = AGX3_CF_ICOND_UGT;
      return true;
   case AGX_ICOND_SLT:
      *out = AGX3_CF_ICOND_SLT;
      return true;
   case AGX_ICOND_SGT:
      *out = AGX3_CF_ICOND_SGT;
      return true;
   }

   return false;
}

/* Canonicalize `immediate relation register` to the already-grounded
 * `register swapped-relation immediate` encoding.  Equality is symmetric;
 * the four ordered relations exchange LT and GT without changing signedness. */
static bool
agx3_cf_swap_icond(enum agx3_cf_icond *cond)
{
   switch (*cond) {
   case AGX3_CF_ICOND_EQ:
      return true;
   case AGX3_CF_ICOND_ULT:
      *cond = AGX3_CF_ICOND_UGT;
      return true;
   case AGX3_CF_ICOND_UGT:
      *cond = AGX3_CF_ICOND_ULT;
      return true;
   case AGX3_CF_ICOND_SLT:
      *cond = AGX3_CF_ICOND_SGT;
      return true;
   case AGX3_CF_ICOND_SGT:
      *cond = AGX3_CF_ICOND_SLT;
      return true;
   }

   return false;
}

/* Normalize Mesa's seven float relations to AGX3's three base conditions,
 * or-NaN selector, and the consuming cnd instruction's invert bit. */
static bool
agx3_cf_map_fcond(enum agx_fcond in, enum agx3_cf_fcond *out, bool *or_nan,
                  bool *invert)
{
   *or_nan = false;

   switch (in) {
   case AGX_FCOND_EQ:
      *out = AGX3_CF_FCOND_EQ;
      return true;
   case AGX_FCOND_LT:
      *out = AGX3_CF_FCOND_LT;
      return true;
   case AGX_FCOND_GT:
      *out = AGX3_CF_FCOND_GT;
      return true;
   case AGX_FCOND_LTN:
      *out = AGX3_CF_FCOND_LT;
      *or_nan = true;
      return true;
   case AGX_FCOND_GTN:
      *out = AGX3_CF_FCOND_GT;
      *or_nan = true;
      return true;
   case AGX_FCOND_GE:
      *out = AGX3_CF_FCOND_LT;
      *or_nan = true;
      *invert = !*invert;
      return true;
   case AGX_FCOND_LE:
      *out = AGX3_CF_FCOND_GT;
      *or_nan = true;
      *invert = !*invert;
      return true;
   }

   return false;
}

static bool
agx3_cf_is_zero(agx_index src)
{
   return src.type == AGX_INDEX_IMMEDIATE && src.value == 0 && !src.abs &&
          !src.neg;
}

/* A constant the witnessed 16-bit compare can carry. Above 255 Apple changes
 * the instruction, so anything wider stays refused. */
static bool
agx3_cf_imm8(agx_index src)
{
   return src.type == AGX_INDEX_IMMEDIATE && src.value <= UINT8_MAX &&
          !src.abs && !src.neg;
}

/* Emit the compare half of a fused AGX IR control-flow instruction. The flag
 * is consumed by the immediately following cnd instruction. Loop predicates
 * use distinct persistent slots by nesting depth; ordinary ifs reuse slot 0. */
static bool
agx3_emit_cf_compare(struct util_dynarray *emission, enum agx_isa_encoding enc,
                     const agx_instr *I, bool is_float, unsigned flag_slot,
                     bool *invert)
{
   uint8_t bytes[AGX3_CF_MAX_LENGTH];
   uint8_t len = 0;

   if (I->nr_srcs != 2)
      return false;

   if (!is_float && agx3_cf_reg16(I->src[0]) && agx3_cf_reg16(I->src[1]) &&
       agx_channels(I->src[0]) == 1 && agx_channels(I->src[1]) == 1) {
      enum agx3_cf_icond condition;
      const uint16_t src[] = {I->src[0].value, I->src[1].value};
      struct agx3_cf_source_hints hints[] = {
         {I->src[0].cache, I->src[0].discard},
         {I->src[1].cache, I->src[1].discard},
      };

      /* agx_lower_pseudo spells an unconditional G17P loop break as the
       * integer identity r0l == r0l.  Register allocation may mark both
       * occurrences cached and killed because the fixed architectural half
       * has no SSA definition.  Native half comparisons ground the live/live
       * register form, and cache/discard are only operand lifetime hints, so
       * normalize this compiler-generated identity back to that form.  Keep
       * every non-identity comparison's independently allocated hints. */
      const bool unconditional_break =
         I->op == AGX_OPCODE_BREAK_IF_ICMP && !I->invert_cond &&
         (I->icond == AGX_ICOND_UEQ || I->icond == AGX_ICOND_SEQ) &&
         I->src[0].type == I->src[1].type &&
         I->src[0].value == I->src[1].value &&
         I->src[0].size == I->src[1].size;
      if (unconditional_break) {
         hints[0] = (struct agx3_cf_source_hints){false, false};
         hints[1] = (struct agx3_cf_source_hints){false, false};
      }
      if ((!unconditional_break && (I->src[0].kill || I->src[1].kill)) ||
          !agx3_cf_map_icond(I->icond, &condition) ||
          !agx3_cf_icmp_reg16(enc, condition, flag_slot, src, hints, 0, bytes,
                              &len))
         return false;
      return agx3_append(emission, bytes, len);
   }

   /* Branching on a 16-bit integer against a constant. Mesa keeps booleans
    * and small ints in 16-bit halves, so `if (b)` arrives as `r0h != 0` and
    * `if (gradientLength < 2)` as `r0h < 2` rather than as 32-bit compares.
    *
    * The constant arrives on either side. `0 < x` and `x > 0` are the same
    * branch, and which one Mesa hands the packer depends only on how the
    * source wrote the comparison -- Hyprland's border, shadow and inner-glow
    * shaders produce both orders. Canonicalize the constant to the right
    * exactly as the 32-bit path below already does; the emitted bytes are
    * the witnessed encoding either way, with the relation swapped, so the
    * normalization admits no new encoding of its own.
    *
    * agx3_cf_icmp_imm16_runtime() refuses the immediates and relations that
    * have no witness, so an ungrounded constant still fails closed here. */
   if (!is_float &&
       ((agx3_cf_reg16(I->src[0]) && agx3_cf_imm8(I->src[1])) ||
        (agx3_cf_imm8(I->src[0]) && agx3_cf_reg16(I->src[1])))) {
      const bool imm_first = !agx3_cf_reg16(I->src[0]);
      const agx_index source = I->src[imm_first ? 1 : 0];
      const agx_index constant = I->src[imm_first ? 0 : 1];
      enum agx3_cf_icond cond16;
      if (source.kill || !agx3_cf_map_icond(I->icond, &cond16) ||
          (imm_first && !agx3_cf_swap_icond(&cond16)) ||
          !agx3_cf_icmp_imm16_runtime(enc, cond16, flag_slot, source.value,
             (uint8_t)constant.value,
             (struct agx3_cf_source_hints){source.cache, source.discard},
             0, bytes, &len))
         return false;
      return agx3_append(emission, bytes, len);
   }

   if (is_float && enc == AGX_ISA_ENCODING_AGX3B &&
       ((agx3_cf_is_zero(I->src[1]) && (agx3_vary_reg32(I->src[0]) || agx3_cf_reg16(I->src[0]))) ||
        (agx3_cf_is_zero(I->src[0]) && (agx3_vary_reg32(I->src[1]) || agx3_cf_reg16(I->src[1]))))) {
      const bool zero_first=agx3_cf_is_zero(I->src[0]);
      const agx_index source=I->src[zero_first?1:0];
      enum agx_fcond relation=I->fcond;
      if(zero_first) {
         switch(relation) {
         case AGX_FCOND_LT:relation=AGX_FCOND_GT;break;
         case AGX_FCOND_GT:relation=AGX_FCOND_LT;break;
         case AGX_FCOND_LE:relation=AGX_FCOND_GE;break;
         case AGX_FCOND_GE:relation=AGX_FCOND_LE;break;
         case AGX_FCOND_LTN:relation=AGX_FCOND_GTN;break;
         case AGX_FCOND_GTN:relation=AGX_FCOND_LTN;break;
         default:break;
         }
      }
      enum agx3_cf_fcond condition;bool or_nan;
      if(source.kill || agx_channels(source)!=1 ||
         !agx3_cf_map_fcond(relation,&condition,&or_nan,invert) ||
         !agx3_cf_fcmp_zero(enc,condition,or_nan,flag_slot,source.value,source.size==AGX_SIZE_16,
            (struct agx3_cf_source_hints){source.cache,source.discard},0,bytes,&len))
         return false;
      return agx3_append(emission,bytes,len);
   }

   if (is_float && enc == AGX_ISA_ENCODING_AGX3B &&
       agx3_src_is_register(I->src[0]) &&
       agx3_src_is_register(I->src[1]) &&
       I->src[0].size == AGX_SIZE_32 &&
       I->src[1].size == AGX_SIZE_32 &&
       agx_channels(I->src[0]) == 1 && agx_channels(I->src[1]) == 1 &&
       !I->src[0].memory && !I->src[1].memory &&
       !I->src[0].abs && !I->src[1].abs &&
       (I->src[0].neg != I->src[1].neg) &&
       !(I->src[0].cache && I->src[0].discard) && !I->src[0].kill &&
       !(I->src[1].cache && I->src[1].discard) && !I->src[1].kill) {
      enum agx3_cf_fcond cond;
      bool or_nan;
      const uint16_t src[] = {I->src[0].value, I->src[1].value};
      const bool neg[] = {I->src[0].neg, I->src[1].neg};
      const struct agx3_cf_source_hints hints[] = {
         {I->src[0].cache, I->src[0].discard},
         {I->src[1].cache, I->src[1].discard}};
      if (!agx3_cf_map_fcond(I->fcond, &cond, &or_nan, invert) ||
          !agx3_cf_fcmp_reg_neg(enc, cond, or_nan, flag_slot, src, neg, hints,
                                bytes, &len))
         return false;
      return agx3_append(emission, bytes, len);
   }

   bool ok;
   if (is_float) {
      if (!agx3_cf_reg32(I->src[0]))
         return false;
      enum agx3_cf_fcond cond;
      bool or_nan;
      if (!agx3_cf_reg32(I->src[1]) ||
          !agx3_cf_map_fcond(I->fcond, &cond, &or_nan, invert))
         return false;

      ok = agx3_cf_fcmp_reg(enc, cond, or_nan, flag_slot, I->src[0].value,
                            I->src[1].value, bytes, &len);
   } else {
      enum agx3_cf_icond cond;
      if (!agx3_cf_map_icond(I->icond, &cond))
         return false;

      agx_index lhs = I->src[0];
      agx_index rhs = I->src[1];
      if (lhs.type == AGX_INDEX_IMMEDIATE && !lhs.abs && !lhs.neg &&
          lhs.value <= UINT8_MAX && agx3_cf_reg32(rhs)) {
         agx_index temporary = lhs;
         lhs = rhs;
         rhs = temporary;
         if (!agx3_cf_swap_icond(&cond))
            return false;
      }

      if (!agx3_cf_reg32(lhs))
         return false;
      if (rhs.type == AGX_INDEX_IMMEDIATE && !rhs.abs && !rhs.neg &&
          rhs.value <= UINT8_MAX) {
         ok = agx3_cf_icmp_imm(enc, cond, flag_slot, lhs.value, rhs.value,
                               bytes, &len);
      } else if (agx3_cf_reg32(rhs)) {
         ok = agx3_cf_icmp_reg(enc, cond, flag_slot, lhs.value, rhs.value,
                               bytes, &len);
      } else {
         return false;
      }
   }

   return ok && agx3_append(emission, bytes, len);
}

static bool
agx3_emit_cnd(struct util_dynarray *emission, enum agx_isa_encoding enc,
              unsigned op, unsigned flag_slot, bool invert, unsigned count)
{
   uint8_t bytes[AGX3_CF_MAX_LENGTH];
   uint8_t len = 0;
   bool ok;

   switch (op) {
   case AGX_OPCODE_IF_ICMP:
   case AGX_OPCODE_IF_FCMP:
      ok = agx3_cf_cndst(enc, flag_slot, invert, count, bytes, &len);
      break;
   case AGX_OPCODE_WHILE_ICMP:
   case AGX_OPCODE_WHILE_FCMP:
   case AGX_OPCODE_BREAK_IF_ICMP:
   case AGX_OPCODE_BREAK_IF_FCMP:
      ok = agx3_cf_cndlt(enc, flag_slot, invert, bytes, &len);
      break;
   case AGX_OPCODE_ELSE_ICMP:
   case AGX_OPCODE_ELSE_FCMP:
      ok = agx3_cf_cndef(enc, bytes, &len);
      break;
   case AGX_OPCODE_POP_EXEC:
      ok = agx3_cf_cndend(enc, count, bytes, &len);
      break;
   default:
      return false;
   }

   return ok && agx3_append(emission, bytes, len);
}

/* Returns 1 when I is a grounded control-flow instruction and was emitted,
 * zero when it is not control flow, and -1 when it is control flow but falls
 * outside the grounded domain. */
/*
 * Retire scoreboard slots ahead of a control-flow boundary.
 *
 * agx3_slot_assign() is a linear pass, so its "waited once = satisfied
 * globally" rule is only sound when every path into a block sees the same
 * scoreboard. The sufficient condition is an empty one. A result that is
 * still in flight at a divergence would otherwise have its wait land inside
 * ONE arm -- the atomic lowering does exactly this, hoisting the descriptor
 * load above the branch and consuming it in the taken arm -- and the other
 * arm would then read it unwaited.
 *
 * Waiting early is always semantically safe (it only costs latency), so
 * rather than refusing, retire everything here. `band(r0, r0, r0)` is a true
 * no-op -- x & x == x preserves every bit -- and it is the same wait-carrying
 * form agx3_emit_alu() already uses as a prelude. Writing r0 is safe even if
 * r0 is an in-flight load's destination, because the wait retires before the
 * write, so the band stores back exactly what the load delivered.
 *
 * Emits one instruction per slot, matching the witnessed prelude usage.
 */
static bool
agx3_retire_slots_noop(struct util_dynarray *emission, enum agx_isa_encoding enc,
                       uint16_t mask)
{
   while (mask != 0) {
      const uint16_t one = mask & (uint16_t)-(int16_t)mask;
      uint8_t bytes[AGX3_ALU_MAX_LENGTH];
      uint8_t len = 0;
      if (!agx3_alu_compose(enc, AGX3_ALU_BAND, 0, 0, 0, 0, 0, one, bytes,
                            &len) ||
          !agx3_append(emission, bytes, len))
         return false;
      mask &= (uint16_t)~one;
   }
   return true;
}

/* The instructions agx3_try_control_flow() claims. */
static bool
agx3_is_cf_op(const agx_instr *I)
{
   switch (I->op) {
   case AGX_OPCODE_BEGIN_CF:
   case AGX_OPCODE_IF_ICMP:
   case AGX_OPCODE_IF_FCMP:
   case AGX_OPCODE_WHILE_ICMP:
   case AGX_OPCODE_WHILE_FCMP:
   case AGX_OPCODE_BREAK_IF_ICMP:
   case AGX_OPCODE_BREAK_IF_FCMP:
   case AGX_OPCODE_ELSE_ICMP:
   case AGX_OPCODE_ELSE_FCMP:
   case AGX_OPCODE_POP_EXEC:
   case AGX_OPCODE_JMP_EXEC_ANY:
   case AGX_OPCODE_JMP_EXEC_NONE:
   case AGX_OPCODE_JMP_EXEC_NONE_AFTER:
      return true;
   default:
      return false;
   }
}

static bool
agx3_is_nested_loop_push(const agx_instr *I)
{
   return I->op == AGX_OPCODE_IF_FCMP && I->target == NULL && I->nest == 1 &&
          I->fcond == AGX_FCOND_EQ && !I->invert_cond && I->nr_srcs == 2 &&
          agx3_cf_is_zero(I->src[0]) && agx3_cf_is_zero(I->src[1]);
}

static int
agx3_try_control_flow(struct util_dynarray *emission,
                      struct util_dynarray *fixups, enum agx_isa_encoding enc,
                      const agx_instr *I, uint16_t wait_mask,
                      uint16_t outstanding, bool after_loop_backedge,
                      unsigned loop_depth)
{
   const bool is_if =
      I->op == AGX_OPCODE_IF_ICMP || I->op == AGX_OPCODE_IF_FCMP;
   const bool is_while =
      I->op == AGX_OPCODE_WHILE_ICMP || I->op == AGX_OPCODE_WHILE_FCMP;
   const bool is_break_if =
      I->op == AGX_OPCODE_BREAK_IF_ICMP || I->op == AGX_OPCODE_BREAK_IF_FCMP;
   const bool is_else =
      I->op == AGX_OPCODE_ELSE_ICMP || I->op == AGX_OPCODE_ELSE_FCMP;
   const bool is_jump = I->op == AGX_OPCODE_JMP_EXEC_ANY ||
                        I->op == AGX_OPCODE_JMP_EXEC_NONE ||
                        I->op == AGX_OPCODE_JMP_EXEC_NONE_AFTER;

   /*
    * Slot assignment is deliberately linear, so a control-flow boundary is
    * only safe when the scoreboard state entering it is the same on every
    * path. The sufficient condition is that nothing is in flight: if every
    * slot issued so far has already been waited, then every block is entered
    * with an empty scoreboard no matter which predecessor ran, and the linear
    * pass's "waited once = satisfied globally" reasoning never depends on a
    * wait that some other path skipped. A result still outstanding here would
    * carry its slot across the branch -- including around a backedge -- and
    * that needs the predecessor-aware analysis this backend does not have, so
    * it stays refused.
    *
    * This replaces a sticky "an async op appeared anywhere earlier" flag,
    * which refused shaders whose loads were all consumed before the branch --
    * the common case, and the one every real kernel with a bounds check hits.
    */
   if (outstanding != 0 &&
       (I->op == AGX_OPCODE_BEGIN_CF || is_if || is_while || is_break_if ||
        is_else || I->op == AGX_OPCODE_POP_EXEC || is_jump))
      return -1;

   if (I->op == AGX_OPCODE_BEGIN_CF)
      return wait_mask == 0 ? 1 : -1; /* architectural state: zero bytes */

   if (agx3_is_nested_loop_push(I)) {
      /* agx_push_exec(1), used before a nested loop, becomes Apple's
       * unconditional TRUE-slot loop entry and opens two architectural
       * exec-mask levels. No compare is emitted. */
      return wait_mask == 0 && agx3_emit_cnd(emission, enc, I->op, 6, false, 2)
                ? 1
                : -1;
   }

   if (is_if || is_while || is_break_if) {
      /* The all-clear compare/cnd forms are witnessed only for register-ready
       * inputs.  A local dependency wait also keeps memory-dependent control
       * flow fail-closed. */
      if (wait_mask != 0)
         return -1;
      /* CNDLT's native operation owns two architectural exec-mask levels.
       * Mesa's ordinary fused loop break arrives with nest=1; a break from
       * the loop's final nested block arrives with nest=2 and uses the same
       * compare+CNDLT transition before the shared backedge/CNDEND(2) footer.
       * Deeper exits need an independently grounded unwind sequence. */
      if ((is_if && I->nest != 1) ||
          (is_break_if && (I->nest == 0 || I->nest > 2)))
         return -1;

      bool invert = I->invert_cond;
      const bool is_float = I->op == AGX_OPCODE_IF_FCMP ||
                            I->op == AGX_OPCODE_WHILE_FCMP ||
                            I->op == AGX_OPCODE_BREAK_IF_FCMP;
      /* CNDLT predicates are persistent across a backedge. Apple assigns
       * outer through inner loop predicates to successive flag slots; NEST
       * is the number of exec-mask levels a break exits and is not this
       * depth. The linear pack walk tracks each native nested-loop push and
       * its matching backedge pop. */
      const unsigned flag_slot = is_break_if || is_while ? loop_depth - 1 : 0;
      if (flag_slot >= 6 ||
          !agx3_emit_cf_compare(emission, enc, I, is_float, flag_slot,
                                &invert))
         return -1;
      /* CNDLT retains lanes for which its consumed predicate is true. A
       * WHILE predicate already means "continue", whereas BREAK_IF names
       * the complementary lanes that must leave the loop. */
      if (is_break_if)
         invert = !invert;
      if (!agx3_emit_cnd(emission, enc, I->op, flag_slot, invert,
                         is_if ? 1 : 2))
         return -1;
      return 1;
   }

   if (is_else) {
      /* The only witnessed else is the canonical unconditional
       * else_fcmp(0, 0, eq, nest=1) emitted by Mesa's structurizer. */
      if (wait_mask != 0 || I->op != AGX_OPCODE_ELSE_FCMP || I->nest != 1 ||
          I->fcond != AGX_FCOND_EQ || I->invert_cond || I->nr_srcs != 2 ||
          !agx3_cf_is_zero(I->src[0]) || !agx3_cf_is_zero(I->src[1]))
         return -1;
      return agx3_emit_cnd(emission, enc, I->op, 0, false, 1) ? 1 : -1;
   }

   if (I->op == AGX_OPCODE_POP_EXEC) {
      /* Mesa's loop footer is jmp_exec_any + pop_exec(1), while AGX3's loop
       * entry/test opens two architectural levels and therefore pops two. */
      const unsigned count = after_loop_backedge ? 2 : I->nest;
      return wait_mask == 0 &&
                   agx3_emit_cnd(emission, enc, I->op, 0, false, count)
                ? 1
                : -1;
   }

   if (is_jump) {
      if (wait_mask != 0 || I->target == NULL)
         return -1;

      const bool is_bn = I->op != AGX_OPCODE_JMP_EXEC_ANY;
      struct agx3_branch_fixup fixup = {
         .block = I->target,
         .offset = emission->size,
         .skip_to_end = I->op == AGX_OPCODE_JMP_EXEC_NONE_AFTER,
         .is_bn = is_bn,
      };
      util_dynarray_append(fixups, fixup);

      uint8_t bytes[AGX3_CF_MAX_LENGTH];
      uint8_t len = 0;
      return agx3_cf_branch(enc, is_bn, bytes, &len) &&
                   agx3_append(emission, bytes, len)
                ? 1
                : -1;
   }

   /* An unconditional break is deliberately not included: no AGX3 witness
    * exists, and agx_lower_pseudo preserves it so it reaches this refusal. */
   if (I->op == AGX_OPCODE_BREAK)
      return -1;

   return 0;
}

static bool
agx3_fixup_branch(struct util_dynarray *emission, enum agx_isa_encoding enc,
                  struct agx3_branch_fixup fixup)
{
   const off_t target =
      fixup.skip_to_end ? fixup.block->last_offset : fixup.block->offset;
   const int64_t displacement = (int64_t)target - (int64_t)fixup.offset;

   /* Only Apple's witnessed directions are accepted: bn skips forward and b
    * is a backwards loop edge. This also rejects zero-displacement branches. */
   if ((fixup.is_bn && displacement <= 0) ||
       (!fixup.is_bn && displacement >= 0))
      return false;
   if (fixup.offset < 0 ||
       (uint64_t)fixup.offset + AGX3_CF_MAX_LENGTH > emission->size)
      return false;

   uint8_t *branch = (uint8_t *)emission->data + fixup.offset;
   return agx3_cf_patch_branch(enc, fixup.is_bn, branch, displacement);
}

/*
 * Apple's complete G17P float2-position shader uses a protocol that is not
 * selected by the generic round-robin scoreboard pass: its sole attribute
 * LDDEV carries f79=2 and signals slot 6. Keep its index last-use exception
 * fenced to the witnessed post-RA shape. All UVSWs, including this shader's,
 * require explicit staging producers; copying their one-hot wait masks alone
 * does not produce the data they consume.
 */
static bool
agx3_is_g17p_float2_position_vs(agx_context *ctx,
                                enum agx_isa_encoding enc)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || ctx->stage != MESA_SHADER_VERTEX)
      return false;

   unsigned instructions = 0, vertex_ids = 0, loads = 0, stores = 0;
   uint8_t uvs_mask = 0;

   agx_foreach_block(ctx, block) {
      agx_foreach_instr_in_block(block, I) {
         if (I->op == AGX_OPCODE_STOP)
            continue;
         instructions++;

         switch (I->op) {
         case AGX_OPCODE_GET_SR:
            if (I->sr != AGX_SR_VERTEX_ID)
               return false;
            vertex_ids++;
            break;
         case AGX_OPCODE_DEVICE_LOAD:
            if (I->nr_dests != 1 || I->nr_srcs != 2 || I->mask != 3 ||
                I->format != AGX_FORMAT_I32 ||
                I->dest[0].type != AGX_INDEX_REGISTER ||
                I->dest[0].size != AGX_SIZE_32 ||
                agx_channels(I->dest[0]) != 2 ||
                I->src[0].type != AGX_INDEX_UNIFORM ||
                I->src[0].size != AGX_SIZE_64 || I->src[0].value != 0 ||
                I->src[1].type != AGX_INDEX_REGISTER ||
                I->src[1].size != AGX_SIZE_32)
               return false;
            loads++;
            break;
         case AGX_OPCODE_ST_VARY:
            if (I->nr_dests != 0 || I->nr_srcs != 2 ||
                I->src[0].type != AGX_INDEX_IMMEDIATE ||
                I->src[0].value > 3 ||
                (uvs_mask & (1u << I->src[0].value)))
               return false;
            uvs_mask |= 1u << I->src[0].value;
            stores++;
            break;
         case AGX_OPCODE_MOV_IMM:
            if (I->imm != 0 && I->imm != 0x3f800000)
               return false;
            break;
         default:
            return false;
         }
      }
   }

   return instructions == 8 && vertex_ids == 1 && loads == 1 && stores == 4 &&
          uvs_mask == 0xf;
}

/* Native 11/16/32/36-output VSs use a dense set of scalar UVSW indices and
 * distinct staging addresses. The stores may be grouped by varying rather
 * than sorted by scalar index. Recycle the six scoreboard slots after waiting
 * for earlier writes; keep the four-position golden independent. */
static unsigned
agx3_g17p_vertex_output_count(agx_context *ctx, enum agx_isa_encoding enc)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || ctx->stage != MESA_SHADER_VERTEX ||
       ctx->is_preamble)
      return 0;

   agx_block *output_block = NULL;
   agx_foreach_block(ctx, block)
      output_block = block;
   uint64_t outputs = 0;
   bool stopped = false;
   agx_foreach_block(ctx, block) {
      agx_foreach_instr_in_block(block, I) {
         if (block != output_block) {
            /* Earlier control flow and producers retain ordinary encoding
             * validation. The terminal output protocol owns only this final
             * block, with no conditional or earlier vertex writeout. */
            if (I->op == AGX_OPCODE_ST_VARY || I->op == AGX_OPCODE_STOP)
               return 0;
            continue;
         }
         if (stopped)
            return 0;
         switch (I->op) {
         case AGX_OPCODE_STOP:
            stopped = true;
            break;
         case AGX_OPCODE_ST_VARY:
            if (I->nr_dests != 0 || I->nr_srcs != 2 ||
                I->src[0].type != AGX_INDEX_IMMEDIATE ||
                I->src[0].value >= AGX3_MAX_VERTEX_OUTPUTS ||
                (outputs & (UINT64_C(1) << I->src[0].value)) ||
                !agx3_vary_reg32(I->src[1]) ||
                agx_channels(I->src[1]) != 1)
               return 0;
            outputs |= UINT64_C(1) << I->src[0].value;
            break;
         case AGX_OPCODE_GET_SR:
            /* Native vertex programs use both IDs as ordinary scalar
             * producers ahead of their terminal UVS bundle. The SR encoder
             * independently validates the witnessed hardware number/width. */
            if (I->sr != AGX_SR_VERTEX_ID && I->sr != AGX_SR_INSTANCE_ID)
               return 0;
            break;
         default:
            if (!agx_opcodes_info[I->op].can_eliminate)
               return 0;
            /* Normal packing still validates each operand/width/encoding. */
            break;
         }
      }
   }
   return stopped && outputs >= 0x1f && !(outputs & (outputs + 1))
             ? util_bitcount64(outputs) : 0;
}

/* The sources of a dense terminal store bundle cannot be overwritten before
 * their injected staging COPY reads retire. Each six-store batch must use all
 * six distinct scoreboard slots; preserve its real output order so grouped
 * Vulkan varyings do not get silently renumbered. Original producers retire
 * on the first COPY and UVSWs retire the staging producers. */
static bool
agx3_outputs_are_terminal(agx_context *ctx, unsigned count,
                          uint8_t order[AGX3_MAX_VERTEX_OUTPUTS])
{
   unsigned next = 0;
   unsigned slot_mask = 0;
   bool stopped = false;
   agx_foreach_instr_global(ctx, I) {
      if (stopped)
         return false;
      if (I->op == AGX_OPCODE_ST_VARY) {
         if (next >= count || I->src[0].type != AGX_INDEX_IMMEDIATE)
            return false;
         const unsigned component = I->src[0].value;
         const unsigned slot = component % 6;
         if (slot_mask & BITFIELD_BIT(slot))
            return false;
         slot_mask |= BITFIELD_BIT(slot);
         order[next++] = component;
         if ((next % 6) == 0)
            slot_mask = 0;
      } else if (I->op == AGX_OPCODE_STOP) {
         if (next != count)
            return false;
         stopped = true;
      } else if (next) {
         return false;
      }
   }
   return next == count && stopped;
}

static bool
agx3_vertex_outputs_share_source(agx_context *ctx)
{
   agx_index first = agx_null();
   bool found = false;
   agx_foreach_instr_global(ctx, I) {
      if (I->op != AGX_OPCODE_ST_VARY)
         continue;
      if (!found) {
         first = I->src[1];
         found = true;
      } else if (!agx_is_equiv(first, I->src[1]) ||
                 first.size != I->src[1].size ||
                 first.abs != I->src[1].abs || first.neg != I->src[1].neg) {
         return false;
      }
   }
   return found;
}

/* Apple's real builtin EOT factory (BTNF kind5, EOT key0x10020300) emits
 * this complete 68-byte fragment program. Its ESL contract is a PBE table
 * pointer in uniform halves0..3 and imageblock byte offset in halves4..5.
 * Coordinates are group position times dispatched group dimensions, not
 * per-thread coordinates from the older compute-wrapper probe.
 * The AGX3 imgwblk is not a standalone translation of AGX2's block store: the
 * preceding pack_imgwir and coordinate-select sequence is part of its ABI.
 * Fresh native RGB10A2 key0x10021300 emits the identical physical program:
 * its normalized conversion happens before W32 tile storage; PBE describes
 * A2B10G10R10 versus A2R10G10B10 channel order. RG16 UNORM is likewise one
 * 32-bit imageblock word,
 * with its two U16-normalized lanes described by PBE. Fresh Apple G17P
 * imgwblk compilations for U16NORM and S16NORM are byte-identical, grounding
 * the same one-word program for signed RG16 while PBE describes signedness.
 * Scalar I32 key0x04021100 uses selector 0x35. R16G16_UINT key0x08020900
 * and R16G16_SINT key0x08020800 use selector 0x15. Fresh G17P
 * R16G16B16A16_UINT key0x10020900 uses the same selector with exactly two
 * imageblock words. R16G16B16A16_SINT key0x10020800 instead has a distinct
 * complete 84-byte builtin. R32G32_SINT key0x08021200 also uses the
 * 0x15 selector, while R32G32_UINT key0x08021300 uses the 0x75 selector,
 * despite both lowering to the same I32 store as scalar R32. Fresh
 * R32G32B32A32_UINT key0x10020b00 has four I32 imageblock words and returns
 * to the scalar 0x35 selector. R32G32B32A32_SINT key0x10020a00 also uses
 * selector 0x35 but requires the distinct complete 84-byte signed program.
 * Fresh R32G32B32A32_SFLOAT key0x10021100 has four I32 imageblock words and
 * is byte-identical to the 68-byte UINT/scalar 0x35 program; its PBE carries
 * FLOAT conversion.
 * Scalar R8_UINT key0x04020500, scalar R8_SINT key0x04020400,
 * R8G8_UINT key0x08020500 and R8G8_SINT
 * key0x08020400 use one I8
 * imageblock word and selector 0x55. R8G8B8A8_UINT/SINT use the same I8
 * builtin with four attachment components. R8_UNORM uses format3/layout2
 * with one component. R8_SNORM, R8G8_SNORM, R8G8B8A8_SNORM and
 * B8G8R8A8_SNORM attachment fields use format2/layout2 with one, two or four
 * components. They emit the normalized program below, matching the exact
 * U8NORM bytes; RGBA/BGRA memory ordering is encoded only in the PBE.
 * R16_UNORM format7 and R16_SNORM/R16G16B16A16_SNORM format6 attachment
 * fields use layout2 with one or four components. All emit a distinct
 * normalized-16 program with selector tail a895.
 * Scalar and paired R16 UINT/SINT formats use one I16 imageblock word and
 * the grounded selector-15 program; wider signed variants remain distinct.
 * R16_SFLOAT, R16G16_SFLOAT and R16G16B16A16_SFLOAT format12 with one,
 * two or four components use that same grounded program.
 * A2B10G10R10_UINT_PACK32 key0x10021200 uses one I16 word and selector0x15.
 * Recognize the exact post-RA semantic graph and replace it atomically, so no
 * partially grounded imgwblk ever escapes for a different format or shape. */
enum agx3_g17p_rt0_eot_kind {
   AGX3_G17P_RT0_EOT_NONE,
   AGX3_G17P_RT0_EOT_SELECTOR_75,
   AGX3_G17P_RT0_EOT_SELECTOR_15,
   AGX3_G17P_RT0_EOT_I32,
   AGX3_G17P_RT0_EOT_SELECTOR_55,
   AGX3_G17P_RT0_EOT_NORM16,
   AGX3_G17P_RT0_EOT_I16_SIGNED_TWO_WORDS,
   AGX3_G17P_RT0_EOT_I32_SIGNED_FOUR_WORDS,
};

static enum agx3_g17p_rt0_eot_kind
agx3_g17p_rt0_eot_kind(const agx_context *ctx, enum agx_isa_encoding enc)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || ctx->stage != MESA_SHADER_COMPUTE ||
       ctx->is_preamble || ctx->key->no_stop)
      return AGX3_G17P_RT0_EOT_NONE;

   unsigned instructions = 0, get_x = 0, get_y = 0, zeros = 0, stores = 0;
   unsigned x_reg = UINT_MAX, y_reg = UINT_MAX, zero_reg = UINT_MAX;
   unsigned coord_reg = UINT_MAX, offset_reg = UINT_MAX;
   enum agx_format store_format = AGX_FORMAT_I8;

   agx_foreach_block(ctx, block) {
      agx_foreach_instr_in_block(block, I) {
         if (I->op == AGX_OPCODE_STOP)
            continue;
         instructions++;

         if (I->op == AGX_OPCODE_GET_SR) {
            if (I->nr_dests != 1 || I->nr_srcs != 0 ||
                !agx3_src_is_register(I->dest[0]) || I->dest[0].memory ||
                I->dest[0].size != AGX_SIZE_32 ||
                agx_channels(I->dest[0]) != 1)
               return false;

            if (I->sr == AGX_SR_THREAD_POSITION_IN_GRID_X) {
               get_x++;
               x_reg = I->dest[0].value;
            } else if (I->sr == AGX_SR_THREAD_POSITION_IN_GRID_Y) {
               get_y++;
               y_reg = I->dest[0].value;
            } else {
               return false;
            }
         } else if (I->op == AGX_OPCODE_MOV_IMM) {
            if (I->nr_dests != 1 || I->nr_srcs != 0 || I->imm != 0 ||
                !agx3_src_is_register(I->dest[0]) || I->dest[0].memory ||
                I->dest[0].size != AGX_SIZE_32 ||
                agx_channels(I->dest[0]) != 1)
               return false;
            zeros++;
            zero_reg = I->dest[0].value;
         } else if (I->op == AGX_OPCODE_BLOCK_IMAGE_STORE) {
            if (I->nr_dests != 0 || I->nr_srcs != 4 ||
                (I->format != AGX_FORMAT_U8NORM &&
                 I->format != AGX_FORMAT_S8NORM &&
                 I->format != AGX_FORMAT_S16NORM &&
                 I->format != AGX_FORMAT_I8 &&
                 I->format != AGX_FORMAT_U16NORM &&
                 I->format != AGX_FORMAT_I16 &&
                 I->format != AGX_FORMAT_F16 &&
                 I->format != AGX_FORMAT_RGB10A2 &&
                 I->format != AGX_FORMAT_I32) || I->dim != AGX_DIM_2D ||
                !I->explicit_coords ||
                I->src[0].type != AGX_INDEX_IMMEDIATE ||
                I->src[0].value != 0 || I->src[0].size != AGX_SIZE_16 ||
                I->src[1].type != AGX_INDEX_IMMEDIATE ||
                I->src[1].value != 0 || I->src[1].size != AGX_SIZE_16 ||
                !agx3_src_is_register(I->src[2]) || I->src[2].memory ||
                I->src[2].abs || I->src[2].neg ||
                I->src[2].size != AGX_SIZE_32 ||
                agx_channels(I->src[2]) != 1 ||
                !agx3_src_is_register(I->src[3]) || I->src[3].memory ||
                I->src[3].abs || I->src[3].neg ||
                I->src[3].size != AGX_SIZE_32 ||
                agx_channels(I->src[3]) != 2)
               return false;
            if ((I->format == AGX_FORMAT_I32) != ctx->key->g17p_rt0_eot_i32)
               return false;
            stores++;
            store_format = I->format;
            offset_reg = I->src[2].value;
            coord_reg = I->src[3].value;
         } else {
            return false;
         }
      }
   }

   if (instructions != 4 || get_x != 1 || get_y != 1 || zeros != 1 ||
       stores != 1 || x_reg != coord_reg || y_reg != coord_reg + 2 ||
       zero_reg != offset_reg)
      return AGX3_G17P_RT0_EOT_NONE;

   if (ctx->key->g17p_rt0_eot_i16_signed_two_words &&
       (store_format != AGX_FORMAT_I16 ||
        ctx->key->g17p_rt0_eot_words != 2))
      return AGX3_G17P_RT0_EOT_NONE;

   if (ctx->key->g17p_rt0_eot_i32_signed_four_words &&
       (store_format != AGX_FORMAT_I32 ||
        ctx->key->g17p_rt0_eot_words != 4))
      return AGX3_G17P_RT0_EOT_NONE;

   if (store_format == AGX_FORMAT_S16NORM) {
      if (ctx->key->g17p_rt0_eot_norm16_selector_95 &&
          (ctx->key->g17p_rt0_eot_words == 1 ||
           ctx->key->g17p_rt0_eot_words == 2))
         return AGX3_G17P_RT0_EOT_NORM16;
      if (ctx->key->g17p_rt0_eot_words == 1)
         return AGX3_G17P_RT0_EOT_SELECTOR_75;

      return AGX3_G17P_RT0_EOT_NONE;
   }

   if (store_format == AGX_FORMAT_U16NORM) {
      if (ctx->key->g17p_rt0_eot_norm16_selector_95)
         return ctx->key->g17p_rt0_eot_words == 1
                   ? AGX3_G17P_RT0_EOT_NORM16
                   : AGX3_G17P_RT0_EOT_NONE;
      if (ctx->key->g17p_rt0_eot_words == 1)
         return AGX3_G17P_RT0_EOT_SELECTOR_75;
      if (ctx->key->g17p_rt0_eot_words == 2)
         return AGX3_G17P_RT0_EOT_SELECTOR_15;

      /* RG16 and RGBA16 lower to the same final instruction shape. Refuse an
       * ambiguous selector instead of silently emitting the wrong store. */
      return AGX3_G17P_RT0_EOT_NONE;
   }

   if (store_format == AGX_FORMAT_I32) {
      if ((ctx->key->g17p_rt0_eot_i32_selector_15 &&
           ctx->key->g17p_rt0_eot_i32_selector_75) ||
          ((ctx->key->g17p_rt0_eot_i32_four_words ||
            ctx->key->g17p_rt0_eot_i32_signed_four_words) &&
           (ctx->key->g17p_rt0_eot_i32_selector_15 ||
            ctx->key->g17p_rt0_eot_i32_selector_75)) ||
          (ctx->key->g17p_rt0_eot_i32_four_words &&
           ctx->key->g17p_rt0_eot_i32_signed_four_words))
         return AGX3_G17P_RT0_EOT_NONE;

      if (ctx->key->g17p_rt0_eot_i32_signed_four_words)
         return AGX3_G17P_RT0_EOT_I32_SIGNED_FOUR_WORDS;

      if (ctx->key->g17p_rt0_eot_i32_four_words)
         return ctx->key->g17p_rt0_eot_words == 4
                   ? AGX3_G17P_RT0_EOT_I32
                   : AGX3_G17P_RT0_EOT_NONE;

      if (ctx->key->g17p_rt0_eot_i32_selector_15) {
         /* Only the exact two-word signed pair has native evidence for this
          * I32 selector. Reject an ambiguous or scalar shape. */
         return ctx->key->g17p_rt0_eot_words == 2
                   ? AGX3_G17P_RT0_EOT_SELECTOR_15
                   : AGX3_G17P_RT0_EOT_NONE;
      }

      if (ctx->key->g17p_rt0_eot_i32_selector_75) {
         /* Only the exact two-word unsigned pair has native evidence for
          * this I32 selector. Reject an ambiguous or scalar shape. */
         return ctx->key->g17p_rt0_eot_words == 2
                   ? AGX3_G17P_RT0_EOT_SELECTOR_75
                   : AGX3_G17P_RT0_EOT_NONE;
      }

      return (ctx->key->g17p_rt0_eot_words == 1 ||
              ctx->key->g17p_rt0_eot_words == 2)
                ? AGX3_G17P_RT0_EOT_I32
                : AGX3_G17P_RT0_EOT_NONE;
   }

   if (store_format == AGX_FORMAT_I8)
      return ctx->key->g17p_rt0_eot_words == 1
                ? AGX3_G17P_RT0_EOT_SELECTOR_55
                : AGX3_G17P_RT0_EOT_NONE;

   if (store_format == AGX_FORMAT_F16)
      return (ctx->key->g17p_rt0_eot_words == 1 ||
              ctx->key->g17p_rt0_eot_words == 2)
                ? AGX3_G17P_RT0_EOT_SELECTOR_15
                : AGX3_G17P_RT0_EOT_NONE;

   if (ctx->key->g17p_rt0_eot_i32_selector_15 ||
       ctx->key->g17p_rt0_eot_i32_selector_75)
      return AGX3_G17P_RT0_EOT_NONE;
   if (store_format == AGX_FORMAT_I16) {
      if (ctx->key->g17p_rt0_eot_i16_signed_two_words)
         return AGX3_G17P_RT0_EOT_I16_SIGNED_TWO_WORDS;

      /* RG16 and RGBA16 integer stores use the same native selector, but only
       * the exact one- and two-word shapes have Apple builtin evidence. */
      return (ctx->key->g17p_rt0_eot_words == 1 ||
              ctx->key->g17p_rt0_eot_words == 2)
                ? AGX3_G17P_RT0_EOT_SELECTOR_15
                : AGX3_G17P_RT0_EOT_NONE;
   }

   return AGX3_G17P_RT0_EOT_SELECTOR_75;
}

/* A fresh applegpu_g17p AIR differential between the 2D and 2D-array
 * imgwblk intrinsics proves the array wire protocol: publish the layer with
 * its own PACK_IMGWIR, move XY to wire destination 6, then select array
 * dimension 2 in IMGWBLK. Match the complete allocated Mesa EOT graph and
 * its registers before substituting the byte-exact 110-byte program; no
 * general array block-store shape is implied. */
static bool
agx3_is_g17p_layered_u8norm_eot(const agx_context *ctx,
                                enum agx_isa_encoding enc)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || ctx->stage != MESA_SHADER_COMPUTE ||
       ctx->is_preamble || ctx->key->no_stop ||
       ctx->key->g17p_rt0_eot_words != 0 || ctx->key->g17p_rt0_eot_i32)
      return false;

   unsigned instruction = 0;
   agx_foreach_instr_global(ctx, I) {
      if (I->op == AGX_OPCODE_STOP)
         continue;

      switch (instruction++) {
      case 0:
      case 1:
      case 2: {
         const enum agx_sr expected[] = {
            AGX_SR_THREAD_POSITION_IN_GRID_X,
            AGX_SR_THREAD_POSITION_IN_GRID_Y,
            AGX_SR_THREADGROUP_POSITION_IN_GRID_Z,
         };
         if (I->op != AGX_OPCODE_GET_SR || I->nr_dests != 1 ||
             I->nr_srcs != 0 || I->sr != expected[instruction - 1] ||
             I->dest[0].type != AGX_INDEX_REGISTER ||
             I->dest[0].size != AGX_SIZE_32 || I->dest[0].memory ||
             agx_channels(I->dest[0]) != 1 ||
             I->dest[0].value != 2 * (instruction - 1))
            return false;
         break;
      }
      case 3:
         if (I->op != AGX_OPCODE_MOV_IMM || I->nr_dests != 1 ||
             I->nr_srcs != 0 || I->imm != 0 ||
             I->dest[0].type != AGX_INDEX_REGISTER ||
             I->dest[0].size != AGX_SIZE_32 || I->dest[0].memory ||
             agx_channels(I->dest[0]) != 1 || I->dest[0].value != 6)
            return false;
         break;
      case 4:
      case 5:
      case 6: {
         const unsigned lane = instruction - 5;
         if (I->op != AGX_OPCODE_BITOP ||
             (I->truth_table & 0xfu) != AGX3_SEL_TT_MOV || I->saturate ||
             I->nr_dests != 1 || I->nr_srcs != 2 ||
             I->dest[0].type != AGX_INDEX_REGISTER ||
             I->dest[0].size != AGX_SIZE_32 || I->dest[0].memory ||
             agx_channels(I->dest[0]) != 1 ||
             I->dest[0].value != 8 + 2 * lane ||
             I->src[0].type != AGX_INDEX_REGISTER ||
             I->src[0].size != AGX_SIZE_32 || I->src[0].memory ||
             I->src[0].abs || I->src[0].neg ||
             agx_channels(I->src[0]) != 1 ||
             I->src[0].value != 2 * lane ||
             I->src[1].type != AGX_INDEX_IMMEDIATE ||
             I->src[1].value != 0)
            return false;
         break;
      }
      case 7:
         if (I->op != AGX_OPCODE_BLOCK_IMAGE_STORE || I->nr_dests != 0 ||
             I->nr_srcs != 4 || I->format != AGX_FORMAT_U8NORM ||
             I->dim != AGX_DIM_2D_ARRAY || !I->explicit_coords ||
             I->src[0].type != AGX_INDEX_IMMEDIATE ||
             I->src[0].value != 0 || I->src[0].size != AGX_SIZE_16 ||
             I->src[1].type != AGX_INDEX_IMMEDIATE ||
             I->src[1].value != 0 || I->src[1].size != AGX_SIZE_16 ||
             I->src[2].type != AGX_INDEX_REGISTER ||
             I->src[2].size != AGX_SIZE_32 || I->src[2].memory ||
             !I->src[2].discard || I->src[2].abs || I->src[2].neg ||
             agx_channels(I->src[2]) != 1 || I->src[2].value != 6 ||
             I->src[3].type != AGX_INDEX_REGISTER ||
             I->src[3].size != AGX_SIZE_32 || I->src[3].memory ||
             !I->src[3].discard || I->src[3].abs || I->src[3].neg ||
             agx_channels(I->src[3]) != 3 || I->src[3].value != 8)
            return false;
         break;
      default:
         return false;
      }
   }

   return instruction == 8;
}

/* Native EOT key {RGBA16F RT0, R8 RT1} (mask 3) has a distinct, complete
 * 120-byte program.  As above, match the whole post-RA graph before replacing
 * it: this is the exact mixed-width layout used by Quake's WBOIT pass. */
static bool
agx3_is_g17p_wboit_mrt_eot(const agx_context *ctx,
                           enum agx_isa_encoding enc)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || ctx->stage != MESA_SHADER_COMPUTE ||
       ctx->is_preamble || ctx->key->no_stop)
      return false;

   unsigned instructions = 0, get_x = 0, get_y = 0, stores = 0;
   unsigned x_reg = UINT_MAX, y_reg = UINT_MAX, coord_reg = UINT_MAX;
   unsigned zero_reg = UINT_MAX, eight_reg = UINT_MAX;

   agx_foreach_block(ctx, block) {
      agx_foreach_instr_in_block(block, I) {
         if (I->op == AGX_OPCODE_STOP)
            continue;
         instructions++;

         if (I->op == AGX_OPCODE_GET_SR) {
            if (I->nr_dests != 1 || I->nr_srcs != 0 ||
                !agx3_src_is_register(I->dest[0]) || I->dest[0].memory ||
                I->dest[0].size != AGX_SIZE_32 ||
                agx_channels(I->dest[0]) != 1)
               return false;

            if (I->sr == AGX_SR_THREAD_POSITION_IN_GRID_X) {
               get_x++;
               x_reg = I->dest[0].value;
            } else if (I->sr == AGX_SR_THREAD_POSITION_IN_GRID_Y) {
               get_y++;
               y_reg = I->dest[0].value;
            } else {
               return false;
            }
         } else if (I->op == AGX_OPCODE_MOV_IMM) {
            if (I->nr_dests != 1 || I->nr_srcs != 0 ||
                !agx3_src_is_register(I->dest[0]) || I->dest[0].memory ||
                I->dest[0].size != AGX_SIZE_32 ||
                agx_channels(I->dest[0]) != 1)
               return false;

            if (I->imm == 0)
               zero_reg = I->dest[0].value;
            else if (I->imm == 8)
               eight_reg = I->dest[0].value;
            else
               return false;
         } else if (I->op == AGX_OPCODE_BLOCK_IMAGE_STORE) {
            if (I->nr_dests != 0 || I->nr_srcs != 4 ||
                I->dim != AGX_DIM_2D || !I->explicit_coords ||
                I->src[0].type != AGX_INDEX_IMMEDIATE ||
                I->src[0].value != 0 || I->src[0].size != AGX_SIZE_16 ||
                I->src[1].type != AGX_INDEX_IMMEDIATE ||
                I->src[1].value != stores || I->src[1].size != AGX_SIZE_16 ||
                !agx3_src_is_register(I->src[2]) || I->src[2].memory ||
                I->src[2].abs || I->src[2].neg ||
                I->src[2].size != AGX_SIZE_32 ||
                agx_channels(I->src[2]) != 1 ||
                !agx3_src_is_register(I->src[3]) || I->src[3].memory ||
                I->src[3].abs || I->src[3].neg ||
                I->src[3].size != AGX_SIZE_32 ||
                agx_channels(I->src[3]) != 2)
               return false;

            if ((stores == 0 &&
                 (I->format != AGX_FORMAT_I16 ||
                  I->src[2].value != zero_reg)) ||
                (stores == 1 &&
                 (I->format != AGX_FORMAT_U8NORM ||
                  I->src[2].value != eight_reg)) ||
                stores >= 2)
               return false;

            if (stores == 0)
               coord_reg = I->src[3].value;
            else if (I->src[3].value != coord_reg)
               return false;
            stores++;
         } else {
            return false;
         }
      }
   }

   return instructions == 6 && get_x == 1 && get_y == 1 && stores == 2 &&
          zero_reg != UINT_MAX && eight_reg != UINT_MAX &&
          x_reg == coord_reg && y_reg == coord_reg + 2;
}

/* Fresh applegpu_g17p builtin EOT with two RGBA8 attachments (mask 3) is a
 * distinct 120-byte program. Its two attachment entries use identical
 * format3/layout2/components4 keys; their vector positions select RT0/RT1,
 * while the imageblock byte offsets are 0/4. Match the complete allocated
 * graph before substituting it so no other MRT layout inherits these target
 * selectors or offsets. */
static bool
agx3_is_g17p_rgba8_mrt_eot(const agx_context *ctx,
                           enum agx_isa_encoding enc)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || ctx->stage != MESA_SHADER_COMPUTE ||
       ctx->is_preamble || ctx->key->no_stop)
      return false;

   unsigned instructions = 0, get_x = 0, get_y = 0, stores = 0;
   unsigned x_reg = UINT_MAX, y_reg = UINT_MAX, coord_reg = UINT_MAX;
   unsigned zero_reg = UINT_MAX, four_reg = UINT_MAX;

   agx_foreach_block(ctx, block) {
      agx_foreach_instr_in_block(block, I) {
         if (I->op == AGX_OPCODE_STOP)
            continue;
         instructions++;

         if (I->op == AGX_OPCODE_GET_SR) {
            if (I->nr_dests != 1 || I->nr_srcs != 0 ||
                !agx3_src_is_register(I->dest[0]) || I->dest[0].memory ||
                I->dest[0].size != AGX_SIZE_32 ||
                agx_channels(I->dest[0]) != 1)
               return false;

            if (I->sr == AGX_SR_THREAD_POSITION_IN_GRID_X) {
               get_x++;
               x_reg = I->dest[0].value;
            } else if (I->sr == AGX_SR_THREAD_POSITION_IN_GRID_Y) {
               get_y++;
               y_reg = I->dest[0].value;
            } else {
               return false;
            }
         } else if (I->op == AGX_OPCODE_MOV_IMM) {
            if (I->nr_dests != 1 || I->nr_srcs != 0 ||
                !agx3_src_is_register(I->dest[0]) || I->dest[0].memory ||
                I->dest[0].size != AGX_SIZE_32 ||
                agx_channels(I->dest[0]) != 1)
               return false;

            if (I->imm == 0)
               zero_reg = I->dest[0].value;
            else if (I->imm == 4)
               four_reg = I->dest[0].value;
            else
               return false;
         } else if (I->op == AGX_OPCODE_BLOCK_IMAGE_STORE) {
            if (I->nr_dests != 0 || I->nr_srcs != 4 ||
                I->format != AGX_FORMAT_U8NORM || I->dim != AGX_DIM_2D ||
                !I->explicit_coords ||
                I->src[0].type != AGX_INDEX_IMMEDIATE ||
                I->src[0].value != 0 || I->src[0].size != AGX_SIZE_16 ||
                I->src[1].type != AGX_INDEX_IMMEDIATE ||
                I->src[1].value != stores || I->src[1].size != AGX_SIZE_16 ||
                !agx3_src_is_register(I->src[2]) || I->src[2].memory ||
                I->src[2].abs || I->src[2].neg ||
                I->src[2].size != AGX_SIZE_32 ||
                agx_channels(I->src[2]) != 1 ||
                !agx3_src_is_register(I->src[3]) || I->src[3].memory ||
                I->src[3].abs || I->src[3].neg ||
                I->src[3].size != AGX_SIZE_32 ||
                agx_channels(I->src[3]) != 2 ||
                (stores == 0 && I->src[2].value != zero_reg) ||
                (stores == 1 && I->src[2].value != four_reg) || stores >= 2)
               return false;

            if (stores == 0)
               coord_reg = I->src[3].value;
            else if (I->src[3].value != coord_reg)
               return false;
            stores++;
         } else {
            return false;
         }
      }
   }

   return instructions == 6 && get_x == 1 && get_y == 1 && stores == 2 &&
          zero_reg != UINT_MAX && four_reg != UINT_MAX &&
          x_reg == coord_reg && y_reg == coord_reg + 2;
}

static bool
agx3_g17p_8norm_format(enum agx_format format)
{
   return format == AGX_FORMAT_U8NORM || format == AGX_FORMAT_S8NORM;
}

/* Native RGBA8 shaders pack before the write fence: opaque shaders then
 * acquire class 12, while blending already acquired it before the read/load.
 * Defer only over FLTNORM in one straight-line block, with exactly the native
 * wait/load/store sequence. Other pixel operations and control flow retain
 * their original ordering.
 */
static const agx_instr *
agx3_g17p_deferred_write_wait(agx_context *ctx, bool blend)
{
   const agx_instr *write = NULL;
   unsigned blocks = 0, phase = blend ? 0 : 2;
   agx_foreach_block(ctx, block) {
      if (++blocks != 1)
         return NULL;
      agx_foreach_instr_in_block(block, I) {
         if (agx3_is_cf_op(I) || I->op == AGX_OPCODE_SIGNAL_PIX)
            return NULL;
         if (I->op == AGX_OPCODE_WAIT_PIX) {
            if (phase == 0 && I->writeout == 8) {
               phase = 1;
            } else if (phase == 2 && I->writeout == (blend ? 4 : 12)) {
               write = I;
               phase = 3;
            } else {
               return NULL;
            }
         } else if (I->op == AGX_OPCODE_LD_TILE ||
                    I->op == AGX_OPCODE_ST_TILE) {
            const bool load = I->op == AGX_OPCODE_LD_TILE;
            if (phase != (load ? 1 : 3) ||
                !((agx3_g17p_8norm_format(I->format) && I->mask == 15) ||
                  (I->format == AGX_FORMAT_RGB10A2 && I->mask == 1)) ||
                I->pixel_offset || I->explicit_coords)
               return NULL;
            phase = load ? 2 : 4;
         } else if (phase == 1 ||
                    (phase == 3 && I->op != AGX_OPCODE_FLTNORM_AGX3 &&
                     I->op != AGX_OPCODE_PACK_RGB10A2_AGX3)) {
            return NULL;
         }
      }
   }
   return phase == 4 ? write : NULL;
}

/* The native WBOIT layout is one RGBA16F target followed by one R8 target in
 * the same tilebuffer. Both colour stores belong to one pixel transaction, so
 * enter once before the first class-12 wait and release only after the second
 * store. Keep this exact until another MRT layout is captured. */
static bool
agx3_is_g17p_wboit_mrt_colour(agx_context *ctx)
{
   unsigned blocks = 0, f16 = 0, r8 = 0, r8_loads = 0;

   agx_foreach_block(ctx, block) {
      if (++blocks != 1)
         return false;

      agx_foreach_instr_in_block(block, I) {
         if (I->op == AGX_OPCODE_LD_TILE) {
            if (I->format != AGX_FORMAT_U8NORM || I->mask != 1 ||
                I->pixel_offset != 8 || I->explicit_coords)
               return false;
            r8_loads++;
         } else if (I->op == AGX_OPCODE_ST_TILE) {
            if (I->explicit_coords)
               return false;
            if (I->format == AGX_FORMAT_F16 && I->mask == 15 &&
                I->pixel_offset == 0)
               f16++;
            else if (I->format == AGX_FORMAT_U8NORM && I->mask == 1 &&
                     I->pixel_offset == 8)
               r8++;
            else
               return false;
         }
      }
   }

   return blocks == 1 && r8_loads == 1 && f16 == 1 && r8 == 1;
}

/* Fresh MetalToolchain v27.1.5194.15 G17P evidence for two opaque RGBA8
 * attachments (mrt.nat SHA256
 * b6484363c120f218f20aaea9d69d0f114479fbc098333c7280e7b232f663f620)
 * ends in four FLTNORMs followed by this exact transaction:
 *
 *   870254000600 870254300800 e70654020002014e00000000
 *   070254300000 8702540c0800 e70654000000014e00000000
 *   0702540c0200
 *
 * The pre-RA pass sinks and reverses the two stores to keep both packed
 * values live.  Revalidate the complete post-RA graph here before selecting
 * the native per-target writeout classes.
 */
static const agx_instr *
agx3_g17p_rgba8_mrt_wait(agx_context *ctx)
{
   unsigned blocks = 0, packs = 0, stores = 0, waits = 0;
   const agx_instr *wait = NULL;

   agx_foreach_block(ctx, block) {
      if (++blocks != 1)
         return NULL;

      agx_foreach_instr_in_block(block, I) {
         if (I->op == AGX_OPCODE_STOP)
            continue;
         if (I->op == AGX_OPCODE_WAIT_PIX) {
            if (I->writeout != 12 || ++waits != 1 || stores || packs)
               return NULL;
            wait = I;
         } else if (I->op == AGX_OPCODE_FLTNORM_AGX3) {
            if (waits != 1 || stores || ++packs > 4)
               return NULL;
         } else if (I->op == AGX_OPCODE_ST_TILE) {
            const unsigned expected_offset = stores == 0 ? 4 : 0;
            if (waits != 1 || packs != 4 || stores >= 2 ||
                I->nr_dests != 0 || I->nr_srcs != 3 ||
                I->format != AGX_FORMAT_U8NORM || I->mask != 15 ||
                I->explicit_coords || I->pixel_offset != expected_offset ||
                !agx3_src_is_register(I->src[0]) || I->src[0].memory ||
                I->src[0].size != AGX_SIZE_16 ||
                agx_channels(I->src[0]) != 2 ||
                I->src[1].type != AGX_INDEX_IMMEDIATE ||
                I->src[1].value != AGX3_TILE_ALL_SAMPLES ||
                !agx_is_null(I->src[2]))
               return NULL;
            stores++;
         } else {
            return NULL;
         }
      }
   }

   return blocks == 1 && waits == 1 && packs == 4 && stores == 2 ? wait
                                                                 : NULL;
}

/* Prove complete shader-part shapes from post-RA IR. EXPORT is an ABI
 * declaration with no machine instruction; every declared source must be
 * supplied by one of the bounded uniform copies. Never infer this from bytes.
 */
static void
agx3_g17p_colour_link_info(agx_context *ctx, bool opaque_protocol)
{
   struct agx3_colour_link_info info = {0};
   memset(&ctx->agx3_colour_link, 0, sizeof(ctx->agx3_colour_link));
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B ||
       ctx->stage != MESA_SHADER_FRAGMENT ||
       ctx->key->fs.ignore_tib_dependencies)
      return;

   unsigned blocks = 0, packs = 0, stores = 0, waits = 0;
   const bool copy_only = !ctx->key->secondary && ctx->key->no_stop;
   const bool pack_only = ctx->key->secondary && opaque_protocol;
   if (!copy_only && !pack_only)
      return;

   agx_foreach_block(ctx, block) {
      if (++blocks != 1)
         return;
      agx_foreach_instr_in_block(block, I) {
         if (I->op == AGX_OPCODE_STOP)
            continue;
         if (copy_only) {
            if (I->op == AGX_OPCODE_EXPORT)
               continue;
            if (I->op != AGX_OPCODE_BITOP ||
                (I->truth_table & 0xfu) != AGX3_SEL_TT_MOV || I->saturate ||
                I->nr_dests != 1 || I->nr_srcs != 2 ||
                I->src[0].type != AGX_INDEX_UNIFORM || I->src[0].memory ||
                I->src[0].abs || I->src[0].neg ||
                I->src[0].size != AGX_SIZE_32 || agx_channels(I->src[0]) != 1 ||
                I->src[1].type != AGX_INDEX_IMMEDIATE || I->src[1].value != 0 ||
                !agx3_src_is_register(I->dest[0]) || I->dest[0].memory ||
                I->dest[0].size != AGX_SIZE_32 || agx_channels(I->dest[0]) != 1 ||
                ((I->dest[0].value | I->src[0].value) & 1) ||
                I->dest[0].value >= 256 || I->src[0].value >= 512 ||
                info.copy_count == 4)
               return;
            for (unsigned j = 0; j < info.copy_count; ++j)
               if (info.copies[j].reg == I->dest[0].value)
                  return;
            info.copies[info.copy_count].reg = I->dest[0].value;
            info.copies[info.copy_count++].uniform = I->src[0].value;
         } else if (I->op == AGX_OPCODE_FLTNORM_AGX3) {
            if (++packs > 2 || stores)
               return;
         } else if (I->op == AGX_OPCODE_WAIT_PIX) {
            if (I->writeout != 12 || ++waits != 1 || stores)
               return;
         } else if (I->op == AGX_OPCODE_ST_TILE) {
            if (++stores != 1 || packs != 2 || waits != 1 ||
                I->nr_srcs != 3 ||
                I->src[0].size != AGX_SIZE_16 || agx_channels(I->src[0]) != 2 ||
                !agx3_src_is_register(I->src[0]) || I->src[0].memory ||
                I->src[1].type != AGX_INDEX_IMMEDIATE || I->src[1].value != 0xff ||
                !agx_is_null(I->src[2]) || I->explicit_coords || I->pixel_offset ||
                !agx3_g17p_8norm_format(I->format) || I->mask != 15)
               return;
            info.store_reg = I->src[0].value;
         } else {
            return;
         }
      }
   }
   if (copy_only) {
      unsigned exported = 0;
      agx_foreach_block(ctx, block) {
         agx_foreach_instr_in_block(block, I) {
            if (I->op != AGX_OPCODE_EXPORT)
               continue;
            if (I->nr_srcs != 1 || !agx3_src_is_register(I->src[0]) ||
                I->src[0].memory || I->src[0].size != AGX_SIZE_32 ||
                agx_channels(I->src[0]) != 1 || I->src[0].value != I->imm)
               return;
            unsigned j;
            for (j = 0; j < info.copy_count; ++j)
               if (info.copies[j].reg == I->src[0].value)
                  break;
            if (j == info.copy_count)
               return;
            exported |= 1u << j;
         }
      }
      info.copy_only = info.copy_count &&
         exported == (1u << info.copy_count) - 1;
   } else {
      info.opaque_pack = packs == 2 && stores == 1 && waits == 1;
   }
   ctx->agx3_colour_link = info;
}

static enum agx3_alu_hint
agx3_post_ra_hint(agx_index index)
{
   return index.discard ? AGX3_ALU_HINT_DISCARD :
          index.cache ? AGX3_ALU_HINT_CACHE : AGX3_ALU_HINT_NONE;
}

bool
agx3_pack_binary(agx_context *ctx, struct util_dynarray *emission)
{
   const enum agx_isa_encoding enc = ctx->key->dev.encoding;
   const unsigned nslots = (enc == AGX_ISA_ENCODING_AGX3C) ? 8 : 6;
   const unsigned initial_size = emission->size;
   ctx->agx3_staging_known = true;
   ctx->agx3_fltnorm_start = false;

   if (agx3_is_g17p_rgba8_mrt_eot(ctx, enc)) {
      /* applegpu_g17p builtin EOT generated from two
       * format3/layout2/components4 attachment keys. SHA256 of the native
       * image: b464eff17eb9adb2a6a68da4ea8a5b43ae6eee16edb28065c474ea2d51026f6c. */
      static const uint8_t g17p_rgba8_mrt_eot[] = {
         0x1c,0x9c,0x10,0x06,0x24,0xa8,0x14,0x06,
         0x0c,0x9d,0x10,0x26,0x24,0xa9,0x10,0x26,
         0x9f,0x10,0x54,0x02,0x03,0x04,0x14,0x00,
         0x50,0x26,0x0a,0x00,0x9f,0x20,0x54,0x00,
         0x03,0x00,0x10,0x00,0x50,0x26,0x0a,0x00,
         0x97,0x05,0x54,0x04,0x00,0x08,0x00,0x00,
         0xd0,0x40,0x10,0x00,0x0b,0x82,0x21,0x04,
         0x1b,0x80,0x21,0x04,0x97,0x05,0x54,0x0c,
         0x00,0x10,0x00,0x00,0xd0,0x44,0x10,0x00,
         0x4b,0x03,0x2f,0x00,0x42,0x00,0x00,0x04,
         0x00,0x00,0x5b,0x01,0x2f,0x00,0x42,0x00,
         0x00,0x04,0x00,0x00,0x57,0x10,0x54,0x00,
         0x00,0x00,0x00,0x80,0xa8,0x75,0x00,0x00,
         0x57,0x20,0x54,0x08,0x00,0x10,0x00,0x90,
         0xa8,0x75,0x00,0x00,0x0e,0x00,0x00,0x00,
      };
      static_assert(sizeof(g17p_rgba8_mrt_eot) == 120,
                    "RGBA8 MRT EOT native length");
      ctx->max_reg = MAX2(ctx->max_reg, 7);
      ctx->agx3_staging_halfs = MAX2(ctx->agx3_staging_halfs, 16);
      return agx3_append(emission, g17p_rgba8_mrt_eot,
                         sizeof(g17p_rgba8_mrt_eot));
   }

   if (agx3_is_g17p_wboit_mrt_eot(ctx, enc)) {
      static const uint8_t g17p_wboit_mrt_eot[] = {
         0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06,
         0x0c, 0x9d, 0x10, 0x26, 0x24, 0xa9, 0x10, 0x26,
         0x9f, 0x10, 0x54, 0x02, 0x03, 0x04, 0x14, 0x00,
         0x50, 0x26, 0x0a, 0x00, 0x9f, 0x20, 0x54, 0x00,
         0x03, 0x00, 0x10, 0x00, 0x50, 0x26, 0x0a, 0x00,
         0x97, 0x05, 0x54, 0x04, 0x00, 0x08, 0x00, 0x00,
         0xd0, 0x40, 0x10, 0x00, 0x0b, 0x82, 0x21, 0x04,
         0x1b, 0x80, 0x21, 0x04, 0x97, 0x05, 0x54, 0x0c,
         0x00, 0x10, 0x00, 0x00, 0xd0, 0x44, 0x10, 0x00,
         0x4b, 0x03, 0x2f, 0x00, 0x42, 0x00, 0x00, 0x04,
         0x00, 0x00, 0x5b, 0x01, 0x2f, 0x00, 0x42, 0x00,
         0x00, 0x04, 0x00, 0x00, 0x57, 0x10, 0x54, 0x00,
         0x00, 0x00, 0x00, 0x80, 0xa8, 0x15, 0x00, 0x00,
         0x57, 0x20, 0x54, 0x08, 0x00, 0x10, 0x00, 0x90,
         0xa8, 0x75, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
      };
      /* Native common field0=3 is three 32-bit GPRs. The text's only GPR
       * COPY destinations are halves4/5, so retain the eight-half bound used
       * by the matching RT0 builtin. D=32 independently needs16 staging. */
      ctx->max_reg = MAX2(ctx->max_reg, 7);
      ctx->agx3_staging_halfs = MAX2(ctx->agx3_staging_halfs, 16);
      return agx3_append(emission, g17p_wboit_mrt_eot,
                         sizeof(g17p_wboit_mrt_eot));
   }

   if (agx3_is_g17p_layered_u8norm_eot(ctx, enc)) {
      static const uint8_t g17p_layered_u8norm_eot[] = {
         /* Exact Mesa GET_SR X/Y/layer, MOV0 and MOV1 prefix. */
         0x0c,0xa0,0x10,0x06,0x1c,0xa1,0x10,0x26,
         0x2c,0x9e,0x10,0x46,0x3b,0x00,0x00,0x00,
         0x4c,0x80,0x02,0x00,0x00,0x20,0x00,0x00,
         0x9f,0x11,0x54,0x08,0x02,0x00,0x00,0x88,0x11,0x04,
         0x5c,0x80,0x02,0x00,0x00,0x40,0x00,0x00,
         0x9f,0x11,0x54,0x0a,0x02,0x00,0x08,0x88,0x11,0x04,
         0x6c,0x80,0x02,0x00,0x00,0x80,0x00,0x00,
         0x9f,0x11,0x54,0x0c,0x02,0x00,0x10,0x88,0x11,0x04,
         /* applegpu_g17p array wire: layer r6, then XY r4:r5. */
         0x97,0x15,0x54,0x04,0x00,0x18,0x10,0x00,0xf0,0xc0,0x00,0x00,
         0x97,0x05,0x54,0x06,0x00,0x08,0x00,0x00,0xd0,0x40,0x10,0x00,
         0x57,0x10,0x54,0x00,0x00,0x00,0x00,0x80,0x28,0x75,0x00,0x00,
         0x0e,0x00,0x00,0x00,
      };
      static_assert(sizeof(g17p_layered_u8norm_eot) == 110,
                    "layered EOT native length");
      ctx->max_reg = MAX2(ctx->max_reg, 13);
      ctx->agx3_staging_halfs = MAX2(ctx->agx3_staging_halfs, 8);
      return agx3_append(emission, g17p_layered_u8norm_eot,
                         sizeof(g17p_layered_u8norm_eot));
   }

   const enum agx3_g17p_rt0_eot_kind rt0_eot =
      agx3_g17p_rt0_eot_kind(ctx, enc);
   if (rt0_eot != AGX3_G17P_RT0_EOT_NONE) {
      static const uint8_t g17p_rgba16_sint_eot[] = {
         0x1c,0x9c,0x10,0x06,0x34,0xa8,0x10,0x06,0x0c,0x9d,0x10,0x26,
         0x24,0xa9,0x14,0x26,0x24,0xc3,0x10,0x46,0x9f,0x10,0x54,0x00,
         0x00,0x04,0x18,0x00,0x50,0x06,0x0b,0x00,0x9f,0x20,0x54,0x02,
         0x00,0x00,0x14,0x00,0x50,0x06,0x0b,0x00,0x97,0x45,0x54,0x04,
         0x00,0x08,0x10,0x00,0xf0,0xc0,0x00,0x00,0x97,0x05,0x54,0x06,
         0x00,0x08,0x00,0x00,0xd0,0x40,0x10,0x00,0x57,0x10,0x54,0x00,
         0x00,0x00,0x00,0x80,0x28,0x15,0x00,0x00,0x0e,0x00,0x00,0x00,
      };
      static const uint8_t g17p_rgba32_sint_eot[] = {
         0x1c,0x9c,0x10,0x06,0x34,0xa8,0x10,0x06,0x0c,0x9d,0x10,0x26,
         0x24,0xa9,0x14,0x26,0x24,0xc3,0x10,0x46,0x9f,0x10,0x54,0x00,
         0x00,0x04,0x18,0x00,0x50,0x06,0x0b,0x00,0x9f,0x20,0x54,0x02,
         0x00,0x00,0x14,0x00,0x50,0x06,0x0b,0x00,0x97,0x45,0x54,0x04,
         0x00,0x08,0x10,0x00,0xf0,0xc0,0x00,0x00,0x97,0x05,0x54,0x06,
         0x00,0x08,0x00,0x00,0xd0,0x40,0x10,0x00,0x57,0x10,0x54,0x00,
         0x00,0x00,0x00,0x80,0x28,0x35,0x00,0x00,0x0e,0x00,0x00,0x00,
      };

      if (rt0_eot == AGX3_G17P_RT0_EOT_I16_SIGNED_TWO_WORDS) {
         ctx->max_reg = MAX2(ctx->max_reg, 7);
         ctx->agx3_staging_halfs = MAX2(ctx->agx3_staging_halfs, 8);
         return agx3_append(emission, g17p_rgba16_sint_eot,
                            sizeof(g17p_rgba16_sint_eot));
      }

      if (rt0_eot == AGX3_G17P_RT0_EOT_I32_SIGNED_FOUR_WORDS) {
         ctx->max_reg = MAX2(ctx->max_reg, 7);
         ctx->agx3_staging_halfs = MAX2(ctx->agx3_staging_halfs, 8);
         return agx3_append(emission, g17p_rgba32_sint_eot,
                            sizeof(g17p_rgba32_sint_eot));
      }

      static const uint8_t g17p_rt0_eot[][68] = {{
         0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06,
         0x0c, 0x9d, 0x10, 0x26, 0x24, 0xa9, 0x10, 0x26,
         0x9f, 0x10, 0x54, 0x00, 0x00, 0x04, 0x14, 0x00,
         0x50, 0x06, 0x0b, 0x00,
         0x9f, 0x20, 0x54, 0x02, 0x00, 0x00, 0x10, 0x00,
         0x50, 0x06, 0x0b, 0x00,
         0x97, 0x05, 0x54, 0x04, 0x00, 0x08, 0x00, 0x00,
         0xd0, 0x40, 0x10, 0x00,
         0x57, 0x10, 0x54, 0x00, 0x00, 0x00, 0x00, 0x80,
         0xa8, 0x75, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
      }, {
         /* Native RG16 integer keys0x08020900/0x08020800. */
         0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06,
         0x0c, 0x9d, 0x10, 0x26, 0x24, 0xa9, 0x10, 0x26,
         0x9f, 0x10, 0x54, 0x00, 0x00, 0x04, 0x14, 0x00,
         0x50, 0x06, 0x0b, 0x00,
         0x9f, 0x20, 0x54, 0x02, 0x00, 0x00, 0x10, 0x00,
         0x50, 0x06, 0x0b, 0x00,
         0x97, 0x05, 0x54, 0x04, 0x00, 0x08, 0x00, 0x00,
         0xd0, 0x40, 0x10, 0x00,
         0x57, 0x10, 0x54, 0x00, 0x00, 0x00, 0x00, 0x80,
         0xa8, 0x15, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
      }, {
         /* applegpu-nt builtin EOT keys 0x04021100 (R32 float) and
          * 0x08021100 (R32G32 float) are byte-identical. */
         0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06,
         0x0c, 0x9d, 0x10, 0x26, 0x24, 0xa9, 0x10, 0x26,
         0x9f, 0x10, 0x54, 0x00, 0x00, 0x04, 0x14, 0x00,
         0x50, 0x06, 0x0b, 0x00,
         0x9f, 0x20, 0x54, 0x02, 0x00, 0x00, 0x10, 0x00,
         0x50, 0x06, 0x0b, 0x00,
         0x97, 0x05, 0x54, 0x04, 0x00, 0x08, 0x00, 0x00,
         0xd0, 0x40, 0x10, 0x00,
         0x57, 0x10, 0x54, 0x00, 0x00, 0x00, 0x00, 0x80,
         0xa8, 0x35, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
      }, {
         /* Native keys0x08020500/0x08020400: one RG8 integer I8 word. */
         0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06,
         0x0c, 0x9d, 0x10, 0x26, 0x24, 0xa9, 0x10, 0x26,
         0x9f, 0x10, 0x54, 0x00, 0x00, 0x04, 0x14, 0x00,
         0x50, 0x06, 0x0b, 0x00,
         0x9f, 0x20, 0x54, 0x02, 0x00, 0x00, 0x10, 0x00,
         0x50, 0x06, 0x0b, 0x00,
         0x97, 0x05, 0x54, 0x04, 0x00, 0x08, 0x00, 0x00,
         0xd0, 0x40, 0x10, 0x00,
         0x57, 0x10, 0x54, 0x00, 0x00, 0x00, 0x00, 0x80,
         0xa8, 0x55, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
      }, {
         /* Native format6/layout2/components4: RGBA16 SNORM. */
         0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06,
         0x0c, 0x9d, 0x10, 0x26, 0x24, 0xa9, 0x10, 0x26,
         0x9f, 0x10, 0x54, 0x00, 0x00, 0x04, 0x14, 0x00,
         0x50, 0x06, 0x0b, 0x00,
         0x9f, 0x20, 0x54, 0x02, 0x00, 0x00, 0x10, 0x00,
         0x50, 0x06, 0x0b, 0x00,
         0x97, 0x05, 0x54, 0x04, 0x00, 0x08, 0x00, 0x00,
         0xd0, 0x40, 0x10, 0x00,
         0x57, 0x10, 0x54, 0x00, 0x00, 0x00, 0x00, 0x80,
         0xa8, 0x95, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
      }};
      /* Native metadata common0=3 means six GPR halves; retain the existing
       * conservative eight-half allocation. Fields38/42 are D=16, requiring
       * eight staging halves independently. Main68 pads to128: LDSHDR f1=1. */
      ctx->max_reg = MAX2(ctx->max_reg, 7);
      ctx->agx3_staging_halfs = MAX2(ctx->agx3_staging_halfs, 8);
      const unsigned index = rt0_eot - AGX3_G17P_RT0_EOT_SELECTOR_75;
      return agx3_append(emission, g17p_rt0_eot[index],
                         sizeof(g17p_rt0_eot[index]));
   }

   struct util_dynarray fixups;
   uint16_t outstanding = 0;
   uint8_t next_uvs_slot = 1;
   unsigned texture_ops_left = 0;
   unsigned tile_loads_left = 0;
   unsigned tile_loads_seen = 0;
   unsigned tile_stores_left = 0;
   unsigned image_loads_left = 0;
   unsigned image_writes_left = 0;
   unsigned texture_samples_left = 0;
   unsigned device_loads_left = 0;
   bool has_coverage_write = false;
   bool has_zs_emit = false;
   bool uniform_half_used[512] = {false};
   unsigned count = 0;

   assert(enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C);
   util_dynarray_init(&fixups, ctx);

   agx_foreach_block(ctx, scan_block) {
      agx_foreach_instr_in_block(scan_block, scan_I) {
         if (scan_I->op != AGX_OPCODE_STOP)
            count++;
         /* Apple's "last texture op" marker goes on the final smp of the
          * whole stream, fetch or sample alike. */
         if (scan_I->op == AGX_OPCODE_TEXTURE_LOAD ||
             scan_I->op == AGX_OPCODE_IMAGE_LOAD ||
             scan_I->op == AGX_OPCODE_TEXTURE_SAMPLE)
            texture_ops_left++;
         if (scan_I->op == AGX_OPCODE_LD_TILE)
            tile_loads_left++;
         if (scan_I->op == AGX_OPCODE_ST_TILE)
            tile_stores_left++;
         if (scan_I->op == AGX_OPCODE_SAMPLE_MASK_1X_AGX3 ||
             scan_I->op == AGX_OPCODE_ZS_EMIT_AGX3 ||
             scan_I->op == AGX_OPCODE_ZS_EMIT)
            has_coverage_write = true;
         if (scan_I->op == AGX_OPCODE_ZS_EMIT)
            has_zs_emit = true;
         if (scan_I->op == AGX_OPCODE_DEVICE_LOAD)
            device_loads_left++;
         if (scan_I->op == AGX_OPCODE_IMAGE_LOAD)
            image_loads_left++;
         if (scan_I->op == AGX_OPCODE_IMAGE_WRITE)
            image_writes_left++;
         if (scan_I->op == AGX_OPCODE_TEXTURE_SAMPLE)
            texture_samples_left++;
         if (scan_I->op == AGX_OPCODE_UNIFORM_STORE &&
             scan_I->nr_srcs == 2 &&
             scan_I->src[1].type == AGX_INDEX_IMMEDIATE) {
            const unsigned base = scan_I->src[1].value;
            const unsigned stride = agx_size_align_16(scan_I->src[0].size);
            for (unsigned c = 0; c < 4; ++c) {
               if (!(scan_I->mask & BITFIELD_BIT(c)))
                  continue;
               for (unsigned h = 0; h < stride; ++h) {
                  if (base + c * stride + h < ARRAY_SIZE(uniform_half_used))
                     uniform_half_used[base + c * stride + h] = true;
               }
            }
         }
      }
   }

   /* The valid G17P colour-only corpus has one class-12 wait/store bracket:
    * an entry gate immediately before the ordinary wait, and a final release
    * immediately after the store. Restrict synthesis to that exact API-shader
    * shape; programmable blending, MRT and special BG/EOT programs retain
    * their existing independently witnessed protocols. */
   const bool g17p_wboit_mrt_colour =
      enc == AGX_ISA_ENCODING_AGX3B && ctx->stage == MESA_SHADER_FRAGMENT &&
      !ctx->key->fs.ignore_tib_dependencies &&
      agx3_is_g17p_wboit_mrt_colour(ctx);
   const agx_instr *g17p_rgba8_mrt_wait =
      enc == AGX_ISA_ENCODING_AGX3B && ctx->stage == MESA_SHADER_FRAGMENT &&
            !ctx->key->fs.ignore_tib_dependencies
         ? agx3_g17p_rgba8_mrt_wait(ctx)
         : NULL;
   const bool g17p_rgba8_mrt_colour = g17p_rgba8_mrt_wait != NULL;
   const bool g17p_colour_protocol =
      (enc == AGX_ISA_ENCODING_AGX3B &&
       ctx->stage == MESA_SHADER_FRAGMENT &&
       !ctx->key->fs.ignore_tib_dependencies && tile_stores_left == 1 &&
       tile_loads_left == 0) || g17p_wboit_mrt_colour ||
      g17p_rgba8_mrt_colour;

   /* Multiple colour stores need an attachment-specific pixel transaction.
    * The captured G17P forms are the WBOIT pair and the opaque two-RGBA8 pair
    * above; emitting individual tile operations without either exact bracket
    * produces a syntactically valid but unusable shader. */
   if (enc == AGX_ISA_ENCODING_AGX3B &&
       ctx->stage == MESA_SHADER_FRAGMENT &&
       !ctx->key->fs.ignore_tib_dependencies && tile_stores_left > 1 &&
       !g17p_wboit_mrt_colour && !g17p_rgba8_mrt_colour)
      goto fail;
   /* Exact native RGBA8 alpha pipeline: one class-8 read wait/load followed
    * by one class-4 write wait/store, under the same class-12 entry/release.
    * Keep MRT, special coordinates, other formats and control flow closed.
    */
   bool g17p_blend_protocol =
      enc == AGX_ISA_ENCODING_AGX3B && ctx->stage == MESA_SHADER_FRAGMENT &&
      !ctx->key->fs.ignore_tib_dependencies && tile_loads_left == 1 &&
      tile_stores_left == 1;
   if (g17p_blend_protocol) {
      unsigned blocks = 0, phase = 0;
      agx_foreach_block(ctx, protocol_block) {
         if (++blocks != 1)
            g17p_blend_protocol = false;
         agx_foreach_instr_in_block(protocol_block, P) {
            if (P->op == AGX_OPCODE_WAIT_PIX) {
               if (phase == 0 && P->writeout == 8)
                  phase = 1;
               else if (phase == 2 && P->writeout == 4)
                  phase = 3;
               else
                  g17p_blend_protocol = false;
            } else if (P->op == AGX_OPCODE_LD_TILE ||
                       P->op == AGX_OPCODE_ST_TILE) {
               const bool load = P->op == AGX_OPCODE_LD_TILE;
               if (phase != (load ? 1 : 3) ||
                   !((agx3_g17p_8norm_format(P->format) &&
                      (P->mask == 3 || P->mask == 15)) ||
                     (P->format == AGX_FORMAT_RGB10A2 && P->mask == 1)) || P->pixel_offset || P->explicit_coords)
                  g17p_blend_protocol = false;
               phase = load ? 2 : 4;
            } else if (P->op == AGX_OPCODE_SIGNAL_PIX) {
               g17p_blend_protocol = false;
            }
         }
      }
      g17p_blend_protocol &= phase == 4;
   }
   const bool g17p_pixel_protocol =
      g17p_colour_protocol || g17p_blend_protocol || has_coverage_write;
   const agx_instr *g17p_deferred_write = g17p_rgba8_mrt_colour
      ? g17p_rgba8_mrt_wait
      : (g17p_pixel_protocol
            ? agx3_g17p_deferred_write_wait(ctx, g17p_blend_protocol)
            : NULL);
   agx3_g17p_colour_link_info(ctx, g17p_colour_protocol &&
                                  g17p_deferred_write != NULL);
   ctx->agx3_colour_link.straight_line = list_length(&ctx->blocks) == 1;
   agx_foreach_instr_global(ctx, entry_I) {
      if (agx3_is_cf_op(entry_I) ||
          entry_I->op == AGX_OPCODE_SAMPLE_MASK_1X_AGX3 ||
          entry_I->op == AGX_OPCODE_ZS_EMIT_AGX3)
         ctx->agx3_colour_link.straight_line = false;
   }
   unsigned g17p_link_packs_seen = 0;
   bool g17p_colour_open = false;
   const agx_instr *g17p_deferred_zs_wait = NULL;
   const bool g17p_position_protocol =
      agx3_is_g17p_float2_position_vs(ctx, enc);
   unsigned g17p_positions_staged = 0;
   const unsigned g17p_output_count = agx3_g17p_vertex_output_count(ctx, enc);
   uint8_t g17p_output_order[AGX3_MAX_VERTEX_OUTPUTS] = {0};
   /* The original six-output/single-source protocol has an exact per-COPY
    * retirement contract even when its stores are terminal. Other terminal
    * six-output programs need batching to preserve distinct values through
    * RA, just like wider programs. */
   const bool g17p_legacy_single_source =
      g17p_output_count == nslots && agx3_vertex_outputs_share_source(ctx);
   const bool g17p_terminal_outputs = g17p_output_count &&
      !g17p_legacy_single_source &&
      agx3_outputs_are_terminal(ctx, g17p_output_count, g17p_output_order);
   /* Only the old six-output fallback supports intervening physical GPR
    * reuse. Wider outputs require the pre-RA terminal bundle. */
   const bool g17p_user_output_protocol = g17p_output_count == 6 ||
      (g17p_output_count >= 5 && g17p_terminal_outputs);
   unsigned g17p_outputs_staged = 0;
   /* User-output graphs outside the qualified scalar shapes must not
    * silently fall back to immediate, potentially out-of-order writeout. */
   if (enc == AGX_ISA_ENCODING_AGX3B && ctx->stage == MESA_SHADER_VERTEX) {
      agx_foreach_block(ctx, block) {
         agx_foreach_instr_in_block(block, I) {
            if (I->op == AGX_OPCODE_ST_VARY && I->nr_srcs >= 1 &&
                I->src[0].type == AGX_INDEX_IMMEDIATE &&
                I->src[0].value >= 4 && !g17p_user_output_protocol)
               AGX3_REFUSE(I);
         }
      }
   }
   /* Across 71 fully decoded own-shader AGX3b vertex programs, f79 marks the
    * final LDDEV with zero exceptions and the first LDDEV uses slot 6 with
    * zero exceptions. Later slots vary with scheduling, so only the two
    * invariant choices are generalized here. UVSW hint bytes likewise vary
    * with liveness and were splice-proven inert; the byte-exact triangle path
    * above retains its native hints, while general shaders keep Mesa's. */
   const bool g17p_vertex_load_policy =
      enc == AGX_ISA_ENCODING_AGX3B && ctx->stage == MESA_SHADER_VERTEX &&
      device_loads_left > 0;
   /* Native read_write and sample-to-write kernels assign their sole texture
    * producer to slot 1 and imgwci waits on that slot. Select slot 1 from the
    * exact one-producer/one-write compute graph; the ordinary dependency pass
    * still derives the write's wait bit from the producer result registers. */
   const bool g17p_storage_image_protocol =
      enc == AGX_ISA_ENCODING_AGX3B && ctx->stage == MESA_SHADER_COMPUTE &&
      image_loads_left + texture_samples_left == 1 &&
      image_writes_left == 1 && texture_ops_left == 1;

   /* Assign one scoreboard timeline over the whole linearized shader, rather
    * than restarting at each basic block.  Fallthrough blocks may consume an
    * async result from their predecessor.  Control flow after an async op is
    * rejected below until CFG-aware state merging is grounded. */
   struct agx3_slot_instr *sl =
      rzalloc_array(ctx, struct agx3_slot_instr, count ? count : 1);
   struct agx3_slot_result *res =
      rzalloc_array(ctx, struct agx3_slot_result, count ? count : 1);
   if (sl == NULL || res == NULL)
      goto fail;

   unsigned k = 0;
   unsigned vertex_loads_scheduled = 0;
   agx_foreach_block(ctx, slot_block) {
      agx_foreach_instr_in_block(slot_block, I) {
         if (I->op == AGX_OPCODE_STOP)
            continue;
         struct agx3_slot_instr *e = &sl[k++];
         /* EXPORT is a zero-byte register-ABI declaration after RA, not a
          * physical read. Letting it consume an async dependency marks the
          * slot waited without emitting any wait, so a linked epilog reads
          * the result too early. Real consumers retain their own waits;
          * outstanding results are synchronized at the shader-part tail. */
         if (I->op == AGX_OPCODE_EXPORT)
            continue;
         e->is_async = agx3_is_async(I);
         /* Apple's register-LOD texel fetch has two independent scoreboard
          * generations: PACK_SMPIR/COPY publish staging on slot 2 and TEX
          * consumes it while returning the sampled vector on slot 1. Model
          * slot 2 as transient so a prior generation is retired before the
          * compound encoding reuses it and later users do not wait it again. */
         if (enc == AGX_ISA_ENCODING_AGX3B &&
             (I->op == AGX_OPCODE_TEXTURE_LOAD ||
              I->op == AGX_OPCODE_IMAGE_LOAD) &&
             I->lod_mode == AGX_LOD_MODE_LOD_MIN && I->nr_srcs >= 2 &&
             agx3_tex_plain_reg(I->src[1]) &&
             I->src[1].size == AGX_SIZE_32 &&
             agx_channels(I->src[1]) == 1) {
            e->preferred_slot = 1;
            e->transient_slots = 1u << 1;
         }
         if ((enc == AGX_ISA_ENCODING_AGX3B &&
              I->op == AGX_OPCODE_IMAGE_LOAD) ||
             (g17p_storage_image_protocol &&
              I->op == AGX_OPCODE_TEXTURE_SAMPLE))
            e->preferred_slot = 1;
         if (enc == AGX_ISA_ENCODING_AGX3B &&
             I->op == AGX_OPCODE_ZS_EMIT_AGX3)
            e->preferred_slot = 1;
         /* Apple's G17P programmable-blend and framebuffer-fetch corpus
          * assigns LLDCol to slot 6, including scalar, pair, and RGBA loads.
          * Keep tile reads in that native hardware domain instead of applying
          * the generic round-robin starting point. */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_LD_TILE)
            e->preferred_slot = 6;
         /* Native multisample reloads keep the texture result on slot 1 and
          * publish the packed sample/layer coordinate independently on slot
          * 2. This prevents the fetch from waiting on its own generation. */
         if (enc == AGX_ISA_ENCODING_AGX3B &&
             I->op == AGX_OPCODE_TEXTURE_LOAD &&
             (I->dim == AGX_DIM_2D_MS ||
              I->dim == AGX_DIM_2D_MS_ARRAY))
            e->preferred_slot = 1;
         if (g17p_vertex_load_policy && I->op == AGX_OPCODE_DEVICE_LOAD &&
             vertex_loads_scheduled++ == 0)
            e->preferred_slot = 6;
         unsigned nd = 0, ns = 0;
         for (unsigned d = 0; d < I->nr_dests; ++d)
            nd = agx3_expand_regs(I->dest[d], e->dst, nd);
         for (unsigned s = 0; s < I->nr_srcs; ++s)
            ns = agx3_expand_regs(I->src[s], e->src, ns);
         e->nr_dsts = (uint8_t)nd;
         e->nr_srcs = (uint8_t)ns;
         if ((I->op == AGX_OPCODE_STACK_LOAD ||
              I->op == AGX_OPCODE_STACK_STORE) &&
             !agx3_append_static_spill_tokens(I, e)) {
            /* Report like every other refusal.  Without this the whole
             * shader fails with no diagnostic at all, which is how a
             * scratch-lowered constant array once looked like a mystery
             * VK_ERROR_FEATURE_NOT_PRESENT from vkCreateComputePipelines. */
            agx3_report_refusal(I, __LINE__);
            goto fail;
         }
      }
   }
   assert(k == count);
   if (!agx3_slot_assign(sl, count, nslots, res))
      goto fail;

   k = 0;
   unsigned vertex_loads_seen = 0;
   unsigned loop_depth = 1;
   bool interpolation_started = false;
   bool coefficient_load_started = false;
   if (has_coverage_write && enc == AGX_ISA_ENCODING_AGX3B &&
       ctx->stage == MESA_SHADER_FRAGMENT &&
       !ctx->key->fs.ignore_tib_dependencies) {
      /* Native memory_discard enters before its first device load, then
       * uses unmarked SR/RHW inputs throughout the coverage operation.
       * This is the same existing entry gate, moved before the main part's
       * root loads; do not restart the input protocol after MOVMSK. */
      uint8_t entry[AGX3_PIXEL_MAX_LENGTH], entry_len;
      if (!agx3_pixel_colour_entry(enc, entry, &entry_len) ||
          !agx3_append(emission, entry, entry_len))
         goto fail;
      ctx->agx3_colour_link.entry_emitted = true;
      ctx->agx3_colour_link.entry_offset = emission->size - entry_len - initial_size;
      g17p_colour_open = true;
      interpolation_started = true;
   }

   agx_foreach_block(ctx, block) {
      block->offset = emission->size;
      bool after_loop_backedge = false;
      agx_foreach_instr_in_block(block, I) {
         block->last_offset = emission->size;

         if (I->op == AGX_OPCODE_STOP)
            continue;

         const uint8_t slot = res[k].slot;
         const uint16_t wait = res[k].wait_mask;
         k++;

         /* A wait retires its slots before this instruction issues its own. */
         outstanding &= (uint16_t)~wait;
         const uint16_t outstanding_here = outstanding;
         if (slot)
            outstanding |= (uint16_t)(1u << (slot - 1));

         /* Give every control-flow boundary an empty scoreboard, rather
          * than refusing the shader when something is still in flight. */
         uint16_t cf_wait = wait;
         uint16_t cf_outstanding = outstanding_here;
         if (agx3_is_cf_op(I)) {
            const uint16_t retire = (uint16_t)(cf_wait | cf_outstanding);
            if (retire != 0) {
               if (!agx3_retire_slots_noop(emission, enc, retire))
                  AGX3_REFUSE(I);
               cf_wait = 0;
               cf_outstanding = 0;
               outstanding = 0;
            }
         }

         const bool nested_loop_push = agx3_is_nested_loop_push(I);
         const bool loop_pop =
            I->op == AGX_OPCODE_POP_EXEC && after_loop_backedge;
         const int cf = agx3_try_control_flow(
            emission, &fixups, enc, I, cf_wait, cf_outstanding,
            after_loop_backedge, loop_depth);
         if (cf < 0)
            AGX3_REFUSE(I);
         if (cf > 0) {
            if (nested_loop_push)
               loop_depth++;
            else if (loop_pop && loop_depth > 1) {
               loop_depth--;
            }
            after_loop_backedge = I->op == AGX_OPCODE_JMP_EXEC_ANY;
            continue;
         }
         after_loop_backedge = false;

         /*
          * `export` is a zero-byte pseudo-op, exactly as on AGX2
          * (agx_pack.c:1242).  Register allocation has already turned it into
          * a parallel copy into its ABI register and rewritten the source to
          * that register (agx_register_allocate.c:1296-1310), so there is
          * nothing left to encode here. The slot pass must not treat this
          * pseudo-op as a hardware read; a no-stop part drains outstanding
          * results before handing its ABI registers to a separate epilog.
          */
         if (I->op == AGX_OPCODE_EXPORT)
            continue;

         if (I->op == AGX_OPCODE_INTERP_AGX3 ||
             I->op == AGX_OPCODE_RHW_INTERP_AGX3) {
            if (I->op == AGX_OPCODE_RHW_INTERP_AGX3 &&
                (enc != AGX_ISA_ENCODING_AGX3B ||
                 ctx->stage != MESA_SHADER_FRAGMENT ||
                 !agx3_private_rhw_use(ctx, block, I)))
               AGX3_REFUSE(I);
            uint16_t interp_wait = wait;
            /* The native projective-RHW opcode has no witnessed scoreboard
             * fields. Real GTK shaders reuse an LDCF destination for RHW,
             * which creates a required WAW wait. Retire that dependency with
             * the grounded band no-op before emitting the unchanged native
             * RHW instruction instead of inventing wait bits. */
            if (I->op == AGX_OPCODE_RHW_INTERP_AGX3 && interp_wait) {
               if (!agx3_retire_slots_noop(emission, enc, interp_wait))
                  AGX3_REFUSE(I);
               outstanding &= (uint16_t)~interp_wait;
               interp_wait = 0;
            }
            const bool first = enc == AGX_ISA_ENCODING_AGX3B &&
               ctx->stage == MESA_SHADER_FRAGMENT && !interpolation_started;
            if (!agx3_emit_interp(emission, enc, I, interp_wait, first,
                                  ctx->stage == MESA_SHADER_FRAGMENT))
               AGX3_REFUSE(I);
            interpolation_started = true;
            continue;
         }

         if (I->op == AGX_OPCODE_PERSP_RCP_AGX3) {
            if (I->nr_dests != 1 || I->nr_srcs != 1 || I->saturate ||
                I->imm > 0x7f || !agx3_vary_reg32(I->dest[0]) ||
                agx_channels(I->dest[0]) != 1 ||
                !agx3_vary_reg32(I->src[0]) ||
                agx_channels(I->src[0]) != 1 || I->src[0].kill ||
                (I->src[0].cache && I->src[0].discard))
               AGX3_REFUSE(I);
            const struct agx3_persp_rcp_desc desc = {
               .dest = I->dest[0].value,
               .rhw = I->src[0].value,
               .coef_index = I->imm,
               .rhw_cache = I->src[0].cache,
               .wait_mask = wait,
            };
            /* Post-RA discard is optional liveness. This operand leaf has no
             * discard bit: those bits carry the coefficient index instead. */
            uint8_t bytes[AGX3_FLTNORM_MAX_LENGTH], len = 0;
            if (!agx3_persp_rcp_encode(enc, &desc, bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         if (I->op == AGX_OPCODE_PERSP_MUL_AGX3) {
            if (I->nr_dests != 1 || I->nr_srcs != 2 || I->saturate ||
                I->imm > 0x7f || !agx3_vary_reg32(I->dest[0]) ||
                agx_channels(I->dest[0]) != 1 ||
                !agx3_vary_reg32(I->src[0]) ||
                !agx3_vary_reg32(I->src[1]) ||
                agx_channels(I->src[0]) != 1 ||
                agx_channels(I->src[1]) != 1 ||
                I->src[0].kill || I->src[1].kill ||
                (I->src[0].cache && I->src[0].discard) ||
                (I->src[1].cache && I->src[1].discard))
               AGX3_REFUSE(I);
            const struct agx3_persp_mul_desc desc = {
               .dest = I->dest[0].value,
               .raw = I->src[0].value,
               .reciprocal = I->src[1].value,
               .raw_liveness = I->src[0].discard ? AGX3_OPERAND_KILL :
                  I->src[0].cache ? AGX3_OPERAND_LIVE : AGX3_OPERAND_UNMARKED,
               .reciprocal_liveness = I->src[1].discard ? AGX3_OPERAND_KILL :
                  I->src[1].cache ? AGX3_OPERAND_LIVE : AGX3_OPERAND_UNMARKED,
               .vary_index = I->imm,
               .dest_hint = true,
               .wait_mask = wait,
            };
            uint8_t bytes[AGX3_FLTNORM_MAX_LENGTH], len = 0;
            if (!agx3_persp_mul_encode(enc, &desc, bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         if (I->op == AGX_OPCODE_LDCF) {
            const bool first = enc == AGX_ISA_ENCODING_AGX3B &&
               ctx->stage == MESA_SHADER_FRAGMENT && !coefficient_load_started;
            if (!agx3_emit_ldcf(emission, enc, I, slot, wait, first))
               AGX3_REFUSE(I);
            coefficient_load_started = true;
            continue;
         }

         if (I->op == AGX_OPCODE_ST_VARY) {
            /* Use a bounded staging allocation independent of GPR indices.
             * Each scalar is consumed immediately, before its slot/address
             * can be reused. A still-live general producer sharing the slot
             * must retire before the staging COPY reuses it. */
            const uint8_t stage_slot =
               (g17p_position_protocol || g17p_user_output_protocol)
                  ? I->src[0].value % nslots + 1 : next_uvs_slot;
            const uint16_t staging_half =
               2 * (g17p_user_output_protocol ? I->src[0].value : stage_slot - 1);
            const uint16_t staging_bit = 1u << (stage_slot - 1);
            uint16_t stage_wait;
            if (g17p_terminal_outputs) {
               /* The first batch drains original GPR producers. Each later
                * batch drains prior UVSWs before recycling their six slots.
                * Staging addresses remain distinct across both batches. */
               stage_wait = g17p_outputs_staged == 0 ? wait | outstanding_here :
                  g17p_outputs_staged % nslots == 0 ? (1u << nslots) - 1 : 0;
            } else {
               stage_wait = wait | (g17p_user_output_protocol ? outstanding_here :
                                     outstanding_here & staging_bit);
            }
            if (!agx3_emit_uvsw(emission, enc, I, staging_half, stage_slot,
                                stage_wait, g17p_position_protocol ||
                                            g17p_user_output_protocol))
               AGX3_REFUSE(I);
            ctx->agx3_staging_halfs =
               MAX2(ctx->agx3_staging_halfs, staging_half + 2);
            outstanding &= (uint16_t)~stage_wait;
            if (g17p_user_output_protocol) {
               /* Nonterminal stores can be followed by GPR/slot reuse, so
                * retain the existing immediate retirement there. A terminal
                * bundle has no intervening IR and uses unique slots per batch;
                * its original producers were all retired by the first COPY.
                * Each batch's UVSWs retire its staging reads instead. */
               if (!g17p_terminal_outputs &&
                   !agx3_retire_slots_noop(emission, enc, staging_bit))
                  AGX3_REFUSE(I);
               ++g17p_outputs_staged;
               if (g17p_outputs_staged % nslots == 0 ||
                   g17p_outputs_staged == g17p_output_count) {
                  const unsigned first =
                     ((g17p_outputs_staged - 1) / nslots) * nslots;
                  for (unsigned position = first;
                       position < g17p_outputs_staged; ++position) {
                     const unsigned component = g17p_terminal_outputs
                        ? g17p_output_order[position] : position;
                     uint8_t bytes[AGX3_VARY_MAX_LENGTH], len = 0;
                     if (!agx3_vary_uvsw(enc, component * 2, component,
                                          component % nslots + 1, true,
                                          1u << (component % nslots), bytes, &len) ||
                         !agx3_append(emission, bytes, len))
                        AGX3_REFUSE(I);
                  }
               }
            }
            /* Match the complete Apple float2 VS's producer-before-writeout
             * schedule: stage x/y before Mesa reuses their GPRs for z/w,
             * then emit the four UVSWs only after all four values exist.
             * This is the existing exact straight-line shape guard, not a
             * fixed shader-body replacement. */
            if (g17p_position_protocol && ++g17p_positions_staged == 4) {
               for (unsigned component = 0; component < 4; ++component) {
                  uint8_t bytes[AGX3_VARY_MAX_LENGTH], len = 0;
                  if (!agx3_vary_uvsw(enc, component * 2, component,
                                       component + 1, true, 1u << component,
                                       bytes, &len) ||
                      !agx3_append(emission, bytes, len))
                     AGX3_REFUSE(I);
               }
            }
            next_uvs_slot = (next_uvs_slot % nslots) + 1;
            continue;
         }

         if (I->op == AGX_OPCODE_TEXTURE_LOAD ||
             I->op == AGX_OPCODE_IMAGE_LOAD) {
            /* Native TEX reads staging coordinates, independently of GPRs.
             * This also applies to the background reload's pixel coordinates. */
            const bool staged = ctx->stage == MESA_SHADER_FRAGMENT ||
                                ctx->stage == MESA_SHADER_COMPUTE;
            if (staged) {
               const unsigned coord_base =
                  (I->op == AGX_OPCODE_IMAGE_LOAD &&
                   I->src[2].type == AGX_INDEX_IMMEDIATE) ||
                        I->dim == AGX_DIM_2D_MS_ARRAY
                     ? 16
                     : 0;
               const unsigned coord_count =
                  (I->dim == AGX_DIM_2D_MS ||
                   I->dim == AGX_DIM_2D_MS_ARRAY)
                     ? 2
                     : agx_channels(I->src[0]);
               ctx->agx3_staging_halfs = MAX2(
                  ctx->agx3_staging_halfs,
                  coord_base + 2 * coord_count);
            }
            else
               ctx->agx3_staging_known = false;
            assert(texture_ops_left > 0);
            const bool final = --texture_ops_left == 0;
            const uint16_t fetch_wait =
               wait | ((I->dim == AGX_DIM_2D_MS ||
                         I->dim == AGX_DIM_2D_MS_ARRAY)
                          ? (outstanding_here & 1u)
                          : 0);
            if (!agx3_emit_texture_fetch(emission, enc, I, slot, fetch_wait,
                                         final, staged))
               AGX3_REFUSE(I);
            continue;
         }

         if (I->op == AGX_OPCODE_IMAGE_WRITE) {
            if (!agx3_emit_storage_image_write(emission, enc, I, wait))
               AGX3_REFUSE(I);
            ctx->agx3_staging_halfs = MAX2(ctx->agx3_staging_halfs, 16);
            continue;
         }

         if (I->op == AGX_OPCODE_TEXTURE_SAMPLE) {
            assert(texture_ops_left > 0);
            const bool final = --texture_ops_left == 0;
            /* The implicit-LOD form needs the 2x2 quad's derivatives; every
             * witness is a fragment shader and the hardware has no quad
             * outside one. Apple's explicit immediate-level form is a
             * compute kernel witness and does not consume derivatives. */
            const bool compute_level = ctx->stage == MESA_SHADER_COMPUTE &&
               I->lod_mode == AGX_LOD_MODE_LOD_MIN &&
               I->src[1].type == AGX_INDEX_IMMEDIATE;
            /* An explicit gradient carries its own derivative, so it needs no
             * quad either; Apple's own witness for the form is a compute
             * kernel. */
            const bool compute_gradient = ctx->stage == MESA_SHADER_COMPUTE &&
               I->lod_mode == AGX_LOD_MODE_LOD_GRAD;
            if ((ctx->stage != MESA_SHADER_FRAGMENT && !compute_level &&
                 !compute_gradient) ||
                !agx3_emit_texture_sample(emission, enc, I, slot, wait,
                                          final))
               AGX3_REFUSE(I);
            /* The sample emitter stages exactly two 32-bit coordinates, plus
             * the auxiliary scalar bank for a per-lane level or bias, plus
             * the four gradient banks that reach half 14. */
            const bool staged_scalar =
               I->lod_mode == AGX_LOD_MODE_LOD_MIN ||
               (I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS &&
                I->src[1].type != AGX_INDEX_IMMEDIATE);
            ctx->agx3_staging_halfs =
               MAX2(ctx->agx3_staging_halfs,
                    I->lod_mode == AGX_LOD_MODE_LOD_GRAD ? 16
                    : (staged_scalar || I->dim == AGX_DIM_2D_ARRAY)
                       ? 8 : I->src[0].value + 4);
            continue;
         }

         if (I->op == AGX_OPCODE_WAIT_PIX ||
             I->op == AGX_OPCODE_SIGNAL_PIX) {
            /* A late depth test for fragment memory side effects stages the
             * pass-through Z value before acquiring class 1 in Apple's fresh
             * G17P atomic fragment. Hold this wait until ZS_EMIT can reproduce
             * that ordering; SIGNAL_PIX remains in its original position. */
            if (has_zs_emit && I->op == AGX_OPCODE_WAIT_PIX &&
                I->writeout == 1) {
               if (g17p_deferred_zs_wait || wait != 0)
                  AGX3_REFUSE(I);
               g17p_deferred_zs_wait = I;
               continue;
            }
            if (I == g17p_deferred_write) {
               /* The scoreboard has already retired these dependencies here.
                * Keep that retirement at the original instruction, independent
                * of where the pixel fence is emitted. */
               if (!agx3_retire_slots_noop(emission, enc, wait))
                  AGX3_REFUSE(I);
               continue;
            }
            if (g17p_pixel_protocol && I->op == AGX_OPCODE_WAIT_PIX &&
                (I->writeout == 1 ||
                 I->writeout == (g17p_blend_protocol ? 8 : 0xc)) &&
                !g17p_colour_open) {
               uint8_t entry[AGX3_PIXEL_MAX_LENGTH];
               uint8_t entry_len = 0;
               if (!agx3_pixel_colour_entry(enc, entry, &entry_len) ||
                   !agx3_append(emission, entry, entry_len))
                  AGX3_REFUSE(I);
               ctx->agx3_colour_link.entry_emitted = true;
               ctx->agx3_colour_link.entry_offset = emission->size - entry_len - initial_size;
               g17p_colour_open = true;
            }
            if (!agx3_emit_pixel(emission, enc, I, wait))
               AGX3_REFUSE(I);
            continue;
         }

         if (I->op == AGX_OPCODE_FLTNORM_AGX3) {
            const bool first = enc == AGX_ISA_ENCODING_AGX3B &&
               ctx->stage == MESA_SHADER_FRAGMENT &&
               !ctx->key->fs.ignore_tib_dependencies &&
               emission->size == initial_size;
            struct agx3_fltnorm_desc *link_desc =
               ctx->agx3_colour_link.opaque_pack ?
                  &ctx->agx3_colour_link.packs[g17p_link_packs_seen++] : NULL;
            if (!agx3_emit_fltnorm(emission, enc, I, wait, first, link_desc))
               AGX3_REFUSE(I);
            ctx->agx3_fltnorm_start |= first;
            continue;
         }

         if (I->op == AGX_OPCODE_PACK_RGB10A2_AGX3 ||
             I->op == AGX_OPCODE_UNPACK_RGB10A2_AGX3) {
            const bool first = ctx->stage == MESA_SHADER_FRAGMENT &&
               !ctx->key->fs.ignore_tib_dependencies &&
               emission->size == initial_size;
            if (enc != AGX_ISA_ENCODING_AGX3B ||
                !agx3_emit_rgb10a2(emission, I, wait, first))
               AGX3_REFUSE(I);
            ctx->agx3_fltnorm_start |= first &&
               I->op == AGX_OPCODE_PACK_RGB10A2_AGX3;
            continue;
         }

         if (I->op == AGX_OPCODE_NORMFLT_AGX3) {
            if (!agx3_emit_normflt(emission, enc, I, wait))
               AGX3_REFUSE(I);
            continue;
         }

         if (I->op == AGX_OPCODE_LD_TILE || I->op == AGX_OPCODE_ST_TILE) {
            bool first = false, last = false;
            if (I->op == AGX_OPCODE_LD_TILE) {
               assert(tile_loads_left > 0);
               first = tile_loads_seen++ == 0;
               last = --tile_loads_left == 0;
            } else {
               assert(tile_stores_left > 0);
               last = --tile_stores_left == 0;
            }
            if (I->op == AGX_OPCODE_ST_TILE && g17p_rgba8_mrt_colour) {
               if (wait != 0 || I->pixel_offset != (last ? 0 : 4))
                  AGX3_REFUSE(I);

               uint8_t bytes[AGX3_TILE_MAX_LENGTH];
               uint8_t len = 0;
               const uint32_t writeout = last ? 12 : 48;
               if (!g17p_colour_open) {
                  if (!agx3_pixel_colour_entry(enc, bytes, &len) ||
                      !agx3_append(emission, bytes, len))
                     AGX3_REFUSE(I);
                  ctx->agx3_colour_link.entry_emitted = true;
                  ctx->agx3_colour_link.entry_offset =
                     emission->size - len - initial_size;
                  g17p_colour_open = true;
               }
               if (!agx3_pixel_wait(enc, writeout, 0, bytes, &len) ||
                   !agx3_append(emission, bytes, len) ||
                   !agx3_emit_tile(emission, enc, I, slot, 0, false, false) ||
                   !agx3_pixel_release(enc, writeout, last, 0, bytes, &len) ||
                   !agx3_append(emission, bytes, len))
                  AGX3_REFUSE(I);
               continue;
            }
            if (I->op == AGX_OPCODE_ST_TILE && g17p_deferred_write) {
               if (!g17p_colour_open) {
                  uint8_t entry[AGX3_PIXEL_MAX_LENGTH], entry_len = 0;
                  if (!agx3_pixel_colour_entry(enc, entry, &entry_len) ||
                      !agx3_append(emission, entry, entry_len))
                     AGX3_REFUSE(I);
                  ctx->agx3_colour_link.entry_emitted = true;
                  ctx->agx3_colour_link.entry_offset = emission->size - entry_len - initial_size;
                  g17p_colour_open = true;
               }
               if (!agx3_emit_pixel(emission, enc, g17p_deferred_write, 0))
                  AGX3_REFUSE(I);
            }
            if (!agx3_emit_tile(emission, enc, I, slot, wait, first, last))
               AGX3_REFUSE(I);

            if (g17p_pixel_protocol && I->op == AGX_OPCODE_ST_TILE && last) {
               uint8_t release[AGX3_PIXEL_MAX_LENGTH];
               uint8_t release_len = 0;
               if (!g17p_colour_open ||
                   !agx3_pixel_release(enc, 0xc, true, 0, release,
                                      &release_len) ||
                   !agx3_append(emission, release, release_len))
                  AGX3_REFUSE(I);
            }
            continue;
         }

         /*
          * Preamble uniform write. Apple's own constant program writes its
          * uniforms with an ordinary `copy` whose f9 is 3 -- the destination
          * moves from f2 into f3 and becomes a uniform half. Witnessed
          * byte-exactly against a write to uniform half 24; see
          * agx3_copy_core.h.
          *
          * Each emitted copy is one grounded 32-bit component. A scalar
          * 64-bit value is therefore two consecutive witnessed copies, just
          * like a two-component 32-bit value; no unobserved wide form is used.
          */
         if (I->op == AGX_OPCODE_UNIFORM_STORE && I->nr_srcs == 2 &&
             (I->mask == 1 || I->mask == 3 || I->mask == 7 ||
              I->mask == 15)) {
            const agx_index value = I->src[0], index = I->src[1];
            const unsigned components = util_bitcount(I->mask);
            const bool scalar16 = value.size == AGX_SIZE_16 && I->mask == 1 &&
                                  agx_channels(value) == 1;
            const bool scalar64 = value.size == AGX_SIZE_64 && I->mask == 1 &&
                                  agx_channels(value) == 1 &&
                                  (value.value & 3) == 0;
            const bool vector32 = value.size == AGX_SIZE_32 &&
                                  agx_channels(value) == components;
            const unsigned copies = scalar64 ? 2 : components;
            /* Apple widens a scalar half preamble result to one aligned
             * 32-bit copy. Permit that exact substitution only when both
             * files are pair-aligned and the extra uniform half is padding;
             * otherwise a neighbouring scalar result could be overwritten. */
            const bool padded16 = scalar16 && !(value.value & 1) &&
                                  !(index.value & 1) && index.value + 2 <= 512 &&
                                  !uniform_half_used[index.value + 1];

            if (agx3_src_is_register(value) && !value.memory && !value.abs &&
                !value.neg && (padded16 || scalar64 || vector32) &&
                index.type == AGX_INDEX_IMMEDIATE && !index.abs &&
                !index.neg && index.value + 2 * copies <= 512) {
               /*
                * Apple's witnessed uniform write carries no wait bits, and
                * the wait bank for this form is not witnessed, so retire any
                * scoreboard dependency with the band no-op first rather than
                * guessing a field. Both halves are witnessed encodings.
                */
               /*
                * One copy per 32-bit component, exactly as Apple's constant
                * program does: its four uniform writes land on halves 20, 22,
                * 24 and 26 -- consecutive 32-bit slots two halves apart.
                */
               bool ok = wait == 0 || agx3_retire_slots_noop(emission, enc, wait);

               for (unsigned i = 0; ok && i < copies; ++i) {
                  uint8_t bytes[AGX3_COPY_MAX_LENGTH];
                  uint8_t len = 0;
                  const bool last = i + 1 == copies;
                  ok = agx3_copy_encode_reg_to_uniform(
                          enc, index.value + 2 * i, value.value + 2 * i,
                          /* A half-register discard does not prove that its
                           * paired upper half is dead. */
                          !padded16 && last && value.kill, bytes, &len) &&
                       agx3_append(emission, bytes, len);
               }

               if (ok) {
                  outstanding &= (uint16_t)~wait;
                  continue;
               }
            }
            AGX3_REFUSE(I);
         }

         if (I->op == AGX_OPCODE_GET_COVERAGE_1X_AGX3) {
            if (enc != AGX_ISA_ENCODING_AGX3B || ctx->stage != MESA_SHADER_FRAGMENT ||
                I->nr_dests != 1 || I->nr_srcs ||
                !agx3_src_is_register(I->dest[0]) || I->dest[0].memory ||
                I->dest[0].size != AGX_SIZE_16 || agx_channels(I->dest[0]) != 1)
               AGX3_REFUSE(I);
            const struct agx3_sr_desc d = {
               .dest = I->dest[0].value, .width = AGX3_SR_W16,
               .sr = 2, .slot = slot, .wait_mask = wait,
            };
            uint8_t bytes[AGX3_SR_MAX_LENGTH], len;
            if (!agx3_sr_encode(enc, &d, bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            interpolation_started = true;
            continue;
         }

         if (I->op == AGX_OPCODE_ZS_EMIT_AGX3) {
            if (enc != AGX_ISA_ENCODING_AGX3B ||
                ctx->stage != MESA_SHADER_FRAGMENT || I->nr_dests ||
                I->nr_srcs != 1 || I->zs != 1 ||
                !agx3_vary_reg32(I->src[0]) ||
                agx_channels(I->src[0]) != 1 || !g17p_colour_open)
               AGX3_REFUSE(I);

            uint8_t bytes[AGX3_STAGE_MAX_LENGTH];
            uint8_t len = 0;
            if (!agx3_stage_copy_source(enc, 0, I->src[0].value,
                                        I->src[0].discard, slot,
                                        wait | outstanding_here, bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);

            const struct agx3_zs_desc zs = {
               .value = 0,
               .value_kill = true,
               .slot = slot,
               .wait_mask = (uint16_t)(1u << (slot - 1)),
            };
            if (!agx3_zs_encode_depth_stencil(enc, &zs, true, false, bytes,
                                              &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);

            outstanding = 0;
            ctx->agx3_staging_halfs = MAX2(ctx->agx3_staging_halfs, 2);
            continue;
         }

         if (I->op == AGX_OPCODE_GET_SR_COVERAGE) {
            /* Fresh Apple G17P fragment-atomic pipeline
             * d150404ac02e.../32cf4963515f... reads SR2 as W16 before the
             * helper-thread predicate (`04 84 10 06`). This is the exact
             * active-lane source used to suppress fragment side effects on
             * helper invocations; other coverage pseudo-registers remain
             * closed until separately witnessed. */
            if (enc != AGX_ISA_ENCODING_AGX3B ||
                ctx->stage != MESA_SHADER_FRAGMENT ||
                I->sr != AGX_SR_IS_ACTIVE_THREAD || I->nr_dests != 1 ||
                I->nr_srcs != 0 || !agx3_src_is_register(I->dest[0]) ||
                I->dest[0].memory || I->dest[0].size != AGX_SIZE_16 ||
                agx_channels(I->dest[0]) != 1)
               AGX3_REFUSE(I);
            const struct agx3_sr_desc d = {
               .dest = I->dest[0].value, .width = AGX3_SR_W16,
               .sr = 2, .slot = slot, .wait_mask = wait,
            };
            uint8_t bytes[AGX3_SR_MAX_LENGTH], len = 0;
            if (!agx3_sr_encode(enc, &d, bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         if (I->op == AGX_OPCODE_ZS_EMIT) {
            /* Fragment stores without early tests inject a pass-through depth
             * write so side effects obey late depth testing. Apple's fresh
             * G17P fragment-atomic pipeline stages that scalar in staging[0]
             * on slot1, then emits pixwait(1), depthf(depth, wait slot1), and
             * the ordinary pixrel. Admit precisely that depth-only graph. */
            const agx_index samples = I->src[0], depth = I->src[1];
            if (enc != AGX_ISA_ENCODING_AGX3B ||
                ctx->stage != MESA_SHADER_FRAGMENT || I->nr_dests != 0 ||
                I->nr_srcs != 2 || I->zs != 1 ||
                samples.type != AGX_INDEX_IMMEDIATE ||
                samples.size != AGX_SIZE_16 || samples.value != 0xff ||
                !agx3_vary_reg32(depth) || agx_channels(depth) != 1 ||
                !depth.discard || depth.kill || depth.cache ||
                !g17p_deferred_zs_wait)
               AGX3_REFUSE(I);

            uint8_t bytes[AGX3_STAGE_MAX_LENGTH], len = 0;
            if (!g17p_colour_open) {
               if (!agx3_pixel_colour_entry(enc, bytes, &len) ||
                   !agx3_append(emission, bytes, len))
                  AGX3_REFUSE(I);
               ctx->agx3_colour_link.entry_emitted = true;
               ctx->agx3_colour_link.entry_offset =
                  emission->size - len - initial_size;
               g17p_colour_open = true;
            }
            if (!agx3_stage_copy_kill(enc, 0, depth.value, true, 1, wait,
                                      bytes, &len) ||
                !agx3_append(emission, bytes, len) ||
                !agx3_emit_pixel(emission, enc, g17p_deferred_zs_wait, 0))
               AGX3_REFUSE(I);
            const struct agx3_zs_desc d = {
               .value = 0, .value_kill = true, .slot = 1, .wait_mask = 1,
            };
            if (!agx3_zs_encode_depth_stencil(enc, &d, true, false, bytes,
                                               &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            outstanding &= (uint16_t)~wait;
            ctx->agx3_staging_halfs = MAX2(ctx->agx3_staging_halfs, 2);
            g17p_deferred_zs_wait = NULL;
            continue;
         }

         if (I->op == AGX_OPCODE_SAMPLE_MASK_1X_AGX3) {
            if (enc != AGX_ISA_ENCODING_AGX3B || ctx->stage != MESA_SHADER_FRAGMENT ||
                I->nr_dests || I->nr_srcs != 2 ||
                !agx3_src_is_register(I->src[0]) || I->src[0].memory ||
                I->src[0].size != AGX_SIZE_16 || agx_channels(I->src[0]) != 1 ||
                !agx3_vary_reg32(I->src[1]) || agx_channels(I->src[1]) != 1)
               AGX3_REFUSE(I);
            if (!g17p_colour_open) {
               uint8_t entry[AGX3_PIXEL_MAX_LENGTH], entry_len;
               if (!agx3_pixel_colour_entry(enc, entry, &entry_len) ||
                   !agx3_append(emission, entry, entry_len))
                  AGX3_REFUSE(I);
               ctx->agx3_colour_link.entry_emitted = true;
               ctx->agx3_colour_link.entry_offset = emission->size - entry_len - initial_size;
               g17p_colour_open = true;
            }
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
            if (!agx3_stage_copy_source(enc, 0, I->src[1].value,
                                         I->src[1].discard, 1,
                                         wait | outstanding_here, bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            /* Native simple and sampled-alpha fragments produce and stage
             * SR2's complete coverage word before acquiring class1. Keep
             * this ordering within the atomic pseudo rather than inheriting
             * the AGX2 wait-before-mask-calculation schedule. */
            if ((I->imm & 2) &&
                (!agx3_pixel_wait(enc, 1, 0, bytes, &len) ||
                 !agx3_append(emission, bytes, len)))
               AGX3_REFUSE(I);
            const struct agx3_zs_desc zs = {
               .value = 0, .value_kill = true, .slot = 1, .wait_mask = 1,
            };
            if (!agx3_zs_encode_sample_mask(enc, &zs, bytes, &len) ||
                !agx3_append(emission, bytes, len) ||
                !agx3_pixel_release(enc, 1, false, 0, bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            outstanding = 0;
            ctx->agx3_staging_halfs = MAX2(ctx->agx3_staging_halfs, 2);
            continue;
         }

         /* Special-register read. */
         if (I->op == AGX_OPCODE_GET_SR) {
            struct agx3_sr_desc d;
            uint8_t bytes[AGX3_SR_MAX_LENGTH];
            uint8_t len = 0;
            if (!agx3_translate_sr(I, slot, wait, &d) ||
                !agx3_sr_encode(enc, &d, bytes, &len))
               AGX3_REFUSE(I);
            memcpy(util_dynarray_grow_bytes(emission, 1, len), bytes, len);
            continue;
         }

         /* Device memory. */
         if (I->op == AGX_OPCODE_DEVICE_LOAD ||
             I->op == AGX_OPCODE_DEVICE_STORE) {
            const bool is_store = I->op == AGX_OPCODE_DEVICE_STORE;
            struct agx3_mem_desc d;
            uint8_t bytes[AGX3_MEM_MAX_LENGTH];
            uint8_t len = 0;
            if (!agx3_translate_mem(I, is_store, slot, wait, &d))
               AGX3_REFUSE(I);
            if (g17p_vertex_load_policy && !is_store) {
               const bool first = vertex_loads_seen++ == 0;
               const bool last = --device_loads_left == 0;
               if (first && d.slot != 6)
                  AGX3_REFUSE(I);
               d.agx3b_last_load = last;
            }
            if (g17p_position_protocol && !is_store) {
               /* These assertions are the descriptor-level half of the shape
                * guard above. Refuse if a later lowering changes it. */
               if (d.data != 0 || d.width != AGX3_MEM_W32 || d.mask != 3 ||
                   d.shift != 3 || !d.base_is_uniform || d.base != 0 ||
                   d.idx_kind != AGX3_MEM_IDX_REG || d.idx_reg != 0 ||
                   d.idx_signed || d.slot != 6 || d.wait_mask != 1 ||
                   !d.agx3b_last_load)
                  AGX3_REFUSE(I);
               d.idx_kill = true;
            }
            const bool ok = is_store
                               ? agx3_mem_encode_store(enc, &d, bytes, &len)
                               : agx3_mem_encode_load(enc, &d, bytes, &len);
            if (!ok)
               AGX3_REFUSE(I);
            memcpy(util_dynarray_grow_bytes(emission, 1, len), bytes, len);
            continue;
         }

         if (I->op == AGX_OPCODE_ATOMIC) {
            struct agx3_device_atomic_desc d;
            uint8_t bytes[AGX3_ATOMIC_MAX_LENGTH];
            uint8_t len = 0;
            if (!agx3_translate_device_atomic(I, enc, block->divergent,
                                              slot, wait, &d) ||
                !agx3_atomic_encode_device(enc, &d, bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         /* Threadgroup memory. The scaled register-indexed axis stays closed;
          * current Mesa IR reaches this path with its byte address in base and
          * an immediate-zero index. */
         if (I->op == AGX_OPCODE_LOCAL_LOAD ||
             I->op == AGX_OPCODE_LOCAL_STORE) {
            const bool is_store = I->op == AGX_OPCODE_LOCAL_STORE;
            struct agx3_local_desc d;
            uint8_t bytes[AGX3_LOCAL_TPS_MAX_LENGTH];
            uint8_t len = 0;
            if (!agx3_translate_local(I, is_store, slot, wait, &d))
               AGX3_REFUSE(I);
            const bool ok =
               is_store ? agx3_local_encode_store(enc, &d, bytes, &len)
                        : agx3_local_encode_load(enc, &d, bytes, &len);
            if (!ok)
               AGX3_REFUSE(I);
            memcpy(util_dynarray_grow_bytes(emission, 1, len), bytes, len);
            continue;
         }

         if (I->op == AGX_OPCODE_LOCAL_ATOMIC) {
            struct agx3_local_atomic_desc d;
            uint8_t bytes[AGX3_ATOMIC_MAX_LENGTH];
            uint8_t len = 0;
            if (!agx3_translate_local_atomic(I, slot, wait, &d) ||
                !agx3_atomic_encode_local(enc, &d, bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         if (I->op == AGX_OPCODE_SCS_LOAD_AGX3 ||
             I->op == AGX_OPCODE_SCS_STORE_AGX3 ||
             I->op == AGX_OPCODE_SCS_LOAD_UNIFORM_INDEXED_AGX3 ||
             I->op == AGX_OPCODE_SCS_STORE_UNIFORM_INDEXED_AGX3) {
            const bool is_store = I->op == AGX_OPCODE_SCS_STORE_AGX3 ||
                                  I->op ==
                                     AGX_OPCODE_SCS_STORE_UNIFORM_INDEXED_AGX3;
            const bool indexed =
               I->op == AGX_OPCODE_SCS_LOAD_UNIFORM_INDEXED_AGX3 ||
               I->op == AGX_OPCODE_SCS_STORE_UNIFORM_INDEXED_AGX3;
            struct agx3_scs_desc d;
            uint8_t bytes[AGX3_SCS_MAX_LENGTH];
            uint8_t len = 0;
            if (!agx3_translate_scs(I, is_store, indexed, slot, wait, &d))
               AGX3_REFUSE(I);
            const bool ok =
               is_store ? agx3_scs_encode_store(enc, &d, bytes, &len)
                        : agx3_scs_encode_load(enc, &d, bytes, &len);
            if (!ok)
               AGX3_REFUSE(I);
            memcpy(util_dynarray_grow_bytes(emission, 1, len), bytes, len);
            continue;
         }

         /* Static TPS spill/reload. Both copy and rld signal scoreboard slots;
          * their synthetic TPS dependency tokens make a reload wait the copy
          * that produced the same stack location. */
         if (I->op == AGX_OPCODE_STACK_LOAD ||
             I->op == AGX_OPCODE_STACK_STORE) {
            const bool is_store = I->op == AGX_OPCODE_STACK_STORE;
            struct agx3_static_spill_store_desc store;
            struct agx3_static_spill_load_desc load;
            uint8_t bytes[AGX3_STATIC_SPILL_MAX_LENGTH];
            uint8_t len = 0;
            if (!agx3_translate_static_spill(I, is_store, slot, wait, &store,
                                             &load))
               AGX3_REFUSE(I);
            const bool ok =
               is_store ? agx3_static_spill_encode_store(enc, &store, bytes,
                                                          &len)
                        : agx3_static_spill_encode_load(enc, &load, bytes,
                                                        &len);
            if (!ok || !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         bool subgroup_quad;
         enum agx3_subgroup_shuffle_op subgroup_shuffle;
         if (agx3_map_shuffle(I->op, &subgroup_quad, &subgroup_shuffle)) {
            if (!agx3_emit_subgroup_shuffle(emission, enc, I, wait))
               AGX3_REFUSE(I);
            continue;
         }

         if (I->op == AGX_OPCODE_SIMD_REDUCE ||
             I->op == AGX_OPCODE_QUAD_REDUCE ||
             I->op == AGX_OPCODE_SIMD_PREFIX ||
             I->op == AGX_OPCODE_QUAD_PREFIX) {
            if (!agx3_emit_subgroup_reduce(emission, enc, I, wait))
               AGX3_REFUSE(I);
            continue;
         }

         if (I->op == AGX_OPCODE_ICMP_BALLOT ||
             I->op == AGX_OPCODE_ICMP_QUAD_BALLOT ||
             I->op == AGX_OPCODE_FCMP_BALLOT ||
             I->op == AGX_OPCODE_FCMP_QUAD_BALLOT) {
            if (!agx3_emit_subgroup_ballot(emission, enc, I, wait))
               AGX3_REFUSE(I);
            continue;
         }

         if (I->op == AGX_OPCODE_SCOPED_BARRIER_AGX3) {
            if (!agx3_emit_scoped_barrier(emission, enc, I, wait))
               AGX3_REFUSE(I);
            continue;
         }

         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_SRSQRT) {
            const agx_index dest=I->dest[0], src=I->src[0];
            if (I->nr_dests != 1 || I->nr_srcs != 1 || I->saturate ||
                !agx3_vary_reg32(dest) ||
                !agx3_src_is_register(src) || src.memory ||
                agx_channels(dest) != 1 || agx_channels(src) != 1 ||
                src.abs || src.neg || src.kill ||
                (src.cache && src.discard))
               AGX3_REFUSE(I);
            uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
            const bool half = src.size == AGX_SIZE_16;
            if ((!half && !agx3_vary_reg32(src)) ||
                !(half ? agx3_alu_sqrt_step_half : agx3_alu_sqrt_step)(
                   enc,dest.value,src.value,agx3_post_ra_hint(src),
                   dest.cache,wait,bytes,&len) ||
                !agx3_append(emission,bytes,len))
               AGX3_REFUSE(I);
            continue;
         }

         /* Reciprocal, at either width. Only the half form was wired up, so a
          * 32-bit float divide -- which NIR lowers to frcp -- refused the
          * whole shader. f26 selects the source width; the 32-bit encoding is
          * witnessed against Apple's own `1.0f / a[i]` (agx3_alu_core.h). */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_RCP &&
             I->nr_dests == 1 && I->nr_srcs == 1 && !I->saturate) {
            const agx_index dest=I->dest[0], src=I->src[0];
            const bool half = dest.size == AGX_SIZE_16;
            if (!agx3_src_is_register(dest) || dest.memory ||
                (dest.size != AGX_SIZE_16 && dest.size != AGX_SIZE_32) ||
                agx_channels(dest) != 1 ||
                !agx3_src_is_register(src) || src.memory ||
                src.size != dest.size || agx_channels(src) != 1 ||
                src.abs || src.neg || src.kill || (src.cache && src.discard))
               AGX3_REFUSE(I);
            uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
            if (!(half ? agx3_alu_hrcp : agx3_alu_f32rcp)(
                   enc,dest.value,src.value,agx3_post_ra_hint(src),
                   dest.cache,wait,bytes,&len) ||
                !agx3_append(emission,bytes,len))
               AGX3_REFUSE(I);
            continue;
         }

         /* Reciprocal square root, at either width. normalize() lowers to a
          * 32-bit frsq, so the half-only gate refused every shader that
          * normalizes a vector. */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_RSQRT &&
             I->nr_dests == 1 && I->nr_srcs == 1 && !I->saturate) {
            const agx_index dest = I->dest[0], src = I->src[0];
            const bool half = dest.size == AGX_SIZE_16;
            if (!agx3_src_is_register(dest) || dest.memory ||
                (dest.size != AGX_SIZE_16 && dest.size != AGX_SIZE_32) ||
                agx_channels(dest) != 1 ||
                !agx3_src_is_register(src) || src.memory ||
                src.size != dest.size || agx_channels(src) != 1 ||
                src.abs || src.neg || src.kill ||
                (src.cache && src.discard))
               AGX3_REFUSE(I);
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
            if (!(half ? agx3_alu_hrsqrt : agx3_alu_f32rsqrt)(
                   enc, dest.value, src.value,
                   agx3_post_ra_hint(src), dest.cache, wait,
                   bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         /* f30 and f31 are the source's absolute-value and negate modifiers,
          * witnessed as single-field differences on this exact form. NIR
          * folds a negation into a float source, so exp(-x) -- vkQuake's fog
          * term, in every one of its fragment shaders -- arrives here as an
          * exp2 of a negated register and refused the shader. Only the 32-bit
          * modifier forms are wired up: that is where both the witnesses and
          * the refusals are. */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_EXP2 &&
             I->nr_dests == 1 && I->nr_srcs == 1 && !I->saturate) {
            const agx_index dest = I->dest[0], src = I->src[0];
            const bool half = dest.size == AGX_SIZE_16;
            if (!agx3_src_is_register(dest) || dest.memory ||
                (dest.size != AGX_SIZE_16 && dest.size != AGX_SIZE_32) ||
                agx_channels(dest) != 1 ||
                !agx3_src_is_register(src) || src.memory ||
                src.size != dest.size || agx_channels(src) != 1 ||
                (half && (src.abs || src.neg)) || src.kill ||
                (src.cache && src.discard))
               AGX3_REFUSE(I);
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
            if (!(half ? agx3_alu_hexp2(enc, dest.value, src.value,
                                        agx3_post_ra_hint(src), dest.cache,
                                        wait, bytes, &len)
                       : agx3_alu_f32exp2_mod(enc, dest.value, src.value,
                                              agx3_post_ra_hint(src),
                                              dest.cache, wait, src.neg,
                                              src.abs, bytes, &len)) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_LOG2 &&
             I->nr_dests == 1 && I->nr_srcs == 1 && !I->saturate) {
            const agx_index dest = I->dest[0], src = I->src[0];
            const bool half = dest.size == AGX_SIZE_16;
            if (!agx3_src_is_register(dest) || dest.memory ||
                (dest.size != AGX_SIZE_16 && dest.size != AGX_SIZE_32) ||
                agx_channels(dest) != 1 ||
                !agx3_src_is_register(src) || src.memory ||
                src.size != dest.size || agx_channels(src) != 1 ||
                (half && (src.abs || src.neg)) || src.kill ||
                (src.cache && src.discard))
               AGX3_REFUSE(I);
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
            if (!(half ? agx3_alu_hlog2(enc, dest.value, src.value,
                                        agx3_post_ra_hint(src), dest.cache,
                                        wait, bytes, &len)
                       : agx3_alu_f32log2_mod(enc, dest.value, src.value,
                                              agx3_post_ra_hint(src),
                                              dest.cache, wait, src.neg,
                                              src.abs, bytes, &len)) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         /* floor() and ceil() at either width. Only the half forms were wired
          * up, so a 32-bit floor -- the quantized lightmap coordinate in
          * vkQuake's world shader -- refused the shader outright. The 32-bit
          * form is the half one with the width selectors raised, and carries
          * the same source modifiers as the rest of this form family. */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_FLOOR &&
             I->nr_dests == 1 && I->nr_srcs == 1 && !I->saturate) {
            const agx_index dest = I->dest[0], src = I->src[0];
            const bool half = dest.size == AGX_SIZE_16;
            if (!agx3_src_is_register(dest) || dest.memory ||
                (dest.size != AGX_SIZE_16 && dest.size != AGX_SIZE_32) ||
                agx_channels(dest) != 1 ||
                !agx3_src_is_register(src) || src.memory ||
                src.size != dest.size || agx_channels(src) != 1 ||
                (half && (src.abs || src.neg)) || src.kill ||
                (src.cache && src.discard))
               AGX3_REFUSE(I);
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
            if (!(half ? agx3_alu_hfloor(enc, dest.value, src.value,
                                         agx3_post_ra_hint(src), dest.cache,
                                         wait, bytes, &len)
                       : agx3_alu_f32floor(enc, dest.value, src.value,
                                           agx3_post_ra_hint(src), dest.cache,
                                           wait, src.neg, src.abs, bytes,
                                           &len)) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_CEIL &&
             I->nr_dests == 1 && I->nr_srcs == 1 && !I->saturate) {
            const agx_index dest = I->dest[0], src = I->src[0];
            const bool half = dest.size == AGX_SIZE_16;
            if (!agx3_src_is_register(dest) || dest.memory ||
                (dest.size != AGX_SIZE_16 && dest.size != AGX_SIZE_32) ||
                agx_channels(dest) != 1 ||
                !agx3_src_is_register(src) || src.memory ||
                src.size != dest.size || agx_channels(src) != 1 ||
                (half && (src.abs || src.neg)) || src.kill ||
                (src.cache && src.discard))
               AGX3_REFUSE(I);
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
            if (!(half ? agx3_alu_hceil(enc, dest.value, src.value,
                                        agx3_post_ra_hint(src), dest.cache,
                                        wait, bytes, &len)
                       : agx3_alu_f32ceil(enc, dest.value, src.value,
                                          agx3_post_ra_hint(src), dest.cache,
                                          wait, src.neg, src.abs, bytes,
                                          &len)) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         /* Transcendental / rounding / gradient funops. The gradients need
          * the 2x2 quad, and every witness is a fragment shader. */
         const int fop = agx3_map_funop(I->op);
         if (fop >= 0) {
            uint16_t funop_wait = wait;
            /* Apple gradient witnesses expose no wait-field map. A real
             * desktop shader may nevertheless reach DFDX/DFDY with an
             * outstanding interpolation/coefficient dependency. Retire it
             * through the already-grounded scoreboard no-op, then preserve
             * the byte-exact zero-wait gradient encoding. */
            if ((fop == AGX3_FUNOP_DFDX || fop == AGX3_FUNOP_DFDY) &&
                funop_wait) {
               if (!agx3_retire_slots_noop(emission, enc, funop_wait))
                  AGX3_REFUSE(I);
               outstanding &= (uint16_t)~funop_wait;
               funop_wait = 0;
            }
            if (((fop == AGX3_FUNOP_DFDX || fop == AGX3_FUNOP_DFDY) &&
                 ctx->stage != MESA_SHADER_FRAGMENT) ||
                !agx3_emit_funop(emission, enc, I, (enum agx3_funop)fop,
                                 funop_wait))
               AGX3_REFUSE(I);
            continue;
         }

         /*
          * A zero immediate has its own instruction, `clr`, at both widths.
          * Apple emits it wherever a register is set to zero, and it is the
          * compact grounded way to materialize a 16-bit zero.
          */
         if (I->op == AGX_OPCODE_MOV_IMM && I->imm == 0 && wait == 0 &&
             I->nr_dests == 1 && I->nr_srcs == 0 &&
             agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
             agx_channels(I->dest[0]) == 1 &&
             (I->dest[0].size == AGX_SIZE_16 ||
              I->dest[0].size == AGX_SIZE_32)) {
            uint8_t bytes[AGX3_STAGE_MAX_LENGTH];
            uint8_t len = 0;
            if (agx3_clr(enc, I->dest[0].value,
                         I->dest[0].size == AGX_SIZE_16, false, bytes, &len)) {
               if (!agx3_append(emission, bytes, len))
                  AGX3_REFUSE(I);
               continue;
            }
            /* Fall through to ldimm when clr cannot express it. */
         }

         /*
          * Uniform -> GPR.  AGX3 ALU operands are register-or-immediate, so
          * agx_lower_uniform_sources() materializes every generic-ALU uniform
          * source as a move; agx_lower_pseudo() then turns that move into a
          * bitop whose source is still a uniform.  Apple moves a uniform into
          * a register with `copy` -- 49 of the 100 G16G corpus copies do
          * exactly this -- so encode that rather than refusing the bitop.
          * Without it no shader that reads a pushed uniform in an ALU op can
          * be compiled at all.
          */
         /*
          * `imad dst32, srcA, #imm, srcC` -- a constant multiplier, which is
          * how every index-derived address is formed (`gid * stride + base`).
          * Like the iadd below it, this used to reach the wait-carrying
          * composer, which would have deposited the multiplier into a
          * register field.
          */
         if (I->op == AGX_OPCODE_IMAD && I->nr_dests == 1 && I->nr_srcs >= 3 &&
             I->shift == 0 && agx3_src_is_register(I->dest[0]) &&
             !I->dest[0].memory &&
             (I->dest[0].size == AGX_SIZE_16 ||
              I->dest[0].size == AGX_SIZE_32 ||
              I->dest[0].size == AGX_SIZE_64) &&
             agx_channels(I->dest[0]) == 1) {
            const agx_index a = I->src[0], m = I->src[1], c = I->src[2];

            /*
             * 32x32 -> 64. This is how `umul_high`, `imul_high` and NIR's
             * `*mul_2x32_64` all reach the backend, and therefore how every
             * integer division by a non-constant does. The source-size
             * convention is the one agx_pack.c already uses for AGX2: an
             * operand narrower than the result is zero-extended when it
             * carries `abs` and sign-extended otherwise, which maps onto the
             * per-slot sign-extend bits f28/f43/f58.
             */
            const bool narrow = I->dest[0].size == AGX_SIZE_16;
            const bool wide = I->dest[0].size == AGX_SIZE_64;

            /* The multiplier is either an 8-bit immediate or a 32-bit
             * register; both are witnessed (test-agx3-iadd-core.c). */
            const bool mult_imm = m.type == AGX_INDEX_IMMEDIATE && !m.abs &&
                                  !m.neg && m.value <= 0xff;
            const bool mult_reg = agx3_src_is_register(m) && !m.memory &&
                                  (wide || !m.abs) && !m.neg &&
                                  m.size == AGX_SIZE_32 &&
                                  agx_channels(m) == 1;

            /* An immediate-zero addend is the pure-multiply form, whose
             * addend operand is absent rather than zero. Native constant
             * and register multiplier variants have distinct baselines. */
            const bool no_addend = c.type == AGX_INDEX_IMMEDIATE &&
                                   c.value == 0 && !c.abs && !c.neg;
            const bool addend_imm =
               c.type == AGX_INDEX_IMMEDIATE && c.value != 0 &&
               c.value <= 0xff && !c.abs && !c.neg && !c.memory;
            const bool addend_ok =
               no_addend || addend_imm ||
               (agx3_src_is_register(c) && !c.memory &&
                agx_channels(c) == 1 && !(c.abs && c.neg) &&
                (c.size == AGX_SIZE_32 || (wide && c.size == AGX_SIZE_64) ||
                 (c.size == AGX_SIZE_16 && (c.abs || c.neg))));
            const bool addend_reg = !no_addend && !addend_imm;

            const bool a_ok = agx3_src_is_register(a) && !a.memory &&
               agx_channels(a) == 1 &&
               (narrow ? (a.size == AGX_SIZE_16 && !a.abs && !a.neg)
                       : ((wide || !a.abs) && !a.neg &&
                          a.size == AGX_SIZE_32));

            if (a_ok &&
                (mult_imm || mult_reg) && addend_ok) {
               const struct agx3_imad_desc d = {
                  .dest = I->dest[0].value,
                  .dest_16 = narrow,
                  .dest_64 = wide,
                  .dest_cache = I->dest[0].cache,
                  .src_a = a.value,
                  .src_a_16 = narrow,
                  .src_a_sign_extend = wide && !a.abs,
                  .multiplier = mult_imm ? (uint8_t)m.value : 0,
                  .multiplier_is_register = mult_reg,
                  .multiplier_reg = mult_reg ? m.value : 0,
                  .multiplier_kill = mult_reg && m.discard,
                  .multiplier_cache = mult_reg && m.cache,
                  .multiplier_sign_extend = wide && mult_reg && !m.abs,
                  .src_a_kill = a.discard,
                  .src_a_cache = a.cache,
                  .src_c_kill = addend_reg && c.discard,
                  .src_c_cache = addend_reg && c.cache,
                  .no_addend = no_addend,
                  .addend_is_immediate = addend_imm,
                  .addend_immediate = addend_imm ? c.value : 0,
                  .src_c = addend_reg ? c.value : 0,
                  .src_c_16 = addend_reg && c.size == AGX_SIZE_16,
                  .src_c_64 = addend_reg && c.size == AGX_SIZE_64,
                  .src_c_sign_extend =
                     addend_reg && (wide ? (c.size != AGX_SIZE_64 && !c.abs)
                                         : c.neg),
                  .wait_mask = wait,
               };
               uint8_t bytes[AGX3_IADD_MAX_LENGTH];
               uint8_t len = 0;
               if (agx3_imad_encode(enc, &d, bytes, &len) &&
                   agx3_append(emission, bytes, len))
                  continue;
            }
         }

         /*
          * Native tied 32-bit comparison/selection is fmin/fmax.  This is the
          * shape NIR hands us for max(x, 0.0) -- normalize(), clamp(),
          * saturate() and the discard compare -- and Apple encodes it as a
          * compact six-byte fsel, taking the register form when the second
          * operand is a register and the minifloat form when it is a
          * constant.  Both were witnessed byte-exactly; see agx3_alu_core.h.
          */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_FCMPSEL &&
             I->nr_dests == 1 && I->nr_srcs == 4 && !I->invert_cond &&
             (I->fcond == AGX_FCOND_LTN || I->fcond == AGX_FCOND_GTN) &&
             !I->saturate &&
             agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
             I->dest[0].size == AGX_SIZE_32 && agx_channels(I->dest[0]) == 1 &&
             I->dest[0].value < 256) {
            /*
             * A minimum and a maximum are commutative -- NIR declares
             * fmin_agx and fmax_agx 2src_commutative -- and a tied GTN/LTN
             * fcmpsel can only have come from one of them, since every other
             * condition a comparison produces is ordered. So the constant may
             * be written on either side, and max(0.0, x) normalizes to the
             * max(x, 0.0) Apple encodes.
             */
            const unsigned reg_pair =
               (agx3_src_is_register(I->src[0]) && !I->src[0].memory &&
                I->src[0].size == AGX_SIZE_32) ? 0 : 1;
            const agx_index reg_a = I->src[reg_pair];
            const agx_index reg_b = I->src[reg_pair + 2];
            const agx_index other_a = I->src[reg_pair ^ 1];
            const agx_index other_b = I->src[(reg_pair ^ 1) + 2];

            const bool tied_reg =
               agx3_src_is_register(reg_a) && agx3_src_is_register(reg_b) &&
               !reg_a.memory && !reg_b.memory &&
               reg_a.size == AGX_SIZE_32 && reg_b.size == AGX_SIZE_32 &&
               agx_channels(reg_a) == 1 && agx_channels(reg_b) == 1 &&
               reg_a.value == reg_b.value && reg_a.value < 256 &&
               agx3_post_ra_hint(reg_a) == agx3_post_ra_hint(reg_b);
            const bool second_imm =
               other_a.type == AGX_INDEX_IMMEDIATE &&
               other_b.type == AGX_INDEX_IMMEDIATE &&
               other_a.value == other_b.value && other_a.value <= 255;
            const bool second_reg =
               !second_imm && agx3_src_is_register(other_a) &&
               agx3_src_is_register(other_b) && !other_a.memory &&
               !other_b.memory && other_a.size == AGX_SIZE_32 &&
               other_b.size == AGX_SIZE_32 &&
               agx_channels(other_a) == 1 && agx_channels(other_b) == 1 &&
               other_a.value == other_b.value && other_a.value < 256 &&
               agx3_post_ra_hint(other_a) == agx3_post_ra_hint(other_b);
            bool clean = tied_reg;
            for (unsigned i = 0; i < 4; ++i) {
               clean &= !I->src[i].abs && !I->src[i].neg && !I->src[i].kill &&
                        !(I->src[i].cache && I->src[i].discard);
            }

            if ((second_imm || second_reg) && clean) {
               /* This form carries one scoreboard slot; retire anything wider
                * first, exactly as the other fsel paths do. */
               uint16_t fsel_wait = wait;
               if (fsel_wait & (fsel_wait - 1)) {
                  if (!agx3_retire_slots_noop(emission, enc, fsel_wait))
                     AGX3_REFUSE(I);
                  fsel_wait = 0;
               }

               const bool maximum = I->fcond == AGX_FCOND_GTN;
               uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
               bool composed;
               if (second_imm) {
                  composed =
                     maximum
                        ? agx3_alu_fmax_f32_imm(
                             enc, I->dest[0].value, reg_a.value,
                             (uint8_t)other_a.value, agx3_post_ra_hint(reg_a),
                             I->dest[0].cache, fsel_wait, bytes, &len)
                        : agx3_alu_fmin_f32_imm(
                             enc, I->dest[0].value, reg_a.value,
                             (uint8_t)other_a.value, agx3_post_ra_hint(reg_a),
                             I->dest[0].cache, fsel_wait, bytes, &len);
               } else {
                  const uint16_t src[4] = {
                     reg_a.value, other_a.value, reg_b.value, other_b.value};
                  const enum agx3_alu_hint hints[4] = {
                     agx3_post_ra_hint(reg_a), agx3_post_ra_hint(other_a),
                     agx3_post_ra_hint(reg_b), agx3_post_ra_hint(other_b)};
                  composed =
                     maximum
                        ? agx3_alu_fmax_f32(enc, I->dest[0].value, src, hints,
                                            I->dest[0].cache, fsel_wait, bytes,
                                            &len)
                        : agx3_alu_fmin_f32(enc, I->dest[0].value, src, hints,
                                            I->dest[0].cache, fsel_wait, bytes,
                                            &len);
               }
               if (!composed || !agx3_append(emission, bytes, len))
                  AGX3_REFUSE(I);
               continue;
            }
         }

         /* Native tied half comparison/selection is scalar fmin/fmax. */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_FCMPSEL &&
             I->nr_dests == 1 && I->nr_srcs == 4 && !I->invert_cond &&
             (I->fcond == AGX_FCOND_LTN || I->fcond == AGX_FCOND_GTN) &&
             !I->saturate &&
             agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
             I->dest[0].size == AGX_SIZE_16 && agx_channels(I->dest[0]) == 1 &&
             I->src[0].value == I->src[2].value &&
             I->src[1].value == I->src[3].value &&
             agx3_fcmpsel_all_half_regs(I)) {
            uint16_t src[4];
            enum agx3_alu_hint hints[4];
            for (unsigned i = 0; i < 4; ++i) {
               src[i] = I->src[i].value;
               hints[i] = agx3_post_ra_hint(I->src[i]);
            }
            if (wait != 0 && wait != 32)
               AGX3_REFUSE(I);
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
            const bool composed = I->fcond == AGX_FCOND_GTN
               ? agx3_alu_fmax_half(enc, I->dest[0].value, src, hints,
                                    I->dest[0].cache, wait, bytes, &len)
               : agx3_alu_fmin_half(enc, I->dest[0].value, src, hints,
                                    I->dest[0].cache, wait, bytes, &len);
            if (!composed ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         /* Hyprland's blur/mask paths compare f32 against a minifloat and
          * select either one full register plus zero, or two half-width byte
          * constants. Both shapes are direct native FSEL forms. */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_FCMPSEL &&
             I->nr_dests == 1 && I->nr_srcs == 4 && !I->invert_cond &&
             !I->saturate &&
             (I->fcond == AGX_FCOND_EQ || I->fcond == AGX_FCOND_LT ||
              I->fcond == AGX_FCOND_GE) &&
             agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
             (I->dest[0].size == AGX_SIZE_16 ||
              I->dest[0].size == AGX_SIZE_32) &&
             agx_channels(I->dest[0]) == 1 &&
             agx3_vary_reg32(I->src[0]) &&
             I->src[1].type == AGX_INDEX_IMMEDIATE &&
             I->src[1].value <= UINT8_MAX &&
             ((I->dest[0].size == AGX_SIZE_16 &&
               I->src[2].type == AGX_INDEX_IMMEDIATE &&
               I->src[3].type == AGX_INDEX_IMMEDIATE) ||
              (I->dest[0].size == AGX_SIZE_32 &&
               ((I->src[2].type == AGX_INDEX_IMMEDIATE) !=
                (I->src[3].type == AGX_INDEX_IMMEDIATE))))) {
            const bool dest_16 = I->dest[0].size == AGX_SIZE_16;
            bool selected_immediate[2];
            uint16_t selected[2];
            enum agx3_alu_hint hints[3] = {
               agx3_post_ra_hint(I->src[0]),
               AGX3_ALU_HINT_NONE,
               AGX3_ALU_HINT_NONE,
            };
            bool invalid = false;
            for (unsigned i = 0; i < 4; ++i)
               invalid |= I->src[i].memory || I->src[i].abs ||
                          I->src[i].neg || I->src[i].kill ||
                          (I->src[i].cache && I->src[i].discard) ||
                          agx_channels(I->src[i]) != 1;
            for (unsigned i = 0; i < 2; ++i) {
               const agx_index value = I->src[i + 2];
               selected_immediate[i] = value.type == AGX_INDEX_IMMEDIATE;
               invalid |= selected_immediate[i]
                             ? value.value > UINT8_MAX
                             : (!agx3_src_is_register(value) ||
                                value.size != I->dest[0].size);
               selected[i] = value.value;
               if (!selected_immediate[i])
                  hints[i + 1] = agx3_post_ra_hint(value);
            }

            uint16_t fsel_wait = wait;
            if (fsel_wait != 0 && fsel_wait != 32) {
               if (!agx3_retire_slots_noop(emission, enc, fsel_wait))
                  AGX3_REFUSE(I);
               fsel_wait = 0;
            }
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
            if (!invalid &&
                agx3_alu_fcmp_minifloat_select_mixed(
                   enc, I->dest[0].value, I->src[0].value,
                   I->src[1].value, selected, selected_immediate, hints,
                   I->fcond, dest_16, I->dest[0].cache, fsel_wait, bytes,
                   &len)) {
               if (!agx3_append(emission, bytes, len))
                  AGX3_REFUSE(I);
               continue;
            }
         }

         /* Native scalar half comparison/select.  This is the general
          * shaderFloat16 ternary form: compare two half registers and select
          * either of two half registers without widening or materializing a
          * boolean.  Keep every operand's physical half and lifetime hint.
          */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_FCMPSEL &&
             I->nr_dests == 1 && I->nr_srcs == 4 && !I->saturate &&
             agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
             I->dest[0].size == AGX_SIZE_16 &&
             agx_channels(I->dest[0]) == 1 &&
             agx3_fcmpsel_all_half_regs(I)) {
            uint16_t src[4];
            enum agx3_alu_hint hints[4];
            for (unsigned i = 0; i < 4; ++i) {
               src[i] = I->src[i].value;
               hints[i] = agx3_post_ra_hint(I->src[i]);
            }

            unsigned condition = I->fcond;
            bool swap = I->invert_cond;
            if (condition == AGX_FCOND_LTN) {
               condition = AGX_FCOND_GE;
               swap = !swap;
            } else if (condition == AGX_FCOND_GTN) {
               condition = AGX_FCOND_LE;
               swap = !swap;
            }
            if (swap) {
               const uint16_t value = src[2];
               const enum agx3_alu_hint hint = hints[2];
               src[2] = src[3];
               src[3] = value;
               hints[2] = hints[3];
               hints[3] = hint;
            }

            uint16_t fsel_wait = wait;
            if (fsel_wait != 0) {
               if (!agx3_retire_slots_noop(emission, enc, fsel_wait))
                  AGX3_REFUSE(I);
               fsel_wait = 0;
            }
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
            if (!agx3_alu_fcmp_select_half(
                   enc, I->dest[0].value, src, hints, condition,
                   I->dest[0].cache, fsel_wait, bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         /* Native mixed-width comparison against floating zero with full
          * float alternatives. Keep the half comparator in its assigned half
          * register; neither widening nor a separate comparison is needed. */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_FCMPSEL &&
             I->nr_dests == 1 && I->nr_srcs == 4 &&
             agx3_mod_src_ok(I->dest[0]) && agx_channels(I->dest[0]) == 1 &&
             agx3_mod_src_ok(I->src[2]) && agx3_mod_src_ok(I->src[3]) &&
             agx_channels(I->src[2]) == 1 && agx_channels(I->src[3]) == 1) {
            const bool zero_first = I->src[0].type == AGX_INDEX_IMMEDIATE;
            const agx_index zero = I->src[zero_first ? 0 : 1];
            const agx_index compare = I->src[zero_first ? 1 : 0];
            if (zero.type == AGX_INDEX_IMMEDIATE && zero.value == 0 &&
                compare.type == AGX_INDEX_REGISTER && !compare.memory &&
                agx_channels(compare) == 1 &&
                (compare.size == AGX_SIZE_16 || compare.size == AGX_SIZE_32)) {
               unsigned condition = I->fcond;
               bool swap = I->invert_cond;
               if (condition == AGX_FCOND_LTN) {
                  condition = AGX_FCOND_GE;
                  swap = !swap;
               } else if (condition == AGX_FCOND_GTN) {
                  condition = AGX_FCOND_LE;
                  swap = !swap;
               }
               if (zero_first) {
                  switch (condition) {
                  case AGX_FCOND_LT: condition = AGX_FCOND_GT; break;
                  case AGX_FCOND_GT: condition = AGX_FCOND_LT; break;
                  case AGX_FCOND_LE: condition = AGX_FCOND_GE; break;
                  case AGX_FCOND_GE: condition = AGX_FCOND_LE; break;
                  default: break;
                  }
               }
               const agx_index selected[] = {
                  I->src[swap ? 3 : 2], I->src[swap ? 2 : 3]};
               const uint16_t sources[] = {
                  compare.value, selected[0].value, selected[1].value};
               const enum agx3_alu_hint hints[] = {
                  agx3_post_ra_hint(compare), agx3_post_ra_hint(selected[0]),
                  agx3_post_ra_hint(selected[1])};
               bool invalid = I->saturate;
               for (unsigned i = 0; i < 4; ++i)
                  invalid |= I->src[i].abs || I->src[i].neg || I->src[i].kill ||
                             (I->src[i].cache && I->src[i].discard);
               if (invalid)
                  AGX3_REFUSE(I);
               uint16_t fsel_wait = wait;
               if (fsel_wait != 0 && fsel_wait != 32) {
                  if (!agx3_retire_slots_noop(emission, enc, fsel_wait))
                     AGX3_REFUSE(I);
                  fsel_wait = 0;
               }
               uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
               if (!agx3_alu_fcmp_select_zero(enc, I->dest[0].value, sources,
                       compare.size == AGX_SIZE_16, hints, condition,
                       I->dest[0].cache, fsel_wait, bytes, &len) ||
                   !agx3_append(emission, bytes, len))
                  AGX3_REFUSE(I);
               continue;
            }
         }

         /*
          * A 32-bit float compared against a minifloat immediate, selecting
          * either of two 32-bit registers. This is what an ordinary
          * `x < k ? a : b` becomes: the comparison stays 32-bit and Mesa
          * carries the constant as a minifloat, so the packer used to see a
          * mixed-width select with nothing to widen and refuse the shader.
          * The immediate may be on either side; a leading immediate reverses
          * the comparison, which does not change what is selected.
          */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_FCMPSEL &&
             I->nr_dests == 1 && I->nr_srcs == 4 && !I->saturate &&
             agx3_fcond_witnessed_select(I->fcond) &&
             agx3_vary_reg32(I->dest[0]) && agx_channels(I->dest[0]) == 1 &&
             agx3_vary_reg32(I->src[2]) && agx3_vary_reg32(I->src[3]) &&
             agx_channels(I->src[2]) == 1 && agx_channels(I->src[3]) == 1 &&
             ((agx3_vary_reg32(I->src[0]) &&
               I->src[1].type == AGX_INDEX_IMMEDIATE) ||
              (agx3_vary_reg32(I->src[1]) &&
               I->src[0].type == AGX_INDEX_IMMEDIATE))) {
            const bool immediate_first =
               I->src[0].type == AGX_INDEX_IMMEDIATE;
            const agx_index immediate = I->src[immediate_first ? 0 : 1];
            const agx_index compare = I->src[immediate_first ? 1 : 0];
            const unsigned true_src = I->invert_cond ? 3 : 2;
            const unsigned false_src = I->invert_cond ? 2 : 3;

            bool invalid = immediate.value > 255 ||
                           agx_channels(compare) != 1;
            for (unsigned i = 0; i < 4; ++i)
               invalid |= I->src[i].abs || I->src[i].neg || I->src[i].kill ||
                          I->src[i].memory ||
                          (I->src[i].cache && I->src[i].discard);
            if (invalid)
               AGX3_REFUSE(I);

            const unsigned condition =
               immediate_first ? agx3_fcond_reverse(I->fcond) : I->fcond;
            const enum agx3_alu_hint hints[4] = {
               agx3_post_ra_hint(compare), AGX3_ALU_HINT_NONE,
               agx3_post_ra_hint(I->src[true_src]),
               agx3_post_ra_hint(I->src[false_src])};

            uint16_t fsel_wait = wait;
            if (fsel_wait & (fsel_wait - 1)) {
               if (!agx3_retire_slots_noop(emission, enc, fsel_wait))
                  AGX3_REFUSE(I);
               fsel_wait = 0;
            }

            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
            if (!agx3_alu_fcmp_select_f32_imm(
                   enc, I->dest[0].value, compare.value,
                   (uint8_t)immediate.value, I->src[true_src].value,
                   I->src[false_src].value, hints, condition,
                   I->dest[0].cache, fsel_wait, bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         /* Full/half comparisons select between full-register alternatives.
          * WBOIT uses EQ against half +inf; update_lightmap uses ordered LT
          * against a half threshold. The condition selector is orthogonal to
          * the native mixed-width routing and is byte-grounded for both. */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_FCMPSEL &&
             I->nr_dests == 1 && I->nr_srcs == 4 &&
             agx3_fcond_witnessed_select(I->fcond) && !I->saturate &&
             agx3_vary_reg32(I->dest[0]) && agx_channels(I->dest[0]) == 1 &&
             agx3_vary_reg32(I->src[2]) && agx3_vary_reg32(I->src[3]) &&
             agx_channels(I->src[2]) == 1 && agx_channels(I->src[3]) == 1 &&
             ((agx3_vary_reg32(I->src[0]) && agx3_half_reg(I->src[1])) ||
              (agx3_vary_reg32(I->src[1]) && agx3_half_reg(I->src[0])))) {
            /* The half may be on either side. Exchanging the compared
             * operands reverses the comparison and nothing else; an inverted
             * condition exchanges the two selected operands, which is how
             * Apple spells `!=`. */
            const bool half_first = agx3_half_reg(I->src[0]);
            const unsigned order[4] = {
               half_first ? 1u : 0u, half_first ? 0u : 1u,
               I->invert_cond ? 3u : 2u, I->invert_cond ? 2u : 3u};
            uint16_t src[4];
            enum agx3_alu_hint hints[4];
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
            bool invalid = agx_channels(I->src[0]) != 1 ||
                           agx_channels(I->src[1]) != 1;
            for (unsigned i = 0; i < 4; ++i) {
               invalid |= I->src[i].abs || I->src[i].neg || I->src[i].kill ||
                          (I->src[i].cache && I->src[i].discard);
               src[i] = I->src[order[i]].value;
               hints[i] = agx3_post_ra_hint(I->src[order[i]]);
            }
            const unsigned condition =
               half_first ? agx3_fcond_reverse(I->fcond) : I->fcond;
            uint16_t fsel_wait = wait;
            if (fsel_wait != 0 && fsel_wait != 32) {
               if (!agx3_retire_slots_noop(emission, enc, fsel_wait))
                  AGX3_REFUSE(I);
               fsel_wait = 0;
            }
            if (invalid ||
                !agx3_alu_fcmp_select_f32_f16(
                   enc, I->dest[0].value, src, hints, condition,
                   I->dest[0].cache, fsel_wait, bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         /* update_lightmap forms a half boolean mask by comparing f32 against
          * a minifloat constant.  Equality is commutative, so normalize the
          * immediate to Apple's second compare slot without changing the
          * selected values. */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_FCMPSEL &&
             I->nr_dests == 1 && I->nr_srcs == 4 && !I->invert_cond &&
             (I->fcond == AGX_FCOND_EQ || I->fcond == AGX_FCOND_LT ||
              I->fcond == AGX_FCOND_GT) && !I->saturate &&
             agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
             I->dest[0].size == AGX_SIZE_16 && agx_channels(I->dest[0]) == 1 &&
             ((I->src[0].type == AGX_INDEX_IMMEDIATE &&
               agx3_vary_reg32(I->src[1])) ||
              (agx3_vary_reg32(I->src[0]) &&
               I->src[1].type == AGX_INDEX_IMMEDIATE)) &&
             agx3_src_is_register(I->src[2]) && !I->src[2].memory &&
             I->src[2].size == AGX_SIZE_16 && agx_channels(I->src[2]) == 1 &&
             I->src[3].type == AGX_INDEX_IMMEDIATE && I->src[3].value <= 255) {
            const bool imm_first = I->src[0].type == AGX_INDEX_IMMEDIATE;
            const agx_index immediate = I->src[imm_first ? 0 : 1];
            const agx_index compare = I->src[imm_first ? 1 : 0];
            unsigned condition = I->fcond;
            if (imm_first) {
               if (condition == AGX_FCOND_LT)
                  condition = AGX_FCOND_GT;
               else if (condition == AGX_FCOND_GT)
                  condition = AGX_FCOND_LT;
            }
            bool invalid = immediate.value > 255;
            for (unsigned i = 0; i < 4; ++i)
               invalid |= I->src[i].abs || I->src[i].neg || I->src[i].kill ||
                          I->src[i].memory ||
                          (I->src[i].cache && I->src[i].discard);
            uint16_t fsel_wait = wait;
            if (fsel_wait != 0 && fsel_wait != 32) {
               if (!agx3_retire_slots_noop(emission, enc, fsel_wait))
                  AGX3_REFUSE(I);
               fsel_wait = 0;
            }
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
            if (invalid ||
                !agx3_alu_fcmp_minifloat_select_half(
                   enc, I->dest[0].value, compare.value, immediate.value,
                   I->src[2].value, I->src[3].value,
                   agx3_post_ra_hint(compare),
                   agx3_post_ra_hint(I->src[2]), condition,
                   I->dest[0].cache,
                   fsel_wait, bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_FCMPSEL &&
             I->nr_dests==1 && I->nr_srcs==4 &&
             agx3_src_is_register(I->dest[0]) && !I->dest[0].memory && agx_channels(I->dest[0])==1 &&
             (I->dest[0].size==AGX_SIZE_16 || I->dest[0].size==AGX_SIZE_32) &&
             I->src[2].type==AGX_INDEX_IMMEDIATE && I->src[3].type==AGX_INDEX_IMMEDIATE &&
             I->src[2].value<=255 && I->src[3].value<=255 &&
             ((I->src[0].type==AGX_INDEX_IMMEDIATE && I->src[0].value==0 && agx3_vary_reg32(I->src[1])) ||
              (I->src[1].type==AGX_INDEX_IMMEDIATE && I->src[1].value==0 && agx3_vary_reg32(I->src[0])))) {
            const bool zero_first=I->src[0].type==AGX_INDEX_IMMEDIATE;
            const agx_index compare=I->src[zero_first?1:0];
            unsigned condition=I->fcond;
            bool swap=I->invert_cond;
            if(condition==AGX_FCOND_LTN) {condition=AGX_FCOND_GE;swap=!swap;}
            else if(condition==AGX_FCOND_GTN) {condition=AGX_FCOND_LE;swap=!swap;}
            if(zero_first) {
               switch(condition) {
               case AGX_FCOND_LT:condition=AGX_FCOND_GT;break;
               case AGX_FCOND_GT:condition=AGX_FCOND_LT;break;
               case AGX_FCOND_LE:condition=AGX_FCOND_GE;break;
               case AGX_FCOND_GE:condition=AGX_FCOND_LE;break;
               default:break;
               }
            }
            const uint16_t src[]={compare.value,0,I->src[swap?3:2].value,I->src[swap?2:3].value};
            const bool compare_abs[]={false,false};
            const bool immediate[]={true,true};
            const enum agx3_alu_hint hints[]={agx3_post_ra_hint(compare),AGX3_ALU_HINT_NONE,
               AGX3_ALU_HINT_NONE,AGX3_ALU_HINT_NONE};
            bool invalid=I->saturate || agx_channels(compare)!=1;
            for(unsigned i=0;i<4;i++)invalid|=I->src[i].abs || I->src[i].neg || I->src[i].kill ||
               (I->src[i].cache && I->src[i].discard);
            if(invalid)AGX3_REFUSE(I);
            uint16_t fsel_wait=wait;
            if(fsel_wait!=0 && fsel_wait!=32) {
               if(!agx3_retire_slots_noop(emission,enc,fsel_wait))AGX3_REFUSE(I);
               fsel_wait=0;
            }
            uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
            if(!agx3_alu_fcmp_select_reg_imm(enc,I->dest[0].value,src,true,
                 compare_abs,immediate,hints,
                 condition,I->dest[0].size==AGX_SIZE_16,I->dest[0].cache,fsel_wait,bytes,&len) ||
               !agx3_append(emission,bytes,len))AGX3_REFUSE(I);
            continue;
         }

         /* Native FSEL selects retained 16/32-bit register values or a register
          * and an immediate directly. Preserve each operand's lifetime hints
          * without materializing a separate comparison result. */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_FCMPSEL &&
             I->nr_dests==1 && I->nr_srcs==4 &&
             agx3_src_is_register(I->dest[0]) && !I->dest[0].memory && agx_channels(I->dest[0])==1 &&
             (I->dest[0].size==AGX_SIZE_16 || I->dest[0].size==AGX_SIZE_32) &&
             ((agx3_mod_src_ok(I->src[0]) && agx3_mod_src_ok(I->src[1])) ||
              (agx3_vary_reg32(I->src[0]) && agx3_src_is_zero_imm(I->src[1])) ||
              (agx3_src_is_zero_imm(I->src[0]) && agx3_vary_reg32(I->src[1]))) &&
             (agx3_src_is_register(I->src[2]) ||
              agx3_src_is_register(I->src[3]))) {
            const bool zero_first=agx3_src_is_zero_imm(I->src[0]);
            const bool compare_zero=zero_first || agx3_src_is_zero_imm(I->src[1]);
            unsigned condition=I->fcond;
            bool swap=I->invert_cond;
            if(condition==AGX_FCOND_LTN) {condition=AGX_FCOND_GE;swap=!swap;}
            else if(condition==AGX_FCOND_GTN) {condition=AGX_FCOND_LE;swap=!swap;}
            if(zero_first) {
               switch(condition) {
               case AGX_FCOND_LT:condition=AGX_FCOND_GT;break;
               case AGX_FCOND_GT:condition=AGX_FCOND_LT;break;
               case AGX_FCOND_LE:condition=AGX_FCOND_GE;break;
               case AGX_FCOND_GE:condition=AGX_FCOND_LE;break;
               default:break;
               }
            }
            const agx_index operands[]={I->src[zero_first?1:0],I->src[zero_first?0:1],
                                       I->src[swap?3:2],I->src[swap?2:3]};
            uint16_t src[4]; enum agx3_alu_hint hints[4]; bool immediate[2];
            bool compare_abs[2];
            bool invalid=I->saturate;
            for(unsigned i=0;i<4;i++) {
               const agx_index x=operands[i];
               const bool imm=x.type==AGX_INDEX_IMMEDIATE;
               invalid|=x.neg || (i>=2 && x.abs) || x.kill || x.memory ||
                  agx_channels(x)!=1 ||
                  (x.cache && x.discard) ||
                  (imm ? x.value>255 : (!agx3_src_is_register(x) ||
                    x.size!=(i<2?AGX_SIZE_32:I->dest[0].size)));
               src[i]=x.value; hints[i]=imm?AGX3_ALU_HINT_NONE:agx3_post_ra_hint(x);
               if(i<2)compare_abs[i]=x.abs;
               if(i>=2)immediate[i-2]=imm;
            }
            if(invalid)AGX3_REFUSE(I);
            uint16_t fsel_wait=wait;
            if(fsel_wait!=0 && fsel_wait!=32) {
               if(!agx3_retire_slots_noop(emission,enc,fsel_wait))AGX3_REFUSE(I);
               fsel_wait=0;
            }
            uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
            if(!agx3_alu_fcmp_select_reg_imm(enc,I->dest[0].value,src,compare_zero,
                 compare_abs,immediate,hints,condition,I->dest[0].size==AGX_SIZE_16,
                 I->dest[0].cache,fsel_wait,bytes,&len) ||
               !agx3_append(emission,bytes,len))AGX3_REFUSE(I);
            continue;
         }

         /* Float comparison selecting integer byte constants, preserving the
          * native 16/32-bit destination width assigned by register allocation. */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_FCMPSEL &&
             I->nr_dests == 1 && I->nr_srcs == 4 &&
             I->dest[0].type == AGX_INDEX_REGISTER && !I->dest[0].memory &&
             agx_channels(I->dest[0]) == 1 &&
             (I->dest[0].size == AGX_SIZE_16 || I->dest[0].size == AGX_SIZE_32) &&
             agx3_src_is_register(I->src[0]) && agx3_src_is_register(I->src[1]) &&
             agx_channels(I->src[0]) == 1 && agx_channels(I->src[1]) == 1 &&
             (I->src[0].size == AGX_SIZE_16 || I->src[0].size == AGX_SIZE_32) &&
             (I->src[1].size == AGX_SIZE_16 || I->src[1].size == AGX_SIZE_32) &&
             !I->src[0].abs && !I->src[1].abs &&
             I->src[2].type == AGX_INDEX_IMMEDIATE && I->src[2].value <= 255 &&
             I->src[3].type == AGX_INDEX_IMMEDIATE && I->src[3].value <= 255) {
            /* Apple spells `-a < b` as the equivalent `b > -a`, placing the
             * only witnessed negate bit on the second full-width comparator.
             * Do not generalize that modifier to another slot or condition. */
            const bool neg_second = I->src[0].neg && !I->src[1].neg &&
                                    I->fcond == AGX_FCOND_LT &&
                                    !I->invert_cond &&
                                    I->dest[0].size == AGX_SIZE_16 &&
                                    I->src[0].size == AGX_SIZE_32 &&
                                    I->src[1].size == AGX_SIZE_32 &&
                                    I->src[2].value == 255 &&
                                    I->src[3].value == 0;
            const unsigned order[2] = {neg_second ? 1u : 0u,
                                       neg_second ? 0u : 1u};
            const uint16_t src[] = {I->src[order[0]].value,
                                    I->src[order[1]].value};
            const enum agx3_alu_hint hints[] = {
               agx3_post_ra_hint(I->src[order[0]]),
               agx3_post_ra_hint(I->src[order[1]])};
            const bool src_16[] = {
               I->src[order[0]].size == AGX_SIZE_16,
               I->src[order[1]].size == AGX_SIZE_16};
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
            unsigned condition = I->fcond;
            bool swap = I->invert_cond;
            if (condition == AGX_FCOND_LTN) {
               condition = AGX_FCOND_GE;
               swap = !swap;
            } else if (condition == AGX_FCOND_GTN) {
               condition = AGX_FCOND_LE;
               swap = !swap;
            }
            if (neg_second)
               condition = agx3_fcond_reverse(condition);
            uint16_t fsel_wait = wait;
            if (fsel_wait != 0 && fsel_wait != 32) {
               if (!agx3_retire_slots_noop(emission, enc, fsel_wait))
                  AGX3_REFUSE(I);
               fsel_wait = 0;
            }
            if (I->saturate || I->src[0].kill || I->src[1].kill ||
                (I->src[0].cache && I->src[0].discard) ||
                (I->src[1].cache && I->src[1].discard) ||
                ((!neg_second) && (I->src[0].neg || I->src[1].neg)) ||
                I->src[2].abs || I->src[2].neg ||
                I->src[3].abs || I->src[3].neg ||
                !(neg_second
                     ? agx3_alu_fcmp_select_imm_width_neg_second(
                          enc, I->dest[0].value, src, src_16, hints,
                          condition, I->src[swap ? 3 : 2].value,
                          I->src[swap ? 2 : 3].value,
                          I->dest[0].size == AGX_SIZE_16,
                          I->dest[0].cache, fsel_wait, bytes, &len)
                     : agx3_alu_fcmp_select_imm_width(
                          enc, I->dest[0].value, src, src_16, hints,
                          condition, I->src[swap ? 3 : 2].value,
                          I->src[swap ? 2 : 3].value,
                          I->dest[0].size == AGX_SIZE_16,
                          I->dest[0].cache, fsel_wait, bytes, &len)) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_IADD &&
             I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
             I->shift == 4 && I->dest[0].size == AGX_SIZE_16 &&
             I->src[0].size == AGX_SIZE_16 &&
             I->src[1].size == AGX_SIZE_16 &&
             agx_channels(I->dest[0]) == 1 &&
             agx_channels(I->src[0]) == 1 && agx_channels(I->src[1]) == 1 &&
             agx3_src_is_register(I->dest[0]) &&
             I->src[0].type == AGX_INDEX_IMMEDIATE &&
             I->src[0].value <= 15 && !I->src[0].memory &&
             !I->src[0].abs && !I->src[0].neg &&
             agx3_src_is_register(I->src[1]) && !I->src[1].memory &&
             !I->src[1].abs && !I->src[1].neg && !I->src[1].kill &&
             !(I->src[1].cache && I->src[1].discard)) {
            const struct agx3_shift_add_desc d = {
               .dest = I->dest[0].value,
               .dest_hint = I->dest[0].cache,
               .src0_immediate = true,
               .src0 = I->src[0].value,
               .src1 = I->src[1].value,
               .shift = 4,
               .wait_mask = wait,
            };
            uint8_t bytes[AGX3_WRITEOUT_MAX_LENGTH], len;
            if (!agx3_shift_add16_encode(enc, &d, bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_IADD &&
             I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
             I->dest[0].size == AGX_SIZE_16 && I->src[0].size == AGX_SIZE_16 &&
             I->src[1].size == AGX_SIZE_16 && agx_channels(I->dest[0]) == 1 &&
             agx_channels(I->src[0]) == 1 && agx_channels(I->src[1]) == 1 &&
             agx3_src_is_register(I->dest[0]) && agx3_src_is_register(I->src[0]) &&
             agx3_src_is_register(I->src[1]) &&
             !I->src[0].abs && !I->src[0].neg && !I->src[1].abs && !I->src[1].neg) {
            const uint16_t src[2] = {I->src[0].value, I->src[1].value};
            const enum agx3_alu_hint hints[2] = {
               agx3_post_ra_hint(I->src[0]), agx3_post_ra_hint(I->src[1])};
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
            if (I->src[0].kill || I->src[1].kill ||
                !agx3_alu_shift_add16(enc, I->dest[0].value, src, hints, I->shift,
                                      I->dest[0].cache, wait, bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         /* Unshifted register arithmetic shares the original canonical
          * form, but its native source hints come from post-RA lifetime. */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_IADD &&
             I->shift == 0 && I->nr_dests == 1 && I->nr_srcs == 2 &&
             !I->saturate && agx3_mod_src_ok(I->dest[0]) &&
             agx3_mod_src_ok(I->src[0]) && agx3_mod_src_ok(I->src[1]) &&
             !I->src[0].abs && !I->src[0].neg && !I->src[1].abs) {
            const enum agx3_alu_op op = I->src[1].neg ? AGX3_ALU_ISUB
                                                    : AGX3_ALU_IADD;
            const uint16_t src[] = {I->src[0].value, I->src[1].value};
            const enum agx3_alu_hint hints[] = {
               agx3_post_ra_hint(I->src[0]), agx3_post_ra_hint(I->src[1])};
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
            if (I->src[0].kill || I->src[1].kill ||
                (I->src[0].cache && I->src[0].discard) ||
                (I->src[1].cache && I->src[1].discard) ||
                !agx3_alu_integer_reg(enc, op, I->dest[0].value, src,
                                      hints, wait, bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         /* A local-memory byte offset widens the 16-bit local invocation
          * index with `iadd dst32, #0, src16.abs`. The native integer iadd
          * immediate form does not admit zero in its witnessed endpoint, but
          * zero-extension needs no new arithmetic encoding: copy the low half
          * and clear the high half, exactly like the existing widened BITOP
          * move below. The copy carries the source's real wait/lifetime. */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_IADD &&
             I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
             I->shift == 0 && agx3_src_is_register(I->dest[0]) &&
             !I->dest[0].memory && I->dest[0].size == AGX_SIZE_32 &&
             agx_channels(I->dest[0]) == 1 && !(I->dest[0].value & 1)) {
            const bool imm_first = I->src[0].type == AGX_INDEX_IMMEDIATE;
            const agx_index zero = I->src[imm_first ? 0 : 1];
            const agx_index half = I->src[imm_first ? 1 : 0];
            /* Post-RA discard is the physical lifetime bit consumed by COPY;
             * SSA kill is allocator bookkeeping and must not reject the same
             * physical source shape. */
            if (zero.type == AGX_INDEX_IMMEDIATE && zero.value == 0 &&
                !zero.memory && !zero.abs && !zero.neg &&
                agx3_src_is_register(half) && !half.memory && half.abs &&
                !half.neg && half.size == AGX_SIZE_16 &&
                agx_channels(half) == 1 && !(half.cache && half.discard)) {
               uint8_t lo[AGX3_COPY_MAX_LENGTH], hi[AGX3_STAGE_MAX_LENGTH];
               uint8_t lo_len = 0, hi_len = 0;
               if (!agx3_copy_encode_reg16(enc, I->dest[0].value, half.value,
                                           I->dest[0].cache, half.cache,
                                           half.discard, wait, lo, &lo_len) ||
                   !agx3_clr(enc, I->dest[0].value + 1, true, false, hi,
                             &hi_len) ||
                   !agx3_append(emission, lo, lo_len) ||
                   !agx3_append(emission, hi, hi_len))
                  AGX3_REFUSE(I);
               continue;
            }
         }

         /*
          * `iadd dst32, #imm, src << shift` -- constant address arithmetic,
          * which is what every shader with more than one memory access is
          * built out of. The source may be a 16-bit half, zero- or
          * sign-extended (abs and neg respectively, as on AGX2).
          *
          * This carries its own scoreboard wait, so it does not need the
          * wait-retiring prelude and does not disturb the operands.
          */
         if (I->op == AGX_OPCODE_IADD && I->nr_dests == 1 && I->nr_srcs == 2 &&
             agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
             I->dest[0].size == AGX_SIZE_32 &&
             agx_channels(I->dest[0]) == 1) {
            const bool imm_first = I->src[0].type == AGX_INDEX_IMMEDIATE;
            const agx_index constant = imm_first ? I->src[0] : I->src[1];
            const agx_index reg = imm_first ? I->src[1] : I->src[0];

            if (!imm_first && I->shift == 0 && !I->saturate &&
                constant.type == AGX_INDEX_IMMEDIATE && constant.neg &&
                !constant.abs && !constant.memory && constant.value <= 0xff &&
                agx3_src_is_register(reg) && !reg.memory && !reg.kill &&
                agx_channels(reg) == 1 && !(reg.cache && reg.discard) &&
                ((reg.size == AGX_SIZE_32 && !reg.abs && !reg.neg) ||
                 (reg.size == AGX_SIZE_16 && (reg.abs != reg.neg)))) {
               const struct agx3_isub_imm_desc d = {
                  .dest=I->dest[0].value, .src=reg.value,
                  .immediate=constant.value, .src_16=reg.size==AGX_SIZE_16,
                  .sign_extend=reg.neg, .src_cache=reg.cache,
                  .src_kill=reg.discard, .wait_mask=wait,
               };
               uint8_t bytes[AGX3_IADD_MAX_LENGTH],len;
               if (agx3_isub_immediate_encode(enc,&d,bytes,&len) &&
                   agx3_append(emission,bytes,len))
                  continue;
            }

            /* Hyprland lowers its signed 16-bit gradient counter to
             * `iadd dst32, #0, src16 << 4`. A bare narrow integer is signed
             * on AGX (ABS denotes unsigned); fresh G17P Metal witnesses the
             * exact zero-immediate, sign-extended shift-4 form, including a
             * cached destination. Keep other bare-half shifts closed. */
            const bool signed_half_shift4 =
               enc == AGX_ISA_ENCODING_AGX3B && !I->saturate &&
               constant.type == AGX_INDEX_IMMEDIATE && constant.value == 0 &&
               reg.size == AGX_SIZE_16 && !reg.abs && !reg.neg &&
               I->shift == 4;
            if (constant.type == AGX_INDEX_IMMEDIATE && !constant.abs &&
                !constant.neg && constant.value <= 0xff &&
                agx3_src_is_register(reg) && !reg.memory &&
                agx_channels(reg) == 1 && !(reg.abs && reg.neg) &&
                (reg.size == AGX_SIZE_32 ||
                 (reg.size == AGX_SIZE_16 &&
                  (reg.abs || reg.neg || signed_half_shift4)))) {
               const struct agx3_iadd_desc d = {
                  .dest = I->dest[0].value,
                  .dest_cache = signed_half_shift4 && I->dest[0].cache,
                  .immediate = (uint8_t)constant.value,
                  .src = reg.value,
                  .src_kill = reg.discard,
                  .src_16 = reg.size == AGX_SIZE_16,
                  .sign_extend = signed_half_shift4 || reg.neg,
                  .shift = I->shift,
                  .wait_mask = wait,
               };
               uint8_t bytes[AGX3_IADD_MAX_LENGTH];
               uint8_t len = 0;
               if (agx3_iadd_encode(enc, &d, bytes, &len) &&
                   agx3_append(emission, bytes, len))
                  continue;
               /* Outside the witnessed domain; fall through to refuse. */
            }

            /*
             * Register + (register << shift). This is the second half of the
             * shifted-imad split in agx_emit_imadshl_agx(): AGX3 has no
             * shifted imad, so the product is added here with the shift on
             * the addend. Apple emits exactly this, and f54 carries the
             * shift (witnessed at 1..4 in test-agx3-iadd-core.c).
             */
            const agx_index a = I->src[0], bsrc = I->src[1];

            /*
             * Only take this path for a SHIFTED add. An unshifted register
             * add already had a working encoding through the select core, and
             * re-routing it here changed its bytes -- which cost two compute
             * probes on hardware. New encodings belong on shapes that were
             * previously refused, not on shapes that already worked.
             */
            if (I->shift != 0 &&
                agx3_src_is_register(a) && !a.memory && !a.abs && !a.neg &&
                a.size == AGX_SIZE_32 && agx_channels(a) == 1 &&
                agx3_src_is_register(bsrc) && !bsrc.memory && !bsrc.abs &&
                !bsrc.neg && bsrc.size == AGX_SIZE_32 &&
                agx_channels(bsrc) == 1) {
               const struct agx3_iadd_desc d = {
                  .dest = I->dest[0].value,
                  .src_a_is_register = true,
                  .src_a_reg = a.value,
                  .src_a_kill = a.discard,
                  .src = bsrc.value,
                  .src_kill = bsrc.discard,
                  .shift = I->shift,
                  .wait_mask = wait,
               };
               uint8_t bytes[AGX3_IADD_MAX_LENGTH];
               uint8_t len = 0;
               if (agx3_iadd_encode(enc, &d, bytes, &len) &&
                   agx3_append(emission, bytes, len))
                  continue;
            }

            /*
             * Register + register where at least one operand is a 16-bit
             * half, zero-extended (abs) or sign-extended (neg) as on AGX2.
             * `gl_LocalInvocationID` arrives as three 16-bit special
             * registers, so summing it is exactly this shape; the whole
             * shader was refused before, because the register slot-A form
             * pinned both operand widths to 32-bit.
             *
             * Apple emits the same instruction for `(uint)a + (uint)b` over
             * two ushorts; see the witnesses quoted in agx3_iadd_core.h.
             *
             * Gated on a narrow operand so the all-32-bit unshifted add keeps
             * the select-core encoding it already had: re-routing a shape
             * that already worked is what cost two hardware probes last time.
             */
            const bool a_ok =
               agx3_src_is_register(a) && !a.memory && agx_channels(a) == 1 &&
               !(a.cache && a.discard) &&
               (a.size == AGX_SIZE_32 ? (!a.abs && !a.neg)
                                      : (a.size == AGX_SIZE_16 &&
                                         (a.abs != a.neg)));
            const bool b_ok =
               agx3_src_is_register(bsrc) && !bsrc.memory &&
               agx_channels(bsrc) == 1 && !(bsrc.cache && bsrc.discard) &&
               (bsrc.size == AGX_SIZE_32 ? (!bsrc.abs && !bsrc.neg)
                                         : (bsrc.size == AGX_SIZE_16 &&
                                            (bsrc.abs != bsrc.neg)));

            if (I->shift == 0 && !I->saturate && a_ok && b_ok &&
                (a.size == AGX_SIZE_16 || bsrc.size == AGX_SIZE_16)) {
               const struct agx3_iadd_desc d = {
                  .dest = I->dest[0].value,
                  .dest_cache = I->dest[0].cache,
                  .src_a_is_register = true,
                  .src_a_reg = a.value,
                  .src_a_kill = a.discard,
                  .src_a_16 = a.size == AGX_SIZE_16,
                  .src_a_sign_extend = a.size == AGX_SIZE_16 && a.neg,
                  .src = bsrc.value,
                  .src_kill = bsrc.discard,
                  .src_16 = bsrc.size == AGX_SIZE_16,
                  .sign_extend = bsrc.size == AGX_SIZE_16 && bsrc.neg,
                  .wait_mask = wait,
               };
               uint8_t bytes[AGX3_IADD_MAX_LENGTH];
               uint8_t len = 0;
               if (agx3_iadd_encode(enc, &d, bytes, &len) &&
                   agx3_append(emission, bytes, len))
                  continue;
            }
         }

         /*
          * Widening move: a 32-bit destination fed by a 16-bit half. Apple
          * does not have a widening `copy`; its compiler moves the half and
          * clears the upper one, which is what a zero-extend is. (Its ALU can
          * also just consume a 16-bit operand in a 32-bit op -- the width is
          * per-operand -- but that folds into the consumer and there is no
          * consumer here.) Both halves of this pair are witnessed encodings.
          */
         if (I->op == AGX_OPCODE_BITOP &&
             (I->truth_table & 0xfu) == AGX3_SEL_TT_MOV &&
             I->nr_dests == 1 && I->nr_srcs == 2 &&
             agx3_src_is_register(I->src[0]) && !I->src[0].memory &&
             !I->src[0].abs && !I->src[0].neg &&
             I->src[0].size == AGX_SIZE_16 && agx_channels(I->src[0]) == 1 &&
             I->src[1].type == AGX_INDEX_IMMEDIATE && I->src[1].value == 0 &&
             agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
             I->dest[0].size == AGX_SIZE_32 && agx_channels(I->dest[0]) == 1 &&
             (I->dest[0].value & 1) == 0) {
            uint8_t lo[AGX3_COPY_MAX_LENGTH], hi[AGX3_STAGE_MAX_LENGTH];
            uint8_t lo_len = 0, hi_len = 0;
            if (!agx3_copy_encode_reg16(enc, I->dest[0].value,
                                        I->src[0].value, I->dest[0].cache,
                                        I->src[0].cache, I->src[0].discard,
                                        wait, lo, &lo_len) ||
                !agx3_clr(enc, I->dest[0].value + 1, true, false, hi,
                          &hi_len) ||
                !agx3_append(emission, lo, lo_len) ||
                !agx3_append(emission, hi, hi_len))
               AGX3_REFUSE(I);
            continue;
         }

         /* A narrowing integer move copies the low half of its 32-bit
          * source. The same native 16-bit COPY used above names that low half
          * directly, so no arithmetic conversion or new encoding is needed. */
         if (I->op == AGX_OPCODE_BITOP &&
             (I->truth_table & 0xfu) == AGX3_SEL_TT_MOV &&
             I->nr_dests == 1 && I->nr_srcs == 2 &&
             agx3_src_is_register(I->src[0]) && !I->src[0].memory &&
             !I->src[0].abs && !I->src[0].neg &&
             I->src[0].size == AGX_SIZE_32 && agx_channels(I->src[0]) == 1 &&
             I->src[1].type == AGX_INDEX_IMMEDIATE && I->src[1].value == 0 &&
             agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
             I->dest[0].size == AGX_SIZE_16 && agx_channels(I->dest[0]) == 1) {
            uint8_t bytes[AGX3_COPY_MAX_LENGTH];
            uint8_t len = 0;
            if (!agx3_copy_encode_reg16(enc, I->dest[0].value,
                                        I->src[0].value, I->dest[0].cache,
                                        I->src[0].cache, I->src[0].discard,
                                        wait, bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         /*
          * The same lowering at 16 bits, register to register: Mesa keeps
          * booleans in halves and the select core's 32-bit `iadd d, s, #0`
          * move cannot express them. Apple copies a half with `copy`.
          */
         if (I->op == AGX_OPCODE_BITOP &&
             (I->truth_table & 0xfu) == AGX3_SEL_TT_MOV &&
             I->nr_dests == 1 && I->nr_srcs == 2 &&
             agx3_src_is_register(I->src[0]) && !I->src[0].memory &&
             !I->src[0].abs && !I->src[0].neg &&
             I->src[0].size == AGX_SIZE_16 && agx_channels(I->src[0]) == 1 &&
             I->src[1].type == AGX_INDEX_IMMEDIATE && I->src[1].value == 0 &&
             agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
             I->dest[0].size == AGX_SIZE_16 && agx_channels(I->dest[0]) == 1) {
            uint8_t bytes[AGX3_COPY_MAX_LENGTH];
            uint8_t len = 0;
            if (!agx3_copy_encode_reg16(enc, I->dest[0].value,
                                        I->src[0].value, I->dest[0].cache,
                                        I->src[0].cache, I->src[0].discard,
                                        wait, bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         if (I->op == AGX_OPCODE_BITOP &&
             (I->truth_table & 0xfu) == AGX3_SEL_TT_MOV &&
             I->nr_dests == 1 && I->nr_srcs == 2 &&
             I->src[0].type == AGX_INDEX_UNIFORM && !I->src[0].memory &&
             !I->src[0].abs && !I->src[0].neg &&
             I->src[0].size == AGX_SIZE_32 && agx_channels(I->src[0]) == 1 &&
             I->src[1].type == AGX_INDEX_IMMEDIATE && I->src[1].value == 0 &&
             agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
             I->dest[0].size == AGX_SIZE_32 && agx_channels(I->dest[0]) == 1) {
            uint8_t bytes[AGX3_COPY_MAX_LENGTH];
            uint8_t len = 0;
            /* Apple's uniform->GPR copy witness has no wait field. Retire a
             * pending producer through the already-grounded band no-op, then
             * emit that exact copy instead of guessing scoreboard bits on a
             * form that has none in the corpus. Meta image copies reach this
             * when global IDs and pushed extents share a scoreboard slot. */
            const bool retired =
               wait == 0 || agx3_retire_slots_noop(emission, enc, wait);
            if (!retired ||
                !agx3_copy_encode_uniform_to_reg32(enc, I->dest[0].value,
                                                   I->src[0].value, bytes,
                                                   &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            outstanding &= (uint16_t)~wait;
            continue;
         }

         /* The same move at 16 bits. Mesa keeps booleans and halves in
          * 16-bit registers, so agx_lower_uniform_sources() emits this shape
          * for every narrow generic-ALU uniform source; without an encoder
          * it stayed a bitop reading the uniform file and the packer refused
          * the whole shader. Apple's own corpus carries the form, and it
          * likewise has no wait field, so retire through the grounded band
          * no-op first. */
         if (I->op == AGX_OPCODE_BITOP &&
             (I->truth_table & 0xfu) == AGX3_SEL_TT_MOV &&
             I->nr_dests == 1 && I->nr_srcs == 2 &&
             I->src[0].type == AGX_INDEX_UNIFORM && !I->src[0].memory &&
             !I->src[0].abs && !I->src[0].neg &&
             I->src[0].size == AGX_SIZE_16 && agx_channels(I->src[0]) == 1 &&
             I->src[1].type == AGX_INDEX_IMMEDIATE && I->src[1].value == 0 &&
             agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
             I->dest[0].size == AGX_SIZE_16 &&
             agx_channels(I->dest[0]) == 1) {
            uint8_t bytes[AGX3_COPY_MAX_LENGTH];
            uint8_t len = 0;
            const bool retired =
               wait == 0 || agx3_retire_slots_noop(emission, enc, wait);
            if (!retired ||
                !agx3_copy_encode_uniform_to_reg16(enc, I->dest[0].value,
                                                   I->src[0].value,
                                                   I->dest[0].cache, bytes,
                                                   &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            outstanding &= (uint16_t)~wait;
            continue;
         }

         /*
          * Fused integer compare-select. agx_lower_pseudo() turns every
          * AGX_OPCODE_ICMP into one of these, so without it no shader with an
          * integer comparison compiles at all. Apple's `isel` is the same
          * instruction with the same operand order and the same condition
          * numbering (probe-witnessed; see agx3_isel_core.h).
          */
         if (I->op == AGX_OPCODE_ICMPSEL && I->nr_dests == 1 &&
             I->nr_srcs == 4) {
            struct agx3_isel_desc d;
            memset(&d, 0, sizeof(d));

            const agx_index dest = I->dest[0];
            agx_index a = I->src[0], b = I->src[1];
            const bool swap_compare = a.type == AGX_INDEX_IMMEDIATE;
            if (swap_compare) {
               const agx_index tmp = a;
               a = b;
               b = tmp;
            }
            const agx_index t = I->src[2], f = I->src[3];

            const bool dest_ok =
               agx3_src_is_register(dest) && !dest.memory &&
               agx_channels(dest) == 1 &&
               (dest.size == AGX_SIZE_32 || dest.size == AGX_SIZE_16);
            const bool cmp_ok =
               agx3_src_is_register(a) && !a.memory && !a.abs && !a.neg &&
               (a.size == AGX_SIZE_32 || a.size == AGX_SIZE_16) &&
               agx_channels(a) == 1 && !b.memory && !b.abs && !b.neg &&
               ((b.type == AGX_INDEX_IMMEDIATE && b.value <= 255) ||
                (agx3_src_is_register(b) && b.size == a.size &&
                 agx_channels(b) == 1));

            if (!dest_ok || !cmp_ok)
               AGX3_REFUSE(I);

            d.dest = dest.value;
            d.dest_16 = dest.size == AGX_SIZE_16;
            d.cmp0 = a.value;
            d.cmp1 = b.value;
            d.cmp1_immediate = b.type == AGX_INDEX_IMMEDIATE;
            d.cmp_16 = a.size == AGX_SIZE_16;
            d.explicit_hints = true;
            const agx_index hinted[4] = {a, b, t, f};
            for (unsigned n = 0; n < 4; ++n) {
               if (hinted[n].type == AGX_INDEX_REGISTER) {
                  d.cache[n] = hinted[n].cache;
                  d.discard[n] = hinted[n].discard;
               }
            }

            /* AGX3 has one equality condition, 4. Mesa spells the same
             * comparison two ways (UEQ = 0 and SEQ = 4) because AGX2 did;
             * the probes compiled `uint ==` and `int ==` and Apple emitted 4
             * for both, equality being sign-agnostic. Every other condition
             * keeps its number. */
            switch (I->icond) {
            case AGX_ICOND_UEQ:
            case AGX_ICOND_SEQ:
               d.cond = AGX3_ISEL_ICOND_EQ;
               break;
            case AGX_ICOND_ULT:
               d.cond = AGX3_ISEL_ICOND_ULT;
               break;
            case AGX_ICOND_UGT:
               d.cond = AGX3_ISEL_ICOND_UGT;
               break;
            case AGX_ICOND_SLT:
               d.cond = AGX3_ISEL_ICOND_SLT;
               break;
            case AGX_ICOND_SGT:
               d.cond = AGX3_ISEL_ICOND_SGT;
               break;
            default:
               AGX3_REFUSE(I);
            }

            /* Canonicalize an immediate LHS by reversing the comparison,
             * retaining the selected operands and their actual lifetimes. */
            if (swap_compare) {
               switch (d.cond) {
               case AGX3_ISEL_ICOND_ULT: d.cond = AGX3_ISEL_ICOND_UGT; break;
               case AGX3_ISEL_ICOND_UGT: d.cond = AGX3_ISEL_ICOND_ULT; break;
               case AGX3_ISEL_ICOND_SLT: d.cond = AGX3_ISEL_ICOND_SGT; break;
               case AGX3_ISEL_ICOND_SGT: d.cond = AGX3_ISEL_ICOND_SLT; break;
               case AGX3_ISEL_ICOND_EQ: break;
               }
            }

            /* Each selected operand is independently a register or an
             * 8-bit immediate; the encoder knows all four combinations. */
            bool selects_ok = true;
            const agx_index sel[2] = {t, f};
            enum agx3_isel_operand kind[2];
            uint16_t value[2];

            for (unsigned n = 0; n < 2; ++n) {
               const agx_index s = sel[n];
               if (s.type == AGX_INDEX_IMMEDIATE && !s.memory && !s.abs &&
                   !s.neg) {
                  kind[n] = AGX3_ISEL_IMMEDIATE;
                  value[n] = s.value;
               } else if (agx3_src_is_register(s) && !s.memory && !s.abs &&
                          !s.neg && s.size == dest.size &&
                          agx_channels(s) == 1) {
                  kind[n] = AGX3_ISEL_REGISTER;
                  value[n] = s.value;
               } else {
                  selects_ok = false;
                  break;
               }
            }

            if (!selects_ok)
               AGX3_REFUSE(I);

            d.true_kind = kind[0];
            d.false_kind = kind[1];
            d.sel_true = value[0];
            d.sel_false = value[1];
            /* The compact isel wait selector encodes one slot or slots1+2.
             * Retire other producer combinations through the established
             * native no-op, keeping the select itself fused. */
            d.wait_mask = wait;
            if (wait != 3 && util_bitcount(wait) > 1) {
               d.wait_mask = 1u << (ffs(wait) - 1);
               if (!agx3_retire_slots_noop(emission, enc, wait & ~d.wait_mask))
                  AGX3_REFUSE(I);
            }

            uint8_t bytes[AGX3_ISEL_MAX_LENGTH];
            uint8_t len = 0;
            if (!agx3_isel_encode(enc, &d, bytes, &len) ||
                !agx3_append(emission, bytes, len)) {
               if (agx_compiler_debug & AGX_DBG_SHADERS)
                  fprintf(stderr,
                          "isel refused: dest=%u/%s cmp=%u,%u cond=%u "
                          "t=%s:%u f=%s:%u wait=%x\n",
                          d.dest, d.dest_16 ? "16" : "32", d.cmp0, d.cmp1,
                          d.cond, d.true_kind ? "imm" : "reg", d.sel_true,
                          d.false_kind ? "imm" : "reg", d.sel_false,
                          d.wait_mask);
               AGX3_REFUSE(I);
            }
            continue;
         }

         if (I->op == AGX_OPCODE_MOV_IMM && I->nr_dests == 1 &&
             I->nr_srcs == 0 && !I->saturate &&
             agx3_src_is_register(I->dest[0]) &&
             I->dest[0].size == AGX_SIZE_16 && agx_channels(I->dest[0]) == 1) {
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
            if (I->imm > 0xffff ||
                !agx3_alu_mov16(enc, I->dest[0].value, I->imm, wait, bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         /* Constant propagation can expose a float add whose two operands
          * are exact minifloats after scheduling. Fold it to the ordinary
          * native LDIMM instead of pretending either immediate is a GPR. */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_FADD &&
             I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate && !wait &&
             agx3_vary_reg32(I->dest[0]) && agx_channels(I->dest[0]) == 1 &&
             I->src[0].type == AGX_INDEX_IMMEDIATE &&
             I->src[1].type == AGX_INDEX_IMMEDIATE &&
             !I->src[0].memory && !I->src[1].memory &&
             !I->src[0].abs && !I->src[1].abs &&
             !I->src[0].neg && !I->src[1].neg) {
            const float value = agx_minifloat_decode(I->src[0].value) +
                                agx_minifloat_decode(I->src[1].value);
            uint32_t bits;
            memcpy(&bits, &value, sizeof(bits));
            struct agx3_encoded_ldimm encoded;
            if (!agx3_pack_ldimm_u32(enc, I->dest[0].value, bits, &encoded) ||
                !agx3_append(emission, encoded.bytes, encoded.length))
               AGX3_REFUSE(I);
            continue;
         }

         /* A constant f32-to-f16 conversion is represented by the same ADD
          * with -0 used for the register form. Fold its finite minifloat
          * inputs to the exact half literal; the saturate bit used by depth
          * output clamps before the single conversion, just like Apple's
          * dynamic narrowing instruction. */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_FADD &&
             I->nr_dests == 1 && I->nr_srcs == 2 && !wait &&
             agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
             I->dest[0].size == AGX_SIZE_16 &&
             agx_channels(I->dest[0]) == 1 &&
             I->src[0].type == AGX_INDEX_IMMEDIATE &&
             I->src[1].type == AGX_INDEX_IMMEDIATE &&
             !I->src[0].memory && !I->src[1].memory &&
             !I->src[0].abs && !I->src[1].abs &&
             !I->src[0].neg && !I->src[1].neg) {
            float value = agx_minifloat_decode(I->src[0].value) +
                          agx_minifloat_decode(I->src[1].value);
            if (!isfinite(value))
               AGX3_REFUSE(I);
            if (I->saturate)
               value = CLAMP(value, 0.0f, 1.0f);
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
            if (!agx3_alu_mov16(enc, I->dest[0].value,
                                _mesa_float_to_half(value), 0, bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         /* AGX models a narrowing float conversion as add with exact -0.
          * Keep one native operation, with actual source lifetime and waits. */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_FADD &&
             I->nr_dests == 1 && I->nr_srcs == 2 &&
             agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
             I->dest[0].size == AGX_SIZE_16 && agx_channels(I->dest[0]) == 1) {
            const bool swap = I->src[0].type == AGX_INDEX_IMMEDIATE;
            const agx_index src = I->src[swap ? 1 : 0];
            const agx_index zero = I->src[swap ? 0 : 1];
            if (agx3_src_is_register(src) && !src.memory && !src.abs && !src.neg &&
                (!src.kill || src.discard) && src.size == AGX_SIZE_32 &&
                agx_channels(src) == 1 &&
                zero.type == AGX_INDEX_IMMEDIATE && zero.value == 0x80 &&
                !zero.memory && !zero.abs && !zero.neg) {
               const struct agx3_falu_runtime runtime = {
                  .cache = {src.cache}, .discard = {src.discard},
                  .dest_cache = I->dest[0].cache, .wait_mask = wait,
               };
               uint8_t bytes[AGX3_FALU_MOD_MAX_LENGTH], len;
               if (!agx3_falu_narrow_f32(enc, I->dest[0].value, src.value,
                                         I->saturate, &runtime, bytes, &len) ||
                   !agx3_append(emission, bytes, len))
                  AGX3_REFUSE(I);
               continue;
            }
         }

         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_FMUL &&
             I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
             agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
             I->dest[0].size == AGX_SIZE_16 && agx_channels(I->dest[0]) == 1 &&
             agx3_vary_reg32(I->src[0]) && agx3_vary_reg32(I->src[1])) {
            uint16_t src[2]; enum agx3_alu_hint hints[2];
            for (unsigned i=0;i<2;i++) {
               agx_index x=I->src[i];
               if (agx_channels(x) != 1 || x.abs || x.neg || x.kill ||
                   (x.cache && x.discard))
                  AGX3_REFUSE(I);
               src[i]=x.value; hints[i]=agx3_post_ra_hint(x);
            }
            uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
            if (!agx3_alu_fmul_narrow(enc,I->dest[0].value,src,hints,
                                      I->dest[0].cache,wait,bytes,&len) ||
                !agx3_append(emission,bytes,len))
               AGX3_REFUSE(I);
            continue;
         }

         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_FMUL &&
             I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
             agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
             I->dest[0].size == AGX_SIZE_16 &&
             agx_channels(I->dest[0]) == 1 &&
             ((I->src[0].size == AGX_SIZE_32 &&
               I->src[1].size == AGX_SIZE_16) ||
              (I->src[0].size == AGX_SIZE_16 &&
               I->src[1].size == AGX_SIZE_32))) {
            const bool half_first = I->src[0].size == AGX_SIZE_16;
            const agx_index f32 = I->src[half_first ? 1 : 0];
            const agx_index f16 = I->src[half_first ? 0 : 1];
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
            uint16_t encoded_wait = wait;
            /* The exact mixed-half endpoint carries only slot 0. If both the
             * texture sample and the uniform load are live, retire slot 1
             * with the established native wait NOP before issuing the fused
             * multiply; do not discard either dependency. */
            if (encoded_wait & ~UINT16_C(1)) {
               if (!agx3_retire_slots_noop(emission, enc,
                                            encoded_wait & ~UINT16_C(1)))
                  AGX3_REFUSE(I);
               encoded_wait &= 1;
            }
            if (!agx3_src_is_register(f32) || !agx3_src_is_register(f16) ||
                f32.memory || f16.memory || (f32.value & 1) ||
                agx_channels(f32) != 1 || agx_channels(f16) != 1 ||
                f32.abs || f32.neg || f32.kill || f16.abs || f16.neg ||
                f16.kill || (f32.cache && f32.discard) ||
                (f16.cache && f16.discard) ||
                !agx3_alu_fmul_mixed_half(
                   enc, I->dest[0].value, f32.value, f16.value,
                   agx3_post_ra_hint(f32), agx3_post_ra_hint(f16),
                   I->dest[0].cache, encoded_wait, bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_HMUL &&
             I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
             agx3_src_is_register(I->src[0]) &&
             I->src[1].type == AGX_INDEX_IMMEDIATE) {
            agx_index dst = I->dest[0], src = I->src[0], imm = I->src[1];
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
            if (!agx3_src_is_register(dst) || dst.memory ||
                dst.size != AGX_SIZE_16 || agx_channels(dst) != 1 ||
                src.memory || src.size != AGX_SIZE_16 ||
                agx_channels(src) != 1 || src.abs || src.neg || src.kill ||
                (src.cache && src.discard) || imm.memory || imm.abs ||
                imm.neg ||
                !agx3_alu_hmul_minifloat(
                   enc, dst.value, src.value, imm.value,
                   agx3_post_ra_hint(src), dst.cache, wait, bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_HMUL &&
             I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate) {
            uint16_t src[2]; enum agx3_alu_hint hints[2];
            agx_index dst=I->dest[0];
            if (!agx3_src_is_register(dst) || dst.memory || dst.size != AGX_SIZE_16 ||
                agx_channels(dst) != 1)
               AGX3_REFUSE(I);
            for (unsigned i=0;i<2;i++) {
               agx_index x=I->src[i];
               if (!agx3_src_is_register(x) || x.memory || x.size != AGX_SIZE_16 ||
                   agx_channels(x) != 1 || x.abs || x.neg || x.kill ||
                   (x.cache && x.discard))
                  AGX3_REFUSE(I);
               src[i]=x.value; hints[i]=agx3_post_ra_hint(x);
            }
            uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
            if (!agx3_alu_hmul(enc,dst.value,src,hints,dst.cache,wait,bytes,&len) ||
                !agx3_append(emission,bytes,len))
               AGX3_REFUSE(I);
            continue;
         }

         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_HADD &&
             I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate) {
            uint16_t src[2];
            enum agx3_alu_hint hints[2];
            agx_index dst = I->dest[0];
            if (!agx3_src_is_register(dst) || dst.memory ||
                dst.size != AGX_SIZE_16 || agx_channels(dst) != 1)
               AGX3_REFUSE(I);
            for (unsigned i = 0; i < 2; ++i) {
               agx_index x = I->src[i];
               if (!agx3_src_is_register(x) || x.memory ||
                   x.size != AGX_SIZE_16 || agx_channels(x) != 1 || x.abs ||
                   x.neg || x.kill || (x.cache && x.discard))
                  AGX3_REFUSE(I);
               src[i] = x.value;
               hints[i] = agx3_post_ra_hint(x);
            }
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
            if (!agx3_alu_hadd(enc, dst.value, src, hints, dst.cache, wait,
                               bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_HFMA &&
             I->nr_dests == 1 && I->nr_srcs == 3 && !I->saturate) {
            const bool immediate_addend =
               I->src[2].type == AGX_INDEX_IMMEDIATE;
            if (immediate_addend) {
               const agx_index dst = I->dest[0];
               const agx_index addend = I->src[2];
               uint16_t src[2];
               enum agx3_alu_hint hints[2];

               if (!agx3_src_is_register(dst) || dst.memory ||
                   dst.size != AGX_SIZE_16 || agx_channels(dst) != 1 ||
                   addend.memory || addend.abs || addend.neg || addend.kill ||
                   addend.cache || addend.discard ||
                   addend.size != AGX_SIZE_16 || agx_channels(addend) != 1 ||
                   addend.value > UINT8_MAX)
                  AGX3_REFUSE(I);

               for (unsigned i = 0; i < 2; ++i) {
                  const agx_index x = I->src[i];
                  if (!agx3_src_is_register(x) || x.memory ||
                      x.size != AGX_SIZE_16 || agx_channels(x) != 1 || x.abs ||
                      x.neg || x.kill || (x.cache && x.discard))
                     AGX3_REFUSE(I);
                  src[i] = x.value;
                  hints[i] = agx3_post_ra_hint(x);
               }

               uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
               if (!agx3_alu_hfma_addend_minifloat(
                      enc, dst.value, src, addend.value, hints, dst.cache,
                      wait, bytes, &len) ||
                   !agx3_append(emission, bytes, len))
                  AGX3_REFUSE(I);
               continue;
            }

            uint16_t src[3];
            enum agx3_alu_hint hints[3];
            agx_index dst = I->dest[0];
            if (!agx3_src_is_register(dst) || dst.memory ||
                dst.size != AGX_SIZE_16 || agx_channels(dst) != 1)
               AGX3_REFUSE(I);
            for (unsigned i = 0; i < 3; ++i) {
               agx_index x = I->src[i];
               if (!agx3_src_is_register(x) || x.memory ||
                   x.size != AGX_SIZE_16 || agx_channels(x) != 1 || x.abs ||
                   x.kill ||
                   (x.cache && x.discard))
                  AGX3_REFUSE(I);
               src[i] = x.value;
               hints[i] = agx3_post_ra_hint(x);
            }
            const bool product_neg = I->src[0].neg != I->src[1].neg;
            if (product_neg && I->src[0].neg) {
               const uint16_t swap_src = src[0];
               src[0] = src[1];
               src[1] = swap_src;
               const enum agx3_alu_hint swap_hint = hints[0];
               hints[0] = hints[1];
               hints[1] = swap_hint;
            }
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
            if (!agx3_alu_hfma(enc, dst.value, src, hints, product_neg,
                               I->src[2].neg, dst.cache, wait, bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         /* Native scalar f32 x f16 -> f32. Keep the half register distinct
          * from its neighbour, preserve post-RA hints and producer waits. */
         if (I->op == AGX_OPCODE_FMUL && I->nr_dests == 1 &&
             I->nr_srcs == 2 && !I->saturate &&
             agx3_src_is_register(I->dest[0]) &&
             I->dest[0].size == AGX_SIZE_32 && agx_channels(I->dest[0]) == 1 &&
             (I->src[0].size == AGX_SIZE_16 || I->src[1].size == AGX_SIZE_16) &&
             agx3_src_is_register(I->src[0]) && agx3_src_is_register(I->src[1])) {
            const bool half_first = I->src[0].size == AGX_SIZE_16;
            const agx_index f32 = I->src[half_first ? 1 : 0];
            const agx_index f16 = I->src[half_first ? 0 : 1];
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
            if (f32.size != AGX_SIZE_32 || f16.size != AGX_SIZE_16 ||
                agx_channels(f32) != 1 || agx_channels(f16) != 1 ||
                f32.abs || f32.neg || f16.abs || f16.neg ||
                f32.kill || f16.kill || (f32.cache && f32.discard) ||
                (f16.cache && f16.discard) ||
                !agx3_alu_fmul_mixed(enc, I->dest[0].value, f32.value,
                    f16.value, agx3_post_ra_hint(f32), agx3_post_ra_hint(f16),
                    wait, bytes, &len) || !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_IADD &&
             I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate && I->shift == 0 &&
             I->dest[0].type == AGX_INDEX_REGISTER && !I->dest[0].memory &&
             I->dest[0].size == AGX_SIZE_16 && agx_channels(I->dest[0]) == 1) {
            const bool imm_first = I->src[0].type == AGX_INDEX_IMMEDIATE;
            const agx_index imm = I->src[imm_first ? 0 : 1];
            const agx_index reg = I->src[imm_first ? 1 : 0];
            if (imm.type == AGX_INDEX_IMMEDIATE && imm.value <= 255 &&
                !imm.abs && !imm.memory &&
                reg.type == AGX_INDEX_REGISTER && !reg.memory &&
                reg.size == AGX_SIZE_16 && agx_channels(reg) == 1 &&
                !reg.abs && !reg.neg && !reg.kill && !(reg.cache && reg.discard)) {
               uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
               uint16_t immediate_wait = wait;
               if (imm.neg && immediate_wait != 0 && immediate_wait != 32) {
                  if (!agx3_retire_slots_noop(emission, enc, immediate_wait))
                     AGX3_REFUSE(I);
                  immediate_wait = 0;
               }
               const bool encoded = imm.neg
                  ? agx3_alu_sub_imm16(enc, I->dest[0].value, reg.value,
                                       imm.value, agx3_post_ra_hint(reg),
                                       I->dest[0].cache, immediate_wait, bytes,
                                       &len)
                  : agx3_alu_add_imm16(enc, I->dest[0].value, reg.value,
                                       imm.value, agx3_post_ra_hint(reg),
                                       I->dest[0].cache, wait, bytes, &len);
               if (!encoded ||
                   !agx3_append(emission,bytes,len))
                  AGX3_REFUSE(I);
               continue;
            }
         }

         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_CONVERT &&
             I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
             I->round == AGX_ROUND_RTZ &&
             I->src[0].type == AGX_INDEX_IMMEDIATE &&
             I->src[0].value == AGX_CONVERT_F_TO_S16 &&
             agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
             I->dest[0].size == AGX_SIZE_16 &&
             agx_channels(I->dest[0]) == 1 &&
             agx3_vary_reg32(I->src[1]) && agx_channels(I->src[1]) == 1 &&
             !I->src[1].abs && !I->src[1].neg && !I->src[1].kill &&
             !(I->src[1].cache && I->src[1].discard)) {
            uint16_t convert_wait = wait;
            if (convert_wait != 0 && convert_wait != 32) {
               if (!agx3_retire_slots_noop(emission, enc, convert_wait))
                  AGX3_REFUSE(I);
               convert_wait = 0;
            }
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
            if (!agx3_alu_f32_to_s16(
                   enc, I->dest[0].value, I->src[1].value,
                   agx3_post_ra_hint(I->src[1]), I->dest[0].cache,
                   convert_wait, bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         /* shaderFloat16 commonly converts an invocation/index value directly
          * from unsigned32 into a scalar half. This has a native result-width
          * form distinct from the established u32-to-f32 conversion. */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_CONVERT &&
             I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
             I->round == AGX_ROUND_RTE &&
             I->src[0].type == AGX_INDEX_IMMEDIATE &&
             I->src[0].value == AGX_CONVERT_U32_TO_F &&
             agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
             I->dest[0].size == AGX_SIZE_16 &&
             agx_channels(I->dest[0]) == 1 &&
             agx3_vary_reg32(I->src[1]) && agx_channels(I->src[1]) == 1 &&
             !I->src[1].abs && !I->src[1].neg && !I->src[1].kill &&
             !(I->src[1].cache && I->src[1].discard)) {
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
            if (!agx3_alu_u32_to_f16(
                   enc, I->dest[0].value, I->src[1].value,
                   agx3_post_ra_hint(I->src[1]), I->dest[0].cache, wait,
                   bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         /* Coordinates and loop counters use16-bit halves. Apple has a direct
          * signed/unsigned half int2flt form, distinct from the32-bit form.
          * Preserve liveness and async producer waits; do not reinterpret the
          * neighbouring half as part of a32-bit integer or widen in place.
          */
         if (I->op == AGX_OPCODE_CONVERT && I->nr_dests == 1 &&
             I->nr_srcs == 2 && !I->saturate &&
             I->round == AGX_ROUND_RTE &&
             I->src[0].type == AGX_INDEX_IMMEDIATE &&
             (I->src[0].value == AGX_CONVERT_U16_TO_F ||
              I->src[0].value == AGX_CONVERT_S16_TO_F) &&
             agx3_src_is_register(I->dest[0]) &&
             I->dest[0].size == AGX_SIZE_32 && agx_channels(I->dest[0]) == 1 &&
             agx3_src_is_register(I->src[1]) && !I->src[1].abs &&
             !I->src[1].neg && !I->src[1].kill &&
             !(I->src[1].cache && I->src[1].discard) &&
             I->src[1].size == AGX_SIZE_16 &&
             agx_channels(I->src[1]) == 1) {
            uint8_t bytes[AGX3_ALU_MAX_LENGTH];
            uint8_t len = 0;
            if (!(I->src[0].value == AGX_CONVERT_S16_TO_F ? agx3_alu_s16_to_f32 :
                   agx3_alu_u16_to_f32)(enc, I->dest[0].value, I->src[1].value,
                                     agx3_post_ra_hint(I->src[1]), wait, bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_FADD &&
             I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
             agx3_vary_reg32(I->dest[0]) && agx_channels(I->dest[0]) == 1) {
            const bool swap=I->src[0].type == AGX_INDEX_IMMEDIATE;
            const agx_index src=I->src[swap?1:0], imm=I->src[swap?0:1];
            if (agx3_src_is_register(src) && !src.memory && src.size == AGX_SIZE_16 &&
                agx_channels(src) == 1 && !src.kill &&
                !(src.abs && src.neg) &&
                imm.type == AGX_INDEX_IMMEDIATE && imm.value == 0x80 &&
                !imm.abs && !imm.neg && !imm.memory) {
               uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
               if (!agx3_alu_widen_half(enc,I->dest[0].value,src.value,
                                         agx3_post_ra_hint(src),src.abs,src.neg,
                                         I->dest[0].cache,
                                         wait,bytes,&len) ||
                   !agx3_append(emission,bytes,len))
                  AGX3_REFUSE(I);
               continue;
            }
         }

         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_FMUL &&
             I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
             agx3_vary_reg32(I->dest[0]) &&
             agx_channels(I->dest[0]) == 1 &&
             (I->src[0].type == AGX_INDEX_IMMEDIATE ||
              I->src[1].type == AGX_INDEX_IMMEDIATE)) {
            const bool immediate_first =
               I->src[0].type == AGX_INDEX_IMMEDIATE;
            const agx_index imm = I->src[immediate_first ? 0 : 1];
            const agx_index half = I->src[immediate_first ? 1 : 0];
            if (imm.value <= UINT8_MAX && !imm.memory && !imm.abs &&
                !imm.neg && agx3_half_reg(half) &&
                agx_channels(half) == 1 && !half.abs && !half.neg &&
                !half.kill && !(half.cache && half.discard)) {
               uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
               if (!agx3_alu_fmul_half_minifloat_to_f32(
                      enc, I->dest[0].value, half.value, imm.value,
                      agx3_post_ra_hint(half), wait, bytes, &len) ||
                   !agx3_append(emission, bytes, len))
                  AGX3_REFUSE(I);
               continue;
            }
         }

         /* Pixel-centre +0.5 and coordinate scaling use minifloat sources,
          * not GPRs. Native f32add/f32mul accept this pair directly, with the
          * register's last-use hint and the compact form's supported waits.
          * Commuting plain operands changes no operation or rounding.
          */
         if ((I->op == AGX_OPCODE_FADD || I->op == AGX_OPCODE_FMUL) &&
             I->nr_dests == 1 &&
             I->nr_srcs == 2 &&
             (I->op == AGX_OPCODE_FADD || !I->saturate) &&
             agx3_src_is_register(I->dest[0]) &&
             I->dest[0].size == AGX_SIZE_32 &&
             agx_channels(I->dest[0]) == 1 &&
             (I->src[0].type == AGX_INDEX_IMMEDIATE ||
              I->src[1].type == AGX_INDEX_IMMEDIATE)) {
            const bool immediate_first =
               I->src[0].type == AGX_INDEX_IMMEDIATE;
            const agx_index imm = I->src[immediate_first ? 0 : 1];
            const agx_index reg = I->src[immediate_first ? 1 : 0];
            uint8_t bytes[AGX3_ALU_MAX_LENGTH];
            uint8_t len = 0;
            const bool multiply = I->op == AGX_OPCODE_FMUL;
            if (!multiply && !I->saturate && reg.abs && !reg.neg && !reg.kill &&
                imm.value == 0x80 && !imm.memory && !imm.abs && !imm.neg &&
                agx3_src_is_register(reg) && !reg.memory &&
                !(reg.cache && reg.discard) && reg.size == AGX_SIZE_32 &&
                agx_channels(reg) == 1) {
               if (!agx3_alu_fadd_abs_negzero(
                      enc, I->dest[0].value, reg.value, I->dest[0].cache,
                      agx3_post_ra_hint(reg), wait, bytes, &len) ||
                   !agx3_append(emission, bytes, len))
                  AGX3_REFUSE(I);
               continue;
            }
            if (!multiply && !I->saturate && reg.abs && reg.neg && !reg.kill &&
                imm.value == 0x30 && !imm.memory && !imm.abs && !imm.neg &&
                agx3_src_is_register(reg) && !reg.memory &&
                !(reg.cache && reg.discard) && reg.size == AGX_SIZE_32 &&
                agx_channels(reg) == 1) {
               if (!agx3_alu_fadd_one_absneg(
                      enc, I->dest[0].value, reg.value, I->dest[0].cache,
                      agx3_post_ra_hint(reg), wait, bytes, &len) ||
                   !agx3_append(emission, bytes, len))
                  AGX3_REFUSE(I);
               continue;
            }
            if (imm.memory || imm.abs || imm.neg ||
                !agx3_src_is_register(reg) ||
                (reg.kill && !reg.discard) ||
                (multiply && reg.neg) ||
                (reg.cache && reg.discard) ||
                reg.size != AGX_SIZE_32 || agx_channels(reg) != 1)
               AGX3_REFUSE(I);

            bool encoded;
            if (!multiply && !reg.abs && (I->saturate || reg.neg)) {
               encoded = agx3_alu_fadd_minifloat_mod(
                  enc, I->dest[0].value, reg.value, imm.value, reg.neg,
                  I->saturate, agx3_post_ra_hint(reg), wait, bytes, &len);
            } else if (multiply) {
               encoded = agx3_alu_fmul_minifloat_mod(
                  enc, I->dest[0].value, reg.value, imm.value,
                  I->dest[0].cache, reg.abs, agx3_post_ra_hint(reg), wait,
                  bytes, &len);
            } else if (reg.abs) {
               encoded = agx3_alu_fadd_abs_minifloat(
                  enc, I->dest[0].value, reg.value, imm.value,
                  I->dest[0].cache, reg.neg, I->saturate,
                  agx3_post_ra_hint(reg), wait, bytes, &len);
            } else {
               encoded = agx3_alu_fadd_minifloat(
                  enc, I->dest[0].value, reg.value, imm.value,
                  agx3_post_ra_hint(reg), wait, bytes, &len);
            }
            if (!encoded || !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_BITOP &&
             I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
             I->dest[0].size == AGX_SIZE_16 &&
             (I->src[0].size == AGX_SIZE_16 || agx3_vary_reg32(I->src[0])) &&
             (I->src[1].size == AGX_SIZE_16 || agx3_vary_reg32(I->src[1])) &&
             agx_channels(I->dest[0]) == 1 &&
             agx_channels(I->src[0]) == 1 && agx_channels(I->src[1]) == 1 &&
             agx3_src_is_register(I->dest[0]) && agx3_src_is_register(I->src[0]) &&
             agx3_src_is_register(I->src[1]) &&
             !I->src[0].abs && !I->src[0].neg && !I->src[1].abs && !I->src[1].neg) {
            const unsigned tt = I->truth_table & 15;
            if (tt == AGX3_SEL_TT_AND || tt == AGX3_SEL_TT_OR || tt == AGX3_SEL_TT_XOR) {
               const enum agx3_alu_op op = tt == AGX3_SEL_TT_AND ? AGX3_ALU_BAND :
                  tt == AGX3_SEL_TT_OR ? AGX3_ALU_BOR : AGX3_ALU_BXOR;
               /* A narrow bitwise result depends only on the low half of
                * each input, including wider IR registers. Native register
                * indices already address halves: retain those indices and
                * the actual lifetime hints without a conversion or copy. */
               const uint16_t src[2] = {I->src[0].value, I->src[1].value};
               const enum agx3_alu_hint hints[2] = {
                  agx3_post_ra_hint(I->src[0]), agx3_post_ra_hint(I->src[1])};
               uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
               if (I->src[0].kill || I->src[1].kill ||
                   !agx3_alu_bit_reg16(enc, op, I->dest[0].value, src, hints, wait,
                                       bytes, &len) ||
                   !agx3_append(emission, bytes, len))
                  AGX3_REFUSE(I);
               continue;
            }
         }

         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_BITOP &&
             I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
             agx3_vary_reg32(I->dest[0]) &&
             agx_channels(I->dest[0]) == 1 &&
             agx3_vary_reg32(I->src[0]) && agx3_vary_reg32(I->src[1]) &&
             agx_channels(I->src[0]) == 1 && agx_channels(I->src[1]) == 1) {
            const unsigned tt = I->truth_table & 0xfu;
            const enum agx3_alu_op op = tt == AGX3_SEL_TT_AND ? AGX3_ALU_BAND :
               tt == AGX3_SEL_TT_OR ? AGX3_ALU_BOR : AGX3_ALU_BXOR;
            if (tt == AGX3_SEL_TT_AND || tt == AGX3_SEL_TT_OR ||
                tt == AGX3_SEL_TT_XOR) {
               if (I->src[0].kill || I->src[1].kill ||
                   (I->src[0].cache && I->src[0].discard) ||
                   (I->src[1].cache && I->src[1].discard))
                  AGX3_REFUSE(I);
               const uint16_t src[2]={I->src[0].value,I->src[1].value};
               const enum agx3_alu_hint hints[2]={
                  agx3_post_ra_hint(I->src[0]),agx3_post_ra_hint(I->src[1])};
               uint8_t bytes[AGX3_ALU_MAX_LENGTH],len=0;
               if (!agx3_alu_bit_reg(enc,op,I->dest[0].value,src,hints,wait,
                                     bytes,&len) ||
                   !agx3_append(emission,bytes,len))
                  AGX3_REFUSE(I);
               continue;
            }
         }

         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_CONVERT &&
             I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
             agx3_vary_reg32(I->dest[0]) && agx_channels(I->dest[0]) == 1 &&
             agx3_vary_reg32(I->src[1]) && agx_channels(I->src[1]) == 1) {
            struct agx3_sel_instr S;
            if (agx3_translate_instr(I, &S) &&
                (S.op == AGX3_SEL_OP_I32_TO_F || S.op == AGX3_SEL_OP_U32_TO_F)) {
               if (I->src[1].kill || (I->src[1].cache && I->src[1].discard))
                  AGX3_REFUSE(I);
               uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
               if (!agx3_alu_int_to_float(enc, I->dest[0].value, I->src[1].value,
                       S.op == AGX3_SEL_OP_I32_TO_F,
                       agx3_post_ra_hint(I->src[1]), wait, bytes, &len) ||
                   !agx3_append(emission, bytes, len))
                  AGX3_REFUSE(I);
               continue;
            }
         }

         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_FFMA &&
             !I->saturate && I->nr_dests == 1 && I->nr_srcs == 3 &&
             agx3_vary_reg32(I->dest[0]) && agx_channels(I->dest[0]) == 1 &&
             agx3_mod_src_ok(I->src[0]) && !I->src[0].abs &&
             agx_channels(I->src[0]) == 1 &&
             agx3_mod_src_ok(I->src[1]) && !I->src[1].abs &&
             agx_channels(I->src[1]) == 1 &&
             I->src[2].type == AGX_INDEX_IMMEDIATE && I->src[2].value <= 255 &&
             !I->src[2].abs && !I->src[2].neg && !I->src[2].memory) {
            struct agx3_falu_runtime runtime = {
               .dest_cache = I->dest[0].cache, .wait_mask = wait,
            };
            for (unsigned i = 0; i < 2; ++i) {
               if (I->src[i].kill)
                  AGX3_REFUSE(I);
               runtime.cache[i] = I->src[i].cache;
               runtime.discard[i] = I->src[i].discard;
            }
            uint8_t bytes[AGX3_FALU_MOD_MAX_LENGTH], length;
            if (!agx3_falu_fma_addend_imm_mod(enc, I->dest[0].value,
                    I->src[0].value, I->src[1].value, I->src[2].value,
                    I->src[0].neg, I->src[1].neg, &runtime, bytes, &length) ||
                !agx3_append(emission, bytes, length))
               AGX3_REFUSE(I);
            continue;
         }

         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_FFMA &&
             !I->saturate && I->nr_dests == 1 && I->nr_srcs == 3 &&
             agx3_vary_reg32(I->dest[0]) && agx_channels(I->dest[0]) == 1 &&
             I->src[2].type == AGX_INDEX_IMMEDIATE && I->src[2].value <= 255 &&
             !I->src[2].abs && !I->src[2].neg && !I->src[2].memory) {
            const bool imm_first = I->src[0].type == AGX_INDEX_IMMEDIATE;
            const agx_index imm = I->src[imm_first ? 0 : 1];
            const agx_index reg = I->src[imm_first ? 1 : 0];
            if (imm.type == AGX_INDEX_IMMEDIATE && imm.value <= 255 &&
                !imm.abs && !imm.neg && !imm.memory &&
                agx3_vary_reg32(reg) && agx_channels(reg) == 1) {
               if (reg.kill)
                  AGX3_REFUSE(I);
               const struct agx3_falu_runtime runtime = {
                  .dest_cache = I->dest[0].cache, .wait_mask = wait,
                  .cache = {reg.cache}, .discard = {reg.discard},
               };
               uint8_t bytes[AGX3_FALU_MOD_MAX_LENGTH], length;
               if (!agx3_falu_fma_two_imm(enc, I->dest[0].value,
                       reg.value, imm.value, I->src[2].value,
                       &runtime, bytes, &length) ||
                   !agx3_append(emission, bytes, length))
                  AGX3_REFUSE(I);
               continue;
            }
         }

         /* Native FMA with a minifloat multiplier. Preserve both async
          * dependencies in the instruction, including the extended two-wait
          * form; do not serialize independent texture operations. */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_FFMA &&
             !I->saturate && I->nr_dests == 1 && I->nr_srcs == 3 &&
             agx3_vary_reg32(I->dest[0]) && agx_channels(I->dest[0]) == 1) {
            const bool imm_first = I->src[0].type == AGX_INDEX_IMMEDIATE;
            const agx_index imm = I->src[imm_first ? 0 : 1];
            const agx_index reg = I->src[imm_first ? 1 : 0];
            const agx_index add = I->src[2];
            agx_index add_plain = add;
            add_plain.neg = false;
            if (imm.type == AGX_INDEX_IMMEDIATE && imm.value <= 255 &&
                !imm.memory && !imm.abs && !imm.neg &&
                agx3_vary_reg32(reg) && agx_channels(reg) == 1 &&
                agx3_vary_reg32(add_plain) && agx_channels(add) == 1 &&
                !add.abs) {
               uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
               if (reg.kill || add.kill || (reg.cache && reg.discard) ||
                   (add.cache && add.discard) ||
                   !agx3_alu_fma_minifloat_mod(enc,I->dest[0].value,reg.value,
                       imm.value,add.value,add.neg,agx3_post_ra_hint(reg),
                       agx3_post_ra_hint(add),I->dest[0].cache,wait,bytes,&len) ||
                   !agx3_append(emission,bytes,len))
                  AGX3_REFUSE(I);
               continue;
            }
         }

         /* The legacy generic packers embed specimen discard flags. Keep
          * G17P's plain register ALU on the post-RA-aware path so an alpha or
          * blend factor reused by later channels is not discarded early. */
         if (enc == AGX_ISA_ENCODING_AGX3B && !I->saturate &&
             (I->op == AGX_OPCODE_FADD || I->op == AGX_OPCODE_FMUL ||
              I->op == AGX_OPCODE_FFMA)) {
            bool plain = I->nr_dests == 1 &&
                         agx3_vary_reg32(I->dest[0]) &&
                         agx_channels(I->dest[0]) == 1 &&
                         I->nr_srcs == (I->op == AGX_OPCODE_FFMA ? 3 : 2);
            for (unsigned i = 0; i < I->nr_srcs; ++i)
               plain &= agx3_vary_reg32(I->src[i]) &&
                        agx_channels(I->src[i]) == 1;
            if (plain) {
               uint16_t src[3] = {0};
               enum agx3_alu_hint hints[3] = {0};
               for (unsigned i = 0; i < I->nr_srcs; ++i) {
                  if (I->src[i].kill ||
                      (I->src[i].cache && I->src[i].discard))
                     AGX3_REFUSE(I);
                  src[i] = I->src[i].value;
                  hints[i] = agx3_post_ra_hint(I->src[i]);
               }
               const enum agx3_alu_op op = I->op == AGX_OPCODE_FADD
                  ? AGX3_ALU_FADD : I->op == AGX_OPCODE_FMUL
                  ? AGX3_ALU_FMUL : AGX3_ALU_FMA;
               uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
               if (!agx3_alu_float_reg(enc, op, I->dest[0].value, src, hints,
                                        wait, bytes, &len)) {
                  /* Compact forms cannot carry every wait combination. Drain
                   * all outstanding work before the r0-preserving prelude,
                   * then retry the same operands/hints without waits. */
                  if (wait == 0 ||
                      !agx3_retire_slots_noop(emission, enc,
                                              wait | outstanding_here) ||
                      !agx3_alu_float_reg(enc, op, I->dest[0].value, src,
                                           hints, 0, bytes, &len))
                     AGX3_REFUSE(I);
                  outstanding = 0;
               }
               if (!agx3_append(emission, bytes, len))
                  AGX3_REFUSE(I);
               continue;
            }
         }

         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_ASR &&
             I->nr_dests == 1 && I->nr_srcs == 2 &&
             I->src[1].type == AGX_INDEX_IMMEDIATE &&
             I->src[1].value >= 1 && I->src[1].value <= 31 &&
             !I->src[1].abs && !I->src[1].neg) {
            const agx_index dest=I->dest[0], src=I->src[0];
            if (!agx3_vary_reg32(dest) || !agx3_vary_reg32(src) ||
                src.abs || src.neg || agx_channels(dest)!=1 || agx_channels(src)!=1)
               AGX3_REFUSE(I);
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], length;
            if (!agx3_alu_asr_imm32(enc,dest.value,src.value,I->src[1].value,
                                     agx3_post_ra_hint(src),dest.cache,wait,bytes,&length) ||
                !agx3_append(emission,bytes,length))
               AGX3_REFUSE(I);
            continue;
         }

         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_EXTR &&
             I->nr_dests==1 && I->nr_srcs==3 && I->bfi_mask==0 &&
             I->src[2].type==AGX_INDEX_IMMEDIATE &&
             I->src[2].value>=1 && I->src[2].value<=31 &&
             !I->src[2].abs && !I->src[2].neg) {
            const agx_index dst=I->dest[0], lo=I->src[0], hi=I->src[1];
            if (!agx3_vary_reg32(dst) || !agx3_vary_reg32(lo) ||
                !agx3_vary_reg32(hi) || lo.abs || lo.neg || hi.abs || hi.neg ||
                agx_channels(dst)!=1 || agx_channels(lo)!=1 || agx_channels(hi)!=1)
               AGX3_REFUSE(I);
            const uint16_t src[2]={lo.value,hi.value};
            const enum agx3_alu_hint hints[2]={agx3_post_ra_hint(lo),agx3_post_ra_hint(hi)};
            uint8_t bytes[AGX3_ALU_MAX_LENGTH],length;
            if (!agx3_alu_extract_imm32(enc,dst.value,src,I->src[2].value,hints,
                                         dst.cache,wait,bytes,&length) ||
                !agx3_append(emission,bytes,length))
               AGX3_REFUSE(I);
            continue;
         }

         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_BFEIL &&
             I->nr_dests == 1 && I->nr_srcs == 3 && I->imm >= 1 && I->imm <= 31 &&
             agx3_src_is_zero_imm(I->src[0])) {
            const agx_index dest=I->dest[0], value=I->src[1], shift=I->src[2];
            const bool reg_shift=agx3_src_is_register(shift);
            if (!agx3_src_is_register(dest) || dest.memory || dest.size != AGX_SIZE_32 ||
                agx_channels(dest) != 1 || !agx3_src_is_register(value) || value.memory ||
                value.size != AGX_SIZE_32 || agx_channels(value) != 1 || value.abs || value.neg ||
                shift.memory || shift.abs || shift.neg || agx_channels(shift) != 1 ||
                (reg_shift ? shift.size != AGX_SIZE_32 : shift.type != AGX_INDEX_IMMEDIATE))
               AGX3_REFUSE(I);
            const enum agx3_alu_hint hints[2] = {
               agx3_post_ra_hint(value), reg_shift ? agx3_post_ra_hint(shift) : AGX3_ALU_HINT_NONE,
            };
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], length;
            if (!agx3_alu_bfeil(enc,dest.value,value.value,shift.value,reg_shift,
                                I->imm,hints,wait,bytes,&length) ||
                !agx3_append(emission,bytes,length))
               AGX3_REFUSE(I);
            continue;
         }

         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_FFS &&
             I->nr_dests == 1 && I->nr_srcs == 1) {
            const agx_index dest = I->dest[0], src = I->src[0];
            if (!agx3_vary_reg32(dest) || !agx3_vary_reg32(src) ||
                agx_channels(dest) != 1 || agx_channels(src) != 1 ||
                dest.value == src.value || src.abs || src.neg)
               AGX3_REFUSE(I);
            /* Apple's G17P clz FT1 carries wait0 and the cached destination
             * form (op-enable 3). The uncached/op-enable-2 combination with
             * a discarded source exposes the internal logarithmic stage
             * instead of the completed bit index. */
            const uint16_t ft1_wait = wait | 1u;
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], length;
            if (!agx3_alu_ft1_u32(enc, dest.value, src.value,
                                  agx3_post_ra_hint(src),
                                  true, ft1_wait, bytes, &length) ||
                !agx3_append(emission, bytes, length))
               AGX3_REFUSE(I);
            continue;
         }

         if (enc == AGX_ISA_ENCODING_AGX3B &&
             I->op == AGX_OPCODE_POPCOUNT && I->nr_dests == 1 &&
             I->nr_srcs == 1) {
            const agx_index dest = I->dest[0], src = I->src[0];
            const bool half = dest.size == AGX_SIZE_16 &&
                              src.size == AGX_SIZE_16;
            const bool word = dest.size == AGX_SIZE_32 &&
                              src.size == AGX_SIZE_32;
            if ((!half && !word) || !agx3_src_is_register(dest) ||
                dest.memory || agx_channels(dest) != 1 ||
                !agx3_src_is_register(src) || src.memory ||
                agx_channels(src) != 1 || src.abs || src.neg)
               AGX3_REFUSE(I);

            uint8_t bytes[AGX3_ALU_MAX_LENGTH], length;
            if (wait && !agx3_retire_slots_noop(emission, enc, wait))
               AGX3_REFUSE(I);
            if (!agx3_alu_popcount(enc, dest.value, src.value, half,
                                   agx3_post_ra_hint(src), dest.cache, 0,
                                   bytes, &length) ||
                !agx3_append(emission, bytes, length))
               AGX3_REFUSE(I);
            continue;
         }

         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_BITOP &&
             (I->truth_table & 0xfu) == 0x2u && I->nr_dests == 1 &&
             I->nr_srcs == 2) {
            const agx_index dest = I->dest[0];
            if (!agx3_vary_reg32(dest) || agx_channels(dest) != 1)
               AGX3_REFUSE(I);
            uint16_t src[2];
            enum agx3_alu_hint hints[2];
            for (unsigned i = 0; i < 2; ++i) {
               if (!agx3_vary_reg32(I->src[i]) ||
                   agx_channels(I->src[i]) != 1 || I->src[i].abs ||
                   I->src[i].neg)
                  AGX3_REFUSE(I);
               src[i] = I->src[i].value;
               hints[i] = agx3_post_ra_hint(I->src[i]);
            }
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], length;
            if (!agx3_alu_bit_andc(enc, dest.value, src, hints, dest.cache,
                                   wait, bytes, &length) ||
                !agx3_append(emission, bytes, length))
               AGX3_REFUSE(I);
            continue;
         }

         /* G17P adds an absolute f32 source directly to a scalar-half
          * register. Keep the half in its assigned register rather than
          * widening it through a temporary. */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_FADD &&
             I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
             agx3_vary_reg32(I->dest[0]) &&
             agx_channels(I->dest[0]) == 1 &&
             agx3_mod_src_ok(I->src[0]) &&
             agx_channels(I->src[0]) == 1 && I->src[0].abs &&
             !I->src[0].neg && agx3_half_reg(I->src[1]) &&
             agx_channels(I->src[1]) == 1 && !I->src[1].abs &&
             !I->src[1].neg) {
            const agx_index dst = I->dest[0], f32 = I->src[0],
                            f16 = I->src[1];
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
            if (f32.kill || f16.kill || (f32.cache && f32.discard) ||
                (f16.cache && f16.discard) ||
                !agx3_alu_fadd_abs_mixed_half(
                   enc, dst.value, f32.value, f16.value,
                   agx3_post_ra_hint(f32), agx3_post_ra_hint(f16), dst.cache,
                   wait, bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         /* Modified G17P ADD/FMA carry their native dependency waits and the
          * actual post-RA hints, without draining unrelated texture slots. */
         const int mod = agx3_try_float_modifier(emission, enc, I, wait);
         if (mod == 1)
            continue;
         if (mod < 0)
            AGX3_REFUSE(I);

         struct agx3_sel_instr S;
         if (!agx3_translate_instr(I, &S) ||
             !agx3_emit_alu(emission, enc, &S, wait)) {
            if (agx_compiler_debug & AGX_DBG_SHADERS)
               fprintf(stderr,
                       "ALUX %s wait=%x shift=%u d{v=%u sz=%u} s0{t=%u v=%u "
                       "sz=%u abs=%u} s1{t=%u v=%u sz=%u abs=%u}\n",
                       agx_opcodes_info[I->op].name, wait, I->shift,
                       I->dest[0].value, I->dest[0].size, I->src[0].type,
                       I->src[0].value, I->src[0].size, I->src[0].abs,
                       I->src[1].type, I->src[1].value, I->src[1].size,
                       I->src[1].abs);
            AGX3_REFUSE(I);
         }
      }
   }

   util_dynarray_foreach(&fixups, struct agx3_branch_fixup, fixup) {
      if (!agx3_fixup_branch(emission, enc, *fixup))
         goto fail;
   }

   /* Separately compiled parts do not share a scoreboard ledger. Apple's
    * monolithic sampled fragment waits for TEX on its first FLTNORM; our
    * epilog cannot know that upstream slot, so finish only the still-live
    * async work at this boundary. Constant-colour parts and parts with real
    * in-part consumers add no instructions. */
   if (ctx->key->no_stop && !ctx->is_preamble && outstanding != 0 &&
       !agx3_retire_slots_noop(emission, enc, outstanding))
      goto fail;

   if (!ctx->key->no_stop || ctx->is_preamble) {
      const struct agx3_sel_instr end = {.op = AGX3_SEL_OP_END};
      if (!agx3_emit(emission, enc, &end))
         goto fail;
   }

   util_dynarray_fini(&fixups);
   return true;

fail:
   /* Do not leave a valid-looking prefix behind.  The compiler entrypoint
    * propagates this failure and clears the shader part transactionally. */
   emission->size = initial_size;
   util_dynarray_fini(&fixups);
   return false;
}
