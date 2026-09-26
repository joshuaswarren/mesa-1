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
#include "agx3_g17p_array_eot.h"
#include "agx3_g17p_quad_eot.h"
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
#include "agx3_cprog_input.h"
#include "agx3_subgroup_core.h"
#include "agx3_tex_core.h"
#include "agx3_tile_core.h"
#include "agx3_uniform_alu.h"
#include "agx3_uniform_integer_instr.h"
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
   bool loop_exit;
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
agx3_report_refusal_why(const agx_instr *I, unsigned line, const char *why)
{
   if (!(agx_compiler_debug & (AGX_DBG_SHADERS | AGX_DBG_REFUSE)))
      return;

   char shape[512];
   int n = snprintf(shape, sizeof(shape), "d%u:%s%u",
                    I->nr_dests ? 16u << I->dest[0].size : 0u,
                    I->nr_dests ? agx3_index_kind(I->dest[0]) : "none",
                    I->nr_dests ? I->dest[0].value : 0u);

   for (unsigned i = 0; i < I->nr_srcs && n > 0 && n < (int)sizeof(shape); ++i) {
      const agx_index x = I->src[i];
      /* An immediate's `value` is a constant, not a register number, so it
       * gets a '#'. Without it `s0=16:imm32` reads as a 32-bit immediate when
       * it is in fact a 16-bit slot holding the minifloat 32 -- and a width
       * and a kind that print as one token is the same class of ambiguity as
       * a field left out entirely. */
      const bool immediate = x.type == AGX_INDEX_IMMEDIATE;
      /* A source's CHANNEL COUNT gates several texture paths -- a 2D fetch
       * wants two coordinate channels and an array or 3D fetch three, and
       * without it two refusals that differ only in dimensionality print
       * identically. Print it whenever it is not the scalar default. */
      const unsigned chans = agx_channels(x);
      n += snprintf(shape + n, sizeof(shape) - n, " s%u=%u:%s%s%u%s%s%s%s%s", i,
                    16u << x.size, agx3_index_kind(x), immediate ? "#" : "",
                    x.value,
                    x.abs ? ".abs" : "", x.neg ? ".neg" : "",
                    x.kill ? ".kill" : "", x.cache ? ".cache" : "",
                    x.discard ? ".discard" : "");
      if (chans != 1 && n > 0 && n < (int)sizeof(shape))
         n += snprintf(shape + n, sizeof(shape) - n, "x%u", chans);
   }

   /* mask, format, shift and nest decide gates that the operand list cannot
    * show. A control-flow refusal turns entirely on `nest` -- a break out of
    * one enclosing block and a break out of five printed identically without
    * it, which is precisely the diagnosis this lane had to redo. (`cond` is
    * the icond/fcond union; which one it is follows from the opcode.)
    * A vector spill differs from a scalar one only in `mask`; and a shifted
    * integer subtract (`a - (b << N)`, which is how a multiply by a small
    * negative constant is strength-reduced) looks exactly like a plain
    * negated add without `shift`. Omitting them costs a wrong diagnosis. */
   fprintf(stderr,
           "agx3_pack_binary.c:%u: no grounded AGX3 encoding for %s "
           "[%s%s%s cond=%u nest=%u chan=%u mask=0x%x fmt=%u "
           "shift=%u dim=%u lod=%u%s%s%s%s%s]%s%s\n",
           line, agx_opcodes_info[I->op].name, shape,
           I->saturate ? " sat" : "", I->invert_cond ? " inv" : "",
           (unsigned)I->fcond, (unsigned)I->nest,
           I->nr_dests ? agx_channels(I->dest[0]) : 0u,
           (unsigned)I->mask, (unsigned)I->format, (unsigned)I->shift,
           (unsigned)I->dim, (unsigned)I->lod_mode,
           I->offset ? " off" : "", I->shadow ? " shadow" : "",
           I->query_lod ? " qlod" : "",
           I->gather != AGX_GATHER_NONE ? " gather" : "",
           I->coherent ? " coherent" : "",
           why ? " -- " : "", why ? why : "");
}

static void
agx3_report_refusal(const agx_instr *I, unsigned line)
{
   agx3_report_refusal_why(I, line, NULL);
}

#define AGX3_REFUSE(I)                                                         \
   do {                                                                        \
      agx3_report_refusal((I), __LINE__);                                      \
      goto fail;                                                               \
   } while (0)

/*
 * Same, naming WHICH stage declined. A memory op passes through a translate
 * step and then an encode step, and when both raise the bare message the log
 * cannot say whether the shape was rejected for a reason the IR could fix or
 * genuinely has no encoding -- which is the difference between IR-side work
 * and ISA work. The line number already separates them; this says it in
 * words so nobody has to map line numbers back to source.
 */
#define AGX3_REFUSE_WHY(I, why)                                                \
   do {                                                                        \
      agx3_report_refusal_why((I), __LINE__, (why));                           \
      goto fail;                                                               \
   } while (0)

/*
 * agx3_emit_texture_sample() screens an operand shape through some thirty
 * gates before it encodes anything, and the caller reports only "no grounded
 * AGX3 encoding for texture_sample". Two shapes that differ in one field --
 * a cube array where the cube encodes, a register descriptor where the
 * immediate one encodes -- then print the same line, so every guess about
 * which field mattered costs a round trip to hardware. Name the gate.
 */
static bool
agx3_tex_decline(const agx_instr *I, unsigned line)
{
   agx3_report_refusal_why(I, line, "texture operand gate");
   return false;
}


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
   S->src[0].live = !value.kill && !value.discard;
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
      S->src[0].live = !I->src[0].kill && !I->src[0].discard;
      S->src[1].reg = I->src[1].value;
      S->src[1].size = I->src[1].size;
      S->src[1].is_reg = true;
      S->src[1].live = !I->src[1].kill && !I->src[1].discard;
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
      S->src[i].live = !I->src[i].kill && !I->src[i].discard;
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

/*
 * An immediate carrying no register-only flag. `memory`, `kill`, `cache` and
 * `discard` are liveness and location, which an immediate operand has none
 * of; honouring the constant while dropping them would encode a different
 * instruction's scoreboard state, so they fail closed instead.
 */
static bool
agx3_plain_immediate(agx_index x)
{
   return x.type == AGX_INDEX_IMMEDIATE && !x.memory && !x.kill && !x.cache &&
          !x.discard;
}

/* A source usable by the modifier long forms: a plain 32-bit register. */
static bool
agx3_mod_src_ok(agx_index s)
{
   return agx3_src_is_register(s) && !s.memory && s.size == AGX_SIZE_32 &&
          (s.value & 1) == 0 && s.value < 256;
}

/*
 * The same, admitting a 16-bit operand on any half. The slot's width is a
 * field of the encoding point that carries its modifier bits (f43 with
 * f47/f48, f26 with f30/f31, f60 with f64/f65), so a narrow operand and a
 * modifier are not competing forms -- see the byte ladder above
 * struct agx3_falu_mods. Modelling them as separate encoders is what refused
 * `half + (-float)`, which is on both axes at once.
 */
static bool
agx3_mod_src_any_width(agx_index s)
{
   return agx3_src_is_register(s) && !s.memory && s.value < 256 &&
          agx_channels(s) == 1 &&
          ((s.size == AGX_SIZE_32 && (s.value & 1) == 0) ||
           s.size == AGX_SIZE_16);
}

/*
 * Route native uniform arithmetic before the existing register-only paths.
 * Return 1 on emission, 0 for other instructions, and -1 on an invalid operand.
 */
static int
agx3_try_uniform_integer(struct util_dynarray *emission,
                         enum agx_isa_encoding enc, const agx_instr *I,
                         unsigned wait)
{
   struct agx3_uniform_integer_desc d;
   if (enc != AGX_ISA_ENCODING_AGX3B ||
       !agx3_uniform_integer_instruction(I, true, &d))
      return 0;
   if (wait >= 64)
      return -1;
   d.wait_mask = wait;
   uint8_t bytes[16], length;
   if (!agx3_uniform_integer_encode(enc, &d, bytes, &length))
      return -1;
   memcpy(util_dynarray_grow_bytes(emission, 1, length), bytes, length);
   return 1;
}

static int
agx3_try_uniform_float(struct util_dynarray *emission,
                       enum agx_isa_encoding enc, const agx_instr *I, unsigned wait)
{
   const bool half = I->op == AGX_OPCODE_HADD || I->op == AGX_OPCODE_HMUL ||
                     I->op == AGX_OPCODE_HFMA;
   if (enc != AGX_ISA_ENCODING_AGX3B ||
       (!half && I->op != AGX_OPCODE_FADD && I->op != AGX_OPCODE_FMUL &&
        I->op != AGX_OPCODE_FFMA))
      return 0;
   bool uniform = false;
   agx_foreach_src(I, s)
      uniform |= I->src[s].type == AGX_INDEX_UNIFORM;
   if (!uniform)
      return 0;

   const bool fma = I->op == AGX_OPCODE_FFMA || I->op == AGX_OPCODE_HFMA;
   const bool add = I->op == AGX_OPCODE_FADD || I->op == AGX_OPCODE_HADD;
   const unsigned count = fma ? 3 : 2;
   const enum agx_size size = half ? AGX_SIZE_16 : AGX_SIZE_32;
   if (I->nr_dests != 1 || I->nr_srcs != count ||
       !agx3_src_is_register(I->dest[0]) || I->dest[0].memory ||
       I->dest[0].size != size || agx_channels(I->dest[0]) != 1)
      return -1;

   struct agx3_uniform_float_src src[3] = {0};
   for (unsigned s = 0; s < count; ++s) {
      const agx_index v = I->src[s];
      if (v.memory || v.kill || agx_channels(v) != 1 ||
          (v.type != AGX_INDEX_IMMEDIATE && v.size != size))
         return -1;
      src[s] = (struct agx3_uniform_float_src){
         .kind = v.type == AGX_INDEX_UNIFORM ? AGX3_FLOAT_UNIFORM :
                 v.type == AGX_INDEX_IMMEDIATE ? AGX3_FLOAT_IMMEDIATE :
                 agx3_src_is_register(v) ? AGX3_FLOAT_REGISTER : 0,
         .value = v.value,
         .abs = v.abs, .neg = v.neg,
         .cache = v.cache, .discard = v.discard,
      };
   }
   const enum agx3_uniform_float_op op = fma ? AGX3_UNIFORM_FMA :
                                           add ? AGX3_UNIFORM_FADD : AGX3_UNIFORM_FMUL;
   uint8_t bytes[16], length;
   bool ok = half
      ? agx3_uniform_half_encode(enc, op, I->dest[0].value, src,
                                 I->saturate, I->dest[0].cache, wait, bytes, &length)
      : agx3_uniform_float_encode(enc, op, I->dest[0].value, src,
                                  I->saturate, I->dest[0].cache, wait, bytes, &length);
   if (!ok)
      return -1;
   memcpy(util_dynarray_grow_bytes(emission, 1, length), bytes, length);
   return 1;
}

/* Same return convention for modified register-source arithmetic. */
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
   /* The destination checks used to be carried by the source ones: a half
    * operand failed agx3_mod_src_ok() and took the whole instruction with it,
    * so a vector or spilled destination never reached the encoder. Admitting
    * a narrow operand removes that accident, so state them. */
   if (I->nr_dests < 1 || I->dest[0].type == AGX_INDEX_NULL ||
       !agx3_src_is_register(I->dest[0]) || I->dest[0].memory ||
       agx_channels(I->dest[0]) != 1 ||
       I->dest[0].size != AGX_SIZE_32 || (I->dest[0].value & 1) != 0 ||
       I->dest[0].value >= 256 || I->nr_srcs != want)
      return -1;
   for (unsigned s = 0; s < want; ++s)
      if (!agx3_mod_src_any_width(I->src[s]))
         return -1;

   struct agx3_falu_mods m = {
      .src0_abs = I->src[0].abs,
      .src0_neg = I->src[0].neg,
      .src0_16 = I->src[0].size == AGX_SIZE_16,
      .src1_abs = I->src[1].abs,
      .src1_neg = I->src[1].neg,
      .src1_16 = I->src[1].size == AGX_SIZE_16,
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
      m.addend_16 = I->src[2].size == AGX_SIZE_16;
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
 * AGX3 form. roundeven keeps its existing select-core route. The native sinc
 * has a dedicated IR opcode, fed by G17P's split-constant range reduction
 * and masked quadrant selects. AGX2 sin_pt_1/2 remain unmapped here.
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
   case AGX_OPCODE_SINC_AGX3:
      return AGX3_FUNOP_SINC;
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
   return agx3_funop_encode_source_lifetime(
             enc, op, I->dest[0].value, I->src[0].value, I->src[0].abs,
             I->src[0].neg, I->saturate, wait_mask, !I->src[0].discard,
             bytes, &len) &&
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
          I->op == AGX_OPCODE_LOAD_CPROG_AGX3 ||
          I->op == AGX_OPCODE_STACK_LOAD ||
          I->op == AGX_OPCODE_STACK_STORE;
}

static bool
agx3_slot_instr_reads(const struct agx3_slot_instr *consumer,
                      const struct agx3_slot_instr *producer)
{
   for (unsigned src = 0; src < consumer->nr_srcs; ++src) {
      for (unsigned dst = 0; dst < producer->nr_dsts; ++dst) {
         if (consumer->src[src] == producer->dst[dst])
            return true;
      }
   }

   return false;
}

/* Map a Mesa special register to the corpus-witnessed Apple SR number and
 * destination width. Only registers tied to a source-level builtin by an
 * isolating probe are mapped; everything else -- including any (SR, width)
 * combination Apple never emitted -- fails closed so the shader is honestly
 * rejected, not miscompiled. */
static bool
agx3_map_sr(enum agx_isa_encoding enc, enum agx_sr sr, uint8_t *apple,
            enum agx3_sr_width *width)
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
   case AGX_SR_INPUT_SAMPLE_MASK:
      if (enc != AGX_ISA_ENCODING_AGX3B)
         return false;
      *apple = 39;
      *width = AGX3_SR_W16;
      return true;
      SR(AGX_SR_COVERAGE_MASK, 39, AGX3_SR_W16);
      SR(AGX_SR_IS_ACTIVE_THREAD, 2, AGX3_SR_W16);
   case AGX_SR_BACKFACING:
      if (enc != AGX_ISA_ENCODING_AGX3B)
         return false;
      *apple = 41;
      *width = AGX3_SR_W16;
      return true;
   case AGX_SR_BACKGROUND_LAYER_ID:
      if (enc != AGX_ISA_ENCODING_AGX3B)
         return false;
      *apple = 12;
      *width = AGX3_SR_W16;
      return true;
      /* Apple's uniform-address atomic election reads the active-thread
       * index (SR 34) and count (SR 36) of the SIMD-group, and quad_is_first
       * reads SR 33, all into half registers in the short form, also after
       * discard_fragment (build/claude-fix/witness-sr59/REPORT.md, 15/15
       * byte-exact re-encodes). */
      SR(AGX_SR_ACTIVE_THREAD_INDEX_IN_QUAD, 33, AGX3_SR_W16);
      SR(AGX_SR_ACTIVE_THREAD_INDEX_IN_SUBGROUP, 34, AGX3_SR_W16);
      SR(AGX_SR_TOTAL_ACTIVE_THREADS_IN_SUBGROUP, 36, AGX3_SR_W16);
      /* The invocation enums carry hardware SR bytes (0xd8/0xdd), while
       * agx3_sr_core consumes Apple's normalized f21 value. */
      SR(AGX_SR_INSTANCE_ID, 46, AGX3_SR_W32);
      SR(AGX_SR_TESS_PATCH_ID, 47, AGX3_SR_W32);
      SR(AGX_SR_TESS_COORD_U, 48, AGX3_SR_W32);
      SR(AGX_SR_TESS_COORD_V, 49, AGX3_SR_W32);
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
agx3_translate_sr(enum agx_isa_encoding enc, const agx_instr *I, uint8_t slot,
                  uint16_t wait_mask, struct agx3_sr_desc *d)
{
   memset(d, 0, sizeof(*d));

   if (I->nr_dests < 1 || I->dest[0].type == AGX_INDEX_NULL)
      return false;
   const agx_index dst = I->dest[0];
   if (!agx3_src_is_register(dst) || dst.memory || dst.value >= 256)
      return false;

   uint8_t apple;
   enum agx3_sr_width width;
   if (!agx3_map_sr(enc, I->sr, &apple, &width))
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

/* Explicit sample and G17P centroid coordinates are an opaque 32-bit token.
 * Require the most recent overlapping write in this block to be ITRCOORDS of
 * the same interpolation, so a legacy ITER sample-index register or an
 * unrelated arithmetic value cannot accidentally select the native f32interp
 * coordinate leaf. A tied final interpolation is valid because sources are
 * read before its destination overwrites the token. */
static bool
agx3_sample_coords_use(const agx_block *block, const agx_instr *consumer)
{
   if (consumer->nr_srcs != 2 || !agx3_vary_reg32(consumer->src[1]))
      return false;

   const unsigned base = consumer->src[1].value;
   const agx_instr *producer = NULL;
   agx_foreach_instr_in_block(block, I) {
      if (I == consumer)
         return producer != NULL;

      agx_foreach_dest(I, d) {
         const agx_index dst = I->dest[d];
         if (!agx3_src_is_register(dst) || dst.memory)
            continue;
         const unsigned end = dst.value + agx_index_size_16(dst);
         if (end <= base || dst.value >= base + 2)
            continue;

         producer = I->op == AGX_OPCODE_ITRCOORDS_AGX3 && dst.value == base &&
                          dst.size == AGX_SIZE_32 && agx_channels(dst) == 1 &&
                          I->interpolation == consumer->interpolation
                       ? I
                       : NULL;
      }
   }

   return false;
}

static bool
agx3_interp_uses_coords(enum agx_isa_encoding enc, const agx_instr *I)
{
   return I->interpolation == AGX_INTERPOLATION_SAMPLE ||
          (I->interpolation == AGX_INTERPOLATION_CENTROID &&
           enc == AGX_ISA_ENCODING_AGX3B);
}

static bool
agx3_emit_interp(struct util_dynarray *emission, enum agx_isa_encoding enc,
                 const agx_instr *I, uint16_t wait_mask, bool first,
                 bool fragment)
{
   if (wait_mask != 0 || I->nr_dests != 1 || I->nr_srcs != 2 ||
       !agx3_vary_reg32(I->dest[0]) || I->src[0].type != AGX_INDEX_IMMEDIATE)
      return false;

   /* Sample and G17P centroid interpolation consume an ITRCOORDS token
    * through the same f32interp leaves. */
   const bool token = agx3_interp_uses_coords(enc, I);
   enum agx3_vary_location location;
   if (token) {
      if (enc != AGX_ISA_ENCODING_AGX3B || first ||
          !agx3_vary_reg32(I->src[1]) || I->src[1].kill ||
          (I->src[1].cache && I->src[1].discard))
         return false;
   } else {
      if (I->src[1].type != AGX_INDEX_IMMEDIATE || I->src[1].value != 0 ||
          !agx3_map_vary_location(I->interpolation, &location))
         return false;
   }

   uint8_t bytes[AGX3_VARY_MAX_LENGTH];
   uint8_t len = 0;
   if (token && I->op == AGX_OPCODE_RHW_INTERP_AGX3) {
      if (I->saturate)
         return false;
      return agx3_vary_projective_rhw_sample(enc, I->dest[0].value,
                                             I->src[0].value, I->src[1].value,
                                             I->src[1].discard, bytes, &len) &&
             agx3_append(emission, bytes, len);
   }
   if (token) {
      return agx3_vary_interp_sample(enc, I->dest[0].value, I->src[0].value,
                                     I->perspective, I->src[1].value,
                                     I->src[1].discard, bytes, &len) &&
             agx3_append(emission, bytes, len);
   }

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
agx3_emit_itrcoords(struct util_dynarray *emission, enum agx_isa_encoding enc,
                    const agx_instr *I, uint16_t wait_mask)
{
   if (I->nr_dests != 1 || I->nr_srcs != 1 || !agx3_vary_reg32(I->dest[0]) ||
       !agx3_src_is_register(I->src[0]) || I->src[0].memory ||
       I->src[0].size != AGX_SIZE_16 || agx_channels(I->src[0]) != 1 ||
       I->src[0].value >= 256 || I->src[0].abs || I->src[0].neg ||
       I->src[0].kill || (I->src[0].cache && I->src[0].discard))
      return false;

   uint8_t bytes[AGX3_VARY_MAX_LENGTH];
   uint8_t len = 0;
   /* Cache is an optional post-RA hint. The fresh ITRCOORDS witnesses cover
    * the unmarked and terminal source forms, so preserve the terminal bit and
    * deliberately leave a merely cached source unmarked. */
   bool encoded;
   switch (I->interpolation) {
   case AGX_INTERPOLATION_SAMPLE:
      encoded = agx3_vary_sample_coords(enc, I->dest[0].value,
                                        I->src[0].value, I->src[0].discard,
                                        wait_mask, bytes, &len);
      break;
   case AGX_INTERPOLATION_CENTROID:
      encoded = agx3_vary_centroid_coords(enc, I->dest[0].value,
                                          I->src[0].value, I->src[0].discard,
                                          wait_mask, bytes, &len);
      break;
   default:
      return false;
   }
   return encoded && agx3_append(emission, bytes, len);
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
   /* An unlinked vertex shader computes its user outputs' indices from the
    * linked varying layout, so the index is then a 16-bit register. */
   const agx_index index = I->src[0];
   const bool register_index = agx3_src_is_register(index) &&
      !index.memory && !index.abs && !index.neg &&
      index.size == AGX_SIZE_16 && agx_channels(index) == 1;
   if (I->nr_dests != 0 || I->nr_srcs != 2 ||
       (index.type != AGX_INDEX_IMMEDIATE && !register_index) ||
       (register_index && defer_writeout) ||
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
   if (defer_writeout)
      return true;
   const uint16_t staged = (uint16_t)(1u << (slot - 1));
   return (register_index
              ? agx3_vary_uvsw_register_index(enc, staging_half, index.value,
                                              slot, true, staged, bytes, &len)
              : agx3_vary_uvsw(enc, staging_half, index.value, slot, true,
                               staged, bytes, &len)) &&
          agx3_append(emission, bytes, len);
}

static bool
agx3_retire_slots_noop(struct util_dynarray *emission,
                       enum agx_isa_encoding enc, uint16_t mask);

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

/* Translate the corpus-grounded texel fetch. Dynamic integer LOD is
 * published into operand bank 6. Native G17P array fetches publish their
 * combined layer/LOD word and return the texture result on slot 1; plain 2D
 * and 3D fetches keep the independently witnessed staging slot 2/result slot
 * 1 pair. G17P 3D reads publish their third integer coordinate in the next
 * staging word; every other texture axis remains a deliberate refusal. */
/* Image-write banks stay below this staging half, which also keeps every
 * staged half within the COPY destination field. */
#define AGX3_STAGED_WRITE_LIMIT 128u

/* Staging banks held by asynchronous consumers still in flight: staged reads
 * and image writes. A bank stays held until the slot its consumer signals is
 * synchronised. */
struct agx3_staging_banks {
   struct {
      uint16_t coord, span;
      uint8_t slot;
   } live[8];
   uint16_t end;
   uint8_t next_write_slot;
};

static void
agx3_staging_release(struct agx3_staging_banks *banks, uint16_t outstanding)
{
   for (unsigned i = 0; i < ARRAY_SIZE(banks->live); ++i) {
      if (banks->live[i].slot &&
          !(outstanding & (1u << (banks->live[i].slot - 1))))
         banks->live[i].slot = 0;
   }
}

/* The slots of held banks overlapping [start, start + span). */
static uint16_t
agx3_staging_conflicts(const struct agx3_staging_banks *banks, unsigned start,
                       unsigned span)
{
   uint16_t conflict = 0;
   for (unsigned i = 0; i < ARRAY_SIZE(banks->live); ++i) {
      if (banks->live[i].slot && banks->live[i].coord < start + span &&
          start < banks->live[i].coord + banks->live[i].span)
         conflict |= (uint16_t)(1u << (banks->live[i].slot - 1));
   }
   return conflict;
}

/* Every slot holding a bank. */
static uint16_t
agx3_staging_held(const struct agx3_staging_banks *banks)
{
   return agx3_staging_conflicts(banks, 0, UINT16_MAX);
}

static void
agx3_staging_forget(struct agx3_staging_banks *banks, uint16_t slots)
{
   for (unsigned i = 0; i < ARRAY_SIZE(banks->live); ++i) {
      if (banks->live[i].slot && (slots & (1u << (banks->live[i].slot - 1))))
         banks->live[i].slot = 0;
   }
}

static bool
agx3_staging_hold(struct agx3_staging_banks *banks, uint16_t coord,
                  uint16_t span, uint8_t slot)
{
   banks->end = MAX2(banks->end, coord + span);
   for (unsigned i = 0; i < ARRAY_SIZE(banks->live); ++i) {
      if (!banks->live[i].slot) {
         banks->live[i].coord = coord;
         banks->live[i].span = span;
         banks->live[i].slot = slot;
         return true;
      }
   }
   return false;
}

static bool
agx3_emit_texture_fetch(struct util_dynarray *emission,
                        enum agx_isa_encoding enc, const agx_instr *I,
                        uint8_t slot, uint16_t wait_mask, bool final,
                        bool stage_coordinates, bool separate_write_coords,
                        uint16_t ms_array_scratch,
                        struct agx3_staging_banks *banks,
                        uint16_t outstanding,
                        uint16_t *staged_end, bool pixel_entry,
                        uint32_t *pixel_entry_offset)
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
   const bool array = I->dim == AGX_DIM_2D_ARRAY;
   const bool three_d = I->dim == AGX_DIM_3D;
   const bool combined_array_lod = register_lod && array;
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
   const uint8_t staging_slot = ms_array || combined_array_lod
      ? 1 : register_lod ? 2 : slot;
   /* A sparse read writes one extra register after its masked channels, so
    * agx_emit_texture reserves an eight-channel destination for it (see
    * agx_validate.c's sparse size rule). The encoded result mask stays the
    * four-bit channel mask; the residency component is implicit. */
   if (I->nr_dests != 1 || I->nr_srcs != 6 ||
       !agx3_tex_plain_reg(I->dest[0]) ||
       (I->dest[0].size != AGX_SIZE_16 &&
        I->dest[0].size != AGX_SIZE_32) ||
       agx_channels(I->dest[0]) != (I->sparse ? 8u : 4u) ||
       I->dest[0].value >= 128 ||
       (I->dest[0].size == AGX_SIZE_32 && (I->dest[0].value & 1)) ||
       !agx3_tex_plain_reg(I->src[0]) ||
       I->src[0].size != AGX_SIZE_32 ||
       agx_channels(I->src[0]) !=
          ((I->dim == AGX_DIM_2D_ARRAY || ms_array || ms_2d || three_d)
              ? 3 : 2) ||
       (stage_coordinates ? ((I->src[0].value & 1) ||
         I->src[0].value + 2 * agx_channels(I->src[0]) > 128) :
         ((I->src[0].value & 3) || I->src[0].value > 60)) ||
       ((ms_array || ms_2d)
           ? !agx3_tex_imm(I->src[1], 0)
           : (!(I->lod_mode == AGX_LOD_MODE_AUTO_LOD &&
                agx3_tex_imm(I->src[1], 0)) &&
              !immediate_lod && !register_lod)) ||
       ((ms_array || ms_2d) && I->lod_mode != AGX_LOD_MODE_AUTO_LOD) ||
       (register_lod && !stage_coordinates) ||
       !(agx3_tex_imm(I->src[2], 0) ||
         ((ms_array || ms_2d) && agx3_tex_plain_uniform64(I->src[2])) ||
         bindless_image) ||
       (immediate ? I->src[3].value > 15 :
        (enc != AGX_ISA_ENCODING_AGX3B ||
         (I->src[3].type != AGX_INDEX_REGISTER && I->src[3].type != AGX_INDEX_UNIFORM) ||
         (I->src[3].size != AGX_SIZE_64 &&
          !((ms_array || ms_2d) && I->src[3].size == AGX_SIZE_32)) ||
         agx_channels(I->src[3]) != 1 ||
         (I->src[3].value & 1) || I->src[3].value > 252)) ||
       I->src[3].memory || I->src[3].abs || I->src[3].neg ||
       !agx3_tex_imm(I->src[4], 0) ||
       !agx_is_null(I->src[5]) ||
       (I->dim != AGX_DIM_2D && I->dim != AGX_DIM_2D_ARRAY && !ms_2d &&
        !ms_array && !three_d) ||
       (three_d && enc != AGX_ISA_ENCODING_AGX3B) ||
       /* agx_instr::mask is four bits wide, so the residency component of a
        * sparse result never appears in it; the hardware writes that
        * component implicitly, after the masked channels. */
       (enc == AGX_ISA_ENCODING_AGX3B ? (I->mask == 0 || (I->mask & ~15u))
                                    : (I->mask != 1 && I->mask != 15)) ||
       I->offset || I->shadow ||
       I->query_lod || I->gather != AGX_GATHER_NONE ||
       /* Only AGX3b witnesses the sparse read, and the multisample form has
        * no witness at all. */
       (I->sparse && (enc != AGX_ISA_ENCODING_AGX3B || ms_2d || ms_array)))
      return false;

   /* A read/write image stages its read coordinates at halves16/18, keeping
    * them disjoint from imgwci's halves0/2, and its PACK_SMPIR operands at
    * half 22 (agx3_tex_stage_*_at()), clear of imgwci's data lanes and layer.
    * Native read-only indexed images, like ordinary texture loads, use the
    * bank0 coordinate ABI. */
   const uint16_t staging_base =
      (I->op == AGX_OPCODE_IMAGE_LOAD && !bindless_image && !ms_array) ||
            separate_write_coords ? 16 : 0;
   if (register_lod && slot != 1)
      return false;

   /* A staged read consumes, and kills, its coordinate bank asynchronously.
    * A later read restaging the same bank while that read is outstanding can
    * have its coordinates killed under it. Apple stages consecutive reads in
    * disjoint banks instead: two texture2d::read calls at different
    * coordinates, metal-nt -arch applegpu_g17p, publish halves 0/2 and 4/6
    * and issue smp f19=0 then smp f19=4. A fragment function reading two
    * indexed textures at different coordinates does the same. Callers that
    * pass no `banks` keep one bank. Halves 4..7 carry this path's
    * PACK_SMPIR operands, so the second plain bank starts at half 8. A read
    * with PACK_SMPIR operands keeps the base bank and also occupies halves
    * 0..7; any outstanding read sharing its halves is retired first. */
   const bool plain_bank = !ms_2d && !ms_array && !array && !register_lod;
   const uint16_t bank_span =
      plain_bank ? 2 * agx_channels(I->src[0]) : staging_base + 8;
   uint16_t staging_coord = staging_base;
   uint16_t bank_retire = 0;
   if (stage_coordinates && banks) {
      agx3_staging_release(banks, outstanding);

      /* A read with PACK_SMPIR operands occupies halves 0..bank_span. */
      const uint16_t first = plain_bank ? staging_base : 0;
      bool placed = false;
      for (unsigned c = 0; c < (plain_bank ? 2u : 1u) && !placed; ++c) {
         const uint16_t start = first + 8 * c;
         const uint16_t conflict = agx3_staging_conflicts(
            banks, plain_bank ? start : 0, bank_span);
         if (c == 0)
            bank_retire = conflict;
         if (!conflict) {
            staging_coord = plain_bank ? start : staging_base;
            placed = true;
         }
      }
      if (placed)
         bank_retire = 0;
      else
         agx3_staging_forget(banks, bank_retire);
   }
   if (bank_retire && !agx3_retire_slots_noop(emission, enc, bank_retire))
      return false;
   if (staged_end)
      *staged_end = plain_bank ? staging_coord + bank_span : bank_span;

   /* The plain 2D fetch publishes its per-lane level after the coordinates,
    * so no PACK_SMPIR leads the group to absorb a combined producer mask. */
   const bool plain_2d_lod =
      register_lod && !combined_array_lod && I->dim == AGX_DIM_2D;

   if (stage_coordinates) {
      uint16_t staging_wait = wait_mask & (uint16_t)~bank_retire;
      /* COPY carries one wait bit, while the image coordinates and heap
       * index may have several independent producers. A runtime LOD's
       * PACK_SMPIR can consume their combined mask; an immediate-level
       * read has no such instruction. Retire only those operand producers
       * before publishing its coordinates.
       */
      if (enc == AGX_ISA_ENCODING_AGX3B &&
          ((I->op == AGX_OPCODE_IMAGE_LOAD && !register_lod) ||
           plain_2d_lod) &&
          util_bitcount(staging_wait) > 1) {
         if (!agx3_retire_slots_noop(emission, enc, staging_wait))
            return false;
         staging_wait = 0;
      }
      if (combined_array_lod) {
         uint8_t packed[AGX3_TEX_MAX_LENGTH], packed_len = 0;
         if (!agx3_tex_stage_array_integer_lod_at(
                enc, staging_base, I->src[0].value + 4, I->src[1].value,
                staging_slot, staging_wait, I->src[0].cache,
                I->src[0].discard, I->src[1].cache, I->src[1].discard,
                packed, &packed_len) ||
             !agx3_append(emission, packed, packed_len))
            return false;
         staging_wait = 0;
      } else if (register_lod && !plain_2d_lod) {
         uint8_t lod[AGX3_TEX_MAX_LENGTH], lod_len = 0;
         if (!agx3_tex_stage_typed_lod_at(
                enc, staging_base, I->src[1].value, staging_slot,
                staging_wait, I->src[1].cache, I->src[1].discard,
                AGX3_TEX_LEVEL_INTEGER, lod, &lod_len) ||
             !agx3_append(emission, lod, lod_len))
            return false;
         staging_wait = 0;
      }

      const unsigned plain_coords =
         (ms_2d || ms_array || array) ? 2 : agx_channels(I->src[0]);
      /* The staging COPY carries exactly one wait bit, so a fetch whose
       * operands are produced on two scoreboard slots had no encoding at all
       * and the packer refused the whole shader.
       *
       * Retire the extra slots by repeating the first coordinate copy, one
       * outstanding bit each. Every repeat writes the same staging register
       * from the same source, so the last one -- issued once every slot has
       * retired -- leaves the correct value, whichever slot produced the
       * coordinate. Keep that source alive until its final repeat: marking an
       * earlier COPY as the last use lets the next COPY reread a killed
       * coordinate. A fetch waiting on one slot or none emits exactly what it
       * emitted before, so no shader that already encoded is affected. */
      uint16_t pending_wait = staging_wait;
      for (unsigned i = 0; i < plain_coords; i++) {
       const unsigned reps =
          (i == 0) ? MAX2(util_bitcount(pending_wait), 1u) : 1u;
       for (unsigned rep = 0; rep < reps; rep++) {
         uint16_t wait_bit = 0;
         if (pending_wait) {
            wait_bit = (uint16_t)(pending_wait & (~pending_wait + 1u));
            pending_wait &= (uint16_t)~wait_bit;
         }
         uint8_t copy[AGX3_TEX_MAX_LENGTH], copy_len = 0;
         if (!agx3_tex_stage_copy_hints(enc, staging_coord + 2 * i,
                                  I->src[0].value + 2 * i,
                                  staging_slot, wait_bit,
                                  enc == AGX_ISA_ENCODING_AGX3B && I->src[0].cache,
                                  enc != AGX_ISA_ENCODING_AGX3B ||
                                     (I->src[0].discard && rep + 1 == reps),
                                  copy, &copy_len) ||
             !agx3_append(emission, copy, copy_len))
            return false;
       }
      }

      /* Apple stages a plain 2D fragment texel fetch's level after its
       * coordinates -- metal-nt's `t.read(uint2(uv), uint(lod))` emits the
       * coordinate staging, then PACK_SMPIR, then SMP -- and into this
       * fetch's own operand bank. A level published outside the
       * sampler-private staging allocation read back as zero for every
       * fragment texel fetch on G17P hardware. */
      if (plain_2d_lod) {
         uint8_t lod[AGX3_TEX_MAX_LENGTH], lod_len = 0;
         if (!agx3_tex_stage_typed_lod_bank(
                enc, staging_coord + 6, I->src[1].value, staging_slot,
                staging_wait, I->src[1].cache, I->src[1].discard,
                AGX3_TEX_LEVEL_INTEGER, lod, &lod_len) ||
             !agx3_append(emission, lod, lod_len))
            return false;
         staging_wait = 0;
      }

      /* Native array fetches consume the layer through PACK_SMPIR's bank6
       * operand, independently of the x/y coordinate COPYs. A third ordinary
       * COPY leaves that operand unwritten and reads layer0. The non-MS
       * lowered layer is the low16 bits of the third coordinate word. */
      if (array && !combined_array_lod) {
         uint8_t packed[AGX3_TEX_MAX_LENGTH], packed_len = 0;
         if (!agx3_tex_stage_array_layer_at(
                enc, staging_base, I->src[0].value + 4, staging_slot, 0,
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
      if (ms_array) {
         uint8_t packed[AGX3_TEX_MAX_LENGTH], packed_len = 0;
         /* Keep the packed source alive through both half reads. The final
          * publication consumes its low sample half after the high layer
          * half has been normalized into a separate native GPR word. */
         if (!agx3_tex_pack_ms_array_layer(
                enc, I->src[0].value + 5, ms_array_scratch,
                I->src[0].cache, false, packed, &packed_len) ||
             !agx3_append(emission, packed, packed_len) ||
             !agx3_tex_stage_ms_array_sample(
                enc, I->src[0].value + 4, ms_array_scratch,
                I->src[0].cache, I->src[0].discard, packed, &packed_len) ||
             !agx3_append(emission, packed, packed_len))
            return false;
      }
   }

   if (stage_coordinates && banks &&
       !agx3_staging_hold(banks, plain_bank ? staging_coord : 0, bank_span,
                          slot))
      return false;

   const struct agx3_tex_fetch_desc d = {
      .dest = I->dest[0].value,
      .coord = stage_coordinates ? staging_coord : I->src[0].value,
      .width = I->dest[0].size == AGX_SIZE_32 ? AGX3_TEX_W32 : AGX3_TEX_W16,
      .mask = I->mask,
      .sparse = I->sparse,
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
      .staged_sample = (ms_2d || ms_array) && stage_coordinates,
      .sampler_value = (ms_2d || ms_array) && stage_coordinates ? 1 : 0,
      .three_d = three_d,
      .lod_x64 = immediate_lod ? I->src[1].value * 64 : 0,
      .register_lod = register_lod,
   };
   uint8_t bytes[AGX3_TEX_MAX_LENGTH];
   uint8_t len = 0;
   /* Apple's fragment texel fetches enter the pixel transaction after all
    * operand staging and immediately before SMP, exactly as its implicit
    * derivative samples do.
    */
   if (pixel_entry) {
      uint8_t entry[AGX3_PIXEL_MAX_LENGTH], entry_len = 0;
      if (pixel_entry_offset == NULL ||
          !agx3_pixel_colour_entry(enc, entry, &entry_len))
         return false;

      *pixel_entry_offset = emission->size;
      if (!agx3_append(emission, entry, entry_len))
         return false;
   }

   return agx3_tex_encode_fetch(enc, &d, bytes, &len) &&
          agx3_append(emission, bytes, len);
}

/* Two bounded public storage-image forms: integer 2D state0 fed by an image
 * load or atomic result, and normalized formatted-float state1 fed by one
 * explicit-LOD sample. The native imgwci operation is the coherent image
 * write form; NIR's coherent bit does not select a different instruction.
 * The PBE descriptor selects RGBA8 versus RGB10A2.
 * The four slot1 COPYs route the packed vec4 into the exact imgwci data lanes.
 * The first COPY consumes the one-shot texture dependency when it is still
 * outstanding. An intervening bounds consumer may already have retired that
 * producer; either way native imgwci waits for the new staging-slot1 writes.
 * Validation happens before the first append, so a refusal leaves the output
 * transaction untouched. */
static bool
agx3_emit_storage_image_write(struct util_dynarray *emission,
                              enum agx_isa_encoding enc, const agx_instr *I,
                              uint16_t wait_mask,
                              struct agx3_staging_banks *banks,
                              uint16_t bank_floor, uint16_t *outstanding)
{
   const bool normalized = I->format == AGX_FORMAT_U8NORM ||
                           I->format == AGX_FORMAT_RGB10A2;
   const bool immediate_state = agx3_tex_imm(I->src[3], 0) &&
      I->src[4].type == AGX_INDEX_IMMEDIATE &&
      !I->src[4].memory && !I->src[4].abs && !I->src[4].neg &&
      ((I->src[4].value == 0 && I->format == AGX_FORMAT_I8) ||
       (I->src[4].value == 1 &&
        (normalized || I->format == AGX_FORMAT_I8)));
   /* A bindless store names its descriptor set's uniform halves as the PBE
    * heap (f79) and carries the PBE record index in a GPR (f92).
    * hk_g17p_lower_storage_records() converts the byte offset to that index.
    * The heap and the index are independent operands: metal-nt -arch
    * applegpu_g17p writing texture(1) of two direct uint write textures emits
    * d74054000040000100040000881a0900, f79 0 with f91 1.
    *
    * The state operand's form and f70 are ORTHOGONAL, so an indexed state
    * does not imply normalized data.  metal-nt -arch applegpu_g17p, indexed
    * `array<texture2d<T, access::write>, 4>` (a uniform index, f93), one
    * variable per kernel:
    *
    *   texture2d<float>  d72054000040001020040000883a0901
    *   texture2d<uint>   d72054000040001020040000881a0901
    *   texture2d<int>    d72054000040001020040000881a0901
    *
    * against the immediate-state controls that are already recorded in
    * agx_compile.c (`...0000 0400 0088 3a09 00` / `...881a0900`): the state
    * operand moves and f70 does not, and int and uint stay byte-identical. */
   const bool heap_state =
      agx3_tex_plain_uniform64(I->src[3]) && !(I->src[3].value & 3) &&
      agx3_tex_plain_reg(I->src[4]) &&
      I->src[4].size == AGX_SIZE_32 && agx_channels(I->src[4]) == 1 &&
      !(I->src[4].value & 1) && I->src[4].value <= 126 &&
      !(I->src[4].cache && (I->src[4].discard || I->src[4].kill));
   /* Cube stores reach the packer as 2D arrays with the face folded into the
    * layer, and 1D stores as 2D. */
   const bool array = I->dim == AGX_DIM_2D_ARRAY;
   const bool three_d = I->dim == AGX_DIM_3D;
   const unsigned coords = (array || three_d) ? 3 : 2;
   if (enc != AGX_ISA_ENCODING_AGX3B || I->nr_dests != 0 ||
       I->nr_srcs != 5 ||
       (I->dim != AGX_DIM_2D && !array && !three_d) ||
       (wait_mask & ~UINT16_C(0x3f)) != 0 ||
       !agx3_tex_plain_reg(I->src[0]) || I->src[0].size != AGX_SIZE_32 ||
       agx_channels(I->src[0]) != 4 || (I->src[0].value & 1) ||
       I->src[0].value > 248 || !agx3_tex_plain_reg(I->src[1]) ||
       I->src[1].size != AGX_SIZE_32 || agx_channels(I->src[1]) != coords ||
       (I->src[1].value & 1) || I->src[1].value + 2 * (coords - 1) > 254 ||
       !agx3_tex_imm(I->src[2], 0) ||
       (!immediate_state && !heap_state)) {
      if (agx_compiler_debug & (AGX_DBG_SHADERS | AGX_DBG_REFUSE))
         fprintf(stderr, "G17P imgwci shape refusal: wait=0x%x data=%u/%u/%u coord=%u/%u/%u dim=%u coherent=%u\n",
                 wait_mask, I->src[0].value, I->src[0].size,
                 agx_channels(I->src[0]), I->src[1].value, I->src[1].size,
                 agx_channels(I->src[1]), I->dim, I->coherent);
      return false;
   }

   /* imgwci consumes its staging bank asynchronously and signals its f12
    * slot when done, like any producer. A later COPY into a bank whose write
    * is still in flight can replace that write's coordinates or data first,
    * so a bank stays held until its write's slot is synchronised.
    *
    * metal-nt -arch applegpu_g17p, writes to array<texture2d<int,
    * access::write>, N> with no wait between them: six conditional scalar
    * writes stage f5 0/8/16/24/32/40 on slots 1..6, three unconditional RGBA
    * writes f5 0/16/32, and two computed-index writes f5 0 on slot 1 and f5
    * 16 on slot 2. After a texture read, two conditional writes stage f5 16
    * and 24, clear of the read's bank. Forty unconditional writes reuse a
    * bank only once its imgwci's slot has been waited.
    *
    * The resolved-PBE forms below keep their witnessed bank 0 and slot 1. */
   const unsigned bank_span = 16;
   uint16_t bank = 0;
   uint8_t slot = 1;
   if (banks) {
      agx3_staging_release(banks, *outstanding);

      uint16_t retire = 0;
      bool placed = false;
      for (unsigned start = heap_state ? bank_floor : 0;
           start + bank_span <= AGX3_STAGED_WRITE_LIMIT && !placed &&
           (heap_state || start == 0);
           start += 8) {
         if (!agx3_staging_conflicts(banks, start, bank_span)) {
            bank = start;
            placed = true;
         }
      }
      if (!placed) {
         bank = heap_state ? bank_floor : 0;
         retire |= agx3_staging_conflicts(banks, bank, bank_span);
      }

      /* Stage on a slot no unrelated producer holds, rotating so writes in
       * flight together use different slots. */
      const uint16_t busy = *outstanding & (uint16_t)~(wait_mask | retire);
      if (heap_state) {
         slot = 0;
         for (unsigned i = 0; i < 6 && !slot; ++i) {
            const uint8_t s = (uint8_t)((banks->next_write_slot + i) % 6 + 1);
            if (!(busy & (1u << (s - 1))))
               slot = s;
         }
         if (!slot) {
            slot = (uint8_t)(banks->next_write_slot % 6 + 1);
            retire |= 1u << (slot - 1);
         }
         banks->next_write_slot = slot % 6;
      } else if (busy & 1u) {
         retire |= 1u;
      }

      if (retire) {
         if (!agx3_retire_slots_noop(emission, enc, retire))
            return false;
         *outstanding &= (uint16_t)~retire;
         agx3_staging_forget(banks, retire);
      }
      /* The write's own slot also names the bank it now holds. */
      agx3_staging_forget(banks, 1u << (slot - 1));
      if (!agx3_staging_hold(banks, bank, bank_span, slot))
         return false;
      *outstanding |= 1u << (slot - 1);
   }

   struct util_dynarray staged;
   util_dynarray_init(&staged, NULL);
   uint8_t coord_bytes[AGX3_STAGE_MAX_LENGTH], coord_len = 0;
   uint8_t layer_bytes[AGX3_TEX_MAX_LENGTH], layer_len = 0;
   /* A 3D write stages z at half 4 and an array write packs its layer at
    * half 6, moving the data lanes out of the way: metal-nt -arch
    * applegpu_g17p COPYs texture3d data to halves 8/10/12/14 and
    * texture2d_array data to halves 4/10/12/14. A bank offsets every half:
    * the second of two texture2d_array writes stages data at 20/26/28/30
    * with f5=16. */
   const uint16_t data_dest[4] = {
      bank + (three_d ? 8 : 4), bank + (array || three_d ? 10 : 6),
      bank + 12, bank + 14,
   };
   /* imgwci consumes integer coordinates from the first two halves of its
    * bank. Native texture/image kernels publish ordinary GPR coordinates
    * with COPY on the same slot the instruction waits for. Keeping the source
    * retained also matches the native write kernel, whose address
    * calculation may reuse it.
    *
    * The scoreboard wait belongs on the FIRST staged COPY of the group, not
    * on the first data COPY: the coordinate copies are emitted ahead of the
    * data copies, so a wait carried by the data copy runs too late to protect
    * the coordinate registers.  A coordinate produced by an outstanding
    * async producer -- `imageStore(img, pc.coords.xy, ...)` compiles to
    * `device_load ... slot 0` feeding image_write's src[1] directly -- was
    * then copied into the staging bank before that load retired, so every
    * store landed at the stale register contents (texel 0,0) regardless of
    * the requested coordinate.  Every other staged producer in this file
    * already waits on its first COPY: the texture-sample coordinate pair
    * (agx3_emit_texture_sample), the ZS_EMIT pair (AGX_OPCODE_ZS_EMIT_AGX3),
    * the UVSW transfer (agx3_emit_uvsw) and SAMPLE_MASK_AGX3.  Waiting here
    * is also strictly conservative for the data copies, which follow. */
   if (!agx3_stage_copy_source(enc, bank, I->src[1].value, false, slot,
                               wait_mask, coord_bytes, &coord_len) ||
       !agx3_append(&staged, coord_bytes, coord_len) ||
       !agx3_stage_copy_source(enc, bank + 2, I->src[1].value + 2, false,
                               slot, 0, coord_bytes, &coord_len) ||
       !agx3_append(&staged, coord_bytes, coord_len) ||
       (three_d &&
        (!agx3_stage_copy_source(enc, bank + 4, I->src[1].value + 4, false,
                                 slot, 0, coord_bytes, &coord_len) ||
         !agx3_append(&staged, coord_bytes, coord_len))) ||
       (array &&
        (!agx3_tex_stage_image_write_layer(enc, I->src[1].value + 4,
                                           bank + 6, slot, 0, false, false,
                                           layer_bytes, &layer_len) ||
         !agx3_append(&staged, layer_bytes, layer_len)))) {
      if (agx_compiler_debug & (AGX_DBG_SHADERS | AGX_DBG_REFUSE))
         fprintf(stderr, "G17P imgwci coordinate compose refusal\n");
      goto fail;
   }

   for (unsigned c = 0; c < 4; ++c) {
      uint8_t bytes[AGX3_STAGE_MAX_LENGTH], len = 0;
      if (!agx3_stage_copy_source(enc, data_dest[c],
                                  I->src[0].value + 2 * c,
                                  I->src[0].discard, slot, 0,
                                  bytes, &len) ||
          !agx3_append(&staged, bytes, len)) {
         if (agx_compiler_debug & (AGX_DBG_SHADERS | AGX_DBG_REFUSE))
            fprintf(stderr, "G17P imgwci data compose refusal: component=%u\n", c);
         goto fail;
      }
   }

   uint8_t imgwci[AGX3_IMAGE_WRITE_LENGTH], imgwci_len = 0;
   const struct agx3_image_write_desc d = {
      .wait_mask = wait_mask,
      .staging_bank = bank,
      .result_slot = slot,
      .image_kind = heap_state ? AGX3_TEX_OPERAND_REGISTER :
                                 AGX3_TEX_OPERAND_IMMEDIATE,
      .image_value = I->src[4].value,
      .image_kill = heap_state && (I->src[4].discard || I->src[4].kill),
      .normalized = normalized,
      .dim = array     ? AGX3_IMAGE_WRITE_2D_ARRAY
             : three_d ? AGX3_IMAGE_WRITE_3D
                       : AGX3_IMAGE_WRITE_2D,
   };
   bool encoded;
   if (heap_state) {
      /* Consume this write's COPYs, as the resolved PBE form below does. */
      encoded = agx3_tex_encode_image_write_2d(
         enc, &d, I->src[3].value, slot, imgwci, &imgwci_len);
   } else if (immediate_state && I->src[4].value == 1 && !normalized) {
      /* Hk's mixed indexed-read/R32-write table contains texture and sampler
       * heap pointers followed by one resolved PBE pointer at uniform half8.
       * Apple's read/write witness instead used a PBE heap at half4 with
       * descriptor index1. Copying that pair selected our sampler heap.
       * Both staging slots passed the real-data hardware gate; retain the
       * existing, shorter slot1 COPYs and consume that same slot here. */
      struct agx3_image_write_desc resolved = d;
      resolved.image_value = 0;
      encoded = agx3_tex_encode_image_write_2d(
         enc, &resolved, 8, 1, imgwci, &imgwci_len);
   } else {
      encoded = agx3_tex_encode_image_write_rgba8_2d(
         enc, &d, imgwci, &imgwci_len);
   }
   if (!encoded || !agx3_append(&staged, imgwci, imgwci_len) ||
       !agx3_append(emission, staged.data, staged.size)) {
      if (agx_compiler_debug & (AGX_DBG_SHADERS | AGX_DBG_REFUSE))
         fprintf(stderr, "G17P imgwci instruction compose refusal\n");
      goto fail;
   }
   util_dynarray_fini(&staged);
   return true;

fail:
   util_dynarray_fini(&staged);
   return false;
}

/* A 3D sample with a bias, a minimum LOD and a runtime offset. */
static bool
agx3_is_staged_3d_bias_min(const agx_instr *I)
{
   return I->op == AGX_OPCODE_TEXTURE_SAMPLE && I->dim == AGX_DIM_3D &&
          I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS_MIN && I->offset &&
          !I->offset_xyz;
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
                         uint8_t slot, uint16_t wait_mask, bool final,
                         bool sample_group_nonterminal, bool sample_group_end,
                         bool pixel_entry, uint32_t *pixel_entry_offset,
                         uint8_t native_coord_slot,
                         const uint16_t cem_scratch[3], bool vertex_stage,
                         bool compute_stage, uint16_t array_scratch,
                         uint16_t bank_base)
{
   const bool explicit_lod = I->lod_mode == AGX_LOD_MODE_LOD_MIN;
   const bool bias_min = I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS_MIN;
   const bool gradient_min = I->lod_mode == AGX_LOD_MODE_LOD_GRAD_MIN;
   const bool min_lod = bias_min || gradient_min;
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
   const bool gradient_lod = I->lod_mode == AGX_LOD_MODE_LOD_GRAD || gradient_min;
   /* Gather uses the implicit-LOD selector for 2D, array and cube forms,
    * including compute where it reads level zero. Map the IR component
    * values 1/3/5/7 explicitly to the native two-bit selector. */
   const bool gather = I->gather != AGX_GATHER_NONE;
   const bool shadow = I->shadow;
   const bool query_lod = I->query_lod;
   const bool offset = I->offset;
   enum agx3_tex_gather gather_component = AGX3_TEX_GATHER_NONE;
   switch (I->gather) {
   case AGX_GATHER_NONE: break;
   case AGX_GATHER_R: gather_component = AGX3_TEX_GATHER_R; break;
   case AGX_GATHER_G: gather_component = AGX3_TEX_GATHER_G; break;
   case AGX_GATHER_B: gather_component = AGX3_TEX_GATHER_B; break;
   case AGX_GATHER_A: gather_component = AGX3_TEX_GATHER_A; break;
   default: return agx3_tex_decline(I, __LINE__);
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
   const bool cube_array = I->dim == AGX_DIM_CUBE_ARRAY;
   const bool array = I->dim == AGX_DIM_2D_ARRAY || cube_array;
   const bool three_d = I->dim == AGX_DIM_3D;
   const bool cube = I->dim == AGX_DIM_CUBE || cube_array;
   /* A vertex stage has no derivatives, so a GLSL `texture()` depth
    * comparison arrives as an explicit level zero. The cube form of that is
    * already grounded (KHR-GL30.ext_texture_shadow_lod
    * texturelod.samplercubeshadow_vertex), and the cube array adds only its
    * layer, staged through the same witnessed face+layer word the implicit
    * cube-array comparison uses (texture.samplercubearrayshadow_fragment).
    * Excluding the cube array refuses texture.samplercubearrayshadow_vertex.
    */
   const bool shadow_level_zero =
      shadow && valid_immediate_lod && immediate_level == 0.0f;
   /* A per-lane level and a per-lane bias share one staging form, so they
    * share one operand test: a live single-channel 16-bit register. */
   const bool scalar_lod_reg = agx3_tex_plain_reg(I->src[1]) &&
      I->src[1].size == AGX_SIZE_16 && agx_channels(I->src[1]) == 1;
   const bool bias_min_operand = agx3_tex_plain_reg(I->src[1]) &&
      I->src[1].size == AGX_SIZE_32 && agx_channels(I->src[1]) == 1 &&
      (I->src[1].value & 1) == 0 && I->src[1].value < 127;
   /* A 2D gradient is four 32-bit scalars in one aligned run, which the four
    * staging copies name two halves apart. */
   /* A 2D gradient is four 32-bit scalars in one aligned run; a 3D one is
    * six, and agx_emit_tex()'s collect already lays both out in the
    * interleaved order the staging banks want. */
   const unsigned gradient_channels = three_d ? 6u : 4u;
   const bool gradient_operand = agx3_tex_plain_reg(I->src[1]) &&
      I->src[1].size == AGX_SIZE_32 &&
      agx_channels(I->src[1]) == gradient_channels + gradient_min &&
      (I->src[1].value & 1) == 0 &&
      I->src[1].value + (2 * (gradient_channels + gradient_min)) - 2 < 128;
   const bool valid_lod = bias_min ?
      (enc == AGX_ISA_ENCODING_AGX3B && bias_min_operand) : explicit_lod ?
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
   const bool texture_immediate =
      I->nr_srcs == 6 && I->src[3].type == AGX_INDEX_IMMEDIATE;
   const bool sampler_immediate =
      I->nr_srcs == 6 && I->src[4].type == AGX_INDEX_IMMEDIATE;
   const bool immediate = texture_immediate && sampler_immediate;
   /* agx_emit_tex() publishes the packed offset on the instruction rather
    * than through src[5], because a comparison sample needs that operand for
    * its reference value. src[5] still carries the immediate when there is no
    * comparator, and the two must agree. */
   const bool immediate_offset = offset && I->offset_xyz > 0 &&
      (shadow || (I->src[5].type == AGX_INDEX_IMMEDIATE &&
                  I->src[5].value == I->offset_xyz &&
                  !I->src[5].memory && !I->src[5].abs && !I->src[5].neg &&
                  !I->src[5].cache && !I->src[5].discard && !I->src[5].kill));
   const bool register_offset = offset && !I->offset_xyz && gather &&
      agx3_tex_plain_reg(I->src[5]) && I->src[5].size == AGX_SIZE_32 &&
      agx_channels(I->src[5]) == (shadow ? 2 : 1) &&
      !(I->src[5].value & 1) && I->src[5].value + (shadow ? 2 : 0) <= 126;
   /* Runtime offsets are contiguous 32-bit register components. A comparison,
    * if present, precedes the 2D pair in the collected operand. A runtime
    * 2D offset combined with register LOD/bias arrives as one prepacked word
    * instead, because those operands share sampler bank 6. */
   const unsigned offset_components = 2;
   const unsigned staged_offset_components =
      offset_components + (shadow ? 1 : 0);
   const bool staged_component_offset = offset && !gather && !three_d && !array &&
      I->offset_xyz == 0 &&
      agx3_tex_plain_reg(I->src[5]) && I->src[5].size == AGX_SIZE_32 &&
      agx_channels(I->src[5]) == staged_offset_components &&
      (I->src[5].value & 1) == 0 &&
      I->src[5].value + 2 * staged_offset_components <= 256;
   const unsigned packed_offset_components = 1 + (shadow ? 1 : 0);
   const bool staged_packed_offset = offset && !gather && I->offset_xyz == 0 &&
      (array || explicit_lod || register_bias || min_lod) &&
      agx3_tex_plain_reg(I->src[5]) && I->src[5].size == AGX_SIZE_32 &&
      agx_channels(I->src[5]) == packed_offset_components &&
      (I->src[5].value & 1) == 0 &&
      I->src[5].value + 2 * packed_offset_components <= 128;
   const bool staged_offset =
      staged_component_offset || staged_packed_offset;
   /* AGX3B coordinates are copied to the fixed staging bank before SMP. COPY
    * and CEM each name a seven-bit source half-register, so the entire vector
    * must end below half 128. AGX3C retains its independently recovered direct
    * coordinate bound. */
   const unsigned max_coord = enc == AGX_ISA_ENCODING_AGX3B
                                 ? 128 - (2 * agx_channels(I->src[0]))
                                 : 60;
   if (I->nr_dests != 1 || I->nr_srcs != 6 ||
       !agx3_tex_plain_reg(I->dest[0]) ||
       (I->dest[0].size != AGX_SIZE_16 &&
        I->dest[0].size != AGX_SIZE_32) ||
       /* Sparse residency widens only the destination reservation: the extra
        * register the instruction writes after its masked channels. */
       agx_channels(I->dest[0]) != (I->sparse ? 8u : 4u) ||
       I->dest[0].value >= 128 ||
       (I->dest[0].size == AGX_SIZE_32 && (I->dest[0].value & 1)) ||
       !agx3_tex_plain_reg(I->src[0]) ||
       I->src[0].size != AGX_SIZE_32 ||
       agx_channels(I->src[0]) != (query_lod ? (three_d || cube ? 3 : 2) :
                                    cube_array ? 4 :
                                    (array || three_d || cube) ? 3 : 2) ||
       (I->src[0].value & 3) || I->src[0].value > max_coord ||
       !valid_lod || !agx3_tex_imm(I->src[2], 0) ||
       (texture_immediate
           ? I->src[3].value > 15
           : ((I->src[3].type != AGX_INDEX_REGISTER &&
               I->src[3].type != AGX_INDEX_UNIFORM) ||
              I->src[3].size != AGX_SIZE_64 ||
              agx_channels(I->src[3]) != 1 || (I->src[3].value & 1))) ||
       (sampler_immediate
           ? I->src[4].value > 15
           : ((I->src[4].type != AGX_INDEX_REGISTER &&
               I->src[4].type != AGX_INDEX_UNIFORM) ||
              I->src[4].size != AGX_SIZE_64 ||
              agx_channels(I->src[4]) != 1 || (I->src[4].value & 1))) ||
       I->src[3].memory || I->src[3].abs || I->src[3].neg ||
       I->src[4].memory || I->src[4].abs || I->src[4].neg ||
       (shadow
           ? (register_offset
                 ? false
                 : staged_offset
                  ? false
                  : (!agx3_tex_plain_reg(I->src[5]) ||
                     I->src[5].size != AGX_SIZE_32 ||
                     agx_channels(I->src[5]) != 1 || I->src[5].value > 126 ||
                     (offset && !immediate_offset)))
           : offset ? (!immediate_offset && !register_offset && !staged_offset)
                     : !agx_is_null(I->src[5])) ||
       (I->dim != AGX_DIM_2D && I->dim != AGX_DIM_2D_ARRAY && !three_d &&
        !cube) ||
       (array && enc != AGX_ISA_ENCODING_AGX3B) ||
       /* f76 = 3 is orthogonal to f139; only the gradient is held back,
        * because gradient3d stages six scalars and that layout has no
        * witness. See agx3_tex_core.c for the five-row table. */
       (three_d &&
        (enc != AGX_ISA_ENCODING_AGX3B ||
         (!valid_immediate_lod && !implicit_lod && !biased_lod &&
          !register_bias && !explicit_lod && !gradient_lod && !bias_min))) ||
       (cube &&
        (enc != AGX_ISA_ENCODING_AGX3B ||
         (!implicit_lod && !explicit_lod && !biased_lod &&
          !register_bias && !bias_min))) ||
       (I->mask == 0 || (I->mask & ~15u)) ||
       /* metal-nt witnesses the sparse sample on AGX3b for the implicit,
        * immediate level, immediate bias, array, 3D, cube and gather shapes.
        * A depth comparison and a LOD query have no witness. */
       (I->sparse && (enc != AGX_ISA_ENCODING_AGX3B || shadow || query_lod)) ||
       /* A minimum LOD and a runtime offset share bank 6 only through the
        * packed word. */
       (min_lod && ((cube && !bias_min) || vertex_stage ||
                    (three_d && offset && !bias_min && !immediate_offset) ||
                    staged_component_offset)) ||
       (query_lod &&
        (enc != AGX_ISA_ENCODING_AGX3B || !implicit_lod || shadow || offset ||
         gather || I->dest[0].size != AGX_SIZE_32 || (I->mask & ~3u))) ||
       /* Immediate 3D offsets carry their third signed nibble in f43.
        * Dynamic 3D offsets have already lowered to coordinate addition;
        * cubes have no offset form. */
       (offset && (enc != AGX_ISA_ENCODING_AGX3B || cube)) ||
       /* A depth comparison is orthogonal to the LOD shape on a plain 2D
        * sample. The array and cube comparisons use their own packed layer /
        * CEM operands, with implicit LOD or the witnessed explicit zero. */
       (shadow && (three_d || enc != AGX_ISA_ENCODING_AGX3B ||
                   I->mask != (gather ? 15u : 1u) ||
                   ((cube || (array && !offset && !vertex_stage)) &&
                    !implicit_lod && !shadow_level_zero &&
                    !(array && !cube && min_lod) &&
                    /* A comparison stages a register level or bias beside
                     * the operand its dimension already owns: the noncube
                     * array with its layer (agx3_tex_stage_array_lod), the
                     * cube with CEM's face, the cube array with both. All
                     * three are KHR-GL30.ext_texture_shadow_lod --
                     * texturelod and bias on sampler2DArrayShadow and on
                     * samplerCubeArrayShadow. Admitting only the cube
                     * refuses the array fragments, which then fail to link
                     * while every offline gate stays green. A bias with a
                     * minimum is witnessed for the cube only. A noncube
                     * array comparison gradient stages its four scalars in
                     * banks 8..14 and keeps the layer on PACK_SMPIR bank 6
                     * and the reference at bank 4, so nothing it publishes
                     * collides; the same shape in a pre-raster stage, which
                     * this term never covered, already reached the encoder. */
                    !((explicit_lod && !immediate_lod) || register_bias ||
                      (cube && bias_min) ||
                      (array && !cube && gradient_lod))))) ||
       (gather && (I->lod_mode != AGX_LOD_MODE_AUTO_LOD || three_d ||
                   enc != AGX_ISA_ENCODING_AGX3B)))
      return agx3_tex_decline(I, __LINE__);

   /* The caller lowers additional required dependencies before staging. */
   if (wait_mask & (wait_mask - 1))
      return agx3_tex_decline(I, __LINE__);

   const bool register_operands = !immediate &&
      (I->src[3].type == AGX_INDEX_REGISTER ||
       I->src[4].type == AGX_INDEX_REGISTER);

   /* A cube register level or bias is published with CEM's face, after CEM
    * has run, so CEM must not overwrite it. */
   const bool cube_register_lod =
      cube && ((explicit_lod && !immediate_lod) || register_bias || bias_min);

   /* The immediate cube corpus uses fixed r0/r7/r2 CEM destinations. Dynamic
    * descriptor forms, and any cube whose staged LOD must survive CEM, use
    * the allocator-owned bank supplied below. */
   const bool fixed_cem = immediate && !cube_register_lod;
   if (cube && fixed_cem) {
      const unsigned coord_first = I->src[0].value;
      const unsigned coord_last = coord_first + (cube_array && !query_lod ? 8 : 6);
      if (coord_first < 2 || (coord_first < 16 && coord_last > 14))
         return agx3_tex_decline(I, __LINE__);
   } else if (cube &&
              (cem_scratch == NULL || (cem_scratch[0] & 1) ||
               cem_scratch[2] > 190)) {
      return agx3_tex_decline(I, __LINE__);
   }

   /* Native register-indexed array and cube samples publish their coordinate /
    * face operand bank on slot 6 and wait slot 6 in SMP, independently of the
    * sample result slot. The dynamic cube witness groups both descriptor
    * loads and face publication on that transaction. Uniform/immediate
    * operands use the ordinary result slot. */
   const bool native_staged_coords = native_coord_slot != 0;
   const uint8_t stage_slot = native_staged_coords
                                 ? native_coord_slot
                                 : (array || cube) && register_operands ? 6
                                                                        : slot;
   /* Every native G17P witness using AGX3B names coordinate bank 0 in SMP,
    * including ordinary implicit 2D samples whose materialized coordinates
    * live in unrelated GPRs.  For example, the attachment-only f_retain_coords
    * witness copies r4/r5 to staging[0]/[2], reports ShaderInfo D=8, and
    * samples bank 0.  The staging address is an ABI location, not the source
    * GPR number.  AGX3c retains its independently decoded coordinate field. */
   /* Sampler-private staging groups. SMP consumes its staged coordinates
    * and comparison reference asynchronously, after issue: a later sample
    * that restages the same bank before the earlier one retires replays the
    * newest operands into every older sample. Apple's own three-reference
    * depth2d fragment (native-quick-textures/2d-three-references) therefore
    * copies consecutive samples' coordinates into banks 0, 8 and 16 and
    * names that bank in each SMP's f19; the comparison reference follows at
    * bank + 4. The stream packer chooses the group from the in-flight
    * samples; only the plain 2D coordinate ABI relocates, every other shape
    * keeps its witnessed fixed banks (gradients 8..18, PACK_SMPIR bank 6,
    * CEM scratch). */
   if (bank_base != 0 &&
       (enc != AGX_ISA_ENCODING_AGX3B || native_staged_coords || array ||
        three_d || cube || gradient_lod || bias_min || gradient_min ||
        query_lod || (explicit_lod && !immediate_lod) || register_bias ||
        register_offset || (bank_base & 7) || bank_base > 16))
      return agx3_tex_decline(I, __LINE__);
   const uint16_t coord = (enc == AGX_ISA_ENCODING_AGX3B ||
                           native_staged_coords || explicit_lod ||
                           register_bias || array || three_d || cube ||
                           gradient_lod || shadow)
                             ? bank_base
                             : I->src[0].value;
   uint8_t bytes[AGX3_TEX_MAX_LENGTH];
   uint8_t len = 0;

   const bool combined_array_lod = array && !cube && !staged_offset &&
      ((explicit_lod && !immediate_lod) || register_bias || min_lod);
   const uint16_t offset_base = I->src[5].value + (shadow ? 2 : 0);
   /* The comparison 2D-array bias and the staged 3D bias read their minimum
    * from a packed register. */
   const bool register_minimum = bias_min &&
      ((array && shadow) || (three_d && staged_packed_offset));
   /* A runtime offset arrives as one NIR-packed bank-6 word that already
    * carries the layer and the bias, or the minimum: a gradient's, or a
    * non-array bias's. Only the 2D-array bias still stages its minimum. */
   const bool packed_array_word = array && staged_packed_offset;
   if (bias_min && array && !cube) {
      /* The minimum moves out of bank 6 so the bias can join the layer
       * there: to half 4 as Apple publishes it, or for a comparison, whose
       * reference owns half 4, to a packed register SMP names. The register
       * pack carries no wait, so it follows the staging that does. The bias
       * and minimum halves belong to one register, so only its last read
       * may discard it. */
      const bool minimum_last = shadow || packed_array_word;
      if (!shadow &&
          (!agx3_tex_stage_array_minimum_lod(
              enc, I->src[1].value + 1, stage_slot, wait_mask,
              I->src[1].cache, minimum_last && I->src[1].discard, bytes,
              &len) ||
           !agx3_append(emission, bytes, len)))
         return agx3_tex_decline(I, __LINE__);
      const bool bank6 = packed_array_word
         ? agx3_tex_stage_copy_hints(
              enc, 6, offset_base, stage_slot, shadow ? wait_mask : 0,
              I->src[5].cache, I->src[5].discard, bytes, &len)
         : agx3_tex_stage_array_lod(
              enc, I->src[0].value + 4, I->src[1].value, stage_slot,
              shadow ? wait_mask : 0, I->src[0].cache, I->src[0].discard,
              I->src[1].cache, !minimum_last && I->src[1].discard, bytes,
              &len);
      if (!bank6 || !agx3_append(emission, bytes, len))
         return agx3_tex_decline(I, __LINE__);
      if (shadow &&
          (!agx3_tex_pack_minimum(enc, I->src[1].value + 1, array_scratch,
                                  I->src[1].cache, I->src[1].discard, bytes,
                                  &len) ||
           !agx3_append(emission, bytes, len)))
         return agx3_tex_decline(I, __LINE__);
      wait_mask = 0;
   } else if (gradient_min && array && !cube) {
      if (!(packed_array_word
               ? agx3_tex_stage_copy_hints(
                    enc, 6, offset_base, stage_slot, wait_mask,
                    I->src[5].cache, I->src[5].discard, bytes, &len)
               : agx3_tex_stage_array_minimum_gradient(
                    enc, I->src[0].value + 4,
                    I->src[1].value + 2 * gradient_channels, stage_slot,
                    wait_mask, I->src[0].cache, I->src[0].discard,
                    I->src[1].cache, I->src[1].discard, bytes, &len)) ||
          !agx3_append(emission, bytes, len))
         return agx3_tex_decline(I, __LINE__);
      wait_mask = 0;
   } else if (bias_min && three_d && staged_packed_offset) {
      /* A 3D word carries the Z nibble where a 2D one carries the minimum,
       * so the minimum goes to the packed register SMP names. The register
       * pack carries no wait, so it follows the staging that does. */
      if (!agx3_tex_stage_copy_hints(
             enc, 6, offset_base, stage_slot, wait_mask, I->src[5].cache,
             I->src[5].discard, bytes, &len) ||
          !agx3_append(emission, bytes, len) ||
          !agx3_tex_pack_minimum(enc, I->src[1].value + 1, array_scratch,
                                 I->src[1].cache, I->src[1].discard, bytes,
                                 &len) ||
          !agx3_append(emission, bytes, len))
         return agx3_tex_decline(I, __LINE__);
      wait_mask = 0;
   } else if (bias_min && !cube && !staged_packed_offset) {
      if (!agx3_tex_stage_bias_min(
             enc, I->src[1].value, I->src[1].value + 1, stage_slot, wait_mask,
             I->src[1].cache, I->src[1].discard, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         return agx3_tex_decline(I, __LINE__);
      wait_mask = 0;
   } else if (gradient_min && !staged_packed_offset) {
      if (!agx3_tex_stage_typed_lod(
             enc, I->src[1].value + 2 * gradient_channels, stage_slot,
             wait_mask, I->src[1].cache, I->src[1].discard,
             AGX3_TEX_LEVEL_MINIMUM, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         return agx3_tex_decline(I, __LINE__);
      wait_mask = 0;
   } else if (staged_packed_offset) {
      if (!agx3_tex_stage_copy_hints(
             enc, 6, offset_base, stage_slot, wait_mask,
             I->src[5].cache, I->src[5].discard, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         return agx3_tex_decline(I, __LINE__);
      wait_mask = 0;
   } else if (combined_array_lod) {
      if (!agx3_tex_stage_array_lod(
             enc, I->src[0].value + 4, I->src[1].value, stage_slot, wait_mask,
             I->src[0].cache, I->src[0].discard,
             I->src[1].cache, I->src[1].discard, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         return agx3_tex_decline(I, __LINE__);
      wait_mask = 0;
   } else if (((explicit_lod && !immediate_lod) || register_bias) &&
              !cube_register_lod) {
      if (!agx3_tex_stage_lod(enc, I->src[1].value, stage_slot, wait_mask,
                             I->src[1].cache, I->src[1].discard,
                             bytes, &len) ||
          !agx3_append(emission, bytes, len))
         return agx3_tex_decline(I, __LINE__);
      wait_mask = 0;
   }

   /* CEM uses fixed r0/r2/r7 scratch registers. Publish the comparison first
    * so a perfectly valid allocator choice in one of those registers cannot
    * be clobbered by the cube-coordinate calculation. Apple's cube and
    * cube-array comparison shaders likewise publish the reference before the
    * final CEM axes. */
   if (shadow && cube &&
       (!agx3_tex_stage_copy_hints(
           enc, 4, I->src[5].value, stage_slot, 0, I->src[5].cache,
           I->src[5].discard, bytes, &len) ||
        !agx3_append(emission, bytes, len)))
      return agx3_tex_decline(I, __LINE__);

   if (cube) {
      /* Native cube sampling consumes CEM's normalized coordinates and face,
       * not the raw XYZ vector. CEM has no wait fields. A register-descriptor
       * witness may leave only slot 6 active: the finalizers/face producer
       * join that transaction and SMP waits for the complete group. */
      const uint16_t staged_wait = (uint16_t)(1u << (stage_slot - 1));
      if (wait_mask && (!register_operands || wait_mask != staged_wait))
         return agx3_tex_decline(I, __LINE__);

      const uint16_t fixed_scratch[3] = {0, 14, 4};
      const uint16_t *scratch = fixed_cem ? fixed_scratch : cem_scratch;
      for (unsigned axis = 0; axis < 3; ++axis) {
         const bool packed = fixed_cem
            ? agx3_tex_cube_cem_calc(
                 enc, I->src[0].value, I->src[0].value + 2,
                 I->src[0].value + 4, axis,
                 axis == 2 && I->src[0].discard, bytes, &len)
            : agx3_tex_cube_cem_calc_to(
                 enc, I->src[0].value, I->src[0].value + 2,
                 I->src[0].value + 4, scratch[axis], axis,
                 axis == 2 && I->src[0].discard, bytes, &len);
         if (!packed ||
             !agx3_append(emission, bytes, len))
            return agx3_tex_decline(I, __LINE__);

         if (axis == 0 &&
             (!(fixed_cem
                   ? agx3_tex_cube_cem_rcp(enc, bytes, &len)
                   : agx3_tex_cube_cem_rcp_to(enc, scratch[0], bytes, &len)) ||
              !agx3_append(emission, bytes, len)))
            return agx3_tex_decline(I, __LINE__);
      }

      for (unsigned component = 0; component < 2; ++component) {
         const bool packed = fixed_cem
            ? agx3_tex_cube_cem_finalize(enc, component, component == 1,
                                         bytes, &len)
            : agx3_tex_cube_cem_finalize_to(
                 enc, component, scratch[0], scratch[component + 1],
                 stage_slot, bytes, &len);
         if (!packed ||
             !agx3_append(emission, bytes, len))
            return agx3_tex_decline(I, __LINE__);
      }
   } else if (!native_staged_coords) {
      if (!agx3_tex_stage_copy_hints(enc, coord, I->src[0].value, stage_slot,
                                    wait_mask,
                                    enc == AGX_ISA_ENCODING_AGX3B &&
                                       I->src[0].cache,
                                    enc != AGX_ISA_ENCODING_AGX3B ||
                                       I->src[0].discard,
                                    bytes, &len) ||
          !agx3_append(emission, bytes, len))
         return agx3_tex_decline(I, __LINE__);
      if (!agx3_tex_stage_copy_hints(enc, coord + 2, I->src[0].value + 2,
                                    stage_slot, 0,
                                    enc == AGX_ISA_ENCODING_AGX3B &&
                                       I->src[0].cache,
                                    enc != AGX_ISA_ENCODING_AGX3B ||
                                       I->src[0].discard,
                                    bytes, &len) ||
          !agx3_append(emission, bytes, len))
         return agx3_tex_decline(I, __LINE__);
   }

   if (three_d &&
       (!agx3_tex_stage_copy_hints(
           enc, coord + 4, I->src[0].value + 4, stage_slot, 0,
           I->src[0].cache, I->src[0].discard, bytes, &len) ||
        !agx3_append(emission, bytes, len)))
      return agx3_tex_decline(I, __LINE__);

   /* The driver's noncube array ABI stages the layer into the sampler-private
    * bank and selects it with SMP f77=2.  Apple's direct f31/f37 form belongs
    * to the incompatible APPLE_FRAGMENT profile and is retained only by the
    * standalone texture encoder.
    */
   if (array && !cube && !query_lod && !combined_array_lod &&
       !register_offset &&
       !staged_offset &&
       (!agx3_tex_stage_array_layer(
           enc, I->src[0].value + 4, stage_slot, 0, I->src[0].cache,
           I->src[0].discard, bytes, &len) ||
        !agx3_append(emission, bytes, len)))
      return agx3_tex_decline(I, __LINE__);

   if (cube) {
      const uint16_t face = fixed_cem ? 2 : cem_scratch[0] + 2;
      const uint16_t layer = I->src[0].value + 6;
      bool packed;
      if (cube_array && cube_register_lod) {
         /* A bias and minimum publish the minimum at half 4, then the bias
          * beside the face. Both halves belong to one register, so only the
          * second read may discard it. */
         packed = agx3_tex_pack_ms_array_layer(
                     enc, layer, array_scratch, I->src[0].cache,
                     I->src[0].discard, bytes, &len) &&
                  agx3_append(emission, bytes, len) &&
                  (!bias_min ||
                   (agx3_tex_stage_array_minimum_lod(
                       enc, I->src[1].value + 1, stage_slot, 0,
                       I->src[1].cache, false, bytes, &len) &&
                    agx3_append(emission, bytes, len))) &&
                  agx3_tex_stage_cube_array_lod(
                     enc, face, I->src[1].value, array_scratch, stage_slot, 0,
                     I->src[1].cache, I->src[1].discard, bytes, &len);
      } else if (bias_min) {
         packed = agx3_tex_pack_cube_face(enc, face, array_scratch, bytes,
                                          &len) &&
                  agx3_append(emission, bytes, len) &&
                  agx3_tex_stage_cube_bias_min(
                     enc, I->src[1].value, I->src[1].value + 1,
                     array_scratch, stage_slot, 0, I->src[1].cache,
                     I->src[1].discard, bytes, &len);
      } else if (cube_array) {
         packed = agx3_tex_stage_cube_array_layer(
            enc, face, layer, stage_slot, 0, I->src[0].cache,
            I->src[0].discard, bytes, &len);
      } else if (cube_register_lod) {
         packed = agx3_tex_stage_cube_lod(
            enc, face, I->src[1].value, stage_slot, 0, I->src[1].cache,
            I->src[1].discard, bytes, &len);
      } else {
         packed = agx3_tex_stage_cube_z(enc, face, stage_slot, 0, false, true,
                                        bytes, &len);
      }
      if (!packed || !agx3_append(emission, bytes, len)) {
         /* The cube staging forms name half-registers in operand fields of
          * different widths, so a refusal here is about WHICH half the CEM
          * scratch and the layer landed on, not about the sample's shape.
          * Printing the shape alone sends the reader back to hardware. */
         char why[96];
         snprintf(why, sizeof(why),
                  "cube staging face=half%u layer=half%u slot=%u", face,
                  layer, stage_slot);
         agx3_report_refusal_why(I, __LINE__, why);
         return false;
      }
   }

   if (register_offset) {
      if (array &&
          (!agx3_tex_pack_ms_array_layer(
              enc, I->src[0].value + 4, array_scratch,
              I->src[0].cache, I->src[0].discard, bytes, &len) ||
           !agx3_append(emission, bytes, len)))
         return agx3_tex_decline(I, __LINE__);
      if (!agx3_tex_stage_gather_offset(
             enc, I->src[5].value + (shadow ? 2 : 0), array, array_scratch,
             stage_slot, I->src[5].cache, I->src[5].discard, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         return agx3_tex_decline(I, __LINE__);
   }

   /* The comparison reference belongs to this sample's own staging group,
    * at bank + 4 (see the sampler-private note above). Staged at a literal
    * half 4, three sampler2DShadow samples in one fragment all read
    * whichever reference was written last:
    * KHR-GL30.texture_repeat_mode's depth cases are exactly that shader. */
   if (shadow && !cube &&
       (!agx3_tex_stage_copy_hints(
           enc, coord + 4, I->src[5].value, stage_slot, 0, I->src[5].cache,
           staged_offset ? false : I->src[5].discard, bytes, &len) ||
        !agx3_append(emission, bytes, len)))
      return agx3_tex_decline(I, __LINE__);

   if (staged_component_offset) {
      if (!agx3_tex_stage_offset_2d(
             enc, offset_base, offset_base + 2, stage_slot, 0,
             I->src[5].cache, I->src[5].discard, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         return agx3_tex_decline(I, __LINE__);
   }

   /* The gradient scalars follow the coordinates, in the interleaved order
    * (ddx.x, ddy.x, ddx.y, ddy.y) the native kernels stage them in. The
    * collect that produced them is laid out the same way, so the run is
    * copied straight through. */
   if (gradient_lod) {
      static const uint16_t bank[6] = {
         AGX3_TEX_GRADIENT_BANK_DDX_X, AGX3_TEX_GRADIENT_BANK_DDY_X,
         AGX3_TEX_GRADIENT_BANK_DDX_Y, AGX3_TEX_GRADIENT_BANK_DDY_Y,
         AGX3_TEX_GRADIENT_BANK_DDX_Z, AGX3_TEX_GRADIENT_BANK_DDY_Z,
      };

      for (unsigned i = 0; i < gradient_channels; ++i) {
         if (!agx3_tex_stage_copy_hints(enc, bank[i],
                                        I->src[1].value + (2 * i), stage_slot,
                                        0, I->src[1].cache, I->src[1].discard,
                                        bytes, &len) ||
             !agx3_append(emission, bytes, len))
            return agx3_tex_decline(I, __LINE__);
      }
   }

   if (pixel_entry) {
      uint8_t entry[AGX3_PIXEL_MAX_LENGTH], entry_len = 0;
      if (pixel_entry_offset == NULL ||
          !agx3_pixel_colour_entry(enc, entry, &entry_len))
         return agx3_tex_decline(I, __LINE__);

      *pixel_entry_offset = emission->size;
      if (!agx3_append(emission, entry, entry_len))
         return agx3_tex_decline(I, __LINE__);
   }

   const struct agx3_tex_sample_desc d = {
      .dest = I->dest[0].value,
      .coord = coord,
      .width = I->dest[0].size == AGX_SIZE_32 ? AGX3_TEX_W32 : AGX3_TEX_W16,
      .mask = I->mask,
      .sparse = I->sparse,
      .texture = texture_immediate ? I->src[3].value : 0,
      .sampler = sampler_immediate ? I->src[4].value : 0,
      .texture_kind = texture_immediate ? AGX3_TEX_OPERAND_IMMEDIATE :
         I->src[3].type == AGX_INDEX_UNIFORM ? AGX3_TEX_OPERAND_UNIFORM : AGX3_TEX_OPERAND_REGISTER,
      .sampler_kind = sampler_immediate ? AGX3_TEX_OPERAND_IMMEDIATE :
         I->src[4].type == AGX_INDEX_UNIFORM ? AGX3_TEX_OPERAND_UNIFORM : AGX3_TEX_OPERAND_REGISTER,
      .texture_value = texture_immediate ? 0 : I->src[3].value,
      .sampler_value = sampler_immediate ? 0 : I->src[4].value,
      .texture_kill = !texture_immediate && I->src[3].kill,
      .sampler_kill = !sampler_immediate && I->src[4].kill,
      .gather = gather_component,
      .shadow = shadow,
      .query_lod = query_lod,
      .offset = offset,
      .offset_staged = staged_offset,
      .offset_staged_array = staged_packed_offset && array,
      .offset_xyz = offset ? I->offset_xyz : 0,
      .offset_register = register_offset,
      .min_lod = min_lod,
      .min_lod_register = register_minimum,
      .min_lod_value = register_minimum ? array_scratch : 0,
      .min_lod_kill = register_minimum,
      .lod = biased_lod       ? AGX3_TEX_LOD_BIAS
             : immediate_lod  ? AGX3_TEX_LOD_LEVEL
             : explicit_lod   ? AGX3_TEX_LOD_REGISTER
             : register_bias || bias_min ? AGX3_TEX_LOD_BIAS_REGISTER
             : gradient_lod   ? AGX3_TEX_LOD_GRADIENT
                              : AGX3_TEX_LOD_AUTO,
      .lod_x64 = immediate_lod ? immediate_x64_field : 0,
      .array = array,
      .array_layer = array && !query_lod
                        ? I->src[0].value + (cube_array ? 6 : 4) : 0,
      .three_d = three_d,
      .cube = cube,
      .slot = slot,
      .wait_mask = (uint16_t)(1u << (stage_slot - 1)),
      /* CEM consumes and discards the source XYZ before SMP. Apple's cube
       * and cube-array samples consequently set SMP f27 even though the
       * sampler reads CEM's private coordinate bank rather than the original
       * AGX IR source. Ordinary dimensions retain their existing last-use
       * mapping. */
      .coord_kill = cube ? true
                         : native_staged_coords
                              ? true
                              : I->src[0].kill,
      .final = final,
      .sample_group_nonterminal = sample_group_nonterminal,
      .sample_group_end = sample_group_end,
      .vertex_stage = vertex_stage,
      .compute_stage = compute_stage,
      .sample_abi = AGX3_TEX_SAMPLE_ABI_DRIVER,
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


/*
 * Translate the IR's sample-mask source into the lstcol/lldcol operand.
 *
 * Mesa's tilebuffer lowering spells "every sample" as the immediate 0xFF, and
 * Apple's own 1x fragments spell the same thing as the immediate 1, so that
 * one immediate is remapped. Everything else at more than one sample arrives
 * as a live 16-bit register -- either the raw coverage word (per-pixel) or
 * coverage ANDed with a one-hot bit (per-sample) -- exactly as Apple's
 * compiler emits it. See agx3_tile_core.h for the witnesses.
 *
 * There is deliberately no immediate "all samples" encoding: no Apple
 * multisample witness uses one, and an immediate would write samples this
 * fragment does not cover.
 */
#define AGX3_TILE_ALL_SAMPLES 0xFFu

static bool
agx3_tile_sample_operand(agx_index sample, struct agx3_tile_sample *out)
{
   if (sample.abs || sample.neg || sample.memory ||
       sample.size != AGX_SIZE_16)
      return false;

   if (sample.type == AGX_INDEX_IMMEDIATE) {
      /* The whole-pixel form. Apple encodes it as immediate 1 at every width
       * and every format; 0xFF is Mesa's spelling of the same request.
       *
       * AGX_TRACE_TILE_COVERAGE found this immediate being emitted for
       * multisample renders and read it as a missing tag. It was not: the
       * 2x/4x stores WERE tagged, and the tag they carried encoded the
       * register alternative of this operand naming a half register nobody
       * defined. That is why making the untagged form unencodable produced
       * no refusal -- there was no untagged store. The tags are gone now;
       * a multisample store carries a live mask register instead.
       */
      if (sample.kill || sample.discard || sample.cache ||
          sample.value != AGX3_TILE_ALL_SAMPLES)
         return false;

      *out = AGX3_TILE_SAMPLE_1X;
      return true;
   }

   if (!agx3_src_is_register(sample) || sample.value >= 192)
      return false;

   /* Register allocation rewrites `kill` into the post-RA `discard` hint on
    * some paths, and the rest of this packer reads the pair together. f55 is
    * that hint for the sample operand. A mask read again by later stores may
    * carry the post-RA `cache` hint instead; like any other live mask it
    * leaves f55 clear, as on all but the last store of Apple's 2x/4x
    * multi-target fragments. */
   *out = agx3_tile_sample_reg(sample.value, sample.kill || sample.discard);
   return true;
}

/* Bind the witnessed tile subset: the 1x whole-pixel form and the 2x/4x
 * register-sourced sample mask. Explicit pixel coordinates stay closed. */
static const char *agx3_trace_shader_name;

static bool
agx3_emit_tile(struct util_dynarray *emission, enum agx_isa_encoding enc,
               const agx_instr *I, uint8_t slot, uint16_t wait_mask,
               bool first_load, bool last_load, bool quad_store,
               bool mrt_store, bool store_f13)
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
   struct agx3_tile_sample sample_op;
   if (getenv("AGX_TRACE_TILE_COVERAGE"))
      fprintf(stderr, "TILE %s sample=%s%u fmt=%u mask=%u shader=%s\n",
              load ? "load" : "store",
              sample.type == AGX_INDEX_IMMEDIATE ? "imm " : "reg ",
              sample.value, I->format, I->mask,
              agx3_trace_shader_name ? agx3_trace_shader_name : "?");
   if (((I->format == AGX_FORMAT_RGB10A2 ||
         I->format == AGX_FORMAT_RGB9E5) &&
        (I->mask != 1 || value.size != AGX_SIZE_32)) ||
       width < 0 || !agx3_tile_value_ok(value, I->mask, width) ||
       !agx3_tile_sample_operand(sample, &sample_op) || !agx_is_null(coords))
      return false;

   /* AGX3b f13 varies independently in high-register store witnesses.  The
    * low-register, single-sample subset is the only one with a stable zero
    * selection that production IR can infer today. */
   /* Fresh two-to-four-target BG and ordinary fragments ground f13=0 for full
    * raw32 payloads starting at half16/24 too. Apple's store-only fragments of
    * three to eight targets and multi-target BG builtins of other formats
    * ground it for every width and mask, reading halves up to 63 once more
    * than four targets are stored; in those fragments f13 follows the store
    * order, not the register (agx3_g17p_mrt_protocol). Other high-bank shapes
    * retain the existing refusal. The caller proves the complete store
    * graph. */
   const bool quad_high = !load &&
      ((value.value < 32 && quad_store &&
        ((I->format == AGX_FORMAT_I32 && (I->mask == 1 || I->mask == 15)) ||
         (I->format == AGX_FORMAT_U8NORM && I->mask == 15))) ||
       (mrt_store && value.value < 64));
   if (!load && enc == AGX_ISA_ENCODING_AGX3B && value.value >= 16 && !quad_high)
      return false;

   uint8_t bytes[AGX3_TILE_MAX_LENGTH];
   uint8_t length = 0;
   /* Every retained G17P multisample LLDCol uses the register sample mask
    * with neither the first-load marker nor an incoming scoreboard wait. The
    * coverage operation that produced the mask has already completed its
    * pixel transaction before the load consumes the architectural register. */
   const uint16_t tile_wait = load && sample_op.reg ? 0 : wait_mask;
   const bool ok =
      load ? agx3_tile_load(enc, value.value, (enum agx3_tile_width)width,
                            slot, I->mask, I->pixel_offset, sample_op,
                            tile_wait, sample_op.reg ? false : first_load,
                            sample_op.reg ? false : last_load, bytes, &length)
           : agx3_tile_store_f13(enc, value.value,
                                 (enum agx3_tile_width)width, I->mask,
                                 I->pixel_offset, sample_op, tile_wait,
                                 store_f13, bytes, &length);
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
      if (src.memory || agx_channels(src) != 1)
         return false;

      /* All three operand classes are witnessed: a register (every
       * interpolated shader), an 8-bit AGX minifloat immediate (a constant
       * colour, which the optimizer folds straight into this source), and a
       * uniform (a colour read from a buffer). A register may also carry the
       * abs and neg modifiers. */
      if (agx3_src_is_register(src)) {
         if (src.size != AGX_SIZE_32)
            return false;
         d.src[i].kind = AGX3_FLTNORM_SRC_REGISTER;
         d.src[i].abs = src.abs;
         d.src[i].neg = src.neg;
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
   const uint8_t layout = I->format == AGX_FORMAT_RGB9E5    ? AGX3_PACKED_RGB9E5
                          : I->format == AGX_FORMAT_RG11B10F ? AGX3_PACKED_RG11B10
                                                             : AGX3_PACKED_RGB10A2;
   const unsigned lanes = layout == AGX3_PACKED_RGB10A2 ? 4 : 3;
   if (I->nr_dests != 1 || I->nr_srcs != (unpack ? 1 : 4))
      return false;
   const agx_index dest = I->dest[0];
   if (!agx3_src_is_register(dest) || dest.memory || dest.size != AGX_SIZE_32 ||
       agx_channels(dest) != (unpack ? lanes : 1))
      return false;
   uint8_t bytes[14], length = 0;
   if (unpack) {
      const agx_index src = I->src[0];
      if (!agx3_src_is_register(src) || src.memory || src.abs || src.neg ||
          src.size != AGX_SIZE_32 || agx_channels(src) != 1)
         return false;
      const struct agx3_rgb10a2_unpack_desc d = {
         .dest = dest.value, .source = src.value,
         .wait_mask = wait, .discard = src.discard, .layout = layout,
      };
      if (!agx3_rgb10a2_unpack_encode(&d, bytes, &length))
         return false;
   } else {
      struct agx3_rgb10a2_desc d = {.dest = dest.value, .layout = layout,
                                   .first_instruction = first, .wait_mask = wait};
      for (unsigned i = 0; i < 4; ++i) {
         const agx_index src = I->src[i];
         if (i >= lanes) {
            if (!agx_is_null(src))
               return false;
            continue;
         }
         if (src.memory || agx_channels(src) != 1 ||
             (agx3_src_is_register(src) && src.size != AGX_SIZE_32))
            return false;
         d.kind[i] = agx3_src_is_register(src) ? AGX3_FLTNORM_SRC_REGISTER :
                     src.type == AGX_INDEX_IMMEDIATE ? AGX3_FLTNORM_SRC_IMMEDIATE : 0;
         d.source[i] = src.value;
         d.discard[i] = src.discard;
         d.cache[i] = src.cache;
         d.abs[i] = src.abs;
         d.neg[i] = src.neg;
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
 * signal_pix releases the late depth/stencil class of a fragment with memory
 * side effects. `last` sets the end-of-sequence bit, which Apple puts on the
 * final release of the whole fragment program.
 */
static bool
agx3_emit_pixel(struct util_dynarray *emission, enum agx_isa_encoding enc,
                const agx_instr *I, uint16_t wait_mask, bool last)
{
   if (I->nr_dests != 0 || I->nr_srcs != 0 ||
       (last && I->op != AGX_OPCODE_SIGNAL_PIX))
      return false;

   uint8_t bytes[AGX3_PIXEL_MAX_LENGTH];
   uint8_t len = 0;
   const bool ok =
      I->op == AGX_OPCODE_WAIT_PIX
         ? agx3_pixel_wait(enc, I->writeout, wait_mask, bytes, &len)
         : agx3_pixel_release(enc, I->writeout, last, wait_mask, bytes, &len);

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
   /* Never silently omit the rest of an operand or later operands. In
    * particular, texture/sampler handles follow the large gradient tuple. */
   if (n > AGX3_SLOT_MAX_REGS || halves > AGX3_SLOT_MAX_REGS - n ||
       idx.value >= AGX3_SLOT_GPR_TOKENS ||
       halves > AGX3_SLOT_GPR_TOKENS - idx.value)
      return AGX3_SLOT_MAX_REGS + 1;
   for (unsigned i = 0; i < halves; ++i)
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
   const bool packed_byte_store =
      is_store && w == AGX3_MEM_W8 && I->memory_is_packed;
   /* Native scalar byte loads write an aligned low half. Stores can also
    * address individual halves: Apple's byte_halves_odd_zero stores
    * ushort2.y from half3 with W8/mask1. A packed vector byte store likewise
    * names its physical 16-bit payload in half-register units, regardless of
    * the logical component mask. */
   const bool scalar_half = data.size == AGX_SIZE_16 &&
      (w == AGX3_MEM_W16 || w == AGX3_MEM_W8) &&
      I->mask == 1 && agx_channels(data) == 1;
   const bool byte_load_half = !is_store && w == AGX3_MEM_W8 && scalar_half;
   const unsigned data_halfs = packed_byte_store
      ? DIV_ROUND_UP(util_bitcount(I->mask), 2)
      : agx_size_align_16(data.size) * util_bitcount(I->mask);
   if (!agx3_src_is_register(data) || data.memory ||
       ((!scalar_half || byte_load_half) && !packed_byte_store &&
        (data.value & 1)) ||
       (packed_byte_store &&
        (data.size != AGX_SIZE_16 || agx_channels(data) != data_halfs)) ||
       data.value + data_halfs > 256)
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
   d->coherent = I->coherent;
   return true;
}

/* Translate the shared-memory shape Mesa currently emits. AGX3 preprocessing
 * converts byte offsets to scalar-element indices, matching Apple's grounded
 * ldtg/sttg register-index convention. */
static bool
agx3_translate_local(enum agx_isa_encoding enc, const agx_instr *I,
                     bool is_store, uint8_t slot, uint16_t wait_mask,
                     struct agx3_local_desc *d)
{
   memset(d, 0, sizeof(*d));

   const int w = agx3_fmt_to_width(I->format);
   if (w < 0 || I->mask < 1 || I->mask > 15 ||
       I->mask != BITFIELD_MASK(util_bitcount(I->mask)))
      return false;
   d->width = (enum agx3_mem_width)w;
   d->mask = (uint8_t)I->mask;
   /* The index counts scalar elements even for vector accesses. Scaling by
    * the total vector width makes successive vec4 stores stride four times
    * too far and breaks explicitly aliased shared-memory views. */
   d->shift = d->width;

   const agx_index data = is_store ? I->src[0] : I->dest[0];
   const enum agx_size expected =
      d->width == AGX3_MEM_W32 ? AGX_SIZE_32 : AGX_SIZE_16;
   const bool packed_bytes = enc == AGX_ISA_ENCODING_AGX3B &&
      d->width == AGX3_MEM_W8 && I->memory_is_packed;
   const unsigned halves = packed_bytes ? DIV_ROUND_UP(util_bitcount(I->mask), 2)
      : agx_size_align_16(expected) * util_bitcount(I->mask);
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
   if (index.size != AGX_SIZE_16 || index.memory || index.abs || index.neg)
      return false;
   if (index.type == AGX_INDEX_IMMEDIATE && index.value < 256) {
      d->index = index.value;
   } else if (agx3_src_is_register(index) && index.value < 256) {
      d->index_is_register = true;
      /* The ldtg/sttg index is a 16-bit scalar, and its field names the
       * source in half-register units just like the compiler IR. Native
       * witnesses include odd indices (for example 5), so converting this
       * to a full-register number selects the wrong half and collapses
       * dynamic stores onto an unrelated value, commonly zero.
       */
      d->index = index.value;
      d->index_kill = index.discard;
   } else {
      return false;
   }


   /*
    * f11, `narrow_extend`, is an AGX3c field. The AGX3b `ldtg`/`sttg` forms
    * have no encoding for it at all -- composing either with f11 = 1 is
    * refused at every width, while f11 = 0 composes -- so a G17P threadgroup
    * access necessarily carries it clear, and Apple's own compiler cannot be
    * emitting anything else. Setting it from the width regardless of encoding
    * therefore refused EVERY 8-bit threadgroup load and store on G17P, which
    * is 1295 of the 1393 refused instructions in the Vulkan compute block:
    * VK_KHR_workgroup_memory_explicit_layout puts 8-bit types in shared
    * memory, and every one of them landed here.
    *
    * Clearing it leaves G17P exactly where the conformant AGX2 path already
    * is: AGX2 has no such bit either, and Mesa's IR contract (an 8-bit value
    * held in a 16-bit register) is the same on both.
    */
   d->narrow_extend =
      d->width == AGX3_MEM_W8 && enc == AGX_ISA_ENCODING_AGX3C;
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

/* AGX3b accepts either a literal byte displacement or a 16-bit GPR containing
 * the byte address.  Vulkan shared atomics use both: static accesses keep the
 * displacement, while dynamically indexed returned operations arrive with a
 * register address after scalar lowering. */
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

/* Mesa's scalar 16-bit or 32-bit stack operation is an exact match for
 * Apple's static copy/rld family: the immediate byte offset names a TPS
 * half-register and the single data operand names one half or one aligned GPR
 * pair, at W16 or W32. Constant vectors are split after RA and dynamic reads
 * use LDTPS. Other modifiers stay on the fail-closed path. */
static bool
agx3_translate_static_spill(const agx_instr *I, bool is_store, uint8_t slot,
                            uint16_t wait_mask, bool tps_live,
                            struct agx3_static_spill_store_desc *store,
                            struct agx3_static_spill_load_desc *load)
{
   if (I->op != (is_store ? AGX_OPCODE_STACK_STORE : AGX_OPCODE_STACK_LOAD) ||
       I->nr_dests != (is_store ? 0 : 1) ||
       I->nr_srcs != (is_store ? 2 : 1) ||
       (I->format != AGX_FORMAT_I16 && I->format != AGX_FORMAT_I32) ||
       I->mask != 1)
      return false;

   const bool wide = I->format == AGX_FORMAT_I32;

   const agx_index data = is_store ? I->src[0] : I->dest[0];
   const agx_index offset = I->src[is_store ? 1 : 0];
   /*
    * A spill store's source is normally DEAD after it -- that is the whole
    * point of spilling -- so agx_opt_register_cache() marks it `.discard`,
    * and refusing that rejected essentially every real spill. f27, which
    * `src_kill` drives, is exactly that last-use bit: the same field the ALU
    * forms set for AGX3_ALU_HINT_DISCARD. Carry it rather than refuse it.
    *
    * Neither form has a cache field. Apple's reloads never set f11 even when
    * an ALU consumes the value next (60 live float4 temporaries under
    * applegpu_g17p), so a cached reload destination is the same instruction.
    * A cached spill source is still live after the copy, which the clear
    * last-use bit already preserves.
    */
   if (!agx3_src_is_register(data) || data.memory || data.abs || data.neg ||
       (data.cache && data.discard) || (!is_store && data.discard) ||
       data.size != (wide ? AGX_SIZE_32 : AGX_SIZE_16) ||
       agx_channels(data) != 1 || (wide && (data.value & 1)) ||
       data.value > (wide ? 254 : 255) ||
       (!is_store && data.kill) || offset.type != AGX_INDEX_IMMEDIATE ||
       /* abs on an immediate only marks the 16-bit form (offsets >= 256) */
       offset.memory || offset.neg || offset.cache ||
       offset.discard || offset.kill || offset.size != AGX_SIZE_16 ||
       agx_channels(offset) != 1 || (offset.value & (wide ? 3 : 1)))
      return false;

   const uint32_t tps_half = offset.value >> 1;
   if (tps_half + (wide ? 2 : 1) > AGX3_SLOT_TPS_HALF_TOKENS)
      return false;

   if (is_store) {
      if (store == NULL)
         return false;
      *store = (struct agx3_static_spill_store_desc){
         .tps_half = tps_half,
         .src = data.value,
         .width = wide ? AGX3_MEM_W32 : AGX3_MEM_W16,
         /* Both spellings of "dead after this use" drive the same f27. */
         .src_kill = data.kill || data.discard,
         .slot = slot,
         .wait_mask = wait_mask,
      };
   } else {
      if (load == NULL)
         return false;
      *load = (struct agx3_static_spill_load_desc){
         .dest = data.value,
         .tps_half = tps_half,
         .width = wide ? AGX3_MEM_W32 : AGX3_MEM_W16,
         .slot = slot,
         .slot_mirror = slot,
         .wait_mask = wait_mask,
         .tps_live = tps_live,
      };
   }

   return true;
}

/*
 * Mark the static reloads whose TPS cell may be read again.
 *
 * A reload's f27 ends the lifetime of the cell it reads. applegpu_g17p sets it
 * only when the cell is written, or never touched, before its next read:
 * across kernels of live ushort3, ushort and half temporaries, each of the 18
 * reloads followed by another reload of the same cell clears it, and all 530
 * other reloads set it. Mesa's spiller can reload one spilled value several times, so a reload
 * keeps its cell live whenever another reload of any of its halves comes
 * later in the linearized program, or when it sits in a loop, whose next
 * iteration may read the cell again. Later stores are not taken as ending
 * the lifetime: on another path the value may still be read.
 */
static bool *
agx3_static_reload_rereads(agx_context *ctx, agx_instr **ordered,
                           unsigned count)
{
   unsigned blocks = 1;
   agx_foreach_block(ctx, block)
      blocks = MAX2(blocks, block->index + 1);

   bool *live = rzalloc_array(ctx, bool, count ? count : 1);
   bool *in_loop = rzalloc_array(ctx, bool, blocks);
   if (!live || !in_loop)
      return NULL;

   /* Structured loops are contiguous: a header and every block up to its
    * last backedge predecessor. */
   agx_foreach_block(ctx, header) {
      if (!header->loop_header)
         continue;
      unsigned last = header->index;
      agx_foreach_predecessor(header, pred)
         last = MAX2(last, (*pred)->index);
      agx_foreach_block(ctx, block) {
         if (block->index >= header->index && block->index <= last)
            in_loop[block->index] = true;
      }
   }

   uint32_t halves = 0;
   for (unsigned i = 0; i < count; ++i) {
      const agx_instr *I = ordered[i];
      if (I->op == AGX_OPCODE_STACK_LOAD &&
          I->src[0].type == AGX_INDEX_IMMEDIATE)
         halves = MAX2(halves, (I->src[0].value >> 1) + 2);
   }

   BITSET_WORD *read_later =
      rzalloc_array(ctx, BITSET_WORD, BITSET_WORDS(halves ? halves : 1));
   if (!read_later)
      return NULL;

   unsigned k = count;
   bool input_read_later[512] = {false};
   agx_foreach_block_rev(ctx, block) {
      agx_foreach_instr_in_block_rev(block, I) {
         if (I->op == AGX_OPCODE_STOP)
            continue;
         --k;
         if (I->op == AGX_OPCODE_LOAD_CPROG_AGX3) {
            if (I->nr_dests != 1 || I->imm >= 512 ||
                (I->dest[0].size != AGX_SIZE_16 && I->dest[0].size != AGX_SIZE_32) ||
                agx_channels(I->dest[0]) != 1)
               return NULL;
            unsigned n = agx_size_align_16(I->dest[0].size);
            if (n > 512 - I->imm)
               return NULL;
            live[k] = in_loop[block->index];
            for (unsigned h = I->imm; h < I->imm + n; ++h) {
               live[k] |= input_read_later[h];
               input_read_later[h] = true;
            }
            continue;
         }
         if (I->op != AGX_OPCODE_STACK_LOAD ||
             I->src[0].type != AGX_INDEX_IMMEDIATE)
            continue;

         const unsigned first = I->src[0].value >> 1;
         const unsigned n = I->format == AGX_FORMAT_I32 ? 2 : 1;
         bool reread = in_loop[block->index];
         for (unsigned h = first; h < first + n; ++h) {
            reread |= BITSET_TEST(read_later, h);
            BITSET_SET(read_later, h);
         }
         live[k] = reread;
      }
   }
   assert(k == 0);
   return live;
}

static bool
agx3_dynamic_stack_load(const agx_instr *I)
{
   return I->op == AGX_OPCODE_STACK_LOAD && I->nr_srcs == 1 &&
          I->src[0].type == AGX_INDEX_REGISTER;
}

static bool
agx3_translate_dynamic_stack_load(const agx_instr *I, uint8_t slot,
                                  struct agx3_tps_desc *d)
{
   if (!agx3_dynamic_stack_load(I) || I->nr_dests != 1 ||
       I->format != AGX_FORMAT_I32 || !I->mask || (I->mask & ~15))
      return false;
   const agx_index dst = I->dest[0], index = I->src[0];
   if (!agx3_tex_plain_reg(dst) || dst.size != AGX_SIZE_32 ||
       agx_channels(dst) != util_bitcount(I->mask) || (dst.value & 1) ||
       dst.value + 2 * agx_channels(dst) > 256 ||
       !agx3_tex_plain_reg(index) || index.size != AGX_SIZE_32 ||
       agx_channels(index) != 1 || (index.value & 1) || index.value > 254 ||
       (index.cache && index.discard))
      return false;
   /* G16G native uint4 private-array witness: ldtps frame1, byte index,
    * mask15. Static COPY half8 and dynamic byte offset16 name the same data. */
   *d = (struct agx3_tps_desc){
      .data = dst.value, .width = AGX3_MEM_W32, .mask = I->mask,
      .index = index.value, .index_kill = index.discard,
      .frame_region = 1, .retain_tps = true, .slot = slot,
   };
   return true;
}

static bool
agx3_dynamic_stack_store(const agx_instr *I)
{
   return I->op == AGX_OPCODE_STACK_STORE && I->nr_srcs == 2 &&
          I->src[1].type == AGX_INDEX_REGISTER;
}

/* G16G private-array witnesses (compiler/test/probes/README-g16g-dynamic-
 * stack-store.md): a dynamically indexed store is STTPS frame 1 with the
 * data mask and a register index, like the dynamic load; a float4 store uses
 * W32 mask 15 with a byte index. STTPS signals no scoreboard slot. Apple
 * issues static RLD reads and dynamic LDTPS reads of the same array right
 * after it without any wait, so the TPS pipeline orders them itself. */
static bool
agx3_translate_dynamic_stack_store(const agx_instr *I, uint16_t wait_mask,
                                   struct agx3_tps_desc *d)
{
   if (!agx3_dynamic_stack_store(I) || I->nr_dests != 0 ||
       I->format != AGX_FORMAT_I32 || !I->mask || (I->mask & ~15))
      return false;
   const agx_index value = I->src[0], index = I->src[1];
   if (!agx3_tex_plain_reg(value) || value.size != AGX_SIZE_32 ||
       agx_channels(value) != util_bitcount(I->mask) || (value.value & 1) ||
       value.value + 2 * agx_channels(value) > 256 ||
       (value.cache && value.discard) ||
       !agx3_tex_plain_reg(index) || index.size != AGX_SIZE_32 ||
       agx_channels(index) != 1 || (index.value & 1) || index.value > 254 ||
       (index.cache && index.discard))
      return false;
   *d = (struct agx3_tps_desc){
      .data = value.value, .width = AGX3_MEM_W32, .mask = I->mask,
      .index = index.value, .index_kill = index.discard,
      .frame_region = 1, .value_kill = value.discard, .slot = 0,
      .wait_mask = wait_mask,
   };
   return true;
}

static bool
agx3_append_static_spill_tokens(const agx_instr *I,
                                struct agx3_slot_instr *slot_instr)
{
   if (agx3_dynamic_stack_load(I)) {
      struct agx3_tps_desc desc;
      if (!agx3_translate_dynamic_stack_load(I, 1, &desc) ||
          slot_instr->nr_dsts >= AGX3_SLOT_MAX_REGS)
         return false;
      slot_instr->dst[slot_instr->nr_dsts++] = AGX3_SLOT_DYNAMIC_TPS_TOKEN;
      return true;
   }
   if (agx3_dynamic_stack_store(I)) {
      /* No slot is signalled; the alias token only records the write so the
       * scoreboard treats the store like any other consumer of pending
       * static TPS traffic. */
      struct agx3_tps_desc desc;
      if (!agx3_translate_dynamic_stack_store(I, 0, &desc) ||
          slot_instr->nr_srcs >= AGX3_SLOT_MAX_REGS)
         return false;
      slot_instr->src[slot_instr->nr_srcs++] = AGX3_SLOT_DYNAMIC_TPS_TOKEN;
      return true;
   }
   if (slot_instr->nr_srcs >= AGX3_SLOT_MAX_REGS)
      return false;
   /* A dynamically addressed read can alias any static write. */
   slot_instr->src[slot_instr->nr_srcs++] = AGX3_SLOT_DYNAMIC_TPS_TOKEN;
   const bool is_store = I->op == AGX_OPCODE_STACK_STORE;
   struct agx3_static_spill_store_desc store;
   struct agx3_static_spill_load_desc load;
   if (!agx3_translate_static_spill(I, is_store, 1, 0, false, &store, &load))
      return false;

   const uint32_t tps_half = is_store ? store.tps_half : load.tps_half;
   const enum agx3_mem_width width = is_store ? store.width : load.width;
   for (unsigned i = 0; i < (width == AGX3_MEM_W32 ? 2 : 1); ++i) {
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
   /* The composer takes register numbers only: a saturate, shift or source
    * modifier would be dropped. Before this check a 32-bit iadd_sat that
    * waited on a load composed as a plain iadd. */
   bool all_sources_registers = !S->saturate && !S->shift;
   for (unsigned i = 0; i < S->nr_srcs; ++i)
      all_sources_registers &= S->src[i].is_reg &&
                               S->src[i].size == AGX3_SEL_SIZE_32 &&
                               !S->src[i].abs && !S->src[i].neg;

   /* The composer's baseline is one Apple specimen, lifetime bits included.
    * Never let it end the lifetime of a source the shader still reads. */
   bool baseline_ends_live_source = false;
   for (unsigned i = 0; op != AGX3_ALU_NUM && i < S->nr_srcs; ++i)
      baseline_ends_live_source |=
         S->src[i].is_reg && S->src[i].live &&
         agx3_alu_form_ends_source(enc, op, i);

   uint8_t bytes[AGX3_ALU_MAX_LENGTH];
   uint8_t len = 0;
   const uint16_t s0 = S->nr_srcs > 0 ? S->src[0].reg : 0;
   const uint16_t s1 = S->nr_srcs > 1 ? S->src[1].reg : 0;
   const uint16_t s2 = S->nr_srcs > 2 ? S->src[2].reg : 0;

   if (op != AGX3_ALU_NUM && all_sources_registers &&
       !baseline_ends_live_source &&
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
      const bool ok =
         exact_in_place
            ? agx3_alu_band_identity_wait(enc, S->dest, prewait, prelude,
                                      &prelude_len)
            : agx3_alu_compose(enc, AGX3_ALU_LDIMM, S->dest, S->dest, S->dest,
                               0, 0, prewait, prelude, &prelude_len);
      if (!ok || !agx3_append(emission, prelude, prelude_len))
         return false;
      wait_mask &= ~prewait;
   }

   /* The wait is retired; emit the plain form. Go through the select core
    * rather than the composer so that immediate operands use their own
    * encodings instead of being mistaken for register numbers. */
   return agx3_emit(emission, enc, S);
}

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

/* The 16-bit counterpart of agx3_subgroup_plain(), in half units. */
static bool
agx3_subgroup_plain_half(agx_index index)
{
   return index.type == AGX_INDEX_REGISTER && !index.memory &&
          index.size == AGX_SIZE_16 && agx_channels(index) == 1 &&
          !index.abs && !index.neg && !index.kill &&
          !(index.cache && index.discard) && index.value < 192;
}

/* A plain immediate operand: an unmodified single-channel constant. */
static bool
agx3_subgroup_imm_lane(agx_index x)
{
   return x.type == AGX_INDEX_IMMEDIATE && !x.memory && !x.abs && !x.neg &&
          agx_channels(x) == 1;
}

static bool
agx3_emit_simd_broadcast(struct util_dynarray *emission,
                         enum agx_isa_encoding enc, const agx_instr *I,
                         uint16_t wait_mask)
{
   if (I->op != AGX_OPCODE_SHUFFLE || I->nr_dests != 1 || I->nr_srcs != 2 ||
       !agx3_subgroup_plain(I->dest[0], AGX_INDEX_REGISTER) ||
       !agx3_subgroup_plain(I->src[0], AGX_INDEX_REGISTER) ||
       !agx3_subgroup_imm_lane(I->src[1]) || I->saturate)
      return false;

   /* G17P puts the lane straight into f29 for every lane the field holds, so
    * route AGX3b through the bounded immediate encoder and let it carry the
    * producer's wait. See agx3_subgroup_core.c for Apple's per-lane sweep. */
   if (enc == AGX_ISA_ENCODING_AGX3B) {
      const struct agx3_subgroup_shuffle_desc d = {
         .op = AGX3_SUBGROUP_SHUFFLE_BROADCAST,
         .lane_kind = AGX3_SUBGROUP_LANE_IMMEDIATE,
         .dest = I->dest[0].value,
         .src = I->src[0].value,
         .lane = I->src[1].value,
         .wait_mask = wait_mask,
         .dest_hint = I->dest[0].cache,
         /* Every other shuffle site publishes this; leaving it out here left
          * the field defaulted to "final read" on EVERY immediate-lane
          * broadcast, and the hardware then destroyed the source register. */
         .src_live = !I->src[0].discard,
      };
      uint8_t bytes[AGX3_SUBGROUP_MAX_LENGTH], length = 0;
      return agx3_subgroup_encode_shuffle(enc, &d, bytes, &length) &&
             agx3_append(emission, bytes, length);
   }

   /* AGX3c keeps its own two pinned broadcast words; nothing else about that
    * encoding has been witnessed, so it stays where it was. */
   if (wait_mask != 0 ||
       (I->src[1].value != 0 && I->src[1].value != 3))
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

   /* The SIMD shuffles from a register lane at either data width, and the
    * 16-bit broadcast from an immediate lane, carry the operands' lifetime
    * hints and the source wait. The lane is read from the low half of its
    * register. */
   const agx_index lane = I->nr_srcs == 2 ? I->src[1] : agx_null();
   const bool register_lane =
      lane.type == AGX_INDEX_REGISTER && !lane.memory &&
      (lane.size == AGX_SIZE_16 || lane.size == AGX_SIZE_32) &&
      agx_channels(lane) == 1 && !lane.abs && !lane.neg && !lane.kill &&
      !(lane.cache && lane.discard) && lane.value < 192;
   if (enc == AGX_ISA_ENCODING_AGX3B && !quad && I->nr_dests == 1 &&
       I->nr_srcs == 2 && !I->saturate &&
       (register_lane ||
        (I->op == AGX_OPCODE_SHUFFLE && I->dest[0].size == AGX_SIZE_16))) {
      const agx_index dest = I->dest[0], src = I->src[0];
      const bool half = dest.size == AGX_SIZE_16;
      if (!(half ? agx3_subgroup_plain_half(dest) &&
                      agx3_subgroup_plain_half(src)
                 : agx3_subgroup_plain(dest, AGX_INDEX_REGISTER) &&
                      agx3_subgroup_plain(src, AGX_INDEX_REGISTER)) ||
          !(register_lane || agx3_subgroup_imm_lane(lane)))
         return false;

      const struct agx3_subgroup_shuffle_desc d = {
         .op = op,
         .lane_kind = register_lane ? AGX3_SUBGROUP_LANE_REGISTER
                                    : AGX3_SUBGROUP_LANE_IMMEDIATE,
         .dest = dest.value,
         .src = src.value,
         .lane = lane.value,
         .wait_mask = wait_mask,
         .dest_hint = dest.cache,
         .src_live = !src.discard,
         .half = half,
         .src_cache = src.cache,
         .src_discard = src.discard,
         .lane_cache = register_lane && lane.cache,
         .lane_discard = register_lane && lane.discard,
      };
      uint8_t bytes[AGX3_SUBGROUP_MAX_LENGTH], length = 0;
      return agx3_subgroup_encode_shuffle(enc, &d, bytes, &length) &&
             agx3_append(emission, bytes, length);
   }

   if (I->op == AGX_OPCODE_SHUFFLE && I->nr_srcs == 2 &&
       I->src[1].type == AGX_INDEX_IMMEDIATE)
      return agx3_emit_simd_broadcast(emission, enc, I, wait_mask);

   /* Every immediate-lane shuffle -- quad or SIMD, broadcast, butterfly or
    * shift -- is the same encoding point with the lane in f29, and Apple's
    * own words for all six mnemonics carry the producer wait in the f44 bank.
    * Route them all here rather than pushing the three non-broadcast SIMD
    * shapes down to the register-lane path, which cannot take a wait and so
    * had to spend a whole synchronous prelude retiring one. */
   if (I->nr_srcs == 2 && agx3_subgroup_imm_lane(I->src[1]) &&
       enc == AGX_ISA_ENCODING_AGX3B) {
      /* 16-bit data narrows this encoding point's width pair, which is a
       * different field from the quad selector, so a half broadcast or
       * butterfly encodes for quad exactly as it does for SIMD -- Apple's
       * applegpu_g17p words are in agx3_subgroup_encode_shuffle().  Reaching
       * for the 32-bit operand predicate here refused every 16-bit
       * subgroupQuadBroadcast and subgroupQuadSwap, which is one refused
       * shader per case.  The shifts keep the 32-bit predicate: their 16-bit
       * width pair is not this composition. */
      const bool half = I->nr_dests == 1 &&
                        I->dest[0].size == AGX_SIZE_16 &&
                        (op == AGX3_SUBGROUP_SHUFFLE_BROADCAST ||
                         op == AGX3_SUBGROUP_SHUFFLE_XOR);
      if (I->nr_dests != 1 || I->saturate ||
          !(half ? agx3_subgroup_plain_half(I->dest[0]) &&
                      agx3_subgroup_plain_half(I->src[0])
                 : agx3_subgroup_plain(I->dest[0], AGX_INDEX_REGISTER) &&
                      agx3_subgroup_plain(I->src[0], AGX_INDEX_REGISTER)))
         return false;

      const struct agx3_subgroup_shuffle_desc d = {
         .quad = quad,
         .op = op,
         .lane_kind = AGX3_SUBGROUP_LANE_IMMEDIATE,
         .dest = I->dest[0].value,
         .src = I->src[0].value,
         .lane = I->src[1].value,
         .wait_mask = wait_mask,
         .dest_hint = I->dest[0].cache,
         .src_live = !I->src[0].discard,
         .half = half,
      };
      uint8_t bytes[AGX3_SUBGROUP_MAX_LENGTH], length = 0;
      return agx3_subgroup_encode_shuffle(enc, &d, bytes, &length) &&
             agx3_append(emission, bytes, length);
   }

   if (quad && I->nr_srcs == 2 && I->src[1].type == AGX_INDEX_IMMEDIATE) {
      if (I->nr_dests != 1 || I->saturate ||
          !agx3_subgroup_plain(I->dest[0], AGX_INDEX_REGISTER) ||
          !agx3_subgroup_plain(I->src[0], AGX_INDEX_REGISTER) ||
          !agx3_subgroup_imm_lane(I->src[1]))
         return false;

      /* AGX3c has no witnessed wait on this form: settle any producer first. */
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
      .dest_hint = I->dest[0].cache,
      .src_live = !I->src[0].discard,
      .lane_live = !I->src[1].discard,
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
   const bool half = I->nr_dests == 1 && I->dest[0].size == AGX_SIZE_16;
   if (I->nr_dests != 1 || I->nr_srcs != 1 || I->saturate ||
       !(half ? agx3_subgroup_plain_half(I->dest[0]) &&
                   agx3_subgroup_plain_half(I->src[0])
              : agx3_subgroup_plain(I->dest[0], AGX_INDEX_REGISTER) &&
                   agx3_subgroup_plain(I->src[0], AGX_INDEX_REGISTER)))
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
      .half = half,
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
   /* A 16-bit source is grounded for the immediate integer form only; the
    * encoder refuses every other shape. */
   const agx_index src = I->src[0];
   const bool src_half =
      !is_float && src.type == AGX_INDEX_REGISTER && !src.memory &&
      src.size == AGX_SIZE_16 && agx_channels(src) == 1 && !src.abs &&
      !src.neg && !src.kill && !(src.cache && src.discard);
   if (I->nr_dests != 1 || I->nr_srcs != 2 || I->saturate ||
       I->src[1].type != AGX_INDEX_IMMEDIATE ||
       I->src[1].memory || I->src[1].abs || I->src[1].neg ||
       I->src[1].cache || I->src[1].discard || I->src[1].kill)
      return false;

   /* The 16-bit quad ballot against zero carries its own wait. */
   if (quad && !is_float && I->dest[0].size == AGX_SIZE_16) {
      if (!agx3_subgroup_plain_half(I->dest[0]) || I->src[1].value != 0 ||
          !(src_half || agx3_subgroup_plain(src, AGX_INDEX_REGISTER)) ||
          (I->icond != AGX_ICOND_UEQ && I->icond != AGX_ICOND_SEQ))
         return false;

      const struct agx3_subgroup_ballot_desc d = {
         .quad = true,
         .type = AGX3_SUBGROUP_BALLOT_INTEGER,
         .dest = I->dest[0].value,
         .src = src.value,
         .src_half = src_half,
         .dest_half = true,
         .dest_cache = I->dest[0].cache,
         .src_live = !src.discard,
         .condition = 4,
         .invert = I->invert_cond,
         .wait_mask = wait_mask,
      };
      uint8_t bytes[AGX3_SUBGROUP_MAX_LENGTH], length = 0;
      return agx3_subgroup_encode_ballot(enc, &d, bytes, &length) &&
             agx3_append(emission, bytes, length);
   }

   if (!agx3_subgroup_plain(I->dest[0], AGX_INDEX_REGISTER) ||
       !(src_half || agx3_subgroup_plain(src, AGX_INDEX_REGISTER)))
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
          * predicate.
          *
          * MEASURED on G17P (A18 Pro): the register-zero equality form -- the
          * one `x != 0` used to take -- is wrong.  A compute shader whose only
          * subgroup op is simd_ballot(x != 0) returns a bogus mask AND loses
          * its execution mask: 31 of 32 lanes stop retiring their stores from
          * the ballot onwards.  The same shader on the immediate form
          * (`x == 0`, `x < 5`) keeps all 32 lanes and returns the right mask.
          * `x != 0` is exactly `x > 0` unsigned, so route the inverted sense
          * through the immediate form the hardware agrees with.
          *
          * A nonzero immediate cannot be inverted this way; agx_optimizer
          * keeps that shape out of here by balloting the opposite compare.
          * The 16-bit quad ballot returns above on its own witnessed inverted
          * encoding, and the immediate form is grounded for AGX3B alone.
          */
         if (I->invert_cond) {
            if (!quad && enc == AGX_ISA_ENCODING_AGX3B) {
               if (compare_value != 0)
                  return false;
               compare_immediate = true;
               condition = AGX_ICOND_UGT;
               break;
            }
            /* The 16-bit quad ballot and AGX3C keep the form they had. */
            compare_immediate = false;
            condition = 4;
            break;
         }
         compare_immediate = true;
         condition = 4;
         break;
      case AGX_ICOND_SEQ:
         if (compare_value != 0)
            return false;
         /* Signed equality against zero is unsigned equality against zero, so
          * the same remap applies rather than the broken register-zero form. */
         if (!quad && enc == AGX_ISA_ENCODING_AGX3B) {
            compare_immediate = true;
            condition = I->invert_cond ? AGX_ICOND_UGT : 4;
            break;
         }
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
      .src = src.value,
      .compare_immediate = compare_immediate,
      .compare_value = compare_value,
      .src_half = src_half,
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
   if (I->op != AGX_OPCODE_SCOPED_BARRIER_AGX3 ||
       I->nr_dests != 0 || I->nr_srcs != 0 ||
       I->imm > AGX3_BARRIER_SCOPE_TEXTURE_FENCE_WAIT)
      return false;

   struct agx3_encoded_scoped_barrier encoded;
   return (!wait_mask || agx3_retire_slots_noop(emission, enc, wait_mask)) &&
          agx3_pack_scoped_barrier(enc, I->imm, &encoded) &&
          agx3_append(emission, encoded.bytes, encoded.length);
}

/* The grounded AGX3 compare forms are 32-bit and address an eight-bit HALF
 * register index, so every one of the 128 architectural 32-bit registers is
 * reachable -- see the witness block above agx3_cf_icmp_reg(), where Apple's
 * own compiler compares halves 148/150 and loads into 178/182 in the same
 * kernel.  This helper stopped at half 128, which quietly halved the usable
 * register file: any shader whose live range crossed r64 had its branch
 * refused rather than encoded.
 *
 * Source modifiers and memory/uniform operands still select forms whose
 * routing has not been proved, and those keep failing closed here. */
static bool
agx3_cf_reg32(agx_index src)
{
   return agx3_src_is_register(src) && !src.memory && src.size == AGX_SIZE_32 &&
          (src.value & 1) == 0 && src.value < 256 && !src.abs && !src.neg;
}

/* Native 16-bit register comparison addresses all 256 individual halves. */
static bool
agx3_cf_reg16(agx_index src)
{
   return agx3_src_is_register(src) && !src.memory && src.size == AGX_SIZE_16 &&
          src.value < 256 && !src.abs && !src.neg;
}

/* A float register that can be canonicalized against a minifloat. Negation
 * is admitted here because agx3_emit_cf_compare() moves it across the
 * relation and constant before calling the unmodified native encoder. */
static bool
agx3_cf_immediate_freg(agx_index src)
{
   if (!agx3_src_is_register(src) || src.memory || src.value >= 256 || src.abs)
      return false;
   if (src.size == AGX_SIZE_16)
      return true;
   return src.size == AGX_SIZE_32 && !(src.value & 1);
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

/* A float constant the control-flow compare can carry in f18: the ordinary
 * minifloat this ISA uses for immediate float operands everywhere else.
 * Zero is included, and keeps producing the bytes it always did. */
static bool
agx3_cf_fminifloat(agx_index src)
{
   return src.type == AGX_INDEX_IMMEDIATE && src.value <= 0xff && !src.abs &&
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

   /* Branching on a float against a CONSTANT. f18 is the compare's second
    * operand; it used to be pinned to zero, so only `x < 0.0` encoded and
    * `x < 1.0` -- what dEQP's workgroup-memory tests branch on, and 136 of
    * the 146 instructions left refused in the Vulkan compute block -- had no
    * encoding. The constant is the ordinary minifloat, so a zero constant
    * still emits exactly the bytes it always did. */
   if (is_float && enc == AGX_ISA_ENCODING_AGX3B &&
       ((agx3_cf_fminifloat(I->src[1]) &&
         agx3_cf_immediate_freg(I->src[0])) ||
        (agx3_cf_fminifloat(I->src[0]) &&
         agx3_cf_immediate_freg(I->src[1])))) {
      const bool zero_first=agx3_cf_fminifloat(I->src[0]);
      const agx_index source=I->src[zero_first?1:0];
      const agx_index constant=I->src[zero_first?0:1];
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
      uint8_t minifloat = (uint8_t)constant.value;
      /* Move a register negate across the comparison: -x REL c is
       * x REVERSE(REL) -c. The minifloat sign is its high bit, so this keeps
       * the existing native compare-immediate instruction and covers half
       * inverse control flow without a materialized negate. */
      if (source.neg) {
         minifloat ^= 0x80;
         switch (relation) {
         case AGX_FCOND_LT: relation=AGX_FCOND_GT; break;
         case AGX_FCOND_GT: relation=AGX_FCOND_LT; break;
         case AGX_FCOND_LE: relation=AGX_FCOND_GE; break;
         case AGX_FCOND_GE: relation=AGX_FCOND_LE; break;
         case AGX_FCOND_LTN: relation=AGX_FCOND_GTN; break;
         case AGX_FCOND_GTN: relation=AGX_FCOND_LTN; break;
         default: break;
         }
      }
      /* Both signed zeros compare identically. Apple's native -half == 0
       * branch removes the negate and keeps the immediate at positive zero,
       * whose compact field form is the one this endpoint accepts. */
      if ((minifloat & 0x7f) == 0)
         minifloat = 0;
      enum agx3_cf_fcond condition;bool or_nan;
      const bool mapped =
         agx3_cf_map_fcond(relation, &condition, &or_nan, invert);
      const bool encoded =
         mapped && !source.kill && !source.abs && agx_channels(source) == 1 &&
         agx3_cf_fcmp_zero_imm(
            enc, condition, or_nan, flag_slot, source.value,
            source.size == AGX_SIZE_16, minifloat,
            (struct agx3_cf_source_hints){source.cache, source.discard}, 0,
            bytes, &len);
      if (!encoded) {
         if (agx_compiler_debug & (AGX_DBG_SHADERS | AGX_DBG_REFUSE))
            fprintf(stderr,
                    "fcmp immediate refused: src=%u size=%u channels=%u "
                    "kill=%u abs=%u neg=%u cache=%u discard=%u imm=%u "
                    "relation=%u mapped=%u condition=%u or_nan=%u slot=%u\n",
                    source.value, source.size, agx_channels(source),
                    source.kill, source.abs, source.neg, source.cache,
                    source.discard, minifloat, relation, mapped,
                    mapped ? condition : 0, mapped ? or_nan : 0, flag_slot);
         return false;
      }
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

   /* An ABSOLUTE comparison source. Like `fsel`, the control-flow compare
    * puts ABS on a longer encoding point rather than on a bit of the short
    * one, so it needs its own encoder; letting an absolute source fall
    * through to the plain one below would drop the modifier and encode a
    * different branch. Qt Quick's conicalgradient.frag is the shape.
    *
    * ABS and NEG are not witnessed together -- NEG lives only on the short
    * form -- so that combination still fails closed. */
   if (is_float && enc == AGX_ISA_ENCODING_AGX3B &&
       agx3_src_is_register(I->src[0]) && agx3_src_is_register(I->src[1]) &&
       I->src[0].size == AGX_SIZE_32 && I->src[1].size == AGX_SIZE_32 &&
       agx_channels(I->src[0]) == 1 && agx_channels(I->src[1]) == 1 &&
       !I->src[0].memory && !I->src[1].memory &&
       (I->src[0].abs || I->src[1].abs) &&
       !I->src[0].neg && !I->src[1].neg &&
       !(I->src[0].cache && I->src[0].discard) && !I->src[0].kill &&
       !(I->src[1].cache && I->src[1].discard) && !I->src[1].kill) {
      enum agx3_cf_fcond cond;
      bool or_nan;
      const uint16_t src[] = {I->src[0].value, I->src[1].value};
      const bool abs[] = {I->src[0].abs, I->src[1].abs};
      const struct agx3_cf_source_hints hints[] = {
         {I->src[0].cache, I->src[0].discard},
         {I->src[1].cache, I->src[1].discard}};
      if (!agx3_cf_map_fcond(I->fcond, &cond, &or_nan, invert) ||
          !agx3_cf_fcmp_reg_abs(enc, cond, or_nan, flag_slot, src, abs, hints,
                                0, bytes, &len))
         return false;
      return agx3_append(emission, bytes, len);
   }

   /* Two half-float registers compare natively, as Apple's G17P compiler
    * does for `if (h0 < h1)`, with both sources' actual lifetimes. */
   if (is_float && enc == AGX_ISA_ENCODING_AGX3B &&
       agx3_src_is_register(I->src[0]) && agx3_src_is_register(I->src[1]) &&
       I->src[0].size == AGX_SIZE_16 && I->src[1].size == AGX_SIZE_16 &&
       agx_channels(I->src[0]) == 1 && agx_channels(I->src[1]) == 1 &&
       !I->src[0].memory && !I->src[1].memory &&
       !I->src[0].abs && !I->src[1].abs && !I->src[0].neg &&
       !I->src[1].neg && !I->src[0].kill && !I->src[1].kill) {
      enum agx3_cf_fcond cond;
      bool or_nan;
      const uint16_t src[] = {I->src[0].value, I->src[1].value};
      const struct agx3_cf_source_hints hints[] = {
         {I->src[0].cache, I->src[0].discard},
         {I->src[1].cache, I->src[1].discard}};
      if (!agx3_cf_map_fcond(I->fcond, &cond, &or_nan, invert) ||
          !agx3_cf_fcmp_reg16(enc, cond, or_nan, flag_slot, src, hints, 0,
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
      /* cndlt IS the two-level cndcf, in a compact four-byte form. A loop
       * owns two architectural levels, so a break with nothing else open --
       * and the loop's own continue test -- lands on it; a break that also
       * has enclosing ifs open needs the wide form's larger count. */
      ok = count <= 2
              ? agx3_cf_cndlt(enc, flag_slot, invert, bytes, &len)
              : agx3_cf_cndcf(enc, flag_slot, invert, count, bytes, &len);
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
 * form agx3_emit_alu() already uses as a prelude. This requires r0 to be ready
 * or its pending producer to be included in the mask. A partial wait for an
 * unrelated destination must use a carrier whose accesses it covers.
 *
 * bit_and carries a six-bit wait mask, so one instruction waits on the whole
 * mask before it reads r0.
 */
static bool
agx3_retire_slots_noop(struct util_dynarray *emission, enum agx_isa_encoding enc,
                       uint16_t mask)
{
   if (mask == 0)
      return true;

   /* Wait on every slot in the one instruction that reads r0. Retiring the
    * slots one instruction at a time reads and writes r0 while a later slot
    * in the mask may still be producing it, which loses that result. */
   uint8_t bytes[AGX3_ALU_MAX_LENGTH];
   uint8_t len = 0;
   return agx3_alu_band_identity_wait(enc, 0, mask, bytes, &len) &&
          agx3_append(emission, bytes, len);
}

/* The COMPACT register+minifloat ALU forms -- agx3_alu_fadd_minifloat(),
 * agx3_alu_fadd_minifloat_mod() at ep1 (unsaturated) and every
 * agx3_alu_fmul_minifloat_narrow() form -- accept only an empty wait, ONE
 * arbitrary slot, or the native slots1+2 pair 0x03. Swept over all 64 masks,
 * each of those encoders takes exactly {0x00, 0x01, 0x02, 0x03, 0x04, 0x08,
 * 0x10, 0x20}; the long forms (fadd_abs_*, saturating fadd_minifloat_mod)
 * take all 64.
 *
 * An operation that has to wait on two unrelated producers therefore has no
 * single compact encoding. Refusing it refuses the whole shader, which is what
 * `mediump` dynamic array and matrix indexing in a vertex shader hit: the
 * first `coords * 0.5` of the if-else-tree-lowered dynamic write reads one
 * async result and overwrites another, and reaches the packer with the mask
 * 0x24 or 0x0c. That refused three of the 760 dEQP-VK.glsl.indexing.* cases
 * -- tmp_array.vec4 and matrix_subscript.mat4 dynamic_write_dynamic_loop_read,
 * and matrix_subscript.mat3x2 dynamic_write_dynamic_read, all _vertex --
 * as VK_ERROR_FEATURE_NOT_PRESENT out of vkCreateGraphicsPipelines.
 *
 * Retire the excess generations through the grounded identity prelude and keep
 * one wait fused on the operation itself. Waiting early is semantically safe,
 * and the extra instruction is emitted only for a mask that has no encoding at
 * all -- a single slot and the native pair still encode in one instruction, so
 * no shader that packs today gains an instruction. */
static bool
agx3_fuse_compact_minifloat_wait(struct util_dynarray *emission,
                                 enum agx_isa_encoding enc, uint16_t *wait)
{
   const uint16_t mask = *wait;

   /* Already encodable: empty, a single slot, or the native pair. */
   if (mask == 3 || !(mask & (mask - 1)))
      return true;

   const uint16_t fused = mask & (uint16_t)-mask;
   if (!agx3_retire_slots_noop(emission, enc, mask & (uint16_t)~fused))
      return false;

   *wait = fused;
   return true;
}

/* Diagnostic (opt-in, AGX_G17P_STAGING_DRAIN=1): a staged sample consumes its
 * sampler-private coordinate/reference banks asynchronously after issue, so a
 * later sample restaging the same bank before the earlier one retires replays
 * the newest operands into the older sample. Shapes that cannot relocate their
 * bank (array, 3D, cube, gradient...) all stage into bank 0, and the only
 * protection is the retire below -- which is keyed on a scoreboard model that
 * tracks registers, not staging banks. This knob forces the full drain so the
 * missed-drain hypothesis can be tested on hardware. It serializes samples and
 * is NOT a production path. */
static bool
agx3_g17p_force_staging_drain(void)
{
   static int value = -1;
   if (value < 0) {
      const char *s = getenv("AGX_G17P_STAGING_DRAIN");
      value = (s && s[0] && s[0] != '0') ? 1 : 0;
   }
   return value == 1;
}

/* Diagnostic (AGX_G17P_DRAIN_TAIL=1): retire every scoreboard generation at
 * the end of a shader part, rather than only the ones the packer's own
 * `outstanding` ledger still believes are live.
 *
 * The production tail drain trusts that ledger. Any place where a slot was
 * marked retired without the wait actually reaching the emitted bytes leaves
 * the ledger clean while the hardware generation is still in flight, and the
 * epilog part then reads the ABI registers early. This knob is blind to the
 * ledger and so cannot share that failure mode: if a mis-tracked generation is
 * the mechanism, this closes it. It costs one instruction per part and is a
 * diagnostic, not a production path. */
static bool
agx3_g17p_drain_tail(void)
{
   static int value = -1;
   if (value < 0) {
      const char *s = getenv("AGX_G17P_DRAIN_TAIL");
      value = (s && s[0] && s[0] != '0') ? 1 : 0;
   }
   return value == 1;
}

/* Diagnostic (AGX_G17P_SINGLE_WAIT=1): give every instruction at most one
 * scoreboard wait. AGX3 has no wait instruction -- the fence is a field inside
 * each instruction -- and the witnessed encodings carry one generation. */
static int
agx3_g17p_single_wait_mode(void)
{
   static int value = -1;
   if (value < 0) {
      const char *e = getenv("AGX_G17P_SINGLE_WAIT");
      value = (e && e[0]) ? atoi(e) : 0;
   }
   return value;
}

static bool
agx3_g17p_single_wait(void)
{
   return agx3_g17p_single_wait_mode() == 1;
}

/* Diagnostic (AGX_G17P_TRACE_SLOTS=1): print the scoreboard slot and wait mask
 * the AGX3 planner assigned to every instruction. agx_insert_waits() is an
 * AGX2-only pass, so this planner is the only thing that orders an async
 * result against its consumers on G17P. */
static bool
agx3_g17p_trace_slots(void)
{
   static int value = -1;
   if (value < 0) {
      const char *e = getenv("AGX_G17P_TRACE_SLOTS");
      value = (e && e[0] && e[0] != '0') ? 1 : 0;
   }
   return value == 1;
}

/* Diagnostic (AGX_G17P_NO_SAMPLE_GROUP=1): G17P fragment SMPs are emitted as
 * one group -- f156 cleared on every member but the last, which terminates it.
 * The last member of a four-sample array-shadow group is exactly the lookup
 * that returns an uncompared texel, so take the grouping away and give every
 * sample the lone-sample encoding it would get on its own. */
static bool
agx3_g17p_no_sample_group(void)
{
   static int value = -1;
   if (value < 0) {
      const char *e = getenv("AGX_G17P_NO_SAMPLE_GROUP");
      value = (e && e[0] && e[0] != '0') ? 1 : 0;
   }
   return value == 1;
}

/* Sampler-private staging groups 0/8/16 (see the note in
 * agx3_emit_texture_sample). Take the lowest group whose previous user has
 * retired; when all three still hold in-flight operations, retire group 0's
 * user first. `slot` is the result slot of the operation about to stage into
 * the chosen group, and `outstanding` is updated for any retirement. */
static bool
agx3_take_staging_group(struct util_dynarray *emission,
                        enum agx_isa_encoding enc, uint8_t stage_group_slot[3],
                        uint16_t *outstanding, uint16_t outstanding_here,
                        uint8_t slot, uint16_t *bank_base)
{
   uint16_t busy[3];
   for (unsigned g = 0; g < 3; ++g)
      busy[g] = stage_group_slot[g]
                   ? (uint16_t)(1u << (stage_group_slot[g] - 1))
                   : 0;

   int chosen = -1;
   for (unsigned g = 0; g < 3 && chosen < 0; ++g)
      if (!(outstanding_here & busy[g]))
         chosen = (int)g;
   if (chosen < 0) {
      if (!agx3_retire_slots_noop(emission, enc, busy[0]))
         return false;
      *outstanding &= (uint16_t)~busy[0];
      *outstanding |= (uint16_t)(1u << (slot - 1));
      chosen = 0;
   }
   *bank_base = (uint16_t)(chosen * 8);
   stage_group_slot[chosen] = slot;
   return true;
}

/* An operand wait covers the instruction's inputs and overwritten destination,
 * not an arbitrary r0. An r0 identity can race an unrelated load still
 * writing r0. Carry the exact dependency mask on one scalar operand,
 * preserving its bits and width. No scratch register or unrelated wait is
 * needed. */
static bool
agx3_retire_select_operand_wait(struct util_dynarray *emission,
                                enum agx_isa_encoding enc,
                                agx_index operand, uint16_t mask)
{
   const uint16_t src[] = {operand.value, operand.value};
   const enum agx3_alu_hint hints[] = {
      AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE};
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
   const bool ok = operand.size == AGX_SIZE_16
      ? agx3_alu_bit_reg16(enc, AGX3_ALU_BAND, operand.value, src, hints,
                           mask, bytes, &len)
      : agx3_alu_bit_reg(enc, AGX3_ALU_BAND, operand.value, src, hints,
                         mask, bytes, &len);
   return ok && agx3_append(emission, bytes, len);
}

static bool
agx3_retire_select_waits(struct util_dynarray *emission,
                         enum agx_isa_encoding enc, const agx_instr *I,
                         uint16_t mask)
{
   for (unsigned s = 0; s < I->nr_srcs; ++s) {
      const agx_index operand = I->src[s];
      if (!agx3_src_is_register(operand) || operand.memory ||
          agx_channels(operand) != 1 ||
          (operand.size != AGX_SIZE_16 && operand.size != AGX_SIZE_32))
         continue;

      return agx3_retire_select_operand_wait(emission, enc, operand, mask);
   }

   return false;
}

/* Find the exact scalar operand carrying a live scoreboard slot, including
 * the previous destination value for a write-after-write dependency. Search
 * backward to its latest physical-register writer; a synchronous overwrite
 * terminates the search for that operand. */
static bool
agx3_find_select_wait_operand(const struct agx3_slot_instr *sl,
                              const struct agx3_slot_result *res,
                              unsigned consumer, const agx_instr *I,
                              uint8_t slot, agx_index *out)
{
   for (unsigned s = 0; s < I->nr_srcs + I->nr_dests; ++s) {
      const agx_index operand = s < I->nr_srcs ? I->src[s]
                                              : I->dest[s - I->nr_srcs];
      if (!agx3_src_is_register(operand) || operand.memory ||
          agx_channels(operand) != 1 ||
          (operand.size != AGX_SIZE_16 && operand.size != AGX_SIZE_32))
         continue;

      const unsigned halves = agx_index_size_16(operand);
      for (unsigned j = consumer; j-- > 0;) {
         bool writes = false;
         for (unsigned d = 0; d < sl[j].nr_dsts; ++d) {
            writes |= sl[j].dst[d] >= operand.value &&
                      sl[j].dst[d] < operand.value + halves;
         }
         if (!writes)
            continue;

         if (res[j].slot == slot) {
            *out = operand;
            return true;
         }
         break;
      }
   }

   return false;
}

/*
 * applegpu_g17p never gives one instruction more than one scoreboard wait.
 * Across 36 kernels whose consumers each read two of four or six loads in
 * crossing orders (short sub/addsat/asr, int*int, int+(short<<2),
 * int-short), Apple reuses a slot for loads consumed together and otherwise
 * orders the loads so each consumer names at most one newly pending slot.
 * A multi-slot ISEL wait was also unreliable on hardware. Retire every slot
 * but the highest through an identity on the exact operand that slot writes,
 * leaving the instruction the single wait Apple's forms carry.
 */
static bool
agx3_retire_extra_waits(struct util_dynarray *emission,
                        enum agx_isa_encoding enc,
                        const struct agx3_slot_instr *sl,
                        const struct agx3_slot_result *res,
                        unsigned consumer, const agx_instr *I,
                        uint16_t *wait)
{
   while (util_bitcount(*wait) > 1) {
      const uint16_t bit = *wait & (uint16_t)-(int16_t)*wait;
      agx_index operand;
      if (!agx3_find_select_wait_operand(sl, res, consumer, I,
                                         (uint8_t)ffs(bit), &operand) ||
          !agx3_retire_select_operand_wait(emission, enc, operand, bit))
         return false;
      *wait &= (uint16_t)~bit;
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
agx3_is_loop_push(const agx_instr *I, unsigned nest)
{
   return I->op == AGX_OPCODE_IF_FCMP && I->target == NULL && I->nest == nest &&
          I->fcond == AGX_FCOND_EQ && !I->invert_cond && I->nr_srcs == 2 &&
          agx3_cf_is_zero(I->src[0]) && agx3_cf_is_zero(I->src[1]);
}

static bool
agx3_is_loop_entry_push(const agx_instr *I)
{
   return agx3_is_loop_push(I, 1);
}

static bool
agx3_is_loop_guard_push(const agx_instr *I)
{
   return agx3_is_loop_push(I, 0);
}

static int
agx3_try_control_flow(struct util_dynarray *emission,
                      struct util_dynarray *fixups, enum agx_isa_encoding enc,
                      enum agx_chip chip,
                      const agx_instr *I, uint16_t wait_mask,
                      uint16_t outstanding, bool after_loop_backedge,
                      bool loop_exit_jump, unsigned loop_depth)
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

   if (agx3_is_loop_guard_push(I)) {
      /* A top-level loop first opens the ordinary outer control-flow scope.
       * Slot 6 is Apple's always-true predicate, so no compare is needed. */
      return wait_mask == 0 && agx3_emit_cnd(emission, enc, I->op, 6, false, 1)
                ? 1
                : -1;
   }

   if (agx3_is_loop_entry_push(I)) {
      /* agx_push_exec(1), used before every AGX3 loop, becomes Apple's
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
      /* A loop owns two architectural exec-mask levels -- Mesa's nest=1
       * break and the loop's own continue test both spend them, which is
       * CNDLT, and the shared backedge footer pops them with CNDEND(2).
       * Every enclosing `if` the break also escapes adds one more level, and
       * the lane has to stay inactive until all of them are popped. Apple
       * spells that with CNDCF, whose count is exactly the number of levels
       * open inside the innermost loop:
       *
       *   nest (levels in Mesa's model, loop counted once)  1  2  3  4 ...
       *   native levels open inside the loop                2  3  4  5 ...
       *
       * so the native count is nest + 1 at every depth. Witnessed byte for
       * byte against `metal-nt -arch applegpu_g17p` for two through seven
       * nested ifs; see agx3_cf_cndcf()'s comment for the bytes. */
      if (is_if && I->nest != 1)
         return -1;
      if (is_break_if && I->nest == 0)
         return -1;

      bool invert = I->invert_cond;
      const bool is_float = I->op == AGX_OPCODE_IF_FCMP ||
                            I->op == AGX_OPCODE_WHILE_FCMP ||
                            I->op == AGX_OPCODE_BREAK_IF_FCMP;
      /* G16G reuses each loop's slot for its breaks and wraps deep loop nests,
       * as witnessed by M4's seven- and eight-deep loop shaders. G17P instead
       * assigns a deep break the next slot so it cannot overwrite the live
       * predicate of an enclosing conditional. NEST counts exec-mask levels
       * exited, not loop depth.
       *
       * There are only six flag slots, so a per-depth assignment runs out at
       * the seventh nested loop and used to refuse the shader. It does not
       * have to: a flag slot is an ordinary allocatable resource, not a
       * per-depth one, and Apple reuses them freely on G17P too. Compiling
       * test/probes/g16g-nested-loops.metal -- the seven- and eight-deep
       * probe already in the tree -- with `metal-nt -arch applegpu_g17p`
       * gives eight CNDLT loop tests in slots 1,1,1,1,1,1,1,0: slot 1 carries
       * seven different loops' exit predicates in one program. See
       * test/probes/README-g17p-deep-loops-and-shift-immediates.md for the
       * decoded bytes.
       *
       * Our own emission is stricter still: every compare is immediately
       * followed by the CND that consumes it (see agx3_emit_cf_compare and
       * agx3_emit_cnd below), so a predicate is never live across another
       * instruction and any slot is reusable.
       *
       * Wrapping modulo six therefore keeps every depth that already encoded
       * byte-identical -- depths 1..6 are unchanged -- and only replaces a
       * refusal at depth seven and beyond. */
      if ((is_break_if || is_while) && loop_depth == 0)
         return -1;
      const unsigned loop_slot = loop_depth - 1;
      const unsigned flag_slot = chip == AGX_CHIP_G16G
         ? (is_break_if || is_while ? loop_slot % 6 : 0)
         : (is_break_if ? (loop_slot + (I->nest > 1)) % 6
                        : (is_while ? loop_slot % 6 : 0));
      if (flag_slot >= 6)
         return -1;
      if (!agx3_emit_cf_compare(emission, enc, I, is_float, flag_slot,
                                &invert)) {
         if (agx_compiler_debug & (AGX_DBG_SHADERS | AGX_DBG_REFUSE))
            fprintf(stderr, "AGX3 control-flow compare refused\n");
         return -1;
      }
      /* CNDLT retains lanes for which its predicate is true, so a shallow
       * BREAK_IF needs the complementary continue predicate. CNDCF instead
       * consumes the break predicate directly. Native break_continue uses
       * EQ(i, 9) followed by CNDCF(invert=0); deep_break uses EQ(guard, 0)
       * followed by CNDCF(invert=1). Flipping a deep unconditional break's
       * true predicate makes it false and leaves the loop running forever. */
      if (is_break_if && I->nest == 1)
         invert = !invert;
      const unsigned levels = is_break_if ? I->nest + 1 : (is_if ? 1 : 2);
      if (!agx3_emit_cnd(emission, enc, I->op, flag_slot, invert, levels)) {
         if (agx_compiler_debug & (AGX_DBG_SHADERS | AGX_DBG_REFUSE))
            fprintf(stderr, "AGX3 control-flow mask transition refused\n");
         return -1;
      }
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
         .loop_exit = loop_exit_jump,
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

   /* agx_lower_pseudo rewrites every AGX3 unconditional break into the
    * always-true BREAK_IF above, at any nesting depth, so one reaching here
    * means that pass did not run. Keep the guard rather than emitting an
    * exec-mask transition with no compare in front of it. */
   if (I->op == AGX_OPCODE_BREAK)
      return -1;

   return 0;
}

static bool
agx3_fixup_branch(struct util_dynarray *emission, enum agx_isa_encoding enc,
                  struct agx3_branch_fixup fixup)
{
   /* JMP_EXEC_NONE_AFTER normally targets the last instruction in a block.
    * A shallow loop break is different: its footer may contain CNDEND(2)
    * followed by CNDEND(1) for an enclosing scope.  The branch has to execute
    * both.  Skipping the loop pop leaves every lane inactive at END and the
    * GPU walks off the executable allocation. */
   if (fixup.loop_exit && fixup.block->loop_exit_offset < 0)
      return false;

   const off_t target =
      fixup.loop_exit
         ? fixup.block->loop_exit_offset
         : fixup.skip_to_end ? fixup.block->last_offset
                             : fixup.block->offset;
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

/* Native 11/16/32/36/52/68-output VSs and the opt-in 72-output qualification
 * shape use a dense set of scalar UVSW indices and distinct staging addresses.
 * Honeykrisp's geometry and XFB vertex variants can also have a sparse suffix,
 * such as point size alone at component 4. The same component-addressed COPY
 * and high-index UVSW protocol covers that suffix without inventing an ISA
 * form. A pure producer may appear between sparse stores (the XFB raster-
 * discard path materializes default point size this way); the nonterminal
 * protocol retires each staged value before its GPR can be reused. The stores
 * may be grouped by varying rather than sorted by scalar index. Recycle the six
 * scoreboard slots after waiting for earlier writes; keep the four-position
 * golden independent. Instructions before the output suffix are encoded and
 * validated normally. In particular, real
 * WebRender vertex shaders perform texture reads before a dense 51-scalar
 * suffix; requiring every preceding producer to be DCE-safe incorrectly
 * excluded that native-shaped stream even though no instruction crosses a
 * writeout.
 */
/* The staging address a user output's COPY writes and its UVSW reads. Apple's
 * half-2k scheme holds for every output it can address; beyond that the
 * addresses rotate, which the drained batch boundary makes safe. See
 * AGX3_G17P_STAGING_OUTPUTS. */
static uint16_t
agx3_g17p_output_staging_half(unsigned output)
{
   return (uint16_t)(2 * (output % AGX3_G17P_STAGING_OUTPUTS));
}

static unsigned
agx3_g17p_vertex_output_count(agx_context *ctx, enum agx_isa_encoding enc)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || ctx->stage != MESA_SHADER_VERTEX ||
       ctx->is_preamble)
      return 0;

   agx_block *output_block = NULL;
   agx_foreach_block(ctx, block)
      output_block = block;
   bool outputs[AGX3_MAX_VERTEX_OUTPUTS] = {false};
   unsigned output_count = 0, maximum_output = 0, minimum_output = UINT_MAX;
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
                outputs[I->src[0].value] ||
                !agx3_vary_reg32(I->src[1]) ||
                agx_channels(I->src[1]) != 1)
               return 0;
            outputs[I->src[0].value] = true;
            output_count++;
            maximum_output = MAX2(maximum_output, I->src[0].value);
            minimum_output = MIN2(minimum_output, I->src[0].value);
            break;
         case AGX_OPCODE_GET_SR:
            /* A system-value read is a pure producer, validated by the SR
             * encoder. Inside the bundle only the two IDs native vertex
             * programs read there are admitted; before it any SR is, such
             * as the subgroup invocation a ballot reads. */
            if (output_count && I->sr != AGX_SR_VERTEX_ID &&
                I->sr != AGX_SR_INSTANCE_ID)
               return 0;
            break;
         default:
            /* Once writeout begins only ST_VARY followed by STOP is safe.
             * Before it begins, side-effecting instructions are fine: they
             * stay in place and normal packing validates their encoding. */
            if (output_count && !agx_opcodes_info[I->op].can_eliminate)
               return 0;
            break;
         }
      }
   }
   const bool grounded_base =
      minimum_output == 0 || minimum_output == AGX3_UVS_POSITION_ENTRIES;
   const bool dense_wide = output_count >= 5 && grounded_base &&
      maximum_output + 1 - minimum_output == output_count;
   const bool sparse_high = ctx->key->g17p_sparse_vertex_outputs &&
      output_count > 0 && maximum_output >= 4;
   return stopped && (dense_wide || sparse_high) ? output_count : 0;
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

/* Firefox WebRender's wide vertex programs write a dense run of scalar
 * outputs, materialize the homogeneous one, then write the last.  That final
 * pure producer makes the stores nonterminal in the IR even though every
 * earlier value has already been copied to its component-addressed staging
 * word before the producer can reuse the GPR -- which is exactly why the late
 * write is safe.
 *
 * That safety argument is structural and does not depend on how many outputs
 * precede it, so the shape rather than the width is what is recognized here:
 * every store dense and in index order, the sole late producer a MOV_IMM of
 * 1.0f, that producer the exclusive source of the final store, and STOP
 * immediately after.  Anything else still fails closed, and the width ceiling
 * is still AGX3_MAX_VERTEX_OUTPUTS.
 *
 * This used to test `count != 40 && count != 51`, the two widths that had been
 * captured.  Firefox 155 emits the identical shape 49 outputs wide, so the
 * pinned pair refused its WebRender vertex program at `st_vary #4` and the
 * browser came up with no WebRender pipeline at all. */
static bool
agx3_is_firefox_dense_output_suffix(agx_context *ctx, unsigned count)
{
   if (count < 5 || count > AGX3_MAX_VERTEX_OUTPUTS)
      return false;

   const unsigned last = count - 1;

   unsigned next = 0;
   bool started = false, saw_one = false, stopped = false;
   agx_index one = agx_null();

   agx_foreach_instr_global(ctx, I) {
      if (stopped)
         return false;

      if (!started && I->op != AGX_OPCODE_ST_VARY)
         continue;
      started = true;

      if (I->op == AGX_OPCODE_ST_VARY) {
         if (I->nr_dests != 0 || I->nr_srcs != 2 ||
             I->src[0].type != AGX_INDEX_IMMEDIATE ||
             I->src[0].value != next || next > last)
            return false;

         if ((next < last && saw_one) ||
             (next == last &&
              (!saw_one || !agx_is_equiv(I->src[1], one) ||
               I->src[1].size != one.size)))
            return false;

         ++next;
         continue;
      }

      if (I->op == AGX_OPCODE_MOV_IMM && next == last && !saw_one &&
          I->nr_dests == 1 && I->nr_srcs == 0 && !I->saturate &&
          agx3_vary_reg32(I->dest[0]) && agx_channels(I->dest[0]) == 1 &&
          I->imm == 0x3f800000) {
         one = I->dest[0];
         saw_one = true;
         continue;
      }

      if (I->op == AGX_OPCODE_STOP && next == count) {
         stopped = true;
         continue;
      }

      return false;
   }

   return next == count && saw_one && stopped;
}

static int
agx3_cmp_u8(const void *a, const void *b)
{
   return (int)*(const uint8_t *)a - (int)*(const uint8_t *)b;
}

/* A dense store bundle whose writes interleave with other work can still be
 * batched when nothing between the first store and STOP stages its own
 * operands: each staged scalar then keeps its component-addressed staging
 * word until its batch's UVSW. Stores retire their staging producer
 * immediately, as the six-output fallback does, so intervening producers may
 * reuse the slots. Texture and image operations and depth/coverage exports
 * publish to the same staging bank and keep refusing the bundle. */
static bool
agx3_outputs_leave_staging_alone(agx_context *ctx)
{
   bool started = false;
   agx_foreach_instr_global(ctx, I) {
      if (I->op == AGX_OPCODE_ST_VARY) {
         started = true;
         continue;
      }
      if (!started)
         continue;
      switch (I->op) {
      case AGX_OPCODE_TEXTURE_SAMPLE:
      case AGX_OPCODE_TEXTURE_LOAD:
      case AGX_OPCODE_IMAGE_LOAD:
      case AGX_OPCODE_IMAGE_WRITE:
      case AGX_OPCODE_ZS_EMIT:
      case AGX_OPCODE_ZS_EMIT_AGX3:
      case AGX_OPCODE_SAMPLE_MASK_AGX3:
         return false;
      default:
         break;
      }
   }
   return started;
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
 *
 * The program differs between formats only in its IMGWBLK selector, which
 * names the class of imageblock lanes the store reads. Sweeping the factory
 * over internal formats 0..22 with one to four components shows the selector
 * depends on the format alone, and Apple's background clears for the same
 * formats store exactly one lane class per selector: 8-bit integer 0x55,
 * 8-bit normalized or packed 32-bit 0x75, 16-bit normalized 0x95, 16-bit
 * integer or float 0x15, and raw 32-bit 0x35. Apple's render pipelines store
 * every MTLPixelFormat in the class the tilebuffer lowering uses for it, so
 * the selector follows the store format: RGB565 and the packed 5551/4444
 * formats keep 16-bit float lanes, RG32 integer formats raw 32-bit lanes,
 * and RGBA16 UNORM normalized lanes. Each witness is a 68-byte 2D program;
 * the 84-byte programs are 2D-array builtins. The word count still tells
 * the admitted RG and RGBA shapes apart after lowering. Recognize the exact
 * post-RA semantic graph and replace it atomically, so no partially grounded
 * imgwblk ever escapes for a different format or shape. */
enum agx3_g17p_rt0_eot_kind {
   AGX3_G17P_RT0_EOT_NONE,
   AGX3_G17P_RT0_EOT_SELECTOR_75,
   AGX3_G17P_RT0_EOT_SELECTOR_15,
   AGX3_G17P_RT0_EOT_SELECTOR_35,
   AGX3_G17P_RT0_EOT_SELECTOR_55,
   AGX3_G17P_RT0_EOT_NORM16,
};

/* The IMGWBLK lane class the RT0 end-of-tile key admits for a store of this
 * format. The 2D, 2D-array and multisample builtins share it. */
static enum agx3_g17p_rt0_eot_kind
agx3_g17p_rt0_eot_store_kind(const struct agx_shader_key *key,
                             enum agx_format store_format)
{
   const unsigned words = key->g17p_rt0_eot_words;
   switch (store_format) {
   case AGX_FORMAT_I8:
      return words == 1 ? AGX3_G17P_RT0_EOT_SELECTOR_55
                        : AGX3_G17P_RT0_EOT_NONE;
   case AGX_FORMAT_U16NORM:
   case AGX_FORMAT_S16NORM:
      return words == 1 || words == 2 ? AGX3_G17P_RT0_EOT_NORM16
                                      : AGX3_G17P_RT0_EOT_NONE;
   case AGX_FORMAT_I16:
   case AGX_FORMAT_F16:
      return words == 1 || words == 2 ? AGX3_G17P_RT0_EOT_SELECTOR_15
                                      : AGX3_G17P_RT0_EOT_NONE;
   case AGX_FORMAT_I32:
      if (key->g17p_rt0_eot_i32_four_words &&
          key->g17p_rt0_eot_i32_signed_four_words)
         return AGX3_G17P_RT0_EOT_NONE;
      if (key->g17p_rt0_eot_i32_four_words ||
          key->g17p_rt0_eot_i32_signed_four_words)
         return words == 4 ? AGX3_G17P_RT0_EOT_SELECTOR_35
                           : AGX3_G17P_RT0_EOT_NONE;
      return words == 1 || words == 2 ? AGX3_G17P_RT0_EOT_SELECTOR_35
                                      : AGX3_G17P_RT0_EOT_NONE;
   default:
      return AGX3_G17P_RT0_EOT_SELECTOR_75;
   }
}

static enum agx3_g17p_rt0_eot_kind
agx3_g17p_rt0_eot_kind(const agx_context *ctx, enum agx_isa_encoding enc)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || ctx->stage != MESA_SHADER_COMPUTE ||
       ctx->is_preamble || ctx->key->no_stop)
      return AGX3_G17P_RT0_EOT_NONE;

   unsigned instructions = 0, get_x = 0, get_y = 0, offsets = 0, stores = 0;
   unsigned x_reg = UINT_MAX, y_reg = UINT_MAX, offset_imm_reg = UINT_MAX;
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
            /* The imageblock offset reaches the program through its offset
             * uniform, so a target need not start at byte zero, and may
             * start past byte 64: Apple's builtin storing any one target of
             * eight RGBA32Float targets is this single-target program. */
            if (I->nr_dests != 1 || I->nr_srcs != 0 ||
                I->imm >= AGX_G17P_MAX_BYTES_PER_SAMPLE ||
                !agx3_src_is_register(I->dest[0]) || I->dest[0].memory ||
                I->dest[0].size != AGX_SIZE_32 ||
                agx_channels(I->dest[0]) != 1)
               return false;
            offsets++;
            offset_imm_reg = I->dest[0].value;
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

   if (instructions != 4 || get_x != 1 || get_y != 1 || offsets != 1 ||
       stores != 1 || x_reg != coord_reg || y_reg != coord_reg + 2 ||
       offset_imm_reg != offset_reg)
      return AGX3_G17P_RT0_EOT_NONE;

   if (ctx->key->g17p_rt0_eot_i32_signed_four_words &&
       (store_format != AGX_FORMAT_I32 ||
        ctx->key->g17p_rt0_eot_words != 4))
      return AGX3_G17P_RT0_EOT_NONE;

   return agx3_g17p_rt0_eot_store_kind(ctx->key, store_format);
}

/* Apple's genuine builtin EOT factory, configured for a 2D-array attachment,
 * computes the tile origin from threadgroup position and dispatched group
 * size before publishing the layer and XY image wires. Match the complete
 * allocated Mesa EOT graph and its registers before substituting that exact
 * 84-byte program; no general array block-store shape is implied. */
static enum agx3_g17p_rt0_eot_kind
agx3_g17p_layered_eot_kind(const agx_context *ctx, enum agx_isa_encoding enc)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || ctx->stage != MESA_SHADER_COMPUTE ||
       ctx->is_preamble || ctx->key->no_stop)
      return AGX3_G17P_RT0_EOT_NONE;

   enum agx_format store_format = AGX_FORMAT_U8NORM;

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
            return AGX3_G17P_RT0_EOT_NONE;
         break;
      }
      case 3:
         if (I->op != AGX_OPCODE_MOV_IMM || I->nr_dests != 1 ||
             I->nr_srcs != 0 || I->imm != 0 ||
             I->dest[0].type != AGX_INDEX_REGISTER ||
             I->dest[0].size != AGX_SIZE_32 || I->dest[0].memory ||
             agx_channels(I->dest[0]) != 1 || I->dest[0].value != 6)
            return AGX3_G17P_RT0_EOT_NONE;
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
            return AGX3_G17P_RT0_EOT_NONE;
         break;
      }
      case 7:
         if (I->op != AGX_OPCODE_BLOCK_IMAGE_STORE || I->nr_dests != 0 ||
             I->nr_srcs != 4 ||
             (I->format == AGX_FORMAT_I32) != ctx->key->g17p_rt0_eot_i32 ||
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
            return AGX3_G17P_RT0_EOT_NONE;
         store_format = I->format;
         break;
      default:
         return AGX3_G17P_RT0_EOT_NONE;
      }
   }

   if (instruction != 8)
      return AGX3_G17P_RT0_EOT_NONE;

   return agx3_g17p_rt0_eot_store_kind(ctx->key, store_format);
}

/* Native EOT key {RGBA16F RT0 at partition 0, R8 RT1 at a nonzero partition}
 * (mask 3) has a distinct, complete 238-byte program.  As above, match the
 * whole post-RA graph before replacing it: this is the exact mixed-width
 * layout used by Quake's WBOIT pass. */
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
agx3_is_g17p_dual_word_mrt_eot(const agx_context *ctx,
                               enum agx_isa_encoding enc,
                               enum agx_format format)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || ctx->stage != MESA_SHADER_COMPUTE ||
       ctx->is_preamble || ctx->key->no_stop ||
       (format == AGX_FORMAT_I32 && !ctx->key->g17p_r11g11b10_mrt_eot) ||
       (format != AGX_FORMAT_I32 && format != AGX_FORMAT_U8NORM))
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
                I->format != format || I->dim != AGX_DIM_2D ||
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
                  ((I->format == AGX_FORMAT_RGB10A2 ||
                    I->format == AGX_FORMAT_RGB9E5) && I->mask == 1)) ||
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

/* G17P allocates two pixel classes per colour-tile partition.  Fresh native
 * two-RGBA8 and RGBA16F+R8 shaders agree on the same partition protocol even
 * though their byte offsets differ: partition 1 uses read/write/combined
 * classes 32/16/48 and partition 0 uses 8/4/12.  The mixed programmable-blend
 * witness is wboit-blend.nat SHA256
 * ae46c7cc23179c4df4f59e081eb10c69666af181da48103c1f0ebcf3d46b85e4;
 * its fragment text SHA256 is
 * ce78a72cac5cf5d4e49dc240b56ac72d7ba89d234de912a7833759d918759c44.
 *
 * Derive partition identity from the sorted physical tile offsets rather than
 * a format or workload name.  Every partition must have one terminal store;
 * loads are optional, and the pre-RA sink guarantees stores descend to the
 * zero-offset final release. The third class pair uses 128/64/192. Every
 * partition from the fourth on reuses partition 0's class pair: Apple's
 * store-only fragments of four to eight targets acquire class 12 before the
 * first store using it and release it after the last, holding it across
 * partitions 2 and 1 when partition 0 is written.
 *
 * Blending fragments of four to eight targets hold the same classes around
 * their loads. Partition 0's read class (8) is acquired once, before the
 * first load of any partition sharing its pair, and held; partitions 2 and 1
 * wait for their own read classes before their loads. The stores then
 * acquire the shared write class (4) before the first shared store when any
 * shared partition was loaded, or all of class 12 otherwise, and partitions 2
 * and 1 acquire their write class if loaded or both of their classes if not.
 * Loads go in descending partition order, as in Apple's RGBA8 fragments; its
 * RGBA16Float fragments order them differently under the same held classes.
 * See the metal-nt applegpu_g17p witnesses in test-agx-g17p-mrt-mixed.c:
 * two to eight RGBA8 and three, four, five and eight RGBA16Float targets all
 * blended, and 17 blended subsets of four, five and eight RGBA8 targets.
 *
 * Those fragments also set lstcol f13 on their leading stores when they store
 * more than five targets including targets 1 and 2: on all but the last five
 * stores, whatever the formats and registers. metal-nt applegpu_g17p
 * witnesses: five to eight RGBA8 targets, six and eight RGBA32F targets, eight
 * mixed targets, and 34 written subsets of eight RGBA8 targets.
 *
 * An epilog writing only some of the framebuffer's targets takes the
 * partitions from the key instead. Apple's store-only fragments for such
 * pipelines keep every written target's classes from its position among all
 * targets and store in descending partition order.
 */
struct agx3_g17p_mrt_protocol {
   bool valid;
   bool raw32_store;
   bool has_loads;
   /* Stores carry the coverage register. When every partition is stored,
    * all of their classes are acquired and released once, around all the
    * stores. */
   bool multisample;
   bool multisample_combined;
   uint16_t combined_classes;
   uint8_t count;
   /* Colour classes of the stored partitions, bit n for class pair n. */
   uint8_t classes;
   /* The first and last stored partitions of class pair 0, in store order. */
   uint8_t shared_first, shared_last;
   /* Whether a partition of class pair 0 is loaded, and the first such load. */
   bool shared_loaded;
   uint8_t shared_load_first;
   /* Number of leading stores carrying lstcol f13. */
   uint8_t f13_stores;
   uint16_t offsets[AGX_MAX_RTS];
   /* The logical colour attachment occupying each offset. An inactive
    * attachment compacts the physical layout but keeps every other
    * attachment's identity, and a fragment selects its synchronization
    * classes from that identity rather than from its position among the
    * written targets. */
   uint8_t targets[AGX_MAX_RTS];
   uint8_t loads[AGX_MAX_RTS];
   uint8_t stores[AGX_MAX_RTS];
};

/* Partitions 1 and 2 own class pairs 1 and 2; every other partition shares
 * partition 0's. */
static unsigned
agx3_g17p_mrt_class_pair(unsigned partition)
{
   return partition < 3 ? partition : 0;
}

/* A multisample background reload writes every sample of one tilebuffer
 * partition explicitly after fetching it from the multisample attachment.
 * These are not multiple render targets: the per-sample MSAA lowering unrolls
 * one store per sample, each carrying its own live sample-mask register
 * (coverage ANDed with that iteration's one-hot bit -- the shape Apple's own
 * unrolled per-sample fragments use), while every store retains the same RT
 * byte offset and format.
 *
 * A fragment unrolled per sample stores the same way inside the ordinary
 * colour transaction. Apple's fs_sample_id_2x/4x (applegpu_g17p, one RGBA8
 * target, reading [[sample_id]]) acquire class 12 once after the 0x200 entry,
 * emit one LSTCOL per sample with that sample's mask register, and release
 * 524 once, exactly as the one-sample fragment brackets its single LSTCOL.
 *
 * Keep this admission as narrow as the native evidence.  A background may
 * emit one two- or four-store set to partition zero with no pixel
 * transaction, and a multi-target background one such set per target, at
 * disjoint whole-sample tile ranges; a colour epilog emits one set to
 * partition zero, at most one per framebuffer sample, after its one class-12
 * wait. API-disabled samples have zero coverage, so their no-op colour stores
 * may already have been eliminated. They do not require a different pixel
 * transaction or a replacement store.
 * Within a target each store's mask must be computed by its own instruction
 * so a repeated one cannot masquerade as a complete sample domain.  Returns
 * the number of targets, or zero.
 *
 * Distinct mask REGISTERS were the first spelling of that, taken from
 * Apple's msaa_f_sampleid_s4, whose four stores land in four registers.  A
 * register is not the mask, though: register allocation is free to reuse one
 * register for masks that are killed at the store that reads them, and on the
 * real four-sample reload it does -- three of the four stores name the same
 * half register, each recomputed in between.  Refusing that refused the only
 * shader this protocol was written for.  Two stores carry the same mask when
 * they name the same register AND nothing redefined it in between, so that is
 * what is checked.
 */
static unsigned
agx3_g17p_sample_store_protocol(agx_context *ctx, bool colour)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B ||
       ctx->stage != MESA_SHADER_FRAGMENT || ctx->is_preamble ||
       ctx->key->fs.ignore_tib_dependencies == colour ||
       list_length(&ctx->blocks) != 1)
      return 0;

   /* Post-RA, the definition that reaches each store is the most recent write
    * of that half register, and the block is straight-line by the test above.
    * 192 is the sample operand's own register ceiling in
    * agx3_tile_sample_operand(). */
   const agx_instr *reaching[192] = {NULL};
   struct {
      uint16_t offset;
      enum agx_format format;
      unsigned component_mask;
      unsigned stores;
      uint16_t mask_regs[4];
      const agx_instr *mask_defs[4];
   } targets[AGX_MAX_RTS];
   unsigned target_count = 0;
   uint64_t written_B = 0;
   bool stopped = false, waited = false;
   /* A colour epilog may first complete one class-1 depth/stencil
    * transaction of every sample's write, as Apple's per-sample depth
    * fragments do before their colour stores: 1 waited, 2 written,
    * 3 released. */
   unsigned depth = 0;

   agx_foreach_instr_global(ctx, I) {
      if (stopped)
         return 0;

      if (I->op == AGX_OPCODE_STOP) {
         stopped = true;
         continue;
      }

      if (colour && !waited && I->op == AGX_OPCODE_WAIT_PIX &&
          I->writeout == 1 && depth == 0) {
         depth = 1;
         continue;
      }
      if (colour && I->op == AGX_OPCODE_ZS_EMIT_AGX3 &&
          (depth == 1 || depth == 2)) {
         depth = 2;
         continue;
      }
      if (colour && I->op == AGX_OPCODE_SIGNAL_PIX && I->writeout == 1 &&
          depth == 2) {
         depth = 3;
         continue;
      }

      if (I->op == AGX_OPCODE_WAIT_PIX && colour && !waited &&
          !target_count && I->writeout == 12 && (depth == 0 || depth == 3)) {
         waited = true;
         continue;
      }

      if (agx3_is_cf_op(I) || I->op == AGX_OPCODE_WAIT_PIX ||
          I->op == AGX_OPCODE_SIGNAL_PIX || I->op == AGX_OPCODE_LD_TILE)
         return 0;

      if (I->op != AGX_OPCODE_ST_TILE) {
         agx_foreach_dest(I, d) {
            const agx_index dest = I->dest[d];
            if (!agx3_src_is_register(dest))
               continue;

            const unsigned halfs = agx_index_size_16(dest);
            for (unsigned h = 0; h < halfs; ++h) {
               const unsigned reg = dest.value + h;
               if (reg < ARRAY_SIZE(reaching))
                  reaching[reg] = I;
            }
         }
         continue;
      }

      struct agx3_tile_sample sample;
      if (I->nr_srcs != 3 || (colour && I->pixel_offset != 0) ||
          I->explicit_coords || !agx_is_null(I->src[2]) ||
          !agx3_tile_sample_operand(I->src[1], &sample) || !sample.reg)
         return 0;

      unsigned t = 0;
      while (t < target_count && targets[t].offset != I->pixel_offset)
         t++;

      if (t == target_count) {
         /* A new target: its whole samples must not overlap another's. */
         const int width = agx3_tile_format_width(I->format);
         const unsigned size_B =
            width < 0 ? 0 : util_last_bit(I->mask) << width;
         const uint64_t bytes =
            BITFIELD64_MASK(size_B) << MIN2(I->pixel_offset, 63);
         if (t == ARRAY_SIZE(targets) || !size_B ||
             I->pixel_offset + size_B > 64 || (written_B & bytes))
            return 0;

         written_B |= bytes;
         targets[t].offset = I->pixel_offset;
         targets[t].format = I->format;
         targets[t].component_mask = I->mask;
         targets[t].stores = 0;
         target_count++;
      } else if (I->format != targets[t].format ||
                 I->mask != targets[t].component_mask) {
         return 0;
      }

      if (targets[t].stores >= ARRAY_SIZE(targets[t].mask_defs))
         return 0;

      /* Same register and the same definition still reaching it is the same
       * mask twice, whatever it holds. Each target writes a sample at most
       * once; background targets of one sample share its mask. */
      const agx_instr *def = sample.value < ARRAY_SIZE(reaching)
                                ? reaching[sample.value] : NULL;

      for (unsigned i = 0; i < targets[t].stores; ++i) {
         if (targets[t].mask_regs[i] == sample.value &&
             targets[t].mask_defs[i] == def)
            return 0;
      }

      targets[t].mask_regs[targets[t].stores] = sample.value;
      targets[t].mask_defs[targets[t].stores++] = def;
   }

   const unsigned samples =
      target_count ? targets[0].stores : 0;
   for (unsigned t = 0; t < target_count; ++t) {
      if (targets[t].stores != samples)
         return 0;
   }

   const bool complete = colour
      ? (ctx->key->fs.nr_samples == 2 || ctx->key->fs.nr_samples == 4) &&
           samples > 0 && samples <= ctx->key->fs.nr_samples
      : samples == 2 || samples == 4;
   return stopped && waited == colour && complete &&
                (depth == 0 || depth == 3) &&
                (target_count == 1 || ctx->key->g17p_mrt_bg)
             ? target_count : 0;
}

static int
agx3_g17p_mrt_partition(const struct agx3_g17p_mrt_protocol *p,
                        uint16_t offset)
{
   for (unsigned i = 0; i < p->count; ++i) {
      if (p->offsets[i] == offset)
         return i;
   }

   return -1;
}

/* Metal has no geometry or tessellation-control stage: Apple compiles both as
 * kernels, and Honeykrisp launches them the same way, through CDM. Their
 * texture fetches must therefore take the kernel path here. The unstaged form
 * returns only x and y on AGX3B -- see the `staged` decision below -- and it
 * also leaves the staging footprint unknown, which makes hk_dispatch_g17p_
 * native() refuse the dispatch outright. */
static inline bool
agx3_stage_runs_as_kernel(mesa_shader_stage stage)
{
   return stage == MESA_SHADER_COMPUTE || stage == MESA_SHADER_TESS_CTRL ||
          stage == MESA_SHADER_GEOMETRY;
}

static struct agx3_g17p_mrt_protocol
agx3_g17p_mrt_refused(unsigned line)
{
   if (agx_compiler_debug & AGX_DBG_REFUSE)
      fprintf(stderr, "G17P MRT protocol refused at %u\n", line);
   return (struct agx3_g17p_mrt_protocol){0};
}

static struct agx3_g17p_mrt_protocol
agx3_g17p_mrt_protocol(agx_context *ctx)
{
   struct agx3_g17p_mrt_protocol p = {0};
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B ||
       ctx->stage != MESA_SHADER_FRAGMENT || ctx->is_preamble ||
       !ctx->agx3_mrt_tile_ops_ordered ||
       (!ctx->key->secondary && !ctx->key->fs.ignore_tib_dependencies) ||
       (!ctx->key->g17p_emrt && list_length(&ctx->blocks) != 1)) {
      /* Not a refusal by itself -- a single-target shader legitimately has no
       * MRT protocol -- so report only when the key names two or more tile
       * partitions, which is when the protocol is the thing a later tile op
       * will need. Reporting every shader buries the one that matters. */
      if ((agx_compiler_debug & AGX_DBG_REFUSE) &&
          __bitset_count(ctx->key->g17p_tile_partitions,
                         BITSET_WORDS(AGX_G17P_MAX_BYTES_PER_SAMPLE)) >= 2) {
         fprintf(stderr,
                 "G17P MRT protocol declined: agx3b=%u frag=%u preamble=%u "
                 "ordered=%u secondary=%u ignore-tib=%u emrt=%u blocks=%u\n",
                 ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B,
                 ctx->stage == MESA_SHADER_FRAGMENT, ctx->is_preamble,
                 ctx->agx3_mrt_tile_ops_ordered, ctx->key->secondary,
                 ctx->key->fs.ignore_tib_dependencies, ctx->key->g17p_emrt,
                 list_length(&ctx->blocks));
      }
      return p;
   }

   const bool partitions =
      !ctx->key->fs.ignore_tib_dependencies &&
      !__bitset_is_empty(ctx->key->g17p_tile_partitions,
                         BITSET_WORDS(AGX_G17P_MAX_BYTES_PER_SAMPLE));
   if (partitions) {
      if (__bitset_count(ctx->key->g17p_tile_partitions,
                         BITSET_WORDS(AGX_G17P_MAX_BYTES_PER_SAMPLE)) >
          ARRAY_SIZE(p.offsets))
         return p;
      unsigned offset;
      BITSET_FOREACH_SET(offset, ctx->key->g17p_tile_partitions,
                         AGX_G17P_MAX_BYTES_PER_SAMPLE) {
         p.targets[p.count] = p.count;
         p.offsets[p.count++] = offset;
      }
   }

   bool depth_written = false, tile_ops = false, tile_loads = false;
   agx_foreach_instr_global(ctx, I) {
      /* A depth export may finish its own class-1 transaction first. */
      if (I->op == AGX_OPCODE_ZS_EMIT_AGX3 && !tile_ops)
         depth_written = true;
      if ((!ctx->key->g17p_emrt && agx3_is_cf_op(I)) ||
          (I->op == AGX_OPCODE_SIGNAL_PIX &&
           (!depth_written || I->writeout != 1 ||
            (tile_ops && !ctx->key->g17p_emrt))))
         return agx3_g17p_mrt_refused(__LINE__);
      if (I->op != AGX_OPCODE_LD_TILE && I->op != AGX_OPCODE_ST_TILE)
         continue;
      if (I->explicit_coords || I->pixel_offset > UINT8_MAX)
         return agx3_g17p_mrt_refused(__LINE__);

      tile_ops = true;
      tile_loads |= I->op == AGX_OPCODE_LD_TILE;
      unsigned i;
      for (i = 0; i < p.count; ++i) {
         if (p.offsets[i] == I->pixel_offset)
            break;
      }
      if (i == p.count) {
         if (partitions || p.count == ARRAY_SIZE(p.offsets))
            return agx3_g17p_mrt_refused(__LINE__);
         p.targets[p.count] = I->colour_target;
         p.offsets[p.count++] = I->pixel_offset;
      } else if (!partitions && p.targets[i] != I->colour_target) {
         return agx3_g17p_mrt_refused(__LINE__);
      }
   }
   /* A lone attachment that is not attachment zero is still a multi-target
    * layout: it owns its own class, not class pair zero.
    * g17p-sparse-mrt.h's fragment-8-active-mask04 is exactly that shape. */
   /* Native id_blend_2x/4x uses the same class-12 lifetime for a single
    * target's unrolled sample loads/stores, acquiring read class 8 once
    * and write class 4 once. Keep the ordinary one-store path unchanged. */
   const bool single_target_blend =
      ctx->key->fs.nr_samples > 1 && tile_loads && !ctx->key->g17p_emrt;

   /* A multisample eMRT epilog may have one resident partition and no other:
    * hk spills a reloaded multisample attachment wider than
    * AGX_G17P_MAX_UNCOVERED_TILE_BYTES_PER_SAMPLE, which leaves exactly one
    * eight-byte target in the tile and every other target in memory. Its tile
    * side is the same one-target unrolled sample set as above -- one LSTCol
    * per sample through the coverage register at partition zero, inside one
    * class-12 bracket -- because the spilled targets are image stores that
    * carry no pixel class at all.
    *
    * agx3_g17p_sample_store_protocol() admits that shape directly when it is
    * resident, but only for a straight-line shader with no tile loads, and an
    * eMRT epilog is never straight-line: the spilled attachments' image
    * accesses bring their own control flow (19 to 43 blocks for the four- and
    * eight-attachment VK_EXT_rasterization_order_attachment_access render
    * passes). So neither protocol claimed it, the packer reached
    * "tile_stores_left > 1 with no proven partition graph", and the whole
    * epilog failed -- reported by hk as VK_ERROR_FEATURE_NOT_PRESENT from
    * vkEndCommandBuffer.
    *
    * The multisample eMRT acquire below already emits one combined wait for
    * the stored partitions and the release already ends at the final store,
    * so no new class or instruction shape is introduced here; only
    * p.count == 1 is new. The resident partition may also be loaded --
    * agx_nir_lower_tilebuffer() preserves an existing-but-unwritten resident
    * attachment of a multisample eMRT layout with a load/store pair -- and
    * that combined acquire covers the read class too. */
   const bool single_target_emrt =
      ctx->key->fs.nr_samples > 1 && ctx->key->g17p_emrt;

   if (!p.count || (p.count == 1 && p.targets[0] == 0 &&
                    !single_target_blend && !single_target_emrt))
      return agx3_g17p_mrt_refused(__LINE__);

   /* A background of four or more targets keeps the constructors' ascending,
    * transaction-free shape; see the note in agx_compile.c. */
   if (p.count >= 4 && ctx->key->fs.ignore_tib_dependencies)
      return agx3_g17p_mrt_refused(__LINE__);

   for (unsigned i = 0; i < p.count; ++i) {
      for (unsigned j = i + 1; j < p.count; ++j) {
         if (p.offsets[j] < p.offsets[i]) {
            const uint16_t tmp = p.offsets[i];
            p.offsets[i] = p.offsets[j];
            p.offsets[j] = tmp;
            const uint8_t target = p.targets[i];
            p.targets[i] = p.targets[j];
            p.targets[j] = target;
         }
      }
   }
   if (p.offsets[0] != 0)
      return agx3_g17p_mrt_refused(__LINE__);

   p.multisample = ctx->key->fs.nr_samples > 1 &&
                   !ctx->key->fs.ignore_tib_dependencies;
   unsigned previous_load = p.count;
   uint8_t load_order[AGX_MAX_RTS * 4];
   unsigned load_instr_count = 0;
   uint8_t store_order[AGX_MAX_RTS * 4];
   unsigned store_instr_count = 0;
   bool stores_started = false;
   bool stopped = false;
   agx_foreach_instr_global(ctx, I) {
      if (stopped)
         return agx3_g17p_mrt_refused(__LINE__);

      if (I->op == AGX_OPCODE_STOP) {
         stopped = true;
         continue;
      }

      if (I->op != AGX_OPCODE_LD_TILE && I->op != AGX_OPCODE_ST_TILE) {
         if (stores_started && !ctx->key->g17p_emrt &&
             (!p.multisample || !agx_opcodes_info[I->op].can_eliminate))
            return agx3_g17p_mrt_refused(__LINE__);
         continue;
      }

      const int partition = agx3_g17p_mrt_partition(&p, I->pixel_offset);
      /* Four to eight targets of any formats share the class-12 lifetime;
       * see the witnesses in test-agx-g17p-mrt-mixed.c. */
      const bool load = I->op == AGX_OPCODE_LD_TILE;
      if (p.count >= 4 &&
          (I->nr_srcs != (load ? 2 : 3) || I->nr_dests != (load ? 1 : 0) ||
           !agx_is_null(I->src[load ? 1 : 2]) || partition < 0))
         return agx3_g17p_mrt_refused(__LINE__);
      struct agx3_tile_sample sample;
      if (partition < 0 || I->nr_srcs < 2 ||
          !agx3_tile_sample_operand(
             I->src[I->op == AGX_OPCODE_LD_TILE ? 0 : 1], &sample) ||
          sample.reg != p.multisample) {
         if (agx_compiler_debug & AGX_DBG_REFUSE) {
            agx_index mask = I->src[load ? 0 : 1];
            fprintf(stderr, "G17P MRT sample: part=%d load=%u offset=%u type=%u value=%u size=%u ms=%u\n",
                    partition, load, I->pixel_offset, mask.type, mask.value,
                    mask.size, p.multisample);
         }
         return agx3_g17p_mrt_refused(__LINE__);
      }

      if (I->op == AGX_OPCODE_LD_TILE) {
         const unsigned max_loads =
            p.multisample ? ctx->key->fs.nr_samples : 1;
         if (++p.loads[partition] > max_loads ||
             (!p.multisample &&
              (stores_started || (unsigned)partition > previous_load)) ||
             load_instr_count >= ARRAY_SIZE(load_order))
            return agx3_g17p_mrt_refused(__LINE__);
         previous_load = partition;
         load_order[load_instr_count++] = partition;
      } else {
         stores_started = true;
         if (store_instr_count >= ARRAY_SIZE(store_order))
            return agx3_g17p_mrt_refused(__LINE__);
         p.stores[partition]++;
         store_order[store_instr_count++] = partition;
      }
   }

   unsigned stored_partitions = 0;
   unsigned stores_per_partition = 0;
   for (unsigned i = 0; i < p.count; ++i) {
      if (!p.stores[i])
         continue;

      stored_partitions++;
      if (!stores_per_partition)
         stores_per_partition = p.stores[i];
      else if (p.stores[i] != stores_per_partition)
         return agx3_g17p_mrt_refused(__LINE__);
   }

   if (!stored_partitions ||
       (stores_per_partition != 1 &&
        !(p.multisample &&
          stores_per_partition == ctx->key->fs.nr_samples)) ||
       store_instr_count != stored_partitions * stores_per_partition)
      return agx3_g17p_mrt_refused(__LINE__);

   unsigned loaded_partitions = 0;
   for (unsigned i = 0; i < p.count; ++i) {
      if (p.loads[i] && p.loads[i] != stores_per_partition)
         return agx3_g17p_mrt_refused(__LINE__);
      loaded_partitions += p.loads[i] != 0;
   }

   if (load_instr_count != loaded_partitions * stores_per_partition)
      return agx3_g17p_mrt_refused(__LINE__);

   if (p.multisample) {
      for (unsigned sample = 0; sample < stores_per_partition; ++sample) {
         unsigned at = sample * loaded_partitions;
         for (int partition = p.count - 1; partition >= 0; --partition) {
            if (p.loads[partition] && load_order[at++] != partition)
               return agx3_g17p_mrt_refused(__LINE__);
         }
      }
   }

   /* Each unrolled sample repeats the same descending target set. This is
    * the multi-target form of the already-grounded per-sample colour-store
    * protocol; no new instruction or modifier shape is introduced. */
   for (unsigned sample = 0; sample < stores_per_partition; ++sample) {
      unsigned at = sample * stored_partitions;
      for (int partition = p.count - 1; partition >= 0; --partition) {
         if (!p.stores[partition])
            continue;
         if (store_order[at++] != partition)
            return agx3_g17p_mrt_refused(__LINE__);
      }
   }

   unsigned store_count = 0;
   p.multisample_combined = p.multisample;
   for (int i = p.count - 1; i >= 0; --i) {
      p.multisample_combined &= p.stores[i] != 0;
      /* `partitions` already enumerates every framebuffer attachment, so its
       * ordinal is the identity; otherwise the attachment names itself.
       * g17p-sparse-mrt.h's six native oracles all disagree with the ordinal:
       * an eight-slot fragment writing only targets 1 and 3 uses classes
       * {1, 3}, not {0, 1}. */
      const unsigned pair = agx3_g17p_mrt_class_pair(p.targets[i]);
      uint32_t read_class, write_class, combined_class;
      if ((p.stores[i] != stores_per_partition &&
           !(partitions && !p.stores[i] && !p.loads[i])) ||
          !agx3_pixel_colour_partition_classes(
             pair, &read_class, &write_class, &combined_class))
         return agx3_g17p_mrt_refused(__LINE__);
      p.has_loads |= p.loads[i];
      if (pair == 0 && p.loads[i] && !p.shared_loaded) {
         p.shared_loaded = true;
         p.shared_load_first = i;
      }
      if (!p.stores[i])
         continue;
      if (pair == 0 && !(p.classes & BITFIELD_BIT(0)))
         p.shared_first = i;
      if (pair == 0)
         p.shared_last = i;
      p.classes |= BITFIELD_BIT(pair);
      p.combined_classes |= combined_class;
      store_count++;
   }
   /* Multisample store-only fragments of up to eight targets carry no f13. */
   if (p.stores[1] && p.stores[2] && store_count > 5 && !p.multisample)
      p.f13_stores = store_count - 5;

   p.raw32_store = true;
   agx_foreach_instr_global(ctx, I) {
      if (I->op == AGX_OPCODE_LD_TILE)
         p.raw32_store = false;
      if (I->op == AGX_OPCODE_ST_TILE) {
         const int partition = agx3_g17p_mrt_partition(&p, I->pixel_offset);
         p.raw32_store &= I->format == AGX_FORMAT_I32 && I->mask == 15 &&
                          partition >= 0 && I->pixel_offset == 16u * partition;
      }
   }
   p.valid = stopped && stores_started;
   return p;
}

/* Fresh G17P BG constructors for two to four identical RGBA32 targets, four
 * R32 targets, and three or four RGBA8 targets have no PIXWAIT/PIXREL in the
 * shader and use loader release 0x200. They store ascending offsets, unlike
 * the ordinary four-target epilog's shared class-12 lifetime. */
static bool
agx3_g17p_multi_background_protocol(const agx_context *ctx)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B ||
       ctx->stage != MESA_SHADER_FRAGMENT || ctx->is_preamble ||
       !ctx->key->fs.ignore_tib_dependencies ||
       list_length(&ctx->blocks) != 1)
      return false;

   unsigned stores = 0, stride = 0;
   bool stopped = false;
   enum agx_format format = 0;
   BITSET_DECLARE(written_B, AGX_G17P_MAX_BYTES_PER_SAMPLE) = {0};
   agx_foreach_instr_global(ctx, I) {
      if (stopped || agx3_is_cf_op(I) || I->op == AGX_OPCODE_WAIT_PIX ||
          I->op == AGX_OPCODE_SIGNAL_PIX || I->op == AGX_OPCODE_LD_TILE)
         return false;
      if (I->op == AGX_OPCODE_STOP) {
         stopped = true;
      } else if (I->op == AGX_OPCODE_ST_TILE && ctx->key->g17p_mrt_bg) {
         /* Apple's BG builtins for other layouts store each target once, in
          * no required order, so only require disjoint whole-pixel stores.
          * A multisample clear stores each target through the coverage
          * register, as the one-target multisample background does. Five
          * and eight RGBA32Float clears store up to offset 112 the same
          * way. */
         const int width = agx3_tile_format_width(I->format);
         const unsigned size_B =
            width < 0 ? 0 : util_last_bit(I->mask) << width;
         struct agx3_tile_sample sample;
         if (!size_B || stores >= AGX_MAX_RTS ||
             I->pixel_offset + size_B > AGX_G17P_MAX_BYTES_PER_SAMPLE ||
             BITSET_TEST_RANGE(written_B, I->pixel_offset,
                               I->pixel_offset + size_B - 1) ||
             I->nr_srcs != 3 || I->nr_dests != 0 ||
             I->explicit_coords ||
             !agx3_tile_sample_operand(I->src[1], &sample) ||
             sample.reg != (ctx->key->fs.nr_samples > 1) ||
             !agx_is_null(I->src[2]))
            return false;
         BITSET_SET_RANGE(written_B, I->pixel_offset,
                          I->pixel_offset + size_B - 1);
         stores++;
      } else if (I->op == AGX_OPCODE_ST_TILE) {
         const bool rgba8 = I->format == AGX_FORMAT_U8NORM && I->mask == 15;
         const unsigned width =
            rgba8 ? 4 : I->mask == 15 ? 16 : I->mask == 1 ? 4 : 0;
         if (!width || (stores && width != stride) || stores >= 4 ||
             (!rgba8 && I->format != AGX_FORMAT_I32) || I->nr_srcs != 3 ||
             I->nr_dests != 0 || I->explicit_coords ||
             I->src[1].type != AGX_INDEX_IMMEDIATE ||
             I->src[1].value != AGX3_TILE_ALL_SAMPLES ||
             !agx_is_null(I->src[2]) || I->pixel_offset != stores * width)
            return false;
         if (stores && I->format != format)
            return false;
         format = I->format;
         stride = width;
         stores++;
      } else if (!agx_opcodes_info[I->op].can_eliminate &&
                 I->op != AGX_OPCODE_TEXTURE_SAMPLE &&
                 I->op != AGX_OPCODE_PRELOAD &&
                 I->op != AGX_OPCODE_COLLECT && I->op != AGX_OPCODE_SPLIT) {
         return false;
      }
   }
   if (ctx->key->g17p_mrt_bg)
      return stopped && stores >= 2;
   return stopped && ((stride == 16 && stores >= 2 && stores <= 4) ||
                      (stride == 4 && (stores == 4 ||
                       (stores == 3 && format == AGX_FORMAT_U8NORM))));
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

/* The formats each target selector of the multi-target EOT builtin is
 * derived for. The logical format, carried by the key, picks the selector;
 * the store format only has to be one that selector was derived for. */
static bool
agx3_g17p_mrt_eot_store_format_ok(enum agx_format format, uint8_t selector)
{
   switch (format) {
   case AGX_FORMAT_I8:
      return selector == 0x55;
   case AGX_FORMAT_U8NORM:
   case AGX_FORMAT_S8NORM:
   case AGX_FORMAT_RGB10A2:
      return selector == 0x75;
   case AGX_FORMAT_I16:
   case AGX_FORMAT_F16:
      return selector == 0x15;
   case AGX_FORMAT_U16NORM:
      return selector == 0x95 || selector == 0x15;
   case AGX_FORMAT_S16NORM:
      return selector == 0x95;
   case AGX_FORMAT_I32:
      return selector == 0x35 || selector == 0x75 || selector == 0x15;
   default:
      return false;
   }
}

/* Recognize EOT shaders with two to four raw32 targets, three to four RGBA8
 * targets, or two to eight targets with key-selected formats. The native
 * constructors use one store per dense attachment. Homogeneous targets have a
 * four-byte or sixteen-byte imageblock stride; key-selected targets take any
 * offset, which reaches the program through its offset uniform. */
static unsigned
agx3_g17p_multi_eot_count(const agx_context *ctx)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B ||
       ctx->stage != MESA_SHADER_COMPUTE || ctx->is_preamble ||
       ctx->key->no_stop ||
       (ctx->key->g17p_rt0_eot_multisample && !ctx->key->g17p_mrt_eot_count) ||
       list_length(&ctx->blocks) != 1)
      return false;

   const unsigned mixed_count = ctx->key->g17p_mrt_eot_count;
   if (mixed_count == 1 || mixed_count > AGX_MAX_RTS)
      return false;

   unsigned values[256];
   for (unsigned i = 0; i < ARRAY_SIZE(values); ++i)
      values[i] = UINT_MAX;
   unsigned x = UINT_MAX, y = UINT_MAX, stores = 0, stride = 0;
   bool stopped = false;
   agx_foreach_instr_global(ctx, I) {
      if (stopped)
         return false;
      if (I->op == AGX_OPCODE_STOP) {
         stopped = true;
      } else if (I->op == AGX_OPCODE_GET_SR || I->op == AGX_OPCODE_MOV_IMM) {
         if (I->nr_dests != 1 || I->nr_srcs ||
             !agx3_src_is_register(I->dest[0]) ||
             I->dest[0].size != AGX_SIZE_32 || agx_channels(I->dest[0]) != 1 ||
             I->dest[0].value >= ARRAY_SIZE(values) || I->dest[0].value % 2)
            return false;
         const unsigned reg = I->dest[0].value;
         /* A later constant must not replace either remembered coordinate.
          * Dword register spans must also be disjoint. */
         if ((x != UINT_MAX && reg < x + 2 && x < reg + 2) ||
             (y != UINT_MAX && reg < y + 2 && y < reg + 2))
            return false;
         values[reg] = UINT_MAX;
         if (I->op == AGX_OPCODE_MOV_IMM) {
            if (mixed_count ? I->imm >= AGX_G17P_MAX_BYTES_PER_SAMPLE
                            : (I->imm > 48 || I->imm % 4))
               return false;
            values[reg] = I->imm;
         } else if (I->sr == AGX_SR_THREAD_POSITION_IN_GRID_X && x == UINT_MAX) {
            x = reg;
         } else if (I->sr == AGX_SR_THREAD_POSITION_IN_GRID_Y && y == UINT_MAX) {
            y = reg;
         } else {
            return false;
         }
      } else if (I->op == AGX_OPCODE_BLOCK_IMAGE_STORE) {
         if (stores >= (mixed_count ? mixed_count : 4) || I->nr_dests ||
             I->nr_srcs != 4 ||
             (mixed_count
                 ? !agx3_g17p_mrt_eot_store_format_ok(
                      I->format, ctx->key->g17p_mrt_eot_selectors[stores])
                 : I->format != (ctx->key->g17p_rt0_eot_i32
                                    ? AGX_FORMAT_I32
                                    : AGX_FORMAT_U8NORM)) ||
             I->dim != AGX_DIM_2D ||
             !I->explicit_coords ||
             I->src[0].type != AGX_INDEX_IMMEDIATE || I->src[0].value != 0 ||
             I->src[0].size != AGX_SIZE_16 ||
             I->src[1].type != AGX_INDEX_IMMEDIATE || I->src[1].value != stores ||
             I->src[1].size != AGX_SIZE_16 ||
             !agx3_src_is_register(I->src[2]) || I->src[2].abs || I->src[2].neg ||
             I->src[2].size != AGX_SIZE_32 || agx_channels(I->src[2]) != 1 ||
             I->src[2].value >= ARRAY_SIZE(values) ||
             !agx3_src_is_register(I->src[3]) || I->src[3].abs || I->src[3].neg ||
             I->src[3].size != AGX_SIZE_32 || agx_channels(I->src[3]) != 2 ||
             x == UINT_MAX || y != x + 2 || I->src[3].value != x)
            return false;
         const unsigned offset = values[I->src[2].value];
         if (stores == 1)
            stride = offset;
         if (mixed_count ? offset == UINT_MAX
             : (stores == 0 && offset != 0) ||
                  (stores > 0 &&
                   ((stride != 4 && stride != 16) || offset != stores * stride)))
            return false;
         stores++;
      } else {
         return false;
      }
   }
   if (mixed_count)
      return stopped && stores == mixed_count ? stores : 0;
   /* A two-word packed-format pair has its own explicit format gate. Do
    * not let its four-byte stride enter through the new RGBA32 admission. */
   if (!ctx->key->g17p_rt0_eot_i32)
      return stopped && stride == 4 && stores >= 3 && stores <= 4 ? stores : 0;
   return stopped && ((stride == 16 && stores >= 2 && stores <= 4) ||
                      (stride == 4 && stores == 4)) ? stores : 0;
}

/* What a register currently holds, for the layered end-of-tile matcher. The
 * coordinate sources are tracked by role rather than by register number,
 * because register allocation reuses a system register's slot for a later tile
 * offset once the coordinate triple has been assembled. */
enum agx3_g17p_eot_role {
   AGX3_G17P_EOT_ROLE_NONE = 0,
   AGX3_G17P_EOT_ROLE_X,
   AGX3_G17P_EOT_ROLE_Y,
   AGX3_G17P_EOT_ROLE_LAYER,
};

/* The layered sibling of agx3_g17p_multi_eot_count().
 *
 * Apple's builtin end-of-tile factory emits a distinct program for 2D-ARRAY
 * attachments: it publishes the layer alongside the XY tile origin, so each
 * block store takes a three-component coordinate vector and AGX_DIM_2D_ARRAY.
 * The generic AGX3 packer has no encoding for an array block image store at
 * all -- it reports "no grounded AGX3 encoding for block_image_store ... dim=3"
 * -- so substituting the native text is the only way such a program can exist.
 * agx3_g17p_layered_eot_kind() already does this for one target; this
 * recognizes the same shape with two through eight.
 *
 * Registers are tracked rather than positionally fixed, because scheduling is
 * free to interleave the per-target offset constants with the stores.
 */
static unsigned
agx3_g17p_array_multi_eot_count(const agx_context *ctx, const char **why)
{
   const unsigned count = ctx->key->g17p_mrt_eot_count;

#define REJECT(reason)                                                         \
   do {                                                                        \
      if (why)                                                                 \
         *why = (reason);                                                      \
      return 0;                                                                \
   } while (0)

   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B ||
       ctx->stage != MESA_SHADER_COMPUTE || ctx->is_preamble ||
       ctx->key->no_stop || list_length(&ctx->blocks) != 1)
      REJECT("not a single-block G17P compute end-of-tile shader");

   if (count < 2 || count > AGX_MAX_RTS)
      REJECT("key does not name two to eight mixed targets");

   /* values[r]: the immediate r holds, or UINT_MAX. role[r]: which coordinate
    * source r currently carries. Both are ordinary forward dataflow, so a
    * register that is reused after its value is dead simply takes its new
    * meaning from the instruction that rewrites it. */
   unsigned values[256];
   uint8_t role[256];
   for (unsigned i = 0; i < ARRAY_SIZE(values); ++i) {
      values[i] = UINT_MAX;
      role[i] = AGX3_G17P_EOT_ROLE_NONE;
   }

   unsigned coords = UINT_MAX, stores = 0;
   bool stopped = false;

   agx_foreach_instr_global(ctx, I) {
      if (stopped)
         REJECT("instruction after stop");

      if (I->op == AGX_OPCODE_STOP) {
         stopped = true;
         continue;
      }

      if (I->op != AGX_OPCODE_BLOCK_IMAGE_STORE) {
         /* Every producer writes one whole aligned 32-bit register. */
         if (I->nr_dests != 1 || !agx3_src_is_register(I->dest[0]) ||
             I->dest[0].size != AGX_SIZE_32 || agx_channels(I->dest[0]) != 1 ||
             I->dest[0].value >= ARRAY_SIZE(values) || I->dest[0].value % 2)
            REJECT("unexpected destination shape");

         const unsigned reg = I->dest[0].value;

         values[reg] = UINT_MAX;
         role[reg] = AGX3_G17P_EOT_ROLE_NONE;

         switch (I->op) {
         case AGX_OPCODE_GET_SR:
            if (I->nr_srcs)
               REJECT("get_sr with sources");
            if (I->sr == AGX_SR_THREAD_POSITION_IN_GRID_X)
               role[reg] = AGX3_G17P_EOT_ROLE_X;
            else if (I->sr == AGX_SR_THREAD_POSITION_IN_GRID_Y)
               role[reg] = AGX3_G17P_EOT_ROLE_Y;
            else if (I->sr == AGX_SR_THREADGROUP_POSITION_IN_GRID_Z)
               role[reg] = AGX3_G17P_EOT_ROLE_LAYER;
            else
               REJECT("unexpected system register");
            break;

         case AGX_OPCODE_MOV_IMM:
            if (I->nr_srcs || I->imm >= AGX_G17P_MAX_BYTES_PER_SAMPLE)
               REJECT("immediate is not a tile offset");
            values[reg] = I->imm;
            break;

         case AGX_OPCODE_BITOP: {
            /* The register move that assembles the coordinate triple. */
            if ((I->truth_table & 0xfu) != AGX3_SEL_TT_MOV || I->saturate ||
                I->nr_srcs != 2 || !agx3_src_is_register(I->src[0]) ||
                I->src[0].abs || I->src[0].neg ||
                I->src[0].size != AGX_SIZE_32 ||
                agx_channels(I->src[0]) != 1 ||
                I->src[0].value >= ARRAY_SIZE(values) ||
                I->src[1].type != AGX_INDEX_IMMEDIATE || I->src[1].value != 0)
               REJECT("bitop is not a plain register move");

            /* A move carries both meanings of its source. */
            const unsigned src = I->src[0].value;
            values[reg] = values[src];
            role[reg] = role[src];
            break;
         }

         default:
            REJECT("unexpected opcode");
         }

         continue;
      }

      if (stores >= count || I->nr_dests || I->nr_srcs != 4 ||
          !agx3_g17p_mrt_eot_store_format_ok(
             I->format, ctx->key->g17p_mrt_eot_selectors[stores]) ||
          I->dim != AGX_DIM_2D_ARRAY || !I->explicit_coords ||
          I->src[0].type != AGX_INDEX_IMMEDIATE || I->src[0].value != 0 ||
          I->src[0].size != AGX_SIZE_16 ||
          I->src[1].type != AGX_INDEX_IMMEDIATE ||
          I->src[1].value != stores || I->src[1].size != AGX_SIZE_16 ||
          !agx3_src_is_register(I->src[2]) || I->src[2].abs || I->src[2].neg ||
          I->src[2].size != AGX_SIZE_32 || agx_channels(I->src[2]) != 1 ||
          I->src[2].value >= ARRAY_SIZE(values) ||
          !agx3_src_is_register(I->src[3]) || I->src[3].abs || I->src[3].neg ||
          I->src[3].size != AGX_SIZE_32 || agx_channels(I->src[3]) != 3 ||
          I->src[3].value + 4 >= ARRAY_SIZE(values))
         REJECT("store is not a native array end-of-tile store");

      /* Every target stores through one coordinate triple holding the grid X,
       * grid Y and layer, in that order. Checking the roles here also rejects
       * any program that overwrote one of them between two stores. */
      const unsigned base = I->src[3].value;
      if (coords == UINT_MAX)
         coords = base;

      if (base != coords || role[base] != AGX3_G17P_EOT_ROLE_X ||
          role[base + 2] != AGX3_G17P_EOT_ROLE_Y ||
          role[base + 4] != AGX3_G17P_EOT_ROLE_LAYER)
         REJECT("coordinate triple is not (grid x, grid y, layer)");

      /* The offset reaches the program through its offset uniform, so it only
       * has to be a tile offset this shader established. */
      if (values[I->src[2].value] == UINT_MAX)
         REJECT("store offset is not a known immediate");

      stores++;
   }

   if (!stopped || stores != count)
      REJECT("target count does not match the key");

#undef REJECT

   return stores;
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
   agx3_trace_shader_name =
      (ctx && ctx->nir && ctx->nir->info.name) ? ctx->nir->info.name : NULL;
   const enum agx_isa_encoding enc = ctx->key->dev.encoding;
   const unsigned nslots = (enc == AGX_ISA_ENCODING_AGX3C) ? 8 : 6;
   const unsigned initial_size = emission->size;
   ctx->agx3_staging_known = true;
   ctx->agx3_fltnorm_start = false;

   /* A layered mixed-format end-of-tile program. Apple's array builtin differs
    * from the 2D one only in publishing the layer wire, and its per-target
    * IMGWBLK lane-class selector is the sole format-dependent byte, exactly as
    * in the 2D family. A multisample array attachment (BTNF dimension 8)
    * differs from the array text only in those selectors, whose low nibble
    * becomes the multisample dimension class 0 -- the same rule the one-target
    * layered program applies above. Unlike the 2D family it needs no
    * PACK_IMGWIR wire-width change; see agx3_g17p_array_eot.h. */
   const char *array_eot_why = NULL;
   const unsigned array_eot_count =
      agx3_g17p_array_multi_eot_count(ctx, &array_eot_why);
   if (array_eot_count) {
      const uint8_t *text = agx3_g17p_array_eot[array_eot_count].text;
      const unsigned size = agx3_g17p_array_eot[array_eot_count].size;
      const bool multisample = ctx->key->g17p_rt0_eot_multisample;
      uint8_t program[sizeof(agx3_g17p_array_eot_8)];

      assert(size <= sizeof(program));
      memcpy(program, text, size);

      for (unsigned rt = 0; rt < array_eot_count; ++rt) {
         const uint8_t selector = ctx->key->g17p_mrt_eot_selectors[rt];
         program[agx3_g17p_array_eot[array_eot_count].selector[rt]] =
            multisample ? (selector & 0xf0) : selector;
      }

      /* The matched synthetic graph used registers that are absent from the
       * replacement, so its register count does not describe what will run.
       * Apple's metadata for the array builtin asks for five GPR dwords and
       * sixteen staging bytes per target. */
      ctx->max_reg = 9;
      ctx->agx3_staging_halfs = MAX2(ctx->agx3_staging_halfs,
                                     8 * array_eot_count);

      /* Six or more targets exceed agx3_append's 8-bit length. */
      memcpy(util_dynarray_grow_bytes(emission, 1, size), program, size);
      return true;
   }

   if (array_eot_why &&
       debug_get_bool_option("AGX_G17P_ARRAY_EOT_TRACE", false)) {
      fprintf(stderr, "G17P array EOT not matched: %s (key count=%u)\n",
              array_eot_why, ctx->key->g17p_mrt_eot_count);
      agx_print_shader(ctx, stderr);
   }

   const unsigned multi_eot_count = agx3_g17p_multi_eot_count(ctx);
   if (multi_eot_count) {
      static_assert(sizeof(agx3_g17p_quad_eot) == 208, "native quad EOT");
      ctx->max_reg = MAX2(ctx->max_reg, 7);
      ctx->agx3_staging_halfs = MAX2(ctx->agx3_staging_halfs, 8 * multi_eot_count);
      if (ctx->key->g17p_mrt_eot_count) {
         /* The builtin for targets of any formats is the raw32 program with
          * each target's IMGWBLK selector replaced. Apple's factory derives
          * each selector from that target's format alone and changes no
          * other byte.
          *
          * Multisample attachments (BTNF texture type 4) change three bytes
          * per target, exactly as the one-target multisample builtin does:
          * the wire width byte of the target's PACK_IMGWIR gains 2, and its
          * IMGWBLK clears bit 7 of byte 8 and replaces the selector's low
          * nibble 5 with the multisample dimension class 2. metal-nt
          * -arch applegpu_g17p builtin EOT BTNF witnesses of two to eight
          * RGBA8 targets in one partition, texture types 2 and 4, differ in
          * exactly those bytes. */
         static const struct {
            const uint8_t *text;
            uint16_t imgwblk[AGX_MAX_RTS];
            uint16_t imgwir_wire[AGX_MAX_RTS];
         } templates[] = {
            [2] = {agx3_g17p_dual_raw32_eot, {92, 104}, {43, 63}},
            [3] = {agx3_g17p_triple_raw32_eot, {124, 136, 148},
                   {43, 63, 95}},
            [4] = {agx3_g17p_quad_eot, {156, 168, 180, 192},
                   {43, 63, 95, 127}},
            [5] = {agx3_g17p_penta_raw32_eot, {188, 200, 212, 224, 236},
                   {43, 63, 95, 127, 159}},
            [6] = {agx3_g17p_hexa_raw32_eot, {220, 232, 244, 256, 268, 280},
                   {43, 63, 95, 127, 159, 191}},
            [7] = {agx3_g17p_hepta_raw32_eot,
                   {60, 264, 276, 288, 300, 312, 324},
                   {43, 75, 107, 139, 171, 203, 235}},
            [8] = {agx3_g17p_octa_raw32_eot,
                   {92, 104, 308, 320, 332, 344, 356, 368},
                   {43, 63, 119, 151, 183, 215, 247, 279}},
         };
         static_assert(sizeof(agx3_g17p_octa_raw32_eot) == 32 + 44 * 8,
                       "native octa EOT");
         const size_t size = 32 + 44 * multi_eot_count;
         const bool multisample = ctx->key->g17p_rt0_eot_multisample;
         uint8_t program[sizeof(agx3_g17p_octa_raw32_eot)];
         memcpy(program, templates[multi_eot_count].text, size);
         for (unsigned rt = 0; rt < multi_eot_count; ++rt) {
            const unsigned imgwblk = templates[multi_eot_count].imgwblk[rt];
            uint8_t selector = ctx->key->g17p_mrt_eot_selectors[rt];
            if (multisample) {
               program[templates[multi_eot_count].imgwir_wire[rt]] |= 0x02;
               program[imgwblk + 8] &= 0x7f;
               selector = (selector & 0xf0) | 0x02;
            }
            /* Byte 9 of the target's 12-byte IMGWBLK. */
            program[imgwblk + 9] = selector;
         }
         /* Six or more targets exceed agx3_append's 8-bit length. */
         memcpy(util_dynarray_grow_bytes(emission, 1, size), program, size);
         return true;
      }
      if (!ctx->key->g17p_rt0_eot_i32) {
         return multi_eot_count == 3
            ? agx3_append(emission, agx3_g17p_rgba8_3_eot,
                            sizeof(agx3_g17p_rgba8_3_eot))
            : agx3_append(emission, agx3_g17p_rgba8_4_eot,
                            sizeof(agx3_g17p_rgba8_4_eot));
      }
      if (multi_eot_count == 2)
         return agx3_append(emission, agx3_g17p_dual_raw32_eot,
                            sizeof(agx3_g17p_dual_raw32_eot));
      if (multi_eot_count == 3)
         return agx3_append(emission, agx3_g17p_triple_raw32_eot,
                            sizeof(agx3_g17p_triple_raw32_eot));
      return agx3_append(emission, agx3_g17p_quad_eot, sizeof(agx3_g17p_quad_eot));
   }

   const bool r11g11b10_mrt_eot =
      agx3_is_g17p_dual_word_mrt_eot(ctx, enc, AGX_FORMAT_I32);
   if (r11g11b10_mrt_eot ||
       agx3_is_g17p_dual_word_mrt_eot(ctx, enc, AGX_FORMAT_U8NORM)) {
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
      uint8_t g17p_r11g11b10_mrt_eot[sizeof(g17p_rgba8_mrt_eot)];
      ctx->max_reg = MAX2(ctx->max_reg, 7);
      ctx->agx3_staging_halfs = MAX2(ctx->agx3_staging_halfs, 16);
      if (r11g11b10_mrt_eot) {
         /* The exact dual-target control/coordinate program is grounded by
          * the native RGBA8 pair above. Each attachment is one tile word;
          * packed R11G11B10 selects the independently grounded scalar-I32
          * imgwblk tail (0x35) for both stores. No other bytes encode the
          * logical format, which remains the PBE descriptor's responsibility. */
         memcpy(g17p_r11g11b10_mrt_eot, g17p_rgba8_mrt_eot,
                sizeof(g17p_rgba8_mrt_eot));
         g17p_r11g11b10_mrt_eot[101] = 0x35;
         g17p_r11g11b10_mrt_eot[113] = 0x35;
         return agx3_append(emission, g17p_r11g11b10_mrt_eot,
                            sizeof(g17p_r11g11b10_mrt_eot));
      }

      return agx3_append(emission, g17p_rgba8_mrt_eot,
                         sizeof(g17p_rgba8_mrt_eot));
   }

   if (agx3_is_g17p_wboit_mrt_eot(ctx, enc)) {
      /* Fresh applegpu_g17p builtin EOT for RGBA16F partition0 plus R8 in a
       * distinct partition. The text is invariant for every nonzero second
       * partition tested. SHA256 of these exact 238 bytes:
       * eef32d3c9b7bca4de2efba861662e32c34f49af5877dabce0ad9da5cd50d5173.
       *
       * Requesting partition0 for both targets instead produces the old
       * 120-byte text. That same-partition specialization stores RT0 but does
       * not select the separate RT1 tile partition used by WBOIT.
       * Metadata: common0=4, D=32; ESL pointer2 plus kind0 count3 at dest2;
       * qword bindpoint0x32 at dest0; kind5 attachment0/1 at dest0/2; mask3.
       */
      static const uint8_t g17p_wboit_mrt_eot[] = {
         0x1c,0x9c,0x10,0x06,0x24,0xa8,0x14,0x06,0x0c,0x9d,0x10,0x26,
         0x24,0xa9,0x10,0x26,0x9f,0x10,0x54,0x02,0x03,0x04,0x14,0x00,
         0x50,0x26,0x0a,0x00,0x9f,0x20,0x54,0x00,0x03,0x00,0x10,0x00,
         0x50,0x26,0x0a,0x00,0x97,0x05,0x54,0x04,0x00,0x08,0x00,0x00,
         0xd0,0x40,0x10,0x00,0x0b,0x82,0x21,0x04,0x1b,0x80,0x21,0x04,
         0x23,0x00,0x20,0x00,0x57,0x10,0x54,0x00,0x00,0x00,0x00,0x80,
         0xa8,0x15,0x00,0x00,0x06,0x04,0x24,0x99,0x15,0x06,0x00,0x01,
         0x00,0x00,0x0a,0x84,0x33,0x04,0x06,0x00,0x17,0x28,0x02,0x00,
         0x8f,0x04,0x54,0x22,0x0f,0x01,0x54,0x60,0x00,0x00,0x00,0x00,
         0x00,0x00,0x23,0x00,0x20,0x01,0x0f,0x05,0x54,0x1a,0x34,0x98,
         0x11,0x06,0x00,0x01,0x00,0x00,0x9f,0x01,0x54,0x05,0x03,0x02,
         0x14,0x00,0x15,0x04,0x0a,0x84,0x23,0x86,0x06,0x00,0x0f,0x28,
         0x02,0x00,0x32,0x04,0x35,0x06,0x26,0x81,0x0f,0x08,0x22,0x80,
         0x9f,0x01,0x54,0x04,0x02,0x08,0x18,0x00,0x17,0x05,0x8f,0x04,
         0x54,0x22,0x0f,0x00,0x54,0xcc,0xff,0xff,0xff,0xff,0xff,0x00,
         0x0f,0x06,0x04,0x02,0x00,0x00,0x0f,0x00,0x54,0x94,0xff,0xff,
         0xff,0xff,0xff,0x00,0x0f,0x06,0x04,0x02,0x00,0x00,0x97,0x05,
         0x54,0x0c,0x00,0x10,0x00,0x00,0xd0,0x40,0x10,0x00,0x4b,0x02,
         0x29,0x04,0x5b,0x00,0x29,0x04,0x57,0x10,0x54,0x08,0x00,0x10,
         0x00,0x80,0xa8,0x75,0x00,0x00,0x0e,0x00,0x00,0x00,
      };
      static_assert(sizeof(g17p_wboit_mrt_eot) == 238,
                    "WBOIT MRT EOT native length");
      /* Native metadata has common field0=4, D=32: four 32-bit GPRs and 16
       * staging halves. */
      ctx->max_reg = MAX2(ctx->max_reg, 7);
      ctx->agx3_staging_halfs = MAX2(ctx->agx3_staging_halfs, 16);
      return agx3_append(emission, g17p_wboit_mrt_eot,
                         sizeof(g17p_wboit_mrt_eot));
   }

   const enum agx3_g17p_rt0_eot_kind layered_eot =
      agx3_g17p_layered_eot_kind(ctx, enc);
   if (layered_eot != AGX3_G17P_RT0_EOT_NONE) {
      /* Apple's builtin end-of-tile factory emits one 84-byte program for a
       * 2D-array attachment of every format (metal-nt -arch applegpu_g17p,
       * BTNF texture type 3), differing only in the IMGWBLK lane-class
       * selector at byte 77. The 2D multisample-array builtin (texture type
       * 8) clears the selector's dimension nibble there and changes nothing
       * else. */
      uint8_t program[] = {
         0x1c,0x9c,0x10,0x06,0x34,0xa8,0x10,0x06,
         0x0c,0x9d,0x10,0x26,0x24,0xa9,0x14,0x26,
         0x24,0xc3,0x10,0x46,
         0x9f,0x10,0x54,0x00,0x00,0x04,0x18,0x00,0x50,0x06,0x0b,0x00,
         0x9f,0x20,0x54,0x02,0x00,0x00,0x14,0x00,0x50,0x06,0x0b,0x00,
         /* Native array wire: layer first, followed by the XY tile origin. */
         0x97,0x45,0x54,0x04,0x00,0x08,0x10,0x00,0xf0,0xc0,0x00,0x00,
         0x97,0x05,0x54,0x06,0x00,0x08,0x00,0x00,0xd0,0x40,0x10,0x00,
         0x57,0x10,0x54,0x00,0x00,0x00,0x00,0x80,0x28,0x75,0x00,0x00,
         0x0e,0x00,0x00,0x00,
      };
      static_assert(sizeof(program) == 84, "layered EOT native length");
      static const uint8_t selectors[] = {
         [AGX3_G17P_RT0_EOT_SELECTOR_75] = 0x75,
         [AGX3_G17P_RT0_EOT_SELECTOR_15] = 0x15,
         [AGX3_G17P_RT0_EOT_SELECTOR_35] = 0x35,
         [AGX3_G17P_RT0_EOT_SELECTOR_55] = 0x55,
         [AGX3_G17P_RT0_EOT_NORM16] = 0x95,
      };
      program[77] = ctx->key->g17p_rt0_eot_multisample
                       ? selectors[layered_eot] & 0xf0
                       : selectors[layered_eot];
      /* The matched synthetic graph used registers that are absent from the
       * replacement. The native metadata requires four GPR dwords. */
      ctx->max_reg = 7;
      ctx->agx3_staging_halfs = MAX2(ctx->agx3_staging_halfs, 8);
      return agx3_append(emission, program, sizeof(program));
   }

   const enum agx3_g17p_rt0_eot_kind rt0_eot =
      agx3_g17p_rt0_eot_kind(ctx, enc);
   if (rt0_eot != AGX3_G17P_RT0_EOT_NONE) {
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
      /* Apple's builtin end-of-tile factory emits a second program for every
       * one of these five selectors when the attachment's texture type is
       * 2D multisample (dimension 4) instead of 2D (dimension 2). The two
       * forms differ in exactly three bytes: the PACK_IMGWIR wire width
       * (field 5, 4 -> 6) and the IMGWBLK dimension class (field 30,
       * 1 -> 2, which straddles bytes 60 and 61). Nothing else moves --
       * no extra instruction computes a sample index, because the block
       * store covers every sample of the tile. Witness generation and the
       * round-tripped field vectors are in isa/README-metal-witness.md.
       */
      static const uint8_t g17p_rt0_eot_ms[][68] = {{
         /* 0x75 -> 0x72: U8NORM/S8NORM/RGB10A2, the RT0 baseline. */
         0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06,
         0x0c, 0x9d, 0x10, 0x26, 0x24, 0xa9, 0x10, 0x26,
         0x9f, 0x10, 0x54, 0x00, 0x00, 0x04, 0x14, 0x00,
         0x50, 0x06, 0x0b, 0x00, 0x9f, 0x20, 0x54, 0x02,
         0x00, 0x00, 0x10, 0x00, 0x50, 0x06, 0x0b, 0x00,
         0x97, 0x05, 0x54, 0x06, 0x00, 0x08, 0x00, 0x00,
         0xd0, 0x40, 0x10, 0x00, 0x57, 0x10, 0x54, 0x00,
         0x00, 0x00, 0x00, 0x80, 0x28, 0x72, 0x00, 0x00,
         0x0e, 0x00, 0x00, 0x00,
      }, {
         /* 0x15 -> 0x12: I16/F16, including RGB565. */
         0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06,
         0x0c, 0x9d, 0x10, 0x26, 0x24, 0xa9, 0x10, 0x26,
         0x9f, 0x10, 0x54, 0x00, 0x00, 0x04, 0x14, 0x00,
         0x50, 0x06, 0x0b, 0x00, 0x9f, 0x20, 0x54, 0x02,
         0x00, 0x00, 0x10, 0x00, 0x50, 0x06, 0x0b, 0x00,
         0x97, 0x05, 0x54, 0x06, 0x00, 0x08, 0x00, 0x00,
         0xd0, 0x40, 0x10, 0x00, 0x57, 0x10, 0x54, 0x00,
         0x00, 0x00, 0x00, 0x80, 0x28, 0x12, 0x00, 0x00,
         0x0e, 0x00, 0x00, 0x00,
      }, {
         /* 0x35 -> 0x32: I32, including the four-word keys. */
         0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06,
         0x0c, 0x9d, 0x10, 0x26, 0x24, 0xa9, 0x10, 0x26,
         0x9f, 0x10, 0x54, 0x00, 0x00, 0x04, 0x14, 0x00,
         0x50, 0x06, 0x0b, 0x00, 0x9f, 0x20, 0x54, 0x02,
         0x00, 0x00, 0x10, 0x00, 0x50, 0x06, 0x0b, 0x00,
         0x97, 0x05, 0x54, 0x06, 0x00, 0x08, 0x00, 0x00,
         0xd0, 0x40, 0x10, 0x00, 0x57, 0x10, 0x54, 0x00,
         0x00, 0x00, 0x00, 0x80, 0x28, 0x32, 0x00, 0x00,
         0x0e, 0x00, 0x00, 0x00,
      }, {
         /* 0x55 -> 0x52: the one-word I8 integer keys. */
         0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06,
         0x0c, 0x9d, 0x10, 0x26, 0x24, 0xa9, 0x10, 0x26,
         0x9f, 0x10, 0x54, 0x00, 0x00, 0x04, 0x14, 0x00,
         0x50, 0x06, 0x0b, 0x00, 0x9f, 0x20, 0x54, 0x02,
         0x00, 0x00, 0x10, 0x00, 0x50, 0x06, 0x0b, 0x00,
         0x97, 0x05, 0x54, 0x06, 0x00, 0x08, 0x00, 0x00,
         0xd0, 0x40, 0x10, 0x00, 0x57, 0x10, 0x54, 0x00,
         0x00, 0x00, 0x00, 0x80, 0x28, 0x52, 0x00, 0x00,
         0x0e, 0x00, 0x00, 0x00,
      }, {
         /* 0x95 -> 0x92: the normalized-16 keys. */
         0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06,
         0x0c, 0x9d, 0x10, 0x26, 0x24, 0xa9, 0x10, 0x26,
         0x9f, 0x10, 0x54, 0x00, 0x00, 0x04, 0x14, 0x00,
         0x50, 0x06, 0x0b, 0x00, 0x9f, 0x20, 0x54, 0x02,
         0x00, 0x00, 0x10, 0x00, 0x50, 0x06, 0x0b, 0x00,
         0x97, 0x05, 0x54, 0x06, 0x00, 0x08, 0x00, 0x00,
         0xd0, 0x40, 0x10, 0x00, 0x57, 0x10, 0x54, 0x00,
         0x00, 0x00, 0x00, 0x80, 0x28, 0x92, 0x00, 0x00,
         0x0e, 0x00, 0x00, 0x00,
      }};
      static_assert(sizeof(g17p_rt0_eot_ms) == sizeof(g17p_rt0_eot),
                    "one multisample program per selector");

      /* Native metadata common0=3 means six GPR halves; retain the existing
       * conservative eight-half allocation. Fields38/42 are D=16, requiring
       * eight staging halves independently. Main68 pads to128: LDSHDR f1=1.
       * The multisample forms have the same length, register and staging
       * metadata, so the launch program is unchanged. */
      ctx->max_reg = MAX2(ctx->max_reg, 7);
      ctx->agx3_staging_halfs = MAX2(ctx->agx3_staging_halfs, 8);
      const unsigned index = rt0_eot - AGX3_G17P_RT0_EOT_SELECTOR_75;
      const uint8_t *program = ctx->key->g17p_rt0_eot_multisample
                                  ? g17p_rt0_eot_ms[index]
                                  : g17p_rt0_eot[index];
      return agx3_append(emission, program, sizeof(g17p_rt0_eot[index]));
   }

   struct util_dynarray fixups;
   uint16_t outstanding = 0;
   /* Subset of `outstanding` whose generations belong to texture samples. */
   uint16_t sample_slots_live = 0;
   /* Result slot of the sample currently occupying each sampler-private
    * staging group (half-registers 0, 8 and 16). A group is free once a
    * wait has retired that slot; see agx3_emit_texture_sample. */
   uint8_t stage_group_slot[3] = {0, 0, 0};
   uint8_t next_uvs_slot = 1;
   unsigned texture_ops_left = 0;
   unsigned tile_loads_left = 0;
   unsigned tile_loads_seen = 0;
   unsigned tile_stores_left = 0;
   unsigned image_writes_left = 0;
   unsigned texture_samples_left = 0;
   unsigned sample_group_members_left = 0;
   unsigned device_loads_left = 0;
   bool has_coverage_write = false;
   bool has_memory_write = false;
   /* The last pixel release of a program-ending fragment part, when no
    * colour store, coverage update, acquisition or branch follows it. */
   const agx_instr *g17p_final_release = NULL;
   bool has_zs_emit = false;
   unsigned zs_staging_halfs = 0;
   bool g17p_fragment_helper_protocol =
      enc == AGX_ISA_ENCODING_AGX3B &&
      ctx->stage == MESA_SHADER_FRAGMENT && ctx->nir &&
      (ctx->nir->info.fs.needs_coarse_quad_helper_invocations ||
       ctx->nir->info.fs.needs_full_quad_helper_invocations);
   bool uniform_half_used[512] = {false};
   for (unsigned h = 0; h < MIN2(ctx->key->reserved_preamble, 512); ++h)
      uniform_half_used[h] = true;
   if (ctx->out) {
      const unsigned base = ctx->out->rodata.base_uniform;
      for (unsigned h = 0; h < ctx->out->rodata.size_16 && base + h < 512; ++h)
         uniform_half_used[base + h] = true;
   }
   bool has_dynamic_cube_sample = false;
   bool has_packed_array_layer = false;
   unsigned observed_max_reg = ctx->max_reg;
   unsigned count = 0;
   bool cprog_imports = false;

   assert(enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C);
   util_dynarray_init(&fixups, ctx);

   agx_foreach_block(ctx, scan_block) {
      agx_foreach_instr_in_block(scan_block, scan_I) {
         /* An imported/read half is not padding either. A scalar export
          * must not overwrite it merely because no shader store defines it. */
         agx_foreach_src(scan_I, s) {
            const agx_index src = scan_I->src[s];
            if (src.type == AGX_INDEX_UNIFORM) {
               for (unsigned h = 0; h < agx_index_size_16(src) &&
                                    src.value + h < 512; ++h)
                  uniform_half_used[src.value + h] = true;
            }
         }
         cprog_imports |= scan_I->op == AGX_OPCODE_LOAD_CPROG_AGX3;
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
         if ((scan_I->op == AGX_OPCODE_TEXTURE_LOAD ||
              scan_I->op == AGX_OPCODE_IMAGE_LOAD) &&
             scan_I->dim == AGX_DIM_2D_MS_ARRAY)
            has_packed_array_layer = true;
         if (scan_I->op == AGX_OPCODE_TEXTURE_SAMPLE &&
             ((scan_I->dim == AGX_DIM_2D_ARRAY &&
               ((scan_I->offset && !scan_I->offset_xyz) ||
                (scan_I->shadow &&
                 scan_I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS_MIN))) ||
              agx3_is_staged_3d_bias_min(scan_I)))
            has_packed_array_layer = true;
         if (scan_I->op == AGX_OPCODE_ST_TILE)
            tile_stores_left++;
         if (scan_I->op == AGX_OPCODE_SIGNAL_PIX)
            g17p_final_release = scan_I;
         else if (scan_I->op == AGX_OPCODE_WAIT_PIX ||
                  scan_I->op == AGX_OPCODE_SAMPLE_MASK_AGX3 ||
                  agx3_is_cf_op(scan_I))
            g17p_final_release = NULL;
         if (scan_I->op == AGX_OPCODE_SAMPLE_MASK_AGX3 ||
             scan_I->op == AGX_OPCODE_ZS_EMIT_AGX3 ||
             scan_I->op == AGX_OPCODE_ZS_EMIT)
            has_coverage_write = true;
         if (scan_I->op == AGX_OPCODE_ZS_EMIT ||
             scan_I->op == AGX_OPCODE_ZS_EMIT_AGX3)
            has_zs_emit = true;
         if (scan_I->op == AGX_OPCODE_ZS_EMIT_AGX3)
            zs_staging_halfs += (scan_I->zs & 1) ? 4 : 2;
         if (scan_I->op == AGX_OPCODE_ATOMIC ||
             scan_I->op == AGX_OPCODE_DEVICE_STORE ||
             scan_I->op == AGX_OPCODE_IMAGE_WRITE)
            has_memory_write = true;
         if (scan_I->op == AGX_OPCODE_DEVICE_LOAD)
            device_loads_left++;
         if (scan_I->op == AGX_OPCODE_IMAGE_WRITE)
            image_writes_left++;
         if (scan_I->op == AGX_OPCODE_TEXTURE_SAMPLE) {
            texture_samples_left++;
            /* Explicit gradients carry their own staged derivatives and sit
             * outside f156 sample groups. Native AUTO/GRAD/AUTO groups the
             * two AUTO samples across the gradient without marking it. */
            if (scan_I->lod_mode != AGX_LOD_MODE_LOD_GRAD &&
                scan_I->lod_mode != AGX_LOD_MODE_LOD_GRAD_MIN)
               sample_group_members_left++;
         }
         if (scan_I->op == AGX_OPCODE_TEXTURE_SAMPLE &&
             (scan_I->dim == AGX_DIM_CUBE ||
              scan_I->dim == AGX_DIM_CUBE_ARRAY)) {
            /* A register level or bias outlives CEM, so it cannot share the
             * fixed immediate-corpus scratch. A cube-array layer is prepacked
             * beside it. */
            const bool bias_min =
               scan_I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS_MIN;
            const bool register_lod = bias_min ||
               ((scan_I->lod_mode == AGX_LOD_MODE_LOD_MIN ||
                 scan_I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS) &&
                scan_I->src[1].type != AGX_INDEX_IMMEDIATE);
            if (register_lod ||
                scan_I->src[3].type != AGX_INDEX_IMMEDIATE ||
                scan_I->src[4].type != AGX_INDEX_IMMEDIATE)
               has_dynamic_cube_sample = true;
            /* So is a cube face packed beside a bias and minimum. */
            if (register_lod &&
                (scan_I->dim == AGX_DIM_CUBE_ARRAY || bias_min))
               has_packed_array_layer = true;
         }
         agx_foreach_reg_dest(scan_I, d) {
            observed_max_reg = MAX2(
               observed_max_reg,
               scan_I->dest[d].value +
                  agx_index_size_16(scan_I->dest[d]) - 1);
         }
         agx_foreach_reg_src(scan_I, s) {
            observed_max_reg = MAX2(
               observed_max_reg,
               scan_I->src[s].value + agx_index_size_16(scan_I->src[s]) - 1);
         }
         /* Native keeps the fragment-input start marker clear in every
          * sampled or derivative fragment in the corpus, including explicit
          * LOD samples.  This conservatively preserves helper-capable lanes;
          * plain varying-only fragments retain the marker below. */
         if (scan_I->op == AGX_OPCODE_TEXTURE_SAMPLE ||
             (scan_I->op == AGX_OPCODE_ITRCOORDS_AGX3 &&
              scan_I->interpolation == AGX_INTERPOLATION_SAMPLE) ||
             scan_I->op == AGX_OPCODE_DFDX ||
             scan_I->op == AGX_OPCODE_DFDY)
            g17p_fragment_helper_protocol = true;
         if ((scan_I->op == AGX_OPCODE_UNIFORM_FADD_AGX3 ||
              scan_I->op == AGX_OPCODE_UNIFORM_FMUL_AGX3 ||
              scan_I->op == AGX_OPCODE_UNIFORM_FFMA_AGX3) &&
             scan_I->imm <= ARRAY_SIZE(uniform_half_used) - 2) {
            uniform_half_used[scan_I->imm] = true;
            uniform_half_used[scan_I->imm + 1] = true;
         }
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

   const uint16_t entry_wait = cprog_imports ? (1u << 5) : 0;
   if (cprog_imports) {
      if (!ctx->is_preamble || !ctx->agx3_cprog_inputs_lowered ||
          enc != AGX_ISA_ENCODING_AGX3B || !ctx->out ||
          !ctx->out->agx3_preamble_inputs.count)
         goto fail;
      outstanding = entry_wait;
      for (unsigned h = 0; h < 512; ++h)
         uniform_half_used[h] |=
            !!(ctx->out->agx3_preamble_inputs.halves[h / 64] &
               (UINT64_C(1) << (h % 64)));
   }

   /* Fresh G17P fragment witnesses with two/three ordinary SMPs clear f156 on
    * every earlier group member and set it only on the final member, including
    * an all-explicit-LEVEL group. Explicit gradients are outside the group.
    * Other stages and AGX3C keep their existing per-op encodings. */
   const bool g17p_sample_group = !agx3_g17p_no_sample_group() &&
                                  enc == AGX_ISA_ENCODING_AGX3B &&
                                  ctx->stage == MESA_SHADER_FRAGMENT &&
                                  sample_group_members_left > 1;

   /* CEM axis 0 writes a magnitude/face pair, while axes 1 and 2 each write
    * one 32-bit coordinate. Native cube_compare_zero uses half8 for axis0
    * and half10 for PACK_SMPIR's face; two_cubes uses half24 and half26.
    * Reserve the implicit face word too, or axis1 overwrites it.
    * The destinations used to be hardcoded to
    * r0/r7/r2 after RA, which could overwrite live control-flow state or a
    * register descriptor (vkQuake's sky shader puts its texture handle in
    * r2:r3). Reserve one reusable bank immediately above RA's allocation and
    * publish those exact registers through every CEM/reciprocal/finalizer
    * field. The CEM destination field reaches half190. */
   uint16_t cem_scratch[3] = {0};
   if (has_dynamic_cube_sample) {
      const unsigned base = (observed_max_reg + 2) & ~1u;
      if (base + 6 > 190)
         goto fail;
      cem_scratch[0] = base;
      cem_scratch[1] = base + 4;
      cem_scratch[2] = base + 6;
      ctx->max_reg = base + 7;
   }

   /* MS-array reads and runtime-offset array gathers merge a normalized
    * layer with another operand. Reserve one transient word above every
    * live RA operand and CEM scratch; successive samples reuse it. */
   uint16_t ms_array_scratch = 0;
   if (has_packed_array_layer) {
      const unsigned base = (MAX2(observed_max_reg, ctx->max_reg) + 2) & ~1u;
      if (base > 190)
         goto fail;
      ms_array_scratch = base;
      ctx->max_reg = base + 1;
   }

   /* Apple's sampled G17P fragment objects use a distinct constant program
    * even when the texture and sampler operands are compact/immediate and no
    * sampler heap is involved.  Publish the semantic instruction class for
    * the linker instead of trying to reconstruct it from resource metadata. */
   if (ctx->out) {
      ctx->out->g17p_sampled_fragment |=
         enc == AGX_ISA_ENCODING_AGX3B &&
         ctx->stage == MESA_SHADER_FRAGMENT && texture_samples_left != 0;
   }

   /* The valid G17P colour-only corpus has one class-12 wait/store bracket.
    * Multi-RT shaders instead select read/write classes by sorted tile
    * partition and release each partition in descending order. */
   const struct agx3_g17p_mrt_protocol g17p_mrt =
      agx3_g17p_mrt_protocol(ctx);
   const bool g17p_multi_background = agx3_g17p_multi_background_protocol(ctx);
   const unsigned g17p_msaa_background_targets =
      agx3_g17p_sample_store_protocol(ctx, false);
   const bool g17p_msaa_background_stores = g17p_msaa_background_targets > 0;
   const bool g17p_sample_colour_stores =
      tile_stores_left > 1 && agx3_g17p_sample_store_protocol(ctx, true);
   const bool g17p_colour_protocol =
      (enc == AGX_ISA_ENCODING_AGX3B &&
       ctx->stage == MESA_SHADER_FRAGMENT &&
       !ctx->key->fs.ignore_tib_dependencies &&
       (tile_stores_left == 1 || g17p_sample_colour_stores) &&
       tile_loads_left == 0) || g17p_mrt.valid;

   /* Multiple colour stores need an attachment-specific pixel transaction.
    * Emitting individual tile operations without a proven partition graph
    * produces a syntactically valid shader that writes only partition zero. */
   if (enc == AGX_ISA_ENCODING_AGX3B &&
       ctx->stage == MESA_SHADER_FRAGMENT &&
       tile_stores_left > 1 && !g17p_mrt.valid && !g17p_multi_background &&
       !g17p_msaa_background_stores && !g17p_sample_colour_stores)
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
                     ((P->format == AGX_FORMAT_RGB10A2 ||
                       P->format == AGX_FORMAT_RGB9E5) && P->mask == 1)) ||
                    P->pixel_offset || P->explicit_coords)
                  g17p_blend_protocol = false;
               phase = load ? 2 : 4;
            } else if (P->op == AGX_OPCODE_SIGNAL_PIX) {
               g17p_blend_protocol = false;
            }
         }
      }
      g17p_blend_protocol &= phase == 4;
   }
   /* A native fragment program ends with a final release: the colour store's
    * class 12 with 0x200, or, with no colour store, its late depth class.
    * metal-nt -arch applegpu_g17p fragments with no colour writes end in
    * pixwait(1), depthf, 070254010200; the same atomic fragment returning a
    * colour releases class 1 plainly and ends in 0702540c0200. */
   if (enc != AGX_ISA_ENCODING_AGX3B || ctx->stage != MESA_SHADER_FRAGMENT ||
       ctx->key->no_stop || ctx->key->fs.ignore_tib_dependencies ||
       tile_stores_left != 0)
      g17p_final_release = NULL;
   const bool g17p_pixel_protocol =
      g17p_colour_protocol || g17p_blend_protocol || has_coverage_write;
   const bool g17p_colourless_fragment =
      enc == AGX_ISA_ENCODING_AGX3B && ctx->stage == MESA_SHADER_FRAGMENT &&
      !ctx->key->fs.ignore_tib_dependencies && tile_stores_left == 0 &&
      !(ctx->nir && (ctx->nir->info.outputs_written &
                     BITFIELD64_RANGE(FRAG_RESULT_DATA0, AGX_MAX_RTS)));
   const agx_instr *g17p_deferred_write =
      g17p_pixel_protocol && !g17p_mrt.valid
         ? agx3_g17p_deferred_write_wait(ctx, g17p_blend_protocol)
         : NULL;
   agx3_g17p_colour_link_info(ctx, g17p_colour_protocol &&
                                  g17p_deferred_write != NULL);
   ctx->agx3_colour_link.colour_classes =
      g17p_mrt.valid ? g17p_mrt.classes
                     : (g17p_colour_protocol || g17p_blend_protocol ? 1 : 0);
   ctx->agx3_colour_link.straight_line = list_length(&ctx->blocks) == 1;
   ctx->agx3_colour_link.branch_free = list_length(&ctx->blocks) == 1;
   agx_foreach_instr_global(ctx, entry_I) {
      if (agx3_is_cf_op(entry_I))
         ctx->agx3_colour_link.branch_free = false;
      if (agx3_is_cf_op(entry_I) ||
          entry_I->op == AGX_OPCODE_SAMPLE_MASK_AGX3 ||
          entry_I->op == AGX_OPCODE_ZS_EMIT_AGX3)
         ctx->agx3_colour_link.straight_line = false;
   }
   unsigned g17p_link_packs_seen = 0;
   unsigned g17p_mrt_stores_seen = 0;
   bool g17p_mrt_acquired = false;
   bool g17p_colour_open = false;
   unsigned texture_samples_remaining = texture_samples_left;
   unsigned sample_group_members_remaining = sample_group_members_left;
   const agx_instr *g17p_deferred_zs_wait = NULL;
   /* Depth/stencil writes in the open class-1 transaction. */
   unsigned g17p_zs_writes = 0, g17p_zs_staging = 0;
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
   /* The staging path has two deliberately separate lifetime modes: terminal
    * bundles recycle each six-slot batch, while the six-output fallback and
    * captured Firefox suffix retire each staging producer immediately before
    * a later pure producer may reuse its physical GPR. */
   const bool g17p_firefox_dense_outputs =
      agx3_is_firefox_dense_output_suffix(ctx, g17p_output_count);
   const bool g17p_user_output_protocol = g17p_terminal_outputs ||
      g17p_output_count == 6 ||
      (g17p_output_count >= 5 &&
       (g17p_terminal_outputs || agx3_outputs_leave_staging_alone(ctx))) ||
      (ctx->key->g17p_sparse_vertex_outputs && g17p_output_count > 0 &&
       agx3_outputs_leave_staging_alone(ctx)) ||
      g17p_firefox_dense_outputs;
   unsigned g17p_outputs_staged = 0;
   /* User-output graphs outside the qualified scalar shapes must not
    * silently fall back to immediate, potentially out-of-order writeout.
    * An unlinked layout indexes its user outputs at run time instead: no
    * output can then join a batched bundle, so every output, including the
    * fixed point size and clip distances behind them, is staged and written
    * where it is stored. */
   if (enc == AGX_ISA_ENCODING_AGX3B && ctx->stage == MESA_SHADER_VERTEX) {
      bool runtime_output_index = false;
      agx_foreach_instr_global(ctx, I) {
         runtime_output_index |= I->op == AGX_OPCODE_ST_VARY &&
                                 I->nr_srcs >= 1 &&
                                 I->src[0].type != AGX_INDEX_IMMEDIATE;
      }
      agx_foreach_block(ctx, block) {
         agx_foreach_instr_in_block(block, I) {
            if (I->op == AGX_OPCODE_ST_VARY && I->nr_srcs >= 1 &&
                I->src[0].type == AGX_INDEX_IMMEDIATE &&
                I->src[0].value >= 4 && !g17p_user_output_protocol &&
                !runtime_output_index)
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
   /* Native read_write, buffer-read-to-write and sample-to-write kernels
    * assign their sole texture
    * producer to slot 1 and imgwci waits on that slot. Select slot 1 from the
    * exact one-producer/one-write compute graph; the ordinary dependency pass
    * still derives the write's wait bit from the producer result registers. */
   const bool g17p_storage_image_protocol =
      enc == AGX_ISA_ENCODING_AGX3B && ctx->stage == MESA_SHADER_COMPUTE &&
      image_writes_left == 1 && texture_ops_left == 1;

   /* Assign one scoreboard timeline over the whole linearized shader, rather
    * than restarting at each basic block.  Fallthrough blocks may consume an
    * async result from their predecessor.  Control flow after an async op is
    * rejected below until CFG-aware state merging is grounded. */
   struct agx3_slot_instr *sl =
      rzalloc_array(ctx, struct agx3_slot_instr, count ? count : 1);
   struct agx3_slot_result *res =
      rzalloc_array(ctx, struct agx3_slot_result, count ? count : 1);
   agx_instr **ordered = rzalloc_array(ctx, agx_instr *, count ? count : 1);
   uint8_t *native_coord_slots =
      rzalloc_array(ctx, uint8_t, count ? count : 1);
   if (sl == NULL || res == NULL || ordered == NULL ||
       native_coord_slots == NULL)
      goto fail;

   uint8_t native_coord_preferred_slot = 0;
   agx_foreach_instr_global(ctx, stage_scan) {
      if (!stage_scan->agx3_tex_coords_staged)
         continue;
      const bool register_operands =
         stage_scan->nr_srcs == 6 &&
         (stage_scan->src[3].type == AGX_INDEX_REGISTER ||
          stage_scan->src[4].type == AGX_INDEX_REGISTER);
      native_coord_preferred_slot = register_operands ? 6 : 1;
      break;
   }

   unsigned k = 0;
   unsigned vertex_loads_scheduled = 0;
   agx_foreach_block(ctx, slot_block) {
      agx_foreach_instr_in_block(slot_block, I) {
         if (I->op == AGX_OPCODE_STOP)
            continue;
         ordered[k] = I;
         struct agx3_slot_instr *e = &sl[k++];
         /* EXPORT is a zero-byte register-ABI declaration after RA, not a
          * physical read. Letting it consume an async dependency marks the
          * slot waited without emitting any wait, so a linked epilog reads
          * the result too early. Real consumers retain their own waits;
          * outstanding results are synchronized at the shader-part tail. */
         if (I->op == AGX_OPCODE_EXPORT)
            continue;
         e->is_async = agx3_is_async(I) || I->agx3_tex_coord_start;
         /* Release ordering includes reads whose results are used after the
          * fence. Register dependencies alone do not order those reads. */
         e->drain = I->op == AGX_OPCODE_SCOPED_BARRIER_AGX3 &&
            (I->imm == AGX3_BARRIER_SCOPE_DEVICE ||
             I->imm == AGX3_BARRIER_SCOPE_DEVICE_FENCE);
         e->reissues_live_slot = I->op == AGX_OPCODE_LDCF;
         if (I->agx3_tex_coord_start)
            e->preferred_slot = native_coord_preferred_slot;
         /* Plain 2D and 3D register-LOD fetches have two independent
          * scoreboard generations: PACK_SMPIR/COPY publish staging on slot 2
          * and TEX returns on slot 1. Native array fetches instead use slot 1
          * for their combined layer/LOD publication and the following TEX,
          * so they have no independent transient slot. */
         if (enc == AGX_ISA_ENCODING_AGX3B &&
             (I->op == AGX_OPCODE_TEXTURE_LOAD ||
              I->op == AGX_OPCODE_IMAGE_LOAD) &&
             I->lod_mode == AGX_LOD_MODE_LOD_MIN && I->nr_srcs >= 2 &&
             agx3_tex_plain_reg(I->src[1]) &&
             I->src[1].size == AGX_SIZE_32 &&
             agx_channels(I->src[1]) == 1) {
            e->preferred_slot = 1;
            if (I->dim != AGX_DIM_2D_ARRAY)
               e->transient_slots = 1u << 1;
         }
         if ((enc == AGX_ISA_ENCODING_AGX3B &&
              I->op == AGX_OPCODE_IMAGE_LOAD) ||
             (g17p_storage_image_protocol &&
              (I->op == AGX_OPCODE_TEXTURE_SAMPLE ||
               I->op == AGX_OPCODE_TEXTURE_LOAD)))
            e->preferred_slot = 1;
         if (enc == AGX_ISA_ENCODING_AGX3B &&
             I->op == AGX_OPCODE_ZS_EMIT_AGX3)
            e->preferred_slot = 1;
         /* A returned G17P device atomic is a two-stage operation in Apple's
          * native stream: atomicdev publishes the old value in TPS half 0 on
          * slot 1, then rld waits slot 1 and publishes the architectural GPR
          * result on slot 6. Model slot 1 as an internal generation and slot
          * 6 as the logical instruction's result dependency. */
         const bool g17p_returned_atomic =
            enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_ATOMIC &&
            I->nr_dests == 1 && !agx_is_null(I->dest[0]);
         if (g17p_returned_atomic) {
            e->preferred_slot = 6;
            e->transient_slots = 1u << 0;
         }
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
         if (I->agx3_tex_coords_staged)
            e->preferred_slot = 1;

         unsigned nd = 0, ns = 0;
         if (!I->agx3_tex_coord_stage) {
            for (unsigned d = 0; d < I->nr_dests; ++d)
               nd = agx3_expand_regs(I->dest[d], e->dst, nd);
         }
         for (unsigned s = 0; s < I->nr_srcs; ++s) {
            if (I->agx3_tex_coords_staged && s == 0)
               continue;
            ns = agx3_expand_regs(I->src[s], e->src, ns);
         }
         if (I->op == AGX_OPCODE_LOAD_CPROG_AGX3) {
            if (ns >= AGX3_SLOT_MAX_REGS)
               goto fail;
            e->src[ns++] = AGX3_SLOT_CPROG_INPUT_TOKEN;
         }
         if (nd > AGX3_SLOT_MAX_REGS || ns > AGX3_SLOT_MAX_REGS ||
             (I->agx3_tex_coord_start && nd == AGX3_SLOT_MAX_REGS) ||
             (I->agx3_tex_coords_staged && ns == AGX3_SLOT_MAX_REGS)) {
            agx3_report_refusal(I, __LINE__);
            goto fail;
         }
         if (I->agx3_tex_coord_start)
            e->dst[nd++] = AGX3_SLOT_TEX_COORD_TOKEN;
         if (I->agx3_tex_coords_staged)
            e->src[ns++] = AGX3_SLOT_TEX_COORD_TOKEN;

         /* rld's asynchronous read of the hidden atomic-return words must
          * complete before a later stack access may reuse them. */
         if (g17p_returned_atomic) {
            const unsigned tps_half = ctx->agx3_atomic_return_tps_half;
            assert(nd + 2 <= AGX3_SLOT_MAX_REGS);
            assert(tps_half + 1 < AGX3_SLOT_TPS_HALF_TOKENS);
            e->dst[nd++] = AGX3_SLOT_TPS_TOKEN_BASE + tps_half;
            e->dst[nd++] = AGX3_SLOT_TPS_TOKEN_BASE + tps_half + 1;
         }

         /* CEM is a compound cube-coordinate pipeline, not a single opaque
          * sampler operation. Its three calculations write the dedicated
          * CEM scratch bank before SMP consumes the staged result.
          * Keep those writes owned by the sample's scoreboard generation so
          * a following cube pipeline cannot clobber them while the first SMP
          * is still in flight.  Apple's two-cube G17P fragment does the
          * equivalent by completing both CEM pipelines before issuing either
          * SMP, with separate staging banks for the two results.
          *
          * A four-channel f32 texture result expands to eight tokens, leaving
          * room for the four scratch words, including axis0's implicit face. */
         if (enc == AGX_ISA_ENCODING_AGX3B &&
             I->op == AGX_OPCODE_TEXTURE_SAMPLE &&
             (I->dim == AGX_DIM_CUBE ||
              I->dim == AGX_DIM_CUBE_ARRAY)) {
            const bool bias_min =
               I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS_MIN;
            const bool register_lod = bias_min ||
               ((I->lod_mode == AGX_LOD_MODE_LOD_MIN ||
                 I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS) &&
                I->src[1].type != AGX_INDEX_IMMEDIATE);
            const bool dynamic_cem = register_lod ||
               I->src[3].type != AGX_INDEX_IMMEDIATE ||
               I->src[4].type != AGX_INDEX_IMMEDIATE;
            const uint16_t fixed_scratch[3] = {0, 14, 4};
            const uint16_t *scratch = dynamic_cem ? cem_scratch
                                                   : fixed_scratch;
            assert(nd + 5 <= AGX3_SLOT_MAX_REGS);
            e->dst[nd++] = scratch[0];
            e->dst[nd++] = scratch[0] + 2;
            e->dst[nd++] = scratch[1];
            e->dst[nd++] = scratch[2];
            /* The cube-array layer prepacked beside a register LOD, or the
             * cube face packed beside a bias and minimum. */
            if (register_lod && (I->dim == AGX_DIM_CUBE_ARRAY || bias_min))
               e->dst[nd++] = ms_array_scratch;
         }
         /* A 2D-array comparison and a staged 3D bias pack their minimum
          * LOD into the same word, which SMP reads after its staging
          * completes. */
         if (enc == AGX_ISA_ENCODING_AGX3B &&
             I->op == AGX_OPCODE_TEXTURE_SAMPLE &&
             ((I->dim == AGX_DIM_2D_ARRAY && I->shadow &&
               I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS_MIN) ||
              agx3_is_staged_3d_bias_min(I))) {
            assert(nd < AGX3_SLOT_MAX_REGS);
            e->dst[nd++] = ms_array_scratch;
         }
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

   /* A common descriptor diamond loads two pointers, dereferences both, then
    * selects between their values. The two value loads must occupy the native
    * slots-1-and-2 pair: an ISEL waiting on two other live slot generations
    * is not a grounded hardware form and intermittently consumes one result
    * early. Reuse slot 3 for the serialized pointer loads, whose generations
    * are retired by their respective dereferences, while the two value loads
    * remain overlapped on slots 2 and 1. */
   if (enc == AGX_ISA_ENCODING_AGX3B) {
      for (unsigned i = 0; i + 4 < count; ++i) {
         if (ordered[i]->op != AGX_OPCODE_DEVICE_LOAD ||
             ordered[i + 1]->op != AGX_OPCODE_DEVICE_LOAD ||
             ordered[i + 2]->op != AGX_OPCODE_DEVICE_LOAD ||
             ordered[i + 3]->op != AGX_OPCODE_DEVICE_LOAD ||
             ordered[i + 4]->op != AGX_OPCODE_ICMPSEL)
            continue;

         bool available = true;
         for (unsigned j = 0; j < 4; ++j) {
            available &= sl[i + j].preferred_slot == 0 &&
                         sl[i + j].transient_slots == 0;
         }
         if (!available ||
             !agx3_slot_instr_reads(&sl[i + 1], &sl[i]) ||
             !agx3_slot_instr_reads(&sl[i + 3], &sl[i + 2]) ||
             !agx3_slot_instr_reads(&sl[i + 4], &sl[i + 1]) ||
             !agx3_slot_instr_reads(&sl[i + 4], &sl[i + 3]))
            continue;

         sl[i].preferred_slot = 3;
         sl[i + 1].preferred_slot = 2;
         sl[i + 2].preferred_slot = 3;
         sl[i + 3].preferred_slot = 1;
      }
   }

   /* Keep adjacent async producers feeding ISEL overlapped on slots 2 and 1.
    * Emission retires that pair through a data-dependent ALU carrier before
    * ISEL, avoiding both a multi-slot ISEL wait and concurrent reuse of one
    * scoreboard slot. Producers with a compound internal slot contract
    * already have a preferred/transient assignment and remain outside this
    * rule. */
   if (enc == AGX_ISA_ENCODING_AGX3B) {
      for (unsigned i = 0; i + 2 < count; ++i) {
         if (!sl[i].is_async || !sl[i + 1].is_async ||
             ordered[i + 2]->op != AGX_OPCODE_ICMPSEL ||
             sl[i].preferred_slot || sl[i + 1].preferred_slot ||
             sl[i].transient_slots || sl[i + 1].transient_slots)
            continue;

         if (agx3_slot_instr_reads(&sl[i + 2], &sl[i]) &&
             agx3_slot_instr_reads(&sl[i + 2], &sl[i + 1])) {
            const bool device_pair = ordered[i]->op == AGX_OPCODE_DEVICE_LOAD &&
               ordered[i + 1]->op == AGX_OPCODE_DEVICE_LOAD;
            const bool texture_then_device =
               (ordered[i]->op == AGX_OPCODE_TEXTURE_SAMPLE ||
                ordered[i]->op == AGX_OPCODE_TEXTURE_LOAD ||
                ordered[i]->op == AGX_OPCODE_IMAGE_LOAD) &&
               ordered[i + 1]->op == AGX_OPCODE_DEVICE_LOAD;
            if (device_pair || texture_then_device) {
               sl[i].preferred_slot = 2;
               sl[i + 1].preferred_slot = 1;
            }
         }
      }
   }

   const struct agx3_slot_entry input_entry = {
      .token = AGX3_SLOT_CPROG_INPUT_TOKEN, .slot = 6,
   };
   if (!agx3_slot_assign_with_entry(sl, count, nslots, &input_entry,
                                    cprog_imports ? 1 : 0, res))
      goto fail;

   /* Emission drains every live scoreboard generation before control flow,
    * but the linear slot assignment above cannot see that architectural
    * boundary. Remove waits belonging to those retired generations until a
    * later producer reuses the slot. Otherwise a post-branch ISEL may carry a
    * stale slot together with two real producers, escaping the native
    * slots-1-and-2 pair even though only that pair is live. */
   uint16_t cf_retired = 0, cf_live = entry_wait;
   for (unsigned i = 0; i < count; ++i) {
      res[i].wait_mask &= (uint16_t)~cf_retired;
      cf_live &= (uint16_t)~res[i].wait_mask;

      if (res[i].slot) {
         const uint16_t bit = (uint16_t)(1u << (res[i].slot - 1));
         cf_retired &= (uint16_t)~bit;
         cf_live |= bit;
      }

      if (agx3_is_cf_op(ordered[i])) {
         cf_retired |= cf_live;
         cf_live = 0;
      }
   }

   uint8_t active_coord_slot = 0;
   unsigned staged_components = 0;
   for (unsigned i = 0; i < count; ++i) {
      const agx_instr *I = ordered[i];
      if (I->agx3_tex_coord_start) {
         if (active_coord_slot || !res[i].slot)
            goto fail;
         active_coord_slot = res[i].slot;
      }
      if (I->agx3_tex_coord_stage) {
         if (!active_coord_slot)
            goto fail;
         native_coord_slots[i] = active_coord_slot;
         staged_components |= BITFIELD_BIT(I->agx3_tex_coord_component);
      }
      if (I->agx3_tex_coords_staged) {
         if (!active_coord_slot || staged_components != 3)
            goto fail;
         native_coord_slots[i] = active_coord_slot;
         active_coord_slot = 0;
         staged_components = 0;
      }
   }
   if (active_coord_slot)
      goto fail;

   const bool *tps_rereads = agx3_static_reload_rereads(ctx, ordered, count);
   if (!tps_rereads)
      goto fail;

   k = 0;
   unsigned vertex_loads_seen = 0;
   unsigned loop_depth = 0;
   /* f63 on F32INTERP and f48 on LDCF are two encodings of the same native
    * fragment-input entry marker. Apple's mixed flat+smooth fragment marks
    * whichever family comes first and leaves the other family's first op
    * unmarked; restarting here corrupts the coefficient-fetch protocol. */
   bool fragment_input_started = false;
   struct agx3_staging_banks staging_banks = {0};
   /* Samples stage fixed halves 0..19. Per-sample DEPTHF exports reserve
    * successive four-half payloads (two for stencil-only), not just the
    * first sample's 0..3. Image writes must not claim any of those halves,
    * including payloads that later unrolled samples have yet to publish. */
   const uint16_t fixed_staging_floor =
      texture_samples_left ? 24
      : (ctx->stage == MESA_SHADER_FRAGMENT &&
         (has_coverage_write || has_zs_emit)) ? 8
                                              : 0;
   const uint16_t image_write_bank_floor =
      MAX2(fixed_staging_floor, ALIGN_POT(zs_staging_halfs, 8));
   if ((has_coverage_write || (has_memory_write && !ctx->key->secondary)) &&
       enc == AGX_ISA_ENCODING_AGX3B &&
       ctx->stage == MESA_SHADER_FRAGMENT &&
       !ctx->key->fs.ignore_tib_dependencies) {
      /* Native memory_discard enters before its first device load, then
       * uses unmarked SR/RHW inputs throughout the coverage operation.
       * This is the same existing entry gate, moved before the main part's
       * root loads; do not restart the input protocol after MOVMSK.
       *
       * An API fragment shader with memory side effects enters before the
       * first of them too. metal-nt -arch applegpu_g17p fragments with no
       * colour writes emit 870254000600 ahead of a device store, a voiding
       * atomic and a returning atomic alike. Without it, only the depth
       * epilog opened the gate, after every side effect of the main part.
       * agx_fast_link removes the epilog's gate. */
      uint8_t entry[AGX3_PIXEL_MAX_LENGTH], entry_len;
      if (!agx3_pixel_colour_entry(enc, entry, &entry_len) ||
          !agx3_append(emission, entry, entry_len))
         goto fail;
      ctx->agx3_colour_link.entry_emitted = true;
      ctx->agx3_colour_link.entry_offset = emission->size - entry_len - initial_size;
      g17p_colour_open = true;
      fragment_input_started = true;
   }

   agx_foreach_block(ctx, block) {
      block->offset = emission->size;
      block->loop_exit_offset = -1;
      bool after_loop_backedge = false;
      bool after_loop_exit_test = false;
      agx_foreach_instr_in_block(block, I) {
         block->last_offset = emission->size;

         if (I->op == AGX_OPCODE_STOP)
            continue;

         const unsigned instr_index = k;
         const uint8_t slot = res[k].slot;
         uint16_t wait = res[k].wait_mask;
         const uint8_t native_coord_slot = native_coord_slots[instr_index];
         k++;

         if (agx3_g17p_trace_slots())
            fprintf(stderr, "G17Pslot %-3u %-22s slot=%u wait=0x%02x\n",
                    instr_index, agx_opcodes_info[I->op].name, slot, wait);

         /* A wait retires its slots before this instruction issues its own. */
         outstanding &= (uint16_t)~wait;
         const uint16_t outstanding_here = outstanding;
         if (slot)
            outstanding |= (uint16_t)(1u << (slot - 1));

         /* Give every control-flow boundary an empty scoreboard, rather
          * than refusing the shader when something is still in flight. */
         /* An instruction waiting on two live slot generations is not a
          * grounded hardware form: the slot planner's own ISEL note records
          * that it "intermittently consumes one result early", and fixes it
          * only for the device-load diamond by pinning preferred slots. Do it
          * generally -- retire every generation but one through the existing
          * carrier, so the instruction carries the single wait the witnessed
          * encodings use. Instructions that already lower their own multi-slot
          * dependencies (textures, atomics, control flow) are left alone. */
         /* Mode 2: move the wait off a BITOP entirely, onto the existing
          * band-identity carrier. The XOR register swap that lowers a post-RA
          * cycle is the only opcode the failing sampler1darrayshadow shader has
          * and the passing sampler2darrayshadow shader does not, and lookup 3's
          * result is what comes out of it. Its wait IS planned and IS encoded
          * (probed: wait 0x08 -> byte[6]=0x20), so this asks the remaining
          * question: whether the hardware honours a wait field on a BITOP that
          * reads a texture destination. */
         /* Modes 5 and 6 split mode 2 by which half of the wait mask is
          * moved onto the carrier. Probing agx3_alu_bit_reg shows a BITOP
          * encodes its 6-bit wait in TWO separate places: slots 1-3 land in
          * byte[5] bits 5..7, slots 4-6 land in byte[7] bits 5..7, and byte[6]
          * is untouched. If only one of those halves is actually read by the
          * hardware, moving just that half onto a carrier should recover all
          * of mode 2's benefit and the other half should be worth nothing.
          *   5 = move only slots 4-6 (mask 0x38), keep 1-3 inline
          *   6 = move only slots 1-3 (mask 0x07), keep 4-6 inline */
         {
            const int swm = agx3_g17p_single_wait_mode();
            if ((swm == 2 || swm == 5 || swm == 6) &&
                I->op == AGX_OPCODE_BITOP && wait != 0) {
               const uint16_t move = swm == 5   ? (uint16_t)(wait & 0x38)
                                     : swm == 6 ? (uint16_t)(wait & 0x07)
                                                : wait;
               if (move) {
                  if (!agx3_retire_slots_noop(emission, enc, move))
                     AGX3_REFUSE(I);
                  wait &= (uint16_t)~move;
               }
            }
         }
         if (agx3_g17p_single_wait() && !agx3_is_cf_op(I) &&
             (wait & (uint16_t)(wait - 1))) {
            const uint16_t keep = wait & (uint16_t)-wait;
            if (!agx3_retire_slots_noop(emission, enc,
                                        (uint16_t)(wait & ~keep)))
               AGX3_REFUSE(I);
            wait = keep;
         }

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

         const bool loop_entry_push = agx3_is_loop_entry_push(I);
         const bool loop_pop =
            I->op == AGX_OPCODE_POP_EXEC && after_loop_backedge;
         if (loop_pop)
            block->loop_exit_offset = emission->size;
         const int cf = agx3_try_control_flow(
            emission, &fixups, enc, ctx->key->dev.chip, I, cf_wait,
            cf_outstanding,
            after_loop_backedge,
            I->op == AGX_OPCODE_JMP_EXEC_NONE_AFTER && after_loop_exit_test,
            loop_depth);
         if (cf < 0)
            AGX3_REFUSE(I);
         if (cf > 0) {
            if (loop_entry_push)
               loop_depth++;
            else if (loop_pop && loop_depth > 0) {
               loop_depth--;
            }
            after_loop_backedge = I->op == AGX_OPCODE_JMP_EXEC_ANY;
            after_loop_exit_test =
               (I->op == AGX_OPCODE_BREAK_IF_ICMP ||
                I->op == AGX_OPCODE_BREAK_IF_FCMP) &&
               I->nest == 1;
            continue;
         }
         after_loop_backedge = false;
         after_loop_exit_test = false;

         if (I->op == AGX_OPCODE_UNIFORM_FADD_AGX3 ||
             I->op == AGX_OPCODE_UNIFORM_FMUL_AGX3 ||
             I->op == AGX_OPCODE_UNIFORM_FFMA_AGX3) {
            const enum agx3_uniform_float_op op =
               I->op == AGX_OPCODE_UNIFORM_FADD_AGX3 ? AGX3_UNIFORM_FADD :
               I->op == AGX_OPCODE_UNIFORM_FMUL_AGX3 ? AGX3_UNIFORM_FMUL :
                                                     AGX3_UNIFORM_FMA;
            const unsigned count = op == AGX3_UNIFORM_FMA ? 3 : 2;
            if (!ctx->is_preamble || I->nr_dests || I->nr_srcs != count ||
                I->saturate || I->imm >= 128 || (I->imm & 1))
               AGX3_REFUSE(I);
            struct agx3_uniform_float_src src[3] = {0};
            for (unsigned s = 0; s < count; ++s) {
               const agx_index index = I->src[s];
               const bool reg = index.type == AGX_INDEX_REGISTER;
               if ((!reg && index.type != AGX_INDEX_IMMEDIATE) ||
                   index.value >= 256 ||
                   index.memory || index.abs || index.neg ||
                   (reg && (index.size != AGX_SIZE_32 ||
                            agx_channels(index) != 1)))
                  AGX3_REFUSE(I);
               src[s] = (struct agx3_uniform_float_src){
                  .kind = reg ? AGX3_FLOAT_REGISTER : AGX3_FLOAT_IMMEDIATE,
                  .value = index.value,
                  .cache = index.cache,
                  .discard = index.discard,
               };
            }
            uint8_t bytes[16], len;
            if (!agx3_uniform_float_store_encode(enc, op, I->imm, src,
                                                  wait, bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

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

         /* AGX2 marks a vertex shader that writes no varying with its own
          * instruction. G17P has none: applegpu_g17p compiles a vertex
          * function that writes nothing to a bare `end` (0e000000). */
         if (I->op == AGX_OPCODE_NO_VARYINGS) {
            if (ctx->stage != MESA_SHADER_VERTEX)
               AGX3_REFUSE(I);
            continue;
         }

         if (I->agx3_tex_coord_stage) {
            if (enc != AGX_ISA_ENCODING_AGX3B || native_coord_slot == 0 ||
                I->nr_dests != 1 || I->dest[0].size != AGX_SIZE_32 ||
                agx_channels(I->dest[0]) != 1)
               AGX3_REFUSE(I);

            const uint16_t staging_dest =
               2 * (uint16_t)I->agx3_tex_coord_component;
            uint8_t bytes[AGX3_ALU_MAX_LENGTH];
            uint8_t len = 0;
            if (I->op == AGX_OPCODE_PERSP_MUL_AGX3) {
               if (I->nr_srcs != 2 || I->saturate || I->imm > 0x7f ||
                   !agx3_vary_reg32(I->src[0]) ||
                   !agx3_vary_reg32(I->src[1]) ||
                   agx_channels(I->src[0]) != 1 ||
                   agx_channels(I->src[1]) != 1 || I->src[0].kill ||
                   I->src[1].kill ||
                   (I->src[0].cache && I->src[0].discard) ||
                   (I->src[1].cache && I->src[1].discard))
                  AGX3_REFUSE(I);
               const struct agx3_persp_mul_desc desc = {
                  .stage_dest = true,
                  .dest = staging_dest,
                  .slot = native_coord_slot,
                  .raw = I->src[0].value,
                  .reciprocal = I->src[1].value,
                  .raw_liveness = I->src[0].discard ? AGX3_OPERAND_KILL
                     : I->src[0].cache ? AGX3_OPERAND_LIVE
                                       : AGX3_OPERAND_UNMARKED,
                  .reciprocal_liveness =
                     I->src[1].discard ? AGX3_OPERAND_KILL
                     : I->src[1].cache ? AGX3_OPERAND_LIVE
                                       : AGX3_OPERAND_UNMARKED,
                  .vary_index = I->imm,
                  .wait_mask = wait,
               };
               if (!agx3_persp_mul_encode(enc, &desc, bytes, &len))
                  AGX3_REFUSE(I);
            } else if (I->op == AGX_OPCODE_FADD) {
               if (I->nr_srcs != 2 || I->saturate)
                  AGX3_REFUSE(I);
               uint16_t src[2];
               enum agx3_alu_hint hints[2];
               for (unsigned s = 0; s < 2; ++s) {
                  if (!agx3_vary_reg32(I->src[s]) ||
                      agx_channels(I->src[s]) != 1 || I->src[s].abs ||
                      I->src[s].neg || I->src[s].kill ||
                      (I->src[s].cache && I->src[s].discard))
                     AGX3_REFUSE(I);
                  src[s] = I->src[s].value;
                  hints[s] = agx3_post_ra_hint(I->src[s]);
               }
               if (!agx3_alu_fadd_stage(enc, staging_dest,
                                        native_coord_slot, src, hints, wait,
                                        bytes, &len))
                  AGX3_REFUSE(I);
            } else {
               AGX3_REFUSE(I);
            }

            if (!agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         if (I->op == AGX_OPCODE_ITRCOORDS_AGX3) {
            if (ctx->stage != MESA_SHADER_FRAGMENT ||
                !agx3_emit_itrcoords(emission, enc, I, wait))
               AGX3_REFUSE(I);
            continue;
         }

         if (I->op == AGX_OPCODE_INTERP_AGX3 ||
             I->op == AGX_OPCODE_RHW_INTERP_AGX3) {
            if (agx3_interp_uses_coords(enc, I) &&
                !agx3_sample_coords_use(block, I))
               AGX3_REFUSE(I);
            if (I->op == AGX_OPCODE_RHW_INTERP_AGX3 &&
                (enc != AGX_ISA_ENCODING_AGX3B ||
                 ctx->stage != MESA_SHADER_FRAGMENT ||
                 !agx3_private_rhw_use(ctx, block, I)))
               AGX3_REFUSE(I);
            uint16_t interp_wait = wait;
            /* Implicit-coordinate interpolation does not read its destination.
             * Its wait therefore covers the old destination's producers, not
             * an arbitrary r0. A synthetic r0 identity can race an unrelated
             * load and write back the old pointer or only one of its halves.
             *
             * Carry exactly these WAW dependencies on a source-free LDIMM
             * into the destination that interpolation immediately replaces.
             * Join the required bits before writing either destination half.
             * No temporary register or additional dependency is needed. */
            if (interp_wait) {
               /* Sample interpolation can read a token tied to its result. */
               if (enc == AGX_ISA_ENCODING_AGX3B &&
                   I->interpolation != AGX_INTERPOLATION_SAMPLE) {
                  if (I->nr_dests != 1 ||
                      !agx3_vary_reg32(I->dest[0]) ||
                      agx_channels(I->dest[0]) != 1)
                     AGX3_REFUSE(I);
                  uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
                  if (!agx3_alu_compose(enc, AGX3_ALU_LDIMM,
                                          I->dest[0].value, 0, 0, 0, 0,
                                          interp_wait, bytes, &len) ||
                      !agx3_append(emission, bytes, len))
                     AGX3_REFUSE(I);
               } else if (!agx3_retire_slots_noop(emission, enc, interp_wait)) {
                  AGX3_REFUSE(I);
               }
               outstanding &= (uint16_t)~interp_wait;
               interp_wait = 0;
            }
            const bool first = enc == AGX_ISA_ENCODING_AGX3B &&
               ctx->stage == MESA_SHADER_FRAGMENT && !fragment_input_started &&
               !g17p_fragment_helper_protocol;
            if (!agx3_emit_interp(emission, enc, I, interp_wait, first,
                                  ctx->stage == MESA_SHADER_FRAGMENT))
               AGX3_REFUSE(I);
            fragment_input_started = true;
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
               ctx->stage == MESA_SHADER_FRAGMENT && !fragment_input_started &&
               !g17p_fragment_helper_protocol;
            if (!agx3_emit_ldcf(emission, enc, I, slot, wait, first))
               AGX3_REFUSE(I);
            fragment_input_started = true;
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
               g17p_user_output_protocol
                  ? agx3_g17p_output_staging_half(I->src[0].value)
                  : (uint16_t)(2 * (stage_slot - 1));
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
               /* Each batch writes the components it staged: a terminal
                * bundle in store order, other stores in component order. */
               g17p_output_order[g17p_outputs_staged] = I->src[0].value;
               ++g17p_outputs_staged;
               if (g17p_outputs_staged % nslots == 0 ||
                   g17p_outputs_staged == g17p_output_count) {
                  const unsigned first =
                     ((g17p_outputs_staged - 1) / nslots) * nslots;
                  if (!g17p_terminal_outputs) {
                     qsort(g17p_output_order + first,
                           g17p_outputs_staged - first,
                           sizeof(g17p_output_order[0]), agx3_cmp_u8);
                  }
                  for (unsigned position = first;
                       position < g17p_outputs_staged; ++position) {
                     const unsigned component = g17p_output_order[position];
                     uint8_t bytes[AGX3_VARY_MAX_LENGTH], len = 0;
                     if (!agx3_vary_uvsw(enc,
                                          agx3_g17p_output_staging_half(component),
                                          component, component % nslots + 1, true,
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
            /* Apple's argument-buffer vertex read publishes uint2 coordinates
             * before SMP.EP4. G17P indexed imageLoad uses the same read-only
             * texture heap and coordinate ABI as texelFetch: its GPR pair is
             * not a staging operand. Legacy immediate IMAGE_LOAD reads keep
             * the unstaged form.
             *
             * A vertex texel fetch does not. The exception used to cover
             * TEXTURE_LOAD too -- staging only when the texture operand is a
             * run-time index -- and on AGX3B that is wrong: the unstaged form
             * returns only the x and y components. Whether the shader asks
             * for the whole vec4 or for .z alone, everything past y reads
             * back as zero, while the identical fetch in a fragment shader
             * (always staged) and a vertex textureLod are both correct.
             * src/asahi/gl-tests/test-g17p-gl-vertex-flat-varying.c is that
             * measurement, and KHR-GL33.texture_swizzle's 363 failures are
             * all this one shape: the CTS samples in the vertex stage and
             * passes the texel through a flat varying. */
            const bool vertex_fetch_stages_coordinates =
               enc == AGX_ISA_ENCODING_AGX3B &&
               ctx->stage == MESA_SHADER_VERTEX &&
               I->op == AGX_OPCODE_TEXTURE_LOAD;
            const bool staged =
               (enc == AGX_ISA_ENCODING_AGX3B &&
                I->dim == AGX_DIM_2D_MS_ARRAY) ||
               vertex_fetch_stages_coordinates ||
               ((I->op == AGX_OPCODE_TEXTURE_LOAD ||
                 enc == AGX_ISA_ENCODING_AGX3B) &&
                ctx->stage == MESA_SHADER_VERTEX &&
                I->src[3].type != AGX_INDEX_IMMEDIATE) ||
               ctx->stage == MESA_SHADER_FRAGMENT ||
               agx3_stage_runs_as_kernel(ctx->stage);
            /* imgwci takes its coordinates from staging halves 0/2, so a
             * read in a shader that also writes images must not stage its
             * coordinates there: an unwaited write COPY can replace them
             * before the asynchronous read consumes them. Apple moves the
             * read to halves 16/18. texture_buffer<uint>::read followed by a
             * 2D write does, as does an indexed read-only image read followed
             * by a write, metal-nt -arch applegpu_g17p:
             *
             *   read only    copy 5:0, copy 5:2, smp ep4
             *   read, write  copy 5:16, copy 5:18, smp ep4 f19 16
             */
            const bool separate_write_coords =
               enc == AGX_ISA_ENCODING_AGX3B && image_writes_left != 0 &&
               ((g17p_storage_image_protocol &&
                 I->op == AGX_OPCODE_TEXTURE_LOAD) ||
                I->op == AGX_OPCODE_IMAGE_LOAD) &&
               (I->dim == AGX_DIM_2D || I->dim == AGX_DIM_2D_ARRAY ||
                I->dim == AGX_DIM_3D);
            /* A fetch whose level is a live register, which is the shape
             * whose level stages into the sampler-private bank. */
            const bool register_fetch_lod =
               I->op == AGX_OPCODE_TEXTURE_LOAD &&
               I->lod_mode == AGX_LOD_MODE_LOD_MIN &&
               I->src[1].type == AGX_INDEX_REGISTER;
            /* A staged read consumes, and kills, its coordinate bank
             * asynchronously, so every stage that stages coordinates needs
             * the bank allocator -- not only the kernel and API-fragment
             * stages it was first written for.
             *
             * A vertex texel fetch always stages (see
             * vertex_fetch_stages_coordinates above) but used to pass no
             * `banks`, so every fetch in the shader republished halves 0..3
             * with nothing tracking the read still consuming them. Measured
             * on G17P in a ycbcr.conversion.chroma_reconstruction vertex
             * shader with three fetches: the third restaged halves 0..3
             * while slot 6 -- the second fetch -- was still outstanding, so
             * the second tap sampled the third's coordinates. Scheduling
             * only decided whether the fetches overlapped, which is why
             * AGX_MESA_DEBUG=nosched hid it.
             *
             * Wherever at most one read is in flight this is a no-op: with
             * no conflicting live bank the allocator returns the base bank
             * and emits no retirement, so the encoding is unchanged.
             *
             * Background, clear and end-of-tile programs need it too. A
             * multi-target background reloads one texel fetch per attachment
             * with nothing tracking the bank an earlier fetch still consumes,
             * so a later fetch restages halves 0..3 under it and that
             * attachment reloads another target's coordinates. */
            const bool staged_bank_reads = staged;
            if (slot)
               sample_slots_live |= (uint16_t)(1u << (slot - 1));
            assert(texture_ops_left > 0);
            const bool final = --texture_ops_left == 0;
            const uint16_t fetch_wait =
               wait | ((I->dim == AGX_DIM_2D_MS ||
                         I->dim == AGX_DIM_2D_MS_ARRAY)
                          ? (outstanding_here & 1u)
                          : 0);
            /* A fragment texel fetch with a per-lane level stages that level
             * into the sampler-private bank; Apple enters the pixel
             * transaction before such an SMP. Keep the same straight-line
             * gate placement rule as the implicit-derivative samples. */
            const bool open_pixel_fetch =
               enc == AGX_ISA_ENCODING_AGX3B &&
               ctx->stage == MESA_SHADER_FRAGMENT &&
               !ctx->key->fs.ignore_tib_dependencies &&
               ctx->agx3_colour_link.straight_line &&
               register_fetch_lod && I->dim == AGX_DIM_2D && !g17p_colour_open;
            uint16_t staged_end = 0;
            if (!agx3_emit_texture_fetch(emission, enc, I, slot, fetch_wait,
                                         final, staged, separate_write_coords,
                                         ms_array_scratch,
                                         staged_bank_reads ? &staging_banks
                                                           : NULL,
                                         outstanding_here, &staged_end,
                                         open_pixel_fetch,
                                         &ctx->agx3_colour_link.entry_offset))
               AGX3_REFUSE(I);
            if (open_pixel_fetch) {
               ctx->agx3_colour_link.entry_emitted = true;
               ctx->agx3_colour_link.entry_offset -= initial_size;
               g17p_colour_open = true;
            }
            if (staged) {
               /* The coordinate bank, or the eight halves of a read with
                * PACK_SMPIR operands: native MS fragments reserve eight
                * halves, not just the four occupied by the x/y COPYs. */
               ctx->agx3_staging_halfs =
                  MAX2(ctx->agx3_staging_halfs, staged_end);
            } else {
               ctx->agx3_staging_known = false;
            }
            continue;
         }

         if (I->op == AGX_OPCODE_IMAGE_WRITE) {
            if (!agx3_emit_storage_image_write(emission, enc, I, wait,
                                               &staging_banks,
                                               image_write_bank_floor,
                                               &outstanding))
               AGX3_REFUSE(I);
            ctx->agx3_staging_halfs =
               MAX2(ctx->agx3_staging_halfs, MAX2(staging_banks.end, 16));
            continue;
         }

         if (I->op == AGX_OPCODE_TEXTURE_SAMPLE) {
            /* Diagnostic (AGX_G17P_DRAIN_BEFORE_SAMPLE=1): retire every
             * outstanding scoreboard generation immediately before each
             * sample.
             *
             * Motivated by the oracle, not by guesswork. Apple's compiler, on
             * the exact CTS shape (.gpu-handoff/oracle/g.metal), emits ONE
             * pack_smpir followed by four back-to-back smp. We emit FOUR
             * pack_smpir, one per sample, interleaved with staging. If
             * pack_smpir publishes SHARED sampler-index state, re-issuing it
             * while an earlier sample is still in flight would corrupt a
             * sample's index -- and the victim would be the last one issued,
             * which is exactly what bug #1 does. Draining here serialises
             * that publication. */
            {
               /* DEFAULT ON. A texture sample's planned scoreboard wait is
                * retired on a carrier instruction instead of being encoded in
                * the sample's own staging copies, whose wait field does not
                * take effect on G17P.
                *
                * Evidence: with the wait left in the copy, the CTS case
                * dEQP-VK.glsl.opaque_type_indexing.sampler.dynamically_uniform
                * .fragment.sampler1darrayshadow fails 18 runs in 20 and the
                * standalone reproducer fails ~270/300. Moving exactly that
                * wait onto a carrier -- two instructions, +20 bytes on a 1792
                * byte shader -- makes both 0 for 300 and 30/30. The emitted
                * difference is precisely two copies losing byte[4] bit 0x80,
                * their wait bit, plus the two carriers.
                *
                * Set AGX_G17P_DRAIN_BEFORE_SAMPLE=0 to restore the old
                * behaviour for A/B; 1 drains everything outstanding, 2 only
                * previous samples' generations. */
               static int drain = -1;
               if (drain < 0) {
                  const char *v = getenv("AGX_G17P_DRAIN_BEFORE_SAMPLE");
                  drain = (v && v[0]) ? atoi(v) : 3;
               }
               /* 1 = retire everything outstanding (correct but a big hammer).
                * 2 = retire only the generations held by PREVIOUS TEXTURE
                *     SAMPLES, which is the minimum the shared-publication
                *     story requires: sample N must not publish its index
                *     while sample N-1 is still using the shared state. */
               /* 3 = retire only this sample's OWN planned wait, on a
                *     carrier, and clear it. That is the minimal form: the
                *     sample's staging copies are what would otherwise carry
                *     the wait in their own encoding, and that field is the
                *     one that does not take effect. Measured identical to
                *     mode 1 on the CTS shape (both add exactly the two
                *     carriers for the two samples whose wait is non-zero)
                *     without serialising unrelated outstanding work. */
               /* A LOD query is not a staged sample: it has its own operand
                * and slot protocol, and retiring its wait here regressed all
                * 45 dEQP-VK.glsl.texture_functions.query.texturequerylod
                * cases while fixing the one array-shadow case. Leave it
                * alone. */
               /* PERSP_MUL/FADD coordinate publications already belong to a
                * native transaction. SMP consumes that transaction through
                * its coordinate-slot wait below; moving the wait to a
                * carrier would retire it before the sample can consume it.
                * Only the late COPY publication needs the carrier. */
               const bool staged_sample =
                  !I->query_lod && !I->agx3_tex_coords_staged;
               const uint16_t live =
                  !staged_sample ? 0
                  : drain == 3 ? wait
                  : drain == 2 ? (uint16_t)(outstanding_here & sample_slots_live)
                  : drain == 1 ? (uint16_t)(wait | outstanding_here)
                               : 0;
               if (live) {
                  if (!agx3_retire_slots_noop(emission, enc, live))
                     AGX3_REFUSE(I);
                  wait &= (uint16_t)~live;
                  outstanding &= (uint16_t)~live;
                  sample_slots_live &= (uint16_t)~live;
               }
            }
            assert(texture_ops_left > 0);
            const bool final = --texture_ops_left == 0;
            assert(texture_samples_remaining > 0);
            texture_samples_remaining--;
            const bool sample_group_member =
               I->lod_mode != AGX_LOD_MODE_LOD_GRAD &&
               I->lod_mode != AGX_LOD_MODE_LOD_GRAD_MIN;
            const bool sample_group_nonterminal =
               (g17p_sample_group && sample_group_member &&
                sample_group_members_remaining > 1) ||
               /* Compute gathers use level zero without a fragment group. */
               (enc == AGX_ISA_ENCODING_AGX3B &&
                ctx->stage == MESA_SHADER_COMPUTE &&
                I->gather != AGX_GATHER_NONE);
            bool sample_group_end = false;
            if (sample_group_member) {
               assert(sample_group_members_remaining > 0);
               sample_group_end = --sample_group_members_remaining == 0 &&
                                  g17p_sample_group;
            }
            /* Apple's implicit-derivative fragment programs enter the pixel
             * transaction immediately before their first SMP, after their
             * coefficient interpolation.  Opening it only in the colour
             * epilog is too late: a scissor edge may then retire uncovered
             * lanes before SMP forms its derivative quad, selecting the next
             * mip on odd-aligned rectangles.  Keep explicit-LOD/gradient
             * samples outside this rule; they do not consume a helper quad.
             *
             * A split fragment epilog carries the same entry gate for its
             * tile write.  Publishing this main-part entry through the
             * existing link metadata lets agx_fast_link remove that later
             * duplicate, leaving the one native gate at the first derivative
             * consumer.  The native corpus is straight-line; do not place an
             * opener inside one control-flow arm and then deduplicate the
             * unconditional epilog opener. */
            const bool implicit_derivatives =
               I->lod_mode == AGX_LOD_MODE_AUTO_LOD ||
               I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS ||
               I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS_MIN;
            const bool native_staged_coords = I->agx3_tex_coords_staged;
            const bool open_pixel =
               enc == AGX_ISA_ENCODING_AGX3B &&
               ctx->stage == MESA_SHADER_FRAGMENT &&
               !ctx->key->fs.ignore_tib_dependencies &&
               ctx->agx3_colour_link.straight_line &&
               implicit_derivatives && !g17p_colour_open;
            /* The implicit-LOD form needs the 2x2 quad's derivatives; every
             * witness is a fragment shader and the hardware has no quad
             * outside one. Apple's compute and vertex explicit-level forms
             * are byte-identical and do not consume derivatives. A compute
             * kernel also stages a runtime level exactly as a fragment
             * function does (metal-nt -arch applegpu_g17p, level(l) with l
             * computed in the kernel):
             *
             *   texture2d         1705540600080004f0c00800
             *                     05800cb8b0000001000010000100
             *   texture2d_array   1705540600040c00f0c02500
             *                     05800cb8b0000801000010000100
             *   texturecube       1705540600040c00f0c01100
             *                     05800cb8b0000003000010000100
             *   texturecube_array 1705540003140000f0880400
             *                     1705540600040c00f0c21700
             *                     05800cb8b0000803000010000100
             */
            const bool explicit_level = ctx->stage != MESA_SHADER_FRAGMENT &&
               I->lod_mode == AGX_LOD_MODE_LOD_MIN &&
               (I->src[1].type == AGX_INDEX_IMMEDIATE ||
                ctx->stage == MESA_SHADER_COMPUTE);
            /* An explicit gradient carries its own derivative, so it needs no
             * quad either; Apple's own witness for the form is a compute
             * kernel. */
            const bool compute_gradient = ctx->stage == MESA_SHADER_COMPUTE &&
               I->lod_mode == AGX_LOD_MODE_LOD_GRAD;
            /* Native runtime-indexed vertex pipelines ground both a per-lane
             * explicit level and explicit gradients in every noncube sample
             * dimension, including depth comparisons and array layers. */
            const bool vertex_explicit_lod =
               (ctx->stage == MESA_SHADER_VERTEX ||
                ctx->stage == MESA_SHADER_GEOMETRY ||
                ctx->stage == MESA_SHADER_TESS_CTRL ||
                ctx->stage == MESA_SHADER_TESS_EVAL) &&
               (I->lod_mode == AGX_LOD_MODE_LOD_MIN ||
                I->lod_mode == AGX_LOD_MODE_LOD_GRAD);
            /* Sampler-private staging groups 0/8/16 (see the note in
             * agx3_emit_texture_sample). A plain 2D sample takes the lowest
             * group whose previous sample has retired; when all three still
             * hold in-flight samples, group 0's is retired first. Every other
             * G17P sample shape keeps its witnessed fixed banks, so it drains
             * the in-flight staged samples and then owns every group. */
            const bool rotatable_staging =
               enc == AGX_ISA_ENCODING_AGX3B && slot != 0 &&
               !native_staged_coords && I->dim == AGX_DIM_2D &&
               !I->query_lod && !(I->offset && !I->offset_xyz) &&
               (I->lod_mode == AGX_LOD_MODE_AUTO_LOD ||
                ((I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS ||
                  I->lod_mode == AGX_LOD_MODE_LOD_MIN) &&
                 I->src[1].type == AGX_INDEX_IMMEDIATE));
            uint16_t bank_base = 0;
            if (enc == AGX_ISA_ENCODING_AGX3B) {
               uint16_t busy[3];
               for (unsigned g = 0; g < 3; ++g)
                  busy[g] = stage_group_slot[g]
                               ? (uint16_t)(1u << (stage_group_slot[g] - 1))
                               : 0;
               if (rotatable_staging) {
                  if (!agx3_take_staging_group(emission, enc,
                                               stage_group_slot, &outstanding,
                                               outstanding_here, slot,
                                               &bank_base))
                     AGX3_REFUSE(I);
               } else {
                  uint16_t retire = 0;
                  for (unsigned g = 0; g < 3; ++g)
                     retire |= (uint16_t)(outstanding_here & busy[g]);
                  if (agx3_g17p_force_staging_drain())
                     retire = (uint16_t)(busy[0] | busy[1] | busy[2]);
                  if (agx3_g17p_force_staging_drain())
                     fprintf(stderr,
                             "G17Pstage dim=%u shadow=%u slot=%u wait=0x%x "
                             "outst_here=0x%x grp=[%u,%u,%u] busy=[0x%x,0x%x,0x%x] "
                             "retire=0x%x bank_base=%u rot=%u\n",
                             I->dim, I->shadow, slot, wait, outstanding_here,
                             stage_group_slot[0], stage_group_slot[1],
                             stage_group_slot[2], busy[0], busy[1], busy[2],
                             retire, bank_base, rotatable_staging);
                  if (retire) {
                     if (!agx3_retire_slots_noop(emission, enc, retire))
                        AGX3_REFUSE(I);
                     outstanding &= (uint16_t)~retire;
                     if (slot)
                        outstanding |= (uint16_t)(1u << (slot - 1));
                  }
                  for (unsigned g = 0; g < 3; ++g)
                     stage_group_slot[g] = slot;
               }
            }
            uint16_t sample_wait = wait;
            if (native_staged_coords) {
               if (!native_coord_slot)
                  AGX3_REFUSE(I);
               const uint16_t coord_wait =
                  (uint16_t)(1u << (native_coord_slot - 1));
               if (!(sample_wait & coord_wait))
                  AGX3_REFUSE(I);

               /* The SMP has one witnessed wait generation. Coordinate
                * staging owns it; drain independent descriptor producers
                * first, then let SMP consume the intact helper-lane staging
                * transaction. */
               const uint16_t other_waits = sample_wait & ~coord_wait;
               if (other_waits &&
                   !agx3_retire_slots_noop(emission, enc, other_waits))
                  AGX3_REFUSE(I);
               sample_wait = coord_wait;
            }
            /* CEM has no scoreboard-wait fields. Keep a register-descriptor
             * producer already assigned slot 6 active so CEM's finalizers and
             * face publication join its native transaction. Retire every
             * other producer before entering the compound pipeline. */
            if ((I->dim == AGX_DIM_CUBE ||
                 I->dim == AGX_DIM_CUBE_ARRAY) &&
                sample_wait != 0) {
               const bool register_operands =
                  I->src[3].type == AGX_INDEX_REGISTER ||
                  I->src[4].type == AGX_INDEX_REGISTER;
               const uint16_t native_wait =
                  register_operands ? (sample_wait & (1u << 5)) : 0;
               const uint16_t retire = sample_wait & ~native_wait;
               if (retire && !agx3_retire_slots_noop(emission, enc, retire))
                  AGX3_REFUSE(I);
               sample_wait = native_wait;
            }
            /* A native staging COPY carries one producer wait. Lower only
             * the extra RAW/WAW dependencies assigned to this sample, using
             * the same grounded wait carrier as other compound instructions.
             * Do not drain unrelated outstanding slots or change allocation.
             * Previously encodable zero/one-wait samples remain unchanged. */
            if (sample_wait & (sample_wait - 1)) {
               const uint16_t staging_wait = sample_wait & -sample_wait;
               if (!agx3_retire_slots_noop(emission, enc,
                                           sample_wait & ~staging_wait))
                  AGX3_REFUSE(I);
               sample_wait = staging_wait;
            }
            if ((ctx->stage != MESA_SHADER_FRAGMENT && !explicit_level &&
                 !compute_gradient && !vertex_explicit_lod &&
                 !(enc == AGX_ISA_ENCODING_AGX3B &&
                   I->gather != AGX_GATHER_NONE)) ||
                /* Keep the pixel-entry gate after all native coordinate
                 * staging and immediately before SMP, matching Apple. */
                !agx3_emit_texture_sample(emission, enc, I, slot, sample_wait,
                                          final, sample_group_nonterminal,
                                          sample_group_end, open_pixel,
                                          &ctx->agx3_colour_link.entry_offset,
                                          native_coord_slot, cem_scratch,
                                          ctx->stage == MESA_SHADER_VERTEX,
                                          ctx->stage == MESA_SHADER_COMPUTE,
                                          ms_array_scratch, bank_base))
               AGX3_REFUSE(I);
            if (open_pixel) {
               ctx->agx3_colour_link.entry_emitted = true;
               ctx->agx3_colour_link.entry_offset -= initial_size;
               g17p_colour_open = true;
            }
            /* ShaderInfo fields 38/42 describe the sampler-private compiler
             * footprint in bytes, not the highest GPR supplying an operand.
             * Fresh attachment-only 2D families report 8 bytes and publish
             * into staging[0]/[2], while cube, cube-array and cube-shadow
             * families report 16 bytes. Fragment gradient pairs report 32
             * bytes for 2D and 64 bytes for 3D, with or without an offset;
             * vertex gradients instead report 48 and 80 bytes. Earlier
             * 24-byte cube witnesses also wrote a device buffer whose
             * independent staging requirement dominated. Keep the extents
             * stage- and dimension-specific, independent of post-RA source
             * placement and scoreboard slot. */
            const bool staged_scalar =
               I->lod_mode == AGX_LOD_MODE_LOD_MIN ||
               I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS_MIN ||
               (I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS &&
                I->src[1].type != AGX_INDEX_IMMEDIATE);
            const bool staged_compare = I->shadow;
            const bool staged_sample_offset = I->offset && I->offset_xyz == 0;
            ctx->agx3_staging_halfs =
               MAX2(ctx->agx3_staging_halfs,
                    bank_base + ((I->dim == AGX_DIM_CUBE ||
                     I->dim == AGX_DIM_CUBE_ARRAY) ? 8
                    : native_staged_coords ? 4
                    : vertex_explicit_lod
                       ? (I->lod_mode == AGX_LOD_MODE_LOD_GRAD
                             ? (I->dim == AGX_DIM_3D ? 40 : 24)
                             : 16)
                    : (I->lod_mode == AGX_LOD_MODE_LOD_GRAD ||
                       I->lod_mode == AGX_LOD_MODE_LOD_GRAD_MIN)
                         ? (I->dim == AGX_DIM_3D ? 32 : 16)
                    : (staged_scalar || staged_sample_offset ||
                       (I->dim == AGX_DIM_2D_ARRAY && !I->query_lod) ||
                       I->dim == AGX_DIM_3D)
                       ? 8
                    : staged_compare ? 6
                    : enc == AGX_ISA_ENCODING_AGX3B ? 4
                                                   : I->src[0].value + 4));
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
            /* Multi-RT colour waits in the generic IR do not identify a tile
             * partition.  Retire their scoreboard dependencies here; the
             * exact partition-specific wait is emitted immediately beside
             * its LLDCol/LSTCol below. */
            if (g17p_mrt.valid && I->op == AGX_OPCODE_WAIT_PIX &&
                I->writeout != 1) {
               if ((g17p_mrt.has_loads || ctx->key->g17p_emrt) &&
                   !g17p_colour_open) {
                  uint8_t entry[AGX3_PIXEL_MAX_LENGTH];
                  uint8_t entry_len = 0;
                  if (!agx3_pixel_colour_entry(enc, entry, &entry_len) ||
                      !agx3_append(emission, entry, entry_len))
                     AGX3_REFUSE(I);
                  ctx->agx3_colour_link.entry_emitted = true;
                  ctx->agx3_colour_link.entry_offset =
                     emission->size - entry_len - initial_size;
                  g17p_colour_open = true;
               }
               if (!agx3_retire_slots_noop(emission, enc, wait))
                  AGX3_REFUSE(I);
               /* The single combined acquire ahead of the stores is the
                * multisample eMRT form: Apple's 2x/4x multi-target fragments
                * hold every stored partition's classes across the whole
                * unrolled sample loop. A one-sample fragment brackets each
                * partition around its own tile access instead, which is what
                * the LLDCol/LSTCol path below emits, so leave the classes
                * unacquired here and let it do that. Anything else -- a
                * multisample eMRT epilog that does not store every partition
                * -- still fails closed. */
               if (ctx->key->g17p_emrt && !g17p_mrt_acquired &&
                   g17p_mrt.multisample) {
                  uint8_t bytes[AGX3_PIXEL_MAX_LENGTH], len;
                  if (!g17p_mrt.multisample_combined ||
                      !agx3_pixel_wait(enc, g17p_mrt.combined_classes, 0,
                                       bytes, &len) ||
                      !agx3_append(emission, bytes, len))
                     AGX3_REFUSE(I);
                  g17p_mrt_acquired = true;
               }
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
            const bool colour_entry_wait =
               I->writeout == 1 ||
               /* Dead blend reads can leave separate class-8/class-4
                * waits around a store-only shader. Acquire before the
                * first surviving colour wait, preserving both waits. */
               (g17p_colour_protocol && (I->writeout & 0xc)) ||
               I->writeout == (g17p_blend_protocol ? 8 : 0xc);
            if (g17p_pixel_protocol && I->op == AGX_OPCODE_WAIT_PIX &&
                colour_entry_wait &&
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
            if (!agx3_emit_pixel(emission, enc, I, wait,
                                 I == g17p_final_release))
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
            if (g17p_mrt.valid && g17p_mrt.multisample_combined) {
               /* Apple's 2x/4x fragments of two to eight targets acquire every
                * stored partition's classes in one PIXWAIT (60, or 252 from
                * three targets), access the partitions in descending order
                * through the coverage register and release those classes once
                * (msaa/native/persample/msmrt*.py). Blended fragments open the
                * same lifetime at their first load and keep it through the
                * final store. Fragments writing only some targets bracket each
                * partition as at one sample (mssubset.py). */
               uint8_t bytes[AGX3_PIXEL_MAX_LENGTH];
               uint8_t len = 0;
               const bool load = I->op == AGX_OPCODE_LD_TILE;
               /* A lone active sparse attachment retains its own classes;
                * only logical target zero owns the class-8/class-4 pair.
                * An eMRT epilog is excluded: its combined acquire has already
                * been emitted at the colour WAIT_PIX above, and splitting the
                * lifetime again here would acquire class 8 and class 4 a
                * second time inside it. */
               const bool single_blend =
                  g17p_mrt.count == 1 && g17p_mrt.targets[0] == 0 &&
                  g17p_mrt.has_loads && !ctx->key->g17p_emrt;
               const bool acquire = single_blend
                  ? (load ? first : g17p_mrt_stores_seen == 0)
                  : !g17p_mrt_acquired &&
                     (load ? first : (!g17p_mrt.has_loads &&
                                      g17p_mrt_stores_seen == 0));
               const uint32_t acquire_classes = single_blend
                  ? (load ? 8 : 4) : g17p_mrt.combined_classes;
               if (!g17p_colour_open) {
                  if (!agx3_pixel_colour_entry(enc, bytes, &len) ||
                      !agx3_append(emission, bytes, len))
                     AGX3_REFUSE(I);
                  ctx->agx3_colour_link.entry_emitted = true;
                  ctx->agx3_colour_link.entry_offset =
                     emission->size - len - initial_size;
                  g17p_colour_open = true;
               }
               if ((acquire &&
                    (!agx3_pixel_wait(enc, acquire_classes, 0,
                                      bytes, &len) ||
                     !agx3_append(emission, bytes, len))) ||
                   !agx3_emit_tile(emission, enc, I, slot, wait, false, false,
                                   g17p_mrt.raw32_store,
                                   /* Native per-sample exports use halves
                                    * 0..31 even for one/two targets. */
                                   true,
                                   ctx->key->g17p_emrt && !load &&
                                      I->src[0].value >= 16) ||
                   (!load && last &&
                    (!agx3_pixel_release(enc, g17p_mrt.combined_classes, true,
                                         0, bytes, &len) ||
                     !agx3_append(emission, bytes, len))))
                  AGX3_REFUSE(I);
               g17p_mrt_stores_seen += !load;
               g17p_mrt_acquired |= acquire;
               continue;
            }
            if (g17p_mrt.valid) {
               const int partition =
                  agx3_g17p_mrt_partition(&g17p_mrt, I->pixel_offset);
               uint32_t read_class, write_class, combined_class;
               if (partition < 0 ||
                   !agx3_pixel_colour_partition_classes(
                      agx3_g17p_mrt_class_pair(g17p_mrt.targets[partition]), &read_class,
                      &write_class, &combined_class))
                  AGX3_REFUSE(I);

               uint8_t bytes[AGX3_PIXEL_MAX_LENGTH];
               uint8_t len = 0;
               if (!g17p_colour_open) {
                  if (!agx3_pixel_colour_entry(enc, bytes, &len) ||
                      !agx3_append(emission, bytes, len))
                     AGX3_REFUSE(I);
                  ctx->agx3_colour_link.entry_emitted = true;
                  ctx->agx3_colour_link.entry_offset =
                     emission->size - len - initial_size;
                  g17p_colour_open = true;
               }

               const bool load = I->op == AGX_OPCODE_LD_TILE;
               /* Class 12 opens before the first load or store using it and
                * stays held until the last store; the other pairs bracket
                * their own loads and store. */
               const bool shared =
                  agx3_g17p_mrt_class_pair(g17p_mrt.targets[partition]) == 0;
               const bool loaded = shared ? g17p_mrt.shared_loaded
                                          : g17p_mrt.loads[partition];
               const uint32_t acquire =
                  load ? read_class : loaded ? write_class : combined_class;
               const bool acquire_class =
                  !shared || partition == (load ? g17p_mrt.shared_load_first
                                                : g17p_mrt.shared_first);
               const bool release_class =
                  !shared || partition == g17p_mrt.shared_last;
               const bool store_f13 =
                  !load && g17p_mrt_stores_seen++ < g17p_mrt.f13_stores;
               /* A multisample store keeps its own coverage dependency. */
               const bool store_waits = !acquire_class || g17p_mrt.multisample;
               if ((acquire_class &&
                    (!agx3_pixel_wait(enc, acquire, store_waits ? 0 : wait,
                                      bytes, &len) ||
                     !agx3_append(emission, bytes, len))) ||
                   !agx3_emit_tile(emission, enc, I, slot,
                                   store_waits ? wait : 0, false, false,
                                   g17p_mrt.raw32_store,
                                   g17p_mrt.count >= 3,
                                   store_f13) ||
                   (!load && release_class &&
                    (!agx3_pixel_release(enc, combined_class, last, 0, bytes,
                                        &len) ||
                     !agx3_append(emission, bytes, len))))
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
               if (!agx3_emit_pixel(emission, enc, g17p_deferred_write, 0,
                                    false))
                  AGX3_REFUSE(I);
            }
            /* Apple's per-sample RGBA32 fragments store from halves 24 and
             * 16 with f13 clear, as the multi-target backgrounds do. */
            if (!agx3_emit_tile(emission, enc, I, slot, wait, first, last,
                                g17p_multi_background ||
                                   g17p_sample_colour_stores,
                                (g17p_multi_background ||
                                 g17p_msaa_background_targets > 1) &&
                                   ctx->key->g17p_mrt_bg,
                                false))
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
            /* SR2 is the one-sample coverage word. A multisample depth
             * export writes each sample's depth separately in Apple's
             * fragments, which this single update does not do. */
            if (enc != AGX_ISA_ENCODING_AGX3B || ctx->stage != MESA_SHADER_FRAGMENT ||
                ctx->key->fs.nr_samples > 1 ||
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
            fragment_input_started = true;
            continue;
         }

         if (I->op == AGX_OPCODE_ZS_EMIT_AGX3) {
            const bool depth = I->zs & 1, stencil = I->zs & 2;
            /* Apple's 2x/4x per-sample depth/stencil writes stage each word,
             * and its depth, after the previous sample's, all under the one
             * class-1 wait taken before the first DEPTHF (msaa/native/zexp2
             * fs_id_{2,4}x, msaa/native/zmrt/zsten fs_id_s_4x). */
            const unsigned staging = g17p_zs_staging;
            if (enc != AGX_ISA_ENCODING_AGX3B ||
                ctx->stage != MESA_SHADER_FRAGMENT || I->nr_dests ||
                I->nr_srcs != 1 || I->zs < 1 || I->zs > 3 ||
                !agx3_vary_reg32(I->src[0]) ||
                agx_channels(I->src[0]) != (depth ? 2 : 1) ||
                (g17p_zs_writes == 0) != (g17p_deferred_zs_wait != NULL) ||
                g17p_zs_writes >= 4)
               AGX3_REFUSE(I);

            uint8_t bytes[AGX3_STAGE_MAX_LENGTH];
            uint8_t len = 0;
            if (!g17p_colour_open) {
               if (!agx3_pixel_colour_entry(enc, bytes, &len) ||
                   !agx3_append(emission, bytes, len))
                  AGX3_REFUSE(I);
               ctx->agx3_colour_link.entry_emitted = true;
               ctx->agx3_colour_link.entry_offset =
                  emission->size - len - initial_size;
               g17p_colour_open = true;
            }
            /* Native DEPTHF producers publish both staging dwords with f13.
             * Keep that flag independent of the allocator's GPR last-use.
             * Apple's stencil-only [[stencil]] fragment publishes its single
             * coverage/stencil dword without it. */
            if (!agx3_stage_copy_liveness(enc, staging, I->src[0].value,
                                        depth, I->src[0].discard, slot,
                                        wait | outstanding_here, bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            if (depth &&
                (!agx3_stage_copy_liveness(enc, staging + 2,
                                           I->src[0].value + 2, true,
                                           I->src[0].discard, slot, 0,
                                           bytes, &len) ||
                 !agx3_append(emission, bytes, len)))
               AGX3_REFUSE(I);
            if (g17p_deferred_zs_wait &&
                !agx3_emit_pixel(emission, enc, g17p_deferred_zs_wait, 0,
                                 false))
               AGX3_REFUSE(I);
            g17p_deferred_zs_wait = NULL;

            const struct agx3_zs_desc zs = {
               .value = staging,
               .value_kill = true,
               .slot = slot,
               .wait_mask = (uint16_t)(1u << (slot - 1)),
            };
            if (!agx3_zs_encode_depth_stencil(enc, &zs, depth, stencil, bytes,
                                              &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);

            outstanding = 0;
            g17p_zs_writes++;
            g17p_zs_staging += depth ? 4 : 2;
            ctx->agx3_staging_halfs =
               MAX2(ctx->agx3_staging_halfs, staging + (depth ? 4 : 2));
            continue;
         }

         if (I->op == AGX_OPCODE_GET_SR_COVERAGE) {
            /* Fresh Apple G17P fragment-atomic pipeline
             * d150404ac02e.../32cf4963515f... reads SR2 as W16 before the
             * helper-thread predicate (`04 84 10 06`). This is the exact
             * active-lane source used to suppress fragment side effects on
             * helper invocations.
             *
             * simd_is_first() reads SR34, the active thread index in the
             * subgroup, as W16 in compute, vertex and fragment shaders. Apple
             * sets f48 on a fragment read only when helper invocations never
             * need the result, so leave it clear.
             *
             * A uniform-address atomic_fetch_add_explicit(c, 7) in a compute
             * or fragment shader performs the whole update from lane SR34 ==
             * 0 as `imadd(SR36, #7)`, so SR36 is the total number of active
             * threads in the subgroup, also W16. Other coverage
             * pseudo-registers remain closed until separately witnessed. */
            const bool active_thread =
               I->sr == AGX_SR_IS_ACTIVE_THREAD &&
               ctx->stage == MESA_SHADER_FRAGMENT;
            const bool sample_coverage =
               I->sr == AGX_SR_INPUT_SAMPLE_MASK &&
               ctx->stage == MESA_SHADER_FRAGMENT;
            /* A G17P tessellation-control program is a compute dispatch:
             * hk_launch_tess() dispatches it, and hk_shader.c already selects
             * the compute texture/sampler heaps for it on exactly that
             * grounds ("Poly executes geometry variants through CDM, so they
             * use the native indexed texture/sampler heaps just like compute
             * shaders"). The stage label is a compiler-side artifact; the
             * hardware program is the same CDM kind already witnessed here,
             * so SR34/SR36 read the same subgroup state. Without this every
             * subgroup op in a TCS refused the shader -- 99 of the 3,912
             * dEQP-VK.subgroups.*tess_control mustpass cases -- which became
             * reachable when tessellationShader was enabled by default. */
            const bool subgroup_stage =
               ctx->stage == MESA_SHADER_COMPUTE ||
               ctx->stage == MESA_SHADER_VERTEX ||
               ctx->stage == MESA_SHADER_TESS_CTRL ||
               ctx->stage == MESA_SHADER_FRAGMENT;
            const bool active_index =
               I->sr == AGX_SR_ACTIVE_THREAD_INDEX_IN_SUBGROUP &&
               subgroup_stage;
            const bool active_count =
               I->sr == AGX_SR_TOTAL_ACTIVE_THREADS_IN_SUBGROUP &&
               subgroup_stage;
            if (enc != AGX_ISA_ENCODING_AGX3B ||
                !(active_thread || sample_coverage || active_index || active_count) ||
                I->nr_dests != 1 ||
                I->nr_srcs != 0 || !agx3_src_is_register(I->dest[0]) ||
                I->dest[0].memory || I->dest[0].size != AGX_SIZE_16 ||
                agx_channels(I->dest[0]) != 1)
               AGX3_REFUSE(I);
            const struct agx3_sr_desc d = {
               .dest = I->dest[0].value, .width = AGX3_SR_W16,
               .sr = sample_coverage ? 39 : active_index ? 34 : active_count ? 36 : 2,
               .slot = slot,
               .wait_mask = wait,
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
                !agx3_emit_pixel(emission, enc, g17p_deferred_zs_wait, 0,
                                 false))
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

         if (I->op == AGX_OPCODE_SAMPLE_MASK_AGX3) {
            if (enc != AGX_ISA_ENCODING_AGX3B || ctx->stage != MESA_SHADER_FRAGMENT ||
                I->nr_dests || I->nr_srcs != 2 ||
                !agx3_src_is_register(I->src[0]) || I->src[0].memory ||
                I->src[0].size != AGX_SIZE_16 || agx_channels(I->src[0]) != 1 ||
                !agx3_vary_reg32(I->src[1]) || agx_channels(I->src[1]) != 1)
               AGX3_REFUSE(I);
            const bool force_early = I->imm & 1;
            if ((I->imm & ~3) || (force_early && !(I->imm & 2)))
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
            if ((I->imm & 2) && !force_early &&
                (!agx3_pixel_wait(enc, 1, 0, bytes, &len) ||
                 !agx3_append(emission, bytes, len)))
               AGX3_REFUSE(I);
            const struct agx3_zs_desc zs = {
               .value = 0, .value_kill = true, .slot = 1, .wait_mask = 1,
            };
            if (!agx3_zs_encode_sample_mask(enc, &zs, bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            /* A no-tests discard does not acquire class 1 above, so it
             * must not release it either. Native G17P kill_fetch and
             * kill_memory emit these early MOVMSKs without PIXREL(1),
             * then bracket the eventual testing update with one wait and
             * release. Keep that release paired with its acquire. */
            if ((I->imm & 2) && !force_early &&
                (!agx3_pixel_release(enc, 1, false, 0, bytes, &len) ||
                 !agx3_append(emission, bytes, len)))
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
            if (!agx3_translate_sr(enc, I, slot, wait, &d))
               AGX3_REFUSE(I);
            /* The coverage word is a fragment input. Apple's programs that
             * open with it put the entry marker on that read. */
            const bool coverage_input = enc == AGX_ISA_ENCODING_AGX3B &&
               ctx->stage == MESA_SHADER_FRAGMENT &&
               I->sr == AGX_SR_INPUT_SAMPLE_MASK;
            d.first = coverage_input && emission->size == initial_size &&
                      !fragment_input_started &&
                      !g17p_fragment_helper_protocol;
            if (!agx3_sr_encode(enc, &d, bytes, &len))
               AGX3_REFUSE(I);
            memcpy(util_dynarray_grow_bytes(emission, 1, len), bytes, len);
            fragment_input_started |= coverage_input;
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
            const bool returned = I->nr_dests == 1 &&
                                  !agx_is_null(I->dest[0]);
            const bool g17p_returned =
               enc == AGX_ISA_ENCODING_AGX3B && returned;
            const uint8_t atomic_slot = g17p_returned ? 1 : slot;
            /* A colourless fragment's returning atomic runs under pixel
             * class 4 and is followed by a barrier; see
             * agx3_pixel_atomic_acquire(). With a colour store, Apple takes
             * colour partition 0's write class instead, and the store then
             * acquires only its read class; that shared protocol is not
             * encoded, so colour-writing fragments keep neither. */
            const bool pixel_bracket = g17p_returned &&
               g17p_colourless_fragment;
            uint8_t pixel[AGX3_PIXEL_MAX_LENGTH], pixel_len = 0;
            /* The old value must be published above the spill frame */
            const unsigned return_tps_half =
               g17p_returned ? ctx->agx3_atomic_return_tps_half : 0;
            if ((g17p_returned &&
                 (slot != 6 || return_tps_half * 2 < ctx->scratch_size_B)) ||
                !agx3_translate_device_atomic(I, enc, block->divergent,
                                              atomic_slot, wait, &d) ||
                (pixel_bracket &&
                 (!agx3_pixel_atomic_acquire(enc, pixel, &pixel_len) ||
                  !agx3_append(emission, pixel, pixel_len))))
               AGX3_REFUSE(I);
            d.result_tps_half = return_tps_half;
            if (!agx3_atomic_encode_device(enc, &d, bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);

            if (g17p_returned) {
               /* Exact applegpu_g17p return sequence. For destination r0,
                * this composes to 0c8019a500200000, the eight-byte native
                * rld immediately following atomicdev. */
               const struct agx3_static_spill_load_desc reload = {
                  .dest = I->dest[0].value,
                  .tps_half = return_tps_half,
                  .width = AGX3_MEM_W32,
                  .slot = 6,
                  .slot_mirror = 6,
                  .wait_mask = 1u << 0,
               };
               uint8_t reload_bytes[AGX3_STATIC_SPILL_MAX_LENGTH];
               uint8_t reload_len = 0;
               if (!agx3_static_spill_encode_load_form(
                      enc, &reload, 1, reload_bytes, &reload_len) ||
                   !agx3_append(emission, reload_bytes, reload_len) ||
                   (pixel_bracket &&
                    (!agx3_pixel_atomic_barrier(enc, pixel, &pixel_len) ||
                     !agx3_append(emission, pixel, pixel_len))))
                  AGX3_REFUSE(I);
            }
            continue;
         }

         /* Threadgroup memory. AGX3 preprocessing has converted the byte
          * address to the element index consumed by the native ldtg/sttg
          * forms. */
         if (I->op == AGX_OPCODE_LOCAL_LOAD ||
             I->op == AGX_OPCODE_LOCAL_STORE) {
            const bool is_store = I->op == AGX_OPCODE_LOCAL_STORE;
            struct agx3_local_desc d;
            uint8_t bytes[AGX3_LOCAL_TPS_MAX_LENGTH];
            uint8_t len = 0;
            if (!agx3_translate_local(enc, I, is_store, slot, wait, &d))
               AGX3_REFUSE_WHY(I, "local: translate declined (operand shape)");
            const bool ok =
               is_store ? agx3_local_encode_store(enc, &d, bytes, &len)
                        : agx3_local_encode_load(enc, &d, bytes, &len);
            if (!ok)
               AGX3_REFUSE_WHY(I, "local: no grounded form for this "
                                  "width/mask/base combination");
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

         if (agx3_dynamic_stack_load(I)) {
            struct agx3_tps_desc desc;
            uint8_t bytes[AGX3_LOCAL_TPS_MAX_LENGTH], len;
            /* The address is unknown at compile time. Complete outstanding
             * static writes before reading; retain this read's alias token
             * until later static accesses have waited for it. */
            const uint16_t retire = wait | outstanding_here;
            if (!agx3_translate_dynamic_stack_load(I, slot, &desc) ||
                (retire && !agx3_retire_slots_noop(emission, enc, retire)) ||
                !agx3_tps_encode_load(enc, &desc, bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            outstanding &= ~retire;
            outstanding |= 1u << (slot - 1);
            continue;
         }

         if (agx3_dynamic_stack_store(I)) {
            struct agx3_tps_desc desc;
            uint8_t bytes[AGX3_LOCAL_TPS_MAX_LENGTH], len;
            /* The address is unknown at compile time: every outstanding
             * static copy or reload may touch the same location, so the
             * store waits for all of them in its own wait field (Apple's
             * STTPS carries the wait of the preceding static copies). */
            const uint16_t retire = wait | outstanding_here;
            if (!agx3_translate_dynamic_stack_store(I, retire, &desc) ||
                !agx3_tps_encode_store(enc, &desc, bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            outstanding &= ~retire;
            continue;
         }

         if (I->op == AGX_OPCODE_LOAD_CPROG_AGX3) {
            if (!cprog_imports || I->nr_dests != 1 || I->nr_srcs ||
                I->saturate || I->imm >= 512 ||
                !agx3_src_is_register(I->dest[0]) || I->dest[0].memory ||
                I->dest[0].abs || I->dest[0].neg || I->dest[0].kill ||
                I->dest[0].discard || agx_channels(I->dest[0]) != 1 ||
                (I->dest[0].size != AGX_SIZE_16 && I->dest[0].size != AGX_SIZE_32))
               AGX3_REFUSE(I);
            const struct agx3_cprog_import import = {
               .uniform_half = I->imm, .dest_half = I->dest[0].value,
               .width = I->dest[0].size == AGX_SIZE_16 ? AGX3_MEM_W16 : AGX3_MEM_W32,
               .slot = slot, .wait_mask = wait,
               .tps_live = tps_rereads[instr_index],
            };
            uint8_t bytes[AGX3_STATIC_SPILL_MAX_LENGTH];
            size_t length;
            if (!agx3_cprog_input_import(&ctx->out->agx3_preamble_inputs,
                                        &import, bytes, sizeof(bytes), &length) ||
                !agx3_append(emission, bytes, length))
               AGX3_REFUSE(I);
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
            if (!agx3_translate_static_spill(I, is_store, slot, wait,
                                             tps_rereads[instr_index], &store,
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
            const bool half = src.size == AGX_SIZE_16;
            /* f30 carries the source's absolute value, so `sqrt(abs(x))` --
             * the standard safe-sqrt idiom -- encodes without a separate
             * move. Only the 32-bit form is witnessed; the half step keeps
             * its narrower grounded domain. */
            if (I->nr_dests != 1 || I->nr_srcs != 1 || I->saturate ||
                !agx3_vary_reg32(dest) ||
                !agx3_src_is_register(src) || src.memory ||
                agx_channels(dest) != 1 || agx_channels(src) != 1 ||
                src.neg || src.kill || (src.abs && half) ||
                (src.cache && src.discard))
               AGX3_REFUSE(I);
            uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
            const bool ok =
               half ? agx3_alu_sqrt_step_half(enc, dest.value, src.value,
                                              agx3_post_ra_hint(src),
                                              true, wait, bytes, &len)
                    : (src.size == AGX_SIZE_32 && !(src.value & 1) &&
                       src.value < 256 &&
                       agx3_alu_sqrt_step_mod(enc, dest.value, src.value,
                                              agx3_post_ra_hint(src),
                                              dest.cache, wait, src.abs,
                                              bytes, &len));
            if (!ok || !agx3_append(emission,bytes,len))
               AGX3_REFUSE(I);
            continue;
         }

         /* Reciprocal widths are independent. In particular, mediump texture
          * data commonly feeds a full-precision divide in GTK's colour
          * conversion shaders. f10/f26 select the result/source widths, while
          * f30/f31 carry source ABS/NEG without a separate move. */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_RCP &&
             I->nr_dests == 1 && I->nr_srcs == 1 && !I->saturate) {
            const agx_index dest=I->dest[0], src=I->src[0];
            if (!agx3_src_is_register(dest) || dest.memory ||
                (dest.size != AGX_SIZE_16 && dest.size != AGX_SIZE_32) ||
                agx_channels(dest) != 1 ||
                !agx3_src_is_register(src) || src.memory ||
                (src.size != AGX_SIZE_16 && src.size != AGX_SIZE_32) ||
                agx_channels(src) != 1 || src.kill ||
                (src.cache && src.discard))
               AGX3_REFUSE(I);
            uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
            if (!agx3_alu_rcp(
                   enc, dest.value, src.value, dest.size == AGX_SIZE_16,
                   src.size == AGX_SIZE_16, src.abs, src.neg,
                   agx3_post_ra_hint(src), dest.cache, wait, bytes, &len) ||
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
            /* f30/f31 carry the source's ABS/NEG. inversesqrt(abs(x)) is the
             * safe form of the idiom and had no encoding at all. The half
             * form has no modifier witness, so it stays width-matched and
             * unmodified. */
            if (!agx3_src_is_register(dest) || dest.memory ||
                (dest.size != AGX_SIZE_16 && dest.size != AGX_SIZE_32) ||
                agx_channels(dest) != 1 ||
                !agx3_src_is_register(src) || src.memory ||
                src.size != dest.size || agx_channels(src) != 1 ||
                src.kill || (half && (src.abs || src.neg)) ||
                (src.cache && src.discard))
               AGX3_REFUSE(I);
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
            const bool ok =
               half ? agx3_alu_hrsqrt(enc, dest.value, src.value,
                                      agx3_post_ra_hint(src), dest.cache,
                                      wait, bytes, &len)
                    : agx3_alu_f32rsqrt_mod(enc, dest.value, src.value,
                                            agx3_post_ra_hint(src),
                                            dest.cache, wait, src.neg,
                                            src.abs, bytes, &len);
            if (!ok || !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         /* f30 and f31 are the source's absolute-value and negate modifiers,
          * witnessed as single-field differences on this exact form. NIR
          * folds a negation into a float source, so exp(-x) -- vkQuake's fog
          * term, in every one of its fragment shaders -- arrives here as an
          * exp2 of a negated register and refused the shader. Only the 32-bit
          * modifier forms are wired up. The half form also has a recovered
          * negate bit, used by mediump exp(-x). */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_EXP2 &&
             I->nr_dests == 1 && I->nr_srcs == 1 && !I->saturate) {
            const agx_index dest = I->dest[0], src = I->src[0];
            const bool half = dest.size == AGX_SIZE_16;
            if (!agx3_src_is_register(dest) || dest.memory ||
                (dest.size != AGX_SIZE_16 && dest.size != AGX_SIZE_32) ||
                agx_channels(dest) != 1 ||
                !agx3_src_is_register(src) || src.memory ||
                src.size != dest.size || agx_channels(src) != 1 ||
                (half && src.abs) || src.kill ||
                (src.cache && src.discard))
               AGX3_REFUSE(I);
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
            if (!(half ? agx3_alu_hexp2_neg(enc, dest.value, src.value,
                                            agx3_post_ra_hint(src), dest.cache,
                                            wait, src.neg, bytes, &len)
                       : agx3_alu_f32exp2_mod(enc, dest.value, src.value,
                                              agx3_post_ra_hint(src),
                                              dest.cache, wait, src.neg,
                                              src.abs, bytes, &len)) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         /* f10 and f26 independently select scalar result/source width. This
          * matters for mediump texture results promoted before logarithmic
          * transfer functions, as in GTK's component-transfer shader. */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_LOG2 &&
             I->nr_dests == 1 && I->nr_srcs == 1 && !I->saturate) {
            const agx_index dest = I->dest[0], src = I->src[0];
            if (!agx3_src_is_register(dest) || dest.memory ||
                (dest.size != AGX_SIZE_16 && dest.size != AGX_SIZE_32) ||
                agx_channels(dest) != 1 ||
                !agx3_src_is_register(src) || src.memory ||
                (src.size != AGX_SIZE_16 && src.size != AGX_SIZE_32) ||
                agx_channels(src) != 1 || src.kill ||
                (src.cache && src.discard))
               AGX3_REFUSE(I);
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
            if (!agx3_alu_log2(
                   enc, dest.value, src.value, dest.size == AGX_SIZE_16,
                   src.size == AGX_SIZE_16, src.abs, src.neg,
                   agx3_post_ra_hint(src), dest.cache, wait, bytes, &len) ||
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
                                         agx3_post_ra_hint(src),
                                         dest.cache, wait, bytes, &len)
                       : agx3_alu_f32floor(enc, dest.value, src.value,
                                           agx3_post_ra_hint(src), dest.cache,
                                           wait, src.neg,
                                           src.abs, bytes, &len)) ||
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

         if (enc == AGX_ISA_ENCODING_AGX3B &&
             (I->op == AGX_OPCODE_TRUNC ||
              I->op == AGX_OPCODE_ROUNDEVEN) &&
             I->nr_dests == 1 && I->nr_srcs == 1 && !I->saturate &&
             agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
             I->dest[0].size == AGX_SIZE_16 &&
             agx_channels(I->dest[0]) == 1 &&
             agx3_src_is_register(I->src[0]) && !I->src[0].memory &&
             I->src[0].size == AGX_SIZE_16 &&
             agx_channels(I->src[0]) == 1 && !I->src[0].abs &&
             !I->src[0].neg && !I->src[0].kill &&
             !(I->src[0].cache && I->src[0].discard)) {
            const agx_index dest = I->dest[0], src = I->src[0];
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
            const bool ok = I->op == AGX_OPCODE_TRUNC
               ? agx3_alu_htrunc(enc, dest.value, src.value,
                                 agx3_post_ra_hint(src), dest.cache, wait,
                                 bytes, &len)
               : agx3_alu_hroundeven(enc, dest.value, src.value,
                                     agx3_post_ra_hint(src), dest.cache, wait,
                                     bytes, &len);
            if (!ok || !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         /* A 32-bit roundeven had no handler here and fell through to the
          * select core, whose f32rint encoding point holds its source in a
          * six-bit field and therefore refuses any source above r63. That is
          * not a rare shape: hk lowers every float texture result through a
          * scaled-view recovery whose integer fetch ends in fround_even, and
          * one unrolled loop of texelFetch() puts those sources well past
          * r63. A refusal is fatal -- the shader does not compile and
          * vkCreateComputePipelines fails -- so it cost 106 robustness2
          * cases. Use the f32rint form family that floor and ceil already
          * use, which carries the family's full register fields, its f30/f31
          * source modifiers and its wait slots. Shapes this does not claim
          * keep their existing route. */
         if (enc == AGX_ISA_ENCODING_AGX3B &&
             I->op == AGX_OPCODE_ROUNDEVEN &&
             I->nr_dests == 1 && I->nr_srcs == 1 && !I->saturate &&
             agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
             I->dest[0].size == AGX_SIZE_32 &&
             agx_channels(I->dest[0]) == 1 &&
             agx3_src_is_register(I->src[0]) && !I->src[0].memory &&
             I->src[0].size == AGX_SIZE_32 &&
             agx_channels(I->src[0]) == 1 && !I->src[0].kill &&
             !(I->src[0].cache && I->src[0].discard)) {
            const agx_index dest = I->dest[0], src = I->src[0];
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
            if (!agx3_alu_f32roundeven(enc, dest.value, src.value,
                                       agx3_post_ra_hint(src), dest.cache,
                                       wait, src.neg, src.abs, bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         if (I->op == AGX_OPCODE_BITTEST_SELECT_AGX3) {
            struct agx3_bittest_desc d = {
               .dest = I->dest[0].value,
               .mask = I->imm,
               .wait_mask = wait,
            };
            if (I->nr_dests != 1 || I->nr_srcs != 3 ||
                !agx3_src_is_register(I->dest[0]) ||
                I->dest[0].size != AGX_SIZE_32 || I->dest[0].memory ||
                I->imm > UINT8_MAX || I->src[0].abs || I->src[0].neg)
               AGX3_REFUSE(I);
            for (unsigned s = 0; s < 3; ++s) {
               if (!agx3_src_is_register(I->src[s]) ||
                   I->src[s].size != AGX_SIZE_32 || I->src[s].memory ||
                   I->src[s].abs)
                  AGX3_REFUSE(I);
               d.src[s] = I->src[s].value;
               d.cache[s] = I->src[s].cache;
               d.discard[s] = I->src[s].discard;
               if (s)
                  d.negate[s - 1] = I->src[s].neg;
            }
            uint8_t bytes[AGX3_ISEL_MAX_LENGTH], len = 0;
            if (!agx3_bittest_encode(enc, &d, bytes, &len) ||
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
         const int uniform_integer = agx3_try_uniform_integer(emission, enc, I, wait);
         if (uniform_integer < 0)
            AGX3_REFUSE(I);
         if (uniform_integer > 0)
            continue;

         if (I->op == AGX_OPCODE_IMAD && I->nr_dests == 1 && I->nr_srcs >= 3 &&
             I->shift == 0 && !I->saturate &&
             agx3_src_is_register(I->dest[0]) &&
             !I->dest[0].memory &&
             (I->dest[0].size == AGX_SIZE_16 ||
              I->dest[0].size == AGX_SIZE_32 ||
              I->dest[0].size == AGX_SIZE_64) &&
             agx_channels(I->dest[0]) == 1) {
            /* Multiplication commutes: NIR may place an immediate first. */
            const bool constant_first = I->src[0].type == AGX_INDEX_IMMEDIATE &&
                                        I->src[1].type == AGX_INDEX_REGISTER;
            agx_index a = I->src[constant_first ? 1 : 0];
            agx_index m = I->src[constant_first ? 0 : 1];
            const agx_index c = I->src[2];

            /* Multiplication commutes. Apple keeps the 32-bit operand of a
             * mixed-width 32-bit product in slot A and the half in slot B. */
            if (I->dest[0].size == AGX_SIZE_32 && a.size == AGX_SIZE_16 &&
                agx3_src_is_register(m) && m.size == AGX_SIZE_32) {
               const agx_index half = a;
               a = m;
               m = half;
            }

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
            const bool widening = !narrow && !wide;

            /* The multiplier is either an 8-bit immediate or a register of
             * the result width, plus the paired 16x16->32 widening form.
             * Both 16-bit and 32-bit forms are witnessed byte-exactly. */
            const bool mult_imm = m.type == AGX_INDEX_IMMEDIATE && !m.abs &&
                                  !m.neg && m.value <= 0xff;
            const bool widening_half_multiplier =
               widening && m.size == AGX_SIZE_16 && !m.neg;
            const bool mult_reg =
               agx3_src_is_register(m) && !m.memory &&
               ((m.size == (narrow ? AGX_SIZE_16 : AGX_SIZE_32) &&
                 (wide || !m.abs) && !m.neg) ||
                widening_half_multiplier) &&
               agx_channels(m) == 1;

            /* An immediate-zero addend is the pure-multiply form, whose
             * addend operand is absent rather than zero. Native constant
             * and register multiplier variants have distinct baselines. */
            const bool no_addend = c.type == AGX_INDEX_IMMEDIATE &&
                                   c.value == 0 && !c.abs && !c.neg;
            /*
             * `.neg` on an IMAD addend is an arithmetic negate, not a source
             * modifier: agx_pack.c gives it a dedicated bit and permits it on
             * src[2] alone.  AGX3 spells the negated form as its own
             * instruction, `imsub`, so admit the modifier here and let
             * agx3_imad_encode() select the opcode.  Without this every
             * `a*b - imm` was refused outright -- 2,692 of 6,486 refused
             * instructions in the 2026-09-11 CTS census, the single largest
             * shape.
             */
            const bool addend_imm =
               c.type == AGX_INDEX_IMMEDIATE && c.value != 0 &&
               c.value <= 0xff && !c.abs && !c.memory;
            /*
             * A bare narrow operand is SIGN-extended, the same rule the
             * comment above names and the same one agx_optimizer_signext()
             * enforces for iadd: it fuses a SIGNEXT into the multiply-add by
             * dropping it and asserting the survivor carries no ABS. So
             * `a*b + int(short c)` arrives with no modifier at all, which
             * this gate used to refuse.
             *
             * Apple, one kernel per variable, everything else held fixed:
             *
             *   a*b + int(short  c)   9f 00 56 00 02 08 00 40 d0 2e 6a 00
             *   a*b + int(ushort c)   9f 00 56 00 02 08 00 40 d0 2e 2a 00
             *   a*b + int c           9f 00 56 00 02 08 00 40 d0 2f 2a 00
             *
             * f54 is slot C's width (1 = 16-bit, 2 = 32-bit) and f58 its
             * sign-extension, set for `short` and clear for `ushort`.
             */
            const bool addend_ok =
               no_addend || addend_imm ||
               (agx3_src_is_register(c) && !c.memory &&
                agx_channels(c) == 1 && !(c.abs && c.neg) &&
                ((!narrow && c.size == AGX_SIZE_32) ||
                 (wide && c.size == AGX_SIZE_64) ||
                 (c.size == AGX_SIZE_16 &&
                   (narrow ? (!c.abs && !c.neg) : (c.abs || c.neg)))));
            const bool addend_reg = !no_addend && !addend_imm;

            const bool widening_half_a =
               widening && a.size == AGX_SIZE_16 && !a.neg;
            const bool a_ok = agx3_src_is_register(a) && !a.memory &&
               agx_channels(a) == 1 &&
               (narrow ? (a.size == AGX_SIZE_16 && !a.abs && !a.neg)
                       : (((wide || !a.abs) && !a.neg &&
                           a.size == AGX_SIZE_32) || widening_half_a));
            /* Each half operand of a 32-bit product widens by its own sign.
             * Apple pairs a half register multiplier only with no addend or
             * a 32-bit register addend, and a half multiplicand times a
             * constant with any register addend. */
            const bool half_a = widening && a.size == AGX_SIZE_16;
            const bool half_m = widening && mult_reg && m.size == AGX_SIZE_16;
            const bool half_operands_ok =
               (!half_a && !half_m) ||
               (!addend_imm &&
                (no_addend || c.size == AGX_SIZE_32 ||
                 (!half_m && c.size == AGX_SIZE_16)));

            if (a_ok && (mult_imm || mult_reg) && addend_ok &&
                half_operands_ok) {
               const struct agx3_imad_desc d = {
                  .dest = I->dest[0].value,
                  .dest_16 = narrow,
                  .dest_64 = wide,
                  .dest_cache = I->dest[0].cache,
                  .src_a = a.value,
                  .src_a_16 = a.size == AGX_SIZE_16,
                  .src_a_sign_extend = (wide || half_a) && !a.abs,
                  .multiplier = mult_imm ? (uint8_t)m.value : 0,
                  .multiplier_is_register = mult_reg,
                  .multiplier_reg = mult_reg ? m.value : 0,
                  .multiplier_16 = mult_reg && m.size == AGX_SIZE_16,
                  .multiplier_kill = mult_reg && m.discard,
                  .multiplier_cache = mult_reg && m.cache,
                  .multiplier_sign_extend =
                     (wide || half_m) && mult_reg && !m.abs,
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
                  /* Bare narrow addend: sign-extend unless ABS marks it
                   * unsigned. Identical to the old `c.neg` for every shape
                   * that already encoded (ABS gave 0, NEG gave 1); only the
                   * bare case, which used to be refused, is new.
                   *
                   * A 16-bit result widens nothing, so the bit has no meaning
                   * there and the core encoder fails such a desc closed --
                   * which refused the whole all-half MAD rather than just the
                   * bit. Keep it to widening results. */
                  .src_c_sign_extend =
                     addend_reg && !narrow &&
                     (wide ? (c.size != AGX_SIZE_64 && !c.abs)
                           : (c.size == AGX_SIZE_16 && !c.abs)),
                  /* Subtract rather than add.  This used to have no
                   * representation at all: a negated REGISTER addend passed
                   * the gate above and then composed as plain `imadd`, so
                   * `a*b - c` silently computed `a*b + c`.  A negated
                   * immediate addend was refused instead.  Both now select
                   * Apple's `imsub`. */
                  .negate_addend = !no_addend && c.neg,
                  .wait_mask = wait,
               };
               uint8_t bytes[AGX3_IADD_MAX_LENGTH];
               uint8_t len = 0;
               if (agx3_imad_encode(enc, &d, bytes, &len) &&
                   agx3_append(emission, bytes, len))
                  continue;
            }
         }

         /* Keep the uniform half of a native tied min/max in the uniform file.
          * Canonicalize just as the register/minifloat path below does. */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_FCMPSEL &&
             I->nr_dests == 1 && I->nr_srcs == 4 && !I->invert_cond &&
             !I->saturate &&
             (I->fcond == AGX_FCOND_LTN || I->fcond == AGX_FCOND_GTN) &&
             (I->src[0].type == AGX_INDEX_UNIFORM || I->src[1].type == AGX_INDEX_UNIFORM)) {
            const unsigned u = I->src[0].type == AGX_INDEX_UNIFORM ? 0 : 1;
            const agx_index uniform = I->src[u], reg = I->src[u ^ 1];
            const bool half = I->dest[0].size == AGX_SIZE_16;
            if (!agx3_src_is_register(I->dest[0]) || I->dest[0].memory ||
                agx_channels(I->dest[0]) != 1 ||
                (!half && I->dest[0].size != AGX_SIZE_32) ||
                !agx3_src_is_register(reg) ||
                !agx_is_equiv(I->src[0], I->src[2]) ||
                !agx_is_equiv(I->src[1], I->src[3]))
               AGX3_REFUSE(I);
            for (unsigned i = 0; i < 4; ++i) {
               if (I->src[i].size != I->dest[0].size ||
                   agx_channels(I->src[i]) != 1 || I->src[i].memory ||
                   I->src[i].abs || I->src[i].neg || I->src[i].kill ||
                   I->src[i].cache != I->src[i ^ 2].cache ||
                   I->src[i].discard != I->src[i ^ 2].discard)
                  AGX3_REFUSE(I);
            }
            if (uniform.cache || uniform.discard)
               AGX3_REFUSE(I);
            uint16_t fsel_wait = wait;
            if (fsel_wait & (fsel_wait - 1)) {
               if (!agx3_retire_select_waits(emission, enc, I, fsel_wait))
                  AGX3_REFUSE(I);
               fsel_wait = 0;
            }
            uint8_t bytes[16], length;
            if (!agx3_uniform_minmax_encode(enc, I->dest[0].value, reg.value,
                    uniform.value, half, I->fcond == AGX_FCOND_GTN,
                    reg.cache, reg.discard, I->dest[0].cache, fsel_wait, bytes, &length) ||
                !agx3_append(emission, bytes, length))
               AGX3_REFUSE(I);
            continue;
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
                  if (!agx3_retire_select_waits(emission, enc, I, fsel_wait))
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
            uint16_t fsel_wait = wait;
            if (fsel_wait & (fsel_wait - 1)) {
               if (!agx3_retire_select_waits(emission, enc, I, fsel_wait))
                  AGX3_REFUSE(I);
               fsel_wait = 0;
            }
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
            const bool composed = I->fcond == AGX_FCOND_GTN
               ? agx3_alu_fmax_half(enc, I->dest[0].value, src, hints,
                                    I->dest[0].cache, fsel_wait, bytes, &len)
               : agx3_alu_fmin_half(enc, I->dest[0].value, src, hints,
                                    I->dest[0].cache, fsel_wait, bytes, &len);
            if (!composed ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         /* Hyprland and GTK compare f32 against a minifloat and select either
          * one full register plus zero, or two half-width byte constants.
          * Both shapes are direct native FSEL forms. */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_FCMPSEL &&
             I->nr_dests == 1 && I->nr_srcs == 4 && !I->saturate &&
             agx3_fcond_witnessed_select(I->fcond) &&
             agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
             (I->dest[0].size == AGX_SIZE_16 ||
              I->dest[0].size == AGX_SIZE_32) &&
             agx_channels(I->dest[0]) == 1 &&
             ((agx3_mod_src_ok(I->src[0]) &&
               I->src[1].type == AGX_INDEX_IMMEDIATE) ||
              (I->src[0].type == AGX_INDEX_IMMEDIATE &&
               agx3_mod_src_ok(I->src[1]))) &&
             ((I->dest[0].size == AGX_SIZE_16 &&
               I->src[2].type == AGX_INDEX_IMMEDIATE &&
               I->src[3].type == AGX_INDEX_IMMEDIATE) ||
              (I->dest[0].size == AGX_SIZE_32 &&
               ((I->src[2].type == AGX_INDEX_IMMEDIATE) !=
                (I->src[3].type == AGX_INDEX_IMMEDIATE))))) {
            const bool dest_16 = I->dest[0].size == AGX_SIZE_16;
            const bool immediate_first =
               I->src[0].type == AGX_INDEX_IMMEDIATE;
            const agx_index compare = I->src[immediate_first ? 1 : 0];
            const agx_index threshold = I->src[immediate_first ? 0 : 1];
            const unsigned selected_index[2] = {
               I->invert_cond ? 3u : 2u, I->invert_cond ? 2u : 3u};
            bool selected_immediate[2];
            uint16_t selected[2];
            enum agx3_alu_hint hints[3] = {
               agx3_post_ra_hint(compare),
               AGX3_ALU_HINT_NONE,
               AGX3_ALU_HINT_NONE,
            };
            const bool compare_neg = compare.neg;
            bool invalid = threshold.value > UINT8_MAX || threshold.abs ||
                           threshold.neg || compare.abs || compare.kill ||
                           threshold.cache || threshold.discard;
            for (unsigned i = 0; i < 4; ++i)
               invalid |= I->src[i].memory || I->src[i].kill ||
                          (I->src[i].cache && I->src[i].discard) ||
                          agx_channels(I->src[i]) != 1;
            for (unsigned i = 2; i < 4; ++i)
               invalid |= I->src[i].abs || I->src[i].neg;
            for (unsigned i = 0; i < 2; ++i) {
               const agx_index value = I->src[selected_index[i]];
               selected_immediate[i] = value.type == AGX_INDEX_IMMEDIATE;
               invalid |= selected_immediate[i]
                             ? (value.value > UINT8_MAX || value.cache ||
                                value.discard)
                             : (!agx3_src_is_register(value) ||
                                value.size != I->dest[0].size);
               selected[i] = value.value;
               if (!selected_immediate[i])
                  hints[i + 1] = agx3_post_ra_hint(value);
            }

            unsigned condition = immediate_first
                                    ? agx3_fcond_reverse(I->fcond)
                                    : I->fcond;
            uint8_t minifloat = threshold.value;
            if (compare_neg) {
               /* Ordered IEEE comparisons preserve their NaN result when
                * both operands are negated, while their direction reverses.
                * Apple's -x < -0 and -x < 1 kernels use exactly this
                * canonical x > +0 / x > -1 form. */
               condition = agx3_fcond_reverse(condition);
               minifloat ^= 0x80;
            }

            uint16_t fsel_wait = wait;
            const bool retire_wait =
               dest_16 ? (fsel_wait > 32 ||
                          (fsel_wait && (fsel_wait & (fsel_wait - 1))))
                       : (fsel_wait != 0 && fsel_wait != 32);
            if (retire_wait)
               fsel_wait = 0;
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
            if (!invalid &&
                agx3_alu_fcmp_minifloat_select_mixed(
                   enc, I->dest[0].value, compare.value,
                   minifloat, selected, selected_immediate, hints,
                   condition, dest_16, I->dest[0].cache, fsel_wait, bytes,
                   &len)) {
               if (retire_wait &&
                   !agx3_retire_slots_noop(emission, enc, wait))
                  AGX3_REFUSE(I);
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
               if (!agx3_retire_select_waits(emission, enc, I, fsel_wait))
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
               /* LTN/GTN are minNum/maxNum conditions. Rewriting them to
                * GE/LE with swapped integer selections returns the NaN when
                * src0 is NaN. Untied condition-3/7 selects are not grounded. */
               if (ctx->key->dev.chip != AGX_CHIP_G17P &&
                   (condition == AGX_FCOND_LTN || condition == AGX_FCOND_GTN))
                  AGX3_REFUSE(I);
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
                  if (!agx3_retire_select_waits(emission, enc, I, fsel_wait))
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

         /* Hyprland's HLG transfer path compares a scalar-half value against
          * a minifloat threshold and selects two full floats.  Keep the half
          * in its assigned physical half; a leading immediate only reverses
          * the comparison, while an inverted condition exchanges results. */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_FCMPSEL &&
             I->nr_dests == 1 && I->nr_srcs == 4 && !I->saturate &&
             agx3_fcond_witnessed_select(I->fcond) &&
             agx3_vary_reg32(I->dest[0]) && agx_channels(I->dest[0]) == 1 &&
             (agx3_vary_reg32(I->src[2]) ||
              I->src[2].type == AGX_INDEX_IMMEDIATE) &&
             (agx3_vary_reg32(I->src[3]) ||
              I->src[3].type == AGX_INDEX_IMMEDIATE) &&
             !(I->src[2].type == AGX_INDEX_IMMEDIATE &&
               I->src[3].type == AGX_INDEX_IMMEDIATE) &&
             agx_channels(I->src[2]) == 1 && agx_channels(I->src[3]) == 1 &&
             ((agx3_half_reg(I->src[0]) &&
               I->src[1].type == AGX_INDEX_IMMEDIATE) ||
              (I->src[0].type == AGX_INDEX_IMMEDIATE &&
               agx3_half_reg(I->src[1])))) {
            const bool immediate_first =
               I->src[0].type == AGX_INDEX_IMMEDIATE;
            const agx_index immediate = I->src[immediate_first ? 0 : 1];
            const agx_index compare = I->src[immediate_first ? 1 : 0];
            const unsigned true_src = I->invert_cond ? 3 : 2;
            const unsigned false_src = I->invert_cond ? 2 : 3;
            bool invalid = immediate.value > UINT8_MAX;
            for (unsigned i = 0; i < 4; ++i)
               invalid |= I->src[i].abs || I->src[i].neg || I->src[i].kill ||
                          I->src[i].memory || agx_channels(I->src[i]) != 1 ||
                          (I->src[i].cache && I->src[i].discard);
            if (invalid)
               AGX3_REFUSE(I);

            const uint16_t selected[] = {
               I->src[true_src].value, I->src[false_src].value};
            const bool selected_immediate[] = {
               I->src[true_src].type == AGX_INDEX_IMMEDIATE,
               I->src[false_src].type == AGX_INDEX_IMMEDIATE};
            const enum agx3_alu_hint hints[] = {
               agx3_post_ra_hint(compare),
               selected_immediate[0]
                  ? AGX3_ALU_HINT_NONE
                  : agx3_post_ra_hint(I->src[true_src]),
               selected_immediate[1]
                  ? AGX3_ALU_HINT_NONE
                  : agx3_post_ra_hint(I->src[false_src])};
            const unsigned condition =
               immediate_first ? agx3_fcond_reverse(I->fcond) : I->fcond;
            uint16_t fsel_wait = wait;
            if (fsel_wait != 0 && fsel_wait != 32) {
               if (!agx3_retire_select_waits(emission, enc, I, fsel_wait))
                  AGX3_REFUSE(I);
               fsel_wait = 0;
            }

            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
            if (!agx3_alu_fcmp_half_minifloat_select_f32(
                   enc, I->dest[0].value, compare.value,
                   (uint8_t)immediate.value, selected, selected_immediate,
                   hints, condition, I->dest[0].cache, fsel_wait, bytes,
                   &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
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
             ((agx3_mod_src_ok(I->src[0]) &&
               I->src[1].type == AGX_INDEX_IMMEDIATE) ||
              (agx3_mod_src_ok(I->src[1]) &&
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
               invalid |= (I->src[i].abs && i != (immediate_first ? 1 : 0)) ||
                          I->src[i].neg || I->src[i].kill || I->src[i].memory ||
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
            if ((compare.abs && fsel_wait != 0 && fsel_wait != 32) ||
                (fsel_wait & (fsel_wait - 1))) {
               if (!agx3_retire_select_waits(emission, enc, I, fsel_wait))
                  AGX3_REFUSE(I);
               fsel_wait = 0;
            }

            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
            if (!agx3_alu_fcmp_select_f32_imm(
                   enc, I->dest[0].value, compare.value,
                   (uint8_t)immediate.value, compare.abs,
                   I->src[true_src].value,
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
               if (!agx3_retire_select_waits(emission, enc, I, fsel_wait))
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

         /* Scalar-half comparisons selecting full values are a native FSEL
          * form of their own.  Vertex-format normalization produces this for
          * `fabs(half) < half ? float : float`; widening the comparison would
          * change mediump rounding and is unnecessary. */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_FCMPSEL &&
             I->nr_dests == 1 && I->nr_srcs == 4 && !I->saturate &&
             agx3_fcond_witnessed_select(I->fcond) &&
             agx3_vary_reg32(I->dest[0]) && agx_channels(I->dest[0]) == 1 &&
             agx3_half_reg(I->src[0]) && agx3_half_reg(I->src[1]) &&
             agx3_vary_reg32(I->src[2]) && agx3_vary_reg32(I->src[3]) &&
             agx_channels(I->src[2]) == 1 && agx_channels(I->src[3]) == 1) {
            const unsigned selected[] = {
               I->invert_cond ? 3u : 2u, I->invert_cond ? 2u : 3u};
            const uint16_t src[] = {
               I->src[0].value, I->src[1].value,
               I->src[selected[0]].value, I->src[selected[1]].value};
            const enum agx3_alu_hint hints[] = {
               agx3_post_ra_hint(I->src[0]),
               agx3_post_ra_hint(I->src[1]),
               agx3_post_ra_hint(I->src[selected[0]]),
               agx3_post_ra_hint(I->src[selected[1]])};
            const bool src_abs[] = {I->src[0].abs, I->src[1].abs};
            bool invalid = I->src[0].neg || I->src[1].neg;
            for (unsigned i = 0; i < 4; ++i)
               invalid |= I->src[i].memory || I->src[i].kill ||
                          (I->src[i].cache && I->src[i].discard) ||
                          (i >= 2 && (I->src[i].abs || I->src[i].neg));

            uint16_t fsel_wait = wait;
            if (fsel_wait != 0 && fsel_wait != 32) {
               if (!agx3_retire_select_waits(emission, enc, I, fsel_wait))
                  AGX3_REFUSE(I);
               fsel_wait = 0;
            }
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
            if (invalid ||
                !agx3_alu_fcmp_select_f16_f16_f32(
                   enc, I->dest[0].value, src, hints, src_abs, I->fcond,
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
               if (!agx3_retire_select_waits(emission, enc, I, fsel_wait))
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
            const bool compare_neg[]={I->src[zero_first?1:0].neg,
                                      I->src[zero_first?0:1].neg};
            const bool immediate[]={true,true};
            const enum agx3_alu_hint hints[]={agx3_post_ra_hint(compare),AGX3_ALU_HINT_NONE,
               AGX3_ALU_HINT_NONE,AGX3_ALU_HINT_NONE};
            bool invalid=I->saturate || agx_channels(compare)!=1;
            for(unsigned i=0;i<4;i++)invalid|=I->src[i].abs || (i>=2 && I->src[i].neg) ||
               I->src[i].kill || (I->src[i].cache && I->src[i].discard);
            if(invalid)AGX3_REFUSE(I);
            uint16_t fsel_wait=wait;
            if(fsel_wait!=0 && fsel_wait!=32) {
               if(!agx3_retire_select_waits(emission,enc,I,fsel_wait))AGX3_REFUSE(I);
               fsel_wait=0;
            }
            uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
            if(!agx3_alu_fcmp_select_reg_imm(enc,I->dest[0].value,src,true,
                 compare_abs,compare_neg,immediate,hints,
                 condition,I->dest[0].size==AGX_SIZE_16,I->dest[0].cache,fsel_wait,bytes,&len) ||
               !agx3_append(emission,bytes,len))AGX3_REFUSE(I);
            continue;
         }

         /* Native FSEL selects retained 16/32-bit register values or a register
          * and an immediate directly. Preserve each operand's lifetime hints
          * without materializing a separate comparison result.
          *
          * A negated comparator rides through to the encoder rather than
          * failing the shader closed: f65 names the second comparator, and
          * every other placement is an operand exchange or a relation flip
          * that agx3_alu_fcmp_select_reg_imm() performs. */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_FCMPSEL &&
             I->nr_dests==1 && I->nr_srcs==4 &&
             agx3_src_is_register(I->dest[0]) && !I->dest[0].memory && agx_channels(I->dest[0])==1 &&
             (I->dest[0].size==AGX_SIZE_16 || I->dest[0].size==AGX_SIZE_32) &&
             ((agx3_mod_src_ok(I->src[0]) && agx3_mod_src_ok(I->src[1])) ||
              (agx3_vary_reg32(I->src[0]) && agx3_src_is_zero_imm(I->src[1])) ||
              (agx3_src_is_zero_imm(I->src[0]) && agx3_vary_reg32(I->src[1]))) &&
             (agx3_src_is_register(I->src[2]) ||
              agx3_src_is_register(I->src[3]))) {
            /* G16G's condition-3/7 full-register form is grounded only when
             * the alternatives are tied min/max operands, handled above.
             * Rewriting an untied all-register form to ordered GE/LE returns
             * the NaN operand. G17P retains its separately qualified legacy
             * form, and immediate-alternative M4 witnesses remain below. */
            if (ctx->key->dev.chip != AGX_CHIP_G17P &&
                I->dest[0].size == AGX_SIZE_32 &&
                (I->fcond == AGX_FCOND_LTN || I->fcond == AGX_FCOND_GTN) &&
                agx3_src_is_register(I->src[0]) &&
                agx3_src_is_register(I->src[1]) &&
                agx3_src_is_register(I->src[2]) &&
                agx3_src_is_register(I->src[3]))
               AGX3_REFUSE(I);
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
            bool compare_abs[2]; bool compare_neg[2];
            bool invalid=I->saturate;
            for(unsigned i=0;i<4;i++) {
               const agx_index x=operands[i];
               const bool imm=x.type==AGX_INDEX_IMMEDIATE;
               /* ADMITTED: abs and neg on EITHER comparator (i < 2). f65 is
                * the one negate field and it names the second comparator, but
                * the first is reachable too -- Apple spells `-a < b` by
                * exchanging the operands and flipping the relation, and
                * agx3_alu_fcmp_select_reg_imm() performs that rewrite rather
                * than handing it back. This gate used to read `x.neg && i!=1`,
                * admitting slot 1 only; slot 0 was refused on the strength of
                * having seen the swap and declined to do it.
                *
                * REFUSED: abs or neg on a SELECTED value (i >= 2). Those slots
                * carry a value, not an arithmetic operand -- there is no
                * modifier field on them at all, and a select may carry an
                * integer. Also refused: a killed or spilled operand, a vector
                * operand, conflicting lifetime hints, an out-of-range
                * immediate, and a selected register whose width does not match
                * the destination. */
               invalid|=(i>=2 && (x.abs || x.neg)) || x.kill || x.memory ||
                  agx_channels(x)!=1 ||
                  (x.cache && x.discard) ||
                  (imm ? x.value>255 : (!agx3_src_is_register(x) ||
                    x.size!=(i<2?AGX_SIZE_32:I->dest[0].size)));
               src[i]=x.value; hints[i]=imm?AGX3_ALU_HINT_NONE:agx3_post_ra_hint(x);
               if(i<2){compare_abs[i]=x.abs;compare_neg[i]=x.neg;}
               if(i>=2)immediate[i-2]=imm;
            }
            if(invalid)AGX3_REFUSE(I);
            uint16_t fsel_wait=wait;
            if(fsel_wait!=0 && fsel_wait!=32) {
               if(!agx3_retire_select_waits(emission,enc,I,fsel_wait))AGX3_REFUSE(I);
               fsel_wait=0;
            }
            uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
            if(!agx3_alu_fcmp_select_reg_imm(enc,I->dest[0].value,src,compare_zero,
                 compare_abs,compare_neg,immediate,hints,condition,I->dest[0].size==AGX_SIZE_16,
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
             I->src[2].type == AGX_INDEX_IMMEDIATE && I->src[2].value <= 255 &&
             I->src[3].type == AGX_INDEX_IMMEDIATE && I->src[3].value <= 255 &&
             /* An ABS source takes the fourteen-byte ep28..35 endpoint, whose
              * encoder here (agx3_alu_fcmp_select_imm_width_abs) carries only
              * the two abs bits and has no argument for a negate, so this
              * specialization cannot spell a comparison that wants one of
              * each. The endpoint itself is not the limit: it holds the same
              * comparator-1 negate as the ten-byte form, and the general
              * compare-select below deposits f47, f64 and f65 independently
              * and encodes the pair byte-for-byte as Apple does (see the
              * witnessed minimal pair in the Hyprland float-select test). So
              * decline the shape here and let it fall through, the way every
              * other shape this block cannot spell already does. Refusing
              * dropped the whole draw instead: `ball(feq(cf, 0))` from the
              * cull-distance emulation folds to `|a| == -(b)` and reaches
              * exactly this, which made every single-cull-distance fragment
              * program unlinkable. */
             !((I->src[0].abs || I->src[1].abs) &&
               (I->src[0].neg || I->src[1].neg))) {
            /* A negated comparison source. f65 names the SECOND comparator
             * and only that one; `-a > b` is Apple's own `b < -a`, so the
             * encoder swaps the operands and reverses the relation instead of
             * refusing. Every relation, both result widths and any pair of
             * selected immediates are witnessed -- see the byte ladder above
             * agx3_alu_fcmp_select_imm_width_internal(). `dot(view - world,
             * normal) > 0.0` is the shape that reaches this. */
            /* An absolute comparison source moves the instruction to the
             * longer ep28..35 endpoint. It is a different form, not a bit on
             * the short one, so it gets its own encoder. Mixed absolute and
             * negate modifiers also use that endpoint, but are handled by the
             * general descriptor below. */
            const bool any_abs = I->src[0].abs || I->src[1].abs;
            const bool any_neg = I->src[0].neg || I->src[1].neg;
            const bool negated = !any_abs && any_neg;
            const uint16_t src[] = {I->src[0].value, I->src[1].value};
            const enum agx3_alu_hint hints[] = {
               agx3_post_ra_hint(I->src[0]), agx3_post_ra_hint(I->src[1])};
            const bool src_16[] = {I->src[0].size == AGX_SIZE_16,
                                   I->src[1].size == AGX_SIZE_16};
            const bool src_neg[] = {I->src[0].neg, I->src[1].neg};
            if (!(any_abs && any_neg)) {
               uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
               unsigned condition = I->fcond;
               bool swap = I->invert_cond;
               /* The tied min/max handler above owns condition 3/7. There is
                * no grounded untied encoding with those NaN semantics. */
               if (ctx->key->dev.chip != AGX_CHIP_G17P &&
                   (condition == AGX_FCOND_LTN || condition == AGX_FCOND_GTN))
                  AGX3_REFUSE(I);
               uint16_t fsel_wait = wait;
               if (fsel_wait != 0 && fsel_wait != 32) {
                  if (!agx3_retire_select_waits(emission, enc, I, fsel_wait))
                     AGX3_REFUSE(I);
                  fsel_wait = 0;
               }
               const bool src_abs[] = {I->src[0].abs, I->src[1].abs};
               if (I->saturate || I->src[0].kill || I->src[1].kill ||
                   (I->src[0].cache && I->src[0].discard) ||
                   (I->src[1].cache && I->src[1].discard) ||
                   I->src[2].abs || I->src[2].neg ||
                   I->src[3].abs || I->src[3].neg ||
                   !(any_abs
                        ? agx3_alu_fcmp_select_imm_width_abs(
                             enc, I->dest[0].value, src, src_16, hints,
                             condition, I->src[swap ? 3 : 2].value,
                             I->src[swap ? 2 : 3].value,
                             I->dest[0].size == AGX_SIZE_16,
                             I->dest[0].cache, fsel_wait, src_abs, bytes, &len)
                     : negated
                        ? agx3_alu_fcmp_select_imm_width_neg(
                             enc, I->dest[0].value, src, src_16, hints, src_neg,
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
         }

         /*
          * The general compare-select: a REGISTER in a select slot, or an
          * IMMEDIATE in a compare slot. The block above requires both selects
          * to be immediates and both compare sources to be registers, so it
          * skips these shapes entirely rather than refusing them. The one
          * overlap is a register/register comparison mixing absolute and
          * negate modifiers, which needs this endpoint's independent modifier
          * fields.
          *
          * Both are ordinary GLSL. dEQP's isOk() written to a mediump output
          * gives the first (a 16-bit destination selecting between a register
          * and a constant); isinf() gives the second (a comparison against an
          * immediate zero selecting between two registers).
          */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_FCMPSEL &&
             I->nr_dests == 1 && I->nr_srcs == 4 && !I->saturate &&
             agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
             agx_channels(I->dest[0]) == 1 &&
             (I->dest[0].size == AGX_SIZE_16 ||
              I->dest[0].size == AGX_SIZE_32)) {
            const agx_index c0 = I->src[0], c1 = I->src[1];
            const bool c0_imm = c0.type == AGX_INDEX_IMMEDIATE;
            const bool c1_imm = c1.type == AGX_INDEX_IMMEDIATE;

            /* At most one compare source may be an immediate, and it is a
             * minifloat byte -- not only zero, which was the limit of what
             * had been measured rather than a property of the field. Apple
             * spells `k < b` by reversing the relation so the immediate lands
             * in slot 1, the only slot whose kind field is witnessed taking
             * the immediate value, and a modifier on that slot is folded into
             * the constant by the encoder. */
            if (!(c0_imm && c1_imm) &&
                (!c0_imm || (c0.value <= 0xff && agx3_plain_immediate(c0))) &&
                (!c1_imm || (c1.value <= 0xff && agx3_plain_immediate(c1))) &&
                (c0_imm || agx3_src_is_register(c0)) &&
                (c1_imm || agx3_src_is_register(c1))) {
               const agx_index sel_t = I->src[2], sel_f = I->src[3];
               const bool t_imm = sel_t.type == AGX_INDEX_IMMEDIATE;
               const bool f_imm = sel_f.type == AGX_INDEX_IMMEDIATE;

               unsigned condition = I->fcond;
               bool swap = I->invert_cond;
               if (condition == AGX_FCOND_LTN) {
                  condition = AGX_FCOND_GE;
                  swap = !swap;
               } else if (condition == AGX_FCOND_GTN) {
                  condition = AGX_FCOND_LE;
                  swap = !swap;
               }
               /* Put the immediate in compare slot 1. */
               const bool reverse = c0_imm;
               const agx_index cmp0 = reverse ? c1 : c0;
               const agx_index cmp1 = reverse ? c0 : c1;
               if (reverse)
                  condition = agx3_fcond_reverse(condition);

               const agx_index tv = swap ? sel_f : sel_t;
               const agx_index fv = swap ? sel_t : sel_f;
               const bool tv_imm = swap ? f_imm : t_imm;
               const bool fv_imm = swap ? t_imm : f_imm;

               const struct agx3_fsel_desc d = {
                  .dest = I->dest[0].value,
                  .dest_16 = I->dest[0].size == AGX_SIZE_16,
                  .dest_cache = I->dest[0].cache,
                  .cmp0 = cmp0.value,
                  .cmp0_hint = agx3_post_ra_hint(cmp0),
                  .cmp0_abs = cmp0.abs,
                  .cmp0_neg = cmp0.neg,
                  .cmp0_16 = cmp0.size == AGX_SIZE_16,
                  .cmp1_is_immediate = reverse ? c0_imm : c1_imm,
                  .cmp1 = cmp1.value,
                  .cmp1_hint = agx3_post_ra_hint(cmp1),
                  .cmp1_abs = cmp1.abs,
                  .cmp1_neg = cmp1.neg,
                  .cmp1_16 = cmp1.size == AGX_SIZE_16,
                  .sel0_kind = tv_imm ? AGX3_FSEL_SEL_IMMEDIATE
                                      : AGX3_FSEL_SEL_REGISTER,
                  .sel1_kind = fv_imm ? AGX3_FSEL_SEL_IMMEDIATE
                                      : AGX3_FSEL_SEL_REGISTER,
                  .sel0 = tv.value,
                  .sel1 = fv.value,
                  .sel0_hint = tv_imm ? AGX3_ALU_HINT_NONE : agx3_post_ra_hint(tv),
                  .sel1_hint = fv_imm ? AGX3_ALU_HINT_NONE : agx3_post_ra_hint(fv),
                  .sel_16 = I->dest[0].size == AGX_SIZE_16,
                  .condition = condition,
                  .wait_mask = 0,
               };

               const bool operands_ok =
                  !tv.neg && !fv.neg && !tv.abs &&
                  !fv.abs && !cmp0.kill && !cmp1.kill &&
                  !tv.kill && !fv.kill && !tv.memory && !fv.memory &&
                  (!tv_imm || agx3_plain_immediate(tv)) &&
                  (!fv_imm || agx3_plain_immediate(fv)) &&
                  !(cmp0.cache && cmp0.discard) &&
                  !(cmp1.cache && cmp1.discard) &&
                  !(tv.cache && tv.discard) && !(fv.cache && fv.discard) &&
                  (tv_imm || agx3_src_is_register(tv)) &&
                  (fv_imm || agx3_src_is_register(fv)) &&
                  agx_channels(cmp0) == 1 && agx_channels(cmp1) == 1;

               uint8_t bytes[AGX3_ALU_MAX_LENGTH];
               uint8_t len = 0;
               if (operands_ok) {
                  /* No witness of this endpoint carries a scoreboard wait, so
                   * retire outstanding slots first rather than dropping the
                   * dependency. */
                  if (wait != 0 &&
                      !agx3_retire_select_waits(emission, enc, I, wait))
                     AGX3_REFUSE(I);
                  if (agx3_alu_fcmp_select(enc, &d, bytes, &len) &&
                      agx3_append(emission, bytes, len))
                     continue;
               }
               AGX3_REFUSE(I);
            }
         }

         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_IADD &&
             I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
             I->shift >= 1 && I->shift <= 4 && I->dest[0].size == AGX_SIZE_16 &&
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
               .src1_kill = I->src[1].discard,
               .shift = I->shift,
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

         /* Narrow iadd_sat and uadd_sat. ABS marks both halves unsigned;
          * Apple's saturating iadd instead marks signed halves, so the
          * signedness is a property of the whole instruction. */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_IADD &&
             I->nr_dests == 1 && I->nr_srcs == 2 && I->saturate &&
             I->shift == 0 && I->dest[0].size == AGX_SIZE_16 &&
             agx3_src_is_register(I->src[0]) &&
             agx3_src_is_register(I->src[1]) &&
             I->src[0].size == AGX_SIZE_16 && !I->src[0].neg &&
             I->src[1].size == AGX_SIZE_16 && !I->src[1].neg) {
            const agx_index dest = I->dest[0];
            const agx_index a = I->src[0], b = I->src[1];
            if (!agx3_src_is_register(dest) || dest.memory ||
                agx_channels(dest) != 1 || a.memory || b.memory ||
                a.abs != b.abs || a.kill || b.kill ||
                agx_channels(a) != 1 || agx_channels(b) != 1 ||
                (a.cache && a.discard) || (b.cache && b.discard))
               AGX3_REFUSE(I);

            const uint16_t src[2] = {a.value, b.value};
            const enum agx3_alu_hint hints[2] = {
               agx3_post_ra_hint(a), agx3_post_ra_hint(b)};
            uint16_t w = wait;
            if (!agx3_retire_extra_waits(emission, enc, sl, res, instr_index,
                                         I, &w))
               AGX3_REFUSE(I);
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
            if (!agx3_alu_add_sat16(enc, dest.value, src, hints, !a.abs,
                                    dest.cache, w, bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         /* NIR's narrow isub is iadd(a, -b), and ineg is iadd(#0, -a). Apple
          * emits both as the scalar-half isub, the negation with an
          * immediate-zero minuend, so neither needs a widened temporary. */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_IADD &&
             I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
             I->shift == 0 && I->dest[0].size == AGX_SIZE_16 &&
             agx3_src_is_register(I->src[1]) &&
             I->src[1].size == AGX_SIZE_16 && I->src[1].neg &&
             (agx3_src_is_zero_imm(I->src[0]) ||
              (agx3_src_is_register(I->src[0]) &&
               I->src[0].size == AGX_SIZE_16))) {
            const agx_index dest = I->dest[0];
            const agx_index minuend = I->src[0], subtrahend = I->src[1];
            const bool negate = minuend.type == AGX_INDEX_IMMEDIATE;
            if (!agx3_src_is_register(dest) || dest.memory ||
                agx_channels(dest) != 1 || minuend.memory || minuend.abs ||
                minuend.neg || minuend.kill ||
                (minuend.cache && minuend.discard) ||
                (!negate && agx_channels(minuend) != 1) ||
                subtrahend.memory || subtrahend.abs || subtrahend.kill ||
                agx_channels(subtrahend) != 1 ||
                (subtrahend.cache && subtrahend.discard))
               AGX3_REFUSE(I);

            uint16_t w = wait;
            if (!agx3_retire_extra_waits(emission, enc, sl, res, instr_index,
                                         I, &w))
               AGX3_REFUSE(I);
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
            bool encoded;
            if (negate) {
               encoded = agx3_alu_neg16(enc, dest.value, subtrahend.value,
                                        agx3_post_ra_hint(subtrahend),
                                        dest.cache, w, bytes, &len);
            } else {
               const uint16_t src[2] = {minuend.value, subtrahend.value};
               const enum agx3_alu_hint hints[2] = {
                  agx3_post_ra_hint(minuend), agx3_post_ra_hint(subtrahend)};
               encoded = agx3_alu_sub16(enc, dest.value, src, hints,
                                        dest.cache, w, bytes, &len);
            }
            if (!encoded || !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         /* Register arithmetic shares the original canonical form, but its
          * native source hints come from post-RA lifetime.
          *
          * The SUBTRACT additionally carries a shift on slot B (f54). NIR
          * strength-reduces a multiply by a small negative constant into
          * `a - (b << N)` -- `x * -7` becomes `x - (x << 3)` -- so a shifted
          * subtract is an ordinary shape, and refusing it failed every such
          * multiply closed. The ADD keeps its `shift == 0` restriction: its
          * shifted form is modelled separately by agx3_iadd_encode, and
          * re-routing a shape that already encodes is what cost two hardware
          * probes before. */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_IADD &&
             (I->shift == 0 || I->src[1].neg) &&
             I->nr_dests == 1 && I->nr_srcs == 2 &&
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
                !agx3_alu_integer_reg_shift(enc, op, I->dest[0].value, src,
                                            hints, (uint8_t)I->shift, wait,
                                            bytes, &len) ||
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

            /* iadd_sat, uadd_sat, isub_sat and usub_sat over 32-bit
             * registers: the register iadd/isub point with f53. ABS on both
             * operands selects unsigned saturation; otherwise both sign
             * fields are set, exactly as applegpu_g17p's addsat/subsat of
             * int and uint. The wait-carrying composer used to drop the
             * saturate bit here. */
            if (enc == AGX_ISA_ENCODING_AGX3B && I->saturate) {
               agx_index a = I->src[0], b = I->src[1];
               if (a.neg && !b.neg) {
                  const agx_index t = a;
                  a = b;
                  b = t;
               }
               if (I->shift != 0 || !agx3_src_is_register(a) ||
                   !agx3_src_is_register(b) || a.memory || b.memory ||
                   a.size != AGX_SIZE_32 || b.size != AGX_SIZE_32 ||
                   agx_channels(a) != 1 || agx_channels(b) != 1 ||
                   a.neg || a.kill || b.kill || a.abs != b.abs ||
                   (a.cache && a.discard) || (b.cache && b.discard))
                  AGX3_REFUSE(I);

               uint16_t w = wait;
               if (!agx3_retire_extra_waits(emission, enc, sl, res,
                                            instr_index, I, &w))
                  AGX3_REFUSE(I);
               const struct agx3_iadd_desc d = {
                  .dest = I->dest[0].value,
                  .dest_cache = I->dest[0].cache,
                  .src_a_is_register = true,
                  .src_a_reg = a.value,
                  .src_a_kill = a.discard,
                  .src_a_cache = a.cache,
                  .src_a_sign_extend = !a.abs,
                  .src = b.value,
                  .src_kill = b.discard,
                  .src_cache = b.cache,
                  .sign_extend = !b.abs,
                  .subtract = b.neg,
                  .saturate = true,
                  .wait_mask = w,
               };
               uint8_t bytes[AGX3_IADD_MAX_LENGTH];
               uint8_t len = 0;
               if (!agx3_iadd_encode(enc, &d, bytes, &len) ||
                   !agx3_append(emission, bytes, len))
                  AGX3_REFUSE(I);
               continue;
            }

            /* reg - #imm. A narrow register is sign-extended when bare and
             * zero-extended under ABS; NEG on it would negate the register
             * too, which this form does not express. */
            if (!imm_first && I->shift == 0 && !I->saturate &&
                constant.type == AGX_INDEX_IMMEDIATE && constant.neg &&
                !constant.abs && !constant.memory && constant.value <= 0xff &&
                agx3_src_is_register(reg) && !reg.memory && !reg.kill &&
                agx_channels(reg) == 1 && !(reg.cache && reg.discard) &&
                !reg.neg &&
                ((reg.size == AGX_SIZE_32 && !reg.abs) ||
                 reg.size == AGX_SIZE_16)) {
               const struct agx3_isub_imm_desc d = {
                  .dest=I->dest[0].value, .src=reg.value,
                  .immediate=constant.value, .src_16=reg.size==AGX_SIZE_16,
                  .sign_extend=reg.size==AGX_SIZE_16 && !reg.abs,
                  .src_cache=reg.cache,
                  .src_kill=reg.discard, .wait_mask=wait,
               };
               uint8_t bytes[AGX3_IADD_MAX_LENGTH],len;
               if (agx3_isub_immediate_encode(enc,&d,bytes,&len) &&
                   agx3_append(emission,bytes,len))
                  continue;
            }

            /* SIGNEXT lowers to `iadd dst32, src16, #0`; Hyprland's signed
             * gradient counter is the same operation with `src16 << 4`.
             * A bare narrow integer is signed on AGX (ABS denotes unsigned),
             * and the G17P iadd corpus witnesses the sign-extended shift-0
             * source width as well as the exact shift-4 form.
             *
             * The addend's value is not part of that rule. Apple emits the
             * same sign-extending form for a non-zero immediate:
             *
             *   o[i] = int(short a[i]) + 7
             *      9f 01 54 00 02 0e 00 08 15 0c       f15 = 7, f43 = 1
             *   o[i] = uint(ushort a[i]) + 7           f43 clear (ABS form)
             *
             * so gate on the source being a bare half, not on the immediate
             * being zero. Every other bare half/shift combination stays
             * closed. */
            const bool signed_half_widen =
               enc == AGX_ISA_ENCODING_AGX3B && !I->saturate &&
               constant.type == AGX_INDEX_IMMEDIATE &&
               reg.size == AGX_SIZE_16 && !reg.abs && !reg.neg &&
               (I->shift == 0 || I->shift == 4);
            /* #imm - reg is AGX IR's iadd(#imm, -reg). NEG is a negation, not
             * a sign extension (on AGX2 too): it selects Apple's isub with
             * the immediate in slot A, `16 - int(short b)` being
             * 1f 01 56 00 02 20 00 08 15 0c. Before, a negated register was
             * added, and a negated half was added sign-extended. Only the
             * unshifted subtraction is witnessed. */
            const bool imm_minus_reg =
               enc == AGX_ISA_ENCODING_AGX3B && !I->saturate &&
               reg.neg && I->shift == 0 &&
               (reg.size == AGX_SIZE_16 ||
                (reg.size == AGX_SIZE_32 && !reg.abs));
            if (constant.type == AGX_INDEX_IMMEDIATE && !constant.abs &&
                !constant.neg && constant.value <= 0xff && !I->saturate &&
                agx3_src_is_register(reg) && !reg.memory &&
                agx_channels(reg) == 1 && (!reg.neg || imm_minus_reg) &&
                ((reg.size == AGX_SIZE_32 && !reg.abs) ||
                 (reg.size == AGX_SIZE_16 &&
                  (reg.abs || reg.neg || signed_half_widen)))) {
               const struct agx3_iadd_desc d = {
                  .dest = I->dest[0].value,
                  .dest_cache = signed_half_widen && I->dest[0].cache,
                  .immediate = (uint8_t)constant.value,
                  .src = reg.value,
                  .src_kill = reg.discard,
                  .src_16 = reg.size == AGX_SIZE_16,
                  .sign_extend = reg.size == AGX_SIZE_16 && !reg.abs,
                  .shift = I->shift,
                  .subtract = reg.neg,
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
            const agx_index original_a = I->src[0];
            const agx_index original_b = I->src[1];
            /* Integer add is commutative. Canonicalize Firefox's reversed
             * signed-half accumulator onto the already-grounded slot-A32 +
             * slot-B16 form instead of inventing a second hardware shape. */
            const bool commute_signed_half =
               enc == AGX_ISA_ENCODING_AGX3B &&
               original_a.size == AGX_SIZE_16 &&
               !original_a.abs && !original_a.neg &&
               original_b.size == AGX_SIZE_32 &&
               !original_b.abs && !original_b.neg;
            const agx_index a = commute_signed_half ? original_b : original_a;
            const agx_index bsrc = commute_signed_half ? original_a : original_b;

            /*
             * Only take this path for a SHIFTED add. An unshifted register
             * add already had a working encoding through the select core, and
             * re-routing it here changed its bytes -- which cost two compute
             * probes on hardware. New encodings belong on shapes that were
             * previously refused, not on shapes that already worked.
             */
            /* A 16-bit struct index scales into an address the same way:
             * the half addend is sign-extended when bare and zero-extended
             * under ABS, through slot B's own width and sign fields. */
            const bool shifted_half =
               enc == AGX_ISA_ENCODING_AGX3B && bsrc.size == AGX_SIZE_16 &&
               !bsrc.neg;
            if (I->shift != 0 &&
                agx3_src_is_register(a) && !a.memory && !a.abs && !a.neg &&
                a.size == AGX_SIZE_32 && agx_channels(a) == 1 &&
                agx3_src_is_register(bsrc) && !bsrc.memory &&
                (shifted_half ||
                 (!bsrc.abs && !bsrc.neg && bsrc.size == AGX_SIZE_32)) &&
                agx_channels(bsrc) == 1) {
               uint16_t w = wait;
               if (shifted_half &&
                   !agx3_retire_extra_waits(emission, enc, sl, res,
                                            instr_index, I, &w))
                  AGX3_REFUSE(I);
               const struct agx3_iadd_desc d = {
                  .dest = I->dest[0].value,
                  .src_a_is_register = true,
                  .src_a_reg = a.value,
                  .src_a_kill = a.discard,
                  .src = bsrc.value,
                  .src_kill = bsrc.discard,
                  .src_16 = shifted_half,
                  .sign_extend = shifted_half && !bsrc.abs,
                  .shift = I->shift,
                  .wait_mask = w,
               };
               uint8_t bytes[AGX3_IADD_MAX_LENGTH];
               uint8_t len = 0;
               if (agx3_iadd_encode(enc, &d, bytes, &len) &&
                   agx3_append(emission, bytes, len))
                  continue;
            }

            /*
             * Register +/- register where at least one operand is a 16-bit
             * half. As on AGX2, a narrow operand is zero-extended under ABS
             * and sign-extended otherwise, and NEG negates it: iadd(a, -b)
             * is Apple's isub, the same fields under their own opcode
             * (`int a - int(short b)` is 1f 01 56 00 02 00 10 28 17 0d).
             * This used to read NEG as the sign extension and drop the
             * negation, so `16 - count` over a narrow count added.
             * `gl_LocalInvocationID` arrives as three 16-bit special
             * registers, so summing it is exactly this shape.
             *
             * Gated on a narrow operand so the all-32-bit unshifted add keeps
             * the select-core encoding it already had.
             */
            agx_index na = original_a, nb = original_b;
            /* Addition commutes: keep a negated operand in slot B, and put a
             * 32-bit operand in slot A of a mixed-width add, the order of
             * the hardware-proven Firefox accumulator. */
            if ((na.neg && !nb.neg) ||
                (!na.neg && !nb.neg && !na.abs && !nb.abs &&
                 na.size == AGX_SIZE_16 && nb.size == AGX_SIZE_32)) {
               const agx_index t = na;
               na = nb;
               nb = t;
            }
            const bool na_ok =
               agx3_src_is_register(na) && !na.memory &&
               agx_channels(na) == 1 && !na.neg &&
               !(na.cache && na.discard) &&
               (na.size == AGX_SIZE_16 ||
                (na.size == AGX_SIZE_32 && !na.abs));
            const bool nb_ok =
               agx3_src_is_register(nb) && !nb.memory &&
               agx_channels(nb) == 1 &&
               !(nb.cache && nb.discard) &&
               (nb.size == AGX_SIZE_16 ||
                (nb.size == AGX_SIZE_32 && !nb.abs));

            if (I->shift == 0 && !I->saturate && na_ok && nb_ok &&
                (na.size == AGX_SIZE_16 || nb.size == AGX_SIZE_16)) {
               const bool b3 = enc == AGX_ISA_ENCODING_AGX3B;
               uint16_t w = wait;
               if (b3 && !agx3_retire_extra_waits(emission, enc, sl, res,
                                                  instr_index, I, &w))
                  AGX3_REFUSE(I);
               const struct agx3_iadd_desc d = {
                  .dest = I->dest[0].value,
                  .dest_cache = I->dest[0].cache,
                  .src_a_is_register = true,
                  .src_a_reg = na.value,
                  .src_a_kill = na.discard,
                  /* Apple's operand cache hints, on the subtraction this
                   * path newly encodes. The additions it already emitted
                   * keep their hardware-proven bytes without them. */
                  .src_a_cache = b3 && nb.neg && na.cache,
                  .src_a_16 = na.size == AGX_SIZE_16,
                  .src_a_sign_extend = na.size == AGX_SIZE_16 && !na.abs,
                  .src = nb.value,
                  .src_kill = nb.discard,
                  .src_cache = b3 && nb.neg && nb.cache,
                  .src_16 = nb.size == AGX_SIZE_16,
                  .sign_extend = nb.size == AGX_SIZE_16 && !nb.abs,
                  .subtract = nb.neg,
                  .wait_mask = w,
               };
               uint8_t bytes[AGX3_IADD_MAX_LENGTH];
               uint8_t len = 0;
               if (!agx3_iadd_encode(enc, &d, bytes, &len) ||
                   !agx3_append(emission, bytes, len))
                  AGX3_REFUSE(I);
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
          * The half-copy corpus does not map every combination of scoreboard
          * waits. Preserve its directly encodable waits. For any other mask,
          * retire the producer through the grounded band no-op first, as for
          * uniform copies and derivatives, then emit the exact zero-wait
          * copy-and-clear pair.
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
            /* The half COPY names a seven-bit destination. Above it, add the
             * zero-extended half to an immediate 0: agx3_iadd_encode()'s
             * narrow slot-B source form, as Apple emits for
             * `uint(ushort a[i]) + 7`, names an eight-bit destination. */
            if (I->dest[0].value >= AGX3_COPY_REG16_DEST_HALVES) {
               const struct agx3_iadd_desc d = {
                  .dest = I->dest[0].value,
                  .dest_cache = I->dest[0].cache,
                  .immediate = 0,
                  .src = I->src[0].value,
                  .src_kill = I->src[0].discard,
                  .src_16 = true,
                  .wait_mask = wait,
               };
               uint8_t bytes[AGX3_IADD_MAX_LENGTH];
               uint8_t len = 0;
               if (!agx3_iadd_encode(enc, &d, bytes, &len) ||
                   !agx3_append(emission, bytes, len))
                  AGX3_REFUSE(I);
               continue;
            }
            uint8_t lo[AGX3_COPY_MAX_LENGTH], hi[AGX3_STAGE_MAX_LENGTH];
            uint8_t lo_len = 0, hi_len = 0;
            bool copied = agx3_copy_encode_reg16(
               enc, I->dest[0].value, I->src[0].value, I->dest[0].cache,
               I->src[0].cache, I->src[0].discard, wait, lo, &lo_len);
            bool retired = false;
            if (!copied && wait) {
               retired = agx3_retire_slots_noop(emission, enc, wait);
               copied = retired && agx3_copy_encode_reg16(
                                     enc, I->dest[0].value, I->src[0].value,
                                     I->dest[0].cache, I->src[0].cache,
                                     I->src[0].discard, 0, lo, &lo_len);
            }
            if (!copied ||
                !agx3_clr(enc, I->dest[0].value + 1, true, false, hi,
                          &hi_len) ||
                !agx3_append(emission, lo, lo_len) ||
                !agx3_append(emission, hi, hi_len))
               AGX3_REFUSE(I);
            if (retired)
               outstanding &= (uint16_t)~wait;
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
            if (!agx3_copy_encode_reg16(
                   enc, I->dest[0].value, I->src[0].value,
                   I->dest[0].cache, I->src[0].cache,
                   I->src[0].discard, wait, bytes, &len) ||
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
            uint8_t bytes[AGX3_ALU_MAX_LENGTH];
            uint8_t len = 0;
            bool encoded = agx3_copy_encode_reg16(
               enc, I->dest[0].value, I->src[0].value, I->dest[0].cache,
               I->src[0].cache, I->src[0].discard, wait, bytes, &len);
            if (!encoded)
               encoded = agx3_alu_bit_imm(
                  enc, AGX3_ALU_BOR, true, I->dest[0].value,
                  I->src[0].value, 0, agx3_post_ra_hint(I->src[0]), wait,
                  bytes, &len);
            if (!encoded || !agx3_append(emission, bytes, len))
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
            d.dest_cache = dest.cache;
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
            /* A multi-slot ISEL wait is not reliable on G17P. After all
             * producers have issued, retire each slot through an identity of
             * the exact operand written by that slot. Each carrier has one
             * grounded wait bit; ISEL itself then needs none. */
            d.wait_mask = wait;
            if (util_bitcount(wait) > 1) {
               uint16_t remaining = wait;
               while (remaining) {
                  const uint16_t bit = remaining & (uint16_t)-(int16_t)remaining;
                  const uint8_t producer_slot = (uint8_t)ffs(bit);
                  agx_index operand;
                  if (!agx3_find_select_wait_operand(
                         sl, res, instr_index, I, producer_slot, &operand) ||
                      !agx3_retire_select_operand_wait(
                         emission, enc, operand, bit))
                     AGX3_REFUSE(I);
                  remaining &= (uint16_t)~bit;
               }
               d.wait_mask = 0;
            }

            uint8_t bytes[AGX3_ISEL_MAX_LENGTH];
            uint8_t len = 0;
            if (!agx3_isel_encode(enc, &d, bytes, &len) ||
                !agx3_append(emission, bytes, len)) {
               if (agx_compiler_debug & (AGX_DBG_SHADERS | AGX_DBG_REFUSE))
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
            /* The exact G17P LDIMM16 form has no witnessed scoreboard field.
             * A materialized half constant can nevertheless be scheduled
             * directly after an async producer, so retire its dependency
             * through the established native no-op before loading it. */
            if (I->imm > 0xffff ||
                (wait && !agx3_retire_slots_noop(emission, enc, wait)) ||
                !agx3_alu_mov16(enc, I->dest[0].value, I->imm, 0, bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         /* Native uniform arithmetic avoids per-use COPY temporaries while
          * preserving the existing GPR dependency mask. Uniform reads do not
          * acquire register cache/discard hints. */
         const int uniform_float = agx3_try_uniform_float(emission, enc, I, wait);
         if (uniform_float > 0)
            continue;
         if (uniform_float < 0)
            AGX3_REFUSE(I);

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
            if (agx3_src_is_register(src) && !src.memory && !src.abs &&
                (!src.neg || !I->saturate) &&
                (!src.kill || src.discard) && src.size == AGX_SIZE_32 &&
                agx_channels(src) == 1 &&
                zero.type == AGX_INDEX_IMMEDIATE && zero.value == 0x80 &&
                !zero.memory && !zero.abs && !zero.neg) {
               const struct agx3_falu_runtime runtime = {
                  .cache = {src.cache}, .discard = {src.discard},
                  .dest_cache = I->dest[0].cache, .wait_mask = wait,
               };
               uint8_t bytes[AGX3_FALU_MOD_MAX_LENGTH], len;
               /* -x is exactly -0.0 - x. */
               const bool encoded =
                  src.neg ? agx3_falu_narrow_imm_sub_f32(
                               enc, I->dest[0].value, 0x80, src.value,
                               &runtime, bytes, &len)
                          : agx3_falu_narrow_f32(enc, I->dest[0].value,
                                                 src.value, I->saturate,
                                                 &runtime, bytes, &len);
               if (!encoded || !agx3_append(emission, bytes, len))
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

         /* Both operands must be registers for the mixed-half endpoint. A
          * 32-bit register times a minifloat IMMEDIATE narrowed to a half is
          * a different, independently witnessed form (f10 selects the
          * destination width on the ordinary f32mul); let it fall through to
          * the minifloat path below instead of being claimed and refused
          * here. */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_FMUL &&
             I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
             agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
             I->dest[0].size == AGX_SIZE_16 &&
             agx_channels(I->dest[0]) == 1 &&
             agx3_src_is_register(I->src[0]) &&
             agx3_src_is_register(I->src[1]) &&
             ((I->src[0].size == AGX_SIZE_32 &&
               I->src[1].size == AGX_SIZE_16) ||
              (I->src[0].size == AGX_SIZE_16 &&
               I->src[1].size == AGX_SIZE_32))) {
            const bool half_first = I->src[0].size == AGX_SIZE_16;
            const agx_index f32 = I->src[half_first ? 1 : 0];
            const agx_index f16 = I->src[half_first ? 0 : 1];
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
            if (!agx3_src_is_register(f32) || !agx3_src_is_register(f16) ||
                f32.memory || f16.memory || (f32.value & 1) ||
                agx_channels(f32) != 1 || agx_channels(f16) != 1 ||
                f32.abs || f32.neg || f32.kill || f16.abs || f16.neg ||
                f16.kill || (f32.cache && f32.discard) ||
                (f16.cache && f16.discard) ||
                !agx3_alu_fmul_mixed_half(
                   enc, I->dest[0].value, f32.value, f16.value,
                   agx3_post_ra_hint(f32), agx3_post_ra_hint(f16),
                   I->dest[0].cache, wait, bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_HMUL &&
             I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
             ((agx3_src_is_register(I->src[0]) &&
               I->src[1].type == AGX_INDEX_IMMEDIATE) ||
              (I->src[0].type == AGX_INDEX_IMMEDIATE &&
               agx3_src_is_register(I->src[1])))) {
            const bool immediate_first =
               I->src[0].type == AGX_INDEX_IMMEDIATE;
            agx_index dst = I->dest[0];
            agx_index src = I->src[immediate_first ? 1 : 0];
            agx_index imm = I->src[immediate_first ? 0 : 1];
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

         /* A saturating half multiply is the same instruction with G17P's
          * f14 set, which moves it to the two-byte-longer ep2 point. This is
          * the shape probes/f16_saturate_texture.frag reaches the packer as
          * (clamp(texture(tex, uv) * vcolor, 0.0, 1.0) -> fmul.sat on halves)
          * and it was refused outright before the encoder existed. */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_HMUL &&
             I->nr_dests == 1 && I->nr_srcs == 2) {
            uint16_t src[2]; enum agx3_alu_hint hints[2];
            agx_index dst=I->dest[0];
            if (!agx3_src_is_register(dst) || dst.memory || dst.size != AGX_SIZE_16 ||
                agx_channels(dst) != 1)
               AGX3_REFUSE(I);
            for (unsigned i=0;i<2;i++) {
               agx_index x=I->src[i];
               if (!agx3_src_is_register(x) || x.memory || x.size != AGX_SIZE_16 ||
                   agx_channels(x) != 1 || x.neg || x.kill ||
                   (x.cache && x.discard))
                  AGX3_REFUSE(I);
               src[i]=x.value; hints[i]=agx3_post_ra_hint(x);
            }
            uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
            const bool src_abs[2] = {I->src[0].abs, I->src[1].abs};
            bool encoded = I->saturate
               ? !src_abs[0] && !src_abs[1] &&
                    agx3_alu_hmul_saturate(enc,dst.value,src,hints,dst.cache,
                                           wait,bytes,&len)
               : (src_abs[0] || src_abs[1])
                    ? agx3_alu_hmul_abs(enc, dst.value, src, hints, src_abs,
                                        dst.cache, wait, bytes, &len)
                    : agx3_alu_hmul(enc,dst.value,src,hints,dst.cache,wait,
                                    bytes,&len);

            /* The compact multiply does not cover every pair of odd half
             * registers.  The grounded three-source FMA does, and an exact
             * multiply is the fused product plus negative zero: it preserves
             * both the product rounding and signed zero.  Use this only when
             * the ordinary non-saturating instruction cannot encode the
             * assigned registers. */
            if (!encoded && !I->saturate && !src_abs[0] && !src_abs[1]) {
               encoded = agx3_alu_hfma_addend_minifloat(
                  enc, dst.value, src, UINT8_C(0x80), hints, dst.cache,
                  wait, bytes, &len);
            }

            if (!encoded || !agx3_append(emission,bytes,len))
               AGX3_REFUSE(I);
            continue;
         }

         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_HADD &&
             I->nr_dests == 1 && I->nr_srcs == 2 && I->saturate &&
             I->src[1].type == AGX_INDEX_IMMEDIATE) {
            const agx_index dst = I->dest[0];
            const agx_index src = I->src[0];
            const agx_index imm = I->src[1];
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
            if (!agx3_src_is_register(dst) || dst.memory ||
                dst.size != AGX_SIZE_16 || agx_channels(dst) != 1 ||
                !agx3_src_is_register(src) || src.memory ||
                src.size != AGX_SIZE_16 || agx_channels(src) != 1 || src.abs ||
                src.neg || src.kill || (src.cache && src.discard) ||
                imm.memory || imm.abs || imm.neg || imm.kill || imm.cache ||
                imm.discard || imm.size != AGX_SIZE_16 ||
                agx_channels(imm) != 1 || imm.value != UINT8_C(0x80) ||
                !agx3_alu_hadd_saturate_negzero(
                   enc, dst.value, src.value, agx3_post_ra_hint(src),
                   dst.cache, wait, bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_HADD &&
             I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
             I->src[1].type == AGX_INDEX_IMMEDIATE) {
            const agx_index dst = I->dest[0];
            const agx_index src = I->src[0];
            const agx_index imm = I->src[1];
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
            if (!agx3_src_is_register(dst) || dst.memory ||
                dst.size != AGX_SIZE_16 || agx_channels(dst) != 1 ||
                !agx3_src_is_register(src) || src.memory ||
                src.size != AGX_SIZE_16 || agx_channels(src) != 1 || src.abs ||
                !src.neg || src.kill || (src.cache && src.discard) ||
                imm.memory || imm.abs || imm.neg || imm.kill || imm.cache ||
                imm.discard || imm.size != AGX_SIZE_16 ||
                agx_channels(imm) != 1 || imm.value != UINT8_C(0x30) ||
                !agx3_alu_hadd_one_neg(enc, dst.value, src.value,
                                       agx3_post_ra_hint(src), dst.cache, wait,
                                       bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         /* Same for the saturating half add. The saturating half add against
          * an immediate is handled above; this is the register/register one. */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_HADD &&
             I->nr_dests == 1 && I->nr_srcs == 2) {
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
            const bool encoded =
               I->saturate ? agx3_alu_hadd_saturate(enc, dst.value, src, hints,
                                                    dst.cache, wait, bytes, &len)
                           : agx3_alu_hadd(enc, dst.value, src, hints,
                                           dst.cache, wait, bytes, &len);
            if (!encoded || !agx3_append(emission, bytes, len))
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

            /* Carry one producer in ep3's selector or several producers in
             * the extended endpoint's independent wait mask. */

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
             !I->dest[0].memory &&
             I->dest[0].size == AGX_SIZE_32 && agx_channels(I->dest[0]) == 1 &&
             (I->src[0].size == AGX_SIZE_16 || I->src[1].size == AGX_SIZE_16) &&
             agx3_src_is_register(I->src[0]) && agx3_src_is_register(I->src[1])) {
            const bool half_first = I->src[0].size == AGX_SIZE_16;
            const agx_index f32 = I->src[half_first ? 1 : 0];
            const agx_index f16 = I->src[half_first ? 0 : 1];
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
            if (f32.size != AGX_SIZE_32 || f16.size != AGX_SIZE_16 ||
                agx_channels(f32) != 1 || agx_channels(f16) != 1 ||
                f32.memory || f16.memory ||
                f32.neg || f16.abs || f16.neg ||
                f32.kill || f16.kill || (f32.cache && f32.discard) ||
                (f16.cache && f16.discard) ||
                !(f32.abs
                     ? agx3_alu_fmul_abs_mixed(
                          enc, I->dest[0].value, f32.value, f16.value,
                          agx3_post_ra_hint(f32), agx3_post_ra_hint(f16),
                          I->dest[0].cache, wait, bytes, &len)
                     : agx3_alu_fmul_mixed(
                          enc, I->dest[0].value, f32.value, f16.value,
                          agx3_post_ra_hint(f32), agx3_post_ra_hint(f16),
                          I->dest[0].cache, wait, bytes, &len)) ||
                !agx3_append(emission, bytes, len))
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

         /* Float to a 16-bit integer. The source width is its own field, so
          * a HALF source is the same instruction: `int(clamp(x, 1.0, 32.0))`
          * written in half precision -- Hyprland's motion_blur.glsl sample
          * count, and the loop bound it then becomes -- reaches this with a
          * 16-bit source and used to fail the whole shader closed. */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_CONVERT &&
             I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
             I->round == AGX_ROUND_RTZ &&
             I->src[0].type == AGX_INDEX_IMMEDIATE &&
             (I->src[0].value == AGX_CONVERT_F_TO_S16 ||
              I->src[0].value == AGX_CONVERT_F_TO_U16) &&
             agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
             I->dest[0].size == AGX_SIZE_16 &&
             agx_channels(I->dest[0]) == 1 &&
             (agx3_vary_reg32(I->src[1]) || agx3_half_reg(I->src[1])) &&
             agx_channels(I->src[1]) == 1 &&
             !I->src[1].abs && !I->src[1].neg && !I->src[1].kill &&
             !(I->src[1].cache && I->src[1].discard)) {
            uint16_t convert_wait = wait;
            if (convert_wait != 0 && convert_wait != 32) {
               if (!agx3_retire_slots_noop(emission, enc, convert_wait))
                  AGX3_REFUSE(I);
               convert_wait = 0;
            }
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
            if (!agx3_alu_float_to_int16(
                   enc, I->dest[0].value, I->src[1].value,
                   I->src[1].size == AGX_SIZE_16,
                   I->src[0].value == AGX_CONVERT_F_TO_S16,
                   agx3_post_ra_hint(I->src[1]), I->dest[0].cache,
                   convert_wait, bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         /* shaderFloat16 commonly converts invocation/index values directly
          * from signed or unsigned32 into a scalar half. This has a native
          * result-width form distinct from the established i32-to-f32 form. */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_CONVERT &&
             I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
             I->round == AGX_ROUND_RTE &&
             I->src[0].type == AGX_INDEX_IMMEDIATE &&
             (I->src[0].value == AGX_CONVERT_U32_TO_F ||
              I->src[0].value == AGX_CONVERT_S32_TO_F) &&
             agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
             I->dest[0].size == AGX_SIZE_16 &&
             agx_channels(I->dest[0]) == 1 &&
             agx3_vary_reg32(I->src[1]) && agx_channels(I->src[1]) == 1 &&
             !I->src[1].abs && !I->src[1].neg && !I->src[1].kill &&
             !(I->src[1].cache && I->src[1].discard)) {
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
            if (!agx3_alu_i32_to_f16(
                   enc, I->dest[0].value, I->src[1].value,
                   I->src[0].value == AGX_CONVERT_S32_TO_F,
                   agx3_post_ra_hint(I->src[1]), I->dest[0].cache, wait,
                   bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         /* int8/int16 conversions keep their input in a half register. The
          * int2flt byte selector reads only the low eight bits, so a byte
          * converts without an extension. Apple has no unsigned byte to half
          * form: `half(uchar)` masks the byte in place and converts the
          * resulting ushort, so that sequence is emitted here too. */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_CONVERT &&
             I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
             I->round == AGX_ROUND_RTE &&
             I->src[0].type == AGX_INDEX_IMMEDIATE &&
             I->src[1].size == AGX_SIZE_16 &&
             (((I->src[0].value == AGX_CONVERT_S8_TO_F ||
                I->src[0].value == AGX_CONVERT_U8_TO_F) &&
               (I->dest[0].size == AGX_SIZE_16 ||
                I->dest[0].size == AGX_SIZE_32)) ||
              ((I->src[0].value == AGX_CONVERT_S16_TO_F ||
                I->src[0].value == AGX_CONVERT_U16_TO_F) &&
               I->dest[0].size == AGX_SIZE_16))) {
            const agx_index dest = I->dest[0], src = I->src[1];
            if (!agx3_src_is_register(dest) || dest.memory ||
                agx_channels(dest) != 1 || !agx3_src_is_register(src) ||
                src.memory || src.abs || src.neg || src.kill ||
                agx_channels(src) != 1 || (src.cache && src.discard))
               AGX3_REFUSE(I);

            const enum agx_convert mode = I->src[0].value;
            const bool byte =
               mode == AGX_CONVERT_S8_TO_F || mode == AGX_CONVERT_U8_TO_F;
            const bool is_signed =
               mode == AGX_CONVERT_S8_TO_F || mode == AGX_CONVERT_S16_TO_F;
            const bool dest_16 = dest.size == AGX_SIZE_16;
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
            if (byte && !is_signed && dest_16) {
               if (!agx3_alu_bit_imm(enc, AGX3_ALU_BAND, true, dest.value,
                                     src.value, 0xff, agx3_post_ra_hint(src),
                                     wait, bytes, &len) ||
                   !agx3_append(emission, bytes, len) ||
                   !agx3_alu_narrow_int_to_float(
                      enc, dest.value, true, dest.value, false, false,
                      AGX3_ALU_HINT_DISCARD, dest.cache, 0, bytes, &len) ||
                   !agx3_append(emission, bytes, len))
                  AGX3_REFUSE(I);
               continue;
            }

            if (!agx3_alu_narrow_int_to_float(
                   enc, dest.value, dest_16, src.value, byte, is_signed,
                   agx3_post_ra_hint(src), dest.cache, wait, bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         /* GTK's turbulence fragment keeps its octave arithmetic in scalar
          * halves, then indexes the permutation table with a direct f16-to-
         * i32 cast. flt2int has a native half-source selector; widening the
         * value would change both register pressure and rounding. Preserve
         * the signed/unsigned mode, post-RA lifetime/cache state and all six
         * producer dependencies in the native conversion itself. */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_CONVERT &&
             I->nr_srcs == 2 &&
             I->src[0].type == AGX_INDEX_IMMEDIATE &&
             (I->src[0].value == AGX_CONVERT_F_TO_S32 ||
              I->src[0].value == AGX_CONVERT_F_TO_U32) &&
             I->src[1].size == AGX_SIZE_16) {
            if (I->nr_dests != 1 || I->saturate ||
                I->round != AGX_ROUND_RTZ ||
                I->src[0].abs || I->src[0].neg || I->src[0].memory ||
                I->src[0].kill || I->src[0].cache || I->src[0].discard ||
                !agx3_vary_reg32(I->dest[0]) ||
                agx_channels(I->dest[0]) != 1 ||
                !agx3_src_is_register(I->src[1]) || I->src[1].memory ||
                agx_channels(I->src[1]) != 1 || I->src[1].abs ||
                I->src[1].neg || I->src[1].kill ||
                (I->src[1].cache && I->src[1].discard))
               AGX3_REFUSE(I);

            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
            if (!agx3_alu_f16_to_int32(
                   enc, I->dest[0].value, I->src[1].value,
                   I->src[0].value == AGX_CONVERT_F_TO_S32,
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
         /* The multiply's destination width is selected independently of its
          * source width (f10 vs f26), so a 32-bit value scaled by a minifloat
          * straight into a half is one instruction. agx_optimizer_fmov_rev()
          * folds the f2f16 of `packHalf2x16(vec2(x * k, ...))` into the
          * multiply and produces exactly that shape; it used to fall off this
          * path and refuse the shader. Only FMUL is witnessed narrowing. */
         if ((I->op == AGX_OPCODE_FADD || I->op == AGX_OPCODE_FMUL) &&
             I->nr_dests == 1 &&
             I->nr_srcs == 2 &&
             (I->op == AGX_OPCODE_FADD || !I->saturate) &&
             agx3_src_is_register(I->dest[0]) &&
             (I->dest[0].size == AGX_SIZE_32 ||
              (I->op == AGX_OPCODE_FMUL &&
               I->dest[0].size == AGX_SIZE_16)) &&
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
               /* Saturation selects the long ep2, which carries all six wait
                * bits; a bare negation stays on the compact ep1. */
               uint16_t encoded_wait = wait;
               if (!I->saturate &&
                   !agx3_fuse_compact_minifloat_wait(emission, enc,
                                                     &encoded_wait))
                  AGX3_REFUSE(I);
               encoded = agx3_alu_fadd_minifloat_mod(
                  enc, I->dest[0].value, reg.value, imm.value, reg.neg,
                  I->saturate, agx3_post_ra_hint(reg), encoded_wait, bytes,
                  &len);
            } else if (multiply) {
               /* The recovered six-byte minifloat endpoint fixes bits 32:33.
                * Treating bit 33 as an ABS modifier produced wrong values on
                * G17P: abs(0.027...) * 2.0 saturated above 1.0 while the
                * identical plain multiply was exact. Use two independently
                * Apple-grounded instructions instead. FABS writes the final
                * destination as a retained intermediate; the plain FMUL then
                * consumes it in place. This adds no register, spill, wait, or
                * serialization and preserves the source's real lifetime. */
               if (reg.abs) {
                  uint8_t abs_bytes[AGX3_ALU_MAX_LENGTH], abs_len = 0;
                  uint8_t mul_bytes[AGX3_ALU_MAX_LENGTH], mul_len = 0;
                  if (I->dest[0].size != AGX_SIZE_32 ||
                      !agx3_alu_fadd_abs_negzero(
                         enc, I->dest[0].value, reg.value, true,
                         agx3_post_ra_hint(reg), wait, abs_bytes, &abs_len) ||
                      !agx3_alu_fmul_minifloat_mod(
                         enc, I->dest[0].value, I->dest[0].value, imm.value,
                         I->dest[0].cache, false, AGX3_ALU_HINT_DISCARD, 0,
                         mul_bytes, &mul_len) ||
                      !agx3_append(emission, abs_bytes, abs_len) ||
                      !agx3_append(emission, mul_bytes, mul_len))
                     AGX3_REFUSE(I);
                  continue;
               }
               /* Every grounded minifloat multiply is a compact form, at both
                * destination widths, so a multi-slot wait needs the identity
                * prelude here exactly as it does on the compact ADD below. */
               uint16_t encoded_wait = wait;
               if (!agx3_fuse_compact_minifloat_wait(emission, enc,
                                                     &encoded_wait))
                  AGX3_REFUSE(I);
               encoded = agx3_alu_fmul_minifloat_narrow(
                  enc, I->dest[0].value, reg.value, imm.value,
                  I->dest[0].cache, false,
                  I->dest[0].size == AGX_SIZE_16, agx3_post_ra_hint(reg),
                  encoded_wait, bytes, &len);
            } else if (reg.abs) {
               encoded = agx3_alu_fadd_abs_minifloat(
                  enc, I->dest[0].value, reg.value, imm.value,
                  I->dest[0].cache, reg.neg, I->saturate,
                  agx3_post_ra_hint(reg), wait, bytes, &len);
            } else {
               /* A real sky shader reaches this operation with one source
                * dependency and an independent write-after-write dependency
                * on its retained destination. */
               uint16_t encoded_wait = wait;
               if (!agx3_fuse_compact_minifloat_wait(emission, enc,
                                                     &encoded_wait))
                  AGX3_REFUSE(I);
               encoded = agx3_alu_fadd_minifloat(
                  enc, I->dest[0].value, reg.value, imm.value,
                  I->dest[0].cache, agx3_post_ra_hint(reg), encoded_wait,
                  bytes, &len);
            }
            if (!encoded || !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         /* A narrow NOT is Apple's unary 16-bit `inv`; the bitop's second
          * source is ignored by the truth table. */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_BITOP &&
             (I->truth_table & 0xfu) == AGX3_SEL_TT_NOT &&
             I->nr_dests == 1 && I->nr_srcs >= 1 && !I->saturate &&
             I->dest[0].size == AGX_SIZE_16 && I->src[0].size == AGX_SIZE_16 &&
             agx_channels(I->dest[0]) == 1 && agx_channels(I->src[0]) == 1 &&
             agx3_src_is_register(I->dest[0]) &&
             agx3_src_is_register(I->src[0]) && !I->src[0].abs &&
             !I->src[0].neg) {
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
            if (I->src[0].kill ||
                !agx3_alu_bit_not_reg16(enc, I->dest[0].value,
                                        I->src[0].value,
                                        agx3_post_ra_hint(I->src[0]),
                                        I->dest[0].cache, wait, bytes, &len) ||
                !agx3_append(emission, bytes, len))
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

         /* AGX3b's compact 32-bit move cannot address full registers above
          * r95. Chromium's animation_keyframes vertex shader moves values in
          * both directions across that boundary. The native 16-bit
          * bitwise-immediate form addresses every half register, so preserve
          * the exact 32-bit value as two ordered `x | 0` operations. Carry
          * the producer wait on the first half; the second half belongs to
          * the same dependency and needs no additional wait. */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_BITOP &&
             (I->truth_table & 0xfu) == AGX3_SEL_TT_MOV &&
             I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
             I->shift == 0 && agx3_vary_reg32(I->dest[0]) &&
             agx_channels(I->dest[0]) == 1 &&
             agx3_vary_reg32(I->src[0]) &&
             agx_channels(I->src[0]) == 1 && !I->src[0].kill &&
             !(I->src[0].cache && I->src[0].discard) &&
             I->src[1].type == AGX_INDEX_IMMEDIATE &&
             I->src[1].value == 0 && !I->src[1].memory &&
             !I->src[1].abs && !I->src[1].neg &&
             (I->dest[0].value >= 192 || I->src[0].value >= 192)) {
            uint8_t low[AGX3_ALU_MAX_LENGTH], high[AGX3_ALU_MAX_LENGTH];
            uint8_t low_len, high_len;
            if (!agx3_alu_bit_imm(
                   enc, AGX3_ALU_BOR, true, I->dest[0].value,
                   I->src[0].value, 0, agx3_post_ra_hint(I->src[0]),
                   wait, low, &low_len) ||
                !agx3_alu_bit_imm(
                   enc, AGX3_ALU_BOR, true, I->dest[0].value + 1,
                   I->src[0].value + 1, 0,
                   agx3_post_ra_hint(I->src[0]), 0, high, &high_len) ||
                !agx3_append(emission, low, low_len) ||
                !agx3_append(emission, high, high_len))
               AGX3_REFUSE(I);
            continue;
         }

         /*
          * The same bitwise operation against an IMMEDIATE. The operand kind
          * is a field of the encoding point the register form already uses
          * (f37, with the constant in f29), so `x & 0xff`, `x | mask` and
          * `x ^ mask` are that instruction with one field changed -- not a
          * separate form, and not something to fail a shader closed over.
          * AND/OR/XOR all commute, so an immediate in either source slot
          * lands in the one slot whose kind field takes it.
          */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_BITOP &&
             I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
             agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
             agx_channels(I->dest[0]) == 1 &&
             (I->dest[0].size == AGX_SIZE_16 ||
              I->dest[0].size == AGX_SIZE_32)) {
            const unsigned tt = I->truth_table & 0xfu;
            const bool immediate_first =
               I->src[0].type == AGX_INDEX_IMMEDIATE;
            const agx_index imm = I->src[immediate_first ? 0 : 1];
            const agx_index reg = I->src[immediate_first ? 1 : 0];
            const bool narrow = I->dest[0].size == AGX_SIZE_16;
            if ((tt == AGX3_SEL_TT_AND || tt == AGX3_SEL_TT_OR ||
                 tt == AGX3_SEL_TT_XOR) &&
                imm.type == AGX_INDEX_IMMEDIATE && imm.value <= 0xff &&
                !imm.abs && !imm.neg && !imm.memory &&
                agx3_src_is_register(reg) && !reg.memory && !reg.abs &&
                !reg.neg && !reg.kill && agx_channels(reg) == 1 &&
                !(reg.cache && reg.discard) &&
                reg.size == I->dest[0].size) {
               const enum agx3_alu_op op = tt == AGX3_SEL_TT_AND
                                              ? AGX3_ALU_BAND
                                              : tt == AGX3_SEL_TT_OR
                                                   ? AGX3_ALU_BOR
                                                   : AGX3_ALU_BXOR;
               uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
               if (!agx3_alu_bit_imm(enc, op, narrow, I->dest[0].value,
                                     reg.value, (uint8_t)imm.value,
                                     agx3_post_ra_hint(reg), wait, bytes,
                                     &len) ||
                   !agx3_append(emission, bytes, len))
                  AGX3_REFUSE(I);
               continue;
            }
         }

         /* The generic selector's original f32-to-i32 vector had a fixed
          * source-discard bit and no cache fields. That destroys a value that
          * remains live for a following compare. Encode the actual post-RA
          * lifetime state through the native flt2int fields instead. */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_CONVERT &&
             I->nr_srcs == 2 &&
             I->src[0].type == AGX_INDEX_IMMEDIATE &&
             (I->src[0].value == AGX_CONVERT_F_TO_S32 ||
              I->src[0].value == AGX_CONVERT_F_TO_U32) &&
             I->src[1].size == AGX_SIZE_32) {
            if (I->nr_dests != 1 || I->saturate ||
                I->round != AGX_ROUND_RTZ ||
                I->src[0].abs || I->src[0].neg || I->src[0].memory ||
                I->src[0].kill || I->src[0].cache || I->src[0].discard ||
                !agx3_vary_reg32(I->dest[0]) ||
                agx_channels(I->dest[0]) != 1 ||
                !agx3_vary_reg32(I->src[1]) ||
                agx_channels(I->src[1]) != 1 || I->src[1].abs ||
                I->src[1].neg || I->src[1].kill ||
                (I->src[1].cache && I->src[1].discard))
               AGX3_REFUSE(I);

            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
            if (!agx3_alu_float_to_int32(
                   enc, I->dest[0].value, I->src[1].value,
                   I->src[0].value == AGX_CONVERT_F_TO_S32,
                   agx3_post_ra_hint(I->src[1]), I->dest[0].cache, wait,
                   bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
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
             agx3_mod_src_ok(I->src[0]) &&
             agx_channels(I->src[0]) == 1 &&
             agx3_mod_src_ok(I->src[1]) &&
             agx_channels(I->src[1]) == 1 &&
             I->src[2].type == AGX_INDEX_IMMEDIATE && I->src[2].value <= 255 &&
             !I->src[2].abs && !I->src[2].neg && !I->src[2].memory) {
            /* Multiplication is commutative. Apple's G17P form puts a lone
             * absolute multiplier in the f35 source slot; match that
             * canonical order while moving the post-RA lifetime hints with
             * their operands. */
            const bool swap = !I->src[0].abs && I->src[1].abs;
            const agx_index src0 = I->src[swap ? 1 : 0];
            const agx_index src1 = I->src[swap ? 0 : 1];
            struct agx3_falu_runtime runtime = {
               .dest_cache = I->dest[0].cache, .wait_mask = wait,
            };
            for (unsigned i = 0; i < 2; ++i) {
               const agx_index src = I->src[swap ? 1 - i : i];
               if (src.kill)
                  AGX3_REFUSE(I);
               runtime.cache[i] = src.cache;
               runtime.discard[i] = src.discard;
            }
            uint8_t bytes[AGX3_FALU_MOD_MAX_LENGTH], length;
            if (!agx3_falu_fma_addend_imm_mod(enc, I->dest[0].value,
                    src0.value, src1.value, I->src[2].value,
                    src0.abs, src0.neg, src1.abs, src1.neg,
                    &runtime, bytes, &length) ||
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
             I->nr_dests == 1 && I->nr_srcs == 3 &&
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
                       imm.value,add.value,add.neg,I->saturate,agx3_post_ra_hint(reg),
                       agx3_post_ra_hint(add),I->dest[0].cache,wait,bytes,&len) ||
                   !agx3_append(emission,bytes,len))
                  AGX3_REFUSE(I);
               continue;
            }
         }

         /* The same native FMA, with the modifiers the compact minifloat form
          * above cannot carry.  Operand KIND (f42 selecting the f34 immediate
          * over the f35 register slot) is a field of the same encoding point
          * as the f30/f31 and f64/f65 modifier bits and the f26/f60 widths,
          * so `k * fabs(a) + fabs(b)` is one instruction, not a shape without
          * an encoding -- Apple composes exactly this cell.  glmark2's
          * terrain noise kernel (`vec4 h = 1.0 - abs(x) - abs(y);`) is the
          * shape that used to take the whole fragment shader down with it.
          *
          * A modifier on the immediate itself is folded into the minifloat:
          * Apple encodes fma(0.5, -b, c) as f34 = -0.5, not as a neg bit,
          * and there is no register slot here to hang one on. */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_FFMA &&
             I->nr_dests == 1 && I->nr_srcs == 3 &&
             agx3_vary_reg32(I->dest[0]) && agx_channels(I->dest[0]) == 1 &&
             (I->src[0].type == AGX_INDEX_IMMEDIATE) !=
                (I->src[1].type == AGX_INDEX_IMMEDIATE)) {
            const bool imm_first = I->src[0].type == AGX_INDEX_IMMEDIATE;
            const agx_index imm = I->src[imm_first ? 0 : 1];
            const agx_index reg = I->src[imm_first ? 1 : 0];
            const agx_index add = I->src[2];

            if (agx3_plain_immediate(imm) && imm.value <= 255 &&
                agx3_mod_src_any_width(reg) && agx3_mod_src_any_width(add)) {
               if (reg.kill || add.kill || (reg.cache && reg.discard) ||
                   (add.cache && add.discard))
                  AGX3_REFUSE(I);

               uint8_t multiplier = imm.value;
               if (imm.abs)
                  multiplier &= 0x7f;
               if (imm.neg)
                  multiplier ^= 0x80;

               /* The multiplicand's negate folds into the multiplier too:
                * k * -b == -k * b, and Apple encodes it that way rather than
                * setting f31, which it never emits beside an immediate. */
               if (reg.neg)
                  multiplier ^= 0x80;

               const struct agx3_falu_mods m = {
                  .src1_abs = reg.abs,
                  .src1_16 = reg.size == AGX_SIZE_16,
                  .addend_abs = add.abs,
                  .addend_neg = add.neg,
                  .addend_16 = add.size == AGX_SIZE_16,
                  .saturate = I->saturate,
               };
               const struct agx3_falu_runtime runtime = {
                  .cache = {false, reg.cache, add.cache},
                  .discard = {false, reg.discard, add.discard},
                  .dest_cache = I->dest[0].cache,
                  .wait_mask = wait,
               };
               uint8_t bytes[AGX3_FALU_MOD_MAX_LENGTH], length;
               if (!agx3_falu_fma_mul_imm_mod(enc, I->dest[0].value,
                       multiplier, reg.value, add.value, &m, &runtime,
                       bytes, &length) ||
                   !agx3_append(emission, bytes, length))
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
             I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
             I->src[1].type == AGX_INDEX_IMMEDIATE &&
             I->src[1].value >= 1 &&
             !I->src[1].abs && !I->src[1].neg) {
            const agx_index dest=I->dest[0], src=I->src[0];
            const bool half = dest.size == AGX_SIZE_16 &&
                              src.size == AGX_SIZE_16;
            const bool word = agx3_vary_reg32(dest) && agx3_vary_reg32(src);
            if ((!half && !word) ||
                I->src[1].value > (half ? 15 : 31) ||
                !agx3_src_is_register(dest) || dest.memory ||
                !agx3_src_is_register(src) || src.memory ||
                src.abs || src.neg || agx_channels(dest)!=1 || agx_channels(src)!=1)
               AGX3_REFUSE(I);
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], length;
            const bool encoded = half
               ? agx3_alu_asr_imm16(enc,dest.value,src.value,I->src[1].value,
                                     agx3_post_ra_hint(src),dest.cache,wait,
                                     bytes,&length)
               : agx3_alu_asr_imm32(enc,dest.value,src.value,I->src[1].value,
                                     agx3_post_ra_hint(src),dest.cache,wait,
                                     bytes,&length);
            if (!encoded ||
                !agx3_append(emission,bytes,length))
               AGX3_REFUSE(I);
            continue;
         }

         /* A dynamic signed-short shift stays narrow on G17P. Apple emits a
          * single mixed-width SRA: half value/result and a full register
          * count. Preserve the two source lifetimes and producer waits. */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_ASR &&
             I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
             I->dest[0].size == AGX_SIZE_16 &&
             I->src[0].size == AGX_SIZE_16 &&
             I->src[1].size == AGX_SIZE_32) {
            const agx_index dest = I->dest[0];
            const agx_index value = I->src[0];
            const agx_index shift = I->src[1];
            if (!agx3_src_is_register(dest) || dest.memory ||
                agx_channels(dest) != 1 ||
                !agx3_src_is_register(value) || value.memory || value.abs ||
                value.neg || value.kill || agx_channels(value) != 1 ||
                !agx3_src_is_register(shift) || shift.memory || shift.abs ||
                shift.neg || shift.kill || agx_channels(shift) != 1 ||
                (value.cache && value.discard) ||
                (shift.cache && shift.discard))
               AGX3_REFUSE(I);

            const enum agx3_alu_hint hints[2] = {
               agx3_post_ra_hint(value), agx3_post_ra_hint(shift),
            };
            uint16_t w = wait;
            if (!agx3_retire_extra_waits(emission, enc, sl, res, instr_index,
                                         I, &w))
               AGX3_REFUSE(I);
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], length = 0;
            if (!agx3_alu_asr_reg16(enc, dest.value, value.value, shift.value,
                                    hints, dest.cache, w, bytes, &length) ||
                !agx3_append(emission, bytes, length))
               AGX3_REFUSE(I);
            continue;
         }

         /* A scalar-half `1 << count` needs no register holding the one: the
          * native `sll` takes an immediate value. Apple reads a half count
          * here; NIR's uint count register supplies its low half. A cached
          * count keeps its register through the clear last-use bit. */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_BFI &&
             I->nr_dests == 1 && I->nr_srcs == 3 && I->bfi_mask == 0 &&
             agx3_src_is_zero_imm(I->src[0]) &&
             I->src[1].type == AGX_INDEX_IMMEDIATE &&
             I->src[1].value == 1 && agx3_src_is_register(I->src[2]) &&
             I->dest[0].size == AGX_SIZE_16 &&
             I->src[1].size == AGX_SIZE_16 &&
             I->src[2].size == AGX_SIZE_32) {
            const agx_index dest = I->dest[0];
            const agx_index one = I->src[1];
            const agx_index shift = I->src[2];
            const agx_index base = I->src[0];
            if (I->saturate || base.memory || base.abs || base.neg ||
                base.kill || base.cache || base.discard || one.memory ||
                one.abs || one.neg || one.kill || one.cache || one.discard ||
                !agx3_src_is_register(dest) || dest.memory ||
                agx_channels(dest) != 1 ||
                !agx3_src_is_register(shift) || shift.memory ||
                agx_channels(shift) != 1 || shift.abs || shift.neg ||
                shift.kill || (shift.cache && shift.discard))
               AGX3_REFUSE(I);

            const enum agx3_alu_hint shift_hint =
               shift.cache ? AGX3_ALU_HINT_NONE : agx3_post_ra_hint(shift);
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], length = 0;
            if (!agx3_alu_sll_one_reg32(enc, dest.value, shift.value,
                                        shift_hint, dest.cache, wait, bytes,
                                        &length) ||
                !agx3_append(emission, bytes, length))
               AGX3_REFUSE(I);
            continue;
         }

         /* A register-count ushort shift is one native `sll`. The shift
          * operand remains uint in NIR; Apple's mixed-width witness pins
          * source A at width 1 and source B at width 2. */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_BFI &&
             I->nr_dests == 1 && I->nr_srcs == 3 && I->bfi_mask == 0 &&
             agx3_src_is_zero_imm(I->src[0]) &&
             agx3_src_is_register(I->src[2]) &&
             I->dest[0].size == AGX_SIZE_16 &&
             I->src[1].size == AGX_SIZE_16 &&
             I->src[2].size == AGX_SIZE_32) {
            const agx_index dest = I->dest[0], value = I->src[1];
            const agx_index shift = I->src[2];
            const agx_index base = I->src[0];
            if (I->saturate || base.memory || base.abs || base.neg ||
                base.kill || base.cache || base.discard ||
                !agx3_src_is_register(dest) || dest.memory ||
                dest.size != AGX_SIZE_16 || agx_channels(dest) != 1 ||
                !agx3_src_is_register(value) || value.memory ||
                value.size != AGX_SIZE_16 || agx_channels(value) != 1 ||
                !agx3_src_is_register(shift) || shift.memory ||
                shift.size != AGX_SIZE_32 || agx_channels(shift) != 1 ||
                value.abs || value.neg || value.kill ||
                shift.abs || shift.neg || shift.kill ||
                (value.cache && value.discard) ||
                (shift.cache && shift.discard))
               AGX3_REFUSE(I);
            const uint16_t src[2] = {value.value, shift.value};
            const bool src_16[2] = {true, false};
            const enum agx3_alu_hint hints[2] = {
               agx3_post_ra_hint(value),
               /* This endpoint has a discard bit but no measured cache bit
                * for the count. NONE retains the physical register. */
               shift.cache ? AGX3_ALU_HINT_NONE : agx3_post_ra_hint(shift),
            };
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
            if (!agx3_alu_sll_reg(enc, dest.value, true, dest.cache, src,
                                  src_16, hints, wait, bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         /*
          * G17P has no single 16-bit shift-left form beyond four bits. Apple
          * lowers ushort << {5,6,7,8} to two witnessed shift-adds: first
          * 0 + (value << (amount - 4)), then 0 + (temporary << 4). The
          * destination itself is the temporary, so this remains post-RA and
          * needs no invented register. Alpha-to-coverage uses shift 7 here.
          */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_BFI &&
             I->nr_dests == 1 && I->nr_srcs == 3 && I->bfi_mask == 0 &&
             agx3_src_is_zero_imm(I->src[0]) &&
             I->src[2].type == AGX_INDEX_IMMEDIATE &&
             I->src[2].value <= 8 && !I->src[2].memory &&
             !I->src[2].abs && !I->src[2].neg && !I->src[2].kill &&
             !I->src[2].cache && !I->src[2].discard) {
            const agx_index dest = I->dest[0], value = I->src[1];
            const agx_index base = I->src[0];
            if (I->saturate || base.memory || base.abs || base.neg ||
                base.kill || base.cache || base.discard ||
                !agx3_src_is_register(dest) || dest.memory ||
                dest.size != AGX_SIZE_16 || agx_channels(dest) != 1 ||
                !agx3_src_is_register(value) || value.memory ||
                value.size != AGX_SIZE_16 || agx_channels(value) != 1 ||
                value.abs || value.neg || value.kill ||
                (value.cache && value.discard))
               AGX3_REFUSE(I);

            /* The shift-add descriptor has no source cache bit. A cached
             * value is still live, which the clear last-use bit preserves. */
            const bool split = I->src[2].value > 4;
            const struct agx3_shift_add_desc first = {
               .dest = dest.value,
               .dest_hint = split || dest.cache,
               .src0_immediate = true,
               .src0 = 0,
               .src1 = value.value,
               .src1_kill = value.discard,
               .shift = split ? I->src[2].value - 4 : I->src[2].value,
               .wait_mask = wait,
            };
            uint8_t bytes[AGX3_WRITEOUT_MAX_LENGTH];
            uint8_t len = 0;
            if (!agx3_shift_add16_encode(enc, &first, bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);

            if (split) {
               const struct agx3_shift_add_desc second = {
                  .dest = dest.value,
                  .dest_hint = dest.cache,
                  .src0_immediate = true,
                  .src0 = 0,
                  .src1 = dest.value,
                  .src1_kill = true,
                  .shift = 4,
               };
               if (!agx3_shift_add16_encode(enc, &second, bytes, &len) ||
                   !agx3_append(emission, bytes, len))
                  AGX3_REFUSE(I);
            }
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

         /* Apple exposes [[sample_mask]] in bits 4..7 of a native ushort
          * special-register read.  Preserve that narrow value through the
          * exact immediate SRL instead of materializing a full-width shift
          * count, which selects a different G17P instruction domain. */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_BFEIL &&
             I->nr_dests == 1 && I->nr_srcs == 3 && I->bfi_mask == 0 &&
             agx3_src_is_zero_imm(I->src[0]) &&
             I->src[2].type == AGX_INDEX_IMMEDIATE &&
             I->src[2].value >= 1 && I->src[2].value <= 31 &&
             !I->src[2].memory && !I->src[2].abs && !I->src[2].neg &&
             !I->src[2].kill && !I->src[2].cache &&
             !I->src[2].discard) {
            const agx_index dest = I->dest[0], value = I->src[1];
            const agx_index base = I->src[0];
            if (I->saturate || base.memory || base.abs || base.neg ||
                base.kill || base.cache || base.discard ||
                !agx3_src_is_register(dest) || dest.memory ||
                dest.size != AGX_SIZE_16 || agx_channels(dest) != 1 ||
                !agx3_src_is_register(value) || value.memory ||
                value.size != AGX_SIZE_16 || agx_channels(value) != 1 ||
                value.abs || value.neg || value.kill ||
                (value.cache && value.discard))
               AGX3_REFUSE(I);

            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
            if (!agx3_alu_srl_imm16(
                   enc, dest.value, value.value, I->src[2].value,
                   agx3_post_ra_hint(value), dest.cache, wait, bytes, &len) ||
                !agx3_append(emission, bytes, len))
               AGX3_REFUSE(I);
            continue;
         }

         /* Dynamic ushort right-shifts use the same mixed-width register
          * domain as SLL. agx_emit_alu has already changed the ordinary NIR
          * count into G17P's signed bitfield boundary. */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_BFEIL &&
             I->nr_dests == 1 && I->nr_srcs == 3 && I->bfi_mask == 0 &&
             agx3_src_is_zero_imm(I->src[0]) &&
             agx3_src_is_register(I->src[2]) &&
             I->dest[0].size == AGX_SIZE_16 &&
             I->src[1].size == AGX_SIZE_16 &&
             I->src[2].size == AGX_SIZE_32) {
            const agx_index dest = I->dest[0], value = I->src[1];
            const agx_index shift = I->src[2];
            const agx_index base = I->src[0];
            if (I->saturate || base.memory || base.abs || base.neg ||
                base.kill || base.cache || base.discard ||
                !agx3_src_is_register(dest) || dest.memory ||
                dest.size != AGX_SIZE_16 || agx_channels(dest) != 1 ||
                !agx3_src_is_register(value) || value.memory ||
                value.size != AGX_SIZE_16 || agx_channels(value) != 1 ||
                shift.memory || shift.size != AGX_SIZE_32 ||
                agx_channels(shift) != 1 || value.abs || value.neg ||
                value.kill || shift.abs || shift.neg || shift.kill ||
                (value.cache && value.discard) ||
                (shift.cache && shift.discard))
               AGX3_REFUSE(I);
            const uint16_t src[2] = {value.value, shift.value};
            const bool src_16[2] = {true, false};
            const enum agx3_alu_hint hints[2] = {
               agx3_post_ra_hint(value),
               /* This endpoint has a discard bit but no grounded cache bit
                * for the shift count. NONE also retains the register, so a
                * cached post-RA source loses no lifetime information. */
               shift.cache ? AGX3_ALU_HINT_NONE : agx3_post_ra_hint(shift),
            };
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
            if (!agx3_alu_srl_reg(enc, dest.value, true, dest.cache, src,
                                  src_16, hints, wait, bytes, &len) ||
                !agx3_append(emission, bytes, len))
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
            /* NIR's bit_count of a 16-bit value returns 32 bits. Apple counts
             * into the low half and clears the high half, as for
             * `uint(popcount(ushort))`. */
            const bool widened = dest.size == AGX_SIZE_32 &&
                                 src.size == AGX_SIZE_16 &&
                                 !(dest.value & 1);
            if ((!half && !word && !widened) ||
                !agx3_src_is_register(dest) || dest.memory ||
                agx_channels(dest) != 1 ||
                !agx3_src_is_register(src) || src.memory ||
                agx_channels(src) != 1 || src.abs || src.neg)
               AGX3_REFUSE(I);

            uint8_t bytes[AGX3_ALU_MAX_LENGTH], length;
            enum agx3_alu_hint hint = agx3_post_ra_hint(src);
            /* An in-place popcount consumes the old register value and then
             * overwrites it. Post-RA has no separate lifetime hint on this
             * tied operand, but that is exactly the witnessed discard form
             * (including Apple's U32 dest=src=0 program). */
            if (hint == AGX3_ALU_HINT_NONE && dest.value == src.value)
               hint = AGX3_ALU_HINT_DISCARD;
            if (!agx3_alu_popcount(enc, dest.value, src.value, !word, hint,
                                   dest.cache, wait, bytes, &length) ||
                !agx3_append(emission, bytes, length))
               AGX3_REFUSE(I);
            if (widened &&
                (!agx3_clr(enc, dest.value + 1, true, false, bytes,
                           &length) ||
                 !agx3_append(emission, bytes, length)))
               AGX3_REFUSE(I);
            continue;
         }

         /* ~(a | b) (truth table 0x1), a & ~b (0x2), ~a & b (0x4),
          * ~(a & b) (0x7), ~(a ^ b) (0x9), a | ~b (0xB) and ~a | b (0xD) are
          * their own two-register instructions; the one-sided forms take
          * their operands in order. */
         const unsigned bit_tt = I->truth_table & 0xfu;
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_BITOP &&
             (bit_tt == 0x1u || bit_tt == 0x2u || bit_tt == 0x4u ||
              bit_tt == 0x7u || bit_tt == 0x9u || bit_tt == 0xbu ||
              bit_tt == 0xdu) &&
             I->nr_dests == 1 && I->nr_srcs == 2) {
            const agx_index dest = I->dest[0];
            if (!agx3_vary_reg32(dest) || agx_channels(dest) != 1)
               AGX3_REFUSE(I);
            const bool swap = bit_tt == 0x4u || bit_tt == 0xdu;
            uint16_t src[2];
            enum agx3_alu_hint hints[2];
            for (unsigned i = 0; i < 2; ++i) {
               const agx_index x = I->src[swap ? 1 - i : i];
               if (!agx3_vary_reg32(x) || agx_channels(x) != 1 || x.abs ||
                   x.neg)
                  AGX3_REFUSE(I);
               src[i] = x.value;
               hints[i] = agx3_post_ra_hint(x);
            }
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], length;
            bool encoded;
            switch (bit_tt) {
            case 0x1u:
               encoded = agx3_alu_bit_nor(enc, dest.value, src, hints,
                                          dest.cache, wait, bytes, &length);
               break;
            case 0x2u:
            case 0x4u:
               encoded = agx3_alu_bit_andc(enc, dest.value, src, hints,
                                           dest.cache, wait, bytes, &length);
               break;
            case 0x7u:
               encoded = agx3_alu_bit_nand(enc, dest.value, src, hints,
                                           dest.cache, wait, bytes, &length);
               break;
            case 0x9u:
               encoded = agx3_alu_bit_xnor(enc, dest.value, src, hints,
                                           dest.cache, wait, bytes, &length);
               break;
            default:
               encoded = agx3_alu_bit_orc(enc, dest.value, src, hints,
                                          dest.cache, wait, bytes, &length);
               break;
            }
            if (!encoded || !agx3_append(emission, bytes, length))
               AGX3_REFUSE(I);
            continue;
         }

         /* Half negation is encoded as -src + -0.0h. This exact f16add.ep2
          * form keeps the source lifetime and producer wait in one native
          * instruction, including the matrix-inverse and faceforward paths. */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_FADD &&
             I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
             agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
             I->dest[0].size == AGX_SIZE_16 &&
             agx_channels(I->dest[0]) == 1) {
            const bool immediate_first =
               I->src[0].type == AGX_INDEX_IMMEDIATE;
            const agx_index imm = I->src[immediate_first ? 0 : 1];
            const agx_index reg = I->src[immediate_first ? 1 : 0];
            if (imm.type == AGX_INDEX_IMMEDIATE && imm.value == 0x80 &&
                !imm.memory && !imm.abs && !imm.neg &&
                agx3_src_is_register(reg) && !reg.memory &&
                reg.size == AGX_SIZE_16 && agx_channels(reg) == 1 &&
                !reg.abs && reg.neg && !reg.kill &&
                !(reg.cache && reg.discard)) {
               uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
               if (!agx3_alu_hadd_neg_negzero(
                      enc, I->dest[0].value, reg.value,
                      agx3_post_ra_hint(reg), I->dest[0].cache, wait,
                      bytes, &len) ||
                   !agx3_append(emission, bytes, len))
                  AGX3_REFUSE(I);
               continue;
            }
         }

         /* Apple keeps exact half constants in scalar halves through an f32
          * add.  This is the form used repeatedly by Hyprland's colour
          * management paths; widening the retained half first costs a live
          * full register and post-RA may legitimately preserve this native
          * mixed operation.  Addition is commutative, so canonicalize either
          * IR order to the native f32/f16 field order. */
         if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_FADD &&
             I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
             agx3_vary_reg32(I->dest[0]) &&
             agx_channels(I->dest[0]) == 1 &&
             !I->src[0].abs && !I->src[0].neg &&
             !I->src[1].abs && !I->src[1].neg &&
             ((agx3_mod_src_ok(I->src[0]) && agx3_half_reg(I->src[1])) ||
              (agx3_half_reg(I->src[0]) && agx3_mod_src_ok(I->src[1])))) {
            const bool half_first = agx3_half_reg(I->src[0]);
            const agx_index f32 = I->src[half_first ? 1 : 0];
            const agx_index f16 = I->src[half_first ? 0 : 1];
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
            if (f32.kill || f16.kill || (f32.cache && f32.discard) ||
                (f16.cache && f16.discard) ||
                !agx3_alu_fadd_mixed_half(
                   enc, I->dest[0].value, f32.value, f16.value,
                   agx3_post_ra_hint(f32), agx3_post_ra_hint(f16),
                   I->dest[0].cache, wait, bytes, &len) ||
                !agx3_append(emission, bytes, len))
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
            if (agx_compiler_debug & (AGX_DBG_SHADERS | AGX_DBG_REFUSE))
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
      if (!agx3_fixup_branch(emission, enc, *fixup)) {
         if (agx_compiler_debug & (AGX_DBG_SHADERS | AGX_DBG_REFUSE))
            fprintf(stderr,
                    "AGX3 branch fixup refused: at=%lld target=%lld last=%lld "
                    "skip_end=%u loop_exit=%u bn=%u\n",
                    (long long)fixup->offset,
                    (long long)fixup->block->offset,
                    (long long)fixup->block->last_offset, fixup->skip_to_end,
                    fixup->loop_exit, fixup->is_bn);
         goto fail;
      }
   }

   /* Separately compiled parts do not share a scoreboard ledger. Apple's
    * monolithic sampled fragment waits for TEX on its first FLTNORM; our
    * epilog cannot know that upstream slot, so finish only the still-live
    * async work at this boundary. Constant-colour parts and parts with real
    * in-part consumers add no instructions. */
   if (agx3_g17p_drain_tail() && !ctx->is_preamble) {
      if (!agx3_retire_slots_noop(emission, enc, 0x3f))
         goto fail;
      outstanding = 0;
   }

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
