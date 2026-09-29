/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * G18G (agx3d) code from the G17G (agx3c) packer.
 *
 * G18G executes the G17G instruction set under a third AGX3 encoding. The
 * field ids of Apple's llvm::AGX3 encoder tables are shared across encodings,
 * and on the 27 G17G/G18G pairs of Apple-compiled witness texts
 * (test/native/g18g) the two are one machine:
 *
 *  - every G17G instruction decodes, and all but the texture samples
 *    (1750 of 1755) encode under agx3d with the same field vector;
 *  - where Apple's G17G and G18G compilers chose the same field vector for an
 *    aligned instruction, composing the G17G vector with the agx3d table in
 *    Apple's form order reproduces Apple's G18G bytes (146 of 146 distinct);
 *  - scoreboard slots and waits are the same fields with the same meaning,
 *    and agx3d encodes every single-slot and two-slot wait agx3c can (tested
 *    on f32add, f32mul and f32fma).
 *
 * So the G17G packer's output is re-encoded instruction by instruction:
 * segment with the agx3c length rule, decode the field vector with
 * agx3_decode(), apply the G18G field rules below, and compose it with the
 * agx3d table. Rules, all measured on the witness texts:
 *
 *  - lddev: G18G sets f87 = 2 whenever f75 is nonzero and leaves it 0 when
 *    f75 is 0 (16 of 16 distinct G18G lddev shapes); G17G never sets it.
 *  - smp, from the five G17G/G18G sample pairs (a fragment sample, level-zero
 *    array samples, and array texel reads with uniform and register indices):
 *    G18G always sets f57 = 1 and f158 = 4; f56 = 2 where f77 is 0 and
 *    f19 = 4 where f77 is 2; and where G17G sets f69 it sets f62 instead.
 *    Mesa's tile reload has f62 = f69 = 0, a pair no witness shows; it is
 *    kept 0 on the reading that f69 folds into f62.
 *
 * Ioto resource programs (agx3_transcode_agx3d_ioto), measured on Apple's
 * G18G Ioto programs in the J873 native captures (every mode of sessions
 * 20260925-a, -b, -g1 and -g2, and compute-add):
 *
 *  - ldshdr: agx3d has no f23 (G17G sets 1); f21 = 2 on every API program
 *    (vertex, fragment, compute) and 0 on the driver's tile programs;
 *  - cndcall: the helper call keeps its target f16 = 27 and drops f15;
 *  - specTPR: agx3d adds f24, the launched shader's register allocation:
 *    the power of two at or above its 32-bit register count, at least 8
 *    (every fragment 8, vertex 8 or 16, the end-of-tile shader 16 for r12).
 *
 * Control flow, measured on the control-flow witnesses (probe
 * m5-control-flow-reference, sixteen kernels of ifs, elses, loops, breaks,
 * continues and switches that Apple cannot if-convert):
 *
 *  - Flag slots move up by one. Every G17G compare that writes slot 0 or 1
 *    (icmp f31, fcmp f35) writes slot 1 or 2 on G18G, and every exec-mask op
 *    that reads it (cndst/cndlt/cndcf f1) reads slot 1 or 2; G17G's
 *    always-true slot 6 (the loop-entry cndst and cndef f1 = 6) is G18G's
 *    slot 0 with the invert bit f0 flipped. No G18G compare ever writes slot
 *    0. With that rule 114 of the 118 aligned cndst/cndlt/cndcf/cndef/cndend
 *    reproduce Apple's G18G bytes -- the other four are cndsts whose invert
 *    Apple chose together with the opposite compare condition -- and so does
 *    every compare whose sources Apple chose alike.
 *  - Slots 2..5 have no G18G witness. The G17G packer emits every compare
 *    immediately before the exec-mask op that consumes it, so no predicate is
 *    live across another instruction and the slot is only a name; such a
 *    program folds them to slot 1. That is checked here, over the whole
 *    program, rather than assumed.
 *  - b/bn keep their field vector apart from the displacement f0: a 48-bit
 *    byte offset from the branch's own first byte on both encodings (35 of
 *    35 aligned branch targets land on the same instruction). Apple encodes
 *    every G18G branch in the ten-byte form (ep 5, all 39 branches), which is
 *    also the G17G branch length, so re-encoding the other instructions moves
 *    the targets but never a branch's own size: the displacements are
 *    recomputed once every agx3c offset has its agx3d offset.
 *
 * Refused (the shader then fails, it is never mis-encoded):
 *
 *  - the control flow no G17G/G18G pair shows: bl, brk, cndret, dbgrtn,
 *    yield, and cndcall outside Ioto programs;
 *  - a branch whose target is not an instruction boundary of the program;
 *  - smp with an f77 other than 0 or 2 (the only values the witnesses show).
 *
 * Not modelled: Apple's G18G vertex programs add a bit_andc/iself32/bit_and/
 * bit_or epilogue on float varyings that G17G does not. Its purpose is not
 * known, so it is not emitted; hardware will say whether it matters.
 */
#include "agx3_transcode.h"

#include <stdlib.h>
#include <string.h>

#include "util/macros.h"
#include "agx3_compose/agx3_compose.h"
#include "agx3_compose/agx3_decode.h"
#include "asahi/isa/disasm.h"

static const char *const refused_control_flow[] = {
   "bl", "brk", "cndcall", "cndret", "dbgrtn", "yield",
};

/* The G17G flag slot that always reads true, and G18G's always-false one. */
#define G17G_TRUE_SLOT  6
#define G18G_FALSE_SLOT 0

/* agx3d's ten-byte b/bn, the only branch form in Apple's G18G code. */
#define G18G_BRANCH_EP   5
#define G18G_BRANCH_SIZE 10

#define DISPLACEMENT_MASK ((UINT64_C(1) << 48) - 1)

static bool
is_control_flow(const char *mnemonic)
{
   for (unsigned i = 0; i < ARRAY_SIZE(refused_control_flow); ++i) {
      if (!strcmp(mnemonic, refused_control_flow[i]))
         return true;
   }
   return false;
}

/* The disassembler's display name, without its padding. */
static void
trim_display(const char *display, char *out, size_t size)
{
   size_t n = 0;
   while (display[n] && display[n] != ' ' && n + 1 < size) {
      out[n] = display[n];
      n++;
   }
   out[n] = '\0';
}

static unsigned
spectpr_allocation(unsigned gprs16)
{
   const unsigned regs = (gprs16 + 1) / 2;
   unsigned alloc = 8;
   while (alloc < regs)
      alloc *= 2;
   return alloc;
}

static bool
apply_ioto_rules(const char *mnemonic, uint64_t fields[AGX3_MAX_FIELDS],
                 const struct agx3_transcode_ioto *ioto, const char **why)
{
   if (!strcmp(mnemonic, "ldshdr")) {
      fields[23] = 0;
      fields[21] = ioto->api_program ? 2 : 0;
   } else if (!strcmp(mnemonic, "specTPR")) {
      if (!ioto->shader_gprs || ioto->shader_gprs > 256) {
         *why = "specTPR without a register count";
         return false;
      }
      fields[24] = spectpr_allocation(ioto->shader_gprs);
   } else if (!strcmp(mnemonic, "cndcall")) {
      /* Only the witnessed helper call: target 27. */
      if (fields[16] != 27 || fields[17] != 1 || fields[25] != 1) {
         *why = "cndcall other than the witnessed Ioto helper call";
         return false;
      }
      fields[15] = 0;
   }
   return true;
}

static bool
apply_g18g_rules(const char *mnemonic, uint64_t fields[AGX3_MAX_FIELDS],
                 const struct agx3_transcode_ioto *ioto, const char **why)
{
   const bool ioto_only = !strcmp(mnemonic, "ldshdr") ||
                          !strcmp(mnemonic, "specTPR") ||
                          !strcmp(mnemonic, "specLM") ||
                          !strcmp(mnemonic, "cndcall");
   if (ioto_only) {
      if (!ioto) {
         *why = "Ioto-only instruction in shader code";
         return false;
      }
      return apply_ioto_rules(mnemonic, fields, ioto, why);
   }
   if (is_control_flow(mnemonic)) {
      *why = "control flow (branch relocation is not grounded for G18G)";
      return false;
   }
   if (!strcmp(mnemonic, "smp")) {
      if (fields[77] != 0 && fields[77] != 2) {
         *why = "texture sample whose f77 no G18G witness shows";
         return false;
      }
      fields[57] = 1;
      fields[158] = 4;
      if (fields[77] == 0)
         fields[56] = 2;
      else
         fields[19] = 4;
      fields[62] |= fields[69];
      fields[69] = 0;
   }
   if (!strcmp(mnemonic, "lddev"))
      fields[87] = fields[75] ? 2 : 0;
   return true;
}

/* The field naming the flag slot a compare writes or an exec-mask op reads,
 * or -1. cndend names none. */
static int
flag_slot_field(const char *mnemonic)
{
   if (!strcmp(mnemonic, "icmp"))
      return 31;
   if (!strcmp(mnemonic, "fcmp"))
      return 35;
   if (!strcmp(mnemonic, "cndst") || !strcmp(mnemonic, "cndlt") ||
       !strcmp(mnemonic, "cndcf") || !strcmp(mnemonic, "cndef"))
      return 1;
   return -1;
}

static bool
is_compare(const char *mnemonic)
{
   return !strcmp(mnemonic, "icmp") || !strcmp(mnemonic, "fcmp");
}

/* Whole-program flag-slot state: which predicate the previous instruction
 * wrote, whether every predicate so far was consumed by the instruction right
 * after its compare, and whether a slot without a G18G witness was folded. */
struct flag_state {
   int pending;
   bool adjacent;
   bool folded;
};

static bool
apply_flag_slot_rule(const char *mnemonic, uint64_t fields[AGX3_MAX_FIELDS],
                     struct flag_state *state, const char **why)
{
   const int field = flag_slot_field(mnemonic);
   const bool compare = is_compare(mnemonic);
   const uint64_t slot = field >= 0 ? fields[field] : 0;
   const bool reads_flag = field >= 0 && !compare && slot != G17G_TRUE_SLOT;

   /* A predicate must be read by the instruction right after its compare,
    * and a read must come right after the compare, for the slot to be only
    * a name. */
   if (state->pending >= 0 && !(reads_flag && slot == (uint64_t)state->pending))
      state->adjacent = false;
   if (reads_flag && state->pending != (int)slot)
      state->adjacent = false;
   state->pending = compare ? (int)slot : -1;

   if (field < 0)
      return true;

   if (slot == G17G_TRUE_SLOT) {
      /* Only the unconditional loop-entry push and the else flip read the
       * always-true slot, in G17G and G18G code alike. */
      if (strcmp(mnemonic, "cndst") && strcmp(mnemonic, "cndef")) {
         *why = "always-true flag slot outside cndst/cndef";
         return false;
      }
      fields[1] = G18G_FALSE_SLOT;
      fields[0] ^= 1;
      return true;
   }
   if (slot > 5) {
      *why = "flag slot above 5";
      return false;
   }
   if (slot <= 1) {
      fields[field] = slot + 1;
   } else {
      fields[field] = 1;
      state->folded = true;
   }
   return true;
}

static int64_t
sign_extend_48(uint64_t v)
{
   return (int64_t)(v << 16) >> 16;
}

/* One re-encoded b/bn waiting for its displacement. */
struct pending_branch {
   unsigned out_offset;
   unsigned in_offset;
   int64_t in_target;
   int32_t form;
   uint64_t fields[AGX3_MAX_FIELDS];
};

/* agx3c offset -> agx3d offset of one instruction boundary. */
struct offset_pair {
   unsigned in, out;
};

static bool
relocate_branches(struct util_dynarray *out, const struct util_dynarray *map,
                  const struct util_dynarray *branches, const char **why,
                  unsigned *fail_offset)
{
   const struct offset_pair *pairs = map->data;
   const unsigned count = util_dynarray_num_elements(map, struct offset_pair);

   util_dynarray_foreach(branches, struct pending_branch, br) {
      *fail_offset = br->in_offset;
      /* The map is in ascending agx3c order; find the target boundary. */
      unsigned lo = 0, hi = count;
      while (lo < hi) {
         const unsigned mid = lo + (hi - lo) / 2;
         if ((int64_t)pairs[mid].in < br->in_target)
            lo = mid + 1;
         else
            hi = mid;
      }
      if (br->in_target < 0 || lo == count ||
          (int64_t)pairs[lo].in != br->in_target) {
         *why = "branch target is not an instruction boundary of the program";
         return false;
      }

      const int64_t displacement =
         (int64_t)pairs[lo].out - (int64_t)br->out_offset;
      uint64_t fields[AGX3_MAX_FIELDS];
      memcpy(fields, br->fields, sizeof(fields));
      fields[0] = (uint64_t)displacement & DISPLACEMENT_MASK;
      uint8_t bytes[2 * AGX3_MAX_WORDS];
      if (!agx3_compose(&agx3d_table, br->form, fields, bytes)) {
         *why = "branch displacement has no agx3d encoding";
         return false;
      }
      memcpy((uint8_t *)out->data + br->out_offset, bytes, G18G_BRANCH_SIZE);
   }
   return true;
}

static bool
transcode(struct util_dynarray *binary, unsigned start,
          const struct agx3_transcode_ioto *ioto, const char **why,
          unsigned *fail_offset)
{
   const char *unused_why;
   unsigned unused_offset;
   if (!why)
      why = &unused_why;
   if (!fail_offset)
      fail_offset = &unused_offset;
   *why = NULL;
   *fail_offset = 0;

   if (start > binary->size) {
      *why = "start past the end of the binary";
      return false;
   }

   const unsigned len = binary->size - start;
   const uint8_t *in = (const uint8_t *)binary->data + start;
   struct util_dynarray out, map, branches;
   util_dynarray_init(&out, NULL);
   util_dynarray_init(&map, NULL);
   util_dynarray_init(&branches, NULL);
   struct flag_state flags = {.pending = -1, .adjacent = true};

   unsigned i = 0;
   while (i < len) {
      const char *display = NULL;
      const signed n = agx3c_instr_length(in + i, len - i, &display);
      *fail_offset = i;
      if (n <= 0) {
         *why = "not an agx3c instruction";
         goto fail;
      }

      char mnemonic[32];
      trim_display(display, mnemonic, sizeof(mnemonic));

      uint64_t fields[AGX3_MAX_FIELDS];
      int32_t form =
         agx3_decode(&agx3c_table, in + i, (unsigned)n, mnemonic, fields);
      /* As the witness-vector generator does: the display name is only a
       * preference, some forms print under another name. */
      if (form < 0)
         form = agx3_decode(&agx3c_table, in + i, (unsigned)n, NULL, fields);
      if (form < 0) {
         *why = "agx3c bytes do not decode to a field vector";
         goto fail;
      }
      /* Encode by the table's own name for the decoded form. */
      const char *name = agx3c_table.forms[form].mnemonic;
      if (!apply_g18g_rules(name, fields, ioto, why) ||
          !apply_flag_slot_rule(name, fields, &flags, why))
         goto fail;

      util_dynarray_append(&map, ((struct offset_pair){i, out.size}));

      uint8_t bytes[2 * AGX3_MAX_WORDS];
      int32_t dform;
      const bool branch = !strcmp(name, "b") || !strcmp(name, "bn");
      if (branch) {
         /* Encode with a zero displacement now, the real one once every
          * target has its agx3d offset. */
         struct pending_branch br = {
            .out_offset = out.size,
            .in_offset = i,
            .in_target = (int64_t)i + sign_extend_48(fields[0]),
            .form = agx3_form_index(&agx3d_table, name, G18G_BRANCH_EP),
         };
         fields[0] = 0;
         memcpy(br.fields, fields, sizeof(br.fields));
         dform = br.form;
         if (dform < 0 || agx3d_table.forms[dform].size != G18G_BRANCH_SIZE ||
             !agx3_compose(&agx3d_table, dform, fields, bytes)) {
            *why = "branch has no ten-byte agx3d encoding";
            goto fail;
         }
         util_dynarray_append(&branches, br);
      } else {
         dform = agx3_compose_apple_order(&agx3d_table, name, fields, bytes);
         if (dform < 0) {
            *why = "field vector has no agx3d encoding";
            goto fail;
         }
      }
      const unsigned size = agx3d_table.forms[dform].size;
      if (agx3d_instr_length(bytes, size, NULL) != (signed)size) {
         *why = "agx3d encoding disagrees with the agx3d length rule";
         goto fail;
      }
      memcpy(util_dynarray_grow_bytes(&out, 1, size), bytes, size);
      i += n;
   }

   /* A branch may also land on the end of the program. */
   util_dynarray_append(&map, ((struct offset_pair){len, out.size}));
   *fail_offset = len;
   if (flags.pending >= 0)
      flags.adjacent = false;
   if (flags.folded && !flags.adjacent) {
      *why = "flag slot above 1 while a predicate lives across instructions";
      goto fail;
   }
   if (!relocate_branches(&out, &map, &branches, why, fail_offset))
      goto fail;

   binary->size = start;
   memcpy(util_dynarray_grow_bytes(binary, 1, out.size), out.data, out.size);
   util_dynarray_fini(&out);
   util_dynarray_fini(&map);
   util_dynarray_fini(&branches);
   return true;

fail:
   util_dynarray_fini(&out);
   util_dynarray_fini(&map);
   util_dynarray_fini(&branches);
   return false;
}

bool
agx3_transcode_agx3d(struct util_dynarray *binary, unsigned start,
                     const char **why, unsigned *fail_offset)
{
   return transcode(binary, start, NULL, why, fail_offset);
}

bool
agx3_transcode_agx3d_ioto(struct util_dynarray *binary, unsigned start,
                          const struct agx3_transcode_ioto *ioto,
                          const char **why, unsigned *fail_offset)
{
   return ioto && transcode(binary, start, ioto, why, fail_offset);
}
