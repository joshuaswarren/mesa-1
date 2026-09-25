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
 * Refused (the shader then fails, it is never mis-encoded):
 *
 *  - control flow: every PC-relative or predicate-stack mnemonic. The
 *    witnesses contain no branch at all, so relocation after a length change
 *    is ungrounded;
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
   "b",      "bl",     "bn",     "brk",    "cndcall", "cndcf", "cndef",
   "cndend", "cndlt",  "cndret", "cndst",  "dbgrtn",  "yield",
};

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
   struct util_dynarray out;
   util_dynarray_init(&out, NULL);

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
      if (!apply_g18g_rules(name, fields, ioto, why))
         goto fail;

      uint8_t bytes[2 * AGX3_MAX_WORDS];
      const int32_t dform =
         agx3_compose_apple_order(&agx3d_table, name, fields, bytes);
      if (dform < 0) {
         *why = "field vector has no agx3d encoding";
         goto fail;
      }
      const unsigned size = agx3d_table.forms[dform].size;
      if (agx3d_instr_length(bytes, size, NULL) != (signed)size) {
         *why = "agx3d encoding disagrees with the agx3d length rule";
         goto fail;
      }
      memcpy(util_dynarray_grow_bytes(&out, 1, size), bytes, size);
      i += n;
   }

   binary->size = start;
   memcpy(util_dynarray_grow_bytes(binary, 1, out.size), out.data, out.size);
   util_dynarray_fini(&out);
   return true;

fail:
   util_dynarray_fini(&out);
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
