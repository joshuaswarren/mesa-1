/*
 * Copyright 2023 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

#include <stdlib.h>
#include <string.h>
#include "util/bitset.h"
#include "util/hash_table.h"
#include "util/ralloc.h"
#include "agx_compiler.h"
#include "agx_debug.h"
#include "agx3_uniform_integer_instr.h"
#include "util/u_debug.h"
#include "agx_opcodes.h"

/*
 * Information about a constant, indexed by its 64-bit value. This describes the
 * value, not the move that generated it. If there are multiple moves in the
 * shader with the same immediate value, they resolve to the same constant.
 */
struct constant_info {
   uint64_t value;

   /* Number of uses of the constant that could be promoted */
   unsigned nr_promotable_uses;

   /* If we push, the uniform used */
   uint16_t uniform;

   /* Alignment in 16-bit units needed for the constant */
   uint8_t align_16;

   /* True if the constant was promoted to a uniform */
   bool promoted;
};

/*
 * Choosing constants to promote is similar to the 0-1 knapsack problem. We use
 * a well-known heuristic: sort by benefit divided by size. We approximate
 * benefit by use count.
 */
static int
constant_priority(const struct constant_info *const info)
{
   int size = info->align_16;
   assert(size == 1 || size == 2 || size == 4);
   int inverse_size = (size == 1) ? 4 : (size == 2) ? 2 : 1;

   return info->nr_promotable_uses * inverse_size;
}

static int
priority_compare(const void *A_, const void *B_)
{
   const struct constant_info *const *A = A_;
   const struct constant_info *const *B = B_;

   /* This is backwards from qsort's documentation, because we want descending
    * order and qsort returns ascending.
    */
   return constant_priority(*B) - constant_priority(*A);
}

static void
record_use(void *memctx, struct hash_table_u64 *constants, uint64_t imm,
           enum agx_size size)
{
   struct constant_info *info = _mesa_hash_table_u64_search(constants, imm);

   if (!info) {
      info = rzalloc(memctx, struct constant_info);
      info->value = imm;
      _mesa_hash_table_u64_insert(constants, imm, info);
   }

   info->nr_promotable_uses++;
   info->align_16 = MAX2(info->align_16, agx_size_align_16(size));
}

/* Reuse the packer's full instruction-domain proof for AGX3 integer operands.
 * Do not broaden the opcode-only predicate: modifiers, operand position and
 * the opposite source's type are part of the native instruction contract.
 * Private qualification switch; shader cache must be disabled in both arms.
 */
static bool
constant_use_accepts_uniform(agx_context *ctx, const agx_instr *I, unsigned s,
                             unsigned uniform, enum agx_size size)
{
   if (agx_instr_accepts_uniform(ctx->key->dev.encoding, I->op, s, uniform, size))
      return true;

   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B || I->nr_srcs > 3 ||
       uniform >= AGX_NUM_UNIFORMS)
      return false;

   agx_instr candidate = *I;
   agx_index sources[3];
   memcpy(sources, I->src, I->nr_srcs * sizeof(*sources));
   candidate.src = sources;
   sources[s] = agx_replace_index(sources[s], agx_uniform(uniform, size));
   struct agx3_uniform_integer_desc desc;
   return agx3_uniform_integer_instruction(&candidate, false, &desc);
}

static void
pass(agx_context *ctx, void *memctx)
{
   /* Map from SSA indices to struct constant_info */
   struct hash_table_u64 *constants = _mesa_hash_table_u64_create(memctx);

   /* Map from SSA indices to immediate values */
   uint64_t *values = rzalloc_array(memctx, uint64_t, ctx->alloc);

   /* Set of SSA indices that map to immediate values */
   BITSET_WORD *is_immediate = BITSET_RZALLOC(memctx, ctx->alloc);

   /* Probe uniform acceptance where the first promoted constant will land:
    * the range starts 32-bit aligned on AGX3, and aligned f32 operands
    * (FFMA, FADD, FMUL, FCMPSEL) refuse an odd index. Probing the raw count
    * after a preamble that ends on a 16-bit export refused every such use.
    */
   const unsigned probe_uniform =
      ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX2
         ? ALIGN_POT(ctx->out->push_count, 2)
         : ctx->out->push_count;
   const unsigned push_before = ctx->out->push_count;

   /* Gather constant definitions and use */
   agx_foreach_instr_global(ctx, I) {
      if (I->op == AGX_OPCODE_MOV_IMM) {
         assert(I->dest[0].type == AGX_INDEX_NORMAL);
         BITSET_SET(is_immediate, I->dest[0].value);
         values[I->dest[0].value] = I->imm;
      } else {
         agx_foreach_ssa_src(I, s) {
            if (BITSET_TEST(is_immediate, I->src[s].value) &&
                constant_use_accepts_uniform(ctx, I, s, probe_uniform,
                                             I->src[s].size)) {

               record_use(memctx, constants, values[I->src[s].value],
                          I->src[s].size);
            }
         }
      }
   }

   /* Early exit if there were no constants */
   unsigned nr_nodes = _mesa_hash_table_u64_num_entries(constants);
   if (nr_nodes == 0)
      return;

   /* Collect nodes that are promotable */
   struct constant_info **flat =
      rzalloc_array(memctx, struct constant_info *, nr_nodes);

   unsigned flat_count = 0;
   hash_table_u64_foreach(constants, entry) {
      flat[flat_count++] = entry.data;
   }

   /* Select constants. Even when we can promote everything, sorting keeps hot
    * constants in lower uniforms, required by some instructions.
    */
   qsort(flat, flat_count, sizeof(*flat), priority_compare);

   /* Promote as many constants as we can */
   for (unsigned i = 0; i < flat_count; ++i) {
      struct constant_info *info = flat[i];
      assert(info->nr_promotable_uses > 0);

      /* Try to assign a uniform */
      /* Native Ioto publishes 32-bit uniform words. Align the beginning of
       * the constant range even when its first constant is only 16 bits. */
      unsigned range_align = ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX2 &&
                             i == 0 ? MAX2(info->align_16, 2) : info->align_16;
      unsigned uniform = ALIGN_POT(ctx->out->push_count, range_align);
      unsigned new_count = uniform + info->align_16;
      if (new_count > AGX_NUM_UNIFORMS)
         break;

      /* Set the base for the first constant. This avoids uploading an extra
       * zero if we needed to pad, compared to doing this before the loop. Same
       * shader code, but slightly more efficient USC record.
       */
      if (i == 0) {
         ctx->out->rodata.base_uniform = uniform;
      }

      info->uniform = uniform;
      info->promoted = true;
      ctx->out->push_count = new_count;

      unsigned size_B = info->align_16 * 2;
      unsigned uniform_offs = uniform - ctx->out->rodata.base_uniform;
      assert(uniform_offs < ARRAY_SIZE(ctx->rodata));
      memcpy(&ctx->rodata[uniform_offs], &info->value, size_B);

      ctx->out->rodata.size_16 = new_count - ctx->out->rodata.base_uniform;
   }

   /* Keep the trailing half inside the owned constant blob and uniform
    * allocation; the native uploader cannot load half of a 32-bit word. */
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX2 &&
       (ctx->out->rodata.size_16 & 1)) {
      ctx->rodata[ctx->out->rodata.size_16++] = 0;
      ctx->out->push_count++;
   }

   /* An inline-blend recompile must leave its constants where the split
    * main program has them (see agx_shader_key::g17p_rodata_hint). Keep the
    * ordinary placement whenever it already does; otherwise take each
    * constant's slot from the main's table. */
   const uint16_t *hint = ctx->key->g17p_rodata_hint;
   const unsigned hint_base = ctx->key->g17p_rodata_hint_base;
   const unsigned hint_size = ctx->key->g17p_rodata_hint_size_16;
   if (hint && hint_size && ctx->out->rodata.size_16 &&
       !(ctx->out->rodata.base_uniform == hint_base &&
         ctx->out->rodata.size_16 <= hint_size &&
         !memcmp(ctx->rodata, hint, ctx->out->rodata.size_16 * 2)) &&
       push_before <= hint_base) {
      unsigned end = 0;
      for (unsigned i = 0; i < flat_count; ++i) {
         struct constant_info *info = flat[i];
         info->promoted = false;
         const unsigned align = info->align_16;
         for (unsigned k = 0; k + align <= hint_size; ++k) {
            if ((hint_base + k) % align ||
                memcmp(&hint[k], &info->value, align * 2))
               continue;
            info->uniform = hint_base + k;
            info->promoted = true;
            end = MAX2(end, k + align);
            break;
         }
      }

      /* Whole 32-bit words, as above, taken from the main's table. */
      if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX2 && (end & 1))
         end++;
      assert(end <= hint_size && end <= ARRAY_SIZE(ctx->rodata));
      memcpy(ctx->rodata, hint, end * 2);
      ctx->out->rodata.base_uniform = end ? hint_base : 0;
      ctx->out->rodata.size_16 = end;
      ctx->out->push_count = end ? hint_base + end : push_before;
   }

   /* Promote in the IR */
   agx_foreach_instr_global(ctx, I) {
      agx_foreach_ssa_src(I, s) {
         if (!BITSET_TEST(is_immediate, I->src[s].value))
            continue;

         struct constant_info *info =
            _mesa_hash_table_u64_search(constants, values[I->src[s].value]);

         if (info && info->promoted &&
             constant_use_accepts_uniform(ctx, I, s,
                                       info->uniform, I->src[s].size)) {

            agx_replace_src(I, s, agx_uniform(info->uniform, I->src[s].size));
         }
      }
   }
}

void
agx_opt_promote_constants(agx_context *ctx)
{
   /* We do not promote constants in preambles since it's pointless and wastes
    * uniform slots.
    */
   if (ctx->is_preamble)
      return;

   void *memctx = ralloc_context(NULL);
   pass(ctx, memctx);
   ralloc_free(memctx);
}
