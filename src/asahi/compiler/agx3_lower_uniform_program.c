/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "agx_builder.h"

/* AGCCodeGenerator.ShaderPhase.cprog_uniform_inputs controls the native
 * external-program prelude. Count live-ins, not every uniform source: a
 * previous export may already define the value. Backward liveness also
 * handles conditional exports and a read on the first loop iteration.
 * Native schema/renderer evidence: test/probes/agx3-uniform-inputs.md. */
bool
agx3_analyze_preamble_uniform_inputs(agx_context *ctx, bool *has_inputs)
{
   if (!has_inputs || !ctx->is_preamble || !ctx->num_blocks ||
       ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B)
      return false;

   struct uniform_block {
      uint64_t use[8], def[8], live[8];
      bool present;
   };
   struct uniform_block *blocks =
      calloc(ctx->num_blocks, sizeof(*blocks));
   if (!blocks)
      return false;

   unsigned count = 0;
   agx_foreach_block(ctx, block) {
      if (block->index >= ctx->num_blocks || blocks[block->index].present)
         goto fail;
      struct uniform_block *state = &blocks[block->index];
      state->present = true;
      ++count;
      agx_foreach_instr_in_block(block, I) {
         agx_foreach_src(I, s) {
            const agx_index src = I->src[s];
            if (src.type != AGX_INDEX_UNIFORM)
               continue;
            const unsigned size = agx_index_size_16(src);
            if (src.value >= 512 || size > 512 - src.value)
               goto fail;
            for (unsigned h = src.value; h < src.value + size; ++h)
               state->use[h / 64] |=
                  (UINT64_C(1) << (h % 64)) & ~state->def[h / 64];
         }

         unsigned base = 0, stride = 0, mask = 0;
         if (I->op == AGX_OPCODE_UNIFORM_STORE) {
            if (I->nr_srcs != 2 || I->src[1].type != AGX_INDEX_IMMEDIATE ||
                I->src[1].abs || I->src[1].neg || I->mask > 15)
               goto fail;
            base = I->src[1].value;
            stride = agx_size_align_16(I->src[0].size);
            mask = I->mask;
         } else if (I->op == AGX_OPCODE_UNIFORM_FADD_AGX3 ||
                    I->op == AGX_OPCODE_UNIFORM_FMUL_AGX3 ||
                    I->op == AGX_OPCODE_UNIFORM_FFMA_AGX3) {
            if (I->imm > 510)
               goto fail;
            base = I->imm;
            stride = 2;
            mask = 1;
         }
         for (unsigned c = 0; c < 4; ++c) {
            if (!(mask & (1u << c)))
               continue;
            if (base >= 512 || (c + 1) * stride > 512 - base)
               goto fail;
            for (unsigned h = base + c * stride;
                 h < base + (c + 1) * stride; ++h)
               state->def[h / 64] |= UINT64_C(1) << (h % 64);
         }
      }
      memcpy(state->live, state->use, sizeof(state->live));
      for (unsigned s = 0; s < 2; ++s) {
         if (block->successors[s] &&
             block->successors[s]->index >= ctx->num_blocks)
            goto fail;
      }
   }
   if (count != ctx->num_blocks)
      goto fail;

   bool progress;
   do {
      progress = false;
      agx_foreach_block_rev(ctx, block) {
         struct uniform_block *state = &blocks[block->index];
         for (unsigned s = 0; s < 2; ++s) {
            const agx_block *succ = block->successors[s];
            if (!succ)
               continue;
            for (unsigned w = 0; w < 8; ++w) {
               uint64_t live = state->live[w] |
                  (blocks[succ->index].live[w] & ~state->def[w]);
               progress |= live != state->live[w];
               state->live[w] = live;
            }
         }
      }
   } while (progress);

   bool inputs = false;
   for (unsigned w = 0; w < 8; ++w)
      inputs |= blocks[agx_start_block(ctx)->index].live[w] != 0;
   if (ctx->out && !agx3_cprog_inputs_init(
          blocks[agx_start_block(ctx)->index].live,
          &ctx->out->agx3_preamble_inputs))
      goto fail;
   free(blocks);
   *has_inputs = inputs;
   return true;

fail:
   free(blocks);
   return false;
}

/* Native scalar-half witnesses export an aligned dword, with an unused upper
 * uniform half. Give RA that real dword operand rather than widening an
 * arbitrary post-RA half-register (which may be odd or have a live neighbour).
 * Existing half-pair coalescing runs first; this handles only proven padding.
 */
bool
agx3_lower_uniform_half_exports(agx_context *ctx)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B || !ctx->is_preamble)
      return false;

   bool occupied[512] = {false};
   if (ctx->agx3_cprog_inputs_lowered && ctx->out) {
      for (unsigned h = 0; h < 512; ++h)
         occupied[h] = !!(ctx->out->agx3_preamble_inputs.halves[h / 64] &
                           (UINT64_C(1) << (h % 64)));
   }
   for (unsigned h = 0; h < MIN2(ctx->key->reserved_preamble, 512); ++h)
      occupied[h] = true;
   if (ctx->out && ctx->out->rodata.size_16) {
      const unsigned base = ctx->out->rodata.base_uniform;
      const unsigned size = ctx->out->rodata.size_16;
      if (base >= 512 || size > 512 - base)
         return false;
      for (unsigned h = 0; h < size; ++h)
         occupied[base + h] = true;
   }

   agx_foreach_instr_global(ctx, I) {
      agx_foreach_src(I, s) {
         const agx_index src = I->src[s];
         if (src.type != AGX_INDEX_UNIFORM)
            continue;
         const unsigned count = agx_channels(src) * agx_size_align_16(src.size);
         if (src.value >= 512 || count > 512 - src.value)
            return false;
         for (unsigned h = 0; h < count; ++h)
            occupied[src.value + h] = true;
      }

      if (I->op == AGX_OPCODE_UNIFORM_FADD_AGX3 ||
          I->op == AGX_OPCODE_UNIFORM_FMUL_AGX3 ||
          I->op == AGX_OPCODE_UNIFORM_FFMA_AGX3) {
         if (I->imm > 510)
            return false;
         occupied[I->imm] = occupied[I->imm + 1] = true;
      } else if (I->op == AGX_OPCODE_UNIFORM_STORE) {
         if (I->nr_srcs != 2 || I->src[1].type != AGX_INDEX_IMMEDIATE ||
             I->src[1].abs || I->src[1].neg)
            return false;
         const unsigned base = I->src[1].value;
         const unsigned stride = agx_size_align_16(I->src[0].size);
         for (unsigned c = 0; c < 4; ++c) {
            if (!(I->mask & BITFIELD_BIT(c)))
               continue;
            if (base >= 512 || (c + 1) * stride > 512 - base)
               return false;
            for (unsigned h = 0; h < stride; ++h)
               occupied[base + c * stride + h] = true;
         }
      }
   }

   bool progress = false;
   agx_foreach_instr_global_safe(ctx, I) {
      if (I->op != AGX_OPCODE_UNIFORM_STORE || I->nr_srcs != 2 || I->mask != 1)
         continue;
      const agx_index value = I->src[0];
      const unsigned base = I->src[1].value;
      if (value.type != AGX_INDEX_NORMAL || value.size != AGX_SIZE_16 ||
          agx_channels(value) != 1 || value.memory || value.abs || value.neg ||
          (base & 1) || base > 510 || occupied[base + 1])
         continue;

      agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
      /* Native scalar-half programs clear the upper half before COPY. */
      agx_index word = agx_temp(ctx, AGX_SIZE_32);
      agx_instr *collect = agx_collect_to(&b, word, 2);
      collect->src[0] = value;
      collect->src[1] = agx_immediate(0);
      I->src[0] = word;
      if (ctx->out)
         ctx->out->push_count = MAX2(ctx->out->push_count, base + 2);
      progress = true;
   }
   return progress;
}

/* Select the constant program's native destination domain before scheduling
 * and RA. Only fold a same-block, single-use scalar value into its uniform
 * export. Native AIR-NT add/mul/fma witnesses ground plain f32 inputs and
 * low-file aligned outputs. Texture/sampler state stores are unrelated.
 */
bool
agx3_lower_uniform_program(agx_context *ctx)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B ||
       !ctx->is_preamble || !ctx->alloc)
      return false;

   agx_instr **defs = rzalloc_array(ctx, agx_instr *, ctx->alloc);
   agx_block **blocks = rzalloc_array(ctx, agx_block *, ctx->alloc);
   unsigned *uses = rzalloc_array(ctx, unsigned, ctx->alloc);
   if (!defs || !blocks || !uses) {
      ralloc_free(defs);
      ralloc_free(blocks);
      ralloc_free(uses);
      return false;
   }
   agx_foreach_block(ctx, block) {
      agx_foreach_instr_in_block(block, I) {
         agx_foreach_ssa_dest(I, d) {
            defs[I->dest[d].value] = I;
            blocks[I->dest[d].value] = block;
         }
         agx_foreach_ssa_src(I, s)
            uses[I->src[s].value]++;
      }
   }

   bool progress = false;
   agx_foreach_block(ctx, block) {
      agx_foreach_instr_in_block_safe(block, I) {
         if (I->op != AGX_OPCODE_UNIFORM_STORE || I->nr_srcs != 2 ||
             (I->mask != 1 && I->mask != 3 && I->mask != 7 && I->mask != 15))
            continue;
         const unsigned count = util_bitcount(I->mask);
         const agx_index value = I->src[0], index = I->src[1];
         if (value.type != AGX_INDEX_NORMAL || value.size != AGX_SIZE_32 ||
             agx_channels(value) != count || value.memory || value.abs || value.neg ||
             index.type != AGX_INDEX_IMMEDIATE || index.value > 128 - 2 * count ||
             (index.value & 1) || index.abs || index.neg ||
             uses[value.value] != 1 || blocks[value.value] != block)
            continue;

         agx_instr *collect = defs[value.value];
         if (count > 1 && (collect->op != AGX_OPCODE_COLLECT ||
                           collect->nr_srcs != count))
            continue;
         agx_instr *alus[4];
         enum agx_opcode ops[4];
         bool admitted = true;
         for (unsigned c = 0; c < count && admitted; ++c) {
            const agx_index scalar = count == 1 ? value : collect->src[c];
            if (scalar.type != AGX_INDEX_NORMAL || scalar.size != AGX_SIZE_32 ||
                agx_channels(scalar) != 1 || scalar.memory || scalar.abs ||
                scalar.neg || uses[scalar.value] != 1 ||
                blocks[scalar.value] != block) {
               admitted = false;
               break;
            }
            agx_instr *alu = alus[c] = defs[scalar.value];
            if (alu->op == AGX_OPCODE_FADD)
               ops[c] = AGX_OPCODE_UNIFORM_FADD_AGX3;
            else if (alu->op == AGX_OPCODE_FMUL)
               ops[c] = AGX_OPCODE_UNIFORM_FMUL_AGX3;
            else if (alu->op == AGX_OPCODE_FFMA)
               ops[c] = AGX_OPCODE_UNIFORM_FFMA_AGX3;
            else {
               admitted = false;
               break;
            }
            admitted &= !alu->saturate && alu->nr_dests == 1 &&
               alu->nr_srcs == (alu->op == AGX_OPCODE_FFMA ? 3 : 2);
            bool has_register = false;
            agx_foreach_src(alu, s) {
               const agx_index src = alu->src[s];
               const bool reg = src.type == AGX_INDEX_NORMAL ||
                                src.type == AGX_INDEX_REGISTER;
               const bool immediate = src.type == AGX_INDEX_IMMEDIATE;
               admitted &= !src.abs && !src.neg && !src.memory &&
                  ((reg && src.size == AGX_SIZE_32 && agx_channels(src) == 1) ||
                   (immediate && src.value <= 255));
               has_register |= reg;
            }
            admitted &= has_register;
         }
         if (!admitted)
            continue;

         agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
         for (unsigned c = 0; c < count; ++c) {
            agx_instr *native = agx_alloc_instr(&b, ops[c], 0, alus[c]->nr_srcs);
            agx_foreach_src(alus[c], s)
               native->src[s] = alus[c]->src[s];
            native->imm = index.value + 2 * c;
            agx_builder_insert(&b.cursor, native);
         }
         agx_remove_instruction(I);
         /* DCE removes the now-unused arithmetic result; other exports and
          * shared computations retain their original GPR representation. */
         progress = true;
      }
   }
   ralloc_free(defs);
   ralloc_free(blocks);
   ralloc_free(uses);
   return progress;
}
