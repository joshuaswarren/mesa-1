/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "agx_builder.h"

static bool
mark(uint64_t *mask, unsigned first, unsigned count)
{
   if (!count || first >= 512 || count > 512 - first)
      return false;
   for (unsigned h = first; h < first + count; ++h)
      mask[h / 64] |= UINT64_C(1) << (h % 64);
   return true;
}

static agx_cursor
import_cursor(agx_block *block, agx_instr *I, unsigned source)
{
   if (I->op == AGX_OPCODE_PHI) {
      agx_block **preds = util_dynarray_begin(&block->predecessors);
      return agx_after_block_logical(preds[source]);
   }
   /* Keep the terminal export run contiguous. */
   if (I->op == AGX_OPCODE_EXPORT) {
      while (I->link.prev != &block->instructions) {
         agx_instr *prev = list_entry(I->link.prev, agx_instr, link);
         if (prev->op != AGX_OPCODE_EXPORT)
            break;
         I = prev;
      }
   }
   return agx_before_instr(I);
}

bool
agx3_lower_cprog_inputs(agx_context *ctx)
{
   if (!ctx->is_preamble || !ctx->out || ctx->agx3_cprog_inputs_lowered ||
       ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B)
      return false;

   uint64_t reads[8] = {0}, writes[8] = {0};
   agx_foreach_block(ctx, block) {
      agx_foreach_instr_in_block(block, I) {
         if (I->op == AGX_OPCODE_PHI &&
             I->nr_srcs != agx_num_predecessors(block))
            return false;
         agx_foreach_src(I, s) {
            agx_index src = I->src[s];
            if (src.type == AGX_INDEX_UNIFORM &&
                (src.memory || !mark(reads, src.value, agx_index_size_16(src))))
               return false;
         }
         if (I->op == AGX_OPCODE_UNIFORM_STORE) {
            if (I->nr_srcs != 2 || I->src[1].type != AGX_INDEX_IMMEDIATE ||
                I->src[1].abs || I->src[1].neg || (I->mask & ~15u))
               return false;
            unsigned stride = agx_size_align_16(I->src[0].size);
            for (unsigned c = 0; c < 4; ++c) {
               if ((I->mask & (1u << c)) &&
                   !mark(writes, I->src[1].value + c * stride, stride))
                  return false;
            }
         } else if (I->op == AGX_OPCODE_UNIFORM_FADD_AGX3 ||
                    I->op == AGX_OPCODE_UNIFORM_FMUL_AGX3 ||
                    I->op == AGX_OPCODE_UNIFORM_FFMA_AGX3) {
            if (I->imm >= 512 || !mark(writes, I->imm, 2))
               return false;
         }
      }
   }
   /* Extracted NIR keeps intermediate values in SSA. Its imported uniform
    * arguments are immutable and disjoint from exported uniform slots.
    * Refuse noncanonical physical-uniform read/modify/write instead of
    * replacing an updated value with an invocation-entry snapshot. */
   for (unsigned w = 0; w < 8; ++w) {
      if (reads[w] & writes[w])
         return false;
   }
   struct agx3_cprog_inputs inputs;
   if (!agx3_cprog_inputs_init(reads, &inputs))
      return false;
   if (!inputs.count)
      return true;

   const unsigned prefix = inputs.storage_bytes;
   if (ctx->scratch_size_B > 65536u - prefix)
      return false;
   /* Validate all existing scratch rebases before mutating the IR. Spiller
    * allocations subsequently start after both this prefix and local data. */
   agx_foreach_instr_global(ctx, I) {
      if (I->op != AGX_OPCODE_STACK_LOAD && I->op != AGX_OPCODE_STACK_STORE)
         continue;
      unsigned s = I->op == AGX_OPCODE_STACK_STORE ? 1 : 0;
      if (I->nr_srcs <= s)
         return false;
      agx_index offset = I->src[s];
      if (offset.memory || offset.neg || offset.cache || offset.discard ||
          offset.kill || agx_channels(offset) != 1)
         return false;
      if (offset.type == AGX_INDEX_IMMEDIATE) {
         if (offset.size != AGX_SIZE_16 || offset.value > UINT16_MAX - prefix)
            return false;
      } else if ((offset.type != AGX_INDEX_NORMAL &&
                  offset.type != AGX_INDEX_REGISTER &&
                  offset.type != AGX_INDEX_UNIFORM) ||
                 offset.size != AGX_SIZE_32 || offset.abs) {
         return false;
      }
   }

   ctx->out->agx3_preamble_inputs = inputs;
   ctx->out->agx3_preamble_uniform_inputs = true;
   ctx->out->agx3_preamble_tps_inputs = true;
   ctx->agx3_cprog_inputs_lowered = true;
   ctx->scratch_size_B += prefix;

   agx_foreach_block(ctx, block) {
      agx_foreach_instr_in_block_safe(block, I) {
         agx_foreach_src(I, s) {
            agx_index src = I->src[s];
            if (src.type != AGX_INDEX_UNIFORM)
               continue;
            agx_builder b = agx_init_builder(ctx, import_cursor(block, I, s));
            uint16_t tps;
            bool found = agx3_cprog_input_tps_half(&inputs, src.value, &tps);
            assert(found);
            (void)found;
            /* A compact word can begin at an odd half after a sparse import.
             * Use two real half loads in that case, with independent SSA
             * results/dependencies, then let RA coalesce the collection. */
            unsigned piece = src.size != AGX_SIZE_16 && !(tps & 1) ? 2 : 1;
            unsigned count = agx_index_size_16(src) / piece;
            agx_index value = agx_vec_temp(ctx, src.size, agx_channels(src));
            agx_index parts[32];
            assert(count <= ARRAY_SIZE(parts));
            for (unsigned p = 0; p < count; ++p) {
               parts[p] = count == 1 ? value :
                  agx_temp(ctx, piece == 2 ? AGX_SIZE_32 : AGX_SIZE_16);
               agx_instr *load = agx_alloc_instr(&b, AGX_OPCODE_LOAD_CPROG_AGX3, 1, 0);
               load->dest[0] = parts[p];
               load->imm = src.value + p * piece;
               agx_builder_insert(&b.cursor, load);
            }
            if (count > 1) {
               agx_instr *collect = agx_collect_to(&b, value, count);
               for (unsigned p = 0; p < count; ++p)
                  collect->src[p] = parts[p];
            }
            agx_replace_src(I, s, value);
         }
      }
   }

   agx_foreach_instr_global_safe(ctx, I) {
      if (I->op != AGX_OPCODE_STACK_LOAD && I->op != AGX_OPCODE_STACK_STORE)
         continue;
      unsigned s = I->op == AGX_OPCODE_STACK_STORE ? 1 : 0;
      agx_index offset = I->src[s];
      if (offset.type == AGX_INDEX_IMMEDIATE) {
         I->src[s] = agx_immediate(offset.value + prefix);
      } else {
         agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
         I->src[s] = agx_iadd(&b, offset, agx_mov_imm(&b, 32, prefix), 0);
      }
   }
   return true;
}
