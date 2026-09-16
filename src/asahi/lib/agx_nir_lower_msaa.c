/*
 * Copyright 2023 Alyssa Rosenzweig
 * Copyright 2021 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#include "agx_abi.h"
#include "agx_compile.h"
#include "agx_tilebuffer.h"
#include "nir.h"
#include "nir_builder.h"
#include "nir_builder_opcodes.h"
#include "nir_intrinsics.h"

struct lower_per_sample_ctx {
   bool force_early_pending;
};

/* Explicit sample interpolation is per-pixel, but its single-sample clamp
 * still depends on rasterizationSamples. It needs native specialization when
 * compiled without that state, just as a sample-shaded fragment does. */
bool
agx_nir_needs_sample_count(const nir_shader *shader)
{
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;
            if (op == nir_intrinsic_load_barycentric_at_sample ||
                op == nir_intrinsic_load_samples_log2_agx)
               return true;
         }
      }
   }
   return false;
}

static bool
specialize_sample_count(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_samples_log2_agx)
      return false;

   unsigned samples = *(const unsigned *)data;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def_replace(&intr->def,
                   nir_imm_intN_t(b, util_logbase2(samples), intr->def.bit_size));
   return true;
}

bool
agx_nir_specialize_sample_count(nir_shader *shader, unsigned samples)
{
   assert(samples == 1 || samples == 2 || samples == 4);
   return nir_shader_intrinsics_pass(shader, specialize_sample_count,
                                     nir_metadata_control_flow, &samples);
}

static bool
lower_to_per_sample(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct lower_per_sample_ctx *ctx = data;
   b->cursor = nir_before_instr(&intr->instr);

   switch (intr->intrinsic) {
   case nir_intrinsic_load_sample_id: {
      nir_def *mask = nir_u2u32(b, nir_load_active_samples_agx(b));
      nir_def *bit = nir_ufind_msb(b, mask);
      nir_def_replace(&intr->def, nir_u2uN(b, bit, intr->def.bit_size));
      return true;
   }

   case nir_intrinsic_load_local_pixel_agx:
   case nir_intrinsic_store_local_pixel_agx:
   case nir_intrinsic_store_zs_agx:
   case nir_intrinsic_demote_samples:
   case nir_intrinsic_sample_mask_agx: {
      /* Explicit early tests are a per-pixel operation even when the shader
       * body is sample shaded.  Keep the all/all update inserted at the
       * function entry intact; the G17P static unroller hoists it ahead of
       * all sample copies below. */
      if (intr->intrinsic == nir_intrinsic_sample_mask_agx &&
          ctx->force_early_pending) {
         ctx->force_early_pending = false;
         return false;
      }

      /* Fragment I/O inside the loop should only affect active samples. */
      unsigned mask_index =
         (intr->intrinsic == nir_intrinsic_store_local_pixel_agx) ? 1 : 0;

      nir_def *mask = intr->src[mask_index].ssa;
      nir_def *id_mask = nir_load_active_samples_agx(b);
      nir_def *converted = nir_u2uN(b, id_mask, mask->bit_size);

      nir_src_rewrite(&intr->src[mask_index], nir_iand(b, mask, converted));
      return true;
   }

   default:
      return false;
   }
}

bool
agx_nir_lower_to_per_sample(nir_shader *shader)
{
   struct lower_per_sample_ctx ctx = {
      .force_early_pending = shader->info.fs.early_fragment_tests,
   };
   return nir_shader_intrinsics_pass(shader, lower_to_per_sample,
                                     nir_metadata_control_flow, &ctx);
}

static bool
lower_active_samples(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_active_samples_agx)
      return false;

   b->cursor = nir_instr_remove(&intr->instr);
   nir_def_rewrite_uses(&intr->def, data);
   return true;
}

/*
 * In a monolithic pixel shader, we wrap the fragment shader in a loop over
 * each sample, and then let optimizations (like loop unrolling) go to town.
 * This lowering is not compatible with fragment epilogues, which require
 * something similar at the binary level since the NIR is long gone by then.
 */
static bool
agx_nir_wrap_per_sample_loop(nir_shader *shader, uint8_t nr_samples)
{
   assert(nr_samples > 1);

   /* Get the original function */
   nir_function_impl *impl = nir_shader_get_entrypoint(shader);

   nir_cf_list list;
   nir_cf_extract(&list, nir_before_impl(impl), nir_after_impl(impl));

   /* Create a builder for the wrapped function */
   nir_builder b = nir_builder_at(nir_after_block(nir_start_block(impl)));

   nir_variable *i =
      nir_local_variable_create(impl, glsl_uintN_t_type(16), NULL);
   nir_store_var(&b, i, nir_imm_intN_t(&b, 1, 16), ~0);
   nir_def *bit = NULL;
   nir_def *end_bit = nir_imm_intN_t(&b, 1 << nr_samples, 16);

   /* Create a loop in the wrapped function */
   nir_loop *loop = nir_push_loop(&b);
   {
      bit = nir_load_var(&b, i);
      nir_break_if(&b, nir_uge(&b, bit, end_bit));

      b.cursor = nir_cf_reinsert(&list, b.cursor);
      nir_store_var(&b, i, nir_ishl_imm(&b, bit, 1), ~0);
   }
   nir_pop_loop(&b, loop);

   /* We've mucked about with control flow */
   nir_progress(true, impl, nir_metadata_none);

   /* Use the loop variable for the active sampple mask each iteration */
   nir_shader_intrinsics_pass(shader, lower_active_samples,
                              nir_metadata_control_flow, bit);
   return true;
}

struct g17p_sample_copy {
   nir_def *active_samples;
   nir_def *samples_log2;
   unsigned colour_offset;
   /* Export base of this sample's depth, or 0 to keep the ABI's. */
   unsigned zs_base;
};

static void
specialize_g17p_sample_copy(nir_cf_list *copy,
                            const struct g17p_sample_copy *sample)
{
   foreach_list_typed(nir_cf_node, node, node, &copy->list) {
      nir_foreach_block_in_cf_node(block, node) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            switch (intr->intrinsic) {
            case nir_intrinsic_load_active_samples_agx:
               nir_def_rewrite_uses(&intr->def, sample->active_samples);
               nir_instr_remove(instr);
               break;

            case nir_intrinsic_load_samples_log2_agx:
               nir_def_rewrite_uses(&intr->def, sample->samples_log2);
               nir_instr_remove(instr);
               break;

            case nir_intrinsic_export_agx:
            case nir_intrinsic_load_exported_agx: {
               const unsigned base = nir_intrinsic_base(intr);
               if (base >= AGX_ABI_FOUT_COLOUR(0)) {
                  nir_intrinsic_set_base(intr, base + sample->colour_offset);
               } else if (sample->zs_base && (base == AGX_ABI_FOUT_Z ||
                                              base == AGX_ABI_FOUT_S)) {
                  nir_intrinsic_set_base(
                     intr, sample->zs_base + base - AGX_ABI_FOUT_Z);
               }
               break;
            }

            default:
               break;
            }
         }
      }
   }
}

/*
 * G17P has no external sample loop around linked fragment parts. Apple's
 * applegpu_g17p fragments that read [[sample_id]] instead repeat the shader
 * body once per sample, with the sample folded in, and store each sample
 * separately. Do the same for one shader part already lowered to per-sample
 * form (agx_nir_lower_to_per_sample): clone the body per sample with that
 * sample active.
 *
 * Colours cross from the main fragment to its epilog in export registers, so
 * each sample gets its own: sample i of every location follows all locations
 * of the samples before it. Depth and stencil of every sample after the first
 * follow all colours, four halves per sample. Both parts must be unrolled with
 * the same count and written locations.
 */
bool
agx_nir_unroll_g17p_samples(nir_shader *shader, unsigned nr_samples,
                            uint8_t loc_written)
{
   assert(shader->info.stage == MESA_SHADER_FRAGMENT);
   assert(nr_samples == 2 || nr_samples == 4);

   nir_function_impl *impl = nir_shader_get_entrypoint(shader);

   /* The forced early Z/S update belongs before the complete unrolled sample
    * body.  Cloning it once per sample runs the same tests repeatedly, which
    * violates the sample-mask contract and kills the later invocations. */
   bool force_early = false;
   if (shader->info.fs.early_fragment_tests) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_sample_mask_agx) {
               nir_instr_remove(instr);
               force_early = true;
               break;
            }
         }
         if (force_early)
            break;
      }
   }

   nir_cf_list body;
   nir_cf_extract(&body, nir_before_impl(impl), nir_after_impl(impl));

   nir_builder b = nir_builder_at(nir_before_impl(impl));
   if (force_early)
      nir_sample_mask_agx(&b, nir_imm_intN_t(&b, 0xff, 16),
                          nir_imm_intN_t(&b, 0xff, 16));

   nir_def *samples_log2 =
      nir_imm_intN_t(&b, util_logbase2(nr_samples), 16);

   for (unsigned i = 0; i < nr_samples; ++i) {
      const struct g17p_sample_copy sample = {
         .active_samples = nir_imm_intN_t(&b, BITFIELD_BIT(i), 16),
         .samples_log2 = samples_log2,
         .colour_offset = i * agx_g17p_sample_colour_stride(loc_written),
         .zs_base = i ? agx_g17p_sample_zs_base(nr_samples, loc_written, i)
                      : 0,
      };

      nir_cf_list copy;
      nir_cf_list_clone(&copy, &body, &impl->cf_node, NULL);
      specialize_g17p_sample_copy(&copy, &sample);
      nir_cf_reinsert(&copy, nir_after_impl(impl));
   }

   nir_cf_delete(&body);

   /* The backend publishes exports after the block that is last when it
    * translates them, so a copy with control flow would leave its exports
    * ahead of the later copies' blocks. Every export's value dominates the
    * end of the function, so gather them all there. */
   struct util_dynarray exports;
   util_dynarray_init(&exports, NULL);
   nir_def *write_samples[4] = {NULL};
   unsigned write_sample_count = 0;
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_export_agx) {
            nir_intrinsic_instr *export = intr;

            /* A forced-early shader exports one software write mask.  The
             * per-sample unroll clones that export along with the body, but
             * the fragment ABI has only one FOUT_WRITE_SAMPLES half.  Fold
             * the clones into one mask: each invocation owns only its active
             * bit and leaves every other sample untouched. */
            if (nir_intrinsic_base(export) ==
                AGX_ABI_FOUT_WRITE_SAMPLES) {
               assert(write_sample_count < nr_samples);
               write_samples[write_sample_count++] = export->src[0].ssa;
               nir_instr_remove(instr);
            } else {
               util_dynarray_append(&exports, instr);
            }
         }
      }
   }

   if (write_sample_count) {
      assert(write_sample_count == nr_samples);
      b = nir_builder_at(nir_after_impl(impl));
      nir_def *combined = nir_imm_intN_t(&b, 0xff, 16);
      for (unsigned i = 0; i < nr_samples; ++i) {
         const unsigned bit = BITFIELD_BIT(i);
         nir_def *owned = nir_iand_imm(&b, write_samples[i], bit);
         nir_def *keep_others = nir_ior_imm(&b, owned, 0xff ^ bit);
         combined = nir_iand(&b, combined, keep_others);
      }
      nir_export_agx(&b, combined, .base = AGX_ABI_FOUT_WRITE_SAMPLES);
   }

   util_dynarray_foreach(&exports, nir_instr *, instr)
      nir_instr_move(nir_after_impl(impl), *instr);
   util_dynarray_fini(&exports);

   return nir_progress(true, impl, nir_metadata_none);
}

static bool movable_zs_source(nir_def *def);

static bool
movable_zs_src_cb(nir_src *src, void *data)
{
   return movable_zs_source(src->ssa);
}

static bool
movable_zs_source(nir_def *def)
{
   nir_instr *instr = nir_def_instr(def);
   switch (instr->type) {
   case nir_instr_type_load_const:
   case nir_instr_type_undef:
      return true;
   case nir_instr_type_alu:
      return nir_foreach_src(instr, movable_zs_src_cb, NULL);
   case nir_instr_type_intrinsic:
      switch (nir_instr_as_intrinsic(instr)->intrinsic) {
      case nir_intrinsic_load_exported_agx:
      /* Alpha-coverage dithering reads FragCoord.xy, later lowered to pixel
       * coordinates. Both inputs are invariant across coverage/Z/S updates. */
      case nir_intrinsic_load_frag_coord:
      case nir_intrinsic_load_pixel_coord:
         return true;
      default:
         return false;
      }
   default:
      return false;
   }
}

struct move_zs_state {
   nir_cursor cursor;
   struct set *moved;
};

static bool
move_zs_source(nir_src *src, void *data)
{
   struct move_zs_state *state = data;
   nir_instr *instr = nir_def_instr(src->ssa);
   if (_mesa_set_search(state->moved, instr))
      return true;

   nir_foreach_src(instr, move_zs_source, state);
   nir_instr_move(state->cursor, instr);
   state->cursor = nir_after_instr(instr);
   _mesa_set_add(state->moved, instr);
   return true;
}

/*
 * An unrolled epilog writes each sample's depth or stencil in its own copy,
 * between the colour stores of the samples around it. Apple's per-sample
 * depth writes all share one class-1 transaction, which a colour transaction
 * cannot interrupt, so move every such write, with the constant and exported
 * values it reads, to the start of the epilog. Pure exported-mask kills can
 * move with them, but must all precede the depth transaction.
 */
bool
agx_nir_hoist_g17p_zs_writes(nir_shader *shader)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   nir_intrinsic_instr *kills[8];
   nir_intrinsic_instr *writes[8];
   unsigned kill_count = 0, count = 0;
   bool progress = false;

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_demote_samples)
            return false;

         if (intr->intrinsic == nir_intrinsic_sample_mask_agx) {
            /* A late split epilog kills the samples discarded by the main
             * shader before opening the class-1 DEPTHF transaction.  The
             * per-sample unroller cloned that update, so gather every clone
             * ahead of every depth write. */
            if (kill_count == ARRAY_SIZE(kills) ||
                block != nir_start_block(impl) ||
                !nir_src_is_const(intr->src[1]) ||
                nir_src_as_uint(intr->src[1]) != 0 ||
                !movable_zs_source(intr->src[0].ssa))
               return false;
            kills[kill_count++] = intr;
            continue;
         }

         if (intr->intrinsic != nir_intrinsic_store_zs_agx)
            continue;

         /* Static sample unrolling can prove that an API mask excludes this
          * sample. A zero-coverage Z/S store is no operation; retaining it
          * would instead make the backend issue a DEPTHF for that sample. */
         if (nir_src_is_const(intr->src[0]) &&
             nir_src_as_uint(intr->src[0]) == 0) {
            nir_instr_remove(instr);
            progress = true;
            continue;
         }

         if (count == ARRAY_SIZE(writes) || block != nir_start_block(impl))
            return false;

         for (unsigned i = 0; i < 3; ++i) {
            if (!movable_zs_source(intr->src[i].ssa))
               return false;
         }
         writes[count++] = intr;
      }
   }

   if (count == 0 || (count == 1 && kill_count == 0))
      return nir_progress(progress, impl, nir_metadata_control_flow);

   struct move_zs_state state = {
      .cursor = nir_before_impl(impl),
      .moved = _mesa_pointer_set_create(NULL),
   };
   nir_def *killed = NULL;
   for (unsigned k = 0; k < kill_count; ++k) {
      nir_foreach_src(&kills[k]->instr, move_zs_source, &state);
      nir_builder b = nir_builder_at(state.cursor);
      nir_def *mask = kills[k]->src[0].ssa;
      killed = killed ? nir_ior(&b, killed, mask) : mask;
      state.cursor = b.cursor;
      nir_instr_remove(&kills[k]->instr);
   }
   if (killed) {
      /* Kill-only updates commute: union their disjoint sample masks and
       * issue one coverage update before any Z/S or colour write. Keeping
       * each unrolled clone would repeat the same coverage transaction. */
      nir_builder b = nir_builder_at(state.cursor);
      nir_sample_mask_agx(&b, killed, nir_imm_intN_t(&b, 0, killed->bit_size));
      state.cursor = b.cursor;
   }
   for (unsigned w = 0; w < count; ++w) {
      nir_foreach_src(&writes[w]->instr, move_zs_source, &state);
      nir_instr_move(state.cursor, &writes[w]->instr);
      state.cursor = nir_after_instr(&writes[w]->instr);
   }
   _mesa_set_destroy(state.moved, NULL);

   return nir_progress(true, impl, nir_metadata_control_flow);
}

/*
 * Lower a fragment shader into a monolithic pixel shader, with static sample
 * count, blend state, and tilebuffer formats in the shader key. For dynamic,
 * epilogs must be used, which have separate lowerings.
 */
bool
agx_nir_lower_monolithic_msaa(nir_shader *shader, uint8_t nr_samples)
{
   assert(shader->info.stage == MESA_SHADER_FRAGMENT);
   assert(nr_samples == 1 || nr_samples == 2 || nr_samples == 4);

   agx_nir_lower_sample_mask(shader);

   /* In single sampled programs, interpolateAtSample needs to return the
    * center pixel.
    */
   if (nr_samples == 1) {
      nir_lower_single_sampled_options lss_opts = {
         .lower_sample_mask_in = true,
      };
      nir_lower_single_sampled(shader, &lss_opts);
   } else if (shader->info.fs.uses_sample_shading) {
      agx_nir_lower_to_per_sample(shader);
      agx_nir_wrap_per_sample_loop(shader, nr_samples);
   }

   return true;
}
