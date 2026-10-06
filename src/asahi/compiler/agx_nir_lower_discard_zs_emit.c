/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "agx_compiler.h"
#include "nir_builder_opcodes.h"

#define ALL_SAMPLES 0xFF
#define BASE_Z      1
#define BASE_S      2

static bool
lower_zs_emit(nir_block *block, bool force_early_z)
{
   nir_intrinsic_instr *zs_emit = NULL;
   bool progress = false;

   nir_foreach_instr_reverse_safe(instr, block) {
      if (instr->type != nir_instr_type_intrinsic)
         continue;

      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      if (intr->intrinsic != nir_intrinsic_store_output)
         continue;

      nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
      if (sem.location != FRAG_RESULT_DEPTH &&
          sem.location != FRAG_RESULT_STENCIL)
         continue;

      /* If early-Z is forced, z/s writes are a no-op (and will cause problems
       * later in the compile). Piglit early-z tests this. Just remove the
       * offending writes.
       */
      if (force_early_z) {
         nir_instr_remove(instr);
         progress = true;
         continue;
      }

      nir_builder b = nir_builder_at(nir_before_instr(instr));

      nir_def *value = intr->src[0].ssa;
      bool z = (sem.location == FRAG_RESULT_DEPTH);

      unsigned src_idx = z ? 1 : 2;
      unsigned base = z ? BASE_Z : BASE_S;

      /* In the hw, depth is 32-bit but stencil is 16-bit. Instruction
       * selection checks this, so emit the conversion now.
       */
      if (z)
         value = nir_f2f32(&b, value);
      else
         value = nir_u2u16(&b, value);

      if (zs_emit == NULL) {
         /* Multisampling will get lowered later if needed, default to
          * broadcast
          */
         nir_def *sample_mask = nir_imm_intN_t(&b, ALL_SAMPLES, 16);
         zs_emit =
            nir_store_zs_agx(&b, sample_mask, nir_undef(&b, 1, 32) /* depth */,
                             nir_undef(&b, 1, 16) /* stencil */);
      }

      assert((nir_intrinsic_base(zs_emit) & base) == 0 &&
             "each of depth/stencil may only be written once");

      nir_src_rewrite(&zs_emit->src[src_idx], value);
      nir_intrinsic_set_base(zs_emit, nir_intrinsic_base(zs_emit) | base);

      nir_instr_remove(instr);
      progress = true;
   }

   return progress;
}

static bool
lower_discard(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_demote &&
       intr->intrinsic != nir_intrinsic_demote_if)
      return false;

   b->cursor = nir_before_instr(&intr->instr);

   nir_def *all_samples = nir_imm_intN_t(b, ALL_SAMPLES, 16);
   nir_def *no_samples = nir_imm_intN_t(b, 0, 16);
   nir_def *killed_samples = all_samples;

   if (intr->intrinsic == nir_intrinsic_demote_if)
      killed_samples = nir_bcsel(b, intr->src[0].ssa, all_samples, no_samples);

   /* This will get lowered later as needed */
   nir_demote_samples(b, killed_samples);
   nir_instr_remove(&intr->instr);
   return true;
}

static bool
agx_nir_lower_discard(nir_shader *s)
{
   if (!s->info.fs.uses_discard)
      return false;

   return nir_shader_intrinsics_pass(s, lower_discard,
                                     nir_metadata_control_flow, NULL);
}

static nir_intrinsic_instr *
as_zs_store(nir_instr *instr)
{
   if (instr->type != nir_instr_type_intrinsic)
      return NULL;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   if (intr->intrinsic != nir_intrinsic_store_output)
      return NULL;

   unsigned location = nir_intrinsic_io_semantics(intr).location;
   if (location != FRAG_RESULT_DEPTH && location != FRAG_RESULT_STENCIL)
      return NULL;

   return intr;
}

/*
 * Depth/stencil become a single store_zs_agx per block, and that becomes
 * export_agx, which the backend requires in the exit block. Frontends store
 * outputs once at the end, but a pass can later move already-lowered output
 * stores into control flow. Honeykrisp's ordinary-descriptor specialization
 * clones the whole fragment body into an if/else, so the G17P pass-through
 * depth write from agx_nir_lower_frag_sidefx lands once in each branch
 * (dEQP-VK.glsl.builtin_var.fragdepth.*: "Invalid block form" abort). Route
 * every store of such a location through a local variable and store it once
 * at the end. Paths that do not write it leave it undefined, as before.
 */
static bool
sink_zs_stores(nir_function_impl *impl)
{
   nir_block *last = nir_impl_last_block(impl);
   nir_intrinsic_instr *template[2] = {NULL, NULL};
   nir_variable *var[2] = {NULL, NULL};
   bool sink[2] = {false, false};

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         nir_intrinsic_instr *intr = as_zs_store(instr);
         if (intr) {
            unsigned i =
               nir_intrinsic_io_semantics(intr).location == FRAG_RESULT_STENCIL;
            sink[i] |= block != last;
            template[i] = intr;
         }
      }
   }

   if (!sink[0] && !sink[1])
      return false;

   nir_builder b = nir_builder_create(impl);

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         nir_intrinsic_instr *intr = as_zs_store(instr);
         if (!intr)
            continue;

         unsigned i =
            nir_intrinsic_io_semantics(intr).location == FRAG_RESULT_STENCIL;
         if (!sink[i])
            continue;

         nir_def *value = intr->src[0].ssa;
         assert(value->num_components == 1 && nir_src_is_const(intr->src[1]));

         if (!var[i]) {
            var[i] = nir_local_variable_create(
               impl, glsl_uintN_t_type(value->bit_size), i ? "stencil" : "z");
         }

         b.cursor = nir_before_instr(instr);
         nir_store_var(&b, var[i], value, 1);
         nir_instr_remove(instr);
      }
   }

   b.cursor = nir_after_impl(impl);

   for (unsigned i = 0; i < 2; ++i) {
      if (!sink[i])
         continue;

      nir_intrinsic_instr *t = template[i];
      nir_store_output(&b, nir_load_var(&b, var[i]), nir_imm_int(&b, 0),
                       .base = nir_intrinsic_base(t),
                       .component = nir_intrinsic_component(t),
                       .write_mask = 1,
                       .src_type = nir_intrinsic_src_type(t),
                       .io_semantics = nir_intrinsic_io_semantics(t));
   }

   nir_progress(true, impl, nir_metadata_none);
   nir_lower_vars_to_ssa(impl->function->shader);
   return true;
}

static bool
agx_nir_lower_zs_emit(nir_shader *s)
{
   /* If depth/stencil isn't written, there's nothing to lower */
   if (!(s->info.outputs_written & (BITFIELD64_BIT(FRAG_RESULT_STENCIL) |
                                    BITFIELD64_BIT(FRAG_RESULT_DEPTH))))
      return false;

   bool any_progress = false;

   nir_foreach_function_impl(impl, s) {
      bool progress = sink_zs_stores(impl);

      nir_foreach_block(block, impl) {
         progress |= lower_zs_emit(block, s->info.fs.early_fragment_tests);
      }

      any_progress |= nir_progress(progress, impl, nir_metadata_none);
   }

   return any_progress;
}

bool
agx_nir_lower_discard_zs_emit(nir_shader *s)
{
   bool progress = false;

   /* Lower depth/stencil writes before discard so the interaction works */
   progress |= agx_nir_lower_zs_emit(s);
   progress |= agx_nir_lower_discard(s);

   return progress;
}
