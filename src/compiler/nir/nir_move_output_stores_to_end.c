/*
 * Copyright © 2025 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 */

/* This pass moves output stores to the end of the shader.
 * (only those that can be moved trivially)
 */

#include "nir.h"
#include "nir_builder.h"

/* Put the position in the last slot to make its store last. */
#define LAST_SLOT NUM_TOTAL_VARYING_SLOTS
#define NUM_SLOTS ((LAST_SLOT + 1) * 4)

typedef struct {
   nir_instr *stores[NUM_SLOTS];
   bool indirect_uvs;
   /* Whether the output component is written only once or multiple times. */
   BITSET_DECLARE(single, NUM_SLOTS);
   BITSET_DECLARE(multiple, NUM_SLOTS);
} output_stores_state;

static bool
gather_output_stores(struct nir_builder *b, nir_intrinsic_instr *intr,
                     void *opaque)
{
   output_stores_state *state = (output_stores_state *)opaque;

   unsigned slot;
   if (intr->intrinsic == nir_intrinsic_store_uvs_agx) {
      /* The scalar UVS stores use an already linked component index. A
       * dynamic index may alias every fixed store, so leave that shader's
       * UVS writes in place. */
      if (!nir_src_is_const(intr->src[1])) {
         state->indirect_uvs = true;
         return false;
      }
      slot = nir_src_as_uint(intr->src[1]);
      if (slot >= NUM_SLOTS) {
         state->indirect_uvs = true;
         return false;
      }
   } else if (intr->intrinsic == nir_intrinsic_store_output) {
      unsigned location = nir_intrinsic_io_semantics(intr).location;
      unsigned component = nir_intrinsic_component(intr);
      assert(location < NUM_TOTAL_VARYING_SLOTS);
      assert(component < 4);
      assert(!nir_intrinsic_io_semantics(intr).high_16bits);
      if (location == VARYING_SLOT_POS)
         location = LAST_SLOT;
      slot = location * 4 + component;
   } else {
      return false;
   }

   /* Conditional stores do not dominate the exit. Count them as multiple
    * writes so another store to that component is not moved across them. */
   if (intr->instr.block->cf_node.parent->type != nir_cf_node_function) {
      BITSET_CLEAR(state->single, slot);
      BITSET_SET(state->multiple, slot);
      return false;
   }

   unsigned num_components = intr->src[0].ssa->num_components;

   /* Each component must be written only once. */
   bool multiple = false;
   for (unsigned i = 0; i < num_components; i++) {
      if (BITSET_TEST(state->multiple, slot + i)) {
         multiple = true;
      } else if (BITSET_TEST(state->single, slot + i)) {
         BITSET_CLEAR(state->single, slot + i);
         BITSET_SET(state->multiple, slot + i);
         multiple = true;
      }
   }

   if (!multiple) {
      state->stores[slot] = &intr->instr;
      BITSET_SET_RANGE_INSIDE_WORD(state->single, slot,
                                   slot + num_components - 1);
   }
   return false;
}

bool
nir_move_output_stores_to_end(nir_shader *nir)
{
   assert(nir->info.stage == MESA_SHADER_VERTEX ||
          nir->info.stage == MESA_SHADER_TESS_EVAL);

   output_stores_state state;
   memset(&state, 0, sizeof(state));

   /* Gather output stores. */
   nir_shader_intrinsics_pass(nir, gather_output_stores, nir_metadata_all,
                              &state);

   if (state.indirect_uvs)
      return false;

   /* Move output stores to the end (only those that we can move). */
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   bool progress = false;
   unsigned i;

   BITSET_FOREACH_SET(i, state.single, NUM_SLOTS) {
      if (!state.stores[i])
         continue;

      nir_instr_remove(state.stores[i]);
      nir_instr_insert(nir_after_impl(impl), state.stores[i]);
      progress = true;
   }

   return nir_progress(progress, impl, nir_metadata_control_flow);
}
