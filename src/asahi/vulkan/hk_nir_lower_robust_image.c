/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_nir_lower_robust_image.h"

#include "nir.h"
#include "nir_builder.h"

struct hk_robust_image_lowering {
   bool lower_loads_and_stores;
};

static bool
should_lower_robust_image(const nir_intrinsic_instr *intr, const void *data)
{
   const struct hk_robust_image_lowering *options = data;

   /* Image atomics lower to raw memory and always need an explicit bounds
    * predicate. Loads and stores follow the per-chip policy supplied by the
    * driver. */
   switch (intr->intrinsic) {
   case nir_intrinsic_image_deref_load:
      /* A formatless load's out-of-bounds value depends on the bound view's
       * format, which the common lowering cannot see. The texture lowering
       * predicates it with the view's descriptor instead. */
      return options->lower_loads_and_stores &&
             nir_intrinsic_format(intr) != PIPE_FORMAT_NONE;
   case nir_intrinsic_image_deref_store:
      return options->lower_loads_and_stores;
   case nir_intrinsic_image_deref_atomic:
   case nir_intrinsic_image_deref_atomic_swap:
      return true;
   default:
      return false;
   }
}

static bool
mark_lowered_access_in_bounds(UNUSED nir_builder *b,
                              nir_intrinsic_instr *intr, void *data)
{
   if (!should_lower_robust_image(intr, data))
      return false;

   nir_intrinsic_set_access(intr,
                            nir_intrinsic_access(intr) | ACCESS_IN_BOUNDS);
   return true;
}

bool
hk_nir_lower_robust_image_access(nir_shader *nir,
                                 bool lower_loads_and_stores)
{
   const struct hk_robust_image_lowering options = {
      .lower_loads_and_stores = lower_loads_and_stores,
   };

   const bool progress =
      nir_lower_robust_access(nir, should_lower_robust_image, &options);

   /* The surviving access is inside the predicate inserted above. Tell the
    * later AGX image lowering that it must not add its own second bounds
    * query around the cloned instruction. */
   if (progress) {
      nir_shader_intrinsics_pass(nir, mark_lowered_access_in_bounds,
                                 nir_metadata_all, (void *)&options);
   }

   return progress;
}
