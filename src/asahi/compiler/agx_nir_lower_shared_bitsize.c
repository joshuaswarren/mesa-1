/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "compiler/nir/nir_builder.h"
#include "agx_compiler.h"

/* Local memory instructions require 16-bit offsets, so we add conversions. */
static bool
pass(struct nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   nir_src *offset = nir_get_io_offset_src(intr);
   if (!nir_is_shared_access(intr))
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *address = offset->ssa;

   /* AGX3 ldtg/sttg register indices count scalar elements, while AGX2's
    * local-memory base remains byte addressed. Keep atomics on their separate
    * byte-addressed encoding and convert only ordinary loads/stores. */
   const bool element_indices = *(const bool *)data;
   if (element_indices &&
       (intr->intrinsic == nir_intrinsic_load_shared ||
        intr->intrinsic == nir_intrinsic_store_shared)) {
      const unsigned bit_size =
         intr->intrinsic == nir_intrinsic_load_shared
            ? intr->def.bit_size
            : nir_src_bit_size(intr->src[0]);
      const unsigned shift = util_logbase2(bit_size / 8);
      address = nir_ushr_imm(b, address, shift);
   }

   if (address->bit_size != 16)
      address = nir_u2u16(b, address);

   if (address == offset->ssa)
      return false;

   nir_src_rewrite(offset, address);
   return true;
}

bool
agx_nir_lower_shared_bitsize(nir_shader *shader, bool element_indices)
{
   return nir_shader_intrinsics_pass(shader, pass, nir_metadata_control_flow,
                                     &element_indices);
}
