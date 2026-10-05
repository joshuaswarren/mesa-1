/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#ifndef D12_NIR_SPARSE_BUFFER_H
#define D12_NIR_SPARSE_BUFFER_H

#include "nir_builder.h"

/* Shared DXIL/DXBC residency query. The map is prebiased to the tile
 * containing the view base. Loads of at most 16 bytes touch at most two
 * 64 KiB tiles. Dense resources have no map; invalid accesses are not
 * reported mapped. Read metadata through the zero page when not needed. */
static inline nir_def *
d12_nir_buffer_residency_status(nir_builder *b, nir_def *map, nir_def *base,
                                nir_def *byte_offset, nir_def *size_B,
                                nir_def *access_size_B, nir_def *element_valid,
                                nir_def *zero_page)
{
   nir_def *offset = nir_u2u64(b, byte_offset);
   nir_def *view_size = nir_u2u64(b, size_B);
   nir_def *last_requested = nir_uadd_sat(
      b, offset,
      nir_u2u64(b, nir_iadd_imm(b, nir_umax_imm(b, access_size_B, 1), -1)));
   nir_def *valid = nir_iand(b, nir_ult(b, offset, view_size),
                            nir_ult(b, last_requested, view_size));
   if (element_valid)
      valid = nir_iand(b, valid, element_valid);
   nir_def *has_map = nir_ine_imm(b, map, 0);
   nir_def *needs_map = nir_iand(b, valid, has_map);

   nir_def *phase = nir_u2u64(b, nir_iand_imm(b, nir_u2u32(b, base), 65535));
   nir_def *first_tile =
      nir_ushr_imm(b, nir_iadd(b, phase, offset), 16);
   nir_def *last_tile =
      nir_ushr_imm(b, nir_iadd(b, phase, last_requested), 16);
   first_tile = nir_bcsel(b, needs_map, first_tile, nir_imm_int64(b, 0));
   last_tile = nir_bcsel(b, needs_map, last_tile, nir_imm_int64(b, 0));

   nir_def *safe_map = nir_bcsel(b, needs_map, map, zero_page);
   nir_def *first_word = nir_load_global(
      b, 1, 32,
      nir_iadd(b, safe_map, nir_imul_imm(b, first_tile, 4)),
      .align_mul = 4);
   nir_def *last_word = nir_load_global(
      b, 1, 32,
      nir_iadd(b, safe_map, nir_imul_imm(b, last_tile, 4)),
      .align_mul = 4);
   nir_def *mapped = nir_iand(b, nir_ine_imm(b, first_word, 0),
                              nir_ine_imm(b, last_word, 0));
   nir_def *resident =
      nir_iand(b, valid, nir_ior(b, nir_inot(b, has_map), mapped));
   return nir_bcsel(b, resident, nir_imm_int(b, ~0u), nir_imm_int(b, 0));
}

#endif
