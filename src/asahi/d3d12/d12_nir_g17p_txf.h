/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT */
#pragma once
#include <stdbool.h>
#include <stdint.h>
#include "nir.h"

/* Ordinary native 2D readonly table fetches only. Validation is read-only;
 * lowering revalidates before mutating. This admits compilation, not dispatch.
 */
bool d12_nir_g17p_txf_supported(nir_shader *nir, bool *has_txf);
bool d12_nir_lower_g17p_txf(nir_shader *nir, uint32_t generation);
bool d12_nir_shift_g17p_txf_root(nir_shader *nir);
