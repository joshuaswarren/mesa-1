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

/* Fragment fetches reuse the native32 array and device sampler0, including
 * multiple resources. Filtered samples use the bounded single-resource form
 * below. The indexed_fetch flag selects the publication contract.
 *
 * Fragment stage, G17P: rewrite the shader's single 2D sample of a root-table
 * texture through a bound API sampler to the native texture0/sampler0 form.
 * Both hardware indices become zero; the driver publishes the selected 32-byte
 * texture descriptor and the 32-byte HAL200 sampler word as the two pointers
 * in uniform halves 0..7 (Honeykrisp's combined-sampler fragment ABI). Runs
 * after agx_nir_lower_texture, which is what gave the sampler its +1 handle.
 * Fails closed on every other shape and leaves the shader unchanged. */
struct d12_nir_g17p_fragment_sample {
   bool indexed_fetch; /* native32 array plus the device texel-fetch sampler */
   uint32_t texture_offset_B;
   uint8_t sampler_slot;
};
bool d12_nir_lower_g17p_fragment_sample(
   nir_shader *nir, uint32_t generation,
   struct d12_nir_g17p_fragment_sample *out);
