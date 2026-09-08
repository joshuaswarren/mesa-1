/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * AGX3 varying / interpolation encoders.
 *
 *   iter     -> f32interp : scalar per-channel linear interpolation of a
 *               fragment input coefficient. Perspective is done in software
 *               (interp RHW, rcp, interp channel, mul), so this is the bare
 *               linear iterator.
 *   ldcf     -> ldcfr     : raw <A,B,C> coefficient load.
 *   st_vary  -> uvsw      : unified-vertex-store write of one 32-bit component.
 *
 * Composer-driven; reproduces Apple's witnessed bytes exactly. Center and
 * centroid locations are explicit; the scheduling/hazard marker fields are
 * emitted zero, which is table-valid but remains unproven on hardware. Evidence:
 * research/gpu/GPU_AGX3_VARYING_OPS_2026-08-24.md.
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "asahi/lib/agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AGX3_VARY_MAX_LENGTH 10u

enum agx3_vary_location {
   AGX3_VARY_CENTER = 0,
   AGX3_VARY_CENTROID = 1,
};

/* Interpolate coefficient `coef_index` into `dest`. `perspective` selects the
 * perspective-correct linear iterator (the caller pairs it with the software
 * rcp/mul sequence). */
bool agx3_vary_interp_at(enum agx_isa_encoding enc, uint16_t dest,
                         uint16_t coef_index, bool perspective,
                         enum agx3_vary_location location,
                         uint8_t out[AGX3_VARY_MAX_LENGTH], uint8_t *len);

/* Native G17P sets f63 when F32INTERP is the first fragment-input operation.
 * LDCF f48 is the alternate encoding of this one shared entry marker, so a
 * later first operation from the other family must not restart the part.
 * This marker is independent of f54, whose optimization/liveness role remains
 * unproven and is deliberately left clear. */
/* G17P center RHW used only by the coefficient-tagged projective reciprocal.
 * Native pure smooth and smooth+position.z witnesses set f54(bit50); ordinary
 * position.w/output/arithmetic witnesses clear it. Numerical semantics beyond
 * that use domain are not established. `first` independently controls f63. */
bool agx3_vary_projective_rhw(enum agx_isa_encoding enc, uint16_t dest,
                             uint16_t coef_index, bool first,
                             uint8_t out[AGX3_VARY_MAX_LENGTH], uint8_t *len);

bool agx3_vary_interp_at_start(enum agx_isa_encoding enc, uint16_t dest,
                               uint16_t coef_index, bool perspective,
                               enum agx3_vary_location location, bool first,
                               uint8_t out[AGX3_VARY_MAX_LENGTH], uint8_t *len);

/* Backward-compatible center-location wrapper. */
bool agx3_vary_interp(enum agx_isa_encoding enc, uint16_t dest,
                      uint16_t coef_index, bool perspective,
                      uint8_t out[AGX3_VARY_MAX_LENGTH], uint8_t *len);

/* Load the raw coefficient triple for `coef_index` into `dest`, on scoreboard
 * slot `slot` (1..6/1..8). */
/* Native G17P sets f48 when LDCF is the first fragment-input operation. */
bool agx3_vary_ldcf_at_start(enum agx_isa_encoding enc, uint16_t dest,
                            uint16_t coef_index, uint8_t slot, bool first,
                            uint8_t out[AGX3_VARY_MAX_LENGTH], uint8_t *len);

bool agx3_vary_ldcf(enum agx_isa_encoding enc, uint16_t dest,
                    uint16_t coef_index, uint8_t slot,
                    uint8_t out[AGX3_VARY_MAX_LENGTH], uint8_t *len);

/* Write `value` to unified-vertex-store index `uvs_index`, on slot `slot`.
 * `value_kill` marks the last write of that register. `wait_mask` bit i waits
 * scoreboard slot i. */
bool agx3_vary_uvsw(enum agx_isa_encoding enc, uint16_t value,
                    uint16_t uvs_index, uint8_t slot, bool value_kill,
                    uint16_t wait_mask, uint8_t out[AGX3_VARY_MAX_LENGTH],
                    uint8_t *len);

#ifdef __cplusplus
}
#endif
