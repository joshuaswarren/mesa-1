/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * AGX3 tile / framebuffer encoders: st_tile -> lstcol (fragment color output)
 * and ld_tile -> lldcol (framebuffer fetch).  This core intentionally exposes
 * the directly witnessed single-sample and 2x/4x color forms. Composer-driven;
 * reproduces Apple's witnessed bytes. Blend and depth-compare are render
 * state, not ISA. Evidence:
 * research/gpu/GPU_AGX3_TILE_OPS_2026-08-24.md.
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "asahi/lib/agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AGX3_TILE_MAX_LENGTH 14u

enum agx3_tile_width {
   AGX3_TILE_W8 = 0,
   AGX3_TILE_W16 = 1,
   AGX3_TILE_W32 = 2,
};

enum agx3_tile_coverage {
   AGX3_TILE_COV_1X = 1,
   AGX3_TILE_COV_2X_BROADCAST,
   AGX3_TILE_COV_2X_SAMPLE_0,
   AGX3_TILE_COV_2X_SAMPLE_1,
   AGX3_TILE_COV_4X_BROADCAST,
   AGX3_TILE_COV_4X_SAMPLE_0,
   AGX3_TILE_COV_4X_SAMPLE_1,
   AGX3_TILE_COV_4X_SAMPLE_2,
   AGX3_TILE_COV_4X_SAMPLE_3,
};

/* Store `data` to the tile buffer. `chan_mask` is the component write mask
 * (R=1, G=2, B=4, A=8). `tile_off` is the byte offset of this render target
 * (sum of the per-sample sizes of the preceding RTs). */
bool agx3_tile_store(enum agx_isa_encoding enc, uint16_t data,
                     enum agx3_tile_width width, uint8_t chan_mask,
                     uint16_t tile_off, enum agx3_tile_coverage cov,
                     uint16_t wait_mask, uint8_t out[AGX3_TILE_MAX_LENGTH],
                     uint8_t *len);

/* Load from the tile buffer into `dest` on scoreboard slot `slot`.  `first`
 * and `last` are the load-sequence boundary markers witnessed in the corpus;
 * AGX3b encodes only `first`, while AGX3c encodes both. */
bool agx3_tile_load(enum agx_isa_encoding enc, uint16_t dest,
                    enum agx3_tile_width width, uint8_t slot, uint8_t chan_mask,
                    uint16_t tile_off, enum agx3_tile_coverage cov,
                    uint16_t wait_mask, bool first, bool last,
                    uint8_t out[AGX3_TILE_MAX_LENGTH], uint8_t *len);

#ifdef __cplusplus
}
#endif
