/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * AGX3 special-register read encoder (get_sr -> rld, source-kind f23=7).
 *
 * A special-register read is an async producer: it signals a scoreboard slot
 * (f12) and its consumers carry wait bits, exactly like a device load. The SR
 * number goes in f21; the width class (16- or 32-bit destination) in f10,
 * mirrored in f24. The compact 4-byte form (ep0) cannot carry wait bits or a
 * wide destination; the encoder escalates to the 8-byte form (ep1) when the
 * composer refuses ep0 -- the same mechanical escalation visible in Apple's
 * own emissions. Only (SR, width) pairs witnessed in the corpus encode; the
 * rest fail closed. Grade A: reproduces all 306 clean Apple SR-read witnesses
 * byte-exactly via the composer.
 * Evidence: research/gpu/GPU_AGX3_SR_OPS_2026-08-25.md.
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "asahi/lib/agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Destination width class, the value of f10/f24. */
enum agx3_sr_width {
   AGX3_SR_W16 = 1,
   AGX3_SR_W32 = 2,
};

#define AGX3_SR_MAX_LENGTH 8u

struct agx3_sr_desc {
   uint16_t dest;      /* destination register, 16-bit half units */
   enum agx3_sr_width width;
   uint8_t sr;         /* Apple's normalized special-register field f21 */
   uint8_t slot;       /* scoreboard slot this read signals: 1..6 (b) / 1..8 (c) */
   uint16_t wait_mask; /* bit i sets wait on slot i+1 */
   /* f48 on a fragment read of SR 34, the active thread index in the
    * subgroup. Apple sets it only when the result reaches nothing but the
    * colour output, never when it feeds a derivative or a quad reduction,
    * which also read helper invocations. Placement is exact; the packer
    * leaves it clear, as Apple does for every other use. */
   bool fragment_active_index;

   /* G17P fragment-input entry marker (f48, 8-byte form only). Apple places
    * it on an SR39 read that is a fragment's first input. */
   bool first;
};

/* True if (sr, width) has a byte-exact corpus witness. Unwitnessed pairs must
 * not be emitted: the composer would accept any f21 value the field can hold,
 * so this table -- not the encoder -- is the fail-closed gate. */
bool agx3_sr_witnessed(uint8_t sr, enum agx3_sr_width width);

bool agx3_sr_encode(enum agx_isa_encoding enc, const struct agx3_sr_desc *d,
                    uint8_t out[AGX3_SR_MAX_LENGTH], uint8_t *len);

#ifdef __cplusplus
}
#endif
