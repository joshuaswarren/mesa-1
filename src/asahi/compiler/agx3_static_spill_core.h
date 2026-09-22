/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * AGX3 static thread-private-storage spill encoders/decoders.
 *
 * Apple's compiler uses copy(f9=4) for an asynchronous GPR -> TPS spill and
 * rld(f23=4) for an asynchronous TPS -> GPR reload.  These are the static
 * counterpart of the separately modeled register-indexed ldtps/sttps family.
 * No SCS or ray-tracing scratch operands are represented here.
 *
 * Evidence: GPU_AGX3_LOCAL_STACK_WAIT_2026-08-24.md and the pinned
 * G15G/G16G/G17S compiler-vector corpus.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "asahi/lib/agx_chip.h"
#include "agx3_mem_core.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AGX3_STATIC_SPILL_MAX_LENGTH   10u
#define AGX3_STATIC_SPILL_MAX_TPS_HALF 0x1ffffu

struct agx3_static_spill_store_desc {
   uint32_t tps_half; /* static TPS destination in half-register units */
   uint16_t src;      /* GPR source in half-register units */
   enum agx3_mem_width width; /* witnessed W16 or W32 */
   bool src_kill;
   uint8_t slot; /* signalled scoreboard slot */
   uint16_t wait_mask;

   /* f39 on AGX3b / f42 on AGX3c.  Placement is grounded, policy is not;
    * production lowering leaves it clear while decode/replay preserves it. */
   bool structural_flag;
};

struct agx3_static_spill_load_desc {
   uint16_t dest;             /* GPR destination in half-register units */
   uint32_t tps_half;         /* static TPS source in half-register units */
   enum agx3_mem_width width; /* witnessed W16 or W32 */
   uint8_t slot;              /* signalled scoreboard slot */
   /* f26: usually mirrors slot, but a grounded long-form subset uses zero.
    * Placement is exact; production lowering uses the mirror-only subset. */
   uint8_t slot_mirror;
   uint16_t wait_mask;

   /* Structural Apple fields with grounded placement but no inferred policy:
    * f11 and f41 (AGX3b) / f44 (AGX3c). */
   bool narrow_extend;
   bool structural_flag;

   /* The TPS cell is read again before it is next written. Clears f27, the
    * reload's last-use bit, which applegpu_g17p sets on every other reload. */
   bool tps_live;
};

bool agx3_static_spill_encode_store(
   enum agx_isa_encoding enc, const struct agx3_static_spill_store_desc *desc,
   uint8_t out[AGX3_STATIC_SPILL_MAX_LENGTH], uint8_t *length);
bool agx3_static_spill_encode_store_form(
   enum agx_isa_encoding enc, const struct agx3_static_spill_store_desc *desc,
   uint8_t ep, uint8_t out[AGX3_STATIC_SPILL_MAX_LENGTH], uint8_t *length);
bool agx3_static_spill_decode_store(enum agx_isa_encoding enc,
                                    const uint8_t *bytes, size_t length,
                                    struct agx3_static_spill_store_desc *out);

bool agx3_static_spill_encode_load(
   enum agx_isa_encoding enc, const struct agx3_static_spill_load_desc *desc,
   uint8_t out[AGX3_STATIC_SPILL_MAX_LENGTH], uint8_t *length);
bool agx3_static_spill_encode_load_form(
   enum agx_isa_encoding enc, const struct agx3_static_spill_load_desc *desc,
   uint8_t ep, uint8_t out[AGX3_STATIC_SPILL_MAX_LENGTH], uint8_t *length);
bool agx3_static_spill_decode_load(enum agx_isa_encoding enc,
                                   const uint8_t *bytes, size_t length,
                                   struct agx3_static_spill_load_desc *out);

#ifdef __cplusplus
}
#endif
