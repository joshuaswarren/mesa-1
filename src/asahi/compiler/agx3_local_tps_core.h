/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * AGX3 threadgroup and thread-private-storage encoders.
 *
 * These descriptors cover the role-complete ldtg/sttg and ldtps/sttps
 * families recovered from Apple G15G/G16G/G17S compiler output.  SCS is
 * deliberately absent: ldscs/stscs are RT-only and their operands are not
 * grounded.  Evidence: GPU_AGX3_LOCAL_STACK_WAIT_2026-08-24.md.
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "agx3_mem_core.h"
#include "asahi/lib/agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AGX3_LOCAL_TPS_MAX_LENGTH 16u

enum agx3_local_base_kind {
   AGX3_LOCAL_BASE_IMMEDIATE = 1,
   AGX3_LOCAL_BASE_REGISTER = 2,
   AGX3_LOCAL_BASE_UNIFORM = 3,
};

struct agx3_local_desc {
   uint16_t data; /* load destination / store value, half-register units */
   enum agx3_mem_width width;
   uint8_t mask;
   uint8_t shift;

   enum agx3_local_base_kind base_kind;
   uint16_t base;
   bool base_kill;

   bool index_is_register;
   uint16_t index;
   bool index_kill;
   uint16_t byte_offset;

   bool value_kill; /* store only */
   uint8_t slot;    /* load only: 1..6 (b), 1..8 (c) */
   uint16_t wait_mask;

   /* Structural Apple fields whose placement is grounded but whose policy is
    * not.  Callers normally leave these clear; decoders/replayers may preserve
    * witnessed values exactly. */
   bool narrow_extend;
   bool c_load_marker;
};

bool agx3_local_encode_load(enum agx_isa_encoding enc,
                            const struct agx3_local_desc *d,
                            uint8_t out[AGX3_LOCAL_TPS_MAX_LENGTH],
                            uint8_t *len);
bool agx3_local_encode_load_form(enum agx_isa_encoding enc,
                                 const struct agx3_local_desc *d, uint8_t ep,
                                 uint8_t out[AGX3_LOCAL_TPS_MAX_LENGTH],
                                 uint8_t *len);
bool agx3_local_encode_store(enum agx_isa_encoding enc,
                             const struct agx3_local_desc *d,
                             uint8_t out[AGX3_LOCAL_TPS_MAX_LENGTH],
                             uint8_t *len);
bool agx3_local_encode_store_form(enum agx_isa_encoding enc,
                                  const struct agx3_local_desc *d, uint8_t ep,
                                  uint8_t out[AGX3_LOCAL_TPS_MAX_LENGTH],
                                  uint8_t *len);

struct agx3_tps_desc {
   uint16_t data; /* load destination / store value, half-register units */
   enum agx3_mem_width width;
   uint8_t mask;
   uint8_t shift;
   uint16_t byte_offset;

   /* All ldtps/sttps witnesses use the grounded dynamic 32-bit index path.
    * Static spills use the separate copy/rld remote-file forms. */
   uint16_t index;
   bool index_kill;
   bool index_signed;

   uint8_t frame_region; /* witnessed 1..6; exact taxonomy remains unknown */
   bool value_kill;      /* store only */
   bool narrow_extend;
   bool structural_bit53;
   bool retain_tps; /* load only: later reads may alias this location */
   uint8_t slot; /* load only */
   uint16_t wait_mask;
};

bool agx3_tps_encode_load(enum agx_isa_encoding enc,
                          const struct agx3_tps_desc *d,
                          uint8_t out[AGX3_LOCAL_TPS_MAX_LENGTH], uint8_t *len);
bool agx3_tps_encode_load_form(enum agx_isa_encoding enc,
                               const struct agx3_tps_desc *d, uint8_t ep,
                               uint8_t out[AGX3_LOCAL_TPS_MAX_LENGTH],
                               uint8_t *len);
bool agx3_tps_encode_store(enum agx_isa_encoding enc,
                           const struct agx3_tps_desc *d,
                           uint8_t out[AGX3_LOCAL_TPS_MAX_LENGTH], uint8_t *len);
bool agx3_tps_encode_store_form(enum agx_isa_encoding enc,
                                const struct agx3_tps_desc *d, uint8_t ep,
                                uint8_t out[AGX3_LOCAL_TPS_MAX_LENGTH],
                                uint8_t *len);

#ifdef __cplusplus
}
#endif
