/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * AGX3 scalar SCS load/store encoders and decoders.
 *
 * The supported address-space(5) forms were recovered with controlled Apple
 * G17P/G17S compiler probes.  Four address kinds are represented, spanning
 * every combination Apple's G17P ray-query setup emits: direct-absolute, a
 * scalar register index plus a uniform-register base, a scalar register index
 * plus a register base, and a register base with no index -- each with a
 * signed 16-bit byte displacement.  Loads and stores may move one, two or four
 * 32-bit components.
 *
 * The two register-base kinds and any component mask other than one are
 * witnessed on AGX3b only and fail closed on AGX3c.  Other scales still fail
 * closed.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "agx3_mem_core.h"
#include "asahi/lib/agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AGX3_SCS_MAX_LENGTH 16u

enum agx3_scs_address_kind {
   /* base 0 + index 0 + byte_offset */
   AGX3_SCS_ADDRESS_DIRECT = 0,

   /* uniform[uniform_base] + gpr[index] * (1 << width) + byte_offset */
   AGX3_SCS_ADDRESS_UNIFORM_INDEXED = 1,

   /* gpr[base_register] + gpr[index] + byte_offset.  Apple's operand-kind
    * selectors read f23=2 (slot A register) and f38=2 (slot B register); the
    * base lands in f31 rather than the uniform field f32. */
   AGX3_SCS_ADDRESS_REGISTER_INDEXED = 2,

   /* gpr[base_register] + byte_offset, no register index (f23=1, f38=2). */
   AGX3_SCS_ADDRESS_REGISTER_BASE = 3,
};

struct agx3_scs_desc {
   uint16_t data; /* load destination / store value, half-register units */
   enum agx3_mem_width width;
   int16_t byte_offset;

   enum agx3_scs_address_kind address_kind;
   uint16_t index;        /* scalar GPR, half-register units */
   uint16_t uniform_base; /* scalar uniform, half-register units */

   /* Slot-B base register (Apple f31), half-register units.  Used only by the
    * two register-base kinds; must be zero otherwise.  Witnessed range is one
    * byte: the composer refuses 256 and above. */
   uint16_t base_register;

   /* Apple f68.  A four-bit field: the composer accepts every value 1..15.
    * Apple's ray-setup code uses only 1, 3 and 15, and those move one, two and
    * four 32-bit components respectively.  Zero is treated as 1 so that a
    * zero-initialised descriptor keeps the previous scalar behaviour, and the
    * encoding is byte-identical to what this core emitted before. */
   uint8_t component_mask;

   /* Apple f70.  The composer accepts exactly 0..4 and refuses 5 and above.
    * Its role is NOT established: it varies independently of `width` inside a
    * single address kind (uniform-indexed W32 stores are witnessed with both
    * f70=0 and f70=2), so it is deliberately not named a scale or a mode.
    * When `has_address_selector` is false the previous derivation from
    * `width` is kept for the two original address kinds; the two register-base
    * kinds always require an explicit value. */
   bool has_address_selector;
   uint8_t address_selector;

   uint8_t slot;       /* load only: 1..6 (b), 1..8 (c) */
   uint16_t wait_mask; /* bit i waits for scoreboard slot i + 1 */
   bool value_kill;    /* store only; witnessed only on AGX3c */

   /* AGX3c narrow loads may zero-extend to the 32-bit destination.  This is
    * exact only for uniform-indexed W8/W16 loads; every other use fails.
    */
   bool zero_extend;

   /* AGX3c ldscs f76=2.  Its placement is exact but policy is not; normal
    * lowering leaves this clear while decode/replay preserves Apple vectors.
    */
   bool c_load_marker;
};

bool agx3_scs_encode_load(enum agx_isa_encoding enc,
                          const struct agx3_scs_desc *desc,
                          uint8_t out[AGX3_SCS_MAX_LENGTH], uint8_t *length);
bool agx3_scs_encode_store(enum agx_isa_encoding enc,
                           const struct agx3_scs_desc *desc,
                           uint8_t out[AGX3_SCS_MAX_LENGTH], uint8_t *length);

#ifdef __cplusplus
}
#endif
