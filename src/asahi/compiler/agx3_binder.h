/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Shared "template + override" helper over the generated AGX3 encoder.
 *
 * Most AGX3 forms carry a fixed set of selector/context field values plus a
 * handful of operand fields. A binder starts from a per-form baseline (the
 * constants, recovered by decoding a witnessed instruction), overrides the
 * operand fields with the caller's values, and asks the composer to encode. The
 * composer fails closed on any contradiction, so a wrong override cannot
 * silently produce a valid-looking instruction.
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "agx3_compose/agx3_compose.h"
#include "asahi/lib/agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

struct agx3_field_kv {
   uint16_t field;
   uint64_t value;
};

/*
 * Encode `mnemonic.ep` for `encoding`, starting from `baseline` and applying
 * `overrides` on top (an override for a field already in the baseline wins).
 * On success writes the form's exact byte length into `*len` and the bytes into
 * `out`, and returns true. Returns false for an unknown encoding, an unknown
 * form, or any value the composer refuses.
 */
/* Byte length of form `mnemonic.ep` for `encoding`, or 0 if there is none. */
unsigned agx3_bind_form_size(enum agx_isa_encoding encoding,
                             const char *mnemonic, int ep);

bool agx3_bind_compose(enum agx_isa_encoding encoding, const char *mnemonic,
                       int ep, const struct agx3_field_kv *baseline,
                       unsigned nbaseline,
                       const struct agx3_field_kv *overrides,
                       unsigned noverrides, uint8_t *out, uint8_t *len);

#ifdef __cplusplus
}
#endif
