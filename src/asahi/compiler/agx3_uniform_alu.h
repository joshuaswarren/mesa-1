/* SPDX-License-Identifier: MIT */
#pragma once

#include <stdbool.h>
#include <stdint.h>
#include "asahi/lib/agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

enum agx3_uniform_float_op {
   AGX3_UNIFORM_FADD,
   AGX3_UNIFORM_FMUL,
   AGX3_UNIFORM_FMA,
};

enum agx3_uniform_float_kind {
   AGX3_FLOAT_IMMEDIATE = 1,
   AGX3_FLOAT_REGISTER = 2,
   AGX3_FLOAT_UNIFORM = 3,
};

struct agx3_uniform_float_src {
   enum agx3_uniform_float_kind kind;
   uint16_t value;
   bool abs, neg;
   bool cache, discard;
};

/* Native constant-program f32 ALU -> uniform-file destination. The initial
 * witness set admits aligned destinations below half128, plain GPR/immediate
 * inputs and the ordinary ALU wait fields. It does not admit half results,
 * saturation, modifiers, or uniform-file source ports. Failure is transactional.
 */
bool agx3_uniform_float_store_encode(enum agx_isa_encoding encoding,
                                     enum agx3_uniform_float_op op,
                                     uint16_t uniform_half,
                                     const struct agx3_uniform_float_src src[3],
                                     uint8_t wait_mask,
                                     uint8_t output[16], uint8_t *length);

/* Scalar f32 arithmetic with native uniform operands. Sources are in ordinary
 * expression order, unlike the historical falu helper's reversed multipliers.
 * Immediates are minifloat codes. Uniform indices name 16-bit halves.
 */
bool agx3_uniform_float_encode(enum agx_isa_encoding encoding,
                               enum agx3_uniform_float_op op, uint16_t dest,
                               const struct agx3_uniform_float_src src[3],
                               bool saturate, bool dest_cache,
                               uint8_t wait_mask, uint8_t output[16],
                               uint8_t *length);

/* Same-width scalar half arithmetic. Mixed widths retain the existing paths. */
bool agx3_uniform_half_encode(enum agx_isa_encoding encoding,
                              enum agx3_uniform_float_op op, uint16_t dest,
                              const struct agx3_uniform_float_src src[3],
                              bool saturate, bool dest_cache, uint8_t wait_mask,
                              uint8_t output[16], uint8_t *length);

/* Tied minNum/maxNum with one GPR and one uniform, as emitted for clamp. */
bool agx3_uniform_minmax_encode(enum agx_isa_encoding encoding, uint16_t dest,
                                uint16_t reg, uint16_t uniform, bool half,
                                bool maximum, bool cache, bool discard,
                                bool dest_cache, uint8_t wait_mask,
                                uint8_t output[16], uint8_t *length);

#ifdef __cplusplus
}
#endif
