/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Bounded AGX3 subgroup encoder over the pinned no-load G15G/G16G/G17S
 * control corpus.  The API exposes only register-source f32 shapes with no
 * unresolved scoreboard dependency.  Unsupported operand kinds, conditions,
 * widths, prefix operations and marker profiles fail closed.
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "asahi/lib/agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AGX3_SUBGROUP_MAX_LENGTH 16u

enum agx3_subgroup_shuffle_op {
   AGX3_SUBGROUP_SHUFFLE_BROADCAST = 0,
   AGX3_SUBGROUP_SHUFFLE_XOR,
   AGX3_SUBGROUP_SHUFFLE_UP,
   AGX3_SUBGROUP_SHUFFLE_DOWN,
   AGX3_SUBGROUP_SHUFFLE_NUM,
};

enum agx3_subgroup_lane_kind {
   /* An immediate lane index, in f29 with the class selector f37 = 1. Apple
    * emits this for the bounded SIMD and quad witnesses accepted below. */
   AGX3_SUBGROUP_LANE_IMMEDIATE = 1,
   AGX3_SUBGROUP_LANE_REGISTER = 2,
   AGX3_SUBGROUP_LANE_UNIFORM = 3,
};

struct agx3_subgroup_shuffle_desc {
   bool quad;
   enum agx3_subgroup_shuffle_op op;
   enum agx3_subgroup_lane_kind lane_kind;
   uint16_t dest;
   uint16_t src;
   uint16_t lane;
   uint16_t wait_mask;
   /* f11, the one-bit general-destination hint. Only consulted on the
    * immediate-lane path; the register and uniform paths keep their own
    * corpus constants pinned. */
   bool dest_hint;
   /* The G17P immediate quad form uses f25 while the source remains live and
    * f27 on its final read.  This is ignored by the older SIMD witnesses. */
   bool src_live;
};

enum agx3_subgroup_ballot_type {
   AGX3_SUBGROUP_BALLOT_INTEGER = 0,
   AGX3_SUBGROUP_BALLOT_FLOAT,
};

struct agx3_subgroup_ballot_desc {
   bool quad;
   enum agx3_subgroup_ballot_type type;
   uint16_t dest;
   uint16_t src;
   uint8_t condition;
   bool invert;
   uint16_t wait_mask;
};

enum agx3_subgroup_reduce_op {
   AGX3_SUBGROUP_REDUCE_AND = 0,
   AGX3_SUBGROUP_REDUCE_FADD,
   AGX3_SUBGROUP_REDUCE_OR,
   AGX3_SUBGROUP_REDUCE_FMUL,
   AGX3_SUBGROUP_REDUCE_XOR,
   AGX3_SUBGROUP_REDUCE_FMIN,
   AGX3_SUBGROUP_REDUCE_FMAX,
   AGX3_SUBGROUP_REDUCE_IADD,
   AGX3_SUBGROUP_REDUCE_SMIN,
   AGX3_SUBGROUP_REDUCE_SMAX,
   AGX3_SUBGROUP_REDUCE_UMIN,
   AGX3_SUBGROUP_REDUCE_UMAX,
   AGX3_SUBGROUP_REDUCE_NUM,
};

struct agx3_subgroup_reduce_desc {
   bool quad;
   bool prefix;
   enum agx3_subgroup_reduce_op op;
   uint16_t dest;
   uint16_t src;
   uint16_t wait_mask;
   /* f11, the general-destination hint. */
   bool dest_hint;
   /*
    * Float reductions carry a two-way liveness selector on their source:
    * f29 = 1 when this is the last read, f27 = 1 when the value stays live.
    * Both are witnessed on g17p -- f29 in the `o[t] = simd_sum(a[t])` probes,
    * where the source dies, and f27 in the corpus csimd shader, where it is
    * reduced twice. The default (false) keeps the f29 the previously pinned
    * corpus expects.
    */
   bool src_live;
};

bool agx3_subgroup_encode_shuffle(enum agx_isa_encoding enc,
                                  const struct agx3_subgroup_shuffle_desc *desc,
                                  uint8_t out[AGX3_SUBGROUP_MAX_LENGTH],
                                  uint8_t *length);

/* The recovered ballot subset compares a register against exact zero. */
bool agx3_subgroup_encode_ballot(enum agx_isa_encoding enc,
                                 const struct agx3_subgroup_ballot_desc *desc,
                                 uint8_t out[AGX3_SUBGROUP_MAX_LENGTH],
                                 uint8_t *length);

bool agx3_subgroup_encode_reduce(enum agx_isa_encoding enc,
                                 const struct agx3_subgroup_reduce_desc *desc,
                                 uint8_t out[AGX3_SUBGROUP_MAX_LENGTH],
                                 uint8_t *length);

#ifdef __cplusplus
}
#endif
