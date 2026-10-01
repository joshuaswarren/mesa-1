/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>
#include "util/format/u_formats.h"
#include "nir.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AGX_MAX_ATTRIBS (16)
#define AGX_MAX_VBUFS   (16)

enum agx_robustness_level {
   /* No robustness */
   AGX_ROBUSTNESS_DISABLED,

   /* Invalid load/store must not fault, but undefined value/effect */
   AGX_ROBUSTNESS_GLES,

   /* Invalid load/store access something from the array (or 0) */
   AGX_ROBUSTNESS_GL,

   /* Invalid loads return 0 and invalid stores are dropped */
   AGX_ROBUSTNESS_D3D,
};

struct agx_robustness {
   enum agx_robustness_level level;

   /* Whether hardware "soft fault" is enabled. */
   bool soft_fault;
};

struct agx_velem_key;

struct agx_vbo_binding {
   uint32_t offset;
   uint8_t binding;
   uint8_t pad[3];
};

/* A contiguous pair of word-sized attributes, read through the first
 * attribute's existing CPU-adjusted base. first/count are in 32-bit words.
 * A zero count keeps the ordinary fetch. */
struct agx_vbo_group {
   uint8_t anchor;
   uint8_t first;
   uint8_t count;
};

void agx_vbo_group_32bit(const struct agx_velem_key *attribs,
                         const struct agx_vbo_binding *bindings, uint16_t valid,
                         struct agx_vbo_group *groups);

void agx_vbo_group_mixed(const struct agx_velem_key *attribs,
                         const struct agx_vbo_binding *bindings, uint16_t valid,
                         struct agx_vbo_group *groups);

bool agx_nir_lower_vbo_raw_grouped(nir_shader *shader,
                                  const struct agx_velem_key *attribs,
                                  struct agx_robustness rs, bool dynamic_strides,
                                  const struct agx_vbo_group *groups);

bool agx_nir_lower_vbo(nir_shader *shader, const struct agx_velem_key *attribs,
                       struct agx_robustness rs, bool dynamic_strides);

bool agx_nir_lower_vbo_raw(nir_shader *shader, const struct agx_velem_key *attribs,
                           struct agx_robustness rs, bool dynamic_strides);

enum pipe_format agx_vbo_internal_format(enum pipe_format format);

bool agx_vbo_supports_format(enum pipe_format format);

#ifdef __cplusplus
} /* extern C */
#endif
