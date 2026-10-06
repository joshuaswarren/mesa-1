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

/* Capacity of every per-attribute key, bitmask and root table (vertex
 * attributes read by a shader, compacted into consecutive slots) and of the
 * vertex buffer binding tables. This is the Vulkan-visible ceiling: an
 * attribute slot index must always be below AGX_MAX_ATTRIBS before it indexes
 * one of these arrays.
 *
 * The number of attributes a device actually advertises is smaller when the
 * vertex prolog register ABI cannot hold them all; see
 * agx_max_vertex_attribs() in agx_compile.h, which hk uses for
 * maxVertexInputAttributes. DXVK's D3D9 fixed-function vertex shader reads 18
 * attributes (dxvk d3d9 BindInputLayout maps every fixed-function input
 * signature entry to an attribute, binding the unused ones to a null stream),
 * and other translation layers assume at least 32, so 16 was too small.
 */
#define AGX_MAX_ATTRIBS (32)
#define AGX_MAX_VBUFS   (32)

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

bool agx_nir_lower_vbo(nir_shader *shader, const struct agx_velem_key *attribs,
                       struct agx_robustness rs, bool dynamic_strides);

bool agx_nir_lower_vbo_raw(nir_shader *shader, const struct agx_velem_key *attribs,
                           struct agx_robustness rs, bool dynamic_strides);

enum pipe_format agx_vbo_internal_format(enum pipe_format format);

bool agx_vbo_supports_format(enum pipe_format format);

#ifdef __cplusplus
} /* extern C */
#endif
