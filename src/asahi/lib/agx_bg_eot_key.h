/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "asahi/compiler/agx_compile.h"
#include "util/format/u_formats.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Bind the logical RT0 format to the exact native G17P EOT selector key. */
void agx_g17p_configure_rt0_eot_key(enum pipe_format format,
                                    struct agx_shader_key *key);

#ifdef __cplusplus
}
#endif
