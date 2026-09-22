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

/* Return the IMGWBLK selector of one target in Apple's two-to-four-target
 * G17P EOT builtin, or zero if the format has no such target. The builtin
 * factory derives each target's selector from that target's format alone,
 * so it equals the selector of the format's single-target program. Formats
 * whose single-target program has a different shape are excluded. */
uint8_t agx_g17p_mrt_eot_selector(enum pipe_format format);

enum pipe_format agx_g17p_rt0_pbe_format(enum pipe_format format);

#ifdef __cplusplus
}
#endif
