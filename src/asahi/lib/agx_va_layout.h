/*
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

bool agx_sparse_layout(uint64_t user_start, uint64_t kernel_start,
                       uint64_t *ro_offset, uint64_t *size);

#ifdef __cplusplus
} /* extern "C" */
#endif
