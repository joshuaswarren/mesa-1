/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Software capacity for one fixed register domain, not a staging-file size.
 * SSA dependencies remain the caller's responsibility. Tokens identify the
 * smallest independently aliased units (uniform halves for the first caller).
 */
#define AGX3_FIXED_DEP_MAX_TOKENS 512u

struct agx3_fixed_access {
   uint32_t instruction;
   uint16_t first;
   uint16_t count;
   bool write;
};

typedef void (*agx3_fixed_dependency_cb)(void *data, uint32_t before,
                                        uint32_t after);

/* Accesses are in original instruction order; multiple accesses may share
 * an instruction. Emit RAW, WAR and WAW dependencies, never read/read edges.
 * The callback may receive a duplicate pair for overlapping multi-half values.
 * Invalid input/allocation failure returns false without invoking callbacks.
 * This is scheduling order only; asynchronous completion belongs to the
 * separate scoreboard and staging-lifetime model.
 */
bool agx3_fixed_dependencies(const struct agx3_fixed_access *accesses,
                             size_t count, unsigned tokens,
                             agx3_fixed_dependency_cb callback, void *data);

#ifdef __cplusplus
}
#endif
