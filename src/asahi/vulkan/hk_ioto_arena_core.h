/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * G17P Ioto programs are tens to a few hundred bytes each, and one is composed
 * per shader per draw and per background/end-of-tile program per render pass.
 * Giving each its own BO costs a whole 16KiB page plus a 16KiB guard page of
 * the 4GiB USC exec window -- more than a hundred times the program -- so a
 * single frame's worth of draws can exhaust the window even though the code it
 * holds is a few hundred kilobytes.
 *
 * Programs are therefore bump-allocated out of chunks, at the 64-byte alignment
 * the launch pointer requires. 128KiB matches HK_CMD_BO_SIZE and holds several
 * hundred programs, so the amortised window cost per program is the program
 * itself plus alignment.
 */
#define HK_IOTO_ARENA_CHUNK_SIZE (128u * 1024u)

struct hk_ioto_arena_span {
   /* Byte offset of the program within the chunk. */
   uint32_t offset;

   /* Bump cursor once the program has been placed. */
   uint32_t next_offset;
};

/*
 * Size of the chunk to request so that a program of `program_size` bytes at
 * `align` fits. Returns 0 for an unrepresentable request.
 */
size_t hk_ioto_arena_chunk_size(size_t program_size, uint32_t align);

/*
 * Place `program_size` bytes at `align` in a chunk of `capacity` bytes whose
 * bump cursor is at `cursor`. Returns false -- leaving *span untouched -- when
 * the program does not fit, which is the caller's signal to start a new chunk.
 */
bool hk_ioto_arena_place(uint32_t cursor, uint32_t capacity,
                         size_t program_size, uint32_t align,
                         struct hk_ioto_arena_span *span);

#ifdef __cplusplus
}
#endif
