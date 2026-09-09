/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Write the 0x40-byte program immediately before a G17P container entry.
 * Apple's HAL200 compiler emits distinct vertex/compute, plain-fragment, and
 * sampled-fragment programs.  sampled_fragment is valid only for fragment. */
#define AGX3_CONTAINER_CONSTANT_PROGRAM_SIZE 0x40u
void agx3_build_container_constant_program(
   uint8_t image[AGX3_CONTAINER_CONSTANT_PROGRAM_SIZE], bool fragment,
   bool sampled_fragment);

/* A CODE block has its size word 0x80 bytes before the main entry and a
 * 0x40-byte constant program immediately before it. Blocks can be packed in
 * a shared code allocation; the independent 0x340-byte allocation lead is
 * deliberately not part of this helper. */
#define AGX3_CODE_BLOCK_HEADER_SIZE 0x40u
#define AGX3_CODE_BLOCK_ENTRY_OFFSET 0x80u
size_t agx3_code_block_size(size_t main_span);
bool agx3_build_code_block(void *image, size_t capacity, const void *main,
                          size_t main_size, size_t main_span,
                          const uint8_t *constant_program,
                          bool fragment);
