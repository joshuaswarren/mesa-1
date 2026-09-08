/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "agx3_ioto_uniform_program.h"

#define AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE 6144u
#define AGX3_G17P_GRAPHICS_MAX_RANGES 5u
#define AGX3_G17P_GRAPHICS_MAX_LOAD_CHUNKS 128u

struct agx3_g17p_graphics_program_desc {
   uint64_t program_va;
   uint64_t shader_va;
   const struct agx3_ioto_uniform_range *uniform_ranges;
   size_t uniform_range_count;
   uint32_t shader_binary_size;
   uint32_t scratch_size;
   uint16_t nr_gprs;
   bool compiler_d_valid;
   uint32_t compiler_d;
   bool fragment;
   bool writes_sample_mask;
};

struct agx3_g17p_graphics_program {
   uint8_t bytes[AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE];
   uint16_t length;
};

bool agx3_g17p_graphics_program_bound(
   enum agx_chip chip, const struct agx3_ioto_uniform_range *ranges,
   size_t range_count, size_t *size);

bool agx3_g17p_graphics_program_generate(
   enum agx_chip chip, const struct agx3_g17p_graphics_program_desc *desc,
   struct agx3_g17p_graphics_program *program);

