/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "agx3_ioto_uniform_program.h"
#include "asahi/lib/agx3_compute_program_core.h"

#define AGX3_G17P_COMPUTE_PROGRAM_MAX_SIZE 6144u
#define AGX3_G17P_COMPUTE_MAX_LOAD_CHUNKS 128u
/* Largest thread-private spill frame a G17P program publishes, in bytes.
 * applegpu_g17p holding 60 live uint4 temporaries publishes 1584. */
#define AGX3_G17P_MAX_SPILL_B 4096u

struct agx3_g17p_compute_program_desc {
   uint64_t program_va;
   uint64_t shader_va;
   const struct agx3_ioto_uniform_range *uniform_ranges;
   size_t uniform_range_count;
   uint32_t shader_binary_size;
   uint32_t scratch_size;
   uint16_t nr_gprs;
   bool compiler_d_valid;
   uint32_t compiler_d;
   uint32_t threadgroup_memory_size_B;
   struct agx3_cdm_direct_template_sources launch_sources;
   enum agx_ioto_g17p_compute_ldshdr_f1 ldshdr_f1;
};

struct agx3_g17p_compute_program {
   uint8_t bytes[AGX3_G17P_COMPUTE_PROGRAM_MAX_SIZE];
   uint16_t length;
   struct agx3_compute_program launch;
};

bool agx3_g17p_compute_program_generate(
   enum agx_chip chip, const struct agx3_g17p_compute_program_desc *desc,
   struct agx3_g17p_compute_program *program);
