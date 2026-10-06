/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT */
#pragma once

#include <wsl/winadapter.h>
#include "agx_compile.h"
#include "agx3_compute_program.h"

struct d12_device;
struct agx_bo;

/* One eight-byte source cell per four-half driver slot. The cells hold
 * pointer values, except DRAW (two u32 values) and VS_OUTPUTS (a u64 mask).
 * Addresses passed here are the cell addresses, never their contents. */
#define D12_NATIVE_COMPUTE_CELLS 20
#define D12_NATIVE_COMPUTE_RANGES (D12_NATIVE_COMPUTE_CELLS + 1)

enum d12_native_stage {
   D12_NATIVE_SW_VS,
   D12_NATIVE_GS_COUNT,
   D12_NATIVE_GS_PRE,
   D12_NATIVE_GS_MAIN,
   D12_NATIVE_SW_TES,
   D12_NATIVE_TCS,
   D12_NATIVE_MS,
   D12_NATIVE_MS_SETUP,
   D12_NATIVE_AS,
   D12_NATIVE_AS_SETUP,
   D12_NATIVE_STAGE_COUNT,
};

/* BO, CODE entry and full compiler metadata remain in their existing PSO
 * slots. Certification proves CODE/Ioto composition, never GPU admission. */
struct d12_native_compute_abi {
   uint32_t required_cells;
   uint64_t rodata_va;
   bool certified;
};

HRESULT d12_native_compute_code_create(
   struct d12_device *device, const struct agx_shader_part *part,
   struct agx_bo **bo, uint64_t *entry, uint64_t *rodata);

bool d12_native_compute_uniforms(const struct agx_shader_info *info,
                                 uint32_t *required_cells);
bool d12_native_compute_ranges(
   const struct agx_shader_info *info, const struct d12_native_compute_abi *abi,
   const uint64_t cells[D12_NATIVE_COMPUTE_CELLS],
   struct agx3_ioto_uniform_range ranges[D12_NATIVE_COMPUTE_RANGES],
   unsigned *count);
bool d12_native_compute_generate(
   const struct agx_shader_info *info, const struct d12_native_compute_abi *abi,
   uint64_t entry, uint64_t program_va,
   const uint64_t cells[D12_NATIVE_COMPUTE_CELLS],
   struct agx3_g17p_compute_program *program);
