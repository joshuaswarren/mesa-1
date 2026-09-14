/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "agx3_ioto_uniform_program.h"
#include "asahi/lib/agx_ioto_state_core.h"

#define AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE 6144u
/* Software composer capacity, not a hardware field: the Ioto loader bounds a
 * program by its total load chunks and body size, never by how many ranges
 * produced them. So this only has to cover every range the driver's collector
 * can emit, and a value below that refuses a legal program fail-closed.
 *
 * A vertex program is the worst case and can request all seven: VBO bases,
 * VBO clamps, draw parameters, draw ID, indexed texture/sampler heaps,
 * descriptor root, and promoted constants. Firefox/WebGL proves six of those,
 * every one but the draw ID. HK_AGX3_G17P_GRAPHICS_MAX_RANGES aliases this
 * value and hk_cmd_buffer.c static-asserts it against the collector. */
#define AGX3_G17P_GRAPHICS_MAX_RANGES 7u
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
   uint8_t colour_partitions;
   struct agx_ioto_g17p_imageblock imageblock;
};

struct agx3_g17p_graphics_program {
   uint8_t bytes[AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE];
   uint16_t length;
};

enum agx3_g17p_rt0_bg_op {
   AGX3_G17P_RT0_BG_CLEAR,
   AGX3_G17P_RT0_BG_RELOAD,
   AGX3_G17P_RT0_BG_NONE,
   /* Reload the leading targets and clear the rest. */
   AGX3_G17P_RT0_BG_CLEAR_RELOAD,
};

#define AGX3_G17P_RT0_CLEAR_PUSH_HALFS 12u
#define AGX3_G17P_WBOIT_MRT_CLEAR_PUSH_HALFS 14u
#define AGX3_G17P_R11G11B10_MRT_CLEAR_PUSH_HALFS 18u
#define AGX3_G17P_RGBA8_MRT_CLEAR_PUSH_HALFS 20u
#define AGX3_G17P_TRIPLE_CLEAR_PUSH_HALFS 28u
#define AGX3_G17P_QUAD_CLEAR_PUSH_HALFS 36u

bool agx3_g17p_rt0_bg_resource_spec(enum agx_chip chip,
                                    enum agx3_g17p_rt0_bg_op op,
                                    uint64_t *resource_spec);

bool agx3_g17p_rt0_bg_shader_resources_match(enum agx3_g17p_rt0_bg_op op,
                                             unsigned push_count,
                                             bool uses_txf);

struct agx3_g17p_rt0_bg_program_desc {
   uint64_t program_va;
   uint64_t shader_va;
   uint64_t clear_value_va;
   uint64_t texture_pointer_va;
   uint64_t sampler_pointer_va;
   uint32_t shader_binary_size;
   uint32_t scratch_size;
   uint16_t nr_gprs;
   uint16_t staging_halfs;
   uint16_t clear_push_halfs;
   /* CLEAR_RELOAD: uniform half of the first cleared target's value, which
    * clear_value_va addresses. The reloaded targets precede it. */
   uint16_t clear_start_halfs;
   bool staging_known;
   enum agx3_g17p_rt0_bg_op op;
   struct agx_ioto_g17p_imageblock imageblock;
};

struct agx3_g17p_rt0_bg_program {
   uint8_t bytes[AGX_IOTO_G17P_CLEAR_RELOAD_PROGRAM_MAX_SIZE];
   uint16_t length;
};

bool agx3_g17p_rt0_bg_program_generate(
   enum agx_chip chip, const struct agx3_g17p_rt0_bg_program_desc *desc,
   struct agx3_g17p_rt0_bg_program *program);

struct agx3_g17p_rt0_eot_program_desc {
   uint64_t program_va;
   uint64_t shader_va;
   uint64_t pbe_argument_va;
   uint8_t imageblock_offset_B;
   uint8_t attachment_count;
   uint8_t imageblock_offset1_B;
   uint8_t imageblock_offset2_B;
   uint8_t imageblock_offset3_B;
   /* Two to four targets of arbitrary formats, each at its own offset. */
   bool mixed_mrt;
   uint32_t shader_binary_size;
   uint32_t scratch_size;
   uint16_t nr_gprs;
   uint16_t staging_halfs;
   struct agx_ioto_g17p_imageblock imageblock;
};

struct agx3_g17p_rt0_eot_program {
   uint8_t bytes[AGX_IOTO_G17P_QUAD_EOT_PROGRAM_MAX_SIZE];
   uint16_t length;
};

bool agx3_g17p_rt0_eot_program_generate(
   enum agx_chip chip, const struct agx3_g17p_rt0_eot_program_desc *desc,
   struct agx3_g17p_rt0_eot_program *program);

bool agx3_g17p_graphics_program_bound(
   enum agx_chip chip, const struct agx3_ioto_uniform_range *ranges,
   size_t range_count, size_t *size);

bool agx3_g17p_graphics_program_generate(
   enum agx_chip chip, const struct agx3_g17p_graphics_program_desc *desc,
   struct agx3_g17p_graphics_program *program);

/* HAL200 native texture/sampler pointers occupy uniform halves 0..7.  Select
 * the first even eight-half range after every main-shader uniform and rodata
 * range for a fragment blend epilog which also uses that resource prefix.
 * The output is transactional on failure. */
bool agx3_g17p_fragment_blend_constant_base(enum agx_chip chip,
                                             unsigned occupied_halfs,
                                             uint16_t *base);
