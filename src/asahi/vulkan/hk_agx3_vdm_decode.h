/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Honeykrisp bridge for bounded AGX3 VDM State Update template packing and
 * strict VDM record decoding.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>

#include "asahi/lib/agx_vdm_decode_core.h"

struct blob;
struct blob_reader;

#ifdef __cplusplus
extern "C" {
#endif

/* Bump whenever the Honeykrisp shader payload below changes.  Vulkan shader
 * binaries carry this version outside the driver-private blob, so stale
 * layouts are rejected before any fields are consumed. */
#define HK_AGX3_SHADER_BINARY_VERSION 32u

/* A G17P vertex program writes at most 128 UVS words: Apple's compiler emits
 * position plus thirty-one float4 user locations. Its fragment linker refuses
 * a 128th coefficient. */
#define HK_AGX3_G17P_MAX_UVS_WORDS 128u
#define HK_AGX3_G17P_MAX_USER_WORDS (HK_AGX3_G17P_MAX_UVS_WORDS - 4u)
#define HK_AGX3_G17P_MAX_COEFFICIENTS 127u

bool hk_agx3_shader_binary_version_supported(uint32_t version);

bool hk_agx3_vdm_program_cache_serialize(
   struct blob *blob, const struct agx_vdm_state_update_program_info *info);

bool hk_agx3_vdm_program_cache_deserialize(
   struct blob_reader *blob, struct agx_vdm_state_update_program_info *info);

enum agx_vdm_state_update_source_layout
hk_agx3_vdm_live_source_layout(
   const struct agx_vdm_state_update_program_info *program,
   bool active_context_selector_valid, bool ordinary_draw_path);

struct hk_agx3_g17p_vertex_compile_info {
   bool hardware_vertex;
   bool has_preamble;
   /* The caller compiled the preamble as a native constant program (inputs
    * imported through its TPS frame). Without it a preamble is refused. */
   bool constant_program_allowed;
   /* The constant program's texture staging footprint, published as its
    * thread-invariant largest-IPR bytes (as for the fragment CProg). */
   bool preamble_staging_known;
   uint16_t preamble_staging_halfs;
   uint16_t nr_preamble_gprs;
   uint32_t preamble_scratch_size;
   uint64_t uvs_written;
   uint8_t position_off;
   uint8_t varyings_off;
   uint8_t psiz_off;
   uint8_t layer_viewport_off;
   uint8_t clip_off;
   uint8_t uvs_size;
   uint8_t user_size;
};

/* Build the exact G17P VertexProgramVariant fields for a hardware vertex
 * shader with position, ordinary user locations, and the Vulkan-bounded
 * eight-word combined clip/cull-distance tail. System outputs are a separate
 * state shape. A constant program is admitted only when
 * `constant_program_allowed` is set; its thread-invariant spill (the CProg's
 * imported-input TPS frame) and staging footprint then populate the two
 * state-load counts (agx_vdm_state_update_vertex_ti_resources). */
bool hk_agx3_g17p_vertex_program_info(
   enum agx_chip chip,
   const struct hk_agx3_g17p_vertex_compile_info *compiler_info,
   struct agx_vdm_state_update_program_info *program);

/* Exact AGX3 VertexProgramVariant selector bridge used by live draw emission. */
bool hk_agx3_encode_vdm_state_update_draw(
   enum agx_chip chip, const struct agx_vdm_state_update_draw_info *info,
   struct agx_vdm_state_update *out);

#ifdef __cplusplus
}
#endif
