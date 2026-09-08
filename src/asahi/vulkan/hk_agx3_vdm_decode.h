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
#define HK_AGX3_SHADER_BINARY_VERSION 9u

/* Native VS witnesses export position plus seven float4 user locations. */
#define HK_AGX3_G17P_MAX_USER_WORDS 28u

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
   bool position_only;
   bool has_preamble;
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
 * shader with position and, optionally, up to two ordinary user-varying
 * slots.  System outputs and preambles are separate state shapes. */
bool hk_agx3_g17p_vertex_program_info(
   enum agx_chip chip,
   const struct hk_agx3_g17p_vertex_compile_info *compiler_info,
   struct agx_vdm_state_update_program_info *program);

/*
 * Decode one record from a potentially longer AGX3 VDM stream through the
 * shared Mesa classifier. This is a validation surface only: it does not
 * follow links, inspect PPP payloads, emit commands, submit work, admit an
 * AGX3 device, or expose a Vulkan capability.
 */
bool hk_agx3_decode_vdm_record(enum agx_chip chip, const void *command,
                               size_t available,
                               struct agx_vdm_decoded_record *record);

/* Validation-only structural bridge. */
bool hk_agx3_encode_vdm_state_update_template(
   enum agx_chip chip, const struct agx_vdm_state_update_template_info *info,
   struct agx_vdm_state_update *out);

/* Exact AGX3 VertexProgramVariant selector bridge used by live draw emission. */
bool hk_agx3_encode_vdm_state_update_draw(
   enum agx_chip chip, const struct agx_vdm_state_update_draw_info *info,
   struct agx_vdm_state_update *out);

#ifdef __cplusplus
}
#endif
