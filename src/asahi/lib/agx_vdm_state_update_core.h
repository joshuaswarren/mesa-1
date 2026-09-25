/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Bounded, target-aware AGX3 VDM State Update representation.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AGX_VDM_STATE_UPDATE_LENGTH 28u
#define AGX_VDM_STATE_UPDATE_HEADER UINT32_C(0x4000002e)

struct agx_vdm_state_update_info {
   uint32_t raw_vertex_shader_word0;
   uint64_t pipeline;
   uint32_t raw_pipeline_template;
   uint32_t raw_vertex_outputs[2];
   uint32_t raw_vertex_unknown;
};

struct agx_vdm_state_update {
   uint8_t bytes[AGX_VDM_STATE_UPDATE_LENGTH];
   uint8_t length;
};

/* The live renderer selects one of six 28-byte templates.  The low selector
 * bit is boolean; the high axis is the three-valued provoking-vertex value
 * stored by setProvokingVertexMode:. */
enum agx_vdm_state_update_template_variant {
   AGX_VDM_STATE_UPDATE_TEMPLATE_00 = 0,
   AGX_VDM_STATE_UPDATE_TEMPLATE_01 = 1,
   AGX_VDM_STATE_UPDATE_TEMPLATE_10 = 2,
   AGX_VDM_STATE_UPDATE_TEMPLATE_11 = 3,
   AGX_VDM_STATE_UPDATE_TEMPLATE_20 = 4,
   AGX_VDM_STATE_UPDATE_TEMPLATE_21 = 5,
};

/*
 * Source-layout and draw-path admission for the native renderer template
 * producer.  The inspected Apple direct, indexed, indirect, and indexed-
 * indirect entry points all invoke the same state producer before emitting
 * their Index List record.  Tessellation and mesh paths are intentionally not
 * represented here.
 *
 * AGX2 Counts/UVS state is not byte-compatible with this producer.  Callers
 * must therefore name the exact AGX3 VertexProgramVariant layout explicitly;
 * zero/legacy/unknown values fail closed.
 */
enum agx_vdm_state_update_source_layout {
   AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_UNSPECIFIED = 0,
   AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX2_LEGACY = 1,
   AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT = 2,
};

enum agx_vdm_state_update_draw_path {
   AGX_VDM_STATE_UPDATE_DRAW_DIRECT_NONINDEXED = 0,
   AGX_VDM_STATE_UPDATE_DRAW_DIRECT_INDEXED = 1,
   AGX_VDM_STATE_UPDATE_DRAW_INDIRECT_NONINDEXED = 2,
   AGX_VDM_STATE_UPDATE_DRAW_INDIRECT_INDEXED = 3,
};

/*
 * Structural inputs used by Apple's VertexProgramVariant template builder.
 *
 * The two state-load counts and the VsStateConfig fields are deliberately
 * named after their proven producer/position.  Their higher-level compiler
 * meanings are not yet grounded, and they are not interchangeable with the
 * AGX2 Counts, VDM_STATE_VERTEX_OUTPUTS, or VDM_STATE_VERTEX_UNKNOWN words.
 */
struct agx_vdm_state_update_template_info {
   uint64_t pipeline;

   uint16_t raw_state_load_count_12;
   uint32_t raw_state_load_count_18;

   uint8_t raw_vs_output_count_0;
   uint8_t raw_vs_output_count_2;
   bool raw_vs_output_count_1_extra;
   uint8_t raw_vs_output_selector;
   bool raw_vs_output_flag_6;

   /* Pre-HAL300 (G15/G16/G17P) only.  These are respectively
    * `(flags & 0xc) != 0` and `(flags & 0x10) != 0` from VsStateConfig.
    * HAL300 does not consume them and rejects nonzero values rather than
    * silently discarding them. */
   bool pre_hal300_flag_group_nonzero;
   bool pre_hal300_flag_bit_4;

   enum agx_vdm_state_update_template_variant variant;

   /* Raw u32 loaded from the live RenderContext before emission.  The pinned
    * G15X_M1/G16X/G17P/G17G/G17X paths compare it with UINT32_MAX: that
    * sentinel preserves the template's zero final word, while every other
    * value emits `(raw_active_context_selector - 1) & 7`.  No narrower
    * producer domain is assumed here. */
   uint32_t raw_active_context_selector;
};

/* Shader-lifetime portion of the VertexProgramVariant state.  Zero-filled
 * shader objects remain UNSPECIFIED and are rejected.  The draw path replaces
 * `template_info.pipeline` with its freshly uploaded pipeline pointer, ignores
 * `template_info.variant` (selected dynamically), and replaces
 * `template_info.raw_active_context_selector` from render-context state. */
struct agx_vdm_state_update_program_info {
   enum agx_vdm_state_update_source_layout source_layout;
   struct agx_vdm_state_update_template_info template_info;
};

/*
 * Exact renderer-side selector inputs.  The low bit is loaded from
 * RenderPipeline+0x2f0, which the pipeline constructor populates from
 * isRasterizationEnabled.  The provoking selector is the exact stored
 * renderer value, whose accepted domain is 0..2.  The pinned renderer computes
 *
 *   variant = rasterization_enabled |
 *             (raw_provoking_selector << 1)
 *
 * before indexing all six 28-byte VertexProgramVariant templates.
 */
struct agx_vdm_state_update_draw_info {
   enum agx_vdm_state_update_source_layout source_layout;
   enum agx_vdm_state_update_draw_path draw_path;
   bool rasterization_enabled;
   uint8_t raw_provoking_selector;
   struct agx_vdm_state_update_template_info template_info;
};

/*
 * Pack/unpack the complete seven-word AGX3 VDM State Update emitted by the
 * inspected Apple VDMEncoderGen5 and VDMEncoderGen6 paths:
 *
 *   word 0: exact 0x4000002e header
 *   word 1: raw vertex-shader word 0
 *   words 2..3: widened 48-bit vertex-pipeline pair
 *   words 4..5: raw two-word vertex-output payload
 *   word 6: raw vertex-unknown payload
 *
 * Only the pipeline field's 64-byte alignment and 48-bit address range are
 * fully named.  The remaining payloads are preserved losslessly because their
 * complete AGX3 bit semantics are not independently grounded.
 *
 * The target allowlist contains only identities directly served by the
 * inspected Apple Metal plugins, plus G18G, whose J873 native renders emit
 * the G17G HAL300 form byte for byte.  G16P and G17A remain closed despite
 * sharing neighboring command-stream generations.  This helper does not emit
 * a live command or enable lowering, scheduling, submission, or device
 * admission.
 */
bool agx_vdm_state_update_supported(enum agx_chip chip);

/* AGX2 USC pointers are stored relative to `shader_base`, while the AGX3
 * VertexProgramVariant pipeline pair carries an absolute 48-bit GPU VA.
 * Resolve that representation boundary explicitly so live producers cannot
 * accidentally feed the legacy packed offset to the AGX3 encoder. */
bool agx_vdm_state_update_resolve_pipeline_va(uint64_t shader_base,
                                              uint32_t packed_usc,
                                              uint64_t *pipeline_va);

bool agx_vdm_state_update_pack(enum agx_chip chip,
                               const struct agx_vdm_state_update_info *info,
                               struct agx_vdm_state_update *out);

bool agx_vdm_state_update_unpack(enum agx_chip chip, const void *bytes,
                                 size_t size,
                                 struct agx_vdm_state_update_info *info);

/*
 * Materialize/pack the exact six-template path observed in:
 *
 *   G15X_M1, G16X, G17P: VertexProgramVariant + VDMEncoderGen5
 *   G17G, G17X:          VertexProgramVariant + VDMEncoderGen6
 *
 * HAL300 intentionally differs: vertex word 0 has no pre-HAL300 bit-31 rule,
 * and the six per-template output constants are 0x20/0/0x21/1/0x22/2 instead
 * of 0x30/0/0x31/1/0x32/2.  Both calls are transactional and fail closed
 * outside the directly observed target/input domains.
 */
bool agx_vdm_state_update_template_materialize(
   enum agx_chip chip, const struct agx_vdm_state_update_template_info *info,
   struct agx_vdm_state_update_info *raw);

bool agx_vdm_state_update_template_pack(
   enum agx_chip chip, const struct agx_vdm_state_update_template_info *info,
   struct agx_vdm_state_update *out);

/*
 * Select and pack the State Update prefix used by the four native draw paths.
 * This emits only the 28-byte State Update record: dirty restart/index state
 * may legally intervene before the Index List record, so this helper does not
 * claim byte adjacency or append a draw.  The output is cleared on every
 * failure.
 */
bool agx_vdm_state_update_draw_pack(
   enum agx_chip chip, const struct agx_vdm_state_update_draw_info *info,
   struct agx_vdm_state_update *out);

#ifdef __cplusplus
}
#endif
