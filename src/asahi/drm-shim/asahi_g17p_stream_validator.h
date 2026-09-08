/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

struct drm_asahi_cmd_render;
struct drm_asahi_cmd_compute;

struct asahi_g17p_stream_memory {
   void *data;
   bool (*read)(void *data, uint64_t address, void *dst, size_t size);
   bool (*owned)(void *data, uint64_t address, size_t size);
   bool (*span)(void *data, uint64_t address, size_t *available);
   bool verbose;
};

struct asahi_g17p_stream_stats {
   unsigned draws;
   unsigned indexed_draws;
   unsigned state_updates;
   unsigned links;
   unsigned viewport_count;
   unsigned max_instance_count;
   uint32_t primitive_mask;
   uint8_t index_size_mask;
   bool clear_only;
   bool body_only;
   bool restart_enabled;
   bool restart_disabled;
   bool indirect_nonindexed;
   bool indirect_indexed;
   bool varying_fs;
   bool no_varying_fs;
   bool viewport_changed;
   bool multi_viewport;
   bool two_viewport;
   bool scissor_changed;
   bool cull_enabled;
   bool depth_enabled;
   bool stencil_enabled;
   bool blend_enabled;
   bool rasterizer_discard;
   bool layer_output;
   uint64_t fragment_code;
};

struct asahi_g17p_stream_config {
   uint64_t root;
   uint64_t scissor_base;
   uint64_t depth_bias_base;
};

struct asahi_g17p_compute_stats {
   unsigned launches;
   unsigned barriers;
   unsigned strong_barriers;
   bool terminated;
};

bool asahi_g17p_stream_validate(
   const struct asahi_g17p_stream_memory *memory,
   const struct asahi_g17p_stream_config *config,
   struct asahi_g17p_stream_stats *stats);

bool asahi_g17p_render_aux_validate(
   const struct asahi_g17p_stream_memory *memory,
   const struct drm_asahi_cmd_render *render);

bool asahi_g17p_compute_validate(
   const struct asahi_g17p_stream_memory *memory,
   const struct drm_asahi_cmd_compute *compute,
   struct asahi_g17p_compute_stats *stats);
