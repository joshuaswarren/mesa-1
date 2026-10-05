/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Frontend-neutral capability policy for the native AGX D3D12 renderer.
 *
 * Keep this header free of COM, WDK and Mesa types.  Both the public D3D12
 * COM frontend and the WDDM DDI frontend translate this single policy into
 * their respective ABI structures.
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define D12_CORE_FEATURE_LEVEL_11_0 UINT32_C(0x0000b000)
#define D12_CORE_FEATURE_LEVEL_11_1 UINT32_C(0x0000b100)
#define D12_CORE_FEATURE_LEVEL_12_0 UINT32_C(0x0000c000)
#define D12_CORE_FEATURE_LEVEL_12_1 UINT32_C(0x0000c100)
#define D12_CORE_SHADER_MODEL_6_0 UINT32_C(0x00000060)
#define D12_CORE_SHADER_MODEL_6_2 UINT32_C(0x00000062)
#define D12_CORE_COMMAND_LIST_SUPPORT_DIRECT UINT32_C(0x1)
#define D12_CORE_COMMAND_LIST_SUPPORT_COMPUTE UINT32_C(0x4)
#define D12_CORE_COMMAND_LIST_SUPPORT_COPY UINT32_C(0x8)

struct d12_core_caps {
   uint32_t max_feature_level;
   uint32_t max_shader_model;
   uint32_t max_root_signature_version;
   uint32_t sample_counts;
   uint32_t max_sampler_heap_size;
   uint32_t max_view_heap_size;
   bool dynamic_depth_bias;
   uint32_t node_count;
   uint32_t resource_binding_tier;
   uint32_t tiled_resources_tier;
   uint32_t resource_heap_tier;
   /* D3D12_RENDER_PASS_TIER. Tier 1 means the driver implements render
    * passes; Tier 2 additionally promises UAV writes inside a pass cost
    * nothing extra. */
   uint32_t render_passes_tier;
   /* D3D12_HEAP_TYPE_GPU_UPLOAD: memory the CPU writes and the GPU reads as
    * if it were a DEFAULT heap. On a UMA part that is what every allocation
    * already is. */
   bool gpu_upload_heap;
   /* D3D12_PROGRAMMABLE_SAMPLE_POSITIONS_TIER. Tier 1 is one pattern for the
    * whole render target; tier 2 varies it per pixel. */
   uint32_t programmable_sample_positions_tier;
   /* D3D12_MESH_SHADER_TIER. Tier 1 is the whole feature; there is no tier 2. */
   uint32_t mesh_shader_tier;
   /* 64-bit integer shader operations. AGX executes iadd64 and imul_2x32_64
    * natively and lowers the rest (agx_compile.h lower_int64_options). */
   bool int64_shader_ops;
   /* Front and back stencil may carry independent reference values and
    * read/write masks. */
   bool independent_front_and_back_stencil;
   /* D3D_PRIMITIVE_TOPOLOGY_TRIANGLEFAN. */
   bool triangle_fan;
   /* IASetIndexBufferStripCutValue outside the pipeline state. */
   bool dynamic_index_buffer_strip_cut;
   /* D3D12_SAMPLER_FLAG_NON_NORMALIZED_COORDINATES. */
   bool non_normalized_coordinate_samplers;
   uint32_t gpu_virtual_address_bits;
   bool uma;
   bool cache_coherent_uma;
   bool isolated_mmu;
   bool tile_based_deferred_renderer;
   bool native_16bit_shader_ops;
   bool typed_uav_load_additional_formats;
   bool full_texture_sampling;
   bool wave_ops;
   uint32_t wave_lane_count_min;
   uint32_t wave_lane_count_max;
   uint32_t total_lane_count;
   bool predication;
   bool hardware_copy;
   bool copy_queue_timestamp_queries;
   bool texture_copy_between_dimensions;
   /* D3D12_BLEND_ALPHA_FACTOR and its inverse consume component A from the
    * four-component OMSetBlendFactor state. */
   bool alpha_blend_factor;
   bool output_merger_logic_op;
   uint32_t write_buffer_immediate_support_flags;
};

/* Inputs copied from the opened AGX device, never inferred from a model name.
 * Compiler bits describe connected translation/backend paths, not silicon. */
struct d12_core_caps_device_info {
   uint32_t generation;
   uint32_t active_cores;
   uint64_t vm_start;
   uint64_t vm_end;
   bool isolated_mmu;
   bool cache_coherent_uma;
   bool compiler_wave_ops;
   bool compiler_int64;
   bool compiler_native_16bit;
};

void d12_core_caps_init(struct d12_core_caps *caps,
                        const struct d12_core_caps_device_info *info);
bool d12_core_caps_supports_feature_level(const struct d12_core_caps *caps,
                                          uint32_t level);
uint32_t d12_core_caps_choose_feature_level(const struct d12_core_caps *caps,
                                           const uint32_t *requested,
                                           uint32_t count);

/* Conservative allocation policy for frontends without an opened adapter.
 * CheckFeatureSupport must use the caps belonging to its device. */
const struct d12_core_caps *d12_core_get_caps(void);

/*
 * The WDDM DDI names feature levels with its own small enumeration
 * (D3D12DDI_3DPIPELINELEVEL) rather than the D3D_FEATURE_LEVEL numbers the
 * COM frontend uses. Translate once so both frontends report the single
 * capability policy instead of drifting apart.
 */
#define D12_CORE_PIPELINE_LEVEL_11_0 10u
#define D12_CORE_PIPELINE_LEVEL_11_1 11u
#define D12_CORE_PIPELINE_LEVEL_12_0 12u
#define D12_CORE_PIPELINE_LEVEL_12_1 13u

static inline uint32_t
d12_core_pipeline_level(uint32_t feature_level)
{
   if (feature_level >= D12_CORE_FEATURE_LEVEL_12_1)
      return D12_CORE_PIPELINE_LEVEL_12_1;
   if (feature_level >= D12_CORE_FEATURE_LEVEL_12_0)
      return D12_CORE_PIPELINE_LEVEL_12_0;
   if (feature_level >= D12_CORE_FEATURE_LEVEL_11_1)
      return D12_CORE_PIPELINE_LEVEL_11_1;
   return D12_CORE_PIPELINE_LEVEL_11_0;
}

bool d12_core_supports_feature_level(uint32_t minimum_feature_level);

uint32_t d12_core_choose_feature_level(const uint32_t *requested,
                                       uint32_t requested_count);

#ifdef __cplusplus
}
#endif
