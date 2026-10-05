/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "d12_core_caps.h"
#include <stddef.h>
#include <string.h>

/* Allocation limits shared with the DDI frontend. These are not adapter
 * feature answers. Adapter queries use d12_core_caps_init below. */
static const struct d12_core_caps d12_allocation_caps = {
   .max_feature_level = D12_CORE_FEATURE_LEVEL_11_0,
   .gpu_virtual_address_bits = 39,
   .gpu_upload_heap = true,
};

void
d12_core_caps_init(struct d12_core_caps *caps,
                   const struct d12_core_caps_device_info *info)
{
   memset(caps, 0, sizeof(*caps));
   if (!info || !info->active_cores || info->vm_end <= info->vm_start)
      return;

   const bool legacy = info->generation == 13 || info->generation == 14;
   const bool native = info->generation == 17;
   if (!legacy && !native)
      return;

   /* bit length of the highest userspace VA, rather than a fixed 39. */
   for (uint64_t addr = info->vm_end - 1; addr; addr >>= 1)
      caps->gpu_virtual_address_bits++;
   caps->node_count = 1;
   caps->uma = true;
   caps->cache_coherent_uma = info->cache_coherent_uma;
   caps->isolated_mmu = info->isolated_mmu;
   /* D3D12 tile-based rendering is an API efficiency promise, distinct
    * from the physical tile buffer. Match the conservative native reference. */
   caps->tile_based_deferred_renderer = false;
   caps->resource_heap_tier = 2; /* one BO allocation kind, mixed placements */
   caps->gpu_upload_heap = true; /* committed/placed GPU_UPLOAD use that BO */
   caps->max_root_signature_version = 2; /* integer static samplers unproven */
   caps->max_shader_model = D12_CORE_SHADER_MODEL_6_0;
   /* Tier 2 requires 2048 distinct samplers. Deduplication cannot meet
    * that contract with the current 1024-entry hardware table. */
   caps->resource_binding_tier = 1;
   caps->max_sampler_heap_size = 1024; /* distinct native states, no dedup promise */
   caps->max_view_heap_size = 1000000; /* rooted descriptor index range */
   /* 8x MSAA is still unavailable, including at FL11_0. Withholding FL12
    * does not close that inherited baseline-conformance gap. */
   caps->sample_counts = legacy ? (1u | 2u | 4u) : 1u;
   caps->wave_ops = info->compiler_wave_ops;
   caps->int64_shader_ops = info->compiler_int64;
   caps->native_16bit_shader_ops = info->compiler_native_16bit;
   if (caps->wave_ops) {
      caps->wave_lane_count_min = 32;
      caps->wave_lane_count_max = 32;
      /* One executable wave per active core is a conservative lower bound.
       * The UAPI does not expose SIMD occupancy; do not copy Proton's guessed
       * 32*subgroupSize fallback as a hardware measurement. */
      caps->total_lane_count = info->active_cores <= UINT32_MAX / 32
                                 ? info->active_cores * 32 : 32;
   }

   /* G17P's bounded raster path is not the full AGX2 path. Do not derive its
    * sample, sparse, stencil, sampler or pre-raster promises from chip age.
    * A compiler accepting a mesh shader does not implement DispatchMesh. */
   /* Sparse feedback/default-mapping witnesses fail, including a missing
    * SM5 texture-feedback opcode. A BO bind hook alone is not a tiled tier. */
   caps->tiled_resources_tier = 0;
   caps->typed_uav_load_additional_formats = legacy;
   caps->full_texture_sampling = legacy;
   caps->programmable_sample_positions_tier = legacy ? 1 : 0;
   caps->render_passes_tier = legacy ? 1 : 0;
   /* The required modern depth/stencil PSO stream is not accepted yet. */
   caps->independent_front_and_back_stencil = false;
   caps->triangle_fan = legacy;
   caps->dynamic_index_buffer_strip_cut = legacy;
   /* The upstream non-normalized-coordinate test fails its integer static
    * sampler variant on G13. Keep this contract off until both routes pass. */
   caps->non_normalized_coordinate_samplers = false;
   caps->dynamic_depth_bias = false;
   /* ExecuteIndirect + predication fails the upstream state witness. */
   caps->predication = false;
   caps->texture_copy_between_dimensions = legacy;
   caps->alpha_blend_factor = legacy;
   caps->output_merger_logic_op = legacy;
   /* Copy-queue timestamps remain off until GPU timestamp execution is
    * qualified; CPU clock values are not a timestamp-query implementation. */
   caps->write_buffer_immediate_support_flags = legacy
      ? D12_CORE_COMMAND_LIST_SUPPORT_DIRECT |
        D12_CORE_COMMAND_LIST_SUPPORT_COMPUTE |
        D12_CORE_COMMAND_LIST_SUPPORT_COPY : 0;
   caps->max_feature_level = caps->resource_binding_tier >= 2 &&
                             caps->tiled_resources_tier >= 2 &&
                             caps->typed_uav_load_additional_formats
                                ? D12_CORE_FEATURE_LEVEL_12_0
                                : D12_CORE_FEATURE_LEVEL_11_0;
}

const struct d12_core_caps *
d12_core_get_caps(void)
{
   return &d12_allocation_caps;
}

bool
d12_core_caps_supports_feature_level(const struct d12_core_caps *caps,
                                     uint32_t level)
{
   /* Reject enum holes, not just numbers inside the interval. */
   return caps && level <= caps->max_feature_level &&
          (level == D12_CORE_FEATURE_LEVEL_11_0 ||
           level == D12_CORE_FEATURE_LEVEL_11_1 ||
           level == D12_CORE_FEATURE_LEVEL_12_0 ||
           level == D12_CORE_FEATURE_LEVEL_12_1);
}

uint32_t
d12_core_caps_choose_feature_level(const struct d12_core_caps *caps,
                                   const uint32_t *requested, uint32_t count)
{
   uint32_t best = 0;
   if (requested)
      for (uint32_t i = 0; i < count; ++i)
         /* FEATURE_LEVELS negotiation accepts arbitrary requested values below
          * the device maximum, unlike D3D12CreateDevice's enum validation. */
         if (caps && requested[i] <= caps->max_feature_level &&
             requested[i] > best)
            best = requested[i];
   return best;
}

bool
d12_core_supports_feature_level(uint32_t level)
{
   return d12_core_caps_supports_feature_level(d12_core_get_caps(), level);
}

uint32_t
d12_core_choose_feature_level(const uint32_t *requested, uint32_t count)
{
   return d12_core_caps_choose_feature_level(d12_core_get_caps(), requested, count);
}
