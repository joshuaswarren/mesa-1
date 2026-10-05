/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT */
#include "../d12_core_caps.h"
#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

static struct d12_core_caps_device_info
adapter(uint32_t generation, uint32_t cores)
{
   return (struct d12_core_caps_device_info){
      .generation = generation, .active_cores = cores,
      .vm_start = UINT64_C(0x100000000), .vm_end = UINT64_C(1) << 39,
      .isolated_mmu = true, .cache_coherent_uma = true,
      .compiler_wave_ops = generation != 17,
      .compiler_int64 = generation != 17,
      .compiler_native_16bit = generation != 17,
   };
}

int main(void)
{
   struct d12_core_caps a, b, c, unknown;
   struct d12_core_caps_device_info g13 = adapter(13, 16);
   struct d12_core_caps_device_info g14 = adapter(14, 8);
   struct d12_core_caps_device_info g17 = adapter(17, 6);
   d12_core_caps_init(&a, &g13);
   d12_core_caps_init(&b, &g14);
   d12_core_caps_init(&c, &g17);
   assert(a.total_lane_count == 512 && b.total_lane_count == 256);
   assert(a.wave_lane_count_min == 32 && a.wave_lane_count_max == 32);
   assert(c.total_lane_count == 0 && !c.wave_ops && !c.int64_shader_ops);
   assert(a.gpu_virtual_address_bits == 39 && c.gpu_virtual_address_bits == 39);
   assert(a.node_count == 1 && a.uma && a.cache_coherent_uma);
   assert(a.isolated_mmu && a.resource_heap_tier == 2 && a.gpu_upload_heap);
   assert(a.sample_counts == 7 && c.sample_counts == 1);
   assert(a.tiled_resources_tier == 0 && c.tiled_resources_tier == 0);
   assert(!a.predication && !a.independent_front_and_back_stencil && a.dynamic_depth_bias);
   assert(a.full_texture_sampling && !c.full_texture_sampling);
   assert(a.mesh_shader_tier == 0 && b.mesh_shader_tier == 0 && c.mesh_shader_tier == 0);
   assert(a.max_shader_model == D12_CORE_SHADER_MODEL_6_0);
   assert(c.max_shader_model == D12_CORE_SHADER_MODEL_5_1);
   assert(d12_core_get_caps()->node_count == 1);
   assert(d12_core_get_caps()->resource_binding_tier == 1);
   assert(d12_core_get_caps()->resource_heap_tier == 2);
   assert(a.native_16bit_shader_ops && !c.native_16bit_shader_ops);
   assert(!a.copy_queue_timestamp_queries && !c.copy_queue_timestamp_queries);
   assert(a.programmable_sample_positions_tier == 0 && !c.programmable_sample_positions_tier);
   assert(a.render_passes_tier == 1 && !c.render_passes_tier);
   assert(a.dynamic_depth_bias && b.dynamic_depth_bias && !c.dynamic_depth_bias);
   assert(!a.non_normalized_coordinate_samplers && !c.non_normalized_coordinate_samplers);
   assert(!a.tile_based_deferred_renderer);
   assert(a.max_root_signature_version == 2);
   assert(a.max_sampler_heap_size == 1024 && a.max_view_heap_size == 1000000);
   assert(a.max_feature_level == D12_CORE_FEATURE_LEVEL_11_0);
   assert(c.max_feature_level == D12_CORE_FEATURE_LEVEL_11_0);
   assert(a.resource_binding_tier == 1);
   const uint32_t requests[] = {D12_CORE_FEATURE_LEVEL_12_1, 0xb001, 0xb000};
   assert(d12_core_caps_choose_feature_level(&a, requests, 3) == 0xb000);
   assert(!d12_core_caps_supports_feature_level(&a, 0xb001));
   assert(!d12_core_caps_supports_feature_level(&a, 0xa100));
   assert(!d12_core_caps_supports_feature_level(&a, 0xc100));
   const uint32_t lower[] = {0x9100, 0x9300};
   assert(d12_core_caps_choose_feature_level(&a, lower, 2) == 0x9300);
   assert(!d12_core_caps_supports_feature_level(&a, 0x9300));
   assert(d12_core_caps_choose_feature_level(&a, NULL, 3) == 0);
   assert(d12_core_caps_choose_feature_level(&a, requests, 0) == 0);

   /* Missing backend support cannot be replaced by newer silicon. */
   g13.compiler_wave_ops = g13.compiler_int64 = g13.compiler_native_16bit = false;
   g13.isolated_mmu = false;
   g13.vm_end = UINT64_C(1) << 36;
   d12_core_caps_init(&unknown, &g13);
   assert(!unknown.wave_ops && !unknown.wave_lane_count_min && !unknown.total_lane_count);
   assert(!unknown.int64_shader_ops && !unknown.native_16bit_shader_ops);
   assert(unknown.max_shader_model == D12_CORE_SHADER_MODEL_5_1);
   assert(!unknown.tiled_resources_tier && !unknown.isolated_mmu);
   assert(unknown.gpu_virtual_address_bits == 36);
   /* Deriving another device cannot mutate existing caps. */
   assert(a.wave_ops && a.tiled_resources_tier == 0 && a.gpu_virtual_address_bits == 39);
   g13.generation = 15;
   d12_core_caps_init(&unknown, &g13);
   assert(!unknown.max_feature_level && !unknown.node_count);
   g13 = adapter(13, 0);
   d12_core_caps_init(&unknown, &g13);
   assert(!unknown.max_feature_level);
   g13 = adapter(13, 16);
   g13.vm_end = g13.vm_start;
   d12_core_caps_init(&unknown, &g13);
   assert(!unknown.max_feature_level);
   memset(&unknown, 0xff, sizeof(unknown));
   d12_core_caps_init(&unknown, NULL);
   assert(!unknown.max_feature_level);
   puts("d12_core_caps_tests: PASS (G13/G14/G17P, backend gates, VA, enum holes, isolation)");
   return 0;
}
