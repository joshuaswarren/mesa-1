/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT */
#include "../d12_core_caps_query.h"
#include <assert.h>
#include <stdio.h>

int
main(void)
{
   for (unsigned generation = 13; generation <= 17; ++generation) {
      if (generation != 13 && generation != 14 && generation != 17)
         continue;
      struct d12_core_caps_device_info info = {
         .generation = generation, .active_cores = 6,
         .vm_start = 0x100000000ull, .vm_end = 1ull << 39,
         .isolated_mmu = true, .cache_coherent_uma = true,
         .compiler_wave_ops = generation != 17,
         .compiler_int64 = generation != 17,
         .compiler_native_16bit = generation != 17,
      };
      struct d12_core_caps caps;
      d12_core_caps_init(&caps, &info);
      D3D12_FEATURE_DATA_D3D12_OPTIONS opts;
      D3D12_FEATURE_DATA_D3D12_OPTIONS1 opts1;
      assert(d12_core_caps_query(&caps, D3D12_FEATURE_D3D12_OPTIONS,
                                 &opts, sizeof(opts)) == S_OK);
      assert(opts.ResourceBindingTier == D3D12_RESOURCE_BINDING_TIER_1);
      assert(opts.ResourceHeapTier == D3D12_RESOURCE_HEAP_TIER_2);
      assert(d12_core_caps_query(&caps, D3D12_FEATURE_D3D12_OPTIONS1,
                                 &opts1, sizeof(opts1)) == S_OK);
      assert(opts1.WaveOps == (generation != 17));
      assert(opts1.WaveLaneCountMin == (generation == 17 ? 0 : 32));
      assert(opts1.WaveLaneCountMax == opts1.WaveLaneCountMin);
      /* Valid versions clamp to the implemented compiler contract. Every
       * other enum spelling must fail, including holes between 5.1/6.0. */
      for (unsigned version = 0; version < 0x100; ++version) {
         D3D12_FEATURE_DATA_SHADER_MODEL sm = {version};
         HRESULT hr = d12_core_caps_query(&caps, D3D12_FEATURE_SHADER_MODEL,
                                          &sm, sizeof(sm));
         bool valid = version == 0x51 || (version >= 0x60 && version <= 0x68);
         assert(hr == (valid ? S_OK : E_INVALIDARG));
         unsigned maximum = generation == 17 ? 0x51 : 0x60;
         assert(sm.HighestShaderModel ==
                (valid && version > maximum ? maximum : version));
      }
      D3D12_FEATURE_DATA_SHADER_MODEL sm = {D3D_SHADER_MODEL_6_6};
      assert(d12_core_caps_query(&caps, D3D12_FEATURE_SHADER_MODEL,
                                 &sm, sizeof(sm) - 1) == E_INVALIDARG);
      assert(sm.HighestShaderModel == D3D_SHADER_MODEL_6_6);
      assert(d12_core_caps_query(&caps, D3D12_FEATURE_SHADER_MODEL,
                                 NULL, sizeof(sm)) == E_INVALIDARG);
   }
   puts("PASS G13/G14/G17P feature queries, compiler contracts and SM enum holes");
   return 0;
}
