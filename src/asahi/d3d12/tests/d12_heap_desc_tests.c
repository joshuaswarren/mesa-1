/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT */

#include "../d12_heap_desc.h"
#include <assert.h>
#include <stdio.h>

int
main(void)
{
   D3D12_HEAP_DESC desc = {.SizeInBytes = 4096};
   assert(!d12_heap_desc_valid(NULL, true));
   assert(!d12_heap_desc_valid(&desc, true)); /* Frozen G1. */
   for (unsigned type = D3D12_HEAP_TYPE_DEFAULT;
        type <= D3D12_HEAP_TYPE_GPU_UPLOAD; ++type) {
      desc = (D3D12_HEAP_DESC){.SizeInBytes = 4096};
      desc.Properties.Type = type;
      if (type == D3D12_HEAP_TYPE_CUSTOM) {
         assert(!d12_heap_desc_valid(&desc, true));
         desc.Properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
         desc.Properties.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
      }
      assert(d12_heap_desc_valid(&desc, true));
      desc.SizeInBytes = 0;
      assert(!d12_heap_desc_valid(&desc, true));
      desc.SizeInBytes = 4096;
      desc.Properties.CreationNodeMask = 2;
      assert(!d12_heap_desc_valid(&desc, true));
      desc.Properties.CreationNodeMask = 1;
      desc.Properties.VisibleNodeMask = 3;
      assert(!d12_heap_desc_valid(&desc, true));
      desc.Properties.VisibleNodeMask = 1;
      assert(d12_heap_desc_valid(&desc, true));
      desc.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
      assert(d12_heap_desc_valid(&desc, true));
      desc.Alignment = D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT;
      assert(d12_heap_desc_valid(&desc, true));
      desc.Alignment = D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT;
      assert(!d12_heap_desc_valid(&desc, true));
      desc.Alignment = 0;
      if (type == D3D12_HEAP_TYPE_CUSTOM) {
         desc.Properties.MemoryPoolPreference = D3D12_MEMORY_POOL_L1;
         assert(!d12_heap_desc_valid(&desc, true));
         desc.Properties.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
         desc.Properties.CPUPageProperty = 4;
         assert(!d12_heap_desc_valid(&desc, true));
      } else {
         desc.Properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
         assert(!d12_heap_desc_valid(&desc, true));
         desc.Properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
         desc.Properties.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
         assert(!d12_heap_desc_valid(&desc, true));
      }
   }
   desc = (D3D12_HEAP_DESC){.SizeInBytes = 4096};
   desc.Properties.Type = D3D12_HEAP_TYPE_GPU_UPLOAD;
   assert(!d12_heap_desc_valid(&desc, false));
   desc.Properties.Type = 6;
   assert(!d12_heap_desc_valid(&desc, true));
   desc.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
   const unsigned invalid_flags[] = {
      2, D3D12_HEAP_FLAG_ALLOW_DISPLAY, D3D12_HEAP_FLAG_HARDWARE_PROTECTED,
      D3D12_HEAP_FLAG_ALLOW_WRITE_WATCH, 0x80000000,
   };
   for (unsigned i = 0; i < sizeof(invalid_flags) / sizeof(invalid_flags[0]); i++) {
      desc.Flags = invalid_flags[i];
      assert(!d12_heap_desc_valid(&desc, true));
   }
   desc.Flags = D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER;
   assert(d12_heap_desc_valid(&desc, true));
   desc.Properties.Type = D3D12_HEAP_TYPE_UPLOAD;
   assert(!d12_heap_desc_valid(&desc, true));
   desc.Properties.Type = D3D12_HEAP_TYPE_CUSTOM;
   desc.Properties.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
   desc.Properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
   assert(!d12_heap_desc_valid(&desc, true));
   desc.Properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE;
   assert(d12_heap_desc_valid(&desc, true));

   /* Exercise all resource-class restrictions and all subsets of the four
    * allocation/atomic/tool hints, with and without process sharing. The
    * documented cross-adapter form has SHARED and no DENY flags. None of
    * these valid combinations may turn CREATE_NOT_ZEROED into a refusal.
    * Protection, display and write-watch remain explicit unsupported cases
    * above, rather than advertised allocation capabilities. */
   const unsigned deny[] = {
      0, D3D12_HEAP_FLAG_DENY_BUFFERS,
      D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES,
      D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES,
      D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS,
      D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES,
      D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES,
   };
   const unsigned hints[] = {
      D3D12_HEAP_FLAG_CREATE_NOT_ZEROED,
      D3D12_HEAP_FLAG_CREATE_NOT_RESIDENT,
      D3D12_HEAP_FLAG_ALLOW_SHADER_ATOMICS,
      D3D12_HEAP_FLAG_TOOLS_USE_MANUAL_WRITE_TRACKING,
   };
   unsigned combinations = 0;
   for (unsigned custom = 0; custom < 2; ++custom) {
      desc.Properties.Type = custom ? D3D12_HEAP_TYPE_CUSTOM
                                    : D3D12_HEAP_TYPE_DEFAULT;
      desc.Properties.CPUPageProperty = custom
         ? D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE
         : D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
      desc.Properties.MemoryPoolPreference = custom ? D3D12_MEMORY_POOL_L0
                                                   : D3D12_MEMORY_POOL_UNKNOWN;
      for (unsigned mask = 0; mask < 16; ++mask) {
         unsigned flags = 0;
         for (unsigned bit = 0; bit < 4; ++bit)
            if (mask & (1u << bit))
               flags |= hints[bit];
         for (unsigned shared = 0; shared < 2; ++shared) {
            for (unsigned i = 0; i < sizeof(deny) / sizeof(deny[0]); ++i) {
               desc.Flags = flags | deny[i] |
                  (shared ? D3D12_HEAP_FLAG_SHARED : 0);
               assert(d12_heap_desc_valid(&desc, true));
               combinations++;
            }
         }
         desc.Flags = flags | D3D12_HEAP_FLAG_SHARED |
                      D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER;
         assert(d12_heap_desc_valid(&desc, true));
         combinations++;
      }
   }
   printf("PASS %u supported heap flag combinations (DEFAULT/CUSTOM)\n",
          combinations);
   puts("PASS D3D12 heap descriptor validation (UMA)");
   return 0;
}
