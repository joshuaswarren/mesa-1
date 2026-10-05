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
   puts("PASS D3D12 heap descriptor validation (UMA)");
   return 0;
}
