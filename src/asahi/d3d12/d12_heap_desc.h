/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include <wsl/winadapter.h>
#include <directx/d3d12.h>

/* Descriptor validation precedes allocation, including validation-only calls.
 * Zero node masks are the D3D12 shorthand for node 0; this device has one node.
 * Heap size need not be a multiple of Alignment (the allocation rounds up).
 */
static inline bool
d12_heap_desc_valid(const D3D12_HEAP_DESC *desc, bool gpu_upload_heap)
{
   if (!desc || !desc->SizeInBytes ||
       (desc->Alignment &&
        desc->Alignment != D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT &&
        desc->Alignment != D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT) ||
       desc->Properties.CreationNodeMask > 1 ||
       desc->Properties.VisibleNodeMask > 1)
      return false;

   const D3D12_HEAP_PROPERTIES *props = &desc->Properties;
   switch (props->Type) {
   case D3D12_HEAP_TYPE_GPU_UPLOAD:
      if (!gpu_upload_heap)
         return false;
      /* fall through */
   case D3D12_HEAP_TYPE_DEFAULT:
   case D3D12_HEAP_TYPE_UPLOAD:
   case D3D12_HEAP_TYPE_READBACK:
      if (props->CPUPageProperty != D3D12_CPU_PAGE_PROPERTY_UNKNOWN ||
          props->MemoryPoolPreference != D3D12_MEMORY_POOL_UNKNOWN)
         return false;
      break;
   case D3D12_HEAP_TYPE_CUSTOM:
      /* UMA has only L0. CUSTOM must explicitly name its CPU access policy. */
      if (props->CPUPageProperty < D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE ||
          props->CPUPageProperty > D3D12_CPU_PAGE_PROPERTY_WRITE_BACK ||
          props->MemoryPoolPreference != D3D12_MEMORY_POOL_L0)
         return false;
      break;
   default:
      return false;
   }

   const unsigned valid_flags = D3D12_HEAP_FLAG_SHARED |
      D3D12_HEAP_FLAG_DENY_BUFFERS | D3D12_HEAP_FLAG_ALLOW_DISPLAY |
      D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER |
      D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES |
      D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES |
      D3D12_HEAP_FLAG_HARDWARE_PROTECTED | D3D12_HEAP_FLAG_ALLOW_WRITE_WATCH |
      D3D12_HEAP_FLAG_ALLOW_SHADER_ATOMICS |
      D3D12_HEAP_FLAG_CREATE_NOT_RESIDENT | D3D12_HEAP_FLAG_CREATE_NOT_ZEROED |
      D3D12_HEAP_FLAG_TOOLS_USE_MANUAL_WRITE_TRACKING;
   if ((unsigned)desc->Flags & ~valid_flags)
      return false;

   /* Display heaps are committed-resource-only. Protection and write-watch
    * require platform allocation support which this driver does not provide. */
   if (desc->Flags & (D3D12_HEAP_FLAG_ALLOW_DISPLAY |
                      D3D12_HEAP_FLAG_HARDWARE_PROTECTED |
                      D3D12_HEAP_FLAG_ALLOW_WRITE_WATCH))
      return false;

   if (desc->Flags & D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER) {
      if (props->Type != D3D12_HEAP_TYPE_DEFAULT &&
          (props->Type != D3D12_HEAP_TYPE_CUSTOM ||
           props->CPUPageProperty != D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE))
         return false;
   }

   return true;
}
