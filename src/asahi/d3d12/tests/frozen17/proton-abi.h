/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 * Compile-only binding for vkd3d-proton's WIDL aggregate COM return ABI.
 * DirectX-Headers uses native struct returns on Linux; WIDL uses an explicit
 * output pointer. Only these calls differ. No fixture/driver data is changed.
 * Preinclude this when linking the unmodified native Proton libraries.
 */
#define COBJMACROS
#include <wsl/winadapter.h>
#include <directx/d3d12.h>
#define D12_TEST_ABI_H

#define AGGREGATE_HELPER(result, helper, interface, method)                    \
   static inline result helper(interface *object)                            \
   {                                                                         \
      result value;                                                          \
      typedef result *(STDMETHODCALLTYPE *call_t)(interface *, result *);     \
      ((call_t)object->lpVtbl->method)(object, &value);                        \
      return value;                                                          \
   }
AGGREGATE_HELPER(D3D12_COMMAND_QUEUE_DESC, d12_test_command_queue_get_desc,
                 ID3D12CommandQueue, GetDesc)
AGGREGATE_HELPER(D3D12_RESOURCE_DESC, d12_test_resource_get_desc,
                 ID3D12Resource, GetDesc)
AGGREGATE_HELPER(D3D12_HEAP_DESC, d12_test_heap_get_desc, ID3D12Heap, GetDesc)
AGGREGATE_HELPER(D3D12_CPU_DESCRIPTOR_HANDLE,
                 d12_test_descriptor_heap_get_cpu_start,
                 ID3D12DescriptorHeap, GetCPUDescriptorHandleForHeapStart)
AGGREGATE_HELPER(D3D12_GPU_DESCRIPTOR_HANDLE,
                 d12_test_descriptor_heap_get_gpu_start,
                 ID3D12DescriptorHeap, GetGPUDescriptorHandleForHeapStart)
AGGREGATE_HELPER(LUID, d12_test_device_get_adapter_luid,
                 ID3D12Device, GetAdapterLuid)
#undef AGGREGATE_HELPER

static inline D3D12_RESOURCE_ALLOCATION_INFO
d12_test_device_get_resource_allocation_info(ID3D12Device *device,
   UINT mask, UINT count, const D3D12_RESOURCE_DESC *descs)
{
   D3D12_RESOURCE_ALLOCATION_INFO value;
   typedef D3D12_RESOURCE_ALLOCATION_INFO *(STDMETHODCALLTYPE *call_t)(
      ID3D12Device *, D3D12_RESOURCE_ALLOCATION_INFO *, UINT, UINT,
      const D3D12_RESOURCE_DESC *);
   ((call_t)device->lpVtbl->GetResourceAllocationInfo)(
      device, &value, mask, count, descs);
   return value;
}
