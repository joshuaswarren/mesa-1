/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * DirectX-Headers models aggregate COM returns with the native platform ABI:
 * Windows passes an explicit return pointer while other platforms return the
 * value directly. Keep the tests source-compatible with both forms.
 */

#ifndef D12_TEST_ABI_H
#define D12_TEST_ABI_H

static inline D3D12_COMMAND_QUEUE_DESC
d12_test_command_queue_get_desc(ID3D12CommandQueue *queue)
{
#ifdef _WIN32
   D3D12_COMMAND_QUEUE_DESC desc;
   ID3D12CommandQueue_GetDesc(queue, &desc);
   return desc;
#else
   return ID3D12CommandQueue_GetDesc(queue);
#endif
}

static inline D3D12_RESOURCE_DESC
d12_test_resource_get_desc(ID3D12Resource *resource)
{
#ifdef _WIN32
   D3D12_RESOURCE_DESC desc;
   ID3D12Resource_GetDesc(resource, &desc);
   return desc;
#else
   return ID3D12Resource_GetDesc(resource);
#endif
}

static inline D3D12_HEAP_DESC
d12_test_heap_get_desc(ID3D12Heap *heap)
{
#ifdef _WIN32
   D3D12_HEAP_DESC desc;
   ID3D12Heap_GetDesc(heap, &desc);
   return desc;
#else
   return ID3D12Heap_GetDesc(heap);
#endif
}

static inline D3D12_CPU_DESCRIPTOR_HANDLE
d12_test_descriptor_heap_get_cpu_start(ID3D12DescriptorHeap *heap)
{
#ifdef _WIN32
   D3D12_CPU_DESCRIPTOR_HANDLE handle;
   ID3D12DescriptorHeap_GetCPUDescriptorHandleForHeapStart(heap, &handle);
   return handle;
#else
   return ID3D12DescriptorHeap_GetCPUDescriptorHandleForHeapStart(heap);
#endif
}

static inline D3D12_GPU_DESCRIPTOR_HANDLE
d12_test_descriptor_heap_get_gpu_start(ID3D12DescriptorHeap *heap)
{
#ifdef _WIN32
   D3D12_GPU_DESCRIPTOR_HANDLE handle;
   ID3D12DescriptorHeap_GetGPUDescriptorHandleForHeapStart(heap, &handle);
   return handle;
#else
   return ID3D12DescriptorHeap_GetGPUDescriptorHandleForHeapStart(heap);
#endif
}

static inline LUID
d12_test_device_get_adapter_luid(ID3D12Device *device)
{
#ifdef _WIN32
   LUID luid;
   ID3D12Device_GetAdapterLuid(device, &luid);
   return luid;
#else
   return ID3D12Device_GetAdapterLuid(device);
#endif
}

static inline D3D12_RESOURCE_ALLOCATION_INFO
d12_test_device_get_resource_allocation_info(
   ID3D12Device *device, UINT visible_mask, UINT resource_desc_count,
   const D3D12_RESOURCE_DESC *resource_descs)
{
#ifdef _WIN32
   D3D12_RESOURCE_ALLOCATION_INFO info;
   ID3D12Device_GetResourceAllocationInfo(
      device, &info, visible_mask, resource_desc_count, resource_descs);
   return info;
#else
   return ID3D12Device_GetResourceAllocationInfo(
      device, visible_mask, resource_desc_count, resource_descs);
#endif
}

#endif /* D12_TEST_ABI_H */
