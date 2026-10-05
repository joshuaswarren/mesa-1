/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT */

#define COBJMACROS
#include <wsl/winadapter.h>
#include <directx/d3d12.h>
#include "d12_test_abi.h"
#include <stdio.h>

#define CHECK(c) do { if (!(c)) { \
   fprintf(stderr, "FAIL CreateHeap validation at line %d\n", __LINE__); \
   failures++; \
} } while (0)

int
main(void)
{
   unsigned failures = 0;
   ID3D12Device *device = NULL;
   HRESULT hr = D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0,
                                   &IID_ID3D12Device, (void **)&device);
   if (FAILED(hr)) {
      fprintf(stderr, "CreateDevice failed: 0x%08x\n", (unsigned)hr);
      return 1;
   }
   D3D12_HEAP_DESC desc = {.SizeInBytes = 4096};
   ID3D12Heap *heap = (void *)(uintptr_t)1;
   hr = ID3D12Device_CreateHeap(device, &desc, &IID_ID3D12Heap, (void **)&heap);
   CHECK(hr == E_INVALIDARG && heap == NULL);
   printf("zeroed G1 descriptor: HRESULT=0x%08x object=%s\n",
          (unsigned)hr, heap ? "yes" : "no");
   CHECK(ID3D12Device_CreateHeap(device, &desc, &IID_ID3D12Heap, NULL) == E_INVALIDARG);
   desc.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
   CHECK(ID3D12Device_CreateHeap(device, &desc, &IID_ID3D12Heap, NULL) == S_FALSE);
   hr = ID3D12Device_CreateHeap(device, &desc, &IID_ID3D12Heap, (void **)&heap);
   CHECK(hr == S_OK && heap != NULL);
   if (hr == S_OK && heap) {
      const D3D12_HEAP_DESC actual = d12_test_heap_get_desc(heap);
      CHECK(actual.SizeInBytes == 4096);
      CHECK(actual.Alignment == D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT);
      CHECK(actual.Properties.CreationNodeMask == 1 &&
            actual.Properties.VisibleNodeMask == 1);
      ID3D12Heap_Release(heap);
   }
   desc.Properties.CreationNodeMask = 2;
   heap = (void *)(uintptr_t)1;
   hr = ID3D12Device_CreateHeap(device, &desc, &IID_ID3D12Heap, (void **)&heap);
   CHECK(hr == E_INVALIDARG && heap == NULL);
   desc.Properties.CreationNodeMask = 0;
   desc.Properties.Type = D3D12_HEAP_TYPE_CUSTOM;
   CHECK(ID3D12Device_CreateHeap(device, &desc, &IID_ID3D12Heap, NULL) == E_INVALIDARG);
   desc.Properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
   desc.Properties.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
   CHECK(ID3D12Device_CreateHeap(device, &desc, &IID_ID3D12Heap, NULL) == S_FALSE);
   desc.Properties.MemoryPoolPreference = D3D12_MEMORY_POOL_L1;
   CHECK(ID3D12Device_CreateHeap(device, &desc, &IID_ID3D12Heap, NULL) == E_INVALIDARG);
   ID3D12Device_Release(device);
   printf("CreateHeap validation: %s\n", failures ? "FAIL" : "PASS");
   return failures ? 1 : 0;
}
