/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT */

#define COBJMACROS
#include <wsl/winadapter.h>
#include <directx/d3d12.h>
#include "d12_test_abi.h"
#include <stdint.h>
#include <stdio.h>

static unsigned failures;
#define CHECK(cond, what) do { \
   if (!(cond)) { printf("FAIL - %s\n", what); failures++; } \
} while (0)

/* DirectX-Headers uses an explicit aggregate output on Windows. */
#ifdef _WIN32
#define QUERY(method, ...) ID3D12Device15_##method(device, &info, __VA_ARGS__)
#else
#define QUERY(method, ...) info = ID3D12Device15_##method(device, __VA_ARGS__)
#endif

int
main(void)
{
   ID3D12Device15 *device = NULL;
   HRESULT hr = D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0,
                                  &IID_ID3D12Device15, (void **)&device);
   if (FAILED(hr) || !device) {
      printf("FAIL - device creation %#x\n", (unsigned)hr);
      return 1;
   }
   D3D12_RESOURCE_DESC buffer = {
      .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER, .Width = 32,
      .Height = 1, .DepthOrArraySize = 1, .MipLevels = 1,
      .SampleDesc = {.Count = 1}, .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
   };
   D3D12_RESOURCE_DESC descs[2] = {buffer, {
      .Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
      .Alignment = D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT,
      .Width = 4, .Height = 4, .DepthOrArraySize = 1, .MipLevels = 1,
      .Format = DXGI_FORMAT_R8_UINT, .SampleDesc = {.Count = 1},
   }};
   D3D12_RESOURCE_DESC1 extended[2] = {0};
   for (unsigned i = 0; i < 2; i++) {
      extended[i] = (D3D12_RESOURCE_DESC1){
         .Dimension = descs[i].Dimension, .Alignment = descs[i].Alignment,
         .Width = descs[i].Width, .Height = descs[i].Height,
         .DepthOrArraySize = descs[i].DepthOrArraySize,
         .MipLevels = descs[i].MipLevels, .Format = descs[i].Format,
         .SampleDesc = descs[i].SampleDesc, .Layout = descs[i].Layout,
      };
   }
   D3D12_RESOURCE_ALLOCATION_INFO info, single[2];
   QUERY(GetResourceAllocationInfo, 0, 1, &descs[0]);
   single[0] = info;
   QUERY(GetResourceAllocationInfo, 0, 1, &descs[1]);
   single[1] = info;
   CHECK(single[0].Alignment == 65536 && single[1].Alignment == 4096 &&
            single[0].SizeInBytes == 65536 && single[1].SizeInBytes < 65536,
         "small texture and buffer have different placement requirements");
   if (single[0].SizeInBytes == UINT64_MAX ||
       single[1].SizeInBytes == UINT64_MAX) {
      ID3D12Device15_Release(device);
      return 1;
   }
   for (unsigned version = 0; version < 4; version++) {
      switch (version) {
      case 0: QUERY(GetResourceAllocationInfo, 0, 0, NULL); break;
      case 1: QUERY(GetResourceAllocationInfo1, 0, 0, NULL, NULL); break;
      case 2: QUERY(GetResourceAllocationInfo2, 0, 0, NULL, NULL); break;
      default: QUERY(GetResourceAllocationInfo3, 0, 0, NULL, NULL, NULL, NULL); break;
      }
      CHECK(info.SizeInBytes == 0 && info.Alignment == 1, "empty allocation query");
      D3D12_RESOURCE_ALLOCATION_INFO1 per[2] = {0};
      switch (version) {
      case 0: QUERY(GetResourceAllocationInfo, 0, 2, descs); break;
      case 1: QUERY(GetResourceAllocationInfo1, 0, 2, descs, per); break;
      case 2: QUERY(GetResourceAllocationInfo2, 0, 2, extended, per); break;
      default: QUERY(GetResourceAllocationInfo3, 0, 2, extended, NULL, NULL, per); break;
      }
      CHECK(info.SizeInBytes == 131072 && info.Alignment == 65536,
            "mixed allocation query pads the aggregate to 64 KiB");
      if (version) {
         CHECK(per[0].Offset == 0 &&
                  per[0].SizeInBytes == single[0].SizeInBytes &&
                  per[0].Alignment == single[0].Alignment &&
                  per[1].Offset == single[0].SizeInBytes &&
                  per[1].SizeInBytes == single[1].SizeInBytes &&
                  per[1].Alignment == single[1].Alignment,
               "individual allocation requirements and offsets");
      }
      D3D12_RESOURCE_DESC huge = buffer;
      D3D12_RESOURCE_DESC1 huge1 = extended[0];
      huge.Width = huge1.Width = UINT64_MAX;
      switch (version) {
      case 0: QUERY(GetResourceAllocationInfo, 0, 1, &huge); break;
      case 1: QUERY(GetResourceAllocationInfo1, 0, 1, &huge, NULL); break;
      case 2: QUERY(GetResourceAllocationInfo2, 0, 1, &huge1, NULL); break;
      default: QUERY(GetResourceAllocationInfo3, 0, 1, &huge1, NULL, NULL, NULL); break;
      }
      CHECK(info.SizeInBytes == UINT64_MAX && info.Alignment == 65536,
            "allocation-size overflow returns failure with a legal alignment");
   }
   D3D12_DESCRIPTOR_HEAP_DESC hd = {
      .Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, .NumDescriptors = 16,
      .Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
   };
   CHECK(ID3D12Device15_CreateDescriptorHeap(device, &hd,
            &IID_ID3D12DescriptorHeap, NULL) == S_FALSE, "validation-only sampler heap");
   ID3D12DescriptorHeap *samplers = NULL;
   hr = ID3D12Device15_CreateDescriptorHeap(device, &hd,
            &IID_ID3D12DescriptorHeap, (void **)&samplers);
   CHECK(hr == S_OK && samplers, "shader-visible sampler heap creation");
   for (unsigned invalid = 0; invalid < 5; invalid++) {
      D3D12_DESCRIPTOR_HEAP_DESC bad = hd;
      switch (invalid) {
      case 0: bad.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV; break;
      case 1: bad.NodeMask = 2; break;
      case 2: bad.Flags = (D3D12_DESCRIPTOR_HEAP_FLAGS)2; break;
      case 3:
         bad.NumDescriptors = D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE + 1;
         break;
      default:
         bad.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
         bad.NumDescriptors = D3D12_MAX_SHADER_VISIBLE_DESCRIPTOR_HEAP_SIZE_TIER_2 + 1;
         break;
      }
      void *out = (void *)(uintptr_t)1;
      hr = ID3D12Device15_CreateDescriptorHeap(device, &bad,
               &IID_ID3D12DescriptorHeap, &out);
      CHECK(hr == E_INVALIDARG && !out, "invalid descriptor heap nulls output");
      CHECK(ID3D12Device15_CreateDescriptorHeap(device, &bad,
               &IID_ID3D12DescriptorHeap, NULL) == E_INVALIDARG, "validation-only rejection");
   }
   if (samplers) {
      CHECK(d12_test_descriptor_heap_get_cpu_start(samplers).ptr != 0,
            "valid sampler heap survives later creation failures");
      ID3D12DescriptorHeap_Release(samplers);
   }
   D3D12_HEAP_DESC heap_desc = {
      .SizeInBytes = 1024, .Properties = {.Type = D3D12_HEAP_TYPE_DEFAULT},
   };
   ID3D12Heap *heap = NULL;
   hr = ID3D12Device15_CreateHeap1(device, &heap_desc, NULL,
                                  &IID_ID3D12Heap, (void **)&heap);
   CHECK(hr == S_OK && heap, "CreateHeap1");
   if (heap) {
      /* Rounded BO storage is not additional application heap space. */
      buffer.Width = 1025;
      void *out = (void *)(uintptr_t)1;
      hr = ID3D12Device15_CreatePlacedResource(device, heap, 0, &buffer,
               D3D12_RESOURCE_STATE_COMMON, NULL, &IID_ID3D12Resource, &out);
      CHECK(hr == E_INVALIDARG && !out, "placement past declared heap size fails");
      buffer.Width = 65536;
      out = (void *)(uintptr_t)1;
      hr = ID3D12Device15_CreatePlacedResource(device, heap, UINT64_MAX - 65535,
               &buffer, D3D12_RESOURCE_STATE_COMMON, NULL, &IID_ID3D12Resource, &out);
      CHECK(hr == E_INVALIDARG && !out, "placement offset overflow fails");
      buffer.Width = 1024;
      ID3D12Resource *resource = NULL;
      hr = ID3D12Device15_CreatePlacedResource(device, heap, 0, &buffer,
               D3D12_RESOURCE_STATE_COMMON, NULL, &IID_ID3D12Resource, (void **)&resource);
      CHECK(hr == S_OK && resource, "heap usable after failed placements");
      if (resource)
         ID3D12Resource_Release(resource);
      ID3D12Heap_Release(heap);
   }
   CHECK(ID3D12Device15_Release(device) == 0, "device references released");
   printf("RESOURCE VALIDATION: %s\n", failures ? "FAILURES" : "ALL PASS");
   return failures ? 1 : 0;
}
