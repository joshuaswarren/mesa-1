/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 * G13/G14 CPU volume-copy qualification, including padded rows/slices.
 */
#define COBJMACROS
#include <wsl/winadapter.h>
#include <directx/d3d12.h>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int
main(void)
{
   ID3D12Device *device = NULL;
   ID3D12Resource *resource = NULL;
   assert(D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0,
                           &IID_ID3D12Device, (void **)&device) == S_OK);
   D3D12_HEAP_PROPERTIES heap = {
      .Type = D3D12_HEAP_TYPE_CUSTOM,
      .CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
      .MemoryPoolPreference = D3D12_MEMORY_POOL_L0,
   };
   D3D12_RESOURCE_DESC desc = {
      .Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D,
      .Width = 7, .Height = 3, .DepthOrArraySize = 4, .MipLevels = 1,
      .Format = DXGI_FORMAT_R8G8B8A8_UNORM, .SampleDesc = {.Count = 1},
   };
   assert(ID3D12Device_CreateCommittedResource(device, &heap, 0, &desc,
          D3D12_RESOURCE_STATE_COMMON, NULL, &IID_ID3D12Resource,
          (void **)&resource) == S_OK);
   uint8_t input[4 * 160], output[4 * 160];
   memset(input, 0xa5, sizeof(input));
   for (unsigned z = 0; z < 4; z++)
      for (unsigned y = 0; y < 3; y++)
         for (unsigned x = 0; x < 28; x++)
            input[z * 160 + y * 40 + x] = z * 53 + y * 17 + x;
   assert(ID3D12Resource_WriteToSubresource(resource, 0, NULL, input,
                                           40, 160) == S_OK);
   memset(output, 0xcd, sizeof(output));
   assert(ID3D12Resource_ReadFromSubresource(resource, output, 40, 160,
                                            0, NULL) == S_OK);
   for (unsigned z = 0; z < 4; z++)
      for (unsigned y = 0; y < 3; y++) {
         assert(!memcmp(input + z * 160 + y * 40,
                        output + z * 160 + y * 40, 28));
         for (unsigned x = 28; x < 40; x++)
            assert(output[z * 160 + y * 40 + x] == 0xcd);
      }
   D3D12_BOX partial = {1, 1, 2, 6, 3, 4};
   memset(input, 0x39, sizeof(input));
   assert(ID3D12Resource_WriteToSubresource(resource, 0, &partial, input,
                                           40, 160) == S_OK);
   memset(output, 0xcd, sizeof(output));
   assert(ID3D12Resource_ReadFromSubresource(resource, output, 40, 160,
                                            0, &partial) == S_OK);
   for (unsigned z = 0; z < 2; z++)
      for (unsigned y = 0; y < 2; y++)
         for (unsigned x = 0; x < 20; x++)
            assert(output[z * 160 + y * 40 + x] == 0x39);
   D3D12_BOX empty = {7, 3, 4, 7, 3, 4};
   assert(ID3D12Resource_WriteToSubresource(resource, 0, &empty, NULL,
                                           0, 0) == S_OK);
   D3D12_BOX outside = {7, 3, 5, 7, 3, 5};
   assert(ID3D12Resource_ReadFromSubresource(resource, output, 40, 160,
                                            0, &outside) == E_INVALIDARG);
   assert(ID3D12Resource_WriteToSubresource(resource, 0, NULL, input,
                                           40, 80) == E_INVALIDARG);
   ID3D12Resource_Release(resource);
   assert(ID3D12Device_Release(device) == 0);
   puts("CPU VOLUME LIVE: ALL PASS");
   return 0;
}
