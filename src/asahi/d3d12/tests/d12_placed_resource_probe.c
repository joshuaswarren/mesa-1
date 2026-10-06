/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Allocation-only integration check. Creates a device, heaps and resources,
 * but no queue or GPU submission. Run separately from the CPU unit suite.
 * The same source builds as native Linux or an x64 Wine consumer.
 */
#define COBJMACROS
#ifdef _WIN32
#include <windows.h>
#else
#include <wsl/winadapter.h>
#endif
#include <directx/d3d12.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static unsigned failures, checks;
#define CHECK(test) do { \
   checks++; \
   if (!(test)) { printf("FAIL line=%d: %s\n", __LINE__, #test); failures++; } \
} while (0)

static void
placed(ID3D12Device *device, ID3D12Heap *heap, uint64_t offset,
       const D3D12_RESOURCE_DESC *desc, HRESULT expected)
{
   ID3D12Resource *resource = (void *)(uintptr_t)1;
   HRESULT hr = ID3D12Device_CreatePlacedResource(
      device, heap, offset, desc, D3D12_RESOURCE_STATE_COMMON, NULL,
      &IID_ID3D12Resource, (void **)&resource);
   CHECK(hr == expected);
   if (hr != expected)
      printf("PLACED flags=%x dimension=%u offset=%llu hr=%08x expected=%08x\n",
             desc->Flags, desc->Dimension, (unsigned long long)offset,
             (unsigned)hr, (unsigned)expected);
   CHECK(SUCCEEDED(hr) ? resource != NULL : resource == NULL);
   if (SUCCEEDED(hr) && resource) {
      D3D12_HEAP_PROPERTIES props;
      D3D12_HEAP_FLAGS flags;
      CHECK(ID3D12Resource_GetHeapProperties(resource, &props, &flags) == S_OK);
      CHECK(props.Type == D3D12_HEAP_TYPE_DEFAULT);
      CHECK(ID3D12Resource_Release(resource) == 0);
   }
}

int
main(int argc, char **argv)
{
   setvbuf(stdout, NULL, _IONBF, 0);
   ID3D12Device *device = NULL;
   CHECK(D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0,
                          &IID_ID3D12Device, (void **)&device) == S_OK);
   if (!device)
      return 1;
   D3D12_FEATURE_DATA_D3D12_OPTIONS opts = {0};
   D3D12_FEATURE_DATA_D3D12_OPTIONS1 opts1 = {0};
   D3D12_FEATURE_DATA_SHADER_MODEL sm = {D3D_SHADER_MODEL_6_6};
   CHECK(ID3D12Device_CheckFeatureSupport(device, D3D12_FEATURE_D3D12_OPTIONS,
                                          &opts, sizeof(opts)) == S_OK);
   CHECK(ID3D12Device_CheckFeatureSupport(device, D3D12_FEATURE_D3D12_OPTIONS1,
                                          &opts1, sizeof(opts1)) == S_OK);
   CHECK(ID3D12Device_CheckFeatureSupport(device, D3D12_FEATURE_SHADER_MODEL,
                                          &sm, sizeof(sm)) == S_OK);
   printf("CAPS heap-tier=%u binding-tier=%u shader-model=%x wave-ops=%u\n",
          opts.ResourceHeapTier, opts.ResourceBindingTier,
          sm.HighestShaderModel, opts1.WaveOps);
   CHECK(opts.ResourceHeapTier == D3D12_RESOURCE_HEAP_TIER_2);

   const unsigned deny[] = {0, 4, 0x40, 0x80, 0xc0, 0x44, 0x84};
   const unsigned hints[] = {
      D3D12_HEAP_FLAG_CREATE_NOT_ZEROED,
      D3D12_HEAP_FLAG_CREATE_NOT_RESIDENT,
      D3D12_HEAP_FLAG_ALLOW_SHADER_ATOMICS,
      D3D12_HEAP_FLAG_TOOLS_USE_MANUAL_WRITE_TRACKING,
   };
   D3D12_RESOURCE_DESC buffer = {
      .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER, .Width = 512,
      .Height = 1, .DepthOrArraySize = 1, .MipLevels = 1,
      .SampleDesc = {.Count = 1}, .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
      .Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
   };
   D3D12_RESOURCE_DESC texture = {
      .Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D, .Width = 4,
      .Height = 4, .DepthOrArraySize = 1, .MipLevels = 1,
      .Format = DXGI_FORMAT_R8G8B8A8_UNORM, .SampleDesc = {.Count = 1},
   };
   D3D12_RESOURCE_ALLOCATION_INFO allocation;
#ifdef _WIN32
   ID3D12Device_GetResourceAllocationInfo(device, &allocation, 0, 1, &buffer);
#else
   allocation = ID3D12Device_GetResourceAllocationInfo(device, 0, 1, &buffer);
#endif
   CHECK(allocation.Alignment == 65536 && allocation.SizeInBytes == 65536);
   if (argc == 2 && !strcmp(argv[1], "--repro")) {
      D3D12_HEAP_DESC hd = {
         .SizeInBytes = 65536,
         .Properties = {.Type = D3D12_HEAP_TYPE_DEFAULT},
         .Flags = D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS |
                  D3D12_HEAP_FLAG_CREATE_NOT_ZEROED,
      };
      ID3D12Heap *heap = NULL;
      CHECK(ID3D12Device_CreateHeap(device, &hd, &IID_ID3D12Heap,
                                    (void **)&heap) == S_OK);
      if (heap) {
         placed(device, heap, 0, &buffer, S_OK);
         CHECK(ID3D12Heap_Release(heap) == 0);
      }
      CHECK(ID3D12Device_Release(device) == 0);
      printf("PLACED_REPRO_RESULT checks=%u failures=%u\n", checks, failures);
      return failures ? 1 : 0;
   }
   unsigned heaps = 0;
   for (unsigned mask = 0; mask < 16; ++mask) {
      unsigned flags = 0;
      for (unsigned bit = 0; bit < 4; ++bit)
         if (mask & (1u << bit))
            flags |= hints[bit];
      for (unsigned shared = 0; shared < 2; ++shared) {
         for (unsigned i = 0; i < sizeof(deny) / sizeof(deny[0]); ++i) {
            D3D12_HEAP_DESC hd = {
               .SizeInBytes = 131072,
               .Properties = {.Type = D3D12_HEAP_TYPE_DEFAULT},
               .Flags = flags | deny[i] | (shared ? D3D12_HEAP_FLAG_SHARED : 0),
            };
            ID3D12Heap *heap = NULL;
            HRESULT hr = ID3D12Device_CreateHeap(device, &hd, &IID_ID3D12Heap,
                                                 (void **)&heap);
            CHECK(hr == S_OK && heap);
            if (!heap)
               continue;
            heaps++;
            const uint64_t invalid_alignments[] = {1, 4096, 4194304};
            for (unsigned a = 0; a < 3; ++a) {
               buffer.Alignment = invalid_alignments[a];
               placed(device, heap, 0, &buffer, E_INVALIDARG);
            }
            for (unsigned explicit_align = 0; explicit_align < 2; ++explicit_align) {
               buffer.Alignment = explicit_align ? 65536 : 0;
               HRESULT expected = (deny[i] & D3D12_HEAP_FLAG_DENY_BUFFERS)
                                     ? E_INVALIDARG : S_OK;
               placed(device, heap, 0, &buffer, expected);
               placed(device, heap, 65536, &buffer, expected);
               placed(device, heap, 1, &buffer, E_INVALIDARG);
               placed(device, heap, 131072, &buffer, E_INVALIDARG);
               placed(device, heap, UINT64_MAX - 65535, &buffer, E_INVALIDARG);
            }
            texture.Flags = 0;
            placed(device, heap, 0, &texture,
               (deny[i] & D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES) ? E_INVALIDARG : S_OK);
            texture.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
            placed(device, heap, 0, &texture,
               (deny[i] & D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES) ? E_INVALIDARG : S_OK);
            CHECK(ID3D12Heap_Release(heap) == 0);
         }
      }
      D3D12_HEAP_DESC cross_desc = {
         .SizeInBytes = 65536,
         .Properties = {.Type = D3D12_HEAP_TYPE_DEFAULT},
         .Flags = flags | D3D12_HEAP_FLAG_SHARED |
                  D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER,
      };
      ID3D12Heap *cross_heap = NULL;
      CHECK(ID3D12Device_CreateHeap(device, &cross_desc, &IID_ID3D12Heap,
                                    (void **)&cross_heap) == S_OK);
      if (cross_heap) {
         heaps++;
         D3D12_RESOURCE_DESC cross_buffer = buffer;
         cross_buffer.Alignment = 0;
         cross_buffer.Flags |= D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER;
         placed(device, cross_heap, 0, &cross_buffer, S_OK);
         CHECK(ID3D12Heap_Release(cross_heap) == 0);
      }
   }
   /* Keep foreign COM objects out, including with tracing installed. */
   buffer.Alignment = 0;
   placed(device, (ID3D12Heap *)device, 0, &buffer, E_INVALIDARG);
   placed(device, NULL, 0, &buffer, E_INVALIDARG);
   CHECK(ID3D12Device_Release(device) == 0);
   printf("PLACED_RESOURCE_RESULT heaps=%u checks=%u failures=%u\n",
          heaps, checks, failures);
   return failures ? 1 : 0;
}
