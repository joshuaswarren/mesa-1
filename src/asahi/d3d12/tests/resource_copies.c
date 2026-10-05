/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT */
#define COBJMACROS
#include <wsl/winadapter.h>
#include <directx/d3d12.h>
#include "d12_test_abi.h"
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

static unsigned failures;
#define CHECK(c) do { if (!(c)) { \
   printf("FAIL resource copies at line %d\n", __LINE__); failures++; \
} } while (0)

static ID3D12Resource *
make_resource(ID3D12Device *device, const D3D12_RESOURCE_DESC *desc,
              D3D12_HEAP_TYPE type, D3D12_RESOURCE_STATES state)
{
   D3D12_HEAP_PROPERTIES properties = {.Type = type};
   ID3D12Resource *resource = NULL;
   HRESULT hr = ID3D12Device_CreateCommittedResource(device, &properties, 0,
      desc, state, NULL, &IID_ID3D12Resource, (void **)&resource);
   CHECK(hr == S_OK && resource);
   return resource;
}

static void
transition(ID3D12GraphicsCommandList *list, ID3D12Resource *resource,
            D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after)
{
   D3D12_RESOURCE_BARRIER barrier = {
      .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
      .Transition = {.pResource = resource,
         .Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
         .StateBefore = before, .StateAfter = after},
   };
   ID3D12GraphicsCommandList_ResourceBarrier(list, 1, &barrier);
}

static uint8_t
pattern(unsigned subresource, unsigned z, unsigned row, unsigned byte)
{
   return (subresource * 59 + z * 31 + row * 13 + byte * 7) ^ (byte >> 8);
}

static void
copy_texture(ID3D12Device *device, ID3D12CommandQueue *queue,
              D3D12_RESOURCE_DESC desc, bool tiled_destination)
{
   ID3D12Resource *source = NULL, *destination = NULL;
   ID3D12Resource *upload = NULL, *readback = NULL;
   ID3D12CommandAllocator *allocator = NULL;
   ID3D12GraphicsCommandList *list = NULL;
   ID3D12Fence *fence = NULL;
   source = make_resource(device, &desc, D3D12_HEAP_TYPE_DEFAULT,
                           D3D12_RESOURCE_STATE_COPY_DEST);
   if (tiled_destination)
      desc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
   destination = make_resource(device, &desc, D3D12_HEAP_TYPE_DEFAULT,
                                D3D12_RESOURCE_STATE_COPY_DEST);
   if (!source || !destination)
      goto done;
   unsigned count = desc.MipLevels *
      (desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D
          ? 1 : desc.DepthOrArraySize);
   D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprints[16];
   UINT rows[16];
   UINT64 row_sizes[16], total;
   CHECK(count <= 16);
   if (count > 16)
      goto done;
   ID3D12Device_GetCopyableFootprints(device, &desc, 0, count, 0,
                                    footprints, rows, row_sizes, &total);
   CHECK(total != UINT64_MAX && total);
   if (total == UINT64_MAX || !total)
      goto done;
   D3D12_RESOURCE_DESC buffer = {
      .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER, .Width = total,
      .Height = 1, .DepthOrArraySize = 1, .MipLevels = 1,
      .SampleDesc = {.Count = 1}, .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
   };
   upload = make_resource(device, &buffer, D3D12_HEAP_TYPE_UPLOAD,
                           D3D12_RESOURCE_STATE_GENERIC_READ);
   readback = make_resource(device, &buffer, D3D12_HEAP_TYPE_READBACK,
                             D3D12_RESOURCE_STATE_COPY_DEST);
   if (!upload || !readback)
      goto done;
   uint8_t *mapped = NULL;
   HRESULT hr = ID3D12Resource_Map(upload, 0, NULL, (void **)&mapped);
   CHECK(hr == S_OK && mapped);
   if (FAILED(hr) || !mapped)
      goto done;
   for (unsigned s = 0; s < count; s++) {
      const D3D12_PLACED_SUBRESOURCE_FOOTPRINT *f = &footprints[s];
      for (unsigned z = 0; z < f->Footprint.Depth; z++)
         for (unsigned y = 0; y < rows[s]; y++)
            for (unsigned x = 0; x < row_sizes[s]; x++)
               mapped[f->Offset + (z * rows[s] + y) * f->Footprint.RowPitch + x] =
                  pattern(s, z, y, x);
   }
   ID3D12Resource_Unmap(upload, 0, NULL);
   hr = ID3D12Device_CreateCommandAllocator(device, D3D12_COMMAND_LIST_TYPE_DIRECT,
      &IID_ID3D12CommandAllocator, (void **)&allocator);
   CHECK(hr == S_OK && allocator);
   if (FAILED(hr) || !allocator)
      goto done;
   hr = ID3D12Device_CreateCommandList(device, 0, D3D12_COMMAND_LIST_TYPE_DIRECT,
      allocator, NULL, &IID_ID3D12GraphicsCommandList, (void **)&list);
   CHECK(hr == S_OK && list);
   if (FAILED(hr) || !list)
      goto done;
   for (unsigned s = 0; s < count; s++) {
      D3D12_TEXTURE_COPY_LOCATION from = {
         .pResource = upload, .Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
         .PlacedFootprint = footprints[s],
      }, to = {
         .pResource = source, .Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
         .SubresourceIndex = s,
      };
      ID3D12GraphicsCommandList_CopyTextureRegion(list, &to, 0, 0, 0, &from, NULL);
   }
   transition(list, source, D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_COPY_SOURCE);
   ID3D12GraphicsCommandList_CopyResource(list, destination, source);
   transition(list, destination, D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_COPY_SOURCE);
   for (unsigned s = 0; s < count; s++) {
      D3D12_TEXTURE_COPY_LOCATION from = {
         .pResource = destination, .Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
         .SubresourceIndex = s,
      }, to = {
         .pResource = readback, .Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
         .PlacedFootprint = footprints[s],
      };
      ID3D12GraphicsCommandList_CopyTextureRegion(list, &to, 0, 0, 0, &from, NULL);
   }
   hr = ID3D12GraphicsCommandList_Close(list);
   CHECK(hr == S_OK);
   if (FAILED(hr))
      goto done;
   hr = ID3D12Device_CreateFence(device, 0, 0, &IID_ID3D12Fence, (void **)&fence);
   CHECK(hr == S_OK && fence);
   if (FAILED(hr) || !fence)
      goto done;
   ID3D12CommandList *lists[] = {(ID3D12CommandList *)list};
   ID3D12CommandQueue_ExecuteCommandLists(queue, 1, lists);
   hr = ID3D12CommandQueue_Signal(queue, fence, 1);
   CHECK(hr == S_OK);
   if (FAILED(hr))
      goto done;
   hr = ID3D12Fence_SetEventOnCompletion(fence, 1, NULL);
   CHECK(hr == S_OK && ID3D12Fence_GetCompletedValue(fence) == 1);
   if (FAILED(hr))
      goto done;
   mapped = NULL;
   hr = ID3D12Resource_Map(readback, 0, NULL, (void **)&mapped);
   CHECK(hr == S_OK && mapped);
   if (FAILED(hr) || !mapped)
      goto done;
   bool matches = true;
   for (unsigned s = 0; s < count; s++) {
      const D3D12_PLACED_SUBRESOURCE_FOOTPRINT *f = &footprints[s];
      for (unsigned z = 0; z < f->Footprint.Depth; z++)
         for (unsigned y = 0; y < rows[s]; y++)
            for (unsigned x = 0; x < row_sizes[s]; x++)
               matches &= mapped[f->Offset +
                  (z * rows[s] + y) * f->Footprint.RowPitch + x] == pattern(s, z, y, x);
   }
   CHECK(matches);
   ID3D12Resource_Unmap(readback, 0, NULL);
done:
   if (fence) ID3D12Fence_Release(fence);
   if (list) ID3D12GraphicsCommandList_Release(list);
   if (allocator) ID3D12CommandAllocator_Release(allocator);
   if (readback) ID3D12Resource_Release(readback);
   if (upload) ID3D12Resource_Release(upload);
   if (destination) ID3D12Resource_Release(destination);
   if (source) ID3D12Resource_Release(source);
}

/* Observe copied DSV descriptors on the GPU, including reuse of overlapping
 * placed resources. Each activation is followed by a full clear as required
 * by D3D12's simple aliasing model; no undefined data inheritance is assumed.
 */
static void
alias_depth(ID3D12Device *device, ID3D12CommandQueue *queue)
{
   ID3D12Heap *heap = NULL;
   ID3D12Resource *resources[2] = {NULL}, *readback = NULL;
   ID3D12DescriptorHeap *src_heap = NULL, *dst_heap = NULL;
   ID3D12CommandAllocator *allocator = NULL;
   ID3D12GraphicsCommandList *list = NULL;
   ID3D12Fence *fence = NULL;
   D3D12_RESOURCE_DESC desc = {
      .Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
      .Width = 8, .Height = 8, .DepthOrArraySize = 1, .MipLevels = 1,
      .Format = DXGI_FORMAT_D32_FLOAT, .SampleDesc = {.Count = 1},
      .Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
   };
#ifdef _WIN32
   D3D12_RESOURCE_ALLOCATION_INFO info;
   ID3D12Device_GetResourceAllocationInfo(device, &info, 0, 1, &desc);
#else
   D3D12_RESOURCE_ALLOCATION_INFO info =
      ID3D12Device_GetResourceAllocationInfo(device, 0, 1, &desc);
#endif
   if (info.SizeInBytes == UINT64_MAX) { CHECK(false); goto done; }
   D3D12_HEAP_DESC hd = {
      .SizeInBytes = info.SizeInBytes, .Alignment = info.Alignment,
      .Properties = {.Type = D3D12_HEAP_TYPE_DEFAULT},
   };
   HRESULT hr = ID3D12Device_CreateHeap(device, &hd, &IID_ID3D12Heap, (void **)&heap);
   CHECK(hr == S_OK && heap);
   if (FAILED(hr) || !heap) goto done;
   for (unsigned i = 0; i < 2; i++) {
      hr = ID3D12Device_CreatePlacedResource(device, heap, 0, &desc,
         D3D12_RESOURCE_STATE_DEPTH_WRITE, NULL, &IID_ID3D12Resource,
         (void **)&resources[i]);
      CHECK(hr == S_OK && resources[i]);
      if (FAILED(hr) || !resources[i]) goto done;
   }
   D3D12_DESCRIPTOR_HEAP_DESC dd = {
      .Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV, .NumDescriptors = 2,
   };
   hr = ID3D12Device_CreateDescriptorHeap(device, &dd, &IID_ID3D12DescriptorHeap,
                                         (void **)&src_heap);
   CHECK(hr == S_OK && src_heap);
   if (FAILED(hr) || !src_heap) goto done;
   dd.NumDescriptors = 4;
   hr = ID3D12Device_CreateDescriptorHeap(device, &dd, &IID_ID3D12DescriptorHeap,
                                         (void **)&dst_heap);
   CHECK(hr == S_OK && dst_heap);
   if (FAILED(hr) || !dst_heap) goto done;
   UINT stride = ID3D12Device_GetDescriptorHandleIncrementSize(device, dd.Type);
   D3D12_CPU_DESCRIPTOR_HANDLE src = d12_test_descriptor_heap_get_cpu_start(src_heap);
   D3D12_CPU_DESCRIPTOR_HANDLE dst = d12_test_descriptor_heap_get_cpu_start(dst_heap);
   D3D12_CPU_DESCRIPTOR_HANDLE sources[2] = {src, {.ptr = src.ptr + stride}};
   for (unsigned i = 0; i < 2; i++)
      ID3D12Device_CreateDepthStencilView(device, resources[i], NULL, sources[i]);
   UINT two = 2;
   ID3D12Device_CopyDescriptors(device, 1, &dst, &two, 2, sources, NULL, dd.Type);
   ID3D12Device_CopyDescriptorsSimple(device, 2,
      (D3D12_CPU_DESCRIPTOR_HANDLE){.ptr = dst.ptr + 2 * stride}, src, dd.Type);

   D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint;
   UINT64 total;
   UINT rows;
   ID3D12Device_GetCopyableFootprints(device, &desc, 0, 1, 0,
                                     &footprint, &rows, NULL, &total);
   CHECK(total != UINT64_MAX && rows == 8);
   if (total == UINT64_MAX || rows != 8) goto done;
   UINT64 step = (total + 511) & ~511ull;
   D3D12_RESOURCE_DESC bd = {
      .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER, .Width = 3 * step,
      .Height = 1, .DepthOrArraySize = 1, .MipLevels = 1,
      .SampleDesc = {.Count = 1}, .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
   };
   readback = make_resource(device, &bd, D3D12_HEAP_TYPE_READBACK,
                             D3D12_RESOURCE_STATE_COPY_DEST);
   if (!readback) goto done;
   hr = ID3D12Device_CreateCommandAllocator(device, D3D12_COMMAND_LIST_TYPE_DIRECT,
      &IID_ID3D12CommandAllocator, (void **)&allocator);
   CHECK(hr == S_OK && allocator);
   if (FAILED(hr) || !allocator) goto done;
   hr = ID3D12Device_CreateCommandList(device, 0, D3D12_COMMAND_LIST_TYPE_DIRECT,
      allocator, NULL, &IID_ID3D12GraphicsCommandList, (void **)&list);
   CHECK(hr == S_OK && list);
   if (FAILED(hr) || !list) goto done;
   const float depths[] = {0.25f, 0.75f, 0.5f};
   for (unsigned phase = 0; phase < 3; phase++) {
      ID3D12Resource *resource = resources[phase & 1];
      D3D12_RESOURCE_BARRIER alias = {
         .Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING,
         .Aliasing = {.pResourceBefore = phase ? resources[(phase - 1) & 1] : NULL,
                      .pResourceAfter = resource},
      };
      /* Also exercise the legal global NULL/NULL aliasing boundary. */
      if (phase == 2) alias.Aliasing.pResourceBefore = alias.Aliasing.pResourceAfter = NULL;
      ID3D12GraphicsCommandList_ResourceBarrier(list, 1, &alias);
      if (phase == 2)
         transition(list, resource, D3D12_RESOURCE_STATE_COPY_SOURCE,
                      D3D12_RESOURCE_STATE_DEPTH_WRITE);
      D3D12_CPU_DESCRIPTOR_HANDLE handle = {.ptr = dst.ptr + phase * stride};
      ID3D12GraphicsCommandList_ClearDepthStencilView(list, handle,
         D3D12_CLEAR_FLAG_DEPTH, depths[phase], 0, 0, NULL);
      transition(list, resource, D3D12_RESOURCE_STATE_DEPTH_WRITE,
                   D3D12_RESOURCE_STATE_COPY_SOURCE);
      D3D12_TEXTURE_COPY_LOCATION from = {
         .pResource = resource, .Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
      }, to = {
         .pResource = readback, .Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
         .PlacedFootprint = footprint,
      };
      to.PlacedFootprint.Offset += phase * step;
      ID3D12GraphicsCommandList_CopyTextureRegion(list, &to, 0, 0, 0, &from, NULL);
   }
   hr = ID3D12GraphicsCommandList_Close(list);
   CHECK(hr == S_OK);
   if (FAILED(hr)) goto done;
   hr = ID3D12Device_CreateFence(device, 0, 0, &IID_ID3D12Fence, (void **)&fence);
   CHECK(hr == S_OK && fence);
   if (FAILED(hr) || !fence) goto done;
   ID3D12CommandList *lists[] = {(ID3D12CommandList *)list};
   ID3D12CommandQueue_ExecuteCommandLists(queue, 1, lists);
   hr = ID3D12CommandQueue_Signal(queue, fence, 1);
   CHECK(hr == S_OK);
   if (FAILED(hr)) goto done;
   hr = ID3D12Fence_SetEventOnCompletion(fence, 1, NULL);
   CHECK(hr == S_OK && ID3D12Fence_GetCompletedValue(fence) == 1);
   if (FAILED(hr)) goto done;
   uint8_t *mapped = NULL;
   hr = ID3D12Resource_Map(readback, 0, NULL, (void **)&mapped);
   CHECK(hr == S_OK && mapped);
   if (FAILED(hr) || !mapped) goto done;
   bool matches = true;
   for (unsigned phase = 0; phase < 3; phase++)
      for (unsigned y = 0; y < 8; y++) {
         const float *row = (const float *)(mapped + phase * step +
                                            y * footprint.Footprint.RowPitch);
         for (unsigned x = 0; x < 8; x++) matches &= row[x] == depths[phase];
      }
   CHECK(matches);
   ID3D12Resource_Unmap(readback, 0, NULL);
done:
   if (fence) ID3D12Fence_Release(fence);
   if (list) ID3D12GraphicsCommandList_Release(list);
   if (allocator) ID3D12CommandAllocator_Release(allocator);
   if (dst_heap) ID3D12DescriptorHeap_Release(dst_heap);
   if (src_heap) ID3D12DescriptorHeap_Release(src_heap);
   if (readback) ID3D12Resource_Release(readback);
   for (unsigned i = 0; i < 2; i++)
      if (resources[i]) ID3D12Resource_Release(resources[i]);
   if (heap) ID3D12Heap_Release(heap);
}

int main(void)
{
   ID3D12Device *device = NULL;
   HRESULT hr = D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0,
                                 &IID_ID3D12Device, (void **)&device);
   if (FAILED(hr) || !device)
      return 1;
   ID3D12CommandQueue *queue = NULL;
   D3D12_COMMAND_QUEUE_DESC qdesc = {.Type = D3D12_COMMAND_LIST_TYPE_DIRECT};
   hr = ID3D12Device_CreateCommandQueue(device, &qdesc,
                                       &IID_ID3D12CommandQueue, (void **)&queue);
   CHECK(hr == S_OK && queue);
   if (queue) {
      D3D12_RESOURCE_DESC desc = {
         .Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
         .Width = 19, .Height = 9, .DepthOrArraySize = 1, .MipLevels = 1,
         .Format = DXGI_FORMAT_R8G8B8A8_UINT, .SampleDesc = {.Count = 1},
      };
      copy_texture(device, queue, desc, true); /* linear -> twiddled */
      desc.MipLevels = 3;
      desc.DepthOrArraySize = 2;
      copy_texture(device, queue, desc, false); /* tiled array mip tails */
      desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE1D;
      desc.Height = 1;
      copy_texture(device, queue, desc, false);
      desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
      desc.Height = 9;
      desc.DepthOrArraySize = 5;
      copy_texture(device, queue, desc, false); /* minified Z slices */
      desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
      desc.Width = 20;
      desc.Height = 12;
      desc.DepthOrArraySize = 2;
      desc.Format = DXGI_FORMAT_BC1_UNORM;
      copy_texture(device, queue, desc, false);
      desc.Format = DXGI_FORMAT_BC3_UNORM;
      copy_texture(device, queue, desc, false);
      alias_depth(device, queue);
      ID3D12CommandQueue_Release(queue);
   }
   CHECK(ID3D12Device_Release(device) == 0);
   printf("RESOURCE COPIES: %s\n", failures ? "FAILURES" : "ALL PASS");
   return failures ? 1 : 0;
}
