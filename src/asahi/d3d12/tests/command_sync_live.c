/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#define _POSIX_C_SOURCE 200809L
#define COBJMACROS
#include <wsl/winadapter.h>
#include <directx/d3d12.h>
#include <inttypes.h>
#include <stdio.h>
#include <time.h>
#include <poll.h>
#include <sys/eventfd.h>
#include <unistd.h>

#define REQUIRE(condition) do { \
   if (!(condition)) { \
      fprintf(stderr, "FAIL command_sync_live:%d: %s\n", __LINE__, #condition); \
      goto cleanup; \
   } \
} while (0)
#define CALL(expression) REQUIRE(SUCCEEDED(expression))
#define RELEASE(type, object) do { if (object) type##_Release(object); } while (0)

static int
run(D3D12_COMMAND_LIST_TYPE type)
{
   int failed = 1, event = -1;
   ID3D12Device *device = NULL;
   ID3D12CommandQueue *queue = NULL;
   ID3D12CommandAllocator *allocator = NULL;
   ID3D12GraphicsCommandList *list = NULL;
   ID3D12QueryHeap *heap = NULL;
   ID3D12Resource *readback = NULL;
   ID3D12Fence *fence = NULL;
   uint64_t *mapped = NULL;
   CALL(D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0, &IID_ID3D12Device,
                          (void **)&device));
   const D3D_ROOT_SIGNATURE_VERSION versions[] = {
      D3D_ROOT_SIGNATURE_VERSION_1_0, D3D_ROOT_SIGNATURE_VERSION_1_1,
      D3D_ROOT_SIGNATURE_VERSION_1_2,
   };
   for (unsigned i = 0; i < 3; i++) {
      D3D12_FEATURE_DATA_ROOT_SIGNATURE support = {.HighestVersion = versions[i]};
      CALL(ID3D12Device_CheckFeatureSupport(device, D3D12_FEATURE_ROOT_SIGNATURE,
                                             &support, sizeof(support)));
      REQUIRE(support.HighestVersion ==
              (i == 0 ? D3D_ROOT_SIGNATURE_VERSION_1_0
                      : D3D_ROOT_SIGNATURE_VERSION_1_1));
   }
   /* Both wire versions must reject ALL/VS aliases and preserve a legal
    * PS/VS reuse of the same constant-buffer register. */
   for (unsigned version = 1; version < 3; version++) {
      D3D12_ROOT_PARAMETER1 parameters[2] = {
         {.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
          .ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL,
          .Constants = {.Num32BitValues = 4}},
         {.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
          .ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX,
          .Constants = {.Num32BitValues = 4}},
      };
      for (unsigned valid = 0; valid < 2; valid++) {
         parameters[0].ShaderVisibility = valid ? D3D12_SHADER_VISIBILITY_PIXEL
                                                : D3D12_SHADER_VISIBILITY_ALL;
         D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc = {.Version = versions[version]};
         if (version == 1) {
            desc.Desc_1_1.NumParameters = 2;
            desc.Desc_1_1.pParameters = parameters;
         } else {
            desc.Desc_1_2.NumParameters = 2;
            desc.Desc_1_2.pParameters = parameters;
         }
         ID3DBlob *serialized = NULL;
         ID3D12RootSignature *signature = NULL;
         CALL(D3D12SerializeVersionedRootSignature(&desc, &serialized, NULL));
         HRESULT hr = ID3D12Device_CreateRootSignature(
            device, 0, ID3D10Blob_GetBufferPointer(serialized),
            ID3D10Blob_GetBufferSize(serialized), &IID_ID3D12RootSignature,
            (void **)&signature);
         RELEASE(ID3D10Blob, serialized);
         RELEASE(ID3D12RootSignature, signature);
         REQUIRE(valid ? hr == S_OK : hr == E_INVALIDARG);
      }
   }
   const D3D12_COMMAND_QUEUE_DESC queue_desc = {.Type = type};
   CALL(ID3D12Device_CreateCommandQueue(device, &queue_desc,
          &IID_ID3D12CommandQueue, (void **)&queue));
   CALL(ID3D12Device_CreateCommandAllocator(device, type,
          &IID_ID3D12CommandAllocator, (void **)&allocator));
   CALL(ID3D12Device_CreateCommandList(device, 0, type, allocator, NULL,
          &IID_ID3D12GraphicsCommandList, (void **)&list));
   /* A list retains allocator storage internally, without adding a public
    * COM reference. Keep the API count separate from pool ownership. */
   REQUIRE(ID3D12CommandAllocator_AddRef(allocator) == 2);
   REQUIRE(ID3D12CommandAllocator_Release(allocator) == 1);
   CALL(ID3D12Device_CreateFence(device, 0, D3D12_FENCE_FLAG_NONE,
          &IID_ID3D12Fence, (void **)&fence));

   event = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
   REQUIRE(event > 0);
   HANDLE handle = (HANDLE)(intptr_t)event;
   struct pollfd pfd = {.fd = event, .events = POLLIN};
   uint64_t token;
   CALL(ID3D12Fence_SetEventOnCompletion(fence, 3, handle));
   CALL(ID3D12Fence_Signal(fence, 2));
   REQUIRE(poll(&pfd, 1, 0) == 0);
   CALL(ID3D12Fence_Signal(fence, 3));
   REQUIRE(poll(&pfd, 1, 1000) == 1 && (pfd.revents & POLLIN));
   REQUIRE(read(event, &token, sizeof(token)) == sizeof(token) && token == 1);
   CALL(ID3D12Fence_SetEventOnCompletion(fence, 3, handle));
   CALL(ID3D12Fence_SetEventOnCompletion(fence, 3, handle));
   REQUIRE(read(event, &token, sizeof(token)) == sizeof(token) && token == 1);
   CALL(ID3D12Fence_Signal(fence, 0));

   const D3D12_QUERY_HEAP_DESC query_desc = {
      .Type = type == D3D12_COMMAND_LIST_TYPE_COPY
                 ? D3D12_QUERY_HEAP_TYPE_COPY_QUEUE_TIMESTAMP
                 : D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
      .Count = 1,
   };
   CALL(ID3D12Device_CreateQueryHeap(device, &query_desc,
          &IID_ID3D12QueryHeap, (void **)&heap));
   const D3D12_HEAP_PROPERTIES properties = {.Type = D3D12_HEAP_TYPE_READBACK};
   const D3D12_RESOURCE_DESC resource_desc = {
      .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
      .Width = sizeof(uint64_t), .Height = 1, .DepthOrArraySize = 1,
      .MipLevels = 1, .SampleDesc = {.Count = 1},
      .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
   };
   CALL(ID3D12Device_CreateCommittedResource(device, &properties,
          D3D12_HEAP_FLAG_NONE, &resource_desc, D3D12_RESOURCE_STATE_COPY_DEST,
          NULL, &IID_ID3D12Resource, (void **)&readback));
   ID3D12GraphicsCommandList_EndQuery(list, heap, D3D12_QUERY_TYPE_TIMESTAMP, 0);
   ID3D12GraphicsCommandList_ResolveQueryData(list, heap,
      D3D12_QUERY_TYPE_TIMESTAMP, 0, 1, readback, 0);
   CALL(ID3D12GraphicsCommandList_Close(list));

   uint64_t first = 0;
   for (uint64_t iteration = 1; iteration <= 2; iteration++) {
      const struct timespec delay = {.tv_nsec = 40 * 1000 * 1000};
      nanosleep(&delay, NULL);
      UINT64 lower_bound, cpu_timestamp;
      CALL(ID3D12CommandQueue_GetClockCalibration(queue, &lower_bound,
                                                   &cpu_timestamp));
      ID3D12CommandList *lists[] = {(ID3D12CommandList *)list};
      ID3D12CommandQueue_ExecuteCommandLists(queue, 1, lists);
      CALL(ID3D12CommandQueue_Signal(queue, fence, iteration));
      CALL(ID3D12Fence_SetEventOnCompletion(fence, iteration, NULL));
      CALL(ID3D12Resource_Map(readback, 0, NULL, (void **)&mapped));
      uint64_t value = *mapped;
      ID3D12Resource_Unmap(readback, 0, NULL);
      mapped = NULL;
      printf("type %u execution %" PRIu64 ": timestamp %" PRIu64
             " >= calibration %" PRIu64 "\n", (unsigned)type, iteration,
             value, (uint64_t)lower_bound);
      REQUIRE(value >= lower_bound);
      REQUIRE(iteration == 1 || value > first);
      first = value;
   }
   failed = 0;
cleanup:
   if (mapped) ID3D12Resource_Unmap(readback, 0, NULL);
   if (event >= 0) close(event);
   RELEASE(ID3D12Fence, fence);
   RELEASE(ID3D12GraphicsCommandList, list);
   RELEASE(ID3D12CommandAllocator, allocator);
   RELEASE(ID3D12QueryHeap, heap);
   RELEASE(ID3D12Resource, readback);
   RELEASE(ID3D12CommandQueue, queue);
   RELEASE(ID3D12Device, device);
   return failed;
}

int
main(void)
{
   int failures = run(D3D12_COMMAND_LIST_TYPE_DIRECT);
   failures += run(D3D12_COMMAND_LIST_TYPE_COMPUTE);
   failures += run(D3D12_COMMAND_LIST_TYPE_COPY);
   printf("COMMAND SYNC: %s\n", failures ? "FAILURES" : "ALL PASS");
   return failures != 0;
}
