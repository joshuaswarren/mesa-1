/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 * A single SO PSO stream omits topology, then captures points, lines and
 * triangles without rebinding the PSO. Check data, filled sizes and queries.
 */
#define COBJMACROS
#include <wsl/winadapter.h>
#include <directx/d3d12.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CALL(expr) do { if (FAILED(expr)) { \
   fprintf(stderr, "FAIL streamout_topology_live:%d: %s\n", __LINE__, #expr); \
   goto cleanup; } } while (0)
#define CHECK(expr) do { if (!(expr)) { \
   fprintf(stderr, "FAIL streamout_topology_live:%d: %s\n", __LINE__, #expr); \
   goto cleanup; } } while (0)
#define RELEASE(type, object) do { if (object) type##_Release(object); } while (0)

static HRESULT
buffer(ID3D12Device *device, D3D12_HEAP_TYPE type, UINT64 size,
       D3D12_RESOURCE_STATES state, ID3D12Resource **out)
{
   const D3D12_HEAP_PROPERTIES heap = {.Type = type};
   const D3D12_RESOURCE_DESC desc = {
      .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER, .Width = size,
      .Height = 1, .DepthOrArraySize = 1, .MipLevels = 1,
      .SampleDesc = {.Count = 1}, .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
   };
   return ID3D12Device_CreateCommittedResource(device, &heap,
      D3D12_HEAP_FLAG_NONE, &desc, state, NULL, &IID_ID3D12Resource, (void **)out);
}

static void
transition(ID3D12GraphicsCommandList *list, ID3D12Resource *resource,
           D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after)
{
   const D3D12_RESOURCE_BARRIER barrier = {
      .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
      .Transition = {.pResource = resource, .StateBefore = before,
                     .StateAfter = after,
                     .Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES},
   };
   ID3D12GraphicsCommandList_ResourceBarrier(list, 1, &barrier);
}

int
main(int argc, char **argv)
{
   if (argc != 2) { fprintf(stderr, "usage: %s tri.dxil\n", argv[0]); return 2; }
   int failed = 1;
   ID3D12Device *device = NULL;
   ID3D12Device2 *device2 = NULL;
   ID3D12RootSignature *root = NULL;
   ID3D12PipelineState *pso = NULL;
   ID3DBlob *blob = NULL;
   ID3D12CommandQueue *queue = NULL;
   ID3D12CommandAllocator *allocator = NULL;
   ID3D12GraphicsCommandList *list = NULL;
   ID3D12Fence *fence = NULL;
   ID3D12QueryHeap *queries = NULL;
   ID3D12Resource *data = NULL, *counters = NULL, *upload = NULL, *readback = NULL;
   void *shader = NULL, *mapped = NULL;
   FILE *input = fopen(argv[1], "rb");
   if (!input) return 2;
   fseek(input, 0, SEEK_END);
   long shader_size = ftell(input);
   rewind(input);
   if (shader_size > 0) shader = malloc((size_t)shader_size);
   bool loaded = shader && fread(shader, 1, (size_t)shader_size, input) == (size_t)shader_size;
   fclose(input);
   CHECK(loaded);

   CALL(D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0, &IID_ID3D12Device, (void **)&device));
   CALL(ID3D12Device_QueryInterface(device, &IID_ID3D12Device2, (void **)&device2));
   const D3D12_ROOT_SIGNATURE_DESC root_desc = {
      .Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_STREAM_OUTPUT,
   };
   CALL(D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1_0, &blob, NULL));
   CALL(ID3D12Device_CreateRootSignature(device, 0, ID3D10Blob_GetBufferPointer(blob),
      ID3D10Blob_GetBufferSize(blob), &IID_ID3D12RootSignature, (void **)&root));
   const D3D12_SO_DECLARATION_ENTRY declaration = {
      .SemanticName = "SV_Position", .ComponentCount = 4,
   };
   const UINT stride = 16;
   const struct {
      struct { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type; ID3D12RootSignature *value; } root;
      struct { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type; D3D12_SHADER_BYTECODE value; } vs;
      struct { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type; D3D12_STREAM_OUTPUT_DESC value; } so;
   } stream = {
      .root = {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, root},
      .vs = {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, {shader, (SIZE_T)shader_size}},
      .so = {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_STREAM_OUTPUT,
             {&declaration, 1, &stride, 1, D3D12_SO_NO_RASTERIZED_STREAM}},
   };
   const D3D12_PIPELINE_STATE_STREAM_DESC stream_desc = {sizeof(stream), (void *)&stream};
   CALL(ID3D12Device2_CreatePipelineState(device2, &stream_desc, &IID_ID3D12PipelineState, (void **)&pso));
   const D3D12_COMMAND_QUEUE_DESC queue_desc = {.Type = D3D12_COMMAND_LIST_TYPE_DIRECT};
   CALL(ID3D12Device_CreateCommandQueue(device, &queue_desc, &IID_ID3D12CommandQueue, (void **)&queue));
   CALL(ID3D12Device_CreateCommandAllocator(device, D3D12_COMMAND_LIST_TYPE_DIRECT,
      &IID_ID3D12CommandAllocator, (void **)&allocator));
   CALL(ID3D12Device_CreateCommandList(device, 0, D3D12_COMMAND_LIST_TYPE_DIRECT,
      allocator, pso, &IID_ID3D12GraphicsCommandList, (void **)&list));
   CALL(ID3D12Device_CreateFence(device, 0, D3D12_FENCE_FLAG_NONE, &IID_ID3D12Fence, (void **)&fence));
   const D3D12_QUERY_HEAP_DESC query_desc = {.Type = D3D12_QUERY_HEAP_TYPE_SO_STATISTICS, .Count = 3};
   CALL(ID3D12Device_CreateQueryHeap(device, &query_desc, &IID_ID3D12QueryHeap, (void **)&queries));
   CALL(buffer(device, D3D12_HEAP_TYPE_DEFAULT, 192, D3D12_RESOURCE_STATE_STREAM_OUT, &data));
   CALL(buffer(device, D3D12_HEAP_TYPE_DEFAULT, 24, D3D12_RESOURCE_STATE_COPY_DEST, &counters));
   CALL(buffer(device, D3D12_HEAP_TYPE_UPLOAD, 24, D3D12_RESOURCE_STATE_GENERIC_READ, &upload));
   CALL(buffer(device, D3D12_HEAP_TYPE_READBACK, 264, D3D12_RESOURCE_STATE_COPY_DEST, &readback));
   CALL(ID3D12Resource_Map(upload, 0, NULL, &mapped));
   memset(mapped, 0, 24);
   ID3D12Resource_Unmap(upload, 0, NULL); mapped = NULL;
   ID3D12GraphicsCommandList_CopyBufferRegion(list, counters, 0, upload, 0, 24);
   transition(list, counters, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_STREAM_OUT);
   ID3D12GraphicsCommandList_SetGraphicsRootSignature(list, root);
   const D3D12_PRIMITIVE_TOPOLOGY topologies[] = {
      D3D_PRIMITIVE_TOPOLOGY_POINTLIST, D3D_PRIMITIVE_TOPOLOGY_LINELIST,
      D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
   };
   const UINT vertices[] = {3, 2, 3}, primitives[] = {3, 1, 1};
   for (UINT i = 0; i < 3; i++) {
      const D3D12_STREAM_OUTPUT_BUFFER_VIEW view = {
         .BufferLocation = ID3D12Resource_GetGPUVirtualAddress(data) + i * 64,
         .SizeInBytes = 64,
         .BufferFilledSizeLocation = ID3D12Resource_GetGPUVirtualAddress(counters) + i * 8,
      };
      ID3D12GraphicsCommandList_SOSetTargets(list, 0, 1, &view);
      ID3D12GraphicsCommandList_IASetPrimitiveTopology(list, topologies[i]);
      ID3D12GraphicsCommandList_BeginQuery(list, queries, D3D12_QUERY_TYPE_SO_STATISTICS_STREAM0, i);
      ID3D12GraphicsCommandList_DrawInstanced(list, vertices[i], 1, 0, 0);
      ID3D12GraphicsCommandList_EndQuery(list, queries, D3D12_QUERY_TYPE_SO_STATISTICS_STREAM0, i);
   }
   transition(list, data, D3D12_RESOURCE_STATE_STREAM_OUT, D3D12_RESOURCE_STATE_COPY_SOURCE);
   transition(list, counters, D3D12_RESOURCE_STATE_STREAM_OUT, D3D12_RESOURCE_STATE_COPY_SOURCE);
   ID3D12GraphicsCommandList_CopyBufferRegion(list, readback, 0, data, 0, 192);
   ID3D12GraphicsCommandList_CopyBufferRegion(list, readback, 192, counters, 0, 24);
   ID3D12GraphicsCommandList_ResolveQueryData(list, queries, D3D12_QUERY_TYPE_SO_STATISTICS_STREAM0, 0, 3, readback, 216);
   CALL(ID3D12GraphicsCommandList_Close(list));
   ID3D12CommandList *lists[] = {(ID3D12CommandList *)list};
   ID3D12CommandQueue_ExecuteCommandLists(queue, 1, lists);
   CALL(ID3D12CommandQueue_Signal(queue, fence, 1));
   CALL(ID3D12Fence_SetEventOnCompletion(fence, 1, NULL));
   CALL(ID3D12Resource_Map(readback, 0, NULL, &mapped));
   const uint64_t *filled = (const void *)((const char *)mapped + 192);
   const D3D12_QUERY_DATA_SO_STATISTICS *statistics = (const void *)((const char *)mapped + 216);
   for (UINT i = 0; i < 3; i++) {
      CHECK(filled[i] == vertices[i] * 16);
      CHECK(statistics[i].NumPrimitivesWritten == primitives[i]);
      CHECK(statistics[i].PrimitivesStorageNeeded == primitives[i]);
      CHECK(!memcmp(mapped, (const char *)mapped + i * 64, vertices[i] * 16));
   }
   CHECK(((const float *)mapped)[3] == 1.0f);
   ID3D12Resource_Unmap(readback, 0, NULL); mapped = NULL;
   failed = 0;
cleanup:
   if (mapped) ID3D12Resource_Unmap(readback, 0, NULL);
   RELEASE(ID3D12GraphicsCommandList, list); RELEASE(ID3D12CommandAllocator, allocator);
   RELEASE(ID3D12Fence, fence); RELEASE(ID3D12CommandQueue, queue);
   RELEASE(ID3D12Resource, data); RELEASE(ID3D12Resource, counters);
   RELEASE(ID3D12Resource, upload); RELEASE(ID3D12Resource, readback);
   RELEASE(ID3D12QueryHeap, queries); RELEASE(ID3D12PipelineState, pso);
   RELEASE(ID3D12RootSignature, root); RELEASE(ID3D10Blob, blob);
   RELEASE(ID3D12Device2, device2); RELEASE(ID3D12Device, device);
   free(shader);
   printf("STREAM OUTPUT TOPOLOGY: %s\n", failed ? "FAILURES" : "ALL PASS");
   return failed;
}
