/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 * Recording delays and replay must not become GPU execution timestamps.
 */
#define COBJMACROS
#include <windows.h>
#include <d3d12.h>
#include <inttypes.h>
#include <stdio.h>

#define CALL(expr) do { HRESULT hr_ = (expr); if (FAILED(hr_)) { \
   fprintf(stderr, "%s: 0x%08lx\n", #expr, (unsigned long)hr_); goto done; } } while (0)

static int run(D3D12_COMMAND_LIST_TYPE type)
{
   int rc = 1;
   ID3D12Device *dev = NULL;
   ID3D12CommandQueue *queue = NULL;
   ID3D12CommandAllocator *alloc = NULL;
   ID3D12GraphicsCommandList *list = NULL;
   ID3D12QueryHeap *heap = NULL;
   ID3D12Resource *readback = NULL;
   ID3D12Resource *target = NULL;
   ID3D12DescriptorHeap *rtv = NULL;
   ID3D12Fence *fence = NULL;
   HANDLE event = NULL;
   CALL(D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0, &IID_ID3D12Device, (void **)&dev));
   D3D12_COMMAND_QUEUE_DESC qdesc = {.Type = type};
   CALL(ID3D12Device_CreateCommandQueue(dev, &qdesc, &IID_ID3D12CommandQueue, (void **)&queue));
   CALL(ID3D12Device_CreateCommandAllocator(dev, type, &IID_ID3D12CommandAllocator, (void **)&alloc));
   CALL(ID3D12Device_CreateCommandList(dev, 0, type, alloc, NULL,
                                    &IID_ID3D12GraphicsCommandList, (void **)&list));
   D3D12_QUERY_HEAP_DESC hdesc = {.Type = type == D3D12_COMMAND_LIST_TYPE_COPY
      ? D3D12_QUERY_HEAP_TYPE_COPY_QUEUE_TIMESTAMP : D3D12_QUERY_HEAP_TYPE_TIMESTAMP, .Count = 1};
   CALL(ID3D12Device_CreateQueryHeap(dev, &hdesc, &IID_ID3D12QueryHeap, (void **)&heap));
   D3D12_HEAP_PROPERTIES hp = {.Type = D3D12_HEAP_TYPE_READBACK};
   D3D12_RESOURCE_DESC rd = {.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
      .Width = 32, .Height = 1, .DepthOrArraySize = 1, .MipLevels = 1,
      .SampleDesc = {.Count = 1}, .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR};
   CALL(ID3D12Device_CreateCommittedResource(dev, &hp, D3D12_HEAP_FLAG_NONE, &rd,
      D3D12_RESOURCE_STATE_COPY_DEST, NULL, &IID_ID3D12Resource, (void **)&readback));
   CALL(ID3D12Device_CreateFence(dev, 0, D3D12_FENCE_FLAG_NONE, &IID_ID3D12Fence, (void **)&fence));
   if (type == D3D12_COMMAND_LIST_TYPE_DIRECT) {
      D3D12_HEAP_PROPERTIES def = {.Type = D3D12_HEAP_TYPE_DEFAULT};
      D3D12_RESOURCE_DESC tex = {.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
         .Width = 32, .Height = 16, .DepthOrArraySize = 1, .MipLevels = 1,
         .Format = DXGI_FORMAT_R8G8B8A8_UNORM, .SampleDesc = {.Count = 1},
         .Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN, .Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET};
      CALL(ID3D12Device_CreateCommittedResource(dev, &def, D3D12_HEAP_FLAG_NONE, &tex,
         D3D12_RESOURCE_STATE_RENDER_TARGET, NULL, &IID_ID3D12Resource, (void **)&target));
      D3D12_DESCRIPTOR_HEAP_DESC hd = {.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV, .NumDescriptors = 1};
      CALL(ID3D12Device_CreateDescriptorHeap(dev, &hd, &IID_ID3D12DescriptorHeap, (void **)&rtv));
   }
   event = CreateEventW(NULL, FALSE, FALSE, NULL);
   if (!event) goto done;
   ID3D12GraphicsCommandList_EndQuery(list, heap, D3D12_QUERY_TYPE_TIMESTAMP, 0);
   ID3D12GraphicsCommandList_ResolveQueryData(list, heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 1, readback, 0);
   Sleep(150);
   if (target) {
      D3D12_CPU_DESCRIPTOR_HANDLE handle;
      rtv->lpVtbl->GetCPUDescriptorHandleForHeapStart(rtv, &handle);
      ID3D12Device_CreateRenderTargetView(dev, target, NULL, handle);
      float color[] = {0.25f, 0.5f, 0.75f, 1.0f};
      ID3D12GraphicsCommandList_ClearRenderTargetView(list, handle, color, 0, NULL);
   }
   ID3D12GraphicsCommandList_EndQuery(list, heap, D3D12_QUERY_TYPE_TIMESTAMP, 0);
   ID3D12GraphicsCommandList_ResolveQueryData(list, heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 1, readback, 8);
   CALL(ID3D12GraphicsCommandList_Close(list));
   UINT64 frequency = 0, previous = 0;
   CALL(ID3D12CommandQueue_GetTimestampFrequency(queue, &frequency));
   if (!frequency) goto done;
   for (UINT64 replay = 1; replay <= 2; replay++) {
      Sleep(150);
      UINT64 before = 0, cpu = 0;
      LARGE_INTEGER qpc_before, qpc_after;
      QueryPerformanceCounter(&qpc_before);
      CALL(ID3D12CommandQueue_GetClockCalibration(queue, &before, &cpu));
      QueryPerformanceCounter(&qpc_after);
      if (cpu < (UINT64)qpc_before.QuadPart || cpu > (UINT64)qpc_after.QuadPart) {
         fprintf(stderr, "clock calibration CPU value is outside the QPC interval\n");
         goto done;
      }
      ID3D12CommandList *lists[] = {(ID3D12CommandList *)list};
      ID3D12CommandQueue_ExecuteCommandLists(queue, 1, lists);
      CALL(ID3D12CommandQueue_Signal(queue, fence, replay));
      CALL(ID3D12Fence_SetEventOnCompletion(fence, replay, event));
      if (WaitForSingleObject(event, 5000) != WAIT_OBJECT_0) goto done;
      UINT64 *values = NULL;
      D3D12_RANGE range = {0, 16};
      CALL(ID3D12Resource_Map(readback, 0, &range, (void **)&values));
      UINT64 first = values[0], last = values[1];
      D3D12_RANGE none = {0, 0};
      ID3D12Resource_Unmap(readback, 0, &none);
      printf("{\"queue_type\":%u,\"replay\":%llu,\"frequency\":%llu,"
             "\"before_submit\":%llu,\"first\":%llu,\"last\":%llu,\"gpu_delta_ms\":%.6f}\n",
             type, (unsigned long long)replay, (unsigned long long)frequency,
             (unsigned long long)before, (unsigned long long)first,
             (unsigned long long)last, 1000.0 * (last - first) / frequency);
      if (first < before || last < first || (previous && first <= previous) ||
          last - first >= frequency / 10) {
         fprintf(stderr, "timestamp execution/replay/resolve ordering failed\n");
         goto done;
      }
      previous = last;
   }
   rc = 0;
done:
   if (event) CloseHandle(event);
   if (fence) ID3D12Fence_Release(fence);
   if (list) ID3D12GraphicsCommandList_Release(list);
   if (alloc) ID3D12CommandAllocator_Release(alloc);
   if (heap) ID3D12QueryHeap_Release(heap);
   if (rtv) ID3D12DescriptorHeap_Release(rtv);
   if (target) ID3D12Resource_Release(target);
   if (readback) ID3D12Resource_Release(readback);
   if (queue) ID3D12CommandQueue_Release(queue);
   if (dev) ID3D12Device_Release(dev);
   return rc;
}
int main(void)
{
   if (run(D3D12_COMMAND_LIST_TYPE_DIRECT) || run(D3D12_COMMAND_LIST_TYPE_COMPUTE) ||
       run(D3D12_COMMAND_LIST_TYPE_COPY)) return 1;
   puts("GPU_EXECUTION_TIMESTAMPS_PASS");
   return 0;
}
