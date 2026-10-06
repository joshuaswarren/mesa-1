/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * d3d12agx G1 acceptance test. Deliberately uses ONLY the public D3D12 ABI
 * (DirectX-Headers), the way a real client would: create a device on the
 * real GPU, query architecture facts, exercise queue+fence semantics.
 *
 * Every check prints ok/FAIL; exit code is the number of failures.
 */

#define COBJMACROS
#include <wsl/winadapter.h>
#include <directx/d3d12.h>

#include "d12_test_abi.h"

#include <inttypes.h>
#include <stdio.h>

static int failures = 0;

#define CHECK(cond, what)                                                    \
   do {                                                                      \
      if (cond) {                                                            \
         printf("ok   - %s\n", what);                                        \
      } else {                                                               \
         printf("FAIL - %s\n", what);                                        \
         failures++;                                                         \
      }                                                                      \
   } while (0)

int
main(void)
{
   ID3D12Device *dev = NULL;
   HRESULT hr;

   /* Capability probe form (NULL out pointer) */
   hr = D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0, &IID_ID3D12Device,
                          NULL);
   CHECK(hr == S_FALSE, "capability probe returns S_FALSE");

   hr = D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0, &IID_ID3D12Device,
                          (void **)&dev);
   CHECK(hr == S_OK && dev, "D3D12CreateDevice on the asahi render node");
   if (FAILED(hr) || !dev)
      return failures;

   CHECK(ID3D12Device_GetNodeCount(dev) == 1, "GetNodeCount == 1");

   D3D12_FEATURE_DATA_ARCHITECTURE1 arch = {.NodeIndex = 0};
   hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_ARCHITECTURE1,
                                         &arch, sizeof(arch));
   CHECK(hr == S_OK, "CheckFeatureSupport(ARCHITECTURE1)");
   printf("       UMA=%d CacheCoherentUMA=%d TileBasedRenderer=%d "
          "IsolatedMMU=%d\n",
          arch.UMA, arch.CacheCoherentUMA, arch.TileBasedRenderer,
          arch.IsolatedMMU);
   CHECK(!arch.CacheCoherentUMA || arch.UMA,
         "cache-coherent UMA implies UMA");

   D3D_FEATURE_LEVEL levels_wanted[] = {
      D3D_FEATURE_LEVEL_11_0,
      D3D_FEATURE_LEVEL_12_0,
   };
   D3D12_FEATURE_DATA_FEATURE_LEVELS levels = {
      .NumFeatureLevels = 2,
      .pFeatureLevelsRequested = levels_wanted,
   };
   hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_FEATURE_LEVELS,
                                         &levels, sizeof(levels));
   CHECK(hr == S_OK &&
            (levels.MaxSupportedFeatureLevel == D3D_FEATURE_LEVEL_11_0 ||
             levels.MaxSupportedFeatureLevel == D3D_FEATURE_LEVEL_12_0),
         "max supported feature level is one of the requested levels");
   if (hr == S_OK) {
      hr = D3D12CreateDevice(NULL, levels.MaxSupportedFeatureLevel,
                            &IID_ID3D12Device, NULL);
      CHECK(hr == S_FALSE, "reported feature level passes device probe");
   }

   /* Queue + fence */
   D3D12_COMMAND_QUEUE_DESC qdesc = {
      .Type = D3D12_COMMAND_LIST_TYPE_DIRECT,
   };
   ID3D12CommandQueue *queue = NULL;
   hr = ID3D12Device_CreateCommandQueue(dev, &qdesc, &IID_ID3D12CommandQueue,
                                        (void **)&queue);
   printf("       CreateCommandQueue HRESULT=0x%08" PRIx32 " object=%s\n",
          (uint32_t)hr, queue ? "yes" : "no");
   CHECK(hr == S_OK && queue, "CreateCommandQueue(DIRECT)");

   ID3D12Fence *fence = NULL;
   hr = ID3D12Device_CreateFence(dev, 5, D3D12_FENCE_FLAG_NONE,
                                 &IID_ID3D12Fence, (void **)&fence);
   CHECK(hr == S_OK && fence, "CreateFence(initial=5)");

   if (queue && fence) {
      CHECK(ID3D12Fence_GetCompletedValue(fence) == 5,
            "GetCompletedValue == initial 5");

      hr = ID3D12Fence_Signal(fence, 7);
      CHECK(hr == S_OK && ID3D12Fence_GetCompletedValue(fence) == 7,
            "CPU Signal(7) completes");

      hr = ID3D12Fence_SetEventOnCompletion(fence, 7, NULL);
      CHECK(hr == S_OK, "NULL-event wait on already-reached value");

      /* Queue-side signal: queue is empty, so the point must appear
       * promptly once the transfer resolves.
       */
      hr = ID3D12CommandQueue_Signal(queue, fence, 9);
      CHECK(hr == S_OK, "queue Signal(fence, 9) accepted");

      hr = ID3D12Fence_SetEventOnCompletion(fence, 9, NULL);
      CHECK(hr == S_OK && ID3D12Fence_GetCompletedValue(fence) >= 9,
            "queue Signal(9) observed on CPU");

      /* Queue-side wait on a materialized point, then a dependent signal */
      hr = ID3D12CommandQueue_Wait(queue, fence, 9);
      CHECK(hr == S_OK, "queue Wait(fence, 9) accepted");

      hr = ID3D12CommandQueue_Signal(queue, fence, 11);
      CHECK(hr == S_OK, "queue Signal(fence, 11) after Wait accepted");

      hr = ID3D12Fence_SetEventOnCompletion(fence, 11, NULL);
      CHECK(hr == S_OK && ID3D12Fence_GetCompletedValue(fence) >= 11,
            "dependent Signal(11) observed on CPU");

      UINT64 freq = 0;
      hr = ID3D12CommandQueue_GetTimestampFrequency(queue, &freq);
      CHECK(hr == S_OK && freq > 0, "GetTimestampFrequency > 0");
      printf("       timestamp frequency: %" PRIu64 " Hz\n", freq);

      D3D12_COMMAND_QUEUE_DESC rt_desc =
         d12_test_command_queue_get_desc(queue);
      CHECK(rt_desc.Type == D3D12_COMMAND_LIST_TYPE_DIRECT,
            "GetDesc round-trips queue type");

      /* GetDevice ref plumbing */
      ID3D12Device *dev2 = NULL;
      hr = ID3D12Fence_GetDevice(fence, &IID_ID3D12Device, (void **)&dev2);
      CHECK(hr == S_OK && dev2 == dev, "fence GetDevice returns the device");
      if (dev2)
         ID3D12Device_Release(dev2);
   }

   /* Honesty checks: these all used to return success while doing
    * nothing, which is worse than failing because nothing surfaces it.
    */
   {
      ID3D12Device *bad = NULL;
      hr = D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_9_1, &IID_ID3D12Device,
                             (void **)&bad);
      CHECK(hr == E_INVALIDARG && !bad,
            "feature level 9_1 is rejected, not silently accepted");

      const D3D_SHADER_MODEL models[] = {
         D3D_SHADER_MODEL_6_7, D3D_SHADER_MODEL_6_6, D3D_SHADER_MODEL_6_5,
         D3D_SHADER_MODEL_6_4, D3D_SHADER_MODEL_6_3, D3D_SHADER_MODEL_6_2,
         D3D_SHADER_MODEL_6_1, D3D_SHADER_MODEL_6_0, D3D_SHADER_MODEL_5_1,
      };
      D3D12_FEATURE_DATA_SHADER_MODEL sm = {0};
      unsigned requested = 0;
      for (; requested < sizeof(models) / sizeof(models[0]); requested++) {
         sm.HighestShaderModel = models[requested];
         hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_SHADER_MODEL,
                                               &sm, sizeof(sm));
         if (hr != E_INVALIDARG)
            break;
      }
      CHECK(hr == S_OK && sm.HighestShaderModel >= D3D_SHADER_MODEL_5_1 &&
               requested < sizeof(models) / sizeof(models[0]) &&
               sm.HighestShaderModel <= models[requested],
            "SHADER_MODEL returns support bounded by the request");
      printf("       HighestShaderModel=0x%x\n", sm.HighestShaderModel);
      if (hr == S_OK) {
         D3D_SHADER_MODEL supported = sm.HighestShaderModel;
         hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_SHADER_MODEL,
                                               &sm, sizeof(sm));
         CHECK(hr == S_OK && sm.HighestShaderModel == supported,
               "reported shader model round-trips");
      }

      D3D12_FEATURE_DATA_D3D12_OPTIONS opts = {0};
      hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_D3D12_OPTIONS,
                                            &opts, sizeof(opts));
      CHECK(hr == S_OK &&
               (opts.ResourceHeapTier == D3D12_RESOURCE_HEAP_TIER_1 ||
                opts.ResourceHeapTier == D3D12_RESOURCE_HEAP_TIER_2),
            "reports a defined resource heap tier");
      printf("       ResourceHeapTier=%u\n", opts.ResourceHeapTier);

      D3D12_FEATURE_DATA_D3D12_OPTIONS3 opts3 = {0};
      hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_D3D12_OPTIONS3,
                                            &opts3, sizeof(opts3));
      CHECK(hr == S_OK, "OPTIONS3 capability query succeeds");

      D3D12_FEATURE_DATA_PREDICATION predication = {0};
      hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_PREDICATION,
                                            &predication,
                                            sizeof(predication));
      CHECK(hr == S_OK || hr == E_INVALIDARG || hr == E_NOTIMPL,
            "optional PREDICATION query succeeds or is unsupported");

      D3D12_FEATURE_DATA_HARDWARE_COPY hardware_copy = {.Supported = TRUE};
      hr = ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_HARDWARE_COPY,
                                            &hardware_copy,
                                            sizeof(hardware_copy));
      CHECK(hr == S_OK || hr == E_INVALIDARG || hr == E_NOTIMPL,
            "optional HARDWARE_COPY query succeeds or is unsupported");

      LUID luid = d12_test_device_get_adapter_luid(dev);
      LUID luid_again = d12_test_device_get_adapter_luid(dev);
      CHECK(luid.LowPart == luid_again.LowPart && luid.HighPart == luid_again.HighPart,
            "adapter LUID is stable (native Linux may have no DXGI LUID)");
      printf("       adapter LUID=%08x:%08x\n", (unsigned)luid.HighPart, luid.LowPart);

      CHECK(ID3D12Device_GetDeviceRemovedReason(dev) == S_OK,
            "device not removed while healthy");

      /* Reset must fail while a list built from the allocator records. */
      ID3D12CommandAllocator *ca = NULL;
      ID3D12GraphicsCommandList *cl = NULL;
      ID3D12Device_CreateCommandAllocator(dev,
                                          D3D12_COMMAND_LIST_TYPE_DIRECT,
                                          &IID_ID3D12CommandAllocator,
                                          (void **)&ca);
      ID3D12Device_CreateCommandList(dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                     ca, NULL,
                                     &IID_ID3D12GraphicsCommandList,
                                     (void **)&cl);
      CHECK(ID3D12CommandAllocator_Reset(ca) == E_FAIL,
            "allocator Reset fails while a list is recording");
      ID3D12GraphicsCommandList_Close(cl);
      CHECK(ID3D12CommandAllocator_Reset(ca) == S_OK,
            "allocator Reset succeeds once the list is closed");
      ID3D12GraphicsCommandList_Release(cl);
      ID3D12CommandAllocator_Release(ca);
   }

   /* Exercise implemented allocation and heap APIs with valid descriptors. */
   {
      D3D12_RESOURCE_DESC rd = {
         .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
         .Width = 4096, .Height = 1, .DepthOrArraySize = 1, .MipLevels = 1,
         .SampleDesc = {.Count = 1}, .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
      };
      D3D12_RESOURCE_ALLOCATION_INFO ai =
         d12_test_device_get_resource_allocation_info(dev, 0, 1, &rd);
      CHECK(ai.SizeInBytes != UINT64_MAX && ai.SizeInBytes >= rd.Width &&
               ai.Alignment == D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT &&
               ai.SizeInBytes % ai.Alignment == 0,
            "valid buffer allocation has sufficient size and default alignment");
      if (ai.SizeInBytes != UINT64_MAX && ai.Alignment) {
         ID3D12Heap *heap = NULL;
         D3D12_HEAP_DESC hd = {
            .SizeInBytes = ai.SizeInBytes, .Alignment = ai.Alignment,
            .Properties = {.Type = D3D12_HEAP_TYPE_DEFAULT,
                           .CreationNodeMask = 1, .VisibleNodeMask = 1},
            .Flags = D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS,
         };
         hr = ID3D12Device_CreateHeap(dev, &hd, &IID_ID3D12Heap, (void **)&heap);
         CHECK(hr == S_OK && heap, "CreateHeap succeeds for a valid buffer heap");
         if (heap) {
            D3D12_HEAP_DESC actual = d12_test_heap_get_desc(heap);
            CHECK(actual.SizeInBytes == hd.SizeInBytes &&
                     actual.Alignment == hd.Alignment &&
                     actual.Properties.Type == hd.Properties.Type &&
                     actual.Flags == hd.Flags,
                  "GetDesc round-trips the heap descriptor");
            ID3D12Heap_Release(heap);
         }
      }
   }

   if (fence)
      CHECK(ID3D12Fence_Release(fence) == 0, "fence refcount reaches 0");
   if (queue)
      CHECK(ID3D12CommandQueue_Release(queue) == 0,
            "queue refcount reaches 0");
   CHECK(ID3D12Device_Release(dev) == 0, "device refcount reaches 0");

   printf(failures ? "G1 SMOKE: %d FAILURES\n" : "G1 SMOKE: ALL PASS\n",
          failures);
   return failures;
}
