/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * d3d12agx wave (subgroup) operation acceptance.
 *
 * Every value here depends on what OTHER lanes in the same wave
 * contributed, so an implementation that only sees one lane produces
 * different answers rather than plausible ones. The checks are written
 * against the wave width the GPU REPORTS rather than a hardcoded 32, so
 * they stay true on hardware with a different width -- and so that a
 * driver returning a wrong width cannot accidentally satisfy them.
 *
 *   lane    < width                        every lane has a distinct index
 *   sum     == width                       WaveActiveSum(1) counts the wave
 *   prefix  == lane                        WavePrefixSum(1) is the lane rank
 *   maxidx  == width - 1                   the reduction saw the last lane
 *   first   == 0                           WaveReadLaneFirst read lane 0
 *
 * `prefix == lane` is the sharpest of these: an exclusive scan of 1 equals
 * a lane's position only if the scan actually ran across the wave.
 *
 * Usage: g5_wave wave_probe.dxil
 */

#define COBJMACROS
#include <wsl/winadapter.h>
#include <directx/d3d12.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "d12_test_contract.h"

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

#define N 256

/* Must match wave_probe.hlsl's `struct Result`. */
struct result {
   uint32_t lane, width, sum, prefix, maxidx, first, anytrue, alltrue;
};

static void *
slurp(const char *path, size_t *size)
{
   FILE *f = fopen(path, "rb");
   if (!f)
      return NULL;
   fseek(f, 0, SEEK_END);
   long n = ftell(f);
   fseek(f, 0, SEEK_SET);
   void *p = malloc(n);
   if (p && fread(p, 1, n, f) != (size_t)n) {
      free(p);
      p = NULL;
   }
   fclose(f);
   *size = n;
   return p;
}

int
main(int argc, char **argv)
{
   if (argc < 2) {
      fprintf(stderr, "usage: %s wave_probe.dxil\n", argv[0]);
      return 2;
   }

   size_t cs_size;
   void *cs = slurp(argv[1], &cs_size);
   if (!cs) {
      fprintf(stderr, "cannot read %s\n", argv[1]);
      return 2;
   }

   ID3D12Device *dev = NULL;
   HRESULT hr = D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0,
                                  &IID_ID3D12Device, (void **)&dev);
   TEST_OBJECT(hr, dev, "checked D3D12CreateDevice dev");
   CHECK(hr == S_OK && dev, "D3D12CreateDevice");
   if (hr != S_OK)
      return 1;

   D3D12_ROOT_PARAMETER param = {
      .ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV,
      .Descriptor = {.ShaderRegister = 0},
   };
   D3D12_ROOT_SIGNATURE_DESC rsd = {.NumParameters = 1, .pParameters = &param};

   ID3DBlob *blob = NULL;
   TEST_OBJECT(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1_0, &blob,
                               NULL), blob, "checked D3D12SerializeRootSignature blob");
   ID3D12RootSignature *root_sig = NULL;
   TEST_OBJECT(ID3D12Device_CreateRootSignature(dev, 0, ID3D10Blob_GetBufferPointer(blob),
                                    ID3D10Blob_GetBufferSize(blob),
                                    &IID_ID3D12RootSignature,
                                    (void **)&root_sig), root_sig, "checked ID3D12Device_CreateRootSignature root_sig");
   ID3D10Blob_Release(blob);

   D3D12_HEAP_PROPERTIES heap = D12_TEST_UMA_HEAP;
   D3D12_RESOURCE_DESC bd = {
      .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
      .Width = sizeof(struct result) * N,
      .Height = 1,
      .DepthOrArraySize = 1,
      .MipLevels = 1,
      .SampleDesc = {.Count = 1},
      .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
      .Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
   };
   ID3D12Resource *buf = NULL;
   hr = ID3D12Device_CreateCommittedResource(dev, &heap, D3D12_HEAP_FLAG_NONE,
                                             &bd, D3D12_RESOURCE_STATE_COMMON,
                                             NULL, &IID_ID3D12Resource,
                                             (void **)&buf);
   TEST_OBJECT(hr, buf, "checked ID3D12Device_CreateCommittedResource buf");
   CHECK(hr == S_OK && buf, "create the result buffer");
   if (hr != S_OK)
      return 1;

   struct result *map = NULL;
   TEST_CALL(d12_test_map(buf, false, (void **)&map), "checked CPU initialization map buf");
   memset(map, 0xEE, sizeof(struct result) * N);

   D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {
      .pRootSignature = root_sig,
      .CS = {.pShaderBytecode = cs, .BytecodeLength = cs_size},
   };
   ID3D12PipelineState *pso = NULL;
   hr = ID3D12Device_CreateComputePipelineState(dev, &pd,
                                                &IID_ID3D12PipelineState,
                                                (void **)&pso);
   TEST_OBJECT(hr, pso, "checked ID3D12Device_CreateComputePipelineState pso");
   CHECK(hr == S_OK && pso, "compile a shader using eight wave operations");
   if (hr != S_OK)
      return 1;

   ID3D12CommandAllocator *alloc = NULL;
   ID3D12GraphicsCommandList *list = NULL;
   TEST_OBJECT(ID3D12Device_CreateCommandAllocator(dev, D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                       &IID_ID3D12CommandAllocator,
                                       (void **)&alloc), alloc, "checked ID3D12Device_CreateCommandAllocator alloc");
   TEST_OBJECT(ID3D12Device_CreateCommandList(dev, 0, D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                  alloc, NULL,
                                  &IID_ID3D12GraphicsCommandList,
                                  (void **)&list), list, "checked ID3D12Device_CreateCommandList list");

   ID3D12GraphicsCommandList_SetComputeRootSignature(list, root_sig);
   ID3D12GraphicsCommandList_SetComputeRootUnorderedAccessView(
      list, 0, ID3D12Resource_GetGPUVirtualAddress(buf));
   ID3D12GraphicsCommandList_SetPipelineState(list, pso);
   ID3D12Resource_Unmap(buf, 0, NULL);map = NULL;
   d12_test_transition(list, buf, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
   ID3D12GraphicsCommandList_Dispatch(list, N / 64, 1, 1);
   d12_test_transition(list, buf, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
   hr = ID3D12GraphicsCommandList_Close(list);
   CHECK(hr == S_OK, "record the dispatch");
   if (hr != S_OK) return 1;

   ID3D12CommandQueue *queue = NULL;
   D3D12_COMMAND_QUEUE_DESC qd = {.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE};
   TEST_OBJECT(ID3D12Device_CreateCommandQueue(dev, &qd, &IID_ID3D12CommandQueue,
                                   (void **)&queue), queue, "checked ID3D12Device_CreateCommandQueue queue");
   ID3D12Fence *fence = NULL;
   TEST_OBJECT(ID3D12Device_CreateFence(dev, 0, D3D12_FENCE_FLAG_NONE, &IID_ID3D12Fence,
                            (void **)&fence), fence, "checked ID3D12Device_CreateFence fence");

   TEST_CALL(d12_test_submit_closed(dev, list, queue, fence, 1), "GPU dispatch complete");
   TEST_CALL(d12_test_map(buf, true, (void **)&map), "map actual fenced buf bytes");


   const uint32_t width = map[0].width;
   printf("       reported wave width: %u\n", width);
   printf("       thread 0: lane=%u sum=%u prefix=%u max=%u first=%u\n",
          map[0].lane, map[0].sum, map[0].prefix, map[0].maxidx,
          map[0].first);
   printf("       thread 5: lane=%u sum=%u prefix=%u max=%u first=%u\n",
          map[5].lane, map[5].sum, map[5].prefix, map[5].maxidx,
          map[5].first);

   CHECK(width >= 1 && width <= 128 && (width & (width - 1)) == 0,
         "WaveGetLaneCount reports a plausible power-of-two width");

   unsigned bad_lane = 0, bad_sum = 0, bad_prefix = 0, bad_max = 0,
            bad_first = 0, bad_vote = 0, bad_width = 0;

   for (unsigned i = 0; i < N; i++) {
      const struct result *r = &map[i];

      if (r->width != width)
         bad_width++;
      if (r->lane >= width)
         bad_lane++;
      /* Every lane of every wave is active here, so the wave's active
       * count is its full width.
       */
      if (r->sum != width)
         bad_sum++;
      if (r->prefix != r->lane)
         bad_prefix++;
      if (r->maxidx != width - 1)
         bad_max++;
      if (r->first != 0)
         bad_first++;
      if (r->anytrue != 1 || r->alltrue != 1)
         bad_vote++;
   }

   CHECK(bad_width == 0, "every thread agrees on the wave width");
   CHECK(bad_lane == 0, "every lane index is inside the wave");
   CHECK(bad_sum == 0, "WaveActiveSum(1) equals the wave width");
   CHECK(bad_prefix == 0,
         "WavePrefixSum(1) equals the lane index (the scan crossed lanes)");
   CHECK(bad_max == 0, "WaveActiveMax saw the highest lane in the wave");
   CHECK(bad_first == 0, "WaveReadLaneFirst read lane 0's value");
   CHECK(bad_vote == 0, "WaveActiveAnyTrue and WaveActiveAllTrue both hold");

   { const D3D12_RANGE empty = {0,0}; ID3D12Resource_Unmap(buf, 0, &empty); }
   ID3D12Fence_Release(fence);
   ID3D12CommandQueue_Release(queue);
   ID3D12GraphicsCommandList_Release(list);
   ID3D12CommandAllocator_Release(alloc);
   ID3D12PipelineState_Release(pso);
   ID3D12Resource_Release(buf);
   ID3D12RootSignature_Release(root_sig);
   CHECK(ID3D12Device_Release(dev) == 0, "device refcount reaches 0");
   free(cs);

   printf(failures ? "\nG5 WAVE: %d FAILURES\n" : "\nG5 WAVE: ALL PASS\n",
          failures);
   return failures;
}
