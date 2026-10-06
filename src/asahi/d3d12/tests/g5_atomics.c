/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * d3d12agx atomics acceptance.
 *
 * Compiling an atomic proves nothing about whether it is atomic. Every
 * case here has 4096 threads contending for ONE address, so a
 * read-modify-write that is not atomic loses updates and lands short --
 * and lands short by a different amount every run, which is the signature
 * of the bug rather than a wrong constant.
 *
 *   o[0]  InterlockedAdd(1)          -> exactly N
 *   o[1]  InterlockedMax(tid)        -> exactly N-1
 *   o[2]  InterlockedAdd(tid)        -> exactly N(N-1)/2
 *   o[3]  InterlockedCompareExchange -> 0xABC, claimed once
 *   o[4]  count of threads that won  -> exactly 1
 *
 * o[4] is the sharpest: compare-exchange must let exactly one thread
 * observe the original value. Any number other than 1 means the
 * comparison and the store were not a single operation.
 *
 * Usage: g5_atomics atomic_probe.dxil
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

/* 64 groups of 64 threads: enough contention that a lost update is
 * overwhelmingly likely if the operations are not atomic.
 */
#define GROUPS 64
#define N (GROUPS * 64)

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
      fprintf(stderr, "usage: %s atomic_probe.dxil\n", argv[0]);
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
   CHECK(root_sig != NULL, "root signature with one UAV at u0");

   D3D12_HEAP_PROPERTIES heap = D12_TEST_UMA_HEAP;
   D3D12_RESOURCE_DESC bd = {
      .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
      .Width = 256,
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
   CHECK(hr == S_OK && buf, "create the contended buffer");
   if (hr != S_OK)
      return 1;

   uint32_t *map = NULL;
   TEST_CALL(d12_test_map(buf, false, (void **)&map), "checked CPU initialization map buf");
   memset(map, 0, 256);

   D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {
      .pRootSignature = root_sig,
      .CS = {.pShaderBytecode = cs, .BytecodeLength = cs_size},
   };
   ID3D12PipelineState *pso = NULL;
   hr = ID3D12Device_CreateComputePipelineState(dev, &pd,
                                                &IID_ID3D12PipelineState,
                                                (void **)&pso);
   TEST_OBJECT(hr, pso, "checked ID3D12Device_CreateComputePipelineState pso");
   CHECK(hr == S_OK && pso, "compile a shader using four atomic operations");
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
   ID3D12GraphicsCommandList_Dispatch(list, GROUPS, 1, 1);
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


   const uint32_t want_sum = (uint32_t)((uint64_t)N * (N - 1) / 2);

   printf("       %u threads: add=%u max=%u sum=%u cmpxchg=0x%X winners=%u\n",
          N, map[0], map[1], map[2], map[3], map[4]);

   CHECK(map[0] == N, "InterlockedAdd(1) counts every thread exactly once");
   if (map[0] != N)
      printf("       lost %d increments\n", (int)N - (int)map[0]);

   CHECK(map[1] == N - 1, "InterlockedMax reaches the highest thread index");
   CHECK(map[2] == want_sum, "InterlockedAdd(tid) sums to N(N-1)/2");
   if (map[2] != want_sum)
      printf("       got %u, want %u\n", map[2], want_sum);

   CHECK(map[3] == 0xABC, "InterlockedCompareExchange stored its value");
   CHECK(map[4] == 1,
         "exactly one thread won the compare-exchange (not 0, not many)");

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

   printf(failures ? "\nG5 ATOMICS: %d FAILURES\n" : "\nG5 ATOMICS: ALL PASS\n",
          failures);
   return failures;
}
