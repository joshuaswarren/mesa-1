/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * d3d12agx control-flow acceptance.
 *
 * DXIL is LLVM: an unstructured CFG. NIR wants structured control flow.
 * The translator emits goto/goto_if and lets Mesa's structurizer rebuild
 * if/loop shapes, with phis carried through registers that are promoted
 * back to SSA. All of that is invisible unless the answers are checked.
 *
 * cf_probe.hlsl, per thread:
 *
 *   count = n[tid]
 *   if (count == 0xFFFF) { o[tid] = 0xDEAD; return; }   // early out
 *   acc = 1
 *   for (i = 0; i < count; i++) acc = acc * 3 + i        // phi on a back edge
 *   tag = (acc & 1) ? 0x100 : 0x200                      // phi at a merge
 *   o[tid] = acc + tag
 *
 * The input array mixes trip counts of 0, 1, small, large and the
 * early-out sentinel, so a wrong loop bound, a phi that picks the wrong
 * incoming value, or an early-out that does not actually skip the tail all
 * produce different answers rather than one uniformly wrong one. Every
 * expected value is computed by running the same algorithm on the CPU.
 *
 * Usage: g5_control_flow cf_probe.dxil
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

/* The shader's algorithm, on the CPU. */
static uint32_t
expected(uint32_t count)
{
   if (count == 0xFFFF)
      return 0xDEAD;

   uint32_t acc = 1;
   for (uint32_t i = 0; i < count; i++)
      acc = acc * 3 + i;

   return acc + ((acc & 1) ? 0x100 : 0x200);
}

static ID3D12Resource *
make_buffer(ID3D12Device *dev, UINT64 size)
{
   D3D12_HEAP_PROPERTIES heap = D12_TEST_UMA_HEAP;
   D3D12_RESOURCE_DESC desc = {
      .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
      .Width = size,
      .Height = 1,
      .DepthOrArraySize = 1,
      .MipLevels = 1,
      .SampleDesc = {.Count = 1},
      .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
      .Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
   };

   ID3D12Resource *res = NULL;
   HRESULT hr = ID3D12Device_CreateCommittedResource(
          dev, &heap, D3D12_HEAP_FLAG_NONE, &desc,
          D3D12_RESOURCE_STATE_COMMON, NULL, &IID_ID3D12Resource,
          (void **)&res);
   if (hr != S_OK || !res) {
      if (res) ID3D12Resource_Release(res);
      return NULL;
   }
   return res;
}

int
main(int argc, char **argv)
{
   if (argc < 2) {
      fprintf(stderr, "usage: %s cf_probe.dxil\n", argv[0]);
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

   /* Two root descriptors: the SRV the shader reads and the UAV it writes. */
   D3D12_ROOT_PARAMETER params[2] = {
      {
         .ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV,
         .Descriptor = {.ShaderRegister = 0},
      },
      {
         .ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV,
         .Descriptor = {.ShaderRegister = 0},
      },
   };
   D3D12_ROOT_SIGNATURE_DESC rsd = {.NumParameters = 2, .pParameters = params};

   ID3DBlob *blob = NULL;
   TEST_OBJECT(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1_0, &blob,
                               NULL), blob, "checked D3D12SerializeRootSignature blob");
   ID3D12RootSignature *root_sig = NULL;
   TEST_OBJECT(ID3D12Device_CreateRootSignature(dev, 0, ID3D10Blob_GetBufferPointer(blob),
                                    ID3D10Blob_GetBufferSize(blob),
                                    &IID_ID3D12RootSignature,
                                    (void **)&root_sig), root_sig, "checked ID3D12Device_CreateRootSignature root_sig");
   ID3D10Blob_Release(blob);
   CHECK(root_sig != NULL, "root signature with a UAV at u0 and an SRV at t0");

   ID3D12Resource *in = make_buffer(dev, sizeof(uint32_t) * N);
   ID3D12Resource *out = make_buffer(dev, sizeof(uint32_t) * N);
   CHECK(in && out, "create input and output buffers");
   if (!in || !out)
      return 1;

   uint32_t *in_map = NULL, *out_map = NULL;
   TEST_CALL(d12_test_map(in, false, (void **)&in_map), "checked CPU initialization map in");
   TEST_CALL(d12_test_map(out, false, (void **)&out_map), "checked CPU initialization map out");

   /* Trip counts that exercise different paths: none, one, a handful, a
    * long run, and the early-out sentinel.
    */
   for (unsigned i = 0; i < N; i++) {
      if (i % 16 == 0)
         in_map[i] = 0xFFFF; /* early out */
      else if (i % 16 == 1)
         in_map[i] = 0; /* loop body never runs */
      else if (i % 16 == 2)
         in_map[i] = 1;
      else
         in_map[i] = i % 23;
   }
   memset(out_map, 0xCD, sizeof(uint32_t) * N);

   D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {
      .pRootSignature = root_sig,
      .CS = {.pShaderBytecode = cs, .BytecodeLength = cs_size},
   };
   ID3D12PipelineState *pso = NULL;
   hr = ID3D12Device_CreateComputePipelineState(dev, &pd,
                                                &IID_ID3D12PipelineState,
                                                (void **)&pso);
   TEST_OBJECT(hr, pso, "checked ID3D12Device_CreateComputePipelineState pso");
   CHECK(hr == S_OK && pso,
         "compile a shader with a loop, an early return and two phis");
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
      list, 0, ID3D12Resource_GetGPUVirtualAddress(out));
   ID3D12GraphicsCommandList_SetComputeRootShaderResourceView(
      list, 1, ID3D12Resource_GetGPUVirtualAddress(in));
   ID3D12GraphicsCommandList_SetPipelineState(list, pso);
   ID3D12Resource_Unmap(in, 0, NULL);in_map = NULL;
   d12_test_transition(list, in, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
   ID3D12Resource_Unmap(out, 0, NULL);out_map = NULL;
   d12_test_transition(list, out, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
   ID3D12GraphicsCommandList_Dispatch(list, N / 64, 1, 1);
   d12_test_transition(list, in, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
   d12_test_transition(list, out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
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
   TEST_CALL(d12_test_map(in, true, (void **)&in_map), "map actual fenced in bytes");
   TEST_CALL(d12_test_map(out, true, (void **)&out_map), "map actual fenced out bytes");


   unsigned wrong = 0, early = 0, zero_trip = 0, looped = 0;
   for (unsigned i = 0; i < N; i++) {
      uint32_t want = expected(in_map[i]);

      if (in_map[i] == 0xFFFF)
         early++;
      else if (in_map[i] == 0)
         zero_trip++;
      else
         looped++;

      if (out_map[i] != want) {
         if (wrong < 4) {
            printf("       [%u] count=%u got %u (0x%X), want %u (0x%X)\n", i,
                   in_map[i], out_map[i], out_map[i], want, want);
         }
         wrong++;
      }
   }

   printf("       %u early-out, %u zero-trip, %u looping threads\n", early,
          zero_trip, looped);
   printf("       spot check: [0]=0x%X [1]=%u [3]=%u\n", out_map[0],
          out_map[1], out_map[3]);

   CHECK(wrong == 0,
         "every thread agrees with the same algorithm run on the CPU");
   if (wrong)
      printf("       %u of %u threads wrong\n", wrong, N);

   /* Called out separately: each is a different way for control flow to be
    * subtly broken while most threads still look right.
    */
   CHECK(out_map[0] == 0xDEAD, "the early return took the early path");
   CHECK(out_map[1] == expected(0),
         "a zero-trip loop runs its body zero times");
   CHECK(out_map[2] == expected(1), "a one-trip loop runs its body once");

   { const D3D12_RANGE empty = {0,0}; ID3D12Resource_Unmap(in, 0, &empty); }
   { const D3D12_RANGE empty = {0,0}; ID3D12Resource_Unmap(out, 0, &empty); }
   ID3D12Fence_Release(fence);
   ID3D12CommandQueue_Release(queue);
   ID3D12GraphicsCommandList_Release(list);
   ID3D12CommandAllocator_Release(alloc);
   ID3D12PipelineState_Release(pso);
   ID3D12Resource_Release(out);
   ID3D12Resource_Release(in);
   ID3D12RootSignature_Release(root_sig);
   CHECK(ID3D12Device_Release(dev) == 0, "device refcount reaches 0");
   free(cs);

   printf(failures ? "\nG5 CONTROL FLOW: %d FAILURES\n"
                   : "\nG5 CONTROL FLOW: ALL PASS\n",
          failures);
   return failures;
}
