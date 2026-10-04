/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * d3d12agx G4 acceptance test: the whole campaign in one program.
 *
 * An application hands the driver DXIL bytecode through the public D3D12
 * API. The driver parses it, translates it to NIR, compiles it to AGX
 * machine code, dispatches it on the Apple GPU, and the results are
 * checked on the CPU. No SPIR-V, no Vulkan, no external compiler.
 *
 * The shader is compute (HLSL written for this project, compiled by dxc):
 *
 *   RWStructuredBuffer<uint> outbuf : register(u0);
 *   [numthreads(64,1,1)]
 *   void main(uint3 tid : SV_DispatchThreadID) {
 *       outbuf[tid.x] = tid.x * 3 + 7;
 *   }
 *
 * so element i must equal i*3+7 — a value the CPU never wrote.
 */

#define COBJMACROS
#include <wsl/winadapter.h>
#include <directx/d3d12.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef _WIN32
#include <errno.h>
#include <time.h>
#endif

static uint64_t
test_monotonic_ms(void)
{
#ifdef _WIN32
   return GetTickCount64();
#else
   struct timespec now;
   if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
      perror("clock_gettime");
      exit(1);
   }
   return (uint64_t)now.tv_sec * 1000 + now.tv_nsec / 1000000;
#endif
}

static void
test_sleep_ms(unsigned milliseconds)
{
#ifdef _WIN32
   Sleep(milliseconds);
#else
   struct timespec delay = {
      .tv_sec = milliseconds / 1000,
      .tv_nsec = (long)(milliseconds % 1000) * 1000000,
   };
   while (nanosleep(&delay, &delay) != 0) {
      if (errno != EINTR) {
         perror("nanosleep");
         exit(1);
      }
   }
#endif
}

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

#define THREADS_PER_GROUP 64
#define NUM_GROUPS        4
#define NUM_ELEMENTS      (THREADS_PER_GROUP * NUM_GROUPS)

int
main(int argc, char **argv)
{
   const char *dxil_path = argc > 1 ? argv[1] : NULL;
   if (!dxil_path) {
      fprintf(stderr, "usage: %s cs.dxil\n", argv[0]);
      return 2;
   }

   FILE *f = fopen(dxil_path, "rb");
   if (!f) {
      fprintf(stderr, "cannot open %s\n", dxil_path);
      return 2;
   }
   fseek(f, 0, SEEK_END);
   long dxil_size = ftell(f);
   fseek(f, 0, SEEK_SET);
   void *dxil = malloc(dxil_size);
   if (!dxil || fread(dxil, 1, dxil_size, f) != (size_t)dxil_size) {
      fprintf(stderr, "cannot read %s\n", dxil_path);
      return 2;
   }
   fclose(f);

   ID3D12Device *dev = NULL;
   ID3D12RootSignature *root_sig = NULL;
   ID3D12PipelineState *pso = NULL;
   ID3D12Resource *buf = NULL;
   uint32_t *map = NULL;
   ID3D12CommandAllocator *alloc = NULL;
   ID3D12GraphicsCommandList *list = NULL;
   ID3D12CommandQueue *queue = NULL;
   ID3D12Fence *fence = NULL;
   const char *blocked_stage = NULL;
   HRESULT blocked_hr = S_OK;
   HRESULT hr = D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0,
                                  &IID_ID3D12Device, (void **)&dev);
   CHECK(hr == S_OK && dev, "D3D12CreateDevice");
   if (FAILED(hr) || !dev)
      return 1;

   /* A real root signature: one root UAV at u0, which is what cs.hlsl
    * declares. The blob is now parsed, so this has to describe the shader
    * -- previously four bytes of the shader itself were passed here and
    * accepted, which described nothing.
    */
   D3D12_ROOT_PARAMETER param = {
      .ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV,
      .Descriptor = {.ShaderRegister = 0, .RegisterSpace = 0},
      .ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL,
   };
   D3D12_ROOT_SIGNATURE_DESC rsd = {.NumParameters = 1, .pParameters = &param};

   ID3DBlob *rs_blob = NULL;
   hr = D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1_0,
                                    &rs_blob, NULL);
   CHECK(hr == S_OK && rs_blob, "D3D12SerializeRootSignature");
   if (FAILED(hr))
      return 1;

   hr = ID3D12Device_CreateRootSignature(
      dev, 0, ID3D10Blob_GetBufferPointer(rs_blob),
      ID3D10Blob_GetBufferSize(rs_blob), &IID_ID3D12RootSignature,
      (void **)&root_sig);
   CHECK(hr == S_OK && root_sig, "CreateRootSignature parses the blob");
   ID3D10Blob_Release(rs_blob);

   /* THE MILESTONE: the driver compiles application DXIL to AGX. */
   D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc = {
      .pRootSignature = root_sig,
      .CS = {.pShaderBytecode = dxil, .BytecodeLength = dxil_size},
   };
   hr = ID3D12Device_CreateComputePipelineState(dev, &pso_desc,
                                                &IID_ID3D12PipelineState,
                                                (void **)&pso);
   CHECK(hr == S_OK && pso,
         "CreateComputePipelineState compiles DXIL to AGX machine code");
   if (FAILED(hr) || !pso)
      return 1;

   /* Output buffer, poisoned so stale zeroes cannot fake a pass. */
   /* UPLOAD is staging and cannot carry ALLOW_UNORDERED_ACCESS. The
    * advertised GPU_UPLOAD heap supports both a real UAV and CPU poison/
    * readback mapping on this UMA device, without relaxing validation. */
   D3D12_HEAP_PROPERTIES heap = {.Type = D3D12_HEAP_TYPE_GPU_UPLOAD};
   D3D12_RESOURCE_DESC bd = {
      .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
      .Width = NUM_ELEMENTS * 4,
      .Height = 1,
      .DepthOrArraySize = 1,
      .MipLevels = 1,
      .SampleDesc = {.Count = 1},
      .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
      .Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
   };
   hr = ID3D12Device_CreateCommittedResource(
      dev, &heap, D3D12_HEAP_FLAG_NONE, &bd,
      D3D12_RESOURCE_STATE_UNORDERED_ACCESS, NULL, &IID_ID3D12Resource,
      (void **)&buf);
   CHECK(hr == S_OK && buf, "create UAV buffer");
   if (FAILED(hr) || !buf) {
      blocked_stage = "CreateCommittedResource UAV";
      blocked_hr = FAILED(hr) ? hr : E_FAIL;
      goto cleanup;
   }

   hr = ID3D12Resource_Map(buf, 0, NULL, (void **)&map);
   CHECK(hr == S_OK && map, "map UAV buffer");
   if (FAILED(hr) || !map) {
      blocked_stage = "Map UAV";
      blocked_hr = FAILED(hr) ? hr : E_FAIL;
      goto cleanup;
   }
   for (unsigned i = 0; i < NUM_ELEMENTS; i++)
      map[i] = 0xDEADBEEF;

   /* Record: bind the pipeline and the UAV, dispatch. */
   hr = ID3D12Device_CreateCommandAllocator(
      dev, D3D12_COMMAND_LIST_TYPE_COMPUTE, &IID_ID3D12CommandAllocator,
      (void **)&alloc);
   CHECK(hr == S_OK && alloc, "CreateCommandAllocator(COMPUTE)");
   if (FAILED(hr) || !alloc) {
      blocked_stage = "CreateCommandAllocator";
      blocked_hr = FAILED(hr) ? hr : E_FAIL;
      goto cleanup;
   }
   hr = ID3D12Device_CreateCommandList(dev, 0, D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                       alloc, NULL,
                                       &IID_ID3D12GraphicsCommandList,
                                       (void **)&list);
   CHECK(hr == S_OK && list, "CreateCommandList(COMPUTE)");
   if (FAILED(hr) || !list) {
      blocked_stage = "CreateCommandList";
      blocked_hr = FAILED(hr) ? hr : E_FAIL;
      goto cleanup;
   }

   ID3D12GraphicsCommandList_SetComputeRootSignature(list, root_sig);
   ID3D12GraphicsCommandList_SetPipelineState(list, pso);
   ID3D12GraphicsCommandList_SetComputeRootUnorderedAccessView(
      list, 0, ID3D12Resource_GetGPUVirtualAddress(buf));
   ID3D12GraphicsCommandList_Dispatch(list, NUM_GROUPS, 1, 1);


   hr = ID3D12GraphicsCommandList_Close(list);
   CHECK(hr == S_OK, "Close command list");
   if (FAILED(hr)) {
      blocked_stage = "Close";
      blocked_hr = hr;
      goto cleanup;
   }

   D3D12_COMMAND_QUEUE_DESC qdesc = {.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE};
   hr = ID3D12Device_CreateCommandQueue(dev, &qdesc, &IID_ID3D12CommandQueue,
                                        (void **)&queue);
   CHECK(hr == S_OK && queue, "CreateCommandQueue");
   if (FAILED(hr) || !queue) {
      blocked_stage = "CreateCommandQueue";
      blocked_hr = FAILED(hr) ? hr : E_FAIL;
      goto cleanup;
   }
   hr = ID3D12Device_CreateFence(dev, 0, D3D12_FENCE_FLAG_NONE, &IID_ID3D12Fence,
                                 (void **)&fence);
   CHECK(hr == S_OK && fence, "CreateFence");
   if (FAILED(hr) || !fence) {
      blocked_stage = "CreateFence";
      blocked_hr = FAILED(hr) ? hr : E_FAIL;
      goto cleanup;
   }

   CHECK(map[0] == 0xDEADBEEF, "buffer still poisoned before execution");

   ID3D12CommandList *lists[] = {(ID3D12CommandList *)list};
   ID3D12CommandQueue_ExecuteCommandLists(queue, 1, lists);
   hr = ID3D12CommandQueue_Signal(queue, fence, 1);
   CHECK(hr == S_OK, "Signal after execution");
   if (FAILED(hr)) {
      blocked_stage = "Signal";
      blocked_hr = hr;
      goto cleanup;
   }

   const uint64_t wait_start_ms = test_monotonic_ms();
   hr = ID3D12Fence_SetEventOnCompletion(fence, 1, NULL);
   printf("       fence wait blocked for %llu ms\n",
          (unsigned long long)(test_monotonic_ms() - wait_start_ms));
   CHECK(hr == S_OK && ID3D12Fence_GetCompletedValue(fence) >= 1,
         "GPU dispatch complete (fence reached 1)");
   if (FAILED(hr) || ID3D12Fence_GetCompletedValue(fence) < 1) {
      blocked_stage = "Wait fence";
      blocked_hr = FAILED(hr) ? hr : E_FAIL;
      goto cleanup;
   }

   /* Diagnostic: is a miss a fence-ordering bug or a visibility delay?
    * If the data appears on a later read, the wait returned too early.
    */
   if (map[0] != 7) {
      for (int spin = 0; spin < 100 && map[0] != 7; spin++) {
         test_sleep_ms(1);
      }
      printf("       NOTE: value after re-read spin: %u (fence returned "
             "before the write was visible)\n",
             map[0]);
   }

   /* The verdict: every element computed by our compiled shader. */
   int bad = -1;
   for (unsigned i = 0; i < NUM_ELEMENTS; i++) {
      if (map[i] != i * 3 + 7) {
         bad = (int)i;
         break;
      }
   }
   CHECK(bad < 0, "all 256 elements equal tid*3+7, computed on the GPU");
   if (bad >= 0) {
      printf("       element %d: got 0x%08x want 0x%08x\n", bad, map[bad],
             bad * 3 + 7);
   } else {
      printf("       spot check: [0]=%u [1]=%u [64]=%u [255]=%u\n", map[0],
             map[1], map[64], map[255]);
   }

cleanup:
   if (blocked_stage)
      printf("G4 SMOKE BLOCKED: stage=%s HRESULT=0x%08x (not a hardware pass)\n",
             blocked_stage, (unsigned)blocked_hr);
   if (map && buf)
      ID3D12Resource_Unmap(buf, 0, NULL);
   if (fence)
      ID3D12Fence_Release(fence);
   if (queue)
      ID3D12CommandQueue_Release(queue);
   if (list)
      ID3D12GraphicsCommandList_Release(list);
   if (alloc)
      ID3D12CommandAllocator_Release(alloc);
   if (buf)
      ID3D12Resource_Release(buf);
   if (pso)
      ID3D12PipelineState_Release(pso);
   if (root_sig)
      ID3D12RootSignature_Release(root_sig);
   CHECK(ID3D12Device_Release(dev) == 0, "device refcount reaches 0");

   printf(failures ? "G4 SMOKE: %d FAILURES\n" : "G4 SMOKE: ALL PASS\n",
          failures);
   free(dxil);
   return failures;
}
