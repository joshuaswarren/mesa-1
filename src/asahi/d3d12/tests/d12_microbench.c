/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * d3d12agx CPU-side microbenchmark.
 *
 * Measures the driver's host cost for the operations a game issues at high
 * frequency: command-list recording per draw and per dispatch, root
 * parameter updates, descriptor writes and copies, PSO creation, allocator
 * and list reset, and the latency of a submit-signal-wait round trip.
 *
 * Everything is timed with CLOCK_MONOTONIC around public D3D12 API calls, so
 * the numbers are what an application sees. GPU time is reported separately
 * (wall time from ExecuteCommandLists to fence completion) and never mixed
 * into a CPU figure.
 *
 * Usage: d3d12agx_microbench tri.dxil solid.dxil res_probe.dxil [--json FILE]
 *
 * The last stdout line is "MICROBENCH_JSON {...}" with every metric, so a
 * host-side collector can file it without parsing prose.
 */

#define COBJMACROS
#include <wsl/winadapter.h>
#include <directx/d3d12.h>

#include "d12_test_abi.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define W 256
#define H 256

#define DRAWS_PER_LIST 20000
#define DISPATCHES_PER_LIST 20000
#define RECORD_REPEATS 5
#define DESCRIPTOR_COUNT 4096
#define PSO_REPEATS 8
#define ROUNDTRIP_REPEATS 100

static unsigned failures;

#define CHECK(cond, what)                                                    \
   do {                                                                      \
      if (!(cond)) {                                                         \
         printf("FAIL - %s\n", what);                                        \
         failures++;                                                         \
      }                                                                      \
   } while (0)

#define REQUIRE(cond, what)                                                  \
   do {                                                                      \
      if (!(cond)) {                                                         \
         printf("FAIL - %s\n", what);                                        \
         printf("MICROBENCH_JSON {\"error\": \"%s\"}\n", what);              \
         return 1;                                                           \
      }                                                                      \
   } while (0)

static double
now_ns(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static int
cmp_double(const void *a, const void *b)
{
   double x = *(const double *)a, y = *(const double *)b;
   return x < y ? -1 : x > y;
}

static double
percentile(double *v, unsigned n, double p)
{
   qsort(v, n, sizeof(*v), cmp_double);
   unsigned i = (unsigned)(p * (n - 1) + 0.5);
   return v[i < n ? i : n - 1];
}

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

static ID3D12RootSignature *
make_root_sig(ID3D12Device *dev, const D3D12_ROOT_PARAMETER *params,
              UINT count)
{
   D3D12_ROOT_SIGNATURE_DESC rsd = {.NumParameters = count,
                                    .pParameters = params};
   ID3DBlob *blob = NULL;
   if (D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1_0,
                                   &blob, NULL) != S_OK)
      return NULL;
   ID3D12RootSignature *rs = NULL;
   HRESULT hr = ID3D12Device_CreateRootSignature(
      dev, 0, ID3D10Blob_GetBufferPointer(blob),
      ID3D10Blob_GetBufferSize(blob), &IID_ID3D12RootSignature, (void **)&rs);
   ID3D10Blob_Release(blob);
   return hr == S_OK ? rs : NULL;
}

static ID3D12Resource *
make_buffer(ID3D12Device *dev, UINT64 size, D3D12_RESOURCE_FLAGS flags)
{
   const bool uav = flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
   D3D12_HEAP_PROPERTIES heap = {
      .Type = uav ? D3D12_HEAP_TYPE_DEFAULT : D3D12_HEAP_TYPE_UPLOAD};
   D3D12_RESOURCE_DESC desc = {
      .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
      .Width = size,
      .Height = 1,
      .DepthOrArraySize = 1,
      .MipLevels = 1,
      .SampleDesc = {.Count = 1},
      .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
      .Flags = flags,
   };
   ID3D12Resource *res = NULL;
   HRESULT hr = ID3D12Device_CreateCommittedResource(
      dev, &heap, D3D12_HEAP_FLAG_NONE, &desc,
      uav ? D3D12_RESOURCE_STATE_COMMON : D3D12_RESOURCE_STATE_GENERIC_READ,
      NULL, &IID_ID3D12Resource, (void **)&res);
   return hr == S_OK ? res : NULL;
}

static void
transition(ID3D12GraphicsCommandList *list, ID3D12Resource *resource,
           D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after)
{
   D3D12_RESOURCE_BARRIER barrier = {
      .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
      .Transition = {.pResource = resource,
                     .Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                     .StateBefore = before,
                     .StateAfter = after},
   };
   ID3D12GraphicsCommandList_ResourceBarrier(list, 1, &barrier);
}

/* Shared state for the recording benchmarks. */
struct bench {
   ID3D12Device *dev;
   ID3D12CommandQueue *queue;
   ID3D12Fence *fence;
   UINT64 fence_value;
   ID3D12CommandAllocator *alloc;
   ID3D12GraphicsCommandList *list;
   ID3D12CommandAllocator *calloc_;
   ID3D12GraphicsCommandList *clist;
   ID3D12RootSignature *rs_empty;
   ID3D12RootSignature *rs_consts;
   ID3D12RootSignature *rs_table;
   ID3D12PipelineState *pso_tri;
   ID3D12PipelineState *pso_tri_consts;
   ID3D12PipelineState *pso_cs;
   ID3D12Resource *rt;
   ID3D12DescriptorHeap *rtv_heap;
   D3D12_CPU_DESCRIPTOR_HANDLE rtv;
   ID3D12DescriptorHeap *cbv_heap;
   D3D12_GPU_DESCRIPTOR_HANDLE table_a, table_b;
};

static HRESULT
wait_idle(struct bench *b)
{
   b->fence_value++;
   HRESULT hr = ID3D12CommandQueue_Signal(b->queue, b->fence, b->fence_value);
   if (hr != S_OK)
      return hr;
   hr = ID3D12Fence_SetEventOnCompletion(b->fence, b->fence_value, NULL);
   if (hr != S_OK)
      return hr;
   if (ID3D12Fence_GetCompletedValue(b->fence) < b->fence_value)
      return E_FAIL;
   return ID3D12Device_GetDeviceRemovedReason(b->dev);
}

static void
bind_graphics(struct bench *b, ID3D12RootSignature *rs,
              ID3D12PipelineState *pso)
{
   ID3D12GraphicsCommandList *list = b->list;
   transition(list, b->rt, D3D12_RESOURCE_STATE_COMMON,
              D3D12_RESOURCE_STATE_RENDER_TARGET);
   const FLOAT clear[4] = {0.0f, 0.0f, 0.0f, 1.0f};
   ID3D12GraphicsCommandList_ClearRenderTargetView(list, b->rtv, clear, 0,
                                                   NULL);
   ID3D12GraphicsCommandList_OMSetRenderTargets(list, 1, &b->rtv, FALSE,
                                                NULL);
   D3D12_VIEWPORT vp = {.Width = W, .Height = H, .MaxDepth = 1.0f};
   ID3D12GraphicsCommandList_RSSetViewports(list, 1, &vp);
   D3D12_RECT sc = {.right = W, .bottom = H};
   ID3D12GraphicsCommandList_RSSetScissorRects(list, 1, &sc);
   ID3D12GraphicsCommandList_SetGraphicsRootSignature(list, rs);
   ID3D12GraphicsCommandList_SetPipelineState(list, pso);
   ID3D12GraphicsCommandList_IASetPrimitiveTopology(
      list, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
}

enum draw_mode {
   DRAW_PLAIN,
   DRAW_ROOT_CONSTANT,
   DRAW_VIEWPORT,
   DRAW_PSO_TOGGLE,
};

/* Records one list of DRAWS_PER_LIST draws. Returns ns per draw for the
 * draw loop only; the fixed per-list setup is reported separately. */
static bool
record_draw_list(struct bench *b, enum draw_mode mode, double *setup_ns,
                 double *per_draw_ns, double *close_ns)
{
   HRESULT hr = ID3D12CommandAllocator_Reset(b->alloc);
   if (hr != S_OK)
      return false;
   hr = ID3D12GraphicsCommandList_Reset(b->list, b->alloc, NULL);
   if (hr != S_OK)
      return false;

   const bool consts = mode == DRAW_ROOT_CONSTANT;
   double t0 = now_ns();
   bind_graphics(b, consts ? b->rs_consts : b->rs_empty,
                 consts ? b->pso_tri_consts : b->pso_tri);
   double t1 = now_ns();

   D3D12_VIEWPORT vps[2] = {
      {.Width = W, .Height = H, .MaxDepth = 1.0f},
      {.Width = W / 2, .Height = H / 2, .MaxDepth = 1.0f},
   };
   for (unsigned i = 0; i < DRAWS_PER_LIST; i++) {
      switch (mode) {
      case DRAW_ROOT_CONSTANT:
         ID3D12GraphicsCommandList_SetGraphicsRoot32BitConstant(b->list, 0,
                                                                i, 0);
         break;
      case DRAW_VIEWPORT:
         ID3D12GraphicsCommandList_RSSetViewports(b->list, 1, &vps[i & 1]);
         break;
      case DRAW_PSO_TOGGLE:
         ID3D12GraphicsCommandList_SetPipelineState(
            b->list, (i & 1) ? b->pso_tri : b->pso_tri_consts);
         if (i & 1)
            ID3D12GraphicsCommandList_SetGraphicsRootSignature(b->list,
                                                               b->rs_empty);
         else
            ID3D12GraphicsCommandList_SetGraphicsRootSignature(b->list,
                                                               b->rs_consts);
         break;
      default:
         break;
      }
      ID3D12GraphicsCommandList_DrawInstanced(b->list, 3, 1, 0, 0);
   }
   double t2 = now_ns();
   transition(b->list, b->rt, D3D12_RESOURCE_STATE_RENDER_TARGET,
              D3D12_RESOURCE_STATE_COMMON);
   hr = ID3D12GraphicsCommandList_Close(b->list);
   double t3 = now_ns();
   if (hr != S_OK) {
      printf("       Close failed: 0x%08x\n", (unsigned)hr);
      return false;
   }
   *setup_ns = t1 - t0;
   *per_draw_ns = (t2 - t1) / DRAWS_PER_LIST;
   *close_ns = t3 - t2;
   return true;
}

static bool
record_dispatch_list(struct bench *b, bool alternate_tables,
                     double *per_dispatch_ns)
{
   HRESULT hr = ID3D12CommandAllocator_Reset(b->calloc_);
   if (hr != S_OK)
      return false;
   hr = ID3D12GraphicsCommandList_Reset(b->clist, b->calloc_, NULL);
   if (hr != S_OK)
      return false;

   ID3D12GraphicsCommandList_SetDescriptorHeaps(b->clist, 1, &b->cbv_heap);
   ID3D12GraphicsCommandList_SetComputeRootSignature(b->clist, b->rs_table);
   ID3D12GraphicsCommandList_SetComputeRootDescriptorTable(b->clist, 0,
                                                           b->table_a);
   ID3D12GraphicsCommandList_SetPipelineState(b->clist, b->pso_cs);

   double t1 = now_ns();
   for (unsigned i = 0; i < DISPATCHES_PER_LIST; i++) {
      if (alternate_tables)
         ID3D12GraphicsCommandList_SetComputeRootDescriptorTable(
            b->clist, 0, (i & 1) ? b->table_b : b->table_a);
      ID3D12GraphicsCommandList_Dispatch(b->clist, 1, 1, 1);
   }
   double t2 = now_ns();
   hr = ID3D12GraphicsCommandList_Close(b->clist);
   if (hr != S_OK) {
      printf("       compute Close failed: 0x%08x\n", (unsigned)hr);
      return false;
   }
   *per_dispatch_ns = (t2 - t1) / DISPATCHES_PER_LIST;
   return true;
}

int
main(int argc, char **argv)
{
   if (argc < 4) {
      fprintf(stderr, "usage: %s tri.dxil solid.dxil res_probe.dxil\n",
              argv[0]);
      return 2;
   }
   size_t vs_size, ps_size, cs_size;
   void *vs = slurp(argv[1], &vs_size);
   void *ps = slurp(argv[2], &ps_size);
   void *cs = slurp(argv[3], &cs_size);
   REQUIRE(vs && ps && cs, "read shader bytecode");

   struct bench b = {0};
   HRESULT hr = D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0,
                                  &IID_ID3D12Device, (void **)&b.dev);
   REQUIRE(hr == S_OK && b.dev, "D3D12CreateDevice");

   D3D12_COMMAND_QUEUE_DESC qd = {.Type = D3D12_COMMAND_LIST_TYPE_DIRECT};
   hr = ID3D12Device_CreateCommandQueue(b.dev, &qd, &IID_ID3D12CommandQueue,
                                        (void **)&b.queue);
   REQUIRE(hr == S_OK, "CreateCommandQueue");
   hr = ID3D12Device_CreateFence(b.dev, 0, D3D12_FENCE_FLAG_NONE,
                                 &IID_ID3D12Fence, (void **)&b.fence);
   REQUIRE(hr == S_OK, "CreateFence");

   /* Root signatures: empty, four root constants, and one descriptor table
    * (SRV t2, UAV u3, CBV b1: res_probe's bindings). */
   b.rs_empty = make_root_sig(b.dev, NULL, 0);
   REQUIRE(b.rs_empty, "empty root signature");
   D3D12_ROOT_PARAMETER consts = {
      .ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
      .Constants = {.ShaderRegister = 0, .Num32BitValues = 4},
      .ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL,
   };
   b.rs_consts = make_root_sig(b.dev, &consts, 1);
   REQUIRE(b.rs_consts, "root-constant root signature");
   D3D12_DESCRIPTOR_RANGE ranges[3] = {
      {.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV, .NumDescriptors = 1,
       .BaseShaderRegister = 2,
       .OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND},
      {.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV, .NumDescriptors = 1,
       .BaseShaderRegister = 3,
       .OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND},
      {.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV, .NumDescriptors = 1,
       .BaseShaderRegister = 1,
       .OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND},
   };
   D3D12_ROOT_PARAMETER table = {
      .ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
      .DescriptorTable = {.NumDescriptorRanges = 3, .pDescriptorRanges = ranges},
      .ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL,
   };
   b.rs_table = make_root_sig(b.dev, &table, 1);
   REQUIRE(b.rs_table, "descriptor-table root signature");

   /* --- PSO creation ------------------------------------------------- */
   D3D12_GRAPHICS_PIPELINE_STATE_DESC pd = {
      .pRootSignature = b.rs_empty,
      .VS = {.pShaderBytecode = vs, .BytecodeLength = vs_size},
      .PS = {.pShaderBytecode = ps, .BytecodeLength = ps_size},
      .NumRenderTargets = 1,
      .RTVFormats = {DXGI_FORMAT_R8G8B8A8_UNORM},
      .SampleDesc = {.Count = 1},
      .SampleMask = ~0u,
      .RasterizerState = {.FillMode = D3D12_FILL_MODE_SOLID,
                          .CullMode = D3D12_CULL_MODE_NONE,
                          .DepthClipEnable = TRUE},
      .PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
   };
   pd.BlendState.RenderTarget[0].RenderTargetWriteMask =
      D3D12_COLOR_WRITE_ENABLE_ALL;

   double pso_ms[PSO_REPEATS];
   for (unsigned i = 0; i < PSO_REPEATS; i++) {
      ID3D12PipelineState *pso = NULL;
      double t0 = now_ns();
      hr = ID3D12Device_CreateGraphicsPipelineState(
         b.dev, &pd, &IID_ID3D12PipelineState, (void **)&pso);
      double t1 = now_ns();
      REQUIRE(hr == S_OK && pso, "CreateGraphicsPipelineState");
      pso_ms[i] = (t1 - t0) / 1e6;
      if (i + 1 == PSO_REPEATS)
         b.pso_tri = pso;
      else
         ID3D12PipelineState_Release(pso);
   }
   double pso_gfx_first_ms = pso_ms[0];
   double pso_gfx_median_ms = percentile(pso_ms, PSO_REPEATS, 0.5);

   pd.pRootSignature = b.rs_consts;
   hr = ID3D12Device_CreateGraphicsPipelineState(
      b.dev, &pd, &IID_ID3D12PipelineState, (void **)&b.pso_tri_consts);
   REQUIRE(hr == S_OK, "CreateGraphicsPipelineState (root constants)");

   D3D12_COMPUTE_PIPELINE_STATE_DESC cd = {
      .pRootSignature = b.rs_table,
      .CS = {.pShaderBytecode = cs, .BytecodeLength = cs_size},
   };
   double cpso_ms[PSO_REPEATS];
   for (unsigned i = 0; i < PSO_REPEATS; i++) {
      ID3D12PipelineState *pso = NULL;
      double t0 = now_ns();
      hr = ID3D12Device_CreateComputePipelineState(
         b.dev, &cd, &IID_ID3D12PipelineState, (void **)&pso);
      double t1 = now_ns();
      REQUIRE(hr == S_OK && pso, "CreateComputePipelineState");
      cpso_ms[i] = (t1 - t0) / 1e6;
      if (i + 1 == PSO_REPEATS)
         b.pso_cs = pso;
      else
         ID3D12PipelineState_Release(pso);
   }
   double pso_cs_first_ms = cpso_ms[0];
   double pso_cs_median_ms = percentile(cpso_ms, PSO_REPEATS, 0.5);
   printf("pso    graphics first %.2f ms, median %.2f ms; compute first %.2f ms, "
          "median %.2f ms\n",
          pso_gfx_first_ms, pso_gfx_median_ms, pso_cs_first_ms,
          pso_cs_median_ms);

   /* --- Resources ---------------------------------------------------- */
   D3D12_HEAP_PROPERTIES rt_heap = {
      .Type = D3D12_HEAP_TYPE_CUSTOM,
      .CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
      .MemoryPoolPreference = D3D12_MEMORY_POOL_L0,
      .CreationNodeMask = 1,
      .VisibleNodeMask = 1,
   };
   D3D12_RESOURCE_DESC td = {
      .Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
      .Width = W,
      .Height = H,
      .DepthOrArraySize = 1,
      .MipLevels = 1,
      .Format = DXGI_FORMAT_R8G8B8A8_UNORM,
      .SampleDesc = {.Count = 1},
      .Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN,
      .Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET |
               D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS,
   };
   hr = ID3D12Device_CreateCommittedResource(
      b.dev, &rt_heap, D3D12_HEAP_FLAG_NONE, &td,
      D3D12_RESOURCE_STATE_COMMON, NULL, &IID_ID3D12Resource, (void **)&b.rt);
   REQUIRE(hr == S_OK, "create render target");
   D3D12_DESCRIPTOR_HEAP_DESC rhd = {.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                                     .NumDescriptors = 1};
   hr = ID3D12Device_CreateDescriptorHeap(b.dev, &rhd, &IID_ID3D12DescriptorHeap,
                                          (void **)&b.rtv_heap);
   REQUIRE(hr == S_OK, "create RTV heap");
   b.rtv = d12_test_descriptor_heap_get_cpu_start(b.rtv_heap);
   ID3D12Device_CreateRenderTargetView(b.dev, b.rt, NULL, b.rtv);

   ID3D12Resource *src = make_buffer(b.dev, 64 * 12, D3D12_RESOURCE_FLAG_NONE);
   ID3D12Resource *dst = make_buffer(b.dev, 64 * 4,
                                     D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
   ID3D12Resource *cb = make_buffer(b.dev, 256, D3D12_RESOURCE_FLAG_NONE);
   ID3D12Resource *cb_big = make_buffer(b.dev, 256 * DESCRIPTOR_COUNT,
                                        D3D12_RESOURCE_FLAG_NONE);
   REQUIRE(src && dst && cb && cb_big, "create compute buffers");

   /* --- Descriptor writes and copies --------------------------------- */
   D3D12_DESCRIPTOR_HEAP_DESC chd = {
      .Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
      .NumDescriptors = DESCRIPTOR_COUNT + 8,
      .Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
   };
   hr = ID3D12Device_CreateDescriptorHeap(b.dev, &chd, &IID_ID3D12DescriptorHeap,
                                          (void **)&b.cbv_heap);
   REQUIRE(hr == S_OK, "create shader-visible CBV/SRV/UAV heap");
   D3D12_DESCRIPTOR_HEAP_DESC shd = chd;
   shd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
   ID3D12DescriptorHeap *staging = NULL;
   hr = ID3D12Device_CreateDescriptorHeap(b.dev, &shd, &IID_ID3D12DescriptorHeap,
                                          (void **)&staging);
   REQUIRE(hr == S_OK, "create staging CBV/SRV/UAV heap");

   const UINT inc = ID3D12Device_GetDescriptorHandleIncrementSize(
      b.dev, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
   D3D12_CPU_DESCRIPTOR_HANDLE vis_cpu =
      d12_test_descriptor_heap_get_cpu_start(b.cbv_heap);
   D3D12_GPU_DESCRIPTOR_HANDLE vis_gpu =
      d12_test_descriptor_heap_get_gpu_start(b.cbv_heap);
   D3D12_CPU_DESCRIPTOR_HANDLE stg_cpu =
      d12_test_descriptor_heap_get_cpu_start(staging);

   /* Two complete tables at the end of the visible heap. */
   const unsigned table_base = DESCRIPTOR_COUNT;
   D3D12_SHADER_RESOURCE_VIEW_DESC srv = {
      .Format = DXGI_FORMAT_UNKNOWN,
      .ViewDimension = D3D12_SRV_DIMENSION_BUFFER,
      .Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
      .Buffer = {.NumElements = 64, .StructureByteStride = 12},
   };
   D3D12_UNORDERED_ACCESS_VIEW_DESC uav = {
      .Format = DXGI_FORMAT_UNKNOWN,
      .ViewDimension = D3D12_UAV_DIMENSION_BUFFER,
      .Buffer = {.NumElements = 64, .StructureByteStride = 4},
   };
   D3D12_CONSTANT_BUFFER_VIEW_DESC cbv = {
      .BufferLocation = ID3D12Resource_GetGPUVirtualAddress(cb),
      .SizeInBytes = 256,
   };
   for (unsigned t = 0; t < 2; t++) {
      D3D12_CPU_DESCRIPTOR_HANDLE h = {
         .ptr = vis_cpu.ptr + (table_base + 3 * t) * inc};
      ID3D12Device_CreateShaderResourceView(b.dev, src, &srv, h);
      h.ptr += inc;
      ID3D12Device_CreateUnorderedAccessView(b.dev, dst, NULL, &uav, h);
      h.ptr += inc;
      ID3D12Device_CreateConstantBufferView(b.dev, &cbv, h);
   }
   b.table_a.ptr = vis_gpu.ptr + table_base * inc;
   b.table_b.ptr = vis_gpu.ptr + (table_base + 3) * inc;

   const UINT64 cb_big_va = ID3D12Resource_GetGPUVirtualAddress(cb_big);
   double t0 = now_ns();
   for (unsigned i = 0; i < DESCRIPTOR_COUNT; i++) {
      D3D12_CONSTANT_BUFFER_VIEW_DESC d = {
         .BufferLocation = cb_big_va + (UINT64)i * 256, .SizeInBytes = 256};
      D3D12_CPU_DESCRIPTOR_HANDLE h = {.ptr = stg_cpu.ptr + i * inc};
      ID3D12Device_CreateConstantBufferView(b.dev, &d, h);
   }
   double t1 = now_ns();
   for (unsigned i = 0; i < DESCRIPTOR_COUNT; i++) {
      D3D12_CONSTANT_BUFFER_VIEW_DESC d = {
         .BufferLocation = cb_big_va + (UINT64)i * 256, .SizeInBytes = 256};
      D3D12_CPU_DESCRIPTOR_HANDLE h = {.ptr = vis_cpu.ptr + i * inc};
      ID3D12Device_CreateConstantBufferView(b.dev, &d, h);
   }
   double t2 = now_ns();
   ID3D12Device_CopyDescriptorsSimple(b.dev, DESCRIPTOR_COUNT, vis_cpu, stg_cpu,
                                      D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
   double t3 = now_ns();
   for (unsigned i = 0; i < DESCRIPTOR_COUNT; i++) {
      D3D12_CPU_DESCRIPTOR_HANDLE d = {.ptr = vis_cpu.ptr + i * inc};
      D3D12_CPU_DESCRIPTOR_HANDLE s = {.ptr = stg_cpu.ptr + i * inc};
      ID3D12Device_CopyDescriptorsSimple(b.dev, 1, d, s,
                                         D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
   }
   double t4 = now_ns();
   double cbv_write_staging_ns = (t1 - t0) / DESCRIPTOR_COUNT;
   double cbv_write_visible_ns = (t2 - t1) / DESCRIPTOR_COUNT;
   double copy_bulk_ns = (t3 - t2) / DESCRIPTOR_COUNT;
   double copy_single_ns = (t4 - t3) / DESCRIPTOR_COUNT;
   printf("desc   CBV write staging %.0f ns, visible %.0f ns; copy bulk %.0f ns, "
          "single %.0f ns (per descriptor)\n",
          cbv_write_staging_ns, cbv_write_visible_ns, copy_bulk_ns,
          copy_single_ns);

   /* --- Command lists ------------------------------------------------ */
   hr = ID3D12Device_CreateCommandAllocator(b.dev, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                            &IID_ID3D12CommandAllocator,
                                            (void **)&b.alloc);
   REQUIRE(hr == S_OK, "CreateCommandAllocator");
   hr = ID3D12Device_CreateCommandList(b.dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                       b.alloc, NULL,
                                       &IID_ID3D12GraphicsCommandList,
                                       (void **)&b.list);
   REQUIRE(hr == S_OK, "CreateCommandList");
   hr = ID3D12GraphicsCommandList_Close(b.list);
   REQUIRE(hr == S_OK, "Close fresh list");
   hr = ID3D12Device_CreateCommandAllocator(b.dev, D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                            &IID_ID3D12CommandAllocator,
                                            (void **)&b.calloc_);
   REQUIRE(hr == S_OK, "CreateCommandAllocator (compute)");
   hr = ID3D12Device_CreateCommandList(b.dev, 0, D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                       b.calloc_, NULL,
                                       &IID_ID3D12GraphicsCommandList,
                                       (void **)&b.clist);
   REQUIRE(hr == S_OK, "CreateCommandList (compute)");
   hr = ID3D12GraphicsCommandList_Close(b.clist);
   REQUIRE(hr == S_OK, "Close fresh compute list");

   /* Empty list: Reset + Close, the fixed per-list cost. */
   double empty_us[RECORD_REPEATS];
   for (unsigned r = 0; r < RECORD_REPEATS; r++) {
      double a = now_ns();
      hr = ID3D12CommandAllocator_Reset(b.alloc);
      HRESULT hr2 = ID3D12GraphicsCommandList_Reset(b.list, b.alloc, NULL);
      HRESULT hr3 = ID3D12GraphicsCommandList_Close(b.list);
      double c = now_ns();
      REQUIRE(hr == S_OK && hr2 == S_OK && hr3 == S_OK, "empty list cycle");
      empty_us[r] = (c - a) / 1e3;
   }
   double empty_list_us = percentile(empty_us, RECORD_REPEATS, 0.5);
   printf("list   empty Reset+Reset+Close %.1f us\n", empty_list_us);

   /* Draw recording in four flavours. Each flavour records RECORD_REPEATS
    * lists; the minimum per-draw time is the driver's steady-state cost. */
   static const char *mode_names[] = {"plain", "rootconst", "viewport",
                                      "psotoggle"};
   double draw_ns[4] = {0}, setup_ns_best[4] = {0}, close_ns_best[4] = {0};
   for (unsigned m = 0; m < 4; m++) {
      double best = 1e18, setup = 0, close = 0;
      for (unsigned r = 0; r < RECORD_REPEATS; r++) {
         double s, d, c;
         REQUIRE(record_draw_list(&b, (enum draw_mode)m, &s, &d, &c),
                 "record draw list");
         if (d < best) {
            best = d;
            setup = s;
            close = c;
         }
      }
      draw_ns[m] = best;
      setup_ns_best[m] = setup;
      close_ns_best[m] = close;
      printf("record %-9s %7.0f ns/draw  (setup %.1f us, close %.1f us, %u draws)\n",
             mode_names[m], best, setup / 1e3, close / 1e3, DRAWS_PER_LIST);
   }

   double dispatch_ns = 1e18, dispatch_table_ns = 1e18;
   for (unsigned r = 0; r < RECORD_REPEATS; r++) {
      double d;
      REQUIRE(record_dispatch_list(&b, false, &d), "record dispatch list");
      if (d < dispatch_ns)
         dispatch_ns = d;
      REQUIRE(record_dispatch_list(&b, true, &d),
              "record dispatch list (tables)");
      if (d < dispatch_table_ns)
         dispatch_table_ns = d;
   }
   printf("record dispatch %7.0f ns/dispatch, with table switch %.0f ns\n",
          dispatch_ns, dispatch_table_ns);

   /* --- Submit: CPU cost of ExecuteCommandLists and GPU wall time ---- */
   double s, d, c;
   REQUIRE(record_draw_list(&b, DRAW_PLAIN, &s, &d, &c),
           "record 20k-draw list for submission");
   ID3D12CommandList *lists[] = {(ID3D12CommandList *)b.list};
   double e0 = now_ns();
   ID3D12CommandQueue_ExecuteCommandLists(b.queue, 1, lists);
   double e1 = now_ns();
   hr = wait_idle(&b);
   double e2 = now_ns();
   REQUIRE(hr == S_OK, "20k-draw list executes and signals");
   double execute_call_us = (e1 - e0) / 1e3;
   double gpu_wall_ms = (e2 - e0) / 1e6;
   printf("submit ExecuteCommandLists(20k draws) call %.1f us, fence after %.2f ms\n",
          execute_call_us, gpu_wall_ms);

   REQUIRE(record_dispatch_list(&b, true, &d),
           "record 20k-dispatch list for submission");
   ID3D12CommandList *clists[] = {(ID3D12CommandList *)b.clist};
   e0 = now_ns();
   ID3D12CommandQueue_ExecuteCommandLists(b.queue, 1, clists);
   e1 = now_ns();
   hr = wait_idle(&b);
   e2 = now_ns();
   REQUIRE(hr == S_OK, "20k-dispatch list executes and signals");
   double execute_call_cs_us = (e1 - e0) / 1e3;
   double gpu_wall_cs_ms = (e2 - e0) / 1e6;
   printf("submit ExecuteCommandLists(20k dispatches) call %.1f us, fence after %.2f ms\n",
          execute_call_cs_us, gpu_wall_cs_ms);

   /* Reset cost after a submitted list: the allocator must take fresh
    * storage rather than rewind what the GPU may still read. */
   double reset_us[RECORD_REPEATS];
   for (unsigned r = 0; r < RECORD_REPEATS; r++) {
      REQUIRE(record_draw_list(&b, DRAW_PLAIN, &s, &d, &c), "record for reset");
      ID3D12CommandQueue_ExecuteCommandLists(b.queue, 1, lists);
      REQUIRE(wait_idle(&b) == S_OK, "submitted list completes");
      double a = now_ns();
      hr = ID3D12CommandAllocator_Reset(b.alloc);
      HRESULT hr2 = ID3D12GraphicsCommandList_Reset(b.list, b.alloc, NULL);
      double z = now_ns();
      REQUIRE(hr == S_OK && hr2 == S_OK, "Reset after submission");
      reset_us[r] = (z - a) / 1e3;
      REQUIRE(ID3D12GraphicsCommandList_Close(b.list) == S_OK, "close");
   }
   double reset_after_submit_us = percentile(reset_us, RECORD_REPEATS, 0.5);
   printf("reset  allocator+list after submit %.1f us\n", reset_after_submit_us);

   /* Round trip: one-draw list, Execute + Signal + wait. */
   double rt_us[ROUNDTRIP_REPEATS];
   for (unsigned r = 0; r < ROUNDTRIP_REPEATS; r++) {
      REQUIRE(ID3D12CommandAllocator_Reset(b.alloc) == S_OK, "rt reset");
      REQUIRE(ID3D12GraphicsCommandList_Reset(b.list, b.alloc, NULL) == S_OK,
              "rt list reset");
      bind_graphics(&b, b.rs_empty, b.pso_tri);
      ID3D12GraphicsCommandList_DrawInstanced(b.list, 3, 1, 0, 0);
      transition(b.list, b.rt, D3D12_RESOURCE_STATE_RENDER_TARGET,
                 D3D12_RESOURCE_STATE_COMMON);
      REQUIRE(ID3D12GraphicsCommandList_Close(b.list) == S_OK, "rt close");
      double a = now_ns();
      ID3D12CommandQueue_ExecuteCommandLists(b.queue, 1, lists);
      hr = wait_idle(&b);
      double z = now_ns();
      REQUIRE(hr == S_OK, "round trip completes");
      rt_us[r] = (z - a) / 1e3;
   }
   double rt_median_us = percentile(rt_us, ROUNDTRIP_REPEATS, 0.5);
   double rt_p95_us = percentile(rt_us, ROUNDTRIP_REPEATS, 0.95);
   double rt_min_us = rt_us[0];
   printf("submit one-draw round trip min %.0f us, median %.0f us, p95 %.0f us\n",
          rt_min_us, rt_median_us, rt_p95_us);

   /* Sanity: the render target should hold the triangle colour somewhere. */
   {
      uint8_t *map = malloc(W * 4 * H);
      bool drew = false;
      if (map && ID3D12Resource_Map(b.rt, 0, NULL, NULL) == S_OK) {
         if (ID3D12Resource_ReadFromSubresource(b.rt, map, W * 4, W * 4 * H, 0,
                                                NULL) == S_OK) {
            for (unsigned i = 0; i < W * H && !drew; i++)
               drew = map[i * 4] == 0x40 && map[i * 4 + 1] == 0x80;
         }
         ID3D12Resource_Unmap(b.rt, 0, NULL);
      }
      free(map);
      CHECK(drew, "render target holds the pixel-shader colour");
   }

   hr = ID3D12Device_GetDeviceRemovedReason(b.dev);
   CHECK(hr == S_OK, "device healthy at end");

   printf("MICROBENCH_JSON {"
          "\"pso_gfx_create_ms_first\": %.3f, \"pso_gfx_create_ms_median\": %.3f, "
          "\"pso_cs_create_ms_first\": %.3f, \"pso_cs_create_ms_median\": %.3f, "
          "\"record_ns_per_draw\": %.0f, \"record_ns_per_draw_rootconst\": %.0f, "
          "\"record_ns_per_draw_viewport\": %.0f, \"record_ns_per_draw_psotoggle\": %.0f, "
          "\"record_list_setup_us\": %.2f, \"record_list_close_us\": %.2f, "
          "\"record_ns_per_dispatch\": %.0f, \"record_ns_per_dispatch_table\": %.0f, "
          "\"descriptor_cbv_write_ns\": %.0f, \"descriptor_cbv_write_visible_ns\": %.0f, "
          "\"descriptor_copy_bulk_ns\": %.0f, \"descriptor_copy_single_ns\": %.0f, "
          "\"empty_list_cycle_us\": %.2f, \"reset_after_submit_us\": %.2f, "
          "\"execute_call_us_20k_draws\": %.2f, \"gpu_wall_ms_20k_draws\": %.3f, "
          "\"execute_call_us_20k_dispatches\": %.2f, \"gpu_wall_ms_20k_dispatches\": %.3f, "
          "\"roundtrip_us_min\": %.1f, \"roundtrip_us_median\": %.1f, \"roundtrip_us_p95\": %.1f, "
          "\"failures\": %u}\n",
          pso_gfx_first_ms, pso_gfx_median_ms, pso_cs_first_ms, pso_cs_median_ms,
          draw_ns[0], draw_ns[1], draw_ns[2], draw_ns[3],
          setup_ns_best[0] / 1e3, close_ns_best[0] / 1e3,
          dispatch_ns, dispatch_table_ns,
          cbv_write_staging_ns, cbv_write_visible_ns, copy_bulk_ns, copy_single_ns,
          empty_list_us, reset_after_submit_us,
          execute_call_us, gpu_wall_ms, execute_call_cs_us, gpu_wall_cs_ms,
          rt_min_us, rt_median_us, rt_p95_us, failures);

   ID3D12GraphicsCommandList_Release(b.list);
   ID3D12GraphicsCommandList_Release(b.clist);
   ID3D12CommandAllocator_Release(b.alloc);
   ID3D12CommandAllocator_Release(b.calloc_);
   ID3D12DescriptorHeap_Release(staging);
   ID3D12DescriptorHeap_Release(b.cbv_heap);
   ID3D12DescriptorHeap_Release(b.rtv_heap);
   ID3D12Resource_Release(src);
   ID3D12Resource_Release(dst);
   ID3D12Resource_Release(cb);
   ID3D12Resource_Release(cb_big);
   ID3D12Resource_Release(b.rt);
   ID3D12PipelineState_Release(b.pso_tri);
   ID3D12PipelineState_Release(b.pso_tri_consts);
   ID3D12PipelineState_Release(b.pso_cs);
   ID3D12RootSignature_Release(b.rs_empty);
   ID3D12RootSignature_Release(b.rs_consts);
   ID3D12RootSignature_Release(b.rs_table);
   ID3D12Fence_Release(b.fence);
   ID3D12CommandQueue_Release(b.queue);
   ID3D12Device_Release(b.dev);
   free(vs);
   free(ps);
   free(cs);
   return failures ? 1 : 0;
}
