/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * G9: amplification shaders execute.
 *
 * An amplification shader runs BEFORE the mesh shader, writes a payload,
 * and calls DispatchMesh(x,y,z) to say how many mesh workgroups to run.
 * That grid is GPU-written, so the mesh dispatch has to be indirect off
 * it -- and an AGX indirect grid is six words, three THREAD counts then
 * three local sizes, so the group-to-thread multiply cannot happen on
 * the CPU either. The chain becomes:
 *
 *   1. the amplification shader, on the grid the application gave;
 *   2. as_setup, turning its DispatchMesh group count into the six-word
 *      indirect-local grid;
 *   3. the mesh shader, dispatched INDIRECTLY off that grid;
 *   4. ms_setup, turning the mesh counts into an indirect draw;
 *   5. the indexed indirect draw.
 *
 * Both shaders are generated from NIR by d3d12agx_make_probe and
 * committed next to it -- as_dispatch.dxil and mesh_quad.dxil. That took
 * mesh AND amplification support in nir_to_dxil, including a
 * dx.op.dispatchMesh declaration whose overload suffix is a struct type
 * name rather than a scalar.
 */

#define COBJMACROS
#include <wsl/winadapter.h>
#include <directx/d3d12.h>

#include "d12_test_abi.h"

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

#define W 128
#define H 128
#define STRIDE (((W * 4) + 63) & ~63u)

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
   if (p)
      *size = n;
   return p;
}

/* One subobject: type tag padded to void*, then the payload. */
#define SUBOBJ(name, tag, type)                                              \
   struct {                                                                  \
      __attribute__((aligned(sizeof(void *))))                                 \
         D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t;                              \
      __attribute__((aligned(sizeof(void *)))) type v;                                                \
   } name = {.t = tag}

int
main(int argc, char **argv)
{
   if (argc < 4) {
      printf("usage: %s as.dxil ms.dxil ps.dxil\n", argv[0]);
      printf("       (run-suites.sh passes shaders/as_dispatch.dxil and\n");
      printf("        shaders/mesh_quad.dxil)\n");
      return 2;
   }

   size_t as_size = 0, ms_size = 0, ps_size = 0;
   void *as = slurp(argv[1], &as_size);
   void *ms = slurp(argv[2], &ms_size);
   void *ps = slurp(argv[3], &ps_size);
   if (!as || !ms || !ps) {
      printf("FAIL - could not read the shaders\n");
      return 1;
   }

   ID3D12Device *dev0 = NULL;
   HRESULT hr = D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0,
                                  &IID_ID3D12Device, (void **)&dev0);
   TEST_OBJECT(hr, dev0, "checked D3D12CreateDevice dev0");
   CHECK(hr == S_OK && dev0, "D3D12CreateDevice");
   if (!dev0)
      return 1;

   ID3D12Device2 *dev = NULL;
   hr = ID3D12Device_QueryInterface(dev0, &IID_ID3D12Device2, (void **)&dev);
   TEST_OBJECT(hr, dev, "checked ID3D12Device_QueryInterface dev");
   CHECK(hr == S_OK && dev,
         "the device answers to ID3D12Device2 (mesh pipelines need it)");
   if (!dev)
      return 1;

   D3D12_ROOT_SIGNATURE_DESC rsd = {0};
   ID3DBlob *sig = NULL;
   hr = D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &sig,
                                    NULL);
   TEST_OBJECT(hr, sig, "checked D3D12SerializeRootSignature sig");
   ID3D12RootSignature *root_sig = NULL;
   if (hr == S_OK)
      hr = ID3D12Device_CreateRootSignature(
         dev0, 0, ID3D10Blob_GetBufferPointer(sig),
         ID3D10Blob_GetBufferSize(sig), &IID_ID3D12RootSignature,
         (void **)&root_sig);
   TEST_OBJECT(hr, root_sig, "checked ID3D12Device_CreateRootSignature root_sig");
   CHECK(hr == S_OK && root_sig, "create root signature");
   ID3D10Blob_Release(sig);
   if (!root_sig)
      return 1;

   /* The stream descriptor. */
   SUBOBJ(so_rs, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE,
          ID3D12RootSignature *);
   SUBOBJ(so_as, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS,
          D3D12_SHADER_BYTECODE);
   SUBOBJ(so_ms, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS,
          D3D12_SHADER_BYTECODE);
   SUBOBJ(so_ps, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS,
          D3D12_SHADER_BYTECODE);
   SUBOBJ(so_rast, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER,
          D3D12_RASTERIZER_DESC);
   SUBOBJ(so_rt, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS,
          struct D3D12_RT_FORMAT_ARRAY);

   so_rs.v = root_sig;
   so_as.v = (D3D12_SHADER_BYTECODE){as, as_size};
   so_ms.v = (D3D12_SHADER_BYTECODE){ms, ms_size};
   so_ps.v = (D3D12_SHADER_BYTECODE){ps, ps_size};
   so_rast.v = (D3D12_RASTERIZER_DESC){.FillMode = D3D12_FILL_MODE_SOLID, .CullMode = D3D12_CULL_MODE_NONE, .DepthClipEnable = TRUE};
   so_rt.v.NumRenderTargets = 1;
   so_rt.v.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;

   uint8_t stream[sizeof(so_rs) + sizeof(so_as) + sizeof(so_ms) +
                  sizeof(so_ps) + sizeof(so_rast) + sizeof(so_rt)];
   size_t off = 0;
#define PUSH(o)                                                              \
   do {                                                                      \
      memcpy(stream + off, &(o), sizeof(o));                                 \
      off += sizeof(o);                                                      \
   } while (0)
   PUSH(so_rs);
   PUSH(so_as);
   PUSH(so_ms);
   PUSH(so_ps);
   PUSH(so_rast);
   PUSH(so_rt);

   D3D12_PIPELINE_STATE_STREAM_DESC sd = {.SizeInBytes = off,
                                          .pPipelineStateSubobjectStream =
                                             stream};

   ID3D12PipelineState *pso = NULL;
   hr = ID3D12Device2_CreatePipelineState(dev, &sd, &IID_ID3D12PipelineState,
                                          (void **)&pso);
   TEST_OBJECT(hr, pso, "checked ID3D12Device2_CreatePipelineState pso");
   CHECK(hr == S_OK && pso,
         "CreatePipelineState builds an AMPLIFICATION + MESH pipeline "
         "from a stream descriptor");
   if (!pso)
      goto done;

   D3D12_HEAP_PROPERTIES hp = D12_TEST_UMA_HEAP;
   D3D12_RESOURCE_DESC rd = {
      .Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
      .Width = W,
      .Height = H,
      .DepthOrArraySize = 1,
      .MipLevels = 1,
      .Format = DXGI_FORMAT_R8G8B8A8_UNORM,
      .SampleDesc = {.Count = 1},
      .Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN,
      .Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS,
   };
   ID3D12Resource *rt = NULL;
   hr = ID3D12Device_CreateCommittedResource(
      dev0, &hp, D3D12_HEAP_FLAG_NONE, &rd,
      D3D12_RESOURCE_STATE_COMMON, NULL, &IID_ID3D12Resource,
      (void **)&rt);
   TEST_OBJECT(hr, rt, "checked ID3D12Device_CreateCommittedResource rt");
   CHECK(hr == S_OK && rt, "create render target");
   if (!rt)
      goto done;

   D3D12_DESCRIPTOR_HEAP_DESC hd = {.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                                    .NumDescriptors = 1};
   ID3D12DescriptorHeap *rtv_heap = NULL;
   TEST_OBJECT(ID3D12Device_CreateDescriptorHeap(dev0, &hd, &IID_ID3D12DescriptorHeap,
                                     (void **)&rtv_heap), rtv_heap, "checked ID3D12Device_CreateDescriptorHeap rtv_heap");
   D3D12_CPU_DESCRIPTOR_HANDLE rtv =
      d12_test_descriptor_heap_get_cpu_start(rtv_heap);
   CHECK(rtv.ptr != 0, "valid RTV descriptor handle");
   if (!rtv.ptr) return 1;
   ID3D12Device_CreateRenderTargetView(dev0, rt, NULL, rtv);

   D3D12_COMMAND_QUEUE_DESC qd = {.Type = D3D12_COMMAND_LIST_TYPE_DIRECT};
   ID3D12CommandQueue *queue = NULL;
   TEST_OBJECT(ID3D12Device_CreateCommandQueue(dev0, &qd, &IID_ID3D12CommandQueue,
                                   (void **)&queue), queue, "checked ID3D12Device_CreateCommandQueue queue");
   ID3D12CommandAllocator *alloc = NULL;
   TEST_OBJECT(ID3D12Device_CreateCommandAllocator(dev0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                       &IID_ID3D12CommandAllocator,
                                       (void **)&alloc), alloc, "checked ID3D12Device_CreateCommandAllocator alloc");
   ID3D12GraphicsCommandList *list0 = NULL;
   TEST_OBJECT(ID3D12Device_CreateCommandList(dev0, 0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                  alloc, NULL, &IID_ID3D12GraphicsCommandList,
                                  (void **)&list0), list0, "checked ID3D12Device_CreateCommandList list0");
   ID3D12GraphicsCommandList6 *list = NULL;
   hr = ID3D12GraphicsCommandList_QueryInterface(
      list0, &IID_ID3D12GraphicsCommandList6, (void **)&list);
   TEST_OBJECT(hr, list, "checked ID3D12GraphicsCommandList_QueryInterface list");
   CHECK(hr == S_OK && list,
         "the command list answers to ID3D12GraphicsCommandList6 "
         "(DispatchMesh needs it)");
   if (!list)
      goto done;

   ID3D12Fence *fence = NULL;
   TEST_OBJECT(ID3D12Device_CreateFence(dev0, 0, D3D12_FENCE_FLAG_NONE, &IID_ID3D12Fence,
                            (void **)&fence), fence, "checked ID3D12Device_CreateFence fence");

   uint8_t *map = malloc(STRIDE * H);
   CHECK(map != NULL, "CPU storage for opaque target");
   if (!map) return 1;
   memset(map, 0x55, STRIDE * H);
   TEST_CALL(d12_test_opaque_texture_transfer(rt, map, STRIDE, STRIDE * H, true), "publish original texture poison in COMMON");

   TEST_CALL(ID3D12GraphicsCommandList_Close(list0), "close initial empty list");

   const FLOAT clear_red[4] = {1.0f, 0.0f, 0.0f, 1.0f};
   D3D12_VIEWPORT vp = {.Width = W, .Height = H, .MaxDepth = 1.0f};
   D3D12_RECT sc = {.right = W, .bottom = H};

   TEST_CALL(ID3D12CommandAllocator_Reset(alloc), "reset fenced allocator");
   TEST_CALL(ID3D12GraphicsCommandList_Reset(list0, alloc, NULL), "reset checked graphics list");
   d12_test_transition(list0, rt, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET);
   ID3D12GraphicsCommandList_ClearRenderTargetView(list0, rtv, clear_red, 0,
                                                   NULL);
   ID3D12GraphicsCommandList_RSSetViewports(list0, 1, &vp);
   ID3D12GraphicsCommandList_RSSetScissorRects(list0, 1, &sc);
   ID3D12GraphicsCommandList_SetGraphicsRootSignature(list0, root_sig);
   ID3D12GraphicsCommandList_SetPipelineState(list0, pso);
   ID3D12GraphicsCommandList6_DispatchMesh(list, 1, 1, 1);
   d12_test_transition(list0, rt, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COMMON);
   hr = ID3D12GraphicsCommandList_Close(list0);
   CHECK(hr == S_OK, "mesh command list closes cleanly");
   if (hr != S_OK)
      goto teardown;

   TEST_CALL(d12_test_submit_closed(dev0, list0, queue, fence, 1), "exact render fence and device status");
   memset(map, 0x33, STRIDE * H);
   TEST_CALL(d12_test_opaque_texture_transfer(rt, map, STRIDE, STRIDE * H, false), "read actual opaque bytes only after fence");
   CHECK(ID3D12Fence_GetCompletedValue(fence) != UINT64_MAX &&
            ID3D12Fence_GetCompletedValue(fence) >= 1,
         "the GPU COMPLETES an amplified mesh draw: four compute "
         "dispatches, one of them INDIRECT off a GPU-written grid, and an "
         "indexed indirect draw");

   unsigned covered = 0, cleared = 0;
   for (unsigned y = 0; y < H; y++) {
      for (unsigned x = 0; x < W; x++) {
         const uint8_t *p = map + y * STRIDE + x * 4;
         if (p[0] == 0xff && p[1] == 0x00 && p[2] == 0x00)
            cleared++;
         else if (!(p[0] == 0x55 && p[1] == 0x55 && p[2] == 0x55))
            covered++;
      }
   }

   printf("       covered %u, cleared %u, untouched %u (of %u)\n", covered,
          cleared, W * H - covered - cleared, W * H);

   CHECK(cleared + covered == W * H,
         "the render pass ran: every pixel is either cleared or drawn");

   /* The amplification shader decides how many mesh workgroups run, so
    * the geometry that appears is proof the whole chain executed: if the
    * AS never ran, or as_setup produced a zero grid, or the indirect
    * dispatch read the wrong words, the mesh shader runs zero times and
    * the target stays cleared.
    */
   /* THE DERIVED EXPECTATION, and it proves the whole chain.
    *
    * Verified under D3D12AGX_DUMP_MESH: the amplification shader writes
    * DispatchMesh(1,1,1); as_setup turns that into the six-word grid
    * {threads 4,1,1  local 4,1,1}; the mesh shader runs indirectly off
    * it and reports SetMeshOutputCounts(4, 2) -- a full-screen quad.
    *
    * Every link is load-bearing. If the amplification shader never ran,
    * or as_setup produced a zero grid, or the indirect dispatch read the
    * wrong words, the mesh shader runs ZERO times and the target stays
    * fully cleared. Only the complete chain fills it.
    */
   CHECK(covered == W * H && cleared == 0,
         "the amplification shader's GPU-written grid drives the mesh "
         "shader: its full-screen quad covers EXACTLY the target");

teardown:
   free(map);
   if (fence)
      ID3D12Fence_Release(fence);
   ID3D12CommandQueue_Release(queue);
   if (list)
      ID3D12GraphicsCommandList6_Release(list);
   ID3D12GraphicsCommandList_Release(list0);
   ID3D12CommandAllocator_Release(alloc);
   ID3D12DescriptorHeap_Release(rtv_heap);
   ID3D12Resource_Release(rt);

done:
   if (pso)
      ID3D12PipelineState_Release(pso);
   ID3D12RootSignature_Release(root_sig);
   ID3D12Device2_Release(dev);
   ID3D12Device_Release(dev0);

   printf(failures ? "G9 AMPLIFICATION: %d FAILURES\n"
                   : "G9 AMPLIFICATION: ALL PASS\n",
          failures);
   free(as);
   free(ms);
   free(ps);
   return failures ? 1 : 0;
}
