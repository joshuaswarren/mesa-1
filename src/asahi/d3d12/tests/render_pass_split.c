/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * A declared render pass the driver has to split keeps its contents.
 *
 * BeginRenderPass declares CLEAR on entry and PRESERVE on exit. Between
 * two draws the list changes the sample positions, which the driver can
 * only honour by ending the hardware render and starting another. The
 * second render used to apply the pass's entry traffic again -- a CLEAR,
 * which threw the first draw away -- because every render opened inside
 * the pass copied the pass's plan. Now a resuming render loads what the
 * previous one stored.
 *
 *   left half   the first draw, before the split
 *   right half  the second draw, after it
 *
 * Both halves must be lit and the rest of the target must hold the clear
 * colour. Needs a device (GPU). Takes tri.dxil and solid.dxil.
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

#define W 256
#define H 256
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
   void *buf = malloc(n > 0 ? (size_t)n : 1);
   if (buf && fread(buf, 1, (size_t)n, f) != (size_t)n) {
      free(buf);
      buf = NULL;
   }
   fclose(f);
   *size = (size_t)n;
   return buf;
}

int
main(int argc, char **argv)
{
   if (argc < 3) {
      printf("usage: %s tri.dxil solid.dxil\n", argv[0]);
      return 2;
   }

   size_t vs_size = 0, ps_size = 0;
   void *vs = slurp(argv[1], &vs_size);
   void *ps = slurp(argv[2], &ps_size);
   if (!vs || !ps) {
      printf("FAIL - could not read the shaders\n");
      return 1;
   }

   ID3D12Device *dev = NULL;
   HRESULT hr = D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0,
                                  &IID_ID3D12Device, (void **)&dev);
   CHECK(hr == S_OK && dev, "D3D12CreateDevice");
   if (!dev)
      return 1;

   D3D12_ROOT_SIGNATURE_DESC rsd = {0};
   ID3DBlob *sig = NULL;
   hr = D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &sig,
                                    NULL);
   ID3D12RootSignature *root_sig = NULL;
   if (hr == S_OK)
      hr = ID3D12Device_CreateRootSignature(
         dev, 0, ID3D10Blob_GetBufferPointer(sig),
         ID3D10Blob_GetBufferSize(sig), &IID_ID3D12RootSignature,
         (void **)&root_sig);
   CHECK(hr == S_OK && root_sig, "create root signature");
   if (sig)
      ID3D10Blob_Release(sig);
   if (!root_sig)
      return 1;

   D3D12_GRAPHICS_PIPELINE_STATE_DESC pd = {
      .pRootSignature = root_sig,
      .VS = {vs, vs_size},
      .PS = {ps, ps_size},
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
   ID3D12PipelineState *pso = NULL;
   hr = ID3D12Device_CreateGraphicsPipelineState(
      dev, &pd, &IID_ID3D12PipelineState, (void **)&pso);
   CHECK(hr == S_OK && pso, "triangle PSO compiles");
   if (!pso)
      return 1;

   D3D12_HEAP_PROPERTIES heap = D12_TEST_UMA_HEAP;
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
   ID3D12Resource *rt = NULL;
   TEST_OBJECT(ID3D12Device_CreateCommittedResource(
                  dev, &heap, D3D12_HEAP_FLAG_NONE, &td,
                  D3D12_RESOURCE_STATE_COMMON, NULL, &IID_ID3D12Resource,
                  (void **)&rt),
               rt, "create the render target");
   D3D12_DESCRIPTOR_HEAP_DESC hd = {.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                                    .NumDescriptors = 1};
   ID3D12DescriptorHeap *rtv_heap = NULL;
   TEST_OBJECT(ID3D12Device_CreateDescriptorHeap(
                  dev, &hd, &IID_ID3D12DescriptorHeap, (void **)&rtv_heap),
               rtv_heap, "create the RTV heap");
   D3D12_CPU_DESCRIPTOR_HANDLE rtv =
      d12_test_descriptor_heap_get_cpu_start(rtv_heap);
   ID3D12Device_CreateRenderTargetView(dev, rt, NULL, rtv);

   D3D12_COMMAND_QUEUE_DESC qd = {.Type = D3D12_COMMAND_LIST_TYPE_DIRECT};
   ID3D12CommandQueue *queue = NULL;
   TEST_OBJECT(ID3D12Device_CreateCommandQueue(dev, &qd,
                                               &IID_ID3D12CommandQueue,
                                               (void **)&queue),
               queue, "create the queue");
   ID3D12CommandAllocator *alloc = NULL;
   TEST_OBJECT(ID3D12Device_CreateCommandAllocator(
                  dev, D3D12_COMMAND_LIST_TYPE_DIRECT,
                  &IID_ID3D12CommandAllocator, (void **)&alloc),
               alloc, "create the allocator");
   ID3D12GraphicsCommandList *list = NULL;
   TEST_OBJECT(ID3D12Device_CreateCommandList(
                  dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, NULL,
                  &IID_ID3D12GraphicsCommandList, (void **)&list),
               list, "create the list");
   ID3D12GraphicsCommandList4 *list4 = NULL;
   hr = ID3D12GraphicsCommandList_QueryInterface(
      list, &IID_ID3D12GraphicsCommandList4, (void **)&list4);
   CHECK(hr == S_OK && list4, "the list answers ID3D12GraphicsCommandList4");
   if (!list4)
      return 1;
   ID3D12Fence *fence = NULL;
   TEST_OBJECT(ID3D12Device_CreateFence(dev, 0, D3D12_FENCE_FLAG_NONE,
                                        &IID_ID3D12Fence, (void **)&fence),
               fence, "create the fence");

   uint8_t *map = malloc(STRIDE * H);
   memset(map, 0x55, STRIDE * H);
   TEST_CALL(d12_test_opaque_texture_transfer(rt, map, STRIDE, STRIDE * H,
                                              true),
             "publish texture poison in COMMON");

   d12_test_transition(list, rt, D3D12_RESOURCE_STATE_COMMON,
                       D3D12_RESOURCE_STATE_RENDER_TARGET);

   D3D12_RENDER_PASS_RENDER_TARGET_DESC pass_rt = {
      .cpuDescriptor = rtv,
      .BeginningAccess = {
         .Type = D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_CLEAR,
         .Clear = {.ClearValue = {.Format = DXGI_FORMAT_R8G8B8A8_UNORM,
                                  .Color = {0.0f, 0.0f, 0.0f, 1.0f}}},
      },
      .EndingAccess = {.Type = D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_PRESERVE},
   };
   ID3D12GraphicsCommandList4_BeginRenderPass(list4, 1, &pass_rt, NULL,
                                              D3D12_RENDER_PASS_FLAG_NONE);

   ID3D12GraphicsCommandList_SetGraphicsRootSignature(list, root_sig);
   ID3D12GraphicsCommandList_SetPipelineState(list, pso);
   ID3D12GraphicsCommandList_IASetPrimitiveTopology(
      list, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
   D3D12_RECT sc = {.right = W, .bottom = H};
   ID3D12GraphicsCommandList_RSSetScissorRects(list, 1, &sc);

   /* The first draw, in the left half. */
   D3D12_VIEWPORT left = {.Width = W / 2, .Height = H, .MaxDepth = 1.0f};
   ID3D12GraphicsCommandList_RSSetViewports(list, 1, &left);
   ID3D12GraphicsCommandList_DrawInstanced(list, 3, 1, 0, 0);

   /* Sample positions are fixed for a whole hardware render, so setting
    * them here is a split the driver cannot avoid. The standard position
    * keeps the rasterisation unchanged. */
   D3D12_SAMPLE_POSITION centre = {0, 0};
   ID3D12GraphicsCommandList4_SetSamplePositions(list4, 1, 1, &centre);

   /* The second draw, in the right half, in the render that resumes. */
   D3D12_VIEWPORT right = {.TopLeftX = W / 2, .Width = W / 2, .Height = H,
                           .MaxDepth = 1.0f};
   ID3D12GraphicsCommandList_RSSetViewports(list, 1, &right);
   ID3D12GraphicsCommandList_DrawInstanced(list, 3, 1, 0, 0);

   ID3D12GraphicsCommandList4_EndRenderPass(list4);
   d12_test_transition(list, rt, D3D12_RESOURCE_STATE_RENDER_TARGET,
                       D3D12_RESOURCE_STATE_COMMON);

   hr = ID3D12GraphicsCommandList_Close(list);
   CHECK(hr == S_OK, "the list closes cleanly");
   if (hr != S_OK)
      goto teardown;
   TEST_CALL(d12_test_submit_closed(dev, list, queue, fence, 1),
             "submit and wait");
   memset(map, 0x33, STRIDE * H);
   TEST_CALL(d12_test_opaque_texture_transfer(rt, map, STRIDE, STRIDE * H,
                                              false),
             "read back the pixels after the fence");

   {
      unsigned lit_left = 0, lit_right = 0, poison = 0, other = 0;
      for (unsigned y = 0; y < H; y++) {
         for (unsigned x = 0; x < W; x++) {
            const uint8_t *c = map + y * STRIDE + x * 4;
            const bool clear = c[0] == 0 && c[1] == 0 && c[2] == 0 &&
                               c[3] == 255;
            const bool stale = c[0] == 0x55 && c[1] == 0x55 &&
                               c[2] == 0x55 && c[3] == 0x55;
            if (stale)
               poison++;
            else if (!clear && x < W / 2)
               lit_left++;
            else if (!clear)
               lit_right++;
            if (!stale && !clear && c[3] != 255)
               other++;
         }
      }
      printf("       left %u lit, right %u lit, %u poison, %u other\n",
             lit_left, lit_right, poison, other);
      CHECK(lit_right > 0, "the draw after the split is present");
      CHECK(lit_left > 0, "the draw before the split survives the split");
      CHECK(poison == 0, "the pass's CLEAR reached every pixel");
      CHECK(lit_left > 0 && lit_right > 0 &&
               lit_left > lit_right - lit_right / 8 &&
               lit_left < lit_right + lit_right / 8,
            "both halves hold one triangle of the same size");
   }

teardown:
   free(map);
   if (fence)
      ID3D12Fence_Release(fence);
   ID3D12GraphicsCommandList4_Release(list4);
   ID3D12GraphicsCommandList_Release(list);
   ID3D12CommandAllocator_Release(alloc);
   ID3D12CommandQueue_Release(queue);
   ID3D12DescriptorHeap_Release(rtv_heap);
   ID3D12Resource_Release(rt);
   ID3D12PipelineState_Release(pso);
   ID3D12RootSignature_Release(root_sig);
   ID3D12Device_Release(dev);
   free(vs);
   free(ps);
   printf(failures ? "RENDER PASS SPLIT: %d FAILURES\n"
                   : "RENDER PASS SPLIT: ALL PASS\n",
          failures);
   return failures ? 1 : 0;
}
