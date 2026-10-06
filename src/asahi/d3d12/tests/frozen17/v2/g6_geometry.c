/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * G6: geometry shaders execute.
 *
 * AGX has no hardware geometry stage. A GS becomes three programs: the
 * app's vertex shader rewritten to write a buffer instead of the
 * rasterizer, the geometry body, and a rasterization shader that replays
 * the body once per output vertex and IS the hardware vertex shader.
 * The first two are compute, dispatched into a stream ordered before the
 * render pass.
 *
 * WHAT THIS TEST CAN AND CANNOT CLAIM.
 *
 * It proves the machinery: that a geometry pipeline is created rather
 * than refused, that the two compute dispatches and the rewritten draw
 * form a command stream the GPU accepts and completes, and that binding
 * a GS changes what is rasterised.
 *
 * It does NOT check the emitted geometry pixel-for-pixel the way
 * g4_triangle checks its triangle. That needs a vertex shader writing a
 * full float4 varying from the vertex ID alone, and no such shader is
 * available: `dxc` does not run on this box, and of the corpus vertex
 * shaders that write VAR0, the ones with a full xyzw write read vertex
 * ATTRIBUTES (which need an attribute-fetch prolog this driver does not
 * have yet) and the attribute-free ones write only .x. Closing that gap
 * is what the NIR->DXIL probe generator in docs/roadmap.md is for. Until
 * then this suite states exactly what it verified and no more.
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

int
main(int argc, char **argv)
{
   if (argc < 4) {
      printf("usage: %s vs.dxil ps.dxil gs.dxil\n", argv[0]);
      printf("       (the GS corpus is LGPL and not vendored; supply one)\n");
      return 2;
   }

   size_t vs_size = 0, ps_size = 0, gs_size = 0;
   void *vs = slurp(argv[1], &vs_size);
   void *ps = slurp(argv[2], &ps_size);
   void *gs = slurp(argv[3], &gs_size);
   if (!vs || !ps || !gs) {
      printf("FAIL - could not read the shaders\n");
      return 1;
   }

   ID3D12Device *dev = NULL;
   HRESULT hr = D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0,
                                  &IID_ID3D12Device, (void **)&dev);
   TEST_OBJECT(hr, dev, "checked D3D12CreateDevice dev");
   CHECK(hr == S_OK && dev, "D3D12CreateDevice");
   if (!dev)
      return 1;

   D3D12_ROOT_SIGNATURE_DESC rsd = {
      .Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
   };
   ID3DBlob *sig = NULL;
   hr = D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &sig,
                                    NULL);
   TEST_OBJECT(hr, sig, "checked D3D12SerializeRootSignature sig");
   ID3D12RootSignature *root_sig = NULL;
   if (hr == S_OK)
      hr = ID3D12Device_CreateRootSignature(
         dev, 0, ID3D10Blob_GetBufferPointer(sig),
         ID3D10Blob_GetBufferSize(sig), &IID_ID3D12RootSignature,
         (void **)&root_sig);
   TEST_OBJECT(hr, root_sig, "checked ID3D12Device_CreateRootSignature root_sig");
   CHECK(hr == S_OK && root_sig, "create root signature");
   ID3D10Blob_Release(sig);
   if (!root_sig)
      return 1;

   D3D12_GRAPHICS_PIPELINE_STATE_DESC pd = {
      .pRootSignature = root_sig,
      .VS = {.pShaderBytecode = vs, .BytecodeLength = vs_size},
      .PS = {.pShaderBytecode = ps, .BytecodeLength = ps_size},
      .NumRenderTargets = 1,
      .RTVFormats = {DXGI_FORMAT_R8G8B8A8_UNORM},
      .RasterizerState = {.FillMode = D3D12_FILL_MODE_SOLID, .CullMode = D3D12_CULL_MODE_NONE, .DepthClipEnable = TRUE},
      .SampleDesc = {.Count = 1},
      .SampleMask = ~0u,
      .PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
   };
   pd.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;

   /* Triangle and point IA input have distinct legal PSO topology types. */
   ID3D12PipelineState *pso_triangle = NULL;
   TEST_OBJECT(ID3D12Device_CreateGraphicsPipelineState(dev, &pd,
      &IID_ID3D12PipelineState, (void **)&pso_triangle), pso_triangle,
      "triangle control pipeline");
   pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT;

   /* The pipeline WITHOUT a geometry shader: the control. */
   ID3D12PipelineState *pso_nogs = NULL;
   hr = ID3D12Device_CreateGraphicsPipelineState(
      dev, &pd, &IID_ID3D12PipelineState, (void **)&pso_nogs);
   TEST_OBJECT(hr, pso_nogs, "checked ID3D12Device_CreateGraphicsPipelineState pso_nogs");
   CHECK(hr == S_OK && pso_nogs, "control pipeline (no geometry shader)");

   /* And with one. This used to return E_NOTIMPL by design, because
    * silently dropping the stage draws the pre-GS geometry and looks
    * like a pass.
    */
   pd.GS = (D3D12_SHADER_BYTECODE){.pShaderBytecode = gs,
                                   .BytecodeLength = gs_size};
   ID3D12PipelineState *pso_gs = NULL;
   hr = ID3D12Device_CreateGraphicsPipelineState(
      dev, &pd, &IID_ID3D12PipelineState, (void **)&pso_gs);
   TEST_OBJECT(hr, pso_gs, "checked ID3D12Device_CreateGraphicsPipelineState pso_gs");
   CHECK(hr == S_OK && pso_gs,
         "a geometry pipeline is CREATED, not refused");
   if (!pso_gs)
      goto done;

   /* Opaque GPU-tiled target; CPU access uses the legal subresource API. */
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
      dev, &hp, D3D12_HEAP_FLAG_NONE, &rd,
      D3D12_RESOURCE_STATE_COMMON, NULL, &IID_ID3D12Resource,
      (void **)&rt);
   TEST_OBJECT(hr, rt, "checked ID3D12Device_CreateCommittedResource rt");
   CHECK(hr == S_OK && rt, "create render target");
   if (!rt)
      goto done;

   D3D12_DESCRIPTOR_HEAP_DESC hd = {.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                                    .NumDescriptors = 1};
   ID3D12DescriptorHeap *rtv_heap = NULL;
   TEST_OBJECT(ID3D12Device_CreateDescriptorHeap(dev, &hd, &IID_ID3D12DescriptorHeap,
                                     (void **)&rtv_heap), rtv_heap, "checked ID3D12Device_CreateDescriptorHeap rtv_heap");
   D3D12_CPU_DESCRIPTOR_HANDLE rtv =
      d12_test_descriptor_heap_get_cpu_start(rtv_heap);
   CHECK(rtv.ptr != 0, "valid RTV descriptor handle");
   if (!rtv.ptr) return 1;
   ID3D12Device_CreateRenderTargetView(dev, rt, NULL, rtv);

   D3D12_COMMAND_QUEUE_DESC qd = {.Type = D3D12_COMMAND_LIST_TYPE_DIRECT};
   ID3D12CommandQueue *queue = NULL;
   TEST_OBJECT(ID3D12Device_CreateCommandQueue(dev, &qd, &IID_ID3D12CommandQueue,
                                   (void **)&queue), queue, "checked ID3D12Device_CreateCommandQueue queue");
   ID3D12CommandAllocator *alloc = NULL;
   TEST_OBJECT(ID3D12Device_CreateCommandAllocator(dev, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                       &IID_ID3D12CommandAllocator,
                                       (void **)&alloc), alloc, "checked ID3D12Device_CreateCommandAllocator alloc");
   ID3D12GraphicsCommandList *list = NULL;
   TEST_OBJECT(ID3D12Device_CreateCommandList(dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                  alloc, NULL, &IID_ID3D12GraphicsCommandList,
                                  (void **)&list), list, "checked ID3D12Device_CreateCommandList list");
   ID3D12Fence *fence = NULL;
   TEST_OBJECT(ID3D12Device_CreateFence(dev, 0, D3D12_FENCE_FLAG_NONE, &IID_ID3D12Fence,
                            (void **)&fence), fence, "checked ID3D12Device_CreateFence fence");

   uint8_t *map = malloc(STRIDE * H);
   CHECK(map != NULL, "CPU storage for opaque target");
   if (!map) return 1;

   TEST_CALL(ID3D12GraphicsCommandList_Close(list), "close initial empty list");

   const FLOAT clear_red[4] = {1.0f, 0.0f, 0.0f, 1.0f};
   D3D12_VIEWPORT vp = {.Width = W, .Height = H, .MaxDepth = 1.0f};
   D3D12_RECT sc = {.right = W, .bottom = H};

   /* Draw the same points through both pipelines. */
   static uint8_t control_img[H][W][4];
   unsigned mismatch = 0;
   unsigned covered[3] = {0, 0, 0};
   ID3D12PipelineState *psos[3] = {pso_triangle, pso_nogs, pso_gs};
   /* Case 0 is a triangle with no geometry shader: it isolates the probe
    * vertex shader and the ordinary raster path, so a failure in the
    * point cases can be attributed. Cases 1 and 2 are the same points
    * without and with the geometry shader.
    */
   D3D12_PRIMITIVE_TOPOLOGY topo[3] = {
      D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
      D3D_PRIMITIVE_TOPOLOGY_POINTLIST,
      D3D_PRIMITIVE_TOPOLOGY_POINTLIST,
   };
   unsigned nverts[3] = {3, 4, 4};
   const char *casename[3] = {"triangle, no GS", "points, no GS",
                              "points, GS"};

   for (unsigned i = 0; i < 3; i++) {
      memset(map, 0x55, STRIDE * H);
      TEST_CALL(d12_test_opaque_texture_transfer(rt, map, STRIDE, STRIDE * H, true), "publish original texture poison in COMMON");
      TEST_CALL(ID3D12CommandAllocator_Reset(alloc), "reset fenced allocator");
      TEST_CALL(ID3D12GraphicsCommandList_Reset(list, alloc, NULL), "reset checked graphics list");
      d12_test_transition(list, rt, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET);
      ID3D12GraphicsCommandList_ClearRenderTargetView(list, rtv, clear_red, 0,
                                                      NULL);
      ID3D12GraphicsCommandList_OMSetRenderTargets(list, 1, &rtv, FALSE, NULL);
   ID3D12GraphicsCommandList_RSSetViewports(list, 1, &vp);
      ID3D12GraphicsCommandList_RSSetScissorRects(list, 1, &sc);
      ID3D12GraphicsCommandList_SetGraphicsRootSignature(list, root_sig);
      ID3D12GraphicsCommandList_SetPipelineState(list, psos[i]);
      ID3D12GraphicsCommandList_IASetPrimitiveTopology(list, topo[i]);
      ID3D12GraphicsCommandList_DrawInstanced(list, nverts[i], 1, 0, 0);
      d12_test_transition(list, rt, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COMMON);
      hr = ID3D12GraphicsCommandList_Close(list);
      CHECK(hr == S_OK, i == 2 ? "geometry command list closes cleanly"
                               : "control command list closes cleanly");
      if (hr != S_OK) return 1;

      TEST_CALL(d12_test_submit_closed(dev, list, queue, fence, i + 1), "exact render fence and device status");
      memset(map, 0x33, STRIDE * H);
      TEST_CALL(d12_test_opaque_texture_transfer(rt, map, STRIDE, STRIDE * H, false), "read actual opaque bytes only after fence");
      CHECK(ID3D12Fence_GetCompletedValue(fence) != UINT64_MAX &&
            ID3D12Fence_GetCompletedValue(fence) >= i + 1,
            i == 2 ? "the GPU COMPLETES a geometry draw: two compute "
                     "dispatches and the rewritten draw are a stream it "
                     "accepts"
                   : "the GPU completes the control draw");

      for (unsigned y = 0; y < H; y++) {
         for (unsigned x = 0; x < W; x++) {
            const uint8_t *p = map + y * STRIDE + x * 4;
            if (!(p[0] == 0xff && p[1] == 0x00 && p[2] == 0x00))
               covered[i]++;

            /* Case 1 is the control; case 2 must reproduce it exactly. */
            if (i == 1)
               memcpy(control_img[y][x], p, 4);
            else if (i == 2 && memcmp(control_img[y][x], p, 4))
               mismatch++;
         }
      }
   }

   for (unsigned i = 0; i < 3; i++)
      printf("       %-18s covered %u of %u\n", casename[i], covered[i],
             W * H);

   /* Coverage is DIAGNOSTIC, not a pass criterion, and deliberately so.
    *
    * The geometry shader copies VAR0.xyzw to POSITION. No available
    * vertex shader writes a full float4 VAR0 without reading vertex
    * attributes (see the header comment), so with the pair this suite is
    * run against, VAR0.yzw -- and therefore the position's w -- are
    * UNDEFINED. Any coverage is then legal, including all of it or none
    * of it, and asserting a bound on it would be asserting on undefined
    * behaviour: it would pass or fail for reasons that have nothing to do
    * with the driver.
    *
    * So this suite proves the geometry PATH and says so precisely. What
    * it cannot yet prove is that the emitted geometry lands in the right
    * pixels, the way g4_triangle proves its triangle against a CPU
    * rasterisation. That needs a probe shader this box cannot currently
    * produce -- docs/roadmap.md, the NIR->DXIL generator.
    */
   CHECK(covered[1] != 0 && covered[1] == nverts[1],
         "points rasterise one pixel each without a geometry shader");

   /* THE STRONG ONE. gs_point.dxil is a pass-through: one point in, the
    * same point out. So the geometry case must produce exactly what the
    * control produced -- not a similar count, the SAME PIXELS. Every one
    * of them is compared, so a geometry shader that emits its points in
    * the wrong places fails here even if it emits the right number.
    */
   printf("       geometry vs control: %u pixels differ of %u\n", mismatch,
          W * H);
   CHECK(covered[2] == covered[1] && mismatch == 0,
         "a PASS-THROUGH geometry shader reproduces the control EXACTLY, "
         "pixel for pixel");

   free(map);
   ID3D12Fence_Release(fence);
   ID3D12CommandQueue_Release(queue);
   ID3D12GraphicsCommandList_Release(list);
   ID3D12CommandAllocator_Release(alloc);
   ID3D12DescriptorHeap_Release(rtv_heap);
   ID3D12Resource_Release(rt);

done:
   if (pso_triangle) ID3D12PipelineState_Release(pso_triangle);
   if (pso_gs)
      ID3D12PipelineState_Release(pso_gs);
   if (pso_nogs)
      ID3D12PipelineState_Release(pso_nogs);
   ID3D12RootSignature_Release(root_sig);
   ID3D12Device_Release(dev);

   printf(failures ? "G6 GEOMETRY: %d FAILURES\n" : "G6 GEOMETRY: ALL PASS\n",
          failures);
   free(vs);
   free(ps);
   free(gs);
   return failures ? 1 : 0;
}
