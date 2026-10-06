/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * d3d12agx G4 final acceptance: THE TRIANGLE.
 *
 * A vertex shader and a pixel shader, both authored in HLSL, compiled to
 * DXIL by dxc, handed to the driver through the public D3D12 API,
 * translated by us to NIR, compiled to AGX machine code, and rasterized
 * by the Apple GPU. No SPIR-V, no Vulkan, no external compiler.
 *
 *   VS: float4 main(uint vid : SV_VertexID) : SV_Position
 *       — a full-screen-ish triangle from the vertex index alone.
 *   PS: returns a constant colour.
 *
 * The target is cleared to one colour and the triangle drawn in another,
 * so the test verifies BOTH that the triangle appears (centre pixel is
 * the triangle colour) and that it is bounded (a corner keeps the clear
 * colour). A solid-filled or empty target fails.
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
#define ARRAY_SIZE_LOCAL(x) (sizeof(x) / sizeof((x)[0]))

/* THE ORACLE. tri.dxil declares, per vertex id:
 *
 *    x = (float)(vid ^ 1) * 2 - 1        y = (vid > 1) ? 1 : -1
 *
 * giving clip-space (1,-1), (-1,-1), (5,1) -- a triangle whose third
 * vertex is far off screen to the right, so the visible part is the
 * wedge below the hypotenuse. Nothing here is calibrated against the
 * GPU: it is the shader's own arithmetic, rasterised on the CPU with
 * D3D's documented pixel-centre sampling and top-left fill rule.
 *
 * This replaces a hand-computed "a quarter of the image" expectation
 * that was wrong, and that only ever passed because the translator was
 * emitting integer adds on float bits -- garbage vertices that happened
 * to cover 25%. A model that derives coverage from the shader cannot
 * agree with a broken translator by luck.
 */
static void
rasterise_reference(bool covered[H][W], unsigned vp_height)
{
   const float ndc[3][2] = {{1.0f, -1.0f}, {-1.0f, -1.0f}, {5.0f, 1.0f}};
   float p[3][2];
   for (int i = 0; i < 3; i++) {
      p[i][0] = (ndc[i][0] * 0.5f + 0.5f) * (float)W;
      p[i][1] = (0.5f - ndc[i][1] * 0.5f) * (float)vp_height;
   }

#define EDGE(a, b, px, py)                                                   \
   ((p[b][0] - p[a][0]) * ((py) - p[a][1]) -                                 \
    (p[b][1] - p[a][1]) * ((px) - p[a][0]))

   /* Interior is EDGE > 0 for this winding; a zero (pixel centre exactly
    * on an edge) is covered only for top and left edges.
    */
   static const int e0[3] = {0, 1, 2}, e1[3] = {1, 2, 0};
   bool topleft[3];
   for (int i = 0; i < 3; i++) {
      int a = e0[i], b = e1[i];
      topleft[i] = (p[b][1] == p[a][1]) ? (p[b][0] > p[a][0])
                                        : (p[b][1] < p[a][1]);
   }

   for (unsigned y = 0; y < H; y++) {
      for (unsigned x = 0; x < W; x++) {
         float cx = (float)x + 0.5f, cy = (float)y + 0.5f;
         bool in = true;
         for (int i = 0; i < 3; i++) {
            float e = EDGE(e0[i], e1[i], cx, cy);
            if (e < 0 || (e == 0 && !topleft[i])) {
               in = false;
               break;
            }
         }
         covered[y][x] = in;
      }
   }
#undef EDGE
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

/* Build a real root signature blob. CreateRootSignature parses it now, so
 * the four bytes of shader bytecode that used to be passed here are no
 * longer accepted -- correctly, since they never described anything.
 */
static ID3D12RootSignature *
make_root_sig(ID3D12Device *dev, const D3D12_ROOT_PARAMETER *params,
              UINT count)
{
   D3D12_ROOT_SIGNATURE_DESC rsd = {
      .NumParameters = count,
      .pParameters = params,
   };

   ID3DBlob *blob = NULL;
   HRESULT hr = D3D12SerializeRootSignature(&rsd,
                                            D3D_ROOT_SIGNATURE_VERSION_1_0,
                                            &blob, NULL);
   if (hr != S_OK || !blob) {
      if (blob)
         ID3D10Blob_Release(blob);
      return NULL;
   }

   ID3D12RootSignature *rs = NULL;
   hr = ID3D12Device_CreateRootSignature(dev, 0,
                                         ID3D10Blob_GetBufferPointer(blob),
                                         ID3D10Blob_GetBufferSize(blob),
                                         &IID_ID3D12RootSignature, (void **)&rs);
   ID3D10Blob_Release(blob);
   if (hr != S_OK || !rs) {
      if (rs)
         ID3D12RootSignature_Release(rs);
      return NULL;
   }

   return rs;
}

/* Execute only a successfully closed list; grade pixels only after the exact
 * signal is observed. UINT64_MAX is the device-removal sentinel, not progress. */
static HRESULT
submit_and_wait(ID3D12Device *dev, ID3D12GraphicsCommandList *list,
                ID3D12CommandQueue *queue, ID3D12Fence *fence, UINT64 value)
{
   HRESULT hr = ID3D12GraphicsCommandList_Close(list);
   if (hr != S_OK)
      return hr;
   ID3D12CommandList *lists[] = {(ID3D12CommandList *)list};
   ID3D12CommandQueue_ExecuteCommandLists(queue, 1, lists);
   hr = ID3D12Device_GetDeviceRemovedReason(dev);
   if (hr != S_OK)
      return hr;
   hr = ID3D12CommandQueue_Signal(queue, fence, value);
   if (hr != S_OK)
      return hr;
   hr = ID3D12Fence_SetEventOnCompletion(fence, value, NULL);
   if (hr != S_OK)
      return hr;
   UINT64 completed = ID3D12Fence_GetCompletedValue(fence);
   printf("       fence requested=%llu completed=%llu\n",
          (unsigned long long)value, (unsigned long long)completed);
   if (completed == UINT64_MAX || completed < value)
      return E_FAIL;
   return ID3D12Device_GetDeviceRemovedReason(dev);
}

/* Opaque CPU-visible UMA textures expose no linear pointer. Transfer real
 * bytes through the D3D12 subresource API only while the texture is COMMON,
 * and after the caller has observed its render fence. */
static HRESULT
cpu_texture_transfer(ID3D12Resource *resource, void *bytes,
                     UINT row_pitch, UINT size, bool write)
{
   HRESULT hr = ID3D12Resource_Map(resource, 0, NULL, NULL);
   if (hr != S_OK)
      return hr;
   if (write)
      hr = ID3D12Resource_WriteToSubresource(resource, 0, NULL, bytes,
                                            row_pitch, size);
   else
      hr = ID3D12Resource_ReadFromSubresource(resource, bytes, row_pitch,
                                              size, 0, NULL);
   ID3D12Resource_Unmap(resource, 0, NULL);
   return hr;
}

static void
transition(ID3D12GraphicsCommandList *list, ID3D12Resource *resource,
           D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after)
{
   D3D12_RESOURCE_BARRIER barrier = {
      .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
      .Transition = {.pResource = resource,
                     .Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                     .StateBefore = before, .StateAfter = after},
   };
   ID3D12GraphicsCommandList_ResourceBarrier(list, 1, &barrier);
}

int
main(int argc, char **argv)
{
   if (argc < 3) {
      fprintf(stderr, "usage: %s tri.dxil solid.dxil\n", argv[0]);
      return 2;
   }

   const char *blocked_stage = NULL;
   HRESULT blocked_hr = S_OK;
   ID3D12Device *dev = NULL;
   ID3D12RootSignature *root_sig = NULL;
   ID3D12PipelineState *pso = NULL;
   ID3D12Resource *rt = NULL, *dstbuf = NULL, *bgra = NULL;
   ID3D12DescriptorHeap *rtv_heap = NULL, *h2 = NULL;
   ID3D12CommandAllocator *alloc = NULL;
   ID3D12GraphicsCommandList *list = NULL;
   ID3D12CommandQueue *queue = NULL;
   ID3D12Fence *fence = NULL;
   uint8_t *map = NULL, *rbmap = NULL, *m2 = NULL;

#define REQUIRE(cond, stage, result)                                         \
   do {                                                                      \
      CHECK(cond, stage);                                                     \
      if (!(cond)) {                                                         \
         blocked_stage = stage;                                              \
         blocked_hr = FAILED(result) ? (result) : E_FAIL;                      \
         goto cleanup;                                                       \
      }                                                                      \
   } while (0)
#define REQUIRE_HR(call, stage)                                              \
   do {                                                                      \
      hr = (call);                                                           \
      REQUIRE(hr == S_OK, stage, hr);                                         \
   } while (0)

   size_t vs_size = 0, ps_size = 0;
   void *vs = slurp(argv[1], &vs_size);
   void *ps = slurp(argv[2], &ps_size);
   if (!vs || !ps) {
      fprintf(stderr, "cannot read shaders\n");
      free(vs);
      free(ps);
      return 2;
   }

   HRESULT hr = D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0,
                                  &IID_ID3D12Device, (void **)&dev);
   REQUIRE(hr == S_OK && dev, "D3D12CreateDevice", hr);

   root_sig = make_root_sig(dev, NULL, 0);
   REQUIRE(root_sig != NULL, "CreateRootSignature", E_FAIL);

   /* Compile BOTH shaders from DXIL to AGX inside the driver. */
   D3D12_GRAPHICS_PIPELINE_STATE_DESC pd = {
      .pRootSignature = root_sig,
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
   pd.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
   hr = ID3D12Device_CreateGraphicsPipelineState(dev, &pd,
                                                 &IID_ID3D12PipelineState,
                                                 (void **)&pso);
   REQUIRE(hr == S_OK && pso,
           "CreateGraphicsPipelineState compiles VS+PS from DXIL to AGX", hr);

   /* Legal opaque GPU-tiled UMA resource, with explicit CPU subresource
    * transfer rather than a DEFAULT heap mapping. */
   D3D12_HEAP_PROPERTIES heap = {
      .Type = D3D12_HEAP_TYPE_CUSTOM,
      .CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
      .MemoryPoolPreference = D3D12_MEMORY_POOL_L0,
      .CreationNodeMask = 1, .VisibleNodeMask = 1,
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
      dev, &heap, D3D12_HEAP_FLAG_NONE, &td,
      D3D12_RESOURCE_STATE_COMMON, NULL, &IID_ID3D12Resource,
      (void **)&rt);
   REQUIRE(hr == S_OK && rt, "create 256x256 render target", hr);

   D3D12_DESCRIPTOR_HEAP_DESC hd = {
      .Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
      .NumDescriptors = 1,
   };
   hr = ID3D12Device_CreateDescriptorHeap(dev, &hd, &IID_ID3D12DescriptorHeap,
                                          (void **)&rtv_heap);
   REQUIRE(hr == S_OK && rtv_heap, "create RTV heap", hr);
   D3D12_CPU_DESCRIPTOR_HANDLE rtv =
      d12_test_descriptor_heap_get_cpu_start(rtv_heap);
   ID3D12Device_CreateRenderTargetView(dev, rt, NULL, rtv);

   map = malloc(STRIDE * H);
   REQUIRE(map, "CPU storage for opaque render target", E_OUTOFMEMORY);
   memset(map, 0x55, STRIDE * H);
   REQUIRE_HR(cpu_texture_transfer(rt, map, STRIDE, STRIDE * H, true),
              "write render-target poison through legal CPU API");

   /* Record: clear to red, then draw a green triangle. */
   hr = ID3D12Device_CreateCommandAllocator(dev, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                       &IID_ID3D12CommandAllocator,
                                       (void **)&alloc);
   REQUIRE(hr == S_OK && alloc, "CreateCommandAllocator", hr);
   hr = ID3D12Device_CreateCommandList(dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                  alloc, NULL,
                                  &IID_ID3D12GraphicsCommandList,
                                  (void **)&list);
   REQUIRE(hr == S_OK && list, "CreateCommandList", hr);

   const FLOAT clear_red[4] = {1.0f, 0.0f, 0.0f, 1.0f};
   transition(list, rt, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET);
   ID3D12GraphicsCommandList_ClearRenderTargetView(list, rtv, clear_red, 0,
                                                   NULL);
   ID3D12GraphicsCommandList_OMSetRenderTargets(list, 1, &rtv, FALSE, NULL);

   D3D12_VIEWPORT vp = {.Width = W, .Height = H, .MaxDepth = 1.0f};
   ID3D12GraphicsCommandList_RSSetViewports(list, 1, &vp);
   D3D12_RECT sc = {.right = W, .bottom = H};
   ID3D12GraphicsCommandList_RSSetScissorRects(list, 1, &sc);

   ID3D12GraphicsCommandList_SetGraphicsRootSignature(list, root_sig);
   ID3D12GraphicsCommandList_SetPipelineState(list, pso);
   ID3D12GraphicsCommandList_IASetPrimitiveTopology(
      list, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
   ID3D12GraphicsCommandList_DrawInstanced(list, 3, 1, 0, 0);

   D3D12_COMMAND_QUEUE_DESC qd = {.Type = D3D12_COMMAND_LIST_TYPE_DIRECT};
   hr = ID3D12Device_CreateCommandQueue(dev, &qd, &IID_ID3D12CommandQueue,
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

   transition(list, rt, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COMMON);
   REQUIRE_HR(submit_and_wait(dev, list, queue, fence, 1),
              "GPU render complete (fence 1)");
   memset(map, 0x33, STRIDE * H);
   REQUIRE_HR(cpu_texture_transfer(rt, map, STRIDE, STRIDE * H, false),
              "read actual fenced opaque render-target bytes");

#define PIXEL(x, y) (map + (y) * STRIDE + (x) * 4)

   /*
    * tri.hlsl covers NDC (-1,+1) (0,+1) (-1,-1): a right triangle whose
    * right angle sits at the TOP-LEFT of the image, because D3D12 clip
    * space has +y up while the framebuffer's first row is the top.
    *
    * These sample points were originally the image centre and a corner,
    * chosen when the viewport transform was missing its Y negation and the
    * rendered triangle was mirrored. They are now taken from the geometry
    * the shader actually declares: (30,30) is well inside, (230,230) is
    * far outside on the other side of the hypotenuse.
    */
   static bool ref[H][W];
   rasterise_reference(ref, H);

   /* Pick the sample points out of the model rather than hardcoding them,
    * so they follow the geometry instead of a memory of it.
    */
   unsigned ix = 0, iy = 0, ox = 0, oy = 0;
   for (unsigned y = 0; y < H; y++)
      for (unsigned x = 0; x < W; x++)
         (ref[y][x] ? (ix = x, iy = y) : (ox = x, oy = y));

   const uint8_t *inside = PIXEL(ix, iy);
   const uint8_t *outside = PIXEL(ox, oy);

   printf("       inside  (%u,%u) = %02x %02x %02x %02x\n", ix, iy, inside[0],
          inside[1], inside[2], inside[3]);
   printf("       outside (%u,%u) = %02x %02x %02x %02x\n", ox, oy, outside[0],
          outside[1], outside[2], outside[3]);

   /* The clear must have happened everywhere. */
   bool cleared =
      outside[0] == 0xff && outside[1] == 0x00 && outside[2] == 0x00;
   CHECK(cleared, "clear survives outside the triangle (red)");

   /* The triangle must cover the inside point with the pixel shader's
    * colour (0.25, 0.5, 0.75, 1.0 from solid.hlsl).
    */
   bool drew = inside[0] == 0x40 && inside[1] == 0x80 && inside[2] == 0xbf;
   CHECK(drew, "triangle covers its interior with the pixel shader colour");

   /* EVERY pixel against the model, not a sample of them. A spot check
    * passes for a triangle that is merely in the right neighbourhood;
    * this one fails on a single mis-rasterised pixel, and it fails
    * loudly enough to say where.
    */
   unsigned covered = 0, mismatch = 0, first_x = 0, first_y = 0;
   for (unsigned y = 0; y < H; y++) {
      for (unsigned x = 0; x < W; x++) {
         const uint8_t *p = PIXEL(x, y);
         bool is_tri = p[0] == 0x40 && p[1] == 0x80 && p[2] == 0xbf;
         bool is_clear = p[0] == 0xff && p[1] == 0x00 && p[2] == 0x00;
         if (!is_clear)
            covered++;
         if (is_tri != ref[y][x] || !(is_tri || is_clear)) {
            if (!mismatch++)
               first_x = x, first_y = y;
         }
      }
   }

   unsigned expect = 0;
   for (unsigned y = 0; y < H; y++)
      for (unsigned x = 0; x < W; x++)
         expect += ref[y][x];

   printf("       covered %u of %u pixels (%.1f%%), model says %u\n", covered,
          W * H, 100.0 * covered / (W * H), expect);
   if (mismatch)
      printf("       first mismatch at (%u,%u)\n", first_x, first_y);
   CHECK(mismatch == 0,
         "every pixel matches a CPU rasterisation of the shader's own "
         "vertices (D3D top-left rule)");

   /* RSSetViewports used to be recorded and never read. Re-render into a
    * half-height viewport: a viewport that is honoured must shrink the
    * covered area, and must leave the bottom of the target cleared.
    */
   {
      memset(map, 0x55, STRIDE * H);
      REQUIRE_HR(cpu_texture_transfer(rt, map, STRIDE, STRIDE * H, true),
                 "poison idle opaque target before viewport reuse");
      REQUIRE_HR(ID3D12CommandAllocator_Reset(alloc), "Reset fenced allocator");
      REQUIRE_HR(ID3D12GraphicsCommandList_Reset(list, alloc, NULL),
                 "Reset command list");
      transition(list, rt, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET);
      ID3D12GraphicsCommandList_OMSetRenderTargets(list, 1, &rtv, FALSE, NULL);
      ID3D12GraphicsCommandList_ClearRenderTargetView(list, rtv, clear_red, 0,
                                                      NULL);
      D3D12_VIEWPORT half = {.Width = W, .Height = H / 2, .MaxDepth = 1.0f};
      ID3D12GraphicsCommandList_RSSetViewports(list, 1, &half);
      ID3D12GraphicsCommandList_RSSetScissorRects(list, 1, &sc);
      ID3D12GraphicsCommandList_SetGraphicsRootSignature(list, root_sig);
      ID3D12GraphicsCommandList_SetPipelineState(list, pso);
      ID3D12GraphicsCommandList_IASetPrimitiveTopology(
         list, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
      ID3D12GraphicsCommandList_DrawInstanced(list, 3, 1, 0, 0);
      transition(list, rt, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COMMON);
      REQUIRE_HR(submit_and_wait(dev, list, queue, fence, 2),
                 "GPU render/copy complete (fence 2)");
      memset(map, 0x33, STRIDE * H);
      REQUIRE_HR(cpu_texture_transfer(rt, map, STRIDE, STRIDE * H, false),
                 "read actual half-height viewport pixels after fence2");

      unsigned half_cov = 0;
      for (unsigned y = 0; y < H; y++) {
         for (unsigned x = 0; x < W; x++) {
            const uint8_t *p = PIXEL(x, y);
            if (!(p[0] == 0xff && p[1] == 0x00 && p[2] == 0x00))
               half_cov++;
         }
      }
      printf("       half-height viewport covered %u (full was %u)\n",
             half_cov, covered);
      CHECK(half_cov > 0 && half_cov < covered,
            "RSSetViewports is honoured: half viewport shrinks coverage");

      /* A point must be inside at full height and outside at half height
       * to discriminate; one the triangle never reaches would pass no
       * matter what the viewport did. Take it from the two models.
       */
      static bool half_ref[H][W];
      rasterise_reference(half_ref, H / 2);
      unsigned half_mismatch = 0;
      for (unsigned y = 0; y < H; y++) for (unsigned x = 0; x < W; x++) {
         const uint8_t tc[4] = {0x40,0x80,0xbf,0xff}, cc[4] = {0xff,0,0,0xff};
         half_mismatch += memcmp(PIXEL(x, y), half_ref[y][x] ? tc : cc, 4) != 0;
      }
      CHECK(!half_mismatch, "every half-height pixel matches the original CPU model");
      unsigned lx = 0, ly = 0;
      bool found = false;
      for (unsigned y = 0; y < H && !found; y++)
         for (unsigned x = 0; x < W && !found; x++)
            if (ref[y][x] && !half_ref[y][x])
               lx = x, ly = y, found = true;

      const uint8_t *low = PIXEL(lx, ly);
      printf("       (%u,%u) is inside at full height, outside at half\n", lx,
             ly);
      CHECK(found && low[0] == 0xff && low[1] == 0x00 && low[2] == 0x00,
            "area below the viewport keeps the clear colour");
   }

   /* THE INDEPENDENT CHECK. Everything above reads the opaque texture
    * through the CPU subresource API. This instead pulls
    * the pixels out the way any real application (and every pixel assertion
    * in the vkd3d suite) does: GetCopyableFootprints to learn the layout,
    * then CopyTextureRegion into a plain buffer.
    */
   {
      D3D12_RESOURCE_DESC rtd = d12_test_resource_get_desc(rt);
      D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp = {0};
      UINT rows = 0;
      UINT64 row_bytes = 0, total = 0;
      ID3D12Device_GetCopyableFootprints(dev, &rtd, 0, 1, 512, &fp, &rows,
                                         &row_bytes, &total);

      REQUIRE(fp.Footprint.Width == W && fp.Footprint.Height == H &&
                 fp.Footprint.Depth == 1 && fp.Offset == 512 &&
                 fp.Footprint.Format == DXGI_FORMAT_R8G8B8A8_UNORM &&
                 rows == H && row_bytes == (UINT64)W * 4,
              "GetCopyableFootprints describes the render target", E_FAIL);
      REQUIRE(fp.Footprint.RowPitch % 256 == 0 &&
                 fp.Footprint.RowPitch >= W * 4 &&
                 total >= (UINT64)fp.Footprint.RowPitch * H,
              "footprint row pitch honours the 256-byte D3D12 alignment", E_FAIL);

      UINT64 copy_size = fp.Offset + fp.Footprint.RowPitch * (UINT64)H + 512;
      D3D12_HEAP_PROPERTIES rb = {.Type = D3D12_HEAP_TYPE_READBACK};
      D3D12_RESOURCE_DESC bd = {
         .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
         .Width = copy_size,
         .Height = 1,
         .DepthOrArraySize = 1,
         .MipLevels = 1,
         .SampleDesc = {.Count = 1},
         .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
      };
      hr = ID3D12Device_CreateCommittedResource(
         dev, &rb, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST,
         NULL, &IID_ID3D12Resource, (void **)&dstbuf);
      REQUIRE(hr == S_OK && dstbuf, "create readback buffer from the footprint", hr);

      const D3D12_RANGE empty = {0, 0};
      hr = ID3D12Resource_Map(dstbuf, 0, &empty, (void **)&rbmap);
      REQUIRE(hr == S_OK && rbmap, "map readback buffer", hr);
      memset(rbmap, 0x7b, (size_t)copy_size);
      memset(rbmap + fp.Offset, 0x33, (size_t)(fp.Footprint.RowPitch * (UINT64)H));
      ID3D12Resource_Unmap(dstbuf, 0, NULL);rbmap = NULL;

      /* Re-render, then copy the texture out through the API. */
      REQUIRE_HR(ID3D12CommandAllocator_Reset(alloc), "Reset fenced allocator");
      REQUIRE_HR(ID3D12GraphicsCommandList_Reset(list, alloc, NULL),
                 "Reset command list");
      transition(list, rt, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET);
      ID3D12GraphicsCommandList_OMSetRenderTargets(list, 1, &rtv, FALSE, NULL);
      ID3D12GraphicsCommandList_ClearRenderTargetView(list, rtv, clear_red, 0,
                                                      NULL);
      ID3D12GraphicsCommandList_RSSetViewports(list, 1, &vp);
      ID3D12GraphicsCommandList_RSSetScissorRects(list, 1, &sc);
      ID3D12GraphicsCommandList_SetGraphicsRootSignature(list, root_sig);
      ID3D12GraphicsCommandList_SetPipelineState(list, pso);
      ID3D12GraphicsCommandList_IASetPrimitiveTopology(
         list, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
      ID3D12GraphicsCommandList_DrawInstanced(list, 3, 1, 0, 0);

      D3D12_TEXTURE_COPY_LOCATION dloc = {
         .pResource = dstbuf,
         .Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
         .PlacedFootprint = fp,
      };
      D3D12_TEXTURE_COPY_LOCATION sloc = {
         .pResource = rt,
         .Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
         .SubresourceIndex = 0,
      };
      transition(list, rt, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
      ID3D12GraphicsCommandList_CopyTextureRegion(list, &dloc, 0, 0, 0, &sloc,
                                                  NULL);
      transition(list, rt, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
      REQUIRE_HR(submit_and_wait(dev, list, queue, fence, 3),
                 "GPU render/copy complete (fence 3)");
      const D3D12_RANGE read_range = {0, (SIZE_T)copy_size};
      REQUIRE_HR(ID3D12Resource_Map(dstbuf, 0, &read_range, (void **)&rbmap),
                 "map actual GPU copy only after fence3");
      REQUIRE(rbmap, "GPU-copy readback pointer", E_FAIL);

      /* Same two sample points as above, read out through the API. */
      const uint8_t *c = rbmap + fp.Offset + iy * fp.Footprint.RowPitch + ix * 4;
      const uint8_t *k = rbmap + fp.Offset + oy * fp.Footprint.RowPitch + ox * 4;
      printf("       via CopyTextureRegion: inside %02x%02x%02x  "
             "outside %02x%02x%02x\n",
             c[0], c[1], c[2], k[0], k[1], k[2]);

      CHECK(c[0] == 0x40 && c[1] == 0x80 && c[2] == 0xbf,
            "triangle read back through CopyTextureRegion, not our mapping");
      CHECK(k[0] == 0xff && k[1] == 0x00 && k[2] == 0x00,
            "clear colour read back through CopyTextureRegion");

      unsigned copy_mismatch = 0, canary_mismatch = 0;
      for (unsigned y = 0; y < H; y++) {
         for (unsigned x = 0; x < W; x++) {
            const uint8_t tc[4] = {0x40,0x80,0xbf,0xff}, cc[4] = {0xff,0,0,0xff};
            const uint8_t *pixel = rbmap + fp.Offset + y * fp.Footprint.RowPitch + x * 4;
            copy_mismatch += memcmp(pixel, ref[y][x] ? tc : cc, 4) != 0;
         }
         for (UINT x = W * 4; x < fp.Footprint.RowPitch; x++)
            canary_mismatch += rbmap[fp.Offset + y * fp.Footprint.RowPitch + x] != 0x33;
      }
      for (UINT64 i = 0; i < fp.Offset; i++) canary_mismatch += rbmap[i] != 0x7b;
      for (UINT64 i = fp.Offset + fp.Footprint.RowPitch * (UINT64)H; i < copy_size; i++)
         canary_mismatch += rbmap[i] != 0x7b;
      printf("       GPU readback all-pixel mismatches=%u canary mismatches=%u\n", copy_mismatch, canary_mismatch);
      CHECK(!copy_mismatch && !canary_mismatch, "every GPU-copy pixel matches original model; canaries intact");
      ID3D12Resource_Unmap(dstbuf, 0, &empty);
      rbmap = NULL;
      ID3D12Resource_Release(dstbuf);
      dstbuf = NULL;
   }

   /* A SECOND FORMAT. Everything above is RGBA8, which the driver used to
    * hardcode in eight places. BGRA8 is the sharpest check available: it
    * has the same size and layout but the opposite channel order, so a
    * missing swizzle shows up as red and blue swapped rather than as a
    * crash.
    */
   {
      D3D12_HEAP_PROPERTIES dh = heap;
      D3D12_RESOURCE_DESC bd2 = {
         .Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
         .Width = 64,
         .Height = 64,
         .DepthOrArraySize = 1,
         .MipLevels = 1,
         .Format = DXGI_FORMAT_B8G8R8A8_UNORM,
         .SampleDesc = {.Count = 1},
         .Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN,
         .Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET |
                  D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS,
      };
      hr = ID3D12Device_CreateCommittedResource(
         dev, &dh, D3D12_HEAP_FLAG_NONE, &bd2,
         D3D12_RESOURCE_STATE_COMMON, NULL, &IID_ID3D12Resource,
         (void **)&bgra);
      REQUIRE(hr == S_OK && bgra, "create a BGRA8 render target", hr);

      D3D12_DESCRIPTOR_HEAP_DESC hd2 = {
         .Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV, .NumDescriptors = 1};
      hr = ID3D12Device_CreateDescriptorHeap(dev, &hd2, &IID_ID3D12DescriptorHeap,
                                             (void **)&h2);
      REQUIRE(hr == S_OK && h2, "create BGRA RTV heap", hr);
      D3D12_CPU_DESCRIPTOR_HANDLE rtv2 =
         d12_test_descriptor_heap_get_cpu_start(h2);
      ID3D12Device_CreateRenderTargetView(dev, bgra, NULL, rtv2);

      m2 = malloc(64 * 64 * 4);
      REQUIRE(m2, "CPU storage for opaque BGRA target", E_OUTOFMEMORY);
      memset(m2, 0x11, 64 * 64 * 4);
      REQUIRE_HR(cpu_texture_transfer(bgra, m2, 64 * 4, 64 * 64 * 4, true),
                 "poison opaque BGRA target through legal CPU API");

      /* Clear to pure red. In BGRA memory order that must land as
       * 00 00 ff ff, NOT ff 00 00 ff.
       */
      REQUIRE_HR(ID3D12CommandAllocator_Reset(alloc), "Reset fenced allocator");
      REQUIRE_HR(ID3D12GraphicsCommandList_Reset(list, alloc, NULL),
                 "Reset command list");
      transition(list, bgra, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET);
      ID3D12GraphicsCommandList_ClearRenderTargetView(list, rtv2, clear_red,
                                                      0, NULL);
      transition(list, bgra, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COMMON);
      REQUIRE_HR(submit_and_wait(dev, list, queue, fence, 4),
                 "GPU render/copy complete (fence 4)");
      memset(m2, 0x33, 64 * 64 * 4);
      REQUIRE_HR(cpu_texture_transfer(bgra, m2, 64 * 4, 64 * 64 * 4, false),
                 "read actual BGRA clear only after fence4");

      printf("       BGRA8 cleared to red -> memory %02x %02x %02x %02x\n",
             m2[0], m2[1], m2[2], m2[3]);
      CHECK(m2[0] == 0x00 && m2[1] == 0x00 && m2[2] == 0xff && m2[3] == 0xff,
            "BGRA8 channel order is honoured (red is not swapped to blue)");

      unsigned bgra_mismatch = 0;
      const uint8_t expected_bgra_red[4] = {0,0,0xff,0xff};
      for (unsigned i = 0; i < 64 * 64; i++)
         bgra_mismatch += memcmp(m2 + i * 4, expected_bgra_red, 4) != 0;
      CHECK(!bgra_mismatch, "every BGRA8 pixel retains red with correct channel order");
      free(m2);
      m2 = NULL;
      ID3D12DescriptorHeap_Release(h2);
      h2 = NULL;
      ID3D12Resource_Release(bgra);
      bgra = NULL;
   }

   /* And a format the hardware cannot render must be REFUSED, not
    * accepted into producing wrong pixels.
    */
   {
      D3D12_HEAP_PROPERTIES dh = {.Type = D3D12_HEAP_TYPE_DEFAULT};
      D3D12_RESOURCE_DESC bad = {
         .Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
         .Width = 16, .Height = 16, .DepthOrArraySize = 1, .MipLevels = 1,
         .Format = DXGI_FORMAT_BC1_UNORM, /* block compressed: not an RT */
         .SampleDesc = {.Count = 1},
         .Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
      };
      ID3D12Resource *r = NULL;
      hr = ID3D12Device_CreateCommittedResource(
         dev, &dh, D3D12_HEAP_FLAG_NONE, &bad,
         D3D12_RESOURCE_STATE_RENDER_TARGET, NULL, &IID_ID3D12Resource,
         (void **)&r);
      CHECK(FAILED(hr) && !r,
            "an unrenderable format is refused, not silently accepted");
      if (r)
         ID3D12Resource_Release(r);
   }

   /* A stage with no hardware equivalent must be refused too. AGX runs
    * geometry and tessellation as compute, and that runtime does not
    * exist yet -- but the descriptions were being dropped on the floor,
    * so a pipeline with a geometry shader quietly rasterised its pre-GS
    * geometry. That draws *something*, and something reads as a pass.
    *
    * The bytecode here is a vertex shader; the refusal happens on the
    * stage being bound at all, before anything parses it.
    */
   {
      /* Geometry and tessellation now DISPATCH (g6_geometry,
       * g7_tessellation), so binding them is no longer a refusal. What is
       * still worth asserting is that an incomplete tessellation pipeline
       * -- a hull shader with no domain shader -- does not silently draw
       * without tessellating.
       */
      const struct {
         const char *name;
         size_t offset;
      } stages[] = {
         {"hull without a domain", offsetof(D3D12_GRAPHICS_PIPELINE_STATE_DESC, HS)},
      };

      for (unsigned i = 0; i < ARRAY_SIZE_LOCAL(stages); i++) {
         D3D12_GRAPHICS_PIPELINE_STATE_DESC pd = {
            .pRootSignature = root_sig,
            .VS = {.pShaderBytecode = vs, .BytecodeLength = vs_size},
            .PS = {.pShaderBytecode = ps, .BytecodeLength = ps_size},
            .NumRenderTargets = 1,
            .RTVFormats = {DXGI_FORMAT_R8G8B8A8_UNORM},
            .RasterizerState = {.CullMode = D3D12_CULL_MODE_NONE},
            .SampleDesc = {.Count = 1},
            .SampleMask = ~0u,
         };
         *(D3D12_SHADER_BYTECODE *)((char *)&pd + stages[i].offset) =
            (D3D12_SHADER_BYTECODE){.pShaderBytecode = vs,
                                    .BytecodeLength = vs_size};

         ID3D12PipelineState *bad_pso = NULL;
         HRESULT ghr = ID3D12Device_CreateGraphicsPipelineState(
            dev, &pd, &IID_ID3D12PipelineState, (void **)&bad_pso);

         char what[128];
         snprintf(what, sizeof(what),
                  "a pipeline with a %s shader is refused, not silently "
                  "drawn without tessellating",
                  stages[i].name);
         CHECK(FAILED(ghr) && !bad_pso, what);
         if (bad_pso)
            ID3D12PipelineState_Release(bad_pso);
      }
   }

cleanup:
   if (blocked_stage)
      printf("G4 TRIANGLE BLOCKED: stage=%s HRESULT=0x%08x (not a hardware pass)\n",
             blocked_stage, (unsigned)blocked_hr);
   free(map);
   if (rbmap) {
      const D3D12_RANGE empty = {0, 0};
      ID3D12Resource_Unmap(dstbuf, 0, &empty);
   }
   free(m2);
   if (dstbuf)
      ID3D12Resource_Release(dstbuf);
   if (bgra)
      ID3D12Resource_Release(bgra);
   if (h2)
      ID3D12DescriptorHeap_Release(h2);
   if (fence)
      ID3D12Fence_Release(fence);
   if (queue)
      ID3D12CommandQueue_Release(queue);
   if (list)
      ID3D12GraphicsCommandList_Release(list);
   if (alloc)
      ID3D12CommandAllocator_Release(alloc);
   if (rtv_heap)
      ID3D12DescriptorHeap_Release(rtv_heap);
   if (rt)
      ID3D12Resource_Release(rt);
   if (pso)
      ID3D12PipelineState_Release(pso);
   if (root_sig)
      ID3D12RootSignature_Release(root_sig);
   if (dev)
      ID3D12Device_Release(dev);

   printf(failures ? "G4 TRIANGLE: %d FAILURES\n" : "G4 TRIANGLE: ALL PASS\n",
          failures);
   free(vs);
   free(ps);
   return failures || blocked_stage;
}
