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
   if (FAILED(D3D12SerializeRootSignature(&rsd,
                                          D3D_ROOT_SIGNATURE_VERSION_1_0,
                                          &blob, NULL)))
      return NULL;

   ID3D12RootSignature *rs = NULL;
   ID3D12Device_CreateRootSignature(dev, 0, ID3D10Blob_GetBufferPointer(blob),
                                    ID3D10Blob_GetBufferSize(blob),
                                    &IID_ID3D12RootSignature, (void **)&rs);
   ID3D10Blob_Release(blob);
   return rs;
}

int
main(int argc, char **argv)
{
   if (argc < 3) {
      fprintf(stderr, "usage: %s tri.dxil solid.dxil\n", argv[0]);
      return 2;
   }

   size_t vs_size, ps_size;
   void *vs = slurp(argv[1], &vs_size);
   void *ps = slurp(argv[2], &ps_size);
   if (!vs || !ps) {
      fprintf(stderr, "cannot read shaders\n");
      return 2;
   }

   ID3D12Device *dev = NULL;
   HRESULT hr = D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0,
                                  &IID_ID3D12Device, (void **)&dev);
   CHECK(hr == S_OK && dev, "D3D12CreateDevice");
   if (FAILED(hr) || !dev)
      return 1;

   ID3D12RootSignature *root_sig = NULL;
   root_sig = make_root_sig(dev, NULL, 0);

   /* Compile BOTH shaders from DXIL to AGX inside the driver. */
   D3D12_GRAPHICS_PIPELINE_STATE_DESC pd = {
      .pRootSignature = root_sig,
      .VS = {.pShaderBytecode = vs, .BytecodeLength = vs_size},
      .PS = {.pShaderBytecode = ps, .BytecodeLength = ps_size},
      .NumRenderTargets = 1,
      .RTVFormats = {DXGI_FORMAT_R8G8B8A8_UNORM},
      .SampleDesc = {.Count = 1},
      .SampleMask = ~0u,
   };
   ID3D12PipelineState *pso = NULL;
   hr = ID3D12Device_CreateGraphicsPipelineState(dev, &pd,
                                                 &IID_ID3D12PipelineState,
                                                 (void **)&pso);
   CHECK(hr == S_OK && pso,
         "CreateGraphicsPipelineState compiles VS+PS from DXIL to AGX");
   if (FAILED(hr) || !pso)
      return 1;

   /* Render target */
   D3D12_HEAP_PROPERTIES heap = {.Type = D3D12_HEAP_TYPE_DEFAULT};
   D3D12_RESOURCE_DESC td = {
      .Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
      .Width = W,
      .Height = H,
      .DepthOrArraySize = 1,
      .MipLevels = 1,
      .Format = DXGI_FORMAT_R8G8B8A8_UNORM,
      .SampleDesc = {.Count = 1},
      .Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
   };
   ID3D12Resource *rt = NULL;
   hr = ID3D12Device_CreateCommittedResource(
      dev, &heap, D3D12_HEAP_FLAG_NONE, &td,
      D3D12_RESOURCE_STATE_RENDER_TARGET, NULL, &IID_ID3D12Resource,
      (void **)&rt);
   CHECK(hr == S_OK && rt, "create 256x256 render target");

   D3D12_DESCRIPTOR_HEAP_DESC hd = {
      .Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
      .NumDescriptors = 1,
   };
   ID3D12DescriptorHeap *rtv_heap = NULL;
   ID3D12Device_CreateDescriptorHeap(dev, &hd, &IID_ID3D12DescriptorHeap,
                                     (void **)&rtv_heap);
   D3D12_CPU_DESCRIPTOR_HANDLE rtv =
      d12_test_descriptor_heap_get_cpu_start(rtv_heap);
   ID3D12Device_CreateRenderTargetView(dev, rt, NULL, rtv);

   uint8_t *map = NULL;
   ID3D12Resource_Map(rt, 0, NULL, (void **)&map);
   CHECK(map != NULL, "map render target");
   memset(map, 0x55, STRIDE * H);

   /* Record: clear to red, then draw a green triangle. */
   ID3D12CommandAllocator *alloc = NULL;
   ID3D12GraphicsCommandList *list = NULL;
   ID3D12Device_CreateCommandAllocator(dev, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                       &IID_ID3D12CommandAllocator,
                                       (void **)&alloc);
   ID3D12Device_CreateCommandList(dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                  alloc, NULL,
                                  &IID_ID3D12GraphicsCommandList,
                                  (void **)&list);

   const FLOAT clear_red[4] = {1.0f, 0.0f, 0.0f, 1.0f};
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

   ID3D12CommandQueue *queue = NULL;
   ID3D12Fence *fence = NULL;
   const char *blocked_stage = NULL;
   HRESULT blocked_hr = S_OK;

   hr = ID3D12GraphicsCommandList_Close(list);
   CHECK(hr == S_OK, "Close command list (clear + draw)");
   if (FAILED(hr)) {
      blocked_stage = "Close";
      blocked_hr = hr;
      goto cleanup;
   }

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

   ID3D12CommandList *lists[] = {(ID3D12CommandList *)list};
   ID3D12CommandQueue_ExecuteCommandLists(queue, 1, lists);
   hr = ID3D12CommandQueue_Signal(queue, fence, 1);
   CHECK(hr == S_OK, "Signal after execution");
   if (FAILED(hr)) {
      blocked_stage = "Signal";
      blocked_hr = hr;
      goto cleanup;
   }
   hr = ID3D12Fence_SetEventOnCompletion(fence, 1, NULL);
   CHECK(hr == S_OK, "GPU render complete");

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
      ID3D12GraphicsCommandList_Reset(list, alloc, NULL);
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
      ID3D12GraphicsCommandList_Close(list);
      ID3D12CommandQueue_ExecuteCommandLists(queue, 1, lists);
      ID3D12CommandQueue_Signal(queue, fence, 2);
      ID3D12Fence_SetEventOnCompletion(fence, 2, NULL);

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

   /* THE INDEPENDENT CHECK. Everything above reads the texture through our
    * own mapping, which only works because our textures happen to be linear
    * -- our harness both draws the image and grades it. This instead pulls
    * the pixels out the way any real application (and every pixel assertion
    * in the vkd3d suite) does: GetCopyableFootprints to learn the layout,
    * then CopyTextureRegion into a plain buffer.
    */
   {
      D3D12_RESOURCE_DESC rtd = d12_test_resource_get_desc(rt);
      D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp = {0};
      UINT rows = 0;
      UINT64 row_bytes = 0, total = 0;
      ID3D12Device_GetCopyableFootprints(dev, &rtd, 0, 1, 0, &fp, &rows,
                                         &row_bytes, &total);

      CHECK(fp.Footprint.Width == W && fp.Footprint.Height == H &&
               rows == H && row_bytes == (UINT64)W * 4,
            "GetCopyableFootprints describes the render target");
      CHECK(fp.Footprint.RowPitch % 256 == 0 && fp.Footprint.RowPitch >= W * 4,
            "footprint row pitch honours the 256-byte D3D12 alignment");

      D3D12_HEAP_PROPERTIES rb = {.Type = D3D12_HEAP_TYPE_READBACK};
      D3D12_RESOURCE_DESC bd = {
         .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
         .Width = fp.Footprint.RowPitch * (UINT64)H,
         .Height = 1,
         .DepthOrArraySize = 1,
         .MipLevels = 1,
         .SampleDesc = {.Count = 1},
         .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
      };
      ID3D12Resource *dstbuf = NULL;
      hr = ID3D12Device_CreateCommittedResource(
         dev, &rb, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COMMON,
         NULL, &IID_ID3D12Resource, (void **)&dstbuf);
      CHECK(hr == S_OK && dstbuf, "create readback buffer from the footprint");

      uint8_t *rbmap = NULL;
      ID3D12Resource_Map(dstbuf, 0, NULL, (void **)&rbmap);
      memset(rbmap, 0x33, (size_t)(fp.Footprint.RowPitch * (UINT64)H));

      /* Re-render, then copy the texture out through the API. */
      ID3D12GraphicsCommandList_Reset(list, alloc, NULL);
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
      ID3D12GraphicsCommandList_CopyTextureRegion(list, &dloc, 0, 0, 0, &sloc,
                                                  NULL);
      ID3D12GraphicsCommandList_Close(list);
      ID3D12CommandQueue_ExecuteCommandLists(queue, 1, lists);
      ID3D12CommandQueue_Signal(queue, fence, 3);
      ID3D12Fence_SetEventOnCompletion(fence, 3, NULL);

      /* Same two sample points as above, read out through the API. */
      const uint8_t *c = rbmap + iy * fp.Footprint.RowPitch + ix * 4;
      const uint8_t *k = rbmap + oy * fp.Footprint.RowPitch + ox * 4;
      printf("       via CopyTextureRegion: inside %02x%02x%02x  "
             "outside %02x%02x%02x\n",
             c[0], c[1], c[2], k[0], k[1], k[2]);

      CHECK(c[0] == 0x40 && c[1] == 0x80 && c[2] == 0xbf,
            "triangle read back through CopyTextureRegion, not our mapping");
      CHECK(k[0] == 0xff && k[1] == 0x00 && k[2] == 0x00,
            "clear colour read back through CopyTextureRegion");

      ID3D12Resource_Unmap(dstbuf, 0, NULL);
      ID3D12Resource_Release(dstbuf);
   }

   /* A SECOND FORMAT. Everything above is RGBA8, which the driver used to
    * hardcode in eight places. BGRA8 is the sharpest check available: it
    * has the same size and layout but the opposite channel order, so a
    * missing swizzle shows up as red and blue swapped rather than as a
    * crash.
    */
   {
      D3D12_HEAP_PROPERTIES dh = {.Type = D3D12_HEAP_TYPE_DEFAULT};
      D3D12_RESOURCE_DESC bd2 = {
         .Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
         .Width = 64,
         .Height = 64,
         .DepthOrArraySize = 1,
         .MipLevels = 1,
         .Format = DXGI_FORMAT_B8G8R8A8_UNORM,
         .SampleDesc = {.Count = 1},
         .Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
      };
      ID3D12Resource *bgra = NULL;
      hr = ID3D12Device_CreateCommittedResource(
         dev, &dh, D3D12_HEAP_FLAG_NONE, &bd2,
         D3D12_RESOURCE_STATE_RENDER_TARGET, NULL, &IID_ID3D12Resource,
         (void **)&bgra);
      CHECK(hr == S_OK && bgra, "create a BGRA8 render target");

      D3D12_DESCRIPTOR_HEAP_DESC hd2 = {
         .Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV, .NumDescriptors = 1};
      ID3D12DescriptorHeap *h2 = NULL;
      ID3D12Device_CreateDescriptorHeap(dev, &hd2, &IID_ID3D12DescriptorHeap,
                                        (void **)&h2);
      D3D12_CPU_DESCRIPTOR_HANDLE rtv2 =
         d12_test_descriptor_heap_get_cpu_start(h2);
      ID3D12Device_CreateRenderTargetView(dev, bgra, NULL, rtv2);

      uint8_t *m2 = NULL;
      ID3D12Resource_Map(bgra, 0, NULL, (void **)&m2);
      memset(m2, 0x11, 64 * ((64 * 4 + 63) & ~63u));

      /* Clear to pure red. In BGRA memory order that must land as
       * 00 00 ff ff, NOT ff 00 00 ff.
       */
      ID3D12GraphicsCommandList_Reset(list, alloc, NULL);
      ID3D12GraphicsCommandList_ClearRenderTargetView(list, rtv2, clear_red,
                                                      0, NULL);
      ID3D12GraphicsCommandList_Close(list);
      ID3D12CommandQueue_ExecuteCommandLists(queue, 1, lists);
      ID3D12CommandQueue_Signal(queue, fence, 4);
      ID3D12Fence_SetEventOnCompletion(fence, 4, NULL);

      printf("       BGRA8 cleared to red -> memory %02x %02x %02x %02x\n",
             m2[0], m2[1], m2[2], m2[3]);
      CHECK(m2[0] == 0x00 && m2[1] == 0x00 && m2[2] == 0xff && m2[3] == 0xff,
            "BGRA8 channel order is honoured (red is not swapped to blue)");

      ID3D12Resource_Unmap(bgra, 0, NULL);
      ID3D12DescriptorHeap_Release(h2);
      ID3D12Resource_Release(bgra);
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
   ID3D12Resource_Unmap(rt, 0, NULL);
   if (fence)
      ID3D12Fence_Release(fence);
   if (queue)
      ID3D12CommandQueue_Release(queue);
   ID3D12GraphicsCommandList_Release(list);
   ID3D12CommandAllocator_Release(alloc);
   ID3D12DescriptorHeap_Release(rtv_heap);
   ID3D12Resource_Release(rt);
   ID3D12PipelineState_Release(pso);
   ID3D12RootSignature_Release(root_sig);
   ID3D12Device_Release(dev);

   printf(failures ? "G4 TRIANGLE: %d FAILURES\n" : "G4 TRIANGLE: ALL PASS\n",
          failures);
   free(vs);
   free(ps);
   return failures;
}
