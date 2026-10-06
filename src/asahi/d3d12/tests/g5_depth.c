/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * d3d12agx depth/stencil acceptance.
 *
 * A depth test is easy to fake: a driver that ignores depth entirely draws
 * everything, and one that mis-encodes the comparison draws nothing. Both
 * look plausible in a screenshot. So every case here comes in a PAIR that
 * differs by ONE number, with opposite expected outcomes -- same geometry,
 * same shaders, same pipeline except the field under test. Only a driver
 * that actually runs the test on the hardware can pass both halves.
 *
 *   depth value   0.75 vs 0.125 against a 0.25 clear   -> rejected / drawn
 *   depth func    LESS vs GREATER, same values         -> drawn / rejected
 *   write mask    ALL vs ZERO on a first draw          -> occludes / does not
 *
 * The last pair is the one that proves depth is really being STORED: the
 * second draw's fate depends entirely on what the first draw left behind.
 *
 * Grading is through the colour buffer, because AGX depth buffers must be
 * twiddled (the ZLS unit has no linear mode) and are not CPU-readable by
 * row. Two pixel shaders in different colours make "which draw won" visible.
 *
 * Usage: g5_depth tri.dxil solid.dxil tri_depth.dxil solid_g.dxil
 *                 tri_depth_cw.dxil
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
#ifdef D12_D32_FIRST_PAIR_PROBE
#include "d12_d32_pair_oracle.h"
#endif

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

/* Where every flat-depth case is graded. It has to be inside BOTH
 * triangles this test draws, and outside neither by accident: grading at
 * a point the triangle never covers makes each case pass whenever the
 * draw was rejected, for any reason, including the right one.
 *
 * Read off the DXIL, not off a memory of the HLSL:
 *
 *   tri.dxil       clip (1,-1) (-1,-1) (5,1)  -> window (256,256)
 *                  (0,256) (768,0); the visible part is the wedge under
 *                  the hypotenuse y = 256 - x/3, so nothing above y=171
 *                  is covered at all.
 *   tri_depth.dxil clip (-0.9,-0.9) (0.9,-0.9) (0,0.9) -> window
 *                  (12.8,243.2) (243.2,243.2) (128,12.8), apex at the top.
 *
 * (220,240) is ~58 px below the first triangle's hypotenuse and ~22 px
 * inside the second's right edge. The previous point, (30,30), was inside
 * NEITHER -- it was chosen against a triangle description that does not
 * match either shader, and survived only because a translator bug made
 * tri.dxil emit garbage vertices. See docs/findings.md.
 */
#define IN_X 220
#define IN_Y 240

/* The clear colour, and the two pixel-shader colours, as RGBA8 bytes. */
static const FLOAT clear_rgba[4] = {1.0f, 0.0f, 0.0f, 1.0f};
static const uint8_t RED[4] = {255, 0, 0, 255};
/* solid.hlsl: float4(0.25, 0.5, 0.75, 1.0) */
static const uint8_t BLUE[4] = {64, 128, 191, 255};
/* solid_g.hlsl: float4(0.0, 1.0, 0.0, 1.0) */
static const uint8_t GREEN[4] = {0, 255, 0, 255};

static void *
slurp(const char *path, size_t *size)
{
   FILE *f = fopen(path, "rb");
   if (!f)
      return NULL;
   if (fseek(f, 0, SEEK_END)) { fclose(f); return NULL; }
   long n = ftell(f);
   if (n <= 0 || fseek(f, 0, SEEK_SET)) { fclose(f); return NULL; }
   void *p = malloc((size_t)n);
   if (p && fread(p, 1, n, f) != (size_t)n) {
      free(p);
      p = NULL;
   }
   fclose(f);
   *size = n;
   return p;
}

static const uint8_t *
px(const uint8_t *map, unsigned x, unsigned y)
{
   return map + y * STRIDE + x * 4;
}

/* Name a pixel's colour for readable failure output. */
static const char *
colour_name(const uint8_t *p)
{
   if (!memcmp(p, RED, 4))
      return "RED (clear)";
   if (!memcmp(p, BLUE, 4))
      return "BLUE (solid.hlsl)";
   if (!memcmp(p, GREEN, 4))
      return "GREEN (solid_g.hlsl)";
   return "other";
}

static int
expect_pixel(const uint8_t *map, unsigned x, unsigned y,
             const uint8_t want[4], const char *what)
{
   const uint8_t *p = px(map, x, y);
   if (!memcmp(p, want, 4)) {
      printf("ok   - %s\n", what);
      return 1;
   }

   printf("FAIL - %s\n", what);
   printf("       (%u,%u) is %02x%02x%02x%02x = %s, wanted "
          "%02x%02x%02x%02x = %s\n",
          x, y, p[0], p[1], p[2], p[3], colour_name(p), want[0], want[1],
          want[2], want[3], colour_name(want));
   failures++;
   return 0;
}

struct ctx {
   ID3D12Device *dev;
   ID3D12CommandQueue *queue;
   ID3D12Fence *fence;
   UINT64 fence_value;
   ID3D12CommandAllocator *alloc;
   ID3D12GraphicsCommandList *list;
   ID3D12Resource *rt, *ds;
   D3D12_CPU_DESCRIPTOR_HANDLE rtv, dsv;
   uint8_t *map;
   ID3D12RootSignature *root;
   const char *blocked_stage;
   HRESULT blocked_hr;
};

static bool
frame_hr(struct ctx *c, HRESULT hr, const char *stage)
{
   if (hr == S_OK) return true;
   c->blocked_stage = stage; c->blocked_hr = hr;
   return false;
}

static void
transition_rt(struct ctx *c, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after)
{
   const D3D12_RESOURCE_BARRIER b = {
      .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
      .Transition = {.pResource = c->rt,
         .Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
         .StateBefore = before, .StateAfter = after},
   };
   ID3D12GraphicsCommandList_ResourceBarrier(c->list, 1, &b);
}

static bool
begin_colour_frame(struct ctx *c)
{
   /* Previous end_frame observed completion before allocator storage reuse. */
   if (!frame_hr(c, ID3D12CommandAllocator_Reset(c->alloc), "Reset fenced allocator") ||
       !frame_hr(c, ID3D12GraphicsCommandList_Reset(c->list, c->alloc, NULL), "Reset graphics list"))
      return false;
   transition_rt(c, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET);
   ID3D12GraphicsCommandList_SetGraphicsRootSignature(c->list, c->root);
   return true;
}

/* One recorded-and-executed frame. Depth state comes from the PSO, depth
 * value from the viewport: setting MinDepth == MaxDepth pins window Z to a
 * chosen constant no matter what the vertex shader emits, which keeps these
 * cases independent of the shader's own Z.
 */
static bool
begin_frame(struct ctx *c, FLOAT clear_depth)
{
   if (!begin_colour_frame(c)) return false;
   ID3D12GraphicsCommandList_ClearRenderTargetView(c->list, c->rtv,
                                                   clear_rgba, 0, NULL);
   ID3D12GraphicsCommandList_ClearDepthStencilView(
      c->list, c->dsv, D3D12_CLEAR_FLAG_DEPTH, clear_depth, 0, 0, NULL);
   ID3D12GraphicsCommandList_OMSetRenderTargets(c->list, 1, &c->rtv, FALSE,
                                                &c->dsv);

   D3D12_RECT sc = {.right = W, .bottom = H};
   ID3D12GraphicsCommandList_RSSetScissorRects(c->list, 1, &sc);
   ID3D12GraphicsCommandList_IASetPrimitiveTopology(
      c->list, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
   return true;
}

static void
draw_at_depth(struct ctx *c, ID3D12PipelineState *pso, FLOAT depth)
{
   D3D12_VIEWPORT vp = {
      .Width = W,
      .Height = H,
      .MinDepth = depth,
      .MaxDepth = depth,
   };
   ID3D12GraphicsCommandList_RSSetViewports(c->list, 1, &vp);
   ID3D12GraphicsCommandList_SetPipelineState(c->list, pso);
   ID3D12GraphicsCommandList_DrawInstanced(c->list, 3, 1, 0, 0);
}

static bool
end_frame(struct ctx *c)
{
   transition_rt(c, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COMMON);
   if (!frame_hr(c, ID3D12GraphicsCommandList_Close(c->list), "Close depth frame"))
      return false; /* Unsupported producers remain blocked before submission. */
   ID3D12CommandList *lists[] = {(ID3D12CommandList *)c->list};
   ID3D12CommandQueue_ExecuteCommandLists(c->queue, 1, lists);
   if (!frame_hr(c, ID3D12Device_GetDeviceRemovedReason(c->dev), "device after submit"))
      return false;
   const UINT64 value = ++c->fence_value;
   if (!frame_hr(c, ID3D12CommandQueue_Signal(c->queue, c->fence, value), "Signal depth fence") ||
       !frame_hr(c, ID3D12Fence_SetEventOnCompletion(c->fence, value, NULL), "wait depth fence"))
      return false;
   UINT64 completed = ID3D12Fence_GetCompletedValue(c->fence);
   if (completed == UINT64_MAX || completed < value)
      return frame_hr(c, E_FAIL, "observed exact depth fence");
   if (!frame_hr(c, ID3D12Device_GetDeviceRemovedReason(c->dev), "device after fence") ||
       !frame_hr(c, ID3D12Resource_Map(c->rt, 0, NULL, NULL), "Map opaque colour target"))
      return false;
   memset(c->map, 0x33, STRIDE * H);
   HRESULT hr = ID3D12Resource_ReadFromSubresource(c->rt, c->map, STRIDE,
                                                   STRIDE * H, 0, NULL);
   ID3D12Resource_Unmap(c->rt, 0, NULL);
   return frame_hr(c, hr, "read actual fenced colour bytes");
}

/* A graphics pipeline differing only in its depth/stencil state. */
static ID3D12PipelineState *
make_pso(ID3D12Device *dev, ID3D12RootSignature *root, const void *vs, size_t vs_size, const void *ps,
         size_t ps_size, BOOL depth_enable, D3D12_DEPTH_WRITE_MASK write,
         D3D12_COMPARISON_FUNC func, DXGI_FORMAT dsv_format)
{
   D3D12_GRAPHICS_PIPELINE_STATE_DESC pd = {
      .pRootSignature = root,
      .PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
      .VS = {.pShaderBytecode = vs, .BytecodeLength = vs_size},
      .PS = {.pShaderBytecode = ps, .BytecodeLength = ps_size},
      .NumRenderTargets = 1,
      .RTVFormats = {DXGI_FORMAT_R8G8B8A8_UNORM},
      .DSVFormat = dsv_format,
      .DepthStencilState =
         {
            .DepthEnable = depth_enable,
            .DepthWriteMask = write,
            .DepthFunc = func,
         },
      .RasterizerState = {.FillMode = D3D12_FILL_MODE_SOLID,
                          .DepthClipEnable = TRUE, .CullMode = D3D12_CULL_MODE_NONE},
      .SampleDesc = {.Count = 1},
      .SampleMask = ~0u,
   };

   pd.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
   ID3D12PipelineState *pso = NULL;
   if (S_OK != (ID3D12Device_CreateGraphicsPipelineState(
          dev, &pd, &IID_ID3D12PipelineState, (void **)&pso))) {
      if (pso) ID3D12PipelineState_Release(pso);
      return NULL;
   }
   return pso;
}

/* A pipeline that differs only in its rasterizer facing/culling state. */
static ID3D12PipelineState *
make_cull_pso(ID3D12Device *dev, ID3D12RootSignature *root, const void *vs, size_t vs_size,
              const void *ps, size_t ps_size, D3D12_CULL_MODE cull,
              BOOL front_ccw)
{
   D3D12_GRAPHICS_PIPELINE_STATE_DESC pd = {
      .pRootSignature = root,
      .PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
      .VS = {.pShaderBytecode = vs, .BytecodeLength = vs_size},
      .PS = {.pShaderBytecode = ps, .BytecodeLength = ps_size},
      .NumRenderTargets = 1,
      .RTVFormats = {DXGI_FORMAT_R8G8B8A8_UNORM},
      .RasterizerState =
         {
            .FillMode = D3D12_FILL_MODE_SOLID,
            .DepthClipEnable = TRUE,
            .CullMode = cull,
            .FrontCounterClockwise = front_ccw,
         },
      .SampleDesc = {.Count = 1},
      .SampleMask = ~0u,
   };

   pd.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
   ID3D12PipelineState *pso = NULL;
   if (S_OK != (ID3D12Device_CreateGraphicsPipelineState(
          dev, &pd, &IID_ID3D12PipelineState, (void **)&pso))) {
      if (pso) ID3D12PipelineState_Release(pso);
      return NULL;
   }
   return pso;
}

/* Draw one triangle with no depth buffer and report how many pixels the
 * rasterizer produced. Used to grade culling, where the question is simply
 * whether the primitive survived.
 */
static unsigned
count_covered(struct ctx *c, ID3D12PipelineState *pso)
{
   if (!pso || !begin_colour_frame(c)) return ~0u;
   ID3D12GraphicsCommandList_ClearRenderTargetView(c->list, c->rtv,
                                                   clear_rgba, 0, NULL);
   ID3D12GraphicsCommandList_OMSetRenderTargets(c->list, 1, &c->rtv, FALSE,
                                                NULL);
   D3D12_RECT sc = {.right = W, .bottom = H};
   ID3D12GraphicsCommandList_RSSetScissorRects(c->list, 1, &sc);
   D3D12_VIEWPORT vp = {.Width = W, .Height = H, .MaxDepth = 1.0f};
   ID3D12GraphicsCommandList_RSSetViewports(c->list, 1, &vp);
   ID3D12GraphicsCommandList_IASetPrimitiveTopology(
      c->list, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
   ID3D12GraphicsCommandList_SetPipelineState(c->list, pso);
   ID3D12GraphicsCommandList_DrawInstanced(c->list, 3, 1, 0, 0);

   if (!end_frame(c))
      return ~0u;

   unsigned n = 0;
   for (unsigned y = 0; y < H; y++)
      for (unsigned x = 0; x < W; x++)
         if (memcmp(px(c->map, x, y), RED, 4))
            n++;
   return n;
}

int
main(int argc, char **argv)
{
   if (argc < 6) {
      fprintf(stderr,
              "usage: %s tri.dxil solid.dxil tri_depth.dxil solid_g.dxil "
              "tri_depth_cw.dxil\n",
              argv[0]);
      return 2;
   }

#ifdef D12_D32_FIRST_PAIR_PROBE
   const char *physical = getenv("AURORA_D3D12_PHYSICAL_D32");
   const char *baseline = getenv("AURORA_D3D12_PHYSICAL_BASELINE");
   const char *preload = getenv("LD_PRELOAD");
   if (!physical || strcmp(physical, "1") || !baseline || strcmp(baseline, "1") ||
       (preload && preload[0])) {
      fprintf(stderr, "D32 physical pair requires explicit admission and no preload\n");
      return 2;
   }
   bool pair_complete = false;
#endif
   struct ctx c = {0};
   ID3D12DescriptorHeap *rtv_heap = NULL, *dsv_heap = NULL;
   ID3D12Resource *ds24 = NULL;
   ID3D12CommandAllocator *negative_alloc = NULL;
   ID3D12GraphicsCommandList *negative_list = NULL;
   ID3DBlob *root_blob = NULL;
   ID3D12PipelineState *pso_less = NULL, *pso_greater = NULL, *pso_write = NULL,
      *pso_nowrite = NULL, *pso_nods = NULL, *pso_slope = NULL, *pso_slope_cw = NULL,
      *cull_none = NULL, *cull_back = NULL, *cull_front = NULL, *cull_front_ccw = NULL;
#define REQUIRE(cond, label) do { bool ok_ = !!(cond); CHECK(ok_, label); \
   if (!ok_) { if (!c.blocked_stage) { c.blocked_stage = label; c.blocked_hr = E_FAIL; } goto cleanup; } } while (0)
#define REQUIRE_HR(call, label) do { hr = (call); REQUIRE(frame_hr(&c, hr, label), label); } while (0)
   size_t vs_size, ps_size, vsd_size, psg_size, vscw_size;
   void *vs = slurp(argv[1], &vs_size);
   void *ps = slurp(argv[2], &ps_size);
   void *vs_depth = slurp(argv[3], &vsd_size);
   void *ps_green = slurp(argv[4], &psg_size);
   void *vs_depth_cw = slurp(argv[5], &vscw_size);
   if (!vs || !ps || !vs_depth || !ps_green || !vs_depth_cw) {
      fprintf(stderr, "cannot read shaders\n");
      free(vs); free(ps); free(vs_depth); free(ps_green); free(vs_depth_cw);
      return 2;
   }

   HRESULT hr = D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0,
                                  &IID_ID3D12Device, (void **)&c.dev);
   REQUIRE(hr == S_OK && c.dev, "D3D12CreateDevice");
   const D3D12_ROOT_SIGNATURE_DESC rs = {0};
   REQUIRE_HR(D3D12SerializeRootSignature(&rs, D3D_ROOT_SIGNATURE_VERSION_1_0,
                 &root_blob, NULL), "serialize empty graphics root");
   REQUIRE(root_blob, "root blob");
   REQUIRE_HR(ID3D12Device_CreateRootSignature(c.dev, 0,
      ID3D10Blob_GetBufferPointer(root_blob), ID3D10Blob_GetBufferSize(root_blob),
      &IID_ID3D12RootSignature, (void **)&c.root), "create empty graphics root");
   REQUIRE(c.root, "root object");

   /* ---- Depth resources and DSV heap ---- */

   D3D12_HEAP_PROPERTIES heap = {.Type = D3D12_HEAP_TYPE_DEFAULT};
   const D3D12_HEAP_PROPERTIES colour_heap = {
      .Type = D3D12_HEAP_TYPE_CUSTOM, .CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
      .MemoryPoolPreference = D3D12_MEMORY_POOL_L0, .CreationNodeMask = 1, .VisibleNodeMask = 1};
   D3D12_RESOURCE_DESC rtd = {
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
   hr = ID3D12Device_CreateCommittedResource(
      c.dev, &colour_heap, D3D12_HEAP_FLAG_NONE, &rtd,
      D3D12_RESOURCE_STATE_COMMON, NULL, &IID_ID3D12Resource,
      (void **)&c.rt);
   REQUIRE(hr == S_OK && c.rt, "create opaque RGBA8 render target");

   D3D12_RESOURCE_DESC dsd = {
      .Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
      .Width = W,
      .Height = H,
      .DepthOrArraySize = 1,
      .MipLevels = 1,
      .Format = DXGI_FORMAT_D32_FLOAT,
      .SampleDesc = {.Count = 1},
      .Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
   };
   hr = ID3D12Device_CreateCommittedResource(
      c.dev, &heap, D3D12_HEAP_FLAG_NONE, &dsd,
      D3D12_RESOURCE_STATE_DEPTH_WRITE, NULL, &IID_ID3D12Resource,
      (void **)&c.ds);
   REQUIRE(hr == S_OK && c.ds, "create D32_FLOAT depth buffer");

   /* A depth format with a stencil plane must also be creatable: it needs a
    * second allocation, since AGX does not interleave depth and stencil.
    */
   D3D12_RESOURCE_DESC d24 = dsd;
   d24.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
   hr = ID3D12Device_CreateCommittedResource(
      c.dev, &heap, D3D12_HEAP_FLAG_NONE, &d24,
      D3D12_RESOURCE_STATE_DEPTH_WRITE, NULL, &IID_ID3D12Resource,
      (void **)&ds24);
   CHECK(hr == S_OK && ds24, "create D24_UNORM_S8_UINT (separate stencil "
                             "plane)");
   REQUIRE(hr == S_OK && ds24, "D24 resource object");
   ID3D12Resource_Release(ds24); ds24 = NULL;

   D3D12_DESCRIPTOR_HEAP_DESC rtv_hd = {
      .Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV, .NumDescriptors = 1};
   hr = ID3D12Device_CreateDescriptorHeap(c.dev, &rtv_hd,
                                     &IID_ID3D12DescriptorHeap,
                                     (void **)&rtv_heap);
   REQUIRE(hr == S_OK && rtv_heap, "CreateDescriptorHeap(RTV)");

   D3D12_DESCRIPTOR_HEAP_DESC dsv_hd = {
      .Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV, .NumDescriptors = 2};
   hr = ID3D12Device_CreateDescriptorHeap(c.dev, &dsv_hd,
                                          &IID_ID3D12DescriptorHeap,
                                          (void **)&dsv_heap);
   REQUIRE(hr == S_OK && dsv_heap, "CreateDescriptorHeap(DSV)");

   c.rtv = d12_test_descriptor_heap_get_cpu_start(rtv_heap);
   c.dsv = d12_test_descriptor_heap_get_cpu_start(dsv_heap);
   REQUIRE(c.rtv.ptr && c.dsv.ptr, "RTV/DSV heap handles");
   ID3D12Device_CreateRenderTargetView(c.dev, c.rt, NULL, c.rtv);
   ID3D12Device_CreateDepthStencilView(c.dev, c.ds, NULL, c.dsv);

   /* A colour texture is not a depth/stencil target. Accepting it here
    * would point the ZLS unit at an RGBA8 buffer and corrupt it.
    */
   UINT dsv_inc = ID3D12Device_GetDescriptorHandleIncrementSize(
      c.dev, D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
   REQUIRE(dsv_inc && c.dsv.ptr <= SIZE_MAX - dsv_inc, "DSV descriptor increment");
   D3D12_CPU_DESCRIPTOR_HANDLE bad_dsv = {.ptr = c.dsv.ptr + dsv_inc};
   ID3D12Device_CreateDepthStencilView(c.dev, c.rt, NULL, bad_dsv);

   c.map = malloc(STRIDE * H);
   REQUIRE(c.map, "CPU storage for actual opaque colour readback");
   memset(c.map, 0x55, STRIDE * H);
   REQUIRE_HR(ID3D12Resource_Map(c.rt, 0, NULL, NULL), "Map opaque target for poison");
   hr = ID3D12Resource_WriteToSubresource(c.rt, 0, NULL, c.map, STRIDE, STRIDE * H);
   ID3D12Resource_Unmap(c.rt, 0, NULL);
   REQUIRE(frame_hr(&c, hr, "write actual opaque target poison"), "publish opaque target poison");

   D3D12_COMMAND_QUEUE_DESC qd = {.Type = D3D12_COMMAND_LIST_TYPE_DIRECT};
   REQUIRE_HR(ID3D12Device_CreateCommandQueue(c.dev, &qd, &IID_ID3D12CommandQueue,
                                   (void **)&c.queue), "create queue");
   REQUIRE(c.queue, "queue object");
   REQUIRE_HR(ID3D12Device_CreateFence(c.dev, 0, D3D12_FENCE_FLAG_NONE, &IID_ID3D12Fence,
                            (void **)&c.fence), "create fence");
   REQUIRE(c.fence, "fence object");
   REQUIRE_HR(ID3D12Device_CreateCommandAllocator(c.dev, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                       &IID_ID3D12CommandAllocator, (void **)&c.alloc), "create allocator");
   REQUIRE(c.alloc, "allocator object");
   REQUIRE_HR(ID3D12Device_CreateCommandList(c.dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                  c.alloc, NULL, &IID_ID3D12GraphicsCommandList,
                                  (void **)&c.list), "create graphics list");
   REQUIRE(c.list, "graphics list object");
   REQUIRE_HR(ID3D12GraphicsCommandList_Close(c.list), "close initial empty list");

   /* A failed Close permanently poisons its list. Keep the rejection oracle
    * on separate, never-submitted storage and preserve the execution list. */
   REQUIRE_HR(ID3D12Device_CreateCommandAllocator(c.dev, D3D12_COMMAND_LIST_TYPE_DIRECT,
      &IID_ID3D12CommandAllocator, (void **)&negative_alloc), "negative DSV allocator");
   REQUIRE(negative_alloc, "negative allocator object");
   REQUIRE_HR(ID3D12Device_CreateCommandList(c.dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT,
      negative_alloc, NULL, &IID_ID3D12GraphicsCommandList, (void **)&negative_list), "negative DSV list");
   REQUIRE(negative_list, "negative list object");
   ID3D12GraphicsCommandList_ClearDepthStencilView(
      negative_list, bad_dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, NULL);
   REQUIRE(FAILED(ID3D12GraphicsCommandList_Close(negative_list)),
         "a colour texture is refused as a depth/stencil view");
   ID3D12GraphicsCommandList_Release(negative_list); negative_list = NULL;
   ID3D12CommandAllocator_Release(negative_alloc); negative_alloc = NULL;

   /* ---- Pipelines ---- */

   pso_less =
      make_pso(c.dev, c.root, vs, vs_size, ps, ps_size, TRUE,
               D3D12_DEPTH_WRITE_MASK_ALL, D3D12_COMPARISON_FUNC_LESS,
               DXGI_FORMAT_D32_FLOAT);
#ifndef D12_D32_FIRST_PAIR_PROBE
   pso_greater =
      make_pso(c.dev, c.root, vs, vs_size, ps, ps_size, TRUE,
               D3D12_DEPTH_WRITE_MASK_ALL, D3D12_COMPARISON_FUNC_GREATER,
               DXGI_FORMAT_D32_FLOAT);
   /* First draw of the write-mask pair: always passes, colour GREEN, and
    * differs between the two runs ONLY in whether it writes depth.
    */
   pso_write =
      make_pso(c.dev, c.root, vs, vs_size, ps_green, psg_size, TRUE,
               D3D12_DEPTH_WRITE_MASK_ALL, D3D12_COMPARISON_FUNC_ALWAYS,
               DXGI_FORMAT_D32_FLOAT);
   pso_nowrite =
      make_pso(c.dev, c.root, vs, vs_size, ps_green, psg_size, TRUE,
               D3D12_DEPTH_WRITE_MASK_ZERO, D3D12_COMPARISON_FUNC_ALWAYS,
               DXGI_FORMAT_D32_FLOAT);
   /* Depth requested, but the pass will have no depth attached. */
   pso_nods =
      make_pso(c.dev, c.root, vs, vs_size, ps, ps_size, TRUE,
               D3D12_DEPTH_WRITE_MASK_ALL, D3D12_COMPARISON_FUNC_LESS,
               DXGI_FORMAT_D32_FLOAT);
   /* Interpolated depth across the primitive, in both windings. */
   pso_slope =
      make_pso(c.dev, c.root, vs_depth, vsd_size, ps_green, psg_size, TRUE,
               D3D12_DEPTH_WRITE_MASK_ALL, D3D12_COMPARISON_FUNC_LESS,
               DXGI_FORMAT_D32_FLOAT);
   pso_slope_cw =
      make_pso(c.dev, c.root, vs_depth_cw, vscw_size, ps_green, psg_size, TRUE,
               D3D12_DEPTH_WRITE_MASK_ALL, D3D12_COMPARISON_FUNC_LESS,
               DXGI_FORMAT_D32_FLOAT);

   REQUIRE(pso_less && pso_greater && pso_write && pso_nowrite && pso_nods &&
            pso_slope && pso_slope_cw,
         "compile seven pipelines differing in depth state");
#else
   REQUIRE(pso_less, "compile original first-pair D32 pipeline");
#endif
   printf("\n-- pair 1: same pipeline, depth 0.75 vs 0.125 against a 0.25 "
          "clear\n");

   REQUIRE(begin_frame(&c, 0.25f), "begin depth frame");
   draw_at_depth(&c, pso_less, 0.75f);
   REQUIRE(end_frame(&c), "far draw executes");
   expect_pixel(c.map, IN_X, IN_Y,RED,
                "0.75 LESS 0.25 is false: the triangle is rejected");
#ifdef D12_D32_FIRST_PAIR_PROBE
   unsigned triangle = 0;
   unsigned bad = d12_d32_pair_pixel_mismatches(c.map, STRIDE, false, &triangle);
   printf("D32 FAR PIXELS total=65536 triangle=%u mismatches=%u fence=%llu\n",
          triangle, bad, (unsigned long long)c.fence_value);
   REQUIRE(bad == 0 && triangle == 10965, "all original far-frame pixels");
#endif

   REQUIRE(begin_frame(&c, 0.25f), "begin depth frame");
   draw_at_depth(&c, pso_less, 0.125f);
   REQUIRE(end_frame(&c), "near draw executes");
   expect_pixel(c.map, IN_X, IN_Y,BLUE,
                "0.125 LESS 0.25 is true: the triangle is drawn");
#ifdef D12_D32_FIRST_PAIR_PROBE
   bad = d12_d32_pair_pixel_mismatches(c.map, STRIDE, true, &triangle);
   printf("D32 NEAR PIXELS total=65536 triangle=%u mismatches=%u fence=%llu\n",
          triangle, bad, (unsigned long long)c.fence_value);
   REQUIRE(bad == 0 && triangle == 10965, "all original near-frame pixels");
   pair_complete = true;
   goto cleanup; /* Separate first-pair probe; mandatory full G5 still runs below. */
#endif

   printf("\n-- pair 2: identical values, LESS vs GREATER\n");

   REQUIRE(begin_frame(&c, 0.5f), "begin depth frame");
   draw_at_depth(&c, pso_less, 0.125f);
   REQUIRE(end_frame(&c), "LESS draw executes");
   expect_pixel(c.map, IN_X, IN_Y,BLUE, "0.125 LESS 0.5 is true: drawn");

   REQUIRE(begin_frame(&c, 0.5f), "begin depth frame");
   draw_at_depth(&c, pso_greater, 0.125f);
   REQUIRE(end_frame(&c), "GREATER draw executes");
   expect_pixel(c.map, IN_X, IN_Y,RED,
                "0.125 GREATER 0.5 is false: rejected (DepthFunc honoured)");

   printf("\n-- pair 3: DepthWriteMask ALL vs ZERO on the occluding draw\n");

   /* Two draws in ONE pass. The second one's fate depends only on what the
    * first left in the depth buffer, so this is where a driver that never
    * actually stores depth gets caught.
    */
   REQUIRE(begin_frame(&c, 1.0f), "begin depth frame");
   draw_at_depth(&c, pso_write, 0.25f);   /* GREEN, writes depth 0.25 */
   draw_at_depth(&c, pso_less, 0.5f);     /* BLUE, 0.5 LESS 0.25 -> no */
   REQUIRE(end_frame(&c), "write-ALL frame executes");
   expect_pixel(c.map, IN_X, IN_Y,GREEN,
                "depth written by draw 1 occludes draw 2");

   REQUIRE(begin_frame(&c, 1.0f), "begin depth frame");
   draw_at_depth(&c, pso_nowrite, 0.25f); /* GREEN, writes no depth */
   draw_at_depth(&c, pso_less, 0.5f);     /* BLUE, 0.5 LESS 1.0 -> yes */
   REQUIRE(end_frame(&c), "write-ZERO frame executes");
   expect_pixel(c.map, IN_X, IN_Y,BLUE,
                "with writes masked off, draw 2 is not occluded");

   printf("\n-- depth interpolated across the primitive\n");

   /* tri_depth.hlsl: apex (128,13) at z=0.1, base y=243 at z=0.9, so z
    * crosses the 0.5 clear at y=128. One primitive, one draw, and the
    * result must be cut in half. A per-draw depth cannot do this.
    */
   REQUIRE(begin_frame(&c, 0.5f), "begin depth frame");
   {
      /* A full 0..1 depth range this time, so the shader's own Z reaches
       * the rasterizer instead of being pinned by the viewport.
       */
      D3D12_VIEWPORT vp = {.Width = W, .Height = H, .MaxDepth = 1.0f};
      ID3D12GraphicsCommandList_RSSetViewports(c.list, 1, &vp);
      ID3D12GraphicsCommandList_SetPipelineState(c.list, pso_slope);
      ID3D12GraphicsCommandList_DrawInstanced(c.list, 3, 1, 0, 0);
   }
   REQUIRE(end_frame(&c), "sloped-depth draw executes");
   expect_pixel(c.map, 128, 60, GREEN,
                "near end of the primitive (z~0.26) passes");
   expect_pixel(c.map, 128, 200, RED,
                "far end of the SAME primitive (z~0.75) is rejected");

   printf("\n-- the same triangle wound the other way round\n");

   /* AGX holds front-face and back-face depth/stencil state in separate
    * slots and picks per primitive. Writing only the front half leaves the
    * back half at whatever it was -- and all-zero decodes as compare NEVER,
    * so back-facing geometry would silently vanish. This is the same test
    * as above on the same pixels, with the winding reversed.
    */
   REQUIRE(begin_frame(&c, 0.5f), "begin depth frame");
   {
      D3D12_VIEWPORT vp = {.Width = W, .Height = H, .MaxDepth = 1.0f};
      ID3D12GraphicsCommandList_RSSetViewports(c.list, 1, &vp);
      ID3D12GraphicsCommandList_SetPipelineState(c.list, pso_slope_cw);
      ID3D12GraphicsCommandList_DrawInstanced(c.list, 3, 1, 0, 0);
   }
   REQUIRE(end_frame(&c), "reversed-winding draw executes");
   expect_pixel(c.map, 128, 60, GREEN,
                "the other face gets the same depth state (near end passes)");
   expect_pixel(c.map, 128, 200, RED,
                "the other face gets the same depth state (far end fails)");

   printf("\n-- facing and culling\n");

   /* Which winding D3D12 calls "front" is a convention our viewport's Y
    * negation inverts relative to the hardware flag, so it is asserted
    * rather than assumed. tri_depth is counter-clockwise in NDC, which is
    * D3D12's FRONT face with the default FrontCounterClockwise = FALSE.
    */
   {
      cull_none = make_cull_pso(
         c.dev, c.root, vs_depth, vsd_size, ps_green, psg_size,
         D3D12_CULL_MODE_NONE, FALSE);
      cull_back = make_cull_pso(
         c.dev, c.root, vs_depth, vsd_size, ps_green, psg_size,
         D3D12_CULL_MODE_BACK, FALSE);
      cull_front = make_cull_pso(
         c.dev, c.root, vs_depth, vsd_size, ps_green, psg_size,
         D3D12_CULL_MODE_FRONT, FALSE);
      /* Same pipeline as cull_front, but the opposite winding convention:
       * the triangle it culls must swap.
       */
      cull_front_ccw = make_cull_pso(
         c.dev, c.root, vs_depth, vsd_size, ps_green, psg_size,
         D3D12_CULL_MODE_FRONT, TRUE);

      REQUIRE(cull_none && cull_back && cull_front && cull_front_ccw,
            "compile pipelines differing only in cull state");

      unsigned n_none = count_covered(&c, cull_none);
      REQUIRE(n_none != ~0u, "fenced culling frame");
      unsigned n_back = count_covered(&c, cull_back);
      REQUIRE(n_back != ~0u, "fenced culling frame");
      unsigned n_front = count_covered(&c, cull_front);
      REQUIRE(n_front != ~0u, "fenced culling frame");
      unsigned n_front_ccw = count_covered(&c, cull_front_ccw);
      REQUIRE(n_front_ccw != ~0u, "fenced culling frame");

      printf("       covered: none=%u cullback=%u cullfront=%u "
             "cullfront+FCC=%u\n",
             n_none, n_back, n_front, n_front_ccw);

      CHECK(n_none > 0, "CULL_MODE_NONE draws the triangle");
      CHECK(n_back == n_none,
            "CULL_MODE_BACK keeps a front-facing triangle");
      CHECK(n_front == 0, "CULL_MODE_FRONT removes it (facing is honoured)");
      CHECK(n_front_ccw == n_none,
            "FrontCounterClockwise flips which face is culled");

   }

   printf("\n-- a depth-testing pipeline in a pass with no depth buffer\n");

   /* No DSV bound. D3D12 says the test is inert; encoding the comparison
    * anyway would test against whatever the ZLS unit last pointed at.
    */
   REQUIRE(begin_colour_frame(&c), "begin no-depth frame");
   ID3D12GraphicsCommandList_ClearRenderTargetView(c.list, c.rtv, clear_rgba,
                                                   0, NULL);
   ID3D12GraphicsCommandList_OMSetRenderTargets(c.list, 1, &c.rtv, FALSE,
                                                NULL);
   {
      D3D12_RECT sc = {.right = W, .bottom = H};
      ID3D12GraphicsCommandList_RSSetScissorRects(c.list, 1, &sc);
      ID3D12GraphicsCommandList_IASetPrimitiveTopology(
         c.list, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
   }
   draw_at_depth(&c, pso_nods, 0.75f);
   REQUIRE(end_frame(&c), "no-depth frame executes");
   expect_pixel(c.map, IN_X, IN_Y,BLUE,
                "depth test is inert with no depth buffer, not NEVER");

cleanup:
   if (c.blocked_stage)
      printf("G5 DEPTH BLOCKED: stage=%s HRESULT=0x%08x (not a hardware pass)\n",
             c.blocked_stage, (unsigned)c.blocked_hr);
   free(c.map);
   if (negative_list) ID3D12GraphicsCommandList_Release(negative_list);
   if (negative_alloc) ID3D12CommandAllocator_Release(negative_alloc);
   if (root_blob) ID3D10Blob_Release(root_blob);
   if (ds24) ID3D12Resource_Release(ds24);
   if (cull_front_ccw) ID3D12PipelineState_Release(cull_front_ccw);
   if (cull_front) ID3D12PipelineState_Release(cull_front);
   if (cull_back) ID3D12PipelineState_Release(cull_back);
   if (cull_none) ID3D12PipelineState_Release(cull_none);
   if (pso_slope_cw) ID3D12PipelineState_Release(pso_slope_cw);
   if (pso_slope) ID3D12PipelineState_Release(pso_slope);
   if (pso_nods) ID3D12PipelineState_Release(pso_nods);
   if (pso_nowrite) ID3D12PipelineState_Release(pso_nowrite);
   if (pso_write) ID3D12PipelineState_Release(pso_write);
   if (pso_greater) ID3D12PipelineState_Release(pso_greater);
   if (pso_less) ID3D12PipelineState_Release(pso_less);
   if (c.list) ID3D12GraphicsCommandList_Release(c.list);
   if (c.alloc) ID3D12CommandAllocator_Release(c.alloc);
   if (c.fence) ID3D12Fence_Release(c.fence);
   if (c.queue) ID3D12CommandQueue_Release(c.queue);
   if (dsv_heap) ID3D12DescriptorHeap_Release(dsv_heap);
   if (rtv_heap) ID3D12DescriptorHeap_Release(rtv_heap);
   if (c.ds) ID3D12Resource_Release(c.ds);
   if (c.rt) ID3D12Resource_Release(c.rt);
   if (c.root) ID3D12RootSignature_Release(c.root);
   if (c.dev) ID3D12Device_Release(c.dev);
   free(vs); free(ps); free(vs_depth); free(ps_green); free(vs_depth_cw);
#ifdef D12_D32_FIRST_PAIR_PROBE
   bool failed = failures || c.blocked_stage || !pair_complete;
   printf(failed ? "\nD32 FIRST PAIR: FAIL\n" :
          "\nD32 FIRST PAIR: PASS ACTUAL_PIXELS_PARTIAL_NOT_FULL_G5\n");
   return failed;
#else
   printf(failures || c.blocked_stage ? "\nG5 DEPTH: %d FAILURES\n" : "\nG5 DEPTH: ALL PASS\n", failures);
   return failures || c.blocked_stage;
#endif
}
