/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * d3d12agx frame-level benchmark.
 *
 * The microbenchmark prices single API calls. This program prices whole
 * frames, in the shape the DirectX-Graphics-Samples "Hello" programs issue
 * them: FrameCount allocators in rotation, one command list, a fence
 * signalled after every ExecuteCommandLists and waited on before an
 * allocator is reused. Rendering is off-screen (no swapchain exists on this
 * harness), so a "frame" ends with the render target transitioned to
 * COMMON, where Present would read it.
 *
 * The D3D12 implementation is loaded with dlopen so the same binary runs
 * against d3d12agx and against a native vkd3d-proton build:
 *
 *   D3D12_LIBRARY=/path/libvkd3d-proton-d3d12.so d3d12agx_framebench SHADERS
 *
 * SHADERS is the tests/shaders directory (tri, solid, vb_vs, vb_ps,
 * res_probe, tex_probe .dxil). Options: --frames N (timed frames per scene,
 * default 200), --scene NAME (one scene), --width W --height H (default
 * 1280x720, HelloWindow's size), --inflight N (frames in flight, default 2,
 * the samples' FrameCount; up to 8), --draws N (draws per frame in the
 * draws scenes, default 1000), --pso-vary MASK (what the draws_pso
 * pipelines vary: 1 cull mode, 2 blending, default 3), --nodepth (draws
 * scenes without the depth buffer).
 *
 * Scenes (every one clears the colour target first):
 *   triangle    HelloTriangle: one vertex-buffer draw
 *   bundles     HelloBundles: the same draw replayed from a bundle
 *   texload     a Texture.Load dispatch over a 2D texture, then the triangle
 *   draws       1000 draws of a triangle a few pixels across from a vertex
 *               buffer, one pipeline state, over a cleared and tested D32
 *               depth buffer (the per-draw cost, not the fill)
 *   draws_rc    the same with a root constant set before every draw
 *   draws_pso   the same with a root constant and four pipeline states
 *   fill        100 full-screen triangles over the depth buffer (fill rate,
 *               for parity: the cost here is the GPU's, not the driver's)
 *   dispatches  64 dispatches with a UAV barrier between each
 *
 * Per scene two loops run: "pipelined" keeps the in-flight count of frames going,
 * which is the throughput an application sees; "serial" waits for every
 * frame before recording the next, which is the latency of one frame on
 * the GPU plus the submission path. CPU record and ExecuteCommandLists
 * times are medians over the pipelined loop. The last line is
 * FRAMEBENCH_JSON with every number, for a collector.
 */

#define COBJMACROS
#include <wsl/winadapter.h>
#include <directx/d3d12.h>

#include "d12_test_abi.h"

#include <dlfcn.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define FRAME_COUNT_MAX 8
#define FRAME_COUNT_DEFAULT 2
#define WARMUP_FRAMES 16
#define DRAW_SCENE_DRAWS_DEFAULT 1000
#define DRAW_SCENE_PSOS 4
#define FILL_SCENE_DRAWS 100
#define DISPATCH_SCENE_DISPATCHES 64
#define TEX_W 16
#define TEX_H 16

typedef HRESULT(WINAPI *pfn_create_device_t)(IUnknown *, D3D_FEATURE_LEVEL,
                                             REFIID, void **);
typedef HRESULT(WINAPI *pfn_serialize_rs_t)(const D3D12_ROOT_SIGNATURE_DESC *,
                                            D3D_ROOT_SIGNATURE_VERSION,
                                            ID3DBlob **, ID3DBlob **);

static unsigned draw_scene_draws = DRAW_SCENE_DRAWS_DEFAULT;
static bool draw_scene_depth = true;
/* Which state the draws_pso pipelines vary: bit 0 cull mode, bit 1 blending. */
static unsigned draw_pso_vary = 3;
static pfn_create_device_t pfn_create_device;
static pfn_serialize_rs_t pfn_serialize_rs;

/* vkd3d-proton's native build keeps the MSVC calling convention for methods
 * that return an aggregate (the caller passes the return slot), where
 * DirectX-Headers returns it in registers on this platform. Set from
 * D3D12_ABI=explicit, or when the library name contains "vkd3d". */
static bool abi_explicit_return;

typedef void(STDMETHODCALLTYPE *pfn_cpu_start_explicit_t)(
   ID3D12DescriptorHeap *, D3D12_CPU_DESCRIPTOR_HANDLE *);
typedef void(STDMETHODCALLTYPE *pfn_gpu_start_explicit_t)(
   ID3D12DescriptorHeap *, D3D12_GPU_DESCRIPTOR_HANDLE *);

static D3D12_CPU_DESCRIPTOR_HANDLE
heap_cpu_start(ID3D12DescriptorHeap *heap)
{
   if (abi_explicit_return) {
      D3D12_CPU_DESCRIPTOR_HANDLE h = {0};
      ((pfn_cpu_start_explicit_t)
          heap->lpVtbl->GetCPUDescriptorHandleForHeapStart)(heap, &h);
      return h;
   }
   return d12_test_descriptor_heap_get_cpu_start(heap);
}

static D3D12_GPU_DESCRIPTOR_HANDLE
heap_gpu_start(ID3D12DescriptorHeap *heap)
{
   if (abi_explicit_return) {
      D3D12_GPU_DESCRIPTOR_HANDLE h = {0};
      ((pfn_gpu_start_explicit_t)
          heap->lpVtbl->GetGPUDescriptorHandleForHeapStart)(heap, &h);
      return h;
   }
   return d12_test_descriptor_heap_get_gpu_start(heap);
}

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
median(double *v, unsigned n)
{
   if (!n)
      return 0;
   qsort(v, n, sizeof(*v), cmp_double);
   return v[n / 2];
}

static void *
slurp(const char *dir, const char *name, size_t *size)
{
   char path[1024];
   snprintf(path, sizeof(path), "%s/%s", dir, name);
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

static unsigned failures;

#define CHECK(cond, what)                                                    \
   do {                                                                      \
      if (!(cond)) {                                                         \
         printf("FAIL - %s\n", what);                                        \
         failures++;                                                         \
         return false;                                                       \
      }                                                                      \
   } while (0)

struct scene;

struct fb {
   ID3D12Device *dev;
   ID3D12CommandQueue *queue;
   ID3D12Fence *fence;
   UINT64 fence_value;
   UINT64 frame_fence[FRAME_COUNT_MAX];
   ID3D12CommandAllocator *alloc[FRAME_COUNT_MAX];
   ID3D12GraphicsCommandList *list;
   unsigned frame_index;
   unsigned frame_count;

   unsigned width, height;
   ID3D12Resource *rt, *ds;
   ID3D12DescriptorHeap *rtv_heap, *dsv_heap;
   D3D12_CPU_DESCRIPTOR_HANDLE rtv, dsv;

   /* triangle / bundles */
   ID3D12RootSignature *rs_empty;
   ID3D12PipelineState *pso_vb;
   ID3D12Resource *vb, *vb_small;
   D3D12_VERTEX_BUFFER_VIEW vbv, vbv_small;
   ID3D12CommandAllocator *bundle_alloc;
   ID3D12GraphicsCommandList *bundle;

   /* draws, fill */
   ID3D12RootSignature *rs_consts;
   ID3D12PipelineState *pso_draws[DRAW_SCENE_PSOS];
   ID3D12PipelineState *pso_fill;

   /* dispatches */
   ID3D12RootSignature *rs_table;
   ID3D12PipelineState *pso_cs;
   ID3D12DescriptorHeap *cbv_heap;
   D3D12_GPU_DESCRIPTOR_HANDLE table;
   ID3D12Resource *src, *dst, *cb;

   /* texload */
   ID3D12RootSignature *rs_tex;
   ID3D12PipelineState *pso_tex;
   ID3D12Resource *tex, *tex_out;
   D3D12_GPU_DESCRIPTOR_HANDLE tex_table;
};

static ID3D12RootSignature *
make_root_sig(struct fb *b, const D3D12_ROOT_PARAMETER *params, UINT count)
{
   D3D12_ROOT_SIGNATURE_DESC rsd = {
      .NumParameters = count,
      .pParameters = params,
      .Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
   };
   ID3DBlob *blob = NULL;
   if (pfn_serialize_rs(&rsd, D3D_ROOT_SIGNATURE_VERSION_1_0, &blob, NULL) !=
       S_OK)
      return NULL;
   ID3D12RootSignature *rs = NULL;
   HRESULT hr = ID3D12Device_CreateRootSignature(
      b->dev, 0, ID3D10Blob_GetBufferPointer(blob),
      ID3D10Blob_GetBufferSize(blob), &IID_ID3D12RootSignature,
      (void **)&rs);
   ID3D10Blob_Release(blob);
   return hr == S_OK ? rs : NULL;
}

static ID3D12Resource *
make_buffer(ID3D12Device *dev, UINT64 size, D3D12_RESOURCE_FLAGS flags,
            D3D12_HEAP_TYPE heap_type)
{
   D3D12_HEAP_PROPERTIES heap = {.Type = heap_type};
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
      heap_type == D3D12_HEAP_TYPE_UPLOAD ? D3D12_RESOURCE_STATE_GENERIC_READ
                                          : D3D12_RESOURCE_STATE_COMMON,
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

static void
uav_barrier(ID3D12GraphicsCommandList *list, ID3D12Resource *resource)
{
   D3D12_RESOURCE_BARRIER barrier = {
      .Type = D3D12_RESOURCE_BARRIER_TYPE_UAV,
      .UAV = {.pResource = resource},
   };
   ID3D12GraphicsCommandList_ResourceBarrier(list, 1, &barrier);
}

static HRESULT
wait_fence(struct fb *b, UINT64 value)
{
   if (ID3D12Fence_GetCompletedValue(b->fence) >= value)
      return S_OK;
   HRESULT hr = ID3D12Fence_SetEventOnCompletion(b->fence, value, NULL);
   if (hr != S_OK)
      return hr;
   if (ID3D12Fence_GetCompletedValue(b->fence) < value)
      return E_FAIL;
   return S_OK;
}

static HRESULT
signal_frame(struct fb *b)
{
   b->fence_value++;
   b->frame_fence[b->frame_index] = b->fence_value;
   return ID3D12CommandQueue_Signal(b->queue, b->fence, b->fence_value);
}

static HRESULT
wait_idle(struct fb *b)
{
   b->fence_value++;
   HRESULT hr = ID3D12CommandQueue_Signal(b->queue, b->fence, b->fence_value);
   if (hr != S_OK)
      return hr;
   hr = wait_fence(b, b->fence_value);
   if (hr != S_OK)
      return hr;
   return ID3D12Device_GetDeviceRemovedReason(b->dev);
}

/* --- Scene setup ------------------------------------------------------ */

struct shaders {
   void *tri, *solid, *vb_vs, *vb_ps, *cs, *tex_cs;
   size_t tri_size, solid_size, vb_vs_size, vb_ps_size, cs_size, tex_cs_size;
};

static D3D12_GRAPHICS_PIPELINE_STATE_DESC
base_gfx_desc(ID3D12RootSignature *rs, const void *vs, size_t vs_size,
              const void *ps, size_t ps_size)
{
   D3D12_GRAPHICS_PIPELINE_STATE_DESC pd = {
      .pRootSignature = rs,
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
   return pd;
}

static bool
setup_targets(struct fb *b)
{
   D3D12_HEAP_PROPERTIES heap = {.Type = D3D12_HEAP_TYPE_DEFAULT};
   /* The application default: a render target with no other flags, as
    * HelloWindow's swapchain buffers and every intermediate target. */
   D3D12_RESOURCE_DESC td = {
      .Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
      .Width = b->width,
      .Height = b->height,
      .DepthOrArraySize = 1,
      .MipLevels = 1,
      .Format = DXGI_FORMAT_R8G8B8A8_UNORM,
      .SampleDesc = {.Count = 1},
      .Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN,
      .Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
   };
   D3D12_CLEAR_VALUE rt_clear = {.Format = DXGI_FORMAT_R8G8B8A8_UNORM,
                                 .Color = {0.0f, 0.2f, 0.4f, 1.0f}};
   HRESULT hr = ID3D12Device_CreateCommittedResource(
      b->dev, &heap, D3D12_HEAP_FLAG_NONE, &td, D3D12_RESOURCE_STATE_COMMON,
      &rt_clear, &IID_ID3D12Resource, (void **)&b->rt);
   CHECK(hr == S_OK, "create colour target");

   td.Format = DXGI_FORMAT_D32_FLOAT;
   td.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
   D3D12_CLEAR_VALUE ds_clear = {.Format = DXGI_FORMAT_D32_FLOAT,
                                 .DepthStencil = {.Depth = 1.0f}};
   hr = ID3D12Device_CreateCommittedResource(
      b->dev, &heap, D3D12_HEAP_FLAG_NONE, &td,
      D3D12_RESOURCE_STATE_DEPTH_WRITE, &ds_clear, &IID_ID3D12Resource,
      (void **)&b->ds);
   CHECK(hr == S_OK, "create depth target");

   D3D12_DESCRIPTOR_HEAP_DESC hd = {
      .Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV, .NumDescriptors = 1};
   hr = ID3D12Device_CreateDescriptorHeap(b->dev, &hd, &IID_ID3D12DescriptorHeap,
                                          (void **)&b->rtv_heap);
   CHECK(hr == S_OK, "create RTV heap");
   hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
   hr = ID3D12Device_CreateDescriptorHeap(b->dev, &hd, &IID_ID3D12DescriptorHeap,
                                          (void **)&b->dsv_heap);
   CHECK(hr == S_OK, "create DSV heap");
   b->rtv = heap_cpu_start(b->rtv_heap);
   b->dsv = heap_cpu_start(b->dsv_heap);
   ID3D12Device_CreateRenderTargetView(b->dev, b->rt, NULL, b->rtv);
   ID3D12Device_CreateDepthStencilView(b->dev, b->ds, NULL, b->dsv);
   return true;
}

static bool
setup_triangle(struct fb *b, const struct shaders *s)
{
   b->rs_empty = make_root_sig(b, NULL, 0);
   CHECK(b->rs_empty, "empty root signature");

   /* vb_vs declares TEXCOORD0 (position, float2) and TEXCOORD1 (colour,
    * float4) on input slot 31, 24 bytes per vertex (g10_vertex_buffers). */
   D3D12_INPUT_ELEMENT_DESC elems[] = {
      {.SemanticName = "TEXCOORD",
       .SemanticIndex = 0,
       .Format = DXGI_FORMAT_R32G32_FLOAT,
       .InputSlot = 31,
       .AlignedByteOffset = 0,
       .InputSlotClass = D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA},
      {.SemanticName = "TEXCOORD",
       .SemanticIndex = 1,
       .Format = DXGI_FORMAT_R32G32B32A32_FLOAT,
       .InputSlot = 31,
       .AlignedByteOffset = 8,
       .InputSlotClass = D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA},
   };
   D3D12_GRAPHICS_PIPELINE_STATE_DESC pd = base_gfx_desc(
      b->rs_empty, s->vb_vs, s->vb_vs_size, s->vb_ps, s->vb_ps_size);
   pd.InputLayout = (D3D12_INPUT_LAYOUT_DESC){elems, 2};
   HRESULT hr = ID3D12Device_CreateGraphicsPipelineState(
      b->dev, &pd, &IID_ID3D12PipelineState, (void **)&b->pso_vb);
   CHECK(hr == S_OK, "vertex-buffer PSO");

   const float verts[3][6] = {
      {0.0f, 0.5f, 1.0f, 0.0f, 0.0f, 1.0f},
      {0.5f, -0.5f, 0.0f, 1.0f, 0.0f, 1.0f},
      {-0.5f, -0.5f, 0.0f, 0.0f, 1.0f, 1.0f},
   };
   b->vb = make_buffer(b->dev, sizeof(verts), D3D12_RESOURCE_FLAG_NONE,
                       D3D12_HEAP_TYPE_UPLOAD);
   CHECK(b->vb, "vertex buffer");
   void *map = NULL;
   D3D12_RANGE none = {0, 0};
   hr = ID3D12Resource_Map(b->vb, 0, &none, &map);
   CHECK(hr == S_OK && map, "map vertex buffer");
   memcpy(map, verts, sizeof(verts));
   ID3D12Resource_Unmap(b->vb, 0, NULL);
   b->vbv = (D3D12_VERTEX_BUFFER_VIEW){
      .BufferLocation = ID3D12Resource_GetGPUVirtualAddress(b->vb),
      .SizeInBytes = sizeof(verts),
      .StrideInBytes = 24,
   };

   /* The draws scenes price the draw, not the fill: a triangle a few
    * pixels across. */
   float small[3][6];
   memcpy(small, verts, sizeof(small));
   for (unsigned v = 0; v < 3; v++) {
      small[v][0] *= 0.02f;
      small[v][1] *= 0.02f;
   }
   b->vb_small = make_buffer(b->dev, sizeof(small), D3D12_RESOURCE_FLAG_NONE,
                             D3D12_HEAP_TYPE_UPLOAD);
   CHECK(b->vb_small, "small vertex buffer");
   hr = ID3D12Resource_Map(b->vb_small, 0, &none, &map);
   CHECK(hr == S_OK && map, "map small vertex buffer");
   memcpy(map, small, sizeof(small));
   ID3D12Resource_Unmap(b->vb_small, 0, NULL);
   b->vbv_small = b->vbv;
   b->vbv_small.BufferLocation =
      ID3D12Resource_GetGPUVirtualAddress(b->vb_small);

   /* HelloBundles: the draw recorded once into a bundle. */
   hr = ID3D12Device_CreateCommandAllocator(
      b->dev, D3D12_COMMAND_LIST_TYPE_BUNDLE, &IID_ID3D12CommandAllocator,
      (void **)&b->bundle_alloc);
   CHECK(hr == S_OK, "bundle allocator");
   hr = ID3D12Device_CreateCommandList(
      b->dev, 0, D3D12_COMMAND_LIST_TYPE_BUNDLE, b->bundle_alloc, b->pso_vb,
      &IID_ID3D12GraphicsCommandList, (void **)&b->bundle);
   CHECK(hr == S_OK, "bundle");
   ID3D12GraphicsCommandList_SetGraphicsRootSignature(b->bundle, b->rs_empty);
   ID3D12GraphicsCommandList_IASetPrimitiveTopology(
      b->bundle, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
   ID3D12GraphicsCommandList_IASetVertexBuffers(b->bundle, 31, 1, &b->vbv);
   ID3D12GraphicsCommandList_DrawInstanced(b->bundle, 3, 1, 0, 0);
   hr = ID3D12GraphicsCommandList_Close(b->bundle);
   CHECK(hr == S_OK, "close bundle");
   return true;
}

static bool
setup_draws(struct fb *b, const struct shaders *s)
{
   D3D12_ROOT_PARAMETER consts = {
      .ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
      .Constants = {.ShaderRegister = 0, .Num32BitValues = 4},
      .ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL,
   };
   b->rs_consts = make_root_sig(b, &consts, 1);
   CHECK(b->rs_consts, "root-constant root signature");

   /* The draws scene uses the small vertex-buffer triangle: tri.dxil's
    * SV_VertexID triangle covers the whole viewport, which makes 1000 of
    * them a fill-rate test (the fill scene) rather than a draw-cost one. */
   D3D12_INPUT_ELEMENT_DESC elems[] = {
      {.SemanticName = "TEXCOORD",
       .SemanticIndex = 0,
       .Format = DXGI_FORMAT_R32G32_FLOAT,
       .InputSlot = 31,
       .AlignedByteOffset = 0,
       .InputSlotClass = D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA},
      {.SemanticName = "TEXCOORD",
       .SemanticIndex = 1,
       .Format = DXGI_FORMAT_R32G32B32A32_FLOAT,
       .InputSlot = 31,
       .AlignedByteOffset = 8,
       .InputSlotClass = D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA},
   };
   for (unsigned i = 0; i < DRAW_SCENE_PSOS; i++) {
      D3D12_GRAPHICS_PIPELINE_STATE_DESC pd = base_gfx_desc(
         b->rs_consts, s->vb_vs, s->vb_vs_size, s->vb_ps, s->vb_ps_size);
      pd.InputLayout = (D3D12_INPUT_LAYOUT_DESC){elems, 2};
      if (draw_scene_depth) {
         pd.DSVFormat = DXGI_FORMAT_D32_FLOAT;
         pd.DepthStencilState.DepthEnable = TRUE;
         pd.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
         pd.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
      }
      /* Four distinct states an engine toggles between: cull mode and
       * blending. */
      pd.RasterizerState.CullMode =
         (i & draw_pso_vary & 1) ? D3D12_CULL_MODE_BACK : D3D12_CULL_MODE_NONE;
      if (i & draw_pso_vary & 2) {
         D3D12_RENDER_TARGET_BLEND_DESC *rt = &pd.BlendState.RenderTarget[0];
         rt->BlendEnable = TRUE;
         rt->SrcBlend = D3D12_BLEND_SRC_ALPHA;
         rt->DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
         rt->BlendOp = D3D12_BLEND_OP_ADD;
         rt->SrcBlendAlpha = D3D12_BLEND_ONE;
         rt->DestBlendAlpha = D3D12_BLEND_ZERO;
         rt->BlendOpAlpha = D3D12_BLEND_OP_ADD;
      }
      HRESULT hr = ID3D12Device_CreateGraphicsPipelineState(
         b->dev, &pd, &IID_ID3D12PipelineState, (void **)&b->pso_draws[i]);
      CHECK(hr == S_OK, "draws-scene PSO");
   }
   D3D12_GRAPHICS_PIPELINE_STATE_DESC fd = base_gfx_desc(
      b->rs_empty, s->tri, s->tri_size, s->solid, s->solid_size);
   fd.DSVFormat = DXGI_FORMAT_D32_FLOAT;
   fd.DepthStencilState.DepthEnable = TRUE;
   fd.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
   fd.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
   HRESULT hr = ID3D12Device_CreateGraphicsPipelineState(
      b->dev, &fd, &IID_ID3D12PipelineState, (void **)&b->pso_fill);
   CHECK(hr == S_OK, "fill-scene PSO");
   return true;
}

static bool
setup_dispatches(struct fb *b, const struct shaders *s)
{
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
   b->rs_table = make_root_sig(b, &table, 1);
   CHECK(b->rs_table, "descriptor-table root signature");

   D3D12_COMPUTE_PIPELINE_STATE_DESC cd = {
      .pRootSignature = b->rs_table,
      .CS = {.pShaderBytecode = s->cs, .BytecodeLength = s->cs_size},
   };
   HRESULT hr = ID3D12Device_CreateComputePipelineState(
      b->dev, &cd, &IID_ID3D12PipelineState, (void **)&b->pso_cs);
   CHECK(hr == S_OK, "compute PSO");

   b->src = make_buffer(b->dev, 64 * 12, D3D12_RESOURCE_FLAG_NONE,
                        D3D12_HEAP_TYPE_UPLOAD);
   b->dst = make_buffer(b->dev, 64 * 4,
                        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                        D3D12_HEAP_TYPE_DEFAULT);
   b->cb = make_buffer(b->dev, 256, D3D12_RESOURCE_FLAG_NONE,
                       D3D12_HEAP_TYPE_UPLOAD);
   CHECK(b->src && b->dst && b->cb, "compute buffers");

   D3D12_DESCRIPTOR_HEAP_DESC chd = {
      .Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
      .NumDescriptors = 8,
      .Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
   };
   hr = ID3D12Device_CreateDescriptorHeap(b->dev, &chd, &IID_ID3D12DescriptorHeap,
                                          (void **)&b->cbv_heap);
   CHECK(hr == S_OK, "shader-visible CBV/SRV/UAV heap");
   const UINT inc = ID3D12Device_GetDescriptorHandleIncrementSize(
      b->dev, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
   D3D12_CPU_DESCRIPTOR_HANDLE h =
      heap_cpu_start(b->cbv_heap);
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
      .BufferLocation = ID3D12Resource_GetGPUVirtualAddress(b->cb),
      .SizeInBytes = 256,
   };
   ID3D12Device_CreateShaderResourceView(b->dev, b->src, &srv, h);
   h.ptr += inc;
   ID3D12Device_CreateUnorderedAccessView(b->dev, b->dst, NULL, &uav, h);
   h.ptr += inc;
   ID3D12Device_CreateConstantBufferView(b->dev, &cbv, h);
   b->table = heap_gpu_start(b->cbv_heap);

   /* texload: tex_probe reads a Texture2D (t0) and writes a UAV (u0),
    * both through one table placed after the compute table. */
   if (s->tex_cs) {
      D3D12_DESCRIPTOR_RANGE tex_ranges[2] = {
         {.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV, .NumDescriptors = 1,
          .BaseShaderRegister = 0,
          .OffsetInDescriptorsFromTableStart =
             D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND},
         {.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV, .NumDescriptors = 1,
          .BaseShaderRegister = 0,
          .OffsetInDescriptorsFromTableStart =
             D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND},
      };
      D3D12_ROOT_PARAMETER tex_table = {
         .ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
         .DescriptorTable = {.NumDescriptorRanges = 2,
                             .pDescriptorRanges = tex_ranges},
         .ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL,
      };
      b->rs_tex = make_root_sig(b, &tex_table, 1);
      CHECK(b->rs_tex, "texture root signature");
      D3D12_COMPUTE_PIPELINE_STATE_DESC tcd = {
         .pRootSignature = b->rs_tex,
         .CS = {.pShaderBytecode = s->tex_cs, .BytecodeLength = s->tex_cs_size},
      };
      hr = ID3D12Device_CreateComputePipelineState(
         b->dev, &tcd, &IID_ID3D12PipelineState, (void **)&b->pso_tex);
      CHECK(hr == S_OK, "texture compute PSO");

      D3D12_HEAP_PROPERTIES def_heap = {.Type = D3D12_HEAP_TYPE_DEFAULT};
      D3D12_RESOURCE_DESC ttd = {
         .Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
         .Width = TEX_W,
         .Height = TEX_H,
         .DepthOrArraySize = 1,
         .MipLevels = 1,
         .Format = DXGI_FORMAT_R8G8B8A8_UNORM,
         .SampleDesc = {.Count = 1},
         .Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN,
      };
      hr = ID3D12Device_CreateCommittedResource(
         b->dev, &def_heap, D3D12_HEAP_FLAG_NONE, &ttd,
         D3D12_RESOURCE_STATE_COMMON, NULL, &IID_ID3D12Resource,
         (void **)&b->tex);
      CHECK(hr == S_OK, "create texture");
      b->tex_out = make_buffer(b->dev, TEX_W * TEX_H * 4,
                               D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                               D3D12_HEAP_TYPE_DEFAULT);
      CHECK(b->tex_out, "texture-probe output");
      D3D12_SHADER_RESOURCE_VIEW_DESC tsrv = {
         .Format = DXGI_FORMAT_R8G8B8A8_UNORM,
         .ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D,
         .Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
         .Texture2D = {.MipLevels = 1},
      };
      D3D12_UNORDERED_ACCESS_VIEW_DESC tuav = {
         .Format = DXGI_FORMAT_UNKNOWN,
         .ViewDimension = D3D12_UAV_DIMENSION_BUFFER,
         .Buffer = {.NumElements = TEX_W * TEX_H, .StructureByteStride = 4},
      };
      h.ptr += inc;
      D3D12_CPU_DESCRIPTOR_HANDLE th = h;
      ID3D12Device_CreateShaderResourceView(b->dev, b->tex, &tsrv, th);
      th.ptr += inc;
      ID3D12Device_CreateUnorderedAccessView(b->dev, b->tex_out, NULL, &tuav,
                                             th);
      b->tex_table.ptr = b->table.ptr + 3 * inc;
   }
   return true;
}

/* --- Frame recording -------------------------------------------------- */

enum scene_id {
   SCENE_TRIANGLE,
   SCENE_BUNDLES,
   SCENE_TEXLOAD,
   SCENE_DRAWS,
   SCENE_DRAWS_RC,
   SCENE_DRAWS_PSO,
   SCENE_FILL,
   SCENE_DISPATCHES,
   SCENE_COUNT,
};

static const char *const scene_names[SCENE_COUNT] = {
   "triangle", "bundles", "texload", "draws", "draws_rc", "draws_pso",
   "fill", "dispatches",
};

static void
begin_frame_graphics(struct fb *b, bool depth)
{
   ID3D12GraphicsCommandList *list = b->list;
   transition(list, b->rt, D3D12_RESOURCE_STATE_COMMON,
              D3D12_RESOURCE_STATE_RENDER_TARGET);
   const FLOAT clear[4] = {0.0f, 0.2f, 0.4f, 1.0f};
   ID3D12GraphicsCommandList_ClearRenderTargetView(list, b->rtv, clear, 0,
                                                   NULL);
   if (depth)
      ID3D12GraphicsCommandList_ClearDepthStencilView(
         list, b->dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, NULL);
   ID3D12GraphicsCommandList_OMSetRenderTargets(list, 1, &b->rtv, FALSE,
                                                depth ? &b->dsv : NULL);
   D3D12_VIEWPORT vp = {
      .Width = (float)b->width, .Height = (float)b->height, .MaxDepth = 1.0f};
   ID3D12GraphicsCommandList_RSSetViewports(list, 1, &vp);
   D3D12_RECT sc = {.right = (LONG)b->width, .bottom = (LONG)b->height};
   ID3D12GraphicsCommandList_RSSetScissorRects(list, 1, &sc);
}

static void
end_frame_graphics(struct fb *b)
{
   transition(b->list, b->rt, D3D12_RESOURCE_STATE_RENDER_TARGET,
              D3D12_RESOURCE_STATE_COMMON);
}

static void
record_triangle(struct fb *b)
{
   ID3D12GraphicsCommandList *list = b->list;
   begin_frame_graphics(b, false);
   ID3D12GraphicsCommandList_SetGraphicsRootSignature(list, b->rs_empty);
   ID3D12GraphicsCommandList_SetPipelineState(list, b->pso_vb);
   ID3D12GraphicsCommandList_IASetPrimitiveTopology(
      list, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
   ID3D12GraphicsCommandList_IASetVertexBuffers(list, 31, 1, &b->vbv);
   ID3D12GraphicsCommandList_DrawInstanced(list, 3, 1, 0, 0);
   end_frame_graphics(b);
}

static void
record_bundles(struct fb *b)
{
   ID3D12GraphicsCommandList *list = b->list;
   begin_frame_graphics(b, false);
   ID3D12GraphicsCommandList_SetGraphicsRootSignature(list, b->rs_empty);
   ID3D12GraphicsCommandList_ExecuteBundle(list, b->bundle);
   end_frame_graphics(b);
}

static void
record_texload(struct fb *b)
{
   ID3D12GraphicsCommandList *list = b->list;
   ID3D12GraphicsCommandList_SetDescriptorHeaps(list, 1, &b->cbv_heap);
   ID3D12GraphicsCommandList_SetComputeRootSignature(list, b->rs_tex);
   ID3D12GraphicsCommandList_SetPipelineState(list, b->pso_tex);
   ID3D12GraphicsCommandList_SetComputeRootDescriptorTable(list, 0,
                                                           b->tex_table);
   ID3D12GraphicsCommandList_Dispatch(list, (TEX_W * TEX_H) / 64, 1, 1);
   uav_barrier(list, b->tex_out);
   record_triangle(b);
}

static void
record_draws(struct fb *b, bool root_constants, bool pso_switches)
{
   ID3D12GraphicsCommandList *list = b->list;
   begin_frame_graphics(b, draw_scene_depth);
   ID3D12GraphicsCommandList_SetGraphicsRootSignature(list, b->rs_consts);
   ID3D12GraphicsCommandList_SetPipelineState(list, b->pso_draws[0]);
   ID3D12GraphicsCommandList_IASetPrimitiveTopology(
      list, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
   ID3D12GraphicsCommandList_IASetVertexBuffers(list, 31, 1, &b->vbv_small);
   const unsigned per_pso = (draw_scene_draws + DRAW_SCENE_PSOS - 1) / DRAW_SCENE_PSOS;
   for (unsigned i = 0; i < draw_scene_draws; i++) {
      if (pso_switches && i % per_pso == 0)
         ID3D12GraphicsCommandList_SetPipelineState(
            list, b->pso_draws[(i / per_pso) % DRAW_SCENE_PSOS]);
      if (root_constants)
         ID3D12GraphicsCommandList_SetGraphicsRoot32BitConstant(list, 0, i,
                                                                0);
      ID3D12GraphicsCommandList_DrawInstanced(list, 3, 1, 0, 0);
   }
   end_frame_graphics(b);
}

static void
record_fill(struct fb *b)
{
   ID3D12GraphicsCommandList *list = b->list;
   begin_frame_graphics(b, true);
   ID3D12GraphicsCommandList_SetGraphicsRootSignature(list, b->rs_empty);
   ID3D12GraphicsCommandList_SetPipelineState(list, b->pso_fill);
   ID3D12GraphicsCommandList_IASetPrimitiveTopology(
      list, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
   for (unsigned i = 0; i < FILL_SCENE_DRAWS; i++)
      ID3D12GraphicsCommandList_DrawInstanced(list, 3, 1, 0, 0);
   end_frame_graphics(b);
}

static void
record_dispatches(struct fb *b)
{
   ID3D12GraphicsCommandList *list = b->list;
   ID3D12GraphicsCommandList_SetDescriptorHeaps(list, 1, &b->cbv_heap);
   ID3D12GraphicsCommandList_SetComputeRootSignature(list, b->rs_table);
   ID3D12GraphicsCommandList_SetPipelineState(list, b->pso_cs);
   ID3D12GraphicsCommandList_SetComputeRootDescriptorTable(list, 0, b->table);
   for (unsigned i = 0; i < DISPATCH_SCENE_DISPATCHES; i++) {
      ID3D12GraphicsCommandList_Dispatch(list, 1, 1, 1);
      uav_barrier(list, b->dst);
   }
}

static void
record_scene(struct fb *b, enum scene_id scene)
{
   switch (scene) {
   case SCENE_TRIANGLE:
      record_triangle(b);
      break;
   case SCENE_BUNDLES:
      record_bundles(b);
      break;
   case SCENE_TEXLOAD:
      record_texload(b);
      break;
   case SCENE_DRAWS:
      record_draws(b, false, false);
      break;
   case SCENE_DRAWS_RC:
      record_draws(b, true, false);
      break;
   case SCENE_DRAWS_PSO:
      record_draws(b, true, true);
      break;
   case SCENE_FILL:
      record_fill(b);
      break;
   case SCENE_DISPATCHES:
      record_dispatches(b);
      break;
   default:
      break;
   }
}

struct scene_result {
   bool ok;
   double pipelined_us; /* wall per frame, frame_count in flight */
   double serial_us;    /* wall per frame, one in flight */
   double record_us;    /* median Reset..Close */
   double execute_us;   /* median ExecuteCommandLists call */
};

/* One frame in the samples' shape. Returns false on a failed Close, Execute
 * or fence. */
static bool
run_frame(struct fb *b, enum scene_id scene, double *record_us,
          double *execute_us)
{
   ID3D12CommandAllocator *alloc = b->alloc[b->frame_index];
   if (wait_fence(b, b->frame_fence[b->frame_index]) != S_OK)
      return false;
   double t0 = now_ns();
   if (ID3D12CommandAllocator_Reset(alloc) != S_OK)
      return false;
   if (ID3D12GraphicsCommandList_Reset(b->list, alloc, NULL) != S_OK)
      return false;
   record_scene(b, scene);
   if (ID3D12GraphicsCommandList_Close(b->list) != S_OK)
      return false;
   double t1 = now_ns();
   ID3D12CommandList *lists[] = {(ID3D12CommandList *)b->list};
   ID3D12CommandQueue_ExecuteCommandLists(b->queue, 1, lists);
   double t2 = now_ns();
   if (signal_frame(b) != S_OK)
      return false;
   b->frame_index = (b->frame_index + 1) % b->frame_count;
   *record_us = (t1 - t0) / 1e3;
   *execute_us = (t2 - t1) / 1e3;
   return true;
}

/* A list whose Close failed stays errored across Reset on some
 * implementations: after a failed scene, idle the queue and record the
 * next scene on a fresh list. */
static bool
replace_list(struct fb *b)
{
   wait_idle(b);
   ID3D12GraphicsCommandList_Release(b->list);
   b->list = NULL;
   if (ID3D12Device_CreateCommandList(
          b->dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, b->alloc[b->frame_index],
          NULL, &IID_ID3D12GraphicsCommandList, (void **)&b->list) != S_OK)
      return false;
   ID3D12GraphicsCommandList_Close(b->list);
   return true;
}

static struct scene_result
run_scene(struct fb *b, enum scene_id scene, unsigned frames)
{
   struct scene_result r = {0};
   double *rec = calloc(frames, sizeof(double));
   double *exe = calloc(frames, sizeof(double));
   if (!rec || !exe) {
      free(rec);
      free(exe);
      return r;
   }
   double a, z;

   /* Warm-up: compiles, allocator growth, first-touch pages. */
   for (unsigned f = 0; f < WARMUP_FRAMES; f++) {
      if (!run_frame(b, scene, &a, &z)) {
         printf("       %s: frame %u failed during warm-up\n",
                scene_names[scene], f);
         goto out;
      }
   }
   if (wait_idle(b) != S_OK) {
      printf("       %s: device lost after warm-up\n", scene_names[scene]);
      goto out;
   }

   double t0 = now_ns();
   for (unsigned f = 0; f < frames; f++) {
      if (!run_frame(b, scene, &rec[f], &exe[f])) {
         printf("       %s: frame %u failed\n", scene_names[scene], f);
         goto out;
      }
   }
   if (wait_idle(b) != S_OK) {
      printf("       %s: device lost\n", scene_names[scene]);
      goto out;
   }
   double t1 = now_ns();
   r.pipelined_us = (t1 - t0) / 1e3 / frames;
   r.record_us = median(rec, frames);
   r.execute_us = median(exe, frames);

   t0 = now_ns();
   for (unsigned f = 0; f < frames; f++) {
      if (!run_frame(b, scene, &a, &z) || wait_fence(b, b->fence_value) != S_OK) {
         printf("       %s: serial frame %u failed\n", scene_names[scene], f);
         goto out;
      }
   }
   t1 = now_ns();
   if (ID3D12Device_GetDeviceRemovedReason(b->dev) != S_OK) {
      printf("       %s: device removed\n", scene_names[scene]);
      goto out;
   }
   r.serial_us = (t1 - t0) / 1e3 / frames;
   r.ok = true;
out:
   free(rec);
   free(exe);
   return r;
}

int
main(int argc, char **argv)
{
   const char *shader_dir = NULL;
   const char *only = NULL;
   unsigned frames = 200;
   unsigned width = 1280, height = 720;
   unsigned inflight = FRAME_COUNT_DEFAULT;
   for (int i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "--frames") && i + 1 < argc)
         frames = (unsigned)atoi(argv[++i]);
      else if (!strcmp(argv[i], "--scene") && i + 1 < argc)
         only = argv[++i];
      else if (!strcmp(argv[i], "--width") && i + 1 < argc)
         width = (unsigned)atoi(argv[++i]);
      else if (!strcmp(argv[i], "--height") && i + 1 < argc)
         height = (unsigned)atoi(argv[++i]);
      else if (!strcmp(argv[i], "--inflight") && i + 1 < argc)
         inflight = (unsigned)atoi(argv[++i]);
      else if (!strcmp(argv[i], "--draws") && i + 1 < argc)
         draw_scene_draws = (unsigned)atoi(argv[++i]);
      else if (!strcmp(argv[i], "--nodepth"))
         draw_scene_depth = false;
      else if (!strcmp(argv[i], "--pso-vary") && i + 1 < argc)
         draw_pso_vary = (unsigned)atoi(argv[++i]) & 3;
      else
         shader_dir = argv[i];
   }
   if (!shader_dir || !frames || !width || !height || !inflight ||
       inflight > FRAME_COUNT_MAX || !draw_scene_draws) {
      fprintf(stderr,
              "usage: %s SHADER_DIR [--frames N] [--scene NAME] "
              "[--width W --height H] [--inflight N] [--draws N] [--nodepth] [--pso-vary MASK]\n"
              "       D3D12_LIBRARY selects the implementation "
              "(default libd3d12agx.so)\n",
              argv[0]);
      return 2;
   }

   const char *library = getenv("D3D12_LIBRARY");
   if (!library || !*library)
      library = "libd3d12agx.so";
   const char *abi = getenv("D3D12_ABI");
   abi_explicit_return =
      (abi && !strcmp(abi, "explicit")) || (!abi && strstr(library, "vkd3d"));
   void *dl = dlopen(library, RTLD_NOW | RTLD_LOCAL);
   if (!dl) {
      printf("FAIL - dlopen %s: %s\n", library, dlerror());
      printf("FRAMEBENCH_JSON {\"error\": \"dlopen\"}\n");
      return 1;
   }
   pfn_create_device = (pfn_create_device_t)dlsym(dl, "D3D12CreateDevice");
   pfn_serialize_rs =
      (pfn_serialize_rs_t)dlsym(dl, "D3D12SerializeRootSignature");
   if (!pfn_create_device || !pfn_serialize_rs) {
      printf("FAIL - %s exports no D3D12CreateDevice/"
             "D3D12SerializeRootSignature\n",
             library);
      printf("FRAMEBENCH_JSON {\"error\": \"exports\"}\n");
      return 1;
   }

   struct shaders s = {0};
   s.tri = slurp(shader_dir, "tri.dxil", &s.tri_size);
   s.solid = slurp(shader_dir, "solid.dxil", &s.solid_size);
   s.vb_vs = slurp(shader_dir, "vb_vs.dxil", &s.vb_vs_size);
   s.vb_ps = slurp(shader_dir, "vb_ps.dxil", &s.vb_ps_size);
   s.cs = slurp(shader_dir, "res_probe.dxil", &s.cs_size);
   s.tex_cs = slurp(shader_dir, "tex_probe.dxil", &s.tex_cs_size);
   if (!s.tri || !s.solid || !s.vb_vs || !s.vb_ps || !s.cs) {
      printf("FAIL - shaders missing in %s\n", shader_dir);
      printf("FRAMEBENCH_JSON {\"error\": \"shaders\"}\n");
      return 1;
   }

   struct fb b = {.width = width, .height = height, .frame_count = inflight};
   HRESULT hr = pfn_create_device(NULL, D3D_FEATURE_LEVEL_11_0,
                                  &IID_ID3D12Device, (void **)&b.dev);
   if (hr != S_OK || !b.dev) {
      printf("FAIL - D3D12CreateDevice 0x%08x\n", (unsigned)hr);
      printf("FRAMEBENCH_JSON {\"error\": \"device\"}\n");
      return 1;
   }
   D3D12_COMMAND_QUEUE_DESC qd = {.Type = D3D12_COMMAND_LIST_TYPE_DIRECT};
   hr = ID3D12Device_CreateCommandQueue(b.dev, &qd, &IID_ID3D12CommandQueue,
                                        (void **)&b.queue);
   if (hr != S_OK) {
      printf("FAIL - CreateCommandQueue 0x%08x\n", (unsigned)hr);
      printf("FRAMEBENCH_JSON {\"error\": \"queue\"}\n");
      return 1;
   }
   hr = ID3D12Device_CreateFence(b.dev, 0, D3D12_FENCE_FLAG_NONE,
                                 &IID_ID3D12Fence, (void **)&b.fence);
   if (hr != S_OK) {
      printf("FAIL - CreateFence\n");
      return 1;
   }
   for (unsigned i = 0; i < inflight; i++) {
      hr = ID3D12Device_CreateCommandAllocator(
         b.dev, D3D12_COMMAND_LIST_TYPE_DIRECT, &IID_ID3D12CommandAllocator,
         (void **)&b.alloc[i]);
      if (hr != S_OK) {
         printf("FAIL - CreateCommandAllocator\n");
         return 1;
      }
   }
   hr = ID3D12Device_CreateCommandList(
      b.dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, b.alloc[0], NULL,
      &IID_ID3D12GraphicsCommandList, (void **)&b.list);
   if (hr != S_OK) {
      printf("FAIL - CreateCommandList\n");
      return 1;
   }
   ID3D12GraphicsCommandList_Close(b.list);

   if (!setup_targets(&b)) {
      printf("FRAMEBENCH_JSON {\"error\": \"setup\"}\n");
      return 1;
   }
   /* Each scene group sets up on its own, so a device that cannot create
    * one group's pipelines (the Neo's native G17P graphics admits no
    * vertex-buffer PSO yet) still measures the others. */
   const bool have_triangle = setup_triangle(&b, &s);
   const bool have_draws = setup_draws(&b, &s);
   const bool have_dispatches = setup_dispatches(&b, &s);

   printf("framebench %s, %ux%u, %u frames per scene, %u in flight\n",
          library, width, height, frames, inflight);
   struct scene_result results[SCENE_COUNT] = {0};
   for (unsigned sc = 0; sc < SCENE_COUNT; sc++) {
      if (only && strcmp(only, scene_names[sc]))
         continue;
      if (sc == SCENE_TEXLOAD && !b.pso_tex)
         continue;
      const bool supported =
         sc == SCENE_TRIANGLE || sc == SCENE_BUNDLES ? have_triangle
         : sc == SCENE_TEXLOAD ? have_triangle && have_dispatches
         : sc == SCENE_DISPATCHES ? have_dispatches
                                  : have_draws;
      if (!supported) {
         printf("frame %-10s UNSUPPORTED (setup failed)\n", scene_names[sc]);
         continue;
      }
      results[sc] = run_scene(&b, sc, frames);
      if (!results[sc].ok && !replace_list(&b)) {
         printf("FAIL - recreate the command list after %s\n",
                scene_names[sc]);
         break;
      }
      const struct scene_result *r = &results[sc];
      if (r->ok)
         printf("frame %-10s pipelined %8.1f us (%6.0f fps), serial %8.1f us; "
                "record %7.1f us, execute %6.1f us\n",
                scene_names[sc], r->pipelined_us, 1e6 / r->pipelined_us,
                r->serial_us, r->record_us, r->execute_us);
      else
         printf("frame %-10s FAILED\n", scene_names[sc]);
   }

   printf("FRAMEBENCH_JSON {\"library\": \"%s\", \"width\": %u, \"height\": %u, "
          "\"frames\": %u, \"inflight\": %u",
          library, width, height, frames, inflight);
   for (unsigned sc = 0; sc < SCENE_COUNT; sc++) {
      const struct scene_result *r = &results[sc];
      printf(", \"%s\": {\"ok\": %s, \"pipelined_us\": %.1f, \"serial_us\": %.1f, "
             "\"record_us\": %.1f, \"execute_us\": %.1f}",
             scene_names[sc], r->ok ? "true" : "false", r->pipelined_us,
             r->serial_us, r->record_us, r->execute_us);
   }
   printf("}\n");
   return failures ? 1 : 0;
}
