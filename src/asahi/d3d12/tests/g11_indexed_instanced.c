/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * G11: indexed draws and per-instance attributes.
 *
 * G10 proved an application's vertex data reaches a shader. Two things it
 * deliberately left unproven, and this suite closes both:
 *
 *   A. INDEXED draws. IASetIndexBuffer and DrawIndexedInstanced were
 *      stubs -- the binding was recorded nowhere and the entry point
 *      reported "not implemented". Nothing indexed could be drawn.
 *
 *   B. PER-INSTANCE attributes. agx_velem_key has an `instanced` flag
 *      and a divisor, and the input-layout resolver filled them in, but
 *      no suite ever drew more than one instance -- so it sat in the
 *      "compiles, unproven" bucket this campaign exists to shrink.
 *
 * Part A is built so that EVERY link is load-bearing. The vertex buffer
 * holds six vertices: three decoys at 0..2 and the real triangle at 3..5.
 * The index buffer holds {9,9,9, 0,1,2}, and the draw uses
 * StartIndexLocation=3 and BaseVertexLocation=3. So:
 *
 *   - ignore the index buffer  -> draws vertices 0,1,2, the decoys
 *   - ignore StartIndexLocation -> reads indices {9,9,9}, out of range
 *   - ignore BaseVertexLocation -> draws vertices 0,1,2, the decoys
 *
 * and only doing all three correctly selects vertices 3,4,5, which are
 * exactly tri.dxil's triangle. The decoys are a small wedge in one
 * corner, so a wrong answer is not subtly wrong.
 *
 * Part B draws two instances of one small triangle with a per-instance
 * offset moving them to opposite sides of the target. They do not
 * overlap, so "only one instance drew" and "both instances landed in the
 * same place" are both visible as a coverage mismatch, and the
 * expectation is a CPU rasterisation of both.
 */

#define COBJMACROS
#include <wsl/winadapter.h>
#include <directx/d3d12.h>

#include "d12_test_abi.h"

#include <math.h>
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

/* tri.dxil's triangle, the same oracle G4 and G10 use. */
static const float TRI_NDC[3][2] = {{1.0f, -1.0f}, {-1.0f, -1.0f}, {5.0f, 1.0f}};

/* Part B: a small triangle, and where each instance puts it. */
static const float SMALL[3][2] = {{-0.4f, -0.4f}, {0.4f, -0.4f}, {0.0f, 0.4f}};
static const float OFFSETS[2][2] = {{-0.5f, 0.0f}, {0.5f, 0.0f}};

#define COL_R 0.25f
#define COL_G 0.50f
#define COL_B 0.75f

struct vertex {
   float x, y;       /* TEXCOORD0, R32G32_FLOAT,       offset 0 */
   float r, g, b, a; /* TEXCOORD1, R32G32B32A32_FLOAT, offset 8 */
};

/*
 * D3D pixel-centre sampling and the top-left fill rule, over whatever
 * three clip-space points it is handed. Accumulates, so part B can union
 * two instances into one reference image.
 */
static void
rasterise_into(bool covered[H][W], const float ndc[3][2], float dx, float dy)
{
   float p[3][2];
   for (int i = 0; i < 3; i++) {
      p[i][0] = ((ndc[i][0] + dx) * 0.5f + 0.5f) * (float)W;
      p[i][1] = (0.5f - (ndc[i][1] + dy) * 0.5f) * (float)H;
   }

   /* Culling is off, so both windings rasterise. Normalise to the one the
    * interior test below assumes rather than assuming the caller's. */
   float area = (p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) -
                (p[1][1] - p[0][1]) * (p[2][0] - p[0][0]);
   if (area < 0.0f) {
      float t0 = p[1][0], t1 = p[1][1];
      p[1][0] = p[2][0];
      p[1][1] = p[2][1];
      p[2][0] = t0;
      p[2][1] = t1;
   }

#define EDGE(a, b, px, py)                                                   \
   ((p[b][0] - p[a][0]) * ((py) - p[a][1]) -                                 \
    (p[b][1] - p[a][1]) * ((px) - p[a][0]))

   for (unsigned y = 0; y < H; y++) {
      for (unsigned x = 0; x < W; x++) {
         float cx = (float)x + 0.5f, cy = (float)y + 0.5f;
         const float e[3] = {EDGE(0, 1, cx, cy), EDGE(1, 2, cx, cy),
                             EDGE(2, 0, cx, cy)};
         const int ia[3] = {0, 1, 2}, ib[3] = {1, 2, 0};

         bool in = true;
         for (int k = 0; k < 3; k++) {
            if (e[k] > 0.0f)
               continue;
            if (e[k] < 0.0f) {
               in = false;
               break;
            }
            float ex = p[ib[k]][0] - p[ia[k]][0];
            float ey = p[ib[k]][1] - p[ia[k]][1];
            bool top_left = (ey < 0.0f) || (ey == 0.0f && ex > 0.0f);
            if (!top_left) {
               in = false;
               break;
            }
         }
         if (in)
            covered[y][x] = true;
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
   if (p)
      *size = n;
   return p;
}

static ID3D12Resource *
upload_buffer(ID3D12Device *dev, const void *data, size_t size)
{
   D3D12_HEAP_PROPERTIES up = {.Type = D3D12_HEAP_TYPE_UPLOAD};
   D3D12_RESOURCE_DESC bd = {
      .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
      .Width = size,
      .Height = 1,
      .DepthOrArraySize = 1,
      .MipLevels = 1,
      .Format = DXGI_FORMAT_UNKNOWN,
      .SampleDesc = {.Count = 1},
      .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
   };
   ID3D12Resource *res = NULL;
   HRESULT hr = ID3D12Device_CreateCommittedResource(
          dev, &up, D3D12_HEAP_FLAG_NONE, &bd,
          D3D12_RESOURCE_STATE_GENERIC_READ, NULL, &IID_ID3D12Resource,
          (void **)&res);
   if (hr != S_OK || !res) {
      if (res) ID3D12Resource_Release(res);
      return NULL;
   }

   void *map = NULL;
   if (d12_test_map(res, false, &map) != S_OK) {
      ID3D12Resource_Release(res);
      return NULL;
   }
   memcpy(map, data, size);
   ID3D12Resource_Unmap(res, 0, NULL);
   return res;
}

int
main(int argc, char **argv)
{
   if (argc < 6) {
      printf("usage: %s tri.dxil solid.dxil vb_vs.dxil vb_ps.dxil "
             "vb_inst_vs.dxil\n",
             argv[0]);
      return 2;
   }

   size_t tri_size = 0, solid_size = 0, vs_size = 0, ps_size = 0, iv_size = 0;
   void *tri = slurp(argv[1], &tri_size);
   void *solid = slurp(argv[2], &solid_size);
   void *vbvs = slurp(argv[3], &vs_size);
   void *vbps = slurp(argv[4], &ps_size);
   void *ivs = slurp(argv[5], &iv_size);
   if (!tri || !solid || !vbvs || !vbps || !ivs) {
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

   D3D12_INPUT_ELEMENT_DESC elems_a[] = {
      {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0,
       D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
      {"TEXCOORD", 1, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 8,
       D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
   };

   /* Slot 1 steps once per INSTANCE. That is the whole point of part B. */
   D3D12_INPUT_ELEMENT_DESC elems_b[] = {
      {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0,
       D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
      {"TEXCOORD", 1, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 8,
       D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
      {"TEXCOORD", 2, DXGI_FORMAT_R32G32_FLOAT, 1, 0,
       D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA, 1},
   };

   D3D12_GRAPHICS_PIPELINE_STATE_DESC base_pd = {
      .pRootSignature = root_sig,
      .NumRenderTargets = 1,
      .RTVFormats = {DXGI_FORMAT_R8G8B8A8_UNORM},
      .SampleDesc = {.Count = 1},
      .SampleMask = ~0u,
         .RasterizerState = {.FillMode = D3D12_FILL_MODE_SOLID, .CullMode = D3D12_CULL_MODE_NONE, .DepthClipEnable = TRUE},
      .PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
};
   base_pd.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;

   D3D12_GRAPHICS_PIPELINE_STATE_DESC ctrl_pd = base_pd;
   ctrl_pd.VS = (D3D12_SHADER_BYTECODE){tri, tri_size};
   ctrl_pd.PS = (D3D12_SHADER_BYTECODE){solid, solid_size};

   D3D12_GRAPHICS_PIPELINE_STATE_DESC idx_pd = base_pd;
   idx_pd.VS = (D3D12_SHADER_BYTECODE){vbvs, vs_size};
   idx_pd.PS = (D3D12_SHADER_BYTECODE){vbps, ps_size};
   idx_pd.InputLayout = (D3D12_INPUT_LAYOUT_DESC){elems_a, 2};

   D3D12_GRAPHICS_PIPELINE_STATE_DESC inst_pd = base_pd;
   inst_pd.VS = (D3D12_SHADER_BYTECODE){ivs, iv_size};
   inst_pd.PS = (D3D12_SHADER_BYTECODE){vbps, ps_size};
   inst_pd.InputLayout = (D3D12_INPUT_LAYOUT_DESC){elems_b, 3};

   ID3D12PipelineState *ctrl_pso = NULL, *idx_pso = NULL, *inst_pso = NULL;
   TEST_OBJECT(ID3D12Device_CreateGraphicsPipelineState(dev, &ctrl_pd,
                                            &IID_ID3D12PipelineState,
                                            (void **)&ctrl_pso), ctrl_pso, "checked ID3D12Device_CreateGraphicsPipelineState ctrl_pso");
   TEST_OBJECT(ID3D12Device_CreateGraphicsPipelineState(dev, &idx_pd,
                                            &IID_ID3D12PipelineState,
                                            (void **)&idx_pso), idx_pso, "checked ID3D12Device_CreateGraphicsPipelineState idx_pso");
   hr = ID3D12Device_CreateGraphicsPipelineState(dev, &inst_pd,
                                                 &IID_ID3D12PipelineState,
                                                 (void **)&inst_pso);
   TEST_OBJECT(hr, inst_pso, "checked ID3D12Device_CreateGraphicsPipelineState inst_pso");
   CHECK(ctrl_pso && idx_pso, "control and indexed PSOs compile");
   CHECK(hr == S_OK && inst_pso,
         "a PSO with a PER_INSTANCE_DATA input element compiles");
   if (!ctrl_pso || !idx_pso || !inst_pso)
      return 1;

   /* --- Part A buffers: decoys first, the real triangle at 3..5 --- */
   struct vertex verts_a[6];
   memset(verts_a, 0, sizeof(verts_a));
   for (int i = 0; i < 3; i++) {
      /* Decoys: a small wedge in the top-left, nothing like the real one. */
      verts_a[i].x = -0.9f + 0.2f * (float)i;
      verts_a[i].y = 0.9f - 0.15f * (float)i;
      verts_a[i].r = 1.0f;
      verts_a[i].a = 1.0f;

      verts_a[3 + i].x = TRI_NDC[i][0];
      verts_a[3 + i].y = TRI_NDC[i][1];
      verts_a[3 + i].r = COL_R;
      verts_a[3 + i].g = COL_G;
      verts_a[3 + i].b = COL_B;
      verts_a[3 + i].a = 1.0f;
   }

   /* Padding indices the draw must SKIP, then the real ones. With
    * BaseVertexLocation=3 these select vertices 3,4,5. */
   const uint16_t indices[6] = {9, 9, 9, 0, 1, 2};

   /* --- Part B buffers --- */
   struct vertex verts_b[3];
   memset(verts_b, 0, sizeof(verts_b));
   for (int i = 0; i < 3; i++) {
      verts_b[i].x = SMALL[i][0];
      verts_b[i].y = SMALL[i][1];
      verts_b[i].r = COL_R;
      verts_b[i].g = COL_G;
      verts_b[i].b = COL_B;
      verts_b[i].a = 1.0f;
   }
   const float offsets[2][2] = {{OFFSETS[0][0], OFFSETS[0][1]},
                                {OFFSETS[1][0], OFFSETS[1][1]}};

   ID3D12Resource *vb_a = upload_buffer(dev, verts_a, sizeof(verts_a));
   ID3D12Resource *ib = upload_buffer(dev, indices, sizeof(indices));
   ID3D12Resource *vb_b = upload_buffer(dev, verts_b, sizeof(verts_b));
   ID3D12Resource *vb_off = upload_buffer(dev, offsets, sizeof(offsets));
   CHECK(vb_a && ib && vb_b && vb_off, "create the vertex and index buffers");
   if (!vb_a || !ib || !vb_b || !vb_off)
      return 1;

   D3D12_VERTEX_BUFFER_VIEW vbv_a = {
      ID3D12Resource_GetGPUVirtualAddress(vb_a), sizeof(verts_a),
      sizeof(struct vertex)};
   D3D12_INDEX_BUFFER_VIEW ibv = {ID3D12Resource_GetGPUVirtualAddress(ib),
                                  sizeof(indices), DXGI_FORMAT_R16_UINT};
   D3D12_VERTEX_BUFFER_VIEW vbv_b = {
      ID3D12Resource_GetGPUVirtualAddress(vb_b), sizeof(verts_b),
      sizeof(struct vertex)};
   D3D12_VERTEX_BUFFER_VIEW vbv_off = {
      ID3D12Resource_GetGPUVirtualAddress(vb_off), sizeof(offsets),
      2 * sizeof(float)};

   /* Three targets: control, indexed, instanced. */
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
      .Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS,
   };
   ID3D12Resource *rt[3] = {NULL, NULL, NULL};
   for (int i = 0; i < 3; i++)
      TEST_OBJECT(ID3D12Device_CreateCommittedResource(
         dev, &heap, D3D12_HEAP_FLAG_NONE, &td,
         D3D12_RESOURCE_STATE_COMMON, NULL, &IID_ID3D12Resource,
         (void **)&rt[i]), rt[i], "checked ID3D12Device_CreateCommittedResource rt[i]");
   CHECK(rt[0] && rt[1] && rt[2], "create three render targets");
   if (!rt[0] || !rt[1] || !rt[2])
      return 1;

   D3D12_DESCRIPTOR_HEAP_DESC hd = {.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                                    .NumDescriptors = 3};
   ID3D12DescriptorHeap *rtv_heap = NULL;
   TEST_OBJECT(ID3D12Device_CreateDescriptorHeap(dev, &hd, &IID_ID3D12DescriptorHeap,
                                     (void **)&rtv_heap), rtv_heap, "checked ID3D12Device_CreateDescriptorHeap rtv_heap");
   UINT inc = ID3D12Device_GetDescriptorHandleIncrementSize(
      dev, D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
   D3D12_CPU_DESCRIPTOR_HANDLE rtv[3];
   rtv[0] = d12_test_descriptor_heap_get_cpu_start(rtv_heap);
   CHECK(inc && rtv[0].ptr && rtv[0].ptr <= SIZE_MAX - (UINT64)inc * 2,
         "RTV handle and descriptor increment bounds");
   if (!inc || !rtv[0].ptr || rtv[0].ptr > SIZE_MAX - (UINT64)inc * 2) return 1;
   rtv[1].ptr = rtv[0].ptr + inc;
   rtv[2].ptr = rtv[0].ptr + 2 * inc;
   for (int i = 0; i < 3; i++)
      ID3D12Device_CreateRenderTargetView(dev, rt[i], NULL, rtv[i]);

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

   uint8_t *map[3] = {NULL, NULL, NULL};
   for (int i = 0; i < 3; i++) {
      map[i] = malloc(STRIDE * H);
      CHECK(map[i] != NULL, "CPU storage for opaque render target");
      if (!map[i]) return 1;
      memset(map[i], 0x55, STRIDE * H);
      TEST_CALL(d12_test_opaque_texture_transfer(rt[i], map[i], STRIDE, STRIDE * H, true),
                "publish original texture poison in COMMON");
   }

   const FLOAT clear[4] = {0.0f, 0.0f, 0.0f, 1.0f};
   D3D12_VIEWPORT vp = {.Width = W, .Height = H, .MaxDepth = 1.0f};
   D3D12_RECT sc = {.right = W, .bottom = H};

   TEST_CALL(ID3D12GraphicsCommandList_Close(list), "close initial empty graphics list");
   TEST_CALL(ID3D12CommandAllocator_Reset(alloc), "reset idle allocator");
   TEST_CALL(ID3D12GraphicsCommandList_Reset(list, alloc, NULL), "reset checked graphics list");
   ID3D12GraphicsCommandList_RSSetViewports(list, 1, &vp);
   ID3D12GraphicsCommandList_RSSetScissorRects(list, 1, &sc);
   ID3D12GraphicsCommandList_SetGraphicsRootSignature(list, root_sig);
   ID3D12GraphicsCommandList_IASetPrimitiveTopology(
      list, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

   /* Control */
   d12_test_transition(list, rt[0], D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET);
   ID3D12GraphicsCommandList_ClearRenderTargetView(list, rtv[0], clear, 0,
                                                   NULL);
   ID3D12GraphicsCommandList_OMSetRenderTargets(list, 1, &rtv[0], FALSE, NULL);
   ID3D12GraphicsCommandList_SetPipelineState(list, ctrl_pso);
   ID3D12GraphicsCommandList_DrawInstanced(list, 3, 1, 0, 0);

   /* A: indexed, with a start index and a base vertex */
   d12_test_transition(list, rt[1], D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET);
   ID3D12GraphicsCommandList_ClearRenderTargetView(list, rtv[1], clear, 0,
                                                   NULL);
   ID3D12GraphicsCommandList_OMSetRenderTargets(list, 1, &rtv[1], FALSE, NULL);
   ID3D12GraphicsCommandList_SetPipelineState(list, idx_pso);
   ID3D12GraphicsCommandList_IASetVertexBuffers(list, 0, 1, &vbv_a);
   ID3D12GraphicsCommandList_IASetIndexBuffer(list, &ibv);
   ID3D12GraphicsCommandList_DrawIndexedInstanced(list, 3, 1, 3, 3, 0);

   /* B: two instances, per-instance offset from slot 1 */
   d12_test_transition(list, rt[2], D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET);
   ID3D12GraphicsCommandList_ClearRenderTargetView(list, rtv[2], clear, 0,
                                                   NULL);
   ID3D12GraphicsCommandList_OMSetRenderTargets(list, 1, &rtv[2], FALSE, NULL);
   ID3D12GraphicsCommandList_SetPipelineState(list, inst_pso);
   {
      D3D12_VERTEX_BUFFER_VIEW both[2] = {vbv_b, vbv_off};
      ID3D12GraphicsCommandList_IASetVertexBuffers(list, 0, 2, both);
   }
   ID3D12GraphicsCommandList_DrawInstanced(list, 3, 2, 0, 0);

   for (int i = 0; i < 3; i++)
      d12_test_transition(list, rt[i], D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COMMON);
   hr = ID3D12GraphicsCommandList_Close(list);
   CHECK(hr == S_OK, "the command list closes cleanly");
   if (hr != S_OK)
      goto teardown;

   TEST_CALL(d12_test_submit_closed(dev, list, queue, fence, 1), "exact fenced indexed/instance draw");
   for (int i = 0; i < 3; i++) {
      memset(map[i], 0x33, STRIDE * H);
      TEST_CALL(d12_test_opaque_texture_transfer(rt[i], map[i], STRIDE, STRIDE * H, false),
                "read actual indexed/instance pixels only after fence");
   }
   CHECK(ID3D12Fence_GetCompletedValue(fence) >= 1,
         "the GPU completes an indexed draw and an instanced draw");

   if (!map[0] || !map[1] || !map[2]) {
      printf("FAIL - could not map the render targets\n");
      failures++;
      goto teardown;
   }

   {
      static bool ref_a[H][W], ref_b[H][W];
      rasterise_into(ref_a, TRI_NDC, 0.0f, 0.0f);
      rasterise_into(ref_b, SMALL, OFFSETS[0][0], OFFSETS[0][1]);
      rasterise_into(ref_b, SMALL, OFFSETS[1][0], OFFSETS[1][1]);

      unsigned ctrl_n = 0, idx_n = 0, inst_n = 0;
      unsigned idx_wrong = 0, inst_wrong = 0, a_differ = 0;
      unsigned left = 0, right = 0, colour_wrong = 0;

      for (unsigned y = 0; y < H; y++) {
         for (unsigned x = 0; x < W; x++) {
            const uint8_t *c = map[0] + y * STRIDE + x * 4;
            const uint8_t *a = map[1] + y * STRIDE + x * 4;
            const uint8_t *b = map[2] + y * STRIDE + x * 4;

            bool cc = !(c[0] == 0 && c[1] == 0 && c[2] == 0);
            bool ca = !(a[0] == 0 && a[1] == 0 && a[2] == 0);
            bool cb = !(b[0] == 0 && b[1] == 0 && b[2] == 0);

            ctrl_n += cc;
            idx_n += ca;
            inst_n += cb;
            idx_wrong += (ca != ref_a[y][x]);
            inst_wrong += (cb != ref_b[y][x]);
            a_differ += (ca != cc);

            if (cb) {
               left += (x < W / 2);
               right += (x >= W / 2);
               int dr = (int)b[0] - 64, dg = (int)b[1] - 128,
                   db = (int)b[2] - 191;
               if (abs(dr) > 1 || abs(dg) > 1 || abs(db) > 1)
                  colour_wrong++;
            }
         }
      }

      printf("       A: control %u, indexed %u, wrong %u, differ %u\n", ctrl_n,
             idx_n, idx_wrong, a_differ);
      printf("       B: instanced %u (left %u, right %u), wrong %u, "
             "colour wrong %u\n",
             inst_n, left, right, inst_wrong, colour_wrong);

      CHECK(idx_wrong == 0,
            "an INDEXED draw through a bound index buffer matches the CPU "
            "rasterisation on all 65536 pixels");
      CHECK(a_differ == 0,
            "the indexed draw reproduces the SV_VertexID control PIXEL FOR "
            "PIXEL, so StartIndexLocation and BaseVertexLocation both "
            "landed");

      CHECK(inst_wrong == 0,
            "TWO INSTANCES with a per-instance offset attribute match the "
            "CPU rasterisation of both on all 65536 pixels");
      CHECK(left > 0 && right > 0,
            "both instances drew, on opposite sides -- a per-instance "
            "attribute read as per-vertex would stack them in one place");
      CHECK(colour_wrong == 0,
            "the per-vertex attributes still arrive correctly alongside "
            "the per-instance one");
   }

teardown:
   for (int i = 0; i < 3; i++) {
      if (map[i])
         free(map[i]);
   }
   if (fence)
      ID3D12Fence_Release(fence);
   ID3D12GraphicsCommandList_Release(list);
   ID3D12CommandAllocator_Release(alloc);
   ID3D12CommandQueue_Release(queue);
   ID3D12DescriptorHeap_Release(rtv_heap);
   for (int i = 0; i < 3; i++) {
      if (rt[i])
         ID3D12Resource_Release(rt[i]);
   }
   ID3D12Resource_Release(vb_a);
   ID3D12Resource_Release(ib);
   ID3D12Resource_Release(vb_b);
   ID3D12Resource_Release(vb_off);
   if (inst_pso)
      ID3D12PipelineState_Release(inst_pso);
   if (idx_pso)
      ID3D12PipelineState_Release(idx_pso);
   if (ctrl_pso)
      ID3D12PipelineState_Release(ctrl_pso);
   ID3D12RootSignature_Release(root_sig);
   ID3D12Device_Release(dev);

   printf(failures ? "G11 INDEXED/INSTANCED: %d FAILURES\n"
                   : "G11 INDEXED/INSTANCED: ALL PASS\n",
          failures);
   free(tri);
   free(solid);
   free(vbvs);
   free(vbps);
   free(ivs);
   return failures ? 1 : 0;
}
