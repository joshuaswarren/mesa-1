/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * ExecuteIndirect with a GPU-written count buffer.
 *
 * The count buffer is never written by the CPU: every value it holds is
 * copied into it by the GPU earlier in the SAME command list, so a driver
 * that read the count on the host at record or submit time would see
 * stale contents and draw the wrong number of commands. Three targets:
 *
 *   rt[0]  count 1 of 2 arguments -> only the LEFT triangle
 *   rt[1]  count 2 of 2           -> both triangles
 *   rt[2]  count 0 of 2           -> nothing but the clear
 *
 * The argument patches run in the pre-graphics compute segment before the
 * render pass, so this also exercises that segment being opened inside a
 * draw. Needs a device (GPU).
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

static const float SMALL[3][2] = {{-0.4f, -0.4f}, {0.4f, -0.4f}, {0.0f, 0.4f}};
static const float OFFSETS[2][2] = {{-0.5f, 0.0f}, {0.5f, 0.0f}};

struct vertex {
   float x, y;       /* TEXCOORD0, R32G32_FLOAT,       offset 0 */
   float r, g, b, a; /* TEXCOORD1, R32G32B32A32_FLOAT, offset 8 */
};

/* D3D pixel-centre sampling and the top-left fill rule (as g11). */
static void
rasterise_into(bool covered[H][W], const float ndc[3][2], float dx, float dy)
{
   float p[3][2];
   for (int i = 0; i < 3; i++) {
      p[i][0] = ((ndc[i][0] + dx) * 0.5f + 0.5f) * (float)W;
      p[i][1] = (0.5f - (ndc[i][1] + dy) * 0.5f) * (float)H;
   }
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
make_buffer(ID3D12Device *dev, D3D12_HEAP_TYPE type, size_t size,
            D3D12_RESOURCE_STATES state)
{
   D3D12_HEAP_PROPERTIES hp = {.Type = type};
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
      dev, &hp, D3D12_HEAP_FLAG_NONE, &bd, state, NULL, &IID_ID3D12Resource,
      (void **)&res);
   return hr == S_OK ? res : NULL;
}

static ID3D12Resource *
upload_buffer(ID3D12Device *dev, const void *data, size_t size)
{
   ID3D12Resource *res = make_buffer(dev, D3D12_HEAP_TYPE_UPLOAD, size,
                                     D3D12_RESOURCE_STATE_GENERIC_READ);
   if (!res)
      return NULL;
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
   if (argc < 3) {
      printf("usage: %s vb_vs.dxil vb_ps.dxil\n", argv[0]);
      return 2;
   }

   size_t vs_size = 0, ps_size = 0;
   void *vbvs = slurp(argv[1], &vs_size);
   void *vbps = slurp(argv[2], &ps_size);
   if (!vbvs || !vbps) {
      printf("FAIL - could not read the shaders\n");
      return 1;
   }

   ID3D12Device *dev = NULL;
   HRESULT hr = D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0,
                                  &IID_ID3D12Device, (void **)&dev);
   CHECK(hr == S_OK && dev, "D3D12CreateDevice");
   if (!dev)
      return 1;

   D3D12_ROOT_SIGNATURE_DESC rsd = {
      .Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
   };
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

   D3D12_INPUT_ELEMENT_DESC elems[] = {
      {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0,
       D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
      {"TEXCOORD", 1, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 8,
       D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
   };
   D3D12_GRAPHICS_PIPELINE_STATE_DESC pd = {
      .pRootSignature = root_sig,
      .VS = {vbvs, vs_size},
      .PS = {vbps, ps_size},
      .InputLayout = {elems, 2},
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
   CHECK(hr == S_OK && pso, "vertex-buffer PSO compiles");
   if (!pso)
      return 1;

   /* Two triangles in one vertex buffer: left at 0..2, right at 3..5. */
   struct vertex verts[6];
   memset(verts, 0, sizeof(verts));
   for (int t = 0; t < 2; t++) {
      for (int i = 0; i < 3; i++) {
         struct vertex *v = &verts[t * 3 + i];
         v->x = SMALL[i][0] + OFFSETS[t][0];
         v->y = SMALL[i][1] + OFFSETS[t][1];
         v->r = 0.25f;
         v->g = 0.5f;
         v->b = 0.75f;
         v->a = 1.0f;
      }
   }
   const D3D12_DRAW_ARGUMENTS args[2] = {
      {.VertexCountPerInstance = 3, .InstanceCount = 1},
      {.VertexCountPerInstance = 3, .InstanceCount = 1,
       .StartVertexLocation = 3},
   };
   const uint32_t counts[3] = {1, 2, 0};

   ID3D12Resource *vb = upload_buffer(dev, verts, sizeof(verts));
   ID3D12Resource *arg_buf = upload_buffer(dev, args, sizeof(args));
   ID3D12Resource *count_src[3];
   for (int i = 0; i < 3; i++)
      count_src[i] = upload_buffer(dev, &counts[i], sizeof(counts[i]));
   /* The count buffer lives in device memory and is only ever written by
    * GPU copies recorded in the list below. */
   ID3D12Resource *count_buf = make_buffer(dev, D3D12_HEAP_TYPE_DEFAULT, 256,
                                           D3D12_RESOURCE_STATE_COPY_DEST);
   CHECK(vb && arg_buf && count_src[0] && count_src[1] && count_src[2] &&
            count_buf,
         "create the vertex, argument and count buffers");
   if (!vb || !arg_buf || !count_src[0] || !count_src[1] || !count_src[2] ||
       !count_buf)
      return 1;

   D3D12_VERTEX_BUFFER_VIEW vbv = {ID3D12Resource_GetGPUVirtualAddress(vb),
                                   sizeof(verts), sizeof(struct vertex)};

   D3D12_INDIRECT_ARGUMENT_DESC arg_desc = {
      .Type = D3D12_INDIRECT_ARGUMENT_TYPE_DRAW,
   };
   D3D12_COMMAND_SIGNATURE_DESC csd = {
      .ByteStride = sizeof(D3D12_DRAW_ARGUMENTS),
      .NumArgumentDescs = 1,
      .pArgumentDescs = &arg_desc,
   };
   ID3D12CommandSignature *csig = NULL;
   hr = ID3D12Device_CreateCommandSignature(
      dev, &csd, NULL, &IID_ID3D12CommandSignature, (void **)&csig);
   CHECK(hr == S_OK && csig, "CreateCommandSignature (DRAW)");
   if (!csig)
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
   ID3D12Resource *rt[3] = {NULL, NULL, NULL};
   for (int i = 0; i < 3; i++)
      TEST_OBJECT(ID3D12Device_CreateCommittedResource(
                     dev, &heap, D3D12_HEAP_FLAG_NONE, &td,
                     D3D12_RESOURCE_STATE_COMMON, NULL, &IID_ID3D12Resource,
                     (void **)&rt[i]),
                  rt[i], "create a render target");
   D3D12_DESCRIPTOR_HEAP_DESC hd = {.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                                    .NumDescriptors = 3};
   ID3D12DescriptorHeap *rtv_heap = NULL;
   TEST_OBJECT(ID3D12Device_CreateDescriptorHeap(
                  dev, &hd, &IID_ID3D12DescriptorHeap, (void **)&rtv_heap),
               rtv_heap, "create the RTV heap");
   UINT inc = ID3D12Device_GetDescriptorHandleIncrementSize(
      dev, D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
   D3D12_CPU_DESCRIPTOR_HANDLE rtv[3];
   rtv[0] = d12_test_descriptor_heap_get_cpu_start(rtv_heap);
   rtv[1].ptr = rtv[0].ptr + inc;
   rtv[2].ptr = rtv[0].ptr + 2 * inc;
   for (int i = 0; i < 3; i++)
      ID3D12Device_CreateRenderTargetView(dev, rt[i], NULL, rtv[i]);

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
   ID3D12Fence *fence = NULL;
   TEST_OBJECT(ID3D12Device_CreateFence(dev, 0, D3D12_FENCE_FLAG_NONE,
                                        &IID_ID3D12Fence, (void **)&fence),
               fence, "create the fence");

   uint8_t *map[3] = {NULL, NULL, NULL};
   for (int i = 0; i < 3; i++) {
      map[i] = malloc(STRIDE * H);
      memset(map[i], 0x55, STRIDE * H);
      TEST_CALL(d12_test_opaque_texture_transfer(rt[i], map[i], STRIDE,
                                                 STRIDE * H, true),
                "publish texture poison in COMMON");
   }

   const FLOAT clear[4] = {0.0f, 0.0f, 0.0f, 1.0f};
   D3D12_VIEWPORT vp = {.Width = W, .Height = H, .MaxDepth = 1.0f};
   D3D12_RECT sc = {.right = W, .bottom = H};

   ID3D12GraphicsCommandList_RSSetViewports(list, 1, &vp);
   ID3D12GraphicsCommandList_RSSetScissorRects(list, 1, &sc);
   ID3D12GraphicsCommandList_SetGraphicsRootSignature(list, root_sig);
   ID3D12GraphicsCommandList_IASetPrimitiveTopology(
      list, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
   ID3D12GraphicsCommandList_SetPipelineState(list, pso);
   ID3D12GraphicsCommandList_IASetVertexBuffers(list, 0, 1, &vbv);

   for (int i = 0; i < 3; i++) {
      /* GPU writes the count, then the draw consumes it. */
      if (i > 0)
         d12_test_transition(list, count_buf,
                             D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                             D3D12_RESOURCE_STATE_COPY_DEST);
      ID3D12GraphicsCommandList_CopyBufferRegion(list, count_buf, 0,
                                                 count_src[i], 0,
                                                 sizeof(uint32_t));
      d12_test_transition(list, count_buf, D3D12_RESOURCE_STATE_COPY_DEST,
                          D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);

      d12_test_transition(list, rt[i], D3D12_RESOURCE_STATE_COMMON,
                          D3D12_RESOURCE_STATE_RENDER_TARGET);
      ID3D12GraphicsCommandList_ClearRenderTargetView(list, rtv[i], clear, 0,
                                                      NULL);
      ID3D12GraphicsCommandList_OMSetRenderTargets(list, 1, &rtv[i], FALSE,
                                                   NULL);
      ID3D12GraphicsCommandList_ExecuteIndirect(list, csig, 2, arg_buf, 0,
                                                count_buf, 0);
      d12_test_transition(list, rt[i], D3D12_RESOURCE_STATE_RENDER_TARGET,
                          D3D12_RESOURCE_STATE_COMMON);
   }

   hr = ID3D12GraphicsCommandList_Close(list);
   CHECK(hr == S_OK, "the command list closes cleanly");
   if (hr != S_OK)
      goto teardown;
   TEST_CALL(d12_test_submit_closed(dev, list, queue, fence, 1),
             "submit and wait");
   for (int i = 0; i < 3; i++) {
      memset(map[i], 0x33, STRIDE * H);
      TEST_CALL(d12_test_opaque_texture_transfer(rt[i], map[i], STRIDE,
                                                 STRIDE * H, false),
                "read back the pixels after the fence");
   }

   {
      static bool ref_left[H][W], ref_both[H][W];
      rasterise_into(ref_left, SMALL, OFFSETS[0][0], OFFSETS[0][1]);
      rasterise_into(ref_both, SMALL, OFFSETS[0][0], OFFSETS[0][1]);
      rasterise_into(ref_both, SMALL, OFFSETS[1][0], OFFSETS[1][1]);

      unsigned lit[3] = {0, 0, 0};
      unsigned wrong[3] = {0, 0, 0};
      for (unsigned y = 0; y < H; y++) {
         for (unsigned x = 0; x < W; x++) {
            for (int i = 0; i < 3; i++) {
               const uint8_t *c = map[i] + y * STRIDE + x * 4;
               bool on = !(c[0] == 0 && c[1] == 0 && c[2] == 0);
               bool want = i == 0 ? ref_left[y][x]
                           : i == 1 ? ref_both[y][x] : false;
               lit[i] += on;
               wrong[i] += on != want;
            }
         }
      }
      printf("       count 1: %u lit, %u wrong; count 2: %u lit, %u wrong; "
             "count 0: %u lit\n",
             lit[0], wrong[0], lit[1], wrong[1], lit[2]);
      CHECK(lit[0] > 0 && wrong[0] == 0,
            "GPU-written count 1 draws exactly the first argument");
      CHECK(lit[1] > lit[0] && wrong[1] == 0,
            "GPU-written count 2 draws both arguments");
      CHECK(lit[2] == 0, "GPU-written count 0 draws nothing");
   }

teardown:
   for (int i = 0; i < 3; i++)
      free(map[i]);
   if (fence)
      ID3D12Fence_Release(fence);
   ID3D12GraphicsCommandList_Release(list);
   ID3D12CommandAllocator_Release(alloc);
   ID3D12CommandQueue_Release(queue);
   ID3D12DescriptorHeap_Release(rtv_heap);
   for (int i = 0; i < 3; i++) {
      if (rt[i])
         ID3D12Resource_Release(rt[i]);
      ID3D12Resource_Release(count_src[i]);
   }
   ID3D12CommandSignature_Release(csig);
   ID3D12Resource_Release(count_buf);
   ID3D12Resource_Release(arg_buf);
   ID3D12Resource_Release(vb);
   ID3D12PipelineState_Release(pso);
   ID3D12RootSignature_Release(root_sig);
   ID3D12Device_Release(dev);
   free(vbvs);
   free(vbps);
   printf(failures ? "EXECUTE INDIRECT COUNT: %d FAILURES\n"
                   : "EXECUTE INDIRECT COUNT: ALL PASS\n",
          failures);
   return failures ? 1 : 0;
}
