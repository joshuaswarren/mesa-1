/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * G10: vertex data reaches a shader from a BOUND BUFFER.
 *
 * Every other probe in this tree derives its geometry from SV_VertexID,
 * and not by preference: until this suite there was no attribute fetch at
 * all. IASetVertexBuffers recorded a binding that nothing read, and the
 * vertex prolog was always `static_vi` -- "just fix up the vertex ID".
 * An application's own vertex data could not reach a shader.
 *
 * The falsification, stated in roadmap.md before any of this was written:
 *
 *   a draw whose vertex data comes only from a bound buffer must produce
 *   the same pixels as the equivalent SV_VertexID shader.
 *
 * So this suite draws ONE triangle TWICE into two targets:
 *
 *   control  tri.dxil    -- computes (1,-1), (-1,-1), (5,1) from the
 *                           vertex id, fetching nothing
 *   subject  vb_vs.dxil  -- reads those same three positions out of a
 *                           vertex buffer, and computes nothing
 *
 * and compares all 65536 pixels. Both are also checked against a CPU
 * rasterisation of those vertices, so a driver that broke BOTH paths
 * identically still fails.
 *
 * The subject's buffer is INTERLEAVED -- position at offset 0, colour at
 * offset 8, stride 24 -- because a fetch that ignored the attribute
 * offset would get the position right and the colour wrong. The colour is
 * constant across the three vertices, so its interpolated value is exact
 * and can be asserted as a number rather than a tolerance; a wrong STRIDE
 * would make vertices 1 and 2 read colour from the wrong place and break
 * that constancy.
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

/* The three vertices tri.dxil computes, and the colour vb_vs carries. */
static const float TRI_NDC[3][2] = {{1.0f, -1.0f}, {-1.0f, -1.0f}, {5.0f, 1.0f}};

#define COL_R 0.25f
#define COL_G 0.50f
#define COL_B 0.75f

struct vertex {
   float x, y;          /* TEXCOORD0, R32G32_FLOAT,       offset 0  */
   float r, g, b, a;    /* TEXCOORD1, R32G32B32A32_FLOAT, offset 8  */
};

/*
 * THE ORACLE, identical to g4_triangle's: D3D pixel-centre sampling and
 * the top-left fill rule, over the shader's own vertices. Not calibrated
 * against the GPU.
 */
static void
rasterise_reference(bool covered[H][W])
{
   float p[3][2];
   for (int i = 0; i < 3; i++) {
      p[i][0] = (TRI_NDC[i][0] * 0.5f + 0.5f) * (float)W;
      p[i][1] = (0.5f - TRI_NDC[i][1] * 0.5f) * (float)H;
   }

#define EDGE(a, b, px, py)                                                   \
   ((p[b][0] - p[a][0]) * ((py) - p[a][1]) -                                 \
    (p[b][1] - p[a][1]) * ((px) - p[a][0]))

   for (unsigned y = 0; y < H; y++) {
      for (unsigned x = 0; x < W; x++) {
         float cx = (float)x + 0.5f, cy = (float)y + 0.5f;
         float e0 = EDGE(0, 1, cx, cy);
         float e1 = EDGE(1, 2, cx, cy);
         float e2 = EDGE(2, 0, cx, cy);

         bool in = true;
         const int ia[3] = {0, 1, 2}, ib[3] = {1, 2, 0};
         const float e[3] = {e0, e1, e2};
         for (int k = 0; k < 3; k++) {
            if (e[k] > 0.0f)
               continue;
            if (e[k] < 0.0f) {
               in = false;
               break;
            }
            /* Exactly on the edge: covered only if it is a top or left
             * edge, which is what D3D's fill rule says. */
            float dx = p[ib[k]][0] - p[ia[k]][0];
            float dy = p[ib[k]][1] - p[ia[k]][1];
            bool top_left = (dy < 0.0f) || (dy == 0.0f && dx > 0.0f);
            if (!top_left) {
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
   if (p)
      *size = n;
   return p;
}

int
main(int argc, char **argv)
{
   if (argc < 5) {
      printf("usage: %s tri.dxil solid.dxil vb_vs.dxil vb_ps.dxil\n", argv[0]);
      return 2;
   }

   size_t tri_size = 0, solid_size = 0, vs_size = 0, ps_size = 0;
   void *tri = slurp(argv[1], &tri_size);
   void *solid = slurp(argv[2], &solid_size);
   void *vbvs = slurp(argv[3], &vs_size);
   void *vbps = slurp(argv[4], &ps_size);
   if (!tri || !solid || !vbvs || !vbps) {
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

   /* THE INPUT LAYOUT. Slot 31 proves that the API's 32 binding slots are
    * distinct from AGX's 16-live-attribute limit. nir_to_dxil names generic
    * vertex inputs
    * TEXCOORD<driver_location>, so these semantic names are what vb_vs
    * actually declares -- they are not a stylistic choice.
    */
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

   D3D12_GRAPHICS_PIPELINE_STATE_DESC control_pd = {
      .pRootSignature = root_sig,
      .VS = {.pShaderBytecode = tri, .BytecodeLength = tri_size},
      .PS = {.pShaderBytecode = solid, .BytecodeLength = solid_size},
      .NumRenderTargets = 1,
      .RTVFormats = {DXGI_FORMAT_R8G8B8A8_UNORM},
      .SampleDesc = {.Count = 1},
      .SampleMask = ~0u,
         .RasterizerState = {.FillMode = D3D12_FILL_MODE_SOLID, .CullMode = D3D12_CULL_MODE_NONE, .DepthClipEnable = TRUE},
      .PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
};
   control_pd.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;

   D3D12_GRAPHICS_PIPELINE_STATE_DESC subject_pd = control_pd;
   subject_pd.VS = (D3D12_SHADER_BYTECODE){vbvs, vs_size};
   subject_pd.PS = (D3D12_SHADER_BYTECODE){vbps, ps_size};
   subject_pd.InputLayout = (D3D12_INPUT_LAYOUT_DESC){elems, 2};

   ID3D12PipelineState *control_pso = NULL, *subject_pso = NULL;
   hr = ID3D12Device_CreateGraphicsPipelineState(
      dev, &control_pd, &IID_ID3D12PipelineState, (void **)&control_pso);
   TEST_OBJECT(hr, control_pso, "checked ID3D12Device_CreateGraphicsPipelineState control_pso");
   CHECK(hr == S_OK && control_pso, "control PSO (SV_VertexID) compiles");

   hr = ID3D12Device_CreateGraphicsPipelineState(
      dev, &subject_pd, &IID_ID3D12PipelineState, (void **)&subject_pso);
   TEST_OBJECT(hr, subject_pso, "checked ID3D12Device_CreateGraphicsPipelineState subject_pso");
   CHECK(hr == S_OK && subject_pso,
         "subject PSO compiles WITH a non-empty D3D12_INPUT_LAYOUT_DESC");
   if (!control_pso || !subject_pso)
      return 1;

   /* The vertex buffer: upload heap so the CPU can write it directly. */
   struct vertex verts[3];
   for (int i = 0; i < 3; i++) {
      verts[i] = (struct vertex){.x = TRI_NDC[i][0],
                                 .y = TRI_NDC[i][1],
                                 .r = COL_R,
                                 .g = COL_G,
                                 .b = COL_B,
                                 .a = 1.0f};
   }

   D3D12_HEAP_PROPERTIES up = {.Type = D3D12_HEAP_TYPE_UPLOAD};
   D3D12_RESOURCE_DESC bd = {
      .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
      .Width = sizeof(verts),
      .Height = 1,
      .DepthOrArraySize = 1,
      .MipLevels = 1,
      .Format = DXGI_FORMAT_UNKNOWN,
      .SampleDesc = {.Count = 1},
      .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
   };
   ID3D12Resource *vb = NULL;
   hr = ID3D12Device_CreateCommittedResource(
      dev, &up, D3D12_HEAP_FLAG_NONE, &bd,
      D3D12_RESOURCE_STATE_GENERIC_READ, NULL, &IID_ID3D12Resource,
      (void **)&vb);
   TEST_OBJECT(hr, vb, "checked ID3D12Device_CreateCommittedResource vb");
   CHECK(hr == S_OK && vb, "create the vertex buffer");
   if (!vb)
      return 1;

   void *vbmap = NULL;
   TEST_CALL(d12_test_map(vb, false, &vbmap), "checked vertex upload map");
   CHECK(vbmap != NULL, "map the vertex buffer");
   if (!vbmap)
      return 1;
   memcpy(vbmap, verts, sizeof(verts));
   ID3D12Resource_Unmap(vb, 0, NULL);

   D3D12_VERTEX_BUFFER_VIEW vbv = {
      .BufferLocation = ID3D12Resource_GetGPUVirtualAddress(vb),
      .SizeInBytes = sizeof(verts),
      .StrideInBytes = sizeof(struct vertex),
   };
   CHECK(vbv.BufferLocation != 0,
         "the vertex buffer has a GPU virtual address");

   /* Two render targets, one per draw. */
   D3D12_HEAP_PROPERTIES heap = {.Type = D3D12_HEAP_TYPE_DEFAULT};
   D3D12_RESOURCE_DESC td = {
      .Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
      .Width = W,
      .Height = H,
      .DepthOrArraySize = 1,
      .MipLevels = 1,
      .Format = DXGI_FORMAT_R8G8B8A8_UNORM,
      .SampleDesc = {.Count = 1},
      .Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE,
      .Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
   };
   ID3D12Resource *rt[2] = {NULL, NULL};
   for (int i = 0; i < 2; i++) {
      TEST_OBJECT(ID3D12Device_CreateCommittedResource(
         dev, &heap, D3D12_HEAP_FLAG_NONE, &td,
         D3D12_RESOURCE_STATE_COMMON, NULL, &IID_ID3D12Resource,
         (void **)&rt[i]), rt[i], "checked ID3D12Device_CreateCommittedResource rt[i]");
   }
   CHECK(rt[0] && rt[1], "create both render targets");
   if (!rt[0] || !rt[1])
      return 1;

   /* DEFAULT resources are intentionally not CPU-mappable.  Grade both
    * draws through the public D3D12 readback path instead of depending on
    * this driver's internal linear render-target layout. */
   D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp = {0};
   UINT rows = 0;
   UINT64 row_bytes = 0, total = 0;
   ID3D12Device_GetCopyableFootprints(dev, &td, 0, 1, 0, &fp, &rows,
                                      &row_bytes, &total);
   CHECK(fp.Footprint.Width == W && fp.Footprint.Height == H && rows == H &&
            row_bytes == (UINT64)W * 4,
         "get a valid render-target readback footprint");
   CHECK(fp.Footprint.RowPitch >= W * 4 && fp.Footprint.RowPitch % 256 == 0 &&
         fp.Footprint.Depth == 1 && total >= (UINT64)fp.Footprint.RowPitch * H,
         "readback pitch/depth/size are valid");
   if (fp.Footprint.Width != W || fp.Footprint.Height != H || rows != H ||
       row_bytes != (UINT64)W * 4 || fp.Footprint.RowPitch < W * 4 ||
       fp.Footprint.RowPitch % 256 || fp.Footprint.Depth != 1 ||
       total < (UINT64)fp.Footprint.RowPitch * H) return 1;

   D3D12_HEAP_PROPERTIES readback_heap = {
      .Type = D3D12_HEAP_TYPE_READBACK,
   };
   D3D12_RESOURCE_DESC readback_desc = {
      .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
      .Width = total,
      .Height = 1,
      .DepthOrArraySize = 1,
      .MipLevels = 1,
      .SampleDesc = {.Count = 1},
      .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
   };
   ID3D12Resource *readback[2] = {NULL, NULL};
   uint8_t *map[2] = {NULL, NULL};
   for (int i = 0; i < 2; i++) {
      hr = ID3D12Device_CreateCommittedResource(
         dev, &readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc,
         D3D12_RESOURCE_STATE_COPY_DEST, NULL, &IID_ID3D12Resource,
         (void **)&readback[i]);
   TEST_OBJECT(hr, readback[i], "checked ID3D12Device_CreateCommittedResource readback[i]");
      TEST_CALL(d12_test_map(readback[i], false, (void **)&map[i]), "checked readback poison map");
      memset(map[i], 0x55, (size_t)total);
   }
   CHECK(readback[0] && readback[1] && map[0] && map[1],
         "create and map both readback buffers");
   if (!readback[0] || !readback[1] || !map[0] || !map[1])
      return 1;

   D3D12_DESCRIPTOR_HEAP_DESC hd = {.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                                    .NumDescriptors = 2};
   ID3D12DescriptorHeap *rtv_heap = NULL;
   TEST_OBJECT(ID3D12Device_CreateDescriptorHeap(dev, &hd, &IID_ID3D12DescriptorHeap,
                                     (void **)&rtv_heap), rtv_heap, "checked ID3D12Device_CreateDescriptorHeap rtv_heap");
   UINT inc = ID3D12Device_GetDescriptorHandleIncrementSize(
      dev, D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
   D3D12_CPU_DESCRIPTOR_HANDLE rtv0 =
      d12_test_descriptor_heap_get_cpu_start(rtv_heap);
   CHECK(rtv0.ptr != 0, "valid RTV descriptor handle");
   if (!rtv0.ptr) return 1;
   CHECK(inc && rtv0.ptr <= SIZE_MAX - inc, "RTV descriptor increment bounds");
   if (!inc || rtv0.ptr > SIZE_MAX - inc) return 1;
   D3D12_CPU_DESCRIPTOR_HANDLE rtv1 = {rtv0.ptr + inc};
   ID3D12Device_CreateRenderTargetView(dev, rt[0], NULL, rtv0);
   ID3D12Device_CreateRenderTargetView(dev, rt[1], NULL, rtv1);

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

   const FLOAT clear[4] = {0.0f, 0.0f, 0.0f, 1.0f};
   D3D12_VIEWPORT vp = {.Width = W, .Height = H, .MaxDepth = 1.0f};
   D3D12_RECT sc = {.right = W, .bottom = H};

   for (int i = 0; i < 2; i++) {
      ID3D12Resource_Unmap(readback[i], 0, NULL);map[i] = NULL;
   }
   TEST_CALL(ID3D12GraphicsCommandList_Close(list), "close initial empty graphics list");
   TEST_CALL(ID3D12CommandAllocator_Reset(alloc), "reset idle allocator");
   TEST_CALL(ID3D12GraphicsCommandList_Reset(list, alloc, NULL), "reset checked graphics list");

   /* Control: no vertex buffer bound at all. */
   d12_test_transition(list, rt[0], D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET);
   ID3D12GraphicsCommandList_ClearRenderTargetView(list, rtv0, clear, 0, NULL);
   ID3D12GraphicsCommandList_OMSetRenderTargets(list, 1, &rtv0, FALSE, NULL);
   ID3D12GraphicsCommandList_RSSetViewports(list, 1, &vp);
   ID3D12GraphicsCommandList_RSSetScissorRects(list, 1, &sc);
   ID3D12GraphicsCommandList_SetGraphicsRootSignature(list, root_sig);
   ID3D12GraphicsCommandList_SetPipelineState(list, control_pso);
   ID3D12GraphicsCommandList_IASetPrimitiveTopology(
      list, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
   ID3D12GraphicsCommandList_DrawInstanced(list, 3, 1, 0, 0);

   /* Subject: identical state, plus the vertex buffer. */
   d12_test_transition(list, rt[1], D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET);
   ID3D12GraphicsCommandList_ClearRenderTargetView(list, rtv1, clear, 0, NULL);
   ID3D12GraphicsCommandList_OMSetRenderTargets(list, 1, &rtv1, FALSE, NULL);
   ID3D12GraphicsCommandList_SetPipelineState(list, subject_pso);
   ID3D12GraphicsCommandList_IASetVertexBuffers(list, 31, 1, &vbv);
   ID3D12GraphicsCommandList_DrawInstanced(list, 3, 1, 0, 0);

   for (int i = 0; i < 2; i++) {
      D3D12_TEXTURE_COPY_LOCATION dst = {
         .pResource = readback[i],
         .Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
         .PlacedFootprint = fp,
      };
      D3D12_TEXTURE_COPY_LOCATION src = {
         .pResource = rt[i],
         .Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
         .SubresourceIndex = 0,
      };
      d12_test_transition(list, rt[i], D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
      ID3D12GraphicsCommandList_CopyTextureRegion(list, &dst, 0, 0, 0, &src,
                                                  NULL);
      d12_test_transition(list, rt[i], D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
   }

   hr = ID3D12GraphicsCommandList_Close(list);
   CHECK(hr == S_OK, "the command list closes cleanly");
   if (hr != S_OK)
      goto teardown;

   TEST_CALL(d12_test_submit_closed(dev, list, queue, fence, 1), "exact fenced vertex-buffer draw/copy");
   for (int i = 0; i < 2; i++)
      TEST_CALL(d12_test_map(readback[i], true, (void **)&map[i]), "read actual GPU copy only after fence");
   CHECK(ID3D12Fence_GetCompletedValue(fence) >= 1,
         "the GPU completes both draws");

   {
      static bool ref[H][W];
      rasterise_reference(ref);

      unsigned ref_n = 0, ctrl_n = 0, subj_n = 0;
      unsigned ctrl_wrong = 0, subj_wrong = 0, differ = 0, colour_wrong = 0;

      for (unsigned y = 0; y < H; y++) {
         for (unsigned x = 0; x < W; x++) {
            const uint8_t *a = map[0] + y * fp.Footprint.RowPitch + x * 4;
            const uint8_t *b = map[1] + y * fp.Footprint.RowPitch + x * 4;

            bool ca = !(a[0] == 0 && a[1] == 0 && a[2] == 0);
            bool cb = !(b[0] == 0 && b[1] == 0 && b[2] == 0);

            ref_n += ref[y][x];
            ctrl_n += ca;
            subj_n += cb;
            ctrl_wrong += (ca != ref[y][x]);
            subj_wrong += (cb != ref[y][x]);
            differ += (ca != cb);

            /* Inside the triangle the subject must carry the colour that
             * came out of the buffer at offset 8. UNORM8 of 0.25/0.5/0.75
             * is 64/128/191, allowing one ULP of rounding either way. */
            if (cb) {
               int dr = (int)b[0] - 64, dg = (int)b[1] - 128,
                   db = (int)b[2] - 191;
               if (abs(dr) > 1 || abs(dg) > 1 || abs(db) > 1)
                  colour_wrong++;
            }
         }
      }

      printf("       reference %u, control %u, subject %u (of %u)\n", ref_n,
             ctrl_n, subj_n, W * H);
      printf("       control wrong %u, subject wrong %u, differ %u, "
             "colour wrong %u\n",
             ctrl_wrong, subj_wrong, differ, colour_wrong);

      CHECK(ctrl_wrong == 0,
            "the SV_VertexID control still matches the CPU rasterisation "
            "on all 65536 pixels");

      /* THE POINT OF THE SUITE. */
      CHECK(subj_wrong == 0,
            "a draw fed ONLY by a bound vertex buffer matches the CPU "
            "rasterisation on all 65536 pixels");

      CHECK(differ == 0,
            "the vertex-buffer draw reproduces the SV_VertexID draw PIXEL "
            "FOR PIXEL");

      CHECK(subj_n > 0,
            "the subject drew something at all (a fetch that returned "
            "zeroes would draw a degenerate triangle)");

      CHECK(colour_wrong == 0,
            "the SECOND attribute arrives too: colour fetched from byte "
            "offset 8 of a 24-byte stride is correct on every covered "
            "pixel");
   }

teardown:
   for (int i = 0; i < 2; i++) {
      if (map[i])
         { const D3D12_RANGE empty = {0,0}; ID3D12Resource_Unmap(readback[i], 0, &empty); }
      if (readback[i])
         ID3D12Resource_Release(readback[i]);
   }
   if (fence)
      ID3D12Fence_Release(fence);
   ID3D12GraphicsCommandList_Release(list);
   ID3D12CommandAllocator_Release(alloc);
   ID3D12CommandQueue_Release(queue);
   ID3D12DescriptorHeap_Release(rtv_heap);
   for (int i = 0; i < 2; i++) {
      if (rt[i])
         ID3D12Resource_Release(rt[i]);
   }
   ID3D12Resource_Release(vb);
   if (subject_pso)
      ID3D12PipelineState_Release(subject_pso);
   if (control_pso)
      ID3D12PipelineState_Release(control_pso);
   ID3D12RootSignature_Release(root_sig);
   ID3D12Device_Release(dev);

   printf(failures ? "G10 VERTEX BUFFERS: %d FAILURES\n"
                   : "G10 VERTEX BUFFERS: ALL PASS\n",
          failures);
   free(tri);
   free(solid);
   free(vbvs);
   free(vbps);
   return failures ? 1 : 0;
}
