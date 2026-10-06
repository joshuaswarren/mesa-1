/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * G12: the three draw SHAPES of Microsoft's D3D12HelloWorld samples, on one
 * vertex-buffer triangle, graded against a CPU rasterisation.
 *
 * The samples were the first real rendering workloads through Proton, and
 * on G17P three of the four stopped at frame 0 on the native graphics
 * admission gate although the frozen g4_triangle (SV_VertexID, empty root
 * signature, no heap) passed. Each sample adds exactly one declaration the
 * frozen suites never combined with a hardware vertex-stage draw:
 *
 *   HelloTriangle      a bound vertex buffer, so the vertex stage consumes
 *                      the vertex-attribute parameter block
 *   HelloConstBuffers  a shader-visible CBV/SRV/UAV heap bound with a
 *                      descriptor-table root parameter, so the binding
 *                      collector uploads the heap's home cells too
 *   HelloBundles       the same draw recorded in a bundle and replayed with
 *                      ExecuteBundle
 *
 * This suite draws the g10 vertex-buffer triangle (vb_vs.dxil, which reads
 * its positions and a constant colour from the buffer and computes nothing)
 * once per shape into three targets and requires every one of them to match
 * the CPU rasterisation on all 65536 pixels, including the colour fetched
 * from byte offset 8 of the 24-byte stride. The descriptor-table shape binds
 * a real constant buffer view the shader never reads: a producer that
 * refuses the draw because a heap is bound, or that publishes the heap in
 * place of the attribute block, fails here and nowhere in the frozen 17.
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
#define SHAPES 3

static const char *const shape_name[SHAPES] = {
   "vertex buffer (HelloTriangle)",
   "vertex buffer + bound heap with a CBV table (HelloConstBuffers)",
   "vertex buffer draw replayed from a bundle (HelloBundles)",
};

/* The vertices vb_vs.dxil fetches, and the colour it carries. */
static const float TRI_NDC[3][2] = {{1.0f, -1.0f}, {-1.0f, -1.0f}, {5.0f, 1.0f}};

#define COL_R 0.25f
#define COL_G 0.50f
#define COL_B 0.75f

struct vertex {
   float x, y;          /* TEXCOORD0, R32G32_FLOAT,       offset 0  */
   float r, g, b, a;    /* TEXCOORD1, R32G32B32A32_FLOAT, offset 8  */
};

/* The oracle shared with g4_triangle and g10: D3D pixel-centre sampling
 * and the top-left fill rule over the shader's own vertices. */
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
   TEST_OBJECT(hr, dev, "checked D3D12CreateDevice dev");
   if (!dev)
      return 1;

   /* Two root signatures: the empty one HelloTriangle/HelloBundles use,
    * and HelloConstBuffers' one descriptor table holding one CBV for the
    * vertex stage. */
   D3D12_ROOT_SIGNATURE_DESC empty_rsd = {
      .Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
   };
   D3D12_DESCRIPTOR_RANGE cbv_range = {
      .RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV,
      .NumDescriptors = 1,
      .BaseShaderRegister = 0,
      .RegisterSpace = 0,
      .OffsetInDescriptorsFromTableStart = 0,
   };
   D3D12_ROOT_PARAMETER table_param = {
      .ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
      .DescriptorTable = {.NumDescriptorRanges = 1,
                          .pDescriptorRanges = &cbv_range},
      .ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX,
   };
   D3D12_ROOT_SIGNATURE_DESC table_rsd = {
      .NumParameters = 1,
      .pParameters = &table_param,
      .Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |
               D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS |
               D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS |
               D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS |
               D3D12_ROOT_SIGNATURE_FLAG_DENY_PIXEL_SHADER_ROOT_ACCESS,
   };
   ID3D12RootSignature *root_sig[2] = {NULL, NULL};
   const D3D12_ROOT_SIGNATURE_DESC *rsds[2] = {&empty_rsd, &table_rsd};
   for (int i = 0; i < 2; i++) {
      ID3DBlob *sig = NULL;
      TEST_OBJECT(D3D12SerializeRootSignature(rsds[i],
                                              D3D_ROOT_SIGNATURE_VERSION_1,
                                              &sig, NULL),
                  sig, "checked D3D12SerializeRootSignature sig");
      TEST_OBJECT(ID3D12Device_CreateRootSignature(
                     dev, 0, ID3D10Blob_GetBufferPointer(sig),
                     ID3D10Blob_GetBufferSize(sig), &IID_ID3D12RootSignature,
                     (void **)&root_sig[i]),
                  root_sig[i], "checked ID3D12Device_CreateRootSignature");
      ID3D10Blob_Release(sig);
   }
   CHECK(root_sig[0] && root_sig[1],
         "create the empty and the CBV-table root signatures");

   /* nir_to_dxil names generic vertex inputs TEXCOORD<driver_location>. */
   D3D12_INPUT_ELEMENT_DESC elems[] = {
      {.SemanticName = "TEXCOORD",
       .SemanticIndex = 0,
       .Format = DXGI_FORMAT_R32G32_FLOAT,
       .InputSlot = 0,
       .AlignedByteOffset = 0,
       .InputSlotClass = D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA},
      {.SemanticName = "TEXCOORD",
       .SemanticIndex = 1,
       .Format = DXGI_FORMAT_R32G32B32A32_FLOAT,
       .InputSlot = 0,
       .AlignedByteOffset = 8,
       .InputSlotClass = D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA},
   };

   D3D12_GRAPHICS_PIPELINE_STATE_DESC pd = {
      .VS = {.pShaderBytecode = vbvs, .BytecodeLength = vs_size},
      .PS = {.pShaderBytecode = vbps, .BytecodeLength = ps_size},
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

   /* One PSO per root signature; the shaders are the same bytes. */
   ID3D12PipelineState *pso[2] = {NULL, NULL};
   for (int i = 0; i < 2; i++) {
      pd.pRootSignature = root_sig[i];
      TEST_OBJECT(ID3D12Device_CreateGraphicsPipelineState(
                     dev, &pd, &IID_ID3D12PipelineState, (void **)&pso[i]),
                  pso[i], "checked ID3D12Device_CreateGraphicsPipelineState");
   }
   CHECK(pso[0] && pso[1], "both PSOs compile with the input layout");

   /* Vertex buffer and the constant buffer the table points at. The
    * constant buffer holds a pattern the shader does not read; its only
    * job is to be a valid CBV in a bound shader-visible heap. */
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
   TEST_OBJECT(ID3D12Device_CreateCommittedResource(
                  dev, &up, D3D12_HEAP_FLAG_NONE, &bd,
                  D3D12_RESOURCE_STATE_GENERIC_READ, NULL, &IID_ID3D12Resource,
                  (void **)&vb),
               vb, "checked ID3D12Device_CreateCommittedResource vb");
   void *vbmap = NULL;
   TEST_CALL(d12_test_map(vb, false, &vbmap), "checked vertex upload map");
   memcpy(vbmap, verts, sizeof(verts));
   ID3D12Resource_Unmap(vb, 0, NULL);
   D3D12_VERTEX_BUFFER_VIEW vbv = {
      .BufferLocation = ID3D12Resource_GetGPUVirtualAddress(vb),
      .SizeInBytes = sizeof(verts),
      .StrideInBytes = sizeof(struct vertex),
   };
   CHECK(vbv.BufferLocation != 0, "the vertex buffer has a GPU virtual address");

   bd.Width = 256; /* D3D12's constant buffer alignment */
   ID3D12Resource *cb = NULL;
   TEST_OBJECT(ID3D12Device_CreateCommittedResource(
                  dev, &up, D3D12_HEAP_FLAG_NONE, &bd,
                  D3D12_RESOURCE_STATE_GENERIC_READ, NULL, &IID_ID3D12Resource,
                  (void **)&cb),
               cb, "checked ID3D12Device_CreateCommittedResource cb");
   void *cbmap = NULL;
   TEST_CALL(d12_test_map(cb, false, &cbmap), "checked constant upload map");
   memset(cbmap, 0x3f, 256);
   ID3D12Resource_Unmap(cb, 0, NULL);

   D3D12_DESCRIPTOR_HEAP_DESC cbv_hd = {
      .Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
      .NumDescriptors = 1,
      .Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
   };
   ID3D12DescriptorHeap *cbv_heap = NULL;
   TEST_OBJECT(ID3D12Device_CreateDescriptorHeap(dev, &cbv_hd,
                                                 &IID_ID3D12DescriptorHeap,
                                                 (void **)&cbv_heap),
               cbv_heap, "checked ID3D12Device_CreateDescriptorHeap cbv_heap");
   D3D12_CONSTANT_BUFFER_VIEW_DESC cbvd = {
      .BufferLocation = ID3D12Resource_GetGPUVirtualAddress(cb),
      .SizeInBytes = 256,
   };
   ID3D12Device_CreateConstantBufferView(
      dev, &cbvd, d12_test_descriptor_heap_get_cpu_start(cbv_heap));
   D3D12_GPU_DESCRIPTOR_HANDLE cbv_gpu =
      d12_test_descriptor_heap_get_gpu_start(cbv_heap);
   CHECK(cbv_gpu.ptr != 0, "the shader-visible heap has a GPU handle");

   /* Three render targets, one per shape, read back through the public
    * copy path. */
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
   ID3D12Resource *rt[SHAPES] = {NULL};
   for (int i = 0; i < SHAPES; i++) {
      TEST_OBJECT(ID3D12Device_CreateCommittedResource(
                     dev, &heap, D3D12_HEAP_FLAG_NONE, &td,
                     D3D12_RESOURCE_STATE_COMMON, NULL, &IID_ID3D12Resource,
                     (void **)&rt[i]),
                  rt[i], "checked ID3D12Device_CreateCommittedResource rt[i]");
   }

   D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp = {0};
   UINT rows = 0;
   UINT64 row_bytes = 0, total = 0;
   ID3D12Device_GetCopyableFootprints(dev, &td, 0, 1, 0, &fp, &rows,
                                      &row_bytes, &total);
   CHECK(fp.Footprint.Width == W && fp.Footprint.Height == H && rows == H &&
            row_bytes == (UINT64)W * 4 && fp.Footprint.RowPitch >= W * 4 &&
            fp.Footprint.RowPitch % 256 == 0 && fp.Footprint.Depth == 1 &&
            total >= (UINT64)fp.Footprint.RowPitch * H,
         "get a valid render-target readback footprint");
   if (fp.Footprint.Width != W || fp.Footprint.Height != H || rows != H ||
       row_bytes != (UINT64)W * 4 || fp.Footprint.RowPitch < W * 4 ||
       fp.Footprint.RowPitch % 256 || fp.Footprint.Depth != 1 ||
       total < (UINT64)fp.Footprint.RowPitch * H)
      return 1;

   D3D12_HEAP_PROPERTIES readback_heap = {.Type = D3D12_HEAP_TYPE_READBACK};
   D3D12_RESOURCE_DESC readback_desc = {
      .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
      .Width = total,
      .Height = 1,
      .DepthOrArraySize = 1,
      .MipLevels = 1,
      .SampleDesc = {.Count = 1},
      .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
   };
   ID3D12Resource *readback[SHAPES] = {NULL};
   uint8_t *map[SHAPES] = {NULL};
   for (int i = 0; i < SHAPES; i++) {
      TEST_OBJECT(ID3D12Device_CreateCommittedResource(
                     dev, &readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc,
                     D3D12_RESOURCE_STATE_COPY_DEST, NULL, &IID_ID3D12Resource,
                     (void **)&readback[i]),
                  readback[i],
                  "checked ID3D12Device_CreateCommittedResource readback[i]");
      TEST_CALL(d12_test_map(readback[i], false, (void **)&map[i]),
                "checked readback poison map");
      memset(map[i], 0x55, (size_t)total);
      ID3D12Resource_Unmap(readback[i], 0, NULL);
      map[i] = NULL;
   }

   D3D12_DESCRIPTOR_HEAP_DESC hd = {.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                                    .NumDescriptors = SHAPES};
   ID3D12DescriptorHeap *rtv_heap = NULL;
   TEST_OBJECT(ID3D12Device_CreateDescriptorHeap(dev, &hd,
                                                 &IID_ID3D12DescriptorHeap,
                                                 (void **)&rtv_heap),
               rtv_heap, "checked ID3D12Device_CreateDescriptorHeap rtv_heap");
   UINT inc = ID3D12Device_GetDescriptorHandleIncrementSize(
      dev, D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
   D3D12_CPU_DESCRIPTOR_HANDLE rtv[SHAPES];
   rtv[0] = d12_test_descriptor_heap_get_cpu_start(rtv_heap);
   CHECK(rtv[0].ptr != 0 && inc != 0, "valid RTV descriptor handle");
   if (!rtv[0].ptr || !inc)
      return 1;
   for (int i = 0; i < SHAPES; i++) {
      rtv[i].ptr = rtv[0].ptr + (SIZE_T)i * inc;
      ID3D12Device_CreateRenderTargetView(dev, rt[i], NULL, rtv[i]);
   }

   D3D12_COMMAND_QUEUE_DESC qd = {.Type = D3D12_COMMAND_LIST_TYPE_DIRECT};
   ID3D12CommandQueue *queue = NULL;
   TEST_OBJECT(ID3D12Device_CreateCommandQueue(dev, &qd, &IID_ID3D12CommandQueue,
                                               (void **)&queue),
               queue, "checked ID3D12Device_CreateCommandQueue queue");
   ID3D12CommandAllocator *alloc = NULL;
   TEST_OBJECT(ID3D12Device_CreateCommandAllocator(
                  dev, D3D12_COMMAND_LIST_TYPE_DIRECT,
                  &IID_ID3D12CommandAllocator, (void **)&alloc),
               alloc, "checked ID3D12Device_CreateCommandAllocator alloc");
   ID3D12GraphicsCommandList *list = NULL;
   TEST_OBJECT(ID3D12Device_CreateCommandList(
                  dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, NULL,
                  &IID_ID3D12GraphicsCommandList, (void **)&list),
               list, "checked ID3D12Device_CreateCommandList list");
   ID3D12Fence *fence = NULL;
   TEST_OBJECT(ID3D12Device_CreateFence(dev, 0, D3D12_FENCE_FLAG_NONE,
                                        &IID_ID3D12Fence, (void **)&fence),
               fence, "checked ID3D12Device_CreateFence fence");

   /* The bundle, recorded exactly as HelloBundles records its draw: PSO,
    * topology, vertex buffer, DrawInstanced. HelloBundles sets the root
    * signature in the bundle too; D3D12 requires a bundle to set one before
    * drawing. */
   ID3D12CommandAllocator *bundle_alloc = NULL;
   TEST_OBJECT(ID3D12Device_CreateCommandAllocator(
                  dev, D3D12_COMMAND_LIST_TYPE_BUNDLE,
                  &IID_ID3D12CommandAllocator, (void **)&bundle_alloc),
               bundle_alloc,
               "checked ID3D12Device_CreateCommandAllocator bundle_alloc");
   ID3D12GraphicsCommandList *bundle = NULL;
   TEST_OBJECT(ID3D12Device_CreateCommandList(
                  dev, 0, D3D12_COMMAND_LIST_TYPE_BUNDLE, bundle_alloc, pso[0],
                  &IID_ID3D12GraphicsCommandList, (void **)&bundle),
               bundle, "checked ID3D12Device_CreateCommandList bundle");
   ID3D12GraphicsCommandList_SetGraphicsRootSignature(bundle, root_sig[0]);
   ID3D12GraphicsCommandList_SetPipelineState(bundle, pso[0]);
   ID3D12GraphicsCommandList_IASetPrimitiveTopology(
      bundle, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
   ID3D12GraphicsCommandList_IASetVertexBuffers(bundle, 0, 1, &vbv);
   ID3D12GraphicsCommandList_DrawInstanced(bundle, 3, 1, 0, 0);
   TEST_CALL(ID3D12GraphicsCommandList_Close(bundle), "close the bundle");

   const FLOAT clear[4] = {0.0f, 0.0f, 0.0f, 1.0f};
   D3D12_VIEWPORT vp = {.Width = W, .Height = H, .MaxDepth = 1.0f};
   D3D12_RECT sc = {.right = W, .bottom = H};

   TEST_CALL(ID3D12GraphicsCommandList_Close(list),
             "close initial empty graphics list");
   TEST_CALL(ID3D12CommandAllocator_Reset(alloc), "reset idle allocator");
   TEST_CALL(ID3D12GraphicsCommandList_Reset(list, alloc, NULL),
             "reset checked graphics list");

   for (int i = 0; i < SHAPES; i++) {
      d12_test_transition(list, rt[i], D3D12_RESOURCE_STATE_COMMON,
                          D3D12_RESOURCE_STATE_RENDER_TARGET);
      ID3D12GraphicsCommandList_ClearRenderTargetView(list, rtv[i], clear, 0,
                                                      NULL);
      ID3D12GraphicsCommandList_OMSetRenderTargets(list, 1, &rtv[i], FALSE,
                                                   NULL);
      ID3D12GraphicsCommandList_RSSetViewports(list, 1, &vp);
      ID3D12GraphicsCommandList_RSSetScissorRects(list, 1, &sc);

      switch (i) {
      case 0:
         ID3D12GraphicsCommandList_SetGraphicsRootSignature(list, root_sig[0]);
         ID3D12GraphicsCommandList_SetPipelineState(list, pso[0]);
         ID3D12GraphicsCommandList_IASetPrimitiveTopology(
            list, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
         ID3D12GraphicsCommandList_IASetVertexBuffers(list, 0, 1, &vbv);
         ID3D12GraphicsCommandList_DrawInstanced(list, 3, 1, 0, 0);
         break;
      case 1: {
         /* HelloConstBuffers' PopulateCommandList order. */
         ID3D12GraphicsCommandList_SetGraphicsRootSignature(list, root_sig[1]);
         ID3D12DescriptorHeap *heaps[] = {cbv_heap};
         ID3D12GraphicsCommandList_SetDescriptorHeaps(list, 1, heaps);
         ID3D12GraphicsCommandList_SetGraphicsRootDescriptorTable(list, 0,
                                                                  cbv_gpu);
         ID3D12GraphicsCommandList_SetPipelineState(list, pso[1]);
         ID3D12GraphicsCommandList_IASetPrimitiveTopology(
            list, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
         ID3D12GraphicsCommandList_IASetVertexBuffers(list, 0, 1, &vbv);
         ID3D12GraphicsCommandList_DrawInstanced(list, 3, 1, 0, 0);
         break;
      }
      case 2:
         ID3D12GraphicsCommandList_SetGraphicsRootSignature(list, root_sig[0]);
         ID3D12GraphicsCommandList_ExecuteBundle(list, bundle);
         break;
      }

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
      d12_test_transition(list, rt[i], D3D12_RESOURCE_STATE_RENDER_TARGET,
                          D3D12_RESOURCE_STATE_COPY_SOURCE);
      ID3D12GraphicsCommandList_CopyTextureRegion(list, &dst, 0, 0, 0, &src,
                                                  NULL);
      d12_test_transition(list, rt[i], D3D12_RESOURCE_STATE_COPY_SOURCE,
                          D3D12_RESOURCE_STATE_COMMON);
   }

   hr = ID3D12GraphicsCommandList_Close(list);
   CHECK(hr == S_OK, "the command list with all three shapes closes cleanly");
   if (hr != S_OK)
      goto teardown;

   TEST_CALL(d12_test_submit_closed(dev, list, queue, fence, 1),
             "exact fenced three-shape draw/copy");
   for (int i = 0; i < SHAPES; i++)
      TEST_CALL(d12_test_map(readback[i], true, (void **)&map[i]),
                "read actual GPU copy only after fence");

   {
      static bool ref[H][W];
      rasterise_reference(ref);

      unsigned ref_n = 0;
      for (unsigned y = 0; y < H; y++)
         for (unsigned x = 0; x < W; x++)
            ref_n += ref[y][x];

      for (int i = 0; i < SHAPES; i++) {
         unsigned n = 0, wrong = 0, colour_wrong = 0, differ = 0;
         for (unsigned y = 0; y < H; y++) {
            for (unsigned x = 0; x < W; x++) {
               const uint8_t *b = map[i] + y * fp.Footprint.RowPitch + x * 4;
               const uint8_t *a = map[0] + y * fp.Footprint.RowPitch + x * 4;
               bool cb_ = !(b[0] == 0 && b[1] == 0 && b[2] == 0);
               bool ca = !(a[0] == 0 && a[1] == 0 && a[2] == 0);
               n += cb_;
               wrong += (cb_ != ref[y][x]);
               differ += (cb_ != ca);
               /* UNORM8 of 0.25/0.5/0.75 is 64/128/191, one ULP either way. */
               if (cb_) {
                  int dr = (int)b[0] - 64, dg = (int)b[1] - 128,
                      db = (int)b[2] - 191;
                  if (abs(dr) > 1 || abs(dg) > 1 || abs(db) > 1)
                     colour_wrong++;
               }
            }
         }
         printf("       shape %d: reference %u, drawn %u (of %u), wrong %u, "
                "colour wrong %u, differ from shape 0 %u\n",
                i, ref_n, n, W * H, wrong, colour_wrong, differ);
         char what[160];
         snprintf(what, sizeof(what),
                  "%s matches the CPU rasterisation on all 65536 pixels",
                  shape_name[i]);
         CHECK(wrong == 0, what);
         snprintf(what, sizeof(what),
                  "%s fetched the colour from byte offset 8 on every pixel",
                  shape_name[i]);
         CHECK(colour_wrong == 0, what);
         snprintf(what, sizeof(what), "%s drew something at all",
                  shape_name[i]);
         CHECK(n > 0, what);
         if (i) {
            snprintf(what, sizeof(what),
                     "%s reproduces the plain vertex-buffer draw pixel for pixel",
                     shape_name[i]);
            CHECK(differ == 0, what);
         }
      }
   }

teardown:
   for (int i = 0; i < SHAPES; i++) {
      if (map[i]) {
         const D3D12_RANGE empty = {0, 0};
         ID3D12Resource_Unmap(readback[i], 0, &empty);
      }
      if (readback[i])
         ID3D12Resource_Release(readback[i]);
      if (rt[i])
         ID3D12Resource_Release(rt[i]);
   }
   ID3D12Fence_Release(fence);
   ID3D12GraphicsCommandList_Release(bundle);
   ID3D12CommandAllocator_Release(bundle_alloc);
   ID3D12GraphicsCommandList_Release(list);
   ID3D12CommandAllocator_Release(alloc);
   ID3D12CommandQueue_Release(queue);
   ID3D12DescriptorHeap_Release(rtv_heap);
   ID3D12DescriptorHeap_Release(cbv_heap);
   ID3D12Resource_Release(cb);
   ID3D12Resource_Release(vb);
   for (int i = 0; i < 2; i++) {
      ID3D12PipelineState_Release(pso[i]);
      ID3D12RootSignature_Release(root_sig[i]);
   }
   ID3D12Device_Release(dev);

   printf(failures ? "G12 SAMPLE SHAPES: %d FAILURES\n"
                   : "G12 SAMPLE SHAPES: ALL PASS\n",
          failures);
   free(vbvs);
   free(vbps);
   return failures ? 1 : 0;
}
