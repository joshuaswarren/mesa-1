/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * G13: a textured triangle, the HelloTexture sample's shape.
 *
 * Every graphics probe before this one was root-only: colours came from
 * vertex data or constants, never from a sampled texture. HelloTexture
 * under Proton failed at pipeline creation on G17P because a fragment
 * sample has no USC texture/sampler words there; it goes through the
 * native texture0/sampler0 pointer ABI or not at all.
 *
 * The draw binds what the sample binds: a shader-visible CBV/SRV/UAV heap
 * holding one SRV of a 2x2 RGBA8 texture reached through a descriptor-table
 * root parameter, and a static point/border sampler (s0) in the root
 * signature. tex_vs carries the vertex buffer's uv to VAR0 and tex_ps
 * samples t0 through s0 with it. The uv is chosen so that it equals the
 * pixel's normalised position across the whole target, so every covered
 * pixel must show the texel of its quadrant exactly: a sampler that was
 * never bound, a texture pointer at the wrong descriptor, or a swapped
 * row stride each produce a different, wrong colour pattern rather than
 * nothing. Coverage is graded against the CPU rasterisation g4/g10/g12 use.
 *
 * With a third argument "hello" the vertex layout is the sample's own
 * (POSITION R32G32B32_FLOAT at 0, TEXCOORD R32G32_FLOAT at 12), so the
 * shaders can be the sample's shaders.hlsl compiled to SM5 DXBC, which is
 * what HelloTexture hands the driver under Wine's d3dcompiler_47.
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

static const float TRI_NDC[3][2] = {{1.0f, -1.0f}, {-1.0f, -1.0f}, {5.0f, 1.0f}};

/* Texel (tx, ty) colours, RGBA8. Four distinct, saturated values so a
 * wrong quadrant is unmistakable. */
static const uint8_t TEXELS[2][2][4] = {
   {{255, 0, 0, 255}, {0, 255, 0, 255}},   /* row 0: red, green */
   {{0, 0, 255, 255}, {255, 255, 0, 255}}, /* row 1: blue, yellow */
};

struct vertex {
   float x, y;   /* TEXCOORD0, R32G32_FLOAT, offset 0 */
   float u, v;   /* TEXCOORD1, R32G32_FLOAT, offset 8 */
};

/* HelloTexture's Vertex: float3 position, float2 uv. */
struct hello_vertex {
   float x, y, z; /* POSITION, R32G32B32_FLOAT, offset 0 */
   float u, v;    /* TEXCOORD, R32G32_FLOAT, offset 12 */
};

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
      printf("usage: %s tex_vs.dxil tex_ps.dxil [hello]\n", argv[0]);
      return 2;
   }
   const bool hello = argc > 3 && !strcmp(argv[3], "hello");
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
   TEST_OBJECT(hr, dev, "checked D3D12CreateDevice dev");

   /* HelloTexture's root signature: one SRV descriptor table for the pixel
    * stage and one static point/border sampler at s0. */
   D3D12_DESCRIPTOR_RANGE srv_range = {
      .RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
      .NumDescriptors = 1,
      .BaseShaderRegister = 0,
      .RegisterSpace = 0,
      .OffsetInDescriptorsFromTableStart = 0,
   };
   D3D12_ROOT_PARAMETER table_param = {
      .ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
      .DescriptorTable = {.NumDescriptorRanges = 1,
                          .pDescriptorRanges = &srv_range},
      .ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL,
   };
   D3D12_STATIC_SAMPLER_DESC sampler = {
      .Filter = D3D12_FILTER_MIN_MAG_MIP_POINT,
      .AddressU = D3D12_TEXTURE_ADDRESS_MODE_BORDER,
      .AddressV = D3D12_TEXTURE_ADDRESS_MODE_BORDER,
      .AddressW = D3D12_TEXTURE_ADDRESS_MODE_BORDER,
      .MipLODBias = 0,
      .MaxAnisotropy = 0,
      .ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER,
      .BorderColor = D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK,
      .MinLOD = 0.0f,
      .MaxLOD = D3D12_FLOAT32_MAX,
      .ShaderRegister = 0,
      .RegisterSpace = 0,
      .ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL,
   };
   D3D12_ROOT_SIGNATURE_DESC rsd = {
      .NumParameters = 1,
      .pParameters = &table_param,
      .NumStaticSamplers = 1,
      .pStaticSamplers = &sampler,
      .Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
   };
   ID3DBlob *sig = NULL;
   TEST_OBJECT(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1,
                                           &sig, NULL),
               sig, "checked D3D12SerializeRootSignature sig");
   ID3D12RootSignature *root_sig = NULL;
   TEST_OBJECT(ID3D12Device_CreateRootSignature(
                  dev, 0, ID3D10Blob_GetBufferPointer(sig),
                  ID3D10Blob_GetBufferSize(sig), &IID_ID3D12RootSignature,
                  (void **)&root_sig),
               root_sig, "create the SRV table + static sampler root signature");
   ID3D10Blob_Release(sig);

   D3D12_INPUT_ELEMENT_DESC elems[] = {
      {.SemanticName = "TEXCOORD",
       .SemanticIndex = 0,
       .Format = DXGI_FORMAT_R32G32_FLOAT,
       .InputSlot = 0,
       .AlignedByteOffset = 0,
       .InputSlotClass = D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA},
      {.SemanticName = "TEXCOORD",
       .SemanticIndex = 1,
       .Format = DXGI_FORMAT_R32G32_FLOAT,
       .InputSlot = 0,
       .AlignedByteOffset = 8,
       .InputSlotClass = D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA},
   };
   D3D12_INPUT_ELEMENT_DESC hello_elems[] = {
      {.SemanticName = "POSITION",
       .SemanticIndex = 0,
       .Format = DXGI_FORMAT_R32G32B32_FLOAT,
       .InputSlot = 0,
       .AlignedByteOffset = 0,
       .InputSlotClass = D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA},
      {.SemanticName = "TEXCOORD",
       .SemanticIndex = 0,
       .Format = DXGI_FORMAT_R32G32_FLOAT,
       .InputSlot = 0,
       .AlignedByteOffset = 12,
       .InputSlotClass = D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA},
   };
   D3D12_GRAPHICS_PIPELINE_STATE_DESC pd = {
      .pRootSignature = root_sig,
      .VS = {.pShaderBytecode = vs, .BytecodeLength = vs_size},
      .PS = {.pShaderBytecode = ps, .BytecodeLength = ps_size},
      .InputLayout = {hello ? hello_elems : elems, 2},
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
   TEST_OBJECT(ID3D12Device_CreateGraphicsPipelineState(
                  dev, &pd, &IID_ID3D12PipelineState, (void **)&pso),
               pso, "the sampling PSO compiles");

   /* uv = pixel-normalised position: (x * 0.5 + 0.5, 0.5 - y * 0.5). */
   struct vertex verts[3];
   struct hello_vertex hello_verts[3];
   for (int i = 0; i < 3; i++) {
      verts[i] = (struct vertex){
         .x = TRI_NDC[i][0], .y = TRI_NDC[i][1],
         .u = TRI_NDC[i][0] * 0.5f + 0.5f, .v = 0.5f - TRI_NDC[i][1] * 0.5f,
      };
      hello_verts[i] = (struct hello_vertex){
         .x = verts[i].x, .y = verts[i].y, .z = 0.0f,
         .u = verts[i].u, .v = verts[i].v,
      };
   }
   const void *vdata = hello ? (const void *)hello_verts : (const void *)verts;
   const UINT vdata_size = hello ? sizeof(hello_verts) : sizeof(verts);
   const UINT vstride =
      hello ? sizeof(struct hello_vertex) : sizeof(struct vertex);
   D3D12_HEAP_PROPERTIES up = {.Type = D3D12_HEAP_TYPE_UPLOAD};
   D3D12_RESOURCE_DESC bd = {
      .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
      .Width = vdata_size,
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
               vb, "create the vertex buffer");
   void *vbmap = NULL;
   TEST_CALL(d12_test_map(vb, false, &vbmap), "map the vertex buffer");
   memcpy(vbmap, vdata, vdata_size);
   ID3D12Resource_Unmap(vb, 0, NULL);
   D3D12_VERTEX_BUFFER_VIEW vbv = {
      .BufferLocation = ID3D12Resource_GetGPUVirtualAddress(vb),
      .SizeInBytes = vdata_size,
      .StrideInBytes = vstride,
   };

   /* The 2x2 texture in a DEFAULT heap, filled through an UPLOAD buffer
    * with the footprint the device reports, exactly as the sample does. */
   D3D12_HEAP_PROPERTIES heap = {.Type = D3D12_HEAP_TYPE_DEFAULT};
   D3D12_RESOURCE_DESC texd = {
      .Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
      .Width = 2,
      .Height = 2,
      .DepthOrArraySize = 1,
      .MipLevels = 1,
      .Format = DXGI_FORMAT_R8G8B8A8_UNORM,
      .SampleDesc = {.Count = 1},
   };
   ID3D12Resource *tex = NULL;
   TEST_OBJECT(ID3D12Device_CreateCommittedResource(
                  dev, &heap, D3D12_HEAP_FLAG_NONE, &texd,
                  D3D12_RESOURCE_STATE_COPY_DEST, NULL, &IID_ID3D12Resource,
                  (void **)&tex),
               tex, "create the 2x2 texture");
   D3D12_PLACED_SUBRESOURCE_FOOTPRINT tex_fp = {0};
   UINT tex_rows = 0;
   UINT64 tex_row_bytes = 0, tex_total = 0;
   ID3D12Device_GetCopyableFootprints(dev, &texd, 0, 1, 0, &tex_fp, &tex_rows,
                                      &tex_row_bytes, &tex_total);
   CHECK(tex_fp.Footprint.Width == 2 && tex_fp.Footprint.Height == 2 &&
            tex_rows == 2 && tex_row_bytes == 8 &&
            tex_fp.Footprint.RowPitch >= 8 && tex_total >= 16,
         "the texture upload footprint is valid");
   bd.Width = tex_total;
   ID3D12Resource *tex_upload = NULL;
   TEST_OBJECT(ID3D12Device_CreateCommittedResource(
                  dev, &up, D3D12_HEAP_FLAG_NONE, &bd,
                  D3D12_RESOURCE_STATE_GENERIC_READ, NULL, &IID_ID3D12Resource,
                  (void **)&tex_upload),
               tex_upload, "create the texture upload buffer");
   uint8_t *tmap = NULL;
   TEST_CALL(d12_test_map(tex_upload, false, (void **)&tmap),
             "map the texture upload buffer");
   memset(tmap, 0, (size_t)tex_total);
   for (unsigned y = 0; y < 2; y++)
      memcpy(tmap + tex_fp.Offset + (size_t)y * tex_fp.Footprint.RowPitch,
             TEXELS[y], 8);
   ID3D12Resource_Unmap(tex_upload, 0, NULL);

   D3D12_DESCRIPTOR_HEAP_DESC srv_hd = {
      .Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
      .NumDescriptors = 1,
      .Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
   };
   ID3D12DescriptorHeap *srv_heap = NULL;
   TEST_OBJECT(ID3D12Device_CreateDescriptorHeap(dev, &srv_hd,
                                                 &IID_ID3D12DescriptorHeap,
                                                 (void **)&srv_heap),
               srv_heap, "create the shader-visible SRV heap");
   D3D12_SHADER_RESOURCE_VIEW_DESC srvd = {
      .Format = DXGI_FORMAT_R8G8B8A8_UNORM,
      .ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D,
      .Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
      .Texture2D = {.MipLevels = 1},
   };
   ID3D12Device_CreateShaderResourceView(
      dev, tex, &srvd, d12_test_descriptor_heap_get_cpu_start(srv_heap));
   D3D12_GPU_DESCRIPTOR_HANDLE srv_gpu =
      d12_test_descriptor_heap_get_gpu_start(srv_heap);
   CHECK(srv_gpu.ptr != 0, "the SRV heap has a GPU handle");

   /* Render target and its readback. */
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
   ID3D12Resource *rt = NULL;
   TEST_OBJECT(ID3D12Device_CreateCommittedResource(
                  dev, &heap, D3D12_HEAP_FLAG_NONE, &td,
                  D3D12_RESOURCE_STATE_COMMON, NULL, &IID_ID3D12Resource,
                  (void **)&rt),
               rt, "create the render target");
   D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp = {0};
   UINT rows = 0;
   UINT64 row_bytes = 0, total = 0;
   ID3D12Device_GetCopyableFootprints(dev, &td, 0, 1, 0, &fp, &rows,
                                      &row_bytes, &total);
   if (fp.Footprint.Width != W || fp.Footprint.Height != H || rows != H ||
       row_bytes != (UINT64)W * 4 || fp.Footprint.RowPitch < W * 4 ||
       fp.Footprint.RowPitch % 256 || fp.Footprint.Depth != 1 ||
       total < (UINT64)fp.Footprint.RowPitch * H) {
      CHECK(false, "get a valid render-target readback footprint");
      return 1;
   }
   D3D12_HEAP_PROPERTIES readback_heap = {.Type = D3D12_HEAP_TYPE_READBACK};
   D3D12_RESOURCE_DESC readback_desc = bd;
   readback_desc.Width = total;
   ID3D12Resource *readback = NULL;
   TEST_OBJECT(ID3D12Device_CreateCommittedResource(
                  dev, &readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc,
                  D3D12_RESOURCE_STATE_COPY_DEST, NULL, &IID_ID3D12Resource,
                  (void **)&readback),
               readback, "create the readback buffer");
   uint8_t *map = NULL;
   TEST_CALL(d12_test_map(readback, false, (void **)&map),
             "poison the readback buffer");
   memset(map, 0x55, (size_t)total);
   ID3D12Resource_Unmap(readback, 0, NULL);
   map = NULL;

   D3D12_DESCRIPTOR_HEAP_DESC hd = {.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                                    .NumDescriptors = 1};
   ID3D12DescriptorHeap *rtv_heap = NULL;
   TEST_OBJECT(ID3D12Device_CreateDescriptorHeap(dev, &hd,
                                                 &IID_ID3D12DescriptorHeap,
                                                 (void **)&rtv_heap),
               rtv_heap, "create the RTV heap");
   D3D12_CPU_DESCRIPTOR_HANDLE rtv =
      d12_test_descriptor_heap_get_cpu_start(rtv_heap);
   ID3D12Device_CreateRenderTargetView(dev, rt, NULL, rtv);

   D3D12_COMMAND_QUEUE_DESC qd = {.Type = D3D12_COMMAND_LIST_TYPE_DIRECT};
   ID3D12CommandQueue *queue = NULL;
   TEST_OBJECT(ID3D12Device_CreateCommandQueue(dev, &qd, &IID_ID3D12CommandQueue,
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
               list, "create the command list");
   ID3D12Fence *fence = NULL;
   TEST_OBJECT(ID3D12Device_CreateFence(dev, 0, D3D12_FENCE_FLAG_NONE,
                                        &IID_ID3D12Fence, (void **)&fence),
               fence, "create the fence");

   /* Upload the texture, then draw, in one list: the sample does the upload
    * in its init list and the draw in the frame list, with a fence between;
    * the barrier here is the same ordering. */
   {
      D3D12_TEXTURE_COPY_LOCATION dst = {
         .pResource = tex,
         .Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
         .SubresourceIndex = 0,
      };
      D3D12_TEXTURE_COPY_LOCATION src = {
         .pResource = tex_upload,
         .Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
         .PlacedFootprint = tex_fp,
      };
      ID3D12GraphicsCommandList_CopyTextureRegion(list, &dst, 0, 0, 0, &src,
                                                  NULL);
      d12_test_transition(list, tex, D3D12_RESOURCE_STATE_COPY_DEST,
                          D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
   }

   const FLOAT clear[4] = {0.0f, 0.0f, 0.0f, 1.0f};
   D3D12_VIEWPORT vp = {.Width = W, .Height = H, .MaxDepth = 1.0f};
   D3D12_RECT sc = {.right = W, .bottom = H};
   d12_test_transition(list, rt, D3D12_RESOURCE_STATE_COMMON,
                       D3D12_RESOURCE_STATE_RENDER_TARGET);
   ID3D12GraphicsCommandList_ClearRenderTargetView(list, rtv, clear, 0, NULL);
   ID3D12GraphicsCommandList_OMSetRenderTargets(list, 1, &rtv, FALSE, NULL);
   ID3D12GraphicsCommandList_RSSetViewports(list, 1, &vp);
   ID3D12GraphicsCommandList_RSSetScissorRects(list, 1, &sc);
   /* HelloTexture's PopulateCommandList order. */
   ID3D12GraphicsCommandList_SetGraphicsRootSignature(list, root_sig);
   ID3D12DescriptorHeap *heaps[] = {srv_heap};
   ID3D12GraphicsCommandList_SetDescriptorHeaps(list, 1, heaps);
   ID3D12GraphicsCommandList_SetGraphicsRootDescriptorTable(list, 0, srv_gpu);
   ID3D12GraphicsCommandList_SetPipelineState(list, pso);
   ID3D12GraphicsCommandList_IASetPrimitiveTopology(
      list, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
   ID3D12GraphicsCommandList_IASetVertexBuffers(list, 0, 1, &vbv);
   ID3D12GraphicsCommandList_DrawInstanced(list, 3, 1, 0, 0);

   {
      D3D12_TEXTURE_COPY_LOCATION dst = {
         .pResource = readback,
         .Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
         .PlacedFootprint = fp,
      };
      D3D12_TEXTURE_COPY_LOCATION src = {
         .pResource = rt,
         .Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
         .SubresourceIndex = 0,
      };
      d12_test_transition(list, rt, D3D12_RESOURCE_STATE_RENDER_TARGET,
                          D3D12_RESOURCE_STATE_COPY_SOURCE);
      ID3D12GraphicsCommandList_CopyTextureRegion(list, &dst, 0, 0, 0, &src,
                                                  NULL);
      d12_test_transition(list, rt, D3D12_RESOURCE_STATE_COPY_SOURCE,
                          D3D12_RESOURCE_STATE_COMMON);
   }

   hr = ID3D12GraphicsCommandList_Close(list);
   CHECK(hr == S_OK, "the textured draw's command list closes cleanly");
   if (hr != S_OK)
      goto teardown;

   TEST_CALL(d12_test_submit_closed(dev, list, queue, fence, 1),
             "exact fenced upload/draw/copy");
   TEST_CALL(d12_test_map(readback, true, (void **)&map),
             "read the GPU copy only after the fence");

   {
      static bool ref[H][W];
      rasterise_reference(ref);
      unsigned ref_n = 0, n = 0, wrong = 0, colour_wrong = 0;
      unsigned quadrant[2][2] = {{0, 0}, {0, 0}};
      for (unsigned y = 0; y < H; y++) {
         for (unsigned x = 0; x < W; x++) {
            const uint8_t *px = map + y * fp.Footprint.RowPitch + x * 4;
            bool c = !(px[0] == 0 && px[1] == 0 && px[2] == 0);
            ref_n += ref[y][x];
            n += c;
            wrong += (c != ref[y][x]);
            if (!c)
               continue;
            /* Point sampling at uv = pixel centre / size: texel = floor(2 *
             * (x + 0.5) / W), which is the quadrant. */
            const unsigned tx = (2 * x) / W, ty = (2 * y) / H;
            const uint8_t *want = TEXELS[ty][tx];
            if (abs((int)px[0] - want[0]) > 1 || abs((int)px[1] - want[1]) > 1 ||
                abs((int)px[2] - want[2]) > 1)
               colour_wrong++;
            else
               quadrant[ty][tx]++;
         }
      }
      printf("       reference %u, drawn %u (of %u), coverage wrong %u, "
             "texel wrong %u, per quadrant %u/%u/%u/%u\n",
             ref_n, n, W * H, wrong, colour_wrong, quadrant[0][0],
             quadrant[0][1], quadrant[1][0], quadrant[1][1]);
      CHECK(wrong == 0,
            "the textured triangle matches the CPU rasterisation on all "
            "65536 pixels");
      CHECK(n > 0, "the textured triangle drew something at all");
      /* THE POINT OF THE SUITE. */
      CHECK(colour_wrong == 0,
            "every covered pixel carries the texel of its quadrant, sampled "
            "from t0 through the static sampler s0");
      /* The triangle covers the lower part of the target, so the blue and
       * yellow texels (row 1, both columns) must both appear; a sampler
       * that ignored v would have returned red/green and failed above. */
      CHECK(quadrant[1][0] && quadrant[1][1],
            "the covered area spans both columns of texel row 1 (blue and "
            "yellow), so the u axis was sampled across the quadrant edge");
   }

teardown:
   if (map) {
      const D3D12_RANGE empty = {0, 0};
      ID3D12Resource_Unmap(readback, 0, &empty);
   }
   ID3D12Resource_Release(readback);
   ID3D12Fence_Release(fence);
   ID3D12GraphicsCommandList_Release(list);
   ID3D12CommandAllocator_Release(alloc);
   ID3D12CommandQueue_Release(queue);
   ID3D12DescriptorHeap_Release(rtv_heap);
   ID3D12DescriptorHeap_Release(srv_heap);
   ID3D12Resource_Release(rt);
   ID3D12Resource_Release(tex_upload);
   ID3D12Resource_Release(tex);
   ID3D12Resource_Release(vb);
   ID3D12PipelineState_Release(pso);
   ID3D12RootSignature_Release(root_sig);
   ID3D12Device_Release(dev);
   printf(failures ? "G13 TEXTURE SAMPLE: %d FAILURES\n"
                   : "G13 TEXTURE SAMPLE: ALL PASS\n",
          failures);
   free(vs);
   free(ps);
   return failures ? 1 : 0;
}
