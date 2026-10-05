/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Concurrent pipeline-state creation. Games create PSOs from many threads
 * at once, mostly over the same handful of shaders with different
 * fixed-function state; the device shader cache has to serve those lookups,
 * insertions, evictions and shared uploads under contention without a
 * failure or a leak. Eight threads create and release graphics and compute
 * PSOs in four fixed-function variants (so the prolog/epilog cache, the
 * stage cache and the fast-link cache all see hits and misses racing),
 * then every variant is created once more on the main thread and checked.
 *
 * Usage: d3d12agx_pso_threads tri.dxil solid.dxil res_probe.dxil
 */

#define COBJMACROS
#include <wsl/winadapter.h>
#include <directx/d3d12.h>

#include "d12_test_abi.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define THREADS 8
#define ITERATIONS 150
#define VARIANTS 4

static unsigned failures;

#define REQUIRE(cond, what)                                                  \
   do {                                                                      \
      if (!(cond)) {                                                         \
         fprintf(stderr, "FAIL: %s\n", what);                                \
         failures++;                                                         \
         return 1;                                                           \
      }                                                                      \
   } while (0)

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

static ID3D12RootSignature *
make_root_sig(ID3D12Device *dev, const D3D12_ROOT_PARAMETER *params,
              UINT count)
{
   D3D12_ROOT_SIGNATURE_DESC rsd = {.NumParameters = count,
                                    .pParameters = params};
   ID3DBlob *blob = NULL;
   if (D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1_0,
                                   &blob, NULL) != S_OK)
      return NULL;
   ID3D12RootSignature *rs = NULL;
   HRESULT hr = ID3D12Device_CreateRootSignature(
      dev, 0, ID3D10Blob_GetBufferPointer(blob),
      ID3D10Blob_GetBufferSize(blob), &IID_ID3D12RootSignature, (void **)&rs);
   ID3D10Blob_Release(blob);
   return hr == S_OK ? rs : NULL;
}

struct shared {
   ID3D12Device *dev;
   ID3D12RootSignature *rs_empty, *rs_table;
   void *vs, *ps, *cs;
   size_t vs_size, ps_size, cs_size;
   atomic_uint errors;
   atomic_uint created;
};

/* Variant 0 is the plain opaque PSO; 1 blends (fragment epilog); 2 masks
 * colour writes (epilog, different key); 3 changes the API sample mask
 * (different fragment compile key). */
static void
graphics_desc(const struct shared *s, unsigned variant,
              D3D12_GRAPHICS_PIPELINE_STATE_DESC *pd)
{
   memset(pd, 0, sizeof(*pd));
   pd->pRootSignature = s->rs_empty;
   pd->VS = (D3D12_SHADER_BYTECODE){s->vs, s->vs_size};
   pd->PS = (D3D12_SHADER_BYTECODE){s->ps, s->ps_size};
   pd->NumRenderTargets = 1;
   pd->RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
   pd->SampleDesc.Count = 1;
   pd->SampleMask = ~0u;
   pd->RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
   pd->RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
   pd->RasterizerState.DepthClipEnable = TRUE;
   pd->PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
   D3D12_RENDER_TARGET_BLEND_DESC *rt = &pd->BlendState.RenderTarget[0];
   rt->RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
   rt->SrcBlend = D3D12_BLEND_ONE;
   rt->DestBlend = D3D12_BLEND_ZERO;
   rt->BlendOp = D3D12_BLEND_OP_ADD;
   rt->SrcBlendAlpha = D3D12_BLEND_ONE;
   rt->DestBlendAlpha = D3D12_BLEND_ZERO;
   rt->BlendOpAlpha = D3D12_BLEND_OP_ADD;
   rt->LogicOp = D3D12_LOGIC_OP_NOOP;

   switch (variant % VARIANTS) {
   case 1:
      rt->BlendEnable = TRUE;
      rt->SrcBlend = D3D12_BLEND_SRC_ALPHA;
      rt->DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
      break;
   case 2:
      rt->RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_RED |
                                  D3D12_COLOR_WRITE_ENABLE_GREEN;
      break;
   case 3:
      pd->SampleMask = 0x1;
      break;
   default:
      break;
   }
}

static void *
worker(void *arg)
{
   struct shared *s = arg;
   for (unsigned i = 0; i < ITERATIONS; i++) {
      D3D12_GRAPHICS_PIPELINE_STATE_DESC pd;
      graphics_desc(s, i, &pd);
      ID3D12PipelineState *pso = NULL;
      HRESULT hr = ID3D12Device_CreateGraphicsPipelineState(
         s->dev, &pd, &IID_ID3D12PipelineState, (void **)&pso);
      if (hr != S_OK || !pso) {
         atomic_fetch_add(&s->errors, 1);
      } else {
         atomic_fetch_add(&s->created, 1);
         ID3D12PipelineState_Release(pso);
      }

      D3D12_COMPUTE_PIPELINE_STATE_DESC cd = {
         .pRootSignature = s->rs_table,
         .CS = {s->cs, s->cs_size},
      };
      pso = NULL;
      hr = ID3D12Device_CreateComputePipelineState(
         s->dev, &cd, &IID_ID3D12PipelineState, (void **)&pso);
      if (hr != S_OK || !pso) {
         atomic_fetch_add(&s->errors, 1);
      } else {
         atomic_fetch_add(&s->created, 1);
         ID3D12PipelineState_Release(pso);
      }
   }
   return NULL;
}

int
main(int argc, char **argv)
{
   if (argc < 4) {
      fprintf(stderr, "usage: %s tri.dxil solid.dxil res_probe.dxil\n",
              argv[0]);
      return 2;
   }

   struct shared s = {0};
   s.vs = slurp(argv[1], &s.vs_size);
   s.ps = slurp(argv[2], &s.ps_size);
   s.cs = slurp(argv[3], &s.cs_size);
   REQUIRE(s.vs && s.ps && s.cs, "read shader bytecode");

   HRESULT hr = D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0,
                                  &IID_ID3D12Device, (void **)&s.dev);
   REQUIRE(hr == S_OK && s.dev, "D3D12CreateDevice");

   s.rs_empty = make_root_sig(s.dev, NULL, 0);
   REQUIRE(s.rs_empty, "empty root signature");
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
   s.rs_table = make_root_sig(s.dev, &table, 1);
   REQUIRE(s.rs_table, "descriptor-table root signature");

   pthread_t threads[THREADS];
   for (unsigned t = 0; t < THREADS; t++)
      REQUIRE(!pthread_create(&threads[t], NULL, worker, &s),
              "pthread_create");
   for (unsigned t = 0; t < THREADS; t++)
      pthread_join(threads[t], NULL);

   printf("pso threads: %u threads x %u iterations, %u PSOs created, %u "
          "errors\n", THREADS, ITERATIONS, atomic_load(&s.created),
          atomic_load(&s.errors));
   REQUIRE(atomic_load(&s.errors) == 0, "concurrent PSO creation");
   REQUIRE(atomic_load(&s.created) == 2 * THREADS * ITERATIONS,
           "every creation accounted for");

   /* Every variant once more, held together, after the races: the cached
    * entries must still produce working pipelines. */
   ID3D12PipelineState *held[VARIANTS + 1] = {0};
   for (unsigned v = 0; v < VARIANTS; v++) {
      D3D12_GRAPHICS_PIPELINE_STATE_DESC pd;
      graphics_desc(&s, v, &pd);
      hr = ID3D12Device_CreateGraphicsPipelineState(
         s.dev, &pd, &IID_ID3D12PipelineState, (void **)&held[v]);
      REQUIRE(hr == S_OK && held[v], "graphics variant after the races");
   }
   D3D12_COMPUTE_PIPELINE_STATE_DESC cd = {
      .pRootSignature = s.rs_table,
      .CS = {s.cs, s.cs_size},
   };
   hr = ID3D12Device_CreateComputePipelineState(
      s.dev, &cd, &IID_ID3D12PipelineState, (void **)&held[VARIANTS]);
   REQUIRE(hr == S_OK && held[VARIANTS], "compute after the races");
   for (unsigned v = 0; v <= VARIANTS; v++)
      ID3D12PipelineState_Release(held[v]);

   ID3D12RootSignature_Release(s.rs_empty);
   ID3D12RootSignature_Release(s.rs_table);
   ID3D12Device_Release(s.dev);
   free(s.vs);
   free(s.ps);
   free(s.cs);

   printf("PSO THREADS: %s\n", failures ? "FAILURES" : "ALL PASS");
   return failures ? 1 : 0;
}
