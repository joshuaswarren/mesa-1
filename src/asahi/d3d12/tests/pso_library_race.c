/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * PSO lifetime under a pipeline library, exercised from several threads.
 *
 * A pipeline library keeps a stored PSO alive after the application has
 * released every public reference, and hands the same object back from
 * Load*. The driver used to track that with two independent counts and a
 * plain bool for the device reference; an application releasing its last
 * PSO reference while another thread released the library could free the
 * object twice or touch it after the free. Ownership is now one atomic
 * count, and this test hammers both transitions:
 *
 *   1. load/release storm: public count oscillates 0 <-> N across threads
 *      while the library holds the object;
 *   2. last-reference race: the application's PSO reference and the
 *      library's last reference are dropped concurrently, many times.
 *
 * The device's own reference count going back to zero at the end proves
 * every PSO was destroyed exactly once. Needs a device (GPU).
 */

#define COBJMACROS
#include <wsl/winadapter.h>
#include <directx/d3d12.h>

#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STORM_THREADS 8
#define STORM_ITERS   2000
#define RACE_ROUNDS   200

static unsigned failures;

#define CHECK(cond, what)                                                    \
   do {                                                                      \
      if (cond) {                                                            \
         printf("ok   - %s\n", what);                                        \
      } else {                                                               \
         printf("FAIL - %s\n", what);                                        \
         failures++;                                                         \
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

/* res_probe's bindings: SRV t2, UAV u3, CBV b1 in one table. */
static ID3D12RootSignature *
make_table_root_sig(ID3D12Device *dev)
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
   D3D12_ROOT_SIGNATURE_DESC rsd = {.NumParameters = 1, .pParameters = &table};
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

/* The wire format of LPCWSTR on every platform the driver runs on. */
static const uint16_t pso_name[] = {'c', 's', 0};
#define PSO_NAME ((LPCWSTR)(const void *)pso_name)

struct storm {
   ID3D12PipelineLibrary *lib;
   const D3D12_COMPUTE_PIPELINE_STATE_DESC *desc;
   ID3D12PipelineState *expect;
   unsigned mismatches;
   unsigned load_failures;
};

static void *
storm_thread(void *arg)
{
   struct storm *s = arg;
   for (unsigned i = 0; i < STORM_ITERS; i++) {
      ID3D12PipelineState *p = NULL;
      HRESULT hr = ID3D12PipelineLibrary_LoadComputePipeline(
         s->lib, PSO_NAME, s->desc, &IID_ID3D12PipelineState, (void **)&p);
      if (hr != S_OK || !p) {
         __atomic_fetch_add(&s->load_failures, 1, __ATOMIC_RELAXED);
         continue;
      }
      if (p != s->expect)
         __atomic_fetch_add(&s->mismatches, 1, __ATOMIC_RELAXED);
      ID3D12PipelineState_AddRef(p);
      ID3D12PipelineState_Release(p);
      ID3D12PipelineState_Release(p);
   }
   return NULL;
}

struct race {
   ID3D12PipelineState *pso;
   ID3D12PipelineLibrary *lib;
   pthread_barrier_t *go;
};

static void *
race_release_pso(void *arg)
{
   struct race *r = arg;
   pthread_barrier_wait(r->go);
   ID3D12PipelineState_Release(r->pso);
   return NULL;
}

static void *
race_release_lib(void *arg)
{
   struct race *r = arg;
   pthread_barrier_wait(r->go);
   ID3D12PipelineLibrary_Release(r->lib);
   return NULL;
}

int
main(int argc, char **argv)
{
   if (argc < 2) {
      fprintf(stderr, "usage: %s res_probe.dxil\n", argv[0]);
      return 2;
   }

   size_t cs_size;
   void *cs = slurp(argv[1], &cs_size);
   if (!cs) {
      printf("FAIL - read %s\n", argv[1]);
      return 1;
   }

   ID3D12Device *dev = NULL;
   HRESULT hr = D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0,
                                  &IID_ID3D12Device, (void **)&dev);
   if (hr != S_OK || !dev) {
      printf("FAIL - D3D12CreateDevice: 0x%08x\n", (unsigned)hr);
      return 1;
   }

   ID3D12Device1 *dev1 = NULL;
   hr = ID3D12Device_QueryInterface(dev, &IID_ID3D12Device1, (void **)&dev1);
   CHECK(hr == S_OK && dev1, "ID3D12Device1 for CreatePipelineLibrary");
   if (!dev1)
      return 1;

   ID3D12RootSignature *rs = make_table_root_sig(dev);
   CHECK(rs != NULL, "descriptor-table root signature");
   if (!rs)
      return 1;

   D3D12_COMPUTE_PIPELINE_STATE_DESC cd = {
      .pRootSignature = rs,
      .CS = {.pShaderBytecode = cs, .BytecodeLength = cs_size},
   };

   /* --- 1. load/release storm ---------------------------------------- */
   ID3D12PipelineState *pso = NULL;
   hr = ID3D12Device_CreateComputePipelineState(
      dev, &cd, &IID_ID3D12PipelineState, (void **)&pso);
   CHECK(hr == S_OK && pso, "CreateComputePipelineState");
   if (!pso)
      return 1;

   ID3D12PipelineLibrary *lib = NULL;
   hr = ID3D12Device1_CreatePipelineLibrary(dev1, NULL, 0,
                                            &IID_ID3D12PipelineLibrary,
                                            (void **)&lib);
   CHECK(hr == S_OK && lib, "CreatePipelineLibrary");
   if (!lib)
      return 1;
   CHECK(ID3D12PipelineLibrary_StorePipeline(lib, PSO_NAME, pso) == S_OK,
         "StorePipeline");

   /* The application lets go; the library alone keeps the PSO. */
   CHECK(ID3D12PipelineState_Release(pso) == 0,
         "public count reaches zero while the library holds the PSO");

   struct storm storm = {.lib = lib, .desc = &cd, .expect = pso};
   pthread_t threads[STORM_THREADS];
   for (unsigned t = 0; t < STORM_THREADS; t++)
      pthread_create(&threads[t], NULL, storm_thread, &storm);
   storm_thread(&storm);
   for (unsigned t = 0; t < STORM_THREADS; t++)
      pthread_join(threads[t], NULL);

   CHECK(storm.load_failures == 0, "every Load from the library succeeded");
   CHECK(storm.mismatches == 0,
         "every Load returned the stored object");

   ID3D12PipelineState *again = NULL;
   hr = ID3D12PipelineLibrary_LoadComputePipeline(
      lib, PSO_NAME, &cd, &IID_ID3D12PipelineState, (void **)&again);
   CHECK(hr == S_OK && again == pso,
         "the stored PSO is still returned after the storm");
   if (again)
      CHECK(ID3D12PipelineState_Release(again) == 0,
            "and its public count returns to zero");
   CHECK(ID3D12PipelineLibrary_Release(lib) == 0,
         "library release destroys the stored PSO");

   /* --- 2. last-reference race --------------------------------------- */
   unsigned rounds_ok = 0;
   for (unsigned round = 0; round < RACE_ROUNDS; round++) {
      ID3D12PipelineState *p = NULL;
      hr = ID3D12Device_CreateComputePipelineState(
         dev, &cd, &IID_ID3D12PipelineState, (void **)&p);
      if (hr != S_OK || !p)
         break;

      ID3D12PipelineLibrary *l = NULL;
      hr = ID3D12Device1_CreatePipelineLibrary(dev1, NULL, 0,
                                               &IID_ID3D12PipelineLibrary,
                                               (void **)&l);
      if (hr != S_OK || !l ||
          ID3D12PipelineLibrary_StorePipeline(l, PSO_NAME, p) != S_OK)
         break;

      pthread_barrier_t go;
      pthread_barrier_init(&go, NULL, 2);
      struct race r = {.pso = p, .lib = l, .go = &go};
      pthread_t a, b;
      pthread_create(&a, NULL, race_release_pso, &r);
      pthread_create(&b, NULL, race_release_lib, &r);
      pthread_join(a, NULL);
      pthread_join(b, NULL);
      pthread_barrier_destroy(&go);
      rounds_ok++;
   }
   printf("       %u concurrent last-reference rounds\n", rounds_ok);
   CHECK(rounds_ok == RACE_ROUNDS, "PSO and library released concurrently");

   /* --- every PSO destroyed exactly once ----------------------------- */
   ID3D12RootSignature_Release(rs);
   ID3D12Device1_Release(dev1);
   ULONG left = ID3D12Device_Release(dev);
   printf("       device references left: %lu\n", (unsigned long)left);
   CHECK(left == 0, "device references reach zero");

   free(cs);
   printf(failures ? "PSO LIBRARY RACE: %u FAILURES\n"
                   : "PSO LIBRARY RACE: ALL PASS\n", failures);
   return failures ? 1 : 0;
}
