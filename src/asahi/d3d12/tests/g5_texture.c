/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * d3d12agx Texture2D.Load acceptance.
 *
 * Every other resource on this driver is reached by address: the root
 * table holds a pointer and the shader dereferences it. A texture is the
 * one thing that is not. AGX reaches a texture through a hardware
 * DESCRIPTOR -- a 24-byte packed structure the driver has to build itself,
 * out of a D3D12 descriptor that never becomes memory on real hardware.
 * So this test is about whether that descriptor is constructed correctly
 * and found correctly, not about filtering: Texture2D.Load takes integer
 * coordinates and no sampler, which removes every other variable.
 *
 * tex_probe.hlsl runs 256 threads over a 16x16 RGBA8 texture:
 *
 *    x = tid % 16, y = tid / 16
 *    c = tex.Load(int3(x, y, 0))
 *    o[tid] = r | (g << 8) | (b << 16) | (a << 24)
 *
 * The texture is filled so that no two texels share a value and no two
 * CHANNELS of one texel share a value:
 *
 *    r = x * 16      unique per column, and only per column
 *    g = y * 16      unique per row, and only per row
 *    b = x + y       a diagonal, so it agrees with neither r nor g
 *    a = 255 - x     descends where r ascends
 *
 * That fill is chosen so each way of being wrong prints as a different
 * number rather than as "not 0":
 *
 *   - A channel swizzle (BGRA read as RGBA, the classic descriptor bug)
 *     leaves the value in range and plausible, but r and b trade places.
 *   - Reading the wrong ROW shows up in g alone, and by exactly 16 per
 *     row, so the offset is readable off the value.
 *   - Reading the wrong COLUMN shows up in r and a moving in opposite
 *     directions, which no single-channel error can imitate.
 *
 * Because all 256 packed values are distinct, a wrong value can be looked
 * up to find WHICH texel the hardware actually read. That turns "wrong
 * pixels" into a coordinate delta, which is the difference between a bug
 * report and a shrug.
 *
 * Usage: g5_texture tex_probe.dxil
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

#define W 16
#define H 16
#define N (W * H)

/* Must mirror the driver's linear stride rule (d12_resource.c): rows are
 * 64-byte aligned, so the CPU cannot assume width*4. At 16 pixels wide
 * this happens to come out at exactly 64, but writing the rule rather
 * than the number is what keeps the test honest if the size changes.
 */
#define STRIDE ((((W * 4)) + 63) & ~63u)

/* Nothing the shader can produce looks like this: every channel of every
 * texel is a value the fill formula can generate, but 0xEE appears in no
 * channel of any texel (r and g are multiples of 16, b <= 30, a >= 240
 * and a is odd only where x is odd -- 0xEE = 238 is below the a range and
 * is not a multiple of 16). So a surviving 0xEEEEEEEE is unambiguous.
 */
#define POISON 0xEEEEEEEEu

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

/* The fill, in one place, so the CPU's expectation and the texture's
 * contents cannot drift apart.
 */
static void
texel(unsigned x, unsigned y, uint8_t out[4])
{
   out[0] = (uint8_t)(x * 16); /* r */
   out[1] = (uint8_t)(y * 16); /* g */
   out[2] = (uint8_t)(x + y);  /* b */
   out[3] = (uint8_t)(255 - x); /* a */
}

/* How the shader packs what it read. RGBA8_UNORM round-trips exactly
 * through uint(c * 255.0 + 0.5), so the expected value is the byte that
 * went in, with no rounding slack to allow for.
 */
static uint32_t
packed(const uint8_t rgba[4])
{
   return (uint32_t)rgba[0] | ((uint32_t)rgba[1] << 8) |
          ((uint32_t)rgba[2] << 16) | ((uint32_t)rgba[3] << 24);
}

int
main(int argc, char **argv)
{
   if (argc < 2) {
      fprintf(stderr, "usage: %s tex_probe.dxil\n", argv[0]);
      return 2;
   }

   size_t cs_size;
   void *cs = slurp(argv[1], &cs_size);
   if (!cs) {
      fprintf(stderr, "cannot read %s\n", argv[1]);
      return 2;
   }

   ID3D12Device *dev = NULL;
   HRESULT hr = D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0,
                                  &IID_ID3D12Device, (void **)&dev);
   TEST_OBJECT(hr, dev, "checked D3D12CreateDevice dev");
   CHECK(hr == S_OK && dev, "D3D12CreateDevice");
   if (hr != S_OK)
      return 1;

   /* ---- Root signature: a TABLE for t0, a root descriptor for u0 ----
    *
    * The mix is deliberate. The texture must come through a descriptor
    * table because that is the only path that carries a resource rather
    * than a bare address, and a hardware texture descriptor cannot be
    * built from an address alone. The output buffer comes through a root
    * UAV, which is the path already proven by G5 atomics -- so if the
    * output is poison, the failure is on the texture side, not in the
    * store.
    */
   D3D12_DESCRIPTOR_RANGE srv_range = {
      .RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
      .NumDescriptors = 1,
      .BaseShaderRegister = 0, /* t0 */
      .RegisterSpace = 0,
      .OffsetInDescriptorsFromTableStart =
         D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
   };

   D3D12_ROOT_PARAMETER params[2] = {
      {
         .ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
         .DescriptorTable = {.NumDescriptorRanges = 1,
                             .pDescriptorRanges = &srv_range},
         .ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL,
      },
      {
         .ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV,
         .Descriptor = {.ShaderRegister = 0}, /* u0 */
         .ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL,
      },
   };
   D3D12_ROOT_SIGNATURE_DESC rsd = {.NumParameters = 2, .pParameters = params};

   ID3DBlob *rs_blob = NULL;
   hr = D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1_0,
                                    &rs_blob, NULL);
   TEST_OBJECT(hr, rs_blob, "checked D3D12SerializeRootSignature rs_blob");
   CHECK(hr == S_OK && rs_blob, "serialize a root signature: SRV table + UAV");
   if (hr != S_OK)
      return 1;

   ID3D12RootSignature *root_sig = NULL;
   hr = ID3D12Device_CreateRootSignature(
      dev, 0, ID3D10Blob_GetBufferPointer(rs_blob),
      ID3D10Blob_GetBufferSize(rs_blob), &IID_ID3D12RootSignature,
      (void **)&root_sig);
   TEST_OBJECT(hr, root_sig, "checked ID3D12Device_CreateRootSignature root_sig");
   CHECK(hr == S_OK && root_sig, "parse it back (round trip)");
   ID3D10Blob_Release(rs_blob);
   if (hr != S_OK)
      return 1;

   /* ---- The texture ---- */

   D3D12_HEAP_PROPERTIES tex_heap = D12_TEST_UMA_HEAP;
   D3D12_RESOURCE_DESC td = {
      .Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
      .Width = W,
      .Height = H,
      .DepthOrArraySize = 1,
      .MipLevels = 1,
      .Format = DXGI_FORMAT_R8G8B8A8_UNORM,
      .SampleDesc = {.Count = 1},
      .Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN,
      .Flags = D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS,
      /* No ALLOW_RENDER_TARGET: this texture is only ever read, and a
       * sampled-only texture is the case that matters here.
       */
   };
   ID3D12Resource *tex = NULL;
   hr = ID3D12Device_CreateCommittedResource(dev, &tex_heap,
                                             D3D12_HEAP_FLAG_NONE, &td,
                                             D3D12_RESOURCE_STATE_COMMON, NULL,
                                             &IID_ID3D12Resource,
                                             (void **)&tex);
   TEST_OBJECT(hr, tex, "checked ID3D12Device_CreateCommittedResource tex");
   CHECK(hr == S_OK && tex, "create a 16x16 RGBA8 texture (sampled only)");
   if (hr != S_OK)
      return 1;

   uint8_t *tex_map = malloc(STRIDE * H);
   CHECK(tex_map != NULL, "CPU input storage for opaque texture");
   if (!tex_map) return 1;

   /* Fill through the 64-byte-aligned stride, not width*4. Indexing by
    * width would put row y at the wrong address for any width whose byte
    * length is not already aligned, and would then AGREE with a driver
    * that made the same mistake -- so the row index is the stride.
    */
   for (unsigned y = 0; y < H; y++) {
      for (unsigned x = 0; x < W; x++)
         texel(x, y, tex_map + y * STRIDE + x * 4);
   }

   TEST_CALL(ID3D12Resource_Map(tex, 0, NULL, NULL), "map opaque texture without exposing pointer");
   hr = ID3D12Resource_WriteToSubresource(tex, 0, NULL, tex_map, STRIDE, STRIDE * H);
   ID3D12Resource_Unmap(tex, 0, NULL);
   TEST_CALL(hr, "publish exact input texels through legal opaque CPU API");
   free(tex_map);tex_map = NULL;

   /* ---- The SRV, in a shader-visible heap ---- */

   D3D12_DESCRIPTOR_HEAP_DESC hd = {
      .Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
      .NumDescriptors = 1,
      .Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
   };
   ID3D12DescriptorHeap *heap = NULL;
   hr = ID3D12Device_CreateDescriptorHeap(dev, &hd, &IID_ID3D12DescriptorHeap,
                                          (void **)&heap);
   TEST_OBJECT(hr, heap, "checked ID3D12Device_CreateDescriptorHeap heap");
   CHECK(hr == S_OK && heap, "create a shader-visible CBV/SRV/UAV heap");
   if (hr != S_OK)
      return 1;

   D3D12_CPU_DESCRIPTOR_HANDLE cpu =
      d12_test_descriptor_heap_get_cpu_start(heap);
   D3D12_GPU_DESCRIPTOR_HANDLE gpu =
      d12_test_descriptor_heap_get_gpu_start(heap);
   CHECK(cpu.ptr && gpu.ptr, "heap start handles are usable");
   if (!cpu.ptr || !gpu.ptr) return 1;

   D3D12_SHADER_RESOURCE_VIEW_DESC srv = {
      .Format = DXGI_FORMAT_R8G8B8A8_UNORM,
      .ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D,
      .Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
      .Texture2D = {.MipLevels = 1},
   };
   ID3D12Device_CreateShaderResourceView(dev, tex, &srv, cpu);

   /* CreateShaderResourceView returns void, so the only way to know it
    * did anything is that the slot is no longer zero. A silently ignored
    * texture SRV would otherwise present later as a fault, at a point far
    * from the cause.
    */
   {
      bool wrote = false;
      const uint8_t *slot = (const uint8_t *)cpu.ptr;
      for (unsigned i = 0; i < 32; i++)
         wrote |= slot[i] != 0;
      CHECK(wrote, "the texture SRV was recorded, not silently dropped");
      if (!wrote) return 1;
   }

   /* ---- The output buffer ---- */

   D3D12_HEAP_PROPERTIES buf_heap = D12_TEST_UMA_HEAP;
   D3D12_RESOURCE_DESC bd = {
      .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
      .Width = sizeof(uint32_t) * N,
      .Height = 1,
      .DepthOrArraySize = 1,
      .MipLevels = 1,
      .SampleDesc = {.Count = 1},
      .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
      .Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
   };
   ID3D12Resource *out = NULL;
   hr = ID3D12Device_CreateCommittedResource(dev, &buf_heap,
                                             D3D12_HEAP_FLAG_NONE, &bd,
                                             D3D12_RESOURCE_STATE_COMMON, NULL,
                                             &IID_ID3D12Resource,
                                             (void **)&out);
   TEST_OBJECT(hr, out, "checked ID3D12Device_CreateCommittedResource out");
   CHECK(hr == S_OK && out, "create the 256-uint output buffer");
   if (hr != S_OK)
      return 1;

   uint32_t *out_map = NULL;
   TEST_CALL(d12_test_map(out, false, (void **)&out_map), "checked CPU initialization map out");

   /* Poison, so that "the dispatch never ran" and "the dispatch ran and
    * read the wrong texels" are different observations rather than both
    * appearing as zeros.
    */
   memset(out_map, 0xEE, sizeof(uint32_t) * N);

   /* ---- Pipeline ---- */

   D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {
      .pRootSignature = root_sig,
      .CS = {.pShaderBytecode = cs, .BytecodeLength = cs_size},
   };
   ID3D12PipelineState *pso = NULL;
   hr = ID3D12Device_CreateComputePipelineState(dev, &pd,
                                                &IID_ID3D12PipelineState,
                                                (void **)&pso);
   TEST_OBJECT(hr, pso, "checked ID3D12Device_CreateComputePipelineState pso");
   CHECK(hr == S_OK && pso, "compile a shader using Texture2D.Load");
   if (hr != S_OK)
      return 1;

   ID3D12CommandAllocator *alloc = NULL;
   ID3D12GraphicsCommandList *list = NULL;
   TEST_OBJECT(ID3D12Device_CreateCommandAllocator(dev, D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                       &IID_ID3D12CommandAllocator,
                                       (void **)&alloc), alloc, "checked ID3D12Device_CreateCommandAllocator alloc");
   TEST_OBJECT(ID3D12Device_CreateCommandList(dev, 0, D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                  alloc, NULL,
                                  &IID_ID3D12GraphicsCommandList,
                                  (void **)&list), list, "checked ID3D12Device_CreateCommandList list");

   ID3D12GraphicsCommandList_SetDescriptorHeaps(list, 1, &heap);
   ID3D12GraphicsCommandList_SetComputeRootSignature(list, root_sig);
   ID3D12GraphicsCommandList_SetComputeRootDescriptorTable(list, 0, gpu);
   ID3D12GraphicsCommandList_SetComputeRootUnorderedAccessView(
      list, 1, ID3D12Resource_GetGPUVirtualAddress(out));
   ID3D12GraphicsCommandList_SetPipelineState(list, pso);
   ID3D12Resource_Unmap(out, 0, NULL);out_map = NULL;
   d12_test_transition(list, out, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
   d12_test_transition(list, tex, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
   ID3D12GraphicsCommandList_Dispatch(list, N / 64, 1, 1);
   d12_test_transition(list, out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
   d12_test_transition(list, tex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
   hr = ID3D12GraphicsCommandList_Close(list);
   CHECK(hr == S_OK, "record the dispatch");
   if (hr != S_OK)
      return 1;

   ID3D12CommandQueue *queue = NULL;
   D3D12_COMMAND_QUEUE_DESC qd = {.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE};
   TEST_OBJECT(ID3D12Device_CreateCommandQueue(dev, &qd, &IID_ID3D12CommandQueue,
                                   (void **)&queue), queue, "checked ID3D12Device_CreateCommandQueue queue");
   ID3D12Fence *fence = NULL;
   TEST_OBJECT(ID3D12Device_CreateFence(dev, 0, D3D12_FENCE_FLAG_NONE, &IID_ID3D12Fence,
                            (void **)&fence), fence, "checked ID3D12Device_CreateFence fence");

   TEST_CALL(d12_test_submit_closed(dev, list, queue, fence, 1), "GPU dispatch complete");
   TEST_CALL(d12_test_map(out, true, (void **)&out_map), "map actual fenced out bytes");


   /* ---- Grade every texel against the CPU ---- */

   uint32_t want[N];
   for (unsigned y = 0; y < H; y++) {
      for (unsigned x = 0; x < W; x++) {
         uint8_t rgba[4];
         texel(x, y, rgba);
         want[y * W + x] = packed(rgba);
      }
   }

   unsigned wrong = 0, poisoned = 0;
   for (unsigned i = 0; i < N; i++) {
      if (out_map[i] == POISON)
         poisoned++;
      if (out_map[i] != want[i]) {
         if (wrong < 6) {
            printf("       (%u,%u) got 0x%08X want 0x%08X\n", i % W, i / W,
                   out_map[i], want[i]);
         }
         wrong++;
      }
   }

   if (getenv("D3D12AGX_DUMP_TEX")) {
      /* Which texel did each thread ACTUALLY read? Every packed value is
       * unique, so an observed value names its source coordinate.
       */
      for (unsigned i = 0; i < N; i++) {
         int sx = -1, sy = -1;
         for (unsigned y = 0; y < H && sx < 0; y++) {
            for (unsigned x = 0; x < W; x++) {
               uint8_t rgba[4];
               texel(x, y, rgba);
               uint32_t v = rgba[0] | (rgba[1] << 8) | (rgba[2] << 16) |
                            ((uint32_t)rgba[3] << 24);
               if (v == out_map[i]) { sx = x; sy = y; break; }
            }
         }
         printf("       want(%2u,%2u) got 0x%08X read (%d,%d)\n", i % W,
                i / W, out_map[i], sx, sy);
      }
   }

   CHECK(wrong == 0, "all 256 texels match Texture2D.Load on the CPU");

   if (wrong == 0) {
      printf("       spot check: (0,0)=0x%08X (15,0)=0x%08X (0,15)=0x%08X "
             "(15,15)=0x%08X\n",
             out_map[0], out_map[15], out_map[15 * W], out_map[N - 1]);
   } else {
      printf("       %u of %u texels wrong\n", wrong, N);

      /* ---- Say WHICH failure this is ----
       *
       * "All wrong" is nearly useless on its own; each of these shapes
       * points at a different piece of the driver, and they are cheap to
       * tell apart because the fill made every texel distinguishable.
       */

      /* (a) Nothing was written. The dispatch never reached the store --
       * a fault, a rejected pipeline, or a root UAV that never bound. The
       * texture path is then untested, not disproven.
       */
      if (poisoned == N) {
         printf("       DIAGNOSIS: nothing was written -- all 256 slots are "
                "still 0x%08X poison. The dispatch did not reach the store; "
                "check dmesg for a GPU fault.\n",
                POISON);
      } else if (poisoned) {
         printf("       DIAGNOSIS: %u of %u slots still hold poison, so the "
                "dispatch wrote only part of the buffer.\n",
                poisoned, N);
      }

      if (poisoned != N) {
         /* (b) Channel swizzle. The values are right but red and blue
          * traded places, which means the descriptor's swizzle fields (or
          * the format's channel order) are wrong -- the ADDRESSING found
          * the correct texel.
          */
         unsigned rb_swapped = 0;
         for (unsigned i = 0; i < N; i++) {
            uint8_t rgba[4];
            texel(i % W, i / W, rgba);
            uint8_t bgra[4] = {rgba[2], rgba[1], rgba[0], rgba[3]};
            if (out_map[i] == packed(bgra))
               rb_swapped++;
         }
         if (rb_swapped > N / 2) {
            printf("       DIAGNOSIS: %u of %u texels match with RED and BLUE "
                   "exchanged. Addressing is correct; the descriptor's "
                   "swizzle or channel order is wrong.\n",
                   rb_swapped, N);
         }

         /* (c) Wrong texel, right texture. Every packed value is unique,
          * so an observed value can be traced back to the texel it really
          * came from. A constant index delta across the whole image is a
          * stride or base-offset bug; a delta that is a whole multiple of
          * the width is specifically the row stride.
          */
         unsigned traced = 0, consistent = 0;
         int delta = 0;
         bool have_delta = false;
         for (unsigned i = 0; i < N; i++) {
            if (out_map[i] == want[i])
               continue;
            for (unsigned j = 0; j < N; j++) {
               if (out_map[i] != want[j])
                  continue;
               traced++;
               if (!have_delta) {
                  delta = (int)j - (int)i;
                  have_delta = true;
                  consistent = 1;
               } else if ((int)j - (int)i == delta) {
                  consistent++;
               }
               break;
            }
         }

         if (traced) {
            printf("       %u wrong values are still real texels of this "
                   "texture (right data, wrong address).\n",
                   traced);
         }
         if (have_delta && consistent == traced && traced > 1) {
            if (delta % (int)W == 0) {
               printf("       DIAGNOSIS: every traced texel came from %+d "
                      "ROWS away, a constant offset. That is a row-stride "
                      "bug: the descriptor's Stride does not match the "
                      "%u-byte aligned rows the resource was allocated "
                      "with.\n",
                      delta / (int)W, (unsigned)STRIDE);
            } else {
               printf("       DIAGNOSIS: every traced texel came from a "
                      "constant %+d texels away, so the base address or "
                      "coordinate origin is offset, not the stride.\n",
                      delta);
            }
         }
      }
   }

   /* A texel per corner, printed unconditionally: even a total failure
    * should leave a human able to see the shape of what came back.
    */
   printf("       raw: [0]=0x%08X [1]=0x%08X [16]=0x%08X [255]=0x%08X\n",
          out_map[0], out_map[1], out_map[16], out_map[255]);

   { const D3D12_RANGE empty = {0,0}; ID3D12Resource_Unmap(out, 0, &empty); }

   ID3D12Fence_Release(fence);
   ID3D12CommandQueue_Release(queue);
   ID3D12GraphicsCommandList_Release(list);
   ID3D12CommandAllocator_Release(alloc);
   ID3D12PipelineState_Release(pso);
   ID3D12Resource_Release(out);
   ID3D12DescriptorHeap_Release(heap);
   ID3D12Resource_Release(tex);
   ID3D12RootSignature_Release(root_sig);
   CHECK(ID3D12Device_Release(dev) == 0, "device refcount reaches 0");
   free(cs);

   printf(failures ? "\nG5 TEXTURE: %d FAILURES\n" : "\nG5 TEXTURE: ALL PASS\n",
          failures);
   return failures;
}
