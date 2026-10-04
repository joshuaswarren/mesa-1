/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * d3d12agx G2 acceptance test: committed buffer resources, command
 * allocator/list, CopyBufferRegion, ExecuteCommandLists through a real
 * drm_asahi_submit, fence-gated readback.
 *
 * GPU-transit proof: the CPU writes a pattern ONLY into the upload buffer.
 * The pattern must arrive in the readback buffer having chained through a
 * default buffer via two GPU copy dispatches, at a deliberately odd size
 * (4097 bytes) and offsets, with canaries proving no overwrite beyond the
 * requested ranges.
 */

#define COBJMACROS
#include <wsl/winadapter.h>
#include <directx/d3d12.h>

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

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

#define COPY_BYTES  4097 /* deliberately not a workgroup multiple */
#define DST_OFFSET  32
#define SRC_OFFSET  16
#define BUF_SIZE    (COPY_BYTES + 128)

static const D3D12_HEAP_PROPERTIES heap_upload = {
   .Type = D3D12_HEAP_TYPE_UPLOAD,
};
static const D3D12_HEAP_PROPERTIES heap_default = {
   .Type = D3D12_HEAP_TYPE_DEFAULT,
};
static const D3D12_HEAP_PROPERTIES heap_readback = {
   .Type = D3D12_HEAP_TYPE_READBACK,
};

static HRESULT
make_buffer(ID3D12Device *dev, const D3D12_HEAP_PROPERTIES *heap,
            UINT64 size, ID3D12Resource **out)
{
   D3D12_RESOURCE_DESC desc = {
      .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
      .Width = size,
      .Height = 1,
      .DepthOrArraySize = 1,
      .MipLevels = 1,
      .SampleDesc = {.Count = 1},
      .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
   };

   return ID3D12Device_CreateCommittedResource(
      dev, heap, D3D12_HEAP_FLAG_NONE, &desc,
      D3D12_RESOURCE_STATE_COMMON, NULL, &IID_ID3D12Resource, (void **)out);
}

int
main(void)
{
   ID3D12Device *dev = NULL;
   ID3D12Resource *upload = NULL, *scratch = NULL, *readback = NULL;
   ID3D12CommandAllocator *alloc = NULL;
   ID3D12GraphicsCommandList *list = NULL;
   ID3D12CommandQueue *queue = NULL;
   ID3D12Fence *fence = NULL;
   uint8_t *up_map = NULL, *rb_map = NULL;
   const char *blocked_stage = NULL;
   HRESULT blocked_hr = S_OK;
   HRESULT hr;

   hr = D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0, &IID_ID3D12Device,
                          (void **)&dev);
   CHECK(hr == S_OK && dev, "D3D12CreateDevice");
   if (FAILED(hr) || !dev)
      return 1;

   hr = make_buffer(dev, &heap_upload, BUF_SIZE, &upload);
   if (SUCCEEDED(hr) && upload)
      hr = make_buffer(dev, &heap_default, BUF_SIZE, &scratch);
   if (SUCCEEDED(hr) && scratch)
      hr = make_buffer(dev, &heap_readback, BUF_SIZE, &readback);
   CHECK(hr == S_OK && upload && scratch && readback,
         "create upload/default/readback buffers");
   if (FAILED(hr) || !upload || !scratch || !readback) {
      blocked_stage = "CreateCommittedResource";
      blocked_hr = FAILED(hr) ? hr : E_FAIL;
      goto cleanup;
   }

   CHECK(ID3D12Resource_GetGPUVirtualAddress(upload) != 0,
         "buffers have GPU virtual addresses");

   /* CPU writes the pattern ONLY here. Poison the other two so stale
    * zeroes can't fake a pass.
    */
   hr = ID3D12Resource_Map(upload, 0, NULL, (void **)&up_map);
   CHECK(hr == S_OK && up_map, "Map upload buffer");
   if (FAILED(hr) || !up_map) {
      blocked_stage = "Map upload";
      blocked_hr = FAILED(hr) ? hr : E_FAIL;
      goto cleanup;
   }

   for (unsigned i = 0; i < BUF_SIZE; i++)
      up_map[i] = (uint8_t)((i * 7 + 13) ^ (i >> 8));

   hr = ID3D12Resource_Map(readback, 0, NULL, (void **)&rb_map);
   CHECK(hr == S_OK && rb_map, "Map readback buffer");
   if (FAILED(hr) || !rb_map) {
      blocked_stage = "Map readback";
      blocked_hr = FAILED(hr) ? hr : E_FAIL;
      goto cleanup;
   }
   memset(rb_map, 0xAA, BUF_SIZE);

   /* Command machinery */
   hr = ID3D12Device_CreateCommandAllocator(
      dev, D3D12_COMMAND_LIST_TYPE_DIRECT, &IID_ID3D12CommandAllocator,
      (void **)&alloc);
   CHECK(hr == S_OK && alloc, "CreateCommandAllocator");
   if (FAILED(hr) || !alloc) {
      blocked_stage = "CreateCommandAllocator";
      blocked_hr = FAILED(hr) ? hr : E_FAIL;
      goto cleanup;
   }

   hr = ID3D12Device_CreateCommandList(dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                       alloc, NULL,
                                       &IID_ID3D12GraphicsCommandList,
                                       (void **)&list);
   CHECK(hr == S_OK && list, "CreateCommandList (recording state)");
   if (FAILED(hr) || !list) {
      blocked_stage = "CreateCommandList";
      blocked_hr = FAILED(hr) ? hr : E_FAIL;
      goto cleanup;
   }

   /* upload[16..16+4097) -> scratch[32..), then scratch -> readback[0..) */
   ID3D12GraphicsCommandList_CopyBufferRegion(list, scratch, DST_OFFSET,
                                              upload, SRC_OFFSET, COPY_BYTES);

   D3D12_RESOURCE_BARRIER barrier = {
      .Type = D3D12_RESOURCE_BARRIER_TYPE_UAV,
      .UAV = {.pResource = scratch},
   };
   ID3D12GraphicsCommandList_ResourceBarrier(list, 1, &barrier);

   ID3D12GraphicsCommandList_CopyBufferRegion(list, readback, 0, scratch,
                                              DST_OFFSET, COPY_BYTES);

   hr = ID3D12GraphicsCommandList_Close(list);
   CHECK(hr == S_OK, "Close command list");
   if (FAILED(hr)) {
      blocked_stage = "Close";
      blocked_hr = FAILED(hr) ? hr : E_FAIL;
      goto cleanup;
   }

   /* Negative control: nothing executed yet, fence must still be 0. */
   D3D12_COMMAND_QUEUE_DESC qdesc = {.Type = D3D12_COMMAND_LIST_TYPE_DIRECT};
   hr = ID3D12Device_CreateCommandQueue(dev, &qdesc, &IID_ID3D12CommandQueue,
                                        (void **)&queue);
   CHECK(hr == S_OK && queue, "CreateCommandQueue");
   if (FAILED(hr) || !queue) {
      blocked_stage = "CreateCommandQueue";
      blocked_hr = FAILED(hr) ? hr : E_FAIL;
      goto cleanup;
   }

   hr = ID3D12Device_CreateFence(dev, 0, D3D12_FENCE_FLAG_NONE,
                                 &IID_ID3D12Fence, (void **)&fence);
   CHECK(hr == S_OK && fence && ID3D12Fence_GetCompletedValue(fence) == 0,
         "fence starts at 0 before execution");
   if (FAILED(hr) || !fence) {
      blocked_stage = "CreateFence";
      blocked_hr = FAILED(hr) ? hr : E_FAIL;
      goto cleanup;
   }

   /* Execute on the real GPU */
   ID3D12CommandList *lists[] = {(ID3D12CommandList *)list};
   ID3D12CommandQueue_ExecuteCommandLists(queue, 1, lists);

   hr = ID3D12CommandQueue_Signal(queue, fence, 1);
   CHECK(hr == S_OK, "queue Signal(fence, 1) after execution");
   if (FAILED(hr)) {
      blocked_stage = "Signal";
      blocked_hr = FAILED(hr) ? hr : E_FAIL;
      goto cleanup;
   }

   hr = ID3D12Fence_SetEventOnCompletion(fence, 1, NULL);
   CHECK(hr == S_OK && ID3D12Fence_GetCompletedValue(fence) >= 1,
         "GPU work complete (fence reached 1)");
   if (FAILED(hr) || ID3D12Fence_GetCompletedValue(fence) < 1) {
      blocked_stage = "Wait fence1";
      blocked_hr = FAILED(hr) ? hr : E_FAIL;
      goto cleanup;
   }

   /* The verdict: byte-exact pattern transit through two GPU copies */
   int mismatch = -1;
   for (unsigned i = 0; i < COPY_BYTES; i++) {
      if (rb_map[i] != up_map[SRC_OFFSET + i]) {
         mismatch = (int)i;
         break;
      }
   }
   CHECK(mismatch < 0, "readback matches upload pattern byte-exact (4097B)");
   if (mismatch >= 0) {
      printf("       first mismatch at %d: got 0x%02x want 0x%02x\n",
             mismatch, rb_map[mismatch], up_map[SRC_OFFSET + mismatch]);
   }

   CHECK(rb_map[COPY_BYTES] == 0xAA,
         "canary after copied range untouched (no overrun)");

   /* Second negative control: an independent poisoned region stays put */
   CHECK(rb_map[BUF_SIZE - 1] == 0xAA, "end-of-buffer canary untouched");

   /* Reset + reuse the list: copy a different slice to overwrite the tail */
   hr = ID3D12GraphicsCommandList_Reset(list, alloc, NULL);
   CHECK(hr == S_OK, "Reset command list for reuse");
   if (FAILED(hr)) {
      blocked_stage = "Reset";
      blocked_hr = FAILED(hr) ? hr : E_FAIL;
      goto cleanup;
   }

   ID3D12GraphicsCommandList_CopyBufferRegion(list, readback, 0, upload, 0,
                                              64);
   hr = ID3D12GraphicsCommandList_Close(list);
   CHECK(hr == S_OK, "Close reused list");
   if (FAILED(hr)) {
      blocked_stage = "Close reused";
      blocked_hr = FAILED(hr) ? hr : E_FAIL;
      goto cleanup;
   }

   ID3D12CommandQueue_ExecuteCommandLists(queue, 1, lists);
   hr = ID3D12CommandQueue_Signal(queue, fence, 2);
   CHECK(hr == S_OK, "queue Signal(fence, 2) after reused execution");
   if (FAILED(hr)) {
      blocked_stage = "Signal reused";
      blocked_hr = hr;
      goto cleanup;
   }
   hr = ID3D12Fence_SetEventOnCompletion(fence, 2, NULL);
   CHECK(hr == S_OK && memcmp(rb_map, up_map, 64) == 0,
         "reused list's copy landed (first 64B now from offset 0)");
   if (FAILED(hr) || ID3D12Fence_GetCompletedValue(fence) < 2) {
      blocked_stage = "Wait fence2";
      blocked_hr = FAILED(hr) ? hr : E_FAIL;
      goto cleanup;
   }

cleanup:
   if (up_map && upload)
      ID3D12Resource_Unmap(upload, 0, NULL);
   if (rb_map && readback)
      ID3D12Resource_Unmap(readback, 0, NULL);
   if (fence)
      CHECK(ID3D12Fence_Release(fence) == 0, "fence released");
   if (queue)
      CHECK(ID3D12CommandQueue_Release(queue) == 0, "queue released");
   if (list)
      ID3D12GraphicsCommandList_Release(list);
   if (alloc)
      ID3D12CommandAllocator_Release(alloc);
   if (upload)
      ID3D12Resource_Release(upload);
   if (scratch)
      ID3D12Resource_Release(scratch);
   if (readback)
      ID3D12Resource_Release(readback);
   CHECK(ID3D12Device_Release(dev) == 0, "device refcount reaches 0");
   if (blocked_stage)
      printf("G2 SMOKE: BLOCKED at %s (%#x); not a hardware pass\n",
             blocked_stage, (unsigned)blocked_hr);

   printf(failures ? "G2 SMOKE: %d FAILURES\n" : "G2 SMOKE: ALL PASS\n",
          failures);
   return failures;
}
