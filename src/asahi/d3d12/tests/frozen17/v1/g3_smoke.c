/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * d3d12agx G3 acceptance test: RTV descriptor heaps and
 * ClearRenderTargetView as a real tiler render pass (background clear +
 * end-of-tile store over PROCESS_EMPTY_TILES), pixel-exact through legal
 * GPU CopyTextureRegion into READBACK buffers.
 *
 * Two sizes: 256x256 (tile-aligned) and 123x77 (deliberately unaligned in
 * both dimensions). Row-padding bytes between width*4 and the 256-byte-
 * aligned readback pitch must keep their poison, and footprint prefix/suffix
 * canaries must remain intact. The opaque target exposes no linear pointer.
 */

#define COBJMACROS
#include <wsl/winadapter.h>
#include <directx/d3d12.h>

#include "d12_test_abi.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
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

/* Original row-padding oracle is retained against the API readback pitch. */
#define STRIDE_B(w) ((((w) * 4) + 63) & ~63u)

static HRESULT
make_rt_texture(ID3D12Device *dev, UINT w, UINT h, ID3D12Resource **out)
{
   D3D12_HEAP_PROPERTIES heap = {
      .Type = D3D12_HEAP_TYPE_CUSTOM,.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
      .MemoryPoolPreference = D3D12_MEMORY_POOL_L0,.CreationNodeMask = 1,.VisibleNodeMask = 1};
   D3D12_RESOURCE_DESC desc = {
      .Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
      .Width = w,
      .Height = h,
      .DepthOrArraySize = 1,
      .MipLevels = 1,
      .Format = DXGI_FORMAT_R8G8B8A8_UNORM,
      .SampleDesc = {.Count = 1},
      .Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN,
      .Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET |
               D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS,
   };

   return ID3D12Device_CreateCommittedResource(
      dev, &heap, D3D12_HEAP_FLAG_NONE, &desc,
      D3D12_RESOURCE_STATE_COMMON, NULL, &IID_ID3D12Resource,
      (void **)out);
}

/* Expect every pixel of the w*h image to equal rgba; padding keeps poison */
static int
check_image(const uint8_t *map, UINT w, UINT h, const uint8_t rgba[4],
            int check_padding, UINT stride)
{

   for (UINT y = 0; y < h; y++) {
      const uint8_t *row = map + y * stride;

      for (UINT x = 0; x < w; x++) {
         if (memcmp(row + x * 4, rgba, 4)) {
            printf("       pixel (%u,%u): got %02x%02x%02x%02x want "
                   "%02x%02x%02x%02x\n",
                   x, y, row[x * 4], row[x * 4 + 1], row[x * 4 + 2],
                   row[x * 4 + 3], rgba[0], rgba[1], rgba[2], rgba[3]);
            return 0;
         }
      }

      if (check_padding) {
         for (UINT b = w * 4; b < stride; b++) {
            if (row[b] != 0x55) {
               printf("       row %u padding byte %u overwritten: %02x\n", y,
                      b, row[b]);
               return 0;
            }
         }
      }
   }

   return 1;
}

/* Keep the original controlled poison in real texture storage, without
 * exposing opaque tiled bytes as a linear application pointer. */
static HRESULT
poison_texture(ID3D12Resource *texture, UINT w, UINT h)
{
   UINT pitch = STRIDE_B(w);size_t size = (size_t)pitch * h;
   uint8_t *bytes = malloc(size);
   if (!bytes) return E_OUTOFMEMORY;
   memset(bytes, 0x55, size);
   HRESULT hr = ID3D12Resource_Map(texture, 0, NULL, NULL);
   if (hr == S_OK) {
      hr = ID3D12Resource_WriteToSubresource(texture, 0, NULL, bytes, pitch, (UINT)size);
      ID3D12Resource_Unmap(texture, 0, NULL);
   }
   free(bytes);return hr;
}

static void
transition(ID3D12GraphicsCommandList *list, ID3D12Resource *resource,
           D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after)
{
   D3D12_RESOURCE_BARRIER b = {.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
      .Transition = {.pResource = resource,
                     .Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                     .StateBefore = before, .StateAfter = after}};
   ID3D12GraphicsCommandList_ResourceBarrier(list, 1, &b);
}

static HRESULT
submit_and_wait(ID3D12Device *dev, ID3D12GraphicsCommandList *list,
                ID3D12CommandQueue *queue, ID3D12Fence *fence, UINT64 value)
{
   HRESULT hr = ID3D12GraphicsCommandList_Close(list);
   if (hr != S_OK) return hr;
   ID3D12CommandList *lists[] = {(ID3D12CommandList *)list};
   ID3D12CommandQueue_ExecuteCommandLists(queue, 1, lists);
   hr = ID3D12Device_GetDeviceRemovedReason(dev);
   if (hr != S_OK) return hr;
   hr = ID3D12CommandQueue_Signal(queue, fence, value);
   if (hr != S_OK) return hr;
   hr = ID3D12Fence_SetEventOnCompletion(fence, value, NULL);
   if (hr != S_OK) return hr;
   UINT64 done = ID3D12Fence_GetCompletedValue(fence);
   printf("       fence requested=%llu completed=%llu\n",
          (unsigned long long)value, (unsigned long long)done);
   if (done == UINT64_MAX || done < value) return E_FAIL;
   return ID3D12Device_GetDeviceRemovedReason(dev);
}

static HRESULT
make_readback(ID3D12Device *dev, ID3D12Resource *texture,
              ID3D12Resource **buffer, D3D12_PLACED_SUBRESOURCE_FOOTPRINT *fp,
              UINT64 *size)
{
   D3D12_RESOURCE_DESC td = d12_test_resource_get_desc(texture);
   UINT rows = 0;UINT64 row_bytes = 0,total = 0;
   ID3D12Device_GetCopyableFootprints(dev, &td, 0, 1, 512, fp, &rows, &row_bytes, &total);
   if (fp->Offset != 512 || rows != td.Height || row_bytes != td.Width * 4 ||
       fp->Footprint.Width != td.Width || fp->Footprint.Height != td.Height ||
       fp->Footprint.Depth != 1 || fp->Footprint.Format != DXGI_FORMAT_R8G8B8A8_UNORM ||
       fp->Footprint.RowPitch < row_bytes || fp->Footprint.RowPitch % 256)
      return E_FAIL;
   *size = fp->Offset + (UINT64)fp->Footprint.RowPitch * rows + 512;
   const D3D12_HEAP_PROPERTIES hp = {.Type = D3D12_HEAP_TYPE_READBACK};
   const D3D12_RESOURCE_DESC bd = {.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
      .Width = *size,.Height = 1,.DepthOrArraySize = 1,.MipLevels = 1,
      .SampleDesc = {.Count = 1},.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR};
   return ID3D12Device_CreateCommittedResource(dev, &hp, D3D12_HEAP_FLAG_NONE,
      &bd, D3D12_RESOURCE_STATE_COPY_DEST, NULL, &IID_ID3D12Resource, (void **)buffer);
}

static void
copy_texture(ID3D12GraphicsCommandList *list, ID3D12Resource *texture,
              ID3D12Resource *buffer, D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp)
{
   const D3D12_TEXTURE_COPY_LOCATION src = {.pResource = texture,
      .Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,.SubresourceIndex = 0};
   const D3D12_TEXTURE_COPY_LOCATION dst = {.pResource = buffer,
      .Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,.PlacedFootprint = fp};
   transition(list, texture, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
   ID3D12GraphicsCommandList_CopyTextureRegion(list, &dst, 0, 0, 0, &src, NULL);
   transition(list, texture, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
}

static int
check_canaries(const uint8_t *map, D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp, UINT64 size)
{
   for (UINT64 i = 0; i < fp.Offset; i++) if (map[i] != 0x55) return 0;
   for (UINT64 i = fp.Offset + (UINT64)fp.Footprint.RowPitch * fp.Footprint.Height; i < size; i++)
      if (map[i] != 0x55) return 0;
   return 1;
}

int
main(int argc, char **argv)
{
   ID3D12Device *dev = NULL;
   ID3D12Resource *tex_a = NULL,*tex_b = NULL,*rb_a = NULL,*rb_b = NULL;
   ID3D12Resource *buf_src = NULL,*buf_dst = NULL;
   ID3D12DescriptorHeap *rtv_heap = NULL;
   ID3D12CommandAllocator *alloc = NULL;
   ID3D12GraphicsCommandList *list = NULL;
   ID3D12CommandQueue *queue = NULL;ID3D12Fence *fence = NULL;
   uint8_t *map_a = NULL,*map_b = NULL,*src_map = NULL,*dst_map = NULL;
   D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp_a = {0},fp_b = {0};
   UINT64 size_a = 0,size_b = 0;
   HRESULT hr = S_OK;const char *blocked = NULL;
   const D3D12_RANGE empty = {0,0};
#define REQUIRE(cond,what) do {CHECK(cond,what);if (!(cond)) {blocked=what;goto cleanup;}} while(0)
#define CALL(call,what) do {hr=(call);REQUIRE(hr==S_OK,what);} while(0)
   unsigned bw = 123,bh = 77;
   if (argc >= 3) {
      char *end_w = NULL,*end_h = NULL;
      unsigned long w = strtoul(argv[1], &end_w, 10),h = strtoul(argv[2], &end_h, 10);
      REQUIRE(end_w && !*end_w && end_h && !*end_h && w && h && w <= 16384 && h <= 16384,
              "valid overridden render dimensions");
      bw = w;bh = h;printf("# unaligned target overridden to %ux%u\n",bw,bh);
   }
   CALL(D3D12CreateDevice(NULL,D3D_FEATURE_LEVEL_11_0,&IID_ID3D12Device,(void **)&dev),"D3D12CreateDevice");
   REQUIRE(dev,"device object");
   const D3D12_DESCRIPTOR_HEAP_DESC hd = {.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV,.NumDescriptors = 2};
   CALL(ID3D12Device_CreateDescriptorHeap(dev,&hd,&IID_ID3D12DescriptorHeap,(void **)&rtv_heap),"CreateDescriptorHeap(RTV, 2)");
   REQUIRE(rtv_heap,"RTV heap object");
   UINT inc = ID3D12Device_GetDescriptorHandleIncrementSize(dev,D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
   D3D12_CPU_DESCRIPTOR_HANDLE rtv_a = d12_test_descriptor_heap_get_cpu_start(rtv_heap);
   D3D12_CPU_DESCRIPTOR_HANDLE rtv_b = {.ptr = rtv_a.ptr + inc};
   REQUIRE(inc >= sizeof(uint64_t) && rtv_a.ptr && rtv_b.ptr != rtv_a.ptr,
           "heap start, and an opaque non-zero handle increment");
   CALL(make_rt_texture(dev,256,256,&tex_a),"create 256x256 RGBA8 render target");
   CALL(make_rt_texture(dev,bw,bh,&tex_b),"create 123x77 RGBA8 render target");
   REQUIRE(tex_a && tex_b,"render-target objects");
   CALL(poison_texture(tex_a,256,256),"publish original 0x55 texture poison for aligned target");
   CALL(poison_texture(tex_b,bw,bh),"publish original 0x55 texture poison for unaligned target");
   ID3D12Device_CreateRenderTargetView(dev,tex_a,NULL,rtv_a);
   ID3D12Device_CreateRenderTargetView(dev,tex_b,NULL,rtv_b);
   CALL(make_readback(dev,tex_a,&rb_a,&fp_a,&size_a),"legal readback footprint for aligned target");
   CALL(make_readback(dev,tex_b,&rb_b,&fp_b,&size_b),"legal readback footprint for unaligned target");
   REQUIRE(rb_a && rb_b,"image readback objects");
   CALL(ID3D12Resource_Map(rb_a,0,&empty,(void **)&map_a),"poison aligned image readback");REQUIRE(map_a,"aligned readback pointer");
   CALL(ID3D12Resource_Map(rb_b,0,&empty,(void **)&map_b),"poison unaligned image readback");REQUIRE(map_b,"unaligned readback pointer");
   memset(map_a,0x55,(size_t)size_a);memset(map_b,0x55,(size_t)size_b);
   ID3D12Resource_Unmap(rb_a,0,NULL);map_a = NULL;ID3D12Resource_Unmap(rb_b,0,NULL);map_b = NULL;
   const D3D12_HEAP_PROPERTIES upload = {.Type = D3D12_HEAP_TYPE_UPLOAD},readback = {.Type = D3D12_HEAP_TYPE_READBACK};
   const D3D12_RESOURCE_DESC bd = {.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,.Width = 256,
      .Height = 1,.DepthOrArraySize = 1,.MipLevels = 1,.SampleDesc = {.Count = 1},.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR};
   CALL(ID3D12Device_CreateCommittedResource(dev,&upload,D3D12_HEAP_FLAG_NONE,&bd,D3D12_RESOURCE_STATE_GENERIC_READ,NULL,&IID_ID3D12Resource,(void **)&buf_src),"create legal UPLOAD source");
   CALL(ID3D12Device_CreateCommittedResource(dev,&readback,D3D12_HEAP_FLAG_NONE,&bd,D3D12_RESOURCE_STATE_COPY_DEST,NULL,&IID_ID3D12Resource,(void **)&buf_dst),"create legal READBACK destination");
   REQUIRE(buf_src && buf_dst,"side-buffer objects");
   CALL(ID3D12Resource_Map(buf_src,0,&empty,(void **)&src_map),"map side source for writing");REQUIRE(src_map,"side source pointer");
   CALL(ID3D12Resource_Map(buf_dst,0,&empty,(void **)&dst_map),"map side destination poison");REQUIRE(dst_map,"side destination pointer");
   for (unsigned i = 0; i < 256; i++) src_map[i] = (uint8_t)(i ^ 0xC3);
   memset(dst_map,0,256);ID3D12Resource_Unmap(buf_src,0,NULL);src_map = NULL;ID3D12Resource_Unmap(buf_dst,0,NULL);dst_map = NULL;
   CALL(ID3D12Device_CreateCommandAllocator(dev,D3D12_COMMAND_LIST_TYPE_DIRECT,&IID_ID3D12CommandAllocator,(void **)&alloc),"CreateCommandAllocator");REQUIRE(alloc,"allocator object");
   CALL(ID3D12Device_CreateCommandList(dev,0,D3D12_COMMAND_LIST_TYPE_DIRECT,alloc,NULL,&IID_ID3D12GraphicsCommandList,(void **)&list),"CreateCommandList");REQUIRE(list,"list object");
   const D3D12_COMMAND_QUEUE_DESC qd = {.Type = D3D12_COMMAND_LIST_TYPE_DIRECT};
   CALL(ID3D12Device_CreateCommandQueue(dev,&qd,&IID_ID3D12CommandQueue,(void **)&queue),"CreateCommandQueue");REQUIRE(queue,"queue object");
   CALL(ID3D12Device_CreateFence(dev,0,D3D12_FENCE_FLAG_NONE,&IID_ID3D12Fence,(void **)&fence),"CreateFence");REQUIRE(fence,"fence object");
   const FLOAT color_a[4] = {1.0f,0.0f,128.0f / 255.0f,1.0f},color_b[4] = {0.0f,1.0f,0.0f,1.0f};
   const uint8_t bytes_a[4] = {255,0,128,255},bytes_b[4] = {0,255,0,255};
   transition(list,tex_a,D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_RENDER_TARGET);
   ID3D12GraphicsCommandList_ClearRenderTargetView(list,rtv_a,color_a,0,NULL);
   transition(list,tex_b,D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_RENDER_TARGET);
   ID3D12GraphicsCommandList_ClearRenderTargetView(list,rtv_b,color_b,0,NULL);
   ID3D12GraphicsCommandList_CopyBufferRegion(list,buf_dst,0,buf_src,0,256);
   copy_texture(list,tex_a,rb_a,fp_a);copy_texture(list,tex_b,rb_b,fp_b);
   CALL(submit_and_wait(dev,list,queue,fence,1),"GPU render + copy complete (fence reached 1)");
   const D3D12_RANGE range_a = {0,(SIZE_T)size_a},range_b = {0,(SIZE_T)size_b},side_range = {0,256};
   CALL(ID3D12Resource_Map(rb_a,0,&range_a,(void **)&map_a),"map actual aligned image after fence1");REQUIRE(map_a,"aligned image pointer");
   CALL(ID3D12Resource_Map(rb_b,0,&range_b,(void **)&map_b),"map actual unaligned image after fence1");REQUIRE(map_b,"unaligned image pointer");
   CALL(ID3D12Resource_Map(buf_src,0,&side_range,(void **)&src_map),"map side source after fence1");REQUIRE(src_map,"side source pointer");
   CALL(ID3D12Resource_Map(buf_dst,0,&side_range,(void **)&dst_map),"map actual side copy after fence1");REQUIRE(dst_map,"side copy pointer");
   CHECK(check_image(map_a + fp_a.Offset,256,256,bytes_a,0,fp_a.Footprint.RowPitch),"256x256 cleared pixel-exact to (255,0,128,255)");
   CHECK(check_image(map_b + fp_b.Offset,bw,bh,bytes_b,1,fp_b.Footprint.RowPitch),"unaligned target cleared pixel-exact, row padding untouched");
   CHECK(check_canaries(map_a,fp_a,size_a) && check_canaries(map_b,fp_b,size_b),"both GPU image-copy prefix/suffix canaries intact");
   CHECK(memcmp(dst_map,src_map,256) == 0,"compute segment after renders landed (256B copy)");
   ID3D12Resource_Unmap(rb_a,0,&empty);map_a = NULL;ID3D12Resource_Unmap(rb_b,0,&empty);map_b = NULL;
   ID3D12Resource_Unmap(buf_src,0,&empty);src_map = NULL;ID3D12Resource_Unmap(buf_dst,0,&empty);dst_map = NULL;
   CALL(ID3D12CommandAllocator_Reset(alloc),"Reset fenced allocator");
   CALL(ID3D12GraphicsCommandList_Reset(list,alloc,NULL),"Reset for re-clear");
   transition(list,tex_a,D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_RENDER_TARGET);
   ID3D12GraphicsCommandList_ClearRenderTargetView(list,rtv_a,color_b,0,NULL);
   copy_texture(list,tex_a,rb_a,fp_a);
   CALL(submit_and_wait(dev,list,queue,fence,2),"exact re-clear fence2");
   CALL(ID3D12Resource_Map(rb_a,0,&range_a,(void **)&map_a),"map actual reused clear after fence2");REQUIRE(map_a,"re-clear image pointer");
   CHECK(check_image(map_a + fp_a.Offset,256,256,bytes_b,0,fp_a.Footprint.RowPitch),"re-clear via reused list lands (256x256 now green)");
   CHECK(check_canaries(map_a,fp_a,size_a),"reused clear GPU-copy canaries intact");
cleanup:
   if (blocked) printf("G3 SMOKE BLOCKED: stage=%s HRESULT=0x%08x\n",blocked,(unsigned)hr);
   if (map_a) ID3D12Resource_Unmap(rb_a,0,&empty);
   if (map_b) ID3D12Resource_Unmap(rb_b,0,&empty);
   if (src_map) ID3D12Resource_Unmap(buf_src,0,&empty);
   if (dst_map) ID3D12Resource_Unmap(buf_dst,0,&empty);
   if (fence) ID3D12Fence_Release(fence);
   if (queue) ID3D12CommandQueue_Release(queue);
   if (list) ID3D12GraphicsCommandList_Release(list);
   if (alloc) ID3D12CommandAllocator_Release(alloc);
   if (buf_src) ID3D12Resource_Release(buf_src);
   if (buf_dst) ID3D12Resource_Release(buf_dst);
   if (rb_a) ID3D12Resource_Release(rb_a);
   if (rb_b) ID3D12Resource_Release(rb_b);
   if (tex_a) ID3D12Resource_Release(tex_a);
   if (tex_b) ID3D12Resource_Release(tex_b);
   if (rtv_heap) ID3D12DescriptorHeap_Release(rtv_heap);
   if (dev) CHECK(ID3D12Device_Release(dev) == 0,"device refcount reaches 0");
   printf(failures ? "G3 SMOKE: %d FAILURES\n" : "G3 SMOKE: ALL PASS\n",failures);
   return failures || blocked;
}
