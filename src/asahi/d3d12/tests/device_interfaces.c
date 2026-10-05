/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Live checks of the device interface ladder. Every ID3D12DeviceN IID must
 * resolve to one object whose vtable is fully populated, and the newer
 * creation entry points must behave like their original forms. Before this
 * test, QueryInterface(ID3D12Device15) returned a tear-off whose inherited
 * slots were NULL, so the first GetNodeCount through it was a segfault.
 */

#define COBJMACROS
#include <wsl/winadapter.h>
#include <directx/d3d12.h>

#include "d12_test_abi.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

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

int
main(void)
{
   ID3D12Device *dev = NULL;
   HRESULT hr = D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0,
                                  &IID_ID3D12Device, (void **)&dev);
   if (hr != S_OK || !dev) {
      printf("FAIL - D3D12CreateDevice: 0x%08x\n", (unsigned)hr);
      return 1;
   }

   CHECK(D3D12EnableExperimentalFeatures(0, NULL, NULL, NULL) == S_OK,
         "D3D12EnableExperimentalFeatures(0) succeeds");

   /* Every device IID resolves to the same pointer. */
   static const struct {
      const IID *iid;
      const char *name;
   } iids[] = {
      {&IID_ID3D12Device1, "ID3D12Device1"},
      {&IID_ID3D12Device2, "ID3D12Device2"},
      {&IID_ID3D12Device3, "ID3D12Device3"},
      {&IID_ID3D12Device4, "ID3D12Device4"},
      {&IID_ID3D12Device5, "ID3D12Device5"},
      {&IID_ID3D12Device6, "ID3D12Device6"},
      {&IID_ID3D12Device7, "ID3D12Device7"},
      {&IID_ID3D12Device8, "ID3D12Device8"},
      {&IID_ID3D12Device9, "ID3D12Device9"},
      {&IID_ID3D12Device10, "ID3D12Device10"},
      {&IID_ID3D12Device11, "ID3D12Device11"},
      {&IID_ID3D12Device12, "ID3D12Device12"},
      {&IID_ID3D12Device13, "ID3D12Device13"},
      {&IID_ID3D12Device14, "ID3D12Device14"},
      {&IID_ID3D12Device15, "ID3D12Device15"},
   };
   for (unsigned i = 0; i < sizeof(iids) / sizeof(iids[0]); i++) {
      void *p = NULL;
      hr = ID3D12Device_QueryInterface(dev, iids[i].iid, &p);
      char what[64];
      snprintf(what, sizeof(what), "QueryInterface %s is the same object",
               iids[i].name);
      CHECK(hr == S_OK && p == (void *)dev, what);
      if (p)
         IUnknown_Release((IUnknown *)p);
   }

   ID3D12Device15 *dev15 = NULL;
   hr = ID3D12Device_QueryInterface(dev, &IID_ID3D12Device15, (void **)&dev15);
   CHECK(hr == S_OK && dev15, "ID3D12Device15 available");
   if (!dev15)
      return 1;

   /* Every vtable slot must be non-NULL: a NULL slot is a crash. */
   {
      void **vtbl = (void **)dev15->lpVtbl;
      const unsigned slots = sizeof(ID3D12Device15Vtbl) / sizeof(void *);
      unsigned null_slots = 0;
      for (unsigned i = 0; i < slots; i++)
         null_slots += vtbl[i] == NULL;
      printf("       %u vtable slots, %u NULL\n", slots, null_slots);
      CHECK(null_slots == 0, "no NULL slot in the ID3D12Device15 vtable");
   }

   /* Inherited methods work through the newest interface. */
   CHECK(ID3D12Device15_GetNodeCount(dev15) == 1,
         "GetNodeCount through ID3D12Device15");

   /* Device4: CreateCommandList1 makes a closed list; Reset opens it. */
   ID3D12CommandAllocator *alloc = NULL;
   hr = ID3D12Device_CreateCommandAllocator(dev, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                            &IID_ID3D12CommandAllocator,
                                            (void **)&alloc);
   CHECK(hr == S_OK, "CreateCommandAllocator");
   ID3D12GraphicsCommandList *list = NULL;
   hr = ID3D12Device15_CreateCommandList1(dev15, 0,
                                          D3D12_COMMAND_LIST_TYPE_DIRECT,
                                          D3D12_COMMAND_LIST_FLAG_NONE,
                                          &IID_ID3D12GraphicsCommandList,
                                          (void **)&list);
   CHECK(hr == S_OK && list, "CreateCommandList1");
   if (list) {
      CHECK(ID3D12GraphicsCommandList_Close(list) != S_OK,
            "a CreateCommandList1 list starts closed");
      CHECK(ID3D12GraphicsCommandList_Reset(list, alloc, NULL) == S_OK,
            "Reset opens the closed list with an allocator");
      CHECK(ID3D12GraphicsCommandList_Close(list) == S_OK,
            "the reset list closes");
      CHECK(ID3D12CommandAllocator_Reset(alloc) == S_OK,
            "allocator resets after the list closed");
   }

   /* Device4/8/10: committed resources through the newer descriptors. */
   D3D12_HEAP_PROPERTIES upload = {.Type = D3D12_HEAP_TYPE_UPLOAD};
   D3D12_RESOURCE_DESC1 bd1 = {
      .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
      .Width = 4096,
      .Height = 1,
      .DepthOrArraySize = 1,
      .MipLevels = 1,
      .SampleDesc = {.Count = 1},
      .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
   };
   D3D12_RESOURCE_DESC bd = {
      .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
      .Width = 4096,
      .Height = 1,
      .DepthOrArraySize = 1,
      .MipLevels = 1,
      .SampleDesc = {.Count = 1},
      .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
   };
   ID3D12Resource *r1 = NULL, *r2 = NULL, *r3 = NULL;
   hr = ID3D12Device15_CreateCommittedResource1(
      dev15, &upload, D3D12_HEAP_FLAG_NONE, &bd,
      D3D12_RESOURCE_STATE_GENERIC_READ, NULL, NULL, &IID_ID3D12Resource,
      (void **)&r1);
   CHECK(hr == S_OK && r1, "CreateCommittedResource1 (no protected session)");
   hr = ID3D12Device15_CreateCommittedResource2(
      dev15, &upload, D3D12_HEAP_FLAG_NONE, &bd1,
      D3D12_RESOURCE_STATE_GENERIC_READ, NULL, NULL, &IID_ID3D12Resource,
      (void **)&r2);
   CHECK(hr == S_OK && r2, "CreateCommittedResource2 (DESC1)");
   hr = ID3D12Device15_CreateCommittedResource3(
      dev15, &upload, D3D12_HEAP_FLAG_NONE, &bd1,
      D3D12_BARRIER_LAYOUT_UNDEFINED, NULL, NULL, 0, NULL,
      &IID_ID3D12Resource, (void **)&r3);
   CHECK(hr == S_OK && r3, "CreateCommittedResource3 (barrier layout)");
   if (r3) {
      D3D12_RESOURCE_DESC got = d12_test_resource_get_desc(r3);
      CHECK(got.Width == 4096 && got.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER,
            "resource created through DESC1 reports the DESC prefix");
   }
   {
      ID3D12Resource *bad = (void *)(uintptr_t)1;
      ID3D12ProtectedResourceSession *fake = (void *)(uintptr_t)1;
      hr = ID3D12Device15_CreateCommittedResource1(
         dev15, &upload, D3D12_HEAP_FLAG_NONE, &bd,
         D3D12_RESOURCE_STATE_GENERIC_READ, NULL, fake, &IID_ID3D12Resource,
         (void **)&bad);
      CHECK(hr == E_INVALIDARG && bad == NULL,
            "a protected session is refused and the out pointer nulled");
   }

   /* SetName(NULL) must not crash. */
   if (r1) {
      hr = ID3D12Resource_SetName(r1, NULL);
      CHECK(FAILED(hr), "SetName(NULL) fails cleanly");
      static const uint16_t name[] = {'b', 'u', 'f', 0};
      hr = ID3D12Resource_SetName(r1, (LPCWSTR)name);
      CHECK(hr == S_OK, "SetName(UTF-16) succeeds");
   }

   /* GetResourceAllocationInfo1 breaks the aggregate down per resource. */
   {
      D3D12_RESOURCE_DESC descs[2] = {bd, bd};
      descs[1].Width = 1000;
      D3D12_RESOURCE_ALLOCATION_INFO1 per[2] = {0};
#ifdef _WIN32
      D3D12_RESOURCE_ALLOCATION_INFO agg;
      ID3D12Device15_GetResourceAllocationInfo1(dev15, &agg, 1, 2, descs, per);
#else
      D3D12_RESOURCE_ALLOCATION_INFO agg =
         ID3D12Device15_GetResourceAllocationInfo1(dev15, 1, 2, descs, per);
#endif
      printf("       aggregate %llu B align %llu; [0] off %llu size %llu, "
             "[1] off %llu size %llu\n",
             (unsigned long long)agg.SizeInBytes,
             (unsigned long long)agg.Alignment,
             (unsigned long long)per[0].Offset,
             (unsigned long long)per[0].SizeInBytes,
             (unsigned long long)per[1].Offset,
             (unsigned long long)per[1].SizeInBytes);
      CHECK(agg.SizeInBytes != UINT64_MAX && per[0].Offset == 0 &&
               per[1].Offset >= per[0].SizeInBytes &&
               agg.SizeInBytes >= per[1].Offset + per[1].SizeInBytes,
            "GetResourceAllocationInfo1 lays resources out in order");
   }

   /* Device9: CreateCommandQueue1 with a creator ID. */
   {
      D3D12_COMMAND_QUEUE_DESC qd = {.Type = D3D12_COMMAND_LIST_TYPE_DIRECT};
      ID3D12CommandQueue *q = NULL;
      hr = ID3D12Device15_CreateCommandQueue1(dev15, &qd, &IID_IUnknown,
                                              &IID_ID3D12CommandQueue,
                                              (void **)&q);
      CHECK(hr == S_OK && q, "CreateCommandQueue1");
      if (q)
         ID3D12CommandQueue_Release(q);
   }

   /* Device5/6: housekeeping calls that engines issue at start-up. */
   {
      UINT n = 7;
      CHECK(ID3D12Device15_EnumerateMetaCommands(dev15, &n, NULL) == S_OK && n == 0,
            "EnumerateMetaCommands reports none");
      BOOL more = TRUE;
      CHECK(ID3D12Device15_SetBackgroundProcessingMode(
               dev15, D3D12_BACKGROUND_PROCESSING_MODE_ALLOWED,
               D3D12_MEASUREMENTS_ACTION_KEEP_ALL, NULL, &more) == S_OK &&
               !more,
            "SetBackgroundProcessingMode succeeds");
      CHECK(ID3D12Device15_ShaderCacheControl(
               dev15, D3D12_SHADER_CACHE_KIND_FLAG_IMPLICIT_D3D_CACHE_FOR_DRIVER,
               D3D12_SHADER_CACHE_CONTROL_FLAG_DISABLE) == S_OK,
            "ShaderCacheControl succeeds");
      void *obj = (void *)(uintptr_t)1;
      hr = ID3D12Device15_CreateStateObject(dev15, NULL, &IID_IUnknown, &obj);
      CHECK(FAILED(hr) && obj == NULL,
            "an unimplemented Device5 method fails instead of crashing");
   }

   /* Device15: TryCreate*View validates and succeeds. */
   {
      D3D12_DESCRIPTOR_HEAP_DESC hd = {
         .Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, .NumDescriptors = 2};
      ID3D12DescriptorHeap *heap = NULL;
      hr = ID3D12Device_CreateDescriptorHeap(dev, &hd, &IID_ID3D12DescriptorHeap,
                                             (void **)&heap);
      CHECK(hr == S_OK, "CreateDescriptorHeap");
      if (heap && r1) {
         D3D12_CPU_DESCRIPTOR_HANDLE h =
            d12_test_descriptor_heap_get_cpu_start(heap);
         D3D12_CONSTANT_BUFFER_VIEW_DESC cbv = {
            .BufferLocation = ID3D12Resource_GetGPUVirtualAddress(r1),
            .SizeInBytes = 256};
         CHECK(ID3D12Device15_TryCreateConstantBufferView(dev15, &cbv, h) == S_OK,
               "TryCreateConstantBufferView");
         D3D12_CPU_DESCRIPTOR_HANDLE null_handle = {0};
         CHECK(ID3D12Device15_TryCreateConstantBufferView(dev15, &cbv,
                                                          null_handle) ==
                  E_INVALIDARG,
               "TryCreateConstantBufferView refuses a null handle");
         ID3D12DescriptorHeap_Release(heap);
      }
   }

   if (r1)
      ID3D12Resource_Release(r1);
   if (r2)
      ID3D12Resource_Release(r2);
   if (r3)
      ID3D12Resource_Release(r3);
   if (list)
      ID3D12GraphicsCommandList_Release(list);
   if (alloc)
      ID3D12CommandAllocator_Release(alloc);
   ID3D12Device15_Release(dev15);
   CHECK(ID3D12Device_Release(dev) == 0, "device references reach zero");

   printf("DEVICE INTERFACES: %s\n", failures ? "FAILURES" : "ALL PASS");
   return failures ? 1 : 0;
}
