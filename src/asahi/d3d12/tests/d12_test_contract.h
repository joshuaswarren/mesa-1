/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 * Public-API fixture setup. These helpers never generate expected results. */
#pragma once

#define D12_TEST_UMA_HEAP { \
   .Type = D3D12_HEAP_TYPE_CUSTOM, \
   .CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK, \
   .MemoryPoolPreference = D3D12_MEMORY_POOL_L0, \
   .CreationNodeMask = 1, .VisibleNodeMask = 1 }

/* Object/call failures end the fixture before any dependent vtable access or
 * grading. The mandatory runner receives a failing process, never a skip. */
#define TEST_OBJECT(call, object, what) do { \
   HRESULT fixture_hr = (call); \
   CHECK(fixture_hr == S_OK && (object), what); \
   if (fixture_hr != S_OK || !(object)) { \
      printf("FIXTURE CONTRACT BLOCKED: %s HRESULT=0x%08x\n", what, (unsigned)fixture_hr); \
      return 1; \
   } \
} while (0)
#define TEST_CALL(call, what) TEST_OBJECT(call, 1, what)

static inline HRESULT
d12_test_map(ID3D12Resource *resource, bool reading, void **out)
{
   if (!resource || !out) return E_INVALIDARG;
   *out = NULL;
   const D3D12_RANGE empty = {0,0};
   HRESULT hr = ID3D12Resource_Map(resource, 0, reading ? NULL : &empty, out);
   return hr == S_OK && !*out ? E_FAIL : hr;
}

static inline void
d12_test_transition(ID3D12GraphicsCommandList *list, ID3D12Resource *resource,
                    D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after)
{
   const D3D12_RESOURCE_BARRIER b = {.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
      .Transition = {.pResource = resource,
         .Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
         .StateBefore = before,.StateAfter = after}};
   ID3D12GraphicsCommandList_ResourceBarrier(list, 1, &b);
}

/* Only call after Close returned exactly S_OK. */
static inline HRESULT
d12_test_submit_closed(ID3D12Device *dev, ID3D12GraphicsCommandList *list,
                       ID3D12CommandQueue *queue, ID3D12Fence *fence, UINT64 value)
{
   if (!dev || !list || !queue || !fence) return E_INVALIDARG;
   ID3D12CommandList *lists[] = {(ID3D12CommandList *)list};
   ID3D12CommandQueue_ExecuteCommandLists(queue, 1, lists);
   HRESULT hr = ID3D12Device_GetDeviceRemovedReason(dev);
   if (hr != S_OK) return hr;
   hr = ID3D12CommandQueue_Signal(queue, fence, value);
   if (hr != S_OK) return hr;
   hr = ID3D12Fence_SetEventOnCompletion(fence, value, NULL);
   if (hr != S_OK) return hr;
   UINT64 completed = ID3D12Fence_GetCompletedValue(fence);
   printf("       fence requested=%llu completed=%llu\n", (unsigned long long)value,
          (unsigned long long)completed);
   if (completed == UINT64_MAX || completed < value) return E_FAIL;
   return ID3D12Device_GetDeviceRemovedReason(dev);
}
