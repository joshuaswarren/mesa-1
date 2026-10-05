/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Query heaps.
 *
 * Before this, CreateQueryHeap was one of the generated E_NOTIMPL stubs,
 * and every one of the eleven tests in vkd3d-proton's d3d12_query.c
 * crashed rather than failed -- because they RELEASE the heap whether or
 * not creation succeeded:
 *
 *     hr = ID3D12Device_CreateQueryHeap(device, &desc, &IID, &heap);
 *     ok(hr == S_OK, ...);
 *     ID3D12QueryHeap_Release(heap);        // unconditional
 *
 * So nulling the out-pointer, which is the right COM behaviour and fixed
 * a large class of crashes elsewhere, could not help here: NULL crashes
 * that Release exactly as garbage did. A test that uses a feature without
 * checking needs the feature or nothing.
 *
 * What a query heap IS on this hardware: storage for `Count` results,
 * plus the type that says how wide each one is. The heap itself is just
 * a buffer; the interesting part is Begin/End/Resolve, which live in
 * d12_cmd.c.
 */

#include "d12_private.h"
#include "d12_query_core.h"

_Static_assert((uint32_t)D3D12_QUERY_HEAP_TYPE_OCCLUSION ==
                  (uint32_t)D12_QUERY_CORE_HEAP_OCCLUSION,
               "public query-heap ABI drift");
_Static_assert((uint32_t)D3D12_QUERY_HEAP_TYPE_COPY_QUEUE_TIMESTAMP ==
                  (uint32_t)D12_QUERY_CORE_HEAP_COPY_QUEUE_TIMESTAMP,
               "public copy-timestamp ABI drift");

static struct d12_query_heap *
d12_query_heap_from_iface(ID3D12QueryHeap *iface)
{
   return (struct d12_query_heap *)iface;
}

static HRESULT STDMETHODCALLTYPE
d12_query_heap_QueryInterface(ID3D12QueryHeap *iface, REFIID riid, void **out)
{
   struct d12_query_heap *heap = d12_query_heap_from_iface(iface);

   if (!out)
      return E_POINTER;

   if (d12_guid_equal(riid, &IID_IUnknown) ||
       d12_guid_equal(riid, &IID_ID3D12Object) ||
       d12_guid_equal(riid, &IID_ID3D12DeviceChild) ||
       d12_guid_equal(riid, &IID_ID3D12Pageable) ||
       d12_guid_equal(riid, &IID_ID3D12QueryHeap)) {
      p_atomic_inc(&heap->refcount);
      *out = iface;
      return S_OK;
   }

   if (d12_private_data_query_notifier(&heap->private_data, riid, out))
      return S_OK;

   d12_log_unknown_riid("ID3D12QueryHeap::QueryInterface", riid);
   *out = NULL;
   return E_NOINTERFACE;
}

static ULONG STDMETHODCALLTYPE
d12_query_heap_AddRef(ID3D12QueryHeap *iface)
{
   return p_atomic_inc_return(&d12_query_heap_from_iface(iface)->refcount);
}

static ULONG STDMETHODCALLTYPE
d12_query_heap_Release(ID3D12QueryHeap *iface)
{
   struct d12_query_heap *heap = d12_query_heap_from_iface(iface);
   uint32_t refs = p_atomic_dec_return(&heap->refcount);

   if (refs == 0) {
      if (heap->timestamp_handle)
         heap->device->dev.ops.bo_unbind_object(&heap->device->dev,
                                               heap->timestamp_handle);
      if (heap->bo)
         agx_bo_unreference(&heap->device->dev, heap->bo);
      d12_private_data_store_destroy(&heap->private_data);
      d12_device_release(heap->device);
      free(heap);
   }

   return refs;
}

D12_DEFINE_PRIVATE_DATA_METHODS(d12_query_heap, ID3D12QueryHeap,
                                d12_query_heap_from_iface)

static HRESULT STDMETHODCALLTYPE
d12_query_heap_GetDevice(ID3D12QueryHeap *iface, REFIID riid, void **out)
{
   struct d12_query_heap *heap = d12_query_heap_from_iface(iface);
   return ID3D12Device_QueryInterface(&heap->device->iface, riid, out);
}

static const ID3D12QueryHeapVtbl d12_query_heap_vtbl = {
   /* ID3D12QueryHeap adds no methods of its own: these eight ARE the
    * whole interface, so there is no stub default to fall back on. */
   .QueryInterface = d12_query_heap_QueryInterface,
   .AddRef = d12_query_heap_AddRef,
   .Release = d12_query_heap_Release,
   .GetPrivateData = d12_query_heap_GetPrivateData,
   .SetPrivateData = d12_query_heap_SetPrivateData,
   .SetPrivateDataInterface = d12_query_heap_SetPrivateDataInterface,
   .SetName = d12_query_heap_SetName,
   .GetDevice = d12_query_heap_GetDevice,
};

/*
 * Bytes one result of this type occupies in the heap. D3D12's resolved
 * layouts are wider for the statistics types, but the heap only has to
 * hold what the GPU accumulates; ResolveQueryData expands it.
 */
unsigned
d12_query_result_size(D3D12_QUERY_HEAP_TYPE type)
{
   switch (type) {
   case D3D12_QUERY_HEAP_TYPE_OCCLUSION:
      return 8; /* a 64-bit sample count */
   case D3D12_QUERY_HEAP_TYPE_TIMESTAMP:
   case D3D12_QUERY_HEAP_TYPE_COPY_QUEUE_TIMESTAMP:
      return 8;
   case D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS:
      return sizeof(D3D12_QUERY_DATA_PIPELINE_STATISTICS);
   case D3D12_QUERY_HEAP_TYPE_SO_STATISTICS:
      return sizeof(D3D12_QUERY_DATA_SO_STATISTICS);
   default:
      return 0;
   }
}

HRESULT
d12_query_heap_create_flags(struct d12_device *device,
                            const D3D12_QUERY_HEAP_DESC *desc,
                            D3D12_QUERY_HEAP_FLAGS flags, REFIID riid,
                            void **out)
{
   if (!out)
      return E_POINTER;

   *out = NULL;

   if (!desc || !desc->Count ||
       (flags & ~D3D12_QUERY_HEAP_FLAG_CPU_RESOLVE))
      return E_INVALIDARG;

   unsigned stride = d12_query_result_size(desc->Type);
   if (!stride) {
      mesa_logw("CreateQueryHeap: type %u is not supported",
                (unsigned)desc->Type);
      return E_NOTIMPL;
   }

   struct d12_query_heap *heap = calloc(1, sizeof(*heap));
   if (!heap)
      return E_OUTOFMEMORY;

   heap->iface.lpVtbl = (ID3D12QueryHeapVtbl *)&d12_query_heap_vtbl;
   heap->refcount = 1;
   heap->desc = *desc;
   heap->flags = flags;
   heap->stride = stride;

   size_t result_size = (size_t)stride * desc->Count;
   const bool timestamp = d12_query_core_is_timestamp_heap(desc->Type);
   /* Firmware timestamp offsets are 32-bit, including for copy queues. */
   if (timestamp && (uint64_t)result_size > (uint64_t)UINT32_MAX + 1) {
      free(heap);
      return E_OUTOFMEMORY;
   }
   size_t allocation_size = result_size;
   if (desc->Type == D3D12_QUERY_HEAP_TYPE_OCCLUSION) {
      /* Final results stay contiguous at the front for ResolveQueryData;
       * BeginQuery snapshots occupy the equally sized second plane.
       */
      heap->start_offset = result_size;
      allocation_size += result_size;
   }

   /* Zeroed, so a query that is resolved without ever being written
    * reads as "nothing happened" rather than as stale memory. */
   enum agx_bo_flags bo_flags = AGX_BO_WRITEBACK;
   /* Like Honeykrisp, firmware-written timestamp storage must be shared. */
   if (timestamp)
      bo_flags |= AGX_BO_SHARED;
   heap->bo = agx_bo_create(&device->dev, allocation_size, 0, bo_flags,
                            "d3d12 query heap");
   if (!heap->bo) {
      free(heap);
      return E_OUTOFMEMORY;
   }

   /* Lock2 can refuse an allocation dxgkrnl just handed us -- memset through
    * the NULL result would fault inside the application. */
   void *mapping = agx_bo_map(heap->bo);
   if (!mapping) {
      agx_bo_unreference(&device->dev, heap->bo);
      free(heap);
      return E_OUTOFMEMORY;
   }
   memset(mapping, 0, allocation_size);

   if (timestamp &&
       agx_bind_timestamps(&device->dev, heap->bo, &heap->timestamp_handle)) {
      agx_bo_unreference(&device->dev, heap->bo);
      free(heap);
      return E_OUTOFMEMORY;
   }

   heap->device = device;
   d12_private_data_store_init(&heap->private_data, (IUnknown *)&heap->iface);
   d12_device_add_ref(device);

   HRESULT hr = d12_query_heap_QueryInterface(&heap->iface, riid, out);
   d12_query_heap_Release(&heap->iface);
   return hr;
}

HRESULT
d12_query_heap_create(struct d12_device *device,
                      const D3D12_QUERY_HEAP_DESC *desc, REFIID riid,
                      void **out)
{
   return d12_query_heap_create_flags(device, desc,
                                      D3D12_QUERY_HEAP_FLAG_NONE,
                                      riid, out);
}
