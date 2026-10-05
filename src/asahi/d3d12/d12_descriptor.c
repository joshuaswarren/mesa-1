/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * ID3D12DescriptorHeap. Slot storage whose handles are pointers into the slot
 * array, with the slot size equal to the advertised 40-byte handle increment.
 * Shader-visible heaps use coherent GPU BOs: finite ranges are snapshotted
 * into root bindings, while unbounded ranges dereference the slots directly.
 */

#include "d12_private.h"

static struct d12_descriptor_heap *
d12_heap_from_iface(ID3D12DescriptorHeap *iface)
{
   return (struct d12_descriptor_heap *)iface;
}

static HRESULT STDMETHODCALLTYPE
d12_heap_QueryInterface(ID3D12DescriptorHeap *iface, REFIID riid, void **out)
{
   struct d12_descriptor_heap *heap = d12_heap_from_iface(iface);

   if (!out)
      return E_POINTER;

   if (d12_guid_equal(riid, &IID_IUnknown) ||
       d12_guid_equal(riid, &IID_ID3D12Object) ||
       d12_guid_equal(riid, &IID_ID3D12DeviceChild) ||
       d12_guid_equal(riid, &IID_ID3D12Pageable) ||
       d12_guid_equal(riid, &IID_ID3D12DescriptorHeap)) {
      p_atomic_inc(&heap->refcount);
      *out = iface;
      return S_OK;
   }

   if (d12_private_data_query_notifier(&heap->private_data, riid, out))
      return S_OK;

   d12_log_unknown_riid("ID3D12DescriptorHeap::QueryInterface", riid);
   *out = NULL;
   return E_NOINTERFACE;
}

static ULONG STDMETHODCALLTYPE
d12_heap_AddRef(ID3D12DescriptorHeap *iface)
{
   struct d12_descriptor_heap *heap = d12_heap_from_iface(iface);
   return p_atomic_inc_return(&heap->refcount);
}

/*
 * Heap registry.
 *
 * A CPU descriptor handle is a bare pointer into a heap's slot array, so a
 * descriptor write has to find the owning heap to mirror the slot into the
 * heap's native tables. Keep every heap's slot range in a table sorted by
 * start address: a write resolves its heap with one binary search, and a
 * seqlock lets it do so without taking the device lock that heap creation
 * and release use. Tables that growth replaced are retired rather than
 * freed because a lock-free search may still be reading one; the retired
 * set is geometric in the largest table, so it stays small.
 */
static bool
d12_descriptor_registry_add(struct d12_device *device,
                            struct d12_descriptor_heap *heap)
{
   struct d12_descriptor_registry *reg = &device->descriptor_registry;
   const struct d12_descriptor_heap_range range = {
      .start = (uintptr_t)heap->slots,
      .end = (uintptr_t)heap->slots +
             (size_t)heap->desc.NumDescriptors * sizeof(*heap->slots),
      .heap = heap,
   };

   simple_mtx_lock(&device->descriptor_heap_lock);

   struct d12_descriptor_heap_range *table = reg->ranges;
   if (reg->count == reg->capacity) {
      const uint32_t capacity = MAX2(reg->capacity * 2, 16);
      table = malloc((size_t)capacity * sizeof(*table));
      struct d12_descriptor_heap_range **retired = NULL;
      if (table && reg->ranges) {
         retired = realloc(reg->retired, (reg->retired_count + 1) *
                                            sizeof(*reg->retired));
         if (!retired) {
            free(table);
            table = NULL;
         }
      }
      if (!table) {
         simple_mtx_unlock(&device->descriptor_heap_lock);
         return false;
      }
      if (reg->count)
         memcpy(table, reg->ranges, (size_t)reg->count * sizeof(*table));
      if (reg->ranges) {
         reg->retired = retired;
         reg->retired[reg->retired_count++] = reg->ranges;
      }
      reg->capacity = capacity;
   }

   uint32_t pos = 0;
   while (pos < reg->count && table[pos].start < range.start)
      pos++;

   /* Writers publish an odd sequence while the table is inconsistent. */
   p_atomic_set(&reg->seq, reg->seq + 1);
   __sync_synchronize();
   if (table != reg->ranges)
      p_atomic_set(&reg->ranges, table);
   memmove(&table[pos + 1], &table[pos],
           (size_t)(reg->count - pos) * sizeof(*table));
   table[pos] = range;
   p_atomic_set(&reg->count, reg->count + 1);
   __sync_synchronize();
   p_atomic_set(&reg->seq, reg->seq + 1);

   simple_mtx_unlock(&device->descriptor_heap_lock);
   return true;
}

static void
d12_descriptor_registry_remove(struct d12_device *device,
                               struct d12_descriptor_heap *heap)
{
   struct d12_descriptor_registry *reg = &device->descriptor_registry;

   simple_mtx_lock(&device->descriptor_heap_lock);
   for (uint32_t i = 0; i < reg->count; i++) {
      if (reg->ranges[i].heap != heap)
         continue;

      p_atomic_set(&reg->seq, reg->seq + 1);
      __sync_synchronize();
      memmove(&reg->ranges[i], &reg->ranges[i + 1],
              (size_t)(reg->count - i - 1) * sizeof(*reg->ranges));
      p_atomic_set(&reg->count, reg->count - 1);
      __sync_synchronize();
      p_atomic_set(&reg->seq, reg->seq + 1);
      break;
   }
   simple_mtx_unlock(&device->descriptor_heap_lock);
}

void
d12_descriptor_registry_destroy(struct d12_device *device)
{
   struct d12_descriptor_registry *reg = &device->descriptor_registry;

   assert(reg->count == 0);
   for (uint32_t i = 0; i < reg->retired_count; i++)
      free(reg->retired[i]);
   free(reg->retired);
   free(reg->ranges);
   memset(reg, 0, sizeof(*reg));
}

/* Lock-free lookup: binary search over the published table and retry when
 * a writer changed it underneath. Returns the heap and the slot index, or
 * NULL when the pointer is not a slot of any registered heap. */
static struct d12_descriptor_heap *
d12_descriptor_registry_lookup(struct d12_device *device,
                               const struct d12_descriptor *slot,
                               uint32_t *index)
{
   struct d12_descriptor_registry *reg = &device->descriptor_registry;
   const uintptr_t ptr = (uintptr_t)slot;

   for (;;) {
      const uint32_t seq = p_atomic_read(&reg->seq);
      if (seq & 1)
         continue;

      const struct d12_descriptor_heap_range *table =
         p_atomic_read(&reg->ranges);
      uint32_t lo = 0, hi = p_atomic_read(&reg->count);
      struct d12_descriptor_heap *heap = NULL;
      uint32_t found = 0;

      /* Last range whose start is at or below the pointer. */
      while (lo < hi) {
         const uint32_t mid = lo + (hi - lo) / 2;
         if (table[mid].start <= ptr)
            lo = mid + 1;
         else
            hi = mid;
      }
      if (lo) {
         const struct d12_descriptor_heap_range r = table[lo - 1];
         if (ptr < r.end && (ptr - r.start) % sizeof(*slot) == 0) {
            heap = r.heap;
            found = (uint32_t)((ptr - r.start) / sizeof(*slot));
         }
      }

      __atomic_thread_fence(__ATOMIC_ACQUIRE);
      if (p_atomic_read_relaxed(&reg->seq) == seq) {
         *index = found;
         return heap;
      }
   }
}

static ULONG STDMETHODCALLTYPE
d12_heap_Release(ID3D12DescriptorHeap *iface)
{
   struct d12_descriptor_heap *heap = d12_heap_from_iface(iface);
   uint32_t refs = p_atomic_dec_return(&heap->refcount);

   if (refs == 0) {
      d12_descriptor_registry_remove(heap->device, heap);

      if (heap->texture_bo)
         agx_bo_unreference(&heap->device->dev, heap->texture_bo);
      if (heap->pbe_bo)
         agx_bo_unreference(&heap->device->dev, heap->pbe_bo);
      if (heap->residency_bo)
         agx_bo_unreference(&heap->device->dev, heap->residency_bo);
      if (heap->sampler_bo)
         agx_bo_unreference(&heap->device->dev, heap->sampler_bo);
      if (heap->bo)
         agx_bo_unreference(&heap->device->dev, heap->bo);
      else
         free(heap->slots);
      free(heap->live_resource_words);
      d12_private_data_store_destroy(&heap->private_data);
      d12_device_release(heap->device);
      free(heap);
   }

   return refs;
}

static void
d12_descriptor_heap_rebuild_samplers(struct d12_descriptor_heap *heap)
{
   if (!heap->sampler_bo)
      return;

   /* Lock2 can refuse after device loss, and this runs on every descriptor
    * update.  A driver must skip the native rebuild, not fault in the
    * application that wrote the descriptor. */
   uint8_t *native = agx_bo_map(heap->sampler_bo);
   if (!native)
      return;

   const unsigned stride =
      agx_chip_sampler_heap_element_size(heap->device->dev.chip);
   if (!stride)
      return;

   uint16_t hash_table[2048];
   uint32_t unique = 0;
   bool overflow = false;
   memset(hash_table, 0xff, sizeof(hash_table));

   static_assert(sizeof(struct agx_sampler_packed) == sizeof(uint64_t),
                 "sampler deduplication key");

   for (UINT logical = 0; logical < heap->desc.NumDescriptors; logical++) {
      struct agx_sampler_packed state;
      struct d12_descriptor *slot = &heap->slots[logical];
      if (slot->sampler_kind == D12_DESC_SAMPLER)
         state = slot->sampler;
      else
         agx_pack_txf_sampler(&state);

      uint64_t key;
      memcpy(&key, &state, sizeof(key));
      uint32_t bucket = ((uint32_t)key ^ (uint32_t)(key >> 32)) & 2047;
      uint32_t index = 0;
      while (hash_table[bucket] != UINT16_MAX) {
         index = hash_table[bucket];
         if (memcmp(native + ((size_t)index * stride), &state,
                    sizeof(state)) == 0)
            break;
         bucket = (bucket + 1) & 2047;
      }

      if (hash_table[bucket] == UINT16_MAX) {
         if (unique >= 1024) {
            overflow = true;
            index = 0;
         } else {
            index = unique++;
            uint8_t *element = native + ((size_t)index * stride);
            memset(element, 0, stride);
            memcpy(element, &state, sizeof(state));
            hash_table[bucket] = index;
         }
      }

      slot->sampler_native_index = index;
   }

   heap->sampler_unique_count = overflow ? UINT32_MAX : unique;
}

/* Mirror one slot into the heap's native tables. The caller orders the
 * writes with a fence once per batch. */
static void
d12_descriptor_heap_refresh_slot(struct d12_descriptor_heap *heap,
                                 uint32_t index, struct d12_descriptor *slot)
{
   struct d12_device *device = heap->device;

   if (heap->live_resource_words) {
      const uint64_t mask = UINT64_C(1) << (index & 63);
      uint64_t *word = &heap->live_resource_words[index >> 6];
      const bool live =
         slot->kind != D12_DESC_NONE && slot->kind != D12_DESC_SAMPLER &&
         (slot->resource || slot->counter_address ||
          (slot->address && slot->address != device->null_va));
      if (live)
         __atomic_fetch_or(word, mask, __ATOMIC_RELAXED);
      else
         __atomic_fetch_and(word, ~mask, __ATOMIC_RELAXED);
   }
   if (heap->texture_bo) {
      uint8_t *textures = agx_bo_map(heap->texture_bo);
      if (textures)
         d12_refresh_texture_descriptor(
            device, textures + index * heap->texture_stride_B, slot);
   }
   if (heap->pbe_bo) {
      uint8_t *pbes = agx_bo_map(heap->pbe_bo);
      if (pbes)
         d12_refresh_pbe_descriptor(
            device, pbes + index * heap->pbe_stride_B, slot);
   }
   if (heap->residency_bo) {
      uint64_t *residency = agx_bo_map(heap->residency_bo);
      if (residency)
         residency[index] = d12_descriptor_sparse_residency_address(slot);
   }
   if (heap->sampler_bo) {
      /* The shader-visible D3D12 limit is 2,048, while one AGX command
       * names at most 1,024 native sampler states.  Rebuild immediately
       * so descriptor updates remain visible to already-recorded command
       * lists, exactly like the coherent logical descriptor heap. */
      d12_descriptor_heap_rebuild_samplers(heap);
   }
}

static inline bool
d12_descriptor_heap_has_native(const struct d12_descriptor_heap *heap)
{
   return heap->texture_bo || heap->pbe_bo || heap->residency_bo ||
          heap->sampler_bo || heap->live_resource_words;
}

void
d12_descriptor_heap_update_slot(struct d12_device *device,
                                struct d12_descriptor *slot)
{
   if (!device || !slot)
      return;

   uint32_t index;
   struct d12_descriptor_heap *heap =
      d12_descriptor_registry_lookup(device, slot, &index);
   if (!heap || !d12_descriptor_heap_has_native(heap))
      return;

   d12_descriptor_heap_refresh_slot(heap, index, slot);

   /* All native BOs are coherent. Order the native descriptor write after
    * the application's 40-byte slot write before another thread records
    * a dispatch that publishes these bases. */
   __sync_synchronize();
}

/* Refresh a contiguous run of slots that was written as one range, as
 * CopyDescriptors does: one heap lookup and one fence for the run. A run
 * that is not inside one registered heap is refreshed slot by slot. */
void
d12_descriptor_heap_update_range(struct d12_device *device,
                                 struct d12_descriptor *first, uint32_t count)
{
   if (!device || !first || !count)
      return;

   uint32_t index;
   struct d12_descriptor_heap *heap =
      d12_descriptor_registry_lookup(device, first, &index);
   if (!heap || count > heap->desc.NumDescriptors - index) {
      for (uint32_t i = 0; i < count; i++)
         d12_descriptor_heap_update_slot(device, &first[i]);
      return;
   }
   if (!d12_descriptor_heap_has_native(heap))
      return;

   for (uint32_t i = 0; i < count; i++)
      d12_descriptor_heap_refresh_slot(heap, index + i, &first[i]);
   __sync_synchronize();
}

D12_DEFINE_PRIVATE_DATA_METHODS(d12_heap, ID3D12DescriptorHeap,
                                d12_heap_from_iface)

static HRESULT STDMETHODCALLTYPE
d12_heap_GetDevice(ID3D12DescriptorHeap *iface, REFIID riid, void **out)
{
   struct d12_descriptor_heap *heap = d12_heap_from_iface(iface);
   return ID3D12Device_QueryInterface(&heap->device->iface, riid, out);
}

static D12_AGGREGATE_RETURN(D3D12_DESCRIPTOR_HEAP_DESC) STDMETHODCALLTYPE
d12_heap_GetDesc(ID3D12DescriptorHeap *iface
                 D12_AGGREGATE_OUT(D3D12_DESCRIPTOR_HEAP_DESC))
{
   struct d12_descriptor_heap *heap = d12_heap_from_iface(iface);
   D12_RETURN_AGGREGATE(heap->desc);
}

static D12_AGGREGATE_RETURN(D3D12_CPU_DESCRIPTOR_HANDLE) STDMETHODCALLTYPE
d12_heap_GetCPUDescriptorHandleForHeapStart(
   ID3D12DescriptorHeap *iface
   D12_AGGREGATE_OUT(D3D12_CPU_DESCRIPTOR_HANDLE))
{
   struct d12_descriptor_heap *heap = d12_heap_from_iface(iface);
   D3D12_CPU_DESCRIPTOR_HANDLE handle = {.ptr = (SIZE_T)heap->slots};
   D12_RETURN_AGGREGATE(handle);
}

static D12_AGGREGATE_RETURN(D3D12_GPU_DESCRIPTOR_HANDLE) STDMETHODCALLTYPE
d12_heap_GetGPUDescriptorHandleForHeapStart(
   ID3D12DescriptorHeap *iface
   D12_AGGREGATE_OUT(D3D12_GPU_DESCRIPTOR_HANDLE))
{
   struct d12_descriptor_heap *heap = d12_heap_from_iface(iface);

   /*
    * The GPU handle is the CPU address of the slot array.
    *
    * The handle stays an opaque CPU token so applications can perform the
    * advertised 40-byte arithmetic. SetDescriptorHeaps supplies the owning
    * object when an unbounded table needs to translate that token to the
    * coherent BO's GPU address.
    */
   D3D12_GPU_DESCRIPTOR_HANDLE handle = {
      .ptr = (heap->desc.Type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ||
              heap->desc.Type == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER)
                ? (UINT64)(uintptr_t)heap->slots
                : 0,
   };
   D12_RETURN_AGGREGATE(handle);
}

static const ID3D12DescriptorHeapVtbl d12_heap_vtbl = {
   /* every unimplemented slot fails safely */
   D12_ID3D12DescriptorHeap_DEFAULTS,
   .QueryInterface = d12_heap_QueryInterface,
   .AddRef = d12_heap_AddRef,
   .Release = d12_heap_Release,
   .GetPrivateData = d12_heap_GetPrivateData,
   .SetPrivateData = d12_heap_SetPrivateData,
   .SetPrivateDataInterface = d12_heap_SetPrivateDataInterface,
   .SetName = d12_heap_SetName,
   .GetDevice = d12_heap_GetDevice,
   .GetDesc = d12_heap_GetDesc,
   .GetCPUDescriptorHandleForHeapStart =
      d12_heap_GetCPUDescriptorHandleForHeapStart,
   .GetGPUDescriptorHandleForHeapStart =
      d12_heap_GetGPUDescriptorHandleForHeapStart,
};

HRESULT
d12_descriptor_heap_create(struct d12_device *device,
                           const D3D12_DESCRIPTOR_HEAP_DESC *desc,
                           REFIID riid, void **out)
{
   if (!out)
      return E_POINTER;
   if (!desc || !desc->NumDescriptors)
      return E_INVALIDARG;

   switch (desc->Type) {
   case D3D12_DESCRIPTOR_HEAP_TYPE_RTV:
   case D3D12_DESCRIPTOR_HEAP_TYPE_DSV:
      if (desc->Flags & D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE) {
         mesa_loge("CreateDescriptorHeap: RTV/DSV heaps cannot be "
                   "shader-visible");
         return E_INVALIDARG;
      }
      break;

   case D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV:
      /* Shader-visible storage is allocated below from a coherent BO so
       * unbounded ranges can index it directly. */
      break;

   case D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER:
      /* Sampler creation is CPU-immediate too.  The packed hardware state
       * lives in the same fixed-size slot; binding/upload is a later
       * texturing concern and does not make heap object creation invalid. */
      break;

   default:
      mesa_loge("CreateDescriptorHeap: invalid type %d", (int)desc->Type);
      return E_INVALIDARG;
   }

   struct d12_descriptor_heap *heap = calloc(1, sizeof(*heap));
   if (!heap)
      return E_OUTOFMEMORY;

   if ((size_t)desc->NumDescriptors >
       SIZE_MAX / sizeof(struct d12_descriptor)) {
      free(heap);
      return E_OUTOFMEMORY;
   }
   size_t slots_size =
      (size_t)desc->NumDescriptors * sizeof(struct d12_descriptor);
   bool shader_visible =
      desc->Flags & D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;

   if (shader_visible &&
       desc->Type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) {
      struct d12_native_descriptor_layout layout;
      const enum agx_descriptor_generation generation =
         agx_chip_descriptor_generation(device->dev.chip);
      if (!d12_native_descriptor_layout((uint32_t)generation, &layout)) {
         mesa_loge("CreateDescriptorHeap: descriptor generation %u has no "
                   "proven native heap layout", (unsigned)generation);
         free(heap);
         return E_FAIL;
      }
      heap->texture_stride_B = layout.texture_stride_B;
      heap->pbe_stride_B = layout.pbe_stride_B;
   }

   if (shader_visible &&
       desc->Type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) {
      const size_t word_count = DIV_ROUND_UP(desc->NumDescriptors, 64);
      if (word_count > UINT32_MAX ||
          word_count > SIZE_MAX / sizeof(uint64_t)) {
         free(heap);
         return E_OUTOFMEMORY;
      }
      heap->live_resource_words = calloc(word_count, sizeof(uint64_t));
      if (!heap->live_resource_words) {
         free(heap);
         return E_OUTOFMEMORY;
      }
      heap->live_resource_word_count = (uint32_t)word_count;
   }

   if (shader_visible) {
      heap->bo = agx_bo_create(&device->dev, slots_size, 0,
                               AGX_BO_WRITEBACK,
                               "d3d12 shader-visible descriptors");
      if (heap->bo)
         heap->slots = agx_bo_map(heap->bo);

      if (desc->Type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) {
         size_t texture_size, pbe_size;
         /* One extra permanently-null PBE slot is the robust sink for
          * dynamically indexed typed-buffer atomics. Unlike texture fetch,
          * AGX buffer image atomics do not suppress coordinates beyond the
          * descriptor's element count. */
         if (!d12_native_descriptor_span(
                heap->texture_stride_B, desc->NumDescriptors, 0,
                &texture_size) ||
             !d12_native_descriptor_span(
                heap->pbe_stride_B, desc->NumDescriptors, 1, &pbe_size)) {
            if (heap->bo)
               agx_bo_unreference(&device->dev, heap->bo);
            free(heap->live_resource_words);
            free(heap);
            return E_OUTOFMEMORY;
         }
         heap->texture_bo =
            agx_bo_create(&device->dev, texture_size, 0, AGX_BO_WRITEBACK,
                          "d3d12 bindless texture descriptors");
         heap->pbe_bo =
            agx_bo_create(&device->dev, pbe_size, 0, AGX_BO_WRITEBACK,
                          "d3d12 bindless PBE descriptors");
         heap->residency_bo = agx_bo_create(
            &device->dev,
            (size_t)desc->NumDescriptors * sizeof(uint64_t), 0,
            AGX_BO_WRITEBACK, "d3d12 bindless sparse residency");
      } else if (desc->Type == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER) {
         const unsigned sampler_stride =
            agx_chip_sampler_heap_element_size(device->dev.chip);
         size_t sampler_size =
            (size_t)MIN2(desc->NumDescriptors, 1024) * sampler_stride;
         if (sampler_stride)
            heap->sampler_bo =
               agx_bo_create(&device->dev, sampler_size, 0, AGX_BO_WRITEBACK,
                             "d3d12 bindless sampler descriptors");
      }
   } else {
      heap->slots = calloc(desc->NumDescriptors,
                           sizeof(struct d12_descriptor));
   }

   if (heap->bo && heap->slots)
      memset(heap->slots, 0, slots_size);

   /*
    * Start every slot as a NULL descriptor rather than as zeroes.
    *
    * A shader may read a descriptor the application never wrote -- D3D12
    * calls the contents undefined, but the read still has to be survivable,
    * and the suite tests exactly that (test_undefined_structured_raw_*). A
    * zeroed slot means address 0, and the GPU faults on it: `address: 0x0,
    * unit: DCMP, reason: Unmapped`, one of the last GPU timeouts in the
    * suite. Pointing undefined slots at the zero-filled null region makes
    * such a read return zeroes, which is what a null descriptor does.
    */
   if (heap->slots) {
      for (UINT i = 0; i < desc->NumDescriptors; i++) {
         heap->slots[i].address = device->null_va;
         heap->slots[i].size_B = D12_NULL_REGION_SIZE;
      }
   }

   /* A BO whose Lock2 fails is unusable: every later native descriptor
    * write would dereference NULL inside the application.  Treat a failed
    * map exactly like a failed allocation and fail heap creation. */
   bool native_mapped = true;
   if (heap->texture_bo && heap->pbe_bo) {
      uint8_t *textures = agx_bo_map(heap->texture_bo);
      uint8_t *pbes = agx_bo_map(heap->pbe_bo);
      if (textures && pbes) {
         for (uint64_t i = 0; i <= desc->NumDescriptors; i++) {
            if (i < desc->NumDescriptors &&
                !d12_set_null_texture_descriptor(
                   device, textures + i * heap->texture_stride_B))
               native_mapped = false;
            if (!d12_set_null_pbe_descriptor(
                   device, pbes + i * heap->pbe_stride_B))
               native_mapped = false;
         }
      } else {
         native_mapped = false;
      }
   }
   if (heap->residency_bo) {
      uint64_t *residency = agx_bo_map(heap->residency_bo);
      if (residency) {
         memset(residency, 0,
                (size_t)desc->NumDescriptors * sizeof(uint64_t));
      } else {
         native_mapped = false;
      }
   }
   if (heap->sampler_bo) {
      uint8_t *samplers = agx_bo_map(heap->sampler_bo);
      const unsigned stride =
         agx_chip_sampler_heap_element_size(device->dev.chip);
      if (samplers) {
         for (UINT i = 0; i < MIN2(desc->NumDescriptors, 1024); i++) {
            uint8_t *element = samplers + ((size_t)i * stride);
            memset(element, 0, stride);
            agx_pack_txf_sampler((struct agx_sampler_packed *)element);
         }
      } else {
         native_mapped = false;
      }
   }

   bool native_ok = native_mapped &&
      (!shader_visible ||
       (desc->Type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
           ? heap->texture_bo && heap->pbe_bo && heap->residency_bo
           : desc->Type == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER
                ? heap->sampler_bo != NULL
                : true));
   if (!heap->slots || !native_ok) {
      if (heap->texture_bo)
         agx_bo_unreference(&device->dev, heap->texture_bo);
      if (heap->pbe_bo)
         agx_bo_unreference(&device->dev, heap->pbe_bo);
      if (heap->residency_bo)
         agx_bo_unreference(&device->dev, heap->residency_bo);
      if (heap->sampler_bo)
         agx_bo_unreference(&device->dev, heap->sampler_bo);
      if (heap->bo)
         agx_bo_unreference(&device->dev, heap->bo);
      else
         free(heap->slots);
      free(heap->live_resource_words);
      free(heap);
      return E_OUTOFMEMORY;
   }

   heap->iface.lpVtbl = (ID3D12DescriptorHeapVtbl *)&d12_heap_vtbl;
   heap->refcount = 1;
   heap->device = device;
   heap->desc = *desc;
   d12_descriptor_heap_rebuild_samplers(heap);
   d12_private_data_store_init(&heap->private_data, (IUnknown *)&heap->iface);
   d12_device_add_ref(device);

   if (!d12_descriptor_registry_add(device, heap)) {
      d12_heap_Release(&heap->iface);
      return E_OUTOFMEMORY;
   }

   HRESULT hr = d12_heap_QueryInterface(&heap->iface, riid, out);
   d12_heap_Release(&heap->iface);
   return hr;
}
