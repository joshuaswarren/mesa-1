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
      free(heap->sampler_hash);
      simple_mtx_destroy(&heap->sampler_lock);
      d12_private_data_store_destroy(&heap->private_data);
      d12_device_release(heap->device);
      free(heap);
   }

   return refs;
}

/*
 * Native sampler table.
 *
 * A shader-visible D3D12 sampler heap holds up to 2,048 logical samplers
 * while one AGX command names at most 1,024 native sampler states, so the
 * native table is the deduplicated set of packed states and each logical
 * slot records its native index. The dedup table lives on the heap: a
 * descriptor write interns one state, and only exhausting the 1,024 native
 * entries forces a rebuild over the whole heap, which recomputes the live
 * set so states that no slot references any more are dropped. All of this
 * runs under sampler_lock.
 */
#define D12_SAMPLER_HASH_BUCKETS 2048
#define D12_SAMPLER_NATIVE_LIMIT 1024

static_assert(sizeof(struct agx_sampler_packed) == sizeof(uint64_t),
              "sampler deduplication key");
static_assert(D12_SAMPLER_HASH_BUCKETS == 1u << 11,
              "bucket index takes the top 11 bits of the mixed key");

static void
d12_sampler_slot_state(const struct d12_descriptor *slot,
                       struct agx_sampler_packed *state)
{
   if (slot->sampler_kind == D12_DESC_SAMPLER)
      *state = slot->sampler;
   else
      agx_pack_txf_sampler(state);
}

/* The native BO holds min(NumDescriptors, 1024) states: a heap with fewer
 * logical slots than that can never reference more states than slots. */
static uint32_t
d12_sampler_heap_native_limit(const struct d12_descriptor_heap *heap)
{
   return MIN2(heap->desc.NumDescriptors, D12_SAMPLER_NATIVE_LIMIT);
}

/* Return the native index of a packed state, appending it to the native
 * table when it is new, or UINT32_MAX when the table is full. */
static uint32_t
d12_sampler_heap_intern(struct d12_descriptor_heap *heap, uint8_t *native,
                        unsigned stride,
                        const struct agx_sampler_packed *state)
{
   uint64_t key;
   memcpy(&key, state, sizeof(key));
   /* Packed samplers differ in a few scattered bits (filters, wrap modes,
    * LOD clamps); multiply by an odd 64-bit constant and keep the top bits
    * so those differences spread over the buckets. */
   uint32_t bucket =
      (uint32_t)((key * UINT64_C(0x9E3779B97F4A7C15)) >> 53);

   /* At most 1,024 of the 2,048 buckets are ever occupied, so the probe
    * always reaches an empty bucket. */
   while (heap->sampler_hash[bucket] != UINT16_MAX) {
      const uint32_t index = heap->sampler_hash[bucket];
      if (memcmp(native + ((size_t)index * stride), state,
                 sizeof(*state)) == 0)
         return index;
      bucket = (bucket + 1) & (D12_SAMPLER_HASH_BUCKETS - 1);
   }

   if (heap->sampler_unique_count >= d12_sampler_heap_native_limit(heap))
      return UINT32_MAX;

   const uint32_t index = heap->sampler_unique_count++;
   uint8_t *element = native + ((size_t)index * stride);
   memset(element, 0, stride);
   memcpy(element, state, sizeof(*state));
   heap->sampler_hash[bucket] = index;
   return index;
}

/* Rebuild the native table from every logical slot. Overflowing slots get
 * native index 0 and the heap is marked with UINT32_MAX so recording
 * refuses it, exactly as before. */
static void
d12_descriptor_heap_rebuild_samplers_locked(struct d12_descriptor_heap *heap)
{
   if (!heap->sampler_bo || !heap->sampler_hash)
      return;

   /* Lock2 can refuse after device loss, and this runs on descriptor
    * updates.  A driver must skip the native rebuild, not fault in the
    * application that wrote the descriptor. */
   uint8_t *native = agx_bo_map(heap->sampler_bo);
   if (!native)
      return;

   const unsigned stride =
      agx_chip_sampler_heap_element_size(heap->device->dev.chip);
   if (!stride)
      return;

   memset(heap->sampler_hash, 0xff,
          D12_SAMPLER_HASH_BUCKETS * sizeof(*heap->sampler_hash));
   heap->sampler_unique_count = 0;
   bool overflow = false;

   for (UINT logical = 0; logical < heap->desc.NumDescriptors; logical++) {
      struct d12_descriptor *slot = &heap->slots[logical];
      struct agx_sampler_packed state;
      d12_sampler_slot_state(slot, &state);

      uint32_t index = d12_sampler_heap_intern(heap, native, stride, &state);
      if (index == UINT32_MAX) {
         overflow = true;
         index = 0;
      }
      slot->sampler_native_index = index;
   }

   if (overflow)
      heap->sampler_unique_count = UINT32_MAX;
}

/* One logical slot was written: intern its state. The native table is
 * updated in place so the change is visible to already-recorded command
 * lists, like the coherent logical heap itself. */
static void
d12_descriptor_heap_write_sampler_locked(struct d12_descriptor_heap *heap,
                                         uint32_t index)
{
   /* An overflowed heap can only find out whether this write ended the
    * overflow by recounting; that is the O(N) rebuild it always paid. */
   if (heap->sampler_unique_count == UINT32_MAX) {
      d12_descriptor_heap_rebuild_samplers_locked(heap);
      return;
   }

   uint8_t *native = agx_bo_map(heap->sampler_bo);
   if (!native)
      return;

   const unsigned stride =
      agx_chip_sampler_heap_element_size(heap->device->dev.chip);
   if (!stride)
      return;

   struct d12_descriptor *slot = &heap->slots[index];
   struct agx_sampler_packed state;
   d12_sampler_slot_state(slot, &state);

   const uint32_t native_index =
      d12_sampler_heap_intern(heap, native, stride, &state);
   if (native_index == UINT32_MAX) {
      /* The table only grows between rebuilds, so states that were
       * overwritten since may still occupy it. Recount before giving up. */
      d12_descriptor_heap_rebuild_samplers_locked(heap);
      return;
   }
   slot->sampler_native_index = native_index;
}

/* Mirror one resource-heap slot into the heap's native tables. */
static void
d12_descriptor_heap_refresh_resource_slot(struct d12_descriptor_heap *heap,
                                          uint32_t index,
                                          struct d12_descriptor *slot)
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
   /* Gen7 mirrors retain both declaration shapes. A physical 2D descriptor
    * cannot accept an array coordinate/LOD ABI, even for a one-layer view. */
   const unsigned windows = d12_native_descriptor_legacy_shapes(
      agx_chip_descriptor_generation(device->dev.chip)) ? 3 : 1;
   uint8_t *textures = heap->texture_bo ? agx_bo_map(heap->texture_bo) : NULL;
   uint8_t *pbes = heap->pbe_bo ? agx_bo_map(heap->pbe_bo) : NULL;
   for (unsigned window = 0; window < windows; ++window) {
      const size_t native_index =
         (size_t)window * ((size_t)heap->desc.NumDescriptors + 1) + index;
      struct d12_descriptor effective = *slot;
      if (window && slot->resource &&
          slot->resource->desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
         if (textures)
            memcpy(textures + native_index * heap->texture_stride_B,
                   textures + index * heap->texture_stride_B, heap->texture_stride_B);
         if (pbes)
            memcpy(pbes + native_index * heap->pbe_stride_B,
                   pbes + index * heap->pbe_stride_B, heap->pbe_stride_B);
         continue;
      }
      if (window)
         d12_apply_texture_shape(&effective, window == 1
            ? D12_TEXTURE_SHAPE_2D : D12_TEXTURE_SHAPE_2D_ARRAY);
      if (textures) {
         uint8_t *out = textures + native_index * heap->texture_stride_B;
         if (!slot->resource)
            memcpy(out, heap->null_texture, heap->texture_stride_B);
         else
            d12_refresh_texture_descriptor(device, out, &effective);
      }
      if (pbes) {
         uint8_t *out = pbes + native_index * heap->pbe_stride_B;
         if (!slot->resource)
            memcpy(out, heap->null_pbe, heap->pbe_stride_B);
         else
            d12_refresh_pbe_descriptor(device, out, &effective);
      }
   }
   if (heap->residency_bo) {
      uint64_t *residency = agx_bo_map(heap->residency_bo);
      if (residency)
         residency[index] = d12_descriptor_sparse_residency_address(slot);
   }
}

static inline bool
d12_descriptor_heap_has_native(const struct d12_descriptor_heap *heap)
{
   return heap->texture_bo || heap->pbe_bo || heap->residency_bo ||
          heap->sampler_bo || heap->live_resource_words;
}

/* Mirror a run of slots that the application has just written. A sampler
 * heap takes its table lock once for the run; every heap orders the native
 * writes with one fence. */
static void
d12_descriptor_heap_refresh_slots(struct d12_descriptor_heap *heap,
                                  uint32_t index, struct d12_descriptor *first,
                                  uint32_t count)
{
   if (heap->sampler_bo) {
      simple_mtx_lock(&heap->sampler_lock);
      for (uint32_t i = 0; i < count; i++)
         d12_descriptor_heap_write_sampler_locked(heap, index + i);
      simple_mtx_unlock(&heap->sampler_lock);
   } else {
      for (uint32_t i = 0; i < count; i++)
         d12_descriptor_heap_refresh_resource_slot(heap, index + i,
                                                   &first[i]);
   }

   /* All native BOs are coherent. Order the native descriptor writes after
    * the application's 40-byte slot writes before another thread records
    * a dispatch that publishes these bases. */
   __sync_synchronize();
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

   d12_descriptor_heap_refresh_slots(heap, index, slot, 1);
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

   d12_descriptor_heap_refresh_slots(heap, index, first, count);
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
   if (out)
      *out = NULL;
   if (!desc || !desc->NumDescriptors || desc->NodeMask > 1 ||
       ((unsigned)desc->Flags & ~D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE))
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
      if ((desc->Flags & D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE) &&
          desc->NumDescriptors >
             D3D12_MAX_SHADER_VISIBLE_DESCRIPTOR_HEAP_SIZE_TIER_2)
         return E_INVALIDARG;
      break;

   case D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER:
      /* Sampler creation is CPU-immediate too.  The packed hardware state
       * lives in the same fixed-size slot; binding/upload is a later
       * texturing concern and does not make heap object creation invalid. */
      if ((desc->Flags & D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE) &&
          desc->NumDescriptors > D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE)
         return E_INVALIDARG;
      break;

   default:
      mesa_loge("CreateDescriptorHeap: invalid type %d", (int)desc->Type);
      return E_INVALIDARG;
   }

   /* D3D12 permits a validation-only call. Do not allocate storage or take
    * a device reference when the caller only asks whether the desc is legal. */
   if (!out)
      return S_FALSE;

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
         const bool variants = d12_native_descriptor_legacy_shapes(
            agx_chip_descriptor_generation(device->dev.chip));
         /* One extra permanently-null PBE slot is the robust sink for
          * dynamically indexed typed-buffer atomics. Unlike texture fetch,
          * AGX buffer image atomics do not suppress coordinates beyond the
          * descriptor's element count. */
         if (!d12_native_descriptor_span(
                heap->texture_stride_B, desc->NumDescriptors, variants ? 1 : 0,
                &texture_size) ||
             !d12_native_descriptor_span(
                heap->pbe_stride_B, desc->NumDescriptors, 1, &pbe_size)) {
            if (heap->bo)
               agx_bo_unreference(&device->dev, heap->bo);
            free(heap->live_resource_words);
            free(heap);
            return E_OUTOFMEMORY;
         }
         if (variants) {
            if (texture_size > SIZE_MAX / 3 || pbe_size > SIZE_MAX / 3) {
               if (heap->bo)
                  agx_bo_unreference(&device->dev, heap->bo);
               free(heap->live_resource_words);
               free(heap);
               return E_OUTOFMEMORY;
            }
            texture_size *= 3;
            pbe_size *= 3;
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
         if (heap->sampler_bo) {
            heap->sampler_hash = malloc(D12_SAMPLER_HASH_BUCKETS *
                                        sizeof(*heap->sampler_hash));
            if (!heap->sampler_hash) {
               agx_bo_unreference(&device->dev, heap->sampler_bo);
               heap->sampler_bo = NULL;
            }
         }
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
      /* Pack the generation's null descriptors once; every slot starts as
       * a copy and resource-less writes reuse them. */
      if (textures && pbes &&
          heap->texture_stride_B <= sizeof(heap->null_texture) &&
          heap->pbe_stride_B <= sizeof(heap->null_pbe) &&
          d12_set_null_texture_descriptor(device, heap->null_texture) &&
          d12_set_null_pbe_descriptor(device, heap->null_pbe)) {
         const bool variants = d12_native_descriptor_legacy_shapes(
            agx_chip_descriptor_generation(device->dev.chip));
         const uint64_t pbe_slots = variants
            ? 3 * ((uint64_t)desc->NumDescriptors + 1)
            : (uint64_t)desc->NumDescriptors + 1;
         const uint64_t texture_slots = variants ? pbe_slots : desc->NumDescriptors;
         for (uint64_t i = 0; i < texture_slots; i++) {
            memcpy(textures + i * heap->texture_stride_B,
                   heap->null_texture, heap->texture_stride_B);
         }
         /* ... plus the permanently-null PBE sink past the last slot. */
         for (uint64_t i = 0; i < pbe_slots; i++) {
            memcpy(pbes + i * heap->pbe_stride_B, heap->null_pbe,
                   heap->pbe_stride_B);
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
      free(heap->sampler_hash);
      free(heap);
      return E_OUTOFMEMORY;
   }

   heap->iface.lpVtbl = (ID3D12DescriptorHeapVtbl *)&d12_heap_vtbl;
   heap->refcount = 1;
   heap->device = device;
   heap->desc = *desc;
   simple_mtx_init(&heap->sampler_lock, mtx_plain);
   d12_descriptor_heap_rebuild_samplers_locked(heap);
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
