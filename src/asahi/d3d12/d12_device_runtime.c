/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Opaque runtime-owned constructor for the shared native AGX renderer core.
 * DDI glue includes only d12_device_runtime.h and never depends on the private
 * renderer header graph.
 */

#include "d12_device_runtime.h"

#include "d12_core_caps.h"
#include "d12_dxbc_wrap.h"
#include "d12_private.h"
#include "disasm.h"

#include "asahi/lib/agx_device_d3dkmt.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool
d12_device_runtime_command_type(
   enum d12_device_runtime_command_type runtime_type,
   D3D12_COMMAND_LIST_TYPE *type)
{
   if (!type)
      return false;

   switch (runtime_type) {
   case D12_DEVICE_RUNTIME_COMMAND_DIRECT:
      *type = D3D12_COMMAND_LIST_TYPE_DIRECT;
      return true;
   case D12_DEVICE_RUNTIME_COMMAND_BUNDLE:
      *type = D3D12_COMMAND_LIST_TYPE_BUNDLE;
      return true;
   case D12_DEVICE_RUNTIME_COMMAND_COMPUTE:
      *type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
      return true;
   case D12_DEVICE_RUNTIME_COMMAND_COPY:
      *type = D3D12_COMMAND_LIST_TYPE_COPY;
      return true;
   default:
      return false;
   }
}

static bool
d12_device_runtime_query_heap_type(uint32_t runtime_type,
                                   D3D12_QUERY_HEAP_TYPE *type)
{
   if (!type)
      return false;

   switch (runtime_type) {
   case 0:
      *type = D3D12_QUERY_HEAP_TYPE_OCCLUSION;
      return true;
   case 1:
      *type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
      return true;
   case 2:
      *type = D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS;
      return true;
   case 3:
      *type = D3D12_QUERY_HEAP_TYPE_SO_STATISTICS;
      return true;
   case 5:
      *type = D3D12_QUERY_HEAP_TYPE_COPY_QUEUE_TIMESTAMP;
      return true;
   default:
      return false;
   }
}

static bool
d12_device_runtime_query_type(uint32_t runtime_type, D3D12_QUERY_TYPE *type)
{
   if (!type)
      return false;

   switch (runtime_type) {
   case 0:
      *type = D3D12_QUERY_TYPE_OCCLUSION;
      return true;
   case 1:
      *type = D3D12_QUERY_TYPE_BINARY_OCCLUSION;
      return true;
   case 2:
      *type = D3D12_QUERY_TYPE_TIMESTAMP;
      return true;
   case 3:
      *type = D3D12_QUERY_TYPE_PIPELINE_STATISTICS;
      return true;
   case 4:
      *type = D3D12_QUERY_TYPE_SO_STATISTICS_STREAM0;
      return true;
   case 5:
      *type = D3D12_QUERY_TYPE_SO_STATISTICS_STREAM1;
      return true;
   case 6:
      *type = D3D12_QUERY_TYPE_SO_STATISTICS_STREAM2;
      return true;
   case 7:
      *type = D3D12_QUERY_TYPE_SO_STATISTICS_STREAM3;
      return true;
   default:
      return false;
   }
}

bool
d12_device_resolve_d3dkmt_adapter_luid(
   const struct agxw_adapter_identity *identity,
   struct agx_kmt_luid *luid)
{
   return agx_d3dkmt_find_adapter_luid(identity, luid);
}

HRESULT
d12_device_create_d3dkmt_runtime(
   const struct agx_d3dkmt_runtime_device *runtime_device,
   struct d12_device **out_device, uint64_t *out_shader_base)
{
   struct d12_device *device;
   HRESULT result;

   if (!runtime_device || !out_device || !out_shader_base)
      return E_INVALIDARG;
   *out_device = NULL;
   *out_shader_base = 0;

   device = calloc(1, sizeof(*device));
   if (!device)
      return E_OUTOFMEMORY;
   /* The DDI runtime submits through D3DKMT, which consumes an exact
    * allocation list for every submission. */
   device->submit_read_set = true;

   device->dev.fd = -1;
   if (!agx_d3dkmt_open_runtime_device(&device->dev, runtime_device)) {
      result = E_FAIL;
      goto fail_device;
   }

   if (!agx_open_device(NULL, &device->dev)) {
      agx_d3dkmt_close_device(&device->dev);
      result = E_FAIL;
      goto fail_device;
   }

   device->sync = util_sync_provider_d3dkmt(&device->dev);
   if (!device->sync) {
      result = E_OUTOFMEMORY;
      goto fail_open_device;
   }

   device->adapter_luid = (LUID){
      .LowPart = runtime_device->process_key.LowPart,
      .HighPart = runtime_device->process_key.HighPart,
   };
   result = d12_device_backend_init(
      device, (D3D_FEATURE_LEVEL)d12_core_get_caps()->max_feature_level);
   if (FAILED(result))
      goto fail_sync;

   *out_shader_base = device->dev.shader_base;
   *out_device = device;
   return S_OK;

fail_sync:
   device->sync->finalize(device->sync);
   device->sync = NULL;
fail_open_device:
   agx_close_device(&device->dev);
fail_device:
   free(device);
   return result;
}

void
d12_device_destroy_d3dkmt_runtime(struct d12_device *device)
{
   if (!device)
      return;

   d12_device_backend_cleanup(device);
   if (device->sync) {
      device->sync->finalize(device->sync);
      device->sync = NULL;
   }
   agx_close_device(&device->dev);
   free(device);
}

bool
d12_device_runtime_set_removal_callback(
   struct d12_device *device, d12_device_removal_callback callback,
   void *userdata)
{
   return device && d12_device_removal_set_callback(
                       &device->removal, callback, userdata);
}

HRESULT
d12_device_runtime_trim_memory(struct d12_device *device, bool periodic,
                               bool restart_periodic,
                               uint64_t bytes_to_trim)
{
   size_t target;

   if (!device || (periodic && restart_periodic))
      return E_INVALIDARG;
   target = bytes_to_trim > SIZE_MAX ? SIZE_MAX : (size_t)bytes_to_trim;
   (void)agx_bo_cache_trim(&device->dev, periodic, restart_periodic, target);
   return S_OK;
}

uint32_t
d12_device_runtime_paging_queue(struct d12_device *device)
{
   return device ? agx_d3dkmt_device_paging_queue(&device->dev) : 0;
}

HRESULT
d12_device_runtime_wait_paging(struct d12_device *device, uint64_t value)
{
   if (!device)
      return E_INVALIDARG;
   return agx_d3dkmt_device_wait_paging(&device->dev, value) ? S_OK : E_FAIL;
}

struct d12_runtime_rotation_entry {
   struct d12_resource *resource;
   struct agx_bo *bo;
   uint64_t old_bo_address;
   uint64_t old_resource_address;
   uint64_t old_stencil_address;
};

static bool
d12_runtime_rotation_compatible(const struct d12_resource *a,
                                const struct d12_resource *b)
{
   const D3D12_RESOURCE_DESC *x = &a->desc;
   const D3D12_RESOURCE_DESC *y = &b->desc;
   const D3D12_HEAP_DESC *hx = &a->placed_heap_desc;
   const D3D12_HEAP_DESC *hy = &b->placed_heap_desc;

   return x->Dimension == y->Dimension && x->Alignment == y->Alignment &&
          x->Width == y->Width && x->Height == y->Height &&
          x->DepthOrArraySize == y->DepthOrArraySize &&
          x->MipLevels == y->MipLevels && x->Format == y->Format &&
          x->SampleDesc.Count == y->SampleDesc.Count &&
          x->SampleDesc.Quality == y->SampleDesc.Quality &&
          x->Layout == y->Layout && x->Flags == y->Flags &&
          a->offset_B == b->offset_B &&
          a->is_depth_stencil == b->is_depth_stencil &&
          memcmp(&a->layout, &b->layout, sizeof(a->layout)) == 0 &&
          memcmp(&a->stencil_layout, &b->stencil_layout,
                 sizeof(a->stencil_layout)) == 0 &&
          hx->SizeInBytes == hy->SizeInBytes &&
          hx->Properties.Type == hy->Properties.Type &&
          hx->Properties.CPUPageProperty ==
             hy->Properties.CPUPageProperty &&
          hx->Properties.MemoryPoolPreference ==
             hy->Properties.MemoryPoolPreference &&
          hx->Properties.CreationNodeMask ==
             hy->Properties.CreationNodeMask &&
          hx->Properties.VisibleNodeMask ==
             hy->Properties.VisibleNodeMask &&
          hx->Alignment == hy->Alignment && hx->Flags == hy->Flags;
}

static struct d12_runtime_rotation_entry *
d12_runtime_rotation_entry_for_resource(
   struct d12_runtime_rotation_entry *entries, uint32_t count,
   const struct d12_resource *resource)
{
   for (uint32_t i = 0; i < count; ++i) {
      if (entries[i].resource == resource)
         return &entries[i];
   }
   return NULL;
}

static bool
d12_runtime_rotate_address(const struct d12_runtime_rotation_entry *entries,
                           uint32_t count, uint64_t address,
                           uint64_t *rotated)
{
   if (!address || !rotated)
      return false;
   for (uint32_t i = 0; i < count; ++i) {
      const uint64_t base = entries[i].old_bo_address;
      if (address >= base && address - base < entries[i].bo->size) {
         *rotated = entries[i].bo->va->addr + (address - base);
         return true;
      }
   }
   return false;
}

HRESULT
d12_device_runtime_rotate_resource_identities(
   struct d12_device *device, void *const *resources, uint32_t count)
{
   struct d12_runtime_rotation_entry stack_entries[8];
   struct agx_bo *stack_bos[8];
   struct d12_runtime_rotation_entry *entries = stack_entries;
   struct agx_bo **bos = stack_bos;
   HRESULT result = E_INVALIDARG;

   if (!device || (count && !resources))
      return E_INVALIDARG;
   if (count < 2)
      return S_OK;
   if (count > ARRAY_SIZE(stack_entries)) {
      if ((size_t)count > SIZE_MAX / sizeof(*entries) ||
          (size_t)count > SIZE_MAX / sizeof(*bos))
         return E_OUTOFMEMORY;
      entries = calloc(count, sizeof(*entries));
      bos = calloc(count, sizeof(*bos));
      if (!entries || !bos) {
         result = E_OUTOFMEMORY;
         goto cleanup;
      }
   } else {
      memset(entries, 0, (size_t)count * sizeof(*entries));
   }

   for (uint32_t i = 0; i < count; ++i) {
      struct d12_resource *resource =
         d12_resource_from_iface((ID3D12Resource *)resources[i]);
      if (!resource || resource->device != device || resource->reserved ||
          !resource->placed || !resource->bo || !resource->bo->va ||
          (resource->stencil_bo && resource->stencil_bo != resource->bo) ||
          (i && !d12_runtime_rotation_compatible(entries[0].resource,
                                                  resource)))
         goto cleanup;
      for (uint32_t j = 0; j < i; ++j) {
         if (entries[j].resource == resource ||
             entries[j].bo == resource->bo)
            goto cleanup;
      }
      entries[i] = (struct d12_runtime_rotation_entry){
         .resource = resource,
         .bo = resource->bo,
         .old_bo_address = resource->bo->va->addr,
         .old_resource_address = resource->addr,
         .old_stencil_address = resource->stencil_addr,
      };
      bos[i] = resource->bo;
   }

   /* Descriptor creation and the parallel AGX texture/PBE tables use this
    * same lock. Holding it across the identity update makes every view move
    * as one observable operation, as required by DXGI. */
   simple_mtx_lock(&device->descriptor_heap_lock);
   if (!agx_d3dkmt_bo_rotate_runtime_identities(&device->dev, bos, count)) {
      simple_mtx_unlock(&device->descriptor_heap_lock);
      result = E_FAIL;
      goto cleanup;
   }

   for (uint32_t i = 0; i < count; ++i) {
      struct d12_resource *resource = entries[i].resource;
      resource->addr = resource->bo->va->addr + resource->offset_B;
      if (resource->stencil_bo) {
         const uint64_t stencil_offset =
            entries[i].old_stencil_address - entries[i].old_bo_address;
         resource->stencil_addr = resource->bo->va->addr + stencil_offset;
      }
   }

   struct d12_descriptor_heap *heap;
   d12_descriptor_registry_foreach(device, heap, heap_index) {
      uint8_t *textures =
         heap->texture_bo ? agx_bo_map(heap->texture_bo) : NULL;
      uint8_t *pbes = heap->pbe_bo ? agx_bo_map(heap->pbe_bo) : NULL;

      for (UINT index = 0; index < heap->desc.NumDescriptors; ++index) {
         struct d12_descriptor *slot = &heap->slots[index];
         const bool resource_slot =
            slot->kind >= D12_DESC_RTV && slot->kind <= D12_DESC_UAV;
         struct d12_runtime_rotation_entry *entry =
            resource_slot ? d12_runtime_rotation_entry_for_resource(
                               entries, count, slot->resource)
                          : NULL;
         bool changed = false;
         if (entry && slot->address >= entry->old_resource_address &&
             slot->address - entry->old_resource_address < entry->bo->size) {
            slot->address = entry->resource->addr +
                            (slot->address - entry->old_resource_address);
            changed = true;
         }
         uint64_t counter_address;
         if (resource_slot &&
             d12_runtime_rotate_address(entries, count,
                                        slot->counter_address,
                                        &counter_address)) {
            slot->counter_address = counter_address;
            changed = true;
         }
         if (!changed)
            continue;
         if (textures)
            d12_refresh_texture_descriptor(
               device, textures + index * heap->texture_stride_B, slot);
         if (pbes)
            d12_refresh_pbe_descriptor(
               device, pbes + index * heap->pbe_stride_B, slot);
      }
   }
   __sync_synchronize();
   simple_mtx_unlock(&device->descriptor_heap_lock);
   result = S_OK;

cleanup:
   if (entries != stack_entries) {
      free(entries);
      free(bos);
   }
   return result;
}

HRESULT
d12_device_runtime_command_queue_create(
   struct d12_device *device, enum d12_device_runtime_command_type runtime_type,
   void **out_queue)
{
   D3D12_COMMAND_LIST_TYPE type;
   D3D12_COMMAND_QUEUE_DESC desc;

   if (!device || !out_queue ||
       !d12_device_runtime_command_type(runtime_type, &type) ||
       type == D3D12_COMMAND_LIST_TYPE_BUNDLE)
      return E_INVALIDARG;
   desc = (D3D12_COMMAND_QUEUE_DESC){
      .Type = type,
      .Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL,
      .Flags = D3D12_COMMAND_QUEUE_FLAG_NONE,
      .NodeMask = 1,
   };
   return d12_queue_create_ddi(device, &desc, &IID_ID3D12CommandQueue,
                               out_queue);
}

void
d12_device_runtime_command_queue_destroy(void *queue)
{
   if (queue)
      ID3D12CommandQueue_Release((ID3D12CommandQueue *)queue);
}

HRESULT
d12_device_runtime_command_queue_execute(void *queue, uint32_t count,
                                         void *const *lists)
{
   ID3D12CommandList *local[64];
   ID3D12CommandList **translated = local;

   if (!queue || (count && !lists))
      return E_INVALIDARG;
   if (count > ARRAY_SIZE(local)) {
      translated = calloc(count, sizeof(*translated));
      if (!translated)
         return E_OUTOFMEMORY;
   }
   for (uint32_t index = 0; index < count; ++index) {
      if (!lists[index]) {
         if (translated != local)
            free(translated);
         return E_INVALIDARG;
      }
      translated[index] = lists[index];
   }
   ID3D12CommandQueue_ExecuteCommandLists(
      (ID3D12CommandQueue *)queue, count, translated);
   if (translated != local)
      free(translated);
   struct d12_queue *native =
      d12_queue_from_iface((ID3D12CommandQueue *)queue);
   return d12_device_removed_reason(native->device);
}

HRESULT
d12_device_runtime_command_queue_update_tile_mappings(
   void *queue, void *resource, uint32_t region_count,
   const struct d12_frontend_tiled_resource_coordinate *region_starts,
   const struct d12_frontend_tile_region_size *region_sizes, void *heap,
   uint32_t range_count, const uint32_t *range_flags,
   const uint32_t *heap_start_offsets, const uint32_t *range_tile_counts,
   uint32_t flags)
{
   D3D12_TILED_RESOURCE_COORDINATE local_starts[16];
   D3D12_TILE_REGION_SIZE local_sizes[16];
   D3D12_TILE_RANGE_FLAGS local_range_flags[16];
   D3D12_TILED_RESOURCE_COORDINATE *starts = local_starts;
   D3D12_TILE_REGION_SIZE *sizes = local_sizes;
   D3D12_TILE_RANGE_FLAGS *translated_flags = local_range_flags;
   HRESULT result;

   if (!queue || !resource ||
       (flags & ~D12_FRONTEND_TILE_MAPPING_NO_HAZARD) ||
       (region_count && !region_starts))
      return E_INVALIDARG;
   if (region_count > ARRAY_SIZE(local_starts)) {
      starts = calloc(region_count, sizeof(*starts));
      if (!starts)
         return E_OUTOFMEMORY;
      if (region_sizes) {
         sizes = calloc(region_count, sizeof(*sizes));
         if (!sizes) {
            free(starts);
            return E_OUTOFMEMORY;
         }
      }
   }
   if (range_count > ARRAY_SIZE(local_range_flags) && range_flags) {
      translated_flags = calloc(range_count, sizeof(*translated_flags));
      if (!translated_flags) {
         if (starts != local_starts)
            free(starts);
         if (sizes != local_sizes)
            free(sizes);
         return E_OUTOFMEMORY;
      }
   }

   for (uint32_t i = 0; i < region_count; ++i) {
      starts[i] = (D3D12_TILED_RESOURCE_COORDINATE){
         .X = region_starts[i].x,
         .Y = region_starts[i].y,
         .Z = region_starts[i].z,
         .Subresource = region_starts[i].subresource,
      };
      if (region_sizes) {
         sizes[i] = (D3D12_TILE_REGION_SIZE){
            .NumTiles = region_sizes[i].num_tiles,
            .UseBox = region_sizes[i].use_box,
            .Width = region_sizes[i].width,
            .Height = region_sizes[i].height,
            .Depth = region_sizes[i].depth,
         };
      }
   }
   for (uint32_t i = 0; range_flags && i < range_count; ++i) {
      if (range_flags[i] != D12_FRONTEND_TILE_RANGE_NONE &&
          range_flags[i] != D12_FRONTEND_TILE_RANGE_NULL &&
          range_flags[i] != D12_FRONTEND_TILE_RANGE_SKIP &&
          range_flags[i] != D12_FRONTEND_TILE_RANGE_REUSE_SINGLE_TILE) {
         result = E_INVALIDARG;
         goto out;
      }
      translated_flags[i] = (D3D12_TILE_RANGE_FLAGS)range_flags[i];
   }

   result = d12_queue_update_tile_mappings(
      d12_queue_from_iface((ID3D12CommandQueue *)queue),
      d12_resource_from_iface((ID3D12Resource *)resource), region_count,
      starts, region_sizes ? sizes : NULL,
      heap ? d12_memheap_from_iface((ID3D12Heap *)heap) : NULL, range_count,
      range_flags ? translated_flags : NULL, heap_start_offsets,
      range_tile_counts, (D3D12_TILE_MAPPING_FLAGS)flags);

out:
   if (starts != local_starts)
      free(starts);
   if (sizes != local_sizes)
      free(sizes);
   if (translated_flags != local_range_flags)
      free(translated_flags);
   return result;
}

HRESULT
d12_device_runtime_command_queue_copy_tile_mappings(
   void *queue, void *destination_resource,
   const struct d12_frontend_tiled_resource_coordinate *destination_start,
   void *source_resource,
   const struct d12_frontend_tiled_resource_coordinate *source_start,
   const struct d12_frontend_tile_region_size *region_size, uint32_t flags)
{
   if (!queue || !destination_resource || !destination_start ||
       !source_resource || !source_start || !region_size ||
       (flags & ~D12_FRONTEND_TILE_MAPPING_NO_HAZARD))
      return E_INVALIDARG;
   const D3D12_TILED_RESOURCE_COORDINATE destination = {
      destination_start->x, destination_start->y, destination_start->z,
      destination_start->subresource,
   };
   const D3D12_TILED_RESOURCE_COORDINATE source = {
      source_start->x, source_start->y, source_start->z,
      source_start->subresource,
   };
   const D3D12_TILE_REGION_SIZE size = {
      region_size->num_tiles, region_size->use_box, region_size->width,
      region_size->height, region_size->depth,
   };
   return d12_queue_copy_tile_mappings(
      d12_queue_from_iface((ID3D12CommandQueue *)queue),
      d12_resource_from_iface((ID3D12Resource *)destination_resource),
      &destination,
      d12_resource_from_iface((ID3D12Resource *)source_resource), &source,
      &size, (D3D12_TILE_MAPPING_FLAGS)flags);
}

HRESULT
d12_device_runtime_fence_create(
   struct d12_device *device, uint32_t count,
   const struct d12_frontend_fence_placement *placements,
   void **out_fence)
{
   struct d12_placement_fence_entry local[32];

   if (!device || !count || count > ARRAY_SIZE(local) || !placements ||
       !out_fence)
      return E_INVALIDARG;
   for (uint32_t index = 0; index < count; ++index) {
      local[index] = (struct d12_placement_fence_entry){
         .current_gpu_address = placements[index].current_gpu_address,
         .monitored_gpu_address = placements[index].monitored_gpu_address,
         .current_cpu_address = placements[index].current_cpu_address,
         .flags = placements[index].flags,
         .lifetime_cookie = placements[index].lifetime_cookie,
         .release_lifetime = placements[index].release_lifetime,
      };
   }
   return d12_placement_fence_create(
      device, count, local, (struct d12_placement_fence **)out_fence);
}

void
d12_device_runtime_fence_destroy(void *fence)
{
   d12_placement_fence_destroy((struct d12_placement_fence *)fence);
}

HRESULT
d12_device_runtime_command_queue_signal_fence(void *queue, void *fence,
                                               uint64_t value)
{
   return d12_queue_signal_placement_fence(
      d12_queue_from_iface((ID3D12CommandQueue *)queue),
      (struct d12_placement_fence *)fence, value);
}

HRESULT
d12_device_runtime_command_queue_wait_fence(void *queue, void *fence,
                                             uint64_t value)
{
   return d12_queue_wait_placement_fence(
      d12_queue_from_iface((ID3D12CommandQueue *)queue),
      (struct d12_placement_fence *)fence, value);
}

uint32_t
d12_device_runtime_command_queue_context_token(void *queue)
{
   struct d12_queue *native =
      d12_queue_from_iface((ID3D12CommandQueue *)queue);
   uint32_t token = 0;

   if (!native || !agx_d3dkmt_queue_context_token(
                     &native->device->dev, native->agx_queue_id, &token))
      return 0;
   return token;
}

HRESULT
d12_device_runtime_command_allocator_create(
   struct d12_device *device, enum d12_device_runtime_command_type runtime_type,
   void **out_allocator)
{
   D3D12_COMMAND_LIST_TYPE type;

   if (!device || !out_allocator ||
       !d12_device_runtime_command_type(runtime_type, &type))
      return E_INVALIDARG;

   return d12_cmd_allocator_create(device, type,
                                   &IID_ID3D12CommandAllocator,
                                   out_allocator);
}

HRESULT
d12_device_runtime_command_allocator_reset(void *allocator)
{
   if (!allocator)
      return E_INVALIDARG;

   return ID3D12CommandAllocator_Reset((ID3D12CommandAllocator *)allocator);
}

void
d12_device_runtime_command_allocator_destroy(void *allocator)
{
   if (allocator)
      ID3D12CommandAllocator_Release((ID3D12CommandAllocator *)allocator);
}

HRESULT
d12_device_runtime_command_list_create(
   struct d12_device *device, enum d12_device_runtime_command_type runtime_type,
   void *allocator, void **out_list)
{
   D3D12_COMMAND_LIST_TYPE type;
   ID3D12CommandAllocator *allocator_iface = allocator;

   if (!device || !allocator_iface || !out_list ||
       !d12_device_runtime_command_type(runtime_type, &type))
      return E_INVALIDARG;

   return d12_cmd_list_create(device, type,
                              d12_cmd_allocator_from_iface(allocator_iface),
                              NULL, &IID_ID3D12GraphicsCommandList, out_list);
}

HRESULT
d12_device_runtime_command_list_close(void *list)
{
   if (!list)
      return E_INVALIDARG;

   return ID3D12GraphicsCommandList_Close((ID3D12GraphicsCommandList *)list);
}

HRESULT
d12_device_runtime_command_list_reset(void *list, void *allocator)
{
   if (!list || !allocator)
      return E_INVALIDARG;

   return ID3D12GraphicsCommandList_Reset(
      (ID3D12GraphicsCommandList *)list,
      (ID3D12CommandAllocator *)allocator, NULL);
}

void
d12_device_runtime_command_list_destroy(void *list)
{
   if (list)
      ID3D12GraphicsCommandList_Release((ID3D12GraphicsCommandList *)list);
}

void
d12_device_runtime_command_list_execute_bundle(void *list, void *bundle)
{
   if (list && bundle)
      ID3D12GraphicsCommandList_ExecuteBundle(
         (ID3D12GraphicsCommandList *)list,
         (ID3D12GraphicsCommandList *)bundle);
}

void
d12_device_runtime_command_list_clear_root_arguments(void *list_handle)
{
   struct d12_cmd_list *list = list_handle;

   if (!list)
      return;
   memset(list->compute_root_table, 0, sizeof(list->compute_root_table));
   memset(list->graphics_root_table, 0, sizeof(list->graphics_root_table));
   memset(list->compute_root_source, 0, sizeof(list->compute_root_source));
   memset(list->graphics_root_source, 0, sizeof(list->graphics_root_source));
   memset(list->compute_sampler_table, 0,
          sizeof(list->compute_sampler_table));
   memset(list->graphics_sampler_table, 0,
          sizeof(list->graphics_sampler_table));
}

HRESULT
d12_device_runtime_command_signature_create(
   struct d12_device *device,
   const struct d12_frontend_command_signature_desc *desc,
   void **out_signature)
{
   D3D12_INDIRECT_ARGUMENT_DESC arguments[65];
   D3D12_COMMAND_SIGNATURE_DESC translated;

   if (!device || !desc || !desc->argument_count ||
       desc->argument_count > ARRAY_SIZE(arguments) || !desc->arguments ||
       desc->node_mask > 1 || !out_signature)
      return E_INVALIDARG;
   for (uint32_t index = 0; index < desc->argument_count; ++index) {
      const struct d12_frontend_indirect_argument_desc *source =
         &desc->arguments[index];
      if (source->type > D3D12_INDIRECT_ARGUMENT_TYPE_UNORDERED_ACCESS_VIEW)
         return E_NOTIMPL;
      arguments[index] = (D3D12_INDIRECT_ARGUMENT_DESC){
         .Type = (D3D12_INDIRECT_ARGUMENT_TYPE)source->type,
      };
      switch (source->type) {
      case D3D12_INDIRECT_ARGUMENT_TYPE_VERTEX_BUFFER_VIEW:
         arguments[index].VertexBuffer.Slot = source->root_parameter_index;
         break;
      case D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT:
         arguments[index].Constant.RootParameterIndex =
            source->root_parameter_index;
         arguments[index].Constant.DestOffsetIn32BitValues =
            source->destination_offset;
         arguments[index].Constant.Num32BitValuesToSet = source->value_count;
         break;
      case D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT_BUFFER_VIEW:
         arguments[index].ConstantBufferView.RootParameterIndex =
            source->root_parameter_index;
         break;
      case D3D12_INDIRECT_ARGUMENT_TYPE_SHADER_RESOURCE_VIEW:
         arguments[index].ShaderResourceView.RootParameterIndex =
            source->root_parameter_index;
         break;
      case D3D12_INDIRECT_ARGUMENT_TYPE_UNORDERED_ACCESS_VIEW:
         arguments[index].UnorderedAccessView.RootParameterIndex =
            source->root_parameter_index;
         break;
      default:
         break;
      }
   }
   translated = (D3D12_COMMAND_SIGNATURE_DESC){
      .ByteStride = desc->byte_stride,
      .NumArgumentDescs = desc->argument_count,
      .pArgumentDescs = arguments,
      .NodeMask = desc->node_mask,
   };
   return ID3D12Device_CreateCommandSignature(
      &device->iface, &translated,
      (ID3D12RootSignature *)desc->root_signature,
      &IID_ID3D12CommandSignature, out_signature);
}

void
d12_device_runtime_command_signature_destroy(void *signature)
{
   if (signature)
      ID3D12CommandSignature_Release((ID3D12CommandSignature *)signature);
}

void
d12_device_runtime_command_list_execute_indirect(
   void *list, void *signature, uint32_t maximum_count,
   void *argument_buffer, uint64_t argument_offset, void *count_buffer,
   uint64_t count_offset)
{
   if (list && signature && argument_buffer && maximum_count)
      ID3D12GraphicsCommandList_ExecuteIndirect(
         (ID3D12GraphicsCommandList *)list,
         (ID3D12CommandSignature *)signature, maximum_count,
         (ID3D12Resource *)argument_buffer, argument_offset,
         (ID3D12Resource *)count_buffer, count_offset);
}

void
d12_device_runtime_command_list_draw_instanced(
   void *list, uint32_t vertex_count_per_instance, uint32_t instance_count,
   uint32_t start_vertex, uint32_t start_instance)
{
   if (!list)
      return;

   ID3D12GraphicsCommandList_DrawInstanced(
      (ID3D12GraphicsCommandList *)list, vertex_count_per_instance,
      instance_count, start_vertex, start_instance);
}

void
d12_device_runtime_command_list_draw_indexed_instanced(
   void *list, uint32_t index_count_per_instance, uint32_t instance_count,
   uint32_t start_index, int32_t base_vertex, uint32_t start_instance)
{
   if (!list)
      return;

   ID3D12GraphicsCommandList_DrawIndexedInstanced(
      (ID3D12GraphicsCommandList *)list, index_count_per_instance,
      instance_count, start_index, base_vertex, start_instance);
}

void
d12_device_runtime_command_list_dispatch(
   void *list, uint32_t group_count_x, uint32_t group_count_y,
   uint32_t group_count_z)
{
   if (!list)
      return;

   ID3D12GraphicsCommandList_Dispatch((ID3D12GraphicsCommandList *)list,
                                      group_count_x, group_count_y,
                                      group_count_z);
}

void
d12_device_runtime_command_list_set_topology(void *list, uint32_t topology)
{
   if (!list)
      return;

   ID3D12GraphicsCommandList_IASetPrimitiveTopology(
      (ID3D12GraphicsCommandList *)list, (D3D12_PRIMITIVE_TOPOLOGY)topology);
}

void
d12_device_runtime_command_list_set_depth_bias(
   void *list, float depth_bias, float depth_bias_clamp,
   float slope_scaled_depth_bias)
{
   if (!list)
      return;

   ID3D12GraphicsCommandList9_RSSetDepthBias(
      (ID3D12GraphicsCommandList9 *)list, depth_bias, depth_bias_clamp,
      slope_scaled_depth_bias);
}

void
d12_device_runtime_command_list_set_index_buffer_strip_cut(
   void *list, uint32_t strip_cut)
{
   if (!list)
      return;

   ID3D12GraphicsCommandList9_IASetIndexBufferStripCutValue(
      (ID3D12GraphicsCommandList9 *)list,
      (D3D12_INDEX_BUFFER_STRIP_CUT_VALUE)strip_cut);
}

void
d12_device_runtime_command_list_set_viewports(
   void *list, uint32_t count, const float (*viewports)[6])
{
   D3D12_VIEWPORT translated[D3D12_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE];

   if (!list || count > ARRAY_SIZE(translated) || (count && !viewports))
      return;
   for (uint32_t index = 0; index < count; ++index) {
      translated[index] = (D3D12_VIEWPORT){
         .TopLeftX = viewports[index][0],
         .TopLeftY = viewports[index][1],
         .Width = viewports[index][2],
         .Height = viewports[index][3],
         .MinDepth = viewports[index][4],
         .MaxDepth = viewports[index][5],
      };
   }
   ID3D12GraphicsCommandList_RSSetViewports(
      (ID3D12GraphicsCommandList *)list, count, translated);
}

void
d12_device_runtime_command_list_set_scissors(
   void *list, uint32_t count, const int32_t (*rects)[4])
{
   D3D12_RECT translated[D3D12_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE];

   if (!list || count > ARRAY_SIZE(translated) || (count && !rects))
      return;
   for (uint32_t index = 0; index < count; ++index) {
      translated[index] = (D3D12_RECT){
         .left = rects[index][0],
         .top = rects[index][1],
         .right = rects[index][2],
         .bottom = rects[index][3],
      };
   }
   ID3D12GraphicsCommandList_RSSetScissorRects(
      (ID3D12GraphicsCommandList *)list, count, translated);
}

void
d12_device_runtime_command_list_set_blend_factor(
   void *list, const float factor[4])
{
   if (list)
      ID3D12GraphicsCommandList_OMSetBlendFactor(
         (ID3D12GraphicsCommandList *)list, factor);
}

void
d12_device_runtime_command_list_set_stencil_ref(void *list,
                                                 uint32_t reference)
{
   if (list)
      ID3D12GraphicsCommandList_OMSetStencilRef(
         (ID3D12GraphicsCommandList *)list, reference);
}

void
d12_device_runtime_command_list_set_front_and_back_stencil_ref(
   void *list, uint32_t front_reference, uint32_t back_reference)
{
   if (list)
      ID3D12GraphicsCommandList9_OMSetFrontAndBackStencilRef(
         (ID3D12GraphicsCommandList9 *)list, front_reference,
         back_reference);
}

void
d12_device_runtime_command_list_set_vertex_buffers(
   void *list, uint32_t start_slot, uint32_t count,
   const struct d12_frontend_vertex_buffer_view *views)
{
   D3D12_VERTEX_BUFFER_VIEW translated[D12_FRONTEND_MAX_VERTEX_BUFFERS];

   if (!list || count > ARRAY_SIZE(translated) || (count && !views))
      return;
   for (uint32_t index = 0; index < count; ++index) {
      translated[index] = (D3D12_VERTEX_BUFFER_VIEW){
         .BufferLocation = views[index].address,
         .SizeInBytes = views[index].size,
         .StrideInBytes = views[index].stride,
      };
   }
   ID3D12GraphicsCommandList_IASetVertexBuffers(
      (ID3D12GraphicsCommandList *)list, start_slot, count, translated);
}

void
d12_device_runtime_command_list_set_index_buffer(
   void *list, const struct d12_frontend_index_buffer_view *view)
{
   D3D12_INDEX_BUFFER_VIEW translated;

   if (!list)
      return;
   if (!view) {
      ID3D12GraphicsCommandList_IASetIndexBuffer(
         (ID3D12GraphicsCommandList *)list, NULL);
      return;
   }
   translated = (D3D12_INDEX_BUFFER_VIEW){
      .BufferLocation = view->address,
      .SizeInBytes = view->size,
      .Format = (DXGI_FORMAT)view->format,
   };
   ID3D12GraphicsCommandList_IASetIndexBuffer(
      (ID3D12GraphicsCommandList *)list, &translated);
}

void
d12_device_runtime_command_list_set_stream_output_targets(
   void *list, uint32_t start_slot, uint32_t count,
   const struct d12_frontend_stream_output_buffer_view *views)
{
   D3D12_STREAM_OUTPUT_BUFFER_VIEW
      translated[D12_FRONTEND_MAX_STREAM_OUTPUT_BUFFERS] = {0};

   if (!list || start_slot > ARRAY_SIZE(translated) ||
       count > ARRAY_SIZE(translated) - start_slot)
      return;
   for (uint32_t index = 0; views && index < count; ++index) {
      translated[index] = (D3D12_STREAM_OUTPUT_BUFFER_VIEW){
         .BufferLocation = views[index].address,
         .SizeInBytes = views[index].size,
         .BufferFilledSizeLocation = views[index].filled_size_address,
      };
   }
   ID3D12GraphicsCommandList_SOSetTargets(
      (ID3D12GraphicsCommandList *)list, start_slot, count, translated);
}

void
d12_device_runtime_command_list_set_sample_positions(
   void *list, uint32_t samples_per_pixel, uint32_t pixel_count,
   const int8_t *positions)
{
   D3D12_SAMPLE_POSITION local[D12_SAMPLE_POSITION_MAX_SAMPLES];
   const uint32_t count = samples_per_pixel <= ARRAY_SIZE(local)
                             ? samples_per_pixel
                             : 0;

   if (!list)
      return;

   for (uint32_t i = 0; i < count && positions; ++i) {
      local[i].X = positions[2 * i];
      local[i].Y = positions[2 * i + 1];
   }

   ID3D12GraphicsCommandList1_SetSamplePositions(
      (ID3D12GraphicsCommandList1 *)list, samples_per_pixel, pixel_count,
      (count && positions) ? local : NULL);
}

void
d12_device_runtime_command_list_begin_render_pass(
   void *list, uint32_t count,
   const struct d12_frontend_render_pass_rt *render_targets,
   const struct d12_frontend_render_pass_ds *depth_stencil)
{
   D3D12_RENDER_PASS_RENDER_TARGET_DESC rts[D12_FRONTEND_MAX_RENDER_TARGETS];
   D3D12_RENDER_PASS_DEPTH_STENCIL_DESC ds;

   if (!list || count > ARRAY_SIZE(rts) || (count && !render_targets))
      return;

   memset(rts, 0, sizeof(rts));
   for (uint32_t i = 0; i < count; ++i) {
      rts[i].cpuDescriptor.ptr = (SIZE_T)render_targets[i].cpu_descriptor;
      rts[i].BeginningAccess.Type =
         (D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE)
            render_targets[i].beginning_access;
      rts[i].EndingAccess.Type = (D3D12_RENDER_PASS_ENDING_ACCESS_TYPE)
                                    render_targets[i].ending_access;
      memcpy(rts[i].BeginningAccess.Clear.ClearValue.Color,
             render_targets[i].clear_color,
             sizeof(rts[i].BeginningAccess.Clear.ClearValue.Color));
   }

   if (depth_stencil) {
      memset(&ds, 0, sizeof(ds));
      ds.cpuDescriptor.ptr = (SIZE_T)depth_stencil->cpu_descriptor;
      ds.DepthBeginningAccess.Type =
         (D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE)
            depth_stencil->depth_beginning_access;
      ds.DepthEndingAccess.Type = (D3D12_RENDER_PASS_ENDING_ACCESS_TYPE)
                                     depth_stencil->depth_ending_access;
      ds.StencilBeginningAccess.Type =
         (D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE)
            depth_stencil->stencil_beginning_access;
      ds.StencilEndingAccess.Type = (D3D12_RENDER_PASS_ENDING_ACCESS_TYPE)
                                       depth_stencil->stencil_ending_access;
      ds.DepthBeginningAccess.Clear.ClearValue.DepthStencil.Depth =
         depth_stencil->clear_depth;
      ds.StencilBeginningAccess.Clear.ClearValue.DepthStencil.Stencil =
         (UINT8)depth_stencil->clear_stencil;
   }

   ID3D12GraphicsCommandList4_BeginRenderPass(
      (ID3D12GraphicsCommandList4 *)list, count, count ? rts : NULL,
      depth_stencil ? &ds : NULL, D3D12_RENDER_PASS_FLAG_NONE);
}

void
d12_device_runtime_command_list_end_render_pass(void *list)
{
   if (!list)
      return;
   ID3D12GraphicsCommandList4_EndRenderPass(
      (ID3D12GraphicsCommandList4 *)list);
}

void
d12_device_runtime_command_list_set_render_targets(
   void *list, uint32_t count, const uint64_t *render_targets,
   bool contiguous, const uint64_t *depth_stencil)
{
   D3D12_CPU_DESCRIPTOR_HANDLE translated[D12_FRONTEND_MAX_RENDER_TARGETS];
   D3D12_CPU_DESCRIPTOR_HANDLE translated_depth;

   if (!list || count > ARRAY_SIZE(translated) ||
       (count && !render_targets))
      return;
   for (uint32_t index = 0; index < count; ++index)
      translated[index].ptr = (SIZE_T)render_targets[index];
   if (depth_stencil)
      translated_depth.ptr = (SIZE_T)*depth_stencil;
   ID3D12GraphicsCommandList_OMSetRenderTargets(
      (ID3D12GraphicsCommandList *)list, count, translated, contiguous,
      depth_stencil ? &translated_depth : NULL);
}

void
d12_device_runtime_command_list_resource_barriers(
   void *list, uint32_t count,
   const struct d12_frontend_resource_barrier *barriers)
{
   D3D12_RESOURCE_BARRIER local[16];

   if (!list || (count && !barriers))
      return;

   for (uint32_t index = 0; index < count; ++index) {
      if (barriers[index].type != D12_FRONTEND_BARRIER_TRANSITION &&
          barriers[index].type != D12_FRONTEND_BARRIER_ALIASING &&
          barriers[index].type != D12_FRONTEND_BARRIER_UAV)
         return;
   }

   for (uint32_t offset = 0; offset < count;) {
      const uint32_t chunk = MIN2(count - offset, ARRAY_SIZE(local));

      for (uint32_t index = 0; index < chunk; ++index) {
         const struct d12_frontend_resource_barrier *source =
            &barriers[offset + index];

         local[index].Type = (D3D12_RESOURCE_BARRIER_TYPE)source->type;
         local[index].Flags = (D3D12_RESOURCE_BARRIER_FLAGS)source->flags;
         switch (source->type) {
         case D12_FRONTEND_BARRIER_TRANSITION:
            local[index].Transition = (D3D12_RESOURCE_TRANSITION_BARRIER){
               .pResource = source->transition.resource,
               .Subresource = source->transition.subresource,
               .StateBefore =
                  (D3D12_RESOURCE_STATES)source->transition.state_before,
               .StateAfter =
                  (D3D12_RESOURCE_STATES)source->transition.state_after,
            };
            break;
         case D12_FRONTEND_BARRIER_ALIASING:
            local[index].Aliasing = (D3D12_RESOURCE_ALIASING_BARRIER){
               .pResourceBefore = source->aliasing.resource_before,
               .pResourceAfter = source->aliasing.resource_after,
            };
            break;
         case D12_FRONTEND_BARRIER_UAV:
            local[index].UAV = (D3D12_RESOURCE_UAV_BARRIER){
               .pResource = source->uav.resource,
            };
            break;
         default:
            return;
         }
      }

      ID3D12GraphicsCommandList_ResourceBarrier(
         (ID3D12GraphicsCommandList *)list, chunk, local);
      offset += chunk;
   }
}

void
d12_device_runtime_command_list_copy_buffer_region(
   void *list, void *destination, uint64_t destination_offset, void *source,
   uint64_t source_offset, uint64_t bytes)
{
   if (list && destination && source)
      ID3D12GraphicsCommandList_CopyBufferRegion(
         (ID3D12GraphicsCommandList *)list, destination, destination_offset,
         source, source_offset, bytes);
}

void
d12_device_runtime_command_list_copy_resource(void *list, void *destination,
                                               void *source)
{
   if (list && destination && source)
      ID3D12GraphicsCommandList_CopyResource(
         (ID3D12GraphicsCommandList *)list, destination, source);
}

HRESULT
d12_device_runtime_command_list_copy_tiles(
   void *list, void *resource,
   const struct d12_frontend_tiled_resource_coordinate *region_start,
   const struct d12_frontend_tile_region_size *region_size, void *buffer,
   uint64_t buffer_offset, uint32_t flags)
{
   if (!list || !resource || !region_start || !region_size || !buffer ||
       (flags & ~(D12_FRONTEND_TILE_COPY_NO_HAZARD |
                  D12_FRONTEND_TILE_COPY_LINEAR_TO_TILED |
                  D12_FRONTEND_TILE_COPY_TILED_TO_LINEAR)) ||
       ((flags & D12_FRONTEND_TILE_COPY_LINEAR_TO_TILED) &&
        (flags & D12_FRONTEND_TILE_COPY_TILED_TO_LINEAR)))
      return E_INVALIDARG;
   const D3D12_TILED_RESOURCE_COORDINATE start = {
      region_start->x, region_start->y, region_start->z,
      region_start->subresource,
   };
   const D3D12_TILE_REGION_SIZE size = {
      region_size->num_tiles, region_size->use_box, region_size->width,
      region_size->height, region_size->depth,
   };
   ID3D12GraphicsCommandList_CopyTiles(
      (ID3D12GraphicsCommandList *)list, resource, &start, &size, buffer,
      buffer_offset, (D3D12_TILE_COPY_FLAGS)flags);

   struct d12_cmd_list *native =
      d12_cmd_list_from_iface((ID3D12GraphicsCommandList6 *)list);
   return native && native->state == D12_LIST_ERROR
             ? (native->record_error ? native->record_error : E_FAIL)
             : S_OK;
}

static bool
d12_device_runtime_texture_copy_location(
   const struct d12_frontend_texture_copy_location *source,
   D3D12_TEXTURE_COPY_LOCATION *destination)
{
   if (!source || !source->resource || !destination)
      return false;
   memset(destination, 0, sizeof(*destination));
   destination->pResource = source->resource;
   switch (source->type) {
   case D12_FRONTEND_TEXTURE_COPY_SUBRESOURCE:
      destination->Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
      destination->SubresourceIndex = source->subresource;
      return true;
   case D12_FRONTEND_TEXTURE_COPY_FOOTPRINT:
      destination->Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
      destination->PlacedFootprint.Offset = source->footprint.offset;
      destination->PlacedFootprint.Footprint = (D3D12_SUBRESOURCE_FOOTPRINT){
         .Format = (DXGI_FORMAT)source->footprint.format,
         .Width = source->footprint.width,
         .Height = source->footprint.height,
         .Depth = source->footprint.depth,
         .RowPitch = source->footprint.row_pitch,
      };
      return true;
   default:
      return false;
   }
}

void
d12_device_runtime_command_list_copy_texture_region(
   void *list,
   const struct d12_frontend_texture_copy_location *destination,
   uint32_t destination_x, uint32_t destination_y, uint32_t destination_z,
   const struct d12_frontend_texture_copy_location *source,
   const struct d12_frontend_box *source_box)
{
   D3D12_TEXTURE_COPY_LOCATION translated_destination;
   D3D12_TEXTURE_COPY_LOCATION translated_source;
   D3D12_BOX translated_box;

   if (!list ||
       !d12_device_runtime_texture_copy_location(destination,
                                                  &translated_destination) ||
       !d12_device_runtime_texture_copy_location(source, &translated_source))
      return;
   if (source_box) {
      translated_box = (D3D12_BOX){
         .left = (UINT)source_box->left,
         .top = (UINT)source_box->top,
         .front = (UINT)source_box->front,
         .right = (UINT)source_box->right,
         .bottom = (UINT)source_box->bottom,
         .back = (UINT)source_box->back,
      };
   }
   ID3D12GraphicsCommandList_CopyTextureRegion(
      (ID3D12GraphicsCommandList *)list, &translated_destination,
      destination_x, destination_y, destination_z, &translated_source,
      source_box ? &translated_box : NULL);
}

void
d12_device_runtime_command_list_resolve_subresource(
   void *list, void *destination, uint32_t destination_subresource,
   void *source, uint32_t source_subresource, uint32_t format)
{
   if (list && destination && source)
      ID3D12GraphicsCommandList_ResolveSubresource(
         (ID3D12GraphicsCommandList *)list, destination,
         destination_subresource, source, source_subresource,
         (DXGI_FORMAT)format);
}

HRESULT
d12_device_runtime_command_list_resolve_subresource_region(
   void *list, void *destination, uint32_t destination_subresource,
   uint32_t destination_x, uint32_t destination_y, void *source,
   uint32_t source_subresource, const struct d12_frontend_rect *source_rect,
   uint32_t format, uint32_t resolve_mode)
{
   D3D12_RECT translated_rect;
   struct d12_cmd_list *native;

   if (!list || !destination || !source)
      return E_INVALIDARG;
   if (source_rect) {
      translated_rect = (D3D12_RECT){
         .left = source_rect->left,
         .top = source_rect->top,
         .right = source_rect->right,
         .bottom = source_rect->bottom,
      };
   }
   ID3D12GraphicsCommandList1_ResolveSubresourceRegion(
      (ID3D12GraphicsCommandList1 *)list, destination,
      destination_subresource, destination_x, destination_y, source,
      source_subresource, source_rect ? &translated_rect : NULL,
      (DXGI_FORMAT)format, (D3D12_RESOLVE_MODE)resolve_mode);

   native = d12_cmd_list_from_iface((ID3D12GraphicsCommandList6 *)list);
   return native && native->state == D12_LIST_ERROR
             ? (native->record_error ? native->record_error : E_FAIL)
             : S_OK;
}

HRESULT
d12_device_runtime_command_list_write_buffer_immediate(
   void *list, uint32_t count,
   const struct d12_frontend_write_buffer_immediate_parameter *parameters,
   const uint32_t *modes)
{
   D3D12_WRITEBUFFERIMMEDIATE_PARAMETER local_parameters[32];
   D3D12_WRITEBUFFERIMMEDIATE_MODE local_modes[32];
   struct d12_cmd_list *native;

   if (!list || (count && !parameters))
      return E_INVALIDARG;

   native = d12_cmd_list_from_iface((ID3D12GraphicsCommandList6 *)list);
   for (uint32_t offset = 0; offset < count;) {
      const uint32_t chunk = MIN2(count - offset,
                                  ARRAY_SIZE(local_parameters));

      for (uint32_t index = 0; index < chunk; ++index) {
         local_parameters[index] =
            (D3D12_WRITEBUFFERIMMEDIATE_PARAMETER){
               .Dest = parameters[offset + index].destination,
               .Value = parameters[offset + index].value,
            };
         local_modes[index] = modes
            ? (D3D12_WRITEBUFFERIMMEDIATE_MODE)modes[offset + index]
            : D3D12_WRITEBUFFERIMMEDIATE_MODE_DEFAULT;
      }

      ID3D12GraphicsCommandList2_WriteBufferImmediate(
         (ID3D12GraphicsCommandList2 *)list, chunk, local_parameters,
         modes ? local_modes : NULL);
      if (native && native->state == D12_LIST_ERROR)
         return native->record_error ? native->record_error : E_FAIL;
      offset += chunk;
   }

   return native && native->state == D12_LIST_ERROR
             ? (native->record_error ? native->record_error : E_FAIL)
             : S_OK;
}

HRESULT
d12_device_runtime_command_list_present_blt(
   void *list, void *destination, void *source,
   const struct d12_frontend_present_blt_desc *desc)
{
   if (!list || !destination || !source || !desc)
      return E_INVALIDARG;

   return d12_cmd_present_blt(
      d12_cmd_list_from_iface(list),
      d12_resource_from_iface((ID3D12Resource *)destination),
      d12_resource_from_iface((ID3D12Resource *)source), desc);
}

_Static_assert(sizeof(struct d12_frontend_rect) == sizeof(D3D12_RECT),
               "frontend rectangle must match D3D12_RECT");

void
d12_device_runtime_command_list_clear_render_target(
   void *list, uint64_t descriptor, const float color[4], uint32_t count,
   const struct d12_frontend_rect *rects)
{
   if (list && descriptor && color && (!count || rects))
      ID3D12GraphicsCommandList_ClearRenderTargetView(
         (ID3D12GraphicsCommandList *)list,
         (D3D12_CPU_DESCRIPTOR_HANDLE){.ptr = (SIZE_T)descriptor}, color,
         count, (const D3D12_RECT *)rects);
}

void
d12_device_runtime_command_list_clear_depth_stencil(
   void *list, uint64_t descriptor, uint32_t flags, float depth,
   uint8_t stencil, uint32_t count, const struct d12_frontend_rect *rects)
{
   if (list && descriptor && (!count || rects))
      ID3D12GraphicsCommandList_ClearDepthStencilView(
         (ID3D12GraphicsCommandList *)list,
         (D3D12_CPU_DESCRIPTOR_HANDLE){.ptr = (SIZE_T)descriptor},
         (D3D12_CLEAR_FLAGS)flags, depth, stencil, count,
         (const D3D12_RECT *)rects);
}

void
d12_device_runtime_command_list_clear_uav_uint(
   void *list, uint64_t gpu_descriptor, uint64_t cpu_descriptor,
   void *resource, const uint32_t values[4], uint32_t count,
   const struct d12_frontend_rect *rects)
{
   if (list && gpu_descriptor && cpu_descriptor && resource && values &&
       (!count || rects))
      ID3D12GraphicsCommandList_ClearUnorderedAccessViewUint(
         (ID3D12GraphicsCommandList *)list,
         (D3D12_GPU_DESCRIPTOR_HANDLE){.ptr = gpu_descriptor},
         (D3D12_CPU_DESCRIPTOR_HANDLE){.ptr = (SIZE_T)cpu_descriptor},
         resource, values, count, (const D3D12_RECT *)rects);
}

void
d12_device_runtime_command_list_clear_uav_float(
   void *list, uint64_t gpu_descriptor, uint64_t cpu_descriptor,
   void *resource, const float values[4], uint32_t count,
   const struct d12_frontend_rect *rects)
{
   if (list && gpu_descriptor && cpu_descriptor && resource && values &&
       (!count || rects))
      ID3D12GraphicsCommandList_ClearUnorderedAccessViewFloat(
         (ID3D12GraphicsCommandList *)list,
         (D3D12_GPU_DESCRIPTOR_HANDLE){.ptr = gpu_descriptor},
         (D3D12_CPU_DESCRIPTOR_HANDLE){.ptr = (SIZE_T)cpu_descriptor},
         resource, values, count, (const D3D12_RECT *)rects);
}

HRESULT
d12_device_runtime_root_signature_create(
   struct d12_device *device,
   const struct d12_frontend_root_signature_desc *frontend,
   void **out_root_signature)
{
   D3D12_ROOT_PARAMETER1 *parameters = NULL;
   D3D12_DESCRIPTOR_RANGE1 *ranges = NULL;
   D3D12_STATIC_SAMPLER_DESC *samplers = NULL;
   size_t range_count = 0;
   size_t next_range = 0;
   void *blob = NULL;
   size_t blob_size = 0;
   HRESULT result;

   if (!device || !frontend || !out_root_signature ||
       (frontend->num_parameters && !frontend->parameters) ||
       (frontend->num_static_samplers && !frontend->static_samplers))
      return E_INVALIDARG;
   *out_root_signature = NULL;

   for (uint32_t index = 0; index < frontend->num_parameters; ++index) {
      const struct d12_frontend_root_parameter *parameter =
         &frontend->parameters[index];
      if (parameter->parameter_type !=
          D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE)
         continue;
      if (parameter->table.num_ranges && !parameter->table.ranges)
         return E_INVALIDARG;
      if (SIZE_MAX - range_count < parameter->table.num_ranges)
         return E_OUTOFMEMORY;
      range_count += parameter->table.num_ranges;
   }

   parameters = calloc(frontend->num_parameters, sizeof(*parameters));
   ranges = calloc(range_count, sizeof(*ranges));
   samplers = calloc(frontend->num_static_samplers, sizeof(*samplers));
   if ((frontend->num_parameters && !parameters) ||
       (range_count && !ranges) ||
       (frontend->num_static_samplers && !samplers)) {
      result = E_OUTOFMEMORY;
      goto out;
   }

   for (uint32_t index = 0; index < frontend->num_parameters; ++index) {
      const struct d12_frontend_root_parameter *source =
         &frontend->parameters[index];
      D3D12_ROOT_PARAMETER1 *destination = &parameters[index];

      destination->ParameterType =
         (D3D12_ROOT_PARAMETER_TYPE)source->parameter_type;
      destination->ShaderVisibility =
         (D3D12_SHADER_VISIBILITY)source->shader_visibility;
      switch (destination->ParameterType) {
      case D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE:
         destination->DescriptorTable.NumDescriptorRanges =
            source->table.num_ranges;
         destination->DescriptorTable.pDescriptorRanges =
            source->table.num_ranges ? &ranges[next_range] : NULL;
         for (uint32_t range_index = 0;
              range_index < source->table.num_ranges; ++range_index) {
            const struct d12_frontend_descriptor_range *source_range =
               &source->table.ranges[range_index];
            ranges[next_range++] = (D3D12_DESCRIPTOR_RANGE1){
               .RangeType =
                  (D3D12_DESCRIPTOR_RANGE_TYPE)source_range->range_type,
               .NumDescriptors = source_range->num_descriptors,
               .BaseShaderRegister = source_range->base_shader_register,
               .RegisterSpace = source_range->register_space,
               .Flags =
                  (D3D12_DESCRIPTOR_RANGE_FLAGS)source_range->flags,
               .OffsetInDescriptorsFromTableStart =
                  source_range->offset_in_descriptors,
            };
         }
         break;
      case D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS:
         destination->Constants = (D3D12_ROOT_CONSTANTS){
            .ShaderRegister = source->constants.shader_register,
            .RegisterSpace = source->constants.register_space,
            .Num32BitValues = source->constants.num_32bit_values,
         };
         break;
      case D3D12_ROOT_PARAMETER_TYPE_CBV:
      case D3D12_ROOT_PARAMETER_TYPE_SRV:
      case D3D12_ROOT_PARAMETER_TYPE_UAV:
         destination->Descriptor = (D3D12_ROOT_DESCRIPTOR1){
            .ShaderRegister = source->descriptor.shader_register,
            .RegisterSpace = source->descriptor.register_space,
            .Flags = (D3D12_ROOT_DESCRIPTOR_FLAGS)source->descriptor.flags,
         };
         break;
      default:
         result = E_INVALIDARG;
         goto out;
      }
   }

   for (uint32_t index = 0; index < frontend->num_static_samplers; ++index) {
      const struct d12_frontend_static_sampler *source =
         &frontend->static_samplers[index];
      samplers[index] = (D3D12_STATIC_SAMPLER_DESC){
         .Filter = (D3D12_FILTER)source->filter,
         .AddressU = (D3D12_TEXTURE_ADDRESS_MODE)source->address_u,
         .AddressV = (D3D12_TEXTURE_ADDRESS_MODE)source->address_v,
         .AddressW = (D3D12_TEXTURE_ADDRESS_MODE)source->address_w,
         .MipLODBias = source->mip_lod_bias,
         .MaxAnisotropy = source->max_anisotropy,
         .ComparisonFunc = (D3D12_COMPARISON_FUNC)source->comparison_func,
         .BorderColor = (D3D12_STATIC_BORDER_COLOR)source->border_color,
         .MinLOD = source->min_lod,
         .MaxLOD = source->max_lod,
         .ShaderRegister = source->shader_register,
         .RegisterSpace = source->register_space,
         .ShaderVisibility =
            (D3D12_SHADER_VISIBILITY)source->shader_visibility,
      };
   }

   const D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc = {
      .Version = D3D_ROOT_SIGNATURE_VERSION_1_1,
      .Desc_1_1 = {
         .NumParameters = frontend->num_parameters,
         .pParameters = parameters,
         .NumStaticSamplers = frontend->num_static_samplers,
         .pStaticSamplers = samplers,
         .Flags = (D3D12_ROOT_SIGNATURE_FLAGS)frontend->flags,
      },
   };
   result = d12_serialize_versioned_root_signature(&desc, &blob, &blob_size);
   if (SUCCEEDED(result))
      result = d12_root_signature_create(
         device, blob, blob_size, &IID_ID3D12RootSignature,
         out_root_signature);

out:
   free(blob);
   free(samplers);
   free(ranges);
   free(parameters);
   return result;
}

void
d12_device_runtime_root_signature_destroy(void *root_signature)
{
   if (root_signature)
      ID3D12RootSignature_Release((ID3D12RootSignature *)root_signature);
}

void
d12_device_runtime_command_list_set_root_signature(
   void *list, enum d12_frontend_root_pipeline pipeline,
   void *root_signature)
{
   ID3D12GraphicsCommandList *command_list = list;
   ID3D12RootSignature *signature = root_signature;

   if (!command_list)
      return;
   if (pipeline == D12_FRONTEND_ROOT_GRAPHICS)
      ID3D12GraphicsCommandList_SetGraphicsRootSignature(command_list,
                                                          signature);
   else if (pipeline == D12_FRONTEND_ROOT_COMPUTE)
      ID3D12GraphicsCommandList_SetComputeRootSignature(command_list,
                                                         signature);
}

void
d12_device_runtime_command_list_set_root_descriptor_table(
   void *list, enum d12_frontend_root_pipeline pipeline,
   uint32_t parameter_index, uint64_t base_descriptor)
{
   ID3D12GraphicsCommandList *command_list = list;
   const D3D12_GPU_DESCRIPTOR_HANDLE descriptor = {.ptr = base_descriptor};

   if (!command_list)
      return;
   if (pipeline == D12_FRONTEND_ROOT_GRAPHICS)
      ID3D12GraphicsCommandList_SetGraphicsRootDescriptorTable(
         command_list, parameter_index, descriptor);
   else if (pipeline == D12_FRONTEND_ROOT_COMPUTE)
      ID3D12GraphicsCommandList_SetComputeRootDescriptorTable(
         command_list, parameter_index, descriptor);
}

void
d12_device_runtime_command_list_set_root_constants(
   void *list, enum d12_frontend_root_pipeline pipeline,
   uint32_t parameter_index, uint32_t count, const void *data,
   uint32_t destination_offset)
{
   ID3D12GraphicsCommandList *command_list = list;

   if (!command_list || (count && !data))
      return;
   if (pipeline == D12_FRONTEND_ROOT_GRAPHICS)
      ID3D12GraphicsCommandList_SetGraphicsRoot32BitConstants(
         command_list, parameter_index, count, data, destination_offset);
   else if (pipeline == D12_FRONTEND_ROOT_COMPUTE)
      ID3D12GraphicsCommandList_SetComputeRoot32BitConstants(
         command_list, parameter_index, count, data, destination_offset);
}

void
d12_device_runtime_command_list_set_root_buffer_view(
   void *list, enum d12_frontend_root_pipeline pipeline,
   enum d12_frontend_root_buffer_type buffer_type,
   uint32_t parameter_index, uint64_t address)
{
   ID3D12GraphicsCommandList *command_list = list;

   if (!command_list)
      return;
   if (pipeline == D12_FRONTEND_ROOT_GRAPHICS) {
      if (buffer_type == D12_FRONTEND_ROOT_CBV)
         ID3D12GraphicsCommandList_SetGraphicsRootConstantBufferView(
            command_list, parameter_index, address);
      else if (buffer_type == D12_FRONTEND_ROOT_SRV)
         ID3D12GraphicsCommandList_SetGraphicsRootShaderResourceView(
            command_list, parameter_index, address);
      else if (buffer_type == D12_FRONTEND_ROOT_UAV)
         ID3D12GraphicsCommandList_SetGraphicsRootUnorderedAccessView(
            command_list, parameter_index, address);
   } else if (pipeline == D12_FRONTEND_ROOT_COMPUTE) {
      if (buffer_type == D12_FRONTEND_ROOT_CBV)
         ID3D12GraphicsCommandList_SetComputeRootConstantBufferView(
            command_list, parameter_index, address);
      else if (buffer_type == D12_FRONTEND_ROOT_SRV)
         ID3D12GraphicsCommandList_SetComputeRootShaderResourceView(
            command_list, parameter_index, address);
      else if (buffer_type == D12_FRONTEND_ROOT_UAV)
         ID3D12GraphicsCommandList_SetComputeRootUnorderedAccessView(
            command_list, parameter_index, address);
   }
}

HRESULT
d12_device_runtime_pipeline_state_create_compute(
   struct d12_device *device,
   const struct d12_frontend_compute_pipeline_desc *frontend,
   void **out_pipeline_state)
{
   const struct d12_binding_map empty_map = {0};
   const struct d12_binding_map *map = &empty_map;

   if (!device || !frontend || !out_pipeline_state ||
       !frontend->shader.data || !frontend->shader.size)
      return E_INVALIDARG;
   if (frontend->root_signature)
      map = &((struct d12_root_signature *)frontend->root_signature)->map;
   return d12_pipeline_state_create_compute(
      device, frontend->shader.data, frontend->shader.size, map,
      &IID_ID3D12PipelineState, out_pipeline_state);
}

static D3D12_RENDER_TARGET_BLEND_DESC
d12_device_runtime_render_target_blend(
   const struct d12_frontend_render_target_blend *source)
{
   return (D3D12_RENDER_TARGET_BLEND_DESC){
      .BlendEnable = source->blend_enable != 0,
      .LogicOpEnable = source->logic_op_enable != 0,
      .SrcBlend = (D3D12_BLEND)source->src_blend,
      .DestBlend = (D3D12_BLEND)source->dest_blend,
      .BlendOp = (D3D12_BLEND_OP)source->blend_op,
      .SrcBlendAlpha = (D3D12_BLEND)source->src_blend_alpha,
      .DestBlendAlpha = (D3D12_BLEND)source->dest_blend_alpha,
      .BlendOpAlpha = (D3D12_BLEND_OP)source->blend_op_alpha,
      .LogicOp = (D3D12_LOGIC_OP)source->logic_op,
      .RenderTargetWriteMask = source->write_mask,
   };
}

static D3D12_DEPTH_STENCILOP_DESC
d12_device_runtime_depth_stencil_op(
   const struct d12_frontend_depth_stencil_op *source)
{
   return (D3D12_DEPTH_STENCILOP_DESC){
      .StencilFailOp = (D3D12_STENCIL_OP)source->stencil_fail_op,
      .StencilDepthFailOp =
         (D3D12_STENCIL_OP)source->stencil_depth_fail_op,
      .StencilPassOp = (D3D12_STENCIL_OP)source->stencil_pass_op,
      .StencilFunc = (D3D12_COMPARISON_FUNC)source->stencil_func,
   };
}

HRESULT
d12_device_runtime_pipeline_state_create_graphics(
   struct d12_device *device,
   const struct d12_frontend_graphics_pipeline_desc *frontend,
   void **out_pipeline_state)
{
   D3D12_INPUT_ELEMENT_DESC input_elements[D12_FRONTEND_MAX_INPUT_ELEMENTS];
   D3D12_SO_DECLARATION_ENTRY
      so_declarations[D12_FRONTEND_MAX_STREAM_OUTPUT_DECLARATIONS];
   D3D12_BLEND_DESC blend = {
      .AlphaToCoverageEnable =
         frontend ? frontend->blend.alpha_to_coverage_enable != 0 : false,
      .IndependentBlendEnable =
         frontend ? frontend->blend.independent_blend_enable != 0 : false,
   };
   D3D12_RASTERIZER_DESC rasterizer;
   D3D12_DEPTH_STENCIL_DESC depth_stencil;
   D3D12_INPUT_LAYOUT_DESC input_layout;
   D3D12_STREAM_OUTPUT_DESC stream_output = {0};
   DXGI_FORMAT rtv_formats[D12_FRONTEND_MAX_RENDER_TARGETS];
   enum pipe_format rt_format = PIPE_FORMAT_NONE;
   const struct d12_binding_map empty_map = {0};
   const struct d12_binding_map *map = &empty_map;

   if (!device || !frontend || !out_pipeline_state ||
       !frontend->vertex_shader.data || !frontend->vertex_shader.size ||
       frontend->num_render_targets > D12_FRONTEND_MAX_RENDER_TARGETS ||
       frontend->element_layout.count > D12_FRONTEND_MAX_INPUT_ELEMENTS ||
       (frontend->element_layout.count && !frontend->element_layout.elements) ||
       frontend->stream_output.declaration_count >
          D12_FRONTEND_MAX_STREAM_OUTPUT_DECLARATIONS ||
       frontend->stream_output.stride_count >
          D12_FRONTEND_MAX_STREAM_OUTPUT_BUFFERS ||
       (frontend->stream_output.declaration_count &&
        !frontend->stream_output.declarations) ||
       (frontend->stream_output.stride_count &&
        !frontend->stream_output.strides))
      return E_INVALIDARG;

   for (uint32_t index = 0; index < D12_FRONTEND_MAX_RENDER_TARGETS; ++index) {
      blend.RenderTarget[index] = d12_device_runtime_render_target_blend(
         &frontend->blend.render_targets[index]);
      rtv_formats[index] = (DXGI_FORMAT)frontend->rtv_formats[index];
   }
   for (uint32_t index = 0; index < frontend->element_layout.count; ++index) {
      const struct d12_frontend_input_element *source =
         &frontend->element_layout.elements[index];
      /* DDI 0003 already resolves application semantics to shader registers.
       * The wrapped shader uses ATTR<register>, preserving the existing
       * semantic resolver without adding a second vertex-fetch path. */
      input_elements[index] = (D3D12_INPUT_ELEMENT_DESC){
         .SemanticName = "ATTR",
         .SemanticIndex = source->input_register,
         .Format = (DXGI_FORMAT)source->format,
         .InputSlot = source->input_slot,
         .AlignedByteOffset = source->aligned_byte_offset,
         .InputSlotClass =
            (D3D12_INPUT_CLASSIFICATION)source->input_slot_class,
         .InstanceDataStepRate = source->instance_data_step_rate,
      };
   }
   for (uint32_t index = 0;
        index < frontend->stream_output.declaration_count; ++index) {
      const struct d12_frontend_stream_output_declaration *source =
         &frontend->stream_output.declarations[index];
      so_declarations[index] = (D3D12_SO_DECLARATION_ENTRY){
         .Stream = source->stream,
         .SemanticName = source->register_index == UINT32_MAX
                            ? NULL
                            : d12_dxbc_signature_semantic(
                                 source->system_value),
         .SemanticIndex = source->register_index == UINT32_MAX
                             ? 0
                             : source->register_index,
         .StartComponent = (BYTE)source->start_component,
         .ComponentCount = (BYTE)source->component_count,
         .OutputSlot = (BYTE)source->output_slot,
      };
   }
   if (frontend->stream_output.declaration_count) {
      stream_output = (D3D12_STREAM_OUTPUT_DESC){
         .pSODeclaration = so_declarations,
         .NumEntries = frontend->stream_output.declaration_count,
         .pBufferStrides = frontend->stream_output.strides,
         .NumStrides = frontend->stream_output.stride_count,
         .RasterizedStream = frontend->stream_output.rasterized_stream,
      };
   }

   rasterizer = (D3D12_RASTERIZER_DESC){
      .FillMode = (D3D12_FILL_MODE)frontend->rasterizer.fill_mode,
      .CullMode = (D3D12_CULL_MODE)frontend->rasterizer.cull_mode,
      .FrontCounterClockwise =
         frontend->rasterizer.front_counter_clockwise != 0,
      .DepthBias = frontend->rasterizer.depth_bias,
      .DepthBiasClamp = frontend->rasterizer.depth_bias_clamp,
      .SlopeScaledDepthBias = frontend->rasterizer.slope_scaled_depth_bias,
      .DepthClipEnable = frontend->rasterizer.depth_clip_enable != 0,
      .MultisampleEnable = frontend->rasterizer.multisample_enable != 0,
      .AntialiasedLineEnable =
         frontend->rasterizer.antialiased_line_enable != 0,
      .ForcedSampleCount = frontend->rasterizer.forced_sample_count,
      .ConservativeRaster = (D3D12_CONSERVATIVE_RASTERIZATION_MODE)
         frontend->rasterizer.conservative_rasterization_mode,
   };
   depth_stencil = (D3D12_DEPTH_STENCIL_DESC){
      .DepthEnable = frontend->depth_stencil.depth_enable != 0,
      .DepthWriteMask =
         (D3D12_DEPTH_WRITE_MASK)frontend->depth_stencil.depth_write_mask,
      .DepthFunc =
         (D3D12_COMPARISON_FUNC)frontend->depth_stencil.depth_func,
      .StencilEnable = frontend->depth_stencil.stencil_enable != 0,
      .StencilReadMask =
         frontend->depth_stencil.front_face_stencil_read_mask,
      .StencilWriteMask =
         frontend->depth_stencil.front_face_stencil_write_mask,
      .FrontFace = d12_device_runtime_depth_stencil_op(
         &frontend->depth_stencil.front_face),
      .BackFace = d12_device_runtime_depth_stencil_op(
         &frontend->depth_stencil.back_face),
   };
   input_layout = (D3D12_INPUT_LAYOUT_DESC){
      .pInputElementDescs = input_elements,
      .NumElements = frontend->element_layout.count,
   };

   if (frontend->num_render_targets &&
       d12_format_is_renderable(rtv_formats[0]))
      rt_format = d12_format_to_pipe(rtv_formats[0]);
   if (frontend->root_signature)
      map = &((struct d12_root_signature *)frontend->root_signature)->map;

   return d12_pipeline_state_create_graphics(
      device, frontend->vertex_shader.data, frontend->vertex_shader.size, false,
      frontend->pixel_shader.data, frontend->pixel_shader.size,
      frontend->geometry_shader.data, frontend->geometry_shader.size,
      frontend->hull_shader.data, frontend->hull_shader.size,
      frontend->domain_shader.data, frontend->domain_shader.size, rt_format,
      frontend->num_render_targets, rtv_formats, frontend->sample_count,
      frontend->sample_mask, &depth_stencil, (DXGI_FORMAT)frontend->dsv_format,
      &rasterizer, frontend->rasterizer.depth_bias,
      frontend->depth_stencil.front_enable != 0,
      frontend->depth_stencil.back_enable != 0,
      frontend->depth_stencil.front_face_stencil_read_mask,
      frontend->depth_stencil.front_face_stencil_write_mask,
      frontend->depth_stencil.back_face_stencil_read_mask,
      frontend->depth_stencil.back_face_stencil_write_mask, &blend,
      &input_layout,
      frontend->stream_output.declaration_count ? &stream_output : NULL,
      (D3D12_PRIMITIVE_TOPOLOGY_TYPE)frontend->primitive_topology_type,
      (D3D12_INDEX_BUFFER_STRIP_CUT_VALUE)frontend->index_buffer_strip_cut_value,
      (D3D12_PIPELINE_STATE_FLAGS)frontend->pipeline_flags, map,
      &IID_ID3D12PipelineState, out_pipeline_state);
}

void
d12_device_runtime_pipeline_state_destroy(void *pipeline_state)
{
   if (pipeline_state)
      ID3D12PipelineState_Release((ID3D12PipelineState *)pipeline_state);
}

void
d12_device_runtime_pipeline_state_retain(void *pipeline_state)
{
   if (pipeline_state)
      ID3D12PipelineState_AddRef((ID3D12PipelineState *)pipeline_state);
}

static bool
d12_device_runtime_xml_write_escaped(FILE *destination, FILE *source)
{
   int character;

   rewind(source);
   while ((character = fgetc(source)) != EOF) {
      const char *replacement = NULL;

      switch (character) {
      case '&': replacement = "&amp;"; break;
      case '<': replacement = "&lt;"; break;
      case '>': replacement = "&gt;"; break;
      default: break;
      }

      if (replacement) {
         if (fputs(replacement, destination) == EOF)
            return false;
      } else if ((character >= 0x20 && character <= 0x7e) ||
                 character == '\n' || character == '\r' ||
                 character == '\t') {
         if (fputc(character, destination) == EOF)
            return false;
      } else if (fputs("&#xfffd;", destination) == EOF) {
         /* Disassembler output is ASCII today. Replace any future non-XML
          * byte instead of allowing malformed diagnostic output. */
         return false;
      }
   }

   return !ferror(source);
}

static bool
d12_device_runtime_emit_program(
   FILE *xml, const char *stage, const char *role, struct agx_bo *bo,
   size_t offset, size_t size, const struct agx_shader_info *info)
{
   FILE *assembly;
   uint8_t *code;
   bool success;

   if (!bo || !size || offset > bo->size || size > bo->size - offset)
      return true;

   code = agx_bo_map(bo);
   if (!code)
      return false;

   assembly = tmpfile();
   if (!assembly)
      return false;
   if (agx2_disassemble(code + offset, size, assembly)) {
      fclose(assembly);
      return false;
   }
   if (fflush(assembly) != 0) {
      fclose(assembly);
      return false;
   }

   success = fprintf(
      xml,
      "  <program stage=\"%s\" role=\"%s\" code-size=\"%zu\" "
      "instructions=\"%u\" gprs=\"%u\" uniforms=\"%u\" "
      "scratch-bytes=\"%u\" max-threads=\"%u\" spills=\"%u\" "
      "fills=\"%u\">\n    <assembly>",
      stage, role, size, info ? info->stats.instrs : 0,
      info ? info->stats.gprs : 0, info ? info->stats.uniforms : 0,
      info ? info->stats.scratch : 0, info ? info->stats.threads : 0,
      info ? info->stats.spills : 0, info ? info->stats.fills : 0) >= 0 &&
      d12_device_runtime_xml_write_escaped(xml, assembly) &&
      fputs("</assembly>\n  </program>\n", xml) != EOF;
   fclose(assembly);
   return success;
}

static bool
d12_device_runtime_emit_main(FILE *xml, const char *stage, const char *role,
                             struct agx_bo *bo,
                             const struct agx_shader_info *info)
{
   return !bo || !info ||
          d12_device_runtime_emit_program(xml, stage, role, bo,
                                          info->main_offset,
                                          info->main_size, info);
}

static bool
d12_device_runtime_emit_preamble(
   FILE *xml, const char *stage, const char *role, struct agx_bo *bo,
   const struct agx_shader_info *info)
{
   size_t size;

   if (!bo || !info || !info->has_preamble)
      return true;
   size = info->main_offset > info->preamble_offset
             ? info->main_offset - info->preamble_offset
             : info->binary_size - info->preamble_offset;
   return d12_device_runtime_emit_program(
      xml, stage, role, bo, info->preamble_offset, size, info);
}

static bool
d12_device_runtime_emit_linked(
   FILE *xml, const char *stage, const char *role,
   const struct d12_gfx_stage *shader)
{
   return !shader->linked.bo ||
          d12_device_runtime_emit_program(
             xml, stage, role, shader->linked.bo, 0, shader->linked.size,
             &shader->info);
}

HRESULT
d12_device_runtime_pipeline_state_comment(void *pipeline_state,
                                          char **out_comment,
                                          size_t *out_size)
{
   struct d12_pipeline_state *pipeline = pipeline_state;
   FILE *xml = NULL;
   char *comment = NULL;
   long signed_size;
   size_t size;
   bool success;

   if (!pipeline || !out_comment || !out_size)
      return E_INVALIDARG;
   *out_comment = NULL;
   *out_size = 0;

   xml = tmpfile();
   if (!xml)
      return E_FAIL;
   success = fputs("<?xml version=\"1.0\" encoding=\"us-ascii\"?>\n"
                   "<shader-disassembly vendor=\"Aurora Silicon\" "
                   "architecture=\"Apple AGX\">\n",
                   xml) != EOF;

   if (pipeline->is_compute) {
      success = success &&
                d12_device_runtime_emit_preamble(
                   xml, "compute", "application-preamble",
                   pipeline->code_bo, &pipeline->info) &&
                d12_device_runtime_emit_main(
                   xml, "compute", "application-main", pipeline->code_bo,
                   &pipeline->info);
   } else {
      success = success &&
                d12_device_runtime_emit_linked(
                   xml, "vertex", "application-linked", &pipeline->vs) &&
                d12_device_runtime_emit_preamble(
                   xml, "vertex", "application-preamble",
                   pipeline->vs.code_bo, &pipeline->vs.info) &&
                d12_device_runtime_emit_linked(
                   xml, "pixel", "application-linked", &pipeline->fs) &&
                d12_device_runtime_emit_preamble(
                   xml, "pixel", "application-preamble",
                   pipeline->fs.code_bo, &pipeline->fs.info) &&
                d12_device_runtime_emit_main(
                   xml, "vertex", "software-prepass", pipeline->sw_vs_bo,
                   &pipeline->sw_vs_info) &&
                d12_device_runtime_emit_main(
                   xml, "domain", "software-prepass", pipeline->sw_tes_bo,
                   &pipeline->sw_tes_info) &&
                d12_device_runtime_emit_main(
                   xml, "hull", "application-main", pipeline->tcs_bo,
                   &pipeline->tcs_info) &&
                d12_device_runtime_emit_linked(
                   xml, "domain", "application-linked", &pipeline->tes) &&
                d12_device_runtime_emit_preamble(
                   xml, "domain", "application-preamble",
                   pipeline->tes.code_bo, &pipeline->tes.info) &&
                d12_device_runtime_emit_main(
                   xml, "geometry", "count", pipeline->gs_count_bo,
                   &pipeline->gs_count_info) &&
                d12_device_runtime_emit_main(
                   xml, "geometry", "prepass", pipeline->gs_pre_bo,
                   &pipeline->gs_pre_info) &&
                d12_device_runtime_emit_main(
                   xml, "geometry", "application-main",
                   pipeline->gs_main_bo, &pipeline->gs_main_info) &&
                d12_device_runtime_emit_linked(
                   xml, "geometry", "rasterization-linked",
                   &pipeline->gs_rast) &&
                d12_device_runtime_emit_preamble(
                   xml, "geometry", "rasterization-preamble",
                   pipeline->gs_rast.code_bo, &pipeline->gs_rast.info) &&
                d12_device_runtime_emit_main(
                   xml, "mesh", "application-main", pipeline->ms_bo,
                   &pipeline->ms_info) &&
                d12_device_runtime_emit_main(
                   xml, "mesh", "setup", pipeline->ms_setup_bo,
                   &pipeline->ms_setup_info) &&
                d12_device_runtime_emit_linked(
                   xml, "mesh", "rasterization-linked",
                   &pipeline->ms_rast) &&
                d12_device_runtime_emit_preamble(
                   xml, "mesh", "rasterization-preamble",
                   pipeline->ms_rast.code_bo, &pipeline->ms_rast.info) &&
                d12_device_runtime_emit_main(
                   xml, "amplification", "application-main",
                   pipeline->as_bo, &pipeline->as_info) &&
                d12_device_runtime_emit_main(
                   xml, "amplification", "setup", pipeline->as_setup_bo,
                   &pipeline->as_setup_info);
   }

   success = success && fputs("</shader-disassembly>\n", xml) != EOF &&
             fflush(xml) == 0 && fseek(xml, 0, SEEK_END) == 0;
   signed_size = success ? ftell(xml) : -1;
   if (signed_size <= 0 || fseek(xml, 0, SEEK_SET) != 0) {
      fclose(xml);
      return E_FAIL;
   }
   size = (size_t)signed_size;
   if (size == SIZE_MAX) {
      fclose(xml);
      return E_OUTOFMEMORY;
   }
   comment = malloc(size + 1);
   if (!comment) {
      fclose(xml);
      return E_OUTOFMEMORY;
   }
   if (fread(comment, 1, size, xml) != size || ferror(xml)) {
      free(comment);
      fclose(xml);
      return E_FAIL;
   }
   fclose(xml);
   comment[size] = '\0';
   *out_comment = comment;
   *out_size = size;
   return S_OK;
}

void
d12_device_runtime_command_list_set_pipeline_state(void *list,
                                                    void *pipeline_state)
{
   if (list)
      ID3D12GraphicsCommandList_SetPipelineState(
         (ID3D12GraphicsCommandList *)list,
         (ID3D12PipelineState *)pipeline_state);
}

static bool
d12_device_runtime_heap_desc(
   const struct d12_frontend_heap_desc *desc, D3D12_HEAP_DESC *translated)
{
   D3D12_CPU_PAGE_PROPERTY cpu_page_property;
   D3D12_MEMORY_POOL memory_pool;
   D3D12_HEAP_FLAGS flags = D3D12_HEAP_FLAG_NONE;
   const uint32_t category_mask =
      D12_FRONTEND_HEAP_ALLOW_NON_RT_DS_TEXTURES |
      D12_FRONTEND_HEAP_ALLOW_BUFFERS |
      D12_FRONTEND_HEAP_ALLOW_RT_DS_TEXTURES;

   if (!desc || !translated || !desc->size || !(desc->flags & category_mask) ||
       (desc->flags & ~(category_mask |
                        D12_FRONTEND_HEAP_COHERENT_SYSTEMWIDE |
                        D12_FRONTEND_HEAP_PRIMARY)))
      return false;
   switch (desc->cpu_page_property) {
   case D12_FRONTEND_CPU_PAGE_NOT_AVAILABLE:
      cpu_page_property = D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE;
      break;
   case D12_FRONTEND_CPU_PAGE_WRITE_COMBINE:
      cpu_page_property = D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE;
      break;
   case D12_FRONTEND_CPU_PAGE_WRITE_BACK:
      cpu_page_property = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
      break;
   default:
      return false;
   }
   switch (desc->memory_pool) {
   case D12_FRONTEND_MEMORY_POOL_L0:
      memory_pool = D3D12_MEMORY_POOL_L0;
      break;
   case D12_FRONTEND_MEMORY_POOL_L1:
      memory_pool = D3D12_MEMORY_POOL_L1;
      break;
   default:
      return false;
   }

   if (!(desc->flags & D12_FRONTEND_HEAP_ALLOW_BUFFERS))
      flags |= D3D12_HEAP_FLAG_DENY_BUFFERS;
   if (!(desc->flags & D12_FRONTEND_HEAP_ALLOW_RT_DS_TEXTURES))
      flags |= D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES;
   if (!(desc->flags & D12_FRONTEND_HEAP_ALLOW_NON_RT_DS_TEXTURES))
      flags |= D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES;
   if (desc->flags & D12_FRONTEND_HEAP_PRIMARY)
      flags |= D3D12_HEAP_FLAG_ALLOW_DISPLAY;

   *translated = (D3D12_HEAP_DESC){
      .SizeInBytes = desc->size,
      .Properties = {
         .Type = D3D12_HEAP_TYPE_CUSTOM,
         .CPUPageProperty = cpu_page_property,
         .MemoryPoolPreference = memory_pool,
         .CreationNodeMask = desc->creation_node_mask,
         .VisibleNodeMask = desc->visible_node_mask,
      },
      .Alignment = desc->alignment,
      .Flags = flags,
   };
   return true;
}

static bool
d12_device_runtime_resource_desc(
   const struct d12_frontend_resource_desc *desc,
   D3D12_RESOURCE_DESC *translated)
{
   D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE;
   const uint32_t supported =
      D12_FRONTEND_RESOURCE_ALLOW_RENDER_TARGET |
      D12_FRONTEND_RESOURCE_ALLOW_DEPTH_STENCIL |
      D12_FRONTEND_RESOURCE_ALLOW_CROSS_ADAPTER |
      D12_FRONTEND_RESOURCE_ALLOW_SIMULTANEOUS_ACCESS |
      D12_FRONTEND_RESOURCE_ALLOW_SHADER_RESOURCE |
      D12_FRONTEND_RESOURCE_ALLOW_UNORDERED_ACCESS;

   if (!desc || !translated || !desc->width ||
       desc->dimension < D3D12_RESOURCE_DIMENSION_BUFFER ||
       desc->dimension > D3D12_RESOURCE_DIMENSION_TEXTURE3D ||
       (desc->flags & ~supported))
      return false;
   if (desc->flags & D12_FRONTEND_RESOURCE_ALLOW_RENDER_TARGET)
      flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
   if (desc->flags & D12_FRONTEND_RESOURCE_ALLOW_DEPTH_STENCIL)
      flags |= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
   if (desc->flags & D12_FRONTEND_RESOURCE_ALLOW_CROSS_ADAPTER)
      flags |= D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER;
   if (desc->flags & D12_FRONTEND_RESOURCE_ALLOW_SIMULTANEOUS_ACCESS)
      flags |= D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;
   if (!(desc->flags & D12_FRONTEND_RESOURCE_ALLOW_SHADER_RESOURCE))
      flags |= D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE;
   if (desc->flags & D12_FRONTEND_RESOURCE_ALLOW_UNORDERED_ACCESS)
      flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

   *translated = (D3D12_RESOURCE_DESC){
      .Dimension = (D3D12_RESOURCE_DIMENSION)desc->dimension,
      .Alignment = desc->alignment,
      .Width = desc->width,
      .Height = desc->height,
      .DepthOrArraySize = desc->depth_or_array_size,
      .MipLevels = desc->mip_levels,
      .Format = (DXGI_FORMAT)desc->format,
      .SampleDesc = {
         .Count = desc->sample_count,
         .Quality = desc->sample_quality,
      },
      .Layout = (D3D12_TEXTURE_LAYOUT)desc->layout,
      .Flags = flags,
   };
   return true;
}

HRESULT
d12_device_runtime_heap_create(
   struct d12_device *device, const struct d12_frontend_heap_desc *desc,
   void *runtime_resource, const void *resource_driver_data,
   uint32_t resource_driver_data_size, uint64_t standard_descriptor,
   void **out_heap,
   struct d12_frontend_heap_allocation *out_allocation)
{
   D3D12_HEAP_DESC heap_desc;
   struct agx_bo *bo;
   ID3D12Heap *heap = NULL;
   enum agx_bo_flags bo_flags =
      AGX_BO_WRITEBACK | AGX_BO_ZERO_ON_REUSE | AGX_BO_SHAREABLE;
   HRESULT result;

   if (!device || !runtime_resource || !out_heap || !out_allocation ||
       !d12_device_runtime_heap_desc(desc, &heap_desc))
      return E_INVALIDARG;
   *out_heap = NULL;
   memset(out_allocation, 0, sizeof(*out_allocation));

   /* AppleAgxWddm contract v3 identifies desktop primaries explicitly and
    * requires them to be uncached, CPU-accessible and shareable. Keep that
    * allocation policy in the shared winsys marker rather than deriving it
    * later from display geometry. */
   if (desc->flags & D12_FRONTEND_HEAP_PRIMARY) {
      bo_flags = (enum agx_bo_flags)
         ((uint32_t)bo_flags & ~(uint32_t)AGX_BO_WRITEBACK);
      bo_flags = (enum agx_bo_flags)
         ((uint32_t)bo_flags | (uint32_t)AGX_BO_PRIMARY);
   }

   bo = agx_d3dkmt_bo_create_runtime_resource_with_driver_data(
      &device->dev, heap_desc.SizeInBytes, (unsigned)heap_desc.Alignment,
      bo_flags, runtime_resource, resource_driver_data,
      resource_driver_data_size, standard_descriptor,
      "D3D12 runtime heap");
   if (!bo)
      return E_OUTOFMEMORY;

   result = d12_heap_create_from_bo(device, &heap_desc, bo,
                                    &IID_ID3D12Heap, (void **)&heap);
   if (SUCCEEDED(result)) {
      uint32_t allocation;
      uint32_t resource;
      if (!agx_d3dkmt_bo_runtime_identity(&device->dev, bo, &allocation,
                                          &resource)) {
         ID3D12Heap_Release(heap);
         result = E_FAIL;
      } else {
         out_allocation->allocation = allocation;
         out_allocation->resource = resource;
         out_allocation->gpu_virtual_address = bo->va ? bo->va->addr : 0;
         out_allocation->size = bo->size;
         *out_heap = heap;
      }
   }
   agx_bo_unreference(&device->dev, bo);
   return result;
}

HRESULT
d12_device_runtime_heap_open(
   struct d12_device *device, const struct d12_frontend_heap_desc *desc,
   void *runtime_resource, uint32_t allocation, uint32_t resource,
   uint32_t allocation_flags, void **out_heap,
   struct d12_frontend_heap_allocation *out_allocation)
{
   D3D12_HEAP_DESC heap_desc;
   enum agx_bo_flags bo_flags = AGX_BO_ZERO_ON_REUSE;
   struct agx_bo *bo;
   ID3D12Heap *heap = NULL;
   HRESULT result;

   if (!device || !runtime_resource || !allocation || !resource || !out_heap ||
       !out_allocation || !d12_device_runtime_heap_desc(desc, &heap_desc))
      return E_INVALIDARG;
   *out_heap = NULL;
   memset(out_allocation, 0, sizeof(*out_allocation));

   if (allocation_flags & AGXW_ALLOC_WRITEBACK)
      bo_flags |= AGX_BO_WRITEBACK;
   if (allocation_flags & AGXW_ALLOC_SHAREABLE)
      bo_flags |= AGX_BO_SHAREABLE;
   if (allocation_flags & AGXW_ALLOC_PRIMARY)
      bo_flags |= AGX_BO_PRIMARY;

   bo = agx_d3dkmt_bo_open_runtime_resource(
      &device->dev, heap_desc.SizeInBytes, (unsigned)heap_desc.Alignment,
      bo_flags, runtime_resource, allocation, resource,
      "D3D12 opened runtime heap");
   if (!bo)
      return E_OUTOFMEMORY;

   result = d12_heap_create_from_bo(device, &heap_desc, bo,
                                    &IID_ID3D12Heap, (void **)&heap);
   if (SUCCEEDED(result)) {
      out_allocation->allocation = bo->uapi_handle;
      out_allocation->resource = resource;
      out_allocation->gpu_virtual_address = bo->va ? bo->va->addr : 0;
      out_allocation->size = bo->size;
      *out_heap = heap;
   }
   agx_bo_unreference(&device->dev, bo);
   return result;
}

void
d12_device_runtime_heap_destroy(void *heap)
{
   if (heap)
      ID3D12Heap_Release((ID3D12Heap *)heap);
}

HRESULT
d12_device_runtime_resource_create_placed(
   struct d12_device *device, void *heap, uint64_t offset,
   const struct d12_frontend_resource_desc *desc,
   const struct d12_frontend_clear_value *clear_value, void **out_resource)
{
   D3D12_RESOURCE_DESC resource_desc;
   D3D12_CLEAR_VALUE translated_clear;
   D3D12_CLEAR_VALUE *clear = NULL;

   if (!device || !heap || !out_resource ||
       !d12_device_runtime_resource_desc(desc, &resource_desc))
      return E_INVALIDARG;
   *out_resource = NULL;
   if (clear_value) {
      translated_clear.Format = (DXGI_FORMAT)clear_value->format;
      if (desc->flags & D12_FRONTEND_RESOURCE_ALLOW_DEPTH_STENCIL) {
         translated_clear.DepthStencil.Depth =
            clear_value->depth_stencil.depth;
         translated_clear.DepthStencil.Stencil =
            clear_value->depth_stencil.stencil;
      } else {
         memcpy(translated_clear.Color, clear_value->color,
                sizeof(translated_clear.Color));
      }
      clear = &translated_clear;
   }
   return d12_resource_create_placed(
      device, d12_memheap_from_iface((ID3D12Heap *)heap), offset,
      &resource_desc, desc->row_pitch, desc->slice_pitch,
      (D3D12_RESOURCE_STATES)desc->initial_state, clear,
      &IID_ID3D12Resource, out_resource);
}

HRESULT
d12_device_runtime_resource_create_reused(
   struct d12_device *device, void *source_resource, uint64_t source_offset,
   const struct d12_frontend_resource_desc *desc,
   const struct d12_frontend_clear_value *clear_value, void **out_resource)
{
   D3D12_RESOURCE_DESC resource_desc;
   D3D12_CLEAR_VALUE translated_clear;
   D3D12_CLEAR_VALUE *clear = NULL;
   struct d12_resource *source =
      d12_resource_from_iface((ID3D12Resource *)source_resource);

   if (!device || !source || !out_resource ||
       !d12_device_runtime_resource_desc(desc, &resource_desc))
      return E_INVALIDARG;
   *out_resource = NULL;
   if (clear_value) {
      translated_clear.Format = (DXGI_FORMAT)clear_value->format;
      if (desc->flags & D12_FRONTEND_RESOURCE_ALLOW_DEPTH_STENCIL) {
         translated_clear.DepthStencil.Depth =
            clear_value->depth_stencil.depth;
         translated_clear.DepthStencil.Stencil =
            clear_value->depth_stencil.stencil;
      } else {
         memcpy(translated_clear.Color, clear_value->color,
                sizeof(translated_clear.Color));
      }
      clear = &translated_clear;
   }
   return d12_resource_create_reused(
      device, source, source_offset, &resource_desc, desc->row_pitch,
      desc->slice_pitch,
      (D3D12_RESOURCE_STATES)desc->initial_state, clear,
      &IID_ID3D12Resource, out_resource);
}

HRESULT
d12_device_runtime_resource_validate_reserved(
   const struct d12_frontend_resource_desc *desc)
{
   D3D12_RESOURCE_DESC resource_desc;

   if (!d12_device_runtime_resource_desc(desc, &resource_desc))
      return E_INVALIDARG;
   return d12_resource_validate_reserved_tier(&resource_desc);
}

HRESULT
d12_device_runtime_resource_create_reserved(
   struct d12_device *device, const struct d12_frontend_resource_desc *desc,
   const struct d12_frontend_clear_value *clear_value, void **out_resource)
{
   D3D12_RESOURCE_DESC resource_desc;
   D3D12_CLEAR_VALUE translated_clear;
   D3D12_CLEAR_VALUE *clear = NULL;

   if (!device || !out_resource || !desc || desc->row_pitch ||
       desc->slice_pitch ||
       !d12_device_runtime_resource_desc(desc, &resource_desc))
      return E_INVALIDARG;
   *out_resource = NULL;
   if (clear_value) {
      translated_clear.Format = (DXGI_FORMAT)clear_value->format;
      if (desc->flags & D12_FRONTEND_RESOURCE_ALLOW_DEPTH_STENCIL) {
         translated_clear.DepthStencil.Depth =
            clear_value->depth_stencil.depth;
         translated_clear.DepthStencil.Stencil =
            clear_value->depth_stencil.stencil;
      } else {
         memcpy(translated_clear.Color, clear_value->color,
                sizeof(translated_clear.Color));
      }
      clear = &translated_clear;
   }
   return d12_resource_create_reserved(
      device, &resource_desc, (D3D12_RESOURCE_STATES)desc->initial_state,
      clear, &IID_ID3D12Resource, out_resource);
}

void
d12_device_runtime_resource_destroy(void *resource)
{
   if (resource)
      ID3D12Resource_Release((ID3D12Resource *)resource);
}

uint64_t
d12_device_runtime_resource_gpu_virtual_address(void *resource)
{
   struct d12_resource *res = d12_resource_from_iface(resource);
   return res ? res->addr : 0;
}

HRESULT
d12_device_runtime_resource_mip_packing(void *resource,
                                        uint32_t *packed_mips,
                                        uint32_t *packed_tiles)
{
   struct d12_resource *res = d12_resource_from_iface(resource);
   if (packed_mips)
      *packed_mips = 0;
   if (packed_tiles)
      *packed_tiles = 0;
   if (!res || !res->reserved || !packed_mips || !packed_tiles)
      return E_INVALIDARG;
   *packed_mips = res->sparse_packed_mips.NumPackedMips;
   *packed_tiles = res->sparse_packed_mips.NumTilesForPackedMips;
   return S_OK;
}

HRESULT
d12_device_runtime_resource_allocation_info(
   struct d12_device *device, const struct d12_frontend_resource_desc *desc,
   uint32_t visible_node_mask,
   struct d12_frontend_resource_allocation_info *info)
{
   D3D12_RESOURCE_DESC resource_desc;
   D3D12_RESOURCE_ALLOCATION_INFO native;
   uint64_t explicit_row_major_size = 0;

   if (!device || !info || visible_node_mask > 1 ||
       !d12_device_runtime_resource_desc(desc, &resource_desc))
      return E_INVALIDARG;
   if (!d12_resource_explicit_row_major_size(
          &resource_desc, desc->row_pitch, desc->slice_pitch,
          &explicit_row_major_size))
      return E_INVALIDARG;
   d12_device_get_resource_allocation_info_runtime(
      device, visible_node_mask, &resource_desc, &native);
   if (native.SizeInBytes == UINT64_MAX)
      return E_INVALIDARG;
   if (explicit_row_major_size) {
      if (!native.Alignment ||
          explicit_row_major_size > UINT64_MAX - (native.Alignment - 1))
         return E_INVALIDARG;
      native.SizeInBytes =
         (explicit_row_major_size + native.Alignment - 1) &
         ~(native.Alignment - 1);
   }
   *info = (struct d12_frontend_resource_allocation_info){
      .size = native.SizeInBytes,
      .alignment = native.Alignment,
      .layout = desc->layout,
   };
   return S_OK;
}

uint64_t
d12_device_runtime_format_support(struct d12_device *device, uint32_t format)
{
   return device ? d12_format_frontend_support((DXGI_FORMAT)format) : 0;
}

uint32_t
d12_device_runtime_multisample_quality_levels(
   struct d12_device *device, uint32_t format, uint32_t sample_count,
   uint32_t flags)
{
   uint64_t support;

   if (!device ||
       (flags & ~D12_FRONTEND_MULTISAMPLE_QUALITY_TILED_RESOURCE) ||
       !sample_count || format == DXGI_FORMAT_UNKNOWN)
      return 0;
   support = d12_format_frontend_support((DXGI_FORMAT)format);
   if (!support)
      return 0;
   if (flags & D12_FRONTEND_MULTISAMPLE_QUALITY_TILED_RESOURCE)
      return sample_count == 1 &&
                    (support & D12_FRONTEND_FORMAT_TEXTURE_2D)
                ? 1
                : 0;
   if (sample_count == 1)
      return 1;
   /* One definition of what AGX can rasterize, shared with the layout
    * admission test: an 8x query is answered with zero, which is the D3D12
    * way to say the hardware does not offer it. */
   if (!d12_sample_count_supported(sample_count))
      return 0;
   return support & (D12_FRONTEND_FORMAT_MULTISAMPLE_RENDER_TARGET |
                     D12_FRONTEND_FORMAT_MULTISAMPLE_LOAD)
             ? 1
             : 0;
}

HRESULT
d12_device_runtime_resource_subresource_info(
   void *resource, uint32_t subresource,
   struct d12_frontend_subresource_info *info)
{
   struct d12_resource *res = d12_resource_from_iface(resource);
   const struct ail_layout *layout;
   uint32_t layers, planes, plane, layer, mip;
   uint64_t plane_offset = 0;

   if (!res || !info)
      return E_INVALIDARG;
   memset(info, 0, sizeof(*info));

   if (res->desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
      if (subresource)
         return E_INVALIDARG;
      info->row_stride = res->desc.Width;
      info->depth_stride = res->desc.Width;
      return S_OK;
   }

   layers = res->desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D
               ? 1
               : res->desc.DepthOrArraySize;
   planes = res->stencil_bo ? 2 : 1;
   if (!res->desc.MipLevels ||
       (uint64_t)subresource >=
          (uint64_t)res->desc.MipLevels * layers * planes)
      return E_INVALIDARG;
   mip = subresource % res->desc.MipLevels;
   layer = (subresource / res->desc.MipLevels) % layers;
   plane = subresource / (res->desc.MipLevels * layers);

   if (plane) {
      layout = &res->stencil_layout;
      if (!res->stencil_addr || res->stencil_addr < res->addr)
         return E_INVALIDARG;
      plane_offset = res->stencil_addr - res->addr;
   } else {
      layout = &res->layout;
   }

   info->offset = plane_offset + ail_get_layer_level_B(layout, layer, mip);
   info->row_stride =
      layout->tiling == AIL_TILING_LINEAR
         ? layout->linear_stride_B
         : (uint64_t)layout->stride_el[mip] *
              util_format_get_blocksize(layout->format);
   info->depth_stride = layout->layer_stride_B
                           ? layout->layer_stride_B
                           : ail_get_level_size_B(layout, mip);
   return S_OK;
}

HRESULT
d12_device_runtime_descriptor_heap_create(
   struct d12_device *device,
   const struct d12_frontend_descriptor_heap_desc *desc, void **out_heap)
{
   D3D12_DESCRIPTOR_HEAP_DESC translated;
   D3D12_DESCRIPTOR_HEAP_FLAGS flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
   const uint32_t supported =
      D12_FRONTEND_DESCRIPTOR_HEAP_CPU_VISIBLE |
      D12_FRONTEND_DESCRIPTOR_HEAP_SHADER_VISIBLE;

   if (!device || !desc || !out_heap || !desc->num_descriptors ||
       desc->type >= D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES ||
       desc->node_mask > 1 || (desc->flags & ~supported))
      return E_INVALIDARG;
   if (desc->flags & D12_FRONTEND_DESCRIPTOR_HEAP_SHADER_VISIBLE)
      flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
   translated = (D3D12_DESCRIPTOR_HEAP_DESC){
      .Type = (D3D12_DESCRIPTOR_HEAP_TYPE)desc->type,
      .NumDescriptors = desc->num_descriptors,
      .Flags = flags,
      .NodeMask = desc->node_mask,
   };
   return d12_descriptor_heap_create(
      device, &translated, &IID_ID3D12DescriptorHeap, out_heap);
}

void
d12_device_runtime_descriptor_heap_destroy(void *heap)
{
   if (heap)
      ID3D12DescriptorHeap_Release((ID3D12DescriptorHeap *)heap);
}

uint64_t
d12_device_runtime_descriptor_heap_cpu_start(void *heap)
{
   const struct d12_descriptor_heap *native = heap;
   return native ? (uint64_t)(uintptr_t)native->slots : 0;
}

uint64_t
d12_device_runtime_descriptor_heap_gpu_start(void *heap)
{
   const struct d12_descriptor_heap *native = heap;
   if (!native ||
       !(native->desc.Flags & D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE))
      return 0;
   return (uint64_t)(uintptr_t)native->slots;
}

void
d12_device_runtime_command_list_set_descriptor_heaps(
   void *list, uint32_t count, void *const *heaps)
{
   ID3D12DescriptorHeap *translated[2];

   if (!list || count > ARRAY_SIZE(translated) || (count && !heaps))
      return;
   for (uint32_t index = 0; index < count; ++index)
      translated[index] = (ID3D12DescriptorHeap *)heaps[index];
   ID3D12GraphicsCommandList_SetDescriptorHeaps(
      (ID3D12GraphicsCommandList *)list, count, translated);
}

void
d12_device_runtime_create_shader_resource_view(
   struct d12_device *device, void *resource,
   const struct d12_frontend_srv_desc *desc, uint64_t destination)
{
   D3D12_SHADER_RESOURCE_VIEW_DESC translated;
   const struct d12_resource *res = resource;
   const bool msaa = res && res->desc.SampleDesc.Count > 1;

   if (!device || !desc || !destination)
      return;
   translated = (D3D12_SHADER_RESOURCE_VIEW_DESC){
      .Format = (DXGI_FORMAT)desc->format,
      .Shader4ComponentMapping = desc->component_mapping,
   };
   switch (desc->dimension) {
   case D3D12_RESOURCE_DIMENSION_BUFFER:
      translated.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
      translated.Buffer = (D3D12_BUFFER_SRV){
         .FirstElement = desc->buffer.first_element,
         .NumElements = desc->buffer.num_elements,
         .StructureByteStride = desc->buffer.stride,
         .Flags = (D3D12_BUFFER_SRV_FLAGS)desc->buffer.flags,
      };
      break;
   case D3D12_RESOURCE_DIMENSION_TEXTURE1D:
      if (desc->texture.first_slice || desc->texture.array_size > 1) {
         translated.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1DARRAY;
         translated.Texture1DArray = (D3D12_TEX1D_ARRAY_SRV){
            .MostDetailedMip = desc->texture.first_mip,
            .MipLevels = desc->texture.mip_levels,
            .FirstArraySlice = desc->texture.first_slice,
            .ArraySize = desc->texture.array_size,
            .ResourceMinLODClamp = desc->texture.min_lod,
         };
      } else {
         translated.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1D;
         translated.Texture1D = (D3D12_TEX1D_SRV){
            .MostDetailedMip = desc->texture.first_mip,
            .MipLevels = desc->texture.mip_levels,
            .ResourceMinLODClamp = desc->texture.min_lod,
         };
      }
      break;
   case D3D12_RESOURCE_DIMENSION_TEXTURE2D:
      if (msaa) {
         if (desc->texture.first_slice || desc->texture.array_size > 1) {
            translated.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY;
            translated.Texture2DMSArray = (D3D12_TEX2DMS_ARRAY_SRV){
               .FirstArraySlice = desc->texture.first_slice,
               .ArraySize = desc->texture.array_size,
            };
         } else {
            translated.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMS;
         }
      } else if (desc->texture.first_slice || desc->texture.array_size > 1) {
         translated.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
         translated.Texture2DArray = (D3D12_TEX2D_ARRAY_SRV){
            .MostDetailedMip = desc->texture.first_mip,
            .MipLevels = desc->texture.mip_levels,
            .FirstArraySlice = desc->texture.first_slice,
            .ArraySize = desc->texture.array_size,
            .PlaneSlice = desc->texture.plane_slice,
            .ResourceMinLODClamp = desc->texture.min_lod,
         };
      } else {
         translated.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
         translated.Texture2D = (D3D12_TEX2D_SRV){
            .MostDetailedMip = desc->texture.first_mip,
            .MipLevels = desc->texture.mip_levels,
            .PlaneSlice = desc->texture.plane_slice,
            .ResourceMinLODClamp = desc->texture.min_lod,
         };
      }
      break;
   case D3D12_RESOURCE_DIMENSION_TEXTURE3D:
      translated.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
      translated.Texture3D = (D3D12_TEX3D_SRV){
         .MostDetailedMip = desc->texture.first_mip,
         .MipLevels = desc->texture.mip_levels,
         .ResourceMinLODClamp = desc->texture.min_lod,
      };
      break;
   case 5:
      if (desc->cube.first_face || desc->cube.num_cubes > 1) {
         translated.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBEARRAY;
         translated.TextureCubeArray = (D3D12_TEXCUBE_ARRAY_SRV){
            .MostDetailedMip = desc->cube.first_mip,
            .MipLevels = desc->cube.mip_levels,
            .First2DArrayFace = desc->cube.first_face,
            .NumCubes = desc->cube.num_cubes,
            .ResourceMinLODClamp = desc->cube.min_lod,
         };
      } else {
         translated.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
         translated.TextureCube = (D3D12_TEXCUBE_SRV){
            .MostDetailedMip = desc->cube.first_mip,
            .MipLevels = desc->cube.mip_levels,
            .ResourceMinLODClamp = desc->cube.min_lod,
         };
      }
      break;
   default:
      return;
   }
   d12_device_create_srv_runtime(
      device, (ID3D12Resource *)resource, &translated,
      (D3D12_CPU_DESCRIPTOR_HANDLE){.ptr = (SIZE_T)destination});
}

void
d12_device_runtime_create_unordered_access_view(
   struct d12_device *device, void *resource, void *counter_resource,
   const struct d12_frontend_uav_desc *desc, uint64_t destination)
{
   D3D12_UNORDERED_ACCESS_VIEW_DESC translated;

   if (!device || !desc || !destination)
      return;
   translated = (D3D12_UNORDERED_ACCESS_VIEW_DESC){
      .Format = (DXGI_FORMAT)desc->format,
   };
   switch (desc->dimension) {
   case D3D12_RESOURCE_DIMENSION_BUFFER:
      translated.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
      translated.Buffer = (D3D12_BUFFER_UAV){
         .FirstElement = desc->buffer.first_element,
         .NumElements = desc->buffer.num_elements,
         .StructureByteStride = desc->buffer.stride,
         .CounterOffsetInBytes = desc->buffer.counter_offset,
         .Flags = (D3D12_BUFFER_UAV_FLAGS)desc->buffer.flags,
      };
      break;
   case D3D12_RESOURCE_DIMENSION_TEXTURE1D:
      if (desc->texture.first_slice || desc->texture.array_size > 1) {
         translated.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
         translated.Texture1DArray = (D3D12_TEX1D_ARRAY_UAV){
            .MipSlice = desc->texture.mip_slice,
            .FirstArraySlice = desc->texture.first_slice,
            .ArraySize = desc->texture.array_size,
         };
      } else {
         translated.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
         translated.Texture1D.MipSlice = desc->texture.mip_slice;
      }
      break;
   case D3D12_RESOURCE_DIMENSION_TEXTURE2D:
      if (desc->texture.first_slice || desc->texture.array_size > 1) {
         translated.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
         translated.Texture2DArray = (D3D12_TEX2D_ARRAY_UAV){
            .MipSlice = desc->texture.mip_slice,
            .FirstArraySlice = desc->texture.first_slice,
            .ArraySize = desc->texture.array_size,
            .PlaneSlice = desc->texture.plane_slice,
         };
      } else {
         translated.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
         translated.Texture2D = (D3D12_TEX2D_UAV){
            .MipSlice = desc->texture.mip_slice,
            .PlaneSlice = desc->texture.plane_slice,
         };
      }
      break;
   case D3D12_RESOURCE_DIMENSION_TEXTURE3D:
      translated.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
      translated.Texture3D = (D3D12_TEX3D_UAV){
         .MipSlice = desc->texture.mip_slice,
         .FirstWSlice = desc->texture.first_slice,
         .WSize = desc->texture.array_size,
      };
      break;
   default:
      return;
   }
   d12_device_create_uav_runtime(
      device, (ID3D12Resource *)resource,
      (ID3D12Resource *)counter_resource, &translated,
      (D3D12_CPU_DESCRIPTOR_HANDLE){.ptr = (SIZE_T)destination});
}

void
d12_device_runtime_create_render_target_view(
   struct d12_device *device, void *resource,
   const struct d12_frontend_rtv_desc *desc, uint64_t destination)
{
   D3D12_RENDER_TARGET_VIEW_DESC translated;
   const struct d12_resource *res = resource;
   const bool msaa = res && res->desc.SampleDesc.Count > 1;

   if (!device || !desc || !destination)
      return;
   translated = (D3D12_RENDER_TARGET_VIEW_DESC){
      .Format = (DXGI_FORMAT)desc->format,
   };
   switch (desc->dimension) {
   case D3D12_RESOURCE_DIMENSION_BUFFER:
      translated.ViewDimension = D3D12_RTV_DIMENSION_BUFFER;
      translated.Buffer = (D3D12_BUFFER_RTV){
         .FirstElement = desc->buffer.first_element,
         .NumElements = desc->buffer.num_elements,
      };
      break;
   case D3D12_RESOURCE_DIMENSION_TEXTURE1D:
      if (desc->texture.first_slice || desc->texture.array_size > 1) {
         translated.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE1DARRAY;
         translated.Texture1DArray = (D3D12_TEX1D_ARRAY_RTV){
            .MipSlice = desc->texture.mip_slice,
            .FirstArraySlice = desc->texture.first_slice,
            .ArraySize = desc->texture.array_size,
         };
      } else {
         translated.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE1D;
         translated.Texture1D.MipSlice = desc->texture.mip_slice;
      }
      break;
   case D3D12_RESOURCE_DIMENSION_TEXTURE2D:
   case 5:
      if (msaa) {
         if (desc->texture.first_slice || desc->texture.array_size > 1) {
            translated.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DMSARRAY;
            translated.Texture2DMSArray = (D3D12_TEX2DMS_ARRAY_RTV){
               .FirstArraySlice = desc->texture.first_slice,
               .ArraySize = desc->texture.array_size,
            };
         } else {
            translated.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DMS;
         }
      } else if (desc->texture.first_slice || desc->texture.array_size > 1 ||
                 desc->dimension == 5) {
         translated.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DARRAY;
         translated.Texture2DArray = (D3D12_TEX2D_ARRAY_RTV){
            .MipSlice = desc->texture.mip_slice,
            .FirstArraySlice = desc->texture.first_slice,
            .ArraySize = desc->texture.array_size,
            .PlaneSlice = desc->texture.plane_slice,
         };
      } else {
         translated.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
         translated.Texture2D = (D3D12_TEX2D_RTV){
            .MipSlice = desc->texture.mip_slice,
            .PlaneSlice = desc->texture.plane_slice,
         };
      }
      break;
   case D3D12_RESOURCE_DIMENSION_TEXTURE3D:
      translated.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE3D;
      translated.Texture3D = (D3D12_TEX3D_RTV){
         .MipSlice = desc->texture.mip_slice,
         .FirstWSlice = desc->texture.first_slice,
         .WSize = desc->texture.array_size,
      };
      break;
   default:
      return;
   }
   d12_device_create_rtv_runtime(
      device, (ID3D12Resource *)resource, &translated,
      (D3D12_CPU_DESCRIPTOR_HANDLE){.ptr = (SIZE_T)destination});
}

void
d12_device_runtime_create_depth_stencil_view(
   struct d12_device *device, void *resource,
   const struct d12_frontend_dsv_desc *desc, uint64_t destination)
{
   D3D12_DEPTH_STENCIL_VIEW_DESC translated;
   const struct d12_resource *res = resource;
   const bool msaa = res && res->desc.SampleDesc.Count > 1;

   if (!device || !desc || !destination)
      return;
   translated = (D3D12_DEPTH_STENCIL_VIEW_DESC){
      .Format = (DXGI_FORMAT)desc->format,
      .Flags = (D3D12_DSV_FLAGS)desc->flags,
   };
   switch (desc->dimension) {
   case D3D12_RESOURCE_DIMENSION_TEXTURE1D:
      if (desc->first_slice || desc->array_size > 1) {
         translated.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE1DARRAY;
         translated.Texture1DArray = (D3D12_TEX1D_ARRAY_DSV){
            .MipSlice = desc->mip_slice,
            .FirstArraySlice = desc->first_slice,
            .ArraySize = desc->array_size,
         };
      } else {
         translated.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE1D;
         translated.Texture1D.MipSlice = desc->mip_slice;
      }
      break;
   case D3D12_RESOURCE_DIMENSION_TEXTURE2D:
   case 5:
      if (msaa) {
         if (desc->first_slice || desc->array_size > 1) {
            translated.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DMSARRAY;
            translated.Texture2DMSArray = (D3D12_TEX2DMS_ARRAY_DSV){
               .FirstArraySlice = desc->first_slice,
               .ArraySize = desc->array_size,
            };
         } else {
            translated.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DMS;
         }
      } else if (desc->first_slice || desc->array_size > 1 ||
                 desc->dimension == 5) {
         translated.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DARRAY;
         translated.Texture2DArray = (D3D12_TEX2D_ARRAY_DSV){
            .MipSlice = desc->mip_slice,
            .FirstArraySlice = desc->first_slice,
            .ArraySize = desc->array_size,
         };
      } else {
         translated.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
         translated.Texture2D.MipSlice = desc->mip_slice;
      }
      break;
   default:
      return;
   }
   d12_device_create_dsv_runtime(
      device, (ID3D12Resource *)resource, &translated,
      (D3D12_CPU_DESCRIPTOR_HANDLE){.ptr = (SIZE_T)destination});
}

void
d12_device_runtime_create_constant_buffer_view(
   struct d12_device *device, uint64_t address, uint32_t size,
   uint64_t destination)
{
   const D3D12_CONSTANT_BUFFER_VIEW_DESC translated = {
      .BufferLocation = address,
      .SizeInBytes = size,
   };
   if (device && destination)
      d12_device_create_cbv_runtime(
         device, &translated,
         (D3D12_CPU_DESCRIPTOR_HANDLE){.ptr = (SIZE_T)destination});
}

void
d12_device_runtime_create_sampler(
   struct d12_device *device,
   const struct d12_frontend_sampler_desc *desc, uint64_t destination)
{
   D3D12_SAMPLER_DESC2 translated;
   struct d12_descriptor *slot;

   if (!device || !desc || !destination)
      return;
   translated = (D3D12_SAMPLER_DESC2){
      .Filter = (D3D12_FILTER)desc->filter,
      .AddressU = (D3D12_TEXTURE_ADDRESS_MODE)desc->address_u,
      .AddressV = (D3D12_TEXTURE_ADDRESS_MODE)desc->address_v,
      .AddressW = (D3D12_TEXTURE_ADDRESS_MODE)desc->address_w,
      .MipLODBias = desc->mip_lod_bias,
      .MaxAnisotropy = desc->max_anisotropy,
      .ComparisonFunc = (D3D12_COMPARISON_FUNC)desc->comparison_func,
      .MinLOD = desc->min_lod,
      .MaxLOD = desc->max_lod,
      .Flags = (D3D12_SAMPLER_FLAGS)desc->flags,
   };
   memcpy(translated.UintBorderColor, desc->border_color,
          sizeof(translated.UintBorderColor));
   slot = (struct d12_descriptor *)(uintptr_t)destination;
   d12_pack_sampler_descriptor(device, slot, &translated);
   d12_descriptor_heap_update_slot(device, slot);
}

void
d12_device_runtime_copy_descriptors_simple(
   struct d12_device *device, uint32_t count, uint64_t destination,
   uint64_t source, uint32_t type)
{
   if (!device || type >= D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES)
      return;
   d12_device_copy_descriptors_simple_runtime(
      device, count,
      (D3D12_CPU_DESCRIPTOR_HANDLE){.ptr = (SIZE_T)destination},
      (D3D12_CPU_DESCRIPTOR_HANDLE){.ptr = (SIZE_T)source},
      (D3D12_DESCRIPTOR_HEAP_TYPE)type);
}

HRESULT
d12_device_runtime_query_heap_create(
   struct d12_device *device, uint32_t runtime_type, uint32_t count,
   uint32_t node_mask, void **out_heap)
{
   D3D12_QUERY_HEAP_TYPE type;
   D3D12_QUERY_HEAP_DESC desc;

   if (!device || !count || node_mask > 1 || !out_heap ||
       !d12_device_runtime_query_heap_type(runtime_type, &type))
      return E_INVALIDARG;
   desc = (D3D12_QUERY_HEAP_DESC){
      .Type = type,
      .Count = count,
      .NodeMask = node_mask,
   };
   return d12_query_heap_create(device, &desc, &IID_ID3D12QueryHeap,
                                out_heap);
}

void
d12_device_runtime_query_heap_destroy(void *heap)
{
   if (heap)
      ID3D12QueryHeap_Release((ID3D12QueryHeap *)heap);
}

void
d12_device_runtime_command_list_begin_query(
   void *list, void *heap, uint32_t runtime_type, uint32_t index)
{
   D3D12_QUERY_TYPE type;

   if (!list || !heap ||
       !d12_device_runtime_query_type(runtime_type, &type))
      return;
   ID3D12GraphicsCommandList_BeginQuery(
      (ID3D12GraphicsCommandList *)list, (ID3D12QueryHeap *)heap, type,
      index);
}

void
d12_device_runtime_command_list_end_query(
   void *list, void *heap, uint32_t runtime_type, uint32_t index)
{
   D3D12_QUERY_TYPE type;

   if (!list || !heap ||
       !d12_device_runtime_query_type(runtime_type, &type))
      return;
   ID3D12GraphicsCommandList_EndQuery(
      (ID3D12GraphicsCommandList *)list, (ID3D12QueryHeap *)heap, type,
      index);
}

void
d12_device_runtime_command_list_resolve_query_data(
   void *list, void *heap, uint32_t runtime_type, uint32_t first,
   uint32_t count, void *destination, uint64_t offset)
{
   D3D12_QUERY_TYPE type;

   if (!list || !heap || !destination || !count ||
       !d12_device_runtime_query_type(runtime_type, &type))
      return;
   ID3D12GraphicsCommandList_ResolveQueryData(
      (ID3D12GraphicsCommandList *)list, (ID3D12QueryHeap *)heap, type,
      first, count, (ID3D12Resource *)destination, offset);
}

void
d12_device_runtime_command_list_set_predication(
   void *list, void *resource, uint64_t offset, uint32_t operation)
{
   D3D12_PREDICATION_OP translated;

   if (!list || operation > 1)
      return;
   translated = operation == 0 ? D3D12_PREDICATION_OP_EQUAL_ZERO
                               : D3D12_PREDICATION_OP_NOT_EQUAL_ZERO;
   ID3D12GraphicsCommandList_SetPredication(
      (ID3D12GraphicsCommandList *)list, (ID3D12Resource *)resource, offset,
      translated);
}
