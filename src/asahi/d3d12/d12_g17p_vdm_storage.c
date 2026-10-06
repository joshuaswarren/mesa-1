/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT */
#include "d12_private.h"
#include "asahi/lib/agx_va_runtime_hole.h"

#define D12_G17P_VDM_SLAB_SIZE (64u * 1024u)

void
d12_g17p_vdm_storage_ref(struct d12_g17p_vdm_storage *storage)
{
   p_atomic_inc(&storage->refcount);
}

/* Slabs kept bound for reuse per device. A command list that records a
 * few passes needs one or two; the bound keeps a burst of retired lists
 * from pinning the low VDM aperture. */
#define D12_G17P_VDM_POOL_MAX 16

void
d12_device_g17p_vdm_pool_init(struct d12_device *device)
{
   simple_mtx_init(&device->g17p_vdm_pool_lock, mtx_plain);
   util_dynarray_init(&device->g17p_vdm_pool, NULL);
}

static void
d12_g17p_vdm_storage_destroy(struct d12_g17p_vdm_storage *storage)
{
   struct agx_device *dev = &storage->device->dev;
   /* Never recycle an alias whose mapping could still exist. Device teardown
    * is the recovery boundary after this exceptional unbind failure. */
   if (agx_bo_bind(dev, NULL, storage->alias, storage->bo->size, 0,
                    DRM_ASAHI_BIND_UNBIND)) {
      mesa_loge("native VDM alias unbind failed; alias/backing BO quarantined");
      free(storage);
      return;
   }
   simple_mtx_lock(&dev->vma_lock);
   util_vma_heap_free(&dev->g17p_vdm_alias_heap, storage->alias, storage->bo->size);
   simple_mtx_unlock(&dev->vma_lock);
   agx_bo_unreference(dev, storage->bo);
   free(storage);
}

void
d12_device_g17p_vdm_pool_fini(struct d12_device *device)
{
   util_dynarray_foreach(&device->g17p_vdm_pool,
                         struct d12_g17p_vdm_storage *, storage)
      d12_g17p_vdm_storage_destroy(*storage);
   util_dynarray_fini(&device->g17p_vdm_pool);
   simple_mtx_destroy(&device->g17p_vdm_pool_lock);
}

void
d12_g17p_vdm_storage_unref(struct d12_g17p_vdm_storage *storage)
{
   if (!storage || p_atomic_dec_return(&storage->refcount))
      return;

   /* The last submitted reader has retired: park the slab, still bound at
    * its alias, for the next command list instead of unbinding it and
    * binding a fresh one per Reset. */
   struct d12_device *device = storage->device;
   simple_mtx_lock(&device->g17p_vdm_pool_lock);
   const bool pooled =
      util_dynarray_num_elements(&device->g17p_vdm_pool,
                                 struct d12_g17p_vdm_storage *) <
      D12_G17P_VDM_POOL_MAX;
   if (pooled) {
      storage->refcount = 0;
      storage->used = 0;
      util_dynarray_append_typed(&device->g17p_vdm_pool,
                                 struct d12_g17p_vdm_storage *, storage);
   }
   simple_mtx_unlock(&device->g17p_vdm_pool_lock);
   if (!pooled)
      d12_g17p_vdm_storage_destroy(storage);
}

void
d12_list_g17p_vdm_storage_reset(struct d12_cmd_list *list)
{
   util_dynarray_foreach(&list->g17p_vdm_storage,
                         struct d12_g17p_vdm_storage *, storage)
      d12_g17p_vdm_storage_unref(*storage);
   util_dynarray_clear(&list->g17p_vdm_storage);
}

static struct d12_g17p_vdm_storage *
d12_g17p_vdm_storage_create(struct d12_device *device)
{
   struct agx_device *dev = &device->dev;

   simple_mtx_lock(&device->g17p_vdm_pool_lock);
   struct d12_g17p_vdm_storage *storage =
      util_dynarray_num_elements(&device->g17p_vdm_pool,
                                 struct d12_g17p_vdm_storage *)
         ? util_dynarray_pop(&device->g17p_vdm_pool,
                             struct d12_g17p_vdm_storage *)
         : NULL;
   simple_mtx_unlock(&device->g17p_vdm_pool_lock);
   if (storage) {
      storage->refcount = 1;
      storage->used = 0;
      return storage;
   }

   storage = calloc(1, sizeof(*storage));
   if (!storage)
      return NULL;
   storage->device = device;
   storage->refcount = 1;
   /* Same ordinary coherent mapping as Honeykrisp's native VDM slabs. */
   storage->bo = agx_bo_create(dev, D12_G17P_VDM_SLAB_SIZE, 0, 0,
                               "d3d12 native VDM payload");
   if (!storage->bo || !agx_bo_map(storage->bo))
      goto fail;
   simple_mtx_lock(&dev->vma_lock);
   storage->alias = util_vma_heap_alloc(&dev->g17p_vdm_alias_heap,
                                      storage->bo->size, AIL_PAGESIZE);
   simple_mtx_unlock(&dev->vma_lock);
   if (!storage->alias ||
       !agx_g17p_vdm_alias_range_valid(storage->alias, storage->bo->size) ||
       agx_bo_bind(dev, storage->bo, storage->alias, storage->bo->size, 0,
                    DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_WRITE))
      goto fail;
   return storage;
fail:
   if (storage->alias) {
      simple_mtx_lock(&dev->vma_lock);
      util_vma_heap_free(&dev->g17p_vdm_alias_heap, storage->alias,
                         storage->bo->size);
      simple_mtx_unlock(&dev->vma_lock);
   }
   if (storage->bo)
      agx_bo_unreference(dev, storage->bo);
   free(storage);
   return NULL;
}

void *
d12_list_alloc_g17p_vdm(struct d12_cmd_list *list, size_t size,
                       size_t alignment, uint64_t *compact_va)
{
   if (compact_va)
      *compact_va = 0;
   if (!list || !compact_va || list->device->dev.chip != AGX_CHIP_G17P ||
       !size || size > D12_G17P_VDM_SLAB_SIZE || !alignment ||
       alignment > D12_G17P_VDM_SLAB_SIZE || (alignment & (alignment - 1)))
      goto fail;
   struct d12_g17p_vdm_storage *storage = NULL;
   const unsigned count = util_dynarray_num_elements(
      &list->g17p_vdm_storage, struct d12_g17p_vdm_storage *);
   if (count)
      storage = util_dynarray_element(&list->g17p_vdm_storage,
                                      struct d12_g17p_vdm_storage *, count - 1)[0];
   size_t offset = storage ? ALIGN_POT(storage->used, alignment) : 0;
   if (!storage || size > storage->bo->size - offset) {
      storage = d12_g17p_vdm_storage_create(list->device);
      if (!storage)
         goto fail;
      struct d12_g17p_vdm_storage **slot = util_dynarray_grow(
         &list->g17p_vdm_storage, struct d12_g17p_vdm_storage *, 1);
      if (!slot) {
         d12_g17p_vdm_storage_unref(storage);
         goto fail;
      }
      *slot = storage;
      offset = 0;
   }
   uint8_t *map = agx_bo_map(storage->bo);
   if (!map)
      goto fail;
   *compact_va = storage->alias - AGX_G17P_VDM_USER_BASE + offset;
   storage->used = offset + size;
   return map + offset;
fail:
   if (list)
      list->state = D12_LIST_ERROR;
   return NULL;
}
