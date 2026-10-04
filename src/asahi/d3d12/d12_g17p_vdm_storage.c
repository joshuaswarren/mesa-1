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

void
d12_g17p_vdm_storage_unref(struct d12_g17p_vdm_storage *storage)
{
   if (!storage || p_atomic_dec_return(&storage->refcount))
      return;
   struct agx_device *dev = storage->dev;
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
d12_list_g17p_vdm_storage_reset(struct d12_cmd_list *list)
{
   util_dynarray_foreach(&list->g17p_vdm_storage,
                         struct d12_g17p_vdm_storage *, storage)
      d12_g17p_vdm_storage_unref(*storage);
   util_dynarray_clear(&list->g17p_vdm_storage);
}

static struct d12_g17p_vdm_storage *
d12_g17p_vdm_storage_create(struct agx_device *dev)
{
   struct d12_g17p_vdm_storage *storage = calloc(1, sizeof(*storage));
   if (!storage)
      return NULL;
   storage->dev = dev;
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
      storage = d12_g17p_vdm_storage_create(&list->device->dev);
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
   *compact_va = storage->alias - AGX_G17P_VDM_USER_BASE + offset;
   storage->used = offset + size;
   return (uint8_t *)agx_bo_map(storage->bo) + offset;
fail:
   if (list)
      list->state = D12_LIST_ERROR;
   return NULL;
}
