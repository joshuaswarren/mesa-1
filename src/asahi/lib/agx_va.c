/*
 * Copyright 2024 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

#include "agx_bo.h"
#include "agx_device.h"

static struct util_vma_heap *
agx_vma_heap(struct agx_device *dev, enum agx_va_flags flags)
{
   if (flags & AGX_VA_USC)
      return &dev->usc_heap;
   if (flags & AGX_VA_CAPTURE_REPLAY) {
      assert(dev->replay_heap_initialized);
      return &dev->replay_heap;
   }
   return &dev->main_heap;
}

struct agx_va *
agx_va_alloc(struct agx_device *dev, uint64_t size_B, uint64_t align_B,
             enum agx_va_flags flags, uint64_t fixed_va)
{
   assert((fixed_va != 0) == !!(flags & AGX_VA_FIXED));
   assert((fixed_va % align_B) == 0);

   /* All allocations need a guard at the end to prevent overreads.
    *
    * TODO: Even with soft fault?
    */
   size_B += dev->guard_size;

   /* Allocate bookkeeping before reserving an address so host-memory
    * failure cannot strand a reservation in either VA allocator. */
   struct agx_va *va = malloc(sizeof(*va));
   if (!va)
      return NULL;

   uint64_t addr = 0;

   {
      struct util_vma_heap *heap = agx_vma_heap(dev, flags);
      simple_mtx_lock(&dev->vma_lock);
      if (flags & AGX_VA_FIXED) {
         if (util_vma_heap_alloc_addr(heap, fixed_va, size_B))
            addr = fixed_va;
      } else {
         const bool high = heap->alloc_high;
         if (flags & AGX_VA_LOW)
            heap->alloc_high = false;
         addr = util_vma_heap_alloc(heap, size_B, align_B);
         heap->alloc_high = high;
      }
      simple_mtx_unlock(&dev->vma_lock);
   }

   if (addr == 0) {
      free(va);
      return NULL;
   }

   *va = (struct agx_va){
      .flags = flags,
      .size_B = size_B,
      .addr = addr,
   };
   return va;
}

void
agx_va_free(struct agx_device *dev, struct agx_va *va, bool unbind)
{
   if (!va)
      return;

   if (unbind) {
      /* Sparse/capture-replay VAs may contain a lower read-write mapping and
       * its upper read-only shadow. Clear both, including unbound holes, in
       * one VM_BIND transaction so neither half can survive VA reuse. */
      assert(!(va->flags & AGX_VA_USC));
      assert(va->size_B >= dev->guard_size);
      const uint64_t bound_size_B = va->size_B - dev->guard_size;
      struct drm_asahi_gem_bind_op ops[2] = {
         {
            .flags = DRM_ASAHI_BIND_UNBIND,
            .range = bound_size_B,
            .addr = va->addr,
         },
         {
            .flags = DRM_ASAHI_BIND_UNBIND,
            .range = bound_size_B,
            .addr = agx_rw_addr_to_ro(dev, va->addr),
         },
      };

      /* Do not recycle an address if either stale mapping could remain. */
      if (dev->ops.bo_bind(dev, ops, ARRAY_SIZE(ops)))
         return;
   }

   {
      struct util_vma_heap *heap = agx_vma_heap(dev, va->flags);
      simple_mtx_lock(&dev->vma_lock);
      util_vma_heap_free(heap, va->addr, va->size_B);
      simple_mtx_unlock(&dev->vma_lock);
   }
   free(va);
}
