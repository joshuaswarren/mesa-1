/*
 * Copyright 2024 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

#include "agx_bo.h"
#include "agx_device.h"
#include "agx_va_layout.h"

static struct util_vma_heap *
agx_vma_heap(struct agx_device *dev, enum agx_va_flags flags)
{
   return (flags & AGX_VA_USC) ? &dev->usc_heap : &dev->main_heap;
}

/*
 * Sparse buffers map every page twice, read-write at X and read-only at
 * X + ro_offset, so the heap [user_start, user_start + size) is split by one
 * address bit: ro_offset is a power of two, every read-write address lies
 * below it (bit clear) and the shadow of the last one still fits below
 * kernel_start. Choose the ro_offset that leaves the largest heap. On a
 * 512 GiB window with the heap at 72 GiB that is 256 GiB and a 152 GiB heap;
 * rounding the window and halving it twice left 64 GiB.
 */
bool
agx_sparse_layout(uint64_t user_start, uint64_t kernel_start,
                  uint64_t *ro_offset, uint64_t *size)
{
   *size = 0;

   for (uint64_t p = util_next_power_of_two64(user_start + 1);
        p && user_start + p < kernel_start; p <<= 1) {
      uint64_t s = MIN2(p - user_start, kernel_start - user_start - p);

      if (s > *size) {
         *size = s;
         *ro_offset = p;
      }
   }

   return *size != 0;
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

   struct util_vma_heap *heap = agx_vma_heap(dev, flags);
   uint64_t addr = 0;

   simple_mtx_lock(&dev->vma_lock);
   if (flags & AGX_VA_FIXED) {
      if (util_vma_heap_alloc_addr(heap, fixed_va, size_B))
         addr = fixed_va;
   } else {
      addr = util_vma_heap_alloc(heap, size_B, align_B);
   }
   simple_mtx_unlock(&dev->vma_lock);

   if (addr == 0)
      return NULL;

   struct agx_va *va = malloc(sizeof(struct agx_va));
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
      agx_bo_bind(dev, NULL, va->addr, va->size_B, 0, DRM_ASAHI_BIND_UNBIND);
   }

   struct util_vma_heap *heap = agx_vma_heap(dev, va->flags);

   simple_mtx_lock(&dev->vma_lock);
   util_vma_heap_free(heap, va->addr, va->size_B);
   simple_mtx_unlock(&dev->vma_lock);
   free(va);
}
