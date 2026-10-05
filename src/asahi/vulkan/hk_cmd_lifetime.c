/* SPDX-License-Identifier: MIT */
#include "hk_cmd_buffer.h"


static inline uint32_t
hk_retained_set_hash(const struct agx_bo *bo)
{
   /* Fibonacci hashing of the pointer: allocator objects are adjacent, so
    * mix all bits before taking the top ones. */
   const uint64_t key = (uint64_t)(uintptr_t)bo;
   return (uint32_t)((key * UINT64_C(0x9e3779b97f4a7c15)) >> 32);
}

static bool
hk_retained_set_grow(struct hk_cmd_buffer *cmd)
{
   const uint32_t capacity =
      cmd->retained_set_capacity ? cmd->retained_set_capacity * 2 : 256;
   struct agx_bo **set = calloc(capacity, sizeof(*set));
   if (!set)
      return false;

   for (uint32_t i = 0; i < cmd->retained_set_capacity; ++i) {
      struct agx_bo *bo = cmd->retained_set[i];
      if (!bo)
         continue;
      uint32_t slot = hk_retained_set_hash(bo) & (capacity - 1);
      while (set[slot])
         slot = (slot + 1) & (capacity - 1);
      set[slot] = bo;
   }

   free(cmd->retained_set);
   cmd->retained_set = set;
   cmd->retained_set_capacity = capacity;
   return true;
}

void
hk_cmd_buffer_retain_bo(struct hk_cmd_buffer *cmd, struct agx_bo *bo)
{
   if (!bo)
      return;

   /* Pointer identity is sufficient: the retained reference prevents reuse
    * until reset. Mix two address regions so adjacent allocator objects do not
    * collapse into one direct-mapped slot. The table size is a power of two. */
   const uintptr_t key = (uintptr_t)bo;
   const unsigned cache_index =
      ((key >> 4) ^ (key >> 12)) & (ARRAY_SIZE(cmd->recent_retained_bos) - 1);
   if (cmd->recent_retained_bos[cache_index] == bo)
      return;

   /* Keep the load factor at or below one half. */
   if ((cmd->retained_set_count + 1) * 2 > cmd->retained_set_capacity &&
       !hk_retained_set_grow(cmd)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
      return;
   }

   const uint32_t mask = cmd->retained_set_capacity - 1;
   uint32_t slot = hk_retained_set_hash(bo) & mask;
   while (cmd->retained_set[slot]) {
      if (cmd->retained_set[slot] == bo) {
         cmd->recent_retained_bos[cache_index] = bo;
         return;
      }
      slot = (slot + 1) & mask;
   }

   agx_bo_reference(bo);
   util_dynarray_append(&cmd->large_bos, bo);
   cmd->retained_set[slot] = bo;
   cmd->retained_set_count++;
   cmd->recent_retained_bos[cache_index] = bo;
}

void
hk_cmd_buffer_release_bos(struct hk_cmd_buffer *cmd, struct agx_device *dev)
{
   util_dynarray_foreach(&cmd->large_bos, struct agx_bo *, bo)
      agx_bo_unreference(dev, *bo);

   util_dynarray_clear(&cmd->large_bos);
   memset(cmd->recent_retained_bos, 0, sizeof(cmd->recent_retained_bos));
   if (cmd->retained_set_count) {
      memset(cmd->retained_set, 0,
             cmd->retained_set_capacity * sizeof(*cmd->retained_set));
      cmd->retained_set_count = 0;
   }
}
