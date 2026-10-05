/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_cmd_buffer.h"
#include "hk_device.h"

#include "asahi/lib/agx_bo.h"
#include "asahi/lib/agx_device.h"
#include "asahi/lib/agx_ioto_program_core.h"

#include "util/log.h"
#include "util/u_dynarray.h"

#include <inttypes.h>
#include <string.h>

void
hk_ioto_arena_finish(struct hk_cmd_buffer *cmd)
{
   struct hk_ioto_arena *arena = &cmd->ioto;

   memset(arena, 0, sizeof(*arena));
}

/*
 * Start a new chunk. The old one keeps whatever it already recorded -- those
 * addresses are still live in the recorded stream -- and stays owned by
 * large_bos.
 */
static bool
hk_ioto_arena_grow(struct hk_cmd_buffer *cmd, size_t size, const char *label)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_ioto_arena *arena = &cmd->ioto;

   const size_t chunk =
      hk_ioto_arena_chunk_size(size, AGX_IOTO_PROGRAM_ALIGNMENT);
   if (chunk == 0)
      return false;

   struct agx_bo *bo =
      agx_bo_create(&dev->dev, chunk, AGX_IOTO_PROGRAM_ALIGNMENT,
                    AGX_BO_EXEC | AGX_BO_LOW_VA, "G17P Ioto program arena");
   if (!bo)
      return false;

   void *map = agx_bo_map(bo);
   if (!map || !bo->va) {
      agx_bo_unreference(&dev->dev, bo);
      return false;
   }

   /* The chunk is retained exactly like the per-program BOs it replaces. */
   util_dynarray_append(&cmd->large_bos, bo);

   arena->bo = bo;
   arena->map = map;
   arena->base = bo->va->addr;
   arena->cursor = 0;
   arena->capacity = bo->size > UINT32_MAX ? UINT32_MAX : (uint32_t)bo->size;
   arena->chunk_bytes += bo->size;

   return true;
}

static_assert(HK_IOTO_PROGRAM_ALIGNMENT == AGX_IOTO_PROGRAM_ALIGNMENT,
              "inline Ioto reservation alignment");

bool
hk_cmd_reserve_ioto_program_slow(struct hk_cmd_buffer *cmd, size_t size,
                                 const char *label,
                                 struct hk_ioto_reservation *reservation)
{
   struct hk_ioto_arena *arena = &cmd->ioto;
   struct hk_ioto_arena_span span;

   if (!reservation || size == 0) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return false;
   }

   if (!arena->map ||
       !hk_ioto_arena_place(arena->cursor, arena->capacity, size,
                            AGX_IOTO_PROGRAM_ALIGNMENT, &span)) {
      if (!hk_ioto_arena_grow(cmd, size, label) ||
          !hk_ioto_arena_place(arena->cursor, arena->capacity, size,
                               AGX_IOTO_PROGRAM_ALIGNMENT, &span)) {
         /* Name what ran out. A bare VK_ERROR_OUT_OF_DEVICE_MEMORY from here
          * is indistinguishable from a texture allocation failing, and the
          * count is the number this recording actually needed. */
         mesa_loge("G17P Ioto arena exhausted requesting %zu bytes for %s "
                   "after %u programs (%" PRIu64 " program bytes "
                   "in %" PRIu64 " bytes of USC window)",
                   size, label ? label : "(program)", arena->programs,
                   arena->program_bytes, arena->chunk_bytes);
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         return false;
      }
   }

   *reservation = (struct hk_ioto_reservation){
      .map = arena->map + span.offset,
      .va = arena->base + span.offset,
   };
   arena->cursor = span.next_offset;
   return true;
}

uint64_t
hk_cmd_publish_ioto_program(struct hk_cmd_buffer *cmd,
                            const struct hk_ioto_reservation *reservation,
                            size_t length)
{
   struct hk_ioto_arena *arena = &cmd->ioto;

   if (!reservation || !reservation->map || length == 0 ||
       length > UINT32_MAX) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return 0;
   }

   assert(reservation->va >= arena->base);
   const uint64_t offset = reservation->va - arena->base;
   assert(offset <= arena->capacity && length <= arena->capacity - offset);

   arena->programs++;

   /* Give back everything the reservation over-reserved: sites reserve the
    * composer's bound and compose something shorter. */
   arena->cursor = (uint32_t)offset + (uint32_t)length;

   if (agx_bo_sync_for_device(arena->bo, offset, length)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_DEVICE_LOST);
      return 0;
   }
   arena->program_bytes += length;
   return reservation->va;
}
