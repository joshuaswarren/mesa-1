/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_cmd_buffer.h"
#include "hk_device.h"

#include "asahi/lib/agx_bo.h"
#include "asahi/lib/agx_device.h"
#include "asahi/lib/agx_ioto_program_core.h"

#include "util/hash_table.h"
#include "util/log.h"
#include "util/ralloc.h"
#include "util/u_dynarray.h"

#include <inttypes.h>
#include <string.h>

/* A published program, keyed by the bytes themselves. The bytes live in the
 * arena chunk that backs the program, which outlives this table. */
struct hk_ioto_published {
   const uint8_t *bytes;
   uint32_t length;
};

static uint32_t
hk_ioto_published_hash(const void *key)
{
   const struct hk_ioto_published *published = key;
   return _mesa_hash_data(published->bytes, published->length);
}

static bool
hk_ioto_published_equal(const void *a, const void *b)
{
   const struct hk_ioto_published *x = a, *y = b;

   return x->length == y->length && !memcmp(x->bytes, y->bytes, x->length);
}

void
hk_ioto_arena_finish(struct hk_cmd_buffer *cmd)
{
   struct hk_ioto_arena *arena = &cmd->ioto;

   if (arena->programs && hk_g17p_verbose_enabled()) {
      fprintf(stderr,
              "asahi: G17P Ioto programs: %u composed, %u distinct, "
              "%" PRIu64 " program bytes in %" PRIu64 " bytes of USC window\n",
              arena->programs, arena->distinct, arena->program_bytes,
              arena->chunk_bytes);
   }

   if (arena->published)
      _mesa_hash_table_destroy(arena->published, NULL);

   memset(arena, 0, sizeof(*arena));
}

/*
 * Start a new chunk. The old one keeps whatever it already published -- those
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

   if (hk_g17p_verbose_enabled()) {
      fprintf(stderr,
              "asahi: G17P Ioto arena chunk %#llx size %#llx for %s\n",
              (unsigned long long)arena->base,
              (unsigned long long)bo->size, label ? label : "(program)");
   }

   return true;
}

bool
hk_cmd_reserve_ioto_program(struct hk_cmd_buffer *cmd, size_t size,
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
                   "after %u programs (%u distinct, %" PRIu64 " program bytes "
                   "in %" PRIu64 " bytes of USC window)",
                   size, label ? label : "(program)", arena->programs,
                   arena->distinct, arena->program_bytes, arena->chunk_bytes);
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         return false;
      }
   }

   *reservation = (struct hk_ioto_reservation){
      .map = arena->map + span.offset,
      .va = arena->base + span.offset,
      .rollback = arena->cursor,
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

   const struct hk_ioto_published key = {
      .bytes = reservation->map,
      .length = (uint32_t)length,
   };

   if (arena->published) {
      struct hash_entry *entry =
         _mesa_hash_table_search(arena->published, &key);
      if (entry) {
         /* Identical bytes: hand back the address that already holds them and
          * release the space this reservation took. */
         arena->cursor = reservation->rollback;
         return (uint64_t)(uintptr_t)entry->data;
      }
   } else {
      arena->published = _mesa_hash_table_create(
         NULL, hk_ioto_published_hash, hk_ioto_published_equal);
      if (!arena->published) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
         return 0;
      }
   }

   if (agx_bo_sync_for_device(arena->bo, offset, length)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_DEVICE_LOST);
      return 0;
   }

   struct hk_ioto_published *stored =
      ralloc(arena->published, struct hk_ioto_published);
   if (!stored) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
      return 0;
   }

   *stored = key;
   _mesa_hash_table_insert(arena->published, stored,
                           (void *)(uintptr_t)reservation->va);

   arena->distinct++;
   arena->program_bytes += length;
   return reservation->va;
}
