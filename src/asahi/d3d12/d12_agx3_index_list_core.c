/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "d12_agx3_index_list_core.h"

#include "asahi/lib/agx_vdm_index_list_core.h"
#include "asahi/lib/agx_abi.h"

#include <string.h>

bool
d12_agx3_encode_native_direct_draw(
   enum agx_chip chip, const struct d12_agx3_native_direct_draw_info *info,
   struct d12_agx3_native_direct_draw *out)
{
   if (out)
      memset(out, 0, sizeof(*out));
   if (!out || !info)
      return false;

   if (info->indexed) {
      /* A null D3D index binding fetches zero for every index. Neither the
       * API index start nor signed vertex bias applies to that zero page. */
      const bool bound = info->index_buffer != 0;
      const struct agx_vdm_hal200_indexed_draw_fields fields = {
         .draw_state_context_offset = info->draw_state_context_offset,
         .primitive = info->primitive,
         .index_size_B = bound ? info->index_size_B : 2,
         .index_buffer = bound ? info->index_buffer : AGX_ZERO_PAGE_ADDRESS,
         .index_buffer_range_B = bound ? info->index_buffer_range_B : 4,
         .first_index = bound ? info->first_index : 0,
         .base_vertex = bound ? (int32_t)info->first_vertex : 0,
         .index_count = info->count,
         .instance_count = info->instances,
      };
      struct agx_vdm_hal200_indexed_draw encoded;
      if (!agx_vdm_hal200_indexed_draw_pack(chip, &fields, &encoded))
         return false;
      memcpy(out->bytes, encoded.bytes, encoded.length);
      out->length = encoded.length;
   } else {
      const struct agx_vdm_hal200_direct_draw_fields fields = {
         .draw_state_context_offset = info->draw_state_context_offset,
         .primitive = info->primitive,
         .vertex_count = info->count,
         .instance_count = info->instances,
         .vertex_start = info->first_vertex,
      };
      struct agx_vdm_hal200_direct_draw encoded;
      if (!agx_vdm_hal200_direct_draw_pack(chip, &fields, &encoded))
         return false;
      memcpy(out->bytes, encoded.bytes, encoded.length);
      out->length = encoded.length;
   }
   return true;
}

bool
d12_agx3_encode_direct_nonindexed_index_list(
   enum agx_chip chip,
   const struct d12_agx3_direct_nonindexed_index_list_info *info,
   struct d12_agx3_direct_nonindexed_index_list *out)
{
   struct agx_vdm_index_list_direct_nonindexed_info shared_info;
   struct agx_vdm_index_list_direct_nonindexed shared;

   if (out)
      memset(out, 0, sizeof(*out));

   if (!out || !info)
      return false;

   shared_info.primitive = info->primitive;
   shared_info.count = info->count;
   shared_info.instances = info->instances;
   shared_info.start = info->start;

   if (!agx_vdm_index_list_direct_nonindexed_pack(chip, &shared_info,
                                                   &shared))
      return false;

   _Static_assert(sizeof(out->bytes) == sizeof(shared.bytes),
                  "D3D and shared non-indexed storage must agree");
   memcpy(out->bytes, shared.bytes, sizeof(out->bytes));
   out->length = shared.length;
   return true;
}

bool
d12_agx3_decode_direct_nonindexed_index_list(
   enum agx_chip chip, const void *bytes, uint32_t size,
   struct d12_agx3_direct_nonindexed_index_list_info *info)
{
   struct agx_vdm_index_list_direct_nonindexed_info shared_info;

   if (info)
      memset(info, 0, sizeof(*info));

   if (!info || !agx_vdm_index_list_direct_nonindexed_unpack(
                   chip, bytes, size, &shared_info))
      return false;

   info->primitive = shared_info.primitive;
   info->count = shared_info.count;
   info->instances = shared_info.instances;
   info->start = shared_info.start;
   return true;
}

bool
d12_agx3_encode_direct_indexed_index_list(
   enum agx_chip chip,
   const struct d12_agx3_direct_indexed_index_list_info *info,
   struct d12_agx3_direct_indexed_index_list *out)
{
   struct agx_vdm_index_list_direct_indexed_info shared_info;
   struct agx_vdm_index_list_direct_indexed shared;

   if (out)
      memset(out, 0, sizeof(*out));

   if (!out || !info)
      return false;

   shared_info.primitive = info->primitive;
   shared_info.restart_enable = info->restart_enable;
   shared_info.index_size =
      (enum agx_vdm_index_list_index_size)info->index_size;
   shared_info.index_buffer = info->index_buffer;
   shared_info.count = info->count;
   shared_info.instances = info->instances;
   shared_info.start = info->start;
   shared_info.index_buffer_size_word = info->index_buffer_size_word;

   if (!agx_vdm_index_list_direct_indexed_pack(chip, &shared_info, &shared))
      return false;

   _Static_assert(sizeof(out->bytes) == sizeof(shared.bytes),
                  "D3D and shared indexed storage must agree");
   memcpy(out->bytes, shared.bytes, sizeof(out->bytes));
   out->length = shared.length;
   return true;
}

bool
d12_agx3_decode_direct_indexed_index_list(
   enum agx_chip chip, const void *bytes, uint32_t size,
   struct d12_agx3_direct_indexed_index_list_info *info)
{
   struct agx_vdm_index_list_direct_indexed_info shared_info;

   if (info)
      memset(info, 0, sizeof(*info));

   if (!info || !agx_vdm_index_list_direct_indexed_unpack(
                   chip, bytes, size, &shared_info))
      return false;

   info->primitive = shared_info.primitive;
   info->restart_enable = shared_info.restart_enable;
   info->index_size = (enum d12_agx3_index_size)shared_info.index_size;
   info->index_buffer = shared_info.index_buffer;
   info->count = shared_info.count;
   info->instances = shared_info.instances;
   info->start = shared_info.start;
   info->index_buffer_size_word = shared_info.index_buffer_size_word;
   return true;
}

bool
d12_agx3_encode_indirect_nonindexed_index_list(
   enum agx_chip chip,
   const struct d12_agx3_indirect_nonindexed_index_list_info *info,
   struct d12_agx3_indirect_nonindexed_index_list *out)
{
   struct agx_vdm_index_list_indirect_nonindexed_info shared_info;
   struct agx_vdm_index_list_indirect_nonindexed shared;

   if (out)
      memset(out, 0, sizeof(*out));

   if (!out || !info)
      return false;

   shared_info.primitive = info->primitive;
   shared_info.indirect_buffer = info->indirect_buffer;

   if (!agx_vdm_index_list_indirect_nonindexed_pack(chip, &shared_info, &shared))
      return false;

   _Static_assert(sizeof(out->bytes) == sizeof(shared.bytes),
                  "D3D and shared indirect non-indexed storage must agree");
   memcpy(out->bytes, shared.bytes, sizeof(out->bytes));
   out->length = shared.length;
   return true;
}

bool
d12_agx3_decode_indirect_nonindexed_index_list(
   enum agx_chip chip, const void *bytes, uint32_t size,
   struct d12_agx3_indirect_nonindexed_index_list_info *info)
{
   struct agx_vdm_index_list_indirect_nonindexed_info shared_info;

   if (info)
      memset(info, 0, sizeof(*info));

   if (!info || !agx_vdm_index_list_indirect_nonindexed_unpack(
                   chip, bytes, size, &shared_info))
      return false;

   info->primitive = shared_info.primitive;
   info->indirect_buffer = shared_info.indirect_buffer;
   return true;
}

bool
d12_agx3_encode_indirect_indexed_index_list(
   enum agx_chip chip,
   const struct d12_agx3_indirect_indexed_index_list_info *info,
   struct d12_agx3_indirect_indexed_index_list *out)
{
   struct agx_vdm_index_list_indirect_indexed_info shared_info;
   struct agx_vdm_index_list_indirect_indexed shared;

   if (out)
      memset(out, 0, sizeof(*out));

   if (!out || !info)
      return false;

   shared_info.primitive = info->primitive;
   shared_info.restart_enable = info->restart_enable;
   shared_info.index_size =
      (enum agx_vdm_index_list_index_size)info->index_size;
   shared_info.index_buffer = info->index_buffer;
   shared_info.indirect_buffer = info->indirect_buffer;
   shared_info.restart_index = info->restart_index;
   shared_info.index_buffer_size_word = info->index_buffer_size_word;

   if (!agx_vdm_index_list_indirect_indexed_pack(chip, &shared_info, &shared))
      return false;

   _Static_assert(sizeof(out->bytes) == sizeof(shared.bytes),
                  "D3D and shared indirect indexed storage must agree");
   memcpy(out->bytes, shared.bytes, sizeof(out->bytes));
   out->length = shared.length;
   return true;
}

bool
d12_agx3_decode_indirect_indexed_index_list(
   enum agx_chip chip, const void *bytes, uint32_t size,
   struct d12_agx3_indirect_indexed_index_list_info *info)
{
   struct agx_vdm_index_list_indirect_indexed_info shared_info;

   if (info)
      memset(info, 0, sizeof(*info));

   if (!info || !agx_vdm_index_list_indirect_indexed_unpack(
                   chip, bytes, size, &shared_info))
      return false;

   info->primitive = shared_info.primitive;
   info->restart_enable = shared_info.restart_enable;
   info->index_size = (enum d12_agx3_index_size)shared_info.index_size;
   info->index_buffer = shared_info.index_buffer;
   info->indirect_buffer = shared_info.indirect_buffer;
   info->restart_index = shared_info.restart_index;
   info->index_buffer_size_word = shared_info.index_buffer_size_word;
   return true;
}
