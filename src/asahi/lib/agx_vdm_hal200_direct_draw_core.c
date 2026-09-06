/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_vdm_hal200_direct_draw_core.h"
#include "agx_vdm_index_list_core.h"

#include <string.h>

static void
store_le32(uint8_t *dst, uint32_t value)
{
   dst[0] = (uint8_t)value;
   dst[1] = (uint8_t)(value >> 8);
   dst[2] = (uint8_t)(value >> 16);
   dst[3] = (uint8_t)(value >> 24);
}

static uint32_t
load_le32(const uint8_t *src)
{
   return (uint32_t)src[0] | ((uint32_t)src[1] << 8) |
          ((uint32_t)src[2] << 16) | ((uint32_t)src[3] << 24);
}

static bool
primitive_supported(uint8_t primitive)
{
   switch (primitive) {
   case 0:
   case 1:
   case 3:
   case 5:
   case 6:
   case 9:
   case 11:
   case 13:
   case 14:
   case 15:
      return true;
   default:
      return false;
   }
}

bool
agx_vdm_hal200_direct_draw_pack(
   enum agx_chip chip, const struct agx_vdm_hal200_direct_draw_fields *fields,
   struct agx_vdm_hal200_direct_draw *out)
{
   if (out)
      memset(out, 0, sizeof(*out));

   if (!out || !fields || chip != AGX_CHIP_G17P ||
       fields->draw_state_context_offset == 0 ||
       (fields->draw_state_context_offset & UINT32_C(3)) != 0 ||
       !primitive_supported(fields->primitive) || fields->vertex_count == 0 ||
       fields->instance_count == 0)
      return false;

   store_le32(&out->bytes[0], fields->draw_state_context_offset);
   store_le32(&out->bytes[4],
              (AGX_VDM_HAL200_DIRECT_DRAW_OPCODE << 16) |
                 ((uint32_t)fields->primitive << 8));
   store_le32(&out->bytes[8], fields->vertex_count);
   store_le32(&out->bytes[12], fields->instance_count);
   store_le32(&out->bytes[16], fields->vertex_start);
   out->length = AGX_VDM_HAL200_DIRECT_DRAW_LENGTH;
   return true;
}

bool
agx_vdm_hal200_direct_draw_unpack(
   enum agx_chip chip, const void *bytes, size_t size,
   struct agx_vdm_hal200_direct_draw_fields *fields)
{
   if (fields)
      memset(fields, 0, sizeof(*fields));

   if (!bytes || !fields || chip != AGX_CHIP_G17P ||
       size != AGX_VDM_HAL200_DIRECT_DRAW_LENGTH)
      return false;

   const uint8_t *raw = bytes;
   const uint32_t header = load_le32(&raw[4]);
   const uint32_t offset = load_le32(&raw[0]);
   const uint8_t primitive = (uint8_t)(header >> 8);
   if (offset == 0 || (offset & 3) != 0 ||
       (header & UINT32_C(0xffff00ff)) !=
          (AGX_VDM_HAL200_DIRECT_DRAW_OPCODE << 16) ||
       !primitive_supported(primitive) || load_le32(&raw[8]) == 0 ||
       load_le32(&raw[12]) == 0)
      return false;

   fields->draw_state_context_offset = offset;
   fields->primitive = primitive;
   fields->vertex_count = load_le32(&raw[8]);
   fields->instance_count = load_le32(&raw[12]);
   fields->vertex_start = load_le32(&raw[16]);
   return true;
}

bool
agx_vdm_hal200_draw_emit(enum agx_chip chip, const void *packed,
                         size_t packed_size, bool render_state_emitted,
                         void *stream, size_t stream_capacity,
                         size_t *emitted_size)
{
   if (emitted_size)
      *emitted_size = 0;

   if (chip != AGX_CHIP_G17P || !packed || !stream || !emitted_size ||
       (packed_size != AGX_VDM_HAL200_DIRECT_DRAW_LENGTH &&
        packed_size != AGX_VDM_HAL200_INDEXED_DRAW_LENGTH))
      return false;

   const size_t skip = render_state_emitted
                          ? 0
                          : AGX_VDM_HAL200_DRAW_STATE_POINTER_LENGTH;
   const size_t size = packed_size - skip;
   if (stream_capacity < size)
      return false;

   memcpy(stream, (const uint8_t *)packed + skip, size);
   *emitted_size = size;
   return true;
}

bool
agx_vdm_hal200_index_range(uint64_t address, uint64_t range_B,
                           uint32_t first_index, uint32_t count,
                           uint8_t index_size_B, uint64_t *relocated,
                           uint32_t *size_word)
{
   if (relocated)
      *relocated = 0;
   if (size_word)
      *size_word = 0;

   const uint64_t limit = UINT64_C(1) << 48;
   const uint64_t offset_B = (uint64_t)first_index * index_size_B;
   const uint64_t count_B = (uint64_t)count * index_size_B;
   if (!relocated || !size_word || !address ||
       (index_size_B != 1 && index_size_B != 2 && index_size_B != 4) ||
       (address & (index_size_B - 1)) ||
       address >= limit || !range_B || range_B > UINT32_MAX ||
       range_B > limit - address || !count || offset_B >= range_B ||
       count_B > range_B - offset_B)
      return false;

   /* Native direct indexed 0x47e57c..0x47e584 rounds the supplied byte
    * length up to dwords, then 0x47e664 subtracts one. Keep arithmetic 64-bit
    * so firstIndex and a near-4GiB range cannot wrap before validation. */
   *relocated = address + offset_B;
   *size_word = (uint32_t)(((range_B - offset_B + 3) >> 2) - 1);
   return true;
}

bool
agx_vdm_hal200_u16_index_range(uint64_t address, uint64_t range_B,
                               uint32_t first_index, uint32_t count,
                               uint64_t *relocated, uint32_t *size_word)
{
   return agx_vdm_hal200_index_range(address, range_B, first_index, count,
                                    2, relocated, size_word);
}

bool
agx_vdm_hal200_indexed_draw_pack(
   enum agx_chip chip, const struct agx_vdm_hal200_indexed_draw_fields *fields,
   struct agx_vdm_hal200_indexed_draw *out)
{
   if (out)
      memset(out, 0, sizeof(*out));

   uint64_t address;
   uint32_t size_word;
   if (!out || !fields || chip != AGX_CHIP_G17P ||
       !fields->draw_state_context_offset ||
       (fields->draw_state_context_offset & 3) || fields->instance_count == 0 ||
       !primitive_supported(fields->primitive) ||
       !agx_vdm_hal200_index_range(
          fields->index_buffer, fields->index_buffer_range_B,
          fields->first_index, fields->index_count, fields->index_size_B,
          &address, &size_word))
      return false;

   struct agx_vdm_index_list_direct_words packet;
   if (!agx_vdm_index_list_direct_pack_words(
          chip, fields->primitive, true, fields->restart_enable,
          fields->index_size_B == 1 ? AGX_VDM_INDEX_LIST_INDEX_SIZE_U8
          : fields->index_size_B == 2 ? AGX_VDM_INDEX_LIST_INDEX_SIZE_U16
                                    : AGX_VDM_INDEX_LIST_INDEX_SIZE_U32, address,
          fields->index_count, fields->instance_count,
          (uint32_t)fields->base_vertex, size_word, &packet) || packet.count != 7)
      return false;

   /* 0x47e618..0x47e678: restart state precedes every native indexed draw,
    * even with restart disabled; baseVertex is copied as a raw signed word. */
   store_le32(out->bytes, fields->draw_state_context_offset);
   store_le32(out->bytes + 4, UINT32_C(0x40000001));
   const uint32_t restart_max = fields->index_size_B == 1 ? UINT32_C(0xff)
                                 : fields->index_size_B == 2 ? UINT32_C(0xffff)
                                                            : UINT32_MAX;
   store_le32(out->bytes + 8, fields->restart_index < restart_max
                                 ? fields->restart_index : restart_max);
   for (unsigned i = 0; i < packet.count; ++i)
      store_le32(out->bytes + 12 + 4 * i, packet.words[i]);
   out->length = AGX_VDM_HAL200_INDEXED_DRAW_LENGTH;
   return true;
}
