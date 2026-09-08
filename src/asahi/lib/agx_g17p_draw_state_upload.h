/* SPDX-License-Identifier: MIT */
#pragma once

#include "agx_va_runtime_hole.h"
#include "agx_vdm_hal200_bind_tail_core.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define AGX_G17P_BIND_GROUP_BYTES 0x80u
#define AGX_G17P_VIEWPORT_RECORD_BYTES 40u
#define AGX_G17P_VARYING_RECORD_BYTES 20u
#define AGX_G17P_VARYING_RECORD_OFFSET                                  \
   (AGX_G17P_BIND_GROUP_BYTES + AGX_G17P_VIEWPORT_RECORD_BYTES)
#define AGX_G17P_BIND_STATE_BYTES                                        \
   (AGX_G17P_VARYING_RECORD_OFFSET + AGX_G17P_VARYING_RECORD_BYTES)

struct agx_g17p_bind_state_upload {
   struct agx_vdm_hal200_bind_pair pairs[AGX_VDM_HAL200_BIND_COUNT];
   struct agx_vdm_hal200_bind_tail tail;
};

/* Compact state lives in the retained +8 MiB..+32 MiB VDM aperture. */
static inline bool
agx_g17p_draw_state_range_valid(uint64_t compact, size_t size)
{
   return size != 0 && compact >= AGX_G17P_VDM_ALIAS_OFFSET_START &&
          compact < AGX_G17P_VDM_ALIAS_OFFSET_END && (compact & 3) == 0 &&
          size <= AGX_G17P_VDM_ALIAS_OFFSET_END - compact;
}

static inline bool
agx_g17p_ppp_indices_pack(size_t scissor_bytes, size_t depth_bias_bytes,
                          uint8_t output[8])
{
   /* HAL200 selector bit8 is followed by a scissor index in the low half and
    * a depth-bias index in the high half. The enables live elsewhere. */
   if (!output || scissor_bytes == 0 || depth_bias_bytes == 0 ||
       scissor_bytes % 16 != 0 || depth_bias_bytes % 12 != 0)
      return false;

   const size_t scissor_index = scissor_bytes / 16 - 1;
   const size_t depth_bias_index = depth_bias_bytes / 12 - 1;
   if (scissor_index > UINT16_MAX || depth_bias_index > UINT16_MAX)
      return false;

   const uint8_t record[8] = {
      0x00, 0x01, 0x00, 0x00,
      (uint8_t)scissor_index, (uint8_t)(scissor_index >> 8),
      (uint8_t)depth_bias_index, (uint8_t)(depth_bias_index >> 8),
   };
   memcpy(output, record, sizeof(record));
   return true;
}

static inline bool
agx_g17p_upload_bind_state(void *cpu, size_t capacity, uint64_t compact,
                           const uint8_t *bind_group,
                           const uint8_t *viewport, size_t viewport_size,
                           uint32_t counts_32, uint32_t counts_16,
                           struct agx_g17p_bind_state_upload *out)
{
   if (!cpu || !bind_group || !viewport || !out ||
       capacity < AGX_G17P_BIND_STATE_BYTES ||
       ((counts_32 | counts_16) & UINT32_C(0xff000000)) != 0 ||
       viewport_size != AGX_G17P_VIEWPORT_RECORD_BYTES ||
       !agx_g17p_draw_state_range_valid(compact,
                                         AGX_G17P_BIND_STATE_BYTES))
      return false;

   const uint32_t group = (uint32_t)compact;
   struct agx_g17p_bind_state_upload encoded = {
      .pairs = {
         {group + AGX_G17P_VARYING_RECORD_OFFSET, 0x700},
         {group + 0x00, 0x500},
         {group + 0x1c, 0x700},
         {group + 0x30, 0x500},
         {group + 0x4c, 0xa00},
         {group + AGX_G17P_BIND_GROUP_BYTES, 0x300},
         {group + 0x60, 0x200},
         {group + 0x6c, 0x200},
      },
   };
   if (!agx_vdm_hal200_bind_tail_pack(AGX_CHIP_G17P, encoded.pairs,
                                       &encoded.tail))
      return false;

   memcpy(cpu, bind_group, AGX_G17P_BIND_GROUP_BYTES);
   memcpy((uint8_t *)cpu + AGX_G17P_BIND_GROUP_BYTES, viewport,
          viewport_size);

   /* Native VertexProgramVariant+0x1008: separate f32/f16 varying counts and
    * two auxiliary words, zero for the admitted user-varying layout. */
   const uint32_t words[5] = {
      UINT32_C(0x10040000), counts_32, counts_16, 0, 0,
   };
   uint8_t *varyings = (uint8_t *)cpu + AGX_G17P_VARYING_RECORD_OFFSET;
   for (unsigned i = 0; i < 5; ++i) {
      for (unsigned byte = 0; byte < 4; ++byte)
         varyings[4 * i + byte] = words[i] >> (8 * byte);
   }

   *out = encoded;
   return true;
}

static inline bool
agx_g17p_upload_draw_indices(void *cpu, size_t capacity, uint64_t compact,
                             size_t scissor_bytes, size_t depth_bias_bytes,
                             uint32_t *offset)
{
   uint8_t indices[8];
   if (!cpu || !offset || capacity < sizeof(indices) ||
       !agx_g17p_draw_state_range_valid(compact, sizeof(indices)) ||
       !agx_g17p_ppp_indices_pack(scissor_bytes, depth_bias_bytes, indices))
      return false;

   memcpy(cpu, indices, sizeof(indices));
   *offset = (uint32_t)compact;
   return true;
}
