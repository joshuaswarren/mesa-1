/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_vdm_state_update_core.h"

#include "agx_vdm_state_core.h"

#include <string.h>

static void
agx_vdm_state_update_store_le32(uint8_t *dst, uint32_t value)
{
   dst[0] = (uint8_t)value;
   dst[1] = (uint8_t)(value >> 8);
   dst[2] = (uint8_t)(value >> 16);
   dst[3] = (uint8_t)(value >> 24);
}

static uint32_t
agx_vdm_state_update_load_le32(const uint8_t *src)
{
   return (uint32_t)src[0] | ((uint32_t)src[1] << 8) |
          ((uint32_t)src[2] << 16) | ((uint32_t)src[3] << 24);
}

bool
agx_vdm_state_update_supported(enum agx_chip chip)
{
   return agx_vdm_state_vertex_pipeline_pair_supported(chip);
}

bool
agx_vdm_state_update_resolve_pipeline_va(uint64_t shader_base,
                                         uint32_t packed_usc,
                                         uint64_t *pipeline_va)
{
   if (pipeline_va)
      *pipeline_va = 0;

   if (!pipeline_va || (shader_base & UINT64_C(0x3f)) != 0 ||
       (packed_usc & UINT32_C(0x3f)) != 0 ||
       shader_base > UINT64_MAX - packed_usc)
      return false;

   const uint64_t candidate = shader_base + packed_usc;
   if (candidate >= (UINT64_C(1) << 48))
      return false;

   *pipeline_va = candidate;
   return true;
}

bool
agx_vdm_state_update_pack(enum agx_chip chip,
                          const struct agx_vdm_state_update_info *info,
                          struct agx_vdm_state_update *out)
{
   struct agx_vdm_state_vertex_pipeline_pair pair;

   if (out)
      memset(out, 0, sizeof(*out));

   if (!out || !info || !agx_vdm_state_vertex_pipeline_pair_pack(
                           chip, info->pipeline, info->raw_pipeline_template,
                           &pair))
      return false;

   _Static_assert(sizeof(pair.bytes) == 8,
                  "AGX3 VDM pipeline pair must occupy two words");
   agx_vdm_state_update_store_le32(&out->bytes[0],
                                   AGX_VDM_STATE_UPDATE_HEADER);
   agx_vdm_state_update_store_le32(&out->bytes[4],
                                   info->raw_vertex_shader_word0);
   memcpy(&out->bytes[8], pair.bytes, sizeof(pair.bytes));
   agx_vdm_state_update_store_le32(&out->bytes[16],
                                   info->raw_vertex_outputs[0]);
   agx_vdm_state_update_store_le32(&out->bytes[20],
                                   info->raw_vertex_outputs[1]);
   agx_vdm_state_update_store_le32(&out->bytes[24],
                                   info->raw_vertex_unknown);
   out->length = AGX_VDM_STATE_UPDATE_LENGTH;
   return true;
}

bool
agx_vdm_state_update_unpack(enum agx_chip chip, const void *bytes, size_t size,
                            struct agx_vdm_state_update_info *info)
{
   if (info)
      memset(info, 0, sizeof(*info));

   if (!bytes || !info || size != AGX_VDM_STATE_UPDATE_LENGTH ||
       !agx_vdm_state_update_supported(chip))
      return false;

   const uint8_t *raw = bytes;
   if (agx_vdm_state_update_load_le32(&raw[0]) !=
       AGX_VDM_STATE_UPDATE_HEADER)
      return false;

   if (!agx_vdm_state_vertex_pipeline_pair_unpack(
          chip, &raw[8], AGX_VDM_STATE_VERTEX_PIPELINE_PAIR_LENGTH,
          &info->pipeline, &info->raw_pipeline_template)) {
      memset(info, 0, sizeof(*info));
      return false;
   }

   info->raw_vertex_shader_word0 = agx_vdm_state_update_load_le32(&raw[4]);
   info->raw_vertex_outputs[0] = agx_vdm_state_update_load_le32(&raw[16]);
   info->raw_vertex_outputs[1] = agx_vdm_state_update_load_le32(&raw[20]);
   info->raw_vertex_unknown = agx_vdm_state_update_load_le32(&raw[24]);
   return true;
}

static bool
agx_vdm_state_update_is_hal300(enum agx_chip chip)
{
   switch (chip) {
   case AGX_CHIP_G17G:
   case AGX_CHIP_G17S:
   case AGX_CHIP_G17C:
   case AGX_CHIP_G18G:
      return true;
   default:
      return false;
   }
}

bool
agx_vdm_state_update_template_materialize(
   enum agx_chip chip, const struct agx_vdm_state_update_template_info *info,
   struct agx_vdm_state_update_info *raw)
{
   if (raw)
      memset(raw, 0, sizeof(*raw));

   if (!raw || !info || !agx_vdm_state_update_supported(chip) ||
       info->raw_state_load_count_12 > UINT16_C(0x0fff) ||
       info->raw_state_load_count_18 > UINT32_C(0x0003ffff) ||
       info->variant < AGX_VDM_STATE_UPDATE_TEMPLATE_00 ||
       info->variant > AGX_VDM_STATE_UPDATE_TEMPLATE_21)
      return false;

   const bool hal300 = agx_vdm_state_update_is_hal300(chip);
   if (hal300 && (info->pre_hal300_flag_group_nonzero ||
                  info->pre_hal300_flag_bit_4))
      return false;

   uint32_t vertex_shader_word0 =
      (uint32_t)info->raw_state_load_count_12 << 8;

   if (!hal300 && (!info->pre_hal300_flag_group_nonzero ||
                   info->pre_hal300_flag_bit_4))
      vertex_shader_word0 |= UINT32_C(1) << 31;

   const uint32_t output_count_1 =
      ((uint32_t)info->raw_vs_output_count_0 +
       (uint32_t)info->raw_vs_output_count_2 +
       (info->raw_vs_output_count_1_extra ? 1u : 0u)) &
      UINT32_C(0xff);

   const uint32_t vertex_outputs_word0 =
      (uint32_t)info->raw_vs_output_count_0 | (output_count_1 << 8) |
      ((uint32_t)info->raw_vs_output_count_2 << 16);

   static const uint8_t pre_hal300_variant_bits[6] = {
      0x30, 0x00, 0x31, 0x01, 0x32, 0x02,
   };
   static const uint8_t hal300_variant_bits[6] = {
      0x20, 0x00, 0x21, 0x01, 0x22, 0x02,
   };
   const uint8_t *variant_bits =
      hal300 ? hal300_variant_bits : pre_hal300_variant_bits;

   const uint32_t vertex_outputs_word1 =
      ((uint32_t)info->raw_vs_output_selector << 2) |
      (info->raw_vs_output_flag_6 ? UINT32_C(0x40) : 0) |
      variant_bits[info->variant];

   const struct agx_vdm_state_update_info candidate = {
      .raw_vertex_shader_word0 = vertex_shader_word0,
      .pipeline = info->pipeline,
      .raw_pipeline_template = info->raw_state_load_count_18 << 3,
      .raw_vertex_outputs = {vertex_outputs_word0, vertex_outputs_word1},
      .raw_vertex_unknown =
         info->raw_active_context_selector == UINT32_MAX
            ? 0
            : (info->raw_active_context_selector - 1) & UINT32_C(0x7),
   };
   struct agx_vdm_state_update validation;

   /* Reuse the raw packer's address and target validation. */
   if (!agx_vdm_state_update_pack(chip, &candidate, &validation))
      return false;

   *raw = candidate;
   return true;
}

bool
agx_vdm_state_update_template_pack(
   enum agx_chip chip, const struct agx_vdm_state_update_template_info *info,
   struct agx_vdm_state_update *out)
{
   struct agx_vdm_state_update_info raw;

   if (out)
      memset(out, 0, sizeof(*out));

   return out &&
          agx_vdm_state_update_template_materialize(chip, info, &raw) &&
          agx_vdm_state_update_pack(chip, &raw, out);
}

bool
agx_vdm_state_update_draw_pack(
   enum agx_chip chip, const struct agx_vdm_state_update_draw_info *info,
   struct agx_vdm_state_update *out)
{
   if (out)
      memset(out, 0, sizeof(*out));

   if (!out || !info ||
       info->source_layout !=
          AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT ||
       info->raw_provoking_selector > 2)
      return false;

   switch (info->draw_path) {
   case AGX_VDM_STATE_UPDATE_DRAW_DIRECT_NONINDEXED:
   case AGX_VDM_STATE_UPDATE_DRAW_DIRECT_INDEXED:
   case AGX_VDM_STATE_UPDATE_DRAW_INDIRECT_NONINDEXED:
   case AGX_VDM_STATE_UPDATE_DRAW_INDIRECT_INDEXED:
      break;
   default:
      return false;
   }

   struct agx_vdm_state_update_template_info selected = info->template_info;
   selected.variant =
      (enum agx_vdm_state_update_template_variant)(
         (info->rasterization_enabled ? 1u : 0u) |
         ((uint32_t)info->raw_provoking_selector << 1));

   return agx_vdm_state_update_template_pack(chip, &selected, out);
}
