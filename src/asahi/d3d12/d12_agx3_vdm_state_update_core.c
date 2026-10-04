/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "d12_agx3_vdm_state_update_core.h"

#include "asahi/lib/agx_vdm_state_update_core.h"

#include <string.h>

enum d12_agx3_vdm_state_update_draw_path
d12_agx3_vdm_state_update_classify_draw(bool indirect, bool indexed)
{
   if (indirect)
      return indexed ? D12_AGX3_VDM_STATE_UPDATE_DRAW_INDIRECT_INDEXED
                     : D12_AGX3_VDM_STATE_UPDATE_DRAW_INDIRECT_NONINDEXED;

   return indexed ? D12_AGX3_VDM_STATE_UPDATE_DRAW_DIRECT_INDEXED
                  : D12_AGX3_VDM_STATE_UPDATE_DRAW_DIRECT_NONINDEXED;
}

bool
d12_agx3_encode_vdm_state_update(
   enum agx_chip chip, const struct d12_agx3_vdm_state_update_info *info,
   struct d12_agx3_vdm_state_update *out)
{
   struct agx_vdm_state_update_info shared_info;
   struct agx_vdm_state_update shared;

   if (out)
      memset(out, 0, sizeof(*out));

   if (!out || !info)
      return false;

   shared_info.raw_vertex_shader_word0 = info->raw_vertex_shader_word0;
   shared_info.pipeline = info->pipeline;
   shared_info.raw_pipeline_template = info->raw_pipeline_template;
   shared_info.raw_vertex_outputs[0] = info->raw_vertex_outputs[0];
   shared_info.raw_vertex_outputs[1] = info->raw_vertex_outputs[1];
   shared_info.raw_vertex_unknown = info->raw_vertex_unknown;

   if (!agx_vdm_state_update_pack(chip, &shared_info, &shared))
      return false;

   _Static_assert(sizeof(out->bytes) == sizeof(shared.bytes),
                  "D3D and shared VDM State Update storage must agree");
   memcpy(out->bytes, shared.bytes, sizeof(out->bytes));
   out->length = shared.length;
   return true;
}

bool
d12_agx3_decode_vdm_state_update(
   enum agx_chip chip, const void *bytes, uint32_t size,
   struct d12_agx3_vdm_state_update_info *info)
{
   struct agx_vdm_state_update_info shared_info;

   if (info)
      memset(info, 0, sizeof(*info));

   if (!info || !agx_vdm_state_update_unpack(chip, bytes, size, &shared_info))
      return false;

   info->raw_vertex_shader_word0 = shared_info.raw_vertex_shader_word0;
   info->pipeline = shared_info.pipeline;
   info->raw_pipeline_template = shared_info.raw_pipeline_template;
   info->raw_vertex_outputs[0] = shared_info.raw_vertex_outputs[0];
   info->raw_vertex_outputs[1] = shared_info.raw_vertex_outputs[1];
   info->raw_vertex_unknown = shared_info.raw_vertex_unknown;
   return true;
}

bool
d12_agx3_encode_vdm_state_update_template(
   enum agx_chip chip,
   const struct d12_agx3_vdm_state_update_template_info *info,
   struct d12_agx3_vdm_state_update *out)
{
   struct agx_vdm_state_update shared;

   if (out)
      memset(out, 0, sizeof(*out));

   if (!out || !info ||
       info->variant < D12_AGX3_VDM_STATE_UPDATE_TEMPLATE_00 ||
       info->variant > D12_AGX3_VDM_STATE_UPDATE_TEMPLATE_21)
      return false;

   const struct agx_vdm_state_update_template_info shared_info = {
      .pipeline = info->pipeline,
      .raw_state_load_count_12 = info->raw_state_load_count_12,
      .raw_state_load_count_18 = info->raw_state_load_count_18,
      .raw_vs_output_count_0 = info->raw_vs_output_count_0,
      .raw_vs_output_count_2 = info->raw_vs_output_count_2,
      .raw_vs_output_count_1_extra = info->raw_vs_output_count_1_extra,
      .raw_vs_output_selector = info->raw_vs_output_selector,
      .raw_vs_output_flag_6 = info->raw_vs_output_flag_6,
      .pre_hal300_flag_group_nonzero =
         info->pre_hal300_flag_group_nonzero,
      .pre_hal300_flag_bit_4 = info->pre_hal300_flag_bit_4,
      .variant =
         (enum agx_vdm_state_update_template_variant)info->variant,
      .raw_active_context_selector = info->raw_active_context_selector,
   };

   if (!agx_vdm_state_update_template_pack(chip, &shared_info, &shared))
      return false;

   _Static_assert(sizeof(out->bytes) == sizeof(shared.bytes),
                  "D3D and shared VDM State Update storage must agree");
   memcpy(out->bytes, shared.bytes, sizeof(out->bytes));
   out->length = shared.length;
   return true;
}

bool
d12_agx3_encode_vdm_state_update_draw(
   enum agx_chip chip, const struct d12_agx3_vdm_state_update_draw_info *info,
   struct d12_agx3_vdm_state_update *out)
{
   struct agx_vdm_state_update shared;

   if (out)
      memset(out, 0, sizeof(*out));

   if (!out || !info ||
       info->source_layout !=
          D12_AGX3_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT)
      return false;

   enum agx_vdm_state_update_draw_path shared_draw_path;
   switch (info->draw_path) {
   case D12_AGX3_VDM_STATE_UPDATE_DRAW_DIRECT_NONINDEXED:
      shared_draw_path = AGX_VDM_STATE_UPDATE_DRAW_DIRECT_NONINDEXED;
      break;
   case D12_AGX3_VDM_STATE_UPDATE_DRAW_DIRECT_INDEXED:
      shared_draw_path = AGX_VDM_STATE_UPDATE_DRAW_DIRECT_INDEXED;
      break;
   case D12_AGX3_VDM_STATE_UPDATE_DRAW_INDIRECT_NONINDEXED:
      shared_draw_path = AGX_VDM_STATE_UPDATE_DRAW_INDIRECT_NONINDEXED;
      break;
   case D12_AGX3_VDM_STATE_UPDATE_DRAW_INDIRECT_INDEXED:
      shared_draw_path = AGX_VDM_STATE_UPDATE_DRAW_INDIRECT_INDEXED;
      break;
   default:
      return false;
   }

   const struct d12_agx3_vdm_state_update_template_info *d12 =
      &info->template_info;
   const struct agx_vdm_state_update_draw_info shared_info = {
      .source_layout =
         AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT,
      .draw_path = shared_draw_path,
      .rasterization_enabled = info->rasterization_enabled,
      .raw_provoking_selector = info->raw_provoking_selector,
      .template_info = {
         .pipeline = d12->pipeline,
         .raw_state_load_count_12 = d12->raw_state_load_count_12,
         .raw_state_load_count_18 = d12->raw_state_load_count_18,
         .raw_vs_output_count_0 = d12->raw_vs_output_count_0,
         .raw_vs_output_count_2 = d12->raw_vs_output_count_2,
         .raw_vs_output_count_1_extra =
            d12->raw_vs_output_count_1_extra,
         .raw_vs_output_selector = d12->raw_vs_output_selector,
         .raw_vs_output_flag_6 = d12->raw_vs_output_flag_6,
         .pre_hal300_flag_group_nonzero =
            d12->pre_hal300_flag_group_nonzero,
         .pre_hal300_flag_bit_4 = d12->pre_hal300_flag_bit_4,
         /* Ignored by the exact renderer selector and overwritten by the
          * shared producer. */
         .variant = AGX_VDM_STATE_UPDATE_TEMPLATE_00,
         .raw_active_context_selector = d12->raw_active_context_selector,
      },
   };

   if (!agx_vdm_state_update_draw_pack(chip, &shared_info, &shared))
      return false;

   _Static_assert(sizeof(out->bytes) == sizeof(shared.bytes),
                  "D3D and shared VDM State Update storage must agree");
   memcpy(out->bytes, shared.bytes, sizeof(out->bytes));
   out->length = shared.length;
   return true;
}

bool
d12_agx3_g17p_vertex_program_info(
   enum agx_chip chip, const struct agx_g17p_vertex_compile_info *compiler_info,
   struct d12_agx3_vdm_state_update_program_info *program)
{
   if (program)
      memset(program, 0, sizeof(*program));
   struct agx_vdm_state_update_program_info shared;
   if (!program || !agx_g17p_vertex_program_info(chip, compiler_info, &shared))
      return false;
   program->source_layout =
      D12_AGX3_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT;
   const struct agx_vdm_state_update_template_info *t = &shared.template_info;
   program->template_info = (struct d12_agx3_vdm_state_update_template_info){
      .pipeline = t->pipeline,
      .raw_state_load_count_12 = t->raw_state_load_count_12,
      .raw_state_load_count_18 = t->raw_state_load_count_18,
      .raw_vs_output_count_0 = t->raw_vs_output_count_0,
      .raw_vs_output_count_2 = t->raw_vs_output_count_2,
      .raw_vs_output_count_1_extra = t->raw_vs_output_count_1_extra,
      .raw_vs_output_selector = t->raw_vs_output_selector,
      .raw_vs_output_flag_6 = t->raw_vs_output_flag_6,
      .pre_hal300_flag_group_nonzero = t->pre_hal300_flag_group_nonzero,
      .pre_hal300_flag_bit_4 = t->pre_hal300_flag_bit_4,
      .raw_active_context_selector = t->raw_active_context_selector,
      .variant = (enum d12_agx3_vdm_state_update_template_variant)t->variant,
   };
   return true;
}
