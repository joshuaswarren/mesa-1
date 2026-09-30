/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_agx3_vdm_decode.h"

#include "compiler/shader_enums.h"
#include "util/blob.h"
#include "util/bitscan.h"

#include <string.h>

bool
hk_agx3_shader_binary_version_supported(uint32_t version)
{
   return version == HK_AGX3_SHADER_BINARY_VERSION;
}

bool
hk_agx3_vdm_program_cache_serialize(
   struct blob *blob, const struct agx_vdm_state_update_program_info *info)
{
   if (!blob || !info)
      return false;

   const struct agx_vdm_state_update_template_info *template =
      &info->template_info;
   blob_write_uint32(blob, info->source_layout);
   blob_write_uint64(blob, template->pipeline);
   blob_write_uint16(blob, template->raw_state_load_count_12);
   blob_write_uint32(blob, template->raw_state_load_count_18);
   blob_write_uint8(blob, template->raw_vs_output_count_0);
   blob_write_uint8(blob, template->raw_vs_output_count_2);
   blob_write_uint8(blob, template->raw_vs_output_count_1_extra);
   blob_write_uint8(blob, template->raw_vs_output_selector);
   blob_write_uint8(blob, template->raw_vs_output_flag_6);
   blob_write_uint8(blob, template->pre_hal300_flag_group_nonzero);
   blob_write_uint8(blob, template->pre_hal300_flag_bit_4);
   blob_write_uint32(blob, template->variant);
   blob_write_uint32(blob, template->raw_active_context_selector);
   return !blob->out_of_memory;
}

bool
hk_agx3_vdm_program_cache_deserialize(
   struct blob_reader *blob, struct agx_vdm_state_update_program_info *info)
{
   if (info)
      memset(info, 0, sizeof(*info));
   if (!blob || !info)
      return false;

   const uint32_t source_layout = blob_read_uint32(blob);
   const uint64_t pipeline = blob_read_uint64(blob);
   const uint16_t state_load_count_12 = blob_read_uint16(blob);
   const uint32_t state_load_count_18 = blob_read_uint32(blob);
   const uint8_t output_count_0 = blob_read_uint8(blob);
   const uint8_t output_count_2 = blob_read_uint8(blob);
   const uint8_t output_count_1_extra = blob_read_uint8(blob);
   const uint8_t output_selector = blob_read_uint8(blob);
   const uint8_t output_flag_6 = blob_read_uint8(blob);
   const uint8_t pre_hal300_flag_group_nonzero = blob_read_uint8(blob);
   const uint8_t pre_hal300_flag_bit_4 = blob_read_uint8(blob);
   const uint32_t variant = blob_read_uint32(blob);
   const uint32_t active_context_selector = blob_read_uint32(blob);

   if (blob->overrun ||
       source_layout >
          AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT ||
       variant > AGX_VDM_STATE_UPDATE_TEMPLATE_21 ||
       output_count_1_extra > 1 || output_flag_6 > 1 ||
       pre_hal300_flag_group_nonzero > 1 || pre_hal300_flag_bit_4 > 1)
      return false;

   *info = (struct agx_vdm_state_update_program_info){
      .source_layout =
         (enum agx_vdm_state_update_source_layout)source_layout,
      .template_info = {
         .pipeline = pipeline,
         .raw_state_load_count_12 = state_load_count_12,
         .raw_state_load_count_18 = state_load_count_18,
         .raw_vs_output_count_0 = output_count_0,
         .raw_vs_output_count_2 = output_count_2,
         .raw_vs_output_count_1_extra = output_count_1_extra,
         .raw_vs_output_selector = output_selector,
         .raw_vs_output_flag_6 = output_flag_6,
         .pre_hal300_flag_group_nonzero =
            pre_hal300_flag_group_nonzero,
         .pre_hal300_flag_bit_4 = pre_hal300_flag_bit_4,
         .variant = (enum agx_vdm_state_update_template_variant)variant,
         .raw_active_context_selector = active_context_selector,
      },
   };
   return true;
}

enum agx_vdm_state_update_source_layout
hk_agx3_vdm_live_source_layout(
   const struct agx_vdm_state_update_program_info *program,
   bool active_context_selector_valid, bool ordinary_draw_path)
{
   return program && ordinary_draw_path && active_context_selector_valid &&
                 program->source_layout ==
                    AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT
             ? AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT
             : AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_UNSPECIFIED;
}

static bool
hk_agx3_vertex_program_info(
   enum agx_chip chip,
   const struct hk_agx3_g17p_vertex_compile_info *compiler_info,
   struct agx_vdm_state_update_program_info *program)
{
   if (program)
      memset(program, 0, sizeof(*program));

   const bool writes_psiz =
      compiler_info &&
      compiler_info->layer_viewport_off == compiler_info->psiz_off + 1;
   const bool writes_layer_viewport =
      compiler_info &&
      compiler_info->clip_off == compiler_info->layer_viewport_off + 1;
   /* Derive this from the final UVS layout. Vulkan I/O lowering clears
    * outputs_written before backend compilation, so a separate semantic bit
    * can be stale even when the emitted position-only layout is complete. */
   const bool position_only =
      compiler_info && compiler_info->user_size == 0 && !writes_psiz &&
      !writes_layer_viewport;

   /* A constant program needs its register allocation, a known staging
    * footprint and a caller that built it as a native CProg; without a
    * preamble there must be no preamble resources at all. */
   const bool preamble_ok =
      compiler_info &&
      (compiler_info->has_preamble
          ? chip == AGX_CHIP_G17P && compiler_info->constant_program_allowed &&
               compiler_info->preamble_staging_known &&
               compiler_info->nr_preamble_gprs != 0 &&
               compiler_info->nr_preamble_gprs <= 256
          : compiler_info->nr_preamble_gprs == 0 &&
               compiler_info->preamble_scratch_size == 0);

   if ((chip != AGX_CHIP_G17P && chip != AGX_CHIP_G17G) ||
       !compiler_info || !program ||
       !compiler_info->hardware_vertex || !preamble_ok ||
       compiler_info->position_off != 0 || compiler_info->varyings_off != 4 ||
       compiler_info->psiz_off != 4 + compiler_info->user_size ||
       (compiler_info->layer_viewport_off != compiler_info->psiz_off &&
        compiler_info->layer_viewport_off != compiler_info->psiz_off + 1) ||
       (compiler_info->clip_off != compiler_info->layer_viewport_off &&
        compiler_info->clip_off != compiler_info->layer_viewport_off + 1) ||
       compiler_info->uvs_size < compiler_info->clip_off ||
       compiler_info->uvs_size - compiler_info->clip_off > 8 ||
       (position_only && compiler_info->uvs_written != 0) ||
       compiler_info->user_size > HK_AGX3_G17P_MAX_USER_WORDS ||
       compiler_info->uvs_size > HK_AGX3_G17P_MAX_UVS_WORDS ||
       util_bitcount64(compiler_info->uvs_written) > compiler_info->user_size)
      return false;

   struct agx_vdm_state_update_program_info staged = {
      .source_layout =
         AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT,
      .template_info =
         {
            /* The no-preamble Ioto loader finishes with both state counts zero.
             * A constant program replaces them below with its TI resources.
             */
            .raw_state_load_count_12 = 0,
            .raw_state_load_count_18 = 0,
            /* VsStateConfig+0xc is the total f32 UVS footprint, including
             * position AND user outputs; +0x14 stays zero in this all-f32
             * path. Apple vs0/vs2/vs4 __GPU_LD_MD root.field1.field3 gives
             * 4/6/8, with field4 absent. The native position+point-size
             * vertex shader likewise gives field3=5 with field4 absent. The former 4/user split described
             * the wrong vertex layout even though its middle sum matched. */
            .raw_vs_output_count_0 = compiler_info->uvs_size,
            .raw_vs_output_count_2 = 0,
            .raw_vs_output_count_1_extra = false,
            .raw_vs_output_selector = 0,
            .raw_vs_output_flag_6 = false,
            .pre_hal300_flag_group_nonzero = chip == AGX_CHIP_G17P,
            .pre_hal300_flag_bit_4 = false,
            .variant = AGX_VDM_STATE_UPDATE_TEMPLATE_00,
            .raw_active_context_selector = UINT32_MAX,
         },
   };

   /* Native CProg inputs live at the start of the preamble's TPS frame, so
    * the preamble scratch size is the thread-invariant spill-buffer size
    * (Apple's terrain VS: eight imported words, 32 bytes). */
   if (compiler_info->has_preamble &&
       !agx_vdm_state_update_vertex_ti_resources(
          chip, compiler_info->preamble_scratch_size,
          (uint32_t)compiler_info->preamble_staging_halfs * 2u,
          &staged.template_info))
      return false;

   *program = staged;
   return true;
}

bool
hk_agx3_g17p_vertex_program_info(
   enum agx_chip chip,
   const struct hk_agx3_g17p_vertex_compile_info *compiler_info,
   struct agx_vdm_state_update_program_info *program)
{
   if (chip != AGX_CHIP_G17P) {
      if (program)
         memset(program, 0, sizeof(*program));
      return false;
   }
   return hk_agx3_vertex_program_info(chip, compiler_info, program);
}

bool
hk_agx3_g17g_vertex_program_info(
   enum agx_chip chip,
   const struct hk_agx3_g17p_vertex_compile_info *compiler_info,
   struct agx_vdm_state_update_program_info *program)
{
   const bool position = compiler_info && compiler_info->uvs_size == 4 &&
                         compiler_info->uvs_written == 0;
   /* Dynamic-topology pipelines retain the default point-size output even
    * for triangle draws. Use the shared UVS footprint encoding for that
    * fifth word; layer/viewport output remains a separate profile.
    */
   const bool position_psiz = compiler_info &&
      compiler_info->uvs_size == 5 && compiler_info->psiz_off == 4 &&
      compiler_info->layer_viewport_off == 5 && compiler_info->clip_off == 5 &&
      compiler_info->uvs_written == 0;
   /* Native applegpu_g17g color and color+point-size vertex pipelines report
    * __GPU_LD_MD root.field1.field3 = 8/9 with field4 absent. This is the
    * same all-f32 VsStateConfig footprint consumed by the shared encoder,
    * with HAL300's pre-HAL300 flags absent. Fragment coefficient state is
    * validated independently before any draw can be submitted.
    */
   const bool color = compiler_info &&
      compiler_info->user_size == 4 &&
      compiler_info->uvs_written == BITFIELD64_BIT(VARYING_SLOT_VAR0) &&
      compiler_info->psiz_off == 8 &&
      (compiler_info->layer_viewport_off == 8 ||
       compiler_info->layer_viewport_off == 9) &&
      compiler_info->clip_off == compiler_info->layer_viewport_off;
   if (chip != AGX_CHIP_G17G || !compiler_info ||
       (!color && ((!position && !position_psiz) || compiler_info->user_size != 0))) {
      if (program)
         memset(program, 0, sizeof(*program));
      return false;
   }
   return hk_agx3_vertex_program_info(chip, compiler_info, program);
}

bool
hk_agx3_decode_vdm_record(enum agx_chip chip, const void *command,
                          size_t available,
                          struct agx_vdm_decoded_record *record)
{
   return agx_vdm_decode_one(chip, command, available, record);
}

bool
hk_agx3_encode_vdm_state_update_template(
   enum agx_chip chip, const struct agx_vdm_state_update_template_info *info,
   struct agx_vdm_state_update *out)
{
   return agx_vdm_state_update_template_pack(chip, info, out);
}

bool
hk_agx3_encode_vdm_state_update_draw(
   enum agx_chip chip, const struct agx_vdm_state_update_draw_info *info,
   struct agx_vdm_state_update *out)
{
   return agx_vdm_state_update_draw_pack(chip, info, out);
}
