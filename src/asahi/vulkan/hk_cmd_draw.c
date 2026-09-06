/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include <assert.h>
#include "agx_abi.h"
#include "agx_bg_eot.h"
#include "agx_bo.h"
#include "agx_compile.h"
#include "agx_compiler.h"
#include "agx_device.h"
#include "agx_helpers.h"
#include "agx_linker.h"
#include "agx_nir_lower_vbo.h"
#include "agx_ppp.h"
#include "agx_ppp_hal200_graphics_core.h"
#include "agx_ppp_hal200_raster_core.h"
#include "agx_ppp_hal300_depth_bias_scissor_core.h"
#include "agx_ppp_viewport_payload_core.h"
#include "agx_ppp_w_clamp_cull2_core.h"
#include "agx_tilebuffer.h"
#include "agx_usc.h"
#include "agx_uvs.h"
#include "agx_vdm_hal200_bind_tail_core.h"
#include "agx_vdm_hal200_direct_draw_core.h"
#include "hk_buffer.h"
#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_image.h"
#include "hk_image_view.h"
#include "hk_agx3_ioto_program_core.h"
#include "hk_g17p_ppp_indices.h"
#include "hk_g17p_full_frame_scissor.h"
#include "hk_g17p_draw_state_upload.h"
#include "hk_agx3_vdm_decode.h"
#include "hk_physical_device.h"
#include "hk_private.h"
#include "hk_shader.h"

#include "asahi/genxml/agx_pack.h"
#include "asahi/libagx/compression.h"
#include "asahi/libagx/libagx.h"
#include "asahi/libagx/query.h"
#include "asahi/libagx/tessellator.h"
#include "poly/geometry.h"
#include "util/blend.h"
#include "util/format/format_utils.h"
#include "util/format/u_formats.h"
#include "util/macros.h"
#include "util/ralloc.h"
#include "util/u_prim.h"
#include "vulkan/vulkan_core.h"
#include "layout.h"
#include "libagx_dgc.h"
#include "libagx_shaders.h"
#include "nir.h"
#include "nir_builder.h"
#include "nir_intrinsics.h"
#include "nir_lower_blend.h"
#include "nir_xfb_info.h"
#include "pool.h"
#include "shader_enums.h"
#include "vk_blend.h"
#include "vk_enum_to_str.h"
#include "vk_format.h"
#include "vk_graphics_state.h"
#include "vk_pipeline.h"
#include "vk_render_pass.h"
#include "vk_shader.h"
#include "vk_standard_sample_locations.h"
#include "vk_util.h"

#define IS_DIRTY(bit) BITSET_TEST(dyn->dirty, MESA_VK_DYNAMIC_##bit)

#define IS_SHADER_DIRTY(bit)                                                   \
   (cmd->state.gfx.shaders_dirty & BITFIELD_BIT(MESA_SHADER_##bit))

#define IS_LINKED_DIRTY(bit)                                                   \
   (cmd->state.gfx.linked_dirty & BITFIELD_BIT(MESA_SHADER_##bit))

/* CTS coverage of indirect draws is pretty bad, so it's helpful to be able to
 * get some extra smoke testing.
 */
#define HK_TEST_INDIRECTS (0)

UNUSED static inline void
print_draw(struct agx_draw d, FILE *fp)
{
   if (agx_is_indirect(d.b))
      fprintf(fp, "indirect (buffer %" PRIx64 "):", d.b.ptr);
   else
      fprintf(fp, "direct (%ux%u):", d.b.count[0], d.b.count[1]);

   if (d.index_size)
      fprintf(fp, " index_size=%u", agx_index_size_to_B(d.index_size));
   else
      fprintf(fp, " non-indexed");

   if (d.restart)
      fprintf(fp, " restart");

   if (d.index_bias)
      fprintf(fp, " index_bias=%u", d.index_bias);

   if (d.start)
      fprintf(fp, " start=%u", d.start);

   if (d.start_instance)
      fprintf(fp, " start_instance=%u", d.start_instance);

   fprintf(fp, "\n");
}

static bool
hk_rast_discard(struct hk_cmd_buffer *cmd)
{
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   /* A non-zero rasterization stream acts as a rasterizer discard unless
    * there's a multistream geometry shader bound.
    */
   if (dyn->rs.rasterization_stream != 0) {
      struct hk_api_shader *gs = cmd->state.gfx.shaders[MESA_SHADER_GEOMETRY];
      return !gs || !gs->variants[HK_GS_VARIANT_COUNT].info.gs.multistream;
   }

   return dyn->rs.rasterizer_discard_enable;
}

static void
hk_cmd_buffer_dirty_render_pass(struct hk_cmd_buffer *cmd)
{
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   /* These depend on color attachment count */
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_CB_COLOR_WRITE_ENABLES);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_CB_BLEND_ENABLES);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_CB_BLEND_EQUATIONS);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_CB_WRITE_MASKS);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_CB_BLEND_ADVANCED);

   /* These depend on the depth/stencil format */
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_DS_DEPTH_TEST_ENABLE);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_DS_DEPTH_WRITE_ENABLE);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_DS_DEPTH_BOUNDS_TEST_ENABLE);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_DS_STENCIL_TEST_ENABLE);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_RS_DEPTH_BIAS_FACTORS);

   /* This may depend on render targets for ESO */
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_MS_RASTERIZATION_SAMPLES);

   /* This may depend on render targets */
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_COLOR_ATTACHMENT_MAP);
}

void
hk_cmd_buffer_begin_graphics(struct hk_cmd_buffer *cmd,
                             const VkCommandBufferBeginInfo *pBeginInfo)
{
   if (cmd->vk.level != VK_COMMAND_BUFFER_LEVEL_PRIMARY &&
       (pBeginInfo->flags & VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT)) {
      char gcbiar_data[VK_GCBIARR_DATA_SIZE(HK_MAX_RTS)];
      const VkRenderingInfo *resume_info =
         vk_get_command_buffer_inheritance_as_rendering_resume(
            cmd->vk.level, pBeginInfo, gcbiar_data);
      if (resume_info) {
         hk_CmdBeginRendering(hk_cmd_buffer_to_handle(cmd), resume_info);
      } else {
         const VkCommandBufferInheritanceRenderingInfo *inheritance_info =
            vk_get_command_buffer_inheritance_rendering_info(cmd->vk.level,
                                                             pBeginInfo);
         assert(inheritance_info);

         struct hk_rendering_state *render = &cmd->state.gfx.render;
         render->flags = inheritance_info->flags;
         render->area = (VkRect2D){};
         render->layer_count = 0;
         render->view_mask = inheritance_info->viewMask;
         render->tilebuffer.nr_samples = inheritance_info->rasterizationSamples;

         render->color_att_count = inheritance_info->colorAttachmentCount;
         for (uint32_t i = 0; i < render->color_att_count; i++) {
            render->color_att[i].vk_format =
               inheritance_info->pColorAttachmentFormats[i];
         }
         render->depth_att.vk_format = inheritance_info->depthAttachmentFormat;
         render->stencil_att.vk_format =
            inheritance_info->stencilAttachmentFormat;

         const VkRenderingAttachmentLocationInfoKHR att_loc_info_default = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_LOCATION_INFO_KHR,
            .colorAttachmentCount = inheritance_info->colorAttachmentCount,
         };
         const VkRenderingAttachmentLocationInfoKHR *att_loc_info =
            vk_get_command_buffer_rendering_attachment_location_info(
               cmd->vk.level, pBeginInfo);
         if (att_loc_info == NULL)
            att_loc_info = &att_loc_info_default;

         vk_cmd_set_rendering_attachment_locations(&cmd->vk, att_loc_info);

         hk_cmd_buffer_dirty_render_pass(cmd);
      }
   }

   hk_cmd_buffer_dirty_all(cmd);

   /* If multiview is disabled, always read 0. If multiview is enabled,
    * hk_set_view_index will dirty the root each draw.
    */
   cmd->state.gfx.descriptors.root.draw.view_index = 0;
   cmd->state.gfx.descriptors.root_dirty = true;
}

void
hk_cmd_invalidate_graphics_state(struct hk_cmd_buffer *cmd)
{
   hk_cmd_buffer_dirty_all(cmd);

   /* From the Vulkan 1.3.275 spec:
    *
    *    "...There is one exception to this rule - if the primary command
    *    buffer is inside a render pass instance, then the render pass and
    *    subpass state is not disturbed by executing secondary command
    *    buffers."
    *
    * We need to reset everything EXCEPT the render pass state.
    */
   struct hk_rendering_state render_save = cmd->state.gfx.render;
   memset(&cmd->state.gfx, 0, sizeof(cmd->state.gfx));
   cmd->state.gfx.render = render_save;
}

static void
hk_attachment_init(struct hk_attachment *att,
                   const VkRenderingAttachmentInfo *info)
{
   if (info == NULL || info->imageView == VK_NULL_HANDLE) {
      *att = (struct hk_attachment){
         .iview = NULL,
      };
      return;
   }

   VK_FROM_HANDLE(hk_image_view, iview, info->imageView);
   *att = (struct hk_attachment){
      .vk_format = iview->vk.format,
      .iview = iview,
   };

   if (info->resolveMode != VK_RESOLVE_MODE_NONE) {
      VK_FROM_HANDLE(hk_image_view, res_iview, info->resolveImageView);
      att->resolve_mode = info->resolveMode;
      att->resolve_iview = res_iview;
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_GetRenderingAreaGranularityKHR(
   VkDevice device, const VkRenderingAreaInfoKHR *pRenderingAreaInfo,
   VkExtent2D *pGranularity)
{
   *pGranularity = (VkExtent2D){.width = 1, .height = 1};
}

static bool
is_attachment_stored(const VkRenderingAttachmentInfo *att)
{
   /* When resolving, we store the intermediate multisampled image as the
    * resolve is a separate control stream. This could be optimized.
    */
   return att->storeOp == VK_ATTACHMENT_STORE_OP_STORE ||
          att->resolveMode != VK_RESOLVE_MODE_NONE;
}

static struct agx_bo *
hk_g17p_alloc_ioto_program(struct hk_cmd_buffer *cmd, size_t size,
                           const char *label)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct agx_bo *bo =
      agx_bo_create(&dev->dev, size, AGX_IOTO_PROGRAM_ALIGNMENT,
                    AGX_BO_EXEC | AGX_BO_LOW_VA, label);
   if (!bo) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return NULL;
   }

   util_dynarray_append(&cmd->large_bos, bo);
   return bo;
}

static bool
hk_g17p_bg_eot_shader_supported(const struct agx_bg_eot_shader *shader)
{
   return shader && shader->bo && shader->bo->va && shader->ptr != 0 &&
          shader->info.binary_size > 0 && shader->info.binary_size <= 256 &&
          shader->info.main_offset < shader->info.binary_size &&
          shader->info.main_size <=
             shader->info.binary_size - shader->info.main_offset &&
          !shader->info.has_preamble && shader->info.rodata.size_16 == 0 &&
          shader->info.nr_gprs > 0;
}

static struct hk_bg_eot
hk_build_g17p_bg_eot(struct hk_cmd_buffer *cmd,
                     struct agx_bg_eot_key *key,
                     const VkRenderingInfo *info, bool store)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (!hk_agx3_g17p_rt0_tile_layout_supported(&key->tib)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return (struct hk_bg_eot){0};
   }

   const bool wboit_mrt =
      key->tib.logical_format[0] == PIPE_FORMAT_R16G16B16A16_FLOAT &&
      key->tib.logical_format[1] == PIPE_FORMAT_R8_UNORM;

   for (unsigned rt = 0; rt < HK_MAX_RTS; ++rt) {
      const enum agx_bg_eot_op op = key->op[rt];
      const bool rt0_supported = store
                                    ? (op == AGX_EOT_STORE || op == AGX_BG_EOT_NONE)
                                    : (op == AGX_BG_EOT_NONE ||
                                       op == AGX_BG_CLEAR || op == AGX_BG_LOAD);
      const bool wboit_mrt_rt1 =
         rt == 1 && wboit_mrt &&
         (store ? op == AGX_EOT_STORE
                : (op == AGX_BG_CLEAR || op == AGX_BG_LOAD));
      if ((rt == 0 && !rt0_supported) ||
          (rt != 0 && op != AGX_BG_EOT_NONE && !wboit_mrt_rt1)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return (struct hk_bg_eot){0};
      }
   }

   const enum agx_bg_eot_op op = key->op[0];
   const bool wboit_mrt_active = wboit_mrt &&
                                 key->op[1] != AGX_BG_EOT_NONE;
   if (wboit_mrt_active &&
       (key->op[1] != op || info->colorAttachmentCount < 2)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return (struct hk_bg_eot){0};
   }
   if (!store && op == AGX_BG_CLEAR)
      key->reserved_preamble = wboit_mrt_active
         ? HK_AGX3_G17P_WBOIT_MRT_CLEAR_PUSH_HALFS
         : HK_AGX3_G17P_RT0_CLEAR_PUSH_HALFS;

   static int rt1_via_slot0 = -1;
   if (rt1_via_slot0 < 0)
      rt1_via_slot0 = debug_get_bool_option(
         "HK_G17P_WBOIT_EOT_RT1_VIA_SLOT0", false);
   const bool diagnostic_rt1_slot0 =
      store && wboit_mrt_active && rt1_via_slot0;
   static int rt1_from_rt0 = -1;
   if (rt1_from_rt0 < 0)
      rt1_from_rt0 = debug_get_bool_option(
         "HK_G17P_WBOIT_EOT_RT1_FROM_RT0", false);
   const bool diagnostic_rt1_from_rt0 =
      diagnostic_rt1_slot0 && rt1_from_rt0;
   struct agx_bg_eot_key shader_key = *key;
   if (diagnostic_rt1_slot0) {
      shader_key.op[1] = AGX_BG_EOT_NONE;
      shader_key.tib.logical_format[0] = PIPE_FORMAT_R8_UNORM;
      shader_key.tib.logical_format[1] = PIPE_FORMAT_NONE;
      shader_key.tib._offset_B[0] = 0;
   }

   struct agx_bg_eot_shader *shader =
      agx_get_bg_eot_shader(&dev->bg_eot, &shader_key);
   const bool shader_resources_match =
      shader &&
      (store || (op == AGX_BG_EOT_NONE
                    ? shader->info.push_count == 0 && !shader->info.uses_txf
                 : hk_agx3_g17p_rt0_bg_shader_resources_match(
                      op == AGX_BG_CLEAR ? HK_AGX3_G17P_RT0_BG_CLEAR
                                         : HK_AGX3_G17P_RT0_BG_RELOAD,
                      shader->info.push_count, shader->info.uses_txf)));
   if (!store && debug_get_bool_option("HK_G17P_LAUNCH_TRACE", false) && shader)
      fprintf(stderr, "G17P BG metadata: op=%u size=%u gprs=%u push=%u "
                      "staging=%u txf=%u supported=%u resources=%u\n",
              op, shader->info.binary_size, shader->info.nr_gprs,
              shader->info.push_count, shader->info.agx3_staging_halfs,
              shader->info.uses_txf, hk_g17p_bg_eot_shader_supported(shader),
              shader_resources_match);
   if (!hk_g17p_bg_eot_shader_supported(shader) || !shader_resources_match) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return (struct hk_bg_eot){0};
   }

   size_t program_capacity;
   const char *program_label;
   if (op == AGX_BG_EOT_NONE) {
      program_capacity = AGX_IOTO_G17P_NO_RESOURCE_PROGRAM_SIZE;
      program_label = "G17P empty tile Ioto program";
   } else if (store) {
      program_capacity = AGX_IOTO_G17P_RT0_EOT_PROGRAM_SIZE;
      program_label = "G17P RT0 EOT Ioto program";
   } else if (op == AGX_BG_CLEAR) {
      program_capacity = wboit_mrt_active
         ? AGX_IOTO_G17P_WBOIT_MRT_CLEAR_PROGRAM_SIZE
         : AGX_IOTO_G17P_RT0_CLEAR_PROGRAM_SIZE;
      program_label = "G17P RT0 clear Ioto program";
   } else {
      program_capacity = AGX_IOTO_G17P_RT0_RELOAD_PROGRAM_MAX_SIZE;
      program_label = "G17P RT0 reload Ioto program";
   }

   struct agx_bo *program_bo =
      hk_g17p_alloc_ioto_program(cmd, program_capacity, program_label);
   if (!program_bo)
      return (struct hk_bg_eot){0};

   void *program_map = agx_bo_map(program_bo);
   if (!program_map) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_MEMORY_MAP_FAILED);
      return (struct hk_bg_eot){0};
   }

   size_t program_length;
   uint64_t resource_spec;
   if (store && op == AGX_EOT_STORE) {
      struct hk_image_view *iview = cmd->state.gfx.render.color_att[0].iview;
      struct hk_image_view *iview1 = wboit_mrt_active
         ? cmd->state.gfx.render.color_att[1].iview : NULL;
      if (!iview || !iview->planes[0].eot_pbe_valid ||
          info->pColorAttachments[0].imageView == VK_NULL_HANDLE ||
          (wboit_mrt_active &&
           (!iview1 || !iview1->planes[0].eot_pbe_valid ||
            info->pColorAttachments[1].imageView == VK_NULL_HANDLE))) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FORMAT_NOT_SUPPORTED);
         return (struct hk_bg_eot){0};
      }

      const unsigned descriptor_generation =
         agx_chip_descriptor_generation(dev->dev.chip);
      if (hk_pbe_descriptor_size(descriptor_generation) !=
          AGX_IOTO_G17P_RT0_PBE_ARGUMENT_SIZE) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return (struct hk_bg_eot){0};
      }
      static_assert(sizeof(struct hk_native_pbe_descriptor) ==
                       AGX_IOTO_G17P_RT0_PBE_ARGUMENT_SIZE,
                    "native PBE table stride");

      struct hk_native_pbe_descriptor pbe_arguments[2] = {
         diagnostic_rt1_slot0 ? iview1->planes[0].eot
                              : iview->planes[0].eot,
      };
      if (wboit_mrt_active && !diagnostic_rt1_slot0)
         pbe_arguments[1] = iview1->planes[0].eot;
      const size_t pbe_argument_size =
         wboit_mrt_active && !diagnostic_rt1_slot0
         ? AGX_IOTO_G17P_WBOIT_MRT_PBE_ARGUMENT_SIZE
         : AGX_IOTO_G17P_RT0_PBE_ARGUMENT_SIZE;
      const uint64_t pbe_argument_va =
         hk_pool_upload(cmd, pbe_arguments, pbe_argument_size, 32);
      if (!pbe_argument_va)
         return (struct hk_bg_eot){0};

      const struct hk_agx3_g17p_rt0_eot_program_desc desc = {
         .program_va = program_bo->va->addr,
         .shader_va = shader->ptr + shader->info.main_offset,
         .pbe_argument_va = pbe_argument_va,
         .imageblock_offset_B = agx_tilebuffer_offset_B(
            &key->tib, diagnostic_rt1_from_rt0 ? 0
                                               : diagnostic_rt1_slot0 ? 1 : 0),
         .attachment_count =
            wboit_mrt_active && !diagnostic_rt1_slot0 ? 2 : 1,
         .imageblock_offset1_B =
            wboit_mrt_active && !diagnostic_rt1_slot0
            ? agx_tilebuffer_offset_B(&key->tib, 1) : 0,
         .shader_binary_size = shader->info.binary_size,
         .scratch_size = shader->info.scratch_size,
         .nr_gprs = shader->info.nr_gprs,
         .staging_halfs = shader->info.agx3_staging_known
                            ? shader->info.agx3_staging_halfs : 0,
      };
      struct hk_agx3_g17p_rt0_eot_program program;
      if (!hk_agx3_g17p_rt0_eot_program_generate(dev->dev.chip, &desc,
                                                  &program)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return (struct hk_bg_eot){0};
      }

      memcpy(program_map, program.bytes, program.length);
      program_length = program.length;
      resource_spec = AGX_IOTO_G17P_EOT_RESOURCE_SPEC;
      if (debug_get_bool_option("HK_G17P_LAUNCH_TRACE", false))
         fprintf(stderr,
                 "G17P EOT launch: program=%" PRIx64 " length=%zu "
                 "pbe=%" PRIx64 " attachments=%u offsets=%u,%u "
                 "gprs=%u staging=%u\n",
                 program_bo->va->addr, program_length, pbe_argument_va,
                 desc.attachment_count, desc.imageblock_offset_B,
                 desc.imageblock_offset1_B, desc.nr_gprs,
                 desc.staging_halfs);
   } else if (op == AGX_BG_EOT_NONE) {
      const struct hk_agx3_g17p_graphics_program_desc desc = {
         .program_va = program_bo->va->addr,
         .shader_va = shader->ptr + shader->info.main_offset,
         .shader_binary_size = shader->info.binary_size,
         .scratch_size = shader->info.scratch_size,
         .nr_gprs = shader->info.nr_gprs,
         .compiler_d_valid = shader->info.agx3_staging_known,
         .compiler_d = (uint32_t)shader->info.agx3_staging_halfs * 2,
         .fragment = true,
      };
      struct hk_agx3_g17p_graphics_program program;
      if (!hk_agx3_g17p_basic_graphics_program_generate(dev->dev.chip, &desc,
                                                        &program)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return (struct hk_bg_eot){0};
      }

      memcpy(program_map, program.bytes, program.length);
      program_length = program.length;
      if (!hk_agx3_g17p_rt0_bg_resource_spec(
             dev->dev.chip, HK_AGX3_G17P_RT0_BG_NONE, &resource_spec)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return (struct hk_bg_eot){0};
      }
   } else {
      const VkRenderingAttachmentInfo *att_info = &info->pColorAttachments[0];
      struct hk_image_view *iview = cmd->state.gfx.render.color_att[0].iview;
      const VkRenderingAttachmentInfo *att_info1 =
         wboit_mrt_active ? &info->pColorAttachments[1] : NULL;
      struct hk_image_view *iview1 =
         wboit_mrt_active ? cmd->state.gfx.render.color_att[1].iview : NULL;
      if (!iview || att_info->imageView == VK_NULL_HANDLE ||
          (wboit_mrt_active &&
           (!iview1 || att_info1->imageView == VK_NULL_HANDLE))) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FORMAT_NOT_SUPPORTED);
         return (struct hk_bg_eot){0};
      }

      const unsigned descriptor_generation =
         agx_chip_descriptor_generation(dev->dev.chip);
      if (hk_texture_descriptor_size(descriptor_generation) != 32 ||
          agx_chip_sampler_heap_element_size(dev->dev.chip) != 32) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return (struct hk_bg_eot){0};
      }

      struct hk_agx3_g17p_rt0_bg_program_desc desc = {
         .program_va = program_bo->va->addr,
         .shader_va = shader->ptr + shader->info.main_offset,
         .shader_binary_size = shader->info.binary_size,
         .scratch_size = shader->info.scratch_size,
         .nr_gprs = shader->info.nr_gprs,
         .staging_halfs = shader->info.agx3_staging_halfs,
         .clear_push_halfs = shader->info.push_count,
         .staging_known = shader->info.agx3_staging_known,
         .op = op == AGX_BG_CLEAR ? HK_AGX3_G17P_RT0_BG_CLEAR
                                  : HK_AGX3_G17P_RT0_BG_RELOAD,
      };

      if (op == AGX_BG_CLEAR) {
         static_assert(sizeof(att_info->clearValue.color) == 16, "fixed ABI");
         VkClearColorValue clear_values[2] = {
            att_info->clearValue.color,
         };
         if (wboit_mrt_active)
            clear_values[1] = att_info1->clearValue.color;
         const size_t clear_size = wboit_mrt_active ? 20 : 16;
         desc.clear_value_va =
            hk_pool_upload(cmd, clear_values, clear_size, 16);
         if (!desc.clear_value_va)
            return (struct hk_bg_eot){0};
      } else {
         /* Native texture-table entries occupy aligned 32-byte slots. RT1 is
          * native texture index2, so the exact MRT table retains index1 as a
          * zero gap. */
         static_assert(sizeof(struct hk_native_texture_descriptor) ==
                          AGX_TEXTURE_GEN9_LENGTH,
                       "native texture table stride");
         struct hk_native_texture_descriptor textures[3] = {
            iview->planes[0].background,
         };
         if (wboit_mrt_active)
            textures[2] = iview1->planes[0].background;
         const size_t texture_size = wboit_mrt_active
            ? 3 * AGX_TEXTURE_GEN9_LENGTH : AGX_TEXTURE_GEN9_LENGTH;
         const uint64_t texture_va =
            hk_pool_upload(cmd, textures, texture_size,
                           AGX_TEXTURE_GEN9_LENGTH);
         const uint64_t sampler_va =
            dev->rodata.bo && dev->rodata.bo->va ? dev->rodata.bo->va->addr : 0;
         if (!texture_va || !sampler_va)
            return (struct hk_bg_eot){0};
         const uint64_t pointers[2] = {texture_va, sampler_va};
         const uint64_t table = hk_pool_upload(cmd, pointers, sizeof(pointers), 8);
         if (!table)
            return (struct hk_bg_eot){0};
         desc.texture_pointer_va = table;
         desc.sampler_pointer_va = table + sizeof(uint64_t);
      }

      struct hk_agx3_g17p_rt0_bg_program program;
      if (!hk_agx3_g17p_rt0_bg_program_generate(dev->dev.chip, &desc,
                                                &program)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return (struct hk_bg_eot){0};
      }

      memcpy(program_map, program.bytes, program.length);
      program_length = program.length;
      if (!hk_agx3_g17p_rt0_bg_resource_spec(dev->dev.chip, desc.op,
                                            &resource_spec)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return (struct hk_bg_eot){0};
      }
   }

   if (!store && debug_get_bool_option("HK_G17P_LAUNCH_TRACE", false)) {
      fprintf(stderr,
              "G17P BG launch: op=%s program=%" PRIx64 " resources=%" PRIx64
              " length=%zu staging_known=%u staging_halfs=%u compiler_d=%u\n",
              op == AGX_BG_CLEAR ? "clear" :
              op == AGX_BG_LOAD ? "reload" : "none",
              program_bo->va->addr, resource_spec, program_length,
              shader->info.agx3_staging_known,
              shader->info.agx3_staging_halfs,
              shader->info.agx3_staging_known
                 ? (unsigned)shader->info.agx3_staging_halfs * 2
                 : shader->info.nr_gprs);
   }

   if (agx_bo_sync_for_device(program_bo, 0, program_length)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_DEVICE_LOST);
      return (struct hk_bg_eot){0};
   }

   return (struct hk_bg_eot){
      .usc = program_bo->va->addr,
      .rsrc_spec = resource_spec,
   };
}

static struct hk_bg_eot
hk_build_bg_eot(struct hk_cmd_buffer *cmd, const VkRenderingInfo *info,
                bool store, bool partial_render, bool incomplete_render_area)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_rendering_state *render = &cmd->state.gfx.render;

   /* Construct the key */
   struct agx_bg_eot_key key = {.tib = render->tilebuffer};
   static_assert(AGX_BG_EOT_NONE == 0, "default initializer");

   key.tib.layered = (render->cr.layers > 1);

   bool needs_textures_for_spilled_rts =
      agx_tilebuffer_spills(&render->tilebuffer) && !partial_render && !store;

   for (unsigned i = 0; i < info->colorAttachmentCount; ++i) {
      const VkRenderingAttachmentInfo *att_info = &info->pColorAttachments[i];
      if (att_info->imageView == VK_NULL_HANDLE)
         continue;

      /* Partial render programs exist only to store/load the tilebuffer to
       * main memory. When render targets are already spilled to main memory,
       * there's nothing to do.
       */
      if (key.tib.spilled[i] && (partial_render || store))
         continue;

      if (store) {
         bool should_store = is_attachment_stored(att_info);

         /* Partial renders always need to flush to memory. */
         should_store |= partial_render;

         if (should_store)
            key.op[i] = AGX_EOT_STORE;
      } else {
         bool load = att_info->loadOp == VK_ATTACHMENT_LOAD_OP_LOAD;
         bool clear = att_info->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR;

         /* The background program used for partial renders must always load
          * whatever was stored in the mid-frame end-of-tile program.
          */
         load |= partial_render;

         /* With an incomplete render area, we're forced to load back tiles and
          * then use the 3D pipe for the clear.
          */
         load |= incomplete_render_area;

         /* Don't read back spilled render targets, they're already in memory */
         load &= !key.tib.spilled[i];

         /* This is a very frustrating corner case. From the spec:
          *
          *     VK_ATTACHMENT_STORE_OP_NONE specifies the contents within the
          *     render area are not accessed by the store operation as long as
          *     no values are written to the attachment during the render pass.
          *
          * With VK_ATTACHMENT_STORE_OP_NONE, we suppress stores on the main
          * end-of-tile program. Unfortunately, that's not enough: we also need
          * to preserve the contents throughout partial renders. The easiest way
          * to do that is forcing a load in the background program, so that
          * partial stores for unused attachments will be no-op'd by writing
          * existing contents.
          *
          * Optimizing this would require nontrivial tracking. Fortunately,
          * this is all Android gunk and we don't have to care too much for
          * dekstop games. So do the simple thing.
          */
         bool no_store = (att_info->storeOp == VK_ATTACHMENT_STORE_OP_NONE);
         bool no_store_wa = no_store && !load && !clear;
         if (no_store_wa) {
            perf_debug(cmd, "STORE_OP_NONE workaround");
         }

         load |= no_store_wa;

         /* Don't apply clears for spilled render targets when we clear the
          * render area explicitly after.
          */
         if (key.tib.spilled[i] && incomplete_render_area)
            continue;

         if (load)
            key.op[i] = AGX_BG_LOAD;
         else if (clear)
            key.op[i] = AGX_BG_CLEAR;
      }
   }

   if (dev->dev.chip == AGX_CHIP_G17P)
      return hk_build_g17p_bg_eot(cmd, &key, info, store);

   const unsigned descriptor_generation =
      agx_chip_descriptor_generation(dev->dev.chip);
   if (!hk_native_pbe_hardware_supported(descriptor_generation)) {
      if (needs_textures_for_spilled_rts) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return (struct hk_bg_eot){0};
      }

      for (unsigned rt = 0; rt < HK_MAX_RTS; ++rt) {
         if (key.op[rt] == AGX_EOT_STORE) {
            vk_command_buffer_set_error(&cmd->vk,
                                        VK_ERROR_FEATURE_NOT_PRESENT);
            return (struct hk_bg_eot){0};
         }
      }
   }

   /* Begin building the pipeline */
   size_t usc_size = agx_usc_size(3 + HK_MAX_RTS);
   struct agx_ptr t = hk_pool_usc_alloc(cmd, usc_size, 64);
   if (!t.cpu)
      return (struct hk_bg_eot){.usc = t.gpu};

   struct agx_usc_builder b = agx_usc_builder(t.cpu, usc_size);

   bool uses_txf = false;
   unsigned uniforms = 0;
   unsigned nr_tex = 0;

   for (unsigned rt = 0; rt < HK_MAX_RTS; ++rt) {
      const VkRenderingAttachmentInfo *att_info = &info->pColorAttachments[rt];
      struct hk_image_view *iview = render->color_att[rt].iview;

      if (key.op[rt] == AGX_BG_LOAD) {
         uses_txf = true;

         const struct hk_native_texture_descriptor *desc =
            key.tib.layered ? &iview->planes[0].layered_background
                            : &iview->planes[0].background;

         agx_usc_pack(&b, TEXTURE, cfg) {
            /* Shifted to match eMRT indexing, could be optimized */
            cfg.start = rt * 2;
            cfg.count = 1;
            cfg.buffer = hk_pool_upload(
               cmd, desc,
               hk_texture_descriptor_size(descriptor_generation), 8);
         }

         nr_tex = (rt * 2) + 1;
      } else if (key.op[rt] == AGX_BG_CLEAR) {
         static_assert(sizeof(att_info->clearValue.color) == 16, "fixed ABI");
         uint64_t colour =
            hk_pool_upload(cmd, &att_info->clearValue.color, 16, 16);

         agx_usc_uniform(&b, 4 + (8 * rt), 8, colour);
         uniforms = MAX2(uniforms, 4 + (8 * rt) + 8);
      } else if (key.op[rt] == AGX_EOT_STORE) {
         const bool valid = key.tib.layered
                               ? iview->planes[0].layered_eot_pbe_valid
                               : iview->planes[0].eot_pbe_valid;
         if (!valid) {
            mesa_loge("render target %u needs an unproven native PBE axis",
                      rt);
            vk_command_buffer_set_error(&cmd->vk,
                                        VK_ERROR_FORMAT_NOT_SUPPORTED);
            return (struct hk_bg_eot){0};
         }

         const struct hk_native_pbe_descriptor *desc =
            key.tib.layered ? &iview->planes[0].layered_eot
                            : &iview->planes[0].eot;

         agx_usc_pack(&b, TEXTURE, cfg) {
            cfg.start = rt;
            cfg.count = 1;
            cfg.buffer = hk_pool_upload(
               cmd, desc, hk_pbe_descriptor_size(descriptor_generation), 8);
         }

         nr_tex = rt + 1;
      }
   }

   if (needs_textures_for_spilled_rts) {
      hk_usc_upload_spilled_rt_descs(&b, cmd);
      uniforms = MAX2(uniforms, 4);
   }

   if (uses_txf) {
      agx_usc_push_packed(&b, SAMPLER, dev->dev.txf_sampler);
   }

   /* For attachmentless rendering, we don't know the sample count until
    * draw-time. But we have trivial bg/eot programs in that case too.
    */
   if (key.tib.nr_samples >= 1) {
      agx_usc_push_packed(&b, SHARED, &key.tib.usc);
   } else {
      assert(key.tib.sample_size_B == 0);
      agx_usc_shared_none(&b);

      key.tib.nr_samples = 1;
   }

   /* Get the shader */
   key.reserved_preamble = uniforms;
   /* XXX: locking? */
   struct agx_bg_eot_shader *shader = agx_get_bg_eot_shader(&dev->bg_eot, &key);

   agx_usc_pack(&b, SHADER, cfg) {
      cfg.code = agx_usc_addr(&dev->dev, shader->ptr);
      cfg.unk_2 = 0;
   }

   agx_usc_pack(&b, REGISTERS, cfg)
      cfg.register_count = shader->info.nr_gprs;

   if (shader->info.has_preamble) {
      agx_usc_pack(&b, PRESHADER, cfg) {
         cfg.code =
            agx_usc_addr(&dev->dev, shader->ptr + shader->info.preamble_offset);
      }
   } else {
      agx_usc_pack(&b, NO_PRESHADER, cfg)
         ;
   }

   struct hk_bg_eot ret = {.usc = t.gpu};

   agx_pack(&ret.counts, COUNTS, cfg) {
      cfg.uniform_register_count = shader->info.push_count;
      cfg.preshader_register_count = shader->info.nr_preamble_gprs;
      cfg.texture_state_register_count = nr_tex;
      cfg.sampler_state_register_count =
         agx_translate_sampler_state_count(uses_txf ? 1 : 0, false);
   }

   return ret;
}

static void
hk_merge_render_iview(struct hk_rendering_state *render,
                      struct hk_image_view *iview, bool zls)
{
   if (iview) {
      unsigned samples = iview->vk.image->samples;
      /* TODO: is this right for ycbcr? */
      unsigned level = iview->vk.base_mip_level;
      unsigned width = u_minify(iview->vk.image->extent.width, level);
      unsigned height = u_minify(iview->vk.image->extent.height, level);

      assert(render->tilebuffer.nr_samples == 0 ||
             render->tilebuffer.nr_samples == samples);
      render->tilebuffer.nr_samples = samples;

      /* TODO: Is this merging logic sound? Not sure how this is supposed to
       * work conceptually.
       */
      render->cr.width = MAX2(render->cr.width, width);
      render->cr.height = MAX2(render->cr.height, height);

      if (zls) {
         render->cr.zls_width = width;
         render->cr.zls_height = height;
      }
   }
}

static void
hk_pack_zls_control(struct agx_zls_control_packed *packed,
                    struct ail_layout *z_layout, struct ail_layout *s_layout,
                    const VkRenderingAttachmentInfo *z,
                    const VkRenderingAttachmentInfo *s,
                    bool incomplete_render_area, bool partial_render)
{
   struct agx_zls zls = {0};

   if (z) {
      /* XXX: Dropping Z stores is wrong if the render pass gets split into
       * multiple control streams (can that ever happen?) We need more ZLS
       * variants. Force || true for now.
       */
      zls.z_store = z->storeOp == VK_ATTACHMENT_STORE_OP_STORE ||
                    z->resolveMode != VK_RESOLVE_MODE_NONE || partial_render ||
                    true;

      zls.z_load = z->loadOp == VK_ATTACHMENT_LOAD_OP_LOAD || partial_render ||
                   incomplete_render_area;
   }

   if (s) {
      /* TODO:
       * Fail
       * dEQP-VK.renderpass.dedicated_allocation.formats.d32_sfloat_s8_uint.input.dont_care.store.self_dep_clear_draw_use_input_aspect
       * without the force
       * .. maybe a VkRenderPass emulation bug.
       */
      zls.s_store = s->storeOp == VK_ATTACHMENT_STORE_OP_STORE ||
                    s->resolveMode != VK_RESOLVE_MODE_NONE || partial_render ||
                    true;

      zls.s_load = s->loadOp == VK_ATTACHMENT_LOAD_OP_LOAD || partial_render ||
                   incomplete_render_area;
   }

   agx_pack_zls_control(packed, z_layout, s_layout, &zls);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBeginRendering(VkCommandBuffer commandBuffer,
                     const VkRenderingInfo *pRenderingInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_rendering_state *render = &cmd->state.gfx.render;

   memset(render, 0, sizeof(*render));

   render->flags = pRenderingInfo->flags;
   render->area = pRenderingInfo->renderArea;
   render->view_mask = pRenderingInfo->viewMask;
   render->layer_count = pRenderingInfo->layerCount;
   render->tilebuffer.nr_samples = 0;

   const uint32_t layer_count = render->view_mask
                                   ? util_last_bit(render->view_mask)
                                   : render->layer_count;

   render->color_att_count = pRenderingInfo->colorAttachmentCount;
   for (uint32_t i = 0; i < render->color_att_count; i++) {
      hk_attachment_init(&render->color_att[i],
                         &pRenderingInfo->pColorAttachments[i]);
   }

   hk_attachment_init(&render->depth_att, pRenderingInfo->pDepthAttachment);
   hk_attachment_init(&render->stencil_att, pRenderingInfo->pStencilAttachment);

   for (uint32_t i = 0; i < render->color_att_count; i++) {
      hk_merge_render_iview(render, render->color_att[i].iview, false);
   }

   hk_merge_render_iview(
      render, render->depth_att.iview ?: render->stencil_att.iview, true);

   /* Infer for attachmentless. samples is inferred at draw-time. */
   render->cr.width =
      MAX2(render->cr.width, render->area.offset.x + render->area.extent.width);

   render->cr.height = MAX2(render->cr.height,
                            render->area.offset.y + render->area.extent.height);

   /* On G17P, zero dimensions are the native inactive-ZLS encoding.  Keep
    * them zero for attachmentless and colour-only rendering; every control
    * stream, including a partial stream, inherits that state from render. */
   if (!render->cr.zls_width && dev->dev.chip != AGX_CHIP_G17P) {
      render->cr.zls_width = render->cr.width;
      render->cr.zls_height = render->cr.height;
   }

   render->cr.layers = layer_count;

   /* Choose a tilebuffer layout given the framebuffer key */
   enum pipe_format formats[HK_MAX_RTS] = {0};
   for (unsigned i = 0; i < render->color_att_count; ++i) {
      formats[i] = hk_format_to_pipe_format(render->color_att[i].vk_format);
   }

   /* G17P's bounded RT0 programs use non-array image operations. Other chips
    * keep the existing always-layered layout.
    */
   const bool layered =
      dev->dev.chip != AGX_CHIP_G17P ||
      !hk_agx3_g17p_single_layer_rendering(render->view_mask, layer_count);
   render->tilebuffer = agx_build_tilebuffer_layout(
      formats, render->color_att_count, render->tilebuffer.nr_samples, layered);

   const VkRenderingAttachmentLocationInfoKHR ral_info = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_LOCATION_INFO_KHR,
      .colorAttachmentCount = pRenderingInfo->colorAttachmentCount,
   };
   vk_cmd_set_rendering_attachment_locations(&cmd->vk, &ral_info);

   hk_cmd_buffer_dirty_render_pass(cmd);

   /* Determine whether the render area is complete, enabling us to use a
    * fast-clear.
    *
    * TODO: If it is incomplete but tile aligned, it should be possibly to fast
    * clear with the appropriate settings. This is critical for performance.
    */
   bool incomplete_render_area =
      render->area.offset.x > 0 || render->area.offset.y > 0 ||
      render->area.extent.width < render->cr.width ||
      render->area.extent.height < render->cr.height ||
      (render->view_mask &&
       render->view_mask != BITFIELD64_MASK(render->cr.layers));

   perf_debug(cmd, "Rendering %ux%ux%u@%u %s%s", render->cr.width,
              render->cr.height, render->cr.layers,
              render->tilebuffer.nr_samples,
              render->view_mask ? " multiview" : "",
              incomplete_render_area ? " incomplete" : "");

   render->cr.bg.main = hk_build_bg_eot(cmd, pRenderingInfo, false, false,
                                        incomplete_render_area);
   if (vk_command_buffer_has_error(&cmd->vk))
      return;

   render->cr.bg.partial =
      hk_build_bg_eot(cmd, pRenderingInfo, false, true, incomplete_render_area);
   if (vk_command_buffer_has_error(&cmd->vk))
      return;

   render->cr.eot.main =
      hk_build_bg_eot(cmd, pRenderingInfo, true, false, incomplete_render_area);
   if (vk_command_buffer_has_error(&cmd->vk))
      return;

   render->cr.eot.partial =
      hk_build_bg_eot(cmd, pRenderingInfo, true, true, incomplete_render_area);
   if (vk_command_buffer_has_error(&cmd->vk))
      return;

   render->cr.isp_bgobjvals = 0x300;

   const VkRenderingAttachmentInfo *attach_z = pRenderingInfo->pDepthAttachment;
   const VkRenderingAttachmentInfo *attach_s =
      pRenderingInfo->pStencilAttachment;

   struct ail_layout *z_layout = NULL, *s_layout = NULL;

   if (attach_z != NULL && attach_z != VK_NULL_HANDLE && attach_z->imageView) {
      struct hk_image_view *view = render->depth_att.iview;
      struct hk_image *image =
         container_of(view->vk.image, struct hk_image, vk);

      z_layout = &image->planes[0].layout;

      unsigned level = view->vk.base_mip_level;
      unsigned first_layer = view->vk.base_array_layer;

      const struct util_format_description *desc =
         util_format_description(hk_format_to_pipe_format(view->vk.format));

      assert(desc->format == PIPE_FORMAT_Z32_FLOAT ||
             desc->format == PIPE_FORMAT_Z16_UNORM ||
             desc->format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT);

      render->cr.depth.buffer =
         hk_image_base_address(image, 0) +
         ail_get_layer_level_B(z_layout, first_layer, level);

      /* Main stride in pages */
      assert((z_layout->depth_px == 1 ||
              util_is_aligned(z_layout->layer_stride_B, AIL_PAGESIZE)) &&
             "Page aligned Z layers");

      unsigned stride_pages = z_layout->layer_stride_B / AIL_PAGESIZE;
      render->cr.depth.stride = ((stride_pages - 1) << 14) | 1;

      assert(z_layout->tiling != AIL_TILING_LINEAR && "must tile");

      if (z_layout->compressed) {
         render->cr.depth.meta =
            hk_image_base_address(image, 0) + z_layout->metadata_offset_B +
            (first_layer * z_layout->compression_layer_stride_B) +
            z_layout->level_offsets_compressed_B[level];

         /* Meta stride in cache lines */
         assert(util_is_aligned(z_layout->compression_layer_stride_B,
                                AIL_CACHELINE) &&
                "Cacheline aligned Z meta layers");

         unsigned stride_lines =
            z_layout->compression_layer_stride_B / AIL_CACHELINE;
         render->cr.depth.meta_stride = (stride_lines - 1) << 14;
      }

      float clear_depth = attach_z->clearValue.depthStencil.depth;

      if (z_layout->format == PIPE_FORMAT_Z16_UNORM) {
         render->cr.isp_bgobjdepth = _mesa_float_to_unorm(clear_depth, 16);
      } else {
         render->cr.isp_bgobjdepth = fui(clear_depth);
      }
   }

   if (attach_s != NULL && attach_s != VK_NULL_HANDLE && attach_s->imageView) {
      struct hk_image_view *view = render->stencil_att.iview;
      struct hk_image *image =
         container_of(view->vk.image, struct hk_image, vk);

      /* Stencil is always the last plane (possibly the only plane) */
      unsigned plane = image->plane_count - 1;
      s_layout = &image->planes[plane].layout;
      assert(s_layout->format == PIPE_FORMAT_S8_UINT);

      unsigned level = view->vk.base_mip_level;
      unsigned first_layer = view->vk.base_array_layer;

      render->cr.stencil.buffer =
         hk_image_base_address(image, plane) +
         ail_get_layer_level_B(s_layout, first_layer, level);

      /* Main stride in pages */
      assert((s_layout->depth_px == 1 ||
              util_is_aligned(s_layout->layer_stride_B, AIL_PAGESIZE)) &&
             "Page aligned S layers");
      unsigned stride_pages = s_layout->layer_stride_B / AIL_PAGESIZE;
      render->cr.stencil.stride = ((stride_pages - 1) << 14) | 1;

      if (s_layout->compressed) {
         render->cr.stencil.meta =
            hk_image_base_address(image, plane) + s_layout->metadata_offset_B +
            (first_layer * s_layout->compression_layer_stride_B) +
            s_layout->level_offsets_compressed_B[level];

         /* Meta stride in cache lines */
         assert(util_is_aligned(s_layout->compression_layer_stride_B,
                                AIL_CACHELINE) &&
                "Cacheline aligned S meta layers");

         unsigned stride_lines =
            s_layout->compression_layer_stride_B / AIL_CACHELINE;

         render->cr.stencil.meta_stride = (stride_lines - 1) << 14;
      }

      render->cr.isp_bgobjvals |= attach_s->clearValue.depthStencil.stencil;
   }

   hk_pack_zls_control(&render->cr.zls_control, z_layout, s_layout, attach_z,
                       attach_s, incomplete_render_area, false);

   hk_pack_zls_control(&render->cr.zls_control_partial, z_layout, s_layout,
                       attach_z, attach_s, incomplete_render_area, true);

   /* If multiview is disabled, always read 0. If multiview is enabled,
    * hk_set_view_index will dirty the root each draw.
    */
   cmd->state.gfx.descriptors.root.draw.view_index = 0;
   cmd->state.gfx.descriptors.root_dirty = true;

   if (render->flags & VK_RENDERING_RESUMING_BIT)
      return;

   /* The first control stream of the render pass is special since it gets
    * the clears. Create it and swap in the clear.
    */
   assert(!cmd->current_cs.gfx && "not already in a render pass");
   struct hk_cs *cs = hk_cmd_buffer_get_cs(cmd, false /* compute */);
   if (!cs)
      return;

   cs->cr.bg.main = render->cr.bg.main;
   cs->cr.zls_control = render->cr.zls_control;

   /* Reordering barrier for post-gfx, in case we had any. */
   hk_cmd_buffer_end_compute_internal(cmd, &cmd->current_cs.post_gfx);

   /* Don't reorder compute across render passes.
    *
    * TODO: Check if this is necessary if the proper PipelineBarriers are
    * handled... there may be CTS bugs...
    */
   hk_cmd_buffer_end_compute(cmd);

   /* If we spill colour attachments, we need to decompress them. This happens
    * at the start of the render; it is not re-emitted when resuming
    * secondaries. It could be hoisted to the start of the command buffer but
    * we're not that clever yet.
    */
   if (agx_tilebuffer_spills(&render->tilebuffer)) {
      const unsigned descriptor_generation =
         agx_chip_descriptor_generation(dev->dev.chip);
      if (!hk_native_pbe_hardware_supported(descriptor_generation)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }

      perf_debug(cmd, "eMRT render pass");

      for (unsigned i = 0; i < render->color_att_count; ++i) {
         struct hk_image_view *view = render->color_att[i].iview;
         if (view) {
            if (!view->planes[0].emrt_pbe_valid) {
               vk_command_buffer_set_error(&cmd->vk,
                                           VK_ERROR_FORMAT_NOT_SUPPORTED);
               return;
            }

            struct hk_image *image =
               container_of(view->vk.image, struct hk_image, vk);

            /* TODO: YCbCr interaction? */
            uint8_t plane = 0;
            uint8_t image_plane = view->planes[plane].image_plane;
            struct ail_layout *layout = &image->planes[image_plane].layout;

            if (ail_is_level_logically_compressed(layout,
                                                  view->vk.base_mip_level)) {

               perf_debug(cmd, "Decompressing in-place");

               unsigned level = view->vk.base_mip_level;
               unsigned layer = view->vk.base_array_layer;
               uint64_t base = hk_image_base_address(image, image_plane);

               struct libagx_decompress_images imgs;
               memcpy(&imgs.compressed, &view->planes[plane].emrt_texture,
                      sizeof(imgs.compressed));
               memcpy(&imgs.uncompressed, &view->planes[plane].emrt_pbe,
                      sizeof(imgs.uncompressed));

               struct agx_grid grid =
                  agx_3d(ail_metadata_width_tl(layout, level) * 32,
                         ail_metadata_height_tl(layout, level), layer_count);

               libagx_decompress(cmd, grid, AGX_BARRIER_ALL | AGX_PREGFX,
                                 layout, layer, level, base,
                                 hk_pool_upload(cmd, &imgs, sizeof(imgs), 64));
            }
         }
      }
   }

   uint32_t clear_count = 0;
   VkClearAttachment clear_att[HK_MAX_RTS + 1];
   bool resolved_clear = false;

   for (uint32_t i = 0; i < pRenderingInfo->colorAttachmentCount; i++) {
      const VkRenderingAttachmentInfo *att_info =
         &pRenderingInfo->pColorAttachments[i];
      if (att_info->imageView == VK_NULL_HANDLE ||
          att_info->loadOp != VK_ATTACHMENT_LOAD_OP_CLEAR)
         continue;

      clear_att[clear_count++] = (VkClearAttachment){
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .colorAttachment = i,
         .clearValue = att_info->clearValue,
      };

      render->color_att[i].clear = true;

      static_assert(sizeof(render->color_att[i].clear_colour) ==
                    sizeof(att_info->clearValue));
      memcpy(render->color_att[i].clear_colour, &att_info->clearValue,
             sizeof(att_info->clearValue));

      resolved_clear |= is_attachment_stored(att_info);
   }

   clear_att[clear_count] = (VkClearAttachment){
      .aspectMask = 0,
   };

   if (attach_z && attach_z->imageView != VK_NULL_HANDLE &&
       attach_z->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) {
      clear_att[clear_count].aspectMask |= VK_IMAGE_ASPECT_DEPTH_BIT;
      clear_att[clear_count].clearValue.depthStencil.depth =
         attach_z->clearValue.depthStencil.depth;

      render->depth_att.clear = true;
      render->depth_att.clear_colour[0] =
         fui(attach_z->clearValue.depthStencil.depth);

      resolved_clear |= is_attachment_stored(attach_z);
   }

   if (attach_s != NULL && attach_s->imageView != VK_NULL_HANDLE &&
       attach_s->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) {
      clear_att[clear_count].aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
      clear_att[clear_count].clearValue.depthStencil.stencil =
         attach_s->clearValue.depthStencil.stencil;

      render->stencil_att.clear = true;
      render->stencil_att.clear_colour[1] =
         attach_s->clearValue.depthStencil.stencil;

      resolved_clear |= is_attachment_stored(attach_s);
   }

   if (clear_att[clear_count].aspectMask != 0)
      clear_count++;

   if (clear_count > 0 && incomplete_render_area) {
      const VkClearRect clear_rect = {
         .rect = render->area,
         .baseArrayLayer = 0,
         .layerCount = render->view_mask ? 1 : render->layer_count,
      };

      hk_CmdClearAttachments(hk_cmd_buffer_to_handle(cmd), clear_count,
                             clear_att, 1, &clear_rect);
   } else {
      /* If a tile is empty, we do not want to process it, as the redundant
       * roundtrip of memory-->tilebuffer-->memory wastes a tremendous amount of
       * memory bandwidth. Any draw marks a tile as non-empty, so we only need
       * to process empty tiles if the background+EOT programs have a side
       * effect. This is the case exactly when there is an attachment we are
       * fast clearing and then storing.
       */
      cs->cr.process_empty_tiles = resolved_clear;
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdEndRendering(VkCommandBuffer commandBuffer)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_rendering_state *render = &cmd->state.gfx.render;

   /* The last control stream of the render pass is special since it gets its
    * stores dropped. Swap it in.
    */
   struct hk_cs *cs = cmd->current_cs.gfx;
   if (cs) {
      cs->cr.eot.main = render->cr.eot.main;
   }

   perf_debug(cmd, "End rendering");
   hk_cmd_buffer_end_graphics(cmd);

   bool need_resolve = false;

   /* Translate render state back to VK for meta */
   VkRenderingAttachmentInfo vk_color_att[HK_MAX_RTS];
   for (uint32_t i = 0; i < render->color_att_count; i++) {
      if (render->color_att[i].resolve_mode != VK_RESOLVE_MODE_NONE)
         need_resolve = true;

      vk_color_att[i] = (VkRenderingAttachmentInfo){
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = hk_image_view_to_handle(render->color_att[i].iview),
         .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
         .resolveMode = render->color_att[i].resolve_mode,
         .resolveImageView =
            hk_image_view_to_handle(render->color_att[i].resolve_iview),
         .resolveImageLayout = VK_IMAGE_LAYOUT_GENERAL,
      };
   }

   const VkRenderingAttachmentInfo vk_depth_att = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = hk_image_view_to_handle(render->depth_att.iview),
      .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
      .resolveMode = render->depth_att.resolve_mode,
      .resolveImageView =
         hk_image_view_to_handle(render->depth_att.resolve_iview),
      .resolveImageLayout = VK_IMAGE_LAYOUT_GENERAL,
   };
   if (render->depth_att.resolve_mode != VK_RESOLVE_MODE_NONE)
      need_resolve = true;

   const VkRenderingAttachmentInfo vk_stencil_att = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = hk_image_view_to_handle(render->stencil_att.iview),
      .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
      .resolveMode = render->stencil_att.resolve_mode,
      .resolveImageView =
         hk_image_view_to_handle(render->stencil_att.resolve_iview),
      .resolveImageLayout = VK_IMAGE_LAYOUT_GENERAL,
   };
   if (render->stencil_att.resolve_mode != VK_RESOLVE_MODE_NONE)
      need_resolve = true;

   const VkRenderingInfo vk_render = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = render->area,
      .layerCount = render->layer_count,
      .viewMask = render->view_mask,
      .colorAttachmentCount = render->color_att_count,
      .pColorAttachments = vk_color_att,
      .pDepthAttachment = &vk_depth_att,
      .pStencilAttachment = &vk_stencil_att,
   };

   if (render->flags & VK_RENDERING_SUSPENDING_BIT)
      need_resolve = false;

   memset(render, 0, sizeof(*render));

   if (need_resolve) {
      perf_debug(cmd, "Resolving render pass, colour store op %u",
                 vk_color_att[0].storeOp);

      hk_meta_resolve_rendering(cmd, &vk_render);
   }
}

static void
hk_init_heap(const void *data)
{
   struct hk_cmd_buffer *cmd = (struct hk_cmd_buffer *)data;
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   perf_debug(cmd, "Allocating heap");

   size_t size = 128 * 1024 * 1024;
   dev->heap = agx_bo_create(&dev->dev, size, 0, 0, "Geometry heap");

   /* The geometry state buffer is initialized here and then is treated by
    * the CPU as rodata, even though the GPU uses it for scratch internally.
    */
   off_t off = dev->rodata.heap - dev->rodata.bo->va->addr;
   struct poly_heap *map = agx_bo_map(dev->rodata.bo) + off;

   *map = (struct poly_heap){
      .base = dev->heap->va->addr,
      .size = size,
   };
}

static uint64_t
hk_heap(struct hk_cmd_buffer *cmd)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   util_call_once_data(&dev->heap_init_once, hk_init_heap, cmd);

   /* We need to free all allocations after each command buffer execution */
   if (!cmd->uses_heap) {
      perf_debug(cmd, "Freeing heap");
      uint64_t addr = dev->rodata.heap;

      /* Zeroing the allocated index frees everything */
      hk_queue_write(cmd, addr + offsetof(struct poly_heap, bottom), 0,
                     true /* after gfx */);

      cmd->uses_heap = true;
   }

   return dev->rodata.heap;
}

static uint64_t
hk_index_buffer(uint64_t index_buffer, uint size_el, uint offset_el,
                uint elsize_B)
{
   if (offset_el < size_el)
      return index_buffer + (offset_el * elsize_B);
   else
      return AGX_ZERO_PAGE_ADDRESS;
}

static uint64_t
hk_upload_vertex_params(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_descriptor_state *desc = &cmd->state.gfx.descriptors;

   const uint32_t wg_size[3] = {64, 1, 1};

   struct poly_vertex_params params;
   poly_vertex_params_init(&params, 0, wg_size);

   /* XXX: We should deduplicate this logic */
   bool indirect = agx_is_indirect(draw.b) || draw.restart;

   if (!indirect)
      poly_vertex_params_set_draw(&params, draw.b.count[0], draw.b.count[1]);

   if (draw.indexed) {
      unsigned index_size_B = agx_index_size_to_B(draw.index_size);
      unsigned range_el = agx_draw_index_range_el(draw);

      params.index_buffer = hk_index_buffer(agx_draw_index_buffer(draw),
                                            range_el, 0, index_size_B);

      params.index_buffer_range_el = range_el;
   }

   if (gfx->shaders[MESA_SHADER_TESS_EVAL] ||
       gfx->shaders[MESA_SHADER_GEOMETRY]) {

      struct hk_shader *vs = hk_bound_sw_vs(gfx);
      params.outputs = vs->b.info.outputs;

      if (!indirect) {
         uint32_t verts = draw.b.count[0], instances = draw.b.count[1];
         unsigned vb_size =
            poly_tcs_in_size(verts * instances, vs->b.info.outputs);

         /* Allocate if there are any outputs, or use the null sink to trap
          * reads if there aren't. Those reads are undefined but should not
          * fault. Affects:
          *
          *    dEQP-VK.pipeline.monolithic.no_position.explicit_declarations.basic.single_view.v0_g1
          */
         params.output_buffer = vb_size ? hk_pool_alloc(cmd, vb_size, 4).gpu
                                        : AGX_SCRATCH_PAGE_ADDRESS;
      }
   }

   desc->root.draw.vertex_outputs = params.outputs;

   return hk_pool_upload(cmd, &params, sizeof(params), 8);
}

static enum mesa_prim
hk_gs_in_prim(struct hk_cmd_buffer *cmd)
{
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_api_shader *tes = gfx->shaders[MESA_SHADER_TESS_EVAL];

   if (tes != NULL)
      return gfx->tess.prim;
   else
      return vk_topology_to_mesa(dyn->ia.primitive_topology);
}

static enum mesa_prim
hk_rast_prim(struct hk_cmd_buffer *cmd)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_api_shader *gs = gfx->shaders[MESA_SHADER_GEOMETRY];
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   if (gs != NULL) {
      return gs->variants[HK_GS_VARIANT_RAST].info.gs.mode;
   } else {
      switch (dyn->ia.primitive_topology) {
      case VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY:
      case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY:
         return MESA_PRIM_LINES;
      case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY:
      case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY:
         return MESA_PRIM_TRIANGLES;
      default:
         return hk_gs_in_prim(cmd);
      }
   }
}

static uint64_t
hk_upload_geometry_params(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_descriptor_state *desc = &cmd->state.gfx.descriptors;
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_api_shader *gs = gfx->shaders[MESA_SHADER_GEOMETRY];
   struct hk_shader *fs = hk_only_variant(gfx->shaders[MESA_SHADER_FRAGMENT]);
   struct hk_shader *count = hk_count_gs_variant(gs);

   /* XXX: We should deduplicate this logic */
   bool indirect = agx_is_indirect(draw.b) ||
                   gfx->shaders[MESA_SHADER_TESS_EVAL] || draw.restart;
   enum mesa_prim mode = hk_gs_in_prim(cmd);

   if (draw.restart) {
      mode = u_decomposed_prim(mode);
   }

   const uint32_t wg_size[3] = {64, 1, 1};

   struct poly_geometry_params params;
   poly_geometry_params_init(&params, mode, wg_size);

   params.flat_outputs = fs->info.fs.interp.flat;

   if (gfx->xfb_enabled) {
      for (unsigned i = 0; i < ARRAY_SIZE(gfx->xfb); ++i) {
         params.xfb_base_original[i] = gfx->xfb[i].addr;
         params.xfb_size[i] = gfx->xfb[i].range;
         params.xfb_offs_ptrs[i] = gfx->xfb_offsets + i * sizeof(uint32_t);
      }
   } else {
      for (unsigned i = 0; i < ARRAY_SIZE(gfx->xfb); ++i) {
         params.xfb_offs_ptrs[i] = AGX_ZERO_PAGE_ADDRESS;
         params.xfb_base_original[i] = AGX_ZERO_PAGE_ADDRESS;
         params.xfb_size[i] = 0;
      }
   }

   for (unsigned i = 0; i < ARRAY_SIZE(gfx->xfb_query); ++i) {
      uint64_t q = gfx->xfb_query[i];

      if (q) {
         params.xfb_prims_generated_counter[i] = q;
         params.prims_generated_counter[i] = q + sizeof(uint64_t);
      } else {
         params.xfb_prims_generated_counter[i] = AGX_SCRATCH_PAGE_ADDRESS;
         params.prims_generated_counter[i] = AGX_SCRATCH_PAGE_ADDRESS;
      }

      /* TODO: Optimize out? */
      params.xfb_overflow[i] = AGX_SCRATCH_PAGE_ADDRESS;
   }

   /* TODO: Optimize out? */
   params.xfb_any_overflow = AGX_SCRATCH_PAGE_ADDRESS;

   /* Calculate input primitive count for direct draws, and allocate the vertex
    * & count buffers. GPU calculates and allocates for indirect draws.
    */
   params.count_buffer_stride = count->info.gs.count_words * 4;

   if (!count->info.gs.prefix_sum && params.count_buffer_stride) {
      struct agx_ptr T = hk_pool_alloc(cmd, 16, 4);
      memset(T.cpu, 0, 16);
      params.count_buffer = T.gpu;
   }

   struct poly_gs_info *gsi = &count->info.gs;

   if (indirect) {
      if (gsi->shape == POLY_GS_SHAPE_DYNAMIC_INDEXED) {
         /* Need to allocate heap if we haven't yet */
         hk_heap(cmd);

         cmd->geom_index_buffer = dev->heap->va->addr;
         cmd->geom_index_count = dev->heap->size;
      } else {
         cmd->geom_index_count =
            poly_gs_rast_vertices(gsi->shape, gsi->max_indices, 1, 0);
      }
   } else {
      poly_geometry_params_set_draw(&params, mode, gsi->shape, gsi->max_indices,
                                    draw.b.count[0], draw.b.count[1]);

      unsigned size = params.input_primitives * params.count_buffer_stride;
      if (count->info.gs.prefix_sum && size) {
         params.count_buffer = hk_pool_alloc(cmd, size, 4).gpu;
      }

      cmd->geom_index_count = params.draw.index_count;
      cmd->geom_instance_count = params.draw.instance_count;

      if (gsi->shape == POLY_GS_SHAPE_DYNAMIC_INDEXED) {
         params.output_index_buffer =
            hk_pool_alloc(cmd, cmd->geom_index_count * 4, 4).gpu;

         cmd->geom_index_buffer = params.output_index_buffer;
      }
   }

   if (gsi->shape == POLY_GS_SHAPE_STATIC_INDEXED) {
      cmd->geom_index_buffer =
         hk_pool_upload(cmd, count->info.gs.topology, gsi->max_indices * 4, 4);
   }

   desc->root_dirty = true;
   return hk_pool_upload(cmd, &params, sizeof(params), 8);
}

static void
hk_upload_tess_params(struct hk_cmd_buffer *cmd, struct poly_tess_params *out,
                      struct agx_draw draw)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_shader *tcs = hk_only_variant(gfx->shaders[MESA_SHADER_TESS_CTRL]);

   enum poly_tess_partitioning partitioning =
      gfx->tess.info.spacing == TESS_SPACING_EQUAL
         ? POLY_TESS_PARTITIONING_INTEGER
      : gfx->tess.info.spacing == TESS_SPACING_FRACTIONAL_ODD
         ? POLY_TESS_PARTITIONING_FRACTIONAL_ODD
         : POLY_TESS_PARTITIONING_FRACTIONAL_EVEN;

   struct poly_tess_params args = {
      .heap = hk_heap(cmd),
      .tcs_stride_el = tcs->info.tess.tcs_output_stride / 4,
      .statistic = hk_pipeline_stat_addr(
         cmd,
         VK_QUERY_PIPELINE_STATISTIC_TESSELLATION_EVALUATION_SHADER_INVOCATIONS_BIT),

      .input_patch_size = dyn->ts.patch_control_points,
      .output_patch_size = tcs->info.tess.tcs_output_patch_size,
      .tcs_patch_constants = tcs->info.tess.tcs_nr_patch_outputs,
      .tcs_per_vertex_outputs = tcs->info.tess.tcs_per_vertex_outputs,
      .partitioning = partitioning,
      .points_mode = gfx->tess.info.points,
      .isolines = gfx->tess.info.mode == TESS_PRIMITIVE_ISOLINES,
   };

   if (!args.points_mode && gfx->tess.info.mode != TESS_PRIMITIVE_ISOLINES) {
      args.ccw = gfx->tess.info.ccw;
      args.ccw ^=
         dyn->ts.domain_origin == VK_TESSELLATION_DOMAIN_ORIGIN_LOWER_LEFT;
   }

   uint32_t draw_stride_el = 5;
   size_t draw_stride_B = draw_stride_el * sizeof(uint32_t);

   /* heap is allocated by hk_heap */
   args.patch_coord_buffer = dev->heap->va->addr;

   if (!agx_is_indirect(draw.b)) {
      unsigned in_patches = draw.b.count[0] / args.input_patch_size;
      unsigned unrolled_patches = in_patches * draw.b.count[1];

      uint32_t alloc = 0;
      uint32_t tcs_out_offs = alloc;
      alloc += unrolled_patches * args.tcs_stride_el * sizeof(uint32_t);

      uint32_t patch_coord_offs = alloc;
      alloc += unrolled_patches * sizeof(uint32_t);

      uint32_t count_offs = alloc;
      alloc += unrolled_patches * sizeof(uint32_t);

      /* Single API draw */
      uint32_t draw_offs = alloc;
      alloc += draw_stride_B;

      struct agx_ptr blob = hk_pool_alloc(cmd, alloc, 4);
      args.tcs_buffer = blob.gpu + tcs_out_offs;
      args.patches_per_instance = in_patches;
      args.coord_allocs = blob.gpu + patch_coord_offs;
      args.nr_patches = unrolled_patches;
      args.out_draws = blob.gpu + draw_offs;
      args.counts = blob.gpu + count_offs;
   } else {
      /* Allocate 3x indirect global+local grids for VS/TCS/tess */
      uint32_t grid_stride = sizeof(uint32_t) * 6;
      gfx->tess.grids = hk_pool_alloc(cmd, grid_stride * 3, 4).gpu;

      args.out_draws = hk_pool_alloc(cmd, draw_stride_B, 4).gpu;
   }

   gfx->tess.out_draws = args.out_draws;
   memcpy(out, &args, sizeof(args));
}

static struct hk_api_shader *
hk_build_meta_shader_locked(struct hk_device *dev, struct hk_internal_key *key,
                            hk_internal_builder_t builder)
{
   /* Try to get the cached shader */
   struct hash_entry *ent = _mesa_hash_table_search(dev->kernels.ht, key);
   if (ent)
      return ent->data;

   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, NULL);
   builder(&b, key->key);

   const struct vk_pipeline_robustness_state rs = {
      .images = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_DISABLED_EXT,
      .storage_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
      .uniform_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
      .vertex_inputs = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
   };

   struct vk_shader_compile_info info = {
      .stage = b.shader->info.stage,
      .nir = b.shader,
      .robustness = &rs,
   };

   hk_preprocess_nir_internal(dev->vk.physical, b.shader);

   struct hk_api_shader *s;
   if (hk_compile_shader(dev, &info, NULL, NULL, NULL, &s) != VK_SUCCESS)
      return NULL;

   /* ..and cache it before we return. The key is on the stack right now, so
    * clone it before using it as a hash table key. The clone is logically owned
    * by the hash table.
    */
   size_t total_key_size = sizeof(*key) + key->key_size;
   void *cloned_key = ralloc_memdup(dev->kernels.ht, key, total_key_size);

   _mesa_hash_table_insert(dev->kernels.ht, cloned_key, s);
   return s;
}

struct hk_api_shader *
hk_meta_shader(struct hk_device *dev, hk_internal_builder_t builder, void *data,
               size_t data_size)
{
   size_t total_key_size = sizeof(struct hk_internal_key) + data_size;

   struct hk_internal_key *key = alloca(total_key_size);
   key->builder = builder;
   key->key_size = data_size;

   if (data_size)
      memcpy(key->key, data, data_size);

   simple_mtx_lock(&dev->kernels.lock);
   struct hk_api_shader *s = hk_build_meta_shader_locked(dev, key, builder);
   simple_mtx_unlock(&dev->kernels.lock);

   return s;
}

static struct agx_draw
hk_draw_as_indexed_indirect(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   assert(draw.indexed);

   if (agx_is_indirect(draw.b))
      return draw;

   VkDrawIndexedIndirectCommand desc = {
      .indexCount = draw.b.count[0],
      .instanceCount = draw.b.count[1],
      .firstIndex = draw.start,
      .vertexOffset = draw.index_bias,
      .firstInstance = draw.start_instance,
   };

   return agx_draw_indexed_indirect(
      hk_pool_upload(cmd, &desc, sizeof(desc), 4), draw.index_buffer,
      draw.index_buffer_range_B, draw.index_size, draw.restart);
}

static struct agx_draw
hk_draw_without_restart(struct hk_cmd_buffer *cmd, struct agx_draw draw,
                        uint32_t draw_count)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   perf_debug(cmd, "Unrolling primitive restart due to GS/XFB");

   /* The unroll kernel assumes an indirect draw. Synthesize one if needed */
   draw = hk_draw_as_indexed_indirect(cmd, draw);

   /* Next, we unroll the index buffer used by the indirect draw */
   enum mesa_prim prim = vk_topology_to_mesa(dyn->ia.primitive_topology);

   assert(draw_count == 1 && "TODO: multidraw");

   struct libagx_unroll_restart_args ia = {
      .heap = hk_heap(cmd),
      .index_buffer = draw.index_buffer,
      .in_draw = draw.b.ptr,
      .out_draw = hk_pool_alloc(cmd, 5 * sizeof(uint32_t) * draw_count, 4).gpu,
      .max_draws = 1 /* TODO: MDI */,
      .restart_index = gfx->index.restart,
      .index_buffer_size_el = agx_draw_index_range_el(draw),
      .index_size_log2 = draw.index_size,
      .flatshade_first =
         dyn->rs.provoking_vertex == VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT,
   };

   libagx_unroll_restart_struct(cmd, agx_1d(1024 * draw_count),
                                AGX_BARRIER_ALL | AGX_PREGFX, ia,
                                poly_compact_prim(prim));

   return agx_draw_indexed_indirect(ia.out_draw, dev->heap->va->addr,
                                    dev->heap->size, draw.index_size,
                                    false /* restart */);
}

static struct agx_draw
hk_launch_gs_prerast(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                     struct agx_draw draw)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_descriptor_state *desc = &cmd->state.gfx.descriptors;
   struct hk_api_shader *gs = gfx->shaders[MESA_SHADER_GEOMETRY];
   struct agx_grid grid_vs, grid_gs;

   struct hk_shader *vs = hk_bound_sw_vs_before_gs(gfx);
   struct hk_shader *main = hk_main_gs_variant(gs);
   struct hk_shader *count = hk_count_gs_variant(gs);
   struct hk_shader *pre_gs = hk_pre_gs_variant(gs);

   uint64_t vertex_params = desc->root.draw.vertex_params;
   uint64_t geometry_params = desc->root.draw.geometry_params;
   unsigned count_words = count->info.gs.count_words;
   struct agx_workgroup wg = agx_workgroup(64, 1, 1);

   if (false /* TODO */)
      perf_debug(cmd, "Transform feedbck");
   else if (count_words)
      perf_debug(cmd, "Geometry shader with counts");
   else
      perf_debug(cmd, "Geometry shader without counts");

   enum mesa_prim mode = hk_gs_in_prim(cmd);

   if (draw.restart) {
      draw = hk_draw_without_restart(cmd, draw, 1);
      mode = u_decomposed_prim(mode);
   }

   /* Setup grids */
   if (agx_is_indirect(draw.b)) {
      struct libagx_gs_setup_indirect_args gsi = {
         .heap = hk_heap(cmd),
         .index_buffer = draw.index_buffer,
         .draw = draw.b.ptr,
         .vp = desc->root.draw.vertex_params,
         .p = desc->root.draw.geometry_params,
         .vs_outputs = vs->b.info.outputs,
         .prim = mode,
         .is_prefix_summing = count->info.gs.prefix_sum,
         .max_indices = count->info.gs.max_indices,
         .shape = count->info.gs.shape,
      };

      if (draw.indexed) {
         gsi.index_size_B = agx_index_size_to_B(draw.index_size);
         gsi.index_buffer_range_el = agx_draw_index_range_el(draw);
      }

      libagx_gs_setup_indirect_struct(cmd, agx_1d(1),
                                      AGX_BARRIER_ALL | AGX_PREGFX, gsi);

      grid_vs = agx_grid_indirect_local(
         vertex_params + offsetof(struct poly_vertex_params, grid));

      grid_gs = agx_grid_indirect_local(
         geometry_params + offsetof(struct poly_geometry_params, grid));
   } else {
      grid_vs = grid_gs = draw.b;
      grid_gs.count[0] = u_decomposed_prims_for_vertices(mode, draw.b.count[0]);
   }

   /* Launch the vertex shader first */
   hk_reserve_scratch(cmd, cs, vs);
   hk_dispatch_with_usc(dev, cs, &vs->b.info,
                        hk_upload_usc_words(cmd, vs,
                                            vs->info.stage == MESA_SHADER_VERTEX
                                               ? gfx->linked[MESA_SHADER_VERTEX]
                                               : vs->only_linked),
                        grid_vs, wg);

   /* Transform feedback and various queries require extra dispatching,
    * determine if we need that here.
    */
   VkQueryPipelineStatisticFlagBits gs_queries =
      VK_QUERY_PIPELINE_STATISTIC_GEOMETRY_SHADER_INVOCATIONS_BIT |
      VK_QUERY_PIPELINE_STATISTIC_GEOMETRY_SHADER_PRIMITIVES_BIT |
      VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT |
      VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT;

   struct hk_root_descriptor_table *root = &cmd->state.gfx.descriptors.root;
   bool xfb_or_queries =
      main->info.gs.xfb || (root->draw.pipeline_stats_flags & gs_queries);

   if (xfb_or_queries) {
      /* If we need counts, launch the count shader and prefix sum the results. */
      if (count_words) {
         perf_debug(dev, "Geometry shader count");
         hk_dispatch_with_local_size(cmd, cs, count, grid_gs, wg);
      }

      if (count->info.gs.prefix_sum) {
         perf_debug(dev, "Geometry shader transform feedback prefix sum");
         libagx_prefix_sum_geom(cmd, agx_1d(1024 * count_words),
                                AGX_BARRIER_ALL | AGX_PREGFX, geometry_params);
      }

      /* Transform feedback / query program */
      perf_debug(dev, "Transform feedback / geometry query");
      hk_dispatch_with_local_size(cmd, cs, pre_gs, agx_1d(1),
                                  agx_workgroup(1, 1, 1));
   }

   /* Pre-rast geometry shader */
   hk_dispatch_with_local_size(cmd, cs, main, grid_gs, wg);

   if (poly_gs_indexed(count->info.gs.shape)) {
      enum agx_index_size index_size =
         agx_translate_index_size(poly_gs_index_size(count->info.gs.shape));

      if (agx_is_indirect(draw.b)) {
         return agx_draw_indexed_indirect(
            geometry_params + offsetof(struct poly_geometry_params, draw),
            cmd->geom_index_buffer, cmd->geom_index_count, index_size, true);
      } else {
         return agx_draw_indexed(cmd->geom_index_count,
                                 cmd->geom_instance_count, 0, 0, 0,
                                 cmd->geom_index_buffer,
                                 cmd->geom_index_count * 4, index_size, true);
      }
   } else {
      if (agx_is_indirect(draw.b)) {
         return agx_draw_indirect(geometry_params +
                                  offsetof(struct poly_geometry_params, draw));
      } else {
         return (struct agx_draw){
            .b = agx_3d(cmd->geom_index_count, cmd->geom_instance_count, 1),
         };
      }
   }
}

static struct agx_draw
hk_launch_tess(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
               struct agx_draw draw, uint64_t c_prims, uint64_t c_inv)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct agx_grid grid_vs, grid_tcs, grid_tess;

   struct hk_shader *vs = hk_bound_sw_vs(gfx);
   struct hk_shader *tcs = hk_only_variant(gfx->shaders[MESA_SHADER_TESS_CTRL]);

   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   uint32_t input_patch_size = dyn->ts.patch_control_points;
   uint64_t state = gfx->descriptors.root.draw.tess_params;
   struct hk_tess_info info = gfx->tess.info;

   perf_debug(cmd, "Tessellation");

   uint64_t tcs_stat = hk_pipeline_stat_addr(
      cmd, VK_QUERY_PIPELINE_STATISTIC_TESSELLATION_CONTROL_SHADER_PATCHES_BIT);

   /* Setup grids */
   if (agx_is_indirect(draw.b)) {
      perf_debug(cmd, "Indirect tessellation");

      struct libagx_tess_setup_indirect_args args = {
         .p = state,
         .grids = gfx->tess.grids,
         .indirect = draw.b.ptr,
         .vp = gfx->descriptors.root.draw.vertex_params,
         .vertex_outputs = vs->b.info.outputs,
         .tcs_statistic = hk_pipeline_stat_addr(
            cmd,
            VK_QUERY_PIPELINE_STATISTIC_TESSELLATION_CONTROL_SHADER_PATCHES_BIT),
      };

      if (draw.indexed) {
         args.in_index_buffer = draw.index_buffer;
         args.in_index_size_B = agx_index_size_to_B(draw.index_size);
         args.in_index_buffer_range_el = agx_draw_index_range_el(draw);
      }

      libagx_tess_setup_indirect_struct(cmd, agx_1d(1),
                                        AGX_BARRIER_ALL | AGX_PREGFX, args);

      uint32_t grid_stride = sizeof(uint32_t) * 6;
      grid_vs = agx_grid_indirect_local(gfx->tess.grids + 0 * grid_stride);
      grid_tcs = agx_grid_indirect_local(gfx->tess.grids + 1 * grid_stride);
      grid_tess = agx_grid_indirect_local(gfx->tess.grids + 2 * grid_stride);
   } else {
      uint32_t patches = draw.b.count[0] / input_patch_size;
      grid_vs = grid_tcs = draw.b;

      grid_tcs.count[0] = patches * tcs->info.tess.tcs_output_patch_size;
      grid_tess = agx_1d(patches * draw.b.count[1]);

      /* TCS invocation counter increments once per-patch */
      if (hk_stat_enabled(tcs_stat)) {
         perf_debug(cmd, "Direct TCS statistic");
         libagx_increment_statistic(
            cmd, agx_1d(1), AGX_BARRIER_ALL | AGX_PREGFX, tcs_stat, patches);
      }
   }

   /* First launch the VS and TCS */
   hk_reserve_scratch(cmd, cs, vs);
   hk_reserve_scratch(cmd, cs, tcs);

   hk_dispatch_with_usc(
      dev, cs, &vs->b.info,
      hk_upload_usc_words(cmd, vs, gfx->linked[MESA_SHADER_VERTEX]), grid_vs,
      agx_workgroup(64, 1, 1));

   hk_dispatch_with_usc(
      dev, cs, &tcs->b.info, hk_upload_usc_words(cmd, tcs, tcs->only_linked),
      grid_tcs, agx_workgroup(tcs->info.tess.tcs_output_patch_size, 1, 1));

   /* First generate counts, then prefix sum them, and then tessellate. */
   libagx_tessellate(cmd, grid_tess, AGX_BARRIER_ALL | AGX_PREGFX, info.mode,
                     POLY_TESS_MODE_COUNT, state);

   libagx_prefix_sum_tess(cmd, agx_1d(1024), AGX_BARRIER_ALL | AGX_PREGFX,
                          state, c_prims, c_inv, c_prims || c_inv);

   libagx_tessellate(cmd, grid_tess, AGX_BARRIER_ALL | AGX_PREGFX, info.mode,
                     POLY_TESS_MODE_WITH_COUNTS, state);

   return agx_draw_indexed_indirect(gfx->tess.out_draws, dev->heap->va->addr,
                                    dev->heap->size, AGX_INDEX_SIZE_U32, false);
}

void
hk_cmd_bind_graphics_shader(struct hk_cmd_buffer *cmd,
                            const mesa_shader_stage stage,
                            struct hk_api_shader *shader)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   /* Null fragment shaders are annoying to handle, because we may still need an
    * actual fragment shader to attach a prolog to. Rather than adding special
    * cases, we just bind an empty fragment shader instead of NULL to make
    * everything work correctly.
    */
   if (stage == MESA_SHADER_FRAGMENT && shader == NULL) {
      shader = dev->null_fs;
   }

   assert(stage < ARRAY_SIZE(cmd->state.gfx.shaders));
   if (cmd->state.gfx.shaders[stage] == shader)
      return;

   cmd->state.gfx.shaders[stage] = shader;
   cmd->state.gfx.shaders_dirty |= BITFIELD_BIT(stage);

   if (stage == MESA_SHADER_FRAGMENT) {
      BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_MS_RASTERIZATION_SAMPLES);
   }
}

static void
hk_flush_shaders(struct hk_cmd_buffer *cmd)
{
   if (cmd->state.gfx.shaders_dirty == 0)
      return;

   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_descriptor_state *desc = &cmd->state.gfx.descriptors;
   desc->root_dirty = true;

   /* Geometry shading overrides the restart index, reemit on rebind */
   if (IS_SHADER_DIRTY(GEOMETRY)) {
      struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
      struct hk_api_shader *gs = gfx->shaders[MESA_SHADER_GEOMETRY];

      desc->root.draw.api_gs = gs && !gs->is_passthrough;
      BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_RS_RASTERIZER_DISCARD_ENABLE);
   }

   struct hk_shader *hw_vs = hk_bound_hw_vs(gfx);
   struct hk_api_shader *fs = gfx->shaders[MESA_SHADER_FRAGMENT];

   /* If we have a new VS/FS pair, UVS locations may have changed so need to
    * relink. We do this here because there's no dependence on the fast linked
    * shaders.
    */
   agx_assign_uvs(&gfx->linked_varyings, &hw_vs->info.uvs,
                  hk_only_variant(fs)->info.fs.interp.flat,
                  hk_only_variant(fs)->info.fs.interp.linear);

   for (unsigned i = 0; i < VARYING_SLOT_MAX; ++i) {
      desc->root.draw.uvs_index[i] = gfx->linked_varyings.slots[i];
   }
}

static struct agx_shader_part *
hk_get_prolog_epilog_locked(struct hk_device *dev, struct hk_internal_key *key,
                            hk_internal_builder_t builder, bool preprocess_nir,
                            bool stop, unsigned cf_base)
{
   /* Try to get the cached shader */
   struct hash_entry *ent = _mesa_hash_table_search(dev->prolog_epilog.ht, key);
   if (ent)
      return ent->data;

   nir_builder b = nir_builder_init_simple_shader(0, &agx_nir_options, NULL);
   builder(&b, key->key);

   if (preprocess_nir)
      agx_preprocess_nir(b.shader);

   struct agx_shader_key backend_key = {
      .dev = agx_gather_device_key(&dev->dev),
      .secondary = true,
      .no_stop = !stop,
   };

   /* We always use dynamic sample shading in the GL driver. Indicate that. */
   if (b.shader->info.stage == MESA_SHADER_FRAGMENT) {
      backend_key.fs.cf_base = cf_base;

      if (b.shader->info.fs.uses_sample_shading)
         backend_key.fs.inside_sample_loop = true;
   }

   struct agx_shader_part *part =
      rzalloc(dev->prolog_epilog.ht, struct agx_shader_part);
   if (!part) {
      ralloc_free(b.shader);
      return NULL;
   }

   const bool compiled = agx_compile_shader_nir(b.shader, &backend_key, part);

   ralloc_free(b.shader);
   if (!compiled) {
      /* A rejected backend instruction leaves an empty part. Caching that
       * as success lets fast-link silently omit required output/blend code. */
      ralloc_free(part);
      return NULL;
   }

   /* ..and cache it before we return. The key is on the stack right now, so
    * clone it before using it as a hash table key. The clone is logically owned
    * by the hash table.
    */
   size_t total_key_size = sizeof(*key) + key->key_size;
   void *cloned_key = ralloc_memdup(dev->prolog_epilog.ht, key, total_key_size);
   if (!cloned_key) {
      free(part->binary);
      ralloc_free(part);
      return NULL;
   }

   _mesa_hash_table_insert(dev->prolog_epilog.ht, cloned_key, part);
   return part;
}

static struct agx_shader_part *
hk_get_prolog_epilog(struct hk_device *dev, void *data, size_t data_size,
                     hk_internal_builder_t builder, bool preprocess_nir,
                     bool stop, unsigned cf_base)
{
   /* Build the meta shader key */
   size_t total_key_size = sizeof(struct hk_internal_key) + data_size;

   struct hk_internal_key *key = alloca(total_key_size);
   key->builder = builder;
   key->key_size = data_size;

   if (data_size)
      memcpy(key->key, data, data_size);

   simple_mtx_lock(&dev->prolog_epilog.lock);

   struct agx_shader_part *part = hk_get_prolog_epilog_locked(
      dev, key, builder, preprocess_nir, stop, cf_base);

   simple_mtx_unlock(&dev->prolog_epilog.lock);
   return part;
}

static struct hk_linked_shader *
hk_get_fast_linked_locked_vs(struct hk_device *dev, struct hk_shader *shader,
                             struct hk_fast_link_key_vs *key)
{
   struct agx_shader_part *prolog =
      hk_get_prolog_epilog(dev, &key->prolog, sizeof(key->prolog),
                           dev->dev.chip == AGX_CHIP_G17P
                              ? agx_nir_vs_prolog_raw : agx_nir_vs_prolog,
                           false, false, 0);
   if (!prolog)
      return NULL;

   struct hk_fast_link_key *key_clone =
      ralloc_memdup(shader->linked.ht, key, sizeof(*key));
   if (!key_clone)
      return NULL;

   struct hk_linked_shader *linked =
      hk_fast_link(dev, false, shader, prolog, NULL, 0);
   if (!linked) {
      ralloc_free(key_clone);
      return NULL;
   }

   /* XXX: Fix this higher up the stack */
   linked->sw_indexing = !key->prolog.hw || key->prolog.adjacency;
   linked->b.uses_base_param |= linked->sw_indexing;

   _mesa_hash_table_insert(shader->linked.ht, key_clone, linked);
   return linked;
}

static bool
lower_fs_root(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_root_agx)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def_replace(&intr->def,
                   nir_load_preamble(b, 1, 64, .base = AGX_ABI_FUNI_ROOT));
   return true;
}

static void
build_fs_prolog(nir_builder *b, const void *key)
{
   agx_nir_fs_prolog(b, key);

   /* Lower load_stat_query_address_agx, needed for FS statistics */
   NIR_PASS(_, b->shader, hk_lower_uvs_index, MESA_SHADER_FRAGMENT, 0);
   NIR_PASS(_, b->shader, nir_shader_intrinsics_pass, lower_fs_root,
            nir_metadata_control_flow, NULL);
}

static void
build_g17p_fs_epilog(nir_builder *b, const void *key_)
{
   const struct agx_fs_epilog_key *key = key_;
   agx_nir_fs_epilog(b, key);

   if (key->nr_samples == 1)
      NIR_PASS(_, b->shader, agx_nir_lower_g17p_1x_tile_samples);
}

static struct hk_linked_shader *
hk_get_fast_linked_locked_fs(struct hk_device *dev, struct hk_shader *shader,
                             struct hk_fast_link_key_fs *key)
{
   /* TODO: prolog without fs needs to work too... */
   bool needs_prolog = key->prolog.statistics ||
                       key->prolog.cull_distance_size ||
                       key->prolog.api_sample_mask != 0xff;

   struct agx_shader_part *prolog = NULL;
   if (needs_prolog) {
      prolog = hk_get_prolog_epilog(dev, &key->prolog, sizeof(key->prolog),
                                    build_fs_prolog, false, false,
                                    key->prolog.cf_base);
      if (!prolog)
         return NULL;
   }

   /* If sample shading is used, don't stop at the epilog, there's a
    * footer that the fast linker will insert to stop.
    */
   bool epilog_stop = (key->nr_samples_shaded == 0);

   struct agx_shader_part *epilog =
      hk_get_prolog_epilog(dev, &key->epilog, sizeof(key->epilog),
                           dev->dev.chip == AGX_CHIP_G17P
                              ? build_g17p_fs_epilog
                              : agx_nir_fs_epilog,
                           true, epilog_stop, 0);
   if (!epilog)
      return NULL;

   struct hk_fast_link_key *key_clone =
      ralloc_memdup(shader->linked.ht, key, sizeof(*key));
   if (!key_clone)
      return NULL;

   struct hk_linked_shader *linked =
      hk_fast_link(dev, true, shader, prolog, epilog, key->nr_samples_shaded);
   if (!linked) {
      ralloc_free(key_clone);
      return NULL;
   }

   _mesa_hash_table_insert(shader->linked.ht, key_clone, linked);
   return linked;
}

/*
 * First, look for a fully linked variant. Else, build the required shader
 * parts and link.
 */
static struct hk_linked_shader *
hk_get_fast_linked(struct hk_device *dev, struct hk_shader *shader, void *key)
{
   struct hk_linked_shader *linked;
   simple_mtx_lock(&shader->linked.lock);

   struct hash_entry *ent = _mesa_hash_table_search(shader->linked.ht, key);

   if (ent)
      linked = ent->data;
   else if (shader->info.stage == MESA_SHADER_VERTEX)
      linked = hk_get_fast_linked_locked_vs(dev, shader, key);
   else if (shader->info.stage == MESA_SHADER_FRAGMENT)
      linked = hk_get_fast_linked_locked_fs(dev, shader, key);
   else
      UNREACHABLE("invalid stage");

   simple_mtx_unlock(&shader->linked.lock);
   return linked;
}

static bool
hk_update_fast_linked(struct hk_cmd_buffer *cmd, struct hk_shader *shader,
                      void *key)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_linked_shader *new = hk_get_fast_linked(dev, shader, key);
   mesa_shader_stage stage = shader->info.stage;
   if (!new) {
      mesa_loge("failed to compile or link required %s shader parts",
                _mesa_shader_stage_to_string(stage));
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return false;
   }

   if (cmd->state.gfx.linked[stage] != new) {
      cmd->state.gfx.linked[stage] = new;
      cmd->state.gfx.linked_dirty |= BITFIELD_BIT(stage);
   }
   return true;
}

static enum agx_polygon_mode
translate_polygon_mode(VkPolygonMode vk_mode)
{
   static_assert((enum agx_polygon_mode)VK_POLYGON_MODE_FILL ==
                 AGX_POLYGON_MODE_FILL);
   static_assert((enum agx_polygon_mode)VK_POLYGON_MODE_LINE ==
                 AGX_POLYGON_MODE_LINE);
   static_assert((enum agx_polygon_mode)VK_POLYGON_MODE_POINT ==
                 AGX_POLYGON_MODE_POINT);

   assert(vk_mode <= VK_POLYGON_MODE_POINT);
   return (enum agx_polygon_mode)vk_mode;
}

static enum agx_zs_func
translate_compare_op(VkCompareOp vk_mode)
{
   static_assert((enum agx_zs_func)VK_COMPARE_OP_NEVER == AGX_ZS_FUNC_NEVER);
   static_assert((enum agx_zs_func)VK_COMPARE_OP_LESS == AGX_ZS_FUNC_LESS);
   static_assert((enum agx_zs_func)VK_COMPARE_OP_EQUAL == AGX_ZS_FUNC_EQUAL);
   static_assert((enum agx_zs_func)VK_COMPARE_OP_LESS_OR_EQUAL ==
                 AGX_ZS_FUNC_LEQUAL);
   static_assert((enum agx_zs_func)VK_COMPARE_OP_GREATER ==
                 AGX_ZS_FUNC_GREATER);
   static_assert((enum agx_zs_func)VK_COMPARE_OP_NOT_EQUAL ==
                 AGX_ZS_FUNC_NOT_EQUAL);
   static_assert((enum agx_zs_func)VK_COMPARE_OP_GREATER_OR_EQUAL ==
                 AGX_ZS_FUNC_GEQUAL);
   static_assert((enum agx_zs_func)VK_COMPARE_OP_ALWAYS == AGX_ZS_FUNC_ALWAYS);

   assert(vk_mode <= VK_COMPARE_OP_ALWAYS);
   return (enum agx_zs_func)vk_mode;
}

static enum agx_stencil_op
translate_stencil_op(VkStencilOp vk_op)
{
   static_assert((enum agx_stencil_op)VK_STENCIL_OP_KEEP ==
                 AGX_STENCIL_OP_KEEP);
   static_assert((enum agx_stencil_op)VK_STENCIL_OP_ZERO ==
                 AGX_STENCIL_OP_ZERO);
   static_assert((enum agx_stencil_op)VK_STENCIL_OP_REPLACE ==
                 AGX_STENCIL_OP_REPLACE);
   static_assert((enum agx_stencil_op)VK_STENCIL_OP_INCREMENT_AND_CLAMP ==
                 AGX_STENCIL_OP_INCR_SAT);
   static_assert((enum agx_stencil_op)VK_STENCIL_OP_DECREMENT_AND_CLAMP ==
                 AGX_STENCIL_OP_DECR_SAT);
   static_assert((enum agx_stencil_op)VK_STENCIL_OP_INVERT ==
                 AGX_STENCIL_OP_INVERT);
   static_assert((enum agx_stencil_op)VK_STENCIL_OP_INCREMENT_AND_WRAP ==
                 AGX_STENCIL_OP_INCR_WRAP);
   static_assert((enum agx_stencil_op)VK_STENCIL_OP_DECREMENT_AND_WRAP ==
                 AGX_STENCIL_OP_DECR_WRAP);

   return (enum agx_stencil_op)vk_op;
}

static struct agx_fragment_stencil_packed
hk_pack_stencil_face(struct vk_stencil_test_face_state s, bool enabled)
{
   struct agx_fragment_stencil_packed packed;

   if (enabled) {
      agx_pack(&packed, FRAGMENT_STENCIL, cfg) {
         cfg.compare = translate_compare_op(s.op.compare);
         cfg.write_mask = s.write_mask;
         cfg.read_mask = s.compare_mask;

         cfg.depth_pass = translate_stencil_op(s.op.pass);
         cfg.depth_fail = translate_stencil_op(s.op.depth_fail);
         cfg.stencil_fail = translate_stencil_op(s.op.fail);
      }
   } else {
      agx_pack(&packed, FRAGMENT_STENCIL, cfg) {
         cfg.compare = AGX_ZS_FUNC_ALWAYS;
         cfg.write_mask = 0xFF;
         cfg.read_mask = 0xFF;

         cfg.depth_pass = AGX_STENCIL_OP_KEEP;
         cfg.depth_fail = AGX_STENCIL_OP_KEEP;
         cfg.stencil_fail = AGX_STENCIL_OP_KEEP;
      }
   }

   return packed;
}

static void
hk_ppp_push_stencil_face(struct agx_ppp_update *ppp,
                         struct vk_stencil_test_face_state s, bool enabled)
{
   struct agx_fragment_stencil_packed packed =
      hk_pack_stencil_face(s, enabled);
   agx_ppp_push_packed(ppp, &packed, FRAGMENT_STENCIL);
}

static bool
hk_stencil_test_enabled(struct hk_cmd_buffer *cmd)
{
   const struct hk_rendering_state *render = &cmd->state.gfx.render;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   return dyn->ds.stencil.test_enable &&
          render->stencil_att.vk_format != VK_FORMAT_UNDEFINED;
}

static bool
hk_emit_hal300_vp_state(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                        enum agx_chip chip, unsigned viewport_count,
                        unsigned scissor_index, const unsigned minx[],
                        const unsigned miny[], const unsigned maxx[],
                        const unsigned maxy[], const VkViewport viewports[],
                        uint8_t **out)
{
   if (!out || !*out || viewport_count == 0 ||
       viewport_count > AGX_PPP_VIEWPORT_PAYLOAD_MAX_VIEWPORTS ||
       !agx_ppp_hal300_depth_bias_scissor_core_supported(chip))
      return false;

   const size_t scissor_offset =
      (size_t)scissor_index * AGX_PPP_HAL300_SCISSOR_RECORD_LENGTH;
   const size_t scissor_size =
      (size_t)viewport_count * AGX_PPP_HAL300_SCISSOR_RECORD_LENGTH;
   if (scissor_offset > cs->scissor.size ||
       scissor_size > cs->scissor.size - scissor_offset)
      return false;

   const uint64_t scissor_base =
      hk_pool_upload(cmd, (const uint8_t *)cs->scissor.data + scissor_offset,
                     scissor_size, 64);

   static const uint8_t zero_depth_bias[AGX_PPP_HAL300_DEPTH_BIAS_RECORD_LENGTH];
   const void *depth_bias = zero_depth_bias;
   if (cs->depth_bias.size >= AGX_PPP_HAL300_DEPTH_BIAS_RECORD_LENGTH) {
      depth_bias = (const uint8_t *)cs->depth_bias.data + cs->depth_bias.size -
                   AGX_PPP_HAL300_DEPTH_BIAS_RECORD_LENGTH;
   }

   const uint64_t depth_bias_base =
      hk_pool_upload(cmd, depth_bias,
                     AGX_PPP_HAL300_DEPTH_BIAS_RECORD_LENGTH, 64);
   if (!scissor_base || !depth_bias_base)
      return false;

   struct agx_ppp_hal300_depth_bias_scissor_bases bases;
   const struct agx_ppp_hal300_depth_bias_scissor_bases_fields base_fields = {
      .depth_bias_base = depth_bias_base,
      .scissor_base = scissor_base,
   };
   if (!agx_ppp_hal300_depth_bias_scissor_bases_pack(chip, &base_fields,
                                                     &bases))
      return false;

   struct agx_ppp_viewport_payload_fields viewport_fields = {
      .viewport_count = viewport_count,
   };

   for (unsigned i = 0; i < viewport_count; ++i) {
      uint8_t region_clip[AGX_REGION_CLIP_LENGTH];
      agx_pack(region_clip, REGION_CLIP, cfg) {
         cfg.enable = true;
         cfg.min_x = minx[i] / 32;
         cfg.min_y = miny[i] / 32;
         cfg.max_x = DIV_ROUND_UP(MAX2(maxx[i], 1), 32);
         cfg.max_y = DIV_ROUND_UP(MAX2(maxy[i], 1), 32);
      }
      memcpy(&viewport_fields.region_clip_raw[i], region_clip,
             sizeof(region_clip));

      const VkViewport *vp = &viewports[i];
      viewport_fields.viewports[i] =
         (struct agx_ppp_viewport_transform_raw){
            .translate_x = fui(vp->x + 0.5f * vp->width),
            .scale_x = fui(vp->width * 0.5f),
            .translate_y = fui(vp->y + 0.5f * vp->height),
            .scale_y = fui(vp->height * 0.5f),
            .translate_z = fui(vp->minDepth),
            .scale_z = fui(vp->maxDepth - vp->minDepth),
         };
   }

   uint8_t viewport_control[AGX_VIEWPORT_CONTROL_LENGTH];
   agx_pack(viewport_control, VIEWPORT_CONTROL, cfg)
      ;
   memcpy(&viewport_fields.viewport_control_raw, viewport_control,
          sizeof(viewport_control));

   struct agx_ppp_viewport_payload_core viewport;
   if (!agx_ppp_viewport_payload_core_pack(chip, &viewport_fields, &viewport))
      return false;

   const uint64_t bases_gpu =
      hk_pool_upload(cmd, bases.bytes, bases.length, AGX_PPP_HEADER_ALIGN);
   const uint64_t viewport_gpu =
      hk_pool_upload(cmd, viewport.bytes, viewport.length,
                     AGX_PPP_HEADER_ALIGN);
   if (!bases_gpu || !viewport_gpu)
      return false;

   /* Keep emission transactional: both exact PPP pointers are staged before
    * the caller's live control-stream cursor is advanced. HAL300's initial
    * render-state path emits the viewport update before the base update. */
   uint8_t records[2 * AGX_PPP_STATE_LENGTH];
   uint8_t *record = records;
   if (!agx_ppp_emit_state(chip, &record, viewport_gpu, viewport.length) ||
       !agx_ppp_emit_state(chip, &record, bases_gpu, bases.length) ||
       record != records + sizeof(records))
      return false;

   memcpy(*out, records, sizeof(records));
   *out += sizeof(records);
   return true;
}

static void
hk_flush_vp_state(struct hk_cmd_buffer *cmd, struct hk_cs *cs, uint8_t **out)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   const bool hal300 =
      agx_chip_command_stream_generation(dev->dev.chip) ==
      AGX_CS_GENERATION_HAL300;
   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;

   static_assert(AGX_SCISSOR_LENGTH ==
                 AGX_PPP_HAL300_SCISSOR_RECORD_LENGTH);

   /* We always need at least 1 viewport for the hardware. With rasterizer
    * discard the app may not supply any, but we can just program garbage.
    */
   unsigned count = MAX2(dyn->vp.viewport_count, 1);

   unsigned minx[HK_MAX_VIEWPORTS] = {0}, miny[HK_MAX_VIEWPORTS] = {0};
   unsigned maxx[HK_MAX_VIEWPORTS] = {0}, maxy[HK_MAX_VIEWPORTS] = {0};

   /* We implicitly scissor to the viewport. We need to do a min/max dance to
    * handle inverted viewports.
    */
   for (uint32_t i = 0; i < dyn->vp.viewport_count; i++) {
      const VkViewport *vp = &dyn->vp.viewports[i];

      minx[i] = MIN2(vp->x, vp->x + vp->width);
      miny[i] = MIN2(vp->y, vp->y + vp->height);
      maxx[i] = MAX2(vp->x, vp->x + vp->width);
      maxy[i] = MAX2(vp->y, vp->y + vp->height);
   }

   /* Additionally clamp to the framebuffer so we don't rasterize
    * off-screen pixels. TODO: Is this necessary? the GL driver does this but
    * it might be cargoculted at this point.
    */
   for (unsigned i = 0; i < count; ++i) {
      minx[i] = MIN2(minx[i], cmd->state.gfx.render.cr.width);
      maxx[i] = MIN2(maxx[i], cmd->state.gfx.render.cr.width);
      miny[i] = MIN2(miny[i], cmd->state.gfx.render.cr.height);
      maxy[i] = MIN2(maxy[i], cmd->state.gfx.render.cr.height);
   }

   /* We additionally apply any API scissors */
   for (unsigned i = 0; i < dyn->vp.scissor_count; ++i) {
      const VkRect2D *s = &dyn->vp.scissors[i];

      minx[i] = MAX2(minx[i], s->offset.x);
      miny[i] = MAX2(miny[i], s->offset.y);
      maxx[i] = MIN2(maxx[i], s->offset.x + s->extent.width);
      maxy[i] = MIN2(maxy[i], s->offset.y + s->extent.height);
   }

   /* Upload a hardware scissor for each viewport, whether there's a
    * corresponding API scissor or not.
    */
   unsigned index = cs->scissor.size / AGX_SCISSOR_LENGTH;
   uint8_t *scissors =
      util_dynarray_grow_bytes(&cs->scissor, count, AGX_SCISSOR_LENGTH);

   for (unsigned i = 0; i < count; ++i) {
      const VkViewport *vp = &dyn->vp.viewports[i];

      /* These settings in conjunction with the PPP control depth clip/clamp
       * settings implement depth clip/clamping. Properly setting them
       * together is required for conformant depth clip enable.
       *
       * TODO: Reverse-engineer the finer interactions here.
       */
      const float min_z = dyn->rs.depth_clamp_enable
                             ? MIN2(vp->minDepth, vp->maxDepth)
                             : 0.0f;
      const float max_z = dyn->rs.depth_clamp_enable
                             ? MAX2(vp->minDepth, vp->maxDepth)
                             : 1.0f;
      uint8_t *record = scissors + i * AGX_SCISSOR_LENGTH;

      if (hal300) {
         const struct agx_ppp_hal300_scissor_record_fields fields = {
            .max_x = maxx[i],
            .min_x = minx[i],
            .max_y = maxy[i],
            .min_y = miny[i],
            .min_z_raw = fui(min_z),
            .max_z_raw = fui(max_z),
         };
         struct agx_ppp_hal300_scissor_record exact;
         if (!agx_ppp_hal300_scissor_record_pack(dev->dev.chip, &fields,
                                                  &exact)) {
            mesa_loge("no grounded HAL300 scissor record for %s",
                      agx_chip_name(dev->dev.chip));
            vk_command_buffer_set_error(&cmd->vk,
                                        VK_ERROR_FEATURE_NOT_PRESENT);
            return;
         }
         memcpy(record, exact.bytes, exact.length);
      } else {
         agx_pack(record, SCISSOR, cfg) {
            cfg.min_x = minx[i];
            cfg.min_y = miny[i];
            cfg.max_x = maxx[i];
            cfg.max_y = maxy[i];
            cfg.min_z = min_z;
            cfg.max_z = max_z;
         }
      }
   }

   if (hal300) {
      if (!hk_emit_hal300_vp_state(cmd, cs, dev->dev.chip, count, index, minx,
                                   miny, maxx, maxy, dyn->vp.viewports, out)) {
         mesa_loge("no grounded HAL300 viewport/base emitter for %s",
                   agx_chip_name(dev->dev.chip));
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      }
      return;
   }

   /* Upload state */
   struct AGX_PPP_HEADER present = {
      .depth_bias_scissor = true,
      .region_clip = true,
      .viewport = true,
      .viewport_count = count,
   };

   size_t size = agx_ppp_update_size(&present);
   struct agx_ptr T = hk_pool_vdm_alloc(cmd, size, AGX_PPP_HEADER_ALIGN);
   if (!T.cpu)
      return;

   struct agx_ppp_update ppp = agx_new_ppp_update(T, size, &present);

   agx_ppp_push(&ppp, DEPTH_BIAS_SCISSOR, cfg) {
      cfg.scissor = index;

      /* Use the current depth bias, we allocate linearly */
      unsigned count = cs->depth_bias.size / AGX_DEPTH_BIAS_LENGTH;
      cfg.depth_bias = count ? count - 1 : 0;
   };

   for (unsigned i = 0; i < count; ++i) {
      agx_ppp_push(&ppp, REGION_CLIP, cfg) {
         cfg.enable = true;
         cfg.min_x = minx[i] / 32;
         cfg.min_y = miny[i] / 32;
         cfg.max_x = DIV_ROUND_UP(MAX2(maxx[i], 1), 32);
         cfg.max_y = DIV_ROUND_UP(MAX2(maxy[i], 1), 32);
      }
   }

   agx_ppp_push(&ppp, VIEWPORT_CONTROL, cfg)
      ;

   /* Upload viewports */
   for (unsigned i = 0; i < count; ++i) {
      const VkViewport *vp = &dyn->vp.viewports[i];

      agx_ppp_push(&ppp, VIEWPORT, cfg) {
         cfg.translate_x = vp->x + 0.5f * vp->width;
         cfg.translate_y = vp->y + 0.5f * vp->height;
         cfg.translate_z = vp->minDepth;

         cfg.scale_x = vp->width * 0.5f;
         cfg.scale_y = vp->height * 0.5f;
         cfg.scale_z = vp->maxDepth - vp->minDepth;
      }
   }

   const bool ppp_ok = agx_ppp_legacy_payload_supported(dev->dev.chip) &&
                       agx_ppp_fini(dev->dev.chip, out, &ppp);
   if (!ppp_ok) {
      mesa_loge("no grounded legacy PPP payload/finalizer for %s",
                agx_chip_name(dev->dev.chip));
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
   } else if (hk_g17p_launch_trace_enabled()) {
      hk_g17p_trace_line("ppp-viewport", ppp.gpu_base, ppp.cpu_base,
                         ppp.total_size);
   }
}

static enum agx_object_type
translate_object_type(enum mesa_prim topology)
{
   static_assert(MESA_PRIM_LINES < MESA_PRIM_LINE_STRIP);
   static_assert(MESA_PRIM_TRIANGLES >= MESA_PRIM_LINE_STRIP);

   if (topology == MESA_PRIM_POINTS)
      return AGX_OBJECT_TYPE_POINT_SPRITE_UV01;
   else if (topology <= MESA_PRIM_LINE_STRIP)
      return AGX_OBJECT_TYPE_LINE;
   else
      return AGX_OBJECT_TYPE_TRIANGLE;
}

static enum agx_primitive
translate_hw_primitive_topology(enum mesa_prim prim)
{
   switch (prim) {
   case MESA_PRIM_POINTS:
      return AGX_PRIMITIVE_POINTS;
   case MESA_PRIM_LINES:
      return AGX_PRIMITIVE_LINES;
   case MESA_PRIM_LINE_STRIP:
      return AGX_PRIMITIVE_LINE_STRIP;
   case MESA_PRIM_TRIANGLES:
      return AGX_PRIMITIVE_TRIANGLES;
   case MESA_PRIM_TRIANGLE_STRIP:
      return AGX_PRIMITIVE_TRIANGLE_STRIP;
   case MESA_PRIM_TRIANGLE_FAN:
      return AGX_PRIMITIVE_TRIANGLE_FAN;
   default:
      UNREACHABLE("Invalid hardware primitive topology");
   }
}

static inline enum agx_vdm_vertex
translate_vdm_vertex(unsigned vtx)
{
   static_assert(AGX_VDM_VERTEX_0 == 0);
   static_assert(AGX_VDM_VERTEX_1 == 1);
   static_assert(AGX_VDM_VERTEX_2 == 2);

   assert(vtx <= 2);
   return vtx;
}

static inline enum agx_ppp_vertex
translate_ppp_vertex(unsigned vtx)
{
   static_assert(AGX_PPP_VERTEX_0 == 0 + 1);
   static_assert(AGX_PPP_VERTEX_1 == 1 + 1);
   static_assert(AGX_PPP_VERTEX_2 == 2 + 1);

   assert(vtx <= 2);
   return vtx + 1;
}

static void
hk_flush_index(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   struct hk_api_shader *gs = cmd->state.gfx.shaders[MESA_SHADER_GEOMETRY];
   uint32_t index = cmd->state.gfx.index.restart;

   if (gs) {
      enum poly_gs_shape shape =
         gs->variants[HK_GS_VARIANT_COUNT].info.gs.shape;
      index = BITFIELD_MASK(8 * poly_gs_index_size(shape));
   }

   /* VDM State updates are relatively expensive, so only emit them when the
    * restart index changes. This is simpler than accurate dirty tracking.
    */
   if (cs->restart_index != index) {
      uint8_t *out = cs->current;
      agx_push(out, VDM_STATE, cfg) {
         cfg.restart_index_present = true;
      }

      agx_push(out, VDM_STATE_RESTART_INDEX, cfg) {
         cfg.value = index;
      }

      cs->current = out;
      cs->restart_index = index;
   }
}

static float hk_depth_bias_factor(VkFormat format, bool exact,
                                  bool force_unorm);

static bool
hk_g17p_emit_native_bind_tail(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                              uint8_t **out)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   const struct hk_rendering_state *render = &gfx->render;

   if (dev->dev.chip != AGX_CHIP_G17P || !out || !*out)
      return false;

   struct hk_shader *fs = hk_only_variant(gfx->shaders[MESA_SHADER_FRAGMENT]);
   struct hk_linked_shader *linked_fs = gfx->linked[MESA_SHADER_FRAGMENT];
   if (!fs || !linked_fs)
      return false;

   const uint64_t fragment_program = hk_upload_usc_words(cmd, fs, linked_fs);
   if (!fragment_program || (fragment_program & UINT64_C(0x3f)) != 0 ||
       fragment_program >= (UINT64_C(1) << 48) ||
       !hk_g17p_smooth_varyings_supported(cmd, linked_fs) ||
       (linked_fs->b.cf.nr_bindings != 0) != (gfx->varyings != 0))
      return false;

   struct agx_ppp_hal200_graphics_core fs_record;
   const struct agx_ppp_hal200_fragment_fields fs_fields = {
      /* HAL200 finalize 0x3f04e4..0x3f0508 puts iteration count at bit 16.
       * The packer merges the complete VIT address, including its high bits. */
      .resource_counts = linked_fs->b.cf.nr_bindings << 16,
      .varying_table = gfx->varyings,
      .unknown_0c = 0,
      .stage_program = fragment_program,
      .stage_program_template = 0,
      .flags = 0,
   };

   const struct hk_api_shader *api_vs = gfx->shaders[MESA_SHADER_VERTEX];
   const struct hk_shader *vs =
      api_vs ? &api_vs->variants[HK_VS_VARIANT_HW] : NULL;
   uint32_t output_size;
   if (!vs || !agx_ppp_hal200_output_size_pack(
                 vs->info.uvs.size, 0, &output_size))
      return false;

   struct agx_ppp_hal200_graphics_core output_record;
   struct agx_ppp_hal200_graphics_fields output_fields = {
      /* Native vertex update 0x481be8..0x481c2c writes zero at
       * context+0x1f90 and viewport_count-1 at +0x1f94; both are zero
       * for this single-viewport layout. VIT supplies interpolation.
       * FragmentProgramVariant 0x373eb4..0x373ee0 sets output-select bit16
       * for nonempty iteration metadata, merged by 0x4824fc..0x482518. */
      .words = {(linked_fs->b.cf.nr_bindings ? UINT32_C(0x10000) : 0) |
                   (linked_fs->b.cf.reads_z ? UINT32_C(0x200000) : 0),
                0, 0, output_size,
                AGX_PPP_HAL200_PRIMARY_FACE_HEADER, 0},
   };

   struct agx_ppp_hal200_graphics_core primary_record;
   struct agx_ppp_hal200_graphics_fields primary_fields = {
      .words = {0, UINT32_C(0x07200f00),
                UINT32_C(0x0e000000), UINT32_C(0x07200f00),
                UINT32_C(0x0e000000), 0},
   };

   /* Each draw already owns its complete PRIMARY_FACE record. Native depth
    * control only changes these two words; retain the allocation and staging
    * layout and do not add launches, cache flushes or device-global state. */
   const bool depth_test = render->depth_att.iview && dyn->ds.depth.test_enable;
   if (dyn->ds.depth.bounds_test.enable ||
       !agx_ppp_hal200_depth_face_pack(
          dev->dev.chip, depth_test, dyn->ds.depth.write_enable,
          dyn->ds.depth.compare_op, primary_fields.words[1],
          &primary_fields.words[1]) ||
       !agx_ppp_hal200_depth_face_pack(
          dev->dev.chip, depth_test, dyn->ds.depth.write_enable,
          dyn->ds.depth.compare_op, primary_fields.words[3],
          &primary_fields.words[3]))
      return false;

   const bool stencil_test = hk_stencil_test_enabled(cmd);
   const struct vk_stencil_test_face_state stencil_faces[2] = {
      dyn->ds.stencil.front, dyn->ds.stencil.back,
   };
   for (unsigned i = 0; i < 2; ++i) {
      const struct vk_stencil_test_face_state *face = &stencil_faces[i];
      const unsigned face_word = 1 + 2 * i;
      /* Native458734/458740 updates only the low reference byte. */
      primary_fields.words[face_word] =
         (primary_fields.words[face_word] & ~0xffu) | (face->reference & 0xffu);
      primary_fields.words[face_word + 1] = agx_ppp_hal200_stencil_pack(
         stencil_test, face->op.compare, face->op.fail, face->op.depth_fail,
         face->op.pass, face->compare_mask, face->write_mask);
   }

   struct agx_ppp_hal200_graphics_core secondary_record;
   struct agx_ppp_hal200_graphics_fields secondary_fields = {
      .words = {UINT32_C(0x00000200), UINT32_C(0x07e00000),
                UINT32_C(0x07e00000), UINT32_C(0x0001ffff)},
   };

   struct agx_ppp_hal200_graphics_core cull_record;
   struct agx_ppp_hal200_graphics_fields cull_fields = {
      .words = {UINT32_C(0x00000480)},
   };
   if (!agx_ppp_hal200_cull_mode_pack(
          dev->dev.chip, dyn->rs.cull_mode,
          dyn->rs.front_face == VK_FRONT_FACE_COUNTER_CLOCKWISE,
          cull_fields.words[0], &cull_fields.words[0]))
      return false;
   if (!agx_ppp_hal200_cull_provoking_pack(dev->dev.chip, gfx->provoking,
                                         cull_fields.words[0],
                                         &cull_fields.words[0]))
      return false;
   if (!agx_ppp_hal200_depth_clip_pack(
          dev->dev.chip,
          vk_rasterization_state_depth_clip_enable(&dyn->rs),
          cull_fields.words[0], &cull_fields.words[0]))
      return false;
   if (!agx_ppp_hal200_rasterizer_discard_pack(
          dev->dev.chip, hk_rast_discard(cmd), secondary_fields.words[0],
          cull_fields.words[0], &secondary_fields.words[0],
          &cull_fields.words[0]))
      return false;

   struct agx_ppp_w_clamp_cull2_core w_cull2_record;
   const struct agx_ppp_w_clamp_cull2_fields w_cull2_fields = {
      .w_clamp_raw = 0,
      .cull2_raw = UINT32_C(0x80),
   };

   struct agx_ppp_viewport_payload_fields viewport_fields = {
      .viewport_count = 1,
   };
   const VkViewport *vp = &dyn->vp.viewports[0];
   unsigned minx = MIN2(vp->x, vp->x + vp->width);
   unsigned miny = MIN2(vp->y, vp->y + vp->height);
   unsigned maxx = MAX2(vp->x, vp->x + vp->width);
   unsigned maxy = MAX2(vp->y, vp->y + vp->height);
   minx = MIN2(minx, render->cr.width);
   miny = MIN2(miny, render->cr.height);
   maxx = MIN2(maxx, render->cr.width);
   maxy = MIN2(maxy, render->cr.height);
   if (dyn->vp.scissor_count > 0) {
      const VkRect2D *s = &dyn->vp.scissors[0];
      minx = MAX2(minx, s->offset.x);
      miny = MAX2(miny, s->offset.y);
      maxx = MIN2(maxx, s->offset.x + s->extent.width);
      maxy = MIN2(maxy, s->offset.y + s->extent.height);
   }

   const struct hk_image_view *color = render->color_att_count == 1
                                         ? render->color_att[0].iview : NULL;
   const VkRect2D api_scissor = dyn->vp.scissor_count
                                  ? dyn->vp.scissors[0] : (VkRect2D){0};
   const struct hk_g17p_full_frame_scissor full_frame = {
      .width = render->cr.width,
      .height = render->cr.height,
      .area_x = render->area.offset.x,
      .area_y = render->area.offset.y,
      .area_width = render->area.extent.width,
      .area_height = render->area.extent.height,
      .scissor_x = api_scissor.offset.x,
      .scissor_y = api_scissor.offset.y,
      .scissor_width = api_scissor.extent.width,
      .scissor_height = api_scissor.extent.height,
      .viewport_x = vp->x,
      .viewport_y = vp->y,
      .viewport_width = vp->width,
      .viewport_height = vp->height,
      .min_depth = vp->minDepth,
      .max_depth = vp->maxDepth,
      .viewport_count = dyn->vp.viewport_count,
      .scissor_count = dyn->vp.scissor_count,
      .single_color_layer_sample =
         color && !render->depth_att.iview && !render->stencil_att.iview &&
         render->layer_count == 1 && !render->view_mask &&
         render->tilebuffer.nr_samples == 1 &&
         u_minify(color->vk.image->extent.width, color->vk.base_mip_level) ==
            render->cr.width &&
         u_minify(color->vk.image->extent.height, color->vk.base_mip_level) ==
            render->cr.height,
      .default_depth =
         vk_rasterization_state_depth_clip_enable(&dyn->rs) &&
         !dyn->rs.depth_clamp_enable && !dyn->vp.depth_clip_negative_one_to_one &&
         !dyn->rs.depth_bias.enable && !hk_rast_discard(cmd),
   };
   const bool scissor_enable = dyn->vp.viewport_count > 0 &&
                              !hk_g17p_full_frame_scissor_can_elide(&full_frame);
   const bool depth_bias_enable =
      dyn->rs.depth_bias.enable &&
      gfx->object_type == AGX_OBJECT_TYPE_TRIANGLE;
   const bool exact = dyn->rs.depth_bias.exact;
   const bool force_unorm =
      dyn->rs.depth_bias.representation ==
      VK_DEPTH_BIAS_REPRESENTATION_LEAST_REPRESENTABLE_VALUE_FORCE_UNORM_EXT;
   const float min_z = dyn->rs.depth_clamp_enable
                          ? MIN2(vp->minDepth, vp->maxDepth)
                          : 0.0f;
   const float max_z = dyn->rs.depth_clamp_enable
                          ? MAX2(vp->minDepth, vp->maxDepth)
                          : 1.0f;
   const float depth_bias =
      depth_bias_enable
         ? dyn->rs.depth_bias.constant_factor /
              hk_depth_bias_factor(render->depth_att.vk_format, exact,
                                   force_unorm)
         : 0.0f;
   const struct agx_ppp_hal200_raster_fields raster_fields = {
      .scissor_enable = scissor_enable,
      .depth_bias_enable = depth_bias_enable,
      .max_x = MIN2(maxx, UINT16_MAX),
      .min_x = MIN2(minx, UINT16_MAX),
      .max_y = MIN2(maxy, UINT16_MAX),
      .min_y = MIN2(miny, UINT16_MAX),
      .min_z_raw = fui(min_z),
      .max_z_raw = fui(max_z),
      .depth_bias_raw = fui(depth_bias),
      .slope_scale_raw =
         depth_bias_enable ? fui(dyn->rs.depth_bias.slope_factor) : 0,
      .clamp_raw = depth_bias_enable ? fui(dyn->rs.depth_bias.clamp) : 0,
   };
   struct agx_ppp_hal200_raster_state raster_state;
   if (!agx_ppp_hal200_raster_state_build(dev->dev.chip, &raster_fields,
                                           &raster_state))
      return false;

   /* Native update 0x46c6ac..0x46c6dc stores pass type at context+0x1f78.
    * The 0x46ff78 path emits it in SECONDARY header 0x02000048's first
    * payload word, masked by 0xe6600000 and ORed with 0x200. It is not the
    * older-generation FRAGMENT_CONTROL location in our PRIMARY record.
    */
   const uint32_t pass_type = linked_fs->b.fragment_control.opaque[0] >> 29;
   if (pass_type != AGX_PASS_TYPE_OPAQUE &&
       pass_type != AGX_PASS_TYPE_TRANSLUCENT &&
       pass_type != AGX_PASS_TYPE_PUNCH_THROUGH)
      return false;
   secondary_fields.words[0] |= pass_type << 29;
   /* Native46c674..46c694 forces bit26 for pass type2, before merging it
    * into the same SECONDARY payload. This prevents triangle merging from
    * treating per-fragment coverage rejection as an opaque pass. Type3's
    * additional draw-buffer-state predicate is a separate native path. */
   if (pass_type == AGX_PASS_TYPE_PUNCH_THROUGH)
      secondary_fields.words[0] |= UINT32_C(1) << 26;
   /* Native458934..45896c retains both stencil-control bits in the
    * same per-draw record as depth. No extra state allocation or launch. */
   if (gfx->occlusion.mode != AGX_VISIBILITY_MODE_NONE &&
       gfx->occlusion.mode != AGX_VISIBILITY_MODE_COUNTING &&
       gfx->occlusion.mode != AGX_VISIBILITY_MODE_BOOLEAN)
      return false;
   if (gfx->occlusion.index >= AGX_MAX_OCCLUSION_QUERIES)
      return false;

   /* HAL200 retained the legacy semantic bit positions inside its fixed
    * PRIMARY_FACE record: visibility mode is fragment-control[15:14], while
    * the final word selects a 64-bit counter from ISP_OCLQRY_BASE.  The native
    * bind path replaces the legacy PPP update wholesale, so both fields must
    * be folded into this per-draw record here. */
   const uint32_t fragment_control = raster_state.fragment_control |
      (stencil_test ? UINT32_C(0x000c0000) : 0) |
      ((uint32_t)gfx->occlusion.mode << 14);
   output_fields.words[5] = fragment_control;
   primary_fields.words[0] = fragment_control;
   primary_fields.words[5] = gfx->occlusion.index << 17;
   if (!agx_ppp_hal200_fragment_core_pack(dev->dev.chip, &fs_fields,
                                          &fs_record) ||
       !agx_ppp_hal200_graphics_core_pack(
          dev->dev.chip, AGX_PPP_HAL200_GRAPHICS_OUTPUT, &output_fields,
          &output_record) ||
       !agx_ppp_hal200_graphics_core_pack(
          dev->dev.chip, AGX_PPP_HAL200_GRAPHICS_PRIMARY_FACE,
          &primary_fields, &primary_record) ||
       !agx_ppp_hal200_graphics_core_pack(
          dev->dev.chip, AGX_PPP_HAL200_GRAPHICS_SECONDARY_FACE,
          &secondary_fields, &secondary_record) ||
       !agx_ppp_hal200_graphics_core_pack(
          dev->dev.chip, AGX_PPP_HAL200_GRAPHICS_CULL, &cull_fields,
          &cull_record) ||
       !agx_ppp_w_clamp_cull2_core_pack(dev->dev.chip, &w_cull2_fields,
                                         &w_cull2_record))
      return false;

   void *scissor = util_dynarray_grow_bytes(
      &cs->scissor, 1, sizeof(raster_state.scissor));
   void *depth_bias_record = util_dynarray_grow_bytes(
      &cs->depth_bias, 1, sizeof(raster_state.depth_bias));
   memcpy(scissor, raster_state.scissor, sizeof(raster_state.scissor));
   memcpy(depth_bias_record, raster_state.depth_bias,
          sizeof(raster_state.depth_bias));

   /* Preserve the physical +0x00..+0x73 layout from Metal.  The complete
    * 28-byte FS record occupies +0x00..+0x1b; in particular, +0x14 is the
    * low stage-program word and must not be overlaid by the old anonymous
    * metadata guess.  The output window at +0x1c intentionally overlaps the
    * primary header/control at +0x30/+0x34.
    */
   static_assert(AGX_PPP_HAL200_FS_LENGTH == 0x1c,
                 "HAL200 FS must end where the output record begins");
   uint8_t bind_group[0x80] = {0};
   memcpy(&bind_group[0x00], fs_record.bytes, fs_record.length);
   memcpy(&bind_group[0x1c], output_record.bytes, output_record.length);
   memcpy(&bind_group[0x30], primary_record.bytes, primary_record.length);
   memcpy(&bind_group[0x4c], secondary_record.bytes, secondary_record.length);
   memcpy(&bind_group[0x60], w_cull2_record.bytes, w_cull2_record.length);
   memcpy(&bind_group[0x6c], cull_record.bytes, cull_record.length);

   uint8_t region_clip[AGX_REGION_CLIP_LENGTH];
   agx_pack(region_clip, REGION_CLIP, cfg) {
      cfg.enable = true;
      cfg.min_x = minx / 32;
      cfg.min_y = miny / 32;
      cfg.max_x = DIV_ROUND_UP(MAX2(maxx, 1), 32);
      cfg.max_y = DIV_ROUND_UP(MAX2(maxy, 1), 32);
   }
   memcpy(&viewport_fields.region_clip_raw[0], region_clip,
          sizeof(region_clip));
   viewport_fields.viewports[0] =
      (struct agx_ppp_viewport_transform_raw){
         .translate_x = fui(vp->x + 0.5f * vp->width),
         .scale_x = fui(vp->width * 0.5f),
         .translate_y = fui(vp->y + 0.5f * vp->height),
         .scale_y = fui(vp->height * 0.5f),
         .translate_z = fui(vp->minDepth),
         .scale_z = fui(vp->maxDepth - vp->minDepth),
      };

   struct agx_ppp_viewport_payload_core viewport_record;
   if (!agx_ppp_viewport_payload_core_pack(dev->dev.chip, &viewport_fields,
                                            &viewport_record))
      return false;

   /* A VDM fetch happens after recording. Never point earlier draws at a
    * device-global record that another draw/CB can overwrite. This upload
    * and its architectural low alias live until command-buffer reset. */
   struct agx_ptr state =
      hk_pool_vdm_alloc(cmd, HK_G17P_BIND_STATE_BYTES, 64);
   struct hk_g17p_bind_state_upload uploaded;
   if (!state.cpu ||
       !hk_g17p_upload_bind_state(state.cpu, HK_G17P_BIND_STATE_BYTES, state.gpu,
                                  bind_group, viewport_record.bytes,
                                  viewport_record.length,
                                  gfx->linked_varyings.counts_32.opaque[0],
                                  gfx->linked_varyings.counts_16.opaque[0],
                                  &uploaded))
      return false;

   if (hk_g17p_launch_trace_enabled()) {
      hk_g17p_trace_line("hal200-bind-group", state.gpu, state.cpu,
                         sizeof(bind_group));
      hk_g17p_trace_line("hal200-fragment-bind", fragment_program,
                         fs_record.bytes, fs_record.length);
      hk_g17p_trace_line("hal200-viewport", state.gpu + HK_G17P_BIND_GROUP_BYTES,
                         (uint8_t *)state.cpu + HK_G17P_BIND_GROUP_BYTES,
                         viewport_record.length);
   }

   memcpy(*out, uploaded.tail.bytes, uploaded.tail.length);
   *out += uploaded.tail.length;
   if (hk_g17p_launch_trace_enabled()) {
      const struct agx_vdm_hal200_bind_pair *pairs = uploaded.pairs;
      fprintf(stderr,
              "G17P HAL200 bind tail: bind0=%08x fs=%08x output=%08x "
              "primary=%08x secondary=%08x viewport=%08x wcull2=%08x "
              "cull=%08x\n",
              pairs[0].context_offset, pairs[1].context_offset,
              pairs[2].context_offset, pairs[3].context_offset,
              pairs[4].context_offset, pairs[5].context_offset,
              pairs[6].context_offset, pairs[7].context_offset);
   }

   (void)cs;
   return true;
}

/*
 * Return the given sample positions, packed into a 32-bit word with fixed
 * point nibbles for each x/y component of the (at most 4) samples. This is
 * suitable for programming the PPP_MULTISAMPLECTL control register.
 */
static uint32_t
hk_pack_ppp_multisamplectrl(const struct vk_sample_locations_state *sl)
{
   uint32_t ctrl = 0;

   for (int32_t i = sl->per_pixel - 1; i >= 0; i--) {
      VkSampleLocationEXT loc = sl->locations[i];

      uint32_t x = CLAMP(loc.x, 0.0f, 0.9375f) * 16.0;
      uint32_t y = CLAMP(loc.y, 0.0f, 0.9375f) * 16.0;

      assert(x <= 15);
      assert(y <= 15);

      /* Push bytes in reverse order so we can use constant shifts. */
      ctrl = (ctrl << 8) | (y << 4) | x;
   }

   return ctrl;
}

/*
 * Return the standard sample positions, prepacked as above for efficiency.
 */
uint32_t
hk_default_sample_positions(unsigned nr_samples)
{
   switch (nr_samples) {
   case 0:
   case 1:
      return 0x88;
   case 2:
      return 0x44cc;
   case 4:
      return 0xeaa26e26;
   default:
      UNREACHABLE("Invalid sample count");
   }
}

static void
hk_flush_ppp_state(struct hk_cmd_buffer *cmd, struct hk_cs *cs, uint8_t **out)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   const struct hk_rendering_state *render = &cmd->state.gfx.render;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_shader *hw_vs = hk_bound_hw_vs(gfx);
   struct hk_shader *fs = hk_only_variant(gfx->shaders[MESA_SHADER_FRAGMENT]);

   bool hw_vs_dirty = IS_SHADER_DIRTY(VERTEX) || IS_SHADER_DIRTY(TESS_EVAL) ||
                      IS_SHADER_DIRTY(GEOMETRY);
   bool fs_dirty = IS_SHADER_DIRTY(FRAGMENT);

   struct hk_linked_shader *linked_fs = gfx->linked[MESA_SHADER_FRAGMENT];
   bool linked_fs_dirty = IS_LINKED_DIRTY(FRAGMENT);

   bool varyings_dirty = gfx->dirty & HK_DIRTY_VARYINGS;

   bool face_dirty =
      IS_DIRTY(DS_DEPTH_TEST_ENABLE) || IS_DIRTY(DS_DEPTH_WRITE_ENABLE) ||
      IS_DIRTY(DS_DEPTH_COMPARE_OP) || IS_DIRTY(DS_STENCIL_REFERENCE) ||
      IS_DIRTY(RS_LINE_WIDTH) || IS_DIRTY(RS_POLYGON_MODE) || fs_dirty;

   bool stencil_face_dirty =
      IS_DIRTY(DS_STENCIL_OP) || IS_DIRTY(DS_STENCIL_COMPARE_MASK) ||
      IS_DIRTY(DS_STENCIL_WRITE_MASK) || IS_DIRTY(DS_STENCIL_TEST_ENABLE);

   struct AGX_PPP_HEADER dirty = {
      .fragment_control =
         IS_DIRTY(DS_STENCIL_TEST_ENABLE) || IS_DIRTY(IA_PRIMITIVE_TOPOLOGY) ||
         IS_DIRTY(RS_DEPTH_BIAS_ENABLE) || gfx->dirty & HK_DIRTY_OCCLUSION,

      .fragment_control_2 =
         IS_DIRTY(RS_RASTERIZER_DISCARD_ENABLE) || linked_fs_dirty,

      .fragment_front_face = face_dirty,
      .fragment_front_face_2 = fs_dirty || IS_DIRTY(IA_PRIMITIVE_TOPOLOGY),
      .fragment_front_stencil = stencil_face_dirty,
      .fragment_back_face = face_dirty,
      .fragment_back_face_2 = fs_dirty || IS_DIRTY(IA_PRIMITIVE_TOPOLOGY),
      .fragment_back_stencil = stencil_face_dirty,
      .output_select = hw_vs_dirty || linked_fs_dirty || varyings_dirty,
      .varying_counts_32 = varyings_dirty,
      .varying_counts_16 = varyings_dirty,
      .cull = IS_DIRTY(RS_CULL_MODE) ||
              IS_DIRTY(RS_RASTERIZER_DISCARD_ENABLE) ||
              IS_DIRTY(RS_FRONT_FACE) || IS_DIRTY(RS_DEPTH_CLIP_ENABLE) ||
              IS_DIRTY(RS_DEPTH_CLAMP_ENABLE) || IS_DIRTY(RS_LINE_MODE) ||
              IS_DIRTY(IA_PRIMITIVE_TOPOLOGY) ||
              (gfx->dirty & HK_DIRTY_PROVOKING) || IS_SHADER_DIRTY(TESS_CTRL) ||
              IS_SHADER_DIRTY(TESS_EVAL) || IS_DIRTY(TS_DOMAIN_ORIGIN),
      .cull_2 = varyings_dirty,

      .fragment_shader = fs_dirty || linked_fs_dirty || varyings_dirty ||
                         gfx->descriptors.root_dirty,

      .occlusion_query = gfx->dirty & HK_DIRTY_OCCLUSION,
      .output_size = hw_vs_dirty,
      .viewport_count = 1, /* irrelevant */
   };

   if (dev->dev.chip != AGX_CHIP_G17P)
      dirty.fragment_shader &= !linked_fs->b.no_op;

   /* The first G17P hardware fault is introduced by this aggregate dirty PPP
    * update. Keep a diagnostic-only selector that emits one independently
    * consumable legacy group at a time so the bad HAL200 block can be
    * identified without changing any of its dynamic values.
    *
    * 1: fragment control and non-stencil faces
    * 2: cull words
    * 3: fragment shader words
    * 4: output selection, varying counts, and output size
    * 5: occlusion query and output size
    */
   if (dev->dev.chip == AGX_CHIP_G17P) {
      static int g17p_dirty_bisect = -1;
      if (g17p_dirty_bisect < 0)
         g17p_dirty_bisect =
            debug_get_num_option("HK_G17P_DIRTY_BISECT", 0);

      if (g17p_dirty_bisect > 0) {
         const struct AGX_PPP_HEADER all = dirty;
         dirty = (struct AGX_PPP_HEADER){.viewport_count = 1};

         switch (g17p_dirty_bisect) {
         case 1:
            dirty.fragment_control = all.fragment_control;
            dirty.fragment_control_2 = all.fragment_control_2;
            dirty.fragment_front_face = all.fragment_front_face;
            dirty.fragment_front_face_2 = all.fragment_front_face_2;
            dirty.fragment_back_face = all.fragment_back_face;
            dirty.fragment_back_face_2 = all.fragment_back_face_2;
            break;
         case 2:
            dirty.cull = all.cull;
            dirty.cull_2 = all.cull_2;
            break;
         case 3:
            dirty.fragment_shader = all.fragment_shader;
            break;
         case 4:
            dirty.output_select = all.output_select;
            dirty.varying_counts_32 = all.varying_counts_32;
            dirty.varying_counts_16 = all.varying_counts_16;
            dirty.output_size = all.output_size;
            break;
         case 5:
            dirty.occlusion_query = all.occlusion_query;
            dirty.output_size = all.output_size;
            break;
         default:
            mesa_loge("HK_G17P_DIRTY_BISECT=%d is outside 1..5",
                      g17p_dirty_bisect);
            dirty = all;
            break;
         }

         fprintf(stderr, "G17P dirty PPP bisect: group=%d\n",
                 g17p_dirty_bisect);
      }
   }

   /* Calculate the update size. If it equals the header, there is nothing to
    * update so early-exit.
    */
   size_t size = agx_ppp_update_size(&dirty);
   if (size == AGX_PPP_HEADER_LENGTH)
      return;

   /* Otherwise, allocate enough space for the update and push it. */
   assert(size > AGX_PPP_HEADER_LENGTH);

   struct agx_ptr T = hk_pool_vdm_alloc(cmd, size, AGX_PPP_HEADER_ALIGN);
   if (!T.cpu)
      return;

   struct agx_ppp_update ppp = agx_new_ppp_update(T, size, &dirty);

   if (dirty.fragment_control) {
      agx_ppp_push(&ppp, FRAGMENT_CONTROL, cfg) {
         cfg.visibility_mode = gfx->occlusion.mode;
         cfg.stencil_test_enable = hk_stencil_test_enabled(cmd);

         /* TODO: Consider optimizing this? */
         cfg.two_sided_stencil = cfg.stencil_test_enable;

         cfg.depth_bias_enable = dyn->rs.depth_bias.enable &&
                                 gfx->object_type == AGX_OBJECT_TYPE_TRIANGLE;

         /* Always enable scissoring so we may scissor to the viewport (TODO:
          * optimize this out if the viewport is the default and the app does
          * not use the scissor test)
          */
         cfg.scissor_enable = true;

         /* This avoids broken derivatives along primitive edges */
         cfg.disable_tri_merging = gfx->object_type != AGX_OBJECT_TYPE_TRIANGLE;
      }
   }

   if (dirty.fragment_control_2) {
      /* Annoying, rasterizer_discard seems to be ignored (sometimes?) in the
       * main fragment control word and has to be combined into the secondary
       * word for reliable behaviour.
       */
      agx_ppp_push_merged(&ppp, FRAGMENT_CONTROL, cfg,
                          linked_fs->b.fragment_control) {

         cfg.tag_write_disable = hk_rast_discard(cmd);
      }
   }

   struct agx_fragment_face_packed fragment_face = {};
   struct agx_fragment_face_2_packed fragment_face_2 = {};

   if (dirty.fragment_front_face) {
      bool has_z = render->depth_att.vk_format != VK_FORMAT_UNDEFINED;
      bool z_test = has_z && dyn->ds.depth.test_enable;

      agx_pack(&fragment_face, FRAGMENT_FACE, cfg) {
         cfg.line_width = agx_pack_line_width(dyn->rs.line.width);
         cfg.polygon_mode = translate_polygon_mode(dyn->rs.polygon_mode);
         cfg.disable_depth_write = !(z_test && dyn->ds.depth.write_enable);

         if (z_test && !gfx->descriptors.root.draw.force_never_in_shader)
            cfg.depth_function = translate_compare_op(dyn->ds.depth.compare_op);
         else
            cfg.depth_function = AGX_ZS_FUNC_ALWAYS;
      };

      agx_ppp_push_merged(&ppp, FRAGMENT_FACE, cfg, fragment_face) {
         cfg.stencil_reference = dyn->ds.stencil.front.reference;
      }
   }

   if (dirty.fragment_front_face_2) {
      agx_pack(&fragment_face_2, FRAGMENT_FACE_2, cfg) {
         cfg.object_type = gfx->object_type;
      }

      agx_merge(fragment_face_2, fs->frag_face, FRAGMENT_FACE_2);
      agx_ppp_push_packed(&ppp, &fragment_face_2, FRAGMENT_FACE_2);
   }

   if (dirty.fragment_front_stencil) {
      hk_ppp_push_stencil_face(&ppp, dyn->ds.stencil.front,
                               hk_stencil_test_enabled(cmd));
   }

   if (dirty.fragment_back_face) {
      assert(dirty.fragment_front_face);

      agx_ppp_push_merged(&ppp, FRAGMENT_FACE, cfg, fragment_face) {
         cfg.stencil_reference = dyn->ds.stencil.back.reference;
      }
   }

   if (dirty.fragment_back_face_2) {
      assert(dirty.fragment_front_face_2);

      agx_ppp_push_packed(&ppp, &fragment_face_2, FRAGMENT_FACE_2);
   }

   if (dirty.fragment_back_stencil) {
      hk_ppp_push_stencil_face(&ppp, dyn->ds.stencil.back,
                               hk_stencil_test_enabled(cmd));
   }

   if (dirty.output_select) {
      struct agx_output_select_packed osel = hw_vs->info.uvs.osel;

      agx_ppp_push_merged_blobs(&ppp, AGX_OUTPUT_SELECT_LENGTH, &osel,
                                &linked_fs->b.osel);
   }

   assert(dirty.varying_counts_32 == dirty.varying_counts_16);

   if (dirty.varying_counts_32) {
      agx_ppp_push_packed(&ppp, &gfx->linked_varyings.counts_32,
                          VARYING_COUNTS);

      agx_ppp_push_packed(&ppp, &gfx->linked_varyings.counts_16,
                          VARYING_COUNTS);
   }

   if (dirty.cull) {
      agx_ppp_push(&ppp, CULL, cfg) {
         cfg.cull_front = dyn->rs.cull_mode & VK_CULL_MODE_FRONT_BIT;
         cfg.cull_back = dyn->rs.cull_mode & VK_CULL_MODE_BACK_BIT;
         cfg.front_face_ccw = dyn->rs.front_face != VK_FRONT_FACE_CLOCKWISE;

         if (gfx->shaders[MESA_SHADER_TESS_CTRL] &&
             !gfx->shaders[MESA_SHADER_GEOMETRY]) {
            cfg.front_face_ccw ^= gfx->tess.info.ccw;
            cfg.front_face_ccw ^= dyn->ts.domain_origin ==
                                  VK_TESSELLATION_DOMAIN_ORIGIN_LOWER_LEFT;
         }

         cfg.flat_shading_vertex = translate_ppp_vertex(gfx->provoking);
         cfg.rasterizer_discard = hk_rast_discard(cmd);

         /* We do not support unrestricted depth, so clamping is inverted from
          * clipping. This implementation seems to pass CTS without unrestricted
          * depth support.
          *
          * TODO: Make sure this is right with gl_FragDepth.
          */
         cfg.depth_clip = vk_rasterization_state_depth_clip_enable(&dyn->rs);
         cfg.depth_clamp = !cfg.depth_clip;

         cfg.primitive_msaa =
            gfx->object_type == AGX_OBJECT_TYPE_LINE &&
            dyn->rs.line.mode == VK_LINE_RASTERIZATION_MODE_BRESENHAM_KHR;
      }
   }

   if (dirty.cull_2) {
      agx_ppp_push(&ppp, CULL_2, cfg) {
         cfg.needs_primitive_id = gfx->generate_primitive_id;
         cfg.clamp_w = true;
      }
   }

   if (dirty.fragment_shader) {
      /* TODO: Do less often? */
      hk_reserve_scratch(cmd, cs, fs);

      const uint64_t uploaded = hk_upload_usc_words(cmd, fs, linked_fs);
      uint32_t fragment_pipeline;
      switch (agx_chip_command_stream_generation(dev->dev.chip)) {
      case AGX_CS_GENERATION_AGX2:
         if (uploaded > UINT32_MAX) {
            vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
            return;
         }
         fragment_pipeline = uploaded;
         break;
      case AGX_CS_GENERATION_HAL200:
         if (uploaded == 0 || uploaded < dev->dev.shader_base ||
             uploaded - dev->dev.shader_base > UINT32_MAX) {
            vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
            return;
         }
         fragment_pipeline = agx_usc_addr(&dev->dev, uploaded);
         break;
      case AGX_CS_GENERATION_G15:
      case AGX_CS_GENERATION_HAL300:
      case AGX_CS_GENERATION_UNKNOWN:
      default:
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }

      agx_ppp_push_packed(&ppp, &linked_fs->fs_counts, FRAGMENT_SHADER_WORD_0);

      agx_ppp_push(&ppp, FRAGMENT_SHADER_WORD_1, cfg) {
         cfg.pipeline = fragment_pipeline;
      }

      agx_ppp_push(&ppp, FRAGMENT_SHADER_WORD_2, cfg) {
         cfg.cf_bindings = gfx->varyings;
      }

      agx_ppp_push(&ppp, FRAGMENT_SHADER_WORD_3, cfg)
         ;
   }

   if (dirty.occlusion_query) {
      agx_ppp_push(&ppp, FRAGMENT_OCCLUSION_QUERY, cfg) {
         cfg.index = gfx->occlusion.index;
      }
   }

   if (dirty.output_size) {
      agx_ppp_push(&ppp, OUTPUT_SIZE, cfg) {
         cfg.count = hw_vs->info.uvs.size;
      }
   }

   const bool ppp_ok = agx_ppp_legacy_payload_supported(dev->dev.chip) &&
                       agx_ppp_fini(dev->dev.chip, out, &ppp);
   if (!ppp_ok) {
      mesa_loge("no grounded legacy PPP payload/finalizer for %s",
                agx_chip_name(dev->dev.chip));
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
   } else if (hk_g17p_launch_trace_enabled()) {
      hk_g17p_trace_line("ppp-dirty", ppp.gpu_base, ppp.cpu_base,
                         ppp.total_size);
   }
}

/*
 * Based somewhat on the calculation in the PowerVR driver, and mostly trial &
 * error to pass CTS. This is a mess.
 */
static float
hk_depth_bias_factor(VkFormat format, bool exact, bool force_unorm)
{
   if (format == VK_FORMAT_D16_UNORM) {
      return exact ? (1 << 16) : (1 << 15);
   } else if (force_unorm) {
      return exact ? (1ull << 24) : (1ull << 23);
   } else {
      return 1.0;
   }
}

static bool
uses_blend_constant(const struct vk_color_blend_state *cb)
{
   static_assert(VK_BLEND_FACTOR_CONSTANT_COLOR + 1 ==
                 VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR);

   static_assert(VK_BLEND_FACTOR_CONSTANT_COLOR + 2 ==
                 VK_BLEND_FACTOR_CONSTANT_ALPHA);

   static_assert(VK_BLEND_FACTOR_CONSTANT_COLOR + 3 ==
                 VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA);

   for (unsigned i = 0; i < cb->attachment_count; ++i) {
      unsigned factors[] = {
         cb->attachments[i].src_color_blend_factor,
         cb->attachments[i].src_alpha_blend_factor,
         cb->attachments[i].dst_color_blend_factor,
         cb->attachments[i].dst_alpha_blend_factor,
      };

      for (unsigned j = 0; j < ARRAY_SIZE(factors); ++j) {
         if (factors[j] >= VK_BLEND_FACTOR_CONSTANT_COLOR &&
             factors[j] <= VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA)
            return true;
      }
   }

   return false;
}

void
agx_fill_velem_keys(const struct vk_vertex_input_state *vi,
                    uint64_t attribs_read, struct agx_velem_key *keys)
{
   u_foreach_bit(a, vi->attributes_valid) {
      struct vk_vertex_attribute_state attr = vi->attributes[a];

      assert(vi->bindings_valid & BITFIELD_BIT(attr.binding));
      struct vk_vertex_binding_state binding = vi->bindings[attr.binding];

      /* nir_assign_io_var_locations compacts vertex inputs, eliminating
       * unused inputs. We need to do the same here to match the locations.
       */
      unsigned slot = util_bitcount64(attribs_read & BITFIELD_MASK(a));

      keys[slot] = (struct agx_velem_key){
         .format = hk_format_to_pipe_format(attr.format),
         .stride = binding.stride,
         .divisor = binding.divisor,
         .instanced = binding.input_rate == VK_VERTEX_INPUT_RATE_INSTANCE,
      };
   }
}

static void
hk_flush_dynamic_state(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                       uint32_t draw_id, struct agx_draw draw,
                       bool ordinary_draw_path,
                       bool *g17p_native_bind_tail_emitted_out)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   const struct hk_rendering_state *render = &cmd->state.gfx.render;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   struct hk_graphics_state *gfx = &cmd->state.gfx;

   struct hk_shader *hw_vs = hk_bound_hw_vs(gfx);
   struct hk_shader *sw_vs = hk_bound_sw_vs(gfx);
   bool g17p_native_bind_tail_emitted = false;
   *g17p_native_bind_tail_emitted_out = false;

   /* The grounded VertexProgramVariant producer covers only the four ordinary
    * Index List entry points.  Reject software-transformed draw paths even if
    * no dynamic state happens to be dirty, otherwise they could inherit a
    * preceding ordinary draw's admitted AGX3 state. */
   if (agx_chip_command_stream_generation(dev->dev.chip) !=
          AGX_CS_GENERATION_AGX2 &&
       !ordinary_draw_path) {
      mesa_loge("no grounded AGX3 VertexProgramVariant state for transformed "
                "draw path on %s",
                agx_chip_name(dev->dev.chip));
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }

   if (!vk_dynamic_graphics_state_any_dirty(dyn) && !gfx->dirty &&
       !gfx->descriptors.root_dirty && !gfx->shaders_dirty &&
       !sw_vs->b.info.uses_draw_id && !sw_vs->b.info.uses_base_param &&
       !(gfx->linked[MESA_SHADER_VERTEX] &&
         gfx->linked[MESA_SHADER_VERTEX]->b.uses_base_param))
      return;

   struct hk_descriptor_state *desc = &cmd->state.gfx.descriptors;

   assert(cs->current + 0x1000 < cs->end && "already ensured space");
   uint8_t *out = cs->current;

   struct hk_shader *fs = hk_only_variant(gfx->shaders[MESA_SHADER_FRAGMENT]);

   bool gt_dirty = IS_SHADER_DIRTY(TESS_CTRL) || IS_SHADER_DIRTY(TESS_EVAL) ||
                   IS_SHADER_DIRTY(GEOMETRY);
   bool vgt_dirty = IS_SHADER_DIRTY(VERTEX) || gt_dirty;
   bool fs_dirty = IS_SHADER_DIRTY(FRAGMENT);

   if (IS_DIRTY(CB_BLEND_CONSTANTS)) {
      static_assert(sizeof(desc->root.draw.blend_constant) ==
                       sizeof(dyn->cb.blend_constants),
                    "common size");

      memcpy(desc->root.draw.blend_constant, dyn->cb.blend_constants,
             sizeof(dyn->cb.blend_constants));
      desc->root_dirty = true;
   }

   if (IS_DIRTY(CB_BLEND_EQUATIONS) || IS_DIRTY(CB_BLEND_ENABLES)) {
      gfx->uses_blend_constant = uses_blend_constant(&dyn->cb);
   }

   if (IS_DIRTY(MS_SAMPLE_MASK)) {
      desc->root.draw.api_sample_mask = dyn->ms.sample_mask;
      desc->root_dirty = true;
   }

   if (IS_DIRTY(RS_RASTERIZATION_STREAM)) {
      desc->root.draw.rasterization_stream = dyn->rs.rasterization_stream;
      desc->root_dirty = true;

      BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_RS_RASTERIZER_DISCARD_ENABLE);
   }

   if (fs_dirty || IS_DIRTY(DS_DEPTH_TEST_ENABLE) ||
       IS_DIRTY(DS_DEPTH_COMPARE_OP)) {

      const struct hk_rendering_state *render = &cmd->state.gfx.render;
      bool has_z = render->depth_att.vk_format != VK_FORMAT_UNDEFINED;
      bool z_test = has_z && dyn->ds.depth.test_enable;

      desc->root.draw.force_never_in_shader =
         z_test && dyn->ds.depth.compare_op == VK_COMPARE_OP_NEVER && fs &&
         fs->info.fs.writes_memory;

      desc->root_dirty = true;
   }

   /* The main shader must not run tests if the epilog will. */
   bool nontrivial_force_early =
      fs && (fs->b.info.early_fragment_tests &&
             (fs->b.info.writes_sample_mask || fs->info.fs.writes_memory));

   bool epilog_discards = dyn->ms.alpha_to_coverage_enable ||
                          (fs && (fs->info.fs.epilog_key.write_z ||
                                  fs->info.fs.epilog_key.write_s));
   epilog_discards &= !nontrivial_force_early;

   if (fs_dirty || IS_DIRTY(MS_ALPHA_TO_COVERAGE_ENABLE)) {
      desc->root.draw.no_epilog_discard = !epilog_discards ? ~0 : 0;
      desc->root_dirty = true;
   }

   if (IS_DIRTY(VI) || IS_DIRTY(VI_BINDINGS_VALID) ||
       IS_DIRTY(VI_BINDING_STRIDES) || vgt_dirty || true /* TODO */) {

      struct hk_fast_link_key_vs key = {
         .prolog.hw = (sw_vs == hw_vs),

         /* FIXME: handle pipeline robustness "properly" */
         .prolog.robustness.level =
            (dev->vk.enabled_features.robustBufferAccess2 ||
             dev->vk.enabled_features.pipelineRobustness)
               ? AGX_ROBUSTNESS_D3D
            : dev->vk.enabled_features.robustBufferAccess
               ? AGX_ROBUSTNESS_GL
               : AGX_ROBUSTNESS_DISABLED,

         .prolog.robustness.soft_fault = agx_has_soft_fault(&dev->dev),
         .prolog.static_vi = !sw_vs->info.vs.use_prolog,
      };

      enum mesa_prim prim = vk_topology_to_mesa(dyn->ia.primitive_topology);

      if (mesa_prim_has_adjacency(prim)) {
         if (draw.restart) {
            prim = u_decomposed_prim(prim);
         }

         key.prolog.adjacency = prim;
      }

      if (key.prolog.adjacency || !key.prolog.hw) {
         key.prolog.sw_index_size_B =
            draw.indexed ? agx_index_size_to_B(draw.index_size) : 0;
      }

      static_assert(sizeof(key.prolog.component_mask) ==
                    sizeof(sw_vs->info.vs.attrib_components_read));
      BITSET_COPY(key.prolog.component_mask,
                  sw_vs->info.vs.attrib_components_read);

      if (sw_vs->info.vs.use_prolog) {
         agx_fill_velem_keys(dyn->vi, sw_vs->info.vs.attribs_read,
                             key.prolog.attribs);

         u_foreach_bit(a, dyn->vi->attributes_valid) {
            unsigned slot =
               util_bitcount64(sw_vs->info.vs.attribs_read & BITFIELD_MASK(a));

            key.prolog.attribs[slot].stride =
               dyn->vi_binding_strides[dyn->vi->attributes[a].binding];
         }
      }

      if (!hk_update_fast_linked(cmd, sw_vs, &key)) {
         cs->current = out;
         return;
      }
   }

   if (IS_DIRTY(VI) || IS_DIRTY(VI_BINDINGS_VALID) ||
       IS_DIRTY(VI_BINDING_STRIDES) || vgt_dirty ||
       (gfx->dirty & HK_DIRTY_VB)) {

      unsigned slot = 0;
      u_foreach_bit(a, sw_vs->info.vs.attribs_read) {
         if (dyn->vi->attributes_valid & BITFIELD_BIT(a)) {
            struct vk_vertex_attribute_state attr = dyn->vi->attributes[a];
            struct hk_addr_range vb = gfx->vb[attr.binding];
            enum pipe_format fmt = hk_format_to_pipe_format(attr.format);
            enum pipe_format interchange_format = agx_vbo_internal_format(fmt);
            unsigned interchange_align =
               util_format_get_blocksize(interchange_format);

            desc->root.draw.attrib_clamps[slot] = agx_calculate_vbo_clamp(
               vb.addr, fmt, vb.range, dyn->vi_binding_strides[attr.binding],
               attr.offset, &desc->root.draw.attrib_base[slot]);

            desc->root.draw.attrib_strides[slot] =
               dyn->vi_binding_strides[attr.binding] / interchange_align;
         } else {
            desc->root.draw.attrib_base[slot] = AGX_ZERO_PAGE_ADDRESS;
            desc->root.draw.attrib_clamps[slot] = 0;
            desc->root.draw.attrib_strides[slot] = 0;
         }

         ++slot;
      }

      desc->root_dirty = true;
   }

   if (vgt_dirty || IS_SHADER_DIRTY(FRAGMENT) ||
       IS_DIRTY(MS_RASTERIZATION_SAMPLES) || IS_DIRTY(MS_SAMPLE_MASK) ||
       IS_DIRTY(MS_ALPHA_TO_COVERAGE_ENABLE) ||
       IS_DIRTY(MS_ALPHA_TO_ONE_ENABLE) || IS_DIRTY(CB_LOGIC_OP) ||
       IS_DIRTY(CB_LOGIC_OP_ENABLE) || IS_DIRTY(CB_WRITE_MASKS) ||
       IS_DIRTY(CB_COLOR_WRITE_ENABLES) || IS_DIRTY(CB_ATTACHMENT_COUNT) ||
       IS_DIRTY(CB_BLEND_ENABLES) || IS_DIRTY(CB_BLEND_EQUATIONS) ||
       IS_DIRTY(CB_BLEND_CONSTANTS) || IS_DIRTY(CB_BLEND_ADVANCED) ||
       desc->root_dirty /* for pipeline stats */ || true) {

      unsigned tib_sample_mask = BITFIELD_MASK(dyn->ms.rasterization_samples);
      unsigned api_sample_mask = dyn->ms.sample_mask & tib_sample_mask;
      bool has_sample_mask = api_sample_mask != tib_sample_mask;

      if (hw_vs->info.cull_distance_array_size) {
         perf_debug(cmd, "Emulating cull distance (size %u)",
                    hw_vs->info.cull_distance_array_size);
      }

      if (has_sample_mask) {
         perf_debug(cmd, "Emulating sample mask");
      }

      unsigned samples_shaded = 0;
      if (fs->info.fs.epilog_key.sample_shading)
         samples_shaded = dyn->ms.rasterization_samples;

      struct hk_fast_link_key_fs key = {
         .prolog.statistics = hk_stat_enabled(hk_pipeline_stat_addr(
            cmd, VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT)),

         .prolog.cull_distance_size = hw_vs->info.cull_distance_array_size,
         .prolog.api_sample_mask = has_sample_mask ? api_sample_mask : 0xff,
         .nr_samples_shaded = samples_shaded,
      };

      bool prolog_discards = has_sample_mask || key.prolog.cull_distance_size;

      bool needs_prolog = key.prolog.statistics || prolog_discards;

      if (needs_prolog) {
         /* With late main shader tests, the prolog runs tests if neither the
          * main shader nor epilog will.
          *
          * With (nontrivial) early main shader tests, the prolog does not
          * run tests, the tests will run at the start of the main shader.
          * This ensures tests are after API sample mask and cull distance
          * discards.
          */
         key.prolog.run_zs_tests = !nontrivial_force_early &&
                                   !fs->b.info.writes_sample_mask &&
                                   !epilog_discards && prolog_discards;

         if (key.prolog.cull_distance_size) {
            key.prolog.cf_base = fs->b.info.varyings.fs.nr_cf;
         }
      }

      key.epilog = (struct agx_fs_epilog_key){
         .link = fs->info.fs.epilog_key,
         .nr_samples = MAX2(dyn->ms.rasterization_samples, 1),
         .blend.alpha_to_coverage = dyn->ms.alpha_to_coverage_enable,
         .blend.alpha_to_one = dyn->ms.alpha_to_one_enable,
         .blend.logicop_enable = dyn->cb.logic_op_enable,
         .blend.logicop_func = vk_logic_op_to_pipe(dyn->cb.logic_op),
         .preserve_zero_mask = dev->dev.chip == AGX_CHIP_G17P,
      };

      for (unsigned rt = 0; rt < ARRAY_SIZE(dyn->cal.color_map); ++rt) {
         int map = dyn->cal.color_map[rt];
         key.epilog.remap[rt] = map == MESA_VK_ATTACHMENT_UNUSED ? -1 : map;
      }

      if (dyn->ms.alpha_to_one_enable || dyn->ms.alpha_to_coverage_enable ||
          dyn->cb.logic_op_enable) {

         perf_debug(
            cmd, "Epilog with%s%s%s",
            dyn->ms.alpha_to_one_enable ? " alpha-to-one" : "",
            dyn->ms.alpha_to_coverage_enable ? " alpha-to-coverage" : "",
            dyn->cb.logic_op_enable ? " logic-op" : "");
      }

      key.epilog.link.already_ran_zs |= nontrivial_force_early;

      struct hk_rendering_state *render = &cmd->state.gfx.render;
      for (uint32_t i = 0; i < render->color_att_count; i++) {
         key.epilog.rt_formats[i] =
            hk_format_to_pipe_format(render->color_att[i].vk_format);

         const struct vk_color_blend_attachment_state *cb =
            &dyn->cb.attachments[i];

         bool write_enable = dyn->cb.color_write_enables & BITFIELD_BIT(i);
         unsigned write_mask = write_enable ? cb->write_mask : 0;

         /* A zero hardware mask deletes the tile store while raster coverage
          * still makes a HAL200 tile nonempty. Materialize the unchanged tile
          * value with the already-grounded blend load/store path instead.
          * nir_lower_blend always blends, so use a default blend state when
          * blending is disabled at an API level.
          */
         if (dev->dev.chip == AGX_CHIP_G17P && !write_enable) {
            key.epilog.blend.rt[i] = (struct agx_blend_rt_key){
               .colormask = BITFIELD_MASK(4),
               .mode = agx_pack_blend_standard(
                  PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ZERO, PIPE_BLENDFACTOR_ONE,
                  PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ZERO, PIPE_BLENDFACTOR_ONE),
            };
         } else if (!dyn->cb.attachments[i].blend_enable) {
            key.epilog.blend.rt[i] = (struct agx_blend_rt_key){
               .colormask = write_mask,
               .mode = agx_pack_blend_standard(
                  PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO,
                  PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO),
            };
         } else if (cb->color_blend_op >= VK_BLEND_OP_ZERO_EXT) {
            key.epilog.blend.rt[i] = (struct agx_blend_rt_key){
               .colormask = write_mask,
               .advanced_blend = 1,
               .mode = agx_pack_blend_advanced(
                  vk_advanced_blend_op_to_pipe(cb->color_blend_op),
                  vk_blend_overlap_to_pipe(cb->blend_overlap),
                  cb->src_premultiplied, cb->dst_premultiplied,
                  cb->clamp_results),
            };

            assert(cb->clamp_results == false);
         } else {
            key.epilog.blend.rt[i] = (struct agx_blend_rt_key){
               .colormask = write_mask,
               .mode = agx_pack_blend_standard(
                  vk_blend_op_to_pipe(cb->color_blend_op),
                  vk_blend_factor_to_pipe(cb->src_color_blend_factor),
                  vk_blend_factor_to_pipe(cb->dst_color_blend_factor),
                  vk_blend_op_to_pipe(cb->alpha_blend_op),
                  vk_blend_factor_to_pipe(cb->src_alpha_blend_factor),
                  vk_blend_factor_to_pipe(cb->dst_alpha_blend_factor)),
            };
         }
      }

      if (!hk_update_fast_linked(cmd, fs, &key)) {
         cs->current = out;
         return;
      }
   }

   /* If the vertex shader uses draw parameters, vertex uniforms are dirty every
    * draw. Fragment uniforms are unaffected.
    *
    * For a direct draw, we upload the draw parameters as-if indirect to
    * avoid keying to indirectness.
    */
   if (gfx->linked[MESA_SHADER_VERTEX]->b.uses_base_param) {
      if (agx_is_indirect(draw.b)) {
         gfx->draw_params = draw.b.ptr;

         if (draw.indexed) {
            gfx->draw_params +=
               offsetof(VkDrawIndexedIndirectCommand, vertexOffset);
         } else {
            gfx->draw_params += offsetof(VkDrawIndirectCommand, firstVertex);
         }
      } else {
         uint32_t params[] = {
            draw.indexed ? draw.index_bias : draw.start,
            draw.start_instance,
         };

         gfx->draw_params = hk_pool_upload(cmd, params, sizeof(params), 4);
      }
   } else {
      gfx->draw_params = 0;
   }

   if (sw_vs->b.info.uses_draw_id) {
      /* TODO: rodata? */
      gfx->draw_id_ptr = hk_pool_upload(cmd, &draw_id, 2, 4);
   } else {
      gfx->draw_id_ptr = 0;
   }

   if (IS_DIRTY(IA_PRIMITIVE_TOPOLOGY) || gt_dirty) {
      enum mesa_prim prim = hk_rast_prim(cmd);

      gfx->topology = translate_hw_primitive_topology(prim);
      gfx->object_type = translate_object_type(prim);
   }

   if (IS_DIRTY(IA_PRIMITIVE_TOPOLOGY) || IS_DIRTY(RS_PROVOKING_VERTEX)) {
      unsigned provoking;
      if (dyn->rs.provoking_vertex == VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT)
         provoking = 2;
      else if (gfx->topology == AGX_PRIMITIVE_TRIANGLE_FAN)
         provoking = 1;
      else
         provoking = 0;

      if (provoking != gfx->provoking) {
         gfx->provoking = provoking;
         gfx->dirty |= HK_DIRTY_PROVOKING;

         gfx->descriptors.root.draw.provoking = provoking;
         gfx->descriptors.root_dirty = true;
      }
   }

   /* With attachmentless rendering, we don't know the sample count until draw
    * time, so we do a late tilebuffer fix up. But with rasterizer discard,
    * rasterization_samples might be 0.
    *
    * Note that we ignore dyn->ms.rasterization_samples when we do have a sample
    * count from an attachment. In Vulkan, these have to match anyway, but DX12
    * drivers are robust against this scenarios and vkd3d-proton will go out of
    * spec here. No reason we can't be robust here too.
    */
   if (dyn->ms.rasterization_samples && !gfx->render.tilebuffer.nr_samples) {
      agx_tilebuffer_set_samples(&gfx->render.tilebuffer,
                                 dyn->ms.rasterization_samples);

      cs->tib = gfx->render.tilebuffer;
   }

   if (IS_DIRTY(MS_SAMPLE_LOCATIONS) || IS_DIRTY(MS_SAMPLE_LOCATIONS_ENABLE) ||
       IS_DIRTY(MS_RASTERIZATION_SAMPLES)) {

      uint32_t ctrl;
      if (dyn->ms.sample_locations_enable) {
         ctrl = hk_pack_ppp_multisamplectrl(dyn->ms.sample_locations);
      } else {
         ctrl = hk_default_sample_positions(dyn->ms.rasterization_samples);
      }

      bool dont_commit = cmd->in_meta || dyn->ms.rasterization_samples == 0;

      if (!cs->has_sample_locations) {
         cs->ppp_multisamplectl = ctrl;

         /* If we're in vk_meta, do not commit to the sample locations yet.
          * vk_meta doesn't care, but the app will!
          */
         cs->has_sample_locations |= !dont_commit;
      } else {
         assert(dont_commit || cs->ppp_multisamplectl == ctrl);
      }

      gfx->descriptors.root.draw.ppp_multisamplectl = ctrl;
      gfx->descriptors.root_dirty = true;
   }

   /* Link varyings before uploading tessellation state, becuase the
    * gfx->generate_primitive_id boolean needs to be plumbed.
    */
   struct hk_linked_shader *linked_vs = gfx->linked[MESA_SHADER_VERTEX];
   struct hk_linked_shader *linked_fs = gfx->linked[MESA_SHADER_FRAGMENT];
   bool linked_vs_dirty = IS_LINKED_DIRTY(VERTEX);
   bool linked_fs_dirty = IS_LINKED_DIRTY(FRAGMENT);

   if ((gfx->dirty & HK_DIRTY_PROVOKING) || vgt_dirty || linked_fs_dirty) {
      unsigned bindings = linked_fs->b.cf.nr_bindings;
      if (bindings) {
         size_t linkage_size =
            AGX_CF_BINDING_HEADER_LENGTH + (bindings * AGX_CF_BINDING_LENGTH);

         const bool native_vit = dev->dev.chip == AGX_CHIP_G17P;
         struct agx_ptr t =
            hk_pool_usc_alloc(cmd, linkage_size, native_vit ? 64 : 16);
         if (!t.cpu)
            return;

         agx_link_varyings_vs_fs(
            t.cpu, &gfx->linked_varyings, hw_vs->info.uvs.user_size,
            &linked_fs->b.cf, gfx->provoking, 0, &gfx->generate_primitive_id);

         gfx->varyings = native_vit ? t.gpu : agx_usc_addr(&dev->dev, t.gpu);
         if (native_vit && hk_g17p_launch_trace_enabled())
            hk_g17p_trace_line("hal200-varying-table", t.gpu, t.cpu,
                               linkage_size);
      } else {
         gfx->varyings = 0;
      }

      gfx->dirty |= HK_DIRTY_VARYINGS;
   }

   if (gfx->shaders[MESA_SHADER_TESS_EVAL] ||
       gfx->shaders[MESA_SHADER_GEOMETRY] || linked_vs->sw_indexing) {
      desc->root.draw.vertex_params = hk_upload_vertex_params(cmd, draw);
      desc->root_dirty = true;
   }

   struct agx_ptr tess_args = {0};
   if (gfx->shaders[MESA_SHADER_TESS_EVAL]) {
      tess_args = hk_pool_alloc(cmd, sizeof(struct poly_tess_params), 4);
      gfx->descriptors.root.draw.tess_params = tess_args.gpu;
      gfx->descriptors.root_dirty = true;
   }

   if (gfx->shaders[MESA_SHADER_GEOMETRY]) {
      gfx->descriptors.root.draw.geometry_params =
         hk_upload_geometry_params(cmd, draw);

      gfx->descriptors.root_dirty = true;
   }

   /* Root must be uploaded after the above, which touch the root */
   if (gfx->descriptors.root_dirty) {
      gfx->root =
         hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);

      /* Tess parameters depend on the root address, so we defer the upload
       * until after uploading root. But the root depends on the tess address,
       * so we allocate tess parameters before uploading root.
       *
       * This whole mechanism is a mess ported over from the GL driver. I'm
       * planning to do a massive rework of indirect geom/tess so I'm trying not
       * to perfectionism it in the mean time.
       */
      if (tess_args.cpu) {
         hk_upload_tess_params(cmd, tess_args.cpu, draw);
      }
   }

   /* Hardware dynamic state must be deferred until after the root and fast
    * linking, since it will use the root address and the linked shaders.
    */
   if ((gfx->dirty & (HK_DIRTY_PROVOKING | HK_DIRTY_VARYINGS)) ||
       IS_DIRTY(RS_RASTERIZER_DISCARD_ENABLE) || linked_vs_dirty || vgt_dirty ||
       gfx->descriptors.root_dirty || gfx->draw_id_ptr || gfx->draw_params ||
       IS_DIRTY(RS_DEPTH_BIAS_ENABLE) || IS_DIRTY(RS_DEPTH_BIAS_FACTORS) ||
       IS_DIRTY(VP_SCISSORS) || IS_DIRTY(VP_SCISSOR_COUNT) ||
       IS_DIRTY(VP_VIEWPORTS) || IS_DIRTY(VP_VIEWPORT_COUNT) ||
       IS_DIRTY(VP_DEPTH_CLIP_NEGATIVE_ONE_TO_ONE) ||
       IS_DIRTY(RS_DEPTH_CLIP_ENABLE) || IS_DIRTY(RS_DEPTH_CLAMP_ENABLE)) {

      /* TODO: Do less often? */
      hk_reserve_scratch(cmd, cs, hw_vs);

      struct hk_linked_shader *linked_hw_vs =
         (hw_vs == sw_vs) ? linked_vs : hw_vs->only_linked;
      const uint64_t pipeline_program =
         hk_upload_usc_words(cmd, hw_vs, linked_hw_vs);

      switch (agx_chip_command_stream_generation(dev->dev.chip)) {
      case AGX_CS_GENERATION_AGX2:
         if (pipeline_program > UINT32_MAX) {
            vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
            cs->current = out;
            return;
         }

         agx_push(out, VDM_STATE, cfg) {
            cfg.vertex_shader_word_0_present = true;
            cfg.vertex_shader_word_1_present = true;
            cfg.vertex_outputs_present = true;
            cfg.vertex_unknown_present = true;
         }

         agx_push_packed(out, hw_vs->counts, VDM_STATE_VERTEX_SHADER_WORD_0);

         agx_push(out, VDM_STATE_VERTEX_SHADER_WORD_1, cfg) {
            cfg.pipeline = pipeline_program;
         }

         agx_push_packed(out, hw_vs->info.uvs.vdm,
                         VDM_STATE_VERTEX_OUTPUTS);

         agx_push(out, VDM_STATE_VERTEX_UNKNOWN, cfg) {
            cfg.flat_shading_control = translate_vdm_vertex(gfx->provoking);
            cfg.unknown_4 = cfg.unknown_5 = hk_rast_discard(cmd);
            cfg.generate_primitive_id = gfx->generate_primitive_id;
         }

         /* Pad up to a multiple of 8 bytes */
         memset(out, 0, 4);
         out += 4;
         break;

      case AGX_CS_GENERATION_G15:
      case AGX_CS_GENERATION_HAL200:
      case AGX_CS_GENERATION_HAL300: {
         const struct agx_vdm_state_update_program_info *program =
            &hw_vs->agx3_vdm_state_update;
         const bool pipeline_va_valid =
            pipeline_program != 0 && pipeline_program < (UINT64_C(1) << 48) &&
            (pipeline_program & UINT64_C(0x3f)) == 0;
         const bool complete_program =
            pipeline_va_valid &&
            hk_agx3_vdm_live_source_layout(
               program, cs->agx3_active_context_selector_valid,
               ordinary_draw_path) ==
               AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT;
         struct agx_vdm_state_update_draw_info info = {
            .source_layout = complete_program
                  ? AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT
                  : AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_UNSPECIFIED,
            .draw_path = agx_is_indirect(draw.b)
               ? (draw.indexed
                     ? AGX_VDM_STATE_UPDATE_DRAW_INDIRECT_INDEXED
                     : AGX_VDM_STATE_UPDATE_DRAW_INDIRECT_NONINDEXED)
               : (draw.indexed
                     ? AGX_VDM_STATE_UPDATE_DRAW_DIRECT_INDEXED
                     : AGX_VDM_STATE_UPDATE_DRAW_DIRECT_NONINDEXED),
            /* RenderPipeline+0x2f0 is populated from
             * isRasterizationEnabled; the selected template carries 0x30
             * when this is false on pre-HAL300. */
            .rasterization_enabled = !hk_rast_discard(cmd),
            /* hk already stores the actual hardware provoking vertex 0..2.
             * Apple's setter maps its API mode to this same value before
             * selecting one of the six templates. */
            .raw_provoking_selector = gfx->provoking,
            .template_info = program->template_info,
         };
         info.template_info.pipeline = pipeline_program;
         info.template_info.raw_active_context_selector =
            cs->agx3_raw_active_context_selector;

         struct agx_vdm_state_update encoded;
         if (!hk_agx3_encode_vdm_state_update_draw(dev->dev.chip, &info,
                                                    &encoded)) {
            mesa_loge("no complete AGX3 VertexProgramVariant state for %s",
                      agx_chip_name(dev->dev.chip));
            vk_command_buffer_set_error(&cmd->vk,
                                        VK_ERROR_FEATURE_NOT_PRESENT);
            cs->current = out;
            return;
         }

         /* Keep both words of the packed 48-bit vertex-pipeline pair.  The
          * Metal witness's 0x01000000/0x00004a00 pair decodes to its own
          * 0x10000128000 pipeline allocation; substituting only the second
          * word redirects Honeykrisp to an unrelated shader container.
          */

         memcpy(out, encoded.bytes, encoded.length);
         out += encoded.length;

         if (dev->dev.chip == AGX_CHIP_G17P) {
            if (!hk_g17p_emit_native_bind_tail(cmd, cs, &out)) {
               mesa_loge("failed to emit native HAL200 bind tail for %s",
                         agx_chip_name(dev->dev.chip));
               vk_command_buffer_set_error(&cmd->vk,
                                           VK_ERROR_FEATURE_NOT_PRESENT);
               cs->current = out;
               return;
            }
            g17p_native_bind_tail_emitted = true;
            *g17p_native_bind_tail_emitted_out = true;
         }
         break;
      }

      default:
         mesa_loge("unknown command-stream generation for %s",
                   agx_chip_name(dev->dev.chip));
         vk_command_buffer_set_error(&cmd->vk,
                                     VK_ERROR_FEATURE_NOT_PRESENT);
         cs->current = out;
         return;
      }
   }

   /* A pure cull/winding update needs new immutable fragment state, but
    * leaves the vertex program and its VDM state unchanged. */
   if (dev->dev.chip == AGX_CHIP_G17P && !g17p_native_bind_tail_emitted &&
       (IS_DIRTY(RS_CULL_MODE) || IS_DIRTY(RS_FRONT_FACE))) {
      if (!hk_g17p_emit_native_bind_tail(cmd, cs, &out)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         cs->current = out;
         return;
      }
      g17p_native_bind_tail_emitted = true;
      *g17p_native_bind_tail_emitted_out = true;
   }

   if (!g17p_native_bind_tail_emitted && IS_DIRTY(RS_DEPTH_BIAS_FACTORS)) {
      bool exact = dyn->rs.depth_bias.exact;
      bool force_unorm =
         dyn->rs.depth_bias.representation ==
         VK_DEPTH_BIAS_REPRESENTATION_LEAST_REPRESENTABLE_VALUE_FORCE_UNORM_EXT;

      const float slope_scale = dyn->rs.depth_bias.slope_factor;
      const float clamp = dyn->rs.depth_bias.clamp;
      const float depth_bias =
         dyn->rs.depth_bias.constant_factor /
         hk_depth_bias_factor(render->depth_att.vk_format, exact, force_unorm);
      uint8_t record[AGX_DEPTH_BIAS_LENGTH];

      static_assert(AGX_DEPTH_BIAS_LENGTH ==
                    AGX_PPP_HAL300_DEPTH_BIAS_RECORD_LENGTH);
      if (agx_chip_command_stream_generation(dev->dev.chip) ==
          AGX_CS_GENERATION_HAL300) {
         const struct agx_ppp_hal300_depth_bias_record_fields fields = {
            .depth_bias_raw = fui(depth_bias),
            .slope_scale_raw = fui(slope_scale),
            .clamp_raw = fui(clamp),
         };
         struct agx_ppp_hal300_depth_bias_record exact_record;
         if (!agx_ppp_hal300_depth_bias_record_pack(
                dev->dev.chip, &fields, &exact_record)) {
            mesa_loge("no grounded HAL300 depth-bias record for %s",
                      agx_chip_name(dev->dev.chip));
            vk_command_buffer_set_error(&cmd->vk,
                                        VK_ERROR_FEATURE_NOT_PRESENT);
            cs->current = out;
            return;
         }
         memcpy(record, exact_record.bytes, exact_record.length);
      } else {
         agx_pack(record, DEPTH_BIAS, cfg) {
            cfg.depth_bias = depth_bias;
            cfg.slope_scale = slope_scale;
            cfg.clamp = clamp;
         }
      }

      void *ptr =
         util_dynarray_grow_bytes(&cs->depth_bias, 1, sizeof(record));
      memcpy(ptr, record, sizeof(record));
   }

   /* Hardware viewport/scissor state is entangled with depth bias. */
   if (!g17p_native_bind_tail_emitted &&
       (IS_DIRTY(RS_DEPTH_BIAS_FACTORS) || IS_DIRTY(VP_SCISSORS) ||
       IS_DIRTY(VP_SCISSOR_COUNT) || IS_DIRTY(VP_VIEWPORTS) ||
       IS_DIRTY(VP_VIEWPORT_COUNT) ||
       IS_DIRTY(VP_DEPTH_CLIP_NEGATIVE_ONE_TO_ONE) ||
       IS_DIRTY(RS_DEPTH_CLIP_ENABLE) || IS_DIRTY(RS_DEPTH_CLAMP_ENABLE))) {

      hk_flush_vp_state(cmd, cs, &out);
      if (vk_command_buffer_has_error(&cmd->vk)) {
         cs->current = out;
         return;
      }
   }

   if (!g17p_native_bind_tail_emitted)
      hk_flush_ppp_state(cmd, cs, &out);
   cs->current = out;

   if (vk_command_buffer_has_error(&cmd->vk))
      return;

   vk_dynamic_graphics_state_clear_dirty(dyn);
   gfx->shaders_dirty = 0;
   gfx->linked_dirty = 0;
   gfx->dirty = 0;
   gfx->descriptors.root_dirty = false;
}

static bool
hk_needs_index_robustness(struct hk_cmd_buffer *cmd, struct agx_draw *draw)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   if (!draw->indexed)
      return false;

   /* Geometry or tessellation use robust software index buffer fetch anyway */
   if (gfx->shaders[MESA_SHADER_GEOMETRY] ||
       gfx->shaders[MESA_SHADER_TESS_EVAL])
      return false;

   /* Soft fault does not cover the hardware index buffer fetch, use the zero
    * page instead.
    */
   if (draw->index_buffer_range_B == 0) {
      draw->index_buffer = AGX_ZERO_PAGE_ADDRESS;
      draw->index_buffer_range_B = 4;
      draw->start = 0;
      return false;
   }

   if (!(dev->vk.enabled_features.robustBufferAccess ||
         dev->vk.enabled_features.robustBufferAccess2 ||
         dev->vk.enabled_features.pipelineRobustness))
      return false;

   if (agx_is_indirect(draw->b))
      return true;

   return agx_direct_draw_overreads_indices(*draw);
}

static void
hk_handle_passthrough_gs(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_api_shader *gs = gfx->shaders[MESA_SHADER_GEOMETRY];

   if (!IS_SHADER_DIRTY(VERTEX) && !IS_SHADER_DIRTY(GEOMETRY))
      return;

   /* If there's an application geometry shader, there's nothing to un/bind */
   if (gs && !gs->is_passthrough)
      return;

   /* Determine if we need a geometry shader to emulate XFB or adjacency */
   struct hk_shader *last_sw = hk_bound_sw_vs_before_gs(gfx);
   uint32_t xfb_outputs = last_sw->info.xfb_info.output_count;
   bool needs_gs = xfb_outputs;

   /* If we don't need a GS but we do have a passthrough, unbind it */
   if (!needs_gs) {
      if (gs != NULL) {
         assert(gs->is_passthrough);
         hk_cmd_bind_graphics_shader(cmd, MESA_SHADER_GEOMETRY, NULL);
      }
      return;
   }

   /* Else, we need to bind a passthrough GS */
   size_t key_size = hk_passthrough_gs_key_size(xfb_outputs);
   struct hk_passthrough_gs_key *key = alloca(key_size);

   *key = (struct hk_passthrough_gs_key){
      .prim = u_decomposed_prim(hk_gs_in_prim(cmd)),
      .outputs = last_sw->b.info.outputs,
      .clip_distance_array_size = last_sw->info.clip_distance_array_size,
      .cull_distance_array_size = last_sw->info.cull_distance_array_size,
   };

   if (xfb_outputs) {
      typed_memcpy(key->xfb_stride, last_sw->info.xfb_stride,
                   ARRAY_SIZE(key->xfb_stride));

      memcpy(&key->xfb_info, &last_sw->info.xfb_info,
             nir_xfb_info_size(xfb_outputs));
   }

   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   perf_debug(cmd, "Binding passthrough GS for%s\n", xfb_outputs ? " XFB" : "");

   gs = hk_meta_shader(dev, hk_nir_passthrough_gs, key, key_size);
   gs->is_passthrough = true;
   hk_cmd_bind_graphics_shader(cmd, MESA_SHADER_GEOMETRY, gs);
}

static struct hk_cs *
hk_flush_gfx_state(struct hk_cmd_buffer *cmd, uint32_t draw_id,
                   struct agx_draw *draw, bool *needs_idx_robust,
                   bool *g17p_native_bind_tail_emitted)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_descriptor_state *desc = &gfx->descriptors;

   struct hk_cs *cs = hk_cmd_buffer_get_cs(cmd, false /* compute */);
   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;

   if (!cs)
      return NULL;

   /* Annoyingly,
    * VK_DEPTH_BIAS_REPRESENTATION_LEAST_REPRESENTABLE_VALUE_FORCE_UNORM_EXT is
    * render pass state on Imaginapple but draw state in Vulkan. In practice,
    * Proton never changes it within a render pass, but we technically need to
    * handle the switch regardless. Do so early since `cs` will be invalidated
    * if we need to split the render pass to switch representation mid-frame.
    *
    * Note we only do this dance with depth bias is actually enabled to avoid
    * senseless control stream splits with DXVK.
    */
   if ((IS_DIRTY(RS_DEPTH_BIAS_FACTORS) || IS_DIRTY(RS_DEPTH_BIAS_ENABLE)) &&
       dyn->rs.depth_bias.enable) {

      bool dbias_is_int =
         (dyn->rs.depth_bias.representation ==
          VK_DEPTH_BIAS_REPRESENTATION_LEAST_REPRESENTABLE_VALUE_FORCE_UNORM_EXT) ||
         (gfx->render.depth_att.vk_format == VK_FORMAT_D16_UNORM);

      /* Attempt to set dbias_is_int per the draw requirement. If this fails,
       * flush the control stream and set it on the new control stream.
       */
      bool succ = u_tristate_set(&cs->cr.dbias_is_int, dbias_is_int);
      if (!succ) {
         perf_debug(cmd, "Splitting control stream due to depth bias");

         hk_cmd_buffer_end_graphics(cmd);
         cs = hk_cmd_buffer_get_cs(cmd, false /* compute */);

         succ = u_tristate_set(&cs->cr.dbias_is_int, dbias_is_int);
         assert(succ && "can always set tri-state on a new control stream");
      }
   }

   /* Native G17P concatenates ordinary draws in one 0x8000-byte VDM root.
    * Its in-stream link packet is not yet established on hardware, so cross
    * that uncommon capacity boundary with the already-correct retained-render
    * transition rather than silently dropping the linked suffix.
    */
   if (hk_g17p_vdm_needs_render_split(
          dev->dev.chip, cs->stats.cmds, cs->end - cs->current,
          AGX_VDM_STREAM_LINK_LENGTH)) {
      if (hk_g17p_launch_trace_enabled())
         fprintf(stderr, "G17P VDM root full: starting a retained render\n");

      hk_cmd_buffer_end_graphics(cmd);
      cs = hk_cmd_buffer_get_cs(cmd, false /* compute */);
      if (!cs)
         return NULL;
   }

   hk_ensure_cs_has_space(cmd, cs, HK_GFX_DRAW_RESERVATION);

#ifndef NDEBUG
   if (unlikely(dev->dev.debug & AGX_DBG_DIRTY)) {
      hk_cmd_buffer_dirty_all(cmd);
   }
#endif

   /* Merge tess info before GS construction since that depends on
    * gfx->tess.prim
    */
   if ((IS_SHADER_DIRTY(TESS_CTRL) || IS_SHADER_DIRTY(TESS_EVAL)) &&
       gfx->shaders[MESA_SHADER_TESS_CTRL]) {
      struct hk_api_shader *tcs = gfx->shaders[MESA_SHADER_TESS_CTRL];
      struct hk_api_shader *tes = gfx->shaders[MESA_SHADER_TESS_EVAL];
      struct hk_shader *tese = hk_any_variant(tes);
      struct hk_shader *tesc = hk_only_variant(tcs);

      gfx->tess.info =
         hk_tess_info_merge(tese->info.tess.info, tesc->info.tess.info);

      /* Determine primitive based on the merged state */
      if (gfx->tess.info.points) {
         gfx->tess.prim = MESA_PRIM_POINTS;
      } else if (gfx->tess.info.mode == TESS_PRIMITIVE_ISOLINES) {
         gfx->tess.prim = MESA_PRIM_LINES;
      } else {
         gfx->tess.prim = MESA_PRIM_TRIANGLES;
      }
   }

   /* TODO: Try to reduce draw overhead of this */
   hk_handle_passthrough_gs(cmd, *draw);

   hk_flush_shaders(cmd);

   const bool geom = gfx->shaders[MESA_SHADER_GEOMETRY];
   const bool tess = gfx->shaders[MESA_SHADER_TESS_EVAL];
   bool adj = mesa_prim_has_adjacency(
      vk_topology_to_mesa(dyn->ia.primitive_topology));
   adj &= !geom;
   *needs_idx_robust = hk_needs_index_robustness(cmd, draw);
   const bool ordinary_draw_path =
      !geom && !tess && !adj && !*needs_idx_robust;

   if (desc->push_dirty)
      hk_cmd_buffer_flush_push_descriptors(cmd, desc);

   if (draw->restart || geom)
      hk_flush_index(cmd, cs);

   hk_flush_dynamic_state(cmd, cs, draw_id, *draw, ordinary_draw_path,
                          g17p_native_bind_tail_emitted);
   return cs;
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBindIndexBuffer2KHR(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                          VkDeviceSize offset, VkDeviceSize size,
                          VkIndexType indexType)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_buffer, buffer, _buffer);

   cmd->state.gfx.index = (struct hk_index_buffer_state){
      .buffer = hk_buffer_addr_range(buffer, offset, size, true),
      .size = agx_translate_index_size(vk_index_type_to_bytes(indexType)),
      .restart = vk_index_to_restart(indexType),
   };

   /* TODO: check if necessary, blob does this */
   cmd->state.gfx.index.buffer.range =
      align(cmd->state.gfx.index.buffer.range, 4);
}

void
hk_cmd_bind_vertex_buffer(struct hk_cmd_buffer *cmd, uint32_t vb_idx,
                          struct hk_addr_range addr_range)
{
   cmd->state.gfx.vb[vb_idx] = addr_range;
   cmd->state.gfx.dirty |= HK_DIRTY_VB;
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBindVertexBuffers2(VkCommandBuffer commandBuffer, uint32_t firstBinding,
                         uint32_t bindingCount, const VkBuffer *pBuffers,
                         const VkDeviceSize *pOffsets,
                         const VkDeviceSize *pSizes,
                         const VkDeviceSize *pStrides)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   if (pStrides) {
      vk_cmd_set_vertex_binding_strides(&cmd->vk, firstBinding, bindingCount,
                                        pStrides);
   }

   for (uint32_t i = 0; i < bindingCount; i++) {
      VK_FROM_HANDLE(hk_buffer, buffer, pBuffers[i]);
      uint32_t idx = firstBinding + i;

      uint64_t size = pSizes ? pSizes[i] : VK_WHOLE_SIZE;
      const struct hk_addr_range addr_range =
         hk_buffer_addr_range(buffer, pOffsets[i], size, true);

      hk_cmd_bind_vertex_buffer(cmd, idx, addr_range);
   }
}

static bool
hk_set_view_index(struct hk_cmd_buffer *cmd, uint32_t view_idx)
{
   if (cmd->state.gfx.render.view_mask) {
      cmd->state.gfx.descriptors.root.draw.view_index = view_idx;
      cmd->state.gfx.descriptors.root_dirty = true;
   }

   return true;
}

/*
 * Iterator macro to duplicate a draw for each enabled view (when multiview is
 * enabled, else always view 0). Along with hk_lower_multiview, this forms the
 * world's worst multiview lowering.
 */
#define hk_foreach_view(cmd)                                                   \
   u_foreach_bit(view_idx, cmd->state.gfx.render.view_mask ?: 1)               \
      if (hk_set_view_index(cmd, view_idx))

static void
hk_ia_update(struct hk_cmd_buffer *cmd, struct agx_draw draw,
             uint64_t ia_vertices, uint64_t ia_prims, uint64_t vs_invocations,
             uint64_t c_prims, uint64_t c_inv)
{
   perf_debug(cmd, "Input assembly counters");

   uint64_t draw_ptr;
   if (agx_is_indirect(draw.b)) {
      draw_ptr = draw.b.ptr;
   } else {
      uint32_t desc[] = {draw.b.count[0], draw.b.count[1], 0};
      draw_ptr = hk_pool_upload(cmd, &desc, sizeof(desc), 4);
   }

   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   enum mesa_prim prim = vk_topology_to_mesa(dyn->ia.primitive_topology);

   bool geom = cmd->state.gfx.shaders[MESA_SHADER_GEOMETRY];
   bool tess = cmd->state.gfx.shaders[MESA_SHADER_TESS_EVAL];

   /* Clipper counters depend on geom/tess outputs and must be written with the
    * geom/tess output. They are updated as IA counters only when geom/tess is
    * not used.
    */
   if (geom || tess) {
      c_prims = 0;
      c_inv = 0;
   }

   uint32_t patch_size = dyn->ts.patch_control_points;

   if (draw.restart) {
      uint32_t index_size_B = agx_index_size_to_B(draw.index_size);

      libagx_increment_ia_restart(
         cmd, agx_1d(1024), AGX_BARRIER_ALL | AGX_PREGFX, ia_vertices, ia_prims,
         vs_invocations, c_prims, c_inv, draw_ptr, draw.index_buffer,
         agx_draw_index_range_el(draw), cmd->state.gfx.index.restart,
         index_size_B, prim, patch_size);
   } else {
      libagx_increment_ia(cmd, agx_1d(1), AGX_BARRIER_ALL | AGX_PREGFX,
                          ia_vertices, ia_prims, vs_invocations, c_prims, c_inv,
                          draw_ptr, prim, patch_size);
   }
}

static void
hk_draw(struct hk_cmd_buffer *cmd, uint16_t draw_id, struct agx_draw draw_)
{
   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;

   /* Filter trivial draws so we don't need to worry about null index buffers */
   if (!agx_is_indirect(draw_.b) &&
       (draw_.b.count[0] == 0 || draw_.b.count[1] == 0))
      return;

   draw_.restart = dyn->ia.primitive_restart_enable && draw_.indexed;
   draw_.index_size = cmd->state.gfx.index.size;

   uint64_t stat_ia_verts = hk_pipeline_stat_addr(
      cmd, VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_VERTICES_BIT);

   uint64_t stat_ia_prims = hk_pipeline_stat_addr(
      cmd, VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_PRIMITIVES_BIT);

   uint64_t stat_vs_inv = hk_pipeline_stat_addr(
      cmd, VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT);

   uint64_t stat_c_inv = hk_pipeline_stat_addr(
      cmd, VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT);

   uint64_t stat_c_prims = hk_pipeline_stat_addr(
      cmd, VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT);

   bool ia_stats = hk_stat_enabled(stat_ia_verts) ||
                   hk_stat_enabled(stat_ia_prims) ||
                   hk_stat_enabled(stat_vs_inv) ||
                   hk_stat_enabled(stat_c_inv) || hk_stat_enabled(stat_c_prims);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   /* Validate the native direct index span before robustness or geometry
    * lowering can introduce a pre-render compute dispatch. */
   if (dev->dev.chip == AGX_CHIP_G17P && draw_.indexed) {
      uint64_t address;
      uint32_t size_word;
      if (agx_is_indirect(draw_.b) ||
          (draw_.index_size != AGX_INDEX_SIZE_U8 &&
           draw_.index_size != AGX_INDEX_SIZE_U16 &&
           draw_.index_size != AGX_INDEX_SIZE_U32) ||
          ia_stats || cmd->state.gfx.shaders[MESA_SHADER_GEOMETRY] ||
          cmd->state.gfx.shaders[MESA_SHADER_TESS_EVAL] ||
          !agx_vdm_hal200_index_range(
             draw_.index_buffer, draw_.index_buffer_range_B, draw_.start,
             draw_.b.count[0], agx_index_size_to_B(draw_.index_size),
             &address, &size_word)) {
         mesa_loge("native G17P indexed draw requires an in-bounds direct "
                   "uint8/uint16/uint32 span without geometry or query lowering");
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }
   }

   hk_foreach_view(cmd) {
      struct agx_draw draw = draw_;
      bool needs_idx_robust = false;
      bool g17p_native_bind_tail_emitted = false;
      struct hk_cs *cs =
         hk_flush_gfx_state(cmd, draw_id, &draw, &needs_idx_robust,
                            &g17p_native_bind_tail_emitted);
      /* If we failed to allocate a control stream, we've already lost the
       * device. Just drop the draw so we don't crash.
       */
      if (!cs)
         return;
      if (vk_command_buffer_has_error(&cmd->vk))
         return;

      struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
      bool geom = cmd->state.gfx.shaders[MESA_SHADER_GEOMETRY];
      bool tess = cmd->state.gfx.shaders[MESA_SHADER_TESS_EVAL];
      bool adj = mesa_prim_has_adjacency(
         vk_topology_to_mesa(dyn->ia.primitive_topology));
      adj &= !geom;
      needs_idx_robust &= !adj;

      struct hk_cs *ccs = NULL;
      uint8_t *out = cs->current;
      assert(cs->current + 0x1000 < cs->end);

      if (tess && HK_PERF(dev, NOTESS))
         continue;

      cs->stats.calls++;

      if (geom || tess) {
         ccs =
            hk_cmd_buffer_get_cs_general(cmd, &cmd->current_cs.pre_gfx, true);
         if (!ccs)
            return;
      }

      if (ia_stats) {
         hk_ia_update(cmd, draw, stat_ia_verts, stat_ia_prims, stat_vs_inv,
                      stat_c_prims, stat_c_inv);
      }

      if (tess) {
         draw = hk_launch_tess(cmd, ccs, draw, geom ? 0 : stat_c_prims,
                               geom ? 0 : stat_c_inv);
      }

      if (geom) {
         draw = hk_launch_gs_prerast(cmd, ccs, draw);
      }

      if (adj) {
         assert(!geom && "geometry shaders handle adj directly");
         enum mesa_prim prim = vk_topology_to_mesa(dyn->ia.primitive_topology);

         if (draw.restart) {
            draw = hk_draw_without_restart(cmd, draw, 1);
            prim = u_decomposed_prim(prim);
         }

         if (agx_is_indirect(draw.b)) {
            const size_t size = sizeof(VkDrawIndexedIndirectCommand);
            static_assert(sizeof(VkDrawIndexedIndirectCommand) >
                             sizeof(VkDrawIndirectCommand),
                          "allocation size is conservative");

            uint64_t out_draw = hk_pool_alloc(cmd, size, 4).gpu;
            struct hk_descriptor_state *desc = &cmd->state.gfx.descriptors;

            libagx_draw_without_adj(
               cmd, agx_1d(1), AGX_BARRIER_ALL | AGX_PREGFX, out_draw,
               draw.b.ptr, desc->root.draw.vertex_params, draw.index_buffer,
               draw.indexed ? agx_draw_index_range_el(draw) : 0,
               draw.indexed ? agx_index_size_to_B(draw.index_size) : 0, prim);

            draw = agx_draw_indirect(out_draw);
         } else {
            unsigned count = libagx_remap_adj_count(draw.b.count[0], prim);

            draw = (struct agx_draw){
               .b = agx_3d(count, draw.b.count[1], 1),
            };
         }
      }

      enum agx_primitive topology = cmd->state.gfx.topology;
      if (!agx_vdm_draw_supported(dev->dev.chip, draw, topology)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }

      if (needs_idx_robust) {
         assert(!geom && !tess && !adj);
         perf_debug(cmd, "lowering robust index buffer");

         cs->current = out;

         draw = hk_draw_as_indexed_indirect(cmd, draw);
         if (!agx_vdm_draw_supported(dev->dev.chip, draw, topology)) {
            vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
            return;
         }

         size_t size_B =
            libagx_draw_robust_index_vdm_size(dev->dev.chip);
         uint64_t target = hk_cs_alloc_for_indirect(cs, size_B);

         libagx_draw_robust_index(cmd, agx_1d(32), AGX_BARRIER_ALL | AGX_PREGFX,
                                  target, hk_heap(cmd), draw.b.ptr,
                                  draw.index_buffer, draw.index_buffer_range_B,
                                  draw.restart, dev->dev.chip, topology,
                                  draw.index_size);
      } else {
         if (dev->dev.chip == AGX_CHIP_G17P) {
            if (agx_is_indirect(draw.b)) {
               mesa_loge("native HAL200 indirect draw framing is not yet admitted");
               vk_command_buffer_set_error(&cmd->vk,
                                           VK_ERROR_FEATURE_NOT_PRESENT);
               return;
            }

            /* This context-relative PPP record selects the latest scissor
             * and depth-bias array entries. Header0x100 is PPP selector bit8,
             * not a direct-draw shape; +4 contains indices, not enable flags.
             * Native bind-state emission appended those owned records above.
             */
            struct agx_ptr draw_state = hk_pool_vdm_alloc(cmd, 8, 4);
            uint32_t draw_state_offset;
            if (!draw_state.cpu ||
                !hk_g17p_upload_draw_indices(draw_state.cpu, 8, draw_state.gpu,
                    cs->scissor.size, cs->depth_bias.size, &draw_state_offset)) {
               vk_command_buffer_set_error(&cmd->vk,
                                           VK_ERROR_FEATURE_NOT_PRESENT);
               return;
            }
            if (draw.indexed) {
               const struct agx_vdm_hal200_indexed_draw_fields fields = {
                  .primitive = (uint8_t)topology,
                  .index_size_B = agx_index_size_to_B(draw.index_size),
                  .restart_enable = draw.restart,
                  .restart_index = cmd->state.gfx.index.restart,
                  .draw_state_context_offset = draw_state_offset,
                  .index_buffer = draw.index_buffer,
                  .index_buffer_range_B = draw.index_buffer_range_B,
                  .first_index = draw.start,
                  .index_count = draw.b.count[0],
                  .instance_count = draw.b.count[1],
                  .base_vertex = (int32_t)draw.index_bias,
               };
               struct agx_vdm_hal200_indexed_draw encoded;
               if (!agx_vdm_hal200_indexed_draw_pack(dev->dev.chip, &fields,
                                                      &encoded)) {
                  vk_command_buffer_set_error(&cmd->vk,
                                              VK_ERROR_FEATURE_NOT_PRESENT);
                  return;
               }
               /* Metal's indexed entry point emits the 36-byte draw body
                * after encodeAndEmitRenderState.  Its leading four-byte PPP
                * selector is emitted only with a fresh native bind tail. */
               size_t emitted;
               if (!agx_vdm_hal200_draw_emit(
                      dev->dev.chip, encoded.bytes, encoded.length,
                      g17p_native_bind_tail_emitted, out,
                      (uint8_t *)cs->end - out,
                      &emitted)) {
                  vk_command_buffer_set_error(&cmd->vk,
                                              VK_ERROR_FEATURE_NOT_PRESENT);
                  return;
               }
               cs->current = out + emitted;
               if (hk_g17p_launch_trace_enabled())
                  hk_g17p_trace_line("hal200-indexed-draw", draw.index_buffer,
                                     out, emitted);
            } else {
               const struct agx_vdm_hal200_direct_draw_fields fields = {
                  .draw_state_context_offset = draw_state_offset,
                  .primitive = (uint8_t)topology,
                  .vertex_count = draw.b.count[0],
                  .instance_count = draw.b.count[1],
                  .vertex_start = draw.start,
               };
               struct agx_vdm_hal200_direct_draw encoded;
               if (!agx_vdm_hal200_direct_draw_pack(dev->dev.chip, &fields,
                                                     &encoded)) {
                  vk_command_buffer_set_error(&cmd->vk,
                                              VK_ERROR_FEATURE_NOT_PRESENT);
                  return;
               }

               /* The native non-indexed entry point likewise emits a bare
                * 16-byte body when render state is unchanged.  A standalone
                * selector before it makes 0x61c4 part of a bogus VDM address. */
               size_t emitted;
               if (!agx_vdm_hal200_draw_emit(
                      dev->dev.chip, encoded.bytes, encoded.length,
                      g17p_native_bind_tail_emitted, out,
                      (uint8_t *)cs->end - out,
                      &emitted)) {
                  vk_command_buffer_set_error(&cmd->vk,
                                              VK_ERROR_FEATURE_NOT_PRESENT);
                  return;
               }
               cs->current = out + emitted;
               if (hk_g17p_launch_trace_enabled())
                  hk_g17p_trace_line("hal200-direct-draw", 0, out, emitted);
            }
            if (hk_g17p_launch_trace_enabled())
               hk_g17p_trace_line("hal200-draw-state",
                                  draw_state.gpu, draw_state.cpu, 8);
         } else {
            cs->current = (void *)agx_vdm_draw((uint32_t *)out,
                                               dev->dev.chip, draw, topology);
         }
      }

      cs->stats.cmds++;
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDraw(VkCommandBuffer commandBuffer, uint32_t vertexCount,
           uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct agx_draw draw;

   if (HK_TEST_INDIRECTS) {
      uint32_t data[] = {
         vertexCount,
         instanceCount,
         firstVertex,
         firstInstance,
      };

      draw = agx_draw_indirect(hk_pool_upload(cmd, data, sizeof(data), 4));
   } else {
      draw = (struct agx_draw){
         .b = agx_3d(vertexCount, instanceCount, 1),
         .start = firstVertex,
         .start_instance = firstInstance,
      };
   }

   hk_draw(cmd, 0, draw);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawMultiEXT(VkCommandBuffer commandBuffer, uint32_t drawCount,
                   const VkMultiDrawInfoEXT *pVertexInfo,
                   uint32_t instanceCount, uint32_t firstInstance,
                   uint32_t stride)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   for (unsigned i = 0; i < drawCount; ++i) {
      struct agx_draw draw = {
         .b = agx_3d(pVertexInfo->vertexCount, instanceCount, 1),
         .start = pVertexInfo->firstVertex,
         .start_instance = firstInstance,
      };

      hk_draw(cmd, i, draw);
      pVertexInfo = ((void *)pVertexInfo) + stride;
   }
}

static void
hk_draw_indexed(VkCommandBuffer commandBuffer, uint16_t draw_id,
                uint32_t indexCount, uint32_t instanceCount,
                uint32_t firstIndex, int32_t vertexOffset,
                uint32_t firstInstance)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct agx_draw draw;
   struct hk_addr_range buf = cmd->state.gfx.index.buffer;

   if (HK_TEST_INDIRECTS && draw_id == 0) {
      uint32_t data[] = {
         indexCount, instanceCount, firstIndex, vertexOffset, firstInstance,
      };
      uint64_t addr = hk_pool_upload(cmd, data, sizeof(data), 4);

      draw = agx_draw_indexed_indirect(addr, buf.addr, buf.range, 0, 0);
   } else {
      draw =
         agx_draw_indexed(indexCount, instanceCount, firstIndex, vertexOffset,
                          firstInstance, buf.addr, buf.range, 0, 0);
   }

   hk_draw(cmd, draw_id, draw);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawIndexed(VkCommandBuffer commandBuffer, uint32_t indexCount,
                  uint32_t instanceCount, uint32_t firstIndex,
                  int32_t vertexOffset, uint32_t firstInstance)
{
   hk_draw_indexed(commandBuffer, 0, indexCount, instanceCount, firstIndex,
                   vertexOffset, firstInstance);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawMultiIndexedEXT(VkCommandBuffer commandBuffer, uint32_t drawCount,
                          const VkMultiDrawIndexedInfoEXT *pIndexInfo,
                          uint32_t instanceCount, uint32_t firstInstance,
                          uint32_t stride, const int32_t *pVertexOffset)
{
   for (unsigned i = 0; i < drawCount; ++i) {
      const int32_t vertex_offset =
         pVertexOffset != NULL ? *pVertexOffset : pIndexInfo->vertexOffset;

      hk_draw_indexed(commandBuffer, i, pIndexInfo->indexCount, instanceCount,
                      pIndexInfo->firstIndex, vertex_offset, firstInstance);

      pIndexInfo = ((void *)pIndexInfo) + stride;
   }
}

static void
hk_draw_indirect_inner(VkCommandBuffer commandBuffer, uint64_t base,
                       uint32_t drawCount, uint32_t stride)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   /* From the Vulkan 1.3.238 spec:
    *
    *    VUID-vkCmdDrawIndirect-drawCount-00476
    *
    *    "If drawCount is greater than 1, stride must be a multiple of 4 and
    *    must be greater than or equal to sizeof(VkDrawIndirectCommand)"
    *
    * and
    *
    *    "If drawCount is less than or equal to one, stride is ignored."
    */
   if (drawCount > 1) {
      assert(stride % 4 == 0);
      assert(stride >= sizeof(VkDrawIndirectCommand));
   }

   for (unsigned draw_id = 0; draw_id < drawCount; ++draw_id) {
      uint64_t addr = base + stride * draw_id;
      hk_draw(cmd, draw_id, agx_draw_indirect(addr));
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawIndirect(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                   VkDeviceSize offset, uint32_t drawCount, uint32_t stride)
{
   VK_FROM_HANDLE(hk_buffer, buffer, _buffer);

   hk_draw_indirect_inner(commandBuffer, hk_buffer_address_ro(buffer, offset),
                          drawCount, stride);
}

static void
hk_draw_indexed_indirect_inner(VkCommandBuffer commandBuffer, uint64_t buffer,
                               uint32_t drawCount, uint32_t stride)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   /* From the Vulkan 1.3.238 spec:
    *
    *    VUID-vkCmdDrawIndexedIndirect-drawCount-00528
    *
    *    "If drawCount is greater than 1, stride must be a multiple of 4 and
    *    must be greater than or equal to
    * sizeof(VkDrawIndexedIndirectCommand)"
    *
    * and
    *
    *    "If drawCount is less than or equal to one, stride is ignored."
    */
   if (drawCount > 1) {
      assert(stride % 4 == 0);
      assert(stride >= sizeof(VkDrawIndexedIndirectCommand));
   }

   for (unsigned draw_id = 0; draw_id < drawCount; ++draw_id) {
      uint64_t addr = buffer + stride * draw_id;
      struct hk_addr_range buf = cmd->state.gfx.index.buffer;

      hk_draw(cmd, draw_id,
              agx_draw_indexed_indirect(addr, buf.addr, buf.range, 0, 0));
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawIndexedIndirect(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                          VkDeviceSize offset, uint32_t drawCount,
                          uint32_t stride)
{
   VK_FROM_HANDLE(hk_buffer, buffer, _buffer);

   hk_draw_indexed_indirect_inner(
      commandBuffer, hk_buffer_address_ro(buffer, offset), drawCount, stride);
}

/*
 * To implement drawIndirectCount generically, we dispatch a compute kernel to
 * patch the indirect buffer and then we dispatch the predicated maxDrawCount
 * indirect draws.
 */
static void
hk_draw_indirect_count(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                       VkDeviceSize offset, VkBuffer countBuffer,
                       VkDeviceSize countBufferOffset, uint32_t maxDrawCount,
                       uint32_t stride, bool indexed)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_buffer, buffer, _buffer);
   VK_FROM_HANDLE(hk_buffer, count_buffer, countBuffer);

   perf_debug(cmd, "Draw indirect count");

   assert((stride % 4) == 0 && "aligned");

   size_t out_stride = sizeof(uint32_t) * (indexed ? 5 : 4);
   uint64_t patched = hk_pool_alloc(cmd, out_stride * maxDrawCount, 4).gpu;
   uint64_t in = hk_buffer_address_ro(buffer, offset);
   uint64_t count_addr = hk_buffer_address_ro(count_buffer, countBufferOffset);

   libagx_predicate_indirect(cmd, agx_1d(maxDrawCount),
                             AGX_BARRIER_ALL | AGX_PREGFX, patched, in,
                             count_addr, stride / 4, indexed);

   if (indexed) {
      hk_draw_indexed_indirect_inner(commandBuffer, patched, maxDrawCount,
                                     out_stride);
   } else {
      hk_draw_indirect_inner(commandBuffer, patched, maxDrawCount, out_stride);
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawIndirectCount(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                        VkDeviceSize offset, VkBuffer countBuffer,
                        VkDeviceSize countBufferOffset, uint32_t maxDrawCount,
                        uint32_t stride)
{
   hk_draw_indirect_count(commandBuffer, _buffer, offset, countBuffer,
                          countBufferOffset, maxDrawCount, stride, false);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawIndexedIndirectCount(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                               VkDeviceSize offset, VkBuffer countBuffer,
                               VkDeviceSize countBufferOffset,
                               uint32_t maxDrawCount, uint32_t stride)
{
   hk_draw_indirect_count(commandBuffer, _buffer, offset, countBuffer,
                          countBufferOffset, maxDrawCount, stride, true);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawIndirectByteCountEXT(VkCommandBuffer commandBuffer,
                               uint32_t instanceCount, uint32_t firstInstance,
                               VkBuffer counterBuffer,
                               VkDeviceSize counterBufferOffset,
                               uint32_t counterOffset, uint32_t vertexStride)
{
   UNREACHABLE("TODO");
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBindTransformFeedbackBuffersEXT(VkCommandBuffer commandBuffer,
                                      uint32_t firstBinding,
                                      uint32_t bindingCount,
                                      const VkBuffer *pBuffers,
                                      const VkDeviceSize *pOffsets,
                                      const VkDeviceSize *pSizes)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_graphics_state *gfx = &cmd->state.gfx;

   for (uint32_t i = 0; i < bindingCount; i++) {
      VK_FROM_HANDLE(hk_buffer, buffer, pBuffers[i]);
      uint32_t idx = firstBinding + i;
      uint64_t size = pSizes ? pSizes[i] : VK_WHOLE_SIZE;

      gfx->xfb[idx] = hk_buffer_addr_range(buffer, pOffsets[i], size, false);
   }
}

static void
hk_begin_end_xfb(VkCommandBuffer commandBuffer, uint32_t firstCounterBuffer,
                 uint32_t counterBufferCount, const VkBuffer *pCounterBuffers,
                 const VkDeviceSize *pCounterBufferOffsets, bool begin)

{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_graphics_state *gfx = &cmd->state.gfx;

   gfx->xfb_enabled = begin;

   /* If we haven't reserved XFB offsets yet for the command buffer, do so. */
   if (!gfx->xfb_offsets) {
      gfx->xfb_offsets = hk_pool_alloc(cmd, 4 * sizeof(uint32_t), 4).gpu;
   }

   struct libagx_xfb_counter_copy params = {};
   unsigned copies = 0;

   /* For CmdBeginTransformFeedbackEXT, we need to initialize everything */
   if (begin) {
      for (copies = 0; copies < 4; ++copies) {
         params.dest[copies] = gfx->xfb_offsets + copies * sizeof(uint32_t);
      }
   }

   for (unsigned i = 0; i < counterBufferCount; ++i) {
      if (pCounterBuffers[i] == VK_NULL_HANDLE)
         continue;

      VK_FROM_HANDLE(hk_buffer, buffer, pCounterBuffers[i]);

      uint64_t offset = pCounterBufferOffsets ? pCounterBufferOffsets[i] : 0;
      uint64_t cb_addr = hk_buffer_address_rw(buffer, offset);
      uint32_t cmd_idx = firstCounterBuffer + i;

      if (begin) {
         params.src[cmd_idx] = cb_addr;
      } else {
         params.dest[copies] = cb_addr;
         params.src[copies] = gfx->xfb_offsets + cmd_idx * sizeof(uint32_t);
         ++copies;
      }
   }

   if (begin)
      copies = 4;

   if (copies > 0) {
      perf_debug(cmd, "XFB counter copy");

      libagx_copy_xfb_counters(cmd, agx_1d(copies),
                               AGX_BARRIER_ALL | AGX_PREGFX,
                               hk_pool_upload(cmd, &params, sizeof(params), 8));
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBeginTransformFeedbackEXT(VkCommandBuffer commandBuffer,
                                uint32_t firstCounterBuffer,
                                uint32_t counterBufferCount,
                                const VkBuffer *pCounterBuffers,
                                const VkDeviceSize *pCounterBufferOffsets)
{
   hk_begin_end_xfb(commandBuffer, firstCounterBuffer, counterBufferCount,
                    pCounterBuffers, pCounterBufferOffsets, true);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdEndTransformFeedbackEXT(VkCommandBuffer commandBuffer,
                              uint32_t firstCounterBuffer,
                              uint32_t counterBufferCount,
                              const VkBuffer *pCounterBuffers,
                              const VkDeviceSize *pCounterBufferOffsets)
{
   hk_begin_end_xfb(commandBuffer, firstCounterBuffer, counterBufferCount,
                    pCounterBuffers, pCounterBufferOffsets, false);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBeginConditionalRenderingEXT(
   VkCommandBuffer commandBuffer,
   const VkConditionalRenderingBeginInfoEXT *pConditionalRenderingBegin)
{
   UNREACHABLE("stub");
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdEndConditionalRenderingEXT(VkCommandBuffer commandBuffer)
{
   UNREACHABLE("stub");
}
