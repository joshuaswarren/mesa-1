/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "hk_cmd_buffer.h"

#include "agx_abi.h"
#include "agx_bo.h"
#include "agx_device.h"
#include "agx_linker.h"
#include "agx_tilebuffer.h"
#include "agx_usc.h"
#include "hk_agx3_ioto_program_core.h"
#include "hk_agx3_vdm_decode.h"
#include "hk_buffer.h"
#include "hk_cmd_pool.h"
#include "hk_descriptor_set.h"
#include "hk_descriptor_set_layout.h"
#include "hk_device.h"
#include "hk_device_memory.h"
#include "hk_entrypoints.h"
#include "hk_image.h"
#include "hk_image_view.h"
#include "hk_physical_device.h"
#include "hk_shader.h"

#include "util/log.h"
#include "util/u_debug.h"

#include <inttypes.h>
#include <stdio.h>

#include "asahi/genxml/agx3_cmdstream.h"
#include "asahi/lib/agx_ppp_hal300_depth_stencil_core.h"
#include "asahi/lib/agx_ppp_w_clamp_cull2_core.h"
#include "asahi/lib/agx_vdm_barrier_core.h"
#include "asahi/lib/agx_vdm_stream_link_core.h"
#include "asahi/lib/agx_vdm_stream_terminate_core.h"
#include "asahi/lib/agx_va_runtime_hole.h"
#include "libagx_dgc.h"
#include "pool.h"
#include "shader_enums.h"
#include "vk_pipeline_layout.h"
#include "vk_common_entrypoints.h"
#include "vk_synchronization.h"

#include "util/list.h"
#include "util/macros.h"
#include "util/u_dynarray.h"
#include "vulkan/vulkan_core.h"

static uint64_t
hk_g17p_vdm_lookup(struct hk_cmd_buffer *cmd, uint64_t primary, bool bind)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (!hk_g17p_vdm_low_window_enabled(&dev->dev))
      return primary;

   list_for_each_entry(struct hk_cmd_bo, bo, &cmd->uploader.main.bos, link) {
      const uint64_t start = bo->bo->va->addr;
      if (primary < start || primary - start >= bo->bo->size)
         continue;

      if (bind) {
         const VkResult result =
            hk_cmd_bo_ensure_vdm_alias(hk_cmd_buffer_pool(cmd), bo);
         if (result != VK_SUCCESS) {
            vk_command_buffer_set_error(&cmd->vk, result);
            return 0;
         }
      }

      const uint64_t alias = bo->g17p_vdm_alias + (primary - start);
      uint32_t compact = 0;
      if (agx_g17p_vdm_alias_compact(alias, &compact))
         return compact;
      break;
   }

   mesa_loge("G17P VDM address %#" PRIx64 " has no low command-BO alias",
             primary);
   return 0;
}

uint64_t
hk_g17p_vdm_compact(struct hk_cmd_buffer *cmd, uint64_t primary)
{
   return hk_g17p_vdm_lookup(cmd, primary, true);
}

uint64_t
hk_g17p_vdm_alias(struct hk_cmd_buffer *cmd, uint64_t primary)
{
   /* Submission may read a simultaneous-use CB from multiple queues. All
    * VDM roots/links were mapped during recording; never mutate them here. */
   const uint64_t compact = hk_g17p_vdm_lookup(cmd, primary, false);
   return compact ? AGX_G17P_VDM_USER_BASE + compact : 0;
}

static void
hk_descriptor_state_fini(struct hk_cmd_buffer *cmd,
                         struct hk_descriptor_state *desc)
{
   struct hk_cmd_pool *pool = hk_cmd_buffer_pool(cmd);

   for (unsigned i = 0; i < HK_MAX_SETS; i++) {
      vk_free(&pool->vk.alloc, desc->push[i]);
      desc->push[i] = NULL;
   }
}

static void
hk_free_resettable_cmd_buffer(struct hk_cmd_buffer *cmd)
{
   struct hk_cmd_pool *pool = hk_cmd_buffer_pool(cmd);
   struct hk_device *dev = hk_cmd_pool_device(pool);

   hk_descriptor_state_fini(cmd, &cmd->state.gfx.descriptors);
   hk_descriptor_state_fini(cmd, &cmd->state.cs.descriptors);

   hk_cmd_pool_free_bo_list(pool, &cmd->uploader.main.bos);
   hk_cmd_pool_free_usc_bo_list(pool, &cmd->uploader.usc.bos);

   list_for_each_entry_safe(struct hk_cs, it, &cmd->control_streams, node) {
      list_del(&it->node);
      hk_cs_destroy(it);
   }

   hk_cmd_buffer_release_bos(cmd, &dev->dev);
}

static void
hk_destroy_cmd_buffer(struct vk_command_buffer *vk_cmd_buffer)
{
   struct hk_cmd_buffer *cmd =
      container_of(vk_cmd_buffer, struct hk_cmd_buffer, vk);
   struct hk_cmd_pool *pool = hk_cmd_buffer_pool(cmd);

   hk_free_resettable_cmd_buffer(cmd);
   util_dynarray_fini(&cmd->large_bos);
   vk_command_buffer_finish(&cmd->vk);
   vk_free(&pool->vk.alloc, cmd);
}

static VkResult
hk_create_cmd_buffer(struct vk_command_pool *vk_pool,
                     VkCommandBufferLevel level,
                     struct vk_command_buffer **cmd_buffer_out)
{
   struct hk_cmd_pool *pool = container_of(vk_pool, struct hk_cmd_pool, vk);
   struct hk_device *dev = hk_cmd_pool_device(pool);
   struct hk_cmd_buffer *cmd;
   VkResult result;

   cmd = vk_zalloc(&pool->vk.alloc, sizeof(*cmd), 8,
                   VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (cmd == NULL)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   result = vk_command_buffer_init_with_params(&cmd->vk,
                                               &(struct vk_command_buffer_init_params) {
                                                  .pool = &pool->vk,
                                                  .ops = &hk_cmd_buffer_ops,
                                                  .level = level,
                                                  .needs_cmd_queue = level == VK_COMMAND_BUFFER_LEVEL_SECONDARY,
                                               });
   if (result != VK_SUCCESS) {
      vk_free(&pool->vk.alloc, cmd);
      return result;
   }

   cmd->large_bos = UTIL_DYNARRAY_INIT;

   cmd->vk.dynamic_graphics_state.vi = &cmd->state.gfx._dynamic_vi;
   cmd->vk.dynamic_graphics_state.ms.sample_locations =
      &cmd->state.gfx._dynamic_sl;

   list_inithead(&cmd->uploader.main.bos);
   list_inithead(&cmd->uploader.usc.bos);
   list_inithead(&cmd->control_streams);

   *cmd_buffer_out = &cmd->vk;

   return VK_SUCCESS;
}

static void
hk_reset_cmd_buffer(struct vk_command_buffer *vk_cmd_buffer,
                    UNUSED VkCommandBufferResetFlags flags)
{
   struct hk_cmd_buffer *cmd =
      container_of(vk_cmd_buffer, struct hk_cmd_buffer, vk);

   vk_command_buffer_reset(&cmd->vk);
   hk_free_resettable_cmd_buffer(cmd);

   cmd->uploader.main.map = NULL;
   cmd->uploader.main.base = 0;
   cmd->uploader.main.offset = 0;
   cmd->uploader.usc.map = NULL;
   cmd->uploader.usc.base = 0;
   cmd->uploader.usc.offset = 0;

   cmd->current_cs.gfx = NULL;
   cmd->current_cs.cs = NULL;
   cmd->current_cs.post_gfx = NULL;
   cmd->current_cs.pre_gfx = NULL;

   assert(!cmd->in_meta);
   cmd->geom_index_buffer = 0;
   cmd->geom_index_count = 0;
   cmd->geom_instance_count = 0;
   cmd->uses_heap = false;

   memset(&cmd->state, 0, sizeof(cmd->state));
}

const struct vk_command_buffer_ops hk_cmd_buffer_ops = {
   .create = hk_create_cmd_buffer,
   .reset = hk_reset_cmd_buffer,
   .destroy = hk_destroy_cmd_buffer,
};

VKAPI_ATTR void VKAPI_CALL
hk_CmdExecuteCommands(VkCommandBuffer commandBuffer,
                      uint32_t commandBufferCount,
                      const VkCommandBuffer *pCommandBuffers)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   /* Secondary command buffers are materialized by replaying their common
    * command queues into the primary.  Their last bound state is therefore
    * also the driver's cached state in the primary after replay.  Vulkan
    * makes that state undefined (except for the active render pass), so an
    * application is required to bind it again.  Invalidate here so binding
    * the same object used by the secondary cannot be incorrectly elided.
    */
   vk_common_CmdExecuteCommands(commandBuffer, commandBufferCount,
                                pCommandBuffers);
   hk_cmd_invalidate_graphics_state(cmd);
   hk_cmd_invalidate_compute_state(cmd);
}

static VkResult
hk_cmd_buffer_alloc_bo(struct hk_cmd_buffer *cmd, bool usc,
                       struct hk_cmd_bo **bo_out)
{
   VkResult result = hk_cmd_pool_alloc_bo(hk_cmd_buffer_pool(cmd), usc, bo_out);
   if (result != VK_SUCCESS)
      return result;

   if (usc)
      list_addtail(&(*bo_out)->link, &cmd->uploader.usc.bos);
   else
      list_addtail(&(*bo_out)->link, &cmd->uploader.main.bos);

   return VK_SUCCESS;
}

struct agx_ptr
hk_pool_alloc_internal(struct hk_cmd_buffer *cmd, uint32_t size,
                       uint32_t alignment, bool usc)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_uploader *uploader =
      usc ? &cmd->uploader.usc : &cmd->uploader.main;

   /* Specially handle large allocations owned by the command buffer, e.g. used
    * for statically allocated vertex output buffers with geometry shaders.
    */
   if (size > HK_CMD_BO_SIZE) {
      uint32_t flags = usc ? AGX_BO_LOW_VA : 0;
      struct agx_bo *bo =
         agx_bo_create(&dev->dev, size, 0, flags, "Large pool allocation");

      if (unlikely(bo == NULL)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         return (struct agx_ptr){0};
      }

      util_dynarray_append(&cmd->large_bos, bo);
      return (struct agx_ptr){
         .gpu = bo->va->addr,
         .cpu = agx_bo_map(bo),
      };
   }

   assert(size <= HK_CMD_BO_SIZE);
   assert(alignment > 0);

   uint32_t offset = align(uploader->offset, alignment);

   assert(offset <= HK_CMD_BO_SIZE);
   if (uploader->map != NULL && size <= HK_CMD_BO_SIZE - offset) {
      uploader->offset = offset + size;

      return (struct agx_ptr){
         .gpu = uploader->base + offset,
         .cpu = uploader->map + offset,
      };
   }

   struct hk_cmd_bo *bo;
   VkResult result = hk_cmd_buffer_alloc_bo(cmd, usc, &bo);
   if (unlikely(result != VK_SUCCESS)) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return (struct agx_ptr){0};
   }

   /* Pick whichever of the current upload BO and the new BO will have more
    * room left to be the BO for the next upload.  If our upload size is
    * bigger than the old offset, we're better off burning the whole new
    * upload BO on this one allocation and continuing on the current upload
    * BO.
    */
   if (uploader->map == NULL || size < uploader->offset) {
      uploader->map = agx_bo_map(bo->bo);
      uploader->base = bo->bo->va->addr;
      uploader->offset = size;
   }

   return (struct agx_ptr){
      .gpu = bo->bo->va->addr,
      .cpu = bo->map,
   };
}

uint64_t
hk_pool_upload(struct hk_cmd_buffer *cmd, const void *data, uint32_t size,
               uint32_t alignment)
{
   struct agx_ptr T = hk_pool_alloc(cmd, size, alignment);
   if (unlikely(T.cpu == NULL))
      return 0;

   memcpy(T.cpu, data, size);
   return T.gpu;
}

uint64_t
hk_pool_usc_upload(struct hk_cmd_buffer *cmd, const void *data, uint32_t size,
                   uint32_t alignment)
{
   struct agx_ptr T = hk_pool_usc_alloc(cmd, size, alignment);
   if (unlikely(T.cpu == NULL))
      return 0;

   memcpy(T.cpu, data, size);
   return T.gpu;
}

void *
hk_cmd_buffer_emit_cs_jump(struct hk_cmd_buffer *cmd, void *current,
                           uint64_t target, bool vdm)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   if (vdm && hk_g17p_vdm_low_window_enabled(&dev->dev))
      target = hk_g17p_vdm_compact(cmd, target);

   switch (agx_chip_command_stream_generation(dev->dev.chip)) {
   case AGX_CS_GENERATION_AGX2:
      return agx_cs_jump(current, target, vdm);

   case AGX_CS_GENERATION_G15:
   case AGX_CS_GENERATION_HAL200:
   case AGX_CS_GENERATION_HAL300:
      if (vdm) {
         struct agx_vdm_stream_link_core link;
         if (agx_vdm_stream_link_core_pack(dev->dev.chip, target, &link)) {
            memcpy(current, link.bytes, link.length);
            return (uint8_t *)current + link.length;
         }
      } else {
         const struct agx3_cdm_stream_link_info info = {
            .chip = dev->dev.chip,
            .target = target,
            .with_return = false,
         };
         uint8_t link[AGX3_CDM_STREAM_LINK_SIZE];
         if (agx3_cdm_stream_link_pack(&info, link)) {
            memcpy(current, link, sizeof(link));
            return (uint8_t *)current + sizeof(link);
         }
      }
      break;

   case AGX_CS_GENERATION_UNKNOWN:
   default:
      break;
   }

   mesa_loge("no grounded %s stream link for %s", vdm ? "VDM" : "CDM",
             agx_chip_name(dev->dev.chip));
   vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
   return current;
}

/* Diagnostics for the G17P native compute/graphics route.
 *
 * The raw add3 probe and a Honeykrisp dispatch drive the *same* kernel path
 * with the same 44-byte launch + 4-byte terminate shape, yet only the
 * Honeykrisp one makes the firmware request a stamp-blamed recovery. The two
 * objects that differ are the CDM stream as a whole and the composed Ioto
 * program; both are small, so dump them verbatim and diff against the
 * known-good 48-byte add3 stream instead of reasoning about them.
 */
bool
hk_g17p_launch_trace_enabled(void)
{
   static int enabled = -1;
   if (enabled < 0)
      enabled = debug_get_bool_option("HK_G17P_LAUNCH_TRACE", false);
   return enabled != 0;
}

/* Single line, distinct tag. The multi-line dump below has twice been reported
 * as "only the header printed", so anything that must not be lost in a log
 * filter goes through this instead. */
void
hk_g17p_trace_line(const char *label, uint64_t base, const void *data,
                   size_t len)
{
   const uint8_t *bytes = data;
   fprintf(stderr, "G17P %s: base=%" PRIx64 " len=%zu bytes=", label, base,
           len);
   for (size_t i = 0; i < len; ++i)
      fprintf(stderr, "%02x", bytes[i]);
   fprintf(stderr, "\n");
   fflush(stderr);
}

void
hk_g17p_trace_hexdump(const char *label, uint64_t base, const void *data,
                      size_t len)
{
   const uint8_t *bytes = data;
   fprintf(stderr, "G17P dump: %s base=%" PRIx64 " len=%zu\n", label, base,
           len);
   for (size_t off = 0; off < len; off += 16) {
      fprintf(stderr, "G17P dump:   +%04zx ", off);
      for (size_t i = 0; i < 16 && off + i < len; ++i)
         fprintf(stderr, "%02x ", bytes[off + i]);
      fprintf(stderr, "\n");
   }
   fflush(stderr);
}

void *
hk_cmd_buffer_emit_cdm_terminate(struct hk_cmd_buffer *cmd, void *current)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   switch (agx_chip_command_stream_generation(dev->dev.chip)) {
   case AGX_CS_GENERATION_AGX2:
      return agx_cdm_terminate(current);

   case AGX_CS_GENERATION_G15:
   case AGX_CS_GENERATION_HAL200:
   case AGX_CS_GENERATION_HAL300: {
      uint8_t terminate[AGX3_CDM_STREAM_TERMINATE_SIZE];
      if (agx3_cdm_stream_terminate_pack(dev->dev.chip, terminate)) {
         memcpy(current, terminate, sizeof(terminate));
         return (uint8_t *)current + sizeof(terminate);
      }
      break;
   }

   case AGX_CS_GENERATION_UNKNOWN:
   default:
      break;
   }

   mesa_loge("no grounded CDM stream terminator for %s",
             agx_chip_name(dev->dev.chip));
   vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
   return current;
}

void *
hk_cmd_buffer_emit_vdm_terminate(struct hk_cmd_buffer *cmd, void *current)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   /* Draw emission leaves the stream open for further state, draws or links.
    * EndCommandStream supplies exactly one terminal record here. Do not infer
    * termination from the preceding dword: it may be a draw's vertex_start.
    */
   switch (agx_chip_command_stream_generation(dev->dev.chip)) {
   case AGX_CS_GENERATION_AGX2:
      return agx_vdm_terminate(current);

   case AGX_CS_GENERATION_G15:
   case AGX_CS_GENERATION_HAL200:
   case AGX_CS_GENERATION_HAL300: {
      struct agx_vdm_stream_terminate_core terminate;
      if (agx_vdm_stream_terminate_core_pack(dev->dev.chip, &terminate)) {
         memcpy(current, terminate.bytes, terminate.length);
         return (uint8_t *)current + terminate.length;
      }
      break;
   }

   case AGX_CS_GENERATION_UNKNOWN:
   default:
      break;
   }

   mesa_loge("no grounded VDM stream terminator for %s",
             agx_chip_name(dev->dev.chip));
   vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
   return current;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_BeginCommandBuffer(VkCommandBuffer commandBuffer,
                      const VkCommandBufferBeginInfo *pBeginInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   hk_reset_cmd_buffer(&cmd->vk, 0);

   perf_debug(cmd, "Begin command buffer");
   hk_cmd_buffer_begin_compute(cmd, pBeginInfo);
   hk_cmd_buffer_begin_graphics(cmd, pBeginInfo);

   return VK_SUCCESS;
}

/*
 * Merge adjacent compute control streams. Except for reading timestamps, there
 * is no reason to submit two CDM streams back-to-back in the same command
 * buffer. However, it is challenging to avoid constructing such sequences due
 * to the gymnastics required to reorder compute around graphics. Merging at
 * EndCommandBuffer is cheap O(# of control streams) and lets us get away with
 * the sloppiness.
 */
static void
merge_control_streams(struct hk_cmd_buffer *cmd)
{
   struct hk_cs *last = NULL;

   list_for_each_entry_safe(struct hk_cs, cs, &cmd->control_streams, node) {
      if (cs->type == HK_CS_CDM && last && last->type == HK_CS_CDM &&
          !last->timestamp.end.handle) {

         hk_cs_merge_cdm(last, cs);
         list_del(&cs->node);
         hk_cs_destroy(cs);
      } else {
         last = cs;
      }
   }
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_EndCommandBuffer(VkCommandBuffer commandBuffer)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   assert(cmd->current_cs.gfx == NULL && cmd->current_cs.pre_gfx == NULL &&
          "must end rendering before ending the command buffer");

   perf_debug(cmd, "End command buffer");
   hk_cmd_buffer_end_compute(cmd);
   hk_cmd_buffer_end_compute_internal(cmd, &cmd->current_cs.post_gfx);

   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (likely(!(dev->dev.debug & AGX_DBG_NOMERGE))) {
      merge_control_streams(cmd);
   }

   /* We cannot terminate CDM control streams until after merging, since merging
    * needs to append stream links late. Now that we've merged, insert all the
    * missing stream terminates.
    */
   list_for_each_entry(struct hk_cs, cs, &cmd->control_streams, node) {
      if (cs->type == HK_CS_CDM) {
         cs->current = hk_cmd_buffer_emit_cdm_terminate(cmd, cs->current);
      }
   }

   return vk_command_buffer_get_record_result(&cmd->vk);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdPipelineBarrier2(VkCommandBuffer commandBuffer,
                       const VkDependencyInfo *pDependencyInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   if (HK_PERF(dev, NOBARRIER))
      return;

   perf_debug(cmd, "Pipeline barrier");

   /* The big hammer. We end both compute and graphics batches. Ending compute
    * here is necessary to properly handle graphics->compute dependencies.
    *
    * XXX: perf. */
   hk_cmd_buffer_end_compute(cmd);
   hk_cmd_buffer_end_graphics(cmd);
}

void
hk_cmd_bind_shaders(struct vk_command_buffer *vk_cmd, uint32_t stage_count,
                    const mesa_shader_stage *stages,
                    struct vk_shader **const shaders)
{
   struct hk_cmd_buffer *cmd = container_of(vk_cmd, struct hk_cmd_buffer, vk);

   for (uint32_t i = 0; i < stage_count; i++) {
      struct hk_api_shader *shader =
         container_of(shaders[i], struct hk_api_shader, vk);

      if (stages[i] == MESA_SHADER_COMPUTE || stages[i] == MESA_SHADER_KERNEL)
         hk_cmd_bind_compute_shader(cmd, shader);
      else
         hk_cmd_bind_graphics_shader(cmd, stages[i], shader);
   }
}

static void
hk_bind_descriptor_sets(UNUSED struct hk_cmd_buffer *cmd,
                        struct hk_descriptor_state *desc,
                        const VkBindDescriptorSetsInfoKHR *info)
{
   VK_FROM_HANDLE(vk_pipeline_layout, pipeline_layout, info->layout);

   /* Fro the Vulkan 1.3.275 spec:
    *
    *    "When binding a descriptor set (see Descriptor Set Binding) to
    *    set number N...
    *
    *    If, additionally, the previously bound descriptor set for set
    *    N was bound using a pipeline layout not compatible for set N,
    *    then all bindings in sets numbered greater than N are
    *    disturbed."
    *
    * This means that, if some earlier set gets bound in such a way that
    * it changes set_dynamic_buffer_start[s], this binding is implicitly
    * invalidated.
    */
   uint8_t dyn_buffer_start =
      pipeline_layout->dynamic_descriptor_offset[info->firstSet];

   uint32_t next_dyn_offset = 0;
   for (uint32_t i = 0; i < info->descriptorSetCount; ++i) {
      unsigned s = i + info->firstSet;
      VK_FROM_HANDLE(hk_descriptor_set, set, info->pDescriptorSets[i]);

      if (desc->sets[s] != set) {
         if (set != NULL) {
            desc->root.sets[s] = hk_descriptor_set_addr(set);
            desc->set_sizes[s] = set->size;
         } else {
            desc->root.sets[s] = 0;
            desc->set_sizes[s] = 0;
         }
         desc->sets[s] = set;
         desc->sets_dirty |= BITFIELD_BIT(s);

         /* Binding descriptors invalidates push descriptors */
         desc->push_dirty &= ~BITFIELD_BIT(s);
      }

      desc->root.set_dynamic_buffer_start[s] = dyn_buffer_start;

      if (pipeline_layout->set_layouts[s] != NULL) {
         const struct hk_descriptor_set_layout *set_layout =
            vk_to_hk_descriptor_set_layout(pipeline_layout->set_layouts[s]);

         if (set != NULL && set_layout->vk.dynamic_descriptor_count > 0) {
            for (uint32_t j = 0; j < set_layout->vk.dynamic_descriptor_count; j++) {
               struct hk_buffer_address addr = set->dynamic_buffers[j];
               addr.base_addr += info->pDynamicOffsets[next_dyn_offset + j];
               desc->root.dynamic_buffers[dyn_buffer_start + j] = addr;
            }
            next_dyn_offset += set->layout->vk.dynamic_descriptor_count;
         }

         dyn_buffer_start += set_layout->vk.dynamic_descriptor_count;
      } else {
         assert(set == NULL);
      }
   }
   assert(dyn_buffer_start <= HK_MAX_DYNAMIC_BUFFERS);
   assert(next_dyn_offset <= info->dynamicOffsetCount);

   desc->root_dirty = true;
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBindDescriptorSets2KHR(
   VkCommandBuffer commandBuffer,
   const VkBindDescriptorSetsInfoKHR *pBindDescriptorSetsInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   if (pBindDescriptorSetsInfo->stageFlags & VK_SHADER_STAGE_ALL_GRAPHICS) {
      hk_bind_descriptor_sets(cmd, &cmd->state.gfx.descriptors,
                              pBindDescriptorSetsInfo);
   }

   if (pBindDescriptorSetsInfo->stageFlags & VK_SHADER_STAGE_COMPUTE_BIT) {
      hk_bind_descriptor_sets(cmd, &cmd->state.cs.descriptors,
                              pBindDescriptorSetsInfo);
   }
}

static void
hk_push_constants(UNUSED struct hk_cmd_buffer *cmd,
                  struct hk_descriptor_state *desc,
                  const VkPushConstantsInfoKHR *info)
{
   memcpy(desc->root.push + info->offset, info->pValues, info->size);
   desc->root_dirty = true;
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdPushConstants2KHR(VkCommandBuffer commandBuffer,
                        const VkPushConstantsInfoKHR *pPushConstantsInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   if (pPushConstantsInfo->stageFlags & VK_SHADER_STAGE_ALL_GRAPHICS)
      hk_push_constants(cmd, &cmd->state.gfx.descriptors, pPushConstantsInfo);

   if (pPushConstantsInfo->stageFlags & VK_SHADER_STAGE_COMPUTE_BIT)
      hk_push_constants(cmd, &cmd->state.cs.descriptors, pPushConstantsInfo);
}

static struct hk_push_descriptor_set *
hk_cmd_push_descriptors(struct hk_cmd_buffer *cmd,
                        struct hk_descriptor_state *desc, uint32_t set)
{
   assert(set < HK_MAX_SETS);
   if (unlikely(desc->push[set] == NULL)) {
      desc->push[set] =
         vk_zalloc(&cmd->vk.pool->alloc, sizeof(*desc->push[set]), 8,
                   VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
      if (unlikely(desc->push[set] == NULL)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
         return NULL;
      }
   }

   /* Pushing descriptors replaces whatever sets are bound */
   desc->sets[set] = NULL;
   desc->push_dirty |= BITFIELD_BIT(set);

   return desc->push[set];
}

static void
hk_push_descriptor_set(struct hk_cmd_buffer *cmd,
                       struct hk_descriptor_state *desc,
                       const VkPushDescriptorSetInfoKHR *info)
{
   VK_FROM_HANDLE(vk_pipeline_layout, pipeline_layout, info->layout);

   struct hk_push_descriptor_set *push_set =
      hk_cmd_push_descriptors(cmd, desc, info->set);
   if (unlikely(push_set == NULL))
      return;

   struct hk_descriptor_set_layout *set_layout =
      vk_to_hk_descriptor_set_layout(pipeline_layout->set_layouts[info->set]);

   hk_push_descriptor_set_update(push_set, set_layout,
                                 info->descriptorWriteCount,
                                 info->pDescriptorWrites);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdPushDescriptorSet2KHR(
   VkCommandBuffer commandBuffer,
   const VkPushDescriptorSetInfoKHR *pPushDescriptorSetInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   if (pPushDescriptorSetInfo->stageFlags & VK_SHADER_STAGE_ALL_GRAPHICS) {
      hk_push_descriptor_set(cmd, &cmd->state.gfx.descriptors,
                             pPushDescriptorSetInfo);
   }

   if (pPushDescriptorSetInfo->stageFlags & VK_SHADER_STAGE_COMPUTE_BIT) {
      hk_push_descriptor_set(cmd, &cmd->state.cs.descriptors,
                             pPushDescriptorSetInfo);
   }
}

void
hk_cmd_buffer_flush_push_descriptors(struct hk_cmd_buffer *cmd,
                                     struct hk_descriptor_state *desc)
{
   u_foreach_bit(set_idx, desc->push_dirty) {
      struct hk_push_descriptor_set *push_set = desc->push[set_idx];
      uint64_t push_set_addr = hk_pool_upload(
         cmd, push_set->data, sizeof(push_set->data), HK_MIN_UBO_ALIGNMENT);

      desc->root.sets[set_idx] = push_set_addr;
      desc->set_sizes[set_idx] = sizeof(push_set->data);
   }

   desc->root_dirty = true;
   desc->push_dirty = 0;
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdPushDescriptorSetWithTemplate2KHR(
   VkCommandBuffer commandBuffer, const VkPushDescriptorSetWithTemplateInfoKHR
                                     *pPushDescriptorSetWithTemplateInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(vk_descriptor_update_template, template,
                  pPushDescriptorSetWithTemplateInfo->descriptorUpdateTemplate);
   VK_FROM_HANDLE(vk_pipeline_layout, pipeline_layout,
                  pPushDescriptorSetWithTemplateInfo->layout);

   struct hk_descriptor_state *desc =
      hk_get_descriptors_state(cmd, template->bind_point);
   struct hk_push_descriptor_set *push_set = hk_cmd_push_descriptors(
      cmd, desc, pPushDescriptorSetWithTemplateInfo->set);
   if (unlikely(push_set == NULL))
      return;

   struct hk_descriptor_set_layout *set_layout = vk_to_hk_descriptor_set_layout(
      pipeline_layout->set_layouts[pPushDescriptorSetWithTemplateInfo->set]);

   hk_push_descriptor_set_update_template(
      push_set, set_layout, template,
      pPushDescriptorSetWithTemplateInfo->pData);
}

uint64_t
hk_cmd_buffer_upload_root(struct hk_cmd_buffer *cmd,
                          VkPipelineBindPoint bind_point)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_descriptor_state *desc = hk_get_descriptors_state(cmd, bind_point);
   struct hk_root_descriptor_table *root = &desc->root;

   struct agx_ptr root_ptr = hk_pool_alloc(cmd, sizeof(*root), 8);
   if (!root_ptr.gpu)
      return 0;

   root->root_desc_addr = root_ptr.gpu;
   root->printf_buffer_address = dev->dev.printf_buffer_address;

   memcpy(root_ptr.cpu, root, sizeof(*root));
   return root_ptr.gpu;
}

void
hk_usc_upload_spilled_rt_descs(struct agx_usc_builder *b,
                               struct hk_cmd_buffer *cmd)
{
   struct hk_rendering_state *render = &cmd->state.gfx.render;
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   const unsigned generation =
      agx_chip_descriptor_generation(dev->dev.chip);

   if (!hk_native_pbe_hardware_supported(generation)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }

   const unsigned texture_size = hk_texture_descriptor_size(generation);
   const unsigned pbe_size = hk_pbe_descriptor_size(generation);
   const unsigned pair_size = texture_size + pbe_size;

   /* Upload texture/PBE descriptors for each render target so we can clear
    * spilled render targets.
    */
   struct agx_ptr descs =
      hk_pool_alloc(cmd, pair_size * render->color_att_count, 64);
   uint8_t *desc = descs.cpu;
   if (!desc)
      return;

   for (unsigned i = 0; i < render->color_att_count; ++i) {
      struct hk_image_view *iview = render->color_att[i].iview;
      if (!iview) {
         /* XXX: probably should emit a null descriptor here...? */
         continue;
      }

      if (!iview->planes[0].emrt_pbe_valid) {
         vk_command_buffer_set_error(&cmd->vk,
                                     VK_ERROR_FORMAT_NOT_SUPPORTED);
         return;
      }

      memcpy(desc + i * pair_size, &iview->planes[0].emrt_texture,
             texture_size);
      memcpy(desc + i * pair_size + texture_size,
             &iview->planes[0].emrt_pbe, pbe_size);
   }

   /* Bind the base as u0_u1 for bindless access */
   agx_usc_uniform(b, 0, 4, hk_pool_upload(cmd, &descs.gpu, 8, 8));
}

void
hk_reserve_scratch(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                   struct hk_shader *s)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   uint32_t max_scratch_size =
      MAX2(s->b.info.scratch_size, s->b.info.preamble_scratch_size);

   /* Not scratch but this is the most convenient place for this... */
   cs->uses_sampler_heap |= s->b.info.uses_sampler_heap;

   if (max_scratch_size == 0)
      return;

   unsigned preamble_size = (s->b.info.preamble_scratch_size > 0) ? 1 : 0;

   /* Note: this uses the hardware stage, not the software stage */
   hk_device_alloc_scratch(dev, s->b.info.stage, max_scratch_size);
   perf_debug(cmd, "Reserving %u (%u) bytes of scratch for stage %s",
              s->b.info.scratch_size, s->b.info.preamble_scratch_size,
              _mesa_shader_stage_to_abbrev(s->b.info.stage));

   switch (s->b.info.stage) {
   case MESA_SHADER_FRAGMENT:
      cs->scratch.fs.main = true;
      cs->scratch.fs.preamble = MAX2(cs->scratch.fs.preamble, preamble_size);
      break;
   case MESA_SHADER_VERTEX:
      cs->scratch.vs.main = true;
      cs->scratch.vs.preamble = MAX2(cs->scratch.vs.preamble, preamble_size);
      break;
   default:
      cs->scratch.cs.main = true;
      cs->scratch.cs.preamble = MAX2(cs->scratch.cs.preamble, preamble_size);
      break;
   }
}

static bool
hk_g17p_basic_fragment_layout(struct hk_cmd_buffer *cmd)
{
   struct agx_tilebuffer_layout *tib = &cmd->state.gfx.render.tilebuffer;
   return hk_agx3_g17p_rt0_tile_layout_supported(tib);
}

/* Keep native interpolation bounded to the f32 UVS layout already supported
 * by hk_agx3_g17p_vertex_program_info: position plus up to 28 user words.
 * HAL200 updateVaryingTable (0x3efa58) uses the same CF entry encoding for
 * smooth, linear and flat inputs, including their mixture in one shader.
 * Flat inputs are restricted to non-fan triangle primitives; the compiler
 * selects C and therefore must not admit the first-fan layout. Position Z and
 * reciprocal W use the native system coefficient sources.
 */
bool
hk_g17p_smooth_varyings_supported(struct hk_cmd_buffer *cmd,
                                 const struct hk_linked_shader *linked)
{
   if (!linked)
      return false;
   const struct agx_varyings_fs *cf = &linked->b.cf;
   if (cf->nr_bindings == 0)
      return cf->nr_cf == 0;

   const struct hk_graphics_state *gfx = &cmd->state.gfx;
   const struct hk_api_shader *api_vs = gfx->shaders[MESA_SHADER_VERTEX];
   const struct hk_shader *vs =
      api_vs ? &api_vs->variants[HK_VS_VARIANT_HW] : NULL;
   if (!vs || gfx->shaders[MESA_SHADER_GEOMETRY] ||
       gfx->shaders[MESA_SHADER_TESS_EVAL] ||
       cf->nr_bindings > HK_AGX3_G17P_MAX_USER_WORDS + 2 ||
       cf->nr_cf > HK_AGX3_G17P_MAX_USER_WORDS + 2 ||
       vs->agx3_vdm_state_update.source_layout !=
          AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT ||
       vs->info.uvs.user_size > HK_AGX3_G17P_MAX_USER_WORDS)
      return false;

   unsigned next_cf = 0;
   bool has_rhw = false, has_z = false;
   unsigned user_bindings = 0;
   for (unsigned i = 0; i < cf->nr_bindings; ++i) {
      const struct agx_cf_binding *b = &cf->bindings[i];
      if (b->count == 0 || b->count > 4 || b->cf_base != next_cf)
         return false;
      if (!b->smooth &&
          (b->perspective ||
           (gfx->topology != AGX_PRIMITIVE_TRIANGLES &&
            gfx->topology != AGX_PRIMITIVE_TRIANGLE_STRIP) ||
           (gfx->provoking != 0 && gfx->provoking != 2)))
         return false;
      if (b->slot == VARYING_SLOT_POS) {
         /* System coefficients precede user coefficients: optional W,
          * then optional Z. agx_link_varyings assigns native Z source1. */
         if (user_bindings || b->count != 1 || b->perspective || !b->smooth)
            return false;
         if (b->offset == 3) {
            if (i != 0 || has_rhw || has_z)
               return false;
            has_rhw = true;
         } else if (b->offset == 2) {
            if (has_z || !cf->reads_z)
               return false;
            has_z = true;
         } else {
            return false;
         }
      } else {
         if (b->slot < VARYING_SLOT_VAR0 || b->slot > VARYING_SLOT_VAR31 ||
             (b->perspective && !has_rhw) ||
             b->offset + b->count > vs->info.uvs.components[b->slot])
            return false;
         ++user_bindings;
         const unsigned slot = gfx->linked_varyings.slots[b->slot];
         if (slot < 4 || slot + b->offset + b->count > vs->info.uvs.size)
            return false;
      }
      next_cf += b->count;
   }
   return (user_bindings || has_rhw || has_z) &&
          has_z == cf->reads_z && next_cf == cf->nr_cf;
}

static bool
hk_g17p_basic_graphics_program_supported(struct hk_cmd_buffer *cmd,
                                         struct hk_shader *s,
                                         struct hk_linked_shader *linked)
{
   if (!linked || !linked->b.bo || !linked->b.bo->va || linked->b.size == 0 ||
       linked->b.size > UINT32_MAX || s->b.info.has_preamble ||
       s->b.info.scratch_size != 0 ||
       s->b.info.preamble_scratch_size != 0 || linked->b.scratch_size != 0 ||
       s->b.info.texture_state_count != 0 ||
       (s->b.info.sampler_state_count != 0 &&
        !(s->info.agx3_indexed_sampler_valid &&
          s->b.info.sampler_state_count == 1)) ||
       s->b.info.uses_sampler_heap ||
       (linked->b.uses_txf && !s->info.agx3_indexed_sampler_valid) ||
       s->data_size != 0 || s->info.set_count > HK_AGX3_G17P_MAX_ROOT_SETS)
      return false;

   if (s->info.stage == MESA_SHADER_VERTEX) {
      const unsigned nr_attribs = DIV_ROUND_UP(
         BITSET_LAST_BIT(s->info.vs.attrib_components_read), 4);

      return !linked->sw_indexing &&
             nr_attribs <= AGX_MAX_VBUFS &&
             s->agx3_vdm_state_update.source_layout ==
                AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT;
   }

   return s->info.stage == MESA_SHADER_FRAGMENT &&
          hk_g17p_smooth_varyings_supported(cmd, linked) &&
          hk_g17p_basic_fragment_layout(cmd);
}

static bool
hk_g17p_graphics_uniform_ranges(
   struct hk_cmd_buffer *cmd, struct hk_shader *s,
   struct agx3_ioto_uniform_range ranges[HK_AGX3_G17P_GRAPHICS_MAX_RANGES],
   size_t *range_count)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   const uint64_t root_ptr = cmd->state.gfx.root;
   size_t count = 0;
   unsigned root_uniform = 0;

   if (s->info.stage == MESA_SHADER_VERTEX) {
      const unsigned nr_attribs = DIV_ROUND_UP(
         BITSET_LAST_BIT(s->info.vs.attrib_components_read), 4);
      if (nr_attribs > AGX_MAX_VBUFS)
         return false;

      if (nr_attribs != 0) {
         if (!root_ptr)
            return false;

         ranges[count++] = (struct agx3_ioto_uniform_range){
            .start_halfs = AGX_ABI_VUNI_VBO_BASE(0),
            .size_halfs = 4 * nr_attribs,
            .address =
               root_ptr + hk_root_descriptor_offset(draw.attrib_base),
         };
         ranges[count++] = (struct agx3_ioto_uniform_range){
            .start_halfs = AGX_ABI_VUNI_VBO_CLAMP(nr_attribs, 0),
            .size_halfs = 2 * nr_attribs,
            .address =
               root_ptr + hk_root_descriptor_offset(draw.attrib_clamps),
         };
      }

      /* The compiler's vertex ABI reserves these independent ranges after
       * attribute bases/clamps. Direct draws already upload baseVertex and
       * firstInstance together; native Ioto can bind them without a prolog
       * specialization, shader patch or pre-render compute dispatch. */
      if (cmd->state.gfx.draw_params) {
         ranges[count++] = (struct agx3_ioto_uniform_range){
            .start_halfs = AGX_ABI_VUNI_FIRST_VERTEX(nr_attribs),
            .size_halfs = 4,
            .address = cmd->state.gfx.draw_params,
         };
      }
      if (cmd->state.gfx.draw_id_ptr) {
         ranges[count++] = (struct agx3_ioto_uniform_range){
            .start_halfs = AGX_ABI_VUNI_DRAW_ID(nr_attribs),
            .size_halfs = 1,
            .address = cmd->state.gfx.draw_id_ptr,
         };
      }

      root_uniform = AGX_ABI_VUNI_COUNT_VK(nr_attribs);
   } else if (s->info.stage == MESA_SHADER_FRAGMENT) {
      root_uniform = AGX_ABI_FUNI_ROOT;
   } else {
      return false;
   }

   if (s->info.agx3_combined_sampler_valid) {
      if (s->info.stage != MESA_SHADER_FRAGMENT || s->info.set_count != 0)
         return false;

      struct hk_descriptor_set *set = cmd->state.gfx.descriptors.sets[0];
      const struct agx_g17p_combined_sampler_binding *binding =
         &s->info.agx3_combined_sampler;
      if (!set || !set->mapped_ptr ||
          binding->texture_offset_B > set->size ||
          set->size - binding->texture_offset_B < 32 ||
          binding->sampler_index_offset_B > set->size ||
          set->size - binding->sampler_index_offset_B < sizeof(uint16_t))
         return false;

      uint16_t sampler_index;
      memcpy(&sampler_index,
             (const uint8_t *)set->mapped_ptr +
                binding->sampler_index_offset_B,
             sizeof(sampler_index));

      const unsigned sampler_element_size =
         agx_chip_sampler_heap_element_size(dev->dev.chip);
      if (!dev->samplers.table.bo || !dev->samplers.table.bo->va ||
          dev->samplers.table.alloc > UINT16_MAX ||
          sampler_element_size > UINT16_MAX)
         return false;

      const struct hk_agx3_g17p_combined_sampler_heap heap = {
         .descriptor_set_va = hk_descriptor_set_addr(set),
         .sampler_heap_va = dev->samplers.table.bo->va->addr,
         .texture_offset_B = binding->texture_offset_B,
         .sampler_index = sampler_index,
         .sampler_count = dev->samplers.table.alloc,
         .sampler_element_size_B = sampler_element_size,
      };
      uint64_t pointers[2];
      if (!hk_agx3_g17p_combined_sampler_pointers(&heap, pointers))
         return false;

      /* HAL200 FragmentProgramVariant::finalize (374968..3749c0) maps
       * compiler bindpoints 0x31/0x35 to driver qwords +0xc0/+0xe0.
       * encodeAndEmitRenderState writes the raw texture/sampler table VAs
       * there (4c5580/4c5598); ESL copies them to uniform dwords 0..3.
       * Keep this upload immutable for the command-buffer lifetime, just
       * like the per-draw graphics state. Both shader indices are zero.
       */
      const uint64_t pointers_va =
         hk_pool_upload(cmd, pointers, sizeof(pointers), 8);
      if (!pointers_va)
         return false;

      ranges[count++] = (struct agx3_ioto_uniform_range){
         .start_halfs = 0,
         .size_halfs = 8,
         .address = pointers_va,
      };
   }

   if (s->info.agx3_indexed_sampler_valid) {
      if (s->info.stage != MESA_SHADER_FRAGMENT ||
          !dev->sampled_images.bo || !dev->samplers.table.bo)
         return false;
      const uint64_t pointers[2] = {
         dev->sampled_images.bo->va->addr,
         dev->samplers.table.bo->va->addr,
      };
      const uint64_t pointers_va = hk_pool_upload(cmd, pointers, sizeof(pointers), 8);
      if (!pointers_va)
         return false;
      ranges[count++] = (struct agx3_ioto_uniform_range){
         .start_halfs = 0, .size_halfs = 8, .address = pointers_va,
      };
   }

   if (s->info.agx3_uses_root || s->info.set_count != 0) {
      if (!root_ptr)
         return false;

      if (!hk_agx3_g17p_root_range(root_uniform, s->info.set_count, root_ptr,
                                   &ranges[count]))
         return false;
      count++;
   }

   /*
    * Promoted constants. agx_opt_promote_constants() parks a shader's
    * constant pool in the binary's rodata section and addresses it through a
    * uniform, so the driver has to publish that address as a uniform range --
    * exactly as the compute path does (hk_cmd_dispatch.c). Without it a
    * fragment shader as simple as a constant colour is refused, because the
    * compiler promotes its three floats.
    *
    * The data is read through the uniform, not the program counter, so it is
    * addressed in the shader's own BO even though graphics executes from the
    * linked binary. Same bounds checks as compute: the range must lie inside
    * that BO, and both halves must be even.
    */
   if (count < HK_AGX3_G17P_GRAPHICS_MAX_RANGES && s->b.info.rodata.size_16 != 0) {
      const struct agx_rodata *rodata = &s->b.info.rodata;
      const uint64_t size_B = (uint64_t)rodata->size_16 * 2;

      if (!s->bo || !s->bo->va || (rodata->base_uniform & 1) != 0 ||
          (rodata->size_16 & 1) != 0 || rodata->size_16 > 512 ||
          rodata->offset > s->bo->size ||
          size_B > s->bo->size - rodata->offset)
         return false;

      ranges[count++] = (struct agx3_ioto_uniform_range){
         .start_halfs = rodata->base_uniform,
         .size_halfs = rodata->size_16,
         .address = s->bo->va->addr + rodata->offset,
      };
   } else if (s->b.info.rodata.size_16 != 0) {
      /* No room left in the program's range budget. */
      return false;
   }

   *range_count = count;
   return count <= HK_AGX3_G17P_GRAPHICS_MAX_RANGES;
}

uint64_t
hk_upload_usc_words(struct hk_cmd_buffer *cmd, struct hk_shader *s,
                    struct hk_linked_shader *linked)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   if (agx_chip_command_stream_generation(dev->dev.chip) ==
       AGX_CS_GENERATION_HAL200) {
      if (dev->dev.chip != AGX_CHIP_G17P ||
          !hk_g17p_basic_graphics_program_supported(cmd, s, linked)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return 0;
      }

      struct agx3_ioto_uniform_range ranges[HK_AGX3_G17P_GRAPHICS_MAX_RANGES];
      size_t range_count;
      if (!hk_g17p_graphics_uniform_ranges(cmd, s, ranges, &range_count)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return 0;
      }

      size_t program_capacity;
      if (!hk_agx3_g17p_graphics_program_bound(dev->dev.chip, ranges,
                                              range_count, &program_capacity)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return 0;
      }

      /* The Ioto program embeds the linked executable address, and promoted
       * constants embed the shader BO address. Pipelines may be destroyed
       * after recording, so retain both until this command buffer is reset.
       */
      hk_cmd_buffer_retain_bo(cmd, linked->b.bo);
      if (s->b.info.rodata.size_16 != 0)
         hk_cmd_buffer_retain_bo(cmd, s->bo);

      struct agx_bo *program_bo =
         agx_bo_create(&dev->dev, program_capacity,
                       AGX_IOTO_PROGRAM_ALIGNMENT, AGX_BO_EXEC | AGX_BO_LOW_VA,
                       "G17P Ioto program");
      if (!program_bo) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         return 0;
      }
      util_dynarray_append(&cmd->large_bos, program_bo);

      void *program_map = agx_bo_map(program_bo);
      if (!program_map) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_MEMORY_MAP_FAILED);
         return 0;
      }

      /* Same fail-closed USC-window containment check as the compute path in
       * hk_cmd_dispatch.c: usc_exec_base is fixed per queue, so an exec
       * pointer outside [shader_base, +4 GiB) can only end as an unattributed
       * GPU page fault.
       */
      if (!agx_va_in_usc_window(&dev->dev, program_bo->va->addr) ||
          !agx_va_in_usc_window(&dev->dev,
                             linked->b.bo->va->addr + linked->b.entry_offset)) {
         mesa_loge("G17P graphics: exec pointer outside the USC window "
                   "[%" PRIx64 ", %" PRIx64 "): program=%" PRIx64
                   " shader=%" PRIx64 "\n",
                   dev->dev.shader_base,
                   dev->dev.shader_base + AGX_USC_WINDOW_SIZE,
                   program_bo->va->addr,
                linked->b.bo->va->addr + linked->b.entry_offset);
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_UNKNOWN);
         return 0;
      }

      const struct hk_agx3_g17p_graphics_program_desc desc = {
         .program_va = program_bo->va->addr,
         /* The container's entry point, not the allocation base: the
       * shader loader reads the block-size dword at entry - 0x80. */
      .shader_va = linked->b.bo->va->addr + linked->b.entry_offset,
         .uniform_ranges = ranges,
         .uniform_range_count = range_count,
         .shader_binary_size = linked->b.size,
         .scratch_size = linked->b.scratch_size,
         .nr_gprs = linked->b.nr_gprs,
         .compiler_d_valid = linked->b.g17p_compiler_d_valid,
         .compiler_d = linked->b.g17p_compiler_d,
         .fragment = s->info.stage == MESA_SHADER_FRAGMENT,
         .writes_sample_mask = s->info.stage == MESA_SHADER_FRAGMENT &&
            ((linked->b.fragment_control.opaque[0] >> 29) == AGX_PASS_TYPE_PUNCH_THROUGH ||
             (linked->b.fragment_control.opaque[0] >> 29) == AGX_PASS_TYPE_TRANSLUCENT_PUNCH_THROUGH),
      };
      struct hk_agx3_g17p_graphics_program program;
      if (!hk_agx3_g17p_basic_graphics_program_generate(dev->dev.chip, &desc,
                                                        &program)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return 0;
      }

      memcpy(program_map, program.bytes, program.length);
      if (hk_g17p_launch_trace_enabled()) {
         const char *label = s->info.stage == MESA_SHADER_FRAGMENT
                                ? "fragment-ioto"
                                : "vertex-ioto";
         hk_g17p_trace_line(label, program_bo->va->addr, program_map,
                            program.length);
      }
      if (agx_bo_sync_for_device(program_bo, 0, program.length)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_DEVICE_LOST);
         return 0;
      }
      return program_bo->va->addr;
   }

   if (agx_chip_command_stream_generation(dev->dev.chip) !=
       AGX_CS_GENERATION_AGX2) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return 0;
   }

   mesa_shader_stage sw_stage = s->info.stage;

   unsigned constant_push_ranges = DIV_ROUND_UP(s->b.info.rodata.size_16, 64);
   unsigned push_ranges = 2;
   unsigned stage_ranges = 3;

   size_t usc_size =
      agx_usc_size(constant_push_ranges + push_ranges + stage_ranges + 4);
   struct agx_ptr t = hk_pool_usc_alloc(cmd, usc_size, 64);
   if (!t.cpu)
      return 0;

   struct agx_usc_builder b = agx_usc_builder(t.cpu, usc_size);

   uint64_t root_ptr;

   if (sw_stage == MESA_SHADER_COMPUTE) {
      root_ptr = hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_COMPUTE);
   } else {
      root_ptr = cmd->state.gfx.root;
   }

   static_assert(offsetof(struct hk_root_descriptor_table, root_desc_addr) == 0,
                 "self-reflective");

   unsigned root_unif = 0;
   if (sw_stage == MESA_SHADER_VERTEX) {
      unsigned count =
         DIV_ROUND_UP(BITSET_LAST_BIT(s->info.vs.attrib_components_read), 4);

      if (count) {
         agx_usc_uniform(
            &b, AGX_ABI_VUNI_VBO_BASE(0), 4 * count,
            root_ptr + hk_root_descriptor_offset(draw.attrib_base));

         agx_usc_uniform(
            &b, AGX_ABI_VUNI_VBO_CLAMP(count, 0), 2 * count,
            root_ptr + hk_root_descriptor_offset(draw.attrib_clamps));
      }

      if (cmd->state.gfx.draw_params) {
         agx_usc_uniform(&b, AGX_ABI_VUNI_FIRST_VERTEX(count), 4,
                         cmd->state.gfx.draw_params);
      }

      if (cmd->state.gfx.draw_id_ptr) {
         agx_usc_uniform(&b, AGX_ABI_VUNI_DRAW_ID(count), 1,
                         cmd->state.gfx.draw_id_ptr);
      }

      if (linked->sw_indexing) {
         agx_usc_uniform(
            &b, AGX_ABI_VUNI_VERTEX_PARAMS(count), 4,
            root_ptr + hk_root_descriptor_offset(draw.vertex_params));
      }

      root_unif = AGX_ABI_VUNI_COUNT_VK(count);
   } else if (sw_stage == MESA_SHADER_FRAGMENT) {
      if (agx_tilebuffer_spills(&cmd->state.gfx.render.tilebuffer)) {
         hk_usc_upload_spilled_rt_descs(&b, cmd);
      }

      if (cmd->state.gfx.uses_blend_constant) {
         agx_usc_uniform(
            &b, 4, 8,
            root_ptr + hk_root_descriptor_offset(draw.blend_constant));
      }

      /* The SHARED state is baked into linked->usc for non-fragment shaders. We
       * don't pass around the information to bake the tilebuffer layout.
       *
       * TODO: We probably could with some refactor.
       */
      agx_usc_push_packed(&b, SHARED, &cmd->state.gfx.render.tilebuffer.usc);
      root_unif = AGX_ABI_FUNI_ROOT;
   }

   /* Address for the root and each set */
   agx_usc_uniform(&b, root_unif, 4 * (1 + s->info.set_count), root_ptr);

   agx_usc_push_blob(&b, linked->usc.data, linked->usc.size);
   return agx_usc_addr(&dev->dev, t.gpu);
}

void
hk_dispatch_precomp(struct hk_cmd_buffer *cmd, struct agx_grid grid,
                    enum agx_barrier barrier, enum libagx_program idx,
                    void *data, size_t data_size)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct agx_precompiled_shader *prog = agx_get_precompiled(&dev->bg_eot, idx);
   if (!prog) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }

   struct hk_cs **target = (barrier & AGX_POSTGFX)  ? &cmd->current_cs.post_gfx
                           : (barrier & AGX_PREGFX) ? &cmd->current_cs.pre_gfx
                                                    : &cmd->current_cs.cs;

   struct hk_cs *cs = hk_cmd_buffer_get_cs_general(cmd, target, true);
   if (!cs)
      return;

   struct agx_ptr t = hk_pool_usc_alloc(cmd, agx_usc_size(15), 64);
   uint64_t uploaded_data = hk_pool_upload(cmd, data, data_size, 4);

   agx_usc_words_precomp(t.cpu, &prog->b, uploaded_data, data_size);

   hk_dispatch_with_usc_launch(dev, cs, prog->b.launch,
                               agx_usc_addr(&dev->dev, t.gpu), grid,
                               prog->b.workgroup);
}

void
hk_cs_init_graphics(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_rendering_state *render = &cmd->state.gfx.render;
   uint8_t *map = cs->current;

   /* Even a clear-only stream needs a stable root alias before EndCommandBuffer.
    * Linked chunks acquire theirs in hk_cmd_buffer_emit_cs_jump. */
   if (hk_g17p_vdm_low_window_enabled(&dev->dev) &&
       !hk_g17p_vdm_compact(cmd, cs->addr))
      return;

   hk_drm_attachment_set_init(&cs->fragment_attachments);

   /* Retain the image allocation spans now. A command buffer may contain
    * several VDM commands with different render state, while queue submission
    * happens only after all of them have been recorded. */
   struct hk_attachment_view_ref {
      struct hk_image_view *view;
      VkImageAspectFlags aspect;
   } views[(2 * HK_MAX_RTS) + 4];
   static_assert(ARRAY_SIZE(views) == HK_DRM_MAX_FRAGMENT_ATTACHMENTS,
                 "fragment attachment capture capacity");
   unsigned view_count = 0;

   for (unsigned i = 0; i < render->color_att_count; ++i) {
      views[view_count++] = (struct hk_attachment_view_ref){
         .view = render->color_att[i].iview,
         .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
      };
      views[view_count++] = (struct hk_attachment_view_ref){
         .view = render->color_att[i].resolve_iview,
         .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
      };
   }

   views[view_count++] = (struct hk_attachment_view_ref){
      .view = render->depth_att.iview,
      .aspect = VK_IMAGE_ASPECT_DEPTH_BIT,
   };
   views[view_count++] = (struct hk_attachment_view_ref){
      .view = render->depth_att.resolve_iview,
      .aspect = VK_IMAGE_ASPECT_DEPTH_BIT,
   };
   views[view_count++] = (struct hk_attachment_view_ref){
      .view = render->stencil_att.iview,
      .aspect = VK_IMAGE_ASPECT_STENCIL_BIT,
   };
   views[view_count++] = (struct hk_attachment_view_ref){
      .view = render->stencil_att.resolve_iview,
      .aspect = VK_IMAGE_ASPECT_STENCIL_BIT,
   };

   assert(view_count <= ARRAY_SIZE(views));
   for (unsigned i = 0; i < view_count; ++i) {
      struct hk_image_view *view = views[i].view;
      if (!view)
         continue;

      struct hk_image *image =
         container_of(view->vk.image, struct hk_image, vk);
      const uint8_t image_plane =
         views[i].aspect == VK_IMAGE_ASPECT_COLOR_BIT
            ? view->planes[0].image_plane
            : hk_image_aspects_to_plane(image, views[i].aspect);
      if (image_plane >= image->plane_count) {
         cs->fragment_attachments.complete = false;
         continue;
      }

      const struct hk_image_plane *plane = &image->planes[image_plane];
      const uint64_t size = plane->layout.size_B;
      if (!plane->addr || !size) {
         cs->fragment_attachments.complete = false;
         continue;
      }

      hk_drm_attachment_set_add(&cs->fragment_attachments, plane->addr, size);
   }

   if (dev->dev.chip == AGX_CHIP_G17P) {
      /* The kext preserves a zero final VDM word when this selector is the
       * UINT32_MAX sentinel.  This reproduces m1n1's direct-triangle prefix. */
      cs->agx3_active_context_selector_valid = true;
      cs->agx3_raw_active_context_selector = UINT32_MAX;
   }

   cs->tib = render->tilebuffer;

   /* Assume this is not the first control stream of the render pass, so
    * initially use the partial background/EOT program and ZLS control.
    * hk_BeginRendering/hk_EndRendering will override.
    */
   cs->cr = render->cr;
   cs->cr.bg.main = render->cr.bg.partial;
   cs->cr.eot.main = render->cr.eot.partial;
   cs->cr.zls_control = render->cr.zls_control_partial;

   /* Barrier to enforce GPU-CPU coherency, in case this batch is back to back
    * with another that caused stale data to be cached and the CPU wrote to it
    * in the meantime.
    */
   switch (agx_chip_command_stream_generation(dev->dev.chip)) {
   case AGX_CS_GENERATION_AGX2:
      agx_push(map, VDM_BARRIER, cfg) {
         cfg.usc_cache_inval = true;
      }
      break;

   case AGX_CS_GENERATION_G15:
   case AGX_CS_GENERATION_HAL200:
   case AGX_CS_GENERATION_HAL300: {
      struct agx_vdm_barrier_core barrier;
      if (!agx_vdm_barrier_core_pack(dev->dev.chip, &barrier)) {
         mesa_loge("no grounded AGX3 initial VDM barrier for %s",
                   agx_chip_name(dev->dev.chip));
         vk_command_buffer_set_error(&cmd->vk,
                                     VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }

      memcpy(map, barrier.bytes, barrier.length);
      map += barrier.length;
      break;
   }

   case AGX_CS_GENERATION_UNKNOWN:
   default:
      mesa_loge("unknown VDM command-stream generation for %s",
                agx_chip_name(dev->dev.chip));
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }

   if (agx_chip_command_stream_generation(dev->dev.chip) ==
       AGX_CS_GENERATION_HAL300) {
      struct agx_ppp_w_clamp_cull2_core initial;
      const struct agx_ppp_w_clamp_cull2_fields fields = {
         /* Exact G17G/G17X default: W clamp 0.0, clamp-W enabled. */
         .w_clamp_raw = 0,
         .cull2_raw = UINT32_C(0x80),
      };
      struct agx_ppp_hal300_depth_stencil_core depth_stencil;
      const struct agx_ppp_hal300_depth_stencil_fields depth_stencil_fields = {
         /* Exact G17G/G17X fixed-layout initial render-state witness. */
         .fragment_control_raw = UINT32_C(0x000007aa),
         .fragment_front_face_raw = UINT32_C(0xe2e00000),
         .fragment_back_face_raw = UINT32_C(0xf0c00000),
         .fragment_front_stencil_raw = UINT32_C(0x00023020),
         .fragment_back_stencil_raw = UINT32_C(0x0001ffff),
      };
      if (!agx_ppp_hal300_depth_stencil_core_pack(
             dev->dev.chip, &depth_stencil_fields, &depth_stencil) ||
          !agx_ppp_w_clamp_cull2_core_pack(dev->dev.chip, &fields,
                                            &initial)) {
         mesa_loge("no grounded HAL300 initial PPP payload for %s",
                   agx_chip_name(dev->dev.chip));
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }

      const uint64_t depth_stencil_gpu = hk_pool_upload(
         cmd, depth_stencil.bytes, depth_stencil.length,
         AGX_PPP_HEADER_ALIGN);
      const uint64_t gpu = hk_pool_upload(
         cmd, initial.bytes, initial.length, AGX_PPP_HEADER_ALIGN);
      uint8_t records[2 * AGX_PPP_STATE_LENGTH];
      uint8_t *record = records;
      if (!depth_stencil_gpu || !gpu ||
          !agx_ppp_emit_state(dev->dev.chip, &record, depth_stencil_gpu,
                              depth_stencil.length) ||
          !agx_ppp_emit_state(dev->dev.chip, &record, gpu, initial.length) ||
          record != records + sizeof(records)) {
         mesa_loge("failed to emit HAL300 initial PPP state for %s",
                   agx_chip_name(dev->dev.chip));
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         return;
      }

      memcpy(map, records, sizeof(records));
      map += sizeof(records);
   } else if (dev->dev.chip == AGX_CHIP_G17P) {
      /* PPPEncoderGen6's inspected G17P fixed-layout opening is one exact
       * W-clamp/Cull-2 payload, not the legacy AGX2 selector graph below.
       * Keep the payload in the compact VDM window just like every other
       * nested HAL200 PPP pointer.
       */
      struct agx_ppp_w_clamp_cull2_core initial;
      const struct agx_ppp_w_clamp_cull2_fields fields = {
         .w_clamp_raw = 0,
         .cull2_raw = UINT32_C(0x80),
      };

      if (!agx_ppp_w_clamp_cull2_core_pack(dev->dev.chip, &fields,
                                            &initial)) {
         mesa_loge("no grounded G17P initial PPP payload for %s",
                   agx_chip_name(dev->dev.chip));
         vk_command_buffer_set_error(&cmd->vk,
                                     VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }

      struct agx_ptr T =
         hk_pool_vdm_alloc(cmd, initial.length, AGX_PPP_HEADER_ALIGN);
      if (!T.cpu)
         return;

      memcpy(T.cpu, initial.bytes, initial.length);
      if (!agx_ppp_emit_state(dev->dev.chip, &map, T.gpu, initial.length)) {
         mesa_loge("failed to emit G17P initial PPP state for %s",
                   agx_chip_name(dev->dev.chip));
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         return;
      }

      if (hk_g17p_launch_trace_enabled())
         hk_g17p_trace_line("ppp-initial", T.gpu, T.cpu, initial.length);
   } else {
      struct AGX_PPP_HEADER present = {
         .w_clamp = true,
         .occlusion_query_2 = true,
         .output_unknown = true,
         .varying_word_2 = true,
         .viewport_count = 1, /* irrelevant */
      };

      size_t size = agx_ppp_update_size(&present);
      struct agx_ptr T = hk_pool_vdm_alloc(cmd, size, AGX_PPP_HEADER_ALIGN);
      if (!T.cpu)
         return;

      struct agx_ppp_update ppp = agx_new_ppp_update(T, size, &present);

      /* clang-format off */
      agx_ppp_push(&ppp, W_CLAMP, cfg) cfg.w_clamp = 1e-10;
      agx_ppp_push(&ppp, FRAGMENT_OCCLUSION_QUERY_2, cfg);
      agx_ppp_push(&ppp, OUTPUT_UNKNOWN, cfg);
      agx_ppp_push(&ppp, VARYING_2, cfg);
      /* clang-format on */

      const bool ppp_ok = agx_ppp_legacy_payload_supported(dev->dev.chip) &&
                          agx_ppp_fini(dev->dev.chip, &map, &ppp);
      if (!ppp_ok) {
         mesa_loge("no grounded legacy PPP payload/finalizer for %s",
                   agx_chip_name(dev->dev.chip));
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }

      if (hk_g17p_launch_trace_enabled())
         hk_g17p_trace_line("ppp-initial", ppp.gpu_base, ppp.cpu_base,
                            ppp.total_size);
   }
   cs->current = map;

   cs->scissor = UTIL_DYNARRAY_INIT;
   cs->depth_bias = UTIL_DYNARRAY_INIT;

   /* All graphics state must be reemited in each control stream */
   hk_cmd_buffer_dirty_all(cmd);

   if (dev->dev.chip == AGX_CHIP_G17P)
      assert((cs->current - cs->start) == HK_G17P_VDM_INITIAL_SIZE);
}

void
hk_ensure_cs_has_space(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                       size_t space)
{
   bool vdm = cs->type == HK_CS_VDM;

   size_t link_length =
      vdm ? AGX_VDM_STREAM_LINK_LENGTH : AGX_CDM_STREAM_LINK_LENGTH;

   /* Assert that we have space for a link tag */
   assert((cs->current + link_length) <= cs->end && "Encoder overflowed");

   /* Always leave room for a link tag, in case we run out of space later,
    * plus padding because VDM apparently overreads?
    *
    * 0x200 is not enough. 0x400 seems to work. 0x800 for safety.
    */
   /* If there is room in the command buffer, we're done */
   if (likely(hk_cs_space_fits(cs->end - cs->current, space, link_length)))
      return;

   /* Otherwise, we need to allocate a new command buffer. We use memory owned
    * by the batch to simplify lifetime management for the BO.
    */
   size_t size = 65536;
   struct agx_ptr T = hk_pool_alloc(cmd, size, 256);

   /* Jump from the old control stream to the new control stream. */
   void *link_end =
      hk_cmd_buffer_emit_cs_jump(cmd, cs->current, T.gpu, vdm);
   const bool linked = link_end != cs->current;

   /* Swap out the control stream */
   cs->current = T.cpu;
   cs->end = cs->current + size;
   cs->chunk = T;
   cs->stream_linked |= linked;
}

static void
clear_attachment_as_image(struct hk_cmd_buffer *cmd,
                          struct hk_rendering_state *render,
                          struct hk_attachment *att, unsigned aspect)
{
   struct hk_image_view *view = att->iview;
   if (!att->clear || !view || !(view->vk.aspects & aspect))
      return;

   const uint32_t layer_count = render->view_mask
                                   ? util_last_bit(render->view_mask)
                                   : render->layer_count;

   const VkImageSubresourceRange range = {
      .layerCount = layer_count,
      .levelCount = 1,
      .baseArrayLayer = view->vk.base_array_layer,
      .baseMipLevel = view->vk.base_mip_level,
      .aspectMask = view->vk.aspects & aspect,
   };

   assert(util_bitcount(range.aspectMask) == 1);
   VkFormat format = att->vk_format;

   if (aspect == VK_IMAGE_ASPECT_DEPTH_BIT) {
      format = vk_format_depth_only(format);
   } else if (aspect == VK_IMAGE_ASPECT_STENCIL_BIT) {
      format = vk_format_stencil_only(format);
   }

   struct hk_image *image = container_of(view->vk.image, struct hk_image, vk);
   hk_clear_image(cmd, image, vk_format_to_pipe_format(format),
                  att->clear_colour, &range, false /* partially clear 3D */);
}

bool
hk_optimize_empty_vdm(struct hk_cmd_buffer *cmd)
{
   struct hk_cs *cs = cmd->current_cs.gfx;
   struct hk_rendering_state *render = &cmd->state.gfx.render;
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   /* G17P already has native BG/EOT programs for the supported render-target
    * clear path, but no precompiled fill_uint4 for the compute replacement
    * below. Do not discard the working graphics representation and record an
    * unsupported compute command instead. Once that precompiled shader is
    * available, the normal optimization becomes eligible again.
    *
    * BeginRendering retained the attachments, initialized TA/PPP and selected
    * the BG clear; EndRendering selected the final EOT. process_empty_tiles
    * is the existing clear-and-store predicate, not a fabricated draw. The
    * common finalizer will append the native VDM terminator without uploading
    * any ordinary vertex or fragment program.
    */
   if (dev->dev.chip == AGX_CHIP_G17P && cs->cr.process_empty_tiles &&
       !agx_get_precompiled(&dev->bg_eot, LIBAGX_FILL_UINT4)) {
      /* Normally supplied by the first draw. A clear has no application
       * sample-location state; use the attachment sample count and leave
       * has_sample_locations uncommitted, just like other meta operations.
       */
      cs->ppp_multisamplectl =
         hk_default_sample_positions(MAX2(cs->tib.nr_samples, 1));

      /* The render submission still binds these arrays even without PPP draw
       * state referencing them. Give it an owned, neutral full-frame entry
       * rather than a zero-byte upload. Render-area handling and empty-tile
       * selection remain in the existing render setup.
       */
      if (cs->scissor.size == 0) {
         uint8_t *scissor =
            util_dynarray_grow_bytes(&cs->scissor, 1, AGX_SCISSOR_LENGTH);
         agx_pack(scissor, SCISSOR, cfg) {
            cfg.min_x = cfg.min_y = 0;
            cfg.max_x = cs->cr.width;
            cfg.max_y = cs->cr.height;
            cfg.min_z = 0.0f;
            cfg.max_z = 1.0f;
         }
      }

      if (cs->depth_bias.size == 0) {
         void *bias = util_dynarray_grow_bytes(&cs->depth_bias, 1,
                                              AGX_DEPTH_BIAS_LENGTH);
         memset(bias, 0, AGX_DEPTH_BIAS_LENGTH);
      }

      if (hk_g17p_launch_trace_enabled()) {
         fprintf(stderr, "G17P clear-only graphics: %ux%u samples=%u "
                         "sample_control=%#x process_empty_tiles=1 "
                         "scissor=%u zbias=%u\n",
                 cs->cr.width, cs->cr.height, MAX2(cs->tib.nr_samples, 1),
                 cs->ppp_multisamplectl, cs->scissor.size,
                 cs->depth_bias.size);
      }
      return false;
   }

   for (unsigned i = 0; i < render->color_att_count; ++i) {
      clear_attachment_as_image(cmd, render, &render->color_att[i], ~0);
   }

   clear_attachment_as_image(cmd, render, &render->depth_att,
                             VK_IMAGE_ASPECT_DEPTH_BIT);

   clear_attachment_as_image(cmd, render, &render->stencil_att,
                             VK_IMAGE_ASPECT_STENCIL_BIT);

   /* Remove the VDM control stream from the command buffer, now that it is
    * replaced by equivalent other operations.
    */
   list_del(&cs->node);
   hk_cs_destroy(cs);
   return true;
}
