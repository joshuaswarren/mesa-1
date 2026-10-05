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
#include "decode.h"
#include "hk_agx3_ioto_program_core.h"
#include "asahi/compiler/agx3_compute_program.h"
#include "asahi/compiler/agx_debug.h"
#include "hk_agx3_vdm_decode.h"
#include "asahi/lib/agx_ioto_load_core.h"
#include "hk_buffer.h"
#include "hk_cmd_pool.h"
#include "hk_descriptor_set.h"
#include "hk_descriptor_set_layout.h"
#include "hk_device.h"
#include "hk_device_memory.h"
#include "hk_entrypoints.h"
#include "hk_g17p_apple_override.h"
#include "hk_image.h"
#include "hk_image_view.h"
#include "hk_physical_device.h"
#include "hk_shader.h"
#include "libagx_shaders.h"

#include "util/cache_ops.h"
#include "util/log.h"
#include "util/hash_table.h"
#include "util/u_debug.h"
#include "util/os_time.h"

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

   struct hk_vdm_allocation *last = cmd->g17p_vdm_last_allocation;
   if (last && primary >= last->addr && primary - last->addr < last->size) {
      uint32_t compact = 0;
      if (agx_g17p_vdm_alias_compact(
             last->alias + (primary - last->addr), &compact))
         return compact;
   }

   /* Allocations never overlap, so the search order cannot change the
    * answer. New control streams live in the most recent allocation: search
    * from the tail so the lookup at each stream start stays constant time
    * instead of growing with every render pass recorded. */
   list_for_each_entry_rev(struct hk_vdm_allocation, allocation,
                           &cmd->vdm_allocations, link) {
      if (primary < allocation->addr ||
          primary - allocation->addr >= allocation->size)
         continue;

      const uint64_t alias =
         allocation->alias + (primary - allocation->addr);
      uint32_t compact = 0;
      if (agx_g17p_vdm_alias_compact(alias, &compact)) {
         /* Recording owns the command buffer, but submission may translate a
          * SIMULTANEOUS_USE buffer from several queues at once.  The latter
          * must remain read-only: it may consume the stable last recording
          * span above, never race to replace it here. */
         if (bind)
            cmd->g17p_vdm_last_allocation = allocation;
         return compact;
      }
   }

   /* The constrained G17P main heap is wholly inside the lower-root compact
    * VDM aperture. Control-stream roots and links can therefore name their
    * primary mapping directly. Keep the scarce +8..+32 MiB aliases above for
    * draw-state payloads whose independent HAL200 fields are narrower. */
   uint64_t direct = 0;
   if (agx_g17p_vdm_payload_compact(primary, &direct))
      return direct;

   {
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

   /* The cached span is only valid while this recording owns its leases. */
   cmd->g17p_vdm_last_allocation = NULL;

   hk_descriptor_state_fini(cmd, &cmd->state.gfx.descriptors);
   hk_descriptor_state_fini(cmd, &cmd->state.cs.descriptors);
   hk_descriptor_state_fini(cmd, &cmd->state.rt.descriptors);

   if (hk_g17p_upload_quarantine_enabled(dev)) {
      /* Diagnostic lifetime falsifier: neither ordinary/USC upload BOs nor
       * compact VDM slab spans may be reused after reset. Their mappings and
       * backing storage remain live until device destruction. */
      hk_g17p_quarantine_command_uploads(
         dev, &cmd->uploader.main.bos, &cmd->uploader.usc.bos,
         &cmd->uploader.root.bos, &cmd->vdm_allocations);
   } else {
      hk_cmd_pool_free_bo_list(pool, HK_CMD_BO_MAIN,
                               &cmd->uploader.main.bos);
      hk_cmd_pool_free_bo_list(pool, HK_CMD_BO_USC,
                               &cmd->uploader.usc.bos);
      hk_cmd_pool_free_bo_list(pool, HK_CMD_BO_ROOT,
                               &cmd->uploader.root.bos);
      list_for_each_entry_safe(struct hk_vdm_allocation, allocation,
                               &cmd->vdm_allocations, link) {
         list_del(&allocation->link);
         hk_cmd_pool_free_vdm(pool, allocation);
      }
   }

   list_for_each_entry_safe(struct hk_cs, it, &cmd->control_streams, node) {
      list_del(&it->node);
      hk_cs_destroy(it);
   }

   hk_ioto_arena_finish(cmd);
   util_dynarray_clear(&cmd->g17p_compute_programs);
   hk_g17p_precomp_lookup_stats_finish(cmd);
   _mesa_hash_table_clear(cmd->g17p_precomp_index, NULL);
   util_dynarray_clear(&cmd->g17p_precomp_programs);
   hk_cmd_buffer_release_bos(cmd, &dev->dev);
}

static void
hk_destroy_cmd_buffer(struct vk_command_buffer *vk_cmd_buffer)
{
   struct hk_cmd_buffer *cmd =
      container_of(vk_cmd_buffer, struct hk_cmd_buffer, vk);
   struct hk_cmd_pool *pool = hk_cmd_buffer_pool(cmd);

   hk_free_resettable_cmd_buffer(cmd);
   util_dynarray_fini(&cmd->g17p_compute_programs);
   _mesa_hash_table_destroy(cmd->g17p_precomp_index, NULL);
   util_dynarray_fini(&cmd->g17p_precomp_programs);
   util_dynarray_fini(&cmd->large_bos);
   free(cmd->retained_set);
   util_dynarray_fini(&cmd->sync_pass_records);
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
   cmd->g17p_compute_programs = UTIL_DYNARRAY_INIT;
   cmd->g17p_precomp_programs = UTIL_DYNARRAY_INIT;

   cmd->vk.dynamic_graphics_state.vi = &cmd->state.gfx._dynamic_vi;
   cmd->vk.dynamic_graphics_state.ms.sample_locations =
      &cmd->state.gfx._dynamic_sl;

   list_inithead(&cmd->uploader.main.bos);
   list_inithead(&cmd->uploader.usc.bos);
   list_inithead(&cmd->uploader.root.bos);
   list_inithead(&cmd->vdm_allocations);
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
   cmd->uploader.root.map = NULL;
   cmd->uploader.root.base = 0;
   cmd->uploader.root.offset = 0;
   cmd->uploader.root.capacity = 0;
   cmd->uploader.vdm.map = NULL;
   cmd->uploader.vdm.base = 0;
   cmd->uploader.vdm.offset = 0;
   cmd->uploader.vdm.capacity = 0;
   cmd->uploader.streams.map = NULL;
   cmd->uploader.streams.base = 0;
   cmd->uploader.streams.offset = 0;

   cmd->current_cs.gfx = NULL;
   cmd->current_cs.cs = NULL;
   cmd->current_cs.post_gfx = NULL;
   cmd->current_cs.pre_gfx = NULL;

   assert(!cmd->in_meta);
   assert(!cmd->conditional_meta);
   cmd->geom_index_buffer = 0;
   cmd->geom_index_count = 0;
   cmd->geom_instance_count = 0;
   cmd->uses_heap = false;
   cmd->g17p_trace_draws = 0;
   hk_sync_reset(cmd);

   memset(&cmd->state, 0, sizeof(cmd->state));
   for (unsigned i = 0; i < HK_G17P_BG_EOT_CACHE_ENTRIES; ++i)
      cmd->g17p_bg_eot_cache[i].valid = false;
   cmd->g17p_bg_eot_cache_next = 0;
   /* v94 bit1: nothing has been recorded since this reset. */
   cmd->g17p_pristine = true;
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

   const struct vk_device_dispatch_table *disp =
      cmd->vk.base.device->command_dispatch_table;
   const struct hk_conditional_render_state primary_conditional =
      cmd->state.conditional;

   /* Secondary command buffers are materialized by replaying their common
    * command queues into the primary.  Their last bound state is therefore
    * also the driver's cached state in the primary after replay.  Vulkan
    * makes that state undefined (except for the active render pass), so an
    * application is required to bind it again.  Invalidate here so binding
    * the same object used by the secondary cannot be incorrectly elided.
    */
   for (uint32_t i = 0; i < commandBufferCount; ++i) {
      VK_FROM_HANDLE(hk_cmd_buffer, secondary, pCommandBuffers[i]);

      /* The queued commands replay into `cmd`, so establish the conditional
       * state promised by the secondary's inheritance declaration. Explicit
       * begin/end scopes recorded inside the secondary may change this state
       * during replay; always restore the primary at the command-buffer
       * boundary. */
      cmd->state.conditional = hk_conditional_render_replay_state(
         &primary_conditional, secondary->state.conditional.inherited);
      vk_cmd_queue_execute(&secondary->vk.cmd_queue, commandBuffer, disp);
      cmd->state.conditional = primary_conditional;
   }

   hk_cmd_invalidate_graphics_state(cmd);
   hk_cmd_invalidate_compute_state(cmd);
}

static VkResult
hk_cmd_buffer_alloc_bo(struct hk_cmd_buffer *cmd, enum hk_cmd_bo_type type,
                       struct hk_cmd_bo **bo_out)
{
   struct hk_cmd_pool *pool = hk_cmd_buffer_pool(cmd);
   const bool perf = cmd->shader_perf_draw_active;
   const bool create = perf && list_is_empty(&pool->free_bos[type]);
   const uint64_t start = perf ? os_time_get_nano() : 0;
   VkResult result = hk_cmd_pool_alloc_bo(pool, type, bo_out);
   if (perf) {
      cmd->shader_perf_draw_bo_allocs++;
      cmd->shader_perf_draw_bo_creates += create;
      cmd->shader_perf_draw_bo_alloc_ns += os_time_get_nano() - start;
   }
   if (result != VK_SUCCESS)
      return result;

   if (type == HK_CMD_BO_USC)
      list_addtail(&(*bo_out)->link, &cmd->uploader.usc.bos);
   else if (type == HK_CMD_BO_ROOT)
      list_addtail(&(*bo_out)->link, &cmd->uploader.root.bos);
   else
      list_addtail(&(*bo_out)->link, &cmd->uploader.main.bos);

   return VK_SUCCESS;
}

static __attribute__((hot)) struct agx_ptr
hk_pool_alloc_from(struct hk_cmd_buffer *cmd, struct hk_uploader *uploader,
                   uint32_t size, uint32_t alignment,
                   enum hk_cmd_bo_type type)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   /* Specially handle large allocations owned by the command buffer, e.g. used
    * for statically allocated vertex output buffers with geometry shaders.
    */
   if (size > HK_CMD_BO_SIZE) {
      enum agx_bo_flags flags =
         type == HK_CMD_BO_USC    ? AGX_BO_LOW_VA
         : type == HK_CMD_BO_ROOT ? AGX_BO_WRITEBACK
                                  : 0;
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
   VkResult result = hk_cmd_buffer_alloc_bo(cmd, type, &bo);
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

__attribute__((hot)) struct agx_ptr
hk_pool_alloc_internal(struct hk_cmd_buffer *cmd, uint32_t size,
                       uint32_t alignment, bool usc)
{
   struct hk_uploader *uploader =
      usc ? &cmd->uploader.usc : &cmd->uploader.main;
   return hk_pool_alloc_from(cmd, uploader, size, alignment,
                             usc ? HK_CMD_BO_USC : HK_CMD_BO_MAIN);
}

struct agx_ptr
hk_pool_cs_alloc(struct hk_cmd_buffer *cmd, uint32_t size, uint32_t alignment)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   /* A G17P compute root consumes a whole command BO. Keep control-stream
    * roots on their own bump cursor so allocating one cannot evict a partially
    * used main-upload BO and move that allocator's slow path into the next
    * application command. Ownership remains on the ordinary main BO list. */
   struct hk_uploader *uploader = dev->dev.chip == AGX_CHIP_G17P
                                     ? &cmd->uploader.streams
                                     : &cmd->uploader.main;
   return hk_pool_alloc_from(cmd, uploader, size, alignment, HK_CMD_BO_MAIN);
}

static struct agx_ptr
hk_pool_root_alloc(struct hk_cmd_buffer *cmd, uint32_t size,
                   uint32_t alignment)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (dev->dev.chip != AGX_CHIP_G17P)
      return hk_pool_alloc(cmd, size, alignment);

   return hk_pool_alloc_from(cmd, &cmd->uploader.root, size, alignment,
                             HK_CMD_BO_ROOT);
}

struct agx_ptr
hk_pool_vdm_primary_alloc(struct hk_cmd_buffer *cmd, uint32_t size,
                          uint32_t alignment)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (!hk_g17p_vdm_low_window_enabled(&dev->dev))
      return hk_pool_alloc(cmd, size, alignment);

   struct hk_uploader *uploader = &cmd->uploader.vdm;
   uint32_t offset = align(uploader->offset, alignment);
   if (uploader->map && offset <= uploader->capacity &&
       size <= uploader->capacity - offset) {
      uploader->offset = offset + size;
      return (struct agx_ptr){
         .gpu = uploader->base + offset,
         .cpu = uploader->map + offset,
      };
   }

   const uint32_t chunk_size =
      align(MAX2(size, HK_G17P_VDM_UPLOAD_CHUNK_SIZE),
            HK_G17P_VDM_UPLOAD_CHUNK_SIZE);
   const uint32_t chunk_alignment =
      MAX2(alignment, HK_G17P_VDM_UPLOAD_CHUNK_SIZE);
   struct agx_ptr chunk = {0};
   struct hk_vdm_allocation *allocation = NULL;
   const VkResult result = hk_cmd_pool_alloc_vdm(
      hk_cmd_buffer_pool(cmd), chunk_size, chunk_alignment, &chunk,
      &allocation);
   if (unlikely(result != VK_SUCCESS)) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return (struct agx_ptr){0};
   }

   list_addtail(&allocation->link, &cmd->vdm_allocations);
   cmd->g17p_vdm_last_allocation = allocation;
   uploader->map = chunk.cpu;
   uploader->base = chunk.gpu;
   uploader->offset = size;
   uploader->capacity = chunk_size;
   return chunk;
}

__attribute__((hot)) uint64_t
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
hk_g17p_texture_state_pointer(struct hk_cmd_buffer *cmd, uint64_t texture_va,
                              const void *descriptor, uint32_t available_B)
{
   if (!(texture_va & (HK_G17P_TEXTURE_STATE_ALIGN - 1)))
      return texture_va;

   if (!descriptor || available_B < HK_TEXTURE_DESCRIPTOR_AGX3_SIZE)
      return 0;

   return hk_pool_upload(
      cmd, descriptor,
      MIN2(available_B, (uint32_t)sizeof(struct hk_sampled_image_descriptor)),
      HK_G17P_TEXTURE_STATE_ALIGN);
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
static bool
hk_g17p_push_descriptor_dedicated_bo(void)
{
   static int enabled = -1;
   if (enabled < 0)
      enabled = debug_get_bool_option("HK_G17P_PUSH_DESCRIPTOR_DEDICATED_BO", false);
   return enabled != 0;
}

static bool
hk_g17p_sync_push_descriptors(void)
{
   static int enabled = -1;
   if (enabled < 0)
      enabled = debug_get_bool_option("HK_G17P_SYNC_PUSH_DESCRIPTORS", false);
   return enabled != 0;
}

__attribute__((hot)) bool
hk_g17p_launch_trace_enabled(void)
{
   static int enabled = -1;
   if (enabled < 0)
      enabled = HK_ENV_BOOL_ONCE("HK_G17P_LAUNCH_TRACE", false);
   return enabled != 0;
}

/* One short line per submitted graphics stream.  This trace exists for real
 * applications where HK_G17P_LAUNCH_TRACE changes timing and produces too
 * much output to find the program whose fragment timestamp never ended. */
__attribute__((hot)) bool
hk_g17p_draw_trace_enabled(void)
{
   static int enabled = -1;
   if (enabled < 0)
      enabled = HK_ENV_BOOL_ONCE("HK_G17P_DRAW_TRACE", false);
   return enabled != 0;
}

/*
 * Terminate every native G17P compute launch with Apple's strong cache flush
 * (barrier flags 5) instead of the weak ordering barrier (flags 0).
 *
 * AGX2 flushes after every launch via hk_cdm_cache_flush(); the G17P native
 * dispatch path does not, so back-to-back dispatches in one CDM stream have no
 * write->read dependency. That is the shape of the G17P meta copy paths. This
 * is a one-boot probe for whether that gap is the Hyprland GMMU read fault
 * ZINK_DEBUG=sync cures; it is off by default because it cannot be decided
 * host-side.
 */
__attribute__((hot)) bool
hk_g17p_strong_dispatch_flush_enabled(void)
{
   static int enabled = -1;
   if (enabled < 0)
      enabled =
         HK_ENV_BOOL_ONCE("HK_G17P_STRONG_DISPATCH_FLUSH", false);
   return enabled != 0;
}

/*
 * Per-submission and per-allocation address tracing. These fire on every queue
 * submit and every command BO, so at desktop frame rates they cost an unbuffered
 * stderr write per frame and bury everything else in the journal. They stay
 * available, they just are not the default any more.
 */
bool
hk_g17p_verbose_enabled(void)
{
   static int enabled = -1;
   if (enabled < 0)
      enabled = HK_ENV_BOOL_ONCE("HK_G17P_VERBOSE", false);
   return enabled != 0;
}

/*
 * Diagnostics for the mid-render-pass control-stream split.
 *
 * A graphics control stream that is not the first of its render pass runs the
 * *partial* background/EOT programs and the partial ZLS control
 * (hk_cs_init_graphics). ZINK_DEBUG=sync ends the Vulkan render pass around
 * every draw, so under sync every stream is a first stream and the partial
 * path is never taken. Without sync a render pass reaches it through the 32 KiB
 * G17P VDM root filling, a depth-bias representation switch, a pipeline
 * barrier, or the one-draw-per-stream rule. Counting them, and naming which
 * rule fired, decides from a log whether a given path is on a failing route
 * instead of inferring it: HK_G17P_SPLIT_TRACE=1 counted *zero* splits across
 * a whole run that lost the GPU, which is how the partial BG/EOT programs were
 * ruled out as the cause.
 */
bool
hk_g17p_split_trace_enabled(void)
{
   static int enabled = -1;
   if (enabled < 0)
      enabled = HK_ENV_BOOL_ONCE("HK_G17P_SPLIT_TRACE", false);
   return enabled != 0;
}

void
hk_g17p_note_gfx_split(struct hk_cmd_buffer *cmd, const char *reason,
                       unsigned commands, size_t remaining)
{
   static unsigned count;

   if (!hk_g17p_split_trace_enabled())
      return;

   fprintf(stderr,
           "G17P gfx split #%u: %s (commands=%u remaining=%zu)\n", ++count,
           reason, commands, remaining);
   fflush(stderr);
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


/* See the macro in hk_cmd_buffer.h. Off unless HK_DEBUG names it, so this
 * costs one relaxed load on a path that is already failing the submission. */
void
hk_trace_cmd_error(const char *file, unsigned line, VkResult error)
{
   static int enabled = -1;
   if (enabled < 0) {
      const char *debug = getenv("HK_DEBUG");
      enabled = debug && strstr(debug, "cmderr") != NULL;
   }
   if (enabled)
      mesa_loge("command buffer error %d set at %s:%u", (int)error, file, line);
}

/* frg-codegen v4: bit1 (redundant reset elision) is the default; an explicit
 * vkResetCommandBuffer followed by vkBeginCommandBuffer no longer resets the
 * buffer twice. HK_G17P_HOST_ENCODE=0 restores the double reset. */
DEBUG_GET_ONCE_NUM_OPTION(hk_g17p_host_encode, "HK_G17P_HOST_ENCODE", 2)

VKAPI_ATTR VkResult VKAPI_CALL
hk_BeginCommandBuffer(VkCommandBuffer commandBuffer,
                      const VkCommandBufferBeginInfo *pBeginInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   /* v94 bit1: an explicit vkResetCommandBuffer/pool reset already left this
    * buffer in exactly the state the implicit reset below produces (nothing
    * can be recorded between that reset and this begin), so doing it again
    * only repeats the teardown and the full state memset. */
   if (!(debug_get_option_hk_g17p_host_encode() & (1u << 1)) ||
       !cmd->g17p_pristine)
      hk_reset_cmd_buffer(&cmd->vk, 0);
   cmd->g17p_pristine = false;
   vk_command_buffer_begin(&cmd->vk, pBeginInfo);

   if (cmd->vk.level == VK_COMMAND_BUFFER_LEVEL_SECONDARY &&
       pBeginInfo->pInheritanceInfo) {
      const VkCommandBufferInheritanceConditionalRenderingInfoEXT *conditional =
         vk_find_struct_const(
            pBeginInfo->pInheritanceInfo->pNext,
            COMMAND_BUFFER_INHERITANCE_CONDITIONAL_RENDERING_INFO_EXT);

      cmd->state.conditional.inherited =
         conditional && conditional->conditionalRenderingEnable;
   }

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
          !last->timestamp.end.handle &&
          !(last->timestamp.start.handle && cs->g17p_timestamp_query) &&
          !(last->timestamp.start.handle && cs->timestamp.start.handle)) {

         hk_sync_merge(cmd, last, cs);
         hk_cs_merge_cdm(last, cs);
         list_del(&cs->node);
         hk_cs_destroy(cs);
      } else {
         last = cs;
      }
   }
}

static void
hk_cmd_buffer_publish_root_uploads(struct hk_cmd_buffer *cmd)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (dev->dev.chip != AGX_CHIP_G17P ||
       list_is_empty(&cmd->uploader.root.bos))
      return;

   if (util_has_cache_ops()) {
      /* Every root range was cleaned when written. Complete those cache
       * operations once, after recording has made all roots immutable. */
      util_post_flush_fence();
      return;
   }

   /* Platforms without userspace cache maintenance publish each root slab
    * through their winsys. Root slabs contain no unrelated command uploads. */
   list_for_each_entry(struct hk_cmd_bo, bo, &cmd->uploader.root.bos, link) {
      if (agx_bo_sync_for_device(bo->bo, 0, bo->bo->size)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_DEVICE_LOST);
         return;
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
   hk_sync_flush_pending_timestamp(cmd);
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

   /* G17P roots use cached mappings. Publish the completed recording once,
    * after every root table is immutable and before submission can consume
    * any address embedded in the command streams. */
   hk_cmd_buffer_publish_root_uploads(cmd);

   hk_g17p_precomp_lookup_stats_finish(cmd);

   return vk_command_buffer_end(&cmd->vk);
}

static bool
hk_dependency_has_host_read(const VkDependencyInfo *info)
{
   for (uint32_t i = 0; i < info->memoryBarrierCount; ++i) {
      if (info->pMemoryBarriers[i].dstAccessMask & VK_ACCESS_2_HOST_READ_BIT)
         return true;
   }

   for (uint32_t i = 0; i < info->bufferMemoryBarrierCount; ++i) {
      if (info->pBufferMemoryBarriers[i].dstAccessMask &
          VK_ACCESS_2_HOST_READ_BIT)
         return true;
   }

   for (uint32_t i = 0; i < info->imageMemoryBarrierCount; ++i) {
      if (info->pImageMemoryBarriers[i].dstAccessMask &
          VK_ACCESS_2_HOST_READ_BIT)
         return true;
   }

   return false;
}

/* Whether any barrier in the dependency makes writes visible to later reads.
 * An execution-only dependency (both access masks zero) inside a render pass
 * instance needs no tile store. */
static bool
hk_dependency_has_memory_scope(const VkDependencyInfo *info)
{
   for (uint32_t i = 0; i < info->memoryBarrierCount; ++i) {
      if (info->pMemoryBarriers[i].dstAccessMask)
         return true;
   }

   for (uint32_t i = 0; i < info->bufferMemoryBarrierCount; ++i) {
      if (info->pBufferMemoryBarriers[i].dstAccessMask)
         return true;
   }

   for (uint32_t i = 0; i < info->imageMemoryBarrierCount; ++i) {
      if (info->pImageMemoryBarriers[i].dstAccessMask)
         return true;
   }

   return false;
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

   /* Inside a render pass instance both scopes are limited to that instance
    * (Vulkan 1.3, 7.6.1: "If vkCmdPipelineBarrier2 is called inside a render
    * pass instance, the first synchronization scope includes only commands
    * that occur earlier in submission order within the same subpass" and
    * likewise for the second). Nothing after the pass learns anything from
    * it. The one thing the driver owes it is memory: input attachments are
    * lowered to texture loads (nir_lower_input_attachments in hk_shader.c),
    * so a dependency with a memory scope needs the tile buffer stored and
    * reloaded, which is a render split. The continuation stream then waits
    * on the stream it continues, see hk_sync_stream_created. */
   if (cmd->sync.in_pass) {
      if (hk_dependency_has_memory_scope(pDependencyInfo) &&
          cmd->current_cs.gfx) {
         hk_g17p_note_gfx_split(cmd, "pipeline-barrier",
                                cmd->current_cs.gfx->stats.cmds, 0);
         hk_cmd_buffer_end_graphics(cmd);
      }
      return;
   }

   uint8_t src_kinds, dst_kinds;
   hk_sync_dependency_kinds(pDependencyInfo, &src_kinds, &dst_kinds);

   /* Compute work that stays in the open compute stream is ordered by the
    * stream itself; what it needs at a barrier is visibility. Every native
    * launch ends in a weak ordering barrier record, and the strong cache
    * flush is that same record with Apple's flags, upgraded in place. */
   struct hk_cs *cs = cmd->current_cs.cs;
   if (cs && (src_kinds & (HK_SYNC_COMPUTE | HK_SYNC_XFER)) && dst_kinds &&
       cs->stats.cmds > 0 && cs->current != cs->strong_flush_end) {
      hk_ensure_cs_has_space(cmd, cs, AGX3_CDM_BARRIER_SIZE);
      hk_cdm_cache_flush(dev, cs);
   }

   hk_sync_barrier(cmd, src_kinds, dst_kinds);

   /* A barrier-only command buffer otherwise records no GPU work, so it
    * cannot make a preceding fragment store visible to the host. Emit the
    * exact G17P strong CDM cache-maintenance record only when the API requests
    * host-read availability, in a stream ordered after the source scope. */
   if (dev->dev.chip == AGX_CHIP_G17P &&
       hk_dependency_has_host_read(pDependencyInfo)) {
      struct hk_sync_dep dep[HK_SYNC_NUM_SQ];
      hk_sync_src_dep(cmd, src_kinds, cmd->current_cs.cs, dep);

      cmd->sync.kind_override = HK_SYNC_ALL;
      struct hk_cs *flush = hk_sync_get_cs_after(cmd, dep);
      cmd->sync.kind_override = 0;
      if (!flush)
         return;

      hk_ensure_cs_has_space(cmd, flush, AGX3_CDM_BARRIER_SIZE);
      hk_cdm_cache_flush(dev, flush);
   }
}

__attribute__((hot)) void
hk_cmd_bind_shaders(struct vk_command_buffer *vk_cmd, uint32_t stage_count,
                    const mesa_shader_stage *stages,
                    struct vk_shader **const shaders)
{
   struct hk_cmd_buffer *cmd = container_of(vk_cmd, struct hk_cmd_buffer, vk);

   for (uint32_t i = 0; i < stage_count; i++) {
      struct hk_api_shader *shader = shaders[i]
         ? container_of(shaders[i], struct hk_api_shader, vk)
         : NULL;
      if (stages[i] == MESA_SHADER_COMPUTE || stages[i] == MESA_SHADER_KERNEL)
         hk_cmd_bind_compute_shader(cmd, shader);
      else
         hk_cmd_bind_graphics_shader(cmd, stages[i], shader);
   }

   if (cmd->current_cs.gfx)
      hk_cmd_prepare_bound_graphics(cmd);
}

/* Only ordinary G17P graphics programs whose descriptor values are fetched
 * through the root/table in GPU memory qualify.  CPU-decoded combined sampler
 * programs and variant selection retain the existing publication path.
 */
static bool
hk_g17p_can_retain_descriptor_root(struct hk_cmd_buffer *cmd,
                                  const struct hk_descriptor_state *desc)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   if (hk_cmd_buffer_device(cmd)->dev.chip != AGX_CHIP_G17P ||
       desc != &gfx->descriptors ||
       gfx->shaders[MESA_SHADER_GEOMETRY] ||
       gfx->shaders[MESA_SHADER_TESS_CTRL] ||
       gfx->shaders[MESA_SHADER_TESS_EVAL])
      return false;

   struct hk_api_shader *vs = gfx->shaders[MESA_SHADER_VERTEX];
   struct hk_api_shader *fs = gfx->shaders[MESA_SHADER_FRAGMENT];
   if (!vs || !fs || vs->g17p_ordinary_variants ||
       fs->g17p_ordinary_variants || fs->g17p_sample_variants)
      return false;

   const struct hk_shader *hw_vs = hk_bound_hw_vs(gfx);
   const struct hk_shader *hw_fs = hk_fs_variant(fs, 1);
   return hw_vs && hw_fs && hw_vs->compiled && hw_fs->compiled &&
          !hw_fs->info.agx3_combined_sampler_valid &&
          !hw_fs->info.g17p_apple_fs;
}

static void
hk_bind_descriptor_sets(struct hk_cmd_buffer *cmd,
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

   bool root_changed = false;
   uint32_t next_dyn_offset = 0;
   for (uint32_t i = 0; i < info->descriptorSetCount; ++i) {
      unsigned s = i + info->firstSet;
      VK_FROM_HANDLE(hk_descriptor_set, set, info->pDescriptorSets[i]);

      if (desc->sets[s] != set ||
          (desc->push_active & BITFIELD_BIT(s))) {
         if (set != NULL) {
            root_changed |= desc->root.sets[s] != hk_descriptor_set_addr(set);
            desc->root.sets[s] = hk_descriptor_set_addr(set);
            desc->set_sizes[s] = set->size;
         } else {
            root_changed |= desc->root.sets[s] != 0;
            desc->root.sets[s] = 0;
            desc->set_sizes[s] = 0;
         }
         desc->sets[s] = set;
         desc->sets_dirty |= BITFIELD_BIT(s);

         /* Binding descriptors invalidates push descriptors */
      }
      desc->push_active &= ~BITFIELD_BIT(s);
      desc->push_dirty &= ~BITFIELD_BIT(s);

      root_changed |=
         desc->root.set_dynamic_buffer_start[s] != dyn_buffer_start;
      desc->root.set_dynamic_buffer_start[s] = dyn_buffer_start;

      if (pipeline_layout->set_layouts[s] != NULL) {
         const struct hk_descriptor_set_layout *set_layout =
            vk_to_hk_descriptor_set_layout(pipeline_layout->set_layouts[s]);

         if (set != NULL && set_layout->vk.dynamic_descriptor_count > 0) {
            for (uint32_t j = 0; j < set_layout->vk.dynamic_descriptor_count; j++) {
               struct hk_buffer_address addr = set->dynamic_buffers[j];
               addr.base_addr += info->pDynamicOffsets[next_dyn_offset + j];
               const struct hk_buffer_address *old =
                  &desc->root.dynamic_buffers[dyn_buffer_start + j];
               root_changed |= old->base_addr != addr.base_addr ||
                               old->size != addr.size ||
                               old->sparse_map_page != addr.sparse_map_page;
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

   /* Preserve all descriptor bookkeeping even if the root's immutable bytes
    * can be reused.  Other paths keep the prior unconditional invalidation.
    */
   desc->root_dirty |= root_changed ||
      !hk_g17p_can_retain_descriptor_root(cmd, desc);
   desc->generation++;
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

   if (pBindDescriptorSetsInfo->stageFlags & HK_RT_SHADER_STAGES) {
      hk_bind_descriptor_sets(cmd, &cmd->state.rt.descriptors,
                              pBindDescriptorSetsInfo);
   }
}

static void
hk_push_constants(UNUSED struct hk_cmd_buffer *cmd,
                  struct hk_descriptor_state *desc,
                  const VkPushConstantsInfoKHR *info)
{
   /* D3D translation layers re-push unchanged constants before most draws.
    * While the root is clean the uploaded root holds exactly these bytes, so
    * an identical push changes nothing; keep the root and every preparation
    * keyed to the current descriptor generation. */
   if (memcmp(desc->root.push + info->offset, info->pValues, info->size) == 0)
      return;

   memcpy(desc->root.push + info->offset, info->pValues, info->size);
   desc->root_dirty = true;
   desc->generation++;
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

   if (pPushConstantsInfo->stageFlags & HK_RT_SHADER_STAGES)
      hk_push_constants(cmd, &cmd->state.rt.descriptors, pPushConstantsInfo);
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
   desc->push_active |= BITFIELD_BIT(set);
   desc->generation++;

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

   if (pPushDescriptorSetInfo->stageFlags & HK_RT_SHADER_STAGES) {
      hk_push_descriptor_set(cmd, &cmd->state.rt.descriptors,
                             pPushDescriptorSetInfo);
   }
}

void
hk_cmd_buffer_flush_push_descriptors(struct hk_cmd_buffer *cmd,
                                     struct hk_descriptor_state *desc)
{
   u_foreach_bit(set_idx, desc->push_dirty) {
      struct hk_push_descriptor_set *push_set = desc->push[set_idx];
      uint64_t push_set_addr;
      struct hk_device *dev = hk_cmd_buffer_device(cmd);
      const bool dedicated_g17p_descriptor_bo =
         dev->dev.chip == AGX_CHIP_G17P &&
         hk_g17p_push_descriptor_dedicated_bo();
      if (dedicated_g17p_descriptor_bo) {
         /* G17P diagnostic: keep the pushed descriptor snapshot out of the
          * mixed command uploader whose BO may also acquire a low VDM alias.
          * Allocated descriptor-pool BOs work for both combined and separate
          * samples; this makes backing/alias class the only changed variable. */
         struct agx_bo *bo = agx_bo_create(
            &dev->dev, sizeof(push_set->data), HK_MIN_UBO_ALIGNMENT, 0,
            "G17P pushed descriptor diagnostic");
         if (!bo) {
            vk_command_buffer_set_error(&cmd->vk,
                                        VK_ERROR_OUT_OF_DEVICE_MEMORY);
            return;
         }

         void *map = agx_bo_map(bo);
         if (!map) {
            agx_bo_unreference(&dev->dev, bo);
            vk_command_buffer_set_error(&cmd->vk,
                                        VK_ERROR_MEMORY_MAP_FAILED);
            return;
         }
         memcpy(map, push_set->data, sizeof(push_set->data));
         if (agx_bo_sync_for_device(bo, 0, sizeof(push_set->data))) {
            agx_bo_unreference(&dev->dev, bo);
            vk_command_buffer_set_error(&cmd->vk, VK_ERROR_DEVICE_LOST);
            return;
         }
         util_dynarray_append(&cmd->large_bos, bo);
         push_set_addr = bo->va->addr;
      } else {
         push_set_addr = hk_pool_upload(
            cmd, push_set->data, sizeof(push_set->data),
            HK_MIN_UBO_ALIGNMENT);
      }

      if (hk_g17p_launch_trace_enabled())
         hk_g17p_trace_line("push-descriptor-set", push_set_addr,
                            push_set->data, sizeof(push_set->data));

      if (!dedicated_g17p_descriptor_bo && hk_g17p_sync_push_descriptors()) {
         bool found = false;
         list_for_each_entry(struct hk_cmd_bo, bo,
                             &cmd->uploader.main.bos, link) {
            const uint64_t base = bo->bo->va->addr;
            if (push_set_addr < base ||
                push_set_addr - base > bo->bo->size - sizeof(push_set->data))
               continue;
            found = true;
            if (agx_bo_sync_for_device(bo->bo, push_set_addr - base,
                                       sizeof(push_set->data)))
               vk_command_buffer_set_error(&cmd->vk, VK_ERROR_DEVICE_LOST);
            break;
         }
         if (!found)
            vk_command_buffer_set_error(&cmd->vk,
                                        VK_ERROR_OUT_OF_DEVICE_MEMORY);
      }

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

__attribute__((hot)) uint64_t
hk_cmd_buffer_upload_root(struct hk_cmd_buffer *cmd,
                          VkPipelineBindPoint bind_point)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_descriptor_state *desc = hk_get_descriptors_state(cmd, bind_point);
   struct hk_root_descriptor_table *root = &desc->root;

   struct agx_ptr root_ptr = {0};
   if (bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS &&
       cmd->state.gfx.prepared_root.gpu) {
      root_ptr = cmd->state.gfx.prepared_root;
      cmd->state.gfx.prepared_root = (struct agx_ptr){0};
   } else {
      root_ptr = hk_pool_root_alloc(cmd, sizeof(*root), 8);

      /* G17P may carry an immutable root across render boundaries.  If state
       * becomes dirty after bind-time preparation, this allocation gives the
       * draw a new root so older draws keep their exact bytes.  Any prepared
       * program or HAL200 tail still names the old root and must be rebuilt
       * before publication. */
      if (bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS &&
          cmd->state.gfx.root && root_ptr.gpu != cmd->state.gfx.root) {
         cmd->state.gfx.prepared_vertex_usc = 0;
         cmd->state.gfx.prepared_fragment_usc = 0;
         cmd->state.gfx.g17p_bind_tail.valid = false;
      }
   }
   if (!root_ptr.gpu)
      return 0;

   root->root_desc_addr = root_ptr.gpu;
   root->printf_buffer_address = dev->dev.printf_buffer_address;

   memcpy(root_ptr.cpu, root, sizeof(*root));

   /* Cached G17P roots need explicit CPU-to-GPU publication. Clean just this
    * table now; vkEndCommandBuffer completes all recorded root cleans with a
    * single fence before submission can consume them. */
   if (dev->dev.chip == AGX_CHIP_G17P && util_has_cache_ops())
      util_flush_range_no_fence(root_ptr.cpu, sizeof(*root));

   /* Diagnostic: which descriptor sets each published graphics root names,
    * so a launch's root range can be matched to the sets its constant
    * program reads (the uniform DMA stale-set gate). */
   if (unlikely(dev->g17p_diagnostics.compact_ioto_trace) &&
       bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS) {
      fprintf(stderr, "G17P root-table: va=%" PRIx64, root_ptr.gpu);
      for (unsigned i = 0; i < HK_MAX_SETS; ++i)
         fprintf(stderr, " set%u=%" PRIx64, i, root->sets[i]);
      fprintf(stderr, "\n");
   }

   return root_ptr.gpu;
}

bool
hk_cmd_buffer_prepare_graphics_root(struct hk_cmd_buffer *cmd)
{
   if (cmd->state.gfx.prepared_root.gpu)
      return true;

   /* A published root is immutable command-buffer storage.  Reuse it across
    * render boundaries while the CPU-side logical root is unchanged.  This
    * preserves the exact root identity used by cached G17P graphics programs
    * and HAL200 state instead of cloning the same table for every pass.
    * hk_cmd_buffer_upload_root() allocates a new table and invalidates those
    * prepared objects if a later dynamic-state update dirties the root. */
   if (!cmd->state.gfx.descriptors.root_dirty && cmd->state.gfx.root)
      return true;

   struct agx_ptr root =
      hk_pool_root_alloc(cmd, sizeof(struct hk_root_descriptor_table), 8);
   if (!root.gpu)
      return false;

   cmd->state.gfx.prepared_root = root;
   cmd->state.gfx.root = root.gpu;
   return true;
}

static uint64_t
hk_upload_native_spilled_rt_descs(struct hk_cmd_buffer *cmd)
{
   const struct hk_rendering_state *render = &cmd->state.gfx.render;
   uint8_t entries[HK_MAX_RTS][AGX_G17P_EMRT_DESCRIPTOR_SIZE] = {0};
   for (unsigned rt = 0; rt < render->color_att_count; ++rt) {
      if (!render->tilebuffer.spilled[rt])
         continue;
      const struct hk_image_view *view = render->color_att[rt].iview;
      if (!view || !view->planes[0].emrt_pbe_valid ||
          !view->planes[0].emrt_read_heap_index) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FORMAT_NOT_SUPPORTED);
         return 0;
      }
      memcpy(entries[rt], &view->planes[0].emrt_texture, 32);
      memcpy(entries[rt] + 32, &view->planes[0].emrt_pbe, 32);
      memcpy(entries[rt] + AGX_G17P_STORAGE_HEAP_INDEX_OFFSET,
             &view->planes[0].emrt_read_heap_index, 4);
      memcpy(entries[rt] + AGX_G17P_SAMPLED_HEAP_INDEX_OFFSET,
             &view->planes[0].emrt_read_heap_index, 4);
      memcpy(entries[rt] + AGX_G17P_IMAGE_ATOMIC_SW_OFFSET,
             &view->planes[0].emrt_atomic_sw, 8);
      memcpy(entries[rt] + AGX_G17P_EMRT_ALIGNED_WIDTH_OFFSET,
             &view->planes[0].emrt_aligned_width, 4);
   }

   const size_t size = render->color_att_count * sizeof(entries[0]);
   __typeof__(cmd->state.gfx.g17p_spilled_rt_descs) *cached =
      &cmd->state.gfx.g17p_spilled_rt_descs;
   if (cached->valid && cached->color_att_count == render->color_att_count &&
       !memcmp(cached->entries, entries, size))
      return cached->pointer;

   const uint64_t address = hk_pool_upload(cmd, entries, size, 32);
   const uint64_t pointer = address ? hk_pool_upload(cmd, &address, 8, 8) : 0;
   if (pointer) {
      memcpy(cached->entries, entries, size);
      cached->address = address;
      cached->pointer = pointer;
      cached->color_att_count = render->color_att_count;
      cached->valid = true;
   }
   return pointer;
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

__attribute__((hot)) void
hk_reserve_scratch(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                   struct hk_shader *s)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   uint32_t max_scratch_size =
      MAX2(s->b.info.scratch_size, s->b.info.preamble_scratch_size);

   /* Not scratch but this is the most convenient place for this... */
   cs->uses_sampler_heap |= s->b.info.uses_sampler_heap;

   /* G17P publishes a shader's spill frame as thread-private storage in its
    * SPECTPR. There is no helper-program stack, and the kernel rejects a
    * command that names one. */
   if (max_scratch_size == 0 ||
       agx_chip_command_stream_generation(dev->dev.chip) ==
          AGX_CS_GENERATION_HAL200)
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
   return cmd->state.gfx.render.g17p_rt0_layout_supported;
}

/* Keep native interpolation bounded to the f32 UVS layout already supported
 * by hk_agx3_g17p_vertex_program_info: position plus up to 64 user words.
 * HAL200 updateVaryingTable (0x3efa58) uses the same CF entry encoding for
 * smooth, linear and flat inputs, including their mixture in one shader.
 * Flat inputs select C. Native G17P Point and Triangle pipelines have identical
 * complete fragment text for flat vec2/vec3/vec4 inputs. The provoking selector
 * is topology-independent, so line lists and strips use the same witnessed
 * first/last selector values as triangles. First-provoking fans use their
 * dedicated selector and the fragment shader's draw-root B/C coefficient
 * choice. Position Z and reciprocal W use native coefficient sources.
 */
static bool
hk_g17p_smooth_varyings_supported_uncached(struct hk_cmd_buffer *cmd,
                                          const struct hk_linked_shader *linked)
{
   if (!linked)
      return false;
   const struct agx_varyings_fs *cf = &linked->b.cf;
   if (cf->nr_bindings == 0)
      return cf->nr_cf == 0;

   /* The coefficients describe the hardware vertex program the draw
    * launches: the API vertex shader, the post-tessellation vertex program,
    * or the geometry raster-copy variant. */
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   const struct hk_shader *vs =
      gfx->shaders[MESA_SHADER_VERTEX] ? hk_bound_hw_vs(gfx) : NULL;
   if (!vs || cf->nr_bindings > HK_AGX3_G17P_MAX_COEFFICIENTS ||
       cf->nr_cf > HK_AGX3_G17P_MAX_COEFFICIENTS ||
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
      if (!b->smooth) {
         const bool point = gfx->topology == AGX_PRIMITIVE_POINTS &&
                            gfx->provoking == 0;
         const bool line =
            (gfx->topology == AGX_PRIMITIVE_LINES ||
             gfx->topology == AGX_PRIMITIVE_LINE_STRIP) &&
            (gfx->provoking == 0 || gfx->provoking == 2);
         const bool triangle =
            (gfx->topology == AGX_PRIMITIVE_TRIANGLES ||
             gfx->topology == AGX_PRIMITIVE_TRIANGLE_STRIP) &&
            (gfx->provoking == 0 || gfx->provoking == 2);
         const bool fan = gfx->topology == AGX_PRIMITIVE_TRIANGLE_FAN &&
                          (gfx->provoking == 1 || gfx->provoking == 2);
         if (b->perspective || (!point && !line && !triangle && !fan))
            return false;
      }
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
         /* Fragment layer and gl_ViewportIndex are ordinary flat varyings
          * that the vertex shader also writes out as user words. Clip-distance
          * inputs use the compiler's compact coefficient layout while the
          * fixed-function clipper consumes the same VS output. Cull-distance
          * emulation similarly links the VS-generated CULL_PRIMITIVE words
          * into its FS prolog as noperspective coefficients. */
         const bool cull_slot =
            b->slot == VARYING_SLOT_CULL_PRIMITIVE ||
            b->slot == VARYING_SLOT_CULL_PRIMITIVE + 1;
         const bool clip_slot = b->slot == VARYING_SLOT_CLIP_DIST0 ||
                                b->slot == VARYING_SLOT_CLIP_DIST1;
         /* gl_PrimitiveID that no vertex shader writes is not a UVS word at
          * all: agx_link_varyings_vs_fs() gives its binding
          * AGX_COEFFICIENT_SOURCE_PRIMITIVE_ID and asks the VDM to generate
          * the value, which hk_flush_dynamic_state() already plumbs through
          * generate_primitive_id. Admit it as a system coefficient beside
          * FragCoord Z and 1/W -- scalar, flat, and indexing no vertex
          * output. A VS-written gl_PrimitiveID would take the ordinary user
          * word path instead and stays refused, having no witness. */
         if (b->slot == VARYING_SLOT_PRIMITIVE_ID) {
            if (gfx->linked_varyings.slots[VARYING_SLOT_PRIMITIVE_ID] ||
                b->count != 1 || b->offset != 0 || b->smooth || b->perspective)
               return false;
            ++user_bindings;
            next_cf += b->count;
            continue;
         }
         /* gl_PointCoord is the third hardware-generated coefficient source,
          * beside FragCoord Z and gl_PrimitiveID: nir_lower_sysvals_to_varyings
          * turns the system value into a VARYING_SLOT_PNTC input and
          * agx_link_varyings_vs_fs() gives that binding
          * AGX_COEFFICIENT_SOURCE_POINT_COORD with base slot 0, so it indexes
          * no vertex output and needs no UVS word. The rasterizer supplies the
          * sprite UV, selected by the Object Type nibble that
          * translate_object_type() already sets to POINT_SPRITE_UV01 for point
          * topology and that hk_g17p_emit_native_bind_tail() already packs into
          * both face words. The CF entry is the same 4-byte encoding HAL200
          * updateVaryingTable consumes for every other source, so nothing else
          * in the native tail changes.
          *
          * agx_link_varyings_vs_fs() asserts base offset 0 and overwrites the
          * base slot, so a partial read starting at .y has no correct encoding
          * on any chip; keep it refused rather than rendering it wrong.
          *
          * The CF entry carries no topology or provoking selector, so this is
          * not gated on point topology: a dynamic-topology pipeline may bind
          * the same fragment shader for a triangle draw, where Vulkan leaves
          * gl_PointCoord undefined. Refusing there would turn a legal draw
          * into VK_ERROR_FEATURE_NOT_PRESENT for no conformance gain. */
         if (b->slot == VARYING_SLOT_PNTC) {
            if (b->offset != 0 || b->count > 2 || !b->smooth ||
                (b->perspective && !has_rhw))
               return false;
            ++user_bindings;
            next_cf += b->count;
            continue;
         }
         const bool user_slot = (b->slot >= VARYING_SLOT_VAR0 &&
                                 b->slot <= VARYING_SLOT_VAR31) ||
                                b->slot == VARYING_SLOT_LAYER ||
                                b->slot == VARYING_SLOT_VIEWPORT || clip_slot ||
                                cull_slot;
         if (!user_slot || (cull_slot && (b->perspective || !b->smooth)) ||
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

/* Evaluated for every draw that republishes its fragment program. The
 * answer depends only on the linked fragment coefficients, the hardware
 * vertex shader's outputs, the topology, the provoking vertex and the
 * current linked_varyings slot table, so remember the last evaluation. */
bool
hk_g17p_smooth_varyings_supported(struct hk_cmd_buffer *cmd,
                                 const struct hk_linked_shader *linked)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   const struct hk_shader *vs =
      gfx->shaders[MESA_SHADER_VERTEX] ? hk_bound_hw_vs(gfx) : NULL;
   __typeof__(gfx->g17p_smooth_memo) *memo = &gfx->g17p_smooth_memo;
   if (linked && memo->valid && memo->linked == linked && memo->vs == vs &&
       memo->topology == gfx->topology && memo->provoking == gfx->provoking)
      return memo->result;

   const bool result = hk_g17p_smooth_varyings_supported_uncached(cmd, linked);
   if (linked) {
      *memo = (__typeof__(*memo)){
         .linked = linked, .vs = vs, .topology = gfx->topology,
         .provoking = gfx->provoking, .valid = true, .result = result,
      };
   }
   return result;
}

static bool
hk_g17p_basic_graphics_program_supported(struct hk_cmd_buffer *cmd,
                                         struct hk_shader *s,
                                         struct hk_linked_shader *linked)
{
   const bool native_spills = s->info.stage == MESA_SHADER_FRAGMENT &&
      agx_tilebuffer_spills(&cmd->state.gfx.render.tilebuffer);
   /* First native CProg qualification covers the imported-input region.
    * Local/spill frame use needs its own native metadata witnesses before
    * admission. This is not a fallback:
    * a candidate outside the bounded protocol is refused. */
   /* Every G17P hardware vertex program (API VS, GS raster and native TES
    * variants) is compiled with a native CProg when anything is hoistable;
    * its VertexProgramVariant carries the TI counts. The stage branches
    * below still require that variant. */
   const bool native_preamble =
      (s->b.info.stage == MESA_SHADER_VERTEX ||
       (s->info.stage == MESA_SHADER_FRAGMENT &&
        agx_g17p_fs_native_uniforms())) &&
      hk_g17p_native_cprog_qualified(&s->b.info);
   if (!linked || !linked->b.bo || !linked->b.bo->va || linked->b.size == 0 ||
       linked->b.size > UINT32_MAX || (s->b.info.has_preamble && !native_preamble) ||
       s->b.info.scratch_size > AGX3_G17P_MAX_SPILL_B ||
       (s->b.info.preamble_scratch_size != 0 && !native_preamble) ||
       linked->b.scratch_size > AGX3_G17P_MAX_SPILL_B ||
       s->b.info.texture_state_count != 0 ||
       (s->b.info.sampler_state_count != 0 &&
        !(s->info.agx3_indexed_sampler_valid &&
          s->b.info.sampler_state_count == 1)) ||
       s->b.info.uses_sampler_heap ||
       (linked->b.uses_txf && !s->info.agx3_indexed_sampler_valid && !native_spills) ||
       s->info.set_count > HK_AGX3_G17P_MAX_ROOT_SETS) {
      mesa_loge_once(
         "G17P graphics program refused: linked=%u bo=%u va=%u size=%zu "
         "preamble=%u scratch=%u/%u/%u textures=%u samplers=%u "
         "indexed_sampler=%u sampler_heap=%u txf=%u sets=%u",
         linked != NULL, linked && linked->b.bo != NULL,
         linked && linked->b.bo && linked->b.bo->va != NULL,
         linked ? linked->b.size : 0, s->b.info.has_preamble,
         s->b.info.scratch_size, s->b.info.preamble_scratch_size,
         linked ? linked->b.scratch_size : 0,
         s->b.info.texture_state_count, s->b.info.sampler_state_count,
         s->info.agx3_indexed_sampler_valid, s->b.info.uses_sampler_heap,
         linked ? linked->b.uses_txf : 0, s->info.set_count);
      return false;
   }

   if (s->info.stage == MESA_SHADER_VERTEX) {
      const unsigned nr_attribs = DIV_ROUND_UP(
         BITSET_LAST_BIT(s->info.vs.attrib_components_read), 4);
      /* Software indexing is admitted when the bound program is still the
       * hardware vertex shader and only fetches its own indices, which is
       * what an adjacency topology without a geometry shader compiles to.
       * Its extra state is the VERTEX_PARAMS uniform, which
       * hk_flush_dynamic_state() already uploads for exactly this case. The
       * geometry and tessellation shape, where the software vertex shader
       * runs as a compute kernel, stays refused. */
      const bool ok =
         (!linked->sw_indexing || linked->sw_indexing_hw_vs) &&
         nr_attribs <= AGX_MAX_VBUFS &&
         s->agx3_vdm_state_update.source_layout ==
            AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT;

      if (!ok) {
         mesa_loge_once("G17P graphics program refused: stage=vs "
                        "sw_indexing=%u attribs=%u/%u source_layout=%u "
                        "(want %u)",
                        linked->sw_indexing, nr_attribs, AGX_MAX_VBUFS,
                        (unsigned)s->agx3_vdm_state_update.source_layout,
                        (unsigned)
                        AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT);
      }

      return ok;
   }

   /* The raster variant produced by poly_nir_lower_gs is a hardware vertex
    * program while retaining the API geometry stage in hk_shader_info.  It
    * pulls generated vertices through the root ABI rather than vertex
    * attributes, then performs the XFB stores before rasterization.  The
    * post-tessellation vertex program likewise keeps the evaluation stage and
    * reads the control shader's outputs through the root ABI. */
   if ((s->info.stage == MESA_SHADER_GEOMETRY ||
        s->info.stage == MESA_SHADER_TESS_EVAL) &&
       s->b.info.stage == MESA_SHADER_VERTEX) {
      const bool ok =
         !linked->sw_indexing &&
         s->agx3_vdm_state_update.source_layout ==
            AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT;

      if (!ok) {
         mesa_loge_once("G17P graphics program refused: stage=gs-raster "
                        "sw_indexing=%u source_layout=%u (want %u)",
                        linked->sw_indexing,
                        (unsigned)s->agx3_vdm_state_update.source_layout,
                        (unsigned)
                        AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT);
      }

      return ok;
   }

   if (s->info.stage != MESA_SHADER_FRAGMENT) {
      mesa_loge_once("G17P graphics program refused: stage=%u has no native "
                     "G17P graphics path (sw stage %u, hw stage %u)",
                     (unsigned)s->info.stage, (unsigned)s->info.stage,
                     (unsigned)s->b.info.stage);
      return false;
   }

   {
      const bool smooth = hk_g17p_smooth_varyings_supported(cmd, linked);
      const bool layout = hk_g17p_basic_fragment_layout(cmd);

      if (!smooth || !layout) {
         mesa_loge_once("G17P graphics program refused: stage=fs "
                        "smooth_varyings=%u basic_fragment_layout=%u",
                        smooth, layout);
      }

      return smooth && layout;
   }
}

/*
 * Every uniform range hk_g17p_graphics_uniform_ranges() below can emit, by the
 * branch that emits it. A vertex program can request all of the first block;
 * a fragment program the second. Keeping the budget derived from this instead
 * of hand-set is what stops a legal program from being refused fail-closed --
 * a budget of five silently refused every Firefox/Zink vertex program that
 * sampled a texture and promoted a constant.
 */
#define HK_G17P_VERTEX_RANGES                                                  \
   (2 /* attribute bases and clamps */ + 1 /* draw parameters */ +             \
    1 /* draw ID */ + 1 /* indexed sampler heap */ + 1 /* root */ +            \
    1 /* promoted constants */)
/* v51b uniform DMA adds one VS range; hk_shader.c enables it only when the
 * draw ID and the indexed heap are not both used, keeping VS <= 7. */

#define HK_G17P_FRAGMENT_RANGES                                                \
   (1 /* blend constants */ + 1 /* combined sampler heap */ +                  \
    1 /* indexed sampler heap */ + 1 /* root */ + 1 /* promoted constants */ + \
    1 /* native spill table */ + 1 /* v51a uniform DMA span */)

static_assert(HK_AGX3_G17P_GRAPHICS_MAX_RANGES >= HK_G17P_VERTEX_RANGES &&
                 HK_AGX3_G17P_GRAPHICS_MAX_RANGES >= HK_G17P_FRAGMENT_RANGES,
              "the graphics range budget must cover every range "
              "hk_g17p_graphics_uniform_ranges() can emit, or a legal "
              "program is refused and ranges[] is written past its end");
static_assert(HK_G17P_GRAPHICS_ROOT_MAX_RANGES ==
                 HK_AGX3_G17P_GRAPHICS_MAX_RANGES,
              "compact root range array must match the graphics collector");

/*
 * Record how many uniform ranges the collector had already committed when it
 * refused. Without it the refusal diagnostic cannot distinguish "this program
 * needs more ranges than the budget allows" from the several structural
 * rejections that share the same message, which is exactly the ambiguity that
 * made the Firefox hardware refusal unreadable.
 */
static bool
hk_g17p_graphics_ranges_refused(size_t *range_count, size_t count)
{
   *range_count = count;
   return false;
}

/*
 * AGX_G17P_UNIFORM_DMA sources its span from a pointer uniform: either the
 * root table (a root-relative address, so it follows the root's identity) or
 * a descriptor-set pointer.  For a set pointer the span address is the set's
 * address *value*, copied from the CPU-side root when the launch program is
 * built.  A program reused on root identity alone (the exact-last cache, a
 * program prepared at vkCmdBindPipeline, or the bind tail that embeds it)
 * therefore keeps reading the set that was bound when it was built.
 * Minecraft binds its pipeline before its descriptor sets, so every draw
 * after a pipeline switch read the previous pipeline's set: wrong UBOs,
 * wrong colours, and UBO-bounded loops that never terminate.
 *
 * Return that value (0 when the span is root-relative or absent) so every
 * reuse path can require it to be unchanged.
 */
uint64_t
hk_g17p_dma_set_base(const struct hk_cmd_buffer *cmd,
                     const struct hk_shader *s)
{
   if (!s || !s->b.info.agx3_uniform_dma_valid)
      return 0;

   const unsigned root_uniform = s->info.agx3_root_uniform;
   const unsigned pointer = s->b.info.agx3_uniform_dma_pointer;
   if (pointer == root_uniform || pointer < root_uniform + 4 ||
       (pointer - root_uniform - 4) % 4 != 0 ||
       (pointer - root_uniform - 4) / 4 >= HK_MAX_SETS)
      return 0;

   const uint64_t base =
      cmd->state.gfx.descriptors.root.sets[(pointer - root_uniform - 4) / 4];
   /* Distinguish "set pointer, currently unbound" (zero page) from "no set
    * span" so that binding a set later still invalidates. */
   return base ? base : AGX_ZERO_PAGE_ADDRESS;
}

/* sources, when not NULL, receives each range's enum hk_g17p_range_source. */
static bool
hk_g17p_graphics_uniform_ranges(
   struct hk_cmd_buffer *cmd, struct hk_shader *s,
   const struct hk_linked_shader *linked,
   struct agx3_ioto_uniform_range ranges[HK_AGX3_G17P_GRAPHICS_MAX_RANGES],
   size_t *range_count, uint8_t *sources)
{
   uint8_t ignored_sources[HK_AGX3_G17P_GRAPHICS_MAX_RANGES];
   if (!sources)
      sources = ignored_sources;
   /* A range whose source is not recorded below is not re-derivable: the
    * compact launcher cache must then republish rather than patch it. */
   memset(sources, HK_G17P_RANGE_SOURCE_UPLOADED,
          HK_AGX3_G17P_GRAPHICS_MAX_RANGES);

   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   const uint64_t root_ptr = cmd->state.gfx.root;
   size_t count = 0;
   const unsigned root_uniform = s->info.agx3_root_uniform;

   /* Diagnostic HK_G17P_APPLE_FS: Apple's own uniform DMA layout. */
   if (s->info.stage == MESA_SHADER_FRAGMENT && s->info.g17p_apple_fs) {
      if (agx_tilebuffer_spills(&cmd->state.gfx.render.tilebuffer) ||
          hk_cmd_uses_blend_constant(cmd))
         return hk_g17p_graphics_ranges_refused(range_count, 0);
      return hk_g17p_apple_fs_ranges(cmd, s, root_ptr, ranges,
                                     HK_AGX3_G17P_GRAPHICS_MAX_RANGES,
                                     range_count);
   }
   const bool native_spills = s->info.stage == MESA_SHADER_FRAGMENT &&
      agx_tilebuffer_spills(&cmd->state.gfx.render.tilebuffer);

   if (s->info.stage == MESA_SHADER_VERTEX) {
      const unsigned nr_attribs = DIV_ROUND_UP(
         BITSET_LAST_BIT(s->info.vs.attrib_components_read), 4);
      const unsigned uniform_base = s->info.agx3_indexed_sampler_valid
         ? AGX_ABI_G17P_TEXTURE_POINTER_HALFS
         : 0;
      if (nr_attribs > AGX_MAX_VBUFS)
         return hk_g17p_graphics_ranges_refused(range_count, count);

      if (nr_attribs != 0) {
         if (!root_ptr)
            return hk_g17p_graphics_ranges_refused(range_count, count);

         sources[count] = HK_G17P_RANGE_SOURCE_ROOT;
         ranges[count++] = (struct agx3_ioto_uniform_range){
            .start_halfs = uniform_base + AGX_ABI_VUNI_VBO_BASE(0),
            .size_halfs = 4 * nr_attribs,
            .address =
               root_ptr + hk_root_descriptor_offset(draw.attrib_base),
         };
         sources[count] = HK_G17P_RANGE_SOURCE_ROOT;
         ranges[count++] = (struct agx3_ioto_uniform_range){
            .start_halfs = uniform_base +
                           AGX_ABI_VUNI_VBO_CLAMP(nr_attribs, 0),
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
         sources[count] = HK_G17P_RANGE_SOURCE_DRAW_PARAMS;
         ranges[count++] = (struct agx3_ioto_uniform_range){
            .start_halfs = uniform_base +
                           AGX_ABI_VUNI_FIRST_VERTEX(nr_attribs),
            .size_halfs = 4,
            .address = cmd->state.gfx.draw_params,
         };
      }
      if (cmd->state.gfx.draw_id_ptr) {
         /* The shader reads gl_DrawID as a single half, but a uniform range
          * is loaded in 32-bit dwords -- agx3_ioto_uniform_program.c rejects
          * an odd size_halfs outright -- so a one-half range could never be
          * published and every gl_DrawID vertex program was refused. Publish
          * the containing dword instead. AGX_ABI_VUNI_DRAW_ID is always
          * FIRST_VERTEX + 4 and AGX_ABI_VUNI_VERTEX_PARAMS is at least
          * FIRST_VERTEX + 8, so the second half is reserved ABI padding that
          * no other range claims. hk_cmd_draw.c uploads the matching dword. */
         sources[count] = HK_G17P_RANGE_SOURCE_DRAW_ID;
         ranges[count++] = (struct agx3_ioto_uniform_range){
            .start_halfs = uniform_base +
                           AGX_ABI_VUNI_DRAW_ID(nr_attribs),
            .size_halfs = 2,
            .address = cmd->state.gfx.draw_id_ptr,
         };
      }

   } else if ((s->info.stage == MESA_SHADER_GEOMETRY ||
               s->info.stage == MESA_SHADER_TESS_EVAL) &&
              s->b.info.stage == MESA_SHADER_VERTEX) {
      /* The compiled geometry/tessellation raster variant owns its root
       * layout. Do not infer it from the original API stage. */
   } else if (s->info.stage == MESA_SHADER_FRAGMENT) {
      if (native_spills) {
         const uint64_t pointer = hk_upload_native_spilled_rt_descs(cmd);
         if (!pointer || !linked->g17p_emrt_heap_base)
            return hk_g17p_graphics_ranges_refused(range_count, count);
         sources[count] = HK_G17P_RANGE_SOURCE_UPLOADED;
         ranges[count++] = (struct agx3_ioto_uniform_range){
            .start_halfs = linked->g17p_emrt_heap_base,
            .size_halfs = 4, .address = pointer,
         };
      }
      if (hk_cmd_uses_blend_constant(cmd)) {
         /* The non-monolithic blend epilog reads the four constants from
          * the base selected by its linked key. It is half4 for the standard
          * ABI and moves beyond the main shader footprint when HAL200 native
          * texture/sampler pointers occupy halves0..7. */
         const uint16_t blend_base = linked->g17p_blend_constant_base
                                        ? linked->g17p_blend_constant_base
                                        : AGX_ABI_FUNI_BLEND_R;
         if (!root_ptr ||
             !hk_agx3_g17p_fragment_blend_constants_range(
                blend_base,
                root_ptr + hk_root_descriptor_offset(draw.blend_constant),
                &ranges[count]))
            return hk_g17p_graphics_ranges_refused(range_count, count);
         sources[count] = HK_G17P_RANGE_SOURCE_ROOT;
         count++;
      }
   } else {
      return hk_g17p_graphics_ranges_refused(range_count, count);
   }

   if (s->info.agx3_combined_sampler_valid) {
      if (s->info.stage != MESA_SHADER_FRAGMENT || s->info.set_count != 0)
         return hk_g17p_graphics_ranges_refused(range_count, count);

      struct hk_descriptor_state *descriptors =
         &cmd->state.gfx.descriptors;
      struct hk_descriptor_set *set = descriptors->sets[0];
      const void *set_map = set ? set->mapped_ptr : NULL;
      uint32_t set_size = set ? set->size : 0;
      uint64_t set_addr = set ? hk_descriptor_set_addr(set) : 0;
      if (!set && descriptors->push[0] && descriptors->root.sets[0]) {
         set_map = descriptors->push[0]->data;
         set_size = sizeof(descriptors->push[0]->data);
         set_addr = descriptors->root.sets[0];
      }
      const struct agx_g17p_combined_sampler_binding *binding =
         &s->info.agx3_combined_sampler;
      if (!set_map || !set_addr ||
          binding->texture_offset_B > set_size ||
          set_size - binding->texture_offset_B < 32 ||
          binding->sampler_index_offset_B > set_size ||
          set_size - binding->sampler_index_offset_B < sizeof(uint16_t))
         return hk_g17p_graphics_ranges_refused(range_count, count);

      uint16_t sampler_index;
      memcpy(&sampler_index,
             (const uint8_t *)set_map +
                binding->sampler_index_offset_B,
             sizeof(sampler_index));

      const unsigned sampler_element_size =
         agx_chip_sampler_heap_element_size(dev->dev.chip);
      if (!dev->samplers.table.bo || !dev->samplers.table.bo->va ||
          dev->samplers.table.alloc > UINT16_MAX ||
          sampler_element_size > UINT16_MAX)
         return hk_g17p_graphics_ranges_refused(range_count, count);

      const struct hk_agx3_g17p_combined_sampler_heap heap = {
         .descriptor_set_va = set_addr,
         .sampler_heap_va = dev->samplers.table.bo->va->addr,
         .texture_offset_B = binding->texture_offset_B,
         .sampler_index = sampler_index,
         .sampler_count = dev->samplers.table.alloc,
         .sampler_element_size_B = sampler_element_size,
      };
      uint64_t pointers[2];
      if (!hk_agx3_g17p_combined_sampler_pointers(&heap, pointers))
         return hk_g17p_graphics_ranges_refused(range_count, count);

      /* HAL200 FragmentProgramVariant::finalize (374968..3749c0) maps
       * compiler bindpoints 0x31/0x35 to driver qwords +0xc0/+0xe0.
       * encodeAndEmitRenderState writes the raw texture/sampler table VAs
       * there (4c5580/4c5598); ESL copies them to uniform dwords 0..3.
       * Keep this upload immutable for the command-buffer lifetime, just
       * like the per-draw graphics state. Both shader indices are zero.
       */
      const unsigned cache_index =
         ((pointers[0] >> 4) ^ (pointers[0] >> 32) ^
          (pointers[1] >> 4) ^ (pointers[1] >> 32)) &
         (ARRAY_SIZE(cmd->state.gfx.g17p_combined_sampler_pointers) - 1);
      __typeof__(cmd->state.gfx.g17p_combined_sampler_pointers[0]) *cached =
         &cmd->state.gfx.g17p_combined_sampler_pointers[cache_index];
      uint64_t pointers_va;
      if (cached->valid && !memcmp(cached->pointers, pointers,
                                   sizeof(pointers))) {
         pointers_va = cached->address;
      } else {
         /* The cache stays keyed on the set address; what is published is
          * the 32-byte-aligned texture state (hk_g17p_texture_state_pointer). */
         const uint64_t published[2] = {
            hk_g17p_texture_state_pointer(
               cmd, pointers[0],
               (const uint8_t *)set_map + binding->texture_offset_B,
               set_size - binding->texture_offset_B),
            pointers[1],
         };
         pointers_va = published[0]
            ? hk_pool_upload(cmd, published, sizeof(published), 8)
            : 0;
         if (pointers_va) {
            memcpy(cached->pointers, pointers, sizeof(pointers));
            memcpy(cached->published, published, sizeof(published));
            cached->address = pointers_va;
            cached->valid = true;
         }
      }
      if (!pointers_va)
         return hk_g17p_graphics_ranges_refused(range_count, count);

      if (hk_g17p_launch_trace_enabled()) {
         fprintf(stderr,
                 "G17P combined sampler: set=%" PRIx64 " size=%u "
                 "texture_off=%u sampler_off=%u sampler_index=%u "
                 "sampler_count=%u sampler_stride=%u pointers=%" PRIx64
                 "/%" PRIx64 "\n",
                 set_addr, set_size,
                 binding->texture_offset_B,
                 binding->sampler_index_offset_B, sampler_index,
                 heap.sampler_count, heap.sampler_element_size_B,
                 pointers[0], pointers[1]);
         hk_g17p_trace_line(
            "combined-texture-desc",
            set_addr + binding->texture_offset_B,
            (const uint8_t *)set_map + binding->texture_offset_B,
            32);
         hk_g17p_trace_line("combined-pointers", pointers_va,
                            cached->published, sizeof(cached->published));
      }

      sources[count] = HK_G17P_RANGE_SOURCE_UPLOADED;
      ranges[count++] = (struct agx3_ioto_uniform_range){
         .start_halfs = 0,
         .size_halfs = 8,
         .address = pointers_va,
      };
   }

   if (native_spills && s->info.agx3_combined_sampler_valid)
      return hk_g17p_graphics_ranges_refused(range_count, count);

   if (s->info.agx3_indexed_sampler_valid || native_spills) {
      if ((s->b.info.stage != MESA_SHADER_FRAGMENT &&
           s->b.info.stage != MESA_SHADER_VERTEX) ||
          !dev->sampled_images.bo || !dev->samplers.table.bo ||
          !dev->rodata.indexed_heap_pointers)
         return hk_g17p_graphics_ranges_refused(range_count, count);
      const uint64_t pointers[2] = {
         dev->sampled_images.bo->va->addr,
         dev->samplers.table.bo->va->addr,
      };
      const uint64_t pointers_va = dev->rodata.indexed_heap_pointers;
      if (hk_g17p_launch_trace_enabled()) {
         fprintf(stderr,
                 "G17P indexed sampler: texture_heap=%" PRIx64
                 " texture_alloc=%u texture_next=%u sampler_heap=%" PRIx64
                 " sampler_alloc=%u sampler_next=%u pointers=%" PRIx64 "\n",
                 pointers[0], dev->sampled_images.alloc,
                 dev->sampled_images.next_desc, pointers[1],
                 dev->samplers.table.alloc, dev->samplers.table.next_desc,
                 pointers_va);
         hk_g17p_trace_line("indexed-pointers", pointers_va, pointers,
                            sizeof(pointers));
         if (dev->sampled_images.map && dev->sampled_images.next_desc)
            hk_g17p_trace_line(
               "indexed-texture-heap", pointers[0], dev->sampled_images.map,
               MIN2((size_t)dev->sampled_images.next_desc *
                        dev->sampled_images.desc_size,
                    (size_t)4096));
         if (dev->samplers.table.map && dev->samplers.table.next_desc)
            hk_g17p_trace_line(
               "indexed-sampler-heap", pointers[1], dev->samplers.table.map,
               MIN2((size_t)dev->samplers.table.next_desc *
                        dev->samplers.table.desc_size,
                    (size_t)4096));
      }
      sources[count] = HK_G17P_RANGE_SOURCE_CONSTANT;
      ranges[count++] = (struct agx3_ioto_uniform_range){
         .start_halfs = 0, .size_halfs = 8, .address = pointers_va,
      };
   }

   if (hk_agx3_g17p_graphics_needs_root(
          s->info.agx3_uses_root, linked->g17p_prolog_uses_root,
          s->info.set_count)) {
      if (!root_ptr)
         return hk_g17p_graphics_ranges_refused(range_count, count);

      if (!hk_agx3_g17p_root_range(root_uniform, s->info.set_count, root_ptr,
                                   &ranges[count]))
         return hk_g17p_graphics_ranges_refused(range_count, count);
      sources[count] = HK_G17P_RANGE_SOURCE_ROOT;
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
    * that BO. G17P loads dwords; an odd final half is covered by the zeroed
    * alignment padding before the shader code.
    */
   if (count < HK_AGX3_G17P_GRAPHICS_MAX_RANGES && s->b.info.rodata.size_16 != 0) {
      const struct agx_rodata *rodata = &s->b.info.rodata;
      if (!s->bo || !s->bo->va ||
          !hk_agx3_g17p_rodata_range(
             rodata->base_uniform, rodata->size_16, s->bo->va->addr,
             rodata->offset, s->bo->size, &ranges[count]))
         return hk_g17p_graphics_ranges_refused(range_count, count);
      sources[count] = HK_G17P_RANGE_SOURCE_CONSTANT;
      count++;
   } else if (s->b.info.rodata.size_16 != 0) {
      /* No room left in the program's range budget. */
      return hk_g17p_graphics_ranges_refused(range_count, count);
   }

   /* AGX_G17P_UNIFORM_DMA (v51a): the compiler replaced constant-index loads
    * through one pointer uniform (root table or descriptor set) with reads
    * of one uniform span. Publish that span from the pointer's current value;
    * the Ioto program reads it at launch, as the loads did. */
   if (s->b.info.agx3_uniform_dma_valid) {
      const unsigned pointer = s->b.info.agx3_uniform_dma_pointer;
      uint64_t base = 0;
      if (pointer == root_uniform) {
         base = root_ptr;
         if (!base)
            return hk_g17p_graphics_ranges_refused(range_count, count);
      } else if (pointer >= root_uniform + 4 &&
                 (pointer - root_uniform - 4) % 4 == 0 &&
                 (pointer - root_uniform - 4) / 4 < HK_MAX_SETS) {
         base = cmd->state.gfx.descriptors.root.sets[
            (pointer - root_uniform - 4) / 4];
      } else {
         return hk_g17p_graphics_ranges_refused(range_count, count);
      }
      /* A set the draw never binds is only read on statically dead paths
       * (reading it is invalid usage); give the launch a mapped zero page
       * instead of a fault. */
      const uint64_t address = base ? base + s->b.info.agx3_uniform_dma_offset_B
                                    : AGX_ZERO_PAGE_ADDRESS;
      if (count >= HK_AGX3_G17P_GRAPHICS_MAX_RANGES || (address & 3) ||
          (s->b.info.agx3_uniform_dma_dest & 1) ||
          (s->b.info.agx3_uniform_dma_halfs & 1))
         return hk_g17p_graphics_ranges_refused(range_count, count);
      /* A root-relative span follows the root. A set-pointer span is the
       * set's address value: constant while hk_g17p_dma_set_base() is
       * unchanged, which every reuse of a published launcher requires. */
      sources[count] = pointer == root_uniform ? HK_G17P_RANGE_SOURCE_ROOT
                                               : HK_G17P_RANGE_SOURCE_CONSTANT;
      ranges[count++] = (struct agx3_ioto_uniform_range){
         .start_halfs = s->b.info.agx3_uniform_dma_dest,
         .size_halfs = s->b.info.agx3_uniform_dma_halfs,
         .address = address,
      };
   }

   *range_count = count;
   return count <= HK_AGX3_G17P_GRAPHICS_MAX_RANGES;
}

/* Convert the legacy absolute-range description into the compact-root ABI.
 * The root contains source GPU addresses, never CPU-copied payload values.
 * The linked-shader setup loads each address from this record and dereferences
 * it at execution time, so GPU-authored indirect draw parameters remain live. */
static bool
hk_g17p_graphics_compact_ranges(
   struct agx3_ioto_uniform_range *ranges, size_t range_count,
   struct hk_g17p_graphics_root_record *root)
{
   if (!ranges || !root ||
       range_count > HK_AGX3_G17P_GRAPHICS_MAX_RANGES)
      return false;

   *root = (struct hk_g17p_graphics_root_record){0};
   root->version = HK_G17P_GRAPHICS_ROOT_VERSION;
   root->size_B = sizeof(*root);
   root->range_count = range_count;

   for (size_t i = 0; i < range_count; ++i) {
      struct agx3_ioto_uniform_range *range = &ranges[i];
      if (!range->address || !range->size_halfs || (range->size_halfs & 1))
         return false;

      root->range_address[i] = range->address;
      root->valid |= HK_G17P_GRAPHICS_ROOT_RANGE0 << i;
      range->address =
         offsetof(struct hk_g17p_graphics_root_record, range_address) +
         i * sizeof(root->range_address[0]);
   }

   return true;
}

struct hk_g17p_graphics_ioto_key {
   uint64_t shader_va;
   uint64_t constant_program_va;
   uint64_t constant_program_size;
   uint64_t range_address[HK_AGX3_G17P_GRAPHICS_MAX_RANGES];
   uint32_t shader_binary_size;
   uint32_t scratch_size;
   uint32_t compiler_d;
   uint16_t range_start_halfs[HK_AGX3_G17P_GRAPHICS_MAX_RANGES];
   uint16_t range_size_halfs[HK_AGX3_G17P_GRAPHICS_MAX_RANGES];
   uint16_t nr_gprs;
   uint16_t tile_size_px;
   uint8_t range_count;
   uint8_t chip;
   uint8_t compiler_d_valid;
   uint8_t fragment;
   uint8_t writes_sample_mask;
   uint8_t colour_classes;
   uint8_t sample_size_B;
   uint8_t nr_samples;
   uint8_t constant_program_flags;
   /* Optional tail: shaders without native inputs retain the original key
    * prefix and comparison size. */
   uint64_t cprog_input_halves[AGX3_CPROG_INPUT_WORDS];
   uint16_t cprog_input_count;
   uint16_t cprog_input_storage;
};

static unsigned
hk_g17p_graphics_ioto_key_init(
   enum agx_chip chip,
   const struct hk_agx3_g17p_graphics_program_desc *desc,
   struct hk_g17p_graphics_ioto_key *key)
{
   static_assert(sizeof(*key) <= HK_G17P_GRAPHICS_IOTO_KEY_MAX,
                 "graphics Ioto key storage");
   assert(desc->uniform_range_count <= HK_AGX3_G17P_GRAPHICS_MAX_RANGES);

   const unsigned size = desc->cprog_inputs && desc->cprog_inputs->count
      ? sizeof(*key) : offsetof(struct hk_g17p_graphics_ioto_key, cprog_input_halves);
   memset(key, 0, size);
   key->shader_va = desc->shader_va;
   /* The shader owns immutable code at this VA for the cache's lifetime. */
   key->constant_program_va = desc->constant_program.code_va;
   key->constant_program_size = desc->constant_program.code_size;
   if (desc->cprog_inputs && desc->cprog_inputs->count) {
      memcpy(key->cprog_input_halves, desc->cprog_inputs->halves,
             sizeof(key->cprog_input_halves));
      key->cprog_input_count = desc->cprog_inputs->count;
      key->cprog_input_storage = desc->cprog_inputs->storage_bytes;
   }
   key->constant_program_flags =
      desc->constant_program.force_external |
      (desc->constant_program.request_prelude << 1) |
      (desc->constant_program.suppress_prelude << 2) |
      (desc->constant_program.absolute_branch << 3);
   key->shader_binary_size = desc->shader_binary_size;
   key->scratch_size = desc->scratch_size;
   key->compiler_d = desc->compiler_d;
   key->nr_gprs = desc->nr_gprs;
   key->range_count = desc->uniform_range_count;
   key->chip = chip;
   key->compiler_d_valid = desc->compiler_d_valid;
   key->fragment = desc->fragment;
   key->writes_sample_mask = desc->writes_sample_mask;
   key->colour_classes = desc->colour_classes;
   /* The native vertex state tail is fixed and never consumes imageblock
    * geometry.  Do not split an otherwise identical vertex program merely
    * because the same shaders are reused across framebuffer layouts. */
   if (desc->fragment) {
      key->tile_size_px = desc->imageblock.tile_size_px;
      key->sample_size_B = desc->imageblock.sample_size_B;
      key->nr_samples = desc->imageblock.nr_samples;
   }

   for (size_t i = 0; i < desc->uniform_range_count; ++i) {
      key->range_address[i] = desc->uniform_ranges[i].address;
      key->range_start_halfs[i] = desc->uniform_ranges[i].start_halfs;
      key->range_size_halfs[i] = desc->uniform_ranges[i].size_halfs;
   }
   return size;
}

enum hk_g17p_graphics_ioto_cache_result {
   HK_G17P_GRAPHICS_IOTO_FAST_HIT,
   HK_G17P_GRAPHICS_IOTO_LAST_HIT,
   HK_G17P_GRAPHICS_IOTO_COMPACT_ROOT,
   HK_G17P_GRAPHICS_IOTO_TEMPLATE_HIT,
   HK_G17P_GRAPHICS_IOTO_GENERATED,
   HK_G17P_GRAPHICS_IOTO_RESULT_COUNT,
};

static inline void
hk_g17p_graphics_ioto_cache_stat(
   const struct hk_device *dev,
   enum hk_g17p_graphics_ioto_cache_result result)
{
   if (likely(!dev->g17p_diagnostics.graphics_ioto_cache_stats))
      return;

   static uint32_t counts[HK_G17P_GRAPHICS_IOTO_RESULT_COUNT];
   static uint32_t total;
   p_atomic_inc(&counts[result]);
   const uint32_t calls = p_atomic_inc_return(&total);

   if ((calls & 0x7fff) == 0) {
      fprintf(stderr,
              "HK_G17P_GRAPHICS_IOTO_CACHE calls=%u fast=%u last=%u "
              "compact=%u template=%u generated=%u\n",
              calls, p_atomic_read(&counts[HK_G17P_GRAPHICS_IOTO_FAST_HIT]),
              p_atomic_read(&counts[HK_G17P_GRAPHICS_IOTO_LAST_HIT]),
              p_atomic_read(&counts[HK_G17P_GRAPHICS_IOTO_COMPACT_ROOT]),
              p_atomic_read(&counts[HK_G17P_GRAPHICS_IOTO_TEMPLATE_HIT]),
              p_atomic_read(&counts[HK_G17P_GRAPHICS_IOTO_GENERATED]));
   }
}

struct hk_g17p_graphics_template_entry {
   struct hk_g17p_graphics_template_entry *next;
   struct agx3_cprog_inputs cprog_inputs;
   struct hk_agx3_g17p_graphics_program_desc desc;
   struct agx3_ioto_uniform_range
      ranges[HK_AGX3_G17P_GRAPHICS_MAX_RANGES];
   struct agx3_g17p_graphics_program_template program;
};

#define HK_G17P_FRAGMENT_TILE_CLASSES 3u
#define HK_G17P_FRAGMENT_SAMPLE_SIZE_CLASSES 32u
#define HK_G17P_FRAGMENT_SAMPLE_COUNT_CLASSES 4u
#define HK_G17P_FRAGMENT_DYNAMIC_CLASSES 4u
#define HK_G17P_FRAGMENT_SETUP_COUNT                                      \
   (HK_G17P_FRAGMENT_TILE_CLASSES * HK_G17P_FRAGMENT_SAMPLE_SIZE_CLASSES * \
    HK_G17P_FRAGMENT_SAMPLE_COUNT_CLASSES *                               \
    HK_G17P_FRAGMENT_DYNAMIC_CLASSES)

struct hk_g17p_graphics_setup {
   struct agx_bo *external_bo;
   uint64_t external_va;
   uint16_t external_length;
   struct agx3_g17p_graphics_external_launcher_template external_launcher;
   uint16_t length;
   uint8_t bytes[];
};

struct hk_g17p_fragment_setup_table {
   struct hk_g17p_graphics_setup *entries[HK_G17P_FRAGMENT_SETUP_COUNT];
};

static bool
hk_g17p_fragment_setup_index(const struct agx_ioto_g17p_imageblock *imageblock,
                             bool blend, bool spills, unsigned *index)
{
   unsigned tile;
   switch (imageblock->tile_size_px) {
   case 32 * 32: tile = 0; break;
   case 32 * 16: tile = 1; break;
   case 16 * 16: tile = 2; break;
   default: return false;
   }

   unsigned samples;
   switch (imageblock->nr_samples) {
   case 0: samples = 0; break;
   case 1: samples = 1; break;
   case 2: samples = 2; break;
   case 4: samples = 3; break;
   default: return false;
   }

   if ((imageblock->sample_size_B & 7) ||
       imageblock->sample_size_B / 8 >=
          HK_G17P_FRAGMENT_SAMPLE_SIZE_CLASSES)
      return false;

   const unsigned size = imageblock->sample_size_B / 8;
   const unsigned dynamic = (blend ? 1u : 0u) | (spills ? 2u : 0u);
   *index = (((tile * HK_G17P_FRAGMENT_SAMPLE_SIZE_CLASSES + size) *
              HK_G17P_FRAGMENT_SAMPLE_COUNT_CLASSES + samples) *
             HK_G17P_FRAGMENT_DYNAMIC_CLASSES) + dynamic;
   assert(*index < HK_G17P_FRAGMENT_SETUP_COUNT);
   return true;
}

static void
hk_g17p_graphics_setup_destroy(struct hk_device *dev,
                               struct hk_g17p_graphics_setup *setup)
{
   if (setup && setup->external_bo)
      agx_bo_unreference(&dev->dev, setup->external_bo);
   free(setup);
}

static struct hk_g17p_graphics_setup *
hk_g17p_graphics_setup_build(
   struct hk_device *dev,
   const struct hk_agx3_g17p_graphics_program_desc *source)
{
   struct hk_agx3_g17p_graphics_program_desc desc = *source;
   /* An immutable setup may move between inline and external allocations.
    * A relative constant-program branch would bake in the sentinel PC. */
   if (desc.constant_program.code_size &&
       !desc.constant_program.absolute_branch)
      return NULL;

   /* With an absent or absolute constant-program suffix, bytes do not depend
    * on program_va. Use a nonzero aligned sentinel for this owned template. */
   desc.program_va = AGX_IOTO_PROGRAM_ALIGNMENT;
   desc.preloaded_root = true;
   uint8_t bytes[HK_AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE];
   uint8_t external[HK_AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE];
   size_t written, external_written = 0;
   if (!hk_agx3_g17p_basic_graphics_program_write_unpublished(
           dev->dev.chip, &desc, bytes, sizeof(bytes), &written) ||
       written > UINT16_MAX) {
      return NULL;
   }

   const bool external_valid =
      hk_agx3_g17p_graphics_external_setup_suffix_generate(
         dev->dev.chip, &desc, external, sizeof(external),
         &external_written) &&
      external_written <= UINT16_MAX;

   struct hk_g17p_graphics_setup *setup =
      malloc(sizeof(*setup) + written);
   if (!setup)
      return NULL;

   setup->external_bo = NULL;
   setup->external_va = 0;
   setup->external_length = 0;
   if (external_valid) {
      setup->external_bo = agx_bo_create(
         &dev->dev, ALIGN_POT(external_written, 64), 0,
         AGX_BO_EXEC | AGX_BO_LOW_VA, "G17P persistent graphics setup");
      if (setup->external_bo) {
         void *external_map = agx_bo_map(setup->external_bo);
         if (external_map) {
            memcpy(external_map, external, external_written);
            memset((uint8_t *)external_map + external_written, 0,
                   ALIGN_POT(external_written, 64) - external_written);
            if (!agx_bo_sync_for_device(
                   setup->external_bo, 0, ALIGN_POT(external_written, 64))) {
               setup->external_va = setup->external_bo->va->addr;
               setup->external_length = external_written;
               if (!hk_agx3_g17p_graphics_external_launcher_template_generate(
                      dev->dev.chip, &desc, setup->external_va,
                      &setup->external_launcher)) {
                  setup->external_va = 0;
                  setup->external_length = 0;
               }
            }
         }
         if (!setup->external_va) {
            agx_bo_unreference(&dev->dev, setup->external_bo);
            setup->external_bo = NULL;
         }
      }
   }

   setup->length = written;
   memcpy(setup->bytes, bytes, written);
   return setup;
}

static struct hk_g17p_graphics_setup *
hk_g17p_graphics_setup_get(
   struct hk_device *dev, struct hk_linked_shader *linked,
   const struct hk_agx3_g17p_graphics_program_desc *desc,
   bool blend, bool spills)
{
   struct hk_g17p_graphics_setup **slot;
   if (!desc->fragment) {
      slot = (struct hk_g17p_graphics_setup **)
         &linked->g17p_graphics_vertex_setup;
   } else {
      unsigned index;
      if (!hk_g17p_fragment_setup_index(&desc->imageblock, blend, spills,
                                         &index))
         return NULL;

      struct hk_g17p_fragment_setup_table *table =
         p_atomic_read(&linked->g17p_graphics_fragment_setups);
      if (!table) {
         struct hk_g17p_fragment_setup_table *candidate =
            calloc(1, sizeof(*candidate));
         if (!candidate)
            return NULL;
         void *old = p_atomic_cmpxchg_ptr(
            &linked->g17p_graphics_fragment_setups, NULL, candidate);
         if (old) {
            free(candidate);
            table = old;
         } else {
            table = candidate;
         }
      }
      slot = &table->entries[index];
   }

   struct hk_g17p_graphics_setup *setup = p_atomic_read(slot);
   if (setup)
      return setup;

   struct hk_g17p_graphics_setup *candidate =
      hk_g17p_graphics_setup_build(dev, desc);
   if (!candidate)
      return NULL;

   void *old = p_atomic_cmpxchg_ptr(slot, NULL, candidate);
   if (old) {
      hk_g17p_graphics_setup_destroy(dev, candidate);
      return old;
   }
   return candidate;
}

void
hk_g17p_linked_graphics_cache_finish(struct hk_device *dev,
                                     struct hk_linked_shader *linked)
{
   struct hk_g17p_graphics_template_entry *entry =
      p_atomic_read(&linked->g17p_graphics_templates);
   while (entry) {
      struct hk_g17p_graphics_template_entry *next = entry->next;
      free(entry);
      entry = next;
   }
   linked->g17p_graphics_templates = NULL;

   hk_g17p_graphics_setup_destroy(
      dev, p_atomic_read(&linked->g17p_graphics_vertex_setup));
   linked->g17p_graphics_vertex_setup = NULL;

   struct hk_g17p_fragment_setup_table *table =
      p_atomic_read(&linked->g17p_graphics_fragment_setups);
   if (table) {
      for (unsigned i = 0; i < ARRAY_SIZE(table->entries); ++i)
         hk_g17p_graphics_setup_destroy(dev, p_atomic_read(&table->entries[i]));
      free(table);
   }
   linked->g17p_graphics_fragment_setups = NULL;
}

static bool
hk_g17p_graphics_template_lookup(
   struct hk_linked_shader *linked,
   const struct hk_agx3_g17p_graphics_program_desc *desc, void *output,
   size_t capacity, uint16_t *written,
   const struct hk_g17p_graphics_template_entry **used)
{
   struct hk_g17p_graphics_template_entry *entry =
      p_atomic_read(&linked->g17p_graphics_templates);
   for (; entry; entry = entry->next) {
      if (hk_agx3_g17p_graphics_program_template_relocate(
             AGX_CHIP_G17P, &entry->desc, desc, &entry->program, output,
             capacity, written)) {
         if (used)
            *used = entry;
         return true;
      }
   }
   return false;
}

static ATTRIBUTE_NOINLINE bool
hk_g17p_graphics_template_generate(
   struct hk_linked_shader *linked,
   const struct hk_agx3_g17p_graphics_program_desc *desc, void *output,
   size_t capacity, uint16_t *written,
   const struct hk_g17p_graphics_template_entry **used)
{
   if (desc->uniform_range_count > HK_AGX3_G17P_GRAPHICS_MAX_RANGES)
      return false;

   struct hk_g17p_graphics_template_entry *candidate =
      calloc(1, sizeof(*candidate));
   if (!candidate)
      return false;

   candidate->desc = *desc;
   if (desc->cprog_inputs) {
      candidate->cprog_inputs = *desc->cprog_inputs;
      candidate->desc.cprog_inputs = &candidate->cprog_inputs;
   }
   memcpy(candidate->ranges, desc->uniform_ranges,
          desc->uniform_range_count * sizeof(candidate->ranges[0]));
   candidate->desc.uniform_ranges = candidate->ranges;
   if (!hk_agx3_g17p_graphics_program_template_generate(
          AGX_CHIP_G17P, &candidate->desc, &candidate->program) ||
       candidate->program.program.length > capacity) {
      free(candidate);
      return false;
   }

   while (true) {
      struct hk_g17p_graphics_template_entry *head =
         p_atomic_read(&linked->g17p_graphics_templates);
      for (struct hk_g17p_graphics_template_entry *entry = head; entry;
           entry = entry->next) {
         if (hk_agx3_g17p_graphics_program_template_relocate(
                AGX_CHIP_G17P, &entry->desc, desc, &entry->program, output,
                capacity, written)) {
            free(candidate);
            if (used)
               *used = entry;
            return true;
         }
      }

      candidate->next = head;
      if (p_atomic_cmpxchg_ptr(&linked->g17p_graphics_templates, head,
                              candidate) == head)
         break;
   }

   memcpy(output, candidate->program.program.bytes,
          candidate->program.program.length);
   *written = candidate->program.program.length;
   if (used)
      *used = candidate;
   return true;
}

/* Publish one compact-root G17P graphics launcher: allocate the draw's
 * immutable root record, reserve and relocate the launcher against it,
 * publish, and retain every BO the launcher names.  Returns 0 when the caller
 * must continue with the non-compact fallback (exactly the cases in which the
 * inline code this was extracted from jumped to compact_fallback). */
static uint64_t
hk_g17p_publish_compact_graphics(
   struct hk_cmd_buffer *cmd, struct hk_shader *s,
   struct hk_linked_shader *linked,
   const struct hk_g17p_graphics_setup *setup,
   const struct hk_g17p_graphics_root_record *compact_root,
   const struct agx3_ioto_uniform_range *compact_ranges, size_t range_count,
   uint64_t linked_shader_va, bool fragment,
   struct agx_ptr *prepared_compact_root, int retained_external,
   bool *external_out)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct agx_ptr compact_root_ptr =
      hk_pool_alloc(cmd, sizeof(*compact_root), 64);
   if (compact_root_ptr.cpu)
      memcpy(compact_root_ptr.cpu, compact_root, sizeof(*compact_root));
   const uint64_t compact_root_va = compact_root_ptr.gpu;
   if (dev->g17p_diagnostics.compact_ioto_trace) {
      fprintf(stderr,
              "G17P compact Ioto: stage=%s shader=%" PRIx64
              " root=%" PRIx64 " setup_len=%u external=%" PRIx64
              "/%u ranges=%zu "
              "valid=%" PRIx64 "\n",
              fragment ? "fragment" : "vertex", linked_shader_va,
              compact_root_va, setup->length, setup->external_va,
              setup->external_length, range_count, compact_root->valid);
      for (size_t r = 0; r < range_count; ++r) {
         fprintf(stderr,
                 "G17P compact range[%zu]: start_halfs=%u "
                 "size_halfs=%u address=%" PRIx64 "\n",
                 r, (unsigned)compact_ranges[r].start_halfs,
                 (unsigned)compact_ranges[r].size_halfs,
                 compact_ranges[r].address);
      }
      hk_g17p_trace_line("compact-root", compact_root_va, compact_root,
                         sizeof(*compact_root));
      hk_g17p_trace_line("compact-template", 0, setup->bytes, setup->length);
      if (setup->external_bo)
         hk_g17p_trace_line("compact-external-setup", setup->external_va,
                            agx_bo_map(setup->external_bo),
                            setup->external_length);
   }
   struct hk_ioto_reservation reservation;
   const size_t launcher_capacity = MAX2(
      (size_t)HK_AGX3_G17P_GRAPHICS_EXTERNAL_LAUNCHER_MAX_SIZE,
      (size_t)AGX3_G17P_GRAPHICS_ROOT_PREFIX_MAX_SIZE + setup->length);
   if (!compact_root_va ||
       !hk_cmd_reserve_ioto_program(cmd, launcher_capacity,
                                    "G17P compact graphics launcher",
                                    &reservation))
      return 0;

   size_t program_length = 0;
   bool external = setup->external_bo &&
      hk_agx3_g17p_graphics_external_launcher_template_relocate(
         dev->dev.chip, &setup->external_launcher, reservation.va,
         compact_root_va, setup->external_va, reservation.map,
         launcher_capacity, &program_length);
   if (!external) {
      struct agx3_g17p_graphics_root_prefix prefix;
      if (!hk_agx3_g17p_graphics_root_prefix_generate(
             dev->dev.chip, compact_root_va, &prefix))
         return 0;
      memcpy(reservation.map, prefix.bytes, prefix.length);
      memcpy(reservation.map + prefix.length, setup->bytes, setup->length);
      program_length = prefix.length + setup->length;
   }
   if (dev->g17p_diagnostics.compact_ioto_trace)
      hk_g17p_trace_line(external ? "compact-external-launcher"
                                  : "compact-inline-program",
                         reservation.va, reservation.map, program_length);
   const uint64_t published =
      hk_cmd_publish_ioto_program(cmd, &reservation, program_length);
   if (!published)
      return 0;

   if (prepared_compact_root)
      *prepared_compact_root = compact_root_ptr;
   if (external_out)
      *external_out = external;
   /* retained_external >= 0: an earlier publication in this recording, with
    * the same linked shader, shader and setup, already retained the linked
    * and shader BOs, and the setup BO when retained_external is 1.  Retaining
    * them again appends nothing to the command buffer's BO list. */
   if (retained_external >= 0 && (!external || retained_external))
      return published;
   hk_cmd_buffer_retain_bo(cmd, linked->b.bo);
   if (external)
      hk_cmd_buffer_retain_bo(cmd, setup->external_bo);
   if (s->b.info.rodata.size_16 != 0 || s->b.info.has_preamble)
      hk_cmd_buffer_retain_bo(cmd, s->bo);
   return published;
}

/* A uniform-range source address as hk_g17p_graphics_compact_ranges() and
 * agx3_ioto_uniform_program_body_bound() accept it: nonzero, dword aligned,
 * and not wrapping over its loaded bytes. */
static inline bool
hk_g17p_compact_range_address_ok(uint64_t address, unsigned size_halfs)
{
   const uint64_t bytes = (uint64_t)(size_halfs / 2) * 4;
   return address != 0 && (address & 3) == 0 &&
          address <= UINT64_MAX - (bytes - 1);
}

/*
 * HK_G17P_MDI_ONE_LEVEL (default on; HK_G17P_MDI_ONE_LEVEL=0 restores the
 * compact two-level launcher).  Qualified by a Minecraft ABBA on hardware
 * (mc-ta, +2.3% FPS over 4 sessions, 0 faults).
 *
 * Every record of an indexed-indirect multi-draw whose vertex shader reads
 * per-draw parameters (gl_BaseInstance/gl_BaseVertex, per-instance vertex
 * attributes, gl_DrawID) republishes its vertex program: the record's draw
 * parameters live at the record's own indirect address, so the vertex setup
 * program that loads them into uniforms differs per record.
 *
 * The production compact launcher (hk_g17p_publish_compact_graphics()) makes
 * that per-record publication cheap on the CPU -- an 80-byte root of source
 * addresses plus a 118-byte relocated launcher -- but its shared external
 * setup chases every uniform range through that root, and every LDDEV
 * (pointer, then each 4-dword payload chunk) is followed by its own 4-NOP
 * drain.  For Minecraft's terrain VS (five ranges) that is 13 serialized
 * device-load round trips at the head of every record, on the TA critical
 * path (offline decode, mc-ta/decode/vext_mcenv.dis).  The linked-shader
 * template path that the compact launcher replaced bakes the same absolute
 * range addresses into LDIMMs and issues all 8 LDDEVs before a single drain
 * (mc-ta/decode/ol_mc.dis): one round trip.  Native Metal's vertex ESL has
 * that one-level shape too (agx3-footguns K.1).
 *
 * With the knob on, indexed-indirect records (hk_g17p_emit_indirect_record)
 * publish that one-level template instead.  The uniform values are loaded at
 * execution time from exactly the same source addresses (root fields,
 * device constants, the record's own indirect parameters), so GPU-written
 * indirect buffers and drawIndirectCount predication keep their semantics.
 * To keep the CPU cost per record close to the compact path, the next record
 * of the same linked shader relocates the same template entry with only the
 * draw-parameter/draw-ID addresses re-derived, exactly as the compact patch
 * path re-derives its root record.
 */
bool
hk_g17p_mdi_one_level_enabled(void)
{
   return HK_ENV_BOOL_ONCE("HK_G17P_MDI_ONE_LEVEL", true);
}

/* The setup composer shares one LDIMM address base between ranges within
 * INT16_MAX bytes of each other (agx3_graphics_program.c,
 * graphics_ranges_close()).  A patched range must own its base: it may be
 * close to no other range, in the template and in every patched record. */
static bool
hk_g17p_one_level_range_isolated(const struct agx3_ioto_uniform_range *ranges,
                                 unsigned count, unsigned index)
{
   const uint64_t a = ranges[index].address;
   const uint64_t a_end = a + (uint64_t)ranges[index].size_halfs * 2;
   for (unsigned r = 0; r < count; ++r) {
      if (r == index)
         continue;
      const uint64_t b = ranges[r].address;
      const uint64_t b_end = b + (uint64_t)ranges[r].size_halfs * 2;
      const uint64_t gap = a_end <= b ? b - a_end
                           : b_end <= a ? a - b_end
                                        : 0;
      if (gap <= (uint64_t)INT16_MAX + 8)
         return false;
   }
   return true;
}

/* HK_G17P_MDI_ONE_LEVEL_VERIFY=1 (diagnostic, CPU-heavy): compose each
 * one-level record program again with the uncached reference composer (the
 * HK_G17P_LAUNCH_TRACE / fail-closed fallback path) for the same descriptor
 * and VA, and compare bytes. */
static unsigned hk_g17p_one_level_verify_calls;
static unsigned hk_g17p_one_level_verify_mismatches;

static void
hk_g17p_one_level_verify_report(void)
{
   fprintf(stderr, "G17P MDI one-level verify: checked=%u mismatched=%u\n",
           p_atomic_read(&hk_g17p_one_level_verify_calls),
           p_atomic_read(&hk_g17p_one_level_verify_mismatches));
}

static void
hk_g17p_one_level_verify(struct hk_device *dev,
                         const struct hk_agx3_g17p_graphics_program_desc *desc,
                         const void *published, uint16_t length,
                         const char *form)
{
   if (!HK_ENV_BOOL_ONCE("HK_G17P_MDI_ONE_LEVEL_VERIFY", false))
      return;

   if (p_atomic_inc_return(&hk_g17p_one_level_verify_calls) == 1)
      atexit(hk_g17p_one_level_verify_report);
   struct hk_agx3_g17p_graphics_program *reference =
      malloc(sizeof(*reference));
   const bool ok = reference &&
      hk_agx3_g17p_basic_graphics_program_generate(dev->dev.chip, desc,
                                                   reference) &&
      reference->length == length &&
      !memcmp(reference->bytes, published, length);
   if (!ok) {
      p_atomic_inc(&hk_g17p_one_level_verify_mismatches);
      mesa_loge("G17P MDI one-level verify MISMATCH (%s): program=%" PRIx64
                " length=%u reference=%u",
                form, desc->program_va, (unsigned)length,
                reference ? (unsigned)reference->length : 0);
   }
   free(reference);
}

static uint64_t
hk_g17p_publish_one_level_record(
   struct hk_cmd_buffer *cmd, struct hk_shader *s,
   struct hk_linked_shader *linked,
   struct hk_g17p_graphics_ioto_cache_entry *cached,
   uint64_t linked_shader_va, uint64_t rodata_va, uint64_t dma_set_base,
   const struct agx_tilebuffer_layout *tib)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   const __typeof__(cached->one_level) *one = &cached->one_level;

   if (!one->valid || one->source_shader != s || one->linked != linked ||
       one->linked_shader_va != linked_shader_va ||
       one->rodata_va != rodata_va || one->root != gfx->root ||
       one->dma_set_base != dma_set_base ||
       one->tile_size_px != tib->tile_size ||
       one->sample_size_B != tib->sample_size_B ||
       one->nr_samples != tib->nr_samples ||
       one->has_draw_params != (gfx->draw_params != 0) ||
       one->has_draw_id != (gfx->draw_id_ptr != 0) || !one->entry ||
       one->range_count > HK_AGX3_G17P_GRAPHICS_MAX_RANGES)
      return 0;

   const struct hk_g17p_graphics_template_entry *entry = one->entry;
   const uint64_t base[4] = {
      [HK_G17P_RANGE_SOURCE_CONSTANT] = 0,
      [HK_G17P_RANGE_SOURCE_ROOT] = gfx->root,
      [HK_G17P_RANGE_SOURCE_DRAW_PARAMS] = gfx->draw_params,
      [HK_G17P_RANGE_SOURCE_DRAW_ID] = gfx->draw_id_ptr,
   };
   struct agx3_ioto_uniform_range ranges[HK_AGX3_G17P_GRAPHICS_MAX_RANGES];
   for (unsigned r = 0; r < one->range_count; ++r) {
      if (one->source[r] > HK_G17P_RANGE_SOURCE_DRAW_ID)
         return 0;
      ranges[r] = one->ranges[r];
      ranges[r].address = base[one->source[r]] + one->source_offset[r];
      if (!hk_g17p_compact_range_address_ok(ranges[r].address,
                                            ranges[r].size_halfs))
         return 0;
   }

   struct hk_ioto_reservation reservation;
   if (one->patch) {
      /* Rewrite only the LDIMM pairs that address the draw-parameter /
       * draw-ID ranges; every other byte is the template's own (the root,
       * constants and linked shader are part of the identity above). */
      uint8_t patch[ARRAY_SIZE(one->sites)][2][AGX_IOTO_LOAD_LDIMM_MAX_SIZE];
      bool ok = one->length != 0 && one->length <= sizeof(one->bytes);
      for (unsigned i = 0; ok && i < one->site_count; ++i) {
         const __typeof__(one->sites[0]) *site = &one->sites[i];
         const uint64_t target = ranges[site->range].address + site->delta;
         uint8_t low_length = 0, high_length = 0;
         ok = site->range < one->range_count &&
              agx_ioto_load_pack_ldimm(dev->dev.chip, site->dest_half,
                                       (uint32_t)target, patch[i][0],
                                       &low_length) &&
              agx_ioto_load_pack_ldimm(dev->dev.chip, site->dest_half + 2,
                                       (uint32_t)(target >> 32), patch[i][1],
                                       &high_length) &&
              low_length == site->low_length &&
              high_length == site->high_length &&
              (size_t)site->offset + low_length + high_length <= one->length;
      }
      for (unsigned i = 0; ok && i < one->site_count; ++i)
         ok = hk_g17p_one_level_range_isolated(ranges, one->range_count,
                                               one->sites[i].range);
      if (ok) {
         if (!hk_cmd_reserve_ioto_program(cmd, one->length,
                                          "G17P MDI one-level vertex setup",
                                          &reservation))
            return 0;
         if (!agx_va_in_usc_window(&dev->dev, reservation.va))
            return 0;
         uint8_t *map = reservation.map;
         memcpy(map, one->bytes, one->length);
         for (unsigned i = 0; i < one->site_count; ++i) {
            const __typeof__(one->sites[0]) *site = &one->sites[i];
            memcpy(map + site->offset, patch[i][0], site->low_length);
            memcpy(map + site->offset + site->low_length, patch[i][1],
                   site->high_length);
         }
         {
            struct hk_agx3_g17p_graphics_program_desc check = entry->desc;
            check.uniform_ranges = ranges;
            check.uniform_range_count = one->range_count;
            check.program_va = reservation.va;
            hk_g17p_one_level_verify(dev, &check, map, one->length, "patch");
         }
         const uint64_t published =
            hk_cmd_publish_ioto_program(cmd, &reservation, one->length);
         if (published && dev->g17p_diagnostics.compact_ioto_trace) {
            fprintf(stderr,
                    "G17P one-level record: shader=%" PRIx64
                    " program=%" PRIx64 " len=%u draw_params=%" PRIx64
                    " draw_id=%" PRIx64 " patched=%u\n",
                    linked_shader_va, published, (unsigned)one->length,
                    gfx->draw_params, gfx->draw_id_ptr, one->site_count);
            hk_g17p_trace_line("one-level-program", published,
                               reservation.map, one->length);
         }
         return published;
      }
   }

   if (!hk_cmd_reserve_ioto_program(cmd, one->capacity,
                                    "G17P MDI one-level vertex setup",
                                    &reservation))
      return 0;
   if (!agx_va_in_usc_window(&dev->dev, reservation.va))
      return 0;

   struct hk_agx3_g17p_graphics_program_desc target = entry->desc;
   target.uniform_ranges = ranges;
   target.uniform_range_count = one->range_count;
   target.program_va = reservation.va;
   uint16_t length = 0;
   if (!hk_agx3_g17p_graphics_program_template_relocate(
          dev->dev.chip, &entry->desc, &target, &entry->program,
          reservation.map, one->capacity, &length) ||
       length == 0 || length > one->capacity)
      return 0;
   hk_g17p_one_level_verify(dev, &target, reservation.map, length,
                            "relocate");

   const uint64_t published =
      hk_cmd_publish_ioto_program(cmd, &reservation, length);
   if (published && dev->g17p_diagnostics.compact_ioto_trace) {
      fprintf(stderr,
              "G17P one-level record: shader=%" PRIx64 " program=%" PRIx64
              " len=%u draw_params=%" PRIx64 " draw_id=%" PRIx64 "\n",
              linked_shader_va, published, (unsigned)length,
              gfx->draw_params, gfx->draw_id_ptr);
      hk_g17p_trace_line("one-level-program", published, reservation.map,
                         length);
   }
   return published;
}

/* agx_usc_uniform() publishes at most 64 halves per record. */
static void
hk_usc_uniform_split(struct agx_usc_builder *b, unsigned start_halfs,
                     unsigned size_halfs, uint64_t buffer)
{
   for (unsigned offset = 0; offset < size_halfs; offset += 64) {
      agx_usc_uniform(b, start_halfs + offset, MIN2(64, size_halfs - offset),
                      buffer + offset * 2);
   }
}

static uint64_t
hk_upload_usc_words_internal(struct hk_cmd_buffer *cmd, struct hk_shader *s,
                             struct hk_linked_shader *linked,
                             struct agx_ptr *prepared_compact_root)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   if (prepared_compact_root)
      *prepared_compact_root = (struct agx_ptr){0};

   if (agx_chip_command_stream_generation(dev->dev.chip) ==
       AGX_CS_GENERATION_HAL200) {
      const unsigned stage = s->info.stage == MESA_SHADER_FRAGMENT ? 1 : 0;
      struct hk_g17p_graphics_ioto_cache_entry *cached =
         &cmd->state.gfx.g17p_ioto[stage];

      /* The admission below depends only on the shader and its linked
       * program for a vertex shader, plus the tile-buffer spill state, the
       * native RT0 layout and the varying-coefficient inputs (topology,
       * provoking vertex, hardware vertex shader) for a fragment shader.  An
       * earlier compact publication in this recording with the same inputs
       * was already admitted. */
      struct hk_graphics_state *agfx = &cmd->state.gfx;
      const bool same_program = dev->dev.chip == AGX_CHIP_G17P &&
                                cached->compact.valid &&
                                cached->compact.source_shader == s &&
                                cached->compact.linked == linked;
      const bool vertex_admitted =
         same_program && s->info.stage == MESA_SHADER_VERTEX;
      const bool fragment_admitted =
         same_program && s->info.stage == MESA_SHADER_FRAGMENT &&
         agfx->render.g17p_rt0_layout_supported &&
         cached->compact.spills ==
            agx_tilebuffer_spills(&agfx->render.tilebuffer) &&
         cached->compact.admit_topology == agfx->topology &&
         cached->compact.admit_provoking == agfx->provoking &&
         agfx->shaders[MESA_SHADER_VERTEX] &&
         cached->compact.admit_hw_vs == hk_bound_hw_vs(agfx);
      if (!vertex_admitted && !fragment_admitted &&
          (dev->dev.chip != AGX_CHIP_G17P ||
           !hk_g17p_basic_graphics_program_supported(cmd, s, linked))) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return 0;
      }

      /* v51x HK_G17P_IOTO_IMMEDIATE takes the uncached inline composer too:
       * its programs embed record-time values, so address-keyed templates,
       * relocation and the compact root launcher do not apply. */
      const bool ioto_immediate =
         HK_ENV_BOOL_ONCE("HK_G17P_IOTO_IMMEDIATE", false);
      const bool diagnostics = (dev->dev.debug & AGX_DBG_FAULTTRACE) ||
                               hk_g17p_launch_trace_enabled() ||
                               hk_g17p_draw_trace_enabled() || ioto_immediate;
      const bool fast_cacheable =
         !diagnostics && !s->info.agx3_combined_sampler_valid &&
         !s->info.g17p_apple_fs &&
         !(s->info.stage == MESA_SHADER_FRAGMENT &&
           agx_tilebuffer_spills(&cmd->state.gfx.render.tilebuffer));
      const bool fragment = s->info.stage == MESA_SHADER_FRAGMENT;
      const bool vertex = s->info.stage == MESA_SHADER_VERTEX;
      const bool root_dependent =
         (vertex &&
          BITSET_LAST_BIT(s->info.vs.attrib_components_read) != 0) ||
         (fragment && hk_cmd_uses_blend_constant(cmd)) ||
         hk_agx3_g17p_graphics_needs_root(
            s->info.agx3_uses_root, linked->g17p_prolog_uses_root,
            s->info.set_count);
      const uint64_t linked_shader_va =
         linked->b.bo->va->addr + linked->b.entry_offset;
      const uint64_t rodata_va =
         s->b.info.rodata.size_16 && s->bo && s->bo->va
            ? s->bo->va->addr
            : 0;
      const struct agx_tilebuffer_layout *tib =
         &cmd->state.gfx.render.tilebuffer;
      const uint64_t dma_set_base = hk_g17p_dma_set_base(cmd, s);

      /* The compact launcher is already address-free with respect to shader
       * payload values: it names a small GPU root whose entries are source
       * addresses.  Reuse the last published launcher when every address and
       * executable identity is unchanged instead of rebuilding ranges,
       * uploading an identical root, reserving IOTO space and relocating the
       * same launcher again.  Root descriptor storage is immutable after
       * upload, so pointer equality is the exact lifetime boundary here. */
      if (fast_cacheable && cached->valid && cached->fast_valid &&
          cached->source_shader == s &&
          cached->linked_shader_va == linked_shader_va &&
          cached->rodata_va == rodata_va &&
          cached->root == cmd->state.gfx.root &&
          /* Only vertex uniform ranges name draw parameters and draw ID.
           * Their per-draw uploads cannot change a fragment launcher whose
           * shader, root, DMA source and render layout still match. */
          (fragment ||
           (cached->draw_params == cmd->state.gfx.draw_params &&
            cached->draw_id_ptr == cmd->state.gfx.draw_id_ptr)) &&
          cached->root_dependent == root_dependent &&
          cached->dma_set_base == dma_set_base &&
          cached->tile_size_px == tib->tile_size &&
          cached->sample_size_B == tib->sample_size_B &&
          cached->nr_samples == tib->nr_samples) {
         hk_g17p_graphics_ioto_cache_stat(
            dev, HK_G17P_GRAPHICS_IOTO_LAST_HIT);
         return cached->program;
      }

      /* HK_G17P_MDI_ONE_LEVEL: an indexed-indirect record publishes the
       * one-level template (see hk_g17p_publish_one_level_record()), never
       * the two-level compact launcher. */
      const bool one_level = vertex && fast_cacheable &&
                             cmd->state.gfx.g17p_vs_one_level;
      if (one_level) {
         const uint64_t published = hk_g17p_publish_one_level_record(
            cmd, s, linked, cached, linked_shader_va, rodata_va,
            dma_set_base, tib);
         if (published) {
            cached->program = published;
            cached->source_shader = s;
            cached->linked_shader_va = linked_shader_va;
            cached->rodata_va = rodata_va;
            cached->root = cmd->state.gfx.root;
            cached->draw_params = cmd->state.gfx.draw_params;
            cached->draw_id_ptr = cmd->state.gfx.draw_id_ptr;
            cached->root_dependent = root_dependent;
            cached->dma_set_base = dma_set_base;
            cached->tile_size_px = tib->tile_size;
            cached->sample_size_B = tib->sample_size_B;
            cached->nr_samples = tib->nr_samples;
            cached->key_size = 0;
            cached->fast_valid = fast_cacheable;
            cached->valid = true;
            hk_g17p_graphics_ioto_cache_stat(
               dev, HK_G17P_GRAPHICS_IOTO_TEMPLATE_HIT);
            return published;
         }
         if (vk_command_buffer_has_error(&cmd->vk))
            return 0;
      }

      const bool compact_blend =
         fragment && hk_cmd_uses_blend_constant(cmd);
      const bool compact_spills =
         fragment && agx_tilebuffer_spills(&cmd->state.gfx.render.tilebuffer);
      const bool compact_same_program =
         !diagnostics && cached->compact.valid &&
         cached->compact.dma_set_base == dma_set_base &&
         cached->compact.source_shader == s &&
         cached->compact.linked == linked &&
         cached->compact.linked_shader_va == linked_shader_va &&
         cached->compact.rodata_va == rodata_va &&
         cached->compact.tile_size_px == tib->tile_size &&
         cached->compact.sample_size_B == tib->sample_size_B &&
         cached->compact.nr_samples == tib->nr_samples &&
         cached->compact.blend == compact_blend &&
         cached->compact.spills == compact_spills;

      /* A vertex program whose every range source is the root, a draw
       * parameter pointer or a constant: hk_g17p_graphics_uniform_ranges()
       * would return the recorded layout with each address re-derived from
       * the current pointers (refusing only a missing root), so derive the
       * root record directly.  This is every record of an indexed-indirect
       * multi-draw and every draw after a descriptor push. */
      bool compact_attempted = false;
      if (!one_level && compact_same_program && vertex &&
          cached->compact.patchable &&
          (!cached->compact.uses_root || cmd->state.gfx.root != 0) &&
          cached->compact.has_draw_params ==
             (cmd->state.gfx.draw_params != 0) &&
          cached->compact.has_draw_id == (cmd->state.gfx.draw_id_ptr != 0)) {
         const uint64_t base[4] = {
            [HK_G17P_RANGE_SOURCE_CONSTANT] = 0,
            [HK_G17P_RANGE_SOURCE_ROOT] = cmd->state.gfx.root,
            [HK_G17P_RANGE_SOURCE_DRAW_PARAMS] = cmd->state.gfx.draw_params,
            [HK_G17P_RANGE_SOURCE_DRAW_ID] = cmd->state.gfx.draw_id_ptr,
         };
         struct hk_g17p_graphics_root_record compact_root =
            cached->compact.root;
         bool valid = true;
         for (unsigned r = 0; r < cached->compact.range_count; ++r) {
            const uint64_t address = base[cached->compact.source[r]] +
                                     cached->compact.source_offset[r];
            valid &= hk_g17p_compact_range_address_ok(
               address, cached->compact.ranges[r].size_halfs);
            compact_root.range_address[r] = address;
         }
         if (valid) {
            bool external;
            const uint64_t published = hk_g17p_publish_compact_graphics(
               cmd, s, linked, cached->compact.setup, &compact_root,
               cached->compact.compact_ranges, cached->compact.range_count,
               linked_shader_va, fragment, prepared_compact_root,
               cached->compact.external, &external);
            if (published) {
               cached->compact.external |= external;
               cached->program = published;
               cached->source_shader = s;
               cached->linked_shader_va = linked_shader_va;
               cached->rodata_va = rodata_va;
               cached->root = cmd->state.gfx.root;
               cached->draw_params = cmd->state.gfx.draw_params;
               cached->draw_id_ptr = cmd->state.gfx.draw_id_ptr;
               cached->root_dependent = root_dependent;
               cached->dma_set_base = dma_set_base;
               cached->tile_size_px = tib->tile_size;
               cached->sample_size_B = tib->sample_size_B;
               cached->nr_samples = tib->nr_samples;
               cached->key_size = 0;
               cached->fast_valid = fast_cacheable;
               cached->valid = true;
               hk_g17p_graphics_ioto_cache_stat(
                  dev, HK_G17P_GRAPHICS_IOTO_COMPACT_ROOT);
               return published;
            }

            /* The publication attempt consumed the same allocations the
             * general path's compact attempt would have; continue with its
             * non-compact fallback. */
            compact_attempted = true;
         }
      }

      struct agx3_ioto_uniform_range ranges[HK_AGX3_G17P_GRAPHICS_MAX_RANGES];
      uint8_t sources[HK_AGX3_G17P_GRAPHICS_MAX_RANGES];
      size_t range_count = 0;
      if (!hk_g17p_graphics_uniform_ranges(cmd, s, linked, ranges,
                                            &range_count, sources)) {
         const unsigned nr_attribs = s->info.stage == MESA_SHADER_VERTEX
            ? DIV_ROUND_UP(
                 BITSET_LAST_BIT(s->info.vs.attrib_components_read), 4)
            : 0;
         mesa_loge_once(
            "G17P graphics uniform ranges refused: stage=%u root=%" PRIx64
            " sets=%u uses_root=%u combined=%u indexed=%u blend_constants=%u "
            "rodata=%u@%u attribs=%u ranges=%zu/%u draw_params=%u "
            "draw_id=%u",
            (unsigned)s->info.stage, cmd->state.gfx.root, s->info.set_count,
            s->info.agx3_uses_root, s->info.agx3_combined_sampler_valid,
            s->info.agx3_indexed_sampler_valid,
            hk_cmd_uses_blend_constant(cmd), s->b.info.rodata.size_16,
            s->b.info.rodata.base_uniform, nr_attribs, range_count,
            HK_AGX3_G17P_GRAPHICS_MAX_RANGES,
            cmd->state.gfx.draw_params != 0, cmd->state.gfx.draw_id_ptr != 0);
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return 0;
      }

      if (hk_g17p_launch_trace_enabled()) {
         for (size_t r = 0; r < range_count; ++r) {
            fprintf(stderr,
                    "G17P graphics uniform range[%zu]: start_halfs=%u "
                    "size_halfs=%u address=%" PRIx64 "\n",
                    r, (unsigned)ranges[r].start_halfs,
                    (unsigned)ranges[r].size_halfs, ranges[r].address);
         }

         /* Keep the source bytes next to the generated Ioto program in an
          * explicit launch trace.  This distinguishes a bad root upload from
          * corruption while LDDEV/COPY publishes the same qwords as vertex
          * uniforms.  It is intentionally behind HK_G17P_LAUNCH_TRACE: the
          * normal desktop path remains silent. */
         if (s->info.stage == MESA_SHADER_VERTEX && cmd->state.gfx.root) {
            const unsigned nr_attribs = DIV_ROUND_UP(
               BITSET_LAST_BIT(s->info.vs.attrib_components_read), 4);
            const size_t bases_size =
               MIN2((size_t)nr_attribs * sizeof(uint64_t),
                    sizeof(cmd->state.gfx.descriptors.root.draw.attrib_base));
            hk_g17p_trace_line(
               "graphics-vbo-bases",
               cmd->state.gfx.root +
                  hk_root_descriptor_offset(draw.attrib_base),
               cmd->state.gfx.descriptors.root.draw.attrib_base, bases_size);
         }
      }

      /* Same linked shader, render layout and uniform-range layout as the
       * last compact publication, with only range source addresses changed
       * (a new root, or the next indirect record's draw parameters): the
       * bounds, constant-program extent and setup derived for that
       * publication are exactly what the checks below would derive again.
       * Rebuild only the root record and launcher, with the identical
       * allocation sequence.  The address checks are the ones those bounds
       * and hk_g17p_graphics_compact_ranges() apply. */
      if (!one_level && !compact_attempted && compact_same_program &&
          cached->compact.range_count == range_count) {
         bool same_layout = true;
         for (size_t r = 0; r < range_count; ++r) {
            same_layout &=
               ranges[r].start_halfs ==
                  cached->compact.ranges[r].start_halfs &&
               ranges[r].size_halfs == cached->compact.ranges[r].size_halfs &&
               hk_g17p_compact_range_address_ok(ranges[r].address,
                                                ranges[r].size_halfs);
         }

         struct agx3_ioto_uniform_range
            compact_ranges[HK_AGX3_G17P_GRAPHICS_MAX_RANGES];
         struct hk_g17p_graphics_root_record compact_root;
         if (same_layout) {
            memcpy(compact_ranges, ranges,
                   range_count * sizeof(compact_ranges[0]));
            same_layout = hk_g17p_graphics_compact_ranges(
               compact_ranges, range_count, &compact_root);
         }
         if (same_layout) {
            compact_attempted = true;
            bool external;
            const uint64_t published = hk_g17p_publish_compact_graphics(
               cmd, s, linked, cached->compact.setup, &compact_root,
               compact_ranges, range_count, linked_shader_va, fragment,
               prepared_compact_root, cached->compact.external, &external);
            if (published) {
               cached->compact.external |= external;
               cached->program = published;
               cached->source_shader = s;
               cached->linked_shader_va = linked_shader_va;
               cached->rodata_va = rodata_va;
               cached->root = cmd->state.gfx.root;
               cached->draw_params = cmd->state.gfx.draw_params;
               cached->draw_id_ptr = cmd->state.gfx.draw_id_ptr;
               cached->root_dependent = root_dependent;
               cached->dma_set_base = dma_set_base;
               cached->tile_size_px = tib->tile_size;
               cached->sample_size_B = tib->sample_size_B;
               cached->nr_samples = tib->nr_samples;
               cached->key_size = 0;
               cached->fast_valid = fast_cacheable;
               cached->valid = true;
               hk_g17p_graphics_ioto_cache_stat(
                  dev, HK_G17P_GRAPHICS_IOTO_COMPACT_ROOT);
               return published;
            }
         }
      }

      struct hk_agx3_g17p_graphics_program_desc desc = {
         /* The container's entry point, not the allocation base: the
       * shader loader reads the block-size dword at entry - 0x80. */
         .shader_va = linked_shader_va,
         .uniform_ranges = ranges,
         .uniform_range_count = range_count,
         .shader_binary_size = linked->b.size,
         .scratch_size =
            MAX2(linked->b.scratch_size,
                 linked->b.agx3_spectpr_scratch_size),
         .nr_gprs = linked->b.nr_gprs,
         .compiler_d_valid = linked->b.g17p_compiler_d_valid,
         .compiler_d = linked->b.g17p_compiler_d,
         .fragment = fragment,
         .writes_sample_mask = fragment &&
            ((linked->b.fragment_control.opaque[0] >> 29) == AGX_PASS_TYPE_PUNCH_THROUGH ||
             (linked->b.fragment_control.opaque[0] >> 29) == AGX_PASS_TYPE_TRANSLUCENT_PUNCH_THROUGH),
         .colour_classes = linked->b.g17p_colour_classes,
         .imageblock = {
            .sample_size_B = cmd->state.gfx.render.tilebuffer.sample_size_B,
            .nr_samples = cmd->state.gfx.render.tilebuffer.nr_samples,
            .tile_size_px = cmd->state.gfx.render.tilebuffer.tile_size,
         },
      };

      /* v51x HK_G17P_IOTO_IMMEDIATE: resolve each range whose source is
       * immutable CPU-known memory. Any unresolved range keeps the ordinary
       * load body for the whole program (the generator requires all). */
      const void *range_cpu[HK_AGX3_G17P_GRAPHICS_MAX_RANGES] = {0};
      uint8_t range_staging[HK_AGX3_G17P_GRAPHICS_MAX_RANGES][256];
      if (ioto_immediate) {
         /* The root is uploaded later in this flush, so only root fields that
          * are final before the draw is flushed qualify: root_desc_addr (by
          * construction the root VA itself), the descriptor-set pointers
          * (bind-time) and the push constants (CmdPushConstants). Draw-time
          * fields written by the flush keep their loads. */
         const struct hk_root_descriptor_table *root_cpu =
            &cmd->state.gfx.descriptors.root;
         const uint64_t root_va = cmd->state.gfx.root;
         const uint64_t sets_end = offsetof(struct hk_root_descriptor_table,
                                            sets) + sizeof(root_cpu->sets);
         const uint64_t push_start = hk_root_descriptor_offset(push);
         const uint64_t push_end = push_start + sizeof(root_cpu->push);
         const uint64_t rodata_va =
            s->bo && s->bo->va && s->b.info.rodata.size_16
               ? s->bo->va->addr + s->b.info.rodata.offset
               : 0;
         bool all = range_count > 0;
         for (size_t i = 0; i < range_count; ++i) {
            const uint64_t a = ranges[i].address;
            const uint64_t bytes = (uint64_t)ranges[i].size_halfs * 2;
            const uint64_t off = a - root_va;
            if (root_va && a >= root_va && bytes <= 256 &&
                ((off + bytes <= sets_end) ||
                 (off >= push_start && off + bytes <= push_end))) {
               memcpy(range_staging[i], (const uint8_t *)root_cpu + off, bytes);
               /* root_desc_addr is published as the root's own VA. */
               for (uint64_t b = off; b < MIN2(off + bytes, 8); ++b)
                  range_staging[i][b - off] = (uint8_t)(root_va >> (8 * b));
               range_cpu[i] = range_staging[i];
            } else if (rodata_va && a == rodata_va && s->code_ptr &&
                       s->b.info.rodata.offset + bytes <= s->code_size) {
               range_cpu[i] = (const uint8_t *)s->code_ptr +
                              s->b.info.rodata.offset;
            } else {
               all = false;
            }
         }
         if (all)
            desc.uniform_range_cpu = range_cpu;
         if (hk_g17p_launch_trace_enabled()) {
            fprintf(stderr, "G17P IOTO immediate: stage=%u ranges=%zu all=%u "
                    "root=%" PRIx64 " rodata=%" PRIx64,
                    (unsigned)s->info.stage, range_count, all, root_va,
                    rodata_va);
            for (size_t i = 0; i < range_count; ++i)
               fprintf(stderr, " [%" PRIx64 "+%u:%s]", ranges[i].address,
                       ranges[i].size_halfs, range_cpu[i] ? "cpu" : "load");
            fprintf(stderr, "\n");
         }
         mesa_logi_once("HK_G17P_IOTO_IMMEDIATE: graphics setup programs "
                        "publish CPU-known uniform ranges with LDIMM");
      }

      if (s->b.info.has_preamble &&
          (!s->bo || !s->bo->va ||
           s->code_size != s->b.info.binary_size ||
           !hk_agx3_g17p_constant_program_from_binary(
              s->code_ptr, s->code_size, s->b.info.preamble_offset,
              s->b.info.preamble_size, s->bo->va->addr, s->bo->size,
              &desc.constant_program))) {
         mesa_loge_once("G17P graphics constant-program extent refused");
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return 0;
      }

      desc.constant_program.request_prelude =
         s->b.info.agx3_preamble_uniform_inputs;
      if (s->b.info.agx3_preamble_tps_inputs) {
         if (!s->b.info.has_preamble ||
             s->b.info.preamble_scratch_size <
                s->b.info.agx3_preamble_inputs.storage_bytes) {
            vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
            return 0;
         }
         desc.cprog_inputs = &s->b.info.agx3_preamble_inputs;
      }

      size_t program_capacity;
      if (!hk_agx3_g17p_graphics_program_desc_bound(
             dev->dev.chip, &desc, &program_capacity)) {
         mesa_loge_once("G17P graphics program bound refused: stage=%u ranges=%zu",
                        (unsigned)s->info.stage, range_count);
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return 0;
      }

      /* Production draws publish a small root-address launcher. It performs
       * the mandatory shader-state load before the native external-preshader
       * branch enters immutable linked-shader setup. The inline template is
       * retained as a fail-closed fallback, not as the ordinary hot path. */
      if (!diagnostics && !compact_attempted && !one_level) {
         struct agx3_ioto_uniform_range
            compact_ranges[HK_AGX3_G17P_GRAPHICS_MAX_RANGES];
         memcpy(compact_ranges, ranges,
                range_count * sizeof(compact_ranges[0]));
         struct hk_g17p_graphics_root_record compact_root;
          if (hk_g17p_graphics_compact_ranges(
                 compact_ranges, range_count, &compact_root)) {
            struct hk_agx3_g17p_graphics_program_desc compact_desc = desc;
            compact_desc.uniform_ranges = compact_ranges;
            compact_desc.preloaded_root = true;
            struct hk_g17p_graphics_setup *setup =
               hk_g17p_graphics_setup_get(dev, linked, &compact_desc,
                                          compact_blend, compact_spills);
            if (setup) {
               bool external;
               const uint64_t published = hk_g17p_publish_compact_graphics(
                  cmd, s, linked, setup, &compact_root, compact_ranges,
                  range_count, linked_shader_va, fragment,
                  prepared_compact_root, -1, &external);
               if (published) {
                  cached->program = published;
                  cached->source_shader = s;
                  cached->linked_shader_va = linked_shader_va;
                  cached->rodata_va = rodata_va;
                  cached->root = cmd->state.gfx.root;
                  cached->draw_params = cmd->state.gfx.draw_params;
                  cached->draw_id_ptr = cmd->state.gfx.draw_id_ptr;
                  cached->root_dependent = root_dependent;
                  cached->dma_set_base = dma_set_base;
                  cached->tile_size_px = tib->tile_size;
                  cached->sample_size_B = tib->sample_size_B;
                  cached->nr_samples = tib->nr_samples;
                  cached->key_size = 0;
                  cached->fast_valid = fast_cacheable;
                  cached->valid = true;

                  /* Record what this publication derived for later draws
                   * that change only range source addresses (see above). */
                  cached->compact.valid =
                     range_count <= HK_G17P_GRAPHICS_ROOT_MAX_RANGES;
                  if (cached->compact.valid) {
                     memcpy(cached->compact.ranges, ranges,
                            range_count * sizeof(ranges[0]));
                     memcpy(cached->compact.compact_ranges, compact_ranges,
                            range_count * sizeof(compact_ranges[0]));
                     cached->compact.root = compact_root;
                     cached->compact.patchable = true;
                     cached->compact.uses_root = false;
                     cached->compact.has_draw_params = false;
                     cached->compact.has_draw_id = false;
                     for (size_t r = 0; r < range_count; ++r) {
                        cached->compact.source[r] = sources[r];
                        switch (sources[r]) {
                        case HK_G17P_RANGE_SOURCE_CONSTANT:
                           cached->compact.source_offset[r] = ranges[r].address;
                           break;
                        case HK_G17P_RANGE_SOURCE_ROOT:
                           cached->compact.source_offset[r] =
                              ranges[r].address - cmd->state.gfx.root;
                           cached->compact.uses_root = true;
                           break;
                        case HK_G17P_RANGE_SOURCE_DRAW_PARAMS:
                           cached->compact.source_offset[r] = 0;
                           cached->compact.has_draw_params = true;
                           break;
                        case HK_G17P_RANGE_SOURCE_DRAW_ID:
                           cached->compact.source_offset[r] = 0;
                           cached->compact.has_draw_id = true;
                           break;
                        default:
                           cached->compact.source_offset[r] = 0;
                           cached->compact.patchable = false;
                           break;
                        }
                     }
                     cached->compact.external = external;
                     cached->compact.dma_set_base = dma_set_base;
                     cached->compact.range_count = range_count;
                     cached->compact.source_shader = s;
                     cached->compact.linked = linked;
                     cached->compact.setup = setup;
                     cached->compact.linked_shader_va = linked_shader_va;
                     cached->compact.rodata_va = rodata_va;
                     cached->compact.tile_size_px = tib->tile_size;
                     cached->compact.sample_size_B = tib->sample_size_B;
                     cached->compact.nr_samples = tib->nr_samples;
                     cached->compact.blend = compact_blend;
                     cached->compact.spills = compact_spills;
                     cached->compact.admit_topology = cmd->state.gfx.topology;
                     cached->compact.admit_provoking =
                        cmd->state.gfx.provoking;
                     cached->compact.admit_hw_vs =
                        cmd->state.gfx.shaders[MESA_SHADER_VERTEX]
                           ? hk_bound_hw_vs(&cmd->state.gfx)
                           : NULL;
                  }
                  hk_g17p_graphics_ioto_cache_stat(
                     dev, HK_G17P_GRAPHICS_IOTO_COMPACT_ROOT);
                  return published;
               }
            }
         }
      }

      struct hk_g17p_graphics_ioto_key key;
      const unsigned key_size = hk_g17p_graphics_ioto_key_init(dev->dev.chip, &desc, &key);
      if (!diagnostics && cached->valid && cached->key_size == key_size &&
          !memcmp(cached->key, &key, key_size)) {
         cached->fast_valid = fast_cacheable;
         if (fast_cacheable) {
            cached->source_shader = s;
            cached->linked_shader_va = linked_shader_va;
            cached->rodata_va = rodata_va;
            cached->root = cmd->state.gfx.root;
            cached->draw_params = cmd->state.gfx.draw_params;
            cached->draw_id_ptr = cmd->state.gfx.draw_id_ptr;
            cached->root_dependent = root_dependent;
            cached->dma_set_base = dma_set_base;
            cached->tile_size_px = tib->tile_size;
            cached->sample_size_B = tib->sample_size_B;
            cached->nr_samples = tib->nr_samples;
         }
         hk_g17p_graphics_ioto_cache_stat(
            dev, HK_G17P_GRAPHICS_IOTO_LAST_HIT);
         return cached->program;
      }

      /* The IOTO program embeds the linked executable address; promoted
       * constants and preambles also reference the shader BO. Pipelines may
       * be destroyed after recording, so a cached program retains both until
       * this command buffer is reset. Cache hits reuse those references.
       */
      hk_cmd_buffer_retain_bo(cmd, linked->b.bo);
      if (s->b.info.rodata.size_16 != 0 || s->b.info.has_preamble)
         hk_cmd_buffer_retain_bo(cmd, s->bo);

      struct hk_ioto_reservation reservation;
      if (!hk_cmd_reserve_ioto_program(cmd, program_capacity,
                                       "G17P Ioto program", &reservation))
         return 0;

      void *program_map = reservation.map;
      const uint64_t program_va = reservation.va;
      desc.program_va = program_va;

      /* Same fail-closed USC-window containment check as the compute path in
       * hk_cmd_dispatch.c: usc_exec_base is fixed per queue, so an exec
       * pointer outside [shader_base, +4 GiB) can only end as an unattributed
       * GPU page fault.
       */
      if (!agx_va_in_usc_window(&dev->dev, program_va) ||
          !agx_va_in_usc_window(&dev->dev, desc.shader_va)) {
         mesa_loge("G17P graphics: exec pointer outside the USC window "
                   "[%" PRIx64 ", %" PRIx64 "): program=%" PRIx64
                   " shader=%" PRIx64 "\n",
                   dev->dev.shader_base,
                   dev->dev.shader_base + AGX_USC_WINDOW_SIZE,
                   program_va, desc.shader_va);
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_UNKNOWN);
         return 0;
      }

      uint16_t program_length = 0;
      const struct hk_g17p_graphics_template_entry *template_entry = NULL;
      const bool template_hit =
         !diagnostics &&
         hk_g17p_graphics_template_lookup(linked, &desc, program_map,
                                           program_capacity, &program_length,
                                           &template_entry);
      bool generated = false;
      if (!template_hit && !diagnostics) {
         generated = hk_g17p_graphics_template_generate(
            linked, &desc, program_map, program_capacity, &program_length,
            &template_entry);
      }
      if (!template_hit && !generated)
         template_entry = NULL;

      /* The linked-shader template is only an optimization.  In particular,
       * its allocation may fail even though ordinary command recording can
       * continue.  Never turn a cache miss/OOM into a Vulkan error: fall back
       * to the uncached composer, which is also the diagnostics path.
       */
      if (!template_hit && !generated) {
         struct hk_agx3_g17p_graphics_program program;
         generated = hk_agx3_g17p_basic_graphics_program_generate(
            dev->dev.chip, &desc, &program);
         if (generated) {
            program_length = program.length;
            memcpy(program_map, program.bytes, program.length);
         }
      }

      if (!template_hit && !generated) {
         mesa_loge_once(
            "G17P graphics program generation refused: stage=%u ranges=%zu "
            "binary=%u scratch=%u gprs=%u compiler_d=%u/%u sample_mask=%u "
            "colour_classes=%#x",
            (unsigned)s->info.stage, range_count, desc.shader_binary_size,
            desc.scratch_size, desc.nr_gprs, desc.compiler_d_valid,
            desc.compiler_d, desc.writes_sample_mask, desc.colour_classes);
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return 0;
      }

      assert(program_length <= program_capacity);

      /* Publish before tracing, so the trace names the address the launch
       * record will actually carry rather than the reserved one -- an
       * identical program already in this recording keeps its own. */
      const uint64_t published =
         hk_cmd_publish_ioto_program(cmd, &reservation, program_length);
      if (!published)
         return 0;

      if (one_level) {
         /* Record how the next indexed-indirect record re-derives these
          * ranges; any range uploaded while collecting is not re-derivable
          * and leaves the record path on the general template lookup. */
         __typeof__(cached->one_level) *one = &cached->one_level;
         bool patchable = template_entry &&
                          range_count <= HK_G17P_GRAPHICS_ROOT_MAX_RANGES;
         const uint64_t base[4] = {
            [HK_G17P_RANGE_SOURCE_CONSTANT] = 0,
            [HK_G17P_RANGE_SOURCE_ROOT] = cmd->state.gfx.root,
            [HK_G17P_RANGE_SOURCE_DRAW_PARAMS] = cmd->state.gfx.draw_params,
            [HK_G17P_RANGE_SOURCE_DRAW_ID] = cmd->state.gfx.draw_id_ptr,
         };
         for (size_t r = 0; patchable && r < range_count; ++r) {
            if (sources[r] > HK_G17P_RANGE_SOURCE_DRAW_ID ||
                (sources[r] != HK_G17P_RANGE_SOURCE_CONSTANT &&
                 (!base[sources[r]] || ranges[r].address < base[sources[r]]))) {
               patchable = false;
               break;
            }
            one->ranges[r] = ranges[r];
            one->source[r] = sources[r];
            one->source_offset[r] = ranges[r].address - base[sources[r]];
         }
         one->valid = patchable;
         hk_g17p_one_level_verify(dev, &desc, program_map, program_length,
                                  "template");
         if (dev->g17p_diagnostics.compact_ioto_trace) {
            fprintf(stderr,
                    "G17P one-level template: shader=%" PRIx64
                    " program=%" PRIx64 " len=%u ranges=%zu patchable=%u"
                    " template=%u\n",
                    linked_shader_va, published, (unsigned)program_length,
                    range_count, patchable, template_hit);
            hk_g17p_trace_line("one-level-program", published, program_map,
                               program_length);
         }
         one->patch = false;
         if (patchable &&
             program_length <= sizeof(one->bytes) &&
             template_entry->program.relocation_count <=
                ARRAY_SIZE(template_entry->program.relocations)) {
            /* Position independence: the same ranges relocated at another
             * program VA must give identical bytes (an external constant
             * program reached through a PC-relative branch would not). */
            struct hk_agx3_g17p_graphics_program_desc moved = desc;
            moved.program_va = published + 0x1000;
            uint8_t again[sizeof(one->bytes)];
            uint16_t again_length = 0;
            bool patch = hk_agx3_g17p_graphics_program_template_relocate(
                            dev->dev.chip, &template_entry->desc, &moved,
                            &template_entry->program, again, sizeof(again),
                            &again_length) &&
                         again_length == program_length &&
                         !memcmp(again, program_map, program_length);
            unsigned sites = 0;
            const struct agx_ioto_load_relocation *rels =
               template_entry->program.relocations;
            for (unsigned i = 0;
                 patch && i < template_entry->program.relocation_count; ++i) {
               const struct agx_ioto_load_relocation *rel = &rels[i];
               unsigned range = UINT_MAX;
               for (unsigned r = 0;
                    r < template_entry->desc.uniform_range_count; ++r) {
                  const struct agx3_ioto_uniform_range *tr =
                     &template_entry->desc.uniform_ranges[r];
                  const uint64_t bytes = (uint64_t)tr->size_halfs * 2;
                  if (bytes >= 4 && rel->address >= tr->address &&
                      rel->address <= tr->address + bytes - 4 &&
                      !((rel->address - tr->address) & 3)) {
                     range = r;
                     break;
                  }
               }
               if (range >= range_count) {
                  patch = false;
                  break;
               }
               if (sources[range] != HK_G17P_RANGE_SOURCE_DRAW_PARAMS &&
                   sources[range] != HK_G17P_RANGE_SOURCE_DRAW_ID)
                  continue;
               if (!hk_g17p_one_level_range_isolated(
                      template_entry->desc.uniform_ranges,
                      template_entry->desc.uniform_range_count, range) ||
                   !hk_g17p_one_level_range_isolated(ranges, range_count,
                                                     range)) {
                  patch = false;
                  break;
               }
               const uint64_t delta =
                  rel->address - template_entry->desc.uniform_ranges[range].address;
               /* The published bytes must already hold exactly this
                * record's address in that LDIMM pair. */
               uint8_t low[AGX_IOTO_LOAD_LDIMM_MAX_SIZE];
               uint8_t high[AGX_IOTO_LOAD_LDIMM_MAX_SIZE];
               uint8_t low_length = 0, high_length = 0;
               const uint64_t now = ranges[range].address + delta;
               if (sites >= ARRAY_SIZE(one->sites) ||
                   rel->offset > UINT16_MAX || rel->dest_half > UINT16_MAX - 2 ||
                   !agx_ioto_load_pack_ldimm(dev->dev.chip, rel->dest_half,
                                             (uint32_t)now, low, &low_length) ||
                   !agx_ioto_load_pack_ldimm(dev->dev.chip,
                                             rel->dest_half + 2,
                                             (uint32_t)(now >> 32), high,
                                             &high_length) ||
                   low_length != rel->low_length ||
                   high_length != rel->high_length ||
                   rel->offset + low_length + high_length > program_length ||
                   memcmp((const uint8_t *)program_map + rel->offset, low,
                          low_length) ||
                   memcmp((const uint8_t *)program_map + rel->offset +
                             low_length,
                          high, high_length)) {
                  patch = false;
                  break;
               }
               one->sites[sites++] = (__typeof__(one->sites[0])){
                  .offset = (uint16_t)rel->offset,
                  .dest_half = rel->dest_half,
                  .low_length = low_length,
                  .high_length = high_length,
                  .range = (uint8_t)range,
                  .delta = delta,
               };
            }
            if (patch) {
               memcpy(one->bytes, program_map, program_length);
               one->length = program_length;
               one->site_count = (uint8_t)sites;
               one->patch = true;
            }
         }
         if (patchable) {
            one->entry = template_entry;
            one->source_shader = s;
            one->linked = linked;
            one->linked_shader_va = linked_shader_va;
            one->rodata_va = rodata_va;
            one->root = cmd->state.gfx.root;
            one->dma_set_base = dma_set_base;
            one->capacity = program_capacity;
            one->tile_size_px = tib->tile_size;
            one->sample_size_B = tib->sample_size_B;
            one->nr_samples = tib->nr_samples;
            one->range_count = (uint8_t)range_count;
            one->has_draw_params = cmd->state.gfx.draw_params != 0;
            one->has_draw_id = cmd->state.gfx.draw_id_ptr != 0;
         }
      }

      if (!diagnostics) {
         memcpy(cached->key, &key, key_size);
         cached->key_size = key_size;
         cached->program = published;
         cached->fast_valid = fast_cacheable;
         if (fast_cacheable) {
            cached->source_shader = s;
            cached->linked_shader_va = linked_shader_va;
            cached->rodata_va = rodata_va;
            cached->root = cmd->state.gfx.root;
            cached->draw_params = cmd->state.gfx.draw_params;
            cached->draw_id_ptr = cmd->state.gfx.draw_id_ptr;
            cached->root_dependent = root_dependent;
            cached->dma_set_base = dma_set_base;
            cached->tile_size_px = tib->tile_size;
            cached->sample_size_B = tib->sample_size_B;
            cached->nr_samples = tib->nr_samples;
         }
         cached->valid = true;
      }

      hk_g17p_graphics_ioto_cache_stat(
         dev, template_hit ? HK_G17P_GRAPHICS_IOTO_TEMPLATE_HIT
                           : HK_G17P_GRAPHICS_IOTO_GENERATED);

      if ((dev->dev.debug & AGX_DBG_FAULTTRACE) &&
          s->info.stage == MESA_SHADER_FRAGMENT && cmd->current_cs.gfx) {
         struct hk_cs *cs = cmd->current_cs.gfx;
         void *shader_map = agx_bo_map(linked->b.bo);
         const uint32_t linked_size = shader_map ? linked->b.size : 0;
         const uint32_t main_size = s->code_ptr ? s->code_size : 0;
         const uint32_t root_size = cmd->state.gfx.root
                                       ? sizeof(cmd->state.gfx.descriptors.root)
                                       : 0;
         struct hk_g17p_faulttrace_snapshot_header header = {
            .magic = HK_G17P_FAULTTRACE_SNAPSHOT_MAGIC,
            .program_va = published,
            .shader_va = desc.shader_va,
            .root_va = cmd->state.gfx.root,
            .ioto_size = program_length,
            .linked_size = linked_size,
            .main_size = main_size,
            .root_size = root_size,
         };
         size_t descriptor_total = 0;
         for (unsigned set = 0; set < HK_MAX_SETS; ++set) {
            header.descriptor_va[set] =
               cmd->state.gfx.descriptors.root.sets[set];
            header.descriptor_size[set] =
               cmd->state.gfx.descriptors.set_sizes[set];

            const size_t available =
               HK_G17P_FAULTTRACE_MAX_DESCRIPTOR_BYTES - descriptor_total;
            const size_t capture =
               MIN2((size_t)header.descriptor_size[set], available);
            if (header.descriptor_va[set] && capture) {
               uint8_t *copy = malloc(capture);
               if (copy && agxdecode_copy_gpu_mem(dev->dev.agxdecode,
                                                  header.descriptor_va[set],
                                                  capture, copy)) {
                  header.descriptor_capture_size[set] = capture;
                  descriptor_total += capture;
               }
               free(copy);
            }
         }

         const size_t total = sizeof(header) + header.ioto_size +
                              header.linked_size + header.main_size +
                              header.root_size + descriptor_total;
         uint8_t *snapshot = util_dynarray_grow_bytes(
            &cs->g17p_faulttrace_snapshots, 1, total);
         if (snapshot) {
            memcpy(snapshot, &header, sizeof(header));
            snapshot += sizeof(header);
            memcpy(snapshot, program_map, header.ioto_size);
            snapshot += header.ioto_size;
            if (header.linked_size) {
               memcpy(snapshot,
                      (uint8_t *)shader_map + linked->b.entry_offset,
                      header.linked_size);
               snapshot += header.linked_size;
            }
            if (header.main_size) {
               memcpy(snapshot, s->code_ptr, header.main_size);
               snapshot += header.main_size;
            }
            if (header.root_size) {
               memcpy(snapshot, &cmd->state.gfx.descriptors.root,
                      header.root_size);
               snapshot += header.root_size;
            }
            for (unsigned set = 0; set < HK_MAX_SETS; ++set) {
               const size_t size = header.descriptor_capture_size[set];
               if (!size)
                  continue;

               if (!agxdecode_copy_gpu_mem(dev->dev.agxdecode,
                                           header.descriptor_va[set], size,
                                           snapshot)) {
                  /* The mapping was present while sizing above.  If it
                   * vanished during this diagnostic-only copy, preserve the
                   * record shape and make the later byte comparison fail. */
                  memset(snapshot, 0, size);
               }
               snapshot += size;
            }
         }
      }

      if (hk_g17p_draw_trace_enabled() &&
          s->info.stage == MESA_SHADER_FRAGMENT && cmd->current_cs.gfx) {
         struct hk_cs *cs = cmd->current_cs.gfx;
         void *shader_map = agx_bo_map(linked->b.bo);

         cs->g17p_draw_trace.fragment_program = published;
         cs->g17p_draw_trace.fragment_ioto_hash =
            _mesa_hash_data(program_map, program_length);
         cs->g17p_draw_trace.fragment_ioto_size = program_length;
         cs->g17p_draw_trace.fragment_shader_size = linked->b.size;
         cs->g17p_draw_trace.fragment_shader_hash =
            shader_map
               ? _mesa_hash_data((uint8_t *)shader_map + linked->b.entry_offset,
                                 linked->b.size)
               : 0;
         /* Keep the main shader identity separate from the linked executable:
          * render-target epilogs can change the latter for one application
          * shader, which makes a timeout fingerprint needlessly ambiguous.
          */
         cs->g17p_draw_trace.fragment_main_size = s->code_size;
         cs->g17p_draw_trace.fragment_main_hash =
            s->code_ptr ? _mesa_hash_data(s->code_ptr, s->code_size) : 0;
         cs->g17p_draw_trace.color_count =
            cmd->state.gfx.render.color_att_count;
         for (unsigned i = 0; i < cmd->state.gfx.render.color_att_count; ++i) {
            cs->g17p_draw_trace.color_formats[i] =
               cmd->state.gfx.render.color_att[i].vk_format;
         }
         cs->g17p_draw_trace.depth_format =
            cmd->state.gfx.render.depth_att.vk_format;
      }

      if (hk_g17p_launch_trace_enabled()) {
         const char *label = s->info.stage == MESA_SHADER_FRAGMENT
                                ? "fragment-ioto"
                                : "vertex-ioto";
         hk_g17p_trace_line(label, published, program_map, program_length);

         const char *shader_label = s->info.stage == MESA_SHADER_FRAGMENT
                                       ? "fragment-shader"
                                       : "vertex-shader";
         void *shader_map = agx_bo_map(linked->b.bo);
         fprintf(stderr,
                 "G17P graphics shader: stage=%u va=%" PRIx64
                 " size=%u entry=%u gprs=%u scratch=%u compiler_d=%u/%u\n",
                 (unsigned)s->info.stage, desc.shader_va,
                 (unsigned)linked->b.size, (unsigned)linked->b.entry_offset,
                 (unsigned)linked->b.nr_gprs,
                 (unsigned)linked->b.scratch_size,
                 (unsigned)linked->b.g17p_compiler_d_valid,
                 (unsigned)linked->b.g17p_compiler_d);
         if (shader_map)
            hk_g17p_trace_hexdump(
               shader_label, desc.shader_va,
               (uint8_t *)shader_map + linked->b.entry_offset,
               MIN2((size_t)linked->b.size, (size_t)512));
         if (s->info.stage == MESA_SHADER_FRAGMENT && cmd->state.gfx.root)
            hk_g17p_trace_line("fragment-root", cmd->state.gfx.root,
                               &cmd->state.gfx.descriptors.root,
                               MIN2(sizeof(cmd->state.gfx.descriptors.root),
                                    (size_t)128));
      }
      return published;
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
         /* One USC uniform record covers at most 64 halves: 16 attribute
          * bases or 32 clamps. Beyond 16 attributes, split like
          * agx_usc_immediates() does for rodata. */
         hk_usc_uniform_split(
            &b, AGX_ABI_VUNI_VBO_BASE(0), 4 * count,
            root_ptr + hk_root_descriptor_offset(draw.attrib_base));

         hk_usc_uniform_split(
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

      if (hk_cmd_uses_blend_constant(cmd)) {
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

uint64_t
hk_upload_usc_words(struct hk_cmd_buffer *cmd, struct hk_shader *s,
                    struct hk_linked_shader *linked)
{
   return hk_upload_usc_words_internal(cmd, s, linked, NULL);
}

uint64_t
hk_prepare_usc_words(struct hk_cmd_buffer *cmd, struct hk_shader *s,
                     struct hk_linked_shader *linked,
                     struct agx_ptr *compact_root)
{
   return hk_upload_usc_words_internal(cmd, s, linked, compact_root);
}

bool
hk_retarget_prepared_usc_root(struct hk_cmd_buffer *cmd, struct hk_shader *s,
                              struct hk_linked_shader *linked,
                              uint64_t prepared_program,
                              struct agx_ptr compact_root)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (dev->dev.chip != AGX_CHIP_G17P || !compact_root.cpu ||
       !compact_root.gpu)
      return false;

   struct agx3_ioto_uniform_range ranges[HK_AGX3_G17P_GRAPHICS_MAX_RANGES];
   size_t range_count = 0;
   if (!hk_g17p_graphics_uniform_ranges(cmd, s, linked, ranges,
                                         &range_count, NULL))
      return false;

   struct hk_g17p_graphics_root_record replacement;
   if (!hk_g17p_graphics_compact_ranges(ranges, range_count, &replacement))
      return false;

   const struct hk_g17p_graphics_root_record *prepared = compact_root.cpu;
   if (prepared->version != replacement.version ||
       prepared->size_B != replacement.size_B ||
       prepared->valid != replacement.valid ||
       prepared->range_count != replacement.range_count)
      return false;

   /* No VDM state record references this prepared launcher yet. Retarget only
    * its command-buffer-owned pointer record, then consume it exactly once.
    * Payload data, including GPU-authored indirect parameters, is never
    * copied or inspected by the CPU. */
   memcpy(compact_root.cpu, &replacement, sizeof(replacement));

   /* Keep the exact-last cache coherent with the record just retargeted. A
    * future hit is legal only for these new source pointers; a different
    * indirect descriptor must allocate its own immutable root. */
   struct hk_g17p_graphics_ioto_cache_entry *cached =
      &cmd->state.gfx.g17p_ioto[0];
   if (cached->valid && cached->program == prepared_program) {
      cached->draw_params = cmd->state.gfx.draw_params;
      cached->draw_id_ptr = cmd->state.gfx.draw_id_ptr;
   } else {
      cached->valid = false;
   }
   return true;
}

void
hk_dispatch_precomp(struct hk_cmd_buffer *cmd, struct agx_grid grid,
                    enum agx_barrier barrier, enum libagx_program idx,
                    void *data, size_t data_size)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (dev->dev.chip == AGX_CHIP_G17P) {
      hk_dispatch_precomp_g17p(cmd, grid, barrier, idx, data, data_size);
      return;
   }

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

uint64_t
hk_apply_conditional_rendering(struct hk_cmd_buffer *cmd, uint64_t source,
                               uint32_t words, enum agx_barrier barrier)
{
   const struct hk_conditional_render_state *conditional =
      &cmd->state.conditional;

   if (!hk_conditional_render_is_active(conditional, cmd->in_meta,
                                         cmd->conditional_meta))
      return source;

   if (!source || !conditional->address || words == 0 ||
       words > HK_CONDITIONAL_RENDER_MAX_WORDS) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return 0;
   }

   const struct agx_ptr output =
      hk_pool_alloc(cmd, words * sizeof(uint32_t), 4);
   if (!output.gpu) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return 0;
   }

   libagx_conditional_render(cmd, agx_1d(1), barrier, output.gpu, source,
                             conditional->address, words,
                             conditional->inverted);
   return vk_command_buffer_has_error(&cmd->vk) ? 0 : output.gpu;
}

bool
hk_cs_init_graphics(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_rendering_state *render = &cmd->state.gfx.render;
   uint8_t *map = cs->current;

   /* Even a clear-only stream needs a stable root alias before EndCommandBuffer.
    * Linked chunks acquire theirs in hk_cmd_buffer_emit_cs_jump. */
   if (hk_g17p_vdm_low_window_enabled(&dev->dev) &&
       !hk_g17p_vdm_compact(cmd, cs->addr))
      return false;

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

   /* Each retained span is a G17P fragment write range, and the kernel
    * encodes at most sixteen of them next to its own. Only a direct colour
    * resolve writes a resolve attachment during the pass; every other resolve
    * is a later meta pass that retains its own attachments. Eight multisample
    * targets and their resolve targets would otherwise overflow the table and
    * fail the submit. */
   const bool g17p = dev->dev.chip == AGX_CHIP_G17P;
   const bool resolves_written = !g17p || render->g17p_direct_color_resolve;

   for (unsigned i = 0; i < render->color_att_count; ++i) {
      views[view_count++] = (struct hk_attachment_view_ref){
         .view = render->color_att[i].iview,
         .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
      };
      views[view_count++] = (struct hk_attachment_view_ref){
         .view = resolves_written ? render->color_att[i].resolve_iview : NULL,
         .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
      };
   }

   views[view_count++] = (struct hk_attachment_view_ref){
      .view = render->depth_att.iview,
      .aspect = VK_IMAGE_ASPECT_DEPTH_BIT,
   };
   views[view_count++] = (struct hk_attachment_view_ref){
      .view = g17p ? NULL : render->depth_att.resolve_iview,
      .aspect = VK_IMAGE_ASPECT_DEPTH_BIT,
   };
   views[view_count++] = (struct hk_attachment_view_ref){
      .view = render->stencil_att.iview,
      .aspect = VK_IMAGE_ASPECT_STENCIL_BIT,
   };
   views[view_count++] = (struct hk_attachment_view_ref){
      .view = g17p ? NULL : render->stencil_att.resolve_iview,
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
         return false;
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
      return false;
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
         return false;
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
         return false;
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
         return false;
      }

      struct agx_ptr T =
         hk_pool_vdm_alloc(cmd, initial.length, AGX_PPP_HEADER_ALIGN);
      if (!T.cpu) {
         /* Returning without an error left the control stream stopped after
          * the VDM barrier, short of HK_G17P_VDM_INITIAL_SIZE, with the
          * command buffer still reporting VK_SUCCESS.  Every following draw
          * then appended to a half-built prelude, and the size assert below
          * aborts a debug build instead of reporting anything. */
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         return false;
      }

      memcpy(T.cpu, initial.bytes, initial.length);
      if (!agx_ppp_emit_state(dev->dev.chip, &map, T.gpu, initial.length)) {
         mesa_loge("failed to emit G17P initial PPP state for %s",
                   agx_chip_name(dev->dev.chip));
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         return false;
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
         return false;

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
         return false;
      }

      if (hk_g17p_launch_trace_enabled())
         hk_g17p_trace_line("ppp-initial", ppp.gpu_base, ppp.cpu_base,
                            ppp.total_size);
   }
   cs->current = map;

   cs->scissor = UTIL_DYNARRAY_INIT;
   cs->depth_bias = UTIL_DYNARRAY_INIT;
   if (!util_dynarray_ensure_cap(&cs->scissor,
                                 HK_MAX_VIEWPORTS * AGX_SCISSOR_LENGTH) ||
       !util_dynarray_ensure_cap(&cs->depth_bias,
                                 HK_MAX_VIEWPORTS * AGX_DEPTH_BIAS_LENGTH)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
      return false;
   }

   /* All graphics state must be reemited in each control stream */
   hk_cmd_buffer_dirty_all(cmd);

   if (dev->dev.chip == AGX_CHIP_G17P)
      assert((cs->current - cs->start) == HK_G17P_VDM_INITIAL_SIZE);

   return true;
}

__attribute__((hot)) void
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
   if (!T.cpu)
      return;

   /* Jump from the old control stream to the new control stream. */
   void *link_end =
      hk_cmd_buffer_emit_cs_jump(cmd, cs->current, T.gpu, vdm);
   const bool linked = link_end != cs->current;
   if (!linked)
      return;

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

   /* Two reasons to keep the graphics representation on G17P rather than
    * replace it with the image clears below.
    *
    * The first is capability: G17P has native BG/EOT programs for the
    * supported render-target clear path, but had no precompiled fill_uint4
    * for the compute replacement. That one retires itself once the shader is
    * available.
    *
    * The second is correctness, and it does not retire. A direct colour
    * resolve is performed by this render's OWN end-of-tile program -- it is
    * selected precisely when storeOp is not STORE, so the multisample
    * attachment is never stored and the pass's only output is the resolve
    * target. Clearing the attachment as an image then writes a surface
    * nobody reads and leaves the resolve target untouched, which is exactly
    * why a clear-only multisample pass resolved to 00 00 00 00 while a pass
    * with any draw in it was correct. Whether fill_uint4 happens to be
    * available cannot be allowed to decide that.
    *
    * BeginRendering retained the attachments, initialized TA/PPP and selected
    * the BG clear; EndRendering selected the final EOT. process_empty_tiles
    * is the existing clear-and-store predicate, not a fabricated draw. The
    * common finalizer will append the native VDM terminator without uploading
    * any ordinary vertex or fragment program.
    */
   const bool g17p_output_only_through_eot =
      dev->dev.chip == AGX_CHIP_G17P && render->g17p_direct_color_resolve;

   /* A timestamp rides on this exact command; replacing the render with
    * image clears would drop it. Keep the render. */
   if (cs->timestamp.start.handle || cs->timestamp.end.handle)
      return false;

   if (dev->dev.chip == AGX_CHIP_G17P && cs->cr.process_empty_tiles &&
       (g17p_output_only_through_eot ||
        !agx_get_precompiled(&dev->bg_eot, LIBAGX_FILL_UINT4))) {
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
                         "scissor=%u zbias=%u direct_resolve=%u\n",
                 cs->cr.width, cs->cr.height, MAX2(cs->tib.nr_samples, 1),
                 cs->ppp_multisamplectl, cs->scissor.size,
                 cs->depth_bias.size, g17p_output_only_through_eot);
      }
      return false;
   }

   /* The clears stand in for the render: they are graphics work for every
    * later dependency, and they were recorded under the same requirements
    * the render was created under. */
   cmd->sync.kind_override = HK_SYNC_GFX;

   for (unsigned i = 0; i < render->color_att_count; ++i) {
      clear_attachment_as_image(cmd, render, &render->color_att[i], ~0);
   }

   clear_attachment_as_image(cmd, render, &render->depth_att,
                             VK_IMAGE_ASPECT_DEPTH_BIT);

   clear_attachment_as_image(cmd, render, &render->stencil_att,
                             VK_IMAGE_ASPECT_STENCIL_BIT);

   cmd->sync.kind_override = 0;

   /* Remove the VDM control stream from the command buffer, now that it is
    * replaced by equivalent other operations.
    */
   hk_sync_forget_stream(cmd, cs);
   list_del(&cs->node);
   hk_cs_destroy(cs);
   return true;
}
