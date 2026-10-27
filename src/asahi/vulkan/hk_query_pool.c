/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "hk_query_pool.h"

#include <errno.h>
#include <xf86drm.h>

#include "agx_compile.h"
#include "agx_pack.h"
#include "hk_buffer.h"
#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_queue.h"
#include "hk_pipeline_statistics_core.h"
#include "hk_shader.h"
#include "hk_acceleration_structure.h"

#include "libagx_dgc.h"
#include "libagx_shaders.h"
#include "vk_alloc.h"
#include "vk_common_entrypoints.h"

#include "asahi/lib/agx_bo.h"
#include "asahi/libagx/libagx.h"
#include "asahi/libagx/query.h"
#include "asahi/libagx/gen1_as.h"
#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"

#include "drm-uapi/asahi_drm.h"
#include "util/os_time.h"
#include "util/u_dynarray.h"
#include "vulkan/vulkan_core.h"

struct hk_query_report {
   uint64_t value;
};
static_assert(sizeof(struct hk_query_report) == sizeof(uint64_t),
              "CPU result ABI requires contiguous u64 reports");

static inline bool
hk_has_available(const struct hk_query_pool *pool)
{
   return pool->vk.query_type != VK_QUERY_TYPE_TIMESTAMP;
}

static inline bool
hk_is_acceleration_structure_query(VkQueryType type)
{
   return type == VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR ||
          type == VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR ||
          type == VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SIZE_KHR ||
          type == VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_BOTTOM_LEVEL_POINTERS_KHR;
}

static uint16_t *
hk_pool_oq_index_ptr(const struct hk_query_pool *pool)
{
   /* A pool whose BO allocation failed (every BO creation fails once the VM
    * is lost) reaches here from hk_CreateQueryPool's error path. */
   if (!pool->bo)
      return NULL;
   void *map = agx_bo_map(pool->bo);
   return map ? (uint16_t *)((char *)map + pool->query_start) : NULL;
}

static uint32_t
hk_reports_per_query(struct hk_query_pool *pool)
{
   switch (pool->vk.query_type) {
   case VK_QUERY_TYPE_OCCLUSION:
   case VK_QUERY_TYPE_TIMESTAMP:
   case VK_QUERY_TYPE_PRIMITIVES_GENERATED_EXT:
   case VK_QUERY_TYPE_MESH_PRIMITIVES_GENERATED_EXT:
   case VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR:
   case VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR:
   case VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SIZE_KHR:
   case VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_BOTTOM_LEVEL_POINTERS_KHR:
      return 1;
   case VK_QUERY_TYPE_PIPELINE_STATISTICS:
      return util_bitcount(pool->vk.pipeline_statistics);
   case VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT:
      // Primitives succeeded and primitives needed
      return 2;
   default:
      UNREACHABLE("Unsupported query type");
   }
}

/*
 * Prepare the compute stream a query command (reset, result copy) records
 * into. Query state is updated in submission order without application
 * barriers (Vulkan 1.3, 17 "Queries": vkCmdCopyQueryPoolResults sees every
 * earlier vkCmdResetQueryPool on the queue, and vkCmdBeginQuery requires the
 * query to be unavailable, which an earlier reset in the same command buffer
 * provides). The stream therefore waits on every earlier stream, and it is
 * never the stream carrying a pending end timestamp. On G17P, both start
 * and end timestamps reach query memory at command retirement, so neither
 * producer may contain the query command or be merged back into it. The caller
 * marks later streams as ordered after it where the command's semantics require
 * that.
 */
static struct hk_cs *
hk_query_command_cs(struct hk_cmd_buffer *cmd)
{
   hk_sync_flush_pending_timestamp(cmd);
   hk_sync_end_cs_if_timestamped(cmd);

   struct hk_sync_dep dep[HK_SYNC_NUM_SQ];
   hk_sync_src_dep(cmd, HK_SYNC_ALL, cmd->current_cs.cs, dep);
   struct hk_cs *cs = hk_sync_get_cs_after(cmd, dep);
   if (cs && agx_native_graphics(&hk_cmd_buffer_device(cmd)->dev))
      cs->g17p_timestamp_query = true;
   return cs;
}

static uint64_t
hk_query_available_size(const struct hk_query_pool *pool)
{
   return sizeof(uint32_t) << pool->avail_shift;
}

static uint64_t
hk_query_available_offset(struct hk_query_pool *pool, uint32_t query)
{
   assert(hk_has_available(pool));
   assert(query < pool->vk.query_count);
   return (uint64_t)query * hk_query_available_size(pool);
}

static uint64_t
hk_query_available_addr(struct hk_query_pool *pool, uint32_t query)
{
   return pool->bo->va->addr + hk_query_available_offset(pool, query);
}

/* The availability slot of a query, nonzero once available. A 64-bit slot
 * holds 1 when a queued store made the query available and a retirement
 * timestamp when the kernel did (hk_cmd_end_query_stamped()). */
static uint64_t
hk_query_available_value(struct hk_query_pool *pool, uint32_t query)
{
   void *slot = (char *)agx_bo_map(pool->bo) +
                hk_query_available_offset(pool, query);
   if (pool->avail_shift)
      return p_atomic_read((uint64_t *)slot);
   return p_atomic_read((uint32_t *)slot);
}

static void
hk_query_set_available_host(struct hk_query_pool *pool, uint32_t query,
                            bool available)
{
   void *slot = (char *)agx_bo_map(pool->bo) +
                hk_query_available_offset(pool, query);
   if (pool->avail_shift)
      *(uint64_t *)slot = available;
   else
      *(uint32_t *)slot = available;
}

static uint64_t
hk_query_offset(struct hk_query_pool *pool, uint32_t query)
{
   assert(query < pool->vk.query_count);
   return pool->query_start + query * pool->query_stride;
}

static uint64_t
hk_query_report_addr(struct hk_device *dev, struct hk_query_pool *pool,
                     uint32_t query)
{
   if (pool->oq_queries) {
      uint16_t *oq_index = hk_pool_oq_index_ptr(pool);
      return dev->occlusion_queries.bo->va->addr +
             (oq_index[query] * sizeof(uint64_t));
   } else {
      return pool->bo->va->addr + hk_query_offset(pool, query);
   }
}

static struct hk_query_report *
hk_query_report_map(struct hk_device *dev, struct hk_query_pool *pool,
                    uint32_t query)
{
   if (pool->oq_queries) {
      uint64_t *queries = (uint64_t *)agx_bo_map(dev->occlusion_queries.bo);
      uint16_t *oq_index = hk_pool_oq_index_ptr(pool);

      return (struct hk_query_report *)&queries[oq_index[query]];
   } else {
      return (void *)((char *)agx_bo_map(pool->bo) +
                      hk_query_offset(pool, query));
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdWriteAccelerationStructuresPropertiesKHR(
   VkCommandBuffer commandBuffer, uint32_t accelerationStructureCount,
   const VkAccelerationStructureKHR *pAccelerationStructures,
   VkQueryType queryType, VkQueryPool queryPool, uint32_t firstQuery)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_query_pool, pool, queryPool);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   /* Live sidecar metadata includes retained update topology. Serialization
    * requires the matching native serialization format and
    * relocation path; do not invent a size for an unimplemented format. */
   if (!dev->g17p_native_as || pool->vk.query_type != queryType ||
       !hk_is_acceleration_structure_query(queryType) ||
       firstQuery > pool->vk.query_count ||
       accelerationStructureCount > pool->vk.query_count - firstQuery) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   if (!accelerationStructureCount || !hk_query_command_cs(cmd))
      return;

   for (uint32_t i = 0; i < accelerationStructureCount; ++i) {
      VK_FROM_HANDLE(vk_acceleration_structure, as, pAccelerationStructures[i]);
      uint64_t metadata = hk_gen1_as_metadata_addr(cmd, as);
      if (!metadata)
         return;
      libagx_gen1_as_property(cmd, agx_1d(1), AGX_BARRIER_ALL,
         vk_acceleration_structure_get_va(as),
         metadata, hk_query_report_addr(dev, pool, firstQuery + i),
         queryType == VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR ? 0 :
         queryType == VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SIZE_KHR ? 1 :
         queryType == VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR ? 2 : 3);
      if (vk_command_buffer_has_error(&cmd->vk))
         return;
      /* The helper's strong barrier completes the value before the existing
       * queue-write path makes availability system-wide coherent. */
      hk_queue_write(cmd, hk_query_available_addr(pool, firstQuery + i), 1,
                      false);
   }
}

void
hk_dispatch_imm_writes(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   /* As soon as we mark a query available, it needs to be available system
    * wide, otherwise a CPU-side get result can query. As such, we cache flush
    * before and then let coherency works its magic. Without this barrier, we
    * get flakes in
    *
    * dEQP-VK.query_pool.occlusion_query.get_results_conservative_size_64_wait_query_without_availability_draw_triangles_discard
    */
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   hk_cdm_cache_flush(dev, cs);

   perf_debug(cmd, "Queued writes");

   uint32_t count =
      util_dynarray_num_elements(&cs->imm_writes, struct libagx_imm_write);
   assert(count > 0);

   uint64_t params =
      hk_pool_upload(cmd, cs->imm_writes.data, cs->imm_writes.size, 16);
   if (vk_command_buffer_has_error(&cmd->vk))
      return;

   if (agx_native_graphics(&dev->dev)) {
      /* End-of-stream writes belong to this exact post-graphics stream.
       * Pin it while the common native helper obtains its CDM stream; it
       * must not move availability ahead of the VDM producer. */
      struct hk_cs *saved = cmd->current_cs.cs;
      bool pinned = cmd->sync.pin_cs;
      cmd->current_cs.cs = cs;
      cmd->sync.pin_cs = true;
      libagx_write_u32s(cmd, agx_1d(count), AGX_BARRIER_ALL, params);
      cmd->current_cs.cs = saved;
      cmd->sync.pin_cs = pinned;

      if (vk_command_buffer_has_error(&cmd->vk))
         return;
      util_dynarray_foreach(&cs->imm_writes, struct libagx_imm_write, write) {
         hk_drm_attachment_set_add(&cs->compute_attachments, write->address,
                                    sizeof(write->value));
      }
      return;
   }

   libagx_write_u32s(cmd, agx_1d(count), AGX_BARRIER_ALL | AGX_POSTGFX, params);
}

void
hk_queue_write(struct hk_cmd_buffer *cmd, uint64_t address, uint32_t value,
               bool after_gfx)
{
   struct hk_cs *cs = hk_cmd_buffer_get_cs_general(
      cmd, after_gfx ? &cmd->current_cs.post_gfx : &cmd->current_cs.cs, true);
   if (!cs)
      return;

   /* TODO: Generalize this mechanism suitably */
   if (after_gfx) {
      struct libagx_imm_write imm = {.address = address, .value = value};

      if (!cs->imm_writes.data) {
         cs->imm_writes = UTIL_DYNARRAY_INIT;
      }

      util_dynarray_append(&cs->imm_writes, imm);
      return;
   }

   /* As soon as we mark a query available, it needs to be available system
    * wide, otherwise a CPU-side get result can query. As such, we cache flush
    * before and then let coherency works its magic. Without this barrier, we
    * get flakes in
    *
    * dEQP-VK.query_pool.occlusion_query.get_results_conservative_size_64_wait_query_without_availability_draw_triangles_discard
    */
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   hk_cdm_cache_flush(dev, cs);

   perf_debug(cmd, "Queued write");
   if (agx_native_graphics(&dev->dev)) {
      hk_g17p_meta_fill_addr(cmd, &cmd->current_cs.cs, address,
                             sizeof(value), value);
      return;
   }

   libagx_write_u32(cmd, agx_1d(1), AGX_BARRIER_ALL, address, value);
}

/*
 * Native AGX3 occlusion availability.
 *
 * A render's retirement record carries four timestamp slots (vertex start
 * and end, fragment start and end). The kernel writes every requested slot
 * when it retires the render, which is after the end-of-render cache
 * maintenance that makes the render's occlusion counts visible, and before
 * it signals the submission. A nonzero timestamp in a query's 64-bit
 * availability slot is therefore exactly "available".
 *
 * The queued post-graphics store remains the general mechanism: it is
 * recorded as usual, and the stream that would carry it is kept empty until
 * the end of the command buffer. Only then is it known whether any later
 * compute work in this command buffer depends on the render; such work would
 * have to wait for the render's host-side retirement once the render writes
 * timestamps, where it now waits in firmware. When nothing does and the
 * render has free slots, the render carries the availability and the extra
 * compute command is dropped. Otherwise the store is emitted as before.
 */
struct hk_oq_stamp {
   uint32_t handle;
   uint32_t offset_B;
   uint32_t render_seq;
};

static void
hk_cmd_end_query_stamped(struct hk_cmd_buffer *cmd, struct hk_query_pool *pool,
                         uint32_t query)
{
   const uint32_t render_seq = cmd->current_cs.gfx->sync.seq;
   hk_queue_write(cmd, hk_query_available_addr(pool, query), 1, true);

   struct hk_cs *cs = cmd->current_cs.post_gfx;
   if (!cs || vk_command_buffer_has_error(&cmd->vk))
      return;

   if (!cs->oq_stamps.data)
      cs->oq_stamps = UTIL_DYNARRAY_INIT;

   const struct hk_oq_stamp stamp = {
      .handle = pool->handle,
      .offset_B = hk_query_available_offset(pool, query),
      .render_seq = render_seq,
   };
   util_dynarray_append(&cs->oq_stamps, stamp);
}

bool
hk_defer_oq_available(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   const unsigned writes =
      util_dynarray_num_elements(&cs->imm_writes, struct libagx_imm_write);
   const unsigned stamps =
      util_dynarray_num_elements(&cs->oq_stamps, struct hk_oq_stamp);

   if (!stamps || stamps != writes || cs->stats.cmds ||
       cs->stats.flushes || cs->timestamp.start.handle ||
       cs->timestamp.end.handle || cs->query_reset_pending)
      return false;

   const struct hk_oq_stamp *s = cs->oq_stamps.data;
   for (unsigned i = 1; i < stamps; ++i) {
      if (s[i].render_seq != s[0].render_seq)
         return false;
   }

   cs->oq_avail_deferred = true;
   return true;
}

static bool
hk_dep_reaches(const struct hk_sync_dep *dep, const struct hk_cs *cs)
{
   return dep->cut && dep->cut->sync.seq >= cs->sync.seq;
}

enum hk_oq_resolution {
   /* Emit the store in place. */
   HK_OQ_KEEP,
   /* The render carries the availability; drop the stream. */
   HK_OQ_STAMPED,
   /* Nothing depends on the stream or (from compute) on its render, but the
    * render has too few free slots: the store may move to a later deferred
    * stream that waits on a later render. */
   HK_OQ_MOVABLE,
};

/* Whether the render can carry the availability of the deferred stream post:
 * nothing later in the command buffer waits on post, no compute waits on the
 * render (or anything after it on the render subqueue), the command buffer's
 * tail requirement does not name it, and enough timestamp slots are free.
 */
static enum hk_oq_resolution
hk_stamp_oq_available(struct hk_cmd_buffer *cmd, struct hk_cs *post)
{
   const struct hk_oq_stamp *stamps = post->oq_stamps.data;
   const unsigned count =
      util_dynarray_num_elements(&post->oq_stamps, struct hk_oq_stamp);

   /* While deferred, the stream may have become the tail an API timestamp
    * rides on (hk_sync_write_timestamp()). It then stays, with its store. */
   if (post->timestamp.start.handle || post->timestamp.end.handle ||
       post->stats.cmds || post->stats.flushes)
      return HK_OQ_KEEP;

   struct hk_cs *render = NULL;
   list_for_each_entry(struct hk_cs, it, &cmd->control_streams, node) {
      if (it == post)
         break;
      if (it->type == HK_CS_VDM && it->sync.seq == stamps[0].render_seq)
         render = it;
   }
   if (!render || post->sync.dep[HK_SYNC_VDM].cut != render)
      return HK_OQ_KEEP;

   list_for_each_entry(struct hk_cs, it, &cmd->control_streams, node) {
      if (it == post || it == render)
         continue;
      if (hk_dep_reaches(&it->sync.dep[HK_SYNC_CDM], post) ||
          hk_dep_reaches(&it->sync.dep_ta[HK_SYNC_CDM], post))
         return HK_OQ_KEEP;
      if (it->type == HK_CS_CDM &&
          (hk_dep_reaches(&it->sync.dep[HK_SYNC_VDM], render) ||
           hk_dep_reaches(&it->sync.dep_ta[HK_SYNC_VDM], render)))
         return HK_OQ_KEEP;
   }

   for (unsigned k = 0; k < HK_SYNC_NUM_KINDS; ++k) {
      if (hk_dep_reaches(&cmd->sync.req[k][HK_SYNC_VDM], render) ||
          hk_dep_reaches(&cmd->sync.req[k][HK_SYNC_CDM], post))
         return HK_OQ_KEEP;
   }

   /* Fragment end first, then the slots no API timestamp uses, then vertex
    * start. A slot an API timestamp rides on stays with it. */
   const unsigned order[] = {3, 1, 2, 0};
   unsigned free_slots[4], nfree = 0;
   for (unsigned i = 0; i < ARRAY_SIZE(order); ++i) {
      const unsigned slot = order[i];
      if ((slot == 0 && render->timestamp.start.handle) ||
          (slot == 3 && render->timestamp.end.handle) ||
          render->oq_stamp[slot].handle)
         continue;
      free_slots[nfree++] = slot;
   }
   if (nfree < count)
      return HK_OQ_MOVABLE;

   for (unsigned i = 0; i < count; ++i) {
      render->oq_stamp[free_slots[i]] = (struct agx_timestamp_req){
         .handle = stamps[i].handle,
         .offset_B = stamps[i].offset_B,
      };
   }
   return HK_OQ_STAMPED;
}

static void
hk_drop_oq_stream(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   hk_sync_forget_stream(cmd, cs);
   list_del(&cs->node);
   hk_cs_destroy(cs);
}

void
hk_resolve_deferred_oq_available(struct hk_cmd_buffer *cmd)
{
   /* Stores that could not ride on their render but that nothing waits on
    * collect in the latest such stream: it waits on a render that completes
    * after theirs, so one compute command serves the whole command buffer. */
   struct hk_cs *movable = NULL;

   list_for_each_entry_safe(struct hk_cs, cs, &cmd->control_streams, node) {
      if (cs->type != HK_CS_CDM || !cs->oq_avail_deferred)
         continue;

      cs->oq_avail_deferred = false;
      switch (hk_stamp_oq_available(cmd, cs)) {
      case HK_OQ_STAMPED:
         hk_drop_oq_stream(cmd, cs);
         break;
      case HK_OQ_MOVABLE:
         /* The later stream's render wait covers every earlier render. */
         if (movable && movable->sync.dep[HK_SYNC_VDM].cut->sync.seq >
                           cs->sync.dep[HK_SYNC_VDM].cut->sync.seq) {
            hk_dispatch_imm_writes(cmd, movable);
            movable = NULL;
         }
         if (movable) {
            util_dynarray_append_dynarray(&cs->imm_writes, &movable->imm_writes);
            hk_drop_oq_stream(cmd, movable);
         }
         movable = cs;
         break;
      case HK_OQ_KEEP:
         hk_dispatch_imm_writes(cmd, cs);
         break;
      }
   }

   if (movable)
      hk_dispatch_imm_writes(cmd, movable);
}

static void
emit_zero_queries_g17p(struct hk_cmd_buffer *cmd, struct hk_query_pool *pool,
                       uint32_t first_index, uint32_t num_queries,
                       bool set_available)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   if (first_index > pool->vk.query_count ||
       num_queries > pool->vk.query_count - first_index) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   if (num_queries == 0)
      return;

   if (pool->vk.query_type != VK_QUERY_TYPE_TIMESTAMP &&
       pool->vk.query_type != VK_QUERY_TYPE_OCCLUSION &&
       pool->vk.query_type != VK_QUERY_TYPE_PIPELINE_STATISTICS &&
       pool->vk.query_type != VK_QUERY_TYPE_MESH_PRIMITIVES_GENERATED_EXT &&
       pool->vk.query_type != VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT &&
       !hk_is_acceleration_structure_query(pool->vk.query_type)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }

   const uint32_t reports = hk_reports_per_query(pool);
   if (!reports || pool->query_stride !=
                      (uint64_t)reports * sizeof(struct hk_query_report) ||
       (pool->vk.query_type == VK_QUERY_TYPE_OCCLUSION &&
        pool->oq_queries != pool->vk.query_count)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }

   /* Use the same indexed batch as AGX2. The hardware counter heap is not
    * necessarily contiguous, so the kernel follows the pool's u16 remap.
    * Publishing availability is a separate operation when synthesizing
    * multiview results: the helper's strong barrier makes every zero report
    * visible before the following availability store. */
   const bool publish_available = set_available && hk_has_available(pool);
   struct libagx_reset_query_args info = {
      .availability = hk_has_available(pool) ? pool->bo->va->addr : 0,
      .results = pool->oq_queries ? dev->occlusion_queries.bo->va->addr
                                  : pool->bo->va->addr + pool->query_start,
      .oq_index = pool->oq_queries ? pool->bo->va->addr + pool->query_start : 0,
      .first_query = first_index,
      .reports_per_query = reports,
      .set_available = set_available && !publish_available,
      .availability_shift = pool->avail_shift,
   };
   struct hk_cs *cs = cmd->current_cs.cs;
   assert(set_available || cs);
   if (!set_available && !cs->query_reset_pending && cs->stats.cmds) {
      /* Earlier copies may still be reading this pool. Only consecutive
       * resets can omit this dependency. */
      hk_ensure_cs_has_space(cmd, cs, AGX3_CDM_BARRIER_SIZE);
      hk_cdm_cache_flush(dev, cs);
   }

   libagx_reset_query_struct(cmd, agx_1d(num_queries),
                             set_available ? AGX_BARRIER_ALL : AGX_BARRIER_NONE,
                             info);
   if (vk_command_buffer_has_error(&cmd->vk))
      return;

   cs = cmd->current_cs.cs;
   assert(cs);
   if (!set_available) {
      cs->query_reset_pending = true;
      cs->stats.query_reset_dispatches++;
      cs->stats.query_reset_queries += num_queries;
   }
   if (hk_has_available(pool)) {
      hk_drm_attachment_set_add(&cs->compute_attachments,
         hk_query_available_addr(pool, first_index),
         (uint64_t)num_queries * hk_query_available_size(pool));
   }
   if (pool->oq_queries) {
      for (uint32_t i = 0; i < num_queries; ++i) {
         hk_drm_attachment_set_add(&cs->compute_attachments,
            hk_query_report_addr(dev, pool, first_index + i), sizeof(uint64_t));
      }
   } else {
      hk_drm_attachment_set_add(&cs->compute_attachments,
         hk_query_report_addr(dev, pool, first_index),
         (uint64_t)num_queries * pool->query_stride);
   }

   if (publish_available) {
      hk_g17p_meta_fill_addr(cmd, &cmd->current_cs.cs,
         hk_query_available_addr(pool, first_index),
         (uint64_t)num_queries * hk_query_available_size(pool), 1);
   }
}

static void
emit_zero_queries(struct hk_cmd_buffer *cmd, struct hk_query_pool *pool,
                  uint32_t first_index, uint32_t num_queries,
                  bool set_available)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   perf_debug(cmd, "Query pool zero");

   if (agx_native_graphics(&dev->dev)) {
      emit_zero_queries_g17p(cmd, pool, first_index, num_queries,
                             set_available);
      return;
   }

   struct libagx_reset_query_args info = {
      .availability = hk_has_available(pool) ? pool->bo->va->addr : 0,
      .results = pool->oq_queries ? dev->occlusion_queries.bo->va->addr
                                  : pool->bo->va->addr + pool->query_start,
      .oq_index = pool->oq_queries ? pool->bo->va->addr + pool->query_start : 0,

      .first_query = first_index,
      .reports_per_query = hk_reports_per_query(pool),
      .set_available = set_available,
      .availability_shift = pool->avail_shift,
   };

   libagx_reset_query_struct(cmd, agx_1d(num_queries), AGX_BARRIER_ALL, info);
}

static void
host_zero_queries(struct hk_device *dev, struct hk_query_pool *pool,
                  uint32_t first_index, uint32_t num_queries,
                  bool set_available)
{
   if (pool->vk.query_type == VK_QUERY_TYPE_PIPELINE_STATISTICS) {
      const struct hk_pipeline_statistics_layout layout = {
         .query_count = pool->vk.query_count,
         .reports_per_query = hk_reports_per_query(pool),
         .query_start_B = pool->query_start,
         .query_stride_B = pool->query_stride,
         .size_B = pool->query_start +
                   ((uint64_t)pool->vk.query_count * pool->query_stride),
      };
      hk_pipeline_statistics_reset_host(agx_bo_map(pool->bo), &layout,
                                        first_index, num_queries,
                                        set_available);
      return;
   }

   for (uint32_t i = 0; i < num_queries; i++) {
      struct hk_query_report *reports =
         hk_query_report_map(dev, pool, first_index + i);

      uint64_t value = 0;
      if (hk_has_available(pool)) {
         hk_query_set_available_host(pool, first_index + i, set_available);
      } else {
         value = set_available ? 0 : LIBAGX_QUERY_UNAVAILABLE;
      }

      for (unsigned j = 0; j < hk_reports_per_query(pool); ++j) {
         reports[j].value = value;
      }
   }
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_CreateQueryPool(VkDevice device, const VkQueryPoolCreateInfo *pCreateInfo,
                   const VkAllocationCallbacks *pAllocator,
                   VkQueryPool *pQueryPool)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   struct hk_query_pool *pool;

   bool occlusion = pCreateInfo->queryType == VK_QUERY_TYPE_OCCLUSION;
   bool timestamp = pCreateInfo->queryType == VK_QUERY_TYPE_TIMESTAMP;
   unsigned occlusion_queries = occlusion ? pCreateInfo->queryCount : 0;

   pool =
      vk_query_pool_create(&dev->vk, pCreateInfo, pAllocator, sizeof(*pool));
   if (!pool)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   /* Native AGX3 occlusion availability lives in 64-bit slots of a bound
    * timestamp buffer (hk_cmd_end_query_stamped()). */
   const bool stamped = occlusion && agx_native_graphics(&dev->dev);
   pool->avail_shift = stamped ? 1 : 0;

   /* We place the availability first and then data. Pipeline-statistics pools
    * use the independently tested 64-bit layout path so large query counts
    * cannot wrap a 32-bit BO size. */
   size_t bo_size = 0;
   if (pool->vk.query_type == VK_QUERY_TYPE_PIPELINE_STATISTICS) {
      struct hk_pipeline_statistics_layout layout;
      if (!hk_pipeline_statistics_layout_init(
             pool->vk.query_count, pool->vk.pipeline_statistics, &layout) ||
          layout.size_B > SIZE_MAX) {
         vk_query_pool_destroy(&dev->vk, pAllocator, &pool->vk);
         return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      }

      pool->query_start = layout.query_start_B;
      pool->query_stride = layout.query_stride_B;
      bo_size = layout.size_B;
   } else {
      pool->query_start = 0;
      if (hk_has_available(pool)) {
         pool->query_start =
            align64((uint64_t)pool->vk.query_count *
                       hk_query_available_size(pool),
                    sizeof(struct hk_query_report));
      }

      const uint32_t reports_per_query = hk_reports_per_query(pool);
      pool->query_stride =
         (uint64_t)reports_per_query * sizeof(struct hk_query_report);
   }

   if (pool->vk.query_count > 0) {
      if (pool->vk.query_type != VK_QUERY_TYPE_PIPELINE_STATISTICS) {
         bo_size = pool->query_start;

         /* For occlusion queries, we stick the query index remapping here */
         if (occlusion_queries)
            bo_size += sizeof(uint16_t) * (size_t)pool->vk.query_count;
         else
            bo_size += pool->query_stride * (size_t)pool->vk.query_count;
      }

      /* The kernel requires that timestamp buffers are SHARED */
      enum agx_bo_flags flags = AGX_BO_WRITEBACK;
      if (timestamp || stamped)
         flags |= AGX_BO_SHARED;

      pool->bo = agx_bo_create(&dev->dev, bo_size, 0, flags, "Query pool");
      if (!pool->bo || !agx_bo_map(pool->bo)) {
         hk_DestroyQueryPool(device, hk_query_pool_to_handle(pool), pAllocator);
         return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      }

      /* Timestamp buffers must be explicitly bound as such before we can use
       * them.
       */
      if (timestamp) {
         int ret = agx_bind_timestamps(&dev->dev, pool->bo, &pool->handle);
         if (ret) {
            hk_DestroyQueryPool(device, hk_query_pool_to_handle(pool),
                                pAllocator);
            return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         }

         assert(pool->handle && "handles are nonzero");
      }

      /* Without a binding, every query end keeps the queued store. */
      if (stamped && agx_bind_timestamps(&dev->dev, pool->bo, &pool->handle))
         pool->handle = 0;
   }

   uint16_t *oq_index = hk_pool_oq_index_ptr(pool);
   assert(oq_index || !occlusion_queries);

   /* See hk_oq_submission_done(): only the native AGX3 path needs the
    * submission point. */
   if (occlusion_queries && agx_native_graphics(&dev->dev)) {
      pool->oq_submit =
         vk_zalloc2(&dev->vk.alloc, pAllocator,
                    sizeof(*pool->oq_submit) * occlusion_queries, 8,
                    VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
      if (!pool->oq_submit) {
         hk_DestroyQueryPool(device, hk_query_pool_to_handle(pool), pAllocator);
         return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
      }
   }

   for (unsigned i = 0; i < occlusion_queries; ++i) {
      uint64_t zero = 0;
      unsigned index;

      VkResult result = hk_descriptor_table_add(
         dev, &dev->occlusion_queries, &zero, sizeof(uint64_t), &index);

      if (result != VK_SUCCESS) {
         hk_DestroyQueryPool(device, hk_query_pool_to_handle(pool), pAllocator);
         return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      }

      /* We increment as we go so we can clean up properly if we run out */
      assert(pool->oq_queries < occlusion_queries);
      oq_index[pool->oq_queries++] = index;
   }

   if (pCreateInfo->flags & VK_QUERY_POOL_CREATE_RESET_BIT_KHR)
      host_zero_queries(dev, pool, 0, pool->vk.query_count, false);

   *pQueryPool = hk_query_pool_to_handle(pool);

   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
hk_DestroyQueryPool(VkDevice device, VkQueryPool queryPool,
                    const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   VK_FROM_HANDLE(hk_query_pool, pool, queryPool);

   if (!pool)
      return;

   uint16_t *oq_index = hk_pool_oq_index_ptr(pool);

   for (unsigned i = 0; oq_index && i < pool->oq_queries; ++i) {
      hk_descriptor_table_remove(dev, &dev->occlusion_queries, oq_index[i]);
   }

   if (pool->handle)
      dev->dev.ops.bo_unbind_object(&dev->dev, pool->handle);

   agx_bo_unreference(&dev->dev, pool->bo);
   vk_free2(&dev->vk.alloc, pAllocator, pool->oq_submit);
   vk_query_pool_destroy(&dev->vk, pAllocator, &pool->vk);
}

VKAPI_ATTR void VKAPI_CALL
hk_ResetQueryPool(VkDevice device, VkQueryPool queryPool, uint32_t firstQuery,
                  uint32_t queryCount)
{
   VK_FROM_HANDLE(hk_query_pool, pool, queryPool);
   VK_FROM_HANDLE(hk_device, dev, device);

   host_zero_queries(dev, pool, firstQuery, queryCount, false);

   for (uint32_t i = 0; pool->oq_submit && i < queryCount; i++)
      p_atomic_set(&pool->oq_submit[firstQuery + i].point, 0);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdResetQueryPool(VkCommandBuffer commandBuffer, VkQueryPool queryPool,
                     uint32_t firstQuery, uint32_t queryCount)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_query_pool, pool, queryPool);

   perf_debug(cmd, "Reset query pool");

   /* The reset runs after every earlier use of the pool and before every
    * later one: it is a transfer operation (Vulkan 1.3, 17.1) whose result
    * every later query command on the queue observes without further
    * synchronization. Zeroing is compute on every generation, so the
    * ordering is one compute stream everything after waits on. */
   if (!queryCount)
      return;

   cmd->sync.kind_override = HK_SYNC_ALL;
   cmd->state.query_reset_recording = true;
   struct hk_cs *cs = hk_query_command_cs(cmd);
   if (cs)
      emit_zero_queries(cmd, pool, firstQuery, queryCount, false);
   cmd->state.query_reset_recording = false;
   cmd->sync.kind_override = 0;
   if (!cs)
      return;

   hk_sync_after_stream(cmd, cmd->current_cs.cs ? cmd->current_cs.cs : cs,
                        HK_SYNC_ALL);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdWriteTimestamp2(VkCommandBuffer commandBuffer,
                      VkPipelineStageFlags2 stage, VkQueryPool queryPool,
                      uint32_t query)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   hk_m5_pass_merge_end_ordered(cmd, "timestamp");
   hk_flush_deferred_clears(cmd);
   VK_FROM_HANDLE(hk_query_pool, pool, queryPool);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   const struct agx_timestamp_req req = {
      .addr = hk_query_report_addr(dev, pool, query),
      .handle = pool->handle,
      .offset_B = hk_query_offset(pool, query),
   };

   /* The stamp rides on a command's native start or end timestamp; which
    * command, and what it must wait on, follows from the stage. */
   hk_sync_write_timestamp(cmd, stage, req);
   if (vk_command_buffer_has_error(&cmd->vk))
      return;

   /* From the Vulkan spec:
    *
    *   "If vkCmdWriteTimestamp2 is called while executing a render pass
    *    instance that has multiview enabled, the timestamp uses N consecutive
    *    query indices in the query pool (starting at query) where N is the
    *    number of bits set in the view mask of the subpass the command is
    *    executed in. The resulting query values are determined by an
    *    implementation-dependent choice of one of the following behaviors:"
    *
    * In our case, only the first query is used, so we emit zeros for the
    * remaining queries, as described in the first behavior listed in the
    * Vulkan spec:
    *
    *   "The first query is a timestamp value and (if more than one bit is set
    *   in the view mask) zero is written to the remaining queries."
    */
   if (cmd->state.gfx.render.view_mask != 0) {
      const uint32_t num_queries =
         util_bitcount(cmd->state.gfx.render.view_mask);
      if (num_queries > 1)
         emit_zero_queries(cmd, pool, query + 1, num_queries - 1, true);
   }
}

static void
hk_cmd_begin_end_query(struct hk_cmd_buffer *cmd, struct hk_query_pool *pool,
                       uint32_t query, uint32_t index,
                       VkQueryControlFlags flags, bool end)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   bool graphics = false;

   switch (pool->vk.query_type) {
   case VK_QUERY_TYPE_OCCLUSION: {
      assert(query < pool->oq_queries);

      if (end) {
         cmd->state.gfx.occlusion.mode = AGX_VISIBILITY_MODE_NONE;
      } else {
         cmd->state.gfx.occlusion.mode = flags & VK_QUERY_CONTROL_PRECISE_BIT
                                            ? AGX_VISIBILITY_MODE_COUNTING
                                            : AGX_VISIBILITY_MODE_BOOLEAN;
         cmd->state.gfx.occlusion.begin_render_seq =
            cmd->current_cs.gfx ? cmd->current_cs.gfx->sync.seq : 0;
      }

      uint16_t *oq_index = hk_pool_oq_index_ptr(pool);
      cmd->state.gfx.occlusion.index = oq_index[query];
      cmd->state.gfx.dirty |= HK_DIRTY_OCCLUSION;
      if (end && pool->oq_submit) {
         const struct hk_oq_end oq_end = {.pool = pool, .query = query};
         util_dynarray_append(&cmd->oq_ends, oq_end);
      }
      graphics = true;
      break;
   }

   case VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT: {
      uint64_t addr = hk_query_report_addr(dev, pool, query);
      cmd->state.gfx.xfb_query[index] = end ? 0 : addr;
      break;
   }

   case VK_QUERY_TYPE_MESH_PRIMITIVES_GENERATED_EXT: {
      cmd->state.gfx.descriptors.root_dirty = true;
      cmd->state.gfx.descriptors.root.mesh_prims_query =
         end ? 0 : hk_query_report_addr(dev, pool, query);
      graphics = true;
      break;
   }

   case VK_QUERY_TYPE_PIPELINE_STATISTICS: {
      struct hk_root_descriptor_table *root = &cmd->state.gfx.descriptors.root;
      cmd->state.gfx.descriptors.root_dirty = true;

      root->draw.pipeline_stats =
         end ? 0 : hk_query_report_addr(dev, pool, query);
      root->draw.pipeline_stats_flags =
         end ? 0 : pool->vk.pipeline_statistics;

      /* IA/VS/clip counters may be written by the pre-graphics helper and FS
       * counters by the VDM workload itself, so any graphics statistic makes
       * availability a post-graphics write. A compute-only query increments
       * and publishes availability in the same ordered general CDM stream.
       * Mixed queries also select post-graphics; when no VDM is live that
       * stream is appended after the current compute stream. */
      graphics = pool->vk.pipeline_statistics &
                 ~VK_QUERY_PIPELINE_STATISTIC_COMPUTE_SHADER_INVOCATIONS_BIT;
      break;
   }

   default:
      UNREACHABLE("Unsupported query type");
   }

   /* We need to set available=1 after the graphics work finishes. */
   if (end) {
      perf_debug(cmd, "Query ending, type %u", pool->vk.query_type);
      /* Renders of one queue may finish out of order: a query that spans
       * renders keeps the store, which waits on every render before it. */
      if (pool->vk.query_type == VK_QUERY_TYPE_OCCLUSION && pool->handle &&
          cmd->current_cs.gfx &&
          cmd->current_cs.gfx->sync.seq ==
             cmd->state.gfx.occlusion.begin_render_seq)
         hk_cmd_end_query_stamped(cmd, pool, query);
      else
         hk_queue_write(cmd, hk_query_available_addr(pool, query), 1,
                        graphics);
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBeginQueryIndexedEXT(VkCommandBuffer commandBuffer, VkQueryPool queryPool,
                           uint32_t query, VkQueryControlFlags flags,
                           uint32_t index)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   hk_m5_pass_merge_end_ordered(cmd, "query");
   hk_flush_deferred_clears(cmd);
   VK_FROM_HANDLE(hk_query_pool, pool, queryPool);

   hk_cmd_begin_end_query(cmd, pool, query, index, flags, false);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdEndQueryIndexedEXT(VkCommandBuffer commandBuffer, VkQueryPool queryPool,
                         uint32_t query, uint32_t index)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   hk_m5_pass_merge_end_ordered(cmd, "query");
   hk_flush_deferred_clears(cmd);
   VK_FROM_HANDLE(hk_query_pool, pool, queryPool);

   hk_cmd_begin_end_query(cmd, pool, query, index, 0, true);

   /* From the Vulkan spec:
    *
    *   "If queries are used while executing a render pass instance that has
    *    multiview enabled, the query uses N consecutive query indices in
    *    the query pool (starting at query) where N is the number of bits set
    *    in the view mask in the subpass the query is used in. How the
    *    numerical results of the query are distributed among the queries is
    *    implementation-dependent."
    *
    * In our case, only the first query is used, so we emit zeros for the
    * remaining queries.
    */
   if (cmd->state.gfx.render.view_mask != 0) {
      const uint32_t num_queries =
         util_bitcount(cmd->state.gfx.render.view_mask);
      if (num_queries > 1) {
         perf_debug(cmd, "Multiview query zeroing");
         emit_zero_queries(cmd, pool, query + 1, num_queries - 1, true);
      }
   }
}

static bool
hk_query_is_available(struct hk_device *dev, struct hk_query_pool *pool,
                      uint32_t query)
{
   if (hk_has_available(pool)) {
      return hk_query_available_value(pool, query) != 0;
   } else {
      const struct hk_query_report *report =
         hk_query_report_map(dev, pool, query);

      return report->value != LIBAGX_QUERY_UNAVAILABLE;
   }
}

#define HK_QUERY_TIMEOUT 2000000000ull

void
hk_query_note_submit(struct hk_cmd_buffer *cmd, struct hk_queue *queue,
                     uint64_t point)
{
   util_dynarray_foreach(&cmd->oq_ends, struct hk_oq_end, e) {
      struct hk_oq_submit *s = &e->pool->oq_submit[e->query];
      /* Readers check the point again after the queue: clear it first so
       * they never pair this queue with the previous point. */
      p_atomic_set(&s->point, 0);
      p_atomic_set(&s->queue, queue);
      p_atomic_set(&s->point, point);
   }
}

static void
hk_queue_note_signalled(struct hk_queue *queue, uint64_t point)
{
   uint64_t seen = p_atomic_read(&queue->drm.signalled_value);
   while (seen < point) {
      const uint64_t prev =
         p_atomic_cmpxchg(&queue->drm.signalled_value, seen, point);
      if (prev == seen)
         break;
      seen = prev;
   }
}

/*
 * A render's occlusion counts reach memory with its end-of-render cache
 * maintenance. The availability word is written by a compute stream that
 * the firmware starts once the render's fragment work is done, which can be
 * before that maintenance: the CPU then reads availability 1 and a count of
 * 0 that is right shortly after. The kernel signals a submission only
 * after the maintenance of all its renders, so a CPU read of an occlusion
 * result also requires the submission that ended the query.
 *
 * This is the native AGX3 firmware path: A18 Pro and M4 Pro both return
 * early zero counts. G13 and G14 read correct counts as soon as the
 * availability word is set (0 of 51,200 reads wrong, where the A18 Pro
 * got 31), so they keep that cheaper and earlier check alone.
 */
static VkResult
hk_oq_submission_done(struct hk_device *dev, struct hk_query_pool *pool,
                      uint32_t query, bool wait, bool *done)
{
   *done = true;
   if (!pool->oq_submit)
      return VK_SUCCESS;

   struct hk_oq_submit *s = &pool->oq_submit[query];
   uint64_t point;
   struct hk_queue *queue;
   do {
      point = p_atomic_read(&s->point);
      queue = p_atomic_read(&s->queue);
   } while (point != p_atomic_read(&s->point));
   /* Timeline values only grow, so a point at or below one this queue has
    * been seen to reach is complete: one syncobj wait per submission, not
    * one per query read. */
   if (!point || point <= p_atomic_read(&queue->drm.signalled_value))
      return VK_SUCCESS;

   uint32_t syncobj = queue->drm.syncobj;

   /* Bounded waits so a lost device is noticed; a hung submission ends in
    * the kernel's own timeout. */
   for (;;) {
      int64_t timeout = wait ? (int64_t)os_time_get_absolute_timeout(100000000)
                             : 0;
      /* WAIT_FOR_SUBMIT: the point is noted just before the submit ioctl. */
      int ret = drmSyncobjTimelineWait(dev->dev.fd, &syncobj, &point, 1,
                                       timeout,
                                       DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT,
                                       NULL);
      if (ret == 0) {
         hk_queue_note_signalled(queue, point);
         return VK_SUCCESS;
      }
      if (ret != -ETIME)
         return vk_device_set_lost(&dev->vk,
                                   "occlusion query submission wait: %d", ret);
      if (!wait) {
         *done = false;
         return VK_SUCCESS;
      }

      VkResult status = vk_device_check_status(&dev->vk);
      if (status != VK_SUCCESS)
         return status;
   }
}

static VkResult
hk_query_wait_for_available(struct hk_device *dev, struct hk_query_pool *pool,
                            uint32_t query)
{
   uint64_t abs_timeout_ns = os_time_get_absolute_timeout(HK_QUERY_TIMEOUT);

   while (os_time_get_nano() < abs_timeout_ns) {
      if (hk_query_is_available(dev, pool, query))
         return VK_SUCCESS;

      VkResult status = vk_device_check_status(&dev->vk);
      if (status != VK_SUCCESS)
         return status;
   }

   return vk_device_set_lost(&dev->vk, "query timeout");
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_GetQueryPoolResults(VkDevice device, VkQueryPool queryPool,
                       uint32_t firstQuery, uint32_t queryCount,
                       size_t dataSize, void *pData, VkDeviceSize stride,
                       VkQueryResultFlags flags)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   VK_FROM_HANDLE(hk_query_pool, pool, queryPool);

   if (vk_device_is_lost(&dev->vk))
      return VK_ERROR_DEVICE_LOST;

   VkResult status = VK_SUCCESS;
   for (uint32_t i = 0; i < queryCount; i++) {
      const uint32_t query = firstQuery + i;

      bool available = hk_query_is_available(dev, pool, query);

      if (!available && (flags & VK_QUERY_RESULT_WAIT_BIT)) {
         status = hk_query_wait_for_available(dev, pool, query);
         if (status != VK_SUCCESS)
            return status;

         available = true;
      }

      /* A retirement timestamp (any value but the queued store's 1) was
       * written after the end-of-render maintenance: no submission wait. */
      if (available &&
          !(pool->avail_shift && hk_query_available_value(pool, query) > 1)) {
         VkResult done = hk_oq_submission_done(
            dev, pool, query, flags & VK_QUERY_RESULT_WAIT_BIT, &available);
         if (done != VK_SUCCESS)
            return done;
      }

      const struct hk_query_report *src = hk_query_report_map(dev, pool, query);
      assert(i * stride < dataSize);
      void *dst = (char *)pData + i * stride;

      uint32_t reports = hk_reports_per_query(pool);
      if (!hk_query_write_cpu_result(dst, &src[0].value, reports, available,
                                     flags))
         status = VK_NOT_READY;
   }

   /* Availability can already be set when entering this call. Check GPU abort
    * status after observing it too, including robust-index allocation failure. */
   VkResult device_status = vk_device_check_status(&dev->vk);
   return device_status == VK_SUCCESS ? status : device_status;
}

static void
hk_cmd_copy_query_pool_results(struct hk_cmd_buffer *cmd,
                               struct hk_query_pool *pool, uint32_t firstQuery,
                               uint32_t queryCount, struct hk_buffer *dst_buffer,
                               VkDeviceSize dstOffset, VkDeviceSize stride,
                               VkQueryResultFlags flags)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   if (!queryCount)
      return;

   /* Query availability writes may be deferred to the post-graphics stream.
    * Finish the producing streams before creating a waiting consumer so it
    * cannot spin ahead of its own availability write. Do this before
    * hk_query_command_cs attaches dependencies to the consumer stream. */
   if (agx_native_graphics(&dev->dev) && (flags & VK_QUERY_RESULT_WAIT_BIT)) {
      hk_cmd_buffer_end_graphics(cmd);
      hk_cmd_buffer_end_compute(cmd);
   }

   /* The copy reads results and availability every earlier stream may still
    * be producing. Ordering the destination buffer against later use is the
    * application's (it is a transfer operation, Vulkan 1.3, 17.2). */
   if (!hk_query_command_cs(cmd))
      return;

   perf_debug(cmd, "Query pool copy");

   if (agx_native_graphics(&dev->dev)) {
      if ((pool->vk.query_type != VK_QUERY_TYPE_TIMESTAMP &&
           pool->vk.query_type != VK_QUERY_TYPE_OCCLUSION &&
           pool->vk.query_type != VK_QUERY_TYPE_PRIMITIVES_GENERATED_EXT &&
           pool->vk.query_type != VK_QUERY_TYPE_MESH_PRIMITIVES_GENERATED_EXT &&
           pool->vk.query_type != VK_QUERY_TYPE_PIPELINE_STATISTICS &&
           pool->vk.query_type != VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT &&
           !hk_is_acceleration_structure_query(pool->vk.query_type)) ||
          firstQuery > pool->vk.query_count ||
          queryCount > pool->vk.query_count - firstQuery) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }

      const unsigned reports = hk_reports_per_query(pool);
      const unsigned word_size = (flags & VK_QUERY_RESULT_64_BIT) ? 8 : 4;
      const uint64_t tuple_size = word_size *
         (reports + !!(flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT));
      if (!reports || dstOffset > dst_buffer->vk.size ||
          tuple_size > dst_buffer->vk.size - dstOffset ||
          (queryCount > 1 &&
           stride > (dst_buffer->vk.size - dstOffset - tuple_size) /
                       (queryCount - 1))) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }

      struct libagx_copy_query_g17p_args info = {
         .availability = hk_has_available(pool) ? pool->bo->va->addr : 0,
         .results = pool->oq_queries ? dev->occlusion_queries.bo->va->addr
                                    : pool->bo->va->addr + pool->query_start,
         .oq_index = pool->oq_queries ? pool->bo->va->addr + pool->query_start : 0,
         .first_query = firstQuery,
         .dst_addr = hk_buffer_address_rw(dst_buffer, dstOffset),
         .dst_stride = stride,
         .reports_per_query = reports,
         .flags = flags,
         .availability_shift = pool->avail_shift,
      };

      struct hk_cs *cs = cmd->current_cs.cs;
      hk_ensure_cs_has_space(cmd, cs, AGX3_CDM_BARRIER_SIZE);
      hk_cdm_cache_flush(dev, cs);
      libagx_copy_query_g17p_struct(cmd, agx_1d(queryCount), AGX_BARRIER_NONE,
                                   info);
      if (vk_command_buffer_has_error(&cmd->vk))
         return;

      cs = cmd->current_cs.cs;
      hk_drm_attachment_set_add(&cs->compute_attachments, info.dst_addr,
         (uint64_t)(queryCount - 1) * stride + tuple_size);
      cs->stats.query_copy_dispatches++;
      cs->stats.query_copy_queries += queryCount;
      cs->stats.query_copy_counters += (uint64_t)queryCount * reports;
      return;
   }

   struct libagx_copy_query_args info = {
      .availability = hk_has_available(pool) ? pool->bo->va->addr : 0,
      .results = pool->oq_queries ? dev->occlusion_queries.bo->va->addr
                                  : pool->bo->va->addr + pool->query_start,
      .oq_index = pool->oq_queries ? pool->bo->va->addr + pool->query_start : 0,

      .first_query = firstQuery,
      .dst_addr = hk_buffer_address_rw(dst_buffer, dstOffset),
      .dst_stride = stride,
      .reports_per_query = hk_reports_per_query(pool),
      .flags = flags,
   };

   libagx_copy_query_struct(cmd, agx_1d(queryCount), AGX_BARRIER_ALL, info);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdCopyQueryPoolResults(VkCommandBuffer commandBuffer, VkQueryPool queryPool,
                           uint32_t firstQuery, uint32_t queryCount,
                           VkBuffer dstBuffer, VkDeviceSize dstOffset,
                           VkDeviceSize stride, VkQueryResultFlags flags)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_query_pool, pool, queryPool);
   VK_FROM_HANDLE(hk_buffer, dst_buffer, dstBuffer);

   cmd->sync.kind_override = HK_SYNC_ALL;
   hk_cmd_copy_query_pool_results(cmd, pool, firstQuery, queryCount,
                                  dst_buffer, dstOffset, stride, flags);
   cmd->sync.kind_override = 0;
}
