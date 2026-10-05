/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 *
 * based in part on anv driver which is:
 * Copyright © 2015 Intel Corporation
 * SPDX-License-Identifier: MIT
 */
#include "hk_queue.h"
#include "hk_buffer.h"

#include "agx_bg_eot.h"
#include "agx_bo.h"
#include "agx_device.h"
#include "agx_pack.h"
#include "asahi/genxml/agx3_cmdstream.h"
#include "agx_vdm_hal200_bind_tail_core.h"
#include "decode.h"
#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_drm_render_core.h"
#include "hk_drm_prior_core.h"
#include "hk_image.h"
#include "hk_physical_device.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <xf86drm.h>
#include <errno.h>
#include "util/cache_ops.h"
#include "util/list.h"
#include "util/macros.h"
#include "util/os_time.h"
#include "util/u_atomic.h"
#include "util/u_debug.h"
#include "util/u_dynarray.h"
#include "vulkan/vulkan_core.h"

#include "hk_private.h"
#include "layout.h"
#include "vk_drm_syncobj.h"
#include "vk_sync.h"

DEBUG_GET_ONCE_BOOL_OPTION(hk_g17p_aux_fb_small_probe,
                          "HK_G17P_AUX_FB_SMALL_PROBE", false)
DEBUG_GET_ONCE_BOOL_OPTION(hk_g17p_mcache_aperture_probe,
                          "HK_G17P_MCACHE_APERTURE_PROBE", false)

DEBUG_GET_ONCE_BOOL_OPTION(hk_g17p_layer_zero_trace,
                          "HK_G17P_LAYER_ZERO_TRACE", false)

static bool
hk_agxdecode_trace_armed(void)
{
   const char *arm_file = getenv("HK_G17P_TRACE_ARM_FILE");

   return arm_file == NULL || arm_file[0] == '\0' ||
          access(arm_file, F_OK) == 0;
}

#define HK_G17P_FAULTTRACE_STREAM_MAGIC UINT64_C(0x314d414552545346)

struct hk_g17p_faulttrace_stream_header {
   uint64_t magic;
   uint64_t submitted_full;
   uint64_t submitted_base;
   uint32_t stream_size;
   uint32_t snapshot_size;
   uint32_t draws;
   uint32_t publications;
};

/* Signal scope includes all earlier commands on this Vulkan queue. A DRM
 * timeline point retains its predecessor fence chain; an individual submit's
 * aggregate fence covers only that submit. Transferring the queue point orders
 * completion without introducing a GPU execution dependency or a CPU wait.
 */
static VkResult
queue_signal_timeline(struct hk_device *dev, struct hk_queue *queue,
                      struct vk_queue_submit *submit)
{
   int fd = dev->dev.fd;

   /* Each output becomes signalled only when this queue point and its
    * predecessor chain complete.
    */
   for (unsigned i = 0; i < submit->signal_count; ++i) {
      struct vk_sync_signal *signal = &submit->signals[i];

      assert(vk_sync_type_is_drm_syncobj(signal->sync->type));
      const struct vk_drm_syncobj *syncobj =
         vk_sync_as_drm_syncobj(signal->sync);

      int ret = drmSyncobjTransfer(
         fd, syncobj->syncobj, signal->signal_value,
         queue->drm.syncobj, queue->drm.timeline_value, 0);
      if (ret)
         return vk_errorf(dev, VK_ERROR_UNKNOWN,
                          "DRM_IOCTL_SYNCOBJ_TRANSFER failed: %m");
   }

   return VK_SUCCESS;
}

/*
 * We need to specially handle submits with no control streams. The kernel
 * can't accept empty submits, but we can end up here in Vulkan for
 * synchronization purposes only. Rather than submit a no-op job (slow),
 * we simply tie the fences together.
 */
static VkResult
queue_submit_empty(struct hk_device *dev, struct hk_queue *queue,
                   struct vk_queue_submit *submit)
{
   int fd = dev->dev.fd;

   /* Transfer the waits into the queue timeline. */
   for (unsigned i = 0; i < submit->wait_count; ++i) {
      struct vk_sync_wait *wait = &submit->waits[i];

      assert(vk_sync_type_is_drm_syncobj(wait->sync->type));
      const struct vk_drm_syncobj *syncobj = vk_sync_as_drm_syncobj(wait->sync);

      int ret = drmSyncobjTransfer(
         fd, queue->drm.syncobj, ++queue->drm.timeline_value,
         syncobj->syncobj, wait->wait_value, 0);
      if (ret)
         return vk_errorf(dev, VK_ERROR_UNKNOWN,
                          "DRM_IOCTL_SYNCOBJ_TRANSFER failed: %m");
   }

   return queue_signal_timeline(dev, queue, submit);
}

static void
asahi_fill_cdm_command(struct hk_device *dev, struct hk_cs *cs,
                       struct drm_asahi_cmd_compute *cmd)
{
   size_t len = cs->stream_linked ? 65536 /* XXX */ : (cs->current - cs->start);

   *cmd = (struct drm_asahi_cmd_compute){
      .cdm_ctrl_stream_base = cs->addr,
      .cdm_ctrl_stream_end = cs->addr + len,

      .ts.start.handle = cs->timestamp.start.handle,
      .ts.start.offset = cs->timestamp.start.offset_B,
      .ts.end.handle = cs->timestamp.end.handle,
      .ts.end.offset = cs->timestamp.end.offset_B,
   };

   if (cs->uses_sampler_heap) {
      cmd->sampler_heap = dev->samplers.table.bo->va->addr;
      cmd->sampler_count = dev->samplers.table.alloc;
   }

   if (cs->scratch.cs.main || cs->scratch.cs.preamble) {
      cmd->helper.data = dev->scratch.cs.buf->va->addr;
      cmd->helper.cfg = cs->scratch.cs.preamble ? (1 << 16) : 0;
      cmd->helper.binary = agx_helper_program(&dev->bg_eot);
   }

   /* Dump the finished CDM stream, terminator included -- this runs after
    * hk_EndCommandBuffer has appended it. The known-good add3 stream is
    * exactly 48 bytes (40-byte direct launch + 4-byte barrier 0x60000160 +
    * 4-byte terminate 0x40000000); anything else here is a structural
    * difference from the submission the firmware is known to complete. The
    * kernel also assumes the last dword of [base, end) is the terminator
    * (translate_compute_command: terminator = end - 4). */
   if (hk_g17p_launch_trace_enabled()) {
      fprintf(stderr,
              "G17P cdm: base=%" PRIx64 " end=%" PRIx64 " len=%zu linked=%d "
              "sampler_heap=%" PRIx64 " sampler_count=%u helper(bin=%" PRIx64
              " data=%" PRIx64 " cfg=%x)\n",
              (uint64_t)cmd->cdm_ctrl_stream_base,
              (uint64_t)cmd->cdm_ctrl_stream_end, len, (int)cs->stream_linked,
              (uint64_t)cmd->sampler_heap, cmd->sampler_count,
              (uint64_t)cmd->helper.binary, (uint64_t)cmd->helper.data,
              cmd->helper.cfg);
      if (!cs->stream_linked) {
         hk_g17p_trace_hexdump("cdm-stream", cs->addr, cs->start, len);
      }
   }
}

/* Diagnostic prefix bisect for the fault-free G17P TA timeout. Each stage
 * replaces the next packet with a terminator, preserving a structurally valid
 * prefix so a fresh-boot run identifies the first state block that firmware
 * cannot execute:
 *
 *   1: barrier + initial PPP
 *   2: + VDM state
 *   3: + native VertexProgramVariant header-control word
 *   4..11: + each of the eight native HAL200 bind pairs
 *   12: + the complete native 24-byte direct draw record, whose last dword is
 *       already the stream terminator
 *
 * The full stream remains the default and every boundary is checked against
 * the exact HAL200 packet header before any byte is changed.
 */
static size_t
asahi_g17p_vdm_bisect(struct hk_device *dev, struct hk_cs *cs, size_t len)
{
   static int stage = -1;
   if (stage < 0)
      stage = debug_get_num_option("HK_G17P_VDM_BISECT", 0);

   if (stage == 0)
      return len;

   if (dev->dev.chip != AGX_CHIP_G17P || cs->stream_linked || stage < 1 ||
       stage > 12) {
      mesa_loge("HK_G17P_VDM_BISECT=%d requires an unlinked G17P stream",
                stage);
      return len;
   }

   const size_t boundary =
      stage == 1 ? 12 : stage == 2 ? 40
                 : stage == 3 ? 44
                 : stage <= 11 ? 44 + ((stage - 3) * 8)
                               : 132;

   const size_t required = stage == 12 ? boundary : boundary + 4;
   if (len < required) {
      mesa_loge("HK_G17P_VDM_BISECT=%d stream too short (%zu)", stage, len);
      return len;
   }

   uint32_t barrier, initial, prefix;
   memcpy(&barrier, (uint8_t *)cs->start, 4);
   memcpy(&initial, (uint8_t *)cs->start + 4, 4);
   memcpy(&prefix, (uint8_t *)cs->start + 12, 4);
   if (barrier != UINT32_C(0x21000960) ||
       (initial & UINT32_C(0xffffff00)) != UINT32_C(0x00000300) ||
       prefix != UINT32_C(0x4000002e)) {
      mesa_loge("HK_G17P_VDM_BISECT=%d opening mismatch: barrier=%#x "
                "initial=%#x prefix=%#x",
                stage, barrier, initial, prefix);
      return len;
   }

   if (stage >= 3) {
      uint32_t header_control;
      memcpy(&header_control, (uint8_t *)cs->start + 40, 4);
      if (header_control != UINT32_C(0x500)) {
         mesa_loge("HK_G17P_VDM_BISECT=%d bind header mismatch: %#x", stage,
                   header_control);
         return len;
      }
   }

   const unsigned pair_count = stage >= 4 ? MIN2((unsigned)(stage - 3), 8) : 0;
   for (unsigned i = 0; i < pair_count; ++i) {
      uint32_t address, control;
      const size_t offset = 44 + (8 * i);
      memcpy(&address, (uint8_t *)cs->start + offset, 4);
      memcpy(&control, (uint8_t *)cs->start + offset + 4, 4);
      if (address == 0 || (address & 3) != 0 ||
          !agx_vdm_hal200_bind_control_valid(i, control)) {
         mesa_loge("HK_G17P_VDM_BISECT=%d bind[%u] mismatch: addr=%#x "
                   "control=%#x",
                   stage, i, address, control);
         return len;
      }
   }

   if (stage == 12) {
      uint32_t draw_state, header, vertex_count, instance_count, tail;
      memcpy(&draw_state, (uint8_t *)cs->start + 108, 4);
      memcpy(&header, (uint8_t *)cs->start + 112, 4);
      memcpy(&vertex_count, (uint8_t *)cs->start + 116, 4);
      memcpy(&instance_count, (uint8_t *)cs->start + 120, 4);
      memcpy(&tail, (uint8_t *)cs->start + 128, 4);
      if (draw_state == 0 || (draw_state & 3) != 0 ||
          header != UINT32_C(0x61c40600) || vertex_count == 0 ||
          instance_count == 0 || tail != UINT32_C(0xc0000000)) {
         mesa_loge("HK_G17P_VDM_BISECT=12 draw mismatch: state=%#x "
                   "header=%#x count=%u instances=%u tail=%#x",
                   draw_state, header, vertex_count, instance_count, tail);
         return len;
      }

      fprintf(stderr, "G17P VDM bisect: stage=%d prefix=%zu total=%zu\n",
              stage, boundary, boundary);
      return boundary;
   }

   static const uint8_t terminate[4] = {0x00, 0x00, 0x00, 0xc0};
   memcpy((uint8_t *)cs->start + boundary, terminate, sizeof(terminate));
   if (hk_g17p_cached_cmd_memory(&hk_cmd_buffer_device(cs->cmd)->dev))
      util_flush_range((uint8_t *)cs->start + boundary, sizeof(terminate));
   fprintf(stderr, "G17P VDM bisect: stage=%d prefix=%zu total=%zu\n", stage,
           boundary, boundary + sizeof(terminate));
   return boundary + sizeof(terminate);
}

static void
hk_g17p_trace_render_command(const struct hk_device *dev,
                              const struct hk_cs *cs,
                              const struct drm_asahi_cmd_render *c)
{
   static int enabled = -1;
   if (enabled < 0)
      enabled = debug_get_bool_option("HK_G17P_RENDER_TRACE", false);

   if (!enabled || dev->dev.chip != AGX_CHIP_G17P)
      return;

   hk_g17p_trace_line("render-uapi", c->vdm_ctrl_stream_base, c,
                      sizeof(*c));
   fprintf(stderr,
           "G17P render-uapi-decoded: flags=%#x size=%ux%u layers=%u "
           "samples=%u sample-size=%u utile=%ux%u vdm=%#" PRIx64
           " scissor=%#" PRIx64 " dbias=%#" PRIx64
           " zls=%#" PRIx64 " zls-pixels=%#x ppp=%#x msaa=%#" PRIx64
           " bg=%#x/%#x eot=%#x/%#x partial-bg=%#x/%#x "
           "partial-eot=%#x/%#x sampler=%#" PRIx64 "/%u "
           "draw-calls=%u draws=%u flushes=%u merged=%u\n",
           c->flags, c->width_px, c->height_px, c->layers, c->samples,
           c->sample_size_B, c->utile_width_px, c->utile_height_px,
           (uint64_t)c->vdm_ctrl_stream_base, (uint64_t)c->isp_scissor_base,
           (uint64_t)c->isp_dbias_base, (uint64_t)c->zls_ctrl,
           c->isp_zls_pixels, c->ppp_ctrl, (uint64_t)c->ppp_multisamplectl,
           c->bg.usc, c->bg.rsrc_spec, c->eot.usc, c->eot.rsrc_spec,
           c->partial_bg.usc, c->partial_bg.rsrc_spec, c->partial_eot.usc,
           c->partial_eot.rsrc_spec, (uint64_t)c->sampler_heap,
           c->sampler_count, cs->stats.calls, cs->stats.cmds,
           cs->stats.flushes, cs->stats.merged);
   fflush(stderr);
}

/* Neither attachment storage nor empty-tile operations may observe the
 * layers eliminated below.  Draws and their XY coverage remain untouched.
 */
static bool
hk_g17p_attachmentless_render(const struct hk_device *dev,
                                    const struct hk_cs *cs)
{
   if (dev->dev.chip != AGX_CHIP_G17P ||
       cs->tib.sample_size_B != 0 || cs->cr.process_empty_tiles ||
       cs->cr.depth.buffer || cs->cr.depth.meta ||
       cs->cr.stencil.buffer || cs->cr.stencil.meta)
      return false;

   /* The tilebuffer builder normalizes an attachmentless sample count to
    * one.  Check attachment formats explicitly; nr_samples is not a tag.
    * This also excludes colour attachments spilled outside the tilebuffer.
    */
   for (unsigned i = 0; i < ARRAY_SIZE(cs->tib.logical_format); ++i) {
      if (cs->tib.logical_format[i] != PIPE_FORMAT_NONE)
         return false;
   }

   return true;
}

static void
asahi_fill_vdm_command(struct hk_device *dev, struct hk_cs *cs,
                       struct drm_asahi_cmd_render *c,
                       struct util_dynarray *faulttrace_extra)
{
   size_t len = cs->stream_linked ? hk_cs_root_size(dev->dev.chip, false)
                                  : (cs->current - cs->start);
   len = asahi_g17p_vdm_bisect(dev, cs, len);
   const uint8_t *submitted_start = cs->start;
   uint64_t submitted_full = cs->addr;

   memset(c, 0, sizeof(*c));

   c->vdm_ctrl_stream_base =
      hk_g17p_vdm_low_window_enabled(&dev->dev)
         ? hk_g17p_vdm_alias(cs->cmd, cs->addr)
         : cs->addr;

   /* Metal's compact direct encoder starts at the 0x4000002e
    * VertexProgramVariant header.  Honeykrisp normally precedes it with a
    * 12-byte barrier/initial-PPP sequence.  Keep this opt-in and validate the
    * exact boundary so one hardware run can isolate that address/order
    * difference without rewriting the command BO.
    */
   static int native_base = -1;
   if (native_base < 0)
      native_base =
         debug_get_bool_option("HK_G17P_VDM_NATIVE_BASE", false) ? 1 : 0;
   if (native_base && dev->dev.chip == AGX_CHIP_G17P && !cs->stream_linked) {
      uint32_t header = 0;
      if (len >= 16)
         memcpy(&header, (const uint8_t *)cs->start + 12, sizeof(header));

      if (header == UINT32_C(0x4000002e)) {
         c->vdm_ctrl_stream_base += 12;
         submitted_full += 12;
         submitted_start += 12;
         len -= 12;
         fprintf(stderr,
                 "G17P VDM native base: skipped 12-byte barrier/PPP prelude\n");
      } else {
         mesa_loge("HK_G17P_VDM_NATIVE_BASE opening mismatch: len=%zu "
                   "header=%#x",
                   len, header);
      }
   }

   /* Capture the exact live VDM stream after its terminator was appended.
    * The nested PPP state records in this dump can then be resolved against
    * the payload dumps emitted while the command buffer was built.
    */
   if (hk_g17p_launch_trace_enabled()) {
      fprintf(stderr,
              "G17P vdm: base=%" PRIx64 " submit=%" PRIx64
              " len=%zu linked=%d\n",
              submitted_full, (uint64_t)c->vdm_ctrl_stream_base, len,
              (int)cs->stream_linked);
      if (!cs->stream_linked)
         hk_g17p_trace_hexdump("vdm-stream", submitted_full, submitted_start,
                               len);
   }

   agx_pack(&c->ppp_ctrl, CR_PPP_CONTROL, cfg) {
      cfg.enable_w_clamp = true;
      cfg.fixed_point_format = 1;
   }

   const bool empty_attachmentless =
      cs->stats.cmds == 0 && hk_g17p_attachmentless_render(dev, cs);
   const bool layer_zero_attachmentless =
      cs->stats.cmds > 0 && !cs->g17p_layer_zero_unknown &&
      cs->g17p_layer_zero_draws == cs->stats.cmds &&
      hk_g17p_attachmentless_render(dev, cs);
   c->width_px = empty_attachmentless ? 1 : cs->cr.width;
   c->height_px = empty_attachmentless ? 1 : cs->cr.height;

   c->isp_bgobjdepth = cs->cr.isp_bgobjdepth;
   c->isp_bgobjvals = cs->cr.isp_bgobjvals;

   static_assert(sizeof(c->zls_ctrl) == sizeof(cs->cr.zls_control));
   memcpy(&c->zls_ctrl, &cs->cr.zls_control, sizeof(cs->cr.zls_control));

   if (dev->dev.chip == AGX_CHIP_G17P) {
      c->isp_zls_pixels = hk_drm_g17p_zls_pixels(
         cs->cr.zls_width, cs->cr.zls_height, cs->cr.depth.buffer,
         cs->cr.depth.meta, cs->cr.stencil.buffer, cs->cr.stencil.meta);
      c->isp_bgobjdepth = hk_drm_g17p_depth_clear(
         c->isp_bgobjdepth, cs->cr.depth.buffer, cs->cr.depth.meta,
         cs->cr.stencil.buffer, cs->cr.stencil.meta);
   } else {
      agx_pack(&c->isp_zls_pixels, CR_ISP_ZLS_PIXELS, cfg) {
         cfg.x = cs->cr.zls_width;
         cfg.y = cs->cr.zls_height;
      }
   }

   c->depth.base = cs->cr.depth.buffer;
   c->depth.stride = cs->cr.depth.stride;
   c->depth.comp_base = cs->cr.depth.meta;
   c->depth.comp_stride = cs->cr.depth.meta_stride;
   c->stencil.base = cs->cr.stencil.buffer;
   c->stencil.stride = cs->cr.stencil.stride;
   c->stencil.comp_base = cs->cr.stencil.meta;
   c->stencil.comp_stride = cs->cr.stencil.meta_stride;

   /* Private matched-benchmark diagnostic, requiring a diagnostic kernel.
    * The caller must know the captured small-primitive predicate applies.
    * Do not enable this for arbitrary applications or desktop workloads.
    */
   if (dev->dev.chip == AGX_CHIP_G17P &&
       debug_get_option_hk_g17p_aux_fb_small_probe())
      c->flags |= (1u << 31);
   if (dev->dev.chip == AGX_CHIP_G17P &&
       debug_get_option_hk_g17p_mcache_aperture_probe())
      c->flags |= (1u << 30);


   if (cs->cr.dbias_is_int == U_TRISTATE_YES) {
      c->flags |= DRM_ASAHI_RENDER_DBIAS_IS_INT;
   }

   if ((dev->dev.debug & AGX_DBG_NOCLUSTER) &&
       dev->dev.chip != AGX_CHIP_G17P) {
      c->flags |= DRM_ASAHI_RENDER_NO_VERTEX_CLUSTERING;
   }

   agx_tilebuffer_set_drm_cmd(c, &cs->tib);

   /* Can be 0 for attachmentless rendering with no draws */
   c->samples = MAX2(cs->tib.nr_samples, 1);
   c->layers = (empty_attachmentless || layer_zero_attachmentless)
                  ? 1 : cs->cr.layers;

   if (dev->dev.chip == AGX_CHIP_G17P && cs->cr.layers > 1 &&
       debug_get_option_hk_g17p_layer_zero_trace()) {
      static int count;
      if (p_atomic_inc_return(&count) <= 64) {
         const int saved_errno = errno;
         fprintf(stderr,
                 "HK_G17P_LAYER_ZERO pid=%ld stream=%#" PRIx64
                 " dims=%ux%u layers=%u->%u draws=%u proven=%" PRIu64
                 " attachmentless=%u normalized=%u\n",
                 (long)getpid(), (uint64_t)c->vdm_ctrl_stream_base,
                 c->width_px, c->height_px, cs->cr.layers, c->layers,
                 cs->stats.cmds, cs->g17p_layer_zero_draws,
                 (unsigned)hk_g17p_attachmentless_render(dev, cs),
                 (unsigned)layer_zero_attachmentless);
         errno = saved_errno;
      }
   }

   /* Drawing max size will OOM and fail submission. But vkd3d-proton does this
    * for emulating no-attachment rendering. Clamp to something reasonable and
    * hope this is good enough in practice. This only affects a case that would
    * otherwise be guaranteed broken.
    *
    * XXX: Hack for vkd3d-proton.
    */
   if (c->layers == 2048 && c->width_px == 16384 && c->height_px == 16384) {
      mesa_log(MESA_LOG_WARN, MESA_LOG_TAG, "Clamping massive framebuffer");
      c->layers = 32;
   }

   c->ppp_multisamplectl = cs->ppp_multisamplectl;
   c->sample_size_B = cs->tib.sample_size_B;

   float tan_60 = 1.732051f;
   c->isp_merge_upper_x = fui(tan_60 / c->width_px);
   c->isp_merge_upper_y = fui(tan_60 / c->height_px);

   c->bg.usc = cs->cr.bg.main.usc | 4;
   c->eot.usc = cs->cr.eot.main.usc | 4;
   c->partial_bg.usc = cs->cr.bg.partial.usc | 4;
   c->partial_eot.usc = cs->cr.eot.partial.usc | 4;

   hk_drm_render_resource_specs_pack(
      &(struct hk_drm_render_resource_specs){
         .bg = cs->cr.bg.main.rsrc_spec,
         .eot = cs->cr.eot.main.rsrc_spec,
         .partial_bg = cs->cr.bg.partial.rsrc_spec,
         .partial_eot = cs->cr.eot.partial.rsrc_spec,
      },
      c);

   c->isp_scissor_base = cs->uploaded_scissor;
   c->isp_dbias_base = cs->uploaded_zbias;

   if (cs->uses_sampler_heap) {
      c->sampler_heap = dev->samplers.table.bo->va->addr;
      c->sampler_count = dev->samplers.table.alloc;
   }

   c->isp_oclqry_base = dev->occlusion_queries.bo->va->addr;

   if (cs->cr.process_empty_tiles)
      c->flags |= DRM_ASAHI_RENDER_PROCESS_EMPTY_TILES;

   if (cs->scratch.vs.main || cs->scratch.vs.preamble) {
      c->flags |= DRM_ASAHI_RENDER_VERTEX_SCRATCH;
      c->vertex_helper.data = dev->scratch.vs.buf->va->addr;
      c->vertex_helper.cfg = cs->scratch.vs.preamble ? (1 << 16) : 0;
      c->vertex_helper.binary = agx_helper_program(&dev->bg_eot);
   }

   if (cs->scratch.fs.main || cs->scratch.fs.preamble) {
      c->fragment_helper.data = dev->scratch.fs.buf->va->addr;
      c->fragment_helper.cfg = cs->scratch.fs.preamble ? (1 << 16) : 0;
      c->fragment_helper.binary = agx_helper_program(&dev->bg_eot);
   }

   if (cs->timestamp.start.handle) {
      c->ts_vtx.start.handle = cs->timestamp.start.handle;
      c->ts_vtx.start.offset = cs->timestamp.start.offset_B;
   }

   if (cs->timestamp.end.handle) {
      c->ts_frag.end.handle = cs->timestamp.end.handle;
      c->ts_frag.end.offset = cs->timestamp.end.offset_B;
   }

   if ((dev->dev.debug & AGX_DBG_FAULTTRACE) && faulttrace_extra) {
      const struct hk_g17p_faulttrace_stream_header header = {
         .magic = HK_G17P_FAULTTRACE_STREAM_MAGIC,
         .submitted_full = submitted_full,
         .submitted_base = c->vdm_ctrl_stream_base,
         .stream_size = cs->stream_linked ? 0 : len,
         .snapshot_size = cs->g17p_faulttrace_snapshots.size,
         .draws = cs->stats.cmds,
         .publications = cs->g17p_state_publications,
      };
      const size_t total = sizeof(header) + header.stream_size +
                           header.snapshot_size;
      uint8_t *record =
         util_dynarray_grow_bytes(faulttrace_extra, 1, total);
      if (record) {
         memcpy(record, &header, sizeof(header));
         record += sizeof(header);
         if (header.stream_size) {
            memcpy(record, submitted_start, header.stream_size);
            record += header.stream_size;
         }
         if (header.snapshot_size)
            memcpy(record, cs->g17p_faulttrace_snapshots.data,
                   header.snapshot_size);
      }
   }

   hk_g17p_trace_render_command(dev, cs, c);
}

static void
asahi_fill_sync(struct drm_asahi_sync *sync, struct vk_sync *vk_sync,
                uint64_t value)
{
   if (unlikely(!vk_sync_type_is_drm_syncobj(vk_sync->type))) {
      UNREACHABLE("Unsupported sync type");
      return;
   }

   const struct vk_drm_syncobj *syncobj = vk_sync_as_drm_syncobj(vk_sync);
   *sync = (struct drm_asahi_sync){.handle = syncobj->syncobj};

   if (vk_sync->flags & VK_SYNC_IS_TIMELINE) {
      sync->sync_type = DRM_ASAHI_SYNC_TIMELINE_SYNCOBJ;
      sync->timeline_value = value;
   } else {
      sync->sync_type = DRM_ASAHI_SYNC_SYNCOBJ;
   }
}

union drm_asahi_cmd {
   struct drm_asahi_cmd_compute compute;
   struct drm_asahi_cmd_render render;
};

/*
 * These are diagnostic knobs, read once per process.
 *
 * Every one of them sat on the submit path and was re-read on EVERY
 * vkQueueSubmit: two inside hk_g17p_write_range_candidates() before its chip
 * check, a third beside its caller, and two more beside the ioctl. getenv() is
 * a linear scan of the environment block, so that is five scans per submission
 * to answer five questions whose answers cannot change after startup.
 */
static bool
hk_g17p_debug_flag(const char *name)
{
   const char *value = getenv(name);
   return value != NULL;
}

static bool
hk_g17p_dump_write_ranges(void)
{
   static int cached = -1;
   if (unlikely(cached < 0))
      cached = hk_g17p_debug_flag("HK_G17P_DUMP_WRITE_RANGES");
   return cached;
}

static bool
hk_g17p_native_write_spans(void)
{
   static int cached = -1;
   if (unlikely(cached < 0))
      cached = hk_g17p_debug_flag("HK_G17P_NATIVE_WRITE_SPANS");
   return cached;
}

static bool
hk_g17p_abort_before_ioctl(void)
{
   static int cached = -1;
   if (unlikely(cached < 0))
      cached = hk_g17p_debug_flag("HK_G17P_ABORT_BEFORE_IOCTL");
   return cached;
}

static bool
hk_g17p_submit_dmb(void)
{
   static int cached = -1;
   if (unlikely(cached < 0))
      cached = hk_g17p_debug_flag("HK_G17P_SUBMIT_DMB");
   return cached;
}

static bool
hk_g17p_write_range_candidates(struct hk_device *dev,
                               struct drm_asahi_attachment spans[2])
{
   /* Chip check first: on anything but G17P this returns without touching the
    * environment at all. */
   if (dev->dev.chip != AGX_CHIP_G17P)
      return false;
   const bool dump = hk_g17p_dump_write_ranges();
   const bool synthesize = hk_g17p_native_write_spans();
   if (!dump && !synthesize)
      return false;

   uint64_t main_start = UINT64_MAX, main_end = 0;
   uint64_t usc_start = UINT64_MAX, usc_end = 0;

   simple_mtx_lock(&dev->dev.bo_map_lock);
   for (uint32_t handle = 0; handle < dev->dev.max_handle; ++handle) {
      struct agx_bo *bo = agx_lookup_bo(&dev->dev, handle);
      if (!bo->size || !bo->va || bo->size > UINT64_MAX - bo->va->addr)
         continue;

      const uint64_t start = bo->va->addr;
      const uint64_t end = start + bo->size;
      const bool usc = bo->va->flags & AGX_VA_USC;
      if (dump) {
         fprintf(stderr,
                 "G17P BO range: handle=%u label=%s va=%#" PRIx64
                 " size=%#" PRIx64 " flags=%#x heap=%s writable=%u\n",
                 handle, bo->label, start, (uint64_t)bo->size, bo->flags,
                 usc ? "usc" : "main", !(bo->flags & AGX_BO_READONLY));
      }

      if (bo->flags & AGX_BO_READONLY)
         continue;

      uint64_t *span_start = usc ? &usc_start : &main_start;
      uint64_t *span_end = usc ? &usc_end : &main_end;
      *span_start = MIN2(*span_start, start);
      *span_end = MAX2(*span_end, end);
   }
   simple_mtx_unlock(&dev->dev.bo_map_lock);

   if (dump) {
      fprintf(stderr,
              "G17P BO writable spans: main=%#" PRIx64 "..%#" PRIx64
              " usc=%#" PRIx64 "..%#" PRIx64 "\n",
              main_start == UINT64_MAX ? 0 : main_start, main_end,
              usc_start == UINT64_MAX ? 0 : usc_start, usc_end);
   }

   if (main_start == UINT64_MAX || usc_start == UINT64_MAX ||
       main_end <= main_start || usc_end <= usc_start)
      return false;

   spans[0] = (struct drm_asahi_attachment){
      .pointer = main_start,
      .size = main_end - main_start,
   };
   spans[1] = (struct drm_asahi_attachment){
      .pointer = usc_start,
      .size = usc_end - usc_start,
   };
   return true;
}

static void
asahi_append_compute_attachments(struct hk_device *dev,
                                 struct util_dynarray *payload,
                                 const struct hk_cs *cs)
{
   const struct hk_drm_attachment_set *set = &cs->compute_attachments;
   const uint32_t count = MIN2(set->count, dev->dev.params.max_attachments);
   const struct drm_asahi_cmd_header header = {
      .cmd_type = DRM_ASAHI_SET_COMPUTE_ATTACHMENTS,
      .size = count * sizeof(set->list[0]),
      .vdm_barrier = DRM_ASAHI_BARRIER_NONE,
      .cdm_barrier = DRM_ASAHI_BARRIER_NONE,
   };
   /* Clear the preceding command's list even for an attachmentless dispatch. */
   util_dynarray_append(payload, header);
   if (count)
      util_dynarray_append_array(payload, struct drm_asahi_attachment,
                                 set->list, count);
}

static void
asahi_append_fragment_attachments(struct hk_device *dev,
                                  struct util_dynarray *payload,
                                  const struct hk_cs *cs)
{
   struct drm_asahi_attachment writable_spans[2];
   const bool have_writable_spans =
      hk_g17p_write_range_candidates(dev, writable_spans);
   struct hk_drm_attachment_set synthetic;
   const struct hk_drm_attachment_set *set = &cs->fragment_attachments;
   if (hk_g17p_native_write_spans() && have_writable_spans) {
      hk_drm_attachment_set_init(&synthetic);
      synthetic.list[0] = writable_spans[0];
      synthetic.list[1] = writable_spans[1];
      synthetic.count = 2;
      set = &synthetic;
      fprintf(stderr,
              "G17P native write spans: main=%#" PRIx64 "+%#" PRIx64
              " usc=%#" PRIx64 "+%#" PRIx64 "\n",
              (uint64_t)writable_spans[0].pointer,
              (uint64_t)writable_spans[0].size,
              (uint64_t)writable_spans[1].pointer,
              (uint64_t)writable_spans[1].size);
   }

   const uint32_t count = MIN2(set->count, dev->dev.params.max_attachments);
   const struct drm_asahi_cmd_header header = {
      .cmd_type = DRM_ASAHI_SET_FRAGMENT_ATTACHMENTS,
      .size = count * sizeof(set->list[0]),
      .vdm_barrier = DRM_ASAHI_BARRIER_NONE,
      .cdm_barrier = DRM_ASAHI_BARRIER_NONE,
   };

   /* Emit even an empty set. Attachment commands affect subsequent hardware
    * commands in the same submit, so an attachmentless render must clear the
    * set retained by the preceding render. */
   util_dynarray_append(payload, header);
   if (count)
      util_dynarray_append_array(payload, struct drm_asahi_attachment,
                                 set->list, count);

   if (!set->complete || count != set->count) {
      perf_debug_dev(&dev->dev,
                     "Fragment attachment hint incomplete: retained %u, emitted %u (kernel limit %u)",
                     set->count, count, dev->dev.params.max_attachments);
   }
}

/* Submit all ordered control streams together up to the kernel's advertised
 * limit.  Splitting every stream into a separate ioctl serializes real games
 * at the queue boundary; keep single-stream submission only as an explicit
 * diagnostic control.
 */
static inline unsigned
max_commands_per_submit(struct hk_device *dev)
{
   return hk_drm_submit_command_limit(
      !HK_PERF(dev, NOBATCH), dev->dev.params.max_commands_per_submission);
}

/* One aggregate line per Vulkan queue submit.  Unlike the per-stream launch
 * and draw diagnostics, this is cheap enough to leave enabled for a complete
 * STK cold start and exposes whether texture work reached CDM without
 * perturbing production batching. */
static bool
hk_g17p_submit_trace_enabled(void)
{
   static int enabled = -1;
   if (enabled < 0)
      enabled = debug_get_bool_option("HK_G17P_SUBMIT_TRACE", false);
   return enabled != 0;
}

static VkResult
queue_submit_single(struct hk_device *dev, struct drm_asahi_submit *submit,
                    unsigned ring_idx)
{
   struct agx_submit_virt virt = {.ring_idx = ring_idx};

   /* Complete the cache cleans of CPU-cached descriptor writes
    * (hk_descriptor_set_publish()) before the GPU can read them. */
   if (dev->dev.chip == AGX_CHIP_G17P && util_has_cache_ops())
      util_post_flush_fence();

   if (dev->dev.chip == AGX_CHIP_G17P && hk_g17p_abort_before_ioctl()) {
      fprintf(stderr,
              "G17P diagnostic: aborting before DRM_IOCTL_ASAHI_SUBMIT\n");
      return vk_device_set_lost(&dev->vk,
                                "G17P diagnostic pre-submit abort");
   }

   if (dev->dev.is_virtio) {
      u_rwlock_rdlock(&dev->external_bos.lock);
      virt.extres_count = util_dynarray_num_elements(
         &dev->external_bos.list, struct asahi_ccmd_submit_res);
      virt.extres = util_dynarray_begin(&dev->external_bos.list);
   }

   /* G17P command, VIT, and HAL200 bind-state uploads use write-combined GEM
    * mappings.  Diagnose whether the GPU can observe a doorbell before those
    * CPU stores have drained, without changing the default submission path. */
   if (dev->dev.chip == AGX_CHIP_G17P && hk_g17p_submit_dmb())
      __sync_synchronize();

   int ret = dev->dev.ops.submit(&dev->dev, submit, &virt);

   if (dev->dev.is_virtio)
      u_rwlock_rdunlock(&dev->external_bos.lock);

   if (ret)
      return vk_device_set_lost(&dev->vk,
                                "DRM_IOCTL_ASAHI_SUBMIT failed: %m");

   return VK_SUCCESS;
}

/* Decode the exact ioctl slice that failed while all command-buffer BOs are
 * still alive.  AGX_DBG_FAULTTRACE already makes libagx retain the allocation
 * map, but until now nothing consumed that map: the ordinary trace runs before
 * submission and the failed payload was freed on return.  Keep this strictly
 * diagnostic and failure-only so production submission timing is unchanged. */
static void
hk_g17p_faulttrace_capture(struct hk_device *dev, struct hk_queue *queue,
                           const void *cmdbuf, size_t cmdbuf_size,
                           const void *extra, size_t extra_size)
{
   if (!(dev->dev.debug & AGX_DBG_FAULTTRACE) || cmdbuf_size == 0)
      return;

   struct hk_g17p_faulttrace_payload *slot =
      &queue->g17p_faulttrace[queue->g17p_faulttrace_head];
   void *data = malloc(cmdbuf_size);
   if (!data)
      return;

   void *extra_data = extra_size ? malloc(extra_size) : NULL;
   if (extra_size && !extra_data) {
      free(data);
      return;
   }

   memcpy(data, cmdbuf, cmdbuf_size);
   if (extra_size)
      memcpy(extra_data, extra, extra_size);
   free(slot->data);
   free(slot->extra);
   slot->data = data;
   slot->size = cmdbuf_size;
   slot->extra = extra_data;
   slot->extra_size = extra_size;
   slot->sequence = ++queue->g17p_faulttrace_sequence;
   queue->g17p_faulttrace_head =
      (queue->g17p_faulttrace_head + 1) % HK_G17P_FAULTTRACE_HISTORY;
   queue->g17p_faulttrace_count =
      MIN2(queue->g17p_faulttrace_count + 1, HK_G17P_FAULTTRACE_HISTORY);
}

enum hk_g17p_faulttrace_compare_status {
   HK_G17P_COMPARE_MATCH,
   HK_G17P_COMPARE_MISMATCH,
   HK_G17P_COMPARE_UNAVAILABLE,
   HK_G17P_COMPARE_NOMEM,
};

static enum hk_g17p_faulttrace_compare_status
hk_g17p_faulttrace_compare_bytes(struct agxdecode_ctx *ctx, uint64_t va,
                                 const void *expected, size_t size,
                                 size_t *first_mismatch)
{
   if (!size)
      return HK_G17P_COMPARE_MATCH;

   uint8_t *current = malloc(size);
   if (!current)
      return HK_G17P_COMPARE_NOMEM;
   if (!agxdecode_copy_gpu_mem(ctx, va, size, current)) {
      free(current);
      return HK_G17P_COMPARE_UNAVAILABLE;
   }

   const uint8_t *recorded = expected;
   for (size_t i = 0; i < size; ++i) {
      if (current[i] != recorded[i]) {
         *first_mismatch = i;
         free(current);
         return HK_G17P_COMPARE_MISMATCH;
      }
   }
   free(current);
   return HK_G17P_COMPARE_MATCH;
}

static const char *
hk_g17p_faulttrace_compare_name(enum hk_g17p_faulttrace_compare_status status)
{
   switch (status) {
   case HK_G17P_COMPARE_MATCH: return "match";
   case HK_G17P_COMPARE_MISMATCH: return "mismatch";
   case HK_G17P_COMPARE_UNAVAILABLE: return "unavailable";
   case HK_G17P_COMPARE_NOMEM: return "nomem";
   }
   UNREACHABLE("invalid faulttrace comparison status");
}

static void
hk_g17p_faulttrace_log_address(struct agxdecode_ctx *ctx, uint64_t sequence,
                               unsigned stream, unsigned snapshot,
                               const char *name, unsigned index,
                               uint64_t address)
{
   if (!address)
      return;

   uint64_t allocation_base = 0;
   size_t allocation_size = 0;
   const bool mapped = agxdecode_gpu_mem_bounds(
      ctx, address, &allocation_base, &allocation_size);
   fprintf(stderr,
           "G17P faulttrace root address: sequence=%" PRIu64
           " stream=%u snapshot=%u field=%s index=%u address=%" PRIx64
           " mapped=%u base=%" PRIx64 " allocation-size=%zu\n",
           sequence, stream, snapshot, name, index, address, mapped,
           allocation_base, allocation_size);
}

static void
hk_g17p_faulttrace_compare_live(struct hk_device *dev,
                                const struct hk_g17p_faulttrace_payload *slot)
{
   const uint8_t *at = slot->extra;
   const uint8_t *end = at + slot->extra_size;
   unsigned stream_index = 0;

   while ((size_t)(end - at) >=
          sizeof(struct hk_g17p_faulttrace_stream_header)) {
      struct hk_g17p_faulttrace_stream_header stream;
      memcpy(&stream, at, sizeof(stream));
      at += sizeof(stream);
      if (stream.magic != HK_G17P_FAULTTRACE_STREAM_MAGIC ||
          stream.stream_size > (size_t)(end - at))
         break;
      at += stream.stream_size;
      if (stream.snapshot_size > (size_t)(end - at))
         break;

      const uint8_t *snapshot_end = at + stream.snapshot_size;
      unsigned snapshot_index = 0;
      while ((size_t)(snapshot_end - at) >=
             sizeof(struct hk_g17p_faulttrace_snapshot_header)) {
         struct hk_g17p_faulttrace_snapshot_header snapshot;
         memcpy(&snapshot, at, sizeof(snapshot));
         at += sizeof(snapshot);
         const size_t sizes[] = {
            snapshot.ioto_size, snapshot.linked_size,
            snapshot.main_size, snapshot.root_size,
         };
         size_t total = 0;
         bool overflow = false;
         for (unsigned i = 0; i < ARRAY_SIZE(sizes); ++i) {
            if (sizes[i] > SIZE_MAX - total) {
               overflow = true;
               break;
            }
            total += sizes[i];
         }
         for (unsigned set = 0; set < HK_MAX_SETS && !overflow; ++set) {
            if (snapshot.descriptor_capture_size[set] > SIZE_MAX - total) {
               overflow = true;
               break;
            }
            total += snapshot.descriptor_capture_size[set];
         }
         if (snapshot.magic != HK_G17P_FAULTTRACE_SNAPSHOT_MAGIC || overflow ||
             total > (size_t)(snapshot_end - at))
            break;

         const uint8_t *ioto = at;
         const uint8_t *linked = ioto + snapshot.ioto_size;
         const uint8_t *main = linked + snapshot.linked_size;
         const uint8_t *root = main + snapshot.main_size;
         const uint8_t *descriptors = root + snapshot.root_size;
         size_t ioto_at = 0, linked_at = 0, root_at = 0;
         const enum hk_g17p_faulttrace_compare_status ioto_status =
            hk_g17p_faulttrace_compare_bytes(
               dev->dev.agxdecode, snapshot.program_va, ioto,
               snapshot.ioto_size, &ioto_at);
         const enum hk_g17p_faulttrace_compare_status linked_status =
            hk_g17p_faulttrace_compare_bytes(
               dev->dev.agxdecode, snapshot.shader_va, linked,
               snapshot.linked_size, &linked_at);
         const enum hk_g17p_faulttrace_compare_status root_status =
            hk_g17p_faulttrace_compare_bytes(
               dev->dev.agxdecode, snapshot.root_va, root,
               snapshot.root_size, &root_at);
         fprintf(stderr,
                 "G17P faulttrace live: sequence=%" PRIu64
                 " stream=%u snapshot=%u program=%" PRIx64
                 " shader=%" PRIx64 " root=%" PRIx64
                 " ioto=%s@%#zx linked=%s@%#zx root=%s@%#zx\n",
                 slot->sequence, stream_index, snapshot_index,
                 snapshot.program_va, snapshot.shader_va, snapshot.root_va,
                 hk_g17p_faulttrace_compare_name(ioto_status), ioto_at,
                 hk_g17p_faulttrace_compare_name(linked_status), linked_at,
                 hk_g17p_faulttrace_compare_name(root_status), root_at);

         if (root_status == HK_G17P_COMPARE_MATCH &&
             snapshot.root_size == sizeof(struct hk_root_descriptor_table)) {
            struct hk_root_descriptor_table launch_root;
            memcpy(&launch_root, root, sizeof(launch_root));

#define LOG_ROOT_ADDRESS(name, index, address)                                 \
   hk_g17p_faulttrace_log_address(dev->dev.agxdecode, slot->sequence,          \
                                  stream_index, snapshot_index, name, index,   \
                                  address)
            LOG_ROOT_ADDRESS("root", 0, launch_root.root_desc_addr);
            for (unsigned i = 0; i < HK_MAX_SETS; ++i)
               LOG_ROOT_ADDRESS("set", i, launch_root.sets[i]);
            for (unsigned i = 0; i < AGX_MAX_VBUFS; ++i)
               LOG_ROOT_ADDRESS("attrib", i,
                                launch_root.draw.attrib_base[i]);
            LOG_ROOT_ADDRESS("vertex-params", 0,
                             launch_root.draw.vertex_params);
            LOG_ROOT_ADDRESS("tess-params", 0,
                             launch_root.draw.tess_params);
            LOG_ROOT_ADDRESS("geometry-params", 0,
                             launch_root.draw.geometry_params);
            LOG_ROOT_ADDRESS("pipeline-stats", 0,
                             launch_root.draw.pipeline_stats);
            LOG_ROOT_ADDRESS("vs-constant", 0,
                             launch_root.draw.vs_constant_data_addr);
            LOG_ROOT_ADDRESS("fs-constant", 0,
                             launch_root.draw.fs_constant_data_addr);
            for (unsigned i = 0; i < HK_MAX_DYNAMIC_BUFFERS; ++i)
               LOG_ROOT_ADDRESS("dynamic-buffer", i,
                                launch_root.dynamic_buffers[i].base_addr);
            LOG_ROOT_ADDRESS("printf", 0,
                             launch_root.printf_buffer_address);
#undef LOG_ROOT_ADDRESS
         }

         for (unsigned set = 0; set < HK_MAX_SETS; ++set) {
            const size_t captured = snapshot.descriptor_capture_size[set];
            if (!snapshot.descriptor_va[set] &&
                !snapshot.descriptor_size[set] && !captured)
               continue;

            size_t descriptor_at = 0;
            const enum hk_g17p_faulttrace_compare_status descriptor_status =
               captured
                  ? hk_g17p_faulttrace_compare_bytes(
                       dev->dev.agxdecode, snapshot.descriptor_va[set],
                       descriptors, captured, &descriptor_at)
                  : HK_G17P_COMPARE_UNAVAILABLE;
            fprintf(stderr,
                    "G17P faulttrace descriptor: sequence=%" PRIu64
                    " stream=%u snapshot=%u set=%u va=%" PRIx64
                    " size=%u captured=%zu live=%s@%#zx\n",
                    slot->sequence, stream_index, snapshot_index, set,
                    snapshot.descriptor_va[set],
                    snapshot.descriptor_size[set], captured,
                    hk_g17p_faulttrace_compare_name(descriptor_status),
                    descriptor_at);

            for (size_t byte = 0; byte + sizeof(uint64_t) <= captured;
                 byte += sizeof(uint64_t)) {
               uint64_t word, allocation_base = 0, allocation_base_shift4 = 0;
               size_t allocation_size = 0, allocation_size_shift4 = 0;
               memcpy(&word, descriptors + byte, sizeof(word));
               if (!word)
                  continue;

               const bool mapped = agxdecode_gpu_mem_bounds(
                  dev->dev.agxdecode, word, &allocation_base,
                  &allocation_size);
               const bool mapped_shift4 = word <= (UINT64_MAX >> 4) &&
                  agxdecode_gpu_mem_bounds(dev->dev.agxdecode, word << 4,
                                           &allocation_base_shift4,
                                           &allocation_size_shift4);
               fprintf(stderr,
                       "G17P faulttrace descriptor word: sequence=%" PRIu64
                       " stream=%u snapshot=%u set=%u offset=%#zx"
                       " value=%" PRIx64 " mapped=%u base=%" PRIx64
                       " allocation-size=%zu shifted4=%u shifted4-base=%" PRIx64
                       " shifted4-allocation-size=%zu\n",
                       slot->sequence, stream_index, snapshot_index, set,
                       byte, word, mapped, allocation_base,
                       allocation_size, mapped_shift4,
                       allocation_base_shift4, allocation_size_shift4);
            }
            descriptors += captured;
         }
         at += total;
         snapshot_index++;
      }
      at = snapshot_end;
      stream_index++;
   }
}

static void
hk_g17p_faulttrace_decode_history(struct hk_device *dev,
                                  struct hk_queue *queue)
{
   if (!(dev->dev.debug & AGX_DBG_FAULTTRACE))
      return;

   const unsigned first =
      (queue->g17p_faulttrace_head + HK_G17P_FAULTTRACE_HISTORY -
       queue->g17p_faulttrace_count) % HK_G17P_FAULTTRACE_HISTORY;

   /* Persist every exact graphics snapshot before verbose decoding.  A
    * watchdog reboot follows the causal firmware fault within seconds, and
    * active camera motion can leave more than a hundred later small submits
    * to decode.  Losing the raw causal record to diagnostic output ordering
    * would make the reproducer non-actionable. */
   const char *raw_base = getenv("AGXDECODE_DUMP_FILE");
   if (!raw_base || !raw_base[0] || !strcmp(raw_base, "stderr"))
      raw_base = "/tmp/hk-g17p-faulttrace";
   for (unsigned i = 0; i < queue->g17p_faulttrace_count; ++i) {
      const unsigned index = (first + i) % HK_G17P_FAULTTRACE_HISTORY;
      const struct hk_g17p_faulttrace_payload *slot =
         &queue->g17p_faulttrace[index];
      if (!slot->extra || !slot->extra_size)
         continue;

      char path[1024];
      snprintf(path, sizeof(path), "%s.raw.%ld.%04" PRIu64, raw_base,
               (long)getpid(), slot->sequence);
      FILE *file = fopen(path, "wb");
      if (file) {
         const size_t written = fwrite(slot->extra, 1, slot->extra_size, file);
         if (written == slot->extra_size)
            fsync(fileno(file));
         fclose(file);
         fprintf(stderr,
                 "G17P faulttrace: raw history sequence=%" PRIu64
                 " path=%s bytes=%zu written=%zu\n",
                 slot->sequence, path, slot->extra_size, written);
      } else {
         fprintf(stderr,
                 "G17P faulttrace: raw history sequence=%" PRIu64
                 " open failed path=%s errno=%d\n",
                 slot->sequence, path, errno);
      }
   }

   for (unsigned i = 0; i < queue->g17p_faulttrace_count; ++i) {
      const unsigned index = (first + i) % HK_G17P_FAULTTRACE_HISTORY;
      const struct hk_g17p_faulttrace_payload *slot =
         &queue->g17p_faulttrace[index];
      if (!slot->data || !slot->size)
         continue;

      struct util_dynarray payload;
      util_dynarray_init(&payload, NULL);
      void *copy = util_dynarray_grow_bytes(&payload, 1, slot->size);
      if (copy) {
         memcpy(copy, slot->data, slot->size);
         fprintf(stderr,
                 "G17P faulttrace: decoding history %u/%u sequence=%" PRIu64
                 " payload=%zu bytes\n",
                 i + 1, queue->g17p_faulttrace_count, slot->sequence,
                 slot->size);
         agxdecode_drm_cmdbuf(dev->dev.agxdecode, &dev->dev.params, &payload,
                              true);
         agxdecode_next_frame();
      }
      util_dynarray_fini(&payload);

      if (slot->extra && slot->extra_size)
         hk_g17p_faulttrace_compare_live(dev, slot);
   }
}

/*
 * The kernel/firmware jointly impose a limit on commands per submit ioctl, but
 * we can build up arbitrarily large command buffers. We handle this here by
 * looping the ioctl, submitting slices of the command buffers that are within
 * bounds.
 */
static VkResult
queue_submit_looped(struct hk_device *dev, struct hk_queue *queue,
                    struct drm_asahi_submit *submit, unsigned command_count,
                    unsigned ring_idx, const uint64_t prior_base[2])
{
   uint8_t *cmdbuf = (uint8_t *)(uintptr_t)submit->cmdbuf;
   uint32_t offs = 0;
   unsigned submitted_vdm = 0, submitted_cdm = 0;
   unsigned commands_remaining = command_count;

   uint64_t out_syncs =
      submit->syncs + sizeof(struct drm_asahi_sync) * submit->in_sync_count;

   while (commands_remaining) {
      bool first = commands_remaining == command_count;
      bool last = commands_remaining <= max_commands_per_submit(dev);

      unsigned count = MIN2(commands_remaining, max_commands_per_submit(dev));
      commands_remaining -= count;

      assert(!last || commands_remaining == 0);
      assert(count > 0);

      unsigned base_offs = offs;
      struct hk_drm_command_span span;
      if (!hk_drm_next_command_span(cmdbuf, submit->cmdbuf_size, base_offs,
                                    count, &span)) {
         return vk_device_set_lost(
            &dev->vk, "Invalid internal Asahi command stream framing");
      }
      offs = span.end_offset;
      unsigned cdm_count = span.compute_count;
      unsigned vdm_count = span.render_count;
      if (last && offs != submit->cmdbuf_size) {
         return vk_device_set_lost(
            &dev->vk, "Trailing synthetic Asahi commands without hardware");
      }

      /* We need to fix up the barriers since barriers are ioctl-relative */
      for (unsigned scan = base_offs; scan < offs;) {
         struct drm_asahi_cmd_header *cmd = (void *)(cmdbuf + scan);
         scan += sizeof(*cmd) + cmd->size;

         if (cmd->cmd_type != DRM_ASAHI_CMD_RENDER &&
             cmd->cmd_type != DRM_ASAHI_CMD_COMPUTE)
            continue;

         if (!hk_drm_rebase_prior(cmd, cmd + 1, submitted_vdm,
                                  submitted_cdm, prior_base,
                                  queue->sync.exact_prior))
            return vk_device_set_lost(&dev->vk,
                                      "Invalid historical Asahi prefix");
      }

      /* We can't signal the out-syncobjs until all prior work finishes. Since
       * only the last ioctl will signal, make sure it waits on prior ioctls.
       *
       * TODO: there might be a more performant way to do this.
      */
      if (last && !first) {
         assert(span.first_hardware_offset != SIZE_MAX);
         struct drm_asahi_cmd_header *cmd =
            (void *)(cmdbuf + span.first_hardware_offset);

         if (cmd->vdm_barrier == DRM_ASAHI_BARRIER_NONE)
            cmd->vdm_barrier = 0;

         if (cmd->cdm_barrier == DRM_ASAHI_BARRIER_NONE)
            cmd->cdm_barrier = 0;
      }

      /* Without the first-command dependency, later chunks have independent
       * roots too. Each must carry the original input syncs. Legacy kernels
       * retain the first-command chain and receive the waits only once. */
      bool has_in_syncs = first ||
         (dev->dev.params.features & DRM_ASAHI_FEATURE_PER_COMMAND_INPUT_SYNCS);
      bool has_out_syncs = last;

      struct drm_asahi_submit submit_ioctl = {
         .flags = submit->flags,
         .queue_id = submit->queue_id,
         .cmdbuf = submit->cmdbuf + base_offs,
         .cmdbuf_size = offs - base_offs,

         .syncs = has_in_syncs ? submit->syncs : out_syncs,
         .in_sync_count = has_in_syncs ? submit->in_sync_count : 0,
         .out_sync_count = has_out_syncs ? submit->out_sync_count : 0,
      };

      hk_g17p_faulttrace_capture(dev, queue, cmdbuf + base_offs,
                                 offs - base_offs, NULL, 0);
      VkResult result = queue_submit_single(dev, &submit_ioctl, ring_idx);
      if (result != VK_SUCCESS) {
         hk_g17p_faulttrace_decode_history(dev, queue);
         return result;
      }

      submitted_cdm += cdm_count;
      submitted_vdm += vdm_count;
   }

   return VK_SUCCESS;
}

struct hk_bind_builder {
   /* Initialized */
   struct hk_device *dev;
   struct vk_object_base *obj_base;
   struct agx_va *va;
   struct hk_buffer *buffer;
   struct hk_image *image;

   /* State */
   struct hk_device_memory *mem;
   VkDeviceSize resourceOffset;
   VkDeviceSize size;
   VkDeviceSize memoryOffset;
   VkResult result;

   /* Array of drm_asahi_gem_bind_op's */
   struct util_dynarray binds;
};

static inline struct hk_bind_builder
hk_bind_builder(struct hk_device *dev, struct vk_object_base *obj_base,
                struct agx_va *va, struct hk_buffer *buffer,
                struct hk_image *image)
{
   struct hk_bind_builder b = {
      .dev = dev,
      .obj_base = obj_base,
      .va = va,
      .buffer = buffer,
      .image = image,
   };

   b.binds = UTIL_DYNARRAY_INIT;
   return b;
}

static VkResult
hk_flush_bind(struct hk_bind_builder *b)
{
   if (b->result != VK_SUCCESS || b->size == 0) {
      return b->result;
   }

   perf_debug(b->dev, "Sparse bind");

   uint64_t va_addr = b->va->addr + b->resourceOffset;

   /* When the app wants to unbind, replace the bound pages with scratch pages
    * so we don't leave a gap.
    */
   struct drm_asahi_gem_bind_op op;
   if (!b->mem) {
      op = (struct drm_asahi_gem_bind_op){
         .handle = b->dev->dev.scratch_bo->uapi_handle,
         .flags = DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_WRITE |
                  DRM_ASAHI_BIND_SINGLE_PAGE,
         .addr = b->va->addr + b->resourceOffset,
         .range = b->size,
      };
   } else {
      op = (struct drm_asahi_gem_bind_op){
         .handle = b->mem->bo->uapi_handle,
         .flags = DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_WRITE,
         .addr = va_addr,
         .offset = b->memoryOffset,
         .range = b->size,
      };
   }

   util_dynarray_append(&b->binds, op);

   /* Shadow a read-only mapping to the upper half */
   op.flags &= ~DRM_ASAHI_BIND_WRITE;
   op.addr = agx_rw_addr_to_ro(&b->dev->dev, op.addr);

   if (!b->mem) {
      op.handle = b->dev->dev.zero_bo->uapi_handle;
   }

   util_dynarray_append(&b->binds, op);

   return VK_SUCCESS;
}

static VkResult
hk_sparse_buffer_map_commit(struct hk_bind_builder *b)
{
   if (!b->buffer || !b->buffer->sparse_map)
      return VK_SUCCESS;

   struct hk_sparse_buffer_map_header *header =
      agx_bo_map(b->buffer->sparse_map);
   uint32_t *map = (uint32_t *)(header + 1);
   struct drm_asahi_gem_bind_op *ops = b->binds.data;
   const unsigned count = util_dynarray_num_elements(
      &b->binds, struct drm_asahi_gem_bind_op);
   assert((count % 2) == 0);

   for (unsigned i = 0; i < count; i += 2) {
      const struct drm_asahi_gem_bind_op *op = &ops[i];
      assert(op->addr >= b->va->addr);
      const uint64_t first = (op->addr - b->va->addr) / AIL_PAGESIZE;
      const uint64_t pages = DIV_ROUND_UP(op->range, AIL_PAGESIZE);
      const bool resident =
         op->handle != b->dev->dev.scratch_bo->uapi_handle;

      for (uint64_t page = 0; page < pages; ++page) {
         if (first + page >= b->buffer->sparse_map_entries)
            break;

         const uint64_t target = first + page;
         const uint32_t mask = BITFIELD_BIT(target & 31);
         if (resident)
            map[target >> 5] |= mask;
         else
            map[target >> 5] &= ~mask;
      }
   }

   if (agx_bo_sync_for_device(b->buffer->sparse_map, 0,
                              b->buffer->sparse_map->size)) {
      return vk_errorf(b->dev, VK_ERROR_DEVICE_LOST,
                       "Sparse buffer residency map upload failed");
   }

   return VK_SUCCESS;
}

static void
hk_sparse_image_map_commit(struct hk_bind_builder *b)
{
   if (!b->image || !b->image->planes[0].sparse_map)
      return;

   assert(b->image->plane_count == 1 && "multiplane sparse not supported");

   uint32_t *map = agx_bo_map(b->image->planes[0].sparse_map);
   struct ail_layout *layout = &b->image->planes[0].layout;
   const uint64_t layer_stride_page =
      ail_bytes_to_pages(layout->layer_stride_B);
   struct drm_asahi_gem_bind_op *ops = b->binds.data;
   const unsigned count = util_dynarray_num_elements(
      &b->binds, struct drm_asahi_gem_bind_op);
   assert((count % 2) == 0);

   /* Each logical sparse bind is followed by its read-only shadow.  Publish
    * residency metadata only after the atomic kernel batch succeeded. */
   for (unsigned i = 0; i < count; i += 2) {
      const struct drm_asahi_gem_bind_op *op = &ops[i];
      assert(op->addr >= b->va->addr);
      const uint64_t resource_offset = op->addr - b->va->addr;
      const uint64_t size_page = ail_bytes_to_pages(op->range);
      const bool enabled =
         op->handle != b->dev->dev.scratch_bo->uapi_handle;

      for (uint64_t offs_page = 0; offs_page < size_page; ++offs_page) {
         const uint64_t target_page =
            ail_bytes_to_pages(resource_offset) + offs_page;
         const uint64_t z = target_page / layer_stride_page;
         if (z >= layout->depth_px)
            break;

         const uint64_t page_in_layer = target_page % layer_stride_page;
         const unsigned idx =
            ail_page_to_sparse_index_el(layout, z, page_in_layer);

         agx_pack(map + idx, SPARSE_BLOCK, cfg) {
            cfg.enabled = enabled;
            cfg.unknown = enabled;
            if (enabled)
               cfg.address = op->addr + (offs_page * AIL_PAGESIZE);
         }
      }
   }
}

static VkResult
hk_bind_builder_finish(struct hk_bind_builder *b)
{
   b->result = hk_flush_bind(b);

   /* Submit everything to the kernel at once */
   if (b->result == VK_SUCCESS && b->binds.size > 0) {
      const unsigned op_count =
         util_dynarray_num_elements(&b->binds, struct drm_asahi_gem_bind_op);

      /* One op per hardware tile plus its read-only shadow, so a large
       * layered mip chain submits thousands.  The kernel copies and stages
       * the whole batch, so the op count is what decides whether this bind
       * can be satisfied when memory is tight -- print it when asked, rather
       * than leaving a VM_BIND ENOMEM with no size attached to it. */
      perf_debug_dev(&b->dev->dev, "Sparse bind batch: %u ops (%u bytes)",
                     op_count, (unsigned)b->binds.size);

      int ret = b->dev->dev.ops.bo_bind(&b->dev->dev, b->binds.data,
                                        op_count);
      if (ret) {
         const int error = errno ? errno : EIO;
         b->result = vk_errorf(
            b->dev,
            error == ENOMEM ? VK_ERROR_OUT_OF_DEVICE_MEMORY : VK_ERROR_UNKNOWN,
            "Sparse VM_BIND batch failed: %s", strerror(error));
      } else {
         b->result = hk_sparse_buffer_map_commit(b);
         if (b->result == VK_SUCCESS)
            hk_sparse_image_map_commit(b);
      }
   }

   util_dynarray_fini(&b->binds);
   return b->result;
}

static void
hk_add_bind(struct hk_bind_builder *b, struct hk_device_memory *mem,
            VkDeviceSize resourceOffset, VkDeviceSize size,
            VkDeviceSize memoryOffset)
{
   /* Discard trivial binds to simplify the below logic. */
   if (size == 0)
      return;

   /* Try to merge with the previous bind */
   if (b->size && b->mem == mem &&
       resourceOffset == b->resourceOffset + b->size &&
       (!mem || memoryOffset == b->memoryOffset + b->size)) {

      b->size += size;
      return;
   }

   /* Otherwise, flush the previous bind and replace with the new one */
   hk_flush_bind(b);
   b->mem = mem;
   b->resourceOffset = resourceOffset;
   b->size = size;
   b->memoryOffset = memoryOffset;
}

static VkResult
hk_sparse_buffer_bind_memory(struct hk_device *device,
                             const VkSparseBufferMemoryBindInfo *bind)
{
   VK_FROM_HANDLE(hk_buffer, buffer, bind->buffer);

   struct hk_bind_builder b =
      hk_bind_builder(device, &buffer->vk.base, buffer->va, buffer, NULL);

   for (uint32_t i = 0; i < bind->bindCount; ++i) {
      struct hk_device_memory *cur_mem = NULL;

      if (bind->pBinds[i].memory != VK_NULL_HANDLE)
         cur_mem = hk_device_memory_from_handle(bind->pBinds[i].memory);

      hk_add_bind(&b, cur_mem, bind->pBinds[i].resourceOffset,
                  bind->pBinds[i].size, bind->pBinds[i].memoryOffset);
   }

   return hk_bind_builder_finish(&b);
}

static VkResult
hk_sparse_image_opaque_bind_memory(
   struct hk_device *device, const VkSparseImageOpaqueMemoryBindInfo *bind)
{
   VK_FROM_HANDLE(hk_image, image, bind->image);

   struct hk_bind_builder b =
      hk_bind_builder(device, &image->vk.base, image->planes[0].va, NULL,
                      image);

   for (uint32_t i = 0; i < bind->bindCount; ++i) {
      struct hk_device_memory *mem = NULL;
      if (bind->pBinds[i].memory != VK_NULL_HANDLE)
         mem = hk_device_memory_from_handle(bind->pBinds[i].memory);

      VkDeviceSize resourceOffset = bind->pBinds[i].resourceOffset;

      /* Conceptually, the miptail is a single region at the end of the image,
       * possibly layered. However, due to alignment requirements we need to
       * use a non-layered miptail and internally fan out to each of the layers.
       * This is facilitated by the HK_MIP_TAIL_START_OFFSET magic offset, see
       * the comment where that is defined for more detail.
       */
      if (resourceOffset >= HK_MIP_TAIL_START_OFFSET) {
         assert(resourceOffset == HK_MIP_TAIL_START_OFFSET &&
                "must bind whole miptail... maybe...");

         const struct ail_layout *layout = &image->planes[0].layout;
         unsigned tail_offset_B =
            layout->level_offsets_B[layout->mip_tail_first_lod];

         for (unsigned z = 0; z < layout->depth_px; ++z) {
            uint64_t image_offs = tail_offset_B + (z * layout->layer_stride_B);
            uint64_t mem_offs =
               bind->pBinds[i].memoryOffset + (z * layout->mip_tail_stride);

            hk_add_bind(&b, mem, image_offs, layout->mip_tail_stride, mem_offs);
         }
      } else {
         hk_add_bind(&b, mem, bind->pBinds[i].resourceOffset,
                     bind->pBinds[i].size, bind->pBinds[i].memoryOffset);
      }
   }

   return hk_bind_builder_finish(&b);
}

static void
bind_hw_tile(struct hk_bind_builder *b, struct hk_device_memory *mem,
             struct ail_layout *layout, unsigned layer, unsigned level,
             VkOffset3D offset, VkExtent3D extent, struct ail_tile std_size_el,
             unsigned mem_offset, unsigned x, unsigned y, unsigned z)
{
   uint64_t bo_offset_B = ail_get_twiddled_block_B(
      layout, level, offset.x + x, offset.y + y, layer + offset.z + z);

   /* Consider the standard tiles in the bound memory to be in raster order, and
    * address accordingly in standard tiles.
    */
   unsigned mem_x_stl = x / std_size_el.width_el;
   unsigned mem_y_stl = y / std_size_el.height_el;
   unsigned extent_w_stl = DIV_ROUND_UP(extent.width, std_size_el.width_el);
   unsigned extent_y_stl = DIV_ROUND_UP(extent.height, std_size_el.height_el);
   unsigned mem_offs_stl = (extent_y_stl * extent_w_stl * z) +
                           (extent_w_stl * mem_y_stl) + mem_x_stl;

   /* There are 4 hardware tiles per standard tile, so offset
    * accordingly for each hardware tile.
    */
   unsigned mem_offset_B = mem_offset + (mem_offs_stl * 4 * AIL_PAGESIZE);

   if (x % std_size_el.width_el)
      mem_offset_B += AIL_PAGESIZE;

   if (y % std_size_el.height_el)
      mem_offset_B += (2 * AIL_PAGESIZE);

   hk_add_bind(b, mem, bo_offset_B, AIL_PAGESIZE, mem_offset_B);
}

static VkResult
hk_sparse_image_bind_memory(struct hk_device *device,
                            const VkSparseImageMemoryBindInfo *bind)
{
   VK_FROM_HANDLE(hk_image, image, bind->image);
   struct ail_layout *layout = &image->planes[0].layout;

   struct hk_bind_builder b =
      hk_bind_builder(device, &image->vk.base, image->planes[0].va, NULL,
                      image);

   for (uint32_t i = 0; i < bind->bindCount; ++i) {
      struct hk_device_memory *mem = NULL;
      if (bind->pBinds[i].memory != VK_NULL_HANDLE)
         mem = hk_device_memory_from_handle(bind->pBinds[i].memory);

      uint64_t mem_offset = bind->pBinds[i].memoryOffset;
      const uint32_t layer = bind->pBinds[i].subresource.arrayLayer;
      const uint32_t level = bind->pBinds[i].subresource.mipLevel;

      VkExtent3D bind_extent = bind->pBinds[i].extent;
      bind_extent.width = DIV_ROUND_UP(
         bind_extent.width, vk_format_get_blockwidth(image->vk.format));
      bind_extent.height = DIV_ROUND_UP(
         bind_extent.height, vk_format_get_blockheight(image->vk.format));

      VkOffset3D bind_offset = bind->pBinds[i].offset;
      bind_offset.x /= vk_format_get_blockwidth(image->vk.format);
      bind_offset.y /= vk_format_get_blockheight(image->vk.format);

      /* Hardware tiles are exactly one page (16K) */
      struct ail_tile tilesize_el = layout->tilesize_el[level];
      unsigned size_B = tilesize_el.width_el * tilesize_el.height_el *
                        ail_get_blocksize_B(layout);

      assert(size_B == AIL_PAGESIZE && "fundamental to AGX");

      /* Standard tiles are exactly 4 pages (65K), consisting of a 2x2 grid of
       * hardware tiles.
       */
      struct ail_tile std_size_el = tilesize_el;
      std_size_el.width_el *= 2;
      std_size_el.height_el *= 2;

      for (unsigned z = 0; z < bind_extent.depth; z += 1) {
         for (unsigned y = 0; y < bind_extent.height;
              y += tilesize_el.height_el) {
            for (unsigned x = 0; x < bind_extent.width;
                 x += tilesize_el.width_el) {
               bind_hw_tile(&b, mem, layout, layer, level, bind_offset,
                            bind_extent, std_size_el, mem_offset, x, y, z);
            }
         }
      }
   }

   return hk_bind_builder_finish(&b);
}

static VkResult
hk_queue_submit_bind_sparse_memory(struct hk_device *device,
                                   struct vk_queue_submit *submission)
{
   assert(submission->command_buffer_count == 0);

   for (uint32_t i = 0; i < submission->buffer_bind_count; ++i) {
      VkResult result =
         hk_sparse_buffer_bind_memory(device, submission->buffer_binds + i);
      if (result != VK_SUCCESS)
         return result;
   }

   for (uint32_t i = 0; i < submission->image_opaque_bind_count; ++i) {
      VkResult result = hk_sparse_image_opaque_bind_memory(
         device, submission->image_opaque_binds + i);
      if (result != VK_SUCCESS)
         return result;
   }

   for (uint32_t i = 0; i < submission->image_bind_count; ++i) {
      VkResult result =
         hk_sparse_image_bind_memory(device, submission->image_binds + i);
      if (result != VK_SUCCESS)
         return result;
   }

   return VK_SUCCESS;
}

static bool
hk_sparse_submission_updates_sstt(const struct vk_queue_submit *submission)
{
   for (uint32_t i = 0; i < submission->image_opaque_bind_count; ++i) {
      VK_FROM_HANDLE(hk_image, image, submission->image_opaque_binds[i].image);
      if (submission->image_opaque_binds[i].bindCount &&
          image->planes[0].sparse_map)
         return true;
   }

   for (uint32_t i = 0; i < submission->image_bind_count; ++i) {
      VK_FROM_HANDLE(hk_image, image, submission->image_binds[i].image);
      if (submission->image_binds[i].bindCount &&
          image->planes[0].sparse_map)
         return true;
   }

   return false;
}

/* v96 HK_G17P_SLOW_SUBMIT_US=N (diagnostic, runtime-only): print every
 * submission whose host-side work took N us or longer, split into command
 * construction (sync resolve, attachment lists, command fill) and the ioctl. */
static uint64_t
hk_g17p_slow_submit_us(void)
{
   static int64_t cached = -1;
   if (unlikely(cached < 0)) {
      const char *v = getenv("HK_G17P_SLOW_SUBMIT_US");
      cached = v ? strtoull(v, NULL, 0) : 0;
   }
   return cached;
}

static VkResult
queue_submit(struct hk_device *dev, struct hk_queue *queue,
             struct vk_queue_submit *submit)
{
   bool flush_sstt = false;
   const uint64_t slow_us = hk_g17p_slow_submit_us();
   const uint64_t slow_t0 = slow_us ? os_time_get_nano() : 0;

   /* TODO: Support asynchronous sparse queue? */
   if (submit->buffer_bind_count || submit->image_bind_count ||
       submit->image_opaque_bind_count) {

      VkResult result = hk_queue_submit_bind_sparse_memory(dev, submit);
      if (result != VK_SUCCESS)
         return result;

      flush_sstt = dev->dev.chip == AGX_CHIP_G17P &&
                   hk_sparse_submission_updates_sstt(submit);
   }

   unsigned command_count = flush_sstt ? 1 : 0;

   /* Gather the number of individual commands to submit up front */
   for (unsigned i = 0; i < submit->command_buffer_count; ++i) {
      struct hk_cmd_buffer *cmdbuf =
         (struct hk_cmd_buffer *)submit->command_buffers[i];

      command_count += list_length(&cmdbuf->control_streams);
   }

   perf_debug_dev(&dev->dev,
                  "Submitting %u control streams (%u command buffers)",
                  command_count, submit->command_buffer_count);

   if (command_count == 0) {
      /* A barrier-only command buffer still has a second scope in later
       * submissions. Unsupported kernels retain their existing path. */
      if (queue->sync.exact_prior &&
          !hk_sync_resolve_submit(
             queue, (struct hk_cmd_buffer *const *)submit->command_buffers,
             submit->command_buffer_count, false, false, 0))
         return vk_device_set_lost(&dev->vk, "Asahi command ordinal overflow");
      return queue_submit_empty(dev, queue, submit);
   }

   const uint64_t prior_base[HK_SYNC_NUM_SQ] = {
      queue->sync.submitted[HK_SYNC_VDM], queue->sync.submitted[HK_SYNC_CDM],
   };

   /* G17P permits independent engines and physical render queues to finish
    * out of order. Publish API signals from the cumulative queue point after
    * the ioctl, preserving independent execution while covering earlier work.
    */
   const bool cumulative_signals = dev->dev.chip == AGX_CHIP_G17P;
   const unsigned kernel_signal_count =
      cumulative_signals ? 0 : submit->signal_count;
   unsigned wait_count = 0;
   struct drm_asahi_sync *syncs =
      alloca((submit->wait_count + kernel_signal_count + 1) *
             sizeof(struct drm_asahi_sync));

   for (unsigned i = 0; i < submit->wait_count; ++i) {
      /* The kernel rejects the submission if we try to wait on the same
       * timeline semaphore at multiple points.
       *
       * TODO: Can we relax the UAPI?
       *
       * XXX: This is quadratic time.
       */
      bool skip = false;
      if (submit->waits[i].sync->flags & VK_SYNC_IS_TIMELINE) {
         const uint64_t v1 = submit->waits[i].wait_value;
         for (unsigned j = 0; j < submit->wait_count; ++j) {
            const uint64_t v2 = submit->waits[j].wait_value;
            if (i != j && submit->waits[i].sync == submit->waits[j].sync &&
                (v1 < v2 || (v1 == v2 && i < j))) {
               skip = true;
               break;
            }
         }

         if (skip)
            continue;
      }

      asahi_fill_sync(&syncs[wait_count++], submit->waits[i].sync,
                      submit->waits[i].wait_value);
   }

   for (unsigned i = 0; i < kernel_signal_count; ++i) {
      asahi_fill_sync(&syncs[wait_count + i], submit->signals[i].sync,
                      submit->signals[i].signal_value);
   }

   /* Signal progress on the queue itself */
   syncs[wait_count + kernel_signal_count] = (struct drm_asahi_sync){
      .sync_type = DRM_ASAHI_SYNC_TIMELINE_SYNCOBJ,
      .handle = queue->drm.syncobj,
      .timeline_value = ++queue->drm.timeline_value,
   };

   /* Now setup the command structs */
   struct util_dynarray payload = UTIL_DYNARRAY_INIT;
   struct util_dynarray faulttrace_extra = UTIL_DYNARRAY_INIT;

   unsigned nr_vdm = 0, nr_cdm = 0;
   unsigned nr_draws = 0, nr_dispatches = 0, nr_flushes = 0, nr_merged = 0;

   if (flush_sstt) {
      assert(queue->g17p_sstt_flush_bo);
      struct drm_asahi_cmd_header header = agx_cmd_header(true, 0, 0);
      const struct drm_asahi_cmd_compute_with_prior cmd = {
         .legacy = {
            .cdm_ctrl_stream_base = queue->g17p_sstt_flush_bo->va->addr,
            .cdm_ctrl_stream_end =
               queue->g17p_sstt_flush_bo->va->addr +
               AGX3_CDM_SSTT_FLUSH_INVALIDATE_SIZE +
               AGX3_CDM_STREAM_TERMINATE_SIZE,
         },
      };
      if (queue->sync.exact_prior)
         header.size = sizeof(cmd);
      util_dynarray_append(&payload, header);
      util_dynarray_append_array(&payload, uint8_t, &cmd, header.size);
      nr_cdm++;
      nr_flushes++;
   }

   /* Tile hazards against this queue's earlier submissions need the point
    * the queue has completed; fetched only when a partial-tile render pass
    * instance is submitted while such a footprint may still be running. On
    * failure every earlier submission is treated as still running. */
   uint64_t completed = 0;
   if (hk_sync_submit_needs_completed(
          queue, (struct hk_cmd_buffer *const *)submit->command_buffers,
          submit->command_buffer_count)) {
      uint64_t point = 0;
      if (drmSyncobjQuery(dev->dev.fd, &queue->drm.syncobj, &point, 1) == 0)
         completed = point;
   }

   /* Every command's barrier fields come from recorded dependencies. Legacy
    * kernels attach input syncs only to the first command, requiring a chain.
    * A kernel advertising per-command waits needs no such execution edge. */
   if (!hk_sync_resolve_submit(queue,
                          (struct hk_cmd_buffer *const *)submit->command_buffers,
                          submit->command_buffer_count, flush_sstt,
                          wait_count > 0 &&
                          !(dev->dev.params.features &
                            DRM_ASAHI_FEATURE_PER_COMMAND_INPUT_SYNCS),
                          completed)) {
      util_dynarray_fini(&payload);
      util_dynarray_fini(&faulttrace_extra);
      return vk_device_set_lost(&dev->vk, "Asahi command ordinal overflow");
   }

   for (unsigned i = 0; i < submit->command_buffer_count; ++i) {
      struct hk_cmd_buffer *cmdbuf =
         (struct hk_cmd_buffer *)submit->command_buffers[i];

      list_for_each_entry(struct hk_cs, cs, &cmdbuf->control_streams, node) {
         struct drm_asahi_cmd_header header =
            agx_cmd_header(cs->type == HK_CS_CDM, cs->sync.barrier[HK_SYNC_VDM],
                           cs->sync.barrier[HK_SYNC_CDM]);
         assert(header.vdm_barrier == DRM_ASAHI_BARRIER_NONE ||
                header.vdm_barrier <= nr_vdm);
         assert(header.cdm_barrier == DRM_ASAHI_BARRIER_NONE ||
                header.cdm_barrier <= nr_cdm);

         if (cs->type == HK_CS_CDM) {
            perf_debug(
               cmdbuf,
               "%u: Submitting CDM with %u API calls, %u dispatches, %u flushes, %u merged",
               i, cs->stats.calls, cs->stats.cmds, cs->stats.flushes,
               cs->stats.merged);

            assert(cs->stats.cmds > 0 || cs->stats.flushes > 0 ||
                   cs->timestamp.end.handle || cs->timestamp.start.handle);

            asahi_append_compute_attachments(dev, &payload, cs);
            struct drm_asahi_cmd_compute_with_prior cmd = {0};
            asahi_fill_cdm_command(dev, cs, &cmd.legacy);
            if (queue->sync.exact_prior) {
               header.size = sizeof(cmd);
               cmd.prior_vdm = cs->sync.prior[HK_SYNC_VDM];
               cmd.prior_cdm = cs->sync.prior[HK_SYNC_CDM];
            }
            util_dynarray_append(&payload, header);
            util_dynarray_append_array(&payload, uint8_t, &cmd, header.size);
            nr_cdm++;
            nr_dispatches += cs->stats.cmds;
            nr_flushes += cs->stats.flushes;
            nr_merged += cs->stats.merged;
         } else {
            assert(cs->type == HK_CS_VDM);
            perf_debug(cmdbuf, "%u: Submitting VDM with %u API draws, %u draws",
                       i, cs->stats.calls, cs->stats.cmds);
            assert(cs->stats.cmds > 0 || cs->cr.process_empty_tiles ||
                   cs->timestamp.end.handle || cs->timestamp.start.handle);

            asahi_append_fragment_attachments(dev, &payload, cs);
            struct drm_asahi_cmd_render_with_prior cmd = {0};
            asahi_fill_vdm_command(dev, cs, &cmd.legacy, &faulttrace_extra);
            header.size = hk_drm_render_command_size(
               dev->dev.chip == AGX_CHIP_G17P, queue->sync.exact_prior);
            if (queue->sync.exact_prior) {
               cmd.prior_vdm = cs->sync.prior[HK_SYNC_VDM];
               cmd.prior_cdm = cs->sync.prior[HK_SYNC_CDM];
            }
            util_dynarray_append(&payload, header);
            util_dynarray_append_array(&payload, uint8_t, &cmd, header.size);
            nr_vdm++;
            nr_draws += cs->stats.cmds;
         }
      }
   }

   if (dev->dev.chip == AGX_CHIP_G17P && hk_g17p_submit_trace_enabled()) {
      static int sequence;
      const unsigned submit_sequence = p_atomic_inc_return(&sequence);
      const unsigned limit = max_commands_per_submit(dev);
      const unsigned ioctls = DIV_ROUND_UP(command_count, limit);
      fprintf(stderr,
              "G17P submit trace: seq=%u streams=%u vdm=%u cdm=%u draws=%u "
              "dispatches=%u flushes=%u merged=%u command-buffers=%u "
              "waits=%u signals=%u ioctls=%u limit=%u\n",
              submit_sequence, command_count, nr_vdm, nr_cdm, nr_draws,
              nr_dispatches, nr_flushes, nr_merged,
              submit->command_buffer_count, wait_count, submit->signal_count,
              ioctls, limit);
      fflush(stderr);
   }

   if ((dev->dev.debug & AGX_DBG_TRACE) && hk_agxdecode_trace_armed()) {
      agxdecode_drm_cmdbuf(dev->dev.agxdecode, &dev->dev.params, &payload,
                           true);

      agxdecode_next_frame();
   }

   struct drm_asahi_submit submit_ioctl = {
      .flags = 0,
      .queue_id = queue->drm.id,
      .in_sync_count = wait_count,
      .out_sync_count = kernel_signal_count + 1,
      .cmdbuf_size = payload.size,
      .syncs = (uint64_t)(uintptr_t)(syncs),
      .cmdbuf = (uint64_t)(uintptr_t)(payload.data),
   };

   VkResult result;
   const uint64_t slow_t1 = slow_us ? os_time_get_nano() : 0;
   if (command_count <= max_commands_per_submit(dev)) {
      hk_g17p_faulttrace_capture(dev, queue, payload.data, payload.size,
                                 faulttrace_extra.data,
                                 faulttrace_extra.size);
      result =
         queue_submit_single(dev, &submit_ioctl, queue->drm.virt_ring_idx);
      if (result != VK_SUCCESS)
         hk_g17p_faulttrace_decode_history(dev, queue);
   } else {
      result = queue_submit_looped(dev, queue, &submit_ioctl, command_count,
                                   queue->drm.virt_ring_idx, prior_base);
   }
   if (cumulative_signals) {
      /* The kernel can publish an error fence even when submission fails.
       * Forward it when present, without waiting for a missing queue point
       * or replacing the original submit error with a transfer error. */
      const VkResult signal_result = queue_signal_timeline(dev, queue, submit);
      if (result == VK_SUCCESS)
         result = signal_result;
   }

   if (slow_us) {
      const uint64_t slow_t2 = os_time_get_nano();
      if ((slow_t2 - slow_t0) / 1000 >= slow_us)
         fprintf(stderr,
                 "G17P slow submit: us=%.1f build_us=%.1f ioctl_us=%.1f "
                 "streams=%u vdm=%u cdm=%u draws=%u command_buffers=%u "
                 "waits=%u signals=%u payload_B=%u result=%d\n",
                 (slow_t2 - slow_t0) / 1000.0, (slow_t1 - slow_t0) / 1000.0,
                 (slow_t2 - slow_t1) / 1000.0, command_count, nr_vdm, nr_cdm,
                 nr_draws, submit->command_buffer_count, wait_count,
                 submit->signal_count, payload.size, result);
   }

   util_dynarray_fini(&payload);
   util_dynarray_fini(&faulttrace_extra);
   return result;
}

static VkResult
hk_queue_submit(struct vk_queue *vk_queue, struct vk_queue_submit *submit)
{
   struct hk_queue *queue = container_of(vk_queue, struct hk_queue, vk);
   struct hk_device *dev = hk_queue_device(queue);

   if (vk_queue_is_lost(&queue->vk))
      return VK_ERROR_DEVICE_LOST;

   VkResult result = queue_submit(dev, queue, submit);
   if (result != VK_SUCCESS)
      result = vk_queue_set_lost(&queue->vk, "Submit failed");

   if (dev->dev.debug & AGX_DBG_SYNC) {
      /* Wait for completion */
      int err = drmSyncobjTimelineWait(
         dev->dev.fd, &queue->drm.syncobj, &queue->drm.timeline_value, 1,
         INT64_MAX, DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT, NULL);

      if (err) {
         result = vk_queue_set_lost(&queue->vk, "Wait failed");
      } else {
         VkResult res = dev->vk.check_status(&dev->vk);
         if (result == VK_SUCCESS)
            result = res;
      }
   }

   return result;
}

static enum drm_asahi_priority
translate_priority(VkQueueGlobalPriorityKHR prio)
{
   switch (prio) {
   case VK_QUEUE_GLOBAL_PRIORITY_REALTIME_KHR:
      return DRM_ASAHI_PRIORITY_REALTIME;

   case VK_QUEUE_GLOBAL_PRIORITY_HIGH_KHR:
      return DRM_ASAHI_PRIORITY_HIGH;

   case VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR:
      return DRM_ASAHI_PRIORITY_MEDIUM;

   case VK_QUEUE_GLOBAL_PRIORITY_LOW_KHR:
      return DRM_ASAHI_PRIORITY_LOW;

   default:
      UNREACHABLE("Invalid VkQueueGlobalPriorityKHR");
   }
}

VkResult
hk_queue_init(struct hk_device *dev, const VkDeviceQueueCreateInfo *pCreateInfo,
              uint32_t index_in_family)
{
   struct hk_physical_device *pdev = hk_device_physical(dev);
   VkResult result;

   assert(pCreateInfo->queueFamilyIndex < pdev->queue_family_count);

   struct hk_queue *queue = vk_zalloc(&dev->vk.alloc, sizeof(struct hk_queue),
                                      8, VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!queue)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   const VkDeviceQueueGlobalPriorityCreateInfoKHR *priority_info =
      vk_find_struct_const(pCreateInfo->pNext,
                           DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_KHR);
   const VkQueueGlobalPriorityKHR priority =
      priority_info ? priority_info->globalPriority
                    : VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR;

   /* High-priority scheduling still needs kernel qualification. */
   enum drm_asahi_priority drm_priority = translate_priority(priority);
   if (drm_priority >= DRM_ASAHI_PRIORITY_HIGH) {
      vk_free(&dev->vk.alloc, queue);
      return VK_ERROR_NOT_PERMITTED_EXT;
   }

   result = vk_queue_init(&queue->vk, &dev->vk, pCreateInfo, index_in_family);
   if (result != VK_SUCCESS) {
      vk_free(&dev->vk.alloc, queue);
      return result;
   }

   queue->vk.driver_submit = hk_queue_submit;
   queue->sync.exact_prior =
      (dev->dev.params.features & DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS) != 0;

   queue->drm.id = agx_create_command_queue(&dev->dev, drm_priority);
   if (queue->drm.id == UINT32_MAX) {
      const int error = errno;
      vk_queue_finish(&queue->vk);
      vk_free(&dev->vk.alloc, queue);
      result = error == ENOMEM ? VK_ERROR_OUT_OF_HOST_MEMORY :
               (error == ENOSPC || error == EBUSY) ? VK_ERROR_OUT_OF_DEVICE_MEMORY :
               (error == EPERM || error == EACCES) ? VK_ERROR_NOT_PERMITTED_EXT :
               VK_ERROR_INITIALIZATION_FAILED;
      return vk_errorf(dev, result, "DRM_IOCTL_ASAHI_QUEUE_CREATE failed: %s",
                       strerror(error));
   }
   queue->drm.virt_ring_idx = drm_priority + 1;

   if (drmSyncobjCreate(dev->dev.fd, 0, &queue->drm.syncobj)) {
      mesa_loge("drmSyncobjCreate() failed %d\n", errno);
      agx_destroy_command_queue(&dev->dev, queue->drm.id);
      vk_queue_finish(&queue->vk);
      vk_free(&dev->vk.alloc, queue);

      return vk_errorf(dev, VK_ERROR_OUT_OF_HOST_MEMORY,
                       "DRM_IOCTL_SYNCOBJ_CREATE failed: %m");
   }

   uint64_t initial_value = 1;
   if (drmSyncobjTimelineSignal(dev->dev.fd, &queue->drm.syncobj,
                                &initial_value, 1)) {
      hk_queue_finish(dev, queue);
      return vk_errorf(dev, VK_ERROR_OUT_OF_HOST_MEMORY,
                       "DRM_IOCTL_TIMELINE_SYNCOBJ_SIGNAL failed: %m");
   }

   /* Point 1 is already signalled.  Keep the software cursor in lockstep so
    * the first real submit advances to point 2 instead of reusing the
    * completed bootstrap point.  QueueWaitIdle transfers from this cursor;
    * reusing point 1 lets it return before the first submit has completed. */
   queue->drm.timeline_value = initial_value;

   if (dev->dev.chip == AGX_CHIP_G17P &&
       dev->vk.enabled_features.sparseBinding) {
      queue->g17p_sstt_flush_bo =
         agx_bo_create(&dev->dev, AIL_PAGESIZE, 0, 0, "SSTT flush stream");
      if (!queue->g17p_sstt_flush_bo) {
         hk_queue_finish(dev, queue);
         return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      }

      uint8_t *stream = agx_bo_map(queue->g17p_sstt_flush_bo);
      if (!stream ||
          !agx3_cdm_sstt_flush_invalidate_pack(AGX_CHIP_G17P, stream) ||
          !agx3_cdm_stream_terminate_pack(
             AGX_CHIP_G17P,
             stream + AGX3_CDM_SSTT_FLUSH_INVALIDATE_SIZE)) {
         hk_queue_finish(dev, queue);
         return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      }
   }

   return VK_SUCCESS;
}

void
hk_queue_finish(struct hk_device *dev, struct hk_queue *queue)
{
   for (unsigned i = 0; i < HK_G17P_FAULTTRACE_HISTORY; ++i) {
      free(queue->g17p_faulttrace[i].data);
      free(queue->g17p_faulttrace[i].extra);
   }
   drmSyncobjDestroy(dev->dev.fd, queue->drm.syncobj);
   agx_destroy_command_queue(&dev->dev, queue->drm.id);
   agx_bo_unreference(&dev->dev, queue->g17p_sstt_flush_bo);
   vk_queue_finish(&queue->vk);
   vk_free(&dev->vk.alloc, queue);
}
