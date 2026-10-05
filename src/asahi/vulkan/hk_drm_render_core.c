/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_drm_render_core.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t
low32(uint64_t value)
{
   return (uint32_t)value;
}

static uint32_t
high32(uint64_t value)
{
   return (uint32_t)(value >> 32);
}

static bool
hk_drm_g17p_has_zls_storage(uint64_t depth_base, uint64_t depth_meta,
                            uint64_t stencil_base, uint64_t stencil_meta)
{
   return (depth_base | depth_meta | stencil_base | stencil_meta) != 0;
}

uint32_t
hk_drm_g17p_zls_pixels(uint32_t width, uint32_t height,
                       uint64_t depth_base, uint64_t depth_meta,
                       uint64_t stencil_base, uint64_t stencil_meta)
{
   /* Native G17P keeps ISP_ZLS_PIXELS zero when there is no depth/stencil
    * storage.  Zero is a special inactive encoding, not minus-one 1x1. */
   if (!hk_drm_g17p_has_zls_storage(depth_base, depth_meta, stencil_base,
                                    stencil_meta))
      return 0;

   assert(width >= 1 && width <= 0x8000);
   assert(height >= 1 && height <= 0x8000);
   return (width - 1) | ((height - 1) << 15);
}

uint32_t
hk_drm_g17p_depth_clear(uint32_t depth_clear,
                        uint64_t depth_base, uint64_t depth_meta,
                        uint64_t stencil_base, uint64_t stencil_meta)
{
   return hk_drm_g17p_has_zls_storage(depth_base, depth_meta, stencil_base,
                                      stencil_meta)
             ? depth_clear
             : UINT32_C(0x3f800000);
}

void
hk_drm_render_resource_specs_pack(
   const struct hk_drm_render_resource_specs *specs,
   struct drm_asahi_cmd_render *cmd)
{
   cmd->bg.rsrc_spec = low32(specs->bg);
   cmd->eot.rsrc_spec = low32(specs->eot);
   cmd->partial_bg.rsrc_spec = low32(specs->partial_bg);
   cmd->partial_eot.rsrc_spec = low32(specs->partial_eot);

   cmd->bg_eot_rsrc_spec_hi = high32(specs->eot);
   cmd->bg_eot_partial_rsrc_spec_hi = high32(specs->partial_eot);
   cmd->bg_rsrc_spec_hi = high32(specs->bg);
   cmd->bg_partial_rsrc_spec_hi = high32(specs->partial_bg);

   const uint32_t high = cmd->bg_eot_rsrc_spec_hi |
                         cmd->bg_eot_partial_rsrc_spec_hi |
                         cmd->bg_rsrc_spec_hi | cmd->bg_partial_rsrc_spec_hi;
   if (high)
      cmd->flags |= DRM_ASAHI_RENDER_RSRC_SPEC_HI;
   else
      cmd->flags &= ~DRM_ASAHI_RENDER_RSRC_SPEC_HI;
}

unsigned
hk_drm_submit_command_limit(bool batch_enabled, uint32_t kernel_limit)
{
   const unsigned userspace_limit = batch_enabled ? 64 : 1;

   if (kernel_limit == 0)
      return 1;

   return userspace_limit < kernel_limit ? userspace_limit : kernel_limit;
}

uint16_t
hk_drm_render_command_size(bool g17p, bool exact_prior)
{
   if (exact_prior)
      return sizeof(struct drm_asahi_cmd_render_with_prior);

   /* Legacy kernels consume only the fields they know, without skipping an
    * appended tail. Even zero G17P resource-specifier high words would then
    * be parsed as another command. Keep their original packet boundary. */
   return g17p ? sizeof(struct drm_asahi_cmd_render)
               : offsetof(struct drm_asahi_cmd_render, bg_eot_rsrc_spec_hi);
}

void
hk_drm_attachment_set_init(struct hk_drm_attachment_set *set)
{
   memset(set, 0, sizeof(*set));
   set->complete = true;
}

bool
hk_drm_attachment_set_add(struct hk_drm_attachment_set *set,
                          uint64_t pointer, uint64_t size)
{
   if (!pointer || !size || size > UINT64_MAX - pointer) {
      set->complete = false;
      return false;
   }

   uint64_t start = pointer;
   uint64_t end = pointer + size;

   /* Coalesce duplicate, overlapping, and adjacent image spans. Vulkan may
    * bind depth/stencil planes or several image views into the same memory
    * allocation, and the UAPI has a finite attachment count. */
   for (uint32_t i = 0; i < set->count;) {
      const uint64_t current_start = set->list[i].pointer;
      const uint64_t current_end = current_start + set->list[i].size;
      if (end < current_start || current_end < start) {
         i++;
         continue;
      }

      start = start < current_start ? start : current_start;
      end = end > current_end ? end : current_end;
      set->list[i] = set->list[--set->count];
   }

   if (set->count == HK_DRM_MAX_FRAGMENT_ATTACHMENTS) {
      set->complete = false;
      return false;
   }

   set->list[set->count++] = (struct drm_asahi_attachment){
      .pointer = start,
      .size = end - start,
   };
   return true;
}

static bool
hk_drm_command_is_attachment(uint16_t type)
{
   return type == DRM_ASAHI_SET_VERTEX_ATTACHMENTS ||
          type == DRM_ASAHI_SET_FRAGMENT_ATTACHMENTS ||
          type == DRM_ASAHI_SET_COMPUTE_ATTACHMENTS;
}

bool
hk_drm_next_command_span(const void *cmdbuf, size_t cmdbuf_size,
                         size_t start_offset, unsigned hardware_limit,
                         struct hk_drm_command_span *span)
{
   if (!cmdbuf || !span || !hardware_limit || start_offset > cmdbuf_size)
      return false;

   *span = (struct hk_drm_command_span){
      .end_offset = start_offset,
      .first_hardware_offset = SIZE_MAX,
   };

   const uint8_t *bytes = cmdbuf;
   size_t offset = start_offset;
   while (span->hardware_count < hardware_limit) {
      if (cmdbuf_size - offset < sizeof(struct drm_asahi_cmd_header))
         return false;

      const struct drm_asahi_cmd_header *header = (const void *)(bytes + offset);
      const size_t command_size = sizeof(*header) + header->size;
      if (command_size > cmdbuf_size - offset)
         return false;

      const bool render = header->cmd_type == DRM_ASAHI_CMD_RENDER;
      const bool compute = header->cmd_type == DRM_ASAHI_CMD_COMPUTE;
      if (render || compute) {
         if (span->first_hardware_offset == SIZE_MAX)
            span->first_hardware_offset = offset;
         span->hardware_count++;
         span->render_count += render;
         span->compute_count += compute;
      } else if (hk_drm_command_is_attachment(header->cmd_type)) {
         if (header->vdm_barrier != DRM_ASAHI_BARRIER_NONE ||
             header->cdm_barrier != DRM_ASAHI_BARRIER_NONE ||
             header->size % sizeof(struct drm_asahi_attachment) != 0)
            return false;
      } else {
         return false;
      }

      offset += command_size;
   }

   span->end_offset = offset;
   return true;
}
