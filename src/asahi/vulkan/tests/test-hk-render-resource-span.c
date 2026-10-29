/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "hk_drm_render_core.h"
#include <assert.h>
#include <string.h>

int main(void)
{
   struct {
      struct drm_asahi_cmd_header resources;
      struct drm_asahi_render_program programs[2];
      struct drm_asahi_cmd_header draw;
      struct drm_asahi_cmd_header reset;
      struct drm_asahi_cmd_header compute;
   } stream = {
      .resources = {
         .cmd_type = DRM_ASAHI_SET_RENDER_RESOURCES,
         .size = 2 * sizeof(struct drm_asahi_render_program),
         .vdm_barrier = DRM_ASAHI_BARRIER_NONE,
         .cdm_barrier = DRM_ASAHI_BARRIER_NONE,
      },
      .programs = {{.stage = 0}, {.stage = 1, .primary_bytes = 272}},
      .draw = {.cmd_type = DRM_ASAHI_CMD_RENDER},
      .reset = {
         .cmd_type = DRM_ASAHI_SET_RENDER_RESOURCES,
         .vdm_barrier = DRM_ASAHI_BARRIER_NONE,
         .cdm_barrier = DRM_ASAHI_BARRIER_NONE,
      },
      .compute = {.cmd_type = DRM_ASAHI_CMD_COMPUTE},
   };
   struct hk_drm_command_span span;
   assert(hk_drm_next_command_span(&stream, sizeof(stream), 0, 2, &span));
   assert(span.hardware_count == 2 && span.render_count == 1 && span.compute_count == 1);
   assert(span.first_hardware_offset == offsetof(__typeof__(stream), draw));
   assert(span.end_offset == sizeof(stream));
   assert(hk_drm_next_command_span(&stream, sizeof(stream), 0, 1, &span));
   assert(span.end_offset == offsetof(__typeof__(stream), reset));
   assert(hk_drm_next_command_span(&stream, sizeof(stream), span.end_offset, 1, &span));
   assert(span.compute_count == 1 && span.render_count == 0);
   stream.resources.vdm_barrier = 0;
   assert(!hk_drm_next_command_span(&stream, sizeof(stream), 0, 2, &span));
   stream.resources.vdm_barrier = DRM_ASAHI_BARRIER_NONE;
   stream.resources.size--;
   assert(!hk_drm_next_command_span(&stream, sizeof(stream), 0, 2, &span));
   return 0;
}
