/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT */

#include "../d12_submit.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

/* Model stock Asahi's read_up_to(), including a following compute command.
 * Oversized zero render tails are not skipped by the public kernel parser. */
static bool
stock_parse(const uint8_t *payload, size_t size)
{
   unsigned renders = 0, computes = 0;
   size_t offset = 0;
   while (offset < size) {
      struct drm_asahi_cmd_header header;
      if (size - offset < sizeof(header))
         return false;
      memcpy(&header, payload + offset, sizeof(header));
      offset += sizeof(header);
      size_t kernel_size;
      if (header.cmd_type == DRM_ASAHI_CMD_RENDER) {
         kernel_size = offsetof(struct drm_asahi_cmd_render, bg_eot_rsrc_spec_hi);
         renders++;
      } else if (header.cmd_type == DRM_ASAHI_CMD_COMPUTE) {
         kernel_size = sizeof(struct drm_asahi_cmd_compute);
         computes++;
      } else {
         return false;
      }
      const size_t consumed = header.size < kernel_size ? header.size : kernel_size;
      if (size - offset < consumed)
         return false;
      offset += consumed;
   }
   return renders == 1 && computes == 1;
}

static size_t
pack(uint8_t *payload, size_t render_size)
{
   const struct drm_asahi_cmd_header render = {
      .cmd_type = DRM_ASAHI_CMD_RENDER, .size = render_size,
      .vdm_barrier = DRM_ASAHI_BARRIER_NONE,
      .cdm_barrier = DRM_ASAHI_BARRIER_NONE,
   };
   const struct drm_asahi_cmd_header compute = {
      .cmd_type = DRM_ASAHI_CMD_COMPUTE,
      .size = sizeof(struct drm_asahi_cmd_compute),
   };
   memcpy(payload, &render, sizeof(render));
   const size_t offset = sizeof(render) + render_size;
   memcpy(payload + offset, &compute, sizeof(compute));
   return offset + sizeof(compute) + compute.size;
}

int
main(void)
{
   const enum agx_chip chips[] = {
      AGX_CHIP_G13G, AGX_CHIP_G13X, AGX_CHIP_G14G, AGX_CHIP_G14X,
   };
   const size_t extended = sizeof(struct drm_asahi_cmd_render);
   const size_t legacy = offsetof(struct drm_asahi_cmd_render, bg_eot_rsrc_spec_hi);
   assert(extended == legacy + 16);
   uint8_t payload[2 * sizeof(struct drm_asahi_cmd_header) +
                   sizeof(struct drm_asahi_cmd_render) +
                   sizeof(struct drm_asahi_cmd_compute)] = {0};
   assert(!stock_parse(payload, pack(payload, extended)));
   for (unsigned i = 0; i < sizeof(chips) / sizeof(chips[0]); ++i) {
      assert(d12_render_cmd_size(chips[i], false, DRM_ASAHI_FEATURE_SOFT_FAULTS) == legacy);
      assert(d12_render_cmd_size(chips[i], true, 0) == extended);
      assert(d12_render_cmd_size(chips[i], false,
                                 DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS) == extended);
      memset(payload, 0, sizeof(payload));
      assert(stock_parse(payload, pack(payload, d12_render_cmd_size(chips[i], false, 0))));
   }
   assert(d12_render_cmd_size(AGX_CHIP_G17P, false, 0) == extended);
   assert(d12_render_cmd_size(AGX_CHIP_G17P, true, UINT64_MAX) == extended);
   printf("PASS stock render packet parsing; legacy=%zu extended=%zu; G17P unchanged\n",
          legacy, extended);
   return 0;
}
