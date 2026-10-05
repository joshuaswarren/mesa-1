/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "drm-uapi/asahi_drm.h"
#include "asahi/lib/agx_chip.h"

static inline size_t
d12_render_cmd_size(enum agx_chip chip, bool rsrc_spec_hi, uint64_t features)
{
   /* G17P backends already require the extended render contract. The prior
    * barrier feature also guarantees the extended base packet on virtio,
    * where the guest DRM version describes virtio_gpu rather than Asahi.
    * G13/G14 on stock kernels must omit the tail entirely: read_up_to()
    * consumes min(kernel size, header.size), leaving it as another header.
    */
   if (chip == AGX_CHIP_G17P || rsrc_spec_hi ||
       (features & DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS))
      return sizeof(struct drm_asahi_cmd_render);

   return offsetof(struct drm_asahi_cmd_render, bg_eot_rsrc_spec_hi);
}
