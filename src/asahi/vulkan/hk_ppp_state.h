/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "agx_m5_va.h"
#include "agx_ppp.h"

/* Vulkan pools hold full GPU VAs. M5 VDM PPP pointers, like its VDM root,
 * encode an offset from the architectural GART base. Native 25G76's first
 * PPP record 00000600/00058000 refers to mapped VA 0x1000058000, not 0x58000.
 * Shader/Ioto and PPP-contained 48-bit scissor bases remain full addresses.
 * G17P's existing low-window allocator already returns its compact address.
 */
static inline bool
hk_ppp_emit_state(enum agx_chip chip, uint8_t **out, uint64_t gpu_base,
                   size_t size)
{
   if (chip == AGX_CHIP_G17G) {
      if (gpu_base < AGX_M5_DRIVER_BASE ||
          gpu_base - AGX_M5_DRIVER_BASE >= (UINT64_C(1) << 39))
         return false;
      gpu_base -= AGX_M5_DRIVER_BASE;
   }
   return agx_ppp_emit_state(chip, out, gpu_base, size);
}
