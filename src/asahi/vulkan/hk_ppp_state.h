/* SPDX-License-Identifier: MIT */
#pragma once

#include "agx_m5_va.h"
#include "agx_ppp.h"

static inline bool
hk_ppp_emit_state(enum agx_chip chip, uint8_t **out, uint64_t gpu_base,
                   size_t size)
{
   if (agx_chip_g17_m5(chip)) {
      if (!agx_m5_vdm_compact(gpu_base, &gpu_base))
         return false;
   }
   return agx_ppp_emit_state(chip, out, gpu_base, size);
}
