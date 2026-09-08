/* SPDX-License-Identifier: MIT */
#pragma once

#include "asahi/lib/agx_chip.h"

#include <stdbool.h>
#include <stddef.h>

#define HK_CS_ROOT_SIZE       65536u
/* Native G17P allocates its ordinary graphics control-stream root as one
 * 0x8000-byte object and concatenates draws in that root. Keeping the native
 * extent also leaves enough room for the conservative per-draw reservation
 * without forcing a link at the ~0x800-byte point reached by Neon.
 */
#define HK_G17P_VDM_ROOT_SIZE    32768u
#define HK_G17P_VDM_INITIAL_SIZE 12u
#define HK_GFX_DRAW_RESERVATION  0x2000u
#define HK_CS_SPACE_PADDING      0x800u

static inline size_t
hk_cs_root_size(enum agx_chip chip, bool compute)
{
   return chip == AGX_CHIP_G17P && !compute ? HK_G17P_VDM_ROOT_SIZE
                                            : HK_CS_ROOT_SIZE;
}

static inline bool
hk_cs_space_fits(size_t remaining, size_t requested, size_t link_length)
{
   return requested <= remaining && link_length <= remaining - requested &&
          HK_CS_SPACE_PADDING <= remaining - requested - link_length;
}

static inline bool
hk_g17p_vdm_needs_render_split(enum agx_chip chip, unsigned commands,
                               size_t remaining, size_t link_length)
{
   return chip == AGX_CHIP_G17P && commands > 0 &&
          !hk_cs_space_fits(remaining, HK_GFX_DRAW_RESERVATION, link_length);
}

static inline bool
hk_graphics_cs_publishable(enum agx_chip chip, bool initialized,
                           bool active_context_selector_valid)
{
   return initialized &&
          (chip != AGX_CHIP_G17P || active_context_selector_valid);
}
