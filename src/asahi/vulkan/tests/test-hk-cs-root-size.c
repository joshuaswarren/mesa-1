/* SPDX-License-Identifier: MIT */
#include "../hk_cs_root_size.h"

#include <assert.h>
#include <stdio.h>

int
main(void)
{
   const size_t command_bo_size = 128 * 1024;
   assert(hk_cs_root_size(AGX_CHIP_G17P, false) == 32768);
   assert(hk_cs_root_size(AGX_CHIP_G17P, true) == 65536);
   assert(hk_cs_root_size(AGX_CHIP_G14G, false) == 65536);
   assert(hk_cs_root_size(AGX_CHIP_G14G, true) == 65536);

   const size_t vdm_link_length = 8;
   assert(!hk_cs_space_fits(4096 - HK_G17P_VDM_INITIAL_SIZE,
                            HK_GFX_DRAW_RESERVATION, vdm_link_length));
   assert(hk_cs_space_fits(HK_G17P_VDM_ROOT_SIZE -
                              HK_G17P_VDM_INITIAL_SIZE,
                           HK_GFX_DRAW_RESERVATION, vdm_link_length));

   /* The exact Neon stream reaches 0x80c bytes including its terminator.
    * That is ordinary in-root traffic for native's 0x8000-byte allocation,
    * not a reason to publish a second VDM root. */
   assert(!hk_g17p_vdm_needs_render_split(
      AGX_CHIP_G17P, 15, HK_G17P_VDM_ROOT_SIZE - 0x80c,
      vdm_link_length));

   const size_t minimum = HK_GFX_DRAW_RESERVATION + vdm_link_length +
                          HK_CS_SPACE_PADDING;
   assert(!hk_g17p_vdm_needs_render_split(AGX_CHIP_G17P, 1, minimum,
                                          vdm_link_length));
   assert(hk_g17p_vdm_needs_render_split(AGX_CHIP_G17P, 1, minimum - 1,
                                         vdm_link_length));
   assert(!hk_g17p_vdm_needs_render_split(AGX_CHIP_G17P, 0, minimum - 1,
                                          vdm_link_length));
   assert(!hk_g17p_vdm_needs_render_split(AGX_CHIP_G14G, 1, minimum - 1,
                                          vdm_link_length));

   /* The command uploader aligns each root to 1 KiB. G17P graphics roots are
    * therefore densely shareable instead of consuming half a 128 KiB BO.
    */
   size_t offset = 0;
   for (unsigned i = 0; i < 4; ++i) {
      offset = (offset + 1023) & ~((size_t)1023);
      assert(HK_G17P_VDM_ROOT_SIZE <= command_bo_size - offset);
      offset += HK_G17P_VDM_ROOT_SIZE;
   }
   assert(offset == 128 * 1024);
   assert(HK_G17P_VDM_ROOT_SIZE > command_bo_size - offset);

   /* vkQuake exposed the previous one-general-BO-per-alias failure only after
    * filling all 192 slots in the native 24 MiB aperture. The device-wide VDM
    * slab allocator must pack more than that many simultaneously live roots
    * without depending on command-buffer or command-pool boundaries. */
   const unsigned live_roots = 193;
   const unsigned alias_slots = (24 * 1024 * 1024) / command_bo_size;
   unsigned packed_bos = 1;
   offset = 0;
   for (unsigned i = 0; i < live_roots; ++i) {
      offset = (offset + 1023) & ~((size_t)1023);
      if (HK_G17P_VDM_ROOT_SIZE > command_bo_size - offset) {
         packed_bos++;
         offset = 0;
      }
      offset += HK_G17P_VDM_ROOT_SIZE;
   }
   assert(alias_slots == 192);
   assert(packed_bos == 49);
   assert(packed_bos < alias_slots);

   /* A failed or reset graphics initialization cannot publish a G17P VDM
    * stream, even if stale storage happens to contain a selector. Conversely,
    * successful initialization still has to establish the selector itself. */
   assert(!hk_graphics_cs_publishable(AGX_CHIP_G17P, false, false));
   assert(!hk_graphics_cs_publishable(AGX_CHIP_G17P, false, true));
   assert(!hk_graphics_cs_publishable(AGX_CHIP_G17P, true, false));
   assert(hk_graphics_cs_publishable(AGX_CHIP_G17P, true, true));
   assert(hk_graphics_cs_publishable(AGX_CHIP_G14G, true, false));

   puts("G17P graphics root sizing and command-BO sharing PASS");
   return 0;
}
