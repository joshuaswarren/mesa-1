/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_vdm_hal200_bind_tail_core.h"

#include <assert.h>
#include <string.h>

int
main(void)
{
   static const struct agx_vdm_hal200_bind_pair pairs[] = {
      {0x00000040, 0x700}, {0x00058000, 0x500},
      {0x0005801c, 0x700}, {0x00058030, 0x500},
      {0x0005804c, 0xa00}, {0x00068900, 0x300},
      {0x00058060, 0x200}, {0x0005806c, 0x200},
   };
   static const uint32_t expected_words[] = {
      0x00000500,
      0x00000040, 0x00000700,
      0x00058000, 0x00000500,
      0x0005801c, 0x00000700,
      0x00058030, 0x00000500,
      0x0005804c, 0x00000a00,
      0x00068900, 0x00000300,
      0x00058060, 0x00000200,
      0x0005806c, 0x00000200,
   };

   struct agx_vdm_hal200_bind_tail packed;
   assert(agx_vdm_hal200_bind_tail_pack(AGX_CHIP_G17P, pairs, &packed));
   assert(packed.length == sizeof(expected_words));
   assert(memcmp(packed.bytes, expected_words, sizeof(expected_words)) == 0);

   struct agx_vdm_hal200_bind_pair unpacked[AGX_VDM_HAL200_BIND_COUNT];
   assert(agx_vdm_hal200_bind_tail_unpack(
      AGX_CHIP_G17P, packed.bytes, packed.length, unpacked));
   assert(memcmp(unpacked, pairs, sizeof(pairs)) == 0);

   /* Context offsets are full unsigned words, including the high bit. */
   struct agx_vdm_hal200_bind_pair wide[AGX_VDM_HAL200_BIND_COUNT];
   memcpy(wide, pairs, sizeof(wide));
   for (unsigned i = 0; i < AGX_VDM_HAL200_BIND_COUNT; ++i)
      wide[i].context_offset = UINT32_C(0xffffffe0) + 4 * i;
   assert(agx_vdm_hal200_bind_tail_pack(AGX_CHIP_G17P, wide, &packed));
   assert(agx_vdm_hal200_bind_tail_unpack(
      AGX_CHIP_G17P, packed.bytes, packed.length, unpacked));
   assert(memcmp(unpacked, wide, sizeof(wide)) == 0);

   assert(!agx_vdm_hal200_bind_tail_pack(AGX_CHIP_G17G, pairs, &packed));

   struct agx_vdm_hal200_bind_pair bad[AGX_VDM_HAL200_BIND_COUNT];
   memcpy(bad, pairs, sizeof(bad));
   bad[3].control = 0x700;
   assert(!agx_vdm_hal200_bind_tail_pack(AGX_CHIP_G17P, bad, &packed));
   memcpy(bad, pairs, sizeof(bad));
   bad[0].context_offset = 2;
   assert(!agx_vdm_hal200_bind_tail_pack(AGX_CHIP_G17P, bad, &packed));

   /* Pair 4 is the size of the viewport payload, 2 + 8 * count words. */
   assert(agx_vdm_hal200_bind_viewport_control(40) == 0xa00);
   assert(agx_vdm_hal200_bind_viewport_control(72) == 0x1200);
   assert(agx_vdm_hal200_bind_viewport_control(520) == 0x8200);
   assert(agx_vdm_hal200_bind_viewport_control(0) == 0);
   assert(agx_vdm_hal200_bind_viewport_control(8) == 0);
   assert(agx_vdm_hal200_bind_viewport_control(56) == 0);
   assert(agx_vdm_hal200_bind_viewport_control(552) == 0);

   struct agx_vdm_hal200_bind_pair multi[AGX_VDM_HAL200_BIND_COUNT];
   for (unsigned count = 1; count <= 16; ++count) {
      memcpy(multi, pairs, sizeof(multi));
      multi[4].control = (2 + 8 * count) << 8;
      multi[0].context_offset = 0x00068900 + 8 + 32 * count;
      assert(agx_vdm_hal200_bind_tail_pack(AGX_CHIP_G17P, multi, &packed));
      assert(agx_vdm_hal200_bind_tail_unpack(
         AGX_CHIP_G17P, packed.bytes, packed.length, unpacked));
      assert(memcmp(unpacked, multi, sizeof(multi)) == 0);
   }

   static const uint32_t bad_viewport_controls[] = {
      0x200, 0x900, 0xa01, 0xb00, 0x1100, 0x8a00, 0x10a00,
   };
   for (unsigned i = 0; i < sizeof(bad_viewport_controls) /
                                sizeof(bad_viewport_controls[0]); ++i) {
      memcpy(bad, pairs, sizeof(bad));
      bad[4].control = bad_viewport_controls[i];
      assert(!agx_vdm_hal200_bind_tail_pack(AGX_CHIP_G17P, bad, &packed));
   }

   /* A viewport size is only accepted in the viewport's own control. */
   for (unsigned i = 0; i < AGX_VDM_HAL200_BIND_COUNT; ++i) {
      if (i == AGX_VDM_HAL200_BIND_VIEWPORT_CONTROL_PAIR)
         continue;
      memcpy(bad, pairs, sizeof(bad));
      bad[i].control = 0x1200;
      assert(!agx_vdm_hal200_bind_tail_pack(AGX_CHIP_G17P, bad, &packed));
      assert(!agx_vdm_hal200_bind_control_valid(i, 0x1200));
   }
   assert(!agx_vdm_hal200_bind_control_valid(AGX_VDM_HAL200_BIND_COUNT,
                                             0x200));
   return 0;
}
