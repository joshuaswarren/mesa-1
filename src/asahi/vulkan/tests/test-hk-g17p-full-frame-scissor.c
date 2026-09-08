/* SPDX-License-Identifier: MIT */
#include "../hk_g17p_full_frame_scissor.h"
#include "../hk_g17p_ppp_indices.h"
#include "asahi/lib/agx_ppp_hal200_graphics_core.h"
#include "asahi/lib/agx_ppp_hal200_raster_core.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

static uint32_t
word(const uint8_t *p)
{
   return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
          ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static struct hk_g17p_full_frame_scissor
full_frame(unsigned width, unsigned height)
{
   return (struct hk_g17p_full_frame_scissor){
      .width = width, .height = height,
      .area_width = width, .area_height = height,
      .scissor_width = width, .scissor_height = height,
      .viewport_width = width, .viewport_height = height,
      .max_depth = 1.0f, .viewport_count = 1, .scissor_count = 1,
      .single_color_layer_sample = true, .default_depth = true,
   };
}

static void
check_sequence(void)
{
   uint8_t scissors[48], biases[36];
   const uint32_t expected[] = {0x40200, 0x50200, 0x40200};
   for (unsigned draw = 0; draw < 3; ++draw) {
      struct hk_g17p_full_frame_scissor s = full_frame(512, 512);
      if (draw == 1)
         s.scissor_width = 511;

      struct agx_ppp_hal200_raster_fields fields = {
         .scissor_enable = !hk_g17p_full_frame_scissor_can_elide(&s),
         .max_x = s.scissor_width, .max_y = s.scissor_height,
         .max_z_raw = 0x3f800000,
      };
      struct agx_ppp_hal200_raster_state state, previous;
      assert(agx_ppp_hal200_raster_state_build(AGX_CHIP_G17P, &fields, &state));
      assert(state.fragment_control == expected[draw]);
      /* Elision changes no array payload, including the Z range. */
      fields.scissor_enable = true;
      assert(agx_ppp_hal200_raster_state_build(AGX_CHIP_G17P, &fields, &previous));
      assert(!memcmp(state.scissor, previous.scissor, sizeof(state.scissor)));
      assert(!memcmp(state.depth_bias, previous.depth_bias, sizeof(state.depth_bias)));
      memcpy(scissors + draw * 16, state.scissor, 16);
      memcpy(biases + draw * 12, state.depth_bias, 12);

      struct agx_ppp_hal200_graphics_fields primary = {
         .words = {state.fragment_control, 0x07200f00, 0x0e000000,
                   0x07200f00, 0x0e000000, 0},
      };
      struct agx_ppp_hal200_graphics_core packet;
      assert(agx_ppp_hal200_graphics_core_pack(AGX_CHIP_G17P,
         AGX_PPP_HAL200_GRAPHICS_PRIMARY_FACE, &primary, &packet));
      assert(packet.length == 28 && word(packet.bytes) == 0x010000b4);
      assert(word(packet.bytes + 4) == expected[draw]);
      const struct agx_ppp_hal200_graphics_fields output = {
         .words = {0, 0, 0, 4, AGX_PPP_HAL200_PRIMARY_FACE_HEADER,
                   state.fragment_control},
      };
      struct agx_ppp_hal200_graphics_core output_packet;
      assert(agx_ppp_hal200_graphics_core_pack(AGX_CHIP_G17P,
         AGX_PPP_HAL200_GRAPHICS_OUTPUT, &output, &output_packet));
      assert(word(output_packet.bytes + 24) == word(packet.bytes + 4));
      primary.words[0] = previous.fragment_control;
      struct agx_ppp_hal200_graphics_core old_packet;
      assert(agx_ppp_hal200_graphics_core_pack(AGX_CHIP_G17P,
         AGX_PPP_HAL200_GRAPHICS_PRIMARY_FACE, &primary, &old_packet));
      for (unsigned byte = 0; byte < 28; ++byte)
         assert((packet.bytes[byte] ^ old_packet.bytes[byte]) ==
                (byte == 6 && draw != 1 ? 1 : 0));

      uint8_t indices[8];
      assert(hk_g17p_ppp_indices_pack((draw + 1) * 16,
                                     (draw + 1) * 12, indices));
      assert(word(indices) == 0x100);
      assert(word(indices + 4) == (draw | (draw << 16)));
   }
   assert(word(scissors) == 512 && word(scissors + 16) == 511);
   assert(!memcmp(scissors, scissors + 32, 16));
   const uint8_t zeros[36] = {0};
   assert(!memcmp(biases, zeros, sizeof(zeros)));
}

int
main(void)
{
   check_sequence();
   struct hk_g17p_full_frame_scissor s = full_frame(257, 129);
   assert(hk_g17p_full_frame_scissor_can_elide(&s));
   s = full_frame(16384, 16384);
   assert(hk_g17p_full_frame_scissor_can_elide(&s));
   assert(!hk_g17p_full_frame_scissor_can_elide(NULL));

   /* These all truncate or round to plausible integer bounds. They must
    * remain on the array path, without widening its current domain. */
   const float invalid_offsets[] = {0.25f, -0.25f, NAN, INFINITY};
   for (unsigned i = 0; i < 4; ++i) {
      s = full_frame(512, 512);
      s.viewport_x = invalid_offsets[i];
      assert(!hk_g17p_full_frame_scissor_can_elide(&s));
   }
#define KEEP_ARRAY(member, value) do { \
   s = full_frame(512, 512); s.member = (value); \
   assert(!hk_g17p_full_frame_scissor_can_elide(&s)); \
} while (0)
   KEEP_ARRAY(area_x, 1);
   KEEP_ARRAY(area_height, 480);
   KEEP_ARRAY(scissor_x, 32);
   KEEP_ARRAY(scissor_height, 511);
   KEEP_ARRAY(scissor_width, 513);
   KEEP_ARRAY(viewport_height, -512.0f);
   KEEP_ARRAY(viewport_width, 512.25f);
   KEEP_ARRAY(min_depth, 0.25f);
   KEEP_ARRAY(max_depth, NAN);
   KEEP_ARRAY(viewport_count, 2);
   KEEP_ARRAY(scissor_count, 0);
   KEEP_ARRAY(single_color_layer_sample, false);
   KEEP_ARRAY(default_depth, false);
   KEEP_ARRAY(width, 0);
   KEEP_ARRAY(height, 16385);
#undef KEEP_ARRAY

   assert(hk_g17p_viewport_state_supported(0, 0));
   assert(hk_g17p_viewport_state_supported(1, 0));
   assert(!hk_g17p_viewport_state_supported(0, 1));
   assert(hk_g17p_viewport_state_supported(1, 1));
   assert(hk_g17p_viewport_state_supported(2, 1));
   assert(!hk_g17p_viewport_state_supported(1, 2));
   assert(hk_g17p_viewport_state_supported(2, 2));
   assert(hk_g17p_viewport_state_supported(16, 16));
   assert(!hk_g17p_viewport_state_supported(17, 16));
   assert(!hk_g17p_viewport_state_supported(UINT32_MAX, 1));
   assert(!hk_g17p_viewport_state_supported(1, UINT32_MAX));

   puts("G17P full-frame scissor: full/crop/full packets, retained arrays, "
        "indices and the 16-viewport bound pass");
   return 0;
}
