/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "../agx_va_runtime_hole.h"
#include "../agx_vdm_barrier_core.h"
#include "../agx_vdm_index_list_core.h"
#include "../agx_vdm_ppp_state_core.h"
#include "../agx_vdm_state_update_core.h"
#include "../agx_vdm_stream_terminate_core.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static const uint8_t gl03_vdm[] = {
   0x60, 0x09, 0x00, 0x21,
   0x00, 0x03, 0x00, 0x00, 0x10, 0x00, 0x88, 0x00,
   0x2e, 0x00, 0x00, 0x40, 0x12, 0x12, 0x00, 0x00,
   0x00, 0x00, 0x00, 0x01, 0x01, 0xda, 0xff, 0x03,
   0x07, 0x07, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
   0x00, 0x00, 0x00, 0x00,
   0x00, 0x0b, 0x00, 0x00, 0x58, 0x0b, 0x88, 0x00,
   0x00, 0x14, 0x00, 0x00, 0x84, 0x0b, 0x88, 0x00,
   0x01, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00,
   0x5f, 0x06, 0xf2, 0x61, 0x00, 0x40, 0xf3, 0xff,
   0x06, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
   0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
   0x00, 0x00, 0x00, 0x00,
   0x00, 0x00, 0x00, 0xc0,
};

static void
check_ppp_pointer(size_t offset, uint64_t expected, uint32_t words)
{
   uint64_t pointer = 0;
   uint32_t size_words = 0;
   assert(agx_vdm_ppp_state_core_unpack(
      AGX_CHIP_G17P, &gl03_vdm[offset], AGX_VDM_PPP_STATE_CORE_LENGTH,
      &pointer, &size_words));
   assert(pointer == expected);
   assert(size_words == words);

   const uint64_t alias = AGX_G17P_VDM_USER_BASE + pointer;
   assert(agx_g17p_vdm_alias_range_valid(alias, (uint64_t)words * 4));
}

static void
check_complete_stream(void)
{
   static_assert(sizeof(gl03_vdm) == 96, "complete GL03 VDM stream");
   const uint64_t root = UINT64_C(0x1000800000);
   assert(agx_g17p_vdm_alias_range_valid(root, UINT64_C(0x80000)));

   assert(agx_vdm_barrier_core_unpack(
      AGX_CHIP_G17P, gl03_vdm, AGX_VDM_BARRIER_CORE_LENGTH));
   check_ppp_pointer(4, UINT64_C(0x880010), 3);

   struct agx_vdm_state_update_info state;
   assert(agx_vdm_state_update_unpack(
      AGX_CHIP_G17P, &gl03_vdm[12], AGX_VDM_STATE_UPDATE_LENGTH, &state));
   assert(state.raw_vertex_shader_word0 == UINT32_C(0x1212));
   assert(state.pipeline == UINT64_C(0x100fff68040));
   assert(state.raw_pipeline_template == 0);
   assert(state.raw_vertex_outputs[0] == UINT32_C(0x707));
   assert(state.raw_vertex_outputs[1] == 2);
   assert(state.raw_vertex_unknown == 0);
   assert(state.pipeline >= UINT64_C(0x10000000000));
   assert(state.pipeline < UINT64_C(0x10100000000));

   check_ppp_pointer(40, UINT64_C(0x880b58), 11);
   check_ppp_pointer(48, UINT64_C(0x880b84), 20);

   uint32_t restart[2];
   memcpy(restart, &gl03_vdm[56], sizeof(restart));
   assert(restart[0] == UINT32_C(0x40000001));
   assert(restart[1] == 0);

   struct agx_vdm_index_list_direct_indexed_info draw;
   assert(agx_vdm_index_list_direct_indexed_unpack(
      AGX_CHIP_G17P, &gl03_vdm[64], 28, &draw));
   assert(draw.index_buffer == UINT64_C(0x5ffff34000));
   assert(draw.count == 6 && draw.instances == 1 && draw.start == 0);
   assert(draw.index_buffer_size_word == 2);
   assert(draw.index_buffer >= UINT64_C(0x4000000000));
   assert(draw.index_buffer < UINT64_C(0x6000000000));

   assert(agx_vdm_stream_terminate_core_unpack(
      AGX_CHIP_G17P, &gl03_vdm[92], AGX_VDM_STREAM_TERMINATE_CORE_LENGTH));
}

static void
check_legacy_phase_regression(void)
{
   static const uint8_t malformed[AGX_VDM_STATE_UPDATE_LENGTH] = {
      0x2e, 0x00, 0x00, 0x40, 0x12, 0x12, 0x00, 0x00,
      0x40, 0x80, 0xf6, 0xff, 0x07, 0x07, 0x00, 0x00,
      0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x0b, 0x00, 0x00,
   };
   struct agx_vdm_state_update_info decoded;
   assert(agx_vdm_state_update_unpack(
      AGX_CHIP_G17P, malformed, sizeof(malformed), &decoded));

   /* The old 24-byte record consumed the following PPP header as word six.
    * Its one-word pipeline field also consumed UVS counts as address bits. */
   assert(decoded.pipeline == UINT64_C(0xffc00001c1c0));
   assert(decoded.raw_vertex_unknown == UINT32_C(0xb00));
   assert(decoded.pipeline < UINT64_C(0x10000000000) ||
          decoded.pipeline >= UINT64_C(0x10100000000));
}

int
main(void)
{
   check_complete_stream();
   check_legacy_phase_regression();
   puts("G17P Gallium GL03 VDM stream and all address fields decode exactly");
   return 0;
}
