/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "../agx_ppp_hal200_graphics_core.h"
#include "../agx_ppp_viewport_payload_core.h"
#include "../agx_ppp_w_clamp_cull2_core.h"
#include "../agx_g17p_draw_state_upload.h"
#include "../agx_va_runtime_hole.h"
#include "../agx_vdm_hal200_bind_tail_core.h"
#include "../agx_vdm_index_list_core.h"
#include "../agx_vdm_state_update_core.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define VDM_LENGTH 152u
#define BIND_STATE_LENGTH 188u

static const uint8_t gl03_vdm[VDM_LENGTH] = {
   0x60, 0x09, 0x00, 0x21,
   0x00, 0x03, 0x00, 0x00, 0x10, 0x00, 0x88, 0x00,
   0x2e, 0x00, 0x00, 0x40, 0x12, 0x12, 0x00, 0x00,
   0x00, 0x00, 0x00, 0x01, 0x01, 0xda, 0xff, 0x03,
   0x07, 0x07, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
   0x00, 0x00, 0x00, 0x00,
   0x00, 0x05, 0x00, 0x00,
   0x28, 0x0c, 0x88, 0x00, 0x00, 0x07, 0x00, 0x00,
   0x80, 0x0b, 0x88, 0x00, 0x00, 0x05, 0x00, 0x00,
   0x9c, 0x0b, 0x88, 0x00, 0x00, 0x07, 0x00, 0x00,
   0xb0, 0x0b, 0x88, 0x00, 0x00, 0x05, 0x00, 0x00,
   0xcc, 0x0b, 0x88, 0x00, 0x00, 0x0a, 0x00, 0x00,
   0x00, 0x0c, 0x88, 0x00, 0x00, 0x03, 0x00, 0x00,
   0xe0, 0x0b, 0x88, 0x00, 0x00, 0x02, 0x00, 0x00,
   0xec, 0x0b, 0x88, 0x00, 0x00, 0x02, 0x00, 0x00,
   0x3c, 0x0c, 0x88, 0x00,
   0x01, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00,
   0x5f, 0x06, 0xf2, 0x61, 0x00, 0x40, 0xf3, 0xff,
   0x06, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
   0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
   0x00, 0x00, 0x00, 0x00,
   0x00, 0x00, 0x00, 0xc0,
};

static const uint8_t gl03_initial_ppp[12] = {
   0x00, 0x00, 0x41, 0x00, 0x00, 0x00, 0x00, 0x00,
   0x80, 0x00, 0x00, 0x00,
};

static const uint8_t gl03_bind_state[BIND_STATE_LENGTH] = {
   0x00, 0x00, 0x80, 0x00, 0x00, 0x01, 0x02, 0x00,
   0x00, 0x80, 0xf6, 0xff, 0x00, 0x00, 0x00, 0x00,
   0x00, 0x00, 0x00, 0x01, 0x03, 0xda, 0xff, 0x03,
   0x00, 0x00, 0x00, 0x00,
   0x00, 0x00, 0x0a, 0x0c, 0x00, 0x00, 0x01, 0x00,
   0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   0x07, 0x00, 0x00, 0x00,
   0xb4, 0x00, 0x00, 0x01, 0x00, 0x02, 0x05, 0x00,
   0x00, 0x0f, 0x20, 0x07, 0x00, 0x00, 0x00, 0x0e,
   0x00, 0x0f, 0x20, 0x07, 0x00, 0x00, 0x00, 0x0e,
   0x00, 0x00, 0x00, 0x00,
   0x48, 0x00, 0x00, 0x02, 0x00, 0x02, 0x00, 0x00,
   0x00, 0x00, 0xe0, 0x07, 0x00, 0x00, 0xe0, 0x07,
   0xff, 0xff, 0x01, 0x00,
   0x00, 0x00, 0x41, 0x00, 0x00, 0x00, 0x00, 0x00,
   0x80, 0x00, 0x00, 0x00,
   0x00, 0x00, 0x20, 0x00, 0x80, 0x05, 0x00, 0x00,
   0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   0x00, 0x00, 0x00, 0x00,
   0x00, 0x0c, 0x00, 0x00, 0x01, 0x00, 0x00, 0x80,
   0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   0x00, 0x00, 0x00, 0x42, 0x00, 0x00, 0x00, 0x42,
   0x00, 0x00, 0x00, 0x42, 0x00, 0x00, 0x00, 0x42,
   0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f,
   0x00, 0x00, 0x04, 0x10, 0x02, 0x00, 0x00, 0x00,
   0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   0x00, 0x00, 0x00, 0x00,
};

static const uint8_t gl03_draw_indices[8] = {
   0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
};

static uint32_t
load_le32(const uint8_t *p)
{
   return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
          ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static void
store_le32(uint8_t *p, uint32_t value)
{
   p[0] = value;
   p[1] = value >> 8;
   p[2] = value >> 16;
   p[3] = value >> 24;
}

static bool
owned(uint64_t address, size_t size, uint64_t base, size_t length)
{
   return address >= base && size <= length &&
          address - base <= length - size;
}

static uint64_t
decode_pipeline_pair(const uint8_t pair[8])
{
   const uint32_t hi = load_le32(pair);
   const uint32_t lo = load_le32(pair + 4);
   assert((hi & UINT32_C(0x003fffff)) == 0);
   return ((uint64_t)(hi >> 22) << 38) | ((uint64_t)lo << 6);
}

static bool
audit_bind_state(const uint8_t *bytes, size_t size)
{
   if (!bytes || size != BIND_STATE_LENGTH)
      return false;

   /* Eight independently addressed native slots, with one fixed padding gap. */
   assert(load_le32(bytes + 0x00) == AGX_PPP_HAL200_FS_HEADER);
   assert(load_le32(bytes + 0x1c) == AGX_PPP_HAL200_OUTPUT_HEADER);
   assert(load_le32(bytes + 0x30) == AGX_PPP_HAL200_PRIMARY_FACE_HEADER);
   assert(load_le32(bytes + 0x4c) == AGX_PPP_HAL200_SECONDARY_FACE_HEADER);
   assert(load_le32(bytes + 0x60) == UINT32_C(0x00410000));
   assert(load_le32(bytes + 0x6c) == AGX_PPP_HAL200_CULL_HEADER);
   for (unsigned i = 0x74; i < 0x80; ++i)
      assert(bytes[i] == 0);

   struct agx_ppp_hal200_graphics_fields graphics;
   assert(agx_ppp_hal200_graphics_core_unpack(
      AGX_CHIP_G17P, AGX_PPP_HAL200_GRAPHICS_FS, bytes + 0x00, 28,
      &graphics));
   assert(agx_ppp_hal200_graphics_core_unpack(
      AGX_CHIP_G17P, AGX_PPP_HAL200_GRAPHICS_OUTPUT, bytes + 0x1c, 28,
      &graphics));
   assert(graphics.words[0] == UINT32_C(0x00010000));
   assert(graphics.words[3] == 7);
   assert(agx_ppp_hal200_graphics_core_unpack(
      AGX_CHIP_G17P, AGX_PPP_HAL200_GRAPHICS_PRIMARY_FACE, bytes + 0x30, 28,
      &graphics));
   assert(agx_ppp_hal200_graphics_core_unpack(
      AGX_CHIP_G17P, AGX_PPP_HAL200_GRAPHICS_SECONDARY_FACE, bytes + 0x4c,
      20, &graphics));
   assert(agx_ppp_hal200_graphics_core_unpack(
      AGX_CHIP_G17P, AGX_PPP_HAL200_GRAPHICS_CULL, bytes + 0x6c, 8,
      &graphics));

   const uint64_t usc_base = UINT64_C(0x100fff68000);
   const size_t usc_size = 0x1000;
   const uint64_t vit = ((uint64_t)(load_le32(bytes + 4) & 0xffff) << 32) |
                        load_le32(bytes + 8);
   const uint64_t fragment = decode_pipeline_pair(bytes + 16);
   assert(vit == UINT64_C(0x100fff68000));
   assert(fragment == UINT64_C(0x100fff680c0));
   assert(owned(vit, 64, usc_base, usc_size));
   assert(owned(fragment, 64, usc_base, usc_size));

   struct agx_ppp_w_clamp_cull2_fields w_cull2;
   assert(agx_ppp_w_clamp_cull2_core_unpack(
      AGX_CHIP_G17P, bytes + 0x60, 12, &w_cull2));
   assert(w_cull2.w_clamp_raw == 0 && w_cull2.cull2_raw == 0x80);

   struct agx_ppp_viewport_payload_fields viewport;
   assert(agx_ppp_viewport_payload_core_unpack(
      AGX_CHIP_G17P, bytes + 0x80, 40, &viewport));
   assert(viewport.viewport_count == 1);
   assert(viewport.region_clip_raw[0] == UINT64_C(0x0000000180000001));
   assert(viewport.viewports[0].translate_x == UINT32_C(0x42000000));
   assert(viewport.viewports[0].scale_x == UINT32_C(0x42000000));
   assert(viewport.viewports[0].translate_y == UINT32_C(0x42000000));
   assert(viewport.viewports[0].scale_y == UINT32_C(0x42000000));
   assert(viewport.viewports[0].translate_z == 0);
   assert(viewport.viewports[0].scale_z == UINT32_C(0x3f800000));

   assert(load_le32(bytes + 0xa8) == UINT32_C(0x10040000));
   assert(load_le32(bytes + 0xac) == 2);
   assert(load_le32(bytes + 0xb0) == 0);
   assert(load_le32(bytes + 0xb4) == 0);
   assert(load_le32(bytes + 0xb8) == 0);
   return true;
}

static bool
audit_vdm(const uint8_t *bytes, size_t size)
{
   if (!bytes || size != VDM_LENGTH)
      return false;

   assert(agx_g17p_vdm_alias_range_valid(UINT64_C(0x1000800000),
                                          UINT64_C(0x80000)));
   size_t cursor = 0;
   assert(cursor == 0); /* Barrier: header 0x21000960, 4 bytes. */
   assert(load_le32(bytes + cursor) == UINT32_C(0x21000960));
   cursor += 4;

   assert(cursor == 4); /* Initial PPP pointer: header + address, 8 bytes. */
   assert(load_le32(bytes + cursor) == UINT32_C(0x00000300));
   assert(load_le32(bytes + cursor + 4) == UINT32_C(0x00880010));
   assert(owned(UINT64_C(0x880010), 12, UINT64_C(0x880000), 0x10000));
   assert(agx_g17p_vdm_alias_range_valid(UINT64_C(0x1000880010), 12));
   cursor += 8;

   assert(cursor == 12); /* VertexProgramVariant state: 28 bytes. */
   struct agx_vdm_state_update_info state;
   assert(agx_vdm_state_update_unpack(AGX_CHIP_G17P, bytes + cursor, 28,
                                       &state));
   assert(state.raw_vertex_shader_word0 == UINT32_C(0x1212));
   assert(state.pipeline == UINT64_C(0x100fff68040));
   assert(owned(state.pipeline, 64, UINT64_C(0x100fff68000), 0x1000));
   assert(state.raw_vertex_outputs[0] == UINT32_C(0x707));
   assert(state.raw_vertex_outputs[1] == 2);
   assert(state.raw_vertex_unknown == 0);
   cursor += 28;

   assert(cursor == 40); /* Native HAL200 bind tail: 68 bytes. */
   assert(load_le32(bytes + cursor) == UINT32_C(0x00000500));
   const uint32_t expected_offsets[8] = {
      0x00880c28, 0x00880b80, 0x00880b9c, 0x00880bb0,
      0x00880bcc, 0x00880c00, 0x00880be0, 0x00880bec,
   };
   const uint32_t expected_controls[8] = {
      0x700, 0x500, 0x700, 0x500, 0xa00, 0x300, 0x200, 0x200,
   };
   const uint8_t expected_lengths[8] = {20, 28, 28, 28, 20, 40, 12, 8};
   struct agx_vdm_hal200_bind_pair decoded_pairs[8];
   assert(agx_vdm_hal200_bind_tail_unpack(
      AGX_CHIP_G17P, bytes + cursor, 68, decoded_pairs));
   for (unsigned i = 0; i < 8; ++i) {
      const uint32_t offset = load_le32(bytes + cursor + 4 + 8 * i);
      const uint32_t control = load_le32(bytes + cursor + 8 + 8 * i);
      assert(offset == expected_offsets[i]);
      assert(control == expected_controls[i]);
      assert(decoded_pairs[i].context_offset == offset);
      assert(decoded_pairs[i].control == control);
      assert(owned(offset, expected_lengths[i], UINT64_C(0x880b80),
                   BIND_STATE_LENGTH));
      assert(agx_g17p_vdm_alias_range_valid(
         AGX_G17P_VDM_USER_BASE + offset, expected_lengths[i]));
   }
   cursor += 68;

   assert(cursor == 108); /* Draw-state selector pointer: 4 bytes. */
   assert(load_le32(bytes + cursor) == UINT32_C(0x00880c3c));
   assert(owned(load_le32(bytes + cursor), sizeof(gl03_draw_indices),
                UINT64_C(0x880c3c), sizeof(gl03_draw_indices)));
   assert(load_le32(gl03_draw_indices) == UINT32_C(0x00000100));
   assert(load_le32(gl03_draw_indices + 4) == 0);
   cursor += 4;

   assert(cursor == 112); /* Indexed restart state: 8 bytes. */
   assert(load_le32(bytes + cursor) == UINT32_C(0x40000001));
   assert(load_le32(bytes + cursor + 4) == 0);
   cursor += 8;

   assert(cursor == 120); /* Indexed Index List: 28 bytes. */
   struct agx_vdm_index_list_direct_indexed_info draw;
   assert(agx_vdm_index_list_direct_indexed_unpack(
      AGX_CHIP_G17P, bytes + cursor, 28, &draw));
   assert(draw.index_buffer == UINT64_C(0x5ffff34000));
   assert(draw.count == 6 && draw.instances == 1 && draw.start == 0);
   assert(draw.index_buffer_size_word == 2);
   assert(owned(draw.index_buffer, 12, UINT64_C(0x5ffff34000), 12));
   cursor += 28;

   assert(cursor == 148); /* Stream terminate: 4 bytes. */
   assert(load_le32(bytes + cursor) == UINT32_C(0xc0000000));
   cursor += 4;
   assert(cursor == size);
   return true;
}

static void
check_legacy_rejections(void)
{
   uint8_t vertex24[24];
   memcpy(vertex24, gl03_vdm + 12, sizeof(vertex24));
   struct agx_vdm_state_update_info state;
   assert(!agx_vdm_state_update_unpack(AGX_CHIP_G17P, vertex24,
                                        sizeof(vertex24), &state));

   static const uint8_t viewport44[44] = {
      0x00, 0x0d, 0x00, 0x00, 0, 0, 0, 0,
      0x01, 0x00, 0x00, 0x80, 0x01, 0, 0, 0,
   };
   struct agx_ppp_viewport_payload_fields viewport;
   assert(!agx_ppp_viewport_payload_core_unpack(
      AGX_CHIP_G17P, viewport44, sizeof(viewport44), &viewport));

   static const uint8_t legacy_bind80[80] = {
      0xff, 0x00, 0xee, 0x09, 0x00, 0x02, 0x01, 0x00,
      0x00, 0x02, 0x00, 0x04, 0x00, 0x0f, 0x20, 0x07,
      0x00, 0x00, 0xe0, 0x07, 0xff, 0xff, 0x00, 0x0e,
      0x00, 0x0f, 0x20, 0x07, 0x00, 0x00, 0xe0, 0x07,
      0xff, 0xff, 0x00, 0x0e, 0x00, 0x05, 0x00, 0x02,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x05,
      0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x00, 0x12,
      0x12, 0x12, 0x02, 0x00, 0xc0, 0x80, 0xf6, 0xff,
      0x00, 0x80, 0xf6, 0xff, 0, 0, 0, 0,
      0, 0, 0, 0, 0x07, 0, 0, 0,
   };
   assert(load_le32(legacy_bind80) != AGX_PPP_HAL200_FS_HEADER);
   assert(load_le32(legacy_bind80) != AGX_PPP_HAL200_OUTPUT_HEADER);
   assert(load_le32(legacy_bind80) != AGX_PPP_HAL200_PRIMARY_FACE_HEADER);
   assert(load_le32(legacy_bind80) != AGX_PPP_HAL200_SECONDARY_FACE_HEADER);
   assert(load_le32(legacy_bind80) != AGX_PPP_HAL200_CULL_HEADER);
   assert(load_le32(legacy_bind80 + 60) == UINT32_C(0xfff680c0));
   assert(!owned(load_le32(legacy_bind80 + 60), 64,
                 UINT64_C(0x100fff68000), 0x1000));

   uint8_t vdm_trailing[VDM_LENGTH + 1];
   memcpy(vdm_trailing, gl03_vdm, VDM_LENGTH);
   vdm_trailing[VDM_LENGTH] = 0;
   assert(!audit_vdm(vdm_trailing, sizeof(vdm_trailing)));

   uint8_t bind_trailing[BIND_STATE_LENGTH + 1];
   memcpy(bind_trailing, gl03_bind_state, BIND_STATE_LENGTH);
   bind_trailing[BIND_STATE_LENGTH] = 0;
   assert(!audit_bind_state(bind_trailing, sizeof(bind_trailing)));
}

static void
check_multi_viewport_state(void)
{
   struct agx_ppp_viewport_payload_fields fields = {
      .viewport_count = AGX_PPP_VIEWPORT_PAYLOAD_MAX_VIEWPORTS,
   };
   for (unsigned i = 0; i < fields.viewport_count; ++i) {
      fields.region_clip_raw[i] = UINT64_C(0x8000000100000001) + i;
      fields.viewports[i] = (struct agx_ppp_viewport_transform_raw){
         .translate_x = UINT32_C(0x41000000) + i,
         .scale_x = UINT32_C(0x40000000) + i,
         .translate_y = UINT32_C(0x42000000) + i,
         .scale_y = UINT32_C(0x3f800000) + i,
         .translate_z = UINT32_C(0x3e800000) + i,
         .scale_z = UINT32_C(0x3f000000) + i,
      };
   }

   struct agx_ppp_viewport_payload_core viewport;
   assert(agx_ppp_viewport_payload_core_pack(AGX_CHIP_G17P, &fields,
                                              &viewport));
   assert(viewport.length == AGX_G17P_VIEWPORT_RECORD_MAX_BYTES);

   uint8_t bind_group[AGX_G17P_BIND_GROUP_BYTES];
   memcpy(bind_group, gl03_bind_state, sizeof(bind_group));
   uint32_t count_word;
   assert(agx_ppp_hal200_viewport_count_pack(
      AGX_CHIP_G17P, fields.viewport_count, &count_word));
   store_le32(bind_group + 0x28, count_word);

   uint8_t state[AGX_G17P_BIND_STATE_MAX_BYTES + 1];
   memset(state, 0xa5, sizeof(state));
   struct agx_g17p_bind_state_upload upload;
   const uint64_t compact = UINT64_C(0x00880b80);
   const size_t state_size = agx_g17p_bind_state_size(viewport.length);
   assert(state_size == AGX_G17P_BIND_STATE_MAX_BYTES);
   assert(agx_g17p_upload_bind_state(
      state, state_size, compact, bind_group, viewport.bytes, viewport.length,
      2, 0, &upload));
   assert(state[state_size] == 0xa5);
   assert(upload.pairs[5].context_offset ==
          compact + AGX_G17P_BIND_GROUP_BYTES);
   assert(upload.pairs[0].context_offset ==
          compact + AGX_G17P_BIND_GROUP_BYTES + viewport.length);
   assert(upload.pairs[0].control == 0x700 &&
          upload.pairs[5].control == 0x300);
   assert(load_le32(state + 0x28) == 15);

   struct agx_ppp_viewport_payload_fields decoded;
   assert(agx_ppp_viewport_payload_core_unpack(
      AGX_CHIP_G17P, state + AGX_G17P_BIND_GROUP_BYTES, viewport.length,
      &decoded));
   assert(decoded.viewport_count == 16);
   /* A shader selecting viewport 1 must land on a distinct complete record. */
   assert(decoded.region_clip_raw[1] == fields.region_clip_raw[1]);
   assert(!memcmp(&decoded.viewports[1], &fields.viewports[1],
                  sizeof(fields.viewports[1])));

   /* Both selector composers emit the same fixed leading word, and the
    * validator compares against that exported constant rather than a retyped
    * copy of it. Pin the derivation from this side too: if the encoder ever
    * changes the word, this fails here instead of the oracle reporting a
    * correct driver as a corrupt command stream. */
   uint8_t indices[8];
   assert(agx_g17p_ppp_indices_base_pack(19 * 16, 21 * 12, 16, indices));
   assert(load_le32(indices) == AGX_G17P_DRAW_STATE_SELECTOR_WORD0);
   assert(load_le32(indices + 4) == UINT32_C(0x00050003));

   uint8_t single[8];
   assert(agx_g17p_ppp_indices_pack(19 * 16, 21 * 12, single));
   assert(load_le32(single) == AGX_G17P_DRAW_STATE_SELECTOR_WORD0);
   assert(load_le32(single + 4) == UINT32_C(0x00140012));

   /* The word is exact, not merely "bit 8 set": nearby values are not what the
    * encoder emits, so an accept-set derived from it still refuses them. */
   assert(AGX_G17P_DRAW_STATE_SELECTOR_WORD0 != UINT32_C(0x00000000));
   assert(AGX_G17P_DRAW_STATE_SELECTOR_WORD0 != UINT32_C(0x00000101));
   assert(AGX_G17P_DRAW_STATE_SELECTOR_WORD0 != UINT32_C(0x00000300));
   assert(AGX_G17P_DRAW_STATE_SELECTOR_WORD0 != UINT32_C(0x01000100));

   /* The varying record's fixed word is exported by the same header and is
    * what the 20-byte varyings slot is validated against. */
   assert(load_le32((const uint8_t *)state + AGX_G17P_BIND_GROUP_BYTES +
                    viewport.length) == AGX_G17P_VARYING_RECORD_WORD0);

   /* Every size/count boundary fails closed before touching live streams. */
   assert(!agx_g17p_upload_bind_state(
      state, state_size - 1, compact, bind_group, viewport.bytes,
      viewport.length, 2, 0, &upload));
   assert(!agx_g17p_upload_bind_state(
      state, state_size, compact, bind_group, viewport.bytes,
      viewport.length - 1, 2, 0, &upload));
   assert(!agx_g17p_upload_bind_state(
      state, state_size, AGX_G17P_VDM_ALIAS_OFFSET_END - state_size + 4,
      bind_group, viewport.bytes, viewport.length, 2, 0, &upload));
   assert(!agx_g17p_ppp_indices_base_pack(16 * 16, 16 * 12, 0, indices));
   assert(!agx_g17p_ppp_indices_base_pack(16 * 16, 16 * 12, 17, indices));
   assert(!agx_g17p_ppp_indices_base_pack(15 * 16, 16 * 12, 16, indices));
   assert(!agx_g17p_ppp_indices_base_pack(16 * 16, 15 * 12, 16, indices));
}

int
main(void)
{
   static_assert(sizeof(gl03_vdm) == VDM_LENGTH, "exact VDM length");
   static_assert(sizeof(gl03_bind_state) == BIND_STATE_LENGTH,
                 "exact bind-state length");

   assert(audit_vdm(gl03_vdm, sizeof(gl03_vdm)));
   assert(audit_bind_state(gl03_bind_state, sizeof(gl03_bind_state)));

   struct agx_ppp_w_clamp_cull2_fields initial;
   assert(agx_ppp_w_clamp_cull2_core_unpack(
      AGX_CHIP_G17P, gl03_initial_ppp, sizeof(gl03_initial_ppp), &initial));
   assert(initial.w_clamp_raw == 0 && initial.cull2_raw == 0x80);
   check_legacy_rejections();
   check_multi_viewport_state();

   puts("G17P Gallium GL03 stream and 16-viewport bind state audit passed");
   return 0;
}
