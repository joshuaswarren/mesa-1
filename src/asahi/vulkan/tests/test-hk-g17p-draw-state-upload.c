/* SPDX-License-Identifier: MIT */
#include "../hk_g17p_draw_state_upload.h"
#include "asahi/lib/agx_g17p_bind0.h"
#include "asahi/lib/agx_vdm_hal200_direct_draw_core.h"
#include <assert.h>
#include <stdio.h>

static void
test_bind0(void)
{
   uint8_t actual[0x440], old[0x400] = {0};
   memset(actual, 0xa5, sizeof(actual));
   /* Independent copy of the former runtime writer, kept byte-for-byte. */
   for (unsigned i = 0; i < 7; ++i) {
      const uint32_t head = 0x80, body = 0x10040000;
      memcpy(old + i * 0x80, &head, 4);
      memcpy(old + i * 0x80 + 0x40, &body, 4);
   }
   const uint32_t direct = 0x0000fcc0;
   memcpy(old + 0x300, &direct, 4);
   memset(old + 0x340, 0, 4);
   assert(agx_g17p_bind0_initialize(actual, sizeof(actual)));
   assert(!memcmp(actual, old, sizeof(old)));
   for (unsigned i = 0x400; i < sizeof(actual); ++i)
      assert(actual[i] == 0xa5);
   memset(actual, 0xa5, sizeof(actual));
   assert(!agx_g17p_bind0_initialize(actual, 0x3ff));
   for (unsigned i = 0; i < sizeof(actual); ++i)
      assert(actual[i] == 0xa5);
}

int main(void)
{
   test_bind0();
   /* Distinct allocations within CB A, followed by an independently recorded
    * CB B. Resolve the emitted real HAL200 fields against retained storage;
    * changing inputs or recording B must never change either earlier record.
    */
   uint8_t retained[3][256], saved[3][256], group[128], viewport[40];
   struct hk_g17p_bind_state_upload uploads[3];
   const uint32_t counts[] = {2, 0x20000, 0};
   const uint32_t bases[] = {0x800000, 0x800100, 0x820000};
   for (unsigned draw = 0; draw < 3; ++draw) {
      memset(retained[draw], 0xa5, sizeof(retained[draw]));
      memset(group, 0x10 + draw, sizeof(group));
      memset(viewport, 0x80 + draw, sizeof(viewport));
      assert(hk_g17p_upload_bind_state(retained[draw], sizeof(retained[draw]),
                bases[draw], group, viewport, sizeof(viewport), counts[draw], 0, &uploads[draw]));
      struct agx_vdm_hal200_bind_pair decoded[8];
      assert(agx_vdm_hal200_bind_tail_unpack(AGX_CHIP_G17P,
             uploads[draw].tail.bytes, uploads[draw].tail.length, decoded));
      assert(decoded[0].context_offset == bases[draw] + HK_G17P_VARYING_RECORD_OFFSET);
      assert(decoded[4].control == 0xa00);
      const uint32_t expected[5] = {0x10040000, counts[draw], 0, 0, 0};
      assert(!memcmp(retained[draw] + HK_G17P_VARYING_RECORD_OFFSET,
                     expected, sizeof(expected)));
      assert(decoded[1].context_offset == bases[draw]);
      assert(decoded[5].context_offset == bases[draw] + 128);
      for (unsigned i = 1; i < 8; ++i) {
         const unsigned offset = decoded[i].context_offset - bases[draw];
         assert(offset < HK_G17P_BIND_STATE_BYTES);
         assert(retained[draw][offset] == (i == 5 ? 0x80 + draw : 0x10 + draw));
      }
      for (unsigned i = HK_G17P_BIND_STATE_BYTES; i < sizeof(retained[draw]); ++i)
         assert(retained[draw][i] == 0xa5);
      memcpy(saved[draw], retained[draw], sizeof(saved[draw]));
      for (unsigned earlier = 0; earlier <= draw; ++earlier)
         assert(!memcmp(saved[earlier], retained[earlier], sizeof(saved[earlier])));
   }
   memset(group, 0xee, sizeof(group));
   memset(viewport, 0xee, sizeof(viewport));
   for (unsigned i = 0; i < 3; ++i)
      assert(!memcmp(saved[i], retained[i], sizeof(saved[i])));

   uint8_t indices[3][12];
   uint32_t pointers[3];
   for (unsigned i = 0; i < 3; ++i) {
      memset(indices[i], 0xa5, sizeof(indices[i]));
      /* A:first0, A:next1, independently recorded B:first0. */
      const unsigned index = i == 1;
      assert(hk_g17p_upload_draw_indices(indices[i], sizeof(indices[i]),
                bases[i] + 0xc0, 16 * (index + 1), 12 * (index + 1), 1,
                &pointers[i]));
      const struct agx_vdm_hal200_direct_draw_fields fields = {
         .draw_state_context_offset = pointers[i], .primitive = 6,
         .vertex_count = 3, .instance_count = 1,
      };
      struct agx_vdm_hal200_direct_draw encoded;
      struct agx_vdm_hal200_direct_draw_fields decoded;
      assert(agx_vdm_hal200_direct_draw_pack(AGX_CHIP_G17P, &fields, &encoded));
      assert(agx_vdm_hal200_direct_draw_unpack(AGX_CHIP_G17P, encoded.bytes,
                                             encoded.length, &decoded));
      assert(decoded.draw_state_context_offset == bases[i] + 0xc0);
      assert(indices[i][4] == index && indices[i][6] == index);
      for (unsigned b = 8; b < 12; ++b) assert(indices[i][b] == 0xa5);
   }
   assert(indices[0][4] == 0 && indices[1][4] == 1 && indices[2][4] == 0);
   assert(pointers[0] != pointers[1] && pointers[1] != pointers[2]);
   const uint64_t bad[] = {0, 3, 0x7ffffc, 0x800001, 0x1fffffc,
                          0x2000000, 0x1000200000, 0x10000000000, UINT64_MAX};
   for (unsigned i = 0; i < sizeof(bad) / sizeof(bad[0]); ++i) {
      uint32_t untouched = 0xdeadbeef;
      assert(!hk_g17p_upload_bind_state(retained[0], sizeof(retained[0]),
                 bad[i], group, viewport, sizeof(viewport), 2, 0, &uploads[0]));
      assert(!memcmp(saved[0], retained[0], sizeof(saved[0])));
      assert(!hk_g17p_upload_draw_indices(indices[0], 8, bad[i], 16, 12, 1,
                                          &untouched));
      assert(untouched == 0xdeadbeef && indices[0][4] == 0);
   }
   assert(!hk_g17p_upload_bind_state(retained[0], sizeof(retained[0]),
              bases[0], group, viewport, sizeof(viewport), 0x1000000, 0, &uploads[0]));
   assert(!memcmp(saved[0], retained[0], sizeof(saved[0])));

   /* A two-viewport PPP record grows from 40 to 72 bytes. Its varying record
    * follows the complete payload and the draw selector names the first of
    * the two consecutive scissor/depth records. */
   uint8_t multi[256], viewport2[72], multi_indices[8];
   memset(multi, 0xa5, sizeof(multi));
   memset(viewport2, 0x6d, sizeof(viewport2));
   assert(hk_g17p_upload_bind_state(
      multi, sizeof(multi), bases[0], group, viewport2, sizeof(viewport2),
      0x12, 0x34, &uploads[0]));
   assert(uploads[0].pairs[0].context_offset == bases[0] + 128 + 72);
   assert(uploads[0].pairs[5].context_offset == bases[0] + 128);
   /* The viewport slot's PPP size follows the payload: 18 words, not 10. */
   assert(uploads[0].pairs[4].control == 0x1200);
   uint32_t multi_tail_control;
   memcpy(&multi_tail_control, uploads[0].tail.bytes + 4 + 8 * 4 + 4, 4);
   assert(multi_tail_control == 0x1200);
   const uint32_t multi_varyings[5] = {0x10040000, 0x12, 0x34, 0, 0};
   assert(!memcmp(multi + 128 + 72, multi_varyings,
                  sizeof(multi_varyings)));
   assert(agx_g17p_bind_state_size(sizeof(viewport2)) == 220);
   for (unsigned i = 220; i < sizeof(multi); ++i)
      assert(multi[i] == 0xa5);
   assert(hk_g17p_upload_draw_indices(multi_indices, sizeof(multi_indices),
                                     bases[0] + 0xc0, 48, 36, 2,
                                     &pointers[0]));
   assert(indices[0][4] == 0);
   assert(multi_indices[4] == 1 && multi_indices[6] == 1);

   assert(hk_g17p_draw_state_range_valid(0x1fffff8, 8));
   assert(!hk_g17p_draw_state_range_valid(0x1fffff8, 9));
   puts("immutable G17P draw state: per-draw smooth/linear/zero counts, Bind0 parity, aliases, retained records and transactional guards PASS");
}
