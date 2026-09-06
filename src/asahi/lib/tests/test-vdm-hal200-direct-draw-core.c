/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_vdm_hal200_direct_draw_core.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static void
check_indexed(void)
{
   /* Valid native194053 doorbell1: PPP draw-state offset0x48000, restart
    * disabled with retained 0xffff state, uint16 indices and a 48-bit VA. */
   struct agx_vdm_hal200_indexed_draw_fields f = {
      .primitive = 6, .index_size_B = 2, .restart_index = 0xffff,
      .draw_state_context_offset = 0x48000,
      .index_buffer = UINT64_C(0x10001bc0180),
      .index_buffer_range_B = 0x3fe00,
      .index_count = 6,
      .instance_count = 1,
   };
   const uint32_t native[] = {
      0x48000, 0x40000001, 0xffff, 0x61f20600, 0x01bc0180,
      6, 1, 0, 0xff7f, 1,
   };
   struct agx_vdm_hal200_indexed_draw packed;
   assert(agx_vdm_hal200_indexed_draw_pack(AGX_CHIP_G17P, &f, &packed));
   assert(packed.length == sizeof(native));
   assert(memcmp(packed.bytes, native, sizeof(native)) == 0);

   uint8_t body[AGX_VDM_HAL200_INDEXED_DRAW_BODY_LENGTH];
   size_t emitted;
   assert(agx_vdm_hal200_draw_emit(AGX_CHIP_G17P, packed.bytes, packed.length,
                                  false, body, sizeof(body), &emitted));
   assert(emitted == sizeof(body));
   assert(!memcmp(body, &native[1], sizeof(body)));

   /* firstIndex is in uint16 elements and reduces the remaining byte range.
    * Relocating an equivalent bound buffer must preserve the complete packet. */
   f.index_buffer -= 6;
   f.index_buffer_range_B += 6;
   f.first_index = 3;
   assert(agx_vdm_hal200_indexed_draw_pack(AGX_CHIP_G17P, &f, &packed));
   assert(memcmp(packed.bytes, native, sizeof(native)) == 0);
   const int32_t biases[] = {INT32_MIN, -7, -1, 0, 1, 7, INT32_MAX};
   for (unsigned i = 0; i < sizeof(biases) / sizeof(biases[0]); ++i) {
      f.base_vertex = biases[i];
      assert(agx_vdm_hal200_indexed_draw_pack(AGX_CHIP_G17P, &f, &packed));
      uint32_t word;
      memcpy(&word, packed.bytes + 28, 4);
      assert(word == (uint32_t)biases[i]);
   }

   uint64_t relocated;
   uint32_t size_word;
   assert(agx_vdm_hal200_u16_index_range(0x1000, 12, 0, 6,
                                         &relocated, &size_word));
   assert(relocated == 0x1000 && size_word == 2);
   assert(agx_vdm_hal200_u16_index_range(0x1000, 14, 1, 6,
                                         &relocated, &size_word));
   assert(relocated == 0x1002 && size_word == 2);
   assert(agx_vdm_hal200_u16_index_range(0x1000, 2, 0, 1,
                                         &relocated, &size_word));
   assert(size_word == 0);
   assert(agx_vdm_hal200_u16_index_range(0x1000, UINT32_MAX, 0, 1,
                                         &relocated, &size_word));
   assert(size_word == 0x3fffffff);
   assert(!agx_vdm_hal200_u16_index_range(0x1000, 12, 1, 6,
                                          &relocated, &size_word));
   assert(!agx_vdm_hal200_u16_index_range(0x1000, UINT32_MAX, UINT32_MAX, 1,
                                          &relocated, &size_word));
   assert(!agx_vdm_hal200_u16_index_range(0x1000, UINT32_MAX, 0, UINT32_MAX,
                                          &relocated, &size_word));
   assert(!agx_vdm_hal200_u16_index_range(0x1000, UINT64_C(1) << 32, 0, 1,
                                          &relocated, &size_word));
   assert(!agx_vdm_hal200_u16_index_range((UINT64_C(1) << 48) - 2, 4, 0, 1,
                                          &relocated, &size_word));
   assert(!agx_vdm_hal200_u16_index_range(0x1001, 12, 0, 6,
                                          &relocated, &size_word));
   assert(!agx_vdm_hal200_u16_index_range(0, 12, 0, 6,
                                          &relocated, &size_word));
   assert(!agx_vdm_hal200_u16_index_range(0x1000, 0, 0, 1,
                                          &relocated, &size_word));
   assert(!agx_vdm_hal200_u16_index_range(0x1000, 12, 0, 0,
                                          &relocated, &size_word));
   assert(!agx_vdm_hal200_u16_index_range(0x1000, 12, 0, 6, NULL, &size_word));
   assert(!agx_vdm_hal200_u16_index_range(0x1000, 12, 0, 6, &relocated, NULL));
   f.instance_count = 0;
   assert(!agx_vdm_hal200_indexed_draw_pack(AGX_CHIP_G17P, &f, &packed));
   f.instance_count = 1;
   f.draw_state_context_offset |= 1;
   assert(!agx_vdm_hal200_indexed_draw_pack(AGX_CHIP_G17P, &f, &packed));
   f.draw_state_context_offset = 0x48000;
   assert(!agx_vdm_hal200_indexed_draw_pack(AGX_CHIP_G17G, &f, &packed));
}

static uint32_t word_at(const uint8_t *p, unsigned word)
{
   uint32_t value;
   memcpy(&value, p + 4 * word, 4);
   return value;
}

static void check_native_indexed_domain(void)
{
   const uint8_t primitives[] = {0, 1, 3, 5, 6, 9, 11, 13, 14, 15};
   const uint32_t instances[] = {1, 2, 17, UINT32_MAX};
   for (unsigned b = 1; b <= 4; b *= 2) {
      for (unsigned p = 0; p < sizeof(primitives); p++) {
         for (unsigned r = 0; r < 2; r++) {
            for (unsigned n = 0; n < sizeof(instances) / sizeof(instances[0]); n++) {
               const struct agx_vdm_hal200_indexed_draw_fields f = {
                  .primitive = primitives[p], .index_size_B = b,
                  .restart_enable = r, .restart_index = UINT32_MAX,
                  .draw_state_context_offset = 0x48000,
                  .index_buffer = UINT64_C(0x10001bc0180),
                  .index_buffer_range_B = 0x3fe00,
                  .first_index = 3, .index_count = 6,
                  .instance_count = instances[n], .base_vertex = -7,
               };
               struct agx_vdm_hal200_indexed_draw packet;
               assert(agx_vdm_hal200_indexed_draw_pack(AGX_CHIP_G17P, &f, &packet));
               assert(packet.length == 40);
               assert(word_at(packet.bytes, 0) == 0x48000);
               assert(word_at(packet.bytes, 1) == 0x40000001);
               assert(word_at(packet.bytes, 2) == (b == 1 ? 0xff : b == 2 ? 0xffff : UINT32_MAX));
               /* Native47e634..47e678 independently identify all packet words. */
               assert(word_at(packet.bytes, 3) ==
                      ((b == 1 ? 0x61f00000u : b == 2 ? 0x61f20000u : 0x61f40000u) |
                       ((uint32_t)primitives[p] << 8) | (r << 16)));
               assert(word_at(packet.bytes, 4) == 0x01bc0180 + 3 * b);
               assert(word_at(packet.bytes, 5) == 6);
               assert(word_at(packet.bytes, 6) == instances[n]);
               assert(word_at(packet.bytes, 7) == (uint32_t)-7);
               assert(word_at(packet.bytes, 8) == (b == 1 ? 0xff7f : b == 2 ? 0xff7e : 0xff7c));
               assert(word_at(packet.bytes, 9) == 1);
            }
         }
      }
   }
   uint64_t addr;
   uint32_t size;
   assert(agx_vdm_hal200_index_range(0x1000, 28, 1, 6, 4, &addr, &size));
   assert(addr == 0x1004 && size == 5);
   assert(!agx_vdm_hal200_index_range(0x1002, 28, 1, 6, 4, &addr, &size));
   assert(!agx_vdm_hal200_index_range(0x1000, 27, 1, 6, 4, &addr, &size));
   assert(!agx_vdm_hal200_index_range(0x1000, UINT32_MAX, UINT32_MAX, 1, 4, &addr, &size));
   assert(!agx_vdm_hal200_index_range(0x1000, 64, 0, 1, 0, &addr, &size));
   assert(!agx_vdm_hal200_index_range(0x1000, 64, 0, 1, 8, &addr, &size));
}

int
main(void)
{
   check_indexed();
   check_native_indexed_domain();
   const struct agx_vdm_hal200_direct_draw_fields fields = {
      .draw_state_context_offset = 0x48000,
      .primitive = 6,
      .vertex_count = 3,
      .instance_count = 1,
      .vertex_start = 0,
   };
   static const uint32_t expected[] = {
      0x00048000, 0x61c40600, 3, 1, 0,
   };

   struct agx_vdm_hal200_direct_draw packed;
   assert(agx_vdm_hal200_direct_draw_pack(AGX_CHIP_G17P, &fields, &packed));
   assert(packed.length == sizeof(expected));
   assert(memcmp(packed.bytes, expected, sizeof(expected)) == 0);

   struct agx_vdm_hal200_direct_draw_fields unpacked;
   assert(agx_vdm_hal200_direct_draw_unpack(
      AGX_CHIP_G17P, packed.bytes, packed.length, &unpacked));
   assert(memcmp(&unpacked, &fields, sizeof(fields)) == 0);

   /* The first draw follows newly emitted render state and therefore carries
    * its four-byte PPP selector.  Native drawPrimitives emits exactly the
    * following 16-byte body.  With unchanged state, a consecutive draw is
    * another body directly; inserting a bare selector makes VDM interpret
    * the 0x61c4 header as part of an address (the hardware fault witness).
    */
   static const uint32_t end = 0xc0000000;
   uint8_t stream[40];
   size_t emitted;
   assert(agx_vdm_hal200_draw_emit(AGX_CHIP_G17P, packed.bytes, packed.length,
                                  true, stream, sizeof(stream), &emitted));
   assert(emitted == AGX_VDM_HAL200_DIRECT_DRAW_LENGTH);
   memcpy(stream + packed.length, &end, sizeof(end));
   static const uint32_t old_single[] = {
      0x00048000, 0x61c40600, 3, 1, 0, 0xc0000000,
   };
   assert(!memcmp(stream, old_single, sizeof(old_single)));
   assert(agx_vdm_hal200_draw_emit(
      AGX_CHIP_G17P, packed.bytes, packed.length, false,
      stream + packed.length, sizeof(stream) - packed.length, &emitted));
   assert(emitted == AGX_VDM_HAL200_DIRECT_DRAW_BODY_LENGTH);
   memcpy(stream + packed.length + AGX_VDM_HAL200_DIRECT_DRAW_BODY_LENGTH,
          &end, sizeof(end));
   static const uint32_t native_two_draw[] = {
      0x00048000, 0x61c40600, 3, 1, 0,
      0x61c40600, 3, 1, 0, 0xc0000000,
   };
   assert(!memcmp(stream, native_two_draw, sizeof(native_two_draw)));
   emitted = 0xdeadbeef;
   assert(!agx_vdm_hal200_draw_emit(
      AGX_CHIP_G17P, packed.bytes, packed.length, false, stream,
      AGX_VDM_HAL200_DIRECT_DRAW_BODY_LENGTH - 1, &emitted));
   assert(emitted == 0);

   /* A payload may equal the terminal opcode. Finalization must not inspect
    * vertex_start and mistake it for an already emitted stream terminator.
    */
   struct agx_vdm_hal200_direct_draw_fields endword_start = fields;
   endword_start.vertex_start = end;
   assert(agx_vdm_hal200_direct_draw_pack(AGX_CHIP_G17P, &endword_start, &packed));
   assert(agx_vdm_hal200_direct_draw_unpack(
      AGX_CHIP_G17P, packed.bytes, packed.length, &unpacked));
   assert(unpacked.vertex_start == end);
   assert(!agx_vdm_hal200_direct_draw_unpack(
      AGX_CHIP_G17P, stream, sizeof(old_single), &unpacked));
   for (size_t size = 0; size < AGX_VDM_HAL200_DIRECT_DRAW_LENGTH; ++size)
      assert(!agx_vdm_hal200_direct_draw_unpack(
         AGX_CHIP_G17P, stream, size, &unpacked));
   assert(!agx_vdm_hal200_direct_draw_pack(AGX_CHIP_G17G, &fields, &packed));
   puts("PASS: preserved native U16 golden; 240 width/topology/restart/instance packets, signed base vertices and range boundaries");
   return 0;
}
