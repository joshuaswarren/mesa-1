/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "agx3_cprog_input.h"
#include "agx3_static_spill_core.h"
#include "agx3_slot_core.h"
#include <stdio.h>
#include <string.h>
#include "probes/agx3-cprog-copy-vectors.h"
#include "probes/agx3-cprog-import-vectors.h"

static unsigned checks;
#define CHECK(x) do { ++checks; if (!(x)) { \
   fprintf(stderr, "FAIL %d: %s\n", __LINE__, #x); return 1; } } while (0)

static void
set_input(uint64_t *halves, unsigned h)
{
   halves[h / 64] |= UINT64_C(1) << (h % 64);
}

int
main(void)
{
   struct agx3_cprog_inputs inputs;
   uint64_t halves[8] = {0};
   CHECK(agx3_cprog_inputs_init(halves, &inputs));
   CHECK(inputs.count == 0 && inputs.storage_bytes == 0);
   uint16_t target = 0xabcd;
   CHECK(!agx3_cprog_input_tps_half(&inputs, 0, &target));
   CHECK(target == 0xabcd);
   for (unsigned h = 0; h < 512; ++h) {
      set_input(halves, h);
      CHECK(agx3_cprog_inputs_init(halves, &inputs));
      CHECK(inputs.count == h + 1);
      CHECK(agx3_cprog_input_tps_half(&inputs, h, &target));
      CHECK(target == 8 + h);
      CHECK(inputs.storage_bytes >= (8 + h + 1) * 2);
      CHECK(!(inputs.storage_bytes & 15));
   }
   CHECK(inputs.storage_bytes == 1040);

   /* Native compiler metadata for 1, 3, 5 and 8 buffer pointers. */
   const unsigned count[] = {4, 12, 20, 32};
   const unsigned storage[] = {32, 48, 64, 80};
   for (unsigned c = 0; c < 4; ++c) {
      memset(halves, 0, sizeof(halves));
      for (unsigned h = 0; h < count[c]; ++h)
         set_input(halves, h);
      CHECK(agx3_cprog_inputs_init(halves, &inputs));
      CHECK(inputs.storage_bytes == storage[c]);
   }

   /* Compare the reused primitive against 28 calls to Apple's pure encoder,
    * including odd halves and both physical GPR and input-frame boundaries. */
   for (unsigned i = 0; i < sizeof(native_cprog_copies) /
                            sizeof(native_cprog_copies[0]); ++i) {
      struct agx3_static_spill_store_desc d = {
         .tps_half = native_cprog_copies[i].tps_half,
         .src = native_cprog_copies[i].src,
         .width = native_cprog_copies[i].width == 32
                     ? AGX3_MEM_W32 : AGX3_MEM_W16,
         .slot = 6, .src_kill = true,
      };
      uint8_t code[10], length;
      CHECK(agx3_static_spill_encode_store_form(
         AGX_ISA_ENCODING_AGX3B, &d, 1, code, &length));
      CHECK(length == 10);
      CHECK(memcmp(code, native_cprog_copies[i].bytes, 10) == 0);
   }

   /* Sparse import IDs must use compact ranks, not the uniform indices. */
   memset(halves, 0, sizeof(halves));
   set_input(halves, 0);
   set_input(halves, 63);
   set_input(halves, 64);
   set_input(halves, 511);
   CHECK(agx3_cprog_inputs_init(halves, &inputs));
   CHECK(inputs.count == 4 && inputs.storage_bytes == 32);
   const unsigned sparse[] = {0, 63, 64, 511};
   for (unsigned i = 0; i < 4; ++i) {
      CHECK(agx3_cprog_input_tps_half(&inputs, sparse[i], &target));
      CHECK(target == 8 + i);
   }

   uint8_t code[20];
   size_t length;
   /* Low and high source-half selection. */
   CHECK(agx3_cprog_input_publish(&inputs, 0, 0, code, sizeof(code), &length));
   CHECK(length == 10 && memcmp(code, native_cprog_copies[0].bytes, 10) == 0);
   CHECK(agx3_cprog_input_publish(&inputs, 62, 0, code, sizeof(code), &length));
   CHECK(length == 10 && memcmp(code, native_cprog_copies[5].bytes, 10) == 0);

   /* Full word at an even compact rank, then a pair at an odd rank. */
   memset(halves, 0, sizeof(halves));
   set_input(halves, 2); set_input(halves, 3);
   CHECK(agx3_cprog_inputs_init(halves, &inputs));
   CHECK(agx3_cprog_input_publish(&inputs, 2, 0, code, sizeof(code), &length));
   CHECK(length == 10 && memcmp(code, native_cprog_copies[16].bytes, 10) == 0);
   set_input(halves, 0);
   CHECK(agx3_cprog_inputs_init(halves, &inputs));
   CHECK(agx3_cprog_input_publish(&inputs, 2, 0, code, sizeof(code), &length));
   CHECK(length == 20);
   struct agx3_static_spill_store_desc first, second;
   CHECK(agx3_static_spill_decode_store(AGX_ISA_ENCODING_AGX3B, code, 10, &first));
   CHECK(agx3_static_spill_decode_store(AGX_ISA_ENCODING_AGX3B, code + 10, 10, &second));
   CHECK(first.src == 0 && first.tps_half == 9 && first.width == AGX3_MEM_W16);
   CHECK(second.src == 1 && second.tps_half == 10 && second.width == AGX3_MEM_W16);
   CHECK(first.slot == 6 && second.slot == 6 && first.src_kill && second.src_kill);

   memset(code, 0xa5, sizeof(code));
   uint8_t before[20]; memcpy(before, code, sizeof(code));
   length = 123;
   CHECK(!agx3_cprog_input_publish(&inputs, 2, 0, code, 19, &length));
   CHECK(length == 123 && memcmp(code, before, sizeof(code)) == 0);
   CHECK(!agx3_cprog_input_publish(&inputs, 1, 0, code, sizeof(code), &length));
   CHECK(!agx3_cprog_input_publish(&inputs, 2, 255, code, sizeof(code), &length));
   CHECK(!agx3_cprog_input_publish(&inputs, 512, 0, code, sizeof(code), &length));
   inputs.storage_bytes = 0;
   CHECK(!agx3_cprog_input_publish(&inputs, 2, 0, code, sizeof(code), &length));
   CHECK(length == 123 && memcmp(code, before, sizeof(code)) == 0);
   memset(halves, 0, sizeof(halves));
   CHECK(agx3_cprog_inputs_init(halves, &inputs));
   CHECK(agx3_cprog_input_publish(&inputs, 2, 0, code, 0, &length));
   CHECK(length == 0 && memcmp(code, before, sizeof(code)) == 0);
   CHECK(!agx3_cprog_inputs_init(NULL, &inputs));
   CHECK(!agx3_cprog_input_tps_half(&inputs, 512, &target));

   /* Native input RLDs: 16/32-bit data, both mirror/structural forms and
    * the first-import slot-6 wait. */
   halves[0] = 0xff;
   CHECK(agx3_cprog_inputs_init(halves, &inputs));
   for (unsigned i = 0; i < sizeof(native_cprog_imports) /
                            sizeof(native_cprog_imports[0]); ++i) {
      CHECK(agx3_cprog_input_import(&inputs, &native_cprog_imports[i].desc,
                                    code, sizeof(code), &length));
      CHECK(length == native_cprog_imports[i].length);
      CHECK(memcmp(code, native_cprog_imports[i].bytes, length) == 0);
   }

   /* Connect the loader publication, external dependency and actual RLD
    * encoding. Waiting the loader does not retire the RLD's new generation. */
   CHECK(agx3_cprog_input_publish(&inputs, 0, 0, code, sizeof(code), &length));
   struct agx3_static_spill_store_desc publication;
   CHECK(agx3_static_spill_decode_store(AGX_ISA_ENCODING_AGX3B, code, length,
                                       &publication));
   const struct agx3_slot_entry entry = {
      .token = AGX3_SLOT_CPROG_INPUT_TOKEN, .slot = publication.slot,
   };
   const struct agx3_slot_instr ops[] = {
      {.is_async=true, .preferred_slot=1, .nr_srcs=1, .nr_dsts=1,
       .src={AGX3_SLOT_CPROG_INPUT_TOKEN}, .dst={0}},
      {.nr_srcs=1, .src={0}},
   };
   struct agx3_slot_result slots[2];
   CHECK(agx3_slot_assign_with_entry(ops, 2, 6, &entry, 1, slots));
   CHECK(slots[0].slot == 1 && slots[0].wait_mask == 32);
   CHECK(slots[1].wait_mask == 1);
   struct agx3_cprog_import load = {
      .uniform_half=0, .dest_half=0, .width=AGX3_MEM_W16,
      .slot=slots[0].slot, .wait_mask=slots[0].wait_mask,
   };
   CHECK(agx3_cprog_input_import(&inputs, &load, code, sizeof(code), &length));
   CHECK(length == 8 && memcmp(code, native_cprog_imports[3].bytes, 8) == 0);

   /* Invalid/unselected words and capacity failures cannot publish a partial
    * instruction or a length. Odd compact words require separate half loads. */
   memset(halves, 0, sizeof(halves));
   set_input(halves, 0); set_input(halves, 2); set_input(halves, 3);
   CHECK(agx3_cprog_inputs_init(halves, &inputs));
   memset(code, 0xa5, sizeof(code));
   memcpy(before, code, sizeof(code));
   length = 123;
   load.uniform_half = 2;
   load.width = AGX3_MEM_W32;
   CHECK(!agx3_cprog_input_import(&inputs, &load, code, sizeof(code), &length));
   load.uniform_half = 0;
   CHECK(!agx3_cprog_input_import(&inputs, &load, code, sizeof(code), &length));
   load.width = AGX3_MEM_W16;
   CHECK(!agx3_cprog_input_import(&inputs, &load, code, 7, &length));
   load.slot = 0;
   CHECK(!agx3_cprog_input_import(&inputs, &load, code, sizeof(code), &length));
   load.slot = 1;
   load.uniform_half = 512;
   CHECK(!agx3_cprog_input_import(&inputs, &load, code, sizeof(code), &length));
   CHECK(length == 123 && memcmp(code, before, sizeof(code)) == 0);
   load.uniform_half = 2;
   load.dest_half = 1;
   CHECK(agx3_cprog_input_import(&inputs, &load, code, sizeof(code), &length));
   struct agx3_static_spill_load_desc decoded;
   CHECK(agx3_static_spill_decode_load(AGX_ISA_ENCODING_AGX3B, code, length, &decoded));
   CHECK(decoded.width == AGX3_MEM_W16 && decoded.dest == 1 && decoded.tps_half == 9);
   const unsigned invalid_publications[][3] = {
      {4, 8, 0}, {3, 519, 0}, {1, 7, 0}, {1, 520, 0},
      {1, 8, 255}, {1, 8, 1},
   };
   memset(code, 0xa5, sizeof(code));
   memcpy(before, code, sizeof(code));
   length = 123;
   for (unsigned i = 0; i < sizeof(invalid_publications) /
                            sizeof(invalid_publications[0]); ++i) {
      const unsigned *v = invalid_publications[i];
      CHECK(!agx3_cprog_input_publish_at(v[0], v[1], v[2],
                                         code, sizeof(code), &length));
      CHECK(length == 123 && !memcmp(code, before, sizeof(code)));
   }
   CHECK(!agx3_cprog_input_publish_at(3, 9, 0, code, 19, &length));
   CHECK(length == 123 && !memcmp(code, before, sizeof(code)));
   printf("AGX3 CProg inputs: %u checks passed\n", checks);
   return 0;
}
