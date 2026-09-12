/* Copyright 2026 Aurora Silicon */
/* SPDX-License-Identifier: MIT */

#include "agx3_ioto_load_packer.h"

#include <stdio.h>
#include <string.h>

static unsigned checks;

#define CHECK(x)                                                               \
   do {                                                                        \
      ++checks;                                                                \
      if (!(x)) {                                                              \
         fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x);                  \
         return false;                                                         \
      }                                                                        \
   } while (0)

#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))

static const enum agx_chip gpu0_targets[] = {
   AGX_CHIP_G15G, AGX_CHIP_G15S, AGX_CHIP_G15C, AGX_CHIP_G15D,
   AGX_CHIP_G16G, AGX_CHIP_G16S, AGX_CHIP_G16C, AGX_CHIP_G17P,
};

static const enum agx_chip gpu1_targets[] = {
   AGX_CHIP_G17G,
   AGX_CHIP_G17S,
   AGX_CHIP_G17C,
};

struct fixture {
   uint8_t bytes[1024];
   struct agx_ioto_inflight_load loads[16];
   struct agx_ioto_inflight_load scratch[16];
   struct agx_ioto_load_cache_entry cache[16];
   struct agx_ioto_load_core core;
};

static bool
fixture_init(struct fixture *f, enum agx_chip chip, bool gpu0_nops)
{
   memset(f, 0, sizeof(*f));
   return agx_ioto_load_core_init(
      &f->core, chip,
      (struct agx_ioto_load_sink){f->bytes, sizeof(f->bytes), 0}, f->loads,
      ARRAY_SIZE(f->loads), f->cache, ARRAY_SIZE(f->cache), 0, false,
      gpu0_nops);
}

static struct agx_ioto_load_workspace
workspace(struct fixture *f)
{
   return (struct agx_ioto_load_workspace){f->scratch, ARRAY_SIZE(f->scratch)};
}

static struct agx_ioto_inflight_load
absolute_record(int16_t offset)
{
   return (struct agx_ioto_inflight_load){
      .data = {.half = 40, .type = 3, .value = 1},
      .base = {.half = 44, .type = 4, .value = 8},
      .byte_offset = offset,
      .count_dwords = 1,
      .native_slot_half = 20,
      .lane_mask = 3,
      .cprog_dest_half = 40,
      .signal_slot = 1,
   };
}

static bool
test_target_gate(void)
{
   struct agx3_ioto_load_packer packer;
   for (size_t i = 0; i < ARRAY_SIZE(gpu0_targets); ++i) {
      CHECK(agx3_ioto_load_absolute_packer_ready(gpu0_targets[i]));
      CHECK(agx3_ioto_load_packer_init(&packer, gpu0_targets[i]));
      CHECK(packer.encoding == AGX_IOTO_LOAD_ENCODING_GPU0);
   }
   for (size_t i = 0; i < ARRAY_SIZE(gpu1_targets); ++i) {
      CHECK(agx3_ioto_load_absolute_packer_ready(gpu1_targets[i]));
      CHECK(agx3_ioto_load_packer_init(&packer, gpu1_targets[i]));
      CHECK(packer.encoding == AGX_IOTO_LOAD_ENCODING_GPU1);
   }
   CHECK(!agx3_ioto_load_absolute_packer_ready(AGX_CHIP_G14G));
   CHECK(!agx3_ioto_load_absolute_packer_ready(AGX_CHIP_G15P));
   CHECK(!agx3_ioto_load_packer_init(&packer, AGX_CHIP_UNKNOWN));
   return true;
}

static bool
pack_load(enum agx_chip chip, struct agx_ioto_inflight_load *load,
          struct agx_ioto_load_raw_record *raw, uint8_t *slot)
{
   struct agx3_ioto_load_packer packer;
   struct agx_ioto_load_emit_ops ops;
   if (!agx3_ioto_load_packer_init(&packer, chip))
      return false;
   agx3_ioto_load_packer_ops(&packer, &ops);
   return ops.pack_load(ops.context, chip, load, raw, slot);
}

static bool
test_load_records(void)
{
   static const uint8_t gpu0_expected[] = {
      0x67, 0x00, 0x54, 0x28, 0x16, 0x00, 0x00,
      0x00, 0x11, 0x00, 0x00, 0x40, 0x22, 0x00,
   };
   static const uint8_t gpu1_ep0[] = {
      0x4f, 0x2c, 0x43, 0x00, 0x80, 0x04, 0x00, 0x20,
   };
   static const uint8_t gpu1_ep2[] = {
      0x4f, 0x2c, 0x43, 0x00, 0x80, 0x04, 0x00,
      0xa0, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01,
   };

   struct agx_ioto_inflight_load load = absolute_record(0);
   struct agx_ioto_load_raw_record raw;
   uint8_t slot = 0;
   for (size_t i = 0; i < ARRAY_SIZE(gpu0_targets); ++i) {
      CHECK(pack_load(gpu0_targets[i], &load, &raw, &slot));
      CHECK(raw.kind == AGX_IOTO_GPU0_INLINE_LDDEV14);
      CHECK(raw.length == sizeof(gpu0_expected));
      CHECK(memcmp(raw.bytes, gpu0_expected, sizeof(gpu0_expected)) == 0);
      CHECK(slot == 1);
   }
   for (size_t i = 0; i < ARRAY_SIZE(gpu1_targets); ++i) {
      CHECK(pack_load(gpu1_targets[i], &load, &raw, &slot));
      CHECK(raw.kind == AGX_IOTO_LOAD_RAW_LDDEV_EP0);
      CHECK(raw.length == sizeof(gpu1_ep0));
      CHECK(memcmp(raw.bytes, gpu1_ep0, sizeof(gpu1_ep0)) == 0);
   }

   load.byte_offset = 256;
   CHECK(pack_load(AGX_CHIP_G17G, &load, &raw, &slot));
   CHECK(raw.kind == AGX_IOTO_LOAD_RAW_LDDEV_EP2);
   CHECK(raw.length == sizeof(gpu1_ep2));
   CHECK(memcmp(raw.bytes, gpu1_ep2, sizeof(gpu1_ep2)) == 0);

   load.byte_offset = -1;
   CHECK(pack_load(AGX_CHIP_G17G, &load, &raw, &slot));
   CHECK(raw.kind == AGX_IOTO_LOAD_RAW_LDDEV_EP2);
   load.data.type = 2;
   CHECK(!pack_load(AGX_CHIP_G17G, &load, &raw, &slot));
   return true;
}

static bool
test_copy_records(void)
{
   static const uint8_t gpu0_dword0[] = {0xab, 0x28, 0x09, 0x04};
   static const uint8_t gpu0_dword1[] = {0xbb, 0x2a, 0x09, 0x04};
   static const uint8_t gpu1_dword0[] = {0xab, 0x28, 0x09, 0x44};
   static const uint8_t gpu1_dword1[] = {0xbb, 0x2a, 0x09, 0x04};

   struct agx_ioto_inflight_load load = absolute_record(0);
   load.count_dwords = 2;
   struct agx_ioto_load_raw_record raw;
   struct agx3_ioto_load_packer packer;
   struct agx_ioto_load_emit_ops ops;

   CHECK(agx3_ioto_load_packer_init(&packer, AGX_CHIP_G15G));
   agx3_ioto_load_packer_ops(&packer, &ops);
   CHECK(ops.pack_copy(ops.context, AGX_CHIP_G15G, &load, 0, 3, &raw));
   CHECK(raw.kind == AGX_IOTO_LOAD_RAW_COPY_EP0 &&
         raw.length == sizeof(gpu0_dword0));
   CHECK(memcmp(raw.bytes, gpu0_dword0, sizeof(gpu0_dword0)) == 0);
   CHECK(ops.pack_copy(ops.context, AGX_CHIP_G15G, &load, 1, 3, &raw));
   CHECK(memcmp(raw.bytes, gpu0_dword1, sizeof(gpu0_dword1)) == 0);

   CHECK(agx3_ioto_load_packer_init(&packer, AGX_CHIP_G17G));
   agx3_ioto_load_packer_ops(&packer, &ops);
   CHECK(ops.pack_copy(ops.context, AGX_CHIP_G17G, &load, 0, 3, &raw));
   CHECK(raw.kind == AGX_IOTO_LOAD_RAW_COPY_EP0 &&
         raw.length == sizeof(gpu1_dword0));
   CHECK(memcmp(raw.bytes, gpu1_dword0, sizeof(gpu1_dword0)) == 0);
   CHECK(ops.pack_copy(ops.context, AGX_CHIP_G17G, &load, 1, 3, &raw));
   CHECK(memcmp(raw.bytes, gpu1_dword1, sizeof(gpu1_dword1)) == 0);
   CHECK(!ops.pack_copy(ops.context, AGX_CHIP_G17G, &load, 2, 3, &raw));
   CHECK(!ops.pack_copy(ops.context, AGX_CHIP_G17G, &load, 0, 0, &raw));

   /* Native G17G EOT Ioto, macOS 25G76, offsets 0xde and 0xe2: a
    * two-dword pointer moves from halves 36/38 to adjacent halves 0/2.
    * An earlier COPY already consumed the load's wait in this sequence.
    */
   load.native_slot_half = 0;
   load.data.half = 36;
   CHECK(ops.pack_copy(ops.context, AGX_CHIP_G17G, &load, 0, 3, &raw));
   CHECK(raw.length == 4 &&
         memcmp(raw.bytes, (uint8_t[]){0x0b, 0x24, 0x09, 0x04}, 4) == 0);
   CHECK(ops.pack_copy(ops.context, AGX_CHIP_G17G, &load, 1, 3, &raw));
   CHECK(raw.length == 4 &&
         memcmp(raw.bytes, (uint8_t[]){0x1b, 0x26, 0x09, 0x04}, 4) == 0);

   load.native_slot_half = 150;
   load.data.half = 190;
   CHECK(agx3_ioto_load_packer_init(&packer, AGX_CHIP_G15G));
   agx3_ioto_load_packer_ops(&packer, &ops);
   CHECK(ops.pack_copy(ops.context, AGX_CHIP_G15G, &load, 0, 3, &raw));
   CHECK(raw.kind == AGX_IOTO_LOAD_RAW_COPY_EP1 && raw.length == 10);
   CHECK(memcmp(raw.bytes,
                (uint8_t[]){0xbb, 0x3f, 0x0f, 0x00, 0x42, 0x11, 0x00, 0x00,
                            0x00, 0x00},
                10) == 0);

   CHECK(agx3_ioto_load_packer_init(&packer, AGX_CHIP_G17G));
   agx3_ioto_load_packer_ops(&packer, &ops);
   CHECK(ops.pack_copy(ops.context, AGX_CHIP_G17G, &load, 0, 3, &raw));
   CHECK(raw.kind == AGX_IOTO_LOAD_RAW_COPY_EP1 && raw.length == 8);
   CHECK(memcmp(raw.bytes,
                /* Wide EP1 forms are DERIVED from the field-27 fix, not
                 * independently witnessed; only the 4-byte EP0 copies above
                 * are pinned by Apple bytes (add3 + the G17S body). */
                (uint8_t[]){0xb3, 0x0a, 0x0f, 0x7a, 0x36, 0xa0, 0xa5, 0x0a},
                8) == 0);
   return true;
}

static bool
queue_two_dwords(struct fixture *f)
{
   const struct agx_ioto_load_absolute request = {
      .native_slot = 10,
      .count_dwords = 2,
      .address = 0x20,
      .lane_mask = 0x0f,
      .cprog_dest_half = 40,
   };
   struct agx_ioto_load_allocation allocation;
   return agx3_ioto_load_absolute_native(&f->core, &request, workspace(f),
                                         &allocation) &&
          allocation.temp_half == 0;
}

static bool
test_native_rounds(void)
{
   static const uint8_t gpu0_tail[] = {
      0x67, 0x00, 0x54, 0x04, 0x00, 0x00, 0x00, 0x00, 0x19, 0x00,
      0x00, 0x40, 0x22, 0x00, 0x06, 0x08, 0x06, 0x00, 0x06, 0x00,
      0x06, 0x00, 0xab, 0x04, 0x09, 0x04, 0xbb, 0x06, 0x09, 0x04,
   };
   static const uint8_t gpu1_tail[] = {
      0x2f, 0x00, 0x03, 0x00, 0xa0, 0x0c, 0x00, 0x20,
      0xab, 0x04, 0x09, 0x44, 0xbb, 0x06, 0x09, 0x04,
   };

   struct fixture f;
   CHECK(fixture_init(&f, AGX_CHIP_G15G, true));
   CHECK(queue_two_dwords(&f));
   CHECK(f.core.sink.length == 4);
   CHECK(agx3_ioto_load_finish_round_native(&f.core, true, workspace(&f)));
   CHECK(f.core.sink.length == 4 + sizeof(gpu0_tail));
   CHECK(memcmp(&f.bytes[4], gpu0_tail, sizeof(gpu0_tail)) == 0);

   CHECK(fixture_init(&f, AGX_CHIP_G17G, false));
   CHECK(queue_two_dwords(&f));
   CHECK(agx3_ioto_load_finish_round_native(&f.core, false, workspace(&f)));
   CHECK(f.core.sink.length == 12);
   CHECK(f.core.loads[0].processed && f.core.loads[0].signal_slot == 1);
   CHECK(agx3_ioto_load_finish_round_native(&f.core, true, workspace(&f)));
   CHECK(f.core.sink.length == 4 + sizeof(gpu1_tail));
   CHECK(memcmp(&f.bytes[4], gpu1_tail, sizeof(gpu1_tail)) == 0);
   CHECK(f.core.load_count == 0 && f.core.cache_count == 0);

   CHECK(fixture_init(&f, AGX_CHIP_G15G, false));
   CHECK(!queue_two_dwords(&f));
   return true;
}

static bool
test_native_direct_sources(void)
{
   static const uint8_t scs_base[] = {
      0x4c, 0x80, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00,
   };
   static const uint8_t gpu0_scs[] = {
      0x4c, 0x80, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x67, 0x00,
      0x54, 0x00, 0x04, 0x00, 0x00, 0x00, 0x19, 0x60, 0x24, 0x40,
      0x22, 0x00, 0x06, 0x08, 0x06, 0x00, 0x06, 0x00, 0x06, 0x00,
      0x7b, 0x00, 0x09, 0x04, 0x8b, 0x02, 0x09, 0x04,
   };
   static const uint8_t gpu1_scs[] = {
      0x4c, 0x80, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x0f, 0x08,
      0x03, 0x00, 0xa0, 0x0c, 0x60, 0xa4, 0x01, 0x00, 0x00, 0x00,
      0x00, 0x01, 0x7b, 0x00, 0x09, 0x44, 0x8b, 0x02, 0x09, 0x04,
   };
   static const uint8_t gpu1_abi_slot2[] = {
      0x0f, 0x04, 0x03, 0x00, 0xa0, 0x0c, 0x60, 0xa4, 0x01, 0x00, 0x00,
      0x00, 0x00, 0x01, 0x7b, 0x00, 0x09, 0x44, 0x8b, 0x02, 0x09, 0x04,
   };
   static const uint8_t gpu1_abi_slot54[] = {
      0x0f, 0x6c, 0x03, 0x00, 0xa0, 0x0c, 0x60, 0xa4, 0x01, 0x00, 0x00,
      0x00, 0x00, 0x01, 0x7b, 0x00, 0x09, 0x44, 0x8b, 0x02, 0x09, 0x04,
   };
   static const enum agx_ioto_abi_buffer_slot abi_slots[] = {
      AGX_IOTO_ABI_BUFFER_SLOT_2,  AGX_IOTO_ABI_BUFFER_SLOT_4,
      AGX_IOTO_ABI_BUFFER_SLOT_6,  AGX_IOTO_ABI_BUFFER_SLOT_8,
      AGX_IOTO_ABI_BUFFER_SLOT_50, AGX_IOTO_ABI_BUFFER_SLOT_52,
      AGX_IOTO_ABI_BUFFER_SLOT_54,
   };
   const struct agx_ioto_load_from_fixed request = {
      .native_slot = 7,
      .count_dwords = 2,
      .byte_offset = 0x123,
      .lane_mask = 0xf,
      .cprog_dest_half = 40,
   };
   struct agx_ioto_load_allocation allocation;
   struct fixture f;

   CHECK(fixture_init(&f, AGX_CHIP_G15G, true));
   CHECK(agx3_ioto_load_from_scs_native(&f.core, &request, workspace(&f),
                                        &allocation));
   CHECK(allocation.temp_half == 0 && f.core.sink.length == sizeof(scs_base));
   CHECK(memcmp(f.bytes, scs_base, sizeof(scs_base)) == 0);
   CHECK(f.core.loads[0].base.half == AGX_IOTO_LOAD_SCS_BASE_HALF);
   CHECK(agx3_ioto_load_finish_round_native(&f.core, true, workspace(&f)));
   CHECK(f.core.sink.length == sizeof(gpu0_scs));
   CHECK(memcmp(f.bytes, gpu0_scs, sizeof(gpu0_scs)) == 0);
   CHECK(f.core.scs_base_loaded);
   CHECK(agx3_ioto_load_from_scs_native(&f.core, &request, workspace(&f),
                                        &allocation));
   CHECK(f.core.sink.length == sizeof(gpu0_scs));
   CHECK(agx3_ioto_load_finish_round_native(&f.core, true, workspace(&f)));
   CHECK(f.core.sink.length == sizeof(gpu0_scs) * 2 - sizeof(scs_base));
   CHECK(memcmp(&f.bytes[sizeof(gpu0_scs)], &gpu0_scs[sizeof(scs_base)],
                sizeof(gpu0_scs) - sizeof(scs_base)) == 0);

   CHECK(fixture_init(&f, AGX_CHIP_G17G, false));
   CHECK(agx3_ioto_load_from_scs_native(&f.core, &request, workspace(&f),
                                        &allocation));
   CHECK(memcmp(f.bytes, scs_base, sizeof(scs_base)) == 0);
   CHECK(agx3_ioto_load_finish_round_native(&f.core, true, workspace(&f)));
   CHECK(f.core.sink.length == sizeof(gpu1_scs));
   CHECK(memcmp(f.bytes, gpu1_scs, sizeof(gpu1_scs)) == 0);

   for (size_t i = 0; i < ARRAY_SIZE(abi_slots); ++i) {
      CHECK(fixture_init(&f, AGX_CHIP_G17G, false));
      CHECK(agx3_ioto_load_from_abi_buffer_native(
         &f.core, abi_slots[i], &request, workspace(&f), &allocation));
      CHECK(f.core.loads[0].base.half == 2u * (unsigned)abi_slots[i]);
      CHECK(!f.core.scs_base_loaded && !f.core.scs_base_ever_loaded);
      CHECK(f.core.abi4_load_pending ==
            (abi_slots[i] == AGX_IOTO_ABI_BUFFER_SLOT_4));
      CHECK(agx3_ioto_load_finish_round_native(&f.core, true, workspace(&f)));
      CHECK(!f.core.abi4_load_pending);
      CHECK(f.core.sink.length == sizeof(gpu1_abi_slot2));
      if (i == 0)
         CHECK(memcmp(f.bytes, gpu1_abi_slot2, sizeof(gpu1_abi_slot2)) == 0);
      if (i + 1 == ARRAY_SIZE(abi_slots))
         CHECK(memcmp(f.bytes, gpu1_abi_slot54, sizeof(gpu1_abi_slot54)) == 0);
   }

   CHECK(fixture_init(&f, AGX_CHIP_G17G, false));
   CHECK(agx3_ioto_load_from_abi_buffer_native(
      &f.core, AGX_IOTO_ABI_BUFFER_SLOT_4, &request, workspace(&f),
      &allocation));
   CHECK(!agx3_ioto_load_from_scs_native(&f.core, &request, workspace(&f),
                                         &allocation));
   CHECK(agx3_ioto_load_finish_round_native(&f.core, true, workspace(&f)));
   CHECK(agx3_ioto_load_from_scs_native(&f.core, &request, workspace(&f),
                                        &allocation));
   CHECK(!agx3_ioto_load_from_abi_buffer_native(
      &f.core, AGX_IOTO_ABI_BUFFER_SLOT_4, &request, workspace(&f),
      &allocation));

   CHECK(fixture_init(&f, AGX_CHIP_G15G, false));
   CHECK(!agx3_ioto_load_from_scs_native(&f.core, &request, workspace(&f),
                                         &allocation));
   CHECK(!agx3_ioto_load_from_abi_buffer_native(
      &f.core, AGX_IOTO_ABI_BUFFER_SLOT_2, &request, workspace(&f),
      &allocation));
   return true;
}

static bool
test_mem_form_bounds(void)
{
   struct agx3_mem_desc desc = {
      .data = 40,
      .width = AGX3_MEM_W32,
      .mask = 1,
      .base = 44,
      .idx_kind = AGX3_MEM_IDX_IMM,
      .byte_off = 0,
      .slot = 1,
   };
   uint8_t bytes[AGX3_MEM_MAX_LENGTH], length;
   CHECK(agx3_mem_encode_load_form(AGX_ISA_ENCODING_AGX3B, &desc, 0, bytes,
                                   &length));
   CHECK(!agx3_mem_encode_load_form(AGX_ISA_ENCODING_AGX3B, &desc, 1, bytes,
                                    &length));
   for (uint8_t ep = 0; ep <= 5; ++ep)
      CHECK(agx3_mem_encode_load_form(AGX_ISA_ENCODING_AGX3C, &desc, ep, bytes,
                                      &length));
   CHECK(!agx3_mem_encode_load_form(AGX_ISA_ENCODING_AGX3C, &desc, 6, bytes,
                                    &length));
   return true;
}

int
main(void)
{
   if (!test_target_gate() || !test_load_records() || !test_copy_records() ||
       !test_native_rounds() || !test_native_direct_sources() ||
       !test_mem_form_bounds())
      return 1;
   printf("agx3_ioto_load_packer: %u checks passed\n", checks);
   return 0;
}
