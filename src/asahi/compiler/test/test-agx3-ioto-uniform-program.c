/* Copyright 2026 Aurora Silicon */
/* SPDX-License-Identifier: MIT */

#include "agx3_ioto_uniform_program.h"
#include "agx3_static_spill_core.h"

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
   uint8_t body[1024];
   uint8_t output[1200];
   struct agx_ioto_inflight_load loads[64];
   struct agx_ioto_inflight_load sort[64];
   struct agx_ioto_load_cache_entry cache[64];
};

static bool
is_gpu1(enum agx_chip chip)
{
   return chip == AGX_CHIP_G17G || chip == AGX_CHIP_G17S ||
          chip == AGX_CHIP_G17C;
}

static struct agx3_ioto_uniform_program_workspace
workspace(struct fixture *f)
{
   return (struct agx3_ioto_uniform_program_workspace){
      .body = f->body,
      .body_capacity = sizeof(f->body),
      .loads = f->loads,
      .load_capacity = ARRAY_SIZE(f->loads),
      .sort = f->sort,
      .sort_capacity = ARRAY_SIZE(f->sort),
      .cache = f->cache,
      .cache_capacity = ARRAY_SIZE(f->cache),
   };
}

static struct agx3_ioto_uniform_program_desc
program_desc(enum agx_chip chip, const struct agx3_ioto_uniform_range *ranges,
             size_t range_count)
{
   struct agx3_ioto_uniform_program_desc desc = {
      .program_va = UINT64_C(0x123456789a80),
      .ranges = ranges,
      .range_count = range_count,
      .profile_payload0 = is_gpu1(chip) ? UINT16_C(0x8098) : UINT16_C(0x0555),
      .profile_payload1 = is_gpu1(chip) ? UINT16_C(0xa400) : 0,
      .preshader.external_va = AGX_IOTO_PRESHADER_NO_VA,
      .tail.type = AGX_IOTO_STATE_TAIL_TYPE0,
   };

   desc.tail.ldshdr.f[0] = UINT64_C(0x123456789ac0);
   if (is_gpu1(chip)) {
      desc.tail.speclm.f[0] = 3;
      desc.tail.speclm.f[1] = 165;
      desc.tail.ldshdr.f[5] = 3;
      desc.tail.ldshdr.f[6] = 107;
      desc.tail.spectpr.f[0] = 3;
      desc.tail.spectpr.f[1] = 166;
   } else {
      desc.tail.ldshdr.f[23] = 1;
   }
   return desc;
}

static bool
test_exact_uniform_program(void)
{
   static const struct agx3_ioto_uniform_range range = {
      .start_halfs = 20,
      .size_halfs = 4,
      .address = 0x20,
   };
   static const uint8_t gpu0_body[] = {
      0x0c, 0x20, 0x1c, 0x00, 0x67, 0x00, 0x54, 0x04, 0x00, 0x00, 0x00, 0x00,
      0x19, 0x00, 0x00, 0x40, 0x22, 0x00, 0x06, 0x08, 0x06, 0x00, 0x06, 0x00,
      0x06, 0x00, 0xab, 0x04, 0x09, 0x04, 0xbb, 0x06, 0x09, 0x04,
   };
   static const uint8_t gpu1_body[] = {
      0x04, 0x20, 0x14, 0x00, 0x2f, 0x00, 0x03, 0x00, 0xa0, 0x0c,
      0x00, 0x20, 0xab, 0x04, 0x09, 0x44, 0xcb, 0x06, 0x09, 0x04,
   };

   const enum agx_chip targets[] = {
      AGX_CHIP_G15G, AGX_CHIP_G15S, AGX_CHIP_G15C, AGX_CHIP_G15D,
      AGX_CHIP_G16G, AGX_CHIP_G16S, AGX_CHIP_G16C, AGX_CHIP_G17P,
      AGX_CHIP_G17G, AGX_CHIP_G17S, AGX_CHIP_G17C,
   };
   for (size_t i = 0; i < ARRAY_SIZE(targets); ++i) {
      const enum agx_chip chip = targets[i];
      struct agx3_ioto_uniform_program_desc desc =
         program_desc(chip, &range, 1);
      struct fixture f;
      memset(&f, 0xa5, sizeof(f));
      size_t written = SIZE_MAX;
      CHECK(agx3_ioto_uniform_program_generate(
         chip, &desc, workspace(&f), f.output, sizeof(f.output), &written));

      const uint8_t *expected_body = is_gpu1(chip) ? gpu1_body : gpu0_body;
      const size_t body_size =
         is_gpu1(chip) ? sizeof(gpu1_body) : sizeof(gpu0_body);
      CHECK(memcmp(f.body, expected_body, body_size) == 0);
      CHECK(memcmp(f.output, expected_body, body_size) == 0);

      const struct agx_ioto_profile_control profile = {
         .chip = chip,
         .payload0 = desc.profile_payload0,
         .payload1 = desc.profile_payload1,
      };
      uint8_t packed_profile[AGX_IOTO_PROFILE_CONTROL_MAX_SIZE];
      struct agx_ioto_state_tail_output packed_tail;
      struct agx_ioto_preshader_output suffix;
      CHECK(agx_ioto_profile_control_pack(&profile, packed_profile));
      CHECK(agx_ioto_state_tail_pack(chip, &desc.tail, &packed_tail));
      CHECK(agx_ioto_preshader_emit(chip, &desc.preshader, &suffix));
      const size_t profile_size = agx_ioto_profile_control_size(chip);
      const size_t expected_size =
         body_size + profile_size + packed_tail.length + suffix.length;
      CHECK(written == expected_size);
      CHECK(memcmp(f.output + body_size, packed_profile, profile_size) == 0);
      CHECK(memcmp(f.output + body_size + profile_size, packed_tail.bytes,
                   packed_tail.length) == 0);
      CHECK(memcmp(f.output + written - suffix.length, suffix.bytes,
                   suffix.length) == 0);
      CHECK(f.output[written] == 0xa5);
      CHECK(!agx_ioto_load_production_ready(chip));
   }
   return true;
}

static bool
test_g17p_background_clear_body(void)
{
   static const struct agx3_ioto_uniform_range range = {
      .start_halfs = 4,
      .size_halfs = 8,
      .address = UINT64_C(0x1000008c000),
   };
   static const uint8_t expected[] = {
      0x0c, 0x80, 0x02, 0x00, 0x00, 0x00, 0x46, 0x00, 0x1c, 0x80, 0x02,
      0x00, 0x04, 0x00, 0x00, 0x00, 0x67, 0x00, 0x54, 0x04, 0x00, 0x00,
      0x00, 0x00, 0x17, 0x00, 0x00, 0x40, 0x22, 0x00, 0x06, 0x08, 0x06,
      0x00, 0x06, 0x00, 0x06, 0x00, 0x2b, 0x04, 0x09, 0x04, 0x3b, 0x06,
      0x09, 0x04, 0x4b, 0x08, 0x09, 0x04, 0x5b, 0x0a, 0x09, 0x04,
   };
   struct agx3_ioto_uniform_program_desc desc =
      program_desc(AGX_CHIP_G17P, &range, 1);
   struct fixture f;
   memset(&f, 0xa5, sizeof(f));
   size_t written = SIZE_MAX, copy_offset = SIZE_MAX;
   CHECK(agx3_ioto_uniform_program_body_generate(
      AGX_CHIP_G17P, &desc, workspace(&f), &written, &copy_offset));
   CHECK(written == sizeof(expected));
   CHECK(copy_offset == 38);
   CHECK(memcmp(f.body, expected, sizeof(expected)) == 0);
   CHECK(f.body[written] == 0xa5);

   written = 27;
   copy_offset = 29;
   struct agx3_ioto_uniform_program_workspace ws = workspace(&f);
   ws.load_capacity = 0;
   CHECK(!agx3_ioto_uniform_program_body_generate(AGX_CHIP_G17P, &desc, ws,
                                                  &written, &copy_offset));
   CHECK(written == 27 && copy_offset == 29);
   return true;
}

static bool
test_bounds_and_cohorts(void)
{
   const struct agx3_ioto_uniform_range ranges[] = {
      {.start_halfs = 0, .size_halfs = 10, .address = 0x1000},
      {.start_halfs = 20, .size_halfs = 4, .address = 0x2000},
   };
   size_t bytes, chunks;
   for (size_t i = 0; i < ARRAY_SIZE(gpu0_targets); ++i) {
      CHECK(agx3_ioto_uniform_program_body_bound(
         gpu0_targets[i], ranges, ARRAY_SIZE(ranges), &bytes, &chunks));
      CHECK(chunks == 3 && bytes == 3 * 78);
   }
   for (size_t i = 0; i < ARRAY_SIZE(gpu1_targets); ++i) {
      CHECK(agx3_ioto_uniform_program_body_bound(
         gpu1_targets[i], ranges, ARRAY_SIZE(ranges), &bytes, &chunks));
      CHECK(chunks == 3 && bytes == 3 * 64);
   }

   CHECK(!agx3_ioto_uniform_program_body_bound(AGX_CHIP_G14G, ranges, 1, &bytes,
                                               &chunks));
   CHECK(!agx3_ioto_uniform_program_body_bound(AGX_CHIP_G15G, NULL, 1, &bytes,
                                               &chunks));
   CHECK(agx3_ioto_uniform_program_body_bound(AGX_CHIP_G15G, NULL, 0, &bytes,
                                              &chunks));
   CHECK(bytes == AGX_IOTO_LOAD_GPU0_ROUND_NOP_SIZE && chunks == 0);
   CHECK(agx3_ioto_uniform_program_body_bound(AGX_CHIP_G17G, NULL, 0, &bytes,
                                              &chunks));
   CHECK(bytes == 0 && chunks == 0);
   return true;
}

static bool
generate_fails_unchanged(enum agx_chip chip,
                         struct agx3_ioto_uniform_program_desc *desc,
                         struct agx3_ioto_uniform_program_workspace ws,
                         size_t capacity)
{
   uint8_t output[256], before[sizeof(output)];
   memset(output, 0x5a, sizeof(output));
   memcpy(before, output, sizeof(output));
   size_t written = 47;
   CHECK(!agx3_ioto_uniform_program_generate(chip, desc, ws, output, capacity,
                                             &written));
   CHECK(memcmp(output, before, sizeof(output)) == 0);
   CHECK(written == 47);
   return true;
}

static bool
test_fail_closed(void)
{
   struct fixture f;
   memset(&f, 0, sizeof(f));
   struct agx3_ioto_uniform_range range = {
      .start_halfs = 20,
      .size_halfs = 4,
      .address = 0x20,
   };
   struct agx3_ioto_uniform_program_desc desc =
      program_desc(AGX_CHIP_G15G, &range, 1);
   struct agx3_ioto_uniform_program_workspace ws = workspace(&f);

   range.start_halfs = 21;
   CHECK(generate_fails_unchanged(AGX_CHIP_G15G, &desc, ws, 256));
   range.start_halfs = 20;
   range.size_halfs = 3;
   CHECK(generate_fails_unchanged(AGX_CHIP_G15G, &desc, ws, 256));
   range.size_halfs = 4;
   range.address = 0x22;
   CHECK(generate_fails_unchanged(AGX_CHIP_G15G, &desc, ws, 256));
   range.address = 0x20;
   range.start_halfs = 510;
   range.size_halfs = 4;
   CHECK(generate_fails_unchanged(AGX_CHIP_G15G, &desc, ws, 256));

   range.start_halfs = 20;
   ws.body_capacity = 33;
   CHECK(generate_fails_unchanged(AGX_CHIP_G15G, &desc, ws, 256));
   ws = workspace(&f);
   ws.load_capacity = 0;
   CHECK(generate_fails_unchanged(AGX_CHIP_G15G, &desc, ws, 256));
   ws = workspace(&f);
   CHECK(generate_fails_unchanged(AGX_CHIP_G14G, &desc, ws, 256));
   CHECK(generate_fails_unchanged(AGX_CHIP_G15G, &desc, ws, 1));

   size_t written = 29;
   CHECK(!agx3_ioto_uniform_program_generate(
      AGX_CHIP_G15G, &desc, workspace(&f), f.body, sizeof(f.body), &written));
   CHECK(written == 29);
   written = 31;
   CHECK(!agx3_ioto_uniform_program_generate(
      AGX_CHIP_G15G, &desc, workspace(&f), f.loads, sizeof(f.loads), &written));
   CHECK(written == 31);
   written = 33;
   CHECK(!agx3_ioto_uniform_program_generate(
      AGX_CHIP_G15G, &desc, workspace(&f), f.sort, sizeof(f.sort), &written));
   CHECK(written == 33);
   written = 35;
   CHECK(!agx3_ioto_uniform_program_generate(
      AGX_CHIP_G15G, &desc, workspace(&f), f.cache, sizeof(f.cache), &written));
   CHECK(written == 35);

   ws = workspace(&f);
   ws.loads = (struct agx_ioto_inflight_load *)f.body;
   CHECK(generate_fails_unchanged(AGX_CHIP_G15G, &desc, ws, 256));
   ws = workspace(&f);
   ws.sort = ws.loads;
   CHECK(generate_fails_unchanged(AGX_CHIP_G15G, &desc, ws, 256));
   ws = workspace(&f);
   ws.cache = (struct agx_ioto_load_cache_entry *)ws.sort;
   CHECK(generate_fails_unchanged(AGX_CHIP_G15G, &desc, ws, 256));
   return true;
}

static bool
test_cprog_program_body(void)
{
   /* Visit high uniforms first. Packed TPS positions must follow the input
    * bitmap, regardless of range order or allocator round boundaries. */
   const struct agx3_ioto_uniform_range ranges[] = {
      {.start_halfs = 20, .size_halfs = 4, .address = 0x20},
      {.start_halfs = 0, .size_halfs = 4, .address = 0x1000},
   };
   const struct agx_ioto_cprog_inputs inputs = {
      .known = true,
      .uniform_halfs = {(UINT64_C(1) << 1) | (UINT64_C(1) << 3) |
                        (UINT64_C(1) << 20) | (UINT64_C(3) << 22)},
   };
   const uint16_t expected_tps[] = {10, 11, 12, 8, 9};
   for (unsigned rollover = 0; rollover < 2; ++rollover) {
      struct fixture baseline, f;
      struct agx3_ioto_uniform_program_desc desc =
         program_desc(AGX_CHIP_G17P, ranges, ARRAY_SIZE(ranges));
      /* Leave room for one four-dword allocation before the native
       * allocator limit. Workspace capacity alone never triggers rollover. */
      desc.initial_gpr_dwords = rollover ? 92 : 0;
      struct agx3_ioto_uniform_program_workspace ws = workspace(&baseline);
      size_t base_length, base_copy_offset;
      CHECK(agx3_ioto_uniform_program_body_generate(
         AGX_CHIP_G17P, &desc, ws, &base_length, &base_copy_offset));

      const struct agx_ioto_cprog_inputs empty = {.known = true};
      desc.cprog_inputs = &empty;
      ws = workspace(&f);
      size_t length, copy_offset;
      CHECK(agx3_ioto_uniform_program_body_generate(
         AGX_CHIP_G17P, &desc, ws, &length, &copy_offset));
      CHECK(length == base_length && copy_offset == base_copy_offset);
      CHECK(memcmp(f.body, baseline.body, length) == 0);

      desc.cprog_inputs = &inputs;
      size_t bound, chunks;
      CHECK(agx3_ioto_uniform_program_body_bound_with_inputs(
         AGX_CHIP_G17P, ranges, ARRAY_SIZE(ranges), &inputs, &bound, &chunks));
      CHECK(chunks == 2 && bound == 2 * 78 + 5 * 10);
      ws.body_capacity = bound - 1;
      length = 19;
      copy_offset = 23;
      CHECK(!agx3_ioto_uniform_program_body_generate(
         AGX_CHIP_G17P, &desc, ws, &length, &copy_offset));
      CHECK(length == 19 && copy_offset == 23);
      ws.body_capacity++;
      CHECK(agx3_ioto_uniform_program_body_generate(
         AGX_CHIP_G17P, &desc, ws, &length, &copy_offset));
      CHECK(length == base_length + 50 && length <= bound);
      CHECK(copy_offset == base_copy_offset + (rollover ? 30 : 0));

      size_t cursor = 0, base_cursor = 0;
      unsigned stores = 0, uniform_copies = 0;
      while (cursor < length) {
         struct agx_ioto_program_record record;
         size_t consumed;
         CHECK(agx_ioto_decode_program_record(
            AGX_CHIP_G17P, f.body + cursor, length - cursor, &record, &consumed));
         CHECK(consumed != 0);
         struct agx3_static_spill_store_desc store;
         if (agx3_static_spill_decode_store(AGX_ISA_ENCODING_AGX3B,
                                             f.body + cursor, consumed, &store)) {
            CHECK(stores < ARRAY_SIZE(expected_tps));
            CHECK(store.tps_half == expected_tps[stores++]);
            CHECK(store.width == AGX3_MEM_W16);
            CHECK(store.src_kill && store.slot == 6 && store.wait_mask == 0);
         } else {
            struct agx_ioto_program_record original;
            size_t original_size;
            CHECK(agx_ioto_decode_program_record(
               AGX_CHIP_G17P, baseline.body + base_cursor,
               base_length - base_cursor, &original, &original_size));
            CHECK(original_size == consumed && original.kind == record.kind);
            if (record.kind == AGX_IOTO_PROGRAM_RECORD_IOTO &&
                (record.ioto.kind == AGX_IOTO_KIND_COPY_EP0 ||
                 record.ioto.kind == AGX_IOTO_KIND_COPY_EP1)) {
               ++uniform_copies;
               /* Ignore only the witnessed source-lifetime flag. */
               record.bytes[2] &= ~0x08;
               original.bytes[2] &= ~0x08;
            }
            CHECK(memcmp(record.bytes, original.bytes, consumed) == 0);
            base_cursor += original_size;
         }
         cursor += consumed;
      }
      CHECK(stores == 5 && uniform_copies == 4 && base_cursor == base_length);
      size_t suffix_offset = SIZE_MAX;
      CHECK(agx_ioto_program_copy_suffix_offset(
         AGX_CHIP_G17P, f.body + copy_offset, length - copy_offset,
         &suffix_offset));
      CHECK(suffix_offset == 0);
      CHECK(agx3_ioto_uniform_program_generate(
         AGX_CHIP_G17P, &desc, ws, f.output, sizeof(f.output), &length));
   }
   return true;
}

static bool
test_cprog_program_failures(void)
{
   struct fixture f;
   const struct agx3_ioto_uniform_range ranges[] = {
      {.start_halfs = 20, .size_halfs = 4, .address = 0x20},
      {.start_halfs = 20, .size_halfs = 4, .address = 0x1000},
   };
   struct agx3_ioto_uniform_program_desc desc =
      program_desc(AGX_CHIP_G17P, ranges, 1);
   struct agx_ioto_cprog_inputs inputs = {
      .known = true, .uniform_halfs = {UINT64_C(1) << 20},
   };
   desc.cprog_inputs = &inputs;
   CHECK(generate_fails_unchanged(AGX_CHIP_G15G, &desc, workspace(&f), 256));
   inputs.known = false;
   CHECK(generate_fails_unchanged(AGX_CHIP_G17P, &desc, workspace(&f), 256));
   inputs.known = true;
   desc.fast_path_enabled = true;
   CHECK(generate_fails_unchanged(AGX_CHIP_G17P, &desc, workspace(&f), 256));
   desc.fast_path_enabled = false;
   desc.range_count = 2;
   CHECK(generate_fails_unchanged(AGX_CHIP_G17P, &desc, workspace(&f), 256));
   desc.range_count = 1;
   inputs.uniform_halfs[7] = UINT64_C(1) << 63;
   CHECK(generate_fails_unchanged(AGX_CHIP_G17P, &desc, workspace(&f), 256));

   size_t bound = 31, chunks = 37;
   CHECK(!agx3_ioto_uniform_program_body_bound_with_inputs(
      AGX_CHIP_G17P, ranges, 1, &inputs, &bound, &chunks));
   CHECK(bound == 31 && chunks == 37);
   inputs.uniform_halfs[0] = 0;
   const struct agx3_ioto_uniform_range last = {510, 2, 0x1000};
   desc.ranges = &last;
   CHECK(agx3_ioto_uniform_program_body_bound_with_inputs(
      AGX_CHIP_G17P, &last, 1, &inputs, &bound, &chunks));
   CHECK(bound == 78 + 10 && chunks == 1);
   size_t length, offset;
   CHECK(agx3_ioto_uniform_program_body_generate(
      AGX_CHIP_G17P, &desc, workspace(&f), &length, &offset));
   return true;
}

int
main(void)
{
   if (!test_exact_uniform_program() || !test_g17p_background_clear_body() ||
       !test_bounds_and_cohorts() || !test_fail_closed() ||
       !test_cprog_program_body() || !test_cprog_program_failures())
      return 1;

   printf("agx3_ioto_uniform_program: %u checks passed\n", checks);
   return 0;
}
