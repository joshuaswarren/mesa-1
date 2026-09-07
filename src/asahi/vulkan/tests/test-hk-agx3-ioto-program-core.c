/* Copyright 2026 Aurora Silicon */
/* SPDX-License-Identifier: MIT */

#include "hk_agx3_ioto_program_core.h"

#include "asahi/lib/agx_ioto_core.h"
#include "asahi/lib/agx_ioto_profile_core.h"
#include "asahi/lib/agx_ioto_state_core.h"
#include "asahi/lib/agx_tilebuffer.h"

#include <stdio.h>
#include <string.h>

static unsigned checks;

#define CHECK(x)                                                               \
   do {                                                                        \
      ++checks;                                                                \
      if (!(x)) {                                                              \
         fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x);                \
         return 1;                                                             \
      }                                                                        \
   } while (0)

#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))

static const enum agx_chip targets[] = {
   AGX_CHIP_G15G, AGX_CHIP_G15S, AGX_CHIP_G15C, AGX_CHIP_G15D,
   AGX_CHIP_G16G, AGX_CHIP_G16S, AGX_CHIP_G16C, AGX_CHIP_G17P,
   AGX_CHIP_G17G, AGX_CHIP_G17S, AGX_CHIP_G17C,
};

static bool
is_gpu1(enum agx_chip chip)
{
   return chip == AGX_CHIP_G17G || chip == AGX_CHIP_G17S ||
          chip == AGX_CHIP_G17C;
}

static size_t
find_unique_bytes(const uint8_t *haystack, size_t haystack_size,
                  const uint8_t *needle, size_t needle_size)
{
   size_t found = SIZE_MAX;

   for (size_t i = 0; i + needle_size <= haystack_size; ++i) {
      if (memcmp(haystack + i, needle, needle_size) != 0)
         continue;
      if (found != SIZE_MAX)
         return SIZE_MAX;
      found = i;
   }

   return found;
}

static struct hk_agx3_ioto_program_desc
make_desc(enum agx_chip chip)
{
   static const uint8_t body[] = {0x06, 0x00};
   struct hk_agx3_ioto_program_desc desc = {
      .program_va = UINT64_C(0x123456789a80),
      .body = body,
      .body_size = sizeof(body),
      .tail.type = AGX_IOTO_STATE_TAIL_TYPE0,
      .profile_payload0 = is_gpu1(chip) ? UINT16_C(0x8098)
                                       : UINT16_C(0x0555),
      .profile_payload1 = is_gpu1(chip) ? UINT16_C(0xa400) : 0,
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

static int
test_exact_target_order(void)
{
   for (size_t i = 0; i < ARRAY_SIZE(targets); ++i) {
      const enum agx_chip chip = targets[i];
      const struct hk_agx3_ioto_program_desc desc = make_desc(chip);
      const struct agx_ioto_profile_control profile = {
         .chip = chip,
         .payload0 = desc.profile_payload0,
         .payload1 = desc.profile_payload1,
      };
      uint8_t packed_profile[AGX_IOTO_PROFILE_CONTROL_MAX_SIZE];
      struct agx_ioto_state_tail_output packed_tail;
      CHECK(agx_ioto_profile_control_pack(&profile, packed_profile));
      CHECK(agx_ioto_state_tail_pack(chip, &desc.tail, &packed_tail));

      const size_t profile_size = agx_ioto_profile_control_size(chip);
      const size_t total = desc.body_size + profile_size + packed_tail.length +
                           AGX_IOTO_PROGRAM_TERMINATE_SIZE;
      uint8_t output[64];
      memset(output, 0xa5, sizeof(output));
      size_t written = SIZE_MAX;
      CHECK(hk_agx3_ioto_program_generate(chip, &desc, output, total,
                                          &written));
      CHECK(written == total);
      CHECK(memcmp(output, desc.body, desc.body_size) == 0);
      CHECK(memcmp(output + desc.body_size, packed_profile, profile_size) ==
            0);
      CHECK(memcmp(output + desc.body_size + profile_size, packed_tail.bytes,
                   packed_tail.length) == 0);
      static const uint8_t terminate[] = {0x0e, 0x00, 0x00, 0x00};
      CHECK(memcmp(output + total - sizeof(terminate), terminate,
                   sizeof(terminate)) == 0);
      CHECK(output[total] == 0xa5);
   }

   return 0;
}

static int
test_terminal_suffix_modes(void)
{
   static const uint8_t inline_code[] = {0x1b, 0x0c, 0x06, 0x00};

   for (size_t i = 0; i < ARRAY_SIZE(targets); ++i) {
      const enum agx_chip chip = targets[i];
      const struct hk_agx3_ioto_program_desc desc = make_desc(chip);
      struct agx_ioto_state_tail_output packed_tail;
      CHECK(agx_ioto_state_tail_pack(chip, &desc.tail, &packed_tail));
      const size_t prefix_size = desc.body_size +
                                 agx_ioto_profile_control_size(chip) +
                                 packed_tail.length;

      for (unsigned mode = 0; mode < 3; ++mode) {
         struct agx_ioto_preshader_desc preshader = {
            .external_va = AGX_IOTO_PRESHADER_NO_VA,
         };
         if (mode == 1) {
            preshader.code = inline_code;
            preshader.code_size = sizeof(inline_code);
         } else if (mode == 2) {
            preshader.force_external = true;
            preshader.request_prelude = true;
            preshader.external_va = UINT64_C(0x123456789abc);
         }

         struct agx_ioto_preshader_output suffix;
         CHECK(agx_ioto_preshader_emit(chip, &preshader, &suffix));
         const size_t total = prefix_size + suffix.length;
         uint8_t output[96];
         memset(output, 0xa5, sizeof(output));
         size_t written = SIZE_MAX;
         CHECK(hk_agx3_ioto_program_generate_no_pixrel(
            chip, &desc, &preshader, output, total, &written));
         CHECK(written == total);
         CHECK(memcmp(output + prefix_size, suffix.bytes, suffix.length) == 0);
         CHECK(output[total] == 0xa5);
         CHECK(suffix.mode == (mode == 0 ? AGX_IOTO_PRESHADER_MODE_ABSENT
                                        : mode == 1
                                             ? AGX_IOTO_PRESHADER_MODE_INLINE
                                             : AGX_IOTO_PRESHADER_MODE_EXTERNAL));
         if (mode != 0) {
            static const uint8_t absent[] = {0x0e, 0x00, 0x00, 0x00};
            CHECK(suffix.length < sizeof(absent) ||
                  memcmp(output + total - sizeof(absent), absent,
                         sizeof(absent)) != 0);
         }

         uint8_t failed[96];
         uint8_t before[sizeof(failed)];
         memset(failed, 0x5a, sizeof(failed));
         memcpy(before, failed, sizeof(failed));
         size_t failed_written = 29;
         CHECK(!hk_agx3_ioto_program_generate_no_pixrel(
            chip, &desc, &preshader, failed, total - 1, &failed_written));
         CHECK(memcmp(failed, before, sizeof(failed)) == 0);
         CHECK(failed_written == 29);
      }
   }

   return 0;
}

static int
expect_failure_unchanged(enum agx_chip chip,
                         const struct hk_agx3_ioto_program_desc *desc,
                         size_t capacity)
{
   uint8_t output[96];
   uint8_t before[sizeof(output)];
   memset(output, 0x5a, sizeof(output));
   memcpy(before, output, sizeof(output));
   size_t written = SIZE_MAX - 7;
   const size_t before_written = written;

   CHECK(!hk_agx3_ioto_program_generate(chip, desc, output, capacity,
                                        &written));
   CHECK(memcmp(output, before, sizeof(output)) == 0);
   CHECK(written == before_written);
   return 0;
}

static int
test_fail_closed(void)
{
   static const enum agx_chip rejected[] = {
      AGX_CHIP_G13G, AGX_CHIP_G14X, AGX_CHIP_G15P,
      AGX_CHIP_G16P, AGX_CHIP_G17A, AGX_CHIP_UNKNOWN,
   };
   struct hk_agx3_ioto_program_desc desc = make_desc(AGX_CHIP_G15G);
   for (size_t i = 0; i < ARRAY_SIZE(rejected); ++i)
      CHECK(expect_failure_unchanged(rejected[i], &desc, 96) == 0);

   static const uint8_t branch[] = {
      0x0f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   desc.body = branch;
   desc.body_size = sizeof(branch);
   CHECK(expect_failure_unchanged(AGX_CHIP_G15G, &desc, 96) == 0);

   desc = make_desc(AGX_CHIP_G15G);
   desc.program_va |= 1;
   CHECK(expect_failure_unchanged(AGX_CHIP_G15G, &desc, 96) == 0);
   desc = make_desc(AGX_CHIP_G15G);
   desc.profile_payload0 = UINT16_C(0x0800);
   CHECK(expect_failure_unchanged(AGX_CHIP_G15G, &desc, 96) == 0);
   desc = make_desc(AGX_CHIP_G15G);
   CHECK(expect_failure_unchanged(AGX_CHIP_G15G, &desc, 1) == 0);
   CHECK(expect_failure_unchanged(AGX_CHIP_G15G, NULL, 96) == 0);

   uint8_t output[96];
   size_t written = 9;
   CHECK(!hk_agx3_ioto_program_generate(AGX_CHIP_G15G, &desc, NULL,
                                         sizeof(output), &written));
   CHECK(written == 9);
   CHECK(!hk_agx3_ioto_program_generate(AGX_CHIP_G15G, &desc, output,
                                         sizeof(output), NULL));

   memset(output, 0x5a, sizeof(output));
   uint8_t before[sizeof(output)];
   memcpy(before, output, sizeof(output));
   written = 11;
   CHECK(!hk_agx3_ioto_program_generate_no_pixrel(
      AGX_CHIP_G15G, &desc, NULL, output, sizeof(output), &written));
   CHECK(memcmp(output, before, sizeof(output)) == 0);
   CHECK(written == 11);
   return 0;
}

static int
expect_g17s_minimal_hk_failure(
   enum agx_chip chip,
   const struct agx3_g17s_minimal_compute_program_desc *desc,
   size_t capacity)
{
   uint8_t output[AGX_IOTO_G17S_MINIMAL_COMPUTE_PROGRAM_SIZE + 1];
   uint8_t before[sizeof(output)];
   memset(output, 0x5a, sizeof(output));
   memcpy(before, output, sizeof(output));
   size_t written = 47;
   struct agx3_compute_program program;
   memset(&program, 0xa5, sizeof(program));
   uint8_t program_before[sizeof(program)];
   memcpy(program_before, &program, sizeof(program));

   CHECK(!hk_agx3_g17s_minimal_compute_program_prepare(
      chip, desc, output, capacity, &written, &program));
   CHECK(memcmp(output, before, sizeof(output)) == 0);
   CHECK(written == 47);
   CHECK(memcmp(&program, program_before, sizeof(program)) == 0);
   return 0;
}

static int
test_g17s_minimal_compute_admission(void)
{
   struct agx3_cdm_direct_template_sources sources;
   CHECK(agx3_cdm_direct_template_sources_compute(false, false, false,
                                                   &sources));
   const struct agx3_g17s_minimal_compute_program_desc desc = {
      .program_va = UINT64_C(0x100000040),
      .shader_va = UINT64_C(0x100000006c0),
      .root_uniform = {
         .user_buffer_binding = 1,
         .byte_offset = 0,
         .uniform_data_destination = 4,
         .uniform_data_count = 1,
         .temporary_base = 0,
      },
      .launch_sources = sources,
   };
   uint8_t direct[AGX_IOTO_G17S_MINIMAL_COMPUTE_PROGRAM_SIZE + 1];
   uint8_t bridged[sizeof(direct)];
   memset(direct, 0xa5, sizeof(direct));
   memset(bridged, 0x5a, sizeof(bridged));
   size_t direct_written = SIZE_MAX, bridged_written = SIZE_MAX;
   struct agx3_compute_program direct_program, bridged_program;

   CHECK(agx3_g17s_minimal_compute_program_prepare(
      &desc, direct, sizeof(direct), &direct_written, &direct_program));
   CHECK(hk_agx3_g17s_minimal_compute_program_prepare(
      AGX_CHIP_G17S, &desc, bridged, sizeof(bridged), &bridged_written,
      &bridged_program));
   CHECK(bridged_written == AGX_IOTO_G17S_MINIMAL_COMPUTE_PROGRAM_SIZE);
   CHECK(bridged_written == direct_written);
   CHECK(memcmp(bridged, direct, bridged_written) == 0);
   CHECK(bridged[bridged_written] == 0x5a);
   CHECK(bridged_program.chip == AGX_CHIP_G17S);
   CHECK(bridged_program.program_va == desc.program_va);
   CHECK(memcmp(&bridged_program.launch_template,
                &direct_program.launch_template,
                sizeof(bridged_program.launch_template)) == 0);

   _Alignas(AGX_IOTO_PROGRAM_ALIGNMENT)
      uint8_t mapped[AGX_IOTO_G17S_MINIMAL_COMPUTE_PROGRAM_SIZE + 1];
   memset(mapped, 0x3c, sizeof(mapped));
   struct agx3_compute_program_upload upload = {
      .cpu = mapped,
      .capacity = sizeof(mapped),
      .gpu_va = desc.program_va,
   };
   size_t mapped_written = SIZE_MAX;
   struct agx3_compute_program mapped_program;
   CHECK(hk_agx3_g17s_minimal_compute_program_upload_prepare(
      AGX_CHIP_G17S, &desc, &upload, &mapped_written, &mapped_program));
   CHECK(mapped_written == direct_written);
   CHECK(memcmp(mapped, direct, mapped_written) == 0);
   CHECK(mapped[mapped_written] == 0x3c);
   CHECK(mapped_program.chip == AGX_CHIP_G17S);
   CHECK(mapped_program.program_va == upload.gpu_va);

   const struct agx_ioto_g17s_minimal_compute_resources resources = {
      .gather_buffer_va = UINT64_C(0x100000e1ba0),
      .user_compute_argument_gpu_va_1 = UINT64_C(0x10000030100),
      .absolute_uniform_data_va =
         AGX_IOTO_G17S_MINIMAL_COMPUTE_WITNESS_ABSOLUTE_VA,
   };
   _Alignas(AGX_IOTO_PROGRAM_ALIGNMENT)
      uint8_t live_mapped[AGX_IOTO_G17S_MINIMAL_COMPUTE_PROGRAM_SIZE + 1];
   memset(live_mapped, 0x6d, sizeof(live_mapped));
   const struct agx3_compute_program_upload live_upload = {
      .cpu = live_mapped,
      .capacity = sizeof(live_mapped),
      .gpu_va = desc.program_va,
   };
   size_t live_written = SIZE_MAX;
   struct agx3_compute_program live_program;
   CHECK(hk_agx3_g17s_minimal_compute_program_live_upload_prepare(
      AGX_CHIP_G17S, &desc, &resources, &live_upload, &live_written,
      &live_program));
   CHECK(live_written == direct_written);
   CHECK(memcmp(live_mapped + 32, direct + 32, live_written - 32) == 0);
   CHECK(memcmp(live_mapped, direct, 32) != 0);
   CHECK(live_mapped[live_written] == 0x6d);
   CHECK(live_program.chip == AGX_CHIP_G17S &&
         live_program.program_va == live_upload.gpu_va);

   const struct agx3_compute_direct_grid grid = {
      .source = AGX3_CDM_DIRECT_GRID_THREADGROUP_COUNTS,
      .size = {2, 3, 4},
      .local_size = {8, 4, 2},
      .dimensions = 3,
   };
   uint8_t expected_launch[AGX3_CDM_DIRECT_LAUNCH_SIZE];
   CHECK(agx3_compute_program_pack_direct_launch(&live_program, &grid,
                                                  expected_launch));
   uint8_t dispatch[AGX3_G17S_MINIMAL_COMPUTE_DISPATCH_SIZE + 3];
   memset(dispatch, 0xb6, sizeof(dispatch));
   struct agx3_compute_dispatch_sink dispatch_sink = {
      .data = dispatch,
      .capacity = sizeof(dispatch),
      .length = 2,
   };
   CHECK(hk_agx3_g17s_minimal_compute_dispatch_append(
      &live_program, &grid, &dispatch_sink));
   CHECK(dispatch_sink.length ==
         2 + AGX3_G17S_MINIMAL_COMPUTE_DISPATCH_SIZE);
   CHECK(dispatch[0] == 0xb6 && dispatch[1] == 0xb6 &&
         dispatch[dispatch_sink.length] == 0xb6);
   CHECK(memcmp(dispatch + 2, expected_launch,
                AGX3_CDM_DIRECT_LAUNCH_SIZE) == 0);
   static const uint8_t strong_barrier[] = {0x60, 0x15, 0x00, 0x60};
   CHECK(memcmp(dispatch + 2 + AGX3_CDM_DIRECT_LAUNCH_SIZE, strong_barrier,
                sizeof(strong_barrier)) == 0);

   uint8_t dispatch_before[sizeof(dispatch)];
   memcpy(dispatch_before, dispatch, sizeof(dispatch));
   struct agx3_compute_dispatch_sink failed_sink = {
      .data = dispatch,
      .capacity = sizeof(dispatch),
      .length = 1,
   };
   CHECK(!hk_agx3_g17s_minimal_compute_dispatch_append(
      &direct_program, &grid, &failed_sink));
   CHECK(failed_sink.length == 1 &&
         memcmp(dispatch, dispatch_before, sizeof(dispatch)) == 0);
   struct agx3_compute_program sibling_program = live_program;
   sibling_program.chip = AGX_CHIP_G17G;
   CHECK(!hk_agx3_g17s_minimal_compute_dispatch_append(
      &sibling_program, &grid, &failed_sink));
   CHECK(failed_sink.length == 1 &&
         memcmp(dispatch, dispatch_before, sizeof(dispatch)) == 0);
   failed_sink.capacity = AGX3_G17S_MINIMAL_COMPUTE_DISPATCH_SIZE;
   CHECK(!hk_agx3_g17s_minimal_compute_dispatch_append(
      &live_program, &grid, &failed_sink));
   CHECK(failed_sink.length == 1 &&
         memcmp(dispatch, dispatch_before, sizeof(dispatch)) == 0);

   uint8_t live_before[sizeof(live_mapped)];
   memcpy(live_before, live_mapped, sizeof(live_mapped));
   const struct agx3_compute_program live_program_before = live_program;

   uint8_t mapped_before[sizeof(mapped)];
   memcpy(mapped_before, mapped, sizeof(mapped));
   const struct agx3_compute_program mapped_program_before = mapped_program;

   static const enum agx_chip rejected[] = {
      AGX_CHIP_G15G, AGX_CHIP_G15S, AGX_CHIP_G15C, AGX_CHIP_G15D,
      AGX_CHIP_G16G, AGX_CHIP_G16S, AGX_CHIP_G16C, AGX_CHIP_G17P,
      AGX_CHIP_G17G, AGX_CHIP_G17C, AGX_CHIP_UNKNOWN,
   };
   for (size_t i = 0; i < ARRAY_SIZE(rejected); ++i) {
      CHECK(expect_g17s_minimal_hk_failure(rejected[i], &desc,
                                            sizeof(bridged)) == 0);
      mapped_written = 47;
      CHECK(!hk_agx3_g17s_minimal_compute_program_upload_prepare(
         rejected[i], &desc, &upload, &mapped_written, &mapped_program));
      CHECK(memcmp(mapped, mapped_before, sizeof(mapped)) == 0 &&
            mapped_written == 47);
      CHECK(memcmp(&mapped_program, &mapped_program_before,
                   sizeof(mapped_program)) == 0);
      live_written = 47;
      CHECK(!hk_agx3_g17s_minimal_compute_program_live_upload_prepare(
         rejected[i], &desc, &resources, &live_upload, &live_written,
         &live_program));
      CHECK(memcmp(live_mapped, live_before, sizeof(live_mapped)) == 0 &&
            live_written == 47);
      CHECK(memcmp(&live_program, &live_program_before,
                   sizeof(live_program)) == 0);
   }

   struct agx3_compute_program_upload invalid_upload = upload;
   invalid_upload.gpu_va += AGX_IOTO_PROGRAM_ALIGNMENT;
   mapped_written = 47;
   CHECK(!hk_agx3_g17s_minimal_compute_program_upload_prepare(
      AGX_CHIP_G17S, &desc, &invalid_upload, &mapped_written,
      &mapped_program));
   CHECK(memcmp(mapped, mapped_before, sizeof(mapped)) == 0 &&
         mapped_written == 47);
   CHECK(memcmp(&mapped_program, &mapped_program_before,
                sizeof(mapped_program)) == 0);

   struct agx_ioto_g17s_minimal_compute_resources invalid_resources =
      resources;
   invalid_resources.absolute_uniform_data_va = 0;
   live_written = 47;
   CHECK(!hk_agx3_g17s_minimal_compute_program_live_upload_prepare(
      AGX_CHIP_G17S, &desc, &invalid_resources, &live_upload, &live_written,
      &live_program));
   CHECK(memcmp(live_mapped, live_before, sizeof(live_mapped)) == 0 &&
         live_written == 47);
   CHECK(memcmp(&live_program, &live_program_before,
                sizeof(live_program)) == 0);

   struct agx3_g17s_minimal_compute_program_desc invalid = desc;
   invalid.root_uniform.user_buffer_binding = 0;
   CHECK(expect_g17s_minimal_hk_failure(AGX_CHIP_G17S, &invalid,
                                         sizeof(bridged)) == 0);
   CHECK(expect_g17s_minimal_hk_failure(
            AGX_CHIP_G17S, &desc,
            AGX_IOTO_G17S_MINIMAL_COMPUTE_PROGRAM_SIZE - 1) == 0);
   CHECK(expect_g17s_minimal_hk_failure(AGX_CHIP_G17S, NULL,
                                         sizeof(bridged)) == 0);
   return 0;
}

static int
test_g17p_dynamic_compute_program(void)
{
   struct agx3_cdm_direct_template_sources sources;
   CHECK(agx3_cdm_direct_template_sources_compute(true, false, false,
                                                   &sources));
   const struct agx3_ioto_uniform_range ranges[] = {
      {
         .start_halfs = 0,
         .size_halfs = 12,
         .address = UINT64_C(0x10000024000),
      },
      {
         .start_halfs = 12,
         .size_halfs = 8,
         .address = UINT64_C(0x10000030000),
      },
   };
   const struct hk_agx3_g17p_compute_program_desc desc = {
      .program_va = UINT64_C(0x100000b0000),
      .shader_va = UINT64_C(0x10000000100),
      .uniform_ranges = ranges,
      .uniform_range_count = ARRAY_SIZE(ranges),
      .shader_binary_size = 96,
      .nr_gprs = 8,
      .compiler_d_valid = true,
      .compiler_d = 32,
      .threadgroup_memory_size_B = 4096,
      .launch_sources = sources,
   };
   struct hk_agx3_g17p_compute_program program;
   memset(&program, 0xa5, sizeof(program));
   CHECK(hk_agx3_g17p_compute_program_generate(AGX_CHIP_G17P, &desc,
                                               &program));
   CHECK(program.length > AGX_IOTO_G17P_NO_RESOURCE_PROGRAM_SIZE);
   CHECK(program.length < sizeof(program.bytes));
   CHECK(program.bytes[program.length - AGX_IOTO_PROGRAM_TERMINATE_SIZE] ==
         0x0e);
   CHECK(program.launch.chip == AGX_CHIP_G17P);
   CHECK(program.launch.program_va == desc.program_va);
   CHECK(program.launch.shader_va == desc.shader_va);
   CHECK(program.launch.g17p_native_compute_live_bound);

   struct agx_ioto_state_tail_output expected_tail;
   /* The direct-dispatch kind. This expectation used to say GRID_HELPER,
    * because it was written against the size-derived f1 that this shader's
    * 118-byte length happened to land on -- a fourth self-pinned expectation
    * rather than a witness. LDSHDR f1 is a program-kind selector (permuted
    * through agx_ioto_e4), so a direct dispatch announces the direct kind. */
   CHECK(agx_ioto_g17p_runtime_compute_tail_build_shared(
      desc.shader_va, AGX_IOTO_G17P_COMPUTE_LDSHDR_F1_DIRECT_ADD3,
      desc.threadgroup_memory_size_B, &expected_tail));
   /* The runtime builder leaves SPECTPR zero (the hand-written add3 witness's
    * footprint). A compiler-produced shader must declare its own, exactly as
    * the graphics path does, so the expectation carries it too. */
   struct agx_ioto_state expected_spectpr;
   CHECK(agx_ioto_g17p_compiler_spectpr_build(desc.scratch_size,
                                              desc.compiler_d,
                                              &expected_spectpr));
   CHECK(agx_ioto_state_pack_spectpr(
      AGX_CHIP_G17P, &expected_spectpr,
      expected_tail.bytes + AGX_IOTO_SPECLM_SIZE + AGX_IOTO_LDSHDR_SIZE,
      AGX_IOTO_SPECTPR_SIZE));
   size_t tail_matches = 0;
   for (size_t i = 0; i + expected_tail.length <= program.length; ++i) {
      if (memcmp(program.bytes + i, expected_tail.bytes,
                 expected_tail.length) == 0)
         tail_matches++;
   }
   CHECK(tail_matches == 1);

   const struct agx3_compute_direct_grid grid = {
      .source = AGX3_CDM_DIRECT_GRID_GLOBAL_THREADS,
      .size = {96, 4, 1},
      .local_size = {32, 2, 1},
      .dimensions = 2,
   };
   static const uint8_t expected_dispatch[] = {
      0x00, 0x00, 0x08, 0x00, 0x40, 0x00, 0x00, 0x01,
      0x00, 0x2c, 0x00, 0x00, 0x01, 0x00, 0x00, 0x40,
      0x60, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
      0x01, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00,
      0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
      0x60, 0x01, 0x00, 0x60,
   };
   uint8_t dispatch[AGX3_G17P_NATIVE_COMPUTE_DISPATCH_SIZE + 1];
   memset(dispatch, 0xb6, sizeof(dispatch));
   struct agx3_compute_dispatch_sink sink = {
      .data = dispatch,
      .capacity = sizeof(dispatch),
   };
   CHECK(hk_agx3_g17p_native_compute_dispatch_append(
      &program.launch, &grid, &sink));
   CHECK(sink.length == sizeof(expected_dispatch));
   CHECK(memcmp(dispatch, expected_dispatch, sizeof(expected_dispatch)) == 0);
   CHECK(dispatch[sink.length] == 0xb6);

   const struct hk_agx3_g17p_compute_program before = program;
   struct hk_agx3_g17p_compute_program_desc invalid = desc;
   invalid.scratch_size = 16;
   CHECK(!hk_agx3_g17p_compute_program_generate(AGX_CHIP_G17P, &invalid,
                                                &program));
   CHECK(memcmp(&program, &before, sizeof(program)) == 0);
   invalid = desc;
   invalid.compiler_d_valid = false;
   CHECK(!hk_agx3_g17p_compute_program_generate(AGX_CHIP_G17P, &invalid,
                                                &program));
   CHECK(memcmp(&program, &before, sizeof(program)) == 0);
   invalid = desc;
   invalid.uniform_ranges = NULL;
   CHECK(!hk_agx3_g17p_compute_program_generate(AGX_CHIP_G17P, &invalid,
                                                &program));
   CHECK(memcmp(&program, &before, sizeof(program)) == 0);
   invalid = desc;
   invalid.threadgroup_memory_size_B = 257;
   CHECK(!hk_agx3_g17p_compute_program_generate(AGX_CHIP_G17P, &invalid,
                                                &program));
   CHECK(memcmp(&program, &before, sizeof(program)) == 0);
   CHECK(!hk_agx3_g17p_compute_program_generate(AGX_CHIP_G17S, &desc,
                                                &program));
   CHECK(memcmp(&program, &before, sizeof(program)) == 0);
   return 0;
}

static int
test_g17p_direct_add3_compute_bridge(void)
{
   struct agx3_cdm_direct_template_sources sources;
   CHECK(agx3_cdm_direct_template_sources_compute(false, false, false,
                                                   &sources));
   const struct agx3_g17p_direct_add3_program_desc desc = {
      .program_va = AGX_IOTO_G17P_DIRECT_ADD3_WITNESS_PROGRAM_VA,
      .shader_va = AGX_IOTO_G17P_DIRECT_ADD3_WITNESS_SHADER_VA,
      .launch_sources = sources,
   };
   _Alignas(AGX_IOTO_PROGRAM_ALIGNMENT)
      uint8_t direct[AGX_IOTO_G17P_DIRECT_ADD3_PROGRAM_SIZE + 1];
   _Alignas(AGX_IOTO_PROGRAM_ALIGNMENT)
      uint8_t bridged[AGX_IOTO_G17P_DIRECT_ADD3_PROGRAM_SIZE + 1];
   memset(direct, 0xa5, sizeof(direct));
   memset(bridged, 0x5a, sizeof(bridged));
   const struct agx3_compute_program_upload direct_upload = {
      .cpu = direct,
      .capacity = sizeof(direct),
      .gpu_va = desc.program_va,
   };
   const struct agx3_compute_program_upload bridged_upload = {
      .cpu = bridged,
      .capacity = sizeof(bridged),
      .gpu_va = desc.program_va,
   };
   size_t direct_written = SIZE_MAX, bridged_written = SIZE_MAX;
   struct agx3_compute_program direct_program, bridged_program;
   CHECK(agx3_g17p_direct_add3_program_upload_prepare(
      AGX_CHIP_G17P, &desc, &direct_upload, &direct_written,
      &direct_program));
   CHECK(hk_agx3_g17p_direct_add3_program_upload_prepare(
      AGX_CHIP_G17P, &desc, &bridged_upload, &bridged_written,
      &bridged_program));
   CHECK(bridged_written == direct_written &&
         bridged_written == AGX_IOTO_G17P_DIRECT_ADD3_PROGRAM_SIZE);
   CHECK(memcmp(bridged, direct, bridged_written) == 0);
   CHECK(bridged[bridged_written] == 0x5a);
   CHECK(bridged_program.chip == AGX_CHIP_G17P);
   CHECK(bridged_program.program_va == desc.program_va);
   CHECK(bridged_program.shader_va == desc.shader_va);
   CHECK(bridged_program.g17p_direct_add3_live_admitted);

   const struct agx3_compute_direct_grid grid = {
      .source = AGX3_CDM_DIRECT_GRID_GLOBAL_THREADS,
      .size = {64, 1, 1},
      .local_size = {32, 1, 1},
      .dimensions = 1,
   };
   static const uint8_t expected_dispatch[] = {
      0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x01,
      0x00, 0x2a, 0x00, 0x00, 0x01, 0x00, 0x00, 0x40,
      0x40, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
      0x01, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00,
      0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
      0x60, 0x01, 0x00, 0x60,
   };
   uint8_t dispatch[AGX3_G17P_DIRECT_ADD3_DISPATCH_SIZE + 1];
   memset(dispatch, 0xb6, sizeof(dispatch));
   struct agx3_compute_dispatch_sink sink = {
      .data = dispatch,
      .capacity = sizeof(dispatch),
   };
   CHECK(hk_agx3_g17p_direct_add3_dispatch_append(
      &bridged_program, &grid, &sink));
   CHECK(sink.length == sizeof(expected_dispatch));
   CHECK(memcmp(dispatch, expected_dispatch, sizeof(expected_dispatch)) == 0);
   CHECK(dispatch[sink.length] == 0xb6);

   uint8_t dispatch_before[sizeof(dispatch)];
   memcpy(dispatch_before, dispatch, sizeof(dispatch));
   struct agx3_compute_dispatch_sink failed_sink = {
      .data = dispatch,
      .capacity = sizeof(dispatch),
   };
   struct agx3_compute_program sibling = bridged_program;
   sibling.chip = AGX_CHIP_G17S;
   CHECK(!hk_agx3_g17p_direct_add3_dispatch_append(
      &sibling, &grid, &failed_sink));
   CHECK(failed_sink.length == 0 &&
         memcmp(dispatch, dispatch_before, sizeof(dispatch)) == 0);

   uint8_t bridged_before[sizeof(bridged)];
   memcpy(bridged_before, bridged, sizeof(bridged));
   const struct agx3_compute_program program_before = bridged_program;
   bridged_written = 47;
   CHECK(!hk_agx3_g17p_direct_add3_program_upload_prepare(
      AGX_CHIP_G17S, &desc, &bridged_upload, &bridged_written,
      &bridged_program));
   CHECK(bridged_written == 47 &&
         memcmp(bridged, bridged_before, sizeof(bridged)) == 0);
   CHECK(memcmp(&bridged_program, &program_before,
                sizeof(bridged_program)) == 0);
   return 0;
}

static int
test_g17p_no_resource_program_bridge(void)
{
   static const uint8_t exact_tail[AGX_IOTO_STATE_TAIL_TYPE0_SIZE] = {
      0x77, 0x00, 0x2a, 0x43, 0x4b, 0x00, 0x00, 0x00, 0x77, 0x01,
      0x2a, 0x01, 0x00, 0x00, 0x00, 0x02, 0x04, 0x00, 0xf7, 0x00,
      0x2a, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   struct agx_ioto_state_tail_output tail = {
      .length = sizeof(exact_tail),
   };
   memcpy(tail.bytes, exact_tail, sizeof(exact_tail));

   uint8_t shared[AGX_IOTO_G17P_NO_RESOURCE_PROGRAM_SIZE];
   uint8_t bridged[sizeof(shared) + 1];
   memset(bridged, 0xa5, sizeof(bridged));
   size_t shared_written = 0, bridged_written = 0;
   CHECK(agx_ioto_g17p_no_resource_program_compose(
      UINT64_C(0x10001990240), &tail, shared, sizeof(shared), &shared_written));
   CHECK(hk_agx3_g17p_no_resource_program_generate(
      AGX_CHIP_G17P, UINT64_C(0x10001990240), &tail, bridged, sizeof(shared),
      &bridged_written));
   CHECK(bridged_written == shared_written &&
         memcmp(bridged, shared, shared_written) == 0);
   CHECK(bridged[bridged_written] == 0xa5);

   uint8_t before[sizeof(bridged)];
   memcpy(before, bridged, sizeof(bridged));
   bridged_written = 31;
   CHECK(!hk_agx3_g17p_no_resource_program_generate(
      AGX_CHIP_G17S, UINT64_C(0x10001990240), &tail, bridged, sizeof(shared),
      &bridged_written));
   CHECK(memcmp(bridged, before, sizeof(bridged)) == 0 &&
         bridged_written == 31);
   return 0;
}

static int
test_g17p_basic_graphics_program(void)
{
   static const uint8_t expected_fragment[] = {
      0x77, 0x00, 0x2a, 0x43, 0x4b, 0x00, 0x00, 0x00, 0x77, 0x01, 0x2a, 0x01,
      0x00, 0x00, 0x00, 0x02, 0x04, 0x00, 0xf7, 0x00, 0x2a, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x1c, 0x80, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x14, 0x81, 0x11, 0x06, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x80, 0x02, 0x00,
      0x04, 0x00, 0x00, 0x00, 0x9f, 0x11, 0x54, 0x00, 0x02, 0x00, 0x08, 0xa8,
      0x10, 0x05, 0x1c, 0x80, 0x02, 0x00, 0x04, 0x00, 0x00, 0x00, 0x0f, 0x12,
      0x54, 0x00, 0x4c, 0x00, 0x07, 0x02, 0x54, 0xf3, 0x02, 0x00, 0x0e, 0x00,
      0x00, 0x00,
   };
   CHECK(sizeof(expected_fragment) ==
         AGX_IOTO_G17P_FRAGMENT_NO_RESOURCE_PROGRAM_SIZE);
   /* Both the valid cold G1a vary0 tail and the matched applegpu_g17p
    * constant-colour/no-varying probe leave compiler ProgramVariant field 38
    * absent. FragmentProgramVariantESLState therefore supplies D=0, making
    * the complete SPECTPR payload above zero. */
   const struct hk_agx3_g17p_graphics_program_desc fragment = {
      .program_va = UINT64_C(0x10001990240),
      .shader_va = UINT64_C(0x10000000080),
      .shader_binary_size = 54,
      .scratch_size = 0,
      .nr_gprs = 8,
      .fragment = true,
   };
   struct hk_agx3_g17p_graphics_program program;
   CHECK(hk_agx3_g17p_basic_graphics_program_generate(AGX_CHIP_G17P, &fragment,
                                                      &program));
   CHECK(program.length == sizeof(expected_fragment));
   CHECK(memcmp(program.bytes, expected_fragment, sizeof(expected_fragment)) ==
         0);

   const struct hk_agx3_g17p_graphics_program_desc vertex = {
      .program_va = UINT64_C(0x10000128000),
      .shader_va = UINT64_C(0x10000000000),
      .shader_binary_size = 74,
      .scratch_size = 0,
      .nr_gprs = 14,
      .fragment = false,
   };
   CHECK(hk_agx3_g17p_basic_graphics_program_generate(AGX_CHIP_G17P, &vertex,
                                                      &program));
   CHECK(program.length == AGX_IOTO_G17P_NO_RESOURCE_PROGRAM_SIZE);
   /* The 74-byte shader spans two 64-byte fetch lines, matching native f1=1
    * and encoded LDSHDR word 4 = 0x0000. */
   static const uint8_t vertex_tail[] = {
      0x77, 0x00, 0x2a, 0x41, 0x00, 0x00, 0x00, 0x00, 0x77, 0x01,
      0x2a, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0xf7, 0x00,
      0x2a, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   CHECK(memcmp(program.bytes, vertex_tail, sizeof(vertex_tail)) == 0);

   /* The exact Apple 88-byte float2 VS has four 32-bit staging values and
    * compiler ProgramVariant field 38 is 16. Preserve the distinct domains:
    * metadata D=16 yields SPECTPR f22=4, whereas using
    * physical GPR halfs=8 here would incorrectly yield f22=2.
    */
   struct hk_agx3_g17p_graphics_program_desc apple_vertex = vertex;
   apple_vertex.shader_binary_size = 88;
   apple_vertex.nr_gprs = 8;
   apple_vertex.compiler_d_valid = true;
   apple_vertex.compiler_d = 16;
   const struct hk_agx3_g17p_graphics_program normal_vertex = program;
   CHECK(hk_agx3_g17p_basic_graphics_program_generate(
      AGX_CHIP_G17P, &apple_vertex, &program));
   CHECK(program.length == normal_vertex.length);
   CHECK(memcmp(program.bytes, normal_vertex.bytes, program.length) == 0);
   apple_vertex.nr_gprs = 4;
   CHECK(hk_agx3_g17p_basic_graphics_program_generate(
      AGX_CHIP_G17P, &apple_vertex, &program));
   CHECK(memcmp(program.bytes, normal_vertex.bytes, program.length) == 0);
   apple_vertex.nr_gprs = 8;
   apple_vertex.compiler_d_valid = false;
   CHECK(hk_agx3_g17p_basic_graphics_program_generate(
      AGX_CHIP_G17P, &apple_vertex, &program));
   static const uint8_t old_d_from_gprs[] = {
      0xf7, 0x00, 0x2a, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   CHECK(memcmp(program.bytes + 18, old_d_from_gprs,
                sizeof(old_d_from_gprs)) == 0);

   static const struct agx3_ioto_uniform_range descriptor_root = {
      .start_halfs = 20,
      .size_halfs = 4,
      .address = 0x20,
   };
   static const uint8_t expected_resource_prefix[] = {
      0x0c, 0x20, 0x1c, 0x00, 0x67, 0x00, 0x54, 0x04, 0x00, 0x00, 0x00, 0x00,
      0x19, 0x00, 0x00, 0x40, 0x22, 0x00, 0x06, 0x08, 0x06, 0x00, 0x06, 0x00,
      0x06, 0x00, 0xab, 0x04, 0x09, 0x04, 0xbb, 0x06, 0x09, 0x04,
   };
   struct hk_agx3_g17p_graphics_program_desc resource = fragment;
   resource.uniform_ranges = &descriptor_root;
   resource.uniform_range_count = 1;
   CHECK(hk_agx3_g17p_basic_graphics_program_generate(AGX_CHIP_G17P,
                                                      &resource, &program));
   CHECK(program.length > AGX_IOTO_G17P_NO_RESOURCE_PROGRAM_SIZE);
   CHECK(program.length <= HK_AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE);
   const size_t pre_copy_size = sizeof(expected_resource_prefix) - 8;
   static const uint8_t expected_resource_tail[] = {
      0x77, 0x00, 0x2a, 0x43, 0x4b, 0x00, 0x00, 0x00, 0x77, 0x01,
      0x2a, 0x01, 0x00, 0x00, 0x00, 0x02, 0x04, 0x00, 0xf7, 0x00,
      0x2a, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   CHECK(memcmp(program.bytes, expected_resource_prefix, pre_copy_size) == 0);
   CHECK(memcmp(program.bytes + pre_copy_size,
                expected_resource_tail, sizeof(expected_resource_tail)) == 0);
   static const uint8_t fragment_termination[] = {
      0x07, 0x02, 0x54, 0xf3, 0x02, 0x00, 0x0e, 0x00, 0x00, 0x00,
   };
   CHECK(memcmp(program.bytes + program.length - sizeof(fragment_termination),
                fragment_termination, sizeof(fragment_termination)) == 0);

   static const struct agx3_ioto_uniform_range two_attribs_and_set[] = {
      {.start_halfs = 0, .size_halfs = 8, .address = 0x1000},
      {.start_halfs = 8, .size_halfs = 4, .address = 0x2000},
      {.start_halfs = 24, .size_halfs = 8, .address = 0x3000},
   };
   resource = vertex;
   resource.uniform_ranges = two_attribs_and_set;
   resource.uniform_range_count = ARRAY_SIZE(two_attribs_and_set);
   resource.shader_binary_size = 344;
   CHECK(hk_agx3_g17p_basic_graphics_program_generate(AGX_CHIP_G17P,
                                                      &resource, &program));
   CHECK(program.length > AGX_IOTO_G17P_NO_RESOURCE_PROGRAM_SIZE);
   CHECK(program.length <= HK_AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE);
   CHECK(program.bytes[program.length - AGX_IOTO_PROGRAM_TERMINATE_SIZE] ==
         0x0e);

   const struct hk_agx3_g17p_graphics_program three_range_program = program;
   for (unsigned i = 0; i < ARRAY_SIZE(two_attribs_and_set); ++i) {
      struct agx3_ioto_uniform_range changed[ARRAY_SIZE(two_attribs_and_set)];
      memcpy(changed, two_attribs_and_set, sizeof(changed));
      changed[i].address += 0x40;
      resource.uniform_ranges = changed;
      CHECK(hk_agx3_g17p_basic_graphics_program_generate(
         AGX_CHIP_G17P, &resource, &program));
      CHECK(program.length == three_range_program.length &&
            memcmp(program.bytes, three_range_program.bytes,
                   program.length) != 0);
   }
   resource.uniform_ranges = two_attribs_and_set;

   /* Full attribute pointers + clamps, descriptor root, and constant pool.
    * Every range must affect actual native CProg bytes independently. */
   const struct agx3_ioto_uniform_range full_vertex[] = {
      {.start_halfs = 0, .size_halfs = 64, .address = 0x1000},
      {.start_halfs = 64, .size_halfs = 32, .address = 0x2000},
      {.start_halfs = 108, .size_halfs = 36, .address = 0x3000},
      {.start_halfs = 144, .size_halfs = 128, .address = 0x4000},
   };
   resource.uniform_ranges = full_vertex;
   resource.uniform_range_count = ARRAY_SIZE(full_vertex);
   size_t bound = 0;
   CHECK(hk_agx3_g17p_graphics_program_bound(AGX_CHIP_G17P, full_vertex,
                                              ARRAY_SIZE(full_vertex), &bound));
   CHECK(hk_agx3_g17p_basic_graphics_program_generate(
      AGX_CHIP_G17P, &resource, &program));
   CHECK(program.length <= bound);
   const struct hk_agx3_g17p_graphics_program full_program = program;
   unsigned copies = 0, copy_then_load = 0;
   bool saw_copy = false;
   const size_t loader_end = program.length - AGX_IOTO_G17P_DEFAULT_PROFILE_SUFFIX_SIZE -
                             AGX_IOTO_PROGRAM_TERMINATE_SIZE;
   for (size_t at = 0; at < loader_end;) {
      struct agx_ioto_program_record record;
      size_t consumed = 0;
      CHECK(agx_ioto_decode_program_record(AGX_CHIP_G17P, program.bytes + at,
                                             loader_end - at, &record,
                                             &consumed));
      CHECK(consumed > 0);
      if (record.kind == AGX_IOTO_PROGRAM_RECORD_IOTO &&
          (record.ioto.kind == AGX_IOTO_KIND_COPY_EP0 ||
           record.ioto.kind == AGX_IOTO_KIND_COPY_EP1)) {
         copies++;
         saw_copy = true;
      }
      if (record.kind == AGX_IOTO_PROGRAM_RECORD_GPU0_INLINE_LDDEV && saw_copy)
         copy_then_load++;
      at += consumed;
   }
   fprintf(stderr, "full-interface native COPY count=%u, copies before later loads=%u\n", copies, copy_then_load);
   CHECK(copies == (64 + 32 + 36 + 128) / 2);
   CHECK(copy_then_load > 0);

   for (unsigned r = 0; r < ARRAY_SIZE(full_vertex); ++r) {
      struct agx3_ioto_uniform_range changed[ARRAY_SIZE(full_vertex)];
      memcpy(changed, full_vertex, sizeof(changed));
      changed[r].address += 0x40;
      resource.uniform_ranges = changed;
      CHECK(hk_agx3_g17p_basic_graphics_program_generate(
         AGX_CHIP_G17P, &resource, &program));
      CHECK(program.length == full_program.length);
      CHECK(memcmp(program.bytes, full_program.bytes, program.length) != 0);
   }
   resource.uniform_ranges = full_vertex;
   size_t untouched = 123;
   CHECK(!hk_agx3_g17p_graphics_program_bound(AGX_CHIP_G17S, full_vertex,
                                               ARRAY_SIZE(full_vertex), &untouched));
   CHECK(untouched == 123);

   struct hk_agx3_g17p_graphics_program_desc boundary = vertex;
   static const struct {
      uint32_t size;
      uint8_t encoded_word4_lo;
   } proven_fetch_lines[] = {
      {1, 0x04},   {64, 0x04},  {65, 0x00},  {128, 0x00},
      {129, 0x00}, {192, 0x00}, {193, 0x02}, {256, 0x02},
      {257, 0x02}, {320, 0x02}, {321, 0x06}, {344, 0x06},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(proven_fetch_lines); ++i) {
      boundary.shader_binary_size = proven_fetch_lines[i].size;
      CHECK(hk_agx3_g17p_basic_graphics_program_generate(
         AGX_CHIP_G17P, &boundary, &program));
      CHECK(program.bytes[16] == proven_fetch_lines[i].encoded_word4_lo &&
            program.bytes[17] == 0x00);
   }
   const struct hk_agx3_g17p_graphics_program before = program;
   struct hk_agx3_g17p_graphics_program_desc invalid = vertex;
   invalid.shader_binary_size = 0;
   CHECK(!hk_agx3_g17p_basic_graphics_program_generate(AGX_CHIP_G17P, &invalid,
                                                       &program));
   CHECK(memcmp(&program, &before, sizeof(program)) == 0);
   CHECK(!hk_agx3_g17p_basic_graphics_program_generate(AGX_CHIP_G17S, &vertex,
                                                       &program));
   invalid = resource;
   invalid.uniform_range_count = HK_AGX3_G17P_GRAPHICS_MAX_RANGES + 1;
   CHECK(!hk_agx3_g17p_basic_graphics_program_generate(AGX_CHIP_G17P, &invalid,
                                                       &program));
   CHECK(memcmp(&program, &before, sizeof(program)) == 0);
   return 0;
}

/* Independently decode the actual emitted LDIMMs, rather than regenerating
 * expected bytes with the encoder under test. No descriptor load is allowed
 * in this prefix: the value published to uniforms is the raw table VA.
 */
static int
check_eot_pointer_prefix(const struct hk_agx3_g17p_rt0_eot_program *program,
                         uint64_t expected, uint8_t expected_offset,
                         size_t *prefix_size)
{
   uint64_t pointer = 0;
   size_t offset = 0;
   for (unsigned i = 0; i < 3; ++i) {
      struct agx_ioto_record record;
      size_t consumed;
      CHECK(agx_ioto_decode_one(AGX_CHIP_G17P, program->bytes + offset,
                                  program->length - offset, &record, &consumed));
      CHECK(record.kind == AGX_IOTO_KIND_LDIMM_EP0 ||
            record.kind == AGX_IOTO_KIND_LDIMM_EP1);
      CHECK(consumed == 2 || consumed == 8);
      const uint8_t *p = record.bytes;
      unsigned dest_half;
      uint32_t immediate;
      if (consumed == 2) {
         dest_half = ((p[0] >> 4) & 15) * 2;
         immediate = p[1];
      } else {
         const uint16_t w0 = p[0] | (uint16_t)p[1] << 8;
         const uint16_t w1 = p[2] | (uint16_t)p[3] << 8;
         const uint16_t w2 = p[4] | (uint16_t)p[5] << 8;
         const uint16_t w3 = p[6] | (uint16_t)p[7] << 8;
         dest_half = ((w0 >> 4) & 15) * 2 + ((w1 >> 6) & 3) * 32 +
                     ((w1 >> 3) & 1) * 128;
         immediate = ((w0 >> 8) & 127) | (uint32_t)((w2 >> 1) & 15) << 7 |
                     (uint32_t)((w2 >> 10) & 3) << 11 |
                     (uint32_t)(w3 & 15) << 13 |
                     (uint32_t)((w3 >> 4) & 255) << 17 |
                     (uint32_t)((w1 >> 9) & 127) << 25;
      }
      CHECK(dest_half == 4 + 2 * i);
      if (i < 2)
         pointer |= (uint64_t)immediate << (32 * i);
      else
         CHECK(immediate == expected_offset);
      offset += consumed;
   }
   CHECK(pointer == expected);
   CHECK(program->length <= 126);
   *prefix_size = offset;
   return 0;
}

static int
test_g17p_rt0_eot_program(void)
{
   CHECK(AGX_IOTO_G17P_EOT_RESOURCE_SPEC == 0);
   const struct hk_agx3_g17p_rt0_eot_program_desc desc = {
      .program_va = UINT64_C(0x10001990640),
      .shader_va = UINT64_C(0x10000000140),
      .pbe_argument_va = UINT64_C(0x10000020220),
      .shader_binary_size = 68,
      .scratch_size = 0,
      .nr_gprs = 8,
      .staging_halfs = 8,
   };
   struct hk_agx3_g17p_rt0_eot_program program;
   CHECK(hk_agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &desc,
                                               &program));
   CHECK(AGX_IOTO_G17P_RT0_EOT_PROGRAM_SIZE == 190);
   CHECK(program.length == 120);
   size_t prefix_size;
   CHECK(check_eot_pointer_prefix(&program, desc.pbe_argument_va, desc.imageblock_offset_B,
                                 &prefix_size) == 0);
   CHECK(prefix_size == 18);

   static const uint8_t source_prefix[] = {
      0x2c, 0xa0, 0x02, 0x00, 0x08, 0x00, 0x10, 0x00,
      0x3c, 0x80, 0x02, 0x00, 0x04, 0x00, 0x00, 0x00,
   };
   CHECK(memcmp(program.bytes, source_prefix, sizeof(source_prefix)) == 0);
   static const uint8_t eot_state_tail[] = {
      0x77, 0x00, 0x2a, 0x43, 0x4b, 0x00, 0x00, 0x00,
      0x77, 0x01, 0xaa, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00,
      0xf7, 0x00, 0x2a, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   CHECK(memcmp(program.bytes + prefix_size, eot_state_tail,
                sizeof(eot_state_tail)) == 0);

   /* Profile scratch occupies0..3; the table pointer survives in4..7.
    * Its only publications are COPY GPR4->uniform0 and GPR6->uniform2.
    * These COPY source-file/no-kill bits and the terminal are unchanged.
    */
   static const uint8_t expected_profile[] = {
      0x1c,0x80,0x02,0x00,0x00,0x00,0x00,0x00,0x14,0x81,0x11,0x06,
      0x00,0x00,0x00,0x00,0x0c,0x80,0x02,0x00,0x04,0x00,0x00,0x00,
      0x9f,0x11,0x54,0x00,0x02,0x00,0x08,0xa8,0x10,0x05,0x1c,0x80,
      0x02,0x00,0x04,0x00,0x00,0x00,0x0f,0x12,0x54,0x00,0x4c,0x00,
   };
   CHECK(sizeof(expected_profile) == AGX_IOTO_G17P_DEFAULT_PROFILE_PREFIX_SIZE);
   CHECK(memcmp(program.bytes + prefix_size + sizeof(eot_state_tail),
                expected_profile, sizeof(expected_profile)) == 0);
   static const uint8_t expected_copies[] = {
      0x0b,0x04,0x01,0x04, 0x1b,0x06,0x01,0x04,
      0x2b,0x08,0x01,0x04, /* COPY32 GPR8 -> uniform half4. */
   };
   const size_t copies_offset = prefix_size + sizeof(eot_state_tail) +
                               AGX_IOTO_G17P_DEFAULT_PROFILE_PREFIX_SIZE;
   CHECK(memcmp(program.bytes + copies_offset, expected_copies,
                sizeof(expected_copies)) == 0);
   const size_t offset_position = prefix_size - 2;
   CHECK(program.bytes[offset_position] == 0x4c);
   CHECK(program.bytes[offset_position + 1] == 0);
   static const uint8_t terminal[] = {
      0x03,0x00,0x07,0x00,0x02,0x00,0x00,0x00,0x60,0x00,
      0x0e,0x00,0x00,0x00,
   };
   CHECK(memcmp(program.bytes + copies_offset + sizeof(expected_copies),
                terminal, sizeof(terminal)) == 0);

   struct hk_agx3_g17p_rt0_eot_program relocated;
   struct hk_agx3_g17p_rt0_eot_program_desc moved = desc;
   /* Actual driver kind5 is an8-bit byte offset, loaded as a complete32-bit
    * uniform. Exercise all values; only this third immediate changes.
    * Runtime supplies the validated tilebuffer layout, not a fixed literal. */
   for (unsigned offset = 0; offset <= UINT8_MAX; ++offset) {
      moved = desc;
      moved.imageblock_offset_B = offset;
      CHECK(hk_agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &moved,
                                                   &relocated));
      size_t relocated_prefix;
      CHECK(check_eot_pointer_prefix(&relocated, desc.pbe_argument_va, offset,
                                     &relocated_prefix) == 0);
      CHECK(relocated.length == program.length + (offset >= 128 ? 6 : 0));
      CHECK(memcmp(relocated.bytes, program.bytes, offset_position) == 0);
      CHECK(memcmp(relocated.bytes + relocated_prefix,
                   program.bytes + prefix_size,
                   program.length - prefix_size) == 0);
   }
   moved = desc;
   moved.pbe_argument_va += UINT64_C(0x10000);
   CHECK(
      hk_agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &moved, &relocated));
   CHECK(memcmp(program.bytes, relocated.bytes, 16) != 0);
   CHECK(memcmp(program.bytes + 16, relocated.bytes + 16,
                program.length - 16) == 0);

   const uint64_t pointers[] = {
      32, 128, UINT64_C(0xffffffe0), UINT64_C(0x100000020),
      UINT64_C(0x8000002000), UINT64_C(0x123456780120),
      UINT64_C(0xffffffffffe0),
   };
   for (unsigned i = 0; i < ARRAY_SIZE(pointers); ++i) {
      moved = desc;
      moved.pbe_argument_va = pointers[i];
      CHECK(hk_agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &moved,
                                                   &relocated));
      size_t moved_prefix;
      CHECK(check_eot_pointer_prefix(&relocated, pointers[i], 0, &moved_prefix) == 0);
      CHECK(relocated.length - moved_prefix == program.length - prefix_size);
      CHECK(memcmp(relocated.bytes + moved_prefix, program.bytes + prefix_size,
                   program.length - prefix_size) == 0);
   }

   const struct hk_agx3_g17p_rt0_eot_program before = program;
   moved = desc;
   moved.staging_halfs = 0;
   CHECK(
      !hk_agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &moved, &program));
   CHECK(memcmp(&program, &before, sizeof(program)) == 0);
   for (unsigned offset = 8; offset < 32; offset += 8) {
      moved = desc;
      moved.pbe_argument_va += offset;
      CHECK(!hk_agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &moved,
                                                    &program));
      CHECK(memcmp(&program, &before, sizeof(program)) == 0);
   }
   moved = desc;
   moved.shader_binary_size = 64;
   CHECK(
      !hk_agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &moved, &program));
   CHECK(memcmp(&program, &before, sizeof(program)) == 0);
   moved.shader_binary_size = 128;
   CHECK(!hk_agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &moved, &program));
   CHECK(memcmp(&program, &before, sizeof(program)) == 0);
   moved = desc;
   moved.nr_gprs = 6;
   CHECK(
      !hk_agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &moved, &program));
   CHECK(memcmp(&program, &before, sizeof(program)) == 0);
   moved = desc;
   moved.pbe_argument_va++;
   CHECK(
      !hk_agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &moved, &program));
   CHECK(memcmp(&program, &before, sizeof(program)) == 0);
   moved = desc;
   moved.pbe_argument_va = UINT64_C(0xfffffffffff0);
   CHECK(!hk_agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &moved,
                                                 &program));
   CHECK(memcmp(&program, &before, sizeof(program)) == 0);
   moved = desc;
   moved.pbe_argument_va = 0;
   CHECK(!hk_agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &moved,
                                                 &program));
   CHECK(memcmp(&program, &before, sizeof(program)) == 0);
   moved = desc;
   moved.pbe_argument_va = 32;
   CHECK(hk_agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &moved,
                                                &relocated));
   CHECK(relocated.length == program.length - 12);
   CHECK(memcmp(relocated.bytes + 4, program.bytes + 16,
                relocated.length - 4) == 0);
   CHECK(
      !hk_agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17S, &desc, &program));

   struct hk_agx3_g17p_rt0_eot_program_desc mrt = desc;
   mrt.shader_binary_size = 120;
   mrt.nr_gprs = 8;
   mrt.staging_halfs = 16;
   mrt.attachment_count = 2;
   mrt.imageblock_offset1_B = 8;
   CHECK(hk_agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &mrt,
                                                &relocated));
   CHECK(relocated.length <= AGX_IOTO_G17P_RT0_EOT_PROGRAM_SIZE);
   mrt.attachment_count = 1;
   CHECK(!hk_agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &mrt,
                                                 &program));
   return 0;
}

static int
test_g17p_combined_sampler_pointers(void)
{
   const struct hk_agx3_g17p_combined_sampler_heap heap = {
      .descriptor_set_va = UINT64_C(0x10000020000),
      .sampler_heap_va = UINT64_C(0x10000080000),
      .texture_offset_B = 64,
      .sampler_index = HK_AGX3_G17P_BINDLESS_SAMPLER_BASE_INDEX + 7,
      .sampler_count = 1024,
      .sampler_element_size_B = 32,
   };
   uint64_t pointers[2] = {0};
   CHECK(hk_agx3_g17p_combined_sampler_pointers(&heap, pointers));
   CHECK(pointers[0] == UINT64_C(0x10000020040));
   CHECK(pointers[1] == UINT64_C(0x100000800e0));

   /* Native ESL consumes these two qwords, not the inline 32-byte records
    * that they address. Both hardware table indices select entry zero.
    */
   const struct agx3_ioto_uniform_range range = {
      .start_halfs = 0,
      .size_halfs = 8,
      .address = UINT64_C(0x10000090000),
   };

   const struct hk_agx3_g17p_graphics_program_desc desc = {
      .program_va = UINT64_C(0x10001990240),
      .shader_va = UINT64_C(0x10000000080),
      .uniform_ranges = &range,
      .uniform_range_count = 1,
      .shader_binary_size = 54,
      .nr_gprs = 8,
      .fragment = true,
   };
   struct hk_agx3_g17p_graphics_program program;
   CHECK(hk_agx3_g17p_basic_graphics_program_generate(AGX_CHIP_G17P, &desc,
                                                      &program));
   CHECK(program.length > AGX_IOTO_G17P_NO_RESOURCE_PROGRAM_SIZE &&
         program.length <= HK_AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE &&
         program.bytes[program.length - AGX_IOTO_PROGRAM_TERMINATE_SIZE] ==
            0x0e);

   const uint64_t before[2] = {pointers[0], pointers[1]};
   struct hk_agx3_g17p_combined_sampler_heap bad = heap;
   bad.sampler_index = HK_AGX3_G17P_BINDLESS_SAMPLER_BASE_INDEX - 1;
   CHECK(!hk_agx3_g17p_combined_sampler_pointers(&bad, pointers));
   CHECK(memcmp(pointers, before, sizeof(before)) == 0);
   bad = heap;
   bad.sampler_index = HK_AGX3_G17P_BINDLESS_SAMPLER_BASE_INDEX + 1024;
   CHECK(!hk_agx3_g17p_combined_sampler_pointers(&bad, pointers));
   CHECK(memcmp(pointers, before, sizeof(before)) == 0);
   bad = heap;
   bad.sampler_element_size_B = 8;
   CHECK(!hk_agx3_g17p_combined_sampler_pointers(&bad, pointers));
   CHECK(memcmp(pointers, before, sizeof(before)) == 0);
   bad = heap;
   bad.texture_offset_B++;
   CHECK(!hk_agx3_g17p_combined_sampler_pointers(&bad, pointers));
   CHECK(memcmp(pointers, before, sizeof(before)) == 0);
   bad = heap;
   bad.descriptor_set_va = UINT64_MAX - 32;
   CHECK(!hk_agx3_g17p_combined_sampler_pointers(&bad, pointers));
   CHECK(memcmp(pointers, before, sizeof(before)) == 0);
   bad = heap;
   bad.sampler_heap_va = UINT64_MAX - 32;
   CHECK(!hk_agx3_g17p_combined_sampler_pointers(&bad, pointers));
   CHECK(memcmp(pointers, before, sizeof(before)) == 0);
   bad = heap;
   bad.texture_offset_B = 0;
   bad.descriptor_set_va = UINT64_MAX - 15;
   CHECK(!hk_agx3_g17p_combined_sampler_pointers(&bad, pointers));
   CHECK(memcmp(pointers, before, sizeof(before)) == 0);
   bad = heap;
   bad.sampler_index = HK_AGX3_G17P_BINDLESS_SAMPLER_BASE_INDEX;
   bad.sampler_heap_va = UINT64_MAX - 15;
   CHECK(!hk_agx3_g17p_combined_sampler_pointers(&bad, pointers));
   CHECK(memcmp(pointers, before, sizeof(before)) == 0);
   CHECK(!hk_agx3_g17p_combined_sampler_pointers(NULL, pointers));
   CHECK(memcmp(pointers, before, sizeof(before)) == 0);
   CHECK(!hk_agx3_g17p_combined_sampler_pointers(&heap, NULL));

   struct hk_agx3_g17p_combined_sampler_heap client = heap;
   client.descriptor_set_va = UINT64_C(0x87ffed0000);
   client.sampler_heap_va = UINT64_C(0x87ffef4000);
   CHECK(hk_agx3_g17p_combined_sampler_pointers(&client, pointers));
   CHECK(pointers[0] == UINT64_C(0x87ffed0040));
   CHECK(pointers[1] == UINT64_C(0x87ffef40e0));

   const struct hk_agx3_g17p_sampled_storage_heap sampled_storage = {
      .sampled = client,
      .pbe_descriptor_set_va = UINT64_C(0x87fff10000),
      .pbe_offset_B = 160,
   };
   uint64_t four[4] = {0};
   CHECK(hk_agx3_g17p_sampled_storage_pointers(&sampled_storage, four));
   CHECK(four[0] == UINT64_C(0x87ffed0040));
   CHECK(four[1] == UINT64_C(0x87ffef40e0));
   CHECK(four[2] == UINT64_C(0x87fff100a0));
   CHECK(four[3] == UINT64_C(0xffff));

   const uint64_t four_before[4] = {four[0], four[1], four[2], four[3]};
   struct hk_agx3_g17p_sampled_storage_heap invalid = sampled_storage;
   invalid.pbe_offset_B++;
   CHECK(!hk_agx3_g17p_sampled_storage_pointers(&invalid, four));
   CHECK(memcmp(four, four_before, sizeof(four)) == 0);
   invalid = sampled_storage;
   invalid.pbe_descriptor_set_va = UINT64_MAX - 64;
   CHECK(!hk_agx3_g17p_sampled_storage_pointers(&invalid, four));
   CHECK(memcmp(four, four_before, sizeof(four)) == 0);

   const struct hk_agx3_g17p_indexed_storage_heap indexed_storage = {
      .texture_heap_va = UINT64_C(0x9000000000),
      .sampler_heap_va = UINT64_C(0x9100000000),
      .pbe_descriptor_set_va = UINT64_C(0x9200000000),
      .pbe_offset_B = 176,
   };
   CHECK(hk_agx3_g17p_indexed_storage_pointers(&indexed_storage, four));
   CHECK(four[0] == indexed_storage.texture_heap_va);
   CHECK(four[1] == indexed_storage.sampler_heap_va);
   CHECK(four[2] == UINT64_C(0x92000000b0));
   CHECK(four[3] == UINT64_C(0xffff));

   const uint64_t indexed_before[4] = {
      four[0], four[1], four[2], four[3],
   };
   struct hk_agx3_g17p_indexed_storage_heap bad_indexed = indexed_storage;
   bad_indexed.pbe_offset_B++;
   CHECK(!hk_agx3_g17p_indexed_storage_pointers(&bad_indexed, four));
   CHECK(memcmp(four, indexed_before, sizeof(four)) == 0);
   return 0;
}

static int
test_g17p_root_range(void)
{
   struct agx3_ioto_uniform_range range = {0};
   CHECK(hk_agx3_g17p_root_range(12, 0, UINT64_C(0x10000024000), &range));
   CHECK(range.start_halfs == 12 && range.size_halfs == 4 &&
         range.address == UINT64_C(0x10000024000));
   CHECK(hk_agx3_g17p_root_range(12, 1, UINT64_C(0x10000024000), &range));
   CHECK(range.start_halfs == 12 && range.size_halfs == 8);
   CHECK(hk_agx3_g17p_root_range(12, 8, UINT64_C(0x10000024000), &range));
   CHECK(range.start_halfs == 12 && range.size_halfs == 36);

   const struct agx3_ioto_uniform_range before = range;
   CHECK(!hk_agx3_g17p_root_range(13, 0, UINT64_C(0x10000024000), &range));
   CHECK(memcmp(&range, &before, sizeof(range)) == 0);
   CHECK(!hk_agx3_g17p_root_range(12, 9, UINT64_C(0x10000024000), &range));
   CHECK(memcmp(&range, &before, sizeof(range)) == 0);
   CHECK(!hk_agx3_g17p_root_range(12, 0, 0, &range));
   CHECK(memcmp(&range, &before, sizeof(range)) == 0);
   return 0;
}

static int
test_g17p_rt0_background_programs(void)
{
   CHECK(hk_agx3_g17p_rt0_bg_shader_resources_match(
      HK_AGX3_G17P_RT0_BG_CLEAR, HK_AGX3_G17P_RT0_CLEAR_PUSH_HALFS, false));
   CHECK(hk_agx3_g17p_rt0_bg_shader_resources_match(
      HK_AGX3_G17P_RT0_BG_CLEAR,
      HK_AGX3_G17P_WBOIT_MRT_CLEAR_PUSH_HALFS, false));
   CHECK(!hk_agx3_g17p_rt0_bg_shader_resources_match(
      HK_AGX3_G17P_RT0_BG_CLEAR, HK_AGX3_G17P_RT0_CLEAR_PUSH_HALFS - 1,
      false));
   CHECK(!hk_agx3_g17p_rt0_bg_shader_resources_match(
      HK_AGX3_G17P_RT0_BG_CLEAR, HK_AGX3_G17P_RT0_CLEAR_PUSH_HALFS + 1,
      false));
   CHECK(hk_agx3_g17p_rt0_bg_shader_resources_match(
      HK_AGX3_G17P_RT0_BG_RELOAD, 0, true));
   CHECK(!hk_agx3_g17p_rt0_bg_shader_resources_match(
      HK_AGX3_G17P_RT0_BG_RELOAD, 0, false));

   const struct hk_agx3_g17p_rt0_bg_program_desc clear = {
      .program_va = UINT64_C(0x10001990240),
      .shader_va = UINT64_C(0x10000000080),
      .clear_value_va = UINT64_C(0x1000008c000),
      .shader_binary_size = 54,
      .scratch_size = 0,
      .nr_gprs = 8,
      .op = HK_AGX3_G17P_RT0_BG_CLEAR,
   };
   struct hk_agx3_g17p_rt0_bg_program program;
   CHECK(hk_agx3_g17p_rt0_bg_program_generate(AGX_CHIP_G17P, &clear, &program));
   CHECK(program.length == AGX_IOTO_G17P_RT0_CLEAR_PROGRAM_SIZE);

   struct hk_agx3_g17p_rt0_bg_program_desc mrt_clear = clear;
   mrt_clear.clear_push_halfs = HK_AGX3_G17P_WBOIT_MRT_CLEAR_PUSH_HALFS;
   struct hk_agx3_g17p_rt0_bg_program mrt_clear_program;
   CHECK(hk_agx3_g17p_rt0_bg_program_generate(
      AGX_CHIP_G17P, &mrt_clear, &mrt_clear_program));
   CHECK(mrt_clear_program.length ==
         AGX_IOTO_G17P_WBOIT_MRT_CLEAR_PROGRAM_SIZE);

   /* m1n1 g17p_triangle_source.py, split where HAL200 inserts LoadShader and
    * the default profile-loader prefix before the trailing COPY records. */
   static const uint8_t clear_pre_copy[] = {
      0x0c, 0x80, 0x02, 0x00, 0x00, 0x00, 0x46, 0x00, 0x1c, 0x80,
      0x02, 0x00, 0x04, 0x00, 0x00, 0x00, 0x67, 0x00, 0x54, 0x04,
      0x00, 0x00, 0x00, 0x00, 0x17, 0x00, 0x00, 0x40, 0x22, 0x00,
      0x06, 0x08, 0x06, 0x00, 0x06, 0x00, 0x06, 0x00,
   };
   static const uint8_t clear_copies[] = {
      0x2b, 0x04, 0x09, 0x04, 0x3b, 0x06, 0x09, 0x04,
      0x4b, 0x08, 0x09, 0x04, 0x5b, 0x0a, 0x09, 0x04,
   };
   CHECK(memcmp(program.bytes, clear_pre_copy, sizeof(clear_pre_copy)) == 0);
   const size_t copy_offset = sizeof(clear_pre_copy) +
                              AGX_IOTO_STATE_TAIL_TYPE0_SIZE +
                              AGX_IOTO_G17P_DEFAULT_PROFILE_PREFIX_SIZE;
   CHECK(memcmp(program.bytes + copy_offset, clear_copies,
                sizeof(clear_copies)) == 0);
   static const uint8_t background_termination[] = {
      0x07, 0x02, 0x54, 0x00, 0x02, 0x00, 0x0e, 0x00, 0x00, 0x00,
   };
   CHECK(memcmp(program.bytes + program.length - sizeof(background_termination),
                background_termination, sizeof(background_termination)) == 0);

   struct hk_agx3_g17p_rt0_bg_program relocated;
   struct hk_agx3_g17p_rt0_bg_program_desc moved = clear;
   moved.clear_value_va += UINT64_C(0x10000);
   CHECK(
      hk_agx3_g17p_rt0_bg_program_generate(AGX_CHIP_G17P, &moved, &relocated));
   CHECK(relocated.length == program.length);
   CHECK(memcmp(relocated.bytes, program.bytes, 16) != 0);
   CHECK(memcmp(relocated.bytes + 16, program.bytes + 16,
                program.length - 16) == 0);

   const struct hk_agx3_g17p_rt0_bg_program_desc reload = {
      .program_va = UINT64_C(0x10001990000),
      .shader_va = UINT64_C(0x10000000100),
      .texture_pointer_va = UINT64_C(0x10000020220),
      .sampler_pointer_va = UINT64_C(0x10000020300),
      .shader_binary_size = 128,
      .scratch_size = 0,
      .nr_gprs = 12,
      .op = HK_AGX3_G17P_RT0_BG_RELOAD,
   };
   CHECK(
      hk_agx3_g17p_rt0_bg_program_generate(AGX_CHIP_G17P, &reload, &program));
   CHECK(program.length == 154);
   static const uint8_t fragment_speclm[] = {
      0x77, 0x00, 0x2a, 0x43, 0x4b, 0x00, 0x00, 0x00,
   };
   const size_t reload_tail =
      find_unique_bytes(program.bytes, program.length, fragment_speclm,
                        sizeof(fragment_speclm));
   CHECK(reload_tail != SIZE_MAX);
   CHECK(program.bytes[reload_tail + 16] == 0x00 &&
         program.bytes[reload_tail + 17] == 0x00);
   static const uint8_t reload_copies[] = {
      0x0b, 0x04, 0x09, 0x04, 0x1b, 0x06, 0x09, 0x04,
      0x2b, 0x0c, 0x09, 0x04, 0x3b, 0x0e, 0x09, 0x04,
   };
   const size_t reload_copy_offset =
      program.length - sizeof(reload_copies) -
      sizeof(background_termination);
   if (memcmp(program.bytes + reload_copy_offset, reload_copies,
              sizeof(reload_copies)) != 0) {
      fprintf(stderr, "ACTUAL4:");
      for (size_t k = 0; k < sizeof(reload_copies); ++k)
         fprintf(stderr, " %02x", program.bytes[reload_copy_offset + k]);
      fprintf(stderr, "\n");
   }
   CHECK(memcmp(program.bytes + reload_copy_offset, reload_copies,
                sizeof(reload_copies)) == 0);
   CHECK(memcmp(program.bytes + program.length - sizeof(background_termination),
                background_termination, sizeof(background_termination)) == 0);

   struct hk_agx3_g17p_rt0_bg_program far_program;
   struct hk_agx3_g17p_rt0_bg_program_desc far_reload = reload;
   far_reload.sampler_pointer_va += UINT64_C(0x100000);
   CHECK(hk_agx3_g17p_rt0_bg_program_generate(AGX_CHIP_G17P, &far_reload,
                                               &far_program));
   CHECK(far_program.length == 170);
   const size_t far_copy_offset =
      far_program.length - sizeof(reload_copies) -
      sizeof(background_termination);
   CHECK(memcmp(far_program.bytes + far_copy_offset, reload_copies,
                sizeof(reload_copies)) == 0);
   CHECK(memcmp(far_program.bytes + far_program.length -
                   sizeof(background_termination),
                background_termination, sizeof(background_termination)) == 0);

   const struct hk_agx3_g17p_rt0_bg_program before = program;
   moved = clear;
   moved.clear_value_va++;
   CHECK(
      !hk_agx3_g17p_rt0_bg_program_generate(AGX_CHIP_G17P, &moved, &program));
   CHECK(memcmp(&program, &before, sizeof(program)) == 0);
   moved = clear;
   moved.texture_pointer_va = UINT64_C(0x10000020220);
   CHECK(
      !hk_agx3_g17p_rt0_bg_program_generate(AGX_CHIP_G17P, &moved, &program));
   CHECK(memcmp(&program, &before, sizeof(program)) == 0);
   CHECK(
      !hk_agx3_g17p_rt0_bg_program_generate(AGX_CHIP_G17S, &reload, &program));
   CHECK(memcmp(&program, &before, sizeof(program)) == 0);
   return 0;
}

static int
test_g17p_rt0_background_staging(void)
{
   struct hk_agx3_g17p_rt0_bg_program_desc desc = {
      .program_va = UINT64_C(0x10001990240),
      .shader_va = UINT64_C(0x100fffbc3c0),
      .clear_value_va = UINT64_C(0x87ffee0000),
      .shader_binary_size = 52,
      .nr_gprs = 6,
      .op = HK_AGX3_G17P_RT0_BG_CLEAR,
   };
   const struct {
      bool known;
      uint16_t halfs;
      uint32_t d;
   } cases[] = {{true, 0, 0}, {true, 8, 16}, {false, 8, 6}};
   struct hk_agx3_g17p_rt0_bg_program baseline;
   CHECK(hk_agx3_g17p_rt0_bg_program_generate(AGX_CHIP_G17P, &desc,
                                               &baseline));
   static const uint8_t speclm[] = {0x77, 0x00, 0x2a, 0x43,
                                  0x4b, 0x00, 0x00, 0x00};
   const size_t tail = find_unique_bytes(baseline.bytes, baseline.length,
                                          speclm, sizeof(speclm));
   CHECK(tail != SIZE_MAX);
   const size_t offset = tail + AGX_IOTO_SPECLM_SIZE + AGX_IOTO_LDSHDR_SIZE;

   for (size_t i = 0; i < ARRAY_SIZE(cases); ++i) {
      desc.staging_known = cases[i].known;
      desc.staging_halfs = cases[i].halfs;
      struct hk_agx3_g17p_rt0_bg_program program;
      CHECK(hk_agx3_g17p_rt0_bg_program_generate(AGX_CHIP_G17P, &desc,
                                                  &program));
      struct agx_ioto_state state;
      uint8_t expected[AGX_IOTO_SPECTPR_SIZE];
      CHECK(agx_ioto_g17p_compiler_spectpr_build(0, cases[i].d, &state));
      CHECK(agx_ioto_state_pack_spectpr(AGX_CHIP_G17P, &state, expected,
                                         sizeof(expected)));
      CHECK(program.length == baseline.length);
      CHECK(memcmp(program.bytes + offset, expected, sizeof(expected)) == 0);
      CHECK(memcmp(program.bytes, baseline.bytes, offset) == 0);
      CHECK(memcmp(program.bytes + offset + sizeof(expected),
                   baseline.bytes + offset + sizeof(expected),
                   baseline.length - offset - sizeof(expected)) == 0);
   }

   /* A caller with unknown staging must use the fallback; changing its
    * untrusted count cannot change that launch. */
   desc.op = HK_AGX3_G17P_RT0_BG_RELOAD;
   desc.clear_value_va = 0;
   desc.texture_pointer_va = UINT64_C(0x87ffee0200);
   desc.sampler_pointer_va = UINT64_C(0x87ffee0240);
   desc.shader_binary_size = 74;
   desc.nr_gprs = 16;
   desc.staging_known = false;
   desc.staging_halfs = 0;
   CHECK(hk_agx3_g17p_rt0_bg_program_generate(AGX_CHIP_G17P, &desc,
                                               &baseline));
   desc.staging_halfs = UINT16_MAX;
   struct hk_agx3_g17p_rt0_bg_program unknown;
   CHECK(hk_agx3_g17p_rt0_bg_program_generate(AGX_CHIP_G17P, &desc,
                                               &unknown));
   CHECK(unknown.length == baseline.length);
   CHECK(memcmp(unknown.bytes, baseline.bytes, baseline.length) == 0);
   return 0;
}

static int
test_g17p_single_layer_rt0_layout(void)
{
   CHECK(hk_agx3_g17p_single_layer_rendering(0, 1));
   CHECK(!hk_agx3_g17p_single_layer_rendering(0, 0));
   CHECK(!hk_agx3_g17p_single_layer_rendering(0, 2));
   CHECK(!hk_agx3_g17p_single_layer_rendering(1, 1));

   enum pipe_format empty[AGX_MAX_RENDER_TARGETS] = {PIPE_FORMAT_NONE};
   struct agx_tilebuffer_layout empty_tib =
      agx_build_tilebuffer_layout(empty, 0, 1, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&empty_tib));
   CHECK(empty_tib.sample_size_B == 0);
   empty_tib.sample_size_B = 8;
   CHECK(!hk_agx3_g17p_rt0_tile_layout_supported(&empty_tib));

   enum pipe_format rgba[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R8G8B8A8_UNORM,
   };
   struct agx_tilebuffer_layout tib =
      agx_build_tilebuffer_layout(rgba, 1, 1, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(agx_tilebuffer_offset_B(&tib, 0) == 0);

   enum pipe_format bgra[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_B8G8R8A8_UNORM,
   };
   tib = agx_build_tilebuffer_layout(bgra, 1, 1, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(agx_tilebuffer_offset_B(&tib, 0) == 0);

   enum pipe_format narrow[] = {
      PIPE_FORMAT_R8_UNORM,
      PIPE_FORMAT_R8G8_UNORM,
      PIPE_FORMAT_R16G16B16A16_FLOAT,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(narrow); ++i) {
      enum pipe_format format[AGX_MAX_RENDER_TARGETS] = {narrow[i]};
      tib = agx_build_tilebuffer_layout(format, 1, 1, false);
      CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
      CHECK(tib.sample_size_B == 8);
      CHECK(agx_tilebuffer_offset_B(&tib, 0) == 0);
   }

   enum pipe_format rg16[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R16G16_UNORM,
   };
   tib = agx_build_tilebuffer_layout(rg16, 1, 1, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(tib.sample_size_B == 8);
   CHECK(agx_tilebuffer_offset_B(&tib, 0) == 0);

   enum pipe_format rg16_snorm[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R16G16_SNORM,
   };
   tib = agx_build_tilebuffer_layout(rg16_snorm, 1, 1, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(tib.sample_size_B == 8);
   CHECK(agx_tilebuffer_offset_B(&tib, 0) == 0);

   /* Keep the signed-normalized path format- and word-count-exact. */
   enum pipe_format rgba16_snorm[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R16G16B16A16_SNORM,
   };
   tib = agx_build_tilebuffer_layout(rgba16_snorm, 1, 1, false);
   CHECK(!hk_agx3_g17p_rt0_tile_layout_supported(&tib));

   enum pipe_format rgba16[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R16G16B16A16_UNORM,
   };
   tib = agx_build_tilebuffer_layout(rgba16, 1, 1, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(tib.sample_size_B == 8);
   CHECK(agx_tilebuffer_offset_B(&tib, 0) == 0);

   tib = agx_build_tilebuffer_layout(rg16, 1, 1, false);

   tib.layered = true;
   CHECK(!hk_agx3_g17p_rt0_tile_layout_supported(&tib));

   enum pipe_format mrt[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R8G8B8A8_UNORM,
      PIPE_FORMAT_R8G8B8A8_UNORM,
   };
   tib = agx_build_tilebuffer_layout(mrt, 2, 1, false);
   CHECK(!hk_agx3_g17p_rt0_tile_layout_supported(&tib));

   enum pipe_format wboit[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R16G16B16A16_FLOAT,
      PIPE_FORMAT_R8_UNORM,
   };
   tib = agx_build_tilebuffer_layout(wboit, 2, 1, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(tib.sample_size_B == 16);
   CHECK(agx_tilebuffer_offset_B(&tib, 0) == 0);
   CHECK(agx_tilebuffer_offset_B(&tib, 1) == 8);

   tib = agx_build_tilebuffer_layout(rgba, 1, 2, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(tib.nr_samples == 2 && tib.sample_size_B == 8 &&
         tib.tile_size == 32 * 32 && !tib.spilled[0]);
   tib = agx_build_tilebuffer_layout(rgba, 1, 4, false);
   CHECK(!hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(!hk_agx3_g17p_rt0_tile_layout_supported(NULL));
   return 0;
}

static int
test_g17p_rt0_background_pass_control(void)
{
   uint64_t resource_spec = UINT64_C(0x123456789abcdef0);
   CHECK(hk_agx3_g17p_rt0_bg_resource_spec(
      AGX_CHIP_G17P, HK_AGX3_G17P_RT0_BG_CLEAR, &resource_spec));
   CHECK(resource_spec == UINT64_C(0xffff800000000040));
   CHECK((resource_spec >> 47) == 0x1ffff);
   CHECK(hk_agx3_g17p_rt0_bg_resource_spec(
      AGX_CHIP_G17P, HK_AGX3_G17P_RT0_BG_RELOAD, &resource_spec));
   CHECK(resource_spec == UINT64_C(0xffff800000000040));

   /* No-op is not a real RT0 write and must not inherit its extended mask. */
   CHECK(hk_agx3_g17p_rt0_bg_resource_spec(
      AGX_CHIP_G17P, HK_AGX3_G17P_RT0_BG_NONE, &resource_spec));
   CHECK(resource_spec == UINT64_C(0x40));
   CHECK((resource_spec >> 47) == 0);
   CHECK(AGX_IOTO_G17P_EOT_RESOURCE_SPEC == UINT64_C(0));

   for (size_t i = 0; i < ARRAY_SIZE(targets); ++i) {
      if (targets[i] == AGX_CHIP_G17P)
         continue;
      CHECK(!hk_agx3_g17p_rt0_bg_resource_spec(
         targets[i], HK_AGX3_G17P_RT0_BG_CLEAR, &resource_spec));
      CHECK(resource_spec == UINT64_C(0x40));
   }
   CHECK(!hk_agx3_g17p_rt0_bg_resource_spec(
      AGX_CHIP_G17P, (enum hk_agx3_g17p_rt0_bg_op)99, &resource_spec));
   CHECK(resource_spec == UINT64_C(0x40));
   CHECK(!hk_agx3_g17p_rt0_bg_resource_spec(
      AGX_CHIP_G17P, HK_AGX3_G17P_RT0_BG_CLEAR, NULL));
   CHECK(!hk_agx3_g17p_rt0_bg_shader_resources_match(
      HK_AGX3_G17P_RT0_BG_NONE, 0, false));
   return 0;
}

static int
test_alias(void)
{
   struct hk_agx3_ioto_program_desc desc = make_desc(AGX_CHIP_G17G);
   uint8_t storage[96] = {0x06, 0x00, 0x06, 0x00};
   static const uint8_t expected[] = {0x06, 0x00, 0x06, 0x00};
   desc.body = storage;
   desc.body_size = sizeof(expected);
   size_t written;
   CHECK(hk_agx3_ioto_program_generate(AGX_CHIP_G17G, &desc, storage + 2,
                                       sizeof(storage) - 2, &written));
   CHECK(memcmp(storage + 2, expected, sizeof(expected)) == 0);

   memset(storage, 0, sizeof(storage));
   memcpy(storage, expected, sizeof(expected));
   static const uint8_t suffix_code[] = {0x1b, 0x0c, 0x06, 0x00};
   memcpy(storage + 4, suffix_code, sizeof(suffix_code));
   desc.body = storage;
   desc.body_size = sizeof(expected);
   const struct agx_ioto_preshader_desc preshader = {
      .code = storage + 4,
      .code_size = sizeof(suffix_code),
      .external_va = AGX_IOTO_PRESHADER_NO_VA,
   };
   CHECK(hk_agx3_ioto_program_generate_no_pixrel(
      AGX_CHIP_G17G, &desc, &preshader, storage + 2, sizeof(storage) - 2,
      &written));
   CHECK(memcmp(storage + 2, expected, sizeof(expected)) == 0);
   CHECK(storage[2 + written - 2] == 0x1b &&
         storage[2 + written - 1] == 0x0c);
   return 0;
}

static int
test_coverage_initial_pixel_dependencies(void)
{
   const struct agx3_ioto_uniform_range root = {
      .start_halfs=12, .size_halfs=4, .address=0x2000,
   };
   struct hk_agx3_g17p_graphics_program_desc desc = {
      .program_va=0x10000040000, .shader_va=0x10000000080,
      .shader_binary_size=280, .nr_gprs=15, .compiler_d_valid=true,
      .compiler_d=4, .fragment=true, .uniform_ranges=&root,
   };
   for(unsigned resources=0;resources<=1;resources++) {
      desc.uniform_range_count=resources;
      struct hk_agx3_g17p_graphics_program opaque,coverage;
      desc.writes_sample_mask=false;
      CHECK(hk_agx3_g17p_basic_graphics_program_generate(AGX_CHIP_G17P,&desc,&opaque));
      desc.writes_sample_mask=true;
      CHECK(hk_agx3_g17p_basic_graphics_program_generate(AGX_CHIP_G17P,&desc,&coverage));
      CHECK(coverage.length==opaque.length);
      const uint8_t native_tail[]={0x07,0x02,0x54,0xf2,0x02,0x00,0x0e,0,0,0};
      CHECK(!memcmp(coverage.bytes+coverage.length-sizeof(native_tail),native_tail,sizeof(native_tail)));
      CHECK(opaque.bytes[opaque.length-7]==0xf3);
      opaque.bytes[opaque.length-7]=0xf2;
      CHECK(!memcmp(opaque.bytes,coverage.bytes,coverage.length));
   }
   desc.fragment=false;
   struct hk_agx3_g17p_graphics_program invalid;
   CHECK(!hk_agx3_g17p_basic_graphics_program_generate(AGX_CHIP_G17P,&desc,&invalid));
   return 0;
}

int
main(void)
{
   if (test_coverage_initial_pixel_dependencies() || test_exact_target_order() || test_terminal_suffix_modes() ||
       test_fail_closed() || test_g17s_minimal_compute_admission() ||
       test_g17p_dynamic_compute_program() ||
       test_g17p_direct_add3_compute_bridge() ||
       test_g17p_no_resource_program_bridge() ||
       test_g17p_basic_graphics_program() || test_g17p_rt0_eot_program() ||
       test_g17p_combined_sampler_pointers() ||
       test_g17p_root_range() ||
       test_g17p_rt0_background_programs() ||
       test_g17p_rt0_background_staging() ||
       test_g17p_rt0_background_pass_control() ||
       test_g17p_single_layer_rt0_layout() || test_alias())
      return 1;

   printf("Honeykrisp AGX3 Ioto program core: %u checks passed\n", checks);
   return 0;
}
