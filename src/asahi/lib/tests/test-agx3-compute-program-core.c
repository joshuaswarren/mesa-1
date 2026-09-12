/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "../agx3_compute_program_core.h"

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

static const enum agx_chip targets[] = {
   AGX_CHIP_G15G, AGX_CHIP_G15S, AGX_CHIP_G15C, AGX_CHIP_G15D,
   AGX_CHIP_G16G, AGX_CHIP_G16S, AGX_CHIP_G16C, AGX_CHIP_G17P,
   AGX_CHIP_G17G, AGX_CHIP_G17S, AGX_CHIP_G17C,
};

static bool
is_hal300(enum agx_chip chip)
{
   return chip == AGX_CHIP_G17G || chip == AGX_CHIP_G17S ||
          chip == AGX_CHIP_G17C;
}

static struct agx3_compute_program_desc
make_desc(enum agx_chip chip)
{
   static const uint8_t nop[] = {0x06, 0x00};
   struct agx3_compute_program_desc desc = {
      .chip = chip,
      .program_va = UINT64_C(0x123456789a80),
      .body = nop,
      .body_size = sizeof(nop),
      .profile =
         {
            .chip = chip,
            .payload0 = is_hal300(chip) ? UINT16_C(0x8098) : UINT16_C(0x0555),
            .payload1 = is_hal300(chip) ? UINT16_C(0xa400) : 0,
         },
      .preshader.external_va = AGX_IOTO_PRESHADER_NO_VA,
      .launch_sources =
         {
            .max_total_threads_per_threadgroup = 0x345,
            .extended_texture_write_mask_qword_1 = 0x124,
            .ray_traversal_depth_nonzero = true,
         },
   };

   desc.tail.type = AGX_IOTO_STATE_TAIL_TYPE0;
   desc.tail.ldshdr.f[0] = UINT64_C(0x123456789ac0);
   if (is_hal300(chip)) {
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
bytes_are(const void *data, size_t size, uint8_t value)
{
   const uint8_t *bytes = data;
   for (size_t i = 0; i < size; ++i) {
      if (bytes[i] != value)
         return false;
   }
   return true;
}

static bool
test_prepare_and_launch(void)
{
   for (size_t i = 0; i < ARRAY_SIZE(targets); ++i) {
      struct agx3_compute_program_desc desc = make_desc(targets[i]);
      uint8_t state[96];
      memset(state, 0xa5, sizeof(state));
      size_t written = SIZE_MAX;
      struct agx3_compute_program program;
      memset(&program, 0xa5, sizeof(program));
      CHECK(agx3_compute_program_prepare(&desc, state, sizeof(state), &written,
                                         &program));
      CHECK(program.chip == desc.chip);
      CHECK(program.program_va == desc.program_va);
      CHECK(written > desc.body_size && written < sizeof(state));
      CHECK(state[written] == 0xa5);

      struct agx3_cdm_direct_template expected;
      CHECK(agx3_cdm_direct_template_build_from_sources(&desc.launch_sources,
                                                        &expected));
      CHECK(memcmp(&program.launch_template, &expected, sizeof(expected)) == 0);

      const struct agx3_compute_direct_grid grid = {
         .source = AGX3_CDM_DIRECT_GRID_THREADGROUP_COUNTS,
         .size = {7, 8, 9},
         .local_size = {4, 5, 6},
         .dimensions = 3,
      };
      uint8_t launch[AGX3_CDM_DIRECT_LAUNCH_SIZE];
      CHECK(agx3_compute_program_pack_direct_launch(&program, &grid, launch));

      struct agx3_cdm_direct_launch_info decoded;
      CHECK(agx3_cdm_direct_launch_unpack(desc.chip, launch, sizeof(launch),
                                          &decoded));
      CHECK(decoded.pipeline == desc.program_va);
      CHECK(decoded.global_size[0] == 28 && decoded.global_size[1] == 40 &&
            decoded.global_size[2] == 54);
      CHECK(decoded.local_size[0] == 4 && decoded.local_size[1] == 5 &&
            decoded.local_size[2] == 6);
      CHECK(decoded.dimensions == 3);
      CHECK(memcmp(&decoded.template_words, &expected, sizeof(expected)) == 0);
   }

   return true;
}

static bool
test_a18_pro_one_dimensional_launch(void)
{
   struct agx3_compute_program_desc desc = make_desc(AGX_CHIP_G17P);
   uint8_t state[96];
   size_t written;
   struct agx3_compute_program program;
   CHECK(agx3_compute_program_prepare(&desc, state, sizeof(state), &written,
                                      &program));

   const struct agx3_compute_direct_grid grid = {
      .source = AGX3_CDM_DIRECT_GRID_GLOBAL_THREADS,
      .size = {257, 1, 1},
      .local_size = {32, 1, 1},
      .dimensions = 1,
   };
   uint8_t launch[AGX3_CDM_DIRECT_LAUNCH_SIZE];
   CHECK(agx3_compute_program_pack_direct_launch(&program, &grid, launch));

   struct agx3_cdm_direct_launch_info decoded;
   CHECK(agx3_cdm_direct_launch_unpack(AGX_CHIP_G17P, launch, sizeof(launch),
                                       &decoded));
   struct agx3_cdm_direct_template_fields fields;
   CHECK(agx3_cdm_direct_template_decode(&decoded.template_words, &fields));
   CHECK(!fields.subkernel_x_target_workitems);
   CHECK(fields.ray_traversal_depth_nonzero);
   CHECK(decoded.pipeline == desc.program_va);
   CHECK(decoded.global_size[0] == 257 && decoded.global_size[1] == 1 &&
         decoded.global_size[2] == 1);
   return true;
}

static bool
test_g17p_native_compute_join(void)
{
   struct agx3_cdm_direct_template_sources sources;
   CHECK(agx3_cdm_direct_template_sources_compute(true, false, false,
                                                   &sources));
   const struct agx3_g17p_native_compute_program_desc desc = {
      .program_va = UINT64_C(0x100000b0000),
      .shader_va = UINT64_C(0x10000000100),
      .launch_sources = sources,
   };
   struct agx3_compute_program program;
   CHECK(agx3_g17p_native_compute_program_bind(&desc, &program));
   CHECK(program.chip == AGX_CHIP_G17P);
   CHECK(program.program_va == desc.program_va);
   CHECK(program.shader_va == desc.shader_va);
   CHECK(program.g17p_native_compute_live_bound);

   const struct agx3_compute_direct_grid grid = {
      .source = AGX3_CDM_DIRECT_GRID_GLOBAL_THREADS,
      .size = {96, 4, 1},
      .local_size = {32, 2, 1},
      .dimensions = 2,
   };
   static const uint8_t expected[] = {
      0x00, 0x00, 0x08, 0x00, 0x40, 0x00, 0x00, 0x01,
      0x00, 0x2c, 0x00, 0x00, 0x01, 0x00, 0x00, 0x40,
      0x60, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
      0x01, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00,
      0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
      0x60, 0x01, 0x00, 0x60,
   };
   uint8_t dispatch[AGX3_G17P_NATIVE_COMPUTE_DISPATCH_SIZE + 2];
   memset(dispatch, 0xa5, sizeof(dispatch));
   struct agx3_compute_dispatch_sink sink = {
      .data = dispatch,
      .capacity = sizeof(dispatch),
      .length = 1,
   };
   CHECK(agx3_g17p_native_compute_dispatch_append(&program, &grid, &sink));
   CHECK(sink.length == 1 + sizeof(expected));
   CHECK(dispatch[0] == 0xa5 && dispatch[sink.length] == 0xa5);
   CHECK(memcmp(dispatch + 1, expected, sizeof(expected)) == 0);

   uint8_t strong_dispatch[AGX3_G17P_NATIVE_COMPUTE_DISPATCH_SIZE + 2];
   memset(strong_dispatch, 0xa5, sizeof(strong_dispatch));
   struct agx3_compute_dispatch_sink strong_sink = {
      .data = strong_dispatch,
      .capacity = sizeof(strong_dispatch),
      .length = 1,
   };
   CHECK(agx3_g17p_native_compute_dispatch_append_with_barrier(
      &program, &grid, 0x5, &strong_sink));
   CHECK(strong_sink.length == 1 + sizeof(expected));
   CHECK(memcmp(strong_dispatch + 1, expected,
                AGX3_CDM_DIRECT_LAUNCH_SIZE) == 0);
   static const uint8_t strong_barrier[4] = {
      0x60, 0x15, 0x00, 0x60,
   };
   CHECK(memcmp(strong_dispatch + 1 + AGX3_CDM_DIRECT_LAUNCH_SIZE,
                strong_barrier, sizeof(strong_barrier)) == 0);
   const size_t strong_length = strong_sink.length;
   CHECK(!agx3_g17p_native_compute_dispatch_append_with_barrier(
      &program, &grid, 0x10, &strong_sink));
   CHECK(strong_sink.length == strong_length);

   struct agx3_cdm_direct_template_sources indirect_sources;
   CHECK(agx3_cdm_direct_template_sources_compute(false, false, false,
                                                   &indirect_sources));
   struct agx3_g17p_native_compute_program_desc indirect_program_desc = desc;
   indirect_program_desc.launch_sources = indirect_sources;
   struct agx3_compute_program indirect_program;
   CHECK(agx3_g17p_native_compute_program_bind(&indirect_program_desc,
                                                &indirect_program));

   const struct agx3_g17p_native_indirect_dispatch indirect = {
      .helper_program_va = UINT64_C(0x100000b0100),
      .geometry_va = UINT64_C(0x100001014c0),
      .inline_global_size = {96, 1, 1},
      .local_size = {32, 1, 1},
   };
   static const uint8_t expected_indirect[] = {
      0x00, 0x00, 0x88, 0x00, 0x00, 0x00, 0x00, 0x01,
      0x00, 0x2c, 0x00, 0x00, 0x01, 0x00, 0x00, 0x40,
      0x60, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
      0x01, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00,
      0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
      0x60, 0x01, 0x00, 0x60,
      0x00, 0x00, 0x08, 0x10, 0x00, 0x00, 0x00, 0x01,
      0x04, 0x2c, 0x00, 0x00, 0x01, 0x00, 0x00, 0x40,
      0x00, 0x01, 0x00, 0x00, 0xc0, 0x14, 0x10, 0x00,
      0x60, 0x01, 0x00, 0x60,
   };
   uint8_t indirect_bytes[sizeof(expected_indirect) + 2];
   memset(indirect_bytes, 0x4d, sizeof(indirect_bytes));
   struct agx3_compute_dispatch_sink indirect_sink = {
      .data = indirect_bytes,
      .capacity = sizeof(indirect_bytes),
      .length = 1,
   };
   CHECK(agx3_g17p_native_indirect_compute_dispatch_append(
      &indirect_program, &indirect, &indirect_sink));
   CHECK(indirect_sink.length == 1 + sizeof(expected_indirect));
   CHECK(indirect_bytes[0] == 0x4d &&
         indirect_bytes[indirect_sink.length] == 0x4d);
   CHECK(memcmp(indirect_bytes + 1, expected_indirect,
                sizeof(expected_indirect)) == 0);

   const uint8_t indirect_before[sizeof(indirect_bytes)] = {
      [0 ... sizeof(indirect_bytes) - 1] = 0x4d,
   };
   struct agx3_g17p_native_indirect_dispatch bad_indirect = indirect;
   bad_indirect.helper_program_va |= 1;
   struct agx3_compute_dispatch_sink bad_indirect_sink = {
      .data = indirect_bytes,
      .capacity = sizeof(indirect_bytes),
   };
   memcpy(indirect_bytes, indirect_before, sizeof(indirect_bytes));
   CHECK(!agx3_g17p_native_indirect_compute_dispatch_append(
      &indirect_program, &bad_indirect, &bad_indirect_sink));
   CHECK(bad_indirect_sink.length == 0 &&
         memcmp(indirect_bytes, indirect_before, sizeof(indirect_bytes)) == 0);

   uint8_t before[sizeof(dispatch)];
   memcpy(before, dispatch, sizeof(dispatch));
   struct agx3_compute_dispatch_sink failed = {
      .data = dispatch,
      .capacity = sizeof(dispatch),
   };
   struct agx3_compute_direct_grid invalid_grid = grid;
   invalid_grid.dimensions = 1;
   CHECK(!agx3_g17p_native_compute_dispatch_append(
      &program, &invalid_grid, &failed));
   CHECK(failed.length == 0 &&
         memcmp(dispatch, before, sizeof(dispatch)) == 0);
   invalid_grid = grid;
   invalid_grid.local_size[0] = 0;
   CHECK(!agx3_g17p_native_compute_dispatch_append(
      &program, &invalid_grid, &failed));
   CHECK(failed.length == 0 &&
         memcmp(dispatch, before, sizeof(dispatch)) == 0);

   const struct agx3_compute_program program_before = program;
   struct agx3_g17p_native_compute_program_desc invalid_desc = desc;
   invalid_desc.program_va |= 1;
   CHECK(!agx3_g17p_native_compute_program_bind(&invalid_desc, &program));
   CHECK(memcmp(&program, &program_before, sizeof(program)) == 0);
   return true;
}

static bool
test_g17p_direct_add3_program_join(void)
{
   struct agx3_cdm_direct_template_sources sources;
   CHECK(agx3_cdm_direct_template_sources_compute(false, false, false,
                                                   &sources));
   const struct agx3_g17p_direct_add3_program_desc desc = {
      .program_va = AGX_IOTO_G17P_DIRECT_ADD3_WITNESS_PROGRAM_VA,
      .shader_va = AGX_IOTO_G17P_DIRECT_ADD3_WITNESS_SHADER_VA,
      .launch_sources = sources,
   };

   uint8_t composed[AGX_IOTO_G17P_DIRECT_ADD3_PROGRAM_SIZE + 1];
   memset(composed, 0xa5, sizeof(composed));
   size_t composed_written = SIZE_MAX;
   struct agx3_compute_program composed_program;
   CHECK(agx3_g17p_direct_add3_program_prepare(
      &desc, composed, sizeof(composed), &composed_written,
      &composed_program));
   CHECK(composed_written == AGX_IOTO_G17P_DIRECT_ADD3_PROGRAM_SIZE);
   CHECK(composed[composed_written] == 0xa5);
   CHECK(composed_program.chip == AGX_CHIP_G17P);
   CHECK(composed_program.program_va == desc.program_va);
   CHECK(composed_program.shader_va == desc.shader_va);
   CHECK(!composed_program.g17p_direct_add3_live_admitted);

   _Alignas(AGX_IOTO_PROGRAM_ALIGNMENT)
      uint8_t mapped[AGX_IOTO_G17P_DIRECT_ADD3_PROGRAM_SIZE + 1];
   memset(mapped, 0x3c, sizeof(mapped));
   const struct agx3_compute_program_upload upload = {
      .cpu = mapped,
      .capacity = sizeof(mapped),
      .gpu_va = desc.program_va,
   };
   size_t mapped_written = SIZE_MAX;
   struct agx3_compute_program live_program;
   CHECK(agx3_g17p_direct_add3_program_upload_prepare(
      AGX_CHIP_G17P, &desc, &upload, &mapped_written, &live_program));
   CHECK(mapped_written == composed_written);
   CHECK(memcmp(mapped, composed, mapped_written) == 0);
   CHECK(mapped[mapped_written] == 0x3c);
   CHECK(live_program.chip == AGX_CHIP_G17P);
   CHECK(live_program.program_va == upload.gpu_va);
   CHECK(live_program.shader_va == desc.shader_va);
   CHECK(live_program.g17p_direct_add3_live_admitted);

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
   CHECK(sizeof(expected_dispatch) == AGX3_G17P_DIRECT_ADD3_DISPATCH_SIZE);

   uint8_t dispatch[AGX3_G17P_DIRECT_ADD3_DISPATCH_SIZE + 3];
   memset(dispatch, 0xb6, sizeof(dispatch));
   struct agx3_compute_dispatch_sink sink = {
      .data = dispatch,
      .capacity = sizeof(dispatch),
      .length = 2,
   };
   CHECK(agx3_g17p_direct_add3_dispatch_append(&live_program, &grid, &sink));
   CHECK(sink.length == 2 + sizeof(expected_dispatch));
   CHECK(dispatch[0] == 0xb6 && dispatch[1] == 0xb6 &&
         dispatch[sink.length] == 0xb6);
   CHECK(memcmp(dispatch + 2, expected_dispatch,
                sizeof(expected_dispatch)) == 0);

   uint8_t dispatch_before[sizeof(dispatch)];
   memcpy(dispatch_before, dispatch, sizeof(dispatch));
   struct agx3_compute_dispatch_sink failed_sink = {
      .data = dispatch,
      .capacity = sizeof(dispatch),
      .length = 1,
   };
   CHECK(!agx3_g17p_direct_add3_dispatch_append(
      &composed_program, &grid, &failed_sink));
   CHECK(failed_sink.length == 1 &&
         memcmp(dispatch, dispatch_before, sizeof(dispatch)) == 0);

   struct agx3_compute_direct_grid invalid_grid = grid;
   invalid_grid.size[0] = 63;
   CHECK(!agx3_g17p_direct_add3_dispatch_append(
      &live_program, &invalid_grid, &failed_sink));
   CHECK(failed_sink.length == 1 &&
         memcmp(dispatch, dispatch_before, sizeof(dispatch)) == 0);

   struct agx3_compute_program sibling_program = live_program;
   sibling_program.chip = AGX_CHIP_G17G;
   CHECK(!agx3_g17p_direct_add3_dispatch_append(
      &sibling_program, &grid, &failed_sink));
   CHECK(failed_sink.length == 1 &&
         memcmp(dispatch, dispatch_before, sizeof(dispatch)) == 0);

   struct agx3_compute_program_upload invalid_upload = upload;
   invalid_upload.gpu_va += AGX_IOTO_PROGRAM_ALIGNMENT;
   mapped_written = 47;
   struct agx3_compute_program program_before = live_program;
   CHECK(!agx3_g17p_direct_add3_program_upload_prepare(
      AGX_CHIP_G17P, &desc, &invalid_upload, &mapped_written,
      &live_program));
   CHECK(mapped_written == 47);
   CHECK(memcmp(&live_program, &program_before, sizeof(live_program)) == 0);
   CHECK(memcmp(mapped, composed, composed_written) == 0);

   struct agx3_g17p_direct_add3_program_desc invalid_desc = desc;
   invalid_desc.launch_sources.max_total_threads_per_threadgroup = 1;
   uint8_t composed_before[sizeof(composed)];
   memcpy(composed_before, composed, sizeof(composed));
   composed_written = 47;
   program_before = composed_program;
   CHECK(!agx3_g17p_direct_add3_program_prepare(
      &invalid_desc, composed, sizeof(composed), &composed_written,
      &composed_program));
   CHECK(composed_written == 47 &&
         memcmp(composed, composed_before, sizeof(composed)) == 0);
   CHECK(memcmp(&composed_program, &program_before,
                sizeof(composed_program)) == 0);

   CHECK(agx3_g17p_direct_add3_dispatch_ready(AGX_CHIP_G17P));
   CHECK(!agx3_g17p_direct_add3_dispatch_ready(AGX_CHIP_G17S));
   return true;
}

static bool
test_g17s_minimal_program_join(void)
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
   uint8_t state[AGX_IOTO_G17S_MINIMAL_COMPUTE_PROGRAM_SIZE + 1];
   memset(state, 0xa5, sizeof(state));
   size_t written = 0;
   struct agx3_compute_program program;
   CHECK(agx3_g17s_minimal_compute_program_prepare(
      &desc, state, sizeof(state), &written, &program));
   CHECK(written == AGX_IOTO_G17S_MINIMAL_COMPUTE_PROGRAM_SIZE);
   CHECK(state[written] == 0xa5);
   CHECK(program.chip == AGX_CHIP_G17S);
   CHECK(program.program_va == desc.program_va);
   CHECK(!program.g17s_minimal_live_admitted);

   const struct agx3_compute_direct_grid grid = {
      .source = AGX3_CDM_DIRECT_GRID_THREADGROUP_COUNTS,
      .size = {2, 3, 4},
      .local_size = {8, 4, 2},
      .dimensions = 3,
   };
   uint8_t launch[AGX3_CDM_DIRECT_LAUNCH_SIZE];
   CHECK(agx3_compute_program_pack_direct_launch(&program, &grid, launch));
   struct agx3_cdm_direct_launch_info decoded;
   CHECK(agx3_cdm_direct_launch_unpack(AGX_CHIP_G17S, launch, sizeof(launch),
                                       &decoded));
   CHECK(decoded.pipeline == desc.program_va);
   CHECK(decoded.global_size[0] == 16 && decoded.global_size[1] == 12 &&
         decoded.global_size[2] == 8);

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
   CHECK(agx3_g17s_minimal_compute_program_upload_prepare(
      AGX_CHIP_G17S, &desc, &upload, &mapped_written, &mapped_program));
   CHECK(mapped_written == written);
   CHECK(memcmp(mapped, state, written) == 0);
   CHECK(mapped[mapped_written] == 0x3c);
   CHECK(mapped_program.chip == AGX_CHIP_G17S);
   CHECK(mapped_program.program_va == upload.gpu_va);
   CHECK(!mapped_program.g17s_minimal_live_admitted);

   CHECK(agx3_compute_program_pack_direct_launch(&mapped_program, &grid,
                                                  launch));
   CHECK(agx3_cdm_direct_launch_unpack(AGX_CHIP_G17S, launch, sizeof(launch),
                                       &decoded));
   CHECK(decoded.pipeline == upload.gpu_va);

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
   CHECK(agx3_g17s_minimal_compute_program_live_upload_prepare(
      AGX_CHIP_G17S, &desc, &resources, &live_upload, &live_written,
      &live_program));
   CHECK(live_written == AGX_IOTO_G17S_MINIMAL_COMPUTE_PROGRAM_SIZE);
   CHECK(live_mapped[live_written] == 0x6d);
   CHECK(memcmp(live_mapped + 32, mapped + 32, live_written - 32) == 0);
   CHECK(memcmp(live_mapped, mapped, 32) != 0);
   CHECK(live_program.chip == AGX_CHIP_G17S &&
         live_program.program_va == live_upload.gpu_va);
   CHECK(live_program.g17s_minimal_live_admitted);
   CHECK(agx3_compute_program_pack_direct_launch(&live_program, &grid,
                                                  launch));
   CHECK(agx3_cdm_direct_launch_unpack(AGX_CHIP_G17S, launch, sizeof(launch),
                                       &decoded));
   CHECK(decoded.pipeline == live_upload.gpu_va);

   uint8_t live_before[sizeof(live_mapped)];
   memcpy(live_before, live_mapped, sizeof(live_mapped));
   const struct agx3_compute_program live_program_before = live_program;
   struct agx_ioto_g17s_minimal_compute_resources invalid_resources =
      resources;
   invalid_resources.absolute_uniform_data_va = 0;
   live_written = 47;
   CHECK(!agx3_g17s_minimal_compute_program_live_upload_prepare(
      AGX_CHIP_G17S, &desc, &invalid_resources, &live_upload, &live_written,
      &live_program));
   CHECK(memcmp(live_mapped, live_before, sizeof(live_mapped)) == 0 &&
         live_written == 47);
   CHECK(memcmp(&live_program, &live_program_before, sizeof(live_program)) ==
         0);

   uint8_t mapped_before[sizeof(mapped)];
   memcpy(mapped_before, mapped, sizeof(mapped));
   struct agx3_compute_program mapped_program_before = mapped_program;
   static const enum agx_chip rejected_upload_targets[] = {
      AGX_CHIP_G15G, AGX_CHIP_G15S, AGX_CHIP_G15C, AGX_CHIP_G15D,
      AGX_CHIP_G16G, AGX_CHIP_G16S, AGX_CHIP_G16C, AGX_CHIP_G17P,
      AGX_CHIP_G17G, AGX_CHIP_G17C, AGX_CHIP_UNKNOWN,
   };
   CHECK(agx3_g17s_minimal_compute_dispatch_ready(AGX_CHIP_G17S));
   for (unsigned i = 0; i < ARRAY_SIZE(rejected_upload_targets); ++i)
      CHECK(!agx3_g17s_minimal_compute_dispatch_ready(
         rejected_upload_targets[i]));

   uint8_t dispatch[AGX3_G17S_MINIMAL_COMPUTE_DISPATCH_SIZE + 5];
   memset(dispatch, 0xa7, sizeof(dispatch));
   struct agx3_compute_dispatch_sink dispatch_sink = {
      .data = dispatch,
      .capacity = sizeof(dispatch),
      .length = 3,
   };
   CHECK(agx3_g17s_minimal_compute_dispatch_append(
      &live_program, &grid, &dispatch_sink));
   CHECK(dispatch_sink.length ==
         3 + AGX3_G17S_MINIMAL_COMPUTE_DISPATCH_SIZE);
   CHECK(dispatch[0] == 0xa7 && dispatch[1] == 0xa7 &&
         dispatch[2] == 0xa7 && dispatch[dispatch_sink.length] == 0xa7);
   CHECK(memcmp(dispatch + 3, launch, AGX3_CDM_DIRECT_LAUNCH_SIZE) == 0);
   static const uint8_t strong_barrier[4] = {
      0x60, 0x15, 0x00, 0x60,
   };
   CHECK(memcmp(dispatch + 3 + AGX3_CDM_DIRECT_LAUNCH_SIZE, strong_barrier,
                sizeof(strong_barrier)) == 0);
   CHECK(agx3_cdm_direct_launch_unpack(
      AGX_CHIP_G17S, dispatch + 3, AGX3_CDM_DIRECT_LAUNCH_SIZE, &decoded));
   CHECK(decoded.pipeline == live_upload.gpu_va);

   uint8_t dispatch_before[sizeof(dispatch)];
   memcpy(dispatch_before, dispatch, sizeof(dispatch));
   struct agx3_compute_dispatch_sink failed_sink = {
      .data = dispatch,
      .capacity = sizeof(dispatch),
      .length = 2,
   };
   CHECK(!agx3_g17s_minimal_compute_dispatch_append(
      &program, &grid, &failed_sink));
   CHECK(failed_sink.length == 2 &&
         memcmp(dispatch, dispatch_before, sizeof(dispatch)) == 0);
   CHECK(!agx3_g17s_minimal_compute_dispatch_append(
      &mapped_program, &grid, &failed_sink));
   CHECK(failed_sink.length == 2 &&
         memcmp(dispatch, dispatch_before, sizeof(dispatch)) == 0);

   struct agx3_compute_program sibling_program = live_program;
   sibling_program.chip = AGX_CHIP_G17G;
   CHECK(!agx3_g17s_minimal_compute_dispatch_append(
      &sibling_program, &grid, &failed_sink));
   CHECK(failed_sink.length == 2 &&
         memcmp(dispatch, dispatch_before, sizeof(dispatch)) == 0);

   failed_sink.length = 3;
   failed_sink.capacity =
      3 + AGX3_G17S_MINIMAL_COMPUTE_DISPATCH_SIZE - 1;
   CHECK(!agx3_g17s_minimal_compute_dispatch_append(
      &live_program, &grid, &failed_sink));
   CHECK(failed_sink.length == 3 &&
         memcmp(dispatch, dispatch_before, sizeof(dispatch)) == 0);
   failed_sink.length = 3;
   failed_sink.capacity = 2;
   CHECK(!agx3_g17s_minimal_compute_dispatch_append(
      &live_program, &grid, &failed_sink));
   CHECK(failed_sink.length == 3 &&
         memcmp(dispatch, dispatch_before, sizeof(dispatch)) == 0);
   failed_sink.data = NULL;
   failed_sink.capacity = AGX3_G17S_MINIMAL_COMPUTE_DISPATCH_SIZE;
   failed_sink.length = 0;
   CHECK(!agx3_g17s_minimal_compute_dispatch_append(
      &live_program, &grid, &failed_sink));
   CHECK(failed_sink.length == 0 &&
         memcmp(dispatch, dispatch_before, sizeof(dispatch)) == 0);

   for (unsigned i = 0; i < ARRAY_SIZE(rejected_upload_targets); ++i) {
      mapped_written = 47;
      CHECK(!agx3_g17s_minimal_compute_program_upload_prepare(
         rejected_upload_targets[i], &desc, &upload, &mapped_written,
         &mapped_program));
      CHECK(memcmp(mapped, mapped_before, sizeof(mapped)) == 0 &&
            mapped_written == 47);
      CHECK(memcmp(&mapped_program, &mapped_program_before,
                   sizeof(mapped_program)) == 0);
      live_written = 47;
      CHECK(!agx3_g17s_minimal_compute_program_live_upload_prepare(
         rejected_upload_targets[i], &desc, &resources, &live_upload,
         &live_written, &live_program));
      CHECK(memcmp(live_mapped, live_before, sizeof(live_mapped)) == 0 &&
            live_written == 47);
      CHECK(memcmp(&live_program, &live_program_before,
                   sizeof(live_program)) == 0);
   }

   struct agx3_compute_program_upload invalid_upload = upload;
   invalid_upload.gpu_va += AGX_IOTO_PROGRAM_ALIGNMENT;
   mapped_written = 47;
   CHECK(!agx3_g17s_minimal_compute_program_upload_prepare(
      AGX_CHIP_G17S, &desc, &invalid_upload, &mapped_written,
      &mapped_program));
   CHECK(memcmp(mapped, mapped_before, sizeof(mapped)) == 0 &&
         mapped_written == 47);
   CHECK(memcmp(&mapped_program, &mapped_program_before,
                sizeof(mapped_program)) == 0);

   invalid_upload = upload;
   invalid_upload.cpu = mapped + 1;
   mapped_written = 47;
   CHECK(!agx3_g17s_minimal_compute_program_upload_prepare(
      AGX_CHIP_G17S, &desc, &invalid_upload, &mapped_written,
      &mapped_program));
   CHECK(memcmp(mapped, mapped_before, sizeof(mapped)) == 0 &&
         mapped_written == 47);
   CHECK(memcmp(&mapped_program, &mapped_program_before,
                sizeof(mapped_program)) == 0);

   invalid_upload = upload;
   invalid_upload.capacity = AGX_IOTO_G17S_MINIMAL_COMPUTE_PROGRAM_SIZE - 1;
   mapped_written = 47;
   CHECK(!agx3_g17s_minimal_compute_program_upload_prepare(
      AGX_CHIP_G17S, &desc, &invalid_upload, &mapped_written,
      &mapped_program));
   CHECK(memcmp(mapped, mapped_before, sizeof(mapped)) == 0 &&
         mapped_written == 47);
   CHECK(memcmp(&mapped_program, &mapped_program_before,
                sizeof(mapped_program)) == 0);

   mapped_written = 47;
   CHECK(!agx3_g17s_minimal_compute_program_upload_prepare(
      AGX_CHIP_G17S, &desc, NULL, &mapped_written, &mapped_program));
   CHECK(memcmp(mapped, mapped_before, sizeof(mapped)) == 0 &&
         mapped_written == 47);
   CHECK(memcmp(&mapped_program, &mapped_program_before,
                sizeof(mapped_program)) == 0);

   uint8_t before[sizeof(state)];
   memcpy(before, state, sizeof(state));
   struct agx3_compute_program program_before = program;
   written = 47;
   struct agx3_g17s_minimal_compute_program_desc invalid = desc;
   invalid.shader_va |= 1;
   CHECK(!agx3_g17s_minimal_compute_program_prepare(
      &invalid, state, sizeof(state), &written, &program));
   CHECK(memcmp(state, before, sizeof(state)) == 0 && written == 47);
   CHECK(memcmp(&program, &program_before, sizeof(program)) == 0);

   static const struct {
      uint32_t user_buffer_binding;
      uint32_t byte_offset;
      uint32_t uniform_data_destination;
      uint32_t uniform_data_count;
      uint32_t temporary_base;
   } invalid_root_uniforms[] = {
      {0, 0, 4, 1, 0},
      {1, 4, 4, 1, 0},
      {1, 0, 2, 1, 0},
      {1, 0, 4, 2, 0},
      {1, 0, 4, 1, 1},
   };

   for (unsigned i = 0; i < ARRAY_SIZE(invalid_root_uniforms); ++i) {
      invalid = desc;
      invalid.root_uniform.user_buffer_binding =
         invalid_root_uniforms[i].user_buffer_binding;
      invalid.root_uniform.byte_offset = invalid_root_uniforms[i].byte_offset;
      invalid.root_uniform.uniform_data_destination =
         invalid_root_uniforms[i].uniform_data_destination;
      invalid.root_uniform.uniform_data_count =
         invalid_root_uniforms[i].uniform_data_count;
      invalid.root_uniform.temporary_base =
         invalid_root_uniforms[i].temporary_base;
      written = 47;
      CHECK(!agx3_g17s_minimal_compute_program_prepare(
         &invalid, state, sizeof(state), &written, &program));
      CHECK(memcmp(state, before, sizeof(state)) == 0 && written == 47);
      CHECK(memcmp(&program, &program_before, sizeof(program)) == 0);
   }
   return true;
}

static bool
prepare_fails_unchanged(struct agx3_compute_program_desc *desc, size_t capacity)
{
   uint8_t state[96], before[sizeof(state)];
   memset(state, 0x5a, sizeof(state));
   memcpy(before, state, sizeof(state));
   size_t written = 37;
   struct agx3_compute_program program, program_before;
   memset(&program, 0xa5, sizeof(program));
   program_before = program;
   CHECK(
      !agx3_compute_program_prepare(desc, state, capacity, &written, &program));
   CHECK(memcmp(state, before, sizeof(state)) == 0);
   CHECK(written == 37);
   CHECK(memcmp(&program, &program_before, sizeof(program)) == 0);
   return true;
}

static bool
test_fail_closed(void)
{
   struct agx3_compute_program_desc desc = make_desc(AGX_CHIP_G17P);
   CHECK(prepare_fails_unchanged(NULL, sizeof((uint8_t[96]){0})));

   desc.profile.chip = AGX_CHIP_G17G;
   CHECK(prepare_fails_unchanged(&desc, 96));
   desc.profile.chip = desc.chip;
   desc.program_va |= 1;
   CHECK(prepare_fails_unchanged(&desc, 96));
   desc.program_va &= ~UINT64_C(63);
   desc.launch_sources.extended_texture_write_mask_qword_1 = UINT32_MAX;
   CHECK(prepare_fails_unchanged(&desc, 96));
   desc = make_desc(AGX_CHIP_G17P);
   CHECK(prepare_fails_unchanged(&desc, 1));
   desc = make_desc(AGX_CHIP_G14X);
   CHECK(prepare_fails_unchanged(&desc, 96));

   desc = make_desc(AGX_CHIP_G17P);
   uint8_t state[96];
   size_t written;
   struct agx3_compute_program program;
   CHECK(agx3_compute_program_prepare(&desc, state, sizeof(state), &written,
                                      &program));
   struct agx3_compute_direct_grid grid = {
      .source = AGX3_CDM_DIRECT_GRID_GLOBAL_THREADS,
      .size = {1, 1, 1},
      .local_size = {1, 1, 1},
      .dimensions = 1,
   };
   uint8_t launch[AGX3_CDM_DIRECT_LAUNCH_SIZE];
   memset(launch, 0xa5, sizeof(launch));
   CHECK(!agx3_compute_program_pack_direct_launch(NULL, &grid, launch));
   CHECK(bytes_are(launch, sizeof(launch), 0));
   memset(launch, 0xa5, sizeof(launch));
   CHECK(!agx3_compute_program_pack_direct_launch(&program, NULL, launch));
   CHECK(bytes_are(launch, sizeof(launch), 0));

   grid.local_size[0] = 0x800;
   memset(launch, 0xa5, sizeof(launch));
   CHECK(!agx3_compute_program_pack_direct_launch(&program, &grid, launch));
   CHECK(bytes_are(launch, sizeof(launch), 0));
   grid.local_size[0] = 1;
   grid.g15x_has_multiple_agcs = true;
   memset(launch, 0xa5, sizeof(launch));
   CHECK(!agx3_compute_program_pack_direct_launch(&program, &grid, launch));
   CHECK(bytes_are(launch, sizeof(launch), 0));

   program.program_va |= 1;
   memset(launch, 0xa5, sizeof(launch));
   CHECK(!agx3_compute_program_pack_direct_launch(&program, &grid, launch));
   CHECK(bytes_are(launch, sizeof(launch), 0));
   CHECK(!agx3_compute_program_pack_direct_launch(&program, &grid, NULL));
   return true;
}

/* Every witnessed G17P launch record was captured with a program VA that has
 * bit 40 set (0x100000a8000, 0x100000b0000), where word 1's `(va >> 38) << 22`
 * and word 3's `va[47:40]` BOTH evaluate to the same 2^40 contribution
 * (4 << 38 == 1 << 40).  Hard-coding either one therefore "reproduces" the
 * witness while injecting 2^40 into any bit-40-clear VA -- which is every
 * address in Mesa's USC window at 0x11xxxxxxxx.  Hardware proved this for
 * word 1 (program at 0x11fffac100 fetched from 0x111fffac100, FAULT_ADDR
 * 0x447ffeb04).  Pin BOTH fields as VA-derived against a USC-window VA:
 *
 *   word 1 bits 22..31 == program_va[47:38]  (derived, so 0 here)
 *   word at +8         == program_va >> 6    (truncated to 32 bits)
 *   word at +12        == 0x40000000 | program_va[47:40]  (so 0x40000000 here)
 *
 * and the same for the indirect record's pointer pair.  Under any OR of the
 * two overlapping high fields the reconstruction is then exactly the VA.
 */
static uint32_t
load_le32(const uint8_t *bytes)
{
   return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) |
          ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
}

static bool
test_g17p_usc_window_va_is_encoded_without_constants(void)
{
   /* A real USC-window address: dev->shader_base is 0x1100000000 on T8140
    * (ALIGN_POT(2^36 + printf, 4 GiB)) and util_vma_heap allocates high, so
    * the first Ioto program BO lands just under shader_base + 4 GiB. */
   const uint64_t program_va = UINT64_C(0x11ffff8000);
   const uint64_t shader_va = UINT64_C(0x11ffff0000);

   struct agx3_cdm_direct_template_sources sources;
   CHECK(agx3_cdm_direct_template_sources_compute(false, false, false,
                                                   &sources));
   const struct agx3_g17p_native_compute_program_desc desc = {
      .program_va = program_va,
      .shader_va = shader_va,
      .launch_sources = sources,
   };
   struct agx3_compute_program program;
   CHECK(agx3_g17p_native_compute_program_bind(&desc, &program));

   const struct agx3_compute_direct_grid grid = {
      .source = AGX3_CDM_DIRECT_GRID_GLOBAL_THREADS,
      .size = {64, 1, 1},
      .local_size = {32, 1, 1},
      .dimensions = 1,
   };
   uint8_t dispatch[AGX3_G17P_NATIVE_COMPUTE_DISPATCH_SIZE];
   struct agx3_compute_dispatch_sink sink = {
      .data = dispatch,
      .capacity = sizeof(dispatch),
   };
   CHECK(agx3_g17p_native_compute_dispatch_append(&program, &grid, &sink));

   /* bit 40 is clear, so the derived high field must be zero -- the old
    * hard-coded 0x01000000 would have injected 2^40 into the fetch. */
   CHECK((load_le32(dispatch + 4) >> 22) == 0);
   CHECK(load_le32(dispatch + 8) == (uint32_t)(program_va >> 6));
   CHECK(load_le32(dispatch + 12) == UINT32_C(0x40000000));

   /* The hardware reconstruction must land exactly on the program VA under
    * the OR of BOTH high fields -- word 1 bits 22..31 at bit 38 and word 3's
    * low byte at bit 40 -- because the witness cannot distinguish which of
    * them the hardware consumes. */
   const uint64_t reconstructed =
      ((uint64_t)(load_le32(dispatch + 4) >> 22) << 38) |
      ((uint64_t)load_le32(dispatch + 8) << 6) |
      ((uint64_t)(load_le32(dispatch + 12) & 0xff) << 40);
   CHECK(reconstructed == program_va);

   /* Indirect record: same two fields, plus the helper's pointer pair. */
   const struct agx3_g17p_native_indirect_dispatch indirect = {
      .helper_program_va = UINT64_C(0x11fffe8000),
      .geometry_va = UINT64_C(0x11fffe0000),
      .inline_global_size = {64, 1, 1},
      .local_size = {32, 1, 1},
   };
   uint8_t indirect_bytes[AGX3_G17P_NATIVE_INDIRECT_COMPUTE_DISPATCH_SIZE];
   struct agx3_compute_dispatch_sink indirect_sink = {
      .data = indirect_bytes,
      .capacity = sizeof(indirect_bytes),
   };
   CHECK(agx3_g17p_native_indirect_compute_dispatch_append(
      &program, &indirect, &indirect_sink));
   CHECK((load_le32(indirect_bytes + 4) >> 22) == 0);
   CHECK(load_le32(indirect_bytes + 8) == (uint32_t)(program_va >> 6));
   CHECK(load_le32(indirect_bytes + 12) == UINT32_C(0x40000000));

   const uint8_t *helper =
      indirect_bytes + AGX3_G17P_NATIVE_COMPUTE_DISPATCH_SIZE;
   CHECK((load_le32(helper + 4) >> 22) == 0);
   CHECK(load_le32(helper + 8) ==
         (uint32_t)(indirect.helper_program_va >> 6));
   CHECK(load_le32(helper + 12) == UINT32_C(0x40000000));
   return true;
}

static bool
test_g17g_native_compute_join(void)
{
   struct agx3_cdm_direct_template_sources sources;
   CHECK(agx3_cdm_direct_template_sources_compute(false, false, false, &sources));
   struct agx3_g17p_native_compute_program_desc desc = {
      .program_va = UINT64_C(0x10000090000),
      .shader_va = UINT64_C(0x100000003c0),
      .launch_sources = sources,
   };
   struct agx3_compute_program program;
   CHECK(agx3_g17g_native_compute_program_bind(&desc, &program));
   CHECK(program.chip == AGX_CHIP_G17G && program.g17g_native_compute_live_bound);
   CHECK(!program.g17p_native_compute_live_bound);
   const struct agx3_compute_direct_grid grid = {
      .source = AGX3_CDM_DIRECT_GRID_GLOBAL_THREADS,
      .size = {1024, 1, 1}, .local_size = {64, 1, 1}, .dimensions = 1,
   };
   /* Captured M5 native add program, through its post-dispatch barrier.
    * The enclosing command stream supplies the final terminate record. */
   static const uint32_t expected[] = {
      0x00080000, 0x01000000, 0x00002400, 0x40000001,
      0x00000400, 1, 1, 0x40, 1, 1, 0x60000160,
   };
   uint8_t bytes[AGX3_G17G_NATIVE_COMPUTE_DISPATCH_SIZE + 2];
   memset(bytes, 0xa5, sizeof(bytes));
   struct agx3_compute_dispatch_sink sink = {
      .data = bytes, .capacity = sizeof(bytes), .length = 1,
   };
   CHECK(agx3_g17g_native_compute_dispatch_append(&program, &grid, 0, &sink));
   CHECK(sink.length == 1 + sizeof(expected));
   for (unsigned i = 0; i < sizeof(expected) / sizeof(expected[0]); i++)
      CHECK(load_le32(bytes + 1 + 4*i) == expected[i]);
   CHECK(bytes[0] == 0xa5 && bytes[sink.length] == 0xa5);
   sink.length = 0;
   CHECK(!agx3_g17p_native_compute_dispatch_append(&program, &grid, &sink));
   program.g17g_native_compute_live_bound = false;
   CHECK(!agx3_g17g_native_compute_dispatch_append(&program, &grid, 0, &sink));
   return true;
}

int
main(void)
{
   if (!test_prepare_and_launch() || !test_a18_pro_one_dimensional_launch() ||
       !test_g17p_native_compute_join() || !test_g17g_native_compute_join() ||
       !test_g17p_usc_window_va_is_encoded_without_constants() ||
       !test_g17p_direct_add3_program_join() ||
       !test_g17s_minimal_program_join() || !test_fail_closed())
      return 1;

   printf("agx3_compute_program_core: %u checks passed\n", checks);
   return 0;
}
