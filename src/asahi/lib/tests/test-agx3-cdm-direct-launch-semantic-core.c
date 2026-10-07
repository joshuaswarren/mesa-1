/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "../agx3_cdm_direct_launch_semantic_core.h"

#include "asahi/genxml/agx3_cmdstream.h"

#include <stdio.h>
#include <string.h>

static unsigned checks;

#define CHECK(x)                                                               \
   do {                                                                        \
      checks++;                                                                \
      if (!(x)) {                                                              \
         fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x);                  \
         return 1;                                                             \
      }                                                                        \
   } while (0)

#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))

static const enum agx_chip grounded_targets[] = {
   AGX_CHIP_G15G, AGX_CHIP_G15S, AGX_CHIP_G15C, AGX_CHIP_G15D,
   AGX_CHIP_G16G, AGX_CHIP_G16S, AGX_CHIP_G16C, AGX_CHIP_G17P,
   AGX_CHIP_G17G, AGX_CHIP_G17S, AGX_CHIP_G17C,
};

static const enum agx_chip unsupported_targets[] = {
   AGX_CHIP_G13G,    AGX_CHIP_G13X,    AGX_CHIP_G14G,
   AGX_CHIP_G14X,    AGX_CHIP_G15P,    AGX_CHIP_G16P,
   AGX_CHIP_G17A,    AGX_CHIP_UNKNOWN, (enum agx_chip)-1,
   (enum agx_chip)999,
};

static bool
bytes_are_zero(const void *data, size_t size)
{
   const uint8_t *bytes = data;
   for (size_t i = 0; i < size; ++i) {
      if (bytes[i] != 0)
         return false;
   }
   return true;
}

static bool
template_is_zero(const struct agx3_cdm_direct_template *template_words)
{
   const struct agx3_cdm_direct_template zero = {0};
   return memcmp(template_words, &zero, sizeof(zero)) == 0;
}

static bool
fields_are_zero(const struct agx3_cdm_direct_template_fields *fields)
{
   const struct agx3_cdm_direct_template_fields zero = {0};
   return memcmp(fields, &zero, sizeof(zero)) == 0;
}

static bool
info_is_zero(const struct agx3_cdm_direct_launch_info *info)
{
   const struct agx3_cdm_direct_launch_info zero = {0};
   return memcmp(info, &zero, sizeof(zero)) == 0;
}

static bool
chip_has_g15x_bit31(enum agx_chip chip)
{
   return chip == AGX_CHIP_G15S || chip == AGX_CHIP_G15C ||
          chip == AGX_CHIP_G15D;
}

static uint32_t
expected_dimension_word(unsigned dimensions)
{
   static const uint32_t words[] = {0, 0x28, 0x30, 0x34};
   return words[dimensions];
}

static struct agx3_cdm_direct_launch_info
make_info(enum agx_chip chip, uint8_t dimensions)
{
   struct agx3_cdm_direct_template template_words;
   const struct agx3_cdm_direct_template_fields fields = {
      .word0_bits_3_14 = 0xabc,
      .word1_bits_3_20 = 0x23456,
      .ray_traversal_depth_nonzero = true,
      .subkernel_x_target_workitems = true,
   };
   if (!agx3_cdm_direct_template_build(&fields, &template_words))
      return (struct agx3_cdm_direct_launch_info){0};

   return (struct agx3_cdm_direct_launch_info){
      .chip = chip,
      .template_words = template_words,
      .pipeline = UINT64_C(0x123456789a80),
      .dimensions = dimensions,
      .global_size = {UINT32_C(0x10203040), 2, 3},
      .local_size = {1, 32, 0x7ff},
   };
}

static int
test_template_contract(void)
{
   const struct agx3_cdm_direct_template_fields zero_fields = {0};
   struct agx3_cdm_direct_template template_words;
   memset(&template_words, 0xa5, sizeof(template_words));
   CHECK(agx3_cdm_direct_template_build(&zero_fields, &template_words));
   CHECK(template_is_zero(&template_words));

   const struct agx3_cdm_direct_template_fields max_fields = {
      .word0_bits_3_14 = 0xffe,
      .word1_bits_3_20 = 0x3ffff,
      .ray_traversal_depth_nonzero = true,
      .subkernel_x_target_workitems = true,
   };
   CHECK(agx3_cdm_direct_template_build(&max_fields, &template_words));
   CHECK(template_words.word0 == AGX3_CDM_DIRECT_TEMPLATE_WORD0_MASK);
   CHECK(template_words.word1 == AGX3_CDM_DIRECT_TEMPLATE_WORD1_MASK);

   struct agx3_cdm_direct_template_fields decoded;
   CHECK(agx3_cdm_direct_template_decode(&template_words, &decoded));
   CHECK(decoded.word0_bits_3_14 == max_fields.word0_bits_3_14);
   CHECK(decoded.word1_bits_3_20 == max_fields.word1_bits_3_20);
   CHECK(decoded.ray_traversal_depth_nonzero ==
         max_fields.ray_traversal_depth_nonzero);
   CHECK(decoded.subkernel_x_target_workitems ==
         max_fields.subkernel_x_target_workitems);

   const struct agx3_cdm_direct_template_fields invalid_fields[] = {
      {.word0_bits_3_14 = 1},
      {.word0_bits_3_14 = 0xfff},
      {.word0_bits_3_14 = 0x1000},
      {.word1_bits_3_20 = 0x40000},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(invalid_fields); ++i) {
      memset(&template_words, 0xa5, sizeof(template_words));
      CHECK(!agx3_cdm_direct_template_build(&invalid_fields[i],
                                             &template_words));
      CHECK(template_is_zero(&template_words));
   }

   for (unsigned bit = 0; bit < 32; ++bit) {
      const struct agx3_cdm_direct_template word0 = {
         .word0 = UINT32_C(1) << bit,
      };
      memset(&decoded, 0xa5, sizeof(decoded));
      const bool word0_allowed =
         (AGX3_CDM_DIRECT_TEMPLATE_WORD0_MASK & (UINT32_C(1) << bit)) != 0;
      CHECK(agx3_cdm_direct_template_decode(&word0, &decoded) ==
            word0_allowed);
      if (!word0_allowed)
         CHECK(fields_are_zero(&decoded));

      const struct agx3_cdm_direct_template word1 = {
         .word1 = UINT32_C(1) << bit,
      };
      memset(&decoded, 0xa5, sizeof(decoded));
      const bool word1_allowed =
         (AGX3_CDM_DIRECT_TEMPLATE_WORD1_MASK & (UINT32_C(1) << bit)) != 0;
      CHECK(agx3_cdm_direct_template_decode(&word1, &decoded) ==
            word1_allowed);
      if (!word1_allowed)
         CHECK(fields_are_zero(&decoded));
   }

   memset(&template_words, 0xa5, sizeof(template_words));
   CHECK(!agx3_cdm_direct_template_build(NULL, &template_words));
   CHECK(template_is_zero(&template_words));
   CHECK(!agx3_cdm_direct_template_build(&zero_fields, NULL));

   memset(&decoded, 0xa5, sizeof(decoded));
   CHECK(!agx3_cdm_direct_template_decode(NULL, &decoded));
   CHECK(fields_are_zero(&decoded));
   CHECK(!agx3_cdm_direct_template_decode(&template_words, NULL));

   const struct agx3_cdm_direct_template_sources source_fields = {
      .ti_spill_bytes = 5,
      .ti_largest_ipr_bytes = 4,
      .ray_traversal_depth_nonzero = true,
   };
   CHECK(agx3_cdm_direct_template_build_from_sources(&source_fields,
                                                      &template_words));
   CHECK(agx3_cdm_direct_template_decode(&template_words, &decoded));
   CHECK(decoded.word0_bits_3_14 == 2);
   CHECK(decoded.word1_bits_3_20 == 2);
   CHECK(decoded.ray_traversal_depth_nonzero);
   CHECK(decoded.subkernel_x_target_workitems);

   const struct {
      bool resource_info;
      bool ray_traversal;
      uint32_t field31;
   } compute_sources[] = {
      {false, false, 0x00},
      {true, false, 0x20},
      {false, true, 0x30},
      {true, true, 0x40},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(compute_sources); ++i) {
      struct agx3_cdm_direct_template_sources sources;
      CHECK(agx3_cdm_direct_template_sources_compute(
         compute_sources[i].resource_info, compute_sources[i].ray_traversal,
         false, &sources));
      CHECK(sources.ti_spill_bytes ==
            compute_sources[i].field31);
      CHECK(sources.ti_largest_ipr_bytes == 0);
      CHECK(!sources.ray_traversal_depth_nonzero);
      CHECK(agx3_cdm_direct_template_build_from_sources(&sources,
                                                         &template_words));
      CHECK(agx3_cdm_direct_template_decode(&template_words, &decoded));
      CHECK(decoded.word1_bits_3_20 ==
            DIV_ROUND_UP(compute_sources[i].field31, 4));
      CHECK(!decoded.ray_traversal_depth_nonzero);

      if (compute_sources[i].ray_traversal) {
         CHECK(agx3_cdm_direct_template_sources_compute(
            compute_sources[i].resource_info, true, true, &sources));
         CHECK(sources.ti_spill_bytes ==
               compute_sources[i].field31);
         CHECK(sources.ray_traversal_depth_nonzero);
      } else {
         CHECK(!agx3_cdm_direct_template_sources_compute(
            compute_sources[i].resource_info, false, true, &sources));
      }
   }
   CHECK(!agx3_cdm_direct_template_sources_compute(false, false, false,
                                                    NULL));

   const uint32_t extended_mask_boundaries[][2] = {
      {0, 0}, {1, 2}, {4, 2}, {5, 2}, {8, 2},
      {9, 4}, {UINT32_C(0x3ff8), 0xffe},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(extended_mask_boundaries); ++i) {
      struct agx3_cdm_direct_template_sources sources = {
         .ti_largest_ipr_bytes =
            extended_mask_boundaries[i][0],
      };
      CHECK(agx3_cdm_direct_template_build_from_sources(&sources,
                                                         &template_words));
      CHECK(agx3_cdm_direct_template_decode(&template_words, &decoded));
      CHECK(decoded.word0_bits_3_14 == extended_mask_boundaries[i][1]);
   }

   const uint32_t max_threads_boundaries[][2] = {
      {0, 0}, {1, 1}, {4, 1}, {5, 2},
      {UINT32_C(0xffffc), 0x3ffff},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(max_threads_boundaries); ++i) {
      struct agx3_cdm_direct_template_sources sources = {
         .ti_spill_bytes = max_threads_boundaries[i][0],
      };
      CHECK(agx3_cdm_direct_template_build_from_sources(&sources,
                                                         &template_words));
      CHECK(agx3_cdm_direct_template_decode(&template_words, &decoded));
      CHECK(decoded.word1_bits_3_20 == max_threads_boundaries[i][1]);
   }

   const struct agx3_cdm_direct_template_sources invalid_sources[] = {
      {.ti_largest_ipr_bytes = UINT32_C(0x3ff9)},
      {.ti_largest_ipr_bytes = UINT32_MAX},
      {.ti_spill_bytes = UINT32_C(0xffffd)},
      {.ti_spill_bytes = UINT32_MAX},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(invalid_sources); ++i) {
      memset(&template_words, 0xa5, sizeof(template_words));
      CHECK(!agx3_cdm_direct_template_build_from_sources(
         &invalid_sources[i], &template_words));
      CHECK(template_is_zero(&template_words));
   }
   memset(&template_words, 0xa5, sizeof(template_words));
   CHECK(!agx3_cdm_direct_template_build_from_sources(NULL,
                                                       &template_words));
   CHECK(template_is_zero(&template_words));
   CHECK(!agx3_cdm_direct_template_build_from_sources(&source_fields, NULL));
   return 0;
}

static int
test_direct_grid_sources(void)
{
   const uint32_t group_counts[3] = {7, 8, 9};
   const uint32_t local_size[3] = {4, 5, 6};
   const uint32_t thread_counts[3] = {28, 40, 54};
   uint32_t resolved[3];

   memset(resolved, 0xa5, sizeof(resolved));
   CHECK(agx3_cdm_direct_grid_resolve(
      AGX3_CDM_DIRECT_GRID_THREADGROUP_COUNTS, group_counts, local_size,
      resolved));
   CHECK(memcmp(resolved, thread_counts, sizeof(resolved)) == 0);

   memset(resolved, 0xa5, sizeof(resolved));
   CHECK(agx3_cdm_direct_grid_resolve(
      AGX3_CDM_DIRECT_GRID_GLOBAL_THREADS, thread_counts, local_size,
      resolved));
   CHECK(memcmp(resolved, thread_counts, sizeof(resolved)) == 0);

   /* The API provenance is not encoded: equivalent source grids must produce
    * byte-identical canonical records. */
   struct agx3_cdm_direct_launch_info group_info =
      make_info(AGX_CHIP_G17S, 3);
   CHECK(agx3_cdm_direct_grid_resolve(
      AGX3_CDM_DIRECT_GRID_THREADGROUP_COUNTS, group_counts, local_size,
      group_info.global_size));
   memcpy(group_info.local_size, local_size, sizeof(local_size));
   struct agx3_cdm_direct_launch_info thread_info = group_info;
   CHECK(agx3_cdm_direct_grid_resolve(
      AGX3_CDM_DIRECT_GRID_GLOBAL_THREADS, thread_counts, local_size,
      thread_info.global_size));
   uint8_t group_record[AGX3_CDM_DIRECT_LAUNCH_SIZE];
   uint8_t thread_record[AGX3_CDM_DIRECT_LAUNCH_SIZE];
   CHECK(agx3_cdm_direct_launch_pack(&group_info, group_record));
   CHECK(agx3_cdm_direct_launch_pack(&thread_info, thread_record));
   CHECK(memcmp(group_record, thread_record, sizeof(group_record)) == 0);

   /* Exact input/output aliasing remains transactional. */
   uint32_t aliased[3] = {7, 8, 9};
   CHECK(agx3_cdm_direct_grid_resolve(
      AGX3_CDM_DIRECT_GRID_THREADGROUP_COUNTS, aliased, local_size, aliased));
   CHECK(memcmp(aliased, thread_counts, sizeof(aliased)) == 0);
   uint32_t aliased_local[3] = {4, 5, 6};
   CHECK(agx3_cdm_direct_grid_resolve(
      AGX3_CDM_DIRECT_GRID_GLOBAL_THREADS, thread_counts, aliased_local,
      aliased_local));
   CHECK(memcmp(aliased_local, thread_counts, sizeof(aliased_local)) == 0);

   for (unsigned axis = 0; axis < 3; ++axis) {
      uint32_t overflowing_groups[3] = {1, 1, 1};
      uint32_t overflow_local[3] = {1, 1, 1};
      overflowing_groups[axis] = UINT32_MAX;
      overflow_local[axis] = 2;
      memset(resolved, 0xa5, sizeof(resolved));
      CHECK(!agx3_cdm_direct_grid_resolve(
         AGX3_CDM_DIRECT_GRID_THREADGROUP_COUNTS, overflowing_groups,
         overflow_local, resolved));
      CHECK(bytes_are_zero(resolved, sizeof(resolved)));

      uint32_t invalid_local[3] = {1, 1, 1};
      invalid_local[axis] = 0x800;
      memset(resolved, 0xa5, sizeof(resolved));
      CHECK(!agx3_cdm_direct_grid_resolve(
         AGX3_CDM_DIRECT_GRID_GLOBAL_THREADS, thread_counts, invalid_local,
         resolved));
      CHECK(bytes_are_zero(resolved, sizeof(resolved)));
   }

   memset(resolved, 0xa5, sizeof(resolved));
   CHECK(!agx3_cdm_direct_grid_resolve(
      (enum agx3_cdm_direct_grid_source)-1, group_counts, local_size,
      resolved));
   CHECK(bytes_are_zero(resolved, sizeof(resolved)));
   memset(resolved, 0xa5, sizeof(resolved));
   CHECK(!agx3_cdm_direct_grid_resolve(
      (enum agx3_cdm_direct_grid_source)2, group_counts, local_size,
      resolved));
   CHECK(bytes_are_zero(resolved, sizeof(resolved)));

   memset(resolved, 0xa5, sizeof(resolved));
   CHECK(!agx3_cdm_direct_grid_resolve(
      AGX3_CDM_DIRECT_GRID_GLOBAL_THREADS, NULL, local_size, resolved));
   CHECK(bytes_are_zero(resolved, sizeof(resolved)));
   memset(resolved, 0xa5, sizeof(resolved));
   CHECK(!agx3_cdm_direct_grid_resolve(
      AGX3_CDM_DIRECT_GRID_GLOBAL_THREADS, thread_counts, NULL, resolved));
   CHECK(bytes_are_zero(resolved, sizeof(resolved)));
   CHECK(!agx3_cdm_direct_grid_resolve(
      AGX3_CDM_DIRECT_GRID_GLOBAL_THREADS, thread_counts, local_size, NULL));
   return 0;
}

static int
test_exact_witness_and_targets(void)
{
   static const uint8_t expected[AGX3_CDM_DIRECT_LAUNCH_SIZE] = {
      0xe0, 0x55, 0x88, 0x00, 0xb0, 0xa2, 0x11, 0x12,
      0x6a, 0xe2, 0x59, 0xd1, 0x34, 0x00, 0x00, 0x40,
      0x40, 0x30, 0x20, 0x10, 0x02, 0x00, 0x00, 0x00,
      0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
      0x20, 0x00, 0x00, 0x00, 0xff, 0x07, 0x00, 0x00,
   };

   for (unsigned t = 0; t < ARRAY_SIZE(grounded_targets); ++t) {
      const enum agx_chip chip = grounded_targets[t];
      CHECK(agx3_cdm_direct_launch_target_grounded(chip));

      {
         for (unsigned dimensions = 1; dimensions <= 3; ++dimensions) {
            const struct agx3_cdm_direct_launch_info info =
               make_info(chip, dimensions);
            uint8_t packed[AGX3_CDM_DIRECT_LAUNCH_SIZE];
            CHECK(agx3_cdm_direct_launch_pack(&info, packed));
            CHECK(agx3_cmdstream_load_le32(packed) ==
                  info.template_words.word0);
            CHECK((agx3_cmdstream_load_le32(packed + 4) & 0x3fffff) ==
                  info.template_words.word1);
            CHECK(agx3_cmdstream_load_le32(packed + 12) ==
                  (expected_dimension_word(dimensions) |
                   (UINT32_C(1) << 30)));

            struct agx3_cdm_direct_launch_info decoded;
            CHECK(agx3_cdm_direct_launch_unpack(chip, packed, sizeof(packed),
                                                &decoded));
            CHECK(decoded.chip == chip);
            CHECK(decoded.template_words.word0 == info.template_words.word0);
            CHECK(decoded.template_words.word1 == info.template_words.word1);
            CHECK(decoded.pipeline == info.pipeline);
            CHECK(decoded.dimensions == info.dimensions);
            CHECK(!decoded.g15x_has_multiple_agcs);
            CHECK(memcmp(decoded.global_size, info.global_size,
                         sizeof(info.global_size)) == 0);
            CHECK(memcmp(decoded.local_size, info.local_size,
                         sizeof(info.local_size)) == 0);

            struct agx3_cdm_launch_info raw = {
               .chip = chip,
               .word0_payload = info.template_words.word0,
               .word1_payload = info.template_words.word1,
               .pipeline = info.pipeline,
               .opaque_word3 = expected_dimension_word(dimensions) |
                               (UINT32_C(1) << 30),
               .global_size = {info.global_size[0], info.global_size[1],
                               info.global_size[2]},
               .local_size = {info.local_size[0], info.local_size[1],
                              info.local_size[2]},
            };
            uint8_t raw_packed[AGX3_CDM_LAUNCH_SIZE];
            CHECK(agx3_cdm_launch_pack(&raw, raw_packed));
            CHECK(memcmp(packed, raw_packed, sizeof(packed)) == 0);
         }
      }

      struct agx3_cdm_direct_launch_info witness =
         make_info(chip, 3);
      uint8_t packed[AGX3_CDM_DIRECT_LAUNCH_SIZE];
      CHECK(agx3_cdm_direct_launch_pack(&witness, packed));
      CHECK(memcmp(packed, expected, sizeof(expected)) == 0);

      /* The byte form does not identify which grounded plugin family emitted
       * it.  Every directly witnessed identity must accept the same canonical
       * record rather than claim a false cross-generation distinction. */
      for (unsigned other = 0; other < ARRAY_SIZE(grounded_targets); ++other) {
         struct agx3_cdm_direct_launch_info decoded;
         CHECK(agx3_cdm_direct_launch_unpack(grounded_targets[other], packed,
                                             sizeof(packed), &decoded));
         CHECK(decoded.chip == grounded_targets[other]);
      }
   }

   const struct agx3_cdm_direct_launch_info witness =
      make_info(AGX_CHIP_G17S, 3);
   uint8_t packed[AGX3_CDM_DIRECT_LAUNCH_SIZE];
   CHECK(agx3_cdm_direct_launch_pack(&witness, packed));
   for (unsigned t = 0; t < ARRAY_SIZE(unsupported_targets); ++t) {
      const enum agx_chip chip = unsupported_targets[t];
      CHECK(!agx3_cdm_direct_launch_target_grounded(chip));

      struct agx3_cdm_direct_launch_info invalid = witness;
      invalid.chip = chip;
      uint8_t rejected[AGX3_CDM_DIRECT_LAUNCH_SIZE];
      memset(rejected, 0xa5, sizeof(rejected));
      CHECK(!agx3_cdm_direct_launch_pack(&invalid, rejected));
      CHECK(bytes_are_zero(rejected, sizeof(rejected)));

      struct agx3_cdm_direct_launch_info decoded;
      memset(&decoded, 0xa5, sizeof(decoded));
      CHECK(!agx3_cdm_direct_launch_unpack(chip, packed, sizeof(packed),
                                           &decoded));
      CHECK(info_is_zero(&decoded));
   }
   return 0;
}

static int
test_enqueue_adjustment_and_multiple_agcs(void)
{
   for (uint32_t y = 0; y <= 2; ++y) {
      for (uint32_t z = 0; z <= 2; ++z) {
         struct agx3_cdm_direct_launch_info info =
            make_info(AGX_CHIP_G17S, 1);
         info.global_size[1] = y;
         info.global_size[2] = z;
         uint8_t packed[AGX3_CDM_DIRECT_LAUNCH_SIZE];
         CHECK(agx3_cdm_direct_launch_pack(&info, packed));
         const bool cleared = y < 2 && z < 2;
         CHECK(((agx3_cmdstream_load_le32(packed) >> 19) & 1) == !cleared);

         struct agx3_cdm_direct_launch_info decoded;
         CHECK(agx3_cdm_direct_launch_unpack(info.chip, packed, sizeof(packed),
                                             &decoded));
         CHECK(((decoded.template_words.word0 >> 19) & 1) == !cleared);

         if (cleared) {
            agx3_cmdstream_store_le32(
               packed, agx3_cmdstream_load_le32(packed) |
                          (UINT32_C(1) << 19));
            memset(&decoded, 0xa5, sizeof(decoded));
            CHECK(!agx3_cdm_direct_launch_unpack(info.chip, packed,
                                                 sizeof(packed), &decoded));
            CHECK(info_is_zero(&decoded));
         }
      }
   }

   for (unsigned t = 0; t < ARRAY_SIZE(grounded_targets); ++t) {
      const enum agx_chip chip = grounded_targets[t];
      struct agx3_cdm_direct_launch_info info =
         make_info(chip, 2);
      info.g15x_has_multiple_agcs = true;
      uint8_t packed[AGX3_CDM_DIRECT_LAUNCH_SIZE];
      memset(packed, 0xa5, sizeof(packed));
      CHECK(agx3_cdm_direct_launch_pack(&info, packed) ==
            chip_has_g15x_bit31(chip));
      if (chip_has_g15x_bit31(chip)) {
         CHECK(agx3_cmdstream_load_le32(packed + 12) == UINT32_C(0xc0000030));
         struct agx3_cdm_direct_launch_info decoded;
         CHECK(agx3_cdm_direct_launch_unpack(chip, packed, sizeof(packed),
                                             &decoded));
         CHECK(decoded.g15x_has_multiple_agcs);
      } else {
         CHECK(bytes_are_zero(packed, sizeof(packed)));
      }
   }
   return 0;
}

static int
test_boundaries_and_failure_clearing(void)
{
   struct agx3_cdm_direct_launch_info info =
      make_info(AGX_CHIP_G17S, 1);
   uint8_t packed[AGX3_CDM_DIRECT_LAUNCH_SIZE + 1];

   info.pipeline = (UINT64_C(1) << 48) - 64;
   info.local_size[0] = 0;
   info.local_size[1] = 0x7ff;
   CHECK(agx3_cdm_direct_launch_pack(&info, packed));

   const uint64_t invalid_pipelines[] = {
      1, 63, UINT64_C(0x123456789a81), UINT64_C(1) << 48,
      (UINT64_C(1) << 48) + 64,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(invalid_pipelines); ++i) {
      struct agx3_cdm_direct_launch_info invalid = info;
      invalid.pipeline = invalid_pipelines[i];
      memset(packed, 0xa5, AGX3_CDM_DIRECT_LAUNCH_SIZE);
      CHECK(!agx3_cdm_direct_launch_pack(&invalid, packed));
      CHECK(bytes_are_zero(packed, AGX3_CDM_DIRECT_LAUNCH_SIZE));
   }

   const uint8_t invalid_dimensions[] = {0, 4, 255};
   for (unsigned i = 0; i < ARRAY_SIZE(invalid_dimensions); ++i) {
      struct agx3_cdm_direct_launch_info invalid = info;
      invalid.dimensions = invalid_dimensions[i];
      memset(packed, 0xa5, AGX3_CDM_DIRECT_LAUNCH_SIZE);
      CHECK(!agx3_cdm_direct_launch_pack(&invalid, packed));
      CHECK(bytes_are_zero(packed, AGX3_CDM_DIRECT_LAUNCH_SIZE));
   }

   for (unsigned axis = 0; axis < 3; ++axis) {
      struct agx3_cdm_direct_launch_info invalid = info;
      invalid.local_size[axis] = 0x800;
      memset(packed, 0xa5, AGX3_CDM_DIRECT_LAUNCH_SIZE);
      CHECK(!agx3_cdm_direct_launch_pack(&invalid, packed));
      CHECK(bytes_are_zero(packed, AGX3_CDM_DIRECT_LAUNCH_SIZE));
   }

   for (unsigned axis = 0; axis < 3; ++axis) {
      for (uint32_t local = 0; local <= 0x7ff; ++local) {
         struct agx3_cdm_direct_launch_info exhaustive = info;
         exhaustive.local_size[axis] = local;
         CHECK(agx3_cdm_direct_launch_pack(&exhaustive, packed));
         struct agx3_cdm_direct_launch_info decoded;
         CHECK(agx3_cdm_direct_launch_unpack(
            exhaustive.chip, packed, AGX3_CDM_DIRECT_LAUNCH_SIZE, &decoded));
         CHECK(decoded.local_size[axis] == local);
      }
   }

   struct agx3_cdm_direct_launch_info global_boundaries = info;
   global_boundaries.global_size[0] = 0;
   global_boundaries.global_size[1] = UINT32_MAX;
   global_boundaries.global_size[2] = UINT32_MAX;
   CHECK(agx3_cdm_direct_launch_pack(&global_boundaries, packed));
   struct agx3_cdm_direct_launch_info global_decoded;
   CHECK(agx3_cdm_direct_launch_unpack(
      global_boundaries.chip, packed, AGX3_CDM_DIRECT_LAUNCH_SIZE,
      &global_decoded));
   CHECK(memcmp(global_decoded.global_size, global_boundaries.global_size,
                sizeof(global_boundaries.global_size)) == 0);

   struct agx3_cdm_direct_launch_info invalid = info;
   invalid.template_words.word0 |= UINT32_C(1);
   memset(packed, 0xa5, AGX3_CDM_DIRECT_LAUNCH_SIZE);
   CHECK(!agx3_cdm_direct_launch_pack(&invalid, packed));
   CHECK(bytes_are_zero(packed, AGX3_CDM_DIRECT_LAUNCH_SIZE));
   invalid = info;
   invalid.template_words.word1 |= UINT32_C(1) << 21;
   memset(packed, 0xa5, AGX3_CDM_DIRECT_LAUNCH_SIZE);
   CHECK(!agx3_cdm_direct_launch_pack(&invalid, packed));
   CHECK(bytes_are_zero(packed, AGX3_CDM_DIRECT_LAUNCH_SIZE));

   CHECK(agx3_cdm_direct_launch_pack(&info, packed));
   packed[AGX3_CDM_DIRECT_LAUNCH_SIZE] = 0xa5;
   struct agx3_cdm_direct_launch_info decoded;
   CHECK(agx3_cdm_direct_launch_unpack(info.chip, packed, sizeof(packed),
                                       &decoded));
   CHECK(decoded.pipeline == info.pipeline);

   for (size_t size = 0; size < AGX3_CDM_DIRECT_LAUNCH_SIZE; ++size) {
      memset(&decoded, 0xa5, sizeof(decoded));
      CHECK(!agx3_cdm_direct_launch_unpack(info.chip, packed, size, &decoded));
      CHECK(info_is_zero(&decoded));
   }

   uint8_t null_result[AGX3_CDM_DIRECT_LAUNCH_SIZE];
   memset(null_result, 0xa5, sizeof(null_result));
   CHECK(!agx3_cdm_direct_launch_pack(NULL, null_result));
   CHECK(bytes_are_zero(null_result, sizeof(null_result)));
   CHECK(!agx3_cdm_direct_launch_pack(&info, NULL));

   memset(&decoded, 0xa5, sizeof(decoded));
   CHECK(!agx3_cdm_direct_launch_unpack(info.chip, NULL, sizeof(packed),
                                        &decoded));
   CHECK(info_is_zero(&decoded));
   CHECK(!agx3_cdm_direct_launch_unpack(info.chip, packed, sizeof(packed),
                                        NULL));
   return 0;
}

static int
test_record_bit_classification(void)
{
   const struct agx3_cdm_direct_launch_info base =
      make_info(AGX_CHIP_G17S, 1);
   uint8_t canonical[AGX3_CDM_DIRECT_LAUNCH_SIZE];
   CHECK(agx3_cdm_direct_launch_pack(&base, canonical));

   for (unsigned bit = 0; bit < 32; ++bit) {
      uint8_t mutated[AGX3_CDM_DIRECT_LAUNCH_SIZE];
      memcpy(mutated, canonical, sizeof(mutated));
      agx3_cmdstream_store_le32(
         mutated, agx3_cmdstream_load_le32(mutated) ^ (UINT32_C(1) << bit));
      struct agx3_cdm_direct_launch_info decoded;
      memset(&decoded, 0xa5, sizeof(decoded));
      const bool allowed =
         bit < 29 &&
         (AGX3_CDM_DIRECT_TEMPLATE_WORD0_MASK & (UINT32_C(1) << bit));
      CHECK(agx3_cdm_direct_launch_unpack(base.chip, mutated, sizeof(mutated),
                                          &decoded) == allowed);
      if (!allowed)
         CHECK(info_is_zero(&decoded));
   }

   for (unsigned bit = 0; bit < 32; ++bit) {
      uint8_t mutated[AGX3_CDM_DIRECT_LAUNCH_SIZE];
      memcpy(mutated, canonical, sizeof(mutated));
      agx3_cmdstream_store_le32(
         mutated + 4,
         agx3_cmdstream_load_le32(mutated + 4) ^ (UINT32_C(1) << bit));
      struct agx3_cdm_direct_launch_info decoded;
      memset(&decoded, 0xa5, sizeof(decoded));
      const bool allowed =
         bit >= 22 ||
         (AGX3_CDM_DIRECT_TEMPLATE_WORD1_MASK & (UINT32_C(1) << bit));
      CHECK(agx3_cdm_direct_launch_unpack(base.chip, mutated, sizeof(mutated),
                                          &decoded) == allowed);
      if (!allowed)
         CHECK(info_is_zero(&decoded));
   }

   for (unsigned bit = 0; bit < 32; ++bit) {
      uint8_t mutated[AGX3_CDM_DIRECT_LAUNCH_SIZE];
      memcpy(mutated, canonical, sizeof(mutated));
      agx3_cmdstream_store_le32(
         mutated + 8,
         agx3_cmdstream_load_le32(mutated + 8) ^ (UINT32_C(1) << bit));
      struct agx3_cdm_direct_launch_info decoded;
      CHECK(agx3_cdm_direct_launch_unpack(base.chip, mutated, sizeof(mutated),
                                          &decoded));
   }

   for (unsigned bit = 0; bit < 32; ++bit) {
      uint8_t mutated[AGX3_CDM_DIRECT_LAUNCH_SIZE];
      memcpy(mutated, canonical, sizeof(mutated));
      agx3_cmdstream_store_le32(
         mutated + 12,
         agx3_cmdstream_load_le32(mutated + 12) ^ (UINT32_C(1) << bit));
      struct agx3_cdm_direct_launch_info decoded;
      memset(&decoded, 0xa5, sizeof(decoded));
      CHECK(!agx3_cdm_direct_launch_unpack(base.chip, mutated,
                                           sizeof(mutated), &decoded));
      CHECK(info_is_zero(&decoded));
   }

   /* 2D and 3D differ by exactly bit 2; this is a canonical reclassification,
    * not malformed data. */
   struct agx3_cdm_direct_launch_info two_d = base;
   two_d.dimensions = 2;
   CHECK(agx3_cdm_direct_launch_pack(&two_d, canonical));
   agx3_cmdstream_store_le32(
      canonical + 12,
      agx3_cmdstream_load_le32(canonical + 12) ^ (UINT32_C(1) << 2));
   struct agx3_cdm_direct_launch_info decoded;
   CHECK(agx3_cdm_direct_launch_unpack(two_d.chip, canonical,
                                       sizeof(canonical), &decoded));
   CHECK(decoded.dimensions == 3);

   /* Every modeled local-size source bit is accepted.  Every unmodeled bit is
    * rejected and clears the complete result. */
   struct agx3_cdm_direct_launch_info zero_locals = base;
   memset(zero_locals.local_size, 0, sizeof(zero_locals.local_size));
   CHECK(agx3_cdm_direct_launch_pack(&zero_locals, canonical));
   for (unsigned axis = 0; axis < 3; ++axis) {
      for (unsigned bit = 0; bit < 32; ++bit) {
         uint8_t mutated[AGX3_CDM_DIRECT_LAUNCH_SIZE];
         memcpy(mutated, canonical, sizeof(mutated));
         agx3_cmdstream_store_le32(mutated + 28 + axis * 4,
                                   UINT32_C(1) << bit);
         memset(&decoded, 0xa5, sizeof(decoded));
         CHECK(agx3_cdm_direct_launch_unpack(
                  zero_locals.chip, mutated, sizeof(mutated), &decoded) ==
               (bit <= 10));
         if (bit > 10)
            CHECK(info_is_zero(&decoded));
      }
   }
   return 0;
}

int
main(void)
{
   if (test_template_contract() || test_direct_grid_sources() ||
       test_exact_witness_and_targets() ||
       test_enqueue_adjustment_and_multiple_agcs() ||
       test_boundaries_and_failure_clearing() ||
       test_record_bit_classification())
      return 1;

   printf("agx3 CDM direct launch semantic core: %u checks passed\n", checks);
   return 0;
}
