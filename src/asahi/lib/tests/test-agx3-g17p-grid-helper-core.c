/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "../agx3_g17p_grid_helper_core.h"
#include "../agx_ioto_state_core.h"

#include <stdbool.h>
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

/*
 * Independently transcribed from the __TEXT,__text section of
 * `metal-nt -arch applegpu_g17p` output for agx3_g17p_grid_helper_source(),
 * which is byte-identical to the trailing 97 bytes of m1n1's captured G17P
 * GRID_SETUP image (g17p_indirect_code.py, GRID_SETUP_PREAMBLE[192:]).
 */
static const uint8_t apple_grid_helper_text[97] = {
   0x2c, 0x8c, 0x09, 0x27, 0x60, 0x00, 0x00, 0x80,
   0x3c, 0x8e, 0x08, 0x27, 0x0c, 0x88, 0x08, 0x47,
   0x1c, 0x8a, 0x08, 0x47, 0x67, 0x20, 0x54, 0x08,
   0x02, 0x00, 0x00, 0x00, 0x1d, 0x04, 0x00, 0x40,
   0x22, 0x00, 0x67, 0x40, 0x44, 0x00, 0x00, 0x00,
   0x00, 0x00, 0x5d, 0x04, 0x00, 0x40, 0x22, 0x00,
   0x6b, 0x88, 0x01, 0x24, 0x9f, 0x20, 0x54, 0x12,
   0x00, 0x10, 0x00, 0x00, 0xd0, 0x26, 0x0a, 0x00,
   0x7b, 0x8a, 0x01, 0x04, 0x9f, 0x00, 0x54, 0x14,
   0x00, 0x14, 0x08, 0x00, 0xd0, 0x26, 0x0a, 0x00,
   0x8b, 0x8c, 0x01, 0x04, 0x9f, 0x00, 0x54, 0x16,
   0x00, 0x18, 0x10, 0x00, 0xd0, 0x26, 0x0a, 0x00,
   0x0e,
};

static bool
test_shader_image(void)
{
   CHECK(AGX3_G17P_GRID_HELPER_SHADER_SIZE == sizeof(apple_grid_helper_text));

   const uint8_t *image = agx3_g17p_grid_helper_shader_image();
   CHECK(image != NULL);
   CHECK(memcmp(image, apple_grid_helper_text,
                sizeof(apple_grid_helper_text)) == 0);

   /* A G17P shader terminates with the one-byte stop record. */
   CHECK(image[AGX3_G17P_GRID_HELPER_SHADER_SIZE - 1] == 0x0e);

   /* The source string is the compile input, so it must actually name the
    * kernel and write all six geometry words in the documented order. */
   const char *msl = agx3_g17p_grid_helper_source();
   CHECK(msl != NULL);
   CHECK(strstr(msl, "kernel void g17p_grid_setup(") != NULL);
   CHECK(strstr(msl, "geometry[0] = threadgroups[0] * threads_per_group[0];")
         != NULL);
   CHECK(strstr(msl, "geometry[3] = threads_per_group[0];") != NULL);
   return true;
}

static bool
test_ldshdr_kinds(void)
{
   /* The indirect pair announces two different program kinds, and neither is
    * the direct-dispatch kind.  Both are accepted by the runtime tail
    * builder, so a program generator can select them. */
   CHECK(AGX3_G17P_INDIRECT_MAIN_LDSHDR_F1 == 3);
   CHECK(AGX3_G17P_GRID_HELPER_LDSHDR_F1 == 1);
   CHECK(AGX3_G17P_INDIRECT_MAIN_LDSHDR_F1 !=
         AGX_IOTO_G17P_COMPUTE_LDSHDR_F1_DIRECT_ADD3);
   CHECK(AGX3_G17P_GRID_HELPER_LDSHDR_F1 !=
         AGX_IOTO_G17P_COMPUTE_LDSHDR_F1_DIRECT_ADD3);

   struct agx_ioto_state_tail_output tail;
   CHECK(agx_ioto_g17p_runtime_compute_tail_build(
      UINT64_C(0x10000000400), AGX3_G17P_GRID_HELPER_LDSHDR_F1, &tail));
   /* Exactly the captured grid-helper tail: shader 0x10000000400, f1 = 1. */
   static const uint8_t witness[] = {
      0x77, 0x00, 0x2a, 0x41, 0x00, 0x00, 0x00, 0x00,
      0x77, 0x01, 0x2a, 0x08, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00,
      0xf7, 0x00, 0x2a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   CHECK(tail.length == sizeof(witness));
   CHECK(memcmp(tail.bytes, witness, sizeof(witness)) == 0);

   CHECK(agx_ioto_g17p_runtime_compute_tail_build(
      UINT64_C(0x10000000740), AGX3_G17P_INDIRECT_MAIN_LDSHDR_F1, &tail));
   return true;
}

static bool
test_geometry_from_groups(void)
{
   struct agx3_g17p_indirect_geometry geometry;

   const uint32_t groups[3] = {3, 5, 7};
   const uint32_t local[3] = {32, 2, 4};
   CHECK(agx3_g17p_indirect_geometry_from_groups(groups, local, &geometry));
   CHECK(geometry.global_size[0] == 96);
   CHECK(geometry.global_size[1] == 10);
   CHECK(geometry.global_size[2] == 28);
   CHECK(geometry.local_size[0] == 32);
   CHECK(geometry.local_size[1] == 2);
   CHECK(geometry.local_size[2] == 4);

   /* The m1n1 witness launch: 2 groups of 32 threads. */
   const uint32_t witness_groups[3] = {2, 1, 1};
   const uint32_t witness_local[3] = {32, 1, 1};
   CHECK(agx3_g17p_indirect_geometry_from_groups(witness_groups,
                                                 witness_local, &geometry));
   CHECK(geometry.global_size[0] == 64 && geometry.global_size[1] == 1 &&
         geometry.global_size[2] == 1);
   CHECK(geometry.local_size[0] == 32 && geometry.local_size[1] == 1 &&
         geometry.local_size[2] == 1);

   const struct agx3_g17p_indirect_geometry before = geometry;

   /* Zero on any axis is refused: the GPU cannot report it back and an
    * all-zero block is indistinguishable from one the helper never wrote. */
   for (unsigned axis = 0; axis < 3; ++axis) {
      uint32_t bad[3] = {1, 1, 1};
      bad[axis] = 0;
      CHECK(!agx3_g17p_indirect_geometry_from_groups(bad, witness_local,
                                                     &geometry));
      CHECK(memcmp(&geometry, &before, sizeof(geometry)) == 0);
      CHECK(!agx3_g17p_indirect_geometry_from_groups(witness_groups, bad,
                                                     &geometry));
      CHECK(memcmp(&geometry, &before, sizeof(geometry)) == 0);
   }

   /* The helper multiplies in wrapping 32-bit.  0x10000000 groups of 16 is
    * exactly 2^32, which the GPU would publish as a global size of zero. */
   const uint32_t overflow_groups[3] = {0x10000000, 1, 1};
   const uint32_t overflow_local[3] = {16, 1, 1};
   CHECK(!agx3_g17p_indirect_geometry_from_groups(
      overflow_groups, overflow_local, &geometry));
   CHECK(memcmp(&geometry, &before, sizeof(geometry)) == 0);

   /* One below the wrap is still legal. */
   const uint32_t edge_groups[3] = {0xffffffff, 1, 1};
   const uint32_t edge_local[3] = {1, 1, 1};
   CHECK(agx3_g17p_indirect_geometry_from_groups(edge_groups, edge_local,
                                                 &geometry));
   CHECK(geometry.global_size[0] == 0xffffffff);

   CHECK(!agx3_g17p_indirect_geometry_from_groups(NULL, edge_local,
                                                  &geometry));
   CHECK(!agx3_g17p_indirect_geometry_from_groups(edge_groups, NULL,
                                                  &geometry));
   CHECK(!agx3_g17p_indirect_geometry_from_groups(edge_groups, edge_local,
                                                  NULL));
   return true;
}

static bool
test_geometry_pack(void)
{
   CHECK(AGX3_G17P_INDIRECT_GEOMETRY_WORDS == 6);
   CHECK(AGX3_G17P_INDIRECT_GEOMETRY_SIZE == 24);
   CHECK(AGX3_G17P_INDIRECT_GEOMETRY_ALIGNMENT == 4);

   const struct agx3_g17p_indirect_geometry geometry = {
      .global_size = {64, 3, 2},
      .local_size = {32, 1, 2},
   };

   /* Six little-endian u32, global first.  This is positionally identical to
    * the direct launch record's inline dimension fields. */
   static const uint8_t expected[AGX3_G17P_INDIRECT_GEOMETRY_SIZE] = {
      0x40, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
      0x02, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00,
      0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
   };

   uint8_t packed[AGX3_G17P_INDIRECT_GEOMETRY_SIZE + 2];
   memset(packed, 0xa5, sizeof(packed));
   CHECK(agx3_g17p_indirect_geometry_pack(&geometry, packed + 1));
   CHECK(packed[0] == 0xa5 &&
         packed[AGX3_G17P_INDIRECT_GEOMETRY_SIZE + 1] == 0xa5);
   CHECK(memcmp(packed + 1, expected, sizeof(expected)) == 0);

   /* The named half offsets must address the halves the packer wrote, not
    * merely divide the block in two: a driver kernel writes through them. */
   CHECK(AGX3_G17P_INDIRECT_GEOMETRY_GLOBAL_OFFSET == 0);
   CHECK(AGX3_G17P_INDIRECT_GEOMETRY_LOCAL_OFFSET +
            3 * sizeof(uint32_t) == AGX3_G17P_INDIRECT_GEOMETRY_SIZE);
   for (unsigned axis = 0; axis < 3; ++axis) {
      uint32_t word;
      memcpy(&word,
             packed + 1 + AGX3_G17P_INDIRECT_GEOMETRY_GLOBAL_OFFSET +
                axis * sizeof(word),
             sizeof(word));
      CHECK(word == geometry.global_size[axis]);
      memcpy(&word,
             packed + 1 + AGX3_G17P_INDIRECT_GEOMETRY_LOCAL_OFFSET +
                axis * sizeof(word),
             sizeof(word));
      CHECK(word == geometry.local_size[axis]);
   }

   struct agx3_g17p_indirect_geometry parsed;
   CHECK(agx3_g17p_indirect_geometry_unpack(expected, &parsed));
   CHECK(memcmp(&parsed, &geometry, sizeof(parsed)) == 0);

   /* An unwritten (all-zero) block must not parse as a launch geometry. */
   static const uint8_t zero[AGX3_G17P_INDIRECT_GEOMETRY_SIZE] = {0};
   const struct agx3_g17p_indirect_geometry before = parsed;
   CHECK(!agx3_g17p_indirect_geometry_unpack(zero, &parsed));
   CHECK(memcmp(&parsed, &before, sizeof(parsed)) == 0);

   uint8_t out[AGX3_G17P_INDIRECT_GEOMETRY_SIZE];
   memset(out, 0x4d, sizeof(out));
   struct agx3_g17p_indirect_geometry bad = geometry;
   bad.local_size[1] = 0;
   CHECK(!agx3_g17p_indirect_geometry_pack(&bad, out));
   for (unsigned i = 0; i < sizeof(out); ++i)
      CHECK(out[i] == 0x4d);

   CHECK(!agx3_g17p_indirect_geometry_pack(NULL, out));
   CHECK(!agx3_g17p_indirect_geometry_pack(&geometry, NULL));
   CHECK(!agx3_g17p_indirect_geometry_unpack(NULL, &parsed));
   CHECK(!agx3_g17p_indirect_geometry_unpack(expected, NULL));
   return true;
}

int
main(void)
{
   const struct {
      const char *name;
      bool (*run)(void);
   } cases[] = {
      {"shader image", test_shader_image},
      {"ldshdr kinds", test_ldshdr_kinds},
      {"geometry from groups", test_geometry_from_groups},
      {"geometry pack", test_geometry_pack},
   };

   for (unsigned i = 0; i < ARRAY_SIZE(cases); ++i) {
      if (!cases[i].run()) {
         fprintf(stderr, "case %s failed\n", cases[i].name);
         return 1;
      }
   }

   printf("agx3 G17P grid helper: %u checks passed\n", checks);
   return 0;
}
