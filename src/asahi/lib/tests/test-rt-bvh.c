/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "../agx_rt_bvh.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned checks;

static void
require(bool condition, const char *message)
{
   ++checks;
   if (!condition) {
      fprintf(stderr, "FAIL: %s\n", message);
      exit(1);
   }
}

static void
set_field(uint64_t *word, unsigned start, unsigned width, uint32_t value)
{
   const uint64_t field_mask = (UINT64_C(1) << width) - 1;
   const uint64_t mask = field_mask << start;

   *word = (*word & ~mask) | (((uint64_t)value & field_mask) << start);
}

static void
set_gen1_control(struct agx_rt_bvh_gen1_header *header, uint32_t control)
{
   set_field(&header->words[7], 0, 6, control);
}

static void
set_gen1_leaf_lane_end(struct agx_rt_bvh_gen1_header *header, unsigned lane,
                       uint32_t low, uint32_t high_field)
{
   set_field(&header->words[lane], 16, 4, low);
   set_field(&header->words[lane], 20, 2, high_field);
}

static void
set_gen1_child_space_slot(struct agx_rt_bvh_gen1_header *header,
                          uint32_t slot)
{
   set_field(&header->words[5], 0, 16, slot >> 16);
   set_field(&header->words[6], 0, 16, slot & 0xffffu);
}

static uint32_t
reference_gen1_inner_child_count(uint32_t control)
{
   uint32_t count = control & 7u;

   if ((control & 0x3fu) == 0x3fu)
      count = 0;
   if (((control | (control >> 3)) & 7u) == 0)
      count = 8;

   return count;
}

static uint32_t
reference_gen1_leaf_box_count(uint32_t control)
{
   const uint32_t fields_intersection = (control & (control >> 3)) & 7u;
   return fields_intersection == 7u ? 8u : ((control >> 3) & 7u);
}

static uint32_t
reference_gen1_leaf_count(const struct agx_rt_bvh_gen1_header *header)
{
   const uint32_t control = (uint32_t)header->words[7] & 0x3fu;
   const uint32_t leaf_boxes = reference_gen1_leaf_box_count(control);

   if ((((uint32_t)header->words[0]) & 1u) != 0)
      return leaf_boxes;

   const uint32_t first = reference_gen1_inner_child_count(control);
   uint32_t maximum = 0;

   for (uint32_t i = 0; i < leaf_boxes && first + i < 8; ++i) {
      const uint32_t lane = (uint32_t)header->words[first + i];
      const uint32_t low = (lane >> 16) & 15u;
      const uint32_t high_field = (lane >> 20) & 3u;
      const uint32_t end = low + (high_field ? high_field : 4u);
      if (end > maximum)
         maximum = end;
   }

   return maximum;
}

static uint64_t
next_random(uint64_t *state)
{
   uint64_t value = *state;
   value ^= value << 13;
   value ^= value >> 7;
   value ^= value << 17;
   *state = value;
   return value;
}

static void
test_gen1_unpack_and_control(void)
{
   uint8_t storage[AGX_RT_BVH_GEN1_HEADER_SIZE + 1] = {0};
   uint8_t *packed = storage + 1;
   struct agx_rt_bvh_gen1_header header = {0};

   for (unsigned word = 0; word < 8; ++word) {
      const uint64_t value = UINT64_C(0x1020304050607080) + word;
      for (unsigned byte = 0; byte < 8; ++byte)
         packed[(word * 8) + byte] = (uint8_t)(value >> (byte * 8));
   }

   require(agx_rt_bvh_gen1_header_unpack(packed, &header),
           "an unaligned little-endian Gen1 header unpacks");
   for (unsigned word = 0; word < 8; ++word) {
      require(header.words[word] == UINT64_C(0x1020304050607080) + word,
              "each Gen1 header word preserves little-endian byte order");
   }
   require(!agx_rt_bvh_gen1_header_unpack(NULL, &header),
           "a null Gen1 packed node is refused");
   require(!agx_rt_bvh_gen1_header_unpack(packed, NULL),
           "a null Gen1 unpack destination is refused");

   for (uint32_t control = 0; control < 64; ++control) {
      set_gen1_control(&header, control);
      const uint32_t inner = reference_gen1_inner_child_count(control);
      const uint32_t leaves = reference_gen1_leaf_box_count(control);

      require(agx_rt_bvh_gen1_control(&header) == control,
              "all six Gen1 control bits round-trip");
      require(agx_rt_bvh_gen1_inner_child_count(&header) == inner,
              "the Gen1 inner-child endpoint encoding matches Apple");
      require(agx_rt_bvh_gen1_leaf_box_count(&header) == leaves,
              "the Gen1 leaf-box endpoint encoding matches Apple");
      require(agx_rt_bvh_gen1_branching_layout_valid(&header) ==
                 (inner + leaves <= 8),
              "the Gen1 BVH8 branching-factor gate matches Apple");
   }

   set_gen1_control(&header, 0);
   require(agx_rt_bvh_gen1_inner_child_count(&header) == 8 &&
              agx_rt_bvh_gen1_leaf_box_count(&header) == 0,
           "control zero is the all-inner endpoint");
   set_gen1_control(&header, 0x3f);
   require(agx_rt_bvh_gen1_inner_child_count(&header) == 0 &&
              agx_rt_bvh_gen1_leaf_box_count(&header) == 8,
           "control 0x3f is the all-leaf endpoint");
}

static void
test_gen1_leaf_layout(void)
{
   struct agx_rt_bvh_gen1_header header = {0};

   for (unsigned lane = 0; lane < 8; ++lane) {
      for (uint32_t low = 0; low < 16; ++low) {
         for (uint32_t high = 0; high < 4; ++high) {
            set_gen1_leaf_lane_end(&header, lane, low, high);
            require(agx_rt_bvh_gen1_leaf_lane_end(&header, lane) ==
                       low + (high ? high : 4u),
                    "every Gen1 4+2-bit leaf-lane end decodes");
         }
      }
   }
   require(agx_rt_bvh_gen1_leaf_lane_end(&header, 8) == 0,
           "an out-of-range Gen1 leaf lane fails closed");

   header = (struct agx_rt_bvh_gen1_header){0};
   set_gen1_control(&header, (3u << 3) | 2u);
   set_gen1_leaf_lane_end(&header, 2, 5, 0);  /* 9 */
   set_gen1_leaf_lane_end(&header, 3, 14, 3); /* 17 */
   set_gen1_leaf_lane_end(&header, 4, 2, 2);  /* 4 */
   set_gen1_child_space_slot(&header, UINT32_C(0xabcd1234));
   set_field(&header.words[0], 1, 4, 6);

   require(agx_rt_bvh_gen1_leaf_count(&header) == 17,
           "count mode takes the maximum leaf-lane end");
   require(agx_rt_bvh_gen1_leaf_slot_capacity(&header) == 6,
           "the Gen1 leaf-slot divisor decodes from w0[1:4]");
   require(agx_rt_bvh_gen1_child_space_slot(&header) == UINT32_C(0xabcd1234),
           "the Gen1 child-space slot uses w5-low:w6-low");
   require(agx_rt_bvh_gen1_leaf_storage_slots(&header) == 3,
           "the Gen1 leaf extent rounds up to three child-space slots");
   require(agx_rt_bvh_gen1_inner_child_offset(&header) ==
              UINT32_C(0xabcd1237),
           "the Gen1 inner-child offset follows rounded leaf storage");

   set_field(&header.words[0], 0, 1, 1);
   require(agx_rt_bvh_gen1_uses_indexed_leaf_counts(&header),
           "w0 bit zero selects indexed Gen1 leaf counts");
   require(agx_rt_bvh_gen1_leaf_count(&header) == 3,
           "indexed mode uses the leaf-box count");
   require(agx_rt_bvh_gen1_leaf_storage_slots(&header) == 1 &&
              agx_rt_bvh_gen1_inner_child_offset(&header) ==
                 UINT32_C(0xabcd1235),
           "indexed leaf storage advances the inner-child offset once");

   set_field(&header.words[0], 1, 4, 0);
   require(agx_rt_bvh_gen1_leaf_storage_slots(&header) == 0 &&
              agx_rt_bvh_gen1_inner_child_offset(&header) ==
                 UINT32_C(0xabcd1234),
           "a zero Gen1 divisor leaves the child-space base unchanged");

   set_gen1_control(&header, (6u << 3) | 7u);
   require(!agx_rt_bvh_gen1_branching_layout_valid(&header),
           "an overfull ordinary Gen1 branch layout is rejected");
}

static void
test_gen1_quantized_bounds(void)
{
   struct agx_rt_bvh_gen1_header header = {0};

   for (unsigned child = 0; child < 8; ++child) {
      uint32_t expected[6];
      for (unsigned component = 0; component < 6; ++component) {
         expected[component] = (child * 17u + component * 11u) & 0x7fu;
         const unsigned shift = 57u - component * 7u;
         set_field(&header.words[child], shift, 7, expected[component]);
      }

      struct agx_rt_bvh_gen1_quantized_bounds bounds;
      require(agx_rt_bvh_gen1_quantized_child_bounds(&header, child, &bounds),
              "each Gen1 child exposes quantized bounds");
      require(memcmp(bounds.component, expected, sizeof(expected)) == 0,
              "Gen1 quantized bounds follow Apple's descending bit order");
   }

   struct agx_rt_bvh_gen1_quantized_bounds bounds;
   require(!agx_rt_bvh_gen1_quantized_child_bounds(NULL, 0, &bounds),
           "a missing Gen1 header has no quantized bounds");
   require(!agx_rt_bvh_gen1_quantized_child_bounds(&header, 8, &bounds),
           "a ninth Gen1 child has no quantized bounds");
   require(!agx_rt_bvh_gen1_quantized_child_bounds(&header, 0, NULL),
           "a missing Gen1 bounds destination is refused");
}

static uint32_t
float_bits(float value)
{
   uint32_t bits;
   memcpy(&bits, &value, sizeof(bits));
   return bits;
}

static void
store_le64(uint8_t *destination, uint64_t value)
{
   for (unsigned byte = 0; byte < 8; ++byte)
      destination[byte] = (uint8_t)(value >> (byte * 8));
}

static void
test_gen1_hw_triangle_leaf(void)
{
   /* Slot 14 from the pinned public-Metal M5 Pro version-3 capture.  The
    * end-to-end walker identifies this as geometry 0, primitive 7 and matches
    * all four SoA lanes against the probe's source vertices. */
   static const uint8_t captured[AGX_RT_BVH_GEN1_HW_TRIANGLE_LEAF_SIZE] = {
      0x50, 0x00, 0x00, 0x00, 0x9c, 0x54, 0xce, 0xbe,
      0x08, 0xec, 0xd2, 0xbd, 0x9c, 0x54, 0xce, 0xbe,
      0x9c, 0x54, 0xce, 0xbe, 0x05, 0xc0, 0x7d, 0xbf,
      0x05, 0xc0, 0x7d, 0xbf, 0x38, 0xf3, 0x30, 0xbf,
      0x38, 0xf3, 0x30, 0xbf, 0x33, 0x33, 0xb3, 0x3e,
      0x33, 0x33, 0xb3, 0x3e, 0x66, 0x66, 0xe6, 0x3e,
      0x66, 0x66, 0xe6, 0x3e, 0x00, 0x00, 0x00, 0x00,
      0x07, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00,
   };
   static const uint32_t expected_coordinate_bits[3][4] = {
      {0xbece549c, 0xbdd2ec08, 0xbece549c, 0xbece549c},
      {0xbf7dc005, 0xbf7dc005, 0xbf30f338, 0xbf30f338},
      {0x3eb33333, 0x3eb33333, 0x3ee66666, 0x3ee66666},
   };

   uint8_t unaligned[AGX_RT_BVH_GEN1_HW_TRIANGLE_LEAF_SIZE + 1];
   memcpy(unaligned + 1, captured, sizeof(captured));

   struct agx_rt_bvh_gen1_hw_triangle_leaf leaf;
   require(agx_rt_bvh_gen1_hw_triangle_leaf_unpack(
              unaligned + 1, sizeof(captured), &leaf),
           "an unaligned real Gen1 hardware triangle leaf decodes");
   for (unsigned axis = 0; axis < 3; ++axis) {
      for (unsigned lane = 0; lane < 4; ++lane) {
         require(float_bits(leaf.coordinate[axis][lane]) ==
                    expected_coordinate_bits[axis][lane],
                 "every captured triangle coordinate bit round-trips");
      }
   }
   require(leaf.geometry_word == 0 && leaf.primitive_word[0] == 7 &&
              leaf.primitive_word[1] == 7,
           "the captured triangle leaf preserves all three raw ID words");
   require(agx_rt_bvh_gen1_hw_triangle_geometry_id(&leaf) == 0 &&
              agx_rt_bvh_gen1_hw_triangle_primitive_id(&leaf, 0) == 7 &&
              agx_rt_bvh_gen1_hw_triangle_primitive_id(&leaf, 1) == 7,
           "the captured triangle leaf exposes its exact low-30-bit IDs");

   uint8_t flagged[sizeof(captured)];
   memcpy(flagged, captured, sizeof(flagged));
   flagged[55] = 0xc0;
   flagged[59] = 0x80;
   flagged[63] = 0x40;
   require(agx_rt_bvh_gen1_hw_triangle_leaf_unpack(flagged, sizeof(flagged),
                                                   &leaf),
           "unknown high ID bits do not invalidate a triangle leaf");
   require(leaf.geometry_word == UINT32_C(0xc0000000) &&
              leaf.primitive_word[0] == UINT32_C(0x80000007) &&
              leaf.primitive_word[1] == UINT32_C(0x40000007),
           "unknown high ID bits remain available to future decoders");
   require(agx_rt_bvh_gen1_hw_triangle_geometry_id(&leaf) == 0 &&
              agx_rt_bvh_gen1_hw_triangle_primitive_id(&leaf, 0) == 7 &&
              agx_rt_bvh_gen1_hw_triangle_primitive_id(&leaf, 1) == 7,
           "unknown high ID bits cannot contaminate decoded IDs");

   const struct agx_rt_bvh_gen1_hw_triangle_leaf zero = {0};
   for (size_t available = 0;
        available < AGX_RT_BVH_GEN1_HW_TRIANGLE_LEAF_SIZE; ++available) {
      memset(&leaf, 0xa5, sizeof(leaf));
      require(!agx_rt_bvh_gen1_hw_triangle_leaf_unpack(captured, available,
                                                       &leaf),
              "every truncated hardware triangle leaf is rejected");
      require(memcmp(&leaf, &zero, sizeof(leaf)) == 0,
              "a truncated triangle leaf leaves no stale output");
   }

   for (unsigned bit = 0; bit < 32; ++bit) {
      uint8_t malformed[sizeof(captured)];
      memcpy(malformed, captured, sizeof(malformed));
      malformed[bit / 8] ^= 1u << (bit % 8);
      memset(&leaf, 0xa5, sizeof(leaf));
      require(!agx_rt_bvh_gen1_hw_triangle_leaf_unpack(
                 malformed, sizeof(malformed), &leaf),
              "every changed triangle-leaf marker bit is rejected");
      require(memcmp(&leaf, &zero, sizeof(leaf)) == 0,
              "a wrong triangle marker leaves no stale output");
   }

   memset(&leaf, 0xa5, sizeof(leaf));
   require(!agx_rt_bvh_gen1_hw_triangle_leaf_unpack(NULL, sizeof(captured),
                                                    &leaf),
           "a missing triangle leaf is rejected");
   require(memcmp(&leaf, &zero, sizeof(leaf)) == 0,
           "a missing triangle leaf leaves no stale output");
   require(!agx_rt_bvh_gen1_hw_triangle_leaf_unpack(captured,
                                                    sizeof(captured), NULL),
           "a missing triangle-leaf destination is rejected");
   require(agx_rt_bvh_gen1_hw_triangle_geometry_id(NULL) == 0 &&
              agx_rt_bvh_gen1_hw_triangle_primitive_id(NULL, 0) == 0 &&
              agx_rt_bvh_gen1_hw_triangle_primitive_id(&leaf, 2) == 0,
           "missing and out-of-range triangle ID queries fail closed");
}

static void
test_gen1_hw_aabb_leaf(void)
{
   /* Exact records from the pinned public-Metal M5 Pro captures.  The
    * one-AABB record equals the second record in the four-AABB capture, but
    * is retained separately to guard both source vectors. */
   static const uint8_t single[AGX_RT_BVH_GEN1_HW_AABB_LEAF_SIZE] = {
      0x18, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x33, 0x33, 0xb3, 0x3f, 0xcd, 0xcc, 0xcc, 0x3e,
      0xcd, 0xcc, 0xcc, 0x3e, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   static const uint8_t captured[4][AGX_RT_BVH_GEN1_HW_AABB_LEAF_SIZE] = {
      {
         0x18, 0x00, 0x00, 0x00, 0x3a, 0xbb, 0x52, 0x3f,
         0x86, 0xd1, 0x15, 0x3f, 0xcd, 0xcc, 0x4c, 0x3d,
         0xd0, 0x90, 0x9c, 0x3f, 0xec, 0x37, 0x7c, 0x3f,
         0x67, 0x66, 0xe6, 0x3e, 0x00, 0x00, 0x00, 0x40,
         0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
         0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      },
      {
         0x18, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f,
         0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
         0x33, 0x33, 0xb3, 0x3f, 0xcd, 0xcc, 0xcc, 0x3e,
         0xcd, 0xcc, 0xcc, 0x3e, 0x00, 0x00, 0x00, 0x00,
         0x00, 0x00, 0x00, 0x00, 0x9c, 0x33, 0x9c, 0x33,
         0x9c, 0x33, 0x9c, 0x33, 0xe7, 0x3d, 0x3e, 0x3c,
      },
      {
         0x18, 0x00, 0x00, 0x00, 0xfa, 0x8f, 0xab, 0x3e,
         0xb3, 0xa0, 0x76, 0x3f, 0xcd, 0xcc, 0xcc, 0x3d,
         0x64, 0x2e, 0x3c, 0x3f, 0x8d, 0x83, 0xae, 0x3f,
         0x00, 0x00, 0x00, 0x3f, 0x00, 0x00, 0x00, 0x00,
         0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
         0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      },
      {
         0x18, 0x00, 0x00, 0x00, 0xdc, 0x69, 0x93, 0xbe,
         0x38, 0x2b, 0x7d, 0x3f, 0x9a, 0x99, 0x19, 0x3e,
         0xc4, 0x8b, 0xe5, 0x3d, 0xcf, 0xc8, 0xb1, 0x3f,
         0xcd, 0xcc, 0x0c, 0x3f, 0x00, 0x00, 0x00, 0x00,
         0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
         0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      },
   };
   static const uint32_t expected_minimum_bits[4][3] = {
      {0x3f52bb3a, 0x3f15d186, 0x3d4ccccd},
      {0x3f800000, 0x00000000, 0x00000000},
      {0x3eab8ffa, 0x3f76a0b3, 0x3dcccccd},
      {0xbe9369dc, 0x3f7d2b38, 0x3e19999a},
   };
   static const uint32_t expected_maximum_bits[4][3] = {
      {0x3f9c90d0, 0x3f7c37ec, 0x3ee66667},
      {0x3fb33333, 0x3ecccccd, 0x3ecccccd},
      {0x3f3c2e64, 0x3fae838d, 0x3f000000},
      {0x3de58bc4, 0x3fb1c8cf, 0x3f0ccccd},
   };
   static const uint32_t expected_primitive_id[4] = {1, 0, 2, 3};
   static const uint32_t expected_word36[4] = {
      0, UINT32_C(0x339c339c), 1, 0,
   };
   static const uint64_t expected_tail_word[4] = {
      0, UINT64_C(0x3c3e3de7339c339c), 0, 0,
   };

   uint8_t unaligned[AGX_RT_BVH_GEN1_HW_AABB_LEAF_SIZE + 2] = {0};
   memcpy(unaligned + 1, single, sizeof(single));
   unaligned[sizeof(unaligned) - 1] = 0xa5;

   struct agx_rt_bvh_gen1_hw_aabb_leaf leaf;
   require(agx_rt_bvh_gen1_hw_aabb_leaf_unpack(
              unaligned + 1, sizeof(unaligned) - 1, &leaf),
           "an unaligned AABB leaf accepts a larger remaining span");
   require(leaf.header == AGX_RT_BVH_GEN1_HW_AABB_LEAF_MARKER &&
              agx_rt_bvh_gen1_hw_aabb_geometry_id(&leaf) == 0 &&
              agx_rt_bvh_gen1_hw_aabb_primitive_id(&leaf) == 0,
           "the one-AABB capture decodes its header and IDs");

   for (unsigned record = 0; record < 4; ++record) {
      require(agx_rt_bvh_gen1_hw_aabb_leaf_unpack(
                 captured[record], sizeof(captured[record]), &leaf),
              "every real four-AABB capture record decodes");
      for (unsigned axis = 0; axis < 3; ++axis) {
         require(float_bits(leaf.minimum[axis]) ==
                    expected_minimum_bits[record][axis] &&
                    float_bits(leaf.maximum[axis]) ==
                    expected_maximum_bits[record][axis],
                 "every captured AABB bound preserves its exact f32 bits");
      }
      require(agx_rt_bvh_gen1_hw_aabb_geometry_id(&leaf) == 0 &&
                 agx_rt_bvh_gen1_hw_aabb_primitive_id(&leaf) ==
                    expected_primitive_id[record],
              "captured low/high 30-bit fields are geometry/primitive IDs");
      require(!agx_rt_bvh_gen1_hw_aabb_is_opaque(&leaf) &&
                 !agx_rt_bvh_gen1_hw_aabb_has_primitive_data(&leaf) &&
                 leaf.primitive_data_relative == 0,
              "default captured AABBs have neither optional proved flag");
      require(leaf.word36 == expected_word36[record] &&
                 leaf.primitive_data_relative_word ==
                    expected_tail_word[record],
              "every still-unassigned captured AABB tail byte is preserved");
   }
   require(leaf.packed_ids_and_flags == UINT64_C(0x0000000300000000),
           "the complete packed ID/flag payload remains visible");

   uint8_t opaque[sizeof(captured[0])];
   memcpy(opaque, captured[0], sizeof(opaque));
   opaque[31] = 0xc0;
   require(agx_rt_bvh_gen1_hw_aabb_leaf_unpack(opaque, sizeof(opaque), &leaf),
           "the controlled public-Metal opaque AABB record decodes");
   require(leaf.packed_ids_and_flags == UINT64_C(0x00000001c0000000) &&
              agx_rt_bvh_gen1_hw_aabb_is_opaque(&leaf),
           "the controlled opaque capture proves packed low-word bit 31");

   /* Controlled two-geometry capture: geometry one used table offset nine. */
   uint8_t geometry_one[sizeof(captured[2])];
   memcpy(geometry_one, captured[2], sizeof(geometry_one));
   geometry_one[1] = 9;
   geometry_one[28] = 1;
   geometry_one[32] = 0;
   require(agx_rt_bvh_gen1_hw_aabb_leaf_unpack(
              geometry_one, sizeof(geometry_one), &leaf),
           "the controlled second-geometry record decodes");
   require(leaf.header == UINT32_C(0x00000918) &&
              agx_rt_bvh_gen1_hw_aabb_intersection_function_table_offset(
                 &leaf) == 9 &&
              agx_rt_bvh_gen1_hw_aabb_geometry_id(&leaf) == 1 &&
              agx_rt_bvh_gen1_hw_aabb_primitive_id(&leaf) == 0,
           "controlled bytes prove table offset and both ID field names");

   /* Exercise all combinations of the four packed bits outside the two
    * proved 30-bit IDs.  Bit 31 is opaque and bit 62 gates the relative
    * primitive-data address; bits 30 and 63 remain raw and unnamed. */
   for (unsigned flags = 0; flags < 16; ++flags) {
      uint8_t flagged[sizeof(single)];
      memcpy(flagged, single, sizeof(flagged));
      const uint64_t raw = UINT64_C(0x0234567801234567) |
                           ((uint64_t)(flags & 1u) << 30) |
                           ((uint64_t)(flags & 2u) << 30) |
                           ((uint64_t)(flags & 4u) << 60) |
                           ((uint64_t)(flags & 8u) << 60);
      const uint64_t relative_bits = UINT64_C(0xfedcba9876543210);
      store_le64(flagged + 28, raw);
      store_le64(flagged + 40, relative_bits);
      require(agx_rt_bvh_gen1_hw_aabb_leaf_unpack(flagged, sizeof(flagged),
                                                   &leaf),
              "every combination of raw packed AABB flags decodes");
      require(leaf.packed_ids_and_flags == raw &&
                 leaf.word36 == 0 &&
                 leaf.primitive_data_relative_word == relative_bits &&
                 agx_rt_bvh_gen1_hw_aabb_geometry_id(&leaf) == 0x01234567 &&
                 agx_rt_bvh_gen1_hw_aabb_primitive_id(&leaf) == 0x02345678,
              "raw flags cannot contaminate either 30-bit AABB ID");
      require(agx_rt_bvh_gen1_hw_aabb_is_opaque(&leaf) ==
                 ((flags & 2u) != 0) &&
                 agx_rt_bvh_gen1_hw_aabb_has_primitive_data(&leaf) ==
                    ((flags & 4u) != 0),
              "only the two proved AABB flags receive semantic names");
      int64_t expected_relative = 0;
      if (flags & 4u)
         memcpy(&expected_relative, &relative_bits, sizeof(relative_bits));
      require(leaf.primitive_data_relative == expected_relative,
              "the relative primitive-data value is consumed only when gated");
   }

   const struct agx_rt_bvh_gen1_hw_aabb_leaf zero = {0};
   for (size_t available = 0;
        available < AGX_RT_BVH_GEN1_HW_AABB_LEAF_SIZE; ++available) {
      memset(&leaf, 0xa5, sizeof(leaf));
      require(!agx_rt_bvh_gen1_hw_aabb_leaf_unpack(single, available, &leaf),
              "every truncated hardware AABB leaf is rejected");
      require(memcmp(&leaf, &zero, sizeof(leaf)) == 0,
              "a truncated AABB leaf leaves no stale output");
   }

   for (unsigned bit = 0; bit < 32; ++bit) {
      uint8_t changed_header[sizeof(single)];
      memcpy(changed_header, single, sizeof(changed_header));
      changed_header[bit / 8] ^= 1u << (bit % 8);
      memset(&leaf, 0xa5, sizeof(leaf));
      const bool valid = agx_rt_bvh_gen1_hw_aabb_leaf_unpack(
         changed_header, sizeof(changed_header), &leaf);
      if (bit < 8) {
         require(!valid, "every changed low-byte AABB marker bit is rejected");
         require(memcmp(&leaf, &zero, sizeof(leaf)) == 0,
                 "a wrong AABB marker leaves no stale output");
      } else {
         require(valid, "every upper AABB header bit remains payload");
         require(leaf.header == (UINT32_C(0x18) | (UINT32_C(1) << bit)),
                 "upper AABB header payload bits are preserved exactly");
      }
   }

   memset(&leaf, 0xa5, sizeof(leaf));
   require(!agx_rt_bvh_gen1_hw_aabb_leaf_unpack(NULL, sizeof(single), &leaf),
           "a missing AABB leaf is rejected");
   require(memcmp(&leaf, &zero, sizeof(leaf)) == 0,
           "a missing AABB leaf leaves no stale output");
   require(!agx_rt_bvh_gen1_hw_aabb_leaf_unpack(single, sizeof(single), NULL),
           "a missing AABB-leaf destination is rejected");
   require(agx_rt_bvh_gen1_hw_aabb_intersection_function_table_offset(NULL) ==
              0 &&
              agx_rt_bvh_gen1_hw_aabb_geometry_id(NULL) == 0 &&
              agx_rt_bvh_gen1_hw_aabb_primitive_id(NULL) == 0 &&
              !agx_rt_bvh_gen1_hw_aabb_is_opaque(NULL) &&
              !agx_rt_bvh_gen1_hw_aabb_has_primitive_data(NULL),
           "all missing AABB field queries fail closed");
}

static void
test_gen1_randomized_assembly_equivalence(void)
{
   uint64_t random = UINT64_C(0xf24c8e91d37ab605);

   for (unsigned iteration = 0; iteration < 1000000; ++iteration) {
      struct agx_rt_bvh_gen1_header header;
      for (unsigned word = 0; word < 8; ++word)
         header.words[word] = next_random(&random);

      const uint32_t control = (uint32_t)header.words[7] & 0x3fu;
      const uint32_t expected_inner =
         reference_gen1_inner_child_count(control);
      const uint32_t expected_boxes = reference_gen1_leaf_box_count(control);
      const uint32_t expected_leaves = reference_gen1_leaf_count(&header);
      const uint32_t capacity = ((uint32_t)header.words[0] >> 1) & 15u;
      const uint32_t expected_storage =
         capacity ? ((expected_leaves + capacity - 1u) / capacity) &
                       0x03ffffffu
                  : 0;

      if (agx_rt_bvh_gen1_inner_child_count(&header) != expected_inner ||
          agx_rt_bvh_gen1_leaf_box_count(&header) != expected_boxes ||
          agx_rt_bvh_gen1_leaf_count(&header) != expected_leaves ||
          agx_rt_bvh_gen1_leaf_storage_slots(&header) != expected_storage ||
          agx_rt_bvh_gen1_inner_child_offset(&header) !=
             agx_rt_bvh_gen1_child_space_slot(&header) + expected_storage) {
         require(false,
                 "random Gen1 header diverged from the Apple assembly model");
      }
   }

   require(true,
           "one million Gen1 headers match the independent assembly model");
}

static void
set_slot(struct agx_rt_bvh_gen5_header *header, unsigned slot,
         uint32_t value)
{
   set_field(&header->words[slot >> 1], 16 + (3 * (slot & 1)), 3, value);
}

static void
set_child_space_slot(struct agx_rt_bvh_gen5_header *header, int32_t slot)
{
   const uint32_t encoded = (uint32_t)slot;

   set_field(&header->words[5], 0, 16, encoded >> 16);
   set_field(&header->words[6], 0, 16, encoded & 0xffffu);
}

static void
set_qtb_occupancy(struct agx_rt_bvh_gen5_header *header, unsigned qtb,
                  uint32_t occupancy)
{
   set_field(&header->words[6], 16 + (2 * qtb), 2, occupancy & 3u);
   set_field(&header->words[7], 16 + (2 * qtb), 2, occupancy >> 2);
}

static struct agx_rt_bvh_gen5_header
make_config4_layout(void)
{
   struct agx_rt_bvh_gen5_header header = {0};

   set_field(&header.words[0], 0, 3, 4); /* configuration */
   set_field(&header.words[0], 3, 4, 3); /* inner-child boundary */
   set_slot(&header, 0, 1);
   set_slot(&header, 1, 2);
   set_slot(&header, 2, 3);
   set_slot(&header, 3, 1);
   set_slot(&header, 4, 2);
   set_slot(&header, 5, 0);
   set_slot(&header, 6, 1);
   set_slot(&header, 7, 0);

   /* Four configuration-4 VRB spans: 7 is the special encoding for 8. */
   set_slot(&header, 8, 7);
   set_slot(&header, 9, 2);
   set_slot(&header, 10, 0);
   set_slot(&header, 11, 1);

   set_qtb_occupancy(&header, 0, 1);
   set_qtb_occupancy(&header, 1, 2);
   set_qtb_occupancy(&header, 2, 3);
   set_child_space_slot(&header, 100);
   return header;
}

static void
test_unpack_and_signed_child_space(void)
{
   uint8_t storage[AGX_RT_BVH_GEN5_HEADER_SIZE + 1] = {0};
   uint8_t *packed = storage + 1; /* exercise an unaligned mapped node */

   const uint64_t words[8] = {
      UINT64_C(0x0123456789abcdef), UINT64_C(0xfedcba9876543210),
      UINT64_C(0x8877665544332211), UINT64_C(0x1020304050607080),
      UINT64_C(0x1112131415161718), UINT64_C(0xabcdefff00008000),
      UINT64_C(0x1234567800000001), UINT64_C(0x5555aaaa0000ffff),
   };

   for (unsigned word = 0; word < 8; ++word) {
      for (unsigned byte = 0; byte < 8; ++byte)
         packed[(word * 8) + byte] = (uint8_t)(words[word] >> (byte * 8));
   }

   struct agx_rt_bvh_gen5_header header;
   require(agx_rt_bvh_gen5_header_unpack(packed, &header),
           "an unaligned little-endian Gen5 header unpacks");
   require(memcmp(header.words, words, sizeof(words)) == 0,
           "all eight words unpack without host-endian or alignment leakage");
   require(agx_rt_bvh_gen5_child_space_slot(&header) ==
              (int32_t)UINT32_C(0x80000001),
           "child space is the signed w5-low:w6-low value");
   require(!agx_rt_bvh_gen5_header_unpack(NULL, &header),
           "a null packed node is refused");
   require(!agx_rt_bvh_gen5_header_unpack(packed, NULL),
           "a null unpack destination is refused");

   struct agx_rt_bvh_gen5_header negative = {0};
   set_child_space_slot(&negative, -2);
   set_slot(&negative, 0, 1);
   require(agx_rt_bvh_gen5_child_space_slot(&negative) == -2,
           "negative child-space slots preserve their sign");
   require(agx_rt_bvh_gen5_offset_child_qtbs_bytes(&negative) == -64,
           "signed slot arithmetic remains signed through byte addressing");
}

static void
test_leaf_and_inner_counts(void)
{
   struct agx_rt_bvh_gen5_header header = {0};
   set_field(&header.words[0], 3, 4, 3);
   set_slot(&header, 0, 0);
   set_slot(&header, 1, 2);
   set_slot(&header, 2, 7);
   set_slot(&header, 3, 1);
   set_slot(&header, 4, 2);
   set_slot(&header, 5, 3);
   set_slot(&header, 6, 0);
   set_slot(&header, 7, 1);

   require(agx_rt_bvh_gen5_leaf_count(&header) == 17,
           "count mode treats zero as eight leaf records");
   require(agx_rt_bvh_gen5_inner_slots(&header) == 7,
           "inner slots are summed from the boundary through slot seven");

   set_field(&header.words[0], 7, 1, 1);
   require(agx_rt_bvh_gen5_uses_indexed_leaf_counts(&header),
           "w0 bit 7 selects indexed leaf counts");
   require(agx_rt_bvh_gen5_leaf_count(&header) == 8,
           "indexed mode decodes the maximum zero-based index plus one");

   set_field(&header.words[0], 3, 4, 0);
   require(agx_rt_bvh_gen5_leaf_count(&header) == 0,
           "a zero boundary owns no leaf records");

   set_field(&header.words[0], 3, 4, 8);
   require(agx_rt_bvh_gen5_inner_slots(&header) == 0,
           "a boundary beyond the eight inner fields has no inner slots");
   require(agx_rt_bvh_gen5_slot(&header, 16) == 0,
           "out-of-range slot reads fail closed");
}

static void
test_qtb_and_vrb_fields(void)
{
   struct agx_rt_bvh_gen5_header header = {0};
   set_field(&header.words[0], 0, 3, 4);
   set_field(&header.words[0], 3, 4, 3);
   set_slot(&header, 0, 5);
   set_slot(&header, 1, 6);
   set_slot(&header, 2, 4);
   set_qtb_occupancy(&header, 0, 5);
   set_qtb_occupancy(&header, 1, 6);
   set_qtb_occupancy(&header, 2, 4);

   uint32_t a, b, c;
   agx_rt_bvh_gen5_qtb_occupancies(&header, &a, &b, &c);
   require(a == 5 && b == 6 && c == 4,
           "QTB occupancies combine low w6 and high w7 halves");
   require(agx_rt_bvh_gen5_qtb_count(&header) == 3,
           "the highest occupied QTB determines a three-QTB section");
   require(agx_rt_bvh_gen5_qtb_layout_valid(&header),
           "matching QTB occupancies are accepted");

   set_qtb_occupancy(&header, 2, 3);
   require(agx_rt_bvh_gen5_qtb_count(&header) ==
              AGX_RT_BVH_GEN5_INVALID_QTB_COUNT,
           "an occupancy sum mismatch produces Apple's sentinel");
   require(!agx_rt_bvh_gen5_qtb_layout_valid(&header),
           "the QTB sentinel is classified as malformed");

   set_field(&header.words[0], 0, 3, 1);
   require(agx_rt_bvh_gen5_qtb_count(&header) == 0,
           "configurations outside 2..4 have no QTB section");

   set_slot(&header, 8, 7);
   set_slot(&header, 9, 2);
   set_slot(&header, 10, 0);
   set_slot(&header, 11, 1);
   require(agx_rt_bvh_gen5_vrb_size_bytes(&header) == 352,
           "VRB spans include the seven-means-eight encoding");
}

static void
test_offsets_for_every_configuration(void)
{
   static const int64_t expected_hw[8] = {
      6656, 6656, 6848, 6656, 6656, 6656, 6656, 6656,
   };
   static const int64_t expected_vrbs[8] = {
      6656, 6656, 6848, 6848, 6848, 6656, 6656, 6656,
   };
   static const int64_t expected_lhb[8] = {
      6656, 6880, 7072, 6848, 7200, 0, 7040, 6656,
   };
   static const int64_t expected_leaves[8] = {
      6680, 6904, 7096, 6872, 7224, 4, 7040, 6680,
   };

   for (uint32_t config = 0; config < 8; ++config) {
      struct agx_rt_bvh_gen5_header header = make_config4_layout();
      set_field(&header.words[0], 0, 3, config);

      require(agx_rt_bvh_gen5_leaf_count(&header) == 6,
              "the common vector owns six leaves");
      require(agx_rt_bvh_gen5_inner_slots(&header) == 4,
              "the common vector owns four inner slots");
      require(agx_rt_bvh_gen5_offset_child_qtbs_bytes(&header) == 6656,
              "child QTBs begin after the inner-child slots");
      require(agx_rt_bvh_gen5_offset_child_hw_leaves_bytes(&header) ==
                 expected_hw[config],
              "hardware-leaf offset matches the configuration table");
      require(agx_rt_bvh_gen5_offset_vrbs_bytes(&header) ==
                 expected_vrbs[config],
              "VRB offset matches the configuration table");
      require(agx_rt_bvh_gen5_offset_lhb_bytes(&header) ==
                 expected_lhb[config],
              "LHB offset matches the configuration table");
      require(agx_rt_bvh_gen5_offset_child_leaves_bytes(&header) ==
                 expected_leaves[config],
              "child-leaf offset matches the configuration table");
   }

   struct agx_rt_bvh_gen5_header special = make_config4_layout();
   set_field(&special.words[0], 0, 3, 7);
   set_field(&special.words[7], 8, 5, 0x1a);
   require(agx_rt_bvh_gen5_offset_child_leaves_bytes(&special) == 6656,
           "config 7 leaf type 0x1a has no LHB extent");

   set_field(&special.words[0], 0, 3, 0);
   require(agx_rt_bvh_gen5_offset_child_leaves_bytes(&special) == 6660,
           "a single-entry leaf type adds one four-byte LHB entry");
}

static bool
all_decoded_fields_equal(const struct agx_rt_bvh_gen5_header *a,
                         const struct agx_rt_bvh_gen5_header *b)
{
   uint32_t aa, ab, ac, ba, bb, bc;
   agx_rt_bvh_gen5_qtb_occupancies(a, &aa, &ab, &ac);
   agx_rt_bvh_gen5_qtb_occupancies(b, &ba, &bb, &bc);

   return agx_rt_bvh_gen5_config(a) == agx_rt_bvh_gen5_config(b) &&
          agx_rt_bvh_gen5_inner_child_boundary(a) ==
             agx_rt_bvh_gen5_inner_child_boundary(b) &&
          agx_rt_bvh_gen5_uses_indexed_leaf_counts(a) ==
             agx_rt_bvh_gen5_uses_indexed_leaf_counts(b) &&
          agx_rt_bvh_gen5_leaf_type(a) == agx_rt_bvh_gen5_leaf_type(b) &&
          agx_rt_bvh_gen5_child_space_slot(a) ==
             agx_rt_bvh_gen5_child_space_slot(b) &&
          agx_rt_bvh_gen5_leaf_count(a) == agx_rt_bvh_gen5_leaf_count(b) &&
          agx_rt_bvh_gen5_inner_slots(a) == agx_rt_bvh_gen5_inner_slots(b) &&
          aa == ba && ab == bb && ac == bc &&
          agx_rt_bvh_gen5_qtb_count(a) == agx_rt_bvh_gen5_qtb_count(b) &&
          agx_rt_bvh_gen5_vrb_size_bytes(a) ==
             agx_rt_bvh_gen5_vrb_size_bytes(b) &&
          agx_rt_bvh_gen5_offset_child_qtbs_bytes(a) ==
             agx_rt_bvh_gen5_offset_child_qtbs_bytes(b) &&
          agx_rt_bvh_gen5_offset_child_hw_leaves_bytes(a) ==
             agx_rt_bvh_gen5_offset_child_hw_leaves_bytes(b) &&
          agx_rt_bvh_gen5_offset_vrbs_bytes(a) ==
             agx_rt_bvh_gen5_offset_vrbs_bytes(b) &&
          agx_rt_bvh_gen5_offset_lhb_bytes(a) ==
             agx_rt_bvh_gen5_offset_lhb_bytes(b) &&
          agx_rt_bvh_gen5_offset_child_leaves_bytes(a) ==
             agx_rt_bvh_gen5_offset_child_leaves_bytes(b);
}

static void
test_unknown_bits_do_not_affect_addressing(void)
{
   const struct agx_rt_bvh_gen5_header reference = make_config4_layout();

   /* Exhaust all 2^11 combinations of the still-unidentified bits. */
   for (uint32_t pattern = 0; pattern < (1u << 11); ++pattern) {
      struct agx_rt_bvh_gen5_header candidate = reference;
      set_field(&candidate.words[0], 8, 8, pattern & 0xffu);
      set_field(&candidate.words[7], 13, 3, pattern >> 8);
      if (!all_decoded_fields_equal(&reference, &candidate)) {
         require(false,
                 "unknown Gen5 bits changed a proved child-layout accessor");
      }
   }

   require(true,
           "all 2^11 unknown-bit combinations leave child addressing unchanged");
}


static float
random_coordinate(uint64_t *state, float centre, float spread)
{
   /* Deliberately NOT a round number: the mantissa is filled from the PRNG so
    * the value almost never lands on a quantisation boundary. */
   const uint32_t bits = (uint32_t)(next_random(state) >> 32);
   const float unit = (float)(bits >> 8) / (float)(1u << 24);

   return centre + (unit - 0.5f) * spread;
}

static void
test_gen1_quantization_frame_roundtrip(void)
{
   struct agx_rt_bvh_gen1_header header;
   struct agx_rt_bvh_gen1_quantization_frame frame, decoded;

   memset(&header, 0, sizeof(header));

   /* Fill every bit the frame does not own, so a packer that clobbers a
    * neighbouring field is caught rather than passing silently. */
   for (unsigned i = 0; i < 8; ++i)
      header.words[i] = UINT64_C(0xa5a5a5a5a5a5a5a5);

   frame.origin[0] = -8.0f;
   frame.origin[1] = 0.5f;
   frame.origin[2] = 1024.0f;
   frame.exponent[0] = 127;
   frame.exponent[1] = 120;
   frame.exponent[2] = 135;

   require(agx_rt_bvh_gen1_quantization_frame_pack(&header, &frame),
           "an exactly-representable frame packs");
   require(agx_rt_bvh_gen1_quantization_frame_unpack(&header, &decoded),
           "a frame unpacks");

   for (unsigned axis = 0; axis < 3; ++axis) {
      require(decoded.origin[axis] == frame.origin[axis],
              "the frame origin survives a pack/unpack round trip");
      require(decoded.exponent[axis] == frame.exponent[axis],
              "the frame exponent survives a pack/unpack round trip");
   }

   require(agx_rt_bvh_gen1_axis_step(127) == 1.0f,
           "exponent 127 is a step of one");
   require(agx_rt_bvh_gen1_axis_step(128) == 2.0f,
           "exponent 128 is a step of two");
   require(agx_rt_bvh_gen1_axis_step(0) == 0.0f,
           "exponent zero is the degenerate step");

   /* The frame must not disturb the fields that share its words. */
   struct agx_rt_bvh_gen1_header reference;
   memset(&reference, 0, sizeof(reference));
   for (unsigned i = 0; i < 8; ++i)
      reference.words[i] = UINT64_C(0xa5a5a5a5a5a5a5a5);

   require((header.words[0] & ~(UINT64_C(0xff) << 8)) ==
              (reference.words[0] & ~(UINT64_C(0xff) << 8)),
           "packing exp.x leaves the rest of word 0 alone");
   require((header.words[4] & ~UINT64_C(0xffff)) ==
              (reference.words[4] & ~UINT64_C(0xffff)),
           "packing exp.y/exp.z leaves the rest of word 4 alone");
   for (unsigned axis = 0; axis < 3; ++axis) {
      require((header.words[1 + axis] & ~UINT64_C(0xffff)) ==
                 (reference.words[1 + axis] & ~UINT64_C(0xffff)),
              "packing an origin leaves the rest of its word alone");
   }
   require(header.words[5] == reference.words[5] &&
              header.words[6] == reference.words[6] &&
              header.words[7] == reference.words[7],
           "packing a frame does not touch words 5 to 7");

   /* A value the decoder could not reproduce must be refused, not truncated. */
   frame.origin[0] = 1.0f + (1.0f / 65536.0f);
   require(!agx_rt_bvh_gen1_quantization_frame_pack(&header, &frame),
           "an origin with low mantissa bits set is refused, not truncated");
}

static void
test_gen1_dequantization_is_apples(void)
{
   struct agx_rt_bvh_gen1_header header;
   float minimum[3], maximum[3];

   memset(&header, 0, sizeof(header));

   /* origin = (0, 0, 0), step = 1 on every axis. */
   header.words[0] = (uint64_t)127u << 8;
   header.words[4] = ((uint64_t)127u << 8) | 127u;

   struct agx_rt_bvh_gen1_quantized_bounds bounds = {{3, 5, 7, 11, 13, 17}};

   require(agx_rt_bvh_gen1_header_set_quantized_child_bounds(&header, 2,
                                                             &bounds),
           "quantised components deposit");

   require(agx_rt_bvh_gen1_child_bounds(&header, 2, minimum, maximum),
           "a child's bounds decode");

   /* Grouped (min.xyz, max.xyz), not interleaved per axis. */
   require(minimum[0] == 3.0f && minimum[1] == 5.0f && minimum[2] == 7.0f,
           "components 0..2 are the minimum corner in x, y, z order");
   require(maximum[0] == 11.0f && maximum[1] == 13.0f && maximum[2] == 17.0f,
           "components 3..5 are the maximum corner in x, y, z order");

   /* The node's own box spans 128 cells, not 127. */
   float node_min[3], node_max[3];
   require(agx_rt_bvh_gen1_node_bounds(&header, node_min, node_max),
           "the node's own box decodes");
   require(node_max[0] == 128.0f && node_max[1] == 128.0f &&
              node_max[2] == 128.0f,
           "the node box is origin plus 128 steps");
}

static void
test_gen1_quantization_contains_off_grid_boxes(void)
{
   uint64_t state = UINT64_C(0x0123456789abcdef);
   unsigned tightened = 0;

   for (unsigned trial = 0; trial < 20000; ++trial) {
      struct agx_rt_bvh_gen1_header header;
      struct agx_rt_bvh_gen1_quantization_frame frame;
      float node_min[3], node_max[3];

      /* A node box at an awkward offset and an awkward size.  The centres are
       * irrational-looking on purpose so neither corner nor extent lands on a
       * power of two. */
      const float centre = (trial & 1) ? -37.317f : 91.733f;
      const float spread = 1.0f + (float)(trial % 53) * 0.7717f;

      for (unsigned axis = 0; axis < 3; ++axis) {
         const float a = random_coordinate(&state, centre, spread);
         const float b = random_coordinate(&state, centre, spread);

         node_min[axis] = a < b ? a : b;
         node_max[axis] = a < b ? b : a;
      }

      if (!agx_rt_bvh_gen1_quantization_frame_fit(node_min, node_max, &frame))
         continue;

      memset(&header, 0, sizeof(header));
      require(agx_rt_bvh_gen1_quantization_frame_pack(&header, &frame),
              "a fitted frame is always exactly representable");

      for (unsigned child = 0; child < 8; ++child) {
         float child_min[3], child_max[3];

         for (unsigned axis = 0; axis < 3; ++axis) {
            const float a = random_coordinate(
               &state, (node_min[axis] + node_max[axis]) * 0.5f,
               node_max[axis] - node_min[axis]);
            const float b = random_coordinate(
               &state, (node_min[axis] + node_max[axis]) * 0.5f,
               node_max[axis] - node_min[axis]);

            child_min[axis] = a < b ? a : b;
            child_max[axis] = a < b ? b : a;

            /* Keep the child inside the node, as a real builder would. */
            if (child_min[axis] < node_min[axis])
               child_min[axis] = node_min[axis];
            if (child_max[axis] > node_max[axis])
               child_max[axis] = node_max[axis];
         }

         if (!agx_rt_bvh_gen1_header_encode_child_bounds(&header, child,
                                                         child_min, child_max))
            continue;

         float decoded_min[3], decoded_max[3];
         require(agx_rt_bvh_gen1_child_bounds(&header, child, decoded_min,
                                              decoded_max),
                 "an encoded child decodes");

         for (unsigned axis = 0; axis < 3; ++axis) {
            const float step = agx_rt_bvh_gen1_axis_step(frame.exponent[axis]);

            /* Containment.  This is the whole correctness property: rounding
             * either corner the wrong way shrinks the box and silently drops
             * intersections. */
            require(decoded_min[axis] <= child_min[axis],
                    "the decoded minimum never rises above the true minimum");
            require(decoded_max[axis] >= child_max[axis],
                    "the decoded maximum never falls below the true maximum");

            /* Tightness: conservative, but only by one grid cell per side, so
             * a lazily-correct encoder that just writes 0 and 127 fails. */
            require(decoded_min[axis] > child_min[axis] - step - step,
                    "the decoded minimum is within one grid cell");
            require(decoded_max[axis] < child_max[axis] + step + step,
                    "the decoded maximum is within one grid cell");

            if (decoded_min[axis] < child_min[axis] ||
                decoded_max[axis] > child_max[axis])
               ++tightened;
         }
      }
   }

   /* Guard against the check above passing vacuously on grid-aligned data:
    * the point of this test is that the boxes are OFF the grid, so the great
    * majority of corners must actually have been rounded outward. */
   require(tightened > 100000,
           "the randomized boxes really are off-grid (rounding was exercised)");
}

static void
test_gen1_quantization_rounding_direction(void)
{
   struct agx_rt_bvh_gen1_header header;
   struct agx_rt_bvh_gen1_quantization_frame frame;
   float decoded_min[3], decoded_max[3];

   memset(&header, 0, sizeof(header));

   /* origin 0, step 1: a box from 2.5 to 6.5 must become 2..7, never 3..6
    * and never a round-to-nearest 2..6 or 3..7. */
   frame.origin[0] = frame.origin[1] = frame.origin[2] = 0.0f;
   frame.exponent[0] = frame.exponent[1] = frame.exponent[2] = 127;
   require(agx_rt_bvh_gen1_quantization_frame_pack(&header, &frame),
           "the unit frame packs");

   const float child_min[3] = {2.5f, 2.5f, 2.5f};
   const float child_max[3] = {6.5f, 6.5f, 6.5f};

   require(agx_rt_bvh_gen1_header_encode_child_bounds(&header, 0, child_min,
                                                      child_max),
           "an off-grid box encodes");
   require(agx_rt_bvh_gen1_child_bounds(&header, 0, decoded_min, decoded_max),
           "the off-grid box decodes");

   for (unsigned axis = 0; axis < 3; ++axis) {
      require(decoded_min[axis] == 2.0f,
              "an off-grid minimum floors rather than rounds to nearest");
      require(decoded_max[axis] == 7.0f,
              "an off-grid maximum ceils rather than rounds to nearest");
   }

   /* Negative coordinates exercise the bfloat16 origin rounding, where
    * truncating the mantissa moves the value the WRONG way. */
   const float negative_min[3] = {-1.0009765625f, -3.5f, -0.1f};
   const float negative_max[3] = {4.25f, 0.0f, 7.7f};
   struct agx_rt_bvh_gen1_quantization_frame negative_frame;

   require(agx_rt_bvh_gen1_quantization_frame_fit(negative_min, negative_max,
                                                  &negative_frame),
           "a box straddling zero fits a frame");

   for (unsigned axis = 0; axis < 3; ++axis) {
      require(negative_frame.origin[axis] <= negative_min[axis],
              "the fitted origin never rises above the box minimum");
   }

   /* A degenerate axis must encode, and must decode back to a flat box. */
   const float flat_min[3] = {1.0f, 2.0f, 3.0f};
   const float flat_max[3] = {1.0f, 2.0f, 3.0f};
   struct agx_rt_bvh_gen1_quantization_frame flat_frame;
   struct agx_rt_bvh_gen1_quantized_bounds flat_bounds;

   require(agx_rt_bvh_gen1_quantization_frame_fit(flat_min, flat_max,
                                                  &flat_frame),
           "a zero-extent box fits a frame");
   require(agx_rt_bvh_gen1_quantize_child_bounds(&flat_frame, flat_min,
                                                 flat_max, &flat_bounds),
           "a zero-extent box quantises");
   for (unsigned i = 0; i < 6; ++i) {
      require(flat_bounds.component[i] == 0,
              "a degenerate axis quantises to the origin cell");
   }

   /* An inverted box is a caller bug and must be refused, not silently
    * normalised. */
   const float inverted_min[3] = {5.0f, 0.0f, 0.0f};
   const float inverted_max[3] = {1.0f, 1.0f, 1.0f};
   struct agx_rt_bvh_gen1_quantization_frame ignored;

   require(!agx_rt_bvh_gen1_quantization_frame_fit(inverted_min, inverted_max,
                                                   &ignored),
           "an inverted box is refused");
}

int
main(void)
{
   require(sizeof(struct agx_rt_bvh_gen1_header) ==
              AGX_RT_BVH_GEN1_HEADER_SIZE,
           "a Gen1 node header is exactly 64 bytes");
   require(sizeof(struct agx_rt_bvh_gen5_header) ==
              AGX_RT_BVH_GEN5_HEADER_SIZE,
           "a Gen5 node header is exactly 64 bytes");
   require(AGX_RT_BVH_GEN1_HW_AABB_LEAF_SIZE == 48,
           "a Gen1 hardware AABB leaf record is exactly 48 bytes");
   test_gen1_unpack_and_control();
   test_gen1_leaf_layout();
   test_gen1_quantized_bounds();
   test_gen1_quantization_frame_roundtrip();
   test_gen1_dequantization_is_apples();
   test_gen1_quantization_contains_off_grid_boxes();
   test_gen1_quantization_rounding_direction();
   test_gen1_hw_triangle_leaf();
   test_gen1_hw_aabb_leaf();
   test_gen1_randomized_assembly_equivalence();
   test_unpack_and_signed_child_space();
   test_leaf_and_inner_counts();
   test_qtb_and_vrb_fields();
   test_offsets_for_every_configuration();
   test_unknown_bits_do_not_affect_addressing();

   printf("agx_rt_bvh: PASS (%u checks)\n", checks);
   return 0;
}
