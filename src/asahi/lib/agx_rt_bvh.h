/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Apple RIA Gen1/Gen5 BVH8 node-header child-layout decoding, plus the Gen1
 * quantisation frame in both directions.
 *
 * G17P's hardware ray-traversal unit walks this format directly, so an
 * acceleration structure we build has to be written in it.  The child-layout
 * accessors below are decode-only; the quantisation frame is the one part
 * that also encodes, because it is the part a builder cannot avoid getting
 * exactly right -- see the rounding note on
 * agx_rt_bvh_gen1_quantize_child_bounds().
 *
 * These accessors are an independent implementation of the six pure
 * RIANodeHeaderGen5 child-layout accessors in AGXMetalG17X.  They were
 * validated against the Apple implementation over 64,000,000 randomized
 * 64-byte headers with no mismatches.  Keep the two still-unidentified bit
 * ranges (w0[8:15] and w7[13:15]) out of this interface: none participates in
 * a child-section address.
 *
 * Gen5 offsets are signed byte offsets relative to the acceleration structure.
 * Its encoded child-space base is a signed 32-bit count of 64-byte slots.
 *
 * Gen1 is the node format used by RIA BVH versions 1 through 4.  The accessors
 * below are a direct transcription of AGXMetalG17X's
 * extractDataFromBVH<RIANodeHeaderGen1> and
 * RIANodeHeaderGen1::getQuantizedChildBounds.  In particular, the two ends of
 * the eight-way branching range have special encodings: control 0 means eight
 * inner children and no leaf boxes, while control 0x3f means no inner children
 * and eight leaf boxes.
 *
 * This header intentionally depends only on the C standard library so it can
 * be shared by Vulkan, Gallium and the native D3D12 frontend.
 */

#ifndef AGX_RT_BVH_H
#define AGX_RT_BVH_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AGX_RT_BVH_GEN1_HEADER_SIZE 64u
#define AGX_RT_BVH_GEN1_HW_TRIANGLE_LEAF_SIZE 64u
#define AGX_RT_BVH_GEN1_HW_TRIANGLE_LEAF_MARKER 0x50u
#define AGX_RT_BVH_GEN1_HW_TRIANGLE_ID_MASK 0x3fffffffu
#define AGX_RT_BVH_GEN1_HW_AABB_LEAF_SIZE 48u
#define AGX_RT_BVH_GEN1_HW_AABB_LEAF_MARKER 0x18u
#define AGX_RT_BVH_GEN1_HW_AABB_ID_MASK 0x3fffffffu
#define AGX_RT_BVH_GEN5_HEADER_SIZE 64u
#define AGX_RT_BVH_GEN5_INVALID_QTB_COUNT 4u

struct agx_rt_bvh_gen1_header {
   uint64_t words[8];
};

struct agx_rt_bvh_gen5_header {
   uint64_t words[8];
};

struct agx_rt_bvh_gen1_hw_triangle_leaf {
   /* Four-lane structure-of-arrays vertex payload: X, Y, then Z.  An
    * unpaired triangle duplicates its third vertex in lane three. */
   float coordinate[3][4];

   /* The low 30 bits are IDs.  Preserve the complete words because the two
    * high bits have not yet been assigned safe public names. */
   uint32_t geometry_word;
   uint32_t primitive_word[2];
};

struct agx_rt_bvh_gen1_hw_aabb_leaf {
   /* The marker occupies only the low byte.  Public-Metal captures prove the
    * upper 24 bits are the geometry's intersection-function-table offset. */
   uint32_t header;
   float minimum[3];
   float maximum[3];

   /* Preserve every packed bit.  The low/high 30-bit fields are geometry and
    * primitive IDs respectively, low-word bit 31 is the opaque flag, and
    * high-word bit 30 gates the signed relative primitive-data address at
    * byte 40.  Low-word bit 30 and high-word bit 31 remain unnamed. */
   uint64_t packed_ids_and_flags;
   uint32_t word36;
   uint64_t primitive_data_relative_word;
   int64_t primitive_data_relative;
};

#ifdef __cplusplus
static_assert(sizeof(float) == sizeof(uint32_t),
              "RIA triangle leaves require 32-bit float");
#else
_Static_assert(sizeof(float) == sizeof(uint32_t),
               "RIA triangle leaves require 32-bit float");
#endif

static inline uint32_t
agx_rt_bvh_load_le32(const uint8_t *src)
{
   uint32_t value = 0;

   for (unsigned i = 0; i < 4; ++i)
      value |= (uint32_t)src[i] << (i * 8);

   return value;
}

static inline uint64_t
agx_rt_bvh_load_le64(const uint8_t *src)
{
   uint64_t value = 0;

   for (unsigned i = 0; i < 8; ++i)
      value |= (uint64_t)src[i] << (i * 8);

   return value;
}

/* Decode the exact 48-byte hardware AABB leaf emitted by the public Metal
 * Gen1 builder.  `available` is the remaining mapped-buffer span, so a larger
 * value is valid; this helper consumes only the first fixed-size record. */
static inline bool
agx_rt_bvh_gen1_hw_aabb_leaf_unpack(
   const void *packed, size_t available,
   struct agx_rt_bvh_gen1_hw_aabb_leaf *leaf)
{
   if (!leaf)
      return false;

   memset(leaf, 0, sizeof(*leaf));

   if (!packed || available < AGX_RT_BVH_GEN1_HW_AABB_LEAF_SIZE)
      return false;

   const uint8_t *bytes = (const uint8_t *)packed;
   leaf->header = agx_rt_bvh_load_le32(bytes);
   if ((leaf->header & 0xffu) != AGX_RT_BVH_GEN1_HW_AABB_LEAF_MARKER) {
      memset(leaf, 0, sizeof(*leaf));
      return false;
   }

   for (unsigned axis = 0; axis < 3; ++axis) {
      uint32_t bits = agx_rt_bvh_load_le32(bytes + 4 + axis * 4);
      memcpy(&leaf->minimum[axis], &bits, sizeof(bits));
      bits = agx_rt_bvh_load_le32(bytes + 16 + axis * 4);
      memcpy(&leaf->maximum[axis], &bits, sizeof(bits));
   }

   leaf->packed_ids_and_flags = agx_rt_bvh_load_le64(bytes + 28);
   leaf->word36 = agx_rt_bvh_load_le32(bytes + 36);
   leaf->primitive_data_relative_word = agx_rt_bvh_load_le64(bytes + 40);
   if (leaf->packed_ids_and_flags & (UINT64_C(1) << 62)) {
      memcpy(&leaf->primitive_data_relative,
             &leaf->primitive_data_relative_word,
             sizeof(leaf->primitive_data_relative_word));
   }

   return true;
}

static inline uint32_t
agx_rt_bvh_gen1_hw_aabb_intersection_function_table_offset(
   const struct agx_rt_bvh_gen1_hw_aabb_leaf *leaf)
{
   return leaf ? leaf->header >> 8 : 0;
}

static inline uint32_t
agx_rt_bvh_gen1_hw_aabb_geometry_id(
   const struct agx_rt_bvh_gen1_hw_aabb_leaf *leaf)
{
   return leaf ? (uint32_t)leaf->packed_ids_and_flags &
                    AGX_RT_BVH_GEN1_HW_AABB_ID_MASK
               : 0;
}

static inline uint32_t
agx_rt_bvh_gen1_hw_aabb_primitive_id(
   const struct agx_rt_bvh_gen1_hw_aabb_leaf *leaf)
{
   return leaf ? (uint32_t)(leaf->packed_ids_and_flags >> 32) &
                    AGX_RT_BVH_GEN1_HW_AABB_ID_MASK
               : 0;
}

static inline bool
agx_rt_bvh_gen1_hw_aabb_is_opaque(
   const struct agx_rt_bvh_gen1_hw_aabb_leaf *leaf)
{
   return leaf && (leaf->packed_ids_and_flags & (UINT64_C(1) << 31));
}

static inline bool
agx_rt_bvh_gen1_hw_aabb_has_primitive_data(
   const struct agx_rt_bvh_gen1_hw_aabb_leaf *leaf)
{
   return leaf && (leaf->packed_ids_and_flags & (UINT64_C(1) << 62));
}

/* Decode the exact 64-byte hardware triangle leaf used by the real Gen1 BVH
 * capture.  This helper models no other leaf type and accepts no alternate
 * marker.  `available` makes mapped-buffer truncation an explicit failure. */
static inline bool
agx_rt_bvh_gen1_hw_triangle_leaf_unpack(
   const void *packed, size_t available,
   struct agx_rt_bvh_gen1_hw_triangle_leaf *leaf)
{
   if (!leaf)
      return false;

   memset(leaf, 0, sizeof(*leaf));

   if (!packed || available < AGX_RT_BVH_GEN1_HW_TRIANGLE_LEAF_SIZE)
      return false;

   const uint8_t *bytes = (const uint8_t *)packed;
   if (agx_rt_bvh_load_le32(bytes) !=
       AGX_RT_BVH_GEN1_HW_TRIANGLE_LEAF_MARKER)
      return false;

   for (unsigned axis = 0; axis < 3; ++axis) {
      for (unsigned lane = 0; lane < 4; ++lane) {
         const uint32_t bits =
            agx_rt_bvh_load_le32(bytes + 4 + 4 * (axis * 4 + lane));
         memcpy(&leaf->coordinate[axis][lane], &bits, sizeof(bits));
      }
   }

   leaf->geometry_word = agx_rt_bvh_load_le32(bytes + 52);
   leaf->primitive_word[0] = agx_rt_bvh_load_le32(bytes + 56);
   leaf->primitive_word[1] = agx_rt_bvh_load_le32(bytes + 60);
   return true;
}

static inline uint32_t
agx_rt_bvh_gen1_hw_triangle_geometry_id(
   const struct agx_rt_bvh_gen1_hw_triangle_leaf *leaf)
{
   return leaf ? leaf->geometry_word & AGX_RT_BVH_GEN1_HW_TRIANGLE_ID_MASK : 0;
}

static inline uint32_t
agx_rt_bvh_gen1_hw_triangle_primitive_id(
   const struct agx_rt_bvh_gen1_hw_triangle_leaf *leaf, unsigned triangle)
{
   return leaf && triangle < 2
             ? leaf->primitive_word[triangle] &
                  AGX_RT_BVH_GEN1_HW_TRIANGLE_ID_MASK
             : 0;
}

/* Unpack explicitly instead of casting mapped GPU memory. */
static inline bool
agx_rt_bvh_gen1_header_unpack(const void *packed,
                              struct agx_rt_bvh_gen1_header *header)
{
   if (!packed || !header)
      return false;

   const uint8_t *bytes = (const uint8_t *)packed;
   for (unsigned i = 0; i < 8; ++i)
      header->words[i] = agx_rt_bvh_load_le64(bytes + (i * 8));

   return true;
}

static inline uint32_t
agx_rt_bvh_gen1_control(const struct agx_rt_bvh_gen1_header *header)
{
   return (uint32_t)(header->words[7] & 0x3fu);
}

/* The all-inner and all-leaf endpoints are encoded as 0 and 0x3f rather
 * than as ordinary three-bit counts. */
static inline uint32_t
agx_rt_bvh_gen1_inner_child_count(
   const struct agx_rt_bvh_gen1_header *header)
{
   const uint32_t control = agx_rt_bvh_gen1_control(header);

   if (control == 0)
      return 8;
   if (control == 0x3f)
      return 0;
   return control & 7u;
}

static inline uint32_t
agx_rt_bvh_gen1_leaf_box_count(
   const struct agx_rt_bvh_gen1_header *header)
{
   const uint32_t control = agx_rt_bvh_gen1_control(header);

   if (control == 0x3f)
      return 8;
   return (control >> 3) & 7u;
}

static inline bool
agx_rt_bvh_gen1_branching_layout_valid(
   const struct agx_rt_bvh_gen1_header *header)
{
   return agx_rt_bvh_gen1_inner_child_count(header) +
             agx_rt_bvh_gen1_leaf_box_count(header) <=
          8;
}

static inline bool
agx_rt_bvh_gen1_uses_indexed_leaf_counts(
   const struct agx_rt_bvh_gen1_header *header)
{
   return (header->words[0] & 1u) != 0;
}

static inline uint32_t
agx_rt_bvh_gen1_leaf_type(const struct agx_rt_bvh_gen1_header *header)
{
   return (uint32_t)((header->words[7] >> 8) & 31u);
}

/* The low halves are deliberately ordered w5:w6.  Unlike the Gen5 byte
 * address accessors, Apple's Gen1 extractor treats this as an unsigned wide
 * node index. */
static inline uint32_t
agx_rt_bvh_gen1_child_space_slot(
   const struct agx_rt_bvh_gen1_header *header)
{
   const uint32_t high = ((uint32_t)header->words[5] & 0xffffu) << 16;
   const uint32_t low = (uint32_t)header->words[6] & 0xffffu;

   return high | low;
}

/* The four-bit field is the divisor used by Apple's extractor when it rounds
 * the decoded leaf extent into 64-byte child-space slots. */
static inline uint32_t
agx_rt_bvh_gen1_leaf_slot_capacity(
   const struct agx_rt_bvh_gen1_header *header)
{
   return (uint32_t)((header->words[0] >> 1) & 15u);
}

/* Each Gen1 child lane has a four-bit low part and a two-bit high part.  Zero
 * in the high part encodes four, so the decoded end lies in 1..18. */
static inline uint32_t
agx_rt_bvh_gen1_leaf_lane_end(
   const struct agx_rt_bvh_gen1_header *header, unsigned lane)
{
   if (lane >= 8)
      return 0;

   const uint32_t word = (uint32_t)header->words[lane];
   const uint32_t low = (word >> 16) & 15u;
   const uint32_t high_field = (word >> 20) & 3u;
   const uint32_t high = high_field ? high_field : 4u;

   return low + high;
}

/* In indexed mode each leaf box contributes one zero-based leaf index.  In
 * count mode the per-box fields are ends, so the maximum is the leaf extent.
 * Leaf lanes follow the inner-child lanes in the eight-way header. */
static inline uint32_t
agx_rt_bvh_gen1_leaf_count(const struct agx_rt_bvh_gen1_header *header)
{
   const uint32_t leaf_boxes = agx_rt_bvh_gen1_leaf_box_count(header);

   if (leaf_boxes == 0)
      return 0;
   if (agx_rt_bvh_gen1_uses_indexed_leaf_counts(header))
      return leaf_boxes;

   const uint32_t first = agx_rt_bvh_gen1_inner_child_count(header);
   uint32_t maximum = 0;

   for (uint32_t i = 0; i < leaf_boxes && first + i < 8; ++i) {
      const uint32_t end = agx_rt_bvh_gen1_leaf_lane_end(header, first + i);
      if (end > maximum)
         maximum = end;
   }

   return maximum;
}

static inline uint32_t
agx_rt_bvh_gen1_leaf_storage_slots(
   const struct agx_rt_bvh_gen1_header *header)
{
   const uint32_t capacity = agx_rt_bvh_gen1_leaf_slot_capacity(header);

   if (capacity == 0)
      return 0;

   return ((agx_rt_bvh_gen1_leaf_count(header) + capacity - 1u) / capacity) &
          0x03ffffffu;
}

/* This is Apple's innerChildOffset: the child-space base follows the rounded
 * leaf storage, and the result remains a 32-bit wide-node index. */
static inline uint32_t
agx_rt_bvh_gen1_inner_child_offset(
   const struct agx_rt_bvh_gen1_header *header)
{
   return agx_rt_bvh_gen1_child_space_slot(header) +
          agx_rt_bvh_gen1_leaf_storage_slots(header);
}

struct agx_rt_bvh_gen1_quantized_bounds {
   uint32_t component[6];
};

/* Apple returns these six seven-bit components in descending bit order.  The
 * axis and min/max roles ARE now correlated -- see
 * agx_rt_bvh_gen1_child_bounds() below -- and are grouped
 * (min.x, min.y, min.z, max.x, max.y, max.z), not interleaved per axis. */
static inline bool
agx_rt_bvh_gen1_quantized_child_bounds(
   const struct agx_rt_bvh_gen1_header *header, unsigned child,
   struct agx_rt_bvh_gen1_quantized_bounds *bounds)
{
   if (!header || !bounds || child >= 8)
      return false;

   const uint64_t word = header->words[child];
   bounds->component[0] = (uint32_t)((word >> 57) & 0x7fu);
   bounds->component[1] = (uint32_t)((word >> 50) & 0x7fu);
   bounds->component[2] = (uint32_t)((word >> 43) & 0x7fu);
   bounds->component[3] = (uint32_t)((word >> 36) & 0x7fu);
   bounds->component[4] = (uint32_t)((word >> 29) & 0x7fu);
   bounds->component[5] = (uint32_t)((word >> 22) & 0x7fu);
   return true;
}

/*
 * The quantisation frame.
 *
 * A Gen1 node stores its children's bounds as six seven-bit numbers in a grid
 * whose origin and per-axis step live in the node header itself.  Measured
 * from Apple's own dequantiser inside extractDataFromBVH<RIANodeHeaderGen1>
 * (AGXMetalG17P), and confirmed independently on the encode side by the
 * shipped gen-2 build kernels:
 *
 *   origin.x = words[1][0:15]   origin.y = words[2][0:15]
 *   origin.z = words[3][0:15]   -- each a bfloat16, i.e. the top half of a
 *                                  binary32, so the float is bits << 16
 *   exp.x    = words[0][8:15]   exp.y    = words[4][8:15]
 *   exp.z    = words[4][0:7]    -- each a RAW binary32 exponent byte, so the
 *                                  step is bitcast<float>(exp << 23)
 *
 * Apple reconstructs BOTH corners with the same origin, the same step and the
 * same fused multiply-add: there is no +1, no half-ULP bias and no asymmetry
 * on the decode side.  The asymmetry lives entirely in the encoder, which
 * floors the minimum triple and ceils the maximum triple.
 *
 * exp == 0 gives a step of +0.0, which is the legal degenerate-axis encoding.
 *
 * Gen5 uses this identical frame with exactly one field moved: its exp.x is at
 * words[7][0:7].  Only Gen1 is encodable here.
 */
struct agx_rt_bvh_gen1_quantization_frame {
   float origin[3];
   uint8_t exponent[3];
};

static inline float
agx_rt_bvh_bitcast_u32_to_f32(uint32_t bits)
{
   float value;

   memcpy(&value, &bits, sizeof(value));
   return value;
}

static inline uint32_t
agx_rt_bvh_bitcast_f32_to_u32(float value)
{
   uint32_t bits;

   memcpy(&bits, &value, sizeof(bits));
   return bits;
}

/*
 * Floating-point helpers done with bit arithmetic rather than <math.h>, so
 * this header keeps its promise to depend only on the freestanding C library
 * and stays linkable from Vulkan, Gallium and the D3D12 frontend without
 * dragging in libm.  Both rounding helpers are restricted to the
 * non-negative, in-range domain the quantiser actually uses.
 */
static inline bool
agx_rt_bvh_is_finite(float value)
{
   return (agx_rt_bvh_bitcast_f32_to_u32(value) & UINT32_C(0x7f800000)) !=
          UINT32_C(0x7f800000);
}

static inline bool
agx_rt_bvh_in_trunc_range(float value)
{
   return value >= 0.0f && value < 2147483648.0f;
}

static inline float
agx_rt_bvh_floor_nonneg(float value)
{
   if (!agx_rt_bvh_in_trunc_range(value))
      return value;

   return (float)(uint32_t)value;
}

static inline float
agx_rt_bvh_ceil_nonneg(float value)
{
   if (!agx_rt_bvh_in_trunc_range(value))
      return value;

   const float truncated = (float)(uint32_t)value;

   return truncated == value ? truncated : truncated + 1.0f;
}

/* The per-axis grid step.  A raw binary32 exponent byte, so exp 127 is 1.0f
 * and exp 0 is +0.0f rather than a subnormal. */
static inline float
agx_rt_bvh_gen1_axis_step(uint8_t exponent)
{
   return agx_rt_bvh_bitcast_u32_to_f32((uint32_t)exponent << 23);
}

static inline bool
agx_rt_bvh_gen1_quantization_frame_unpack(
   const struct agx_rt_bvh_gen1_header *header,
   struct agx_rt_bvh_gen1_quantization_frame *frame)
{
   if (!header || !frame)
      return false;

   for (unsigned axis = 0; axis < 3; ++axis) {
      const uint32_t half = (uint32_t)(header->words[1 + axis] & 0xffffu);
      frame->origin[axis] = agx_rt_bvh_bitcast_u32_to_f32(half << 16);
   }

   frame->exponent[0] = (uint8_t)((header->words[0] >> 8) & 0xffu);
   frame->exponent[1] = (uint8_t)((header->words[4] >> 8) & 0xffu);
   frame->exponent[2] = (uint8_t)(header->words[4] & 0xffu);
   return true;
}

/* Decode one child's bounds to world space.  Apple's exact arithmetic. */
static inline bool
agx_rt_bvh_gen1_child_bounds(const struct agx_rt_bvh_gen1_header *header,
                             unsigned child, float minimum[3], float maximum[3])
{
   struct agx_rt_bvh_gen1_quantization_frame frame;
   struct agx_rt_bvh_gen1_quantized_bounds bounds;

   if (!minimum || !maximum)
      return false;
   if (!agx_rt_bvh_gen1_quantization_frame_unpack(header, &frame))
      return false;
   if (!agx_rt_bvh_gen1_quantized_child_bounds(header, child, &bounds))
      return false;

   for (unsigned axis = 0; axis < 3; ++axis) {
      const float step = agx_rt_bvh_gen1_axis_step(frame.exponent[axis]);

      minimum[axis] = frame.origin[axis] + (float)bounds.component[axis] * step;
      maximum[axis] =
         frame.origin[axis] + (float)bounds.component[3 + axis] * step;
   }

   return true;
}

/*
 * Quantise one child box against a frame.
 *
 * The minimum triple floors and the maximum triple ceils, so the encoded box
 * always contains the real one.  That asymmetry is the whole correctness
 * argument: the decoder applies one affine map to both corners, so rounding
 * either corner the wrong way yields a box strictly smaller than the true one
 * and silently drops intersections.
 */
static inline bool
agx_rt_bvh_gen1_quantize_child_bounds(
   const struct agx_rt_bvh_gen1_quantization_frame *frame,
   const float minimum[3], const float maximum[3],
   struct agx_rt_bvh_gen1_quantized_bounds *bounds)
{
   if (!frame || !minimum || !maximum || !bounds)
      return false;

   for (unsigned axis = 0; axis < 3; ++axis) {
      const float step = agx_rt_bvh_gen1_axis_step(frame->exponent[axis]);

      if (!agx_rt_bvh_is_finite(minimum[axis]) ||
          !agx_rt_bvh_is_finite(maximum[axis]))
         return false;
      if (!(minimum[axis] <= maximum[axis]))
         return false;
      if (minimum[axis] < frame->origin[axis])
         return false;

      if (!(step > 0.0f)) {
         /* Degenerate axis: the frame carries no extent, so the box must not
          * need any. */
         if (minimum[axis] != frame->origin[axis] ||
             maximum[axis] != frame->origin[axis])
            return false;

         bounds->component[axis] = 0;
         bounds->component[3 + axis] = 0;
         continue;
      }

      const float low =
         agx_rt_bvh_floor_nonneg((minimum[axis] - frame->origin[axis]) / step);
      const float high =
         agx_rt_bvh_ceil_nonneg((maximum[axis] - frame->origin[axis]) / step);

      if (low < 0.0f || high > 127.0f)
         return false;

      bounds->component[axis] = (uint32_t)low;
      bounds->component[3 + axis] = (uint32_t)high;
   }

   return true;
}

/* Unpack explicitly instead of casting mapped GPU memory.  This accepts
 * unaligned nodes and gives the same result on either host byte order. */
static inline bool
agx_rt_bvh_gen5_header_unpack(const void *packed,
                              struct agx_rt_bvh_gen5_header *header)
{
   if (!packed || !header)
      return false;

   const uint8_t *bytes = (const uint8_t *)packed;
   for (unsigned i = 0; i < 8; ++i)
      header->words[i] = agx_rt_bvh_load_le64(bytes + (i * 8));

   return true;
}

static inline uint32_t
agx_rt_bvh_gen5_config(const struct agx_rt_bvh_gen5_header *header)
{
   return (uint32_t)(header->words[0] & 7u);
}

/* Index of the first inner-child slot.  Leaf-box fields occupy [0, n), and
 * inner-child contributions occupy [n, 8). */
static inline uint32_t
agx_rt_bvh_gen5_inner_child_boundary(
   const struct agx_rt_bvh_gen5_header *header)
{
   return (uint32_t)((header->words[0] >> 3) & 15u);
}

static inline bool
agx_rt_bvh_gen5_uses_indexed_leaf_counts(
   const struct agx_rt_bvh_gen5_header *header)
{
   return ((header->words[0] >> 7) & 1u) != 0;
}

static inline uint32_t
agx_rt_bvh_gen5_leaf_type(const struct agx_rt_bvh_gen5_header *header)
{
   return (uint32_t)((header->words[7] >> 8) & 31u);
}

/* The low halves are deliberately ordered w5:w6. */
static inline int32_t
agx_rt_bvh_gen5_child_space_slot(
   const struct agx_rt_bvh_gen5_header *header)
{
   const uint32_t high = ((uint32_t)header->words[5] & 0xffffu) << 16;
   const uint32_t low = (uint32_t)header->words[6] & 0xffffu;

   return (int32_t)(high | low);
}

/* Sixteen 3-bit fields: bits 16..18 and 19..21 of words 0..7. */
static inline uint32_t
agx_rt_bvh_gen5_slot(const struct agx_rt_bvh_gen5_header *header,
                     unsigned slot)
{
   if (slot >= 16)
      return 0;

   return (uint32_t)((header->words[slot >> 1] >>
                      (16 + (3 * (slot & 1)))) &
                     7u);
}

static inline uint32_t
agx_rt_bvh_gen5_leaf_count(const struct agx_rt_bvh_gen5_header *header)
{
   const uint32_t boundary =
      agx_rt_bvh_gen5_inner_child_boundary(header);

   if (boundary == 0)
      return 0;

   if (agx_rt_bvh_gen5_uses_indexed_leaf_counts(header)) {
      uint32_t maximum = 0;

      for (uint32_t i = 0; i < boundary; ++i) {
         const uint32_t value = agx_rt_bvh_gen5_slot(header, i);
         if (value > maximum)
            maximum = value;
      }

      return maximum + 1;
   }

   uint32_t count = 0;
   for (uint32_t i = 0; i < boundary; ++i) {
      const uint32_t value = agx_rt_bvh_gen5_slot(header, i);
      count += value ? value : 8u;
   }

   return count;
}

static inline uint32_t
agx_rt_bvh_gen5_inner_slots(const struct agx_rt_bvh_gen5_header *header)
{
   const uint32_t boundary =
      agx_rt_bvh_gen5_inner_child_boundary(header);

   if (boundary > 7)
      return 0;

   uint32_t count = 0;
   for (uint32_t i = boundary; i < 8; ++i)
      count += agx_rt_bvh_gen5_slot(header, i);

   return count;
}

/* The three 4-bit QTB occupancies are split across w6/w7. */
static inline void
agx_rt_bvh_gen5_qtb_occupancies(
   const struct agx_rt_bvh_gen5_header *header, uint32_t *a, uint32_t *b,
   uint32_t *c)
{
   const uint32_t w6 = (uint32_t)header->words[6];
   const uint32_t w7 = (uint32_t)header->words[7];
   const uint32_t occupancy_a =
      ((w6 >> 16) & 3u) | (((w7 >> 16) & 3u) << 2);
   const uint32_t occupancy_b =
      ((w6 >> 18) & 3u) | (((w7 >> 18) & 3u) << 2);
   const uint32_t occupancy_c =
      ((w6 >> 20) & 3u) | (((w7 >> 20) & 3u) << 2);

   if (a)
      *a = occupancy_a;
   if (b)
      *b = occupancy_b;
   if (c)
      *c = occupancy_c;
}

/* Apple returns 4 as an invalid-occupancy sentinel for configurations 2..4. */
static inline uint32_t
agx_rt_bvh_gen5_qtb_count(const struct agx_rt_bvh_gen5_header *header)
{
   const uint32_t config = agx_rt_bvh_gen5_config(header);

   if (config < 2 || config > 4)
      return 0;

   uint32_t a, b, c;
   agx_rt_bvh_gen5_qtb_occupancies(header, &a, &b, &c);

   if (a + b + c != (agx_rt_bvh_gen5_leaf_count(header) & 0xffffu))
      return AGX_RT_BVH_GEN5_INVALID_QTB_COUNT;
   if (c)
      return 3;
   if (b)
      return 2;
   if (a)
      return 1;
   return 0;
}

static inline bool
agx_rt_bvh_gen5_qtb_layout_valid(
   const struct agx_rt_bvh_gen5_header *header)
{
   return agx_rt_bvh_gen5_qtb_count(header) !=
          AGX_RT_BVH_GEN5_INVALID_QTB_COUNT;
}

/* Configuration 4's four VRB spans use slots 8..11.  Seven encodes eight
 * 32-byte units rather than seven. */
static inline uint32_t
agx_rt_bvh_gen5_vrb_size_bytes(
   const struct agx_rt_bvh_gen5_header *header)
{
   uint32_t units = 0;

   for (unsigned i = 8; i < 12; ++i) {
      const uint32_t value = agx_rt_bvh_gen5_slot(header, i);
      units += value == 7 ? 8u : value;
   }

   return units * 32u;
}

static inline int64_t
agx_rt_bvh_gen5_base_slots(const struct agx_rt_bvh_gen5_header *header)
{
   return (int64_t)agx_rt_bvh_gen5_child_space_slot(header) +
          (int64_t)agx_rt_bvh_gen5_inner_slots(header);
}

static inline int64_t
agx_rt_bvh_gen5_offset_child_qtbs_bytes(
   const struct agx_rt_bvh_gen5_header *header)
{
   return agx_rt_bvh_gen5_base_slots(header) * 64;
}

static inline int64_t
agx_rt_bvh_gen5_offset_child_hw_leaves_bytes(
   const struct agx_rt_bvh_gen5_header *header)
{
   int64_t slots = agx_rt_bvh_gen5_base_slots(header);

   if (agx_rt_bvh_gen5_config(header) == 2)
      slots += (int64_t)agx_rt_bvh_gen5_qtb_count(header);

   return slots * 64;
}

static inline int64_t
agx_rt_bvh_gen5_offset_vrbs_bytes(
   const struct agx_rt_bvh_gen5_header *header)
{
   const int64_t slots =
      agx_rt_bvh_gen5_base_slots(header) +
      (int64_t)agx_rt_bvh_gen5_qtb_count(header);

   return slots * 64;
}

static inline int64_t
agx_rt_bvh_gen5_offset_lhb_bytes(
   const struct agx_rt_bvh_gen5_header *header)
{
   const uint32_t config = agx_rt_bvh_gen5_config(header);
   const int64_t base_slots = agx_rt_bvh_gen5_base_slots(header);

   switch (config) {
   case 0:
   case 7:
      return base_slots * 64;
   case 1:
   case 2: {
      const int64_t hw =
         agx_rt_bvh_gen5_offset_child_hw_leaves_bytes(header);
      const int64_t end =
         hw + (36 * (int64_t)(agx_rt_bvh_gen5_leaf_count(header) & 0xffffu));
      return (int64_t)(((uint64_t)end + 31u) & ~UINT64_C(31));
   }
   case 3:
      return (base_slots +
              (int64_t)agx_rt_bvh_gen5_qtb_count(header)) *
             64;
   case 4:
      return (base_slots +
              (int64_t)agx_rt_bvh_gen5_qtb_count(header)) *
                64 +
             (int64_t)agx_rt_bvh_gen5_vrb_size_bytes(header);
   case 5:
      return 0;
   case 6:
      return agx_rt_bvh_gen5_offset_child_hw_leaves_bytes(header) +
             ((int64_t)(agx_rt_bvh_gen5_inner_child_boundary(header) &
                        0xffffu) *
              128);
   default:
      return 0;
   }
}

static inline int64_t
agx_rt_bvh_gen5_offset_child_leaves_bytes(
   const struct agx_rt_bvh_gen5_header *header)
{
   const uint32_t config = agx_rt_bvh_gen5_config(header);
   const uint32_t leaf_type = agx_rt_bvh_gen5_leaf_type(header);
   uint32_t extra;

   if (config == 6)
      extra = 0;
   else if (config == 5)
      extra = 4;
   else if (config == 7 && leaf_type == 0x1a)
      extra = 0;
   else {
      const uint32_t type_index = leaf_type - 0x0cu;
      const bool single_entry =
         type_index <= 0x10u && ((UINT32_C(0x0001d0c3) >> type_index) & 1u);

      extra = single_entry
                 ? 4
                 : ((agx_rt_bvh_gen5_leaf_count(header) & 0xffu) * 4);
   }

   return agx_rt_bvh_gen5_offset_lhb_bytes(header) + (int64_t)extra;
}

#ifdef __cplusplus
}
#endif

#endif
