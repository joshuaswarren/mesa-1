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
#include "agx_bytes.h"
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
   /* Historical API name: this is the full-precision geometry record read by
    * the native CPU extractor, not a QuantizedTriangleBlock. The native
    * layout reserves separate traversal blocks before these records. */
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
   leaf->header = agx_load_le32(bytes);
   if ((leaf->header & 0xffu) != AGX_RT_BVH_GEN1_HW_AABB_LEAF_MARKER) {
      memset(leaf, 0, sizeof(*leaf));
      return false;
   }

   for (unsigned axis = 0; axis < 3; ++axis) {
      uint32_t bits = agx_load_le32(bytes + 4 + axis * 4);
      memcpy(&leaf->minimum[axis], &bits, sizeof(bits));
      bits = agx_load_le32(bytes + 16 + axis * 4);
      memcpy(&leaf->maximum[axis], &bits, sizeof(bits));
   }

   leaf->packed_ids_and_flags = agx_load_le64(bytes + 28);
   leaf->word36 = agx_load_le32(bytes + 36);
   leaf->primitive_data_relative_word = agx_load_le64(bytes + 40);
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
   if (agx_load_le32(bytes) !=
       AGX_RT_BVH_GEN1_HW_TRIANGLE_LEAF_MARKER)
      return false;

   for (unsigned axis = 0; axis < 3; ++axis) {
      for (unsigned lane = 0; lane < 4; ++lane) {
         const uint32_t bits =
            agx_load_le32(bytes + 4 + 4 * (axis * 4 + lane));
         memcpy(&leaf->coordinate[axis][lane], &bits, sizeof(bits));
      }
   }

   leaf->geometry_word = agx_load_le32(bytes + 52);
   leaf->primitive_word[0] = agx_load_le32(bytes + 56);
   leaf->primitive_word[1] = agx_load_le32(bytes + 60);
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
      header->words[i] = agx_load_le64(bytes + (i * 8));

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
 * in the high part encodes four, so the decoded end lies in 1..19. */
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

   struct agx_rt_bvh_gen1_quantized_bounds result;

   for (unsigned axis = 0; axis < 3; ++axis) {
      const float step = agx_rt_bvh_gen1_axis_step(frame->exponent[axis]);

      if (!agx_rt_bvh_is_finite(minimum[axis]) ||
          !agx_rt_bvh_is_finite(maximum[axis]) ||
          !agx_rt_bvh_is_finite(frame->origin[axis]) ||
          frame->exponent[axis] == 255)
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

         result.component[axis] = 0;
         result.component[3 + axis] = 0;
         continue;
      }

      const float low =
         agx_rt_bvh_floor_nonneg((minimum[axis] - frame->origin[axis]) / step);
      const float high =
         agx_rt_bvh_ceil_nonneg((maximum[axis] - frame->origin[axis]) / step);

      if (!agx_rt_bvh_is_finite(low) || !agx_rt_bvh_is_finite(high) ||
          low < 0.0f || high > 127.0f)
         return false;

      uint32_t qmin = (uint32_t)low, qmax = (uint32_t)high;
      /* Division may round an off-grid coordinate onto an integer. Check
       * the actual reconstructed float before accepting either boundary. */
      while (frame->origin[axis] + (float)qmin * step > minimum[axis]) {
         if (!qmin)
            return false;
         --qmin;
      }
      while (frame->origin[axis] + (float)qmax * step < maximum[axis]) {
         if (qmax == 127)
            return false;
         ++qmax;
      }
      result.component[axis] = qmin;
      result.component[3 + axis] = qmax;
   }

   *bounds = result;
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
      header->words[i] = agx_load_le64(bytes + (i * 8));

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

/* Bounded native-format node and leaf construction. */
static inline bool
agx_rt_bvh_gen1_hw_triangle_leaf_pack(
   void *packed, size_t available,
   const struct agx_rt_bvh_gen1_hw_triangle_leaf *leaf)
{
   if (!packed || !leaf || available < AGX_RT_BVH_GEN1_HW_TRIANGLE_LEAF_SIZE)
      return false;

   uint8_t bytes[AGX_RT_BVH_GEN1_HW_TRIANGLE_LEAF_SIZE];
   agx_store_le32(bytes, AGX_RT_BVH_GEN1_HW_TRIANGLE_LEAF_MARKER);
   for (unsigned axis = 0; axis < 3; ++axis) {
      for (unsigned lane = 0; lane < 4; ++lane) {
         uint32_t bits;
         memcpy(&bits, &leaf->coordinate[axis][lane], sizeof(bits));
         agx_store_le32(bytes + 4 + 4 * (axis * 4 + lane), bits);
      }
   }
   agx_store_le32(bytes + 52, leaf->geometry_word);
   agx_store_le32(bytes + 56, leaf->primitive_word[0]);
   agx_store_le32(bytes + 60, leaf->primitive_word[1]);
   memcpy(packed, bytes, sizeof(bytes));
   return true;
}

static inline bool
agx_rt_bvh_gen1_hw_aabb_leaf_pack(
   void *packed, size_t available,
   const struct agx_rt_bvh_gen1_hw_aabb_leaf *leaf)
{
   if (!packed || !leaf || available < AGX_RT_BVH_GEN1_HW_AABB_LEAF_SIZE ||
       (leaf->header & 0xffu) != AGX_RT_BVH_GEN1_HW_AABB_LEAF_MARKER)
      return false;

   uint8_t bytes[AGX_RT_BVH_GEN1_HW_AABB_LEAF_SIZE];
   agx_store_le32(bytes, leaf->header);
   for (unsigned axis = 0; axis < 3; ++axis) {
      uint32_t bits;
      memcpy(&bits, &leaf->minimum[axis], sizeof(bits));
      agx_store_le32(bytes + 4 + axis * 4, bits);
      memcpy(&bits, &leaf->maximum[axis], sizeof(bits));
      agx_store_le32(bytes + 16 + axis * 4, bits);
   }
   agx_store_le64(bytes + 28, leaf->packed_ids_and_flags);
   agx_store_le32(bytes + 36, leaf->word36);
   agx_store_le64(bytes + 40, leaf->primitive_data_relative_word);
   memcpy(packed, bytes, sizeof(bytes));
   return true;
}

static inline bool
agx_rt_bvh_gen1_header_pack(
   void *packed, size_t available,
   const struct agx_rt_bvh_gen1_header *header)
{
   if (!packed || !header || available < AGX_RT_BVH_GEN1_HEADER_SIZE)
      return false;

   uint8_t bytes[AGX_RT_BVH_GEN1_HEADER_SIZE];
   for (unsigned i = 0; i < 8; ++i)
      agx_store_le64(bytes + i * 8, header->words[i]);
   memcpy(packed, bytes, sizeof(bytes));
   return true;
}

static inline bool
agx_rt_bvh_gen1_header_set_child_layout(
   struct agx_rt_bvh_gen1_header *header, unsigned inner_count,
   unsigned leaf_boxes, uint32_t child_space_slot, unsigned leaf_type,
   unsigned leaf_slot_capacity, bool indexed_leaf_counts)
{
   /* Indexed counts do not need a fixed slot capacity. Native M5 instance
    * roots use this form with capacity zero and two-slot instance records. */
   if (!header || inner_count > 8 || leaf_boxes > 8 ||
       inner_count + leaf_boxes == 0 || inner_count + leaf_boxes > 8 ||
       leaf_type > 31 || leaf_slot_capacity > 15 ||
       (leaf_boxes && !leaf_slot_capacity && !indexed_leaf_counts))
      return false;

   const uint32_t control = inner_count == 8 ? 0 :
                            leaf_boxes == 8 ? 0x3f :
                            (leaf_boxes << 3) | inner_count;
   header->words[0] = (header->words[0] & ~UINT64_C(0x1f)) |
                      (leaf_slot_capacity << 1) | indexed_leaf_counts;
   header->words[5] = (header->words[5] & ~UINT64_C(0xffff)) |
                      (child_space_slot >> 16);
   header->words[6] = (header->words[6] & ~UINT64_C(0xffff)) |
                      (child_space_slot & 0xffffu);
   header->words[7] = (header->words[7] & ~UINT64_C(0x1f3f)) |
                      (leaf_type << 8) | control;
   return true;
}

static inline bool
agx_rt_bvh_gen1_header_set_leaf_lane_end(
   struct agx_rt_bvh_gen1_header *header, unsigned lane, unsigned end)
{
   if (!header || lane >= 8 || !end || end > 19)
      return false;

   const unsigned low = end <= 16 ? end - 1 : 15;
   const unsigned high = (end - low) & 3u;
   header->words[lane] = (header->words[lane] & ~UINT64_C(0x3f0000)) |
                         ((uint64_t)low << 16) | ((uint64_t)high << 20);
   return true;
}

static inline unsigned
agx_rt_bvh_ceil_pow2_exponent(float value)
{
   const uint32_t bits = agx_rt_bvh_bitcast_f32_to_u32(value);
   const uint32_t exponent = (bits >> 23) & 0xffu;

   if (exponent == 0)
      return 0; /* subnormal: below the representable step range */

   /* An exact power of two is already the answer; anything else needs the
    * next one up. */
   return (bits & UINT32_C(0x7fffff)) ? exponent + 1u : exponent;
}

static inline bool
agx_rt_bvh_gen1_quantization_frame_pack(
   struct agx_rt_bvh_gen1_header *header,
   const struct agx_rt_bvh_gen1_quantization_frame *frame)
{
   if (!header || !frame)
      return false;

   for (unsigned axis = 0; axis < 3; ++axis) {
      const uint32_t bits = agx_rt_bvh_bitcast_f32_to_u32(frame->origin[axis]);

      /* The decoder drops the low sixteen mantissa bits, so only a value that
       * is already exactly representable may be written here.  Fitting the
       * frame is what rounds; this only deposits. */
      if ((bits & 0xffffu) || !agx_rt_bvh_is_finite(frame->origin[axis]) ||
          frame->exponent[axis] == 255)
         return false;
   }
   for (unsigned axis = 0; axis < 3; ++axis) {
      const uint32_t bits = agx_rt_bvh_bitcast_f32_to_u32(frame->origin[axis]);
      header->words[1 + axis] = (header->words[1 + axis] & ~UINT64_C(0xffff)) |
                                (uint64_t)(bits >> 16);
   }

   header->words[0] = (header->words[0] & ~(UINT64_C(0xff) << 8)) |
                      ((uint64_t)frame->exponent[0] << 8);
   header->words[4] = (header->words[4] & ~UINT64_C(0xffff)) |
                      ((uint64_t)frame->exponent[1] << 8) |
                      (uint64_t)frame->exponent[2];
   return true;
}

static inline bool
agx_rt_bvh_gen1_node_bounds(const struct agx_rt_bvh_gen1_header *header,
                            float minimum[3], float maximum[3])
{
   struct agx_rt_bvh_gen1_quantization_frame frame;

   if (!minimum || !maximum)
      return false;
   if (!agx_rt_bvh_gen1_quantization_frame_unpack(header, &frame))
      return false;

   for (unsigned axis = 0; axis < 3; ++axis) {
      const float step = agx_rt_bvh_gen1_axis_step(frame.exponent[axis]);

      minimum[axis] = frame.origin[axis];
      maximum[axis] = frame.origin[axis] + 128.0f * step;
   }

   return true;
}

static inline float
agx_rt_bvh_gen1_origin_round_down(float value)
{
   const uint32_t bits = agx_rt_bvh_bitcast_f32_to_u32(value);
   uint32_t half = bits >> 16;

   if ((bits & UINT32_C(0x80000000)) && (bits & 0xffffu))
      ++half;

   return agx_rt_bvh_bitcast_u32_to_f32(half << 16);
}

static inline bool
agx_rt_bvh_gen1_quantization_frame_fit(
   const float minimum[3], const float maximum[3],
   struct agx_rt_bvh_gen1_quantization_frame *frame)
{
   if (!minimum || !maximum || !frame)
      return false;

   struct agx_rt_bvh_gen1_quantization_frame result = {0};

   for (unsigned axis = 0; axis < 3; ++axis) {
      if (!agx_rt_bvh_is_finite(minimum[axis]) ||
          !agx_rt_bvh_is_finite(maximum[axis]))
         return false;
      if (!(minimum[axis] <= maximum[axis]))
         return false;

      const float origin = agx_rt_bvh_gen1_origin_round_down(minimum[axis]);

      if (!agx_rt_bvh_is_finite(origin) || origin > minimum[axis])
         return false;

      const float extent = maximum[axis] - origin;
      unsigned exponent;

      if (extent <= 0.0f) {
         /* A degenerate axis: step +0.0 puts both corners on the origin. */
         result.origin[axis] = origin;
         result.exponent[axis] = 0;
         continue;
      }

      /* Smallest power of two at least extent / 127. */
      exponent = agx_rt_bvh_ceil_pow2_exponent(extent / 127.0f);

      if (exponent < 1)
         exponent = 1;
      else if (exponent > 254)
         return false;

      /* Verify rather than trust the exponent arithmetic: bump the step until
       * the maximum corner actually quantises inside the grid.  This settles
       * the boundary cases frexpf alone does not. */
      for (;;) {
         const float step = agx_rt_bvh_gen1_axis_step((uint8_t)exponent);

         if (step > 0.0f &&
             agx_rt_bvh_ceil_nonneg((maximum[axis] - origin) / step) <= 127.0f)
            break;

         if (++exponent > 254)
            return false;
      }

      result.origin[axis] = origin;
      result.exponent[axis] = (uint8_t)exponent;
   }

   *frame = result;
   return true;
}

static inline bool
agx_rt_bvh_gen1_header_set_quantized_child_bounds(
   struct agx_rt_bvh_gen1_header *header, unsigned child,
   const struct agx_rt_bvh_gen1_quantized_bounds *bounds)
{
   static const unsigned shift[6] = {57, 50, 43, 36, 29, 22};

   if (!header || !bounds || child >= 8)
      return false;

   uint64_t word = header->words[child] & ((UINT64_C(1) << 22) - 1);

   for (unsigned i = 0; i < 6; ++i) {
      if (bounds->component[i] > 0x7fu)
         return false;

      word |= (uint64_t)(bounds->component[i] & 0x7fu) << shift[i];
   }

   header->words[child] = word;
   return true;
}

static inline bool
agx_rt_bvh_gen1_header_encode_child_bounds(
   struct agx_rt_bvh_gen1_header *header, unsigned child,
   const float minimum[3], const float maximum[3])
{
   struct agx_rt_bvh_gen1_quantization_frame frame;
   struct agx_rt_bvh_gen1_quantized_bounds bounds;

   if (!agx_rt_bvh_gen1_quantization_frame_unpack(header, &frame))
      return false;
   if (!agx_rt_bvh_gen1_quantize_child_bounds(&frame, minimum, maximum,
                                              &bounds))
      return false;

   return agx_rt_bvh_gen1_header_set_quantized_child_bounds(header, child,
                                                            &bounds);
}

/* Gen3 QTB encoding type 0, three-vertex block. This packs already quantized
 * coordinates in the native-tested 0..126 range; frame selection, primitive IDs and the other encoding types
 * are separate operations. Native encodeVertex places vertex 0 at bit 107,
 * then vertices 1 and 2 at bits 86 and 65. Within each vertex Z is lowest.
 * Preserve the frame/header and every other bit in the caller's block.
 */
static inline bool
agx_rt_qtb_gen3_type0_pack_vertex(void *packed, size_t available,
                                 unsigned vertex, const uint32_t xyz[3])
{
   if (!packed || !xyz || available < 16 || vertex >= 3 ||
       xyz[0] > 126 || xyz[1] > 126 || xyz[2] > 126)
      return false;

   uint8_t *bytes = packed;
   const unsigned first = 107 - 21 * vertex;
   const uint32_t value = (xyz[0] << 14) | (xyz[1] << 7) | xyz[2];
   for (unsigned bit = 0; bit < 21; ++bit) {
      const unsigned dst = first + bit;
      const uint8_t mask = 1u << (dst % 8);
      bytes[dst / 8] = (bytes[dst / 8] & ~mask) |
                       (((value >> bit) & 1u) ? mask : 0);
   }
   return true;
}

/* Type-2 QTB coordinates are 10-bit unsigned fixed point with one
 * fractional bit. All nine fields and the split exponent header were checked
 * against 64 native TriFast builder outputs, including non-planar triangles.
 * Vertex zero has four extra preceding bits. */
static inline bool
agx_rt_qtb_gen3_type2_pack_vertex_fp1(void *packed, size_t available,
                                    unsigned vertex, const uint32_t xyz[3])
{
   if (!packed || !xyz || available < 32 || vertex >= 3 ||
       xyz[0] > 1023 || xyz[1] > 1023 || xyz[2] > 1023)
      return false;

   static const unsigned first_bit[3] = {196, 162, 132};
   const uint32_t value = (xyz[0] << 20) | (xyz[1] << 10) | xyz[2];
   uint8_t *bytes = packed;
   for (unsigned bit = 0; bit < 30; ++bit) {
      const unsigned dst = first_bit[vertex] + bit;
      const uint8_t mask = 1u << (dst % 8);
      bytes[dst / 8] = (bytes[dst / 8] & ~mask) |
                       (((value >> bit) & 1u) ? mask : 0);
   }
   return true;
}

/* Compatibility entry point for integral quantized coordinates. */
static inline bool
agx_rt_qtb_gen3_type2_pack_vertex(void *packed, size_t available,
                                 unsigned vertex, const uint32_t xyz[3])
{
   if (!xyz || xyz[0] > 511 || xyz[1] > 511 || xyz[2] > 511)
      return false;
   const uint32_t fp1[3] = {xyz[0] << 1, xyz[1] << 1, xyz[2] << 1};
   return agx_rt_qtb_gen3_type2_pack_vertex_fp1(packed, available, vertex, fp1);
}

/* Unindexed, one-triangle native TriFast type-2 block. Exponents are ordered
 * XYZ at the API boundary; the wire order is Z at byte 1, X/Y at bytes 8/9.
 * Coordinates are fixed point with one fractional bit. */
static inline bool
agx_rt_qtb_gen3_type2_single_triangle_fp1(
   void *packed, size_t available, const uint16_t origin[3],
   const uint8_t exponent[3], const uint32_t xyz[3][3])
{
   if (!packed || !origin || !exponent || !xyz || available < 64)
      return false;

   for (unsigned axis = 0; axis < 3; ++axis) {
      if ((origin[axis] & 0x7f80) == 0x7f80 || exponent[axis] == 255)
         return false;
      for (unsigned vertex = 0; vertex < 3; ++vertex) {
         if (xyz[vertex][axis] > 1023)
            return false;
      }
   }

   uint8_t result[64] = {2};
   result[1] = exponent[2];
   result[8] = exponent[0];
   result[9] = exponent[1];
   for (unsigned axis = 0; axis < 3; ++axis) {
      result[2 + 2 * axis] = origin[axis];
      result[3 + 2 * axis] = origin[axis] >> 8;
   }
   for (unsigned vertex = 0; vertex < 3; ++vertex)
      agx_rt_qtb_gen3_type2_pack_vertex_fp1(result, sizeof(result), vertex,
                                            xyz[vertex]);

   memcpy(packed, result, sizeof(result));
   return true;
}

static inline bool
agx_rt_qtb_gen3_type2_single_triangle(
   void *packed, size_t available, const uint16_t origin[3],
   const uint8_t exponent[3], const uint32_t xyz[3][3])
{
   if (!xyz || !exponent)
      return false;
   uint32_t fp1[3][3];
   uint8_t native_exponent[3];
   for (unsigned axis = 0; axis < 3; ++axis) {
      /* Earlier integral callers used zero for a degenerate axis. */
      native_exponent[axis] = exponent[axis] ? exponent[axis] : 1;
      for (unsigned vertex = 0; vertex < 3; ++vertex) {
         if (xyz[vertex][axis] > 511)
            return false;
         fp1[vertex][axis] = xyz[vertex][axis] << 1;
      }
   }
   return agx_rt_qtb_gen3_type2_single_triangle_fp1(
      packed, available, origin, native_exponent, fp1);
}

/* Fit and encode the single-triangle block observed from the native builder.
 * This does not construct a BVH root or qualify traversal. Reject nonfinite
 * inputs, origins and overflowed extents before touching the destination. */
static inline bool
agx_rt_qtb_gen3_type2_from_triangle(void *packed, size_t available,
                                   const float vertices[3][3])
{
   if (!packed || !vertices || available < 64)
      return false;
   uint16_t origin[3];
   uint8_t exponent[3];
   uint32_t quantized[3][3];
   for (unsigned axis = 0; axis < 3; ++axis) {
      float minimum = vertices[0][axis], maximum = minimum;
      for (unsigned vertex = 0; vertex < 3; ++vertex) {
         const float value = vertices[vertex][axis];
         if (!agx_rt_bvh_is_finite(value))
            return false;
         if (value < minimum) minimum = value;
         if (value > maximum) maximum = value;
      }
      const float base = agx_rt_bvh_gen1_origin_round_down(minimum);
      const float extent = maximum - base;
      if (!agx_rt_bvh_is_finite(base) || !agx_rt_bvh_is_finite(extent))
         return false;
      origin[axis] = agx_rt_bvh_bitcast_f32_to_u32(base) >> 16;
      const unsigned e = (agx_rt_bvh_bitcast_f32_to_u32(extent) >> 23) & 0xff;
      exponent[axis] = e > 8 ? e - 7 : 1;
      /* Double arithmetic retains the fractional bit without overflowing the
       * reciprocal for very small float exponents. */
      const double scale = 4.0 / agx_rt_bvh_gen1_axis_step(exponent[axis]);
      for (unsigned vertex = 0; vertex < 3; ++vertex) {
         const double q = ((double)vertices[vertex][axis] - base) * scale;
         quantized[vertex][axis] = q <= 0.0 ? 0 : q >= 1023.0 ? 1023 : (uint32_t)q;
      }
   }
   return agx_rt_qtb_gen3_type2_single_triangle_fp1(
      packed, available, origin, exponent, quantized);
}

/* Unindexed Gen3 type-6 block, up to four unpaired triangles. The native
 * two/eight-triangle captures use four 24-bit XYZ slots per triangle, in
 * vertex order 2, 2, 1, 0. Each component is eight bits, unlike type 2's
 * ten-bit fixed-point coordinates. This is only the traversal payload;
 * explicit primitive IDs live in the associated full-precision records. */
static inline bool
agx_rt_qtb_gen3_type6_from_triangles(void *packed, size_t available,
                                  const float (*vertices)[3][3], unsigned count)
{
   if (!packed || !vertices || available < 64 || !count || count > 4)
      return false;

   float origin[3];
   uint8_t result[64] = {6};
   uint8_t exponent[3];
   for (unsigned axis = 0; axis < 3; ++axis) {
      float minimum = vertices[0][0][axis], maximum = minimum;
      for (unsigned t = 0; t < count; ++t)
         for (unsigned v = 0; v < 3; ++v) {
            float x = vertices[t][v][axis];
            if (!agx_rt_bvh_is_finite(x))
               return false;
            if (x < minimum) minimum = x;
            if (x > maximum) maximum = x;
         }
      origin[axis] = agx_rt_bvh_gen1_origin_round_down(minimum);
      float extent = maximum - origin[axis];
      if (!agx_rt_bvh_is_finite(origin[axis]) || !agx_rt_bvh_is_finite(extent))
         return false;
      unsigned e = (agx_rt_bvh_bitcast_f32_to_u32(extent) >> 23) & 0xff;
      exponent[axis] = e > 8 ? e - 7 : 1;
      uint16_t bf16 = agx_rt_bvh_bitcast_f32_to_u32(origin[axis]) >> 16;
      result[2 + axis * 2] = bf16;
      result[3 + axis * 2] = bf16 >> 8;
   }
   result[1] = exponent[2];
   result[8] = exponent[0];
   result[9] = exponent[1];
   static const unsigned order[4] = {2, 2, 1, 0};
   for (unsigned t = 0; t < count; ++t)
      for (unsigned slot = 0; slot < 4; ++slot)
         for (unsigned axis = 0; axis < 3; ++axis) {
            double q = ((double)vertices[t][order[slot]][axis] - origin[axis]) /
                       agx_rt_bvh_gen1_axis_step(exponent[axis]);
            result[16 + t * 12 + slot * 3 + 2 - axis] =
               q <= 0 ? 0 : q >= 255 ? 255 : (uint8_t)q;
         }
   memcpy(packed, result, sizeof(result));
   return true;
}

#ifdef __cplusplus
}
#endif

#endif
