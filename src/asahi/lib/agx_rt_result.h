/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Shared Apple RIA3 traversal-result ABI recovered from the pinned runtime IR.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Exact IntersectRayResult byte offsets.  The two trailing transforms follow
 * the last i32 at the standard LLVM struct offsets (+60 and +108). */
#define AGX_RT_HIT_OFF_TYPE               0u
#define AGX_RT_HIT_OFF_T                  4u
#define AGX_RT_HIT_OFF_PRIMITIVE_ID       8u
#define AGX_RT_HIT_OFF_GEOMETRY_ID        12u
#define AGX_RT_HIT_OFF_PRIMITIVE_DATA     16u
#define AGX_RT_HIT_OFF_INSTANCE_ID        24u
#define AGX_RT_HIT_OFF_USER_INSTANCE_ID   28u
#define AGX_RT_HIT_OFF_INSTANCE_COUNT     32u
#define AGX_RT_HIT_OFF_BARYCENTRICS       40u
#define AGX_RT_HIT_OFF_FRONT_FACE         48u
#define AGX_RT_HIT_OFF_CURVE_PARAM        52u
#define AGX_RT_HIT_OFF_INTERSECTION_FN_ID 56u
#define AGX_RT_HIT_OFF_OBJECT_TO_WORLD    60u
#define AGX_RT_HIT_OFF_WORLD_TO_OBJECT    108u
#define AGX_RT_HIT_RESULT_SIZE            156u
#define AGX_RT_TRANSFORM_FLOATS            12u

/* Exact size of Apple's inline ray-query state object. */
#define AGX_RT_RAY_QUERY_STATE_SIZE 168u

/*
 * Apple's inline ray-query state layout (`intersection_query`).
 *
 * THIS IS A DIFFERENT ABI FROM THE `IntersectRayResult` OFFSETS ABOVE.  Those
 * describe the struct the `intersector` path returns; these describe the
 * inline `intersection_query` object that the ray-query path keeps.  A
 * Vulkan `rq_load` reads the latter.  The two do not agree on a single field
 * -- committed t is at 4 in IntersectRayResult and at 112 here -- so using
 * the IntersectRayResult offsets to lower a ray query produces a lowering
 * that looks right and reads the wrong words.
 *
 * Recovered from the RIA3 runtime (raytracing_runtime_ria3.metallib, shipped
 * in the Metal toolchain at metal/32023/lib/applegpu-nt/) and cross-checked
 * two independent ways that agree on every field below:
 *
 *  - the TBAA type node `_ZTS18intersection_query` in
 *    agx.intersection_query.reset, which gives member offsets and types, and
 *    whose stores bind reset's own arguments: min_distance -> 0,
 *    max_distance -> 4, origin[0..2] -> 12/16/20, direction[0..2] ->
 *    24/28/32.  The node's members end at 160 + 8, i.e. exactly the 168 above.
 *  - the individual `agx.get_committed_*_intersection_query` accessors, whose
 *    load offsets match the committed sub-record member-for-member.
 */
#define AGX_RT_QUERY_OFF_TMIN       0u
#define AGX_RT_QUERY_OFF_TMAX       4u
#define AGX_RT_QUERY_OFF_ORIGIN    12u /* 3 x f32 */
#define AGX_RT_QUERY_OFF_DIRECTION 24u /* 3 x f32 */
#define AGX_RT_QUERY_OFF_PARAMS    48u /* 10 bytes of intersection_params */
#define AGX_RT_QUERY_OFF_STATE     68u /* intersection_query::State */
#define AGX_RT_QUERY_OFF_CANDIDATE 88u /* ray_query_candidate, 20 bytes */
#define AGX_RT_QUERY_OFF_COMMITTED 112u

/* Committed-hit record, absolute offsets into the 168-byte query object.
 * Each is the load offset of the matching agx.get_committed_* accessor. */
#define AGX_RT_QUERY_OFF_COMMITTED_T              112u /* f32   */
#define AGX_RT_QUERY_OFF_COMMITTED_GEOMETRY_ID    120u /* i32   */
#define AGX_RT_QUERY_OFF_COMMITTED_PRIMITIVE_ID   124u /* i32   */
#define AGX_RT_QUERY_OFF_COMMITTED_INSTANCE_COUNT 128u /* i32   */
#define AGX_RT_QUERY_OFF_COMMITTED_TYPE           132u /* i8, hit_kind */
#define AGX_RT_QUERY_OFF_COMMITTED_FRONT_FACE     133u /* i8, bool     */
#define AGX_RT_QUERY_OFF_COMMITTED_BARYCENTRICS   136u /* 2 x f32 */
#define AGX_RT_QUERY_OFF_COMMITTED_CURVE_PARAM    144u /* f32   */
#define AGX_RT_QUERY_OFF_COMMITTED_PRIMITIVE_DATA 152u /* pointer */

/*
 * Apple's SCS ray staging segment -- the block the hardware intersect_ray
 * reads, distinct again from both ABIs above.
 *
 * Recovered from agx.intersection_query.next in the RIA3 runtime, whose
 * payload segment type is
 *
 *   <{ [3 x [2 x float]], float, i64, ... }>
 *
 * and whose stores bind each element by intersection_query field index:
 * pair[i][0] comes from field 3 (origin) and pair[i][1] from field 4
 * (direction), so the ray is interleaved PER AXIS rather than laid out as
 * origin.xyz then direction.xyz.  The trailing float comes from field 1,
 * tmax.
 *
 * This is cross-checked against native code: G17P ray-query setup emits
 * exactly two four-component SCS stores, at byte offsets 0 and 16, which is
 * precisely what this layout predicts -- bytes 0..15 carry
 * (origin.x, direction.x, origin.y, direction.y) and bytes 16..31 carry
 * (origin.z, direction.z, tmax, the low half of the packed word).
 *
 * tmin is NOT staged here.  next() reads intersection_query field 0
 * separately; it is not part of this segment.
 */
#define AGX_RT_SCS_RAY_AXIS_STRIDE   8u /* one (origin, direction) pair */
#define AGX_RT_SCS_RAY_ORIGIN(axis)    ((axis) * AGX_RT_SCS_RAY_AXIS_STRIDE)
#define AGX_RT_SCS_RAY_DIRECTION(axis)                                       \
   ((axis) * AGX_RT_SCS_RAY_AXIS_STRIDE + 4u)
#define AGX_RT_SCS_RAY_TMAX         24u
/* An i64 at 28 that next() read-modify-writes from several OR-ed values.  Its
 * contents are not mapped, so it is named only by position. */
#define AGX_RT_SCS_RAY_PACKED_WORD  28u
#define AGX_RT_SCS_RAY_SEGMENT_SIZE 36u

/*
 * SCS addressing and the rest of the map, measured against
 * `metal-nt -arch applegpu_g17p` over 27 purpose-written MSL kernels.
 *
 * stscs/ldscs address the SCS as (bank, byte offset), not as one flat space:
 * f70 is a bank selector with exactly five legal values 0..4 (5 and above are
 * refused), f71 is a contiguous 16-bit byte offset at instruction bits 77..92,
 * f68 is a 4-bit component mask (0 refused; Apple emits 1/3/15), and f10 is
 * the element size with 1 = 16-bit and 2 = 32-bit.
 *
 * THE BANK-AND-OFFSET ASSIGNMENT IS PER SHADER, NOT ARCHITECTURAL.
 * agc.scs.get_layout() returns a packed i64 that agx.scs.get_unpacked_layout
 * splits into five fields, and Apple's own layout printer inside
 * libapplegpu-nt.dylib carries the note "The layout above only applies to this
 * shader in isolation."  Measured consequence: the curve parameter lands at
 * (bank 2, 896) under <instancing, curve_data> while the barycentrics land at
 * (bank 3, 512) under <instancing, triangle_data>.  Across all 27 kernels the
 * ONLY invariant placement was bank 0 offset 0, the staged ray.  So none of
 * the offsets below may be treated as a fixed ISA constant; they are what one
 * layout produced, and a lowering must read the layout rather than assume.
 *
 * The read side, for the <instancing, triangle_data> layout:
 *
 *   bank 0 +24   the committed hit distance.  This is NOT a separate result
 *                field: it is the ray's own tmax slot read back.
 *   bank 0 +2560/+2562/+2564   instance-leaf pointer, three 16-bit reads
 *   bank 3 +256  (type << 30) | (primitive_id & 0x3fffffff)
 *   bank 3 +260  front_face in bit 31, geometry_id in bits 0..29
 *   bank 3 +512  float2 barycentrics
 *   bank 3 +263  the completion poll, read with the wait bit f65=1
 *                immediately after intersect_ray
 *   bank 4 +7    traversal status, read-modify-written around continue_ray
 *   bank 2 +896  curve parameter, in the curve_data layout only
 *
 * That +260 word is why an earlier differential saw geometry_id and
 * front_face "both add the same ldscs": they are two bitfields of one dword.
 *
 * intersect_ray RETURNS NOTHING -- agc.intersect.ray is declared void.  The
 * only read the hardware forces is the 4-byte poll at (bank 3, +263); every
 * other read above is an ordinary SCS load of an address the compiler
 * computed.
 *
 * The packed word at +28 is now mapped, from the payload segment's TBAA field
 * histogram cross-checked against the read/write arithmetic in
 * agx.intersect.ray, agx.intersection_query.next and agx.commit_*:
 *
 *   bits 2-3    woop axis index A          bits 4-5    woop axis index B
 *   bit  10     inline-query mode flag     bit  13     ray slot selector
 *   bits 18-25  instance mask (runtime)    bits 26-29  CoreId / clique
 *   bits 32-39  quantized motion time      bits 48-49  woop major axis
 *   bits 50-63  woop_info field 2
 *   bits 0-1, 6-9, 30-31 exist in the TBAA histogram but no RIA3 code writes
 *   them; 30-31 survive init_core's clear mask, which is the shape of a
 *   hardware writeback field.  Settling those needs hardware.
 *
 * There is NO node or acceleration-structure pointer in this word: the
 * acceleration structure goes to its own segment.
 *
 * CORRECTION to the segment size above: SEG0 is 64 bytes, not 36.  Its full
 * type is <{ [3 x [2 x float]], float, i64, {[3 x [2 x float]], i32} }> -- a
 * SECOND ray slot at bytes 36..59 with its own state word at 60..63, selected
 * by bit 13 of the packed word.  Apple's compiler only ever wrote slot A in
 * any of the 27 kernels, so slot B's use is unknown.
 */
#define AGX_RT_SCS_RAY_SLOT_B       36u
#define AGX_RT_SCS_SEGMENT0_SIZE    64u

/*
 * Deliberately absent: committed instance id and user instance id.  Their
 * accessors (agx.get_committed_instance_id / _user_instance_id) do not read a
 * single committed field -- they resolve through the candidate record at
 * 92/96/100/104 -- so no exact committed offset is established for them and
 * none is invented here.
 */

/* Exact TLAS instance-leaf offsets from the RIA3 module's TBAA metadata. */
#define AGX_RT_INSTANCE_LEAF_OFF_HEADER           0u
#define AGX_RT_INSTANCE_LEAF_OFF_OBJECT_TO_WORLD 12u
#define AGX_RT_INSTANCE_LEAF_OFF_AABB             60u
#define AGX_RT_INSTANCE_LEAF_OFF_IDS              84u
#define AGX_RT_INSTANCE_LEAF_OFF_INSTANCE_FLAGS  108u
#define AGX_RT_INSTANCE_LEAF_SIZE                124u

struct agx_rt_hit_result {
   uint32_t intersection_type;
   float t;
   uint32_t primitive_id;
   uint32_t geometry_id;
   uint64_t primitive_data;
   uint32_t instance_id;
   uint32_t user_instance_id;
   uint8_t instance_count;
   float barycentrics[2];
   uint8_t front_face;
   float curve_param;
   uint32_t intersection_function_id;
   float object_to_world[AGX_RT_TRANSFORM_FLOATS];
   float world_to_object[AGX_RT_TRANSFORM_FLOATS];
};

static inline uint32_t
agx_rt_read_u32_le(const void *base, uint32_t offset)
{
   const uint8_t *p = (const uint8_t *)base + offset;
   return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
          ((uint32_t)p[3] << 24);
}

static inline float
agx_rt_read_f32_le(const void *base, uint32_t offset)
{
   const uint32_t bits = agx_rt_read_u32_le(base, offset);
   float value;
   memcpy(&value, &bits, sizeof(value));
   return value;
}

static inline uint64_t
agx_rt_read_u64_le(const void *base, uint32_t offset)
{
   return (uint64_t)agx_rt_read_u32_le(base, offset) |
          ((uint64_t)agx_rt_read_u32_le(base, offset + 4) << 32);
}

static inline void
agx_rt_read_transform(const void *base, uint32_t offset,
                      float out[AGX_RT_TRANSFORM_FLOATS])
{
   for (uint32_t i = 0; i < AGX_RT_TRANSFORM_FLOATS; ++i)
      out[i] = agx_rt_read_f32_le(base, offset + i * 4u);
}

/* Decode the exact public result prefix.  The two transforms are decoded only
 * when the complete 156-byte object is present and otherwise remain zero. */
static inline bool
agx_rt_hit_result_decode(const void *raw, size_t available,
                         struct agx_rt_hit_result *out)
{
   if (!raw || !out || available < AGX_RT_HIT_OFF_INTERSECTION_FN_ID + 4u)
      return false;

   memset(out, 0, sizeof(*out));
   out->intersection_type = agx_rt_read_u32_le(raw, AGX_RT_HIT_OFF_TYPE);
   out->t = agx_rt_read_f32_le(raw, AGX_RT_HIT_OFF_T);
   out->primitive_id = agx_rt_read_u32_le(raw, AGX_RT_HIT_OFF_PRIMITIVE_ID);
   out->geometry_id = agx_rt_read_u32_le(raw, AGX_RT_HIT_OFF_GEOMETRY_ID);
   out->primitive_data = agx_rt_read_u64_le(raw, AGX_RT_HIT_OFF_PRIMITIVE_DATA);
   out->instance_id = agx_rt_read_u32_le(raw, AGX_RT_HIT_OFF_INSTANCE_ID);
   out->user_instance_id =
      agx_rt_read_u32_le(raw, AGX_RT_HIT_OFF_USER_INSTANCE_ID);
   out->instance_count = ((const uint8_t *)raw)[AGX_RT_HIT_OFF_INSTANCE_COUNT];
   out->barycentrics[0] =
      agx_rt_read_f32_le(raw, AGX_RT_HIT_OFF_BARYCENTRICS);
   out->barycentrics[1] =
      agx_rt_read_f32_le(raw, AGX_RT_HIT_OFF_BARYCENTRICS + 4u);
   out->front_face = ((const uint8_t *)raw)[AGX_RT_HIT_OFF_FRONT_FACE];
   out->curve_param = agx_rt_read_f32_le(raw, AGX_RT_HIT_OFF_CURVE_PARAM);
   out->intersection_function_id =
      agx_rt_read_u32_le(raw, AGX_RT_HIT_OFF_INTERSECTION_FN_ID);

   if (available >= AGX_RT_HIT_RESULT_SIZE) {
      agx_rt_read_transform(raw, AGX_RT_HIT_OFF_OBJECT_TO_WORLD,
                            out->object_to_world);
      agx_rt_read_transform(raw, AGX_RT_HIT_OFF_WORLD_TO_OBJECT,
                            out->world_to_object);
   }

   return true;
}

static inline bool
agx_rt_instance_leaf_object_to_world(const void *raw, size_t available,
                                     float out[AGX_RT_TRANSFORM_FLOATS])
{
   if (!raw || !out || available < AGX_RT_INSTANCE_LEAF_OFF_AABB)
      return false;

   agx_rt_read_transform(raw, AGX_RT_INSTANCE_LEAF_OFF_OBJECT_TO_WORLD, out);
   return true;
}

#ifdef __cplusplus
}
#endif
