/*
 * Copyright © 2021 Google
 * Copyright © 2023 Valve Corporation
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Honeykrisp software-traversal BVH node format.
 *
 * The current builder produces this software format for shader ALU traversal.
 * It cannot be passed to M5's native intersect_ray/continue_ray path. Native
 * M5 uses the Gen1 node layout for BVH version 3 and an SCS query protocol;
 * the complete acceleration-structure envelope still needs integrating.
 * The byte encoders and their regression fixtures live in agx_rt_bvh.h and
 * tests/test-rt-bvh.c. Hardware traversal must use that native format.
 *
 * This software layout is byte-identical to
 * lavapipe's (src/gallium/frontends/lavapipe/lvp_acceleration_structure.h),
 * because the NIR traversal in hk_nir_ray_tracing.c is a port of lavapipe's
 * and keeping the layouts identical keeps that port auditable field by field.
 *
 * A node reference is `byte_offset_from_bvh_base | type` with the type in the
 * low 2 bits, so every node must be at least 4-byte aligned.
 */

#ifndef HK_BVH_H
#define HK_BVH_H

#ifdef VULKAN
/* vk_bvh_helpers.h has already pulled in vk_bvh_defines.h and the GLSL scalar
 * typedefs this header depends on. */
#else
#include "bvh/vk_bvh_defines.h"
#endif

/* Geometry flags, packed into the high bits of geometry_id_and_flags. */
#define HK_GEOMETRY_OPAQUE (1u << 31)

/* Instance flags, packed into the high bits of sbt_offset_and_flags.  The
 * top two are adjacent on purpose: an unsigned compare against
 * HK_INSTANCE_NO_FORCE_NOT_OPAQUE decides opacity in one instruction. */
#define HK_INSTANCE_FORCE_OPAQUE                 (1u << 31)
#define HK_INSTANCE_NO_FORCE_NOT_OPAQUE          (1u << 30)
#define HK_INSTANCE_TRIANGLE_FACING_CULL_DISABLE (1u << 29)
#define HK_INSTANCE_TRIANGLE_FLIP_FACING         (1u << 28)

#define hk_bvh_node_triangle 0
#define hk_bvh_node_internal 1
#define hk_bvh_node_instance 2
#define hk_bvh_node_aabb     3

/* 48 bytes */
struct hk_bvh_triangle_node {
   float coords[3][3];

   uint32_t padding;

   uint32_t primitive_id;
   /* flags in the upper 4 bits */
   uint32_t geometry_id_and_flags;
};

/* 32 bytes */
struct hk_bvh_aabb_node {
   vk_aabb bounds;

   uint32_t primitive_id;
   /* flags in the upper 4 bits */
   uint32_t geometry_id_and_flags;
};

/* 120 bytes */
struct hk_bvh_instance_node {
   uint64_t bvh_ptr;

   /* lower 24 bits are the custom instance index, upper 8 the visibility mask */
   uint32_t custom_instance_and_mask;
   /* lower 24 bits are the sbt offset, upper 8 VkGeometryInstanceFlagsKHR */
   uint32_t sbt_offset_and_flags;

   mat3x4 wto_matrix;
   uint32_t padding;

   uint32_t instance_id;

   /* Object-to-world matrix, transposed from the initial transform. */
   mat3x4 otw_matrix;
};

/* 56 bytes */
struct hk_bvh_box_node {
   vk_aabb bounds[2];
   uint32_t children[2];
};

/*
 * The traversal prefetches a whole node unconditionally before it knows the
 * node type, so the allocation is padded to a multiple of this and every node
 * type must be no larger than a box node.
 */
#define HK_BVH_NODE_PREFETCH_SIZE 56

struct hk_bvh_header {
   vk_aabb bounds;

   uint32_t compacted_size;
   uint32_t serialization_size;
   uint32_t instance_count;
   uint32_t leaf_nodes_offset;
};

/* The root node is the first node after the header. */
#define HK_BVH_ROOT_NODE_OFFSET (sizeof(struct hk_bvh_header))
#define HK_BVH_ROOT_NODE        (HK_BVH_ROOT_NODE_OFFSET | hk_bvh_node_internal)
#define HK_BVH_INVALID_NODE     0xFFFFFFFF

/*
 * Traversal stack depth.  A ray query keeps this many uints live in shader
 * temporaries, so the numbers directly set register pressure: G17P has a
 * known scratch/spill gap, and these are the first knob to turn if the
 * traversal spills badly.  They bound how deep the encoder is allowed to make
 * the tree, not just how deep traversal may go.
 */
#define HK_MAX_TLAS_DEPTH 24
#define HK_MAX_BLAS_DEPTH 29

/*
 * KNOWN GAP -- deeper trees are not yet rejected or flattened.
 *
 * The encoder does not cap tree depth, so a tree deeper than
 * HK_MAX_TLAS_DEPTH + HK_MAX_BLAS_DEPTH is possible.  Lavapipe avoids this by
 * measuring depth during its CPU-side encode and rebuilding the offending
 * subtrees flatter (lvp_flatten_as); our encode is a GPU shader and has no
 * such pass.
 *
 * This is reachable, not theoretical.  The common builder picks PLOC for
 * every TLAS above four instances and for most BLASes
 * (vk_acceleration_structure.c, vk_acceleration_structure_build_state_init),
 * and PLOC is agglomerative: nothing in it bounds the depth of the tree it
 * merges together.
 *
 * The consequence is bounded but wrong.  hk_build_push_stack() refuses to
 * push past the end, so an over-deep tree loses the subtree it could not
 * defer and returns missed intersections.  It does not corrupt memory and it
 * does not read an undefined node -- but it does silently return the wrong
 * answer, which is the part still to fix.
 *
 * CLOSING IT.  The tractable route is a depth pass in the encoder, and the
 * fact that makes it cheap is easy to miss, so it is written down here.
 *
 * encode_internal() maps IR index to output index as
 *
 *    dst_index = internal_count - src_index - 1
 *
 * over an IR internal array that is root-last, with every child allocated
 * before its parent.  A parent therefore has a *larger* IR index and hence a
 * *smaller* output index than either of its children:
 *
 *    output index of parent  <  output index of both children
 *
 * Output order is already a topological order.  So depth needs no sort, no
 * atomics and no second traversal: sweep output indices in increasing order,
 * seed depth[0] = 0 for the root, and when visiting node i write
 * depth[child] = depth[i] + 1 for each internal child.  Every child is
 * reached after its parent is final, so one linear pass suffices.
 *
 * What remains after that is the flatten itself, which is the expensive half:
 * lavapipe gathers each offending subtree's leaves and re-links them into a
 * balanced tree over the internal node slots that subtree already owns (so it
 * needs no extra allocation).  Porting that to a compute shader means doing
 * without recursion and without per-subtree dynamic arrays, and it is the
 * reason this gap is still open rather than the depth measurement.
 */

#endif /* HK_BVH_H */
