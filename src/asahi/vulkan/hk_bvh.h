/*
 * Copyright © 2021 Google
 * Copyright © 2023 Valve Corporation
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Honeykrisp software-traversal BVH node format.
 *
 * This is a SOFTWARE format, traversed by shader ALU and global loads.  It is
 * deliberately NOT Apple's RIA BVH layout and it is deliberately not consumed
 * by the AGX3 `intersect_ray` node-test instruction:
 *
 *  - Apple's `intersect_ray` is a per-node test accelerator, not a traversal
 *    engine (the loop, the stack and commit are software either way), so a
 *    software traversal gives up far less than it sounds like.
 *  - The SCS slot layout that `intersect_ray` reads and writes is only
 *    partially mapped, and the canonical `intersect_ray` context encoded in
 *    agx3_rt_core.c matches no witnessed G17P ray-query site, so it is not a
 *    lowering template.
 *  - agx_rt_bvh.h's Gen5 layout carries a "validated over 64,000,000
 *    randomized headers" claim with no harness anywhere in the tree; it must
 *    not be adopted on that strength.
 *
 * So this format is ours to choose, and it is chosen to be byte-identical to
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
/* vk_build_helpers.h has already pulled in vk_bvh.h and the GLSL scalar
 * typedefs this header depends on. */
#else
#include "bvh/vk_bvh.h"
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
 * The traversal pushes onto a fixed HK_MAX_TLAS_DEPTH + HK_MAX_BLAS_DEPTH
 * array with no bounds check, exactly as lavapipe's does.  Lavapipe makes
 * that safe by detecting an over-deep tree during its CPU-side encode and
 * rebuilding the offending subtrees flatter.  Our encode is a GPU shader and
 * does no such pass, so a tree deeper than this is currently possible.
 *
 * It is not a memory-safety problem on this driver: the stack is a shader
 * temporary array indexed dynamically, and hk_lower_nir runs
 * nir_lower_indirect_derefs_to_if_else_trees, so an out-of-range push matches
 * no branch and is dropped rather than writing anywhere.  The consequence is
 * a silently missed subtree, i.e. missed intersections, not corruption.
 *
 * Closing it means either a depth-limiting pass in the encoder or a
 * stack-overflow restart in the traversal.  Until then, a scene degenerate
 * enough to build a 50-deep binary tree will return wrong hits.
 */

#endif /* HK_BVH_H */
