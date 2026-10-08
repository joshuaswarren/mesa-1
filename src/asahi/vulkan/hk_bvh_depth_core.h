/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Depth analysis and depth-limiting for an encoded Honeykrisp BVH.
 *
 * The traversal stack is a fixed-size shader array (hk_bvh.h), so a tree
 * deeper than it holds loses subtrees and returns missed intersections.  This
 * is the logic that measures the depth and decides what to rebuild; it lives
 * here, separate from the compute shader that will run it, because the shader
 * cannot be unit tested and this can.
 *
 * Everything operates on the *encoded* output nodes rather than the common
 * builder's IR, for one reason: the output index order is a topological order.
 * encode_internal() maps IR index to output index as
 *
 *    dst_index = internal_count - src_index - 1
 *
 * over an IR internal array that is root-last, with every child allocated
 * before its parent, so a parent has a larger IR index and hence a *smaller*
 * output index than either child:
 *
 *    output index of parent  <  output index of both children
 *
 * Depth therefore falls out of a single forward sweep and height/leaf counts
 * out of a single backward sweep -- no sort, no atomics, no tree walk.  That
 * property is what makes a GPU port tractable at all, so it is asserted by the
 * unit tests rather than assumed.
 *
 * All scratch is caller-provided so the GLSL port can hand it a scratch
 * buffer, and no function here recurses.
 */

#ifndef HK_BVH_DEPTH_CORE_H
#define HK_BVH_DEPTH_CORE_H

#include "hk_bvh.h"

#include <stdbool.h>
#include <stdint.h>

/*
 * Decode a box node child reference into an internal node index.
 *
 * Returns false when the child is absent or is a leaf, i.e. when there is no
 * internal node to descend into.
 */
bool hk_bvh_internal_child(uint32_t child_ref, uint32_t *index);

/*
 * depth[i] = number of internal edges from the root to internal node i, so the
 * root is 0.  Returns the maximum over the tree.
 *
 * depth must have internal_count entries.  Nodes unreachable from the root
 * keep depth 0; a well-formed encode produces none.
 */
uint32_t hk_bvh_compute_depths(const struct hk_bvh_box_node *nodes,
                               uint32_t internal_count, uint32_t *depth);

/*
 * height[i] = internal edges from i down to its deepest internal descendant,
 * so a node whose children are all leaves is 0.
 * leaves[i] = number of leaf children in the subtree rooted at i.
 *
 * Both are filled by one backward sweep.  Either output may be NULL.
 */
void hk_bvh_compute_subtree_info(const struct hk_bvh_box_node *nodes,
                                 uint32_t internal_count, uint32_t *height,
                                 uint32_t *leaves);

/*
 * Choose the subtree roots to rebuild so the tree fits `stack_slots`.
 *
 * Walks down from the root and, for each subtree that does not fit, marks the
 * shallowest node where rebuilding balanced *would* fit.  Rebuilding balanced
 * is the minimum achievable height for a given leaf count, so a subtree that
 * still does not fit when balanced cannot be fixed at that node and the walk
 * continues into its children, where the leaf counts are smaller.
 *
 * roots must have room for internal_count entries; stack likewise.  Returns
 * the number of roots written, and sets *feasible to false if some branch
 * cannot be made to fit at any node (only reachable with more leaves than
 * 2^(stack_slots-1), which the encoder should have refused long before).
 */
uint32_t hk_bvh_select_flatten_roots(const struct hk_bvh_box_node *nodes,
                                     uint32_t internal_count,
                                     const uint32_t *depth,
                                     const uint32_t *height,
                                     const uint32_t *leaves,
                                     uint32_t stack_slots, uint32_t *roots,
                                     uint32_t *stack, bool *feasible);

/*
 * Collect every leaf reference and every internal slot in the subtree rooted
 * at `root`.  The slots come back so the rebuild can reuse exactly the storage
 * the subtree already owns -- a binary tree with k leaves has at least k - 1
 * internal nodes, so there is always enough and no allocation is needed.
 *
 * Slots come back in ascending index order, which is what keeps a rebuilt
 * child's index above its parent's.  leaf_refs/leaf_bounds/slots/stack must
 * each have room for `capacity` entries and mark for internal_count; returns
 * false if the subtree does not fit (malformed).
 */
bool hk_bvh_gather_subtree(const struct hk_bvh_box_node *nodes,
                           uint32_t internal_count, uint32_t root,
                           uint32_t capacity, uint32_t *leaf_refs,
                           vk_aabb *leaf_bounds, uint32_t *leaf_count,
                           uint32_t *slots, uint32_t *slot_count,
                           uint32_t *stack, uint8_t *mark);

/*
 * Re-link the gathered leaves into a balanced tree over the gathered slots.
 * slots[0] must be the subtree root, so the parent that points at it stays
 * correct.  Slots beyond the k - 1 needed are left unreferenced.
 */
void hk_bvh_rebuild_balanced(struct hk_bvh_box_node *nodes,
                             const uint32_t *slots, uint32_t slot_count,
                             const uint32_t *leaf_refs,
                             const vk_aabb *leaf_bounds, uint32_t leaf_count,
                             uint32_t *stack);

#endif /* HK_BVH_DEPTH_CORE_H */
