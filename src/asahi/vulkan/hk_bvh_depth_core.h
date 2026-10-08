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

#endif /* HK_BVH_DEPTH_CORE_H */
