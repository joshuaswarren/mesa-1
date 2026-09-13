/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_bvh_depth_core.h"

#include "util/macros.h"

#include <assert.h>
#include <string.h>

bool
hk_bvh_internal_child(uint32_t child_ref, uint32_t *index)
{
   if (child_ref == HK_BVH_INVALID_NODE)
      return false;

   if ((child_ref & 3u) != hk_bvh_node_internal)
      return false;

   /* Inverse of encode_internal(): a box node lives at
    * sizeof(hk_bvh_header) + dst_index * sizeof(hk_bvh_box_node). */
   uint32_t offset = child_ref & ~3u;
   if (offset < sizeof(struct hk_bvh_header))
      return false;

   *index = (offset - (uint32_t)sizeof(struct hk_bvh_header)) /
            (uint32_t)sizeof(struct hk_bvh_box_node);
   return true;
}

uint32_t
hk_bvh_compute_depths(const struct hk_bvh_box_node *nodes,
                      uint32_t internal_count, uint32_t *depth)
{
   if (!internal_count)
      return 0;

   memset(depth, 0, internal_count * sizeof(*depth));

   uint32_t max_depth = 0;

   /* Forward sweep. Every child has a larger index than its parent, so each
    * node's depth is final by the time the sweep reaches it. */
   for (uint32_t i = 0; i < internal_count; i++) {
      for (uint32_t c = 0; c < 2; c++) {
         uint32_t child;
         if (!hk_bvh_internal_child(nodes[i].children[c], &child))
            continue;

         if (child <= i || child >= internal_count)
            continue; /* malformed: refuse to loop rather than trust it */

         depth[child] = depth[i] + 1;
         max_depth = MAX2(max_depth, depth[child]);
      }
   }

   return max_depth;
}

void
hk_bvh_compute_subtree_info(const struct hk_bvh_box_node *nodes,
                            uint32_t internal_count, uint32_t *height,
                            uint32_t *leaves)
{
   if (!internal_count)
      return;

   if (height)
      memset(height, 0, internal_count * sizeof(*height));
   if (leaves)
      memset(leaves, 0, internal_count * sizeof(*leaves));

   /* Backward sweep, the mirror of the forward one: every child has a larger
    * index, so all of a node's children are final before it is reached. */
   for (uint32_t n = internal_count; n-- > 0;) {
      for (uint32_t c = 0; c < 2; c++) {
         uint32_t ref = nodes[n].children[c];
         uint32_t child;

         if (hk_bvh_internal_child(ref, &child)) {
            if (child <= n || child >= internal_count)
               continue;

            if (height)
               height[n] = MAX2(height[n], height[child] + 1);
            if (leaves)
               leaves[n] += leaves[child];
         } else if (ref != HK_BVH_INVALID_NODE) {
            if (leaves)
               leaves[n] += 1;
         }
      }
   }
}

/* Smallest h with (1 << h) >= k, i.e. the height of a balanced tree over k
 * leaves. */
static uint32_t
ceil_log2(uint32_t k)
{
   uint32_t h = 0;
   while ((1u << h) < k)
      h++;
   return h;
}

uint32_t
hk_bvh_select_flatten_roots(const struct hk_bvh_box_node *nodes,
                            uint32_t internal_count, const uint32_t *depth,
                            const uint32_t *height, const uint32_t *leaves,
                            uint32_t stack_slots, uint32_t *roots,
                            uint32_t *stack, bool *feasible)
{
   *feasible = true;

   if (!internal_count || !stack_slots)
      return 0;

   uint32_t count = 0;
   uint32_t sp = 0;
   stack[sp++] = 0; /* the root */

   while (sp) {
      uint32_t i = stack[--sp];

      /* Already within budget: this whole subtree is fine. */
      if (depth[i] + height[i] + 1 <= stack_slots)
         continue;

      /* Balanced is the shallowest this subtree can possibly be. If that
       * fits, rebuilding here is both sufficient and the shallowest place
       * that is sufficient, so stop descending. */
      if (depth[i] + ceil_log2(leaves[i]) + 1 <= stack_slots) {
         roots[count++] = i;
         continue;
      }

      /* Cannot be fixed at i. The children hold strictly fewer leaves, so
       * try them; if none are internal, nothing can be done here. */
      bool descended = false;
      for (uint32_t c = 0; c < 2; c++) {
         uint32_t child;
         if (!hk_bvh_internal_child(nodes[i].children[c], &child))
            continue;
         if (child <= i || child >= internal_count)
            continue;

         stack[sp++] = child;
         descended = true;
      }

      if (!descended)
         *feasible = false;
   }

   return count;
}

bool
hk_bvh_gather_subtree(const struct hk_bvh_box_node *nodes,
                      uint32_t internal_count, uint32_t root, uint32_t capacity,
                      uint32_t *leaf_refs, vk_aabb *leaf_bounds,
                      uint32_t *leaf_count, uint32_t *slots,
                      uint32_t *slot_count, uint32_t *stack, uint8_t *mark)
{
   memset(mark, 0, internal_count);

   uint32_t sp = 0;
   stack[sp++] = root;
   mark[root] = 1;

   while (sp) {
      uint32_t i = stack[--sp];

      for (uint32_t c = 0; c < 2; c++) {
         uint32_t child;
         if (!hk_bvh_internal_child(nodes[i].children[c], &child))
            continue;
         if (child <= i || child >= internal_count)
            continue;

         if (sp >= capacity)
            return false;

         mark[child] = 1;
         stack[sp++] = child;
      }
   }

   /*
    * Collect in ascending index order, not in the order the walk found them.
    *
    * hk_bvh_rebuild_balanced() hands these slots out to parents before
    * children, so ascending slots are what keep a rebuilt child's index above
    * its parent's -- the parent-before-child property the whole file depends
    * on.  Handing out slots in walk order silently inverts it for some nodes
    * and the depth sweep then reads the rebuilt tree wrong.
    *
    * A scan is also what the GPU port wants here: sorting the gathered set
    * would be the only sort in the algorithm.
    */
   uint32_t leaves = 0;
   uint32_t n_slots = 0;

   for (uint32_t i = root; i < internal_count; i++) {
      if (!mark[i])
         continue;

      if (n_slots >= capacity)
         return false;
      slots[n_slots++] = i;

      for (uint32_t c = 0; c < 2; c++) {
         uint32_t ref = nodes[i].children[c];
         uint32_t child;

         if (hk_bvh_internal_child(ref, &child))
            continue;
         if (ref == HK_BVH_INVALID_NODE)
            continue;

         if (leaves >= capacity)
            return false;

         leaf_refs[leaves] = ref;
         leaf_bounds[leaves] = nodes[i].bounds[c];
         leaves++;
      }
   }

   *leaf_count = leaves;
   *slot_count = n_slots;
   return true;
}

static vk_aabb
aabb_union(vk_aabb a, vk_aabb b)
{
   vk_aabb r;
   r.min.x = MIN2(a.min.x, b.min.x);
   r.min.y = MIN2(a.min.y, b.min.y);
   r.min.z = MIN2(a.min.z, b.min.z);
   r.max.x = MAX2(a.max.x, b.max.x);
   r.max.y = MAX2(a.max.y, b.max.y);
   r.max.z = MAX2(a.max.z, b.max.z);
   return r;
}

static vk_aabb
range_bounds(const vk_aabb *bounds, uint32_t begin, uint32_t end)
{
   vk_aabb r = bounds[begin];
   for (uint32_t i = begin + 1; i < end; i++)
      r = aabb_union(r, bounds[i]);
   return r;
}

static uint32_t
node_ref(uint32_t slot)
{
   return ((uint32_t)sizeof(struct hk_bvh_header) +
           slot * (uint32_t)sizeof(struct hk_bvh_box_node)) |
          hk_bvh_node_internal;
}

void
hk_bvh_rebuild_balanced(struct hk_bvh_box_node *nodes, const uint32_t *slots,
                        uint32_t slot_count, const uint32_t *leaf_refs,
                        const vk_aabb *leaf_bounds, uint32_t leaf_count,
                        uint32_t *stack)
{
   if (!leaf_count || !slot_count)
      return;

   /* A single leaf cannot occupy a box node on its own; the caller only ever
    * selects subtrees with an internal child, so this is defensive. */
   if (leaf_count == 1) {
      nodes[slots[0]].children[0] = leaf_refs[0];
      nodes[slots[0]].bounds[0] = leaf_bounds[0];
      nodes[slots[0]].children[1] = HK_BVH_INVALID_NODE;
      return;
   }

   /* Work items are (slot, leaf_begin, leaf_end) triples on the caller's
    * stack, so the split is iterative. next_slot hands out the remaining
    * gathered slots; a k-leaf balanced tree needs k - 1 of them and the
    * gather always returns at least that many. */
   uint32_t next_slot = 1;
   uint32_t sp = 0;

   stack[sp++] = slots[0];
   stack[sp++] = 0;
   stack[sp++] = leaf_count;

   while (sp) {
      uint32_t end = stack[--sp];
      uint32_t begin = stack[--sp];
      uint32_t slot = stack[--sp];

      uint32_t mid = begin + (end - begin) / 2;
      const uint32_t halves[2][2] = {{begin, mid}, {mid, end}};

      for (uint32_t c = 0; c < 2; c++) {
         uint32_t lo = halves[c][0], hi = halves[c][1];

         if (hi - lo == 1) {
            nodes[slot].children[c] = leaf_refs[lo];
            nodes[slot].bounds[c] = leaf_bounds[lo];
            continue;
         }

         uint32_t child_slot = slots[next_slot++];
         nodes[slot].children[c] = node_ref(child_slot);
         nodes[slot].bounds[c] = range_bounds(leaf_bounds, lo, hi);

         stack[sp++] = child_slot;
         stack[sp++] = lo;
         stack[sp++] = hi;
      }
   }

   assert(next_slot <= slot_count);
}
