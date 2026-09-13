/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * The depth analysis in hk_bvh_depth_core.c exists to stop the traversal
 * stack overflowing, and it will eventually run as a compute shader where
 * none of it can be observed.  So the interesting properties are checked
 * here, against trees built independently of the code under test:
 *
 *  - the depth sweep agrees with a recursive walk;
 *  - the stack requirement it predicts is the peak occupancy of an actual
 *    simulation of the traversal's push/pop discipline, which is the only
 *    reason any of this matters;
 *  - flattening produces a tree that fits the budget, keeps every leaf, and
 *    preserves the parent-before-child index order the sweep depends on.
 *
 * Every case runs at several budgets rather than only the real one.  A rule
 * that accidentally hardcodes HK_MAX_TLAS_DEPTH passes at 24 and nowhere
 * else, and that is precisely the regression worth catching.
 */

#include "hk_bvh_depth_core.h"

#include "util/macros.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_NODES 512

struct tree {
   struct hk_bvh_box_node nodes[MAX_NODES];
   uint32_t internal_count;
   uint32_t next_leaf;
};

static uint32_t
leaf_ref(struct tree *t)
{
   /* Distinct, non-zero, and typed as a triangle so it is never mistaken for
    * an internal reference. */
   return ((t->next_leaf++ + 1) * 64u) | hk_bvh_node_triangle;
}

static uint32_t
internal_ref(uint32_t slot)
{
   return ((uint32_t)sizeof(struct hk_bvh_header) +
           slot * (uint32_t)sizeof(struct hk_bvh_box_node)) |
          hk_bvh_node_internal;
}

static void
set_bounds(struct hk_bvh_box_node *n, uint32_t c, float v)
{
   n->bounds[c].min.x = n->bounds[c].min.y = n->bounds[c].min.z = v;
   n->bounds[c].max.x = n->bounds[c].max.y = n->bounds[c].max.z = v + 1.0f;
}

/* A chain: node i holds one leaf and descends into node i + 1. Depth = n - 1. */
static void
build_chain(struct tree *t, uint32_t n)
{
   memset(t, 0, sizeof(*t));
   t->internal_count = n;

   for (uint32_t i = 0; i < n; i++) {
      t->nodes[i].children[0] = leaf_ref(t);
      set_bounds(&t->nodes[i], 0, (float)i);

      if (i + 1 < n)
         t->nodes[i].children[1] = internal_ref(i + 1);
      else
         t->nodes[i].children[1] = leaf_ref(t);

      set_bounds(&t->nodes[i], 1, (float)i + 0.5f);
   }
}

/* A perfect tree in heap order: node i has children 2i+1 and 2i+2, so the
 * parent index is always below both children. 2^h leaves, depth h - 1. */
static void
build_balanced(struct tree *t, uint32_t h)
{
   memset(t, 0, sizeof(*t));
   uint32_t n = (1u << h) - 1;
   t->internal_count = n;

   for (uint32_t i = 0; i < n; i++) {
      for (uint32_t c = 0; c < 2; c++) {
         uint32_t child = 2 * i + 1 + c;

         t->nodes[i].children[c] =
            child < n ? internal_ref(child) : leaf_ref(t);
         set_bounds(&t->nodes[i], c, (float)(i * 2 + c));
      }
   }
}

/*
 * A random tree. Nodes are handed out from a queue as they are created, so a
 * parent is always allocated before its children and keeps the lower index --
 * the same order the encoder produces.
 */
static void
build_random(struct tree *t, uint32_t n, unsigned *seed)
{
   memset(t, 0, sizeof(*t));
   t->internal_count = n;

   uint32_t next = 1;
   for (uint32_t i = 0; i < n; i++) {
      for (uint32_t c = 0; c < 2; c++) {
         /* next > i is not cosmetic: without it a node whose predecessors
          * took no internal children gets handed its own index and points at
          * itself, which hangs every walk in this file. */
         bool want_internal =
            next < n && next > i && (rand_r(seed) % 100) < 65;

         t->nodes[i].children[c] =
            want_internal ? internal_ref(next++) : leaf_ref(t);
         set_bounds(&t->nodes[i], c, (float)(i * 2 + c));
      }
   }

   /* Any node never handed out is unreachable; trim so counts stay honest. */
   t->internal_count = next;
}

/* Independent recursive depth, deliberately not the sweep under test. */
static void
ref_depths(const struct tree *t, uint32_t i, uint32_t d, uint32_t *out,
           uint32_t *max_depth)
{
   out[i] = d;
   *max_depth = MAX2(*max_depth, d);

   for (uint32_t c = 0; c < 2; c++) {
      uint32_t child;
      if (hk_bvh_internal_child(t->nodes[i].children[c], &child) &&
          child < t->internal_count)
         ref_depths(t, child, d + 1, out, max_depth);
   }
}

/*
 * Independent subtree height, used only to pick the worst-case ray below.
 *
 * The recursion is hoisted out of MAX2 deliberately: MAX2 is a macro that
 * evaluates both arguments twice, so calling it inline recurses twice per
 * child and costs 2^depth on a chain.
 */
static uint32_t
local_height(const struct tree *t, uint32_t i)
{
   uint32_t h = 0;
   for (uint32_t c = 0; c < 2; c++) {
      uint32_t child;
      if (hk_bvh_internal_child(t->nodes[i].children[c], &child) &&
          child < t->internal_count) {
         uint32_t child_height = local_height(t, child) + 1;
         h = MAX2(h, child_height);
      }
   }
   return h;
}

/*
 * Simulate hk_build_ray_traversal()'s stack discipline: descend into one
 * child, defer the other, pop when the current node runs out.
 *
 * Which child it descends into is decided per ray by the box test, so peak
 * occupancy is a property of the ray, not of the tree alone -- a ray that
 * takes the shallow child at every step retires it immediately and never
 * stacks anything up.  The stack has to be sized for the worst ray, so this
 * drives the worst one: always descend the taller child, leaving the other
 * outstanding at every level of the deepest path.
 */
static uint32_t
simulate_worst_peak_stack(const struct tree *t)
{
   uint32_t stack[MAX_NODES * 2];
   uint32_t sp = 0, peak = 0;
   uint32_t current = internal_ref(0);

   /* A malformed tree must fail the test, never hang the suite. Every node is
    * visited once in a well-formed tree, so this can only trip on a cycle. */
   uint32_t fuel = MAX_NODES * 8;

   for (;;) {
      if (fuel-- == 0)
         return UINT32_MAX;

      if (current == HK_BVH_INVALID_NODE) {
         if (!sp)
            break;
         current = stack[--sp];
      }

      uint32_t node = current;
      current = HK_BVH_INVALID_NODE;

      uint32_t index;
      if (!hk_bvh_internal_child(node, &index) || index >= t->internal_count)
         continue; /* a leaf: the traversal tests it and moves on */

      uint32_t child_ref[2] = {t->nodes[index].children[0],
                               t->nodes[index].children[1]};

      /* Order so child_ref[0] is the one the worst-case ray descends into. */
      int h[2];
      for (uint32_t c = 0; c < 2; c++) {
         uint32_t ci;
         h[c] = (hk_bvh_internal_child(child_ref[c], &ci) &&
                 ci < t->internal_count)
                   ? (int)local_height(t, ci)
                   : -1;
      }
      if (h[1] > h[0]) {
         uint32_t tmp = child_ref[0];
         child_ref[0] = child_ref[1];
         child_ref[1] = tmp;
      }

      if (child_ref[0] != HK_BVH_INVALID_NODE)
         current = child_ref[0];

      if (child_ref[1] != HK_BVH_INVALID_NODE) {
         if (current == HK_BVH_INVALID_NODE) {
            current = child_ref[1];
         } else {
            stack[sp++] = child_ref[1];
            peak = MAX2(peak, sp);
         }
      }
   }

   return peak;
}

static void
collect_leaves(const struct tree *t, uint32_t *out, uint32_t *count)
{
   *count = 0;
   for (uint32_t i = 0; i < t->internal_count; i++) {
      for (uint32_t c = 0; c < 2; c++) {
         uint32_t ref = t->nodes[i].children[c], child;
         if (ref == HK_BVH_INVALID_NODE)
            continue;
         if (hk_bvh_internal_child(ref, &child) && child < t->internal_count)
            continue;
         out[(*count)++] = ref;
      }
   }
}

static int
cmp_u32(const void *a, const void *b)
{
   uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
   return (x > y) - (x < y);
}

static bool
parent_before_child(const struct tree *t)
{
   for (uint32_t i = 0; i < t->internal_count; i++) {
      for (uint32_t c = 0; c < 2; c++) {
         uint32_t child;
         if (!hk_bvh_internal_child(t->nodes[i].children[c], &child))
            continue;
         if (child >= t->internal_count)
            continue;
         if (child <= i)
            return false;
      }
   }
   return true;
}

#define FAIL(...)                                                              \
   do {                                                                        \
      fprintf(stderr, "%s: ", name);                                           \
      fprintf(stderr, __VA_ARGS__);                                            \
      fprintf(stderr, "\n");                                                   \
      return false;                                                            \
   } while (0)

static bool
check_tree(const char *name, struct tree *t, uint32_t budget)
{
   if (!t->internal_count)
      return true;

   if (!parent_before_child(t))
      FAIL("the test's own builder broke parent-before-child");

   uint32_t depth[MAX_NODES], height[MAX_NODES], leaves[MAX_NODES];

   /* 1. The sweep agrees with a recursive walk. */
   uint32_t max_depth = hk_bvh_compute_depths(t->nodes, t->internal_count, depth);

   uint32_t ref[MAX_NODES] = {0}, ref_max = 0;
   ref_depths(t, 0, 0, ref, &ref_max);

   if (max_depth != ref_max)
      FAIL("max depth %u, recursive walk says %u", max_depth, ref_max);

   for (uint32_t i = 0; i < t->internal_count; i++) {
      if (depth[i] != ref[i])
         FAIL("depth[%u] = %u, recursive walk says %u", i, depth[i], ref[i]);
   }

   /* 2. The predicted stack requirement is the simulated peak. This is the
    *    claim the whole file rests on. */
   uint32_t predicted = hk_bvh_stack_slots_required(max_depth);
   uint32_t peak = simulate_worst_peak_stack(t);

   /* Safety: the array must be big enough for the worst ray. */
   if (peak > predicted)
      FAIL("worst ray needs %u stack slots, sizing predicts only %u", peak,
           predicted);

   /* Tightness: every node here has two children, so the worst ray defers at
    * every level of the deepest path and the bound is exactly met.  Without
    * this a sizing rule that simply overestimated would pass the line above. */
   if (peak != predicted)
      FAIL("predicted %u stack slots, worst ray peaks at %u", predicted, peak);

   hk_bvh_compute_subtree_info(t->nodes, t->internal_count, height, leaves);

   if (depth[0] + height[0] != max_depth)
      FAIL("root height %u inconsistent with max depth %u", height[0],
           max_depth);

   uint32_t total_leaves = 0;
   uint32_t before[MAX_NODES * 2];
   collect_leaves(t, before, &total_leaves);
   if (leaves[0] != total_leaves)
      FAIL("root leaf count %u, walk finds %u", leaves[0], total_leaves);

   /* 3. Flatten to the budget, then re-measure. */
   uint32_t roots[MAX_NODES], stack[MAX_NODES * 4];
   bool feasible;
   uint32_t n_roots =
      hk_bvh_select_flatten_roots(t->nodes, t->internal_count, depth, height,
                                  leaves, budget, roots, stack, &feasible);

   if (peak <= budget && n_roots)
      FAIL("tree already fits %u slots but %u subtrees were selected", budget,
           n_roots);

   for (uint32_t r = 0; r < n_roots; r++) {
      uint32_t leaf_refs[MAX_NODES * 2], slots[MAX_NODES], n_leaves, n_slots;
      vk_aabb bounds[MAX_NODES * 2];
      uint8_t mark[MAX_NODES];

      if (!hk_bvh_gather_subtree(t->nodes, t->internal_count, roots[r],
                                 MAX_NODES * 2, leaf_refs, bounds, &n_leaves,
                                 slots, &n_slots, stack, mark))
         FAIL("gather failed at root %u", roots[r]);

      if (slots[0] != roots[r])
         FAIL("gather did not put the subtree root first");

      /* A binary tree with k leaves always owns at least k - 1 internal
       * nodes, which is exactly what a balanced rebuild needs. */
      if (n_slots + 1 < n_leaves)
         FAIL("subtree has %u slots for %u leaves", n_slots, n_leaves);

      hk_bvh_rebuild_balanced(t->nodes, slots, n_slots, leaf_refs, bounds,
                              n_leaves, stack);
   }

   if (!n_roots)
      return true;

   if (!parent_before_child(t))
      FAIL("flatten inverted parent-before-child");

   uint32_t after_max = hk_bvh_compute_depths(t->nodes, t->internal_count, depth);
   uint32_t after_peak = simulate_worst_peak_stack(t);

   if (hk_bvh_stack_slots_required(after_max) != after_peak)
      FAIL("after flatten: predicted %u, simulated %u",
           hk_bvh_stack_slots_required(after_max), after_peak);

   if (feasible && after_peak > budget)
      FAIL("after flatten still needs %u slots, budget %u", after_peak, budget);

   /* 4. No leaf gained, lost or duplicated. */
   uint32_t after[MAX_NODES * 2], after_count;
   collect_leaves(t, after, &after_count);
   if (after_count != total_leaves)
      FAIL("flatten changed the leaf count: %u -> %u", total_leaves,
           after_count);

   qsort(before, total_leaves, sizeof(uint32_t), cmp_u32);
   qsort(after, after_count, sizeof(uint32_t), cmp_u32);
   if (memcmp(before, after, total_leaves * sizeof(uint32_t)) != 0)
      FAIL("flatten changed which leaves are present");

   return true;
}

int
main(void)
{
   bool ok = true;
   char name[128];

   /* Budgets deliberately include values either side of the real limits, so a
    * rule that hardcoded HK_MAX_TLAS_DEPTH or HK_MAX_BLAS_DEPTH stands out. */
   const uint32_t budgets[] = {4, 6, 9, HK_MAX_TLAS_DEPTH, HK_MAX_BLAS_DEPTH};

   for (unsigned b = 0; b < ARRAY_SIZE(budgets); b++) {
      for (uint32_t n = 1; n <= 40; n++) {
         struct tree t;
         build_chain(&t, n);
         snprintf(name, sizeof(name), "chain n=%u budget=%u", n, budgets[b]);
         ok &= check_tree(name, &t, budgets[b]);
      }

      for (uint32_t h = 1; h <= 7; h++) {
         struct tree t;
         build_balanced(&t, h);
         snprintf(name, sizeof(name), "balanced h=%u budget=%u", h, budgets[b]);
         ok &= check_tree(name, &t, budgets[b]);
      }

      unsigned seed = 12345u + b;
      for (uint32_t trial = 0; trial < 200; trial++) {
         struct tree t;
         build_random(&t, 1 + (rand_r(&seed) % 120), &seed);
         snprintf(name, sizeof(name), "random trial=%u budget=%u", trial,
                  budgets[b]);
         ok &= check_tree(name, &t, budgets[b]);
      }
   }

   return ok ? 0 : 1;
}
