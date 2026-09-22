/* Copyright 2026 Aurora Silicon; SPDX-License-Identifier: MIT */

/*
 * Host oracle for poly_map_adj_vertex(), the single source of truth for
 * adjacency decomposition.
 *
 * Two consumers depend on it agreeing with the Vulkan spec's adjacency
 * sequences: the VS prolog (via poly_map_to_line_adj() and friends in
 * geometry.cl) and libagx_write_adj_indices(), the kernel that materialises
 * the mapping as a real index buffer because a G17P hardware vertex shader
 * reads its invocation index from SR 0xdd and therefore never observes a remap
 * the prolog performed.
 *
 * The expected sequences below are written out literally from the spec's
 * "discarding the vertices only a geometry shader can see" rule, NOT derived
 * from the implementation, so this test can actually fail.
 */

#include "poly/geometry.h"

#include <stdio.h>

static unsigned checks;
static unsigned failures;

static void
check_sequence(const char *name, enum mesa_prim prim, const unsigned *expected,
               unsigned n)
{
   for (unsigned i = 0; i < n; ++i) {
      unsigned got = poly_map_adj_vertex(i, prim);

      ++checks;
      if (got != expected[i]) {
         fprintf(stderr, "FAIL %s: map(%u) = %u, expected %u\n", name, i, got,
                 expected[i]);
         ++failures;
      }
   }
}

int
main(void)
{
   /* Triangle list with adjacency: each primitive is six vertices and the
    * triangle keeps 0, 2, 4 of them. So (0,2,4), (6,8,10), (12,14,16). */
   static const unsigned tri_list[] = {0, 2, 4, 6, 8, 10, 12, 14, 16};
   check_sequence("triangle_list_adj", MESA_PRIM_TRIANGLES_ADJACENCY, tri_list,
                  ARRAY_SIZE(tri_list));

   /* Line list with adjacency: four vertices per primitive, the line keeps
    * 1 and 2. So (1,2), (5,6), (9,10), (13,14). */
   static const unsigned line_list[] = {1, 2, 5, 6, 9, 10, 13, 14};
   check_sequence("line_list_adj", MESA_PRIM_LINES_ADJACENCY, line_list,
                  ARRAY_SIZE(line_list));

   /* Line strip with adjacency: line i keeps vertices i+1 and i+2.
    * So (1,2), (2,3), (3,4), (4,5). */
   static const unsigned line_strip[] = {1, 2, 2, 3, 3, 4, 4, 5};
   check_sequence("line_strip_adj", MESA_PRIM_LINE_STRIP_ADJACENCY, line_strip,
                  ARRAY_SIZE(line_strip));

   /* Triangle strip with adjacency: after discarding the geometry-shader-only
    * vertices this is a plain triangle strip at even indices, with the winding
    * order of odd triangles flipped.
    * So (0,2,4), (2,6,4), (4,6,8), (6,10,8). */
   static const unsigned tri_strip[] = {0, 2, 4, 2, 6, 4, 4, 6, 8, 6, 10, 8};
   check_sequence("triangle_strip_adj", MESA_PRIM_TRIANGLE_STRIP_ADJACENCY,
                  tri_strip, ARRAY_SIZE(tri_strip));

   /* No mapping may read past the input vertices the API actually supplied.
    * The output counts below are libagx_remap_adj_count()'s formulas, written
    * out independently here so a change to either side is caught. */
   for (unsigned count = 0; count < 64; ++count) {
      struct {
         enum mesa_prim prim;
         unsigned out;
      } cases[] = {
         {MESA_PRIM_TRIANGLES_ADJACENCY, count / 2},
         {MESA_PRIM_LINES_ADJACENCY, count / 2},
         {MESA_PRIM_LINE_STRIP_ADJACENCY, 2 * (count >= 3 ? count - 3 : 0)},
         {MESA_PRIM_TRIANGLE_STRIP_ADJACENCY,
          3 * ((count >= 4 ? count - 4 : 0) / 2)},
      };

      for (unsigned c = 0; c < ARRAY_SIZE(cases); ++c) {
         for (unsigned i = 0; i < cases[c].out; ++i) {
            unsigned v = poly_map_adj_vertex(i, cases[c].prim);

            ++checks;
            if (v >= count) {
               fprintf(stderr,
                       "FAIL prim %u, %u input vertices: output %u of %u maps "
                       "to input %u, which is out of range\n",
                       (unsigned)cases[c].prim, count, i, cases[c].out, v);
               ++failures;
            }
         }
      }
   }

   printf("test-poly-adj-map: %u checks, %u failures\n", checks, failures);
   return failures ? 1 : 0;
}
