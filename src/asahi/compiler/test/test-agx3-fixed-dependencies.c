/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "agx3_fixed_dependencies.h"
#include <stdio.h>
#include <string.h>

static unsigned checks, failures;
#define CHECK(x) do { \
   ++checks; \
   if (!(x)) { \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); \
      ++failures; \
   } \
} while (0)

struct graph { bool edge[6][6]; unsigned callbacks; };
static void
edge(void *data, uint32_t before, uint32_t after)
{
   struct graph *g = data;
   CHECK(before < after && after < 6);
   if (before < 6 && after < 6)
      g->edge[before][after] = true;
   g->callbacks++;
}

static void
hazards_and_independence(void)
{
   struct agx3_fixed_access a[] = {
      {0, 0, 4, true}, {1, 2, 2, false}, {2, 1, 2, true},
      {3, 0, 1, false}, {4, 4, 1, true},
   };
   struct graph g = {0};
   CHECK(agx3_fixed_dependencies(a, 5, 8, edge, &g));
   CHECK(g.edge[0][1]); /* RAW, upper half of the original vector */
   CHECK(g.edge[0][2]); /* WAW, partial overlap */
   CHECK(g.edge[1][2]); /* WAR, overlap of read and replacement */
   CHECK(g.edge[0][3]); /* The untouched lower half still has writer0 */
   CHECK(!g.edge[2][3] && !g.edge[1][3]);
   for (unsigned i = 0; i < 4; ++i)
      CHECK(!g.edge[i][4]); /* Independent destination may move freely. */

   struct agx3_fixed_access readers[] = {
      {0, 0, 2, true}, {1, 0, 2, false}, {2, 0, 2, false},
      {3, 0, 2, true}, {4, 0, 2, false},
   };
   memset(&g, 0, sizeof(g));
   CHECK(agx3_fixed_dependencies(readers, 5, 8, edge, &g));
   CHECK(g.edge[0][1] && g.edge[0][2]);
   CHECK(!g.edge[1][2] && !g.edge[2][1]);
   CHECK(g.edge[1][3] && g.edge[2][3] && g.edge[0][3]);
   CHECK(g.edge[3][4]);
   CHECK(!g.edge[0][4] && !g.edge[1][4] && !g.edge[2][4]);

   struct agx3_fixed_access self[] = {
      {0, 511, 1, false}, {0, 511, 1, true},
      {1, 511, 1, false}, {1, 511, 1, true},
   };
   memset(&g, 0, sizeof(g));
   CHECK(agx3_fixed_dependencies(self, 4, 512, edge, &g));
   CHECK(g.edge[0][1]);
   CHECK(!g.edge[0][0] && !g.edge[1][1]);
}

static void
reject_before_emitting_edges(void)
{
   struct agx3_fixed_access a[] = {
      {0, 0, 2, true}, {1, 0, 2, false}, {2, 7, 2, true},
   };
   struct graph g = {0};
   CHECK(!agx3_fixed_dependencies(a, 3, 8, edge, &g));
   CHECK(!g.callbacks);
   a[2].first = 0;
   a[2].instruction = 0;
   CHECK(!agx3_fixed_dependencies(a, 3, 8, edge, &g));
   a[2].instruction = UINT32_MAX;
   CHECK(!agx3_fixed_dependencies(a, 3, 8, edge, &g));
   a[2].instruction = 2;
   a[2].count = 0;
   CHECK(!agx3_fixed_dependencies(a, 3, 8, edge, &g));
   CHECK(!g.callbacks);
   CHECK(!agx3_fixed_dependencies(NULL, 1, 8, edge, &g));
   CHECK(!agx3_fixed_dependencies(a, 2, 0, edge, &g));
   CHECK(!agx3_fixed_dependencies(a, 2, 513, edge, &g));
   CHECK(!agx3_fixed_dependencies(a, 2, 8, NULL, &g));
   CHECK(agx3_fixed_dependencies(NULL, 0, 8, edge, &g));
   CHECK(!g.callbacks);
}

/* Replay all topological permutations of six instructions against the
 * original program's read values. The interpreter knows nothing about the
 * dependency algorithm: it just reads/writes eight half-register cells. */
struct replay {
   struct agx3_fixed_access accesses[12];
   unsigned count;
   struct graph graph;
   unsigned expected[12][8];
   unsigned order[6];
   unsigned schedules;
};

static void
interpret(struct replay *r, bool reference)
{
   unsigned memory[8] = {0};
   for (unsigned pos = 0; pos < 6; ++pos) {
      unsigned instruction = reference ? pos : r->order[pos];
      for (unsigned i = 0; i < r->count; ++i) {
         const struct agx3_fixed_access *a = &r->accesses[i];
         if (a->instruction != instruction)
            continue;
         for (unsigned h = a->first; h < a->first + a->count; ++h) {
            if (a->write)
               memory[h] = 100 * (instruction + 1) + h;
            else if (reference)
               r->expected[i][h] = memory[h];
            else
               CHECK(memory[h] == r->expected[i][h]);
         }
      }
   }
}

static void
permutations(struct replay *r, unsigned used, unsigned pos)
{
   if (pos == 6) {
      ++r->schedules;
      interpret(r, false);
      return;
   }
   for (unsigned i = 0; i < 6; ++i) {
      if (used & (1u << i))
         continue;
      bool ready = true;
      for (unsigned p = 0; p < 6; ++p)
         if (r->graph.edge[p][i] && !(used & (1u << p)))
            ready = false;
      if (ready) {
         r->order[pos] = i;
         permutations(r, used | (1u << i), pos + 1);
      }
   }
}

static void
schedule_replay(void)
{
   uint32_t seed = 0x31415926;
   unsigned schedules = 0;
   for (unsigned test = 0; test < 100; ++test) {
      struct replay r = {0};
      for (unsigned i = 0; i < 6; ++i) {
         seed = seed * 1664525u + 1013904223u;
         unsigned halves = 1 + ((seed >> 16) & 1);
         unsigned first = (seed >> 20) % (9 - halves);
         r.accesses[r.count++] = (struct agx3_fixed_access){
            i, first, halves, (seed >> 18) & 1,
         };
         /* Some instructions read and then replace their own input cells. */
         if ((seed >> 24) % 3 == 0)
            r.accesses[r.count++] = (struct agx3_fixed_access){i, first, halves, true};
      }
      CHECK(agx3_fixed_dependencies(r.accesses, r.count, 8, edge, &r.graph));
      interpret(&r, true);
      permutations(&r, 0, 0);
      CHECK(r.schedules != 0);
      schedules += r.schedules;
   }
   printf("replayed %u valid schedules\n", schedules);
}

int main(void)
{
   hazards_and_independence();
   reject_before_emitting_edges();
   schedule_replay();
   printf("AGX3 fixed dependencies: %u checks, %u failures\n", checks, failures);
   return failures != 0;
}
