/* Copyright 2026 Aurora Silicon. SPDX-License-Identifier: MIT */
#include "agx3_resource_ir.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 64
#define V 24
#define U 128

static uint32_t rng = 0x7064c0de;
static uint32_t random32(void)
{
   rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
   return rng;
}

/* Independent completion model: publications write only when their event
 * completes; late reads inspect the file again on completion. Values are
 * distinguishable, including each half. An omitted RAW wait or WAR lifetime
 * therefore changes a consumer's observed data.
 */
struct machine {
   const struct agx3_resource_value *values;
   const struct agx3_resource_use *uses;
   const uint16_t *bases;
   unsigned nv, nu;
   uint32_t cells[256];
   int pending[9];
};

static uint32_t expected(unsigned value, unsigned half)
{
   return 1 + value * 256 + half;
}

static bool check_reads(struct machine *m, unsigned instruction, bool late_only)
{
   for (unsigned u = 0; u < m->nu; ++u) {
      const struct agx3_resource_use *R = &m->uses[u];
      if (R->instruction != instruction || (late_only && !R->through_completion))
         continue;
      for (unsigned h = R->first; h < R->first + R->count; ++h)
         if (m->cells[m->bases[R->value] + h] != expected(R->value, h))
            return false;
   }
   return true;
}

static bool complete(struct machine *m, unsigned slot)
{
   int instruction = m->pending[slot];
   if (instruction < 0)
      return true;
   if (!check_reads(m, instruction, true))
      return false;
   for (unsigned v = 0; v < m->nv; ++v) {
      if (m->values[v].producer == (unsigned)instruction)
         for (unsigned h = 0; h < m->values[v].halves; ++h)
            m->cells[m->bases[v] + h] = expected(v, h);
   }
   m->pending[slot] = -1;
   return true;
}

static bool simulate(const struct agx3_resource_value *values, unsigned nv,
                     const struct agx3_resource_use *uses, unsigned nu,
                     const uint16_t *bases, const struct agx3_slot_result *res,
                     unsigned n)
{
   struct machine m = {.values = values, .uses = uses, .bases = bases,
                        .nv = nv, .nu = nu};
   for (unsigned s = 0; s < 9; ++s)
      m.pending[s] = -1;
   for (unsigned i = 0; i < n; ++i) {
      for (unsigned s = 1; s <= 8; ++s)
         if ((res[i].wait_mask & (1u << (s - 1))) && !complete(&m, s))
            return false;
      if (!check_reads(&m, i, false))
         return false;
      if (res[i].slot) {
         if (m.pending[res[i].slot] >= 0)
            return false;
         m.pending[res[i].slot] = i;
      }
   }
   for (unsigned s = 1; s <= 8; ++s)
      if (!complete(&m, s))
         return false;
   return true;
}

static void check_intervals(const struct agx3_resource_value *v, unsigned nv,
                            const struct agx3_resource_use *u, unsigned nu,
                            const uint16_t *bases, unsigned capacity)
{
   unsigned end[V];
   for (unsigned i = 0; i < nv; ++i) {
      end[i] = v[i].producer;
      for (unsigned j = 0; j < nu; ++j)
         if (u[j].value == i && u[j].instruction > end[i])
            end[i] = u[j].instruction;
      assert(bases[i] % v[i].alignment == 0);
      assert(bases[i] + v[i].halves <= capacity);
      assert(v[i].fixed_base == UINT16_MAX || bases[i] == v[i].fixed_base);
   }
   for (unsigned i = 0; i < nv; ++i)
      for (unsigned j = 0; j < i; ++j)
         if (end[i] >= v[j].producer && end[j] >= v[i].producer)
            assert(bases[i] + v[i].halves <= bases[j] ||
                   bases[j] + v[j].halves <= bases[i]);
}

int main(void)
{
   struct agx3_slot_instr base[N] = {0}, lowered[N], bad[N];
   struct agx3_slot_result res[N], broken[N];
   uint16_t placements[V];
   for (unsigned i = 0; i < N; ++i) {
      base[i].is_async = true;
      base[i].nr_dsts = 1;
      base[i].dst[0] = i;
   }

   /* Apple linked7064: publish once, then four independent samples. A fifth
    * publication reuses the range only after all four readers have retired.
    */
   struct agx3_resource_value values[V] = {
      {0, 4, 4, UINT16_MAX}, {5, 4, 4, UINT16_MAX},
   };
   struct agx3_resource_use uses[U] = {
      {1, 0, 0, 4, true}, {2, 0, 0, 4, true},
      {3, 0, 0, 4, true}, {4, 0, 0, 4, true}, {6, 1, 0, 4, true},
   };
   assert(agx3_resource_lower(values, 2, uses, 5, base, 7, 4,
                              placements, lowered));
   assert(placements[0] == 0 && placements[1] == 0);
   assert(agx3_slot_assign(lowered, 7, 6, res));
   assert(res[1].wait_mask == 1);
   for (unsigned i = 2; i <= 4; ++i)
      assert(res[i].wait_mask == 0);
   unsigned reader_mask = 0;
   for (unsigned i = 1; i <= 4; ++i)
      reader_mask |= 1u << (res[i].slot - 1);
   assert((res[5].wait_mask & reader_mask) == reader_mask);
   assert(simulate(values, 2, uses, 5, placements, res, 7));
   memcpy(broken, res, sizeof(res));
   broken[1].wait_mask = 0;
   assert(!simulate(values, 2, uses, 5, placements, broken, 7));
   memcpy(bad, lowered, sizeof(lowered));
   for (unsigned i = 0; i < 7; ++i)
      bad[i].late_src_mask = 0;
   assert(agx3_slot_assign(bad, 7, 6, broken));
   assert(!simulate(values, 2, uses, 5, placements, broken, 7));

   /* Logical last use does not mean its async read has completed. With
    * additional space available, preserve overlap instead of forcing a WAR
    * wait just to reuse the lowest-numbered range.
    */
   values[0] = (struct agx3_resource_value){0, 4, 4, UINT16_MAX};
   values[1] = (struct agx3_resource_value){2, 4, 4, UINT16_MAX};
   uses[0] = (struct agx3_resource_use){1, 0, 0, 4, true};
   uses[1] = (struct agx3_resource_use){3, 1, 0, 4, true};
   assert(agx3_resource_lower(values, 2, uses, 2, base, 4, 8,
                              placements, lowered));
   assert(placements[0] == 0 && placements[1] == 4);
   assert(agx3_slot_assign(lowered, 4, 6, res));
   assert(res[2].wait_mask == 0);
   assert(simulate(values, 2, uses, 2, placements, res, 4));

   /* A future fixed ABI publication must have room throughout its lifetime,
    * even when an earlier movable value remains live across its definition.
    */
   values[0] = (struct agx3_resource_value){0, 4, 4, UINT16_MAX};
   values[1] = (struct agx3_resource_value){1, 4, 4, 0};
   uses[0] = (struct agx3_resource_use){2, 0, 0, 4, true};
   uses[1] = (struct agx3_resource_use){3, 1, 0, 4, true};
   assert(agx3_resource_lower(values, 2, uses, 2, base, 4, 8,
                              placements, lowered));
   assert(placements[0] == 4 && placements[1] == 0);
   memset(bad, 0xa5, sizeof(bad));
   memcpy(lowered, bad, sizeof(lowered));
   placements[0] = 0xf00d;
   assert(!agx3_resource_lower(values, 2, uses, 2, base, 4, 4,
                               placements, lowered));
   assert(placements[0] == 0xf00d && !memcmp(lowered, bad, sizeof(bad)));
   uses[0].instruction = 0;
   assert(!agx3_resource_lower(values, 2, uses, 2, base, 4, 8,
                               placements, lowered));

   /* Apple's other linked7064 shape: four publications before their samples.
    * No unrelated reader needs to finish merely to publish another value.
    */
   for (unsigned i = 0; i < 4; ++i) {
      values[i] = (struct agx3_resource_value){i, 4, 4, UINT16_MAX};
      uses[i] = (struct agx3_resource_use){i + 4, i, 0, 4, true};
   }
   assert(agx3_resource_lower(values, 4, uses, 4, base, 8, 16,
                              placements, lowered));
   assert(agx3_slot_assign(lowered, 8, 6, res));
   for (unsigned i = 0; i < 4; ++i) {
      assert(placements[i] == 4 * i);
      assert(res[i].wait_mask == 0);
   }
   assert(simulate(values, 4, uses, 4, placements, res, 8));

   /* In-place lowering is required by the packer migration. Non-resource
    * TPS dependencies retain their separate namespace, and repeated partial
    * views must merge completion semantics without duplicating aliases.
    */
   memcpy(bad, base, sizeof(bad));
   bad[0].nr_srcs = 1;
   bad[0].src[0] = AGX3_SLOT_TPS_TOKEN_BASE + 7;
   values[0] = (struct agx3_resource_value){0, 4, 4, UINT16_MAX};
   uses[0] = (struct agx3_resource_use){1, 0, 0, 4, false};
   uses[1] = (struct agx3_resource_use){1, 0, 1, 2, true};
   assert(agx3_resource_lower(values, 1, uses, 2, bad, 2, 4,
                              placements, bad));
   assert(bad[0].src[0] == AGX3_SLOT_TPS_TOKEN_BASE + 7);
   assert(bad[1].nr_srcs == 4 && bad[1].late_src_mask == 6);
   assert(agx3_resource_lower(NULL, 0, NULL, 0, NULL, 0, 256, NULL, NULL));
   values[0].fixed_base = 1;
   assert(!agx3_resource_lower(values, 1, uses, 2, base, 2, 4,
                               placements, lowered));
   values[0].fixed_base = UINT16_MAX;
   uses[0].count = 5;
   assert(!agx3_resource_lower(values, 1, uses, 1, base, 2, 8,
                               placements, lowered));

   for (unsigned round = 0; round < 4000; ++round) {
      unsigned nv = 1 + random32() % V, nu = 0;
      for (unsigned v = 0; v < nv; ++v)
         values[v] = (struct agx3_resource_value){v * 2, 2 + 2 * (random32() % 4),
                                                2, UINT16_MAX};
      for (unsigned i = 1; i < N; ++i) {
         unsigned defined = (i + 1) / 2;
         if (defined > nv) defined = nv;
         unsigned refs = 1 + random32() % 2;
         for (unsigned j = 0; j < refs; ++j) {
            unsigned v = random32() % defined;
            unsigned first = random32() % values[v].halves;
            uses[nu++] = (struct agx3_resource_use){
               i, v, first, 1 + random32() % (values[v].halves - first),
               (bool)(random32() & 1),
            };
         }
      }
      assert(agx3_resource_lower(values, nv, uses, nu, base, N, 256,
                                 placements, lowered));
      check_intervals(values, nv, uses, nu, placements, 256);
      assert(agx3_slot_assign(lowered, N, 1 + random32() % 8, res));
      assert(simulate(values, nv, uses, nu, placements, res, N));
   }
   puts("resource IR: shared publication, future uses, fixed ABI, negative controls and 4000 completion-oracle programs passed");
   return 0;
}
