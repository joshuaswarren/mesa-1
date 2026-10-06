/* Copyright 2026 Aurora Silicon. SPDX-License-Identifier: MIT */
#include "agx3_staging.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

struct interval { unsigned start, end; uint16_t readers; };
static struct interval oracle[4096];
static unsigned count;

static uint16_t
conflicts(unsigned start, unsigned span)
{
   uint16_t result = 0;
   for (unsigned i = 0; i < count; ++i)
      if (oracle[i].start < start + span && start < oracle[i].end)
         result |= oracle[i].readers;
   return result;
}

static uint32_t rng = 0xa187064;
static uint32_t random32(void)
{
   rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
   return rng;
}

int main(void)
{
   struct agx3_staging_banks b = {0}, before;
   uint16_t base, waits;

   /* Independent outstanding coordinate ranges in Apple's linked 7064 FS.
    * More than eight ranges can be live: range count is not slot count.
    */
   const unsigned native[] = {0, 4, 8, 12, 16, 84, 128, 132, 140, 152};
   for (unsigned i = 0; i < sizeof(native) / sizeof(native[0]); ++i)
      assert(agx3_staging_hold(&b, native[i], 4, 1 + i % 6));
   assert(agx3_staging_plan(&b, 4, 4, 256, &base, &waits));
   assert(base == 20 && waits == 0);
   assert(agx3_staging_hold(&b, 84, 4, 8));
   assert(agx3_staging_conflicts(&b, 84, 4) == ((1u << 5) | (1u << 7)));
   agx3_staging_forget(&b, 1u << 5);
   assert(agx3_staging_conflicts(&b, 84, 4) == (1u << 7));
   agx3_staging_release(&b, 0);
   assert(agx3_staging_held(&b) == 0);
   assert(agx3_staging_plan(&b, 4, 4, 256, &base, &waits) && base == 0);

   before = b;
   assert(!agx3_staging_hold(&b, 253, 4, 1));
   assert(!agx3_staging_hold(&b, 0, 0, 1));
   assert(!agx3_staging_hold(&b, 0, 4, 9));
   assert(!agx3_staging_hold(&b, 0, 4, 0));
   assert(!memcmp(&before, &b, sizeof(b)));
   assert(!agx3_staging_plan(&b, 257, 4, 256, &base, &waits));
   assert(!agx3_staging_plan(&b, 4, 3, 256, &base, &waits));
   assert(!agx3_staging_plan(&b, 4, 4, 257, &base, &waits));

   unsigned cases = 0;
   for (unsigned round = 0; round < 100; ++round) {
      memset(&b, 0, sizeof(b)); count = 0;
      for (unsigned step = 0; step < 1000; ++step) {
         unsigned span = 1 + random32() % 40;
         unsigned start = random32() % (257 - span);
         unsigned slot = 1 + random32() % 8;
         if (random32() % 3) {
            assert(agx3_staging_hold(&b, start, span, slot));
            oracle[count++] = (struct interval){start, start + span, 1u << (slot - 1)};
         } else {
            uint16_t outstanding = random32() & 255;
            agx3_staging_release(&b, outstanding);
            for (unsigned i = 0; i < count; ++i)
               oracle[i].readers &= outstanding;
         }
         assert(agx3_staging_conflicts(&b, start, span) == conflicts(start, span));
         unsigned align = 1u << (random32() % 5);
         unsigned limit = 64 + random32() % 193;
         assert(agx3_staging_plan(&b, span, align, limit, &base, &waits));
         assert(base % align == 0 && base + span <= limit);
         assert(waits == conflicts(base, span));
         for (unsigned i = 0; i <= limit - span; i += align) {
            unsigned n = __builtin_popcount(conflicts(i, span));
            assert(n >= (unsigned)__builtin_popcount(waits));
            assert(i >= base || n > (unsigned)__builtin_popcount(waits));
         }
         /* The plan itself must not retire any producer. */
         assert(agx3_staging_conflicts(&b, start, span) == conflicts(start, span));
         ++cases;
      }
   }
   printf("staging allocator: %u interval-oracle cases passed\n", cases);
   return 0;
}
