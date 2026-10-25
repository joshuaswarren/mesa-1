/* Offline check for hk_g18_merge_compute_resources().
 * cc -Wall -Wextra -I../../../../include -I.. test-hk-g18-resources.c && ./a.out */
#include <stdio.h>
#include <string.h>

#include "hk_g18_resources.h"

static struct drm_asahi_g18_compute_resources
launch(uint32_t regs, uint32_t staging, uint32_t stride, uint32_t local_bytes,
       uint32_t x, uint32_t y, uint32_t z)
{
   struct drm_asahi_g18_compute_resources r;
   memset(&r, 0, sizeof(r));
   r.version = 1;
   r.flags = 1;
   r.registers_16 = regs;
   r.staging_16 = staging;
   r.imageblock_stride = stride;
   r.local_bytes = local_bytes;
   r.local_size[0] = x;
   r.local_size[1] = y;
   r.local_size[2] = z;
   return r;
}

int
main(void)
{
   int failures = 0;
#define CHECK(c) do { if (!(c)) { failures++; printf("FAIL line %d: %s\n", __LINE__, #c); } } while (0)

   struct drm_asahi_g18_compute_resources acc;
   memset(&acc, 0xa5, sizeof(acc)); /* the first launch must not read the old contents */

   /* Big launch first, small helper last (the job 131 shape): the plan keeps the big launch's needs. */
   struct drm_asahi_g18_compute_resources big = launch(96, 40, 8, 16384, 128, 8, 1);
   struct drm_asahi_g18_compute_resources small = launch(8, 4, 0, 0, 32, 1, 1);
   small.scratch_bytes = 512;
   hk_g18_merge_compute_resources(&acc, &big, true);
   hk_g18_merge_compute_resources(&acc, &small, false);
   CHECK(acc.version == 1 && acc.flags == 1);
   CHECK(acc.registers_16 == 96 && acc.staging_16 == 40 && acc.imageblock_stride == 8);
   CHECK(acc.local_bytes == 16384);        /* shared memory of the big launch kept */
   CHECK(acc.scratch_bytes == 512);        /* the helper's scratch is raised, not lost */
   CHECK(acc.local_size[0] == 128 && acc.local_size[1] == 8 && acc.local_size[2] == 1);

   /* Small launch first, big launch later: everything is raised. */
   hk_g18_merge_compute_resources(&acc, &small, true);
   hk_g18_merge_compute_resources(&acc, &big, false);
   CHECK(acc.registers_16 == 96 && acc.local_bytes == 16384 && acc.scratch_bytes == 512);
   CHECK(acc.local_size[0] == 128 && acc.local_size[1] == 8 && acc.local_size[2] == 1);

   /* Local size comes from the launch with the largest x*y*z, never a per-axis maximum
    * (256x1x1 and 1x256x1 would give 256x256x1 = 65536 threads, over the validator bound). */
   struct drm_asahi_g18_compute_resources wide = launch(1, 1, 0, 0, 256, 1, 1);
   /* Control: the old last-wins assignment loses the big launch's needs (this is the
    * failure the merge above prevents). */
   struct drm_asahi_g18_compute_resources old = big;
   old = small;
   CHECK(old.registers_16 == 8 && old.local_bytes == 0 && old.local_size[0] == 32);

   struct drm_asahi_g18_compute_resources tall = launch(1, 1, 0, 0, 1, 256, 1);
   hk_g18_merge_compute_resources(&acc, &wide, true);
   hk_g18_merge_compute_resources(&acc, &tall, false);
   CHECK((uint64_t)acc.local_size[0] * acc.local_size[1] * acc.local_size[2] == 256);

   /* A smaller product never replaces the dims, flags are ORed per bit. */
   struct drm_asahi_g18_compute_resources a = launch(1, 1, 0, 0, 64, 4, 1);
   struct drm_asahi_g18_compute_resources b = launch(1, 1, 0, 0, 2, 2, 1);
   a.flags = 1;
   b.flags = 2;
   hk_g18_merge_compute_resources(&acc, &a, true);
   hk_g18_merge_compute_resources(&acc, &b, false);
   CHECK(acc.local_size[0] == 64 && acc.local_size[1] == 4);
   CHECK(acc.flags == 3);

   if (failures)
      return 1;
   printf("HK_G18_RESOURCES_OK\n");
   return 0;
}
