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

   /* Every merged field, in both orders: a launch that raises one field and lowers
    * another must raise only the first (one field per launch, so a field that is
    * overwritten instead of maximised is caught). */
   struct drm_asahi_g18_compute_resources hi = launch(0, 0, 0, 0, 1, 1, 1);
   struct drm_asahi_g18_compute_resources lo = launch(0, 0, 0, 0, 1, 1, 1);
   hi.registers_16 = 10;           lo.registers_16 = 2;
   hi.scratch_bytes = 20;          lo.scratch_bytes = 3;
   hi.spectpr_scratch_bytes = 30;  lo.spectpr_scratch_bytes = 4;
   hi.staging_16 = 40;             lo.staging_16 = 5;
   hi.constant_registers_16 = 50;  lo.constant_registers_16 = 6;
   hi.constant_scratch_bytes = 60; lo.constant_scratch_bytes = 7;
   hi.constant_staging_16 = 70;    lo.constant_staging_16 = 8;
   hi.local_bytes = 80;            lo.local_bytes = 9;
   hi.imageblock_stride = 90;      lo.imageblock_stride = 11;
   hi.variable_shared_bytes = 100; lo.variable_shared_bytes = 12;
   for (int order = 0; order < 2; ++order) {
      hk_g18_merge_compute_resources(&acc, order ? &lo : &hi, true);
      hk_g18_merge_compute_resources(&acc, order ? &hi : &lo, false);
      CHECK(acc.registers_16 == 10 && acc.scratch_bytes == 20);
      CHECK(acc.spectpr_scratch_bytes == 30 && acc.staging_16 == 40);
      CHECK(acc.constant_registers_16 == 50 && acc.constant_scratch_bytes == 60);
      CHECK(acc.constant_staging_16 == 70 && acc.local_bytes == 80);
      CHECK(acc.imageblock_stride == 90 && acc.variable_shared_bytes == 100);
   }

   /* Shared bytes fold as a PAIR by sum (kernel check i bounds
    * (local_bytes + variable_shared) * scale <= 32768; separate maxima summed
    * to 120832 on the op-8 attention shapes and were rejected). The launch
    * with the largest SUM contributes BOTH words; a losing launch's larger
    * single field never leaks into the pair. */
   struct drm_asahi_g18_compute_resources lb_big = launch(1, 1, 0, 30000, 8, 1, 1);
   lb_big.variable_shared_bytes = 0;
   struct drm_asahi_g18_compute_resources vs_big = launch(1, 1, 0, 1000, 8, 1, 1);
   vs_big.variable_shared_bytes = 4000;
   hk_g18_merge_compute_resources(&acc, &lb_big, true);
   hk_g18_merge_compute_resources(&acc, &vs_big, false);
   CHECK(acc.local_bytes == 30000 && acc.variable_shared_bytes == 0);
   /* Control: separate maxima would give 30000 + 4000 = 34000 > 32768
    * (v1 scale x4: 136000) — the measured op-8 rejection class. */
   /* The SUM decides, not local_bytes alone: a launch with small local and
    * large variable (sum 41000) beats the (30000, 0) accumulator — the
    * local-only mutant would keep (30000, 0) and fail here. */
   struct drm_asahi_g18_compute_resources vs_wins = launch(1, 1, 0, 1000, 8, 1, 1);
   vs_wins.variable_shared_bytes = 40000;
   hk_g18_merge_compute_resources(&acc, &vs_wins, false);
   CHECK(acc.local_bytes == 1000 && acc.variable_shared_bytes == 40000);
   struct drm_asahi_g18_compute_resources pair2 = launch(1, 1, 0, 6000, 8, 1, 1);
   pair2.variable_shared_bytes = 100;
   hk_g18_merge_compute_resources(&acc, &pair2, false);
   CHECK(acc.local_bytes == 1000 && acc.variable_shared_bytes == 40000);
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
