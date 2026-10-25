/*
 * Resource words of a batched G18G compute stream.
 *
 * The kernel sizes the stream's compute plan (pool, SMA, counters) from one
 * set of resource words per stream, so every launch that joins the stream,
 * user dispatch or precompiled helper, must raise the accumulated words, never
 * replace them: a small last launch after 1024-thread launches would otherwise
 * under-size the plan. Scalars take the running maximum, flags are ORed bit by
 * bit, and the local size comes from the launch with the largest x*y*z (a
 * per-axis maximum can exceed the validator's 1024-thread bound).
 *
 * Pure C on purpose: the offline test links the production code.
 */
#ifndef HK_G18_RESOURCES_H
#define HK_G18_RESOURCES_H

#include <stdbool.h>
#include <stdint.h>

#include "drm-uapi/asahi_drm.h"

static inline void
hk_g18_merge_compute_resources(struct drm_asahi_g18_compute_resources *acc,
                               const struct drm_asahi_g18_compute_resources *launch,
                               bool first)
{
   if (first) {
      *acc = *launch;
      return;
   }

#define HK_G18_MAX_FIELD(f) \
   do { if (launch->f > acc->f) acc->f = launch->f; } while (0)
   HK_G18_MAX_FIELD(registers_16);
   HK_G18_MAX_FIELD(scratch_bytes);
   HK_G18_MAX_FIELD(spectpr_scratch_bytes);
   HK_G18_MAX_FIELD(staging_16);
   HK_G18_MAX_FIELD(constant_registers_16);
   HK_G18_MAX_FIELD(constant_scratch_bytes);
   HK_G18_MAX_FIELD(constant_staging_16);
   HK_G18_MAX_FIELD(imageblock_stride);
#undef HK_G18_MAX_FIELD

   /* local_bytes and variable_shared_bytes move as a PAIR selected by their
    * SUM: the kernel's shared check (w7P check i) bounds
    * (local_bytes + variable_shared) * scale <= 32768, so separate maxima
    * can sum past the limit although every launch alone fits (the op-8
    * regression: 120832 > 32768). The launch with the largest sum
    * contributes both words — which can LOWER one of them: a later launch
    * whose sum wins runs with the pair's own variable_shared, even if it is
    * smaller than an earlier launch's. This rests on the kernel sizing the
    * plan from the pair TOTAL, not the individual fields. */
   if ((uint64_t)launch->local_bytes + launch->variable_shared_bytes >
       (uint64_t)acc->local_bytes + acc->variable_shared_bytes) {
      acc->local_bytes = launch->local_bytes;
      acc->variable_shared_bytes = launch->variable_shared_bytes;
   }

   acc->flags |= launch->flags;

   const uint64_t acc_prod = (uint64_t)acc->local_size[0] *
                             acc->local_size[1] * acc->local_size[2];
   const uint64_t launch_prod = (uint64_t)launch->local_size[0] *
                                launch->local_size[1] * launch->local_size[2];
   if (launch_prod >= acc_prod) {
      acc->local_size[0] = launch->local_size[0];
      acc->local_size[1] = launch->local_size[1];
      acc->local_size[2] = launch->local_size[2];
   }
}

#endif
