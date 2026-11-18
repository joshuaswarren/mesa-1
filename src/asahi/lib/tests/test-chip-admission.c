/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/* Every check in this test is an assert(); keep them in NDEBUG builds. */
#undef NDEBUG

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

#include "agx_chip.h"
#include "agx_va_runtime_hole.h"
#include "util/macros.h"
#include "util/u_math.h"

static struct drm_asahi_params_global
g15_params(char variant, uint32_t chip_id, uint32_t clusters)
{
   struct drm_asahi_params_global params = {
      .gpu_generation = 15,
      .gpu_variant = variant,
      .chip_id = chip_id,
      .num_dies = 1,
      .num_clusters_total = clusters,
      .num_cores_per_cluster = 10,
      .usc_generation = 3,
      .gpu_hal_generation = DRM_ASAHI_GPU_HAL_LEGACY,
   };

   for (unsigned i = 0; i < clusters; ++i)
      params.core_masks[i] = 0x3ff;
   return params;
}

static bool
overlaps(uint64_t a0, uint64_t a1, uint64_t b0, uint64_t b1)
{
   return a0 < b1 && b0 < a1;
}

/* Rounded main heap span agx_open_device derives for a native chip below
 * the USC window (full window, no RO alias while sparse is unsupported). */
static void
native_user_window(enum agx_chip chip, uint64_t *start, uint64_t *end)
{
   const uint64_t limit = agx_va_user_limit(chip, true, UINT64_C(0x2f7ffff8000));
   uint64_t base = MAX2(UINT64_C(0x100000000), AGX_G17P_USER_LIMIT / 2);
   base = util_next_power_of_two64(base);
   assert(base < limit);
   *start = base;
   *end = limit;
}

static void
check_native_g15(enum agx_chip chip)
{
   assert(agx_chip_native_agx3(chip));
   assert(agx_chip_native_g15(chip));
   assert(agx_chip_native_imageblock(chip));
   assert(agx_chip_native_indirect_compute(chip));
   assert(agx_chip_driver_supported(chip));
   assert(agx_chip_usc_helper_table_slots(chip, 0) == 10);
}

int
main(void)
{
   check_native_g15(AGX_CHIP_G15S);
   check_native_g15(AGX_CHIP_G15C);
   check_native_g15(AGX_CHIP_G15G);

   /* The other M3 parts stay off the native path. */
   assert(!agx_chip_native_agx3(AGX_CHIP_G15D));
   assert(!agx_chip_driver_supported(AGX_CHIP_G15D));

   /* T6030 and T6031 are each admitted with their own identity only. */
   struct drm_asahi_params_global g15s = g15_params('S', 0x6030, 2);
   struct drm_asahi_params_global g15c = g15_params('C', 0x6031, 4);
   assert(agx_chip_from_params(&g15s) == AGX_CHIP_G15S);
   assert(agx_chip_from_params(&g15c) == AGX_CHIP_G15C);
   assert(agx_params_driver_supported(&g15s));
   assert(agx_params_driver_supported(&g15c));

   /* A binned part reports fewer cores in its masks and is still admitted. */
   struct drm_asahi_params_global binned = g15c;
   binned.core_masks[0] = binned.core_masks[3] = 0x1ff;
   assert(agx_params_driver_supported(&binned));

   struct drm_asahi_params_global bad = g15c;
   bad.chip_id = 0x6030;
   assert(!agx_params_driver_supported(&bad));

   bad = g15c;
   bad.num_clusters_total = 2;
   assert(!agx_params_driver_supported(&bad));

   bad = g15c;
   bad.num_dies = 2;
   assert(!agx_params_driver_supported(&bad));

   bad = g15c;
   bad.gpu_hal_generation = DRM_ASAHI_GPU_HAL_200;
   assert(!agx_params_driver_supported(&bad));

   bad = g15c;
   bad.usc_generation = 2;
   assert(!agx_params_driver_supported(&bad));

   bad = g15s;
   bad.chip_id = 0x6031;
   assert(!agx_params_driver_supported(&bad));

   bad = g15s;
   bad.num_clusters_total = 4;
   assert(!agx_params_driver_supported(&bad));

   /* Fresh J514M/T6034 identity: four physical clusters, 30 enabled cores. */
   struct drm_asahi_params_global t6034 = g15_params('C', 0x6034, 4);
   t6034.gpu_revision = 0x11;
   t6034.core_masks[0] = t6034.core_masks[1] = 0x1fd;
   t6034.core_masks[2] = t6034.core_masks[3] = 0x1ed;
   assert(agx_params_driver_supported(&t6034));
   bad = t6034; bad.chip_id = 0x6032;
   assert(!agx_params_driver_supported(&bad));
   bad = t6034; bad.gpu_revision = 0x12;
   assert(!agx_params_driver_supported(&bad));
   bad = t6034; bad.num_cores_per_cluster = 8;
   assert(!agx_params_driver_supported(&bad));
   bad = t6034; bad.num_clusters_total = 3;
   assert(!agx_params_driver_supported(&bad));
   bad = t6034; bad.num_dies = 2;
   assert(!agx_params_driver_supported(&bad));
   bad = t6034; bad.usc_generation = 2;
   assert(!agx_params_driver_supported(&bad));
   bad = t6034; bad.gpu_hal_generation = DRM_ASAHI_GPU_HAL_200;
   assert(!agx_params_driver_supported(&bad));
   bad = t6034; bad.core_masks[0] |= UINT64_C(1) << 10;
   assert(!agx_params_driver_supported(&bad));
   bad = t6034; bad.core_masks[0] = 0;
   assert(!agx_params_driver_supported(&bad));
   bad = t6034; bad.core_masks[0] ^= 1;
   assert(!agx_params_driver_supported(&bad));
   bad = t6034; bad.core_masks[0] |= 2;
   assert(!agx_params_driver_supported(&bad));
   assert(!agx_params_native_kernel_abi_supported(&t6034));
   t6034.features = DRM_ASAHI_FEATURE_PER_COMMAND_INPUT_SYNCS |
                    DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS;
   assert(agx_params_native_kernel_abi_supported(&t6034));

   /* The same GPU identity must not admit the incompatible legacy payloads. */
   assert(!agx_params_native_kernel_abi_supported(NULL));
   assert(!agx_params_native_kernel_abi_supported(&g15s));
   const uint64_t native = DRM_ASAHI_FEATURE_PER_COMMAND_INPUT_SYNCS |
                           DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS;
   g15s.features = native;
   g15c.features = native;
   assert(agx_params_native_kernel_abi_supported(&g15s));
   assert(agx_params_native_kernel_abi_supported(&g15c));
   /* Extra advertised capabilities do not change this packet contract. */
   for (unsigned bit = 3; bit <= 16; ++bit) {
      bad = g15c;
      bad.features |= UINT64_C(1) << bit;
      assert(agx_params_native_kernel_abi_supported(&bad));
   }
   bad = g15c;
   bad.features = (UINT64_C(1) << 3) | (UINT64_C(1) << 5);
   assert(!agx_params_native_kernel_abi_supported(&bad));
   bad = g15c;
   bad.features &= ~DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS;
   assert(!agx_params_native_kernel_abi_supported(&bad));
   bad.features = DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS;
   assert(!agx_params_native_kernel_abi_supported(&bad));
   bad = (struct drm_asahi_params_global){
      .gpu_generation = 17, .gpu_variant = 'P', .num_clusters_total = 1,
      .chip_id = 0x8140, .num_dies = 1, .usc_generation = 3,
      .gpu_hal_generation = DRM_ASAHI_GPU_HAL_200, .features = native,
   };
   assert(agx_params_driver_supported(&bad));
   assert(agx_params_native_kernel_abi_supported(&bad));
   bad.features = UINT64_C(0xfc00) | native;
   assert(agx_params_native_kernel_abi_supported(&bad));
   bad.features = (UINT64_C(1) << 3) | (UINT64_C(1) << 5);
   assert(!agx_params_native_kernel_abi_supported(&bad));
   bad = (struct drm_asahi_params_global){
      .gpu_generation = 13, .gpu_variant = 'G', .num_clusters_total = 1,
   };
   assert(agx_params_native_kernel_abi_supported(&bad));
   /* The M4 Pro (T6040, G16S, AGX3, HAL200) runs as G17P with its two
    * clusters; other G16S identities stay G16S and unsupported. */
   struct drm_asahi_params_global t6040 = g15_params('S', 0x6040, 2);
   t6040.gpu_generation = 16;
   t6040.gpu_hal_generation = DRM_ASAHI_GPU_HAL_200;
   assert(agx_chip_from_params(&t6040) == AGX_CHIP_G17P);
   assert(!agx_params_driver_supported(&t6040));

   bad = t6040;
   bad.num_clusters_total = 3;
   assert(!agx_params_driver_supported(&bad));

   bad = t6040;
   bad.chip_id = 0x6041;
   assert(agx_chip_from_params(&bad) == AGX_CHIP_G16S);
   assert(!agx_params_driver_supported(&bad));

   bad = t6040;
   bad.gpu_hal_generation = DRM_ASAHI_GPU_HAL_LEGACY;
   assert(agx_chip_from_params(&bad) == AGX_CHIP_G16S);

   /* G17P helper tables have a slot per core of a cluster, never fewer than
    * the A18 Pro's six: six there, ten on the M4 Pro. */
   assert(agx_chip_usc_helper_table_slots(AGX_CHIP_G17P, 0) == 6);
   assert(agx_chip_usc_helper_table_slots(AGX_CHIP_G17P, 6) == 6);
   assert(agx_chip_usc_helper_table_slots(AGX_CHIP_G17P, 10) == 10);

   struct drm_asahi_params_global m4 = t6040;
   m4.gpu_variant = 'G';
   m4.chip_id = 0x8132;
   m4.num_clusters_total = 1;
   assert(agx_chip_driver_supported(AGX_CHIP_G16G));
   assert(!agx_params_driver_supported(&m4));

   /* T8122 (G15G) is admitted by default for its exact identity and
    * supported legacy firmware/kernel contract. */
   struct drm_asahi_params_global g15g = g15_params('G', 0x8122, 1);
   assert(agx_chip_from_params(&g15g) == AGX_CHIP_G15G);
   assert(agx_params_driver_supported(&g15g));
   assert(!agx_params_native_kernel_abi_supported(&g15g));
   /* m3_drm.rs Backend::feature_flags: the Air runtime does not advertise
    * PER_COMMAND_INPUT_SYNCS or EXACT_PRIOR_BARRIERS. */
   g15g.features = DRM_ASAHI_FEATURE_VM_STATUS |
                   DRM_ASAHI_FEATURE_SCHEDULED_QUEUES;
   assert(agx_params_native_kernel_abi_supported(&g15g));
   bad = g15g;
   bad.features &= ~DRM_ASAHI_FEATURE_VM_STATUS;
   assert(!agx_params_native_kernel_abi_supported(&bad));
   bad = g15g;
   bad.features &= ~DRM_ASAHI_FEATURE_SCHEDULED_QUEUES;
   assert(!agx_params_native_kernel_abi_supported(&bad));
   bad = g15g;
   bad.features |= DRM_ASAHI_FEATURE_COMPUTE_WIDE_VISIBILITY;
   assert(agx_params_native_kernel_abi_supported(&bad));
   bad = g15g;
   bad.features = (UINT64_C(1) << 3) | (UINT64_C(1) << 5);
   assert(!agx_params_native_kernel_abi_supported(&bad));

   bad = g15g;
   bad.chip_id = 0x6030;
   assert(!agx_params_driver_supported(&bad));

   bad = g15g;
   bad.num_clusters_total = 2;
   assert(!agx_params_driver_supported(&bad));

   bad = g15g;
   bad.gpu_hal_generation = DRM_ASAHI_GPU_HAL_200;
   assert(!agx_params_driver_supported(&bad));

   bad = g15g;
   bad.usc_generation = 2;
   assert(!agx_params_driver_supported(&bad));

   /* T8122 (G15G): the heap stays below 2^39, the
    * absolute USC Texture/Sampler bound, and clear of every per-VM
    * reservation of the kernel's T8122 runtime. */
   uint64_t start, end;
   native_user_window(AGX_CHIP_G15G, &start, &end);
   assert(start == UINT64_C(0x4000000000) && end == UINT64_C(0x7000000000));
   assert(end <= UINT64_C(1) << 39);
   assert(!overlaps(start, end, AGX_T8122_RUNTIME_RESERVED_START,
                    AGX_T8122_RUNTIME_RESERVED_END));
   assert(!overlaps(start, end, UINT64_C(0x1000000000),
                    UINT64_C(0x1008000000)));
   assert(!overlaps(start, end, UINT64_C(0x1074000000),
                    UINT64_C(0x1074028000)));
   assert(!overlaps(start, end, AGX_T8122_RUNTIME_WINDOW_HIGH,
                    AGX_T8122_RUNTIME_WINDOW_END));

   /* G15S keeps the G17P-derived limit, and non-native chips the kernel's. */
   assert(agx_va_user_limit(AGX_CHIP_G15S, true, 1) == AGX_G17P_USER_LIMIT);
   assert(agx_va_user_limit(AGX_CHIP_G13G, false, 1234) == 1234);

   printf("chip admission: ok\n");
   return 0;
}
