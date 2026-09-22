/* SPDX-License-Identifier: MIT */
/* Link against each generated libagx_shaders.c to compare compiler changes. */
#include "asahi/compiler/agx_compile.h"
#include "libagx_shaders.h"
#include <inttypes.h>
#include <stdio.h>

int
main(void)
{
   const uint32_t **tables[] = {libagx_g13g, libagx_g13x, libagx_g17p};
   for (unsigned arch = 0; arch < ARRAY_SIZE(tables); ++arch) {
      for (unsigned i = 0; i < LIBAGX_NUM_PROGRAMS; ++i) {
         const struct agx_precompiled_kernel_info *info =
            (const void *)tables[arch][i];
         if (!info)
            continue;
         const uint8_t *bytes = (const void *)(info + 1);
         uint64_t hash = UINT64_C(14695981039346656037);
         for (unsigned j = 0; j < info->binary_size; ++j) {
            hash ^= bytes[j];
            hash *= UINT64_C(1099511628211);
         }
         printf("%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%016"
                PRIx64 "\n",
                arch, i, info->main_size, info->binary_size, info->nr_gprs,
                info->nr_preamble_gprs, info->agx3_tps_size, info->push_count,
                info->main_offset, info->preamble_offset,
                info->workgroup_size[0], info->workgroup_size[1],
                info->workgroup_size[2], info->local_size, info->imageblock_stride,
                info->uses_txf, info->agx3_staging_known,
                info->agx3_staging_halfs, info->printf_uniform, hash);
      }
   }
   return 0;
}
