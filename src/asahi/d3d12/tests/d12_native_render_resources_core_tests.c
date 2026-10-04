/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT */
#include "../d12_native_render_resources_core.h"
#include "asahi/vulkan/hk_drm_render_core.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(void)
{
   assert(!getenv("AGX_NO_RSRC_SPEC_HI"));
   const uint64_t values[] = {0, 1, UINT32_MAX, UINT64_C(0x100000000),
                            UINT64_C(0xffff0123456789ab), UINT64_MAX};
   unsigned cases = 0;
   for (unsigned a=0;a<6;a++) for (unsigned b=0;b<6;b++)
      for (unsigned c=0;c<6;c++) for (unsigned d=0;d<6;d++) {
         struct drm_asahi_cmd_render actual={0}, expected={0};
         actual.flags=expected.flags=DRM_ASAHI_RENDER_PROCESS_EMPTY_TILES;
         const struct d12_native_render_resource_specs ours={values[a],values[b],values[c],values[d]};
         const struct hk_drm_render_resource_specs hk={values[a],values[b],values[c],values[d]};
         d12_native_render_resource_specs_pack(&ours,&actual);
         hk_drm_render_resource_specs_pack(&hk,&expected);
         assert(!memcmp(&actual,&expected,sizeof(actual)));
         assert(actual.flags & DRM_ASAHI_RENDER_PROCESS_EMPTY_TILES);
         assert(((uint64_t)actual.bg_rsrc_spec_hi<<32 | actual.bg.rsrc_spec)==ours.bg);
         assert(((uint64_t)actual.bg_eot_rsrc_spec_hi<<32 | actual.eot.rsrc_spec)==ours.eot);
         assert(((uint64_t)actual.bg_partial_rsrc_spec_hi<<32 | actual.partial_bg.rsrc_spec)==ours.partial_bg);
         assert(((uint64_t)actual.bg_eot_partial_rsrc_spec_hi<<32 | actual.partial_eot.rsrc_spec)==ours.partial_eot);
         cases++;
      }
   printf("PASS native BG/EOT UAPI resource widths/flag match HK: %u cases\n",cases);
   return 0;
}
