/*
 * Copyright 2026 Joshua Warren
 * SPDX-License-Identifier: MIT
 */

/* Standalone self-test for the DRM driver-name allowlist. Build and run:
 *
 *   cc -Isrc -o agx-drm-name-selftest \
 *      src/asahi/lib/tests/agx-drm-name-selftest.c && ./agx-drm-name-selftest
 *
 * It needs only the C standard library, so it runs without a mesa build.
 */

#include "asahi/lib/agx_drm_name.h"

#include <stdio.h>
#include <string.h>

static unsigned failures;

static void
expect(bool value, const char *name)
{
   if (!value) {
      failures++;
      printf("FAIL: %s\n", name);
   }
}

int
main(void)
{
   expect(agx_drm_name_is_native("aurora"), "aurora is admitted");
   expect(agx_drm_name_is_native("asahi"), "legacy name is admitted");
   expect(!agx_drm_name_is_native("virtio_gpu"), "virtio is not native");
   expect(!agx_drm_name_is_native(NULL), "NULL name rejected");
   expect(!agx_drm_name_is_native("aurora-dri"), "prefix match rejected");
   expect(!agx_drm_name_is_native(""), "empty name rejected");

   if (failures) {
      printf("%u failure(s)\n", failures);
      return 1;
   }
   printf("agx-drm-name-selftest: all checks passed\n");
   return 0;
}
