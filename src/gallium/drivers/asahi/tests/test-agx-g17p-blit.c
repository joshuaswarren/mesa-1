/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/*
 * Offline reproduction of the native-Gallium G17P blit crash, under the
 * no-op DRM shim -- no GPU, no window system, no submission.
 *
 * Core 983 caught the driver dereferencing a null `ctx->vs->bo` inside
 * agx_draw_vbo(), reached from util_blitter_blit() for a 4x8
 * BGRA8_UNORM -> RGBA8_UNORM blit with uses_txf. The cause is that the AGX3
 * backend is fail-closed: agx_compile_shader_nir() returns false and zeroes
 * its output for a shader it cannot encode, so agx_compile_nir() produced a
 * compiled shader with binary_size == 0 and therefore no executable BO, and
 * the draw path used it anyway.
 *
 * Two things are asserted here, and they are separable:
 *
 *   1. The blit must never take the process down. A refused shader is a
 *      driver gap; it must surface as a skipped draw, not a SIGSEGV. This
 *      holds whether or not the shaders encode, and is the regression that
 *      guards the failure-propagation path.
 *
 *   2. With AGX_G17P_BLIT_REQUIRE_ENCODE=1, every shader the blit needs must
 *      actually encode. That is the stronger claim, and it is checked
 *      separately so a compiler regression cannot hide behind (1).
 *
 * The chip comes from ASAHI_SHIM_CHIP, so the same binary runs the G13G
 * control and the G17P case.
 */

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "pipe/p_context.h"
#include "pipe/p_defines.h"
#include "pipe/p_screen.h"
#include "pipe/p_state.h"
#include "util/format/u_formats.h"
#include "util/u_memory.h"
#include "util/driconf.h"
#include "util/xmlconfig.h"

#include "agx_public.h"
#include "agx_state.h"

static struct pipe_resource *
make_2d(struct pipe_screen *screen, enum pipe_format format, unsigned w,
        unsigned h, unsigned bind)
{
   struct pipe_resource templ = {
      .target = PIPE_TEXTURE_2D,
      .format = format,
      .width0 = w,
      .height0 = h,
      .depth0 = 1,
      .array_size = 1,
      .last_level = 0,
      .nr_samples = 1,
      .nr_storage_samples = 1,
      .usage = PIPE_USAGE_DEFAULT,
      .bind = bind,
   };

   return screen->resource_create(screen, &templ);
}

int
main(void)
{
   const char *node = getenv("AGX_TEST_RENDER_NODE");
   int fd = open(node ? node : "/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
   if (fd < 0) {
      fprintf(stderr, "cannot open render node (is the drm-shim preloaded?)\n");
      return 77;
   }

   /* agx_screen_create() parses driconf unconditionally, so give it the
    * driver's own option descriptions rather than a null cache. */
   static const driOptionDescription asahi_driconf[] = {
#include "driinfo_asahi.h"
   };
   driOptionCache option_info, option_cache;
   driParseOptionInfo(&option_info, asahi_driconf,
                      ARRAY_SIZE(asahi_driconf));

   struct pipe_screen_config config = {
      .options = &option_cache,
      .options_info = &option_info,
   };
   struct pipe_screen *screen = agx_screen_create(fd, NULL, &config);
   if (!screen) {
      fprintf(stderr, "agx_screen_create failed\n");
      return 77;
   }

   struct pipe_context *pctx = screen->context_create(screen, NULL, 0);
   if (!pctx) {
      fprintf(stderr, "context_create failed\n");
      return 1;
   }

   /* Exactly the blit core 983 died on: 4x8, BGRA8_UNORM source, RGBA8_UNORM
    * destination, one level, one layer, nearest. u_blitter turns this into a
    * texel-fetch fragment shader over a pass-through vertex shader and a
    * triangle fan, which is where the null program was consumed. */
   struct pipe_resource *src =
      make_2d(screen, PIPE_FORMAT_B8G8R8A8_UNORM, 4, 8, PIPE_BIND_SAMPLER_VIEW);
   struct pipe_resource *dst =
      make_2d(screen, PIPE_FORMAT_R8G8B8A8_UNORM, 4, 8, PIPE_BIND_RENDER_TARGET);
   if (!src || !dst) {
      fprintf(stderr, "resource_create failed\n");
      return 1;
   }

   struct pipe_blit_info info = {
      .src = {
         .resource = src,
         .level = 0,
         .format = PIPE_FORMAT_B8G8R8A8_UNORM,
         .box = {.x = 0, .y = 0, .z = 0, .width = 4, .height = 8, .depth = 1},
      },
      .dst = {
         .resource = dst,
         .level = 0,
         .format = PIPE_FORMAT_R8G8B8A8_UNORM,
         .box = {.x = 0, .y = 0, .z = 0, .width = 4, .height = 8, .depth = 1},
      },
      .mask = PIPE_MASK_RGBA,
      .filter = PIPE_TEX_FILTER_NEAREST,
      .scissor_enable = false,
      .render_condition_enable = false,
   };

   pctx->blit(pctx, &info);

   struct agx_context *ctx = agx_context(pctx);
   struct agx_device *dev = agx_device(screen);
   const bool agx2 =
      agx_chip_command_stream_generation(dev->chip) == AGX_CS_GENERATION_AGX2;
   const bool vs_ok = ctx->vs != NULL && ctx->vs->bo != NULL;
   const bool fs_ok = ctx->fs != NULL && ctx->fs->bo != NULL;
   const bool linked_ok = ctx->linked.vs != NULL && ctx->linked.fs != NULL;

   printf("blit returned on %s; vs=%s fs=%s linked=%s\n",
          agx_chip_name(dev->chip), vs_ok ? "ok" : "refused",
          fs_ok ? "ok" : "refused", linked_ok ? "ok" : "refused");

   int status = 0;

   /* CommandEncodingNext parts must not take Gallium's AGX2 compute blitter.
    * The render blitter binds a vertex stage; the compute one never does, so
    * a null ctx->vs here means asahi_compute_blit() ran. */
   if (!agx2 && ctx->vs == NULL) {
      fprintf(stderr, "blit took the compute path on a non-AGX2 chip\n");
      status = 1;
   }

   /* The stronger claim, kept separate so a compiler regression cannot hide
    * behind "at least it did not crash". */
   if (!agx2 && getenv("AGX_G17P_BLIT_REQUIRE_ENCODE")) {
      if (!vs_ok || !fs_ok || !linked_ok) {
         fprintf(stderr,
                 "blit shaders did not encode (vs=%d fs=%d linked=%d)\n",
                 vs_ok, fs_ok, linked_ok);
         status = 1;
      }
   }

   pipe_resource_reference(&src, NULL);
   pipe_resource_reference(&dst, NULL);
   pctx->destroy(pctx);
   screen->destroy(screen);
   close(fd);
   return status;
}
