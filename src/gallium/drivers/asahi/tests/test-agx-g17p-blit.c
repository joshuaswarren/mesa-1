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
 * BGRA8_UNORM -> RGBA8_UNORM region with uses_txf. The cause is that the AGX3
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
#include "util/u_inlines.h"
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
   /* The native hardware oracle uses a 2408x1506 target. Unlike the old 4x8
    * fixture, that extent is compression-eligible, so it also reproduces the
    * real end-of-tile PBE descriptor path while the blit rectangle stays 4x8.
    */
   struct pipe_resource *dst = make_2d(screen, PIPE_FORMAT_R8G8B8A8_UNORM,
                                       2408, 1506, PIPE_BIND_RENDER_TARGET);
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

   /* A clear and a flush first: no API shader is involved, so this is the
    * background/end-of-tile programs, the PBE render-target descriptor and
    * the batch submit path on their own. It used to abort the process in
    * agx_native_descriptor_rejected("end-of-tile PBE") before any draw
    * happened at all.
    */
   struct pipe_surface cbuf = {
      .texture = dst,
      .format = PIPE_FORMAT_R8G8B8A8_UNORM,
      .first_layer = 0,
      .last_layer = 0,
      .level = 0,
   };
   struct pipe_framebuffer_state fb = {
      .width = 4,
      .height = 8,
      .layers = 1,
      .samples = 1,
      .nr_cbufs = 1,
   };
   fb.cbufs[0] = cbuf;
   pctx->set_framebuffer_state(pctx, &fb);

   const union pipe_color_union colour = {.f = {0.25f, 0.5f, 0.75f, 1.0f}};
   pctx->clear(pctx, PIPE_CLEAR_COLOR0, 0xf, 0, NULL, &colour, 0.0, 0);
   pctx->flush(pctx, NULL, 0);
   printf("clear and flush completed\n");

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

   if (!agx2 && agx_resource(dst)->layout.compressed) {
      fprintf(stderr,
              "AGX3 render target selected an unencodable compressed PBE\n");
      status = 1;
   }

   /* CommandEncodingNext parts must not take Gallium's AGX2 compute blitter.
    * The render blitter binds a vertex stage; the compute one never does, so
    * a null ctx->vs here means asahi_compute_blit() ran. */
   if (!agx2 && ctx->vs == NULL) {
      fprintf(stderr, "blit took the compute path on a non-AGX2 chip\n");
      status = 1;
   }

   /* A ratchet on what already works. u_blitter's pass-through vertex shader
    * moves a 16-bit uniform, which had no encoding until the AGX3b uniform
    * copy landed; if this regresses, every blit loses its vertex program
    * again. The fragment program has its own encode ratchet below so the
    * crash-safety check remains independently useful.
    */
   if (!agx2 && !vs_ok) {
      fprintf(stderr, "the blit's vertex program no longer encodes\n");
      status = 1;
   }

   /* Whatever the shaders did, a refused program must have been counted as a
    * skipped draw rather than silently rendering nothing. */
   if (!fs_ok && ctx->skipped_no_program == 0) {
      fprintf(stderr, "a refused program did not skip its draw\n");
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
