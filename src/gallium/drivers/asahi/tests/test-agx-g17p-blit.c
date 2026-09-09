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

#include "asahi/compiler/agx_compile.h"
#include "asahi/genxml/agx_pack.h"
#include "compiler/nir/nir_builder.h"

#include "agx_public.h"
#include "agx_state.h"

static bool
check_g17p_sampler_bias_record(void)
{
   /* applegpu-nt's HAL200 record for a compare-enabled sampler with a
    * nonzero bias. This is the native evidence that G17P consumes sampler
    * state bias in the descriptor rather than requiring nir_lower_tex to add
    * a second per-instruction bias. The heap stride is 32 bytes; only the
    * first eight are populated. */
   static const uint8_t expected[32] = {
      0x40, 0x00, 0xbe, 0x30, 0x5a, 0xfc, 0x3d, 0xc0,
   };
   uint8_t record[32] = {0};
   agx_pack(record, SAMPLER_HAL200, cfg) {
      cfg.minimum_lod = 1.0f;
      cfg.maximum_lod = 100.0f;
      cfg.lod_bias = -1.015625f;
      cfg.maximum_anisotropy = 8;
      cfg.magnify = AGX_FILTER_LINEAR;
      cfg.minify = AGX_FILTER_NEAREST;
      cfg.mip_filter = AGX_MIP_FILTER_LINEAR;
      cfg.wrap_s = AGX_WRAP_REPEAT;
      cfg.wrap_t = AGX_WRAP_MIRRORED_REPEAT;
      cfg.wrap_r = AGX_WRAP_CLAMP_TO_BORDER;
      cfg.compare_func = AGX_COMPARE_FUNC_LEQUAL;
      cfg.border_colour = AGX_BORDER_COLOUR_OPAQUE_WHITE;
      cfg.pixel_coordinates = true;
      cfg.compare_enable = true;
      cfg.seamful_cube_maps = true;
   }

   if (memcmp(record, expected, sizeof(record)) != 0) {
      fprintf(stderr, "G17P HAL200 sampler bias record changed\n");
      return false;
   }
   return true;
}

static nir_shader *
build_es3_bitops_txf(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options,
      "G17P ES3 bit operations plus texelFetch fragment shader");

   nir_def *frag = nir_load_frag_coord(&b);
   nir_def *word = nir_f2u32(&b, nir_channel(&b, frag, 0));
   nir_def *reversed = nir_bitfield_reverse(&b, word);
   nir_def *count = nir_bit_count(&b, reversed);
   nir_def *top = nir_ufind_msb(&b, nir_ior_imm(&b, reversed, 1));
   nir_def *lod = nir_iand_imm(&b, nir_iadd(&b, count, top), 3);
   nir_def *coord = nir_f2i32(&b, nir_channels(&b, frag, 0x3));

   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 2);
   tex->op = nir_texop_txf;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->dest_type = nir_type_float32;
   tex->coord_components = 2;
   tex->texture_index = 0;
   tex->sampler_index = 0;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord, coord);
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_lod, lod);
   nir_builder_instr_insert(&b, &tex->instr);
   nir_def_init(&tex->instr, &tex->def, 4, 32);

   /* Keep each integer operation live in the colour. This is the GLSL ES
    * 3.x/WebGL2 family behind bitfieldReverse(), bitCount(), findMSB() and a
    * dynamic texelFetch() level, rather than four independent dead probes. */
   nir_def *bit_value = nir_iand_imm(
      &b, nir_ixor(&b, reversed, nir_ishl_imm(&b, count, 3)), 0xff);
   nir_def *scale = nir_fmul_imm(&b, nir_u2f32(&b, bit_value), 1.0f / 255.0f);
   /* This exact f16 -> f32 -> multiply chain is the remaining ext.frag
    * failure from Hyprland's built-in corpus. The widening must stay explicit
    * through the backend optimizer when the post-RA mixed multiply cannot
    * carry the combined producer wait and lifetime hints. */
   nir_def *rounded_half = nir_f2f16(&b, nir_channel(&b, &tex->def, 3));
   nir_def *mixed_product = nir_fmul(
      &b, nir_f2f32(&b, rounded_half), nir_fadd_imm(&b, scale, 1.0f));
   nir_def *colour = nir_fadd(
      &b, &tex->def,
      nir_vec4(&b, scale, scale, scale, mixed_product));
   nir_store_output(&b, colour, nir_imm_int(&b, 0), .base = 0,
                    .write_mask = 0xf,
                    .io_semantics.location = FRAG_RESULT_DATA0);

   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
   BITSET_SET(b.shader->info.textures_used, 0);
   BITSET_SET(b.shader->info.samplers_used, 0);
   return b.shader;
}

static nir_shader *
build_webgl_shadow_fs(enum glsl_sampler_dim dim, bool array)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options,
      dim == GLSL_SAMPLER_DIM_CUBE
         ? (array ? "G17P WebGL cube-array shadow fragment shader"
                  : "G17P WebGL cube shadow fragment shader")
         : (array ? "G17P WebGL 2D-array shadow fragment shader"
                  : "G17P WebGL 2D shadow fragment shader"));
   nir_def *frag = nir_load_frag_coord(&b);

   /* Keep every desktop/WebGL comparison dimension in one API-realistic
    * family. All four forms are byte-grounded on G17P. */
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 2);
   tex->op = nir_texop_tex;
   tex->sampler_dim = dim;
   tex->is_array = array;
   tex->is_shadow = true;
   tex->dest_type = nir_type_float32;
   const unsigned coord_components =
      dim == GLSL_SAMPLER_DIM_CUBE ? (array ? 4 : 3) : (array ? 3 : 2);
   tex->coord_components = coord_components;
   tex->texture_index = 0;
   tex->sampler_index = 0;
   tex->src[0] = nir_tex_src_for_ssa(
      nir_tex_src_coord,
      nir_fmul_imm(&b, nir_channels(&b, frag,
                                    BITFIELD_MASK(coord_components)),
                   1.0f / 64.0f));
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_comparator,
                                     nir_channel(&b, frag, 2));
   nir_builder_instr_insert(&b, &tex->instr);
   nir_def_init(&tex->instr, &tex->def, 4, 32);

   nir_def *colour = nir_replicate(&b, nir_channel(&b, &tex->def, 0), 4);
   nir_store_output(&b, colour, nir_imm_int(&b, 0), .base = 0,
                    .write_mask = 0xf,
                    .io_semantics.location = FRAG_RESULT_DATA0);
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
   BITSET_SET(b.shader->info.textures_used, 0);
   BITSET_SET(b.shader->info.samplers_used, 0);
   return b.shader;
}

static bool
check_es3_compile_contract(struct pipe_context *pctx)
{
   struct pipe_shader_state accepted = {
      .type = PIPE_SHADER_IR_NIR,
      .report_compile_error = true,
      .ir.nir = build_es3_bitops_txf(),
   };
   void *ok = pctx->create_fs_state(pctx, &accepted);
   if (!ok || accepted.error_message) {
      fprintf(stderr, "ES3 bitops/texelFetch shader did not compile: %s\n",
              accepted.error_message ? accepted.error_message : "no shader");
      free(accepted.error_message);
      if (ok)
         pctx->delete_fs_state(pctx, ok);
      return false;
   }
   pctx->delete_fs_state(pctx, ok);

   static const struct {
      enum glsl_sampler_dim dim;
      bool array;
      const char *name;
   } comparisons[] = {
      {GLSL_SAMPLER_DIM_2D, false, "sampler2DShadow"},
      {GLSL_SAMPLER_DIM_2D, true, "sampler2DArrayShadow"},
      {GLSL_SAMPLER_DIM_CUBE, false, "samplerCubeShadow"},
      {GLSL_SAMPLER_DIM_CUBE, true, "samplerCubeArrayShadow"},
   };

   bool compiled = true;
   for (unsigned i = 0; i < ARRAY_SIZE(comparisons); ++i) {
      struct pipe_shader_state shadow = {
         .type = PIPE_SHADER_IR_NIR,
         .report_compile_error = true,
         .ir.nir = build_webgl_shadow_fs(comparisons[i].dim,
                                         comparisons[i].array),
      };
      void *shader = pctx->create_fs_state(pctx, &shadow);
      if (!shader || shadow.error_message) {
         fprintf(stderr, "WebGL2 %s did not compile: %s\n",
                 comparisons[i].name,
                 shadow.error_message ? shadow.error_message : "no shader");
         compiled = false;
      }
      if (shader)
         pctx->delete_fs_state(pctx, shader);
      free(shadow.error_message);
   }
   return compiled;
}

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

static struct pipe_resource *
make_buffer(struct pipe_screen *screen, unsigned size)
{
   struct pipe_resource templ = {
      .target = PIPE_BUFFER,
      .format = PIPE_FORMAT_R8_UNORM,
      .width0 = size,
      .height0 = 1,
      .depth0 = 1,
      .array_size = 1,
      .last_level = 0,
      .nr_samples = 1,
      .nr_storage_samples = 1,
      .usage = PIPE_USAGE_DEFAULT,
      .bind = PIPE_BIND_GLOBAL,
   };

   return screen->resource_create(screen, &templ);
}

static bool
check_direct_resolve_shapes(void)
{
   struct pipe_resource src = {
      .target = PIPE_TEXTURE_2D,
      .format = PIPE_FORMAT_R8G8B8A8_UNORM,
      .width0 = 8,
      .height0 = 8,
      .depth0 = 1,
      .array_size = 1,
      .nr_samples = 2,
      .nr_storage_samples = 2,
      .bind = PIPE_BIND_RENDER_TARGET,
   };
   struct pipe_resource dst = {
      .target = PIPE_TEXTURE_2D,
      .format = PIPE_FORMAT_R8G8B8A8_UNORM,
      .width0 = 8,
      .height0 = 8,
      .depth0 = 1,
      .array_size = 1,
      .nr_samples = 1,
      .nr_storage_samples = 1,
      .bind = PIPE_BIND_RENDER_TARGET,
   };
   struct pipe_blit_info info = {
      .src = {
         .resource = &src,
         .format = PIPE_FORMAT_R8G8B8A8_UNORM,
         .box = {.width = 8, .height = 8, .depth = 1},
      },
      .dst = {
         .resource = &dst,
         .format = PIPE_FORMAT_R8G8B8A8_UNORM,
         .box = {.width = 8, .height = 8, .depth = 1},
      },
      .mask = PIPE_MASK_RGBA,
      .filter = PIPE_TEX_FILTER_NEAREST,
   };

   if (!agx_g17p_direct_resolve_blit_shape(&info, true)) {
      fprintf(stderr, "exact offscreen 2x resolve shape was rejected\n");
      return false;
   }

   src.nr_samples = src.nr_storage_samples = 4;
   if (!agx_g17p_direct_resolve_blit_shape(&info, true)) {
      fprintf(stderr, "exact offscreen 4x resolve shape was rejected\n");
      return false;
   }
   src.nr_samples = src.nr_storage_samples = 2;

   info.src.box.width = info.dst.box.width = 4;
   if (agx_g17p_direct_resolve_blit_shape(&info, true)) {
      fprintf(stderr, "partial offscreen resolve was accepted\n");
      return false;
   }
   info.src.box.width = info.dst.box.width = 8;
   info.dst.format = PIPE_FORMAT_B8G8R8A8_UNORM;
   if (agx_g17p_direct_resolve_blit_shape(&info, true)) {
      fprintf(stderr, "format-changing offscreen resolve was accepted\n");
      return false;
   }
   info.dst.format = PIPE_FORMAT_R8G8B8A8_UNORM;
   dst.target = PIPE_TEXTURE_2D_ARRAY;
   dst.array_size = 2;
   if (agx_g17p_direct_resolve_blit_shape(&info, true)) {
      fprintf(stderr, "array offscreen resolve was accepted\n");
      return false;
   }

   dst.target = PIPE_TEXTURE_2D;
   dst.array_size = 1;
   src.nr_samples = src.nr_storage_samples = 1;
   dst.nr_samples = dst.nr_storage_samples = 2;
   src.bind |= PIPE_BIND_SHARED;
   if (!agx_g17p_direct_resolve_blit_shape(&info, false)) {
      fprintf(stderr, "exact reverse WSI seed shape was rejected\n");
      return false;
   }
   src.bind &= ~PIPE_BIND_SHARED;
   if (agx_g17p_direct_resolve_blit_shape(&info, false)) {
      fprintf(stderr, "offscreen reverse seed shape was accepted\n");
      return false;
   }
   info.src.resource = NULL;
   if (agx_g17p_direct_resolve_blit_shape(&info, true)) {
      fprintf(stderr, "resolve without a source was accepted\n");
      return false;
   }

   return true;
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

   if (!check_g17p_sampler_bias_record())
      return 1;

   if (!check_direct_resolve_shapes())
      return 1;

   if (!check_es3_compile_contract(pctx))
      return 1;
   printf("ES3 bitops/texelFetch and WebGL2 shadow compile contract passed\n");

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

   /* G17P's one admitted precompiled kernel is the vector buffer fill used
    * by clear_buffer. It feeds later graphics, so AGX_BARRIER_ALL must survive
    * native CDM lowering as the strong flags-5 cache barrier. The shim audits
    * the actual production stream and its nested Ioto argument source. */
   struct pipe_resource *filled = make_buffer(screen, 256);
   const uint32_t fill[4] = {
      UINT32_C(0x11223344), UINT32_C(0x55667788),
      UINT32_C(0x99aabbcc), UINT32_C(0xddeeff00),
   };
   if (!filled) {
      fprintf(stderr, "buffer resource_create failed\n");
      return 1;
   }
   pctx->clear_buffer(pctx, filled, 0, 256, fill, sizeof(fill));
   pctx->flush(pctx, NULL, 0);

   struct agx_context *ctx = agx_context(pctx);
   struct agx_device *dev = agx_device(screen);
   const bool agx2 =
      agx_chip_command_stream_generation(dev->chip) == AGX_CS_GENERATION_AGX2;
   int status = 0;

   /* The state tracker must not insert this same ungrounded render blit for
    * BGRA default-surface readback on AGX3.  The explicit pctx->blit above is
    * retained as a compiler/stream ratchet, while API transfers take the
    * mapped CPU conversion path until the pixel result is hardware-proven. */
   if (!agx2 &&
       screen->caps.texture_transfer_modes != PIPE_TEXTURE_TRANSFER_DEFAULT) {
      fprintf(stderr, "AGX3 still advertises blit-based texture transfers\n");
      status = 1;
   }

   const bool vs_ok = ctx->vs != NULL && ctx->vs->bo != NULL;
   const bool fs_ok = ctx->fs != NULL && ctx->fs->bo != NULL;
   const bool linked_ok = ctx->linked.vs != NULL && ctx->linked.fs != NULL;

   printf("blit returned on %s; vs=%s fs=%s linked=%s\n",
          agx_chip_name(dev->chip), vs_ok ? "ok" : "refused",
          fs_ok ? "ok" : "refused", linked_ok ? "ok" : "refused");

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
   pipe_resource_reference(&filled, NULL);
   pctx->destroy(pctx);
   screen->destroy(screen);
   close(fd);
   return status;
}
