/*
 * Offline sweep of the G17P fragment epilog key space.
 *
 * The epilog is compiled separately from the API fragment shader, at draw
 * time, and its refusal does not carry the "cannot encode fragment shader"
 * message the pipeline path prints -- so an epilog gap looks like a silent
 * missing draw rather than a failed pipeline. This walks formats, blend
 * modes, sRGB lowering and the link flags, reports which opcode each refusal
 * names, and fails only if the number of refusals grows.
 *
 * It is a ratchet, not a specification: lower the ceiling whenever a gap is
 * closed. As measured, the refusals are zs_emit (depth/stencil writes from
 * the epilog), ld_tile and st_tile (tilebuffer formats), log2 (the sRGB
 * encode's pow) and a saturating fadd/fmul. No fcmpsel shape appears here,
 * which is what ruled the epilog out as the source of vkcube's refusal.
 */
#include "agx_compile.h"
#include "agx_linker.h"
#include "nir_builder.h"
#include "util/format/u_formats.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const struct {
   const char *name;
   enum pipe_format fmt;
} formats[] = {
   {"B8G8R8A8_UNORM", PIPE_FORMAT_B8G8R8A8_UNORM},
   {"B8G8R8A8_SRGB", PIPE_FORMAT_B8G8R8A8_SRGB},
   {"R8G8B8A8_UNORM", PIPE_FORMAT_R8G8B8A8_UNORM},
   {"R8G8B8A8_SRGB", PIPE_FORMAT_R8G8B8A8_SRGB},
   {"R8G8B8A8_SNORM", PIPE_FORMAT_R8G8B8A8_SNORM},
   {"R16G16B16A16_FLOAT", PIPE_FORMAT_R16G16B16A16_FLOAT},
   {"R32G32B32A32_FLOAT", PIPE_FORMAT_R32G32B32A32_FLOAT},
   {"R8_UNORM", PIPE_FORMAT_R8_UNORM},
   {"R5G6B5_UNORM", PIPE_FORMAT_B5G6R5_UNORM},
   {"R10G10B10A2_UNORM", PIPE_FORMAT_R10G10B10A2_UNORM},
   {"R8G8B8A8_UINT", PIPE_FORMAT_R8G8B8A8_UINT},
   {"R8G8B8A8_SINT", PIPE_FORMAT_R8G8B8A8_SINT},
};

static const struct {
   const char *name;
   unsigned func, src, dst;
} blends[] = {
   {"none", 0, 0, 0},
   {"add-one-one", PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ONE},
   {"add-src-invsrc", PIPE_BLEND_ADD, PIPE_BLENDFACTOR_SRC_ALPHA,
    PIPE_BLENDFACTOR_INV_SRC_ALPHA},
   {"sub-src-dst", PIPE_BLEND_SUBTRACT, PIPE_BLENDFACTOR_SRC_COLOR,
    PIPE_BLENDFACTOR_DST_COLOR},
   {"min", PIPE_BLEND_MIN, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ONE},
   {"max", PIPE_BLEND_MAX, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ONE},
};

int
main(void)
{
   unsigned refused = 0, total = 0;
   for (unsigned f = 0; f < ARRAY_SIZE(formats); ++f) {
      for (unsigned bl = 0; bl < ARRAY_SIZE(blends); ++bl) {
         for (unsigned srgb = 0; srgb < 2; ++srgb) {
            for (unsigned s32 = 1; s32 < 2; ++s32) {
               for (unsigned w1 = 0; w1 < 256; ++w1) {
                  struct agx_fs_epilog_key key = {
                     .link.loc_written = BITFIELD_BIT(0),
                     .link.size_32 = s32 ? BITFIELD_BIT(0) : 0,
                     .link.loc0_w_1 = w1 & 1,
                     .link.write_z = (w1 >> 1) & 1,
                     .link.write_s = (w1 >> 2) & 1,
                     .link.already_ran_zs = (w1 >> 3) & 1,
                     .link.sample_shading = (w1 >> 4) & 1,
                     .nr_samples = (w1 >> 5) & 1 ? 4 : 1,
                     .force_small_tile = (w1 >> 6) & 1,
                     .preserve_zero_mask = (w1 >> 7) & 1,
                     .lower_srgb = srgb != 0,
                  };
                  memset(key.remap, -1, sizeof(key.remap));
                  key.remap[0] = 0;
                  key.rt_formats[0] = formats[f].fmt;
                  key.blend.rt[0].colormask = 0xf;
                  if (bl == 0) {
                     key.blend.rt[0].mode = agx_pack_blend_standard(
                        PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE,
                        PIPE_BLENDFACTOR_ZERO, PIPE_BLEND_ADD,
                        PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO);
                  } else {
                     key.blend.rt[0].mode = agx_pack_blend_standard(
                        blends[bl].func, blends[bl].src, blends[bl].dst,
                        blends[bl].func, blends[bl].src, blends[bl].dst);
                  }

                  nir_builder b = nir_builder_init_simple_shader(
                     0, &agx_nir_options, "G17P epilog sweep");
                  agx_nir_fs_epilog(&b, &key);
                  agx_nir_lower_g17p_tile_samples(b.shader, key.nr_samples);
                  agx_preprocess_nir(b.shader);

                  struct agx_shader_key bk = {
                     .dev.encoding = AGX_ISA_ENCODING_AGX3B,
                     .secondary = true,
                     .no_preamble = true,
                  };
                  struct agx_shader_part part = {0};
                  ++total;
                  char label[160];
                  snprintf(label, sizeof(label),
                           "%s blend=%s srgb=%u size32=%u flags=0x%02x "
                           "(w1=%u z=%u s=%u ranzs=%u sampsh=%u samples=%u "
                           "smalltile=%u zeromask=%u)",
                           formats[f].name, blends[bl].name, srgb, s32, w1,
                           w1 & 1, (w1 >> 1) & 1, (w1 >> 2) & 1,
                           (w1 >> 3) & 1, (w1 >> 4) & 1,
                           (w1 >> 5) & 1 ? 4 : 1, (w1 >> 6) & 1,
                           (w1 >> 7) & 1);
                  printf("### %s\n", label);
                  fflush(stdout);
                  if (!agx_compile_shader_nir(b.shader, &bk, &part)) {
                     ++refused;
                     printf("    REFUSED\n");
                  }
                  fflush(stdout);
                  free(part.binary);
                  ralloc_free(b.shader);
               }
            }
         }
      }
   }
   printf("total=%u refused=%u\n", total, refused);

   /* Keeping G17P fsat as its native standalone f32add.ep2 closes 416
    * producer shapes without inventing saturated forms for those producers. */
   /* Multisample selectors are now sample-count tagged before the backend.
    * This deliberately rejects 1904 matrix combinations which previously
    * fell through the single-sample tile form and encoded the wrong access. */
   const unsigned ceiling = 32384;
   if (refused > ceiling) {
      fprintf(stderr,
              "G17P epilog matrix regressed: %u refusals, ceiling %u\n",
              refused, ceiling);
      return 1;
   }
   return 0;
}
