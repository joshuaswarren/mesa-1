/*
 * Offline sweep of the G17P fragment epilog key space.
 *
 * The epilog is compiled separately from the API fragment shader, at draw
 * time, and its refusal does not carry the "cannot encode fragment shader"
 * message the pipeline path prints -- so an epilog gap looks like a silent
 * missing draw rather than a failed pipeline. This walks formats, blend
 * modes, sRGB lowering, alpha-to-coverage/one and the link flags. Each alpha
 * mode has its own refusal ceiling so losing the whole coverage path cannot
 * hide behind an improvement elsewhere in the matrix.
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

static bool
check_rgb565_blend(void)
{
   struct agx_fs_epilog_key key = {
      .link.loc_written = BITFIELD_BIT(0),
      .link.size_32 = BITFIELD_BIT(0),
      .nr_samples = 1,
   };
   agx_fs_epilog_apply_device_policy(AGX_CHIP_G17P, &key);
   memset(key.remap, -1, sizeof(key.remap));
   key.remap[0] = 0;
   key.rt_formats[0] = PIPE_FORMAT_B5G6R5_UNORM;
   key.blend.rt[0].colormask = 0xf;
   key.blend.rt[0].mode = agx_pack_blend_standard(
      PIPE_BLEND_ADD, PIPE_BLENDFACTOR_SRC_ALPHA,
      PIPE_BLENDFACTOR_INV_SRC_ALPHA, PIPE_BLEND_ADD,
      PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO);

   nir_builder b = nir_builder_init_simple_shader(
      0, &agx_nir_options, "G17P RGB565 source-alpha epilog");
   agx_nir_fs_epilog(&b, &key);

   unsigned loads = 0, stores = 0;
   unsigned load_components = 0, load_bits = 0, load_format = 0;
   unsigned store_mask = 0, store_format = 0;
   bool lowering_ok = true;
   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_load_local_pixel_agx) {
               loads++;
               load_components = intr->def.num_components;
               load_bits = intr->def.bit_size;
               load_format = nir_intrinsic_format(intr);
               lowering_ok &= intr->def.num_components == 3 &&
                              intr->def.bit_size == 16 &&
                              nir_intrinsic_format(intr) ==
                                 PIPE_FORMAT_R16_UINT;
            } else if (intr->intrinsic ==
                       nir_intrinsic_store_local_pixel_agx) {
               stores++;
               store_mask = nir_intrinsic_write_mask(intr);
               store_format = nir_intrinsic_format(intr);
               lowering_ok &= nir_intrinsic_write_mask(intr) == 0x7 &&
                              nir_intrinsic_format(intr) ==
                                 PIPE_FORMAT_R16_FLOAT;
            }
         }
      }
   }
   lowering_ok &= loads == 1 && stores == 1;

   agx_nir_lower_g17p_1x_tile_samples(b.shader);
   agx_preprocess_nir(b.shader);
   struct agx_shader_key backend_key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .secondary = true,
      .no_preamble = true,
   };
   struct agx_shader_part part = {0};
   const bool compiled = agx_compile_shader_nir(b.shader, &backend_key, &part);
   static const uint8_t apple_rgb565_load[] = {
      0x67,0x0e,0x54,0x00,0x00,0x00,0x01,0x9a,0x02,0x00,0x00,0x00,
   };
   bool exact_load = false;
   for (unsigned offset = 0;
        part.binary && offset + sizeof(apple_rgb565_load) <=
                          part.info.binary_size;
        ++offset) {
      if (!memcmp((const uint8_t *)part.binary + offset, apple_rgb565_load,
                  sizeof(apple_rgb565_load))) {
         exact_load = true;
         break;
      }
   }

   const bool ok = lowering_ok && compiled && exact_load;
   if (!ok) {
      fprintf(stderr,
              "G17P RGB565 source-alpha epilog failed: lowering=%u "
              "loads=%u(%u/%u/%u) stores=%u(%u/%u) compiled=%u "
              "exact_load=%u\n",
              lowering_ok, loads, load_components, load_bits, load_format,
              stores, store_mask, store_format, compiled, exact_load);
   }
   free(part.binary);
   ralloc_free(b.shader);
   return ok;
}

int
main(void)
{
   if (!check_rgb565_blend())
      return 1;

   const bool verbose = getenv("AGX_G17P_EPILOG_VERBOSE") != NULL;
   unsigned refused = 0, total = 0;
   unsigned refused_alpha[2][2] = {{0}}, total_alpha[2][2] = {{0}};
   for (unsigned f = 0; f < ARRAY_SIZE(formats); ++f) {
      for (unsigned bl = 0; bl < ARRAY_SIZE(blends); ++bl) {
         for (unsigned srgb = 0; srgb < 2; ++srgb) {
            for (unsigned s32 = 1; s32 < 2; ++s32) {
               for (unsigned w1 = 0; w1 < 1024; ++w1) {
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
                     .blend.alpha_to_coverage = (w1 >> 8) & 1,
                     .blend.alpha_to_one = (w1 >> 9) & 1,
                     .g17p_tile_formats = srgb != 0,
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
                  ++total_alpha[key.blend.alpha_to_coverage]
                               [key.blend.alpha_to_one];
                  if (verbose) {
                     printf("### %s blend=%s srgb=%u size32=%u flags=0x%03x "
                            "(w1=%u z=%u s=%u ranzs=%u sampsh=%u "
                            "samples=%u smalltile=%u zeromask=%u atoc=%u "
                            "ato1=%u)\n",
                            formats[f].name, blends[bl].name, srgb, s32, w1,
                            w1 & 1, (w1 >> 1) & 1, (w1 >> 2) & 1,
                            (w1 >> 3) & 1, (w1 >> 4) & 1,
                            (w1 >> 5) & 1 ? 4 : 1, (w1 >> 6) & 1,
                            (w1 >> 7) & 1, (w1 >> 8) & 1,
                            (w1 >> 9) & 1);
                     fflush(stdout);
                  }
                  if (!agx_compile_shader_nir(b.shader, &bk, &part)) {
                     ++refused;
                     ++refused_alpha[key.blend.alpha_to_coverage]
                                    [key.blend.alpha_to_one];
                     if (verbose)
                        printf("    REFUSED\n");
                  }
                  if (verbose)
                     fflush(stdout);
                  free(part.binary);
                  ralloc_free(b.shader);
               }
            }
         }
      }
   }
   printf("total=%u refused=%u\n", total, refused);
   for (unsigned atoc = 0; atoc < 2; ++atoc) {
      for (unsigned ato1 = 0; ato1 < 2; ++ato1) {
         printf("atoc=%u ato1=%u total=%u refused=%u accepted=%u\n", atoc,
                ato1, total_alpha[atoc][ato1], refused_alpha[atoc][ato1],
                total_alpha[atoc][ato1] - refused_alpha[atoc][ato1]);
      }
   }

   /* Keeping G17P fsat as its native standalone f32add.ep2 closes 416
    * producer shapes without inventing saturated forms for those producers. */
   /* The multisample sample mask is now the register operand Apple's own
    * 2x/4x fragments use (a live coverage read), so every 2x and 4x epilogue
    * shape that the fabricated "coverage selector" immediate had to refuse
    * now encodes: 110848 refusals -> 93952. */
   /* Apple's RGB565 framebuffer-fetch form grounds the W16 mask7 tile load,
    * closing the corresponding single-sample blend epilogs. */
   const unsigned ceiling = 93952;
   if (refused > ceiling) {
      fprintf(stderr,
              "G17P epilog matrix regressed: %u refusals, ceiling %u\n",
              refused, ceiling);
      return 1;
   }
   const unsigned alpha_ceiling[2][2] = {
      {23488, 23488},
      {23488, 23488},
   };
   for (unsigned atoc = 0; atoc < 2; ++atoc) {
      for (unsigned ato1 = 0; ato1 < 2; ++ato1) {
         if (refused_alpha[atoc][ato1] > alpha_ceiling[atoc][ato1]) {
            fprintf(stderr,
                    "G17P epilog alpha mode %u/%u regressed: %u refusals, "
                    "ceiling %u\n",
                    atoc, ato1, refused_alpha[atoc][ato1],
                    alpha_ceiling[atoc][ato1]);
            return 1;
         }
      }
   }
   if (refused_alpha[1][0] == total_alpha[1][0] ||
       refused_alpha[1][1] == total_alpha[1][1]) {
      fprintf(stderr, "G17P alpha-to-coverage epilogs all refused\n");
      return 1;
   }
   return 0;
}
