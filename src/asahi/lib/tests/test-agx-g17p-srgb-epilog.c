/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_compile.h"
#include "agx_linker.h"
#include "nir_builder.h"
#include "util/format/u_formats.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* G17P keeps an 8-bit sRGB target as U8NORM bytes in the tile. Colour reads
 * and writes keep the sRGB format so the backend converts with the native
 * normflt/fltnorm sRGB modes; no transfer-function arithmetic may appear in
 * NIR. Only the destination-preserving copy moves raw U8NORM bytes. */
static bool
check_lowering(nir_shader *shader, enum pipe_format expected)
{
   unsigned loads = 0, stores = 0, transcendental = 0;

   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_alu) {
               const nir_op op = nir_instr_as_alu(instr)->op;
               transcendental += op == nir_op_fpow || op == nir_op_flog2 ||
                                 op == nir_op_fexp2;
               continue;
            }

            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_load_local_pixel_agx) {
               loads++;
               if (nir_intrinsic_format(intr) != expected)
                  return false;
            } else if (intr->intrinsic == nir_intrinsic_store_local_pixel_agx) {
               stores++;
               if (nir_intrinsic_format(intr) != expected)
                  return false;
            }
         }
      }
   }

   return loads == 1 && stores == 1 && transcendental == 0;
}

static bool
compile_epilog(nir_shader *shader, const char *name, struct agx_shader_part *part)
{
   agx_preprocess_nir(shader);
   struct agx_shader_key backend_key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .secondary = true,
      .no_preamble = true,
   };
   memset(part, 0, sizeof(*part));
   const bool ok = agx_compile_shader_nir(shader, &backend_key, part);
   if (!ok)
      fprintf(stderr, "%s did not compile\n", name);
   return ok;
}

static nir_shader *
build_epilog(struct agx_fs_epilog_key *key, enum pipe_format format,
             const char *name)
{
   key->rt_formats[0] = format;
   nir_builder b = nir_builder_init_simple_shader(0, &agx_nir_options, "%s",
                                                  name);
   agx_nir_fs_epilog(&b, key);
   return b.shader;
}

/* Apple's sRGB and linear RGBA8 fragment programs differ only in the
 * conversion-mode fields of the packs (agx3_fltnorm_core.h): same length,
 * different bytes. */
static bool
same_shape_as_linear(struct agx_fs_epilog_key *key, enum pipe_format srgb,
                     enum pipe_format linear, const char *name)
{
   struct agx_shader_part s_part, l_part;
   nir_shader *s = build_epilog(key, srgb, name);
   /* Every 8-bit sRGB format renders through the SRGBA8 tile format. */
   if (!check_lowering(s, PIPE_FORMAT_R8G8B8A8_SRGB)) {
      fprintf(stderr, "%s: software sRGB conversion or wrong tile format\n",
              name);
      return false;
   }
   bool ok = compile_epilog(s, name, &s_part);
   nir_shader *l = build_epilog(key, linear, name);
   ok &= compile_epilog(l, name, &l_part);
   /* An alpha-only write need not convert anything, so only require a
    * difference when a colour channel is written. */
   const bool converts = key->blend.rt[0].colormask & 0x7;
   const bool identical =
      ok && s_part.info.main_size == l_part.info.main_size &&
      !memcmp((const uint8_t *)s_part.binary + s_part.info.main_offset,
              (const uint8_t *)l_part.binary + l_part.info.main_offset,
              s_part.info.main_size);
   if (ok && (s_part.info.main_size != l_part.info.main_size ||
              (converts && identical))) {
      fprintf(stderr, "%s: sRGB epilog is %u bytes vs linear %u "
              "(identical bytes: %d)\n", name, s_part.info.main_size,
              l_part.info.main_size, identical);
      ok = false;
   }
   free(s_part.binary);
   free(l_part.binary);
   ralloc_free(s);
   ralloc_free(l);
   return ok;
}

int
main(void)
{
   struct agx_fs_epilog_key policy = {
      .g17p_tile_formats = true,
      .preserve_zero_mask = true,
   };
   agx_fs_epilog_apply_device_policy(AGX_CHIP_G13G, &policy);
   if (policy.g17p_tile_formats || policy.preserve_zero_mask) {
      fprintf(stderr, "legacy device inherited G17P epilog policy\n");
      return 1;
   }
   agx_fs_epilog_apply_device_policy(AGX_CHIP_G17P, &policy);
   if (!policy.g17p_tile_formats || !policy.preserve_zero_mask) {
      fprintf(stderr, "G17P framebuffer epilog policy is incomplete\n");
      return 1;
   }
   agx_fs_epilog_apply_device_policy(AGX_CHIP_G17P, NULL);

   struct agx_fs_epilog_key key = {
      .link.loc_written = BITFIELD_BIT(0),
      .link.size_32 = BITFIELD_BIT(0),
      .nr_samples = 1,
   };
   agx_fs_epilog_apply_device_policy(AGX_CHIP_G17P, &key);
   memset(key.remap, -1, sizeof(key.remap));
   key.remap[0] = 0;
   key.blend.rt[0].colormask = 0xf;
   key.blend.rt[0].mode =
      agx_pack_blend_standard(PIPE_BLEND_ADD, PIPE_BLENDFACTOR_SRC_ALPHA,
                              PIPE_BLENDFACTOR_INV_SRC_ALPHA, PIPE_BLEND_ADD,
                              PIPE_BLENDFACTOR_SRC_ALPHA,
                              PIPE_BLENDFACTOR_INV_SRC_ALPHA);

   /* Blending reads, converts and writes back every channel. */
   bool ok = same_shape_as_linear(&key, PIPE_FORMAT_R8G8B8A8_SRGB,
                                  PIPE_FORMAT_R8G8B8A8_UNORM,
                                  "G17P RGBA8 sRGB blend epilog");
   ok &= same_shape_as_linear(&key, PIPE_FORMAT_B8G8R8A8_SRGB,
                              PIPE_FORMAT_B8G8R8A8_UNORM,
                              "G17P BGRA8 sRGB blend epilog");
   ok &= same_shape_as_linear(&key, PIPE_FORMAT_R8G8_SRGB,
                              PIPE_FORMAT_R8G8_UNORM,
                              "G17P RG8 sRGB blend epilog");
   ok &= same_shape_as_linear(&key, PIPE_FORMAT_R8_SRGB,
                              PIPE_FORMAT_R8_UNORM,
                              "G17P R8 sRGB blend epilog");

   /* Partial colour masks compact the stored channels: (G, A) is an encoded
    * lane over a linear alpha lane, (G, B) two encoded lanes, A alone linear. */
   static const unsigned masks[] = {0xa, 0x6, 0x8, 0x7, 0xd};
   for (unsigned i = 0; i < sizeof(masks) / sizeof(masks[0]); ++i) {
      key.blend.rt[0].colormask = masks[i];
      char name[64];
      snprintf(name, sizeof(name), "G17P sRGB blend epilog mask %#x", masks[i]);
      ok &= same_shape_as_linear(&key, PIPE_FORMAT_R8G8B8A8_SRGB,
                                 PIPE_FORMAT_R8G8B8A8_UNORM, name);
   }

   /* A disabled colour mask must preserve the raw encoded tile. It needs one
    * real U8 load/store pair but no sRGB round trip. */
   key.blend.rt[0].colormask = 0;
   key.blend.rt[0].mode =
      agx_pack_blend_standard(PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE,
                              PIPE_BLENDFACTOR_ZERO, PIPE_BLEND_ADD,
                              PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO);
   nir_shader *masked = build_epilog(&key, PIPE_FORMAT_R8G8B8A8_SRGB,
                                     "G17P masked sRGB epilog");
   if (!check_lowering(masked, PIPE_FORMAT_R8_UNORM)) {
      fprintf(stderr, "G17P zero-mask epilog did not preserve the encoded tile\n");
      return 1;
   }
   struct agx_shader_part part;
   ok &= compile_epilog(masked, "G17P zero-mask sRGB epilog", &part);
   free(part.binary);
   ralloc_free(masked);
   return ok ? 0 : 1;
}
