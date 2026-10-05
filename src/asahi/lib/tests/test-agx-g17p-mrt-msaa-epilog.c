/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/* G17P multi-target multisample fragment epilog: sweep the key space and
 * report which configurations the AGX3 packer refuses.
 *
 * Two known refusing points anchor the sweep, both transcribed from real
 * driver failure lines on Neo:
 *
 *   uniform 4-target (renderpasses.*.multisample_resolve.*):
 *     link=0x400f0f00 samples=2 ... formats=54/54/54/54 remap=0/1/2/3
 *   mixed 3-target (pipeline.*.multisample.misc.* / MSRTSS):
 *     link=0x09070700 samples=2 ... formats=34/54/38/0 remap=0/1/2/3
 *
 * 0x54 is PIPE_FORMAT_R8G8B8A8_UNORM.  Device policy (preserve_zero_mask,
 * g17p_tile_formats) comes from agx_fs_epilog_apply_device_policy() rather than being
 * assigned here, so the swept keys are ones the driver actually emits.
 */

#include "agx_compile.h"
#include "agx_chip.h"
#include "agx_linker.h"
#include "agx_nir.h"
#include "agx_tilebuffer.h"
#include "nir_builder.h"
#include "util/format/u_formats.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t
opaque_blend(void)
{
   return agx_pack_blend_standard(PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE,
                                  PIPE_BLENDFACTOR_ZERO, PIPE_BLEND_ADD,
                                  PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO);
}

/* Mirror hk_get_prolog_epilog_locked()'s G17P epilog path exactly. */
static bool
compile_epilog(const enum pipe_format fmts[8], unsigned nr_samples,
               bool sample_mask_after_force_early, const int8_t remap[8],
               unsigned written_mask, bool emrt, struct agx_fs_epilog_key *out)
{
   unsigned mask = 0;
   for (unsigned rt = 0; rt < 8; ++rt)
      if (fmts[rt] != PIPE_FORMAT_NONE)
         mask |= BITFIELD_BIT(rt);

   if (written_mask)
      mask = written_mask;
   struct agx_fs_epilog_key key = {
      .link =
         {
            .size_32 = mask,
            .loc_written = mask,
            .sample_mask_after_force_early = sample_mask_after_force_early,
         },
      .nr_samples = nr_samples,
      /* hk_cmd_draw.c:5559-5561 carries the render pass's limits in the key;
       * agx_nir_fs_epilog() builds its own tilebuffer layout from them at
       * agx_nir_prolog_epilog.c:629.  Leaving them zero silently selects the
       * 64-byte default and spills targets the driver keeps resident. */
      .max_sample_size_B = AGX_G17P_MAX_BYTES_PER_SAMPLE,
      .max_bytes_per_tile = 32768,
      .max_samples_per_tile = 2048,
   };
   agx_fs_epilog_apply_device_policy(AGX_CHIP_G17P, &key);

   memset(key.remap, -1, sizeof(key.remap));
   for (unsigned rt = 0; rt < 8; ++rt) {
      if (fmts[rt] == PIPE_FORMAT_NONE)
         continue;
      key.remap[rt] = remap ? remap[rt] : (int8_t)rt;
      key.rt_formats[rt] = fmts[rt];
      key.blend.rt[rt].colormask = 0xf;
      key.blend.rt[rt].mode = opaque_blend();
   }
   *out = key;

   /* Exactly what hk_cmd_draw.c:3490 calls: the limits travel in the epilog
    * key, and plain agx_build_tilebuffer_layout() defaults them to a 64-byte
    * budget the driver never uses on G17P. */
   const struct agx_tilebuffer_layout tib =
      agx_build_tilebuffer_layout_with_limits(
         key.rt_formats, ARRAY_SIZE(key.rt_formats), nr_samples, true,
         AGX_G17P_MAX_BYTES_PER_SAMPLE, 32768, 2048);

   struct agx_shader_key backend_key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .fs.nr_samples = nr_samples,
      .secondary = true,
      .g17p_emrt = emrt,
   };
   unsigned partitions = 0;
   for (unsigned rt = 0; rt < ARRAY_SIZE(tib.logical_format); ++rt) {
      if (tib.logical_format[rt] != PIPE_FORMAT_NONE && !tib.spilled[rt]) {
         BITSET_SET(backend_key.g17p_tile_partitions, tib._offset_B[rt]);
         partitions++;
      }
   }
   if (partitions < 2)
      BITSET_ZERO(backend_key.g17p_tile_partitions);

   nir_builder b = nir_builder_init_simple_shader(0, &agx_nir_options, NULL);
   struct agx_fs_epilog_key pixel = key;
   pixel.defer_monolithic_msaa =
      key.blend.alpha_to_coverage && key.nr_samples > 1;
   agx_nir_fs_epilog(&b, &pixel);
   NIR_PASS(_, b.shader, agx_nir_lower_g17p_tile_samples, key.nr_samples);
   agx_preprocess_nir(b.shader);
   NIR_PASS(_, b.shader, agx_nir_lower_g17p_tile_samples, key.nr_samples);

   unsigned max_off = 0;
   for (unsigned rt = 0; rt < ARRAY_SIZE(tib.logical_format); ++rt)
      if (tib.logical_format[rt] != PIPE_FORMAT_NONE && !tib.spilled[rt])
         max_off = MAX2(max_off, tib._offset_B[rt]);
   unsigned live = 0, nspill = 0;
   for (unsigned rt = 0; rt < ARRAY_SIZE(tib.logical_format); ++rt) {
      if (tib.logical_format[rt] == PIPE_FORMAT_NONE) continue;
      if (tib.spilled[rt]) nspill++; else live++;
   }
   printf("TIB ss=%u maxoff=%u live=%u nspill=%u parts=%u ",
          tib.sample_size_B, max_off, live, nspill, partitions);

   struct agx_shader_part part = {0};
   const bool ok = agx_compile_shader_nir(b.shader, &backend_key, &part);
   free(part.binary);
   ralloc_free(b.shader);
   return ok;
}

struct fmt_name { enum pipe_format f; const char *n; };
static const struct fmt_name kFormats[] = {
   {PIPE_FORMAT_R8G8B8A8_UNORM, "rgba8un"},
   {PIPE_FORMAT_R16G16B16A16_FLOAT, "rgba16f"},
   {PIPE_FORMAT_R16G16B16A16_SINT, "rgba16i"},
   {PIPE_FORMAT_R32G32B32A32_UINT, "rgba32u"},
};
#define NFMT ARRAY_SIZE(kFormats)

static const char *
fname(enum pipe_format f)
{
   for (unsigned i = 0; i < NFMT; ++i)
      if (kFormats[i].f == f)
         return kFormats[i].n;
   return "none";
}

static bool g_emrt;

static void
probe(const char *label, const enum pipe_format f[4], unsigned samples,
      unsigned written_mask)
{
   struct agx_fs_epilog_key key;
   const bool ok = compile_epilog(f, samples, false, NULL, written_mask, g_emrt, &key);
   printf("%-44s samples=%u mask=0x%02x -> %s\n", label, samples,
          written_mask, ok ? "compiled" : "REFUSED");
}

int
main(void)
{
   printf("=== block count vs attachment count (r32g32b32a32_uint, 1 sample, emrt=1) ===\n");
   g_emrt = true;
   for (unsigned n = 1; n <= 8; ++n) {
      enum pipe_format f[8] = {0};
      for (unsigned i = 0; i < n; ++i)
         f[i] = PIPE_FORMAT_R32G32B32A32_UINT;
      char lbl[64];
      snprintf(lbl, sizeof(lbl), "%u attachments", n);
      probe(lbl, f, 1, (1u << n) - 1);
   }
   printf("\n=== same, rgba8 (4 B/sample each) ===\n");
   for (unsigned n = 1; n <= 8; ++n) {
      enum pipe_format f[8] = {0};
      for (unsigned i = 0; i < n; ++i)
         f[i] = PIPE_FORMAT_R8G8B8A8_UNORM;
      char lbl[64];
      snprintf(lbl, sizeof(lbl), "%u attachments rgba8", n);
      probe(lbl, f, 1, (1u << n) - 1);
   }
   return 0;
}
