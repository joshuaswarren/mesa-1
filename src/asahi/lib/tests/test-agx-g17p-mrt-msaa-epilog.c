/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/* A four-target multisample epilog: the shape every
 * dEQP-VK.renderpasses.*.multisample_resolve.*.samples_N case builds, and the
 * shape the AGX3 packer refused.  The key below is transcribed from the
 * driver's own failure line on Neo:
 *
 *   fragment epilog compilation failed: link=0x400f0f00 samples=2 shaded=0
 *   stop=1 small=0 preserve-zero=1 srgb=1 formats=54/54/54/54 remap=0/1/2/3
 *
 * link 0x400f0f00 is rt_spill_base 0, size_32 0x0f, loc_written 0x0f and
 * sample_mask_after_force_early; 0x54 is PIPE_FORMAT_R8G8B8A8_UNORM.
 */

#include "agx_compile.h"
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

/* Mirror hk_get_prolog_epilog_locked()'s G17P epilog path exactly: the NIR
 * builder, then preprocessing, then the sample lowering, then the backend key
 * carrying nr_samples and the tile partitions. */
struct variant {
   bool sample_mask_after_force_early;
   bool lower_srgb;
   bool preserve_zero_mask;
};

static bool
compile_mrt_msaa_epilog(unsigned nr_targets, unsigned nr_samples,
                        struct variant v, struct agx_shader_part *part)
{
   struct agx_fs_epilog_key key = {
      .link =
         {
            .size_32 = BITFIELD_MASK(nr_targets),
            .loc_written = BITFIELD_MASK(nr_targets),
            .sample_mask_after_force_early = v.sample_mask_after_force_early,
         },
      .nr_samples = nr_samples,
      .lower_srgb = v.lower_srgb,
      .preserve_zero_mask = v.preserve_zero_mask,
   };
   memset(key.remap, -1, sizeof(key.remap));
   for (unsigned rt = 0; rt < nr_targets; ++rt) {
      key.remap[rt] = rt;
      key.rt_formats[rt] = PIPE_FORMAT_R8G8B8A8_UNORM;
      key.blend.rt[rt].colormask = 0xf;
      key.blend.rt[rt].mode = opaque_blend();
   }

   const struct agx_tilebuffer_layout tib = agx_build_tilebuffer_layout(
      key.rt_formats, ARRAY_SIZE(key.rt_formats), nr_samples, true);

   struct agx_shader_key backend_key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .fs.nr_samples = nr_samples,
      .secondary = true,
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

   const bool ok = agx_compile_shader_nir(b.shader, &backend_key, part);
   ralloc_free(b.shader);
   return ok;
}

static bool
run(unsigned nr_targets, unsigned nr_samples, struct variant v)
{
   struct agx_shader_part part = {0};
   const bool ok =
      compile_mrt_msaa_epilog(nr_targets, nr_samples, v, &part);
   printf("  %u rt x %u samples  mask-after-early=%u srgb=%u preserve=%u: %s\n",
          nr_targets, nr_samples, v.sample_mask_after_force_early, v.lower_srgb,
          v.preserve_zero_mask, ok ? "compiled" : "REFUSED");
   free(part.binary);
   return ok;
}

int
main(void)
{
   const struct variant reported = {true, true, true};
   const struct variant plain = {false, false, false};
   bool pass = true;

   printf("G17P multi-target multisample epilog\n");
   /* Controls first: a diagnostic that cannot compile anything proves
    * nothing about the case it refuses. */
   pass &= run(1, 1, plain);
   pass &= run(1, 2, plain);
   pass &= run(4, 1, plain);
   pass &= run(2, 1, plain);
   printf("-- sample_mask_after_force_early by target count, one sample\n");
   run(1, 1, (struct variant){true, false, false});
   run(2, 1, (struct variant){true, false, false});
   run(3, 1, (struct variant){true, false, false});
   printf("-- one field at a time on four targets, one sample\n");
   run(4, 1, (struct variant){true, false, false});
   run(4, 1, (struct variant){false, true, false});
   run(4, 1, (struct variant){false, false, true});
   run(4, 1, reported);
   printf("-- multisample\n");
   pass &= run(2, 2, plain);
   pass &= run(4, 2, plain);
   pass &= run(4, 4, plain);
   pass &= run(4, 2, reported);

   printf("%s\n", pass ? "PASS" : "FAIL");
   return pass ? 0 : 1;
}
