/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * A G17P multisample colour store must write every sample the fragment
 * covers. The hardware form that does so takes its sample mask from a live
 * register holding the coverage word, not from an immediate -- Apple's own
 * applegpu_g17p fragments emit `rld hN <- SR39` and then hand hN to LSTCOL at
 * both 2x and 4x, while every 1x fragment uses the whole-pixel immediate.
 *
 * This checks the whole epilogue path end to end: the NIR lowering asks for
 * the coverage word exactly when there is more than one sample, and the
 * backend encodes the register alternative of the operand rather than the
 * immediate one. An immediate at 2x or 4x is the original bug -- it writes
 * sample zero alone, so the resolve returns colour/N.
 */
#include "agx_compile.h"
#include "agx_linker.h"
#include "agx_nir.h"
#include "nir_builder.h"
#include "util/format/u_formats.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* form 211 lstcol, size 12: word 0 is fixed 0x06e7 under mask 0x07ff. */
static bool
is_lstcol(const uint8_t *b, size_t left)
{
   return left >= 12 && b[0] == 0xe7 && (b[1] & 0x07) == 0x06 && b[2] == 0x54;
}

/* Choice group 2 alternative 1 (the register operand) sets word 5 bit 11. */
static bool
lstcol_sample_is_register(const uint8_t *b)
{
   return (b[11] & 0x08) != 0;
}

/* Both alternatives deposit the operand in word 3 bits 0..7: the immediate
 * mask (f43) or the half-register index (f44). */
static uint8_t
lstcol_sample_operand(const uint8_t *b)
{
   return b[6];
}

static unsigned
count_intrinsic(nir_shader *s, nir_intrinsic_op op)
{
   unsigned count = 0;
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == op)
               count++;
         }
      }
   }
   return count;
}

static bool
check_samples(unsigned nr_samples)
{
   struct agx_fs_epilog_key key = {
      .nr_samples = nr_samples,
      .link.loc_written = BITFIELD_BIT(0),
   };
   memset(key.remap, -1, sizeof(key.remap));
   key.remap[0] = 0;
   key.rt_formats[0] = PIPE_FORMAT_R8G8B8A8_UNORM;
   key.blend.rt[0].colormask = 0xf;
   key.blend.rt[0].mode = agx_pack_blend_standard(
      PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO,
      PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO);

   nir_builder b =
      nir_builder_init_simple_shader(0, &agx_nir_options, "G17P MSAA epilog");
   agx_nir_fs_epilog(&b, &key);
   agx_nir_lower_g17p_tile_samples(b.shader, nr_samples);

   const unsigned coverage_reads =
      count_intrinsic(b.shader, nir_intrinsic_load_active_samples_agx);
   const unsigned want_reads = nr_samples == 1 ? 0 : 1;
   if (coverage_reads != want_reads) {
      fprintf(stderr,
              "%ux epilog reads the coverage word %u times, expected %u\n",
              nr_samples, coverage_reads, want_reads);
      ralloc_free(b.shader);
      return false;
   }

   agx_preprocess_nir(b.shader);

   struct agx_shader_key backend_key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .secondary = true,
      .no_preamble = true,
   };
   struct agx_shader_part part = {0};
   const bool compiled = agx_compile_shader_nir(b.shader, &backend_key, &part);
   ralloc_free(b.shader);

   if (!compiled) {
      fprintf(stderr, "%ux epilog did not compile\n", nr_samples);
      free(part.binary);
      return false;
   }

   unsigned stores = 0;
   bool ok = true;
   for (unsigned i = 0; i + 12 <= part.info.binary_size; ++i) {
      const uint8_t *at = (const uint8_t *)part.binary + i;
      if (!is_lstcol(at, part.info.binary_size - i))
         continue;

      stores++;
      const bool reg = lstcol_sample_is_register(at);
      if (nr_samples == 1) {
         /* Apple's 1x whole-pixel immediate. */
         if (reg || lstcol_sample_operand(at) != 1) {
            fprintf(stderr, "1x store is not the whole-pixel immediate: ");
            ok = false;
         }
      } else if (!reg) {
         fprintf(stderr,
                 "%ux store still takes an immediate sample mask (%#x): ",
                 nr_samples, lstcol_sample_operand(at));
         ok = false;
      }

      if (!ok) {
         for (unsigned j = 0; j < 12; ++j)
            fprintf(stderr, "%02x", at[j]);
         fprintf(stderr, "\n");
         break;
      }
      i += 11;
   }

   free(part.binary);

   if (stores != 1) {
      fprintf(stderr, "%ux epilog emitted %u colour stores, expected 1\n",
              nr_samples, stores);
      return false;
   }

   return ok;
}

int
main(void)
{
   bool ok = true;
   for (unsigned samples = 1; samples <= 4; samples *= 2)
      ok &= check_samples(samples);

   printf("%s\n", ok ? "PASS" : "FAIL");
   return ok ? 0 : 1;
}
