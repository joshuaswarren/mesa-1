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
#include "agx_tilebuffer.h"
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

static bool
is_lldcol(const uint8_t *b, size_t left)
{
   return left >= 12 && b[0] == 0x67 && (b[1] & 0x07) == 0x06 &&
          b[2] == 0x54;
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
      count_intrinsic(b.shader, nir_intrinsic_load_sample_mask);
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

static bool
check_alpha_coverage_blend(unsigned nr_samples)
{
   struct agx_fs_epilog_key key = {
      .nr_samples = nr_samples,
      .link.loc_written = BITFIELD_BIT(0),
      .link.size_32 = BITFIELD_BIT(0),
      .defer_monolithic_msaa = true,
      .blend.alpha_to_coverage = true,
   };
   agx_fs_epilog_apply_device_policy(AGX_CHIP_G17P, &key);
   memset(key.remap, -1, sizeof(key.remap));
   key.remap[0] = 0;
   key.rt_formats[0] = PIPE_FORMAT_R8G8B8A8_UNORM;
   key.blend.rt[0].colormask = 0xf;
   key.blend.rt[0].mode = agx_pack_blend_standard(
      PIPE_BLEND_ADD, PIPE_BLENDFACTOR_SRC_ALPHA,
      PIPE_BLENDFACTOR_INV_SRC_ALPHA, PIPE_BLEND_ADD,
      PIPE_BLENDFACTOR_ZERO, PIPE_BLENDFACTOR_ONE);

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options,
      "G17P alpha-to-coverage blend epilog");
   agx_nir_fs_epilog(&b, &key);
   agx_nir_lower_g17p_tile_samples(b.shader, nr_samples);
   agx_preprocess_nir(b.shader);

   struct agx_shader_key backend_key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .fs.nr_samples = nr_samples,
      .secondary = true,
      .no_preamble = true,
   };
   struct agx_shader_part part = {0};
   const bool compiled = agx_compile_shader_nir(b.shader, &backend_key, &part);
   ralloc_free(b.shader);
   if (!compiled) {
      fprintf(stderr, "%ux alpha-to-coverage blend epilog did not compile\n",
              nr_samples);
      return false;
   }

   unsigned loads = 0, stores = 0;
   for (unsigned i = 0; i + 12 <= part.info.binary_size; ++i) {
      const uint8_t *at = (const uint8_t *)part.binary + i;
      loads += is_lldcol(at, part.info.binary_size - i);
      stores += is_lstcol(at, part.info.binary_size - i);
   }
   free(part.binary);
   if (loads != 1 || stores != 1) {
      fprintf(stderr,
              "%ux alpha-to-coverage blend emitted %u loads/%u stores\n",
              nr_samples, loads, stores);
      return false;
   }
   return true;
}

static bool
check_unrolled_single_blend(unsigned samples)
{
   struct agx_fs_epilog_key key = {
      .nr_samples = samples,
      .link.loc_written = 1,
      .link.size_32 = 1,
      .rt_formats = {PIPE_FORMAT_R8G8B8A8_UNORM},
   };
   memset(key.remap, -1, sizeof(key.remap));
   key.remap[0] = 0;
   key.blend.rt[0].colormask = 15;
   key.blend.rt[0].mode = agx_pack_blend_standard(
      PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ONE,
      PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ONE);
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "G17P unrolled single blend");
   agx_nir_fs_epilog(&b, &key);
   agx_preprocess_nir(b.shader);
   agx_nir_lower_g17p_tile_samples(b.shader, samples);
   agx_preprocess_nir(b.shader);
   struct agx_shader_key backend = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .secondary = true,
      .no_preamble = true,
      .fs.nr_samples = samples,
   };
   struct agx_shader_part part = {0};
   bool ok = agx_compile_shader_nir(b.shader, &backend, &part);
   unsigned loads = 0, stores = 0, read_waits = 0, write_waits = 0, releases = 0;
   const uint8_t *bytes = part.binary;
   for (unsigned i = 0; ok && i + 6 <= part.info.binary_size; ++i) {
      loads += is_lldcol(bytes + i, part.info.binary_size - i);
      stores += is_lstcol(bytes + i, part.info.binary_size - i);
      read_waits += !memcmp(bytes + i, "\x87\x02\x54\x08\x08\x00", 6);
      write_waits += !memcmp(bytes + i, "\x87\x02\x54\x04\x08\x00", 6);
      releases += !memcmp(bytes + i, "\x07\x02\x54\x0c\x02\x00", 6);
   }
   ok &= loads == samples && stores == samples && read_waits == 1 &&
         write_waits == 1 && releases == 1;
   if (!ok)
      fprintf(stderr, "%ux single blend: loads=%u stores=%u waits=%u/%u releases=%u\n",
              samples, loads, stores, read_waits, write_waits, releases);
   free(part.binary);
   ralloc_free(b.shader);
   return ok;
}

static bool
check_coverage_after_depth(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "G17P ordered tile coverage");
   for (unsigned sample = 0; sample < 4; ++sample) {
      nir_def *mask = nir_imm_intN_t(&b, 1u << sample, 16);
      nir_store_zs_agx(&b, mask, nir_imm_float(&b, 0.25),
                       nir_imm_intN_t(&b, 0, 16), .base = 1);
      nir_def *value = nir_imm_vec4(&b, 0.1, 0.2, 0.3, 0.4);
      for (unsigned rt = 0; rt < 2; ++rt)
         nir_store_local_pixel_agx(&b, value, mask, nir_undef(&b, 2, 16),
                                   .base = 8 * rt, .write_mask = 15,
                                   .format = PIPE_FORMAT_R16G16B16A16_FLOAT,
                                   .target = rt);
   }
   agx_nir_lower_g17p_tile_samples(b.shader, 4);
   agx_preprocess_nir(b.shader);
   unsigned reads = count_intrinsic(b.shader, nir_intrinsic_load_sample_mask);
   unsigned stale = count_intrinsic(b.shader,
                                   nir_intrinsic_load_active_samples_agx);
   bool ok = reads == 4 && stale == 0;
   if (!ok)
      fprintf(stderr, "depth coverage: %u ordered reads, %u reusable reads\n",
              reads, stale);
   ralloc_free(b.shader);
   return ok;
}

static bool
check_half2_sample_exports(void)
{
   struct agx_fs_epilog_key key = {
      .nr_samples = 4,
      .defer_monolithic_msaa = true,
      .link.loc_written = 12,
      .link.size_32 = 12,
      .rt_formats = {PIPE_FORMAT_R16G16B16A16_SINT,
                     PIPE_FORMAT_R16G16B16A16_FLOAT},
   };
   agx_fs_epilog_apply_device_policy(AGX_CHIP_G17P, &key);
   memset(key.remap, -1, sizeof(key.remap));
   for (unsigned i = 0; i < 2; ++i) {
      key.remap[i] = i + 2;
      key.blend.rt[i].colormask = 15;
      key.blend.rt[i].mode = agx_pack_blend_standard(
         PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO,
         PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO);
   }
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "G17P distinct half2 sample exports");
   agx_nir_fs_epilog(&b, &key);
   agx_nir_lower_to_per_sample(b.shader);
   agx_nir_unroll_g17p_samples(b.shader, 4, key.link.loc_written);
   bool progress;
   do {
      progress = nir_opt_constant_folding(b.shader);
      progress |= nir_opt_algebraic(b.shader);
   } while (progress);
   agx_nir_hoist_g17p_zs_writes(b.shader);
   agx_nir_lower_g17p_tile_samples(b.shader, 4);
   agx_preprocess_nir(b.shader);
   struct agx_shader_key backend = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .secondary = true,
      .no_preamble = true,
      .fs.nr_samples = 4,
   };
   BITSET_SET(backend.g17p_tile_partitions, 0);
   BITSET_SET(backend.g17p_tile_partitions, 8);
   struct agx_shader_part part = {0};
   bool ok = agx_compile_shader_nir(b.shader, &backend, &part);
   if (!ok)
      fprintf(stderr, "distinct half2 per-sample epilog did not compile\n");
   free(part.binary);
   ralloc_free(b.shader);
   return ok;
}

static bool
check_native_spill(unsigned write_mask, bool blend, bool depth, bool resident)
{
   struct agx_fs_epilog_key key = {
      .nr_samples = 4,
      .native_spills = true,
      .max_bytes_per_tile = resident ? 32768 : 0,
      .max_samples_per_tile = resident ? 2048 : 0,
      .emrt_heap_base = 28,
      .link.loc_written = 7,
      .link.size_32 = 7,
      .link.write_z = depth,
      .rt_formats = {PIPE_FORMAT_R8G8B8A8_UNORM,
                     PIPE_FORMAT_R16G16B16A16_FLOAT,
                     PIPE_FORMAT_R32G32B32A32_UINT},
   };
   agx_fs_epilog_apply_device_policy(AGX_CHIP_G17P, &key);
   memset(key.remap, -1, sizeof(key.remap));
   for (unsigned i = 0; i < 3; ++i) {
      key.remap[i] = i;
      key.blend.rt[i].colormask = write_mask & BITFIELD_BIT(i) ? 15 : 0;
      key.blend.rt[i].mode = agx_pack_blend_standard(
         PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE,
         blend && i < 2 ? PIPE_BLENDFACTOR_ONE : PIPE_BLENDFACTOR_ZERO,
         PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE,
         blend && i < 2 ? PIPE_BLENDFACTOR_ONE : PIPE_BLENDFACTOR_ZERO);
   }
   nir_builder b = nir_builder_init_simple_shader(
      0, &agx_nir_options, "G17P native spill epilog");
   agx_nir_fs_epilog(&b, &key);
   agx_nir_lower_g17p_tile_samples(b.shader, 4);
   agx_preprocess_nir(b.shader);
   agx_nir_lower_g17p_tile_samples(b.shader, 4);
   struct agx_shader_key backend = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .secondary = true,
      .no_preamble = true,
      .g17p_emrt = true,
      .fs.nr_samples = 4,
   };
   BITSET_SET(backend.g17p_tile_partitions, 0);
   BITSET_SET(backend.g17p_tile_partitions, 16);
   if (resident)
      BITSET_SET(backend.g17p_tile_partitions, 24);
   struct agx_shader_part part = {0};
   const bool compiled = agx_compile_shader_nir(b.shader, &backend, &part);
   const bool staging_ok = !depth || resident || part.info.agx3_staging_halfs >= 32;
   if (compiled && !staging_ok)
      fprintf(stderr, "four depth exports and image write share %u staging halves\n",
              part.info.agx3_staging_halfs);
   if (!compiled)
      fprintf(stderr, "native spill epilog mask=%u blend=%u did not compile\n",
              write_mask, blend);
   free(part.binary);
   ralloc_free(b.shader);
   return compiled && staging_ok;
}

static bool
check_late_mask_split(bool alpha_coverage)
{
   nir_builder main = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "late native mask");
   nir_demote_samples(&main, nir_imm_intN_t(&main, 3, 16));
   main.shader->info.fs.uses_discard = true;
   main.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_SAMPLE_MASK);
   struct agx_fs_epilog_link_info link = {0};
   agx_nir_lower_fs_output_to_epilog_for_device(main.shader, AGX_CHIP_G17P,
                                                &link);
   bool ok = link.sample_mask_after_force_early &&
      count_intrinsic(main.shader, nir_intrinsic_demote_samples) == 0;
   ralloc_free(main.shader);

   struct agx_fs_epilog_key key = {
      .nr_samples = 4,
      .link.loc_written = 1,
      .link.size_32 = 1,
      .link.sample_mask_after_force_early = true,
      .defer_monolithic_msaa = true,
      .blend.alpha_to_coverage = alpha_coverage,
   };
   memset(key.remap, -1, sizeof(key.remap));
   key.remap[0] = 0;
   key.rt_formats[0] = PIPE_FORMAT_R8G8B8A8_UNORM;
   key.blend.rt[0].colormask = 0xf;
   key.blend.rt[0].mode = agx_pack_blend_standard(
      PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO,
      PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO);
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "late mask epilog");
   agx_nir_fs_epilog(&b, &key);
   ok &= count_intrinsic(b.shader, nir_intrinsic_sample_mask_agx) == 1;
   ok &= count_intrinsic(b.shader, nir_intrinsic_demote_samples) == 0;
   agx_nir_lower_g17p_tile_samples(b.shader, 4);
   agx_preprocess_nir(b.shader);
   struct agx_shader_key backend = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .secondary = true,
      .no_preamble = true,
      .fs.nr_samples = 4,
   };
   struct agx_shader_part part = {0};
   ok &= agx_compile_shader_nir(b.shader, &backend, &part);
   free(part.binary);
   ralloc_free(b.shader);
   return ok;
}

static bool
check_alpha_one_after_preprocess(unsigned samples)
{
   struct agx_fs_epilog_key key = {
      .nr_samples = samples,
      .link.loc_written = 1,
      .link.size_32 = 1,
      .rt_formats = {PIPE_FORMAT_R8G8B8A8_UNORM},
      .blend.alpha_to_one = true,
      .preserve_zero_mask = true,
      .g17p_tile_formats = true,
   };
   memset(key.remap, -1, sizeof(key.remap));
   key.remap[0] = 0;
   key.blend.rt[0].colormask = 15;
   key.blend.rt[0].mode = agx_pack_blend_standard(
      PIPE_BLEND_ADD, PIPE_BLENDFACTOR_SRC_ALPHA, PIPE_BLENDFACTOR_INV_SRC_ALPHA,
      PIPE_BLEND_ADD, PIPE_BLENDFACTOR_SRC_ALPHA, PIPE_BLENDFACTOR_INV_SRC_ALPHA);
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "G17P late alpha-one masks");
   agx_nir_fs_epilog(&b, &key);
   agx_nir_lower_g17p_tile_samples(b.shader, samples);
   agx_preprocess_nir(b.shader);
   /* The earlier pass must not be mistaken for a complete lowering: loop
    * optimization has just exposed constant sample operands. */
   bool ok = agx_nir_lower_g17p_tile_samples(b.shader, samples);
   ok &= count_intrinsic(b.shader, nir_intrinsic_load_sample_mask) == 1;
   struct agx_shader_key backend = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .secondary = true,
      .no_preamble = true,
      .fs.nr_samples = samples,
   };
   struct agx_shader_part part = {0};
   ok &= agx_compile_shader_nir(b.shader, &backend, &part);
   unsigned stores = 0;
   const uint8_t *bytes = part.binary;
   for (unsigned i = 0; ok && i + 12 <= part.info.binary_size; ++i) {
      if (is_lstcol(bytes + i, part.info.binary_size - i)) {
         stores++;
         ok &= lstcol_sample_is_register(bytes + i);
      }
   }
   ok &= stores == samples;
   if (!ok)
      fprintf(stderr, "%ux alpha-one late mask lowering failed (%u stores)\n",
              samples, stores);
   free(part.binary);
   ralloc_free(b.shader);
   return ok;
}

/* form 38 depthf, size 6, as the early-demote epilog test spells it. */
static bool
is_depthf(const uint8_t *b, size_t left)
{
   return left >= 6 && b[0] == 0xd7 && (b[1] & 0x0f) == 0x04 && b[2] == 0x54;
}

/*
 * A late gl_SampleMask shader that also writes gl_FragDepth arrives at the
 * epilog with its surviving samples in a register. The depth/stencil write is
 * the one test-triggering update, so at several samples it must carry that
 * register itself and reach the hardware as one write per sample. A separate
 * coverage update ahead of it runs the depth/stencil test on every surviving
 * sample a second time, and a single write naming several samples at once
 * only ever lands sample zero's depth: both spellings failed
 * renderpasses.*.multisample.<depth format>.samples_{2,4}. At one sample the
 * write's word has no mask field, so there the kill is kept, ahead of it.
 */
static bool
check_late_mask_depth(unsigned samples)
{
   nir_builder main = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "late native mask and depth");
   nir_demote_samples(&main, nir_imm_intN_t(&main, 1, 16));
   nir_store_zs_agx(&main, nir_imm_intN_t(&main, 0xff, 16),
                    nir_imm_float(&main, 0.5f), nir_undef(&main, 1, 16),
                    .base = 1);
   main.shader->info.fs.uses_discard = true;
   main.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_SAMPLE_MASK) |
                                       BITFIELD64_BIT(FRAG_RESULT_DEPTH);

   struct agx_fs_epilog_link_info link = {0};
   agx_nir_lower_fs_output_to_epilog_for_device(main.shader, AGX_CHIP_G17P,
                                                &link);
   bool ok = link.sample_mask_after_force_early && link.write_z;
   ralloc_free(main.shader);

   struct agx_fs_epilog_key key = {
      .link = link,
      .nr_samples = samples,
   };
   agx_fs_epilog_apply_device_policy(AGX_CHIP_G17P, &key);
   memset(key.remap, -1, sizeof(key.remap));

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "late mask depth epilog");
   agx_nir_fs_epilog(&b, &key);

   /* Whatever the sample count, no coverage update may follow the write that
    * triggered the tests. */
   bool seen_write = false;
   unsigned masks = 0;
   nir_foreach_block(block, b.impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_store_zs_agx)
            seen_write = true;
         else if (intr->intrinsic == nir_intrinsic_sample_mask_agx ||
                  intr->intrinsic == nir_intrinsic_demote_samples) {
            masks++;
            ok &= !seen_write;
         }
      }
   }
   ok &= seen_write && masks == (samples > 1 ? 0 : 1);

   NIR_PASS(_, b.shader, agx_nir_lower_g17p_tile_samples, samples);
   agx_preprocess_nir(b.shader);
   struct agx_shader_key backend = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .secondary = true,
      .no_preamble = true,
      .fs.nr_samples = samples,
   };
   struct agx_shader_part part = {0};
   ok &= agx_compile_shader_nir(b.shader, &backend, &part);

   unsigned writes = 0;
   const uint8_t *bytes = part.binary;
   for (unsigned i = 0; bytes && i + 6 <= part.info.binary_size; ++i)
      writes += is_depthf(bytes + i, part.info.binary_size - i);

   /* One word per sample, each naming its own sample alone. */
   ok &= writes == samples;
   if (!ok)
      fprintf(stderr, "%ux late mask depth epilog: %u masks, %u depth writes\n",
              samples, masks, writes);

   free(part.binary);
   ralloc_free(b.shader);
   return ok;
}

int
main(int argc, char **argv)
{
   if (argc == 2 && !strcmp(argv[1], "alpha-one"))
      return (check_alpha_one_after_preprocess(2) &&
              check_alpha_one_after_preprocess(4)) ? 0 : 1;
   if (argc == 2 && !strcmp(argv[1], "half2"))
      return check_half2_sample_exports() ? 0 : 1;
   if (argc == 3)
      return check_native_spill(atoi(argv[1]), atoi(argv[2]), false, false) ? 0 : 1;
   if (argc == 4)
      return check_native_spill(atoi(argv[1]), atoi(argv[2]), false,
                                atoi(argv[3])) ? 0 : 1;

   bool ok = true;
   ok &= check_alpha_one_after_preprocess(2);
   ok &= check_alpha_one_after_preprocess(4);
   ok &= check_late_mask_split(false);
   ok &= check_late_mask_split(true);
   for (unsigned samples = 1; samples <= 4; samples *= 2)
      ok &= check_late_mask_depth(samples);
   for (unsigned samples = 1; samples <= 4; samples *= 2)
      ok &= check_samples(samples);
   for (unsigned samples = 2; samples <= 4; samples *= 2)
      ok &= check_alpha_coverage_blend(samples);
   for (unsigned samples = 2; samples <= 4; samples *= 2)
      ok &= check_unrolled_single_blend(samples);
   ok &= check_coverage_after_depth();
   ok &= check_half2_sample_exports();
   for (unsigned mask = 1; mask < 8; ++mask) {
      ok &= check_native_spill(mask, false, false, false);
      ok &= check_native_spill(mask, true, false, false);
   }
   ok &= check_native_spill(7, true, true, false);
   for (unsigned mask = 1; mask < 8; ++mask) {
      ok &= check_native_spill(mask, false, false, true);
      ok &= check_native_spill(mask, true, false, true);
   }
   ok &= check_native_spill(7, true, true, true);

   printf("%s\n", ok ? "PASS" : "FAIL");
   return ok ? 0 : 1;
}
