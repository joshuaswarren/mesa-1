/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/* G17P sample-shaded fragment epilog over a spilling (eMRT) multisample
 * tilebuffer.
 *
 * dEQP-VK.rasterization.rasterization_order_attachment_access.format_float.
 * attachments_{4,8}_samples_{2,4}.multi_draw_barriers renders four or eight
 * VK_FORMAT_R32G32_SFLOAT attachments with loadOp LOAD and storeOp STORE at
 * 2x/4x, reading each one back as a per-sample input attachment.  loadOp LOAD
 * puts hk on the AGX_G17P_MAX_UNCOVERED_TILE_BYTES_PER_SAMPLE budget
 * (hk_cmd_draw.c), so exactly one eight-byte attachment stays resident in the
 * tilebuffer and the rest spill, and the epilog compiles as eMRT.
 *
 * On hardware those four cases fail with
 *   MESA: error: fragment epilog compilation failed: link=0x010f0f00 samples=2
 *   ... preserve-zero=1 srgb=1 formats=16/16/16/16 remap=0/1/2/3
 * surfacing as VK_ERROR_FEATURE_NOT_PRESENT from vkEndCommandBuffer.
 *
 * This reproduces that compile on the host with no GPU: `REFUSED` here is the
 * same packer refusal.  Mirrors build_g17p_fs_epilog() and
 * hk_get_prolog_epilog_locked() in hk_cmd_draw.c exactly.
 */

#include "agx_chip.h"
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

struct probe_result {
   bool compiled;
   unsigned resident;
   unsigned spilled;
   unsigned partitions;
   unsigned sample_size_B;
   unsigned tile_size;
   /* Pixel protocol of the compiled epilog: class-1 PIXWAITs and PIXRELs,
    * and class-1 waits issued after the first colour-class wait. */
   unsigned class1_waits;
   unsigned class1_releases;
   unsigned class1_after_colour;
};

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

/* Scan the six-byte PIXWAIT/PIXREL forms: 87 02 54 <classes> <flags> 00 and
 * 07 02 54 <classes> <flags> 00. Colour waits carry flags 0x08. */
static void
scan_pixel_protocol(const struct agx_shader_part *part,
                    struct probe_result *r)
{
   const uint8_t *code = part->binary;
   bool colour_seen = false;
   for (size_t at = 0; code && at + 6 <= part->info.binary_size; at += 2) {
      if (code[at + 1] != 0x02 || code[at + 2] != 0x54 || code[at + 5] != 0)
         continue;
      if (code[at] == 0x87 && code[at + 3] == 0x01 && code[at + 4] == 0x00) {
         r->class1_waits++;
         r->class1_after_colour += colour_seen;
      } else if (code[at] == 0x87 && code[at + 4] == 0x08) {
         colour_seen = true;
      } else if (code[at] == 0x07 && code[at + 3] == 0x01 &&
                 code[at + 4] == 0x00) {
         r->class1_releases++;
      }
   }
}

/* build_g17p_fs_epilog(), hk_cmd_draw.c:3470. */
static void
build_g17p_fs_epilog(nir_builder *b, const struct agx_fs_epilog_key *key)
{
   if (key->link.sample_shading) {
      struct agx_fs_epilog_key pixel = *key;
      pixel.link.sample_shading = false;
      pixel.defer_monolithic_msaa = true;
      agx_nir_fs_epilog(b, &pixel);

      NIR_PASS(_, b->shader, agx_nir_lower_to_per_sample);
      NIR_PASS(_, b->shader, agx_nir_unroll_g17p_samples, key->nr_samples,
               key->link.loc_written);
      bool opt_progress;
      do {
         opt_progress = false;
         NIR_PASS(opt_progress, b->shader, nir_opt_constant_folding);
         NIR_PASS(opt_progress, b->shader, nir_opt_algebraic);
      } while (opt_progress);
      NIR_PASS(_, b->shader, agx_nir_hoist_g17p_zs_writes);
   } else {
      struct agx_fs_epilog_key pixel = *key;
      pixel.defer_monolithic_msaa =
         key->blend.alpha_to_coverage && key->nr_samples > 1;
      agx_nir_fs_epilog(b, &pixel);
   }

   NIR_PASS(_, b->shader, agx_nir_lower_g17p_tile_samples, key->nr_samples);
}

static struct probe_result
probe_mask(unsigned nr_attachments, const enum pipe_format *formats,
           unsigned nr_samples, bool sample_shading, uint8_t max_sample_size_B,
           unsigned written_mask, bool api_sample_mask)
{
   struct probe_result r = {0};

   enum pipe_format fmts[8] = {PIPE_FORMAT_NONE};
   unsigned mask = 0;
   for (unsigned i = 0; i < nr_attachments; ++i) {
      fmts[i] = formats[i];
      mask |= BITFIELD_BIT(i);
   }
   if (written_mask)
      mask &= written_mask;

   /* hk_cmd_draw.c:1765 -- the render pass's tilebuffer. */
   const struct agx_tilebuffer_layout tib =
      agx_build_tilebuffer_layout_with_limits(fmts, 8, nr_samples, true,
                                              max_sample_size_B, 32768, 2048);
   r.sample_size_B = tib.sample_size_B;
   r.tile_size = tib.tile_size;

   struct agx_fs_epilog_key key = {
      .link =
         {
            .size_32 = mask,
            .loc_written = mask,
            .sample_shading = sample_shading,
            /* The main shader wrote gl_SampleMask (late tests): the G17P
             * split exports it as FOUT_WRITE_SAMPLES. */
            .sample_mask_after_force_early = api_sample_mask,
         },
      .nr_samples = nr_samples,
      .max_sample_size_B = max_sample_size_B,
      .max_bytes_per_tile = 32768,
      .max_samples_per_tile = 2048,
      .native_spills = agx_tilebuffer_spills(&tib),
   };
   agx_fs_epilog_apply_device_policy(AGX_CHIP_G17P, &key);

   memset(key.remap, -1, sizeof(key.remap));
   for (unsigned rt = 0; rt < nr_attachments; ++rt) {
      key.remap[rt] = (int8_t)rt;
      key.rt_formats[rt] = formats[rt];
      key.blend.rt[rt].colormask = 0xf;
      key.blend.rt[rt].mode = opaque_blend();
   }
   if (key.native_spills) {
      /* hk_cmd_draw.c:5570: a relocated uniform pair above the blend
       * constants. The exact base does not change the tile protocol. */
      key.emrt_heap_base = 200;
      key.blend_constant_base = 204;
   }

   /* hk_get_fast_linked_locked_fs(), hk_cmd_draw.c:3539. */
   BITSET_DECLARE(partitions, AGX_G17P_MAX_BYTES_PER_SAMPLE) = {0};
   for (unsigned rt = 0; rt < ARRAY_SIZE(tib.logical_format); ++rt) {
      if (tib.logical_format[rt] == PIPE_FORMAT_NONE)
         continue;
      if (tib.spilled[rt]) {
         r.spilled++;
      } else {
         r.resident++;
         BITSET_SET(partitions, tib._offset_B[rt]);
      }
   }
   r.partitions =
      __bitset_count(partitions, BITSET_WORDS(AGX_G17P_MAX_BYTES_PER_SAMPLE));
   if (r.partitions < 2)
      BITSET_ZERO(partitions);

   /* hk_get_prolog_epilog_locked(), hk_cmd_draw.c:3289. */
   nir_builder b = nir_builder_init_simple_shader(0, &agx_nir_options, NULL);
   build_g17p_fs_epilog(&b, &key);
   agx_preprocess_nir(b.shader);
   /* hk_get_prolog_epilog_locked(): an unrolled pixel-shaded epilog with
    * several Z/S writes or coverage updates takes class 1 once. */
   if (!key.link.sample_shading &&
       (count_intrinsic(b.shader, nir_intrinsic_store_zs_agx) > 1 ||
        count_intrinsic(b.shader, nir_intrinsic_sample_mask_agx) > 1))
      NIR_PASS(_, b.shader, agx_nir_hoist_g17p_zs_writes);

   struct agx_shader_key backend_key = {
      .dev.chip = AGX_CHIP_G17P,
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .fs.nr_samples = nr_samples,
      .secondary = true,
      .g17p_emrt = key.native_spills,
   };
   NIR_PASS(_, b.shader, agx_nir_lower_g17p_tile_samples, nr_samples);
   if (backend_key.g17p_emrt) {
      nir_metadata_require(b.impl, nir_metadata_dominance);
      nir_foreach_block(block, b.impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;
            if ((op == nir_intrinsic_store_local_pixel_agx ||
                 op == nir_intrinsic_load_local_pixel_agx) &&
                !nir_block_dominates(block, nir_impl_last_block(b.impl)))
               backend_key.g17p_emrt = false;
         }
      }
   }
   if (r.partitions >= 2) {
      memcpy(backend_key.g17p_tile_partitions, partitions,
             sizeof(backend_key.g17p_tile_partitions));
   }
   if (b.shader->info.fs.uses_sample_shading)
      backend_key.fs.inside_sample_loop = true;

   struct agx_shader_part part = {0};
   r.compiled = agx_compile_shader_nir(b.shader, &backend_key, &part);
   if (r.compiled)
      scan_pixel_protocol(&part, &r);
   free(part.binary);
   ralloc_free(b.shader);
   return r;
}

static struct probe_result
probe(unsigned nr_attachments, const enum pipe_format *formats,
      unsigned nr_samples, bool sample_shading, uint8_t max_sample_size_B,
      unsigned written_mask)
{
   return probe_mask(nr_attachments, formats, nr_samples, sample_shading,
                     max_sample_size_B, written_mask, false);
}

static unsigned failures;

/* A pixel-shaded multisample epilog over spilled targets runs its colour
 * stores per sample. An API sample mask must still update coverage once,
 * with one class-1 PIXWAIT/PIXREL pair before the first colour wait, as
 * every native G17P fragment does. Cloning the update per sample took class
 * 1 again under the held colour classes: renderpass*.suballocation.
 * multisample_resolve.r32g32b32a32_sint.samples_4 (four RGBA32 targets at 4x,
 * two spilled, gl_SampleMask written) hung the 3D pipeline within a second
 * at six concurrent processes. */
static void
expect_one_coverage_update(const char *label, unsigned nr_attachments,
                           enum pipe_format format, unsigned nr_samples,
                           uint8_t max_sample_size_B)
{
   enum pipe_format formats[8];
   for (unsigned i = 0; i < ARRAY_SIZE(formats); ++i)
      formats[i] = format;

   printf("%-52s samples=%u mask budget=%3u ... ", label, nr_samples,
          max_sample_size_B);
   fflush(stdout);

   const struct probe_result r = probe_mask(
      nr_attachments, formats, nr_samples, false, max_sample_size_B, 0, true);
   const bool ok = r.compiled && r.spilled > 0 && r.class1_waits == 1 &&
                   r.class1_releases == 1 && r.class1_after_colour == 0;

   printf("resident=%u spilled=%u class1 waits=%u releases=%u after-colour=%u "
          "%s%s\n",
          r.resident, r.spilled, r.class1_waits, r.class1_releases,
          r.class1_after_colour, r.compiled ? "compiled" : "REFUSED",
          ok ? "" : "  <-- UNEXPECTED");

   if (!ok)
      failures++;
}

static void
expect_mixed(const char *label, unsigned nr_attachments,
             const enum pipe_format *formats, unsigned nr_samples,
             bool sample_shading, uint8_t max_sample_size_B,
             unsigned written_mask, bool want)
{
   printf("%-52s samples=%u ss=%u budget=%3u ... ", label, nr_samples,
          sample_shading, max_sample_size_B);
   fflush(stdout);

   const struct probe_result r =
      probe(nr_attachments, formats, nr_samples, sample_shading,
            max_sample_size_B, written_mask);

   printf("resident=%u spilled=%u parts=%u ss_B=%u tile=%u %s%s\n",
          r.resident, r.spilled, r.partitions, r.sample_size_B, r.tile_size,
          r.compiled ? "compiled" : "REFUSED",
          r.compiled == want ? "" : "  <-- UNEXPECTED");

   if (r.compiled != want)
      failures++;
}

static void
expect_masked(const char *label, unsigned nr_attachments,
              enum pipe_format format, unsigned nr_samples,
              bool sample_shading, uint8_t max_sample_size_B,
              unsigned written_mask, bool want)
{
   enum pipe_format formats[8];
   for (unsigned i = 0; i < ARRAY_SIZE(formats); ++i)
      formats[i] = format;

   printf("%-52s samples=%u ss=%u budget=%3u ... ", label, nr_samples,
          sample_shading, max_sample_size_B);
   fflush(stdout);

   const struct probe_result r =
      probe(nr_attachments, formats, nr_samples, sample_shading,
            max_sample_size_B, written_mask);

   printf("resident=%u spilled=%u parts=%u ss_B=%u tile=%u %s%s\n",
          r.resident, r.spilled, r.partitions, r.sample_size_B, r.tile_size,
          r.compiled ? "compiled" : "REFUSED",
          r.compiled == want ? "" : "  <-- UNEXPECTED");

   if (r.compiled != want)
      failures++;
}

static void
expect(const char *label, unsigned nr_attachments, enum pipe_format format,
       unsigned nr_samples, bool sample_shading, uint8_t max_sample_size_B,
       bool want)
{
   expect_masked(label, nr_attachments, format, nr_samples, sample_shading,
                 max_sample_size_B, 0, want);
}

int
main(void)
{
   const enum pipe_format rg32f = PIPE_FORMAT_R32G32_FLOAT;
   const uint8_t wide = AGX_G17P_MAX_BYTES_PER_SAMPLE;
   const uint8_t uncovered = AGX_G17P_MAX_UNCOVERED_TILE_BYTES_PER_SAMPLE;

   printf("PIPE_FORMAT_R32G32_FLOAT = %u\n\n", (unsigned)rg32f);

   printf("=== ROAA multi_draw_barriers: loadOp LOAD, %u B/sample budget ===\n",
          uncovered);
   /* Three of the four failing CTS cases. attachments_8 samples_4 is NOT
    * here: eight locations unrolled over four samples reserve
    * 8 * 4 * 8 = 256 export halves before depth/stencil, so
    * hk_g17p_unrolls_sample_shading() declines it and the draw is refused at
    * a different place (hk_cmd_draw.c:5480, "per-sample fragment execution
    * needs a fixed sample count"). Compiling it here would not model that;
    * it runs the register allocator out of registers instead. */
   expect("roaa attachments_4 samples_2", 4, rg32f, 2, true, uncovered, true);
   expect("roaa attachments_4 samples_4", 4, rg32f, 4, true, uncovered, true);
   expect("roaa attachments_8 samples_2", 8, rg32f, 2, true, uncovered, true);

   printf("\n=== the passing neighbours: no spill, same shader shape ===\n");
   expect("roaa attachments_1 samples_2", 1, rg32f, 2, true, uncovered, true);
   expect("roaa attachments_1 samples_4", 1, rg32f, 4, true, uncovered, true);
   /* A one-sample pipeline is never unrolled, so its epilog key carries
    * sample_shading = 0 (hk_g17p_unrolls_sample_shading()). */
   expect("roaa attachments_4 samples_1", 4, rg32f, 1, false, wide, true);
   expect("roaa attachments_8 samples_1", 8, rg32f, 1, false, wide, true);

   printf("\n=== pixel-shaded eMRT multisample: same single-partition shape ===\n");
   expect("pixel-shaded 4x rg32f samples_2", 4, rg32f, 2, false, uncovered,
          true);
   expect("pixel-shaded 8x rg32f samples_4", 8, rg32f, 4, false, uncovered,
          true);

   printf("\n=== eMRT multisample whose resident partition is unwritten ===\n");
   /* agx_nir_lower_tilebuffer() preserves an existing-but-unwritten resident
    * attachment of a multisample eMRT layout with a load/store pair, so the
    * same single-partition epilog also occurs WITH tile loads. */
   expect_masked("eMRT 4x rg32f samples_2, RT0 unwritten", 4, rg32f, 2, true,
                 uncovered, 0xe, true);
   expect_masked("eMRT 8x rg32f samples_2, RT0 unwritten", 8, rg32f, 2, true,
                 uncovered, 0xfe, true);

   printf("\n=== pipeline.*.multisample.misc: mixed-format MSAA passes ===\n");
   /* dEQP-VK.pipeline.<mode>.multisample.misc.{multi_subpass,multi_renderpass,
    * dynamic_rendering.multi_renderpass}.<formats>.random_N, 282 cases across
    * three pipeline construction modes. Nothing about the shape is specific to
    * VK_EXT_rasterization_order_attachment_access: an ordinary multi-attachment
    * MSAA render pass that loads and stores a wide attachment lands on the same
    * 8 B/sample budget, and one attachment stays resident while the rest spill.
    *
    * Attachment ORDER decides which one stays, and only a resident 8-byte
    * target reproduces: the measured failing layout put R16G16B16A16_FLOAT
    * (pipe format 56) resident at offset 0 with R8G8B8A8_UNORM (84) and
    * R16G16B16A16_SINT (52) spilled. Put the 4-byte RGBA8 first instead and
    * the same attachment set compiles even on the unfixed packer.
    *
    * Only two budgets exist on G17P -- hk_cmd_draw.c picks
    * AGX_G17P_MAX_UNCOVERED_TILE_BYTES_PER_SAMPLE or
    * AGX_G17P_MAX_BYTES_PER_SAMPLE and nothing between -- so these probe the
    * uncovered one, which is the only budget that spills. */
   const enum pipe_format misc_resident_wide[3] = {
      PIPE_FORMAT_R16G16B16A16_FLOAT,
      PIPE_FORMAT_R8G8B8A8_UNORM,
      PIPE_FORMAT_R16G16B16A16_SINT,
   };
   expect_mixed("misc rgba16f-resident samples_2", 3, misc_resident_wide, 2,
                true, uncovered, 0, true);
   expect_mixed("misc rgba16f-resident samples_4", 3, misc_resident_wide, 4,
                true, uncovered, 0, true);
   expect_mixed("misc rgba16f-resident samples_4 pixel", 3, misc_resident_wide,
                4, false, uncovered, 0, true);
   /* RT0 loaded but unwritten, as a subpass that only reads it. */
   expect_mixed("misc rgba16f-resident samples_4, RT0 unwritten", 3,
                misc_resident_wide, 4, true, uncovered, 0x6, true);

   /* The same attachments with the narrow one first keep a 4-byte resident
    * target and were never refused; they pin that this is about the resident
    * partition, not the format set. */
   const enum pipe_format misc_resident_narrow[3] = {
      PIPE_FORMAT_R8G8B8A8_UNORM,
      PIPE_FORMAT_R16G16B16A16_FLOAT,
      PIPE_FORMAT_R16G16B16A16_SINT,
   };
   expect_mixed("misc rgba8-resident samples_4 (always compiled)", 3,
                misc_resident_narrow, 4, true, uncovered, 0, true);

   printf("\n=== wide budget: several resident partitions, no spill ===\n");
   expect("wide rgba8 x8 samples_2", 8, PIPE_FORMAT_R8G8B8A8_UNORM, 2, true,
          wide, true);
   expect("wide rgba8 x4 samples_4", 4, PIPE_FORMAT_R8G8B8A8_UNORM, 4, true,
          wide, true);
   expect("wide rg32f x4 samples_2", 4, rg32f, 2, true, wide, true);
   expect("wide rg32f x8 samples_2", 8, rg32f, 2, true, wide, true);

   printf("\n=== pixel-shaded eMRT multisample with an API sample mask ===\n");
   expect_one_coverage_update("rgba32i x4 samples_4 (multisample_resolve)", 4,
                              PIPE_FORMAT_R32G32B32A32_SINT, 4, wide);
   expect_one_coverage_update("rgba32i x3 samples_4", 3,
                              PIPE_FORMAT_R32G32B32A32_SINT, 4, wide);
   expect_one_coverage_update("rgba32f x4 samples_2 (uncovered budget)", 4,
                              PIPE_FORMAT_R32G32B32A32_FLOAT, 2, uncovered);
   expect_one_coverage_update("rg32f x4 samples_4 (uncovered budget)", 4, rg32f,
                              4, uncovered);

   printf("\n%u unexpected result(s)\n", failures);
   return failures ? 1 : 0;
}
