/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_abi.h"
#include "agx_compile.h"
#include "agx_linker.h"
#include "agx_tilebuffer.h"
#include "nir_builder.h"
#include "util/format/u_formats.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
test_single_sample_shading(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "G17P 1x sample shading");
   b.shader->info.fs.uses_sample_shading = true;
   BITSET_SET(b.shader->info.system_values_read, SYSTEM_VALUE_SAMPLE_ID);
   BITSET_SET(b.shader->info.system_values_read, SYSTEM_VALUE_SAMPLE_POS);
   BITSET_SET(b.shader->info.system_values_read, SYSTEM_VALUE_SAMPLE_MASK_IN);

   nir_def *id = nir_load_sample_id(&b);
   nir_def *pos = nir_load_sample_pos(&b);
   nir_def *mask = nir_load_sample_mask_in(&b);
   nir_def *value = nir_vec4(&b, nir_u2f32(&b, id),
                             nir_channel(&b, pos, 0),
                             nir_channel(&b, pos, 1), nir_u2f32(&b, mask));
   nir_store_output(&b, value, nir_imm_int(&b, 0), .write_mask = 0xf,
                    .io_semantics.location = FRAG_RESULT_DATA0,
                    .src_type = nir_type_float32);

   const bool progress = agx_nir_lower_g17p_1x_sample_shading(b.shader);
   const bool ok = progress && !b.shader->info.fs.uses_sample_shading &&
      count_intrinsic(b.shader, nir_intrinsic_load_sample_id) == 0 &&
      count_intrinsic(b.shader, nir_intrinsic_load_sample_pos) == 0 &&
      count_intrinsic(b.shader, nir_intrinsic_load_sample_mask_in) == 0;
   ralloc_free(b.shader);
   return ok;
}

static bool
test_epilog_discard_policy(void)
{
   struct agx_fs_epilog_link_info link = {0};
   if (agx_fs_epilog_may_discard(&link, false, false) ||
       !agx_fs_epilog_may_discard(NULL, true, false) ||
       agx_fs_epilog_may_discard(NULL, true, true))
      return false;

   link.write_z = true;
   if (!agx_fs_epilog_may_discard(&link, false, false) ||
       agx_fs_epilog_may_discard(&link, false, true))
      return false;

   link.write_z = false;
   link.write_s = true;
   return agx_fs_epilog_may_discard(&link, false, false);
}

static bool
test_all_live_early_test_prolog(void)
{
   struct agx_fs_prolog_key key = {
      .api_sample_mask = 0xff,
      .run_zs_tests = true,
      .nr_samples = 1,
   };
   nir_builder b = nir_builder_init_simple_shader(
      0, &agx_nir_options, "G17P all-live early-test FS prolog");
   agx_nir_fs_prolog(&b, &key);

   bool ok = count_intrinsic(b.shader, nir_intrinsic_sample_mask_agx) == 1;
   nir_foreach_block(block, b.impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_sample_mask_agx)
            continue;
         ok &= nir_src_is_const(intr->src[0]) &&
               nir_src_as_uint(intr->src[0]) == 0xff &&
               nir_src_is_const(intr->src[1]) &&
               nir_src_as_uint(intr->src[1]) == 0xff;
      }
   }
   nir_validate_shader(b.shader, "after constructing all-live early test");
   agx_preprocess_nir(b.shader);

   struct agx_shader_key backend_key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .secondary = true,
      .no_preamble = true,
      .fs.nr_samples = 1,
   };
   struct agx_shader_part part = {0};
   ok &= agx_compile_shader_nir(b.shader, &backend_key, &part) &&
         part.info.writes_sample_mask;
   free(part.binary);
   ralloc_free(b.shader);
   return ok;
}

static nir_shader *
build_early_demote_fs(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "G17P early-demote split FS");
   b.shader->info.fs.early_fragment_tests = true;
   b.shader->info.fs.uses_discard = true;
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);

   nir_def *discarded = nir_load_preamble(&b, 1, 16, .base = 0);
   nir_demote_samples(&b, discarded);
   nir_store_output(&b, nir_load_preamble(&b, 4, 32, .base = 2),
                    nir_imm_int(&b, 0), .write_mask = 0xf,
                    .io_semantics.location = FRAG_RESULT_DATA0,
                    .src_type = nir_type_float32);
   return b.shader;
}

static nir_shader *
build_alpha_one_fs(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "G17P alpha-one split FS");
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
   nir_store_output(&b, nir_imm_vec4(&b, 0.25f, 0.5f, 0.75f, 1.0f),
                    nir_imm_int(&b, 0), .write_mask = 0xf,
                    .io_semantics.location = FRAG_RESULT_DATA0,
                    .src_type = nir_type_float32);
   return b.shader;
}

static bool
has_lstcol(const struct agx_shader_part *part)
{
   const uint8_t *binary = part->binary;
   for (unsigned i = 0; binary && i + 12 <= part->info.binary_size; ++i) {
      if (binary[i] == 0xe7 && binary[i + 1] == 0x06 &&
          binary[i + 2] == 0x54)
         return true;
   }
   return false;
}

static unsigned
count_dynamic_tile_samples(nir_shader *s)
{
   unsigned count = 0;
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            unsigned src;
            if (intr->intrinsic == nir_intrinsic_load_local_pixel_agx)
               src = 0;
            else if (intr->intrinsic == nir_intrinsic_store_local_pixel_agx)
               src = 1;
            else
               continue;

            count += !nir_src_is_const(intr->src[src]);
         }
      }
   }
   return count;
}

static bool
exported_bases(nir_shader *s, nir_intrinsic_op op, unsigned count,
               const unsigned *bases)
{
   unsigned seen = 0;
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic ||
                nir_instr_as_intrinsic(instr)->intrinsic != op)
               continue;

            if (seen == count ||
                nir_intrinsic_base(nir_instr_as_intrinsic(instr)) !=
                   bases[seen])
               return false;
            seen++;
         }
      }
   }
   return seen == count;
}

static unsigned
count_bytes(const struct agx_shader_part *part, const uint8_t *pattern,
            unsigned len)
{
   unsigned count = 0;
   const uint8_t *binary = part->binary;
   for (unsigned i = 0; binary && i + len <= part->info.binary_size; ++i)
      count += memcmp(binary + i, pattern, len) == 0;
   return count;
}

/* A sample-shaded G17P fragment runs its body once per sample inside the
 * program, exporting each sample's colours to their own registers, and its
 * epilog stores each sample from those registers in one class-12 transaction:
 * Apple's fs_sample_id_4x brackets four masked LSTCOLs with one PIXWAIT 12 and
 * one PIXREL 524. */
static bool
test_unrolled_sample_shading(void)
{
   static const unsigned colour_bases[] = {
      AGX_ABI_FOUT_COLOUR(0),
      AGX_ABI_FOUT_COLOUR(0) + 8,
      AGX_ABI_FOUT_COLOUR(0) + 16,
      AGX_ABI_FOUT_COLOUR(0) + 24,
   };

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "G17P unrolled main FS");
   nir_def *active = nir_u2f32(&b, nir_load_active_samples_agx(&b));
   nir_def *log2 = nir_u2f32(&b, nir_load_samples_log2_agx(&b));
   nir_export_agx(&b, nir_vec4(&b, active, log2, active, log2),
                  .base = AGX_ABI_FOUT_COLOUR(0));
   nir_export_agx(&b, nir_imm_intN_t(&b, 0, 32), .base = AGX_ABI_FOUT_Z);

   bool ok = agx_nir_unroll_g17p_samples(b.shader, 4, BITFIELD_BIT(0));
   nir_validate_shader(b.shader, "after unrolling samples");
   ok &= count_intrinsic(b.shader, nir_intrinsic_load_active_samples_agx) ==
            0 &&
         count_intrinsic(b.shader, nir_intrinsic_load_samples_log2_agx) == 0;
   ok &= count_intrinsic(b.shader, nir_intrinsic_export_agx) == 8;

   /* Every export gathers after the last copy; sample i's depth follows all
    * four samples' colours for i > 0. */
   static const unsigned main_bases[] = {
      AGX_ABI_FOUT_COLOUR(0),     AGX_ABI_FOUT_Z,
      AGX_ABI_FOUT_COLOUR(0) + 8, AGX_ABI_FOUT_COLOUR(0) + 32,
      AGX_ABI_FOUT_COLOUR(0) + 16, AGX_ABI_FOUT_COLOUR(0) + 36,
      AGX_ABI_FOUT_COLOUR(0) + 24, AGX_ABI_FOUT_COLOUR(0) + 40,
   };
   ok &= exported_bases(b.shader, nir_intrinsic_export_agx, 8, main_bases) &&
         agx_g17p_sample_zs_base(4, BITFIELD_BIT(0), 1) ==
            AGX_ABI_FOUT_COLOUR(0) + 32;
   nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
      nir_foreach_instr(instr, block) {
         ok &= instr->type != nir_instr_type_intrinsic ||
               nir_instr_as_intrinsic(instr)->intrinsic !=
                  nir_intrinsic_export_agx ||
               block == nir_impl_last_block(nir_shader_get_entrypoint(
                           b.shader));
      }
   }
   ralloc_free(b.shader);
   if (!ok)
      return false;

   struct agx_fs_epilog_key key = {
      .link.loc_written = BITFIELD_BIT(0),
      .link.size_32 = BITFIELD_BIT(0),
      .link.already_ran_zs = true,
      .nr_samples = 4,
   };
   memset(key.remap, -1, sizeof(key.remap));
   key.remap[0] = 0;
   key.rt_formats[0] = PIPE_FORMAT_R8G8B8A8_UNORM;
   key.blend.rt[0].colormask = 0xf;
   key.blend.rt[0].mode =
      agx_pack_blend_standard(PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE,
                              PIPE_BLENDFACTOR_ZERO, PIPE_BLEND_ADD,
                              PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO);

   b = nir_builder_init_simple_shader(0, &agx_nir_options,
                                      "G17P unrolled FS epilog");
   agx_nir_fs_epilog(&b, &key);
   agx_nir_lower_to_per_sample(b.shader);
   ok &= agx_nir_unroll_g17p_samples(b.shader, 4, key.link.loc_written);
   nir_opt_constant_folding(b.shader);
   agx_nir_lower_g17p_tile_samples(b.shader, 4);
   ok &= exported_bases(b.shader, nir_intrinsic_load_exported_agx, 4,
                        colour_bases) &&
         count_dynamic_tile_samples(b.shader) == 4;
   agx_preprocess_nir(b.shader);

   struct agx_shader_key backend_key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .secondary = true,
      .no_preamble = true,
      .fs.nr_samples = 4,
   };
   struct agx_shader_part part = {0};
   static const uint8_t wait12[] = {0x87, 0x02, 0x54, 0x0c, 0x08, 0x00};
   static const uint8_t release524[] = {0x07, 0x02, 0x54, 0x0c, 0x02, 0x00};
   static const uint8_t lstcol[] = {0xe7, 0x06, 0x54};
   ok &= agx_compile_shader_nir(b.shader, &backend_key, &part) &&
         count_bytes(&part, wait12, sizeof(wait12)) == 1 &&
         count_bytes(&part, release524, sizeof(release524)) == 1 &&
         count_bytes(&part, lstcol, sizeof(lstcol)) == 4;
   free(part.binary);
   ralloc_free(b.shader);
   return ok;
}

/* Per-sample depth: Apple's fs_id_4x (applegpu_g17p, [[depth(any)]] per
 * [[sample_id]]) stages sample i's word and depth at staging halves 4i and
 * 4i + 2 and writes all four DEPTHFs under one class-1 transaction. The
 * unrolled epilog completes that transaction before its colour stores. */
static bool
test_unrolled_sample_depth(bool alpha_coverage, bool writes_depth)
{
   struct agx_fs_epilog_key key = {
      .link.loc_written = BITFIELD_BIT(0),
      .link.size_32 = BITFIELD_BIT(0),
      .link.write_z = writes_depth,
      .nr_samples = 4,
      .defer_monolithic_msaa = true,
      .blend.alpha_to_coverage = alpha_coverage,
      .blend.alpha_to_one = alpha_coverage,
   };
   memset(key.remap, -1, sizeof(key.remap));
   key.remap[0] = 0;
   key.rt_formats[0] = PIPE_FORMAT_R8G8B8A8_UNORM;
   key.blend.rt[0].colormask = 0xf;
   key.blend.rt[0].mode =
      agx_pack_blend_standard(PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE,
                              PIPE_BLENDFACTOR_ZERO, PIPE_BLEND_ADD,
                              PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO);

   nir_builder b = nir_builder_init_simple_shader(
      0, &agx_nir_options, "G17P unrolled depth FS epilog");
   agx_nir_fs_epilog(&b, &key);
   agx_nir_lower_to_per_sample(b.shader);
   bool ok = agx_nir_unroll_g17p_samples(b.shader, 4, key.link.loc_written);
   bool progress;
   do {
      progress = nir_opt_constant_folding(b.shader);
      progress |= nir_opt_algebraic(b.shader);
   } while (progress);
   ok &= agx_nir_hoist_g17p_zs_writes(b.shader);
   ok &= count_intrinsic(b.shader, nir_intrinsic_sample_mask_agx) ==
         (alpha_coverage ? 1 : 0);
   agx_nir_lower_g17p_tile_samples(b.shader, 4);
   agx_preprocess_nir(b.shader);

   struct agx_shader_key backend_key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .secondary = true,
      .no_preamble = true,
      .fs.nr_samples = 4,
   };
   struct agx_shader_part part = {0};
   ok &= agx_compile_shader_nir(b.shader, &backend_key, &part);

   static const uint8_t wait1[] = {0x87, 0x02, 0x54, 0x01, 0x00, 0x00};
   static const uint8_t release1[] = {0x07, 0x02, 0x54, 0x01, 0x00, 0x00};
   static const uint8_t wait12[] = {0x87, 0x02, 0x54, 0x0c, 0x08, 0x00};
   /* Without a depth export, the combined testing mask owns class 1. With
    * a depth export, DEPTHF owns it instead. Both require one transaction. */
   ok &= count_bytes(&part, wait1, sizeof(wait1)) == 1 &&
         count_bytes(&part, release1, sizeof(release1)) == 1 &&
         count_bytes(&part, wait12, sizeof(wait12)) == 1;

   /* DEPTHF f5, the staging half, is byte 3 of the six-byte form. */
   const uint8_t *text = part.binary;
   unsigned depth_writes = 0;
   size_t first_depthf = 0, colour_wait = 0;
   for (size_t at = 0; text && at + 6 <= part.info.binary_size; ++at) {
      if (text[at] == 0xd7 && (text[at + 1] & 0x0f) == 0x04 &&
          text[at + 2] == 0x54 && text[at + 5] == 0x03) {
         ok &= text[at + 3] == 4 * depth_writes;
         first_depthf = depth_writes++ ? first_depthf : at;
      }
      if (!memcmp(text + at, wait12, sizeof(wait12)))
         colour_wait = at;
   }
   ok &= depth_writes == (writes_depth ? 4 : 0) &&
         (!writes_depth || first_depthf < colour_wait);

   if (!ok)
      fprintf(stderr, "depth fixture: alpha=%u binary=%u depth=%u first=%zu colour=%zu masks=%u waits=%u releases=%u colour-waits=%u\n",
              alpha_coverage, part.info.binary_size, depth_writes, first_depthf,
              colour_wait, count_intrinsic(b.shader, nir_intrinsic_sample_mask_agx),
              count_bytes(&part, wait1, sizeof(wait1)),
              count_bytes(&part, release1, sizeof(release1)),
              count_bytes(&part, wait12, sizeof(wait12)));

   free(part.binary);
   ralloc_free(b.shader);
   return ok;
}

/* Interpret the pure scalar mask expressions used in these epilog fixtures.
 * Unknown operations fail the check rather than becoming assumed constants. */
static bool
eval_sample_mask(nir_def *def, unsigned exported_mask, unsigned *value)
{
   if (def->num_components != 1 || def->bit_size != 16)
      return false;
   nir_instr *instr = nir_def_instr(def);
   if (instr->type == nir_instr_type_load_const) {
      *value = nir_const_value_as_uint(nir_instr_as_load_const(instr)->value[0], 16);
      return true;
   }
   if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      if (intr->intrinsic != nir_intrinsic_load_exported_agx ||
          nir_intrinsic_base(intr) != AGX_ABI_FOUT_WRITE_SAMPLES)
         return false;
      *value = exported_mask;
      return true;
   }
   if (instr->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(instr);
   unsigned a, b;
   if (alu->op != nir_op_iand && alu->op != nir_op_ior && alu->op != nir_op_ixor)
      return false;
   if (!eval_sample_mask(alu->src[0].src.ssa, exported_mask, &a) ||
       !eval_sample_mask(alu->src[1].src.ssa, exported_mask, &b))
      return false;
   *value = alu->op == nir_op_iand ? a & b :
            alu->op == nir_op_ior ? a | b : a ^ b;
   return true;
}

static bool
test_disjoint_testing_masks(void)
{
   for (unsigned samples = 2; samples <= 4; samples *= 2) {
      for (unsigned dynamic = 0; dynamic < 2; ++dynamic) {
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_FRAGMENT, &agx_nir_options, "disjoint testing masks");
         nir_def *input = nir_load_exported_agx(
            &b, 1, 16, .base = AGX_ABI_FOUT_WRITE_SAMPLES);
         for (unsigned i = 0; i < samples; ++i) {
            nir_def *bit = nir_imm_intN_t(&b, 1u << i, 16);
            nir_def *target = dynamic ? nir_iand(&b, input, bit) : bit;
            /* Other samples' LIVE bits must never leak into this target. */
            nir_sample_mask_agx(&b, target, nir_ixor(&b, input, bit));
         }
         bool ok = agx_nir_hoist_g17p_zs_writes(b.shader) &&
                   count_intrinsic(b.shader, nir_intrinsic_sample_mask_agx) == 1;
         nir_intrinsic_instr *mask = NULL;
         nir_foreach_block(block, b.impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type == nir_instr_type_intrinsic &&
                   nir_instr_as_intrinsic(instr)->intrinsic ==
                      nir_intrinsic_sample_mask_agx)
                  mask = nir_instr_as_intrinsic(instr);
            }
         }
         const unsigned all = (1u << samples) - 1;
         for (unsigned input_mask = 0; mask && input_mask < 256; ++input_mask) {
            unsigned target = 0, live = 0;
            ok &= eval_sample_mask(mask->src[0].ssa, input_mask, &target) &&
                  eval_sample_mask(mask->src[1].ssa, input_mask, &live);
            ok &= target == (dynamic ? input_mask & all : all);
            ok &= live == (dynamic ? 0 : (~input_mask & all));
         }
         ok &= mask != NULL;
         ralloc_free(b.shader);
         if (!ok)
            return false;
      }
   }

   /* Repeated targets and mutable coverage inputs do not commute. */
   for (unsigned bad = 0; bad < 2; ++bad) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "noncommuting testing masks");
      nir_def *live = bad ? nir_u2u16(&b, nir_load_sample_mask(&b))
                         : nir_imm_intN_t(&b, 0xff, 16);
      nir_sample_mask_agx(&b, nir_imm_intN_t(&b, 1, 16), live);
      nir_sample_mask_agx(&b, nir_imm_intN_t(&b, bad ? 2 : 1, 16), live);
      bool ok = !agx_nir_hoist_g17p_zs_writes(b.shader) &&
                count_intrinsic(b.shader, nir_intrinsic_sample_mask_agx) == 2;
      ralloc_free(b.shader);
      if (!ok)
         return false;
   }
   return true;
}

/* A late-discard main shader exports its surviving-sample mask to the split
 * epilog. Apply all per-sample kills once, before the depth transaction, and
 * keep only API-enabled depth writes. Exercise every possible 8-bit exported
 * mask to check the semantic union, not just the number of instructions. */
static bool
test_unrolled_dynamic_depth_mask_case(unsigned samples, unsigned api_mask)
{
   struct agx_fs_epilog_key key = {
      .link.loc_written = BITFIELD_BIT(0),
      .link.size_32 = BITFIELD_BIT(0),
      .link.sample_shading = true,
      .link.write_z = true,
      .link.sample_mask_after_force_early = true,
      .nr_samples = samples,
      .preserve_zero_mask = true,
      .defer_api_sample_mask = true,
      .api_sample_mask = api_mask,
   };
   memset(key.remap, -1, sizeof(key.remap));
   key.remap[0] = 0;
   key.rt_formats[0] = PIPE_FORMAT_R8G8B8A8_UNORM;
   key.blend.rt[0].colormask = 0xf;
   key.blend.rt[0].mode =
      agx_pack_blend_standard(PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE,
                              PIPE_BLENDFACTOR_ZERO, PIPE_BLEND_ADD,
                              PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO);

   nir_builder b = nir_builder_init_simple_shader(
      0, &agx_nir_options, "G17P dynamic-mask depth FS epilog");
   struct agx_fs_epilog_key pixel = key;
   pixel.link.sample_shading = false;
   pixel.defer_monolithic_msaa = true;
   agx_nir_fs_epilog(&b, &pixel);
   agx_nir_lower_to_per_sample(b.shader);
   bool ok = agx_nir_unroll_g17p_samples(b.shader, samples,
                                         key.link.loc_written);
   bool opt_progress;
   do {
      opt_progress = false;
      opt_progress |= nir_opt_constant_folding(b.shader);
      opt_progress |= nir_opt_algebraic(b.shader);
   } while (opt_progress);
   const bool hoisted = agx_nir_hoist_g17p_zs_writes(b.shader);
   const unsigned sample_masks =
      count_intrinsic(b.shader, nir_intrinsic_sample_mask_agx);
   ok &= hoisted && sample_masks == 1;
   bool saw_output = false;
   unsigned seen_masks = 0;
   nir_foreach_block(block, b.impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_store_zs_agx ||
             intr->intrinsic == nir_intrinsic_store_local_pixel_agx) {
            saw_output = true;
            ok &= seen_masks == 1;
         }
         if (intr->intrinsic != nir_intrinsic_sample_mask_agx)
            continue;
         seen_masks++;
         ok &= !saw_output && nir_src_is_const(intr->src[1]) &&
               nir_src_as_uint(intr->src[1]) == 0;
         for (unsigned mask = 0; mask < 256; ++mask) {
            unsigned killed = 0;
            ok &= eval_sample_mask(intr->src[0].ssa, mask, &killed) &&
                  killed == ((~(mask & api_mask)) & ((1u << samples) - 1));
         }
      }
   }
   nir_validate_shader(b.shader, "after hoisting G17P sample kills and depth");
   agx_nir_lower_g17p_tile_samples(b.shader, samples);
   agx_preprocess_nir(b.shader);

   struct agx_shader_key backend_key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .secondary = true,
      .no_preamble = true,
      .fs.nr_samples = samples,
   };
   struct agx_shader_part part = {0};
   const bool compiled = agx_compile_shader_nir(b.shader, &backend_key, &part);
   ok &= compiled;

   static const uint8_t wait1[] = {0x87, 0x02, 0x54, 0x01, 0x00, 0x00};
   static const uint8_t release1[] = {0x07, 0x02, 0x54, 0x01, 0x00, 0x00};
   const unsigned waits = count_bytes(&part, wait1, sizeof(wait1));
   const unsigned releases = count_bytes(&part, release1, sizeof(release1));
   ok &= waits == 1 && releases == 1;

   unsigned depth_writes = 0;
   const uint8_t *text = part.binary;
   for (size_t at = 0; text && at + 6 <= part.info.binary_size; ++at) {
      if (text[at] == 0xd7 && (text[at + 1] & 0x0f) == 0x04 &&
          text[at + 2] == 0x54 && text[at + 5] == 0x03)
         depth_writes++;
   }
   ok &= depth_writes == util_bitcount(api_mask);

   if (!ok) {
      fprintf(stderr,
              "dynamic depth metrics: samples=%u api_mask=%x hoisted=%u masks=%u waits=%u "
              "releases=%u depth=%u binary=%u compiled=%u\n",
              samples, api_mask, hoisted, sample_masks, waits, releases, depth_writes,
              part.info.binary_size, compiled);
   }

   free(part.binary);
   ralloc_free(b.shader);
   return ok;
}

static bool
test_unrolled_dynamic_depth_mask(void)
{
   for (unsigned samples = 2; samples <= 4; samples *= 2) {
      for (unsigned mask = 1; mask < (1u << samples); ++mask) {
         if (!test_unrolled_dynamic_depth_mask_case(samples, mask))
            return false;
      }
   }
   return true;
}

int
main(void)
{
   if (!test_disjoint_testing_masks()) {
      fprintf(stderr, "G17P testing-mask coalescing changed sample semantics\n");
      return 1;
   }
   if (!test_single_sample_shading()) {
      fprintf(stderr, "G17P 1x sample shading was not folded to pixel shading\n");
      return 1;
   }
   if (!test_unrolled_sample_shading()) {
      fprintf(stderr, "G17P samples were not unrolled into one transaction\n");
      return 1;
   }
   if (!test_unrolled_sample_depth(false, true) ||
       !test_unrolled_sample_depth(true, true) ||
       !test_unrolled_sample_depth(true, false)) {
      fprintf(stderr, "G17P per-sample depth writes were not staged per sample\n");
      return 1;
   }
   if (!test_unrolled_dynamic_depth_mask()) {
      fprintf(stderr, "G17P dynamic sample mask did not reach multisample DEPTHF\n");
      return 1;
   }
   if (!test_epilog_discard_policy()) {
      fprintf(stderr, "fragment epilog discard policy lost a tested axis\n");
      return 1;
   }
   if (!test_all_live_early_test_prolog()) {
      fprintf(stderr, "G17P all-live early-test prolog was not preserved\n");
      return 1;
   }

   nir_shader *legacy = build_early_demote_fs();
   struct agx_fs_epilog_link_info legacy_link = {0};
   if (!agx_nir_lower_fs_output_to_epilog_for_device(
          legacy, AGX_CHIP_G13G, &legacy_link) ||
       !legacy_link.sample_mask_after_force_early ||
       count_intrinsic(legacy, nir_intrinsic_demote_samples) != 0) {
      fprintf(stderr, "legacy split did not retain its epilog sample mask\n");
      return 1;
   }
   ralloc_free(legacy);

   nir_shader *main = build_early_demote_fs();
   struct agx_fs_epilog_link_info link = {0};
   if (!agx_nir_lower_fs_output_to_epilog_for_device(
          main, AGX_CHIP_G17P, &link) ||
       !link.sample_mask_after_force_early ||
       count_intrinsic(main, nir_intrinsic_demote_samples) != 0) {
      fprintf(stderr, "G17P split did not export its post-test sample mask\n");
      return 1;
   }

   nir_shader *legacy_alpha = build_alpha_one_fs();
   legacy_link = (struct agx_fs_epilog_link_info){0};
   agx_nir_lower_fs_output_to_epilog_for_device(
      legacy_alpha, AGX_CHIP_G13G, &legacy_link);
   nir_shader *g17p_alpha = build_alpha_one_fs();
   struct agx_fs_epilog_link_info g17p_alpha_link = {0};
   agx_nir_lower_fs_output_to_epilog_for_device(
      g17p_alpha, AGX_CHIP_G17P, &g17p_alpha_link);
   if (!legacy_link.loc0_w_1 || g17p_alpha_link.loc0_w_1 ||
       g17p_alpha_link.loc_written != BITFIELD_BIT(0) ||
       g17p_alpha_link.size_32 != BITFIELD_BIT(0)) {
      fprintf(stderr, "G17P split did not preserve its native alpha export\n");
      return 1;
   }
   ralloc_free(legacy_alpha);
   ralloc_free(g17p_alpha);

   agx_nir_lower_sample_mask(main);
   if (count_intrinsic(main, nir_intrinsic_demote_samples) != 0 ||
       count_intrinsic(main, nir_intrinsic_sample_mask_agx) != 1) {
      fprintf(stderr, "G17P forced-early test was not retained exactly once\n");
      return 1;
   }

   struct agx_fs_epilog_key epilog_key = {
      .link = link,
      .nr_samples = 1,
   };
   epilog_key.link.sample_shading = true;
   epilog_key.link.already_ran_zs = true;
   memset(epilog_key.remap, -1, sizeof(epilog_key.remap));
   epilog_key.remap[0] = 0;
   epilog_key.rt_formats[0] = PIPE_FORMAT_R8G8B8A8_UNORM;
   epilog_key.blend.rt[0].colormask = 0xf;
   epilog_key.blend.rt[0].mode =
      agx_pack_blend_standard(PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE,
                              PIPE_BLENDFACTOR_ZERO, PIPE_BLEND_ADD,
                              PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO);

   nir_builder b =
      nir_builder_init_simple_shader(0, &agx_nir_options, "G17P FS epilog");
   agx_nir_fs_epilog(&b, &epilog_key);
   if (count_dynamic_tile_samples(b.shader) != 1 ||
       agx_nir_lower_g17p_tile_samples(b.shader, 2) ||
       agx_nir_lower_g17p_tile_samples(b.shader, 4) ||
       agx_nir_lower_g17p_tile_samples(b.shader, 8) ||
       count_dynamic_tile_samples(b.shader) != 1 ||
       !agx_nir_lower_g17p_1x_tile_samples(b.shader) ||
       count_dynamic_tile_samples(b.shader) != 0 ||
       agx_nir_lower_g17p_1x_tile_samples(b.shader)) {
      fprintf(stderr, "G17P 1x sample-shaded epilog was not canonicalized\n");
      return 1;
   }
   agx_preprocess_nir(b.shader);

   struct agx_shader_key backend_key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .secondary = true,
      .no_preamble = true,
   };
   struct agx_shader_part part = {0};
   bool ok = agx_compile_shader_nir(b.shader, &backend_key, &part) &&
             has_lstcol(&part);
   if (!ok)
      fprintf(stderr, "G17P early-demote epilog did not compile to LSTCOL\n");

   free(part.binary);
   ralloc_free(b.shader);

   /* Fully disabled G17P colour writes use an identity blend to materialize
    * the existing tile value. Validate the exact load/store shape and backend
    * widths independently of the Vulkan key construction. */
   epilog_key.link.sample_shading = false;
   epilog_key.blend.rt[0].mode =
      agx_pack_blend_standard(PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ZERO,
                              PIPE_BLENDFACTOR_ONE, PIPE_BLEND_ADD,
                              PIPE_BLENDFACTOR_ZERO, PIPE_BLENDFACTOR_ONE);
   b = nir_builder_init_simple_shader(0, &agx_nir_options,
                                      "G17P masked colour identity epilog");
   agx_nir_fs_epilog(&b, &epilog_key);
   ok &= count_intrinsic(b.shader, nir_intrinsic_load_local_pixel_agx) == 1;
   ok &= count_intrinsic(b.shader, nir_intrinsic_store_local_pixel_agx) == 1;
   agx_preprocess_nir(b.shader);
   memset(&part, 0, sizeof(part));
   ok &= agx_compile_shader_nir(b.shader, &backend_key, &part) &&
         has_lstcol(&part);
   if (!ok)
      fprintf(stderr, "G17P masked colour identity epilog failed\n");
   free(part.binary);
   ralloc_free(b.shader);
   ralloc_free(main);
   return ok ? 0 : 1;
}
