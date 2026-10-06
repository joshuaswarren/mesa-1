/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_compile.h"
#include "agx_nir.h"
#include "nir_builder.h"
#include "util/half_float.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *
op_name(nir_texop op)
{
   switch (op) {
   case nir_texop_tex: return "tex";
   case nir_texop_txb: return "txb";
   case nir_texop_txl: return "txl";
   case nir_texop_txd: return "txd";
   default: UNREACHABLE("unexpected op");
   }
}

static bool
lower_sysvals(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   unsigned value;
   switch (intr->intrinsic) {
   case nir_intrinsic_load_api_sample_mask_agx:
   case nir_intrinsic_load_active_samples_agx:
      value = 0xff;
      break;
   case nir_intrinsic_load_shader_part_tests_zs_agx:
      value = 0;
      break;
   default:
      return false;
   }
   b->cursor = nir_before_instr(&intr->instr);
   nir_def_replace(&intr->def, nir_imm_intN_t(b, value, 16));
   return true;
}

static bool
check(nir_texop op, bool clamp, bool g17p, bool dynamic, bool reproduce)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "texture min LOD");
   b.shader->info.internal = true;

   unsigned count = 1 + clamp + (op == nir_texop_txd ? 2 : op != nir_texop_tex);
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, count);
   tex->op = op;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->dest_type = nir_type_float32;
   tex->coord_components = 2;
   unsigned i = 0;
   tex->src[i++] = nir_tex_src_for_ssa(
      nir_tex_src_coord, nir_load_preamble(&b, 2, 32, .base = 0));
   if (op == nir_texop_txb)
      tex->src[i++] = nir_tex_src_for_ssa(nir_tex_src_bias, nir_imm_float(&b, 0.5));
   if (op == nir_texop_txl)
      tex->src[i++] = nir_tex_src_for_ssa(nir_tex_src_lod, nir_imm_float(&b, 0.5));
   if (op == nir_texop_txd) {
      tex->src[i++] = nir_tex_src_for_ssa(nir_tex_src_ddx, nir_imm_vec2(&b, 0.01, 0));
      tex->src[i++] = nir_tex_src_for_ssa(nir_tex_src_ddy, nir_imm_vec2(&b, 0, 0.01));
   }
   if (clamp)
      tex->src[i++] = nir_tex_src_for_ssa(
         nir_tex_src_min_lod, dynamic ? nir_load_preamble(&b, 1, 32, .base = 2)
                                     : nir_imm_float(&b, 1.5));
   assert(i == count);
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);
   nir_store_local_pixel_agx(&b, &tex->def, nir_imm_intN_t(&b, 0xff, 16),
                             nir_undef(&b, 2, 16), .base = 0, .write_mask = 0xf,
                             .format = PIPE_FORMAT_R8_UNORM);
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);

   /* Image-view min LOD is added after early texture lowering in Honeykrisp.
    * Exercise the late pass directly, with sampler-bias lowering absent. */
   agx_nir_lower_texture_for_generation(b.shader, g17p ? 9 : 0);
   nir_validate_shader(b.shader, "after min LOD lowering");
   const int min = nir_tex_instr_src_index(tex, nir_tex_src_min_lod);
   const int packed = nir_tex_instr_src_index(tex, nir_tex_src_lod_bias_min_agx);
   const int lod = nir_tex_instr_src_index(tex, nir_tex_src_lod);
   bool ok = true;
   if (clamp && (op == nir_texop_tex || op == nir_texop_txb)) {
      ok = tex->op == nir_texop_txb && min < 0 && packed >= 0;
      if (ok && !dynamic) {
         nir_opt_constant_folding(b.shader);
         uint32_t bits = nir_src_as_uint(tex->src[packed].src);
         ok = (bits >> 16) == _mesa_float_to_half(1.5) &&
              (bits & 0xffff) == _mesa_float_to_half(op == nir_texop_txb ? 0.5 : 0);
      }
   } else if (clamp && op == nir_texop_txl) {
      ok = min < 0 && packed < 0 && lod >= 0;
      if (ok && !dynamic) {
         nir_opt_constant_folding(b.shader);
         ok = nir_src_as_float(tex->src[lod].src) == 1.5;
      }
   } else {
      ok = tex->op == op && (min >= 0) == (clamp && op == nir_texop_txd) &&
           packed < 0;
   }
   if (reproduce)
      nir_print_shader(b.shader, stderr);
   if (!ok && !reproduce) {
      fprintf(stderr, "FAIL lowering op=%s clamp=%u g17p=%u dynamic=%u\n",
              op_name(op), clamp, g17p, dynamic);
      nir_print_shader(b.shader, stderr);
      ralloc_free(b.shader);
      return false;
   }

   agx_preprocess_nir(b.shader);
   NIR_PASS(_, b.shader, agx_nir_lower_discard_zs_emit);
   NIR_PASS(_, b.shader, agx_nir_lower_sample_mask);
   NIR_PASS(_, b.shader, agx_nir_lower_interpolation, false);
   NIR_PASS(_, b.shader, nir_shader_intrinsics_pass, lower_sysvals,
            nir_metadata_control_flow, NULL);
   struct agx_shader_key key = {
      .dev.encoding = g17p ? AGX_ISA_ENCODING_AGX3B : AGX_ISA_ENCODING_AGX2,
      .dev.chip = g17p ? AGX_CHIP_G17P : AGX_CHIP_G13G,
      .no_preamble = true,
      .reserved_preamble = 4,
   };
   struct agx_shader_part part = {0};
   ok = agx_compile_shader_nir(b.shader, &key, &part) &&
        part.binary && part.info.binary_size > 0;
   printf("%s op=%s clamp=%u dynamic=%u: %s\n", g17p ? "G17P" : "G13",
          op_name(op), clamp, dynamic, ok ? "PASS" : "FAIL");
   free(part.binary);
   ralloc_free(b.shader);
   return ok;
}

int
main(int argc, char **argv)
{
   /* On an unfixed compiler, bypass the structural check and reach the
    * original backend assertion. The fixed compiler must emit a binary. */
   if (argc == 2 && !strcmp(argv[1], "--reproduce"))
      return !check(nir_texop_tex, true, true, false, true);
   if (argc != 1)
      return 2;
   static const nir_texop ops[] = {
      nir_texop_tex, nir_texop_txb, nir_texop_txl, nir_texop_txd,
   };
   for (unsigned g17p = 0; g17p < 2; ++g17p) {
      for (unsigned i = 0; i < ARRAY_SIZE(ops); ++i) {
         for (unsigned clamp = 0; clamp < 2; ++clamp) {
            for (unsigned dynamic = 0; dynamic < (clamp ? 2 : 1); ++dynamic) {
               if (!check(ops[i], clamp, g17p, dynamic, false))
                  return 1;
            }
         }
      }
   }
   return 0;
}
