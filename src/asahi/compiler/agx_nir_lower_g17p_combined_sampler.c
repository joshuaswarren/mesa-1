/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <stdlib.h>
#include <stdio.h>
#include "agx_compiler.h"
#include "agx_debug.h"
#include "agx_nir.h"
#include "agx_nir_texture.h"
#include "agx_compiler.h"
#include "agx_debug.h"

#include "nir.h"
#include "nir_builder.h"

#include <stdio.h>

/* Diagnostic (AGX_G17P_FORCE_SAMPLER_INDEX=<n>): publish the sampler heap index as a
 * wave-uniform scalar instead of a per-lane value. G17P returns the raw texel
 * with no depth comparison applied, for whole 32-lane waves at a time, on the
 * last of four indexed array-shadow samples; the texture index is read
 * correctly in the same sample, so the two descriptor paths behave
 * differently. Pinning the heap index to a constant removes the per-lane index path
 * entirely and tells a bad sampler index apart from a bad comparison. This is only semantically
 * valid where the index is already dynamically uniform, so it is a diagnostic,
 * never a production path. */
static int
agx3_g17p_forced_sampler_index(void)
{
   static int value = -2;
   if (value == -2) {
      const char *e = getenv("AGX_G17P_FORCE_SAMPLER_INDEX");
      value = (e && e[0]) ? atoi(e) : -1;
   }
   return value;
}

/* Pre-raster graphics stages have no implicit derivatives. Their Poly
 * compute variants use the same native explicit-LOD texture operations. */
static bool
g17p_explicit_lod_stage(mesa_shader_stage stage)
{
   return stage == MESA_SHADER_VERTEX || stage == MESA_SHADER_GEOMETRY ||
          stage == MESA_SHADER_TESS_CTRL || stage == MESA_SHADER_TESS_EVAL;
}

/* agx_nir_lower_g17p_indexed_samplers() is all-or-nothing per shader: one
 * unadmitted texture or image instruction and NOTHING in that shader gets a
 * heap index, so every texture instruction in it then reaches the packer with
 * a 32-bit bindless handle it has no encoding for.  A refusal census therefore
 * measures shaders, not instructions, and says nothing about which shape
 * actually caused it.  Name the instruction that did.
 */
static bool
g17p_decline(const char *why, const nir_tex_instr *tex,
             const nir_intrinsic_instr *intr, mesa_shader_stage stage)
{
   if (!(agx_compiler_debug & (AGX_DBG_SHADERS | AGX_DBG_REFUSE)))
      return false;

   if (tex) {
      fprintf(stderr,
              "G17P indexed-sampler decline (%s): stage=%u op=%u dim=%u "
              "array=%u shadow=%u sparse=%u coords=%u srcs=%u dest_type=%u "
              "comps=%u bits=%u flags=0x%x tex_idx=%u smp_idx=%u\n",
              why, stage, tex->op, tex->sampler_dim, tex->is_array,
              tex->is_shadow, tex->is_sparse, tex->coord_components,
              tex->num_srcs, tex->dest_type, tex->def.num_components,
              tex->def.bit_size, tex->backend_flags, tex->texture_index,
              tex->sampler_index);
   } else if (intr) {
      fprintf(stderr,
              "G17P indexed-sampler decline (%s): stage=%u image_load dim=%u "
              "array=%u comps=%u bits=%u fmt=%u access=0x%x coordc=%u "
              "coordb=%u lod_const=%u ms_const=%u handle_const=%u\n",
              why, stage, nir_intrinsic_image_dim(intr),
              nir_intrinsic_image_array(intr), intr->def.num_components,
              intr->def.bit_size, (unsigned)nir_intrinsic_format(intr),
              nir_intrinsic_access(intr),
              nir_src_num_components(intr->src[1]),
              nir_src_bit_size(intr->src[1]),
              nir_src_is_const(intr->src[2]),
              nir_src_is_const(intr->src[3]),
              nir_src_as_intrinsic(intr->src[0]) &&
                 nir_src_is_const(nir_src_as_intrinsic(intr->src[0])->src[0]));
   }
   return false;
}

static bool
g17p_sample_shape(const nir_tex_instr *tex)
{
   const bool query_lod = tex->op == nir_texop_lod;
   const bool three_d = tex->sampler_dim == GLSL_SAMPLER_DIM_3D;
   const bool cube = tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE;
   const bool cube_array = cube && tex->is_array;
   const bool offset_array = tex->is_array && !cube &&
      (nir_tex_instr_src_index(tex, nir_tex_src_offset) >= 0 ||
       nir_tex_instr_src_index(tex, nir_tex_src_backend2) >= 0);
   const int level_src = nir_tex_instr_src_index(tex, nir_tex_src_lod);
   const int bias_src = nir_tex_instr_src_index(tex, nir_tex_src_bias);
   /* Native G17P compute depthcube/depth2d_array sample_compare(level(0))
    * keeps the ordinary face/layer staging and selects SMP f139=2. */
   const bool shadow_level_zero = tex->is_shadow && !cube_array &&
      tex->op == nir_texop_txl && level_src >= 0 &&
      nir_src_is_const(tex->src[level_src].src) &&
      nir_src_as_float(tex->src[level_src].src) == 0.0;
   /* A per-lane level or bias is published beside the operand its dimension
    * already owns -- the layer for a noncube array (agx3_tex_stage_array_lod),
    * CEM's face for a cube (agx3_tex_stage_cube_lod) -- so it is the runtime
    * value, not a constant, that the comparison forms admit. A constant level
    * or bias would instead have to share SMP's immediate f29/f43 slot with
    * that same operand, which is the collision agx3_emit_texture_sample()
    * declines. */
   const bool register_level = tex->op == nir_texop_txl && level_src >= 0 &&
      !nir_src_is_const(tex->src[level_src].src);
   const bool register_bias = tex->op == nir_texop_txb && bias_src >= 0 &&
      !nir_src_is_const(tex->src[bias_src].src);

   if ((tex->op != nir_texop_tex && tex->op != nir_texop_txb &&
       tex->op != nir_texop_txl && tex->op != nir_texop_txd &&
       tex->op != nir_texop_tg4 && !query_lod) ||
       (tex->sampler_dim != GLSL_SAMPLER_DIM_2D && !three_d && !cube) ||
       /* 3D takes the implicit, immediate-bias and per-lane LOD shapes the
        * 2D dimension takes; only gradient3d, which stages six scalars
        * rather than four, has no witness. */
       (three_d &&
        (tex->is_array || tex->is_shadow ||
         (tex->op != nir_texop_tex && tex->op != nir_texop_txb &&
          tex->op != nir_texop_txl && tex->op != nir_texop_txd && !query_lod))) ||
       /* A cube comparison publishes its reference through the same CEM
        * bracket the colour cube uses, with implicit or explicit zero LOD.
        * A colour cube also takes an explicit per-lane level, on the
        * plain cube exactly as on the cube array -- Apple's own
        * t.sample(s, dir, level(reg)) for texturecube<float> is
        *
        *   05800cb8b0000003000010000100
        *
        * which is the instruction agx3_tex_encode_sample() already emits for
        * AGX3_TEX_LOD_REGISTER on a cube (test-agx3-tex-core.c, "native cube
        * register level"), field for field. Admitting only the array form
        * here was what refused textureLod(samplerCube, dir, lod) -- the
        * roughness-indexed probe lookup every image-based lighting shader
        * does, SuperTuxKart's IBL.frag included. */
       (cube &&
        (tex->op != nir_texop_tex && tex->op != nir_texop_tg4 && !query_lod &&
         !((!tex->is_shadow || shadow_level_zero ||
            (!cube_array && level_src >= 0 &&
             !nir_src_is_const(tex->src[level_src].src))) &&
            tex->op == nir_texop_txl) &&
         !(tex->op == nir_texop_txb &&
           (!tex->is_shadow ||
            nir_tex_instr_src_index(tex, nir_tex_src_lod_bias_min_agx) >= 0 ||
            register_bias) &&
           !(tex->is_shadow && cube_array)))) ||
       /* Sparse residency rides on the ordinary sample: the encoder sets
        * three extra fields and the instruction writes one more register.
        * A depth comparison and a LOD query have no sparse witness. */
       (tex->is_sparse && (tex->is_shadow || query_lod)) ||
       tex->coord_components != (query_lod ? (three_d || cube ? 3 : 2) :
                                cube_array ? 4 :
                                (tex->is_array || three_d || cube) ? 3 : 2) ||
       (tex->backend_flags & ~(AGX_TEXTURE_FLAG_CLAMP_TO_0 |
                               AGX_TEXTURE_FLAG_NO_CLAMP)) != 0 ||
       tex->backend_flags == (AGX_TEXTURE_FLAG_CLAMP_TO_0 |
                              AGX_TEXTURE_FLAG_NO_CLAMP) ||
       (tex->dest_type != nir_type_float32 &&
        tex->dest_type != nir_type_uint32 &&
        tex->dest_type != nir_type_int32) ||
       tex->def.num_components !=
          ((query_lod ? 2 : (tex->is_shadow && tex->op != nir_texop_tg4) ? 1 : 4) +
           (tex->is_sparse ? 1 : 0)) ||
       (query_lod && tex->dest_type != nir_type_float32) ||
       tex->def.bit_size != 32)
      return false;

   const int packed_bias_min =
      nir_tex_instr_src_index(tex, nir_tex_src_lod_bias_min_agx);
   const int minimum = nir_tex_instr_src_index(tex, nir_tex_src_min_lod);

   if (tex->is_shadow && !query_lod) {
      /* The depth comparison is orthogonal to the LOD shape: Apple's
       * depth2d sample_compare with level(k), level(reg), bias(k), bias(reg)
       * and gradient2d() each differ from the corresponding colour sample by
       * f143 alone, and gather_compare is the ordinary gather plus f143.
       * See the witness table in agx3_tex_core.c. The array and cube
       * comparisons use separately packed layer / CEM operands and admit
       * only implicit LOD, the native explicit-zero form above, or a
       * clamped bias or gradient, which publish their minimum beside the
       * layer or in a packed register. */
      const int comparator =
         nir_tex_instr_src_index(tex, nir_tex_src_comparator);
      if (comparator < 0 ||
          nir_src_num_components(tex->src[comparator].src) != 1 ||
          nir_src_bit_size(tex->src[comparator].src) != 32 ||
          (tex->is_array && !offset_array && tex->op != nir_texop_tex &&
           tex->op != nir_texop_tg4 &&
           !(tex->op == nir_texop_txb && packed_bias_min >= 0) &&
           /* GLSL's remaining comparison LOD shapes for an array depth
            * sampler: textureLod and texture(..., bias) on
            * sampler1DArrayShadow, and textureGrad on sampler1DArrayShadow
            * and sampler2DArrayShadow -- a gradient now with or without a
            * minimum. Each stages its scalar in a word the packer has
            * emitted since the colour forms landed: the level or bias beside
            * the layer, the gradient in banks 8..14 with the layer still on
            * PACK_SMPIR bank 6. Only this gate held them back, and refusing
            * them here does not fall back to anything -- the pass is
            * all-or-nothing per shader, so every texture instruction in the
            * shader then reached the packer as a software-heap bindless
            * handle, which has no AGX3 encoding, and the pipeline failed
            * with VK_ERROR_UNKNOWN. */
           !register_level && !register_bias &&
           tex->op != nir_texop_txd &&
           !shadow_level_zero))
         return false;
   }

   const bool has_minimum = packed_bias_min >= 0 || minimum >= 0;
   const bool has_bias = packed_bias_min >= 0 ||
      nir_tex_instr_src_index(tex, nir_tex_src_bias) >= 0;
   const bool has_lod = nir_tex_instr_src_index(tex, nir_tex_src_lod) >= 0;
   const int ddx = nir_tex_instr_src_index(tex, nir_tex_src_ddx);
   const int ddy = nir_tex_instr_src_index(tex, nir_tex_src_ddy);
   const bool has_grad = ddx >= 0 && ddy >= 0;
   /* Constant coordinate offsets use one immediate nibble per coordinate in
    * SMP (f43/f51/f141=1). Runtime offsets keep their component vector for
    * PACK_SMPIR and select the staged form with f141=2; this includes native
    * gathers as well as ordinary 2D samples. */
   int offset_src = nir_tex_instr_src_index(tex, nir_tex_src_offset);
   const bool backend_offset = offset_src < 0;
   if (backend_offset)
      offset_src = nir_tex_instr_src_index(tex, nir_tex_src_backend2);
   const bool has_offset = offset_src >= 0;
   const bool packed_offset =
      has_offset && backend_offset &&
      nir_src_num_components(tex->src[offset_src].src) == 1;
   const bool staged_packed_offset =
      packed_offset && !nir_src_is_const(tex->src[offset_src].src) &&
      (tex->op == nir_texop_tg4 || tex->is_array || has_lod || has_bias ||
       has_minimum);
   const uint32_t constant_offset =
      has_offset && nir_src_is_const(tex->src[offset_src].src)
         ? nir_src_as_uint(tex->src[offset_src].src)
         : 0;
   /* lower_regular_texture() keeps G17P gather X/Y as two 16-bit lanes so a
    * runtime value can feed PACK_SMPIR directly.  A constant therefore has
    * its signed nibbles in bits 0..3 and 16..19 (for example, CTS offset
    * (+8,+8) is 0x00080008), not in the compact SMP byte used after
    * agx_emit_tex() folds it.  Admit exactly that generated representation;
    * the backend later maps it to (y << 4) | x. */
   const bool valid_constant_offset =
      tex->op == nir_texop_tg4
         ? (constant_offset & ~0x000f000fu) == 0
         : constant_offset <= (three_d ? 0xfffu : 0xffu);
   /* Native bank-6 minimum operands are grounded for 2D, 2D-array and 3D
    * fragment samples, and for biased cube samples. The shared texture lowering has already combined a
    * bias/minimum pair into one f16x2 source, or left the gradient minimum
    * as a scalar half. Preserve that source while selecting native handles.
    * A 2D runtime offset carries the bias or gradient minimum in its packed
    * bank-6 word; an unpacked component offset would overwrite it. A runtime
    * 3D offset is witnessed beside a bias only. */
   if (has_minimum &&
       ((cube && packed_bias_min < 0) ||
        (three_d && has_offset && packed_bias_min < 0 &&
         !nir_src_is_const(tex->src[offset_src].src)) ||
        (has_offset && !three_d && !packed_offset &&
         !nir_src_is_const(tex->src[offset_src].src)) ||
        (packed_bias_min >= 0 &&
         (tex->op != nir_texop_txb || minimum >= 0 ||
          nir_tex_instr_src_index(tex, nir_tex_src_bias) >= 0 ||
          nir_src_num_components(tex->src[packed_bias_min].src) != 1 ||
          nir_src_bit_size(tex->src[packed_bias_min].src) != 32)) ||
        (minimum >= 0 &&
         (tex->op != nir_texop_txd ||
          nir_src_num_components(tex->src[minimum].src) != 1 ||
          nir_src_bit_size(tex->src[minimum].src) != 16))))
      return false;
   if (query_lod && has_offset)
      return false;
   if (has_offset &&
       (nir_src_num_components(tex->src[offset_src].src) !=
           (packed_offset ? 1u : three_d ? 3u : 2u) ||
        (!packed_offset && three_d) ||
        (packed_offset && !staged_packed_offset &&
         (!nir_src_is_const(tex->src[offset_src].src) ||
           !valid_constant_offset)) ||
         cube))
      return false;

   const unsigned expected_srcs =
      (has_grad ? 5 : ((has_bias || has_lod) ? 4 : 3)) +
      (tex->is_shadow && !query_lod) +
      has_offset + (minimum >= 0);
   if (tex->num_srcs != expected_srcs ||
       has_lod != (tex->op == nir_texop_txl) ||
       has_bias != (tex->op == nir_texop_txb) ||
       has_grad != (tex->op == nir_texop_txd))
      return false;

   /* Metal's gather takes no LOD argument, so no gather carries one. The
    * array layer, the coordinate offset and the depth comparison all
    * compose with it. Cube gathers use the existing CEM coordinate path;
    * 3D gathers remain unsupported. */
   if (tex->op == nir_texop_tg4 && (three_d || tex->component > 3))
      return false;

   /* An explicit gradient is four 32-bit scalars staged into their own
    * operand banks. G17P array gradients name the independent dynamic layer
    * register directly in the sample instruction. */
   const unsigned grad_components = three_d ? 3u : 2u;
   if (has_grad &&
       (nir_src_num_components(tex->src[ddx].src) != grad_components ||
        nir_src_num_components(tex->src[ddy].src) != grad_components ||
        nir_src_bit_size(tex->src[ddx].src) != 32 ||
        nir_src_bit_size(tex->src[ddy].src) != 32))
      return false;

   int coord = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   if (coord < 0)
      coord = nir_tex_instr_src_index(tex, nir_tex_src_backend1);

   return coord >= 0 &&
          nir_src_num_components(tex->src[coord].src) ==
             tex->coord_components &&
          nir_src_bit_size(tex->src[coord].src) == 32 &&
          nir_tex_instr_src_index(tex, nir_tex_src_texture_handle) >= 0 &&
          nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle) >= 0;
}

static bool
g17p_handle_binding(const nir_tex_instr *tex, nir_tex_src_type source,
                    nir_intrinsic_op intrinsic, uint8_t *set,
                    uint32_t *offset_B)
{
   const int index = nir_tex_instr_src_index(tex, source);
   if (index < 0)
      return false;

   nir_intrinsic_instr *handle = nir_src_as_intrinsic(tex->src[index].src);
   if (!handle || handle->intrinsic != intrinsic ||
       nir_intrinsic_desc_set(handle) > UINT8_MAX ||
       !nir_src_is_const(handle->src[0]))
      return false;

   *set = nir_intrinsic_desc_set(handle);
   *offset_B = nir_src_as_uint(handle->src[0]);
   return true;
}

static bool
g17p_handle_offset(const nir_tex_instr *tex, nir_tex_src_type source,
                   nir_intrinsic_op intrinsic, uint32_t *offset_B)
{
   uint8_t set;
   return g17p_handle_binding(tex, source, intrinsic, &set, offset_B) &&
          set == 0;
}

static bool
g17p_is_set_address(nir_def *def, uint8_t set)
{
   nir_scalar address = nir_scalar_resolved(def, 0);
   nir_intrinsic_instr *load =
      nir_def_as_intrinsic_or_null(address.def);

   return address.comp == 0 && load &&
          load->intrinsic == nir_intrinsic_load_descriptor_set_agx &&
          nir_intrinsic_desc_set(load) == set;
}

static bool
g17p_is_constant_scalar(nir_scalar offset, uint32_t expected)
{
   offset = nir_scalar_chase_movs(offset);
   if (nir_scalar_is_const(offset))
      return nir_scalar_as_uint(offset) == expected;

   nir_alu_instr *convert = nir_def_as_alu_or_null(offset.def);
   if (offset.comp != 0 || !convert || convert->op != nir_op_u2u64)
      return false;

   return g17p_is_constant_scalar(nir_scalar_chase_alu_src(offset, 0),
                                  expected);
}

static bool
g17p_is_constant_offset(nir_def *def, uint32_t expected)
{
   return g17p_is_constant_scalar(nir_get_scalar(def, 0), expected);
}

/* Match only the driver-generated software LOD-bias field belonging to this
 * sampled descriptor. API samplers carry the same bias in their G17P native
 * sampler word, so this zero-valued software load must not become a txb.
 * A shader with no bias source at all has nothing to classify and passes;
 * g17p_sample_shape() has already tied the presence of the source to txb.
 */
static bool
g17p_is_sampler_bias(const nir_tex_instr *tex, uint8_t set,
                     uint32_t expected_offset_B)
{
   const int index = nir_tex_instr_src_index(tex, nir_tex_src_bias);
   if (index < 0)
      return tex->op != nir_texop_txb;

   nir_intrinsic_instr *load = nir_src_as_intrinsic(tex->src[index].src);
   if (!load || load->def.num_components != 1 || load->def.bit_size != 16)
      return false;

   if (load->intrinsic == nir_intrinsic_load_global_constant_offset ||
       load->intrinsic == nir_intrinsic_load_global_constant_bounded) {
      return g17p_is_set_address(load->src[0].ssa, set) &&
             g17p_is_constant_offset(load->src[1].ssa, expected_offset_B);
   }

   if (load->intrinsic != nir_intrinsic_load_global_constant)
      return false;

   nir_scalar address = nir_scalar_resolved(load->src[0].ssa, 0);
   nir_alu_instr *add = nir_def_as_alu_or_null(address.def);
   if (address.comp != 0 || !add || add->op != nir_op_iadd)
      return false;

   nir_scalar lhs = nir_scalar_chase_alu_src(address, 0);
   nir_scalar rhs = nir_scalar_chase_alu_src(address, 1);

   return (g17p_is_set_address(lhs.def, set) &&
           g17p_is_constant_scalar(rhs, expected_offset_B)) ||
          (g17p_is_set_address(rhs.def, set) &&
           g17p_is_constant_scalar(lhs, expected_offset_B));
}

/* The native texture/sampler table prefix is separate from the fragment
 * push-root ABI. Residual descriptor accesses need their own lowering, but a
 * root load used by push constants is valid alongside this sampled resource. */
static bool
g17p_has_descriptor_access(const nir_shader *shader)
{
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            const nir_intrinsic_op op =
               nir_instr_as_intrinsic(instr)->intrinsic;
            if (op == nir_intrinsic_bindless_image_agx ||
                op == nir_intrinsic_bindless_sampler_agx ||
                op == nir_intrinsic_load_descriptor_set_agx)
               return true;
         }
      }
   }

   return false;
}

/* Array-coordinate legalization reads the texture descriptor to clamp the
 * API layer before the sample.  The combined-sampler fragment ABI already
 * publishes that exact 32-byte descriptor pointer in preamble halves 0..3;
 * retaining the original descriptor-set handle as well would require a
 * second, unavailable root.  Replace only compiler-generated descriptor
 * crawls of the selected texture with the native pointer.  Any independent
 * handle/use remains below and makes the specialization fail closed. */
static bool
g17p_rewrite_combined_texture_crawls(nir_shader *shader, uint8_t set,
                                     uint32_t texture_offset_B)
{
   nir_foreach_function_impl(impl, shader) {
      nir_builder b = nir_builder_create(impl);
      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *load = nir_instr_as_intrinsic(instr);
            if (load->intrinsic !=
                nir_intrinsic_load_from_texture_handle_agx)
               continue;

            nir_intrinsic_instr *handle = nir_src_as_intrinsic(load->src[0]);
            if (!handle ||
                handle->intrinsic != nir_intrinsic_bindless_image_agx ||
                nir_intrinsic_desc_set(handle) != set ||
                !nir_src_is_const(handle->src[0]) ||
                nir_src_as_uint(handle->src[0]) != texture_offset_B)
               return false;

            b.cursor = nir_before_instr(instr);
            nir_def_replace(&load->def,
                            nir_load_preamble(&b, 1, 64, .base = 0));
         }
      }
   }

   return true;
}

bool
agx_nir_lower_g17p_combined_sampler(
   nir_shader *shader, struct agx_g17p_combined_sampler_binding *binding)
{
   if (!shader || !binding ||
       (shader->info.stage != MESA_SHADER_FRAGMENT &&
        shader->info.stage != MESA_SHADER_COMPUTE))
      return false;

   nir_tex_instr *sample = NULL;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_tex)
               continue;

            if (sample || !g17p_sample_shape(nir_instr_as_tex(instr)))
               return false;
            sample = nir_instr_as_tex(instr);
         }
      }
   }

   if (!sample)
      return false;

   uint32_t texture_offset_B, sampler_state_offset_B;
   if (!g17p_handle_offset(sample, nir_tex_src_texture_handle,
                           nir_intrinsic_bindless_image_agx,
                           &texture_offset_B) ||
       !g17p_handle_offset(sample, nir_tex_src_sampler_handle,
                           nir_intrinsic_bindless_sampler_agx,
                           &sampler_state_offset_B))
      return false;

   /* Gen9 sampled descriptors contain one 32-byte texture followed by the
    * eight-byte native sampler word and its 16-bit global-heap index.
    */
   if ((texture_offset_B & 7) != 0 ||
       sampler_state_offset_B != texture_offset_B + 32)
      return false;

   /* The Gen9 software LOD bias is at byte 12 of sampler state, and the
    * native sampler word already applies it, so that load -- and only that
    * load -- is dropped. A genuine API bias is a different value entirely:
    * keep it, and keep the sample a txb, so the hardware applies both the
    * descriptor bias and the shader's. */
   nir_steal_tex_src(sample, nir_tex_src_texture_handle);
   nir_steal_tex_src(sample, nir_tex_src_sampler_handle);
   if (sample->op == nir_texop_txb &&
       g17p_is_sampler_bias(sample, 0, sampler_state_offset_B + 12)) {
      nir_steal_tex_src(sample, nir_tex_src_bias);
      sample->op = nir_texop_tex;
   }
   sample->texture_index = 0;
   /* HAL200's bounded fragment ABI binds independent texture/sampler table
    * pointers. Each table is rebased to the selected API resource, so both
    * hardware indices are zero; the legacy texel-fetch reservation is not
    * an extra entry in this sampler table.
    */
   sample->sampler_index = 0;

   if (!g17p_rewrite_combined_texture_crawls(shader, 0, texture_offset_B))
      return false;

   NIR_PASS(_, shader, nir_opt_dce);
   /* The initial fragment ABI replaces the whole descriptor root. Compute
    * reserves the same native texture/sampler pointer prefix but may keep
    * unrelated storage-image/SSBO descriptors in the ordinary root. */
   if (shader->info.stage == MESA_SHADER_FRAGMENT &&
       g17p_has_descriptor_access(shader))
      return false;

   *binding = (struct agx_g17p_combined_sampler_binding){
      .texture_offset_B = texture_offset_B,
      .sampler_index_offset_B = sampler_state_offset_B + 8,
   };
   return true;
}

/* Ordinary native integer-coordinate read. The descriptor selects float,
 * signed, or unsigned texel interpretation; the result is a 32-bit register
 * value in each case. Input attachments arrive as 2D-array reads. */
static bool
g17p_indexed_fetch_shape(const nir_tex_instr *tex)
{
   const bool multisample = tex->op == nir_texop_txf_ms;
   const bool three_d = tex->sampler_dim == GLSL_SAMPLER_DIM_3D;
   int coord = nir_tex_instr_src_index(tex, nir_tex_src_backend1);
   if (coord < 0)
      coord = nir_tex_instr_src_index(tex, nir_tex_src_coord);

   /* Buffer textures are lowered to a 2D hardware texture before this pass.
    * That lowering deliberately preserves coord_components == 1 while its
    * backend coordinate is the remapped vec2.  Admit exactly that canonical
    * mismatch alongside ordinary 2D/2D-array fetches.
    */
   const bool lowered_buffer =
      tex->sampler_dim == GLSL_SAMPLER_DIM_2D && !tex->is_array &&
      tex->coord_components == 1 && coord >= 0 &&
      tex->src[coord].src.ssa->num_components == 2;

   if ((tex->op != nir_texop_txf && !multisample) ||
       tex->sampler_dim !=
          (multisample ? GLSL_SAMPLER_DIM_MS
                       : three_d ? GLSL_SAMPLER_DIM_3D
                                 : GLSL_SAMPLER_DIM_2D) ||
       (three_d && tex->is_array) ||
       tex->is_shadow || tex->texture_index ||
       /* No sparse multisample read is witnessed. */
       (tex->is_sparse && multisample) ||
       tex->sampler_index || tex->def.bit_size != 32 ||
       tex->def.num_components != (tex->is_sparse ? 5u : 4u) ||
       (tex->dest_type != nir_type_float32 && tex->dest_type != nir_type_int32 &&
        tex->dest_type != nir_type_uint32) ||
       (!lowered_buffer &&
        tex->coord_components != (tex->is_array || three_d ? 3 : 2)) ||
       (tex->backend_flags & ~AGX_TEXTURE_FLAG_NO_CLAMP))
      return false;
   const int handle = nir_tex_instr_src_index(tex, nir_tex_src_texture_handle);
   const int lod = nir_tex_instr_src_index(tex, nir_tex_src_lod);
   const int sample =
      nir_tex_instr_src_index(tex, nir_tex_src_ms_index);
   /* The backend stages a dynamic uint level independently of the texture
    * result for 2D, array, and 3D fetches. Keep that SSA value through descriptor
    * lowering; leaving the old base/byte-offset handle in place is not an
    * encoding of the native texture heap index. The fetch instruction's 3D
    * selector is orthogonal to the integer-LOD PACK_SMPIR transaction. */
   const bool valid_lod = lod >= 0 &&
      nir_src_num_components(tex->src[lod].src) == 1 &&
      (nir_src_bit_size(tex->src[lod].src) == 16 ||
       nir_src_bit_size(tex->src[lod].src) == 32) &&
      (nir_src_is_const(tex->src[lod].src)
         ? nir_src_as_uint(tex->src[lod].src) <= 63
         : (nir_src_num_components(tex->src[lod].src) == 1 &&
            (nir_src_bit_size(tex->src[lod].src) == 16 ||
             nir_src_bit_size(tex->src[lod].src) == 32)));
   const unsigned lowered_coord_components =
      coord >= 0 ? nir_src_num_components(tex->src[coord].src) : 0;
   return handle >= 0 && coord >= 0 &&
      (multisample
          ? (tex->num_srcs == (lod >= 0 ? 3 : 2) && sample < 0 &&
             (lod < 0 || (nir_src_is_const(tex->src[lod].src) &&
                          nir_src_as_uint(tex->src[lod].src) == 0)) &&
             /* agx_nir_lower_texture packs sample and optional array layer
              * into the third 32-bit backend coordinate. */
             lowered_coord_components == 3)
          : (tex->num_srcs == 3 && sample < 0 && valid_lod)) &&
      nir_src_bit_size(tex->src[coord].src) == 32 &&
      (multisample || lowered_coord_components ==
                         (lowered_buffer ? 2 : tex->coord_components));
}

/* Descriptor lowering normally wraps the software heap index in a bindless
 * sampler. The custom-border zero-clamp sample already carries the companion
 * index directly: its descriptor load and nonnegative runtime branch must
 * remain intact. Both indices include the same AGX2 reservation of 28. */
static nir_def *
g17p_sample_heap_index(const nir_tex_instr *tex)
{
   const int index = nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle);
   if (index < 0)
      return NULL;

   nir_src handle = tex->src[index].src;
   nir_intrinsic_instr *sampler = nir_src_as_intrinsic(handle);
   nir_def *value = NULL;
   if (sampler && sampler->intrinsic == nir_intrinsic_bindless_sampler_agx)
      value = sampler->src[1].ssa;
   else if (tex->backend_flags & AGX_TEXTURE_FLAG_CLAMP_TO_0)
      value = handle.ssa;

   return value && value->num_components == 1 && value->bit_size == 16
             ? value : NULL;
}

/* Read-only storage images use the same native texture heap and texel-fetch
 * sampler as ordinary indexed txf. The software descriptor tail supplies the
 * heap index; the original handle remains live for robust descriptor bounds
 * loads, which are independent of the hardware fetch operand. */
static bool
g17p_indexed_image_fetch_shape(const nir_intrinsic_instr *intr)
{
   /* No format condition.  This predicate only decides whether the image's
    * descriptor handle is replaced by a heap index; the rewrite is
    * format-agnostic and agx3_emit_texture_fetch() never reads I->format --
    * its only uses of the field are in agx3_emit_storage_image_write(), a
    * different instruction.  Apple agrees: metal-nt -arch applegpu_g17p emits
    * BYTE-IDENTICAL code for texture2d<float>, texture2d<uint> and
    * texture2d<int> read(), one variable per kernel,
    *
    *   05 80 0c b8 b0 00 17 01 00 00 00 00 01 00
    *
    * so the component type lives in the descriptor and the fetch has no field
    * for it.  The old PIPE_FORMAT_R8G8B8A8_UNORM equality was the format of
    * the kernel this was first witnessed against, not a constraint, and
    * because the pass is all-or-nothing per shader it took every OTHER
    * texture instruction in the shader down with it: an r32ui or r32i
    * imageLoad -- what every compute shader and every WebGPU storage texture
    * uses -- refused the whole shader.
    */
   /* Admit exactly the dimensions agx3_emit_texture_fetch() encodes for an
    * image load: 2D, 2D array and 3D. Vulkan storage cube views are packed as
    * 2D-array descriptors by hk_image_view, and agx_emit_image_load() already
    * maps both cube forms onto AGX_DIM_2D_ARRAY. They therefore use the same
    * qualified fetch as an ordinary array image; rejecting the original NIR
    * dimension here only leaves its unencodable software descriptor handle in
    * place. Multisample uses the emitter's native sample/layer packing and
    * may have a nonzero or dynamic sample index. */
   const enum glsl_sampler_dim image_dim = nir_intrinsic_image_dim(intr);
   const bool ms = image_dim == GLSL_SAMPLER_DIM_MS;
   /* OpImageSparseRead arrives as bindless_image_sparse_load: the same fetch
    * with one more destination component for the residency code.  Without
    * this the pass leaves the software heap-index handle in place and the
    * packer refuses the shape it cannot encode. */
   const bool sparse = intr->intrinsic == nir_intrinsic_bindless_image_sparse_load;
   if ((intr->intrinsic != nir_intrinsic_bindless_image_load && !sparse) ||
       (sparse && ms) ||
       !(ms || image_dim == GLSL_SAMPLER_DIM_2D ||
         image_dim == GLSL_SAMPLER_DIM_CUBE ||
         (image_dim == GLSL_SAMPLER_DIM_3D &&
          !nir_intrinsic_image_array(intr))) ||
       intr->def.num_components < 1 ||
       intr->def.num_components > (sparse ? 5u : 4u) ||
       (sparse && intr->def.num_components != 5) ||
       intr->def.bit_size != 32 ||
       nir_src_num_components(intr->src[1]) != 4 ||
       nir_src_bit_size(intr->src[1]) != 32 ||
       (!ms && (!nir_src_is_const(intr->src[2]) ||
                nir_src_as_uint(intr->src[2]) != 0)) ||
       !nir_src_is_const(intr->src[3]) ||
       nir_src_as_uint(intr->src[3]) != 0 ||
       (nir_intrinsic_access(intr) & ACCESS_NON_READABLE))
      return false;

   /* Coherent and volatile reads take the same fetch.  G17P's texture read
    * has no coherency field: metal-nt -arch applegpu_g17p compiles a
    * read_write texture's read() after a write() to the same smp form as an
    * access::read texture's read(), differing only in the staged coordinate
    * halves and the result register,
    *
    *   texture2d<int, access::read>        05800cb8b0001701000000000100
    *   texture2d<int, access::read_write>  05800cb9b0001701000000000200
    *
    * and the preceding store's fence already retires the write before any
    * later texture instruction is issued. */

   /* The descriptor offset does not have to be a constant.  The rewrite below
    * is pure SSA arithmetic -- nir_iadd_imm() onto the offset, then one
    * nir_load_global_constant() with .align_mul = 4 -- and runs the same for a
    * dynamically indexed descriptor as for a constant one.  The texture path
    * (g17p_indexed_fetch_shape) never required constness in the first place,
    * so requiring it here was an asymmetry, not a constraint.
    *
    * Keep the 8-alignment assertion wherever it can still be evaluated.  It is
    * stronger than the load needs; every descriptor stride the layout produces
    * is a multiple of 8, so base + index * stride + 64 stays 8-aligned for a
    * dynamic index too.
    */
   nir_intrinsic_instr *image = nir_src_as_intrinsic(intr->src[0]);
   return image && image->intrinsic == nir_intrinsic_bindless_image_agx &&
          nir_intrinsic_desc_set(image) <= UINT8_MAX &&
          (!nir_src_is_const(image->src[0]) ||
           (nir_src_as_uint(image->src[0]) & 7) == 0);
}

/* Native indexed samples consume zero-extended heap indices, loaded by the
 * GPU from the descriptor set. Keep descriptor-set roots alive so updates
 * after command recording and dynamically selected array elements are visible.
 */
bool
agx_nir_lower_g17p_indexed_samplers(nir_shader *shader)
{
   if (!shader || (!g17p_explicit_lod_stage(shader->info.stage) &&
                   shader->info.stage != MESA_SHADER_FRAGMENT &&
                   shader->info.stage != MESA_SHADER_COMPUTE))
      return false;

   unsigned samples = 0, image_reads = 0;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic) {
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic == nir_intrinsic_bindless_image_load ||
                   intr->intrinsic ==
                      nir_intrinsic_bindless_image_sparse_load) {
                  if (!g17p_indexed_image_fetch_shape(intr))
                     return g17p_decline("image_fetch_shape", NULL, intr,
                                         shader->info.stage);
                  image_reads++;
               }
               continue;
            }
            if (instr->type != nir_instr_type_tex)
               continue;
            nir_tex_instr *tex = nir_instr_as_tex(instr);
            const bool fetch = g17p_indexed_fetch_shape(tex);
            /* txb and txd reach the packer as the same sample instruction
             * with a different LOD mode; the heap-indexed descriptor operand
             * is orthogonal to it. */
            /* Dynamic argument-buffer vertex witnesses cover explicit level
             * and gradient samples. Vertex implicit sampling has no fragment
             * quad derivatives; bias likewise remains fragment-only. */
            if (!fetch && g17p_explicit_lod_stage(shader->info.stage) &&
                tex->op != nir_texop_txl && tex->op != nir_texop_txd)
               return g17p_decline("vertex_implicit_sample", tex, NULL,
                                   shader->info.stage);
            if (!fetch && !g17p_sample_shape(tex))
               return g17p_decline("sample_shape", tex, NULL,
                                   shader->info.stage);
            nir_intrinsic_instr *image = nir_src_as_intrinsic(
               tex->src[nir_tex_instr_src_index(tex, nir_tex_src_texture_handle)].src);
            if (!image || image->intrinsic != nir_intrinsic_bindless_image_agx)
               return g17p_decline("handle_not_bindless", tex, NULL,
                                   shader->info.stage);
            if (!fetch && !g17p_sample_heap_index(tex))
               return g17p_decline("no_sampler_heap_index", tex, NULL,
                                   shader->info.stage);
            samples++;
         }
      }
   }
   if (!samples && !image_reads) {
      /* A shader with no texture instruction at all is the ordinary case and
       * says nothing; reporting it once per compile buries the shaders that
       * did have something to lower and did not get it. Report only when the
       * shader has texture instructions this pass declined to count. */
      if (agx_compiler_debug & (AGX_DBG_SHADERS | AGX_DBG_REFUSE)) {
         unsigned tex = 0;
         nir_foreach_function_impl(impl, shader) {
            nir_foreach_block(block, impl) {
               nir_foreach_instr(instr, block)
                  tex += instr->type == nir_instr_type_tex;
            }
         }
         if (tex) {
            fprintf(stderr,
                    "G17P indexed samplers: %u texture instructions in stage "
                    "%u, none admitted\n",
                    tex, (unsigned)shader->info.stage);
         }
      }
      return false;
   }

   nir_foreach_function_impl(impl, shader) {
      nir_builder b = nir_builder_create(impl);
      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type == nir_instr_type_intrinsic) {
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic != nir_intrinsic_bindless_image_load &&
                   intr->intrinsic !=
                      nir_intrinsic_bindless_image_sparse_load)
                  continue;

               b.cursor = nir_before_instr(instr);
               nir_intrinsic_instr *image =
                  nir_src_as_intrinsic(intr->src[0]);
               assert(image &&
                      image->intrinsic == nir_intrinsic_bindless_image_agx);
               nir_def *set = nir_load_descriptor_set_agx(
                  &b, nir_intrinsic_desc_set(image));
               nir_def *offset = nir_iadd_imm(
                  &b, image->src[0].ssa,
                  AGX_G17P_STORAGE_HEAP_INDEX_OFFSET);
               nir_def *address =
                  nir_iadd(&b, set, nir_u2u64(&b, offset));
               nir_def *texture_index = nir_load_global_constant(
                  &b, 1, 32, address, .align_mul = 4);
               texture_index = nir_u2u64(&b, texture_index);
               nir_def *native_handle = nir_bindless_image_agx(
                  &b, texture_index, .desc_set = 0);
               nir_src_rewrite(&intr->src[0], native_handle);
               continue;
            }
            if (instr->type != nir_instr_type_tex)
               continue;
            nir_tex_instr *tex = nir_instr_as_tex(instr);
            b.cursor = nir_before_instr(instr);
            nir_intrinsic_instr *image = nir_src_as_intrinsic(
               tex->src[nir_tex_instr_src_index(tex, nir_tex_src_texture_handle)].src);
            const bool fetch = tex->op == nir_texop_txf ||
                               tex->op == nir_texop_txf_ms;
            nir_def *sampler = fetch ? NULL : g17p_sample_heap_index(tex);

            /* texture2d::read takes a uint level on AGX3.  The shared AGX2
             * texture lowering narrows every NIR LOD source to 16 bits, but
             * G17P's integer PACK_SMPIR form consumes a full 32-bit scalar.
             * Leaving the upper half unwritten makes the native sampler see
             * an indeterminate, usually out-of-range level and return zero.
             * Restore the API integer width before AIR is formed. */
            if (tex->op == nir_texop_txf) {
               const int lod =
                  nir_tex_instr_src_index(tex, nir_tex_src_lod);
               assert(lod >= 0);
               nir_def *level = tex->src[lod].src.ssa;
               nir_src_rewrite(&tex->src[lod].src,
                               nir_src_is_const(tex->src[lod].src)
                                  ? nir_imm_intN_t(
                                       &b, nir_src_as_uint(tex->src[lod].src), 32)
                                  : nir_u2u32(&b, level));
            }

            nir_def *set = nir_load_descriptor_set_agx(&b, nir_intrinsic_desc_set(image));
            /* Descriptor offsets are 32-bit and valid sets fit far below
             * 4GiB. Add the field offset before widening so native LDDEV can
             * consume base+byte-index without a separate 64-bit base add. */
            nir_def *offset = nir_iadd_imm(&b, image->src[0].ssa,
                                          AGX_G17P_SAMPLED_HEAP_INDEX_OFFSET);
            nir_def *address = nir_iadd(&b, set, nir_u2u64(&b, offset));
            nir_def *texture_index = nir_load_global_constant(&b, 1, 32, address, .align_mul = 4);
            /* The software sampler index includes AGX2's reservation of 28.
             * Native G17P heaps are zero-based, as the immediate path is.
             *
             * Rebase by saturating, not wrapping. The descriptor ABI writes a
             * null sampler as the unbiased base itself and leaves an untouched
             * descriptor at all-zero, so indices below the bias do reach here.
             * hk_agx3_g17p_combined_sampler_pointers() resolves the identical
             * heap on the immediate path as
             *   index < HK_AGX3_G17P_BINDLESS_SAMPLER_BASE_INDEX ? 0
             *                                                    : index - base
             * and native entry zero is the reserved robust sampler both
             * encodings must land on. Wrapping instead sends 0 to 0xffffffe4,
             * a register sampler descriptor far outside the heap.
             *
             * G17P has no saturating-subtract encoding; agx_compile_shader_nir
             * sets lower_usub_sat for AGX3B, so this becomes the witnessed
             * compare/subtract/select before isel. */
            texture_index = nir_u2u64(&b, texture_index);
            nir_steal_tex_src(tex, nir_tex_src_texture_handle);
            nir_tex_instr_add_src(tex, nir_tex_src_texture_offset, texture_index);
            if (!fetch) {
               nir_def *sampler_index = nir_usub_sat(
                  &b, nir_u2u32(&b, sampler),
                  nir_imm_int(&b, AGX_G17P_BINDLESS_SAMPLER_BASE_INDEX));
               if (agx3_g17p_forced_sampler_index() >= 0)
                  sampler_index =
                     nir_imm_int(&b, agx3_g17p_forced_sampler_index());
               sampler_index = nir_u2u64(&b, sampler_index);
               nir_steal_tex_src(tex, nir_tex_src_sampler_handle);
               nir_tex_instr_add_src(tex, nir_tex_src_sampler_handle, sampler_index);
            }
            /* Fetch uses native sampler0, reserved in the same device heap
             * that ordinary samples use, so mixed reads share one prefix. */
         }
      }
      nir_progress(true, impl, nir_metadata_control_flow);
   }
   NIR_PASS(_, shader, nir_opt_dce);
   return true;
}

static bool
g17p_cube_sample_shape(const nir_tex_instr *tex)
{
   if (tex->op != nir_texop_tex ||
       tex->sampler_dim != GLSL_SAMPLER_DIM_CUBE || tex->is_array ||
       tex->is_shadow || tex->is_sparse || tex->coord_components != 3 ||
       tex->num_srcs != 3 ||
       (tex->backend_flags & ~AGX_TEXTURE_FLAG_CLAMP_TO_0) != 0 ||
       tex->dest_type != nir_type_float32 || tex->def.num_components != 4 ||
       tex->def.bit_size != 32 ||
       nir_tex_instr_src_index(tex, nir_tex_src_texture_handle) < 0 ||
       nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle) < 0)
      return false;

   int coord = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   if (coord < 0)
      coord = nir_tex_instr_src_index(tex, nir_tex_src_backend1);

   return coord >= 0 && nir_src_num_components(tex->src[coord].src) == 3 &&
          nir_src_bit_size(tex->src[coord].src) == 32;
}

/* Lower the exact implicit, non-array cube sample to the already-qualified
 * G17P 2D-array path. Cube images are six consecutive 2D layers; select the
 * major axis and map its face coordinate to [0, 1] before the ordinary array
 * layer marshalling pass. The face ordering and transforms are the Vulkan/GL
 * cube convention (the same mapping used by Mesa's software samplers). */
bool
agx_nir_lower_g17p_cube_samples(nir_shader *shader)
{
   if (!shader || shader->info.stage != MESA_SHADER_FRAGMENT)
      return false;

   unsigned cubes = 0;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_tex)
               continue;

            nir_tex_instr *tex = nir_instr_as_tex(instr);
            if (tex->sampler_dim != GLSL_SAMPLER_DIM_CUBE)
               continue;
            if (!g17p_cube_sample_shape(tex))
               return false;
            cubes++;
         }
      }
   }

   if (!cubes)
      return false;

   nir_foreach_function_impl(impl, shader) {
      nir_builder b = nir_builder_create(impl);
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_tex)
               continue;

            nir_tex_instr *tex = nir_instr_as_tex(instr);
            if (tex->sampler_dim != GLSL_SAMPLER_DIM_CUBE)
               continue;

            int coord_index = nir_tex_instr_src_index(tex, nir_tex_src_coord);
            if (coord_index < 0)
               coord_index =
                  nir_tex_instr_src_index(tex, nir_tex_src_backend1);

            nir_def *coord = tex->src[coord_index].src.ssa;
            b.cursor = nir_before_instr(instr);
            nir_def *x = nir_channel(&b, coord, 0);
            nir_def *y = nir_channel(&b, coord, 1);
            nir_def *z = nir_channel(&b, coord, 2);
            nir_def *x2 = nir_fmul(&b, x, x);
            nir_def *y2 = nir_fmul(&b, y, y);
            nir_def *z2 = nir_fmul(&b, z, z);
            nir_def *half = nir_imm_float(&b, 0.5f);

            nir_def *x_positive = nir_fge_imm(&b, x, 0.0f);
            nir_def *y_positive = nir_fge_imm(&b, y, 0.0f);
            nir_def *z_positive = nir_fge_imm(&b, z, 0.0f);

            /* rsqrt(c*c) gives 1/abs(c) without materializing fabs/fneg,
             * whose standalone AGX3 forms are not yet qualified. */
            nir_def *rx = nir_frsq(&b, x2);
            nir_def *ry = nir_frsq(&b, y2);
            nir_def *rz = nir_frsq(&b, z2);
            nir_def *px = nir_fmul_imm(&b, rx, 0.5f);
            nir_def *nx = nir_fmul_imm(&b, rx, -0.5f);
            nir_def *py = nir_fmul_imm(&b, ry, 0.5f);
            nir_def *ny = nir_fmul_imm(&b, ry, -0.5f);
            nir_def *pz = nir_fmul_imm(&b, rz, 0.5f);
            nir_def *nz = nir_fmul_imm(&b, rz, -0.5f);

            nir_def *face_x = nir_vec3(
               &b,
               nir_bcsel(&b, x_positive, nir_ffma(&b, nx, z, half),
                          nir_ffma(&b, px, z, half)),
               nir_ffma(&b, nx, y, half),
               nir_bcsel(&b, x_positive, nir_imm_int(&b, 0),
                          nir_imm_int(&b, 1)));
            nir_def *face_y = nir_vec3(
               &b, nir_ffma(&b, py, x, half),
               nir_bcsel(&b, y_positive, nir_ffma(&b, py, z, half),
                          nir_ffma(&b, ny, z, half)),
               nir_bcsel(&b, y_positive, nir_imm_int(&b, 2),
                          nir_imm_int(&b, 3)));
            nir_def *face_z = nir_vec3(
               &b,
               nir_bcsel(&b, z_positive, nir_ffma(&b, pz, x, half),
                          nir_ffma(&b, nz, x, half)),
               nir_ffma(&b, nz, y, half),
               nir_bcsel(&b, z_positive, nir_imm_int(&b, 4),
                          nir_imm_int(&b, 5)));

            nir_def *use_x = nir_iand(&b, nir_fge(&b, x2, y2),
                                      nir_fge(&b, x2, z2));
            nir_def *use_y = nir_iand(&b, nir_fge(&b, y2, x2),
                                      nir_fge(&b, y2, z2));
            nir_def *mapped = nir_bcsel(
               &b, use_x, face_x, nir_bcsel(&b, use_y, face_y, face_z));

            nir_src_rewrite(&tex->src[coord_index].src, mapped);
            tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
            tex->is_array = true;
            tex->array_is_lowered_cube = true;
         }
      }

      nir_progress(true, impl, nir_metadata_control_flow);
   }

   NIR_PASS(_, shader, nir_opt_dce);
   return true;
}

static bool
g17p_txf_shape(const nir_tex_instr *tex)
{
   if (tex->op != nir_texop_txf ||
       tex->sampler_dim != GLSL_SAMPLER_DIM_2D || tex->is_shadow ||
       tex->is_sparse || (tex->coord_components != 2 &&
                          tex->coord_components != 3) ||
       tex->is_array != (tex->coord_components == 3) ||
       (tex->backend_flags & ~AGX_TEXTURE_FLAG_NO_CLAMP) != 0 ||
       tex->dest_type != nir_type_uint32 || tex->def.num_components != 4 ||
       tex->def.bit_size != 32 || tex->texture_index != 0 ||
       tex->sampler_index != 0)
      return false;

   const int handle =
      nir_tex_instr_src_index(tex, nir_tex_src_texture_handle);
   const int lod = nir_tex_instr_src_index(tex, nir_tex_src_lod);
   int coord = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   if (coord < 0)
      coord = nir_tex_instr_src_index(tex, nir_tex_src_backend1);

   if (handle < 0 || lod < 0 || coord < 0 || tex->num_srcs != 3 ||
       nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle) >= 0 ||
       nir_tex_instr_src_index(tex, nir_tex_src_sampler_offset) >= 0 ||
       nir_tex_instr_src_index(tex, nir_tex_src_texture_offset) >= 0 ||
       !nir_src_is_const(tex->src[lod].src) ||
       nir_src_as_uint(tex->src[lod].src) != 0)
      return false;

   return nir_src_num_components(tex->src[coord].src) ==
             tex->coord_components &&
          nir_src_bit_size(tex->src[coord].src) == 32;
}

static bool
g17p_has_bindless_texture_access(const nir_shader *shader)
{
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            const nir_intrinsic_op op =
               nir_instr_as_intrinsic(instr)->intrinsic;
            if (op == nir_intrinsic_bindless_image_agx ||
                op == nir_intrinsic_bindless_sampler_agx)
               return true;
         }
      }
   }

   return false;
}

/* A robustness bounds check lowers imageSize() into a descriptor crawl before
 * storage-image specialization runs. The load/store handles are removed by
 * the specialization, but that read-only crawl and image atomics still need
 * their bindless handles for address calculation. Admit only the exact paired
 * texture/PBE descriptors and those two consumers; any independently indexed
 * image or sampler access still fails closed. */
static bool
g17p_has_unsupported_storage_descriptor_access(const nir_shader *shader,
                                                uint32_t texture_offset_B,
                                                uint32_t pbe_offset_B)
{
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_bindless_sampler_agx)
               return true;
            if (intr->intrinsic != nir_intrinsic_bindless_image_agx)
               continue;

            if (nir_intrinsic_desc_set(intr) != 0 ||
                !nir_src_is_const(intr->src[0]))
               return true;

            const uint32_t offset_B = nir_src_as_uint(intr->src[0]);
            if (offset_B != texture_offset_B && offset_B != pbe_offset_B)
               return true;

            nir_foreach_use(use, &intr->def) {
               nir_instr *parent = nir_src_parent_instr(use);
               if (parent->type != nir_instr_type_intrinsic)
                  return true;

               nir_intrinsic_instr *user = nir_instr_as_intrinsic(parent);
               const bool descriptor_address =
                  user->intrinsic ==
                     nir_intrinsic_load_from_texture_handle_agx &&
                  user->def.num_components == 1 && user->def.bit_size == 64;
               const bool image_atomic =
                  offset_B == pbe_offset_B &&
                  (user->intrinsic == nir_intrinsic_bindless_image_atomic ||
                   user->intrinsic ==
                      nir_intrinsic_bindless_image_atomic_swap);
               if (!descriptor_address && !image_atomic)
                  return true;
            }
         }
      }
   }

   return false;
}

bool
agx_nir_lower_g17p_txf(nir_shader *shader,
                        struct agx_g17p_txf_binding *binding)
{
   if (!shader || !binding || shader->info.stage != MESA_SHADER_COMPUTE)
      return false;

   nir_tex_instr *fetch = NULL;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_tex)
               continue;

            if (fetch || !g17p_txf_shape(nir_instr_as_tex(instr)))
               return false;
            fetch = nir_instr_as_tex(instr);
         }
      }
   }

   if (!fetch)
      return false;

   uint32_t texture_offset_B;
   if (!g17p_handle_offset(fetch, nir_tex_src_texture_handle,
                           nir_intrinsic_bindless_image_agx,
                           &texture_offset_B) ||
       (texture_offset_B & 7) != 0)
      return false;

   nir_steal_tex_src(fetch, nir_tex_src_texture_handle);
   fetch->texture_index = 0;

   NIR_PASS(_, shader, nir_opt_dce);
   if (g17p_has_bindless_texture_access(shader))
      return false;

   *binding = (struct agx_g17p_txf_binding){
      .texture_offset_B = texture_offset_B,
   };
   return true;
}

static bool
g17p_storage_handle_binding(const nir_intrinsic_instr *intr, uint8_t *set,
                            uint32_t *offset_B)
{
   nir_intrinsic_instr *handle = nir_src_as_intrinsic(intr->src[0]);
   if (!handle || handle->intrinsic != nir_intrinsic_bindless_image_agx ||
       nir_intrinsic_desc_set(handle) > UINT8_MAX ||
       !nir_src_is_const(handle->src[0]))
      return false;

   *set = nir_intrinsic_desc_set(handle);
   *offset_B = nir_src_as_uint(handle->src[0]);
   return true;
}

static bool
g17p_storage_handle_offset(const nir_intrinsic_instr *intr,
                           uint32_t *offset_B)
{
   uint8_t set;
   return g17p_storage_handle_binding(intr, &set, offset_B) && set == 0;
}

/* Native imgwci accepts a zero-extended descriptor-heap index in a GPR
 * pair. Load it from the bound descriptor set on the GPU so descriptor arrays
 * and updates use their current contents. Sampled and storage descriptors
 * share the native 32-byte heap addressed through u0.
 */
bool
agx_nir_lower_g17p_bindless_storage(nir_shader *shader)
{
   if (shader->info.stage != MESA_SHADER_COMPUTE &&
       shader->info.stage != MESA_SHADER_FRAGMENT &&
       !g17p_explicit_lod_stage(shader->info.stage))
      return false;
   bool progress = false;
   nir_foreach_function_impl(impl, shader) {
      nir_builder b = nir_builder_create(impl);
      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_bindless_image_store)
               continue;
            const bool cube = nir_intrinsic_image_dim(intr) == GLSL_SAMPLER_DIM_CUBE;
            const enum pipe_format format = nir_intrinsic_format(intr);
            nir_intrinsic_instr *handle = nir_src_as_intrinsic(intr->src[0]);
            if (!handle || handle->intrinsic != nir_intrinsic_bindless_image_agx ||
                (nir_intrinsic_image_dim(intr) != GLSL_SAMPLER_DIM_2D &&
                 nir_intrinsic_image_dim(intr) != GLSL_SAMPLER_DIM_3D && !cube) ||
                (nir_intrinsic_image_array(intr) && !cube &&
                 nir_intrinsic_image_dim(intr) != GLSL_SAMPLER_DIM_2D) ||
                (format != PIPE_FORMAT_R10G10B10A2_UINT &&
                 format != PIPE_FORMAT_B10G10R10A2_UINT &&
                 format != PIPE_FORMAT_R8_UINT &&
                 format != PIPE_FORMAT_R8G8_UINT &&
                 format != PIPE_FORMAT_R8G8_UNORM &&
                 format != PIPE_FORMAT_R16_UNORM &&
                 format != PIPE_FORMAT_R32_UINT &&
                 format != PIPE_FORMAT_R32_SINT &&
                 format != PIPE_FORMAT_R32G32_UINT &&
                 format != PIPE_FORMAT_R32G32_SINT &&
                 format != PIPE_FORMAT_R32G32B32A32_SINT &&
                 format != PIPE_FORMAT_R16_SINT &&
                 format != PIPE_FORMAT_R16G16_UINT &&
                 format != PIPE_FORMAT_R16G16_SINT &&
                 format != PIPE_FORMAT_R8G8B8A8_UINT &&
                 format != PIPE_FORMAT_R8_SINT &&
                 format != PIPE_FORMAT_R8G8_SINT &&
                 format != PIPE_FORMAT_R8G8B8A8_SINT &&
                 format != PIPE_FORMAT_R16_UINT &&
                 format != PIPE_FORMAT_R16G16B16A16_SINT &&
                 format != PIPE_FORMAT_R16G16B16A16_UINT &&
                 format != PIPE_FORMAT_R32G32B32A32_UINT &&
                 format != PIPE_FORMAT_R11G11B10_FLOAT &&
                 format != PIPE_FORMAT_R32_FLOAT &&
                 format != PIPE_FORMAT_R32G32_FLOAT &&
                 format != PIPE_FORMAT_R32G32B32A32_FLOAT &&
                 format != PIPE_FORMAT_R16G16_UNORM &&
                 format != PIPE_FORMAT_R16_SNORM &&
                 format != PIPE_FORMAT_R16G16_SNORM &&
                 format != PIPE_FORMAT_R16G16B16A16_UNORM &&
                 format != PIPE_FORMAT_R16G16B16A16_SNORM &&
                 format != PIPE_FORMAT_R8_UNORM &&
                 format != PIPE_FORMAT_R8_SNORM &&
                 format != PIPE_FORMAT_R8G8_SNORM &&
                 format != PIPE_FORMAT_R8G8B8A8_UNORM &&
                 format != PIPE_FORMAT_B8G8R8A8_UNORM &&
                 format != PIPE_FORMAT_R8G8B8A8_SNORM &&
                 format != PIPE_FORMAT_R10G10B10A2_UNORM &&
                 /* Same RGB10A2 store lane layout as R10G10B10A2; the PBE
                  * descriptor swizzles, as for the accepted UINT twin
                  * (dynamic_rendering ... multisample_resolve.
                  * a2r10g10b10_unorm_pack32.max_attachments_8_samples_4). */
                 format != PIPE_FORMAT_B10G10R10A2_UNORM &&
                 format != PIPE_FORMAT_R5G6B5_UNORM &&
                 format != PIPE_FORMAT_B5G6R5_UNORM &&
                 format != PIPE_FORMAT_R16_FLOAT &&
                 format != PIPE_FORMAT_R16G16_FLOAT &&
                 format != PIPE_FORMAT_R16G16B16A16_FLOAT &&
                 !(format == PIPE_FORMAT_NONE &&
                   (nir_intrinsic_src_type(intr) == nir_type_float32 ||
                    nir_intrinsic_src_type(intr) == nir_type_int32 ||
                    nir_intrinsic_src_type(intr) == nir_type_uint32)))) {
               if (agx_compiler_debug & AGX_DBG_SHADERS)
                  fprintf(stderr,
                          "G17P bindless store left legacy: handle=%s dim=%u "
                          "array=%u format=%u src_type=%u\n",
                          handle ? nir_intrinsic_infos[handle->intrinsic].name
                                 : "none",
                          nir_intrinsic_image_dim(intr),
                          nir_intrinsic_image_array(intr), format,
                          nir_intrinsic_src_type(intr));
               continue;
            }
            /* Storage cube coordinates address faces as ordinary array
             * layers, including a non-array cube's six faces. */
            if (cube) {
               nir_intrinsic_set_image_dim(intr, GLSL_SAMPLER_DIM_2D);
               nir_intrinsic_set_image_array(intr, true);
            }
            b.cursor = nir_before_instr(instr);
            nir_def *set = nir_load_descriptor_set_agx(&b, nir_intrinsic_desc_set(handle));
            if (nir_intrinsic_image_array(intr)) {
               /* Native array stores pack the layer into a 16-bit auxiliary
                * word. Reject negative and out-of-range layers before that
                * truncation by routing x outside the maximum image width,
                * as Apple's array-write kernel does. The inline software
                * PBE has layers-1 in bits[25:12] of the word at byte 12. */
               nir_def *pbe_address = nir_iadd(&b, set,
                  nir_u2u64(&b, nir_iadd_imm(&b, handle->src[0].ssa, 12)));
               nir_def *extent = nir_load_global_constant(&b, 1, 32,
                  pbe_address, .align_mul = 4);
               nir_def *last_layer = nir_iand_imm(&b,
                  nir_ushr_imm(&b, extent, 12), 0x3fff);
               nir_def *coords = intr->src[1].ssa;
               nir_def *valid = nir_uge(&b, last_layer, nir_channel(&b, coords, 2));
               nir_def *x = nir_bcsel(&b, valid, nir_channel(&b, coords, 0),
                                      nir_imm_int(&b, 0xffff));
               nir_src_rewrite(&intr->src[1], nir_vector_insert_imm(&b, coords, x, 0));
            }
            /* The image handle points at the inline PBE descriptor. The
             * native heap index follows the complete descriptor pair. */
            nir_def *offset = nir_iadd_imm(&b, handle->src[0].ssa,
               AGX_G17P_STORAGE_HEAP_INDEX_OFFSET - 32);
            nir_def *address = nir_iadd(&b, set, nir_u2u64(&b, offset));
            nir_def *index = nir_load_global_constant(&b, 1, 32, address,
                                                       .align_mul = 4);
            nir_rewrite_image_intrinsic(intr, nir_u2u64(&b, index),
                                        nir_image_intrinsic_type_default);
            nir_intrinsic_set_range_base(intr, 0);
            progress = true;
         }
      }
      nir_progress(progress, impl, nir_metadata_control_flow);
   }
   if (progress)
      NIR_PASS(_, shader, nir_opt_dce);
   return progress;
}

bool
agx_nir_lower_g17p_indexed_storage(
   nir_shader *shader, struct agx_g17p_indexed_storage_binding *binding)
{
   if (!shader || !binding || shader->info.stage != MESA_SHADER_COMPUTE)
      return false;

   nir_intrinsic_instr *store = NULL;
   bool has_buffer_fetch = false;
   uint8_t pbe_set = 0;
   uint32_t pbe_offset_B = 0;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_tex) {
               nir_tex_instr *tex = nir_instr_as_tex(instr);
               /* Buffer textures are lowered to a 2D txf whose coordinate is
                * carried in backend1; ordinary 2D texelFetch keeps coord. */
               has_buffer_fetch |=
                  tex->op == nir_texop_txf &&
                  nir_tex_instr_src_index(tex, nir_tex_src_backend1) >= 0;
               continue;
            }

            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_bindless_image_store) {
               const enum pipe_format format = nir_intrinsic_format(intr);
               const bool r32 = format == PIPE_FORMAT_R32_UINT;
               if (store ||
                   nir_intrinsic_image_dim(intr) != GLSL_SAMPLER_DIM_2D ||
                   nir_intrinsic_image_array(intr) ||
                   (format != PIPE_FORMAT_R8G8B8A8_UNORM &&
                    format != PIPE_FORMAT_R10G10B10A2_UNORM && !r32) ||
                   nir_src_num_components(intr->src[1]) != 4 ||
                   nir_src_bit_size(intr->src[1]) != 32 ||
                   (nir_src_num_components(intr->src[3]) != 4 &&
                    !(r32 && nir_src_num_components(intr->src[3]) == 1)) ||
                   nir_src_bit_size(intr->src[3]) != 32 ||
                   !nir_src_is_const(intr->src[2]) ||
                   nir_src_as_uint(intr->src[2]) != 0 ||
                   !nir_src_is_const(intr->src[4]) ||
                   nir_src_as_uint(intr->src[4]) != 0 ||
                   (nir_intrinsic_access(intr) &
                    (ACCESS_COHERENT | ACCESS_VOLATILE)) ||
                   !g17p_storage_handle_binding(intr, &pbe_set,
                                                 &pbe_offset_B) ||
                   (pbe_offset_B & 7) != 0)
                  return false;
               store = intr;
            } else if (intr->intrinsic == nir_intrinsic_bindless_image_load ||
                       intr->intrinsic ==
                          nir_intrinsic_bindless_image_sparse_load ||
                       intr->intrinsic == nir_intrinsic_bindless_image_atomic ||
                       intr->intrinsic ==
                          nir_intrinsic_bindless_image_atomic_swap ||
                       intr->intrinsic == nir_intrinsic_bindless_image_size ||
                       intr->intrinsic == nir_intrinsic_bindless_image_samples) {
               return false;
            }
         }
      }
   }

   if (!store)
      return false;

   /* The indexed texture path occupies the first native state pair with the
    * texture/sampler heap roots.  A normalized immediate-state write after a
    * lowered texel-buffer fetch currently resolves state 1 as empty on G17P,
    * producing an all-zero destination.  Keep the working bindless PBE form
    * for this mixed shape; integer state 1 and straight-line sampled-storage
    * retain their independently witnessed specialized paths. */
   if (has_buffer_fetch &&
       (nir_intrinsic_format(store) == PIPE_FORMAT_R8G8B8A8_UNORM ||
        nir_intrinsic_format(store) == PIPE_FORMAT_R10G10B10A2_UNORM))
      return false;

   nir_builder b = nir_builder_at(nir_before_instr(&store->instr));
   nir_src_rewrite(&store->src[0], nir_imm_intN_t(&b, 1, 16));
   store->intrinsic = nir_intrinsic_image_store;
   nir_progress(true, nir_shader_get_entrypoint(shader),
                nir_metadata_control_flow);
   NIR_PASS(_, shader, nir_opt_dce);

   *binding = (struct agx_g17p_indexed_storage_binding){
      .pbe_offset_B = pbe_offset_B,
      .pbe_set = pbe_set,
   };
   return true;
}

static bool
g17p_storage_image_shape(const nir_intrinsic_instr *intr, bool load)
{
   const enum pipe_format format = nir_intrinsic_format(intr);
   if (intr->intrinsic != (load ? nir_intrinsic_bindless_image_load
                                : nir_intrinsic_bindless_image_store) ||
       (nir_intrinsic_access(intr) & ACCESS_BUFFER_AGX) ||
       nir_intrinsic_image_dim(intr) != GLSL_SAMPLER_DIM_2D ||
       nir_intrinsic_image_array(intr) ||
       (format != PIPE_FORMAT_R8G8B8A8_UINT &&
        format != PIPE_FORMAT_R32_UINT) ||
       nir_src_num_components(intr->src[1]) != 4 ||
       nir_src_bit_size(intr->src[1]) != 32 ||
       !nir_src_is_const(intr->src[2]) ||
       nir_src_as_uint(intr->src[2]) != 0)
      return false;

   if (load) {
      return intr->def.num_components == 4 && intr->def.bit_size == 32 &&
             !nir_intrinsic_image_array(intr) &&
             nir_src_is_const(intr->src[3]) &&
             nir_src_as_uint(intr->src[3]) == 0 &&
             !(nir_intrinsic_access(intr) &
               (ACCESS_COHERENT | ACCESS_VOLATILE));
   }

   return nir_src_num_components(intr->src[3]) == 4 &&
          nir_src_bit_size(intr->src[3]) == 32 &&
          nir_src_is_const(intr->src[4]) &&
          nir_src_as_uint(intr->src[4]) == 0 &&
          !(nir_intrinsic_access(intr) & ACCESS_VOLATILE);
}

static bool
g17p_storage_atomic_shape(const nir_intrinsic_instr *intr,
                          uint32_t pbe_offset_B)
{
   const bool atomic =
      intr->intrinsic == nir_intrinsic_bindless_image_atomic ||
      intr->intrinsic == nir_intrinsic_bindless_image_atomic_swap;
   const enum pipe_format format = nir_intrinsic_format(intr);

   uint32_t atomic_pbe_offset_B;
   return atomic && nir_intrinsic_image_dim(intr) == GLSL_SAMPLER_DIM_2D &&
          !nir_intrinsic_image_array(intr) &&
          (format == PIPE_FORMAT_R32_UINT ||
           format == PIPE_FORMAT_R32_SINT) &&
          nir_src_num_components(intr->src[1]) == 4 &&
          nir_src_bit_size(intr->src[1]) == 32 &&
          nir_src_is_const(intr->src[2]) &&
          nir_src_as_uint(intr->src[2]) == 0 &&
          !(nir_intrinsic_access(intr) & ACCESS_VOLATILE) &&
          g17p_storage_handle_offset(intr, &atomic_pbe_offset_B) &&
          atomic_pbe_offset_B == pbe_offset_B;
}

bool
agx_nir_lower_g17p_storage_image(
   nir_shader *shader, struct agx_g17p_storage_image_binding *binding)
{
   if (!shader || !binding || shader->info.stage != MESA_SHADER_COMPUTE)
      return false;

   nir_intrinsic_instr *load = NULL, *store = NULL;
   unsigned atomics = 0;
   uint32_t texture_offset_B = 0, pbe_offset_B = 0;
   unsigned impl_count = 0;
   nir_foreach_function_impl(impl, shader) {
      impl_count++;
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_tex)
               return false;
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            /* The single-image state-0 route predates the general indexed
             * storage ABI. It is correct for an isolated image roundtrip, but
             * a shader memory barrier does not order its native state-0 image
             * accesses at all -- the pairing of the barrier with buffer or
             * global memory was never what mattered.
             *
             * Measured on G17P with dEQP-VK.memory_model.write_after_read:
             * the image + buffer-guard variants pass only because the check
             * below already routed them to the indexed lowering, and forcing
             * those same 80 passing cases back onto the state-0 route fails
             * all 80, deterministically, with nothing else changed. The
             * image + workgroup-guard variants name only image and shared
             * memory, so they stayed on the state-0 route and fail.
             *
             * The barrier instructions the two routes emit are identical, so
             * WHY the native state-0 pair is unordered is not understood.
             * Only that it is. Decline any barrier that names the image and
             * leave ordered images to the indexed lowering. */
            if (intr->intrinsic == nir_intrinsic_barrier) {
               const nir_variable_mode modes =
                  nir_intrinsic_memory_modes(intr);
               if (modes & nir_var_image)
                  return false;
            }
            if (intr->intrinsic == nir_intrinsic_bindless_image_load) {
               if (load || !g17p_storage_image_shape(intr, true) ||
                   !g17p_storage_handle_offset(intr, &texture_offset_B))
                  return false;
               load = intr;
            } else if (intr->intrinsic == nir_intrinsic_bindless_image_store) {
               if (store || !g17p_storage_image_shape(intr, false) ||
                   !g17p_storage_handle_offset(intr, &pbe_offset_B))
                  return false;
               store = intr;
            } else if (intr->intrinsic ==
                          nir_intrinsic_bindless_image_atomic ||
                       intr->intrinsic ==
                          nir_intrinsic_bindless_image_atomic_swap) {
               uint32_t atomic_pbe_offset_B;
               if (!g17p_storage_handle_offset(intr,
                                                &atomic_pbe_offset_B))
                  return false;
               if (atomics++ == 0)
                  pbe_offset_B = atomic_pbe_offset_B;
               else if (pbe_offset_B != atomic_pbe_offset_B)
                  return false;
            } else if (intr->intrinsic == nir_intrinsic_bindless_image_sparse_load ||
                       intr->intrinsic == nir_intrinsic_bindless_image_size ||
                       intr->intrinsic == nir_intrinsic_bindless_image_samples) {
               return false;
            }
         }
      }
   }

   /* A Gen9 storage descriptor is texture32 followed by PBE32. A shader with
    * only PBE stores still needs the same native state-0 pair; derive the
    * unused texture address from the PBE handle instead of leaving imgwci
    * state0 unbound. Atomics keep the PBE handle live because they lower to a
    * raw texel address, while the paired imageStore uses native state 0. */
   if (impl_count != 1 || !store || (pbe_offset_B & 7) != 0)
      return false;

   uint32_t store_pbe_offset_B;
   if (!g17p_storage_handle_offset(store, &store_pbe_offset_B) ||
       store_pbe_offset_B != pbe_offset_B)
      return false;

   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if ((intr->intrinsic == nir_intrinsic_bindless_image_atomic ||
                 intr->intrinsic ==
                    nir_intrinsic_bindless_image_atomic_swap) &&
                !g17p_storage_atomic_shape(intr, pbe_offset_B))
               return false;
         }
      }
   }

   if (load) {
      if ((texture_offset_B & 7) != 0 ||
          pbe_offset_B != texture_offset_B + 32)
         return false;
   } else {
      if (pbe_offset_B < 32)
         return false;
      texture_offset_B = pbe_offset_B - 32;
   }

   nir_foreach_function_impl(impl, shader) {
      nir_builder b = nir_builder_create(impl);
      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if ((!load || instr != &load->instr) && instr != &store->instr)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            b.cursor = nir_before_instr(instr);
            nir_src_rewrite(&intr->src[0],
                            nir_imm_intN_t(&b, 0, 16));
            intr->intrinsic = load && intr == load
                                 ? nir_intrinsic_image_load
                                 : nir_intrinsic_image_store;
         }
      }
      nir_progress(true, impl, nir_metadata_control_flow);
   }

   NIR_PASS(_, shader, nir_opt_dce);
   if (g17p_has_unsupported_storage_descriptor_access(
          shader, texture_offset_B, pbe_offset_B))
      return false;

   *binding = (struct agx_g17p_storage_image_binding){
      .texture_offset_B = texture_offset_B,
      .pbe_offset_B = pbe_offset_B,
      .write_only = load == NULL && atomics == 0,
   };
   return true;
}

static bool
g17p_sampled_storage_store_shape(const nir_intrinsic_instr *intr)
{
   return intr->intrinsic == nir_intrinsic_bindless_image_store &&
          nir_intrinsic_image_dim(intr) == GLSL_SAMPLER_DIM_2D &&
          !nir_intrinsic_image_array(intr) &&
          nir_intrinsic_format(intr) == PIPE_FORMAT_R8G8B8A8_UNORM &&
          nir_src_num_components(intr->src[1]) == 4 &&
          nir_src_bit_size(intr->src[1]) == 32 &&
          nir_src_num_components(intr->src[3]) == 4 &&
          nir_src_bit_size(intr->src[3]) == 32 &&
          nir_src_is_const(intr->src[2]) &&
          nir_src_as_uint(intr->src[2]) == 0 &&
          nir_src_is_const(intr->src[4]) &&
          nir_src_as_uint(intr->src[4]) == 0 &&
          !(nir_intrinsic_access(intr) &
            (ACCESS_COHERENT | ACCESS_VOLATILE));
}

bool
agx_nir_lower_g17p_sampled_storage(
   nir_shader *shader, struct agx_g17p_sampled_storage_binding *binding)
{
   if (!shader || !binding || shader->info.stage != MESA_SHADER_COMPUTE)
      return false;

   nir_tex_instr *sample = NULL;
   nir_intrinsic_instr *store = NULL;
   unsigned impl_count = 0, block_count = 0;
   nir_foreach_function_impl(impl, shader) {
      impl_count++;
      nir_foreach_block(block, impl) {
         block_count++;
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_tex) {
               if (sample || !g17p_sample_shape(nir_instr_as_tex(instr)))
                  return false;
               sample = nir_instr_as_tex(instr);
               continue;
            }

            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_bindless_image_store) {
               if (store || !g17p_sampled_storage_store_shape(intr))
                  return false;
               store = intr;
            } else if (intr->intrinsic == nir_intrinsic_bindless_image_load ||
                       intr->intrinsic == nir_intrinsic_bindless_image_sparse_load ||
                       intr->intrinsic == nir_intrinsic_bindless_image_atomic ||
                       intr->intrinsic == nir_intrinsic_bindless_image_atomic_swap ||
                       intr->intrinsic == nir_intrinsic_bindless_image_size ||
                       intr->intrinsic == nir_intrinsic_bindless_image_samples) {
               return false;
            }
         }
      }
   }

   /* This protocol is derived from Apple's straight-line kernel with one
    * explicit level-zero sample feeding one rgba8 write.  Keep all control
    * flow and independent data transforms outside the qualified shape. */
   if (impl_count != 1 || block_count != 1 || !sample || !store ||
       store->src[3].ssa != &sample->def || sample->op != nir_texop_txl)
      return false;

   const int lod = nir_tex_instr_src_index(sample, nir_tex_src_lod);
   if (lod < 0 || !nir_src_is_const(sample->src[lod].src) ||
       nir_src_as_uint(sample->src[lod].src) != 0)
      return false;

   uint8_t texture_set, sampler_set, pbe_set;
   uint32_t texture_offset_B, sampler_state_offset_B, storage_offset_B;
   if (!g17p_handle_binding(sample, nir_tex_src_texture_handle,
                            nir_intrinsic_bindless_image_agx, &texture_set,
                            &texture_offset_B) ||
       !g17p_handle_binding(sample, nir_tex_src_sampler_handle,
                            nir_intrinsic_bindless_sampler_agx, &sampler_set,
                            &sampler_state_offset_B) ||
       !g17p_storage_handle_binding(store, &pbe_set, &storage_offset_B) ||
       texture_set != sampler_set || (texture_offset_B & 7) != 0 ||
       sampler_state_offset_B != texture_offset_B + 32 ||
       !g17p_is_sampler_bias(sample, texture_set,
                             sampler_state_offset_B + 12) ||
       (storage_offset_B & 7) != 0)
      return false;

   nir_steal_tex_src(sample, nir_tex_src_texture_handle);
   nir_steal_tex_src(sample, nir_tex_src_sampler_handle);
   if (sample->op == nir_texop_txb) {
      nir_steal_tex_src(sample, nir_tex_src_bias);
      sample->op = nir_texop_tex;
   }
   sample->texture_index = 0;
   sample->sampler_index = 0;

   nir_builder b = nir_builder_create(nir_shader_get_entrypoint(shader));
   b.cursor = nir_before_instr(&store->instr);
   nir_src_rewrite(&store->src[0], nir_imm_intN_t(&b, 1, 16));
   store->intrinsic = nir_intrinsic_image_store;

   NIR_PASS(_, shader, nir_opt_dce);
   if (g17p_has_bindless_texture_access(shader))
      return false;

   *binding = (struct agx_g17p_sampled_storage_binding){
      .texture_offset_B = texture_offset_B,
      .sampler_index_offset_B = sampler_state_offset_B + 8,
      .pbe_offset_B = storage_offset_B,
      .texture_set = texture_set,
      .pbe_set = pbe_set,
   };
   return true;
}
