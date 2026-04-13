/*
 * Copyright (C) 2026 Collabora, Ltd.
 * SPDX-License-Identifier: MIT
 */

#include "pan_nir.h"
#include "bifrost/valhall/valhall.h"
#include "panfrost/model/pan_model.h"

struct tex_srcs {
   nir_def *tex_h;
   nir_def *samp_h;
   nir_def *coord;
   nir_def *ms_idx;
   nir_def *offset;
   nir_def *lod;
   nir_def *bias;
   nir_def *min_lod;
   nir_def *ddx;
   nir_def *ddy;
   nir_def *z_cmpr;
};

static struct tex_srcs
steal_tex_srcs(nir_builder *b, nir_tex_instr *tex)
{
   struct tex_srcs srcs = { NULL, };
   for (unsigned i = 0; i < tex->num_srcs; i++) {
      nir_def *def = tex->src[i].src.ssa;
      switch (tex->src[i].src_type) {
      case nir_tex_src_texture_handle: srcs.tex_h = def;    break;
      case nir_tex_src_sampler_handle: srcs.samp_h = def;   break;
      case nir_tex_src_texture_offset: srcs.tex_h = def;    break;
      case nir_tex_src_sampler_offset: srcs.samp_h = def;   break;
      case nir_tex_src_coord:          srcs.coord = def;    break;
      case nir_tex_src_ms_index:       srcs.ms_idx = def;   break;
      case nir_tex_src_comparator:     srcs.z_cmpr = def;   break;
      case nir_tex_src_offset:         srcs.offset = def;   break;
      case nir_tex_src_lod:            srcs.lod = def;      break;
      case nir_tex_src_bias:           srcs.bias = def;     break;
      case nir_tex_src_min_lod:        srcs.min_lod = def;  break;
      case nir_tex_src_ddx:            srcs.ddx = def;      break;
      case nir_tex_src_ddy:            srcs.ddy = def;      break;
      default:
         UNREACHABLE("Unsupported texture source");
      }
      /* Remove sources as we walk them.  We'll add them back later */
      nir_instr_clear_src(&tex->instr, &tex->src[i].src);
   }
   tex->num_srcs = 0;

   /* If we don't have a texture or sampler handle, grab it from the
    * immediate texture/sampler_index.
    */
   if (!srcs.tex_h)
      srcs.tex_h = nir_imm_int(b, tex->texture_index);
   if (!srcs.samp_h)
      srcs.samp_h = nir_imm_int(b, tex->sampler_index);

   return srcs;
}

static nir_def *
build_cube_coord2_face(nir_builder *b, nir_def *coord)
{
   nir_def *x = nir_channel(b, coord, 0);
   nir_def *y = nir_channel(b, coord, 1);
   nir_def *z = nir_channel(b, coord, 2);

   nir_def *cf = nir_cubeface_pan(b, x, y, z);
   nir_def *max_xyz = nir_channel(b, cf, 0);
   nir_def *face = nir_channel(b, cf, 1);

   nir_def *s = nir_cube_ssel_pan(b, z, x, face);
   nir_def *t = nir_cube_tsel_pan(b, y, z, face);

   /* The OpenGL ES specification requires us to transform an input vector
    * (x, y, z) to the coordinate, given the selected S/T:
    *
    * (1/2 ((s / max{x,y,z}) + 1), 1/2 ((t / max{x, y, z}) + 1))
    *
    * We implement (s shown, t similar) in a form friendlier to FMA
    * instructions, and clamp coordinates at the end for correct
    * NaN/infinity handling:
    *
    * fsat(s * (0.5 / max{x, y, z}) + 0.5)
    */

   /* Calculate 0.5 / max{x, y, z} */
   nir_def *fma1 = nir_fdiv(b, nir_imm_float(b, 0.5), max_xyz);

   s = nir_fsat(b, nir_ffma(b, s, fma1, nir_imm_float(b, 0.5)));
   t = nir_fsat(b, nir_ffma(b, t, fma1, nir_imm_float(b, 0.5)));

   return nir_vec3(b, s, t, face);
}

/* Emits a cube map descriptor, returning a vec2.  The packing of the face
 * with the S coordinate exploits the redundancy of floating points with the
 * range restriction of CUBEFACE output.
 *
 *     struct cube_map_descriptor {
 *         float s : 29;
 *         unsigned face : 3;
 *         float t : 32;
 *     }
 *
 * Since the cube face index is preshifted, this is easy to pack with a
 * bitwise MUX.i32 and a fixed mask, selecting the lower bits 29 from s and
 * the upper 3 bits from face.
 */
static nir_def *
build_cube_desc(nir_builder *b, nir_def *coord)
{
   nir_def *coord2_face = build_cube_coord2_face(b, coord);
   nir_def *s = nir_channel(b, coord2_face, 0);
   nir_def *t = nir_channel(b, coord2_face, 1);
   nir_def *face = nir_channel(b, coord2_face, 2);

   s = nir_bitfield_select(b, nir_imm_int(b, BITFIELD_MASK(29)), s, face);

   return nir_vec2(b, s, t);
}

/* TEXC's explicit and bias LOD modes requires the LOD to be transformed to a
 * 16-bit 8:8 fixed-point format. We lower as:
 *
 * F32_TO_S32(clamp(x, -16.0, +16.0) * 256.0) & 0xFFFF =
 * MKVEC(F32_TO_S32(clamp(x * 1.0/16.0, -1.0, 1.0) * (16.0 * 256.0)), #0)
 */
static nir_def *
build_sfixed_8_8(nir_builder *b, nir_def *x)
{
   nir_def *x_div16_sat = nir_fsat_signed(b, nir_fmul_imm(b, x, 1.0 / 16.0));
   return nir_f2i16(b, nir_fmul_imm(b, x_div16_sat, 16.0 * 256.0));
}

static nir_def *
build_lod_bias_clamp(nir_builder *b, nir_def *bias, nir_def *min)
{
   bias = bias ? build_sfixed_8_8(b, bias) : nir_imm_intN_t(b, 0, 16);
   min = min ? build_sfixed_8_8(b, min) : nir_imm_intN_t(b, 0, 16);
   return nir_pack_32_2x16(b, nir_vec2(b, bias, min));
}

static bool
scalar_is_imm_i4(nir_scalar s, bool is_signed)
{
   if (!nir_scalar_is_const(s))
      return false;

   if (is_signed) {
      int32_t i = nir_scalar_as_int(s);
      return -8 <= i && i <= 7;
   } else {
      uint32_t i = nir_scalar_as_uint(s);
      return i <= 15;
   }
}

static uint32_t
scalar_as_imm_i4(nir_scalar s)
{
   return nir_scalar_as_uint(s) & 0xf;
}

#define PAN_AS_U32(x) ({\
   static_assert(sizeof(x) == 4, "x must be 4 bytes"); \
   uint32_t _u; \
   memcpy(&_u, &(x), 4); \
   _u; \
})

static nir_def *
va_tex_handle(nir_builder *b, nir_def *tex_h, nir_def *samp_h)
{
   if (!nir_def_is_const(tex_h) || !nir_def_is_const(samp_h))
      return nir_vec2(b, samp_h, tex_h);

   uint32_t imm_tex_h = nir_scalar_as_uint(nir_get_scalar(tex_h, 0));
   uint32_t tex_table = pan_res_handle_get_table(imm_tex_h);
   uint32_t tex_index = pan_res_handle_get_index(imm_tex_h);

   uint32_t imm_samp_h = nir_scalar_as_uint(nir_get_scalar(samp_h, 0));
   uint32_t samp_table = pan_res_handle_get_table(imm_samp_h);
   uint32_t samp_index = pan_res_handle_get_index(imm_samp_h);

   if (!va_is_valid_const_table(tex_table) || tex_index >= 1024 ||
       !va_is_valid_const_table(samp_table) || samp_index >= 1024)
      return nir_vec2(b, samp_h, tex_h);

   uint32_t packed_h = (tex_table << 27) | (tex_index << 16) |
                       (samp_table << 11) | samp_index;

   return nir_imm_int(b, packed_h);
}

static nir_def *
build_va_gradient_desc(nir_builder *b, nir_def *tex_h,
                       enum glsl_sampler_dim dim,
                       nir_def *ddx, nir_def *ddy)
{
   struct pan_va_tex_flags flags = {
      .wide_indices = tex_h->num_components > 1,
      .derivative_enable = true,
      .force_delta_enable = false,
      .lod_clamp_disable = true,
      .lod_bias_disable = true,
   };

   nir_def *sr[6] = {};
   unsigned sr_count = 0;

   assert(ddx->num_components == ddy->num_components);
   for (unsigned i = 0; i < ddx->num_components; i++) {
      sr[sr_count++] = nir_channel(b, ddx, i);
      sr[sr_count++] = nir_channel(b, ddy, i);
   }
   assert(sr_count <= ARRAY_SIZE(sr));

   tex_h = nir_pad_vector_imm_int(b, tex_h, 0, 2);

   nir_def *sr0 = NULL, *sr1 = NULL;
   sr0 = nir_vec(b, sr, MIN2(4, sr_count));
   if (sr_count > 4)
      sr1 = nir_vec(b, sr + 4, sr_count - 4);

   return nir_build_tex(b, nir_texop_gradient_pan,
                        .dim = dim,
                        .dest_type = nir_type_uint32,
                        .backend_flags = PAN_AS_U32(flags),
                        .texture_handle = tex_h,
                        .backend1 = sr0,
                        .backend2 = sr1);
}

/* Staging registers required by texturing in the order they appear (Valhall) */
enum valhall_tex_sreg {
   VA_TEX_SR_COORD_FIRST = 0,
   VA_TEX_SR_COORD_S = VA_TEX_SR_COORD_FIRST,
   VA_TEX_SR_COORD_T,
   VA_TEX_SR_COORD_R,
   VA_TEX_SR_COORD_Q,
   VA_TEX_SR_ARRAY,
   VA_TEX_SR_SHADOW,
   VA_TEX_SR_OFFSET,
   VA_TEX_SR_LOD,
   VA_TEX_SR_GRDESC0,
   VA_TEX_SR_GRDESC1,
   VA_TEX_SR_COUNT,
};

static bool
va_lower_tex(nir_builder *b, nir_tex_instr *tex, uint64_t gpu_id)
{
   b->cursor = nir_before_instr(&tex->instr);
   struct tex_srcs srcs = steal_tex_srcs(b, tex);
   nir_def *tex_h = va_tex_handle(b, srcs.tex_h, srcs.samp_h);

   struct pan_va_tex_flags flags = {
      .wide_indices = tex_h->num_components > 1,
   };
   uint32_t narrow = 0;

   nir_def *sr[VA_TEX_SR_COUNT] = {};

   /* TEX_FETCH doesn't have CUBE support. This is not a problem as a cube is
    * just a 2D array in any cases.
    */
   if (tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE && tex->op == nir_texop_txf) {
      tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
      tex->is_array = true;
   }

   const unsigned coord_comps = tex->coord_components - tex->is_array;
   if (tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE) {
      assert(coord_comps == 3);
      nir_def *desc = build_cube_desc(b, srcs.coord);
      sr[VA_TEX_SR_COORD_S] = nir_channel(b, desc, 0);
      sr[VA_TEX_SR_COORD_T] = nir_channel(b, desc, 1);
   } else {
      for (unsigned i = 0; i < coord_comps; i++)
         sr[VA_TEX_SR_COORD_S + i] = nir_channel(b, srcs.coord, i);
   }

   if (tex->is_array) {
      nir_scalar arr_idx = nir_get_scalar(srcs.coord, coord_comps);
      arr_idx = nir_scalar_chase_movs(arr_idx);
      /* On v11+, narrow_array_index is a U4 in bits [15:12]
       *
       * On v9 and v10, narrow_array_index is a U16 in bits [31:16].  However,
       * it does not appear to bounds-check correctly so we can't use it.
       */
      if (pan_arch(gpu_id) >= 11 && scalar_is_imm_i4(arr_idx, false)) {
         narrow |= scalar_as_imm_i4(arr_idx) << 12;
      } else {
         sr[VA_TEX_SR_ARRAY] = nir_mov_scalar(b, arr_idx);
         flags.array_enable = true;
      }
   }

   if (srcs.z_cmpr) {
      sr[VA_TEX_SR_SHADOW] = srcs.z_cmpr;
      flags.compare_enable = true;
   }

   /* On v9 and v10, narrow_lod is a U4 in bits [15:12] and is not affected
    * by texel_offset
    */
   if (pan_arch(gpu_id) < 11 && !flags.wide_indices &&
       tex->op == nir_texop_txf && srcs.lod &&
       nir_scalar_is_const(nir_get_scalar(srcs.lod, 0))) {
      uint32_t imm_lod = nir_scalar_as_uint(nir_get_scalar(srcs.lod, 0));
      narrow |= MIN2(imm_lod, 15) << 12;
      srcs.lod = NULL;
   }

   if (srcs.offset || srcs.ms_idx || tex->op == nir_texop_txf) {
      /* The hardware specifies the offset, MS index, and lod (for TXF) in a
       * u8vec4 <off_s, off_t, off_r_or_ms_idx, txf_lod>.
       */
      nir_scalar comps[4] = { };
      if (srcs.offset) {
         assert(srcs.offset->num_components == coord_comps);
         for (unsigned i = 0; i < coord_comps; i++)
            comps[i] = nir_get_scalar(srcs.offset, i);
      }

      /* The MS index goes in .z */
      if (srcs.ms_idx) {
         assert(coord_comps == 2);
         comps[2] = nir_get_scalar(srcs.ms_idx, 0);
      }

      uint32_t narrow_offset = 0;
      bool is_narrow = true;
      for (unsigned i = 0; i < ARRAY_SIZE(comps); i++) {
         if (comps[i].def) {
            comps[i] = nir_scalar_chase_movs(comps[i]);

            if (scalar_is_imm_i4(comps[i], true)) {
               narrow_offset |= scalar_as_imm_i4(comps[i]) << (i * 4);
            } else {
               is_narrow = false;
               break;
            }
         }
      }

      if (tex->op == nir_texop_txf && srcs.lod) {
         comps[3] = nir_get_scalar(srcs.lod, 0);
         if (pan_arch(gpu_id) >= 11 && nir_scalar_is_const(comps[3])) {
            /* On v11+, narrow_array_index is a 8.8 fixed-point value in
             * bits [31:16]
             */
            uint32_t imm_lod = nir_scalar_as_uint(comps[3]);
            narrow_offset |= MIN2(imm_lod, UINT8_MAX) << 24;
         } else {
            /* Clamp the LOD so it doesn't wrap around */
            comps[3].def = nir_umin_imm(b, comps[3].def, UINT8_MAX);
            is_narrow = false;
         }
      }

      if (is_narrow && !flags.wide_indices) {
         narrow |= narrow_offset;
      } else {
         for (unsigned i = 0; i < ARRAY_SIZE(comps); i++) {
            if (!comps[i].def)
               comps[i] = nir_get_scalar(nir_imm_int(b, 0), 0);
         }

         sr[VA_TEX_SR_OFFSET] =
            nir_pack_32_4x8(b, nir_i2i8(b, nir_vec_scalars(b, comps, 4)));
         flags.texel_offset = true;
      }
   }

   if (tex->op != nir_texop_txf) {
      if (srcs.lod) {
         if (nir_scalar_is_zero(nir_get_scalar(srcs.lod, 0))) {
            flags.lod_mode = BI_VA_LOD_MODE_ZERO_LOD;
         } else {
            flags.lod_mode = BI_VA_LOD_MODE_EXPLICIT;
            sr[VA_TEX_SR_LOD] = nir_u2u32(b, build_sfixed_8_8(b, srcs.lod));
         }
      } else if (srcs.bias || srcs.min_lod) {
         flags.lod_mode = BI_VA_LOD_MODE_COMPUTED_BIAS;
         sr[VA_TEX_SR_LOD] = build_lod_bias_clamp(b, srcs.bias, srcs.min_lod);
      } else if (srcs.ddx || srcs.ddy) {
         flags.lod_mode = BI_VA_LOD_MODE_GRDESC;
         nir_def *grdesc = build_va_gradient_desc(b, tex_h, tex->sampler_dim,
                                                  srcs.ddx, srcs.ddy);
         sr[VA_TEX_SR_GRDESC0] = nir_channel(b, grdesc, 0);
         sr[VA_TEX_SR_GRDESC1] = nir_channel(b, grdesc, 1);
      } else {
         flags.lod_mode = BI_VA_LOD_MODE_COMPUTED_LOD;
      }
   }

   /* Now, fill out the lowered instruction */

   tex->backend_flags = PAN_AS_U32(flags);

   /* If !wide_indices, we put the narrow bits in tex_h.hi */
   if (!flags.wide_indices)
      tex_h = nir_vec2(b, tex_h, nir_imm_int(b, narrow));
   nir_tex_instr_add_src(tex, nir_tex_src_texture_handle, tex_h);

   unsigned sr_count = 0;
   for (unsigned i = 0; i < VA_TEX_SR_COUNT; i++) {
      if (sr[i])
         sr[sr_count++] = sr[i];
   }
   assert(sr_count <= 8);

   nir_def *sr0 = nir_vec(b, sr, MIN2(4, sr_count));
   nir_tex_instr_add_src(tex, nir_tex_src_backend1, sr0);
   if (sr_count > 4) {
      nir_def *sr1 = nir_vec(b, sr + 4, sr_count - 4);
      nir_tex_instr_add_src(tex, nir_tex_src_backend2, sr1);
   }

   return true;
}

static bool
va_lower_lod(nir_builder *b, nir_tex_instr *tex, uint64_t gpu_id)
{
   b->cursor = nir_before_instr(&tex->instr);
   struct tex_srcs srcs = steal_tex_srcs(b, tex);
   nir_def *tex_h = va_tex_handle(b, srcs.tex_h, srcs.samp_h);

   struct pan_va_tex_flags flags = {
      .wide_indices = tex_h->num_components > 1,
      .derivative_enable = false,
      .force_delta_enable = true,
   };

   tex_h = nir_pad_vector_imm_int(b, tex_h, 0, 2);

   nir_def *coord = srcs.coord;
   if (tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE)
      coord = build_cube_desc(b, coord);

   nir_def *comps[2];
   for (unsigned i = 0; i < 2; i++) {
      flags.lod_clamp_disable = i != 0;
      nir_def *grdesc = nir_build_tex(b, nir_texop_gradient_pan,
                                      .dim = tex->sampler_dim,
                                      .dest_type = nir_type_int32,
                                      .backend_flags = PAN_AS_U32(flags),
                                      .texture_handle = tex_h,
                                      .backend1 = coord);

      nir_def *lod_i16 = nir_unpack_32_2x16_split_x(b, grdesc);

      assert(tex->dest_type == nir_type_float32);
      nir_def *lod = nir_i2f32(b, lod_i16);

      lod = nir_fdiv_imm(b, lod, 256.0);
      if (i == 0)
         lod = nir_fround_even(b, lod);

      comps[i] = lod;
   }

   nir_def_replace(&tex->def, nir_vec2(b, comps[0], comps[1]));
   return true;
}

static bool
va_lower_tex_instr(nir_builder *b, nir_tex_instr *tex, void *cb_data)
{
   uint64_t gpu_id = *(uint64_t *)cb_data;

   switch (tex->op) {
   case nir_texop_tex:
   case nir_texop_txb:
   case nir_texop_txl:
   case nir_texop_txd:
   case nir_texop_txf:
   case nir_texop_txf_ms:
   case nir_texop_tg4:
      return va_lower_tex(b, tex, gpu_id);

   case nir_texop_lod:
      return va_lower_lod(b, tex, gpu_id);

   default:
      return false;
   }
}

bool
pan_nir_lower_tex(nir_shader *nir, uint64_t gpu_id)
{
   if (pan_arch(gpu_id) >= 9) {
      return nir_shader_tex_pass(nir, va_lower_tex_instr,
                                 nir_metadata_control_flow,
                                 &gpu_id);
   } else {
      UNREACHABLE("Midgard and Bifrost are not supported by this pass");
   }
}
