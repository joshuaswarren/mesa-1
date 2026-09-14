/*
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2024 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

#include "gallium/include/pipe/p_defines.h"
#include "poly/cl/libpoly.h"
#include "poly/nir/poly_nir.h"
#include "util/format/u_formats.h"
#include "agx_abi.h"
#include "agx_linker.h"
#include "agx_nir.h"
#include "agx_nir_lower_vbo.h"
#include "agx_pack.h"
#include "agx_tilebuffer.h"
#include "libagx.h"
#include "nir.h"
#include "nir_builder.h"
#include "nir_builder_opcodes.h"
#include "nir_lower_blend.h"
#include "shader_enums.h"

/* Mesa's "every sample" spelling, shared with agx_nir_lower_tilebuffer. */
#define ALL_SAMPLES 0xFF

/*
 * Insert code into a fragment shader to lower polygon stipple. The stipple is
 * passed in a sideband, rather than requiring a texture binding. This is
 * simpler for drivers to integrate and might be more efficient.
 */
static bool
agx_nir_lower_poly_stipple(nir_shader *s)
{
   assert(s->info.stage == MESA_SHADER_FRAGMENT);

   /* Insert at the beginning for performance. */
   nir_builder b_ =
      nir_builder_at(nir_before_impl(nir_shader_get_entrypoint(s)));
   nir_builder *b = &b_;

   /* The stipple coordinate is defined at the window coordinate mod 32. It's
    * reversed along the X-axis to simplify the driver, hence the NOT.
    */
   nir_def *raw = nir_u2u32(b, nir_load_pixel_coord(b));
   nir_def *coord = nir_umod_imm(
      b,
      nir_vec2(b, nir_inot(b, nir_channel(b, raw, 0)), nir_channel(b, raw, 1)),
      32);

   /* Extract the column from the packed bitfield */
   nir_def *pattern = nir_load_polygon_stipple_agx(b, nir_channel(b, coord, 1));
   nir_def *bit = nir_ubitfield_extract(b, pattern, nir_channel(b, coord, 0),
                                        nir_imm_int(b, 1));

   /* Discard fragments where the pattern is 0 */
   nir_demote_if(b, nir_ieq_imm(b, bit, 0));
   s->info.fs.uses_discard = true;

   return nir_progress(true, b->impl, nir_metadata_control_flow);
}

static int
map_vs_part_uniform(nir_intrinsic_instr *intr, unsigned nr_attribs,
                    unsigned uniform_base)
{
   int uniform;
   switch (intr->intrinsic) {
   case nir_intrinsic_load_vbo_base_agx:
      uniform = AGX_ABI_VUNI_VBO_BASE(nir_src_as_uint(intr->src[0]));
      break;

   case nir_intrinsic_load_attrib_clamp_agx:
      uniform =
         AGX_ABI_VUNI_VBO_CLAMP(nr_attribs, nir_src_as_uint(intr->src[0]));
      break;

   case nir_intrinsic_load_first_vertex:
      uniform = AGX_ABI_VUNI_FIRST_VERTEX(nr_attribs);
      break;

   case nir_intrinsic_load_base_instance:
      uniform = AGX_ABI_VUNI_BASE_INSTANCE(nr_attribs);
      break;

   case nir_intrinsic_load_vertex_param_buffer_poly:
      uniform = AGX_ABI_VUNI_VERTEX_PARAMS(nr_attribs);
      break;

   default:
      return -1;
   }

   return uniform_base + uniform;
}

static int
map_fs_part_uniform(nir_intrinsic_instr *intr,
                    const struct agx_fs_epilog_key *key)
{
   const unsigned blend_base = key && key->blend_constant_base
                                  ? key->blend_constant_base
                                  : AGX_ABI_FUNI_BLEND_R;

   switch (intr->intrinsic) {
   case nir_intrinsic_load_blend_const_color_r_float:
      return blend_base;

   case nir_intrinsic_load_blend_const_color_g_float:
      return blend_base + 2;

   case nir_intrinsic_load_blend_const_color_b_float:
      return blend_base + 4;

   case nir_intrinsic_load_blend_const_color_a_float:
      return blend_base + 6;

   default:
      return -1;
   }
}

static bool
lower_non_monolithic_uniforms(nir_builder *b, nir_intrinsic_instr *intr,
                              void *data)
{
   int unif;
   if (b->shader->info.stage == MESA_SHADER_VERTEX) {
      const struct agx_vs_prolog_key *layout = data;
      unif = map_vs_part_uniform(intr,
                                 DIV_ROUND_UP(BITSET_LAST_BIT(
                                                 layout->component_mask),
                                              4),
                                 layout->uniform_base);
   } else {
      unif = map_fs_part_uniform(intr, data);
   }

   if (unif >= 0) {
      b->cursor = nir_instr_remove(&intr->instr);
      nir_def *load = nir_load_preamble(b, 1, intr->def.bit_size, .base = unif);
      nir_def_rewrite_uses(&intr->def, load);
      return true;
   } else if (intr->intrinsic == nir_intrinsic_load_texture_handle_agx) {
      b->cursor = nir_instr_remove(&intr->instr);
      nir_def *offs =
         nir_imul_imm(b, nir_u2u32(b, intr->src[0].ssa), AGX_TEXTURE_LENGTH);
      nir_def_rewrite_uses(&intr->def,
                           nir_bindless_image_agx(b, offs, .desc_set = 0));
      return true;
   } else {
      return false;
   }
}

bool
agx_nir_lower_non_monolithic_uniforms(nir_shader *nir, unsigned nr)
{
   struct agx_vs_prolog_key layout = {0};
   if (nr)
      BITSET_SET_COUNT(layout.component_mask, 0, nr * 4);

   return nir_shader_intrinsics_pass(nir, lower_non_monolithic_uniforms,
                                     nir_metadata_control_flow, &layout);
}

static bool
lower_adjacency(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const struct agx_vs_prolog_key *key = data;
   b->cursor = nir_before_instr(&intr->instr);

   if (intr->intrinsic != nir_intrinsic_load_vertex_id)
      return false;

   nir_def *id = nir_load_vertex_id(b);

   if (key->adjacency == MESA_PRIM_LINES_ADJACENCY) {
      id = poly_map_to_line_adj(b, id);
   } else if (key->adjacency == MESA_PRIM_TRIANGLE_STRIP_ADJACENCY) {
      id = poly_map_to_tri_strip_adj(b, id);
   } else if (key->adjacency == MESA_PRIM_LINE_STRIP_ADJACENCY) {
      id = poly_map_to_line_strip_adj(b, id);
   } else if (key->adjacency == MESA_PRIM_TRIANGLES_ADJACENCY) {
      /* Sequence (0, 2, 4), (6, 8, 10), ... */
      id = nir_imul_imm(b, id, 2);
   } else {
      UNREACHABLE("unknown");
   }

   id = poly_nir_load_vertex_id(b, id);

   nir_def_replace(&intr->def, id);
   return true;
}

static void
agx_nir_vs_prolog_impl(nir_builder *b, const void *key_, bool native_raw)
{
   const struct agx_vs_prolog_key *key = key_;
   b->shader->info.stage = MESA_SHADER_VERTEX;
   b->shader->info.name = "VS prolog";

   /* First, construct a passthrough shader reading each attribute and exporting
    * the value. We also need to export vertex/instance ID in their usual regs.
    */
   if (!key->static_vi) {
      unsigned i = 0;
      nir_def *vec = NULL;
      unsigned vec_idx = ~0;
      BITSET_FOREACH_SET(i, key->component_mask, AGX_MAX_ATTRIBS * 4) {
         unsigned a = i / 4;
         unsigned c = i % 4;

         if (vec_idx != a) {
            vec = nir_load_input(b, 4, 32, nir_imm_int(b, 0), .base = a);
            vec_idx = a;
         }

         nir_export_agx(b, nir_channel(b, vec, c),
                        .base = AGX_ABI_VIN_ATTRIB(i));
      }
   }

   if (!key->hw) {
      nir_export_agx(b, nir_channel(b, nir_load_global_invocation_id(b, 32), 0),
                     .base = AGX_ABI_VIN_VERTEX_ID_ZERO_BASE);
   }

   nir_export_agx(b, nir_load_vertex_id(b), .base = AGX_ABI_VIN_VERTEX_ID);
   nir_export_agx(b, nir_load_instance_id(b), .base = AGX_ABI_VIN_INSTANCE_ID);

   /* Now lower the resulting program using the key */
   if (!key->static_vi) {
      if (native_raw)
         agx_nir_lower_vbo_raw(b->shader, key->attribs, key->robustness, false);
      else
         agx_nir_lower_vbo(b->shader, key->attribs, key->robustness, false);

      /* Clean up redundant vertex ID loads */
      if (!key->hw || key->adjacency) {
         NIR_PASS(_, b->shader, nir_opt_cse);
         NIR_PASS(_, b->shader, nir_opt_dce);
      }
   }

   if (!key->hw) {
      b->cursor = nir_before_impl(nir_shader_get_entrypoint(b->shader));
      poly_nir_lower_sw_vs(b->shader);
   } else if (key->adjacency) {
      nir_shader_intrinsics_pass(b->shader, lower_adjacency,
                                 nir_metadata_control_flow, (void *)key);
   }
   nir_inline_sysval(b->shader, nir_intrinsic_load_index_size_poly,
                     key->sw_index_size_B);

   /* Finally, lower uniforms according to our ABI */
   nir_shader_intrinsics_pass(b->shader, lower_non_monolithic_uniforms,
                              nir_metadata_control_flow, (void *)key);
   b->shader->info.io_lowered = true;
}

void
agx_nir_vs_prolog(nir_builder *b, const void *key)
{
   agx_nir_vs_prolog_impl(b, key, false);
}

void
agx_nir_vs_prolog_raw(nir_builder *b, const void *key)
{
   agx_nir_vs_prolog_impl(b, key, true);
}

/*
 * Map a lowered load_input to its slot in the vertex input ABI.
 *
 * The prolog publishes every attribute component as a full 32-bit value in
 * its own pair of half-registers: agx_nir_vs_prolog_impl() fetches with
 * nir_load_input(b, 4, 32, ...) and exports each channel at
 * AGX_ABI_VIN_ATTRIB(4 * attrib + comp) = 2 * (8 + i), two halves apart.
 * struct agx_vs_prolog_key carries no bit sizes at all -- only a component
 * mask in these same 32-bit slot units -- so the prolog cannot publish
 * anything narrower even in principle.
 *
 * A main shader may still declare a narrower input, since we advertise
 * VK_KHR_16bit_storage's storageInputOutput16. That does not change the slot
 * arithmetic: lowered NIR IO keeps nir_intrinsic_component() in 32-bit slot
 * components even for 16-bit loads (nir_validate asserts component +
 * num_components <= 4), and the only way two 16-bit values share one
 * component is nir_io_semantics::high_16bits, which nir_lower_io never sets
 * and which is produced for varyings alone (nir_lower_mediump's 16-bit
 * varying slots and nir_opt_vectorize_io, neither of which we run). So one
 * declared component is still exactly one 32-bit ABI slot, and all the main
 * shader has to do is narrow the value the prolog handed it.
 */
static unsigned
vs_input_abi_slot(nir_intrinsic_instr *intr)
{
   ASSERTED nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   assert(!sem.high_16bits && "VS inputs never share a 32-bit slot");
   assert(intr->def.bit_size <= 32 && "64-bit attributes need two ABI slots");

   unsigned idx = nir_src_as_uint(intr->src[0]) + nir_intrinsic_base(intr);
   unsigned comp = nir_intrinsic_component(intr);
   assert(comp + intr->def.num_components <= 4);

   return 4 * idx + comp;
}

static bool
gather_inputs(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_input)
      return false;

   unsigned base = vs_input_abi_slot(intr);

   b->cursor = nir_before_instr(&intr->instr);
   BITSET_WORD *comps_read = data;
   nir_component_mask_t mask = nir_def_components_read(&intr->def);

   u_foreach_bit(c, mask) {
      BITSET_SET(comps_read, base + c);
   }

   return false;
}

bool
agx_nir_gather_vs_inputs(nir_shader *s, BITSET_WORD *attrib_components_read)
{
   return nir_shader_intrinsics_pass(
      s, gather_inputs, nir_metadata_control_flow, attrib_components_read);
}

static bool
lower_input_to_prolog(nir_builder *b, nir_intrinsic_instr *intr, void *_data)
{
   if (intr->intrinsic != nir_intrinsic_load_input)
      return false;

   unsigned base = vs_input_abi_slot(intr);

   b->cursor = nir_before_instr(&intr->instr);

   /* Always read the ABI slot at its published width. Loading at the narrow
    * width would be silently wrong rather than merely slow: agx_load_exported
    * strides the base by the DESTINATION size, so a 16-bit load would take
    * component 1 out of component 0's high half.
    */
   nir_def *val = nir_load_exported_agx(b, intr->def.num_components, 32,
                                        .base = AGX_ABI_VIN_ATTRIB(base));

   if (intr->def.bit_size != 32) {
      /* The prolog's vertex fetch already converted the attribute into the
       * 32-bit slot with the format's own numeric type (agx_nir_lower_vbo
       * emits f2f32/i2i32/u2u32), so narrowing by the declared base type
       * recovers the value exactly.
       */
      nir_alu_type base_type =
         nir_alu_type_get_base_type(nir_intrinsic_dest_type(intr));

      assert((base_type == nir_type_float || base_type == nir_type_int ||
              base_type == nir_type_uint) &&
             "narrow vertex input without a numeric type");

      val = nir_convert_to_bit_size(b, val, base_type, intr->def.bit_size);
   }

   nir_def_replace(&intr->def, val);
   return true;
}

bool
agx_nir_lower_vs_input_to_prolog(nir_shader *s)
{
   return nir_shader_intrinsics_pass(s, lower_input_to_prolog,
                                     nir_metadata_control_flow, NULL);
}

static bool
lower_active_samples_to_register(nir_builder *b, nir_intrinsic_instr *intr,
                                 void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_active_samples_agx)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *id =
      nir_load_exported_agx(b, 1, 16, .base = AGX_ABI_FIN_SAMPLE_MASK);

   nir_def_replace(&intr->def, id);
   return true;
}

static bool
lower_tests_zs_intr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   bool *value = data;
   if (intr->intrinsic != nir_intrinsic_load_shader_part_tests_zs_agx)
      return false;

   b->cursor = nir_instr_remove(&intr->instr);
   nir_def_rewrite_uses(&intr->def, nir_imm_intN_t(b, *value ? 0xFF : 0, 16));
   return true;
}

static bool
lower_tests_zs(nir_shader *s, bool value)
{
   if (!s->info.fs.uses_discard)
      return false;

   return nir_shader_intrinsics_pass(s, lower_tests_zs_intr,
                                     nir_metadata_control_flow, &value);
}

static inline bool
blend_uses_2src(struct agx_blend_rt_key rt)
{
   assert(rt.advanced_blend == false);
   const struct agx_blend_standard blend = agx_unpack_blend_standard(rt.mode);

   enum pipe_blendfactor factors[] = {
      blend.rgb_src_factor,
      blend.rgb_dst_factor,
      blend.alpha_src_factor,
      blend.alpha_dst_factor,
   };

   for (unsigned i = 0; i < ARRAY_SIZE(factors); ++i) {
      switch (factors[i]) {
      case PIPE_BLENDFACTOR_SRC1_COLOR:
      case PIPE_BLENDFACTOR_SRC1_ALPHA:
      case PIPE_BLENDFACTOR_INV_SRC1_COLOR:
      case PIPE_BLENDFACTOR_INV_SRC1_ALPHA:
         return true;
      default:
         break;
      }
   }

   return false;
}

static void
copy_colour(nir_builder *b, const struct agx_fs_epilog_key *key,
            unsigned out_rt, unsigned in_loc, bool dual_src)
{
   unsigned size = (key->link.size_32 & BITFIELD_BIT(in_loc)) ? 32 : 16;

   nir_def *value =
      nir_load_exported_agx(b, 4, size, .base = AGX_ABI_FOUT_COLOUR(in_loc));

   if (key->link.loc0_w_1 && in_loc == 0) {
      value =
         nir_vector_insert_imm(b, value, nir_imm_floatN_t(b, 1.0, size), 3);
   }

   if (key->smooth_line_alpha && in_loc == 0 && !dual_src) {
      nir_def *alpha = nir_fmul_imm(b, nir_channel(b, value, 3), 0.5);
      value = nir_vector_insert_imm(b, value, alpha, 3);
   }

   nir_store_output(b, value, nir_imm_int(b, 0),
                    .io_semantics.location = FRAG_RESULT_DATA0 + out_rt,
                    .io_semantics.dual_source_blend_index = dual_src,
                    .src_type = nir_type_float | size);
}

void
agx_nir_fs_epilog(nir_builder *b, const void *key_)
{
   const struct agx_fs_epilog_key *key = key_;
   b->shader->info.stage = MESA_SHADER_FRAGMENT;
   b->shader->info.name = "FS epilog";

   /* First, construct a passthrough shader reading each colour and outputting
    * the value.
    */
   for (unsigned rt = 0; rt < ARRAY_SIZE(key->remap); ++rt) {
      int location = key->remap[rt];

      /* Negative remaps indicate the attachment isn't written. */
      if (location >= 0 && key->link.loc_written & BITFIELD_BIT(location)) {
         copy_colour(b, key, rt, location, false);

         /* If this render target uses dual source blending, also copy the dual
          * source colour. While the copy_colour above is needed even for
          * missing attachments to handle alpha-to-coverage, this copy is only
          * for blending so should be suppressed for missing attachments to keep
          * the assert from blowing up on OpenGL.
          */
         if (!key->blend.rt[rt].advanced_blend &&
             blend_uses_2src(key->blend.rt[rt]) &&
             key->rt_formats[rt] != PIPE_FORMAT_NONE) {

            assert(location == 0);
            copy_colour(b, key, rt, 1, true);
         }
      }
   }

   /* Grab registers early, this has to happen in the first block. */
   nir_def *sample_id = NULL, *write_samples = NULL;
   if (key->link.sample_shading) {
      sample_id =
         nir_load_exported_agx(b, 1, 16, .base = AGX_ABI_FOUT_SAMPLE_MASK);
   }

   if (key->link.sample_mask_after_force_early) {
      write_samples =
         nir_load_exported_agx(b, 1, 16, .base = AGX_ABI_FOUT_WRITE_SAMPLES);
   }

   /* Now lower the resulting program using the key */
   struct agx_tilebuffer_layout tib = agx_build_tilebuffer_layout_bounded(
      key->rt_formats, ARRAY_SIZE(key->rt_formats), key->nr_samples, true,
      key->max_sample_size_B);

   if (key->force_small_tile)
      tib.tile_size = 16 * 16;

   bool force_translucent = false;
   nir_lower_blend_options opts = {
      .scalar_blend_const = true,
      .logicop_enable = key->blend.logicop_enable,
      .logicop_func = key->blend.logicop_func,
   };

   static_assert(ARRAY_SIZE(opts.rt) == 8, "max RTs out of sync");

   for (unsigned i = 0; i < 8; ++i) {
      if (key->blend.rt[i].advanced_blend) {
         const struct agx_blend_advanced blend =
            agx_unpack_blend_advanced(key->blend.rt[i].mode);

         opts.rt[i] = (nir_lower_blend_rt){
            .format = key->rt_formats[i],
            .advanced_blend = true,
            .colormask = key->blend.rt[i].colormask,
            .blend_mode = blend.op,
            .src_premultiplied = blend.src_premultiplied,
            .dst_premultiplied = blend.dst_premultiplied,
            .overlap = blend.overlap,
         };
      } else {
         const struct agx_blend_standard blend =
            agx_unpack_blend_standard(key->blend.rt[i].mode);

         opts.rt[i] = (nir_lower_blend_rt){
            .format = key->rt_formats[i],

            .rgb.src_factor = blend.rgb_src_factor,
            .rgb.dst_factor = blend.rgb_dst_factor,
            .rgb.func = blend.rgb_func,

            .alpha.src_factor = blend.alpha_src_factor,
            .alpha.dst_factor = blend.alpha_dst_factor,
            .alpha.func = blend.alpha_func,

            .colormask = key->blend.rt[i].colormask,
         };
      }
   }

   /* It's more efficient to use masked stores (with
    * agx_nir_lower_tilebuffer) than to emulate colour masking with
    * nir_lower_blend.
    */
   uint8_t colormasks[8] = {0};

   for (unsigned i = 0; i < 8; ++i) {
      if (key->rt_formats[i] == PIPE_FORMAT_NONE)
         continue;

      if (agx_tilebuffer_supports_mask(&tib, i)) {
         colormasks[i] = key->blend.rt[i].colormask;
         opts.rt[i].colormask = (uint8_t)BITFIELD_MASK(4);
      } else {
         colormasks[i] = (uint8_t)BITFIELD_MASK(4);
      }

      /* If not all bound RTs are fully written to, we need to force
       * translucent pass type. agx_nir_lower_tilebuffer will take
       * care of this for its own colormasks input.
       */
      unsigned comps = util_format_get_nr_components(key->rt_formats[i]);
      if ((opts.rt[i].colormask & BITFIELD_MASK(comps)) !=
          BITFIELD_MASK(comps)) {
         force_translucent = true;
      }
   }

   /* Alpha-to-coverage must be lowered before alpha-to-one */
   if (key->blend.alpha_to_coverage)
      NIR_PASS(_, b->shader, nir_lower_alpha_to_coverage, false,
               NULL);

   /* Depth/stencil writes must be deferred until after all discards,
    * particularly alpha-to-coverage.
    */
   if (key->link.write_z || key->link.write_s) {
      nir_store_zs_agx(
         b, nir_imm_intN_t(b, 0xFF, 16),
         nir_load_exported_agx(b, 1, 32, .base = AGX_ABI_FOUT_Z),
         nir_load_exported_agx(b, 1, 16, .base = AGX_ABI_FOUT_S),
         .base = (key->link.write_z ? 1 : 0) | (key->link.write_s ? 2 : 0));

      if (key->link.write_z)
         b->shader->info.outputs_written |= BITFIELD64_BIT(FRAG_RESULT_DEPTH);

      if (key->link.write_s)
         b->shader->info.outputs_written |= BITFIELD64_BIT(FRAG_RESULT_STENCIL);
   }

   /* Alpha-to-one must be lowered before blending */
   if (key->blend.alpha_to_one)
      NIR_PASS(_, b->shader, nir_lower_alpha_to_one);

   NIR_PASS(_, b->shader, nir_lower_blend, &opts);

   unsigned rt_spill = key->link.rt_spill_base;
   NIR_PASS(_, b->shader, agx_nir_lower_tilebuffer, &tib, colormasks, &rt_spill,
            write_samples, &force_translucent, key->preserve_zero_mask,
            key->lower_srgb);
   NIR_PASS(_, b->shader, agx_nir_lower_texture);
   NIR_PASS(_, b->shader, agx_nir_lower_multisampled_image_store);

   /* If the API shader runs once per sample, then the epilog runs once per
    * sample as well, so we need to lower our code to run for a single sample.
    *
    * If the API shader runs once per pixel, then the epilog runs once per
    * pixel. So we run through the monolithic MSAA lowering, which wraps the
    * epilog in the sample loop if needed. This localizes sample shading
    * to the epilog, when sample shading is not used but blending is.
    */
   if (key->link.sample_shading) {
      /* Lower the resulting discards. Done in agx_nir_lower_monolithic_msaa for
       * the pixel shaded path. Must be done before agx_nir_lower_to_per_sample
       * to avoid duplicating tests.
       */
      if (key->blend.alpha_to_coverage) {
         NIR_PASS(_, b->shader, agx_nir_lower_sample_mask);
      }

      NIR_PASS(_, b->shader, agx_nir_lower_to_per_sample);
      NIR_PASS(_, b->shader, agx_nir_lower_fs_active_samples_to_register);

      /* Ensure the sample ID is preserved in register. We do this late since it
       * has to go in the last block, and the above passes might add control
       * flow when lowering.
       */
      b->cursor = nir_after_impl(b->impl);
      nir_export_agx(b, sample_id, .base = AGX_ABI_FIN_SAMPLE_MASK);
   } else if (!key->defer_monolithic_msaa) {
      NIR_PASS(_, b->shader, agx_nir_lower_monolithic_msaa, key->nr_samples);
   }

   /* Finally, lower uniforms according to our ABI */
   nir_shader_intrinsics_pass(b->shader, lower_non_monolithic_uniforms,
                              nir_metadata_control_flow, (void *)key);

   /* There is no shader part after the epilog, so we're always responsible for
    * running our own tests, unless the fragment shader forced early tests.
    */
   NIR_PASS(_, b->shader, lower_tests_zs, !key->link.already_ran_zs);

   b->shader->info.io_lowered = true;
   b->shader->info.fs.uses_fbfetch_output |= force_translucent;
   b->shader->info.fs.uses_sample_shading = key->link.sample_shading;
}

struct lower_epilog_ctx {
   struct agx_fs_epilog_link_info *info;
   nir_variable *masked_samples;
   bool preserve_early_demote;
};

static bool
lower_output_to_epilog(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct lower_epilog_ctx *ctx = data;
   struct agx_fs_epilog_link_info *info = ctx->info;

   if (intr->intrinsic == nir_intrinsic_store_zs_agx) {
      assert(nir_src_as_uint(intr->src[0]) == 0xff && "msaa not yet lowered");
      b->cursor = nir_instr_remove(&intr->instr);

      unsigned base = nir_intrinsic_base(intr);
      info->write_z = !!(base & 1);
      info->write_s = !!(base & 2);

      if (info->write_z)
         nir_export_agx(b, intr->src[1].ssa, .base = AGX_ABI_FOUT_Z);

      if (info->write_s)
         nir_export_agx(b, intr->src[2].ssa, .base = AGX_ABI_FOUT_S);

      return true;
   }

   if (intr->intrinsic == nir_intrinsic_demote_samples &&
       b->shader->info.fs.early_fragment_tests) {

      /* G17P has no sample operand on LSTCOL.  Keep the demote in the main
       * shader so its grounded MOVMSK lowering updates coverage before the
       * linked epilog emits the ordinary whole-pixel store. */
      if (ctx->preserve_early_demote)
         return false;

      if (!ctx->masked_samples) {
         b->cursor = nir_before_impl(nir_shader_get_entrypoint(b->shader));

         ctx->masked_samples =
            nir_local_variable_create(b->impl, glsl_uint16_t_type(), NULL);

         nir_store_var(b, ctx->masked_samples, nir_imm_intN_t(b, 0xFF, 16),
                       nir_component_mask(1));
      }

      b->cursor = nir_before_instr(&intr->instr);

      nir_def *mask = nir_load_var(b, ctx->masked_samples);
      nir_def *mask_2 =
         nir_ixor(b, intr->src[0].ssa, nir_imm_intN_t(b, 0xff, 16));

      mask = nir_iand(b, mask, mask_2);
      nir_store_var(b, ctx->masked_samples, mask, nir_component_mask(1));

      nir_instr_remove(&intr->instr);
      return true;
   }

   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;

   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);

   /* Fix up gl_FragColor */
   if (sem.location == FRAG_RESULT_COLOR) {
      sem.location = FRAG_RESULT_DATA0;
      info->broadcast_rt0 = true;
   }

   /* We don't use the epilog for sample mask writes */
   if (sem.location < FRAG_RESULT_DATA0)
      return false;

   /* Determine the ABI location. Dual source blending aliases a second
    * render target, so get that out of the way now.
    */
   unsigned loc = sem.location - FRAG_RESULT_DATA0;
   loc += nir_src_as_uint(intr->src[1]);

   if (sem.dual_source_blend_index) {
      assert(loc == 0);
      loc = 1;
   }

   b->cursor = nir_instr_remove(&intr->instr);
   nir_def *vec = intr->src[0].ssa;

   info->loc_written |= BITFIELD_BIT(loc);

   if (vec->bit_size == 32)
      info->size_32 |= BITFIELD_BIT(loc);
   else
      assert(vec->bit_size == 16);

   uint32_t one_f = (vec->bit_size == 32 ? fui(1.0) : _mesa_float_to_half(1.0));
   unsigned comp = nir_intrinsic_component(intr);

   u_foreach_bit(c, nir_intrinsic_write_mask(intr)) {
      nir_scalar s = nir_scalar_resolved(vec, c);
      /* G17P's mixed-width MRT epilog cannot materialize the elided alpha
       * through AGX2's 16-bit immediate move. Keep it in the ordinary export
       * ABI, which is also what native multi-target shaders use. */
      if (!ctx->preserve_early_demote && loc == 0 && c == 3 &&
          nir_scalar_is_const(s) &&
          nir_scalar_as_uint(s) == one_f) {

         info->loc0_w_1 = true;
      } else {
         unsigned stride = vec->bit_size / 16;

         nir_export_agx(b, nir_channel(b, vec, c),
                        .base = AGX_ABI_FOUT_COLOUR(loc) + (comp + c) * stride);
      }
   }

   return true;
}

static bool
lower_fs_output_to_epilog(nir_shader *s, struct agx_fs_epilog_link_info *out,
                          bool preserve_early_demote)
{
   struct lower_epilog_ctx ctx = {
      .info = out,
      .preserve_early_demote = preserve_early_demote,
   };

   nir_shader_intrinsics_pass(s, lower_output_to_epilog,
                              nir_metadata_control_flow, &ctx);

   if (ctx.masked_samples) {
      nir_builder b =
         nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(s)));

      nir_export_agx(&b, nir_load_var(&b, ctx.masked_samples),
                     .base = AGX_ABI_FOUT_WRITE_SAMPLES);
      out->sample_mask_after_force_early = true;

      bool progress;
      do {
         progress = false;
         NIR_PASS(progress, s, nir_lower_vars_to_ssa);
         NIR_PASS(progress, s, nir_opt_dce);
      } while (progress);
   }

   out->sample_shading = s->info.fs.uses_sample_shading;
   return true;
}

bool
agx_nir_lower_fs_output_to_epilog(nir_shader *s,
                                  struct agx_fs_epilog_link_info *out)
{
   return lower_fs_output_to_epilog(s, out, false);
}

bool
agx_nir_lower_fs_output_to_epilog_g17p(
   nir_shader *s, struct agx_fs_epilog_link_info *out)
{
   return lower_fs_output_to_epilog(s, out, true);
}

bool
agx_nir_lower_fs_output_to_epilog_for_device(
   nir_shader *s, enum agx_chip chip, struct agx_fs_epilog_link_info *out)
{
   return lower_fs_output_to_epilog(s, out, chip == AGX_CHIP_G17P);
}

/*
 * Give a G17P tile access the sample-mask source its hardware form actually
 * takes.
 *
 * At one sample the operand is the whole-pixel immediate, which is what Mesa
 * already spells as ALL_SAMPLES, so nothing has to happen.
 *
 * At two or four samples there is no immediate form at all. Apple's compiler
 * reads the live coverage word out of a special register and passes that
 * register as the mask -- raw for a per-pixel shader ("every sample this
 * fragment covers"), ANDed with a one-hot bit for each iteration of an
 * unrolled per-sample shader. Reproduce exactly those two shapes.
 *
 * A non-constant mask is already an expression in a register, so it is left
 * alone and reaches the packer as the register form by itself. An epilog
 * unrolled per sample (agx_nir_unroll_g17p_samples) has folded each sample's
 * `mask & active_samples` to that sample's one-hot constant beforehand, so it
 * too becomes coverage ANDed with the sample's bit.
 */
static bool
lower_g17p_tile_sample(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   unsigned src;
   switch (intr->intrinsic) {
   case nir_intrinsic_load_local_pixel_agx:
      src = 0;
      break;
   case nir_intrinsic_store_local_pixel_agx:
      src = 1;
      break;
   default:
      return false;
   }

   const unsigned samples = *(const unsigned *)data;
   if (samples == 1) {
      /* Per-sample epilogs still carry a dynamic active-sample mask, but
       * there is only sample zero. The native 1x tile instruction takes the
       * whole-pixel immediate; preserve its form for these epilogs. */
      if (nir_src_is_const(intr->src[src]) &&
          nir_src_as_uint(intr->src[src]) == ALL_SAMPLES)
         return false;

      b->cursor = nir_before_instr(&intr->instr);
      nir_src_rewrite(&intr->src[src],
                      nir_imm_intN_t(b, ALL_SAMPLES, 16));
      return true;
   }

   /* Already a live expression: the per-sample lowerings have ANDed the mask
    * with an active-sample register, which is the register form. */
   if (!nir_src_is_const(intr->src[src]))
      return false;

   const unsigned mask = nir_src_as_uint(intr->src[src]);
   if (mask != ALL_SAMPLES &&
       (mask == 0 || mask >= (1u << samples)))
      return false;

   b->cursor = nir_before_instr(&intr->instr);

   nir_def *coverage = nir_load_active_samples_agx(b);
   nir_def *sel = (mask == ALL_SAMPLES)
                     ? coverage
                     : nir_iand(b, coverage, nir_imm_intN_t(b, mask, 16));

   nir_src_rewrite(&intr->src[src], sel);
   return true;
}

bool
agx_nir_lower_g17p_tile_samples(nir_shader *s, unsigned nr_samples)
{
   if (nr_samples != 1 && nr_samples != 2 && nr_samples != 4)
      return false;

   return nir_shader_intrinsics_pass(s, lower_g17p_tile_sample,
                                     nir_metadata_control_flow, &nr_samples);
}

bool
agx_nir_lower_g17p_1x_tile_samples(nir_shader *s)
{
   /* Honeykrisp advertises only one sample on G17P.  Per-sample epilog
    * lowering still materializes `1 << sample_id`, but sample_id can only be
    * zero.  Native 1x fragments encode the ordinary whole-pixel LSTCOL/LLDCOL
    * forms, which carry no sample operand. */
   return agx_nir_lower_g17p_tile_samples(s, 1);
}

bool
agx_nir_lower_g17p_1x_sample_shading(nir_shader *s)
{
   assert(s->info.stage == MESA_SHADER_FRAGMENT);

   const bool sample_shading = s->info.fs.uses_sample_shading;
   nir_lower_single_sampled_options options = {
      .lower_sample_mask_in = true,
   };
   const bool progress = nir_lower_single_sampled(s, &options);

   /* With one rasterization sample, per-sample and per-pixel invocation are
    * identical. Leaving this set makes agx_fast_link append its AGX2 sample
    * loop header/footer (including 0x0088 STOP) to AGX3 code, whereas G17P's
    * grounded terminator is 0x0000000e. */
   s->info.fs.uses_sample_shading = false;
   return progress || sample_shading;
}

bool
agx_nir_lower_fs_active_samples_to_register(nir_shader *s)
{
   return nir_shader_intrinsics_pass(s, lower_active_samples_to_register,
                                     nir_metadata_control_flow, NULL);
}

static bool
agx_nir_lower_stats_fs(nir_shader *s)
{
   assert(s->info.stage == MESA_SHADER_FRAGMENT);
   nir_builder b_ =
      nir_builder_at(nir_before_impl(nir_shader_get_entrypoint(s)));
   nir_builder *b = &b_;

   nir_push_if(b, nir_inot(b, nir_load_helper_invocation(b, 1)));
   nir_def *samples = nir_bit_count(b, nir_load_sample_mask_in(b));
   unsigned query = PIPE_STAT_QUERY_PS_INVOCATIONS;

   nir_def *addr = nir_load_stat_query_address_poly(b, .base = query);
   nir_global_atomic(b, 32, addr, samples, .atomic_op = nir_atomic_op_iadd);

   nir_pop_if(b, NULL);
   return nir_progress(true, b->impl, nir_metadata_control_flow);
}

void
agx_nir_fs_prolog(nir_builder *b, const void *key_)
{
   const struct agx_fs_prolog_key *key = key_;
   b->shader->info.stage = MESA_SHADER_FRAGMENT;
   b->shader->info.name = "FS prolog";

   /* First, insert code for any emulated features */
   if (key->api_sample_mask != 0xff) {
      /* Kill samples that are NOT covered by the mask */
      nir_demote_samples(b, nir_imm_intN_t(b, key->api_sample_mask ^ 0xff, 16));
      b->shader->info.fs.uses_discard = true;
   }

   if (key->statistics) {
      NIR_PASS(_, b->shader, agx_nir_lower_stats_fs);
   }

   if (key->cull_distance_size) {
      NIR_PASS(_, b->shader, agx_nir_lower_cull_distance_fs,
               key->cull_distance_size);
   }

   if (key->polygon_stipple) {
      NIR_PASS(_, b->shader, agx_nir_lower_poly_stipple);
   }

   /* Then, lower the prolog */
   NIR_PASS(_, b->shader, agx_nir_lower_discard_zs_emit);
   NIR_PASS(_, b->shader, agx_nir_lower_sample_mask);
   NIR_PASS(_, b->shader, nir_shader_intrinsics_pass,
            lower_non_monolithic_uniforms, nir_metadata_control_flow, NULL);
   NIR_PASS(_, b->shader, lower_tests_zs, key->run_zs_tests);

   b->shader->info.io_lowered = true;
}
