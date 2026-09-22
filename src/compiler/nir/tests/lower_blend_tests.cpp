/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <array>

#include "nir_test.h"
extern "C" {
#include "nir_lower_blend.h"
}
#include "util/format/u_formats.h"

class nir_lower_blend_test : public nir_test {
protected:
   nir_lower_blend_test() : nir_test("lower_blend", MESA_SHADER_FRAGMENT) {}

   void evaluate(unsigned bits, enum pipe_format format,
                 const std::array<float, 4> &src_color,
                 const std::array<float, 4> &dst_color,
                 const std::array<float, 4> &expected,
                 enum pipe_blend_overlap_mode overlap = PIPE_BLEND_OVERLAP_UNCORRELATED,
                 bool dst_premultiplied = true, float tolerance = 0.0f)
   {
      nir_def *src = nir_f2fN(b, nir_imm_vec4(b, src_color[0], src_color[1],
                                             src_color[2], src_color[3]), bits);
      nir_intrinsic_instr *store = nir_store_output(b, src, nir_imm_int(b, 0));
      nir_intrinsic_set_src_type(store, (nir_alu_type)(nir_type_float | bits));
      nir_io_semantics semantics = {};
      semantics.location = FRAG_RESULT_DATA0;
      semantics.num_slots = 1;
      nir_intrinsic_set_io_semantics(store, semantics);
      b->shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
      b->shader->info.io_lowered = true;

      nir_lower_blend_options options = {};
      options.rt[0].format = format;
      options.rt[0].advanced_blend = true;
      options.rt[0].blend_mode = PIPE_ADVANCED_BLEND_HARDMIX;
      options.rt[0].colormask = 0xf;
      options.rt[0].src_premultiplied = true;
      options.rt[0].dst_premultiplied = dst_premultiplied;
      options.rt[0].overlap = overlap;
      ASSERT_TRUE(nir_lower_blend(b->shader, &options));

      unsigned loads = 0;
      nir_foreach_block(block, b->impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_load_output)
               continue;

            b->cursor = nir_before_instr(instr);
            nir_def *dst = nir_f2fN(
               b, nir_imm_vec4(b, dst_color[0], dst_color[1],
                               dst_color[2], dst_color[3]), bits);
            nir_def_replace(&intr->def, dst);
            loads++;
         }
      }
      ASSERT_EQ(loads, 1u);

      while (nir_opt_constant_folding(b->shader)) {}
      nir_validate_shader(b->shader, "constant advanced blend");

      ASSERT_TRUE(nir_src_is_const(store->src[0]));
      const nir_const_value *result = nir_src_as_const_value(store->src[0]);
      for (unsigned c = 0; c < store->num_components; ++c) {
         EXPECT_NEAR(nir_const_value_as_float(result[c], bits), expected[c],
                     tolerance) << "component " << c;
      }
   }
};

/* Exact binary16/binary32 values on both sides of the discontinuity. */
static const std::array<float, 4> half_color = {0.5, 0.5, 0.5, 1.0};
static const std::array<float, 4> boundary_color = {
   0.5 - 1.0 / 1024, 0.5, 0.5 + 1.0 / 1024, 1.0};

TEST_F(nir_lower_blend_test, hardmix_threshold_f16)
{
   evaluate(16, PIPE_FORMAT_R16G16B16A16_FLOAT, half_color, boundary_color,
            {0, 1, 1, 1});
}

TEST_F(nir_lower_blend_test, hardmix_threshold_f32)
{
   evaluate(32, PIPE_FORMAT_R32G32B32A32_FLOAT, half_color, boundary_color,
            {0, 1, 1, 1});
}

TEST_F(nir_lower_blend_test, dual_source_mixed_precision)
{
   nir_def *src = nir_imm_vec4(b, 0.25, 0.5, 0.75, 1.0);
   nir_def *src1 = nir_f2f16(b, nir_imm_vec4(b, 0.0, 0.0, 0.0, 0.5));
   nir_def *dst = nir_imm_vec4(b, 0.75, 0.5, 0.25, 1.0);

   nir_lower_blend_rt rt = {};
   rt.format = PIPE_FORMAT_B8G8R8A8_UNORM;
   rt.rgb.func = PIPE_BLEND_ADD;
   rt.rgb.src_factor = PIPE_BLENDFACTOR_ONE;
   rt.rgb.dst_factor = PIPE_BLENDFACTOR_INV_SRC1_ALPHA;
   rt.alpha = rt.rgb;
   rt.colormask = 0xf;

   nir_def *result = nir_color_blend(b, src, src1, dst, &rt, false);
   ASSERT_EQ(result->bit_size, 32);
   nir_validate_shader(b->shader, "mixed-precision dual-source blend");
}

TEST_F(nir_lower_blend_test, hardmix_threshold_unorm8)
{
   evaluate(32, PIPE_FORMAT_R8G8B8A8_UNORM, half_color, boundary_color,
            {1, 1, 1, 1});
}

TEST_F(nir_lower_blend_test, hardmix_threshold_unorm16)
{
   evaluate(32, PIPE_FORMAT_R16G16B16A16_UNORM, half_color, boundary_color,
            {0, 1, 1, 1});
}

TEST_F(nir_lower_blend_test, hardmix_threshold_rgb10)
{
   evaluate(32, PIPE_FORMAT_R10G10B10A2_UNORM, half_color, boundary_color,
            {0, 1, 1, 1});
}

TEST_F(nir_lower_blend_test, hardmix_mixed_channel_precision)
{
   evaluate(32, PIPE_FORMAT_R5G6B5_UNORM, {0.99f, 0.99f, 0.99f, 1},
            {0, 0, 0, 1}, {1, 0, 1, 1});
}

TEST_F(nir_lower_blend_test, hardmix_snorm_precision)
{
   evaluate(32, PIPE_FORMAT_R8G8B8A8_SNORM, {0.997f, 0.997f, 0.997f, 1},
            {0, 0, 0, 1}, {1, 1, 1, 1});
}

TEST_F(nir_lower_blend_test, hardmix_unorm_tie_to_even)
{
   evaluate(32, PIPE_FORMAT_R8G8B8A8_UNORM,
            {254.5f / 255, 254.75f / 255, 254.25f / 255, 1},
            {0, 0, 0, 1}, {0, 1, 0, 1});
}

/* CTS pixel 17: a non-premultiplied destination of 0.75 is stored as
 * 191/255. With source R/A = 0.1875/0.75, the hard-mix sum is just below 1.
 * Check the final colour against the one-byte tolerance of UNORM storage.
 */
static const std::array<float, 4> quantized_destination = {
   191.0f / 255, 191.0f / 255, 191.0f / 255, 191.0f / 255};

TEST_F(nir_lower_blend_test, hardmix_nonpremultiplied_dst_uncorrelated)
{
   evaluate(32, PIPE_FORMAT_R8G8B8A8_UNORM, {0.1875, 0.09375, 0, 0.75},
            quantized_destination, {204.0f / 255, 45.0f / 255, 38.0f / 255,
                                     239.0f / 255},
            PIPE_BLEND_OVERLAP_UNCORRELATED, false, 0.5f / 255);
}

TEST_F(nir_lower_blend_test, hardmix_nonpremultiplied_dst_disjoint)
{
   evaluate(32, PIPE_FORMAT_R8G8B8A8_UNORM, {0.1875, 0.09375, 0, 0.75},
            quantized_destination, {191.0f / 255, 56.0f / 255, 48.0f / 255, 1},
            PIPE_BLEND_OVERLAP_DISJOINT, false, 0.5f / 255);
}
