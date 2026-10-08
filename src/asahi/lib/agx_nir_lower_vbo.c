/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "agx_nir_lower_vbo.h"

#include "asahi/compiler/agx_nir.h"
#include "asahi/layout/layout.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_format_convert.h"
#include "util/bitset.h"
#include "util/u_math.h"
#include "agx_linker.h"
#include "shader_enums.h"

struct ctx {
   const struct agx_velem_key *attribs;
   struct agx_robustness rs;
   bool dynamic_strides;
   bool native_raw;
};


/* G17P's general multiply-high encoding is not hardware-qualified yet, while
 * nir_opt_idiv_const lowers non-power-of-two udiv to multiply-high. Vertex
 * divisors are pipeline constants, so use exact binary long division composed
 * only of the already-grounded integer shift, compare/select and subtract
 * families. The remainder never underflows and every shifted divisor fits in
 * 32 bits by construction.
 */
static nir_def *
g17p_udiv_imm(nir_builder *b, nir_def *numerator, uint32_t divisor)
{
   assert(divisor != 0);

   if (util_is_power_of_two_or_zero(divisor))
      return nir_ushr_imm(b, numerator, util_logbase2(divisor));

   nir_def *remainder = numerator;
   nir_def *quotient = nir_imm_int(b, 0);

   for (int shift = __builtin_clz(divisor); shift >= 0; --shift) {
      const uint32_t shifted_divisor = divisor << shift;
      nir_def *ge = nir_uge(b, remainder, nir_imm_int(b, shifted_divisor));
      remainder = nir_bcsel(
         b, ge, nir_isub(b, remainder, nir_imm_int(b, shifted_divisor)),
         remainder);
      quotient = nir_ior(
         b, quotient,
         nir_bcsel(b, ge, nir_imm_int(b, UINT32_C(1) << shift),
                   nir_imm_int(b, 0)));
   }

   return quotient;
}

static bool
is_rgb10_a2(const struct util_format_description *desc)
{
   return desc->channel[0].shift == 0 && desc->channel[0].size == 10 &&
          desc->channel[1].shift == 10 && desc->channel[1].size == 10 &&
          desc->channel[2].shift == 20 && desc->channel[2].size == 10 &&
          desc->channel[3].shift == 30 && desc->channel[3].size == 2;
}

enum pipe_format
agx_vbo_internal_format(enum pipe_format format)
{
   const struct util_format_description *desc = util_format_description(format);

   /* RGB10A2 formats are native for UNORM and unpacked otherwise */
   if (is_rgb10_a2(desc)) {
      if (desc->is_unorm)
         return PIPE_FORMAT_R10G10B10A2_UNORM;
      else
         return PIPE_FORMAT_R32_UINT;
   }

   /* R11G11B10F is native and special */
   if (format == PIPE_FORMAT_R11G11B10_FLOAT)
      return format;

   /* No other non-array formats handled */
   if (!desc->is_array)
      return PIPE_FORMAT_NONE;

   /* Otherwise look at one (any) channel */
   int idx = util_format_get_first_non_void_channel(format);
   if (idx < 0)
      return PIPE_FORMAT_NONE;

   /* We only handle RGB formats (we could do SRGB if we wanted though?) */
   if ((desc->colorspace != UTIL_FORMAT_COLORSPACE_RGB) ||
       (desc->layout != UTIL_FORMAT_LAYOUT_PLAIN))
      return PIPE_FORMAT_NONE;

   /* We have native 8-bit and 16-bit normalized formats */
   struct util_format_channel_description chan = desc->channel[idx];

   if (chan.normalized) {
      if (chan.size == 8)
         return desc->is_unorm ? PIPE_FORMAT_R8_UNORM : PIPE_FORMAT_R8_SNORM;
      else if (chan.size == 16)
         return desc->is_unorm ? PIPE_FORMAT_R16_UNORM : PIPE_FORMAT_R16_SNORM;
   }

   /* Otherwise map to the corresponding integer format */
   switch (chan.size) {
   case 32:
      return PIPE_FORMAT_R32_UINT;
   case 16:
      return PIPE_FORMAT_R16_UINT;
   case 8:
      return PIPE_FORMAT_R8_UINT;
   default:
      return PIPE_FORMAT_NONE;
   }
}

bool
agx_vbo_supports_format(enum pipe_format format)
{
   return agx_vbo_internal_format(format) != PIPE_FORMAT_NONE;
}

/*
 * How G17P converts a vertex format. G17P has no fixed-function vertex fetch
 * and its LDDEV carries only an element width (8/16/32 bits), never a format.
 * Apple's G17P vertex programs (metal-nt -arch applegpu_g17p, one pipeline per
 * MTLVertexFormat) convert with a native ALU instruction straight after the
 * load, and only fall back to integer shifts and masks where no such
 * instruction exists:
 *
 *   8/16-bit UNORM/SNORM   LDDEV + `normflt` (two lanes per instruction)
 *   R10G10B10A2_UNORM,     LDDEV W32 + `unpack` (all lanes in one instruction)
 *   R11G11B10_FLOAT
 *   16-bit FLOAT/SINT/     LDDEV W16; the consumer reads the 16-bit half with
 *   UINT/[SU]SCALED        its own widening source (f16->f32, sext, zext,
 *                          int->float)
 *   8-bit integer/scaled,  shifts and masks (same as the software path)
 *   other 10:10:10:2
 *
 * On a G17P-class GPU (M5 Pro, applegpu_g17s), Apple's fetch of every code of
 * every lane of these formats is exact: correctly rounded c / (2^b - 1), SNORM
 * clamped to -1, exact half/11-bit float and integer values. The software
 * unpack computes u2f32(c) * fl(1 / (2^b - 1)) after lower_fdiv and is up to
 * 1 ULP off for some codes, so the native conversion is also the exact one.
 */
enum g17p_fetch {
   G17P_FETCH_SOFTWARE = 0,
   G17P_FETCH_NORMALIZED,
   G17P_FETCH_PACKED,
   G17P_FETCH_HALVES,
};

static enum g17p_fetch
g17p_fetch_class(const struct util_format_description *desc, int chan)
{
   if (is_rgb10_a2(desc))
      return desc->is_unorm ? G17P_FETCH_PACKED : G17P_FETCH_SOFTWARE;

   if (desc->format == PIPE_FORMAT_R11G11B10_FLOAT)
      return G17P_FETCH_PACKED;

   if (!desc->is_array || desc->layout != UTIL_FORMAT_LAYOUT_PLAIN ||
       desc->colorspace != UTIL_FORMAT_COLORSPACE_RGB)
      return G17P_FETCH_SOFTWARE;

   const struct util_format_channel_description *c = &desc->channel[chan];
   if (c->type != UTIL_FORMAT_TYPE_UNSIGNED &&
       c->type != UTIL_FORMAT_TYPE_SIGNED && c->type != UTIL_FORMAT_TYPE_FLOAT)
      return G17P_FETCH_SOFTWARE;

   if (c->normalized && (c->size == 8 || c->size == 16))
      return G17P_FETCH_NORMALIZED;

   if (c->size == 16)
      return G17P_FETCH_HALVES;

   return G17P_FETCH_SOFTWARE;
}

/* Convert one 16-bit lane with the widening conversion its format needs. */
static nir_def *
g17p_convert_half(nir_builder *b, nir_def *half,
                  const struct util_format_channel_description *c,
                  unsigned bit_size)
{
   if (c->type == UTIL_FORMAT_TYPE_FLOAT)
      return nir_f2fN(b, half, bit_size);

   const bool is_signed = c->type == UTIL_FORMAT_TYPE_SIGNED;
   if (c->pure_integer)
      return is_signed ? nir_i2iN(b, half, bit_size)
                       : nir_u2uN(b, half, bit_size);

   return is_signed ? nir_i2fN(b, half, bit_size)
                    : nir_u2fN(b, half, bit_size);
}

static nir_def *
apply_swizzle_channel(nir_builder *b, nir_def *vec, unsigned swizzle,
                      bool is_int)
{
   switch (swizzle) {
   case PIPE_SWIZZLE_X:
      return nir_channel(b, vec, 0);
   case PIPE_SWIZZLE_Y:
      return nir_channel(b, vec, 1);
   case PIPE_SWIZZLE_Z:
      return nir_channel(b, vec, 2);
   case PIPE_SWIZZLE_W:
      return nir_channel(b, vec, 3);
   case PIPE_SWIZZLE_0:
      return nir_imm_intN_t(b, 0, vec->bit_size);
   case PIPE_SWIZZLE_1:
      return is_int ? nir_imm_intN_t(b, 1, vec->bit_size)
                    : nir_imm_floatN_t(b, 1.0, vec->bit_size);
   default:
      UNREACHABLE("Invalid swizzle channel");
   }
}

static bool
pass(struct nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_input)
      return false;

   struct ctx *ctx = data;
   const struct agx_velem_key *attribs = ctx->attribs;
   b->cursor = nir_instr_remove(&intr->instr);

   nir_src *offset_src = nir_get_io_offset_src(intr);
   assert(nir_src_is_const(*offset_src) && "no attribute indirects");
   unsigned index = nir_intrinsic_base(intr) + nir_src_as_uint(*offset_src);

   /* Slots are compacted to the attributes the shader reads, so the caller
    * must have refused any shader reading more than AGX_MAX_ATTRIBS of them
    * (hk_compile_shader does). Reading attribs[] past that was the Fallout:
    * New Vegas new-game crash: DXVK's 18-attribute fixed-function vertex
    * shader fetched a garbage key from the stack here.
    */
   assert(index < AGX_MAX_ATTRIBS && "attribute slot beyond key capacity");

   struct agx_velem_key attrib = attribs[index];
   uint32_t stride = attrib.stride;

   const struct util_format_description *desc =
      util_format_description(attrib.format);
   int chan = util_format_get_first_non_void_channel(attrib.format);
   assert(chan >= 0);

   bool is_float = desc->channel[chan].type == UTIL_FORMAT_TYPE_FLOAT;
   bool is_unsigned = desc->channel[chan].type == UTIL_FORMAT_TYPE_UNSIGNED;
   bool is_signed = desc->channel[chan].type == UTIL_FORMAT_TYPE_SIGNED;
   bool is_fixed = desc->channel[chan].type == UTIL_FORMAT_TYPE_FIXED;
   bool is_int = util_format_is_pure_integer(attrib.format);

   assert((is_float ^ is_unsigned ^ is_signed ^ is_fixed) && "Invalid format");
   (void)is_float;

   enum pipe_format interchange_format = agx_vbo_internal_format(attrib.format);
   assert(interchange_format != PIPE_FORMAT_NONE);

   unsigned interchange_align = util_format_get_blocksize(interchange_format);
   unsigned interchange_comps = util_format_get_nr_components(attrib.format);

   /* In the hardware, uint formats zero-extend and float formats convert.
    * However, non-uint formats using a uint interchange format shouldn't be
    * zero extended.
    */
   unsigned interchange_register_size =
      util_format_is_pure_uint(interchange_format) &&
            !util_format_is_pure_uint(attrib.format)
         ? (interchange_align * 8)
         : intr->def.bit_size;

   /* Non-UNORM R10G10B10A2 loaded as a scalar and unpacked */
   if (interchange_format == PIPE_FORMAT_R32_UINT && !desc->is_array)
      interchange_comps = 1;

   /* Calculate the element to fetch the vertex for. Divide the instance ID by
    * the divisor for per-instance data. Divisor=0 specifies per-vertex data.
    */
   nir_def *el;
   if (attrib.instanced) {
      if (attrib.divisor > 0) {
         nir_def *instance_id = nir_load_instance_id(b);
         el = ctx->native_raw
                 ? g17p_udiv_imm(b, instance_id, attrib.divisor)
                 : nir_udiv_imm(b, instance_id, attrib.divisor);
      } else {
         el = nir_imm_int(b, 0);
      }

      el = nir_iadd(b, el, nir_load_base_instance(b));

      BITSET_SET(b->shader->info.system_values_read,
                 SYSTEM_VALUE_BASE_INSTANCE);
   } else {
      el = nir_load_vertex_id(b);
   }

   /* VBO bases are per-attribute, otherwise they're per-buffer. This allows
    * memory sinks to work properly with robustness, allows folding
    * the src_offset into the VBO base to save an add in the shader, and reduces
    * the size of the vertex fetch key. That last piece allows reusing a linked
    * VS with both separate and interleaved attributes.
    */
   nir_def *buf_handle = nir_imm_int(b, index);

   /* Robustness is handled at the ID level */
   nir_def *bounds = nir_load_attrib_clamp_agx(b, buf_handle);
   nir_def *oob = nir_ult(b, bounds, el);

   /* We clamp to handle GL robustness. This should be optimized further.
    * However, with the fix up after the load for D3D robustness, we don't need
    * this clamp if we can ignore the fault.
    */
   if (ctx->rs.level >= AGX_ROBUSTNESS_GL &&
       !(ctx->rs.level >= AGX_ROBUSTNESS_D3D && ctx->rs.soft_fault)) {

      el = nir_bcsel(b, oob, nir_imm_int(b, 0), el);
   }

   nir_def *base = nir_load_vbo_base_agx(b, buf_handle);

   assert((stride % interchange_align) == 0 && "must be aligned");

   unsigned stride_el = stride / interchange_align;
   unsigned shift = 0;

   /* Try to use the small shift on the load itself when possible. This can save
    * an instruction. Shifts are only available for regular interchange formats,
    * i.e. the set of formats that support masking.
    */
   if ((stride_el == 2 || stride_el == 4) &&
       ail_isa_format_supports_mask((enum ail_isa_format)interchange_format)) {

      shift = util_logbase2(stride_el);
      stride_el = 1;
   }

   nir_def *stride_el_def = nir_imm_int(b, stride_el);
   if (ctx->dynamic_strides) {
      assert(stride_el == 0);
      stride_el_def = nir_load_vbo_stride_agx(b, buf_handle);
   }

   nir_def *stride_offset_el = nir_imul(b, el, stride_el_def);

   /* Fixing up the address is expected to be profitable for vec3 and above, as
    * it requires 2 instructions. It is implemented with a 64GiB carveout at the
    * bottom of memory, using soft fault to return zeroes.
    */
   bool rs_address_fixup = interchange_comps > 2 && ctx->rs.soft_fault;

   if (ctx->rs.level >= AGX_ROBUSTNESS_D3D && rs_address_fixup) {
      base = nir_bcsel(b, oob, nir_imm_int64(b, 0), base);
   }


   /* G17P formats with a native conversion (g17p_fetch_class above). The
    * load keeps the software path's element size and index, so the address
    * math is unchanged; only the conversion after it differs.
    */
   const enum g17p_fetch fetch =
      ctx->native_raw ? g17p_fetch_class(desc, chan) : G17P_FETCH_SOFTWARE;
   if (fetch != G17P_FETCH_SOFTWARE) {
      const unsigned count = util_format_get_nr_components(attrib.format);
      nir_def *value;

      if (fetch == G17P_FETCH_HALVES) {
         /* One W16 LDDEV of every lane into consecutive halves, exactly the
          * load the software path emits; each half is then read directly by
          * its widening conversion. */
         nir_def *raw = nir_load_constant_agx(
            b, DIV_ROUND_UP(count, 2), 32, base, stride_offset_el,
            .format = PIPE_FORMAT_R16_UINT,
            .base = shift | (count << AGX_NIR_MEMORY_PACKED_COUNT_SHIFT));
         if (ctx->rs.level >= AGX_ROBUSTNESS_D3D && !rs_address_fixup)
            raw = nir_bcsel(b, oob, nir_imm_zero(b, raw->num_components, 32),
                            raw);

         nir_def *lanes[4];
         for (unsigned c = 0; c < count; ++c) {
            nir_def *word = nir_channel(b, raw, c / 2);
            nir_def *half = (c & 1) ? nir_unpack_32_2x16_split_y(b, word)
                                    : nir_unpack_32_2x16_split_x(b, word);
            lanes[c] = g17p_convert_half(b, half, &desc->channel[c],
                                         intr->def.bit_size);
         }
         value = nir_vec(b, lanes, count);
      } else {
         /* A load in the normalized or packed format itself: the backend
          * emits the raw-width LDDEV followed by `normflt` or `unpack`
          * (agx_emit_load). Every such conversion maps 0 to 0.0, so the
          * robustness select may follow it. */
         const enum pipe_format load_format =
            agx_vbo_internal_format(attrib.format);
         const unsigned load_count =
            fetch == G17P_FETCH_PACKED
               ? (load_format == PIPE_FORMAT_R11G11B10_FLOAT ? 3 : 4)
               : count;
         value = nir_load_constant_agx(b, load_count, 32, base,
                                       stride_offset_el, .format = load_format,
                                       .base = shift);
         if (ctx->rs.level >= AGX_ROBUSTNESS_D3D && !rs_address_fixup)
            value = nir_bcsel(b, oob, nir_imm_zero(b, load_count, 32), value);
         value = nir_f2fN(b, value, intr->def.bit_size);
      }

      nir_def *channels[4] = {NULL};
      for (unsigned i = 0; i < intr->num_components; ++i) {
         unsigned c = nir_intrinsic_component(intr) + i;
         channels[i] =
            apply_swizzle_channel(b, value, desc->swizzle[c], is_int);
      }

      nir_def_rewrite_uses(&intr->def,
                           nir_vec(b, channels, intr->num_components));
      return true;
   }

   /* Native G17P LDDEV returns packed raw bytes/halves. Keep one word for
    * RGBA8 even at byte-aligned base/stride (Apple packed_uchar4 stride5,
    * offset1 witness), then perform Vulkan format conversion explicitly for
    * the formats without a native conversion (8-bit integer and scaled
    * channels, non-UNORM 10:10:10:2). */
   if (ctx->native_raw && (!desc->is_array || desc->channel[chan].size < 32)) {
      const unsigned bits = desc->is_array ? desc->channel[chan].size : 32;
      const unsigned count = desc->is_array ? desc->nr_channels : 1;
      assert((bits == 8 || bits == 16 || bits == 32) && count <= 4);
      enum pipe_format raw_format = bits == 8 ? PIPE_FORMAT_R8_UINT :
                                    bits == 16 ? PIPE_FORMAT_R16_UINT :
                                                 PIPE_FORMAT_R32_UINT;
      nir_def *raw = nir_load_constant_agx(
         b, DIV_ROUND_UP(bits * count, 32), 32, base, stride_offset_el,
         .format = raw_format,
         .base = shift | (count << AGX_NIR_MEMORY_PACKED_COUNT_SHIFT));
      if (ctx->rs.level >= AGX_ROBUSTNESS_D3D && !rs_address_fixup)
         raw = nir_bcsel(b, oob, nir_imm_zero(b, raw->num_components, 32), raw);

      nir_def *rgba = nir_format_unpack_rgba(b, raw, attrib.format);
      rgba = is_int ? (is_signed ? nir_i2iN(b, rgba, intr->def.bit_size)
                                 : nir_u2uN(b, rgba, intr->def.bit_size))
                    : nir_f2fN(b, rgba, intr->def.bit_size);
      unsigned components[4];
      for (unsigned i = 0; i < intr->def.num_components; ++i)
         components[i] = nir_intrinsic_component(intr) + i;
      nir_def_rewrite_uses(&intr->def,
                          nir_swizzle(b, rgba, components,
                                      intr->def.num_components));
      return true;
   }

   /* Load the raw vector */
   nir_def *memory = nir_load_constant_agx(
      b, interchange_comps, interchange_register_size, base, stride_offset_el,
      .format = interchange_format, .base = shift);

   /* For scalar loads, it's faster to fix up the output than the address. */
   if (ctx->rs.level >= AGX_ROBUSTNESS_D3D && !rs_address_fixup) {
      nir_def *zero = nir_imm_zero(b, memory->num_components, memory->bit_size);
      memory = nir_bcsel(b, oob, zero, memory);
   }

   unsigned dest_size = intr->def.bit_size;

   /* Unpack but do not convert non-native non-array formats */
   if (is_rgb10_a2(desc) && interchange_format == PIPE_FORMAT_R32_UINT) {
      unsigned bits[] = {10, 10, 10, 2};

      if (is_signed)
         memory = nir_format_unpack_sint(b, memory, bits, 4);
      else
         memory = nir_format_unpack_uint(b, memory, bits, 4);
   }

   if (desc->channel[chan].normalized) {
      /* 8/16-bit normalized formats are native, others converted here */
      if (is_rgb10_a2(desc) && is_signed) {
         unsigned bits[] = {10, 10, 10, 2};
         memory = nir_format_snorm_to_float(b, memory, bits);
      } else if (desc->channel[chan].size == 32) {
         assert(desc->is_array && "no non-array 32-bit norm formats");
         unsigned bits[] = {32, 32, 32, 32};

         if (is_signed)
            memory = nir_format_snorm_to_float(b, memory, bits);
         else
            memory = nir_format_unorm_to_float(b, memory, bits);
      }
   } else if (desc->channel[chan].pure_integer) {
      /* Zero-extension is native, may need to sign extend */
      if (is_signed)
         memory = nir_i2iN(b, memory, dest_size);
   } else {
      if (is_unsigned)
         memory = nir_u2fN(b, memory, dest_size);
      else if (is_signed || is_fixed)
         memory = nir_i2fN(b, memory, dest_size);
      else
         memory = nir_f2fN(b, memory, dest_size);

      /* 16.16 fixed-point weirdo GL formats need to be scaled */
      if (is_fixed) {
         assert(desc->is_array && desc->channel[chan].size == 32);
         assert(dest_size == 32 && "overflow if smaller");
         memory = nir_fmul_imm(b, memory, 1.0 / 65536.0);
      }
   }

   /* We now have a properly formatted vector of the components in memory. Apply
    * the format swizzle forwards to trim/pad/reorder as needed.
    */
   nir_def *channels[4] = {NULL};

   for (unsigned i = 0; i < intr->num_components; ++i) {
      unsigned c = nir_intrinsic_component(intr) + i;
      channels[i] = apply_swizzle_channel(b, memory, desc->swizzle[c], is_int);
   }

   nir_def *logical = nir_vec(b, channels, intr->num_components);
   nir_def_rewrite_uses(&intr->def, logical);
   return true;
}

static bool
lower_vbo(nir_shader *shader, const struct agx_velem_key *attribs,
          struct agx_robustness robustness, bool dynamic_strides, bool native_raw)
{
   assert(shader->info.stage == MESA_SHADER_VERTEX);

   /* To implement null vertex buffer descriptors, we need either soft fault or
    * GL robustness with a vertex buffer at 0x0.
    */
   if (!robustness.soft_fault) {
      robustness.level = MAX2(robustness.level, AGX_ROBUSTNESS_GL);
   }

   struct ctx ctx = {
      .attribs = attribs,
      .rs = robustness,
      .dynamic_strides = dynamic_strides,
      .native_raw = native_raw,
   };

   return nir_shader_intrinsics_pass(shader, pass, nir_metadata_control_flow,
                                     &ctx);
}

bool
agx_nir_lower_vbo(nir_shader *shader, const struct agx_velem_key *attribs,
                  struct agx_robustness robustness, bool dynamic_strides)
{
   return lower_vbo(shader, attribs, robustness, dynamic_strides, false);
}

bool
agx_nir_lower_vbo_raw(nir_shader *shader, const struct agx_velem_key *attribs,
                      struct agx_robustness robustness, bool dynamic_strides)
{
   return lower_vbo(shader, attribs, robustness, dynamic_strides, true);
}
